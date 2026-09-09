import math
import threading
from pathlib import Path

import numpy as np
import torch

from .neural_waveform import build_neural_conditioning, load_neural_waveform_decoder


SAMPLE_RATE = 48000
STRUCTURE_LOWPASS_HZ = 9000.0
STRUCTURE_TRANSITION_HZ = 1500.0


def smooth_lowpass_structure(x, cutoff_hz=STRUCTURE_LOWPASS_HZ, transition_hz=STRUCTURE_TRANSITION_HZ):
    if x.ndim == 2:
        x = x.unsqueeze(1)
    n = int(x.shape[-1])
    spec = torch.fft.rfft(x, dim=-1)
    freqs = torch.linspace(0.0, SAMPLE_RATE * 0.5, spec.shape[-1], device=x.device, dtype=x.dtype)
    cutoff = float(cutoff_hz)
    end = min(SAMPLE_RATE * 0.5, cutoff + float(transition_hz))
    mask = torch.ones_like(freqs)
    mask = torch.where(freqs >= end, torch.zeros_like(mask), mask)
    if end > cutoff:
        t = torch.clamp((freqs - cutoff) / (end - cutoff), 0.0, 1.0)
        taper = 0.5 * (1.0 + torch.cos(math.pi * t))
        mask = torch.where((freqs > cutoff) & (freqs < end), taper, mask)
    return torch.fft.irfft(spec * mask.view(1, 1, -1), n=n, dim=-1)


class NeuralWaveformRuntimeRoute:
    """Voicebank-scoped v0.3 neural waveform route with exact legacy fallback."""

    def __init__(self, runtime_root, config, device="cpu"):
        self.runtime_root = Path(runtime_root).expanduser().resolve()
        self.config = dict(config or {})
        self.device = torch.device(device)
        self.local = threading.local()
        self.original_decode = None
        self.checkpoint = self._resolve_checkpoint()
        self.model = None
        self.metadata = {}
        self.load_error = None
        if self.checkpoint is not None:
            try:
                model, metadata = load_neural_waveform_decoder(self.checkpoint, device=self.device)
                if str(metadata.get("trainer_generation") or "") != "conditioned-v3":
                    raise RuntimeError("runtime neural waveform checkpoint is not a conditioned-v3 model")
                self.model = model
                self.metadata = dict(metadata or {})
            except Exception as exc:
                self.load_error = str(exc)
                self.model = None
                self.metadata = {}
        self.select_record(None)

    def _resolve_checkpoint(self):
        if not bool(self.config.get("neural_waveform_enabled", True)):
            return None
        explicit = self.config.get("neural_waveform_checkpoint")
        candidates = []
        if explicit:
            p = Path(str(explicit)).expanduser()
            if not p.is_absolute():
                p = self.runtime_root / p
            candidates.append(p)
        candidates.extend([
            self.runtime_root / "control_models" / "neural-waveform-v0.3.0-conditioned-v3-pareto-best.pt",
            self.runtime_root / "control_models" / "neural-waveform-v0.3.0-conditioned-v3-multipitch-best.pt",
            self.runtime_root / "control_models" / "neural-waveform-v0.3.0-conditioned-v3.pt",
        ])
        seen = set()
        for candidate in candidates:
            try:
                candidate = candidate.resolve()
            except Exception:
                candidate = Path(candidate)
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _norm_path(value):
        if not value:
            return ""
        try:
            return str(Path(str(value)).expanduser().resolve())
        except Exception:
            return str(value)

    def select_record(self, record):
        trained_voicebank = self._norm_path(self.metadata.get("voicebank"))
        record_voicebank = self._norm_path((record or {}).get("voicebank_root"))
        active = bool(
            self.model is not None
            and trained_voicebank
            and record_voicebank
            and trained_voicebank == record_voicebank
        )
        self.local.active = active
        if self.model is None:
            reason = "checkpoint-unavailable" if self.checkpoint is None else "checkpoint-load-error"
        elif not record:
            reason = "no-ai14-record"
        elif not trained_voicebank:
            reason = "checkpoint-missing-voicebank-provenance"
        elif trained_voicebank != record_voicebank:
            reason = "voicebank-mismatch"
        else:
            reason = "selected"
        self.local.selection_reason = reason
        self.local.last_stats = {
            "neural_waveform_loaded": bool(self.model is not None),
            "neural_waveform_used": False,
            "neural_waveform_selection_reason": reason,
            "neural_waveform_checkpoint": str(self.checkpoint or ""),
            "neural_waveform_checkpoint_role": str(self.metadata.get("checkpoint_role") or ""),
        }
        return active

    def is_active(self):
        return bool(getattr(self.local, "active", False))

    def stats(self):
        return dict(getattr(self.local, "last_stats", {}) or {})

    def install_patch(self, core_module):
        if self.original_decode is not None:
            return
        self.original_decode = core_module.deterministic_decode_dualrate
        route = self
        original = self.original_decode

        def decode_with_neural(
            decoder, f0, z, seed, synthesis_sample_rate, adapter=None, detail=None,
            prototype_index=None, timbre_shift_semitones=0.0, detail_strength=1.0,
            frame_controls=None, ai_control_adapter=None,
        ):
            if not route.is_active():
                result = original(
                    decoder, f0, z, seed, synthesis_sample_rate,
                    adapter=adapter, detail=detail, prototype_index=prototype_index,
                    timbre_shift_semitones=timbre_shift_semitones,
                    detail_strength=detail_strength, frame_controls=frame_controls,
                    ai_control_adapter=ai_control_adapter,
                )
                stats = route.stats()
                stats["neural_waveform_used"] = False
                route.local.last_stats = stats
                return result

            if int(synthesis_sample_rate) != SAMPLE_RATE:
                raise RuntimeError(
                    f"conditioned-v3 neural runtime requires {SAMPLE_RATE} Hz DDSP synthesis, "
                    f"got {int(synthesis_sample_rate)}"
                )
            if detail is None:
                raise RuntimeError("conditioned-v3 neural runtime requires warped detail conditioning")

            torch.manual_seed(int(seed))
            with torch.inference_mode():
                raw_structure, aux = decoder(
                    f0, z, adapter=adapter, detail=detail,
                    prototype_index=prototype_index,
                    timbre_shift_semitones=timbre_shift_semitones,
                    detail_strength=detail_strength,
                    frame_controls=frame_controls,
                    ai_control_adapter=ai_control_adapter,
                    synthesis_sample_rate=int(synthesis_sample_rate),
                    return_aux=True,
                )
                legacy = aux.get("legacy_wav")
                if legacy is None:
                    raise RuntimeError("neural runtime DDSP conditioning did not return legacy_wav")
                conditioning = build_neural_conditioning(z, detail, f0, aux)
                if int(conditioning.shape[1]) != int(route.model.condition_channels):
                    raise RuntimeError(
                        "neural runtime conditioning width mismatch: "
                        f"model={route.model.condition_channels} runtime={conditioning.shape[1]}"
                    )
                structure = smooth_lowpass_structure(raw_structure.detach())
                neural = route.model(conditioning, structure)

            neural_np = neural[0, 0].detach().cpu().numpy().astype(np.float32)
            legacy_samples = int(legacy.shape[-1])
            legacy_np = core_module.resample_exact(
                neural_np,
                int(synthesis_sample_rate),
                int(decoder.sample_rate),
                legacy_samples,
            )
            full_stats = dict(aux.get("fullband_stats") or {})
            full_stats.update({
                "neural_waveform_used": True,
                "neural_waveform_backend": "conditioned-v3-direct-waveform",
                "neural_waveform_checkpoint": str(route.checkpoint),
                "neural_waveform_checkpoint_role": str(route.metadata.get("checkpoint_role") or ""),
                "neural_waveform_structure_lowpass_hz": STRUCTURE_LOWPASS_HZ,
                "neural_waveform_structure_transition_hz": STRUCTURE_TRANSITION_HZ,
            })
            route.local.last_stats = {
                "neural_waveform_loaded": True,
                "neural_waveform_used": True,
                "neural_waveform_selection_reason": "selected",
                "neural_waveform_checkpoint": str(route.checkpoint),
                "neural_waveform_checkpoint_role": str(route.metadata.get("checkpoint_role") or ""),
                "neural_waveform_backend": "conditioned-v3-direct-waveform",
                "neural_waveform_structure_lowpass_hz": STRUCTURE_LOWPASS_HZ,
                "neural_waveform_structure_transition_hz": STRUCTURE_TRANSITION_HZ,
            }
            return legacy_np, neural_np, full_stats

        core_module.deterministic_decode_dualrate = decode_with_neural

    def describe(self):
        return {
            "loaded": bool(self.model is not None),
            "checkpoint": str(self.checkpoint or ""),
            "checkpoint_role": str(self.metadata.get("checkpoint_role") or ""),
            "trained_voicebank": str(self.metadata.get("voicebank") or ""),
            "load_error": self.load_error,
        }
