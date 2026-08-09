#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .adapter import DEFAULT_DETAIL_DIM, VoicebankAdapter, load_adapter, save_adapter
from .articulation import (
    analyze_articulation_regions,
    combine_canonical_articulation_templates,
    extract_neutral_articulation_template,
    save_canonical_articulation,
)
from .fidelity import TinyFidelityRefiner, load_refiner, save_refiner
from .loudness import active_rms_dbfs, oto_loudness_signature, source_gain_to_target_db
from .core import (
    YuazDDSPResamplerEngine,
    crop_oto,
    derive_voiced_mask,
    extract_detail_features,
    extract_f0,
    read_audio,
    stable_seed,
)
from .voicebank import annotate_utau_subbanks, cache_key, entry_to_dict, file_sha256, pcm_fingerprint, scan_voicebank, voicebank_id


PROFILE_VERSION = 10
CLARITY_TRAINING_VERSION = 6
ANTI_LEAK_TRAINING_VERSION = 3
FIDELITY_TRAINING_VERSION = 4
ARTICULATION_TRAINING_VERSION = 3
CANONICAL_ARTICULATION_VERSION = 1
LOUDNESS_PROFILE_VERSION = 2
CACHE_FORMAT = 5


def exact_length(x, n):
    n = int(n)
    if x.shape[-1] == n:
        return x
    if x.shape[-1] > n:
        return x[..., :n]
    return F.pad(x, (0, n - x.shape[-1]))


def stft_logmag(x, n_fft):
    hop = n_fft // 4
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
    return torch.log1p(spec.abs())


def _frequency_mask(sr, bins, low, high, device):
    freqs = torch.linspace(0.0, sr * 0.5, bins, device=device)
    return (freqs >= float(low)) & (freqs <= float(high))


def _local_contrast(x, kernel=7):
    b, f, t = x.shape
    y = x.transpose(1, 2).reshape(b * t, 1, f)
    k = min(int(kernel), f if f % 2 == 1 else max(1, f - 1))
    k = max(1, k)
    if k % 2 == 0:
        k -= 1
    if k <= 1:
        return torch.zeros_like(x)
    pad = k // 2
    smooth = F.avg_pool1d(F.pad(y, (pad, pad), mode="replicate"), kernel_size=k, stride=1)
    return (y - smooth).reshape(b, t, f).transpose(1, 2)


def spectral_shape_loss(target, pred, sr, n_fft=1024):
    if target.shape[-1] < n_fft:
        return target.new_tensor(0.0), target.new_tensor(0.0), target.new_tensor(0.0)
    t = stft_logmag(target, n_fft)
    p = stft_logmag(pred, n_fft)
    mask = _frequency_mask(sr, t.shape[-2], 250.0, 6500.0, t.device)
    if not bool(mask.any()):
        return target.new_tensor(0.0), target.new_tensor(0.0), target.new_tensor(0.0)
    tb = t[:, mask, :]
    pb = p[:, mask, :]
    if tb.shape[1] < 3:
        return target.new_tensor(0.0), target.new_tensor(0.0), target.new_tensor(0.0)
    grad_t = torch.diff(tb, dim=1)
    grad_p = torch.diff(pb, dim=1)
    curvature_t = torch.diff(grad_t, dim=1)
    curvature_p = torch.diff(grad_p, dim=1)
    contrast_t = _local_contrast(tb)
    contrast_p = _local_contrast(pb)
    return (
        F.l1_loss(grad_p, grad_t),
        F.l1_loss(curvature_p, curvature_t),
        F.l1_loss(contrast_p, contrast_t),
    )


def articulation_trajectory_loss(target, pred, sr, start_sample, end_sample):
    n = min(target.shape[-1], pred.shape[-1])
    a = int(np.clip(start_sample, 0, max(0, n - 1)))
    b = int(np.clip(end_sample, a + 1, n))
    if b - a < 384:
        zero = target.new_tensor(0.0)
        return zero, {"articulation_temporal": 0.0, "articulation_centroid": 0.0, "articulation_bands": 0.0, "articulation_energy": 0.0}
    t = target[..., a:b]
    p = pred[..., a:b]
    n_fft = 256 if b - a < 900 else 512
    hop = n_fft // 8
    window = torch.hann_window(n_fft, device=t.device, dtype=t.dtype)
    ts = torch.stft(t, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True).abs()
    ps = torch.stft(p, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True).abs()
    tl = torch.log1p(ts)
    pl = torch.log1p(ps)
    if tl.shape[-1] > 1:
        temporal = F.l1_loss(torch.diff(pl, dim=-1), torch.diff(tl, dim=-1))
    else:
        temporal = target.new_tensor(0.0)
    freqs = torch.linspace(0.0, sr * 0.5, ts.shape[-2], device=t.device, dtype=t.dtype).view(1, -1, 1)
    tc = (ts * freqs).sum(dim=-2) / (ts.sum(dim=-2) + 1e-6) / (sr * 0.5)
    pc = (ps * freqs).sum(dim=-2) / (ps.sum(dim=-2) + 1e-6) / (sr * 0.5)
    centroid = F.l1_loss(pc, tc)
    bands = []
    for lo, hi in ((180.0, 900.0), (900.0, 1800.0), (1800.0, 3200.0), (3200.0, min(7600.0, sr * 0.5))):
        mask = (freqs[0, :, 0] >= lo) & (freqs[0, :, 0] < hi)
        if bool(mask.any()):
            tb = torch.log1p(ts[:, mask, :].mean(dim=-2))
            pb = torch.log1p(ps[:, mask, :].mean(dim=-2))
            bands.append(F.l1_loss(pb, tb))
    band_loss = torch.stack(bands).mean() if bands else target.new_tensor(0.0)
    kernel = max(3, int(round(sr * 0.008)))
    if kernel % 2 == 0:
        kernel += 1
    te = F.avg_pool1d(t.abs().unsqueeze(1) if t.ndim == 2 else t.abs(), kernel_size=kernel, stride=max(1, kernel // 4), padding=kernel // 2)
    pe = F.avg_pool1d(p.abs().unsqueeze(1) if p.ndim == 2 else p.abs(), kernel_size=kernel, stride=max(1, kernel // 4), padding=kernel // 2)
    frames = min(te.shape[-1], pe.shape[-1])
    energy = F.l1_loss(pe[..., :frames], te[..., :frames]) if frames else target.new_tensor(0.0)
    loss = 0.44 * temporal + 0.15 * centroid + 0.29 * band_loss + 0.12 * energy
    return loss, {
        "articulation_temporal": float(temporal.detach()),
        "articulation_centroid": float(centroid.detach()),
        "articulation_bands": float(band_loss.detach()),
        "articulation_energy": float(energy.detach()),
    }


def high_frequency_loss(target, pred, sr, low_hz=2500.0, n_fft=512):
    if target.shape[-1] < n_fft:
        return F.l1_loss(pred, target)
    t = stft_logmag(target, n_fft)
    p = stft_logmag(pred, n_fft)
    mask = _frequency_mask(sr, t.shape[-2], low_hz, sr * 0.5, t.device)
    if not bool(mask.any()):
        return F.l1_loss(p, t)
    return F.l1_loss(p[:, mask, :], t[:, mask, :])


def clarity_reconstruction_loss(target, pred, sr, first_voiced_sample, regularization=None, pair_mode=False, articulation_end_sample=None):
    n = min(target.shape[-1], pred.shape[-1])
    target = target[..., :n]
    pred = pred[..., :n]
    fv = int(np.clip(first_voiced_sample, 0, max(0, n - 1)))
    voiced_start = max(0, fv - int(0.015 * sr))
    voiced_target = target[..., voiced_start:]
    voiced_pred = pred[..., voiced_start:]

    spectral = []
    for n_fft in (512, 1024, 2048):
        if voiced_target.shape[-1] >= n_fft:
            spectral.append(F.l1_loss(stft_logmag(voiced_pred, n_fft), stft_logmag(voiced_target, n_fft)))
    base = torch.stack(spectral).mean() if spectral else F.l1_loss(voiced_pred, voiced_target)
    hf = high_frequency_loss(voiced_target, voiced_pred, sr, low_hz=2600.0, n_fft=512)
    grad, curvature, contrast = spectral_shape_loss(voiced_target, voiced_pred, sr, n_fft=1024)

    onset_a = max(0, fv - int(0.045 * sr))
    onset_b = min(n, fv + int(0.110 * sr))
    if onset_b - onset_a >= 512:
        onset_hf = high_frequency_loss(target[..., onset_a:onset_b], pred[..., onset_a:onset_b], sr, low_hz=3200.0, n_fft=256)
    else:
        onset_hf = target.new_tensor(0.0)

    wave = F.l1_loss(voiced_pred, voiced_target)
    articulation_end = int(articulation_end_sample if articulation_end_sample is not None else min(n, fv + int(0.18 * sr)))
    articulation_start = max(0, fv - int(0.020 * sr))
    articulation_loss, articulation_parts = articulation_trajectory_loss(target, pred, sr, articulation_start, articulation_end)
    if pair_mode:
        loss = base + 0.22 * hf + 0.16 * grad + 0.06 * curvature + 0.10 * contrast + 0.035 * wave + 0.12 * articulation_loss
    else:
        loss = base + 0.28 * hf + 0.18 * grad + 0.07 * curvature + 0.12 * contrast + 0.16 * onset_hf + 0.045 * wave + 0.22 * articulation_loss
    if regularization is not None:
        loss = loss + 0.002 * regularization
    parts = {
        "spectral": float(base.detach()),
        "high_frequency": float(hf.detach()),
        "formant_gradient": float(grad.detach()),
        "formant_curvature": float(curvature.detach()),
        "spectral_contrast": float(contrast.detach()),
        "onset_high_frequency": float(onset_hf.detach()),
        "wave": float(wave.detach()),
        "articulation_trajectory": float(articulation_loss.detach()),
        **articulation_parts,
    }
    return loss, parts


def load_project_config(path):
    path = Path(path)
    if not path.exists():
        raise RuntimeError("Run scripts/configure-macos.command first.")
    return json.loads(path.read_text(encoding="utf-8"))


def semitone_distance(a, b):
    a = float(a)
    b = float(b)
    if a <= 0 or b <= 0:
        return 0.0
    return abs(12.0 * math.log2(b / a))


def timbre_perturb_audio(audio, sr, seed):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size < 32:
        return audio.copy()
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    n = len(audio)
    spec = np.fft.rfft(audio.astype(np.float64))
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sr))
    nyq = max(1.0, sr * 0.5)
    x = np.clip(freqs / nyq, 0.0, 1.0)
    tilt_db = rng.uniform(-4.0, 4.0) * (x - 0.35)
    center = rng.uniform(700.0, min(4200.0, nyq * 0.78))
    width = rng.uniform(500.0, 1500.0)
    bump_db = rng.uniform(-2.5, 2.5) * np.exp(-0.5 * ((freqs - center) / max(120.0, width)) ** 2)
    air_db = rng.uniform(-2.0, 2.0) * np.clip((freqs - 3500.0) / max(1.0, nyq - 3500.0), 0.0, 1.0)
    gain = np.power(10.0, (tilt_db + bump_db + air_db) / 20.0)
    out = np.fft.irfft(spec * gain, n=n).real.astype(np.float32)
    in_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-10))
    out_rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2) + 1e-10))
    if out_rms > 1e-8:
        out *= np.float32(np.clip(in_rms / out_rms, 0.75, 1.33))
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def content_consistency_loss(adapter, latent, perturbed_latent, perturbed_latent_b=None, first_voiced=0, last_voiced=None):
    variants = [latent, perturbed_latent]
    if perturbed_latent_b is not None:
        variants.append(perturbed_latent_b)
    contents = [adapter.content_representation(x) for x in variants]
    frames = min(x.shape[-1] for x in contents)
    a = int(np.clip(first_voiced, 0, max(0, frames - 1)))
    b = int(np.clip(last_voiced if last_voiced is not None else frames, a + 1, frames))
    trimmed = []
    bases = []
    for content, base in zip(contents, variants):
        content = content[..., :frames]
        base = base[..., :frames]
        if b - a >= 4:
            content = content[..., a:b]
            base = base[..., a:b]
        trimmed.append(content)
        bases.append(base)
    center = torch.stack(trimmed, dim=0).mean(dim=0)
    invariance = torch.stack([F.smooth_l1_loss(x, center.detach()) for x in trimmed]).mean()
    anchor = torch.stack([F.smooth_l1_loss(x, base.detach()) for x, base in zip(trimmed, bases)]).mean()
    spread = torch.stack([(x - center).pow(2).mean() for x in trimmed]).mean()
    return invariance + 0.5 * spread, anchor


def pair_content_consistency_loss(adapter, source_latent, target_latent):
    target_frames = target_latent.shape[-1]
    src = F.interpolate(source_latent, size=target_frames, mode="linear", align_corners=False)
    c_src = adapter.content_representation(src)
    c_tgt = adapter.content_representation(target_latent)
    c_src = F.layer_norm(c_src.transpose(1, 2), (c_src.shape[1],)).transpose(1, 2)
    c_tgt = F.layer_norm(c_tgt.transpose(1, 2), (c_tgt.shape[1],)).transpose(1, 2)
    return 0.5 * (F.smooth_l1_loss(c_src, c_tgt.detach()) + F.smooth_l1_loss(c_tgt, c_src.detach()))


class VoicebankPreparer:
    def __init__(self, project_root, voicebank_root, mode="quick"):
        self.project_root = Path(project_root).resolve()
        self.voicebank_root = Path(voicebank_root).expanduser().resolve()
        self.mode = mode
        self.config = load_project_config(self.project_root / "config.json")
        self.engine = YuazDDSPResamplerEngine(
            self.config["yuaz_repo"], self.config["checkpoint"],
            transition_ms=self.config.get("transition_ms", 70),
            use_rvq=self.config.get("use_rvq", False),
            output_sr=self.config.get("output_sr", 44100),
            registry_path=self.config.get("registry_path"),
        )
        self.scan = scan_voicebank(self.voicebank_root)
        self.yuaz_dir = self.voicebank_root / ".yuaz"
        self.cache_dir = self.yuaz_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.articulation_dir = self.yuaz_dir / "articulation"
        self.canonical_articulation_dir = self.articulation_dir / "canonical"
        self.canonical_articulation_dir.mkdir(parents=True, exist_ok=True)
        self.bank_id = voicebank_id(self.voicebank_root)
        self.subbank_info = None
        self.cache_subbank_index = {}
        self.loudness_enabled = bool(self.config.get("normalize_voicebank_loudness", True))
        self.loudness_target_dbfs = float(self.config.get("normalization_target_dbfs", -18.0))
        self.loudness_peak_ceiling_dbfs = float(self.config.get("normalization_peak_ceiling_dbfs", -1.0))
        self.loudness_peak_guard_knee_db = float(self.config.get("normalization_peak_guard_knee_db", 3.0))
        self.loudness_emergency_max_abs_gain_db = float(self.config.get("normalization_emergency_max_abs_gain_db", 30.0))
        self.loudness_tolerance_db = float(self.config.get("normalization_tolerance_db", 0.05))

    def profile(self):
        entries = self.scan["entries"]
        print(f"Found {len(self.scan['oto_files'])} oto.ini files.")
        print(f"Found {len(entries)} valid oto entries.")
        if self.scan["malformed"]:
            print(f"Skipped {len(self.scan['malformed'])} missing WAV references.")
        manifest_entries = []
        valid_caches = []
        f0_values = []
        voiced_ratios = []
        durations = []
        articulation_durations = []
        articulation_confidences = []
        loudness_values = []
        for index, entry in enumerate(entries, 1):
            print(f"[{index}/{len(entries)}] {entry.relative_wav} :: {entry.alias}", flush=True)
            cache_path = self.cache_dir / f"{cache_key(entry)}.npz"
            wav_path = Path(entry.wav_path)
            sha = file_sha256(wav_path)
            pcm = pcm_fingerprint(wav_path)
            reused = False
            upgraded = False
            if cache_path.exists():
                try:
                    upgrade_payload = None
                    with np.load(cache_path, allow_pickle=False) as old:
                        base_valid = str(old["sha256"].item()) == sha and "detail" in old.files and "timbre_aug_latent" in old.files and "timbre_aug_latent_b" in old.files and "audio" in old.files and "f0" in old.files
                        current = base_valid and "articulation_end" in old.files and "cache_format" in old.files and int(old["cache_format"].item()) >= CACHE_FORMAT
                        if current:
                            reused = True
                        elif base_valid:
                            payload = {key: np.asarray(old[key]) for key in old.files}
                            audio_old = np.asarray(payload["audio"], dtype=np.float32)
                            f0_old = np.asarray(payload["f0"], dtype=np.float32)
                            detail_old = np.asarray(payload["detail"], dtype=np.float32)
                            articulation = analyze_articulation_regions(audio_old, self.engine.sr, self.engine.hop, f0_old, detail_old, entry.consonant)
                            payload.update({
                                "cache_format": np.asarray(CACHE_FORMAT),
                                "articulation_raw_end": np.asarray(articulation["raw_end_frame"]),
                                "articulation_transition_end": np.asarray(articulation["transition_end_frame"]),
                                "articulation_end": np.asarray(articulation["articulation_end_frame"]),
                                "articulation_confidence": np.asarray(articulation["confidence"], dtype=np.float32),
                                "consonant": np.asarray(entry.consonant, dtype=np.float32),
                            })
                            upgrade_payload = payload
                    if upgrade_payload is not None:
                        np.savez_compressed(cache_path, **upgrade_payload)
                        reused = True
                        upgraded = True
                except Exception:
                    reused = False
                    upgraded = False
            if not reused:
                try:
                    audio = read_audio(wav_path, self.engine.sr)
                    audio = crop_oto(audio, self.engine.sr, entry.offset, entry.cutoff)
                    if len(audio) < int(0.08 * self.engine.sr):
                        raise RuntimeError("sample too short after oto crop")
                    f0 = extract_f0(audio, self.engine.sr, self.engine.hop)
                    f0_t = torch.from_numpy(f0).float().view(1, 1, -1)
                    audio_t = torch.from_numpy(audio).float().view(1, 1, -1)
                    with torch.inference_mode():
                        z, f0_aligned = self.engine.encoder(audio_t, f0_override=f0_t)
                    f0_np = f0_aligned[0, 0].cpu().numpy().astype(np.float32)
                    z_np = z[0].cpu().numpy().astype(np.float16)
                    detail_np = extract_detail_features(audio, self.engine.sr, self.engine.hop).astype(np.float16)
                    perturbed = timbre_perturb_audio(audio, self.engine.sr, stable_seed(entry.relative_wav, sha, "timbre-perturb-a"))
                    perturbed_b = timbre_perturb_audio(audio, self.engine.sr, stable_seed(entry.relative_wav, sha, "timbre-perturb-b"))
                    perturbed_t = torch.from_numpy(perturbed).float().view(1, 1, -1)
                    perturbed_b_t = torch.from_numpy(perturbed_b).float().view(1, 1, -1)
                    with torch.inference_mode():
                        z_aug, _ = self.engine.encoder(perturbed_t, f0_override=f0_aligned)
                        z_aug_b, _ = self.engine.encoder(perturbed_b_t, f0_override=f0_aligned)
                    z_aug_np = z_aug[0].cpu().numpy().astype(np.float16)
                    z_aug_b_np = z_aug_b[0].cpu().numpy().astype(np.float16)
                    voiced_mask, first_voiced, last_voiced = derive_voiced_mask(f0_np)
                    articulation = analyze_articulation_regions(audio, self.engine.sr, self.engine.hop, f0_np, detail_np.astype(np.float32), entry.consonant)
                    np.savez_compressed(
                        cache_path,
                        cache_format=np.asarray(CACHE_FORMAT), sha256=np.asarray(sha), pcm=np.asarray(pcm),
                        latent=z_np, timbre_aug_latent=z_aug_np, timbre_aug_latent_b=z_aug_b_np, f0=f0_np.astype(np.float16), detail=detail_np,
                        audio=audio.astype(np.float32), sr=np.asarray(self.engine.sr),
                        first_voiced=np.asarray(first_voiced), last_voiced=np.asarray(last_voiced),
                        articulation_raw_end=np.asarray(articulation["raw_end_frame"]),
                        articulation_transition_end=np.asarray(articulation["transition_end_frame"]),
                        articulation_end=np.asarray(articulation["articulation_end_frame"]),
                        articulation_confidence=np.asarray(articulation["confidence"], dtype=np.float32),
                        consonant=np.asarray(entry.consonant, dtype=np.float32),
                    )
                except Exception as exc:
                    manifest_entries.append({**entry_to_dict(entry), "sha256": sha, "pcm": pcm, "status": "error", "error": str(exc)})
                    continue
            try:
                data = np.load(cache_path, allow_pickle=False)
                f0_np = data["f0"].astype(np.float32)
                audio_np = data["audio"].astype(np.float32)
                voiced = f0_np > 1.0
                if np.any(voiced):
                    f0_values.extend(f0_np[voiced].tolist())
                voiced_ratios.append(float(np.mean(voiced)))
                durations.append(float(len(audio_np) / self.engine.sr))
                articulation_end_frame = int(data["articulation_end"].item())
                first_voiced_frame = int(data["first_voiced"].item())
                articulation_ms = max(0.0, (articulation_end_frame - first_voiced_frame) * self.engine.hop * 1000.0 / self.engine.sr)
                articulation_confidence = float(data["articulation_confidence"].item()) if "articulation_confidence" in data.files else 0.0
                articulation_durations.append(articulation_ms)
                articulation_confidences.append(articulation_confidence)
                measured_dbfs = active_rms_dbfs(audio_np, self.engine.sr)
                diagnostic_gain_db = source_gain_to_target_db(measured_dbfs, self.loudness_target_dbfs)
                loudness_values.append(measured_dbfs)
                valid_caches.append(str(cache_path))
                status = "upgraded" if upgraded else ("cached" if reused else "analyzed")
                median_f0 = float(np.median(f0_np[voiced])) if np.any(voiced) else 0.0
                manifest_entries.append({
                    **entry_to_dict(entry), "sha256": sha, "pcm": pcm, "status": status,
                    "cache": str(cache_path.relative_to(self.voicebank_root)), "median_f0_hz": median_f0,
                    "first_voiced_ms": float(first_voiced_frame * self.engine.hop * 1000.0 / self.engine.sr),
                    "articulation_end_ms": float(articulation_end_frame * self.engine.hop * 1000.0 / self.engine.sr),
                    "articulation_voiced_span_ms": float(articulation_ms),
                    "articulation_confidence": articulation_confidence,
                    "active_rms_dbfs": float(measured_dbfs),
                    "diagnostic_gain_to_target_db": float(diagnostic_gain_db),
                    "loudness_signature": oto_loudness_signature(entry.offset, entry.consonant, entry.cutoff),
                })
            except Exception as exc:
                manifest_entries.append({**entry_to_dict(entry), "sha256": sha, "pcm": pcm, "status": "error", "error": str(exc)})

        self.subbank_info = annotate_utau_subbanks(self.voicebank_root, manifest_entries, self.scan["prefix_map"])
        self.cache_subbank_index = {}
        for item in manifest_entries:
            if item.get("status") == "error" or not item.get("cache"):
                continue
            self.cache_subbank_index[str((self.voicebank_root / item["cache"]).resolve())] = int(item.get("subbank_index", 0))

        groups = {}
        for item in manifest_entries:
            if item.get("status") == "error":
                continue
            groups.setdefault(item.get("base_alias") or item.get("alias", ""), []).append(item)
        multipitch_groups = []
        pair_candidates = 0
        for base_alias, items in groups.items():
            pitched = [x for x in items if float(x.get("median_f0_hz", 0.0)) > 0]
            subbanks = sorted({str(x.get("subbank_label", "")) for x in pitched})
            pitches = sorted({round(float(x.get("median_f0_hz", 0.0)), 1) for x in pitched})
            if len(pitched) >= 2 and len(pitches) >= 2 and len(subbanks) >= 2:
                pair_candidates += len(pitched) * (len(pitched) - 1)
                multipitch_groups.append({"base_alias": base_alias, "samples": len(pitched), "subbanks": subbanks, "median_f0_hz": pitches})
        multipitch_groups.sort(key=lambda x: (-x["samples"], x["base_alias"]))

        canonical_articulation = self._build_canonical_articulations(manifest_entries)
        print(
            f"Canonical articulation: {canonical_articulation['alias_count']} aliases "
            f"({canonical_articulation['multipitch_canonical_count']} multipitch, "
            f"{canonical_articulation['single_neutral_fallback_count']} neutral fallback)."
        )

        subbank_loudness = {}
        loudness_entries = []
        for item in manifest_entries:
            if item.get("status") == "error" or "active_rms_dbfs" not in item:
                continue
            label = item.get("subbank_label") or "unassigned"
            subbank_loudness.setdefault(label, []).append(float(item["active_rms_dbfs"]))
            loudness_entries.append({
                "relative_wav": item.get("relative_wav"),
                "alias": item.get("alias"),
                "offset": float(item.get("offset", 0.0)),
                "consonant": float(item.get("consonant", 0.0)),
                "cutoff": float(item.get("cutoff", 0.0)),
                "signature": item.get("loudness_signature"),
                "active_rms_dbfs": float(item.get("active_rms_dbfs", -120.0)),
                "diagnostic_gain_to_target_db": float(item.get("diagnostic_gain_to_target_db", 0.0)),
                "subbank_label": item.get("subbank_label"),
            })
        loudness_report = {
            "format": LOUDNESS_PROFILE_VERSION,
            "mode": "strict_final_render",
            "enabled": bool(self.loudness_enabled),
            "target_active_rms_dbfs": float(self.loudness_target_dbfs),
            "peak_ceiling_dbfs": float(self.loudness_peak_ceiling_dbfs),
            "peak_guard_knee_db": float(self.loudness_peak_guard_knee_db),
            "emergency_max_abs_gain_db": float(self.loudness_emergency_max_abs_gain_db),
            "tolerance_db": float(self.loudness_tolerance_db),
            "source_measured_median_dbfs": float(np.median(loudness_values)) if loudness_values else 0.0,
            "source_quietest_dbfs": float(np.min(loudness_values)) if loudness_values else 0.0,
            "source_loudest_dbfs": float(np.max(loudness_values)) if loudness_values else 0.0,
            "entry_count": len(loudness_entries),
            "subbanks": {
                label: {
                    "count": len(values),
                    "median_active_rms_dbfs": float(np.median(values)),
                }
                for label, values in sorted(subbank_loudness.items())
            },
            "entries": loudness_entries,
        }
        (self.yuaz_dir / "loudness.json").write_text(json.dumps(loudness_report, indent=2, ensure_ascii=False), encoding="utf-8")

        profile = {
            "format": PROFILE_VERSION,
            "clarity_training_version": CLARITY_TRAINING_VERSION,
            "anti_leak_training_version": ANTI_LEAK_TRAINING_VERSION,
            "voicebank_id": self.bank_id,
            "voicebank_root": str(self.voicebank_root),
            "created_at": time.time(),
            "oto_count": len(self.scan["oto_files"]),
            "entry_count": len(entries),
            "valid_cache_count": len(valid_caches),
            "missing_reference_count": len(self.scan["malformed"]),
            "median_f0_hz": float(np.median(f0_values)) if f0_values else 0.0,
            "median_voiced_ratio": float(np.median(voiced_ratios)) if voiced_ratios else 0.0,
            "median_duration_sec": float(np.median(durations)) if durations else 0.0,
            "median_articulation_voiced_span_ms": float(np.median(articulation_durations)) if articulation_durations else 0.0,
            "median_articulation_confidence": float(np.median(articulation_confidences)) if articulation_confidences else 0.0,
            "articulation_training_version": ARTICULATION_TRAINING_VERSION,
            "canonical_articulation_version": CANONICAL_ARTICULATION_VERSION,
            "canonical_articulation_alias_count": int(canonical_articulation.get("alias_count", 0)),
            "canonical_articulation_multipitch_count": int(canonical_articulation.get("multipitch_canonical_count", 0)),
            "canonical_articulation_single_fallback_count": int(canonical_articulation.get("single_neutral_fallback_count", 0)),
            "canonical_articulation_mean_coherence": float(canonical_articulation.get("mean_coherence", 0.0)),
            "loudness_profile_version": LOUDNESS_PROFILE_VERSION,
            "loudness_normalization_enabled": bool(self.loudness_enabled),
            "loudness_target_active_rms_dbfs": float(self.loudness_target_dbfs),
            "loudness_source_measured_median_dbfs": loudness_report["source_measured_median_dbfs"],
            "loudness_normalization_mode": "strict_final_render",
            "detail_feature_dim": DEFAULT_DETAIL_DIM,
            "cache_format": CACHE_FORMAT,
            "timbre_perturbation_latent": True,
            "dual_timbre_perturbation": True,
            "fidelity_training_version": FIDELITY_TRAINING_VERSION,
            "prefix_map": self.scan["prefix_map"],
            "utau_subbank_strategy": self.subbank_info.get("strategy", "utau_native"),
            "prefix_map_authoritative": bool(self.subbank_info.get("prefix_map_authoritative", False)),
            "fallback_created_prototypes": int(self.subbank_info.get("fallback_created_prototypes", 0)),
            "fallback_assignment_count": int(self.subbank_info.get("fallback_assignment_count", 0)),
            "utau_subbank_count": int(self.subbank_info.get("prototype_count", 0)),
            "utau_subbanks": self.subbank_info.get("subbanks", []),
            "multipitch_group_count": len(multipitch_groups),
            "multipitch_pair_candidates": int(pair_candidates),
            "multipitch_groups": multipitch_groups[:200],
            "oto_encodings": self.scan["encodings"],
        }
        (self.yuaz_dir / "profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.yuaz_dir / "subbanks.json").write_text(json.dumps(self.subbank_info, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.yuaz_dir / "manifest.json").write_text(json.dumps({"profile": profile, "entries": manifest_entries}, indent=2, ensure_ascii=False), encoding="utf-8")
        return profile, manifest_entries, valid_caches

    def _build_canonical_articulations(self, manifest_entries):
        groups = {}
        for item in manifest_entries:
            if item.get("status") == "error" or not item.get("cache"):
                continue
            base_alias = str(item.get("base_alias") or item.get("alias") or "").strip()
            if not base_alias:
                continue
            groups.setdefault(base_alias, []).append(item)

        aliases = {}
        used_files = set()
        multipitch_count = 0
        single_count = 0
        coherence_values = []
        for base_alias, items in groups.items():
            by_subbank = {}
            for item in items:
                label = str(item.get("subbank_label") or "unassigned")
                score = (
                    float(item.get("articulation_confidence", 0.0)),
                    float(item.get("articulation_voiced_span_ms", 0.0)),
                )
                current = by_subbank.get(label)
                if current is None or score > current[0]:
                    by_subbank[label] = (score, item)
            representatives = [value[1] for value in by_subbank.values()]
            templates = []
            source_labels = []
            for item in representatives:
                try:
                    cache_path = (self.voicebank_root / item["cache"]).resolve()
                    with np.load(cache_path, allow_pickle=False) as data:
                        audio = data["audio"].astype(np.float32)
                        first = int(data["first_voiced"].item())
                        end = int(data["articulation_end"].item())
                    a = int(np.clip(first * self.engine.hop, 0, len(audio)))
                    b = int(np.clip(end * self.engine.hop, a, len(audio)))
                    if b - a < 256:
                        continue
                    template = extract_neutral_articulation_template(audio[a:b], self.engine.sr, frames=32, n_fft=256)
                    if template is None:
                        continue
                    templates.append(template)
                    source_labels.append(str(item.get("subbank_label") or "unassigned"))
                except Exception:
                    continue
            canonical = combine_canonical_articulation_templates(templates)
            if canonical is None:
                continue
            distinct_subbanks = sorted(set(source_labels))
            multipitch = len(distinct_subbanks) >= 2
            if multipitch:
                multipitch_count += 1
            else:
                single_count += 1
                canonical["coherence"] = min(float(canonical.get("coherence", 0.5)), 0.58)
            digest = hashlib.sha1(base_alias.encode("utf-8", "replace")).hexdigest()[:20]
            filename = f"{digest}.npz"
            path = self.canonical_articulation_dir / filename
            source_kind = "multipitch_canonical" if multipitch else "single_neutral_fallback"
            save_canonical_articulation(path, canonical, {
                "base_alias": base_alias,
                "source": source_kind,
                "subbanks": distinct_subbanks,
                "version": CANONICAL_ARTICULATION_VERSION,
            })
            used_files.add(filename)
            coherence = float(canonical.get("coherence", 0.5))
            coherence_values.append(coherence)
            rel = str(path.relative_to(self.voicebank_root))
            aliases[base_alias] = {
                "file": rel,
                "source": source_kind,
                "source_count": int(canonical.get("source_count", len(templates))),
                "subbank_count": len(distinct_subbanks),
                "subbanks": distinct_subbanks,
                "coherence": coherence,
            }
            for item in items:
                item["canonical_articulation"] = rel
                item["canonical_articulation_source"] = source_kind
                item["canonical_articulation_coherence"] = coherence

        for path in self.canonical_articulation_dir.glob("*.npz"):
            if path.name not in used_files:
                try:
                    path.unlink()
                except Exception:
                    pass
        report = {
            "format": CANONICAL_ARTICULATION_VERSION,
            "strategy": "multipitch_common_trajectory_with_timbre_neutral_fallback",
            "alias_count": len(aliases),
            "multipitch_canonical_count": multipitch_count,
            "single_neutral_fallback_count": single_count,
            "mean_coherence": float(np.mean(coherence_values)) if coherence_values else 0.0,
            "clarity_guard": "3-9kHz broad attenuation floor",
            "aliases": aliases,
        }
        self.articulation_dir.mkdir(parents=True, exist_ok=True)
        (self.articulation_dir / "index.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    def _training_caches(self, valid_caches):
        paths = [Path(p) for p in valid_caches]
        if self.mode == "profile":
            return []
        rng = random.Random(20260808)
        rng.shuffle(paths)
        if self.mode == "quick":
            return paths[: min(128, len(paths))]
        return paths

    def _multipitch_pairs(self, manifest_entries):
        groups = {}
        for item in manifest_entries:
            if item.get("status") == "error" or not item.get("cache"):
                continue
            f0 = float(item.get("median_f0_hz", 0.0))
            if f0 <= 0:
                continue
            groups.setdefault(item.get("base_alias") or item.get("alias", ""), []).append(item)
        pairs = []
        seen = set()
        for base_alias, items in groups.items():
            items = sorted(items, key=lambda x: float(x.get("median_f0_hz", 0.0)))
            if len(items) < 2:
                continue
            candidates = []
            for i in range(len(items) - 1):
                candidates.append((items[i], items[i + 1]))
            if len(items) > 2:
                candidates.append((items[0], items[-1]))
            for a, b in candidates:
                if int(a.get("subbank_index", -1)) == int(b.get("subbank_index", -1)):
                    continue
                if semitone_distance(a["median_f0_hz"], b["median_f0_hz"]) < 1.5:
                    continue
                for src, tgt in ((a, b), (b, a)):
                    key = (src["cache"], tgt["cache"])
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append({
                        "alias": base_alias,
                        "source_subbank_index": int(src.get("subbank_index", 0)),
                        "target_subbank_index": int(tgt.get("subbank_index", 0)),
                        "source": self.voicebank_root / src["cache"],
                        "target": self.voicebank_root / tgt["cache"],
                        "semitones": semitone_distance(src["median_f0_hz"], tgt["median_f0_hz"]),
                    })
        rng = random.Random(20260809)
        rng.shuffle(pairs)
        if self.mode == "quick":
            return pairs[: min(16, len(pairs))]
        limit = min(256, max(32, len(self.scan["entries"]) // 3))
        return pairs[: min(limit, len(pairs))]

    def _load_training_sample(self, path):
        data = np.load(path, allow_pickle=False)
        aug = data["timbre_aug_latent"].astype(np.float32) if "timbre_aug_latent" in data.files else data["latent"].astype(np.float32)
        aug_b = data["timbre_aug_latent_b"].astype(np.float32) if "timbre_aug_latent_b" in data.files else aug
        return {
            "latent": torch.from_numpy(data["latent"].astype(np.float32)).unsqueeze(0).to(self.engine.device),
            "timbre_aug_latent": torch.from_numpy(aug).unsqueeze(0).to(self.engine.device),
            "timbre_aug_latent_b": torch.from_numpy(aug_b).unsqueeze(0).to(self.engine.device),
            "f0": torch.from_numpy(data["f0"].astype(np.float32)).view(1, 1, -1).to(self.engine.device),
            "detail": torch.from_numpy(data["detail"].astype(np.float32)).unsqueeze(0).to(self.engine.device),
            "audio": torch.from_numpy(data["audio"].astype(np.float32)).float().view(1, 1, -1).to(self.engine.device),
            "first_voiced": int(data["first_voiced"].item()),
            "last_voiced": int(data["last_voiced"].item()),
            "articulation_raw_end": int(data["articulation_raw_end"].item()) if "articulation_raw_end" in data.files else int(data["first_voiced"].item()),
            "articulation_transition_end": int(data["articulation_transition_end"].item()) if "articulation_transition_end" in data.files else int(data["first_voiced"].item()) + 4,
            "articulation_end": int(data["articulation_end"].item()) if "articulation_end" in data.files else int(data["first_voiced"].item()) + 12,
            "articulation_confidence": float(data["articulation_confidence"].item()) if "articulation_confidence" in data.files else 0.0,
            "subbank_index": self.cache_subbank_index.get(str(Path(path).resolve())),
        }

    def _train_reconstruction_epoch(self, adapter, optimizer, paths, epoch, epochs):
        total = 0.0
        count = 0
        metrics = {}
        for idx, path in enumerate(paths, 1):
            sample = self._load_training_sample(path)
            if sample["last_voiced"] <= sample["first_voiced"] + 2:
                continue
            torch.manual_seed(stable_seed(str(path), "clarity", epoch))
            optimizer.zero_grad(set_to_none=True)
            pred = self.engine.decoder(sample["f0"], sample["latent"], adapter=adapter, detail=sample["detail"], prototype_index=sample["subbank_index"])
            target = exact_length(sample["audio"], pred.shape[-1])
            first_sample = sample["first_voiced"] * self.engine.hop
            loss, parts = clarity_reconstruction_loss(
                target.squeeze(1), pred.squeeze(1), self.engine.sr, first_sample, adapter.regularization(), pair_mode=False,
                articulation_end_sample=sample["articulation_end"] * self.engine.hop,
            )
            invariance, scrub_anchor = content_consistency_loss(
                adapter, sample["latent"], sample["timbre_aug_latent"], sample["timbre_aug_latent_b"],
                sample["first_voiced"], sample["last_voiced"],
            )
            anti_leak = 0.15 * invariance + 0.014 * scrub_anchor
            loss = loss + anti_leak
            parts["timbre_perturb_invariance"] = float(invariance.detach())
            parts["content_scrub_anchor"] = float(scrub_anchor.detach())
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.5)
            optimizer.step()
            total += float(loss.detach())
            count += 1
            for key, value in parts.items():
                metrics[key] = metrics.get(key, 0.0) + float(value)
            if idx == 1 or idx % 10 == 0 or idx == len(paths):
                print(f"reconstruction epoch {epoch + 1}/{epochs} [{idx}/{len(paths)}] loss={float(loss.detach()):.4f}", flush=True)
        return total, count, {k: v / max(1, count) for k, v in metrics.items()}

    def _train_pair_epoch(self, adapter, optimizer, pairs, epoch):
        if not pairs:
            return 0.0, 0, {}
        total = 0.0
        count = 0
        metrics = {}
        for idx, pair in enumerate(pairs, 1):
            src = self._load_training_sample(pair["source"])
            tgt = self._load_training_sample(pair["target"])
            target_frames = tgt["f0"].shape[-1]
            src_latent = F.interpolate(src["latent"], size=target_frames, mode="linear", align_corners=False)
            src_detail = F.interpolate(src["detail"], size=target_frames, mode="linear", align_corners=False)
            torch.manual_seed(stable_seed(str(pair["source"]), str(pair["target"]), epoch))
            optimizer.zero_grad(set_to_none=True)
            pred = self.engine.decoder(tgt["f0"], src_latent, adapter=adapter, detail=src_detail, prototype_index=tgt["subbank_index"])
            target = exact_length(tgt["audio"], pred.shape[-1])
            first_sample = tgt["first_voiced"] * self.engine.hop
            loss, parts = clarity_reconstruction_loss(
                target.squeeze(1), pred.squeeze(1), self.engine.sr, first_sample,
                0.5 * adapter.regularization(), pair_mode=True,
                articulation_end_sample=tgt["articulation_end"] * self.engine.hop,
            )
            pair_content = pair_content_consistency_loss(adapter, src["latent"], tgt["latent"])
            loss = 0.45 * loss + 0.090 * pair_content
            parts["pair_content_consistency"] = float(pair_content.detach())
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.25)
            optimizer.step()
            total += float(loss.detach())
            count += 1
            for key, value in parts.items():
                metrics[key] = metrics.get(key, 0.0) + float(value)
            if idx == 1 or idx % 8 == 0 or idx == len(pairs):
                print(
                    f"multipitch pair [{idx}/{len(pairs)}] alias={pair['alias']} "
                    f"distance={pair['semitones']:.1f}st loss={float(loss.detach()):.4f}",
                    flush=True,
                )
        return total, count, {k: v / max(1, count) for k, v in metrics.items()}

    def train(self, valid_caches, manifest_entries):
        paths = self._training_caches(valid_caches)
        if not paths:
            return None
        adapter_path = self.yuaz_dir / "adapter.pt"
        migrated_from = None
        if adapter_path.exists():
            try:
                adapter, old_meta = load_adapter(adapter_path, device=self.engine.device)
                migrated_from = old_meta.get("loaded_adapter_format")
                print(f"Continuing from existing voicebank adapter (format {migrated_from}).")
            except Exception:
                adapter = VoicebankAdapter(latent_dim=128, spectral_bands=64, ap_bands=16).to(self.engine.device)
        else:
            adapter = VoicebankAdapter(latent_dim=128, spectral_bands=64, ap_bands=16).to(self.engine.device)
        pitched = [float(x.get("median_f0_hz", 0.0)) for x in manifest_entries if float(x.get("median_f0_hz", 0.0)) > 20.0]
        if pitched:
            adapter.set_bank_median_f0(float(np.median(pitched)))
        subbanks = (self.subbank_info or {}).get("subbanks", [])
        anchors = [float(x["anchor_midi"]) for x in subbanks] or [69.0 + 12.0 * math.log2(max(20.0, float(adapter.bank_median_f0_hz)) / 440.0)]
        adapter.configure_pitch_prototypes(anchors)
        routing_name = "Prefix-authoritative" if (self.subbank_info or {}).get("prefix_map_authoritative") else "UTAU-native fallback"
        print(f"{routing_name} timbre prototypes: {len(anchors)} -> " + ", ".join(f"{x.get('label', i)}@{x.get('anchor_note', '?')}" for i, x in enumerate(subbanks)))
        adapter.train()
        for param in self.engine.encoder.parameters():
            param.requires_grad_(False)
        for param in self.engine.decoder.parameters():
            param.requires_grad_(False)

        lr = 0.0018 if self.mode == "quick" else 0.0012
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
        epochs = 1 if self.mode == "quick" else 3
        pairs = self._multipitch_pairs(manifest_entries)
        print(f"Using {len(paths)} reconstruction samples and {len(pairs)} real multipitch pair directions.")
        history = []
        for epoch in range(epochs):
            total, count, parts = self._train_reconstruction_epoch(adapter, optimizer, paths, epoch, epochs)
            pair_total, pair_count, pair_parts = self._train_pair_epoch(adapter, optimizer, pairs, epoch)
            mean = total / max(1, count)
            pair_mean = pair_total / max(1, pair_count) if pair_count else 0.0
            history.append({
                "epoch": epoch + 1,
                "mean_loss": mean,
                "samples": count,
                "clarity_metrics": parts,
                "multipitch_pair_mean_loss": pair_mean,
                "multipitch_pair_updates": pair_count,
                "multipitch_pair_metrics": pair_parts,
            })
            print(f"epoch {epoch + 1} reconstruction mean: {mean:.4f}; pair mean: {pair_mean:.4f}")
        metadata = {
            "voicebank_id": self.bank_id,
            "voicebank_root": str(self.voicebank_root),
            "mode": self.mode,
            "training_version": CLARITY_TRAINING_VERSION,
            "anti_leak_training_version": ANTI_LEAK_TRAINING_VERSION,
            "articulation_training_version": ARTICULATION_TRAINING_VERSION,
            "trained_samples": len(paths),
            "epochs": epochs,
            "multipitch_pair_directions": len(pairs),
            "detail_feature_dim": DEFAULT_DETAIL_DIM,
            "migrated_from_adapter_format": migrated_from,
            "losses": [
                "multi_resolution_log_spectral",
                "high_frequency_detail",
                "formant_gradient",
                "formant_curvature",
                "spectral_contrast",
                "oto_onset_high_frequency",
                "real_multipitch_pair",
                "timbre_perturb_content_invariance",
                "multipitch_content_consistency",
                "explicit_voicebank_timbre_code",
                "dual_timbre_perturbation_invariance",
                "utau_native_dynamic_timbre_prototypes",
                "prefix_map_base_alias_multipitch_pairs",
                "source_subbank_routing",
                "prefix_map_authoritative_routing",
                "fallback_prototype_suppression",
                "oto_guided_articulation_trajectory",
                "voiced_onset_temporal_spectral_trajectory",
            ],
            "utau_subbank_count": len(subbanks),
            "utau_subbanks": subbanks,
            "history": history,
            "adapter_summary": adapter.summary(),
            "created_at": time.time(),
        }
        save_adapter(adapter_path, adapter, metadata)
        torch.save({
            "format": 3,
            "bank_median_f0_hz": float(adapter.bank_median_f0_hz.detach().cpu()),
            "global_timbre_code": adapter.timbre_code.detach().cpu(),
            "pitch_timbre_codes": adapter.pitch_timbre_codes.detach().cpu(),
            "pitch_prototype_midi": adapter.pitch_prototype_midi.detach().cpu(),
            "subbanks": subbanks,
            "labels": [x.get("label", f"subbank-{i}") for i, x in enumerate(subbanks)],
        }, self.yuaz_dir / "timbre_profiles.pt")
        (self.yuaz_dir / "training.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return metadata

    def _fidelity_training_paths(self, valid_caches):
        paths = [Path(p) for p in valid_caches]
        rng = random.Random(20260810)
        rng.shuffle(paths)
        limit = 64 if self.mode == "quick" else 256
        return paths[: min(limit, len(paths))]

    def train_fidelity_refiner(self, valid_caches):
        if self.mode == "profile":
            return None
        adapter_path = self.yuaz_dir / "adapter.pt"
        if not adapter_path.exists():
            return None
        adapter, _ = load_adapter(adapter_path, device=self.engine.device)
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad_(False)

        refiner_path = self.yuaz_dir / "fidelity_refiner.pt"
        migrated = None
        if refiner_path.exists():
            try:
                refiner, old_meta = load_refiner(refiner_path, device=self.engine.device)
                migrated = old_meta.get("loaded_refiner_format")
                print(f"Continuing from existing fidelity refiner (format {migrated}).")
            except Exception:
                refiner = TinyFidelityRefiner(detail_dim=DEFAULT_DETAIL_DIM).to(self.engine.device)
        else:
            refiner = TinyFidelityRefiner(detail_dim=DEFAULT_DETAIL_DIM).to(self.engine.device)
        refiner.train()
        paths = self._fidelity_training_paths(valid_caches)
        epochs = 1 if self.mode == "quick" else 2
        optimizer = torch.optim.AdamW(refiner.parameters(), lr=8e-4 if self.mode == "quick" else 5e-4, weight_decay=1e-5)
        history = []
        for epoch in range(epochs):
            total = 0.0
            count = 0
            residual_ratios = []
            for idx, path in enumerate(paths, 1):
                sample = self._load_training_sample(path)
                if sample["last_voiced"] <= sample["first_voiced"] + 2:
                    continue
                torch.manual_seed(stable_seed(str(path), "fidelity-base", epoch))
                with torch.no_grad():
                    base = self.engine.decoder(sample["f0"], sample["latent"], adapter=adapter, detail=sample["detail"], prototype_index=sample["subbank_index"])
                target = exact_length(sample["audio"], base.shape[-1])
                optimizer.zero_grad(set_to_none=True)
                articulation_end_sample = sample["articulation_end"] * self.engine.hop
                refined, residual = refiner(
                    base.detach(), sample["detail"], sample["f0"], articulation_end_sample=articulation_end_sample
                )
                first_sample = sample["first_voiced"] * self.engine.hop
                loss, parts = clarity_reconstruction_loss(
                    target.squeeze(1), refined.squeeze(1), self.engine.sr, first_sample, pair_mode=False,
                    articulation_end_sample=sample["articulation_end"] * self.engine.hop,
                )
                base_rms = torch.sqrt(torch.mean(base.detach().pow(2)) + 1e-8)
                residual_rms = torch.sqrt(torch.mean(residual.pow(2)) + 1e-8)
                ratio = residual_rms / (base_rms + 1e-8)
                residual_penalty = torch.relu(ratio - 0.09).pow(2)
                loss = 0.80 * loss + 0.035 * residual_penalty + 0.002 * refiner.regularization()
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach())
                count += 1
                residual_ratios.append(float(ratio.detach()))
                if idx == 1 or idx % 10 == 0 or idx == len(paths):
                    print(f"fidelity epoch {epoch + 1}/{epochs} [{idx}/{len(paths)}] loss={float(loss.detach()):.4f} residual={float(ratio.detach()):.3f}", flush=True)
            history.append({
                "epoch": epoch + 1,
                "mean_loss": total / max(1, count),
                "updates": count,
                "mean_residual_rms_ratio": float(np.mean(residual_ratios)) if residual_ratios else 0.0,
            })
        metadata = {
            "training_version": FIDELITY_TRAINING_VERSION,
            "articulation_training_version": ARTICULATION_TRAINING_VERSION,
            "mode": self.mode,
            "trained_samples": len(paths),
            "epochs": epochs,
            "migrated_from_refiner_format": migrated,
            "residual_rms_hard_limit": 0.11,
            "history": history,
            "refiner_summary": refiner.summary(),
            "created_at": time.time(),
        }
        save_refiner(refiner_path, refiner, metadata)
        (self.yuaz_dir / "fidelity_training.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return metadata

    def register(self, manifest_entries):
        registry_path = Path(self.config.get("registry_path") or (self.project_root / "voicebank_registry.json")).expanduser().resolve()
        registry = {"format": 4, "samples": {}}
        if registry_path.exists():
            try:
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        samples = registry.setdefault("samples", {})
        for key in list(samples.keys()):
            record = samples.get(key)
            if isinstance(record, dict) and record.get("voicebank_id") == self.bank_id:
                samples.pop(key, None)
        adapter = self.yuaz_dir / "adapter.pt"
        refiner = self.yuaz_dir / "fidelity_refiner.pt"
        loudness_path = self.yuaz_dir / "loudness.json"
        articulation_index_path = self.articulation_dir / "index.json"

        def add_record(key, item):
            if not key:
                return
            record = samples.get(key)
            if not isinstance(record, dict) or record.get("voicebank_id") != self.bank_id:
                record = {
                    "voicebank_id": self.bank_id,
                    "voicebank_root": str(self.voicebank_root),
                    "profile": str(self.yuaz_dir / "profile.json"),
                    "adapter": str(adapter) if adapter.exists() else "",
                    "refiner": str(refiner) if refiner.exists() else "",
                    "loudness": str(loudness_path) if loudness_path.exists() else "",
                    "articulation_index": str(articulation_index_path) if articulation_index_path.exists() else "",
                    "loudness_enabled": bool(self.loudness_enabled),
                    "loudness_target_dbfs": float(self.loudness_target_dbfs),
                    "loudness_peak_ceiling_dbfs": float(self.loudness_peak_ceiling_dbfs),
                    "loudness_peak_guard_knee_db": float(self.loudness_peak_guard_knee_db),
                    "loudness_emergency_max_abs_gain_db": float(self.loudness_emergency_max_abs_gain_db),
                    "loudness_tolerance_db": float(self.loudness_tolerance_db),
                    "adapter_generation": "canonical-articulation-v4" if adapter.exists() else "base",
                    "subbank_id": item.get("subbank_id"),
                    "subbank_index": item.get("subbank_index"),
                    "subbank_label": item.get("subbank_label"),
                    "subbank_anchor_midi": item.get("subbank_anchor_midi"),
                    "base_alias": item.get("base_alias"),
                    "source_loudness_variants": [],
                }
            variants = record.setdefault("source_loudness_variants", [])
            signature = item.get("loudness_signature") or oto_loudness_signature(item.get("offset", 0.0), item.get("consonant", 0.0), item.get("cutoff", 0.0))
            variants[:] = [v for v in variants if v.get("signature") != signature]
            variants.append({
                "signature": signature,
                "alias": item.get("alias"),
                "offset": float(item.get("offset", 0.0)),
                "consonant": float(item.get("consonant", 0.0)),
                "cutoff": float(item.get("cutoff", 0.0)),
                "active_rms_dbfs": float(item.get("active_rms_dbfs", -120.0)),
                "diagnostic_gain_to_target_db": float(item.get("diagnostic_gain_to_target_db", 0.0)),
                "base_alias": item.get("base_alias"),
                "subbank_label": item.get("subbank_label"),
                "subbank_index": item.get("subbank_index"),
                "canonical_articulation": item.get("canonical_articulation"),
                "canonical_articulation_source": item.get("canonical_articulation_source"),
                "canonical_articulation_coherence": item.get("canonical_articulation_coherence"),
            })
            samples[key] = record

        for item in manifest_entries:
            if item.get("status") == "error":
                continue
            if item.get("sha256"):
                add_record("sha256:" + item["sha256"], item)
            if item.get("pcm"):
                add_record("pcm:" + item["pcm"], item)
        registry["format"] = 4
        registry["updated_at"] = time.time()
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
        return registry_path

    def run(self):
        profile, entries, caches = self.profile()
        metadata = self.train(caches, entries)
        fidelity = self.train_fidelity_refiner(caches)
        registry = self.register(entries)
        result = {"profile": profile, "training": metadata, "fidelity": fidelity, "registry": str(registry), "yuaz_dir": str(self.yuaz_dir)}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("voicebank")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--mode", choices=("profile", "quick", "deep"), default="quick")
    args = parser.parse_args()
    VoicebankPreparer(args.project_root, args.voicebank, args.mode).run()


if __name__ == "__main__":
    main()
