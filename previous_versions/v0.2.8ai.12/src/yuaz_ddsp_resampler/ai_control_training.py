#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from .ai_vocal_controls import AIControlAdapter, AP_BANDS, SPECTRAL_BANDS, save_ai_control_adapter
from .control_training import discover_gtsinger_pairs
from .core import YuazDDSPResamplerEngine, extract_f0, read_audio
from .state import sha256

SR = 24000
N_FFT = 1024
HOP = 256


def _resize_freq(x, bands):
    x = np.asarray(x, dtype=np.float32)
    old = np.linspace(0.0, 1.0, x.shape[0], dtype=np.float64)
    new = np.linspace(0.0, 1.0, int(bands), dtype=np.float64)
    out = np.empty((int(bands), x.shape[1]), dtype=np.float32)
    for i in range(x.shape[1]):
        out[:, i] = np.interp(new, old, x[:, i]).astype(np.float32)
    return out


def _resize_time(x, frames):
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] == int(frames):
        return x
    old = np.linspace(0.0, 1.0, x.shape[-1], dtype=np.float64)
    new = np.linspace(0.0, 1.0, int(frames), dtype=np.float64)
    flat = x.reshape(-1, x.shape[-1])
    out = np.empty((flat.shape[0], int(frames)), dtype=np.float32)
    for i, row in enumerate(flat):
        out[i] = np.interp(new, old, row).astype(np.float32)
    return out.reshape(x.shape[:-1] + (int(frames),))


def _logit_np(x, eps=1e-4):
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x) - np.log1p(-x)


class NativeYuazDDSPExtractor:
    """Frozen Yuaz analysis used as the primary foundation-training backend."""

    def __init__(self, project_root):
        project_root = Path(project_root).expanduser().resolve()
        cfg_path = project_root / "config.json"
        if not cfg_path.is_file():
            raise RuntimeError("config.json not found; run configure-macos.command before AI foundation training.")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.project_root = project_root
        self.checkpoint = Path(cfg["checkpoint"]).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise RuntimeError(f"Yuaz checkpoint not found: {self.checkpoint}")
        self.engine = YuazDDSPResamplerEngine(
            cfg["yuaz_repo"], self.checkpoint,
            transition_ms=cfg.get("transition_ms", 70),
            use_rvq=cfg.get("use_rvq", False),
            output_sr=cfg.get("output_sr", 44100),
            registry_path=None,
        )
        self.checkpoint_sha256 = sha256(self.checkpoint)
        self.sample_rate = int(self.engine.sr)
        self.hop = int(self.engine.hop)

    def features(self, path):
        audio = read_audio(path, self.sample_rate)
        audio = np.asarray(audio, dtype=np.float32)
        # Keep the original segment time axis intact. GTSinger technique JSON uses
        # per-phoneme times relative to the WAV, so trimming would misalign direct
        # technique supervision with the Yuaz DDSP frames.
        if audio.size < int(0.25 * self.sample_rate):
            raise ValueError("audio too short")
        f0_np = extract_f0(audio, self.sample_rate, self.hop).astype(np.float32)
        audio_t = torch.from_numpy(audio).float().view(1, 1, -1)
        f0_t = torch.from_numpy(f0_np).float().view(1, 1, -1)
        with torch.inference_mode():
            z, f0_aligned = self.engine.encoder(audio_t, f0_override=f0_t)
            state = self.engine.decoder.extract_neural_ddsp_state(f0_aligned, z)
        spectral = state["spectral_envelope"][0].cpu().numpy().astype(np.float32)
        ap = state["ap"][0].cpu().numpy().astype(np.float32)
        gate = state["gate"][0].cpu().numpy().astype(np.float32)
        f0 = state["f0"][0].cpu().numpy().astype(np.float32)
        log_spec = _resize_freq(np.log(np.maximum(spectral, 1e-7)), SPECTRAL_BANDS)
        ap = _resize_freq(ap, AP_BANDS)
        # Yuaz S is a spectral envelope, so use its high/total envelope energy
        # ratio only as a continuous tension-coordinate proxy. The actual target
        # residual remains the full native Yuaz DDSP state difference.
        hz = np.linspace(0.0, self.sample_rate / 2.0, spectral.shape[0], dtype=np.float32)
        hi = hz >= 2200.0
        spow = spectral * spectral
        tension = (np.sum(spow[hi], axis=0) / (np.sum(spow, axis=0) + 1e-8)).reshape(1, -1).astype(np.float32)
        return {
            "log_spec": log_spec, "ap": ap, "gate": gate, "f0": f0, "tension": tension,
        }


def _frame_features(path):
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    y = np.asarray(y, dtype=np.float32)
    # Preserve the source time axis for phoneme-level technique masks.
    if y.size < int(0.25 * SR):
        raise ValueError("audio too short")
    mag = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, center=True)).astype(np.float32)
    mag = np.maximum(mag, 1e-7)
    harmonic, residual = librosa.decompose.hpss(mag, kernel_size=(17, 17))
    total = harmonic + residual + 1e-7
    log_spec = _resize_freq(np.log(total), SPECTRAL_BANDS)
    ap = _resize_freq(residual / total, AP_BANDS)
    h_energy = np.sqrt(np.mean(harmonic * harmonic, axis=0) + 1e-12)
    r_energy = np.sqrt(np.mean(residual * residual, axis=0) + 1e-12)
    gate = (h_energy / (h_energy + r_energy + 1e-8)).reshape(1, -1).astype(np.float32)
    try:
        f0 = librosa.yin(y, fmin=55.0, fmax=1200.0, sr=SR, frame_length=N_FFT, hop_length=HOP)
        f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32).reshape(1, -1)
    except Exception:
        f0 = np.zeros((1, log_spec.shape[-1]), dtype=np.float32)
    f0 = _resize_time(f0, log_spec.shape[-1])
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    hi = freqs >= 2200.0
    harmonic_pow = harmonic * harmonic
    tension = (np.sum(harmonic_pow[hi], axis=0) / (np.sum(harmonic_pow, axis=0) + 1e-8)).reshape(1, -1).astype(np.float32)
    return {"log_spec": log_spec, "ap": ap, "gate": gate, "f0": f0, "tension": tension}


SUPPORTED_TECHNIQUES = {
    "breathy": "breathiness",
    "falsetto": "falsetto",
    "mixed_voice": "mixed_voice",
    "pharyngeal": "pharyngeal",
}
CONTROL_ORDER = ("breathiness", "falsetto", "mixed_voice", "pharyngeal")
TECHNIQUE_JSON_KEYS = {
    "breathy": "breathy",
    "falsetto": "falsetto",
    "mixed_voice": "mix",
    "pharyngeal": "pharyngeal",
}


def _label_on(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    try:
        return float(value) > 0.5
    except (TypeError, ValueError):
        return False


def _technique_control_curve(technique_wav, technique, frames):
    """Return [control, time] direct labels aligned to this WAV's frame axis."""
    axis = SUPPORTED_TECHNIQUES.get(str(technique))
    if axis is None:
        return None, {"annotation": "unsupported", "active_fraction": 0.0}
    controls = np.zeros((len(CONTROL_ORDER), int(frames)), dtype=np.float32)
    axis_index = CONTROL_ORDER.index(axis)
    json_path = Path(technique_wav).with_suffix(".json")
    if not json_path.is_file():
        controls[axis_index, :] = 1.0
        return controls, {"annotation": "group-fallback", "active_fraction": 1.0, "json": ""}
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        duration_s = float(librosa.get_duration(path=str(technique_wav)))
        if duration_s <= 1e-6:
            raise ValueError("invalid WAV duration")
        key = TECHNIQUE_JSON_KEYS[str(technique)]
        intervals = 0
        active_s = 0.0
        for word in payload if isinstance(payload, list) else []:
            starts = list(word.get("ph_start") or [])
            ends = list(word.get("ph_end") or [])
            labels = list(word.get(key) or [])
            for start, end, label in zip(starts, ends, labels):
                if not _label_on(label):
                    continue
                start = max(0.0, min(duration_s, float(start)))
                end = max(start, min(duration_s, float(end)))
                if end <= start:
                    continue
                i0 = max(0, min(int(frames) - 1, int(math.floor(start / duration_s * int(frames)))))
                i1 = max(i0 + 1, min(int(frames), int(math.ceil(end / duration_s * int(frames)))))
                controls[axis_index, i0:i1] = 1.0
                intervals += 1
                active_s += end - start
        fraction = float(np.mean(controls[axis_index])) if int(frames) else 0.0
        if intervals == 0 or fraction <= 0.0:
            return None, {"annotation": "json-no-active-label", "active_fraction": 0.0, "json": str(json_path)}
        return controls, {
            "annotation": "phoneme-json",
            "active_fraction": fraction,
            "active_seconds": float(active_s),
            "active_intervals": int(intervals),
            "json": str(json_path),
        }
    except Exception as exc:
        controls[axis_index, :] = 1.0
        return controls, {
            "annotation": "group-fallback-json-error",
            "active_fraction": 1.0,
            "json": str(json_path),
            "annotation_error": str(exc),
        }


def _pair_controls(technique, src=None, tgt=None):
    del src, tgt
    controls = np.zeros(4, dtype=np.float32)
    axis = SUPPORTED_TECHNIQUES.get(str(technique))
    if axis is None:
        return None
    index = CONTROL_ORDER.index(axis)
    controls[index] = 1.0
    return controls


def _safe_stem(text):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text))[:90]


def build_gtsinger_dataset(root, out_dir, max_pairs=0, project_root=None, feature_backend="yuaz-native"):
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    native = None
    backend = str(feature_backend).strip().lower()
    if backend == "yuaz-native":
        if not project_root:
            raise RuntimeError("yuaz-native feature backend requires --project-root")
        native = NativeYuazDDSPExtractor(project_root)
        provenance = {
            "feature_backend": "yuaz-native-ddsp-v1",
            "checkpoint_sha256": native.checkpoint_sha256,
            "sample_rate": native.sample_rate,
            "hop": native.hop,
        }
    elif backend == "stft-proxy":
        provenance = {"feature_backend": "stft-proxy-v1", "sample_rate": SR, "hop": HOP}
    else:
        raise RuntimeError(f"Unknown feature backend: {feature_backend}")
    manifest = {
        "format": 4,
        "source": "GTSinger paired Control_Group/Technique_Group",
        "supervision": "phoneme-level-direct-technique-labels",
        "direct_controls": ["breathiness", "falsetto", "mixed_voice", "pharyngeal"],
        "created_at": time.time(),
        "provenance": provenance,
        "pairs": [], "errors": [], "skipped_techniques": {},
    }
    count = 0
    for technique, control_wav, technique_wav in discover_gtsinger_pairs(root):
        if technique not in SUPPORTED_TECHNIQUES:
            manifest["skipped_techniques"][technique] = int(manifest["skipped_techniques"].get(technique, 0)) + 1
            continue
        if max_pairs and count >= int(max_pairs):
            break
        key = hashlib.sha1((str(control_wav) + "|" + str(technique_wav)).encode()).hexdigest()[:16]
        shard = out_dir / f"{count:06d}-{_safe_stem(technique)}-{key}.npz"
        if shard.exists():
            manifest["pairs"].append({"shard": shard.name, "technique": technique, "cached": True})
            count += 1
            continue
        try:
            if native is not None:
                src = native.features(control_wav)
                tgt = native.features(technique_wav)
            else:
                src = _frame_features(control_wav)
                tgt = _frame_features(technique_wav)
            frames = min(src["log_spec"].shape[-1], 1800)
            if frames < 20:
                raise ValueError("too few frames")
            for name in ("log_spec", "ap", "gate", "f0", "tension"):
                src[name] = _resize_time(src[name], frames)
                tgt[name] = _resize_time(tgt[name], frames)
            controls, annotation = _technique_control_curve(technique_wav, technique, frames)
            if controls is None:
                raise ValueError(f"no active direct {technique} annotation in technique JSON")
            ds = np.clip((tgt["log_spec"] - src["log_spec"]) / 0.72, -1.5, 1.5).astype(np.float32)
            da = np.clip((_logit_np(tgt["ap"]) - _logit_np(src["ap"])) / 1.35, -1.5, 1.5).astype(np.float32)
            dg = np.clip((_logit_np(tgt["gate"]) - _logit_np(src["gate"])) / 1.35, -1.5, 1.5).astype(np.float32)
            np.savez_compressed(
                shard,
                spectral=np.exp(src["log_spec"]).astype(np.float32),
                ap=src["ap"].astype(np.float32), gate=src["gate"].astype(np.float32), f0=src["f0"].astype(np.float32),
                controls=controls, target_ds=ds, target_da=da, target_dg=dg,
                annotation=np.asarray(annotation.get("annotation", "")),
                active_fraction=np.asarray(annotation.get("active_fraction", 0.0), dtype=np.float32),
                technique=np.asarray(technique),
                feature_backend=np.asarray(provenance["feature_backend"]),
                checkpoint_sha256=np.asarray(provenance.get("checkpoint_sha256", "")),
            )
            manifest["pairs"].append({
                "shard": shard.name, "technique": technique, "control": str(control_wav), "target": str(technique_wav),
                "annotation": annotation.get("annotation", ""), "active_fraction": annotation.get("active_fraction", 0.0),
            })
            count += 1
            if count % 10 == 0:
                print(f"Prepared {count} paired examples", flush=True)
        except Exception as exc:
            manifest["errors"].append({"control": str(control_wav), "target": str(technique_wav), "error": str(exc)})
    (out_dir / "dataset.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if count == 0:
        raise RuntimeError("No usable GTSinger pairs were prepared.")
    print(f"Prepared dataset: {out_dir} ({count} pairs)")
    return out_dir


def _tensor(a):
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).unsqueeze(0)


def _windows(npz, window=160, max_windows=6):
    data = np.load(npz, allow_pickle=False)
    T = int(data["spectral"].shape[-1])
    if T <= window:
        starts = [0]
        window = T
    else:
        n = min(max_windows, max(1, T // window))
        starts = np.linspace(0, T - window, n).astype(int).tolist()
        random.shuffle(starts)
    for st in starts:
        en = st + window
        yield {
            "spectral": _tensor(data["spectral"][:, st:en]),
            "ap": _tensor(data["ap"][:, st:en]),
            "gate": _tensor(data["gate"][:, st:en]),
            "f0": _tensor(data["f0"][:, st:en]),
            "controls": (
                np.asarray(data["controls"], dtype=np.float32)[:, st:en]
                if np.asarray(data["controls"]).ndim == 2
                else np.asarray(data["controls"], dtype=np.float32)
            ),
            "target_ds": _tensor(data["target_ds"][:, st:en]),
            "target_da": _tensor(data["target_da"][:, st:en]),
            "target_dg": _tensor(data["target_dg"][:, st:en]),
        }


def _control_dict(controls, frames):
    c = torch.from_numpy(np.asarray(controls, dtype=np.float32)).float()
    if c.dim() == 1:
        return {name: torch.full((1, 1, int(frames)), float(c[i])) for i, name in enumerate(CONTROL_ORDER)}
    if c.dim() != 2 or c.shape[0] != len(CONTROL_ORDER):
        raise ValueError(f"invalid control tensor shape: {tuple(c.shape)}")
    if c.shape[-1] != int(frames):
        c = F.interpolate(c.unsqueeze(0), size=int(frames), mode="nearest")[0]
    return {name: c[i].view(1, 1, -1) for i, name in enumerate(CONTROL_ORDER)}


def train(dataset_dir, output, epochs=10, lr=2e-4, window=160, max_windows=6, seed=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    shards = sorted(dataset_dir.glob("*.npz"))
    dataset_manifest_path = dataset_dir / "dataset.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8")) if dataset_manifest_path.is_file() else {}
    provenance = dict(dataset_manifest.get("provenance") or {})
    if not shards:
        raise RuntimeError("No training shards found. Run build-gtsinger first.")
    val = [p for p in shards if int(hashlib.sha1(p.name.encode()).hexdigest()[:4], 16) % 10 == 0]
    train_shards = [p for p in shards if p not in val]
    if not val:
        val = shards[:1]
        train_shards = shards[1:] if len(shards) > 1 else shards[:1]
    if not train_shards:
        train_shards = shards[:1]
    model = AIControlAdapter()
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    checkpoint = Path(str(output) + ".training.pt")
    start_epoch = 0
    best = float("inf")
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        opt.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", -1)) + 1
        best = float(payload.get("best", best))
        print(f"Resuming foundation training at epoch {start_epoch + 1}")

    def run_files(files, training):
        model.train(training)
        total = 0.0; n = 0
        order = list(files)
        if training: random.shuffle(order)
        for shard in order:
            for batch in _windows(shard, window=window, max_windows=max_windows if training else 2):
                frames = batch["spectral"].shape[-1]
                controls = _control_dict(batch["controls"], frames)
                ds, da, dg = model.predict_residuals(batch["spectral"], batch["ap"], batch["gate"], batch["f0"], controls)
                voiced = (batch["f0"] > 1.0).to(batch["spectral"].dtype)
                activity = torch.amax(torch.cat([controls[name] for name in CONTROL_ORDER], dim=1), dim=1, keepdim=True)
                supervised = voiced * (activity > 0.5).to(voiced.dtype)
                def masked_huber(pred, target):
                    raw = F.smooth_l1_loss(torch.tanh(pred), torch.clamp(target, -1.0, 1.0), reduction="none")
                    mask = supervised.expand(raw.shape[0], raw.shape[1], raw.shape[2])
                    return torch.sum(raw * mask) / torch.clamp(mask.sum(), min=1.0)
                loss_s = masked_huber(ds, batch["target_ds"])
                loss_a = masked_huber(da, batch["target_da"])
                loss_g = masked_huber(dg, batch["target_dg"])
                # Explicit zero-control identity regularizer prevents the learned
                # controller from becoming another timbre adapter.
                zeros = {k: torch.zeros_like(v) for k, v in controls.items()}
                zds, zda, zdg = model.predict_residuals(batch["spectral"], batch["ap"], batch["gate"], batch["f0"], zeros)
                zero_loss = zds.abs().mean() + zda.abs().mean() + zdg.abs().mean()
                loss = loss_s + 0.85 * loss_a + 0.75 * loss_g + 0.15 * zero_loss
                if training:
                    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
                total += float(loss.detach()); n += 1
        return total / max(1, n)

    history = []
    for epoch in range(start_epoch, int(epochs)):
        train_loss = run_files(train_shards, True)
        with torch.inference_mode(): val_loss = run_files(val, False)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch + 1}/{epochs}: train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch, "best": min(best, val_loss)}, checkpoint)
        if val_loss <= best:
            best = val_loss
            save_ai_control_adapter(output, model, metadata={
                "training_source": "GTSinger paired singing",
                "training_method": "paired Yuaz-native DDSP direct-technique residual foundation v2",
                "feature_backend": provenance.get("feature_backend", "unknown"),
                "checkpoint_sha256": provenance.get("checkpoint_sha256", ""),
                "sample_rate": provenance.get("sample_rate"),
                "hop": provenance.get("hop"),
                "controls": ["breathiness", "falsetto", "mixed_voice", "pharyngeal"],
                "pair_count": len(shards), "validation_pair_count": len(val),
                "best_validation_loss": best, "epoch": epoch + 1,
                "created_at": time.time(),
                "direct_controls": ["breathiness", "falsetto", "mixed_voice", "pharyngeal"],
                "supervision": "phoneme-level-direct-technique-labels",
                "license_note": "Model provenance includes GTSinger; preserve compatible dataset/derived-weight rights.",
            })
    meta = {"format": 2, "best_validation_loss": best, "history": history, "output": str(Path(output).resolve())}
    Path(str(output) + ".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return output



def print_gtsinger_coverage(root):
    counts = {}
    total = 0
    for technique, _control_wav, _technique_wav in discover_gtsinger_pairs(root):
        counts[technique] = counts.get(technique, 0) + 1
        total += 1
    print("GTSinger paired coverage:")
    for name in sorted(counts):
        direct = "DIRECT" if name in SUPPORTED_TECHNIQUES else "future/ignored"
        print(f"  {name:24s} {counts[name]:6d} pairs  [{direct}]")
    direct_total = sum(counts.get(k, 0) for k in SUPPORTED_TECHNIQUES)
    print(f"Direct-technique training pairs: {direct_total} / {total}")
    missing = [k for k in SUPPORTED_TECHNIQUES if counts.get(k, 0) == 0]
    if missing:
        print("Missing direct techniques: " + ", ".join(missing))
    return 0 if direct_total else 2

def main():
    p = argparse.ArgumentParser(description="Train Yuaz neural DDSP vocal-control foundation model.")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-gtsinger")
    b.add_argument("dataset_root"); b.add_argument("output_dir"); b.add_argument("--max-pairs", type=int, default=0)
    b.add_argument("--project-root", default=None)
    b.add_argument("--feature-backend", choices=("yuaz-native", "stft-proxy"), default="yuaz-native")
    c = sub.add_parser("coverage")
    c.add_argument("dataset_root")
    t = sub.add_parser("train")
    t.add_argument("dataset_dir"); t.add_argument("output")
    t.add_argument("--epochs", type=int, default=10); t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--window", type=int, default=160); t.add_argument("--max-windows", type=int, default=6)
    a = p.parse_args()
    if a.cmd == "build-gtsinger":
        build_gtsinger_dataset(a.dataset_root, a.output_dir, a.max_pairs, a.project_root, a.feature_backend)
    elif a.cmd == "coverage":
        raise SystemExit(print_gtsinger_coverage(a.dataset_root))
    else:
        train(a.dataset_dir, a.output, a.epochs, a.lr, a.window, a.max_windows)


if __name__ == "__main__":
    main()
