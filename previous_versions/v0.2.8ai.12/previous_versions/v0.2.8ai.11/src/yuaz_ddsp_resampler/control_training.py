#!/usr/bin/env python3
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np


TARGET_SR = 24000
N_FFT = 1024
HOP = 256
SPECTRAL_BANDS = 64
AP_BANDS = 16


def _resample_vector(x, n):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.zeros(int(n), dtype=np.float32)
    if x.size == 1:
        return np.full(int(n), float(x[0]), dtype=np.float32)
    src = np.linspace(0.0, 1.0, x.size, dtype=np.float64)
    dst = np.linspace(0.0, 1.0, int(n), dtype=np.float64)
    return np.interp(dst, src, x.astype(np.float64)).astype(np.float32)


def _normalize_technique(name):
    text = re.sub(r"_group$", "", str(name), flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    aliases = {
        "mixed_voice_and_falsetto": "mixed_voice_falsetto",
        "mixed_voice": "mixed_voice",
        "falsetto": "falsetto",
        "breathy": "breathy",
        "pharyngeal": "pharyngeal",
    }
    return aliases.get(text, text)


def _audio_signature(path):
    y, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    y = np.asarray(y, dtype=np.float32)
    if y.size < int(0.18 * TARGET_SR):
        raise ValueError("audio too short")
    y, _ = librosa.effects.trim(y, top_db=48)
    if y.size < int(0.12 * TARGET_SR):
        raise ValueError("trimmed audio too short")
    mag = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, center=True)).astype(np.float32)
    mag = np.maximum(mag, 1e-7)
    harmonic, residual = librosa.decompose.hpss(mag, kernel_size=(17, 17))
    total = harmonic + residual + 1e-7
    ap = residual / total
    spec = np.median(np.log(mag), axis=1)
    ap_profile = np.median(ap, axis=1)
    harmonic_energy = float(np.sqrt(np.mean(harmonic * harmonic) + 1e-12))
    residual_energy = float(np.sqrt(np.mean(residual * residual) + 1e-12))
    gate = harmonic_energy / (harmonic_energy + residual_energy + 1e-9)

    # OpenVPI-compatible proxy statistics. The exact DiffSinger extractor uses a
    # harmonic/aperiodic decomposition and isolates the base harmonic. Here we
    # keep the offline trainer dependency-light while preserving the same
    # directionality: breathiness=residual energy, voicing=harmonic energy,
    # tension=harmonic energy excluding the fundamental relative to total.
    try:
        f0 = librosa.yin(
            y, fmin=65.0, fmax=min(1200.0, TARGET_SR / 2.5), sr=TARGET_SR,
            frame_length=N_FFT, hop_length=HOP,
        )
        n_frames = min(harmonic.shape[1], len(f0))
        freqs = librosa.fft_frequencies(sr=TARGET_SR, n_fft=N_FFT)
        total_pow = np.sum(harmonic[:, :n_frames] ** 2, axis=0) + 1e-12
        base_pow = np.zeros(n_frames, dtype=np.float64)
        bin_hz = float(TARGET_SR) / float(N_FFT)
        for i in range(n_frames):
            hz = float(f0[i])
            if not np.isfinite(hz) or hz <= 1.0:
                continue
            mask = np.abs(freqs - hz) <= max(1.5 * bin_hz, 0.06 * hz)
            base_pow[i] = float(np.sum(harmonic[mask, i] ** 2))
        ratio = np.sqrt(np.maximum(total_pow - base_pow, 0.0) / total_pow)
        tension = float(np.median(ratio[np.isfinite(ratio)])) if np.any(np.isfinite(ratio)) else 0.0
    except Exception:
        tension = 0.0
    return {
        "spectral_log": _resample_vector(spec, SPECTRAL_BANDS),
        "ap": _resample_vector(ap_profile, AP_BANDS),
        "gate": float(gate),
        "rms": float(np.sqrt(np.mean(y * y) + 1e-12)),
        "breathiness_rms": residual_energy,
        "voicing_rms": harmonic_energy,
        "tension": float(tension),
    }


def _best_pairs(control_dir, technique_dir):
    controls = sorted(control_dir.rglob("*.wav"))
    techniques = sorted(technique_dir.rglob("*.wav"))
    by_relative = {str(p.relative_to(technique_dir)).lower(): p for p in techniques}
    by_stem = defaultdict(list)
    for p in techniques:
        by_stem[p.stem.lower()].append(p)
    pairs = []
    used = set()
    for c in controls:
        rel = str(c.relative_to(control_dir)).lower()
        candidate = by_relative.get(rel)
        if candidate in used:
            candidate = None
        if candidate is None:
            candidates = by_stem.get(c.stem.lower(), [])
            # Prefer the candidate whose parent path resembles the control path.
            parent_parts = set(part.lower() for part in c.relative_to(control_dir).parent.parts)
            ranked = sorted(
                (p for p in candidates if p not in used),
                key=lambda p: -len(parent_parts.intersection(
                    part.lower() for part in p.relative_to(technique_dir).parent.parts
                )),
            )
            candidate = ranked[0] if ranked else None
        if candidate is not None:
            pairs.append((c, candidate))
            used.add(candidate)
    if pairs:
        return pairs
    return list(zip(controls, techniques))


def discover_gtsinger_pairs(root):
    root = Path(root).expanduser().resolve()
    for control_dir in root.rglob("Control_Group"):
        parent = control_dir.parent
        siblings = [p for p in parent.iterdir() if p.is_dir() and p.name.lower().endswith("_group")]
        for technique_dir in siblings:
            if technique_dir.name.lower() in {"control_group", "paired_speech_group"}:
                continue
            technique = _normalize_technique(technique_dir.name)
            for control_wav, technique_wav in _best_pairs(control_dir, technique_dir):
                yield technique, control_wav, technique_wav


def train_gtsinger(root, output, max_pairs_per_technique=0):
    deltas = defaultdict(lambda: {
        "spectral": [], "ap": [], "gate": [], "rms_db": [],
        "breathiness_db": [], "voicing_db": [], "tension": [], "pairs": [],
    })
    seen = defaultdict(int)
    errors = []
    for technique, control_wav, technique_wav in discover_gtsinger_pairs(root):
        if max_pairs_per_technique and seen[technique] >= int(max_pairs_per_technique):
            continue
        try:
            c = _audio_signature(control_wav)
            t = _audio_signature(technique_wav)
        except Exception as exc:
            errors.append({"control": str(control_wav), "technique": str(technique_wav), "error": str(exc)})
            continue
        entry = deltas[technique]
        entry["spectral"].append(t["spectral_log"] - c["spectral_log"])
        entry["ap"].append(t["ap"] - c["ap"])
        entry["gate"].append(t["gate"] - c["gate"])
        entry["rms_db"].append(20.0 * np.log10((t["rms"] + 1e-8) / (c["rms"] + 1e-8)))
        entry["breathiness_db"].append(20.0 * np.log10((t["breathiness_rms"] + 1e-8) / (c["breathiness_rms"] + 1e-8)))
        entry["voicing_db"].append(20.0 * np.log10((t["voicing_rms"] + 1e-8) / (c["voicing_rms"] + 1e-8)))
        entry["tension"].append(float(t["tension"] - c["tension"]))
        entry["pairs"].append((str(control_wav), str(technique_wav)))
        seen[technique] += 1

    if not deltas:
        raise RuntimeError("No GTSinger Control_Group / Technique_Group WAV pairs were found.")

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    techniques = sorted(deltas)
    spectral = []
    ap = []
    gate = []
    rms_db = []
    breathiness_db = []
    voicing_db = []
    tension_delta = []
    counts = []
    metadata = {
        "format": 1,
        "source": "GTSinger paired Control_Group/Technique_Group",
        "target_sr": TARGET_SR,
        "spectral_bands": SPECTRAL_BANDS,
        "ap_bands": AP_BANDS,
        "techniques": {},
        "errors": errors[:100],
    }
    for technique in techniques:
        entry = deltas[technique]
        spectral.append(np.median(np.stack(entry["spectral"]), axis=0))
        ap.append(np.median(np.stack(entry["ap"]), axis=0))
        gate.append(float(np.median(entry["gate"])))
        rms_db.append(float(np.median(entry["rms_db"])))
        breathiness_db.append(float(np.median(entry["breathiness_db"])))
        voicing_db.append(float(np.median(entry["voicing_db"])))
        tension_delta.append(float(np.median(entry["tension"])))
        counts.append(len(entry["spectral"]))
        metadata["techniques"][technique] = {
            "pair_count": len(entry["spectral"]),
            "median_gate_delta": gate[-1],
            "median_rms_delta_db": rms_db[-1],
            "median_breathiness_delta_db": breathiness_db[-1],
            "median_voicing_delta_db": voicing_db[-1],
            "median_tension_delta": tension_delta[-1],
        }
    np.savez_compressed(
        output,
        techniques=np.asarray(techniques),
        spectral_log_gain=np.asarray(spectral, dtype=np.float32),
        ap_bias=np.asarray(ap, dtype=np.float32),
        gate_bias=np.asarray(gate, dtype=np.float32),
        rms_delta_db=np.asarray(rms_db, dtype=np.float32),
        breathiness_delta_db=np.asarray(breathiness_db, dtype=np.float32),
        voicing_delta_db=np.asarray(voicing_db, dtype=np.float32),
        tension_delta=np.asarray(tension_delta, dtype=np.float32),
        pair_count=np.asarray(counts, dtype=np.int32),
    )
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train a compact Yuaz singing-technique delta library.")
    sub = parser.add_subparsers(dest="command", required=True)
    gt = sub.add_parser("gtsinger")
    gt.add_argument("dataset_root")
    gt.add_argument("output")
    gt.add_argument("--max-pairs-per-technique", type=int, default=0)
    args = parser.parse_args()
    if args.command == "gtsinger":
        meta = train_gtsinger(args.dataset_root, args.output, args.max_pairs_per_technique)
        print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
