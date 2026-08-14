#!/usr/bin/env python3
import argparse
import hashlib
import io
import json
import math
import os
import random
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
import torch.nn.functional as F

from .highband_foundation import (
    HighBandFoundation,
    DEFAULT_CROSSOVER_HZ,
    DEFAULT_FULL_HZ,
    DEFAULT_MAX_HZ,
    highpass_residual_torch,
    save_highband_foundation,
)


AUDIT_FORMAT = 1
SHARD_FORMAT = 1
TARGET_SR = 48000
SEGMENT_SECONDS = 1.0


def _json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _mono(x):
    x = np.asarray(x)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    return np.nan_to_num(np.asarray(x, dtype=np.float32))


def _band_rms_db(audio, sr, lo, hi):
    audio = np.asarray(audio, dtype=np.float64)
    if audio.size < 1024 or sr * 0.5 <= lo:
        return -160.0
    n = min(audio.size, int(sr * 8.0))
    if audio.size > n:
        start = max(0, (audio.size - n) // 2)
        audio = audio[start:start+n]
    win = np.hanning(audio.size)
    spec = np.fft.rfft(audio * win)
    p = np.abs(spec) ** 2
    f = np.fft.rfftfreq(audio.size, 1.0 / float(sr))
    full = np.sqrt(np.mean(p[(f >= 80.0) & (f < min(sr*0.5, 22000.0))]) + 1e-20)
    m = (f >= lo) & (f < min(hi, sr*0.5))
    if not np.any(m) or full <= 1e-20:
        return -160.0
    band = np.sqrt(np.mean(p[m]) + 1e-20)
    return float(20.0 * np.log10(max(band / full, 1e-12)))


def _estimate_f0(audio, sr):
    audio = _mono(audio)
    if audio.size < int(0.25 * sr):
        return 0.0
    if sr != 12000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=12000)
        sr = 12000
    if audio.size > 6 * sr:
        start = max(0, (audio.size - 6 * sr) // 2)
        audio = audio[start:start + 6 * sr]
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12))
    if rms < 1e-4:
        return 0.0
    try:
        f0 = librosa.yin(audio, fmin=45.0, fmax=900.0, sr=sr, frame_length=1024, hop_length=256)
        f0 = f0[np.isfinite(f0)]
        if not f0.size:
            return 0.0
        return float(np.median(f0))
    except Exception:
        return 0.0


def _speaker_from_path(path, source):
    p = Path(path)
    parts = p.parts
    if source == "gtsinger":
        for part in parts:
            if any(tag in part for tag in ("Alto", "Soprano", "Tenor", "Bass")):
                return part
    stem = p.stem
    if source == "vocalset":
        import re
        m = re.search(r"(?:^|[_-])([fm]\d+)(?:[_-]|$)", stem.lower())
        if m:
            return m.group(1)
    if source == "phonation":
        return "ND357A"
    return p.parent.name or source


def _audit_audio(audio, sr):
    audio = _mono(audio)
    duration = audio.size / max(1.0, float(sr))
    metrics = {
        "duration_sec": float(duration),
        "sample_rate": int(sr),
        "nyquist_hz": float(sr * 0.5),
        "band_10_12_db": _band_rms_db(audio, sr, 10000.0, 12000.0),
        "band_12_16_db": _band_rms_db(audio, sr, 12000.0, 16000.0),
        "band_16_20_db": _band_rms_db(audio, sr, 16000.0, 20000.0),
        "band_20_22_db": _band_rms_db(audio, sr, 20000.0, 22000.0),
    }
    metrics["median_f0_hz"] = _estimate_f0(audio, sr)
    # Reject nominally high-rate audio whose upper band is effectively empty.
    metrics["accepted"] = bool(
        sr >= 40000 and duration >= 0.70 and
        metrics["band_12_16_db"] > -62.0 and metrics["band_16_20_db"] > -70.0
    )
    return metrics


def _iter_file_audio(root, source):
    root = Path(root).expanduser()
    if not root.is_dir():
        return
    exts = {".wav", ".flac", ".aif", ".aiff"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield {"source": source, "path": str(p.resolve()), "id": str(p.relative_to(root))}


def _extract_vocalset_records(root, cache_dir):
    root = Path(root).expanduser()
    if not root.is_dir():
        return
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("VocalSet audit needs pyarrow. Run setup-gender-training.command once, or install pyarrow from the configured mirror.") from exc
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(root.rglob("*.parquet"))
    count = 0
    for parquet_path in parquet_files:
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=32):
            rows = batch.to_pylist()
            for row in rows:
                audio_obj = None
                for key in ("audio", "file", "wav", "waveform"):
                    if key in row:
                        audio_obj = row[key]
                        break
                raw = None
                source_path = None
                if isinstance(audio_obj, dict):
                    raw = audio_obj.get("bytes")
                    source_path = audio_obj.get("path")
                elif isinstance(audio_obj, (bytes, bytearray, memoryview)):
                    raw = bytes(audio_obj)
                if raw is None and source_path:
                    sp = Path(source_path)
                    if not sp.is_absolute():
                        sp = parquet_path.parent / sp
                    if sp.is_file():
                        raw = sp.read_bytes()
                if raw is None:
                    continue
                try:
                    audio, sr = sf.read(io.BytesIO(raw), always_2d=False, dtype="float32")
                except Exception:
                    continue
                rid = None
                for key in ("id", "name", "filename", "file_name", "path"):
                    if row.get(key):
                        rid = str(row[key])
                        break
                if not rid:
                    rid = f"{parquet_path.stem}-{count:06d}"
                digest = hashlib.sha1((str(parquet_path)+"|"+rid).encode("utf-8", "replace")).hexdigest()[:16]
                out = cache_dir / f"{digest}.flac"
                if not out.exists():
                    sf.write(out, _mono(audio), int(sr), format="FLAC", subtype="PCM_16")
                count += 1
                yield {"source": "vocalset", "path": str(out.resolve()), "id": rid, "cache_owned": True}


def audit_datasets(gtsinger=None, vocalset=None, phonation=None, out=None, vocalset_cache=None):
    records = []
    sources = []
    if gtsinger and Path(gtsinger).expanduser().is_dir():
        sources.append(("gtsinger", _iter_file_audio(gtsinger, "gtsinger")))
    if phonation and Path(phonation).expanduser().is_dir():
        sources.append(("phonation", _iter_file_audio(phonation, "phonation")))
    if vocalset and Path(vocalset).expanduser().is_dir():
        sources.append(("vocalset", _extract_vocalset_records(vocalset, vocalset_cache)))
    idx = 0
    for source, iterator in sources:
        for rec in iterator or ():
            idx += 1
            p = Path(rec["path"])
            try:
                audio, sr = sf.read(p, always_2d=False, dtype="float32")
                metrics = _audit_audio(audio, int(sr))
            except Exception as exc:
                metrics = {"accepted": False, "error": str(exc)}
            item = dict(rec)
            item.update(metrics)
            item["speaker_id"] = _speaker_from_path(rec["path"], source)
            records.append(item)
            if source == "vocalset" and rec.get("cache_owned") and not item.get("accepted"):
                try:
                    Path(rec["path"]).unlink(missing_ok=True)
                except Exception:
                    pass
            if idx % 50 == 0:
                accepted = sum(bool(x.get("accepted")) for x in records)
                print(f"Audited {idx} files; accepted {accepted}")
    accepted = [x for x in records if x.get("accepted")]
    summary = {"total": len(records), "accepted": len(accepted), "sources": {}}
    for source in ("gtsinger", "vocalset", "phonation"):
        xs = [x for x in records if x.get("source") == source]
        ys = [x for x in xs if x.get("accepted")]
        if xs:
            summary["sources"][source] = {
                "total": len(xs), "accepted": len(ys),
                "accepted_hours": float(sum(x.get("duration_sec", 0.0) for x in ys) / 3600.0),
                "low_f0_under_160": int(sum(0 < float(x.get("median_f0_hz", 0.0)) < 160.0 for x in ys)),
                "median_16_20_db": float(np.median([x.get("band_16_20_db", -160.0) for x in ys])) if ys else -160.0,
            }
    payload = {"format": AUDIT_FORMAT, "created_at": time.time(), "target_sr": TARGET_SR, "summary": summary, "records": records}
    if out:
        _json_write(out, payload)
    return payload


def _resample_48k(audio, sr):
    audio = _mono(audio)
    if int(sr) == TARGET_SR:
        return audio
    g = math.gcd(int(sr), TARGET_SR)
    return resample_poly(audio, TARGET_SR // g, int(sr) // g).astype(np.float32)


def _degrade_24k_to_48k(target):
    low = resample_poly(np.asarray(target, dtype=np.float32), 1, 2)
    up = resample_poly(low, 2, 1).astype(np.float32)
    if up.size < target.size:
        up = np.pad(up, (0, target.size - up.size))
    return up[:target.size]


def _segment_energy(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-12))


def prepare_shards(audit_path, out_dir, segments=6000, val_segments=800, seed=1337, segment_seconds=SEGMENT_SECONDS, shard_size=32):
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    records = [r for r in audit.get("records", []) if r.get("accepted") and Path(r.get("path", "")).is_file()]
    if not records:
        raise RuntimeError("No accepted full-band audio records. Run the bandwidth audit first.")
    rng = random.Random(int(seed))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.npz"):
        old.unlink()
    samples = int(round(TARGET_SR * float(segment_seconds)))
    speakers = sorted({str(r.get("speaker_id") or "unknown") for r in records})
    val_speakers = {s for s in speakers if int(hashlib.sha1(s.encode()).hexdigest()[:8], 16) % 5 == 0}
    if not val_speakers and len(speakers) > 1:
        val_speakers = {speakers[-1]}
    train_records = [r for r in records if str(r.get("speaker_id") or "unknown") not in val_speakers]
    val_records = [r for r in records if str(r.get("speaker_id") or "unknown") in val_speakers]
    if not train_records:
        train_records = records
    if not val_records:
        val_records = records[:max(1, min(20, len(records)))]

    cache = {}
    def choose_record(pool):
        weights = []
        for r in pool:
            f0 = float(r.get("median_f0_hz", 0.0) or 0.0)
            w = 2.5 if 0 < f0 < 160.0 else (1.4 if 0 < f0 < 220.0 else 1.0)
            if r.get("source") == "gtsinger": w *= 1.20
            if r.get("source") == "phonation": w *= 0.75
            weights.append(w)
        return rng.choices(pool, weights=weights, k=1)[0]

    def load_record(r):
        p = r["path"]
        if p not in cache:
            audio, sr = sf.read(p, always_2d=False, dtype="float32")
            cache[p] = _resample_48k(audio, int(sr))
            if len(cache) > 12:
                cache.pop(next(iter(cache)))
        return cache[p]

    manifest = {"format": SHARD_FORMAT, "target_sr": TARGET_SR, "segment_samples": samples, "val_speakers": sorted(val_speakers), "shards": []}
    for split, pool, total in (("train", train_records, int(segments)), ("val", val_records, int(val_segments))):
        shard_index = 0
        made = 0
        while made < total:
            targets=[]; inputs=[]; f0s=[]; sources=[]; speakers_out=[]
            attempts = 0
            while len(targets) < min(int(shard_size), total-made) and attempts < int(shard_size)*30:
                attempts += 1
                r = choose_record(pool)
                audio = load_record(r)
                if audio.size < samples:
                    continue
                max_start = audio.size - samples
                start = rng.randint(0, max_start) if max_start > 0 else 0
                seg = np.asarray(audio[start:start+samples], dtype=np.float32)
                if _segment_energy(seg) < 0.004:
                    continue
                degraded = _degrade_24k_to_48k(seg)
                # Keep only examples with a measurable missing upper band.
                missing = seg - degraded
                if _segment_energy(missing) < 2e-5:
                    continue
                f0 = _estimate_f0(seg, TARGET_SR)
                targets.append(seg.astype(np.float16))
                inputs.append(degraded.astype(np.float16))
                f0s.append(np.float16(f0))
                sources.append(str(r.get("source", "")))
                speakers_out.append(str(r.get("speaker_id", "")))
            if not targets:
                raise RuntimeError(f"Could not prepare usable {split} segments from accepted records.")
            path = out_dir / f"{split}-{shard_index:04d}.npz"
            np.savez_compressed(
                path,
                target=np.stack(targets), input=np.stack(inputs), f0=np.asarray(f0s, dtype=np.float16),
                source=np.asarray(sources), speaker=np.asarray(speakers_out),
            )
            manifest["shards"].append({"split":split,"path":str(path.resolve()),"count":len(targets)})
            made += len(targets)
            shard_index += 1
            print(f"Prepared {split}: {made}/{total}")
    _json_write(out_dir / "manifest.json", manifest)
    return manifest


def _stft_pair(pred, target, sr, n_fft, hop):
    win = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
    ps = torch.stft(pred[:, 0], n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True)
    ts = torch.stft(target[:, 0], n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / float(sr), device=pred.device)
    return ps, ts, freqs


def _stft_highband_loss(pred, target, sr=TARGET_SR, n_fft=2048, hop=512):
    ps, ts, freqs = _stft_pair(pred, target, sr, n_fft, hop)
    mask = (freqs >= DEFAULT_CROSSOVER_HZ) & (freqs <= DEFAULT_MAX_HZ)
    if not torch.any(mask):
        return pred.new_tensor(0.0)
    pm = torch.log1p(4.0 * torch.abs(ps[:, mask]))
    tm = torch.log1p(4.0 * torch.abs(ts[:, mask]))
    return torch.mean(torch.abs(pm - tm))


def _multires_highband_loss(pred, target, sr=TARGET_SR):
    losses = []
    for n_fft, hop in ((1024, 256), (2048, 384), (4096, 768)):
        losses.append(_stft_highband_loss(pred, target, sr=sr, n_fft=n_fft, hop=hop))
    return torch.mean(torch.stack(losses))


def _band_envelope_loss(pred, target, sr=TARGET_SR, n_fft=2048, hop=256):
    ps, ts, freqs = _stft_pair(pred, target, sr, n_fft, hop)
    pp = torch.abs(ps).pow(2)
    tp = torch.abs(ts).pow(2)
    bands = (
        (9500.0, 12000.0),
        (12000.0, 15000.0),
        (15000.0, 18000.0),
        (18000.0, 22000.0),
    )
    losses = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        if not torch.any(mask):
            continue
        p = torch.mean(pp[:, mask], dim=1)
        t = torch.mean(tp[:, mask], dim=1)
        p_db = 0.5 * torch.log(p + 1e-9)
        t_db = 0.5 * torch.log(t + 1e-9)
        losses.append(torch.mean(torch.abs(p_db - t_db)))
    if not losses:
        return pred.new_tensor(0.0)
    return torch.mean(torch.stack(losses))


def _training_objective(pred, target):
    wave = F.smooth_l1_loss(pred, target, beta=0.005)
    mag = _multires_highband_loss(pred, target)
    envelope = _band_envelope_loss(pred, target)
    return 0.10 * wave + 0.56 * mag + 0.34 * envelope, {
        "wave": wave,
        "magnitude": mag,
        "envelope": envelope,
    }


def _iter_shard_batches(manifest, split, batch_size, rng):
    shards = [x for x in manifest.get("shards", []) if x.get("split") == split]
    rng.shuffle(shards)
    for s in shards:
        with np.load(s["path"], allow_pickle=False) as data:
            inp = np.asarray(data["input"], dtype=np.float32)
            target = np.asarray(data["target"], dtype=np.float32)
            f0 = np.asarray(data["f0"], dtype=np.float32)
        order = list(range(len(inp))); rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            idx = order[i:i+batch_size]
            yield inp[idx], target[idx], f0[idx]


def train_foundation(shard_manifest, out_path, epochs=10, batch_size=4, lr=1.5e-4, device="auto", seed=1337):
    manifest = json.loads(Path(shard_manifest).read_text(encoding="utf-8"))
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    dev = torch.device(device)
    torch.manual_seed(int(seed)); np.random.seed(int(seed)); random.seed(int(seed))
    model = HighBandFoundation(hidden=40, dilations=(1, 2, 4, 8, 16, 32, 64, 128)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    rng = random.Random(int(seed))
    best = None; best_state = None
    history=[]
    for epoch in range(1, int(epochs)+1):
        model.train(); train_losses=[]; train_parts={"wave":[],"magnitude":[],"envelope":[]}
        for inp_np, target_np, f0_np in _iter_shard_batches(manifest, "train", int(batch_size), rng):
            inp = torch.from_numpy(inp_np).to(dev).unsqueeze(1)
            target = torch.from_numpy(target_np).to(dev).unsqueeze(1)
            f0 = torch.from_numpy(f0_np).to(dev).view(-1,1,1)
            raw = model(inp, f0=f0)
            pred_hb = highpass_residual_torch(raw, TARGET_SR, DEFAULT_CROSSOVER_HZ, DEFAULT_FULL_HZ, DEFAULT_MAX_HZ)
            true_hb = highpass_residual_torch(target - inp, TARGET_SR, DEFAULT_CROSSOVER_HZ, DEFAULT_FULL_HZ, DEFAULT_MAX_HZ)
            objective, parts = _training_objective(pred_hb, true_hb)
            low_w = 1.0 + 0.55 * ((f0[:,0,0] > 1.0) & (f0[:,0,0] < 160.0)).float().mean()
            loss = low_w * objective
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step()
            train_losses.append(float(loss.detach().cpu()))
            for key, value in parts.items():
                train_parts[key].append(float(value.detach().cpu()))
        model.eval(); vals=[]; low_vals=[]
        with torch.inference_mode():
            for inp_np, target_np, f0_np in _iter_shard_batches(manifest, "val", int(batch_size), rng):
                inp = torch.from_numpy(inp_np).to(dev).unsqueeze(1)
                target = torch.from_numpy(target_np).to(dev).unsqueeze(1)
                f0 = torch.from_numpy(f0_np).to(dev).view(-1,1,1)
                pred = highpass_residual_torch(model(inp, f0=f0), TARGET_SR, DEFAULT_CROSSOVER_HZ, DEFAULT_FULL_HZ, DEFAULT_MAX_HZ)
                true = highpass_residual_torch(target - inp, TARGET_SR, DEFAULT_CROSSOVER_HZ, DEFAULT_FULL_HZ, DEFAULT_MAX_HZ)
                val_obj, _ = _training_objective(pred, true)
                vals.append(float(val_obj.cpu()))
                mask = (f0[:,0,0] > 1.0) & (f0[:,0,0] < 160.0)
                if torch.any(mask):
                    low_obj, _ = _training_objective(pred[mask], true[mask])
                    low_vals.append(float(low_obj.cpu()))
        tr=float(np.mean(train_losses)) if train_losses else float("inf")
        va=float(np.mean(vals)) if vals else float("inf")
        lv=float(np.mean(low_vals)) if low_vals else va
        score=va + 0.35*lv
        row={
            "epoch":epoch,"train":tr,"val":va,"low_f0_val":lv,"score":score,
            "train_wave":float(np.mean(train_parts["wave"])) if train_parts["wave"] else None,
            "train_magnitude":float(np.mean(train_parts["magnitude"])) if train_parts["magnitude"] else None,
            "train_envelope":float(np.mean(train_parts["envelope"])) if train_parts["envelope"] else None,
        }
        history.append(row)
        print(f"Epoch {epoch:02d}: train={tr:.6f} val={va:.6f} lowF0={lv:.6f} mag={row['train_magnitude']:.6f} env={row['train_envelope']:.6f}")
        if best is None or score < best:
            best=score
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    meta={
        "foundation_revision":2,
        "runtime_backend":"highband-foundation-v2",
        "training_source":"paired full-band singing -> 24k degraded input",
        "training_objective":"phase-tolerant multi-resolution magnitude + framewise band-envelope + light waveform residual",
        "target_is_runtime_masked_residual":True,
        "target_sample_rate":TARGET_SR,
        "crossover_hz":DEFAULT_CROSSOVER_HZ,
        "full_hz":DEFAULT_FULL_HZ,
        "max_hz":DEFAULT_MAX_HZ,
        "low_f0_oversampling":True,
        "low_f0_validation_under_hz":160.0,
        "history":history,
        "license_note":"Derived weight rights follow the source datasets used to build the shards. Keep WEIGHTS/provenance metadata with distributed checkpoints.",
        "created_at":time.time(),
    }
    save_highband_foundation(out_path, model.cpu(), meta)
    return meta

def main(argv=None):
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("audit")
    a.add_argument("--gtsinger"); a.add_argument("--vocalset"); a.add_argument("--phonation"); a.add_argument("--out",required=True); a.add_argument("--vocalset-cache",required=True)
    s=sub.add_parser("prepare")
    s.add_argument("--audit",required=True); s.add_argument("--out-dir",required=True); s.add_argument("--segments",type=int,default=6000); s.add_argument("--val-segments",type=int,default=800); s.add_argument("--seed",type=int,default=1337)
    t=sub.add_parser("train")
    t.add_argument("--manifest",required=True); t.add_argument("--out",required=True); t.add_argument("--epochs",type=int,default=10); t.add_argument("--batch-size",type=int,default=4); t.add_argument("--lr",type=float,default=1.5e-4); t.add_argument("--device",default="auto")
    args=p.parse_args(argv)
    if args.cmd=="audit":
        data=audit_datasets(args.gtsinger,args.vocalset,args.phonation,args.out,args.vocalset_cache)
        print(json.dumps(data["summary"],ensure_ascii=False,indent=2))
    elif args.cmd=="prepare":
        prepare_shards(args.audit,args.out_dir,args.segments,args.val_segments,args.seed)
        print("Prepared:",Path(args.out_dir)/"manifest.json")
    else:
        train_foundation(args.manifest,args.out,args.epochs,args.batch_size,args.lr,args.device)
        print("Saved:",args.out)


if __name__ == "__main__":
    main()
