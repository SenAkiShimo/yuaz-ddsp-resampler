#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .ai_control_training import NativeYuazDDSPExtractor, _resize_time
from .ai_vocal_controls import AIControlAdapter, save_ai_control_adapter

CONTROL_NAME = "gender_formant"
STRAIGHT_LABEL = 12
VOWELS = {"a", "e", "i", "o", "u"}


def _audio_cell(value):
    if isinstance(value, dict):
        return value.get("bytes"), str(value.get("path") or "")
    try:
        d = value.as_py()
        if isinstance(d, dict):
            return d.get("bytes"), str(d.get("path") or "")
    except Exception:
        pass
    return None, ""


def _identity_from_path(path):
    name = Path(str(path)).name.lower()
    m = re.match(r"([fm]\d+)[ _-]", name)
    if not m:
        raise ValueError(f"cannot recover VocalSet singer id from filename: {name!r}")
    singer = m.group(1)
    sex = "female" if singer.startswith("f") else "male"
    stem = Path(name).stem
    tokens = re.split(r"[_ -]+", stem)
    vowel = next((x for x in reversed(tokens) if x in VOWELS), None)
    if vowel is None:
        raise ValueError(f"cannot recover VocalSet vowel from filename: {name!r}")
    return singer, sex, vowel


def _midi(f0):
    f0 = np.asarray(f0, dtype=np.float32).reshape(-1)
    out = np.full(f0.shape, np.nan, dtype=np.float32)
    mask = f0 > 20.0
    out[mask] = 69.0 + 12.0 * np.log2(f0[mask] / 440.0)
    return out


def _pitch_bin(midi, width=3.0):
    return int(round(float(midi) / width) * width)


def _shape_logspec(log_spec):
    x = np.asarray(log_spec, dtype=np.float32)
    return x - np.mean(x, axis=0, keepdims=True)


def _read_parquet_rows(dataset_root):
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required. Run setup-gender-training.command first.") from exc
    files = sorted(Path(dataset_root).expanduser().resolve().glob("data/*.parquet"))
    if not files:
        raise RuntimeError(f"No VocalSet parquet shards found under: {dataset_root}/data")
    for file in files:
        pf = pq.ParquetFile(file)
        cols = set(pf.schema_arrow.names)
        if "audio" not in cols or "label" not in cols:
            raise RuntimeError(f"Unexpected VocalSet mirror schema in {file.name}: {sorted(cols)}")
        for batch in pf.iter_batches(columns=["audio", "label"], batch_size=16):
            audio_col = batch.column(0).to_pylist()
            label_col = batch.column(1).to_pylist()
            for audio, label in zip(audio_col, label_col):
                yield audio, int(label)


def prepare(dataset_root, out_dir, project_root, max_items=0):
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = out_dir / "features"
    shard_dir = out_dir / "shards"
    feature_dir.mkdir(exist_ok=True); shard_dir.mkdir(exist_ok=True)
    native = NativeYuazDDSPExtractor(project_root)
    records = []
    errors = []
    processed = 0
    for audio_cell, label in _read_parquet_rows(dataset_root):
        if label != STRAIGHT_LABEL:
            continue
        raw, path = _audio_cell(audio_cell)
        try:
            if not path:
                raise ValueError("VocalSet mirror row has no original audio.path; refusing to infer gender from acoustics")
            singer, sex, vowel = _identity_from_path(path)
            key = hashlib.sha1(path.encode("utf-8")).hexdigest()[:18]
            cache = feature_dir / f"{singer}-{vowel}-{key}.npz"
            if cache.is_file():
                z = np.load(cache, allow_pickle=False)
                rec = {"cache": str(cache), "path": path, "singer": singer, "sex": sex, "vowel": vowel,
                       "frames": int(z["log_spec"].shape[-1])}
                records.append(rec); processed += 1
                continue
            if raw is None:
                # Some parquet exporters keep a path but omit embedded bytes. We deliberately
                # do not fall back to an overseas URL here.
                raise ValueError("audio bytes are absent from the China-mirror parquet row")
            suffix = Path(path).suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(raw); temp = Path(f.name)
            try:
                feat = native.features(temp)
            finally:
                temp.unlink(missing_ok=True)
            frames = min(int(feat["log_spec"].shape[-1]), 2400)
            if frames < 20:
                raise ValueError("too few Yuaz DDSP frames")
            log_spec = _resize_time(feat["log_spec"], frames).astype(np.float32)
            ap = _resize_time(feat["ap"], frames).astype(np.float32)
            gate = _resize_time(feat["gate"], frames).astype(np.float32)
            f0 = _resize_time(feat["f0"], frames).astype(np.float32)
            np.savez_compressed(cache, log_spec=log_spec, ap=ap, gate=gate, f0=f0,
                                singer=np.asarray(singer), sex=np.asarray(sex), vowel=np.asarray(vowel), path=np.asarray(path))
            records.append({"cache": str(cache), "path": path, "singer": singer, "sex": sex, "vowel": vowel, "frames": frames})
            processed += 1
            if processed % 20 == 0:
                print(f"Extracted {processed} straight VocalSet recordings into frozen Yuaz DDSP features", flush=True)
            if max_items and processed >= int(max_items):
                break
        except Exception as exc:
            errors.append({"path": path, "error": str(exc)})
    if not records:
        preview = errors[:5]
        raise RuntimeError(f"No usable straight VocalSet rows were extracted. First errors: {preview}")

    singers = sorted({r["singer"] for r in records})
    female = [s for s in singers if s.startswith("f")]
    male = [s for s in singers if s.startswith("m")]
    if len(female) < 2 or len(male) < 2:
        raise RuntimeError(f"Need at least two singers per sex for speaker-disjoint validation; got female={female}, male={male}")
    # Deterministic held-out speakers. No clip from these singers enters training.
    val_singers = {female[-1], male[-1]}

    # singer -> (vowel,pitchbin) -> list spectral frame shapes
    singer_bins = defaultdict(lambda: defaultdict(list))
    record_data = []
    for r in records:
        z = np.load(r["cache"], allow_pickle=False)
        shape = _shape_logspec(z["log_spec"])
        f0 = np.asarray(z["f0"], dtype=np.float32).reshape(-1)
        midi = _midi(f0)
        bins = np.full(midi.shape, -999, dtype=np.int32)
        for i, m in enumerate(midi):
            if np.isfinite(m): bins[i] = _pitch_bin(m)
        for pb in sorted(set(int(x) for x in bins if x > -900)):
            idx = np.where(bins == pb)[0]
            if len(idx) >= 4:
                singer_bins[r["singer"]][(r["vowel"], pb)].append(np.mean(shape[:, idx], axis=1))
        record_data.append((r, z["log_spec"].astype(np.float32), z["ap"].astype(np.float32), z["gate"].astype(np.float32), z["f0"].astype(np.float32), bins))

    # First average inside each singer; then average singers. This avoids singers with more clips dominating.
    group = defaultdict(lambda: {"female": [], "male": []})
    singer_means = {}
    for singer, bmap in singer_bins.items():
        sex = "female" if singer.startswith("f") else "male"
        for key, vals in bmap.items():
            mean = np.mean(np.stack(vals), axis=0).astype(np.float32)
            singer_means[(singer, key)] = mean
            if singer not in val_singers:
                group[key][sex].append(mean)
    deltas = {}
    for key, sexes in group.items():
        if len(sexes["female"]) >= 2 and len(sexes["male"]) >= 2:
            f = np.mean(np.stack(sexes["female"]), axis=0)
            m = np.mean(np.stack(sexes["male"]), axis=0)
            deltas[key] = (m - f).astype(np.float32)
    # Vowel-level fallback when a pitch bin lacks enough cross-sex coverage.
    vowel_group = {}
    for vowel in VOWELS:
        vals = [v for (vow, _), v in deltas.items() if vow == vowel]
        if vals: vowel_group[vowel] = np.mean(np.stack(vals), axis=0).astype(np.float32)
    if not deltas and not vowel_group:
        raise RuntimeError("Could not build any cross-sex VocalSet spectral centroids.")

    made = 0
    per_singer = defaultdict(int)
    for r, log_spec, ap, gate, f0, bins in record_data:
        frames = log_spec.shape[-1]
        target = np.zeros((log_spec.shape[0], frames), dtype=np.float32)
        active = np.zeros((1, frames), dtype=np.float32)
        sign = 1.0 if r["sex"] == "female" else -1.0
        for i, pb in enumerate(bins):
            if pb <= -900: continue
            delta = deltas.get((r["vowel"], int(pb)), vowel_group.get(r["vowel"]))
            if delta is None: continue
            # Adapter applies exp(0.72*tanh(ds)); supervise in that residual coordinate.
            target[:, i] = np.clip(sign * delta / 0.72, -1.25, 1.25)
            active[0, i] = sign
        if np.count_nonzero(active) < 12:
            continue
        name = f"{made:05d}-{r['singer']}-{r['vowel']}-{hashlib.sha1(r['path'].encode()).hexdigest()[:10]}.npz"
        np.savez_compressed(shard_dir / name,
            spectral=np.exp(log_spec).astype(np.float32), ap=ap, gate=gate, f0=f0,
            controls=active, target_ds=target,
            target_da=np.zeros((16, frames), dtype=np.float32), target_dg=np.zeros((1, frames), dtype=np.float32),
            singer=np.asarray(r["singer"]), sex=np.asarray(r["sex"]), vowel=np.asarray(r["vowel"]), path=np.asarray(r["path"]),
            is_validation=np.asarray(r["singer"] in val_singers),
        )
        made += 1; per_singer[r["singer"]] += 1
    if made < 8:
        raise RuntimeError(f"Too few gender training shards after centroid matching: {made}")
    manifest = {
        "format": 1,
        "source": "VocalSet mirror of Zenodo 1442513",
        "transport_mirror": "Bill13579/vocalset-mirror via hf-mirror.com",
        "license": "CC BY 4.0",
        "supervision": "speaker-aggregate opposite-sex Yuaz spectral-envelope centroid residual",
        "control": CONTROL_NAME,
        "positive_direction": "masculine/lower-formant statistical direction",
        "output_scopes": ["spectral"],
        "checkpoint_sha256": native.checkpoint_sha256,
        "feature_backend": "yuaz-native-ddsp-v1",
        "sample_rate": native.sample_rate,
        "hop": native.hop,
        "straight_label": STRAIGHT_LABEL,
        "singers": singers,
        "validation_singers": sorted(val_singers),
        "shard_count": made,
        "shards_per_singer": dict(sorted(per_singer.items())),
        "errors": errors,
        "created_at": time.time(),
    }
    (out_dir / "dataset.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Prepared VocalSet gender dataset: {shard_dir} ({made} shards)")
    print("Speaker-disjoint validation singers:", ", ".join(sorted(val_singers)))
    return shard_dir


def _tensor(a):
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).unsqueeze(0)


def _windows(path, window=160, max_windows=5):
    z = np.load(path, allow_pickle=False)
    T = int(z["spectral"].shape[-1]); w = min(int(window), T)
    starts = [0] if T <= w else np.linspace(0, T-w, min(max_windows, max(1, T//w))).astype(int).tolist()
    for st in starts:
        en=st+w
        yield {k:_tensor(z[k][:,st:en]) for k in ("spectral","ap","gate","f0","target_ds","target_da","target_dg")}, torch.from_numpy(np.asarray(z["controls"][:,st:en],dtype=np.float32)).view(1,1,-1)


def train(dataset_dir, output, epochs=12, lr=2e-4, window=160, max_windows=5, seed=2801):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dataset_dir=Path(dataset_dir).expanduser().resolve()
    manifest_path=dataset_dir.parent / "dataset.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    shards=sorted(dataset_dir.glob("*.npz"))
    train_shards=[]; val_shards=[]
    for p in shards:
        z=np.load(p,allow_pickle=False)
        (val_shards if bool(z["is_validation"].item()) else train_shards).append(p)
    if not train_shards or not val_shards:
        raise RuntimeError(f"Speaker-disjoint split invalid: train={len(train_shards)} val={len(val_shards)}")
    model=AIControlAdapter(control_names=(CONTROL_NAME,), control_modes=("signed",), output_scopes=("spectral",))
    opt=torch.optim.AdamW(model.parameters(),lr=float(lr),weight_decay=1e-4)
    checkpoint=Path(str(output)+".training.pt"); start=0; best=float("inf")
    if checkpoint.is_file():
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
        model.load_state_dict(payload["model"]); opt.load_state_dict(payload["optimizer"])
        start=int(payload.get("epoch",-1))+1; best=float(payload.get("best",best))
        print(f"Resuming gender training at epoch {start+1}")
    def run(files, training):
        model.train(training); total=0.0; n=0; order=list(files)
        if training: random.shuffle(order)
        for shard in order:
            for b,c in _windows(shard,window,max_windows if training else 2):
                controls={CONTROL_NAME:c}
                ds,da,dg=model.predict_residuals(b["spectral"],b["ap"],b["gate"],b["f0"],controls)
                voiced=(b["f0"]>1.0).to(ds.dtype); active=(c.abs()>0.5).to(ds.dtype); mask=(voiced*active).expand_as(ds)
                raw=F.smooth_l1_loss(torch.tanh(ds),torch.clamp(b["target_ds"],-1,1),reduction="none")
                loss_s=torch.sum(raw*mask)/torch.clamp(mask.sum(),min=1.0)
                # Hard guard: a Gender pack must never learn AP/gate residuals.
                scope_loss=da.abs().mean()+dg.abs().mean()
                zeros={CONTROL_NAME:torch.zeros_like(c)}
                zds,zda,zdg=model.predict_residuals(b["spectral"],b["ap"],b["gate"],b["f0"],zeros)
                zero_loss=zds.abs().mean()+zda.abs().mean()+zdg.abs().mean()
                loss=loss_s+0.25*scope_loss+0.15*zero_loss
                if training:
                    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step()
                total+=float(loss.detach()); n+=1
        return total/max(1,n)
    history=[]
    for epoch in range(start,int(epochs)):
        tr=run(train_shards,True)
        with torch.inference_mode(): va=run(val_shards,False)
        history.append({"epoch":epoch+1,"train_loss":tr,"val_loss":va})
        print(f"Gender Epoch {epoch+1}/{epochs}: train={tr:.6f} val={va:.6f}",flush=True)
        torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"epoch":epoch,"best":min(best,va)},checkpoint)
        if va<=best:
            best=va
            save_ai_control_adapter(output,model,metadata={
                "training_source":"VocalSet: A Singing Voice Dataset",
                "canonical_source":"Zenodo 1442513",
                "download_transport":"Bill13579/vocalset-mirror repository; transport selected at setup time",
                "license":"CC BY 4.0",
                "feature_backend":manifest["feature_backend"],
                "checkpoint_sha256":manifest["checkpoint_sha256"],
                "sample_rate":manifest["sample_rate"],"hop":manifest["hop"],
                "controls":[CONTROL_NAME],"control_modes":["signed"],"output_scopes":["spectral"],
                "positive_direction":manifest["positive_direction"],
                "target_design":"opposite-sex aggregate centroid residual; no target singer embedding; straight singing only",
                "speaker_disjoint_validation":True,"validation_singers":manifest["validation_singers"],
                "train_shards":len(train_shards),"validation_shards":len(val_shards),
                "best_validation_loss":best,"epoch":epoch+1,"created_at":time.time(),
            })
    Path(str(output)+".json").write_text(json.dumps({"format":1,"best_validation_loss":best,"history":history},indent=2)+"\n",encoding="utf-8")
    print(f"Gender foundation trained: {Path(output).resolve()}")
    return output


def main():
    ap=argparse.ArgumentParser(description="Train Yuaz learned Gender/Formant DDSP control from VocalSet.")
    sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("prepare"); p.add_argument("dataset_root"); p.add_argument("output_dir"); p.add_argument("--project-root",required=True); p.add_argument("--max-items",type=int,default=0)
    t=sub.add_parser("train"); t.add_argument("dataset_dir"); t.add_argument("output"); t.add_argument("--epochs",type=int,default=12); t.add_argument("--lr",type=float,default=2e-4); t.add_argument("--window",type=int,default=160); t.add_argument("--max-windows",type=int,default=5)
    a=ap.parse_args()
    if a.cmd=="prepare": prepare(a.dataset_root,a.output_dir,a.project_root,a.max_items)
    else: train(a.dataset_dir,a.output,a.epochs,a.lr,a.window,a.max_windows)
if __name__=="__main__": main()
