#!/usr/bin/env python3
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .ai_vocal_controls import AIControlAdapter, save_ai_control_adapter

CONTROL_NAMES = ("tension", "voicing")
CONTROL_MODES = ("signed", "signed")
OUTPUT_SCOPES = ("spectral", "ap", "gate")


def _tensor(a):
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).unsqueeze(0)


def _windows(path, window=160, max_windows=5):
    z = np.load(path, allow_pickle=False)
    T = int(z["spectral"].shape[-1])
    w = min(int(window), T)
    starts = [0] if T <= w else np.linspace(0, T - w, min(max_windows, max(1, T // w))).astype(int).tolist()
    source = str(z["source"].item()) if "source" in z else ""
    for st in starts:
        en = st + w
        batch = {
            k: _tensor(z[k][:, st:en])
            for k in ("spectral", "ap", "gate", "f0", "target_ds", "target_da", "target_dg", "mask_s", "mask_a", "mask_g")
        }
        c = np.asarray(z["controls"][:, st:en], dtype=np.float32)
        controls = {name: torch.from_numpy(c[i]).view(1, 1, -1) for i, name in enumerate(CONTROL_NAMES)}
        yield batch, controls, source


def _masked_huber(pred, target, mask):
    expanded = mask.expand_as(pred)
    raw = F.smooth_l1_loss(torch.tanh(pred), torch.clamp(target, -1.0, 1.0), reduction="none")
    return torch.sum(raw * expanded) / torch.clamp(expanded.sum(), min=1.0)


def _masked_mean_abs(x, mask):
    expanded = mask.expand_as(x)
    return torch.sum(torch.abs(torch.tanh(x)) * expanded) / torch.clamp(expanded.sum(), min=1.0)


def _voicing_only_mask(controls, dtype):
    v = torch.abs(controls["voicing"])
    t = torch.abs(controls["tension"])
    return ((v > 1e-4) & (t <= 1e-4)).to(dtype)


def _odd_ap_loss(model, batch, controls, mask):
    magnitude = torch.abs(controls["voicing"])
    positive = {
        "tension": torch.zeros_like(controls["tension"]),
        "voicing": magnitude,
    }
    negative = {
        "tension": torch.zeros_like(controls["tension"]),
        "voicing": -magnitude,
    }
    _, ap_pos, _ = model.predict_residuals(batch["spectral"], batch["ap"], batch["gate"], batch["f0"], positive)
    _, ap_neg, _ = model.predict_residuals(batch["spectral"], batch["ap"], batch["gate"], batch["f0"], negative)
    residual = torch.tanh(ap_pos) + torch.tanh(ap_neg)
    expanded = mask.expand_as(residual)
    return torch.sum(torch.abs(residual) * expanded) / torch.clamp(expanded.sum(), min=1.0)


def _signed_probe(model, batch):
    ones = torch.ones((1, 1, batch["spectral"].shape[-1]), dtype=batch["spectral"].dtype)
    zeros = torch.zeros_like(ones)
    neg = {"tension": zeros, "voicing": -ones}
    pos = {"tension": zeros, "voicing": ones}
    with torch.inference_mode():
        n = model.predict_residuals(batch["spectral"], batch["ap"], batch["gate"], batch["f0"], neg)
        p = model.predict_residuals(batch["spectral"], batch["ap"], batch["gate"], batch["f0"], pos)
    rows = {}
    for name, a, b in zip(("spectral", "ap", "gate"), n, p):
        af = a.reshape(-1).double(); bf = b.reshape(-1).double()
        na = float(torch.linalg.vector_norm(af)); nb = float(torch.linalg.vector_norm(bf))
        cosine = float(torch.dot(af, bf) / (na * nb)) if na * nb > 1e-12 else 0.0
        rms_a = float(torch.sqrt(torch.mean(af * af) + 1e-12))
        rms_b = float(torch.sqrt(torch.mean(bf * bf) + 1e-12))
        mean = max(1e-8, 0.5 * (rms_a + rms_b))
        rows[name] = {
            "rms_negative": rms_a,
            "rms_positive": rms_b,
            "cosine": cosine,
            "difference_ratio": float(torch.sqrt(torch.mean((af - bf) ** 2) + 1e-12)) / mean,
            "sum_ratio": float(torch.sqrt(torch.mean((af + bf) ** 2) + 1e-12)) / mean,
        }
    return rows


def train(direct_dir, mocha_dir, output, checkpoint_sha, epochs=12, lr=2e-4, seed=619):
    direct = sorted(Path(direct_dir).expanduser().resolve().glob("*.npz"))
    mocha = sorted(Path(mocha_dir).expanduser().resolve().glob("*.npz"))
    files = [(p, "direct") for p in direct] + [(p, "mocha") for p in mocha]
    if len(direct) < 4 or len(mocha) < 4:
        raise RuntimeError(f"Too few shards: direct={len(direct)} mocha={len(mocha)}")
    random.Random(seed).shuffle(files)
    nval = max(1, int(round(len(files) * 0.12)))
    val = files[:nval]
    train_files = files[nval:]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = AIControlAdapter(
        control_names=CONTROL_NAMES,
        control_modes=CONTROL_MODES,
        output_scopes=OUTPUT_SCOPES,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    best = float("inf")
    history = []
    probe_batch = None

    def run(items, training):
        nonlocal probe_batch
        model.train(training)
        total = 0.0
        count = 0
        order = list(items)
        if training:
            random.shuffle(order)
        for path, _kind in order:
            for batch, controls, source in _windows(path, max_windows=5 if training else 2):
                ds, da, dg = model.predict_residuals(
                    batch["spectral"], batch["ap"], batch["gate"], batch["f0"], controls
                )
                voicing_only = _voicing_only_mask(controls, batch["spectral"].dtype)
                is_mocha_voicing = bool("MOCHA" in source.upper()) and float(voicing_only.max()) > 0.5

                mask_s = batch["mask_s"]
                mask_a = batch["mask_a"]
                mask_g = batch["mask_g"]
                loss_s = _masked_huber(ds, batch["target_ds"], mask_s)
                loss_a = _masked_huber(da, batch["target_da"], mask_a)
                loss_g = _masked_huber(dg, batch["target_dg"], mask_g)
                loss = loss_s + loss_a + loss_g

                if is_mocha_voicing:
                    voiced = (batch["f0"] > 1.0).to(batch["spectral"].dtype)
                    active = voicing_only * voiced
                    loss = loss_s + loss_a
                    loss = loss + 0.85 * _masked_mean_abs(ds, active)
                    loss = loss + 0.70 * _masked_mean_abs(dg, active)
                    loss = loss + 0.38 * _odd_ap_loss(model, batch, controls, active)
                    if probe_batch is None:
                        probe_batch = {k: v.detach().clone() for k, v in batch.items()}

                zeros = {k: torch.zeros_like(v) for k, v in controls.items()}
                zds, zda, zdg = model.predict_residuals(
                    batch["spectral"], batch["ap"], batch["gate"], batch["f0"], zeros
                )
                loss = loss + 0.12 * (zds.abs().mean() + zda.abs().mean() + zdg.abs().mean())

                if training:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    opt.step()
                total += float(loss.detach())
                count += 1
        return total / max(1, count)

    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(int(epochs)):
        tr = run(train_files, True)
        with torch.inference_mode():
            va = run(val, False)
        row = {"epoch": epoch + 1, "train": tr, "val": va}
        if probe_batch is not None:
            row["yv_probe"] = _signed_probe(model, probe_batch)
        history.append(row)
        probe_text = ""
        if "yv_probe" in row:
            q = row["yv_probe"]
            probe_text = (
                f" yv(spec={q['spectral']['rms_positive']:.4f},"
                f" ap_cos={q['ap']['cosine']:.3f}, ap_sum={q['ap']['sum_ratio']:.3f},"
                f" gate={q['gate']['rms_positive']:.4f})"
            )
        print(f"YT+YV signedfix Epoch {epoch + 1}/{epochs}: train={tr:.6f} val={va:.6f}{probe_text}", flush=True)
        if va <= best:
            best = va
            metadata = {
                "training_sources": ["OSF Phonation Modes Dataset", "MOCHA-TIMIT"],
                "training_method": "phonation signedfix v2",
                "feature_backend": "yuaz-native-ddsp-v1",
                "checkpoint_sha256": str(checkpoint_sha),
                "controls": list(CONTROL_NAMES),
                "control_modes": list(CONTROL_MODES),
                "output_scopes": list(OUTPUT_SCOPES),
                "voicing_design": "MOCHA AP-only learned signed residual; neural spectral/gate suppressed; AP odd-symmetry regularized; deterministic carrier retains signed spectral/gate",
                "best_validation_loss": best,
                "epoch": epoch + 1,
                "created_at": time.time(),
                "runtime_gain": 2.0,
            }
            save_ai_control_adapter(output, model, metadata)
    Path(str(output) + ".json").write_text(
        json.dumps({"format": 1, "best_validation_loss": best, "history": history}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Trained:", output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("direct_shards")
    p.add_argument("mocha_shards")
    p.add_argument("output")
    p.add_argument("--checkpoint-sha", required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=2e-4)
    a = p.parse_args()
    train(a.direct_shards, a.mocha_shards, a.output, a.checkpoint_sha, a.epochs, a.lr)


if __name__ == "__main__":
    main()
