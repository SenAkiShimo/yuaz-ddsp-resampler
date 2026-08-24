#!/bin/bash
set -euo pipefail
RUNTIME="$HOME/Library/Application Support/YuazDDSP/0.2.9"
PY="$RUNTIME/.venv/bin/python"
[ -x "$PY" ] || { echo "Runtime Python not found: $PY" >&2; exit 1; }
export PYTHONPATH="$RUNTIME/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$RUNTIME" <<'PY'
import copy
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from yuaz_ddsp_resampler import client

root = Path(sys.argv[1]).expanduser().resolve()
log = root / "logs" / "render_requests.jsonl"
if not log.is_file():
    raise SystemExit("No render_requests.jsonl found. Render one note in OpenUtau first.")
lines = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
if not lines:
    raise SystemExit("render_requests.jsonl is empty. Render one note in OpenUtau first.")
base = None
for line in reversed(lines):
    try:
        record = json.loads(line)
    except Exception:
        continue
    request = dict(record.get("request") or {})
    input_path = request.get("input")
    if input_path and Path(input_path).expanduser().is_file():
        base = request
        break
if base is None:
    raise SystemExit(
        "No logged OpenUtau request still has an existing input WAV. "
        "Render one note in OpenUtau now, then rerun this probe immediately."
    )

config, config_path = client.load_config(root)
host = config.get("host", "127.0.0.1")
port = int(config.get("port", client.DEFAULT_PORT))
runtime_id = str(config.get("runtime_id") or client.ENGINE_VERSION)
client.start_server(root, config_path, host, port, runtime_id)

pat = re.compile(r"YT[+-]?(?:\d+(?:\.\d*)?|\.\d+)", re.I)

def with_yt(flags, value):
    flags = str(flags or "")
    replacement = f"YT{value}"
    if pat.search(flags):
        return pat.sub(replacement, flags, count=1)
    return flags + replacement

def load_audio(path):
    y, sr = sf.read(path, always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = np.mean(y, axis=1)
    return np.asarray(y, dtype=np.float64), int(sr)

def stats(path):
    y, sr = load_audio(path)
    if y.size:
        steps = np.abs(np.diff(y)) if y.size > 1 else np.zeros(1)
        max_step_index = int(np.argmax(steps)) if steps.size else 0
        return {
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16],
            "samples": int(y.size),
            "sr": sr,
            "max_abs": float(np.max(np.abs(y))),
            "rms": float(np.sqrt(np.mean(y * y) + 1e-18)),
            "max_step": float(np.max(steps)) if steps.size else 0.0,
            "p999_step": float(np.quantile(steps, 0.999)) if steps.size else 0.0,
            "max_step_index": max_step_index,
        }
    return {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16], "samples": 0, "sr": sr}

with tempfile.TemporaryDirectory(prefix="yuaz-yt01-") as td:
    td = Path(td)
    outputs = {}
    arrays = {}
    for value in (0, 1):
        outputs[value] = []
        arrays[value] = []
        for repeat in range(2):
            req = copy.deepcopy(base)
            req["flags"] = with_yt(req.get("flags", ""), value)
            out = td / f"yt{value}-{repeat}.wav"
            req["output"] = str(out)
            response = client.send(host, port, {"action": "render", "request": req, "runtime_id": runtime_id}, timeout=600)
            if not response.get("ok"):
                raise RuntimeError(f"YT{value} render failed: {response.get('error')}")
            outputs[value].append({"response": {k: response.get(k) for k in ("ok", "diagnostic", "yuaz_tension")}, "stats": stats(out)})
            arrays[value].append(load_audio(out)[0])

    def diff(a, b):
        n = min(len(a), len(b))
        if n <= 0:
            return {"samples": 0}
        d = np.asarray(a[:n] - b[:n], dtype=np.float64)
        ds = np.abs(np.diff(d)) if n > 1 else np.zeros(1)
        return {
            "samples": int(n),
            "max_abs": float(np.max(np.abs(d))),
            "rms": float(np.sqrt(np.mean(d * d) + 1e-18)),
            "max_step": float(np.max(ds)) if ds.size else 0.0,
        }

    result = {
        "source_request": {
            "input": base.get("input"),
            "tone": base.get("tone"),
            "length": base.get("length"),
            "flags": base.get("flags"),
        },
        "yt0": outputs[0],
        "yt1": outputs[1],
        "repeat_diff_yt0": diff(arrays[0][0], arrays[0][1]),
        "repeat_diff_yt1": diff(arrays[1][0], arrays[1][1]),
        "yt0_vs_yt1": diff(arrays[0][0], arrays[1][0]),
    }
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
PY
