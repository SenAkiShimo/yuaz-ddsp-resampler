#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || exit 1
WAV="${1:-}"
if [ -z "$WAV" ]; then
  read -r WAV
fi
[ -f "$WAV" ] || exit 1
pkill -f yuaz_ddsp_resampler.server 2>/dev/null || true
sleep 1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$ROOT" "$WAV" <<'PY'
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

from yuaz_ddsp_resampler import client

root = Path(sys.argv[1]).resolve()
wav = Path(sys.argv[2]).resolve()
config, config_path = client.load_config(root)
host = config.get("host", "127.0.0.1")
port = int(config.get("port", client.DEFAULT_PORT))
runtime_id = str(config.get("runtime_id") or client.ENGINE_VERSION)

client.start_server(root, config_path, host, port, runtime_id)
status = client.ping(host, port) or {}

base_request = {
    "input": str(wav),
    "tone": "C4",
    "velocity": 100.0,
    "flags": "YT100",
    "offset": 0.0,
    "length": 1000.0,
    "consonant": 0.0,
    "cutoff": 0.0,
    "volume": 100.0,
    "modulation": 0.0,
    "tempo": "!120",
    "pitch": "AA",
}

barrier = threading.Barrier(3)
results = [None, None]


def run(index):
    with tempfile.NamedTemporaryFile(suffix=f"-{index}.wav", delete=False) as f:
        out = Path(f.name)
    request = dict(base_request)
    request["output"] = str(out)
    barrier.wait()
    started = time.perf_counter()
    try:
        response = client.send(
            host,
            port,
            {"action": "render", "request": request, "runtime_id": runtime_id},
            timeout=600,
        )
        elapsed = time.perf_counter() - started
        results[index] = {
            "ok": bool(response.get("ok")),
            "error": response.get("error"),
            "elapsed": round(elapsed, 4),
            "output_exists": out.is_file(),
            "output_bytes": out.stat().st_size if out.is_file() else 0,
            "backend": response.get("yuaz_tension_backend"),
            "packs": response.get("yuaz_ai_direct_controls"),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        results[index] = {
            "ok": False,
            "error": repr(exc),
            "elapsed": round(elapsed, 4),
            "output_exists": out.is_file(),
            "output_bytes": out.stat().st_size if out.is_file() else 0,
        }
    finally:
        out.unlink(missing_ok=True)


threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(2)]
for thread in threads:
    thread.start()
barrier.wait()
for thread in threads:
    thread.join()

status = client.ping(host, port) or {}
print(json.dumps({
    "server_pid": status.get("pid"),
    "concurrent": results,
    "server_ready": status.get("ready"),
    "server_error": status.get("error"),
    "active_renders": status.get("active_renders"),
}, ensure_ascii=False, separators=(",", ":")))
PY
