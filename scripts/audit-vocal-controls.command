#!/bin/bash
set -euo pipefail
ROOT="$HOME/Library/Application Support/YuazDDSP/0.2.8ai.16"
PY="$ROOT/.venv/bin/python"
[ -d "$ROOT" ] || { echo "Installed ai16 runtime not found: $ROOT" >&2; exit 1; }
[ -x "$PY" ] || { echo "Installed ai16 Python not found: $PY" >&2; exit 1; }
WAV="${1:-}"
if [ -z "$WAV" ]; then
  read -r WAV
fi
[ -f "$WAV" ] || { echo "WAV not found: $WAV" >&2; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$ROOT" "$WAV" <<'PY'
import json
import sys
import tempfile
from pathlib import Path

from yuaz_ddsp_resampler import client
from yuaz_ddsp_resampler.ai_vocal_controls import load_ai_control_adapter
from yuaz_ddsp_resampler.controls import parse_yuaz_controls
from yuaz_ddsp_resampler.state import find_voicebank_for_input, lookup_local_record, resolve_active_state

root = Path(sys.argv[1]).resolve()
wav = Path(sys.argv[2]).expanduser().resolve()
bank = find_voicebank_for_input(wav)
if bank is None:
    raise RuntimeError("voicebank not found")
state, info = resolve_active_state(bank, verify=True)
if state is None:
    raise RuntimeError("active ai14 state not found")
record = lookup_local_record(wav) or {}

pack_fields = {
    "technique": "ai_control_adapter",
    "gender": "ai_gender_adapter",
    "phonation": "ai_phonation_adapter",
    "mouth": "ai_mouth_adapter",
}
pack_inventory = []
for label, field in pack_fields.items():
    raw = record.get(field)
    entry = {"label": label, "field": field, "path": str(raw or ""), "exists": False}
    if raw:
        p = Path(str(raw)).expanduser()
        entry["exists"] = p.is_file()
        if p.is_file():
            try:
                pack, meta = load_ai_control_adapter(p, device="cpu")
                entry.update({
                    "controls": list(pack.control_names),
                    "modes": list(pack.control_modes),
                    "scopes": list(pack.output_scopes),
                    "runtime_gain": float(pack.runtime_gain),
                    "checkpoint_sha256": str(meta.get("checkpoint_sha256") or ""),
                })
            except Exception as exc:
                entry["load_error"] = str(exc)
    pack_inventory.append(entry)

print(json.dumps({
    "type": "state",
    "bank": str(bank),
    "generation": state.name,
    "state_source": (info or {}).get("source"),
    "packs": pack_inventory,
}, ensure_ascii=False, separators=(",", ":")))

config, config_path = client.load_config(root)
host = config.get("host", "127.0.0.1")
port = int(config.get("port", client.DEFAULT_PORT))
runtime_id = str(config.get("runtime_id") or client.ENGINE_VERSION)
status = client.ping(host, port)
if not (status and status.get("ready") and status.get("runtime_id") == runtime_id and Path(status.get("runtime_root", "")).expanduser().resolve() == root):
    client.start_server(root, config_path, host, port, runtime_id)

controls = [
    ("YB", "breathiness", "yuaz_breathiness", "yuaz_breathiness_backend"),
    ("YG", "gender_formant", "yuaz_gender_formant", "yuaz_gender_backend"),
    ("YO", "mouth", "yuaz_mouth", "yuaz_mouth_backend"),
    ("YX", "mixed_voice", None, None),
    ("YP", "pharyngeal", None, None),
    ("YV", "voicing", "yuaz_voicing", "yuaz_voicing_backend"),
    ("YF", "falsetto", None, None),
    ("YT", "tension", "yuaz_tension", "yuaz_tension_backend"),
]

mode_map = {}
for item in pack_inventory:
    for name, mode in zip(item.get("controls") or [], item.get("modes") or []):
        mode_map[name] = mode

for flag, name, parsed_key, backend_key in controls:
    mode = mode_map.get(name)
    values = (-100, 100) if mode == "signed" else (100,)
    if mode is None:
        values = (-100, 100)
    for value in values:
        parsed = parse_yuaz_controls(f"{flag}{value}")
        parsed_value = float(getattr(parsed, name))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = Path(f.name)
        req = {
            "input": str(wav), "output": str(out), "tone": "C4", "velocity": 100.0,
            "flags": f"{flag}{value}", "offset": 0.0, "length": 1000.0,
            "consonant": 0.0, "cutoff": 0.0, "volume": 100.0,
            "modulation": 0.0, "tempo": "!120", "pitch": "AA",
        }
        result = client.send(host, port, {"action": "render", "request": req, "runtime_id": runtime_id}, timeout=600)
        out.unlink(missing_ok=True)
        if not result.get("ok"):
            print(json.dumps({
                "type": "control", "flag": flag, "control": name, "value": value,
                "mode": mode, "parser_value": parsed_value, "render_ok": False,
                "error": result.get("error", "render failed"),
            }, ensure_ascii=False, separators=(",", ":")))
            continue
        effect = None
        for item in result.get("yuaz_ai_effects") or []:
            if name in (item.get("pack_controls") or []):
                effect = item
                break
        raw = 0.0
        applied = 0.0
        collapsed = None
        gain = None
        controls_seen = None
        if effect is not None:
            raw = max(float(effect.get("raw_spectral_rms", 0.0)), float(effect.get("raw_ap_rms", 0.0)), float(effect.get("raw_gate_rms", 0.0)))
            applied = max(float(effect.get("applied_spectral_log_rms", 0.0)), float(effect.get("applied_ap_rms", 0.0)), float(effect.get("applied_gate_rms", 0.0)))
            collapsed = bool(effect.get("collapsed", False))
            gain = float(effect.get("runtime_gain", 0.0))
            controls_seen = effect.get("controls")
        payload = {
            "type": "control",
            "flag": flag,
            "control": name,
            "value": value,
            "mode": mode,
            "parser_value": parsed_value,
            "render_ok": True,
            "direct_controls": result.get("yuaz_ai_direct_controls") or [],
            "effect_found": effect is not None,
            "effect_controls": controls_seen,
            "raw_max": raw,
            "applied_max": applied,
            "gain": gain,
            "collapsed": collapsed,
        }
        if parsed_key:
            payload["runtime_parsed"] = result.get(parsed_key)
        if backend_key:
            payload["backend"] = result.get(backend_key)
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
PY