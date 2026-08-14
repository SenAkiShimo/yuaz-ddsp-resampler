#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
strip_path() {
  python3 - "$1" <<'PY'
import shlex, sys
parts = shlex.split(sys.argv[1].strip())
print(parts[0] if parts else "")
PY
}
YUAZ="${YUAZ_SGR_ROOT:-$HOME/Downloads/yuaz-sgr}"
if [ ! -d "$YUAZ/yuaz_sgr/models" ]; then
  echo "Drop the Yuaz SGR repository folder here, then press Return:"
  read -r RAW
  YUAZ="$(strip_path "$RAW")"
fi
[ -d "$YUAZ/yuaz_sgr/models" ] || { echo "Yuaz SGR model directory not found: $YUAZ/yuaz_sgr/models"; exit 1; }
CKPT="${YUAZ_SGR_CHECKPOINT:-$YUAZ/yuaz_sgr_inference/checkpoint_300k.pt}"
if [ ! -f "$CKPT" ]; then
  echo "Drop checkpoint_300k.pt here, then press Return:"
  read -r RAW
  CKPT="$(strip_path "$RAW")"
fi
[ -f "$CKPT" ] || { echo "Checkpoint not found: $CKPT"; exit 1; }
APP_STATE="$HOME/Library/Application Support/YuazDDSP/state"
mkdir -p "$APP_STATE"
python3 - "$YUAZ" "$CKPT" "$APP_STATE/voicebank_registry-0.2.8ai13.json" <<'PY'
import json, sys
from pathlib import Path
config = {
    "yuaz_repo": str(Path(sys.argv[1]).resolve()),
    "checkpoint": str(Path(sys.argv[2]).resolve()),
    "host": "127.0.0.1",
    "port": 47885,
    "engine_version": "0.2.8ai.13",
    "runtime_id": "yuaz-0.2.8ai.13-control-v13",
    "acoustic_base": "0.2.8ai.13-slope-continuity-topguard-v2",
    "transition_ms": 70.0,
    "use_rvq": False,
    "output_sr": 44100,
    "ddsp_synthesis_sr": 48000,
    "ddsp_fullband_crossover_start_hz": 8800.0,
    "ddsp_fullband_crossover_full_hz": 12100.0,
    "ai12_upperband_head_enabled": True,
    "ai12_upperband_head_start_hz": 8400.0,
    "ai12_upperband_head_full_hz": 12400.0,
    "ai13_upperband_guard_enabled": True,
    "ai13_upperband_head_start_hz": 8200.0,
    "ai13_upperband_head_full_hz": 13800.0,
    "normalize_voicebank_loudness": True,
    "normalization_target_dbfs": -18.0,
    "normalization_peak_ceiling_dbfs": -1.0,
    "normalization_peak_guard_knee_db": 3.0,
    "normalization_emergency_max_abs_gain_db": 30.0,
    "normalization_tolerance_db": 0.05,
    "enable_fidelity_refiner": True,
    "fidelity_residual_hard_limit": 0.085,
    "registry_path": str(Path(sys.argv[3]).expanduser().resolve()),
    "state_namespace": ".yuaz-0.2.8ai13",
    "previous_028ai12_readonly_state_namespace": ".yuaz-0.2.8ai12",
    "previous_028ai11_readonly_state_namespace": ".yuaz-0.2.8ai11",
    "previous_028ai10_readonly_state_namespace": ".yuaz-0.2.8ai10",
    "previous_028ai9_readonly_state_namespace": ".yuaz-0.2.8ai9",
    "previous_028ai8_readonly_state_namespace": ".yuaz-0.2.8ai8",
    "previous_028ai7_readonly_state_namespace": ".yuaz-0.2.8ai7",
    "previous_028ai6_readonly_state_namespace": ".yuaz-0.2.8ai6",
    "previous_028ai5_readonly_state_namespace": ".yuaz-0.2.8ai5",
    "previous_028ai4_readonly_state_namespace": ".yuaz-0.2.8ai4",
    "previous_028ai3_readonly_state_namespace": ".yuaz-0.2.8ai3",
    "previous_028ai2_readonly_state_namespace": ".yuaz-0.2.8ai2",
    "previous_028ai1_readonly_state_namespace": ".yuaz-0.2.8ai1",
    "previous_028_readonly_state_namespace": ".yuaz-0.2.8ai",
    "predecessor_readonly_state_namespace": ".yuaz-alpha8-rc4-3-ai3",
    "stable_readonly_state_namespace": ".yuaz-alpha8-rc3-3",
    "ai_control_model_policy": "voicebank-pinned-modular-foundations",
}
Path("config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
print("Wrote config.json")
print("Registry accelerator:", config["registry_path"])
print("Migration source order before purge: 0.2.8ai.12 / 0.2.8ai.11 / 0.2.8ai.10 / 0.2.8ai.9 / 0.2.8ai.8 / 0.2.8ai.7 / 0.2.8ai.6 / 0.2.8ai.5 / 0.2.8ai.4 / 0.2.8ai.3 / 0.2.8ai.2 / 0.2.8ai.1 / 0.2.8ai / AI.3 / RC4.2; new state writes only .yuaz-0.2.8ai13")
print("Dual-rate DDSP: preserved ai.12 upper-band head + ai.13 8.2-13.8 kHz slope-continuity crossover + output-rate-aware top-band guard; YH is refinement")
print("Stage C Fidelity Refiner: enabled for Deep modes")
PY
chmod +x yuaz-ddsp-resampler scripts/*.command *.command 2>/dev/null || true
