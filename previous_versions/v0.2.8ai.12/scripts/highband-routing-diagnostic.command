#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INST="$HOME/Library/Application Support/YuazDDSP/0.2.8ai.12/logs/render_requests.jsonl"
LOCAL="$ROOT/logs/render_requests.jsonl"
LOG=""
[ -f "$INST" ] && LOG="$INST"
[ -z "$LOG" ] && [ -f "$LOCAL" ] && LOG="$LOCAL"
if [ -z "$LOG" ]; then
  echo "No 0.2.8ai.12 render log yet. Render one note with YH0/YH100 first."
  exit 0
fi
python3 - "$LOG" <<'PY'
import json,re,sys
from pathlib import Path
p=Path(sys.argv[1])
rows=[]
for line in p.read_text(encoding='utf-8',errors='replace').splitlines():
    try:d=json.loads(line)
    except Exception:continue
    q=d.get('request',{});r=d.get('result',{})
    flags=str(q.get('flags',''))
    if re.search(r'YH[+-]?(?:\d+(?:\.\d*)?|\.\d+)',flags,re.I): rows.append((q,r))
if not rows:
    print('No YH render found in',p);raise SystemExit(0)
q,r=rows[-1]
print('Yuaz 0.2.8ai.12 latest YH routing')
print('flags:',repr(q.get('flags','')))
print('input:',q.get('input'))
for k in (
 'analysis_sr','ddsp_synthesis_sr','ddsp_synthesis_backend','ddsp_fullband_body_used',
 'ddsp_fullband_crossover_start_hz','ddsp_fullband_crossover_full_hz','ddsp_fullband_branch_rms',
 'ddsp_fullband_safety_gain','ddsp_fullband_harmonic_count','ddsp_fullband_fft_size',
 'ddsp_upperband_parameter_head_enabled','ddsp_upperband_parameter_head_used','ddsp_upperband_parameter_head_revision',
 'ddsp_upperband_head_start_hz','ddsp_upperband_head_full_hz','ddsp_upperband_spectral_slope_db_per_oct',
 'ddsp_upperband_ap_mean','ddsp_upperband_harmonic_weight_mean','ddsp_upperband_noise_weight_mean','ddsp_upperband_mix_scale_mean',
 'ddsp_upperband_edge_rms','ddsp_upperband_seam_gain','ddsp_upperband_presence_gain','ddsp_upperband_air_gain',
 'output_sr','yuaz_highband_strength','yuaz_highband_assist_start_hz',
 'learned_highband_db_found','learned_highband_db_source','learned_highband_db_path',
 'learned_highband_requested_base_alias','learned_highband_profile_match_mode','learned_highband_selected_base_alias',
 'learned_highband_profile_found','learned_highband_used','learned_highband_rms',
 'learned_highband_harmonic_count','learned_highband_harmonic_mix','learned_highband_temporal_used',
 'highband_backend','highband_foundation_found','highband_foundation_source','highband_foundation_path',
 'highband_continuity_hybrid_used','highband_temporal_coverage_before','highband_temporal_coverage_after',
 'highband_continuity_assist_rms','highband_foundation_branch_rms','highband_continuity_branch_rms',
 'highband_nyquist_seam_edge_rms','highband_nyquist_seam_gain','highband_nyquist_air_gain',
 'highband_nyquist_body_taper_amount'):
    print(f'{k}:',r.get(k))
PY
