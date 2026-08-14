#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INST="$HOME/Library/Application Support/YuazDDSP/0.2.8ai.12/logs/render_requests.jsonl"
LOCAL="$ROOT/logs/render_requests.jsonl"
LOG=""
[ -f "$INST" ] && LOG="$INST"
[ -z "$LOG" ] && [ -f "$LOCAL" ] && LOG="$LOCAL"
if [ -z "$LOG" ]; then
  echo "No 0.2.8ai.12 render log yet. Render at least one note in OpenUtau first."
  exit 0
fi
python3 - "$LOG" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); lines=p.read_text(encoding='utf-8',errors='replace').splitlines()[-12:]
print('Yuaz 0.2.8ai.12 recent render routing:',p)
for line in lines:
 try: d=json.loads(line)
 except Exception: continue
 q=d.get('request',{});r=d.get('result',{})
 print('\nflags=',repr(q.get('flags','')),'tone=',q.get('tone'),'input=',Path(q.get('input','')).name)
 print('  raw_bypass=',r.get('yuaz_raw_bypass',False),'packs=',r.get('yuaz_ai_control_packs'),'controls=',r.get('yuaz_ai_direct_controls'))
 print('  parsed: YT=',r.get('yuaz_tension'),'YB=',r.get('yuaz_breathiness'),'YV=',r.get('yuaz_voicing'),'YG=',r.get('yuaz_gender_formant'),'YO=',r.get('yuaz_mouth'))
 print('  highband: YH=',r.get('yuaz_highband_strength'),'db_found=',r.get('learned_highband_db_found'),'profile_found=',r.get('learned_highband_profile_found'),'used=',r.get('learned_highband_used'),'rms=',r.get('learned_highband_rms'))
 print('  highband-route: source=',r.get('learned_highband_db_source'),'match=',r.get('learned_highband_profile_match_mode'),'requested=',repr(r.get('learned_highband_requested_base_alias')),'selected=',repr(r.get('learned_highband_selected_base_alias')))
 for eff in r.get('yuaz_ai_effects') or []:
  print('  effect',eff.get('pack_controls'),'ctrl=',eff.get('controls'),'raw=',max(eff.get('raw_spectral_rms',0),eff.get('raw_ap_rms',0),eff.get('raw_gate_rms',0)),'gain=',eff.get('runtime_gain'),'applied=',max(eff.get('applied_spectral_log_rms',0),eff.get('applied_ap_rms',0),eff.get('applied_gate_rms',0)),'collapsed=',eff.get('collapsed'))
 if r.get('yuaz_ai_collapsed_packs'): print('  COLLAPSED:',r.get('yuaz_ai_collapsed_packs'))
PY
