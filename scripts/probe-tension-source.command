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
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$ROOT" "$WAV" <<'PY'
import json,sys
from pathlib import Path
import torch
import torch.nn.functional as F
from yuaz_ddsp_resampler.ai_control_training import NativeYuazDDSPExtractor
from yuaz_ddsp_resampler.ai_vocal_controls import load_ai_control_adapter
from yuaz_ddsp_resampler.state import find_voicebank_for_input,resolve_active_state
root=Path(sys.argv[1]).resolve()
wav=Path(sys.argv[2]).expanduser().resolve()
bank=find_voicebank_for_input(wav)
if bank is None:
    raise RuntimeError('voicebank not found')
state,_=resolve_active_state(bank,verify=True)
if state is None:
    raise RuntimeError('active ai14 state not found')
pack,meta=load_ai_control_adapter(state/'ai_phonation_adapter.ai14.pt',device='cpu',expected_controls=('tension','voicing'))
native=NativeYuazDDSPExtractor(root)
feat=native.features(wav)
T=min(feat['log_spec'].shape[-1],max(1,int(round(native.sample_rate/native.hop))))
S=torch.from_numpy(feat['log_spec'][:,:T]).unsqueeze(0).exp().float()
AP=torch.from_numpy(feat['ap'][:,:T]).unsqueeze(0).float()
G=torch.from_numpy(feat['gate'][:,:T]).unsqueeze(0).float()
SRCF0=torch.from_numpy(feat['f0'][:,:T]).unsqueeze(0).float()
F0=torch.full((1,1,T),261.625565)
strict=(SRCF0>1.0)
score_all=G.clamp(0,1)*(1-AP.mean(dim=1,keepdim=True).clamp(0,1))
score_low4=G.clamp(0,1)*(1-AP[:,:4].mean(dim=1,keepdim=True).clamp(0,1))
score_low8=G.clamp(0,1)*(1-AP[:,:8].mean(dim=1,keepdim=True).clamp(0,1))
score_min=G.clamp(0,1)*(1-AP.amin(dim=1,keepdim=True).clamp(0,1))
score_q25=G.clamp(0,1)*(1-torch.quantile(AP,0.25,dim=1,keepdim=True).clamp(0,1))
def metrics(score,threshold):
    pred=score>threshold
    tp=float((pred&strict).float().sum())
    fp=float((pred&~strict).float().sum())
    fn=float((~pred&strict).float().sum())
    precision=tp/max(1.0,tp+fp)
    recall=tp/max(1.0,tp+fn)
    return {
        'fraction':float(pred.float().mean()),
        'precision_vs_source_f0':precision,
        'recall_vs_source_f0':recall,
    }
print(json.dumps({
    'frames':T,
    'source_f0_voiced_fraction':float(strict.float().mean()),
    'gate_mean':float(G.mean()),
    'ap_mean':float(AP.mean()),
    'ap_band_means':[float(x) for x in AP.mean(dim=(0,2))],
    'all_mean':metrics(score_all,0.10),
    'low4_mean':metrics(score_low4,0.10),
    'low8_mean':metrics(score_low8,0.10),
    'min_band':metrics(score_min,0.10),
    'q25':metrics(score_q25,0.10),
    'score_all_max':float(score_all.max()),
    'score_low4_max':float(score_low4.max()),
    'score_low8_max':float(score_low8.max()),
    'score_min_max':float(score_min.max()),
    'score_q25_max':float(score_q25.max()),
},separators=(',',':')))
for value in (-1.0,1.0):
    controls={'tension':torch.full((1,1,T),value),'voicing':torch.zeros((1,1,T))}
    with torch.inference_mode():
        x,c,mask=pack._context(S,AP,G,F0,controls)
        h=F.silu(pack.input_proj(x))
        h=h+0.45*F.silu(pack.temporal1(h))
        h=h+0.45*F.silu(pack.temporal2(h))
        h=h+0.30*F.silu(pack.temporal3(h))
        y=pack.output_proj(h)
        pre_s=y[:,:pack.spectral_bands]
        pre_ap=y[:,pack.spectral_bands:pack.spectral_bands+pack.ap_bands]
        pre_gate=y[:,-1:]
        ds,da,dg=pack.predict_residuals(S,AP,G,F0,controls)
    def rms(z):
        return float(torch.sqrt(torch.mean(torch.tanh(z).pow(2))+1e-12))
    print(json.dumps({
        'yt':int(value*100),
        'mask_fraction':float(mask.mean()),
        'pre_s':rms(pre_s),
        'pre_ap':rms(pre_ap),
        'pre_gate':rms(pre_gate),
        'post_s':rms(ds),
        'post_ap':rms(da),
        'post_gate':rms(dg),
    },separators=(',',':')))
PY