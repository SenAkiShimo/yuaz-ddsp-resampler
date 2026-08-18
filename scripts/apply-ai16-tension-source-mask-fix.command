#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3)"
fi
"$PY" - "$ROOT" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])

def replace_once(path, old, new):
    text=path.read_text(encoding='utf-8')
    count=text.count(old)
    if count != 1:
        raise SystemExit(f'{path}: expected 1 match, found {count}')
    path.write_text(text.replace(old,new,1),encoding='utf-8')

core=root/'src/yuaz_ddsp_resampler/core.py'
ai=root/'src/yuaz_ddsp_resampler/ai_vocal_controls.py'
vocal=root/'src/yuaz_ddsp_resampler/vocal_controls.py'

replace_once(core,
'''            frame_controls = None
            if controls.vocal_controls_active or req.get("control_curves"):
                frame_controls = controls.frame_controls(
                    target_frames, self.device, f0_t.dtype, curves=req.get("control_curves")
                )
            canonical_template = self._canonical_articulation_for_variant(bank_record, variant)
''',
'''            frame_controls = None
            if controls.vocal_controls_active or req.get("control_curves"):
                frame_controls = controls.frame_controls(
                    target_frames, self.device, f0_t.dtype, curves=req.get("control_curves")
                )
                if "tension" in frame_controls:
                    source_voiced = torch.from_numpy((src_f0_raw > 1.0).astype(np.float32)).view(1, 1, -1).to(
                        device=self.device, dtype=f0_t.dtype
                    )
                    frame_controls["tension"] = frame_controls["tension"] * source_voiced
            canonical_template = self._canonical_articulation_for_variant(bank_record, variant)
''')

replace_once(ai,
'''        voiced = (f > 1.0).to(f.dtype)
        periodic = (g.clamp(0.0, 1.0) * (1.0 - ap.mean(dim=1, keepdim=True).clamp(0.0, 1.0)) > 0.10).to(f.dtype)
        log_f0 = torch.log2(torch.clamp(f, min=40.0) / 220.0) / 3.0
''',
'''        voiced = (f > 1.0).to(f.dtype)
        log_f0 = torch.log2(torch.clamp(f, min=40.0) / 220.0) / 3.0
''')
replace_once(ai,
'''        return torch.cat([log_s, ap, g, log_f0, c], dim=1), c, voiced * periodic
''',
'''        return torch.cat([log_s, ap, g, log_f0, c], dim=1), c, voiced
''')
replace_once(ai,
'''        f = _resize_time(f0, frames)
        ap = _resize_freq(_resize_time(ap_bands, frames), self.ap_bands)
        g = _resize_time(gate, frames)
        periodic = g.clamp(0.0, 1.0) * (1.0 - ap.mean(dim=1, keepdim=True).clamp(0.0, 1.0)) > 0.10
        active_s = (activity > 0) & (f > 1.0) & periodic
''',
'''        f = _resize_time(f0, frames)
        active_s = (activity > 0) & (f > 1.0)
''')
replace_once(ai,
'''                "control_gate_mode": "periodic-active",
''',
'''                "control_gate_mode": "source-tension-active-voiced",
''')

replace_once(vocal,
'''def _periodic_mask(ap_bands, gate, frames, device, dtype):
    g = _interp_curve(gate, frames, device, dtype).clamp(0.0, 1.0)
    ap = ap_bands.to(device=device, dtype=dtype)
    if ap.shape[-1] != int(frames):
        ap = F.interpolate(ap, size=int(frames), mode="linear", align_corners=False)
    score = g * (1.0 - ap.mean(dim=1, keepdim=True).clamp(0.0, 1.0))
    return (score > 0.10).to(dtype)


''','')
replace_once(vocal,
'''    source_periodic = _periodic_mask(ap_bands, gate, frames, device, dtype)
    tension_eff = torch.sign(tension) * torch.pow(torch.abs(tension), 0.72) * tension_scale * source_periodic
''',
'''    tension_eff = torch.sign(tension) * torch.pow(torch.abs(tension), 0.72) * tension_scale
''')
PY

"$PY" -m py_compile \
  src/yuaz_ddsp_resampler/core.py \
  src/yuaz_ddsp_resampler/ai_vocal_controls.py \
  src/yuaz_ddsp_resampler/vocal_controls.py

git diff --check

git add \
  src/yuaz_ddsp_resampler/core.py \
  src/yuaz_ddsp_resampler/ai_vocal_controls.py \
  src/yuaz_ddsp_resampler/vocal_controls.py

git commit -m update
git push origin HEAD:agent/ai16-tension-probe
