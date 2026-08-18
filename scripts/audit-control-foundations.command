#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || exit 1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
from yuaz_ddsp_resampler.ai_vocal_controls import load_ai_control_adapter
from yuaz_ddsp_resampler.checkpoint_identity import checkpoint_identity, checkpoint_identity_sha

root=Path(sys.argv[1]).resolve()
cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
checkpoint=Path(cfg['checkpoint']).expanduser().resolve()
ident=checkpoint_identity(checkpoint)
current=checkpoint_identity_sha(checkpoint)
print(json.dumps({
    'type':'checkpoint',
    'runtime_sha256':ident.get('runtime_sha256'),
    'source_checkpoint_sha256':ident.get('source_checkpoint_sha256'),
    'identity_sha256':current,
    'source_checkpoint':ident.get('source_checkpoint'),
    'source_step':ident.get('source_step'),
},separators=(',',':')))

home=Path.home()
specs=[
    ('technique','ai_control_foundation-v2.pt',('breathiness','falsetto','mixed_voice','pharyngeal')),
    ('gender','ai_gender_foundation-v1.pt',('gender_formant',)),
    ('mouth','ai_mouth_foundation-v1.pt',('mouth',)),
    ('phonation','ai_phonation_foundation-v1.pt',('tension','voicing')),
]
extra={
    'technique':['ai_control_foundation-v2-Chinese-Core.pt'],
    'gender':['ai_gender_foundation-v1-VocalSet.pt'],
    'mouth':['ai_mouth_foundation-v1-MOCHA.pt'],
    'phonation':['ai_phonation_foundation-v1-PhonationModes-MOCHA.pt','ai_phonation_foundation-v1-VQS-MOCHA.pt'],
}
roots=[
    root/'control_models',
    home/'Documents'/'Yuaz-DDSP-Backups'/'control-models',
    home/'Library'/'Application Support'/'YuazDDSP'/'0.2.8ai.14'/'control_models',
    home/'Downloads'/'yuaz-ddsp-resampler-v0.2.8ai.14'/'control_models',
    home/'Downloads'/'yuaz-ddsp-resampler-v0.2.8ai.14-phonation-fix'/'control_models',
]
for label,filename,expected in specs:
    names=[filename]+extra[label]
    seen=set(); found=[]
    for base in roots:
        for name in names:
            p=(base/name).expanduser()
            try: key=str(p.resolve())
            except Exception: key=str(p)
            if key in seen: continue
            seen.add(key)
            if p.is_file(): found.append(p.resolve())
    if not found:
        print(json.dumps({'type':'foundation','label':label,'found':False},separators=(',',':')))
        continue
    for p in found:
        row={'type':'foundation','label':label,'found':True,'path':str(p)}
        try:
            pack,meta=load_ai_control_adapter(p,device='cpu')
            controls=tuple(getattr(pack,'control_names',()))
            modes=tuple(getattr(pack,'control_modes',()))
            scopes=tuple(getattr(pack,'output_scopes',()))
            model_sha=str(meta.get('checkpoint_sha256') or '')
            backend=str(meta.get('feature_backend') or '')
            row.update({
                'controls':list(controls),
                'modes':list(modes),
                'scopes':list(scopes),
                'feature_backend':backend,
                'checkpoint_sha256':model_sha,
                'control_set_ok':controls==expected,
                'backend_ok':backend=='yuaz-native-ddsp-v1',
                'checkpoint_ok':bool(model_sha and model_sha==current),
                'compatible':bool(controls==expected and backend=='yuaz-native-ddsp-v1' and model_sha==current),
            })
        except Exception as exc:
            row.update({'load_error':str(exc),'compatible':False})
        print(json.dumps(row,separators=(',',':')))
PY
