#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "Python not found: $PY" >&2; exit 1; }
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
import numpy as np
from yuaz_ddsp_resampler.checkpoint_identity import checkpoint_identity_sha

root=Path(sys.argv[1]).resolve()
config=json.loads((root/'config.json').read_text(encoding='utf-8'))
sha=checkpoint_identity_sha(Path(config['checkpoint']).expanduser())
short=sha[:16]
data_root=Path.home()/'YuazControlDatasets'/'_yuaz_ai_cache'
tech_dir=data_root/f'gtsinger-ddsp-v2-direct-{short}'


def cosine(a,b):
    a=np.asarray(a,dtype=np.float64).reshape(-1); b=np.asarray(b,dtype=np.float64).reshape(-1)
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den>1e-12 else 0.0

def rms(a):
    a=np.asarray(a,dtype=np.float64)
    return float(np.sqrt(np.mean(a*a)+1e-12))

def compare(a,b):
    ra=rms(a); rb=rms(b); mean=max(1e-8,.5*(ra+rb))
    return {'rms_a':ra,'rms_b':rb,'cosine':cosine(a,b),'difference_ratio':rms(np.asarray(a)-np.asarray(b))/mean,'sum_ratio':rms(np.asarray(a)+np.asarray(b))/mean}

def aggregate_technique(name):
    sums={'spectral':None,'ap':None,'gate':None}; n=0; shards=0
    if not tech_dir.is_dir(): return None
    for p in sorted(tech_dir.glob('*.npz')):
        z=np.load(p,allow_pickle=False)
        t=str(z['technique'].item()) if 'technique' in z else ''
        if t!=name: continue
        controls=np.asarray(z['controls'],dtype=np.float32)
        idx={'breathy':0,'falsetto':1,'mixed_voice':2,'pharyngeal':3}[name]
        mask=controls[idx]>0.5
        if not np.any(mask): continue
        vals={'spectral':np.asarray(z['target_ds'],dtype=np.float64)[:,mask],
              'ap':np.asarray(z['target_da'],dtype=np.float64)[:,mask],
              'gate':np.asarray(z['target_dg'],dtype=np.float64)[:,mask]}
        for k,v in vals.items():
            s=v.sum(axis=1)
            sums[k]=s if sums[k] is None else sums[k]+s
        n+=int(mask.sum()); shards+=1
    if n==0:return None
    return {'frames':n,'shards':shards,'centroids':{k:v/n for k,v in sums.items()}}

print(json.dumps({'type':'config','checkpoint_sha256':sha,'technique_cache':str(tech_dir),'technique_cache_exists':tech_dir.is_dir()},separators=(',',':')))
b=aggregate_technique('breathy'); f=aggregate_technique('falsetto')
if b and f:
    print(json.dumps({'type':'technique_counts','breathy_frames':b['frames'],'breathy_shards':b['shards'],'falsetto_frames':f['frames'],'falsetto_shards':f['shards']},separators=(',',':')))
    for scope in ('spectral','ap','gate'):
        row={'type':'target_yb_yf','scope':scope}; row.update(compare(b['centroids'][scope],f['centroids'][scope])); print(json.dumps(row,separators=(',',':')))
else:
    print(json.dumps({'type':'skip','target':'YB/YF','reason':'current technique cache or targets missing'},separators=(',',':')))

phon_dirs=[]
for d in sorted(data_root.glob(f'*{short}*')):
    p=d/'phonation_shards'
    if p.is_dir(): phon_dirs.append(p)
print(json.dumps({'type':'phonation_dirs','dirs':[str(x) for x in phon_dirs]},separators=(',',':')))

pos={'ap':None,'gate':None}; neg={'ap':None,'gate':None}; np_=nn_=0; ps=ns=0
for d in phon_dirs:
    for p in sorted(d.glob('*.npz')):
        z=np.load(p,allow_pickle=False)
        c=np.asarray(z['controls'],dtype=np.float32)
        if c.ndim!=2 or c.shape[0]<2: continue
        v=c[1]
        ma=np.asarray(z['mask_a'],dtype=np.float32).reshape(-1)>0.5
        mg=np.asarray(z['mask_g'],dtype=np.float32).reshape(-1)>0.5
        for sign,label in ((1,'pos'),(-1,'neg')):
            m=(v*sign>0.18)
            target=pos if sign>0 else neg
            count_name='pos' if sign>0 else 'neg'
            for scope,key,mask in (('ap','target_da',ma),('gate','target_dg',mg)):
                mm=m & mask
                if not np.any(mm): continue
                arr=np.asarray(z[key],dtype=np.float64)[:,mm]
                s=arr.sum(axis=1)
                target[scope]=s if target[scope] is None else target[scope]+s
            if np.any(m & (ma|mg)):
                if sign>0: np_+=int(np.count_nonzero(m & (ma|mg))); ps+=1
                else: nn_+=int(np.count_nonzero(m & (ma|mg))); ns+=1
if np_ and nn_:
    print(json.dumps({'type':'voicing_counts','positive_frames':np_,'negative_frames':nn_,'positive_shards':ps,'negative_shards':ns},separators=(',',':')))
    for scope in ('ap','gate'):
        if pos[scope] is None or neg[scope] is None: continue
        pc=pos[scope]/max(1,np_); nc=neg[scope]/max(1,nn_)
        row={'type':'target_yv_signed','scope':scope}; row.update(compare(nc,pc)); print(json.dumps(row,separators=(',',':')))
else:
    print(json.dumps({'type':'skip','target':'YV signed targets','reason':'current-checkpoint MOCHA phonation shards missing or have no signed frames'},separators=(',',':')))
PY
