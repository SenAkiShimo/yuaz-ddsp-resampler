#!/usr/bin/env python3
import argparse, json, math, os, random, re, tempfile, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from .ai_control_training import NativeYuazDDSPExtractor, _resize_time, _resize_freq
from .ai_vocal_controls import AIControlAdapter, save_ai_control_adapter, SPECTRAL_BANDS, AP_BANDS


def _logit(x,eps=1e-4):
 x=np.clip(x,eps,1-eps); return np.log(x)-np.log1p(-x)
def _residual(src,tgt):
 ds=np.clip((tgt['log_spec']-src['log_spec'])/0.72,-1.4,1.4).astype(np.float32)
 da=np.clip((_logit(tgt['ap'])-_logit(src['ap']))/1.35,-1.4,1.4).astype(np.float32)
 dg=np.clip((_logit(tgt['gate'])-_logit(src['gate']))/1.35,-1.4,1.4).astype(np.float32)
 return ds,da,dg

def _align(a,b):
 T=min(a['log_spec'].shape[-1],b['log_spec'].shape[-1])
 out=[]
 for d in (a,b):out.append({k:_resize_time(d[k],T).astype(np.float32) for k in ('log_spec','ap','gate','f0')})
 return out

def _quality(text):
 t=str(text).lower()
 for q,pat in (("breathy",r'(^|[^a-z])breathy([^a-z]|$)'),("modal",r'(^|[^a-z])(neutral|modal)([^a-z]|$)'),("pressed",r'(^|[^a-z])pressed(?!ta)([^a-z]|$)')):
  if re.search(pat,t):return q
 return None


def _phonation_key(text):
 t=Path(str(text)).as_posix().lower()
 t=re.sub(r'(^|[/_. -])(breathy|neutral|modal|flow|pressedta|pressed)(?=($|[/_. -]))',' ',t)
 t=re.sub(r'\.(wav|flac|aiff?|wave)$','',t)
 t=re.sub(r'[^a-z0-9]+',' ',t).strip()
 return t


def prepare_phonation_modes(phonation_root,out_dir,project_root,max_groups=0):
 root=Path(phonation_root).expanduser().resolve();out=Path(out_dir).expanduser().resolve();out.mkdir(parents=True,exist_ok=True);shards=out/'phonation_direct_shards';shards.mkdir(exist_ok=True)
 mp=root/'.yuaz-phonation-modes-osf-manifest.json'
 if not mp.is_file():raise RuntimeError('OSF Phonation Modes manifest missing; run setup-training.command first.')
 manifest=json.loads(mp.read_text());groups=defaultdict(dict)
 for item in manifest.get('files',[]):
  q=item.get('quality') or _quality(item.get('path',''))
  if q not in ('breathy','modal','pressed'):continue
  path=root/item['path']
  if not path.is_file():continue
  key=str(item.get('pair_key') or _phonation_key(item['path']))
  if key:groups[key][q]=path
 triples=[(k,v) for k,v in groups.items() if all(q in v for q in ('breathy','modal','pressed'))]
 if max_groups:triples=triples[:int(max_groups)]
 if not triples:raise RuntimeError('No breathy/modal/pressed OSF phonation triplets recovered from the validated manifest.')
 native=NativeYuazDDSPExtractor(project_root);made=0;errors=[]
 for gi,(key,g) in enumerate(triples):
  try:
   modal=native.features(g['modal'])
   for q,sign in (('breathy',-1.0),('pressed',1.0)):
    tech=native.features(g[q]);src,tgt=_align(modal,tech);T=src['log_spec'].shape[-1];ds,da,dg=_residual(src,tgt)
    voiced=((src['f0']>1)&(tgt['f0']>1)).astype(np.float32)
    controls=np.zeros((2,T),np.float32);controls[0]=sign*voiced.reshape(-1)
    np.savez_compressed(shards/f'{made:05d}-{q}.npz',spectral=np.exp(src['log_spec']).astype(np.float32),ap=src['ap'],gate=src['gate'],f0=src['f0'],controls=controls,target_ds=ds,target_da=da,target_dg=dg,mask_s=voiced,mask_a=voiced,mask_g=voiced,source=np.asarray('OSF-PhonationModes'),group=np.asarray(key))
    made+=1
   if (gi+1)%10==0:print(f'Prepared OSF Phonation Modes groups {gi+1}/{len(triples)}',flush=True)
  except Exception as e:errors.append({'group':key,'error':str(e)})
 if made<4:raise RuntimeError(f'Too few usable OSF phonation shards: {made}; first errors={errors[:3]}')
 meta={'format':1,'source':'OSF Phonation Modes Dataset','controls':['tension','voicing'],'direct_groups':len(triples),'direct_shards':made,'feature_backend':'yuaz-native-ddsp-v1','checkpoint_sha256':native.checkpoint_sha256,'sample_rate':native.sample_rate,'hop':native.hop,'errors':errors,'created_at':time.time()}
 (out/'phonation_modes_dataset.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
 print(f'Prepared OSF tension shards: {made}')
 return shards,meta

# ---- MOCHA readers ----
def _nist_pcm(path):
 raw=Path(path).read_bytes(); hdr=raw[:1024].decode('ascii','ignore')
 def field(name,default=''):
  m=re.search(r'^'+re.escape(name)+r'\s+-\w+\s+(.+)$',hdr,re.M);return m.group(1).strip() if m else default
 nbytes=int(field('sample_n_bytes','2')); bytefmt=field('sample_byte_format','01'); endian='<' if bytefmt in ('01','1') else '>'
 if nbytes!=2:raise ValueError('MOCHA NIST reader currently expects 16-bit PCM')
 x=np.frombuffer(raw[1024:],dtype=endian+'i2').astype(np.float32)/32768.0
 return x,16000

def _est_track(path):
 data=Path(path).read_bytes(); marker=b'EST_Header_End\n'; j=data.find(marker)
 if j<0:raise ValueError('EST header end not found')
 head=data[:j+len(marker)].decode('ascii','ignore'); body=data[j+len(marker):]
 def hv(name,default=''):
  m=re.search(r'^'+re.escape(name)+r'\s+(.+)$',head,re.M);return m.group(1).strip() if m else default
 n=int(hv('NumFrames','0')); c=int(hv('NumChannels','20')); bo=hv('ByteOrder','01');
 candidates=[]
 for endian in (('<' if bo in ('01','1') else '>'),('>' if bo in ('01','1') else '<')):
  arr=np.frombuffer(body,dtype=endian+'f4')
  width=c+2
  if n>0 and arr.size>=n*width:
   a=arr[:n*width].reshape(n,width)
   score=float(np.mean(np.diff(a[:,0])>=-1e-5)) if n>1 else 0
   candidates.append((score,a))
 if not candidates:raise ValueError('EMA dimensions do not match EST header')
 a=max(candidates,key=lambda z:z[0])[1]
 return a[:,0].astype(np.float32),a[:,2:].astype(np.float32),head

def _coil_xy(ch,coil):
 # MOCHA raw EST: x for coils 1-5, y 1-5, x 6-10, y 6-10.
 i=coil-1
 if i<5:return ch[:,i],ch[:,5+i]
 i-=5;return ch[:,10+i],ch[:,15+i]

def _mouth_aperture(ema_path):
 t,ch,_=_est_track(ema_path)
 if ch.shape[1]<20:raise ValueError('MOCHA EMA has fewer than 20 coordinate channels')
 # Recording inventory order: upper incisor, lower incisor, upper lip, lower lip, tongue tip, tongue blade, tongue dorsum, velum, references.
 ulx,uly=_coil_xy(ch,3); llx,lly=_coil_xy(ch,4)
 aperture=np.sqrt((ulx-llx)**2+(uly-lly)**2).astype(np.float32)
 return t,aperture

def _egg_periodicity(lar_path,f0_times,f0_values):
 x,sr=_nist_pcm(lar_path); x=x-np.mean(x); out=np.zeros(len(f0_times),np.float32)
 for i,(t,f) in enumerate(zip(f0_times,f0_values)):
  if f<40:continue
  center=int(t*sr); half=int(0.025*sr); seg=x[max(0,center-half):min(len(x),center+half)]
  if len(seg)<int(.02*sr):continue
  seg=seg-np.mean(seg); denom=float(np.dot(seg,seg)+1e-8); lag=max(2,int(round(sr/f)))
  if lag>=len(seg)//2:continue
  out[i]=float(np.clip(np.dot(seg[:-lag],seg[lag:])/denom,-1,1))
 return out

def _labels(path,times):
 seg=[]
 for line in Path(path).read_text(errors='ignore').splitlines():
  p=line.strip().split()
  if len(p)>=3:
   try:seg.append((float(p[0]),float(p[1]),p[2]))
   except:pass
 out=[];j=0
 for t in times:
  while j+1<len(seg) and t>=seg[j][1]:j+=1
  out.append(seg[j][2] if seg and seg[j][0]<=t<seg[j][1] else 'sil')
 return np.asarray(out)

def _standard_wav(nist_path,tmpdir):
 y,sr=_nist_pcm(nist_path); p=Path(tmpdir)/(Path(nist_path).stem+'.wav');sf.write(p,y,sr,subtype='PCM_16');return p

def prepare_mocha(mocha_root,out_dir,project_root,max_utterances=0):
 root=Path(mocha_root).expanduser().resolve(); out=Path(out_dir).expanduser().resolve();out.mkdir(parents=True,exist_ok=True); cache=out/'mocha_features';cache.mkdir(exist_ok=True)
 native=NativeYuazDDSPExtractor(project_root); records=[];errors=[]; count=0
 wavs=sorted((root/'extracted').rglob('*.wav'))
 if not wavs:raise RuntimeError('No extracted MOCHA .wav files found; run setup-training.command first.')
 with tempfile.TemporaryDirectory(prefix='yuaz-mocha-wav-') as td:
  for wav in wavs:
   stem=wav.with_suffix(''); lar=stem.with_suffix('.lar');ema=stem.with_suffix('.ema');lab=stem.with_suffix('.lab')
   if not (lar.is_file() and ema.is_file() and lab.is_file()):continue
   try:
    std=_standard_wav(wav,td); feat=native.features(std); T=feat['log_spec'].shape[-1]; times=(np.arange(T,dtype=np.float32)*native.hop/native.sample_rate)
    f0=np.asarray(feat['f0']).reshape(-1); f0=_resize_time(f0.reshape(1,-1),T).reshape(-1)
    et,aperture=_mouth_aperture(ema); aperture=np.interp(times,et,aperture,left=aperture[0],right=aperture[-1]).astype(np.float32)
    periodic=_egg_periodicity(lar,times,f0); phones=_labels(lab,times); speaker=wav.name.split('_',1)[0].lower()
    cpath=cache/(wav.stem+'.npz');np.savez_compressed(cpath,log_spec=feat['log_spec'],ap=feat['ap'],gate=feat['gate'],f0=feat['f0'],aperture=aperture,periodic=periodic,phones=phones,speaker=np.asarray(speaker))
    records.append(cpath);count+=1
    if count%25==0:print(f'Extracted MOCHA Yuaz features {count}',flush=True)
    if max_utterances and count>=int(max_utterances):break
   except Exception as e:errors.append({'file':str(wav),'error':str(e)})
 if len(records)<4:raise RuntimeError(f'Too few MOCHA feature records: {len(records)}; first errors={errors[:3]}')
 # Build robust neutral centroids per speaker+phone+3-semitone F0 bin, sampling frames.
 groups=defaultdict(lambda:{'spec':[],'ap':[],'gate':[],'mouth':[],'period':[]})
 for p in records:
  z=np.load(p,allow_pickle=False);sp=str(z['speaker'].item());f0=np.asarray(z['f0']).reshape(-1);phones=z['phones'];
  midi=np.where(f0>20,69+12*np.log2(np.maximum(f0,20)/440),np.nan)
  for i in range(0,len(f0),2):
   if not np.isfinite(midi[i]) or phones[i] in ('sil','pau','#'):continue
   key=(sp,str(phones[i]),int(round(float(midi[i])/3)*3));g=groups[key]
   if len(g['mouth'])>=256:continue
   g['spec'].append(z['log_spec'][:,i]);g['ap'].append(z['ap'][:,i]);g['gate'].append(z['gate'][:,i]);g['mouth'].append(float(z['aperture'][i]));g['period'].append(float(z['periodic'][i]))
 stats={}
 for k,g in groups.items():
  if len(g['mouth'])<12:continue
  stats[k]={'spec':np.median(np.stack(g['spec']),axis=0),'ap':np.median(np.stack(g['ap']),axis=0),'gate':np.median(np.stack(g['gate']),axis=0),'mouth_med':float(np.median(g['mouth'])),'mouth_scale':float(np.percentile(g['mouth'],85)-np.percentile(g['mouth'],15)+1e-6),'period_med':float(np.median(g['period'])),'period_scale':float(np.percentile(g['period'],85)-np.percentile(g['period'],15)+1e-6)}
 psh=out/'phonation_shards';msh=out/'mouth_shards';psh.mkdir(exist_ok=True);msh.mkdir(exist_ok=True); pmade=mmade=0
 for p in records:
  z=np.load(p,allow_pickle=False);T=z['log_spec'].shape[-1];sp=str(z['speaker'].item());f0=np.asarray(z['f0']).reshape(-1);phones=z['phones'];midi=np.where(f0>20,69+12*np.log2(np.maximum(f0,20)/440),np.nan)
  base_spec=np.array(z['log_spec'],copy=True);base_ap=np.array(z['ap'],copy=True);base_gate=np.array(z['gate'],copy=True)
  vc=np.zeros((2,T),np.float32);mc=np.zeros((1,T),np.float32); tds=np.zeros_like(base_spec);tda=np.zeros_like(base_ap);tdg=np.zeros_like(base_gate);vma=np.zeros((1,T),np.float32);mma=np.zeros((1,T),np.float32)
  for i in range(T):
   if not np.isfinite(midi[i]):continue
   key=(sp,str(phones[i]),int(round(float(midi[i])/3)*3)); st=stats.get(key)
   if not st:continue
   mouth=float(np.clip((float(z['aperture'][i])-st['mouth_med'])/(0.5*st['mouth_scale']),-1,1));voi=float(np.clip((float(z['periodic'][i])-st['period_med'])/(0.5*st['period_scale']),-1,1))
   neutral={'log_spec':st['spec'][:,None],'ap':st['ap'][:,None],'gate':st['gate'][:,None]}; obs={'log_spec':z['log_spec'][:,i:i+1],'ap':z['ap'][:,i:i+1],'gate':z['gate'][:,i:i+1]};ds,da,dg=_residual(neutral,obs)
   base_spec[:,i]=st['spec'];base_ap[:,i]=st['ap'];base_gate[:,i]=st['gate']
   if abs(voi)>=0.18:vc[1,i]=voi;tda[:,i]=da[:,0];tdg[:,i]=dg[:,0];vma[0,i]=1
   if abs(mouth)>=0.18:mc[0,i]=mouth;tds[:,i]=ds[:,0];mma[0,i]=1
  if np.count_nonzero(vma)>=16:
   np.savez_compressed(psh/f'mocha-{pmade:05d}.npz',spectral=np.exp(base_spec).astype(np.float32),ap=base_ap,gate=base_gate,f0=z['f0'],controls=vc,target_ds=np.zeros_like(tds),target_da=tda,target_dg=tdg,mask_s=np.zeros_like(vma),mask_a=vma,mask_g=vma,source=np.asarray('MOCHA'));pmade+=1
  if np.count_nonzero(mma)>=16:
   np.savez_compressed(msh/f'mocha-{mmade:05d}.npz',spectral=np.exp(base_spec).astype(np.float32),ap=base_ap,gate=base_gate,f0=z['f0'],controls=mc,target_ds=tds,target_da=np.zeros_like(tda),target_dg=np.zeros_like(tdg),mask_s=mma,mask_a=np.zeros_like(mma),mask_g=np.zeros_like(mma),source=np.asarray('MOCHA'));mmade+=1
 meta={'format':1,'source':'MOCHA-TIMIT CSTR','feature_backend':'yuaz-native-ddsp-v1','checkpoint_sha256':native.checkpoint_sha256,'sample_rate':native.sample_rate,'hop':native.hop,'feature_records':len(records),'phonation_shards':pmade,'mouth_shards':mmade,'errors':errors,'created_at':time.time()};(out/'mocha_dataset.json').write_text(json.dumps(meta,indent=2)+'\n')
 print(f'Prepared MOCHA: voicing shards={pmade}, mouth shards={mmade}')
 return psh,msh,meta


def _tensor(a):return torch.from_numpy(np.asarray(a,dtype=np.float32)).unsqueeze(0)
def _windows(path,control_names,window=160,max_windows=5):
 z=np.load(path,allow_pickle=False);T=z['spectral'].shape[-1];w=min(window,T);starts=[0] if T<=w else np.linspace(0,T-w,min(max_windows,max(1,T//w))).astype(int).tolist()
 for st in starts:
  en=st+w;b={k:_tensor(z[k][:,st:en]) for k in ('spectral','ap','gate','f0','target_ds','target_da','target_dg','mask_s','mask_a','mask_g')};c=np.asarray(z['controls'][:,st:en],np.float32);controls={n:torch.from_numpy(c[i]).view(1,1,-1) for i,n in enumerate(control_names)};yield b,controls

def train_pack(shard_dirs,output,control_names,control_modes,output_scopes,metadata,epochs=12,lr=2e-4,seed=283):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);files=[]
 for d in shard_dirs:files+=sorted(Path(d).glob('*.npz'))
 if len(files)<4:raise RuntimeError(f'Too few training shards: {len(files)}')
 random.Random(seed).shuffle(files);nval=max(1,int(round(len(files)*.12)));val=files[:nval];train=files[nval:]
 model=AIControlAdapter(control_names=tuple(control_names),control_modes=tuple(control_modes),output_scopes=tuple(output_scopes));opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4);best=float('inf')
 def run(fs,tr):
  model.train(tr);tot=n=0
  for p in fs:
   for b,c in _windows(p,control_names,160,5 if tr else 2):
    ds,da,dg=model.predict_residuals(b['spectral'],b['ap'],b['gate'],b['f0'],c);loss=0
    for pred,tgt,mk in ((ds,b['target_ds'],b['mask_s']),(da,b['target_da'],b['mask_a']),(dg,b['target_dg'],b['mask_g'])):
     mask=mk.expand_as(pred);raw=F.smooth_l1_loss(torch.tanh(pred),torch.clamp(tgt,-1,1),reduction='none');loss=loss+torch.sum(raw*mask)/torch.clamp(mask.sum(),min=1)
    zeros={k:torch.zeros_like(v) for k,v in c.items()};zds,zda,zdg=model.predict_residuals(b['spectral'],b['ap'],b['gate'],b['f0'],zeros);loss=loss+.12*(zds.abs().mean()+zda.abs().mean()+zdg.abs().mean())
    if tr:opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
    tot+=float(loss.detach());n+=1
  return tot/max(1,n)
 hist=[]
 for e in range(epochs):
  tr=run(train,True)
  with torch.inference_mode():va=run(val,False)
  hist.append({'epoch':e+1,'train':tr,'val':va});print(f"{'+'.join(control_names)} Epoch {e+1}/{epochs}: train={tr:.6f} val={va:.6f}",flush=True)
  if va<=best:
   best=va;m=dict(metadata);m.update({'feature_backend':'yuaz-native-ddsp-v1','controls':list(control_names),'control_modes':list(control_modes),'output_scopes':list(output_scopes),'best_validation_loss':best,'epoch':e+1,'created_at':time.time()});save_ai_control_adapter(output,model,m)
 Path(str(output)+'.json').write_text(json.dumps({'best_validation_loss':best,'history':hist},indent=2)+'\n');print('Trained:',Path(output).resolve())


def main():
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True)
 p=sub.add_parser('prepare-phonation');p.add_argument('root');p.add_argument('out');p.add_argument('--project-root',required=True);p.add_argument('--max-groups',type=int,default=0)
 m=sub.add_parser('prepare-mocha');m.add_argument('root');m.add_argument('out');m.add_argument('--project-root',required=True);m.add_argument('--max-utterances',type=int,default=0)
 t=sub.add_parser('train-phonation');t.add_argument('direct_shards');t.add_argument('mocha_shards');t.add_argument('output');t.add_argument('--checkpoint-sha',required=True);t.add_argument('--epochs',type=int,default=12)
 o=sub.add_parser('train-mouth');o.add_argument('shards');o.add_argument('output');o.add_argument('--checkpoint-sha',required=True);o.add_argument('--epochs',type=int,default=12)
 a=ap.parse_args()
 if a.cmd=='prepare-phonation':prepare_phonation_modes(a.root,a.out,a.project_root,a.max_groups)
 elif a.cmd=='prepare-mocha':prepare_mocha(a.root,a.out,a.project_root,a.max_utterances)
 elif a.cmd=='train-phonation':train_pack([a.direct_shards,a.mocha_shards],a.output,('tension','voicing'),('signed','signed'),('spectral','ap','gate'),{'training_sources':['OSF Phonation Modes Dataset','MOCHA-TIMIT'],'checkpoint_sha256':a.checkpoint_sha,'target_design':'OSF direct breathy/modal/pressed singing tension residuals + MOCHA laryngograph-periodicity voicing residuals; orthogonal axis masks'},a.epochs)
 else:train_pack([a.shards],a.output,('mouth',),('signed',),('spectral',),{'training_sources':['MOCHA-TIMIT EMA'],'checkpoint_sha256':a.checkpoint_sha,'target_design':'within-speaker/phone/F0-bin lip-aperture residual; spectral-only'},a.epochs)
if __name__=='__main__':main()
