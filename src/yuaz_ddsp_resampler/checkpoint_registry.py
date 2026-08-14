#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch

from .checkpoint_identity import checkpoint_identity, file_sha256, load_checkpoint_raw
from .core import build_modules, load_config as load_yuaz_config, normalize_checkpoint_state

MODEL_PREFIXES = ('encoder.', 'ddsp_decoder.', 'rvq.')
EXPECTED_COMPONENTS = ('encoder', 'ddsp_decoder', 'rvq')


def default_models_root():
    return Path.home() / 'Library' / 'Application Support' / 'YuazDDSP' / 'models'


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def _model_dict(raw):
    if not isinstance(raw, dict):
        return raw
    for key in ('model', 'state_dict', 'model_state_dict'):
        candidate = raw.get(key)
        if isinstance(candidate, dict) and candidate:
            return candidate
    return raw


def extract_runtime_state(raw):
    state = normalize_checkpoint_state(raw)
    return {k: v.cpu() for k, v in state.items() if torch.is_tensor(v) and k.startswith(MODEL_PREFIXES)}


def probe_checkpoint(path, yuaz_repo):
    path = Path(path).expanduser().resolve()
    yuaz_repo = Path(yuaz_repo).expanduser().resolve()
    raw = load_checkpoint_raw(path)
    ident = checkpoint_identity(path)
    state = normalize_checkpoint_state(raw)
    cfg = load_yuaz_config(yuaz_repo)
    encoder, decoder, rvq, sr, hop = build_modules(yuaz_repo, cfg, torch.device('cpu'))
    targets = {'encoder': encoder.state_dict(), 'ddsp_decoder': decoder.state_dict(), 'rvq': rvq.state_dict()}
    components = {}
    expected = set()
    for prefix, target in targets.items():
        total = sum(v.numel() for v in target.values())
        loaded = 0
        matched = 0
        missing = []
        mismatches = []
        for local, tensor in target.items():
            full = prefix + '.' + local
            expected.add(full)
            value = state.get(full)
            if value is None:
                missing.append(local)
            elif tuple(value.shape) != tuple(tensor.shape):
                mismatches.append({'key': local, 'checkpoint': list(value.shape), 'expected': list(tensor.shape)})
            else:
                matched += 1
                loaded += tensor.numel()
        components[prefix] = {
            'coverage': loaded / max(1, total),
            'matched_tensors': matched,
            'target_tensors': len(target),
            'missing_count': len(missing),
            'mismatch_count': len(mismatches),
            'missing_first': missing[:12],
            'mismatch_first': mismatches[:12],
        }
    extra = sorted(set(state) - expected)
    compatible = (
        components['encoder']['coverage'] >= 0.95 and
        components['ddsp_decoder']['coverage'] >= 0.80 and
        components['rvq']['coverage'] >= 0.80
    )
    return {
        **ident,
        'size_bytes': path.stat().st_size,
        'tensor_count': len(state),
        'source_step': ident.get('source_step') if ident.get('source_step') is not None else (raw.get('step') if isinstance(raw, dict) else None),
        'components': components,
        'extra_tensor_count': len(extra),
        'extra_first': extra[:30],
        'model_sample_rate': int(sr),
        'model_hop': int(hop),
        'compatible': bool(compatible),
    }


def import_checkpoint(path, yuaz_repo, models_root=None):
    path = Path(path).expanduser().resolve()
    yuaz_repo = Path(yuaz_repo).expanduser().resolve()
    models_root = Path(models_root or default_models_root()).expanduser().resolve()
    probe = probe_checkpoint(path, yuaz_repo)
    if not probe['compatible']:
        raise RuntimeError('Checkpoint is not compatible with the current Yuaz Encoder/DDSP/RVQ architecture.')
    raw = load_checkpoint_raw(path)
    selected = extract_runtime_state(raw)
    source_sha = probe['source_checkpoint_sha256']
    model_id = source_sha[:16]
    model_dir = models_root / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = model_dir / 'runtime.pt'
    payload = {
        'format': 'yuaz-ddsp-resampler-runtime-checkpoint-v1',
        'source_checkpoint': probe.get('source_checkpoint') or path.name,
        'source_checkpoint_sha256': source_sha,
        'source_step': probe.get('source_step'),
        'model': selected,
    }
    tmp = model_dir / f'.runtime.pt.tmp-{os.getpid()}'
    torch.save(payload, tmp)
    os.replace(tmp, runtime_path)
    runtime_sha = file_sha256(runtime_path)
    metadata = {
        'format': 1,
        'model_id': model_id,
        'source_checkpoint': payload['source_checkpoint'],
        'source_checkpoint_sha256': source_sha,
        'source_step': payload['source_step'],
        'source_input_path': str(path),
        'source_input_size_bytes': int(path.stat().st_size),
        'source_input_runtime_sha256': probe['runtime_sha256'],
        'runtime_path': str(runtime_path),
        'runtime_sha256': runtime_sha,
        'runtime_size_bytes': int(runtime_path.stat().st_size),
        'components': probe['components'],
        'compatible': True,
        'yuaz_repo': str(yuaz_repo),
        'imported_at': time.time(),
    }
    _atomic_json(model_dir / 'metadata.json', metadata)
    registry_path = models_root / 'registry.json'
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text(encoding='utf-8'))
        except Exception:
            registry = {'format': 1, 'models': {}}
    else:
        registry = {'format': 1, 'models': {}}
    registry.setdefault('models', {})[model_id] = metadata
    registry['updated_at'] = time.time()
    _atomic_json(registry_path, registry)
    return metadata


def load_registry(models_root=None):
    root = Path(models_root or default_models_root()).expanduser().resolve()
    p = root / 'registry.json'
    if not p.exists():
        return {'format': 1, 'models': {}}
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {'format': 1, 'models': {}}
    if not isinstance(data.get('models'), dict):
        data['models'] = {}
    return data


def select_model(project_root, model_id, models_root=None):
    project_root = Path(project_root).expanduser().resolve()
    cfg_path = project_root / 'config.json'
    if not cfg_path.is_file():
        raise RuntimeError('config.json not found; run configure-macos.command first.')
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    registry = load_registry(models_root)
    models = registry.get('models') or {}
    matches = [k for k in models if k == model_id or k.startswith(model_id)]
    if len(matches) != 1:
        raise RuntimeError(f'Model selector {model_id!r} matched {len(matches)} entries.')
    meta = models[matches[0]]
    runtime = Path(meta['runtime_path']).expanduser().resolve()
    if not runtime.is_file():
        raise RuntimeError(f'Registered runtime checkpoint is missing: {runtime}')
    if file_sha256(runtime) != meta.get('runtime_sha256'):
        raise RuntimeError('Registered runtime checkpoint SHA-256 changed; refusing selection.')
    cfg['checkpoint'] = str(runtime)
    cfg['base_checkpoint_model_id'] = meta['model_id']
    cfg['base_checkpoint_source_name'] = meta.get('source_checkpoint')
    cfg['base_checkpoint_sha256'] = meta['source_checkpoint_sha256']
    cfg['base_checkpoint_runtime_sha256'] = meta['runtime_sha256']
    cfg['base_checkpoint_step'] = meta.get('source_step')
    cfg['base_checkpoint_registry'] = str(Path(models_root or default_models_root()).expanduser().resolve() / 'registry.json')
    _atomic_json(cfg_path, cfg)
    _atomic_json(default_models_root() / 'ACTIVE.json', {
        'format': 1, 'model_id': meta['model_id'], 'source_checkpoint_sha256': meta['source_checkpoint_sha256'],
        'runtime_path': str(runtime), 'selected_at': time.time(), 'project_root': str(project_root),
    })
    return meta


def _print_probe(data):
    print('========== YUAZ CHECKPOINT PROBE ==========')
    print('checkpoint      :', data['path'])
    print('size MiB        :', round(data['size_bytes']/1024/1024, 2))
    print('source SHA256   :', data['source_checkpoint_sha256'])
    print('runtime SHA256  :', data['runtime_sha256'])
    print('source step     :', data.get('source_step'))
    print('tensor count    :', data['tensor_count'])
    print()
    for name in EXPECTED_COMPONENTS:
        c=data['components'][name]
        print(f'{name:14s}: {c["coverage"]:.4%}  tensors {c["matched_tensors"]}/{c["target_tensors"]}  missing={c["missing_count"]} mismatch={c["mismatch_count"]}')
    print('extra tensors   :', data['extra_tensor_count'])
    print('COMPATIBLE      :', 'YES' if data['compatible'] else 'NO')
    print('===========================================')


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='cmd', required=True)
    a=sub.add_parser('probe'); a.add_argument('checkpoint'); a.add_argument('--yuaz-repo', required=True)
    a=sub.add_parser('import'); a.add_argument('checkpoint'); a.add_argument('--yuaz-repo', required=True); a.add_argument('--models-root')
    a=sub.add_parser('list'); a.add_argument('--models-root')
    a=sub.add_parser('select'); a.add_argument('model_id'); a.add_argument('--project-root', required=True); a.add_argument('--models-root')
    a=p.parse_args()
    if a.cmd=='probe':
        data=probe_checkpoint(a.checkpoint,a.yuaz_repo); _print_probe(data); raise SystemExit(0 if data['compatible'] else 2)
    if a.cmd=='import':
        meta=import_checkpoint(a.checkpoint,a.yuaz_repo,a.models_root)
        print('Imported model:', meta['model_id'])
        print('Source SHA256 :', meta['source_checkpoint_sha256'])
        print('Source step   :', meta.get('source_step'))
        print('Runtime       :', meta['runtime_path'])
        print('Runtime MiB   :', round(meta['runtime_size_bytes']/1024/1024,2))
        print('Runtime SHA256:', meta['runtime_sha256'])
        return
    if a.cmd=='list':
        models=load_registry(a.models_root).get('models') or {}
        if not models:
            print('No imported Yuaz base models.')
            return
        for key,meta in sorted(models.items(), key=lambda kv: (kv[1].get('source_step') or 0, kv[0])):
            print(f'{key}  step={meta.get("source_step")}  source={meta.get("source_checkpoint")}  runtime={round((meta.get("runtime_size_bytes") or 0)/1024/1024,2)} MiB')
        return
    if a.cmd=='select':
        meta=select_model(a.project_root,a.model_id,a.models_root)
        print('Selected Yuaz base model:', meta['model_id'])
        print('Source:', meta.get('source_checkpoint'), 'step', meta.get('source_step'))
        print('Source SHA256:', meta['source_checkpoint_sha256'])
        print('Existing ai.14 voicebank generations are NOT modified. A state trained under another SHA will be rejected until a matching Deep generation is active.')

if __name__=='__main__':
    main()
