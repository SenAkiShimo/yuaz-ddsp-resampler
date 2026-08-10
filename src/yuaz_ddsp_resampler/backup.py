#!/usr/bin/env python3

import argparse

import hashlib

import json

import os

import shutil

import tarfile

import time

from pathlib import Path


PREDECESSOR_STATE = '.yuaz-alpha8-rc3-1'

CURRENT_STATE = '.yuaz-alpha8-rc3-2'

PREDECESSOR_PROJECT = 'yuaz-ddsp-resampler-v0.2.7-alpha.8-rc.3.1'

CRITICAL_HASH_NAMES = {

    'adapter.pt', 'timbre_profiles.pt', 'highband_profiles_v3.json',

    'profile.json', 'training.json', 'loudness.json', 'manifest.json',

    'fidelity_refiner.pt', 'fidelity_training.json',

}

EXCLUDED_CACHE_DIRS = {'cache', 'highband_cache_v3', 'highband_cache', '__pycache__'}


def _sha256(path, chunk=1024 * 1024):

    h = hashlib.sha256()

    with Path(path).open('rb') as f:

        while True:

            b = f.read(chunk)

            if not b:

                break

            h.update(b)

    return h.hexdigest()


def _safe_name(text):

    keep = []

    for ch in str(text):

        if ch.isalnum() or ch in ('-', '_', '.', ' '):

            keep.append(ch)

        else:

            keep.append('_')

    value = ''.join(keep).strip().replace(' ', '_')

    return value or 'voicebank'


def _tar_filter(ti):

    parts = Path(ti.name).parts

    if any(part in EXCLUDED_CACHE_DIRS for part in parts):

        return None

    return ti


def _state_inventory(state):

    state = Path(state)

    info = {'exists': state.exists(), 'path': str(state), 'critical_files': {}, 'cache_summary': {}}

    if not state.exists():

        return info

    for p in state.rglob('*'):

        if not p.is_file():

            continue

        rel = p.relative_to(state).as_posix()

        if any(part in EXCLUDED_CACHE_DIRS for part in p.relative_to(state).parts):

            top = p.relative_to(state).parts[0]

            entry = info['cache_summary'].setdefault(top, {'files': 0, 'bytes': 0})

            entry['files'] += 1

            entry['bytes'] += p.stat().st_size

            continue

        if p.name in CRITICAL_HASH_NAMES or rel.startswith('articulation/'):

            try:

                info['critical_files'][rel] = {'bytes': p.stat().st_size, 'sha256': _sha256(p)}

            except Exception as exc:

                info['critical_files'][rel] = {'bytes': p.stat().st_size, 'sha256_error': str(exc)}

    return info


def _write_layout_snapshot(bank, dest):

    bank = Path(bank)

    members = []

    prefix = bank / 'prefix.map'

    if prefix.exists():

        members.append(prefix)

    members.extend(sorted(bank.rglob('oto.ini')))

    character = bank / 'character.txt'

    if character.exists():

        members.append(character)

    if not members:

        return None

    out = dest / 'voicebank-layout.tar.gz'

    with tarfile.open(out, 'w:gz') as tf:

        for p in members:

            tf.add(p, arcname=p.relative_to(bank).as_posix())

    return out


def _registry_snapshot(project_root, bank, bank_id, dest):

    project_root = Path(project_root).resolve()

    bank = Path(bank).resolve()

    candidates = [project_root.parent / PREDECESSOR_PROJECT / 'voicebank_registry.json']

    current_registry = project_root / 'voicebank_registry.json'

    candidates.append(current_registry)

    payload = {'voicebank_root': str(bank), 'voicebank_id': bank_id, 'registries': []}

    for path in candidates:

        entry = {'path': str(path), 'exists': path.exists(), 'records': {}}

        if path.exists():

            try:

                data = json.loads(path.read_text(encoding='utf-8'))

                samples = data.get('samples', {}) if isinstance(data, dict) else {}

                for key, value in samples.items():

                    if not isinstance(value, dict):

                        continue

                    root_match = False

                    try:

                        root_match = Path(value.get('voicebank_root', '')).expanduser().resolve() == bank

                    except Exception:

                        pass

                    if root_match or value.get('voicebank_id') == bank_id:

                        entry['records'][key] = value

            except Exception as exc:

                entry['error'] = str(exc)

        payload['registries'].append(entry)

    out = dest / 'registry-snapshot.json'

    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    return out


def create_backup(project_root, voicebank, reason='manual'):

    project_root = Path(project_root).resolve()

    bank = Path(voicebank).expanduser().resolve()

    if not bank.is_dir():

        raise RuntimeError(f'Voicebank folder not found: {bank}')

    from .voicebank import voicebank_id

    bank_id = voicebank_id(bank)

    backup_base = Path.home() / 'Documents' / 'Yuaz-DDSP-Backups' / _safe_name(bank.name)

    backup_base.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime('%Y%m%d-%H%M%S')

    dest = backup_base / f'{stamp}-before-rc3-2-{_safe_name(reason)}'

    suffix = 1

    while dest.exists():

        dest = backup_base / f'{stamp}-before-rc3-2-{_safe_name(reason)}-{suffix}'

        suffix += 1

    dest.mkdir(parents=True, exist_ok=False)


    predecessor = bank / PREDECESSOR_STATE

    current = bank / CURRENT_STATE

    states = []

    for label, state in (('rc3-1', predecessor), ('rc3-2', current)):

        inventory = _state_inventory(state)

        inventory['label'] = label

        states.append(inventory)

        if state.exists():

            archive = dest / f'{label}-runtime-state.tar.gz'

            with tarfile.open(archive, 'w:gz') as tf:

                tf.add(state, arcname=state.name, filter=_tar_filter)

            if archive.stat().st_size <= 0:

                raise RuntimeError(f'Backup archive is empty: {archive}')


    _registry_snapshot(project_root, bank, bank_id, dest)

    _write_layout_snapshot(bank, dest)


    predecessor_project = project_root.parent / PREDECESSOR_PROJECT

    project_meta = {

        'rc3_1_project_path': str(predecessor_project),

        'rc3_1_project_exists': predecessor_project.exists(),

        'rc3_1_version': None,

        'rc3_1_config': None,

    }

    if predecessor_project.exists():

        version = predecessor_project / 'VERSION'

        config = predecessor_project / 'config.json'

        if version.exists():

            project_meta['rc3_1_version'] = version.read_text(encoding='utf-8', errors='replace').strip()

        if config.exists():

            try:

                project_meta['rc3_1_config'] = json.loads(config.read_text(encoding='utf-8'))

            except Exception:

                project_meta['rc3_1_config'] = config.read_text(encoding='utf-8', errors='replace')


    manifest = {

        'format': 1,

        'created_at': time.time(),

        'created_local': time.strftime('%Y-%m-%d %H:%M:%S'),

        'reason': reason,

        'voicebank_root': str(bank),

        'voicebank_name': bank.name,

        'voicebank_id': bank_id,

        'project_root': str(project_root),

        'states': states,

        'project_metadata': project_meta,

        'cache_policy': 'analysis caches excluded; runtime/training-critical state archived exactly',

    }

    manifest_path = dest / 'backup-manifest.json'

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')

    if not manifest_path.exists() or manifest_path.stat().st_size < 100:

        raise RuntimeError('Backup manifest validation failed')

    (backup_base / 'LATEST_BACKUP.txt').write_text(str(dest) + '\n', encoding='utf-8')

    return dest, manifest


def _extract_state(archive, bank, expected_dir):

    archive = Path(archive)

    bank = Path(bank)

    if not archive.exists():

        raise RuntimeError(f'Archive not found: {archive}')

    with tarfile.open(archive, 'r:gz') as tf:

        top = {Path(m.name).parts[0] for m in tf.getmembers() if m.name and Path(m.name).parts}

        if expected_dir not in top:

            raise RuntimeError(f'Archive does not contain {expected_dir}: {archive}')

        tf.extractall(bank, filter="data")


def restore_backup(backup_dir, voicebank=None, target='rc3-1'):

    backup_dir = Path(backup_dir).expanduser().resolve()

    manifest_path = backup_dir / 'backup-manifest.json'

    if not manifest_path.exists():

        raise RuntimeError(f'Not a Yuaz backup folder: {backup_dir}')

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    bank = Path(voicebank or manifest['voicebank_root']).expanduser().resolve()

    if not bank.is_dir():

        raise RuntimeError(f'Voicebank folder not found: {bank}')

    if target not in ('rc3-1', 'rc3-2'):

        raise RuntimeError('target must be rc3-1 or rc3-2')

    state_name = PREDECESSOR_STATE if target == 'rc3-1' else CURRENT_STATE

    archive = backup_dir / f'{target}-runtime-state.tar.gz'

    if not archive.exists():

        raise RuntimeError(f'Backup does not contain {target} state')

    existing = bank / state_name

    if existing.exists():

        moved = bank / f'{state_name}-pre-restore-{time.strftime("%Y%m%d-%H%M%S")}'

        existing.rename(moved)

        print(f'Existing state moved to: {moved}')

    _extract_state(archive, bank, state_name)


    if target == 'rc3-1':

        snap = backup_dir / 'registry-snapshot.json'

        if snap.exists():

            data = json.loads(snap.read_text(encoding='utf-8'))

            for reg in data.get('registries', []):

                path = Path(reg.get('path', '')).expanduser()

                if PREDECESSOR_PROJECT not in str(path):

                    continue

                records = reg.get('records', {})

                if not records:

                    continue

                try:

                    current = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'format': 4, 'samples': {}}

                except Exception:

                    current = {'format': 4, 'samples': {}}

                samples = current.setdefault('samples', {})

                bank_id = data.get('voicebank_id')

                for key in list(samples):

                    value = samples.get(key)

                    if isinstance(value, dict) and (value.get('voicebank_id') == bank_id or value.get('voicebank_root') == str(bank)):

                        samples.pop(key, None)

                samples.update(records)

                path.parent.mkdir(parents=True, exist_ok=True)

                path.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding='utf-8')

                print(f'Restored previous registry records: {path}')

    return bank / state_name


def _latest_backup_for(bank):

    base = Path.home() / 'Documents' / 'Yuaz-DDSP-Backups' / _safe_name(Path(bank).name)

    marker = base / 'LATEST_BACKUP.txt'

    if marker.exists():

        p = Path(marker.read_text(encoding='utf-8').strip()).expanduser()

        if p.exists():

            return p

    dirs = sorted([p for p in base.glob('*') if p.is_dir()], reverse=True) if base.exists() else []

    return dirs[0] if dirs else None


def main():

    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(dest='command', required=True)

    p_backup = sub.add_parser('backup')

    p_backup.add_argument('voicebank')

    p_backup.add_argument('--project-root', required=True)

    p_backup.add_argument('--reason', default='manual')

    p_restore = sub.add_parser('restore')

    p_restore.add_argument('voicebank')

    p_restore.add_argument('--backup-dir')

    p_restore.add_argument('--target', choices=('rc3-1', 'rc3-2'), default='rc3-1')

    args = parser.parse_args()

    if args.command == 'backup':

        dest, manifest = create_backup(args.project_root, args.voicebank, reason=args.reason)

        pred = next((x for x in manifest['states'] if x['label'] == 'rc3-1'), {})

        cur = next((x for x in manifest['states'] if x['label'] == 'rc3-2'), {})

        print(f'Backup completed successfully: {dest}')

        print(f'  Previous compatible state backed up: {bool(pred.get("exists"))}')

        print(f'  Current state backed up: {bool(cur.get("exists"))}')

    else:

        backup_dir = Path(args.backup_dir).expanduser() if args.backup_dir else _latest_backup_for(args.voicebank)

        if backup_dir is None:

            raise SystemExit('No backup was found for this voicebank.')

        restored = restore_backup(backup_dir, args.voicebank, target=args.target)

        print(f'Restored {args.target} runtime state: {restored}')


if __name__ == '__main__':

    main()

