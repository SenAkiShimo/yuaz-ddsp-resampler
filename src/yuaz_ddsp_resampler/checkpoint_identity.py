#!/usr/bin/env python3
import hashlib
from pathlib import Path

import torch


def file_sha256(path, chunk=8 * 1024 * 1024):
    path = Path(path).expanduser().resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_checkpoint_raw(path):
    path = Path(path).expanduser().resolve()
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        return torch.load(path, map_location='cpu', weights_only=False)


def checkpoint_identity(path):
    path = Path(path).expanduser().resolve()
    raw = load_checkpoint_raw(path)
    runtime_sha = file_sha256(path)
    source_sha = runtime_sha
    source_name = path.name
    source_step = None
    runtime_format = None
    if isinstance(raw, dict):
        runtime_format = raw.get('format')
        source_sha = str(raw.get('source_checkpoint_sha256') or runtime_sha)
        source_name = str(raw.get('source_checkpoint') or path.name)
        source_step = raw.get('source_step', raw.get('step'))
    return {
        'path': str(path),
        'runtime_sha256': runtime_sha,
        'source_checkpoint_sha256': source_sha,
        'source_checkpoint': source_name,
        'source_step': source_step,
        'runtime_format': runtime_format,
    }


def checkpoint_identity_sha(path):
    return checkpoint_identity(path)['source_checkpoint_sha256']
