#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import tarfile
import time
from pathlib import Path

from .state import (
    ACTIVE_FILE, LEGACY_STATE, STATE_CONTAINER, PREVIOUS_028AI12_STATE_CONTAINER, PREVIOUS_028AI7_STATE_CONTAINER, PREVIOUS_028AI5_STATE_CONTAINER, PREVIOUS_028AI4_STATE_CONTAINER, PREVIOUS_028AI3_STATE_CONTAINER, PREVIOUS_028AI2_STATE_CONTAINER, PREVIOUS_028AI1_STATE_CONTAINER, PREVIOUS_028_STATE_CONTAINER, PREDECESSOR_AI_STATE_CONTAINER, STABLE_STATE_CONTAINER, atomic_write_json,
    begin_generation, commit_generation, resolve_active_state, resolve_ai_state, resolve_previous_028ai12_state, resolve_previous_028ai7_state, resolve_previous_028ai5_state, resolve_previous_028ai4_state, resolve_previous_028ai3_state, resolve_previous_028ai2_state, resolve_previous_028ai1_state, resolve_previous_028_state, resolve_predecessor_ai_state, resolve_stable_state,
)

CRITICAL_HASH_NAMES = {
    "adapter.pt", "timbre_profiles.pt", "highband_profiles_v3.json",
    "profile.json", "training.json", "loudness.json", "manifest.json",
    "fidelity_refiner.pt", "fidelity_training.json", "clarity_calibration.json",
    "ai_control_adapter.pt", "ai_control_training.json", "ai_gender_adapter.pt", "ai_gender_training.json",
    "ai_phonation_adapter.pt", "ai_phonation_training.json", "ai_mouth_adapter.pt", "ai_mouth_training.json",
    "state_fingerprint.json", "runtime_registry.json",
}
EXCLUDED_CACHE_DIRS = {"cache", "highband_cache_v3", "highband_cache_v2", "highband_cache", "__pycache__"}


def _sha256(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _safe_name(text):
    value = "".join(ch if (ch.isalnum() or ch in "-_. ") else "_" for ch in str(text)).strip().replace(" ", "_")
    return value or "voicebank"


def _tar_filter(info):
    parts = Path(info.name).parts
    if any(part in EXCLUDED_CACHE_DIRS for part in parts):
        return None
    return info


def _inventory(state):
    state = Path(state)
    result = {"exists": state.exists(), "path": str(state), "critical_files": {}, "cache_summary": {}}
    if not state.exists():
        return result
    for p in state.rglob("*"):
        if p.is_symlink():
            if p.name in EXCLUDED_CACHE_DIRS:
                result["cache_summary"][p.name] = {"linked_to": str(p.resolve())}
            continue
        if not p.is_file():
            continue
        rel = p.relative_to(state).as_posix()
        if any(part in EXCLUDED_CACHE_DIRS for part in p.relative_to(state).parts):
            top = p.relative_to(state).parts[0]
            item = result["cache_summary"].setdefault(top, {"files": 0, "bytes": 0})
            item["files"] += 1
            item["bytes"] += p.stat().st_size
            continue
        if p.name in CRITICAL_HASH_NAMES or rel.startswith("articulation/"):
            result["critical_files"][rel] = {"bytes": p.stat().st_size, "sha256": _sha256(p)}
    return result


def _write_layout_snapshot(bank, dest):
    bank = Path(bank)
    members = []
    for p in [bank / "prefix.map", bank / "character.txt"]:
        if p.exists():
            members.append(p)
    members.extend(sorted(bank.rglob("oto.ini")))
    if not members:
        return None
    out = dest / "voicebank-layout.tar.gz"
    with tarfile.open(out, "w:gz") as tf:
        for p in members:
            tf.add(p, arcname=p.relative_to(bank).as_posix())
    return out


def create_backup(project_root, voicebank, reason="manual"):
    project_root = Path(project_root).resolve()
    bank = Path(voicebank).expanduser().resolve()
    if not bank.is_dir():
        raise RuntimeError(f"Voicebank folder not found: {bank}")
    ai_active, ai_info = resolve_ai_state(bank, verify=True)
    previous_028ai12_active, previous_028ai12_info = resolve_previous_028ai12_state(bank, verify=True)
    previous_028ai7_active, previous_028ai7_info = resolve_previous_028ai7_state(bank, verify=True)
    previous_028ai5_active, previous_028ai5_info = resolve_previous_028ai5_state(bank, verify=True)
    previous_028ai4_active, previous_028ai4_info = resolve_previous_028ai4_state(bank, verify=True)
    previous_028ai3_active, previous_028ai3_info = resolve_previous_028ai3_state(bank, verify=True)
    previous_028ai2_active, previous_028ai2_info = resolve_previous_028ai2_state(bank, verify=True)
    previous_028ai1_active, previous_028ai1_info = resolve_previous_028ai1_state(bank, verify=True)
    previous_028_active, previous_028_info = resolve_previous_028_state(bank, verify=True)
    predecessor_active, predecessor_info = resolve_predecessor_ai_state(bank, verify=True)
    stable_active, stable_info = resolve_stable_state(bank, verify=True)
    legacy = bank / LEGACY_STATE
    base = Path.home() / "Documents" / "Yuaz-DDSP-Backups" / _safe_name(bank.name)
    base.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = base / f"{stamp}-before-0.2.8ai.13-{_safe_name(reason)}"
    suffix = 1
    while dest.exists():
        dest = base / f"{stamp}-before-0.2.8ai.13-{_safe_name(reason)}-{suffix}"
        suffix += 1
    dest.mkdir(parents=True, exist_ok=False)

    states = []
    def archive_state(label, state, info, archive_name, container_name, include_caches=False):
        if state is None:
            states.append({"label": label, "exists": False, "path": None, "critical_files": {}, "cache_summary": {}})
            return
        inv = _inventory(state); inv["label"] = label; inv["generation"] = info.get("generation")
        states.append(inv)
        with tarfile.open(dest / archive_name, "w:gz") as tf:
            tf.add(state, arcname=label + "-state", filter=None if include_caches else _tar_filter)
        active_file = bank / container_name / ACTIVE_FILE
        if active_file.exists():
            shutil.copy2(active_file, dest / f"{label}-ACTIVE.json")

    archive_state("0.2.8ai.13", ai_active, ai_info, "0.2.8ai.13-runtime-state.tar.gz", STATE_CONTAINER, include_caches=False)
    archive_state("0.2.8ai.12", previous_028ai12_active, previous_028ai12_info, "0.2.8ai.12-active-critical.tar.gz", PREVIOUS_028AI12_STATE_CONTAINER, include_caches=False)
    archive_state("0.2.8ai.7", previous_028ai7_active, previous_028ai7_info, "0.2.8ai.7-active-critical.tar.gz", PREVIOUS_028AI7_STATE_CONTAINER, include_caches=False)
    archive_state("0.2.8ai.5", previous_028ai5_active, previous_028ai5_info, "0.2.8ai.5-active-critical.tar.gz", PREVIOUS_028AI5_STATE_CONTAINER, include_caches=False)
    archive_state("0.2.8ai.4", previous_028ai4_active, previous_028ai4_info, "0.2.8ai.4-active-critical.tar.gz", PREVIOUS_028AI4_STATE_CONTAINER, include_caches=False)
    archive_state("0.2.8ai.3", previous_028ai3_active, previous_028ai3_info, "0.2.8ai.3-active-critical.tar.gz", PREVIOUS_028AI3_STATE_CONTAINER, include_caches=False)
    archive_state("0.2.8ai.2", previous_028ai2_active, previous_028ai2_info, "0.2.8ai.2-active-critical.tar.gz", PREVIOUS_028AI2_STATE_CONTAINER, include_caches=False)
    archive_state("0.2.8ai.1", previous_028ai1_active, previous_028ai1_info, "0.2.8ai.1-active-critical.tar.gz", PREVIOUS_028AI1_STATE_CONTAINER, include_caches=False)
    archive_state("0.2.8ai", previous_028_active, previous_028_info, "0.2.8ai-active-critical.tar.gz", PREVIOUS_028_STATE_CONTAINER, include_caches=False)
    archive_state("0.2.7-ai3", predecessor_active, predecessor_info, "0.2.7-ai3-active-critical.tar.gz", PREDECESSOR_AI_STATE_CONTAINER, include_caches=False)
    archive_state("rc4-2-stable", stable_active, stable_info, "rc4-2-stable-active-critical.tar.gz", STABLE_STATE_CONTAINER, include_caches=False)
    previous_028ai12_container = bank / PREVIOUS_028AI12_STATE_CONTAINER
    previous_028ai12_container_full = None
    if previous_028ai12_container.is_dir():
        previous_028ai12_container_full = dest / "0.2.8ai.12-CONTAINER-FULL.tar.gz"
        with tarfile.open(previous_028ai12_container_full, "w:gz") as tf:
            tf.add(previous_028ai12_container, arcname=PREVIOUS_028AI12_STATE_CONTAINER)
    previous_028ai7_container = bank / PREVIOUS_028AI7_STATE_CONTAINER
    previous_028ai7_container_full = None
    if previous_028ai7_container.is_dir():
        previous_028ai7_container_full = dest / "0.2.8ai.7-CONTAINER-FULL.tar.gz"
        with tarfile.open(previous_028ai7_container_full, "w:gz") as tf:
            tf.add(previous_028ai7_container, arcname=PREVIOUS_028AI7_STATE_CONTAINER)
    previous_028ai5_container = bank / PREVIOUS_028AI5_STATE_CONTAINER
    previous_028ai5_container_full = None
    if previous_028ai5_container.is_dir():
        previous_028ai5_container_full = dest / "0.2.8ai.5-CONTAINER-FULL.tar.gz"
        with tarfile.open(previous_028ai5_container_full, "w:gz") as tf:
            tf.add(previous_028ai5_container, arcname=PREVIOUS_028AI5_STATE_CONTAINER)
    previous_028ai4_container = bank / PREVIOUS_028AI4_STATE_CONTAINER
    previous_028ai4_container_full = None
    if previous_028ai4_container.is_dir():
        previous_028ai4_container_full = dest / "0.2.8ai.4-CONTAINER-FULL.tar.gz"
        with tarfile.open(previous_028ai4_container_full, "w:gz") as tf:
            tf.add(previous_028ai4_container, arcname=PREVIOUS_028AI4_STATE_CONTAINER)
    previous_028ai3_container = bank / PREVIOUS_028AI3_STATE_CONTAINER
    previous_028ai3_container_full = None
    if previous_028ai3_container.is_dir():
        previous_028ai3_container_full = dest / "0.2.8ai.3-CONTAINER-FULL.tar.gz"
        with tarfile.open(previous_028ai3_container_full, "w:gz") as tf:
            tf.add(previous_028ai3_container, arcname=PREVIOUS_028AI3_STATE_CONTAINER)
    previous_028ai2_container = bank / PREVIOUS_028AI2_STATE_CONTAINER
    previous_028ai2_container_full = None
    if previous_028ai2_container.is_dir():
        previous_028ai2_container_full = dest / "0.2.8ai.2-CONTAINER-FULL.tar.gz"
        with tarfile.open(previous_028ai2_container_full, "w:gz") as tf:
            tf.add(previous_028ai2_container, arcname=PREVIOUS_028AI2_STATE_CONTAINER)
    previous_028ai1_container = bank / PREVIOUS_028AI1_STATE_CONTAINER
    previous_028ai1_container_full = None
    if previous_028ai1_container.is_dir():
        previous_028ai1_container_full = dest / "0.2.8ai.1-CONTAINER-FULL.tar.gz"
        with tarfile.open(previous_028ai1_container_full, "w:gz") as tf:
            tf.add(previous_028ai1_container, arcname=PREVIOUS_028AI1_STATE_CONTAINER)
    previous_028_container = bank / PREVIOUS_028_STATE_CONTAINER
    previous_028_container_full = None
    if previous_028_container.is_dir():
        previous_028_container_full = dest / "0.2.8ai-CONTAINER-FULL.tar.gz"
        with tarfile.open(previous_028_container_full, "w:gz") as tf:
            tf.add(previous_028_container, arcname=PREVIOUS_028_STATE_CONTAINER)
    predecessor_container = bank / PREDECESSOR_AI_STATE_CONTAINER
    predecessor_container_full = None
    if predecessor_container.is_dir():
        predecessor_container_full = dest / "0.2.7-ai3-CONTAINER-FULL.tar.gz"
        with tarfile.open(predecessor_container_full, "w:gz") as tf:
            tf.add(predecessor_container, arcname=PREDECESSOR_AI_STATE_CONTAINER)
    stable_container = bank / STABLE_STATE_CONTAINER
    stable_container_full = None
    if stable_container.is_dir():
        # Full immutable snapshot including retained generations and derived caches.
        stable_container_full = dest / "rc4-2-stable-CONTAINER-FULL.tar.gz"
        with tarfile.open(stable_container_full, "w:gz") as tf:
            tf.add(stable_container, arcname=STABLE_STATE_CONTAINER)

    if legacy.exists():
        inv = _inventory(legacy); inv["label"] = "rc3-2"; states.append(inv)
        with tarfile.open(dest / "rc3-2-runtime-state.tar.gz", "w:gz") as tf:
            tf.add(legacy, arcname=LEGACY_STATE, filter=_tar_filter)
    else:
        states.append({"label": "rc3-2", "exists": False, "path": str(legacy), "critical_files": {}, "cache_summary": {}})

    registry = Path(project_root / "config.json")
    registry_path = None
    if registry.exists():
        try:
            config = json.loads(registry.read_text(encoding="utf-8"))
            registry_path = Path(config.get("registry_path", "")).expanduser()
            if registry_path.exists(): shutil.copy2(registry_path, dest / "ai-registry-snapshot.json")
        except Exception:
            registry_path = None
    _write_layout_snapshot(bank, dest)
    manifest = {
        "format": 3, "created_at": time.time(), "created_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": reason, "voicebank_root": str(bank), "voicebank_name": bank.name,
        "project_root": str(project_root), "registry_path": str(registry_path) if registry_path else None,
        "states": states,
        "stable_container_full_archive": str(stable_container_full) if stable_container_full else None,
        "previous_028ai12_container_full_archive": str(previous_028ai12_container_full) if previous_028ai12_container_full else None,
        "previous_028ai7_container_full_archive": str(previous_028ai7_container_full) if previous_028ai7_container_full else None,
        "previous_028ai5_container_full_archive": str(previous_028ai5_container_full) if previous_028ai5_container_full else None,
        "previous_028ai4_container_full_archive": str(previous_028ai4_container_full) if previous_028ai4_container_full else None,
        "previous_028ai3_container_full_archive": str(previous_028ai3_container_full) if previous_028ai3_container_full else None,
        "previous_028ai2_container_full_archive": str(previous_028ai2_container_full) if previous_028ai2_container_full else None,
        "previous_028ai1_container_full_archive": str(previous_028ai1_container_full) if previous_028ai1_container_full else None,
        "previous_028_container_full_archive": str(previous_028_container_full) if previous_028_container_full else None,
        "predecessor_ai_container_full_archive": str(predecessor_container_full) if predecessor_container_full else None,
        "cache_policy": "Current 0.2.8ai.13 routine cache is excluded; predecessor 0.2.8ai.12, 0.2.8ai.7, 0.2.8ai.5, 0.2.8ai.4, 0.2.8ai.3, 0.2.8ai.2, 0.2.8ai.1, 0.2.8ai, AI.3 and RC4.2 containers are archived separately in full, including retained generations and caches",
        "raw_voicebank_policy": "WAV/OTO files are never modified by AI Deep; layout metadata is archived separately",
    }
    atomic_write_json(dest / "backup-manifest.json", manifest)
    (base / "LATEST_BACKUP.txt").write_text(str(dest) + "\n", encoding="utf-8")
    return dest, manifest

def _latest_backup_for(bank):
    base = Path.home() / "Documents" / "Yuaz-DDSP-Backups" / _safe_name(Path(bank).name)
    marker = base / "LATEST_BACKUP.txt"
    if marker.exists():
        p = Path(marker.read_text(encoding="utf-8").strip()).expanduser()
        if p.exists():
            return p
    dirs = sorted([p for p in base.glob("*") if p.is_dir()], reverse=True) if base.exists() else []
    return dirs[0] if dirs else None


def restore_backup(backup_dir, voicebank, target="ai"):
    backup_dir = Path(backup_dir).expanduser().resolve()
    bank = Path(voicebank).expanduser().resolve()
    if target == "rc3-2":
        archive = backup_dir / "rc3-2-runtime-state.tar.gz"
        if not archive.exists():
            raise RuntimeError("Backup has no RC3.2 state.")
        existing = bank / LEGACY_STATE
        if existing.exists():
            moved = bank / f"{LEGACY_STATE}-pre-restore-{time.strftime('%Y%m%d-%H%M%S')}"
            existing.rename(moved)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(bank, filter="data")
        return bank / LEGACY_STATE
    if target != "ai":
        raise RuntimeError("0.2.8ai.13 restore only supports target=ai or rc3-2; stable RC4.2 is deliberately not modified by this branch.")
    archive = backup_dir / "0.2.8ai.13-runtime-state.tar.gz"
    if not archive.exists():
        raise RuntimeError("Backup has no 0.2.8ai.13 state. Stable RC4.2 snapshots are read-only safety copies in this branch.")
    generation, staging = begin_generation(bank, "restore-ai")
    temp = staging / "_extract"
    temp.mkdir()
    try:
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(temp, filter="data")
        src = temp / "0.2.8ai.13-state"
        if not src.is_dir():
            raise RuntimeError("0.2.8ai.13 backup archive is malformed.")
        for child in src.iterdir():
            shutil.move(str(child), str(staging / child.name))
        shutil.rmtree(temp)
        (staging / "runtime_registry.json").unlink(missing_ok=True)
        final, _ = commit_generation(bank, generation, staging, reason="external-ai-backup-restore", acoustic_base="0.2.8ai.13-twelve-control-modular-ai-ddsp")
        return final
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("backup")
    b.add_argument("voicebank")
    b.add_argument("--project-root", required=True)
    b.add_argument("--reason", default="manual")
    r = sub.add_parser("restore")
    r.add_argument("voicebank")
    r.add_argument("--backup-dir")
    r.add_argument("--target", choices=("rc3-2", "ai"), default="ai")
    args = parser.parse_args()
    if args.command == "backup":
        dest, manifest = create_backup(args.project_root, args.voicebank, args.reason)
        print(f"Backup completed successfully: {dest}")
        for state in manifest["states"]:
            print(f"  {state['label']} state backed up: {bool(state.get('exists'))}")
    else:
        backup_dir = Path(args.backup_dir).expanduser() if args.backup_dir else _latest_backup_for(args.voicebank)
        if backup_dir is None:
            raise SystemExit("No backup was found for this voicebank.")
        restored = restore_backup(backup_dir, args.voicebank, args.target)
        print(f"Restored {args.target} state: {restored}")


if __name__ == "__main__":
    main()
