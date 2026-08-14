#!/usr/bin/env python3
import argparse
import json
import shutil
import time
from pathlib import Path

from .backup import create_backup
from .client import ENGINE_VERSION, ping, send
from .learned_highband import build_profile_database, save_profile_database
from .ai_vocal_controls import load_ai_control_adapter
from .highband_foundation import load_highband_foundation
from .prepare import VoicebankPreparer
from .checkpoint_identity import checkpoint_identity_sha, checkpoint_identity
from .state import (
    begin_generation, clone_state, commit_generation, link_analysis_caches,
    legacy_dir, merge_global_registry, resolve_active_state, resolve_stable_state, sha256,
    validate_state, write_local_registry, atomic_write_json,
)


def load_config(project_root):
    path = Path(project_root) / "config.json"
    if not path.exists():
        raise RuntimeError("config.json not found; run configure-macos.command first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_remove_staging(staging):
    staging = Path(staging)
    if not staging.exists():
        return
    # Never follow cache symlinks when cleaning a failed staging generation.
    for child in list(staging.iterdir()):
        if child.is_symlink():
            child.unlink(missing_ok=True)
    shutil.rmtree(staging, ignore_errors=True)


def _force_highband_in_state(bank, state):
    state = Path(state)
    cache = state / "highband_cache_v3_ai14"
    if cache.is_symlink():
        cache.unlink()
    elif cache.exists():
        shutil.rmtree(cache)
    manifest = json.loads((state / "manifest.json").read_text(encoding="utf-8"))
    db = build_profile_database(
        Path(bank), manifest.get("entries") or [], model_hop=256, model_sr=24000, state_dir=state,
    )
    out = state / "highband_profiles_v3.ai14.json"
    save_profile_database(out, db)
    return out, db



def _quiesce_runtime(config, phase):
    host = str(config.get("host", "127.0.0.1"))
    port = int(config.get("port", 47886))
    runtime_id = str(config.get("runtime_id") or ENGINE_VERSION)
    status = ping(host, port)
    if not status:
        return
    if status.get("ready"):
        if status.get("engine_version") != ENGINE_VERSION or status.get("runtime_id") != runtime_id:
            raise RuntimeError(
                f"Port {port} is occupied by another Yuaz runtime during {phase}; refusing a state switch."
            )
        active = int(status.get("active_renders") or 0)
        if active > 0:
            raise RuntimeError(
                f"OpenUtau/Yuaz is actively rendering ({active} request(s)) during {phase}. "
                "Close/stop rendering and run Prepare again. ACTIVE was not changed."
            )
        try:
            send(host, port, {"action":"shutdown", "runtime_id":runtime_id}, timeout=2)
        except Exception:
            pass
        deadline=time.time()+5.0
        while time.time()<deadline:
            if not ping(host,port):
                return
            time.sleep(0.1)
        raise RuntimeError(f"0.2.8ai.14 engine would not stop cleanly during {phase}; ACTIVE was not changed.")


def _find_ai_control_foundation(project_root, source_state=None):
    project_root = Path(project_root).expanduser().resolve()
    home = Path.home()
    candidates = [
        project_root / "control_models" / "ai_control_foundation-v2.pt",
    ]
    if source_state is not None:
        candidates.append(Path(source_state) / "ai_control_adapter.ai14.pt")
    candidates.extend([
        home / "Documents" / "Yuaz-DDSP-Backups" / "control-models" / "ai_control_foundation-v2-Chinese-Core.pt",
        home / "Library" / "Application Support" / "YuazDDSP" / "0.2.7-alpha.8-rc.4.3-ai.3" / "control_models" / "ai_control_foundation-v2.pt",
        home / "Downloads" / "yuaz-ddsp-resampler-v0.2.7-alpha.8-rc.4.3-ai.3" / "control_models" / "ai_control_foundation-v2.pt",
    ])
    seen = set()
    for candidate in candidates:
        try:
            key = str(candidate.expanduser().resolve())
        except Exception:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate
    return None


def _attach_ai_control_foundation(project_root, staging, source_state=None):
    project_root = Path(project_root)
    staging = Path(staging)
    foundation = _find_ai_control_foundation(project_root, source_state=source_state)
    meta_path = staging / "ai_control_training.ai14.json"
    if foundation is None:
        atomic_write_json(meta_path, {
            "format": 3, "accepted": False, "backend": "deterministic-fallback",
            "reason": "no-trained-foundation-model",
            "created_at": time.time(),
        })
        print("AI Control Foundation: no compatible predecessor/current model found.")
        print("YB/YF/YX/YP learned controls stay unavailable; the deterministic controls remain usable.")
        return None
    model, metadata = load_ai_control_adapter(foundation, device="cpu")
    learned_controls = tuple(getattr(model, "control_names", ()))
    del model
    expected_controls = ("breathiness", "falsetto", "mixed_voice", "pharyngeal")
    if learned_controls != expected_controls:
        atomic_write_json(meta_path, {
            "format": 3, "accepted": False, "backend": "deterministic-fallback",
            "reason": "foundation-control-set-mismatch",
            "controls": list(learned_controls), "expected_controls": list(expected_controls),
            "source": str(foundation), "created_at": time.time(),
        })
        print("AI Control Foundation rejected: predecessor model has an incompatible control set.")
        return None
    config = load_config(project_root)
    checkpoint = Path(config["checkpoint"]).expanduser().resolve()
    current_checkpoint_sha = checkpoint_identity_sha(checkpoint) if checkpoint.is_file() else "missing:" + str(checkpoint)
    feature_backend = str(metadata.get("feature_backend") or "")
    foundation_checkpoint_sha = str(metadata.get("checkpoint_sha256") or "")
    if feature_backend != "yuaz-native-ddsp-v1":
        atomic_write_json(meta_path, {
            "format": 3, "accepted": False, "backend": "deterministic-fallback",
            "reason": "foundation-not-native-yuaz-ddsp",
            "feature_backend": feature_backend, "source": str(foundation), "created_at": time.time(),
        })
        print("AI Control Foundation rejected: 0.2.8ai.14 requires Yuaz-native DDSP training features.")
        return None
    if not foundation_checkpoint_sha or foundation_checkpoint_sha != current_checkpoint_sha:
        atomic_write_json(meta_path, {
            "format": 3, "accepted": False, "backend": "deterministic-fallback",
            "reason": "foundation-checkpoint-mismatch",
            "foundation_checkpoint_sha256": foundation_checkpoint_sha,
            "current_checkpoint_sha256": current_checkpoint_sha,
            "source": str(foundation), "created_at": time.time(),
        })
        print("AI Control Foundation rejected: it was trained against a different Yuaz checkpoint.")
        return None
    target = staging / "ai_control_adapter.ai14.pt"
    shutil.copy2(foundation, target)
    atomic_write_json(meta_path, {
        "format": 3, "accepted": True, "backend": "ai-ddsp",
        "direct_controls": ["breathiness", "falsetto", "mixed_voice", "pharyngeal"],
        "foundation_path": str(foundation.resolve()),
        "foundation_sha256": sha256(foundation),
        "foundation_metadata": metadata,
        "migration_policy": "0.2.8ai.14 may reuse compatible frozen predecessor foundations without modifying their source",
        "frozen_during_voicebank_deep": True,
        "created_at": time.time(),
    })
    print(f"AI Control Foundation pinned into 0.2.8ai.14 generation from: {foundation}")
    print(f"Pinned copy: {target}")
    return target


def _find_ai_gender_foundation(project_root, source_state=None):
    project_root = Path(project_root).expanduser().resolve()
    home = Path.home()
    candidates = [project_root / "control_models" / "ai_gender_foundation-v1.pt"]
    if source_state is not None:
        candidates.append(Path(source_state) / "ai_gender_adapter.ai14.pt")
    candidates.extend([
        home / "Documents" / "Yuaz-DDSP-Backups" / "control-models" / "ai_gender_foundation-v1-VocalSet.pt",
        home / "Library" / "Application Support" / "YuazDDSP" / "0.2.8ai.14" / "control_models" / "ai_gender_foundation-v1.pt",
        home / "Downloads" / "yuaz-ddsp-resampler-v0.2.8ai.14" / "control_models" / "ai_gender_foundation-v1.pt",
        home / "Library" / "Application Support" / "YuazDDSP" / "0.2.8ai.11" / "control_models" / "ai_gender_foundation-v1.pt",
        home / "Downloads" / "yuaz-ddsp-resampler-v0.2.8ai.11" / "control_models" / "ai_gender_foundation-v1.pt",
        home / "Library" / "Application Support" / "YuazDDSP" / "0.2.8ai.2" / "control_models" / "ai_gender_foundation-v1.pt",
        home / "Downloads" / "yuaz-ddsp-resampler-v0.2.8ai.2" / "control_models" / "ai_gender_foundation-v1.pt",
        home / "Library" / "Application Support" / "YuazDDSP" / "0.2.8ai.1" / "control_models" / "ai_gender_foundation-v1.pt",
        home / "Downloads" / "yuaz-ddsp-resampler-v0.2.8ai.1" / "control_models" / "ai_gender_foundation-v1.pt",
    ])
    seen=set()
    for candidate in candidates:
        try: key=str(candidate.expanduser().resolve())
        except Exception: key=str(candidate)
        if key in seen: continue
        seen.add(key)
        if candidate.is_file(): return candidate
    return None


def _attach_ai_gender_foundation(project_root, staging, source_state=None):
    staging=Path(staging)
    foundation=_find_ai_gender_foundation(project_root, source_state=source_state)
    meta_path=staging / "ai_gender_training.ai14.json"
    if foundation is None:
        atomic_write_json(meta_path, {"format":1,"accepted":False,"backend":"deterministic-fallback","reason":"no-trained-gender-foundation","created_at":time.time()})
        print("AI Gender Foundation: not found; YG keeps deterministic DDSP Gender/Formant.")
        return None
    try:
        model, metadata=load_ai_control_adapter(foundation, device="cpu", expected_controls=("gender_formant",))
        modes=tuple(getattr(model,"control_modes",()))
        scopes=tuple(getattr(model,"output_scopes",()))
        del model
    except Exception as exc:
        atomic_write_json(meta_path,{"format":1,"accepted":False,"backend":"deterministic-fallback","reason":"invalid-gender-foundation","error":str(exc),"source":str(foundation),"created_at":time.time()})
        print(f"AI Gender Foundation rejected: {exc}")
        return None
    if modes != ("signed",) or scopes != ("spectral",):
        atomic_write_json(meta_path,{"format":1,"accepted":False,"backend":"deterministic-fallback","reason":"gender-model-scope-mismatch","control_modes":list(modes),"output_scopes":list(scopes),"source":str(foundation),"created_at":time.time()})
        print("AI Gender Foundation rejected: YG foundation must be signed and spectral-only.")
        return None
    config=load_config(project_root)
    checkpoint=Path(config["checkpoint"]).expanduser().resolve()
    current_sha=checkpoint_identity_sha(checkpoint) if checkpoint.is_file() else "missing:"+str(checkpoint)
    if str(metadata.get("feature_backend") or "") != "yuaz-native-ddsp-v1" or str(metadata.get("checkpoint_sha256") or "") != current_sha:
        atomic_write_json(meta_path,{"format":1,"accepted":False,"backend":"deterministic-fallback","reason":"gender-foundation-provenance-mismatch","source":str(foundation),"current_checkpoint_sha256":current_sha,"foundation_checkpoint_sha256":metadata.get("checkpoint_sha256"),"created_at":time.time()})
        print("AI Gender Foundation rejected: Yuaz checkpoint/provenance mismatch.")
        return None
    target=staging / "ai_gender_adapter.ai14.pt"
    shutil.copy2(foundation,target)
    atomic_write_json(meta_path,{"format":1,"accepted":True,"backend":"ai-ddsp","direct_controls":["gender_formant"],"output_scopes":["spectral"],"foundation_path":str(foundation.resolve()),"foundation_sha256":sha256(foundation),"foundation_metadata":metadata,"frozen_during_voicebank_deep":True,"created_at":time.time()})
    print(f"AI Gender Foundation pinned into 0.2.8ai.14 generation from: {foundation}")
    print(f"Pinned gender copy: {target}")
    return target


def _find_modular_foundation(project_root, filename, backup_names=(), source_state=None, state_filename=None):
    project_root = Path(project_root).expanduser().resolve()
    home = Path.home()
    candidates = [project_root / "control_models" / filename]
    if source_state is not None and state_filename:
        candidates.append(Path(source_state) / state_filename)
    for b in backup_names:
        candidates.append(home / "Documents" / "Yuaz-DDSP-Backups" / "control-models" / b)
    for ver in ("0.2.8ai.14", "0.2.8ai.12", "0.2.8ai.11", "0.2.8ai.3", "0.2.8ai.2", "0.2.8ai.1", "0.2.8ai"):
        candidates.append(home / "Library" / "Application Support" / "YuazDDSP" / ver / "control_models" / filename)
        candidates.append(home / "Downloads" / f"yuaz-ddsp-resampler-v{ver}" / "control_models" / filename)
    seen=set()
    for candidate in candidates:
        try: key=str(candidate.expanduser().resolve())
        except Exception: key=str(candidate)
        if key in seen: continue
        seen.add(key)
        if candidate.is_file(): return candidate
    return None


def _attach_modular_foundation(project_root, staging, *, filename, target_filename, metadata_filename,
                               expected_controls, expected_modes, expected_scopes, backup_names=(), source_state=None, label="AI pack"):
    staging=Path(staging)
    foundation=_find_modular_foundation(project_root, filename, backup_names=backup_names, source_state=source_state, state_filename=target_filename)
    meta_path=staging / metadata_filename
    if foundation is None:
        atomic_write_json(meta_path,{"format":1,"accepted":False,"backend":"deterministic-fallback","reason":"no-trained-foundation","pack":label,"created_at":time.time()})
        print(f"{label}: not found; corresponding controls keep deterministic DDSP fallback.")
        return None
    try:
        model, metadata=load_ai_control_adapter(foundation, device="cpu", expected_controls=tuple(expected_controls))
        modes=tuple(getattr(model,"control_modes",()))
        scopes=tuple(getattr(model,"output_scopes",()))
        del model
    except Exception as exc:
        atomic_write_json(meta_path,{"format":1,"accepted":False,"backend":"deterministic-fallback","reason":"invalid-foundation","error":str(exc),"source":str(foundation),"created_at":time.time()})
        print(f"{label} rejected: {exc}")
        return None
    if tuple(modes)!=tuple(expected_modes) or tuple(scopes)!=tuple(expected_scopes):
        atomic_write_json(meta_path,{"format":1,"accepted":False,"backend":"deterministic-fallback","reason":"scope-or-mode-mismatch","modes":list(modes),"scopes":list(scopes),"source":str(foundation),"created_at":time.time()})
        print(f"{label} rejected: control mode/output scope mismatch.")
        return None
    config=load_config(project_root)
    checkpoint=Path(config["checkpoint"]).expanduser().resolve()
    current_sha=checkpoint_identity_sha(checkpoint) if checkpoint.is_file() else "missing:"+str(checkpoint)
    foundation_sha=str(metadata.get("checkpoint_sha256") or "")
    if str(metadata.get("feature_backend") or "")!="yuaz-native-ddsp-v1" or not foundation_sha or foundation_sha!=current_sha:
        atomic_write_json(meta_path,{"format":1,"accepted":False,"backend":"deterministic-fallback","reason":"provenance-mismatch","source":str(foundation),"created_at":time.time()})
        print(f"{label} rejected: Yuaz-native checkpoint provenance mismatch.")
        return None
    target=staging/target_filename
    shutil.copy2(foundation,target)
    atomic_write_json(meta_path,{"format":1,"accepted":True,"backend":"ai-ddsp","controls":list(expected_controls),"source":str(foundation.resolve()),"foundation_sha256":sha256(foundation),"foundation_metadata":metadata,"frozen_during_voicebank_deep":True,"created_at":time.time()})
    print(f"{label} pinned into 0.2.8ai.14 generation from: {foundation}")
    return target


def _attach_ai_phonation_foundation(project_root, staging, source_state=None):
    return _attach_modular_foundation(project_root, staging, filename="ai_phonation_foundation-v1.pt", target_filename="ai_phonation_adapter.ai14.pt", metadata_filename="ai_phonation_training.ai14.json", expected_controls=("tension","voicing"), expected_modes=("signed","signed"), expected_scopes=("spectral","ap","gate"), backup_names=("ai_phonation_foundation-v1-PhonationModes-MOCHA.pt","ai_phonation_foundation-v1-VQS-MOCHA.pt"), source_state=source_state, label="AI Phonation Foundation")


def _attach_ai_mouth_foundation(project_root, staging, source_state=None):
    return _attach_modular_foundation(project_root, staging, filename="ai_mouth_foundation-v1.pt", target_filename="ai_mouth_adapter.ai14.pt", metadata_filename="ai_mouth_training.ai14.json", expected_controls=("mouth",), expected_modes=("signed",), expected_scopes=("spectral",), backup_names=("ai_mouth_foundation-v1-MOCHA.pt",), source_state=source_state, label="AI Mouth Foundation")



def _find_highband_foundation(project_root, source_state=None):
    project_root = Path(project_root).expanduser().resolve()
    home = Path.home()
    candidates = [project_root / "control_models" / "highband_foundation-v2.pt", project_root / "control_models" / "highband_foundation-v1.pt"]
    if source_state is not None:
        candidates.append(Path(source_state) / "highband_foundation.ai14.pt")
    candidates.extend([
        home / "Documents" / "Yuaz-DDSP-Backups" / "control-models" / "0.2.8ai.14" / "highband_foundation-v2.pt",
        home / "Documents" / "Yuaz-DDSP-Backups" / "control-models" / "0.2.8ai.14" / "highband_foundation-v1.pt",
        home / "Library" / "Application Support" / "YuazDDSP" / "0.2.8ai.14" / "control_models" / "highband_foundation-v2.pt",
        home / "Library" / "Application Support" / "YuazDDSP" / "0.2.8ai.14" / "control_models" / "highband_foundation-v1.pt",
        home / "Downloads" / "yuaz-ddsp-resampler-v0.2.8ai.14" / "control_models" / "highband_foundation-v2.pt",
        home / "Downloads" / "yuaz-ddsp-resampler-v0.2.8ai.14" / "control_models" / "highband_foundation-v1.pt",
        home / "Documents" / "Yuaz-DDSP-Backups" / "control-models" / "0.2.8ai.12" / "highband_foundation-v2.pt",
        home / "Documents" / "Yuaz-DDSP-Backups" / "control-models" / "0.2.8ai.12" / "highband_foundation-v1.pt",
        home / "Library" / "Application Support" / "YuazDDSP" / "0.2.8ai.12" / "control_models" / "highband_foundation-v2.pt",
        home / "Library" / "Application Support" / "YuazDDSP" / "0.2.8ai.12" / "control_models" / "highband_foundation-v1.pt",
        home / "Downloads" / "yuaz-ddsp-resampler-v0.2.8ai.12" / "control_models" / "highband_foundation-v2.pt",
        home / "Downloads" / "yuaz-ddsp-resampler-v0.2.8ai.12" / "control_models" / "highband_foundation-v1.pt",
        home / "Documents" / "Yuaz-DDSP-Backups" / "control-models" / "0.2.8ai.11" / "highband_foundation-v2.pt",
        home / "Documents" / "Yuaz-DDSP-Backups" / "control-models" / "0.2.8ai.11" / "highband_foundation-v1.pt",
        home / "Library" / "Application Support" / "YuazDDSP" / "0.2.8ai.11" / "control_models" / "highband_foundation-v2.pt",
        home / "Library" / "Application Support" / "YuazDDSP" / "0.2.8ai.11" / "control_models" / "highband_foundation-v1.pt",
        home / "Downloads" / "yuaz-ddsp-resampler-v0.2.8ai.11" / "control_models" / "highband_foundation-v2.pt",
        home / "Downloads" / "yuaz-ddsp-resampler-v0.2.8ai.11" / "control_models" / "highband_foundation-v1.pt",
    ])
    seen=set()
    for candidate in candidates:
        try: key=str(candidate.expanduser().resolve())
        except Exception: key=str(candidate)
        if key in seen: continue
        seen.add(key)
        if candidate.is_file(): return candidate
    return None


def _attach_highband_foundation(project_root, staging, source_state=None):
    staging = Path(staging)
    foundation = _find_highband_foundation(project_root, source_state=source_state)
    meta_path = staging / "highband_foundation_training.ai14.json"
    if foundation is None:
        atomic_write_json(meta_path, {
            "format": 1, "accepted": False, "backend": "voicebank-profile-fallback",
            "reason": "no-trained-highband-foundation", "created_at": time.time(),
        })
        print("High-Band Foundation: not found; YH keeps the voicebank-profile fallback.")
        return None
    try:
        model, metadata = load_highband_foundation(foundation, device="cpu")
        del model
        target_sr = int(metadata.get("target_sample_rate", 48000))
        if target_sr != 48000:
            raise RuntimeError(f"expected 48 kHz foundation, got {target_sr}")
    except Exception as exc:
        atomic_write_json(meta_path, {
            "format": 1, "accepted": False, "backend": "voicebank-profile-fallback",
            "reason": "foundation-validation-failed", "error": str(exc),
            "source": str(foundation), "created_at": time.time(),
        })
        print(f"High-Band Foundation rejected: {exc}")
        return None
    target = staging / "highband_foundation.ai14.pt"
    shutil.copy2(foundation, target)
    revision = int(metadata.get("foundation_revision", 1) or 1)
    backend = str(metadata.get("runtime_backend") or f"highband-foundation-v{revision}")
    atomic_write_json(meta_path, {
        "format": 1, "accepted": True, "backend": backend,
        "foundation_revision": revision,
        "foundation_path": str(foundation.resolve()),
        "foundation_sha256": sha256(foundation),
        "foundation_metadata": metadata,
        "voicebank_profile_role": "style-and-tilt-conditioning",
        "frozen_during_voicebank_deep": True,
        "created_at": time.time(),
    })
    print(f"High-Band Foundation r{revision} pinned into 0.2.8ai.14 generation from: {foundation}")
    print(f"Pinned high-band copy: {target}")
    return target

def _state_base_sha(state):
    if state is None:
        return ""
    p = Path(state) / "base_model.json"
    if not p.is_file():
        return ""
    try:
        return str(json.loads(p.read_text(encoding="utf-8")).get("source_checkpoint_sha256") or "")
    except Exception:
        return ""


def _write_base_model_metadata(staging, config):
    ident = checkpoint_identity(config["checkpoint"])
    payload = {
        "format": 1,
        "engine_version": "0.2.8ai.14",
        "model_id": str(config.get("base_checkpoint_model_id") or ident["source_checkpoint_sha256"][:16]),
        "source_checkpoint": str(config.get("base_checkpoint_source_name") or ident.get("source_checkpoint") or ""),
        "source_checkpoint_sha256": str(config.get("base_checkpoint_sha256") or ident["source_checkpoint_sha256"]),
        "runtime_sha256": str(config.get("base_checkpoint_runtime_sha256") or ident["runtime_sha256"]),
        "source_step": config.get("base_checkpoint_step", ident.get("source_step")),
        "runtime_path": str(Path(config["checkpoint"]).expanduser().resolve()),
        "created_at": time.time(),
    }
    atomic_write_json(Path(staging) / "base_model.json", payload)
    return payload


def run_transaction(project_root, voicebank, mode):
    project_root = Path(project_root).expanduser().resolve()
    bank = Path(voicebank).expanduser().resolve()
    if not bank.is_dir():
        raise RuntimeError(f"Voicebank folder not found: {bank}")
    config = load_config(project_root)
    global_registry = Path(config["registry_path"]).expanduser().resolve()

    _quiesce_runtime(config, "preparation start")
    backup_dir, _ = create_backup(project_root, bank, reason=f"0.2.8ai.14-{mode}")
    print(f"Safety backup completed: {backup_dir}")

    source, source_info = resolve_active_state(bank, allow_legacy=True, verify=True)
    current_base_sha = str(config.get("base_checkpoint_sha256") or checkpoint_identity_sha(config["checkpoint"]))
    source_base_sha = _state_base_sha(source)
    source_compatible = bool(source is not None and source_base_sha and source_base_sha == current_base_sha)
    generation, staging = begin_generation(bank, mode)
    print(f"Building isolated generation: {generation}")
    print(f"Current render state remains untouched until commit: {source}")

    try:
        if mode == "adopt":
            raise RuntimeError("ai.14 does not adopt predecessor trained state. Use Clean Deep so ai.13 remains read-only and checkpoint-isolated.")
        elif mode == "deep":
            print("Production Deep: rebuilding ai.14 analysis and learned state from source WAVs.")
            print("ai.13 state is never cloned, linked, renamed, or modified.")
            preparer = VoicebankPreparer(project_root, bank, "deep", state_dir=staging)
            preparer.run(register=False)
            reusable_source = source if source_compatible else None
            _attach_ai_control_foundation(project_root, staging, source_state=reusable_source)
            _attach_ai_gender_foundation(project_root, staging, source_state=reusable_source)
            _attach_ai_phonation_foundation(project_root, staging, source_state=reusable_source)
            _attach_ai_mouth_foundation(project_root, staging, source_state=reusable_source)
            _attach_highband_foundation(project_root, staging, source_state=reusable_source)
            reason = "0.2.8ai.14-clean-deep-checkpoint-isolated"
        elif mode == "continue":
            if source is None:
                raise RuntimeError("No ai.14 prepared state exists to continue. Use Clean Deep instead.")
            if not source_compatible:
                raise RuntimeError("The active ai.14 generation was trained under another Yuaz base checkpoint. Use Clean Deep for the selected base model.")
            clone_state(source, staging, link_caches=False)
            print("Continue Deep: copied the active generation into isolated staging; no mutable cache symlinks are used.")
            preparer = VoicebankPreparer(project_root, bank, "deep", state_dir=staging)
            preparer.run(register=False)
            _attach_ai_control_foundation(project_root, staging, source_state=source)
            _attach_ai_gender_foundation(project_root, staging, source_state=source)
            _attach_ai_phonation_foundation(project_root, staging, source_state=source)
            _attach_ai_mouth_foundation(project_root, staging, source_state=source)
            _attach_highband_foundation(project_root, staging, source_state=source)
            reason = "continue-0.2.8ai.14-deep-isolated-copy-plus-modular-control-packs"
        elif mode == "highband":
            if source is None:
                raise RuntimeError("No ai.14 prepared state exists. Use Clean Deep first.")
            if not source_compatible:
                raise RuntimeError("The active ai.14 generation belongs to another base checkpoint. Use Clean Deep first.")
            clone_state(source, staging, link_caches=True)
            _force_highband_in_state(bank, staging)
            _attach_highband_foundation(project_root, staging, source_state=source)
            reason = "relearn-highband-v3-plus-foundation-v2-compatible"
        else:
            raise RuntimeError(f"Unknown transaction mode: {mode}")

        # Do not switch acoustic generations while OpenUtau is rendering. If the
        # engine restarted during a long training job, quiesce it again now.
        base_meta = _write_base_model_metadata(staging, config)
        print(f"Pinned ai.14 base provenance: {base_meta['source_checkpoint_sha256'][:16]} step={base_meta.get('source_step')}")
        _quiesce_runtime(config, "ACTIVE commit")
        final, active = commit_generation(
            bank, generation, staging, reason=reason, acoustic_base="0.2.8ai.14-checkpoint-isolated-twelve-control-ddsp",
        )
        print(f"Committed ACTIVE generation: {final}")
        print(f"Previous generation retained for automatic rollback: {active.get('previous_generation')}")
        try:
            payload = write_local_registry(bank, final)
            print(f"Local registry: {final / 'runtime_registry.json'}")
            try:
                merge_global_registry(global_registry, payload)
                print(f"Global registry accelerator: {global_registry}")
            except Exception as exc:
                print(f"WARNING: global registry accelerator refresh failed: {exc}")
                print("Rendering remains safe because the voicebank-local generation is authoritative.")
        except Exception as exc:
            print(f"WARNING: local registry cache creation failed: {exc}")
            print("Rendering will reconstruct routing from the pinned manifest instead of changing acoustic state.")
        return final
    except Exception:
        _safe_remove_staging(staging)
        print("Preparation failed before ACTIVE switch. The previous render state was preserved.")
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("voicebank")
    p.add_argument("--project-root", required=True)
    p.add_argument("--mode", choices=("adopt", "deep", "continue", "highband"), required=True)
    a = p.parse_args()
    run_transaction(a.project_root, a.voicebank, a.mode)


if __name__ == "__main__":
    main()
