#!/usr/bin/env python3
import fcntl
import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path

from .loudness import oto_loudness_signature
from .voicebank import file_sha256, pcm_fingerprint, voicebank_id

ENGINE_VERSION = "0.2.8ai.13"
STATE_CONTAINER = ".yuaz-0.2.8ai13"
PREVIOUS_028AI12_STATE_CONTAINER = ".yuaz-0.2.8ai12"
PREVIOUS_028AI11_STATE_CONTAINER = ".yuaz-0.2.8ai11"
PREVIOUS_028AI10_STATE_CONTAINER = ".yuaz-0.2.8ai10"
PREVIOUS_028AI9_STATE_CONTAINER = ".yuaz-0.2.8ai9"
PREVIOUS_028AI8_STATE_CONTAINER = ".yuaz-0.2.8ai8"
PREVIOUS_028AI7_STATE_CONTAINER = ".yuaz-0.2.8ai7"
PREVIOUS_028AI6_STATE_CONTAINER = ".yuaz-0.2.8ai6"
PREVIOUS_028AI5_STATE_CONTAINER = ".yuaz-0.2.8ai5"
PREVIOUS_028AI4_STATE_CONTAINER = ".yuaz-0.2.8ai4"
PREVIOUS_028AI3_STATE_CONTAINER = ".yuaz-0.2.8ai3"
PREVIOUS_028AI2_STATE_CONTAINER = ".yuaz-0.2.8ai2"
PREVIOUS_028AI1_STATE_CONTAINER = ".yuaz-0.2.8ai1"
PREVIOUS_028_STATE_CONTAINER = ".yuaz-0.2.8ai"
PREDECESSOR_AI_STATE_CONTAINER = ".yuaz-alpha8-rc4-3-ai3"
STABLE_STATE_CONTAINER = ".yuaz-alpha8-rc3-3"
LEGACY_STATE = ".yuaz-alpha8-rc3-2"
ACTIVE_FILE = "ACTIVE.json"
GENERATIONS_DIR = "generations"
RUNTIME_REGISTRY = "runtime_registry.json"
FINGERPRINT_FILE = "state_fingerprint.json"

_LOCAL_CACHE = {}
_VERIFY_CACHE = {}


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        with tmp.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_json(path, payload):
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def sha256(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def container_dir(bank):
    return Path(bank).expanduser().resolve() / STATE_CONTAINER


def stable_container_dir(bank):
    return Path(bank).expanduser().resolve() / STABLE_STATE_CONTAINER

def legacy_dir(bank):
    return Path(bank).expanduser().resolve() / LEGACY_STATE


def generation_dir(bank, generation):
    return container_dir(bank) / GENERATIONS_DIR / generation


def _critical_files(state):
    names = [
        "profile.json", "manifest.json", "subbanks.json", "loudness.json",
        "highband_profiles_v3.json", "highband_foundation.pt", "highband_foundation_training.json", "adapter.pt", "timbre_profiles.pt",
        "ai_control_adapter.pt", "ai_control_training.json", "ai_gender_adapter.pt", "ai_gender_training.json",
        "ai_phonation_adapter.pt", "ai_phonation_training.json", "ai_mouth_adapter.pt", "ai_mouth_training.json",
        "training.json", "clarity_calibration.json", "fidelity_refiner.pt",
        "fidelity_training.json", "deep_validation.json", "articulation/index.json",
    ]
    return [state / name for name in names if (state / name).is_file()]


def validate_state(state, verify_hashes=True):
    state = Path(state)
    required = [
        state / "profile.json",
        state / "manifest.json",
        state / "subbanks.json",
        state / "loudness.json",
        state / "highband_profiles_v3.json",
        state / "articulation" / "index.json",
    ]
    missing = [str(p.relative_to(state)) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError("State is incomplete; missing: " + ", ".join(missing))
    for p in required:
        if p.suffix == ".json":
            _json(p)
    manifest = _json(state / "manifest.json")
    entries = manifest.get("entries") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("State manifest has no usable entries.")
    if (state / "training.json").exists():
        training = _json(state / "training.json")
        if not (state / "adapter.pt").is_file() or not (state / "timbre_profiles.pt").is_file():
            raise RuntimeError("Training metadata exists but adapter/timbre profiles are incomplete.")
        if training.get("mode") == "deep" and int(training.get("deep_training_version", 0) or 0) >= 1:
            deep_validation_path = state / "deep_validation.json"
            if not deep_validation_path.is_file():
                raise RuntimeError("Deep training state is missing deep_validation.json; refusing activation.")
            deep_validation = _json(deep_validation_path)
            if not bool(deep_validation.get("activation_safe", False)):
                raise RuntimeError("Deep training validation did not mark this generation activation-safe.")
    if (state / "fidelity_training.json").exists():
        fidelity = _json(state / "fidelity_training.json")
        if bool(fidelity.get("accepted", True)) and not (state / "fidelity_refiner.pt").is_file():
            raise RuntimeError("Accepted Fidelity metadata exists but fidelity_refiner.pt is missing.")
    if (state / "highband_foundation_training.json").exists():
        meta = _json(state / "highband_foundation_training.json")
        if bool(meta.get("accepted", False)) and not (state / "highband_foundation.pt").is_file():
            raise RuntimeError("High-band foundation metadata is accepted but highband_foundation.pt is missing.")
    if (state / "ai_control_training.json").exists():
        ai_meta = _json(state / "ai_control_training.json")
        if bool(ai_meta.get("accepted", False)) and not (state / "ai_control_adapter.pt").is_file():
            raise RuntimeError("AI technique metadata is accepted but ai_control_adapter.pt is missing.")
    if (state / "ai_gender_training.json").exists():
        gender_meta = _json(state / "ai_gender_training.json")
        if bool(gender_meta.get("accepted", False)) and not (state / "ai_gender_adapter.pt").is_file():
            raise RuntimeError("AI gender metadata is accepted but ai_gender_adapter.pt is missing.")
    if (state / "ai_phonation_training.json").exists():
        meta = _json(state / "ai_phonation_training.json")
        if bool(meta.get("accepted", False)) and not (state / "ai_phonation_adapter.pt").is_file():
            raise RuntimeError("AI phonation metadata is accepted but ai_phonation_adapter.pt is missing.")
    if (state / "ai_mouth_training.json").exists():
        meta = _json(state / "ai_mouth_training.json")
        if bool(meta.get("accepted", False)) and not (state / "ai_mouth_adapter.pt").is_file():
            raise RuntimeError("AI mouth metadata is accepted but ai_mouth_adapter.pt is missing.")
    art = _json(state / "articulation" / "index.json")
    aliases = art.get("aliases", {}) if isinstance(art, dict) else {}
    missing_templates = 0
    for item in aliases.values():
        rel = item.get("file") if isinstance(item, dict) else None
        if rel:
            p = Path(rel)
            if not p.is_absolute():
                # Stored paths are voicebank-relative in RC3.2. Resolve by locating the state parent.
                bank = state
                while bank.name.startswith(".yuaz") or bank.parent.name == GENERATIONS_DIR:
                    bank = bank.parent
                    if bank.name == GENERATIONS_DIR:
                        bank = bank.parent.parent
                        break
                candidate = bank / p
                if not candidate.exists():
                    # For generation-cloned states, canonical files are also available inside state/articulation.
                    candidate = state / "articulation" / "canonical" / Path(rel).name
                if not candidate.exists():
                    missing_templates += 1
    if missing_templates:
        raise RuntimeError(f"State articulation dictionary is incomplete: {missing_templates} templates missing.")

    fp = state / FINGERPRINT_FILE
    if verify_hashes and fp.exists():
        data = _json(fp)
        tracked = data.get("files") or {}
        stat_signature = []
        for rel in sorted(tracked):
            p = state / rel
            if not p.is_file():
                raise RuntimeError(f"Pinned state file disappeared: {rel}")
            st = p.stat()
            stat_signature.append((rel, st.st_size, st.st_mtime_ns))
        key = (str(state.resolve()), fp.stat().st_mtime_ns, tuple(stat_signature))
        cached = _VERIFY_CACHE.get(key)
        if cached is True:
            return True
        for rel, expected in tracked.items():
            p = state / rel
            actual = sha256(p)
            if actual != expected.get("sha256"):
                raise RuntimeError(f"Pinned state file changed unexpectedly: {rel}")
        if len(_VERIFY_CACHE) > 32:
            _VERIFY_CACHE.clear()
        _VERIFY_CACHE[key] = True
    return True


def write_fingerprint(state, reason, acoustic_base="rc3.2"):
    state = Path(state)
    validate_state(state, verify_hashes=False)
    files = {}
    for path in _critical_files(state):
        rel = path.relative_to(state).as_posix()
        files[rel] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    payload = {
        "format": 1,
        "engine_version": ENGINE_VERSION,
        "acoustic_base": acoustic_base,
        "reason": reason,
        "created_at": time.time(),
        "files": files,
    }
    atomic_write_json(state / FINGERPRINT_FILE, payload)
    return payload


def _candidate_from_active(bank, name):
    if not name:
        return None
    p = generation_dir(bank, name)
    if not p.is_dir():
        return None
    try:
        validate_state(p, verify_hashes=True)
        return p
    except Exception:
        return None


def _resolve_generation_container(bank, container_name, verify=True, source_label="generation"):
    bank = Path(bank).expanduser().resolve()
    container = bank / container_name
    active = container / ACTIVE_FILE
    if active.is_file():
        try:
            data = _json(active)
        except Exception:
            data = {}
        for key in ("generation", "previous_generation"):
            name = data.get(key)
            if not name:
                continue
            p = container / GENERATIONS_DIR / name
            if not p.is_dir():
                continue
            try:
                validate_state(p, verify_hashes=verify)
                return p, {"source": source_label, "generation": name, "active": key == "generation"}
            except Exception:
                continue
    if container.is_dir() and (container / "manifest.json").is_file():
        try:
            validate_state(container, verify_hashes=verify)
            return container, {"source": source_label + "-direct", "generation": "direct", "active": True}
        except Exception:
            pass
    return None, {"source": None, "generation": None, "active": False}

def resolve_ai_state(bank, verify=True):
    return _resolve_generation_container(bank, STATE_CONTAINER, verify=verify, source_label="0.2.8ai.13")

def resolve_previous_028ai12_state(bank, verify=True):
    return _resolve_generation_container(bank, PREVIOUS_028AI12_STATE_CONTAINER, verify=verify, source_label="0.2.8ai.12")

def resolve_previous_028ai11_state(bank, verify=True):
    return _resolve_generation_container(bank, PREVIOUS_028AI11_STATE_CONTAINER, verify=verify, source_label="0.2.8ai.11")

def resolve_previous_028ai10_state(bank, verify=True):
    return _resolve_generation_container(bank, PREVIOUS_028AI10_STATE_CONTAINER, verify=verify, source_label="0.2.8ai.10")

def resolve_previous_028ai9_state(bank, verify=True):
    return _resolve_generation_container(bank, PREVIOUS_028AI9_STATE_CONTAINER, verify=verify, source_label="0.2.8ai.9")

def resolve_previous_028ai8_state(bank, verify=True):
    return _resolve_generation_container(bank, PREVIOUS_028AI8_STATE_CONTAINER, verify=verify, source_label="0.2.8ai.8")

def resolve_previous_028ai7_state(bank, verify=True):
    return _resolve_generation_container(bank, PREVIOUS_028AI7_STATE_CONTAINER, verify=verify, source_label="0.2.8ai.7")

def resolve_previous_028ai6_state(bank, verify=True):
    return _resolve_generation_container(bank, PREVIOUS_028AI6_STATE_CONTAINER, verify=verify, source_label="0.2.8ai.6")

def resolve_previous_028ai5_state(bank, verify=True):
    return _resolve_generation_container(bank, PREVIOUS_028AI5_STATE_CONTAINER, verify=verify, source_label="0.2.8ai.5")

def resolve_previous_028ai4_state(bank, verify=True):
    return _resolve_generation_container(bank, PREVIOUS_028AI4_STATE_CONTAINER, verify=verify, source_label="0.2.8ai.4")

def resolve_previous_028ai3_state(bank, verify=True):
    return _resolve_generation_container(bank, PREVIOUS_028AI3_STATE_CONTAINER, verify=verify, source_label="0.2.8ai.3")

def resolve_previous_028ai2_state(bank, verify=True):
    return _resolve_generation_container(bank, PREVIOUS_028AI2_STATE_CONTAINER, verify=verify, source_label="0.2.8ai.2")

def resolve_previous_028ai1_state(bank, verify=True):
    return _resolve_generation_container(bank, PREVIOUS_028AI1_STATE_CONTAINER, verify=verify, source_label="0.2.8ai.1")

def resolve_previous_028_state(bank, verify=True):
    return _resolve_generation_container(bank, PREVIOUS_028_STATE_CONTAINER, verify=verify, source_label="0.2.8ai")

def resolve_predecessor_ai_state(bank, verify=True):
    return _resolve_generation_container(bank, PREDECESSOR_AI_STATE_CONTAINER, verify=verify, source_label="0.2.7-ai3")

def resolve_stable_state(bank, verify=True):
    return _resolve_generation_container(bank, STABLE_STATE_CONTAINER, verify=verify, source_label="rc4.2-stable")

def resolve_active_state(bank, allow_legacy=True, verify=True):
    # Predecessor states are read-only fallbacks until the current namespace is prepared.
    p, info = resolve_ai_state(bank, verify=verify)
    if p is not None:
        return p, info
    p, info = resolve_previous_028ai12_state(bank, verify=verify)
    if p is not None:
        info = dict(info)
        info["read_only_fallback"] = True
        info["predecessor_028ai12"] = True
        return p, info
    p, info = resolve_previous_028ai11_state(bank, verify=verify)
    if p is not None:
        info = dict(info)
        info["read_only_fallback"] = True
        info["predecessor_028ai11"] = True
        return p, info
    p, info = resolve_previous_028ai10_state(bank, verify=verify)
    if p is not None:
        info = dict(info)
        info["read_only_fallback"] = True
        info["predecessor_028ai10"] = True
        return p, info
    p, info = resolve_previous_028ai9_state(bank, verify=verify)
    if p is not None:
        info = dict(info)
        info["read_only_fallback"] = True
        info["predecessor_028ai9"] = True
        return p, info
    p, info = resolve_previous_028ai8_state(bank, verify=verify)
    if p is not None:
        info = dict(info)
        info["read_only_fallback"] = True
        info["predecessor_028ai8"] = True
        return p, info
    p, info = resolve_previous_028ai7_state(bank, verify=verify)
    if p is not None:
        info = dict(info)
        info["read_only_fallback"] = True
        info["predecessor_028ai7"] = True
        return p, info
    p, info = resolve_previous_028ai6_state(bank, verify=verify)
    if p is not None:
        info = dict(info)
        info["read_only_fallback"] = True
        info["predecessor_028ai6"] = True
        return p, info
    p, info = resolve_previous_028ai5_state(bank, verify=verify)
    if p is not None:
        info = dict(info)
        info["read_only_fallback"] = True
        info["predecessor_028ai5"] = True
        return p, info
    p, info = resolve_previous_028ai4_state(bank, verify=verify)
    if p is not None:
        info = dict(info)
        info["read_only_fallback"] = True
        info["predecessor_028ai4"] = True
        return p, info
    p, info = resolve_previous_028ai3_state(bank, verify=verify)
    if p is not None:
        info = dict(info)
        info["read_only_fallback"] = True
        info["predecessor_028ai3"] = True
        return p, info
    p, info = resolve_previous_028ai2_state(bank, verify=verify)
    if p is not None:
        info = dict(info)
        info["read_only_fallback"] = True
        info["predecessor_028ai2"] = True
        return p, info
    p, info = resolve_previous_028ai1_state(bank, verify=verify)
    if p is not None:
        info = dict(info)
        info["read_only_fallback"] = True
        info["predecessor_028ai1"] = True
        return p, info
    p, info = resolve_previous_028_state(bank, verify=verify)
    if p is not None:
        info = dict(info)
        info["read_only_fallback"] = True
        info["predecessor_028"] = True
        return p, info
    p, info = resolve_predecessor_ai_state(bank, verify=verify)
    if p is not None:
        info = dict(info)
        info["read_only_fallback"] = True
        info["predecessor_ai"] = True
        return p, info
    p, info = resolve_stable_state(bank, verify=verify)
    if p is not None:
        info = dict(info)
        info["read_only_fallback"] = True
        return p, info
    if allow_legacy:
        legacy = legacy_dir(bank)
        if legacy.is_dir() and (legacy / "manifest.json").is_file():
            try:
                validate_state(legacy, verify_hashes=False)
                return legacy, {"source": "rc3.2-legacy", "generation": "legacy-rc3.2", "active": True, "read_only_fallback": True}
            except Exception:
                pass
    return None, {"source": None, "generation": None, "active": False}

def begin_generation(bank, reason):
    bank = Path(bank).expanduser().resolve()
    root = container_dir(bank) / GENERATIONS_DIR
    root.mkdir(parents=True, exist_ok=True)
    generation = time.strftime("%Y%m%d-%H%M%S") + f"-{reason}-{uuid.uuid4().hex[:8]}"
    staging = root / f".staging-{generation}"
    staging.mkdir(parents=True, exist_ok=False)
    return generation, staging


def _copy_tree_item(src, dst):
    src = Path(src)
    dst = Path(dst)
    if src.is_symlink():
        dst.symlink_to(os.readlink(src), target_is_directory=src.is_dir())
    elif src.is_dir():
        shutil.copytree(src, dst, symlinks=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def clone_state(source, staging, link_caches=True, skip_caches=False):
    source = Path(source)
    staging = Path(staging)
    cache_names = {"cache", "highband_cache", "highband_cache_v2", "highband_cache_v3"}
    for child in source.iterdir():
        if child.name in {FINGERPRINT_FILE, RUNTIME_REGISTRY}:
            continue
        target = staging / child.name
        if skip_caches and child.name in cache_names:
            continue
        if child.name in cache_names and child.exists():
            if link_caches:
                target.symlink_to(child.resolve(), target_is_directory=True)
            else:
                resolved = child.resolve() if child.is_symlink() else child
                if resolved.is_dir():
                    shutil.copytree(resolved, target, symlinks=False)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(resolved, target)
        else:
            _copy_tree_item(child, target)
    return staging


def link_analysis_caches(source, staging):
    if source is None:
        return
    source = Path(source)
    staging = Path(staging)
    for name in ("cache", "highband_cache", "highband_cache_v2", "highband_cache_v3"):
        src = source / name
        dst = staging / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve(), target_is_directory=True)



def _rewrite_manifest_generation_paths(bank, staging, final):
    """Rewrite transient staging cache paths to the immutable generation path."""
    bank = Path(bank).resolve()
    staging = Path(staging).resolve()
    final = Path(final).resolve()
    manifest_path = staging / "manifest.json"
    if not manifest_path.is_file():
        return
    data = _json(manifest_path)
    old_rel = staging.relative_to(bank).as_posix()
    new_rel = final.relative_to(bank).as_posix()
    old_abs = str(staging)
    new_abs = str(final)
    def rewrite(value):
        if isinstance(value, str):
            if value.startswith(old_rel + "/"):
                return new_rel + value[len(old_rel):]
            if value.startswith(old_abs + os.sep):
                return new_abs + value[len(old_abs):]
            return value
        if isinstance(value, list):
            return [rewrite(x) for x in value]
        if isinstance(value, dict):
            return {k: rewrite(v) for k, v in value.items()}
        return value
    atomic_write_json(manifest_path, rewrite(data))

def commit_generation(bank, generation, staging, reason, acoustic_base="rc3.2"):
    bank = Path(bank).expanduser().resolve()
    staging = Path(staging)
    final = generation_dir(bank, generation)
    if final.exists():
        raise RuntimeError(f"Generation already exists: {final}")
    _rewrite_manifest_generation_paths(bank, staging, final)
    # Fingerprint after every transient path has been normalized.
    write_fingerprint(staging, reason=reason, acoustic_base=acoustic_base)
    validate_state(staging, verify_hashes=True)
    os.replace(staging, final)
    active_path = container_dir(bank) / ACTIVE_FILE
    previous = None
    if active_path.exists():
        try:
            previous = _json(active_path).get("generation")
        except Exception:
            previous = None
    payload = {
        "format": 1,
        "engine_version": ENGINE_VERSION,
        "acoustic_base": acoustic_base,
        "generation": generation,
        "previous_generation": previous,
        "committed_at": time.time(),
        "reason": reason,
    }
    atomic_write_json(active_path, payload)
    return final, payload



def rollback_to_previous(bank):
    bank = Path(bank).expanduser().resolve()
    active_path = container_dir(bank) / ACTIVE_FILE
    if not active_path.is_file():
        raise RuntimeError("No RC3.3 ACTIVE.json exists for this voicebank.")
    data = _json(active_path)
    current = data.get("generation")
    previous = data.get("previous_generation")
    if not previous:
        raise RuntimeError("No previous 0.2.8ai.13 generation is available to roll back to.")
    target = generation_dir(bank, previous)
    validate_state(target, verify_hashes=True)
    payload = {
        "format": 1,
        "engine_version": ENGINE_VERSION,
        "acoustic_base": data.get("acoustic_base", "rc3.2-highband-articulation-clarity"),
        "generation": previous,
        "previous_generation": current,
        "committed_at": time.time(),
        "reason": "manual-generation-rollback",
    }
    atomic_write_json(active_path, payload)
    return target, payload

def _state_loudness(state):
    path = Path(state) / "loudness.json"
    if not path.exists():
        return {}
    try:
        return _json(path)
    except Exception:
        return {}


def build_registry_payload(bank, state):
    bank = Path(bank).expanduser().resolve()
    state = Path(state).expanduser().resolve()
    manifest = _json(state / "manifest.json")
    entries = manifest.get("entries") or []
    profile = manifest.get("profile") or {}
    bank_id = profile.get("voicebank_id") or voicebank_id(bank)
    loud = _state_loudness(state)
    adapter = state / "adapter.pt"
    refiner = state / "fidelity_refiner.pt"
    ai_control = state / "ai_control_adapter.pt"
    ai_gender = state / "ai_gender_adapter.pt"
    ai_phonation = state / "ai_phonation_adapter.pt"
    ai_mouth = state / "ai_mouth_adapter.pt"
    highband = state / "highband_profiles_v3.json"
    highband_foundation = state / "highband_foundation.pt"
    articulation_index = state / "articulation" / "index.json"
    training = {}
    if (state / "training.json").exists():
        try:
            training = _json(state / "training.json")
        except Exception:
            training = {}
    generation = state.name
    payload = {"format": 5, "voicebank_id": bank_id, "voicebank_root": str(bank), "state_generation": generation, "samples": {}}
    samples = payload["samples"]

    def add_record(key, item):
        if not key:
            return
        record = samples.get(key)
        if not isinstance(record, dict):
            record = {
                "voicebank_id": bank_id,
                "voicebank_root": str(bank),
                "state_generation": generation,
                "state_path": str(state),
                "profile": str(state / "profile.json"),
                "adapter": str(adapter) if adapter.exists() else "",
                "refiner": str(refiner) if refiner.exists() else "",
                "ai_control_adapter": str(ai_control) if ai_control.exists() else "",
                "ai_gender_adapter": str(ai_gender) if ai_gender.exists() else "",
                "ai_phonation_adapter": str(ai_phonation) if ai_phonation.exists() else "",
                "ai_mouth_adapter": str(ai_mouth) if ai_mouth.exists() else "",
                "ai_control_backend": "ai-ddsp" if ai_control.exists() else "deterministic-fallback",
                "ai_gender_backend": "ai-ddsp" if ai_gender.exists() else "deterministic-fallback",
                "loudness": str(state / "loudness.json") if (state / "loudness.json").exists() else "",
                "articulation_index": str(articulation_index) if articulation_index.exists() else "",
                "highband_profiles": str(highband) if highband.exists() else "",
                "highband_foundation": str(highband_foundation) if highband_foundation.exists() else "",
                "highband_profile_format": 3 if highband.exists() else 0,
                "loudness_enabled": bool(loud.get("enabled", profile.get("loudness_normalization_enabled", True))),
                "loudness_target_dbfs": float(loud.get("target_active_rms_dbfs", profile.get("loudness_target_active_rms_dbfs", -18.0))),
                "loudness_peak_ceiling_dbfs": float(loud.get("peak_ceiling_dbfs", -1.0)),
                "loudness_peak_guard_knee_db": float(loud.get("peak_guard_knee_db", 3.0)),
                "loudness_emergency_max_abs_gain_db": float(loud.get("emergency_max_abs_gain_db", 30.0)),
                "loudness_tolerance_db": float(loud.get("tolerance_db", 0.05)),
                "adapter_generation": training.get("adapter_generation", "alpha8-rc3-3-two-stage-clarity") if adapter.exists() else "base",
                "subbank_id": item.get("subbank_id"),
                "subbank_index": item.get("subbank_index"),
                "subbank_label": item.get("subbank_label"),
                "subbank_anchor_midi": item.get("subbank_anchor_midi"),
                "base_alias": item.get("base_alias"),
                "source_loudness_variants": [],
            }
        variants = record.setdefault("source_loudness_variants", [])
        signature = item.get("loudness_signature") or oto_loudness_signature(
            item.get("offset", 0.0), item.get("consonant", 0.0), item.get("cutoff", 0.0)
        )
        variants[:] = [v for v in variants if v.get("signature") != signature]
        canon = item.get("canonical_articulation")
        if canon:
            # Generation states keep canonical files inside their own articulation directory.
            candidate = state / "articulation" / "canonical" / Path(canon).name
            canon = str(candidate) if candidate.exists() else canon
        variants.append({
            "signature": signature,
            "alias": item.get("alias"),
            "offset": float(item.get("offset", 0.0)),
            "consonant": float(item.get("consonant", 0.0)),
            "cutoff": float(item.get("cutoff", 0.0)),
            "active_rms_dbfs": float(item.get("active_rms_dbfs", -120.0)),
            "diagnostic_gain_to_target_db": float(item.get("diagnostic_gain_to_target_db", 0.0)),
            "base_alias": item.get("base_alias"),
            "subbank_label": item.get("subbank_label"),
            "subbank_index": item.get("subbank_index"),
            "canonical_articulation": canon,
            "canonical_articulation_source": item.get("canonical_articulation_source"),
            "canonical_articulation_coherence": item.get("canonical_articulation_coherence"),
        })
        samples[key] = record

    for item in entries:
        if item.get("status") == "error":
            continue
        if item.get("sha256"):
            add_record("sha256:" + item["sha256"], item)
        if item.get("pcm"):
            add_record("pcm:" + item["pcm"], item)
    payload["updated_at"] = time.time()
    return payload


def write_local_registry(bank, state):
    payload = build_registry_payload(bank, state)
    atomic_write_json(Path(state) / RUNTIME_REGISTRY, payload)
    return payload


def merge_global_registry(global_path, payload):
    global_path = Path(global_path).expanduser().resolve()
    global_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = global_path.with_suffix(global_path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        current = {"format": 5, "samples": {}}
        if global_path.exists():
            try:
                current = _json(global_path)
            except Exception:
                current = {"format": 5, "samples": {}}
        samples = current.setdefault("samples", {})
        bank_id = payload.get("voicebank_id")
        bank_root = payload.get("voicebank_root")
        for key in list(samples):
            value = samples.get(key)
            if isinstance(value, dict) and (value.get("voicebank_id") == bank_id or value.get("voicebank_root") == bank_root):
                samples.pop(key, None)
        samples.update(payload.get("samples") or {})
        current["format"] = 5
        current["updated_at"] = time.time()
        atomic_write_json(global_path, current)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return global_path


def find_voicebank_for_input(input_path, max_levels=10):
    p = Path(input_path).expanduser().resolve()
    cur = p.parent
    for _ in range(max_levels):
        if any((cur / x).exists() for x in (STATE_CONTAINER, PREVIOUS_028AI12_STATE_CONTAINER, PREVIOUS_028AI11_STATE_CONTAINER, PREVIOUS_028AI10_STATE_CONTAINER, PREVIOUS_028AI9_STATE_CONTAINER, PREVIOUS_028AI8_STATE_CONTAINER, PREVIOUS_028AI7_STATE_CONTAINER, PREVIOUS_028AI6_STATE_CONTAINER, PREVIOUS_028AI5_STATE_CONTAINER, PREVIOUS_028AI4_STATE_CONTAINER, PREVIOUS_028AI3_STATE_CONTAINER, PREVIOUS_028AI2_STATE_CONTAINER, PREVIOUS_028AI1_STATE_CONTAINER, PREVIOUS_028_STATE_CONTAINER, PREDECESSOR_AI_STATE_CONTAINER, STABLE_STATE_CONTAINER, LEGACY_STATE)):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def _registry_for_state(bank, state, read_only=False):
    state = Path(state)
    local = state / RUNTIME_REGISTRY
    marker = local.stat().st_mtime_ns if local.exists() else (state / "manifest.json").stat().st_mtime_ns
    key = (str(state.resolve()), marker)
    if key in _LOCAL_CACHE:
        return _LOCAL_CACHE[key]
    payload = None
    if local.exists():
        try:
            candidate = _json(local)
            if (
                isinstance(candidate, dict)
                and candidate.get("state_generation") == state.name
                and Path(candidate.get("voicebank_root", "")).expanduser().resolve() == Path(bank).resolve()
            ):
                payload = candidate
        except Exception:
            payload = None
    if payload is None:
        payload = build_registry_payload(bank, state)
        # This is a cache only. A predecessor/stable fallback is strictly read-only:
        # build an in-memory registry but never write runtime_registry.json into it.
        if not read_only:
            try:
                atomic_write_json(local, payload)
            except Exception:
                pass
    if len(_LOCAL_CACHE) > 16:
        _LOCAL_CACHE.clear()
    _LOCAL_CACHE[key] = payload
    return payload


def lookup_local_record(input_path):
    bank = find_voicebank_for_input(input_path)
    if bank is None:
        return None
    state, info = resolve_active_state(bank, allow_legacy=True, verify=True)
    if state is None:
        raise RuntimeError(
            f"Input belongs to a Yuaz-prepared voicebank but no valid pinned state can be resolved: {bank}"
        )
    payload = _registry_for_state(bank, state, read_only=bool(info.get("read_only_fallback", False)))
    samples = payload.get("samples", {}) if isinstance(payload, dict) else {}
    hash_errors = []
    try:
        key = "sha256:" + file_sha256(input_path)
        record = samples.get(key)
        if record:
            return record
    except Exception as exc:
        hash_errors.append(str(exc))
    try:
        key = "pcm:" + pcm_fingerprint(input_path)
        record = samples.get(key)
        if record:
            return record
    except Exception as exc:
        hash_errors.append(str(exc))
    detail = f" ({'; '.join(hash_errors)})" if hash_errors else ""
    raise RuntimeError(
        "Prepared voicebank state is valid, but this source WAV is not present in its pinned manifest. "
        "Refusing generic fallback because that could silently change the rendered voice. "
        "Run prepare-voicebank.command to create a new validated generation." + detail
    )
