#!/usr/bin/env python3
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "cp932", "shift_jis", "gb18030")
NOTE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Ga-g])([#b♯♭]?)(-?\d{1,2})(?!\d)")


@dataclass
class OtoEntry:
    oto_path: str
    wav_path: str
    relative_wav: str
    alias: str
    offset: float
    consonant: float
    cutoff: float
    preutterance: float
    overlap: float


def read_text_fallback(path):
    raw = Path(path).read_bytes()
    for encoding in TEXT_ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8-replace"


def _float(value, default=0.0):
    try:
        return float(str(value).strip())
    except Exception:
        return float(default)


def parse_oto_file(path, voicebank_root):
    path = Path(path)
    voicebank_root = Path(voicebank_root)
    text, encoding = read_text_fallback(path)
    entries = []
    malformed = []
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip().lstrip("\ufeff")
        if not line or line.startswith("#") or "=" not in line:
            continue
        wav_name, rhs = line.split("=", 1)
        fields = rhs.split(",")
        while len(fields) < 6:
            fields.append("")
        alias = fields[0].strip() or Path(wav_name.strip()).stem
        wav_path = (path.parent / wav_name.strip()).resolve()
        entry = OtoEntry(
            oto_path=str(path.resolve()),
            wav_path=str(wav_path),
            relative_wav=os.path.relpath(wav_path, voicebank_root),
            alias=alias,
            offset=_float(fields[1]),
            consonant=_float(fields[2]),
            cutoff=_float(fields[3]),
            preutterance=_float(fields[4]),
            overlap=_float(fields[5]),
        )
        if wav_path.exists():
            entries.append(entry)
        else:
            malformed.append({"line": line_no, "wav": str(wav_path), "reason": "missing_wav"})
    return entries, malformed, encoding


def scan_voicebank(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    oto_files = sorted(root.rglob("oto.ini"))
    if not oto_files:
        raise RuntimeError("No oto.ini files were found under the selected voicebank.")
    entries = []
    malformed = []
    encodings = {}
    for oto in oto_files:
        parsed, bad, enc = parse_oto_file(oto, root)
        entries.extend(parsed)
        malformed.extend({"oto": str(oto), **item} for item in bad)
        encodings[str(oto.relative_to(root))] = enc
    unique = {}
    for entry in entries:
        key = (entry.wav_path, entry.alias, entry.offset, entry.cutoff)
        unique.setdefault(key, entry)
    entries = list(unique.values())
    return {
        "root": root,
        "oto_files": oto_files,
        "entries": entries,
        "malformed": malformed,
        "encodings": encodings,
        "prefix_map": parse_prefix_map(root / "prefix.map"),
    }


def parse_prefix_map(path):
    path = Path(path)
    if not path.exists():
        return []
    text, encoding = read_text_fallback(path)
    rows = []
    for raw in text.splitlines():
        parts = raw.rstrip("\r\n").split("\t")
        while len(parts) < 3:
            parts.append("")
        if parts[0].strip():
            rows.append({"tone": parts[0].strip(), "prefix": parts[1], "suffix": parts[2]})
    return rows


def note_to_midi(value):
    m = re.fullmatch(r"\s*([A-Ga-g])([#b♯♭]?)(-?\d{1,2})\s*", str(value))
    if not m:
        return None
    note = m.group(1).upper()
    accidental = m.group(2)
    octave = int(m.group(3))
    pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[note]
    if accidental in ("#", "♯"):
        pc += 1
    elif accidental in ("b", "♭"):
        pc -= 1
    return float((octave + 1) * 12 + pc)


def midi_to_note(midi):
    midi = int(round(float(midi)))
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[midi % 12]}{midi // 12 - 1}"


def hz_to_midi(hz):
    hz = float(hz)
    if hz <= 0:
        return None
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def _note_from_text(text):
    matches = list(NOTE_RE.finditer(str(text)))
    if not matches:
        return None, None
    m = matches[-1]
    note = f"{m.group(1).upper()}{m.group(2).replace('♯', '#').replace('♭', 'b')}{m.group(3)}"
    return note, note_to_midi(note)


def _prefix_groups(prefix_map):
    groups = {}
    for row in prefix_map or []:
        prefix = str(row.get("prefix", ""))
        suffix = str(row.get("suffix", ""))
        if not prefix and not suffix:
            continue
        midi = note_to_midi(row.get("tone", ""))
        key = (prefix, suffix)
        g = groups.setdefault(key, {"prefix": prefix, "suffix": suffix, "tones": [], "midis": []})
        g["tones"].append(str(row.get("tone", "")))
        if midi is not None:
            g["midis"].append(float(midi))
    for g in groups.values():
        affix_note, affix_midi = _note_from_text(g["prefix"] + " " + g["suffix"])
        if affix_midi is not None:
            g["anchor_midi"] = float(affix_midi)
            g["anchor_source"] = "affix_note"
            g["anchor_note"] = affix_note
        elif g["midis"]:
            g["anchor_midi"] = float(np.median(g["midis"]))
            g["anchor_source"] = "prefix_map_range"
            g["anchor_note"] = midi_to_note(g["anchor_midi"])
        else:
            g["anchor_midi"] = None
            g["anchor_source"] = "unknown"
            g["anchor_note"] = None
    return list(groups.values())


def _match_prefix_group(alias, groups):
    alias = str(alias)
    candidates = []
    for g in groups:
        prefix = g["prefix"]
        suffix = g["suffix"]
        if prefix and not alias.startswith(prefix):
            continue
        if suffix and not alias.endswith(suffix):
            continue
        candidates.append((len(prefix) + len(suffix), g))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _strip_affixes(alias, prefix, suffix):
    out = str(alias)
    if prefix and out.startswith(prefix):
        out = out[len(prefix):]
    if suffix and out.endswith(suffix):
        out = out[:-len(suffix)] if len(suffix) else out
    return out or str(alias)


def _note_from_folder_component(text):
    note, midi = _note_from_text(text)
    if midi is not None:
        return note, midi
    m = re.search(r"([A-Ga-g])([#b♯♭]?)(-?\d{1,2})$", str(text))
    if not m:
        return None, None
    note = f"{m.group(1).upper()}{m.group(2).replace('♯', '#').replace('♭', 'b')}{m.group(3)}"
    return note, note_to_midi(note)


def _folder_pitch(relative_wav):
    parent = Path(relative_wav).parent
    if str(parent) in ("", "."):
        return None
    parts = list(parent.parts)
    for part in reversed(parts):
        note, midi = _note_from_folder_component(part)
        if midi is not None:
            return {
                "folder": str(parent),
                "pitch_folder": part,
                "anchor_note": note,
                "anchor_midi": float(midi),
            }
    return {"folder": str(parent), "pitch_folder": None, "anchor_note": None, "anchor_midi": None}


def annotate_utau_subbanks(voicebank_root, manifest_entries, prefix_map):
    root = Path(voicebank_root).resolve()
    prefix_groups = _prefix_groups(prefix_map)
    valid_items = [x for x in manifest_entries if x.get("status") != "error"]

    if prefix_groups:
        group_by_key = {}
        for g in prefix_groups:
            key = f"affix:{g['prefix']}\x1f{g['suffix']}"
            group_by_key[key] = {**g, "id": key}

        folder_votes = {}
        for item in valid_items:
            alias = str(item.get("alias", ""))
            matched = _match_prefix_group(alias, prefix_groups)
            if matched is None:
                continue
            folder = str(Path(item.get("relative_wav", "")).parent)
            if folder in ("", "."):
                continue
            key = f"affix:{matched['prefix']}\x1f{matched['suffix']}"
            votes = folder_votes.setdefault(folder, {})
            votes[key] = votes.get(key, 0) + 1

        folder_choice = {}
        for folder, votes in folder_votes.items():
            folder_choice[folder] = max(votes.items(), key=lambda x: (x[1], x[0]))[0]

        anchored = [(key, g) for key, g in group_by_key.items() if g.get("anchor_midi") is not None]

        def nearest_key(midi):
            if midi is None or not anchored:
                return None
            return min(anchored, key=lambda x: abs(float(x[1]["anchor_midi"]) - float(midi)))[0]

        prelim = []
        fallback_assignments = 0
        unassigned = 0
        for item in valid_items:
            alias = str(item.get("alias", ""))
            matched = _match_prefix_group(alias, prefix_groups)
            folder = _folder_pitch(item.get("relative_wav", ""))
            folder_path = str(Path(item.get("relative_wav", "")).parent)
            assignment_source = None
            key = None

            if matched is not None:
                key = f"affix:{matched['prefix']}\x1f{matched['suffix']}"
                assignment_source = "prefix_map_alias"
            elif folder_path in folder_choice:
                key = folder_choice[folder_path]
                assignment_source = "prefix_map_folder"
                fallback_assignments += 1
            elif folder is not None and folder.get("anchor_midi") is not None:
                key = nearest_key(folder.get("anchor_midi"))
                if key is not None:
                    assignment_source = "prefix_map_folder_pitch"
                    fallback_assignments += 1
            else:
                f0 = float(item.get("median_f0_hz", 0.0) or 0.0)
                key = nearest_key(hz_to_midi(f0))
                if key is not None:
                    assignment_source = "prefix_map_observed_f0"
                    fallback_assignments += 1

            if key is None:
                unassigned += 1
                prelim.append({"item": item, "key": None, "base_alias": alias, "assignment_source": "unassigned"})
                continue

            g = group_by_key[key]
            prelim.append({
                "item": item,
                "key": key,
                "base_alias": _strip_affixes(alias, g["prefix"], g["suffix"]),
                "assignment_source": assignment_source,
            })

        groups = {}
        for row in prelim:
            if row["key"] is None:
                continue
            spec = group_by_key[row["key"]]
            g = groups.setdefault(row["key"], {
                "id": row["key"],
                "label": (spec["prefix"] + "*" + spec["suffix"]).strip("*") or spec.get("anchor_note") or "prefix-map",
                "source": "prefix_map",
                "prefix": spec["prefix"],
                "suffix": spec["suffix"],
                "folder_counts": {},
                "declared_midis": [],
                "observed_midis": [],
                "sample_count": 0,
                "fallback_sample_count": 0,
            })
            if spec.get("anchor_midi") is not None:
                g["declared_midis"].append(float(spec["anchor_midi"]))
            folder = str(Path(row["item"].get("relative_wav", "")).parent)
            if folder not in ("", "."):
                g["folder_counts"][folder] = g["folder_counts"].get(folder, 0) + 1
            f0 = float(row["item"].get("median_f0_hz", 0.0) or 0.0)
            midi = hz_to_midi(f0)
            if midi is not None:
                g["observed_midis"].append(float(midi))
            g["sample_count"] += 1
            if row["assignment_source"] != "prefix_map_alias":
                g["fallback_sample_count"] += 1

        subbanks = []
        for g in groups.values():
            if g["declared_midis"]:
                anchor_midi = float(np.median(g["declared_midis"]))
                anchor_source = "utau_structure"
            elif g["observed_midis"]:
                anchor_midi = float(np.median(g["observed_midis"]))
                anchor_source = "observed_f0_fallback"
            else:
                anchor_midi = 60.0
                anchor_source = "fallback"
            observed_midi = float(np.median(g["observed_midis"])) if g["observed_midis"] else None
            folder = max(g["folder_counts"].items(), key=lambda x: (x[1], x[0]))[0] if g["folder_counts"] else None
            subbanks.append({
                "id": g["id"],
                "label": g["label"],
                "source": g["source"],
                "prefix": g["prefix"],
                "suffix": g["suffix"],
                "folder": folder,
                "anchor_midi": anchor_midi,
                "anchor_note": midi_to_note(anchor_midi),
                "anchor_source": anchor_source,
                "observed_median_midi": observed_midi,
                "observed_median_note": midi_to_note(observed_midi) if observed_midi is not None else None,
                "sample_count": int(g["sample_count"]),
                "fallback_sample_count": int(g["fallback_sample_count"]),
            })

        subbanks.sort(key=lambda x: (float(x["anchor_midi"]), str(x["label"]), str(x["id"])))
        index_by_id = {g["id"]: i for i, g in enumerate(subbanks)}
        for row in prelim:
            item = row["item"]
            item["base_alias"] = row["base_alias"]
            item["subbank_assignment_source"] = row["assignment_source"]
            if row["key"] is None:
                item["subbank_id"] = None
                item["subbank_index"] = -1
                item["subbank_label"] = None
                item["subbank_source"] = "unassigned"
                item["subbank_anchor_midi"] = None
                item["subbank_anchor_note"] = None
                continue
            idx = int(index_by_id[row["key"]])
            proto = subbanks[idx]
            item["subbank_id"] = row["key"]
            item["subbank_index"] = idx
            item["subbank_label"] = proto["label"]
            item["subbank_source"] = proto["source"]
            item["subbank_anchor_midi"] = float(proto["anchor_midi"])
            item["subbank_anchor_note"] = proto["anchor_note"]

        return {
            "format": 2,
            "strategy": "utau_native_prefix_authoritative",
            "prototype_count": len(subbanks),
            "subbanks": subbanks,
            "prefix_map_group_count": len(prefix_groups),
            "uses_prefix_map": True,
            "prefix_map_authoritative": True,
            "fallback_created_prototypes": 0,
            "fallback_assignment_count": int(fallback_assignments),
            "unassigned_entry_count": int(unassigned),
            "voicebank_root": str(root),
        }

    prelim = []
    non_root_folders = {str(Path(x.get("relative_wav", "")).parent) for x in valid_items if str(Path(x.get("relative_wav", "")).parent) not in ("", ".")}
    for item in valid_items:
        alias = str(item.get("alias", ""))
        folder = _folder_pitch(item.get("relative_wav", ""))
        if folder is not None and folder.get("anchor_midi") is not None:
            key = f"folder:{folder['folder']}"
            label = folder["pitch_folder"] or folder["folder"]
            anchor_midi = float(folder["anchor_midi"])
            anchor_note = folder["anchor_note"]
            source = "pitch_folder"
            folder_name = folder["folder"]
        elif folder is not None and len(non_root_folders) > 1:
            key = f"folder:{folder['folder']}"
            label = Path(folder["folder"]).name or folder["folder"]
            anchor_midi = None
            anchor_note = None
            source = "oto_folder"
            folder_name = folder["folder"]
        else:
            key = "root"
            label = "root"
            anchor_midi = None
            anchor_note = None
            source = "single_bank"
            folder_name = None
        prelim.append({
            "item": item,
            "key": key,
            "label": label,
            "source": source,
            "declared_anchor_midi": anchor_midi,
            "declared_anchor_note": anchor_note,
            "base_alias": alias,
            "prefix": "",
            "suffix": "",
            "folder": folder_name,
        })

    groups = {}
    for row in prelim:
        g = groups.setdefault(row["key"], {
            "id": row["key"],
            "label": row["label"],
            "source": row["source"],
            "prefix": row["prefix"],
            "suffix": row["suffix"],
            "folder": row["folder"],
            "declared_midis": [],
            "observed_midis": [],
            "sample_count": 0,
        })
        if row["declared_anchor_midi"] is not None:
            g["declared_midis"].append(float(row["declared_anchor_midi"]))
        f0 = float(row["item"].get("median_f0_hz", 0.0) or 0.0)
        midi = hz_to_midi(f0)
        if midi is not None:
            g["observed_midis"].append(float(midi))
        g["sample_count"] += 1

    subbanks = []
    for g in groups.values():
        if g["declared_midis"]:
            anchor_midi = float(np.median(g["declared_midis"]))
            anchor_source = "utau_structure"
        elif g["observed_midis"]:
            anchor_midi = float(np.median(g["observed_midis"]))
            anchor_source = "observed_f0_fallback"
        else:
            anchor_midi = 60.0
            anchor_source = "fallback"
        observed_midi = float(np.median(g["observed_midis"])) if g["observed_midis"] else None
        subbanks.append({
            "id": g["id"],
            "label": g["label"],
            "source": g["source"],
            "prefix": g["prefix"],
            "suffix": g["suffix"],
            "folder": g["folder"],
            "anchor_midi": anchor_midi,
            "anchor_note": midi_to_note(anchor_midi),
            "anchor_source": anchor_source,
            "observed_median_midi": observed_midi,
            "observed_median_note": midi_to_note(observed_midi) if observed_midi is not None else None,
            "sample_count": int(g["sample_count"]),
        })
    subbanks.sort(key=lambda x: (float(x["anchor_midi"]), str(x["label"]), str(x["id"])))
    index_by_id = {g["id"]: i for i, g in enumerate(subbanks)}
    for row in prelim:
        item = row["item"]
        idx = int(index_by_id[row["key"]])
        proto = subbanks[idx]
        item["base_alias"] = row["base_alias"]
        item["subbank_id"] = row["key"]
        item["subbank_index"] = idx
        item["subbank_label"] = proto["label"]
        item["subbank_source"] = proto["source"]
        item["subbank_assignment_source"] = proto["source"]
        item["subbank_anchor_midi"] = float(proto["anchor_midi"])
        item["subbank_anchor_note"] = proto["anchor_note"]
    return {
        "format": 2,
        "strategy": "utau_native_fallback",
        "prototype_count": len(subbanks),
        "subbanks": subbanks,
        "prefix_map_group_count": 0,
        "uses_prefix_map": False,
        "prefix_map_authoritative": False,
        "fallback_created_prototypes": len(subbanks),
        "fallback_assignment_count": len(valid_items),
        "unassigned_entry_count": 0,
        "voicebank_root": str(root),
    }

def file_sha256(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def pcm_fingerprint(path):
    audio, sr = sf.read(path, always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    peak = max(1e-9, float(np.max(np.abs(audio))) if audio.size else 1.0)
    q = np.clip(np.round(audio / peak * 32767.0), -32768, 32767).astype("<i2")
    h = hashlib.sha256()
    h.update(str(int(sr)).encode())
    h.update(str(len(q)).encode())
    h.update(q.tobytes())
    return h.hexdigest()


def voicebank_id(root):
    root = Path(root).resolve()
    h = hashlib.sha256(str(root).encode("utf-8"))
    character = root / "character.txt"
    if character.exists():
        h.update(character.read_bytes())
    return h.hexdigest()[:16]


def cache_key(entry):
    payload = json.dumps(asdict(entry), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def entry_to_dict(entry):
    return asdict(entry)
