#!/usr/bin/env python3
import math
import random

import numpy as np

from . import train_high_detail_router as base


def _safe_median_f0(engine, entry):
    try:
        audio = base.read_audio(entry.wav_path, engine.sr)
        audio = base.crop_oto(audio, engine.sr, entry.offset, entry.cutoff)
        if len(audio) < int(0.08 * engine.sr):
            return 0.0
        try:
            f0 = base.extract_f0(audio, engine.sr, engine.hop)
        except RuntimeError:
            return 0.0
        voiced = np.asarray(f0, dtype=np.float32)
        voiced = voiced[np.isfinite(voiced) & (voiced > 1.0)]
        return float(np.median(voiced)) if voiced.size else 0.0
    except Exception:
        return 0.0


def _anchor_hz(item):
    midi = item.get("subbank_anchor_midi")
    if midi is None:
        return 0.0
    try:
        midi = float(midi)
    except Exception:
        return 0.0
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _groups_from_manifest(manifest):
    groups = {}
    for item in manifest:
        base_alias = str(item.get("base_alias") or item.get("alias") or "").strip()
        subbank = int(item.get("subbank_index", -1))
        if not base_alias or subbank < 0:
            continue
        groups.setdefault(base_alias, []).append(item)
    return groups


def _structural_pair_aliases(groups):
    return [
        alias for alias, items in groups.items()
        if len({int(x.get("subbank_index", -1)) for x in items if int(x.get("subbank_index", -1)) >= 0}) >= 2
    ]


def _choose_rep(engine, items, stats):
    items = list(items)
    if not items:
        return None
    anchor = _anchor_hz(items[0])
    ranked = []
    for item in items[:4]:
        entry = stats["entry_by_key"][int(item["entry_key"])]
        f0 = _safe_median_f0(engine, entry)
        stats["f0_scanned"] += 1
        if f0 > 0.0:
            item["median_f0_hz"] = float(f0)
            item["pitch_source"] = "observed_f0"
            distance = abs(math.log2(max(f0, 1.0) / max(anchor, f0, 1.0))) if anchor > 0 else 0.0
            ranked.append((distance, item))
        else:
            stats["f0_unreliable"] += 1
    if ranked:
        ranked.sort(key=lambda x: x[0])
        return ranked[0][1]
    if anchor > 0.0:
        item = items[0]
        item["median_f0_hz"] = float(anchor)
        item["pitch_source"] = "subbank_anchor_fallback"
        stats["anchor_fallback"] += 1
        return item
    return None


def _build_candidates(engine, groups, aliases, stats, pool_target):
    rng = random.Random(20260902)
    aliases = list(aliases)
    rng.shuffle(aliases)
    candidates = []
    aliases_used = 0

    for alias in aliases:
        by_subbank = {}
        for item in groups[alias]:
            idx = int(item.get("subbank_index", -1))
            if idx >= 0:
                by_subbank.setdefault(idx, []).append(item)

        reps = []
        for idx in sorted(by_subbank):
            rep = _choose_rep(engine, by_subbank[idx], stats)
            if rep is not None and float(rep.get("median_f0_hz", 0.0) or 0.0) > 0.0:
                reps.append(rep)
        reps.sort(key=lambda x: float(x.get("median_f0_hz", 0.0)))
        if len(reps) < 2:
            continue

        local = []
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                a = reps[i]
                b = reps[j]
                fa = float(a["median_f0_hz"])
                fb = float(b["median_f0_hz"])
                if fa <= 0.0 or fb <= 0.0:
                    continue
                semitones = abs(12.0 * math.log2(fb / fa))
                if semitones < 1.5:
                    continue
                local.append((semitones, a, b))
        local.sort(key=lambda x: x[0], reverse=True)
        if not local:
            continue

        aliases_used += 1
        for semitones, a, b in local[:2]:
            candidates.append({"alias": alias, "source": a, "target": b, "semitones": semitones})
            candidates.append({"alias": alias, "source": b, "target": a, "semitones": semitones})
        if pool_target > 0 and len(candidates) >= pool_target:
            break

    # Keep alias diversity first, then favor wider pitch jumps inside the sampled pool.
    rng.shuffle(candidates)
    candidates.sort(key=lambda x: float(x["semitones"]), reverse=True)
    return candidates, aliases_used


def build_multipitch_pairs_fast(engine, voicebank, limit):
    scan = base.scan_voicebank(voicebank)
    entries = list(scan["entries"])
    if not entries:
        raise RuntimeError("No usable OTO entries found.")

    manifest = []
    entry_by_key = {}
    for i, entry in enumerate(entries):
        manifest.append({
            "status": "ok",
            "relative_wav": entry.relative_wav,
            "alias": entry.alias,
            "median_f0_hz": 0.0,
            "entry_key": i,
        })
        entry_by_key[i] = entry

    print(f"Indexing pitch structure for {len(entries)} OTO entries...", flush=True)
    subbanks = base.annotate_utau_subbanks(voicebank, manifest, scan["prefix_map"])
    groups = _groups_from_manifest(manifest)
    aliases = _structural_pair_aliases(groups)

    stats = {
        "entry_by_key": entry_by_key,
        "f0_scanned": 0,
        "f0_unreliable": 0,
        "anchor_fallback": 0,
    }

    # Normally multipitch UTAU banks expose enough structure through prefix.map or
    # pitch folders that we only need to analyze entries which can actually form
    # cross-pitch same-alias pairs.
    if aliases:
        print(
            f"Found {len(aliases)} same-alias multipitch groups from voicebank structure; "
            "scanning F0 only for pair candidates.",
            flush=True,
        )
        pool_target = max(96, int(limit) * 4) if int(limit) > 0 else 0
        candidates, aliases_used = _build_candidates(
            engine, groups, aliases, stats, pool_target
        )
    else:
        # Structure-less fallback: robustly scan every entry. Unvoiced/noisy OTOs
        # are expected and simply get median_f0_hz=0 instead of aborting training.
        print(
            "No structural multipitch index was found; falling back to safe F0 scan.",
            flush=True,
        )
        for i, item in enumerate(manifest, 1):
            entry = entry_by_key[int(item["entry_key"])]
            f0 = _safe_median_f0(engine, entry)
            stats["f0_scanned"] += 1
            if f0 > 0.0:
                item["median_f0_hz"] = float(f0)
                item["pitch_source"] = "observed_f0"
            else:
                stats["f0_unreliable"] += 1
            if i == 1 or i % 250 == 0 or i == len(manifest):
                print(
                    f"safe pitch scan [{i}/{len(manifest)}] "
                    f"unreliable={stats['f0_unreliable']}",
                    flush=True,
                )
        subbanks = base.annotate_utau_subbanks(voicebank, manifest, scan["prefix_map"])
        groups = _groups_from_manifest(manifest)
        aliases = _structural_pair_aliases(groups)
        candidates, aliases_used = _build_candidates(engine, groups, aliases, stats, 0)

    if not candidates:
        raise RuntimeError(
            "No usable same-alias multipitch pairs were found after safe pitch indexing."
        )

    if int(limit) > 0:
        candidates = candidates[: min(int(limit), len(candidates))]
    for pair in candidates:
        pair["source_entry"] = entry_by_key[int(pair["source"]["entry_key"])]
        pair["target_entry"] = entry_by_key[int(pair["target"]["entry_key"])]

    print(
        f"Pair index ready: pairs={len(candidates)} aliases={aliases_used} "
        f"F0_scanned={stats['f0_scanned']} unreliable={stats['f0_unreliable']} "
        f"anchor_fallback={stats['anchor_fallback']}",
        flush=True,
    )
    return candidates, {
        "oto_entries": len(entries),
        "subbanks": int(subbanks.get("prototype_count", 0)),
        "aliases_with_pitch_pairs": len(aliases),
        "candidate_pairs": len(candidates),
        "f0_scanned": int(stats["f0_scanned"]),
        "f0_unreliable": int(stats["f0_unreliable"]),
        "anchor_fallback": int(stats["anchor_fallback"]),
        "pair_index_mode": "structural-first-safe-f0-v2",
    }


def _safe_prepare_base(engine, source_entry, target_entry):
    try:
        return _ORIGINAL_PREPARE_BASE(engine, source_entry, target_entry)
    except RuntimeError as exc:
        text = str(exc)
        if "没有检测到可靠 F0" in text or "reliable F0" in text:
            return None
        raise


_ORIGINAL_PREPARE_BASE = base._prepare_base
base._median_f0 = _safe_median_f0
base.build_multipitch_pairs = build_multipitch_pairs_fast
base._prepare_base = _safe_prepare_base


if __name__ == "__main__":
    base.main()
