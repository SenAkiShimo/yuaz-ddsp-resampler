#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -x .venv/bin/python ]; then
  echo "Run scripts/setup-macos.command first."
  exit 1
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python - <<'PY'
import tempfile
from pathlib import Path

import numpy as np
import torch

from yuaz_ddsp_resampler.core import YuazDDSPResamplerEngine, decode_int12_pitch, tone_to_midi, crop_oto, build_target_f0, extract_detail_features, piecewise_warp
from yuaz_ddsp_resampler.articulation import (
    analyze_articulation_regions, apply_articulation_template, combine_canonical_articulation_templates,
    extract_neutral_articulation_template, load_canonical_articulation, map_articulation_regions,
    save_canonical_articulation, single_source_articulation_hybrid, transfer_articulation_trajectory,
)
from yuaz_ddsp_resampler.adapter import VoicebankAdapter, load_adapter, save_adapter
from yuaz_ddsp_resampler.controls import parse_yuaz_controls
from yuaz_ddsp_resampler.learned_highband import synthesize_learned_highband, select_learned_profile
from yuaz_ddsp_resampler.fidelity import TinyFidelityRefiner, load_refiner, save_refiner
from yuaz_ddsp_resampler.loudness import active_rms_dbfs, normalize_final_render
from yuaz_ddsp_resampler.prepare import timbre_perturb_audio, content_consistency_loss
from yuaz_ddsp_resampler.voicebank import annotate_utau_subbanks

assert tone_to_midi("C4") == 60
assert tone_to_midi("G4") == 67
assert tone_to_midi("C5") == 72
assert decode_int12_pitch("AA").tolist() == [0.0]
assert decode_int12_pitch("//").tolist() == [-1.0]
assert parse_yuaz_controls("").timbre_shift_semitones == 0.0
assert parse_yuaz_controls("").highband_enabled is False
assert parse_yuaz_controls("").highband_yuaz_only_hz == 12000.0
assert parse_yuaz_controls("YH80").highband_yuaz_only_hz == 8000.0
assert parse_yuaz_controls("YH120").highband_yuaz_only_hz == 12000.0
assert parse_yuaz_controls("YH0").highband_enabled is False
assert parse_yuaz_controls("YM0YD0").detail_strength == 1.0
assert parse_yuaz_controls("g-10YM50YD-50").timbre_shift_semitones == 6.0
assert parse_yuaz_controls("g-10YM50YD-50").detail_strength == 0.5
assert parse_yuaz_controls("YD100").detail_strength == 1.5

hb_sr = 44100
hb_t = np.arange(hb_sr, dtype=np.float32) / hb_sr
hb_gen = (0.16 * np.sin(2 * np.pi * 1000.0 * hb_t) + 0.015 * np.sin(2 * np.pi * 9000.0 * hb_t)).astype(np.float32)
hb_profile = {
    "band_centers_hz": [9000.0, 11000.0, 13000.0, 15000.0, 17000.0, 19000.0],
    "voiced_db_to_full": [-28.0, -29.0, -30.0, -31.0, -33.0, -35.0],
    "unvoiced_db_to_full": [-24.0, -25.0, -27.0, -29.0, -31.0, -33.0],
    "voiced_harmonic_mix": 0.78,
}
hb_f0 = np.full(100, 220.0, dtype=np.float32)
hb_profile["temporal"] = {"fixed_bins":16,"tail_bins":32,"low_delta_db":[0.0]*16+[6.0]*32,"upper_delta_db":[0.0]*16+[9.0]*32,"harmonic_mix":[0.65]*48,"voicing":[1.0]*48}
hb_out, hb_stats = synthesize_learned_highband(hb_gen, hb_sr, hb_f0, hb_profile, 1234, assist_start_hz=10000.0, detail_strength=1.0, target_fixed_ms=500.0)
assert hb_stats.get("temporal_used") is True
assert hb_stats.get("temporal_upper_peak_gain",1.0) > 1.5

def _band_energy(x, lo, hi):
    spec = np.fft.rfft(np.asarray(x, dtype=np.float64) * np.hanning(len(x)))
    freq = np.fft.rfftfreq(len(x), 1.0 / hb_sr)
    mask = (freq >= lo) & (freq < hi)
    return float(np.mean(np.abs(spec[mask]) ** 2))

assert hb_stats["used"] is True
assert hb_stats["harmonic_count"] > 0
assert _band_energy(hb_out, 12000.0, 20000.0) > _band_energy(hb_gen, 12000.0, 20000.0) * 100.0
hb_low, _ = synthesize_learned_highband(hb_gen, hb_sr, hb_f0, hb_profile, 1234, assist_start_hz=10000.0, detail_strength=0.0, target_fixed_ms=500.0)
assert _band_energy(hb_out, 12000.0, 20000.0) > _band_energy(hb_low, 12000.0, 20000.0) * 1.8
fake_db = {"groups": {"a": {"prototypes": [
    {"subbank_index":0,"anchor_midi":60.0,"voiced_db_to_full":[-30]*6,"unvoiced_db_to_full":[-28]*6,"voiced_harmonic_mix":0.6},
    {"subbank_index":1,"anchor_midi":72.0,"voiced_db_to_full":[-18]*6,"unvoiced_db_to_full":[-20]*6,"voiced_harmonic_mix":0.8},
]}}}
lo_profile = select_learned_profile(fake_db, "a", 60.0, -6.0, 0)
hi_profile = select_learned_profile(fake_db, "a", 60.0, 12.0, 0)
assert hi_profile["voiced_db_to_full"][3] > lo_profile["voiced_db_to_full"][3]
x = np.arange(1000, dtype=np.float32)
assert len(crop_oto(x, 1000, 100, 200)) == 700
source = np.ones(100, dtype=np.float32) * 220
mask = np.ones(100, dtype=np.float32)
values = []
for tone in ("C4", "G4", "C5"):
    f0, _, midi, hz = build_target_f0(tone, "AA", "!120", 100, 24000, 256, source, 0, mask)
    median = float(np.median(f0[f0 > 0]))
    values.append(median)
    print(f"{tone}: MIDI {midi}, base {hz:.3f} Hz, target {median:.3f} Hz")
assert values[0] < values[1] < values[2]

loud_t = np.arange(24000, dtype=np.float32) / 24000.0
for amplitude in (0.02, 0.08, 0.22):
    loud = amplitude * np.sin(2 * np.pi * 220.0 * loud_t)
    loud[:2400] = 0.0
    before = active_rms_dbfs(loud, 24000)
    normalized, stats = normalize_final_render(loud, 24000, target_dbfs=-18.0, peak_ceiling_dbfs=-1.0)
    after = active_rms_dbfs(normalized, 24000)
    assert stats["used"] is True
    assert stats["target_reached"] is True
    assert abs(after - (-18.0)) <= 0.06, (amplitude, before, after, stats)
    assert np.max(np.abs(normalized)) <= 10 ** (-1.0 / 20.0) + 1e-5

spiky = 0.06 * np.sin(2 * np.pi * 220.0 * loud_t)
spiky[5000] = 1.0
normalized_spiky, spiky_stats = normalize_final_render(spiky, 24000, target_dbfs=-18.0, peak_ceiling_dbfs=-1.0)
assert np.max(np.abs(normalized_spiky)) <= 10 ** (-1.0 / 20.0) + 1e-5
assert abs(active_rms_dbfs(normalized_spiky, 24000) + 18.0) <= 0.06
assert spiky_stats["peak_guard_samples"] > 0

manifest = []
for folder, note, suffix, f0 in (("A3", "A3", "_A3", 220.0), ("D4", "D4", "_D4", 293.66), ("G4", "G4", "_G4", 392.0), ("C5", "C5", "_C5", 523.25)):
    manifest.append({
        "status": "analyzed",
        "relative_wav": f"{folder}/a.wav",
        "wav_path": f"/tmp/{folder}/a.wav",
        "alias": f"a{suffix}",
        "median_f0_hz": f0,
        "cache": f".yuaz/cache/{folder}.npz",
    })
prefix = []
for tone, suffix in (("A3", "_A3"), ("D4", "_D4"), ("G4", "_G4"), ("C5", "_C5")):
    prefix.append({"tone": tone, "prefix": "", "suffix": suffix})
info = annotate_utau_subbanks("/tmp/TestBank", manifest, prefix)
assert info["prototype_count"] == 4
assert [x["anchor_note"] for x in info["subbanks"]] == ["A3", "D4", "G4", "C5"]
assert all(x["base_alias"] == "a" for x in manifest)
assert [x["subbank_index"] for x in manifest] == [0, 1, 2, 3]

folder_manifest = [
    {"status": "analyzed", "relative_wav": "soft_F3/a.wav", "wav_path": "/tmp/soft_F3/a.wav", "alias": "a", "median_f0_hz": 174.6, "cache": ".yuaz/cache/f3.npz"},
    {"status": "analyzed", "relative_wav": "soft_C4/a.wav", "wav_path": "/tmp/soft_C4/a.wav", "alias": "a", "median_f0_hz": 261.6, "cache": ".yuaz/cache/c4.npz"},
]
folder_info = annotate_utau_subbanks("/tmp/TestBank2", folder_manifest, [])
assert folder_info["prototype_count"] == 2
assert [x["anchor_note"] for x in folder_info["subbanks"]] == ["F3", "C4"]

strict_prefix = [
    {"tone": "G3", "prefix": "G3", "suffix": ""},
    {"tone": "D4", "prefix": "D4", "suffix": ""},
    {"tone": "G4", "prefix": "G4", "suffix": ""},
    {"tone": "D5", "prefix": "D5", "suffix": ""},
]
strict_manifest = []
for folder, prefix_text, f0 in (("SingerG3", "G3", 196.0), ("SingerD4", "D4", 293.66), ("Singer_G4", "G4", 392.0), ("SingerD5", "D5", 587.33)):
    for alias in ("a", "- ba", "shi"):
        strict_manifest.append({
            "status": "analyzed",
            "relative_wav": f"{folder}/{alias.replace(' ', '_')}.wav",
            "wav_path": f"/tmp/{folder}/{alias.replace(' ', '_')}.wav",
            "alias": prefix_text + alias,
            "median_f0_hz": f0,
            "cache": f".yuaz/cache/{folder}-{alias.replace(' ', '_')}.npz",
        })
strict_manifest.append({"status": "analyzed", "relative_wav": "SingerD4/special.wav", "wav_path": "/tmp/SingerD4/special.wav", "alias": "special", "median_f0_hz": 294.0, "cache": ".yuaz/cache/special.npz"})
strict_manifest.append({"status": "analyzed", "relative_wav": "sample.wav", "wav_path": "/tmp/sample.wav", "alias": "sample", "median_f0_hz": 247.0, "cache": ".yuaz/cache/sample.npz"})
strict_info = annotate_utau_subbanks("/tmp/TestBank3", strict_manifest, strict_prefix)
assert strict_info["prototype_count"] == 4
assert strict_info["prefix_map_authoritative"] is True
assert strict_info["fallback_created_prototypes"] == 0
assert [x["anchor_note"] for x in strict_info["subbanks"]] == ["G3", "D4", "G4", "D5"]
assert all(x["label"] not in ("root", "SingerD4") for x in strict_info["subbanks"])
assert strict_manifest[-2]["subbank_label"] == "D4"
assert strict_manifest[-1]["subbank_index"] >= 0


art_sr = 24000
art_hop = 256
art_n = int(0.70 * art_sr)
art_t = np.arange(art_n, dtype=np.float32) / art_sr
rng = np.random.default_rng(123)
art_audio = np.zeros(art_n, dtype=np.float32)
art_audio[: int(0.08 * art_sr)] = 0.02 * rng.normal(size=int(0.08 * art_sr))
art_audio[int(0.08 * art_sr):] = 0.15 * np.sin(2 * np.pi * 220.0 * art_t[int(0.08 * art_sr):])
art_frames = max(16, int(np.ceil(art_n / art_hop)))
art_f0 = np.zeros(art_frames, dtype=np.float32)
art_f0[8:] = 220.0
art_detail = np.zeros((25, art_frames), dtype=np.float32)
art_detail[-1, :13] = 1.2
art_detail[-1, 13:] = 0.12
regions = analyze_articulation_regions(art_audio, art_sr, art_hop, art_f0, art_detail, 180.0)
assert regions["first_voiced_ms"] > 50.0
assert regions["articulation_end_ms"] > regions["transition_end_ms"] > regions["first_voiced_ms"]
source_voiced = art_audio[int(0.08 * art_sr): int(0.26 * art_sr)]
target_t = np.arange(len(source_voiced), dtype=np.float32) / art_sr
target_voiced = 0.15 * np.sin(2 * np.pi * 330.0 * target_t)
shaped, used, transfer_stats = transfer_articulation_trajectory(source_voiced, target_voiced, art_sr)
assert used and shaped.shape == target_voiced.shape
assert np.isfinite(shaped).all()
assert transfer_stats["trajectory_strength"] > 0.0
generated = 0.15 * np.sin(2 * np.pi * 330.0 * np.arange(int(0.8 * art_sr), dtype=np.float32) / art_sr)
hybrid, art_stats = single_source_articulation_hybrid(art_audio, generated, art_sr, regions, 180.0, 180.0, 800.0)
assert len(hybrid) == len(generated)
assert art_stats["psola_used"] is False
assert art_stats["single_periodic_source"] is True
assert art_stats["trajectory_transfer_used"] is True
assert art_stats["target_articulation_end_ms"] > art_stats["target_onset_ms"]
mapped = map_articulation_regions(regions, 180.0, 160.0, 700.0, 800.0)
assert mapped["target_articulation_end_ms"] > mapped["target_onset_ms"]

canon_t = np.arange(int(0.19 * art_sr), dtype=np.float32) / art_sr
common = 0.12 * np.sin(2 * np.pi * 220.0 * canon_t)
common += (0.07 * np.exp(-canon_t * 18.0)) * np.sin(2 * np.pi * 660.0 * canon_t)
common += (0.04 * np.exp(-canon_t * 12.0)) * np.sin(2 * np.pi * 1100.0 * canon_t)

def tilt_audio(x, tilt_db):
    spec = np.fft.rfft(x.astype(np.float64))
    f = np.linspace(0.0, 1.0, spec.size)
    gain = 10.0 ** ((tilt_db * (f - 0.5)) / 20.0)
    return np.fft.irfft(spec * gain, n=len(x)).astype(np.float32)

templates = [extract_neutral_articulation_template(tilt_audio(common, db), art_sr) for db in (-7.0, -2.0, 3.0, 8.0)]
canonical = combine_canonical_articulation_templates(templates)
assert canonical is not None
assert canonical["source_count"] == 4
assert 0.25 <= canonical["coherence"] <= 1.0
canon_target = 0.11 * np.sin(2 * np.pi * 330.0 * canon_t)
canon_target += 0.035 * np.sin(2 * np.pi * 990.0 * canon_t)
canon_shaped, canon_used, canon_stats = apply_articulation_template(canonical, canon_target, art_sr)
assert canon_used and canon_shaped.shape == canon_target.shape
assert np.isfinite(canon_shaped).all()
assert canon_stats["canonical_coherence"] > 0.0
canonical_hybrid, canonical_hybrid_stats = single_source_articulation_hybrid(
    art_audio, generated, art_sr, regions, 180.0, 180.0, 800.0, canonical_template=canonical
)
assert canonical_hybrid_stats["trajectory_source"] == "canonical"
assert canonical_hybrid_stats["single_periodic_source"] is True

xwarp = torch.arange(20, dtype=torch.float32).view(1, 1, -1)
warped = piecewise_warp(xwarp, 6, 30, target_fixed_frames=4)
assert warped.shape[-1] == 30

sample_audio = np.sin(2 * np.pi * 220 * np.arange(24000, dtype=np.float32) / 24000)
perturbed_a = timbre_perturb_audio(sample_audio, 24000, 1234)
perturbed_b = timbre_perturb_audio(sample_audio, 24000, 5678)
assert perturbed_a.shape == sample_audio.shape and perturbed_b.shape == sample_audio.shape
detail_np = extract_detail_features(sample_audio, 24000, 256)
assert detail_np.shape[0] == 25
legacy_adapter = VoicebankAdapter(pitch_prototype_count=6, pitch_prototype_midi=[55.0, 59.45, 62.0, 62.05, 67.0, 74.0])
with torch.no_grad():
    for i in range(6):
        legacy_adapter.pitch_timbre_codes[i].fill_(i + 1)
legacy_adapter.configure_pitch_prototypes([55.0, 62.0, 67.0, 74.0])
assert legacy_adapter.pitch_prototype_count == 4
assert [round(float(x)) for x in legacy_adapter.pitch_prototype_midi] == [55, 62, 67, 74]
assert [round(float(x.detach().mean())) for x in legacy_adapter.pitch_timbre_codes] == [1, 3, 5, 6]

adapter = VoicebankAdapter()
adapter.set_bank_median_f0(293.66)
adapter.configure_pitch_prototypes([57.0, 62.0, 67.0, 72.0])
assert adapter.pitch_prototype_count == 4
z = torch.randn(1, 128, 32) * 0.1
z_aug = z + torch.randn_like(z) * 0.01
z_aug_b = z + torch.randn_like(z) * 0.012
detail = torch.from_numpy(detail_np[:, :32]).unsqueeze(0)
f0 = torch.full((1, 1, 32), 392.0)
assert adapter.content_representation(z).shape == z.shape
assert adapter.apply_latent(z, detail=detail, f0=f0, source_prototype_index=2).shape == z.shape
assert adapter.spectral_gain(1025, detail=detail, frames=32, f0=f0, batch=1, source_prototype_index=2).shape == (1, 1025, 32)
ap = torch.full((1, 16, 32), 0.3)
assert adapter.apply_ap(ap, detail=detail, f0=f0, source_prototype_index=2).shape == ap.shape
inv, anchor = content_consistency_loss(adapter, z, z_aug, z_aug_b, 2, 28)
assert torch.isfinite(inv) and torch.isfinite(anchor)
weights = adapter._pitch_weights(f0, batch=1, source_prototype_index=2)
assert weights.shape == (1, 4)
assert int(torch.argmax(weights, dim=1)[0]) == 2
weights_up = adapter._pitch_weights(f0, batch=1, source_prototype_index=2, timbre_shift_semitones=12.0)
weights_down = adapter._pitch_weights(f0, batch=1, source_prototype_index=2, timbre_shift_semitones=-12.0)
assert float((weights_up - weights).abs().sum()) > 1e-4
assert float((weights_down - weights).abs().sum()) > 1e-4
base_latent = adapter.apply_latent(z, detail=detail, f0=f0, source_prototype_index=2)
zero_latent = adapter.apply_latent(z, detail=detail, f0=f0, source_prototype_index=2, timbre_shift_semitones=0.0, detail_strength=1.0)
assert torch.equal(base_latent, zero_latent)

refiner = TinyFidelityRefiner()
wave = torch.randn(1, 1, 8192) * 0.05
refined, residual = refiner(wave, detail, f0, articulation_end_sample=2048)
assert refined.shape == wave.shape and residual.shape == wave.shape
ratio = torch.sqrt(torch.mean(residual.pow(2)) + 1e-8) / (torch.sqrt(torch.mean(wave.pow(2)) + 1e-8))
assert float(ratio.detach()) <= 0.111
loss = (refined - wave * 0.98).abs().mean()
loss.backward()
assert refiner.output.weight.grad is not None

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    canonical_path = td / "canonical.npz"
    save_canonical_articulation(canonical_path, canonical, {"base_alias": "a"})
    loaded_canonical = load_canonical_articulation(canonical_path)
    assert loaded_canonical["trajectory"].shape == canonical["trajectory"].shape
    dummy = YuazDDSPResamplerEngine.__new__(YuazDDSPResamplerEngine)
    dummy._canonical_articulation_cache = {}
    record = {
        "voicebank_root": str(td),
        "source_loudness_variants": [
            {"signature": "0.000|100.000|-200.000", "offset": 0.0, "consonant": 100.0, "cutoff": -200.0, "base_alias": "a", "canonical_articulation": "canonical.npz", "subbank_index": 2},
            {"signature": "50.000|120.000|-180.000", "offset": 50.0, "consonant": 120.0, "cutoff": -180.0, "base_alias": "a k", "subbank_index": 2},
        ],
    }
    variant = dummy._request_variant(record, {"offset": 50.0, "consonant": 120.0, "cutoff": -180.0})
    assert variant["base_alias"] == "a k"
    variant = record["source_loudness_variants"][0]
    loaded_by_engine = dummy._canonical_articulation_for_variant(record, variant)
    assert loaded_by_engine is not None
    apath = td / "adapter.pt"
    rpath = td / "fidelity_refiner.pt"
    save_adapter(apath, adapter, {"test": True})
    loaded, meta = load_adapter(apath)
    assert loaded.pitch_prototype_count == 4
    assert [round(float(x)) for x in loaded.pitch_prototype_midi] == [57, 62, 67, 72]
    assert meta["loaded_adapter_format"] == 5
    save_refiner(rpath, refiner, {"test": True})
    rloaded, rmeta = load_refiner(rpath)
    assert rloaded.detail_dim == 25
    assert rmeta["loaded_refiner_format"] == 2
print("Native controls + Learned High-Band v3 continuity trajectories + articulation + normalization self-test OK")
PY
