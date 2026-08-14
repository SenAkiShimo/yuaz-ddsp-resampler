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
import hashlib
import json
import tempfile
import wave
from pathlib import Path

import numpy as np
import torch
import yaml

from yuaz_ddsp_resampler.core import YuazDDSPResamplerEngine, make_adaptive_decoder_class, decode_int12_pitch, tone_to_midi, crop_oto, build_target_f0, extract_detail_features, piecewise_warp, blend_dualrate_fullband_body
from yuaz_ddsp_resampler.articulation import (
    analyze_articulation_regions, apply_articulation_template, combine_canonical_articulation_templates,
    extract_neutral_articulation_template, load_canonical_articulation, map_articulation_regions,
    save_canonical_articulation, single_source_articulation_hybrid, transfer_articulation_trajectory,
)
from yuaz_ddsp_resampler.adapter import VoicebankAdapter, load_adapter, save_adapter
from yuaz_ddsp_resampler.ai_vocal_controls import AIControlAdapter, save_ai_control_adapter, load_ai_control_adapter
from yuaz_ddsp_resampler.ai_control_training import _technique_control_curve, _control_dict
from yuaz_ddsp_resampler.controls import parse_yuaz_controls
from yuaz_ddsp_resampler.vocal_controls import apply_decoder_vocal_controls
from yuaz_ddsp_resampler.learned_highband import synthesize_learned_highband, select_learned_profile
from yuaz_ddsp_resampler.highband_foundation import HighBandFoundation, save_highband_foundation, load_highband_foundation, apply_highband_foundation, highpass_residual_torch, blend_foundation_with_continuity
from yuaz_ddsp_resampler.fidelity import TinyFidelityRefiner, load_refiner, save_refiner, FIDELITY_HARD_LIMIT
from yuaz_ddsp_resampler.loudness import active_rms_dbfs, normalize_final_render
from yuaz_ddsp_resampler.prepare import timbre_perturb_audio, content_consistency_loss
from yuaz_ddsp_resampler.voicebank import annotate_utau_subbanks
from yuaz_ddsp_resampler.state import begin_generation, commit_generation, clone_state, resolve_active_state, resolve_stable_state, STATE_CONTAINER, PREVIOUS_028AI10_STATE_CONTAINER, PREVIOUS_028AI9_STATE_CONTAINER, PREVIOUS_028AI8_STATE_CONTAINER, PREVIOUS_028AI7_STATE_CONTAINER, PREVIOUS_028AI6_STATE_CONTAINER, PREVIOUS_028AI5_STATE_CONTAINER, PREVIOUS_028AI4_STATE_CONTAINER, PREVIOUS_028AI3_STATE_CONTAINER, PREVIOUS_028AI2_STATE_CONTAINER, PREVIOUS_028AI1_STATE_CONTAINER, PREVIOUS_028_STATE_CONTAINER, STABLE_STATE_CONTAINER, PREDECESSOR_AI_STATE_CONTAINER

# Verify that the user-requested RC3.2 acoustic baseline modules were not
# accidentally altered while RC3.3 runtime/state safety was implemented.
baseline = json.loads(Path("ACOUSTIC_BASELINE.json").read_text(encoding="utf-8"))
for rel, expected in baseline["unchanged_modules"].items():
    actual = hashlib.sha256(Path(rel).read_bytes()).hexdigest()
    assert actual == expected, f"RC3.2 acoustic baseline drift: {rel}"
prepare_source = Path("src/yuaz_ddsp_resampler/prepare.py").read_text(encoding="utf-8")
transaction_source = Path("src/yuaz_ddsp_resampler/transaction.py").read_text(encoding="utf-8")
assert "CACHE_FORMAT = 6" in prepare_source
assert "DEEP_TRAINING_VERSION = 1" in prepare_source
assert "analysis_signature" in prepare_source
assert "_validate_stage_a" in prepare_source
assert "link_analysis_caches(source, staging)" not in transaction_source, "Production Deep must not symlink an old analysis cache"
assert "clone_state(source, staging, link_caches=False)" in transaction_source, "Continue Deep must use an isolated cache copy"

assert tone_to_midi("C4") == 60
assert tone_to_midi("G4") == 67
assert tone_to_midi("C5") == 72
assert decode_int12_pitch("AA").tolist() == [0.0]
assert decode_int12_pitch("//").tolist() == [-1.0]
assert parse_yuaz_controls("").timbre_shift_semitones == 0.0
assert parse_yuaz_controls("").highband_enabled is False
assert parse_yuaz_controls("").highband_yuaz_only_hz == 12000.0
assert abs(parse_yuaz_controls("YH100").highband_strength - 1.0) < 1e-8
assert parse_yuaz_controls("YH100").highband_yuaz_only_hz < parse_yuaz_controls("YH25").highband_yuaz_only_hz
assert parse_yuaz_controls("YH120").highband_strength == 1.0
assert parse_yuaz_controls("YH-10").highband_enabled is False
assert parse_yuaz_controls("YH0").highband_enabled is False
assert parse_yuaz_controls("YM0YD0").detail_strength == 1.0
assert parse_yuaz_controls("g-10YM50YD-50").timbre_shift_semitones == 6.0
assert parse_yuaz_controls("g-10YM50YD-50").detail_strength == 0.5
assert parse_yuaz_controls("YD100").detail_strength == 1.5
vc = parse_yuaz_controls("YT35YB-20YV40YG-15YO10YF70YX55YP40")
assert vc.tension == 35.0
assert vc.breathiness == -20.0
assert vc.voicing == 40.0
assert vc.gender_formant == -15.0
assert vc.mouth == 10.0
assert vc.falsetto == 70.0 and vc.mixed_voice == 55.0 and vc.pharyngeal == 40.0
assert parse_yuaz_controls("YA100") == parse_yuaz_controls(""), "YA Attack must be fully retired in 0.2.8ai.11"
_manifest = yaml.safe_load(Path("resampler-manifest.yaml").read_text(encoding="utf-8"))
_exprs = _manifest.get("expressions", {})
assert len(_exprs) == 12, f"0.2.8ai.11 must expose exactly 12 Yuaz controls, got {len(_exprs)}"
_flags = {str(v.get("flag")) for v in _exprs.values()}
assert _flags == {"YM","YD","YH","YT","YB","YV","YG","YO","YF","YX","YP","YR"}
assert "YA" not in _flags
assert "YR" in _flags
assert parse_yuaz_controls("YR1").raw_bypass_enabled
assert not parse_yuaz_controls("YR0").raw_bypass_enabled
assert parse_yuaz_controls("").vocal_controls_active is False
curves = vc.frame_controls(12, torch.device("cpu"), torch.float32, curves={"YT": [-100, 100], "YB": [0, 50]})
assert curves["tension"].shape == (1, 1, 12)
assert float(curves["tension"].min()) <= -0.99 and float(curves["tension"].max()) >= 0.99

S = torch.ones(1, 64, 12)
A = torch.full((1, 16, 12), 0.35)
G = torch.full((1, 1, 12), 0.55)
F0 = torch.full((1, 1, 12), 220.0)
neutral = parse_yuaz_controls("").frame_controls(12, torch.device("cpu"), torch.float32)
S0, A0, G0 = apply_decoder_vocal_controls(S, A, G, F0, neutral)
assert torch.allclose(S0, S, atol=1e-7)
assert torch.allclose(A0, A, atol=1e-7)
assert torch.allclose(G0, G, atol=1e-7)
# RC4.2 parameter-axis separation: Tension no longer changes AP/gate;
# Breathiness owns AP; Voicing owns broad harmonic gate without touching AP.
def one_control(flag):
    fc = parse_yuaz_controls(flag).frame_controls(12, torch.device("cpu"), torch.float32)
    return apply_decoder_vocal_controls(S, A, G, F0, fc)
ST, AT, GT = one_control("YT100")
assert not torch.allclose(ST, S)
assert torch.allclose(AT, A, atol=1e-7) and torch.allclose(GT, G, atol=1e-7)
SB, AB, GB = one_control("YB100")
assert torch.allclose(SB, S, atol=1e-7) and not torch.allclose(AB, A) and not torch.allclose(GB, G)
SV, AV, GV = one_control("YV100")
assert not torch.allclose(SV, S) and torch.allclose(AV, A, atol=1e-7) and not torch.allclose(GV, G)
SG, AG, GG = one_control("YG100")
assert torch.allclose(AG, A, atol=1e-7) and torch.allclose(GG, G, atol=1e-7)
SM, AM, GM = one_control("YO100")
assert not torch.allclose(SM, S) and torch.allclose(AM, A, atol=1e-7) and torch.allclose(GM, G, atol=1e-7)
SF, AF, GF = one_control("YF100")
SX, AX, GX = one_control("YX100")
SP, APH, GP = one_control("YP100")
assert not torch.allclose(SF, S) and not torch.allclose(AF, A) and not torch.allclose(GF, G)
assert not torch.allclose(SX, S) and not torch.allclose(GX, G)
assert not torch.allclose(SP, S)
assert not torch.allclose(SF, SX) and not torch.allclose(SX, SP), "YF/YX/YP carriers must remain distinct"
STN, ATN, GTN = one_control("YT-100")
SVN, AVN, GVN = one_control("YV-100")
SGN, AGN, GGN = one_control("YG-100")
assert not torch.allclose(ST, STN), "YT +/- directions must differ"
assert not torch.allclose(GV, GVN), "YV +/- directions must differ"
assert not torch.allclose(SG, SGN), "YG +/- directions must differ"
active = parse_yuaz_controls("YT60YB40YV20YG-30YO35").frame_controls(12, torch.device("cpu"), torch.float32)
S1, A1, G1 = apply_decoder_vocal_controls(S, A, G, F0, active)
assert not torch.allclose(S1, S)
assert not torch.allclose(A1, A)
assert not torch.allclose(G1, G)
# AI controller regression: zero control is an exact bypass, while a learned
# nonzero residual manipulates DDSP features without generating waveform audio.
ai = AIControlAdapter()
assert tuple(ai.control_names) == ("breathiness", "falsetto", "mixed_voice", "pharyngeal")
assert tuple(ai.output_scopes) == ("spectral", "ap", "gate")
phonation_ai = AIControlAdapter(control_names=("tension", "voicing"), control_modes=("signed", "signed"), output_scopes=("spectral", "ap", "gate"))
assert tuple(phonation_ai.control_names) == ("tension", "voicing")
assert tuple(phonation_ai.control_modes) == ("signed", "signed")
assert tuple(phonation_ai.output_scopes) == ("spectral", "ap", "gate")
mouth_ai = AIControlAdapter(control_names=("mouth",), control_modes=("signed",), output_scopes=("spectral",))
assert tuple(mouth_ai.control_names) == ("mouth",) and tuple(mouth_ai.output_scopes) == ("spectral",)
with tempfile.TemporaryDirectory() as mptd:
    mptd = Path(mptd)
    pp = mptd / "ai_phonation_foundation-v1.pt"
    mp = mptd / "ai_mouth_foundation-v1.pt"
    save_ai_control_adapter(pp, phonation_ai, {"feature_backend":"yuaz-native-ddsp-v1","checkpoint_sha256":"selftest"})
    save_ai_control_adapter(mp, mouth_ai, {"feature_backend":"yuaz-native-ddsp-v1","checkpoint_sha256":"selftest"})
    pl, pm = load_ai_control_adapter(pp)
    ml, mm = load_ai_control_adapter(mp)
    assert tuple(pl.control_names) == ("tension", "voicing") and tuple(pl.control_modes) == ("signed", "signed")
    assert tuple(ml.control_names) == ("mouth",) and tuple(ml.output_scopes) == ("spectral",)
gender_ai = AIControlAdapter(control_names=("gender_formant",), control_modes=("signed",), output_scopes=("spectral",))
assert tuple(gender_ai.control_names) == ("gender_formant",)
assert tuple(gender_ai.control_modes) == ("signed",)
assert tuple(gender_ai.output_scopes) == ("spectral",)
with torch.no_grad():
    gender_ai.output_proj.bias[:gender_ai.spectral_bands].fill_(0.35)
    gender_ai.output_proj.bias[gender_ai.spectral_bands:].fill_(0.9)
gfc = parse_yuaz_controls("YG100").frame_controls(12, torch.device("cpu"), torch.float32)
gs, ga, gg = gender_ai.apply(S, A, G, F0, gfc)
assert not torch.allclose(gs, S), "learned gender pack must be able to change spectral envelope"
assert torch.allclose(ga, A, atol=1e-7), "spectral-only gender pack must never change AP"
assert torch.allclose(gg, G, atol=1e-7), "spectral-only gender pack must never change gate"
# 0.2.8ai.11 keeps a low-gain interpretable carrier underneath learned packs so
# a collapsed checkpoint can never make the UI axis silently dead.
ms, ma, mg = apply_decoder_vocal_controls(S, A, G, F0, gfc, learned_controls=("gender_formant",))
assert not torch.allclose(ms, S) and torch.allclose(ma, A, atol=1e-7) and torch.allclose(mg, G, atol=1e-7)
# Positive Breathiness also retains a restrained carrier; negative Breathiness
# keeps the stronger deterministic de-breathing direction because GTSinger has no
# directly supervised negative target.
_fc_pos_b = parse_yuaz_controls("YB100").frame_controls(12, torch.device("cpu"), torch.float32)
_ms, _ma, _mg = apply_decoder_vocal_controls(S, A, G, F0, _fc_pos_b, learned_controls=ai.control_names)
assert torch.allclose(_ms, S, atol=1e-7) and not torch.allclose(_ma, A) and not torch.allclose(_mg, G)
_fc_neg_b = parse_yuaz_controls("YB-100").frame_controls(12, torch.device("cpu"), torch.float32)
_ms, _ma, _mg = apply_decoder_vocal_controls(S, A, G, F0, _fc_neg_b, learned_controls=ai.control_names)
assert not torch.allclose(_ma, A) and not torch.allclose(_mg, G)
_fc_tv = parse_yuaz_controls("YT100YV100").frame_controls(12, torch.device("cpu"), torch.float32)
_ms, _ma, _mg = apply_decoder_vocal_controls(S, A, G, F0, _fc_tv, learned_controls=ai.control_names)
assert not torch.allclose(_ms, S) and not torch.allclose(_mg, G)
ai_active = parse_yuaz_controls("YB80YF60YX30YP40").frame_controls(12, torch.device("cpu"), torch.float32)
ai_zero = parse_yuaz_controls("").frame_controls(12, torch.device("cpu"), torch.float32)
AS0, AA0, AG0 = ai.apply(S, A, G, F0, ai_zero)
assert torch.allclose(AS0, S, atol=1e-7) and torch.allclose(AA0, A, atol=1e-7) and torch.allclose(AG0, G, atol=1e-7)
with torch.no_grad():
    ai.output_proj.bias[:ai.spectral_bands].fill_(0.25)
    ai.output_proj.bias[ai.spectral_bands:ai.spectral_bands+ai.ap_bands].fill_(0.20)
    ai.output_proj.bias[-1:].fill_(-0.18)
AS1, AA1, AG1 = ai.apply(S, A, G, F0, ai_active)
assert not torch.allclose(AS1, S) and not torch.allclose(AA1, A) and not torch.allclose(AG1, G)
with tempfile.TemporaryDirectory() as aitd:
    ap = Path(aitd) / "ai_control_adapter.pt"
    save_ai_control_adapter(ap, ai, {"selftest": True})
    ail, aim = load_ai_control_adapter(ap)
    assert aim["selftest"] is True and sum(p.numel() for p in ail.parameters()) == sum(p.numel() for p in ai.parameters())
# GTSinger direct supervision must respect its per-phoneme JSON labels rather
# than treating an entire technique-group WAV as technique=1.
with tempfile.TemporaryDirectory() as gtd:
    gtd = Path(gtd)
    gwav = gtd / "0000.wav"
    with wave.open(str(gwav), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(24000)
        wf.writeframes((np.zeros(48000, dtype=np.int16)).tobytes())
    gjson = [{
        "ph": ["a", "<AP>", "i"],
        "ph_start": [0.0, 0.5, 1.0],
        "ph_end": [0.5, 1.0, 1.5],
        "breathy": ["1", "0", "1"],
        "mix": ["0", "0", "0"],
        "falsetto": ["0", "0", "0"],
        "pharyngeal": ["0", "0", "0"],
    }]
    gwav.with_suffix(".json").write_text(json.dumps(gjson), encoding="utf-8")
    gcurve, gmeta = _technique_control_curve(gwav, "breathy", 200)
    assert gcurve.shape == (4, 200) and gmeta["annotation"] == "phoneme-json"
    assert 0.45 < float(gcurve[0].mean()) < 0.55
    assert float(gcurve[1:].sum()) == 0.0
    gdict = _control_dict(gcurve[:, :80], 80)
    assert gdict["breathiness"].shape == (1, 1, 80)
# The offline foundation path must be able to read Yuaz's own pre-synthesis
# DDSP state without invoking oscillator/noise waveform generation.
class _IdentityEmformer(torch.nn.Module):
    def forward(self, x): return x, None
class _FakeDDSPBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sample_rate = 24000; self.fft_size = 30; self.encoder_hop_length = 256
        self.emformer_input_proj = torch.nn.Linear(5, 8)
        self.emformer = _IdentityEmformer()
        self.emformer_proj = torch.nn.Linear(8, 8)
        self.env_net = torch.nn.Linear(8, 16)
        self.ap_net = torch.nn.Linear(8, 16)
        self.weight_net = torch.nn.Sequential(torch.nn.Linear(8, 1), torch.nn.Sigmoid())
    def _decompress_envelope(self, x): return torch.exp(torch.clamp(x, -3.0, 3.0))
    def _smooth_ap_bands(self, x): return torch.sigmoid(x)
_NativeTestDecoder = make_adaptive_decoder_class(_FakeDDSPBase)
_native_decoder = _NativeTestDecoder().eval()
with torch.inference_mode():
    _native_state = _native_decoder.extract_neural_ddsp_state(torch.full((1,1,20), 220.0), torch.randn(1,4,20))
assert _native_state["spectral_envelope"].shape == (1,16,20)
assert _native_state["ap"].shape == (1,16,20)
assert _native_state["gate"].shape == (1,1,20)

# Public training/install scripts must use the native Yuaz backend and must not
# contain destructive paths for the stable RC4.2 runtime/wrapper.
_train_script = Path("scripts/train-ai-control-foundation.command").read_text(encoding="utf-8")
assert "--feature-backend yuaz-native" in _train_script and '--project-root "$ROOT"' in _train_script
assert "YF Falsetto" in _train_script and "YX Mixed Voice" in _train_script and "YP Pharyngeal" in _train_script
_setup_all = Path("scripts/setup-all-training.command").read_text(encoding="utf-8")
assert "setup-gender-training.command" in _setup_all and "setup-multimodal-training.command" in _setup_all
_train_all = Path("scripts/train-all-learned-packs.command").read_text(encoding="utf-8")
assert "ai_phonation_foundation-v1.pt" in _train_all and "ai_mouth_foundation-v1.pt" in _train_all
_setup_training = Path("scripts/setup-ai-training.command").read_text(encoding="utf-8")
assert "chinese-core" in _setup_training and "hf-mirror.com" in _setup_training and "huggingface.co" in _setup_training and "--dry-run" in _setup_training and "same manifest/.part files" in _setup_training
_install_txt = Path("scripts/install-openutau-macos.command").read_text(encoding="utf-8")
_purge_txt = Path("scripts/migrate-and-purge-previous.command").read_text(encoding="utf-8")
assert "migrate-and-purge-previous.command" in _install_txt
_configure_txt = Path("scripts/configure-macos.command").read_text(encoding="utf-8")
assert "voicebank_registry-0.2.8ai11.json" in _configure_txt
assert "voicebank_registry-0.2.8ai6.json" not in _configure_txt
assert 'voicebank_registry-0.2.8ai11.json' in _purge_txt
assert ".yuaz-0.2.8ai11" in _purge_txt and ".yuaz-0.2.8ai9" in _purge_txt and ".yuaz-0.2.8ai8" in _purge_txt and ".yuaz-0.2.8ai7" in _purge_txt and ".yuaz-0.2.8ai6" in _purge_txt
assert "~/Documents/Yuaz-DDSP-Backups" in _purge_txt
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
hb_out, hb_stats = synthesize_learned_highband(hb_gen, hb_sr, hb_f0, hb_profile, 1234, assist_start_hz=8800.0, detail_strength=1.0, target_fixed_ms=500.0, restoration_strength=1.0)
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
# Regression for the user's real failure: a 24 kHz DDSP body can be blank above 12 kHz,
# and a source profile can also report near-zero upper bands. YH100 must still reconstruct
# measurable 13–20 kHz energy instead of reproducing that blank region.
hb_dead = dict(hb_profile)
hb_dead["voiced_db_to_full"] = [-28.0, -34.0, -80.0, -80.0, -80.0, -80.0]
hb_dead["unvoiced_db_to_full"] = [-30.0, -38.0, -80.0, -80.0, -80.0, -80.0]
hb_dead.pop("temporal", None)
hb_dead_out, hb_dead_stats = synthesize_learned_highband(hb_gen, hb_sr, hb_f0, hb_dead, 4321, assist_start_hz=parse_yuaz_controls("YH100").highband_yuaz_only_hz, restoration_strength=1.0)
assert hb_dead_stats.get("reconstruction_floor_ratio", 0.0) >= 0.0025
hb_dead_before = _band_energy(hb_gen, 13000.0, 20000.0)
hb_dead_after = _band_energy(hb_dead_out, 13000.0, 20000.0)
print(f"High-band dead-profile regression 13-20k: before={hb_dead_before:.8g} after={hb_dead_after:.8g} ratio={hb_dead_after/max(hb_dead_before,1e-20):.2f}x")
assert hb_dead_stats.get("branch_rms", 0.0) > 1e-4
assert hb_dead_after > max(hb_dead_before * 40.0, 1e-5)
# Low-note regression: the old fixed partial ceiling could stop a 45–60 Hz note
# around 12–14 kHz. Source-texture synthesis must fill the upper band without a
# harmonic-index ceiling.
hb_low_f0 = np.full(100, 52.0, dtype=np.float32)
hb_low_note_out, hb_low_note_stats = synthesize_learned_highband(hb_gen, hb_sr, hb_low_f0, hb_dead, 9876, assist_start_hz=8800.0, restoration_strength=1.0)
assert hb_low_note_stats.get("synthesis_mode") == "bandlimited_source_texture"
assert hb_low_note_stats.get("harmonic_count", 0) > 150
hb_low_before = _band_energy(hb_gen, 16000.0, 20000.0)
hb_low_after = _band_energy(hb_low_note_out, 16000.0, 20000.0)
print(f"High-band low-note regression 16-20k: before={hb_low_before:.8g} after={hb_low_after:.8g} ratio={hb_low_after/max(hb_low_before,1e-20):.2f}x")
assert hb_low_note_stats.get("branch_rms", 0.0) > 1e-4
assert hb_low_after > max(hb_low_before * 40.0, 1e-5)
hb_low, _ = synthesize_learned_highband(hb_gen, hb_sr, hb_f0, hb_profile, 1234, assist_start_hz=8800.0, detail_strength=0.0, target_fixed_ms=500.0, restoration_strength=1.0)
assert _band_energy(hb_out, 12000.0, 20000.0) > _band_energy(hb_low, 12000.0, 20000.0) * 1.8
fake_db = {"groups": {"a": {"prototypes": [
    {"subbank_index":0,"anchor_midi":60.0,"voiced_db_to_full":[-30]*6,"unvoiced_db_to_full":[-28]*6,"voiced_harmonic_mix":0.6},
    {"subbank_index":1,"anchor_midi":72.0,"voiced_db_to_full":[-18]*6,"unvoiced_db_to_full":[-20]*6,"voiced_harmonic_mix":0.8},
]}}}
lo_profile = select_learned_profile(fake_db, "a", 60.0, -6.0, 0)
hi_profile = select_learned_profile(fake_db, "a", 60.0, 12.0, 0)
assert hi_profile["voiced_db_to_full"][3] > lo_profile["voiced_db_to_full"][3]
# If OpenUtau cache routing loses the exact base alias, YH must not turn off.
# Fall back to a bank-wide profile assembled only from this voicebank's own learned profiles.
fallback_profile = select_learned_profile(fake_db, "missing-cache-alias", 64.0, 0.0, 0)
assert fallback_profile is not None
assert fallback_profile.get("match_mode") == "bank-wide-fallback"
assert fallback_profile.get("selected_base_alias") == "<bank-wide>"

# A stale registry file path must self-heal via the record's current state_path.
with tempfile.TemporaryDirectory() as hbtd:
    hbstate = Path(hbtd) / "generation"
    hbstate.mkdir()
    hbdb = {"format":3,"groups":{"a":{"prototypes":[{"subbank_index":0,"anchor_midi":60.0,"voiced_db_to_full":[-30]*6,"unvoiced_db_to_full":[-28]*6,"voiced_harmonic_mix":0.6}]}}}
    (hbstate / "highband_profiles_v3.json").write_text(json.dumps(hbdb), encoding="utf-8")
    dummy_hb = object.__new__(YuazDDSPResamplerEngine)
    dummy_hb._highband_profile_cache = {}
    loaded_db, route = dummy_hb._highband_db({"highband_profiles":"/deleted/old/highband_profiles_v3.json","state_path":str(hbstate)})
    assert loaded_db is not None and route.get("db_found") is True
    assert route.get("db_source") == "registry-state-path"
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
assert abs(FIDELITY_HARD_LIMIT - 0.085) < 1e-9
assert float(ratio.detach()) <= 0.086
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
# High-Band Foundation v1 smoke: checkpoint round-trip, hard high-band mask,
# 44.1 kHz runtime resampling, and exact YH0 bypass behavior.
with tempfile.TemporaryDirectory() as hbtd:
    hbtd = Path(hbtd)
    hb_model = HighBandFoundation(hidden=16, dilations=(1,2,4,8))
    with torch.no_grad():
        hb_model.out_proj.bias.fill_(0.08)
    hb_path = hbtd / "highband_foundation-v1.pt"
    save_highband_foundation(hb_path, hb_model, {"target_sample_rate":48000,"selftest":True})
    hb_loaded, hb_meta = load_highband_foundation(hb_path)
    assert hb_meta["selftest"] is True
    n = 48000
    tt = np.arange(n, dtype=np.float32) / 48000.0
    hb_in = (0.12*np.sin(2*np.pi*220*tt) + 0.03*np.sin(2*np.pi*4400*tt)).astype(np.float32)
    hb_out, hb_stats = apply_highband_foundation(hb_in, 48000, np.full(188,220.0,np.float32), hb_loaded, strength=1.0)
    assert hb_stats["used"] is True and hb_stats["backend"].startswith("highband-foundation")
    assert hb_out.shape == hb_in.shape
    hb_zero, hb_zero_stats = apply_highband_foundation(hb_in, 48000, None, hb_loaded, strength=0.0)
    assert np.array_equal(hb_zero, hb_in) and hb_zero_stats["used"] is False
    hb_441, hb_441_stats = apply_highband_foundation(hb_in[:44100], 44100, np.full(172,110.0,np.float32), hb_loaded, strength=1.0)
    assert hb_441.shape == hb_in[:44100].shape and hb_441_stats["model_sample_rate"] == 48000
    raw = hb_loaded(torch.from_numpy(hb_in).view(1,1,-1), torch.tensor([[[220.0]]]))
    masked = highpass_residual_torch(raw, 48000)
    sp = torch.abs(torch.fft.rfft(masked[0,0]))
    ff = torch.fft.rfftfreq(masked.shape[-1], 1/48000.0)
    low = float(torch.mean(sp[ff < 8500.0]))
    high = float(torch.mean(sp[(ff > 12000.0) & (ff < 20000.0)]))
    assert high > low * 10.0, (low, high)

# High-band continuity regression: a learned foundation that fires only in short
# bursts must no longer leave most voiced frames black above the 24 kHz body's
# Nyquist edge. The source-texture branch acts as a continuous floor, while the
# learned branch still owns its strong events.
    cont_profile = {
        "band_centers_hz": [9000.0,11000.0,13000.0,15000.0,17000.0,19000.0],
        "voiced_db_to_full": [-27.0,-28.0,-30.0,-32.0,-34.0,-36.0],
        "unvoiced_db_to_full": [-25.0,-26.0,-28.0,-30.0,-32.0,-34.0],
        "voiced_harmonic_mix": 0.68,
    }
    cont_out, _ = synthesize_learned_highband(hb_gen, hb_sr, hb_f0, cont_profile, 2468, assist_start_hz=8800.0, restoration_strength=1.0)
    cont_branch = cont_out - hb_gen
    burst_gate = np.zeros_like(cont_branch)
    burst_gate[int(0.10*hb_sr):int(0.24*hb_sr)] = 1.0
    burst_gate[int(0.56*hb_sr):int(0.68*hb_sr)] = 1.0
    sparse_foundation = hb_gen + cont_branch * burst_gate
    hybrid_out, hybrid_stats = blend_foundation_with_continuity(hb_gen, sparse_foundation, cont_out, hb_sr, strength=1.0)
    assert hybrid_stats.get("hybrid_used") is True
    assert hybrid_stats.get("upper_temporal_coverage_after",0.0) > hybrid_stats.get("upper_temporal_coverage_before",0.0) + 0.35
    assert hybrid_stats.get("upper_temporal_coverage_after",0.0) > 0.75
    assert _band_energy(hybrid_out, 12000.0, 20000.0) > _band_energy(sparse_foundation, 12000.0, 20000.0) * 1.6

# Previous 0.2.8ai.10 is the immediate read-only predecessor and must win over older state.
with tempfile.TemporaryDirectory() as td:
    bank = Path(td) / "prev028ai10-bank"; bank.mkdir()
    prev10 = bank / PREVIOUS_028AI10_STATE_CONTAINER
    (prev10 / "articulation").mkdir(parents=True)
    (prev10 / "profile.json").write_text('{"voicebank_id":"prev028ai10"}', encoding="utf-8")
    (prev10 / "manifest.json").write_text('{"profile":{"voicebank_id":"prev028ai10"},"entries":[{"status":"ok","relative_wav":"a.wav"}]}', encoding="utf-8")
    (prev10 / "subbanks.json").write_text('{"format":2,"subbanks":[]}', encoding="utf-8")
    (prev10 / "loudness.json").write_text('{"enabled":true}', encoding="utf-8")
    (prev10 / "highband_profiles_v3.json").write_text('{"format":3,"stats":{}}', encoding="utf-8")
    (prev10 / "articulation" / "index.json").write_text('{"aliases":{}}', encoding="utf-8")
    before = hashlib.sha256((prev10 / "profile.json").read_bytes()).hexdigest()
    resolved, info = resolve_active_state(bank, allow_legacy=False, verify=True)
    assert resolved.resolve() == prev10.resolve() and info.get("predecessor_028ai10") is True and info.get("read_only_fallback") is True
    from yuaz_ddsp_resampler.state import _registry_for_state
    _registry_for_state(bank, prev10, read_only=True)
    assert not (prev10 / "runtime_registry.json").exists(), "0.2.8ai.11 must not write into 0.2.8ai.10 fallback state"
    assert hashlib.sha256((prev10 / "profile.json").read_bytes()).hexdigest() == before

# Previous 0.2.8ai.9 is the immediate read-only predecessor.
with tempfile.TemporaryDirectory() as td:
    bank = Path(td) / "prev028ai9-bank"; bank.mkdir()
    prev9 = bank / PREVIOUS_028AI9_STATE_CONTAINER
    (prev9 / "articulation").mkdir(parents=True)
    (prev9 / "profile.json").write_text('{"voicebank_id":"prev028ai9"}', encoding="utf-8")
    (prev9 / "manifest.json").write_text('{"profile":{"voicebank_id":"prev028ai9"},"entries":[{"status":"ok","relative_wav":"a.wav"}]}', encoding="utf-8")
    (prev9 / "subbanks.json").write_text('{"format":2,"subbanks":[]}', encoding="utf-8")
    (prev9 / "loudness.json").write_text('{"enabled":true}', encoding="utf-8")
    (prev9 / "highband_profiles_v3.json").write_text('{"format":3,"stats":{},"groups":{}}', encoding="utf-8")
    (prev9 / "articulation" / "index.json").write_text('{"aliases":{}}', encoding="utf-8")
    resolved, info = resolve_active_state(bank, allow_legacy=False, verify=True)
    assert resolved.resolve() == prev9.resolve() and info.get("predecessor_028ai9") is True and info.get("read_only_fallback") is True

# Transactional-state regression: an acoustic generation is committed only after
# validation, and a later corrupted active generation falls back to the previous
# pinned generation instead of silently changing sound.
with tempfile.TemporaryDirectory() as td:
    bank = Path(td) / "bank"
    bank.mkdir()
    def make_minimal(st):
        (st / "articulation").mkdir(parents=True, exist_ok=True)
        (st / "profile.json").write_text('{"voicebank_id":"selftest"}', encoding="utf-8")
        (st / "manifest.json").write_text('{"profile":{"voicebank_id":"selftest"},"entries":[{"status":"ok","relative_wav":"a.wav"}]}', encoding="utf-8")
        (st / "subbanks.json").write_text('{"format":2,"subbanks":[]}', encoding="utf-8")
        (st / "loudness.json").write_text('{"enabled":true}', encoding="utf-8")
        (st / "highband_profiles_v3.json").write_text('{"format":3,"stats":{}}', encoding="utf-8")
        (st / "articulation" / "index.json").write_text('{"aliases":{}}', encoding="utf-8")
    g1,s1=begin_generation(bank,"selftest-a")
    make_minimal(s1)
    f1,_=commit_generation(bank,g1,s1,"selftest-a")
    g2,s2=begin_generation(bank,"selftest-b")
    make_minimal(s2)
    f2,_=commit_generation(bank,g2,s2,"selftest-b")
    resolved,info=resolve_active_state(bank,allow_legacy=False,verify=True)
    assert resolved == f2 and info["active"] is True
    # Tamper with a pinned acoustic file. The resolver must reject it and use g1.
    (f2 / "profile.json").write_text('{"voicebank_id":"tampered"}', encoding="utf-8")
    resolved,info=resolve_active_state(bank,allow_legacy=False,verify=True)
    assert resolved == f1 and info["active"] is False
# Migration primitive regression: model/training state is copied from ai.10,
# while derived caches are deliberately omitted and regenerated by ai.11.
with tempfile.TemporaryDirectory() as td:
    bank = Path(td) / "migration-bank"; bank.mkdir()
    src = bank / PREVIOUS_028AI10_STATE_CONTAINER / "generations" / "oldgen"
    make_minimal(src)
    (src / "highband_foundation.pt").write_bytes(b"foundation-r2-selftest")
    (src / "adapter.pt").write_bytes(b"adapter-selftest")
    (src / "timbre_profiles.pt").write_bytes(b"timbre-selftest")
    (src / "cache").mkdir(); (src / "cache" / "derived.bin").write_bytes(b"derived")
    generation, staging = begin_generation(bank, "migrate10")
    clone_state(src, staging, link_caches=False, skip_caches=True)
    final, _ = commit_generation(bank, generation, staging, reason="migrated-from-0.2.8ai.10", acoustic_base="0.2.8ai.11-dual-rate-48k-ddsp-body")
    assert (final / "highband_foundation.pt").read_bytes() == b"foundation-r2-selftest"
    assert (final / "adapter.pt").read_bytes() == b"adapter-selftest"
    assert not (final / "cache").exists(), "ai.11 migration must rebuild derived analysis caches"

# Previous 0.2.8ai.7 fallback has highest predecessor priority and must remain read-only.
with tempfile.TemporaryDirectory() as td:
    bank = Path(td) / "prev028ai7-bank"; bank.mkdir()
    prev7 = bank / PREVIOUS_028AI7_STATE_CONTAINER
    (prev7 / "articulation").mkdir(parents=True)
    (prev7 / "profile.json").write_text('{"voicebank_id":"prev028ai7"}', encoding="utf-8")
    (prev7 / "manifest.json").write_text('{"profile":{"voicebank_id":"prev028ai7"},"entries":[{"status":"ok","relative_wav":"a.wav"}]}', encoding="utf-8")
    (prev7 / "subbanks.json").write_text('{"format":2,"subbanks":[]}', encoding="utf-8")
    (prev7 / "loudness.json").write_text('{"enabled":true}', encoding="utf-8")
    (prev7 / "highband_profiles_v3.json").write_text('{"format":3,"stats":{},"groups":{}}', encoding="utf-8")
    (prev7 / "articulation" / "index.json").write_text('{"aliases":{}}', encoding="utf-8")
    before = hashlib.sha256((prev7 / "profile.json").read_bytes()).hexdigest()
    resolved, info = resolve_active_state(bank, allow_legacy=False, verify=True)
    assert resolved.resolve() == prev7.resolve() and info.get("predecessor_028ai7") is True and info.get("read_only_fallback") is True
    from yuaz_ddsp_resampler.state import _registry_for_state
    _registry_for_state(bank, prev7, read_only=True)
    assert not (prev7 / "runtime_registry.json").exists()
    assert hashlib.sha256((prev7 / "profile.json").read_bytes()).hexdigest() == before

# Previous 0.2.8ai.5 fallback has highest predecessor priority and must remain read-only.
with tempfile.TemporaryDirectory() as td:
    bank = Path(td) / "prev028ai5-bank"; bank.mkdir()
    prev5 = bank / PREVIOUS_028AI5_STATE_CONTAINER
    (prev5 / "articulation").mkdir(parents=True)
    (prev5 / "profile.json").write_text('{"voicebank_id":"prev028ai5"}', encoding="utf-8")
    (prev5 / "manifest.json").write_text('{"profile":{"voicebank_id":"prev028ai5"},"entries":[{"status":"ok","relative_wav":"a.wav"}]}', encoding="utf-8")
    (prev5 / "subbanks.json").write_text('{"format":2,"subbanks":[]}', encoding="utf-8")
    (prev5 / "loudness.json").write_text('{"enabled":true}', encoding="utf-8")
    (prev5 / "highband_profiles_v3.json").write_text('{"format":3,"stats":{}}', encoding="utf-8")
    (prev5 / "articulation" / "index.json").write_text('{"aliases":{}}', encoding="utf-8")
    before = hashlib.sha256((prev5 / "profile.json").read_bytes()).hexdigest()
    resolved, info = resolve_active_state(bank, allow_legacy=False, verify=True)
    assert resolved.resolve() == prev5.resolve() and info.get("predecessor_028ai5") is True and info.get("read_only_fallback") is True
    from yuaz_ddsp_resampler.state import _registry_for_state
    _registry_for_state(bank, prev5, read_only=True)
    assert not (prev5 / "runtime_registry.json").exists(), "0.2.8ai.11 must not write into 0.2.8ai.5 fallback state"
    assert hashlib.sha256((prev5 / "profile.json").read_bytes()).hexdigest() == before

# Previous 0.2.8ai.4 fallback has highest predecessor priority and must remain read-only.
with tempfile.TemporaryDirectory() as td:
    bank = Path(td) / "prev028ai3-bank"; bank.mkdir()
    prev4 = bank / PREVIOUS_028AI4_STATE_CONTAINER
    (prev4 / "articulation").mkdir(parents=True)
    (prev4 / "profile.json").write_text('{"voicebank_id":"prev028ai3"}', encoding="utf-8")
    (prev4 / "manifest.json").write_text('{"profile":{"voicebank_id":"prev028ai3"},"entries":[{"status":"ok","relative_wav":"a.wav"}]}', encoding="utf-8")
    (prev4 / "subbanks.json").write_text('{"format":2,"subbanks":[]}', encoding="utf-8")
    (prev4 / "loudness.json").write_text('{"enabled":true}', encoding="utf-8")
    (prev4 / "highband_profiles_v3.json").write_text('{"format":3,"stats":{}}', encoding="utf-8")
    (prev4 / "articulation" / "index.json").write_text('{"aliases":{}}', encoding="utf-8")
    before = hashlib.sha256((prev4 / "profile.json").read_bytes()).hexdigest()
    resolved, info = resolve_active_state(bank, allow_legacy=False, verify=True)
    assert resolved.resolve() == prev4.resolve() and info.get("predecessor_028ai4") is True and info.get("read_only_fallback") is True
    from yuaz_ddsp_resampler.state import _registry_for_state
    _registry_for_state(bank, prev4, read_only=True)
    assert not (prev4 / "runtime_registry.json").exists(), "0.2.8ai.11 must not write into 0.2.8ai.4 fallback state"
    assert hashlib.sha256((prev4 / "profile.json").read_bytes()).hexdigest() == before


# Previous 0.2.8ai.3 fallback has highest predecessor priority and must remain read-only.
with tempfile.TemporaryDirectory() as td:
    bank = Path(td) / "prev028ai3-bank"; bank.mkdir()
    prev3 = bank / PREVIOUS_028AI3_STATE_CONTAINER
    (prev3 / "articulation").mkdir(parents=True)
    (prev3 / "profile.json").write_text('{"voicebank_id":"prev028ai3"}', encoding="utf-8")
    (prev3 / "manifest.json").write_text('{"profile":{"voicebank_id":"prev028ai3"},"entries":[{"status":"ok","relative_wav":"a.wav"}]}', encoding="utf-8")
    (prev3 / "subbanks.json").write_text('{"format":2,"subbanks":[]}', encoding="utf-8")
    (prev3 / "loudness.json").write_text('{"enabled":true}', encoding="utf-8")
    (prev3 / "highband_profiles_v3.json").write_text('{"format":3,"stats":{}}', encoding="utf-8")
    (prev3 / "articulation" / "index.json").write_text('{"aliases":{}}', encoding="utf-8")
    before = hashlib.sha256((prev3 / "profile.json").read_bytes()).hexdigest()
    resolved, info = resolve_active_state(bank, allow_legacy=False, verify=True)
    assert resolved.resolve() == prev3.resolve() and info.get("predecessor_028ai3") is True and info.get("read_only_fallback") is True
    from yuaz_ddsp_resampler.state import _registry_for_state
    _registry_for_state(bank, prev3, read_only=True)
    assert not (prev3 / "runtime_registry.json").exists(), "0.2.8ai.11 must not write into 0.2.8ai.3 fallback state"
    assert hashlib.sha256((prev3 / "profile.json").read_bytes()).hexdigest() == before

# Previous 0.2.8ai.2 fallback has next predecessor priority and must remain read-only.
with tempfile.TemporaryDirectory() as td:
    bank = Path(td) / "prev028ai2-bank"; bank.mkdir()
    prev2 = bank / PREVIOUS_028AI2_STATE_CONTAINER
    (prev2 / "articulation").mkdir(parents=True)
    (prev2 / "profile.json").write_text('{"voicebank_id":"prev028ai2"}', encoding="utf-8")
    (prev2 / "manifest.json").write_text('{"profile":{"voicebank_id":"prev028ai2"},"entries":[{"status":"ok","relative_wav":"a.wav"}]}', encoding="utf-8")
    (prev2 / "subbanks.json").write_text('{"format":2,"subbanks":[]}', encoding="utf-8")
    (prev2 / "loudness.json").write_text('{"enabled":true}', encoding="utf-8")
    (prev2 / "highband_profiles_v3.json").write_text('{"format":3,"stats":{}}', encoding="utf-8")
    (prev2 / "articulation" / "index.json").write_text('{"aliases":{}}', encoding="utf-8")
    before = hashlib.sha256((prev2 / "profile.json").read_bytes()).hexdigest()
    resolved, info = resolve_active_state(bank, allow_legacy=False, verify=True)
    assert resolved.resolve() == prev2.resolve() and info.get("predecessor_028ai2") is True and info.get("read_only_fallback") is True
    from yuaz_ddsp_resampler.state import _registry_for_state
    _registry_for_state(bank, prev2, read_only=True)
    assert not (prev2 / "runtime_registry.json").exists(), "0.2.8ai.11 must not write into 0.2.8ai.2 fallback state"
    assert hashlib.sha256((prev2 / "profile.json").read_bytes()).hexdigest() == before

# Previous 0.2.8ai.1 fallback has next predecessor priority and must remain read-only.
with tempfile.TemporaryDirectory() as td:
    bank = Path(td) / "prev028ai1-bank"; bank.mkdir()
    prev1 = bank / PREVIOUS_028AI1_STATE_CONTAINER
    (prev1 / "articulation").mkdir(parents=True)
    (prev1 / "profile.json").write_text('{"voicebank_id":"prev028ai1"}', encoding="utf-8")
    (prev1 / "manifest.json").write_text('{"profile":{"voicebank_id":"prev028ai1"},"entries":[{"status":"ok","relative_wav":"a.wav"}]}', encoding="utf-8")
    (prev1 / "subbanks.json").write_text('{"format":2,"subbanks":[]}', encoding="utf-8")
    (prev1 / "loudness.json").write_text('{"enabled":true}', encoding="utf-8")
    (prev1 / "highband_profiles_v3.json").write_text('{"format":3,"stats":{}}', encoding="utf-8")
    (prev1 / "articulation" / "index.json").write_text('{"aliases":{}}', encoding="utf-8")
    before = hashlib.sha256((prev1 / "profile.json").read_bytes()).hexdigest()
    resolved, info = resolve_active_state(bank, allow_legacy=False, verify=True)
    assert resolved.resolve() == prev1.resolve() and info.get("predecessor_028ai1") is True and info.get("read_only_fallback") is True
    from yuaz_ddsp_resampler.state import _registry_for_state
    _registry_for_state(bank, prev1, read_only=True)
    assert not (prev1 / "runtime_registry.json").exists(), "0.2.8ai.11 must not write into 0.2.8ai.1 fallback state"
    assert hashlib.sha256((prev1 / "profile.json").read_bytes()).hexdigest() == before

# Previous 0.2.8ai fallback has higher priority than AI.3 and must remain read-only.
with tempfile.TemporaryDirectory() as td:
    bank = Path(td) / "prev028-bank"; bank.mkdir()
    prev = bank / PREVIOUS_028_STATE_CONTAINER
    (prev / "articulation").mkdir(parents=True)
    (prev / "profile.json").write_text('{"voicebank_id":"prev028"}', encoding="utf-8")
    (prev / "manifest.json").write_text('{"profile":{"voicebank_id":"prev028"},"entries":[{"status":"ok","relative_wav":"a.wav"}]}', encoding="utf-8")
    (prev / "subbanks.json").write_text('{"format":2,"subbanks":[]}', encoding="utf-8")
    (prev / "loudness.json").write_text('{"enabled":true}', encoding="utf-8")
    (prev / "highband_profiles_v3.json").write_text('{"format":3,"stats":{}}', encoding="utf-8")
    (prev / "articulation" / "index.json").write_text('{"aliases":{}}', encoding="utf-8")
    before = hashlib.sha256((prev / "profile.json").read_bytes()).hexdigest()
    resolved, info = resolve_active_state(bank, allow_legacy=False, verify=True)
    assert resolved.resolve() == prev.resolve() and info.get("predecessor_028") is True and info.get("read_only_fallback") is True
    from yuaz_ddsp_resampler.state import _registry_for_state
    _registry_for_state(bank, prev, read_only=True)
    assert not (prev / "runtime_registry.json").exists(), "0.2.8ai.11 must not write into 0.2.8ai fallback state"
    assert hashlib.sha256((prev / "profile.json").read_bytes()).hexdigest() == before

# Predecessor AI.3 fallback is higher priority than RC4.2 and must remain read-only.
with tempfile.TemporaryDirectory() as td:
    bank = Path(td) / "predecessor-bank"; bank.mkdir()
    pred = bank / PREDECESSOR_AI_STATE_CONTAINER
    (pred / "articulation").mkdir(parents=True)
    (pred / "profile.json").write_text('{"voicebank_id":"pred"}', encoding="utf-8")
    (pred / "manifest.json").write_text('{"profile":{"voicebank_id":"pred"},"entries":[{"status":"ok","relative_wav":"a.wav"}]}', encoding="utf-8")
    (pred / "subbanks.json").write_text('{"format":2,"subbanks":[]}', encoding="utf-8")
    (pred / "loudness.json").write_text('{"enabled":true}', encoding="utf-8")
    (pred / "highband_profiles_v3.json").write_text('{"format":3,"stats":{}}', encoding="utf-8")
    (pred / "articulation" / "index.json").write_text('{"aliases":{}}', encoding="utf-8")
    before = hashlib.sha256((pred / "profile.json").read_bytes()).hexdigest()
    resolved, info = resolve_active_state(bank, allow_legacy=False, verify=True)
    assert resolved.resolve() == pred.resolve() and info.get("predecessor_ai") is True and info.get("read_only_fallback") is True
    assert not (pred / "runtime_registry.json").exists()
    from yuaz_ddsp_resampler.state import _registry_for_state
    _registry_for_state(bank, pred, read_only=True)
    assert not (pred / "runtime_registry.json").exists(), "0.2.8ai.11 must not write cache files into AI.3 fallback state"
    assert hashlib.sha256((pred / "profile.json").read_bytes()).hexdigest() == before

# Predecessor/stable fallback isolation: 0.2.8ai.11 may READ 0.2.8ai.4/0.2.8ai.3/0.2.8ai.2/0.2.8ai.1/0.2.8ai/AI.3/RC4.2 state when its own state is absent,
# but committing a generation must create only .yuaz-0.2.8ai11.
with tempfile.TemporaryDirectory() as td:
    bank = Path(td) / "stable-bank"; bank.mkdir()
    stable = bank / STABLE_STATE_CONTAINER
    (stable / "articulation").mkdir(parents=True)
    (stable / "profile.json").write_text('{"voicebank_id":"stable"}', encoding="utf-8")
    (stable / "manifest.json").write_text('{"profile":{"voicebank_id":"stable"},"entries":[{"status":"ok","relative_wav":"a.wav"}]}', encoding="utf-8")
    (stable / "subbanks.json").write_text('{"format":2,"subbanks":[]}', encoding="utf-8")
    (stable / "loudness.json").write_text('{"enabled":true}', encoding="utf-8")
    (stable / "highband_profiles_v3.json").write_text('{"format":3,"stats":{}}', encoding="utf-8")
    (stable / "articulation" / "index.json").write_text('{"aliases":{}}', encoding="utf-8")
    before = hashlib.sha256((stable / "profile.json").read_bytes()).hexdigest()
    resolved, info = resolve_active_state(bank, allow_legacy=False, verify=True)
    assert resolved.resolve() == stable.resolve() and info.get("read_only_fallback") is True
    g, st = begin_generation(bank, "ai-isolation")
    make_minimal(st)
    commit_generation(bank, g, st, "ai-isolation")
    assert (bank / STATE_CONTAINER / "ACTIVE.json").is_file()
    assert hashlib.sha256((stable / "profile.json").read_bytes()).hexdigest() == before

# 0.2.8ai.11 dual-rate DDSP regression. The frozen 24 kHz neural envelope is
# extended only above its original Nyquist while the oscillator/noise synthesis
# itself runs at 48 kHz. YH0 therefore has a real full-band body.
class _DummyDecoderBase:
    pass
_Dual = make_adaptive_decoder_class(_DummyDecoderBase)
d = object.__new__(_Dual)
d.sample_rate = 24000
d.n_harmonics = 64
d.fft_size = 1024
d.hop_length = 256
d.encoder_hop_length = 320
frames = 10
f0_fb = torch.full((1,1,frames), 220.0)
S_fb = torch.ones((1, d.fft_size//2 + 1, frames), dtype=torch.float32) * 0.20
A_fb = torch.ones((1, 16, frames), dtype=torch.float32) * 0.30
G_fb = torch.ones((1,1,frames), dtype=torch.float32) * 0.75
torch.manual_seed(1234)
fb_wav, fb_stats = d._synthesize_fullband_body(f0_fb, S_fb, A_fb, A_fb, G_fb, 48000)
assert fb_wav.ndim == 3 and fb_wav.shape[-1] == frames * d.encoder_hop_length * 2
assert fb_stats["sample_rate"] == 48000 and fb_stats["fft_size"] == 2048
assert fb_stats["harmonic_count"] >= 100, fb_stats
fb_np = fb_wav[0,0].detach().cpu().numpy().astype(np.float32)
# Construct a 24 kHz compatibility body and confirm the complementary crossover
# keeps the output finite and retains measurable energy above 12 kHz.
legacy_len = int(round(len(fb_np) * 44100 / 48000))
legacy_src = np.sin(2*np.pi*440*np.arange(max(1, int(round(len(fb_np)*24000/48000))))/24000).astype(np.float32) * 0.05
import librosa as _librosa
legacy_44 = _librosa.resample(legacy_src, orig_sr=24000, target_sr=44100).astype(np.float32)[:legacy_len]
if len(legacy_44) < legacy_len: legacy_44 = np.pad(legacy_44,(0,legacy_len-len(legacy_44)))
fb_44 = _librosa.resample(fb_np, orig_sr=48000, target_sr=44100).astype(np.float32)[:legacy_len]
if len(fb_44) < legacy_len: fb_44 = np.pad(fb_44,(0,legacy_len-len(fb_44)))
fb_mix, fb_mix_stats = blend_dualrate_fullband_body(legacy_44, fb_44, 44100, 9000.0, 12100.0)
assert fb_mix_stats.get("used") is True
assert np.isfinite(fb_mix).all()
def _band_fft_rms(x, sr, lo, hi):
    x=np.asarray(x,dtype=np.float64)
    X=np.fft.rfft(x*np.hanning(len(x)))
    F=np.fft.rfftfreq(len(x),1.0/sr)
    m=(F>=lo)&(F<hi)
    return float(np.sqrt(np.mean(np.abs(X[m])**2)+1e-18))
fb_upper=_band_fft_rms(fb_mix,44100,12500,18000)
assert fb_upper > 1e-7, fb_upper
print(f"Dual-rate 48 kHz DDSP body regression: harmonics={fb_stats['harmonic_count']} fft={fb_stats['fft_size']} upper12.5-18k={fb_upper:.6g}")

# Complementary Nyquist-crossover regression: the seam immediately above a
# 24 kHz-style body edge must become materially stronger without turning the
# whole 15-20 kHz band into a broadband block.
seam_sr = 44100
seam_n = seam_sr
seam_t = np.arange(seam_n, dtype=np.float64) / seam_sr
seam_base = np.zeros(seam_n, dtype=np.float64)
for k in range(1, 52):
    hz = 220.0 * k
    if hz >= 11200.0:
        break
    seam_base += (0.28 / (k ** 0.72)) * np.sin(2*np.pi*hz*seam_t)
seam_base = seam_base.astype(np.float32)
rng = np.random.default_rng(991)
donor_noise = rng.standard_normal(seam_n)
donor_spec = np.fft.rfft(donor_noise)
donor_freq = np.fft.rfftfreq(seam_n, 1.0/seam_sr)
donor_mask = np.clip((donor_freq-9000.0)/1800.0,0,1)
donor_mask = donor_mask*donor_mask*(3-2*donor_mask)
donor_mask *= np.clip((20500.0-donor_freq)/2500.0,0,1)
donor = np.fft.irfft(donor_spec*donor_mask, n=seam_n).real
donor *= 0.0012 / max(np.sqrt(np.mean(donor**2)),1e-9)
seam_cont = seam_base + donor.astype(np.float32)
seam_found = seam_base + (0.18*donor).astype(np.float32)
seam_out, seam_stats = blend_foundation_with_continuity(seam_base, seam_found, seam_cont, seam_sr, strength=1.0)
def _br(x,lo,hi):
    X=np.fft.rfft(np.asarray(x,dtype=np.float64)*np.hanning(len(x)))
    F=np.fft.rfftfreq(len(x),1.0/seam_sr)
    m=(F>=lo)&(F<hi)
    return float(np.sqrt(np.mean(np.abs(X[m])**2)+1e-18))
pre_ratio=_br(seam_found,12000,14500)/max(_br(seam_base,8500,10500),1e-12)
post_ratio=_br(seam_out,12000,14500)/max(_br(seam_out,8500,10500),1e-12)
assert seam_stats.get('hybrid_used') is True
assert seam_stats.get('nyquist_body_taper_amount',0) > 0.20
assert post_ratio > pre_ratio * 1.8, (pre_ratio, post_ratio, seam_stats)
assert _br(seam_out,16000,19500) < _br(seam_out,12000,14500) * 0.85
print(f"Nyquist seam crossover regression: before={pre_ratio:.6g} after={post_ratio:.6g} ratio={post_ratio/max(pre_ratio,1e-12):.2f}x")

print("0.2.8ai.11 dual-rate 48 kHz DDSP body + Foundation refinement self-test OK")
PY
