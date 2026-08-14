#!/usr/bin/env python3
import torch
import torch.nn.functional as F


def _interp_curve(curve, frames, device, dtype):
    if curve is None:
        return torch.zeros((1, 1, int(frames)), device=device, dtype=dtype)
    if not torch.is_tensor(curve):
        curve = torch.as_tensor(curve, device=device, dtype=dtype)
    curve = curve.to(device=device, dtype=dtype)
    if curve.dim() == 0:
        curve = curve.view(1, 1, 1)
    elif curve.dim() == 1:
        curve = curve.view(1, 1, -1)
    elif curve.dim() == 2:
        curve = curve.unsqueeze(1)
    if curve.shape[-1] != int(frames):
        curve = F.interpolate(curve, size=int(frames), mode="linear", align_corners=False)
    return curve.clamp(-1.0, 1.0)


def _warp_with_source_frequency(envelope, source_freq):
    b, c, t = envelope.shape
    if c < 2:
        return envelope
    source_freq = source_freq.clamp(0.0, 1.0)
    grid_y = source_freq * 2.0 - 1.0
    time = torch.linspace(-1.0, 1.0, t, device=envelope.device, dtype=envelope.dtype).view(1, 1, t)
    grid_x = time.expand(b, c, t)
    grid = torch.stack([grid_x, grid_y.expand(b, c, t)], dim=-1)
    warped = F.grid_sample(
        envelope.unsqueeze(1), grid, mode="bilinear",
        padding_mode="border", align_corners=True,
    )
    return warped[:, 0]


def _frequency_warp(envelope, shift_semitones, voiced=None):
    b, c, t = envelope.shape
    if c < 2:
        return envelope
    shift = _interp_curve(shift_semitones, t, envelope.device, envelope.dtype)
    if float(torch.max(torch.abs(shift)).detach().cpu()) < 1e-7:
        return envelope
    scale = torch.pow(envelope.new_tensor(2.0), shift / 12.0)
    freq = torch.linspace(0.0, 1.0, c, device=envelope.device, dtype=envelope.dtype).view(1, c, 1)
    # Keep the extreme high band progressively anchored so stronger formant shifts
    # do not drag unvoiced/high-frequency articulation with the vocal-tract warp.
    anchor = 1.0 - 0.78 * torch.clamp((freq - 0.46) / 0.44, 0.0, 1.0)
    local_scale = 1.0 + (scale - 1.0) * anchor
    source_freq = freq * local_scale
    warped = _warp_with_source_frequency(envelope, source_freq)
    if voiced is None:
        return warped
    mask = voiced
    if mask.shape[-1] != t:
        mask = F.interpolate(mask, size=t, mode="nearest")
    return envelope * (1.0 - mask) + warped * mask


def _mouth_formant_warp(envelope, mouth, sample_rate, voiced):
    b, c, t = envelope.shape
    if c < 2:
        return envelope
    control = _interp_curve(mouth, t, envelope.device, envelope.dtype)
    if float(torch.max(torch.abs(control)).detach().cpu()) < 1e-7:
        return envelope
    nyquist = max(1.0, float(sample_rate) * 0.5)
    freq = torch.linspace(0.0, 1.0, c, device=envelope.device, dtype=envelope.dtype).view(1, c, 1)
    hz = freq * nyquist
    # Jaw/opening control: move the first-formant region most strongly, with a
    # gentler secondary displacement in the F2 region. Positive = more open.
    f1_weight = torch.exp(-0.5 * torch.square((hz - 760.0) / 620.0))
    f2_weight = 0.38 * torch.exp(-0.5 * torch.square((hz - 2050.0) / 1050.0))
    shift_hz = (255.0 * f1_weight + 115.0 * f2_weight) * control
    source_hz = hz - shift_hz
    source_freq = source_hz / nyquist
    warped = _warp_with_source_frequency(envelope, source_freq)
    mask = voiced
    if mask.shape[-1] != t:
        mask = F.interpolate(mask, size=t, mode="nearest")
    return envelope * (1.0 - mask) + warped * mask


def _move_toward(value, control, target, amount):
    positive = torch.clamp(control, 0.0, 1.0)
    negative = torch.clamp(-control, 0.0, 1.0)
    if target >= 0.5:
        out = value + positive * amount * (1.0 - value)
        out = out - negative * amount * value
    else:
        out = value - positive * amount * value
        out = out + negative * amount * (1.0 - value)
    return out


def apply_decoder_vocal_controls(spectral_envelope, ap_bands, gate, f0, frame_controls, sample_rate=24000, learned_controls=()):
    if not frame_controls:
        return spectral_envelope, ap_bands, gate
    frames = spectral_envelope.shape[-1]
    device = spectral_envelope.device
    dtype = spectral_envelope.dtype
    learned_controls = set(str(x) for x in (learned_controls or ()))
    # 0.2.8ai.11 uses an interpretable low-gain carrier underneath learned packs.
    # A valid learned checkpoint must never be able to make a UI axis silently dead.
    # The carrier is deliberately restrained when a learned pack owns the axis and
    # becomes the full deterministic fallback only when no learned pack is present.
    def carrier(name, learned_scale=0.34, fallback_scale=1.0):
        return float(learned_scale if name in learned_controls else fallback_scale)
    tension_scale = carrier("tension", 0.34)
    breathiness_scale = carrier("breathiness", 0.30)
    voicing_scale = carrier("voicing", 0.34)
    gender_scale = carrier("gender_formant", 0.34)
    mouth_scale = carrier("mouth", 0.34)
    falsetto_scale = carrier("falsetto", 0.30, 0.62)
    mixed_scale = carrier("mixed_voice", 0.30, 0.62)
    pharyngeal_scale = carrier("pharyngeal", 0.30, 0.62)
    tension = _interp_curve(frame_controls.get("tension"), frames, device, dtype)
    breathiness = _interp_curve(frame_controls.get("breathiness"), frames, device, dtype)
    voicing = _interp_curve(frame_controls.get("voicing"), frames, device, dtype)
    gender = _interp_curve(frame_controls.get("gender_formant"), frames, device, dtype)
    mouth = _interp_curve(frame_controls.get("mouth"), frames, device, dtype)
    falsetto = _interp_curve(frame_controls.get("falsetto"), frames, device, dtype)
    mixed_voice = _interp_curve(frame_controls.get("mixed_voice"), frames, device, dtype)
    pharyngeal = _interp_curve(frame_controls.get("pharyngeal"), frames, device, dtype)
    tension_eff = tension * tension_scale
    voicing_eff = voicing * voicing_scale
    gender_eff = gender * gender_scale
    mouth_eff = mouth * mouth_scale
    falsetto_eff = torch.clamp(falsetto, 0.0, 1.0) * falsetto_scale
    mixed_eff = torch.clamp(mixed_voice, 0.0, 1.0) * mixed_scale
    pharyngeal_eff = torch.clamp(pharyngeal, 0.0, 1.0) * pharyngeal_scale
    voiced = (f0 > 1.0).to(dtype)
    if voiced.shape[-1] != frames:
        voiced = F.interpolate(voiced, size=frames, mode="nearest")

    out_s = spectral_envelope

    # Gender/Formant is a vocal-tract-envelope warp, not a pitch control.  The
    # stronger rc.4.2 range is restricted to voiced frames and anchored in the
    # high band so consonants stay close to the preserved articulation path.
    if float(torch.max(torch.abs(gender_eff)).detach().cpu()) > 1e-7:
        gender_curve = torch.sign(gender_eff) * torch.pow(torch.abs(gender_eff), 0.86)
        formant_shift = 5.8 * gender_curve
        out_s = _frequency_warp(out_s, formant_shift, voiced=voiced)

    # Mouth/Resonance now moves low/mid formant regions instead of behaving like
    # a tiny broad EQ.  It remains voiced-only to protect consonant identity.
    if float(torch.max(torch.abs(mouth_eff)).detach().cpu()) > 1e-7:
        out_s = _mouth_formant_warp(out_s, mouth_eff, sample_rate, voiced)

    c = out_s.shape[1]
    freq = torch.linspace(0.0, 1.0, c, device=device, dtype=dtype).view(1, c, 1)
    nyquist = max(1.0, float(sample_rate) * 0.5)
    hz = freq * nyquist

    # Tension is intentionally independent from aperiodicity/noise in rc.4.2.
    # It changes spectral tilt/presence only: positive values redistribute energy
    # toward higher harmonic regions, negative values soften that distribution.
    low = torch.exp(-0.5 * torch.square((hz - 520.0) / 700.0))
    presence = torch.exp(-0.5 * torch.square((hz - 3200.0) / 2300.0))
    upper = torch.exp(-0.5 * torch.square((hz - 6500.0) / 3000.0))
    tension_shape = -0.34 * low + 0.72 * presence + 0.38 * upper
    tension_gain = torch.exp(0.92 * tension_eff * tension_shape * voiced)

    # Mouth gets an additional restrained resonance component after its formant
    # displacement so +/-100 is clearly audible without becoming an EQ-only flag.
    mouth_presence = (
        0.70 * torch.exp(-0.5 * torch.square((hz - 1700.0) / 1050.0))
        + 0.28 * torch.exp(-0.5 * torch.square((hz - 3100.0) / 1500.0))
        - 0.18 * torch.exp(-0.5 * torch.square((hz - 6200.0) / 2600.0))
    )
    mouth_gain = torch.exp(0.42 * mouth_eff * mouth_presence * voiced)

    # Gender gets a restrained low-vs-upper vocal-tract tilt in addition to the
    # frequency warp. This keeps +/- directions measurable even on very smooth
    # envelopes where a pure warp can be nearly invariant.
    gender_shape = (
        0.42 * torch.exp(-0.5 * torch.square((hz - 900.0) / 950.0))
        - 0.34 * torch.exp(-0.5 * torch.square((hz - 3150.0) / 1900.0))
    )
    gender_gain = torch.exp(0.62 * gender_eff * gender_shape * voiced)

    # Positive voicing adds body/presence rather than changing AP.  Its actual
    # harmonic-vs-noise control is handled by gate below.
    voice_body = (
        0.55 * torch.exp(-0.5 * torch.square((hz - 820.0) / 850.0))
        + 0.32 * torch.exp(-0.5 * torch.square((hz - 2100.0) / 1500.0))
    )
    voicing_gain = torch.exp(0.28 * voicing_eff * voice_body * voiced)
    # Low-gain technique carriers keep YF/YX/YP perceptually distinct even if a
    # learned residual collapses. The learned pack remains the dominant nonlinear
    # component; these fixed bases only define an interpretable direction.
    falsetto_shape = (-0.46 * torch.exp(-0.5 * torch.square((hz - 720.0) / 900.0))
                      + 0.30 * torch.exp(-0.5 * torch.square((hz - 4200.0) / 2600.0)))
    mixed_shape = (0.42 * torch.exp(-0.5 * torch.square((hz - 1700.0) / 1300.0))
                   + 0.20 * torch.exp(-0.5 * torch.square((hz - 3600.0) / 2200.0)))
    pharyngeal_shape = (0.52 * torch.exp(-0.5 * torch.square((hz - 1350.0) / 850.0))
                        + 0.38 * torch.exp(-0.5 * torch.square((hz - 2850.0) / 1250.0))
                        - 0.12 * torch.exp(-0.5 * torch.square((hz - 650.0) / 700.0)))
    technique_gain = torch.exp((0.72 * falsetto_eff * falsetto_shape
                                + 0.62 * mixed_eff * mixed_shape
                                + 0.62 * pharyngeal_eff * pharyngeal_shape) * voiced)
    out_s = (out_s * tension_gain * mouth_gain * gender_gain * voicing_gain * technique_gain).clamp(min=1e-7)

    ap_frames = ap_bands.shape[-1]
    b_ap = _interp_curve(breathiness, ap_frames, ap_bands.device, ap_bands.dtype)
    voiced_ap = (f0 > 1.0).to(ap_bands.dtype)
    if voiced_ap.shape[-1] != ap_frames:
        voiced_ap = F.interpolate(voiced_ap, size=ap_frames, mode="nearest")
    ap_freq = torch.linspace(0.0, 1.0, ap_bands.shape[1], device=ap_bands.device, dtype=ap_bands.dtype).view(1, -1, 1)
    out_ap = ap_bands
    # The learned model owns only directly supervised positive technique axes.
    # Deterministic fallback remains for unsupported/opposite directions.
    b_pos = torch.clamp(b_ap, 0.0, 1.0) * breathiness_scale
    b_neg = torch.clamp(-b_ap, 0.0, 1.0)
    if float(torch.max(b_pos + b_neg).detach().cpu()) > 1e-7:
        ap_shape = (0.20 + 0.80 * torch.pow(ap_freq, 0.72)) * voiced_ap
        out_ap = out_ap + b_pos * 0.58 * ap_shape * (1.0 - out_ap)
        out_ap = out_ap - b_neg * 0.46 * ap_shape * out_ap
        out_ap = out_ap.clamp(0.012, 0.988)
    # Falsetto carrier adds restrained aperiodicity; mixed voice slightly reduces it.
    f_ap = _interp_curve(falsetto_eff, ap_frames, ap_bands.device, ap_bands.dtype)
    x_ap = _interp_curve(mixed_eff, ap_frames, ap_bands.device, ap_bands.dtype)
    if float(torch.max(f_ap + x_ap).detach().cpu()) > 1e-7:
        air_shape = (0.30 + 0.70 * torch.pow(ap_freq, 0.78)) * voiced_ap
        out_ap = out_ap + f_ap * 0.24 * air_shape * (1.0 - out_ap)
        out_ap = out_ap - x_ap * 0.08 * air_shape * out_ap
        out_ap = out_ap.clamp(0.012, 0.988)

    gate_frames = gate.shape[-1]
    b_g = _interp_curve(breathiness, gate_frames, gate.device, gate.dtype)
    v_g = _interp_curve(voicing, gate_frames, gate.device, gate.dtype)
    voiced_g = (f0 > 1.0).to(gate.dtype)
    if voiced_g.shape[-1] != gate_frames:
        voiced_g = F.interpolate(voiced_g, size=gate_frames, mode="nearest")
    out_gate = gate
    v_pos = torch.clamp(v_g, 0.0, 1.0) * voiced_g * voicing_scale
    v_neg = torch.clamp(-v_g, 0.0, 1.0) * voiced_g * voicing_scale
    if float(torch.max(v_pos + v_neg).detach().cpu()) > 1e-7:
        out_gate = out_gate + 0.52 * v_pos * (1.0 - out_gate)
        out_gate = out_gate - 0.52 * v_neg * out_gate
        out_gate = out_gate.clamp(0.02, 0.98)
    b_pos_g = torch.clamp(b_g, 0.0, 1.0) * voiced_g * breathiness_scale
    b_neg_g = torch.clamp(-b_g, 0.0, 1.0) * voiced_g
    if float(torch.max(b_pos_g + b_neg_g).detach().cpu()) > 1e-7:
        out_gate = out_gate - 0.18 * b_pos_g * out_gate
        out_gate = out_gate + 0.10 * b_neg_g * (1.0 - out_gate)
        out_gate = out_gate.clamp(0.02, 0.98)
    f_g = _interp_curve(falsetto_eff, gate_frames, gate.device, gate.dtype) * voiced_g
    x_g = _interp_curve(mixed_eff, gate_frames, gate.device, gate.dtype) * voiced_g
    if float(torch.max(f_g + x_g).detach().cpu()) > 1e-7:
        out_gate = out_gate - 0.24 * f_g * out_gate
        out_gate = out_gate + 0.20 * x_g * (1.0 - out_gate)
        out_gate = out_gate.clamp(0.02, 0.98)
    return out_s, out_ap, out_gate
