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
    control = torch.sign(control) * torch.pow(torch.abs(control), 0.78)
    nyquist = max(1.0, float(sample_rate) * 0.5)
    freq = torch.linspace(0.0, 1.0, c, device=envelope.device, dtype=envelope.dtype).view(1, c, 1)
    hz = freq * nyquist
    f1_weight = torch.exp(-0.5 * torch.square((hz - 720.0) / 500.0))
    f2_weight = torch.exp(-0.5 * torch.square((hz - 1850.0) / 760.0))
    shift_hz = (430.0 * f1_weight + 105.0 * f2_weight) * control
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

    def carrier(name, learned_scale=0.60, fallback_scale=1.0):
        return float(learned_scale if name in learned_controls else fallback_scale)

    tension_scale = carrier("tension", 0.88)
    breathiness_scale = carrier("breathiness", 0.34)
    voicing_pos_scale = carrier("voicing", 1.25)
    voicing_neg_scale = carrier("voicing", 0.70)
    gender_scale = carrier("gender_formant", 0.85)
    mouth_scale = carrier("mouth", 0.95)
    falsetto_spectral_scale = carrier("falsetto", 0.88, 0.96)
    falsetto_noise_scale = carrier("falsetto", 0.0, 0.18)
    mixed_scale = carrier("mixed_voice", 0.95, 0.95)
    pharyngeal_scale = carrier("pharyngeal", 0.95, 0.95)

    tension = _interp_curve(frame_controls.get("tension"), frames, device, dtype)
    breathiness = _interp_curve(frame_controls.get("breathiness"), frames, device, dtype)
    voicing = _interp_curve(frame_controls.get("voicing"), frames, device, dtype)
    gender = _interp_curve(frame_controls.get("gender_formant"), frames, device, dtype)
    mouth = _interp_curve(frame_controls.get("mouth"), frames, device, dtype)
    falsetto = _interp_curve(frame_controls.get("falsetto"), frames, device, dtype)
    mixed_voice = _interp_curve(frame_controls.get("mixed_voice"), frames, device, dtype)
    pharyngeal = _interp_curve(frame_controls.get("pharyngeal"), frames, device, dtype)

    tension_pos_eff = torch.pow(torch.clamp(tension, 0.0, 1.0), 1.05) * tension_scale
    tension_neg_eff = torch.pow(torch.clamp(-tension, 0.0, 1.0), 0.95) * tension_scale * 1.08
    tension_eff = tension_pos_eff - tension_neg_eff
    voicing_pos_eff = torch.pow(torch.clamp(voicing, 0.0, 1.0), 0.72) * voicing_pos_scale
    voicing_neg_eff = torch.clamp(-voicing, 0.0, 1.0) * voicing_neg_scale
    gender_eff = gender * gender_scale
    mouth_eff = mouth * mouth_scale
    falsetto_eff = torch.pow(torch.clamp(falsetto, 0.0, 1.0), 0.72) * falsetto_spectral_scale
    falsetto_noise_eff = torch.clamp(falsetto, 0.0, 1.0) * falsetto_noise_scale
    mixed_eff = torch.pow(torch.clamp(mixed_voice, 0.0, 1.0), 0.72) * mixed_scale
    pharyngeal_eff = torch.pow(torch.clamp(pharyngeal, 0.0, 1.0), 0.72) * pharyngeal_scale
    voiced = (f0 > 1.0).to(dtype)
    if voiced.shape[-1] != frames:
        voiced = F.interpolate(voiced, size=frames, mode="nearest")

    f0_env = f0.to(device=device, dtype=dtype)
    if f0_env.dim() == 1:
        f0_env = f0_env.view(1, 1, -1)
    elif f0_env.dim() == 2:
        f0_env = f0_env.unsqueeze(1)
    if f0_env.shape[-1] != frames:
        f0_env = F.interpolate(f0_env, size=frames, mode="linear", align_corners=False)
    f0_env = torch.clamp(f0_env, min=60.0)

    out_s = spectral_envelope

    if float(torch.max(torch.abs(gender_eff)).detach().cpu()) > 1e-7:
        gender_curve = torch.sign(gender_eff) * torch.pow(torch.abs(gender_eff), 0.86)
        formant_shift = 5.2 * gender_curve
        out_s = _frequency_warp(out_s, formant_shift, voiced=voiced)

    if float(torch.max(torch.abs(mouth_eff)).detach().cpu()) > 1e-7:
        out_s = _mouth_formant_warp(out_s, mouth_eff, sample_rate, voiced)

    c = out_s.shape[1]
    freq = torch.linspace(0.0, 1.0, c, device=device, dtype=dtype).view(1, c, 1)
    nyquist = max(1.0, float(sample_rate) * 0.5)
    hz = freq * nyquist

    low = torch.exp(-0.5 * torch.square((hz - 520.0) / 620.0))
    mid = torch.exp(-0.5 * torch.square((hz - 1850.0) / 1200.0))
    upper = torch.exp(-0.5 * torch.square((hz - 4800.0) / 2500.0))
    tension_shape = -0.34 * low + 0.28 * mid + 0.42 * upper
    tension_gain = torch.exp(0.86 * tension_eff * tension_shape * voiced)

    gender_shape = (
        0.22 * torch.exp(-0.5 * torch.square((hz - 900.0) / 950.0))
        - 0.12 * torch.exp(-0.5 * torch.square((hz - 3150.0) / 1900.0))
    )
    gender_gain = torch.exp(0.22 * gender_eff * gender_shape * voiced)

    voice_body = (
        0.72 * torch.exp(-0.5 * torch.square((hz - 700.0) / 860.0))
        + 0.48 * torch.exp(-0.5 * torch.square((hz - 1900.0) / 1350.0))
    )
    voice_presence = torch.exp(-0.5 * torch.square((hz - 3400.0) / 1750.0))
    voice_air = torch.exp(-0.5 * torch.square((hz - 6500.0) / 2600.0))
    voicing_shape = voice_body + 0.22 * voice_presence - 0.26 * voice_air
    voicing_gain = torch.exp(
        (0.95 * voicing_pos_eff - 0.38 * voicing_neg_eff) * voicing_shape * voiced
    )

    harmonic_order = hz / f0_env
    source_tilt = torch.log2(torch.clamp(harmonic_order, min=1.0)).clamp(0.0, 4.2)
    h1 = torch.exp(-0.5 * torch.square((harmonic_order - 1.15) / 0.58))
    h2 = torch.exp(-0.5 * torch.square((harmonic_order - 2.0) / 0.72))
    register_shape = 0.34 * h1 + 0.10 * h2 - 0.31 * source_tilt
    falsetto_gain = torch.exp(1.08 * falsetto_eff * register_shape * voiced)

    mixed_shape = (
        0.30 * torch.exp(-0.5 * torch.square((hz - 900.0) / 850.0))
        + 0.90 * torch.exp(-0.5 * torch.square((hz - 2600.0) / 1500.0))
        + 0.34 * torch.exp(-0.5 * torch.square((hz - 5200.0) / 2500.0))
    )
    pharyngeal_shape = (
        -0.34 * torch.exp(-0.5 * torch.square((hz - 620.0) / 620.0))
        + 1.05 * torch.exp(-0.5 * torch.square((hz - 1450.0) / 650.0))
        + 0.52 * torch.exp(-0.5 * torch.square((hz - 2600.0) / 820.0))
        - 0.20 * torch.exp(-0.5 * torch.square((hz - 5200.0) / 2300.0))
    )
    technique_gain = torch.exp((1.02 * mixed_eff * mixed_shape
                                + 1.12 * pharyngeal_eff * pharyngeal_shape) * voiced)
    out_s = (out_s * tension_gain * gender_gain * voicing_gain * falsetto_gain * technique_gain).clamp(min=1e-7)

    ap_frames = ap_bands.shape[-1]
    b_ap = _interp_curve(breathiness, ap_frames, ap_bands.device, ap_bands.dtype)
    t_ap = _interp_curve(tension_eff, ap_frames, ap_bands.device, ap_bands.dtype)
    v_ap = _interp_curve(voicing, ap_frames, ap_bands.device, ap_bands.dtype)
    voiced_ap = (f0 > 1.0).to(ap_bands.dtype)
    if voiced_ap.shape[-1] != ap_frames:
        voiced_ap = F.interpolate(voiced_ap, size=ap_frames, mode="nearest")
    ap_freq = torch.linspace(0.0, 1.0, ap_bands.shape[1], device=ap_bands.device, dtype=ap_bands.dtype).view(1, -1, 1)
    out_ap = ap_bands

    b_pos = torch.clamp(b_ap, 0.0, 1.0) * breathiness_scale
    b_neg = torch.clamp(-b_ap, 0.0, 1.0)
    if float(torch.max(b_pos + b_neg).detach().cpu()) > 1e-7:
        ap_shape = (0.20 + 0.80 * torch.pow(ap_freq, 0.72)) * voiced_ap
        out_ap = out_ap + b_pos * 0.58 * ap_shape * (1.0 - out_ap)
        out_ap = out_ap - b_neg * 0.46 * ap_shape * out_ap
        out_ap = out_ap.clamp(0.012, 0.988)

    t_pos = torch.clamp(t_ap, 0.0, 1.0) * voiced_ap
    t_neg = torch.clamp(-t_ap, 0.0, 1.0) * voiced_ap
    if float(torch.max(t_pos + t_neg).detach().cpu()) > 1e-7:
        tension_ap_shape = 0.94 - 0.22 * ap_freq
        out_ap = out_ap - 0.10 * t_pos * tension_ap_shape * out_ap
        out_ap = out_ap + 0.14 * t_neg * tension_ap_shape * (1.0 - out_ap)
        out_ap = out_ap.clamp(0.012, 0.988)

    v_pos_ap = torch.pow(torch.clamp(v_ap, 0.0, 1.0), 0.68) * voicing_pos_scale * voiced_ap
    v_neg_ap = torch.pow(torch.clamp(-v_ap, 0.0, 1.0), 0.72) * voicing_neg_scale * voiced_ap
    if float(torch.max(v_pos_ap + v_neg_ap).detach().cpu()) > 1e-7:
        periodic_shape = 0.30 + 0.70 * torch.pow(ap_freq, 0.66)
        out_ap = out_ap - 0.72 * v_pos_ap * periodic_shape * out_ap
        out_ap = out_ap + 0.44 * v_neg_ap * periodic_shape * (1.0 - out_ap)
        out_ap = out_ap.clamp(0.012, 0.988)

    f_ap = _interp_curve(falsetto_noise_eff, ap_frames, ap_bands.device, ap_bands.dtype)
    x_ap = _interp_curve(mixed_eff, ap_frames, ap_bands.device, ap_bands.dtype)
    if float(torch.max(f_ap + x_ap).detach().cpu()) > 1e-7:
        air_shape = (0.30 + 0.70 * torch.pow(ap_freq, 0.78)) * voiced_ap
        out_ap = out_ap + f_ap * 0.18 * air_shape * (1.0 - out_ap)
        out_ap = out_ap - x_ap * 0.32 * air_shape * out_ap
        out_ap = out_ap.clamp(0.012, 0.988)

    gate_frames = gate.shape[-1]
    b_g = _interp_curve(breathiness, gate_frames, gate.device, gate.dtype)
    v_g = _interp_curve(voicing, gate_frames, gate.device, gate.dtype)
    t_g = _interp_curve(tension_eff, gate_frames, gate.device, gate.dtype)
    voiced_g = (f0 > 1.0).to(gate.dtype)
    if voiced_g.shape[-1] != gate_frames:
        voiced_g = F.interpolate(voiced_g, size=gate_frames, mode="nearest")
    out_gate = gate

    v_pos = torch.pow(torch.clamp(v_g, 0.0, 1.0), 0.68) * voiced_g * voicing_pos_scale
    v_neg = torch.pow(torch.clamp(-v_g, 0.0, 1.0), 0.72) * voiced_g * voicing_neg_scale
    if float(torch.max(v_pos + v_neg).detach().cpu()) > 1e-7:
        out_gate = out_gate + 1.10 * v_pos * (1.0 - out_gate)
        out_gate = out_gate - 0.58 * v_neg * out_gate
        out_gate = out_gate.clamp(0.02, 0.98)

    t_pos_g = torch.clamp(t_g, 0.0, 1.0) * voiced_g
    t_neg_g = torch.clamp(-t_g, 0.0, 1.0) * voiced_g
    if float(torch.max(t_pos_g + t_neg_g).detach().cpu()) > 1e-7:
        out_gate = out_gate + 0.05 * t_pos_g * (1.0 - out_gate)
        out_gate = out_gate - 0.08 * t_neg_g * out_gate
        out_gate = out_gate.clamp(0.02, 0.98)

    b_pos_g = torch.clamp(b_g, 0.0, 1.0) * voiced_g * breathiness_scale
    b_neg_g = torch.clamp(-b_g, 0.0, 1.0) * voiced_g
    if float(torch.max(b_pos_g + b_neg_g).detach().cpu()) > 1e-7:
        out_gate = out_gate - 0.18 * b_pos_g * out_gate
        out_gate = out_gate + 0.10 * b_neg_g * (1.0 - out_gate)
        out_gate = out_gate.clamp(0.02, 0.98)

    f_g = _interp_curve(falsetto_noise_eff, gate_frames, gate.device, gate.dtype) * voiced_g
    x_g = _interp_curve(mixed_eff, gate_frames, gate.device, gate.dtype) * voiced_g
    if float(torch.max(f_g + x_g).detach().cpu()) > 1e-7:
        out_gate = out_gate - 0.14 * f_g * out_gate
        out_gate = out_gate + 0.48 * x_g * (1.0 - out_gate)
        out_gate = out_gate.clamp(0.02, 0.98)
    return out_s, out_ap, out_gate
