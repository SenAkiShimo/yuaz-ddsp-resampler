#!/usr/bin/env python3
import torch
import torch.nn.functional as F

from . import train_high_detail_router_entry as entry

base = entry.base


def _target_nonperiodic_mask(freqs, target_f0, frames):
    tf0 = F.interpolate(target_f0, size=int(frames), mode="linear", align_corners=False)
    voiced = (tf0 > 1.0).to(tf0.dtype)
    safe_f0 = torch.clamp(tf0, min=55.0)
    freq_grid = freqs.view(1, -1, 1)
    order = torch.round(freq_grid / safe_f0)
    distance = torch.abs(freq_grid - order * safe_f0) / safe_f0
    harmonic = torch.exp(-0.5 * torch.pow(distance / 0.16, 2.0))
    harmonic = harmonic * (order >= 1.0).to(harmonic.dtype) * voiced
    # In voiced regions ignore the harmonic cores; in unvoiced regions supervise
    # the whole high band because that is exactly where articulation noise lives.
    return torch.clamp(1.0 - 0.92 * harmonic, 0.08, 1.0)


def highband_loss_v2(pred, target, source_f0, target_f0, sr):
    p, freqs = base._stft_mag(pred, sr)
    t, _ = base._stft_mag(target, sr)
    hi = min(20000.0, sr * 0.5 - 100.0)
    band = (freqs >= 7200.0) & (freqs <= hi)
    if not bool(band.any()):
        return F.smooth_l1_loss(pred, target), {}

    frames = p.shape[-1]
    nonperiodic = _target_nonperiodic_mask(freqs, target_f0, frames)
    band_weight = band.view(1, -1, 1).to(p.dtype)
    weight = band_weight * nonperiodic
    den = torch.sum(weight) + 1e-8

    # Missing real target detail is substantially worse than extra energy. This
    # prevents the old shortcut where the router reduced loss by muting highband.
    delta = p - t
    under = torch.relu(-delta)
    over = torch.relu(delta)
    spectral = torch.sum(weight * (2.40 * under.pow(2) + 0.75 * over.pow(2))) / den

    # Explicit target-presence floor: bins/frames with meaningful target detail
    # should not be allowed to collapse toward silence.
    target_level = t.detach()
    active_threshold = torch.quantile(target_level[:, band, :].reshape(-1), 0.62)
    active = (target_level >= active_threshold).to(p.dtype) * weight
    active_den = torch.sum(active) + 1e-8
    presence = torch.sum(active * torch.relu(target_level - p - 0.025).pow(2)) / active_den

    pb = p[:, band, :]
    tb = t[:, band, :]
    wb = nonperiodic[:, band, :]
    if pb.shape[-1] > 1:
        pflux = torch.relu(torch.diff(pb, dim=-1))
        tflux = torch.relu(torch.diff(tb, dim=-1))
        fw = torch.minimum(wb[..., 1:], wb[..., :-1])
        flux = torch.sum(fw * torch.abs(pflux - tflux)) / (torch.sum(fw) + 1e-8)
    else:
        flux = pred.new_tensor(0.0)

    if pb.shape[1] > 2:
        ps = torch.diff(pb, dim=1)
        ts = torch.diff(tb, dim=1)
        sw = torch.minimum(wb[:, 1:, :], wb[:, :-1, :])
        shape = torch.sum(sw * torch.abs(ps - ts)) / (torch.sum(sw) + 1e-8)
    else:
        shape = pred.new_tensor(0.0)

    band_losses = []
    for lo, band_hi in ((7200.0, 10000.0), (10000.0, 14000.0), (14000.0, hi)):
        m = (freqs >= lo) & (freqs < band_hi)
        if not bool(m.any()):
            continue
        w = nonperiodic[:, m, :]
        pred_energy = torch.sum(w * p[:, m, :], dim=1) / (torch.sum(w, dim=1) + 1e-8)
        target_energy = torch.sum(w * t[:, m, :], dim=1) / (torch.sum(w, dim=1) + 1e-8)
        # Again make underfill more expensive than overfill.
        d = pred_energy - target_energy
        band_losses.append((2.0 * torch.relu(-d).pow(2) + 0.65 * torch.relu(d).pow(2)).mean())
    bands = torch.stack(band_losses).mean() if band_losses else pred.new_tensor(0.0)

    # Source-F0 leakage stays explicitly penalized when the leaked ridge does not
    # coincide with a target-F0 harmonic.
    sf0 = F.interpolate(source_f0, size=frames, mode="linear", align_corners=False).clamp_min(1.0)
    tf0 = F.interpolate(target_f0, size=frames, mode="linear", align_corners=False).clamp_min(1.0)
    freq_grid = freqs.view(1, -1, 1)
    source_order = torch.round(freq_grid / sf0)
    target_order = torch.round(freq_grid / tf0)
    source_distance = torch.abs(freq_grid - source_order * sf0) / sf0
    target_distance = torch.abs(freq_grid - target_order * tf0) / tf0
    source_harm = torch.exp(-0.5 * torch.pow(source_distance / 0.13, 2.0))
    target_harm = torch.exp(-0.5 * torch.pow(target_distance / 0.13, 2.0))
    leakage_region = (source_harm > 0.45) & (target_harm < 0.20) & band.view(1, -1, 1)
    if bool(leakage_region.any()):
        excess = torch.relu((p - t) - 0.05)
        leak = excess[leakage_region.expand_as(excess)].pow(2).mean()
    else:
        leak = pred.new_tensor(0.0)

    loss = spectral + 0.85 * presence + 0.26 * flux + 0.16 * shape + 0.24 * bands + 0.65 * leak
    return loss, {
        "spectral": float(spectral.detach()),
        "presence": float(presence.detach()),
        "flux": float(flux.detach()),
        "shape": float(shape.detach()),
        "bands": float(bands.detach()),
        "source_f0_leak": float(leak.detach()),
    }


base.highband_loss = highband_loss_v2
base.TRAINING_FORMAT = 2


if __name__ == "__main__":
    base.main()
