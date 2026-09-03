from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_UPSAMPLE_RATES = (8, 8, 4, 2)
DEFAULT_CHANNELS = (192, 160, 128, 96, 64)
DEFAULT_SPECTRAL_BANDS = 48
DEFAULT_AP_BANDS = 16


def _match_length(x, length):
    length = int(length)
    if x.shape[-1] == length:
        return x
    if x.shape[-1] > length:
        return x[..., :length]
    return F.pad(x, (0, length - x.shape[-1]))


def _resize_frequency(x, bands):
    if x.ndim != 3:
        raise ValueError("expected [B, C, T]")
    if x.shape[1] == int(bands):
        return x
    return F.interpolate(
        x.unsqueeze(1),
        size=(int(bands), x.shape[-1]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def _frame_length(x, frames):
    if x.shape[-1] == int(frames):
        return x
    return F.interpolate(x, size=int(frames), mode="linear", align_corners=False)


def build_neural_conditioning(
    latent,
    detail,
    f0,
    aux,
    spectral_bands=DEFAULT_SPECTRAL_BANDS,
    ap_bands=DEFAULT_AP_BANDS,
):
    """Build frame-rate conditioning without exposing raw voiced source waveform."""
    if f0.ndim == 2:
        f0 = f0.unsqueeze(1)
    frames = int(f0.shape[-1])
    latent = _frame_length(latent, frames)
    detail = _frame_length(detail, frames)

    voiced = (f0 > 1.0).to(f0.dtype)
    log_f0 = torch.zeros_like(f0)
    safe = torch.clamp(f0, min=1.0)
    log_f0 = torch.where(voiced > 0.5, torch.log2(safe / 440.0), log_f0)
    log_f0 = torch.clamp(log_f0, -5.0, 3.0) / 5.0

    spectral = aux.get("spectral_envelope")
    ap = aux.get("ap_after_soft_mvf")
    gate = aux.get("gate")
    if spectral is None or ap is None or gate is None:
        raise RuntimeError("DDSP auxiliary state is incomplete for neural waveform conditioning")

    spectral = _frame_length(spectral, frames)
    spectral = _resize_frequency(spectral, int(spectral_bands))
    spectral = torch.log1p(torch.clamp(spectral, min=0.0))
    spectral = spectral - spectral.mean(dim=1, keepdim=True)
    spectral = spectral / (spectral.std(dim=1, keepdim=True) + 1e-4)
    spectral = torch.clamp(spectral, -4.0, 4.0)

    ap = _frame_length(ap, frames)
    ap = _resize_frequency(ap, int(ap_bands))
    ap = torch.clamp(ap, 0.0, 1.0) * 2.0 - 1.0
    gate = _frame_length(gate, frames)

    return torch.cat([
        latent,
        detail,
        log_f0,
        voiced,
        spectral,
        ap,
        gate,
    ], dim=1)


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels, dilations=(1, 3, 9)):
        super().__init__()
        self.layers = nn.ModuleList()
        for dilation in dilations:
            self.layers.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, 3, padding=int(dilation), dilation=int(dilation)),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, 1),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class YuazNeuralWaveformDecoder(nn.Module):
    """Small DDSP-conditioned direct waveform generator for the v0.3 branch."""

    def __init__(
        self,
        condition_channels,
        upsample_rates=DEFAULT_UPSAMPLE_RATES,
        channels=DEFAULT_CHANNELS,
        residual_dilations=(1, 3, 9),
    ):
        super().__init__()
        rates = tuple(int(x) for x in upsample_rates)
        widths = tuple(int(x) for x in channels)
        if len(widths) != len(rates) + 1:
            raise ValueError("channels must contain one more entry than upsample_rates")
        if any(x <= 0 for x in rates) or any(x <= 0 for x in widths):
            raise ValueError("upsample rates and channels must be positive")

        self.condition_channels = int(condition_channels)
        self.upsample_rates = rates
        self.channels = widths
        hop = 1
        for rate in rates:
            hop *= rate
        self.output_hop = int(hop)

        self.pre = nn.Conv1d(self.condition_channels, widths[0], 1)
        self.stages = nn.ModuleList()
        self.structure_projections = nn.ModuleList()
        for rate, in_ch, out_ch in zip(rates, widths[:-1], widths[1:]):
            self.stages.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(in_ch, out_ch, 7, padding=3),
                ResidualDilatedBlock(out_ch, residual_dilations),
            ))
            self.structure_projections.append(nn.Conv1d(1, out_ch, 7, padding=3))

        mid = max(24, widths[-1] // 2)
        self.post = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(widths[-1], mid, 7, padding=3),
            nn.LeakyReLU(0.1),
            nn.Conv1d(mid, 1, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, conditioning, structure):
        if conditioning.ndim != 3:
            raise ValueError("conditioning must be [B, C, T]")
        if structure.ndim == 2:
            structure = structure.unsqueeze(1)
        if structure.ndim != 3 or structure.shape[1] != 1:
            raise ValueError("structure must be [B, 1, samples]")
        if conditioning.shape[0] != structure.shape[0]:
            raise ValueError("conditioning and structure batch sizes must match")
        if conditioning.shape[1] != self.condition_channels:
            raise ValueError(
                f"expected {self.condition_channels} conditioning channels, got {conditioning.shape[1]}"
            )

        target_samples = int(conditioning.shape[-1]) * self.output_hop
        structure = _match_length(structure, target_samples)
        x = self.pre(conditioning)
        for rate, stage, source_proj in zip(self.upsample_rates, self.stages, self.structure_projections):
            x = F.interpolate(x, scale_factor=int(rate), mode="linear", align_corners=False)
            x = stage(x)
            source = F.interpolate(structure, size=x.shape[-1], mode="linear", align_corners=False)
            x = x + source_proj(source)
        return _match_length(self.post(x), target_samples)


def save_neural_waveform_decoder(path, model, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "yuaz-neural-waveform-v1",
        "condition_channels": int(model.condition_channels),
        "upsample_rates": list(model.upsample_rates),
        "channels": list(model.channels),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }, path)


def load_neural_waveform_decoder(path, device="cpu"):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != "yuaz-neural-waveform-v1":
        raise RuntimeError("unsupported neural waveform decoder format")
    model = YuazNeuralWaveformDecoder(
        condition_channels=int(payload["condition_channels"]),
        upsample_rates=tuple(payload.get("upsample_rates") or DEFAULT_UPSAMPLE_RATES),
        channels=tuple(payload.get("channels") or DEFAULT_CHANNELS),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().to(device)
    return model, dict(payload.get("metadata") or {})
