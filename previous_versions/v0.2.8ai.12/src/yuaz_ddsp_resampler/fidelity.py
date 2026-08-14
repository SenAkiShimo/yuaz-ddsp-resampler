#!/usr/bin/env python3
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


REFINER_FORMAT = 2
DEFAULT_DETAIL_DIM = 25
FIDELITY_HARD_LIMIT = 0.085


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        pad = 2 * int(dilation)
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=5, padding=pad, dilation=int(dilation), groups=channels
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(4, channels)
        nn.init.kaiming_uniform_(self.depthwise.weight, a=0.2)
        nn.init.zeros_(self.depthwise.bias)
        nn.init.kaiming_uniform_(self.pointwise.weight, a=0.2)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x):
        y = self.depthwise(x)
        y = self.pointwise(F.silu(y))
        y = self.norm(y)
        return x + 0.35 * F.silu(y)


class TinyFidelityRefiner(nn.Module):
    def __init__(self, detail_dim=DEFAULT_DETAIL_DIM, hidden=32):
        super().__init__()
        self.detail_dim = int(detail_dim)
        self.hidden = int(hidden)
        self.detail_proj = nn.Conv1d(self.detail_dim, 10, kernel_size=1)
        self.input_proj = nn.Conv1d(12, self.hidden, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList([
            ResidualBlock(self.hidden, 1),
            ResidualBlock(self.hidden, 2),
            ResidualBlock(self.hidden, 4),
        ])
        self.output = nn.Conv1d(self.hidden, 1, kernel_size=5, padding=2)
        nn.init.kaiming_uniform_(self.detail_proj.weight, a=0.2)
        nn.init.zeros_(self.detail_proj.bias)
        nn.init.kaiming_uniform_(self.input_proj.weight, a=0.2)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _prepare_detail(self, detail, samples):
        if detail is None:
            return None
        if detail.dim() == 2:
            detail = detail.unsqueeze(0)
        if detail.shape[1] > self.detail_dim:
            detail = detail[:, : self.detail_dim]
        elif detail.shape[1] < self.detail_dim:
            detail = F.pad(detail, (0, 0, 0, self.detail_dim - detail.shape[1]))
        detail = F.interpolate(detail, size=int(samples), mode="linear", align_corners=False)
        return torch.tanh(self.detail_proj(detail))

    @staticmethod
    def _prepare_f0(f0, samples):
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        f0 = F.interpolate(f0, size=int(samples), mode="linear", align_corners=False)
        voiced = (f0 > 1.0).to(f0.dtype)
        logf0 = torch.where(voiced > 0.5, torch.log2(torch.clamp(f0, min=40.0) / 220.0), torch.zeros_like(f0))
        logf0 = torch.clamp(logf0 / 3.0, -1.0, 1.0)
        return logf0, voiced

    @staticmethod
    def _highpass(x):
        kernel = 17
        pad = kernel // 2
        smooth = F.avg_pool1d(F.pad(x, (pad, pad), mode="replicate"), kernel_size=kernel, stride=1)
        return x - smooth

    def forward(self, waveform, detail, f0, transient_strength=1.0, articulation_end_sample=None):
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        samples = waveform.shape[-1]
        d = self._prepare_detail(detail, samples)
        if d is None:
            d = waveform.new_zeros((waveform.shape[0], 10, samples))
        logf0, voiced = self._prepare_f0(f0, samples)
        x = torch.cat([waveform, d, logf0], dim=1)
        h = F.silu(self.input_proj(x))
        for block in self.blocks:
            h = block(h)
        residual = 0.10 * torch.tanh(self.output(h))
        residual = self._highpass(residual)

        flux = d[:, -1:, :].abs() if d.shape[1] else torch.zeros_like(waveform)
        transient_gate = torch.clamp(0.32 + float(transient_strength) * 0.16 * flux, 0.22, 0.66)
        voiced_gate = 0.52 + 0.48 * voiced
        residual = residual * transient_gate * voiced_gate

        if articulation_end_sample is not None:
            end = int(max(0, min(samples, int(articulation_end_sample))))
            if end > 0:
                gate = waveform.new_ones((1, 1, samples))
                gate[..., :end] = 0.48
                ramp = min(samples - end, max(1, int(round(0.045 * 24000))))
                if ramp > 1:
                    gate[..., end:end + ramp] = torch.linspace(0.48, 1.0, ramp, device=waveform.device, dtype=waveform.dtype)
                residual = residual * gate

        base_rms = torch.sqrt(torch.mean(waveform.pow(2), dim=-1, keepdim=True) + 1e-8)
        res_rms = torch.sqrt(torch.mean(residual.pow(2), dim=-1, keepdim=True) + 1e-8)
        max_rms = FIDELITY_HARD_LIMIT * base_rms + 1e-7
        scale = torch.clamp(max_rms / (res_rms + 1e-8), max=1.0)
        residual = residual * scale
        return torch.clamp(waveform + residual, -1.2, 1.2), residual

    def regularization(self):
        return 0.05 * self.output.weight.pow(2).mean()

    def summary(self):
        with torch.no_grad():
            return {
                "format": REFINER_FORMAT,
                "detail_dim": self.detail_dim,
                "hidden": self.hidden,
                "output_weight_rms": float(self.output.weight.pow(2).mean().sqrt()),
            }


def save_refiner(path, refiner, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": REFINER_FORMAT,
            "state_dict": {k: v.detach().cpu() for k, v in refiner.state_dict().items()},
            "metadata": metadata or {},
        },
        path,
    )


def load_refiner(path, device="cpu"):
    path = Path(path)
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError(f"Unsupported fidelity refiner file: {path}")
    state = payload["state_dict"]
    detail_dim = int(state.get("detail_proj.weight", torch.empty(10, DEFAULT_DETAIL_DIM, 1)).shape[1])
    hidden = int(state.get("input_proj.weight", torch.empty(32, 12, 7)).shape[0])
    refiner = TinyFidelityRefiner(detail_dim=detail_dim, hidden=hidden).to(device)
    refiner.load_state_dict(state, strict=False)
    refiner.eval()
    metadata = payload.get("metadata", {})
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    metadata["loaded_refiner_format"] = int(payload.get("format", 0))
    return refiner, metadata
