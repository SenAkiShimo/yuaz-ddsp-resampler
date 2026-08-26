#!/usr/bin/env python3
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


CLARITY_REFINER_FORMAT = 1
DEFAULT_DETAIL_DIM = 25
CLARITY_RESIDUAL_LIMIT = 0.12


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        pad = 2 * int(dilation)
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=5, padding=pad,
            dilation=int(dilation), groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(4, channels)

    def forward(self, x):
        y = self.depthwise(x)
        y = self.pointwise(F.silu(y))
        y = self.norm(y)
        return x + 0.32 * F.silu(y)


class ClarityRefiner(nn.Module):
    def __init__(self, detail_dim=DEFAULT_DETAIL_DIM, hidden=48, sample_rate=24000):
        super().__init__()
        self.detail_dim = int(detail_dim)
        self.hidden = int(hidden)
        self.sample_rate = int(sample_rate)
        self.detail_proj = nn.Conv1d(self.detail_dim, 12, kernel_size=1)
        self.input_proj = nn.Conv1d(16, self.hidden, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList([
            ResidualBlock(self.hidden, 1),
            ResidualBlock(self.hidden, 2),
            ResidualBlock(self.hidden, 4),
            ResidualBlock(self.hidden, 8),
        ])
        self.output = nn.Conv1d(self.hidden, 1, kernel_size=5, padding=2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _prepare_detail(self, detail, samples):
        if detail is None:
            return None, None
        if detail.dim() == 2:
            detail = detail.unsqueeze(0)
        if detail.shape[1] > self.detail_dim:
            detail = detail[:, :self.detail_dim]
        elif detail.shape[1] < self.detail_dim:
            detail = F.pad(detail, (0, 0, 0, self.detail_dim - detail.shape[1]))
        raw_flux = detail[:, -1:, :] if detail.shape[1] else None
        detail = F.interpolate(detail, size=int(samples), mode="linear", align_corners=False)
        detail = torch.tanh(self.detail_proj(detail))
        if raw_flux is not None:
            raw_flux = F.interpolate(raw_flux, size=int(samples), mode="linear", align_corners=False)
        return detail, raw_flux

    @staticmethod
    def _prepare_f0(f0, samples):
        if f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        f0 = F.interpolate(f0, size=int(samples), mode="linear", align_corners=False)
        voiced = (f0 > 1.0).to(f0.dtype)
        logf0 = torch.where(
            voiced > 0.5,
            torch.log2(torch.clamp(f0, min=40.0) / 220.0),
            torch.zeros_like(f0),
        )
        logf0 = torch.clamp(logf0 / 3.0, -1.0, 1.0)
        return logf0, voiced

    @staticmethod
    def _highpass(x):
        kernel = 21
        pad = kernel // 2
        smooth = F.avg_pool1d(
            F.pad(x, (pad, pad), mode="replicate"),
            kernel_size=kernel, stride=1,
        )
        return x - smooth

    def articulation_gate(self, f0, samples, flux=None, external_mask=None):
        if external_mask is not None:
            mask = external_mask
            if mask.dim() == 1:
                mask = mask.view(1, 1, -1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1)
            if mask.shape[-1] != samples:
                mask = F.interpolate(mask, size=int(samples), mode="linear", align_corners=False)
            return torch.clamp(mask, 0.0, 1.0)

        _, voiced = self._prepare_f0(f0, samples)
        gate = torch.zeros_like(voiced)
        pre = int(round(0.120 * self.sample_rate))
        hold = int(round(0.090 * self.sample_rate))
        decay = int(round(0.140 * self.sample_rate))
        for i in range(voiced.shape[0]):
            index = torch.nonzero(voiced[i, 0] > 0.5, as_tuple=False).reshape(-1)
            if index.numel() == 0:
                continue
            onset = int(index[0].item())
            a = max(0, onset - pre)
            b = min(samples, onset + hold)
            c = min(samples, b + decay)
            gate[i, 0, a:b] = 1.0
            if c > b:
                u = torch.linspace(0.0, 1.0, c - b, device=gate.device, dtype=gate.dtype)
                gate[i, 0, b:c] = 0.5 + 0.5 * torch.cos(torch.pi * u)
        if flux is not None:
            flux = torch.clamp(flux / 2.5, 0.0, 1.0)
            gate = torch.clamp(gate * (0.72 + 0.28 * flux), 0.0, 1.0)
        return gate

    def forward(self, waveform, detail, f0, articulation_mask=None):
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        samples = waveform.shape[-1]
        detail_features, flux = self._prepare_detail(detail, samples)
        if detail_features is None:
            detail_features = waveform.new_zeros((waveform.shape[0], 12, samples))
        logf0, voiced = self._prepare_f0(f0, samples)
        wave_hp = self._highpass(waveform)
        x = torch.cat([waveform, wave_hp, detail_features, logf0, voiced], dim=1)
        h = F.silu(self.input_proj(x))
        for block in self.blocks:
            h = block(h)
        residual = 0.12 * torch.tanh(self.output(h))
        residual = self._highpass(residual)
        gate = self.articulation_gate(f0, samples, flux=flux, external_mask=articulation_mask)
        residual = residual * gate

        weight = torch.clamp(gate, min=0.05)
        base_rms = torch.sqrt(
            torch.sum(waveform.pow(2) * weight, dim=-1, keepdim=True)
            / (torch.sum(weight, dim=-1, keepdim=True) + 1e-8)
            + 1e-8
        )
        residual_rms = torch.sqrt(
            torch.sum(residual.pow(2) * weight, dim=-1, keepdim=True)
            / (torch.sum(weight, dim=-1, keepdim=True) + 1e-8)
            + 1e-8
        )
        limit = CLARITY_RESIDUAL_LIMIT * base_rms + 1e-7
        scale = torch.clamp(limit / (residual_rms + 1e-8), max=1.0)
        residual = residual * scale
        refined = torch.clamp(waveform + residual, -1.2, 1.2)
        return refined, residual, gate

    def summary(self):
        with torch.no_grad():
            return {
                "format": CLARITY_REFINER_FORMAT,
                "detail_dim": self.detail_dim,
                "hidden": self.hidden,
                "sample_rate": self.sample_rate,
                "output_weight_rms": float(self.output.weight.pow(2).mean().sqrt()),
            }


def save_clarity_refiner(path, model, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": CLARITY_REFINER_FORMAT,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metadata": metadata or {},
    }, path)


def load_clarity_refiner(path, device="cpu"):
    path = Path(path)
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError(f"Unsupported clarity refiner file: {path}")
    state = payload["state_dict"]
    detail_dim = int(state.get("detail_proj.weight", torch.empty(12, DEFAULT_DETAIL_DIM, 1)).shape[1])
    hidden = int(state.get("input_proj.weight", torch.empty(48, 16, 7)).shape[0])
    metadata = payload.get("metadata") or {}
    sample_rate = int(metadata.get("sample_rate", 24000))
    model = ClarityRefiner(detail_dim=detail_dim, hidden=hidden, sample_rate=sample_rate).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, dict(metadata)
