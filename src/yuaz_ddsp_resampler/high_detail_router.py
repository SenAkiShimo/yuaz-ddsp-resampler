#!/usr/bin/env python3
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


HIGH_DETAIL_ROUTER_FORMAT = 3


class RouterBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        d = int(dilation)
        self.dw = nn.Conv1d(
            channels, channels, kernel_size=5, padding=2 * d,
            dilation=d, groups=channels,
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(4, channels)

    def forward(self, x):
        y = self.dw(x)
        y = self.pw(F.silu(y))
        y = self.norm(y)
        return x + 0.30 * F.silu(y)


class HighDetailRouter(nn.Module):
    def __init__(self, sample_rate=44100, hidden=32, frame_hop=128):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.hidden = int(hidden)
        self.frame_hop = int(frame_hop)
        self.input_proj = nn.Conv1d(9, self.hidden, kernel_size=5, padding=2)
        self.blocks = nn.ModuleList([
            RouterBlock(self.hidden, 1),
            RouterBlock(self.hidden, 2),
            RouterBlock(self.hidden, 4),
            RouterBlock(self.hidden, 8),
        ])
        self.output = nn.Conv1d(self.hidden, 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.output.bias[0] = 0.18
            self.output.bias[1] = -4.0

    def _frame_envelope(self, x):
        hop = max(16, int(self.frame_hop))
        kernel = hop * 2
        pad = kernel // 2
        env = F.avg_pool1d(
            F.pad(x.abs(), (pad, pad), mode="replicate"),
            kernel_size=kernel,
            stride=hop,
        )
        return torch.log1p(28.0 * env)

    @staticmethod
    def _frame_flux(env):
        flux = torch.zeros_like(env)
        if env.shape[-1] > 1:
            flux[..., 1:] = torch.relu(env[..., 1:] - env[..., :-1])
        return torch.clamp(flux * 3.0, 0.0, 2.0)

    @staticmethod
    def _prepare_f0(f0, frames):
        if f0 is None:
            raise ValueError("HighDetailRouter requires F0 conditioning")
        if f0.dim() == 1:
            f0 = f0.view(1, 1, -1)
        elif f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        f0 = F.interpolate(f0, size=int(frames), mode="linear", align_corners=False)
        voiced = (f0 > 1.0).to(f0.dtype)
        logf0 = torch.where(
            voiced > 0.5,
            torch.log2(torch.clamp(f0, min=40.0) / 220.0) / 3.0,
            torch.zeros_like(f0),
        )
        return torch.clamp(logf0, -1.25, 1.25), voiced

    def _highpass_fft(self, waveform, low_hz=7600.0, full_hz=8400.0, top_hz=20000.0):
        n = waveform.shape[-1]
        if n < 32:
            return torch.zeros_like(waveform)
        spec = torch.fft.rfft(waveform, dim=-1)
        freqs = torch.linspace(
            0.0,
            self.sample_rate * 0.5,
            spec.shape[-1],
            device=waveform.device,
            dtype=waveform.dtype,
        )
        mask = torch.zeros_like(freqs)
        rise = (freqs >= low_hz) & (freqs < full_hz)
        if bool(rise.any()):
            u = (freqs[rise] - low_hz) / max(1.0, full_hz - low_hz)
            mask[rise] = 0.5 - 0.5 * torch.cos(torch.pi * torch.clamp(u, 0.0, 1.0))
        hi = min(float(top_hz), self.sample_rate * 0.5 - 50.0)
        mask[(freqs >= full_hz) & (freqs <= hi)] = 1.0
        fall_end = min(self.sample_rate * 0.5, hi + 1200.0)
        fall = (freqs > hi) & (freqs < fall_end)
        if bool(fall.any()):
            u = (freqs[fall] - hi) / max(1.0, fall_end - hi)
            mask[fall] = 0.5 + 0.5 * torch.cos(torch.pi * torch.clamp(u, 0.0, 1.0))
        return torch.fft.irfft(spec * mask.view(1, 1, -1), n=n, dim=-1)

    def forward(self, base, source_detail, source_f0, target_f0):
        if base.dim() == 2:
            base = base.unsqueeze(1)
        if source_detail.dim() == 2:
            source_detail = source_detail.unsqueeze(1)
        n = min(base.shape[-1], source_detail.shape[-1])
        base = base[..., :n]
        source_detail = source_detail[..., :n]

        base_high = self._highpass_fft(base)
        src_env = self._frame_envelope(source_detail)
        base_env = self._frame_envelope(base_high)
        frames = min(src_env.shape[-1], base_env.shape[-1])
        src_env = src_env[..., :frames]
        base_env = base_env[..., :frames]
        src_flux = self._frame_flux(src_env)
        base_flux = self._frame_flux(base_env)
        src_logf0, src_voiced = self._prepare_f0(source_f0, frames)
        tgt_logf0, tgt_voiced = self._prepare_f0(target_f0, frames)
        ratio = torch.clamp(tgt_logf0 - src_logf0, -1.25, 1.25)

        features = torch.cat([
            src_env,
            base_env,
            src_flux,
            base_flux,
            src_logf0,
            tgt_logf0,
            ratio,
            src_voiced,
            tgt_voiced,
        ], dim=1)
        h = F.silu(self.input_proj(features))
        for block in self.blocks:
            h = block(h)
        controls = self.output(h)

        # v3 learns only how much source high-detail to route. There is no learned
        # generated-highband suppression path, so muting Yuaz cannot minimize loss.
        inject_frames = 1.35 * torch.sigmoid(controls[:, 0:1, :])
        inject = F.interpolate(inject_frames, size=n, mode="linear", align_corners=False)
        suppress_frames = torch.zeros_like(inject_frames)

        residual = inject * source_detail
        base_rms = torch.sqrt(torch.mean(base.pow(2), dim=-1, keepdim=True) + 1e-8)
        residual_rms = torch.sqrt(torch.mean(residual.pow(2), dim=-1, keepdim=True) + 1e-8)
        limit = 0.30 * base_rms + 1e-7
        scale = torch.clamp(limit / (residual_rms + 1e-8), max=1.0)
        residual = residual * scale
        refined = torch.clamp(base + residual, -1.2, 1.2)

        # The legacy trainer adds generic control/residual regularizers. v3's
        # teacher already constrains detail energy, so return detached monitoring
        # tensors to prevent those old regularizers from recreating a zero-detail
        # shortcut while preserving their reported values.
        return (
            refined,
            residual.detach(),
            inject_frames.detach(),
            suppress_frames.detach(),
            base_high,
        )

    def summary(self):
        return {
            "format": HIGH_DETAIL_ROUTER_FORMAT,
            "sample_rate": self.sample_rate,
            "hidden": self.hidden,
            "frame_hop": self.frame_hop,
            "output_weight_rms": float(self.output.weight.detach().pow(2).mean().sqrt().cpu()),
        }


def save_high_detail_router(path, model, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": HIGH_DETAIL_ROUTER_FORMAT,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metadata": metadata or {},
    }, path)


def load_high_detail_router(path, device="cpu"):
    path = Path(path)
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError(f"Unsupported high-detail router file: {path}")
    meta = dict(payload.get("metadata") or {})
    model = HighDetailRouter(
        sample_rate=int(meta.get("sample_rate", 44100)),
        hidden=int(meta.get("hidden", 32)),
        frame_hop=int(meta.get("frame_hop", 128)),
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=False)
    model.eval()
    return model, meta
