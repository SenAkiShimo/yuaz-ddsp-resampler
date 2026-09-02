#!/usr/bin/env python3
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


HIGH_DETAIL_TF_FORMAT = 1


class TFBlock(nn.Module):
    def __init__(self, channels, dilation_t=1):
        super().__init__()
        d = int(dilation_t)
        self.conv1 = nn.Conv2d(channels, channels, (3, 5), padding=(1, 2 * d), dilation=(1, d))
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.GroupNorm(4, channels)

    def forward(self, x):
        y = F.silu(self.conv1(x))
        y = self.norm(self.conv2(y))
        return x + 0.28 * F.silu(y)


class HighDetailTFGenerator(nn.Module):
    def __init__(self, sample_rate=44100, n_fft=1024, hop=128, hidden=24):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.hidden = int(hidden)
        self.input = nn.Conv2d(7, hidden, 3, padding=1)
        self.blocks = nn.ModuleList([
            TFBlock(hidden, 1),
            TFBlock(hidden, 2),
            TFBlock(hidden, 4),
            TFBlock(hidden, 8),
            TFBlock(hidden, 1),
        ])
        self.output = nn.Conv2d(hidden, 1, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, -0.25)

    def _window(self, x):
        return torch.hann_window(self.n_fft, device=x.device, dtype=x.dtype)

    def _stft(self, x):
        return torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.n_fft,
            window=self._window(x),
            return_complex=True,
        )

    def _istft(self, spec, length, ref):
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.n_fft,
            window=self._window(ref),
            length=int(length),
        ).unsqueeze(1)

    @staticmethod
    def _f0_frames(f0, frames):
        if f0.dim() == 1:
            f0 = f0.view(1, 1, -1)
        elif f0.dim() == 2:
            f0 = f0.unsqueeze(1)
        return F.interpolate(f0, size=int(frames), mode="linear", align_corners=False)

    def _harmonic_map(self, freqs, f0, frames):
        track = self._f0_frames(f0, frames)
        voiced = (track > 1.0).to(track.dtype)
        safe = torch.clamp(track, min=55.0)
        grid = freqs.view(1, -1, 1)
        order = torch.round(grid / safe)
        distance = torch.abs(grid - order * safe) / safe
        harm = torch.exp(-0.5 * torch.pow(distance / 0.14, 2.0))
        return harm * (order >= 1.0).to(harm.dtype) * voiced

    @staticmethod
    def _norm_feature(x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True).clamp_min(0.12)
        return torch.clamp((x - mean) / std, -4.0, 4.0)

    def forward(self, base, source_detail, source_f0, target_f0):
        if base.dim() == 2:
            base = base.unsqueeze(1)
        if source_detail.dim() == 2:
            source_detail = source_detail.unsqueeze(1)
        n = min(base.shape[-1], source_detail.shape[-1])
        base = base[..., :n]
        source_detail = source_detail[..., :n]

        src_spec = self._stft(source_detail)
        base_spec = self._stft(base)
        frames = min(src_spec.shape[-1], base_spec.shape[-1])
        src_spec = src_spec[..., :frames]
        base_spec = base_spec[..., :frames]
        freqs = torch.linspace(
            0.0,
            self.sample_rate * 0.5,
            src_spec.shape[1],
            device=base.device,
            dtype=base.dtype,
        )

        src_log = torch.log1p(32.0 * src_spec.abs())
        base_log = torch.log1p(32.0 * base_spec.abs())
        src_harm = self._harmonic_map(freqs, source_f0, frames)
        tgt_harm = self._harmonic_map(freqs, target_f0, frames)

        src_f0 = self._f0_frames(source_f0, frames)
        tgt_f0 = self._f0_frames(target_f0, frames)
        src_logf0 = torch.where(src_f0 > 1.0, torch.log2(torch.clamp(src_f0, min=40.0) / 220.0), torch.zeros_like(src_f0))
        tgt_logf0 = torch.where(tgt_f0 > 1.0, torch.log2(torch.clamp(tgt_f0, min=40.0) / 220.0), torch.zeros_like(tgt_f0))
        ratio = torch.clamp((tgt_logf0 - src_logf0) / 2.0, -1.5, 1.5)
        ratio = ratio.expand(-1, src_log.shape[1], -1)

        flux = torch.zeros_like(src_log)
        if frames > 1:
            flux[..., 1:] = torch.relu(src_log[..., 1:] - src_log[..., :-1])

        x = torch.stack([
            self._norm_feature(src_log),
            self._norm_feature(base_log),
            src_harm,
            tgt_harm,
            ratio,
            torch.clamp(flux, 0.0, 2.0),
            torch.clamp(src_harm - tgt_harm, -1.0, 1.0),
        ], dim=1)
        h = F.silu(self.input(x))
        for block in self.blocks:
            h = block(h)
        mask = 1.65 * torch.sigmoid(self.output(h)).squeeze(1)

        high = (freqs >= 7000.0) & (freqs <= min(20500.0, self.sample_rate * 0.5 - 50.0))
        mask = mask * high.view(1, -1, 1).to(mask.dtype)
        residual_spec = src_spec * mask
        residual = self._istft(residual_spec, n, base)

        base_rms = torch.sqrt(torch.mean(base.pow(2), dim=-1, keepdim=True) + 1e-8)
        residual_rms = torch.sqrt(torch.mean(residual.pow(2), dim=-1, keepdim=True) + 1e-8)
        scale = torch.clamp((0.32 * base_rms + 1e-7) / (residual_rms + 1e-8), max=1.0)
        residual = residual * scale
        refined = torch.clamp(base + residual, -1.2, 1.2)
        return refined, residual, mask, src_spec, base_spec

    def summary(self):
        return {
            "format": HIGH_DETAIL_TF_FORMAT,
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop": self.hop,
            "hidden": self.hidden,
        }


def save_high_detail_tf(path, model, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": HIGH_DETAIL_TF_FORMAT,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metadata": metadata or {},
    }, path)


def load_high_detail_tf(path, device="cpu"):
    path = Path(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    meta = dict(payload.get("metadata") or {})
    model = HighDetailTFGenerator(
        sample_rate=int(meta.get("sample_rate", 44100)),
        n_fft=int(meta.get("n_fft", 1024)),
        hop=int(meta.get("hop", 128)),
        hidden=int(meta.get("hidden", 24)),
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, meta
