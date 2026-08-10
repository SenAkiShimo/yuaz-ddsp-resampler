#!/usr/bin/env python3

import math

from pathlib import Path


import torch

import torch.nn as nn

import torch.nn.functional as F


ADAPTER_FORMAT = 5

DEFAULT_DETAIL_DIM = 25

DEFAULT_TIMBRE_DIM = 32


def _hz_to_midi(hz):

    hz = float(hz)

    if hz <= 0:

        return 69.0

    return 69.0 + 12.0 * math.log2(hz / 440.0)


class VoicebankAdapter(nn.Module):

    def __init__(

        self,

        latent_dim=128,

        spectral_bands=64,

        ap_bands=16,

        detail_dim=DEFAULT_DETAIL_DIM,

        timbre_dim=DEFAULT_TIMBRE_DIM,

        pitch_prototype_count=3,

        pitch_prototype_midi=None,

    ):

        super().__init__()

        self.latent_dim = int(latent_dim)

        self.spectral_bands = int(spectral_bands)

        self.ap_bands = int(ap_bands)

        self.detail_dim = int(detail_dim)

        self.timbre_dim = int(timbre_dim)

        pitch_prototype_count = max(1, int(pitch_prototype_count))


        self.latent_scale_raw = nn.Parameter(torch.zeros(self.latent_dim))

        self.latent_bias_raw = nn.Parameter(torch.zeros(self.latent_dim))

        self.spectral_log_gain_raw = nn.Parameter(torch.zeros(self.spectral_bands))

        self.ap_bias_raw = nn.Parameter(torch.zeros(self.ap_bands))

        self.gate_bias_raw = nn.Parameter(torch.zeros(1))


        detail_hidden = 48

        self.detail_pre = nn.Conv1d(self.detail_dim, detail_hidden, 1)

        self.detail_to_latent = nn.Conv1d(detail_hidden, self.latent_dim, 1)

        self.detail_to_spectral = nn.Conv1d(detail_hidden, self.spectral_bands, 1)

        self.detail_to_ap = nn.Conv1d(detail_hidden, self.ap_bands, 1)

        nn.init.kaiming_uniform_(self.detail_pre.weight, a=0.2)

        nn.init.zeros_(self.detail_pre.bias)

        for layer in (self.detail_to_latent, self.detail_to_spectral, self.detail_to_ap):

            nn.init.zeros_(layer.weight)

            nn.init.zeros_(layer.bias)


        scrub_hidden = 64

        self.content_pre = nn.Conv1d(self.latent_dim, scrub_hidden, 1)

        self.content_delta = nn.Conv1d(scrub_hidden, self.latent_dim, 1)

        nn.init.kaiming_uniform_(self.content_pre.weight, a=0.2)

        nn.init.zeros_(self.content_pre.bias)

        nn.init.zeros_(self.content_delta.weight)

        nn.init.zeros_(self.content_delta.bias)


        self.timbre_code = nn.Parameter(torch.empty(self.timbre_dim))

        self.pitch_timbre_codes = nn.Parameter(torch.zeros(pitch_prototype_count, self.timbre_dim))

        if pitch_prototype_midi is None:

            if pitch_prototype_count == 1:

                pitch_prototype_midi = [57.0]

            else:

                pitch_prototype_midi = torch.linspace(48.0, 66.0, pitch_prototype_count).tolist()

        self.register_buffer("pitch_prototype_midi", torch.as_tensor(pitch_prototype_midi, dtype=torch.float32).reshape(-1))

        self.register_buffer("bank_median_f0_hz", torch.tensor(220.0, dtype=torch.float32))

        nn.init.normal_(self.timbre_code, mean=0.0, std=0.02)


        self.timbre_to_latent = nn.Linear(self.timbre_dim, self.latent_dim)

        self.timbre_to_spectral = nn.Linear(self.timbre_dim, self.spectral_bands)

        self.timbre_to_ap = nn.Linear(self.timbre_dim, self.ap_bands)

        self.timbre_to_gate = nn.Linear(self.timbre_dim, 1)

        for layer in (self.timbre_to_latent, self.timbre_to_spectral, self.timbre_to_ap, self.timbre_to_gate):

            nn.init.zeros_(layer.weight)

            nn.init.zeros_(layer.bias)


    @property

    def pitch_prototype_count(self):

        return int(self.pitch_timbre_codes.shape[0])


    def set_bank_median_f0(self, hz):

        hz = float(hz)

        if hz > 20.0:

            self.bank_median_f0_hz.fill_(hz)


    def _legacy_default_anchors(self):

        center = _hz_to_midi(float(self.bank_median_f0_hz.detach().cpu()))

        count = self.pitch_prototype_count

        if count == 1:

            return torch.tensor([center], dtype=self.pitch_timbre_codes.dtype, device=self.pitch_timbre_codes.device)

        if count == 3:

            return torch.tensor([center - 9.0, center, center + 9.0], dtype=self.pitch_timbre_codes.dtype, device=self.pitch_timbre_codes.device)

        return torch.linspace(center - 9.0, center + 9.0, count, dtype=self.pitch_timbre_codes.dtype, device=self.pitch_timbre_codes.device)


    def configure_pitch_prototypes(self, anchor_midis):

        anchors = torch.as_tensor(anchor_midis, dtype=self.pitch_timbre_codes.dtype, device=self.pitch_timbre_codes.device).reshape(-1)

        if anchors.numel() == 0:

            anchors = torch.tensor([_hz_to_midi(float(self.bank_median_f0_hz))], dtype=self.pitch_timbre_codes.dtype, device=self.pitch_timbre_codes.device)

        old_codes = self.pitch_timbre_codes.detach()

        old_anchors = self.pitch_prototype_midi.detach().to(old_codes.device, old_codes.dtype)

        if old_anchors.numel() != old_codes.shape[0]:

            old_anchors = self._legacy_default_anchors()

        if old_codes.shape[0] == 1:

            new_codes = old_codes.expand(anchors.numel(), -1).clone()

        else:

            order = torch.argsort(old_anchors)

            xa = old_anchors[order]

            ya = old_codes[order]

            rows = []

            for x in anchors:

                if x <= xa[0]:

                    rows.append(ya[0])

                    continue

                if x >= xa[-1]:

                    rows.append(ya[-1])

                    continue

                hi = int(torch.searchsorted(xa, x).item())

                lo = max(0, hi - 1)

                span = torch.clamp(xa[hi] - xa[lo], min=1e-6)

                w = (x - xa[lo]) / span

                rows.append(ya[lo] * (1.0 - w) + ya[hi] * w)

            new_codes = torch.stack(rows, dim=0)

        self.pitch_timbre_codes = nn.Parameter(new_codes.clone())

        self.pitch_prototype_midi = anchors.detach().clone()

        return self


    def _detail_hidden(self, detail, frames):

        if detail is None:

            return None

        if detail.dim() == 2:

            detail = detail.unsqueeze(0)

        if detail.shape[1] != self.detail_dim:

            if detail.shape[1] > self.detail_dim:

                detail = detail[:, :self.detail_dim]

            else:

                detail = F.pad(detail, (0, 0, 0, self.detail_dim - detail.shape[1]))

        if detail.shape[-1] != int(frames):

            detail = F.interpolate(detail, size=int(frames), mode="linear", align_corners=False)

        return F.silu(self.detail_pre(detail))


    def content_representation(self, z):

        h = F.silu(self.content_pre(z))

        delta = 0.075 * torch.tanh(self.content_delta(h))

        return z + delta


    def _target_midi(self, f0, batch, device, dtype):

        if f0 is None:

            center = _hz_to_midi(float(self.bank_median_f0_hz.detach().cpu()))

            return torch.full((int(batch),), center, device=device, dtype=dtype)

        if f0.dim() == 2:

            f0 = f0.unsqueeze(1)

        f0 = f0.to(device=device, dtype=dtype)

        voiced = f0 > 1.0

        midi = 69.0 + 12.0 * torch.log2(torch.clamp(f0, min=20.0) / 440.0)

        mask = voiced.to(dtype)

        return (midi * mask).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1)).clamp(min=1.0)


    def _pitch_weights(self, f0, batch=1, device=None, dtype=None, source_prototype_index=None, timbre_shift_semitones=0.0):

        device = device or self.pitch_timbre_codes.device

        dtype = dtype or self.pitch_timbre_codes.dtype

        anchors = self.pitch_prototype_midi.to(device=device, dtype=dtype).reshape(1, -1)

        count = int(anchors.shape[1])

        if count == 1:

            return torch.ones((int(batch), 1), device=device, dtype=dtype)

        target = self._target_midi(f0, batch, device, dtype).reshape(-1, 1)

        if float(timbre_shift_semitones) != 0.0:

            target = target + float(timbre_shift_semitones)

        sorted_anchors, _ = torch.sort(anchors.reshape(-1))

        spacing = torch.diff(sorted_anchors)

        spacing = spacing[spacing > 0.25]

        sigma = torch.median(spacing) * 0.60 if spacing.numel() else torch.tensor(3.6, device=device, dtype=dtype)

        sigma = torch.clamp(sigma, min=1.6, max=5.2)

        logits = -0.5 * ((target - anchors) / sigma).pow(2)

        if source_prototype_index is not None:

            if not torch.is_tensor(source_prototype_index):

                source_prototype_index = torch.tensor([int(source_prototype_index)], device=device)

            idx = source_prototype_index.to(device=device).long().reshape(-1)

            if idx.numel() == 1 and int(batch) > 1:

                idx = idx.expand(int(batch))

            valid = (idx >= 0) & (idx < count)

            safe_idx = idx.clamp(0, count - 1)

            src_anchor = anchors[0, safe_idx].reshape(-1, 1)

            proximity = torch.exp(-0.5 * ((target - src_anchor) / 8.0).pow(2)).reshape(-1)

            boost = 1.55 * proximity * valid.to(dtype)

            logits = logits.scatter_add(1, safe_idx.reshape(-1, 1), boost.reshape(-1, 1))

        return torch.softmax(logits, dim=1)


    def timbre_representation(self, f0=None, batch=1, source_prototype_index=None, timbre_shift_semitones=0.0):

        device = self.timbre_code.device

        dtype = self.timbre_code.dtype

        weights = self._pitch_weights(

            f0, batch=batch, device=device, dtype=dtype, source_prototype_index=source_prototype_index,

            timbre_shift_semitones=timbre_shift_semitones,

        )

        local = weights @ self.pitch_timbre_codes

        return self.timbre_code.view(1, -1) + 0.70 * local


    def apply_latent(self, z, detail=None, f0=None, source_prototype_index=None, timbre_shift_semitones=0.0, detail_strength=1.0):

        out = self.content_representation(z)

        scale = 1.0 + 0.18 * torch.tanh(self.latent_scale_raw).view(1, -1, 1)

        bias = 0.12 * torch.tanh(self.latent_bias_raw).view(1, -1, 1)

        out = out * scale + bias

        timbre = self.timbre_representation(

            f0=f0, batch=z.shape[0], source_prototype_index=source_prototype_index,

            timbre_shift_semitones=timbre_shift_semitones,

        )

        out = out + 0.10 * torch.tanh(self.timbre_to_latent(timbre)).unsqueeze(-1)

        h = self._detail_hidden(detail, z.shape[-1])

        if h is not None:

            dyn = torch.tanh(self.detail_to_latent(h))

            if float(detail_strength) != 1.0:

                dyn = dyn * float(detail_strength)

            out = out + 0.08 * dyn

        return out


    def spectral_gain(self, bins, device=None, dtype=None, detail=None, frames=None, f0=None, batch=1, source_prototype_index=None, timbre_shift_semitones=0.0, detail_strength=1.0):

        global_log_gain = torch.tanh(self.spectral_log_gain_raw) * 0.40

        timbre = self.timbre_representation(

            f0=f0, batch=batch, source_prototype_index=source_prototype_index,

            timbre_shift_semitones=timbre_shift_semitones,

        )

        timbre_log_gain = 0.22 * torch.tanh(self.timbre_to_spectral(timbre))

        log_gain = global_log_gain.view(1, -1) + timbre_log_gain

        log_gain = F.interpolate(log_gain.unsqueeze(1), size=int(bins), mode="linear", align_corners=False).transpose(1, 2)

        target_frames = int(frames or (detail.shape[-1] if detail is not None else 1))

        log_gain = log_gain.expand(-1, -1, target_frames)

        h = self._detail_hidden(detail, target_frames)

        if h is not None:

            dyn = 0.20 * torch.tanh(self.detail_to_spectral(h))

            dyn = F.interpolate(dyn.unsqueeze(1), size=(int(bins), target_frames), mode="bilinear", align_corners=False).squeeze(1)

            if float(detail_strength) != 1.0:

                dyn = dyn * float(detail_strength)

            log_gain = log_gain + dyn

        gain = torch.exp(log_gain)

        if device is not None:

            gain = gain.to(device)

        if dtype is not None:

            gain = gain.to(dtype=dtype)

        return gain


    def apply_ap(self, ap, detail=None, f0=None, source_prototype_index=None, timbre_shift_semitones=0.0, detail_strength=1.0):

        timbre = self.timbre_representation(

            f0=f0, batch=ap.shape[0], source_prototype_index=source_prototype_index,

            timbre_shift_semitones=timbre_shift_semitones,

        )

        bias = 0.18 * torch.tanh(self.ap_bias_raw).view(1, -1)

        bias = bias + 0.10 * torch.tanh(self.timbre_to_ap(timbre))

        bias = bias.unsqueeze(-1)

        if bias.shape[1] != ap.shape[1]:

            bias = F.interpolate(bias.transpose(1, 2), size=ap.shape[1], mode="linear", align_corners=False).transpose(1, 2)

        out = ap + bias

        h = self._detail_hidden(detail, ap.shape[-1])

        if h is not None:

            dyn = 0.08 * torch.tanh(self.detail_to_ap(h))

            if dyn.shape[1] != ap.shape[1]:

                dyn = F.interpolate(dyn.transpose(1, 2), size=ap.shape[1], mode="linear", align_corners=False).transpose(1, 2)

            if float(detail_strength) != 1.0:

                dyn = dyn * float(detail_strength)

            out = out + dyn

        return out.clamp(0.005, 0.995)


    def apply_gate(self, gate, f0=None, source_prototype_index=None, timbre_shift_semitones=0.0):

        timbre = self.timbre_representation(

            f0=f0, batch=gate.shape[0], source_prototype_index=source_prototype_index,

            timbre_shift_semitones=timbre_shift_semitones,

        )

        bias = 0.12 * torch.tanh(self.gate_bias_raw).view(1, 1)

        bias = bias + 0.08 * torch.tanh(self.timbre_to_gate(timbre))

        return (gate + bias.unsqueeze(-1)).clamp(0.01, 0.99)


    def anti_leak_regularization(self):

        scrub_reg = self.content_delta.weight.pow(2).mean() + self.content_delta.bias.pow(2).mean()

        timbre_proj_reg = (

            self.timbre_to_latent.weight.pow(2).mean()

            + self.timbre_to_spectral.weight.pow(2).mean()

            + self.timbre_to_ap.weight.pow(2).mean()

            + self.timbre_to_gate.weight.pow(2).mean()

        )

        if self.pitch_timbre_codes.shape[0] > 1:

            pitch_smooth = (self.pitch_timbre_codes[1:] - self.pitch_timbre_codes[:-1]).pow(2).mean()

        else:

            pitch_smooth = self.pitch_timbre_codes.new_tensor(0.0)

        pitch_center = self.pitch_timbre_codes.mean(dim=0).pow(2).mean()

        return 0.18 * scrub_reg + 0.05 * timbre_proj_reg + 0.02 * self.timbre_code.pow(2).mean() + 0.012 * pitch_smooth + 0.01 * pitch_center


    def regularization(self):

        detail_reg = self.detail_to_latent.weight.pow(2).mean() + self.detail_to_spectral.weight.pow(2).mean() + self.detail_to_ap.weight.pow(2).mean()

        return (

            self.latent_scale_raw.pow(2).mean()

            + self.latent_bias_raw.pow(2).mean()

            + 0.5 * self.spectral_log_gain_raw.pow(2).mean()

            + 0.5 * self.ap_bias_raw.pow(2).mean()

            + 0.25 * self.gate_bias_raw.pow(2).mean()

            + 0.10 * detail_reg

            + self.anti_leak_regularization()

        )


    def summary(self):

        with torch.no_grad():

            timbre = self.timbre_representation(None, batch=1)

            spec_log = torch.tanh(self.spectral_log_gain_raw) * 0.40

            spec_log = spec_log + 0.22 * torch.tanh(self.timbre_to_spectral(timbre))[0]

            spec_db = (20.0 / torch.log(torch.tensor(10.0))) * spec_log

            scrub_rms = float((0.075 * torch.tanh(self.content_delta.weight)).pow(2).mean().sqrt())

            return {

                "format": ADAPTER_FORMAT,

                "latent_scale_abs_mean": float((0.18 * torch.tanh(self.latent_scale_raw)).abs().mean()),

                "latent_bias_abs_mean": float((0.12 * torch.tanh(self.latent_bias_raw)).abs().mean()),

                "spectral_gain_db_min": float(spec_db.min()),

                "spectral_gain_db_max": float(spec_db.max()),

                "ap_bias_abs_mean": float((0.18 * torch.tanh(self.ap_bias_raw)).abs().mean()),

                "gate_bias": float(0.12 * torch.tanh(self.gate_bias_raw)[0]),

                "detail_latent_weight_rms": float(self.detail_to_latent.weight.pow(2).mean().sqrt()),

                "detail_spectral_weight_rms": float(self.detail_to_spectral.weight.pow(2).mean().sqrt()),

                "detail_ap_weight_rms": float(self.detail_to_ap.weight.pow(2).mean().sqrt()),

                "content_scrubber_weight_rms": scrub_rms,

                "timbre_code_rms": float(self.timbre_code.pow(2).mean().sqrt()),

                "pitch_prototype_count": self.pitch_prototype_count,

                "pitch_prototype_midi": [float(x) for x in self.pitch_prototype_midi.detach().cpu()],

                "pitch_timbre_code_rms": [float(x.pow(2).mean().sqrt()) for x in self.pitch_timbre_codes],

                "bank_median_f0_hz": float(self.bank_median_f0_hz),

            }


def save_adapter(path, adapter, metadata=None):

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {

        "format": ADAPTER_FORMAT,

        "state_dict": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},

        "metadata": metadata or {},

    }

    torch.save(payload, path)


def load_adapter(path, device="cpu"):

    path = Path(path)

    try:

        payload = torch.load(path, map_location=device, weights_only=True)

    except Exception:

        payload = torch.load(path, map_location=device, weights_only=False)

    if not isinstance(payload, dict) or "state_dict" not in payload:

        raise RuntimeError(f"Unsupported adapter file: {path}")

    state = payload["state_dict"]

    latent_dim = int(state["latent_scale_raw"].numel())

    spectral_bands = int(state["spectral_log_gain_raw"].numel())

    ap_bands = int(state["ap_bias_raw"].numel())

    detail_dim = int(state["detail_pre.weight"].shape[1]) if "detail_pre.weight" in state else DEFAULT_DETAIL_DIM

    timbre_dim = int(state["timbre_code"].numel()) if "timbre_code" in state else DEFAULT_TIMBRE_DIM

    prototype_count = int(state["pitch_timbre_codes"].shape[0]) if "pitch_timbre_codes" in state else 3

    prototype_midi = state.get("pitch_prototype_midi")

    adapter = VoicebankAdapter(latent_dim, spectral_bands, ap_bands, detail_dim, timbre_dim, prototype_count, prototype_midi).to(device)

    adapter.load_state_dict(state, strict=False)

    loaded_format = int(payload.get("format", 0))

    if "pitch_prototype_midi" not in state:

        adapter.pitch_prototype_midi = adapter._legacy_default_anchors().detach().clone()

    adapter.eval()

    metadata = payload.get("metadata", {})

    metadata = dict(metadata) if isinstance(metadata, dict) else {}

    metadata["loaded_adapter_format"] = loaded_format

    return adapter, metadata

