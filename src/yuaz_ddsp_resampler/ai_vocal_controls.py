#!/usr/bin/env python3
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

AI_CONTROL_FORMAT = 3
LEGACY_AI_CONTROL_FORMAT = 2
TECHNIQUE_CONTROL_NAMES = ("breathiness", "falsetto", "mixed_voice", "pharyngeal")
CONTROL_NAMES = TECHNIQUE_CONTROL_NAMES
SPECTRAL_BANDS = 64
AP_BANDS = 16


def _resize_freq(x, bands):
    if x.shape[1] == int(bands):
        return x
    return F.interpolate(x.unsqueeze(1), size=(int(bands), x.shape[-1]), mode="bilinear", align_corners=False)[:, 0]


def _resize_time(x, frames):
    if x.shape[-1] == int(frames):
        return x
    return F.interpolate(x, size=int(frames), mode="linear", align_corners=False)


def _curve(value, frames, device, dtype):
    if value is None:
        return torch.zeros((1, 1, int(frames)), device=device, dtype=dtype)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=device, dtype=dtype)
    value = value.to(device=device, dtype=dtype)
    if value.dim() == 0:
        value = value.view(1, 1, 1)
    elif value.dim() == 1:
        value = value.view(1, 1, -1)
    elif value.dim() == 2:
        value = value.unsqueeze(1)
    return _resize_time(value, frames).clamp(-1.0, 1.0)


def _logit(x, eps=1e-4):
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


class AIControlAdapter(nn.Module):
    def __init__(
        self,
        hidden=112,
        spectral_bands=SPECTRAL_BANDS,
        ap_bands=AP_BANDS,
        control_names=TECHNIQUE_CONTROL_NAMES,
        control_modes=None,
        output_scopes=("spectral", "ap", "gate"),
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.spectral_bands = int(spectral_bands)
        self.ap_bands = int(ap_bands)
        self.control_names = tuple(control_names)
        if control_modes is None:
            control_modes = ["positive"] * len(self.control_names)
        self.control_modes = tuple(str(x) for x in control_modes)
        if len(self.control_modes) != len(self.control_names):
            raise ValueError("control_modes length must match control_names")
        if any(x not in {"positive", "signed"} for x in self.control_modes):
            raise ValueError(f"unsupported control mode: {self.control_modes}")
        self.output_scopes = tuple(str(x) for x in output_scopes)
        allowed_scopes = {"spectral", "ap", "gate"}
        if not self.output_scopes or any(x not in allowed_scopes for x in self.output_scopes):
            raise ValueError(f"unsupported output scopes: {self.output_scopes}")
        in_ch = self.spectral_bands + self.ap_bands + 1 + 1 + len(self.control_names)
        out_ch = self.spectral_bands + self.ap_bands + 1
        self.input_proj = nn.Conv1d(in_ch, self.hidden, 1)
        self.temporal1 = nn.Conv1d(self.hidden, self.hidden, 5, padding=2)
        self.temporal2 = nn.Conv1d(self.hidden, self.hidden, 5, padding=2)
        self.temporal3 = nn.Conv1d(self.hidden, self.hidden, 3, padding=1)
        self.output_proj = nn.Conv1d(self.hidden, out_ch, 1)
        nn.init.kaiming_uniform_(self.input_proj.weight, a=0.2)
        nn.init.zeros_(self.input_proj.bias)
        for layer in (self.temporal1, self.temporal2, self.temporal3):
            nn.init.kaiming_uniform_(layer.weight, a=0.2)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        self.runtime_gain = 1.0
        self.runtime_gain_cap = 4.0
        self.runtime_effect_floor = 0.055
        self.last_effect_stats = {}

    def _context(self, spectral_envelope, ap_bands, gate, f0, controls):
        frames = spectral_envelope.shape[-1]
        log_s = torch.log(spectral_envelope.clamp(min=1e-7))
        log_s = _resize_freq(log_s, self.spectral_bands)
        log_s = log_s - log_s.mean(dim=1, keepdim=True)
        ap = _resize_freq(_resize_time(ap_bands, frames), self.ap_bands)
        g = _resize_time(gate, frames)
        f = _resize_time(f0, frames)
        voiced = (f > 1.0).to(f.dtype)
        log_f0 = torch.log2(torch.clamp(f, min=40.0) / 220.0) / 3.0
        log_f0 = torch.clamp(log_f0, -1.5, 1.5) * voiced
        cs = []
        for name, mode in zip(self.control_names, self.control_modes):
            c = _curve(controls.get(name), frames, spectral_envelope.device, spectral_envelope.dtype)
            if mode == "positive":
                c = torch.clamp(c, 0.0, 1.0)
            else:
                c = torch.clamp(c, -1.0, 1.0)
            cs.append(c)
        c = torch.cat(cs, dim=1)
        return torch.cat([log_s, ap, g, log_f0, c], dim=1), c, voiced

    def predict_residuals(self, spectral_envelope, ap_bands, gate, f0, controls):
        x, c, voiced = self._context(spectral_envelope, ap_bands, gate, f0, controls)
        h = F.silu(self.input_proj(x))
        h = h + 0.45 * F.silu(self.temporal1(h))
        h = h + 0.45 * F.silu(self.temporal2(h))
        h = h + 0.30 * F.silu(self.temporal3(h))
        y = self.output_proj(h)
        ds = y[:, :self.spectral_bands]
        da = y[:, self.spectral_bands:self.spectral_bands + self.ap_bands]
        dg = y[:, -1:]
        strength = torch.amax(torch.abs(c), dim=1, keepdim=True)
        mask = (strength > 1e-6).to(c.dtype) * voiced
        ds = ds * mask if "spectral" in self.output_scopes else torch.zeros_like(ds)
        da = da * mask if "ap" in self.output_scopes else torch.zeros_like(da)
        dg = dg * mask if "gate" in self.output_scopes else torch.zeros_like(dg)
        return ds, da, dg

    def _phonation_routed_residuals(self, spectral_envelope, ap_bands, gate, f0, controls):
        if tuple(self.control_names) != ("tension", "voicing"):
            return None
        frames = spectral_envelope.shape[-1]
        t = _curve(controls.get("tension"), frames, spectral_envelope.device, spectral_envelope.dtype)
        v = _curve(controls.get("voicing"), frames, spectral_envelope.device, spectral_envelope.dtype)
        zero = torch.zeros_like(v)
        tension_controls = dict(controls)
        tension_controls["voicing"] = zero
        t_ds, t_da, t_dg = self.predict_residuals(
            spectral_envelope, ap_bands, gate, f0, tension_controls
        )

        t_amount = torch.clamp(torch.abs(t), 0.0, 1.0)
        t_progress = t_amount * t_amount * (3.0 - 2.0 * t_amount)
        t_ds = t_ds * t_progress * (0.10 + 0.20 * t_progress)
        t_da = t_da * t_progress * (0.08 + 0.16 * t_progress)
        t_dg = t_dg * t_progress * (0.015 + 0.035 * t_progress)

        if float(torch.max(torch.abs(v)).detach().cpu()) <= 1e-6:
            return t_ds, t_da, t_dg

        magnitude = torch.abs(v)
        pos_controls = dict(controls)
        pos_controls["tension"] = zero
        pos_controls["voicing"] = magnitude
        neg_controls = dict(controls)
        neg_controls["tension"] = zero
        neg_controls["voicing"] = -magnitude
        _, pos_da, _ = self.predict_residuals(
            spectral_envelope, ap_bands, gate, f0, pos_controls
        )
        _, neg_da, _ = self.predict_residuals(
            spectral_envelope, ap_bands, gate, f0, neg_controls
        )
        yv_neural_ap_scale = 0.25
        signed_odd_ap = 0.5 * (pos_da - neg_da) * torch.sign(v) * yv_neural_ap_scale
        return t_ds, t_da + signed_odd_ap, t_dg

    def _technique_routed_residuals(self, spectral_envelope, ap_bands, gate, f0, controls):
        if tuple(self.control_names) != TECHNIQUE_CONTROL_NAMES:
            return None
        frames = spectral_envelope.shape[-1]
        falsetto = _curve(controls.get("falsetto"), frames, spectral_envelope.device, spectral_envelope.dtype)
        falsetto = torch.clamp(falsetto, 0.0, 1.0)
        if float(torch.max(falsetto).detach().cpu()) <= 1e-6:
            return None

        zero = torch.zeros_like(falsetto)
        base_controls = dict(controls)
        base_controls["falsetto"] = zero
        base_ds, base_da, base_dg = self.predict_residuals(
            spectral_envelope, ap_bands, gate, f0, base_controls
        )
        full_ds, full_da, full_dg = self.predict_residuals(
            spectral_envelope, ap_bands, gate, f0, controls
        )
        yf_ds = full_ds - base_ds
        yf_da = full_da - base_da
        yf_dg = full_dg - base_dg

        yb_controls = {name: zero for name in self.control_names}
        yb_controls["breathiness"] = falsetto
        yb_ds, _, _ = self.predict_residuals(
            spectral_envelope, ap_bands, gate, f0, yb_controls
        )
        dot = torch.sum(yf_ds * yb_ds, dim=1, keepdim=True)
        norm = torch.sum(yb_ds * yb_ds, dim=1, keepdim=True).clamp(min=1e-8)
        projection = torch.clamp(dot / norm, min=0.0) * yb_ds

        yf_spectral_overlap = 0.78
        yf_ap_scale = 0.18
        yf_gate_scale = 0.12
        return (
            base_ds + yf_ds - yf_spectral_overlap * projection,
            base_da + yf_ap_scale * yf_da,
            base_dg + yf_gate_scale * yf_dg,
        )

    def apply(self, spectral_envelope, ap_bands, gate, f0, controls):
        routed = self._phonation_routed_residuals(
            spectral_envelope, ap_bands, gate, f0, controls
        )
        route_mode = "standard"
        if routed is None:
            routed = self._technique_routed_residuals(
                spectral_envelope, ap_bands, gate, f0, controls
            )
            if routed is None:
                ds, da, dg = self.predict_residuals(spectral_envelope, ap_bands, gate, f0, controls)
            else:
                ds, da, dg = routed
                route_mode = "technique-yf-register-v2"
        else:
            ds, da, dg = routed
            route_mode = "phonation-yv-odd-ap-v2"
        with torch.no_grad():
            rms_s = float(torch.sqrt(torch.mean(torch.tanh(ds).pow(2)) + 1e-12).cpu()) if "spectral" in self.output_scopes else 0.0
            rms_a = float(torch.sqrt(torch.mean(torch.tanh(da).pow(2)) + 1e-12).cpu()) if "ap" in self.output_scopes else 0.0
            rms_g = float(torch.sqrt(torch.mean(torch.tanh(dg).pow(2)) + 1e-12).cpu()) if "gate" in self.output_scopes else 0.0
            raw_effect = max(rms_s, rms_a, rms_g)
            base_gain = max(1.0, float(getattr(self, "runtime_gain", 1.0)))
            if raw_effect > 1e-6 and raw_effect < float(getattr(self, "runtime_effect_floor", 0.055)):
                base_gain *= min(2.5, float(getattr(self, "runtime_effect_floor", 0.055)) / raw_effect)
            gain = min(float(getattr(self, "runtime_gain_cap", 4.0)), base_gain)
            if tuple(self.control_names) == ("tension", "voicing"):
                tension_curve = _curve(controls.get("tension"), spectral_envelope.shape[-1], spectral_envelope.device, spectral_envelope.dtype)
                voicing_curve = _curve(controls.get("voicing"), spectral_envelope.shape[-1], spectral_envelope.device, spectral_envelope.dtype)
                if float(torch.max(torch.abs(tension_curve)).detach().cpu()) > 1e-6 and float(torch.max(torch.abs(voicing_curve)).detach().cpu()) <= 1e-6:
                    gain = min(gain, 2.0)
        ds_full = _resize_freq(ds * gain, spectral_envelope.shape[1])
        da_full = _resize_freq(da * gain, ap_bands.shape[1])
        da_full = _resize_time(da_full, ap_bands.shape[-1])
        dg_full = _resize_time(dg * gain, gate.shape[-1])
        out_s = spectral_envelope * torch.exp(0.82 * torch.tanh(ds_full))
        out_ap = torch.sigmoid(_logit(ap_bands) + 1.55 * torch.tanh(da_full))
        out_gate = torch.sigmoid(_logit(gate) + 1.55 * torch.tanh(dg_full))
        frames = spectral_envelope.shape[-1]
        cs = []
        active_values = {}
        for name, mode in zip(self.control_names, self.control_modes):
            c = _curve(controls.get(name), frames, spectral_envelope.device, spectral_envelope.dtype)
            c = torch.clamp(c, 0.0, 1.0) if mode == "positive" else torch.clamp(c, -1.0, 1.0)
            cs.append(c)
            active_values[name] = float(torch.max(torch.abs(c)).detach().cpu())
        activity = torch.amax(torch.abs(torch.cat(cs, dim=1)), dim=1, keepdim=True)
        f = _resize_time(f0, frames)
        active_s = (activity > 0) & (f > 1.0)
        active_ap = _resize_time(active_s.to(ap_bands.dtype), ap_bands.shape[-1]) > 0.5
        active_gate = _resize_time(active_s.to(gate.dtype), gate.shape[-1]) > 0.5
        out_s = torch.where(active_s, out_s.clamp(min=1e-7), spectral_envelope)
        out_ap = torch.where(active_ap, out_ap.clamp(0.012, 0.988), ap_bands)
        out_gate = torch.where(active_gate, out_gate.clamp(0.02, 0.98), gate)
        with torch.no_grad():
            ds_out = torch.log(out_s.clamp(min=1e-7) / spectral_envelope.clamp(min=1e-7))
            da_out = out_ap - ap_bands
            dg_out = out_gate - gate
            self.last_effect_stats = {
                "controls": dict(active_values),
                "raw_spectral_rms": rms_s,
                "raw_ap_rms": rms_a,
                "raw_gate_rms": rms_g,
                "runtime_gain": float(gain),
                "control_gate_mode": "source-active-voiced",
                "runtime_route": route_mode,
                "applied_spectral_log_rms": float(torch.sqrt(torch.mean(ds_out.pow(2)) + 1e-12).cpu()),
                "applied_ap_rms": float(torch.sqrt(torch.mean(da_out.pow(2)) + 1e-12).cpu()),
                "applied_gate_rms": float(torch.sqrt(torch.mean(dg_out.pow(2)) + 1e-12).cpu()),
                "collapsed": bool(max(rms_s, rms_a, rms_g) < 1e-5 and max(active_values.values(), default=0.0) > 0.0),
            }
        return out_s, out_ap, out_gate


def save_ai_control_adapter(path, model, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": AI_CONTROL_FORMAT,
        "model": {
            "hidden": model.hidden,
            "spectral_bands": model.spectral_bands,
            "ap_bands": model.ap_bands,
            "control_names": list(model.control_names),
            "control_modes": list(model.control_modes),
            "output_scopes": list(model.output_scopes),
        },
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, path)
    return path


def load_ai_control_adapter(path, device="cpu", expected_controls=None):
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    fmt = int(payload.get("format", 0))
    if fmt not in {LEGACY_AI_CONTROL_FORMAT, AI_CONTROL_FORMAT}:
        raise RuntimeError(f"Unsupported AI control model format: {payload.get('format')}")
    cfg = payload.get("model") or {}
    controls = tuple(cfg.get("control_names") or ())
    if expected_controls is not None and tuple(expected_controls) != controls:
        raise RuntimeError(f"AI control model has incompatible controls: {controls}, expected {tuple(expected_controls)}")
    if not controls:
        raise RuntimeError("AI control model declares no controls")
    modes = tuple(cfg.get("control_modes") or (["positive"] * len(controls)))
    model = AIControlAdapter(
        hidden=int(cfg.get("hidden", 112)),
        spectral_bands=int(cfg.get("spectral_bands", SPECTRAL_BANDS)),
        ap_bands=int(cfg.get("ap_bands", AP_BANDS)),
        control_names=controls,
        control_modes=modes,
        output_scopes=tuple(cfg.get("output_scopes") or ("spectral", "ap", "gate")),
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    metadata = dict(payload.get("metadata") or {})
    defaults = {
        ("breathiness", "falsetto", "mixed_voice", "pharyngeal"): 2.15,
        ("gender_formant",): 2.10,
        ("tension", "voicing"): 2.00,
        ("mouth",): 2.05,
    }
    model.runtime_gain = float(metadata.get("runtime_gain", defaults.get(tuple(controls), 1.8)))
    model.runtime_gain_cap = float(metadata.get("runtime_gain_cap", 4.0))
    model.runtime_effect_floor = float(metadata.get("runtime_effect_floor", 0.055))
    model.eval()
    return model, metadata


def inspect_ai_control_adapter(path):
    model, meta = load_ai_control_adapter(path, device="cpu")
    return {"controls": list(model.control_names), "control_modes": list(model.control_modes), "output_scopes": list(model.output_scopes), "metadata": meta}
