#!/usr/bin/env python3

import hashlib

import json

import math

import os

import platform

import re

import sys

import threading

import time

from pathlib import Path


import librosa

import numpy as np

import soundfile as sf

import torch

import torch.nn.functional as F

import yaml


from .adapter import load_adapter

from .controls import parse_yuaz_controls

from .articulation import analyze_articulation_regions, load_canonical_articulation, map_articulation_regions, single_source_articulation_hybrid

from .fidelity import load_refiner

from .loudness import normalize_final_render, oto_loudness_signature

from .learned_highband import load_profile_database, select_learned_profile, synthesize_learned_highband

from .voicebank import file_sha256, pcm_fingerprint


B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def load_yaml(path):

    with open(path, "r", encoding="utf-8") as f:

        return yaml.safe_load(f) or {}


def load_config(repo):

    config_dir = repo / "yuaz_sgr" / "config"

    if not config_dir.exists():

        raise FileNotFoundError(f"找不到 Yuaz 配置目录: {config_dir}")

    config = {}

    for name in ("base", "model", "train"):

        p = config_dir / f"{name}.yaml"

        if p.exists():

            config.update(load_yaml(p))

    return config


def normalize_checkpoint_state(ckpt):

    if isinstance(ckpt, dict):

        for key in ("model", "state_dict", "model_state_dict"):

            candidate = ckpt.get(key)

            if isinstance(candidate, dict) and candidate:

                ckpt = candidate

                break

    if not isinstance(ckpt, dict):

        raise RuntimeError("无法识别 checkpoint 结构。")

    state = {}

    for key, value in ckpt.items():

        if not torch.is_tensor(value):

            continue

        clean = key

        changed = True

        while changed:

            changed = False

            for prefix in ("_orig_mod.", "module.", "model."):

                if clean.startswith(prefix):

                    clean = clean[len(prefix):]

                    changed = True

        state[clean] = value

    return state


def load_checkpoint(path):

    try:

        ckpt = torch.load(path, map_location="cpu", weights_only=True)

    except Exception:

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    return ckpt, normalize_checkpoint_state(ckpt)


def load_component(module, full_state, prefix):

    target = module.state_dict()

    selected = {}

    for key, value in full_state.items():

        if key.startswith(prefix + "."):

            local = key[len(prefix) + 1:]

            if local in target and target[local].shape == value.shape:

                selected[local] = value

    module.load_state_dict(selected, strict=False)

    total_numel = sum(v.numel() for v in target.values())

    loaded_numel = sum(target[k].numel() for k in selected)

    return float(loaded_numel / max(1, total_numel))


def import_yuaz_modules(repo):

    core = repo / "yuaz_sgr"

    if not (core / "models").exists():

        raise FileNotFoundError(f"这不像 Yuaz SGR 仓库: {repo}")

    sys.path.insert(0, str(core))

    from models.encoder import Encoder

    from models.ddsp_decoder import DDSPDecoder

    from models.rvq import ResidualVectorQuantizer

    return Encoder, DDSPDecoder, ResidualVectorQuantizer


def make_adaptive_decoder_class(base_cls):

    class AdaptiveDDSPDecoder(base_cls):

        def harmonic_oscillator(self, f0, n_samples):

            B, _, _ = f0.shape

            device = f0.device

            f0_up = F.interpolate(f0, size=n_samples, mode="linear", align_corners=False)

            voiced = (f0_up > 1.0).to(f0_up.dtype)

            safe_f0 = torch.clamp(f0_up, min=0.0)

            phase = 2 * np.pi * torch.cumsum(safe_f0 / self.sample_rate, dim=-1)

            phase = phase.unsqueeze(2)

            k = torch.arange(1, self.n_harmonics + 1, device=device, dtype=f0.dtype).view(1, 1, -1, 1)

            harmonic_hz = safe_f0.unsqueeze(2) * k

            valid = ((harmonic_hz < (self.sample_rate * 0.5 - 1.0)) & (harmonic_hz > 0.0)).to(f0.dtype)

            harmonics = ((torch.sin(k * phase) / k) * valid).sum(dim=2)

            return harmonics * voiced


        @staticmethod

        def _smooth_ap_bands(A_c):

            B, C, T = A_c.shape

            x = A_c.reshape(B * C, 1, T)

            if T >= 3:

                x = F.pad(x, (2, 2), mode="replicate")

                x = F.avg_pool1d(x, kernel_size=5, stride=1)

            x = x.reshape(B, C, T)

            y = x.transpose(1, 2).reshape(B * T, 1, C)

            if C >= 3:

                kernel = torch.tensor([0.25, 0.50, 0.25], device=A_c.device, dtype=A_c.dtype).view(1, 1, 3)

                y = F.pad(y, (1, 1), mode="replicate")

                y = F.conv1d(y, kernel)

            return y.reshape(B, T, C).transpose(1, 2).clamp(0.0, 1.0)


        @staticmethod

        def _soft_mvf_ap(A_band, voiced_frames):

            B, C, T = A_band.shape

            periodic_prob = torch.sigmoid((0.50 - A_band) * 7.0)

            expected_count = periodic_prob.sum(dim=1, keepdim=True).clamp(1.0, float(C))

            mvf_band = expected_count - 0.5

            idx = torch.arange(C, device=A_band.device, dtype=A_band.dtype).view(1, C, 1)

            harmonic_gate = torch.sigmoid((mvf_band - idx) / 1.20)

            mvf_ap = 1.0 - harmonic_gate

            adaptive = 0.82 * A_band + 0.18 * mvf_ap

            adaptive = adaptive.clamp(0.015, 0.985)

            return torch.where(voiced_frames > 0.5, adaptive, A_band)


        @staticmethod

        def _energy_match(reference, candidate, max_db=3.0):

            ref2 = reference.pow(2)

            can2 = candidate.pow(2)

            kernel = 1024 if reference.shape[-1] >= 1024 else max(64, reference.shape[-1] // 4 * 2 + 1)

            if kernel % 2 == 0:

                kernel += 1

            pad = kernel // 2

            ref_rms = torch.sqrt(F.avg_pool1d(F.pad(ref2, (pad, pad), mode="replicate"), kernel, stride=1) + 1e-9)

            can_rms = torch.sqrt(F.avg_pool1d(F.pad(can2, (pad, pad), mode="replicate"), kernel, stride=1) + 1e-9)

            gain = ref_rms / can_rms

            limit = float(10.0 ** (max_db / 20.0))

            gain = gain.clamp(1.0 / limit, limit)

            gk = 257 if gain.shape[-1] >= 257 else max(17, gain.shape[-1] // 8 * 2 + 1)

            if gk % 2 == 0:

                gk += 1

            gp = gk // 2

            gain = F.avg_pool1d(F.pad(gain, (gp, gp), mode="replicate"), gk, stride=1)

            return candidate * gain


        def forward(self, f0, z, target_samples=None, cond=None, adapter=None, detail=None, prototype_index=None, timbre_shift_semitones=0.0, detail_strength=1.0, return_aux=False):

            if f0.dim() == 2:

                f0 = f0.unsqueeze(1)

            B, _, T = f0.shape

            if adapter is not None:

                z = adapter.apply_latent(

                    z, detail=detail, f0=f0, source_prototype_index=prototype_index,

                    timbre_shift_semitones=timbre_shift_semitones, detail_strength=detail_strength,

                )

            feat = torch.cat([z, f0], dim=1).transpose(1, 2)

            feat = self.emformer_input_proj(feat)

            h, _ = self.emformer(feat)

            h = self.emformer_proj(h).transpose(1, 2)


            S_c = self.env_net(h.transpose(1, 2)).transpose(1, 2)

            A_c_raw = self.ap_net(h.transpose(1, 2)).transpose(1, 2)

            if cond is not None:

                if cond.shape[-1] != h.shape[-1]:

                    cond = F.interpolate(cond, size=h.shape[-1], mode="linear", align_corners=False)

                h_w = torch.cat([h, cond], dim=1)

                h_w = self.cond_fuse(h_w.transpose(1, 2)).transpose(1, 2)

            else:

                h_w = h

            gate = self.weight_net(h_w.transpose(1, 2)).transpose(1, 2)

            if adapter is not None:

                gate = adapter.apply_gate(

                    gate, f0=f0, source_prototype_index=prototype_index,

                    timbre_shift_semitones=timbre_shift_semitones,

                )


            S_lin = self._decompress_envelope(S_c)

            if adapter is not None:

                S_lin = S_lin * adapter.spectral_gain(

                    S_lin.shape[1], device=S_lin.device, dtype=S_lin.dtype, detail=detail, frames=S_lin.shape[-1],

                    f0=f0, batch=S_lin.shape[0], source_prototype_index=prototype_index,

                    timbre_shift_semitones=timbre_shift_semitones, detail_strength=detail_strength,

                )

            voiced_frames = (f0 > 1.0).to(f0.dtype)

            A_smooth_base = self._smooth_ap_bands(A_c_raw)

            A_smooth = A_smooth_base

            if adapter is not None:

                A_smooth = adapter.apply_ap(

                    A_smooth, detail=detail, f0=f0, source_prototype_index=prototype_index,

                    timbre_shift_semitones=timbre_shift_semitones, detail_strength=detail_strength,

                )

            A_band = self._soft_mvf_ap(A_smooth, voiced_frames)


            def expand_ap(A_band_in):

                x = A_band_in.transpose(1, 2)

                x = F.interpolate(x, size=self.fft_size // 2 + 1, mode="linear", align_corners=False)

                return x.transpose(1, 2)


            A_lin = expand_ap(A_band)

            A_raw_lin = expand_ap(A_c_raw)


            n_samples = T * self.encoder_hop_length

            e_h = self.harmonic_oscillator(f0, n_samples)

            spec_h = torch.stft(

                e_h.squeeze(1), self.fft_size, self.hop_length,

                window=self.hann_window.to(z.device), return_complex=True,

            )

            spec_h = spec_h[:, :self.fft_size // 2 + 1, :]

            n_spec_frames = spec_h.shape[-1]

            S_spec = F.interpolate(S_lin, size=n_spec_frames, mode="linear", align_corners=False)

            A_spec = F.interpolate(A_lin, size=n_spec_frames, mode="linear", align_corners=False)

            A_raw_spec = F.interpolate(A_raw_lin, size=n_spec_frames, mode="linear", align_corners=False)


            e_n = torch.randn(B, n_samples, device=z.device)

            spec_n = torch.stft(

                e_n, self.fft_size, self.hop_length,

                window=self.hann_window.to(z.device), return_complex=True,

            )

            spec_n = spec_n[:, :self.fft_size // 2 + 1, :n_spec_frames]


            def synth_with_ap(A_use):

                spec_h_filt = spec_h * (1.0 - A_use) * S_spec

                x_h = torch.istft(

                    spec_h_filt, self.fft_size, self.hop_length,

                    window=self.hann_window.to(z.device), length=n_samples,

                )

                spec_n_filt = spec_n * A_use[:, :, :n_spec_frames] * S_spec[:, :, :n_spec_frames]

                x_n = torch.istft(

                    spec_n_filt, self.fft_size, self.hop_length,

                    window=self.hann_window.to(z.device), length=n_samples,

                )

                lh = F.interpolate(gate, size=n_samples, mode="linear", align_corners=False)

                ln = F.interpolate(1.0 - gate, size=n_samples, mode="linear", align_corners=False)

                return lh * x_h.unsqueeze(1) + ln * x_n.unsqueeze(1)


            wav = synth_with_ap(A_spec)

            raw_wav = synth_with_ap(A_raw_spec)

            wav = self._energy_match(raw_wav, wav, max_db=3.0)


            if target_samples is not None and wav.shape[-1] != target_samples:

                wav = wav[..., :target_samples]

            if return_aux:

                return wav, {

                    "spectral_envelope": S_lin,

                    "ap_raw": A_c_raw,

                    "ap_smoothed_base": A_smooth_base,

                    "ap_after_adapter": A_smooth,

                    "ap_after_soft_mvf": A_band,

                    "gate": gate,

                    "voiced_frames": voiced_frames,

                }

            return wav

    return AdaptiveDDSPDecoder


def build_modules(repo, config, device):

    Encoder, BaseDDSPDecoder, ResidualVectorQuantizer = import_yuaz_modules(repo)

    Decoder = make_adaptive_decoder_class(BaseDDSPDecoder)

    enc = config["encoder"]

    ddsp = config["ddsp"]

    rvq = config["rvq"]

    sample_rate = int(config.get("sample_rate", enc.get("sample_rate", 24000)))

    encoder = Encoder(

        hidden_dim=enc["hidden_dim"], latent_dim=enc["latent_dim"], sample_rate=sample_rate,

        n_mfcc=enc["n_mfcc"], n_fft=enc["n_fft"], hop_length=enc["hop_length"],

    )

    decoder = Decoder(

        n_harmonics=ddsp["n_harmonics"], sample_rate=sample_rate, fft_size=ddsp["fft_size"],

        hop_length=ddsp["hop_length"], hidden_dim=ddsp["hidden_dim"], encoder_hop_length=enc["hop_length"],

        emformer_num_layers=ddsp["emformer_num_layers"], emformer_num_heads=ddsp["emformer_num_heads"],

        emformer_ffn_dim=ddsp["emformer_ffn_dim"], emformer_segment_length=ddsp["emformer_segment_length"],

        emformer_left_context=ddsp["emformer_left_context"], emformer_right_context=ddsp["emformer_right_context"],

    )

    quantizer = ResidualVectorQuantizer(

        num_quantizers=rvq["num_quantizers"], codebook_size=rvq["codebook_size"],

        codebook_dim=rvq["codebook_dim"], commitment_weight=rvq["commitment_weight"],

    )

    for module in (encoder, decoder, quantizer):

        module.eval().to(device)

    return encoder, decoder, quantizer, sample_rate, int(enc["hop_length"])


def decode_int12_pitch(text):

    text = text or "AA"

    values = []

    i = 0

    last = None

    while i < len(text):

        if text[i] == "#":

            j = text.find("#", i + 1)

            if j < 0:

                raise ValueError("pitch string 的 RLE # 没有闭合")

            if last is None:

                raise ValueError("pitch string 在首值前出现 RLE")

            count = int(text[i + 1:j] or "0")

            values.extend([last] * count)

            i = j + 1

            continue

        if i + 1 >= len(text):

            raise ValueError("pitch string 长度不完整")

        try:

            a = B64.index(text[i])

            b = B64.index(text[i + 1])

        except ValueError as e:

            raise ValueError("pitch string 含非法 Base64 字符") from e

        v = (a << 6) | b

        if v >= 2048:

            v -= 4096

        values.append(v)

        last = v

        i += 2

    return np.asarray(values, dtype=np.float32)


def tone_to_midi(tone):

    m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", str(tone).strip())

    if not m:

        raise ValueError(f"无法识别音符名: {tone}")

    note, accidental, octave = m.group(1).upper(), m.group(2), int(m.group(3))

    pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[note]

    if accidental == "#":

        pc += 1

    elif accidental == "b":

        pc -= 1

    return (octave + 1) * 12 + pc


def midi_to_hz(midi):

    return 440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0))


def parse_tempo(value):

    s = str(value).strip()

    if s.startswith("!"):

        s = s[1:]

    try:

        t = float(s)

    except Exception:

        t = 120.0

    return max(1.0, t)


def crop_oto(audio, sr, offset_ms, cutoff_ms):

    start = int(round(max(0.0, float(offset_ms)) * sr / 1000.0))

    start = min(start, len(audio))

    cutoff_ms = float(cutoff_ms)

    if cutoff_ms >= 0:

        end = len(audio) - int(round(cutoff_ms * sr / 1000.0))

    else:

        end = start + int(round((-cutoff_ms) * sr / 1000.0))

    end = int(np.clip(end, start, len(audio)))

    return audio[start:end]


def read_audio(path, target_sr):

    audio, sr = sf.read(path, always_2d=False)

    if audio.ndim > 1:

        audio = np.mean(audio, axis=1)

    audio = np.asarray(audio, dtype=np.float32)

    audio = np.nan_to_num(audio)

    if sr != target_sr:

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0

    if peak > 1.0:

        audio /= peak

    return audio


def extract_detail_features(audio, sr, hop, n_mels=24):

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size < 32:

        return np.zeros((int(n_mels) + 1, 1), dtype=np.float32)

    n_fft = 512

    fmax = min(float(sr) * 0.5 - 20.0, 11000.0)

    fmin = min(1800.0, max(80.0, fmax * 0.25))

    mel = librosa.feature.melspectrogram(

        y=audio, sr=int(sr), n_fft=n_fft, hop_length=int(hop), win_length=n_fft,

        n_mels=int(n_mels), fmin=fmin, fmax=fmax, power=1.0, center=True,

    ).astype(np.float32)

    x = np.log1p(12.0 * mel)

    center = float(np.median(x))

    scale = float(np.std(x)) + 1e-4

    x = np.clip((x - center) / scale, -4.0, 4.0)

    if x.shape[1] > 1:

        flux = np.mean(np.abs(np.diff(x, axis=1, prepend=x[:, :1])), axis=0, keepdims=True)

    else:

        flux = np.zeros((1, x.shape[1]), dtype=np.float32)

    flux = np.clip(flux / (float(np.median(flux)) + 0.25), 0.0, 4.0)

    return np.concatenate([x, flux.astype(np.float32)], axis=0).astype(np.float32)


def extract_f0(audio, sr, hop):

    frame_length = 2048

    try:

        f0, voiced_flag, _ = librosa.pyin(

            audio, fmin=50.0, fmax=min(1100.0, sr / 2 - 1), sr=sr,

            frame_length=frame_length, hop_length=hop, center=True,

        )

        if f0 is None:

            raise RuntimeError("pYIN 未返回 F0")

        f0 = np.asarray(f0, dtype=np.float32)

        voiced = np.isfinite(f0)

        if voiced_flag is not None:

            voiced &= np.asarray(voiced_flag, dtype=bool)

        f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)

        f0[~voiced] = 0.0

    except Exception:

        f0 = librosa.yin(

            audio, fmin=50.0, fmax=min(1100.0, sr / 2 - 1), sr=sr,

            frame_length=frame_length, hop_length=hop,

        ).astype(np.float32)

        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop).squeeze()

        threshold = max(1e-5, float(np.percentile(rms, 20)) * 1.5)

        voiced = np.asarray(rms > threshold, dtype=bool)

        n = min(len(f0), len(voiced))

        f0, voiced = f0[:n], voiced[:n]

        f0[~voiced] = 0.0

    if np.count_nonzero(f0 > 0) == 0:

        raise RuntimeError("没有检测到可靠 F0")

    return f0.astype(np.float32)


def interpolate_tensor(x, frames, mode="linear"):

    frames = int(max(1, frames))

    if x.shape[-1] == frames:

        return x

    if mode == "nearest":

        return F.interpolate(x, size=frames, mode="nearest")

    return F.interpolate(x, size=frames, mode="linear", align_corners=False)


def piecewise_warp(x, source_fixed_frames, target_frames, mode="linear", target_fixed_frames=None):

    source_frames = int(x.shape[-1])

    target_frames = int(max(2, target_frames))

    if source_frames <= 2:

        return interpolate_tensor(x, target_frames, mode=mode)

    source_fixed = int(np.clip(source_fixed_frames, 0, source_frames - 2))

    if target_fixed_frames is None:

        target_fixed = min(source_fixed, max(0, target_frames - 2))

    else:

        target_fixed = int(np.clip(target_fixed_frames, 0, max(0, target_frames - 2)))

    prefix = x[..., :source_fixed]

    prefix_warped = interpolate_tensor(prefix, target_fixed, mode=mode) if target_fixed > 0 and source_fixed > 0 else x[..., :0]

    tail = x[..., source_fixed:]

    tail_warped = interpolate_tensor(tail, target_frames - target_fixed, mode=mode)

    return torch.cat([prefix_warped, tail_warped], dim=-1)


def fill_f0(f0, onset_frames):

    arr = np.asarray(f0, dtype=np.float32).copy()

    valid = np.flatnonzero((arr > 0) & (np.arange(len(arr)) >= onset_frames))

    if len(valid) == 0:

        return arr

    first, last = int(valid[0]), int(valid[-1])

    arr[first:last + 1] = np.interp(np.arange(first, last + 1), valid, arr[valid]).astype(np.float32)

    if last < len(arr) - 1:

        arr[last + 1:] = arr[last]

    arr[:onset_frames] = 0.0

    return arr


def build_target_f0(tone, pitch_encoded, tempo, frames, sr, hop, source_f0, modulation, voiced_mask=None):

    pitch = decode_int12_pitch(pitch_encoded)

    if pitch.size == 0:

        pitch = np.zeros(1, dtype=np.float32)

    interval_ms = 60000.0 / parse_tempo(tempo) / 480.0 * 5.0

    frame_ms = np.arange(frames, dtype=np.float32) * hop * 1000.0 / sr

    pitch_ms = np.arange(len(pitch), dtype=np.float32) * interval_ms

    cents = np.interp(frame_ms, pitch_ms, pitch, left=float(pitch[0]), right=float(pitch[-1])).astype(np.float32)


    base_midi = tone_to_midi(tone)

    base_hz = midi_to_hz(base_midi)

    target = base_hz * np.power(2.0, cents / 1200.0)


    mod = float(modulation) / 100.0

    if abs(mod) > 1e-6:

        src = np.asarray(source_f0, dtype=np.float32)

        voiced = src > 0

        if np.any(voiced):

            med = float(np.median(src[voiced]))

            deviation_cents = np.zeros_like(src)

            deviation_cents[voiced] = 1200.0 * np.log2(np.maximum(src[voiced], 1e-6) / med)

            target *= np.power(2.0, deviation_cents * mod / 1200.0)


    if voiced_mask is not None:

        mask = np.asarray(voiced_mask, dtype=np.float32)

        if len(mask) != len(target):

            x0 = np.linspace(0.0, 1.0, len(mask), dtype=np.float32)

            x1 = np.linspace(0.0, 1.0, len(target), dtype=np.float32)

            mask = np.interp(x1, x0, mask).astype(np.float32)

        target *= (mask >= 0.5).astype(np.float32)


    return target.astype(np.float32), cents, base_midi, base_hz


def derive_voiced_mask(f0):

    arr = np.asarray(f0, dtype=np.float32)

    valid = np.flatnonzero(arr > 1.0)

    mask = np.zeros(len(arr), dtype=np.float32)

    if len(valid) == 0:

        return mask, 0, 0

    first, last = int(valid[0]), int(valid[-1])

    mask[first:last + 1] = 1.0

    return mask, first, last


def rms(x):

    x = np.asarray(x, dtype=np.float64)

    return float(np.sqrt(np.mean(x * x) + 1e-12)) if x.size else 0.0


def _shift_with_zeros(x, shift):

    x = np.asarray(x, dtype=np.float32)

    if shift == 0:

        return x

    y = np.zeros_like(x)

    if shift > 0:

        y[shift:] = x[:-shift]

    else:

        k = -shift

        y[:-k] = x[k:]

    return y


def phase_aware_align(original, generated, sr, boundary_samples, target_f0_hz):

    if target_f0_hz <= 20.0 or boundary_samples < 8:

        return generated, 0.0

    max_by_period = int(round(sr / max(target_f0_hz, 1.0) * 0.5))

    max_shift = max(1, min(int(round(0.0015 * sr)), max_by_period))

    win = int(round(min(0.025, max(0.010, 3.0 / target_f0_hz)) * sr))

    start = max(0, boundary_samples - win // 4)

    stop = min(len(original), len(generated), start + win)

    if stop - start < 64:

        return generated, 0.0

    ref = np.asarray(original[start:stop], dtype=np.float64)

    ref = ref - np.mean(ref)

    ref_norm = np.linalg.norm(ref) + 1e-9

    best_shift, best_score = 0, -1e9

    for shift in range(-max_shift, max_shift + 1):

        gs, ge = start + shift, stop + shift

        if gs < 0 or ge > len(generated):

            continue

        cand = np.asarray(generated[gs:ge], dtype=np.float64)

        cand = cand - np.mean(cand)

        score = float(np.dot(ref, cand) / (ref_norm * (np.linalg.norm(cand) + 1e-9)))

        if score > best_score:

            best_score, best_shift = score, shift

    if best_score < 0.12 or best_shift == 0:

        return generated, 0.0

    aligned = _shift_with_zeros(generated, -best_shift)

    return aligned, float(-best_shift * 1000.0 / sr)


def hybrid_mix(original, generated, sr, consonant_ms, transition_ms, target_f0_hz=0.0):

    n = len(generated)

    canvas = np.zeros(n, dtype=np.float32)

    copy_n = min(n, len(original))

    canvas[:copy_n] = original[:copy_n]

    c_end = int(np.clip(round(consonant_ms * sr / 1000.0), 0, n))

    t_end = int(np.clip(round((consonant_ms + transition_ms) * sr / 1000.0), c_end, n))


    gen, phase_shift_ms = phase_aware_align(original, generated, sr, c_end, float(target_f0_hz))


    match_start = max(0, c_end)

    match_end = min(n, match_start + int(0.09 * sr))

    gain = 1.0

    if match_end > match_start + 32 and match_start < len(original):

        src_end = min(len(original), match_end)

        ro = rms(original[match_start:src_end])

        rg = rms(gen[match_start:match_end])

        if ro > 1e-5 and rg > 1e-7:

            gain = float(np.clip(ro / rg, 10 ** (-3 / 20), 10 ** (3 / 20)))

    gen = gen * gain


    w = np.zeros(n, dtype=np.float32)

    w[:c_end] = 1.0

    if t_end > c_end:

        x = np.linspace(0.0, 1.0, t_end - c_end, endpoint=True, dtype=np.float32)

        w[c_end:t_end] = 0.5 * (1.0 + np.cos(np.pi * x))

    mixed = (canvas * w + gen * (1.0 - w)).astype(np.float32)

    return mixed, phase_shift_ms, gain


def articulation_hybrid_mix(original, generated, sr, source_f0, target_f0, regions, source_fixed_ms, target_fixed_ms, target_ms, canonical_template=None):

    mixed, stats = single_source_articulation_hybrid(

        original, generated, sr, regions, source_fixed_ms, target_fixed_ms, target_ms, canonical_template=canonical_template

    )

    return mixed, stats


def deterministic_decode(decoder, f0, z, seed, adapter=None, detail=None, prototype_index=None, timbre_shift_semitones=0.0, detail_strength=1.0):

    torch.manual_seed(int(seed))

    with torch.inference_mode():

        wav = decoder(

            f0, z, adapter=adapter, detail=detail, prototype_index=prototype_index,

            timbre_shift_semitones=timbre_shift_semitones, detail_strength=detail_strength,

        )

    return wav[0, 0].detach().cpu().numpy().astype(np.float32)


def stable_seed(*parts):

    h = hashlib.sha1("|".join(str(x) for x in parts).encode("utf-8", "replace")).digest()

    return int.from_bytes(h[:4], "little")


def resample_exact(audio, orig_sr, out_sr, samples):

    y = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=orig_sr, target_sr=out_sr).astype(np.float32)

    samples = max(1, int(samples))

    if len(y) < samples:

        y = np.pad(y, (0, samples - len(y)))

    else:

        y = y[:samples]

    return y


def write_wav(path, audio, sr, volume=100.0):

    y = np.asarray(audio, dtype=np.float32) * (float(volume) / 100.0)

    y = np.nan_to_num(y)

    peak = float(np.max(np.abs(y))) if y.size else 0.0

    if peak > 0.98:

        y *= 0.95 / peak

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    sf.write(path, y, sr, subtype="PCM_16")


class YuazDDSPResamplerEngine:

    def __init__(self, repo, checkpoint, transition_ms=70.0, use_rvq=False, output_sr=44100, registry_path=None):

        self.repo = Path(repo).expanduser().resolve()

        self.checkpoint = Path(checkpoint).expanduser().resolve()

        self.transition_ms = float(transition_ms)

        self.use_rvq = bool(use_rvq)

        self.output_sr = int(output_sr)

        self.device = torch.device("cpu")

        self.registry_path = Path(registry_path).expanduser().resolve() if registry_path else None

        self._registry_cache = None

        self._registry_mtime = None

        self._adapter_cache = {}

        self._refiner_cache = {}

        self._canonical_articulation_cache = {}

        self._highband_profile_cache = {}

        config = load_config(self.repo)

        self.encoder, self.decoder, self.quantizer, self.sr, self.hop = build_modules(self.repo, config, self.device)

        _, state = load_checkpoint(self.checkpoint)

        enc_ratio = load_component(self.encoder, state, "encoder")

        dec_ratio = load_component(self.decoder, state, "ddsp_decoder")

        rvq_ratio = load_component(self.quantizer, state, "rvq")

        if enc_ratio < 0.95:

            raise RuntimeError(f"Encoder 权重不完整: {enc_ratio:.1%}")

        if dec_ratio < 0.80:

            raise RuntimeError(f"DDSP 权重不完整: {dec_ratio:.1%}")

        self.rvq_available = rvq_ratio >= 0.80

        self.analysis_cache = {}

        self.render_lock = threading.Lock()


    def _load_registry(self):

        if self.registry_path is None or not self.registry_path.exists():

            return {"samples": {}}

        mtime = self.registry_path.stat().st_mtime_ns

        if self._registry_cache is not None and self._registry_mtime == mtime:

            return self._registry_cache

        try:

            self._registry_cache = json.loads(self.registry_path.read_text(encoding="utf-8"))

        except Exception:

            self._registry_cache = {"samples": {}}

        self._registry_mtime = mtime

        return self._registry_cache


    def _models_for_input(self, path):

        registry = self._load_registry()

        samples = registry.get("samples", {}) if isinstance(registry, dict) else {}

        record = None

        try:

            sha = file_sha256(path)

            record = samples.get("sha256:" + sha)

        except Exception:

            record = None

        if record is None:

            try:

                pcm = pcm_fingerprint(path)

                record = samples.get("pcm:" + pcm)

            except Exception:

                record = None

        if not record:

            return None, None, None


        adapter = None

        adapter_path = record.get("adapter")

        if adapter_path:

            adapter_path = str(Path(adapter_path).expanduser())

            if Path(adapter_path).exists():

                mtime = Path(adapter_path).stat().st_mtime_ns

                cached = self._adapter_cache.get(adapter_path)

                if cached and cached[0] == mtime:

                    adapter = cached[1]

                else:

                    adapter, metadata = load_adapter(adapter_path, device=self.device)

                    self._adapter_cache[adapter_path] = (mtime, adapter, metadata)


        refiner = None

        refiner_path = record.get("refiner")

        if refiner_path:

            refiner_path = str(Path(refiner_path).expanduser())

            if Path(refiner_path).exists():

                mtime = Path(refiner_path).stat().st_mtime_ns

                cached = self._refiner_cache.get(refiner_path)

                if cached and cached[0] == mtime:

                    refiner = cached[1]

                else:

                    refiner, metadata = load_refiner(refiner_path, device=self.device)

                    self._refiner_cache[refiner_path] = (mtime, refiner, metadata)

        return adapter, refiner, record


    def _request_variant(self, record, req):

        variants = record.get("source_loudness_variants") or []

        if not variants:

            return None

        signature = oto_loudness_signature(req.get("offset", 0.0), req.get("consonant", 0.0), req.get("cutoff", 0.0))

        for variant in variants:

            if variant.get("signature") == signature:

                return variant

        target = np.asarray([float(req.get("offset", 0.0)), float(req.get("consonant", 0.0)), float(req.get("cutoff", 0.0))], dtype=np.float64)

        best = None

        best_distance = None

        for variant in variants:

            values = np.asarray([

                float(variant.get("offset", 0.0)),

                float(variant.get("consonant", 0.0)),

                float(variant.get("cutoff", 0.0)),

            ], dtype=np.float64)

            distance = float(np.sum(np.abs(values - target)))

            if best_distance is None or distance < best_distance:

                best = variant

                best_distance = distance

        return best


    def _canonical_articulation_for_variant(self, record, variant):

        if not record or not variant:

            return None

        rel = variant.get("canonical_articulation")

        if not rel:

            return None

        path = Path(str(rel)).expanduser()

        if not path.is_absolute():

            root = record.get("voicebank_root")

            if not root:

                return None

            path = Path(root).expanduser() / path

        if not path.exists():

            return None

        key = str(path.resolve())

        mtime = path.stat().st_mtime_ns

        cached = self._canonical_articulation_cache.get(key)

        if cached and cached[0] == mtime:

            return cached[1]

        try:

            template = load_canonical_articulation(path)

        except Exception:

            return None

        self._canonical_articulation_cache[key] = (mtime, template)

        if len(self._canonical_articulation_cache) > 1024:

            self._canonical_articulation_cache.pop(next(iter(self._canonical_articulation_cache)))

        return template


    def _highband_db(self, record):

        if not record:

            return None

        path = record.get("highband_profiles")

        if not path:

            return None

        path = str(Path(path).expanduser())

        p = Path(path)

        if not p.exists():

            return None

        mtime = p.stat().st_mtime_ns

        cached = self._highband_profile_cache.get(path)

        if cached and cached[0] == mtime:

            return cached[1]

        try:

            db = load_profile_database(p)

        except Exception:

            return None

        self._highband_profile_cache[path] = (mtime, db)

        if len(self._highband_profile_cache) > 32:

            self._highband_profile_cache.pop(next(iter(self._highband_profile_cache)))

        return db


    def _learned_highband_profile(self, record, variant, target_f0, timbre_shift_semitones, prototype_index):

        db = self._highband_db(record)

        if not db:

            return None

        base_alias = None

        if variant:

            base_alias = variant.get("base_alias") or variant.get("alias")

        if not base_alias and record:

            base_alias = record.get("base_alias")

        voiced = np.asarray(target_f0, dtype=np.float32)

        voiced = voiced[voiced > 1.0]

        if voiced.size:

            target_midi = 69.0 + 12.0 * math.log2(float(np.median(voiced)) / 440.0)

        else:

            target_midi = float((variant or {}).get("subbank_anchor_midi") or (record or {}).get("subbank_anchor_midi") or 60.0)

        return select_learned_profile(

            db, base_alias, target_midi,

            timbre_shift_semitones=float(timbre_shift_semitones),

            source_prototype_index=prototype_index,

        )


    def _analysis_key(self, req):

        p = Path(req["input"]).resolve()

        st = p.stat()

        return (str(p), st.st_size, st.st_mtime_ns, float(req["offset"]), float(req["cutoff"]), float(req["velocity"]))


    def analyze(self, req):

        key = self._analysis_key(req)

        if key in self.analysis_cache:

            return self.analysis_cache[key]

        audio = read_audio(req["input"], self.sr)

        audio = crop_oto(audio, self.sr, req["offset"], req["cutoff"])

        if len(audio) < int(0.08 * self.sr):

            raise RuntimeError("裁切后的样本太短")

        f0 = extract_f0(audio, self.sr, self.hop)

        f0_t = torch.from_numpy(f0).float().view(1, 1, -1)

        audio_t = torch.from_numpy(audio).float().view(1, 1, -1)

        with torch.inference_mode():

            z1, f0_aligned = self.encoder(audio_t, f0_override=f0_t)

            latent = z1

            if self.use_rvq and self.rvq_available:

                latent, _, _ = self.quantizer(z1)

        detail = torch.from_numpy(extract_detail_features(audio, self.sr, self.hop)).float().unsqueeze(0)

        result = {"audio": audio, "latent": latent.cpu(), "f0": f0_aligned.cpu(), "detail": detail.cpu()}

        self.analysis_cache[key] = result

        if len(self.analysis_cache) > 512:

            self.analysis_cache.pop(next(iter(self.analysis_cache)))

        return result


    def render(self, req):

        with self.render_lock:

            a = self.analyze(req)

            audio = a["audio"]

            latent = a["latent"].to(self.device)

            source_f0_t = a["f0"].to(self.device)

            detail = a["detail"].to(self.device)


            velocity = max(1.0, float(req["velocity"]))

            stretch_ratio = 2.0 ** (1.0 - velocity * 0.01)


            source_fixed_ms = max(0.0, float(req["consonant"]))

            fixed_region_ms = max(0.0, source_fixed_ms * stretch_ratio)

            target_ms = max(50.0, float(req["length"]))

            target_frames = max(4, int(round(target_ms * self.sr / (1000.0 * self.hop))))

            source_fixed_frames = int(round(source_fixed_ms * self.sr / (1000.0 * self.hop)))

            target_fixed_frames = int(round(fixed_region_ms * self.sr / (1000.0 * self.hop)))


            source_f0_raw = source_f0_t[0, 0].cpu().numpy().astype(np.float32)

            source_detail = detail[0].cpu().numpy().astype(np.float32)

            articulation = analyze_articulation_regions(audio, self.sr, self.hop, source_f0_raw, source_detail, source_fixed_ms)


            latent_warp = piecewise_warp(latent, source_fixed_frames, target_frames, target_fixed_frames=target_fixed_frames)

            f0_warp = piecewise_warp(source_f0_t, source_fixed_frames, target_frames, target_fixed_frames=target_fixed_frames)

            detail_warp = piecewise_warp(detail, source_fixed_frames, target_frames, target_fixed_frames=target_fixed_frames)

            src_f0_raw = f0_warp[0, 0].numpy()

            voiced_mask, first_voiced_frame, last_voiced_frame = derive_voiced_mask(src_f0_raw)

            src_f0 = fill_f0(src_f0_raw, first_voiced_frame)


            target_f0, cents, base_midi, base_hz = build_target_f0(

                req["tone"], req.get("pitch", "AA"), req.get("tempo", "!120"), target_frames,

                self.sr, self.hop, src_f0, req.get("modulation", 0), voiced_mask=voiced_mask,

            )

            f0_t = torch.from_numpy(target_f0).float().view(1, 1, -1).to(self.device)

            adapter, refiner, bank_record = self._models_for_input(req["input"])

            variant = self._request_variant(bank_record, req) if bank_record else None

            seed = stable_seed(req["input"], req["tone"], req.get("pitch", ""), req["length"], req["consonant"], req["offset"], req["cutoff"], bank_record.get("voicebank_id") if bank_record else "base")

            prototype_index = variant.get("subbank_index") if variant and variant.get("subbank_index") is not None else (bank_record.get("subbank_index") if bank_record else None)

            controls = parse_yuaz_controls(req.get("flags", ""))

            timbre_shift_semitones = controls.timbre_shift_semitones

            detail_strength = controls.detail_strength

            canonical_template = self._canonical_articulation_for_variant(bank_record, variant)

            generated = deterministic_decode(

                self.decoder, f0_t, latent_warp, seed, adapter=adapter, detail=detail_warp, prototype_index=prototype_index,

                timbre_shift_semitones=timbre_shift_semitones, detail_strength=detail_strength,

            )

            source_total_ms = len(audio) * 1000.0 / float(self.sr)

            mapped_articulation = map_articulation_regions(

                articulation, source_fixed_ms, fixed_region_ms, source_total_ms, target_ms

            )

            target_articulation_end_sample = int(np.clip(

                round(mapped_articulation["target_articulation_end_ms"] * self.sr / 1000.0), 0, len(generated)

            ))

            fidelity_residual_rms = 0.0

            if refiner is not None:

                generated_t = torch.from_numpy(generated).float().view(1, 1, -1).to(self.device)

                with torch.inference_mode():

                    refined_t, residual_t = refiner(

                        generated_t, detail_warp, f0_t, articulation_end_sample=target_articulation_end_sample

                    )

                if float(detail_strength) == 1.0:

                    output_t = refined_t

                    applied_residual_t = residual_t

                else:

                    applied_residual_t = residual_t * float(detail_strength)

                    output_t = torch.clamp(generated_t + applied_residual_t, -1.2, 1.2)

                generated = output_t[0, 0].detach().cpu().numpy().astype(np.float32)

                fidelity_residual_rms = float(torch.sqrt(torch.mean(applied_residual_t.pow(2)) + 1e-12).cpu())


            hybrid, articulation_stats = articulation_hybrid_mix(

                audio, generated, self.sr, source_f0_raw, target_f0, articulation,

                source_fixed_ms, fixed_region_ms, target_ms, canonical_template=canonical_template,

            )


            exact_out_samples = int(round(target_ms * self.output_sr / 1000.0))

            final = resample_exact(hybrid, self.sr, self.output_sr, exact_out_samples)

            highband_stats = {

                "used": False,

                "assist_start_hz": float(controls.highband_yuaz_only_hz),

                "profile_found": False,

            }

            if controls.highband_enabled and self.output_sr > 24000:

                hb_profile = self._learned_highband_profile(

                    bank_record, variant, target_f0, timbre_shift_semitones, prototype_index

                )

                if hb_profile is not None:

                    highband_stats["profile_found"] = True

                    final, hb_stats = synthesize_learned_highband(

                        final, self.output_sr, target_f0, hb_profile,

                        seed=stable_seed(seed, "learned-highband"),

                        assist_start_hz=controls.highband_yuaz_only_hz,

                        detail_strength=detail_strength,

                        target_fixed_ms=fixed_region_ms,

                    )

                    highband_stats.update(hb_stats)

                    highband_stats["ym_profile_weights"] = hb_profile.get("weights", [])

            loudness_stats = {

                "used": False,

                "target_dbfs": None,

                "before_active_rms_dbfs": None,

                "after_active_rms_dbfs": None,

                "gain_db": 0.0,

                "target_error_db": None,

                "peak_guard_samples": 0,

                "target_reached": False,

                "safety_limited": False,

            }

            if bank_record and bool(bank_record.get("loudness_enabled", False)):

                final, loudness_stats = normalize_final_render(

                    final,

                    self.output_sr,

                    target_dbfs=float(bank_record.get("loudness_target_dbfs", -18.0)),

                    peak_ceiling_dbfs=float(bank_record.get("loudness_peak_ceiling_dbfs", -1.0)),

                    peak_guard_knee_db=float(bank_record.get("loudness_peak_guard_knee_db", 3.0)),

                    emergency_max_abs_gain_db=float(bank_record.get("loudness_emergency_max_abs_gain_db", 30.0)),

                    tolerance_db=float(bank_record.get("loudness_tolerance_db", 0.05)),

                )

            write_wav(req["output"], final, self.output_sr, req.get("volume", 100))

            return {

                "ok": True,

                "output": req["output"],

                "source_sr": self.sr,

                "output_sr": self.output_sr,

                "target_ms": target_ms,

                "utau_source_fixed_region_ms": source_fixed_ms,

                "utau_fixed_region_ms": fixed_region_ms,

                "raw_preserve_ms": float(articulation_stats["target_onset_ms"]),

                "transition_ms": float(articulation_stats["target_transition_end_ms"] - articulation_stats["target_onset_ms"]),

                "articulation_end_ms": float(articulation_stats["target_articulation_end_ms"]),

                "articulation_psola_used": False,

                "articulation_trajectory_transfer_used": bool(articulation_stats.get("trajectory_transfer_used", False)),

                "single_periodic_source": bool(articulation_stats.get("single_periodic_source", True)),

                "articulation_trajectory_gain_rms_db": float(articulation_stats.get("trajectory_gain_rms_db", 0.0)),

                "articulation_trajectory_source": articulation_stats.get("trajectory_source"),

                "canonical_articulation_used": bool(articulation_stats.get("trajectory_source") == "canonical"),

                "canonical_articulation_coherence": float(articulation_stats.get("canonical_coherence", 0.0)),

                "articulation_base_alias": variant.get("base_alias") if variant else (bank_record.get("base_alias") if bank_record else None),

                "phase_shift_ms": float(articulation_stats.get("phase_shift_ms", 0.0)),

                "hybrid_gain": float(articulation_stats["hybrid_gain"]),

                "tone": req["tone"],

                "base_midi": int(base_midi),

                "base_hz": float(base_hz),

                "target_f0_median": float(np.median(target_f0[target_f0 > 0])) if np.any(target_f0 > 0) else 0.0,

                "target_f0_min": float(np.min(target_f0[target_f0 > 0])) if np.any(target_f0 > 0) else 0.0,

                "target_f0_max": float(np.max(target_f0[target_f0 > 0])) if np.any(target_f0 > 0) else 0.0,

                "pitch_points": int(len(decode_int12_pitch(req.get("pitch", "AA")))),

                "cache_entries": len(self.analysis_cache),

                "voicebank_adapter": bool(adapter is not None),

                "fidelity_refiner_used": bool(refiner is not None),

                "fidelity_residual_rms": float(fidelity_residual_rms),

                "voicebank_id": bank_record.get("voicebank_id") if bank_record else None,

                "utau_subbank_index": prototype_index,

                "utau_subbank_label": variant.get("subbank_label") if variant else (bank_record.get("subbank_label") if bank_record else None),

                "yuaz_timbre_morph": float(controls.timbre_morph),

                "yuaz_timbre_shift_semitones": float(timbre_shift_semitones),

                "yuaz_learned_detail": float(controls.learned_detail),

                "yuaz_learned_detail_strength": float(detail_strength),

                "yuaz_highband_assist_start": float(controls.highband_crossover),

                "yuaz_highband_assist_start_hz": float(controls.highband_yuaz_only_hz),

                "learned_highband_used": bool(highband_stats.get("used", False)),

                "learned_highband_profile_found": bool(highband_stats.get("profile_found", False)),

                "learned_highband_rms": float(highband_stats.get("branch_rms", 0.0)),

                "learned_highband_safety_gain": float(highband_stats.get("safety_gain", 1.0)),

                "learned_highband_harmonic_count": int(highband_stats.get("harmonic_count", 0)),

                "learned_highband_harmonic_mix": float(highband_stats.get("voiced_harmonic_mix", 0.0)),

                "learned_highband_temporal_used": bool(highband_stats.get("temporal_used", False)),

                "learned_highband_temporal_upper_peak_gain": float(highband_stats.get("temporal_upper_peak_gain", 1.0)),

                "learned_highband_temporal_low_peak_gain": float(highband_stats.get("temporal_low_peak_gain", 1.0)),

                "learned_highband_profile_weights": highband_stats.get("ym_profile_weights", []),

                "loudness_normalization_used": bool(loudness_stats.get("used", False)),

                "loudness_target_active_rms_dbfs": loudness_stats.get("target_dbfs"),

                "loudness_before_active_rms_dbfs": loudness_stats.get("before_active_rms_dbfs"),

                "loudness_gain_db": float(loudness_stats.get("gain_db", 0.0)),

                "loudness_final_active_rms_dbfs": loudness_stats.get("after_active_rms_dbfs"),

                "loudness_target_error_db": loudness_stats.get("target_error_db"),

                "loudness_peak_guard_samples": int(loudness_stats.get("peak_guard_samples", 0)),

                "loudness_target_reached": bool(loudness_stats.get("target_reached", False)),

                "loudness_safety_limited": bool(loudness_stats.get("safety_limited", False)),

            }

