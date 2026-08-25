#!/usr/bin/env python3
import re
from dataclasses import dataclass


_CONTROL_RE = re.compile(
    r"(YM|YD|YH|YT|YB|YV|YG|YO|YF|YX|YP|YR|YC)([+-]?(?:\d+(?:\.\d*)?|\.\d+))",
    re.IGNORECASE,
)


def _clamp(value, lo=-100.0, hi=100.0):
    return max(lo, min(hi, float(value)))


@dataclass(frozen=True)
class YuazControls:
    timbre_morph: float = 0.0
    learned_detail: float = 0.0
    highband_crossover: float = 0.0
    tension: float = 0.0
    breathiness: float = 0.0
    voicing: float = 0.0
    gender_formant: float = 0.0
    mouth: float = 0.0
    falsetto: float = 0.0
    mixed_voice: float = 0.0
    pharyngeal: float = 0.0
    raw_bypass: float = 0.0
    clarity_ab: float = 0.0

    @property
    def timbre_shift_semitones(self):
        return 12.0 * (_clamp(self.timbre_morph) / 100.0)

    @property
    def detail_strength(self):
        value = _clamp(self.learned_detail)
        if value <= 0.0:
            return 1.0 + value / 100.0
        return 1.0 + 0.5 * value / 100.0

    @property
    def highband_enabled(self):
        return float(self.highband_crossover) != 0.0

    @property
    def highband_strength(self):
        return max(0.0, min(1.0, float(self.highband_crossover) / 100.0))

    @property
    def highband_yuaz_only_hz(self):
        if not self.highband_enabled:
            return 12000.0
        strength = self.highband_strength
        return 11800.0 - 3000.0 * (strength ** 0.5)

    @property
    def raw_bypass_enabled(self):
        return float(self.raw_bypass) >= 0.5 or float(self.clarity_ab) >= 99.0

    @property
    def vocal_controls_active(self):
        return any(abs(float(v)) > 1e-8 for v in (
            self.tension, self.breathiness, self.voicing,
            self.gender_formant, self.mouth,
            self.falsetto, self.mixed_voice, self.pharyngeal,
        ))

    def frame_controls(self, frames, device, dtype, curves=None):
        import torch
        import torch.nn.functional as F
        from .post_gender import set_gender_amount

        frames = max(1, int(frames))
        set_gender_amount(self.gender_formant)
        aliases = {
            "YT": "tension", "tension": "tension",
            "YB": "breathiness", "breathiness": "breathiness",
            "YV": "voicing", "voicing": "voicing",
            "YG": "gender_formant", "gender": "gender_formant", "gender_formant": "gender_formant",
            "YO": "mouth", "mouth": "mouth",
            "YF": "falsetto", "falsetto": "falsetto",
            "YX": "mixed_voice", "mixed": "mixed_voice", "mixed_voice": "mixed_voice",
            "YP": "pharyngeal", "pharyngeal": "pharyngeal",
        }
        defaults = {
            "tension": self.tension,
            "breathiness": self.breathiness,
            "voicing": self.voicing,
            "gender_formant": self.gender_formant,
            "mouth": self.mouth,
            "falsetto": self.falsetto,
            "mixed_voice": self.mixed_voice,
            "pharyngeal": self.pharyngeal,
        }
        supplied = {}
        if isinstance(curves, dict):
            for key, value in curves.items():
                name = aliases.get(str(key), aliases.get(str(key).upper()))
                if name is not None:
                    supplied[name] = value

        def shape_curve(name, value):
            amount = {
                "gender_formant": 0.08,
                "mouth": 0.22,
            }.get(name, 0.0)
            if amount > 0.0:
                value = value * (1.0 - amount * torch.square(torch.abs(value)))
            return value

        def make_curve(name):
            if name not in supplied:
                value = torch.full((1, 1, frames), _clamp(defaults[name]) / 100.0, device=device, dtype=dtype)
                return shape_curve(name, value)
            value = torch.as_tensor(supplied[name], device=device, dtype=dtype).reshape(1, 1, -1)
            value = torch.clamp(value, -100.0, 100.0) / 100.0
            if value.shape[-1] != frames:
                value = F.interpolate(value, size=frames, mode="linear", align_corners=False)
            return shape_curve(name, value)

        return {name: make_curve(name) for name in defaults}


def parse_yuaz_controls(flags):
    values = {
        "YM": 0.0, "YD": 0.0, "YH": 0.0,
        "YT": 0.0, "YB": 0.0, "YV": 0.0, "YG": 0.0,
        "YO": 0.0, "YF": 0.0, "YX": 0.0, "YP": 0.0, "YR": 0.0,
        "YC": 0.0,
    }
    for match in _CONTROL_RE.finditer(str(flags or "")):
        key = match.group(1).upper()
        raw = float(match.group(2))
        if key == "YH":
            values[key] = max(0.0, min(100.0, raw))
        elif key == "YR":
            values[key] = 1.0 if raw >= 0.5 else 0.0
        elif key == "YC":
            values[key] = max(0.0, min(100.0, raw))
        else:
            values[key] = _clamp(raw)

    from .clarity_ab import set_mode
    set_mode(values["YC"])

    return YuazControls(
        timbre_morph=values["YM"], learned_detail=values["YD"], highband_crossover=values["YH"],
        tension=values["YT"], breathiness=values["YB"], voicing=values["YV"],
        gender_formant=values["YG"], mouth=values["YO"],
        falsetto=values["YF"], mixed_voice=values["YX"], pharyngeal=values["YP"],
        raw_bypass=values["YR"], clarity_ab=values["YC"],
    )
