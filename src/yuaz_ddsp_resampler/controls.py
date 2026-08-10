#!/usr/bin/env python3

import re

from dataclasses import dataclass


_CONTROL_RE = re.compile(r"(YM|YD|YH)([+-]?(?:\d+(?:\.\d*)?|\.\d+))", re.IGNORECASE)


def _clamp(value, lo=-100.0, hi=100.0):

    return max(lo, min(hi, float(value)))


@dataclass(frozen=True)

class YuazControls:

    timbre_morph: float = 0.0

    learned_detail: float = 0.0

    highband_crossover: float = 0.0


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

    def highband_yuaz_only_hz(self):

        if not self.highband_enabled:

            return 12000.0

        return 100.0 * max(80.0, min(120.0, float(self.highband_crossover)))


def parse_yuaz_controls(flags):

    values = {"YM": 0.0, "YD": 0.0, "YH": 0.0}

    for match in _CONTROL_RE.finditer(str(flags or "")):

        key = match.group(1).upper()

        raw = float(match.group(2))

        if key == "YH":

            values[key] = 0.0 if raw == 0.0 else max(80.0, min(120.0, raw))

        else:

            values[key] = _clamp(raw)

    return YuazControls(

        timbre_morph=values["YM"],

        learned_detail=values["YD"],

        highband_crossover=values["YH"],

    )

