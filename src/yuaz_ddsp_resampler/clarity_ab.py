#!/usr/bin/env python3
import gc
import threading

import numpy as np


_state = threading.local()
_patched = False
_original_smoothers = {}


def set_mode(value):
    global _patched
    _state.mode = float(np.clip(float(value), 0.0, 100.0))
    if not _patched:
        _patch_ap_smoothers()


def get_mode():
    return float(getattr(_state, "mode", 0.0))


def _patch_ap_smoothers():
    global _patched
    found = 0
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, type):
                continue
            if obj.__name__ != "AdaptiveDDSPDecoder":
                continue
            if not hasattr(obj, "_smooth_ap_bands"):
                continue
            key = id(obj)
            if key in _original_smoothers:
                continue
            original = getattr(obj, "_smooth_ap_bands")
            _original_smoothers[key] = original

            def wrapper(A_c, _original=original):
                smoothed = _original(A_c)
                mode = get_mode()
                if mode < 25.0:
                    return smoothed
                if mode < 75.0:
                    return 0.35 * smoothed + 0.65 * A_c
                return A_c

            setattr(obj, "_smooth_ap_bands", staticmethod(wrapper))
            found += 1
        except Exception:
            continue
    _patched = found > 0
