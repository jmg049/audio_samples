"""Resampling and mu-law companding.

``librosa.resample`` with its default ``res_type="soxr_hq"`` delegates to the
soxr C library (python-soxr, pinned in ``pyproject.toml``). The outputs are too
long to store in full, so each case keeps the exact output length plus three
windows: the first and last ``EDGE`` samples (start/end alignment and the
flushed tail) and ``MID`` samples from the middle. As an independent second
reference, ``scipy.signal.resample_poly`` (a different, Kaiser-windowed
polyphase FIR, also delay-free) is stored for the middle window.

The mu-law cases use a short ramp on [-1, 1] and cover both librosa modes:
``quantize=False`` (float companding) and the default ``quantize=True``
(integer codes).
"""

from __future__ import annotations

from fractions import Fraction

import librosa
import numpy as np
import scipy.signal

NAME = "resample"

EDGE = 256
MID = 1024
RATES = [16000, 44100, 11025]
SIGNALS = ["mix", "noise"]


def _windows(y: np.ndarray) -> dict:
    n = y.shape[-1]
    m0 = n // 2 - MID // 2
    return {"head": y[:EDGE], "mid": y[m0 : m0 + MID], "tail": y[n - EDGE :], "mid_start": m0}


def build(signals, sr):
    out: dict = {"edge": EDGE, "mid": MID, "resample": {}, "mu_law": {}}
    for name in SIGNALS:
        x = signals[name].astype(np.float64)
        for target in RATES:
            y = librosa.resample(x, orig_sr=sr, target_sr=target)  # soxr_hq
            frac = Fraction(target, sr)
            poly = scipy.signal.resample_poly(x, frac.numerator, frac.denominator)
            w = _windows(y)
            w["len"] = int(y.shape[-1])
            w["len_exact"] = int(-(-len(x) * target // sr))
            w["scipy_mid"] = poly[w["mid_start"] : w["mid_start"] + MID]
            out["resample"][f"{name}_{target}"] = {"orig_sr": sr, "target_sr": target, **w}

    ramp = np.linspace(-1.0, 1.0, 101)
    for mu in (255, 15):
        comp = librosa.mu_compress(ramp, mu=mu, quantize=False)
        codes = librosa.mu_compress(ramp, mu=mu, quantize=True)
        out["mu_law"][str(mu)] = {
            "mu": float(mu),
            "x": ramp,
            "compress": comp,
            "expand": librosa.mu_expand(comp, mu=mu, quantize=False),
            "codes": codes,
            "decoded": librosa.mu_expand(codes, mu=mu, quantize=True),
        }
    return out
