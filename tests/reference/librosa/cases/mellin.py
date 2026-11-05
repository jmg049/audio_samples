"""References for ``librosa.fmt``. The interpolant is ``scipy.interpolate.interp1d`` and
the transform ``scipy.fft.rfft``."""

from __future__ import annotations

import numpy as np

import librosa

NAME = "mellin"


def _c(a):
    a = np.asarray(a)
    return {"re": a.real.tolist(), "im": a.imag.tolist()}


def build(signals, sr) -> dict:
    y = signals["mix"][:2048].astype(np.float64)
    stereo = np.stack([y, y[::-1]])
    out: dict = {"y": y.tolist()}
    cases = {}
    for name, kw in {
        "default": dict(),
        "beta0": dict(beta=0.0),
        "t_min": dict(t_min=2.0),
        "n_fmt": dict(n_fmt=512),
        "n_fmt_over": dict(n_fmt=512, over_sample=2.0),
        "over": dict(over_sample=2.0),
        "linear": dict(kind="linear"),
        "quadratic": dict(kind="quadratic", n_fmt=256),
    }.items():
        r = librosa.fmt(y, **kw)
        cases[name] = {"kw": kw, "out": _c(r[:64]), "n": int(r.size)}
    out["cases"] = cases
    r = librosa.fmt(stereo, n_fmt=128)
    out["stereo"] = _c(r)
    return out
