"""References for ``librosa.segment`` (``path_enhance``, ``agglomerative``, ``subsegment``,
``timelag_filter``), ``filters.diagonal_filter`` and ``sequence.path_to_steps``.
``agglomerative`` uses scikit-learn's ward tree with a chain connectivity.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage

import librosa

NAME = "segment"


def _l(a):
    return np.asarray(a).tolist()


def build(signals, sr) -> dict:
    rng = np.random.default_rng(20261106)
    out: dict = {}

    kernels: dict = {}
    for name, (win, n, kw) in {
        "hann_diag": ("hann", 9, {}),
        "hann_slope2": ("hann", 9, {"slope": 2.0}),
        "hann_slope_half": ("hann", 11, {"slope": 0.5}),
        "boxcar_angle": ("boxcar", 7, {"angle": 0.3}),
        "hamming_zero_mean": ("hamming", 8, {"slope": 1.5, "zero_mean": True}),
        "kaiser": (("kaiser", 5.0), 9, {"slope": 1.26}),
    }.items():
        kernels[name] = {"window": win if isinstance(win, str) else list(win), "n": n, "kw": kw, "k": _l(librosa.filters.diagonal_filter(win, n, **kw))}
    out["diagonal_filter"] = kernels

    # A recurrence-like matrix: diagonal stripes plus noise.
    N = 40
    R = rng.uniform(0, 0.2, size=(N, N))
    for offset in (0, 10):
        R += np.eye(N, k=offset) + np.eye(N, k=-offset)
    enh: dict = {"R": _l(R)}
    for name, kw in {
        "default": dict(n=7),
        "wide": dict(n=11, max_ratio=1.5, n_filters=3),
        "minmax": dict(n=5, max_ratio=2.0, min_ratio=1.0, n_filters=2),
        "zero_mean": dict(n=7, zero_mean=True),
        "no_clip": dict(n=7, zero_mean=True, clip=False),
        "constant": dict(n=7, mode="constant", cval=0.25),
        "nearest": dict(n=7, mode="nearest"),
        "mirror": dict(n=7, mode="mirror"),
        "wrap": dict(n=7, mode="wrap"),
        "window": dict(n=7, window="boxcar"),
        "origin": dict(n=7, n_filters=3, origin=1),
    }.items():
        args = dict(kw)
        n = args.pop("n")
        enh[name] = {"kw": kw, "out": _l(librosa.segment.path_enhance(R, n, **args))}
    out["path_enhance"] = enh

    x = rng.normal(size=(6, 80))
    x[:, 20:45] += 4.0
    x[:, 60:] -= 3.0
    ag: dict = {"x": _l(x)}
    for k in (1, 2, 3, 5, 12, 80):
        ag[str(k)] = _l(librosa.segment.agglomerative(x, k))
    ag["axis0"] = _l(librosa.segment.agglomerative(x.T, 4, axis=0))
    y = x[0]
    ag["one_d"] = _l(librosa.segment.agglomerative(y, 4))
    out["agglomerative"] = ag
    out["subsegment"] = {
        "frames": [10, 40, 70],
        "default": _l(librosa.segment.subsegment(x, np.array([10, 40, 70]))),
        "n2": _l(librosa.segment.subsegment(x, np.array([10, 40, 70]), n_segments=2)),
        "many": _l(librosa.segment.subsegment(x, np.array([3, 5]), n_segments=4)),
    }

    # Timelag filter: a median filter along the time axis of the lag matrix.
    rec = librosa.segment.recurrence_matrix(x, mode="affinity", k=8)
    med = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    filtered = med(rec, size=(1, 3))
    nopad = librosa.segment.timelag_filter(scipy.ndimage.gaussian_filter, pad=False)
    out["timelag"] = {
        "rec": _l(rec),
        "median": _l(filtered),
        "lag_nopad_sum": float(np.sum(nopad(rec, sigma=1.0))),
    }

    c = rng.normal(size=(5, 60))
    d = rng.normal(size=(5, 45))
    _, wp = librosa.sequence.dtw(c, d)
    out["path_to_steps"] = {
        "path": _l(wp),
        "forward": _l(librosa.sequence.path_to_steps(wp)),
        "inverse": _l(librosa.sequence.path_to_steps(wp, inverse=True)),
        "gap_path": [[0, 0], [2, 4], [2, 5], [9, 7]],
        "gap_forward": _l(librosa.sequence.path_to_steps(np.array([[0, 0], [2, 4], [2, 5], [9, 7]]))),
        "gap_inverse": _l(librosa.sequence.path_to_steps(np.array([[0, 0], [2, 4], [2, 5], [9, 7]]), inverse=True)),
    }
    return out
