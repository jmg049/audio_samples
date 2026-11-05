"""References for ``audio_samples.operations.linalg``.

* ``nnls`` on a vector: ``librosa.util.nnls`` delegates to ``scipy.optimize.nnls``,
  whose active-set solution is unique for a full-rank matrix with more rows than
  columns.
* ``nnls`` on a matrix: ``librosa.util.nnls`` runs L-BFGS-B from the projected
  pseudo-inverse with SciPy's default stopping rule, so it stops short of the optimum.
  The fixture stores its solution and its objective for an upper bound.
* ``pinv``, ``eigh`` and ``svd`` come from SciPy and NumPy (LAPACK) as the reference.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

import librosa

NAME = "linalg"


def _l(a):
    return np.asarray(a).tolist()


def build(signals, sr) -> dict:
    rng = np.random.default_rng(20261105)
    out: dict = {}

    a = rng.normal(size=(30, 12))
    b = rng.normal(size=30)
    out["nnls_vector"] = {"a": _l(a), "b": _l(b), "x": _l(librosa.util.nnls(a, b))}

    pos = np.abs(rng.normal(size=(20, 8)))
    target = pos @ np.abs(rng.normal(size=8))
    out["nnls_vector_exact"] = {"a": _l(pos), "b": _l(target), "x": _l(librosa.util.nnls(pos, target))}

    mel = librosa.filters.mel(sr=sr, n_fft=512, n_mels=32, dtype=np.float64)
    gains = np.abs(rng.normal(size=(mel.shape[1], 6))) ** 2
    m = mel @ gains
    x = librosa.util.nnls(mel, m)
    out["nnls_matrix"] = {
        "b": _l(m),
        "n_fft": 512,
        "n_mels": 32,
        "x": _l(x),
        "objective": float(0.5 * np.sum((mel @ x - m) ** 2)),
    }

    w = rng.normal(size=(9, 20))
    out["pinv"] = {"a": _l(w), "p": _l(scipy.linalg.pinv(w, rtol=1e-15))}
    r = rng.normal(size=(12, 12))
    sym = r + r.T
    vals, vecs = np.linalg.eigh(sym)
    out["eigh"] = {"a": _l(sym), "w": _l(vals), "v": _l(vecs)}
    t = rng.normal(size=(15, 7))
    u, s, vt = np.linalg.svd(t, full_matrices=False)
    out["svd"] = {"a": _l(t), "s": _l(s), "u": _l(u), "vt": _l(vt)}
    return out
