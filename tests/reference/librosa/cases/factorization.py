"""References for ``librosa.decompose.decompose`` (scikit-learn's ``NMF``).

Inputs are small magnitude spectrograms. The starting point of the SVD inits comes from
a randomised SVD with scikit-learn's own generator, so factors are compared through the
reconstruction error, and the templates through their cosine similarity after matching.
"""

from __future__ import annotations

import numpy as np
import sklearn.decomposition

import librosa

NAME = "factorization"


def _l(a):
    return np.asarray(a).tolist()


def build(signals, sr) -> dict:
    out: dict = {}
    y = signals["mix"][:16384].astype(np.float64)
    S = np.abs(librosa.stft(y, n_fft=256, hop_length=512, dtype=np.complex128))
    out["S"] = _l(S)
    cases = {}
    for name, kw in {
        "cd": dict(n_components=4, random_state=0),
        "cd_nndsvd": dict(n_components=4, init="nndsvd", random_state=0),
        "cd_l1": dict(n_components=4, alpha_W=0.001, l1_ratio=1.0, random_state=0),
        "mu_frobenius": dict(n_components=4, solver="mu", random_state=0),
        "mu_kl": dict(n_components=4, solver="mu", beta_loss="kullback-leibler", random_state=0),
        "mu_is": dict(n_components=4, solver="mu", beta_loss="itakura-saito", max_iter=400, random_state=0),
    }.items():
        model = sklearn.decomposition.NMF(**kw)
        comps, acts = librosa.decompose.decompose(S, transformer=model)
        cases[name] = {
            "kw": {k: v for k, v in kw.items() if k != "random_state"},
            "components": _l(comps),
            "activations": _l(acts),
            "err": float(model.reconstruction_err_),
            "n_iter": int(model.n_iter_),
        }
    out["cases"] = cases

    comps, acts = librosa.decompose.decompose(S, n_components=3, sort=True, random_state=0)
    out["sorted"] = {
        "peaks": _l(np.argmax(comps, axis=0)),
        "err": float(np.linalg.norm(S - comps @ acts)),
    }

    # Fixed templates: the activations are a convex problem, so they agree closely.
    model = sklearn.decomposition.NMF(n_components=4, random_state=0, tol=1e-8, max_iter=2000)
    model.fit_transform(S.T)
    out["transform"] = {
        "H": _l(model.components_),
        "W": _l(model.transform(S.T)),
    }
    return out
