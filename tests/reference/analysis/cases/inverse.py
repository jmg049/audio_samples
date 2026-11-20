"""References for ``librosa.feature.inverse``.

``mel_to_stft``: librosa minimises by L-BFGS-B from the projected pseudo-inverse and
stops early, and the problem has many solutions, so the fixture stores its result and
objective: the check is on the objective and the fit, not on each unknown.

``mfcc_to_mel``: a closed form, checked directly across DCT types, norms and lifters.

``mel_to_audio``: librosa's Griffin-Lim starts from random phases, so the fixture runs
``griffinlim(init=None)`` on the magnitude and stores the mel spectrogram of the result.
"""

from __future__ import annotations

import numpy as np

import librosa

NAME = "inverse"


def _l(a):
    return np.asarray(a).tolist()


def build(signals, sr) -> dict:
    out: dict = {}
    y = signals["mix"][:8192].astype(np.float64)

    cases = {}
    for name, kw in {
        "default": dict(n_fft=512, n_mels=40, power=2.0),
        "power1": dict(n_fft=512, n_mels=40, power=1.0),
        "htk": dict(n_fft=1024, n_mels=32, power=2.0, htk=True, fmin=50.0, fmax=8000.0),
    }.items():
        mel_kw = {k: v for k, v in kw.items() if k not in ("n_fft", "power")}
        M = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=kw["n_fft"], hop_length=256, power=kw["power"], dtype=np.float64, **mel_kw
        )
        M = M[:, :12]
        s = librosa.feature.inverse.mel_to_stft(
            M, sr=sr, n_fft=kw["n_fft"], power=kw["power"], **{k: v for k, v in mel_kw.items() if k != "n_mels"}
        )
        basis = librosa.filters.mel(sr=sr, n_fft=kw["n_fft"], dtype=np.float64, **mel_kw)
        cases[name] = {
            "M": _l(M),
            "kw": kw,
            "objective": float(0.5 * np.sum((basis @ s ** kw["power"] - M) ** 2)),
            "energy": float(np.sum(s**2)),
        }
    out["mel_to_stft"] = cases

    M = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=128, n_mels=40, dtype=np.float64)
    s = librosa.feature.inverse.mel_to_stft(M, sr=sr, n_fft=512)
    a = librosa.griffinlim(s, n_iter=8, hop_length=128, n_fft=512, init=None, length=8192, dtype=np.float64)
    out["mel_to_audio"] = {
        "M": _l(M),
        "n_fft": 512,
        "hop_length": 128,
        "n_iter": 8,
        "mel_of_audio": _l(
            librosa.feature.melspectrogram(y=a, sr=sr, n_fft=512, hop_length=128, n_mels=40, dtype=np.float64)
        ),
    }

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256, n_mels=40, dtype=np.float64)[:, :10]
    cases = {}
    for name, kw in {
        "default": dict(n_mels=40),
        "lifter": dict(n_mels=40, lifter=22.0),
        "ref": dict(n_mels=40, ref=2.0),
        "dct2_none": dict(n_mels=40, norm=None),
        "dct1_ortho": dict(n_mels=40, dct_type=1),
        "dct1_none": dict(n_mels=40, dct_type=1, norm=None),
        "dct3_ortho": dict(n_mels=40, dct_type=3),
        "dct3_none": dict(n_mels=40, dct_type=3, norm=None),
        "fewer_bands": dict(n_mels=8),
    }.items():
        # Unnormalised transforms scale the log values: keep the result finite.
        scale = 1e-3 if kw.get("norm", "ortho") is None else 1.0
        cases[name] = {
            "kw": kw,
            "scale": scale,
            "mel": _l(librosa.feature.inverse.mfcc_to_mel(mfcc * scale, **kw)),
        }
    out["mfcc_to_mel"] = {"mfcc": _l(mfcc), "cases": cases}
    return out
