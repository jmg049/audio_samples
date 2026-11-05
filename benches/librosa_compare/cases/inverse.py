"""librosa.feature.inverse, librosa.util.nnls and librosa.decompose.decompose.

Inputs come from the audio of the ``short_mono`` configurations (1 s and 10 s) and are
computed outside the timed call. ``mel_to_audio`` and ``mfcc_to_audio`` use ``n_iter=8``
on both sides. The ``_tiny`` rows use a 16 x 8 input (``const``). librosa's ``decompose``
needs scikit-learn, which is its own dependency.
"""

import librosa
import numpy as np

from audio_samples import factorization as Z
from audio_samples import inverse as I
from audio_samples import linalg as L

from . import Case, lcg_matrix, partial, register

G = "inverse"
SR = 22050
N_MELS = 64


def _mel(c):
    return c.cached("mel_", lambda: librosa.feature.melspectrogram(y=c.y, sr=SR, n_mels=N_MELS))


def _mfcc(c):
    return c.cached("mfcc_", lambda: librosa.feature.mfcc(y=c.y, sr=SR, n_mels=N_MELS, n_mfcc=20))


def _basis(c):
    return c.cached(
        "basis_", lambda: librosa.filters.mel(sr=SR, n_fft=2048, n_mels=N_MELS, dtype=c.dtype)
    )


def _mag(c):
    return c.cached("mag_", lambda: np.abs(librosa.stft(c.y, n_fft=512, hop_length=256)))


def _tiny_a():
    return lcg_matrix(16, 8, 21)


def _tiny_b():
    return lcg_matrix(16, 4, 22)


def pair(name, configs, lib, ours, func, note=""):
    return Case(name, G, configs, librosa=lib, ours=ours, func=func, note=note)


register(
    pair("mel_to_stft", "short_mono", lambda c: partial(librosa.feature.inverse.mel_to_stft, _mel(c), sr=SR), lambda c: partial(I.mel_to_stft, _mel(c), sr=SR), "feature.inverse.mel_to_stft"),
    pair("mel_to_audio", "short_mono", lambda c: partial(librosa.feature.inverse.mel_to_audio, _mel(c), sr=SR, n_iter=8), lambda c: partial(I.mel_to_audio, _mel(c), sr=SR, n_iter=8), "feature.inverse.mel_to_audio"),
    pair("mfcc_to_mel", "short_mono", lambda c: partial(librosa.feature.inverse.mfcc_to_mel, _mfcc(c), n_mels=N_MELS), lambda c: partial(I.mfcc_to_mel, _mfcc(c), n_mels=N_MELS), "feature.inverse.mfcc_to_mel"),
    pair("mfcc_to_audio", "short_mono", lambda c: partial(librosa.feature.inverse.mfcc_to_audio, _mfcc(c), n_mels=N_MELS, sr=SR, n_iter=8), lambda c: partial(I.mfcc_to_audio, _mfcc(c), n_mels=N_MELS, sr=SR, n_iter=8), "feature.inverse.mfcc_to_audio"),
    pair("nnls_matrix", "short_mono", lambda c: partial(librosa.util.nnls, _basis(c), _mel(c)), lambda c: partial(L.nnls, _basis(c), _mel(c)), "util.nnls"),
    pair("nnls_vector", "const", lambda c: partial(librosa.util.nnls, lcg_matrix(200, 60, 23), lcg_matrix(200, 1, 24)[:, 0]), lambda c: partial(L.nnls, lcg_matrix(200, 60, 23), lcg_matrix(200, 1, 24)[:, 0]), "util.nnls"),
    pair("nnls_vector_tiny", "const", lambda c: partial(librosa.util.nnls, _tiny_a(), _tiny_b()[:, 0]), lambda c: partial(L.nnls, _tiny_a(), _tiny_b()[:, 0]), "util.nnls"),
    pair("nnls_matrix_tiny", "const", lambda c: partial(librosa.util.nnls, _tiny_a(), _tiny_b()), lambda c: partial(L.nnls, _tiny_a(), _tiny_b()), "util.nnls"),
    pair("decompose", "short_mono", lambda c: partial(librosa.decompose.decompose, _mag(c), n_components=8, random_state=0), lambda c: partial(Z.decompose, _mag(c), n_components=8), "decompose.decompose"),
    pair("decompose_tiny", "const", lambda c: partial(librosa.decompose.decompose, _tiny_a(), n_components=3, random_state=0), lambda c: partial(Z.decompose, _tiny_a(), n_components=3), "decompose.decompose"),
)
