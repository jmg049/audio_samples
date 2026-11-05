"""librosa.iirt, librosa.fmt, librosa.filters.semitone_filterbank and mr_frequencies.

``fmt`` runs on ``vec`` (VEC_N samples) and on 64 samples (``const``). The ``iirt`` rows use
1 s and 10 s of audio.
"""

import librosa
import numpy as np

from audio_samples import pitch_filterbank as P
from audio_samples import spectral as S

from . import Case, partial, register, vec_values

G = "pitch_filterbank"
SR = 22050
TINY = 64


def _v(c):
    return c.cached("pf_v", lambda: np.exp(-vec_values(0.0, 20.0, c.dtype)))


def _t():
    return np.exp(-np.linspace(0.0, 4.0, TINY))


def pair(name, configs, lib, ours, func, note=""):
    return Case(name, G, configs, librosa=lib, ours=ours, func=func, note=note)


register(
    pair("iirt", "short_mono", lambda c: partial(librosa.iirt, c.y, sr=SR), lambda c: partial(P.iirt, c.y, sr=SR), "iirt"),
    pair("fmt", "vec", lambda c: partial(librosa.fmt, _v(c), n_fmt=16384), lambda c: partial(S.mellin_transform, _v(c), n_fmt=16384), "fmt"),
    pair("fmt_tiny", "const", lambda c: partial(librosa.fmt, _t()), lambda c: partial(S.mellin_transform, _t()), "fmt"),
    pair("mr_frequencies", "const", lambda c: partial(librosa.filters.mr_frequencies, 0.0), lambda c: partial(P.mr_frequencies, 0.0), "filters.mr_frequencies"),
    pair("semitone_filterbank", "const", lambda c: partial(librosa.filters.semitone_filterbank, flayout="sos"), lambda c: partial(P.semitone_filterbank), "filters.semitone_filterbank"),
)
