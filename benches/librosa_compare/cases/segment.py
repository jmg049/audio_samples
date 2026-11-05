"""librosa.segment, filters.diagonal_filter and sequence.path_to_steps.

Matrices are fixed-size (``seq``, ``const``). ``agglomerative`` needs scikit-learn on the
librosa side. ``timelag_filter`` wraps a filter that returns its input, so the row times
the two lag conversions and the call.
"""

import librosa
import numpy as np

from audio_samples import segment as G_
from audio_samples import sequence as Q

from . import Case, lcg_matrix, partial, register

G = "segment"
N = 300


def _r(c):
    return c.cached("r", lambda: lcg_matrix(N, N, 31, c.dtype))


def _tr():
    return lcg_matrix(12, 12, 31)


def _x(c):
    return c.cached("x", lambda: lcg_matrix(12, 2000, 32, c.dtype))


def _tx():
    return lcg_matrix(4, 24, 32)


def _path(n):
    return np.stack([np.arange(n - 1, -1, -1), np.arange(n - 1, -1, -1) // 2 * 2], axis=1)


def _same(m):
    return m


def _timelag_librosa(r):
    return librosa.segment.timelag_filter(_same)(r)


def _timelag_ours(r):
    return G_.timelag_filter(_same)(r)


def pair(name, configs, lib, ours, func, note=""):
    return Case(name, G, configs, librosa=lib, ours=ours, func=func, note=note)


register(
    pair("diagonal_filter", "const", lambda c: partial(librosa.filters.diagonal_filter, "hann", 51, slope=1.5), lambda c: partial(G_.diagonal_filter, "hann", 51, slope=1.5), "filters.diagonal_filter"),
    pair("diagonal_filter_tiny", "const", lambda c: partial(librosa.filters.diagonal_filter, "hann", 5), lambda c: partial(G_.diagonal_filter, "hann", 5), "filters.diagonal_filter"),
    pair("path_enhance", "seq", lambda c: partial(librosa.segment.path_enhance, _r(c), 15), lambda c: partial(G_.path_enhance, _r(c), 15), "segment.path_enhance"),
    pair("path_enhance_tiny", "const", lambda c: partial(librosa.segment.path_enhance, _tr(), 3, n_filters=2), lambda c: partial(G_.path_enhance, _tr(), 3, n_filters=2), "segment.path_enhance"),
    pair("timelag_filter", "seq", lambda c: partial(_timelag_librosa, _r(c)), lambda c: partial(_timelag_ours, _r(c)), "segment.timelag_filter"),
    pair("timelag_filter_tiny", "const", lambda c: partial(_timelag_librosa, _tr()), lambda c: partial(_timelag_ours, _tr()), "segment.timelag_filter"),
    pair("agglomerative", "seq", lambda c: partial(librosa.segment.agglomerative, _x(c), 20), lambda c: partial(G_.agglomerative, _x(c), 20), "segment.agglomerative"),
    pair("agglomerative_tiny", "const", lambda c: partial(librosa.segment.agglomerative, _tx(), 3), lambda c: partial(G_.agglomerative, _tx(), 3), "segment.agglomerative"),
    pair("subsegment", "seq", lambda c: partial(librosa.segment.subsegment, _x(c), np.arange(0, 2000, 200)), lambda c: partial(G_.subsegment, _x(c), np.arange(0, 2000, 200)), "segment.subsegment"),
    pair("subsegment_tiny", "const", lambda c: partial(librosa.segment.subsegment, _tx(), np.array([8, 16]), n_segments=2), lambda c: partial(G_.subsegment, _tx(), np.array([8, 16]), n_segments=2), "segment.subsegment"),
    pair("path_to_steps", "const", lambda c: partial(librosa.sequence.path_to_steps, _path(20000)), lambda c: partial(Q.path_to_steps, _path(20000)), "sequence.path_to_steps"),
    pair("path_to_steps_tiny", "const", lambda c: partial(librosa.sequence.path_to_steps, _path(16)), lambda c: partial(Q.path_to_steps, _path(16)), "sequence.path_to_steps"),
)
