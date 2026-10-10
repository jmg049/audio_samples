"""librosa constant-Q functions: cqt / vqt and their inverses, CQT chroma
features and the wavelet filterbank.

Skipped rows:

* ``feature.chroma_cqt(C=...)``, ``feature.chroma_cens(C=...)``: no
  precomputed-CQT form in audio_samples (``chroma_cens_from_chroma`` takes a
  chromagram, which librosa has no public function for).
"""

import librosa
import numpy as np

import audio_samples as aus
from audio_samples import features as F

from . import Case, partial, register

G = "constant_q"

C1 = librosa.note_to_hz("C1")


def _cqt_freqs():
    """``cqt_frequencies(84, fmin=C1)``: the default CQT's centre frequencies."""
    return librosa.cqt_frequencies(n_bins=84, fmin=C1)


def _cqt_mag(c):
    return c.cached("C_mag", lambda: np.abs(c.C))


def _chroma_cqt(c):
    return c.cached("chroma_cqt", lambda: librosa.feature.chroma_cqt(y=c.y, sr=c.sr))


register(
    Case(
        "cqt", G, "full",
        librosa=lambda c: partial(librosa.cqt, c.y, sr=c.sr),
        ours=lambda c: c.a.cqt,
        func="cqt",
    ),
    Case(
        "vqt", G, "full",
        librosa=lambda c: partial(librosa.vqt, c.y, sr=c.sr),
        ours=lambda c: c.a.vqt,
        func="vqt",
    ),
    Case(
        "pseudo_cqt", G, "full",
        librosa=lambda c: partial(librosa.pseudo_cqt, c.y, sr=c.sr),
        ours=lambda c: c.a.pseudo_cqt,
        func="pseudo_cqt",
    ),
    Case(
        "hybrid_cqt", G, "full",
        librosa=lambda c: partial(librosa.hybrid_cqt, c.y, sr=c.sr),
        ours=lambda c: c.a.hybrid_cqt,
        func="hybrid_cqt",
    ),
    Case(
        "icqt", G, "full",
        librosa=lambda c: partial(librosa.icqt, c.C, sr=c.sr, length=c.n),
        ours=lambda c: partial(aus.AudioSamples.icqt, c.C, sample_rate=c.sr, length=c.n),
        func="icqt",
        note="input: librosa.cqt(y)",
    ),
    Case(
        "griffinlim_cqt", G, "short",
        librosa=lambda c: partial(librosa.griffinlim_cqt, _cqt_mag(c), sr=c.sr, length=c.n, init=None),
        ours=lambda c: partial(aus.AudioSamples.griffinlim_cqt, _cqt_mag(c), sample_rate=c.sr, length=c.n),
        func="griffinlim_cqt",
        note="init=None for both (audio_samples has only the zero-phase start; librosa's default is random)",
    ),
    Case(
        "chroma_cqt", G, "full",
        librosa=lambda c: partial(librosa.feature.chroma_cqt, y=c.y, sr=c.sr),
        # tuning=None: estimate it, librosa's default (the binding defaults to 0.0)
        ours=lambda c: partial(c.a.chroma_cqt, tuning=None),
        func="feature.chroma_cqt",
    ),
    Case(
        "chroma_cens", G, "full",
        librosa=lambda c: partial(librosa.feature.chroma_cens, y=c.y, sr=c.sr),
        ours=lambda c: partial(c.a.chroma_cens, tuning=None),
        func="feature.chroma_cens",
    ),
    Case(
        "chroma_vqt", G, "full",
        librosa=lambda c: partial(librosa.feature.chroma_vqt, y=c.y, sr=c.sr, intervals="equal"),
        ours=lambda c: c.a.chroma_vqt,
        func="feature.chroma_vqt",
        note="intervals='equal' (required by librosa; the only form audio_samples has)",
    ),
    Case(
        "tonnetz", G, "full",
        librosa=lambda c: partial(librosa.feature.tonnetz, y=c.y, sr=c.sr),
        ours=lambda c: partial(c.a.tonnetz, tuning=None),
        func="feature.tonnetz",
    ),
    Case(
        "tonnetz_chroma", G, "derived",
        librosa=lambda c: partial(librosa.feature.tonnetz, chroma=_chroma_cqt(c)),
        ours=lambda c: partial(F.tonnetz_from_chroma, _chroma_cqt(c)),
        func="feature.tonnetz",
        note="chroma= form; input: librosa.feature.chroma_cqt(y)",
    ),
    Case(
        "wavelet", G, "const",
        librosa=lambda c: partial(librosa.filters.wavelet, freqs=_cqt_freqs(), sr=22050),
        ours=lambda c: partial(F.wavelet, freqs=_cqt_freqs(), sr=22050),
        func="filters.wavelet",
        note="84 CQT frequencies from C1 at 22050 Hz; librosa's default dtype is complex64, ours complex128",
    ),
    Case(
        "wavelet_lengths", G, "const",
        librosa=lambda c: partial(librosa.filters.wavelet_lengths, freqs=_cqt_freqs(), sr=22050),
        ours=lambda c: partial(F.wavelet_lengths, freqs=_cqt_freqs(), sr=22050),
        func="filters.wavelet_lengths",
        note="84 CQT frequencies from C1 at 22050 Hz",
    ),
    Case(
        "cq_to_chroma", G, "const",
        librosa=lambda c: partial(librosa.filters.cq_to_chroma, 84),
        ours=lambda c: partial(F.cq_to_chroma, 84),
        func="filters.cq_to_chroma",
        note="n_input=84; librosa float32, ours float64",
    ),
)
