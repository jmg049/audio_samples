"""Tonal analysis: ``feature.chroma_stft``, ``filters.chroma``,
``piptrack``, ``estimate_tuning`` and ``pitch_tuning``, from audio and
(``_S`` cases) from a spectrogram.

No function of these tables is skipped.
"""

import librosa
import numpy as np

from audio_samples import features as F

from . import Case, partial, register, vec_values

G = "tonal"


def _pitches(c):
    """Positive frequencies for pitch_tuning: 50 Hz .. 4 kHz, log-spaced."""
    return c.cached("pitches", lambda: vec_values(50.0, 4000.0, np.float64))


register(
    Case(
        "chroma_stft", G, "full",
        librosa=lambda c: partial(librosa.feature.chroma_stft, y=c.y, sr=c.sr),
        ours=lambda c: partial(c.a.chroma_stft, tuning=None),
        func="feature.chroma_stft",
        note="tuning estimated (librosa's default; the binding's default is 0.0)",
    ),
    Case(
        "chroma_stft_S", G, "derived",
        librosa=lambda c: partial(librosa.feature.chroma_stft, S=c.S_pow, sr=c.sr),
        ours=lambda c: partial(F.chroma_stft_from_spectrogram, c.S_pow, sr=c.sr, tuning=None),
        func="feature.chroma_stft",
        note="S= form on |stft|**2; tuning estimated",
    ),
    Case(
        "filters_chroma", G, "const",
        librosa=lambda c: partial(librosa.filters.chroma, sr=22050, n_fft=2048),
        ours=lambda c: partial(F.chroma_filterbank, sr=22050, n_fft=2048),
        func="filters.chroma",
        note="sr=22050, n_fft=2048; librosa's default dtype is float32, ours float64",
    ),
    Case(
        "piptrack", G, "full",
        librosa=lambda c: partial(librosa.piptrack, y=c.y, sr=c.sr),
        ours=lambda c: c.a.piptrack,
        func="piptrack",
    ),
    Case(
        "piptrack_S", G, "derived",
        librosa=lambda c: partial(librosa.piptrack, S=c.S_mag, sr=c.sr),
        ours=lambda c: partial(F.piptrack_from_spectrogram, c.S_mag, sr=c.sr),
        func="piptrack",
        note="S= form on |stft|",
    ),
    Case(
        "estimate_tuning", G, "full",
        librosa=lambda c: partial(librosa.estimate_tuning, y=c.y, sr=c.sr),
        ours=lambda c: c.a.estimate_tuning,
        func="estimate_tuning",
    ),
    Case(
        "estimate_tuning_S", G, "derived",
        librosa=lambda c: partial(librosa.estimate_tuning, S=c.S_mag, sr=c.sr),
        ours=lambda c: partial(F.estimate_tuning_from_spectrogram, c.S_mag, sr=c.sr),
        func="estimate_tuning",
        note="S= form on |stft|",
    ),
    Case(
        "pitch_tuning", G, "vec_f64",
        librosa=lambda c: partial(librosa.pitch_tuning, _pitches(c)),
        ours=lambda c: partial(F.pitch_tuning, _pitches(c)),
        func="pitch_tuning",
        note="100000 frequencies, 50 Hz .. 4 kHz",
    ),
)
