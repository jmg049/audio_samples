"""`librosa.feature`: MFCC, deltas, frame-wise spectral features, RMS and
zero-crossing rate, from audio and (``_S`` cases) from a spectrogram.

Covered elsewhere: ``melspectrogram`` (spectrum group), ``chroma_stft``
(tonal group), the constant-Q chroma / tonnetz (constant_q group), the
rhythm features (rhythm group) and ``feature.inverse`` (inverse group).
``zero_crossing_rate`` has no ``S=`` form in librosa, so it has no ``_S``
case.
"""

import librosa

from audio_samples import features as F
from audio_samples import spectral as S

from . import Case, partial, register

G = "features"


def _y_case(name, fy, ours, note=""):
    return Case(
        name, G, "full",
        librosa=lambda c: partial(fy, y=c.y, sr=c.sr),
        ours=lambda c: getattr(c.a, ours),
        func=f"feature.{name}",
        note=note,
    )


def _s_case(name, fy, fs, note="S= form on |stft|"):
    return Case(
        f"{name}_S", G, "derived",
        librosa=lambda c: partial(fy, S=c.S_mag, sr=c.sr),
        ours=lambda c: partial(fs, c.S_mag, sr=c.sr),
        func=f"feature.{name}",
        note=note,
    )


lf = librosa.feature

register(
    Case(
        "mfcc", G, "full",
        librosa=lambda c: partial(lf.mfcc, y=c.y, sr=c.sr),
        ours=lambda c: partial(c.a.mfcc_frames, top_db=80.0),
        func="feature.mfcc",
    ),
    Case(
        "mfcc_S", G, "derived",
        librosa=lambda c: partial(lf.mfcc, S=c.mel_db),
        ours=lambda c: partial(S.mfcc_from_mel_db, c.mel_db),
        func="feature.mfcc",
        note="S= form on power_to_db(mel)",
    ),
    Case(
        "delta", G, "derived",
        librosa=lambda c: partial(lf.delta, c.mfcc),
        ours=lambda c: partial(S.delta, c.mfcc),
        func="feature.delta",
        note="input: mfcc (20 x frames)",
    ),
    Case(
        "stack_memory", G, "derived",
        librosa=lambda c: partial(lf.stack_memory, c.mfcc),
        ours=lambda c: partial(S.stack_memory, c.mfcc),
        func="feature.stack_memory",
        note="input: mfcc (20 x frames)",
    ),
    _y_case("spectral_centroid", lf.spectral_centroid, "spectral_centroid_frames"),
    _s_case("spectral_centroid", lf.spectral_centroid, F.spectral_centroid_from_spectrogram),
    _y_case("spectral_bandwidth", lf.spectral_bandwidth, "spectral_bandwidth_frames"),
    _s_case("spectral_bandwidth", lf.spectral_bandwidth, F.spectral_bandwidth_from_spectrogram),
    _y_case("spectral_contrast", lf.spectral_contrast, "spectral_contrast_frames"),
    _s_case("spectral_contrast", lf.spectral_contrast, F.spectral_contrast_from_spectrogram),
    Case(
        "spectral_flatness", G, "full",
        librosa=lambda c: partial(lf.spectral_flatness, y=c.y),
        ours=lambda c: c.a.spectral_flatness_frames,
        func="feature.spectral_flatness",
    ),
    Case(
        "spectral_flatness_S", G, "derived",
        librosa=lambda c: partial(lf.spectral_flatness, S=c.S_mag),
        ours=lambda c: partial(F.spectral_flatness_from_spectrogram, c.S_mag),
        func="feature.spectral_flatness",
        note="S= form on |stft|",
    ),
    _y_case("spectral_rolloff", lf.spectral_rolloff, "spectral_rolloff_frames"),
    _s_case("spectral_rolloff", lf.spectral_rolloff, F.spectral_rolloff_from_spectrogram),
    _y_case("poly_features", lf.poly_features, "poly_features"),
    _s_case("poly_features", lf.poly_features, F.poly_features_from_spectrogram),
    Case(
        "rms", G, "full",
        librosa=lambda c: partial(lf.rms, y=c.y),
        ours=lambda c: c.a.rms_frames,
        func="feature.rms",
        note="librosa's default dtype is float32 (also for float64 input)",
    ),
    Case(
        "rms_S", G, "derived",
        librosa=lambda c: partial(lf.rms, S=c.S_mag),
        ours=lambda c: partial(F.rms_from_spectrogram, c.S_mag),
        func="feature.rms",
        note="S= form on |stft|; librosa's default dtype is float32",
    ),
    Case(
        "zero_crossing_rate", G, "full",
        librosa=lambda c: partial(lf.zero_crossing_rate, c.y),
        ours=lambda c: c.a.zero_crossing_rate_frames,
        func="feature.zero_crossing_rate",
    ),
)
