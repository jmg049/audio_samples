"""librosa core spectral representations (STFT, magphase, reassigned
spectrogram, phase vocoder, Griffin-Lim, PCEN, dB conversions, frequency
weightings), the mel spectrogram and the `librosa.filters` window and mel
functions.

Covered elsewhere: ``iirt``, ``fmt``, ``filters.mr_frequencies`` and
``filters.semitone_filterbank`` (pitch_filterbank group) and
``filters.diagonal_filter`` (segment group).
"""

import librosa
import numpy as np

import audio_samples as aus
from audio_samples import effects as E
from audio_samples import features as F
from audio_samples import spectral as S

from . import Case, partial, register, vec_values

G = "spectrum"

# Phase-vocoder stretch rate (required argument; the same in all three paths).
PV_RATE = 1.5
# Frames of a 10 s signal at 22050 Hz with librosa's default STFT (hop 512).
WSS_FRAMES = 431


def _mel_scaled(c):
    """The mel spectrogram scaled to integer-like magnitudes, as librosa's
    pcen documentation and the reference tests do."""
    return c.cached("mel_pcen", lambda: c.mel * c.dtype(2.0**31))


def _amp_db(c):
    return c.cached("amp_db", lambda: librosa.amplitude_to_db(c.S_mag))


def _freqs_hz(c):
    """10 Hz .. 20 kHz, log-spaced (frequency weighting input)."""
    return c.cached("freqs_hz", lambda: vec_values(10.0, 20000.0, np.float64))


register(
    # --- STFT ---------------------------------------------------------------
    Case(
        "stft", G, "full",
        librosa=lambda c: partial(librosa.stft, c.y),
        ours=lambda c: c.a.stft_matrix,
        func="stft",
    ),
    Case(
        "istft", G, "full",
        librosa=lambda c: partial(librosa.istft, c.D, length=c.n),
        ours=lambda c: partial(S.istft, c.D, length=c.n),
        func="istft",
    ),
    Case(
        "magphase", G, "derived_f64",
        librosa=lambda c: partial(librosa.magphase, c.D),
        ours=lambda c: partial(aus.AudioSamples.magphase, c.D),
        func="magphase",
        note="complex128 input",
    ),
    Case(
        "magphase_c64", G, "derived_f32",
        librosa=lambda c: partial(librosa.magphase, c.D),
        ours=lambda c: partial(aus.AudioSamples.magphase, c.D),
        func="magphase",
        note="complex64 input",
    ),
    Case(
        "reassigned_spectrogram", G, "full",
        librosa=lambda c: partial(librosa.reassigned_spectrogram, y=c.y, sr=c.sr),
        ours=lambda c: c.a.reassigned_spectrogram,
        func="reassigned_spectrogram",
        note="frequencies and times are NaN below ref_power (both libraries)",
    ),
    Case(
        "phase_vocoder", G, "derived",
        librosa=lambda c: partial(librosa.phase_vocoder, c.D, rate=PV_RATE),
        ours=lambda c: partial(E.phase_vocoder, c.D, rate=PV_RATE),
        func="phase_vocoder",
        note=f"rate={PV_RATE}",
    ),
    Case(
        "griffinlim", G, "short",
        librosa=lambda c: partial(librosa.griffinlim, c.S_mag, init=None),
        ours=lambda c: partial(E.griffinlim, c.S_mag, init=None),
        func="griffinlim",
        note="init=None (zero phase) in all paths, so the outputs are comparable; "
        "librosa's default random init draws unseeded phases",
    ),
    Case(
        "pcen", G, "derived",
        librosa=lambda c: partial(librosa.pcen, _mel_scaled(c), sr=c.sr),
        ours=lambda c: partial(F.pcen, _mel_scaled(c), sr=c.sr, hop_length=512),
        func="pcen",
        note="input: mel spectrogram * 2**31",
    ),
    # --- mel spectrogram ------------------------------------------------------
    Case(
        "melspectrogram", G, "full",
        librosa=lambda c: partial(librosa.feature.melspectrogram, y=c.y, sr=c.sr),
        ours=lambda c: c.a.melspectrogram,
        func="feature.melspectrogram",
    ),
    Case(
        "melspectrogram_S", G, "derived",
        librosa=lambda c: partial(librosa.feature.melspectrogram, S=c.S_pow, sr=c.sr),
        ours=lambda c: partial(S.melspectrogram_from_spectrogram, c.S_pow, sr=c.sr),
        func="feature.melspectrogram",
        note="S= form on |stft|**2",
    ),
    # --- dB conversions -------------------------------------------------------
    Case(
        "power_to_db", G, "derived",
        librosa=lambda c: partial(librosa.power_to_db, c.mel),
        ours=lambda c: partial(S.power_to_db, c.mel, top_db=80.0),
        func="power_to_db",
    ),
    Case(
        "amplitude_to_db", G, "derived",
        librosa=lambda c: partial(librosa.amplitude_to_db, c.S_mag),
        ours=lambda c: partial(S.amplitude_to_db, c.S_mag, top_db=80.0),
        func="amplitude_to_db",
    ),
    Case(
        "db_to_power", G, "derived",
        librosa=lambda c: partial(librosa.db_to_power, c.mel_db),
        ours=lambda c: partial(S.db_to_power, c.mel_db),
        func="db_to_power",
        note="input: power_to_db(mel)",
    ),
    Case(
        "db_to_amplitude", G, "derived",
        librosa=lambda c: partial(librosa.db_to_amplitude, _amp_db(c)),
        ours=lambda c: partial(S.db_to_amplitude, _amp_db(c)),
        func="db_to_amplitude",
        note="input: amplitude_to_db(|stft|)",
    ),
    # --- weightings -----------------------------------------------------------
    Case(
        "perceptual_weighting", G, "derived",
        librosa=lambda c: partial(librosa.perceptual_weighting, c.S_pow, c.freqs),
        ours=lambda c: partial(F.perceptual_weighting, c.S_pow, c.freqs, top_db=80.0),
        func="perceptual_weighting",
    ),
    Case(
        "frequency_weighting", G, "vec_f64",
        librosa=lambda c: partial(librosa.frequency_weighting, _freqs_hz(c)),
        ours=lambda c: partial(F.frequency_weighting, _freqs_hz(c), kind="A", min_db=-80.0),
        func="frequency_weighting",
        note="100000 frequencies, 10 Hz .. 20 kHz",
    ),
    Case(
        "multi_frequency_weighting", G, "vec_f64",
        librosa=lambda c: partial(librosa.multi_frequency_weighting, _freqs_hz(c)),
        ours=lambda c: partial(F.multi_frequency_weighting, _freqs_hz(c), min_db=-80.0),
        func="multi_frequency_weighting",
        note="100000 frequencies, kinds ZAC",
    ),
    *[
        Case(
            f"{k}_weighting", G, "vec_f64",
            librosa=(lambda fn: lambda c: partial(fn, _freqs_hz(c)))(getattr(librosa, f"{k}_weighting")),
            ours=(lambda fn: lambda c: partial(fn, _freqs_hz(c)))(getattr(F, f"{k.lower()}_weighting")),
            func=f"{k}_weighting",
            note="100000 frequencies, 10 Hz .. 20 kHz",
        )
        for k in "ABCDZ"
    ],
    # --- librosa.filters ------------------------------------------------------
    Case(
        "get_window", G, "const",
        librosa=lambda c: partial(librosa.filters.get_window, "hann", 2048),
        ours=lambda c: partial(S.get_window, "hann", 2048),
        func="filters.get_window",
        note="hann, 2048 points",
    ),
    Case(
        "filters_mel", G, "const",
        librosa=lambda c: partial(librosa.filters.mel, sr=22050, n_fft=2048),
        ours=lambda c: partial(S.mel_filterbank, sr=22050, n_fft=2048),
        func="filters.mel",
        note="sr=22050, n_fft=2048; librosa's default dtype is float32, ours float64",
    ),
    Case(
        "window_bandwidth", G, "const",
        librosa=lambda c: partial(librosa.filters.window_bandwidth, "hann"),
        ours=lambda c: partial(S.window_bandwidth, "hann"),
        func="filters.window_bandwidth",
        note="librosa looks up a table value for named windows; ours evaluates the "
        "formula over 1000 points",
    ),
    Case(
        "window_sumsquare", G, "const",
        librosa=lambda c: partial(librosa.filters.window_sumsquare, window="hann", n_frames=WSS_FRAMES),
        ours=lambda c: partial(S.window_sumsquare, window="hann", n_frames=WSS_FRAMES, dtype=np.float32),
        func="filters.window_sumsquare",
        note=f"hann, {WSS_FRAMES} frames, hop 512, n_fft 2048; float32 in all paths (librosa's default "
        "dtype; the binding's default is float64)",
    ),
)
