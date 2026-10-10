"""librosa rhythm functions: onset strength and detection, tempograms, tempo,
beat tracking, PLP, metrogram and ``util.peak_pick``.

The plain names time the call on audio (librosa's ``y=`` form, the
``AudioSamples`` method). The ``_S`` / ``_env`` / ``_tg`` variants time the
free function on a prepared input (librosa's ``S=``, ``onset_envelope=`` or
``tg=`` form) computed with librosa and handed to both libraries.

Skipped rows: none. ``beat_track`` and ``onset_detect`` run on mono configs
only (audio_samples' event detectors take mono audio). ``tempogram_ratio`` on
audio is mono only: librosa 1.0 raises on multichannel input (its ``freqs``
shape check fails).

``hybrid_tempogram`` differs by ~2e-4 peak-relative in float32 (~1e-8 in
float64): ``sqrt(|F| * |A|)`` turns FFT round-off where ``A`` is zero into
``sqrt(eps)``-sized values on both sides.
"""

import librosa
import numpy as np

from audio_samples import rhythm as R

from . import Case, partial, register

G = "rhythm"


def _tg(c):
    """Autocorrelation tempogram of the onset envelope (librosa)."""
    return c.cached("tg", lambda: librosa.feature.tempogram(onset_envelope=c.onset_env, sr=c.sr))


def _ftg_mag(c):
    """|Fourier tempogram| without its last frame, the metrogram input."""
    return c.cached(
        "ftg_mag",
        lambda: np.abs(librosa.feature.fourier_tempogram(onset_envelope=c.onset_env, sr=c.sr))[..., :-1],
    )


def _onsets(c):
    """Onset frames of the envelope (librosa)."""
    return c.cached("onsets", lambda: librosa.onset.onset_detect(onset_envelope=c.onset_env, sr=c.sr))


# The parameters of librosa's peak_pick docstring example.
PEAK = dict(pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
# Mel sub-band boundaries for onset_strength_multi_S.
BANDS = [0, 16, 32, 64, 128]

register(
    # --- onset strength -------------------------------------------------------
    Case(
        "onset_strength", G, "full",
        librosa=lambda c: partial(librosa.onset.onset_strength, y=c.y, sr=c.sr),
        ours=lambda c: c.a.onset_strength,
        func="onset.onset_strength",
    ),
    Case(
        "onset_strength_S", G, "derived",
        librosa=lambda c: partial(librosa.onset.onset_strength, S=c.mel_db, sr=c.sr),
        ours=lambda c: partial(R.onset_strength, S=c.mel_db, sr=c.sr),
        func="onset.onset_strength",
        note="S = power_to_db(melspectrogram(y))",
    ),
    Case(
        "onset_strength_multi", G, "full",
        librosa=lambda c: partial(librosa.onset.onset_strength_multi, y=c.y, sr=c.sr),
        ours=lambda c: c.a.onset_strength_multi,
        func="onset.onset_strength_multi",
    ),
    Case(
        "onset_strength_multi_S", G, "derived",
        librosa=lambda c: partial(librosa.onset.onset_strength_multi, S=c.mel_db, sr=c.sr, channels=BANDS),
        ours=lambda c: partial(R.onset_strength_multi, S=c.mel_db, sr=c.sr, channels=BANDS),
        func="onset.onset_strength_multi",
        note="S = power_to_db(melspectrogram(y)); four mel sub-bands",
    ),
    # --- onset events -----------------------------------------------------------
    Case(
        "onset_detect", G, "mono",
        librosa=lambda c: partial(librosa.onset.onset_detect, y=c.y, sr=c.sr),
        ours=lambda c: c.a.onset_detect,
        func="onset.onset_detect",
    ),
    Case(
        "onset_detect_env", G, "derived",
        librosa=lambda c: partial(librosa.onset.onset_detect, onset_envelope=c.onset_env, sr=c.sr),
        ours=lambda c: partial(R.onset_detect, onset_envelope=c.onset_env, sr=c.sr),
        func="onset.onset_detect",
    ),
    Case(
        "onset_backtrack", G, "derived",
        librosa=lambda c: partial(librosa.onset.onset_backtrack, _onsets(c), c.onset_env),
        ours=lambda c: partial(R.onset_backtrack, _onsets(c), c.onset_env),
        func="onset.onset_backtrack",
        note="events = onset_detect(onset_envelope); energy = the envelope",
    ),
    Case(
        "peak_pick", G, "derived",
        librosa=lambda c: partial(librosa.util.peak_pick, c.onset_env, **PEAK),
        ours=lambda c: partial(R.peak_pick, c.onset_env, **PEAK),
        func="util.peak_pick",
        note="on the onset envelope, with librosa's docstring parameters",
    ),
    # --- tempograms -------------------------------------------------------------
    Case(
        "tempogram", G, "full",
        librosa=lambda c: partial(librosa.feature.tempogram, y=c.y, sr=c.sr),
        ours=lambda c: c.a.tempogram,
        func="feature.tempogram",
    ),
    Case(
        "tempogram_env", G, "derived",
        librosa=lambda c: partial(librosa.feature.tempogram, onset_envelope=c.onset_env, sr=c.sr),
        ours=lambda c: partial(R.tempogram, onset_envelope=c.onset_env, sr=c.sr),
        func="feature.tempogram",
    ),
    Case(
        "fourier_tempogram", G, "full",
        librosa=lambda c: partial(librosa.feature.fourier_tempogram, y=c.y, sr=c.sr),
        ours=lambda c: c.a.fourier_tempogram,
        func="feature.fourier_tempogram",
    ),
    Case(
        "fourier_tempogram_env", G, "derived",
        librosa=lambda c: partial(librosa.feature.fourier_tempogram, onset_envelope=c.onset_env, sr=c.sr),
        ours=lambda c: partial(R.fourier_tempogram, onset_envelope=c.onset_env, sr=c.sr),
        func="feature.fourier_tempogram",
    ),
    Case(
        "hybrid_tempogram", G, "full",
        librosa=lambda c: partial(librosa.feature.hybrid_tempogram, y=c.y, sr=c.sr),
        ours=lambda c: c.a.hybrid_tempogram,
        func="feature.hybrid_tempogram",
    ),
    Case(
        "hybrid_tempogram_env", G, "derived",
        librosa=lambda c: partial(librosa.feature.hybrid_tempogram, onset_envelope=c.onset_env, sr=c.sr),
        ours=lambda c: partial(R.hybrid_tempogram, onset_envelope=c.onset_env, sr=c.sr),
        func="feature.hybrid_tempogram",
    ),
    Case(
        "tempogram_ratio", G, "mono",
        librosa=lambda c: partial(librosa.feature.tempogram_ratio, y=c.y, sr=c.sr),
        ours=lambda c: c.a.tempogram_ratio,
        func="feature.tempogram_ratio",
    ),
    Case(
        "tempogram_ratio_tg", G, "derived",
        librosa=lambda c: partial(librosa.feature.tempogram_ratio, tg=_tg(c), sr=c.sr),
        ours=lambda c: partial(R.tempogram_ratio, tg=_tg(c), sr=c.sr),
        func="feature.tempogram_ratio",
        note="tg = tempogram(onset_envelope)",
    ),
    Case(
        "metrogram", G, "derived",
        librosa=lambda c: partial(
            librosa.feature.metrogram, tg=_ftg_mag(c), freqs=librosa.fourier_tempo_frequencies(sr=c.sr)
        ),
        ours=lambda c: partial(R.metrogram, tg=_ftg_mag(c), freqs=librosa.fourier_tempo_frequencies(sr=c.sr)),
        func="feature.metrogram",
        note="tg = |fourier_tempogram(onset_envelope)| without the last frame",
    ),
    # --- tempo, beats, PLP --------------------------------------------------------
    Case(
        "tempo", G, "full",
        librosa=lambda c: partial(librosa.feature.tempo, y=c.y, sr=c.sr),
        ours=lambda c: c.a.tempo,
        func="feature.tempo",
    ),
    Case(
        "tempo_env", G, "derived",
        librosa=lambda c: partial(librosa.feature.tempo, onset_envelope=c.onset_env, sr=c.sr),
        ours=lambda c: partial(R.tempo, onset_envelope=c.onset_env, sr=c.sr),
        func="feature.tempo",
    ),
    Case(
        "tempo_tg", G, "derived",
        librosa=lambda c: partial(librosa.feature.tempo, tg=_tg(c), sr=c.sr),
        ours=lambda c: partial(R.tempo, tg=_tg(c), sr=c.sr),
        func="feature.tempo",
        note="tg = tempogram(onset_envelope)",
    ),
    Case(
        "beat_track", G, "mono",
        librosa=lambda c: partial(librosa.beat.beat_track, y=c.y, sr=c.sr),
        ours=lambda c: partial(c.a.beat_track, trim=True),
        func="beat.beat_track",
        note="the binding defaults to trim=False; librosa's trim=True is passed",
    ),
    Case(
        "beat_track_env", G, "derived",
        librosa=lambda c: partial(librosa.beat.beat_track, onset_envelope=c.onset_env, sr=c.sr),
        ours=lambda c: partial(R.beat_track, onset_envelope=c.onset_env, sr=c.sr, trim=True),
        func="beat.beat_track",
        note="the binding defaults to trim=False; librosa's trim=True is passed",
    ),
    Case(
        "plp", G, "full",
        librosa=lambda c: partial(librosa.beat.plp, y=c.y, sr=c.sr),
        ours=lambda c: c.a.plp,
        func="beat.plp",
    ),
    Case(
        "plp_env", G, "derived",
        librosa=lambda c: partial(librosa.beat.plp, onset_envelope=c.onset_env, sr=c.sr),
        ours=lambda c: partial(R.plp, onset_envelope=c.onset_env, sr=c.sr),
        func="beat.plp",
    ),
)
