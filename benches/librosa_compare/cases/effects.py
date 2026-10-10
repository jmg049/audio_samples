"""librosa core audio functions (resample, channel conversion, lpc, signal
generation, mu-law), ``librosa.effects``,
``librosa.decompose`` and ``util.softmask`` / ``util.fix_length``.

Skipped rows:

* ``get_duration``: ``AudioSamples.duration_seconds`` is a property read
  (``n / sr``); there is no work to time.
* ``phase_vocoder``, ``griffinlim``: timed in the ``spectrum`` group.
"""

import librosa
import numpy as np

import audio_samples as aus
from audio_samples import effects as E

from . import Case, partial, register

G = "effects"

# The previous benchmark's settings.
STRETCH_RATE = 1.25
PITCH_STEPS = 4.0
# The pitch-shift resampling ratio for n_steps = 4: 2^(-4/12).
SHIFT_RATIO = 2.0 ** (-PITCH_STEPS / 12.0)
LPC_ORDER = 16
AUTOCORR_SIZE = 2048
TOP_DB = 60.0  # librosa's trim / split default (required by the binding)
MU = 255.0


def _resample(target):
    """Builders for ``resample`` to ``target`` Hz: soxr_hq (librosa's default)
    against ResamplingQuality.high."""
    lib = lambda c: partial(librosa.resample, c.y, orig_sr=c.sr, target_sr=target)  # noqa: E731
    ours = lambda c: partial(c.a.resample, target, "high")  # noqa: E731
    return lib, ours


def _resample_case(src, dst, configs):
    lib, ours = _resample(dst)
    return Case(
        f"resample_{src}_{dst}", G, configs, librosa=lib, ours=ours, func="resample",
        note="librosa res_type='soxr_hq' (default) vs ResamplingQuality.high",
    )


def _split_intervals(c):
    """``librosa.effects.split(y, top_db=8)`` (about one interval per note) in
    reverse order: a remix that reorders the notes."""
    return c.cached("split_rev", lambda: np.ascontiguousarray(librosa.effects.split(c.y, top_db=8)[::-1]))


def _channels(c):
    """The two channels of a stereo signal as separate mono AudioSamples."""
    return c.cached(
        "channels",
        lambda: [aus.AudioSamples.new_mono(np.ascontiguousarray(c.y[i]), c.sr) for i in range(c.y.shape[0])],
    )


def _hpss_mag(c):
    """``librosa.decompose.hpss(|stft|)``: harmonic and percussive magnitudes."""
    return c.cached("hpss_mag", lambda: librosa.decompose.hpss(c.S_mag))


def _mu_codes(c):
    return c.cached("mu_codes", lambda: librosa.mu_compress(c.y, mu=MU))


def _mu_float(c):
    return c.cached("mu_float", lambda: librosa.mu_compress(c.y, mu=MU, quantize=False))


def _mu_float_audio(c):
    def make():
        x = _mu_float(c)
        if x.ndim == 1:
            return aus.AudioSamples.new_mono(x, c.sr)
        return aus.AudioSamples.new_multi(x, c.sr)

    return c.cached("mu_float_a", make)


CLICK_TIMES = np.arange(0.0, 60.0, 0.5)  # 120 clicks over 60 s
CLICK_SR = 22050
CLICK_LEN = 60 * CLICK_SR
GEN_SR = 22050
GEN_SECONDS = 10.0

register(
    # --- core: audio ----------------------------------------------------------
    _resample_case(22050, 16000, "rate_22050"),
    _resample_case(22050, 44100, "rate_22050"),
    _resample_case(22050, 11025, "rate_22050"),
    _resample_case(16000, 22050, "rate_16000"),
    Case(
        "resample_by_ratio", G, "full",
        librosa=lambda c: partial(librosa.resample, c.y, orig_sr=c.sr / SHIFT_RATIO, target_sr=c.sr),
        ours=lambda c: partial(c.a.resample_by_ratio, SHIFT_RATIO, "high"),
        func="resample",
        note="ratio 2^(-4/12) (pitch_shift n_steps=4); librosa resample(orig_sr=sr/ratio, target_sr=sr), "
        "soxr_hq vs ResamplingQuality.high; ours rounds the target rate to an integer",
    ),
    Case(
        "to_mono", G, "stereo",
        librosa=lambda c: partial(librosa.to_mono, c.y),
        ours=lambda c: c.a.to_mono,
        func="to_mono",
    ),
    Case(
        "to_stereo", G, "mono",
        librosa=lambda c: partial(librosa.to_stereo, left=c.y, right=c.y),
        ours=lambda c: c.a.to_stereo,
        func="to_stereo",
        note="mono duplicated to both channels (StereoConversionMethod.duplicate)",
    ),
    Case(
        "to_multi", G, "stereo",
        librosa=lambda c: partial(librosa.to_multi, c.y[0], c.y[1]),
        ours=lambda c: partial(aus.AudioSamples.stack, _channels(c)),
        func="to_multi",
        note="two mono signals stacked as channels (AudioSamples.stack)",
    ),
    Case(
        "autocorrelate", G, "mono",
        librosa=lambda c: partial(librosa.autocorrelate, c.y, max_size=AUTOCORR_SIZE),
        ours=lambda c: partial(c.a.autocorrelation, AUTOCORR_SIZE - 1),
        func="autocorrelate",
        note="max_size=2048; ours is the unbiased estimate r[k] = raw[k] / (n - k), "
        "f64 and a Python list (first channel only)",
    ),
    Case(
        "lpc", G, "full",
        librosa=lambda c: partial(librosa.lpc, c.y, order=LPC_ORDER),
        ours=lambda c: partial(c.a.lpc, order=LPC_ORDER),
        func="lpc",
        note="order=16",
    ),
    Case(
        "zero_crossings", G, "full",
        librosa=lambda c: lambda y=c.y: np.count_nonzero(librosa.zero_crossings(y)),
        ours=lambda c: c.a.zero_crossings,
        func="zero_crossings",
        note="librosa returns a per-sample mask, ours a count: librosa's side is "
        "count_nonzero(zero_crossings(y)) (default threshold 1e-10, zero_pos)",
    ),
    Case(
        "clicks", G, "const",
        librosa=lambda c: partial(librosa.clicks, times=CLICK_TIMES, sr=CLICK_SR, length=CLICK_LEN),
        ours=lambda c: partial(E.clicks, times=CLICK_TIMES, sr=CLICK_SR, length=CLICK_LEN),
        func="clicks",
        note="120 clicks, 60 s at 22050 Hz; librosa float32, ours float64",
    ),
    Case(
        "tone", G, "const",
        librosa=lambda c: partial(librosa.tone, 440.0, sr=GEN_SR, duration=GEN_SECONDS),
        ours=lambda c: partial(aus.sine_wave, 440.0, GEN_SECONDS, GEN_SR, 1.0, np.float64),
        func="tone",
        note="440 Hz, 10 s at 22050 Hz, float64",
    ),
    Case(
        "chirp", G, "const",
        librosa=lambda c: partial(librosa.chirp, fmin=110.0, fmax=8000.0, sr=GEN_SR, duration=GEN_SECONDS),
        ours=lambda c: partial(aus.exponential_chirp, 110.0, 8000.0, GEN_SECONDS, GEN_SR, 1.0, np.float64),
        func="chirp",
        note="exponential (librosa's default linear=False), 110..8000 Hz, 10 s at 22050 Hz",
    ),
    Case(
        "chirp_linear", G, "const",
        librosa=lambda c: partial(
            librosa.chirp, fmin=110.0, fmax=8000.0, sr=GEN_SR, duration=GEN_SECONDS, linear=True
        ),
        ours=lambda c: partial(aus.chirp, 110.0, 8000.0, GEN_SECONDS, GEN_SR, 1.0, np.float64),
        func="chirp",
        note="linear=True, 110..8000 Hz, 10 s at 22050 Hz",
    ),
    Case(
        "mu_compress", G, "full",
        librosa=lambda c: partial(librosa.mu_compress, c.y, mu=MU),
        ours=lambda c: partial(aus.mu_law_encode, c.y, mu=MU),
        func="mu_compress",
        note="quantize=True (librosa's default): integer codes, mu_law_encode",
    ),
    Case(
        "mu_compress_float", G, "full",
        librosa=lambda c: partial(librosa.mu_compress, c.y, mu=MU, quantize=False),
        ours=lambda c: partial(c.a.mu_compress, MU),
        func="mu_compress",
        note="quantize=False: AudioSamples.mu_compress",
    ),
    Case(
        "mu_expand", G, "full",
        librosa=lambda c: partial(librosa.mu_expand, _mu_codes(c), mu=MU),
        ours=lambda c: partial(aus.mu_law_decode, _mu_codes(c), mu=MU),
        func="mu_expand",
        note="quantize=True (librosa's default): input librosa.mu_compress(y) codes, mu_law_decode",
    ),
    Case(
        "mu_expand_float", G, "full",
        librosa=lambda c: partial(librosa.mu_expand, _mu_float(c), mu=MU, quantize=False),
        ours=lambda c: partial(_mu_float_audio(c).mu_expand, MU),
        func="mu_expand",
        note="quantize=False: input librosa.mu_compress(y, quantize=False), AudioSamples.mu_expand",
    ),
    # --- librosa.effects --------------------------------------------------------
    Case(
        "time_stretch", G, "full",
        librosa=lambda c: partial(librosa.effects.time_stretch, c.y, rate=STRETCH_RATE),
        ours=lambda c: partial(c.a.time_stretch, STRETCH_RATE),
        func="effects.time_stretch",
        note="rate=1.25",
    ),
    Case(
        "pitch_shift", G, "full",
        librosa=lambda c: partial(librosa.effects.pitch_shift, c.y, sr=c.sr, n_steps=PITCH_STEPS),
        ours=lambda c: partial(c.a.pitch_shift, PITCH_STEPS),
        func="effects.pitch_shift",
        note="n_steps=4; soxr_hq vs ResamplingQuality.high in the final resampling",
    ),
    Case(
        "preemphasis", G, "full",
        librosa=lambda c: partial(librosa.effects.preemphasis, c.y),
        ours=lambda c: c.a.preemphasis,
        func="effects.preemphasis",
    ),
    Case(
        "deemphasis", G, "full",
        librosa=lambda c: partial(librosa.effects.deemphasis, c.y),
        ours=lambda c: c.a.deemphasis,
        func="effects.deemphasis",
    ),
    Case(
        "trim", G, "full",
        librosa=lambda c: partial(librosa.effects.trim, c.y),
        ours=lambda c: partial(c.a.trim_db, top_db=TOP_DB),
        func="effects.trim",
    ),
    Case(
        "split", G, "full",
        librosa=lambda c: partial(librosa.effects.split, c.y),
        ours=lambda c: partial(c.a.split_db, top_db=TOP_DB),
        func="effects.split",
    ),
    Case(
        "remix", G, "full",
        librosa=lambda c: partial(librosa.effects.remix, c.y, _split_intervals(c)),
        # align_zeros=True: librosa's default (the binding defaults to False)
        ours=lambda c: partial(c.a.remix, _split_intervals(c), align_zeros=True),
        func="effects.remix",
        note="intervals: librosa.effects.split(y, top_db=8) in reverse order; align_zeros=True",
    ),
    Case(
        "hpss", G, "full",
        librosa=lambda c: partial(librosa.effects.hpss, c.y),
        ours=lambda c: c.a.harmonic_percussive,
        func="effects.hpss",
    ),
    Case(
        "harmonic", G, "full",
        librosa=lambda c: partial(librosa.effects.harmonic, c.y),
        ours=lambda c: c.a.harmonic,
        func="effects.harmonic",
    ),
    Case(
        "percussive", G, "full",
        librosa=lambda c: partial(librosa.effects.percussive, c.y),
        ours=lambda c: c.a.percussive,
        func="effects.percussive",
    ),
    # --- librosa.decompose -------------------------------------------------------
    Case(
        "decompose_hpss", G, "full",
        librosa=lambda c: partial(librosa.decompose.hpss, c.D),
        ours=lambda c: partial(E.hpss, c.D),
        func="decompose.hpss",
        note="input: complex librosa.stft(y)",
    ),
    Case(
        "nn_filter", G, "derived",
        librosa=lambda c: partial(librosa.decompose.nn_filter, c.chroma),
        ours=lambda c: partial(E.nn_filter, c.chroma),
        func="decompose.nn_filter",
        note="input: librosa.feature.chroma_stft(y); recurrence matrix computed inside (k nearest neighbours)",
    ),
    # --- librosa.util ------------------------------------------------------------
    Case(
        "softmask", G, "derived",
        librosa=lambda c: partial(librosa.util.softmask, *_hpss_mag(c)),
        ours=lambda c: partial(E.softmask, *_hpss_mag(c)),
        func="util.softmask",
        note="X, X_ref: harmonic and percussive magnitudes of librosa.decompose.hpss(|stft|)",
    ),
    Case(
        "fix_length", G, "full",
        librosa=lambda c: partial(librosa.util.fix_length, c.y, size=c.n + c.sr),
        ours=lambda c: partial(c.a.pad_samples_right, c.n + c.sr, 0.0),
        func="util.fix_length",
        note="pads one second of zeros (AudioSamples.pad_samples_right)",
    ),
)
