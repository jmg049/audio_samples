"""Rhythm area: onset strength, peak picking, onset detection / backtracking,
tempograms, tempo, beat tracking and PLP.

Signals: ``clicks120`` (3 s click track at 120 BPM), ``notes``, ``mix`` and the
stereo pair. Onset envelopes are computed with ``dtype=np.float64`` (forwarded to
``librosa.filters.mel``) because audio_samples builds the mel basis in float64;
librosa's default float32 basis only adds ~6e-8 relative rounding to the weights.
The float32-basis beat tracker output is stored too (``beat_track.*_f32basis``)
to show that the frame indices do not depend on it.

Algorithms that librosa delegates:
* the mel power spectrogram and dB scaling (``spectrum`` area);
* ``scipy.ndimage.maximum_filter1d`` (``max_size > 1``, reflect boundary);
* ``scipy.signal.lfilter`` (``detrend``);
* ``scipy.fft`` (tempogram autocorrelation, Fourier tempogram / PLP STFT);
* numba kernels for peak picking and the dynamic-programming beat tracker.

``float32`` cases (``float32.<sig>``) run librosa on ``y.astype(np.float32)`` with
its default ``float32`` mel basis: the onset envelope, Fourier tempogram and PLP
stay ``float32``; the autocorrelation tempogram (and so the tempo) is computed in
``float64`` because librosa's ``float64`` window promotes the ``float32`` envelope.
audio_samples analyses ``f32`` audio in ``f32`` throughout.

Large matrices are stored as strided subsets (``[::bin_step, ::frame_step]``).
Envelopes used as direct inputs (``inputs``) are stored in full so the Rust side
can test peak picking, tempo and beat tracking on librosa's exact input.
"""

from __future__ import annotations

import librosa
import numpy as np

NAME = "rhythm"
F64 = dict(dtype=np.float64)


def _onset_strength(signals, sr):
    out = {}
    cases = {
        "default": dict(),
        "lag2": dict(lag=2),
        "max_size3": dict(max_size=3),
        "median": dict(aggregate=np.median),
        "nocenter": dict(center=False),
        "detrend": dict(detrend=True),
        "hop256_nfft1024": dict(hop_length=256, n_fft=1024),
    }
    for sig in ("clicks120", "notes", "mix"):
        y = signals[sig]
        out[sig] = {name: librosa.onset.onset_strength(y=y, sr=sr, **kw, **F64) for name, kw in cases.items()}
    stereo = np.stack([signals["stereo_l"], signals["stereo_r"]])
    out["stereo"] = {"default": librosa.onset.onset_strength(y=stereo, sr=sr, **F64)}
    return out


def _onset_strength_multi(signals, sr):
    notes = signals["notes"]
    stereo = np.stack([signals["stereo_l"], signals["stereo_r"]])
    bounds = [0, 16, 40, 80, 128]
    return {
        "bounds": bounds,
        "notes_default": librosa.onset.onset_strength_multi(y=notes, sr=sr, **F64),
        "notes_bands": librosa.onset.onset_strength_multi(y=notes, sr=sr, channels=bounds, **F64),
        # unsorted / duplicated / out-of-range boundaries are normalised by librosa
        "notes_bands_messy": librosa.onset.onset_strength_multi(
            y=notes, sr=sr, channels=[64, 0, 64, 200, 20], aggregate=np.max, **F64
        ),
        "stereo_bands": librosa.onset.onset_strength_multi(y=stereo, sr=sr, channels=bounds, **F64),
        "notes_no_aggregate_row_step": 16,
        "notes_no_aggregate": librosa.onset.onset_strength_multi(y=notes, sr=sr, aggregate=False, **F64)[::16],
    }


def _peak_pick(rng):
    x = rng.uniform(0.0, 1.0, 300)
    # a few plateaus (ties) to exercise the x[n] == max(...) test
    x[50:53] = 0.97
    x[120:122] = 0.99
    params = {
        "onset_defaults": dict(pre_max=1, post_max=1, pre_avg=4, post_avg=5, delta=0.07, wait=1),
        "wide": dict(pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10),
        "zero_wait": dict(pre_max=0, post_max=1, pre_avg=0, post_avg=1, delta=0.0, wait=0),
        "long_post": dict(pre_max=10, post_max=20, pre_avg=7, post_avg=20, delta=0.1, wait=5),
        "dp_count": dict(pre_max=2, post_max=2, pre_avg=4, post_avg=5, delta=0.05, wait=4, method="dp_count"),
        "dp_value": dict(pre_max=2, post_max=2, pre_avg=4, post_avg=5, delta=0.05, wait=4, method="dp_value"),
    }
    return {
        "x": x,
        "cases": {
            name: {"params": p, "peaks": librosa.util.peak_pick(x, **p)} for name, p in params.items()
        },
    }


def _onset_detect(signals, sr):
    out = {}
    for sig in ("clicks120", "notes", "mix"):
        y = signals[sig]
        env = librosa.onset.onset_strength(y=y, sr=sr, **F64)
        rms = librosa.feature.rms(y=y)[0]
        out[sig] = {
            "frames": librosa.onset.onset_detect(onset_envelope=env, sr=sr),
            "backtrack": librosa.onset.onset_detect(onset_envelope=env, sr=sr, backtrack=True),
            "backtrack_rms": librosa.onset.onset_detect(onset_envelope=env, sr=sr, backtrack=True, energy=rms),
            "rms": rms,
            "no_normalize": librosa.onset.onset_detect(onset_envelope=env, sr=sr, normalize=False, delta=0.5),
            "custom": librosa.onset.onset_detect(
                onset_envelope=env, sr=sr, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.2, wait=4
            ),
            "samples": librosa.onset.onset_detect(onset_envelope=env, sr=sr, units="samples"),
            "time": librosa.onset.onset_detect(onset_envelope=env, sr=sr, units="time"),
            # From the signal (default float32 mel basis): frame indices should agree.
            "from_y_f32basis": librosa.onset.onset_detect(y=y, sr=sr),
        }
    return out


def _tempograms(signals, sr):
    out = {}
    for sig in ("clicks120", "notes"):
        y = signals[sig]
        env = librosa.onset.onset_strength(y=y, sr=sr, **F64)
        tg = librosa.feature.tempogram(onset_envelope=env, sr=sr)
        ftg = librosa.feature.fourier_tempogram(onset_envelope=env, sr=sr)
        small = librosa.feature.tempogram(onset_envelope=env, sr=sr, win_length=32, center=False, norm=2.0)
        small_center = librosa.feature.tempogram(onset_envelope=env, sr=sr, win_length=33, window="hamming", norm=None)
        out[sig] = {
            "envelope": env,
            "tempogram_shape": list(tg.shape),
            "bin_step": 8,
            "frame_step": 4,
            "tempogram": tg[::8, ::4],
            "tempogram_w32_nocenter_l2": small,
            "tempogram_w33_hamming_nonorm": small_center,
            "fourier_shape": list(ftg.shape),
            "fourier_mag": np.abs(ftg[::8, ::4]),
            "fourier": ftg[::24, ::8],
            "fourier_w40_nocenter": librosa.feature.fourier_tempogram(
                onset_envelope=env, sr=sr, win_length=40, center=False
            )[::4, ::4],
        }
    stereo = np.stack([signals["stereo_l"], signals["stereo_r"]])
    tg = librosa.feature.tempogram(onset_envelope=librosa.onset.onset_strength(y=stereo, sr=sr, **F64), sr=sr)
    out["stereo"] = {"tempogram_shape": list(tg.shape), "tempogram": tg[:, ::16, ::4]}
    out["tempo_frequencies"] = {
        # bin 0 is +inf (zero lag) and is not stored
        "t384_from1": librosa.tempo_frequencies(384, sr=sr, hop_length=512)[1:],
        "t10_hop256_sr16000_from1": librosa.tempo_frequencies(10, sr=16000, hop_length=256)[1:],
        "f384": librosa.fourier_tempo_frequencies(sr=sr, win_length=384, hop_length=512),
        "f63_hop256_sr16000": librosa.fourier_tempo_frequencies(sr=16000, win_length=63, hop_length=256),
    }
    return out


def _tempo(signals, sr):
    out = {}
    for sig in ("clicks120", "notes", "mix"):
        y = signals[sig]
        env = librosa.onset.onset_strength(y=y, sr=sr, **F64)
        out[sig] = {
            "default": librosa.feature.tempo(onset_envelope=env, sr=sr),
            "time_varying": librosa.feature.tempo(onset_envelope=env, sr=sr, aggregate=None),
            "start60_std05_nomax_ac4": librosa.feature.tempo(
                onset_envelope=env, sr=sr, start_bpm=60, std_bpm=0.5, max_tempo=None, ac_size=4.0
            ),
            "median_max200": librosa.feature.tempo(onset_envelope=env, sr=sr, aggregate=np.median, max_tempo=200.0),
            "from_y_f32basis": librosa.feature.tempo(y=y, sr=sr),
        }
    stereo = np.stack([signals["stereo_l"], signals["stereo_r"]])
    out["stereo"] = {"default": librosa.feature.tempo(y=stereo, sr=sr)}
    out["stereo_f64"] = {
        "default": librosa.feature.tempo(onset_envelope=librosa.onset.onset_strength(y=stereo, sr=sr, **F64), sr=sr)
    }
    return out


def _beat_track(signals, sr, rng):
    out = {}
    for sig in ("clicks120", "notes", "mix"):
        y = signals[sig]
        env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median, **F64)
        bt = lambda **kw: librosa.beat.beat_track(onset_envelope=env, sr=sr, **kw)  # noqa: E731
        t0, b0 = bt()
        t1, b1 = bt(trim=False)
        t2, b2 = bt(tightness=400)
        t3, b3 = bt(bpm=100.0)
        t4, b4 = bt(start_bpm=90.0, tightness=50)
        tf, bf = librosa.beat.beat_track(y=y, sr=sr)
        out[sig] = {
            "envelope": env,
            "default": {"tempo": t0, "beats": b0},
            "no_trim": {"tempo": t1, "beats": b1},
            "tightness400": {"tempo": t2, "beats": b2},
            "bpm100": {"tempo": t3, "beats": b3},
            "start90_tight50": {"tempo": t4, "beats": b4},
            "default_f32basis": {"tempo": tf, "beats": bf},
            "times": librosa.beat.beat_track(onset_envelope=env, sr=sr, units="time")[1],
        }
    # Time-varying tempo curve and random envelopes: the DP on exact inputs.
    env = np.abs(rng.standard_normal(400)) ** 2
    env[::20] += 4.0
    curve = np.linspace(100.0, 140.0, env.shape[-1])
    out["random"] = {
        "envelope": env,
        "curve": curve,
        "default": librosa.beat.beat_track(onset_envelope=env, sr=sr)[1],
        "tempo": librosa.beat.beat_track(onset_envelope=env, sr=sr)[0],
        "bpm_curve": librosa.beat.beat_track(onset_envelope=env, sr=sr, bpm=curve)[1],
        "bpm_curve_no_trim": librosa.beat.beat_track(onset_envelope=env, sr=sr, bpm=curve, trim=False)[1],
        "bpm_odd_fpb": librosa.beat.beat_track(onset_envelope=env, sr=sr, bpm=111.0)[1],
    }
    return out


def _plp(signals, sr):
    out = {}
    for sig in ("clicks120", "notes"):
        y = signals[sig]
        env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median, **F64)
        out[sig] = {
            "default": librosa.beat.plp(onset_envelope=env, sr=sr),
            "range_60_200_w256": librosa.beat.plp(onset_envelope=env, sr=sr, tempo_min=60, tempo_max=200, win_length=256),
            "unbounded": librosa.beat.plp(onset_envelope=env, sr=sr, tempo_min=None, tempo_max=None),
        }
    return out


def _tempogram_variants(signals, sr):
    """tempogram_ratio / hybrid_tempogram / metrogram on librosa's stored mean
    envelope (``tempogram.<sig>.envelope``)."""
    out = {}
    for sig in ("clicks120", "notes"):
        y = signals[sig]
        env = librosa.onset.onset_strength(y=y, sr=sr, **F64)
        tg = librosa.feature.tempogram(onset_envelope=env, sr=sr)
        ftg = np.abs(librosa.feature.fourier_tempogram(onset_envelope=env, sr=sr))[..., :-1]
        ffreqs = librosa.fourier_tempo_frequencies(sr=sr)
        hyb = librosa.feature.hybrid_tempogram(onset_envelope=env, sr=sr)
        out[sig] = {
            "ratio": librosa.feature.tempogram_ratio(tg=tg, sr=sr),
            "ratio_mean": librosa.feature.tempogram_ratio(tg=tg, sr=sr, aggregate=np.mean),
            "ratio_factors": librosa.feature.tempogram_ratio(
                tg=tg, sr=sr, factors=np.array([1.0, 0.5, 2.0]), start_bpm=90.0, fill_value=-1.0
            ),
            "hybrid_shape": list(hyb.shape),
            "hybrid": hyb[::8, ::4],
            "metrogram": librosa.feature.metrogram(tg=ftg, freqs=ffreqs),
            "metrogram_max": librosa.feature.metrogram(
                tg=ftg, freqs=ffreqs, factors=np.array([0.5, 1.0 / 3.0]), aggregate=np.max
            ),
        }
    return out


def _float32(signals, sr):
    out = {}
    for sig in ("clicks120", "notes", "mix"):
        y = signals[sig].astype(np.float32)
        env = librosa.onset.onset_strength(y=y, sr=sr)
        env_median = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
        tg = librosa.feature.tempogram(onset_envelope=env, sr=sr)
        ftg = librosa.feature.fourier_tempogram(onset_envelope=env, sr=sr)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_nt, beats_nt = librosa.beat.beat_track(y=y, sr=sr, trim=False)
        out[sig] = {
            "dtypes": [str(env.dtype), str(tg.dtype), str(ftg.dtype)],
            "onset_strength": env,
            "onset_strength_median": env_median,
            "tempogram_shape": list(tg.shape),
            "tempogram": tg[::8, ::4],
            "fourier_mag": np.abs(ftg[::8, ::4]),
            "tempo": librosa.feature.tempo(y=y, sr=sr),
            "onset_detect": librosa.onset.onset_detect(y=y, sr=sr),
            "beat_track": {"tempo": tempo, "beats": beats},
            "beat_track_no_trim": {"tempo": tempo_nt, "beats": beats_nt},
            "plp": librosa.beat.plp(y=y, sr=sr),
        }
    return out


def build(signals, sr):
    rng = np.random.default_rng(1234)
    return {
        "onset_strength": _onset_strength(signals, sr),
        "onset_strength_multi": _onset_strength_multi(signals, sr),
        "peak_pick": _peak_pick(rng),
        "onset_detect": _onset_detect(signals, sr),
        "tempogram": _tempograms(signals, sr),
        "tempo": _tempo(signals, sr),
        "beat_track": _beat_track(signals, sr, rng),
        "plp": _plp(signals, sr),
        "tempogram_variants": _tempogram_variants(signals, sr),
        "float32": _float32(signals, sr),
    }
