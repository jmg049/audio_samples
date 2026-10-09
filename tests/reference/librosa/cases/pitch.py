"""Pitch area: librosa.yin, librosa.pyin, and the harmonic helpers
librosa.salience / librosa.interp_harmonics / librosa.f0_harmonics.

YIN and pYIN run on the shared signals (``notes``, ``mix``, the stereo pair),
so only their outputs are stored. The harmonic helpers work on a small
magnitude spectrogram, which is stored as input (65 bins x 9 frames) so the
Rust side does not depend on its own STFT for these cases.

Non-finite values (NaN f0 of unvoiced frames, NaN fill of salience) are stored
as ``null``.
"""

from __future__ import annotations

import librosa
import numpy as np

NAME = "pitch"

FMIN = 65.0
FMAX = 2093.0


def _nullable(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 0:
        return float(a) if np.isfinite(a) else None
    return [_nullable(x) for x in a]


def _yin(signals, sr):
    notes, mix = signals["notes"], signals["mix"]
    stereo = np.stack([signals["stereo_l"], signals["stereo_r"]])
    specs = {
        "notes_default": ("notes", dict()),
        "mix_default": ("mix", dict()),
        "notes_fl1024_nocenter_th02": (
            "notes",
            dict(frame_length=1024, center=False, trough_threshold=0.2),
        ),
        "mix_hop300_reflect": ("mix", dict(hop_length=300, pad_mode="reflect")),
        "mix_narrow_range": ("mix", dict(fmin=150.0, fmax=500.0, frame_length=1024)),
    }
    inputs = {"notes": notes, "mix": mix}
    out = {}
    for name, (sig, kw) in specs.items():
        cfg = dict(fmin=FMIN, fmax=FMAX, frame_length=2048, hop_length=None,
                   trough_threshold=0.1, center=True, pad_mode="constant")
        cfg.update(kw)
        f0 = librosa.yin(inputs[sig], sr=sr, **cfg)
        out[name] = {"signal": sig, "config": cfg, "f0": f0}
    cfg = dict(fmin=FMIN, fmax=FMAX, frame_length=2048, hop_length=None,
               trough_threshold=0.1, center=True, pad_mode="constant")
    out["stereo_default"] = {"signal": "stereo", "config": cfg, "f0": librosa.yin(stereo, sr=sr, **cfg)}
    return out


def _pyin(signals, sr):
    notes, mix = signals["notes"], signals["mix"]
    stereo = np.stack([signals["stereo_l"], signals["stereo_r"]])
    base = dict(
        fmin=FMIN, fmax=FMAX, frame_length=2048, hop_length=None, n_thresholds=100,
        beta_parameters=[2.0, 18.0], boltzmann_parameter=2.0, resolution=0.1,
        max_transition_rate=35.92, switch_prob=0.01, no_trough_prob=0.01,
        center=True, pad_mode="constant", transition_min_prob=1e-4,
    )
    specs = {
        "notes_default": ("notes", dict()),
        "mix_default": ("mix", dict()),
        "notes_exact_viterbi": ("notes", dict(transition_min_prob=None)),
        "notes_custom": (
            "notes",
            dict(frame_length=1024, hop_length=256, center=False, n_thresholds=50,
                 beta_parameters=[1.5, 12.5], boltzmann_parameter=1.0, resolution=0.25,
                 max_transition_rate=20.0, switch_prob=0.05, no_trough_prob=0.02,
                 fmin=100.0, fmax=1500.0),
        ),
        "stereo_default": ("stereo", dict()),
    }
    inputs = {"notes": notes, "mix": mix, "stereo": stereo}
    out = {}
    for name, (sig, kw) in specs.items():
        cfg = dict(base)
        cfg.update(kw)
        call = dict(cfg)
        call["beta_parameters"] = tuple(call["beta_parameters"])
        # fill_na=None: unvoiced frames keep the best-guess pitch, so every value is
        # finite and the NaN-filled default is recovered from voiced_flag.
        f0, flag, prob = librosa.pyin(inputs[sig], sr=sr, fill_na=None, **call)
        out[name] = {"signal": sig, "config": cfg, "f0": f0, "voiced_flag": flag, "voiced_prob": prob}
    return out


def _harmonics(signals, sr):
    y = signals["notes"][:4096]
    n_fft = 128
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_frames = S.shape[1]
    # A time-varying, strictly increasing grid (as from a reassigned spectrogram).
    freqs_tv = freqs[:, None] * (1.0 + 0.002 * np.arange(n_frames))[None, :] + 3.0 * np.arange(n_frames)[None, :]
    f0 = librosa.yin(y, fmin=FMIN, fmax=FMAX, sr=sr, frame_length=1024, hop_length=512)[:n_frames]
    f0 = f0.copy()
    f0[3] = np.nan  # an "unvoiced" frame: f0_harmonics fills it
    h_interp = [0.5, 1.0, 2.0, 3.0]
    h_sal = [1.0, 2.0, 3.0, 4.0]
    w_sal = [1.0, 0.5, 0.33, 0.25]
    h_f0 = [1.0, 2.0, 3.0, 4.5, 30.0]

    out = {
        "S": S,
        "freqs": freqs,
        "freqs_tv": freqs_tv,
        "f0": _nullable(f0),
        "interp_harmonics": {
            "harmonics": h_interp,
            "fixed": librosa.interp_harmonics(S, freqs=freqs, harmonics=h_interp),
            "fixed_fill": librosa.interp_harmonics(S, freqs=freqs, harmonics=h_interp, fill_value=-1.0),
            "time_varying": librosa.interp_harmonics(S, freqs=freqs_tv, harmonics=h_interp),
        },
        "salience": {
            "harmonics": h_sal,
            "weights": w_sal,
            "default": _nullable(librosa.salience(S, freqs=freqs, harmonics=h_sal)),
            "weighted": _nullable(librosa.salience(S, freqs=freqs, harmonics=h_sal, weights=w_sal)),
            "weighted_fill0": librosa.salience(S, freqs=freqs, harmonics=h_sal, weights=w_sal, fill_value=0.0),
            "max_nofilter": librosa.salience(S, freqs=freqs, harmonics=h_sal, aggregate=np.max, filter_peaks=False),
            "sum_nofilter": librosa.salience(S, freqs=freqs, harmonics=h_sal, aggregate=np.sum, filter_peaks=False),
            "median_nofilter": librosa.salience(S, freqs=freqs, harmonics=h_sal, aggregate=np.median, filter_peaks=False),
            "min_nofilter": librosa.salience(S, freqs=freqs, harmonics=h_sal, aggregate=np.min, filter_peaks=False),
            "time_varying": _nullable(librosa.salience(S, freqs=freqs_tv, harmonics=h_sal, weights=w_sal)),
        },
        "f0_harmonics": {
            "harmonics": h_f0,
            "fixed": librosa.f0_harmonics(S, f0=f0, freqs=freqs, harmonics=h_f0),
            "fixed_fill": librosa.f0_harmonics(S, f0=f0, freqs=freqs, harmonics=h_f0, fill_value=-2.0),
            "time_varying": librosa.f0_harmonics(S, f0=f0, freqs=freqs_tv, harmonics=h_f0),
        },
    }
    return out


def build(signals, sr):
    return {
        "fmin": FMIN,
        "fmax": FMAX,
        "yin": _yin(signals, sr),
        "pyin": _pyin(signals, sr),
        "harmonics": _harmonics(signals, sr),
    }
