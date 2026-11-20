"""Constant-Q area: cqt / vqt / pseudo_cqt / hybrid_cqt / icqt, cq_to_chroma,
chroma_cqt / chroma_cens / chroma_vqt and tonnetz.

librosa builds its wavelet filters in complex64 (``filters.wavelet`` default
dtype) even for float64 input, so besides the stock outputs one case is computed
with the filters forced to complex128 (``cqt_f64_basis``): it isolates the
algorithm from single-precision rounding. Large matrices are stored as strided
frame subsets (``frame_step``); ``mix`` default CQT is stored in full because the
icqt case and the per-octave error analysis use it.

``tuning=None`` cases also store librosa's estimated tuning, so the Rust side can
reproduce the rest of the pipeline with that value.

``float32`` holds results for the PCM signals cast to ``np.float32``: librosa
then computes the CQT in complex64 (wavelets, FFTs and resampling in single
precision). The Rust side compares ``AudioSamples<f32>`` output with them.
"""

from __future__ import annotations

import functools

import librosa
import librosa.filters
import numpy as np

NAME = "constant_q"

CQT_CASES = {
    "default": dict(),
    "fmin65_bpo24_n48": dict(fmin=65.0, bins_per_octave=24, n_bins=48),
    "filter_scale2": dict(filter_scale=2, n_bins=72),
    "scale_false": dict(scale=False),
    "norm2": dict(norm=2),
    "early_downsample": dict(n_bins=24),
    "hop256_reflect": dict(hop_length=256, pad_mode="reflect", n_bins=60),
    "sparsity0": dict(sparsity=0.0, n_bins=36, fmin=110.0),
}

VQT_CASES = {
    "gamma_none": dict(),
    "gamma0": dict(gamma=0),
    "gamma10_bpo24": dict(gamma=10.0, bins_per_octave=24, n_bins=120),
}

FRAME_STEP = 3


def _sub(x, step=FRAME_STEP):
    return x[..., ::step]


def _f64_wavelet():
    """Context: librosa.filters.wavelet forced to complex128."""
    orig = librosa.filters.wavelet

    class _Ctx:
        def __enter__(self):
            librosa.filters.wavelet = functools.partial(orig, dtype=np.complex128)

        def __exit__(self, *exc):
            librosa.filters.wavelet = orig

    return _Ctx()


def build(signals, sr):
    mix = signals["mix"]
    notes = signals["notes"]
    stereo = np.stack([signals["stereo_l"], signals["stereo_r"]])
    out: dict = {"frame_step": FRAME_STEP}

    c_default = librosa.cqt(mix, sr=sr)
    with _f64_wavelet():
        c_f64 = librosa.cqt(mix, sr=sr)
    out["cqt_full_mix_default"] = c_default
    out["cqt_f64_basis_mix_default"] = c_f64

    cqt = {}
    for name, kw in CQT_CASES.items():
        if name == "default":
            continue
        c = librosa.cqt(mix, sr=sr, **kw)
        params = {k: v for k, v in kw.items()}
        cqt[name] = {"params": params, "shape": list(c.shape), "subset": _sub(c)}
    c = librosa.cqt(notes, sr=sr)
    cqt["notes_default"] = {"params": {}, "shape": list(c.shape), "subset": _sub(c)}
    c = librosa.cqt(stereo, sr=sr)
    cqt["stereo_default"] = {"params": {}, "shape": list(c.shape), "subset": _sub(c)}
    out["cqt"] = cqt

    vqt = {}
    for name, kw in VQT_CASES.items():
        v = librosa.vqt(mix, sr=sr, **kw)
        vqt[name] = {"params": kw, "shape": list(v.shape), "subset": _sub(v)}
    out["vqt"] = vqt

    out["pseudo_cqt"] = {
        "default": _sub(librosa.pseudo_cqt(mix, sr=sr)),
        "fmin220_scale_false": _sub(librosa.pseudo_cqt(mix, sr=sr, fmin=220.0, n_bins=48, scale=False)),
    }
    out["hybrid_cqt"] = {
        "default": _sub(librosa.hybrid_cqt(mix, sr=sr)),
        "notes_bpo36": _sub(librosa.hybrid_cqt(notes, sr=sr, bins_per_octave=36, n_bins=252)),
    }

    # icqt of librosa's own CQT (stored in full above): isolates the inverse.
    y_inv = librosa.icqt(c_default, sr=sr, length=len(mix))
    y_inv_noscale = librosa.icqt(librosa.cqt(mix, sr=sr, scale=False), sr=sr, scale=False, length=len(mix))
    out["icqt"] = {
        "step": 4,
        "default": y_inv[::4],
        "noscale": y_inv_noscale[::4],
    }

    out["cq_to_chroma"] = {
        "default_84": librosa.filters.cq_to_chroma(84, dtype=np.float64),
        "bpo36_108": librosa.filters.cq_to_chroma(108, bins_per_octave=36, dtype=np.float64),
        "fmin65_bpo24_50": librosa.filters.cq_to_chroma(50, bins_per_octave=24, fmin=65.0, dtype=np.float64),
        "base_a_fmin100": librosa.filters.cq_to_chroma(40, fmin=100.0, base_c=False, dtype=np.float64),
        "n24_bpo24": librosa.filters.cq_to_chroma(48, bins_per_octave=24, n_chroma=24, dtype=np.float64),
    }

    tuning_notes = float(librosa.estimate_tuning(y=notes, sr=sr, bins_per_octave=36))
    tuning_mix = float(librosa.estimate_tuning(y=mix, sr=sr, bins_per_octave=36))
    out["estimated_tuning"] = {"notes": tuning_notes, "mix": tuning_mix}
    out["chroma_cqt"] = {
        "notes_tuning0": librosa.feature.chroma_cqt(y=notes, sr=sr, tuning=0.0),
        "notes_tuning_none": librosa.feature.chroma_cqt(y=notes, sr=sr),
        "mix_tuning0": librosa.feature.chroma_cqt(y=mix, sr=sr, tuning=0.0),
        "notes_hybrid_l2_bpo12": librosa.feature.chroma_cqt(
            y=notes, sr=sr, tuning=0.0, cqt_mode="hybrid", norm=2, bins_per_octave=12
        ),
        "stereo_tuning0": librosa.feature.chroma_cqt(y=stereo, sr=sr, tuning=0.0),
    }
    out["chroma_cens"] = {
        "notes_tuning0": librosa.feature.chroma_cens(y=notes, sr=sr, tuning=0.0),
        "notes_tuning_none": librosa.feature.chroma_cens(y=notes, sr=sr),
        "mix_tuning0_win10": librosa.feature.chroma_cens(y=mix, sr=sr, tuning=0.0, win_len_smooth=10),
    }
    out["chroma_vqt"] = {
        "mix_equal": librosa.feature.chroma_vqt(y=mix, sr=sr, intervals="equal"),
        "notes_gamma_none": librosa.feature.chroma_vqt(y=notes, sr=sr, intervals="equal", gamma=None),
    }
    mix32 = mix.astype(np.float32)
    notes32 = notes.astype(np.float32)
    out["float32"] = {
        "cqt_mix_default": librosa.cqt(mix32, sr=sr),
        "chroma_cqt_notes_tuning0": librosa.feature.chroma_cqt(y=notes32, sr=sr, tuning=0.0),
        "chroma_cqt_mix_tuning0": librosa.feature.chroma_cqt(y=mix32, sr=sr, tuning=0.0),
    }
    out["tonnetz"] = {
        "notes_tuning0": librosa.feature.tonnetz(y=notes, sr=sr, tuning=0.0),
        "notes_tuning_none": librosa.feature.tonnetz(y=notes, sr=sr),
    }
    return out
