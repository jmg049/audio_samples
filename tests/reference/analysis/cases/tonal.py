"""Tonal area: filters.chroma, piptrack, estimate_tuning, pitch_tuning,
chroma_stft, util.normalize, and the tuning / octave / key helpers.

Chroma bases are requested as float64 (librosa's default float32 basis only
tests float32 rounding); one chroma_stft case keeps the float32 default to
document that difference. piptrack outputs are sparse, so only the non-zero
entries are stored (as flat indices + values). The detuned test tone is stored
as 16-bit PCM codes so both sides analyse identical samples.

``chroma_stft_float32`` runs chroma_stft on the PCM signals cast to
``np.float32`` with librosa's defaults (float32 STFT and float32 basis); the
Rust side compares ``AudioSamples<f32>`` output with it.
"""

from __future__ import annotations

import librosa
import numpy as np

NAME = "tonal"

PCM_SCALE = 32768.0


def _pcm(x):
    return np.clip(np.round(x * PCM_SCALE), -32768, 32767) / PCM_SCALE


def _chroma_filters():
    cfgs = {
        "default": dict(sr=22050, n_fft=2048),
        "n24_tuning": dict(sr=22050, n_fft=2048, n_chroma=24, tuning=0.3),
        "flat_nonorm_base_a": dict(sr=16000, n_fft=1024, octwidth=None, norm=None, base_c=False),
        "linf_ctroct4_odd": dict(sr=44100, n_fft=2047, norm=np.inf, ctroct=4.0, octwidth=1.0, tuning=-0.2),
        "n7_l1": dict(sr=22050, n_fft=512, n_chroma=7, norm=1.0),
    }
    out = {}
    for name, kw in cfgs.items():
        m = librosa.filters.chroma(dtype=np.float64, **kw)
        step = 8 if m.shape[0] > 12 else 4
        params = dict(kw)
        params.setdefault("n_chroma", 12)
        params.setdefault("tuning", 0.0)
        params.setdefault("ctroct", 5.0)
        params.setdefault("octwidth", 2.0)
        params.setdefault("norm", 2.0)
        params.setdefault("base_c", True)
        if params["norm"] == np.inf:
            params["norm"] = "inf"
        out[name] = {
            "params": params,
            "shape": list(m.shape),
            "col_step": step,
            "subset": m[:, ::step],
            "row_sums": m.sum(axis=1),
        }
    return out


def _sparse(p, m):
    idx = np.flatnonzero(p)
    return {"shape": list(p.shape), "index": idx, "pitch": p.ravel()[idx], "mag": m.ravel()[idx]}


def build(signals, sr) -> dict:
    notes = signals["notes"]
    mix = signals["mix"]
    stereo = np.stack([signals["stereo_l"], signals["stereo_r"]])
    t = np.arange(sr // 2) / sr
    f_det = 440.0 * 2 ** (25 / 1200)
    detuned = _pcm(0.3 * np.sin(2 * np.pi * f_det * t) + 0.15 * np.sin(2 * np.pi * 2 * f_det * t))
    detuned_codes = np.round(detuned * PCM_SCALE).astype(np.int64)

    out = {"chroma_filters": _chroma_filters()}

    # piptrack
    out["piptrack"] = {
        "notes_default": _sparse(*librosa.piptrack(y=notes, sr=sr)),
        "notes_range_thr": _sparse(*librosa.piptrack(y=notes, sr=sr, fmin=100.0, fmax=1500.0, threshold=0.3, n_fft=1024)),
        "notes_ref_mean": _sparse(*librosa.piptrack(y=notes, sr=sr, threshold=1.0, ref=np.mean)),
        "notes_ref_value": _sparse(*librosa.piptrack(y=notes, sr=sr, ref=2.0)),
        # The noise channel has peaks everywhere: keep every 4th frame.
        "stereo_frame_step": 4,
        "stereo_default": _sparse(*(a[..., ::4] for a in librosa.piptrack(y=stereo, sr=sr))),
    }

    # estimate_tuning / pitch_tuning
    S_notes_pow = np.abs(librosa.stft(notes)) ** 2
    out["estimate_tuning"] = {
        "detuned_pcm": detuned_codes,
        "notes": librosa.estimate_tuning(y=notes, sr=sr),
        "notes_res_0001": librosa.estimate_tuning(y=notes, sr=sr, resolution=1e-3),
        "notes_bpo24": librosa.estimate_tuning(y=notes, sr=sr, bins_per_octave=24),
        "notes_power_S": librosa.estimate_tuning(S=S_notes_pow, sr=sr),
        "detuned": librosa.estimate_tuning(y=detuned, sr=sr),
        "detuned_nfft4096": librosa.estimate_tuning(y=detuned, sr=sr, n_fft=4096),
        "mix": librosa.estimate_tuning(y=mix, sr=sr),
        "stereo": librosa.estimate_tuning(y=stereo, sr=sr),
    }
    rng = np.random.default_rng(20261009)
    rand = rng.uniform(50.0, 5000.0, 400)
    cqt25 = librosa.cqt_frequencies(n_bins=24, fmin=55.0, tuning=0.25)
    jitter = librosa.midi_to_hz(np.round(rng.uniform(40, 90, 300)) - 0.13 + rng.normal(0, 0.02, 300))
    out["pitch_tuning"] = {
        "random": {"freqs": rand, "res": 0.01, "bpo": 12, "tuning": librosa.pitch_tuning(rand)},
        "random_bpo36_res005": {
            "freqs": rand, "res": 0.05, "bpo": 36,
            "tuning": librosa.pitch_tuning(rand, resolution=0.05, bins_per_octave=36),
        },
        "cqt_plus25": {"freqs": cqt25, "res": 0.01, "bpo": 12, "tuning": librosa.pitch_tuning(cqt25)},
        "jitter_minus13": {"freqs": jitter, "res": 0.01, "bpo": 12, "tuning": librosa.pitch_tuning(jitter)},
        "jitter_res0001": {
            "freqs": jitter, "res": 0.001, "bpo": 12,
            "tuning": librosa.pitch_tuning(jitter, resolution=0.001),
        },
    }

    # chroma_stft
    f64 = dict(dtype=np.float64)
    out["chroma_stft"] = {
        "notes_default": librosa.feature.chroma_stft(y=notes, sr=sr, **f64),
        "notes_default_float32_basis": librosa.feature.chroma_stft(y=notes, sr=sr),
        "notes_tuning0": librosa.feature.chroma_stft(y=notes, sr=sr, tuning=0.0, **f64),
        "notes_norm_none": librosa.feature.chroma_stft(y=notes, sr=sr, norm=None, **f64),
        "notes_n24": librosa.feature.chroma_stft(y=notes, sr=sr, n_chroma=24, **f64),
        "notes_nfft4096_l2": librosa.feature.chroma_stft(y=notes, sr=sr, n_fft=4096, norm=2, **f64),
        "mix_default": librosa.feature.chroma_stft(y=mix, sr=sr, **f64),
        "stereo_default": librosa.feature.chroma_stft(y=stereo, sr=sr, **f64),
        "detuned_default": librosa.feature.chroma_stft(y=detuned, sr=sr, **f64),
        "notes_magnitude_S": librosa.feature.chroma_stft(S=np.abs(librosa.stft(notes)), sr=sr, **f64),
    }

    notes32 = notes.astype(np.float32)
    out["chroma_stft_float32"] = {
        "notes_tuning0": librosa.feature.chroma_stft(y=notes32, sr=sr, tuning=0.0),
        "notes_default": librosa.feature.chroma_stft(y=notes32, sr=sr),
        "notes_estimated_tuning": float(librosa.estimate_tuning(y=notes32, sr=sr)),
        "mix_tuning0": librosa.feature.chroma_stft(y=mix.astype(np.float32), sr=sr, tuning=0.0),
    }

    # util.normalize
    x = rng.normal(0, 1, (6, 9))
    x[:, 3] = 0.0
    x[2, :] = 1e-6
    out["normalize"] = {
        "input": x,
        "inf_axis0": librosa.util.normalize(x),
        "l2_axis1": librosa.util.normalize(x, norm=2, axis=1),
        "l1_axis0": librosa.util.normalize(x, norm=1, axis=0),
        "l3_axis1": librosa.util.normalize(x, norm=3, axis=1),
        "l0_axis0": librosa.util.normalize(x, norm=0, axis=0),
        "neg_inf_axis1": librosa.util.normalize(x, norm=-np.inf, axis=1),
        "none_axis_l2": librosa.util.normalize(x, norm=2, axis=None),
        "thr_fill_true_l2_axis1": librosa.util.normalize(x, norm=2, axis=1, threshold=1e-3, fill=True),
        "thr_fill_false_inf_axis1": librosa.util.normalize(x, axis=1, threshold=1e-3, fill=False),
        "thr_fill_none_inf_axis1": librosa.util.normalize(x, axis=1, threshold=1e-3),
        "fill_true_inf_axis0": librosa.util.normalize(x, axis=0, fill=True),
    }

    # conversions
    freqs = np.array([8.0, 27.5, 55.0, 100.0, 261.63, 440.0, 1000.0, 4186.0, 11025.0])
    octs = np.linspace(-1.0, 9.0, 21)
    a4 = np.array([415.0, 432.0, 440.0, 443.0, 466.16, 880.0])
    tunings = np.linspace(-0.5, 0.5, 11)
    out["conversions"] = {
        "freqs": freqs,
        "hz_to_octs": librosa.hz_to_octs(freqs),
        "hz_to_octs_t03_b24": librosa.hz_to_octs(freqs, tuning=0.3, bins_per_octave=24),
        "octs": octs,
        "octs_to_hz": librosa.octs_to_hz(octs),
        "octs_to_hz_tm02_b36": librosa.octs_to_hz(octs, tuning=-0.2, bins_per_octave=36),
        "a4": a4,
        "a4_to_tuning": librosa.A4_to_tuning(a4),
        "a4_to_tuning_b36": librosa.A4_to_tuning(a4, bins_per_octave=36),
        "tunings": tunings,
        "tuning_to_a4": librosa.tuning_to_A4(tunings),
        "tuning_to_a4_b24": librosa.tuning_to_A4(tunings, bins_per_octave=24),
        "cqt_default": librosa.cqt_frequencies(n_bins=84, fmin=librosa.note_to_hz("C1")),
        "cqt_fmin_c1": librosa.note_to_hz("C1"),
        "cqt_b36_t025": librosa.cqt_frequencies(n_bins=100, fmin=55.0, bins_per_octave=36, tuning=0.25),
    }

    keys = [
        "C:maj", "A:min", "C:major", "a:minor", "G:maj", "F:maj", "Bb:maj", "Eb:min", "A♯:min", "G♯:maj",
        "F♭:min", "B#:maj", "Cb:maj", "Dbb:min", "E𝄪:phryg", "C#b#:min", "G:loc", "D:dorian", "F:lydian",
        "B:mixolyd", "A:aeolian", "E:phrygian", "C:ionian", "Bb:dor", "F#:locrian", "Gb:maj", "D#:min",
        "Ab:min", "E##:maj", "Fbb:maj", "Cn:min", "D!:maj", "C♮:mix",
    ]
    out["keys"] = {
        k: {
            "unicode": librosa.key_to_notes(k),
            "ascii": librosa.key_to_notes(k, unicode=False),
            "natural": librosa.key_to_notes(k, natural=True),
            "degrees": librosa.key_to_degrees(k),
        }
        for k in keys
    }
    return out
