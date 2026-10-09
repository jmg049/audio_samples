"""Spectrum area: windows, mel filterbanks, unit conversions, STFT/ISTFT, dB,
melspectrogram, MFCC, delta, stack_memory, window_sumsquare.

Large matrices are stored as strided subsets (``[::bin_step, ::frame_step]``) to
keep the fixture small; the step sizes are stored next to the data and the Rust
test applies the same slicing. Mel bases are requested as float64 (librosa's
default float32 basis would only test float32 rounding).
"""

from __future__ import annotations

import librosa
import numpy as np

NAME = "spectrum"


def _windows():
    specs = {
        "hann": "hann",
        "hamming": "hamming",
        "blackman": "blackman",
        "blackmanharris": "blackmanharris",
        "boxcar": "boxcar",
        "bartlett": "bartlett",
        "triang": "triang",
        "kaiser": ("kaiser", 8.6),
        "gaussian": ("gaussian", 7.0),
    }
    out = {}
    for name, spec in specs.items():
        for n in (16, 17):
            for periodic in (True, False):
                key = f"{name}_{n}_{'periodic' if periodic else 'symmetric'}"
                out[key] = librosa.filters.get_window(spec, n, fftbins=periodic)
    return out


def _bandwidth_formula(win):
    n = len(win)
    return n * np.sum(win**2) / (np.sum(win) ** 2 + np.finfo(win.dtype).tiny)


def _mel_bases():
    cfgs = {
        "default": dict(sr=22050, n_fft=2048),
        "htk": dict(sr=22050, n_fft=2048, htk=True),
        "range_nonorm": dict(sr=16000, n_fft=512, n_mels=40, fmin=50.0, fmax=7000.0, norm=None),
        "odd_htk_nonorm": dict(sr=44100, n_fft=2047, n_mels=64, fmin=20.0, fmax=10000.0, htk=True, norm=None),
        "l1": dict(sr=22050, n_fft=1024, n_mels=32, norm=1.0),
        "linf": dict(sr=22050, n_fft=1024, n_mels=32, norm=np.inf),
    }
    out = {}
    for name, kw in cfgs.items():
        m = librosa.filters.mel(dtype=np.float64, **kw)
        params = {k: v for k, v in kw.items()}
        params.setdefault("n_mels", 128)
        params.setdefault("fmin", 0.0)
        params.setdefault("fmax", None)
        params.setdefault("htk", False)
        params.setdefault("norm", "slaney")
        if params["norm"] == np.inf:
            params["norm"] = "inf"
        out[name] = {
            "params": params,
            "shape": list(m.shape),
            "row_step": 8,
            "col_step": 4,
            "subset": m[::8, ::4],
            "row_sums": m.sum(axis=1),
            "row_max": m.max(axis=1),
        }
    return out


def _conversions(sr):
    freqs = np.array([0.0, 20.0, 100.0, 440.0, 999.0, 1000.0, 1001.0, 4000.0, 11025.0, 22050.0])
    mels = np.linspace(0.0, 4000.0, 21)
    mels_slaney = np.linspace(0.0, 60.0, 21)
    times = np.linspace(-0.5, 3.0, 301)
    samples = np.arange(-3000, 3000, 37)
    frames = np.arange(-5, 50)
    return {
        "freqs": freqs,
        "hz_to_mel_slaney": librosa.hz_to_mel(freqs),
        "hz_to_mel_htk": librosa.hz_to_mel(freqs, htk=True),
        "mels_htk_in": mels,
        "mel_to_hz_htk": librosa.mel_to_hz(mels, htk=True),
        "mels_slaney_in": mels_slaney,
        "mel_to_hz_slaney": librosa.mel_to_hz(mels_slaney),
        "mel_frequencies_slaney": librosa.mel_frequencies(),
        "mel_frequencies_htk": librosa.mel_frequencies(htk=True),
        "mel_frequencies_40_slaney": librosa.mel_frequencies(40, fmin=133.0, fmax=6854.0),
        "mel_frequencies_40_htk": librosa.mel_frequencies(40, fmin=133.0, fmax=6854.0, htk=True),
        "times": times,
        "time_to_frames": librosa.time_to_frames(times, sr=sr, hop_length=512),
        "time_to_frames_nfft": librosa.time_to_frames(times, sr=sr, hop_length=512, n_fft=2048),
        "time_to_samples": librosa.time_to_samples(times, sr=sr),
        "samples": samples,
        "samples_to_frames": librosa.samples_to_frames(samples, hop_length=512, n_fft=2047),
        "frames": frames,
        "frames_to_time_nfft": librosa.frames_to_time(frames, sr=sr, hop_length=512, n_fft=2048),
        "frames_to_samples": librosa.frames_to_samples(frames, hop_length=300),
    }


STFT_CASES = {
    "default": dict(),
    "nocenter": dict(n_fft=1024, hop_length=256, center=False),
    "winlen_hamming": dict(n_fft=2048, hop_length=300, win_length=1000, window="hamming"),
    "reflect": dict(n_fft=512, hop_length=128, pad_mode="reflect"),
    "odd_edge": dict(n_fft=1023, hop_length=256, pad_mode="edge"),
}


def _stft(y):
    out = {}
    for name, kw in STFT_CASES.items():
        d = librosa.stft(y, **kw)
        out[name] = {
            "params": kw,
            "shape": list(d.shape),
            "bin_step": 24,
            "frame_step": 3,
            "subset": d[::24, ::3],
        }
    return out


def build(signals, sr):
    mix = signals["mix"]
    stereo = np.stack([signals["stereo_l"], signals["stereo_r"]])
    short = mix[:3000]

    out = {
        "windows": _windows(),
        # librosa.filters.window_bandwidth returns a hard-coded table for named windows
        # (ignoring n); "formula" is its definition evaluated at n=1000 on the periodic
        # window, which is what audio_samples computes.
        "window_bandwidth": {
            w: {
                "table": librosa.filters.window_bandwidth(w),
                "formula": _bandwidth_formula(librosa.filters.get_window(w, 1000)),
            }
            for w in ("hann", "hamming", "blackman", "boxcar")
        },
        "window_sumsquare": {
            "step": 4,
            "hann_winlen": librosa.filters.window_sumsquare(
                window="hann", n_frames=10, hop_length=256, win_length=800, n_fft=1024, dtype=np.float64
            )[::4],
            "hamming": librosa.filters.window_sumsquare(
                window="hamming", n_frames=6, hop_length=512, n_fft=2048, dtype=np.float64
            )[::4],
        },
        "mel": _mel_bases(),
        "convert": _conversions(sr),
        "stft": _stft(mix),
    }

    d_st = librosa.stft(stereo)
    out["stft_stereo"] = {"shape": list(d_st.shape), "bin_step": 64, "frame_step": 3, "subset": d_st[:, ::64, ::3]}

    # ISTFT of a modified (not round-trip) spectrum: a smooth per-bin gain.
    d = librosa.stft(short)
    gain = 1.0 / (1.0 + np.arange(d.shape[0]) / 100.0)
    dm = d * gain[:, None]
    d2 = librosa.stft(short, n_fft=1024, hop_length=256, win_length=800, center=False)
    out["istft"] = {
        "length_given_step2": librosa.istft(dm, length=len(short))[::2],
        "no_length_step4": librosa.istft(dm)[::4],
        "no_length_len": len(librosa.istft(dm)),
        "nocenter_winlen_step2": librosa.istft(
            d2 * gain[: d2.shape[0], None], n_fft=1024, hop_length=256, win_length=800, center=False
        )[::2],
        "longer_len": len(short) + 1000,
        "longer": librosa.istft(dm, length=len(short) + 1000)[-1500:],
    }

    # melspectrogram / dB
    mel = librosa.feature.melspectrogram(y=mix, sr=sr, dtype=np.float64)
    mel_f32 = librosa.feature.melspectrogram(y=mix, sr=sr)
    mel_htk = librosa.feature.melspectrogram(
        y=mix, sr=sr, n_mels=64, fmin=50.0, fmax=8000.0, htk=True, power=1.0, dtype=np.float64
    )
    mel_st = librosa.feature.melspectrogram(y=stereo, sr=sr, dtype=np.float64)
    out["melspectrogram"] = {
        "row_step": 4,
        "default": mel[::4],
        "default_float32_basis": mel_f32[::4],
        "htk_power1": mel_htk[::4],
        "stereo_row_step": 8,
        "stereo": mel_st[:, ::8, :],
    }
    # dB: inputs are stored subsets (mel[::4] above, |stft default subset|), so the
    # Rust side feeds librosa's exact input.
    mel_in = mel[::4]
    mag = np.abs(out["stft"]["default"]["subset"])
    out["db"] = {
        "power_default": librosa.power_to_db(mel_in),
        "power_ref_max": librosa.power_to_db(mel_in, ref=np.max),
        "power_no_top_db": librosa.power_to_db(mel_in, top_db=None),
        "power_stereo_ref_max_row_step": 8,
        "power_stereo_ref_max": librosa.power_to_db(mel_st, ref=np.max)[:, ::8, :],
        "amplitude_ref_max": librosa.amplitude_to_db(mag, ref=np.max),
        "amplitude_default": librosa.amplitude_to_db(mag),
        "db_in": np.linspace(-120.0, 20.0, 29),
        "db_to_power_ref2": librosa.db_to_power(np.linspace(-120.0, 20.0, 29), ref=2.0),
        "db_to_amplitude_ref2": librosa.db_to_amplitude(np.linspace(-120.0, 20.0, 29), ref=2.0),
    }

    # MFCC (mel basis in float64 via **kwargs -> melspectrogram -> filters.mel)
    f64 = dict(dtype=np.float64)
    mfcc = librosa.feature.mfcc(y=mix, sr=sr, **f64)
    out["mfcc"] = {
        "default": mfcc,
        "lifter22": librosa.feature.mfcc(y=mix, sr=sr, lifter=22, **f64),
        "dct3": librosa.feature.mfcc(y=mix, sr=sr, dct_type=3, **f64),
        "dct1": librosa.feature.mfcc(y=mix, sr=sr, dct_type=1, **f64),
        "norm_none": librosa.feature.mfcc(y=mix, sr=sr, norm=None, **f64),
        "htk_13_40": librosa.feature.mfcc(y=mix, sr=sr, n_mfcc=13, n_mels=40, htk=True, **f64),
        "stereo": librosa.feature.mfcc(y=stereo, sr=sr, **f64),
    }

    # delta / stack_memory on the stored default MFCC (exact input for the Rust side)
    out["delta"] = {
        "order1": librosa.feature.delta(mfcc),
        "order2": librosa.feature.delta(mfcc, order=2),
        "width5": librosa.feature.delta(mfcc, width=5),
        "nearest": librosa.feature.delta(mfcc, mode="nearest"),
        "mirror_w7_o2": librosa.feature.delta(mfcc, width=7, order=2, mode="mirror"),
        "wrap": librosa.feature.delta(mfcc, mode="wrap"),
        "constant": librosa.feature.delta(mfcc, mode="constant"),
        "axis0": librosa.feature.delta(mfcc, axis=0, width=5),
    }
    out["stack_memory"] = {
        "n3_d2": librosa.feature.stack_memory(mfcc[:5], n_steps=3, delay=2),
        "n2_dm1": librosa.feature.stack_memory(mfcc[:5], n_steps=2, delay=-1),
    }
    out["frame"] = {"frames": librosa.util.frame(mix[:3000], frame_length=512, hop_length=256)[::8, ::3]}
    return out
