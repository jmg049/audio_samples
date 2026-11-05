"""References for options reached through array forms and explicit parameters: wavelet
bandwidths, tuned variable-Q bins and chroma, smoothing, custom axes and interpolants."""

from __future__ import annotations

import numpy as np

import librosa

NAME = "options"


def _c(a):
    a = np.asarray(a)
    return {"re": a.real.tolist(), "im": a.imag.tolist()}


def build(signals, sr) -> dict:
    y = signals["mix"][: sr // 2].astype(np.float64)
    out: dict = {"y": y.tolist(), "sr": sr}

    freqs = 110.0 * 2.0 ** (np.arange(24) / 12.0)
    alpha = 0.05 + 0.002 * np.arange(24)
    lengths, cutoff = librosa.filters.wavelet_lengths(freqs=freqs, sr=sr, alpha=alpha)
    filters, _ = librosa.filters.wavelet(freqs=freqs, sr=sr, alpha=alpha, dtype=np.complex128)
    out["wavelet_alpha"] = {
        "freqs": freqs.tolist(),
        "alpha": alpha.tolist(),
        "lengths": lengths.tolist(),
        "cutoff": float(cutoff),
        "filters": _c(filters),
    }

    ratios = librosa.plimit_intervals(primes=[3, 5], bins_per_octave=12)
    kw = dict(sr=sr, n_bins=48, fmin=110.0, gamma=0.0, tuning=0.0, intervals=ratios)
    out["vqt_intervals"] = {"ratios": ratios.tolist(), "v": _c(librosa.vqt(y, **kw))}
    out["vqt_named"] = {
        name: _c(librosa.vqt(y, sr=sr, n_bins=24, fmin=220.0, gamma=0.0, tuning=0.0, intervals=name))
        for name in ("pythagorean", "ji3", "ji5", "ji7")
    }

    C = np.abs(librosa.cqt(y, sr=sr, fmin=55.0, n_bins=72, bins_per_octave=36, tuning=0.0))
    window = np.array([0.25, 0.5, 0.25])
    out["chroma"] = {
        "C": C.tolist(),
        "window": window.tolist(),
        "cq_to_chroma_window": librosa.filters.cq_to_chroma(72, bins_per_octave=36, fmin=55.0, window=window).tolist(),
        "chroma_cqt": librosa.feature.chroma_cqt(C=C, fmin=55.0, bins_per_octave=36, window=window).tolist(),
        "chroma_cens": librosa.feature.chroma_cens(C=C, fmin=55.0, bins_per_octave=36, window=window).tolist(),
        "chroma_vqt": librosa.feature.chroma_vqt(
            V=C, sr=sr, fmin=55.0, intervals="equal", bins_per_octave=36
        ).tolist(),
    }

    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    harmonics = [1.0, 2.0, 3.0]
    f0 = np.linspace(100.0, 400.0, S.shape[-1])
    out["harmonics"] = {
        "S": S[:64, :32].tolist(),
        "freqs": freqs[:64].tolist(),
        "harmonics": harmonics,
        "f0": f0[:32].tolist(),
        "kinds": {
            kind: {
                "interp": librosa.interp_harmonics(
                    S[:64, :32], freqs=freqs[:64], harmonics=harmonics, kind=kind, axis=0
                ).tolist(),
                "f0": librosa.f0_harmonics(
                    S[:64, :32], freqs=freqs[:64], f0=f0[:32], harmonics=harmonics, kind=kind, axis=0
                ).tolist(),
                "salience": librosa.salience(
                    S[:64, :32], freqs=freqs[:64], harmonics=harmonics, kind=kind, fill_value=0.0
                ).tolist(),
            }
            for kind in ("linear", "nearest", "cubic")
        },
    }

    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
    tg = librosa.feature.tempogram(onset_envelope=env, sr=sr, hop_length=256, win_length=128)
    tg_freqs = np.linspace(1.0, 640.0, tg.shape[0])
    bpm = np.full(tg.shape[-1], 120.0)
    out["tempogram_ratio"] = {
        "tg": tg.tolist(),
        "bpm": bpm.tolist(),
        "custom_freqs": tg_freqs.tolist(),
        "result": librosa.feature.tempogram_ratio(
            tg=tg, sr=sr, bpm=bpm, freqs=tg_freqs, hop_length=256
        ).tolist(),
    }

    def feature(y, sr, n_fft, hop_length):
        return np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)) ** 2

    S_db = librosa.power_to_db(feature(y, sr, 1024, 256))
    ref = np.roll(S_db, 1, axis=0)
    out["onset_options"] = {
        "power_feature": librosa.onset.onset_strength(y=y, sr=sr, n_fft=1024, hop_length=256, feature=feature).tolist(),
        "with_ref": librosa.onset.onset_strength(S=S_db, sr=sr, hop_length=256, n_fft=1024, ref=ref).tolist(),
    }

    freqs_r, times_r, mags_r = librosa.reassigned_spectrogram(y, sr=sr, n_fft=512, hop_length=128)
    S_c = librosa.stft(y, n_fft=512, hop_length=128)
    freqs_s, times_s, mags_s = librosa.reassigned_spectrogram(
        y, sr=sr, S=S_c * 2.0, n_fft=512, hop_length=128, fill_nan=True
    )
    out["reassigned"] = {
        "freqs": np.nan_to_num(freqs_s[:64, :16]).tolist(),
        "times": np.nan_to_num(times_s[:64, :16]).tolist(),
        "mags": mags_s[:64, :16].tolist(),
    }
    stereo = np.stack([y, y[::-1]])
    short = y[: y.size // 2]
    env_y = signals["clicks120"].astype(np.float64)
    out["forms"] = {
        "to_mono": librosa.to_mono(stereo).tolist(),
        "to_mono_sum": librosa.to_mono(y, short, pad=True, norm=False).tolist(),
        "to_stereo": librosa.to_stereo(left=y, right=short, pad=True).tolist(),
        "to_multi": librosa.to_multi(y, short, stereo, pad=False, norm=False).tolist(),
        "resample_scale": librosa.resample(y, orig_sr=sr, target_sr=8000, scale=True).tolist(),
        "pitch_shift_scale": librosa.effects.pitch_shift(y, sr=sr, n_steps=2.0, scale=True).tolist(),
        "frame_axis0": librosa.util.frame(np.stack([y[:64], y[:64]], axis=1), frame_length=8, hop_length=4, axis=0).tolist(),
        "frame_axis1": librosa.util.frame(y[:64], frame_length=8, hop_length=4, axis=-1).tolist(),
        "duration_S": librosa.get_duration(S=np.zeros((513, 40)), sr=sr, n_fft=1024, hop_length=256),
        "duration_S_uncentred": librosa.get_duration(S=np.zeros((513, 40)), sr=sr, n_fft=1024, hop_length=256, center=False),
        "onset_mask": librosa.onset.onset_detect(y=env_y, sr=sr, sparse=False).astype(float).tolist(),
        "beat_mask": librosa.beat.beat_track(y=env_y, sr=sr, sparse=False)[1].astype(float).tolist(),
    }
    return out
