"""Features area: frame-wise spectral descriptors (centroid, bandwidth, rolloff,
flatness, contrast, poly_features), rms, zero_crossing_rate, pcen, frequency
weightings, perceptual_weighting and reassigned_spectrogram.

Per-frame scalar features are stored whole (1 s signals give 44 frames). Larger
matrices (pcen, perceptual weighting, reassigned spectrogram) are stored as
strided subsets whose steps are stored next to the data. librosa's ``rms``
defaults to float32 arithmetic; it is called with ``dtype=np.float64`` here
(audio_samples computes in f64; the difference is documented). Mel spectrograms
are built with a float64 basis for the same reason.
"""

from __future__ import annotations

import librosa
import numpy as np

NAME = "features"

SIGNALS = ("mix", "noise", "notes")


def _spectral(y, sr):
    """Default-parameter spectral descriptors (librosa.feature.* on y)."""
    return {
        "centroid": librosa.feature.spectral_centroid(y=y, sr=sr),
        "bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr),
        "rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr),
        "flatness": librosa.feature.spectral_flatness(y=y),
        "contrast": librosa.feature.spectral_contrast(y=y, sr=sr),
        "poly1": librosa.feature.poly_features(y=y, sr=sr),
        "rms": librosa.feature.rms(y=y, dtype=np.float64),
        "zcr": librosa.feature.zero_crossing_rate(y),
    }


def _nondefault(y, sr):
    """Non-default parameters, each case named after what it changes."""
    nc = dict(n_fft=1024, hop_length=256, center=False)
    s = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    centroid = librosa.feature.spectral_centroid(S=s, freq=freqs)
    return {
        "centroid_nocenter": librosa.feature.spectral_centroid(y=y, sr=sr, **nc),
        "centroid_hamming_reflect": librosa.feature.spectral_centroid(
            y=y, sr=sr, window="hamming", pad_mode="reflect"
        ),
        "bandwidth_p1": librosa.feature.spectral_bandwidth(y=y, sr=sr, p=1),
        "bandwidth_p3_nonorm": librosa.feature.spectral_bandwidth(y=y, sr=sr, p=3, norm=False),
        # S= path with an explicit centroid (n_fft 1024, hop 256).
        "bandwidth_S_centroid": librosa.feature.spectral_bandwidth(S=s, freq=freqs, centroid=centroid),
        "rolloff_050": librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.5),
        "rolloff_099_nocenter": librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99, **nc),
        "flatness_power1": librosa.feature.spectral_flatness(y=y, power=1.0),
        "flatness_amin": librosa.feature.spectral_flatness(y=y, amin=1e-4),
        "contrast_linear": librosa.feature.spectral_contrast(y=y, sr=sr, linear=True),
        "contrast_b4_f300_q01": librosa.feature.spectral_contrast(
            y=y, sr=sr, n_bands=4, fmin=300.0, quantile=0.1
        ),
        "contrast_nfft1024": librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=1024, hop_length=256),
        "poly0": librosa.feature.poly_features(y=y, sr=sr, order=0),
        "poly2": librosa.feature.poly_features(y=y, sr=sr, order=2),
        "rms_nocenter": librosa.feature.rms(y=y, frame_length=1024, hop_length=256, center=False, dtype=np.float64),
        "rms_reflect_odd": librosa.feature.rms(
            y=y, frame_length=1001, hop_length=300, pad_mode="reflect", dtype=np.float64
        ),
        "rms_S": librosa.feature.rms(S=np.abs(librosa.stft(y)), dtype=np.float64),
        "zcr_nocenter": librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256, center=False),
        "zcr_thresh_max": librosa.feature.zero_crossing_rate(y, threshold=0.05, ref_magnitude=np.max),
        "zcr_zero_neg_pad": librosa.feature.zero_crossing_rate(y, threshold=0.01, zero_pos=False, pad=True),
    }


def _pcen(signals, sr):
    mix = signals["mix"]
    stereo = np.stack([signals["stereo_l"], signals["stereo_r"]])
    mel = librosa.feature.melspectrogram(y=mix, sr=sr, dtype=np.float64) * (2**31)
    mel_st = librosa.feature.melspectrogram(y=stereo, sr=sr, dtype=np.float64) * (2**31)
    mag = np.abs(librosa.stft(signals["notes"]))
    step = 8
    return {
        "row_step": step,
        "default": librosa.pcen(mel, sr=sr)[::step],
        "max_size3": librosa.pcen(mel, sr=sr, max_size=3)[::step],
        "power0": librosa.pcen(mel, sr=sr, power=0.0)[::step],
        "bias0": librosa.pcen(mel, sr=sr, bias=0.0, gain=0.8)[::step],
        "b_explicit_hop256": librosa.pcen(mel, sr=sr, hop_length=256, b=0.2, time_constant=0.1)[::step],
        "stereo": librosa.pcen(mel_st, sr=sr)[:, ::step],
        "stft_notes_mag_step": 32,
        "stft_notes": librosa.pcen(mag * (2**31), sr=sr)[::32],
    }


def _weighting(signals, sr):
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)[::8]
    special = np.array([0.0, 1.0, 10.0, 31.5, 100.0, 1000.0, 1018.7, 3136.5, 10000.0, 20000.0, 30000.0])
    out = {"freqs": freqs, "special": special}
    for k in "ABCD":
        out[k] = librosa.frequency_weighting(freqs, kind=k)
        out[k + "_special_nomin"] = librosa.frequency_weighting(special[1:], kind=k, min_db=None)
        out[k + "_special_min40"] = librosa.frequency_weighting(special, kind=k, min_db=-40.0)
    out["multi_ZAC"] = librosa.multi_frequency_weighting(freqs)
    out["multi_DBA_min60"] = librosa.multi_frequency_weighting(freqs, kinds="DBA", min_db=-60.0)

    mix = signals["mix"]
    stereo = np.stack([signals["stereo_l"], signals["stereo_r"]])
    p = np.abs(librosa.stft(mix)) ** 2
    pst = np.abs(librosa.stft(stereo)) ** 2
    all_freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    out["perceptual"] = {
        "bin_step": 16,
        "A_default": librosa.perceptual_weighting(p, all_freqs)[::16],
        "C_ref_max": librosa.perceptual_weighting(p, all_freqs, kind="C", ref=np.max)[::16],
        "D_no_top_db": librosa.perceptual_weighting(p, all_freqs, kind="D", top_db=None)[::16],
        "A_stereo": librosa.perceptual_weighting(pst, all_freqs)[:, ::16],
    }
    return out


def _nan_split(a):
    """JSON cannot hold NaN: return (values with NaN -> 0, NaN mask)."""
    mask = np.isnan(a)
    return np.where(mask, 0.0, a), mask


def _reassigned(signals, sr):
    out = {"bin_step": 16, "frame_step": 3}
    cases = {
        "default": (signals["mix"], dict()),
        "fill_nan_noclip": (signals["notes"], dict(fill_nan=True, clip=False)),
        "nocenter_n1024_hamming": (
            signals["mix"],
            dict(n_fft=1024, hop_length=256, center=False, window="hamming", ref_power=1e-3),
        ),
        "odd_nfft_winlen": (signals["mix"], dict(n_fft=1023, win_length=800, hop_length=200)),
        "freq_only_ref_max": (signals["notes"], dict(reassign_times=False, ref_power=np.max)),
        "time_only": (signals["mix"], dict(reassign_frequencies=False)),
    }
    for name, (y, kw) in cases.items():
        freqs, times, mags = librosa.reassigned_spectrogram(y=y, sr=sr, **kw)
        sl = (slice(None, None, 16), slice(None, None, 3))
        f, fmask = _nan_split(freqs[sl])
        t, tmask = _nan_split(times[sl])
        out[name] = {
            "params": {k: (v if not callable(v) else "max") for k, v in kw.items()},
            "shape": list(mags.shape),
            "freqs": f,
            "freqs_nan": fmask,
            "times": t,
            "times_nan": tmask,
            "mags": mags[sl],
        }
    return out


def build(signals, sr):
    out = {"spectral": {}, "nondefault": {}}
    for name in SIGNALS:
        out["spectral"][name] = _spectral(signals[name], sr)
    stereo = np.stack([signals["stereo_l"], signals["stereo_r"]])
    out["spectral"]["stereo"] = _spectral(stereo, sr)
    for name in ("mix", "noise"):
        out["nondefault"][name] = _nondefault(signals[name], sr)
    out["pcen"] = _pcen(signals, sr)
    out["weighting"] = _weighting(signals, sr)
    out["reassigned"] = _reassigned(signals, sr)
    return out
