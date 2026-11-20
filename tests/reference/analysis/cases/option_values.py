"""References for the option values reached through ``*_with`` and ``*_slices`` functions:
fractional ``magphase`` powers, time-varying ``freq``, slice channels and indices, the
per-bin metrogram, sparse and custom ``nn_filter``, custom ``trim`` and ``split``, and the
quality of a random-phase ``griffinlim_cqt``."""

from __future__ import annotations

import numpy as np

import librosa

NAME = "option_values"


def _c(a):
    a = np.asarray(a)
    return {"re": a.real.tolist(), "im": a.imag.tolist()}



def build(signals, sr) -> dict:
    y = signals["mix"][: sr // 2].astype(np.float64)
    out: dict = {"y": y.tolist(), "sr": sr}

    # magphase: a fractional power.
    D = librosa.stft(y, n_fft=256, hop_length=128)[:20, :10]
    D[3, 2] = 0.0
    out["magphase"] = {
        "D": _c(D),
        **{f"mag_{str(p).replace('.', '_').replace('-', 'm')}": np.nan_to_num(librosa.magphase(D, power=p)[0], posinf=-1.0).tolist() for p in (0.5, 1.5, 0.25, -0.5)},
        "phase": _c(librosa.magphase(D, power=1.5)[1]),
    }

    # Time-varying frequencies: reassigned frequencies with the NaNs filled.
    freqs, _, mags = librosa.reassigned_spectrogram(y, sr=sr, n_fft=256, hop_length=128, fill_nan=True)
    S = np.abs(mags)
    freq = np.asarray(freqs, dtype=np.float64)
    out["time_varying_freq"] = {
        "S": S.tolist(),
        "freq": freq.tolist(),
        "centroid": librosa.feature.spectral_centroid(S=S, freq=freq).tolist(),
        "bandwidth": librosa.feature.spectral_bandwidth(S=S, freq=freq).tolist(),
        "bandwidth_p3_nonorm": librosa.feature.spectral_bandwidth(S=S, freq=freq, p=3, norm=False).tolist(),
        "rolloff": librosa.feature.spectral_rolloff(S=S, freq=freq, roll_percent=0.7).tolist(),
        "poly2": librosa.feature.poly_features(S=S, freq=freq, order=2).tolist(),
    }

    # Slices: channels of onset_strength_multi and indices of sync.
    Sdb = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=24, n_fft=512, hop_length=128))
    slices = [slice(0, 6), slice(4, 20, 3), slice(20, None), slice(10, 2, -2)]
    out["slices"] = {
        "Sdb": Sdb.tolist(),
        "onset_mean": librosa.onset.onset_strength_multi(S=Sdb, n_fft=512, hop_length=128, channels=slices).tolist(),
        "onset_median": librosa.onset.onset_strength_multi(S=Sdb, n_fft=512, hop_length=128, channels=slices, aggregate=np.median).tolist(),
        "onset_max_detrend": librosa.onset.onset_strength_multi(S=Sdb, n_fft=512, hop_length=128, channels=slices, aggregate=np.max, detrend=True, lag=2).tolist(),
        "sync_mean": librosa.util.sync(Sdb, [slice(0, 5), slice(3, 12, 2), slice(30, 8, -3)]).tolist(),
        "sync_median": librosa.util.sync(Sdb, [slice(0, 5), slice(3, 12, 2), slice(30, 8, -3)], aggregate=np.median).tolist(),
        "sync_max": librosa.util.sync(Sdb, [slice(0, 5), slice(3, 12, 2), slice(30, 8, -3)], aggregate=np.max).tolist(),
    }

    # Per-bin metrogram.
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=128)
    tg = librosa.feature.tempogram(onset_envelope=env, sr=sr, hop_length=128, win_length=48)[1:]
    bpm = librosa.tempo_frequencies(tg.shape[0] + 1, sr=sr, hop_length=128)[1:]
    out["metrogram"] = {
        "tg": tg.tolist(),
        "freqs": bpm.tolist(),
        "per_bin": librosa.feature.metrogram(tg=tg, freqs=bpm, aggregate=None).tolist(),
        "per_bin_factors": librosa.feature.metrogram(tg=tg, freqs=bpm, factors=np.array([0.5, 0.25]), aggregate=None, fill_value=0.25).tolist(),
    }

    # nn_filter on a sparse matrix.
    data = np.abs(librosa.stft(y, n_fft=64, hop_length=256))
    rec = librosa.segment.recurrence_matrix(data, k=5, mode="affinity", sparse=True).tocsc()
    out["nn_filter"] = {
        "S": data.tolist(),
        "indptr": rec.indptr.tolist(),
        "indices": rec.indices.tolist(),
        "values": rec.data.tolist(),
        "mean": librosa.decompose.nn_filter(data, rec=rec).tolist(),
        "median": librosa.decompose.nn_filter(data, rec=rec, aggregate=np.median).tolist(),
        "average": librosa.decompose.nn_filter(data, rec=rec, aggregate=np.average).tolist(),
        "max": librosa.decompose.nn_filter(data, rec=rec, aggregate=np.max).tolist(),
        "min": librosa.decompose.nn_filter(data, rec=rec, aggregate=np.min).tolist(),
        "sum": librosa.decompose.nn_filter(data, rec=rec, aggregate=np.sum).tolist(),
        "percentile": librosa.decompose.nn_filter(data, rec=rec, aggregate=lambda x, axis: np.percentile(x, 75, axis=axis)).tolist(),
    }

    # trim and split with callables.
    burst = np.concatenate([np.zeros(4096), signals["mix"][: sr // 4], np.zeros(4096), signals["mix"][: sr // 8] * 0.05, np.zeros(2048)]).astype(np.float64)
    stereo = np.stack([burst, burst[::-1] * 0.25])
    percentile_agg = lambda x, axis: np.percentile(x, 30, axis=axis)
    out["silence"] = {
        "y": burst.tolist(),
        "trim_median_ref": librosa.effects.trim(burst, top_db=20, ref=np.median)[1].tolist(),
        "split_median_ref": librosa.effects.split(burst, top_db=20, ref=np.median).tolist(),
        "split_stereo_median": librosa.effects.split(stereo, top_db=25, aggregate=np.median).tolist(),
        "split_stereo_percentile": librosa.effects.split(stereo, top_db=25, aggregate=percentile_agg).tolist(),
        "trim_stereo_both": librosa.effects.trim(stereo, top_db=25, ref=np.median, aggregate=np.median)[1].tolist(),
    }

    # griffinlim_cqt with random phases: the random stream differs, so the reference
    # is how close the reconstruction comes to the target magnitude.
    C = np.abs(librosa.cqt(y, sr=sr, n_bins=60, tuning=0.0))
    y_r = librosa.griffinlim_cqt(C, sr=sr, n_iter=8, length=len(y), init="random", rng=0, tuning=0.0)
    rec_mag = np.abs(librosa.cqt(y_r, sr=sr, n_bins=60, tuning=0.0))
    out["griffinlim_cqt_random"] = {
        "n_iter": 8,
        "n_bins": 60,
        "spectral_convergence": float(np.linalg.norm(rec_mag - C) / np.linalg.norm(C)),
    }
    return out
