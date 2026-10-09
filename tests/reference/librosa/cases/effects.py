"""librosa.effects / decompose / phase_vocoder / griffinlim / lpc / clicks references.

Long outputs are stored as strided subsets (``[::step]`` for signals,
``[::bin_step, ::frame_step]`` for spectrograms); the step sizes are stored next
to the data and the Rust test applies the same slicing.

librosa delegates:
* ``pitch_shift`` resampling to ``soxr`` (``res_type="soxr_hq"``), so the
  time-stretched intermediate (the phase-vocoder part, exact) is stored too;
* HPSS median filtering to ``scipy.ndimage.median_filter(mode="reflect")``;
* ``preemphasis`` / ``deemphasis`` to ``scipy.signal.lfilter``;
* ``phase_vocoder`` magnitude interpolation to ``scipy.interpolate.interp1d``;
* ``nn_filter``'s recurrence matrix to sklearn nearest neighbours.
"""

from __future__ import annotations

import numpy as np

import librosa

NAME = "effects"

SIG_STEP = 10
# Exact, cheap recursions (emphasis) and HPSS audio are sampled more sparsely.
SPARSE_STEP = 20


def _spec_subset(d, bin_step=16, frame_step=2):
    return {"shape": list(d.shape), "bin_step": bin_step, "frame_step": frame_step, "subset": d[..., ::bin_step, ::frame_step]}


def build(signals, sr):
    mix = signals["mix"]
    noise = signals["noise"]
    notes = signals["notes"]
    burst = signals["burst"]
    stereo = np.stack([signals["stereo_l"], signals["stereo_r"]])
    out: dict = {"sig_step": SIG_STEP, "sparse_step": SPARSE_STEP}

    # Phase vocoder on the default STFT of `mix`.
    d = librosa.stft(mix)
    pv = {}
    for rate in (0.8, 1.25):
        pv[str(rate)] = _spec_subset(librosa.phase_vocoder(d, rate=rate))
    t_out = np.array([0.0, 0.3, 1.0, 2.5, 7.25, 20.9, d.shape[-1] - 0.5])
    pv["t_out"] = {"t_out": t_out, **_spec_subset(librosa.phase_vocoder(d, t_out=t_out))}
    out["phase_vocoder"] = pv

    # time_stretch (librosa defaults: n_fft 2048, hop 512).
    out["time_stretch"] = {
        str(rate): {"len": len(y), "y": y[::SIG_STEP]}
        for rate in (1.5, 0.75)
        for y in [librosa.effects.time_stretch(mix, rate=rate)]
    }
    ts_st = librosa.effects.time_stretch(stereo, rate=1.5)
    out["time_stretch_stereo"] = {"rate": 1.5, "len": ts_st.shape[-1], "y": ts_st[:, ::SIG_STEP]}

    # pitch_shift: final output (soxr) and the resampler-free intermediate.
    ps = {}
    for n_steps in (2, -3):
        rate = 2.0 ** (-n_steps / 12)
        stretched = librosa.effects.time_stretch(mix, rate=rate)
        y = librosa.effects.pitch_shift(mix, sr=sr, n_steps=n_steps)
        ps[str(n_steps)] = {
            "rate": rate,
            "stretched_len": len(stretched),
            "stretched": stretched[::SIG_STEP],
            "len": len(y),
            "y": y[::SIG_STEP],
        }
    out["pitch_shift"] = ps

    # pre-/de-emphasis.
    pre, pre_zf = librosa.effects.preemphasis(mix, return_zf=True)
    pre_zi, pre_zi_zf = librosa.effects.preemphasis(mix, coef=0.5, zi=[0.25], return_zf=True)
    de, de_zf = librosa.effects.deemphasis(mix, return_zf=True)
    de_zi, de_zi_zf = librosa.effects.deemphasis(mix, coef=0.8, zi=[-0.1], return_zf=True)
    pre_st = librosa.effects.preemphasis(stereo)
    de_st = librosa.effects.deemphasis(stereo)
    out["emphasis"] = {
        "pre": pre[::SPARSE_STEP],
        "pre_zf": float(pre_zf[0]),
        "pre_coef05_zi025": pre_zi[::SPARSE_STEP],
        "pre_coef05_zi025_zf": float(pre_zi_zf[0]),
        "de": de[::SPARSE_STEP],
        "de_zf": float(de_zf[0]),
        "de_coef08_zim01": de_zi[::SPARSE_STEP],
        "de_coef08_zim01_zf": float(de_zi_zf[0]),
        "pre_stereo_head": pre_st[:, :50],
        "de_stereo_head": de_st[:, :50],
    }

    # trim / split.
    tr = {}
    for name, kw in {
        "default": {},
        "top_db30": dict(top_db=30),
        "frame1024_hop256": dict(frame_length=1024, hop_length=256),
        "ref_abs": dict(ref=0.01, top_db=20),
    }.items():
        _, idx = librosa.effects.trim(burst, **kw)
        tr[name] = {
            "params": {k: float(v) for k, v in kw.items()},
            "trim": idx,
            "split": librosa.effects.split(burst, **kw),
        }
    tr["mix_split"] = librosa.effects.split(mix, top_db=20)
    tr["stereo_split_top_db20"] = librosa.effects.split(stereo, top_db=20)
    out["trim_split"] = tr

    # remix.
    intervals = [(0, 5000), (12000, 15000), (6000, 9001)]
    rm = {"intervals": intervals}
    for align in (True, False):
        y = librosa.effects.remix(mix, np.asarray(intervals), align_zeros=align)
        rm[f"align_{align}"] = {"len": len(y), "y": y[::SIG_STEP]}
    out["remix"] = rm

    # HPSS on the STFT of `notes` (decompose) and on the audio (effects).
    dn = librosa.stft(notes)
    hp = {}
    for name, kw in {
        "default": {},
        "margin2": dict(margin=2.0),
        "kernel_17_9_power1": dict(kernel_size=(17, 9), power=1.0),
        "hard": dict(power=np.inf),
        "kernel_even": dict(kernel_size=(10, 12), margin=(1.0, 3.0)),
    }.items():
        h, p = librosa.decompose.hpss(dn, **kw)
        mh, mp = librosa.decompose.hpss(dn, mask=True, **kw)
        hp[name] = {
            "params": {k: (list(v) if isinstance(v, tuple) else (None if v == np.inf else v)) for k, v in kw.items()},
            "harm": _spec_subset(h, 32, 3),
            "perc": _spec_subset(p, 32, 3),
            "mask_harm": _spec_subset(mh.astype(np.float64), 32, 3),
            "mask_perc": _spec_subset(mp.astype(np.float64), 32, 3),
        }
    hp["magnitude_default"] = {
        "harm": _spec_subset(librosa.decompose.hpss(np.abs(dn))[0], 32, 3),
        "perc": _spec_subset(librosa.decompose.hpss(np.abs(dn))[1], 32, 3),
    }
    out["hpss_spectrogram"] = hp
    yh, yp = librosa.effects.hpss(notes)
    out["hpss_audio"] = {
        "harmonic": yh[::SPARSE_STEP],
        "percussive": yp[::SPARSE_STEP],
        "harmonic_margin2": librosa.effects.harmonic(notes, margin=2.0)[::SPARSE_STEP],
        "percussive_margin2": librosa.effects.percussive(notes, margin=2.0)[::SPARSE_STEP],
    }

    # softmask.
    rng = np.random.default_rng(7)
    x = rng.random((6, 7))
    x_ref = rng.random((6, 7))
    x[0, :3] = 0.0
    x_ref[0, :3] = 0.0
    out["softmask"] = {
        "x": x,
        "x_ref": x_ref,
        "power1": librosa.util.softmask(x, x_ref, power=1),
        "power2_split": librosa.util.softmask(x, x_ref, power=2, split_zeros=True),
        "power3_5": librosa.util.softmask(x, x_ref, power=3.5),
        "inf": librosa.util.softmask(x, x_ref, power=np.inf).astype(np.float64),
    }

    # Griffin-Lim with init=None (deterministic).
    s = np.abs(librosa.stft(mix[:8000], n_fft=512))
    out["griffinlim"] = {
        "n_fft": 512,
        "n_iter": 8,
        "len": 8000,
        "momentum099": librosa.griffinlim(s, n_iter=8, init=None, n_fft=512, length=8000)[::SIG_STEP],
        "momentum0": librosa.griffinlim(s, n_iter=8, init=None, n_fft=512, momentum=0.0, length=8000)[::SIG_STEP],
        "no_length": librosa.griffinlim(s, n_iter=4, init=None, n_fft=512)[::SIG_STEP],
    }

    # nn_filter on a chroma of `notes` (input stored).
    chroma = librosa.feature.chroma_stft(y=notes, sr=sr)
    chroma = chroma.astype(np.float64)
    rec_aff = librosa.segment.recurrence_matrix(chroma, mode="affinity", sym=True)
    out["nn_filter"] = {
        "chroma": chroma,
        "mean": librosa.decompose.nn_filter(chroma),
        "median": librosa.decompose.nn_filter(chroma, aggregate=np.median),
        "affinity_sym_average": librosa.decompose.nn_filter(chroma, rec=rec_aff, aggregate=np.average),
    }

    # LPC (Burg).
    out["lpc"] = {
        f"{name}_{order}": librosa.lpc(sig, order=order)
        for name, sig in (("mix", mix), ("noise", noise))
        for order in (4, 16)
    }

    # clicks.
    out["clicks"] = {
        "times": librosa.clicks(times=[0.1, 0.5, 0.52], sr=sr).astype(np.float64)[::SIG_STEP],
        "frames_len": librosa.clicks(frames=[3, 20, 40], sr=sr, hop_length=256, click_freq=2000.0, click_duration=0.05, length=11000).astype(np.float64)[::SIG_STEP],
        "times_len": len(librosa.clicks(times=[0.1, 0.5, 0.52], sr=sr)),
    }
    return out
