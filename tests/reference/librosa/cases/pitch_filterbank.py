"""References for ``librosa.iirt``, ``librosa.filters.semitone_filterbank`` and
``librosa.filters.mr_frequencies``.

The filterbank uses ``scipy.signal.iirdesign`` (order selection and design), compared
section by section. ``iirt`` includes ``soxr_hq`` resampling for the two lower rates,
where the crate's resampler differs; the top rate needs no resampling, so its bands are
stored separately and agree more closely.
"""

from __future__ import annotations

import numpy as np

import librosa

NAME = "pitch_filterbank"


def _l(a):
    return np.asarray(a).tolist()


def build(signals, sr) -> dict:
    out: dict = {}
    for tuning in (0.0, 0.25):
        f, r = librosa.filters.mr_frequencies(tuning)
        out[f"mr_{tuning}"] = {"freqs": _l(f), "rates": _l(r)}

    bank: dict = {}
    f, r = librosa.filters.mr_frequencies(0.0)
    picks = [0, 20, 35, 36, 60, 69, 70, 84]
    for ftype in ("ellip", "butter", "cheby1", "cheby2"):
        kw = {} if ftype == "ellip" else {"ftype": ftype}
        fb, rates = librosa.filters.semitone_filterbank(flayout="sos", **kw)
        bank[ftype] = {str(i): _l(fb[i]) for i in picks}
        assert np.array_equal(rates, r)
    ba, _ = librosa.filters.semitone_filterbank(flayout="ba")
    bank["ba"] = {str(i): {"b": _l(ba[i][0]), "a": _l(ba[i][1])} for i in (36, 69, 84)}
    q, _ = librosa.filters.semitone_filterbank(flayout="sos", Q=10.0, passband_ripple=0.5, stopband_attenuation=30)
    bank["q10"] = {str(i): _l(q[i]) for i in picks}
    cf, cr = np.array([300.0, 600.0]), np.array([4410.0, 4410.0])
    custom, _ = librosa.filters.semitone_filterbank(center_freqs=cf, sample_rates=cr, flayout="sos")
    bank["custom"] = {"freqs": _l(cf), "rates": _l(cr), "sos": [_l(s) for s in custom]}
    out["bank"] = bank

    y = signals["mix"].astype(np.float64)
    cases: dict = {}
    for name, kw in {
        "default": dict(),
        "tuned": dict(tuning=0.25, win_length=1024),
        "nocenter": dict(center=False, hop_length=256),
        "reflect": dict(pad_mode="reflect", win_length=1024, hop_length=512),
        "ba": dict(flayout="ba"),
    }.items():
        cases[name] = {"kw": kw, "power": _l(librosa.iirt(y, sr=sr, **kw)[70:, :20])}
    full = librosa.iirt(y, sr=sr)
    cases["default"]["all"] = _l(full[:, ::8])
    out["iirt"] = cases
    return out
