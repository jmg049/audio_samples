"""References for the interval, FJS, svara, mela and thaat helpers of librosa.

``interval_to_fjs`` is run on every ratio in librosa's interval table, which fixes the
factorisation of each. Names are stored as strings.
"""

from __future__ import annotations

import numpy as np

import librosa
import librosa.core.intervals as intervals_module

NAME = "notation"


def _l(a):
    return np.asarray(a).tolist()


def _factors(values):
    return [{str(k): int(v) for k, v in d.items()} for d in values]


def build(signals, sr) -> dict:
    out: dict = {}
    tune: dict = {}
    for b in (7, 12, 24, 53):
        for sort in (True, False):
            tune[f"pyth_{b}_{sort}"] = _l(librosa.pythagorean_intervals(bins_per_octave=b, sort=sort))
            tune[f"pyth_factors_{b}_{sort}"] = _factors(
                librosa.pythagorean_intervals(bins_per_octave=b, sort=sort, return_factors=True)
            )
    for primes, b in (([3], 12), ([3, 5], 12), ([3, 5], 24), ([3, 5, 7], 16), ([3, 11], 9), ([5], 5)):
        for sort in (True, False):
            key = "_".join(map(str, primes)) + f"_{b}_{sort}"
            tune[f"plimit_{key}"] = _l(librosa.plimit_intervals(primes=primes, bins_per_octave=b, sort=sort))
            tune[f"plimit_factors_{key}"] = _factors(
                librosa.plimit_intervals(primes=primes, bins_per_octave=b, sort=sort, return_factors=True)
            )
    out["tuning"] = tune

    freqs: dict = {}
    for name in ("equal", "pythagorean", "ji3", "ji5", "ji7"):
        for sort in (True, False):
            freqs[f"{name}_{sort}"] = _l(
                librosa.interval_frequencies(30, fmin=55.0, intervals=name, bins_per_octave=12, sort=sort)
            )
    freqs["equal_tuned"] = _l(librosa.interval_frequencies(30, fmin=55.0, intervals="equal", tuning=0.25))
    freqs["custom"] = _l(librosa.interval_frequencies(7, fmin=100.0, intervals=[1.0, 1.25, 1.5]))
    out["frequencies"] = freqs

    out["fifths"] = [
        {"unison": u, "fifths": f, "unicode": uni, "out": librosa.fifths_to_note(unison=u, fifths=f, unicode=uni)}
        for u in ("C", "F#", "Bb", "Ebb", "g")
        for f in (-15, -8, -2, 0, 1, 7, 13, 26)
        for uni in (True, False)
    ]

    keys = sorted(intervals_module.INTERVALS.keys())
    out["fjs_table"] = {
        "intervals": [float(k) for k in keys],
        "unicode": [librosa.interval_to_fjs(float(k), unison="C") for k in keys],
        "ascii": [librosa.interval_to_fjs(float(k), unison="F#", unicode=False) for k in keys],
    }
    ratios = [1.0, 2.0, 3.0, 1.5, 0.75, 5.0 / 4, 16.0 / 5, 7.0 / 4]
    out["fjs_octaves"] = {"intervals": ratios, "out": [librosa.interval_to_fjs(r) for r in ratios]}
    fs = [220.0, 330.0, 275.0, 385.0]
    out["hz_to_fjs"] = {
        "hz": fs,
        "default": _l(librosa.hz_to_fjs(fs)),
        "fmin": _l(librosa.hz_to_fjs(fs, fmin=110.0, unison="D", unicode=True)),
    }

    out["list_thaat"] = librosa.list_thaat()
    out["thaat_degrees"] = {t: _l(librosa.thaat_to_degrees(t)) for t in librosa.list_thaat()}
    mela = librosa.list_mela()
    out["list_mela"] = {k: int(v) for k, v in mela.items()}
    out["mela_degrees"] = [_l(librosa.mela_to_degrees(i)) for i in range(1, 73)]
    out["mela_svara"] = {
        f"{abbr}_{uni}": [librosa.mela_to_svara(i, abbr=abbr, unicode=uni) for i in range(1, 73)]
        for abbr in (True, False)
        for uni in (True, False)
    }

    midis = [48.0, 59.4, 60.0, 61.6, 66.0, 71.0, 72.0, 83.0, 84.0, 96.0, 47.0, float("nan")]
    svara: dict = {"midi": [m if m == m else None for m in midis]}
    for abbr in (True, False):
        for octave in (True, False):
            for uni in (True, False):
                kw = dict(abbr=abbr, octave=octave, unicode=uni)
                key = f"{abbr}_{octave}_{uni}"
                svara[f"h_{key}"] = [librosa.midi_to_svara_h(m, Sa=60.0, **kw) for m in midis]
                svara[f"c_{key}"] = [librosa.midi_to_svara_c(m, Sa=60.0, mela=31, **kw) for m in midis]
    hzs = [220.0, 261.63, 440.0, 523.25, 110.0]
    svara["hz"] = hzs
    svara["hz_h"] = _l(librosa.hz_to_svara_h(np.array(hzs), Sa=261.63))
    svara["hz_c"] = _l(librosa.hz_to_svara_c(np.array(hzs), Sa=261.63, mela="kosalam"))
    notes = ["C4", "C#4", "G4", "C5", "B3", "D4+25"]
    svara["notes"] = notes
    svara["note_h"] = _l(librosa.note_to_svara_h(notes, Sa="C4"))
    svara["note_c"] = _l(librosa.note_to_svara_c(notes, Sa="C4", mela=1))
    out["svara"] = svara
    return out
