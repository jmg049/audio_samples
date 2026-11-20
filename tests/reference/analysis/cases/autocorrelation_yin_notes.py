"""References for autocorrelation scaling, YIN and note-name parsing.

* ``autocorrelate``: ``librosa.autocorrelate(mix, max_size=2048)`` - the raw
  sums ``sum_j x[j] x[j+k]`` (librosa delegates to ``scipy.fft``; numpy's
  ``correlate`` gives the same values).
* ``yin``: ``librosa.yin(notes, fmin=65, fmax=2093, frame_length=2048)``
  (hop 512, centred, zero padding). audio_samples' ``track_pitch`` frames are
  not centred (frame ``i`` covers ``[512 i, 512 i + 2048)``, i.e. librosa frame
  ``i + 2``), so only the "confident" frames are compared: those whose whole
  2048-sample window lies inside one note of the sequence and where librosa is
  within 50 cents of that note's nominal pitch.
* ``note_to_midi``: ``librosa.note_to_midi`` (``round_midi=True``) on note
  strings covering case, ASCII/Unicode/double accidentals, enharmonics across
  the octave boundary, negative octaves and cents suffixes.
"""

from __future__ import annotations

import numpy as np

import librosa

NAME = "autocorrelation_yin_notes"

NOTES_HZ = [261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 493.88, 523.25]

NOTE_NAMES = [
    "C4", "c4", "A4", "a#3", "Bb2", "B#3", "Cb4", "E#4", "Fb4", "C##4", "Dbb4",
    "C♯4", "D♭4", "C𝄪4", "D𝄫4", "C♮4", "C!4", "C-1", "G9", "C+4", "A4+50",
    "A4-50", "A4+49", "A4-49", "G#-1", "f♯5+25", "eb6-30",
]


def build(signals, sr) -> dict:
    out: dict = {}

    mix = signals["mix"]
    out["autocorrelate"] = {
        "signal": "mix",
        "max_size": 2048,
        "values": librosa.autocorrelate(mix, max_size=2048),
    }

    notes = signals["notes"]
    frame_length, hop = 2048, 512
    f0 = librosa.yin(notes, fmin=65.0, fmax=2093.0, sr=sr, frame_length=frame_length, hop_length=hop)
    step = len(notes) // len(NOTES_HZ)
    frames, expected, nominal = [], [], []
    for t, f in enumerate(f0):
        centre = t * hop
        lo, hi = centre - frame_length // 2, centre + frame_length // 2
        k = lo // step
        if lo < 0 or k >= len(NOTES_HZ) or hi > (k + 1) * step:
            continue
        if abs(1200 * np.log2(f / NOTES_HZ[k])) > 50:
            continue
        frames.append(t)
        expected.append(f)
        nominal.append(NOTES_HZ[k])
    out["yin"] = {
        "signal": "notes",
        "fmin": 65.0,
        "fmax": 2093.0,
        "trough_threshold": 0.1,
        "frame_length": frame_length,
        "hop_length": hop,
        "frames": np.asarray(frames, dtype=np.int64),
        "f0": np.asarray(expected),
        "nominal": np.asarray(nominal),
    }

    out["note_to_midi"] = {
        "names": NOTE_NAMES,
        "midi": [int(librosa.note_to_midi(n)) for n in NOTE_NAMES],
    }
    return out
