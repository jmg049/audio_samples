"""librosa.convert: unit conversions and frequency tables.

Array conversions run on ``vec_f64`` (VEC_N values; the audio_samples
conversions are float64 for every input). Frequency tables run on ``const``.

``samples_to_time`` (VEC_N values) and the note-name functions ``hz_to_note``,
``midi_to_note``, ``note_to_hz``, ``note_to_midi`` (N_STR inputs) take the whole
array in one call in both libraries. ``key_to_notes`` and ``key_to_degrees``
take one key in both libraries and are looped in both.

Note names: audio_samples spells sharps in ASCII (``#``), so librosa is called
with ``unicode=False`` where it has the option.

Skipped rows: the "not provided" rows only (blocks_to_*, interval/FJS/svara
helpers).
"""

import librosa
import numpy as np

from audio_samples import rhythm as R
from audio_samples import utils as U

from . import VEC_N, Case, partial, register, vec_values

G = "convert"
SR = 22050
HOP = 512
# Inputs of the string conversions.
N_STR = 1000
KEYS = [f"{p}:{m}" for m in ("maj", "min") for p in ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")]
C1 = 32.70319566257483  # note_to_hz('C1')


def _frames(c):
    return c.cached("frames", lambda: np.arange(VEC_N, dtype=np.int64))


def _samples(c):
    return c.cached("samples", lambda: np.arange(VEC_N, dtype=np.int64) * 13)


def _times(c):
    return c.cached("times", lambda: vec_values(0.0, 60.0, np.float64))


def _hz(c):
    return c.cached("hz", lambda: vec_values(20.0, 11025.0, np.float64))


def _str_hz():
    return np.geomspace(30.0, 4000.0, N_STR)


def _str_midi():
    return np.arange(N_STR) % 128


def _str_notes():
    return list(librosa.midi_to_note(_str_midi(), unicode=False))


def _loop(f, xs):
    """One call of ``f`` per input."""
    return lambda: [f(x) for x in xs]


def _keys():
    return [KEYS[i % len(KEYS)] for i in range(N_STR)]


register(
    # --- frames, samples, time ---------------------------------------------------
    Case(
        "frames_to_samples", G, "vec_f64",
        librosa=lambda c: partial(librosa.frames_to_samples, _frames(c)),
        ours=lambda c: partial(U.frames_to_samples, _frames(c), hop_length=HOP),
        func="frames_to_samples",
    ),
    Case(
        "frames_to_time", G, "vec_f64",
        librosa=lambda c: partial(librosa.frames_to_time, _frames(c), sr=SR),
        ours=lambda c: partial(U.frames_to_time, _frames(c), SR, HOP),
        func="frames_to_time",
    ),
    Case(
        "samples_to_frames", G, "vec_f64",
        librosa=lambda c: partial(librosa.samples_to_frames, _samples(c)),
        ours=lambda c: partial(U.samples_to_frames, _samples(c), hop_length=HOP),
        func="samples_to_frames",
    ),
    Case(
        "samples_to_time", G, "vec_f64",
        librosa=lambda c: partial(librosa.samples_to_time, _samples(c), sr=SR),
        ours=lambda c: partial(U.samples_to_time, _samples(c), SR),
        func="samples_to_time",
    ),
    Case(
        "time_to_frames", G, "vec_f64",
        librosa=lambda c: partial(librosa.time_to_frames, _times(c), sr=SR),
        ours=lambda c: partial(U.time_to_frames_floor, _times(c), sr=SR, hop_length=HOP),
        func="time_to_frames",
    ),
    Case(
        "time_to_samples", G, "vec_f64",
        librosa=lambda c: partial(librosa.time_to_samples, _times(c), sr=SR),
        ours=lambda c: partial(U.time_to_samples, _times(c), sr=SR),
        func="time_to_samples",
    ),
    Case(
        "times_like", G, "vec_f64",
        librosa=lambda c: partial(librosa.times_like, VEC_N, sr=SR),
        ours=lambda c: partial(U.times_like, VEC_N, sr=SR, hop_length=HOP),
        func="times_like",
        note="VEC_N frames",
    ),
    Case(
        "samples_like", G, "vec_f64",
        librosa=lambda c: partial(librosa.samples_like, VEC_N),
        ours=lambda c: partial(U.samples_like, VEC_N, hop_length=HOP),
        func="samples_like",
        note="VEC_N frames",
    ),
    # --- frequency tables --------------------------------------------------------------
    Case(
        "fft_frequencies", G, "const",
        librosa=lambda c: partial(librosa.fft_frequencies, sr=SR, n_fft=2048),
        ours=lambda c: partial(U.fft_frequencies, 2048, SR),
        func="fft_frequencies",
    ),
    Case(
        "mel_frequencies", G, "const",
        librosa=lambda c: partial(librosa.mel_frequencies, n_mels=128, fmin=0.0, fmax=11025.0),
        ours=lambda c: partial(U.mel_frequencies, 128, 0.0, 11025.0, htk=False),
        func="mel_frequencies",
        note="the binding defaults to htk=True; librosa's Slaney scale (htk=False) is passed",
    ),
    Case(
        "cqt_frequencies", G, "const",
        librosa=lambda c: partial(librosa.cqt_frequencies, 84, fmin=C1),
        ours=lambda c: partial(U.cqt_frequencies, 84, fmin=C1),
        func="cqt_frequencies",
    ),
    Case(
        "tempo_frequencies", G, "const",
        librosa=lambda c: partial(librosa.tempo_frequencies, 384, sr=SR),
        ours=lambda c: partial(R.tempo_frequencies, 384, sr=SR),
        func="tempo_frequencies",
    ),
    Case(
        "fourier_tempo_frequencies", G, "const",
        librosa=lambda c: partial(librosa.fourier_tempo_frequencies, sr=SR),
        ours=lambda c: partial(R.fourier_tempo_frequencies, sr=SR),
        func="fourier_tempo_frequencies",
    ),
    # --- Hz, mel, MIDI, octaves, tuning ---------------------------------------------------
    Case(
        "hz_to_mel", G, "vec_f64",
        librosa=lambda c: partial(librosa.hz_to_mel, _hz(c)),
        ours=lambda c: partial(U.hz_to_mel, _hz(c), htk=False),
        func="hz_to_mel",
        note="the binding defaults to htk=True; librosa's Slaney scale (htk=False) is passed",
    ),
    Case(
        "mel_to_hz", G, "vec_f64",
        librosa=lambda c: partial(librosa.mel_to_hz, vec_values(0.0, 45.0, np.float64)),
        ours=lambda c: partial(U.mel_to_hz, vec_values(0.0, 45.0, np.float64), htk=False),
        func="mel_to_hz",
        note="the binding defaults to htk=True; librosa's Slaney scale (htk=False) is passed",
    ),
    Case(
        "hz_to_midi", G, "vec_f64",
        librosa=lambda c: partial(librosa.hz_to_midi, _hz(c)),
        ours=lambda c: partial(U.hz_to_midi, _hz(c)),
        func="hz_to_midi",
    ),
    Case(
        "midi_to_hz", G, "vec_f64",
        librosa=lambda c: partial(librosa.midi_to_hz, vec_values(0.0, 127.0, np.float64)),
        ours=lambda c: partial(U.midi_to_hz, vec_values(0.0, 127.0, np.float64)),
        func="midi_to_hz",
    ),
    Case(
        "hz_to_octs", G, "vec_f64",
        librosa=lambda c: partial(librosa.hz_to_octs, _hz(c)),
        ours=lambda c: partial(U.hz_to_octs, _hz(c)),
        func="hz_to_octs",
    ),
    Case(
        "octs_to_hz", G, "vec_f64",
        librosa=lambda c: partial(librosa.octs_to_hz, vec_values(0.0, 9.0, np.float64)),
        ours=lambda c: partial(U.octs_to_hz, vec_values(0.0, 9.0, np.float64)),
        func="octs_to_hz",
    ),
    Case(
        "A4_to_tuning", G, "vec_f64",
        librosa=lambda c: partial(librosa.A4_to_tuning, vec_values(430.0, 450.0, np.float64)),
        ours=lambda c: partial(U.a4_to_tuning, vec_values(430.0, 450.0, np.float64)),
        func="A4_to_tuning",
    ),
    Case(
        "tuning_to_A4", G, "vec_f64",
        librosa=lambda c: partial(librosa.tuning_to_A4, vec_values(-0.5, 0.5, np.float64)),
        ours=lambda c: partial(U.tuning_to_a4, vec_values(-0.5, 0.5, np.float64)),
        func="tuning_to_A4",
    ),
    # --- note names and keys -----------------------------------------------------------------
    Case(
        "hz_to_note", G, "const",
        librosa=lambda c: partial(librosa.hz_to_note, _str_hz(), unicode=False),
        ours=lambda c: partial(U.frequency_to_note, _str_hz()),
        func="hz_to_note",
        note="1000 frequencies; ours also returns the cents",
    ),
    Case(
        "midi_to_note", G, "const",
        librosa=lambda c: partial(librosa.midi_to_note, _str_midi(), unicode=False),
        ours=lambda c: partial(U.midi_to_note, _str_midi()),
        func="midi_to_note",
        note="1000 MIDI numbers",
    ),
    Case(
        "note_to_hz", G, "const",
        librosa=lambda c: partial(librosa.note_to_hz, _str_notes()),
        ours=lambda c: partial(U.note_to_frequency, _str_notes()),
        func="note_to_hz",
        note="1000 note names",
    ),
    Case(
        "note_to_midi", G, "const",
        librosa=lambda c: partial(librosa.note_to_midi, _str_notes()),
        ours=lambda c: partial(U.note_to_midi, _str_notes()),
        func="note_to_midi",
        note="1000 note names",
    ),
    Case(
        "key_to_notes", G, "const",
        librosa=lambda c: _loop(librosa.key_to_notes, _keys()),
        ours=lambda c: _loop(U.key_to_notes, _keys()),
        func="key_to_notes",
        note="1000 keys (the 24 major/minor keys repeated), a loop in both",
    ),
    Case(
        "key_to_degrees", G, "const",
        librosa=lambda c: _loop(librosa.key_to_degrees, _keys()),
        ours=lambda c: _loop(U.key_to_degrees, _keys()),
        func="key_to_degrees",
        note="1000 keys (the 24 major/minor keys repeated), a loop in both",
    ),
)
