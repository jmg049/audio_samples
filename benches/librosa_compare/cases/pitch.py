"""librosa pitch tracking and harmonics: yin, pyin, salience, interp_harmonics,
f0_harmonics. (piptrack, estimate_tuning and pitch_tuning are in ``tonal``.)

No rows of these tables are skipped.
"""

import librosa

from audio_samples import features as F

from . import Case, partial, register

G = "pitch"

# librosa has no default fmin / fmax for yin and pyin; the three paths use C2..C7.
FMIN = float(librosa.note_to_hz("C2"))
FMAX = float(librosa.note_to_hz("C7"))
# librosa requires `harmonics` for the harmonic functions.
HARMONICS = [1, 2, 3, 4, 5]


def _f0(c):
    """``librosa.yin(y)``: one f0 per STFT frame (same hop and centring)."""
    return c.cached("yin_f0", lambda: librosa.yin(c.y, fmin=FMIN, fmax=FMAX, sr=c.sr))


register(
    Case(
        "yin", G, "full",
        librosa=lambda c: partial(librosa.yin, c.y, fmin=FMIN, fmax=FMAX, sr=c.sr),
        ours=lambda c: partial(c.a.yin, fmin=FMIN, fmax=FMAX),
        func="yin",
        note="fmin=C2, fmax=C7",
    ),
    Case(
        "pyin", G, "short",
        librosa=lambda c: partial(librosa.pyin, c.y, fmin=FMIN, fmax=FMAX, sr=c.sr),
        ours=lambda c: partial(c.a.pyin, fmin=FMIN, fmax=FMAX),
        func="pyin",
        note="fmin=C2, fmax=C7",
    ),
    Case(
        "salience", G, "derived",
        librosa=lambda c: partial(librosa.salience, c.S_mag, freqs=c.freqs, harmonics=HARMONICS),
        ours=lambda c: partial(F.salience, c.S_mag, freqs=c.freqs, harmonics=HARMONICS),
        func="salience",
        note="input |stft|; harmonics 1..5",
    ),
    Case(
        "interp_harmonics", G, "derived",
        librosa=lambda c: partial(librosa.interp_harmonics, c.S_mag, freqs=c.freqs, harmonics=HARMONICS),
        ours=lambda c: partial(F.interp_harmonics, c.S_mag, freqs=c.freqs, harmonics=HARMONICS),
        func="interp_harmonics",
        note="input |stft|; harmonics 1..5",
    ),
    Case(
        "f0_harmonics", G, "derived",
        librosa=lambda c: partial(
            librosa.f0_harmonics, c.S_mag, f0=_f0(c), freqs=c.freqs, harmonics=HARMONICS
        ),
        ours=lambda c: partial(F.f0_harmonics, c.S_mag, f0=_f0(c), freqs=c.freqs, harmonics=HARMONICS),
        func="f0_harmonics",
        note="input |stft| and librosa.yin f0 (C2..C7); harmonics 1..5",
    ),
)
