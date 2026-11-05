"""librosa's interval, FJS, svara, mela and thaat helpers, and librosa.display.

The note and svara functions take arrays in both libraries. Display rows draw on a new
figure each with the Agg backend and have no native half.
"""

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from audio_samples import mpl  # noqa: E402
from audio_samples import notation as N  # noqa: E402

from . import Case, lcg_matrix, partial, register  # noqa: E402

G = "notation"
N_NAMES = 1000
RATIOS = [1.0, 1.25, 1.5, 7.0 / 4, 9.0 / 8, 5.0 / 3, 4.0 / 3, 16.0 / 15]


def _midi():
    return 48.0 + 48.0 * np.arange(N_NAMES) / N_NAMES


def _hz():
    return 110.0 * 2.0 ** (np.arange(N_NAMES) * 3.0 / N_NAMES)


def _notes():
    names = ["C4", "D4", "E4", "G4", "A4", "C5", "F#3", "Bb4"]
    return (names * (N_NAMES // len(names)))[:N_NAMES]


def _mat():
    return lcg_matrix(128, 200, 41)


def _wave():
    return np.sin(np.linspace(0.0, 3000.0, 44100))


def _closed(call):
    def run():
        call()
        plt.close("all")

    return run


def pair(name, lib, ours, func, configs="const"):
    return Case(name, G, configs, librosa=lib, ours=ours, func=func)


register(
    pair("pythagorean_intervals", lambda c: partial(librosa.pythagorean_intervals, bins_per_octave=53), lambda c: partial(N.pythagorean_intervals, bins_per_octave=53), "pythagorean_intervals"),
    pair("plimit_intervals", lambda c: partial(librosa.plimit_intervals, primes=[3, 5], bins_per_octave=24), lambda c: partial(N.plimit_intervals, primes=[3, 5], bins_per_octave=24), "plimit_intervals"),
    pair("interval_frequencies", lambda c: partial(librosa.interval_frequencies, 84, fmin=32.7, intervals="ji5"), lambda c: partial(N.interval_frequencies, 84, fmin=32.7, intervals="ji5"), "interval_frequencies"),
    pair("fifths_to_note", lambda c: partial(librosa.fifths_to_note, unison="C", fifths=7), lambda c: partial(N.fifths_to_note, unison="C", fifths=7), "fifths_to_note"),
    pair("interval_to_fjs", lambda c: partial(librosa.interval_to_fjs, RATIOS), lambda c: partial(N.interval_to_fjs, RATIOS), "interval_to_fjs"),
    pair("interval_to_fjs_tiny", lambda c: partial(librosa.interval_to_fjs, 1.5), lambda c: partial(N.interval_to_fjs, 1.5), "interval_to_fjs"),
    pair("hz_to_fjs", lambda c: partial(librosa.hz_to_fjs, [220.0 * r for r in RATIOS], fmin=220.0, unison="A"), lambda c: partial(N.hz_to_fjs, [220.0 * r for r in RATIOS], fmin=220.0, unison="A"), "hz_to_fjs"),
    pair("list_mela", lambda c: librosa.list_mela, lambda c: N.list_mela, "list_mela"),
    pair("list_thaat", lambda c: librosa.list_thaat, lambda c: N.list_thaat, "list_thaat"),
    pair("thaat_to_degrees", lambda c: partial(librosa.thaat_to_degrees, "kafi"), lambda c: partial(N.thaat_to_degrees, "kafi"), "thaat_to_degrees"),
    pair("mela_to_degrees", lambda c: partial(librosa.mela_to_degrees, 29), lambda c: partial(N.mela_to_degrees, 29), "mela_to_degrees"),
    pair("mela_to_svara", lambda c: partial(librosa.mela_to_svara, 29), lambda c: partial(N.mela_to_svara, 29), "mela_to_svara"),
    pair("midi_to_svara_h", lambda c: partial(librosa.midi_to_svara_h, _midi(), Sa=60.0), lambda c: partial(N.midi_to_svara_h, _midi(), Sa=60.0), "midi_to_svara_h"),
    pair("midi_to_svara_h_tiny", lambda c: partial(librosa.midi_to_svara_h, 67.0, Sa=60.0), lambda c: partial(N.midi_to_svara_h, 67.0, Sa=60.0), "midi_to_svara_h"),
    pair("midi_to_svara_c", lambda c: partial(librosa.midi_to_svara_c, _midi(), Sa=60.0, mela=29), lambda c: partial(N.midi_to_svara_c, _midi(), Sa=60.0, mela=29), "midi_to_svara_c"),
    pair("hz_to_svara_h", lambda c: partial(librosa.hz_to_svara_h, _hz(), Sa=261.63), lambda c: partial(N.hz_to_svara_h, _hz(), Sa=261.63), "hz_to_svara_h"),
    pair("hz_to_svara_c", lambda c: partial(librosa.hz_to_svara_c, _hz(), Sa=261.63, mela=29), lambda c: partial(N.hz_to_svara_c, _hz(), Sa=261.63, mela=29), "hz_to_svara_c"),
    pair("note_to_svara_h", lambda c: partial(librosa.note_to_svara_h, _notes(), Sa="C4"), lambda c: partial(N.note_to_svara_h, _notes(), Sa="C4"), "note_to_svara_h"),
    pair("note_to_svara_c", lambda c: partial(librosa.note_to_svara_c, _notes(), Sa="C4", mela=29), lambda c: partial(N.note_to_svara_c, _notes(), Sa="C4", mela=29), "note_to_svara_c"),
    pair("specshow", lambda c: _closed(partial(librosa.display.specshow, _mat(), x_axis="time", y_axis="mel")), lambda c: _closed(partial(mpl.plot_matrix, _mat(), x_axis="time", y_axis="mel")), "display.specshow"),
    pair("waveshow", lambda c: _closed(partial(librosa.display.waveshow, _wave(), sr=22050)), lambda c: _closed(partial(mpl.plot_waveform_adaptive, _wave(), sample_rate=22050)), "display.waveshow"),
    pair("wavebars", lambda c: _closed(partial(librosa.display.wavebars, _wave(), sr=22050, n_bars=50)), lambda c: _closed(partial(mpl.plot_waveform_bars, _wave(), sample_rate=22050, n_bars=50)), "display.wavebars"),
    pair("infer_cmap", lambda c: partial(librosa.display.infer_cmap, _mat()), lambda c: partial(mpl.infer_colormap, _mat()), "display.infer_cmap"),
)
