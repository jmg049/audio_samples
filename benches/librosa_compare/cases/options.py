"""Options reached through array forms and explicit parameters.

Rows that call a form of ``audio_samples.forms`` have no native half: they are NumPy and
existing kernels. The file rows read 10 s of audio written once per process.
"""

import os
import tempfile

import librosa
import numpy as np

from audio_samples import AudioSamples, arrays, effects, features, forms, io, notation, rhythm, sequence

from . import Case, lcg_matrix, partial, register, vec_values

G = "options"
SR = 22050
RATIOS = librosa.plimit_intervals(primes=[3, 5], bins_per_octave=12)
FREQS = 110.0 * 2.0 ** (np.arange(24) / 12.0)
ALPHA = 0.05 + 0.002 * np.arange(24)
WINDOW = np.array([0.25, 0.5, 0.25])
_PATH = []


def _v(c):
    return c.cached("opt_v", lambda: np.sin(vec_values(0.0, 20.0, c.dtype)))


def _stereo(c):
    return c.cached("opt_stereo", lambda: np.stack([c.y, c.y[::-1]]))


def _midi():
    return 48.0 + 48.0 * np.arange(1000) / 1000


def _hz():
    return 110.0 * 2.0 ** (np.arange(1000) * 3.0 / 1000)


def _notes():
    names = ["C4+25", "D4", "E4-10", "G4", "A4", "C5", "F#3", "Bb4"]
    return (names * 125)[:1000]


def _wav():
    if not _PATH:
        directory = tempfile.mkdtemp(prefix="pg-options-", dir=os.environ.get("BENCH_TMP", "/var/tmp"))
        path = os.path.join(directory, "clip.wav")
        y = np.sin(np.linspace(0.0, 20000.0, 10 * SR)).astype(np.float32)
        io.save(path, AudioSamples.new_mono(y, SR))
        _PATH.append(path)
    return _PATH[0]


def _C(c):
    return c.cached("opt_C", lambda: np.abs(librosa.cqt(c.y, sr=SR, fmin=55.0, n_bins=72, bins_per_octave=36, tuning=0.0)))


def _S(c):
    return c.cached("opt_S", lambda: np.abs(librosa.stft(c.y, n_fft=1024, hop_length=256)))


def _Sc(c):
    return c.cached("opt_Sc", lambda: librosa.stft(c.y, n_fft=512, hop_length=128))


def _hf(c):
    return librosa.fft_frequencies(sr=SR, n_fft=1024)


def _tg(c):
    def make():
        env = librosa.onset.onset_strength(y=c.y, sr=SR, hop_length=256)
        return librosa.feature.tempogram(onset_envelope=env, sr=SR, hop_length=256, win_length=128)

    return c.cached("opt_tg", make)


def _tg_args(c):
    tg = _tg(c)
    return dict(tg=tg, bpm=np.full(tg.shape[-1], 120.0), freqs=np.linspace(1.0, 640.0, tg.shape[0]))


def _power(y, sr, n_fft, hop_length):
    return np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)) ** 2


def _mat():
    return lcg_matrix(12, 300, 17)


def _out(c):
    frames = 1 + c.y.size // 512
    return np.zeros((1025, frames), dtype=np.complex64 if c.y.dtype == np.float32 else np.complex128)


def pair(name, configs, lib, ours, func, note=""):
    return Case(name, G, configs, librosa=lib, ours=ours, func=func, note=note)


register(
    pair("zero_crossings_opt", "vec", lambda c: partial(librosa.zero_crossings, _v(c), threshold=0.1, zero_pos=False), lambda c: partial(arrays.zero_crossings, _v(c), threshold=0.1, zero_pos=False), "zero_crossings"),
    pair("tone_phi", "const", lambda c: partial(librosa.tone, 440.0, sr=SR, length=220500, phi=0.3), lambda c: partial(arrays.tone, 440.0, sr=SR, length=220500, phi=0.3), "tone"),
    pair("chirp_phi", "const", lambda c: partial(librosa.chirp, fmin=110.0, fmax=3520.0, sr=SR, length=220500, phi=0.3), lambda c: partial(arrays.chirp, fmin=110.0, fmax=3520.0, sr=SR, length=220500, phi=0.3), "chirp"),
    pair("midi_to_note_key", "const", lambda c: partial(librosa.midi_to_note, _midi(), cents=True, key="F:maj"), lambda c: partial(notation.midi_to_note, _midi(), cents=True, key="F:maj"), "midi_to_note"),
    pair("hz_to_note_key", "const", lambda c: partial(librosa.hz_to_note, _hz(), octave=False, key="F:maj"), lambda c: partial(notation.hz_to_note, _hz(), octave=False, key="F:maj"), "hz_to_note"),
    pair("note_to_midi_fractional", "const", lambda c: partial(librosa.note_to_midi, _notes(), round_midi=False), lambda c: partial(notation.note_to_midi, _notes(), round_midi=False), "note_to_midi"),
    pair("note_to_hz_round", "const", lambda c: partial(librosa.note_to_hz, _notes(), round_midi=True), lambda c: partial(notation.note_to_hz, _notes(), round_midi=True), "note_to_hz"),
    pair("wavelet_lengths_alpha", "const", lambda c: partial(librosa.filters.wavelet_lengths, freqs=FREQS, sr=SR, alpha=ALPHA), lambda c: partial(features.wavelet_lengths, freqs=FREQS, sr=SR, alpha=ALPHA), "filters.wavelet_lengths"),
    pair("wavelet_alpha", "const", lambda c: partial(librosa.filters.wavelet, freqs=FREQS, sr=SR, alpha=ALPHA, dtype=np.complex128), lambda c: partial(features.wavelet, freqs=FREQS, sr=SR, alpha=ALPHA), "filters.wavelet"),
    pair("cq_to_chroma_window", "const", lambda c: partial(librosa.filters.cq_to_chroma, 72, bins_per_octave=36, fmin=55.0, window=WINDOW, dtype=np.float64), lambda c: partial(features.cq_to_chroma, 72, bins_per_octave=36, fmin=55.0, window=WINDOW), "filters.cq_to_chroma"),
    pair("vqt_intervals", "short_mono", lambda c: partial(librosa.vqt, c.y, sr=c.sr, fmin=110.0, n_bins=48, gamma=0.0, tuning=0.0, intervals=RATIOS), lambda c: partial(AudioSamples.new_mono(c.y, c.sr).vqt, fmin=110.0, n_bins=48, gamma=0.0, tuning=0.0, intervals=RATIOS), "vqt"),
    pair("to_mono_form", "short_mono", lambda c: partial(librosa.to_mono, _stereo(c)), lambda c: partial(forms.to_mono, _stereo(c)), "to_mono"),
    pair("to_stereo_form", "short_mono", lambda c: partial(librosa.to_stereo, left=c.y, right=c.y), lambda c: partial(forms.to_stereo, left=c.y, right=c.y), "to_stereo"),
    pair("to_multi_form", "short_mono", lambda c: partial(librosa.to_multi, c.y, c.y), lambda c: partial(forms.to_multi, c.y, c.y), "to_multi"),
    pair("resample_scale", "short_mono", lambda c: partial(librosa.resample, c.y, orig_sr=c.sr, target_sr=8000, scale=True), lambda c: partial(forms.resample, c.y, orig_sr=c.sr, target_sr=8000, scale=True), "resample"),
    pair("pitch_shift_scale", "short_mono", lambda c: partial(librosa.effects.pitch_shift, c.y, sr=c.sr, n_steps=2.0, scale=True), lambda c: partial(forms.pitch_shift, c.y, sr=c.sr, n_steps=2.0, scale=True), "effects.pitch_shift"),
    pair("frame_axis", "short_mono", lambda c: partial(librosa.util.frame, _stereo(c).T, frame_length=2048, hop_length=512, axis=0), lambda c: partial(forms.frame, _stereo(c).T, frame_length=2048, hop_length=512, axis=0), "util.frame"),
    pair("get_duration_S", "const", lambda c: partial(librosa.get_duration, S=np.zeros((513, 40)), n_fft=1024, hop_length=256), lambda c: partial(forms.get_duration, S=np.zeros((513, 40)), n_fft=1024, hop_length=256), "get_duration"),
    pair("stft_out", "short_mono", lambda c: partial(librosa.stft, c.y, out=_out(c)), lambda c: partial(forms.stft, c.y, out=_out(c)), "stft"),
    pair("reassigned_S", "short_mono", lambda c: partial(librosa.reassigned_spectrogram, c.y, sr=c.sr, S=_Sc(c), n_fft=512, hop_length=128), lambda c: partial(forms.reassigned_spectrogram, c.y, sr=c.sr, S=_Sc(c), n_fft=512, hop_length=128), "reassigned_spectrogram"),
    pair("chroma_cqt_C", "short_mono", lambda c: partial(librosa.feature.chroma_cqt, C=_C(c), fmin=55.0, bins_per_octave=36, window=WINDOW), lambda c: partial(forms.chroma_cqt, C=_C(c), fmin=55.0, bins_per_octave=36, window=WINDOW), "feature.chroma_cqt"),
    pair("chroma_cens_C", "short_mono", lambda c: partial(librosa.feature.chroma_cens, C=_C(c), fmin=55.0, bins_per_octave=36, window=WINDOW), lambda c: partial(forms.chroma_cens, C=_C(c), fmin=55.0, bins_per_octave=36, window=WINDOW), "feature.chroma_cens"),
    pair("chroma_vqt_V", "short_mono", lambda c: partial(librosa.feature.chroma_vqt, V=_C(c), sr=SR, fmin=55.0, intervals="equal", bins_per_octave=36), lambda c: partial(forms.chroma_vqt, V=_C(c), fmin=55.0, bins_per_octave=36), "feature.chroma_vqt"),
    pair("tempogram_ratio_freqs", "short_mono", lambda c: partial(librosa.feature.tempogram_ratio, sr=SR, hop_length=256, **_tg_args(c)), lambda c: partial(forms.tempogram_ratio, **_tg_args(c)), "feature.tempogram_ratio"),
    pair("onset_feature", "short_mono", lambda c: partial(librosa.onset.onset_strength, y=c.y, sr=c.sr, n_fft=1024, hop_length=256, feature=_power), lambda c: partial(forms.onset_strength, y=c.y, sr=c.sr, n_fft=1024, hop_length=256, feature=_power), "onset.onset_strength"),
    pair("onset_detect_dense", "short_mono", lambda c: partial(librosa.onset.onset_detect, y=c.y, sr=c.sr, sparse=False), lambda c: partial(forms.onset_detect, y=c.y, sr=c.sr, sparse=False), "onset.onset_detect"),
    pair("beat_track_dense", "short_mono", lambda c: partial(librosa.beat.beat_track, y=c.y, sr=c.sr, sparse=False), lambda c: partial(forms.beat_track, y=c.y, sr=c.sr, sparse=False, trim=True), "beat.beat_track"),
    pair("salience_cubic", "short_mono", lambda c: partial(librosa.salience, _S(c), freqs=_hf(c), harmonics=[1, 2, 3], kind="cubic", fill_value=0.0), lambda c: partial(forms.salience, _S(c), freqs=_hf(c), harmonics=[1, 2, 3], kind="cubic", fill_value=0.0), "salience"),
    pair("interp_harmonics_cubic", "short_mono", lambda c: partial(librosa.interp_harmonics, _S(c), freqs=_hf(c), harmonics=[1, 2, 3], kind="cubic"), lambda c: partial(features.interp_harmonics, _S(c), freqs=_hf(c), harmonics=[1, 2, 3], kind="cubic"), "interp_harmonics"),
    pair("recurrence_sparse", "const", lambda c: partial(librosa.segment.recurrence_matrix, _mat(), k=5, sparse=True), lambda c: partial(sequence.recurrence_matrix, _mat(), k=5, sparse=True), "segment.recurrence_matrix"),
    pair("cross_similarity_sparse", "const", lambda c: partial(librosa.segment.cross_similarity, _mat(), _mat()[:, ::2], k=5, sparse=True), lambda c: partial(sequence.cross_similarity, _mat(), _mat()[:, ::2], k=5, sparse=True), "segment.cross_similarity"),
    pair("load_segment", "const", lambda c: partial(librosa.load, _wav(), sr=None, offset=1.0, duration=1.0), lambda c: partial(forms.read_segment, _wav(), offset=1.0, duration=1.0), "load", "reads the whole file: the reader has no seek"),
    pair("stream_blocks", "const", lambda c: (lambda: list(librosa.stream(_wav(), block_length=8, frame_length=2048, hop_length=512))), lambda c: (lambda: list(forms.stream_blocks(_wav(), block_length=8, frame_length=2048, hop_length=512))), "stream", "reads the whole file: the reader has no seek"),
)


# Option values: fractional powers, random CQT phases, time-varying frequencies, slices,
# the per-bin metrogram, sparse and custom nn_filter, callables, NumPy generators and
# file objects.

SLICES = [slice(0, 32), slice(32, 96, 2), slice(96, None)]


def _reassigned(c):
    def make():
        freqs, _, mags = librosa.reassigned_spectrogram(c.y, sr=c.sr, fill_nan=True)
        return np.abs(mags), np.asarray(freqs)

    return c.cached("opt_reassigned", make)


def _ftg(c):
    def make():
        tg = np.abs(librosa.feature.fourier_tempogram(onset_envelope=c.onset_env, sr=c.sr))[..., :-1]
        return tg, librosa.fourier_tempo_frequencies(sr=c.sr)

    return c.cached("opt_ftg", make)


def _sync_slices(c):
    n = c.mfcc.shape[1]
    return [slice(i, i + 42, 2) for i in range(0, n, 21)]


def _sparse_rec(c):
    return c.cached("opt_sparse_rec", lambda: librosa.segment.recurrence_matrix(c.chroma, mode="affinity", sparse=True))


def _bytes(fmt):
    key = "bytes_" + fmt
    if key not in _BYTES:
        path = _wav()
        if fmt == "flac":
            flac = path[:-4] + ".flac"
            io.save(flac, io.read(path, as_type=_I16()))
            path = flac
        with open(path, "rb") as f:
            _BYTES[key] = f.read()
    return _BYTES[key]


_BYTES: dict = {}


def _I16():
    from audio_samples import SampleType

    return SampleType.I16


def _F32():
    from audio_samples import SampleType

    return SampleType.F32


def _p75(x, axis):
    return np.percentile(x, 75, axis=axis)


def _from_file(read, fmt):
    import io as _io

    data = _bytes(fmt)
    return lambda: read(_io.BytesIO(data))


register(
    pair("magphase_power", "short_mono", lambda c: partial(librosa.magphase, _Sc(c), power=1.5), lambda c: partial(AudioSamples.magphase, _Sc(c), power=1.5), "magphase"),
    pair("griffinlim_cqt_random", "short_mono", lambda c: partial(librosa.griffinlim_cqt, np.abs(c.C), sr=c.sr, length=c.n, init="random", rng=0), lambda c: partial(AudioSamples.griffinlim_cqt, np.abs(c.C), sample_rate=c.sr, length=c.n, init="random", rng=0), "griffinlim_cqt"),
    pair("griffinlim_rng_generator", "short_mono", lambda c: (lambda: librosa.griffinlim(_S(c), rng=np.random.default_rng(0))), lambda c: (lambda: effects.griffinlim(_S(c), rng=np.random.default_rng(0))), "griffinlim", "rng is a fresh numpy.random.Generator on each call"),
    pair("centroid_freq2d", "short_mono", lambda c: partial(librosa.feature.spectral_centroid, S=_reassigned(c)[0], freq=_reassigned(c)[1]), lambda c: partial(features.spectral_centroid_from_spectrogram, _reassigned(c)[0], sr=c.sr, freq=_reassigned(c)[1]), "feature.spectral_centroid", "freq: reassigned frequencies"),
    pair("bandwidth_freq2d", "short_mono", lambda c: partial(librosa.feature.spectral_bandwidth, S=_reassigned(c)[0], freq=_reassigned(c)[1]), lambda c: partial(features.spectral_bandwidth_from_spectrogram, _reassigned(c)[0], sr=c.sr, freq=_reassigned(c)[1]), "feature.spectral_bandwidth", "freq: reassigned frequencies"),
    pair("rolloff_freq2d", "short_mono", lambda c: partial(librosa.feature.spectral_rolloff, S=_reassigned(c)[0], freq=_reassigned(c)[1]), lambda c: partial(features.spectral_rolloff_from_spectrogram, _reassigned(c)[0], sr=c.sr, freq=_reassigned(c)[1]), "feature.spectral_rolloff", "freq: reassigned frequencies"),
    pair("poly_freq2d", "short_mono", lambda c: partial(librosa.feature.poly_features, S=_reassigned(c)[0], freq=_reassigned(c)[1]), lambda c: partial(features.poly_features_from_spectrogram, _reassigned(c)[0], sr=c.sr, freq=_reassigned(c)[1]), "feature.poly_features", "freq: reassigned frequencies"),
    pair("onset_multi_slices", "short_mono", lambda c: partial(librosa.onset.onset_strength_multi, y=c.y, sr=c.sr, channels=SLICES), lambda c: partial(AudioSamples.new_mono(c.y, c.sr).onset_strength_multi, channels=SLICES), "onset.onset_strength_multi"),
    pair("metrogram_none", "derived", lambda c: partial(librosa.feature.metrogram, tg=_ftg(c)[0], freqs=_ftg(c)[1], aggregate=None), lambda c: partial(rhythm.metrogram, tg=_ftg(c)[0], freqs=_ftg(c)[1], aggregate=None), "feature.metrogram"),
    pair("sync_slices", "derived", lambda c: partial(librosa.util.sync, c.mfcc, _sync_slices(c)), lambda c: partial(sequence.sync, c.mfcc, _sync_slices(c)), "util.sync", "mfcc, slice(i, i + 42, 2) every 21 frames"),
    pair("nn_filter_sparse", "derived", lambda c: partial(librosa.decompose.nn_filter, c.chroma, rec=_sparse_rec(c), aggregate=np.average), lambda c: partial(effects.nn_filter, c.chroma, rec=_sparse_rec(c), aggregate=np.average), "decompose.nn_filter", "rec: sparse affinity recurrence of the chroma"),
    pair("nn_filter_max", "derived", lambda c: partial(librosa.decompose.nn_filter, c.chroma, aggregate=np.max), lambda c: partial(effects.nn_filter, c.chroma, aggregate=np.max), "decompose.nn_filter"),
    pair("nn_filter_callable", "derived", lambda c: partial(librosa.decompose.nn_filter, c.chroma, aggregate=_p75), lambda c: partial(effects.nn_filter, c.chroma, aggregate=_p75), "decompose.nn_filter", "a Python callable (75th percentile)"),
    pair("trim_ref_callable", "short", lambda c: partial(librosa.effects.trim, c.y, top_db=30, ref=np.median), lambda c: partial(c.a.trim_db, top_db=30, ref=np.median), "effects.trim"),
    pair("split_aggregate_callable", "short", lambda c: partial(librosa.effects.split, c.y, top_db=30, aggregate=np.median), lambda c: partial(c.a.split_db, top_db=30, aggregate=np.median), "effects.split"),
    pair("read_fileobj_wav", "const", lambda c: _from_file(lambda f: librosa.load(f, sr=None), "wav"), lambda c: _from_file(lambda f: io.read(f, as_type=_F32()), "wav"), "load", "10 s mono float32 WAV in a BytesIO"),
    pair("read_fileobj_flac", "const", lambda c: _from_file(lambda f: librosa.load(f, sr=None), "flac"), lambda c: _from_file(lambda f: io.read(f, as_type=_F32()), "flac"), "load", "10 s mono 16-bit FLAC in a BytesIO"),
)
