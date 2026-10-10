# Migrating from librosa

This guide maps [librosa](https://librosa.org) 1.0 onto `audio_samples`. It covers
every public function of `librosa.core` (including `convert`), `feature`,
`effects`, `onset`, `beat`, `decompose`, `filters`, `util`, `segment`,
`sequence` and `display`. For each one it lists the Rust equivalent, the cargo
feature that enables it, and how closely it matches. It then collects every
documented behavioural difference and explains how agreement with librosa is
verified.

The librosa-compatible API is checked against golden fixtures produced by
librosa 1.0.0 (see [Verification](#verification)). Where a result differs
from librosa, the cause is listed under
[Differences from librosa](#differences-from-librosa).

## Conventions

- **Audio carries its sample rate.** librosa passes `y` and `sr` separately.
  Here an `AudioSamples<T>` holds the samples, the sample rate and the channel
  layout, so methods take no `sr` argument. Free functions that work on bare
  arrays (filterbanks, unit conversions, `*_from_spectrogram` forms) take the
  rate explicitly.
- **Defaults are librosa's defaults.** Every librosa-compatible operation takes a
  config struct (`StftConfig`, `MelSpectrogramConfig`, `OnsetStrengthConfig`,
  ...) whose `Default` (and `new()`) reproduces librosa 1.0's keyword defaults.
  `X::default()` therefore gives librosa's output. Change one field with
  struct-update syntax (`StftConfig { n_fft: nzu!(1024), ..Default::default() }`)
  or with the chainable `with_<field>` builders. The operation validates the config
  and returns a typed error for invalid values. It does not panic.
- **`op` / `op_in_place`.** Every audio-to-audio operation (`time_stretch`,
  `pitch_shift`, `preemphasis`, `trim_db`, `harmonic`, ...) borrows and returns a
  new `AudioSamples`. Each also has an `_in_place` twin that mutates the receiver.
  Analysis methods return arrays or structs and have no in-place form.
- **Shapes always include the channel axis.** Frame-wise outputs are
  `ndarray::Array3<f64>` (or `Array3<Complex<f64>>`) shaped
  `(channels, n_features, n_frames)`. Mono input gives `channels == 1`, where
  librosa would drop the leading axis. Per-frame scalars (centroid, RMS, ZCR)
  are `(channels, 1, n_frames)`. Onset envelopes and pitch tracks are
  `(channels, n_frames)`.
- **Non-finite audio is an error.** Like librosa's `valid_audio` ("Audio
  buffer is not finite everywhere"), every audio-level method of
  `AudioSpectrum`, `AudioFeatures`, `AudioTonal`, `AudioConstantQ`,
  `AudioRhythm`, `AudioEffects` and `AudioPitchTracking` returns a
  `ParameterError` when the input contains a NaN or infinite sample. This is
  slightly stricter than librosa, which skips the check in a few functions
  (e.g. `preemphasis`, `remix` without `align_zeros`). Free functions on bare
  arrays (`power_to_db`, `*_from_spectrogram`, ...) do not check.
- **Everything is `f64`.** Input samples of any type (`i16`, `f32`, ...) are
  converted with the crate's audio-aware conversion. Outputs are always `f64` or
  `Complex<f64>`; librosa keeps `float32`/`complex64` for `float32` input.
- **Names.** librosa's name is kept unless it clashes with an existing method.
  Clashing frame-wise variants take a `_frames` suffix (`mfcc_frames`,
  `spectral_centroid_frames`, `rms_frames`, ...), and the complex STFT is
  `stft_matrix` / `istft_matrix`. The older methods with the plain names
  (`AudioTransforms::mfcc`, `AudioStatistics::spectral_centroid`, ...) keep
  their original behaviour, which is *not* librosa's (see
  [Existing APIs that differ from librosa](#existing-apis-that-differ-from-librosa)).
- **Where things live.** The traits (`AudioSpectrum`, `AudioFeatures`,
  `AudioTonal`, `AudioConstantQ`, `AudioRhythm`, `AudioEffects`,
  `AudioPitchTracking`) and their config types are re-exported at the crate
  root. Free functions live in their modules: `operations::spectrum`,
  `operations::filterbanks`, `operations::features`, `operations::tonal`,
  `operations::constant_q`, `operations::rhythm`, `operations::effects`,
  `operations::pitch_tracking` and `operations::sequence`. Unit conversions live
  in `utils::audio_math`.

### Feature flags

| Feature | Enables |
|---|---|
| `transforms` | `AudioSpectrum`, `operations::spectrum`, `operations::filterbanks` |
| `transforms` + `resampling` | `AudioConstantQ` (CQT, VQT, iCQT); `chroma_cqt`, `chroma_cens`, `chroma_vqt` and `tonnetz` also need `feature-extraction` |
| `feature-extraction` | `AudioFeatures`, `AudioTonal` (implies `transforms`) |
| `beat-tracking` | `AudioRhythm` (onset strength and detection, tempograms, tempo, beats, PLP) |
| `effects` | `AudioEffects` and the effects free functions (implies `transforms`, `resampling`, `editing`, `sequence`) |
| `pitch-analysis` | `AudioPitchTracking` (YIN, pYIN, salience; implies `transforms`, `sequence`) |
| `sequence` | `operations::sequence` (DTW, Viterbi, recurrence, `sync`) |
| `resampling` | `resample`, `resample_by_ratio`, `AudioProcessing::resample` |
| `processing` | `mu_law_encode`, `mu_law_decode`, `AudioProcessing::mu_compress` |
| (none) | `utils::audio_math` conversions, `sine_wave` / `chirp` generators |

`full_no_plotting` (and `full`) enable all of them.

## Common operations side by side

Each example below compiles and runs as a doctest with
`--features full_no_plotting`. In place of `librosa.load`, read a file with
[`audio_samples_io`](https://crates.io/crates/audio_samples_io) or build an
`AudioSamples` from an `ndarray` (`AudioSamples::new_mono`,
`AudioSamples::new_multi_channel`). The examples use a generated tone.

### STFT and inverse STFT

```rust
use audio_samples::{AudioSamples, AudioSpectrum, StftConfig, sample_rate, sine_wave};
use std::time::Duration;

let y = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(22050), 0.5);

// D = librosa.stft(y)  -> librosa shape (1025, 44); here (channels, 1025, 44)
let d = y.stft_matrix(&StftConfig::default()).unwrap();
assert_eq!(d.dim(), (1, 1025, 44));

// librosa.istft(D, length=len(y))
let n = y.samples_per_channel();
let y_hat =
    AudioSamples::<f32>::istft_matrix(&d, &StftConfig::default(), sample_rate!(22050), Some(n))
        .unwrap();
assert_eq!(y_hat.samples_per_channel(), n);
```

### Mel spectrogram in dB

```rust
use audio_samples::operations::spectrum::power_to_db;
use audio_samples::{AudioSpectrum, DbConfig, DbReference, MelSpectrogramConfig, sample_rate, sine_wave};
use std::time::Duration;

let y = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(22050), 0.5);

// S = librosa.feature.melspectrogram(y=y, sr=sr)   (128 Slaney mel bands, power 2)
let s = y.melspectrogram(&MelSpectrogramConfig::default()).unwrap();
assert_eq!(s.dim(), (1, 128, 44));

// librosa.power_to_db(S, ref=np.max)   (top_db = 80, applied per channel)
let db = DbConfig { reference: DbReference::Max, ..DbConfig::librosa_power() };
let s_db = power_to_db(&s, &db).unwrap();
let max = s_db.iter().copied().fold(f32::NEG_INFINITY, f32::max);
let min = s_db.iter().copied().fold(f32::INFINITY, f32::min);
assert!(max.abs() < 1e-9 && min >= -80.0 - 1e-9);
```

### MFCC and deltas

```rust
use audio_samples::operations::spectrum::delta;
use audio_samples::{AudioSpectrum, DeltaConfig, MfccConfig, sample_rate, sine_wave};
use std::time::Duration;

let y = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(22050), 0.5);

// mfcc = librosa.feature.mfcc(y=y, sr=sr)   (20 coefficients, DCT-II ortho)
let mfcc = y.mfcc_frames(&MfccConfig::librosa()).unwrap();
assert_eq!(mfcc.dim(), (1, 20, 44));

// librosa.feature.delta(mfcc) and librosa.feature.delta(mfcc, order=2)
let d1 = delta(&mfcc, &DeltaConfig::default()).unwrap();
let d2 = delta(&mfcc, &DeltaConfig { order: 2, ..DeltaConfig::default() }).unwrap();
assert_eq!(d1.dim(), mfcc.dim());
assert_eq!(d2.dim(), mfcc.dim());
```

### Spectral features and chroma

```rust
use audio_samples::{
    AudioFeatures, AudioTonal, ChromaStftConfig, RmsConfig, SpectralCentroidConfig,
    ZeroCrossingRateConfig, sample_rate, sine_wave,
};
use std::time::Duration;

let y = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(22050), 0.5);

// librosa.feature.spectral_centroid / rms / zero_crossing_rate: (1, 1, 44) each
let centroid = y.spectral_centroid_frames(&SpectralCentroidConfig::default()).unwrap();
let rms = y.rms_frames(&RmsConfig::default()).unwrap();
let zcr = y.zero_crossing_rate_frames(&ZeroCrossingRateConfig::default()).unwrap();
assert_eq!(centroid.dim(), (1, 1, 44));
assert_eq!(rms.dim(), (1, 1, 44));
assert_eq!(zcr.dim(), (1, 1, 44));
assert!((centroid[[0, 0, 20]] - 440.0).abs() < 20.0);

// librosa.feature.chroma_stft(y=y, sr=sr): A (pitch class 9) is the maximum
let chroma = y.chroma_stft(&ChromaStftConfig::default()).unwrap();
assert_eq!(chroma.dim(), (1, 12, 44));
assert_eq!(chroma[[0, 9, 20]], 1.0);
```

### Onsets, tempo and beats

```rust
use audio_samples::{
    AudioRhythm, AudioSamples, BeatTrackConfig, OnsetDetectConfig, OnsetStrengthConfig,
    sample_rate,
};
use ndarray::Array1;

// A click every 0.5 s (120 BPM) for 4 s.
let sr = 22050;
let mut y = Array1::<f32>::zeros(4 * sr);
for k in 0..8 {
    for n in 0..200 {
        y[k * sr / 2 + n] = (-(n as f32) / 30.0).exp();
    }
}
let audio = AudioSamples::new_mono(y, sample_rate!(22050)).unwrap();

// librosa.onset.onset_strength(y=y, sr=sr): (channels, n_frames)
let env = audio.onset_strength(&OnsetStrengthConfig::librosa()).unwrap();
assert_eq!(env.nrows(), 1);

// librosa.onset.onset_detect(y=y, sr=sr, units="time")
let onsets = audio.onset_detect(&OnsetDetectConfig::librosa()).unwrap();
let onset_times: Vec<f64> = onsets.times();
assert!(onset_times.len() >= 7);

// tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
let track = audio.beat_track(&BeatTrackConfig::librosa()).unwrap();
assert!((track.tempo - 120.0).abs() < 5.0);
let beat_times = track.beats.times();
assert!(!beat_times.is_empty());
```

### Effects and resampling

```rust
use audio_samples::operations::types::ResamplingQuality;
use audio_samples::{
    AudioEffects, AudioProcessing, PitchShiftConfig, SilenceConfig, StftConfig, sample_rate,
    sine_wave,
};
use std::time::Duration;

let y = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(22050), 0.5);
let n = y.samples_per_channel().get();

// librosa.effects.time_stretch(y, rate=2.0): half the length
let fast = y.time_stretch(2.0, &StftConfig::default()).unwrap();
assert_eq!(fast.samples_per_channel().get(), n / 2);

// librosa.effects.pitch_shift(y, sr=sr, n_steps=2): same length
let up = y.pitch_shift(2.0, &PitchShiftConfig::default()).unwrap();
assert_eq!(up.samples_per_channel().get(), n);

// y_trimmed, (start, end) = librosa.effects.trim(y)  (top_db = 60)
let (trimmed, (start, end)) = y.trim_db(&SilenceConfig::default()).unwrap();
assert_eq!(trimmed.samples_per_channel().get(), end - start);

// librosa.resample(y, orig_sr=22050, target_sr=16000)
let y16 = y.resample(sample_rate!(16000), ResamplingQuality::High).unwrap();
assert_eq!(y16.samples_per_channel().get(), 16000);
```

### Pitch tracking

```rust
use audio_samples::{AudioPitchTracking, PyinConfig, YinConfig, sample_rate, sine_wave};
use std::time::Duration;

let y = sine_wave::<f32>(220.0, Duration::from_millis(500), sample_rate!(22050), 0.5);

// librosa.yin(y, fmin=65.4, fmax=2093.0, sr=sr): (channels, n_frames)
let f0 = y.yin(&YinConfig::new(65.4, 2093.0)).unwrap();
assert!((f0[[0, 11]] - 220.0).abs() < 1.0);

// f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=65.4, fmax=2093.0, sr=sr)
let r = y.pyin(&PyinConfig::new(65.4, 2093.0)).unwrap();
assert!(r.voiced_flag[[0, 11]]);
assert!((r.f0[[0, 11]] - 220.0).abs() < 2.0);
assert_eq!(r.voiced_prob.dim(), r.f0.dim());
```

## Equivalence tables

Status values:

- **equivalent**: same results as librosa with the same parameters, to
  floating-point round-off. Event outputs (frames, beats, paths) are identical.
- **equivalent with differences**: the same algorithm, with a documented
  difference in output range, dtype, options or edge cases (see the notes column
  and [Differences from librosa](#differences-from-librosa)).
- **not provided**: no counterpart, with the reason.

In the second column, `Trait::method` is a trait method on `AudioSamples`
(traits are re-exported at the crate root), and `module::function` is a free
function under `audio_samples::`.

### `librosa` core: audio

| librosa | audio_samples | feature | status | notes |
|---|---|---|---|---|
| `load` | not in this crate | – | not provided: file I/O | Use `audio_samples_io` |
| `loadx` | – | – | not provided: file I/O | |
| `stream` | – | – | not provided: file I/O | Block streaming belongs to `audio_samples_io`; `StreamingResampler` resamples chunked audio |
| `get_samplerate` | – | – | not provided: file I/O | `AudioSamples::sample_rate` gives the rate of loaded audio |
| `get_duration` | `AudioSamples::duration_seconds` | – | equivalent with differences | Audio only; no `S=` or `path=` forms |
| `resample` | `AudioProcessing::resample`, `AudioProcessing::resample_by_ratio`, `resample`, `resample_by_ratio` | `resampling` | equivalent with differences | Own Kaiser-windowed sinc instead of soxr; output length is exactly `ceil(n * out / in)` |
| `to_mono` | `AudioChannelOps::to_mono(MonoConversionMethod::Average)` | `channels` | equivalent | |
| `to_stereo` | `AudioChannelOps::to_stereo(StereoConversionMethod::...)` | `channels` | equivalent with differences | `Duplicate` is `to_stereo(left=m, right=m)`; `Left`/`Right` give `[m, 0]` / `[0, m]` for mono input (on multi-channel input they extract channel 0 / 1 as mono, kept from 2.0). There is no two-signal form; use `AudioEditing::stack` |
| `to_multi` | `AudioEditing::stack` | `editing` | equivalent with differences | No pad or trim to a common length |
| `autocorrelate` | `AudioStatistics::autocorrelation` | `statistics` | equivalent with differences | Returns the unbiased estimate: `r[k] * (n - k)` equals librosa's raw sum. librosa also autocorrelates every channel |
| `lpc` | `AudioEffects::lpc`, `operations::effects::lpc` | `effects` | equivalent | Burg's method, one row per channel |
| `zero_crossings` | `AudioStatistics::zero_crossings` | `statistics` | equivalent with differences | Returns a count, not a per-sample boolean mask; no `threshold` / `ref_magnitude` / `zero_pos` options. The librosa-compatible frame-wise rate is `AudioFeatures::zero_crossing_rate_frames` |
| `clicks` | `operations::effects::clicks` (`ClicksConfig`, `ClickPositions`) | `effects` | equivalent with differences | Returns `f64` audio. No multichannel custom click |
| `tone` | `sine_wave` | – | equivalent with differences | No `phi`; the duration is a `Duration` |
| `chirp` | `chirp` (linear), `exponential_chirp` (logarithmic, librosa's default `linear=False`) | – | equivalent with differences | No `phi` |
| `mu_compress` | `AudioProcessing::mu_compress`, `mu_law_encode` | `processing` | equivalent | `mu_compress` equals `quantize=False`; `mu_law_encode` returns librosa's default integer codes (`quantize=True`) |
| `mu_expand` | `AudioProcessing::mu_expand`, `mu_law_decode` | `processing` | equivalent | As above |

### `librosa` core: spectral representations

| librosa | audio_samples | feature | status | notes |
|---|---|---|---|---|
| `stft` | `AudioSpectrum::stft_matrix`, `operations::spectrum::stft` | `transforms` | equivalent | All `pad_mode`s, `win_length`, any `Window`. `AudioSpectrum::spectrogram_matrix` gives `abs(stft) ** power` |
| `istft` | `AudioSpectrum::istft_matrix`, `operations::spectrum::istft` | `transforms` | equivalent with differences | `n_fft` comes from the config, and a bin-count mismatch is an error; `length = 0` is rejected |
| `magphase` | `AudioTransforms::magphase` | `transforms` | equivalent with differences | 2-D input only; `power` is an integer (librosa accepts any float) |
| `reassigned_spectrogram` | `AudioFeatures::reassigned_spectrogram` (`ReassignedSpectrogramConfig`) | `feature-extraction` | equivalent | |
| `phase_vocoder` | `operations::effects::phase_vocoder`, `operations::effects::phase_vocoder_at` (`t_out`) | `effects` | equivalent with differences | The deprecated `hop_length` / `n_fft` arguments are not accepted |
| `griffinlim` | `operations::effects::griffinlim`, `AudioEffects::griffinlim_audio` (`GriffinLimConfig`) | `effects` | equivalent with differences | `GriffinLimInit::ZeroPhase` matches `init=None`. Random phases come from this crate's RNG |
| `iirt` | – | – | not provided: deferred | Needs elliptic order selection (`ellipord`) for its semitone filterbank |
| `fmt` | – | – | not provided: deferred | Rarely used; needs a cubic-spline exponential resampler |
| `pcen` | `operations::features::pcen`, `operations::features::pcen_with_state` (`zi`/`zf`) | `feature-extraction` | equivalent | |
| `amplitude_to_db` | `operations::spectrum::amplitude_to_db` (`DbConfig`) | `transforms` | equivalent | `ref` as `DbReference::Value` or `DbReference::Max`; `amin`, `top_db`, per-channel `DbAxes` |
| `power_to_db` | `operations::spectrum::power_to_db` | `transforms` | equivalent | As above |
| `db_to_amplitude` | `operations::spectrum::db_to_amplitude` | `transforms` | equivalent | The scalar `utils::audio_math::db_to_amplitude` has no `ref` |
| `db_to_power` | `operations::spectrum::db_to_power` | `transforms` | equivalent | As above |
| `perceptual_weighting` | `operations::features::perceptual_weighting` | `feature-extraction` | equivalent | |
| `frequency_weighting` | `operations::features::frequency_weighting` (`FrequencyWeighting`) | `feature-extraction` | equivalent | |
| `multi_frequency_weighting` | `operations::features::multi_frequency_weighting` | `feature-extraction` | equivalent | |
| `A_weighting` | `operations::features::a_weighting` | `feature-extraction` | equivalent | |
| `B_weighting` | `operations::features::b_weighting` | `feature-extraction` | equivalent | |
| `C_weighting` | `operations::features::c_weighting` | `feature-extraction` | equivalent | |
| `D_weighting` | `operations::features::d_weighting` | `feature-extraction` | equivalent | |
| `Z_weighting` | `operations::features::z_weighting` | `feature-extraction` | equivalent | |

### `librosa` core: constant-Q

| librosa | audio_samples | feature | status | notes |
|---|---|---|---|---|
| `cqt` | `AudioConstantQ::cqt` (`CqtConfig`) | `transforms` + `resampling` | equivalent with differences | Octave decimation uses the crate's resampler, not soxr |
| `vqt` | `AudioConstantQ::vqt` (`VqtConfig`) | `transforms` + `resampling` | equivalent with differences | As `cqt`; only equal-tempered `intervals` |
| `pseudo_cqt` | `AudioConstantQ::pseudo_cqt` | `transforms` + `resampling` | equivalent | No resampling involved |
| `hybrid_cqt` | `AudioConstantQ::hybrid_cqt` | `transforms` + `resampling` | equivalent with differences | Low octaves as `cqt` |
| `icqt` | `AudioConstantQ::icqt` | `transforms` + `resampling` | equivalent with differences | Upsampling uses the crate's resampler |
| `griffinlim_cqt` | `AudioConstantQ::griffinlim_cqt` | `transforms` + `resampling` | equivalent with differences | Only the deterministic `init=None` start |

### `librosa` core: pitch, tuning and harmonics

| librosa | audio_samples | feature | status | notes |
|---|---|---|---|---|
| `yin` | `AudioPitchTracking::yin`, `operations::pitch_tracking::yin` (`YinConfig`) | `pitch-analysis` | equivalent | `fmin`/`fmax` are required, as in librosa: `YinConfig::new(fmin, fmax)` |
| `pyin` | `AudioPitchTracking::pyin`, `operations::pitch_tracking::pyin` (`PyinConfig`, `PyinResult`) | `pitch-analysis` | equivalent | `UnvoicedFill` is librosa's `fill_na` |
| `piptrack` | `AudioTonal::piptrack`, `operations::tonal::piptrack_from_spectrogram` | `feature-extraction` | equivalent | |
| `estimate_tuning` | `AudioTonal::estimate_tuning`, `operations::tonal::estimate_tuning_from_signal`, `operations::tonal::estimate_tuning_from_spectrogram` | `feature-extraction` | equivalent | Pools all channels, as librosa does |
| `pitch_tuning` | `operations::tonal::pitch_tuning` | `feature-extraction` | equivalent | |
| `salience` | `operations::pitch_tracking::salience` (`SalienceConfig`) | `pitch-analysis` | equivalent | |
| `interp_harmonics` | `operations::pitch_tracking::interp_harmonics` (`FrequencyGrid`) | `pitch-analysis` | equivalent | |
| `f0_harmonics` | `operations::pitch_tracking::f0_harmonics` | `pitch-analysis` | equivalent | |

### `librosa` core: unit conversions (`librosa.convert`)

All of these are in `utils::audio_math` and need no feature. They are scalar
functions; the frame and time helpers also have `_slice` forms for arrays.

| librosa | audio_samples | status | notes |
|---|---|---|---|
| `frames_to_samples` | `frames_to_samples`, `frames_to_samples_slice` | equivalent | Negative frames and the `n_fft` offset are supported |
| `frames_to_time` | `frames_to_time_with_offset`, `frames_to_time_with_offset_slice` | equivalent | The older `frames_to_time` has no `n_fft` offset |
| `samples_to_frames` | `samples_to_frames`, `samples_to_frames_slice` | equivalent | |
| `samples_to_time` | `samples_to_time` | equivalent | |
| `time_to_frames` | `time_to_frames_floor`, `time_to_frames_floor_slice` | equivalent | The older `time_to_frames` rounds to the nearest frame |
| `time_to_samples` | `time_to_samples`, `time_to_samples_slice` | equivalent | |
| `blocks_to_frames`, `blocks_to_samples`, `blocks_to_time` | – | not provided: block streaming is file I/O | One multiplication each |
| `times_like` | `times_like` | equivalent | |
| `samples_like` | `samples_like` | equivalent | |
| `fft_frequencies` | `fft_frequencies` | equivalent | |
| `mel_frequencies` | `mel_frequencies_with_scale(.., MelScale::Slaney)` | equivalent | The older `mel_frequencies` / `mel_scale` use the HTK scale |
| `hz_to_mel` | `hz_to_mel_with_scale` | equivalent | `MelScale::Slaney` is librosa's default, `MelScale::Htk` is `htk=True` |
| `mel_to_hz` | `mel_to_hz_with_scale` | equivalent | As above |
| `cqt_frequencies` | `cqt_frequencies` | equivalent | |
| `tempo_frequencies` | `operations::rhythm::tempo_frequencies` (`beat-tracking`) | equivalent | |
| `fourier_tempo_frequencies` | `operations::rhythm::fourier_tempo_frequencies` (`beat-tracking`) | equivalent | |
| `hz_to_midi` | `hz_to_midi` | equivalent | |
| `midi_to_hz` | `midi_to_hz` | equivalent | |
| `hz_to_note` | `frequency_to_note` | equivalent with differences | Returns `(name, cents)`; ASCII `#`; MIDI 0..=127 only (otherwise an error) |
| `midi_to_note` | `midi_to_note` | equivalent with differences | ASCII `#` (librosa `unicode=False`); integer MIDI 0..=127 only; no cents, no `octave=False` |
| `note_to_hz` | `note_to_frequency` | equivalent with differences | Same grammar and limits as `note_to_midi` |
| `note_to_midi` | `note_to_midi` | equivalent with differences | librosa's full grammar (Unicode accidentals, cents suffix, signed octaves); the octave is required and the result is a `u8` in 0..=127 |
| `hz_to_octs` | `hz_to_octs` | equivalent | |
| `octs_to_hz` | `octs_to_hz` | equivalent | |
| `A4_to_tuning` | `a4_to_tuning` | equivalent | |
| `tuning_to_A4` | `tuning_to_a4` | equivalent | |
| `key_to_notes` | `key_to_notes` | equivalent | |
| `key_to_degrees` | `key_to_degrees` | equivalent | |
| `interval_frequencies`, `pythagorean_intervals`, `plimit_intervals` | – | not provided: deferred | Tuning-system generators; low use |
| `interval_to_fjs`, `hz_to_fjs`, `fifths_to_note` | – | not provided: deferred | Functional Just System notation; low use |
| `hz_to_svara_c`, `hz_to_svara_h`, `midi_to_svara_c`, `midi_to_svara_h`, `note_to_svara_c`, `note_to_svara_h` | – | not provided: deferred | Carnatic / Hindustani notation tables; low use |
| `list_mela`, `list_thaat`, `mela_to_degrees`, `mela_to_svara`, `thaat_to_degrees` | – | not provided: deferred | As above |

### `librosa.feature`

| librosa | audio_samples | feature | status | notes |
|---|---|---|---|---|
| `melspectrogram` | `AudioSpectrum::melspectrogram`, `operations::spectrum::melspectrogram_from_spectrogram` | `transforms` | equivalent with differences | `f64` mel basis (librosa: `float32`, ~1e-7 relative) |
| `mfcc` | `AudioSpectrum::mfcc_frames` (`MfccConfig`), `operations::spectrum::mfcc_from_mel_db` | `transforms` | equivalent with differences | `f64` mel basis; `n_mfcc > n_mels` is an error |
| `delta` | `operations::spectrum::delta` (`DeltaConfig`, `DeltaMode`) | `transforms` | equivalent | |
| `stack_memory` | `operations::spectrum::stack_memory` | `transforms` | equivalent | |
| `spectral_centroid` | `AudioFeatures::spectral_centroid_frames`, `operations::features::spectral_centroid_from_spectrogram` | `feature-extraction` | equivalent | |
| `spectral_bandwidth` | `AudioFeatures::spectral_bandwidth_frames`, `operations::features::spectral_bandwidth_from_spectrogram` | `feature-extraction` | equivalent | |
| `spectral_contrast` | `AudioFeatures::spectral_contrast_frames`, `operations::features::spectral_contrast_from_spectrogram` | `feature-extraction` | equivalent with differences | A band with no FFT bin is an error |
| `spectral_flatness` | `AudioFeatures::spectral_flatness_frames`, `operations::features::spectral_flatness_from_spectrogram` | `feature-extraction` | equivalent | |
| `spectral_rolloff` | `AudioFeatures::spectral_rolloff_frames`, `operations::features::spectral_rolloff_from_spectrogram` | `feature-extraction` | equivalent | |
| `poly_features` | `AudioFeatures::poly_features`, `operations::features::poly_features_from_spectrogram` | `feature-extraction` | equivalent with differences | Rank-deficient fits are an error (NumPy warns and returns a minimum-norm solution) |
| `rms` | `AudioFeatures::rms_frames`, `operations::features::rms_from_spectrogram` | `feature-extraction` | equivalent with differences | Computed in `f64` (librosa: `float32`, ~1e-7 relative) |
| `zero_crossing_rate` | `AudioFeatures::zero_crossing_rate_frames` | `feature-extraction` | equivalent | |
| `chroma_stft` | `AudioTonal::chroma_stft`, `operations::tonal::chroma_stft_from_spectrogram` | `feature-extraction` | equivalent with differences | `f64` chroma basis; with odd `n_fft` the configured `n_fft` is used for tuning |
| `chroma_cqt` | `AudioConstantQ::chroma_cqt` | `transforms` + `resampling` + `feature-extraction` | equivalent with differences | Inherits the CQT resampler difference |
| `chroma_cens` | `AudioConstantQ::chroma_cens`, `operations::constant_q::chroma_cens_from_chroma` | as above | equivalent with differences | As `chroma_cqt` |
| `chroma_vqt` | `AudioConstantQ::chroma_vqt` | as above | equivalent with differences | As `vqt` |
| `tonnetz` | `AudioConstantQ::tonnetz`, `operations::constant_q::tonnetz_from_chroma` | as above | equivalent with differences | As `chroma_cqt` (from audio) |
| `tempogram` | `AudioRhythm::tempogram`, `operations::rhythm::tempogram_from_envelope` | `beat-tracking` | equivalent | |
| `fourier_tempogram` | `AudioRhythm::fourier_tempogram`, `operations::rhythm::fourier_tempogram_from_envelope` | `beat-tracking` | equivalent | |
| `tempo` | `AudioRhythm::tempo`, `operations::rhythm::tempo_from_envelope`, `operations::rhythm::tempo_from_tempogram` | `beat-tracking` | equivalent | Returns `(channels, 1)`, or `(channels, n_frames)` when time-varying |
| `tempogram_ratio` | `AudioRhythm::tempogram_ratio`, `operations::rhythm::tempogram_ratio_from_tempogram` | `beat-tracking` | equivalent | |
| `hybrid_tempogram` | `AudioRhythm::hybrid_tempogram`, `operations::rhythm::hybrid_tempogram_from_envelope` | `beat-tracking` | equivalent | |
| `metrogram` | `operations::rhythm::metrogram` | `beat-tracking` | equivalent with differences | `aggregate=None` (the 4-D per-bin product) is not offered |
| `inverse.mel_to_stft`, `inverse.mel_to_audio` | – | – | not provided: deferred | Needs a non-negative least-squares solver; compose `griffinlim` with your own mel inversion |
| `inverse.mfcc_to_mel`, `inverse.mfcc_to_audio` | – | – | not provided: deferred | As above |

### `librosa.effects`

All of these are `AudioEffects` methods (feature `effects`), each with an
`_in_place` variant where the output is audio.

| librosa | audio_samples | status | notes |
|---|---|---|---|
| `time_stretch` | `AudioEffects::time_stretch` | equivalent | Round-off is amplified on input containing exact digital silence (ill-conditioned phases of ~0-magnitude bins): ~30 dB SNR there, the same as librosa against itself with a 1e-15 perturbation |
| `pitch_shift` | `AudioEffects::pitch_shift` (`PitchShiftConfig`) | equivalent with differences | The final resampling uses the crate's resampler, not soxr (about 68 dB SNR against librosa) |
| `preemphasis` | `AudioEffects::preemphasis`, `operations::effects::preemphasis_with_state` (`zi`, returns `zf`) | equivalent | |
| `deemphasis` | `AudioEffects::deemphasis`, `operations::effects::deemphasis_with_state` | equivalent | |
| `trim` | `AudioEffects::trim_db` (`SilenceConfig`) | equivalent with differences | Returns `(trimmed, (start, end))`. Frame RMS is `f64` (librosa: `float32`), so a frame within ~1e-6 dB of the threshold can be classified differently |
| `split` | `AudioEffects::split_db`, `operations::effects::nonsilent_frames` | equivalent with differences | As `trim` |
| `remix` | `AudioEffects::remix` | equivalent | |
| `hpss` | `AudioEffects::harmonic_percussive` (`HarmonicPercussiveConfig`) | equivalent with differences | `config.stft.window` is used (librosa accepts `window` but always uses Hann) |
| `harmonic` | `AudioEffects::harmonic` | equivalent with differences | As `hpss` |
| `percussive` | `AudioEffects::percussive` | equivalent with differences | As `hpss` |

### `librosa.onset` and `librosa.beat`

| librosa | audio_samples | feature | status | notes |
|---|---|---|---|---|
| `onset.onset_strength` | `AudioRhythm::onset_strength` (`OnsetStrengthConfig`), `operations::rhythm::onset_strength_from_db` | `beat-tracking` | equivalent | `f64` mel basis |
| `onset.onset_strength_multi` | `AudioRhythm::onset_strength_multi`, `operations::rhythm::onset_strength_multi_from_db` | `beat-tracking` | equivalent | |
| `onset.onset_detect` | `AudioRhythm::onset_detect` (`OnsetDetectConfig`), `operations::rhythm::onset_detect_from_envelope` | `beat-tracking` | equivalent with differences | Mono audio only. Returns `FrameEvents` (frames, plus `samples()` and `times()` for librosa's `units=`) |
| `onset.onset_backtrack` | `operations::rhythm::onset_backtrack`, or `OnsetDetectConfig::backtrack` | `beat-tracking` | equivalent with differences | An empty event list returns an empty vector (librosa raises) |
| `beat.beat_track` | `AudioRhythm::beat_track` (`BeatTrackConfig`, `BeatTrack`), `operations::rhythm::beat_track_from_envelope`, `operations::rhythm::beat_track_with_bpm` | `beat-tracking` | equivalent with differences | Mono audio only. `beat_track_with_bpm` takes a fixed or time-varying tempo |
| `beat.plp` | `AudioRhythm::plp` (`PlpConfig`), `operations::rhythm::plp_from_envelope` | `beat-tracking` | equivalent | |

### `librosa.decompose`

| librosa | audio_samples | feature | status | notes |
|---|---|---|---|---|
| `decompose` | – | – | not provided: deferred | NMF through scikit-learn; no decomposition library in this crate |
| `hpss` | `operations::effects::hpss_spectrogram`, `operations::effects::hpss_magnitude`, `operations::effects::hpss_masks` (`HpssMaskConfig`) | `effects` | equivalent | Kernel sizes, power, margins, `mask=True` |
| `nn_filter` | `operations::effects::nn_filter`, `operations::effects::nn_filter_recurrence` (`NnAggregate`) | `effects` | equivalent | |

### `librosa.filters`

| librosa | audio_samples | feature | status | notes |
|---|---|---|---|---|
| `get_window` | `operations::filterbanks::get_window` (`Window`) | `transforms` | equivalent | Periodic (`fftbins=True`) or symmetric |
| `mel` | `operations::filterbanks::mel_filterbank` (`MelFilterbankConfig`, `MelFilterNorm`, `MelScale`) | `transforms` | equivalent with differences | `f64` weights (librosa: `float32`); empty filters are returned silently as zero rows (librosa warns) |
| `chroma` | `operations::tonal::chroma_filterbank` (`ChromaFilterbankConfig`) | `feature-extraction` | equivalent with differences | `f64` weights (librosa: `float32`) |
| `cq_to_chroma` | `operations::constant_q::cq_to_chroma` | `transforms` + `resampling` | equivalent with differences | No smoothing `window` |
| `wavelet` | `operations::constant_q::wavelet` (`WaveletConfig`) | `transforms` + `resampling` | equivalent with differences | `Complex<f64>` filters (librosa: `complex64`) |
| `wavelet_lengths` | `operations::constant_q::wavelet_lengths` | `transforms` + `resampling` | equivalent with differences | The single-frequency-with-explicit-`alpha` form is internal |
| `window_bandwidth` | `operations::filterbanks::window_bandwidth` | `transforms` | equivalent with differences | Always evaluates the formula; librosa returns a hard-coded table value for named windows (up to ~1e-4 relative apart) |
| `window_sumsquare` | `operations::filterbanks::window_sumsquare` | `transforms` | equivalent with differences | No `norm`; `f64` output |
| `diagonal_filter` | – | – | not provided: deferred | Needs SciPy's spline image rotation; only used by `segment.path_enhance` |
| `mr_frequencies` | – | – | not provided: deferred | Only used by `iirt` |
| `semitone_filterbank` | – | – | not provided: deferred | Needs elliptic order selection (`ellipord`) |

### `librosa.util`

| librosa | audio_samples | feature | status | notes |
|---|---|---|---|---|
| `frame` | `operations::spectrum::frame` | `transforms` | equivalent with differences | 1-D input, last-axis framing. `AudioSampleIterators::windows` pads trailing frames; `windows_ref` / `PaddingMode::Skip` gives librosa's count |
| `normalize` | `operations::tonal::normalize_array` (`NormalizeConfig`) | `feature-extraction` | equivalent | inf / -inf / 0 / p norms, `axis`, `threshold`, `fill` |
| `peak_pick` | `operations::rhythm::peak_pick` (`PeakPickConfig`, `PeakPickMethod`) | `beat-tracking` | equivalent | |
| `sync` | `operations::sequence::sync`, `operations::sequence::sync_with` (`SyncAggregate`) | `sequence` | equivalent with differences | Frame indices only (no slices); an empty index list with `pad = true` gives one segment |
| `softmask` | `operations::effects::softmask` | `effects` | equivalent | |
| `fix_length` | `AudioEditing::pad_samples_right` | `editing` | equivalent with differences | Pads only; it does not truncate a longer signal |
| `localmax`, `localmin` | – | – | not provided: internal helpers | Used privately by peak picking and beat tracking |
| `match_events` | – | – | not provided: internal helper | Used privately by `remix(align_zeros)` |
| `nnls` | – | – | not provided: deferred | Only needed by `feature.inverse` |
| `pad_center` | – | – | not provided: internal | `StftConfig::win_length` pads the window to `n_fft` |
| `fix_frames` | – | – | not provided: out of scope | Index-set helper; `sync` takes frame indices directly |
| `buf_to_float` | – | – | not provided: out of scope | Byte decoding is file I/O. For integer-to-float use `to_format` / `as_f64`, which scale asymmetrically (see below) |
| `valid_audio` | – | – | not provided: out of scope | `AudioSamples` enforces its invariants by construction |
| `stack` | – | – | not provided: out of scope | Generic NumPy stacking (`AudioEditing::stack` stacks audio channels) |
| `abs2`, `phasor`, `tiny`, `expand_to`, `dtype_c2r`, `dtype_r2c`, `MAX_MEM_BLOCK` | – | – | not provided: out of scope | NumPy one-liners and dtype plumbing (`Complex::from_polar`, `f64::MIN_POSITIVE`) |
| `axis_sort`, `sparsify_rows`, `shear`, `cyclic_gradient`, `fill_off_diagonal`, `index_to_slice`, `count_unique`, `is_unique`, `interp_broadcast` | – | – | not provided: out of scope | Generic array utilities; the ones the ported algorithms need exist privately |
| `match_intervals` | – | – | not provided: out of scope | Annotation alignment, no audio meaning |
| `is_positive_int`, `valid_int`, `valid_intervals`, `rename_kw`, `Deprecated` | – | – | not provided: out of scope | Python argument validation and deprecation plumbing; Rust uses types and typed errors |
| `example`, `ex`, `example_info`, `list_examples`, `find_files`, `cite` | – | – | not provided: out of scope | Example-data download, file discovery and citation helpers |

### `librosa.segment` and `librosa.sequence`

All of these are free functions in `operations::sequence` (feature
`sequence`) on 2-D `(n_features, n_frames)` arrays.

| librosa | audio_samples | status | notes |
|---|---|---|---|
| `segment.recurrence_matrix` | `recurrence_matrix` (`RecurrenceConfig`, `RecurrenceMode`, `Bandwidth`) | equivalent with differences | Dense only (no `sparse=True`); connectivity entries are `0.0`/`1.0`; `axis=-1` only |
| `segment.cross_similarity` | `cross_similarity` (`CrossSimilarityConfig`) | equivalent with differences | As `recurrence_matrix` |
| `segment.recurrence_to_lag` | `recurrence_to_lag` | equivalent | |
| `segment.lag_to_recurrence` | `lag_to_recurrence` | equivalent | |
| `segment.path_enhance` | – | not provided: deferred | Needs `filters.diagonal_filter` |
| `segment.timelag_filter` | – | not provided: out of scope | A Python decorator; compose `recurrence_to_lag`, a filter and `lag_to_recurrence` |
| `segment.agglomerative` | – | not provided: out of scope | Wraps scikit-learn clustering |
| `segment.subsegment` | – | not provided: out of scope | Needs `agglomerative` |
| `sequence.dtw` | `dtw`, `dtw_with_cost` (`DtwConfig`, `DistanceMetric`, `DtwResult`) | equivalent | Euclidean, squared Euclidean, cosine, cityblock, Chebyshev; custom steps and weights; subsequence; global band |
| `sequence.dtw_backtracking` | `dtw_backtracking` | equivalent | |
| `sequence.rqa` | `rqa` (`RqaConfig`, `RqaResult`) | equivalent with differences | Same scores; backtracking through the second column follows the move actually taken (librosa stores a wrong pointer there) |
| `sequence.viterbi` | `viterbi` | equivalent | |
| `sequence.viterbi_discriminative` | `viterbi_discriminative` | equivalent | |
| `sequence.viterbi_binary` | `viterbi_binary` | equivalent | |
| `sequence.transition_uniform` | `transition_uniform` | equivalent | |
| `sequence.transition_loop` | `transition_loop` | equivalent | |
| `sequence.transition_cycle` | `transition_cycle` | equivalent | |
| `sequence.transition_local` | `transition_local` (`TransitionWindow`) | equivalent | |
| `sequence.path_to_steps` | – | not provided: low value | Differences of consecutive path points |

### `librosa.display`

`librosa.display` is matplotlib plotting and is out of scope. The crate has
its own plotly-based plots on the `AudioPlotting` trait (feature `plotting`).

| librosa | audio_samples | status |
|---|---|---|
| `waveshow`, `AdaptiveWaveplot`, `wavebars`, `wavef0` | `AudioPlotting::plot_waveform` (closest) | not provided: out of scope (matplotlib) |
| `specshow` | `AudioPlotting::plot_spectrogram` (closest) | not provided: out of scope (matplotlib) |
| `multiplot`, `highlight`, `legend_for_axes`, `colorbar_db`, `colorbar_phase`, `infer_cmap` | – | not provided: out of scope (matplotlib) |
| `TimeFormatter`, `NoteFormatter`, `LogHzFormatter`, `ChromaFormatter`, `ChromaFJSFormatter`, `ChromaSvaraFormatter`, `FJSFormatter`, `TonnetzFormatter`, `Transformf0` | – | not provided: out of scope (matplotlib axis formatters) |

## Differences from librosa

These differences are the ones documented under `# Differences from librosa`
in the API docs, gathered here by theme.

### Shapes and types

- **The channel axis is always present.** librosa drops the leading axis for
  mono input; here frame-wise outputs are `(channels, n_features, n_frames)`,
  envelopes and pitch tracks `(channels, n_frames)`, and tempi `(channels, 1)`.
- **Outputs are `f64` / `Complex<f64>`.** librosa preserves `float32` and
  `complex64`, and several librosa bases default to single precision even for
  `float64` input. Here they are `f64`:
  - the mel basis (`filters.mel`, so `melspectrogram`, `mfcc`, `onset_strength`):
    about 1e-7 relative to librosa's default. Pass `dtype=np.float64` to librosa
    to agree to round-off;
  - the chroma basis (`filters.chroma`, `chroma_stft`): about 1e-7 relative;
  - CQT wavelets (`filters.wavelet`): librosa's `complex64` filters move its
    top-octave CQT by up to ~4e-8 of the peak;
  - `rms` and the RMS behind `trim` / `split`: librosa computes the mean square
    in `float32`;
  - `window_sumsquare`, `cq_to_chroma` and `clicks` return `f64`.
- **Event outputs are typed.** `onset_detect` and `beat_track` return
  `FrameEvents` (frame indices, convertible with `samples()` / `times()`) in place
  of librosa's `units=` argument. `trim_db` returns `(audio, (start, end))`, and
  `split_db` returns `(start, end)` sample pairs.
- **`operations::sequence` takes 2-D input only.** librosa broadcasts over
  leading axes; loop over channels yourself. Recurrence and cross-similarity
  matrices are dense `f64` (no `sparse=True`, connectivity is `0.0`/`1.0` rather
  than `bool`), and only the default `axis=-1` layout is supported.

### Resampling (no soxr)

- `librosa.resample` defaults to soxr (`res_type="soxr_hq"`). This crate has no
  soxr dependency. Its `ResamplingQuality::High` resampler is a delay-free
  Kaiser-windowed sinc designed after soxr_hq's passband (91.3 % of Nyquist)
  and attenuation. The two agree closely on band-limited content but are not
  bit-identical. Measured against `librosa.resample` from 22050 Hz to 11025,
  16000 and 44100 Hz: 129 to 135 dB SNR in the middle of a band-limited test
  signal, 59 to 104 dB at its abrupt start and end, and 50 to 58 dB on white
  noise (where the two transition bands differ). `Fast` upsampling is linear
  interpolation.
- **Integer output lengths.** The output length is exactly
  `ceil(n * target / orig)`, computed in integers. librosa computes it in
  floating point, which can be one sample longer (96001 instead of 96000 for
  88200 samples from 44100 Hz to 48000 Hz). Fractional sample rates and librosa's
  `scale` option are not supported.
- **Operations that resample internally inherit the difference:**
  - `pitch_shift`: the time-stretch stage matches librosa to round-off (except
    on exact digital silence, see `time_stretch`); the final signal agrees to
    about 68 dB SNR.
  - `cqt`, `vqt`, `hybrid_cqt`, `chroma_cqt`, `chroma_cens`, `chroma_vqt`,
    `tonnetz`: the top octave (no decimation) matches librosa to round-off.
    Decimated octaves agree to about 1e-4 of the peak, which is the same size as
    the difference between librosa's own `soxr_hq` and `soxr_vhq` (up to 4.7e-4).
  - `icqt`, `griffinlim_cqt`: the upsampling stages, as above.

### Errors where librosa warns or fails late

| Situation | librosa | audio_samples |
|---|---|---|
| `mfcc` with `n_mfcc > n_mels` | silently returns `n_mels` rows | error |
| `istft` whose bin count does not match `n_fft` | infers `n_fft` from the bins | error (`config.n_fft` is authoritative) |
| `istft(length=0)` | treated as `None` | error |
| `spectral_contrast` band with no FFT bin | `IndexError` or `NaN` with a warning | error |
| rank-deficient `poly_features` fit | `RankWarning`, minimum-norm solution | error |
| `onset_backtrack` with no events | raises | empty result |
| empty onset envelope, tempo too fast for one frame per beat | crashes or reads out of bounds | empty result or `ParameterError` |
| `onset_detect` / `beat_track` on multichannel audio | dense (`sparse=False`) broadcasting only | `LayoutError::ChannelCountUnsupported` (mono only) |
| `griffinlim` with `momentum > 1` | warns | accepted silently |
| `filters.mel` with empty filters | warns | zero rows, silently |
| `hz_to_note` outside MIDI 0..=127 | names any frequency (`"C-2"` for 4 Hz) | error |

### Randomness

- `griffinlim` with `GriffinLimInit::Random` (librosa's default `init="random"`)
  draws phases from this crate's RNG, not NumPy's, so results differ from
  librosa's for the same seed (both are valid reconstructions).
  `GriffinLimInit::ZeroPhase` is deterministic and matches `init=None` to
  round-off. `Random(Some(seed))` is reproducible.
- `griffinlim_cqt` offers only the deterministic `init=None` start.

### Options that are not supported

- `vqt`: custom `intervals` (just intonation). Only equal temperament is supported.
- `filters.cq_to_chroma`: the smoothing `window`.
- `filters.window_sumsquare`: `norm`.
- `filters.wavelet_lengths`: the public single-frequency-with-explicit-`alpha` form.
- `feature.metrogram`: `aggregate=None`.
- `phase_vocoder`: the deprecated `hop_length` / `n_fft` (ignored by librosa 1.0).
- `clicks`: multichannel custom clicks.
- `util.sync`: slice indices (frame indices only).
- `magphase`: a fractional `power`.

### Deliberate behavioural differences

- `filters.window_bandwidth`: librosa returns a hard-coded table value for named
  windows and ignores `n`. Here the formula is always evaluated (up to ~1e-4
  relative apart, for example Blackman 1.72697 vs 1.72676).
- `effects.hpss` / `harmonic` / `percussive`: librosa accepts a `window`
  argument but ignores it (always Hann). Here `config.stft.window` is used.
- `sequence.rqa`: librosa stores the wrong back-pointer for a `(-2, -1)` move
  into the second column, so its backtracking steps outside the matrix. The
  scores are identical; the path here follows the move actually taken.
- `util.sync` with an empty index list and `pad = true` returns one segment
  `[0, n_frames)`; librosa returns zero columns.
- `chroma_stft` with an odd `n_fft`: librosa's tuning estimate re-infers the FFT
  length as `2 * (bins - 1)`. Here the configured `n_fft` is used throughout.
- `recurrence_matrix`: exact brute-force neighbour search (librosa uses
  scikit-learn, whose distances can differ by round-off). librosa's
  connectivity-mode tie-breaking (lowest frame index wins) is reproduced exactly.

### Existing APIs that differ from librosa

The crate's older APIs keep their behaviour for compatibility. They are not
librosa-compatible. Use the librosa-compatible API listed in the table above
instead.

| Existing API | Difference from librosa | librosa-compatible API |
|---|---|---|
| `AudioTransforms::stft`, `istft`, `mel_spectrogram`, `chromagram`, `constant_q_transform` | Delegate to the `spectrograms` crate: symmetric windows, mono only, absolute dB floor, `MelNorm::None` by default, a different chroma filterbank and CQT kernel | `AudioSpectrum`, `AudioTonal`, `AudioConstantQ` |
| `AudioTransforms::mfcc` | Five independent differences: symmetric window, mel norm none, absolute -80 dB floor, unnormalised DCT-II, HTK-style lifter 22 by default | `AudioSpectrum::mfcc_frames` |
| `AudioStatistics::spectral_centroid`, `spectral_rolloff`, `spectral_bandwidth`, `spectral_flatness`, `spectral_contrast` | One value (or one vector) per signal from a single FFT of the whole signal; centroid and rolloff are power-weighted (librosa: magnitude); a different contrast band layout | `AudioFeatures::*_frames` |
| `AudioStatistics::zero_crossing_rate` | One whole-signal rate in Hz (librosa: per-frame fraction) | `AudioFeatures::zero_crossing_rate_frames` |
| `AudioStatistics::autocorrelation` | Unbiased estimate (multiply `r[k]` by `n - k` for librosa's raw sums); librosa autocorrelates every channel | – |
| `AudioOnsetDetection` (`detect_onsets`, `onset_detection_function`, `spectral_flux`, `complex_onset_detection`) | Different detection functions (CQT energy flux) and peak picker | `AudioRhythm::onset_strength`, `onset_detect` |
| `AudioBeatTracking` (`detect_beats`, `estimate_tempo`) | A beat grid at a user-supplied tempo and a global-autocorrelation tempo estimate, not dynamic programming | `AudioRhythm::beat_track`, `tempo` |
| `AudioDecomposition::hpss` | Its own mask blend (`mask_softness`) and window conventions | `AudioEffects::harmonic_percussive`, `operations::effects::hpss_spectrogram` |
| `AudioEditing::trim_silence` | Thresholds individual samples in absolute dBFS (librosa: frame RMS relative to the maximum) | `AudioEffects::trim_db` |
| `AudioEditing::split` | Fixed-length chunking | `AudioEffects::split_db` |
| `AudioPitchAnalysis::detect_pitch_yin`, `track_pitch`, `operations::pitch_analysis::yin_pitch_detection` | One estimate per buffer or uncentred frames; the difference function sums the `n - tau` overlapping products; no global-minimum fallback (returns `None`) | `AudioPitchTracking::yin`, `pyin` |
| `utils::audio_math::hz_to_mel`, `mel_to_hz`, `mel_scale`, `mel_frequencies` | HTK scale (librosa's default is Slaney) | `hz_to_mel_with_scale`, `mel_frequencies_with_scale` |
| `utils::audio_math::time_to_frames` | Rounds to the nearest frame and saturates negative times at 0 (librosa floors; about half of all times differ by one frame) | `time_to_frames_floor` |
| `utils::audio_math::seconds_to_samples` vs `utils::seconds_to_samples` | The first rounds, the second truncates (librosa's `time_to_samples` truncates) | `utils::audio_math::time_to_samples` |
| `utils::audio_math::frames_to_time` | No `n_fft` offset, no negative frames | `frames_to_time_with_offset` |
| `utils::audio_math::amplitude_to_db`, `power_to_db` (scalar) | No `ref` / `amin` / `top_db` clamping | `operations::spectrum::amplitude_to_db`, `power_to_db` |
| `AudioProcessing::mu_compress`, `mu_expand` | Equal `quantize=False` (librosa's default is `quantize=True`) | `mu_law_encode`, `mu_law_decode` |
| `AudioSampleIterators::windows` | Zero-pads trailing windows (ceil count) | `operations::spectrum::frame`, `windows_ref` |
| Integer to float conversion (`as_f64`, `to_format`) | Asymmetric: positive values divided by `MAX`, negative by `-MIN` (librosa's `buf_to_float` divides both by `2^(bits-1)`; up to 1 LSB apart); `u8` is unsigned PCM (librosa `n_bytes=1` reads signed `int8`) | – |

## Verification

The librosa-compatible API is checked against golden fixtures generated by
librosa itself:

- **Generator:** `tests/reference/librosa/generate_librosa_fixtures.py`, a `uv`
  project that pins `librosa==1.0.0`, `numpy==2.5.3`, `scipy==1.18.1` and
  `soxr==1.1.0`. It builds deterministic test signals (`signals.json`, stored as
  16-bit PCM codes so they are exact in every sample type) and runs one case
  module per area (`cases/<area>.py`).
- **Fixtures:** `tests/fixtures/librosa/<area>.json`, one per area: `spectrum`,
  `features`, `tonal`, `constant_q`, `rhythm`, `effects`, `pitch`, `sequence`,
  `resample`, `analysis_fixes`.
- **Tests:** `tests/librosa_reference/<area>.rs`. Every tolerance has a comment
  that says why it is needed.

Run the suite (add `-- --nocapture` to print the measured error of every case):

```sh
cargo test --release --features full_no_plotting --test librosa_reference
```

Regenerate the fixtures, or check that the committed ones match the generator
(the first run installs the pinned Python environment):

```sh
uv run --project tests/reference/librosa tests/reference/librosa/generate_librosa_fixtures.py
uv run --project tests/reference/librosa tests/reference/librosa/generate_librosa_fixtures.py --check
```

The `pre-commit` hooks `librosa-reference-tests` and `librosa-fixtures-fresh`
(scripts in `tests/reference/hooks/`) run these automatically when the
librosa-compatible code, the generator or the fixtures change.

### What the tolerances mean

- **Array outputs are compared peak-relative:** `|actual - expected| <= rtol *
  max|expected|` over the compared array. `rtol` is set by the error source
  named in the test:
  - **About 1e-15 to 1e-12** where the algorithm is a step-by-step port. The
    only differences are FFT butterfly order (pocketfft vs realfft, about 1e-16
    of the peak per transform), matrix-product summation order and libm ulps.
    This covers windows, filterbanks, STFT/ISTFT, mel, MFCC, dB, delta,
    spectral features, PCEN, chroma, pitch, onset envelopes and tempograms.
  - **About 1e-9** where such errors are fed back through iterations or long
    phase accumulations (Griffin-Lim, phase vocoder, time stretch).
  - **Looser, justified bounds** for different algorithms: CQT octaves below the
    top (resampler difference, about 1e-4 of the peak, bounds 5e-4 to 1e-2
    depending on the case) and `float32`-basis comparisons (1e-6).
- **Discrete outputs are compared exactly:** onset, peak and beat frames, tempi,
  DTW and RQA paths, Viterbi state sequences, trim and split boundaries,
  mu-law codes and tuning estimates.
- **Resampling is compared by SNR** against soxr_hq: at least 125 dB in the
  middle of the band-limited signal, 55 dB at its edges and 47 dB on white
  noise. There is an independent cross-check against
  `scipy.signal.resample_poly` (at least 62 dB).

### Measured agreement

Worst case per area, from `cargo test --release --features full_no_plotting
--test librosa_reference -- --nocapture` (peak-relative unless stated):

| Area | Worst measured | Notes |
|---|---|---|
| Windows, mel / chroma filterbanks, unit conversions, `normalize_array` | ≤ 6e-16 | Most cases bit-identical |
| STFT / ISTFT / dB / melspectrogram / MFCC / delta | ≤ 2e-12 | MFCC 2.7e-15 (default). melspectrogram against librosa's default `float32` mel basis: 3.3e-8 |
| Spectral features, PCEN, weightings, reassigned spectrogram | ≤ 5e-11 | Contrast (log of band ratios) is the largest |
| chroma_stft, piptrack | ≤ 1.5e-15 | Tuning estimates identical |
| yin / pyin / salience / harmonics | f0 ≤ 3.3e-16 | pYIN f0 and voiced flags bit-identical; voicing probability ≤ 3e-15 absolute |
| Onset strength, tempograms, PLP | ≤ 3.1e-14 | Onset, peak and beat frames and tempi identical. `hybrid_tempogram`: 3.9e-9 (square root of round-off where the tempogram is zero) |
| Effects (HPSS, softmask, emphasis, remix, LPC, nn_filter) | ≤ 6e-13 | trim / split boundaries identical |
| Phase vocoder, time stretch, Griffin-Lim (zero phase) | ≤ 7e-11 | |
| pitch_shift | 67.6 to 68.5 dB SNR | Resampler difference |
| CQT top octave / pseudo-CQT | 1.7e-16 (with `complex128` filters) / 3e-8 | Lower octaves: up to 1.3e-4 of the peak; chroma_cqt ≤ 1e-4 |
| DTW, Viterbi, RQA, recurrence, sync | ≤ 3.5e-16 | Paths and state sequences identical |
| resample (`High`) | 129 to 135 dB SNR (band-limited, middle) | 50 to 58 dB on white noise |
