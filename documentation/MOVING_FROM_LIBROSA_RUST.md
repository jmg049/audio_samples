# Moving from librosa: the Rust side

Companion to the guide
[Moving from librosa](https://github.com/jmg049/audio_samples_python/blob/main/docs/source/moving_from_librosa.rst),
which has the measurements, the pipeline in Python, the differences that catch people out and
the list of what is missing. This page is for porting a librosa pipeline to Rust. Every code
block here is a doctest (`cargo test --doc --features full_no_plotting`) except file reading,
which needs a separate crate and is a runnable example in the Python bindings' repository.
[LIBROSA_MIGRATION.md](LIBROSA_MIGRATION.md) is the function-by-function lookup.

In the guide's tables, "native" is this path: the Rust library called directly, with no Python
in the process. It is what the benchmark's `examples/librosa_bench` measures.

## The pipeline

`X::librosa()` is a config with librosa 1.0's keyword defaults. Use it when you need to
reproduce a librosa result. `X::default()` is this crate's default.

```rust
use audio_samples::operations::spectrum::power_to_db;
use audio_samples::{
    AudioRhythm, AudioSamples, AudioSpectrum, BeatTrackConfig, DbConfig, DbReference,
    MelSpectrogramConfig, MfccConfig, OnsetDetectConfig, StftConfig, sample_rate,
};
use ndarray::Array1;

// A click every 0.5 s (120 BPM) for 8 s. Replaces `y, sr = librosa.load(path)`.
let sr = 22050;
let mut y = Array1::<f32>::zeros(8 * sr);
for k in 0..16 {
    for n in 0..200 {
        y[k * sr / 2 + n] = (-(n as f32) / 30.0).exp();
    }
}
let audio = AudioSamples::new_mono(y, sample_rate!(22050)).unwrap();

// D = librosa.stft(y): the channel axis is always there.
let d = audio.stft_matrix(&StftConfig::librosa()).unwrap();
assert_eq!(d.dim().0, 1);
assert_eq!(d.dim().1, 1025);

// S = librosa.feature.melspectrogram(y=y, sr=sr)
let s = audio.melspectrogram(&MelSpectrogramConfig::librosa()).unwrap();
assert_eq!(s.dim().1, 128);

// S_db = librosa.power_to_db(S, ref=np.max)
let db = DbConfig { reference: DbReference::Max, ..DbConfig::librosa_power() };
let s_db = power_to_db(&s, &db).unwrap();
assert!(s_db.iter().all(|&v| v >= -80.0 - 1e-3));

// mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
let cfg = MfccConfig { n_mfcc: audio_samples::nzu!(13), ..MfccConfig::librosa() };
let mfcc = audio.mfcc_frames(&cfg).unwrap();
assert_eq!(mfcc.dim().1, 13);

// librosa.onset.onset_detect(y=y, sr=sr, units="time")
let onsets = audio.onset_detect(&OnsetDetectConfig::librosa()).unwrap().times();
assert!(onsets.len() >= 15);

// tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
let track = audio.beat_track(&BeatTrackConfig::librosa()).unwrap();
assert!((track.tempo - 120.0).abs() < 5.0);
assert!(!track.beats.times().is_empty());
```

## Defaults and presets

`Default` differs from `librosa()` only where librosa's default changes the data without being
asked, or where no default suits every input. The full table is in
[LIBROSA_MIGRATION.md](LIBROSA_MIGRATION.md#defaults-that-differ-from-librosa).

```rust
use audio_samples::{BeatTrackConfig, ChromaStftConfig, DbConfig, MfccConfig, PcenConfig, SilenceConfig};
use audio_samples::nzu;

// librosa's 80 dB floor is opt-in.
assert_eq!(DbConfig::power().top_db, None);
assert_eq!(DbConfig::librosa_power().top_db, Some(80.0));
assert_ne!(MfccConfig::default().db.top_db, MfccConfig::librosa().db.top_db);

// So is estimating the tuning from the signal, and trimming weak beats.
assert_eq!(ChromaStftConfig::default().tuning, Some(0.0));
assert_eq!(ChromaStftConfig::librosa().tuning, None);
assert!(!BeatTrackConfig::default().trim);
assert!(BeatTrackConfig::librosa().trim);

// Where no default suits every input the value is required, and `librosa()` is librosa's.
assert_eq!(SilenceConfig::librosa().top_db, 60.0);
assert_eq!(PcenConfig::librosa().hop_length, nzu!(512));
```

## Precision follows the sample type

Analysis runs in `T::Float`: `f32` for `u8`, `i16`, `I24` and `f32` audio, `f64` for `i32` and
`f64` audio. The type system shows the result, and audio-returning operations keep `T`.

```rust
use audio_samples::{AudioEffects, AudioSamples, AudioSpectrum, MfccConfig, StftConfig, sample_rate};
use ndarray::{Array1, Array3};

let y = Array1::<i16>::from_shape_fn(22050, |i| ((i as f32 * 0.05).sin() * 8000.0) as i16);
let audio = AudioSamples::new_mono(y, sample_rate!(22050)).unwrap();

// i16 audio: f32 features, computed in f32.
let mfcc: Array3<f32> = audio.mfcc_frames(&MfccConfig::librosa()).unwrap();
assert_eq!(mfcc.dim().0, 1);

// An effect returns i16 audio again.
let stretched: AudioSamples<'static, i16> = audio.time_stretch(1.25, &StftConfig::librosa()).unwrap();
assert_eq!(stretched.samples_per_channel().get(), 17640);
```

## Results that are bool

`recurrence_matrix` and `cross_similarity` in connectivity mode return `Array2<bool>`, like
librosa's dense result. Distances and affinities come from the `_weighted` functions.
`viterbi_binary` returns `bool` states where librosa returns `uint16`.

```rust
use audio_samples::operations::sequence::{RecurrenceConfig, recurrence_matrix};
use ndarray::Array2;

let features = Array2::<f32>::from_shape_fn((4, 30), |(i, j)| ((i * 7 + j * 13) % 11) as f32);
let links: Array2<bool> = recurrence_matrix(features.view(), &RecurrenceConfig::default()).unwrap();
assert_eq!(links.dim(), (30, 30));
```

## Reading files

File reading is in the separate crate [`audio_samples_io`](https://crates.io/crates/audio_samples_io),
which this crate does not depend on. It reads and writes WAV and FLAC. Unlike `librosa.load`,
`audio_samples_io::read::<_, T>` returns the file's rate and channels; you choose `T`. The mix-down
and resampling that `librosa.load` does by default are explicit:

```rust
use audio_samples::operations::types::{MonoConversionMethod, ResamplingQuality};
use audio_samples::{AudioChannelOps, AudioProcessing, AudioSamples, sample_rate, sine_wave};
use std::time::Duration;

// let audio: AudioSamples<'static, f32> = audio_samples_io::read("file.wav")?;
let audio: AudioSamples<'static, f32> =
    sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(44100), 0.5);

// y, sr = librosa.load("file.wav")
let mono = audio.to_mono(MonoConversionMethod::Average).unwrap();
let y = mono.resample(sample_rate!(22050), ResamplingQuality::High).unwrap();
assert_eq!(y.sample_rate().get(), 22050);
assert_eq!(y.samples_per_channel().get(), 22050);
```

`examples/rust_pipeline.rs` in the
[Python bindings' repository](https://github.com/jmg049/audio_samples_python) does this on a real
file (`cargo run --release --example rust_pipeline -- file.wav`).

## Allocation-free forms

Some functions have forms that write into your buffer or return static data, which is what the
benchmark's memory rows use where librosa's result comes from NumPy's small-block cache.

```rust
use audio_samples::operations::constant_q::{WaveletConfig, wavelet_lengths_into};
use audio_samples::operations::spectrum::frame_view;
use audio_samples::utils::audio_math::{cqt_frequencies_into, midi_to_note_static};
use audio_samples::nzu;

let mut freqs = vec![0.0f64; 84];
cqt_frequencies_into(&mut freqs, 32.703195662574764, nzu!(12), 0.0);
assert!((freqs[12] - 2.0 * freqs[0]).abs() < 1e-9);

let mut lengths = vec![0.0f64; 84];
wavelet_lengths_into(&freqs, 22050.0, &WaveletConfig::librosa(), &mut lengths).unwrap();
assert!(lengths[0] > lengths[83]);

assert_eq!(midi_to_note_static(69).unwrap(), "A4");

// librosa.util.frame(x, frame_length=4, hop_length=2): a view, no copy.
let x: Vec<f32> = (0..10).map(|i| i as f32).collect();
let frames = frame_view(&x, nzu!(4), nzu!(2)).unwrap();
assert_eq!(frames.dim(), (4, 4));
```

## Checking a port

Generate reference output from your librosa pipeline with `numpy.save`, read it in a test and compare
peak-relative, the way `tests/analysis_reference/` does. Use `f64` audio and librosa's
`dtype=np.float64` to compare algorithms rather than `f32` round-off. Expect about 1e-15 to 1e-12 for
step-by-step ports, about 1e-9 where error feeds back through iterations (phase vocoder, Griffin-Lim),
and about 1e-4 of the peak where soxr is involved (`pitch_shift`, CQT octaves below the top). The
guide lists what deliberately does not agree.

## What is not here

Everything listed under "When to stay on librosa" in the guide also applies in Rust. In addition, the
crate has no `librosa.display`: `AudioPlotting` (feature `plotting`) has its own plotly plots.
Arrays are `ndarray`, shaped `(channels, n_features, n_frames)`.
