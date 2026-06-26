<div align="center">

# AudioSamples

## Typed, invariant-preserving audio processing for Rust

<img src="logo.png" title="AudioSamples Logo -- Ferrous' Mustachioed Cousin From East Berlin, Eisenhaltig" width="200"/>

[![Crates.io][crate-img]][crate] [![Docs.rs][docs-img]][docs] [![License: MIT][license-img]][license]

</div>

---

## Overview

Most audio libraries hand you a bare buffer. In Python it is a NumPy array
whose `dtype` is explicit but whose meaning is not: the sample rate, amplitude
range, channel layout, and interleaving live elsewhere, if they are tracked at
all. Rust libraries give you fast, safe primitives but still leave you holding
the raw buffer, writing your own conversions, and carrying those conventions in
your head. Mismatches between them, such as a stale sample rate after
resampling or a buffer that was interleaved when the next stage expected planar,
are a recurring source of bugs.

`audio_samples` puts that metadata in the type. [`AudioSamples<T>`] pairs the
PCM data with its sample rate, channel count, and memory layout, and every
operation either preserves those invariants or updates them explicitly. The
sample type `T` (`u8`, `i16`, [`I24`], `i32`, `f32`, `f64`) is part of the type,
so format and bit-depth conversions are checked and scaled correctly rather than
left to ad hoc casts.

### API conventions

`AudioSamples` keeps its fields private; reach the buffer and metadata through
`data()`, `data_mut()`, `into_data()`, and `sample_rate()`. Every transforming
operation has two forms:

- `op(&self, ..) -> AudioSampleResult<Self>` borrows and returns a new, modified
  value. This is the unsuffixed, canonical name, and these forms chain into
  pipelines without consuming the binding. Infallible operations such as `scale`
  return `Self` directly.
- `op_in_place(&mut self, ..) -> AudioSampleResult<()>` mutates the receiver.

Analysis operations that do not modify the audio (statistics, transforms, pitch,
and so on) take `&self` and return their result.

---

## Installation

```bash
cargo add audio_samples
```

The default feature set (`bare-bones`) is the core types and traits with no
optional dependencies. Add features for the operations you need; see
[Features](#features).

Upgrading from 1.x? See [the migration guide](documentation/MIGRATING_TO_2.0.md).

---

## Quick Start

### Generating and mixing signals

```rust
use audio_samples::{sample_rate, AudioTypeConversion, cosine_wave, sine_wave};
use std::time::Duration;

fn main() {
    let sr = sample_rate!(44_100);
    let duration = Duration::from_secs_f64(1.0);

    // A 440 Hz sine as i16 PCM, converted to f32.
    let float_sine = sine_wave::<i16>(440.0, duration, sr, 0.5).as_f32();

    // Mixed with a 220 Hz cosine.
    let cosine = cosine_wave::<f32>(220.0, duration, sr, 0.5);
    let mixed = float_sine + cosine;
}
```

Besides the classic oscillators, the generators include logarithmic chirps
(`exponential_chirp`), band-limited (alias-free) square, sawtooth, and triangle
waves (`square_wave_bandlimited`, `sawtooth_wave_bandlimited`,
`triangle_wave_bandlimited`), and frequency modulation (`fm_signal`).

### Processing pipelines

Enable the `processing` feature. Each borrowing operation returns a new owned
value, so they chain; use the `*_in_place` variants to mutate instead.

```rust
use audio_samples::{AudioProcessing, AudioSamples, NormalizationConfig, sample_rate};
use ndarray::array;

fn main() -> audio_samples::AudioSampleResult<()> {
    let audio = AudioSamples::new_mono(
        array![0.1f32, 0.5, -0.3, 0.8, -0.2],
        sample_rate!(44100),
    )?;

    // `scale` is infallible (no `?`); `normalize` and `remove_dc_offset` are fallible.
    let processed = audio
        .normalize(NormalizationConfig::peak(1.0))?
        .scale(0.5)
        .remove_dc_offset()?;

    // The same work, mutating in place:
    let mut buf = processed;
    buf.scale_in_place(2.0);
    Ok(())
}
```

### Spectral transforms

Enable the `transforms` feature:

```bash
cargo add audio_samples --features transforms
```

```rust
use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
use spectrograms::{ChromaParams, CqtParams, MfccParams, StftParams, WindowType};
use std::time::Duration;

fn main() -> audio_samples::AudioSampleResult<()> {
    let sr = sample_rate!(44100);
    let audio: AudioSamples<'static, f64> =
        sine_wave::<f64>(440.0, Duration::from_millis(200), sr, 0.8);

    let fft = audio.fft(nzu!(8192))?;

    let stft_params = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true)?;
    let stft = audio.stft(&stft_params)?;
    let mfcc = audio.mfcc(&stft_params, nzu!(40), &MfccParams::speech_standard())?;
    let chroma = audio.chromagram(&stft_params, &ChromaParams::music_standard())?;

    // power_spectral_density returns a `Psd` with frequency and density axes.
    let psd = audio.power_spectral_density(nzu!(1024), 0.5)?;
    let (_freqs, _density) = (psd.frequencies(), psd.density());

    let _cqt = audio.constant_q_transform(
        &CqtParams::new(nzu!(12), nzu!(7), 32.7)?,
        nzu!(256),
    )?;

    // Round-trip via the inverse STFT.
    let _reconstructed = AudioSamples::<f64>::istft(stft)?;
    Ok(())
}
```

### Psychoacoustic analysis

Enable the `psychoacoustic` feature:

```bash
cargo add audio_samples --features psychoacoustic
```

```rust
use audio_samples::{
    AudioPerceptualAnalysis, BandLayout, PsychoacousticConfig, sample_rate, sine_wave,
};
use non_empty_slice::NonEmptySlice;
use spectrograms::WindowType;
use std::num::NonZeroUsize;
use std::time::Duration;

fn main() -> audio_samples::AudioSampleResult<()> {
    let signal = sine_wave::<f32>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.8);

    // 24 Bark critical bands mapped onto 1024 MDCT bins.
    let layout = BandLayout::bark(
        NonZeroUsize::new(24).unwrap(),
        44100.0,
        NonZeroUsize::new(1024).unwrap(),
    );

    let weights = vec![1.0_f32; 24];
    let config = PsychoacousticConfig::new(
        -60.0, 14.5, 0.4, 25.0, 6.0,
        NonEmptySlice::from_slice(&weights).unwrap(),
        1e-10,
    );

    let result = signal.analyse_psychoacoustic(WindowType::Hanning, &layout, &config)?;

    // Bands audible above the masking threshold.
    for metric in result.band_metrics.as_slice().iter() {
        if metric.signal_to_mask_ratio > 0.0 {
            println!(
                "{:.0} Hz, SMR {:.1} dB, importance {:.2}",
                metric.band.centre_frequency,
                metric.signal_to_mask_ratio,
                metric.importance,
            );
        }
    }
    Ok(())
}
```

---

## Creating AudioSamples

Construction returns a `Result`: a valid `AudioSamples` requires a consistent
buffer length, channel count, and sample rate. The `sample_rate!` macro and
`non_empty_vec!` enforce the non-zero and non-empty invariants at the call site.

```rust
use audio_samples::{AudioSamples, sample_rate};
use non_empty_slice::non_empty_vec;

let audio = AudioSamples::from_mono_vec(
    non_empty_vec![0.1f32, 0.2, 0.3],
    sample_rate!(44100),
);
```

For multi-channel audio (channels x samples):

```rust
use audio_samples::{AudioSamples, sample_rate};
use ndarray::array;

let stereo = AudioSamples::<f32>::new_multi_channel(
    array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    sample_rate!(44100),
).unwrap();
```

---

## <a name="features">Features</a>

The default feature is `bare-bones`: the core types and traits, no optional
dependencies. Enable the rest as needed.

### Core operations

| Feature         | Description                                                                           |
| --------------- | ------------------------------------------------------------------------------------- |
| `statistics`    | Peak, RMS, mean, variance, zero-crossings; the spectral feature suite (centroid, rolloff, bandwidth, flatness, contrast, slope, crest) when `transforms` is also enabled |
| `processing`    | Normalization, scaling, clipping, DC-offset removal (requires `statistics`)           |
| `editing`       | Trim, pad, reverse, fade, concatenate, perturb (requires `statistics`, `random-generation`) |
| `channels`      | Interleave/deinterleave, mono/stereo conversion, channel extraction                   |
| `iir-filtering` | Butterworth, Chebyshev I and II, Elliptic (Cauer), and Bessel filters; low-, high-, band-pass and band-stop responses; zero-phase `filtfilt`; design-once streaming `SosFilter` |
| `parametric-eq` | Parametric EQ bands and `ThreeBandEqConfig` (requires `iir-filtering`)                 |
| `dynamic-range` | Compression, limiting, gating, expansion via config structs, with side-chain support  |
| `envelopes`     | Amplitude, RMS, and attack-decay envelope followers                                   |
| `vad`           | Voice activity detection                                                              |

### Spectral and analysis

| Feature           | Description                                                                 |
| ----------------- | --------------------------------------------------------------------------- |
| `transforms`      | FFT, STFT and inverse STFT, MFCC, chromagram, CQT, power spectral density   |
| `psychoacoustic`  | Bark/Mel band layouts, absolute threshold of hearing, masking thresholds, SMR (requires `transforms`) |
| `pitch-analysis`  | YIN and autocorrelation pitch detection and tracking (requires `transforms`) |
| `onset-detection` | Onset detection (requires `transforms`, `peak-picking`, `processing`)        |
| `beat-tracking`   | Beat tracking and tempo estimation (`estimate_tempo`)                        |
| `peak-picking`    | Peak picking on onset-strength envelopes                                     |
| `decomposition`   | Harmonic/percussive source separation (requires `onset-detection`)           |

### Utility

| Feature             | Description                                                          |
| ------------------- | -------------------------------------------------------------------- |
| `resampling`        | Sample-rate conversion via rubato                                    |
| `random-generation` | White, pink, and brown noise generators                              |
| `fixed-size-audio`  | Stack-allocated fixed-size buffers                                   |
| `plotting`          | Interactive HTML plots (waveform, spectrum, phase, spectrogram, Lissajous) via plotly |
| `static-plots`      | PNG/SVG export (requires `plotting`; see [PLOTTING.md](PLOTTING.md)) |
| `simd`              | SIMD-accelerated sample conversions via the `wide` crate (stable; results are bit-identical to the scalar path) |

### Bundles

| Feature            | Description                  |
| ------------------ | ---------------------------- |
| `full`             | Everything below             |
| `full_no_plotting` | Everything except plotting   |

---

## Documentation

Full API documentation: [https://docs.rs/audio_samples](https://docs.rs/audio_samples).
Architecture notes are in [documentation/ARCHITECTURE.md](documentation/ARCHITECTURE.md).

---

## Examples

The repository includes runnable examples in `examples/`, each annotated with
its required feature flags. Run one with, for example:

```bash
cargo run --example transforms --features transforms,statistics
```

A larger demo lives in a separate repository:

- [DTMF encoder and decoder](https://github.com/jmg049/dtmf-demo)

---

## Companion Crates

- [`audio_samples_io`](https://github.com/jmg049/audio_samples_io): audio file decoding and encoding
- [`audio_samples_playback`](https://github.com/jmg049/audio_playback): device-level playback
- [`audio_samples_python`](https://github.com/jmg049/audio_python): Python bindings
- [`spectrograms`](https://github.com/jmg049/Spectrograms): spectrograms and time-frequency transforms (used by the `transforms` feature)
- [`i24`](https://github.com/jmg049/i24): 24-bit signed integer type for Rust
- [`dtmf_tones`](https://github.com/jmg049/dtmf_tones): `no_std` DTMF keypad frequencies

---

## License

MIT.

---

## Citing

If you use AudioSamples in research, please cite:

```bibtex
@inproceedings{geraghty2026audio,
  author    = {Geraghty, Jack and Golpayegani, Fatemeh and Hines, Andrew},
  title     = {Audio Made Simple: A Modern Framework for Audio Processing},
  booktitle = {ACM Multimedia Systems Conference 2026 (MMSys '26)},
  year      = {2026},
  month     = apr,
  publisher = {ACM},
  address   = {Hong Kong, Hong Kong},
  doi       = {10.1145/3793853.3799811},
  note      = {Accepted for publication}
}
```

---

## Contributing

Contributions are welcome. Please open an issue or pull request, and see
[CONTRIBUTING.md](CONTRIBUTING.md) for guidance.

[crate]: https://crates.io/crates/audio_samples
[crate-img]: https://img.shields.io/crates/v/audio_samples?style=for-the-badge&color=009E73&label=crates.io
[docs]: https://docs.rs/audio_samples
[docs-img]: https://img.shields.io/badge/docs.rs-online-009E73?style=for-the-badge&labelColor=gray
[license-img]: https://img.shields.io/crates/l/audio_samples?style=for-the-badge&label=license&labelColor=gray
[license]: https://github.com/jmg049/audio_samples/blob/main/LICENSE
