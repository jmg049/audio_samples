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

### Creating audio

Build `AudioSamples` from an `ndarray`, or generate a signal. Construction is
fallible because the data, channel count, and sample rate must agree.

```rust
use audio_samples::{AudioSamples, sample_rate, sine_wave};
use ndarray::array;
use std::time::Duration;

fn main() -> audio_samples::AudioSampleResult<()> {
    // Mono, from a slice of samples.
    let _mono = AudioSamples::new_mono(array![0.1f32, 0.5, -0.3, 0.8], sample_rate!(44_100))?;

    // Stereo, from a 2-D (channels x samples) array.
    let _stereo = AudioSamples::<f32>::new_multi_channel(
        array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        sample_rate!(44_100),
    )?;

    // Or generate one: a 440 Hz sine, 1 second, amplitude 0.5.
    let _tone = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(44_100), 0.5);
    Ok(())
}
```

Generators cover the classic oscillators plus logarithmic chirps
(`exponential_chirp`), band-limited square/sawtooth/triangle waves, FM
(`fm_signal`), and white/pink/brown noise.

### A processing pipeline

Operations carry the sample rate and channel layout with the data and chain
through the borrowing API. Here two tones are mixed, the low one is filtered
out, the result is peak-normalized, and its level is read back. Enable the
`processing` feature.

```rust
use audio_samples::{AudioProcessing, AudioStatistics, NormalizationConfig, sample_rate, sine_wave};
use std::time::Duration;

fn main() -> audio_samples::AudioSampleResult<()> {
    let sr = sample_rate!(44_100);
    let dur = Duration::from_secs(1);

    // 100 Hz and 5 kHz, mixed.
    let signal = sine_wave::<f32>(100.0, dur, sr, 0.5) + sine_wave::<f32>(5_000.0, dur, sr, 0.5);

    // High-pass above 1 kHz, then peak-normalize. Each step returns a new value.
    let cleaned = signal
        .high_pass_filter(1_000.0)?
        .normalize(NormalizationConfig::peak(1.0))?;

    println!(
        "{} channel(s) at {} Hz, peak {:.3}, rms {:.3}",
        cleaned.num_channels().get(),
        cleaned.sample_rate().get(),
        cleaned.peak(),
        cleaned.rms(),
    );

    // The `*_in_place` variants mutate instead of returning a new value.
    let mut buf = cleaned;
    buf.scale_in_place(0.5);
    Ok(())
}
```

### Type conversions

Conversions are audio-aware: integer PCM is scaled to and from the float
`[-1.0, 1.0]` range rather than cast blindly.

```rust
use audio_samples::{AudioTypeConversion, sample_rate, sine_wave};
use std::time::Duration;

let pcm = sine_wave::<i16>(440.0, Duration::from_secs(1), sample_rate!(44_100), 0.8); // i16 PCM
let as_float = pcm.as_f32();   // scaled into [-1.0, 1.0]
let _back = as_float.as_i16(); // and back to i16
```

### Spectral analysis

Enable the `transforms` feature for FFT/STFT and the spectral feature suite.

```rust
use audio_samples::{AudioStatistics, AudioTransforms, ChannelReduction, nzu, sample_rate, sine_wave};
use spectrograms::{StftParams, WindowType};
use std::time::Duration;

fn main() -> audio_samples::AudioSampleResult<()> {
    let audio = sine_wave::<f32>(440.0, Duration::from_millis(200), sample_rate!(44_100), 0.8);

    // Short-time Fourier transform.
    let params = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true)?;
    let _stft = audio.stft(&params)?;

    // A derived spectral feature (brightness), in Hz.
    let centroid = audio.spectral_centroid(ChannelReduction::Average)?;
    println!("spectral centroid: {centroid:.0} Hz");
    Ok(())
}
```

The full transform set (FFT, MFCC, chromagram, CQT, PSD, inverse STFT) and the
rest of the spectral features are listed under [Features](#features).

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
- [`audio_samples_streaming](https://github.com/jmg049/audio_samples_streaming): streaming functionality
- [`audio_samples_ml](https://github.com/jmg049/audio_samples_ml): audio machine learning such as STT and TTS.
- [`audio_samples_qoe](https://github.com/jmg049/audio_samples_streaming): audio Quality of Experience (qoe) metrics.
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
