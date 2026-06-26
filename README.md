<div align="center">

# AudioSamples

## Fast, simple, and expressive audio in Rust

<img src="logo.png" title="AudioSamples Logo -- Ferrous' Mustachioed Cousin From East Berlin, Eisenhaltig" width="200"/>

[![Crates.io][crate-img]][crate] [![Docs.rs][docs-img]][docs] [![License: MIT][license-img]][license]

</div>

---

## Overview

Most audio libraries expose samples as raw numeric buffers. In Python,
audio is typically represented as a NumPy array whose `dtype` is
explicit, but whose meaning is not: sample rate, amplitude range,
memory interleaving, and PCM versus floating-point semantics are tracked
externally, if at all. In Rust, the situation is reversed but not
resolved. Libraries provide fast and safe low-level primitives, yet
users are still responsible for managing raw buffers, writing ad hoc
conversion code, and manually preserving invariants across crates.

AudioSamples closes this gap with a strongly typed audio representation
that encodes sample format, numeric domain, channel structure, and
layout in the type system. All operations preserve or explicitly update
these invariants, supporting both exploratory workflows and system-level
use without requiring users to remember hidden conventions or reimplement
common audio logic.

The `AudioSamples` type keeps its fields private; access the underlying
buffer and metadata through accessors (`data()`, `data_mut()`,
`into_data()`, `into_data_borrowed()`, `sample_rate()`). Every transforming
operation comes in two flavours: the canonical, unsuffixed name borrows and
returns a **new** modified copy (`op(&self, ..) -> AudioSampleResult<Self>`),
while the `op_in_place` counterpart mutates the receiver
(`op_in_place(&mut self, ..) -> AudioSampleResult<()>`). The borrowing
variants compose into pipelines without consuming the binding; infallible
operations such as `scale` return `Self` directly.

---

## Installation

```bash
cargo add audio_samples
```

The default feature set (`bare-bones`) includes only the core types and
traits. Add features for the operations you need — see [Features](#features).

---

## Quick Start

### Generating and mixing signals

```rust
use audio_samples::{sample_rate, AudioTypeConversion, cosine_wave, sine_wave};
use std::time::Duration;

fn main() {
    let sr = sample_rate!(44_100);
    let duration = Duration::from_secs_f64(1.0);

    // Generate a 440 Hz sine wave as i16 PCM, then convert to f32
    let float_sine = sine_wave::<i16>(440.0, duration, sr, 0.5).as_f32();

    // Mix with a 220 Hz cosine wave
    let cosine = cosine_wave::<f32>(220.0, duration, sr, 0.5);
    let mixed = float_sine + cosine;
}
```

Alongside the classic oscillators, the generators include exponential/log
chirps (`exponential_chirp`), band-limited square/sawtooth/triangle waves
(`square_wave_bandlimited`, `sawtooth_wave_bandlimited`,
`triangle_wave_bandlimited`), and FM signals (`fm_signal`).

### Processing pipelines

Enable the `processing` feature. Each borrowing operation returns a new owned
value, so they chain directly; use the `*_in_place` variants to mutate instead.

```rust
use audio_samples::{AudioProcessing, AudioSamples, NormalizationConfig, sample_rate};
use ndarray::array;

fn main() -> audio_samples::AudioSampleResult<()> {
    let audio = AudioSamples::new_mono(
        array![0.1f32, 0.5, -0.3, 0.8, -0.2],
        sample_rate!(44100),
    )?;

    // `scale` is infallible (no `?`); `normalize`/`remove_dc_offset` are fallible.
    let processed = audio
        .normalize(NormalizationConfig::peak(1.0))?
        .scale(0.5)
        .remove_dc_offset()?;

    // Or mutate in place:
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
    let (_freqs, _psd) = audio.power_spectral_density(nzu!(1024), 0.5)?;
    let _cqt = audio.constant_q_transform(
        &CqtParams::new(nzu!(12), nzu!(7), 32.7)?,
        nzu!(256),
    )?;

    // Round-trip via inverse STFT
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

    // Print bands audible above the masking threshold.
    for metric in result.band_metrics.as_slice().iter() {
        if metric.signal_to_mask_ratio > 0.0 {
            println!(
                "{:.0} Hz — SMR {:.1} dB, importance {:.2}",
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

AudioSamples creation returns a `Result` because validity requires
consistent buffer length, channel count, and sample rate. The
`sample_rate!` macro and `non_empty_vec!` guarantee invariants at
construction:

```rust
use audio_samples::{AudioSamples, sample_rate};
use non_empty_slice::non_empty_vec;

let audio = AudioSamples::from_mono_vec(
    non_empty_vec![0.1f32, 0.2, 0.3],
    sample_rate!(44100),
);
```

For multi-channel audio:

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

The default feature is `bare-bones` — the core types and traits with no
optional dependencies. Enable features as needed:

### Core operations

| Feature         | Description                                                                           |
| --------------- | ------------------------------------------------------------------------------------- |
| `statistics`    | Descriptive statistics: peak, RMS, mean, variance; spectral feature suite (centroid, rolloff, bandwidth, flatness, contrast, slope, crest) when `transforms` is also enabled |
| `processing`    | Normalization, scaling, clipping (requires `statistics`)                              |
| `editing`       | Trim, pad, reverse, perturb, concatenate (requires `statistics`, `random-generation`) |
| `channels`      | Interleave/deinterleave, mono↔stereo conversion                                       |
| `iir-filtering` | IIR filters: Butterworth, Chebyshev I/II, Elliptic (Cauer), Bessel; low/high/band-pass and band-stop; zero-phase `filtfilt`; design-once streaming `SosFilter` |
| `parametric-eq` | Parametric EQ bands and `ThreeBandEqConfig` (requires `iir-filtering`)                |
| `dynamic-range` | Compression, limiting, gating, expansion (config structs, side-chain support)         |
| `envelopes`     | Amplitude, RMS, and attack-decay envelopes                                            |
| `vad`           | Voice activity detection                                                              |

### Spectral and analysis

| Feature           | Description                                                                 |
| ----------------- | --------------------------------------------------------------------------- |
| `transforms`      | FFT, STFT, MFCC, chromagram, CQT, PSD                                       |
| `psychoacoustic`  | Bark/Mel band layouts, ATH, masking thresholds, SMR (requires `transforms`) |
| `pitch-analysis`  | YIN and autocorrelation pitch detection (requires `transforms`)             |
| `onset-detection` | Onset detection (requires `transforms`, `peak-picking`, `processing`)       |
| `beat-tracking`   | Beat tracking and tempo estimation (`estimate_tempo`)                       |
| `peak-picking`    | Peak picking on onset envelopes                                             |
| `decomposition`   | Audio decomposition (requires `onset-detection`)                            |

### Utility

| Feature             | Description                                                           |
| ------------------- | --------------------------------------------------------------------- |
| `resampling`        | Sample-rate conversion via rubato                                     |
| `random-generation` | Noise and random audio generation                                     |
| `fixed-size-audio`  | Fixed-size buffer support (no heap allocation)                        |
| `plotting`          | Interactive HTML plots via plotly                                     |
| `static-plots`      | PNG/SVG export (requires `plotting` — see [PLOTTING.md](PLOTTING.md)) |
| `simd`              | SIMD acceleration (nightly only)                                      |

### Bundles

| Feature            | Description                  |
| ------------------ | ---------------------------- |
| `full`             | All features                 |
| `full_no_plotting` | All features except plotting |

---

## Documentation

Full API documentation: [https://docs.rs/audio_samples](https://docs.rs/audio_samples)

---

## Examples

The repository includes runnable examples in `examples/`. Each is
self-contained and annotated with the required feature flags.

Additional demos:

- [DTMF encoder and decoder](https://github.com/jmg049/dtmf-demo)

---

## Companion Crates

- [`audio_samples_io`](https://github.com/jmg049/audio_samples_io) — Audio file decoding and encoding
- [`audio_samples_playback`](https://github.com/jmg049/audio_playback) — Device-level playback
- [`audio_samples_python`](https://github.com/jmg049/audio_python) — Python bindings
- [`spectrograms`](https://github.com/jmg049/Spectrograms) — Spectrogram and time–frequency transforms (used by the `transforms` feature)
- [`i24`](https://github.com/jmg049/i24) — 24-bit signed integer type for Rust
- [`dtmf_tones`](https://github.com/jmg049/dtmf_tones) — `no_std` DTMF keypad frequencies

---

## License

MIT License

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

Contributions are welcome. Please submit a pull request and see
[CONTRIBUTING.md](CONTRIBUTING.md) for guidance.

[crate]: https://crates.io/crates/audio_samples
[crate-img]: https://img.shields.io/crates/v/audio_samples?style=for-the-badge&color=009E73&label=crates.io
[docs]: https://docs.rs/audio_samples
[docs-img]: https://img.shields.io/badge/docs.rs-online-009E73?style=for-the-badge&labelColor=gray
[license-img]: https://img.shields.io/crates/l/audio_samples?style=for-the-badge&label=license&labelColor=gray
[license]: https://github.com/jmg049/audio_samples/blob/main/LICENSE
