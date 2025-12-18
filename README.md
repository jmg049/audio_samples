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
explicit, but whose meaning is not: sample rate, channel layout,
amplitude range, memory interleaving, and PCM versus floating-point
semantics are tracked externally, if at all. In Rust, the situation is
reversed but not resolved. Libraries provide fast and safe low-level
primitives, yet users are still responsible for managing raw buffers,
writing ad hoc conversion code, and manually preserving invariants
across crates.

AudioSamples is designed to close this gap by providing a strongly
typed audio representation that makes audio semantics explicit and
enforced by construction. Sample format, numeric domain, channel
structure, and layout are encoded in the type system, and all
operations preserve or explicitly update these invariants.

The result is an API that supports both exploratory workflows and
reliable system-level use, without requiring users to remember hidden
conventions or reimplement common audio logic.

AudioSamples is the core data and processing layer of the broader
audio related crates. It defines the canonical audio object and the
operations that act upon it.

Other crates that build on this foundation:

- `audio_samples_io` for decoding and encoding audio containers into typed
  audio objects
- `audio_samples_playback` for device-level output
- `audio_samples_python` for Python bindings, enabling AudioSamples to act as a
  type-safe backend for Python workflows
- `html_view` for lightweight visualisation and inspection, generating
  self-contained HTML outputs suitable for analysis and reporting
  
By separating representation from I/O, playback, and visualisation,
AudioRs remains modular while enforcing a single, consistent audio
model throughout the stack.

---

## Installation

```bash
cargo add audio_samples
```

See the [Features](#features) for more details.

---

## Quick Start

### Generating and mixing signals

This example generates a sine wave in a target sample format, converts
it to floating-point samples, and mixes it with a second signal.

```rust
use audio_samples::{
    AudioProcessing, AudioTypeConversion, cosine_wave, operations::types::NormalizationMethod,
    sine_wave,
};
use std::time::Duration;

fn main() {
    let sample_rate = 44_100;
    let duration = Duration::from_secs_f64(1.0);
    let frequency = 440.0;
    let amplitude = 0.5;

    // Generate a sine wave with i16 output samples.
    // The waveform is computed in f32 and converted into i16.
    let pcm_sine = sine_wave::<i16, f32>(frequency, duration, sample_rate, amplitude);

    // Convert to floating-point representation
    let float_sine = pcm_sine.to_format::<f32>();

    // Generate a second signal directly as floating-point samples
    let cosine = cosine_wave::<f32, f32>(frequency / 2.0, duration, sample_rate, amplitude);

    // Mix the two signals
    let mixed = (float_sine + cosine).normalize(-1.0, 1.0, NormalizationMethod::MinMax);
}
```

---

### Spectral transforms and analysis

AudioSamples supports spectral and time–frequency transforms via the
`AudioTransforms` trait, enabled by the `spectral-analysis` feature.
These operations produce standard frequency-domain and
time–frequency representations used in audio analysis and research.

Enable the feature:

```bash
cargo add audio_samples --features spectral-analysis
```

#### Example: STFT, spectrogram, and MFCC computation

```rust
use audio_samples::{
    AudioProcessing, AudioTypeConversion, cosine_wave, operations::types::NormalizationMethod,
    sine_wave,
};
use std::time::Duration;

fn main() {
    let sample_rate = 44_100;
    let duration = Duration::from_secs_f64(1.0);
    let frequency = 440.0;
    let amplitude = 0.5;

    // Generate a sine wave with i16 output samples.
    // The waveform is computed in f32 and converted into i16.
    let pcm_sine = sine_wave::<i16, f32>(frequency, duration, sample_rate, amplitude);

    // Convert to floating-point representation
    let float_sine = pcm_sine.to_format::<f32>();

    // Generate a second signal directly as floating-point samples
    let cosine = cosine_wave::<f32, f32>(frequency / 2.0, duration, sample_rate, amplitude);

    // Mix the two signals
    let mixed = (float_sine + cosine).normalize(-1.0, 1.0, NormalizationMethod::MinMax);
}
```

## Why Use audio_samples?

AudioSamples exists to make audio semantics explicit and enforceable.

In many audio libraries, audio data is represented as a numeric buffer
with metadata tracked separately or implicitly. Sample rate, channel
layout, amplitude domain, and sample representation often exist outside
the type system and are maintained by convention. As a result,
mismatches between representations can propagate silently through
pipelines, particularly when converting between integer PCM and
floating-point formats or combining signals from different sources.

AudioSamples addresses this by treating audio as a structured object.
An `AudioSamples<'a, T>` value couples sample data with its sample rate
and channel layout, and operations on audio explicitly preserve or
update these invariants. Conversions between sample formats are defined
in terms of semantic transformations rather than raw casts, ensuring
that changes in numerical representation are intentional and
well-defined.

This design supports workflows where correctness matters: research
pipelines, long-lived systems code, and multi-stage audio processing
where buffers pass through several components. Rather than relying on
discipline or external documentation, AudioSamples encodes audio
assumptions directly in the API.

AudioSamples is the core data and processing layer of the broader
audio related crates. It defines the canonical audio object and the
operations that act upon it.

Other crates in the ecosystem build on this foundation:

- `audio_samples_io` for decoding and encoding audio containers into typed
  audio objects
- `audio_samples_playback` for device-level output
- `audio_samples_python` for Python bindings, enabling AudioSamples to act as a
  type-safe backend for Python workflows
- `html_view` for lightweight visualisation and inspection, generating
  self-contained HTML outputs suitable for analysis and reporting

By separating representation from I/O, playback, and visualisation,
AudioRs remains modular while enforcing a single, consistent audio
model throughout the stack.

---

## <a name="features">Features</a>

### Default features

- `statistics`
- `processing`
- `editing`
- `channels`

### Major functionality groups

- `fft`
- `resampling`
- `serialization`
- `plotting`

### Transform and analysis features

- `spectral-analysis`
- `beat-detection` (requires `spectral-analysis`)

### Plotting sub-features

- `static-plots` (PNG output)

### Performance features

- `parallel-processing`
- `simd` (nightly only)
- `mkl`
- `fixed-size-audio`

### Utility features

- `formatting`
- `random-generation`
- `utilities-full`

---

## Documentation

Full API documentation is available at
[https://docs.rs/audio_samples](https://docs.rs/audio_samples)

---

## Examples

A range of examples is included in the repository.

Additional demos include:

- DTMF encoder and decoder
- Basic synthesis examples
- Audio inspection utilities

More advanced I/O and playback examples are provided in the companion
crates.

---

## AudioRs — Companion Crates

### [`audio_samples_io`](https://github.com/jmg049/audio_samples_io)

Rust crate providing audio file I/O utilities and helpers.
`audio_samples_io` is the IO extension of the [audio_samples](https://crates.io/crates/audio_samples) crate.

### [`audio_samples_playback`](https://github.com/jmg049/audio_playback)

Device-level playback built on AudioSamples.

### [`audio_samples_python`](https://github.com/jmg049/audio_python)

Python bindings exposing AudioSamples, AudioIO and AudioPlayback.

### [`html_view`](https://github.com/jmg049/HTMLView)

A lightweight, cross-platform HTML viewer for Rust.

`html_view` provides a minimal, ergonomic API for rendering HTML content in a native window, similar in spirit to `matplotlib.pyplot.show()` for visualisation rather than UI development.

### [`dtmf_tones`](https://github.com/jmg049/dtmf_tones)

A zero-heap, `no_std` friendly, **const-first** implementation of the standard DTMF (Dual-Tone Multi-Frequency) keypad used in telephony systems.  
This crate provides compile-time safe mappings between keypad keys and their canonical low/high frequencies, along with **runtime helpers** for practical audio processing.

### [`i24`](https://github.com/jmg049/i24)

i24 provides a 24-bit signed integer type for Rust, filling the gap between i16 and i32. This type is particularly useful in audio processing, certain embedded systems, and other scenarios where 24-bit precision is required but 32 bits would be excessive

---

## License

MIT License

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
