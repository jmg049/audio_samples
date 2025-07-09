<div align="center">

# Audio Sample Processing & Conversion Library

<img src="logo.png" alt="audio_samples Logo" width="200"/>

[![Crates.io](https://img.shields.io/crates/v/audio_samples.svg)](https://crates.io/crates/audio_samples) [![Docs.rs](https://docs.rs/audio_sample/badge.svg)](https://docs.rs/audio_samples) ![MSRV: 1.70+](https://img.shields.io/badge/MSRV-1.70+-blue) [![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://pypi.org/project/audio-samples/)

</div>

A high-performance audio processing library for Rust with Python bindings.

This library provides a comprehensive set of tools for working with audio data,
including type-safe sample format conversions, statistical analysis, and various
audio processing operations.

## Core Features

- **Type-safe audio sample conversions** between i16, I24, i32, f32, and f64
- **High-performance operations** leveraging ndarray for efficient computation
- **Comprehensive metadata** tracking (sample rate, channels, duration)
- **Flexible data structures** supporting both mono and multi-channel audio
- **Python integration** via PyO3 bindings

## Example Usage

```rust
    use audio_samples::AudioSamples;
    use ndarray::array;

    // Create mono audio with sample rate
    let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let audio = AudioSamples::new_mono(data, 44100);

    assert_eq!(audio.sample_rate(), 44100);
    assert_eq!(audio.channels(), 1);
    assert_eq!(audio.samples_per_channel(), 5);
```
