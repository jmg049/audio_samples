<div align="center">

# Audio Sample Processing & Conversion Library

<img src="logo.png" alt="audio_samples Logo" width="200"/>

[![Crates.io](https://img.shields.io/crates/v/audio_samples.svg)](https://crates.io/crates/audio_samples) [![Docs.rs](https://docs.rs/audio_sample/badge.svg)](https://docs.rs/audio_samples) ![MSRV: 1.70+](https://img.shields.io/badge/MSRV-1.70+-blue) [![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://pypi.org/project/audio-samples/)

</div>

A high-performance audio processing library for Rust with Python bindings, providing type-safe audio sample conversions, comprehensive statistical analysis, real-time streaming, and professional audio processing capabilities.

## Quick Start

### Rust

```toml
[dependencies]
audio_samples = "0.1"
```

```rust
use audio_samples::{AudioSamples, operations::*};
use ndarray::array;

// Create and process audio data
let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
let audio = AudioSamples::new_mono(data, 44100);

// Statistical analysis
let peak = audio.peak(); // Returns f32 directly
let rms = audio.rms()?;  // Returns Result<f64>

// Type conversion
let audio_i16 = audio.as_type::<i16>()?;
```

## Features Overview

### Core Audio Processing

- **Type-safe conversions** between all major audio formats (i16, I24, i32, f32, f64)
- **Statistical analysis** (RMS, peak, variance, zero-crossing rate, autocorrelation)
- **Audio processing** (normalization, filtering, effects, dynamic range control)
- **Spectral analysis** (FFT, transforms, frequency domain operations)
- **Multi-channel support** with flexible channel layouts

### Real-time Capabilities

- **Streaming audio** from TCP/UDP sources with adaptive buffering
- **Real-time playback** with low-latency audio output
- **Signal generation** (sine, square, sawtooth, noise, chirps)
- **Effects processing** (reverb, delay, compression, EQ)
