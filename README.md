<div align="center">

# AudioSamples

## Fast, simple, and expressive audio in Rust

<img src="logo.png" title="AudioSamples Logo -- Ferrous' Mustachioed Cousin From East Berlin, Eisenhaltig" width="200"/>

[![Crates.io][crate-img]][crate] [![Docs.rs][docs-img]][docs] [![License: MIT][license-img]][license]
</div>

A high-performance audio processing library for Rust that provides type-safe sample format conversions, statistical analysis, and various audio processing operations.

Core building block of the wider [AudioRs](link_to_website_in_development) ecosystem.

## Overview

<!-- This section is reserved for the project's purpose and motivation. -->


## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
audio_samples = "1.0.0"
```

or more easily with:

```bash
cargo add audio_samples
```


For specific features, enable only what you need:

```toml
[dependencies]
audio_samples = { version = "1.0.0", features = ["fft", "plotting"] }
```

Or enable everything:

```toml
[dependencies]
audio_samples = { version = "1.0.0", features = ["full"] }
```

## Features

The library uses a modular feature system to keep dependencies minimal:

- **`core-ops`** (default) - Basic audio operations and statistics
- **`fft`** - Fast Fourier Transform and spectral analysis
- **`plotting`** - Audio visualization capabilities
- **`resampling`** - High-quality audio resampling
- **`parallel-processing`** - Multi-threaded processing with Rayon
- **`simd`** - SIMD acceleration for supported operations
- **`beat-detection`** - Tempo and beat tracking (requires `fft`)
- **`full`** - Enables all features

## Full Examples

A range of examples demonstrating this crate and its companion [audio_io](https://ghithub.com/jmg049/audio_io) crate can be found at [here]().

- [DTMF tone generation and decoding]()
- [Basic synthesizer]()
- [Silence Trimming CLI tool]()
- [Audio file information CLI tool]()

## Available Operations

The library organizes functionality into focused traits:

### Core Audio Operations (`core-ops`)

- **Statistics** (`AudioStatistics`) - Peak, RMS, mean, variance, zero-crossings, autocorrelation
- **Processing** (`AudioProcessing`) - Normalize, scale, clip, filtering, compression, DC removal
- **Channel Operations** (`AudioChannelOps`) - Mono/stereo conversion, channel extraction, pan, balance
- **Editing** (`AudioEditing`) - Trim, pad, reverse, fade, split, concatenate, mix

### Signal Processing Features

- **IIR Filtering** (`AudioIirFiltering`) - Biquad filters, shelving, peaking
- **Parametric EQ** (`AudioParametricEq`) - Multi-band equalizer with adjustable Q
- **Dynamic Range** (`AudioDynamicRange`) - Compression, limiting, expansion, gating

### Advanced Analysis (Optional Features)

- **Spectral Analysis** (`AudioTransforms`) - FFT, STFT, spectrogram, mel-spectrogram, MFCC, CQT
- **Pitch Analysis** (`AudioPitchAnalysis`) - Fundamental frequency, harmonic analysis
- **Beat Detection** - Tempo analysis and beat tracking
- **Plotting** (`AudioPlottingUtils`) - Waveform, spectrogram, and frequency domain visualization

## Quick Start

### Creating Audio Data

```rust
use audio_samples::AudioSamples;
use ndarray::array;

// Create mono audio
let data = array![0.1f32, 0.5, -0.3, 0.8, -0.2];
let audio = AudioSamples::new_mono(data, 44100);

// Create stereo audio
let stereo_data = array![
    [0.1f32, 0.5, -0.3],  // Left channel
    [0.8f32, -0.2, 0.4]   // Right channel
];
let stereo_audio = AudioSamples::new_multi_channel(stereo_data, 44100);
```

### Basic Statistics

```rust
use audio_samples::AudioStatistics;

// Simple statistics (no Result needed)
let peak = audio.peak();
let min = audio.min_sample();
let max = audio.max_sample();
let mean = audio.mean();

// More complex statistics (return Result)
let rms = audio.rms()?;
let variance = audio.variance()?;
let zero_crossings = audio.zero_crossings();
```

### Processing Operations

```rust
use audio_samples::{AudioProcessing, NormalizationMethod};

let mut audio = AudioSamples::new_mono(data, 44100);

// Basic processing (in-place)
audio.normalize(-1.0, 1.0, NormalizationMethod::Peak)?;
audio.scale(0.5); // Reduce volume by half
audio.remove_dc_offset();
```

### Type Conversions

```rust
// Convert between sample types
let audio_f32 = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0], 44100);
let audio_i16 = audio_f32.as_type::<i16>()?;
let audio_f64 = audio_f32.as_type::<f64>()?;
```

### Iterating Over Audio Data

```rust
use audio_samples::AudioSampleIterators;

// Iterate by frames (one sample from each channel)
for frame in audio.frames() {
    println!("Frame: {:?}", frame);
}

// Iterate by channels
for channel in audio.channels() {
    println!("Channel: {:?}", channel);
}

// Windowed iteration for analysis
for window in audio.windows(1024, 512) {
    // Process 1024-sample windows with 50% overlap
    let window_rms = window.rms()?;
    println!("Window RMS: {:.3}", window_rms);
}
```

## Builder Pattern for Complex Processing

For more complex operations, use the fluent builder API:

```rust
use audio_samples::{AudioSamples, NormalizationMethod};

let mut audio = AudioSamples::new_mono(data, 44100);

// Chain multiple operations
audio.processing()
    .normalize(-1.0, 1.0, NormalizationMethod::Peak)
    .scale(0.8)
    .remove_dc_offset()
    .apply()?;
```


## Documentation

Full API documentation is available at [docs.rs/audio_samples](https://docs.rs/audio_samples).

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

[crate]: https://crates.io/crates/audio_samples  
[crate-img]: https://img.shields.io/crates/v/audio_samples?style=for-the-badge&color=009E73&label=crates.io

[docs]: https://docs.rs/audio_samples  
[docs-img]: https://img.shields.io/badge/docs.rs-online-009E73?style=for-the-badge&labelColor=gray

[license-img]: https://img.shields.io/crates/l/audio_samples?style=for-the-badge&label=license&labelColor=gray  
[license]: https://github.com/jmg049/audio_samples/blob/main/LICENSE
