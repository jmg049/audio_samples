# audio_samples

A high-performance audio processing library for Rust that provides type-safe sample format conversions, statistical analysis, and various audio processing operations.

## Overview

`audio_samples` is a comprehensive Rust library designed for efficient audio data manipulation, supporting a wide range of audio processing tasks with a focus on performance, type safety, and flexible design.

### Key Features

- **Type-safe audio sample conversions** between `i16`, `I24`, `i32`, `f32`, and `f64`
- **High-performance operations** leveraging `ndarray` for efficient computation
- **Comprehensive metadata tracking** (sample rate, channels, duration)
- **Flexible data structures** supporting both mono and multi-channel audio
- **Modular feature system** with fine-grained optional components
- **Advanced error handling** with chainable result types

### Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
audio_samples = "1.0.0"
```

Select features as needed:

```toml
audio_samples = { version = "1.0.0", features = ["full"] }
```

### Feature Flags

- `core-ops` (default): Basic audio operations
- `fft`: Spectral analysis and Fast Fourier Transform
- `plotting`: Visualization capabilities
- `resampling`: Advanced audio resampling
- `parallel-processing`: Parallel computation using Rayon
- `full`: Enables all features

## Quick Examples

### Basic Audio Creation and Manipulation

```rust
use audio_samples::AudioSamples;
use ndarray::array;

// Create mono audio with sample rate
let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
let audio = AudioSamples::new_mono(data, 44100);

assert_eq!(audio.sample_rate(), 44100);
assert_eq!(audio.channels(), 1);
```

### Chainable Processing with Error Handling

```rust
let result = audio
    .try_apply(|sample| sample * 0.5)
    .chain(|_| audio.try_apply(|sample| sample.clamp(-1.0, 1.0)))
    .map(|_| audio.samples_per_channel())
    .into_result();
```

### Beat Detection with Progress Tracking

```rust
let beat_tracker = audio.detect_beats_with_progress(
    &BeatConfig::new(120.0).with_tolerance(0.1),
    Some(0.5), // log compression
    Some(&progress_callback)
)?;

println!("Detected tempo: {} BPM", beat_tracker.tempo_bpm);
```

## Supported Sample Types

- `i16`
- `I24`
- `i32`
- `f32`
- `f64`

## Performance Considerations

- Uses `ndarray` for efficient computation
- Optional SIMD and parallel processing support
- Zero-cost abstractions for audio metadata tracking

## Error Handling

Implements a `ChainableResult` type for:
- Fluent method chaining
- Improved error handling ergonomics
- Built-in logging capabilities

## Documentation

Full API documentation is available at [docs.rs/audio_samples](https://docs.rs/audio_samples)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.