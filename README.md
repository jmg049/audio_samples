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

### Python

```bash
pip install audio-samples
```

```python
import audio_samples as aus
import numpy as np

# Create audio data
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
audio = aus.AudioSamples(data, sample_rate=44100)

# Analysis and processing
peak = audio.peak()
rms = audio.rms()
normalized = audio.normalize()
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

### Python Integration

- **Full API coverage** via PyO3 bindings
- **NumPy integration** for seamless array operations
- **Type safety** maintained across the Python boundary

### Installation

#### Rust (Cargo)

```toml
# Basic installation
[dependencies]
audio_samples = "0.1"

# With streaming and playback
[dependencies]
audio_samples = { version = "0.1", features = ["streaming", "playback"] }

# All features
[dependencies]
audio_samples = { version = "0.1", features = ["python", "parallel-processing", "streaming", "playback", "progress-tracking"] }
```

#### Python (pip)

```bash
# Install from PyPI
pip install audio-samples

# Install development version
pip install git+https://github.com/your-repo/audio_samples.git
```

#### From Source

```bash
# Clone and build Rust library
git clone https://github.com/your-repo/audio_samples.git
cd audio_samples
cargo build --release

# Build Python bindings
pip install maturin
maturin develop --features python
```

## Examples

### Audio Type Conversion

```rust
use audio_samples::{AudioSamples, ConvertTo};
use ndarray::array;

// Convert between different sample formats
let data_f32 = array![0.5f32, -0.3, 0.8, -1.0];
let audio = AudioSamples::new_mono(data_f32, 48000);

// Convert to different formats
let audio_i16 = audio.as_type::<i16>()?;  // 16-bit integers
let audio_i24 = audio.as_type::<I24>()?;  // 24-bit integers
let audio_i32 = audio.as_type::<i32>()?;  // 32-bit integers
```

### Statistical Analysis

```rust
use audio_samples::operations::*;

// Comprehensive audio statistics
let peak = audio.peak();
let rms = audio.rms()?;
let variance = audio.variance()?;
let zero_crossings = audio.zero_crossings();
let zcr = audio.zero_crossing_rate();
```

### Real-time Streaming

```rust
use audio_samples::streaming::{sources::generator::GeneratorSource, traits::AudioSource};

// Generate and stream audio in real-time
let mut generator = GeneratorSource::<f32>::sine(440.0, 48000, 2);

while let Ok(Some(chunk)) = generator.next_chunk().await {
    // Process audio chunk in real-time
    println!("Generated {} samples", chunk.samples_per_channel());
}
```

### Python Usage

```python
import audio_samples as aus
import numpy as np

# Load and process audio
audio = aus.load_audio("input.wav")

# Statistical analysis
print(f"Peak: {audio.peak()}")
print(f"RMS: {audio.rms()}")
print(f"Duration: {audio.duration_seconds():.2f}s")

# Processing operations
normalized = audio.normalize()
filtered = audio.apply_lowpass_filter(cutoff_freq=8000)

# Type conversions
audio_16bit = audio.as_type("i16")
audio_24bit = audio.as_type("I24")
```

## Performance Characteristics

- **Zero-copy operations** where possible using ndarray views
- **SIMD optimizations** for mathematical operations
- **Lock-free streaming** buffers for real-time performance
- **Memory-efficient** sample format conversions
- **Parallel processing** support via rayon (optional)

## Documentation

- **[API Documentation](https://docs.rs/audio_samples)** - Complete Rust API reference
- **[Examples Directory](examples/)** - Comprehensive usage examples
- **[Python Guide](docs/python.md)** - Python-specific documentation
- **[Streaming Guide](STREAMING_PLAYBACK_GUIDE.md)** - Real-time audio processing
- **[Performance Guide](docs/performance.md)** - Optimization tips and benchmarks

## Supported Audio Formats

| Format | Description | Range | Precision |
|--------|-------------|-------|-----------|
| i16 | 16-bit signed integer | -32,768 to 32,767 | Most common format |
| I24 | 24-bit signed integer | -8,388,608 to 8,388,607 | Professional audio |
| i32 | 32-bit signed integer | -2³¹ to 2³¹-1 | High precision integer |
| f32 | 32-bit float | -1.0 to 1.0 | Single precision float |
| f64 | 64-bit float | -1.0 to 1.0 | Double precision float |

All conversions maintain mathematical precision and handle edge cases properly.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/audio_samples.git
cd audio_samples

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Run tests
cargo test

# Run tests with all features
cargo test --all-features

# Format code
cargo fmt

# Run linting
cargo clippy
```

### Building Python Bindings

```bash
# Install maturin
pip install maturin

# Build and install for development
maturin develop --features python

# Build wheel for distribution
maturin build --features python
```

## Community

- **[Issues](https://github.com/your-repo/audio_samples/issues)** - Bug reports and feature requests
- **[Discussions](https://github.com/your-repo/audio_samples/discussions)** - General questions and community chat
- **[Documentation](https://docs.rs/audio_samples)** - Complete API reference

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Acknowledgments

This library builds on the excellent work of:

- [ndarray](https://github.com/rust-ndarray/ndarray) - N-dimensional arrays for Rust
- [PyO3](https://github.com/PyO3/pyo3) - Rust bindings for Python
- [i24](https://crates.io/crates/i24) - 24-bit integer support
- [cpal](https://github.com/RustAudio/cpal) - Cross-platform audio I/O

---

**[Documentation](https://docs.rs/audio_samples)** | **[Crates.io](https://crates.io/crates/audio_samples)** | **[PyPI](https://pypi.org/project/audio-samples/)**
