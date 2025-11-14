# AudioSamples Architecture

This document provides detailed information on how the audio_samples crate is architected from its most fundamental building blocks up to the highest-level API. The architecture follows core principles of **type safety**, **zero-allocation efficiency**, **trait-based composition**, and **modular feature design**.

## Core Design Principles

### 1. Type Safety Through Strong Typing

All audio data is strongly typed with the sample format (`i16`, `I24`, `i32`, `f32`, `f64`), ensuring mathematical operations are performed with appropriate precision and range. The type system prevents common audio processing errors like mixing incompatible sample formats.

### 2. Zero-Allocation Efficiency

The library leverages `ndarray`'s view system to enable zero-allocation access patterns wherever possible. Operations prefer in-place modifications and views over copying data.

### 3. Trait-Based Composition

Functionality is organized into focused, composable traits rather than monolithic implementations. Each trait handles a specific aspect of audio processing with clear separation of concerns.

### 4. Metadata Integration

Audio samples are always paired with essential metadata (sample rate, channel layout) to prevent common audio processing errors and enable automatic format conversions.

### 5. Feature-Gated Modularity

The library uses cargo features extensively to keep dependencies minimal, allowing users to enable only the functionality they need.

## Building Blocks

### AudioSample (./traits.rs)

The `AudioSample` trait is the foundation of the entire type system. It defines the interface for all supported audio sample formats:

**Supported Types:**

- `i16`: 16-bit signed integer samples (most common for audio files)
- `I24`: 24-bit signed integer samples (professional audio)
- `i32`: 32-bit signed integer samples (high precision)
- `f32`: 32-bit floating-point samples (normalized -1.0 to 1.0)
- `f64`: 64-bit floating-point samples (highest precision)

**Key Requirements:**

- Standard arithmetic operations (`Add`, `Sub`, `Mul`, `Div`)
- Memory safety guarantees (`NoUninit` for safe byte serialization)
- Numeric operations (`Num`, `Zero`, `One`, `Signed`)
- Serialization support (`Serialize`, `Deserialize`)
- Constants for range information (`MAX`, `MIN`, `BITS`)

**API Contracts:**

- All sample types must provide consistent arithmetic behavior
- Byte serialization must be safe and deterministic
- Range constants must accurately represent the format's dynamic range

### ConvertTo (./traits.rs)

The `ConvertTo<T>` trait provides audio-aware conversions between different sample formats with proper scaling:

**Conversion Behavior:**

- **Integer ↔ Integer**: Bit-shift scaling to preserve full dynamic range
- **Integer ↔ Float**: Normalized scaling (-1.0 to 1.0 for floats)
- **Float ↔ Float**: Direct casting with precision conversion
- **I24 Special Handling**: Custom methods for 24-bit operations

**Design Patterns:**

- Returns `AudioSampleResult<T>` to handle conversion failures
- Uses macro-generated implementations for consistency
- Maintains mathematical precision across format boundaries
- Handles edge cases like range overflows gracefully

**API Contracts:**

- Conversions must preserve audio dynamic range proportionally
- Round-trip conversions should minimize precision loss
- Error cases must be clearly documented and handled

### CastInto, CastFrom and Castable (./traits.rs)

The casting trait family provides raw numeric conversions without audio-specific scaling:

**Purpose:**

- Raw numeric casting for non-audio operations
- Direct type conversions without range normalization
- Performance-critical paths where scaling is not needed

**Design Patterns:**

Sometimes you just need to cast an int to a float.

- `CastFrom<S>`: Cast from source type to Self
- `CastInto<T>`: Cast self into target type
- `Castable`: Marker trait for types that can cast to all audio formats


**API Contracts:**

- Casting preserves numeric values without audio scaling
- Out-of-range values are clamped to target type's limits
- No error handling - assumes well-formed input

### AudioSamples<'_, T: AudioSample> (repr.rs)

The main data container that combines audio samples with essential metadata:

```rust
pub struct AudioSamples<'a, T: AudioSample> {
    pub data: AudioData<'a, T>,
    pub sample_rate: u32,
    pub layout: ChannelLayout,
}
```

**Key Features:**

- Generic over any `AudioSample` type
- Lifetime parameter `'a` enables zero-copy views
- Always includes sample rate and channel layout
- Provides uniform interface for mono and multi-channel audio

**Memory Layout:**

- Mono audio: 1D arrays via `MonoData<'a, T>`
- Multi-channel audio: 2D arrays via `MultiData<'a, T>` with channels as rows
- Both support borrowed (`ArrayView`) and owned (`Array`) data

**API Contracts:**

- Sample rate must be positive
- Channel layout must match data dimensions
- Lifetime safety ensured through Rust's borrow checker
- Metadata consistency maintained across operations

### AudioSamples Iteration (iterators.rs)

Provides multiple iteration patterns for efficient audio processing:

**Iterator Types:**

- `frames()`: Iterate by frames (one sample from each channel)
- `channels()`: Iterate by complete channels
- `windows(size, hop)`: Windowed iteration with configurable overlap
- Support for different padding modes: `Zero`, `None`, `Skip`

**Design Patterns:**

- Zero-allocation views where possible
- Configurable windowing for FFT and analysis operations
- Type-safe iterator adaptors
- Memory-efficient streaming for large audio files

**API Contracts:**

- Iterator stability guaranteed for immutable operations
- Window boundaries handled consistently
- Padding modes clearly defined and documented

### AudioSamples Conversion (conversions.rs)

Implements the `AudioTypeConversion` trait for safe type transformations:

**In-Domain Conversions:**

- `as_type<O>()`: Borrows original, returns new type
- `to_type<O>()`: Consumes original, returns new type
- Convenience methods: `as_f32()`, `as_i16()`, `as_i24()`, etc.
- Uses `ConvertTo` trait for audio-aware scaling

**Out-of-Domain Conversions:**

- `cast_as<O>()`: Borrows original, raw numeric casting
- `cast_to<O>()`: Consumes original, raw numeric casting
- Uses `CastFrom` trait for direct numeric conversion

**API Contracts:**

- Clear distinction between audio-aware and raw conversions
- Lifetime management ensures memory safety
- Type bounds enforce conversion compatibility

### Utilities (./utils)

Provides supporting functionality organized by purpose:

- `generation.rs`: Signal generation (sine waves, noise, etc.)
- `detection.rs`: Feature detection algorithms
- `comparison.rs`: Audio comparison and similarity metrics

**Design Patterns:**

- Pure functions where possible
- Consistent error handling patterns
- Performance-optimized implementations
- Feature-gated advanced functionality

### Errors (./error.rs)

Comprehensive error handling with specific error types:

```rust
pub enum AudioSampleError {
    ConversionError(String, String, String, String),
    InvalidRange(String),
    InvalidParameter(String),
    DimensionMismatch(String),
    InvalidInput { msg: String },
    ProcessingError { msg: String },
    FeatureNotEnabled { feature: String },
    ArrayLayoutError { message: String },
    OptionError { message: String },
    BorrowedDataError { message: String },
    InternalError(String),
}
```

**Error Handling Strategy:**

- Specific error types for different failure modes
- Rich context information in error messages
- Integration with `thiserror` for ergonomic handling
- `AudioSampleResult<T>` type alias for consistency

## Trait Extensions

### The `operations` Module (./operations)

The operations module organizes audio processing functionality into focused, composable traits. Each trait addresses a specific domain of audio processing:

#### Core Organization

- **`traits.rs`**: Trait definitions and type bounds
- **Implementation files**: Named after traits (e.g., `AudioChannelOps` → `channels.rs`)
- **Supporting types**: Enums and structs in `types.rs`
- **Feature gating**: Advanced functionality behind cargo features

#### `AudioStatistics` (statistics.rs)

**Purpose**: Statistical analysis operations for audio data.

**Why use it?**: Essential for audio analysis, level monitoring, and processing decisions.

**How to use it?**:

```rust
let peak = audio.peak();          // Returns T directly
let rms: f64 = audio.rms()?;      // Returns Result<f64>
let crossings = audio.zero_crossings(); // Signal analysis
```

**Key Operations**:

- `peak()`, `min_sample()`, `max_sample()`: Level measurements
- `mean()`, `variance()`, `std_dev()`: Statistical measures
- `rms()`: Perceptually relevant loudness measure
- `zero_crossings()`, `zero_crossing_rate()`: Periodicity analysis
- `autocorrelation()`, `cross_correlation()`: Correlation analysis (requires `fft`)
- `spectral_centroid()`: Brightness measure (requires `fft`)

**API Contracts**:

- Simple measures return values directly (never fail)
- Complex computations return `Result` types
- Generic over float types for numerical operations
- Consistent behavior across mono and multi-channel audio

#### `AudioProcessing` (processing.rs)

**Purpose**: Core signal processing operations with fluent builder API.

**Why use it?**: Fundamental audio modifications like normalization, scaling, filtering.

**How to use it?**:

```rust
// Individual operations
audio.normalize(-1.0, 1.0, NormalizationMethod::Peak)?;
audio.scale(0.8)?;

// Fluent builder API
audio.processing()
    .normalize(-1.0, 1.0, NormalizationMethod::Peak)
    .scale(0.8)
    .clip(-0.5, 0.5)
    .apply()?;
```

**Key Operations**:

- `normalize()`: Multiple normalization strategies
- `scale()`, `gain()`: Amplitude adjustments
- `clip()`: Hard limiting to prevent overflows
- `remove_dc_offset()`: DC bias removal
- `fade_in()`, `fade_out()`: Envelope operations

**ProcessingBuilder Pattern**:

- Chains operations efficiently
- Validates parameters before application
- Atomic application (all or nothing)
- Memory-efficient operation sequencing

#### `AudioTransforms` (transforms.rs)

**Purpose**: Frequency-domain analysis and transformations (requires `fft`).

**Why use it?**: Spectral analysis, filtering, and frequency-domain processing.

**How to use it?**:

```rust
let spectrum = audio.fft()?;
let stft = audio.stft(window_size, hop_size)?;
let filtered = audio.spectral_filter(cutoff_freq)?;
```

**Key Operations**:

- `fft()`, `ifft()`: Fast Fourier Transform and inverse
- `stft()`, `istft()`: Short-Time Fourier Transform
- `spectrogram()`: Time-frequency representation
- `spectral_filter()`: Frequency domain filtering
- `phase_vocoder()`: Time/pitch manipulation

**Performance Considerations**:

- Uses `RustFFT` by default
- Optional Intel MKL backend (`mkl` feature)
- Real-valued FFT optimization for audio signals
- Memory-efficient windowing operations

#### `AudioEditing` (editing.rs)

**Purpose**: Time-domain editing and manipulation operations.

**Why use it?**: Audio arrangement, timing modifications, and content editing.

**How to use it?**:

```rust
let trimmed = audio.trim_start_end(1.0, 2.0)?;  // Remove first 1s, last 2s
let padded = audio.pad(PadSide::Both, 0.5)?;   // Add 0.5s silence
let reversed = audio.reverse()?;                // Reverse audio
let combined = audio1.concatenate(&audio2)?;   // Join audio
```

**Key Operations**:

- `trim()`, `trim_start_end()`: Remove audio segments
- `pad()`: Add silence or repeat edge samples
- `reverse()`: Reverse audio timeline
- `concatenate()`: Join multiple audio clips
- `insert_silence()`: Add silence at specific points
- `repeat()`: Loop audio content

**API Contracts**:

- Time-based operations accept seconds or samples
- Sample rate consistency enforced across operations
- Memory-efficient implementation using views where possible

#### `AudioChannelOps` (channels.rs)

**Purpose**: Channel manipulation and spatial audio operations.

**Why use it?**: Mono/stereo conversion, channel mixing, spatial processing.

**How to use it?**:

```rust
let mono = stereo_audio.to_mono(MonoConversionMethod::Average)?;
let stereo = mono_audio.to_stereo(StereoConversionMethod::Duplicate)?;
let extracted = multi_audio.extract_channel(0)?;
```

**Key Operations**:

- `to_mono()`: Multiple mono conversion strategies
- `to_stereo()`: Stereo expansion from mono
- `extract_channel()`: Extract specific channels
- `mix_channels()`: Custom channel mixing
- `swap_channels()`: Channel reordering

**Conversion Methods**:

- **Mono**: `Average`, `Left`, `Right`, `Sum`, `WeightedSum`
- **Stereo**: `Duplicate`, `Spread`, `Custom`

#### `AudioIirFiltering` (iir_filtering.rs)

**Purpose**: Infinite Impulse Response filter implementations.

**Why use it?**: Real-time filtering, tone shaping, frequency response control.

**How to use it?**:

```rust
let filtered = audio.lowpass_filter(cutoff_hz, q_factor)?;
let shaped = audio.highpass_filter(cutoff_hz, q_factor)?;
let custom = audio.biquad_filter(coefficients)?;
```

**Filter Types**:

- `lowpass_filter()`, `highpass_filter()`: Basic frequency separation
- `bandpass_filter()`, `bandstop_filter()`: Band-limited filtering
- `biquad_filter()`: Custom biquad coefficients
- `butterworth_filter()`: Smooth response filters

#### `AudioParametricEq` (parametric_eq.rs)

**Purpose**: Multi-band parametric equalization.

**Why use it?**: Tone shaping, frequency response correction, creative EQ.

**How to use it?**:

```rust
let eq = ParametricEq::new()
    .add_band(EqBand::new(1000.0, 2.0, 3.0)) // +3dB at 1kHz
    .add_band(EqBand::new(5000.0, 1.0, -6.0)); // -6dB at 5kHz

let equalized = audio.apply_parametric_eq(&eq)?;
```

**Key Features**:

- Multiple simultaneous frequency bands
- Independent gain, frequency, and Q controls
- Real-time parameter updates
- Efficient cascaded biquad implementation

#### `AudioDynamicRange` (dynamic_range.rs)

**Purpose**: Dynamic range processing (compression, limiting, expansion).

**Why use it?**: Level control, dynamics processing, mastering operations.

**How to use it?**:

```rust
let compressed = audio.compressor(CompressorConfig::new(
    threshold: -12.0,
    ratio: 4.0,
    attack: 0.003,
    release: 0.1
))?;

let limited = audio.limiter(LimiterConfig::new(-1.0, 0.001, 0.05))?;
```

**Processor Types**:

- `compressor()`: Reduce dynamic range above threshold
- `limiter()`: Hard limiting to prevent peaks
- `expander()`: Increase dynamic range below threshold
- `gate()`: Remove low-level noise

#### Audio Resampling (resampling.rs)

**Purpose**: High-quality sample rate conversion (requires `resampling`).

**Why use it?**: Sample rate conversion, format compatibility, anti-aliasing.

**How to use it?**:

```rust
let resampled = audio.resample(48000, ResamplingQuality::VeryHigh)?;
let converted = audio.resample_to_target_rate(target_audio.sample_rate())?;
```

**Quality Levels**:

- `Fast`: Quick conversion with acceptable quality
- `Medium`: Balanced quality and performance
- `High`: High-quality anti-aliasing
- `VeryHigh`: Maximum quality for critical applications

### Plotting (./operations/plotting)

**Purpose**: Comprehensive audio visualization capabilities (requires `plotting`).

**Architecture**:

- **Composable API**: Build complex plots from simple elements
- **Builder Pattern**: Fluent configuration of plot appearance
- **Multiple Backends**: Plotly for interactive plots, static generation support

**Core Components**:

#### `PlotComposer` (composer.rs)

Orchestrates the creation of complex, multi-element plots:

```rust
let plot = PlotComposer::new()
    .add_waveform(&audio, "Waveform")
    .add_spectrogram(&audio, window_size)
    .add_onsets(&onset_times)
    .set_layout(layout_config)
    .build()?;
```

#### Plotting Elements (elements.rs)

- `Waveform`: Time-domain amplitude plots
- `Spectrogram`: Time-frequency representations
- `Spectrum`: Frequency-domain magnitude plots
- `OnsetMarkers`: Event detection visualization
- `BeatMarkers`: Tempo and rhythm visualization
- `PitchContour`: Fundamental frequency tracking

#### Styling System (builders.rs)

- `ColorPalette`: Consistent color schemes
- `LineStyle`: Customizable line appearance
- `MarkerStyle`: Point and event markers
- `LayoutConfig`: Plot layout and formatting

**Feature Integration**:

- Automatic sample rate handling for time axis
- Frequency axis scaling for spectral plots
- Real-time plot updates for streaming audio
- Export capabilities (PNG, SVG, HTML)

**API Contracts**:

- Consistent time/frequency axis handling
- Memory-efficient plot generation
- Thread-safe plot composition
- Graceful fallbacks for missing features

### SIMD Conversions (simd_conversions.rs)

**Purpose**: Vectorized high-performance sample format conversions (requires `simd`).

**Optimization Strategy**:

- Process 8 samples simultaneously using AVX2 instructions
- Automatic fallback to scalar operations for remainder samples
- Platform-specific optimizations with consistent API

**Key Functions**:

```rust
convert_f32_to_i16_simd(input: &[f32], output: &mut [i16]) -> AudioSampleResult<()>
convert_i16_to_f32_simd(input: &[i16], output: &mut [f32]) -> AudioSampleResult<()>
```

**Performance Benefits**:

- Up to 8x speedup for bulk conversions
- Maintains numerical precision of scalar operations
- Zero-allocation implementation
- Automatic SIMD capability detection

### Realtime (realtime.rs)

**Purpose**: Real-time audio processing capabilities with low-latency guarantees.

**Key Features**:

- Lock-free data structures for audio threading
- Configurable buffer sizes for latency control
- Real-time safe memory allocation patterns
- Integration with real-time audio frameworks

**Core Types**:

- `RealtimeProcessor`: Non-blocking audio processing
- `AudioBuffer`: Circular buffer for streaming
- `LatencyMeasurement`: Performance monitoring
- `ProcessingCallback`: User-defined processing functions

**Design Patterns**:

- No heap allocation in audio callback
- Pre-allocated buffer pools
- Atomic operations for thread communication
- Graceful degradation under load

## API Design Contracts

### Error Handling Strategy

1. **Simple operations never fail**: Basic getters return values directly
2. **Complex operations return Results**: Type conversions, processing operations
3. **Rich error context**: Specific error types with detailed messages
4. **Graceful degradation**: Optional features disabled cleanly

### Memory Management

1. **Zero-allocation views**: Use `ArrayView` for read-only access
2. **In-place operations**: Prefer modification over copying
3. **Owned data when needed**: Automatic conversion from borrowed to owned
4. **Memory safety**: Rust's ownership system prevents data races

### Type System Guarantees

1. **Sample format safety**: Strong typing prevents format mix-ups
2. **Lifetime correctness**: Borrowed data cannot outlive its source
3. **Feature consistency**: Trait bounds enforce feature requirements
4. **Conversion validity**: Type-safe conversions with error handling

### Performance Characteristics

1. **Predictable allocation**: Clearly documented allocation behavior
2. **SIMD optimization**: Automatic vectorization where beneficial
3. **Cache efficiency**: ndarray layouts optimized for memory access
4. **Scalable algorithms**: Linear complexity for core operations

This architecture provides a foundation for high-performance, type-safe audio processing while maintaining ergonomic APIs and flexible feature composition.
