# AudioSamples Architecture

This document provides detailed information on how the audio_samples crate is architected from its building blocks up to the highest-level API.
The architecture follows core principles of **type safety**, **zero-allocation efficiency**, **trait-based composition**, and **modular feature design**.

## Core Design Principles

### 1. Type Safety Through Strong Typing

All audio data is strongly typed with the sample format (`i16`, `I24`, `i32`, `f32`, `f64`), ensuring mathematical operations are performed with appropriate precision and range.
The type system prevents common audio processing errors like mixing incompatible sample formats.

### 2. Zero-Allocation Efficiency

The library leverages `ndarray`'s view system to enable zero-allocation access patterns wherever possible.
Operations prefer in-place modifications and views over copying data.
The library adds several wrappers, ``MonoRepr/MultiRepr`` and ``MonoData/MultiData``, around `ndarray` arrays to facilitate owned and borrowed data automatically.
The end user should never have to deal with these wrappers.
The top layer of the wrappers is the ``AudioSamples<'_, T: AudioSample>`` struct.
This is the entry point for a user and the impl blocks for ``AudioSamples`` connect to the right backend wrapper.

### 3. Trait-Based Composition

Functionality is organized into focused, composable traits rather than monolithic implementations.
Each trait handles a specific aspect of audio processing with clear separation of concerns.
Traits also allow for easy feature-gating.

### 4. Metadata Integration

Audio samples are always paired with essential metadata (sample rate) to prevent common audio processing errors and enable automatic format conversions.

### 5. Feature-Gated Modularity

The library uses cargo features extensively to keep dependencies minimal, allowing users to enable only the functionality they need.

### 6. Dual-Variant Operations

Every transforming operation is exposed as a matched pair. The canonical, unsuffixed name **borrows** and returns a new owned copy:

```rust,ignore
fn op(&self, ..) -> AudioSampleResult<Self>;        // returns a NEW modified value
fn op_in_place(&mut self, ..) -> AudioSampleResult<()>; // mutates the receiver
```

The `*_in_place` variant is the primitive; the borrowing variant clones the receiver and delegates to it. Because the borrowing form does not consume `self`, results compose into pipelines (`audio.normalize(cfg)?.scale(0.5).remove_dc_offset()?`) while leaving the original binding usable. Operations that cannot fail (e.g. `scale`) return `Self` directly rather than `AudioSampleResult<Self>`, so no `?` is needed.

## Building Blocks

### AudioSample/StandardSample (./traits.rs)

The `AudioSample` and `StandardSample` traits are the foundation of the entire type system.
It defines the interface for all supported audio sample formats:

**Supported Types:**

- `i16`: 16-bit signed integer samples (most common for audio files)
- `I24`: 24-bit signed integer samples (professional audio) -- **Always use the re-export of ``I24`` from the crate.**
- `i32`: 32-bit signed integer samples (high precision)
- `f32`: 32-bit floating-point samples (normalized -1.0 to 1.0)
- `f64`: 64-bit floating-point samples (highest precision)

**Key Requirements:**

- Standard arithmetic operations (`Add`, `Sub`, `Mul`, `Div`)
- Memory safety guarantees (`NoUninit` for safe byte serialization)
- Numeric operations (`Num`, `Zero`, `One`, `Signed`)
- Serialization support (`Serialize`, `Deserialize`)
- Constants for range information (`MAX`, `MIN`, `BITS`)

The `StandardSample` trait extends the `AudioSample` trait with conversion functionality provided via the `ConvertTo` and `ConvertFrom` traits.

**API Contracts:**

- All sample types must provide consistent arithmetic behavior
- Byte serialization must be safe and deterministic
- Range constants must accurately represent the format's dynamic range

### ConvertTo/ConvertFrom (./traits.rs)

The `ConvertTo<T>` and `ConvertFrom<T>` trait provides audio-aware conversions between different sample formats with proper scaling:

**Conversion Behavior:**

- **Integer ↔ Integer**: Bit-shift scaling to preserve full dynamic range
- **Integer ↔ Float**: Normalized scaling (-1.0 to 1.0 for floats)
- **Float ↔ Float**: Direct casting with precision conversion
- **I24 Special Handling**: Custom methods for 24-bit operations

**Design Patterns:**

- Returns `T`, designed not to fail.
- Uses macro-generated implementations for consistency and to cut down on manual code.
- Maintains mathematical precision across format boundaries
- Handles edge cases like range overflows gracefully
- The two traits provide functionality like `Into` and `From` in the standard library.

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

Sometimes you just need to cast an int to a float and back again.
These traits **DO NOT** perform any audio-specific scaling/conversions.

- `CastFrom<S>`: Cast from source type to Self
- `CastInto<T>`: Cast self into target type
- `Castable`: Marker trait for types that can cast to all audio formats

**API Contracts:**

- Casting preserves numeric values without audio scaling
- Out-of-range values are clamped to target type's limits
- No error handling - assumes well-formed input.

If something like this fails then things are bad.

### AudioSamples<'_, T: StandardSample> (repr.rs)

The main data container that combines audio samples with essential metadata:

```rust,ignore
pub struct AudioSamples<'a, T: StandardSample> {
    data: AudioData<'a, T>,        // private
    sample_rate: SampleRate,       // private; guaranteed non-zero
}
```

**Sealed fields and accessors:** the fields are **private**. Callers reach the underlying data and metadata through accessors:

- `data(&self) -> &AudioData<'a, T>` and `data_mut(&mut self) -> &mut AudioData<'a, T>`
- `into_data(self) -> AudioData<'static, T>` (owned) and `into_data_borrowed(self) -> AudioData<'a, T>` (preserves the borrow)
- `sample_rate(&self) -> SampleRate` (plus the convenience `sample_rate_hz() -> f64`)

Channel **indices** are plain `usize`; channel **counts** use the dedicated `ChannelCount` type, keeping the two concepts from being confused at call sites.

**Key Features:**

- Generic over any `StandardSample` type
- Lifetime parameter `'a` enables zero-copy views
- Always includes a non-zero sample rate (`SampleRate`)
- Provides uniform interface for mono and multi-channel audio

**Memory Layout:**

- Mono audio: 1D arrays via `MonoData<'a, T>`
- Multi-channel audio: 2D arrays via `MultiData<'a, T>` with channels as rows
- Both support borrowed and owned data

**API Contracts:**

- Sample rate must be positive
- Lifetime safety ensured through Rust's borrow checker.
But at the user level of the API, unless they are really concerned with lifetime management and reuse in their program, lifetimes should not be a concern.
- Metadata consistency maintained across operations

### AudioSamples Iteration (iterators.rs)

Provides multiple iteration patterns for efficient audio processing:

**Iterator Types:**

- `frames()`: Iterate by frames (one sample from each channel)
- `channels()`: Iterate by complete channels
- `windows(size, hop)`: Windowed iteration with configurable overlap (owning `WindowIterator`)
- `windows_ref(size, hop)`: Zero-copy windowed iteration that borrows each window (`WindowRefIterator`) instead of allocating
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

Implements the `AudioTypeConversion` trait for safe type transformations on ``AudioSamples``:

**In-Domain Conversions:**

- `to_format<O>()`: Borrows original, returns new type
- `to_type<O>()`: Consumes original, returns new type
- Convenience methods: `as_f32()`, `as_i16()`, `as_i24()`, etc.
- Uses `ConvertTo` and `ConvertFrom` traits for audio-aware scaling

**Out-of-Domain Conversions:**

- `cast_as<O>()`: Borrows original, raw numeric casting
- `cast_to<O>()`: Consumes original, raw numeric casting
- Uses `CastFrom` trait for direct numeric conversion

**API Contracts:**

- Clear distinction between audio-aware and raw conversions
- Lifetime management ensures memory safety
- Type bounds enforce conversion compatibility

### NdResult (lib.rs)

Some operations (for example envelope followers) produce an array whose dimensionality mirrors the input: a mono input yields a 1-D array, a multi-channel input yields a 2-D `channels × samples` array. `NdResult<T>` is the tagged return type that expresses this **1-D / 2-D by layout** distinction in a single value: it is either `Mono(Array1<T>)` or `MultiChannel(Array2<T>)`. Callers pattern-match, or use the `into_array1()` / `into_array2()` helpers when the expected shape is known. Both variants are guaranteed non-empty.

### Utilities (./utils)

Provides supporting functionality organized by purpose:

- `generation.rs`: Signal generation. Beyond the classic oscillators (`sine_wave`, `cosine_wave`, `square_wave`, `triangle_wave`, `sawtooth_wave`, `chirp`, `impulse`, `silence`, `compound_tone`, `am_signal`), this now includes exponential/log chirps (`exponential_chirp`), band-limited oscillators (`square_wave_bandlimited`, `sawtooth_wave_bandlimited`, `triangle_wave_bandlimited`), FM signals (`fm_signal`), and the noise sources `white_noise`/`pink_noise`/`brown_noise` (under `random-generation`).
- `detection.rs`: Feature detection algorithms
- `comparison.rs`: Audio comparison and similarity metrics — `correlation`, `mse`, `snr`, `psnr`, `segmental_snr`, `log_spectral_distance`, plus per-channel variants of the first three (`correlation_per_channel`, `mse_per_channel`, `snr_per_channel`).
- `audio_math.rs`: Audio-domain mathematical utilities and canonical unit conversions.
In future, this may become a more general ``algorithms`` module.

**Design Patterns:**

- Pure functions where possible
- Consistent error handling patterns
- Performance-optimized implementations
- Feature-gated advanced functionality

### Errors (./error.rs)

Hierarchical error handling. `AudioSampleError` is the `#[non_exhaustive]` root, and each variant wraps a specialised sub-enum that covers one failure domain:

```rust,ignore
pub enum AudioSampleError {
    Conversion(ConversionError),   // type conversion and casting failures
    Parameter(ParameterError),     // invalid parameters and configuration values
    Layout(LayoutError),           // memory layout / array-structure issues
    Processing(ProcessingError),   // DSP algorithm and arithmetic failures
    Feature(FeatureError),         // missing/misconfigured optional cargo features
}
```

Grouping by domain lets callers match a broad category in a single arm while still drilling into the specific cause when needed.

**Error Handling Strategy:**

- Built on `thiserror`; every error also implements [`miette::Diagnostic`].
- Each variant carries a stable, namespaced **code** (`audio_samples::<domain>::<variant>`) for programmatic matching and an actionable **help** hint.
- Text-parsing failures (note names, enum `FromStr`) carry a `SourceSpan` pointing a caret at the offending character.
- The coloured caret-underline rendering is gated behind the `fancy` feature, so library consumers pay nothing for graphical dependencies unless they opt in.
- `AudioSampleResult<T>` (= `Result<T, AudioSampleError>`) is the consistent return alias for fallible operations.

## Trait Extensions

Available:

- AudioStatistics (descriptive stats + spectral feature suite)
- AudioVoiceActivityDetection
- AudioProcessing
- AudioTransforms
- AudioPitchAnalysis (`track_pitch`, `estimate_key`)
- AudioBeatTracking (`estimate_tempo`)
- AudioOnsetDetection
- AudioIirFiltering (filter design, `filtfilt`, streaming)
- AudioParametricEq
- AudioDynamicRange
- AudioEditing
- AudioChannelOps
- AudioPerceptualAnalysis (psychoacoustic)
- AudioPlottingUtils
- AudioSamplesSerialise
- AudioDecomposition
- AudioTypeConversion

Each transforming trait method follows the dual-variant convention (`op` / `op_in_place`) described in the Core Design Principles.

### The `operations` Module (./operations)

The operations module organizes audio processing functionality into focused, composable traits.
Each trait addresses a specific domain of audio processing:

#### Core Organization

- **`traits.rs`**: Trait definitions and type bounds
- **Implementation files**: Named after traits (e.g., `AudioChannelOps` → `channels.rs`)
- **Supporting types**: Enums and structs in `types.rs`
- **Feature gating**: Advanced functionality behind cargo features

#### `AudioStatistics` (statistics.rs)

**Purpose**: Statistical analysis operations for audio data.

**Why use it?**: Essential for audio analysis, level monitoring, and processing decisions.

**How to use it?**:

```rust,ignore
let peak = audio.peak();                  // Returns T directly
let rms: f64 = audio.rms();               // Returns f64 directly (infallible)
let crossings = audio.zero_crossings();   // Signal analysis

// Spectral features fold multi-channel input via a ChannelReduction policy.
let centroid = audio.spectral_centroid(ChannelReduction::Average)?;
let flatness = audio.spectral_flatness(ChannelReduction::First)?;
```

**Key Operations**:

- `peak()`, `min_sample()`, `max_sample()`: Level measurements
- `mean()`, `variance()`, `std_dev()`: Statistical measures
- `rms()`: Perceptually relevant loudness measure
- `zero_crossings()`, `zero_crossing_rate()`: Periodicity analysis
- `autocorrelation()`, `cross_correlation()`: Correlation analysis (requires `transforms`)
- Spectral feature suite (requires `transforms`): `spectral_centroid()`, `spectral_rolloff()`, `spectral_bandwidth()`, `spectral_flatness()`, `spectral_contrast()`, `spectral_slope()`, `spectral_crest()` — each takes a `ChannelReduction` argument

**ChannelReduction policy**: spectral analysis on multi-channel audio must decide how to collapse channels to a single value. `ChannelReduction` is the parameter that chooses: `Error` (default — refuse multi-channel input), `First`, `Average`, or `Channel(usize)`.

**API Contracts**:

- Simple statistics (`peak`, `rms`, `mean`, `variance`, `zero_crossings`, …) return values **directly** — no `Result` — leveraging the non-empty invariant of `AudioSamples`.
- Generic over float types for numerical operations
- Consistent behavior across mono and multi-channel audio

#### `AudioProcessing` (processing.rs)

**Purpose**: Core signal processing operations following the dual-variant convention.

**Why use it?**: Fundamental audio modifications like normalization, scaling, DC removal.

**How to use it?**:

```rust,ignore
// Borrowing variants return a NEW value and compose into a pipeline. `scale` is
// infallible (returns Self, no `?`); the others are fallible.
let processed = audio
    .normalize(NormalizationConfig::peak(1.0))?
    .scale(0.8)
    .remove_dc_offset()?;

// In-place variants mutate the receiver instead of allocating a copy.
let mut buf = audio.clone();
buf.normalize_in_place(NormalizationConfig::peak(1.0))?;
buf.scale_in_place(0.8);
```

**Key Operations**:

- `normalize()`: Multiple normalization strategies, configured via `NormalizationConfig` (e.g. `NormalizationConfig::peak(target)`)
- `scale()`, `gain()`: Amplitude adjustments (`scale` is infallible)
- `clip()`: Hard limiting to prevent overflows
- `remove_dc_offset()`: DC bias removal
- `fade_in()`, `fade_out()`: Envelope operations

**Convention**: each operation above has both the borrowing form shown here and an `*_in_place` counterpart, as described under *Dual-Variant Operations*. Multi-parameter behaviour is captured in dedicated config structs rather than long positional argument lists.

#### `AudioTransforms` (transforms.rs)

**Purpose**: Frequency-domain analysis and transformations (requires `transforms`).

**Why use it?**: Spectral analysis, filtering, and frequency-domain processing. The heavy time–frequency machinery lives in the companion `spectrograms` crate.

**How to use it?**:

```rust,ignore
let spectrum = audio.fft(nzu!(8192))?;
let stft = audio.stft(&stft_params)?;
let mfcc = audio.mfcc(&stft_params, nzu!(40), &MfccParams::speech_standard())?;
let psd = audio.power_spectral_density(nzu!(1024), 0.5)?; // -> Psd
```

**Key Operations**:

- `fft()`, `ifft()`: Fast Fourier Transform and inverse
- `stft()`, `istft()`: Short-Time Fourier Transform (parameterised by `StftParams`)
- `mfcc()`, `chromagram()`, `constant_q_transform()`: higher-level spectral representations
- `power_spectral_density()`: returns a structured `Psd { frequencies, density }` rather than a bare tuple (access via `frequencies()` / `density()` / `into_parts()`)
- `spectral_filter()`: Frequency domain filtering

**Structured returns**: where a pre-2.0 API returned a loose tuple, the modern API returns a named type — for example `power_spectral_density` yields `Psd`, and (in the pitch-analysis trait) `track_pitch` yields `PitchContour` and `estimate_key` yields `Key { tonic, mode, confidence }`.

**Performance Considerations**:

- Uses `rustfft` by default
- Real-valued FFT optimization for audio signals
- Memory-efficient windowing operations (`windows()` / zero-copy `windows_ref()`)

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
- `stack()`: Stack audio samples on top of each other
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
- `extract_channel()`: Extract a specific channel.
allocates a new ``AudioSamples`` for the extracted channel
- `borrow_channel()`: Borrows a specific channel.
Does not allocate a new ``AudioSamples`` for the extracted channel, it just borrows the data.
- `mix_channels()`: Custom channel mixing
- `swap_channels()`: Channel reordering
- `duplicate_to_channels(N)` - Duplicates mono audio to N channels
- `pan(x)` - Applies pan control to stereo audio
- `balance(x)` - Adjusts balance between left and right channels
- `apply_to_channel(i, f)` - Apply a function to a specific channel
- `interleave_channels(c)` - Interleave multiple channels into one audio sample
- `deinterleave_channels()` - Deinterleave audio into separate channel samples
**Conversion Methods**:

- **Mono**: `Average`, `Left`, `Right`, `Sum`, `WeightedSum`
- **Stereo**: `Duplicate`, `Spread`, `Custom`

#### `AudioIirFiltering` (iir_filtering.rs)

**Purpose**: Infinite Impulse Response filter design and application.

**Why use it?**: Real-time filtering, tone shaping, frequency response control.

**How to use it?**:

```rust,ignore
// Convenience constructors per family + response, in borrowing or in-place form.
let filtered = audio.butterworth_lowpass(nzu!(4), 1_000.0)?;
audio.butterworth_lowpass_in_place(nzu!(4), 1_000.0)?;

// Full control via an IirFilterDesign struct.
let design = IirFilterDesign::chebyshev_i(FilterResponse::HighPass, nzu!(6), 2_000.0, 1.0);
let shaped = audio.apply_iir_filter(&design)?;

// Zero-phase (forward-backward) filtering: no group delay.
let flat_phase = audio.filtfilt(&design)?;

// Design once, stream many blocks (real-time / chunked input).
let mut sos = SosFilter::from_design(&design, 44_100.0)?;
sos.process_block(&mut block);
```

**Filter families** (`IirFilterType`): `Butterworth` (default), `ChebyshevI`, `ChebyshevII`, `Elliptic` (Cauer), and `Bessel`.

**Responses** (`FilterResponse`): `LowPass`, `HighPass`, `BandPass`, `BandStop`.

**Key capabilities**:

- `IirFilterDesign`: a single struct describing family + response + order + cut-off(s) + ripple/attenuation, built with constructors such as `butterworth_lowpass`, `chebyshev_i`, `chebyshev_ii`, `bessel`, `elliptic`.
- `apply_iir_filter()` / `apply_iir_filter_in_place()`: apply a design.
- `filtfilt()` / `filtfilt_in_place()`: zero-phase forward-backward filtering (SciPy `sosfiltfilt`-style), doubling the magnitude response while cancelling group delay.
- `SosFilter::from_design(..)` + `process_block(&mut [f64])`: a design-once, second-order-sections filter that carries state across calls for streaming use.
- `frequency_response()`: magnitude/phase of the configured filter at probe frequencies.

#### `AudioParametricEq` (parametric_eq.rs)

**Purpose**: Multi-band parametric equalization.

**Why use it?**: Tone shaping, frequency response correction, creative EQ.

**How to use it?**:

```rust,ignore
let mut eq = ParametricEq::new();
eq.add_band(EqBand::peak(1000.0, 3.0, 2.0));  // +3 dB at 1 kHz, Q = 2.0
eq.add_band(EqBand::peak(5000.0, -6.0, 1.0)); // -6 dB at 5 kHz, Q = 1.0

let equalized = audio.apply_parametric_eq(&eq)?;          // borrowing variant
// audio.apply_parametric_eq_in_place(&eq)?;              // in-place variant

// Or the fixed three-band shelf/peak EQ via a config struct:
let config = ThreeBandEqConfig::new(200.0, 2.0, 1000.0, -1.0, 1.0, 5000.0, 3.0);
let shaped = audio.apply_three_band_eq(&config)?;
```

**Key Features**:

- Multiple simultaneous frequency bands (`ParametricEq` / `EqBand`)
- A convenience `ThreeBandEqConfig` for the common low/mid/high case
- Independent gain, frequency, and Q controls
- Both borrowing and `*_in_place` application variants
- Efficient cascaded biquad implementation

#### `AudioDynamicRange` (dynamic_range.rs)

**Purpose**: Dynamic range processing (compression, limiting, expansion).

**Why use it?**: Level control, dynamics processing, mastering operations.

**How to use it?**: each processor is driven by a dedicated config struct.

```rust,ignore
// CompressorConfig::new() yields a sensible default; presets like vocal()/drum()/
// bus() and field setters tailor it further.
let compressed = audio.compressor(&CompressorConfig::vocal())?;

// GateConfig / ExpanderConfig expose with_params(threshold, ratio, attack, release).
let gated = audio.gate(&GateConfig::with_params(-40.0, 4.0, 1.0, 100.0))?;

// LimiterConfig / ThreeBandEqConfig take explicit positional parameters.
let limited = audio.limiter(&LimiterConfig::default())?;
```

**Configuration structs**: `CompressorConfig`, `LimiterConfig`, `GateConfig`, and `ExpanderConfig` replace long positional argument lists. Each exposes a `new()`/`default()` baseline plus either presets or a `with_params(..)` helper, and side-chain behaviour is described by `SideChainConfig`.

**Processor Types** (each with a borrowing and an `*_in_place` variant):

- `compressor()`: Reduce dynamic range above threshold
- `limiter()`: Hard limiting to prevent peaks
- `expander()`: Increase dynamic range below threshold
- `gate()`: Remove low-level noise

#### Audio Resampling (resampling.rs)

**Purpose**: High-quality sample rate conversion (requires `resampling`).

**Why use it?**: Sample rate conversion, format compatibility, anti-aliasing.

**How to use it?**:

```rust,ignore
let resampled = audio.resample(48000, ResamplingQuality::High)?;
```

**Quality Levels**:

- `Fast`: Quick conversion with acceptable quality
- `Medium`: Balanced quality and performance
- `High`: High-quality anti-aliasing

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
- `PhaseSpectrum`: Frequency-domain phase plots
- `Lissajous`: Stereo X-Y / Lissajous (phase-correlation) plots
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

## API Design Contracts

### Error Handling Strategy

1. **Simple operations never fail**: Basic getters return values directly
2. **Complex operations return Results**: Type conversions, processing operations
3. **Rich error context**: Specific error types with detailed messages
4. **Graceful degradation**: Optional features disabled cleanly

### Memory Management

1. **Zero-allocation views**: Borrow AudioSamples when possible
2. **In-place operations**: Prefer modification over copying
3. **Owned data when needed**: Automatic conversion from borrowed to owned
4. **Memory safety**: Rust's ownership system prevents data races

### Type System Guarantees

1 .**Sample format safety**: Strong typing prevents format mix-ups
2. **Lifetime correctness**: Borrowed data cannot outlive its source
3. **Feature consistency**: Trait bounds enforce feature requirements
4. **Conversion validity**: Type-safe conversions with error handling

### Performance Characteristics

1. **Predictable allocation**: Clearly documented allocation behavior
2. **SIMD optimization**: Automatic vectorization where beneficial
3. **Cache efficiency**: ndarray layouts optimized for memory access
4. **Scalable algorithms**: Linear complexity for core operations
