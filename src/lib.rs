#![cfg_attr(feature = "simd", feature(portable_simd))]
#![feature(const_type_name)]
// Correctness and logic
#![warn(clippy::unit_cmp)] // Detects comparing unit types
#![warn(clippy::match_same_arms)] // Duplicate match arms
#![warn(clippy::unreachable)] // Detects unreachable code

// Performance-focused
#![warn(clippy::inefficient_to_string)] // `format!("{}", x)` vs `x.to_string()`
#![warn(clippy::map_clone)] // Cloning inside `map()` unnecessarily
#![warn(clippy::unnecessary_to_owned)] // Detects redundant `.to_owned()` or `.clone()`
#![warn(clippy::large_stack_arrays)] // Helps avoid stack overflows
#![warn(clippy::box_collection)] // Warns on boxed `Vec`, `String`, etc.
#![warn(clippy::vec_box)] // Avoids using `Vec<Box<T>>` when unnecessary
#![warn(clippy::needless_collect)] // Avoids `.collect().iter()` chains

// Style and idiomatic Rust
#![warn(clippy::redundant_clone)] // Detects unnecessary `.clone()`
#![warn(clippy::identity_op)] // e.g., `x + 0`, `x * 1`
#![warn(clippy::needless_return)] // Avoids `return` at the end of functions
#![warn(clippy::let_unit_value)] // Avoids binding `()` to variables
#![warn(clippy::manual_map)] // Use `.map()` instead of manual `match`
#![warn(clippy::unwrap_used)] // Avoids using `unwrap()`
#![warn(clippy::panic)] // Avoids using `panic!` in production code

// Maintainability
#![warn(clippy::missing_panics_doc)] // Docs for functions that might panic
#![warn(clippy::missing_safety_doc)] // Docs for `unsafe` functions
#![warn(clippy::missing_const_for_fn)] // Suggests making eligible functions `const`
#![allow(clippy::too_many_arguments)]
// Allow functions with many parameters (very few and far between)
// #![deny(missing_docs)] // Documentation is a must for release

//! # AudioSamples
//!
//! A high-performance audio processing library for Rust that provides type-safe sample format conversions, statistical analysis, and various audio processing operations.
//!
//! Core building block of the wider [AudioRs](link_to_website_in_development) ecosystem.
//!
//! ## Overview
//!
//! <!-- This section is reserved for the project's purpose and motivation. -->
//! <!-- The author will fill this in later. -->
//!
//! ## Installation
//!
//! Add this to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! audio_samples = "1.0.0"
//! ```
//!
//! or more easily with:
//! ```bash
//! cargo add audio_samples
//! ```
//!
//! For specific features, enable only what you need:
//!
//! ```toml
//! [dependencies]
//! audio_samples = { version = "1.0.0", features = ["fft", "plotting"] }
//! ```
//!
//! Or enable everything:
//!
//! ```toml
//! [dependencies]
//! audio_samples = { version = "1.0.0", features = ["full"] }
//! ```
//!
//! ## Features
//!
//! The library uses a modular feature system to keep dependencies minimal:
//!
//! - **`core-ops`** (default) - Basic audio operations and statistics
//! - **`fft`** - Fast Fourier Transform and spectral analysis
//! - **`plotting`** - Audio visualization capabilities
//! - **`resampling`** - High-quality audio resampling
//! - **`parallel-processing`** - Multi-threaded processing with Rayon
//! - **`simd`** - SIMD acceleration for supported operations
//! - **`beat-detection`** - Tempo and beat tracking (requires `fft`)
//! - **`full`** - Enables all features
//!
//! ## Full Examples
//!
//! A range of examples demonstrating this crate and its companion [audio_io](https://ghithub.com/jmg049/audio_io) crate can be found at [here]().
//!
//! - [DTMF tone generation and decoding]()
//! - [Basic synthesizer]()
//! - [Silence Trimming CLI tool]()
//! - [Audio file information CLI tool]()
//!
//! ## Available Operations
//!
//! The library organizes functionality into focused traits:
//!
//! ### Core Audio Operations (`core-ops`)
//!
//! - **Statistics** (`AudioStatistics`) - Peak, RMS, mean, variance, zero-crossings, autocorrelation
//! - **Processing** (`AudioProcessing`) - Normalize, scale, clip, filtering, compression, DC removal
//! - **Channel Operations** (`AudioChannelOps`) - Mono/stereo conversion, channel extraction, pan, balance
//! - **Editing** (`AudioEditing`) - Trim, pad, reverse, fade, split, concatenate, mix
//!
//! ### Signal Processing Features
//!
//! - **IIR Filtering** (`AudioIirFiltering`) - Biquad filters, shelving, peaking
//! - **Parametric EQ** (`AudioParametricEq`) - Multi-band equalizer with adjustable Q
//! - **Dynamic Range** (`AudioDynamicRange`) - Compression, limiting, expansion, gating
//!
//! ### Advanced Analysis (Optional Features)
//!
//! - **Spectral Analysis** (`AudioTransforms`) - FFT, STFT, spectrogram, mel-spectrogram, MFCC, CQT
//! - **Pitch Analysis** (`AudioPitchAnalysis`) - Fundamental frequency, harmonic analysis
//! - **Beat Detection** - Tempo analysis and beat tracking
//! - **Plotting** (`AudioPlottingUtils`) - Waveform, spectrogram, and frequency domain visualization
//!
//! ## Quick Start
//!
//! ### Creating Audio Data
//!
//! ```rust
//! use audio_samples::AudioSamples;
//! use ndarray::array;
//!
//! // Create mono audio
//! let data = array![0.1f32, 0.5, -0.3, 0.8, -0.2];
//! let audio = AudioSamples::new_mono(data, 44100);
//!
//! // Create stereo audio
//! let stereo_data = array![
//!     [0.1f32, 0.5, -0.3],  // Left channel
//!     [0.8f32, -0.2, 0.4]   // Right channel
//! ];
//! let stereo_audio = AudioSamples::new_multi_channel(stereo_data, 44100);
//! ```
//!
//! ### Basic Statistics
//!
//! ```rust
//! use audio_samples::AudioStatistics;
//!
//! // Simple statistics (no Result needed)
//! let peak = audio.peak();
//! let min = audio.min_sample();
//! let max = audio.max_sample();
//! let mean = audio.mean();
//!
//! // More complex statistics (return Result)
//! let rms = audio.rms()?;
//! let variance = audio.variance()?;
//! let zero_crossings = audio.zero_crossings();
//! ```
//!
//! ### Processing Operations
//!
//! ```rust
//! use audio_samples::{AudioProcessing, NormalizationMethod};
//!
//! let mut audio = AudioSamples::new_mono(data, 44100);
//!
//! // Basic processing (in-place)
//! audio.normalize(-1.0, 1.0, NormalizationMethod::Peak)?;
//! audio.scale(0.5); // Reduce volume by half
//! audio.remove_dc_offset();
//! ```
//!
//! ### Type Conversions
//!
//! ```rust
//! // Convert between sample types
//! let audio_f32 = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0], 44100);
//! let audio_i16 = audio_f32.as_type::<i16>()?;
//! let audio_f64 = audio_f32.as_type::<f64>()?;
//! ```
//!
//! ### Iterating Over Audio Data
//!
//! ```rust
//! use audio_samples::AudioSampleIterators;
//!
//! // Iterate by frames (one sample from each channel)
//! for frame in audio.frames() {
//!     println!("Frame: {:?}", frame);
//! }
//!
//! // Iterate by channels
//! for channel in audio.channels() {
//!     println!("Channel: {:?}", channel);
//! }
//!
//! // Windowed iteration for analysis
//! for window in audio.windows(1024, 512) {
//!     // Process 1024-sample windows with 50% overlap
//!     let window_rms = window.rms()?;
//!     println!("Window RMS: {:.3}", window_rms);
//! }
//! ```
//!
//! ## Builder Pattern for Complex Processing
//!
//! For more complex operations, use the fluent builder API:
//!
//! ```rust
//! use audio_samples::{AudioSamples, NormalizationMethod};
//!
//! let mut audio = AudioSamples::new_mono(data, 44100);
//!
//! // Chain multiple operations
//! audio.processing()
//!     .normalize(-1.0, 1.0, NormalizationMethod::Peak)
//!     .scale(0.8)
//!     .remove_dc_offset()
//!     .apply()?;
//! ```
//!
//! ## Error Handling
//!
//! The library uses standard Rust error handling with `Result<T, AudioSampleError>`:
//!
//! ```rust
//! use audio_samples::{AudioSampleResult, AudioSamples};
//!
//! fn process_audio() -> AudioSampleResult<f64> {
//!     let audio = AudioSamples::new_mono(/* ... */, 44100);
//!     let rms = audio.rms()?;
//!     let variance = audio.variance()?;
//!     Ok(variance.sqrt())
//! }
//! ```
//!
//! ## Core Type System
//!
//! ### Supported Sample Types
//!
//! - `i16` - 16-bit signed integer
//! - `I24` - 24-bit signed integer (from `i24` crate)
//! - `i32` - 32-bit signed integer
//! - `f32` - 32-bit floating point
//! - `f64` - 64-bit floating point
//!
//! ### Type System Traits
//!
//! The library provides a rich trait system for working with different audio sample types:
//!
//! #### `AudioSample` Trait
//!
//! Core trait that all audio sample types implement. Provides common operations and constraints needed for audio processing.
//!
//! #### Conversion Traits
//!
//! - **`ConvertTo<T>`** - Type-safe conversions between audio sample types with proper scaling
//!   - `i16 -> f32`: Normalized to -1.0 to 1.0 range
//!   - `f32 -> i16`: Scaled and clamped to integer range
//!   - Handles bit depth differences automatically
//!
//! - **`CastFrom<T>` / `CastInto<T>`** - Direct type casting without audio-specific scaling
//!   - For computational operations where you need the raw numeric value
//!   - Example: `i16` sample `1334` casts to `f32` as `1334.0` (not normalized)
//!   - Use when you need to work with samples as regular numbers, not audio values
//!
//! ## Utility Functions
//!
//! The `utils` module provides convenient functions for common audio tasks:
//!
//! ### Signal Generation (`utils::generation`)
//!
//! - **Test Signals**: `generate_sine()`, `generate_white_noise()`, `generate_chirp()`
//! - **Complex Signals**: `generate_multi_tone()` for multiple frequencies
//! - **Calibration**: Known reference signals for testing
//!
//! ### Audio Analysis (`utils::detection`)
//!
//! - **Format Detection**: `detect_sample_rate()`, `detect_channel_layout()`
//! - **Content Analysis**: `detect_speech_activity()`, `detect_silence_ratio()`
//! - **Quality Metrics**: `estimate_dynamic_range()`
//!
//! ### Audio Comparison (`utils::comparison`)
//!
//! - **Similarity**: `compute_similarity()`, `cross_correlate()`
//! - **Quality**: `signal_to_noise_ratio()`, `total_harmonic_distortion()`
//! - **Perceptual**: `perceptual_audio_distance()`
//!
//! ```rust
//! use audio_samples::utils::*;
//!
//! // Generate test signal
//! let test_tone = generation::generate_sine(440.0, 44100, 1.0);
//!
//! // Analyze audio content
//! let has_speech = detection::detect_speech_activity(&audio)?;
//! let snr = comparison::signal_to_noise_ratio(&signal, &noise)?;
//! ```
//!
//! ## Development
//!
//! ### Building
//!
//! ```bash
//! cargo build                    # Core features only
//! cargo build --features full    # All features
//! cargo build --features fft     # Specific features
//! ```
//!
//! ### Testing
//!
//! ```bash
//! cargo test                     # Core tests
//! cargo test --features full     # All tests
//! cargo clippy                   # Linting
//! cargo fmt                      # Code formatting
//! ```
//!
//! ### Examples
//!
//! ```bash
//! cargo run --example iterators_demo
//! cargo run --example processing_builder_demo --features full
//! cargo run --example beat_tracking_with_progress --features full
//! ```
//!
//! ## Documentation
//!
//! Full API documentation is available at [docs.rs/audio_samples](https://docs.rs/audio_samples).
//!
//! ## License
//!
//! MIT License
//!
//! ## Contributing
//!
//! Contributions are welcome! Please feel free to submit a Pull Request.
//!
mod error;

#[cfg(feature = "core-ops")]
pub mod operations;

pub mod conversions;
pub mod iterators;
// pub mod realtime;
mod repr;
#[cfg(feature = "resampling")]
pub mod resampling;
pub mod simd_conversions;
/// Core traits for audio processing.
pub mod traits;
pub mod utils;

use std::fmt::Debug;

pub use crate::error::{AudioSampleError, AudioSampleResult};
pub use crate::iterators::{
    AudioSampleIterators, ChannelIterator, ChannelIteratorMut, FrameIterator, FrameIteratorMut,
    FrameMut, PaddingMode, WindowIterator, WindowIteratorMut, WindowMut,
};
#[cfg(feature = "core-ops")]
pub use crate::operations::{
    AudioChannelOps, AudioEditing, AudioProcessing, AudioSamplesOperations, AudioStatistics,
    NormalizationMethod,
};

#[cfg(feature = "spectral-analysis")]
pub use crate::operations::AudioTransforms;

#[cfg(feature = "plotting")]
pub use crate::operations::AudioPlottingUtils;
pub use crate::repr::{AudioData, AudioSamples, StereoAudioSamples};
pub use crate::traits::{
    AudioSample, AudioSampleFamily, AudioTypeConversion, CastFrom, CastInto, Castable, ConvertTo,
};
pub use crate::utils::*;

pub use i24::I24; // Re-export I24 type that has the AudioSample implementation

use num_traits::{Float, FloatConst, NumCast};
#[cfg(feature = "resampling")]
pub use resampling::{resample, resample_by_ratio};

/// Array of supported audio sample data types as string identifiers
pub const SUPPORTED_DTYPES: [&str; 5] = ["i16", "I24", "i32", "f32", "f64"];
/// Left channel index.
pub const LEFT: usize = 0;
/// Right channel index.
pub const RIGHT: usize = 1;

/// Marker trait for real floating-point types (f32, f64)
pub trait RealFloat: Float + FloatConst + NumCast + AudioSample {}
impl RealFloat for f32 {}
impl RealFloat for f64 {}

/// Describes how multi-channel audio data is organized in memory
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ChannelLayout {
    /// Samples from different channels are stored sequentially (LRLRLR...)
    /// This is the most common format for audio files and streaming
    #[default]
    Interleaved,
    /// Samples from each channel are stored in separate contiguous blocks (LLL...RRR...)
    /// This format is often preferred for digital signal processing
    NonInterleaved,
}

impl ChannelLayout {
    /// Returns true if the layout is interleaved
    pub const fn is_interleaved(&self) -> bool {
        matches!(self, ChannelLayout::Interleaved)
    }

    /// Returns true if the layout is non-interleaved
    pub const fn is_non_interleaved(&self) -> bool {
        matches!(self, ChannelLayout::NonInterleaved)
    }
}

/// Casts a numeric value into the target floating-point type `F`.
///
/// This function provides a *transparent* conversion mechanism
/// for any numeric type (`T`) into a chosen floating-point type (`F`),
/// typically `f32` or `f64`. It behaves identically to the built-in
/// Rust `as` cast between those types, following the same IEEE-754
/// rounding, overflow, and infinity rules.
///
/// The main purpose is to **abstract over floating-point precision**
/// in generic code where the target type `F: RealFloat` may vary.
/// This enables you to write a single numeric implementation that
/// automatically adapts to either `f32` or `f64` precision without
/// explicit `as` conversions.
///
/// # Arguments
/// * `value` - The numeric value to convert to the target floating-point type
///
/// # Returns
/// The input value converted to the target floating-point type `F`.
///
/// # Behaviour
/// - Performs the same operation as `value as F` for primitive numeric types.
/// - Rounds ties to even, consistent with IEEE-754.
/// - On overflow, produces `∞` with the same sign as the input.
/// - Panics only if the cast is invalid (which should never occur
///   for standard numeric types).
///
/// In practice, if `F` and `T` are the same type (e.g. `f32 → f32`),
/// this operation is a **compile-time no-op** with no runtime overhead.
///
/// # Examples
/// ```
/// use audio_samples::to_precision;
///
/// let value_i32 = 42i32;
/// let value_f32: f32 = to_precision(value_i32);
/// assert_eq!(value_f32, 42.0);
///
/// let value_f64: f64 = to_precision(value_i32);
/// assert_eq!(value_f64, 42.0);
/// ```
///
/// # Panics
/// Panics if the numeric conversion fails, which should never occur
/// for standard numeric types but is theoretically possible for invalid values.
#[inline(always)]
pub fn to_precision<F, T>(value: T) -> F
where
    F: RealFloat + NumCast,
    T: NumCast,
{
    NumCast::from(value).expect("safe_cast: valid numeric conversion")
}
