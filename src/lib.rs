#![cfg_attr(feature = "simd", feature(portable_simd))]
// Correctness and logic
#![warn(clippy::unit_cmp)] // Detects comparing unit types
#![warn(clippy::match_same_arms)]
// Duplicate match arms
// #![warn(clippy::unreachable)] // Detects unreachable code

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

// Maintainability
#![warn(clippy::missing_panics_doc)] // Docs for functions that might panic
#![warn(clippy::missing_safety_doc)] // Docs for `unsafe` functions
#![warn(clippy::missing_const_for_fn)] // Suggests making eligible functions `const`
#![allow(clippy::too_many_arguments)]
// Allow functions with many parameters (very few and far between)
#![deny(missing_docs)] // Documentation is a must for release

//! # AudioSamples
//!
//! A high-performance audio processing library for Rust that provides type-safe sample format conversions, statistical analysis, and various audio processing operations.
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
//! audio_samples = "0.10.0"
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
//! audio_samples = { version = "*", features = ["fft", "plotting"] }
//! ```
//!
//! Or enable everything:
//!
//! ```toml
//! [dependencies]
//! audio_samples = { version = "*", features = ["full"] }
//! ```
//!
//! ## Features
//!
//! The library uses a modular feature system to keep dependencies minimal:
//!
//! - `statistics`: statistics utilities (peak, RMS, variance, etc.)
//! - `processing`: processing operations (normalize, scale, clip, etc.)
//! - `editing`: time-domain editing operations (trim, pad, reverse, etc.)
//! - `channels`: channel operations (mono/stereo conversion)
//! - `fft`: FFT-backed transforms (adds FFT dependencies)
//! - `plotting`: plotting utilities (adds signal plotting dependencies)
//! - `resampling`: resampling utilities (using `rubato` crate)
//! - `serialization`: serialization utilities (using `serde` crate)
//!
//! See `Cargo.toml` for the complete feature list and feature groups.
//!
//!
//! ## Error Handling
//!
//! The library uses a hierarchical error system designed for precise error handling:
//!
//! ```rust
//! use audio_samples::{AudioSampleError, AudioSampleResult, ParameterError};
//!
//! let audio_result: AudioSampleResult<()> = Err(AudioSampleError::Parameter(
//!     ParameterError::invalid_value("window_size", "must be > 0"),
//! ));
//!
//! match audio_result {
//!     Ok(()) => {}
//!     Err(AudioSampleError::Conversion(err)) => eprintln!("Conversion failed: {err}"),
//!     Err(AudioSampleError::Parameter(err)) => eprintln!("Invalid parameter: {err}"),
//!     Err(other_err) => eprintln!("Other error: {other_err}"),
//! }
//! ```
//!
//! ## Full Examples
//!
//! Examples live in the repository (see `examples/`) and in the crate-level docs on
//! <https://docs.rs/audio_samples>.
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
//! use audio_samples::{AudioStatistics, sine_wave};
//! use std::time::Duration;
//!
//! // frequency, sample rate, duration, amplitude
//! // <f32, f32> indicates the sine wave will be
//! // generated using single float precision and
//! // the resulting samples will also be f32
//! let audio = sine_wave::<f32, f32>(440.0, Duration::from_secs_f32(1.0), 44100, 1.0);
//! // Simple statistics
//! let peak = audio.peak();
//! let min = audio.min_sample();
//! let max = audio.max_sample();
//! let mean = audio.mean();
//!
//! // More complex statistics (return Result)
//! let rms = audio.rms().unwrap();
//! let variance = audio.variance().unwrap();
//! let zero_crossings = audio.zero_crossings();
//! ```
//!
//! ### Processing Operations
//!
//! ```rust
//! use audio_samples::{AudioProcessing, NormalizationMethod};
//! use ndarray::array;
//!
//! let data = array![0.1f32, 0.5, -0.3, 0.8, -0.2];
//! let mut audio = audio_samples::AudioSamples::new_mono(data, 44100);
//!
//! // Basic processing (in-place)
//! audio
//!     .normalize(-1.0, 1.0, NormalizationMethod::Peak)
//!     .unwrap();
//! audio.scale(0.5); // Reduce volume by half
//! audio.remove_dc_offset();
//! ```
//!
//! ### Type Conversions
//!
//! ```rust
//! use audio_samples::AudioSamples;
//! use ndarray::array;
//!
//! // Convert between sample types
//! let audio_f32 = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0], 44100);
//! let audio_i16 = audio_f32.as_type::<i16>().unwrap();
//! let audio_f64 = audio_f32.as_type::<f64>().unwrap();
//! ```
//!
//! ### Iterating Over Audio Data
//!
//! ```rust
//! use audio_samples::AudioSampleIterators;
//! use ndarray::array;
//!
//! let audio = audio_samples::AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0], 44100);
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
//!     let window_rms = window.rms().unwrap();
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
//! use ndarray::array;
//!
//! let data = array![0.1f32, 0.5, -0.3, 0.8, -0.2];
//! let mut audio = AudioSamples::new_mono(data, 44100);
//!
//! // Chain multiple operations
//! audio.processing()
//!     .normalize(-1.0, 1.0, NormalizationMethod::Peak)
//!     .scale(0.8)
//!     .remove_dc_offset()
//!     .apply()
//!     .unwrap();
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
//!

//! ```rust
//! use audio_samples::{comparison, sine_wave};
//! use std::time::Duration;
//!
//! let a = sine_wave::<f32, f32>(440.0, Duration::from_secs(1), 44100, 1.0);
//! let b = sine_wave::<f32, f32>(440.0, Duration::from_secs(1), 44100, 1.0);
//! let corr: f32 = comparison::correlation::<f32, f32>(&a, &b).unwrap();
//! assert!(corr > 0.99);
//! ```
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

mod error;

#[cfg(any(
    feature = "core-ops",
    feature = "statistics",
    feature = "processing",
    feature = "editing",
    feature = "channels"
))]
#[cfg(any(
    feature = "core-ops",
    feature = "statistics",
    feature = "processing",
    feature = "editing",
    feature = "channels"
))]
pub mod operations;

pub mod conversions;
pub mod iterators;
mod repr;
#[cfg(feature = "resampling")]
pub mod resampling;
pub mod simd_conversions;
/// Core traits for audio processing.
pub mod traits;
pub mod utils;

use std::fmt::Debug;

pub use crate::error::{
    AudioSampleError, AudioSampleResult, ConversionError, FeatureError, LayoutError,
    ParameterError, ProcessingError,
};
pub use crate::iterators::{
    AudioSampleIterators, ChannelIterator, ChannelIteratorMut, FrameIterator, FrameIteratorMut,
    FrameMut, PaddingMode, WindowIterator, WindowIteratorMut, WindowMut,
};
#[cfg(feature = "statistics")]
pub use crate::operations::AudioStatistics;

#[cfg(feature = "processing")]
pub use crate::operations::{AudioProcessing, NormalizationMethod};

#[cfg(feature = "editing")]
pub use crate::operations::AudioEditing;

#[cfg(feature = "channels")]
pub use crate::operations::AudioChannelOps;

#[cfg(feature = "spectral-analysis")]
pub use crate::operations::AudioTransforms;

#[cfg(feature = "plotting")]
pub use crate::operations::AudioPlottingUtils;
#[cfg(feature = "fixed-size-audio")]
pub use crate::repr::FixedSizeAudioSamples;
pub use crate::repr::{AudioBytes, AudioData, AudioSamples, SampleType, StereoAudioSamples};
pub use crate::traits::{
    AudioSample, AudioTypeConversion, CastFrom, CastInto, Castable, ConvertTo,
};
pub use crate::utils::{
    audio_math::{
        amplitude_to_db, db_to_amplitude, fft_frequencies, frames_to_time, hz_to_mel, hz_to_midi,
        mel_to_hz, mel_scale, midi_to_hz, midi_to_note, note_to_midi, power_to_db, time_to_frames,
    },
    comparison, detection,
    generation::{
        ToneComponent, chirp, compound_tone, cosine_wave, impulse, multichannel_compound_tone,
        sawtooth_wave, silence, sine_wave, square_wave, stereo_chirp, stereo_silence,
        stereo_sine_wave, triangle_wave,
    },
    samples_to_seconds, seconds_to_samples,
};

// Re-export noise generation functions with feature gating
#[cfg(feature = "random-generation")]
pub use crate::utils::generation::{brown_noise, pink_noise, white_noise};

pub use i24::{I24, PackedStruct}; // Re-export I24 type that has the AudioSample implementation

// Re-export NonZero types used in the API
pub use core::num::{NonZeroU32, NonZeroUsize};

use num_traits::{Float, FloatConst, NumCast};
#[cfg(feature = "resampling")]
pub use resampling::{resample, resample_by_ratio};
#[cfg(feature = "resampling")]
use rubato::Sample;

/// Array of supported audio sample data types as string identifiers
pub const SUPPORTED_DTYPES: [&str; 5] = ["i16", "I24", "i32", "f32", "f64"];
/// Left channel index.
pub const LEFT: usize = 0;
/// Right channel index.
pub const RIGHT: usize = 1;

#[cfg(not(feature = "resampling"))]
/// Marker trait for real floating-point types (f32, f64)
pub trait RealFloat: Float + FloatConst + NumCast + AudioSample {}

#[cfg(feature = "resampling")]
/// Marker trait for real floating-point types (f32, f64)
pub trait RealFloat: Float + FloatConst + NumCast + AudioSample + Sample {}

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
/// This function provides a *transparent* conversion mechanism for numeric
/// values (`T`) into a chosen target type (`F`), typically `f32` or `f64`.
///
/// Internally it uses `num_traits::NumCast::from` and will **panic** if the
/// cast is not representable by the target type (e.g. out-of-range values,
/// or non-finite floats when converting to an integer type).
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
/// - Uses `NumCast::from(value)`.
/// - Panics if the conversion fails.
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
/// Panics if the numeric conversion fails.
#[inline(always)]
pub fn to_precision<F, T>(value: T) -> F
where
    F: RealFloat + NumCast,
    T: NumCast,
{
    NumCast::from(value).expect("safe_cast: valid numeric conversion")
}
