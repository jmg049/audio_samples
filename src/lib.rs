#![cfg_attr(feature = "simd", feature(portable_simd))]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
// Correctness and logic
#![warn(clippy::unit_cmp)] // Detects comparing unit types
#![warn(clippy::match_same_arms)]
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
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::needless_borrows_for_generic_args)]
#![allow(clippy::unsafe_derive_deserialize)]
#![allow(clippy::multiple_unsafe_ops_per_block)]
#![allow(clippy::doc_markdown)]
#![warn(clippy::exhaustive_enums)]
#![warn(clippy::exhaustive_structs)]
#![warn(clippy::missing_inline_in_public_items)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::iter_cloned_collect)]
#![warn(clippy::panic_in_result_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
// Allow functions with many parameters (very few and far between)
// #![deny(missing_docs)] // Documentation is a must for release

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
//! audio_samples = { version = "*", features = ["transforms", "plotting"] }
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
//! let audio = AudioSamples::new_mono(data, 44100).unwrap();
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
//! use audio_samples::{AudioProcessing, NormalizationConfig, sample_rate};
//! use ndarray::array;
//!
//! let data = array![0.1f32, 0.5, -0.3, 0.8, -0.2];
//! // new_mono and similar functions return Result types since the data cannot be empty
//! let audio = audio_samples::AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
//!
//! // Basic processing with method chaining
//! let audio = audio
//!     .normalize(NormalizationConfig::peak(1.0))
//!     .unwrap()
//!     .scale(0.5) // Reduce volume by half
//!     .remove_dc_offset()
//!     .unwrap();
//! # let _ = audio;
//! ```
//!
//! ### Type Conversions
//!
//! ```rust
//! use audio_samples::AudioSamples;
//! use ndarray::array;
//!
//! // Convert between sample types
//! let audio_f32 = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0], 44100).unwrap();
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
//! let audio = audio_samples::AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0], 44100).unwrap();
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
//! ## Method Chaining for Complex Processing
//!
//! For complex operations, chain methods directly:
//!
//! ```rust
//! use audio_samples::{AudioSamples, AudioProcessing, NormalizationConfig};
//! use ndarray::array;
//!
//! let data = array![0.1f32, 0.5, -0.3, 0.8, -0.2];
//! let audio = AudioSamples::new_mono(data, 44100).unwrap();
//!
//! // Chain multiple operations - methods consume and return Self
//! let audio = audio
//!     .normalize(NormalizationConfig::peak(1.0))
//!     .unwrap()
//!     .scale(0.8)
//!     .remove_dc_offset()
//!     .unwrap();
//! # let _ = audio;
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

// General todos in audio_samples:
// - Define a layout / template for all docs
// - Define a standard way of using type parameters in functions, all type info should be in where clauses and not <>. i.e. should be  where F: RealFloat
// - Improve the feature gating
// - Improve user experience for F: RealFloat constrained functions/types via convenience wrappers/type declarations
// - Outside of the core AudioSample structs there could be better use of the type system
// - Aim to remove as many expect, unwrap, unwrap_or etc. as possible. As many things as possible should be constrained at compile time and anything else (for example sample rates which use the NonZero variant) should be validated ONCE. Beyond this we should be able to assume that any parameter value is valid.
// - Plotting
// - Serialisation / Deserialisation of structs
// - Define / decide on a policy for parallel processing
// - Improve the error system, it has breadth currently, but not depth or aethetics. Rust set a new standard for error handling (i.e. nice errors that tell you exactly what went wrong, where and how to possible fix things.) which other libraries in other languages have tried to emulate.
// - Better constants and better use of them -- e.g. the LEFT, RIGHT and SUPPORTED_DTYPES are rarely used

#[macro_export]
macro_rules! nzu {
    ($rate:expr) => {{
        const RATE: usize = $rate;
        const { assert!(RATE > 0, "non zero usize must be greater than 0") };
        // SAFETY: We just asserted RATE > 0 at compile time
        unsafe { ::core::num::NonZeroUsize::new_unchecked(RATE) }
    }};
}

pub mod conversions;
pub mod iterators;
pub mod operations;
#[cfg(feature = "resampling")]
pub mod resampling;
pub mod traits;
pub mod utils;

mod error;
mod repr;

pub mod simd_conversions;

use std::fmt::Debug;

pub use crate::error::{
    AudioSampleError, AudioSampleResult, ConversionError, FeatureError, LayoutError,
    ParameterError, ProcessingError,
};
pub use crate::repr::AudioData;
pub use crate::repr::{AudioSamples, SampleType};
pub use crate::traits::{
    AudioSample, AudioTypeConversion, CastFrom, CastInto, ConvertFrom, ConvertTo, StandardSample,
};

pub use crate::iterators::{AudioSampleIterators, ChannelIterator, FrameIterator, PaddingMode};

#[cfg(feature = "editing")]
pub use crate::iterators::WindowIterator;

pub use crate::utils::generation::{
    ToneComponent, am_signal, chirp, compound_tone, cosine_wave, impulse, sawtooth_wave, silence,
    sine_wave, square_wave, triangle_wave,
};

#[cfg(feature = "channels")]
pub use crate::utils::generation::{
    multichannel_compound_tone, stereo_chirp, stereo_silence, stereo_sine_wave,
};

#[cfg(feature = "statistics")]
pub use crate::operations::AudioStatistics;

#[cfg(feature = "processing")]
pub use crate::operations::AudioProcessing;

#[cfg(any(feature = "processing", feature = "peak-picking"))]
pub use crate::operations::types::{NormalizationConfig, NormalizationMethod};

#[cfg(feature = "channels")]
pub use crate::operations::AudioChannelOps;

#[cfg(feature = "editing")]
pub use crate::operations::AudioEditing;

#[cfg(feature = "transforms")]
pub use crate::operations::AudioTransforms;

#[cfg(feature = "fixed-size-audio")]
pub use crate::repr::FixedSizeAudioSamples;

#[cfg(feature = "onset-detection")]
pub use crate::operations::traits::AudioOnsetDetection;

#[cfg(feature = "decomposition")]
pub use crate::operations::traits::AudioDecomposition;

#[cfg(feature = "dynamic-range")]
pub use crate::operations::traits::AudioDynamicRange;

#[cfg(feature = "iir-filtering")]
pub use crate::operations::traits::AudioIirFiltering;

#[cfg(feature = "parametric-eq")]
pub use crate::operations::traits::AudioParametricEq;

#[cfg(feature = "pitch-analysis")]
pub use crate::operations::traits::AudioPitchAnalysis;

#[cfg(feature = "vad")]
pub use crate::operations::traits::AudioVoiceActivityDetection;

#[cfg(feature = "random-generation")]
pub use crate::utils::generation::{brown_noise, pink_noise, white_noise};

pub use i24::{I24, PackedStruct}; // Re-export I24 type that has the AudioSample implementation

use ndarray::{Array1, Array2};
#[cfg(feature = "resampling")]
pub use resampling::{resample, resample_by_ratio};

/// Array of supported audio sample data types as string identifiers
pub const SUPPORTED_DTYPES: [&str; 6] = ["u8", "i16", "I24", "i32", "f32", "f64"];
/// Left channel index.
pub const LEFT: usize = 0;
/// Right channel index.
pub const RIGHT: usize = 1;

/// Describes how multi-channel audio data is organized in memory
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[allow(clippy::exhaustive_enums)]
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
    #[inline]
    #[must_use]
    pub const fn is_interleaved(&self) -> bool {
        matches!(self, Self::Interleaved)
    }

    /// Returns true if the layout is non-interleaved
    #[inline]
    #[must_use]
    pub const fn is_non_interleaved(&self) -> bool {
        matches!(self, Self::NonInterleaved)
    }
}

/// Enum specific for representing a result which can be either 1D or 2D.
/// Intended to be used for returning results from operations which can return a single result (mono) or a sequence of results (multichannel)
pub enum NdResult<T>
where
    T: StandardSample,
{
    Mono(Array1<T>),
    MultiChannel(Array2<T>),
}

impl<T> NdResult<T>
where
    T: StandardSample,
{
    /// Converts the result into a 1D array if it is mono, otherwise returns None.
    ///
    /// # Returns
    ///
    /// Some(1D array) if the result is mono, otherwise None.
    #[inline]
    pub fn into_array1(self) -> Option<Array1<T>> {
        match self {
            Self::Mono(arr) => Some(arr),
            Self::MultiChannel(_) => None,
        }
    }

    /// # Safety
    /// This function is unsafe because it assumes that the caller has already verified that the result is mono (1D). If the result is actually multi-channel (2D), this will panic.
    ///
    /// # Returns
    ///
    /// The 1D array if the result is mono, otherwise panics.
    ///
    /// # Panics
    ///
    /// Panics if the result is multi-channel (2D).
    #[inline]
    pub unsafe fn into_array1_unchecked(self) -> Array1<T> {
        match self {
            Self::Mono(arr) => arr,
            Self::MultiChannel(_) => panic!("Cannot convert MultiChannel to Array1"),
        }
    }

    /// Converts the result into a 2D array if it is multi-channel, otherwise returns None.
    ///
    /// # Returns
    ///
    /// Some(2D array) if the result is multi-channel, otherwise None.
    #[inline]
    pub fn into_array2(self) -> Option<Array2<T>> {
        match self {
            Self::Mono(_) => None,
            Self::MultiChannel(arr) => Some(arr),
        }
    }

    /// # Safety
    /// This function is unsafe because it assumes that the caller has already verified that the result is multi-channel (2D). If the result is actually mono (1D), this will panic.
    ///
    /// # Returns
    ///
    /// The 2D array if the result is multi-channel, otherwise panics.
    ///
    /// # Panics
    ///
    /// Panics if the result is mono (1D).
    #[inline]
    pub unsafe fn into_array2_unchecked(self) -> Array2<T> {
        match self {
            Self::Mono(_) => panic!("Cannot convert Mono to Array2"),
            Self::MultiChannel(arr) => arr,
        }
    }
}
