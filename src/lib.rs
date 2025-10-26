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
#![allow(clippy::too_many_arguments)] // Allow functions with many parameters (very few and far between)

//! # Audio Samples
//!
//! A high-performance audio processing library for Rust with Python bindings.
//!
//! This library provides a comprehensive set of tools for working with audio data,
//! including type-safe sample format conversions, statistical analysis, and various
//! audio processing operations.
//!
//! ## Core Features
//!
//! - **Type-safe audio sample conversions** between i16, I24, i32, f32, and f64
//! - **High-performance operations** leveraging ndarray for efficient computation
//! - **Comprehensive metadata** tracking (sample rate, channels, duration)
//! - **Flexible data structures** supporting both mono and multi-channel audio
//! - **Python integration** via PyO3 bindings
//!
//! ## Example Usage
//!
//! ```rust
//! use audio_samples::AudioSamples;
//! use ndarray::array;
//!
//! // Create mono audio with sample rate
//! let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
//! let audio = AudioSamples::new_mono(data, 44100);
//!
//! assert_eq!(audio.sample_rate(), 44100);
//! assert_eq!(audio.channels(), 1);
//! assert_eq!(audio.samples_per_channel(), 5);
//! ```

mod error;

#[cfg(feature = "operations")]
pub mod operations;

pub mod conversions;
pub mod iterators;
pub mod realtime;
mod repr;
pub mod resampling;
pub mod simd_conversions;
pub mod traits;
pub mod utils;
pub mod views;

use std::fmt::Debug;

pub use crate::error::{AudioSampleError, AudioSampleResult, ChainableResult, IntoChainable};
pub use crate::iterators::{
    AudioSampleIterators, ChannelIterator, ChannelIteratorMut, FrameIterator, FrameIteratorMut,
    FrameMut, PaddingMode, WindowIterator, WindowIteratorMut, WindowMut,
};
pub use crate::operations::{
    AudioChannelOps, AudioEditing, AudioPlottingUtils, AudioProcessing, AudioSamplesOperations,
    AudioStatistics, AudioTransforms, NormalizationMethod,
};
pub use crate::repr::{AudioData, AudioSamples};
pub use crate::traits::{
    AudioSample, AudioSampleFamily, AudioTypeConversion, CastFrom, CastInto, Castable, ConvertTo,
};
pub use crate::utils::*;
pub use crate::views::{AudioDataRead, AudioView, AudioViewMut, SameLayout};

pub use i24::I24; // Re-export I24 type that has the AudioSample implementation

#[cfg(feature = "python")]
pub use i24::PyI24; // Re-export PyI24 for Python bindings

#[cfg(feature = "operations")]
pub use resampling::{resample, resample_by_ratio};

/// Array of supported audio sample data types as string identifiers
pub const SUPPORTED_DTYPES: [&str; 5] = ["i16", "I24", "i32", "f32", "f64"];
pub const LEFT: usize = 0;
pub const RIGHT: usize = 1;

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
    pub fn is_interleaved(&self) -> bool {
        matches!(self, ChannelLayout::Interleaved)
    }

    /// Returns true if the layout is non-interleaved
    pub fn is_non_interleaved(&self) -> bool {
        matches!(self, ChannelLayout::NonInterleaved)
    }
}
