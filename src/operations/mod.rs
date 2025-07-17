//! Audio processing operations and transformations.
//!
//! This module provides a comprehensive set of audio processing capabilities
//! organized into focused, composable traits. Each trait handles a specific
//! aspect of audio processing to maintain clean separation of concerns.
//!
//! ## Module Organization
//!
//! - [`traits`] - Core trait definitions
//! - [`statistics`] - Statistical analysis operations  
//! - [`processing`] - Signal processing operations
//! - [`transforms`] - FFT and spectral analysis
//! - [`editing`] - Time-domain editing operations
//! - [`channels`] - Channel manipulation operations
//! - [`conversions`] - Type conversion operations
//! - [`types`] - Supporting types and enums
//!
//! ## Quick Start
//!
//! ```rust
//! use audio_samples::{AudioSamples, operations::*};
//! use ndarray::array;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
//! let audio = AudioSamples::new_mono(data, 44100);
//!
//! // Statistical analysis
//! let peak = audio.peak();
//! let rms = audio.rms();
//!
//! // Signal processing
//! let mut audio_copy = audio.clone();
//! audio_copy.normalize(-1.0, 1.0, NormalizationMethod::MinMax)?;
//!
//! // Type conversion
//! let audio_i16 = audio.as_type::<i16>()?;
//! # Ok(())
//! # }
//! ```

// Public module declarations
pub mod traits;
pub mod types;

// Implementation modules (will be created as needed)
pub mod channels;
pub mod conversions;
pub mod editing;
pub mod processing;
pub mod statistics;
pub mod transforms;

// Re-export main traits for convenience
pub use traits::{
    AudioChannelOps, AudioEditing, AudioProcessing, AudioSamplesOperations, AudioStatistics,
    AudioTransforms, AudioTypeConversion,
};

// Re-export supporting types
pub use types::{
    ChannelConversionMethod, FadeCurve, MonoConversionMethod, NormalizationMethod,
    StereoConversionMethod, WindowType,
};
