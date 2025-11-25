//! Minimal API for basic audio processing.
//!
//! This module provides a streamlined API focused on the most common audio operations
//! with minimal dependencies. It's designed for users who need basic audio processing
//! without the overhead of advanced features.
//!
//! ## Features
//!
//! - Basic statistics: peak, RMS, mean
//! - Essential processing: normalize, scale, clip
//! - Sample type conversions
//!
//! ## Usage
//!
//! Enable this API with:
//! ```toml
//! [dependencies]
//! audio_samples = { version = "0.10", features = ["statistics", "processing"] }
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use audio_samples::minimal::*;
//! use ndarray::array;
//!
//! let data = array![0.5f32, -0.3, 0.8, -0.1];
//! let mut audio = AudioSamples::new_mono(data, 44100);
//!
//! // Basic analysis
//! let peak = audio.peak();
//! let rms = audio.rms()?;
//!
//! // Basic processing
//! audio.normalize(-1.0, 1.0, NormalizationMethod::Peak)?;
//! audio.scale(0.8);
//! ```

// Re-export core types
pub use crate::{AudioSampleError, AudioSampleResult, AudioSamples};

// Re-export sample types
pub use crate::{AudioSample, ConvertTo, I24};

// Re-export basic operations
#[cfg(feature = "statistics")]
pub use crate::AudioStatistics;

#[cfg(feature = "processing")]
pub use crate::{AudioProcessing, NormalizationMethod};

// Re-export essential utilities
pub use crate::utils::{detection, generation};
