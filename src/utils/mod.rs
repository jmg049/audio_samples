//! Utility functions and helpers for audio processing.
//!
//! This module groups convenience utilities that complement the core types and operations.
//!
//! # Module organization
//! - [`generation`]: synthesize common test signals (sine, chirp, impulse, silence, …)
//! - [`comparison`]: similarity metrics (correlation, MSE, SNR, …)
//! - [`detection`]: lightweight analysis helpers (fundamental frequency, silence regions, …)
//! - [`audio_math`]: mathematical utility functions for audio processing
//!
//! Some utilities are feature-gated:
//! - `detection::detect_sample_rate` requires `fft`.
//! - noise generators like `generation::white_noise` require `random-generation`.
//!
//! # Examples
//!
//! ## Generate a test tone
//! ```rust,no_run
//! use audio_samples::utils::generation;
//! use std::time::Duration;
//!
//! let tone = generation::sine_wave::<f32, f32>(440.0, Duration::from_secs(1), 44_100, 1.0);
//! assert_eq!(tone.sample_rate(), 44_100);
//! ```
//!
//! ## Compare two signals
//! ```rust,no_run
//! use audio_samples::utils::{comparison, generation};
//! use std::time::Duration;
//!
//! let a = generation::sine_wave::<f32, f32>(440.0, Duration::from_secs(1), 44_100, 1.0);
//! let b = generation::sine_wave::<f32, f32>(440.0, Duration::from_secs(1), 44_100, 1.0);
//! let corr: f32 = comparison::correlation::<f32, f32>(&a, &b).unwrap();
//! assert!(corr > 0.99);
//! ```
//!
//! ## Detect silence regions
//! ```rust,no_run
//! use audio_samples::utils::{detection, generation};
//! use std::time::Duration;
//!
//! let audio = generation::silence::<f32, f32>(Duration::from_millis(200), 44_100);
//! let regions = detection::detect_silence_regions::<f32, f32>(&audio, 0.001f32).unwrap();
//! assert!(!regions.is_empty());
//! ```

pub mod audio_math;
pub mod comparison;
pub mod detection;
pub mod generation;

// Re-export utility modules (not individual functions)
// Modules are already public above, no need for additional re-exports

use crate::{RealFloat, to_precision};

/// Helper function to convert seconds to samples
/// Converts time in seconds to number of samples at given sample rate
///
/// # Arguments
/// - `seconds`: Duration in seconds
/// - `sample_rate`: Sampling frequency in Hz (accepts `u32` or `NonZeroU32`)
///
/// # Returns
/// Number of samples representing the specified duration
///
/// # Panics
/// Panics if the computed sample count cannot be converted to `usize`,
/// typically when the result would overflow or is infinite/NaN.
pub fn seconds_to_samples<F: RealFloat>(seconds: F, sample_rate: impl Into<u32>) -> usize {
    let rate: u32 = sample_rate.into();
    (seconds * to_precision::<F, _>(rate))
        .to_usize()
        .expect("Invalid seconds or sample_rate")
}

/// Converts a number of samples to duration in seconds.
pub fn samples_to_seconds<F: RealFloat>(num_samples: usize, sample_rate: impl Into<u32>) -> F {
    let rate: u32 = sample_rate.into();
    to_precision::<F, _>(num_samples) / to_precision::<F, _>(rate)
}
