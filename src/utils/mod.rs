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

pub use audio_math::*;
pub use comparison::*;
pub use detection::*;
pub use generation::*;

// Re-export utility modules (not individual functions)
// Modules are already public above, no need for additional re-exports

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
#[inline]
pub fn seconds_to_samples(seconds: f64, sample_rate: impl Into<u32>) -> usize {
    let rate: u32 = sample_rate.into();
    (seconds * f64::from(rate)) as usize
}

/// Converts a number of samples to duration in seconds.
#[inline]
pub fn samples_to_seconds(num_samples: usize, sample_rate: impl Into<u32>) -> f64 {
    let rate: u32 = sample_rate.into();
    num_samples as f64 / f64::from(rate)
}

#[allow(dead_code)]
pub(crate) fn min_max_single_pass<A: AsRef<[f64]>>(data: A) -> (f64, f64) {
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for &val in data.as_ref() {
        if val < min_val {
            min_val = val;
        }
        if val > max_val {
            max_val = val;
        }
    }
    (min_val, max_val)
}
