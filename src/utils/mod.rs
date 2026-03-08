//! Utility functions and helpers for audio processing.
//!
//! This module groups convenience utilities that complement the core types and operations.

//! Four sub-modules expose signal generation, signal comparison, audio analysis, and
//! mathematical building blocks respectively:
//!
//! - [`generation`]: synthesize common test signals (sine, chirp, impulse, silence, …)
//! - [`comparison`]: similarity metrics (correlation, MSE, SNR, signal alignment)
//! - [`detection`]: lightweight analysis helpers (fundamental frequency, silence regions, …)
//! - [`audio_math`]: domain-level conversions between frequency, amplitude, MIDI, mel, and time
//!
//! All sub-modules are re-exported at the `utils` level for convenience. Top-level helpers
//! for sample-to-time conversion are also provided directly.

//! Audio pipelines typically require many small numeric helpers — converting frequencies,
//! detecting silence, generating test tones — that do not belong to the core sample-buffer
//! API. This module isolates those helpers so they remain composable and easy to discover
//! without polluting the primary `AudioSamples` interface.

//! Import individual functions by path or use the module prefix directly:
//!
//! ```rust
//! use audio_samples::utils::generation;
//! use audio_samples::sample_rate;
//! use std::time::Duration;
//!
//! // Generate a one-second 440 Hz sine wave at 44 100 Hz sample rate.
//! let tone = generation::sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(44100), 1.0);
//! assert_eq!(tone.sample_rate().get(), 44100);
//! ```
//!
//! Some utilities are feature-gated:
//! - `detection::detect_sample_rate` requires `feature = "transforms"`.
//! - Noise generators (`white_noise`, `pink_noise`, `brown_noise`) require `feature = "random-generation"`.
//!
//! # Examples
//!
//! ## Compare two identical signals
//! ```rust
//! use audio_samples::utils::{comparison, generation};
//! use audio_samples::sample_rate;
//! use std::time::Duration;
//!
//! let sr = sample_rate!(44100);
//! let a = generation::sine_wave::<f32>(440.0, Duration::from_secs(1), sr, 1.0);
//! let b = generation::sine_wave::<f32>(440.0, Duration::from_secs(1), sr, 1.0);
//! let corr: f64 = comparison::correlation(&a, &b).unwrap();
//! assert!(corr > 0.99);
//! ```
//!
//! ## Detect silence regions
//! ```rust
//! use audio_samples::utils::{detection, generation};
//! use audio_samples::sample_rate;
//! use std::time::Duration;
//!
//! let sr = sample_rate!(44100);
//! let audio = generation::silence::<f32>(Duration::from_millis(200), sr);
//! let regions = detection::detect_silence_regions(&audio, 0.001f32).unwrap();
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

/// Converts a duration expressed in seconds into a sample count.
///
/// # Arguments
///
/// - `seconds` – Duration in seconds.
/// - `sample_rate` – Sampling frequency in Hz. Accepts any type that converts into `u32`,
///   including `u32` and `NonZeroU32`.
///
/// # Returns
///
/// The number of samples corresponding to the given duration at the specified sample rate.
/// The result is truncated (not rounded) to the nearest integer sample.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::seconds_to_samples;
/// use audio_samples::sample_rate;
///
/// let n = seconds_to_samples(1.0, sample_rate!(44100));
/// assert_eq!(n, 44100);
/// ```
#[inline]
pub fn seconds_to_samples(seconds: f64, sample_rate: impl Into<u32>) -> usize {
    let rate: u32 = sample_rate.into();
    (seconds * f64::from(rate)) as usize
}

/// Converts a sample count into a duration expressed in seconds.
///
/// # Arguments
///
/// - `num_samples` – The number of samples.
/// - `sample_rate` – Sampling frequency in Hz. Accepts any type that converts into `u32`,
///   including `u32` and `NonZeroU32`.
///
/// # Returns
///
/// The duration in seconds corresponding to the given number of samples at the specified
/// sample rate.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::samples_to_seconds;
/// use audio_samples::sample_rate;
///
/// let t = samples_to_seconds(44100, sample_rate!(44100));
/// assert!((t - 1.0).abs() < 1e-9);
/// ```
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
