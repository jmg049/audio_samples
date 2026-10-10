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

//! Audio pipelines typically require many small numeric helpers - converting frequencies,
//! detecting silence, generating test tones - that do not belong to the core sample-buffer
//! API. This module keeps those helpers separate from the `AudioSamples` interface.

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
pub(crate) mod fast_math;
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
/// - `seconds` - Duration in seconds.
/// - `sample_rate` - Sampling frequency in Hz. Accepts any type that converts into `u32`,
///   including `u32` and `NonZeroU32`.
///
/// # Returns
///
/// The number of samples corresponding to the given duration at the specified sample rate.
/// The result is truncated (not rounded) to the nearest integer sample.
///
/// # Rounding
///
/// This function **truncates** toward zero, which matches librosa's `time_to_samples`
/// for non-negative times. The similarly named
/// [`audio_math::seconds_to_samples`]
/// **rounds** to the nearest sample instead, so the two can differ by one sample. For
/// librosa-compatible conversion of possibly negative times, use
/// [`audio_math::time_to_samples`].
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
/// - `num_samples` - The number of samples.
/// - `sample_rate` - Sampling frequency in Hz. Accepts any type that converts into `u32`,
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

/// Largest number of elements of a signal-sized buffer whose size is derived
/// from user parameters (2^31, 16 GiB of `f64`): output signals, click tracks,
/// frame matrices.
#[allow(dead_code)] // Users are behind optional features.
pub(crate) const MAX_SIGNAL_ELEMENTS: f64 = 2_147_483_648.0;

/// Largest number of elements of an auxiliary table whose size depends only
/// on parameters, not on the data (2^28, 2 GiB of `f64`): histograms, HMM
/// transition matrices, codebooks.
#[allow(dead_code)] // Users are behind optional features.
pub(crate) const MAX_TABLE_ELEMENTS: f64 = 268_435_456.0;

/// Checks a buffer length derived from user parameters before allocating it.
///
/// Extreme but type-valid parameters (a pitch resolution of 1e-9, a click at
/// 1e12 s, a stretch rate of 1e-6, ...) would otherwise abort the process on
/// an allocation failure. `n` is computed in `f64` by the caller so that the
/// computation itself cannot overflow; `limit` is [`MAX_SIGNAL_ELEMENTS`] or
/// [`MAX_TABLE_ELEMENTS`].
///
/// # Errors
/// [`crate::ParameterError`] naming `parameter` if `n` is not finite, negative
/// or above `limit`.
#[allow(dead_code)] // Users are behind optional features.
pub(crate) fn check_derived_len(
    parameter: &str,
    what: &str,
    n: f64,
    limit: f64,
) -> crate::AudioSampleResult<usize> {
    if n.is_finite() && (0.0..=limit).contains(&n) {
        Ok(n as usize)
    } else {
        Err(crate::AudioSampleError::Parameter(
            crate::ParameterError::invalid_value(
                parameter,
                format!(
                    "{what} would need {n:.3e} elements (limit {limit:.3e}); the value is too extreme"
                ),
            ),
        ))
    }
}
