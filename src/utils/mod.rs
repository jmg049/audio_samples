//! Helpers that complement the core type and the operation traits: test signals,
//! unit conversions, comparison metrics and quick checks. All of them are in the
//! default build.
//!
//! - [`generation`]: test signals (sine, chirp, impulse, silence, noise, …) and
//!   [`example_clips`], named clips of realistic music and speech.
//! - [`audio_math`]: conversions between time, samples and frames, frequency, MIDI and
//!   note names, mel, decibels and cents.
//! - [`comparison`]: correlation, MSE, SNR and signal alignment.
//! - [`detection`]: silence, clipping, dynamic range, noise floor and a quick
//!   fundamental-frequency estimate.
//! - [`array_ops`]: small array operations used by feature pipelines (local extrema,
//!   event matching, interpolation, padding).
//! - [`notation`]: tuning systems and musical notation (just intonation, the
//!   Functional Just System, Hindustani and Carnatic names).
//!
//! The items of `generation`, `audio_math`, `comparison` and `detection` are also
//! re-exported here, so `utils::sine_wave` and `utils::generation::sine_wave` name the
//! same function. [`seconds_to_samples`] and [`samples_to_seconds`] are defined here.
//!
//! ```rust
//! use audio_samples::utils::{correlation, seconds_to_samples};
//! use audio_samples::{sample_rate, sine_wave};
//! use std::time::Duration;
//!
//! let a = sine_wave::<f32>(440.0, Duration::from_millis(200), sample_rate!(44100), 1.0);
//! assert_eq!(a.samples_per_channel().get(), seconds_to_samples(0.2, 44100u32));
//!
//! // A signal correlates perfectly with itself.
//! let r: f64 = correlation(&a, &a).unwrap();
//! assert!((r - 1.0).abs() < 1e-9);
//! ```

pub mod array_ops;
pub mod audio_math;
pub mod comparison;
pub mod detection;
pub mod example_clips;
pub(crate) mod fast_math;
pub mod generation;
pub mod notation;
// Used by the spectrum, feature, pitch, rhythm (`beat-tracking`) and effects
// modules; with only some of those features enabled, parts of these helpers are
// unused.
#[cfg_attr(
    not(all(
        feature = "transforms",
        feature = "feature-extraction",
        feature = "pitch-analysis",
        feature = "beat-tracking",
        feature = "effects"
    )),
    allow(dead_code)
)]
pub(crate) mod layout;
#[cfg_attr(
    not(all(
        feature = "transforms",
        feature = "feature-extraction",
        feature = "pitch-analysis",
        feature = "beat-tracking",
        feature = "effects"
    )),
    allow(dead_code)
)]
pub(crate) mod vmath;

pub use audio_math::*;
pub use comparison::*;
pub use detection::*;
pub use generation::*;

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
/// This function **truncates** toward zero. The similarly named
/// [`audio_math::seconds_to_samples`] **rounds** to the nearest sample instead, so the
/// two can differ by one sample. For conversion of possibly negative times, use
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
