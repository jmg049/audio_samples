//! Low-level DSP kernels for onset detection.
//!
//! This module provides optimized, low-level building blocks for onset detection algorithms.
//! These functions operate directly on spectrograms and detection functions to compute onset
//! features and apply adaptive thresholding.
//!
//! The kernels include:
//! - The "energy" onset detection function (half-wave-rectified magnitude flux)
//! - Adaptive thresholding using median-based local thresholds
//!
//! These functions are typically used internally by higher-level onset detection implementations
//! and are optimized for performance.

use ndarray::Array2;
use non_empty_slice::{NonEmptySlice, NonEmptyVec};

/// Computes the "energy" onset detection function from a magnitude spectrogram.
///
/// For each frame, sums the positive frame-to-frame differences of the values in `mag`
/// across all frequency bins. No squaring is done: with a magnitude spectrogram (as passed
/// by [`AudioOnsetDetection::onset_detection_function`](crate::operations::traits::AudioOnsetDetection::onset_detection_function))
/// this is the half-wave-rectified magnitude flux; pass a power spectrogram to obtain a
/// true energy flux. Peaks correspond to sudden increases in spectral content.
///
/// Formula: `odf[t] = Σ_b max(0, mag[b,t] − mag[b,t−1])`, `odf[0] = 0`.
///
/// # Arguments
///
/// * `mag` - 2D magnitude spectrogram with shape `(frequency_bins, frames)`
///
/// # Returns
///
/// Onset detection function values for each frame. The first frame is always 0.0.
///
/// # Example
///
/// ```rust,ignore
/// use ndarray::Array2;
/// use audio_samples::operations::onset::kernels::energy_odf;
///
/// let mag = Array2::from_shape_vec((128, 100), vec![0.0; 12800]).unwrap();
/// let odf = energy_odf(&mag);
/// assert_eq!(odf.len().get(), 100);
/// assert_eq!(odf[0], 0.0);
/// ```
#[inline]
#[must_use]
pub fn energy_odf(mag: &Array2<f64>) -> NonEmptyVec<f64> {
    let (bins, frames) = mag.dim();
    let mut odf = Vec::with_capacity(frames);
    odf.push(0.0);

    for t in 1..frames {
        let mut acc = 0.0;
        for b in 0..bins {
            let diff = mag[[b, t]] - mag[[b, t - 1]];
            if diff > 0.0 {
                acc += diff;
            }
        }
        odf.push(acc);
    }
    // safety: frames > 1 ensures odf is non-empty
    unsafe { NonEmptyVec::new_unchecked(odf) }
}

/// Applies adaptive thresholding to a signal in-place using a median-based local threshold.
///
/// For each sample, if its value is below `median[i] * multiplier`, it is set to zero.
/// Otherwise, it remains unchanged (the median is only a reference level; values above it
/// are not reduced). This technique removes low-amplitude fluctuations while
/// preserving strong peaks, improving onset detection robustness against varying background
/// energy levels.
///
/// # Arguments
///
/// * `signal` - Signal to threshold (modified in-place)
/// * `median` - Median-filtered version of the signal (same length as `signal`)
/// * `multiplier` - Threshold multiplier (typically 1.0-2.0, higher = more aggressive)
///
/// # Panics
///
/// Panics in debug builds if `signal` and `median` have different lengths.
///
/// # Example
///
/// ```rust,ignore
/// use non_empty_slice::non_empty_slice;
/// use audio_samples::operations::onset::kernels::apply_adaptive_threshold;
///
/// let mut signal = non_empty_slice![1.0, 5.0, 2.0, 8.0, 1.5].to_non_empty_vec();
/// let median = non_empty_slice![1.5, 2.0, 2.5, 3.0, 1.8];
/// apply_adaptive_threshold(&mut signal, &median, 2.0);
/// // Values below median * 2.0 are set to zero
/// ```
#[inline]
pub fn apply_adaptive_threshold(
    signal: &mut NonEmptySlice<f64>,
    median: &NonEmptySlice<f64>,
    multiplier: f64,
) {
    debug_assert!(signal.len() == median.len());

    for i in 0..signal.len().get() {
        let thresh = median[i] * multiplier;
        if signal[i] < thresh {
            signal[i] = 0.0;
        }
    }
}
