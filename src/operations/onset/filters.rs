//! Signal processing utilities for onset detection.
//!
//! This module provides basic signal processing functions commonly used in onset detection
//! pipelines. These utilities operate on 1D signals (typically onset detection functions)
//! to improve detection accuracy and robustness.
//!
//! The functions include:
//! - Median filtering for smoothing and adaptive thresholding
//! - Rectification for keeping only positive values
//! - Logarithmic compression for dynamic range reduction
//!
//! These operations are typically applied in sequence to condition onset detection functions
//! before peak picking. For example: compute ODF → rectify → log compress → median filter →
//! adaptive threshold → peak pick.

use std::num::NonZeroUsize;

use non_empty_slice::{NonEmptySlice, NonEmptyVec};

use crate::{AudioSampleError, AudioSampleResult, ParameterError};

/// Applies a median filter to smooth a signal.
///
/// For each sample, replaces its value with the median of a window centered around it.
/// This nonlinear filter effectively removes outliers and noise while preserving edges,
/// making it useful for adaptive thresholding in onset detection.
///
/// # Arguments
///
/// * `signal` — Input signal to filter
/// * `filter_length` — Length of the median filter kernel (must be odd and > 0)
///
/// # Returns
///
/// Filtered signal with the same length as the input.
///
/// # Errors
///
/// - [crate::AudioSampleError::Parameter] if `filter_length` is even (must be odd for symmetry)
///
/// # Example
///
/// ```rust,ignore
/// use non_empty_slice::non_empty_slice;
/// use audio_samples::operations::onset::filters::median_filter;
///
/// let signal = non_empty_slice![1.0, 5.0, 2.0, 3.0, 100.0, 4.0, 5.0];
/// let filtered = median_filter(&signal, crate::nzu!(3))?;
/// // The outlier 100.0 is replaced by the median of its window
/// ```
#[inline]
pub fn median_filter(
    signal: &NonEmptySlice<f64>,
    filter_length: NonZeroUsize,
) -> AudioSampleResult<NonEmptyVec<f64>> {
    if filter_length.get().is_multiple_of(2) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "filter_length",
            "Must be odd and > 0",
        )));
    }

    if filter_length.get() == 1 {
        return Ok(signal.to_non_empty_vec());
    }

    let half = filter_length.get() / 2;
    let mut out = Vec::with_capacity(signal.len().get());

    for i in 0..signal.len().get() {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(signal.len().get());
        let mut window = signal[lo..hi].to_vec();
        window.sort_by(f64::total_cmp);

        out.push(window[window.len() / 2]);
    }
    // safety: signal is non-empty and filter_length > 0 ensures out is non-empty
    let out = unsafe { NonEmptyVec::new_unchecked(out) };
    Ok(out)
}

/// Rectifies a signal in-place by replacing negative values with zero.
///
/// Half-wave rectification keeps only positive values, setting all negative values to zero.
/// This is commonly used in onset detection to focus on spectral increases rather than
/// decreases.
///
/// # Arguments
///
/// * `signal` — Signal to rectify (modified in-place)
///
/// # Example
///
/// ```rust,ignore
/// use non_empty_slice::non_empty_slice;
/// use audio_samples::operations::onset::filters::rectify_inplace;
///
/// let mut signal = non_empty_slice![1.0, -2.0, 3.0, -4.0, 5.0].to_non_empty_vec();
/// rectify_inplace(&mut signal);
/// // signal is now [1.0, 0.0, 3.0, 0.0, 5.0]
/// ```
#[inline]
pub fn rectify_inplace(signal: &mut NonEmptySlice<f64>) {
    for v in signal {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Applies logarithmic compression to a signal in-place.
///
/// Compresses the dynamic range using the formula: `v = log(1 + alpha * v)`.
/// Higher `alpha` values produce more aggressive compression, bringing down peaks and
/// emphasizing quieter values. This helps onset detection be more sensitive to subtle onsets.
///
/// # Arguments
///
/// * `signal` — Signal to compress (modified in-place)
/// * `alpha` — Compression factor (typically 10.0-1000.0, higher = more compression)
///
/// # Example
///
/// ```rust,ignore
/// use non_empty_slice::non_empty_slice;
/// use audio_samples::operations::onset::filters::log_compress_inplace;
///
/// let mut signal = non_empty_slice![1.0, 10.0, 100.0, 1000.0].to_non_empty_vec();
/// log_compress_inplace(&mut signal, 100.0);
/// // Large peaks are compressed more than small values
/// ```
#[inline]
pub fn log_compress_inplace(signal: &mut NonEmptySlice<f64>, alpha: f64) {
    for v in signal {
        *v = alpha.mul_add(*v, 1.0).ln();
    }
}
