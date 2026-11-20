//! Conditioning of a 1-D onset detection function before peak picking: median
//! filtering, half-wave rectification and logarithmic compression. A typical chain
//! is ODF, rectify, compress, median filter, adaptive threshold, peak picking.

use std::num::NonZeroUsize;

use non_empty_slice::{NonEmptySlice, NonEmptyVec};

use crate::{AudioSampleError, AudioSampleResult, ParameterError};

/// Applies a median filter to smooth a signal.
///
/// For each sample i, replaces its value with the median of the window
/// `i − ⌊L/2⌋ ..= i + ⌊L/2⌋` (L = `filter_length`), truncated at the signal
/// edges. Where truncation leaves an even number of elements the upper of the
/// two middle values is used. This nonlinear filter removes outliers while
/// preserving edges; it is used to form the local reference level for
/// adaptive thresholding in onset detection.
///
/// # Arguments
///
/// * `signal` - Input signal to filter
/// * `filter_length` - Length of the median filter kernel (must be odd and > 0)
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
/// ```rust
/// use non_empty_slice::non_empty_vec;
/// use audio_samples::nzu;
/// use audio_samples::operations::onset::filters::median_filter;
///
/// let signal = non_empty_vec![1.0, 5.0, 2.0, 3.0, 100.0, 4.0, 5.0];
/// let filtered = median_filter(&signal, nzu!(3))?;
/// // The outlier 100.0 is replaced by the median of its window [3, 100, 4].
/// assert_eq!(filtered[4], 4.0);
/// // An even length is an error.
/// assert!(median_filter(&signal, nzu!(4)).is_err());
/// # Ok::<(), audio_samples::AudioSampleError>(())
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
        window.sort_unstable_by(f64::total_cmp);

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
/// * `signal` - Signal to rectify (modified in-place)
///
/// # Example
///
/// ```rust
/// use non_empty_slice::non_empty_vec;
/// use audio_samples::operations::onset::filters::rectify_inplace;
///
/// let mut signal = non_empty_vec![1.0, -2.0, 3.0, -4.0, 5.0];
/// rectify_inplace(&mut signal);
/// assert_eq!(signal.as_slice(), &[1.0, 0.0, 3.0, 0.0, 5.0]);
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
/// * `signal` - Signal to compress (modified in-place)
/// * `alpha` - Compression factor (typically 10.0-1000.0, higher = more compression)
///
/// # Example
///
/// ```rust
/// use non_empty_slice::non_empty_vec;
/// use audio_samples::operations::onset::filters::log_compress_inplace;
///
/// let mut signal = non_empty_vec![1.0, 10.0, 100.0, 1000.0];
/// log_compress_inplace(&mut signal, 100.0);
/// // ln(1 + 100·x): each tenfold step adds about ln 10 ≈ 2.3.
/// assert!((signal[0] - 101f64.ln()).abs() < 1e-12);
/// assert!((signal[3] - signal[2] - 10f64.ln()).abs() < 1e-3);
/// ```
#[inline]
pub fn log_compress_inplace(signal: &mut NonEmptySlice<f64>, alpha: f64) {
    for v in signal {
        *v = alpha.mul_add(*v, 1.0).ln();
    }
}
