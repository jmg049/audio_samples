//! Peak picking and signal enhancement utilities for onset detection.
//!
//! This module provides the core signal analysis building blocks used by onset
//! detection: adaptive thresholding, peak picking with temporal constraints, and
//! signal pre-processing (pre-emphasis, median filtering, normalisation, smoothing).
//!
//! Raw onset strength functions are noisy and have varying amplitude across signals
//! and recording conditions. A fixed detection threshold fails in quiet sections and
//! produces false positives in loud ones. This module handles signal conditioning and
//! peak selection so that onset detectors can focus on their specific spectral or
//! temporal features.
//!
//! The primary entry point is [`pick_peaks`], which applies a configurable pipeline:
//! optional pre-emphasis → optional median filtering → optional normalisation →
//! adaptive thresholding → local maximum detection → temporal constraint filtering.
//! Each step is also available as a standalone function for use in custom pipelines.
//!
//! # Example
//!
//! ```
//! use audio_samples::operations::peak_picking::pick_peaks;
//! use audio_samples::operations::types::PeakPickingConfig;
//! use non_empty_slice::NonEmptySlice;
//!
//! let data = [0.1f64, 0.3, 0.8, 0.2, 0.4, 0.9, 0.1];
//! let onset_strength = NonEmptySlice::from_slice(&data).unwrap();
//! // Disable pre-processing steps that can suppress peaks on short signals.
//! let mut config = PeakPickingConfig::default();
//! config.pre_emphasis = false;
//! config.median_filter = false;
//! config.normalize_onset_strength = false;
//! config.adaptive_threshold.window_size = 3;
//! let peaks = pick_peaks(onset_strength, &config).unwrap();
//! assert!(!peaks.is_empty());
//! assert!(peaks.iter().all(|&i| i < data.len()));
//! ```
use std::num::NonZeroUsize;

use non_empty_iter::{IntoNonEmptyIterator, NonEmptyIterator};
use non_empty_slice::{NonEmptySlice, NonEmptyVec};

use crate::operations::types::{
    AdaptiveThresholdConfig, AdaptiveThresholdMethod, NormalizationMethod, PeakPickingConfig,
};
use crate::{AudioSampleError, AudioSampleResult, ParameterError};

/// Computes an adaptive threshold for each sample of an onset strength function.
///
/// Three thresholding methods are available, selected by `config.method`:
///
/// - `Delta` – threshold = local max − δ. Responsive to sudden level changes.
/// - `Percentile` – threshold = percentile(local window, p). More robust to noise.
/// - `Combined` – maximum of the delta and percentile thresholds. Balances both.
///
/// The computed threshold at each position is clamped to
/// `[config.min_threshold, config.max_threshold]`.
///
/// # Arguments
///
/// - `onset_strength` – Non-empty slice of onset strength values.
/// - `config` – Thresholding parameters: method, window size, delta, percentile,
///   and threshold bounds.
///
/// # Returns
///
/// A `Vec<f64>` of threshold values with the same length as `onset_strength`.
///
/// # Errors
///
/// Returns [crate::AudioSampleError::Parameter] if `config` fails validation
/// (e.g. delta < 0, percentile outside `[0, 1]`, or `min_threshold > max_threshold`).
///
/// # Example
///
/// ```
/// use audio_samples::operations::peak_picking::adaptive_threshold;
/// use audio_samples::operations::types::AdaptiveThresholdConfig;
/// use non_empty_slice::NonEmptySlice;
///
/// let data = [0.1f64, 0.3, 0.8, 0.2, 0.4, 0.9, 0.1];
/// let onset_strength = NonEmptySlice::from_slice(&data).unwrap();
/// let config = AdaptiveThresholdConfig::default();
/// let thresholds = adaptive_threshold(onset_strength, &config).unwrap();
/// assert_eq!(thresholds.len(), data.len());
/// ```
#[inline]
pub fn adaptive_threshold(
    onset_strength: &NonEmptySlice<f64>,
    config: &AdaptiveThresholdConfig,
) -> AudioSampleResult<Vec<f64>> {
    config.validate()?;

    let len = onset_strength.len().get();
    let mut thresholds = Vec::with_capacity(len);
    let half_window = config.window_size / 2;

    for i in 0..len {
        // Determine local window bounds
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(len);
        let window = &onset_strength[start..end];
        let window = non_empty_slice::non_empty_slice!(window);

        let threshold = match config.method {
            AdaptiveThresholdMethod::Delta => {
                // Delta-based: threshold = local_max - delta
                let local_max = window.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
                local_max - config.delta
            }
            AdaptiveThresholdMethod::Percentile => {
                // Percentile-based: threshold = percentile(window, p)
                percentile(window, config.percentile)
            }
            AdaptiveThresholdMethod::Combined => {
                // Combined: max(delta_threshold, percentile_threshold)
                let local_max = window.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
                let delta_threshold = local_max - config.delta;
                let percentile_threshold = percentile(window, config.percentile);
                delta_threshold.max(percentile_threshold)
            }
        };

        // Apply min/max threshold bounds
        let bounded_threshold = threshold
            .max(config.min_threshold)
            .min(config.max_threshold);

        thresholds.push(bounded_threshold);
    }

    Ok(thresholds)
}

/// Picks local maxima from an onset strength function using adaptive thresholding.
///
/// Applies a configurable pipeline to `onset_strength` and returns the indices of
/// detected peaks in ascending order:
///
/// 1. Pre-emphasis filtering (optional) — high-pass `y[n] = x[n] − α·x[n-1]`
/// 2. Median filtering (optional) — noise reduction
/// 3. Normalisation (optional)
/// 4. Adaptive threshold computation
/// 5. Local maximum detection above threshold (interior samples only)
/// 6. Temporal constraint filtering — retains only the strongest peak within each
///    minimum-separation window
///
/// Configuration window sizes and minimum separation are automatically adjusted
/// downward when the signal is shorter than the configured values.
///
/// # Arguments
///
/// - `onset_strength` – Non-empty slice of onset strength values.
/// - `config` – Peak picking parameters including threshold configuration,
///   minimum peak separation, pre-processing flags, and normalisation method.
///
/// # Returns
///
/// A `Vec<usize>` of peak indices into `onset_strength`, sorted in ascending order.
/// Returns an empty `Vec` if no peaks are found.
///
/// # Errors
///
/// Returns [crate::AudioSampleError::Parameter] if `config` fails validation.
///
/// # Example
///
/// ```
/// use audio_samples::operations::peak_picking::pick_peaks;
/// use audio_samples::operations::types::PeakPickingConfig;
/// use non_empty_slice::NonEmptySlice;
///
/// let data = [0.1f64, 0.3, 0.8, 0.2, 0.4, 0.9, 0.1];
/// let onset_strength = NonEmptySlice::from_slice(&data).unwrap();
/// // Disable pre-processing steps that can suppress peaks on short signals.
/// let mut config = PeakPickingConfig::default();
/// config.pre_emphasis = false;
/// config.median_filter = false;
/// config.normalize_onset_strength = false;
/// config.adaptive_threshold.window_size = 3;
/// let peaks = pick_peaks(onset_strength, &config).unwrap();
/// // All returned indices are valid positions in the onset strength function.
/// assert!(peaks.iter().all(|&i| i < data.len()));
/// ```
#[inline]
pub fn pick_peaks(
    onset_strength: &NonEmptySlice<f64>,
    config: &PeakPickingConfig,
) -> AudioSampleResult<Vec<usize>> {
    config.validate().map_err(|e| {
        AudioSampleError::Parameter(ParameterError::invalid_value(
            "peak_picking_config",
            format!("Invalid peak picking config: {e}"),
        ))
    })?;
    // Adapt configuration for short signals
    let mut adjusted_config = *config;
    let signal_len = onset_strength.len().get();

    // Adjust window size if it's larger than the signal
    if adjusted_config.adaptive_threshold.window_size > signal_len {
        adjusted_config.adaptive_threshold.window_size = signal_len.max(3); // Minimum window size of 3
    }

    // Adjust min_peak_separation if it's unreasonably large for the signal
    if adjusted_config.min_peak_separation.get() >= signal_len / 2 {
        // safety: guaranteed to be non-zero due to max(1)
        adjusted_config.min_peak_separation =
            unsafe { NonZeroUsize::new_unchecked((signal_len / 4).max(1)) };
    }

    // Step 1: Apply signal enhancement
    let mut processed_strength = onset_strength.to_non_empty_vec();

    // Pre-emphasis filtering
    if config.pre_emphasis {
        processed_strength = apply_pre_emphasis(&processed_strength, config.pre_emphasis_coeff)?;
    }

    // Median filtering
    if config.median_filter {
        processed_strength = apply_median_filter(&processed_strength, config.median_filter_length)?;
    }

    // Normalization
    if config.normalize_onset_strength {
        processed_strength =
            normalize_onset_strength(&processed_strength, config.normalization_method);
    }

    // Step 2: Compute adaptive threshold
    let thresholds = adaptive_threshold(&processed_strength, &config.adaptive_threshold)?;

    // Step 3: Find candidate peaks (local maxima above threshold)
    let mut candidates = Vec::new();

    for i in 1..processed_strength.len().get() - 1 {
        let current = processed_strength[i];
        let prev = processed_strength[i - 1];
        let next = processed_strength[i + 1];

        // Check if it's a local maximum above threshold
        if current > prev && current > next && current > thresholds[i] {
            candidates.push((i, current));
        }
    }

    // Early return if no peaks found
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    // safety: candidates is non-empty due to check above
    let candidates = unsafe { NonEmptyVec::new_unchecked(candidates) };
    // Step 4: Apply temporal constraints
    let peaks = apply_temporal_constraints(&candidates, adjusted_config.min_peak_separation.get());

    Ok(peaks)
}

/// Applies a first-order high-pass pre-emphasis filter to enhance transients.
///
/// Each output sample `y[n] = x[n] − α·x[n-1]`, where α is `coeff`.
/// The first sample is passed through unchanged. A coefficient of `0.0` is
/// the identity; higher values provide stronger high-frequency emphasis.
///
/// # Arguments
///
/// - `signal` – Non-empty slice of input values.
/// - `coeff` – Pre-emphasis coefficient in `[0.0, 1.0]`.
///
/// # Returns
///
/// A new `NonEmptyVec<f64>` the same length as `signal` with the filter applied.
///
/// # Errors
///
/// Returns [crate::AudioSampleError::Parameter] if `coeff` is outside `[0.0, 1.0]`.
///
/// # Example
///
/// ```
/// use audio_samples::operations::peak_picking::apply_pre_emphasis;
/// use non_empty_slice::NonEmptySlice;
///
/// let data = [1.0f64, 2.0, 3.0, 2.0, 1.0];
/// let signal = NonEmptySlice::from_slice(&data).unwrap();
/// let filtered = apply_pre_emphasis(signal, 0.97).unwrap();
/// assert_eq!(filtered.len(), signal.len());
/// // First sample passes through unchanged.
/// assert_eq!(filtered[0], data[0]);
/// ```
#[inline]
pub fn apply_pre_emphasis(
    signal: &NonEmptySlice<f64>,
    coeff: f64,
) -> AudioSampleResult<NonEmptyVec<f64>> {
    if !(0.0..=1.0).contains(&coeff) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "pre_emphasis_coefficient",
            "Pre-emphasis coefficient must be between 0.0 and 1.0",
        )));
    }

    let mut filtered = Vec::with_capacity(signal.len().get());

    // First sample unchanged
    filtered.push(signal[0]);

    // Apply pre-emphasis filter
    for i in 1..signal.len().get() {
        filtered.push(coeff.mul_add(-signal[i - 1], signal[i]));
    }
    // safety: filtered is guaranteed to be non-empty since signal is non-empty
    let filtered = unsafe { NonEmptyVec::new_unchecked(filtered) };
    Ok(filtered)
}

/// Applies a median filter to reduce impulse noise while preserving peaks.
///
/// Each output sample is replaced by the median of a symmetric window centred
/// on that sample, using edge-clamped boundaries. A `filter_length` of `1`
/// returns the signal unchanged.
///
/// # Arguments
///
/// - `signal` – Non-empty slice of input values.
/// - `filter_length` – Length of the median window; must be an odd `NonZeroUsize`.
///
/// # Returns
///
/// A new `NonEmptyVec<f64>` the same length as `signal` with the filter applied.
///
/// # Errors
///
/// Returns [crate::AudioSampleError::Parameter] if `filter_length` is even.
///
/// # Example
///
/// ```
/// use std::num::NonZeroUsize;
/// use audio_samples::operations::peak_picking::apply_median_filter;
/// use non_empty_slice::NonEmptySlice;
///
/// let data = [1.0f64, 5.0, 2.0, 8.0, 3.0]; // contains outliers
/// let signal = NonEmptySlice::from_slice(&data).unwrap();
/// let filtered = apply_median_filter(signal, NonZeroUsize::new(3).unwrap()).unwrap();
/// assert_eq!(filtered.len(), signal.len());
/// // The spike at index 1 should be reduced.
/// assert!(filtered[1] < data[1]);
/// ```
#[inline]
pub fn apply_median_filter(
    signal: &NonEmptySlice<f64>,
    filter_length: NonZeroUsize,
) -> AudioSampleResult<NonEmptyVec<f64>> {
    if filter_length.get().is_multiple_of(2) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "filter_length",
            "Median filter length must be odd and greater than 0",
        )));
    }

    if filter_length.get() == 1 {
        return Ok(signal.to_owned());
    }

    let mut filtered = Vec::with_capacity(signal.len().get());
    let half_length = filter_length.get() / 2;

    for i in 0..signal.len().get() {
        // Determine window bounds
        let start = i.saturating_sub(half_length);
        let end = (i + half_length + 1).min(signal.len().get());

        // Extract window and compute median
        let mut window: Vec<f64> = signal[start..end].to_vec();
        window.sort_by(|a, b| {
            a.partial_cmp(b)
                .map_or(std::cmp::Ordering::Equal, |order| order)
        });
        let median = window[window.len() / 2];
        filtered.push(median);
    }
    // safety: filtered is guaranteed to be non-empty since signal is non-empty
    let filtered = unsafe { NonEmptyVec::new_unchecked(filtered) };
    Ok(filtered)
}

/// Normalises an onset strength function to a consistent amplitude scale.
///
/// Available methods (selected via `method`):
///
/// - `Peak` – divides by the maximum absolute value; output peaks at ±1.
/// - `MinMax` – scales to `[0, 1]`.
/// - `ZScore` – zero mean, unit variance.
/// - `Mean` – subtracts the mean.
/// - `Median` – subtracts the median.
///
/// When the signal is constant (or all zeros for `Peak`), the input is returned
/// unchanged.
///
/// # Arguments
///
/// - `onset_strength` – Non-empty slice of onset strength values.
/// - `method` – Normalisation method to apply.
///
/// # Returns
///
/// A new `NonEmptyVec<f64>` the same length as `onset_strength`.
///
/// # Example
///
/// ```
/// use audio_samples::operations::peak_picking::normalize_onset_strength;
/// use audio_samples::operations::types::NormalizationMethod;
/// use non_empty_slice::NonEmptySlice;
///
/// let data = [0.2f64, 0.5, 1.0, 0.3];
/// let onset_strength = NonEmptySlice::from_slice(&data).unwrap();
/// let normalized = normalize_onset_strength(onset_strength, NormalizationMethod::Peak);
/// // Peak normalisation maps the maximum to 1.0.
/// let max_abs = normalized.iter().cloned().fold(0.0f64, f64::max);
/// assert!((max_abs - 1.0).abs() < 1e-10);
/// ```
#[inline]
#[must_use]
pub fn normalize_onset_strength(
    onset_strength: &NonEmptySlice<f64>,
    method: NormalizationMethod,
) -> NonEmptyVec<f64> {
    match method {
        NormalizationMethod::Peak => {
            // Peak normalization: divide by maximum absolute value
            let max_abs = onset_strength
                .iter()
                .fold(0.0, |acc: f64, &x| acc.max(x.abs()));

            if max_abs == 0.0 {
                return onset_strength.to_owned();
            }

            onset_strength
                .into_non_empty_iter()
                .map(|&x| x / max_abs)
                .collect_non_empty()
        }
        NormalizationMethod::MinMax => {
            // Min-max normalization: scale to [0, 1]
            let min_val = onset_strength
                .iter()
                .fold(f64::INFINITY, |acc, &x| acc.min(x));
            let max_val = onset_strength
                .iter()
                .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

            if (max_val - min_val).abs() < f64::EPSILON {
                return onset_strength.to_owned();
            }

            onset_strength
                .into_non_empty_iter()
                .map(|&x| (x - min_val) / (max_val - min_val))
                .collect_non_empty()
        }
        NormalizationMethod::ZScore => {
            // Z-score normalization: zero mean, unit variance
            let mean = onset_strength.iter().fold(0.0, |acc, &x| acc + x)
                / onset_strength.len().get() as f64;
            let variance = onset_strength
                .iter()
                .map(|&x| (x - mean).powi(2))
                .fold(0.0, |acc, x| acc + x)
                / onset_strength.len().get() as f64;

            if variance == 0.0 {
                return onset_strength.to_owned();
            }

            let std_dev = variance.sqrt();

            onset_strength
                .into_non_empty_iter()
                .map(|&x| (x - mean) / std_dev)
                .collect_non_empty()
        }
        NormalizationMethod::Mean => {
            // Mean normalization: subtract mean
            let mean = onset_strength.iter().fold(0.0, |acc, &x| acc + x)
                / onset_strength.len().get() as f64;

            onset_strength
                .into_non_empty_iter()
                .map(|&x| x - mean)
                .collect_non_empty()
        }
        NormalizationMethod::Median => {
            // Median normalization: subtract median
            let mut sorted = onset_strength.to_vec();
            sorted.sort_by(|a, b| {
                a.partial_cmp(b)
                    .map_or(std::cmp::Ordering::Equal, |order| order)
            });
            let median = sorted[sorted.len() / 2];

            onset_strength
                .into_non_empty_iter()
                .map(|&x| x - median)
                .collect_non_empty()
        }
    }
}

/// Filters candidate peaks to enforce a minimum sample separation.
///
/// Candidates are processed in descending strength order. A candidate is kept
/// if it is at least `min_separation` samples away from all already-selected
/// peaks. The returned indices are sorted in ascending order.
///
/// # Arguments
///
/// - `candidates` – Non-empty slice of `(index, strength)` tuples.
/// - `min_separation` – Minimum distance between any two kept peaks in samples.
fn apply_temporal_constraints(
    candidates: &NonEmptySlice<(usize, f64)>,
    min_separation: usize,
) -> Vec<usize> {
    // Sort candidates by strength (descending)
    let mut sorted_candidates = candidates.to_vec();
    sorted_candidates.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .map_or(std::cmp::Ordering::Equal, |order| order)
    });

    let mut selected_peaks = Vec::new();

    for &(index, _strength) in &sorted_candidates {
        // Check if this peak is far enough from all selected peaks
        let mut valid = true;
        for &selected_index in &selected_peaks {
            if index.abs_diff(selected_index) < min_separation {
                valid = false;
                break;
            }
        }

        if valid {
            selected_peaks.push(index);
        }
    }

    // Sort selected peaks by index
    selected_peaks.sort_unstable();
    selected_peaks
}

/// Applies two-stage smoothing: median filtering followed by a moving average.
///
/// Median filtering is applied first to remove impulse noise, then a centred
/// moving average is applied to reduce remaining high-frequency variation.
/// Passing `window_size = 1` skips the moving-average step.
///
/// # Arguments
///
/// - `onset_strength` – Non-empty slice of onset strength values.
/// - `window_size` – Moving-average window size. A value of `1` skips the step.
/// - `median_length` – Median filter window length; must be an odd `NonZeroUsize`.
///
/// # Returns
///
/// A new `NonEmptyVec<f64>` the same length as `onset_strength`.
///
/// # Errors
///
/// Returns [crate::AudioSampleError::Parameter] if `median_length` is even.
///
/// # Example
///
/// ```
/// use std::num::NonZeroUsize;
/// use audio_samples::operations::peak_picking::smooth_onset_strength;
/// use non_empty_slice::NonEmptySlice;
///
/// let data = [0.1f64, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3];
/// let onset_strength = NonEmptySlice::from_slice(&data).unwrap();
/// let smoothed = smooth_onset_strength(
///     onset_strength,
///     NonZeroUsize::new(3).unwrap(),
///     NonZeroUsize::new(3).unwrap(),
/// ).unwrap();
/// assert_eq!(smoothed.len(), onset_strength.len());
/// ```
#[inline]
pub fn smooth_onset_strength(
    onset_strength: &NonEmptySlice<f64>,
    window_size: NonZeroUsize,
    median_length: NonZeroUsize,
) -> AudioSampleResult<NonEmptyVec<f64>> {
    // Apply median filtering first
    let mut smoothed: NonEmptyVec<f64> = apply_median_filter(onset_strength, median_length)?;

    // Apply moving average smoothing
    if window_size.get() > 1 {
        smoothed = apply_moving_average(&smoothed, window_size);
    }
    Ok(smoothed)
}

/// Applies a centred moving-average filter to smooth `signal`.
///
/// A `window_size` of `1` returns the signal unchanged.
fn apply_moving_average(
    signal: &NonEmptySlice<f64>,
    window_size: NonZeroUsize,
) -> NonEmptyVec<f64> {
    if window_size.get() == 1 {
        return signal.to_owned();
    }

    let mut smoothed = Vec::with_capacity(signal.len().get());
    let half_window = window_size.get() / 2;

    for i in 0..signal.len().get() {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(signal.len().get());

        let sum: f64 = signal[start..end].iter().sum();
        let average = sum / (end - start) as f64;
        smoothed.push(average);
    }
    // safety: smoothed is guaranteed to be non-empty since signal is non-empty

    unsafe { NonEmptyVec::new_unchecked(smoothed) }
}

/// Returns the `percentile`-th quantile of `values` using linear interpolation.
///
/// `percentile` must be in `[0.0, 1.0]` (not validated; caller is responsible).
fn percentile(values: &NonEmptySlice<f64>, percentile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| {
        a.partial_cmp(b)
            .map_or(std::cmp::Ordering::Equal, |order| order)
    });

    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }

    let index = percentile * (n - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;

    if lower == upper {
        sorted[lower]
    } else {
        let weight = index - lower as f64;
        sorted[lower].mul_add(1.0 - weight, sorted[upper] * weight)
    }
}

#[cfg(test)]
mod tests {
    use non_empty_slice::{NonEmptyVec, non_empty_vec};

    use super::*;
    use crate::operations::types::NormalizationMethod;

    #[test]
    fn test_adaptive_threshold_delta() {
        let onset_strength =
            NonEmptySlice::from_slice(&[0.1, 0.3, 0.8, 0.2, 0.4, 0.9, 0.1]).unwrap();
        let config = AdaptiveThresholdConfig::delta(0.1, 3);

        let thresholds = adaptive_threshold(&onset_strength, &config).unwrap();
        assert_eq!(thresholds.len(), onset_strength.len().get());

        // All thresholds should be above minimum
        for &threshold in &thresholds {
            assert!(threshold >= config.min_threshold);
        }
    }

    #[test]
    fn test_adaptive_threshold_percentile() {
        let onset_strength = non_empty_vec![0.1, 0.3, 0.8, 0.2, 0.4, 0.9, 0.1];
        let config = AdaptiveThresholdConfig::percentile(0.8, 3);

        let thresholds = adaptive_threshold(&onset_strength, &config).unwrap();
        assert_eq!(thresholds.len(), onset_strength.len().get());
        // Thresholds should be within bounds
        for &threshold in &thresholds {
            assert!(threshold >= config.min_threshold);
            assert!(threshold <= config.max_threshold);
        }
    }

    #[test]
    fn test_adaptive_threshold_combined() {
        let onset_strength = non_empty_vec![0.1, 0.3, 0.8, 0.2, 0.4, 0.9, 0.1];
        let config = AdaptiveThresholdConfig::combined(0.1, 0.8, 3);

        let thresholds = adaptive_threshold(&onset_strength, &config).unwrap();
        assert_eq!(thresholds.len(), onset_strength.len().get());
    }

    #[test]
    fn test_pick_peaks_basic() {
        let onset_strength = non_empty_vec![0.1, 0.3, 0.8, 0.2, 0.4, 0.9, 0.1];
        let mut config = PeakPickingConfig::default();

        // Configure for small test data
        config.adaptive_threshold.window_size = 3; // Small window for 7 samples
        config.min_peak_separation = crate::nzu!(1); // Allow adjacent peaks for testing
        config.pre_emphasis = false; // Disable to avoid edge effects
        config.median_filter = false; // Disable to avoid smoothing
        config.normalize_onset_strength = false; // Keep original values

        let peaks = pick_peaks(&onset_strength, &config).unwrap();

        // Should detect some peaks (indices 2 and 5 should be peaks: 0.8 and 0.9)
        assert!(!peaks.is_empty());

        // All peaks should be valid indices
        for &peak in &peaks {
            assert!(peak < onset_strength.len().get());
        }
    }

    #[test]
    fn test_pick_peaks_with_constraints() {
        let onset_strength = non_empty_vec![0.1, 0.5, 0.6, 0.7, 0.2, 0.8, 0.1];
        let mut config = PeakPickingConfig::default();
        config.min_peak_separation = crate::nzu!(3);

        let peaks = pick_peaks(&onset_strength, &config).unwrap();

        // Check minimum separation constraint
        for i in 1..peaks.len() {
            assert!(peaks[i] - peaks[i - 1] >= config.min_peak_separation.get());
        }
    }

    #[test]
    fn test_pre_emphasis() {
        let signal: NonEmptyVec<f64> = non_empty_vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let coeff: f64 = 0.97;

        let filtered = apply_pre_emphasis(&signal, coeff).unwrap();
        assert_eq!(filtered.len(), signal.len());

        // First sample should be unchanged
        assert_eq!(filtered[0], signal[0]);

        // Check pre-emphasis formula
        for i in 1..signal.len().get() {
            let expected = signal[i] - coeff * signal[i - 1];
            assert!((filtered[i] - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_median_filter() {
        let signal: NonEmptyVec<f64> = non_empty_vec![1.0, 5.0, 2.0, 8.0, 3.0]; // Contains outlier
        let filtered = apply_median_filter(&signal, crate::nzu!(3)).unwrap();

        assert_eq!(filtered.len(), signal.len());

        // Median filter should reduce the outlier effect
        assert!(filtered[1] < signal[1]); // Outlier should be reduced
    }

    #[test]
    fn test_normalize_onset_strength_peak() {
        let onset_strength: NonEmptyVec<f64> = non_empty_vec![0.1, 0.5, 1.0, 0.3];
        let normalized = normalize_onset_strength(&onset_strength, NormalizationMethod::Peak);

        assert_eq!(normalized.len(), onset_strength.len());

        // Maximum absolute value should be 1.0
        let max_abs = normalized.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
        assert!((max_abs - 1.0f64).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_onset_strength_minmax() {
        let onset_strength: NonEmptyVec<f64> = non_empty_vec![0.1, 0.5, 1.0, 0.3];
        let normalized = normalize_onset_strength(&onset_strength, NormalizationMethod::MinMax);

        assert_eq!(normalized.len(), onset_strength.len());

        // Range should be [0, 1]
        let min_val = normalized.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max_val = normalized
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

        assert!(min_val >= 0.0);
        assert!(max_val <= 1.0);
        assert!((max_val - 1.0).abs() < 1e-10);
        assert!((min_val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_smooth_onset_strength() {
        let onset_strength: NonEmptyVec<f64> = non_empty_vec![0.1, 0.9, 0.1, 0.8, 0.2]; // Noisy signal
        let smoothed =
            smooth_onset_strength(&onset_strength, crate::nzu!(3), crate::nzu!(3)).unwrap();

        assert_eq!(smoothed.len(), onset_strength.len());

        // Smoothed signal should have less variation
        let original_std = standard_deviation(&onset_strength);
        let smoothed_std = standard_deviation(&smoothed);
        assert!(smoothed_std <= original_std);
    }

    #[test]
    fn test_temporal_constraints() {
        let candidates = non_empty_vec![(1, 0.8), (2, 0.6), (5, 0.9), (6, 0.7)];
        let min_separation = 2;

        let selected = apply_temporal_constraints(&candidates, min_separation);

        // Check separation constraint
        for i in 1..selected.len() {
            assert!(selected[i] - selected[i - 1] >= min_separation);
        }
    }

    #[test]
    fn test_percentile() {
        let values: NonEmptyVec<f64> = non_empty_vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(percentile(&values, 0.0), 1.0);
        assert_eq!(percentile(&values, 1.0), 5.0);
        assert_eq!(percentile(&values, 0.5), 3.0); // Median of odd-length array

        // Test interpolation
        let p25 = percentile(&values, 0.25);
        assert!(p25 > 1.0 && p25 < 3.0);
    }

    #[test]
    fn test_edge_cases() {
        // Single sample
        let config = AdaptiveThresholdConfig::default();
        let single: NonEmptyVec<f64> = non_empty_vec![1.0];
        let thresholds = adaptive_threshold(&single, &config).unwrap();
        assert_eq!(thresholds.len(), 1);

        // All zeros
        let zeros: NonEmptyVec<f64> = non_empty_vec![0.0, 0.0, 0.0];
        let normalized = normalize_onset_strength(&zeros, NormalizationMethod::Peak);
        assert_eq!(normalized, zeros);
    }

    #[test]
    fn test_config_validation() {
        // Invalid delta
        let mut config = AdaptiveThresholdConfig::default();
        config.delta = -0.1;
        assert!(config.validate().is_err());

        // Invalid percentile
        config = AdaptiveThresholdConfig::default();
        config.percentile = 1.5;
        assert!(config.validate().is_err());

        // Invalid window size
        config = AdaptiveThresholdConfig::default();
        config.window_size = 0;
        assert!(config.validate().is_err());

        // Invalid threshold bounds
        config = AdaptiveThresholdConfig::default();
        config.min_threshold = 0.5;
        config.max_threshold = 0.3;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_peak_picking_presets() {
        let onset_strength = non_empty_vec![0.1, 0.3, 0.8, 0.2, 0.4, 0.9, 0.1];

        // Test different presets
        let configs = vec![
            PeakPickingConfig::default(),
            PeakPickingConfig::music(),
            PeakPickingConfig::speech(),
            PeakPickingConfig::drums(),
        ];

        for config in configs {
            let peaks = pick_peaks(&onset_strength, &config).unwrap();
            // All presets should produce valid results
            for &peak in &peaks {
                assert!(peak < onset_strength.len().get());
            }
        }
    }

    // Helper function for testing
    fn standard_deviation(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }
}
