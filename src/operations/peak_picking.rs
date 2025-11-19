//! Peak picking and post-processing utilities for onset detection.
//!
//! This module provides comprehensive peak picking and post-processing utilities
//! that serve as the foundation for all onset detection methods. It implements
//! adaptive thresholding, temporal constraints, and signal enhancement techniques
//! based on established research in music information retrieval.
//!
//! ## Mathematical Foundation
//!
//! ### Adaptive Thresholding
//!
//! Adaptive thresholding dynamically adjusts the detection threshold based on
//! local characteristics of the onset strength function. Three methods are supported:
//!
//! 1. **Delta-based**: `threshold(n) = max(onset_strength[n-W:n+W]) - δ`
//!    - Responsive to sudden changes but may be sensitive to noise
//!    - Good for signals with varying dynamics
//!
//! 2. **Percentile-based**: `threshold(n) = percentile(onset_strength[n-W:n+W], p)`
//!    - More robust to noise and outliers
//!    - Slower to adapt to sudden changes
//!
//! 3. **Combined**: `threshold(n) = max(delta_threshold(n), percentile_threshold(n))`
//!    - Balances responsiveness and robustness
//!    - Recommended for general use
//!
//! ### Peak Picking with Temporal Constraints
//!
//! Peak picking identifies local maxima in the onset strength function that exceed
//! the adaptive threshold. Temporal constraints ensure detected peaks are separated
//! by minimum time intervals to prevent multiple detections of the same onset event.
//!
//! The algorithm:
//! 1. Find all local maxima where `onset_strength[n] > threshold[n]`
//! 2. Apply minimum separation constraint: keep only peaks separated by ≥ `min_separation`
//! 3. When multiple peaks violate the constraint, keep the one with highest strength
//!
//! ### Signal Enhancement
//!
//! Pre-processing techniques enhance onset detection:
//!
//! - **Pre-emphasis**: High-pass filtering to emphasize transients
//!   `y[n] = x[n] - α * x[n-1]` where α is the pre-emphasis coefficient
//!
//! - **Median filtering**: Noise reduction while preserving peak structure
//!   `y[n] = median(x[n-k:n+k])` where k = (filter_length - 1) / 2
//!
//! - **Normalization**: Ensures consistent detection across signal levels
//!   Peak normalization: `y = x / max(|x|)`
//!
//! ## References
//!
//! - Bello, J.P., et al. "A tutorial on onset detection in music signals." IEEE TSALP 2005.
//! - Böck, S., et al. "Evaluating the online capabilities of onset detection methods." ISMIR 2012.
//! - Dixon, S. "Onset detection revisited." DAFx 2006.
use crate::operations::types::{
    AdaptiveThresholdConfig, AdaptiveThresholdMethod, NormalizationMethod, PeakPickingConfig,
};
use crate::{AudioSampleError, AudioSampleResult, ParameterError, RealFloat, to_precision};

/// Compute adaptive threshold for onset strength function.
///
/// This function implements adaptive thresholding algorithms that dynamically
/// adjust the detection threshold based on local characteristics of the onset
/// strength function. The threshold adapts to varying signal conditions to
/// improve detection accuracy.
///
/// # Mathematical Theory
///
/// Adaptive thresholding is essential for robust onset detection because:
/// - Fixed thresholds fail when signal dynamics vary
/// - Local adaptation handles both quiet and loud sections
/// - Multiple methods provide different trade-offs between responsiveness and robustness
///
/// # Arguments
///
/// * `onset_strength` - The onset strength function values
/// * `config` - Configuration for adaptive thresholding
///
/// # Returns
///
/// A vector of threshold values with the same length as the input
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::{AdaptiveThresholdConfig, AdaptiveThresholdMethod};
/// use audio_samples::operations::peak_picking::adaptive_threshold;
///
/// let onset_strength = vec![0.1, 0.3, 0.8, 0.2, 0.4, 0.9, 0.1];
/// let config = AdaptiveThresholdConfig::new();
/// let thresholds = adaptive_threshold(&onset_strength, &config).unwrap();
/// assert_eq!(thresholds.len(), onset_strength.len());
/// ```
pub fn adaptive_threshold<F: RealFloat>(
    onset_strength: &[F],
    config: &AdaptiveThresholdConfig<F>,
) -> AudioSampleResult<Vec<F>> {
    config.validate()?;

    if onset_strength.is_empty() {
        return Ok(Vec::new());
    }

    let len = onset_strength.len();
    let mut thresholds = Vec::with_capacity(len);
    let half_window = config.window_size / 2;

    for i in 0..len {
        // Determine local window bounds
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(len);
        let window = &onset_strength[start..end];

        let threshold = match config.method {
            AdaptiveThresholdMethod::Delta => {
                // Delta-based: threshold = local_max - delta
                let local_max = window.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x));
                local_max - config.delta
            }
            AdaptiveThresholdMethod::Percentile => {
                // Percentile-based: threshold = percentile(window, p)
                percentile(window, config.percentile)
            }
            AdaptiveThresholdMethod::Combined => {
                // Combined: max(delta_threshold, percentile_threshold)
                let local_max = window.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x));
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

/// Pick peaks from onset strength function using adaptive thresholding and temporal constraints.
///
/// This function identifies local maxima in the onset strength function that exceed
/// the adaptive threshold and satisfy temporal separation constraints. It implements
/// the complete peak picking pipeline with optional signal enhancement.
///
/// # Mathematical Theory
///
/// Peak picking is crucial for onset detection because:
/// - Converts continuous onset strength to discrete onset times
/// - Temporal constraints prevent multiple detections of the same event
/// - Signal enhancement improves detection in noisy conditions
///
/// The algorithm implements a two-stage process:
/// 1. **Candidate detection**: Find all local maxima above threshold
/// 2. **Temporal filtering**: Apply minimum separation constraints
///
/// # Arguments
///
/// * `onset_strength` - The onset strength function values
/// * `config` - Configuration for peak picking
///
/// # Returns
///
/// A vector of peak indices (sample positions) in the onset strength function
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::{PeakPickingConfig};
/// use audio_samples::operations::peak_picking::pick_peaks;
///
/// let onset_strength = vec![0.1f64, 0.3, 0.8, 0.2, 0.4, 0.9, 0.1];
/// let config = PeakPickingConfig::new();
/// let peaks = pick_peaks(&onset_strength, &config).unwrap();
/// // peaks contains indices of detected onset locations
/// ```
pub fn pick_peaks<F: RealFloat>(
    onset_strength: &[F],
    config: &PeakPickingConfig<F>,
) -> AudioSampleResult<Vec<usize>> {
    config.validate().map_err(|e| {
        AudioSampleError::Parameter(ParameterError::invalid_value(
            "peak_picking_config",
            format!("Invalid peak picking config: {}", e),
        ))
    })?;

    if onset_strength.is_empty() {
        return Ok(Vec::new());
    }

    // Step 1: Apply signal enhancement
    let mut processed_strength = onset_strength.to_vec();

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
            normalize_onset_strength(&processed_strength, config.normalization_method)?;
    }

    // Step 2: Compute adaptive threshold
    let thresholds = adaptive_threshold(&processed_strength, &config.adaptive_threshold)?;

    // Step 3: Find candidate peaks (local maxima above threshold)
    let mut candidates = Vec::new();

    for i in 1..processed_strength.len() - 1 {
        let current = processed_strength[i];
        let prev = processed_strength[i - 1];
        let next = processed_strength[i + 1];

        // Check if it's a local maximum above threshold
        if current > prev && current > next && current > thresholds[i] {
            candidates.push((i, current));
        }
    }

    // Step 4: Apply temporal constraints
    let peaks = apply_temporal_constraints(&candidates, config.min_peak_separation)?;

    Ok(peaks)
}

/// Apply pre-emphasis filtering to enhance transients.
///
/// Pre-emphasis is a high-pass filter that emphasizes high-frequency components
/// and transients in the signal. This is particularly useful for onset detection
/// as it enhances the characteristics that distinguish onsets from steady-state signals.
///
/// # Mathematical Theory
///
/// Pre-emphasis implements a first-order high-pass filter:
/// `y[n] = x[n] - α * x[n-1]`
///
/// Where α is the pre-emphasis coefficient (typically 0.95-0.99):
/// - Higher α values provide stronger emphasis
/// - Lower α values provide gentler emphasis
///
/// # Arguments
///
/// * `signal` - Input signal to filter
/// * `coeff` - Pre-emphasis coefficient (0.0-1.0)
///
/// # Returns
///
/// Filtered signal with enhanced transients
pub fn apply_pre_emphasis<F: RealFloat>(signal: &[F], coeff: F) -> AudioSampleResult<Vec<F>> {
    if !(F::zero()..=F::one()).contains(&coeff) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "pre_emphasis_coefficient",
            "Pre-emphasis coefficient must be between 0.0 and 1.0",
        )));
    }

    if signal.is_empty() {
        return Ok(Vec::new());
    }

    let mut filtered = Vec::with_capacity(signal.len());

    // First sample unchanged
    filtered.push(signal[0]);

    // Apply pre-emphasis filter
    for i in 1..signal.len() {
        filtered.push(signal[i] - coeff * signal[i - 1]);
    }

    Ok(filtered)
}

/// Apply median filtering for noise reduction.
///
/// Median filtering is a non-linear filter that replaces each sample with the
/// median of its neighborhood. It effectively reduces noise while preserving
/// edge structure, making it ideal for onset strength processing.
///
/// # Mathematical Theory
///
/// Median filtering: `y[n] = median(x[n-k:n+k])`
/// where k = (filter_length - 1) / 2
///
/// Properties:
/// - Preserves edges and peaks
/// - Removes impulse noise
/// - Non-linear operation
///
/// # Arguments
///
/// * `signal` - Input signal to filter
/// * `filter_length` - Length of median filter (must be odd)
///
/// # Returns
///
/// Filtered signal with reduced noise
pub fn apply_median_filter<F: RealFloat>(
    signal: &[F],
    filter_length: usize,
) -> AudioSampleResult<Vec<F>> {
    if filter_length == 0 || filter_length.is_multiple_of(2) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "filter_length",
            "Median filter length must be odd and greater than 0",
        )));
    }

    if signal.is_empty() {
        return Ok(Vec::new());
    }

    if filter_length == 1 {
        return Ok(signal.to_vec());
    }

    let mut filtered = Vec::with_capacity(signal.len());
    let half_length = filter_length / 2;

    for i in 0..signal.len() {
        // Determine window bounds
        let start = i.saturating_sub(half_length);
        let end = (i + half_length + 1).min(signal.len());

        // Extract window and compute median
        let mut window: Vec<F> = signal[start..end].to_vec();
        window.sort_by(|a, b| {
            match a.partial_cmp(b) {
                Some(order) => order,
                None => std::cmp::Ordering::Equal, // Handle NaN values gracefully
            }
        });
        let median = window[window.len() / 2];
        filtered.push(median);
    }

    Ok(filtered)
}

/// Normalize onset strength function.
///
/// Normalization ensures consistent detection thresholds across signals with
/// different amplitude levels. This is crucial for robust onset detection in
/// varied acoustic conditions.
///
/// # Arguments
///
/// * `onset_strength` - Input onset strength function
/// * `method` - Normalization method to apply
///
/// # Returns
///
/// Normalized onset strength function
pub fn normalize_onset_strength<F: RealFloat>(
    onset_strength: &[F],
    method: NormalizationMethod,
) -> AudioSampleResult<Vec<F>> {
    if onset_strength.is_empty() {
        return Ok(Vec::new());
    }

    match method {
        NormalizationMethod::Peak => {
            // Peak normalization: divide by maximum absolute value
            let max_abs = onset_strength
                .iter()
                .fold(F::zero(), |acc, &x| acc.max(x.abs()));

            if max_abs == F::zero() {
                return Ok(onset_strength.to_vec());
            }

            let normalized = onset_strength.iter().map(|&x| x / max_abs).collect();
            Ok(normalized)
        }
        NormalizationMethod::MinMax => {
            // Min-max normalization: scale to [0, 1]
            let min_val = onset_strength
                .iter()
                .fold(F::infinity(), |acc, &x| acc.min(x));
            let max_val = onset_strength
                .iter()
                .fold(F::neg_infinity(), |acc, &x| acc.max(x));

            if (max_val - min_val).abs() < F::epsilon() {
                return Ok(onset_strength.to_vec());
            }

            let normalized = onset_strength
                .iter()
                .map(|&x| (x - min_val) / (max_val - min_val))
                .collect();
            Ok(normalized)
        }
        NormalizationMethod::ZScore => {
            // Z-score normalization: zero mean, unit variance
            let mean = onset_strength.iter().fold(F::zero(), |acc: F, x| acc + *x)
                / to_precision::<F, _>(onset_strength.len());
            let variance = onset_strength
                .iter()
                .map(|&x| (x - mean).powi(2))
                .fold(F::zero(), |acc, x| acc + x)
                / to_precision::<F, _>(onset_strength.len());

            if variance == F::zero() {
                return Ok(onset_strength.to_vec());
            }

            let std_dev = variance.sqrt();
            let normalized = onset_strength
                .iter()
                .map(|&x| (x - mean) / std_dev)
                .collect();
            Ok(normalized)
        }
        NormalizationMethod::Mean => {
            // Mean normalization: subtract mean
            let mean = onset_strength.iter().fold(F::zero(), |acc: F, x| acc + *x)
                / to_precision::<F, _>(onset_strength.len());
            let normalized = onset_strength.iter().map(|&x| x - mean).collect();
            Ok(normalized)
        }
        NormalizationMethod::Median => {
            // Median normalization: subtract median
            let mut sorted = onset_strength.to_vec();
            sorted.sort_by(|a, b| {
                match a.partial_cmp(b) {
                    Some(order) => order,
                    None => std::cmp::Ordering::Equal, // Handle NaN values gracefully
                }
            });
            let median = sorted[sorted.len() / 2];
            let normalized = onset_strength.iter().map(|&x| x - median).collect();
            Ok(normalized)
        }
    }
}

/// Apply temporal constraints to candidate peaks.
///
/// This function implements temporal filtering to ensure detected peaks are
/// separated by minimum time intervals. When multiple peaks violate the
/// separation constraint, the one with highest strength is kept.
///
/// # Mathematical Theory
///
/// Temporal constraints prevent multiple detections of the same onset event:
/// - Minimum separation ensures peaks are spaced ≥ min_separation samples
/// - When conflicts occur, higher-strength peaks are preferred
/// - This is a greedy algorithm that processes peaks in strength order
///
/// # Arguments
///
/// * `candidates` - Vector of (index, strength) tuples for candidate peaks
/// * `min_separation` - Minimum separation between peaks in samples
///
/// # Returns
///
/// Vector of peak indices that satisfy temporal constraints
fn apply_temporal_constraints<F: RealFloat>(
    candidates: &[(usize, F)],
    min_separation: usize,
) -> AudioSampleResult<Vec<usize>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Sort candidates by strength (descending)
    let mut sorted_candidates = candidates.to_vec();
    sorted_candidates.sort_by(|a, b| {
        match b.1.partial_cmp(&a.1) {
            Some(order) => order,
            None => std::cmp::Ordering::Equal, // Handle NaN values gracefully
        }
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
    Ok(selected_peaks)
}

/// Smooth onset strength function using temporal processing.
///
/// This function applies temporal smoothing to the onset strength function
/// to reduce noise and improve peak detection. It combines multiple smoothing
/// techniques for optimal results.
///
/// # Arguments
///
/// * `onset_strength` - Input onset strength function
/// * `window_size` - Size of smoothing window
/// * `median_length` - Length of median filter (must be odd)
///
/// # Returns
///
/// Smoothed onset strength function
pub fn smooth_onset_strength(
    onset_strength: &[f64],
    window_size: usize,
    median_length: usize,
) -> AudioSampleResult<Vec<f64>> {
    if onset_strength.is_empty() {
        return Ok(Vec::new());
    }

    // Apply median filtering first
    let mut smoothed = apply_median_filter(onset_strength, median_length)?;

    // Apply moving average smoothing
    if window_size > 1 {
        smoothed = apply_moving_average(&smoothed, window_size)?;
    }

    Ok(smoothed)
}

/// Apply moving average smoothing.
///
/// This function applies a moving average filter to smooth the signal
/// while preserving overall trends and reducing high-frequency noise.
///
/// # Arguments
///
/// * `signal` - Input signal to smooth
/// * `window_size` - Size of moving average window
///
/// # Returns
///
/// Smoothed signal
fn apply_moving_average(signal: &[f64], window_size: usize) -> AudioSampleResult<Vec<f64>> {
    if window_size == 0 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "window_size",
            "Window size must be greater than 0",
        )));
    }

    if signal.is_empty() {
        return Ok(Vec::new());
    }

    if window_size == 1 {
        return Ok(signal.to_vec());
    }

    let mut smoothed = Vec::with_capacity(signal.len());
    let half_window = window_size / 2;

    for i in 0..signal.len() {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(signal.len());

        let sum: f64 = signal[start..end].iter().sum();
        let average = sum / (end - start) as f64;
        smoothed.push(average);
    }

    Ok(smoothed)
}

/// Compute percentile of a slice of values.
///
/// This helper function computes the percentile of a slice without
/// modifying the original data. It uses linear interpolation for
/// percentiles that fall between sample points.
///
/// # Arguments
///
/// * `values` - Slice of values to compute percentile from
/// * `percentile` - Percentile to compute (0.0-1.0)
///
/// # Returns
///
/// The percentile value
fn percentile<F: RealFloat>(values: &[F], percentile: F) -> F {
    if values.is_empty() {
        return F::zero();
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| {
        match a.partial_cmp(b) {
            Some(order) => order,
            None => std::cmp::Ordering::Equal, // Handle NaN values gracefully
        }
    });

    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }

    let index = percentile * to_precision::<F, _>(n - 1);
    let lower = index.floor().to_usize().expect("should not fail");
    let upper = index.ceil().to_usize().expect("should not fail");

    if lower == upper {
        sorted[lower]
    } else {
        let weight = index - to_precision::<F, _>(lower);
        sorted[lower] * (F::one() - weight) + sorted[upper] * weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::types::NormalizationMethod;

    #[test]
    fn test_adaptive_threshold_delta() {
        let onset_strength = vec![0.1, 0.3, 0.8, 0.2, 0.4, 0.9, 0.1];
        let config = AdaptiveThresholdConfig::delta(0.1, 3);

        let thresholds = adaptive_threshold(&onset_strength, &config).unwrap();
        assert_eq!(thresholds.len(), onset_strength.len());

        // All thresholds should be above minimum
        for &threshold in &thresholds {
            assert!(threshold >= config.min_threshold);
        }
    }

    #[test]
    fn test_adaptive_threshold_percentile() {
        let onset_strength = vec![0.1, 0.3, 0.8, 0.2, 0.4, 0.9, 0.1];
        let config = AdaptiveThresholdConfig::percentile(0.8, 3);

        let thresholds = adaptive_threshold(&onset_strength, &config).unwrap();
        assert_eq!(thresholds.len(), onset_strength.len());

        // Thresholds should be within bounds
        for &threshold in &thresholds {
            assert!(threshold >= config.min_threshold);
            assert!(threshold <= config.max_threshold);
        }
    }

    #[test]
    fn test_adaptive_threshold_combined() {
        let onset_strength = vec![0.1, 0.3, 0.8, 0.2, 0.4, 0.9, 0.1];
        let config = AdaptiveThresholdConfig::combined(0.1, 0.8, 3);

        let thresholds = adaptive_threshold(&onset_strength, &config).unwrap();
        assert_eq!(thresholds.len(), onset_strength.len());
    }

    #[test]
    fn test_pick_peaks_basic() {
        let onset_strength = vec![0.1, 0.3, 0.8, 0.2, 0.4, 0.9, 0.1];
        let mut config = PeakPickingConfig::new();

        // Configure for small test data
        config.adaptive_threshold.window_size = 3; // Small window for 7 samples
        config.min_peak_separation = 1; // Allow adjacent peaks for testing
        config.pre_emphasis = false; // Disable to avoid edge effects
        config.median_filter = false; // Disable to avoid smoothing
        config.normalize_onset_strength = false; // Keep original values

        let peaks = pick_peaks(&onset_strength, &config).unwrap();

        // Should detect some peaks (indices 2 and 5 should be peaks: 0.8 and 0.9)
        assert!(!peaks.is_empty());

        // All peaks should be valid indices
        for &peak in &peaks {
            assert!(peak < onset_strength.len());
        }
    }

    #[test]
    fn test_pick_peaks_with_constraints() {
        let onset_strength = vec![0.1, 0.5, 0.6, 0.7, 0.2, 0.8, 0.1];
        let mut config = PeakPickingConfig::new();
        config.min_peak_separation = 3;

        let peaks = pick_peaks(&onset_strength, &config).unwrap();

        // Check minimum separation constraint
        for i in 1..peaks.len() {
            assert!(peaks[i] - peaks[i - 1] >= config.min_peak_separation);
        }
    }

    #[test]
    fn test_pre_emphasis() {
        let signal: Vec<f64> = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let coeff: f64 = 0.97;

        let filtered = apply_pre_emphasis(&signal, coeff).unwrap();
        assert_eq!(filtered.len(), signal.len());

        // First sample should be unchanged
        assert_eq!(filtered[0], signal[0]);

        // Check pre-emphasis formula
        for i in 1..signal.len() {
            let expected = signal[i] - coeff * signal[i - 1];
            assert!((filtered[i] - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_median_filter() {
        let signal: Vec<f64> = vec![1.0, 5.0, 2.0, 8.0, 3.0]; // Contains outlier
        let filtered = apply_median_filter(&signal, 3).unwrap();

        assert_eq!(filtered.len(), signal.len());

        // Median filter should reduce the outlier effect
        assert!(filtered[1] < signal[1]); // Outlier should be reduced
    }

    #[test]
    fn test_normalize_onset_strength_peak() {
        let onset_strength: Vec<f64> = vec![0.1, 0.5, 1.0, 0.3];
        let normalized =
            normalize_onset_strength(&onset_strength, NormalizationMethod::Peak).unwrap();

        assert_eq!(normalized.len(), onset_strength.len());

        // Maximum absolute value should be 1.0
        let max_abs = normalized.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
        assert!((max_abs - 1.0f64).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_onset_strength_minmax() {
        let onset_strength: Vec<f64> = vec![0.1, 0.5, 1.0, 0.3];
        let normalized =
            normalize_onset_strength(&onset_strength, NormalizationMethod::MinMax).unwrap();

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
        let onset_strength = vec![0.1, 0.9, 0.1, 0.8, 0.2]; // Noisy signal
        let smoothed = smooth_onset_strength(&onset_strength, 3, 3).unwrap();

        assert_eq!(smoothed.len(), onset_strength.len());

        // Smoothed signal should have less variation
        let original_std = standard_deviation(&onset_strength);
        let smoothed_std = standard_deviation(&smoothed);
        assert!(smoothed_std <= original_std);
    }

    #[test]
    fn test_temporal_constraints() {
        let candidates = vec![(1, 0.8), (2, 0.6), (5, 0.9), (6, 0.7)];
        let min_separation = 2;

        let selected = apply_temporal_constraints(&candidates, min_separation).unwrap();

        // Should select peaks with highest strength that satisfy constraints
        assert!(!selected.is_empty());

        // Check separation constraint
        for i in 1..selected.len() {
            assert!(selected[i] - selected[i - 1] >= min_separation);
        }
    }

    #[test]
    fn test_percentile() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(percentile(&values, 0.0), 1.0);
        assert_eq!(percentile(&values, 1.0), 5.0);
        assert_eq!(percentile(&values, 0.5), 3.0);

        // Test interpolation
        let p25 = percentile(&values, 0.25);
        assert!(p25 > 1.0 && p25 < 3.0);
    }

    #[test]
    fn test_edge_cases() {
        // Empty input
        let empty: Vec<f64> = vec![];
        let config = AdaptiveThresholdConfig::new();

        assert!(adaptive_threshold(&empty, &config).unwrap().is_empty());
        assert!(
            pick_peaks(&empty, &PeakPickingConfig::new())
                .unwrap()
                .is_empty()
        );

        // Single sample
        let single = vec![1.0];
        let thresholds = adaptive_threshold(&single, &config).unwrap();
        assert_eq!(thresholds.len(), 1);

        // All zeros
        let zeros = vec![0.0, 0.0, 0.0];
        let normalized = normalize_onset_strength(&zeros, NormalizationMethod::Peak).unwrap();
        assert_eq!(normalized, zeros);
    }

    #[test]
    fn test_config_validation() {
        // Invalid delta
        let mut config = AdaptiveThresholdConfig::new();
        config.delta = -0.1;
        assert!(config.validate().is_err());

        // Invalid percentile
        config = AdaptiveThresholdConfig::new();
        config.percentile = 1.5;
        assert!(config.validate().is_err());

        // Invalid window size
        config = AdaptiveThresholdConfig::new();
        config.window_size = 0;
        assert!(config.validate().is_err());

        // Invalid threshold bounds
        config = AdaptiveThresholdConfig::new();
        config.min_threshold = 0.5;
        config.max_threshold = 0.3;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_peak_picking_presets() {
        let onset_strength = vec![0.1, 0.3, 0.8, 0.2, 0.4, 0.9, 0.1];

        // Test different presets
        let configs = vec![
            PeakPickingConfig::new(),
            PeakPickingConfig::music(),
            PeakPickingConfig::speech(),
            PeakPickingConfig::drums(),
        ];

        for config in configs {
            let peaks = pick_peaks(&onset_strength, &config).unwrap();
            // All presets should produce valid results
            for &peak in &peaks {
                assert!(peak < onset_strength.len());
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
