//! Audio signal comparison and similarity metrics.
//!
//! This module provides a small collection of signal-level comparison operators for
//! analysing similarity, error, and alignment between audio signals. The exposed functions
//! operate on [`AudioSamples`] and return scalar metrics or derived signals that are suitable
//! for validation, evaluation, diagnostics, and lightweight analysis workflows.
//!
//! The primary purpose of this module is to centralise common comparison semantics.
//!
//! # Usage
//!
//! Typical usage consists of passing two compatible audio signals into a metric function
//! and interpreting the returned scalar or aligned signal. All comparison functions require
//! matching dimensionality and compatible channel layouts unless explicitly documented
//! otherwise.
//!
//! ```rust
//! use audio_samples::utils::comparison::{correlation, mse};
//! use audio_samples::utils::generation::sine_wave;
//! use audio_samples::sample_rate;
//! use std::time::Duration;
//!
//! let sr = sample_rate!(44100);
//! let a = sine_wave::<f32>(440.0, Duration::from_millis(100), sr, 1.0);
//! let b = sine_wave::<f32>(440.0, Duration::from_millis(100), sr, 1.0);
//!
//! let corr: f64 = correlation(&a, &b).unwrap();
//! let error: f64 = mse(&a, &b).unwrap();
//!
//! assert!((corr - 1.0).abs() < 1e-6);
//! assert!(error < 1e-10);
//! ```
use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, LayoutError,
    ParameterError, traits::StandardSample,
};
use ndarray::{Array1, ArrayView1};

/// Computes the Pearson correlation coefficient between two audio signals.
///
/// This function measures linear similarity between two signals on a per-sample basis and
/// returns a scalar correlation coefficient in the range `[-1, 1]`. Positive values indicate
/// positive correlation, negative values indicate inverse correlation, and values near zero
/// indicate weak linear relationship.
///
/// For mono signals, the correlation is computed directly over the single channel.\
/// For multi-channel signals, the correlation is computed independently for each channel and
/// the results are averaged to produce a single scalar score.
///
/// # Arguments
///
/// * `a`\
///   The first audio signal.
///
/// * `b`\
///   The second audio signal.
///
/// Both signals must have identical channel counts and the same number of samples per
/// channel.
///
/// # Returns
///
/// A scalar correlation coefficient in the range `[-1, 1]`, expressed in the requested
/// floating-point type.
///
/// # Errors
///
/// Returns an error in the following cases:
///
/// * The two signals have different channel counts.
/// * The two signals have different numbers of samples per channel.
/// * The underlying sample layout is non-contiguous and cannot be viewed as a slice.
/// * The channel configuration of the two signals does not match (mono vs multi-channel).
///
/// # Behavioural Guarantees
///
/// * The returned value lies in the closed interval `[-1, 1]` for all finite inputs.
/// * For identical signals, the returned value is `1.0` (up to floating-point rounding).
/// * For multi-channel inputs, the result is the arithmetic mean of per-channel correlations.
/// * If either signal has zero variance over a channel, the correlation for that channel is
///   defined as `0.0`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::comparison::correlation;
/// use audio_samples::utils::generation::sine_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let sr = sample_rate!(44100);
/// let a = sine_wave::<f32>(440.0, Duration::from_millis(100), sr, 1.0);
/// let b = sine_wave::<f32>(440.0, Duration::from_millis(100), sr, 1.0);
///
/// let corr = correlation(&a, &b).unwrap();
/// assert!((corr - 1.0).abs() < 1e-6); // identical signals → correlation 1.0
/// ```
#[inline]
pub fn correlation<T>(a: &AudioSamples<T>, b: &AudioSamples<T>) -> AudioSampleResult<f64>
where
    T: StandardSample,
{
    if a.num_channels() != b.num_channels() || a.samples_per_channel() != b.samples_per_channel() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "audio_signals",
            "Signals must have the same dimensions for correlation",
        )));
    }

    let a_f = a.as_float();
    let b_f = b.as_float();

    match (a_f.as_mono(), b_f.as_mono()) {
        (Some(a_mono), Some(b_mono)) => {
            let corr = correlation_1d(&a_mono.view(), &b_mono.view())?;
            Ok(corr)
        }
        (Some(_), None) | (None, Some(_)) => {
            Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Signals must have the same channel configuration",
            )))
        }
        (None, None) => {
            // Multi-channel correlation - compute average correlation across channels
            let a_multi = a_f.as_multi_channel().ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Must be multi-channel audio",
                ))
            })?;
            let b_multi = b_f.as_multi_channel().ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Must be multi-channel audio",
                ))
            })?;

            let mut correlations = Vec::new();
            for i in 0..a_multi.nrows().get() {
                let a_channel = a_multi.row(i);
                let b_channel = b_multi.row(i);
                let corr = correlation_1d_slice(
                    a_channel.as_slice().ok_or_else(|| {
                        AudioSampleError::Layout(LayoutError::NonContiguous {
                            operation: "signal processing".to_string(),
                            layout_type: "non-contiguous multi-channel samples".to_string(),
                        })
                    })?,
                    b_channel.as_slice().ok_or_else(|| {
                        AudioSampleError::Layout(LayoutError::NonContiguous {
                            operation: "signal processing".to_string(),
                            layout_type: "non-contiguous multi-channel samples".to_string(),
                        })
                    })?,
                )?;
                correlations.push(corr);
            }

            Ok(correlations.iter().fold(0.0, |acc, x| acc + *x) / correlations.len() as f64)
        }
    }
}

/// Computes the mean squared error (MSE) between two audio signals.
///
/// This function measures the average squared per-sample difference between two signals.
/// Lower values indicate higher similarity, with a value of `0.0` indicating identical
/// signals under exact arithmetic. The metric is unnormalised and scales quadratically with
/// signal amplitude.
///
/// For mono signals, the MSE is computed directly over the single channel.\
/// For multi-channel signals, the MSE is computed independently for each channel and the
/// results are averaged to produce a single scalar value.
///
/// # Arguments
///
/// * `a`\
///   The first audio signal.
///
/// * `b`\
///   The second audio signal.
///
/// Both signals must have identical channel counts and the same number of samples per
/// channel.
///
/// # Returns
///
/// The mean squared error as a non-negative scalar value expressed in the requested
/// floating-point type.
///
/// # Errors
///
/// Returns an error in the following cases:
///
/// * The two signals have different channel counts.
/// * The two signals have different numbers of samples per channel.
/// * The underlying sample layout is non-contiguous and cannot be viewed as a slice.
/// * The channel configuration of the two signals does not match (mono vs multi-channel).
///
/// # Behavioural Guarantees
///
/// * The returned value is always greater than or equal to `0.0` for all finite inputs.
/// * For identical signals, the returned value is `0.0` up to floating-point rounding.
/// * For multi-channel inputs, the result is the arithmetic mean of per-channel MSE values.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::comparison::mse;
/// use audio_samples::utils::generation::sine_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let sr = sample_rate!(44100);
/// let a = sine_wave::<f32>(440.0, Duration::from_millis(100), sr, 1.0);
/// let b = sine_wave::<f32>(440.0, Duration::from_millis(100), sr, 1.0);
///
/// let error = mse(&a, &b).unwrap();
/// assert!(error < 1e-10); // identical signals → zero MSE
/// ```
#[inline]
pub fn mse<T>(a: &AudioSamples<T>, b: &AudioSamples<T>) -> AudioSampleResult<f64>
where
    T: StandardSample,
{
    if a.num_channels() != b.num_channels() || a.samples_per_channel() != b.samples_per_channel() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "audio_signals",
            "Signals must have the same dimensions for MSE",
        )));
    }

    let a_f = a.as_float();
    let b_f = b.as_float();

    match (a_f.as_mono(), b_f.as_mono()) {
        (Some(a_mono), Some(b_mono)) => mse_1d(&a_mono.view(), &b_mono.view()),
        (Some(_), None) | (None, Some(_)) => {
            Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Signals must have the same channel configuration",
            )))
        }
        (None, None) => {
            // Multi-channel MSE - compute average MSE across channels
            let a_multi = a_f.as_multi_channel().ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Must be multi-channel audio",
                ))
            })?;
            let b_multi = b_f.as_multi_channel().ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Must be multi-channel audio",
                ))
            })?;

            let mut mses = Vec::new();
            for i in 0..a_multi.nrows().get() {
                let a_channel = a_multi.row(i);
                let b_channel = b_multi.row(i);
                let mse = mse_1d_slice(
                    a_channel.as_slice().ok_or_else(|| {
                        AudioSampleError::Layout(LayoutError::NonContiguous {
                            operation: "signal processing".to_string(),
                            layout_type: "non-contiguous multi-channel samples".to_string(),
                        })
                    })?,
                    b_channel.as_slice().ok_or_else(|| {
                        AudioSampleError::Layout(LayoutError::NonContiguous {
                            operation: "signal processing".to_string(),
                            layout_type: "non-contiguous multi-channel samples".to_string(),
                        })
                    })?,
                )?;
                mses.push(mse);
            }

            Ok(mses.iter().fold(0.0, |acc, x| acc + *x) / mses.len() as f64)
        }
    }
}

/// Computes the signal-to-noise ratio (SNR) between a signal and a noise signal.
///
/// This function measures the ratio between the average power of a signal and the average
/// power of a corresponding noise signal, expressed in decibels. Higher values indicate
/// greater dominance of the signal relative to noise.
///
/// The function assumes that `signal` and `noise` are already aligned and represent
/// compatible components. No validation is performed to determine whether the inputs
/// actually correspond to a signal–noise decomposition.
///
/// For mono inputs, power is computed over the single channel.\
/// For multi-channel inputs, power is computed over all samples across all channels and
/// aggregated into a single scalar value. No per-channel SNR values are returned.
///
/// # Arguments
///
/// * `signal`\
///   The signal component.
///
/// * `noise`\
///   The noise component.
///
/// Both inputs must have identical channel counts and the same number of samples per
/// channel.
///
/// # Returns
///
/// The signal-to-noise ratio in decibels. Larger values indicate higher signal quality.
///
/// If the estimated noise power is zero, the function returns positive infinity.
///
/// # Errors
///
/// Returns an error in the following cases:
///
/// * The two inputs have different channel counts.
/// * The two inputs have different numbers of samples per channel.
/// * The underlying sample layout is non-contiguous and cannot be viewed as a slice.
/// * The channel configuration of the two signals does not match (mono vs multi-channel).
///
/// # Behavioural Guarantees
///
/// * For finite non-zero noise power, the returned value is finite.
/// * If the noise power is exactly zero, the returned value is positive infinity.
/// * The returned value increases monotonically with increasing signal power for fixed
///   noise power.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::comparison::snr;
/// use audio_samples::utils::generation::sine_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let sr = sample_rate!(44100);
/// // Signal with amplitude 1.0, noise with amplitude 0.01.
/// let signal = sine_wave::<f32>(440.0, Duration::from_millis(100), sr, 1.0);
/// let noise  = sine_wave::<f32>(880.0, Duration::from_millis(100), sr, 0.01);
///
/// let db = snr(&signal, &noise).unwrap();
/// assert!(db > 30.0); // high-power signal relative to low-power noise → high SNR
/// ```
#[inline]
pub fn snr<T>(signal: &AudioSamples<T>, noise: &AudioSamples<T>) -> AudioSampleResult<f64>
where
    T: StandardSample,
{
    if signal.num_channels() != noise.num_channels()
        || signal.samples_per_channel() != noise.samples_per_channel()
    {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "audio_signals",
            "Signal and noise must have the same dimensions for SNR",
        )));
    }

    let signal_f = signal.as_float();
    let noise_f = noise.as_float();

    // Calculate signal power
    let signal_power = if let Some(mono) = signal_f.as_mono() {
        mono.iter().map(|&x| x * x).fold(0.0, |acc, x| acc + x) / mono.len().get() as f64
    } else {
        let multi = signal_f.as_multi_channel().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Must be multi-channel audio",
            ))
        })?;
        multi.iter().map(|&x| x * x).fold(0.0, |acc, x| acc + x) / multi.len().get() as f64
    };

    // Calculate noise power
    let noise_power = if let Some(mono) = noise_f.as_mono() {
        mono.iter().map(|x| *x * *x).fold(0.0, |acc, x| acc + x) / mono.len().get() as f64
    } else {
        let multi = noise_f.as_multi_channel().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Must be multi-channel audio",
            ))
        })?;
        multi.iter().map(|&x| x * x).fold(0.0, |acc, x| acc + x) / multi.len().get() as f64
    };

    if noise_power == 0.0 {
        return Ok(f64::INFINITY);
    }

    let snr_db = 10.0 * (signal_power / noise_power).log10();
    Ok(snr_db)
}
/// Aligns two audio signals by estimating a time offset that maximises correlation.
///
/// This function attempts to temporally align `signal` to `reference` by searching for a
/// non-negative sample offset that maximises cross-correlation between the two signals. The
/// returned signal is a shifted version of the input signal padded at the start, together
/// with the estimated offset in samples.
///
/// This is a heuristic alignment utility intended for coarse synchronisation, diagnostics,
/// and preprocessing. It does not guarantee globally optimal alignment, sub-sample accuracy,
/// or robustness under heavy noise, reverberation, or non-linear distortion.
///
/// # Channel Handling
///
/// * For mono signals, alignment is computed directly on the single channel.
/// * For multi-channel signals, all channels are averaged to a single derived signal before
///   alignment. The estimated offset is then applied uniformly to all channels.
///
/// No per-channel alignment is performed. Channel-specific delays are not detected.
///
/// # Search Behaviour
///
/// * Only non-negative offsets are considered. The signal is shifted forward in time only.
/// * The maximum offset searched is half of the shorter signal length.
/// * For each candidate offset, correlation is computed over the overlapping region only.
///
/// These limits are part of the public contract and constrain the class of alignments that
/// can be detected.
///
/// # Padding Semantics
///
/// When a positive offset is selected, the aligned signal is padded at the start using the
/// default value of the sample type. This is a structural padding operation rather than an
/// explicit silence model.
///
/// # Arguments
///
/// * `reference`\
///   The reference signal that defines the target alignment.
///
/// * `signal`\
///   The signal to be aligned to the reference.
///
/// Both signals must have identical channel counts.
///
/// # Returns
///
/// A tuple `(aligned_signal, offset_samples)` where:
///
/// * `aligned_signal` is a newly allocated signal shifted by the estimated offset.
/// * `offset_samples` is the number of samples by which the signal was shifted.
///
/// # Errors
///
/// Returns an error in the following cases:
///
/// * The two signals have different channel counts.
/// * The channel configuration of the two signals does not match (mono vs multi-channel).
/// * The underlying sample layout is non-contiguous and cannot be viewed as a slice.
/// * Internal array construction fails due to invalid shape assumptions.
///
/// # Behavioural Guarantees
///
/// * The returned offset is always greater than or equal to zero.
/// * If no offset improves correlation, the returned offset is zero and the original signal
///   is returned unchanged (cloned).
/// * The alignment decision is deterministic for fixed inputs.
/// * The function allocates a new signal when a non-zero offset is applied.
///
/// # Limitations
///
/// This function does not:
///
/// * Search negative offsets or bidirectional shifts.
/// * Perform sub-sample alignment or interpolation.
/// * Account for phase offsets, time stretching, or sample-rate mismatch.
/// * Preserve original sample padding semantics beyond default value initialisation.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::comparison::align_signals;
/// use audio_samples::AudioSamples;
///
/// fn example<T: audio_samples::traits::StandardSample>(
///     reference: AudioSamples<'static, T>,
///      signal: AudioSamples<'static, T>,
///  ) {
/// let (aligned, offset) = align_signals::<T>(&reference, &signal).unwrap();
///
/// println!("Estimated offset: {offset} samples");
/// assert!(offset >= 0);
///  }
/// ```
#[inline]
pub fn align_signals<T>(
    reference: &AudioSamples<'_, T>,
    signal: &AudioSamples<'_, T>,
) -> AudioSampleResult<(AudioSamples<'static, T>, usize)>
where
    T: StandardSample,
{
    if reference.num_channels() != signal.num_channels() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "audio_channels",
            "Signals must have the same number of channels for alignment",
        )));
    }

    let sample_rate = signal.sample_rate();
    let ref_f = reference.as_float();
    let sig_f = signal.as_float();

    // For simplicity, we'll work with the first channel for mono or the average for multi-channel
    let (ref_data, sig_data) = match (ref_f.as_mono(), sig_f.as_mono()) {
        (Some(ref_mono), Some(sig_mono)) => (ref_mono.to_vec(), sig_mono.to_vec()),
        (None, None) => {
            // Average across channels for multi-channel signals
            let ref_multi = ref_f.as_multi_channel().ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Must be multi-channel audio",
                ))
            })?;
            let sig_multi = sig_f.as_multi_channel().ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Must be multi-channel audio",
                ))
            })?;

            let ref_avg: Vec<f64> = (0..ref_multi.ncols().get())
                .map(|i| {
                    ref_multi.column(i).iter().fold(0.0, |acc, x| acc + *x)
                        / ref_multi.nrows().get() as f64
                })
                .collect();

            let sig_avg: Vec<f64> = (0..sig_multi.ncols().get())
                .map(|i| {
                    sig_multi.column(i).iter().fold(0.0, |acc, x| acc + *x)
                        / sig_multi.nrows().get() as f64
                })
                .collect();

            (ref_avg, sig_avg)
        }
        _ => {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Signals must have the same channel configuration",
            )));
        }
    };

    // Find the best alignment using cross-correlation
    let max_offset = ref_data.len().min(sig_data.len()) / 2;
    let mut best_offset = 0;
    let mut best_correlation = f64::NEG_INFINITY;

    for offset in 0..max_offset {
        let correlation = if offset < sig_data.len() {
            let end = (ref_data.len() - offset).min(sig_data.len() - offset);
            correlation_1d_slice(
                &ref_data[offset..offset + end],
                &sig_data[offset..offset + end],
            )?
        } else {
            0.0
        };

        if correlation > best_correlation {
            best_correlation = correlation;
            best_offset = offset;
        }
    }

    // Create aligned signal by shifting
    let aligned_signal = if best_offset > 0 {
        if let Some(mono) = signal.as_mono() {
            let mut aligned_data = vec![T::default(); best_offset];
            aligned_data.extend_from_slice(
                &mono.as_slice().ok_or_else(|| {
                    AudioSampleError::Layout(LayoutError::NonContiguous {
                        operation: "signal alignment".to_string(),
                        layout_type: "non-contiguous mono samples".to_string(),
                    })
                })?[..mono.len().get() - best_offset],
            );
            let aligned_array = Array1::from_vec(aligned_data);
            AudioSamples::new_mono(aligned_array, sample_rate)?
        } else {
            let multi = signal.as_multi_channel().ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Must be multi-channel audio",
                ))
            })?;
            let mut aligned_data = Vec::new();
            for i in 0..multi.nrows().get() {
                let mut row = vec![T::default(); best_offset];
                row.extend_from_slice(
                    &multi.row(i).as_slice().ok_or_else(|| {
                        AudioSampleError::Layout(LayoutError::NonContiguous {
                            operation: "signal alignment".to_string(),
                            layout_type: "non-contiguous multi-channel samples".to_string(),
                        })
                    })?[..multi.ncols().get() - best_offset],
                );
                aligned_data.push(row);
            }
            let aligned_array = ndarray::Array2::from_shape_vec(
                (aligned_data.len(), aligned_data[0].len()),
                aligned_data.into_iter().flatten().collect(),
            )
            .map_err(|e| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "array_shape",
                    format!("Array shape error: {e}"),
                ))
            })?;
            AudioSamples::new_multi_channel(aligned_array, sample_rate)?
        }
    } else {
        signal.clone().into_owned()
    };

    Ok((aligned_signal, best_offset))
}

// Helper functions for 1D correlation and MSE
fn correlation_1d(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> AudioSampleResult<f64> {
    correlation_1d_slice(
        a.as_slice().ok_or_else(|| {
            AudioSampleError::Layout(LayoutError::NonContiguous {
                operation: "correlation calculation".to_string(),
                layout_type: "non-contiguous mono samples".to_string(),
            })
        })?,
        b.as_slice().ok_or_else(|| {
            AudioSampleError::Layout(LayoutError::NonContiguous {
                operation: "correlation calculation".to_string(),
                layout_type: "non-contiguous mono samples".to_string(),
            })
        })?,
    )
}

fn correlation_1d_slice(a: &[f64], b: &[f64]) -> AudioSampleResult<f64> {
    if a.len() != b.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "array_length",
            "Arrays must have the same length for correlation",
        )));
    }

    let n = a.len();
    let mean_a = a.iter().fold(0.0, |acc, x| acc + *x) / n as f64;
    let mean_b = b.iter().fold(0.0, |acc, x| acc + *x) / n as f64;

    let mut num = 0.0;
    let mut den_a = 0.0;
    let mut den_b = 0.0;

    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        num += dx * dy;
        den_a += dx * dx;
        den_b += dy * dy;
    }

    let denominator = (den_a * den_b).sqrt();
    if denominator == 0.0 {
        Ok(0.0)
    } else {
        Ok(num / denominator)
    }
}

fn mse_1d(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> AudioSampleResult<f64> {
    if a.len() != b.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "array_length",
            "Arrays must have the same length for MSE",
        )));
    }

    let n = a.len();
    let sum_squared_diff: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .fold(0.0, |acc, val| acc + val);

    Ok(sum_squared_diff / n as f64)
}

fn mse_1d_slice(a: &[f64], b: &[f64]) -> AudioSampleResult<f64> {
    if a.len() != b.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "array_length",
            "Arrays must have the same length for MSE",
        )));
    }
    let n = a.len();

    let sum_squared_diff: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .fold(0.0, |acc, val| acc + val)
        / n as f64;

    Ok(sum_squared_diff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample_rate;
    use approx_eq::assert_approx_eq;
    use non_empty_slice::non_empty_vec;

    #[test]
    fn test_correlation_identical_signals() {
        let data: non_empty_slice::NonEmptyVec<f64> = non_empty_vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let audio1: AudioSamples<'static, f64> =
            AudioSamples::from_mono_vec::<f64>(data.clone(), sample_rate!(44100));
        let audio2: AudioSamples<'static, f64> =
            AudioSamples::from_mono_vec::<f64>(data, sample_rate!(44100));

        let corr: f64 = correlation(&audio1, &audio2).unwrap();
        assert_approx_eq!(corr, 1.0, 1e-10);
    }

    #[test]
    fn test_correlation_opposite_signals() {
        let audio1: AudioSamples<'static, f64> = AudioSamples::from_mono_vec::<f64>(
            non_empty_vec![1.0f64, 2.0, 3.0, 4.0, 5.0],
            sample_rate!(44100),
        );
        let audio2 = AudioSamples::from_mono_vec::<f64>(
            non_empty_vec![-1.0f64, -2.0, -3.0, -4.0, -5.0],
            sample_rate!(44100),
        );

        let corr: f64 = correlation(&audio1, &audio2).unwrap();
        assert_approx_eq!(corr, -1.0, 1e-10);
    }

    #[test]
    fn test_mse_identical_signals() {
        let audio1: AudioSamples<'static, f64> = AudioSamples::from_mono_vec::<f64>(
            non_empty_vec![1.0f64, 2.0, 3.0, 4.0, 5.0],
            sample_rate!(44100),
        );
        let audio2 = AudioSamples::from_mono_vec::<f64>(
            non_empty_vec![1.0f64, 2.0, 3.0, 4.0, 5.0],
            sample_rate!(44100),
        );

        let mse_val = mse(&audio1, &audio2).unwrap();
        assert_approx_eq!(mse_val, 0.0_f64, 1e-10);
    }

    #[test]
    fn test_snr_calculation() {
        let signal: AudioSamples<'static, f64> = AudioSamples::from_mono_vec::<f64>(
            non_empty_vec![1.0f64, 2.0, 3.0, 4.0, 5.0],
            sample_rate!(44100),
        );
        let noise = AudioSamples::from_mono_vec::<f64>(
            non_empty_vec![0.1f64, 0.2, 0.1, 0.2, 0.1],
            sample_rate!(44100),
        );

        let snr_val: f64 = snr(&signal, &noise).unwrap();
        assert!(snr_val > 0.0_f64); // Signal should have higher power than noise
    }

    #[test]
    fn test_align_signals_no_offset() {
        let data = non_empty_vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let reference = AudioSamples::from_mono_vec::<f64>(data.clone(), sample_rate!(44100));
        let signal = AudioSamples::from_mono_vec::<f64>(data, sample_rate!(44100));

        let (aligned, offset) = align_signals::<f64>(&reference, &signal).unwrap();
        assert_eq!(offset, 0);
        assert_eq!(aligned.samples_per_channel(), signal.samples_per_channel());
    }
}
