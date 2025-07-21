//! Audio comparison and similarity utilities.
//!
//! This module provides functions for comparing audio signals and measuring
//! their similarity using various metrics.

use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ConvertTo,
    I24,
};
use ndarray::Array1;

/// Computes the Pearson correlation coefficient between two audio signals.
///
/// The correlation coefficient ranges from -1 to 1, where:
/// - 1 indicates perfect positive correlation
/// - 0 indicates no correlation
/// - -1 indicates perfect negative correlation
///
/// # Arguments
/// * `a` - First audio signal
/// * `b` - Second audio signal
///
/// # Returns
/// Correlation coefficient as f64
///
/// # Errors
/// Returns an error if the signals have different lengths or channels.
pub fn correlation<T: AudioSample>(
    a: &AudioSamples<T>,
    b: &AudioSamples<T>,
) -> AudioSampleResult<f64>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    if a.channels() != b.channels() || a.samples_per_channel() != b.samples_per_channel() {
        return Err(AudioSampleError::InvalidParameter(
            "Signals must have the same dimensions for correlation".to_string(),
        ));
    }

    let a_f64 = a.as_f64()?;
    let b_f64 = b.as_f64()?;

    match (a_f64.as_mono(), b_f64.as_mono()) {
        (Some(a_mono), Some(b_mono)) => {
            let corr = correlation_1d(a_mono, b_mono)?;
            Ok(corr)
        }
        (Some(_), None) | (None, Some(_)) => Err(AudioSampleError::InvalidParameter(
            "Signals must have the same channel configuration".to_string(),
        )),
        (None, None) => {
            // Multi-channel correlation - compute average correlation across channels
            let a_multi = a_f64
                .as_multi_channel()
                .ok_or(AudioSampleError::InvalidInput {
                    msg: "Must be multi-channel audio".to_string(),
                })?;
            let b_multi = b_f64
                .as_multi_channel()
                .ok_or(AudioSampleError::InvalidInput {
                    msg: "Must be multi-channel audio".to_string(),
                })?;

            let mut correlations = Vec::new();
            for i in 0..a_multi.nrows() {
                let a_channel = a_multi.row(i);
                let b_channel = b_multi.row(i);
                let corr = correlation_1d_slice(
                    a_channel
                        .as_slice()
                        .ok_or(AudioSampleError::ArrayLayoutError {
                            message: "Multi-channel samples must be contiguous".to_string(),
                        })?,
                    b_channel
                        .as_slice()
                        .ok_or(AudioSampleError::ArrayLayoutError {
                            message: "Multi-channel samples must be contiguous".to_string(),
                        })?,
                )?;
                correlations.push(corr);
            }

            Ok(correlations.iter().sum::<f64>() / correlations.len() as f64)
        }
    }
}

/// Computes the Mean Squared Error (MSE) between two audio signals.
///
/// MSE measures the average squared difference between corresponding samples.
/// Lower values indicate higher similarity.
///
/// # Arguments
/// * `a` - First audio signal
/// * `b` - Second audio signal
///
/// # Returns
/// MSE value as f64
pub fn mse<T: AudioSample>(a: &AudioSamples<T>, b: &AudioSamples<T>) -> AudioSampleResult<f64>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    if a.channels() != b.channels() || a.samples_per_channel() != b.samples_per_channel() {
        return Err(AudioSampleError::InvalidParameter(
            "Signals must have the same dimensions for MSE".to_string(),
        ));
    }

    let a_f64 = a.as_f64()?;
    let b_f64 = b.as_f64()?;

    match (a_f64.as_mono(), b_f64.as_mono()) {
        (Some(a_mono), Some(b_mono)) => mse_1d(a_mono, b_mono),
        (Some(_), None) | (None, Some(_)) => Err(AudioSampleError::InvalidParameter(
            "Signals must have the same channel configuration".to_string(),
        )),
        (None, None) => {
            // Multi-channel MSE - compute average MSE across channels
            let a_multi = a_f64
                .as_multi_channel()
                .ok_or(AudioSampleError::InvalidInput {
                    msg: "Must be multi-channel audio".to_string(),
                })?;
            let b_multi = b_f64
                .as_multi_channel()
                .ok_or(AudioSampleError::InvalidInput {
                    msg: "Must be multi-channel audio".to_string(),
                })?;

            let mut mses = Vec::new();
            for i in 0..a_multi.nrows() {
                let a_channel = a_multi.row(i);
                let b_channel = b_multi.row(i);
                let mse = mse_1d_slice(
                    a_channel
                        .as_slice()
                        .ok_or(AudioSampleError::ArrayLayoutError {
                            message: "Multi-channel samples must be contiguous".to_string(),
                        })?,
                    b_channel
                        .as_slice()
                        .ok_or(AudioSampleError::ArrayLayoutError {
                            message: "Multi-channel samples must be contiguous".to_string(),
                        })?,
                )?;
                mses.push(mse);
            }

            Ok(mses.iter().sum::<f64>() / mses.len() as f64)
        }
    }
}

/// Computes the Signal-to-Noise Ratio (SNR) between a signal and noise.
///
/// SNR is expressed in decibels (dB) and measures the ratio of signal power
/// to noise power. Higher values indicate better signal quality.
///
/// # Arguments
/// * `signal` - The clean signal
/// * `noise` - The noise component
///
/// # Returns
/// SNR value in dB
pub fn snr<T: AudioSample>(
    signal: &AudioSamples<T>,
    noise: &AudioSamples<T>,
) -> AudioSampleResult<f64>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    if signal.channels() != noise.channels()
        || signal.samples_per_channel() != noise.samples_per_channel()
    {
        return Err(AudioSampleError::InvalidParameter(
            "Signal and noise must have the same dimensions for SNR".to_string(),
        ));
    }

    let signal_f64 = signal.as_f64()?;
    let noise_f64 = noise.as_f64()?;

    // Calculate signal power
    let signal_power = match signal_f64.as_mono() {
        Some(mono) => mono.iter().map(|&x| x * x).sum::<f64>() / mono.len() as f64,
        None => {
            let multi = signal_f64
                .as_multi_channel()
                .ok_or(AudioSampleError::InvalidInput {
                    msg: "Must be multi-channel audio".to_string(),
                })?;
            multi.iter().map(|&x| x * x).sum::<f64>() / multi.len() as f64
        }
    };

    // Calculate noise power
    let noise_power = match noise_f64.as_mono() {
        Some(mono) => mono.iter().map(|&x| x * x).sum::<f64>() / mono.len() as f64,
        None => {
            let multi = noise_f64
                .as_multi_channel()
                .ok_or(AudioSampleError::InvalidInput {
                    msg: "Must be multi-channel audio".to_string(),
                })?;
            multi.iter().map(|&x| x * x).sum::<f64>() / multi.len() as f64
        }
    };

    if noise_power == 0.0 {
        return Ok(f64::INFINITY);
    }

    let snr_db = 10.0 * (signal_power / noise_power).log10();
    Ok(snr_db)
}

/// Aligns two audio signals by finding the best time offset.
///
/// This function finds the time offset that maximizes the cross-correlation
/// between the two signals, effectively aligning them in time.
///
/// # Arguments
/// * `reference` - The reference signal
/// * `signal` - The signal to align
///
/// # Returns
/// A tuple of (aligned_signal, offset_samples) where offset_samples is the
/// number of samples by which the signal was shifted.
pub fn align_signals<T: AudioSample>(
    reference: &AudioSamples<T>,
    signal: &AudioSamples<T>,
) -> AudioSampleResult<(AudioSamples<T>, usize)>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    if reference.channels() != signal.channels() {
        return Err(AudioSampleError::InvalidParameter(
            "Signals must have the same number of channels for alignment".to_string(),
        ));
    }

    let ref_f64 = reference.as_f64()?;
    let sig_f64 = signal.as_f64()?;

    // For simplicity, we'll work with the first channel for mono or the average for multi-channel
    let (ref_data, sig_data) = match (ref_f64.as_mono(), sig_f64.as_mono()) {
        (Some(ref_mono), Some(sig_mono)) => (ref_mono.to_vec(), sig_mono.to_vec()),
        (None, None) => {
            // Average across channels for multi-channel signals
            let ref_multi = ref_f64
                .as_multi_channel()
                .ok_or(AudioSampleError::InvalidInput {
                    msg: "Must be multi-channel audio".to_string(),
                })?;
            let sig_multi = sig_f64
                .as_multi_channel()
                .ok_or(AudioSampleError::InvalidInput {
                    msg: "Must be multi-channel audio".to_string(),
                })?;

            let ref_avg: Vec<f64> = (0..ref_multi.ncols())
                .map(|i| ref_multi.column(i).iter().sum::<f64>() / ref_multi.nrows() as f64)
                .collect();

            let sig_avg: Vec<f64> = (0..sig_multi.ncols())
                .map(|i| sig_multi.column(i).iter().sum::<f64>() / sig_multi.nrows() as f64)
                .collect();

            (ref_avg, sig_avg)
        }
        _ => {
            return Err(AudioSampleError::InvalidParameter(
                "Signals must have the same channel configuration".to_string(),
            ));
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
        match signal.as_mono() {
            Some(mono) => {
                let mut aligned_data = vec![T::default(); best_offset];
                aligned_data.extend_from_slice(
                    &mono.as_slice().ok_or(AudioSampleError::ArrayLayoutError {
                        message: "Mono samples must be contiguous".to_string(),
                    })?[..mono.len() - best_offset],
                );
                let aligned_array = Array1::from_vec(aligned_data);
                AudioSamples::new_mono(aligned_array, signal.sample_rate())
            }
            None => {
                let multi = signal
                    .as_multi_channel()
                    .ok_or(AudioSampleError::InvalidInput {
                        msg: "Must be multi-channel audio".to_string(),
                    })?;
                let mut aligned_data = Vec::new();
                for i in 0..multi.nrows() {
                    let mut row = vec![T::default(); best_offset];
                    row.extend_from_slice(
                        &multi
                            .row(i)
                            .as_slice()
                            .ok_or(AudioSampleError::ArrayLayoutError {
                                message: "Multi-channel samples must be contiguous".to_string(),
                            })?[..multi.ncols() - best_offset],
                    );
                    aligned_data.push(row);
                }
                let aligned_array = ndarray::Array2::from_shape_vec(
                    (aligned_data.len(), aligned_data[0].len()),
                    aligned_data.into_iter().flatten().collect(),
                )
                .map_err(|e| {
                    AudioSampleError::InvalidParameter(format!("Array shape error: {}", e))
                })?;
                AudioSamples::new_multi_channel(aligned_array, signal.sample_rate())
            }
        }
    } else {
        signal.clone()
    };

    Ok((aligned_signal, best_offset))
}

// Helper functions for 1D correlation and MSE
fn correlation_1d(a: &Array1<f64>, b: &Array1<f64>) -> AudioSampleResult<f64> {
    correlation_1d_slice(
        a.as_slice().ok_or(AudioSampleError::ArrayLayoutError {
            message: "Mono samples must be contiguous".to_string(),
        })?,
        b.as_slice().ok_or(AudioSampleError::ArrayLayoutError {
            message: "Mono samples must be contiguous".to_string(),
        })?,
    )
}

fn correlation_1d_slice(a: &[f64], b: &[f64]) -> AudioSampleResult<f64> {
    if a.len() != b.len() {
        return Err(AudioSampleError::InvalidParameter(
            "Arrays must have the same length for correlation".to_string(),
        ));
    }

    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;

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

fn mse_1d(a: &Array1<f64>, b: &Array1<f64>) -> AudioSampleResult<f64> {
    if a.len() != b.len() {
        return Err(AudioSampleError::InvalidParameter(
            "Arrays must have the same length for MSE".to_string(),
        ));
    }

    let n = a.len() as f64;
    let sum_squared_diff: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum();

    Ok(sum_squared_diff / n)
}

fn mse_1d_slice(a: &[f64], b: &[f64]) -> AudioSampleResult<f64> {
    if a.len() != b.len() {
        return Err(AudioSampleError::InvalidParameter(
            "Arrays must have the same length for MSE".to_string(),
        ));
    }
    let n = a.len() as f64;

    let sum_squared_diff: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f64>()
        / n;

    Ok(sum_squared_diff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;
    use ndarray::array;

    #[test]
    fn test_correlation_identical_signals() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio1 = AudioSamples::new_mono(data.clone(), 44100);
        let audio2 = AudioSamples::new_mono(data, 44100);

        let corr = correlation(&audio1, &audio2).unwrap();
        assert_approx_eq!(corr, 1.0, 1e-10);
    }

    #[test]
    fn test_correlation_opposite_signals() {
        let data1 = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let data2 = array![-1.0f32, -2.0, -3.0, -4.0, -5.0];
        let audio1 = AudioSamples::new_mono(data1, 44100);
        let audio2 = AudioSamples::new_mono(data2, 44100);

        let corr = correlation(&audio1, &audio2).unwrap();
        assert_approx_eq!(corr, -1.0, 1e-10);
    }

    #[test]
    fn test_mse_identical_signals() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio1 = AudioSamples::new_mono(data.clone(), 44100);
        let audio2 = AudioSamples::new_mono(data, 44100);

        let mse_val = mse(&audio1, &audio2).unwrap();
        assert_approx_eq!(mse_val, 0.0, 1e-10);
    }

    #[test]
    fn test_snr_calculation() {
        let signal_data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let noise_data = array![0.1f32, 0.2, 0.1, 0.2, 0.1];
        let signal = AudioSamples::new_mono(signal_data, 44100);
        let noise = AudioSamples::new_mono(noise_data, 44100);

        let snr_val = snr(&signal, &noise).unwrap();
        assert!(snr_val > 0.0); // Signal should have higher power than noise
    }

    #[test]
    fn test_align_signals_no_offset() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let reference = AudioSamples::new_mono(data.clone(), 44100);
        let signal = AudioSamples::new_mono(data, 44100);

        let (aligned, offset) = align_signals(&reference, &signal).unwrap();
        assert_eq!(offset, 0);
        assert_eq!(aligned.samples_per_channel(), signal.samples_per_channel());
    }
}
