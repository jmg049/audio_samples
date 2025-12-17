//! Audio comparison and similarity utilities.
//!
//! This module provides functions for comparing audio signals and measuring
//! their similarity using various metrics.

use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ConvertTo,
    I24, LayoutError, ParameterError, RealFloat, repr::MultiData, to_precision,
};
use ndarray::{Array1, ArrayView1};

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
/// The correlation coefficient.
///
/// # Errors
/// Returns an error if the signals have different lengths or channels.
pub fn correlation<T, F>(a: &AudioSamples<T>, b: &AudioSamples<T>) -> AudioSampleResult<F>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    F: RealFloat + ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    for<'b> AudioSamples<'b, T>: AudioTypeConversion<'b, T>,
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
            let a_multi = a_f.as_multi_channel().ok_or(AudioSampleError::Parameter(
                ParameterError::invalid_value("audio_format", "Must be multi-channel audio"),
            ))?;
            let b_multi = b_f.as_multi_channel().ok_or(AudioSampleError::Parameter(
                ParameterError::invalid_value("audio_format", "Must be multi-channel audio"),
            ))?;

            let mut correlations = Vec::new();
            for i in 0..a_multi.nrows() {
                let a_channel = a_multi.row(i);
                let b_channel = b_multi.row(i);
                let corr = correlation_1d_slice(
                    a_channel.as_slice().ok_or(AudioSampleError::Layout(
                        LayoutError::NonContiguous {
                            operation: "signal processing".to_string(),
                            layout_type: "non-contiguous multi-channel samples".to_string(),
                        },
                    ))?,
                    b_channel.as_slice().ok_or(AudioSampleError::Layout(
                        LayoutError::NonContiguous {
                            operation: "signal processing".to_string(),
                            layout_type: "non-contiguous multi-channel samples".to_string(),
                        },
                    ))?,
                )?;
                correlations.push(corr);
            }

            Ok(correlations.iter().fold(F::zero(), |acc, x| acc + *x)
                / to_precision::<F, _>(correlations.len()))
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
/// The mean squared error.
pub fn mse<T, F>(a: &AudioSamples<T>, b: &AudioSamples<T>) -> AudioSampleResult<F>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    F: RealFloat + ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    for<'b> AudioSamples<'b, T>: AudioTypeConversion<'b, T>,
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
            let a_multi = a_f.as_multi_channel().ok_or(AudioSampleError::Parameter(
                ParameterError::invalid_value("audio_format", "Must be multi-channel audio"),
            ))?;
            let b_multi = b_f.as_multi_channel().ok_or(AudioSampleError::Parameter(
                ParameterError::invalid_value("audio_format", "Must be multi-channel audio"),
            ))?;

            let mut mses = Vec::new();
            for i in 0..a_multi.nrows() {
                let a_channel = a_multi.row(i);
                let b_channel = b_multi.row(i);
                let mse = mse_1d_slice(
                    a_channel.as_slice().ok_or(AudioSampleError::Layout(
                        LayoutError::NonContiguous {
                            operation: "signal processing".to_string(),
                            layout_type: "non-contiguous multi-channel samples".to_string(),
                        },
                    ))?,
                    b_channel.as_slice().ok_or(AudioSampleError::Layout(
                        LayoutError::NonContiguous {
                            operation: "signal processing".to_string(),
                            layout_type: "non-contiguous multi-channel samples".to_string(),
                        },
                    ))?,
                )?;
                mses.push(mse);
            }

            Ok(mses.iter().fold(F::zero(), |acc, x| acc + *x) / to_precision::<F, _>(mses.len()))
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
pub fn snr<T, F>(signal: &AudioSamples<T>, noise: &AudioSamples<T>) -> AudioSampleResult<F>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    F: RealFloat + ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    for<'b> AudioSamples<'b, T>: AudioTypeConversion<'b, T>,
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
    let noise_f: AudioSamples<'static, F> = noise.as_float();

    // Calculate signal power
    let signal_power = match signal_f.as_mono() {
        Some(mono) => {
            mono.iter()
                .map(|&x| x * x)
                .fold(F::zero(), |acc, x| acc + x)
                / to_precision::<F, _>(mono.len())
        }
        None => {
            let multi: &MultiData<'static, F> =
                signal_f
                    .as_multi_channel()
                    .ok_or(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "audio_format",
                        "Must be multi-channel audio",
                    )))?;
            multi
                .iter()
                .map(|&x| x * x)
                .fold(F::zero(), |acc, x| acc + x)
                / to_precision::<F, _>(multi.len())
        }
    };

    // Calculate noise power
    let noise_power = match noise_f.as_mono() {
        Some(mono) => {
            mono.iter()
                .map(|x| *x * *x)
                .fold(F::zero(), |acc, x| acc + x)
                / to_precision::<F, _>(mono.len())
        }
        None => {
            let multi = noise_f
                .as_multi_channel()
                .ok_or(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Must be multi-channel audio",
                )))?;
            multi
                .iter()
                .map(|&x| x * x)
                .fold(F::zero(), |acc, x| acc + x)
                / to_precision::<F, _>(multi.len())
        }
    };

    if noise_power == F::zero() {
        return Ok(F::infinity());
    }

    let snr_db = to_precision::<F, _>(10.0) * (signal_power / noise_power).log10();
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
pub fn align_signals<T, F>(
    reference: &AudioSamples<'_, T>,
    signal: &AudioSamples<'_, T>,
) -> AudioSampleResult<(AudioSamples<'static, T>, usize)>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    F: RealFloat + ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    for<'b> AudioSamples<'b, T>: AudioTypeConversion<'b, T>,
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
            let ref_multi = ref_f.as_multi_channel().ok_or(AudioSampleError::Parameter(
                ParameterError::invalid_value("audio_format", "Must be multi-channel audio"),
            ))?;
            let sig_multi = sig_f.as_multi_channel().ok_or(AudioSampleError::Parameter(
                ParameterError::invalid_value("audio_format", "Must be multi-channel audio"),
            ))?;

            let ref_avg: Vec<F> = (0..ref_multi.ncols())
                .map(|i| {
                    ref_multi
                        .column(i)
                        .iter()
                        .fold(F::zero(), |acc, x| acc + *x)
                        / to_precision::<F, _>(ref_multi.nrows())
                })
                .collect();

            let sig_avg: Vec<F> = (0..sig_multi.ncols())
                .map(|i| {
                    sig_multi
                        .column(i)
                        .iter()
                        .fold(F::zero(), |acc, x| acc + *x)
                        / to_precision::<F, _>(sig_multi.nrows())
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
    let mut best_correlation = F::neg_infinity();

    for offset in 0..max_offset {
        let correlation = if offset < sig_data.len() {
            let end = (ref_data.len() - offset).min(sig_data.len() - offset);
            correlation_1d_slice(
                &ref_data[offset..offset + end],
                &sig_data[offset..offset + end],
            )?
        } else {
            F::zero()
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
                    &mono.as_slice().ok_or(AudioSampleError::Layout(
                        LayoutError::NonContiguous {
                            operation: "signal alignment".to_string(),
                            layout_type: "non-contiguous mono samples".to_string(),
                        },
                    ))?[..mono.len() - best_offset],
                );
                let aligned_array = Array1::from_vec(aligned_data);
                AudioSamples::new_mono(aligned_array, sample_rate)
            }
            None => {
                let multi = signal
                    .as_multi_channel()
                    .ok_or(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "audio_format",
                        "Must be multi-channel audio",
                    )))?;
                let mut aligned_data = Vec::new();
                for i in 0..multi.nrows() {
                    let mut row = vec![T::default(); best_offset];
                    row.extend_from_slice(
                        &multi.row(i).as_slice().ok_or(AudioSampleError::Layout(
                            LayoutError::NonContiguous {
                                operation: "signal alignment".to_string(),
                                layout_type: "non-contiguous multi-channel samples".to_string(),
                            },
                        ))?[..multi.ncols() - best_offset],
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
                        format!("Array shape error: {}", e),
                    ))
                })?;
                AudioSamples::new_multi_channel(aligned_array, sample_rate)
            }
        }
    } else {
        signal.clone().into_owned()
    };

    Ok((aligned_signal, best_offset))
}

// Helper functions for 1D correlation and MSE
fn correlation_1d<F: RealFloat>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> AudioSampleResult<F> {
    correlation_1d_slice(
        a.as_slice()
            .ok_or(AudioSampleError::Layout(LayoutError::NonContiguous {
                operation: "correlation calculation".to_string(),
                layout_type: "non-contiguous mono samples".to_string(),
            }))?,
        b.as_slice()
            .ok_or(AudioSampleError::Layout(LayoutError::NonContiguous {
                operation: "correlation calculation".to_string(),
                layout_type: "non-contiguous mono samples".to_string(),
            }))?,
    )
}

fn correlation_1d_slice<F: RealFloat>(a: &[F], b: &[F]) -> AudioSampleResult<F> {
    if a.len() != b.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "array_length",
            "Arrays must have the same length for correlation",
        )));
    }

    let n = to_precision::<F, _>(a.len());
    let mean_a = a.iter().fold(F::zero(), |acc, x| acc + *x) / n;
    let mean_b = b.iter().fold(F::zero(), |acc, x| acc + *x) / n;

    let mut num = F::zero();
    let mut den_a = F::zero();
    let mut den_b = F::zero();

    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        num += dx * dy;
        den_a += dx * dx;
        den_b += dy * dy;
    }

    let denominator = (den_a * den_b).sqrt();
    if denominator == F::zero() {
        Ok(F::zero())
    } else {
        Ok(num / denominator)
    }
}

fn mse_1d<F: RealFloat>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> AudioSampleResult<F> {
    if a.len() != b.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "array_length",
            "Arrays must have the same length for MSE",
        )));
    }

    let n = to_precision::<F, _>(a.len());
    let sum_squared_diff: F = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .fold(F::zero(), |acc, val| acc + val);

    Ok(sum_squared_diff / n)
}

fn mse_1d_slice<F: RealFloat>(a: &[F], b: &[F]) -> AudioSampleResult<F> {
    if a.len() != b.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "array_length",
            "Arrays must have the same length for MSE",
        )));
    }
    let n = to_precision::<F, _>(a.len());

    let sum_squared_diff: F = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .fold(F::zero(), |acc, val| acc + val)
        / n;

    Ok(sum_squared_diff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample_rate;
    use approx_eq::assert_approx_eq;
    use ndarray::array;

    #[test]
    fn test_correlation_identical_signals() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio1 = AudioSamples::new_mono(data.clone(), sample_rate!(44100));
        let audio2 = AudioSamples::new_mono(data, sample_rate!(44100));

        let corr: f64 = correlation(&audio1, &audio2).unwrap();
        assert_approx_eq!(corr, 1.0, 1e-10);
    }

    #[test]
    fn test_correlation_opposite_signals() {
        let data1 = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let data2 = array![-1.0f32, -2.0, -3.0, -4.0, -5.0];
        let audio1 = AudioSamples::new_mono(data1, sample_rate!(44100));
        let audio2 = AudioSamples::new_mono(data2, sample_rate!(44100));

        let corr: f64 = correlation(&audio1, &audio2).unwrap();
        assert_approx_eq!(corr, -1.0, 1e-10);
    }

    #[test]
    fn test_mse_identical_signals() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio1 = AudioSamples::new_mono(data.clone(), sample_rate!(44100));
        let audio2 = AudioSamples::new_mono(data, sample_rate!(44100));

        let mse_val = mse(&audio1, &audio2).unwrap();
        assert_approx_eq!(mse_val, 0.0_f64, 1e-10);
    }

    #[test]
    fn test_snr_calculation() {
        let signal_data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let noise_data = array![0.1f32, 0.2, 0.1, 0.2, 0.1];
        let signal = AudioSamples::new_mono(signal_data, sample_rate!(44100));
        let noise = AudioSamples::new_mono(noise_data, sample_rate!(44100));

        let snr_val: f64 = snr(&signal, &noise).unwrap();
        assert!(snr_val > 0.0_f64); // Signal should have higher power than noise
    }

    #[test]
    fn test_align_signals_no_offset() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let reference = AudioSamples::new_mono(data.clone(), sample_rate!(44100));
        let signal = AudioSamples::new_mono(data, sample_rate!(44100));

        let (aligned, offset) = align_signals::<f32, f64>(&reference, &signal).unwrap();
        assert_eq!(offset, 0);
        assert_eq!(aligned.samples_per_channel(), signal.samples_per_channel());
    }
}
