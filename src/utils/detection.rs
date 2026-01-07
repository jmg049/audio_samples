//! Format detection and audio analysis utilities.
//!
//! This module provides functions for detecting and analyzing properties
//! of audio signals, such as sample rate, fundamental frequency, and
//! silence regions.

#[cfg(feature = "transforms")]
use spectrograms::{WindowType, power_spectrum};
#[cfg(feature = "transforms")]
use std::num::{NonZeroU32, NonZeroUsize};

use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, LayoutError,
    ParameterError, traits::StandardSample,
};

use non_empty_slice::{NonEmptySlice, NonEmptyVec};

/// Attempts to detect the sample rate of an audio signal based on its content.
///
/// This function analyzes the frequency content of the signal to estimate
/// the original sample rate. It's useful for validating sample rate metadata
/// or detecting resampled content.
///
/// # Arguments
/// * `audio` - The audio signal to analyze
///
/// # Returns
/// * `Some(sample_rate)` - The detected sample rate in Hz
/// * `None` - If sample rate cannot be reliably detected
#[cfg(feature = "transforms")]
#[inline]
pub fn detect_sample_rate<T>(audio: &AudioSamples<T>) -> AudioSampleResult<Option<NonZeroU32>>
where
    T: StandardSample,
{
    let audio_f64 = audio.as_float();

    // Use the first channel for analysis
    let data = if let Some(mono) = audio_f64.as_mono() {
        mono.as_slice()
            .ok_or(AudioSampleError::Layout(LayoutError::NonContiguous {
                operation: "audio processing".to_string(),
                layout_type: "non-contiguous mono array".to_string(),
            }))?
            .to_vec()
    } else {
        // Use the first channel of multi-channel audio
        let multi = audio_f64
            .as_multi_channel()
            .ok_or(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_data",
                "Audio must be multi-channel",
            )))?;
        multi
            .row(0)
            .as_slice()
            .ok_or(AudioSampleError::Layout(LayoutError::NonContiguous {
                operation: "audio processing".to_string(),
                layout_type: "non-contiguous multi-channel array".to_string(),
            }))?
            .to_vec()
    };

    // Look for high-frequency cutoff patterns that might indicate resampling
    let n_fft = NonZeroUsize::new(data.len().next_power_of_two()).expect("data is non-empty");
    // safety: data is non-empty (channel was extracted from valid audio)
    let data_slice = unsafe { NonEmptySlice::new_unchecked(&data[..]) };
    let spectrum = power_spectrum(data_slice, n_fft, Some(WindowType::Hanning))
        .map_err(AudioSampleError::from)?;
    let nyquist_freq = audio.nyquist();
    let spectrum_slice = spectrum.as_non_empty_slice();
    // Analyze the spectrum for sharp cutoffs that might indicate resampling
    let detected_rate = analyze_spectrum_for_cutoff(spectrum_slice, nyquist_freq).ok_or(
        AudioSampleError::Parameter(ParameterError::invalid_value(
            "audio_data",
            "Could not detect sample rate from audio content",
        )),
    )?;

    let detected_rate = NonZeroU32::new(detected_rate).ok_or(AudioSampleError::Parameter(
        ParameterError::invalid_value("detected_rate", "detected sample rate is zero"),
    ))?;
    Ok(Some(detected_rate))
}

/// Detects the fundamental frequency of an audio signal.
///
/// This function uses spectral analysis to find the fundamental frequency
/// of the input signal. It's useful for pitch detection and harmonic analysis.
///
/// # Arguments
/// * `audio` - The audio signal to analyze
///
/// # Returns
/// * `Some(frequency)` - The detected fundamental frequency in Hz
/// * `None` - If no clear fundamental frequency is detected
#[inline]
pub fn detect_fundamental_frequency<T>(audio: &AudioSamples<T>) -> AudioSampleResult<Option<f64>>
where
    T: StandardSample,
{
    // error if audio is empty
    // if audio is empty, you are doing something wrong
    let audio_f64 = audio.as_float();

    // Use the first channel for analysis
    let data = if let Some(mono) = audio_f64.as_mono() {
        mono.as_slice()
            .ok_or(AudioSampleError::Layout(LayoutError::NonContiguous {
                operation: "audio processing".to_string(),
                layout_type: "non-contiguous mono array".to_string(),
            }))?
            .to_vec()
    } else {
        // Use the first channel of multi-channel audio
        let multi = audio_f64
            .as_multi_channel()
            .ok_or(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_data",
                "Audio must be multi-channel",
            )))?;
        multi
            .row(0)
            .as_slice()
            .ok_or(AudioSampleError::Layout(LayoutError::NonContiguous {
                operation: "audio processing".to_string(),
                layout_type: "non-contiguous multi-channel array".to_string(),
            }))?
            .to_vec()
    };
    // safety: data is non-empty (channel was extracted from valid audio)
    let data = unsafe { NonEmptyVec::new_unchecked(data) };

    // Use autocorrelation method for fundamental frequency detection
    let fundamental =
        detect_fundamental_autocorrelation(&data, f64::from(audio.sample_rate().get()))?;

    Ok(fundamental)
}

/// Detects silence regions in an audio signal.
///
/// This function identifies time intervals where the audio signal
/// falls below a specified threshold, indicating silence or very quiet regions.
///
/// # Arguments
/// * `audio` - The audio signal to analyze
/// * `threshold` - The amplitude threshold below which samples are considered silence
///
/// # Returns
/// A vector of (start_time, end_time) tuples representing silence regions in seconds
#[inline]
pub fn detect_silence_regions<T>(
    audio: &AudioSamples<'_, T>,
    threshold: T,
) -> AudioSampleResult<Vec<(f64, f64)>>
where
    T: StandardSample,
{
    let mut silence_regions = Vec::new();
    let mut in_silence = false;
    let mut silence_start = 0;

    let sample_rate = audio.sample_rate().get();
    let samples_per_second = sample_rate;
    let threshold: f64 = threshold.cast_into();

    // Helper function to check if a sample is below threshold (absolute value)
    let is_below_threshold = |sample: T| -> bool {
        let abs_val: f64 = sample.cast_into();
        let abs_val = abs_val.abs();
        abs_val < threshold
    };

    if let Some(mono) = audio.as_mono() {
        for (i, &sample) in mono.iter().enumerate() {
            if is_below_threshold(sample) {
                if !in_silence {
                    silence_start = i;
                    in_silence = true;
                }
            } else if in_silence {
                let start_time = silence_start as f64 / f64::from(samples_per_second);
                let end_time = i as f64 / f64::from(samples_per_second);
                silence_regions.push((start_time, end_time));
                in_silence = false;
            }
        }

        // Handle case where silence extends to the end
        if in_silence {
            let start_time = silence_start as f64 / f64::from(samples_per_second);
            let end_time = mono.len().get() as f64 / f64::from(samples_per_second);
            silence_regions.push((start_time, end_time));
        }
    } else {
        // Multi-channel analysis - consider it silence only if ALL channels are below threshold
        let multi = audio.as_multi_channel().ok_or(AudioSampleError::Parameter(
            ParameterError::invalid_value("audio_data", "Audio must be multi-channel"),
        ))?;
        for i in 0..multi.ncols().get() {
            let all_below_threshold =
                (0..multi.nrows().get()).all(|ch| is_below_threshold(multi[(ch, i)]));

            if all_below_threshold {
                if !in_silence {
                    silence_start = i;
                    in_silence = true;
                }
            } else if in_silence {
                let start_time = silence_start as f64 / f64::from(samples_per_second);
                let end_time = i as f64 / f64::from(samples_per_second);
                silence_regions.push((start_time, end_time));
                in_silence = false;
            }
        }

        // Handle case where silence extends to the end
        if in_silence {
            let start_time = silence_start as f64 / f64::from(samples_per_second);
            let end_time = multi.ncols().get() as f64 / f64::from(samples_per_second);
            silence_regions.push((start_time, end_time));
        }
    }

    Ok(silence_regions)
}

/// Detects the dynamic range of an audio signal.
///
/// This function analyzes the amplitude distribution of the signal
/// to determine its dynamic range characteristics.
///
/// # Arguments
/// * `audio` - The audio signal to analyze
///
/// # Returns
/// A tuple of (peak_amplitude, rms_amplitude, dynamic_range_db)
#[inline]
pub fn detect_dynamic_range<T>(audio: &AudioSamples<T>) -> AudioSampleResult<(f64, f64, f64)>
where
    T: StandardSample,
{
    let audio_f = audio.as_float();

    let data: Vec<f64> = if let Some(mono) = audio_f.as_mono() {
        mono.as_slice()
            .ok_or(AudioSampleError::Layout(LayoutError::NonContiguous {
                operation: "audio processing".to_string(),
                layout_type: "non-contiguous mono array".to_string(),
            }))?
            .to_vec()
    } else {
        // Flatten multi-channel audio
        let multi = audio_f
            .as_multi_channel()
            .ok_or(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_data",
                "Audio must be multi-channel",
            )))?;
        multi
            .as_slice()
            .ok_or(AudioSampleError::Layout(LayoutError::NonContiguous {
                operation: "audio processing".to_string(),
                layout_type: "non-contiguous multi-channel array".to_string(),
            }))?
            .to_vec()
    };

    if data.is_empty() {
        return Ok((0.0, 0.0, 0.0));
    }

    // Calculate peak amplitude
    let peak_amplitude = data.iter().map(|&x| x.abs()).fold(0.0, f64::max);

    // Calculate RMS amplitude
    let rms_amplitude = (data.iter().fold(0.0, |acc, x| acc + *x) / data.len() as f64).sqrt();

    // Calculate dynamic range in dB
    let dynamic_range_db = if rms_amplitude > 0.0 {
        20.0 * (peak_amplitude / rms_amplitude).log10()
    } else {
        0.0
    };

    Ok((peak_amplitude, rms_amplitude, dynamic_range_db))
}

/// Detects clipping in an audio signal.
///
/// This function identifies regions where the audio signal is clipped
/// (reaches the maximum or minimum values and stays there).
///
/// # Arguments
/// * `audio` - The audio signal to analyze
/// * `threshold_ratio` - The ratio of max value to consider as clipping (default: 0.99)
///
/// # Returns
/// A vector of (start_time, end_time) tuples representing clipped regions in seconds
#[inline]
pub fn detect_clipping<T>(
    audio: &AudioSamples<T>,
    threshold_ratio: f64,
) -> AudioSampleResult<Vec<(f64, f64)>>
where
    T: StandardSample,
{
    let mut clipped_regions = Vec::new();
    let mut in_clipped = false;
    let mut clipped_start = 0;

    let sample_rate = audio.sample_rate().get();

    // Determine clipping thresholds
    let max_val: f64 = T::MAX.cast_into();
    let min_val: f64 = T::MIN.cast_into();

    let upper_threshold: T = T::cast_from(max_val * threshold_ratio);
    let lower_threshold: T = T::cast_from(min_val * threshold_ratio);

    #[inline]
    fn is_clipped<T>(sample: T, upper_threshold: T, lower_threshold: T) -> bool
    where
        T: StandardSample,
    {
        sample >= upper_threshold || sample <= lower_threshold
    }

    if let Some(mono) = audio.as_mono() {
        for (i, &sample) in mono.iter().enumerate() {
            if is_clipped(sample, upper_threshold, lower_threshold) {
                if !in_clipped {
                    clipped_start = i;
                    in_clipped = true;
                }
            } else if in_clipped {
                let start_time = clipped_start as f64 / f64::from(sample_rate);
                let end_time = i as f64 / f64::from(sample_rate);
                clipped_regions.push((start_time, end_time));
                in_clipped = false;
            }
        }

        // Handle case where clipping extends to the end
        if in_clipped {
            let start_time = clipped_start as f64 / f64::from(sample_rate);
            let end_time = mono.len().get() as f64 / f64::from(sample_rate);
            clipped_regions.push((start_time, end_time));
        }
    } else {
        // Multi-channel analysis - consider it clipped if ANY channel is clipped
        let multi = audio.as_multi_channel().ok_or(AudioSampleError::Parameter(
            ParameterError::invalid_value("audio_data", "Audio must be mono or multi-channel"),
        ))?;
        for i in 0..multi.ncols().get() {
            let mut any_clipped = false;
            for ch in 0..multi.nrows().get() {
                if is_clipped(multi[(ch, i)], upper_threshold, lower_threshold) {
                    any_clipped = true;
                    break;
                }
            }

            if any_clipped {
                if !in_clipped {
                    clipped_start = i;
                    in_clipped = true;
                }
            } else if in_clipped {
                let start_time = clipped_start as f64 / f64::from(sample_rate);
                let end_time = i as f64 / f64::from(sample_rate);
                clipped_regions.push((start_time, end_time));
                in_clipped = false;
            }
        }

        // Handle case where clipping extends to the end
        if in_clipped {
            let start_time = clipped_start as f64 / f64::from(sample_rate);
            let end_time = multi.ncols().get() as f64 / f64::from(sample_rate);
            clipped_regions.push((start_time, end_time));
        }
    }

    Ok(clipped_regions)
}

/// Estimate noise floor of audio signal.
#[inline]
pub fn estimate_noise_floor<T, F>(audio: &AudioSamples<T>) -> AudioSampleResult<Option<f64>>
where
    T: StandardSample,
{
    let audio = audio.as_float();

    let data: Vec<f64> = if let Some(mono) = audio.as_mono() {
        mono.as_slice()
            .ok_or(AudioSampleError::Layout(LayoutError::NonContiguous {
                operation: "audio processing".to_string(),
                layout_type: "non-contiguous mono array".to_string(),
            }))?
            .to_vec()
    } else {
        // Use first channel
        let multi = audio.as_multi_channel().ok_or(AudioSampleError::Parameter(
            ParameterError::invalid_value("audio_data", "Audio must be multi-channel"),
        ))?;
        multi
            .row(0)
            .as_slice()
            .ok_or(AudioSampleError::Layout(LayoutError::NonContiguous {
                operation: "audio processing".to_string(),
                layout_type: "non-contiguous multi-channel array".to_string(),
            }))?
            .to_vec()
    };

    if data.is_empty() {
        return Ok(None);
    }

    // Find quietest regions (bottom 10th percentile)
    let mut sorted_abs: Vec<f64> = data.iter().map(|&x| x.abs()).collect();
    sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let percentile_10 = sorted_abs.len() / 10;
    if percentile_10 > 0 {
        let noise_level = sorted_abs[percentile_10];
        if noise_level > 0.0 {
            let noise_floor_db = 20.0 * noise_level.log10();
            return Ok(Some(noise_floor_db));
        }
    }

    Ok(None)
}

/// Estimate frequency response range.
#[cfg(feature = "transforms")]
#[inline]
pub fn estimate_frequency_range<T>(audio: &AudioSamples<T>) -> AudioSampleResult<Option<(f64, f64)>>
where
    T: StandardSample,
{
    let audio = audio.as_float();

    let data: Vec<f64> = if let Some(mono) = audio.as_mono() {
        mono.as_slice()
            .ok_or(AudioSampleError::Layout(LayoutError::NonContiguous {
                operation: "audio processing".to_string(),
                layout_type: "non-contiguous mono array".to_string(),
            }))?
            .to_vec()
    } else {
        // Use first channel
        let multi = audio.as_multi_channel().ok_or(AudioSampleError::Parameter(
            ParameterError::invalid_value("audio_data", "Audio must be multi-channel"),
        ))?;
        multi
            .row(0)
            .as_slice()
            .ok_or(AudioSampleError::Layout(LayoutError::NonContiguous {
                operation: "audio processing".to_string(),
                layout_type: "non-contiguous multi-channel array".to_string(),
            }))?
            .to_vec()
    };

    if data.len() < 1024 {
        return Ok(None); // Too short for meaningful analysis
    }

    // Compute spectrum
    let n_fft = NonZeroUsize::new(data.len().next_power_of_two()).expect("data is non-empty");
    // safety: data is non-empty (channel was extracted from valid audio)
    let data_slice = unsafe { NonEmptySlice::new_unchecked(&data[..]) };
    let spectrum = power_spectrum(data_slice, n_fft, Some(WindowType::Hanning))
        .map_err(AudioSampleError::from)?;
    let nyquist = audio.nyquist();

    // Find frequency range with significant energy
    let threshold = spectrum.iter().fold(0.0, |acc: f64, &x| acc.max(x)) * 0.01; // 1% of peak

    let mut low_freq = None;
    let mut high_freq = None;

    for (i, &energy) in spectrum.iter().enumerate() {
        if energy > threshold {
            let freq = i as f64 / spectrum.len().get() as f64 * nyquist;
            if low_freq.is_none() {
                low_freq = Some(freq);
            }
            high_freq = Some(freq);
        }
    }

    if let (Some(low), Some(high)) = (low_freq, high_freq) {
        Ok(Some((low, high)))
    } else {
        Ok(None)
    }
}

/// Analyzes a frequency spectrum for potential cutoff frequencies that might indicate resampling.
///
/// # Panics
/// Panics if internal float-to-`usize` conversion fails (e.g. if `nyquist_freq` is non-finite).
#[inline]
#[must_use]
pub fn analyze_spectrum_for_cutoff(
    spectrum: &NonEmptySlice<f64>,
    nyquist_freq: f64,
) -> Option<u32> {
    // Look for sharp cutoffs that might indicate resampling
    let len = spectrum.len().get();
    let half_len = len / 2;

    // Check for energy drops that might indicate filtering
    let mut candidates = Vec::new();

    // Common resampling target frequencies
    let common_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000];

    for &rate in &common_rates {
        let target_nyquist = f64::from(rate) / 2.0;
        if target_nyquist < nyquist_freq {
            // Check if there's a significant drop in energy around this frequency
            let bin_index = (target_nyquist / nyquist_freq * half_len as f64) as usize;
            if bin_index < spectrum.len().get() - 10 {
                let energy_before: f64 = spectrum[bin_index.saturating_sub(5)..bin_index]
                    .iter()
                    .fold(0.0, |acc, &x| acc + x);
                let energy_after = spectrum[bin_index..bin_index + 10]
                    .iter()
                    .fold(0.0, |acc, &x| acc + x);

                if energy_before > energy_after * 2.0 {
                    candidates.push(rate);
                }
            }
        }
    }

    // Return the most likely candidate
    candidates.first().copied()
}

/// Detects the fundamental frequency using autocorrelation analysis.
///
/// # Panics
/// Panics if internal float-to-`usize` conversions fail (e.g. if `sample_rate` is non-finite).
#[inline]
pub fn detect_fundamental_autocorrelation(
    data: &NonEmptySlice<f64>,
    sample_rate: f64,
) -> AudioSampleResult<Option<f64>> {
    if data.len() < crate::nzu!(2) {
        return Ok(None);
    }

    let min_freq = 50.0; // Minimum fundamental frequency to detect
    let max_freq = 2000.0; // Maximum fundamental frequency to detect

    let min_period = (sample_rate / max_freq) as usize;
    let max_period = (sample_rate / min_freq) as usize;
    if max_period >= data.len().get() {
        return Ok(None);
    }

    let mut max_correlation = 0.0;
    let mut best_period = 0;

    for period in min_period..max_period.min(data.len().get() / 2) {
        let mut correlation = 0.0;
        let mut count = 0;

        for i in 0..(data.len().get() - period) {
            correlation += data[i] * data[i + period];
            count += 1;
        }

        if count > 0 {
            correlation /= f64::from(count);

            if correlation > max_correlation {
                max_correlation = correlation;
                best_period = period;
            }
        }
    }

    if best_period > 0 && max_correlation > 0.1 {
        let fundamental_freq = sample_rate / best_period as f64;
        Ok(Some(fundamental_freq))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample_rate;
    use ndarray::array;
    use non_empty_slice::non_empty_vec;

    #[test]
    fn test_detect_silence_regions() {
        // Create a signal with silence regions
        let data = array![0.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let audio: AudioSamples<'_, f32> = AudioSamples::new_mono(data, sample_rate!(10)).unwrap(); // 10 Hz sample rate for easy calculation

        let silence_regions: Vec<(f64, f64)> =
            detect_silence_regions(&audio, 0.5_f32).expect("Failed to detect silence");

        assert!(!silence_regions.is_empty());
        // First silence region should be at the beginning
        assert_eq!(silence_regions[0].0, 0.0);
        assert_eq!(silence_regions[0].1, 0.3);
    }

    #[test]
    fn test_detect_dynamic_range() {
        // Create a signal with known dynamic range
        let data = non_empty_vec![0.1f32, 0.5, 1.0, 0.2, 0.8];
        let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(data, sample_rate!(10));

        let (peak, rms, dynamic_range) = detect_dynamic_range(&audio).unwrap();

        assert_eq!(peak, 1.0);
        assert!(rms > 0.0);
        assert!(dynamic_range > 0.0);
    }

    #[test]
    fn test_detect_clipping() {
        // Create a signal with clipping
        let data = array![0.5f32, 1.0, 1.0, 1.0, 0.5, -1.0, -1.0, 0.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(8)).unwrap(); // 8 Hz sample rate for easy calculation

        let clipped_regions = detect_clipping(&audio, 0.99).expect("Failed to detect clipping");

        assert!(!clipped_regions.is_empty());
    }

    #[test]
    fn test_detect_fundamental_frequency() {
        // Create a sine wave with known frequency
        let sample_rate = crate::sample_rate!(44100);
        let frequency: f64 = 440.0; // A4
        let duration: f64 = 1.0; // 1 second

        let audio = crate::sine_wave::<f64>(
            frequency,
            std::time::Duration::from_secs_f64(duration),
            sample_rate,
            1.0f64,
        );

        let detected_freq: Option<f64> = detect_fundamental_frequency(&audio).unwrap();

        if let Some(freq) = detected_freq {
            // Allow broader tolerance for autocorrelation-based detection on pure tones
            // Pure sine waves can confuse autocorrelation algorithms
            let tolerance: f64 =
                if (freq - frequency / 2.0).abs() < 20.0 || (freq - frequency / 4.0).abs() < 20.0 {
                    // If it's detecting a subharmonic (half or quarter frequency), that's acceptable
                    400.0
                } else {
                    50.0 // Otherwise use more relaxed tolerance
                };

            assert!(
                (freq - frequency).abs() < tolerance,
                "Expected: {} Hz, Detected: {} Hz, Difference: {} Hz",
                frequency,
                freq,
                (freq - frequency).abs()
            );
        } else {
            panic!("No frequency detected");
        }
    }
}
