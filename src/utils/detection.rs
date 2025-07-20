//! Format detection and audio analysis utilities.
//!
//! This module provides functions for detecting and analyzing properties
//! of audio signals, such as sample rate, fundamental frequency, and
//! silence regions.

use crate::{AudioSample, AudioSampleResult, AudioSamples, AudioTypeConversion, ConvertTo, I24};
use rustfft::{FftPlanner, num_complex::Complex};

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
pub fn detect_sample_rate<T: AudioSample>(audio: &AudioSamples<T>) -> AudioSampleResult<Option<u32>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    let audio_f64 = audio.as_f64()?;

    // Use the first channel for analysis
    let data = match audio_f64.as_mono() {
        Some(mono) => mono.as_slice().unwrap().to_vec(),
        None => {
            // Use the first channel of multi-channel audio
            let multi = audio_f64.as_multi_channel().unwrap();
            multi.row(0).as_slice().unwrap().to_vec()
        }
    };

    // Look for high-frequency cutoff patterns that might indicate resampling
    let spectrum = compute_spectrum(&data)?;
    let nyquist_freq = audio.sample_rate() as f64 / 2.0;

    // Analyze the spectrum for sharp cutoffs that might indicate resampling
    let detected_rate = analyze_spectrum_for_cutoff(&spectrum, nyquist_freq);

    Ok(detected_rate)
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
pub fn detect_fundamental_frequency<T: AudioSample>(
    audio: &AudioSamples<T>,
) -> AudioSampleResult<Option<f64>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    let audio_f64 = audio.as_f64()?;

    // Use the first channel for analysis
    let data = match audio_f64.as_mono() {
        Some(mono) => mono.as_slice().unwrap().to_vec(),
        None => {
            // Use the first channel of multi-channel audio
            let multi = audio_f64.as_multi_channel().unwrap();
            multi.row(0).as_slice().unwrap().to_vec()
        }
    };

    if data.is_empty() {
        return Ok(None);
    }

    // Use autocorrelation method for fundamental frequency detection
    let fundamental = detect_fundamental_autocorrelation(&data, audio.sample_rate() as f64)?;

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
pub fn detect_silence_regions<T: AudioSample>(
    audio: &AudioSamples<T>,
    threshold: T,
) -> Vec<(f64, f64)> {
    let mut silence_regions = Vec::new();
    let mut in_silence = false;
    let mut silence_start = 0;

    let sample_rate = audio.sample_rate() as f64;
    let samples_per_second = sample_rate;

    // Helper function to check if a sample is below threshold (absolute value)
    let is_below_threshold = |sample: T| -> bool {
        let abs_val = if sample < T::default() {
            T::default() - sample
        } else {
            sample
        };
        abs_val < threshold
    };

    match audio.as_mono() {
        Some(mono) => {
            for (i, &sample) in mono.iter().enumerate() {
                if is_below_threshold(sample) {
                    if !in_silence {
                        silence_start = i;
                        in_silence = true;
                    }
                } else {
                    if in_silence {
                        let start_time = silence_start as f64 / samples_per_second;
                        let end_time = i as f64 / samples_per_second;
                        silence_regions.push((start_time, end_time));
                        in_silence = false;
                    }
                }
            }

            // Handle case where silence extends to the end
            if in_silence {
                let start_time = silence_start as f64 / samples_per_second;
                let end_time = mono.len() as f64 / samples_per_second;
                silence_regions.push((start_time, end_time));
            }
        }
        None => {
            // Multi-channel analysis - consider it silence only if ALL channels are below threshold
            let multi = audio.as_multi_channel().unwrap();
            for i in 0..multi.ncols() {
                let all_below_threshold =
                    (0..multi.nrows()).all(|ch| is_below_threshold(multi[(ch, i)]));

                if all_below_threshold {
                    if !in_silence {
                        silence_start = i;
                        in_silence = true;
                    }
                } else {
                    if in_silence {
                        let start_time = silence_start as f64 / samples_per_second;
                        let end_time = i as f64 / samples_per_second;
                        silence_regions.push((start_time, end_time));
                        in_silence = false;
                    }
                }
            }

            // Handle case where silence extends to the end
            if in_silence {
                let start_time = silence_start as f64 / samples_per_second;
                let end_time = multi.ncols() as f64 / samples_per_second;
                silence_regions.push((start_time, end_time));
            }
        }
    }

    silence_regions
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
pub fn detect_dynamic_range<T: AudioSample>(
    audio: &AudioSamples<T>,
) -> AudioSampleResult<(f64, f64, f64)>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    let audio_f64 = audio.as_f64()?;

    let data: Vec<f64> = match audio_f64.as_mono() {
        Some(mono) => mono.as_slice().unwrap().to_vec(),
        None => {
            // Flatten multi-channel audio
            let multi = audio_f64.as_multi_channel().unwrap();
            multi.as_slice().unwrap().to_vec()
        }
    };

    if data.is_empty() {
        return Ok((0.0, 0.0, 0.0));
    }

    // Calculate peak amplitude
    let peak_amplitude = data.iter().map(|&x| x.abs()).fold(0.0, f64::max);

    // Calculate RMS amplitude
    let rms_amplitude = (data.iter().map(|&x| x * x).sum::<f64>() / data.len() as f64).sqrt();

    // Calculate dynamic range in dB
    let dynamic_range_db = if rms_amplitude > 0.0_f64 {
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
pub fn detect_clipping<T: AudioSample>(
    audio: &AudioSamples<T>,
    threshold_ratio: f64,
) -> Vec<(f64, f64)>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    let mut clipped_regions = Vec::new();
    let mut in_clipped = false;
    let mut clipped_start = 0;

    let sample_rate = audio.sample_rate() as f64;

    // Determine clipping thresholds
    let max_val = T::MAX;
    let min_val = T::MIN;

    // Convert to f64 for threshold calculation
    let max_f64 = max_val.convert_to().unwrap_or(1.0);
    let min_f64 = min_val.convert_to().unwrap_or(-1.0);

    let upper_threshold = max_f64 * threshold_ratio;
    let lower_threshold = min_f64 * threshold_ratio;

    // TODO: Make constant function
    let is_clipped = |sample: T| -> bool {
        let sample_f64 = sample.convert_to().unwrap_or(0.0);
        sample_f64 >= upper_threshold || sample_f64 <= lower_threshold
    };

    match audio.as_mono() {
        Some(mono) => {
            for (i, &sample) in mono.iter().enumerate() {
                if is_clipped(sample) {
                    if !in_clipped {
                        clipped_start = i;
                        in_clipped = true;
                    }
                } else {
                    if in_clipped {
                        let start_time = clipped_start as f64 / sample_rate;
                        let end_time = i as f64 / sample_rate;
                        clipped_regions.push((start_time, end_time));
                        in_clipped = false;
                    }
                }
            }

            // Handle case where clipping extends to the end
            if in_clipped {
                let start_time = clipped_start as f64 / sample_rate;
                let end_time = mono.len() as f64 / sample_rate;
                clipped_regions.push((start_time, end_time));
            }
        }
        None => {
            // Multi-channel analysis - consider it clipped if ANY channel is clipped
            let multi = audio.as_multi_channel().unwrap();
            for i in 0..multi.ncols() {
                let any_clipped = (0..multi.nrows()).any(|ch| is_clipped(multi[(ch, i)]));

                if any_clipped {
                    if !in_clipped {
                        clipped_start = i;
                        in_clipped = true;
                    }
                } else {
                    if in_clipped {
                        let start_time = clipped_start as f64 / sample_rate;
                        let end_time = i as f64 / sample_rate;
                        clipped_regions.push((start_time, end_time));
                        in_clipped = false;
                    }
                }
            }

            // Handle case where clipping extends to the end
            if in_clipped {
                let start_time = clipped_start as f64 / sample_rate;
                let end_time = multi.ncols() as f64 / sample_rate;
                clipped_regions.push((start_time, end_time));
            }
        }
    }

    clipped_regions
}

// Helper functions

fn compute_spectrum(data: &[f64]) -> AudioSampleResult<Vec<f64>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(data.len());

    let mut buffer: Vec<Complex<f64>> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut buffer);

    let spectrum: Vec<f64> = buffer.iter().map(|c| c.norm()).collect();
    Ok(spectrum)
}

fn analyze_spectrum_for_cutoff(spectrum: &[f64], nyquist_freq: f64) -> Option<u32> {
    // Look for sharp cutoffs that might indicate resampling
    let len = spectrum.len();
    let half_len = len / 2;

    // Check for energy drops that might indicate filtering
    let mut candidates = Vec::new();

    // Common resampling target frequencies
    let common_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000];

    for &rate in &common_rates {
        let target_nyquist = rate as f64 / 2.0;
        if target_nyquist < nyquist_freq {
            // Check if there's a significant drop in energy around this frequency
            let bin_index = (target_nyquist / nyquist_freq * half_len as f64) as usize;
            if bin_index < spectrum.len() - 10 {
                let energy_before = spectrum[bin_index.saturating_sub(5)..bin_index]
                    .iter()
                    .sum::<f64>();
                let energy_after = spectrum[bin_index..bin_index + 10].iter().sum::<f64>();

                if energy_before > energy_after * 2.0 {
                    candidates.push(rate);
                }
            }
        }
    }

    // Return the most likely candidate
    candidates.first().copied()
}

fn detect_fundamental_autocorrelation(
    data: &[f64],
    sample_rate: f64,
) -> AudioSampleResult<Option<f64>> {
    if data.len() < 2 {
        return Ok(None);
    }

    let min_freq = 50.0; // Minimum fundamental frequency to detect
    let max_freq = 2000.0; // Maximum fundamental frequency to detect

    let min_period = (sample_rate / max_freq) as usize;
    let max_period = (sample_rate / min_freq) as usize;

    if max_period >= data.len() {
        return Ok(None);
    }

    let mut max_correlation = 0.0;
    let mut best_period = 0;

    for period in min_period..max_period.min(data.len() / 2) {
        let mut correlation = 0.0;
        let mut count = 0;

        for i in 0..(data.len() - period) {
            correlation += data[i] * data[i + period];
            count += 1;
        }

        if count > 0 {
            correlation /= count as f64;

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
    use ndarray::array;
    use std::f64::consts::PI;

    #[test]
    fn test_detect_silence_regions() {
        // Create a signal with silence regions
        let data = array![0.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let audio = AudioSamples::new_mono(data, 10); // 10 Hz sample rate for easy calculation

        let silence_regions = detect_silence_regions(&audio, 0.5);

        assert!(!silence_regions.is_empty());
        // First silence region should be at the beginning
        assert_eq!(silence_regions[0].0, 0.0);
        assert_eq!(silence_regions[0].1, 0.3);
    }

    #[test]
    fn test_detect_dynamic_range() {
        // Create a signal with known dynamic range
        let data = array![0.1f32, 0.5, 1.0, 0.2, 0.8];
        let audio = AudioSamples::new_mono(data, 44100);

        let (peak, rms, dynamic_range) = detect_dynamic_range(&audio).unwrap();

        assert_eq!(peak, 1.0);
        assert!(rms > 0.0);
        assert!(dynamic_range > 0.0);
    }

    #[test]
    fn test_detect_clipping() {
        // Create a signal with clipping
        let data = array![0.5f32, 1.0, 1.0, 1.0, 0.5, -1.0, -1.0, 0.0];
        let audio = AudioSamples::new_mono(data, 8); // 8 Hz sample rate for easy calculation

        let clipped_regions = detect_clipping(&audio, 0.99);

        assert!(!clipped_regions.is_empty());
    }

    #[test]
    fn test_detect_fundamental_frequency() {
        // Create a sine wave with known frequency
        let sample_rate = 44100.0;
        let frequency = 440.0; // A4
        let duration = 1.0; // 1 second

        let samples: Vec<f32> = (0..(sample_rate * duration) as usize)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (2.0 * PI * frequency * t).sin() as f32
            })
            .collect();

        let data = ndarray::Array1::from_vec(samples);
        let audio = AudioSamples::new_mono(data, sample_rate as u32);

        let detected_freq = detect_fundamental_frequency(&audio).unwrap();

        if let Some(freq) = detected_freq {
            // Allow some tolerance in frequency detection
            assert!((freq - frequency).abs() < 10.0);
        }
    }
}
