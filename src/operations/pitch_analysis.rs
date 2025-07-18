//! Pitch detection and fundamental frequency analysis implementations.
//!
//! This module provides robust pitch detection algorithms including YIN and
//! autocorrelation-based methods, as well as harmonic analysis and key estimation.

use super::traits::AudioPitchAnalysis;
use super::types::PitchDetectionMethod;
use crate::repr::AudioData;
use crate::{AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, ConvertTo, I24};
use ndarray::Array1;

impl<T: AudioSample> AudioPitchAnalysis<T> for AudioSamples<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: ConvertTo<f64>,
{
    fn detect_pitch_yin(
        &self,
        threshold: f64,
        min_frequency: f64,
        max_frequency: f64,
    ) -> AudioSampleResult<Option<f64>> {
        if threshold < 0.0 || threshold > 1.0 {
            return Err(AudioSampleError::InvalidParameter(
                "Threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        if min_frequency <= 0.0 || max_frequency <= min_frequency {
            return Err(AudioSampleError::InvalidParameter(
                "Invalid frequency range".to_string(),
            ));
        }

        let samples = self.to_mono_samples_f64()?;
        let sample_rate = self.sample_rate() as f64;

        // Calculate tau (lag) limits based on frequency constraints
        let min_tau = (sample_rate / max_frequency) as usize;
        let max_tau = (sample_rate / min_frequency) as usize;

        if max_tau >= samples.len() / 2 {
            return Ok(None);
        }

        let result = yin_pitch_detection(&samples, min_tau, max_tau, threshold);

        Ok(result.map(|tau| sample_rate / tau))
    }

    fn detect_pitch_autocorr(
        &self,
        min_frequency: f64,
        max_frequency: f64,
    ) -> AudioSampleResult<Option<f64>> {
        if min_frequency <= 0.0 || max_frequency <= min_frequency {
            return Err(AudioSampleError::InvalidParameter(
                "Invalid frequency range".to_string(),
            ));
        }

        let samples = self.to_mono_samples_f64()?;
        let sample_rate = self.sample_rate() as f64;

        // Calculate tau (lag) limits based on frequency constraints
        let min_tau = (sample_rate / max_frequency) as usize;
        let max_tau = (sample_rate / min_frequency) as usize;

        if max_tau >= samples.len() / 2 {
            return Ok(None);
        }

        let result = autocorr_pitch_detection(&samples, min_tau, max_tau);

        Ok(result.map(|tau| sample_rate / tau))
    }

    fn track_pitch(
        &self,
        window_size: usize,
        hop_size: usize,
        method: PitchDetectionMethod,
        threshold: f64,
        min_frequency: f64,
        max_frequency: f64,
    ) -> AudioSampleResult<Vec<(f64, Option<f64>)>> {
        if window_size == 0 || hop_size == 0 {
            return Err(AudioSampleError::InvalidParameter(
                "Window size and hop size must be greater than 0".to_string(),
            ));
        }

        let samples = self.to_mono_samples_f64()?;
        let sample_rate = self.sample_rate() as f64;
        let mut results = Vec::new();

        // Process each windowed segment
        for start in (0..samples.len()).step_by(hop_size) {
            let end = (start + window_size).min(samples.len());
            if end - start < window_size / 2 {
                break; // Skip windows that are too short
            }

            let window = &samples[start..end];
            let time_seconds = start as f64 / sample_rate;

            // Create temporary AudioSamples for this window
            let window_data: Vec<T> = window
                .iter()
                .map(|&x| {
                    // Convert f64 back to T using the existing conversion system
                    x.convert_to().unwrap_or_else(|_| {
                        // Fallback to zero value
                        let zero_f64 = 0.0f64;
                        zero_f64.convert_to().unwrap()
                    })
                })
                .collect();
            let window_audio =
                AudioSamples::new_mono(Array1::from(window_data), self.sample_rate());

            let frequency = match method {
                PitchDetectionMethod::Yin => {
                    window_audio.detect_pitch_yin(threshold, min_frequency, max_frequency)?
                }
                PitchDetectionMethod::Autocorrelation => {
                    window_audio.detect_pitch_autocorr(min_frequency, max_frequency)?
                }
                _ => None, // Other methods not implemented yet
            };

            results.push((time_seconds, frequency));
        }

        Ok(results)
    }

    fn harmonic_to_noise_ratio(
        &self,
        fundamental_freq: f64,
        num_harmonics: usize,
    ) -> AudioSampleResult<f64> {
        if fundamental_freq <= 0.0 || num_harmonics == 0 {
            return Err(AudioSampleError::InvalidParameter(
                "Invalid fundamental frequency or number of harmonics".to_string(),
            ));
        }

        let samples = self.to_mono_samples_f64()?;
        let sample_rate = self.sample_rate() as f64;

        // Compute power spectrum
        let spectrum = compute_power_spectrum(&samples)?;
        let freq_resolution = sample_rate / samples.len() as f64;

        // Calculate harmonic and noise power
        let mut harmonic_power = 0.0;
        let mut total_power = 0.0;

        for (i, &power) in spectrum.iter().enumerate() {
            let freq = i as f64 * freq_resolution;
            total_power += power;

            // Check if this frequency is close to any harmonic
            for harmonic in 1..=num_harmonics {
                let harmonic_freq = fundamental_freq * harmonic as f64;
                if (freq - harmonic_freq).abs() < freq_resolution {
                    harmonic_power += power;
                    break;
                }
            }
        }

        let noise_power = total_power - harmonic_power;
        if noise_power <= 0.0 {
            return Ok(f64::INFINITY);
        }

        Ok(10.0 * (harmonic_power / noise_power).log10())
    }

    fn harmonic_analysis(
        &self,
        fundamental_freq: f64,
        num_harmonics: usize,
        tolerance: f64,
    ) -> AudioSampleResult<Vec<f64>> {
        if fundamental_freq <= 0.0 || num_harmonics == 0 {
            return Err(AudioSampleError::InvalidParameter(
                "Invalid fundamental frequency or number of harmonics".to_string(),
            ));
        }

        if tolerance < 0.0 || tolerance > 1.0 {
            return Err(AudioSampleError::InvalidParameter(
                "Tolerance must be between 0.0 and 1.0".to_string(),
            ));
        }

        let samples = self.to_mono_samples_f64()?;
        let sample_rate = self.sample_rate() as f64;

        // Compute power spectrum
        let spectrum = compute_power_spectrum(&samples)?;
        let freq_resolution = sample_rate / samples.len() as f64;

        let mut harmonic_magnitudes = vec![0.0; num_harmonics];

        // Find magnitude at each harmonic frequency
        for harmonic in 1..=num_harmonics {
            let harmonic_freq = fundamental_freq * harmonic as f64;
            let target_bin = (harmonic_freq / freq_resolution).round() as usize;

            // Search within tolerance range
            let tolerance_bins = (tolerance * harmonic_freq / freq_resolution) as usize;
            let start_bin = target_bin.saturating_sub(tolerance_bins);
            let end_bin = (target_bin + tolerance_bins).min(spectrum.len() - 1);

            // Find maximum magnitude in the tolerance range
            let max_magnitude = spectrum[start_bin..=end_bin]
                .iter()
                .fold(0.0f64, |acc, &x| acc.max(x));

            harmonic_magnitudes[harmonic - 1] = max_magnitude;
        }

        // Normalize relative to fundamental
        if harmonic_magnitudes[0] > 0.0 {
            let fundamental_magnitude = harmonic_magnitudes[0];
            for magnitude in &mut harmonic_magnitudes {
                *magnitude /= fundamental_magnitude;
            }
        }

        Ok(harmonic_magnitudes)
    }

    fn estimate_key(&self, window_size: usize, hop_size: usize) -> AudioSampleResult<(usize, f64)> {
        if window_size == 0 || hop_size == 0 {
            return Err(AudioSampleError::InvalidParameter(
                "Window size and hop size must be greater than 0".to_string(),
            ));
        }

        // This is a simplified key estimation using chromagram
        // In a full implementation, this would use more sophisticated methods

        let samples = self.to_mono_samples_f64()?;
        let sample_rate = self.sample_rate() as f64;

        // Accumulate chroma features across time
        let mut chroma_sum = vec![0.0; 12];
        let mut num_windows = 0;

        for start in (0..samples.len()).step_by(hop_size) {
            let end = (start + window_size).min(samples.len());
            if end - start < window_size / 2 {
                break;
            }

            let window = &samples[start..end];
            let chroma = compute_chroma(window, sample_rate)?;

            for (i, &value) in chroma.iter().enumerate() {
                chroma_sum[i] += value;
            }
            num_windows += 1;
        }

        // Average the chroma features
        if num_windows > 0 {
            for value in &mut chroma_sum {
                *value /= num_windows as f64;
            }
        }

        // Find the key with maximum energy
        let (key_index, &max_energy) = chroma_sum
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        // Calculate confidence as the ratio of max energy to total energy
        let total_energy: f64 = chroma_sum.iter().sum();
        let confidence = if total_energy > 0.0 {
            max_energy / total_energy
        } else {
            0.0
        };

        Ok((key_index, confidence))
    }
}

/// Implementation of the YIN pitch detection algorithm.
///
/// YIN uses a difference function and cumulative mean normalized difference
/// to find the most likely fundamental period in the signal.
fn yin_pitch_detection(
    samples: &[f64],
    min_tau: usize,
    max_tau: usize,
    threshold: f64,
) -> Option<f64> {
    let n = samples.len();
    if max_tau >= n / 2 {
        return None;
    }

    // Step 1: Compute difference function
    let mut diff_fn = vec![0.0; max_tau + 1];
    for tau in min_tau..=max_tau {
        let mut sum = 0.0;
        for j in 0..(n - tau) {
            let delta = samples[j] - samples[j + tau];
            sum += delta * delta;
        }
        diff_fn[tau] = sum;
    }

    // Step 2: Compute cumulative mean normalized difference
    let mut cmnd = vec![1.0; max_tau + 1];
    let mut running_sum = 0.0;

    for tau in 1..=max_tau {
        running_sum += diff_fn[tau];
        if running_sum > 0.0 {
            cmnd[tau] = diff_fn[tau] / (running_sum / tau as f64);
        }
    }

    // Step 3: Find the first minimum below threshold
    for tau in min_tau..=max_tau {
        if cmnd[tau] < threshold {
            // Find the actual minimum around this tau
            let mut min_tau = tau;
            let mut min_val = cmnd[tau];

            // Search in a small neighborhood
            let start = (tau.saturating_sub(5)).max(min_tau);
            let end = (tau + 5).min(max_tau);

            for t in start..=end {
                if cmnd[t] < min_val {
                    min_val = cmnd[t];
                    min_tau = t;
                }
            }

            return Some(min_tau as f64);
        }
    }

    None
}

/// Simple autocorrelation-based pitch detection.
///
/// Finds the lag with maximum autocorrelation within the specified range.
fn autocorr_pitch_detection(samples: &[f64], min_tau: usize, max_tau: usize) -> Option<f64> {
    let n = samples.len();
    if max_tau >= n / 2 {
        return None;
    }

    let mut max_corr = 0.0;
    let mut best_tau = 0;

    for tau in min_tau..=max_tau {
        let mut corr = 0.0;
        for i in 0..(n - tau) {
            corr += samples[i] * samples[i + tau];
        }

        if corr > max_corr {
            max_corr = corr;
            best_tau = tau;
        }
    }

    if best_tau > 0 {
        Some(best_tau as f64)
    } else {
        None
    }
}

/// Compute power spectrum using FFT.
fn compute_power_spectrum(samples: &[f64]) -> AudioSampleResult<Vec<f64>> {
    use rustfft::{FftPlanner, num_complex::Complex};

    let n = samples.len();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    // Convert to complex numbers
    let mut buffer: Vec<Complex<f64>> = samples.iter().map(|&x| Complex::new(x, 0.0)).collect();

    // Compute FFT
    fft.process(&mut buffer);

    // Compute power spectrum (magnitude squared)
    let power_spectrum: Vec<f64> = buffer
        .iter()
        .take(n / 2) // Only take positive frequencies
        .map(|c| c.norm_sqr())
        .collect();

    Ok(power_spectrum)
}

/// Compute chroma features (simplified version).
///
/// This is a basic implementation that maps frequency bins to chroma classes.
fn compute_chroma(samples: &[f64], sample_rate: f64) -> AudioSampleResult<Vec<f64>> {
    let spectrum = compute_power_spectrum(samples)?;
    let n = samples.len();
    let freq_resolution = sample_rate / n as f64;

    let mut chroma = vec![0.0; 12];
    let a4_freq = 440.0; // A4 frequency

    for (i, &power) in spectrum.iter().enumerate() {
        let freq = i as f64 * freq_resolution;
        if freq <= 0.0 {
            continue;
        }

        // Convert frequency to MIDI note number
        let midi_note = 12.0 * (freq / a4_freq).log2() + 69.0;

        // Map to chroma class (0-11)
        let chroma_class = ((midi_note.round() as i32) % 12) as usize;

        if chroma_class < 12 {
            chroma[chroma_class] += power;
        }
    }

    Ok(chroma)
}

impl<T: AudioSample> AudioSamples<T>
where
    T: ConvertTo<f64>,
{
    /// Helper method to convert to mono f64 samples for processing.
    pub fn to_mono_samples_f64(&self) -> AudioSampleResult<Vec<f64>> {
        let samples = match &self.data {
            AudioData::Mono(samples) => samples
                .iter()
                .map(|&x| x.convert_to().unwrap_or(0.0))
                .collect(),
            AudioData::MultiChannel(samples) => {
                // Average all channels to create mono
                let num_channels = samples.nrows();
                let num_samples = samples.ncols();
                let mut mono_samples = vec![0.0; num_samples];

                for channel in 0..num_channels {
                    for sample in 0..num_samples {
                        mono_samples[sample] +=
                            samples[[channel, sample]].convert_to().unwrap_or(0.0);
                    }
                }

                // Average the channels
                for sample in &mut mono_samples {
                    *sample /= num_channels as f64;
                }

                mono_samples
            }
        };

        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::AudioPitchAnalysis;
    use ndarray::Array1;
    use std::f64::consts::PI;

    #[test]
    fn test_pitch_detection_yin() {
        // Create a simple sine wave at 440 Hz
        let sample_rate = 44100;
        let duration = 1.0; // 1 second
        let frequency = 440.0; // A4
        let samples_count = (sample_rate as f64 * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate as f64;
            let value = (2.0 * PI * frequency * t).sin();
            samples.push(value as f32);
        }

        let audio = AudioSamples::new_mono(Array1::from(samples), sample_rate);

        // Test YIN pitch detection
        let detected_pitch = audio.detect_pitch_yin(0.1, 80.0, 1000.0).unwrap();

        // Should detect approximately 440 Hz
        assert!(detected_pitch.is_some());
        let pitch = detected_pitch.unwrap();
        assert!((pitch - 440.0).abs() < 10.0); // Allow 10 Hz tolerance
    }

    #[test]
    fn test_pitch_detection_autocorr() {
        // Create a simple sine wave at 220 Hz
        let sample_rate = 44100;
        let duration = 1.0; // 1 second
        let frequency = 220.0; // A3
        let samples_count = (sample_rate as f64 * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate as f64;
            let value = (2.0 * PI * frequency * t).sin();
            samples.push(value as f32);
        }

        let audio = AudioSamples::new_mono(Array1::from(samples), sample_rate);

        // Test autocorrelation pitch detection
        let detected_pitch = audio.detect_pitch_autocorr(80.0, 1000.0).unwrap();

        // Should detect approximately 220 Hz
        assert!(detected_pitch.is_some());
        let pitch = detected_pitch.unwrap();
        assert!((pitch - 220.0).abs() < 10.0); // Allow 10 Hz tolerance
    }

    #[test]
    fn test_pitch_tracking() {
        // Create a simple sine wave at 440 Hz
        let sample_rate = 44100;
        let duration = 0.5; // 0.5 seconds
        let frequency = 440.0; // A4
        let samples_count = (sample_rate as f64 * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate as f64;
            let value = (2.0 * PI * frequency * t).sin();
            samples.push(value as f32);
        }

        let audio = AudioSamples::new_mono(Array1::from(samples), sample_rate);

        // Test pitch tracking
        let window_size = 2048;
        let hop_size = 512;
        let pitch_track = audio
            .track_pitch(
                window_size,
                hop_size,
                PitchDetectionMethod::Yin,
                0.1,
                80.0,
                1000.0,
            )
            .unwrap();

        // Should have multiple pitch estimates
        assert!(!pitch_track.is_empty());

        // Most should detect around 440 Hz
        let detected_pitches: Vec<f64> =
            pitch_track.iter().filter_map(|(_, pitch)| *pitch).collect();

        assert!(!detected_pitches.is_empty());

        // Average should be close to 440 Hz
        let avg_pitch = detected_pitches.iter().sum::<f64>() / detected_pitches.len() as f64;
        assert!((avg_pitch - 440.0).abs() < 20.0); // Allow 20 Hz tolerance for windowed analysis
    }

    #[test]
    fn test_harmonic_analysis() {
        // Create a sawtooth wave (rich in harmonics) at 220 Hz
        let sample_rate = 44100;
        let duration = 1.0; // 1 second
        let frequency = 220.0; // A3
        let samples_count = (sample_rate as f64 * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate as f64;
            // Sawtooth wave approximation using first few harmonics
            let mut value = 0.0;
            for harmonic in 1..10 {
                value += (2.0 * PI * frequency * harmonic as f64 * t).sin() / harmonic as f64;
            }
            samples.push(value as f32);
        }

        let audio = AudioSamples::new_mono(Array1::from(samples), sample_rate);

        // Test harmonic analysis
        let harmonics = audio.harmonic_analysis(frequency, 5, 0.1).unwrap();

        // Should have 5 harmonic components
        assert_eq!(harmonics.len(), 5);

        // Fundamental should be normalized to 1.0
        assert!((harmonics[0] - 1.0).abs() < 0.1);

        // Higher harmonics should be present but weaker
        for i in 1..harmonics.len() {
            assert!(harmonics[i] > 0.0);
            assert!(harmonics[i] <= harmonics[0]); // Should be weaker than fundamental
        }
    }

    #[test]
    fn test_silence_detection() {
        // Test with silence (should return None)
        let audio = AudioSamples::new_mono(Array1::<f32>::zeros(44100), 44100);

        let detected_pitch = audio.detect_pitch_yin(0.1, 80.0, 1000.0).unwrap();
        assert!(detected_pitch.is_none());

        let detected_pitch_autocorr = audio.detect_pitch_autocorr(80.0, 1000.0).unwrap();
        assert!(detected_pitch_autocorr.is_none());
    }

    #[test]
    fn test_noise_robustness() {
        // Create a noisy sine wave at 440 Hz
        let sample_rate = 44100;
        let duration = 1.0; // 1 second
        let frequency = 440.0; // A4
        let samples_count = (sample_rate as f64 * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate as f64;
            let sine_value = (2.0 * PI * frequency * t).sin();
            // Add some noise
            let noise = (i as f64 * 0.1).sin() * 0.1; // Small amount of noise
            let value = sine_value + noise;
            samples.push(value as f32);
        }

        let audio = AudioSamples::new_mono(Array1::from(samples), sample_rate);

        // YIN should be more robust to noise
        let detected_pitch = audio.detect_pitch_yin(0.1, 80.0, 1000.0).unwrap();

        // Should still detect approximately 440 Hz despite noise
        assert!(detected_pitch.is_some());
        let pitch = detected_pitch.unwrap();
        assert!((pitch - 440.0).abs() < 20.0); // Allow larger tolerance for noisy signal
    }

    #[test]
    fn test_parameter_validation() {
        let audio = AudioSamples::new_mono(Array1::from(vec![1.0f32, 2.0, 3.0]), 44100);

        // Test invalid threshold
        assert!(audio.detect_pitch_yin(-0.1, 80.0, 1000.0).is_err());
        assert!(audio.detect_pitch_yin(1.5, 80.0, 1000.0).is_err());

        // Test invalid frequency range
        assert!(audio.detect_pitch_yin(0.1, -80.0, 1000.0).is_err());
        assert!(audio.detect_pitch_yin(0.1, 1000.0, 800.0).is_err());

        // Test invalid window parameters
        assert!(
            audio
                .track_pitch(0, 512, PitchDetectionMethod::Yin, 0.1, 80.0, 1000.0)
                .is_err()
        );
        assert!(
            audio
                .track_pitch(2048, 0, PitchDetectionMethod::Yin, 0.1, 80.0, 1000.0)
                .is_err()
        );

        // Test invalid harmonic analysis parameters
        assert!(audio.harmonic_analysis(-440.0, 5, 0.1).is_err());
        assert!(audio.harmonic_analysis(440.0, 0, 0.1).is_err());
        assert!(audio.harmonic_analysis(440.0, 5, -0.1).is_err());
        assert!(audio.harmonic_analysis(440.0, 5, 1.5).is_err());
    }
}
