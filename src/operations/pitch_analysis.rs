//! Pitch detection and fundamental frequency analysis implementations.
//!
//! This module provides robust pitch detection algorithms including YIN and
//! autocorrelation-based methods, as well as harmonic analysis and key estimation.

use crate::operations::traits::AudioPitchAnalysis;
use crate::operations::types::PitchDetectionMethod;
use crate::repr::AudioData;
use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ConvertTo,
    I24, RealFloat, to_precision,
};

use ndarray::Array1;

impl<'a, T: AudioSample> AudioPitchAnalysis<T> for AudioSamples<'a, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'b> AudioSamples<'b, T>: AudioTypeConversion<'b, T>,
{
    fn detect_pitch_yin<F>(
        &self,
        threshold: F,
        min_frequency: F,
        max_frequency: F,
    ) -> AudioSampleResult<Option<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        if !(F::zero()..=F::one()).contains(&threshold) {
            return Err(AudioSampleError::InvalidParameter(
                "Threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        if min_frequency <= F::zero() || max_frequency <= min_frequency {
            return Err(AudioSampleError::InvalidParameter(
                "Invalid frequency range".to_string(),
            ));
        }

        let samples: Vec<F> = self.to_mono_float_samples()?;
        let sample_rate_f: F = to_precision(self.sample_rate);

        // Calculate tau (lag) limits based on frequency constraints
        let min_tau = (sample_rate_f / max_frequency).to_usize().expect("both sample_rate and max_frequency are non-zero positive numbers so their product will be non-zero and positve. ");
        let max_tau = (sample_rate_f / min_frequency).to_usize().expect("both sample_rate and min_frequency are non-zero positive numbers so their product will be non-zero and positve. ");

        if max_tau >= samples.len() / 2 {
            return Ok(None);
        }

        let result = yin_pitch_detection(&samples, min_tau, max_tau, threshold);

        Ok(result.map(|tau| sample_rate_f / tau))
    }

    fn detect_pitch_autocorr<F>(
        &self,
        min_frequency: F,
        max_frequency: F,
    ) -> AudioSampleResult<Option<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        if min_frequency <= F::one() || max_frequency <= min_frequency {
            return Err(AudioSampleError::InvalidParameter(
                "Invalid frequency range".to_string(),
            ));
        }

        let samples: Vec<F> = self.to_mono_float_samples()?;
        let sample_rate = to_precision::<F, _>(self.sample_rate);

        // Calculate tau (lag) limits based on frequency constraints
        let min_tau = (sample_rate / max_frequency).to_usize().expect("both sample_rate and max_frequency are non-zero positive numbers so their product will be non-zero and positve. ");
        let max_tau = (sample_rate / min_frequency).to_usize().expect("both sample_rate and min_frequency are non-zero positive numbers so their product will be non-zero and positve. ");

        if max_tau >= samples.len() / 2 {
            return Ok(None);
        }

        let result = autocorr_pitch_detection(&samples, min_tau, max_tau);

        Ok(result.map(|tau| sample_rate / tau))
    }

    fn track_pitch<F>(
        &self,
        window_size: usize,
        hop_size: usize,
        method: PitchDetectionMethod,
        threshold: F,
        min_frequency: F,
        max_frequency: F,
    ) -> AudioSampleResult<Vec<(F, Option<F>)>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        if window_size == 0 || hop_size == 0 {
            return Err(AudioSampleError::InvalidParameter(
                "Window size and hop size must be greater than 0".to_string(),
            ));
        }

        let samples: Vec<F> = self.to_mono_float_samples()?;
        let sample_rate: F = to_precision(self.sample_rate());
        let mut results = Vec::new();

        // Process each windowed segment
        for start in (0..samples.len()).step_by(hop_size) {
            let end = (start + window_size).min(samples.len());
            if end - start < window_size / 2 {
                break; // Skip windows that are too short
            }

            let window = &samples[start..end];
            let time_seconds = to_precision::<F, _>(start) / sample_rate;

            // Create temporary AudioSamples for this window
            let mut window_data = Vec::with_capacity(window.len());

            for x in window {
                // Convert f64 to T using the existing conversion system
                window_data.push(x.convert_to()?);
            }

            let window_audio =
                AudioSamples::new_mono(Array1::from(window_data).into(), self.sample_rate());

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

    fn harmonic_to_noise_ratio<F>(
        &self,
        fundamental_freq: F,
        num_harmonics: usize,
    ) -> AudioSampleResult<F>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        if fundamental_freq <= F::zero() || num_harmonics == 0 {
            return Err(AudioSampleError::InvalidParameter(
                "Invalid fundamental frequency or number of harmonics".to_string(),
            ));
        }

        let samples = self.to_mono_float_samples()?;
        let sample_rate = to_precision::<F, _>(self.sample_rate);

        // Compute power spectrum
        let spectrum = compute_power_spectrum(&samples)?;
        let freq_resolution = sample_rate / to_precision::<F, _>(samples.len());

        // Calculate harmonic and noise power
        let mut harmonic_power = F::zero();
        let mut total_power = F::zero();

        for (i, &power) in spectrum.iter().enumerate() {
            let freq = to_precision::<F, _>(i) * freq_resolution;
            total_power += power;

            // Check if this frequency is close to any harmonic
            for harmonic in 1..=num_harmonics {
                let harmonic_freq = fundamental_freq * to_precision::<F, _>(harmonic);
                if (freq - harmonic_freq).abs() < freq_resolution {
                    harmonic_power += power;
                    break;
                }
            }
        }

        let noise_power = total_power - harmonic_power;
        if noise_power <= F::zero() {
            return Ok(F::infinity());
        }

        Ok(to_precision::<F, _>(10.0) * (harmonic_power / noise_power).log10())
    }

    fn harmonic_analysis<F>(
        &self,
        fundamental_freq: F,
        num_harmonics: usize,
        tolerance: F,
    ) -> AudioSampleResult<Vec<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        if fundamental_freq <= F::zero() || num_harmonics == 0 {
            return Err(AudioSampleError::InvalidParameter(
                "Invalid fundamental frequency or number of harmonics".to_string(),
            ));
        }

        if !(F::zero()..=F::one()).contains(&tolerance) {
            return Err(AudioSampleError::InvalidParameter(
                "Tolerance must be between 0.0 and 1.0".to_string(),
            ));
        }

        let samples: Vec<F> = self.to_mono_float_samples()?;
        let sample_rate = to_precision::<F, _>(self.sample_rate);

        // Compute power spectrum
        let spectrum = compute_power_spectrum(&samples)?;
        let freq_resolution = sample_rate / to_precision(samples.len());

        let mut harmonic_magnitudes = vec![F::zero(); num_harmonics];

        // Find magnitude at each harmonic frequency
        for harmonic in 1..=num_harmonics {
            let harmonic_freq = fundamental_freq * to_precision(harmonic);
            let target_bin = (harmonic_freq / freq_resolution).round().to_usize().expect("both harmonic and freq are non-zero positive numbers so their product will be non-zero and positve. ");

            // Search within tolerance range
            let tolerance_bins = (tolerance * harmonic_freq / freq_resolution).to_usize().expect("both harmonic and freq are non-zero positive numbers so their product will be non-zero and positve. ");
            let start_bin = target_bin.saturating_sub(tolerance_bins);
            let end_bin = (target_bin + tolerance_bins).min(spectrum.len() - 1);

            // Find maximum magnitude in the tolerance range
            let max_magnitude = spectrum[start_bin..=end_bin]
                .iter()
                .fold(F::zero(), |acc, &x| acc.max(x));

            harmonic_magnitudes[harmonic - 1] = max_magnitude;
        }

        // Normalize relative to fundamental
        if harmonic_magnitudes[0] > F::zero() {
            let fundamental_magnitude = harmonic_magnitudes[0];
            for magnitude in &mut harmonic_magnitudes {
                *magnitude /= fundamental_magnitude;
            }
        }

        Ok(harmonic_magnitudes)
    }

    fn estimate_key<F>(&self, window_size: usize, hop_size: usize) -> AudioSampleResult<(usize, F)>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        if window_size == 0 || hop_size == 0 {
            return Err(AudioSampleError::InvalidParameter(
                "Window size and hop size must be greater than 0".to_string(),
            ));
        }

        // Krumhansl-Schmuckler key-finding algorithm implementation
        let samples: Vec<F> = self.to_mono_float_samples()?;
        let sample_rate: F = to_precision(self.sample_rate);

        // Krumhansl-Schmuckler key profiles (major and minor)
        let major_profile: [f64; 12] = [
            6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
        ];
        let minor_profile: [f64; 12] = [
            6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
        ];

        // Accumulate chroma features across time with better windowing
        let mut chroma_sum = vec![F::zero(); 12];
        let mut num_windows = 0;

        let two_pi = to_precision::<F, _>(2.0) * F::PI();

        for start in (0..samples.len()).step_by(hop_size) {
            let end = (start + window_size).min(samples.len());
            if end - start < window_size / 2 {
                break;
            }

            let window = &samples[start..end];

            // Apply Hann window for better spectral analysis
            let windowed: Vec<F> = window
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let hann = to_precision::<F, _>(0.5)
                        * (F::one()
                            - (two_pi * to_precision(i) / to_precision(window.len() - 1)).cos());
                    x * hann
                })
                .collect();

            let chroma = compute_enhanced_chroma(&windowed, sample_rate)?;

            for (i, &value) in chroma.iter().enumerate() {
                chroma_sum[i] += value;
            }
            num_windows += 1;
        }

        // Average and normalize the chroma features
        if num_windows > 0 {
            for value in &mut chroma_sum {
                *value /= to_precision(num_windows);
            }
        }

        // Normalize chroma vector
        let chroma_sum_total: F = chroma_sum.iter().fold(F::zero(), |acc, &x| acc + x);
        if chroma_sum_total > F::zero() {
            for value in &mut chroma_sum {
                *value /= chroma_sum_total;
            }
        }

        // Calculate correlation with all 24 keys (12 major + 12 minor)
        let mut best_correlation = -F::one();
        let mut best_key = 0;
        let mut is_major = true;

        // Test all major keys
        for tonic in 0..12 {
            let correlation = calculate_correlation(&chroma_sum, &major_profile, tonic);
            if correlation > best_correlation {
                best_correlation = correlation;
                best_key = tonic;
                is_major = true;
            }
        }

        // Test all minor keys
        for tonic in 0..12 {
            let correlation = calculate_correlation(&chroma_sum, &minor_profile, tonic);
            if correlation > best_correlation {
                best_correlation = correlation;
                best_key = tonic;
                is_major = false;
            }
        }

        // Encode key: 0-11 for major keys (C, C#, D, ..., B), 12-23 for minor keys
        let encoded_key = if is_major { best_key } else { best_key + 12 };

        // Convert correlation to confidence (normalize to 0-1 range)
        let confidence = (best_correlation + F::one()) / to_precision::<F, _>(2.0);

        Ok((encoded_key, confidence))
    }
}

/// Implementation of the YIN pitch detection algorithm.
///
/// YIN uses a difference function and cumulative mean normalized difference
/// to find the most likely fundamental period in the signal.
pub fn yin_pitch_detection<F: RealFloat>(
    samples: &[F],
    min_tau: usize,
    max_tau: usize,
    threshold: F,
) -> Option<F> {
    let n = samples.len();
    if max_tau >= n / 2 {
        return None;
    }

    // Step 1: Compute difference function
    let mut diff_fn = vec![F::zero(); max_tau + 1];
    for tau in min_tau..=max_tau {
        let mut sum = F::zero();
        for j in 0..(n - tau) {
            let delta = samples[j] - samples[j + tau];
            sum += delta * delta;
        }
        diff_fn[tau] = sum;
    }

    // Step 2: Compute cumulative mean normalized difference
    let mut cmnd = vec![F::one(); max_tau + 1];
    let mut running_sum = F::zero();

    for tau in 1..=max_tau {
        running_sum += diff_fn[tau];
        if running_sum > F::zero() {
            cmnd[tau] = diff_fn[tau] / (running_sum / to_precision::<F, _>(tau));
        }
    }

    // Step 3: Find the first minimum below threshold
    for tau in min_tau..=max_tau {
        if cmnd[tau] < threshold {
            // Find the actual minimum around this tau
            let mut min_tau = tau;
            let mut min_val = cmnd[tau];

            // Search in a small neighborhood
            let start = tau.saturating_sub(5).max(min_tau);
            let end = (tau + 5).min(max_tau);

            for (t, &val) in cmnd.iter().enumerate().skip(start).take(end - start + 1) {
                if val < min_val {
                    min_val = val;
                    min_tau = t;
                }
            }

            return Some(to_precision(min_tau));
        }
    }

    None
}

/// Simple autocorrelation-based pitch detection.
///
/// Finds the lag with maximum autocorrelation within the specified range.
pub fn autocorr_pitch_detection<F: RealFloat>(
    samples: &[F],
    min_tau: usize,
    max_tau: usize,
) -> Option<F> {
    let n = samples.len();
    if max_tau >= n / 2 {
        return None;
    }

    let mut max_corr = F::zero();
    let mut best_tau = 0;

    for tau in min_tau..=max_tau {
        let mut corr = F::zero();
        for i in 0..(n - tau) {
            corr += samples[i] * samples[i + tau];
        }

        if corr > max_corr {
            max_corr = corr;
            best_tau = tau;
        }
    }

    if best_tau > 0 {
        Some(to_precision::<F, _>(best_tau))
    } else {
        None
    }
}

/// Compute power spectrum using FFT.
pub fn compute_power_spectrum<F: RealFloat>(samples: &[F]) -> AudioSampleResult<Vec<F>> {
    use rustfft::{FftPlanner, num_complex::Complex};

    let n = samples.len();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    // Convert to complex numbers
    let mut buffer: Vec<Complex<F>> = samples
        .iter()
        .map(|&x| Complex::new(x, F::zero()))
        .collect();

    // Compute FFT
    fft.process(&mut buffer);

    // Compute power spectrum (magnitude squared)
    let power_spectrum: Vec<F> = buffer
        .iter()
        .take(n / 2) // Only take positive frequencies
        .map(|c| c.norm_sqr())
        .collect();

    Ok(power_spectrum)
}

/// Enhanced chroma computation with better harmonic weighting
fn compute_enhanced_chroma<F: RealFloat>(
    samples: &[F],
    sample_rate: F,
) -> AudioSampleResult<Vec<F>> {
    let spectrum = compute_power_spectrum(samples)?;
    let n = samples.len();
    let freq_resolution = sample_rate / to_precision(n);

    let mut chroma = vec![F::zero(); 12];
    let a4_freq: F = to_precision::<F, _>(440.0); // A4 frequency

    for (i, &power) in spectrum.iter().enumerate() {
        let freq = to_precision::<F, _>(i) * freq_resolution;
        if freq <= F::zero() || freq > sample_rate / to_precision::<F, _>(2.0) {
            continue;
        }

        // Convert frequency to MIDI note number
        let midi_note =
            to_precision::<F, _>(12.0) * (freq / a4_freq).log2() + to_precision::<F, _>(69.0);

        // Map to chroma class (0-11) with better rounding
        let chroma_class =
            (midi_note.round().to_i32().expect("should not fail")).rem_euclid(12) as usize;

        // Apply harmonic weighting - higher harmonics contribute less
        let octave = (midi_note / to_precision::<F, _>(12.0)).floor();
        let harmonic_weight = F::one() / (F::one() + octave.abs() / to_precision::<F, _>(3.0)); // Decay with octave distance

        if chroma_class < 12 {
            chroma[chroma_class] += power * harmonic_weight;
        }
    }

    Ok(chroma)
}

/// Calculate Pearson correlation between chroma vector and key profile
fn calculate_correlation<F: RealFloat>(chroma: &[F], profile: &[f64; 12], tonic: usize) -> F {
    // Rotate the profile to match the tonic
    let rotated_profile: Vec<F> = (0..12)
        .map(|i| to_precision::<F, _>(profile[(i + tonic) % 12]))
        .collect();

    // Calculate means
    let chroma_mean: F =
        chroma.iter().fold(F::zero(), |acc, &x| acc + x) / to_precision::<F, _>(chroma.len());
    let profile_mean: F = rotated_profile.iter().fold(F::zero(), |acc, &x| acc + x)
        / to_precision(rotated_profile.len());

    // Calculate correlation components
    let mut numerator = F::zero();
    let mut chroma_variance = F::zero();
    let mut profile_variance = F::zero();

    for i in 0..12 {
        let chroma_diff = chroma[i] - chroma_mean;
        let profile_diff = rotated_profile[i] - profile_mean;

        numerator += chroma_diff * profile_diff;
        chroma_variance += chroma_diff * chroma_diff;
        profile_variance += profile_diff * profile_diff;
    }

    let denominator = (chroma_variance * profile_variance).sqrt();

    if denominator > F::zero() {
        numerator / denominator
    } else {
        F::zero()
    }
}

/// Compute chroma features (simplified version).
///
/// This is a basic implementation that maps frequency bins to chroma classes.
pub fn compute_chroma<F: RealFloat>(samples: &[F], sample_rate: F) -> AudioSampleResult<Vec<F>> {
    let spectrum = compute_power_spectrum(samples)?;
    let n = samples.len();
    let freq_resolution = sample_rate / to_precision(n);

    let mut chroma = vec![F::zero(); 12];
    let a4_freq: F = to_precision::<F, _>(440.0); // A4 frequency

    for (i, &power) in spectrum.iter().enumerate() {
        let freq = to_precision::<F, _>(i) * freq_resolution;
        if freq <= F::zero() {
            continue;
        }

        // Convert frequency to MIDI note number
        let midi_note: F =
            to_precision::<F, _>(12.0) * (freq / a4_freq).log2() + to_precision::<F, _>(69.0);

        // Map to chroma class (0-11)
        let chroma_class = ((midi_note
            .round()
            .to_i32()
            .expect("Midi note should be valid"))
            % 12) as usize;

        if chroma_class < 12 {
            chroma[chroma_class] += power;
        }
    }

    Ok(chroma)
}

impl<'a, T: AudioSample> AudioSamples<'a, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'b> AudioSamples<'b, T>: AudioTypeConversion<'b, T>,
{
    /// Helper method to convert to mono f64 samples for processing.
    pub fn to_mono_float_samples<F>(&self) -> AudioSampleResult<Vec<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        let samples = match &self.data {
            AudioData::Mono(samples) => samples
                .iter()
                .map(|&x| x.convert_to().unwrap_or_default())
                .collect(),
            AudioData::Multi(_) => {
                // Average all channels to create mono
                let num_samples = self.samples_per_channel();
                let mut mono_samples = vec![F::zero(); num_samples];

                // Access channels directly to avoid lifetime issues
                if let AudioData::Multi(channels) = &self.data {
                    for channel in channels.axis_iter(ndarray::Axis(0)) {
                        for (i, sample) in channel.iter().enumerate() {
                            let x = sample.convert_to()?;
                            mono_samples[i] += x;
                        }
                    }
                }

                // Average the channels
                let num_channels = self.num_channels();
                for sample in &mut mono_samples {
                    *sample /= to_precision(num_channels);
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
    use crate::operations::traits::AudioPitchAnalysis;
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

        let audio = AudioSamples::new_mono(Array1::from(samples).into(), sample_rate);

        // Test YIN pitch detection
        let detected_pitch = audio.detect_pitch_yin(0.1, 80.0, 1000.0).unwrap();

        // Should detect approximately 440 Hz
        assert!(detected_pitch.is_some());
        let pitch: f64 = detected_pitch.unwrap();
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

        let audio = AudioSamples::new_mono(Array1::from(samples).into(), sample_rate);

        // Test autocorrelation pitch detection
        let detected_pitch = audio.detect_pitch_autocorr(80.0, 1000.0).unwrap();

        // Should detect approximately 220 Hz
        assert!(detected_pitch.is_some());
        let pitch: f64 = detected_pitch.unwrap();
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

        let audio = AudioSamples::new_mono(Array1::from(samples).into(), sample_rate);

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

        let audio = AudioSamples::new_mono(Array1::from(samples).into(), sample_rate);

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
        let audio = AudioSamples::new_mono(Array1::<f32>::zeros(44100).into(), 44100);

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

        let audio = AudioSamples::new_mono(Array1::from(samples).into(), sample_rate);

        // YIN should be more robust to noise
        let detected_pitch = audio.detect_pitch_yin(0.1, 80.0, 1000.0).unwrap();

        // Should still detect approximately 440 Hz despite noise
        assert!(detected_pitch.is_some());
        let pitch: f64 = detected_pitch.unwrap();
        assert!((pitch - 440.0).abs() < 20.0); // Allow larger tolerance for noisy signal
    }

    #[test]
    fn test_parameter_validation() {
        let audio = AudioSamples::new_mono(Array1::from(vec![1.0f32, 2.0, 3.0]).into(), 44100);

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
