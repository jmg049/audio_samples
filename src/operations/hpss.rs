//! Harmonic/Percussive Source Separation (HPSS) for AudioSamples.
//!
//! HPSS separates audio signals into harmonic and percussive components using
//! Short-Time Fourier Transform (STFT) magnitude median filtering. This module
//! implements the core HPSS algorithm based on Fitzgerald's method.
//!
//! ## Algorithm Overview
//!
//! 1. Compute STFT magnitude spectrogram of the input signal
//! 2. Apply median filtering along time axis to enhance harmonic content
//! 3. Apply median filtering along frequency axis to enhance percussive content
//! 4. Generate binary or soft masks based on the filtered spectrograms
//! 5. Apply masks to the original complex STFT and reconstruct time-domain signals
//!
//! ## References
//!
//! - Fitzgerald, D. (2010). "Harmonic/percussive separation using median filtering"
//! - Müller, M. (2015). "Fundamentals of Music Processing", Section 8.4

use crate::operations::traits::{AudioDecomposition, AudioTransforms};
use crate::operations::types::{HpssConfig, WindowType};
use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ConvertTo,
    I24, ParameterError, RealFloat,
};

use ndarray::{Array2, s};
use rustfft::{FftNum, num_complex::Complex};

impl<T: AudioSample> AudioDecomposition<T> for AudioSamples<'_, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
{
    fn hpss<F: RealFloat>(
        &self,
        config: &HpssConfig<F>,
    ) -> AudioSampleResult<(AudioSamples<'static, T>, AudioSamples<'static, T>)>
    where
        F: FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>,
        for<'a> AudioSamples<'a, T>: AudioTypeConversion<'a, T> + AudioTransforms<T>,
    {
        // Validate configuration
        config.validate(crate::to_precision::<F, f64>(self.sample_rate.get() as f64))?;

        // Check minimum signal length
        let min_length = config.win_size;
        if self.samples_per_channel() < min_length {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "signal_length",
                format!(
                    "Signal too short ({} samples), need at least {} samples for window size",
                    self.samples_per_channel(),
                    min_length
                ),
            )));
        }

        // Perform HPSS separation
        let (harmonic_audio, percussive_audio) = perform_hpss(self, config)?;

        Ok((harmonic_audio, percussive_audio))
    }
}

/// Core HPSS implementation function.
///
/// Performs the actual harmonic/percussive separation using STFT and median filtering.
fn perform_hpss<T, F>(
    audio: &AudioSamples<'_, T>,
    config: &HpssConfig<F>,
) -> AudioSampleResult<(AudioSamples<'static, T>, AudioSamples<'static, T>)>
where
    T: AudioSample + ConvertTo<F>,
    F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<'a, T>: AudioTransforms<T>,
{
    // Step 1: Compute STFT of the input signal
    let stft_result = audio.stft(config.win_size, config.hop_size, WindowType::Hanning)?;

    // Step 2: Compute magnitude spectrogram
    let magnitude_spec = compute_magnitude_spectrogram(&stft_result);

    // Step 3: Apply median filtering to create harmonic and percussive enhanced spectrograms
    let harmonic_spec = median_filter_time_axis(&magnitude_spec, config.median_filter_harmonic);
    let percussive_spec = median_filter_freq_axis(&magnitude_spec, config.median_filter_percussive);

    // Step 4: Generate separation masks
    let (harmonic_mask, percussive_mask) =
        generate_separation_masks(&harmonic_spec, &percussive_spec, config.mask_softness);

    // Step 5: Apply masks and reconstruct signals
    let harmonic_stft = apply_mask_to_stft(&stft_result, &harmonic_mask);
    let percussive_stft = apply_mask_to_stft(&stft_result, &percussive_mask);

    // Step 6: Inverse STFT to get time domain signals
    let harmonic_audio: AudioSamples<'static, T> = AudioSamples::istft(
        &harmonic_stft,
        config.hop_size,
        WindowType::Hanning,
        audio.sample_rate.get() as usize,
        true,
    )?;
    let percussive_audio: AudioSamples<'static, T> = AudioSamples::istft(
        &percussive_stft,
        config.hop_size,
        WindowType::Hanning,
        audio.sample_rate.get() as usize,
        true,
    )?;

    Ok((harmonic_audio, percussive_audio))
}

/// Compute magnitude spectrogram from complex STFT result.
fn compute_magnitude_spectrogram<F: RealFloat>(stft: &Array2<Complex<F>>) -> Array2<F> {
    stft.mapv(|c| c.norm())
}

/// Apply median filtering along the time axis to enhance harmonic components.
///
/// Harmonic components tend to be stable over time, so median filtering along
/// the time axis preserves sustained tonal content while suppressing transients.
fn median_filter_time_axis<F: RealFloat>(spectrogram: &Array2<F>, kernel_size: usize) -> Array2<F> {
    let (n_freq_bins, n_time_frames) = spectrogram.dim();
    let mut filtered = Array2::zeros((n_freq_bins, n_time_frames));

    // Process each frequency bin independently
    for freq_idx in 0..n_freq_bins {
        let freq_row = spectrogram.slice(s![freq_idx, ..]);
        let filtered_row = median_filter_1d(&freq_row.to_vec(), kernel_size);
        for (time_idx, &val) in filtered_row.iter().enumerate() {
            filtered[[freq_idx, time_idx]] = val;
        }
    }

    filtered
}

/// Apply median filtering along the frequency axis to enhance percussive components.
///
/// Percussive components tend to have broadband characteristics, so median filtering
/// along the frequency axis preserves transients while suppressing tonal content.
fn median_filter_freq_axis<F: RealFloat>(spectrogram: &Array2<F>, kernel_size: usize) -> Array2<F> {
    let (n_freq_bins, n_time_frames) = spectrogram.dim();
    let mut filtered = Array2::zeros((n_freq_bins, n_time_frames));

    // Process each time frame independently
    for time_idx in 0..n_time_frames {
        let time_col = spectrogram.slice(s![.., time_idx]);
        let filtered_col = median_filter_1d(&time_col.to_vec(), kernel_size);
        for (freq_idx, &val) in filtered_col.iter().enumerate() {
            filtered[[freq_idx, time_idx]] = val;
        }
    }

    filtered
}

/// Simple 1D median filter implementation.
///
/// Uses a sliding window approach with border handling via reflection.
fn median_filter_1d<F: RealFloat>(signal: &[F], kernel_size: usize) -> Vec<F> {
    if kernel_size == 0 {
        return signal.to_vec();
    }
    if kernel_size == 1 {
        return signal.to_vec();
    }

    let len = signal.len();
    if len == 0 {
        return Vec::new();
    }

    let half_kernel = kernel_size / 2;
    let mut filtered = Vec::with_capacity(len);

    for i in 0..len {
        let mut window = Vec::with_capacity(kernel_size);

        // Collect values in the kernel window with reflection padding
        for j in 0..kernel_size {
            let idx = i as i32 + j as i32 - half_kernel as i32;
            let reflected_idx = if idx < 0 {
                (-idx) as usize
            } else if idx >= len as i32 {
                len - 2 - (idx - len as i32) as usize
            } else {
                idx as usize
            };

            // Clamp to valid range
            let safe_idx = reflected_idx.min(len - 1);
            window.push(signal[safe_idx]);
        }

        // Compute median
        window.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if kernel_size % 2 == 1 {
            window[kernel_size / 2]
        } else {
            (window[kernel_size / 2 - 1] + window[kernel_size / 2]) / crate::to_precision(2.0)
        };

        filtered.push(median);
    }

    filtered
}

/// Generate separation masks from harmonic and percussive spectrograms.
///
/// Creates binary or soft masks based on the relative strengths of harmonic
/// and percussive components at each time-frequency bin.
fn generate_separation_masks<F: RealFloat>(
    harmonic_spec: &Array2<F>,
    percussive_spec: &Array2<F>,
    mask_softness: F,
) -> (Array2<F>, Array2<F>) {
    let (n_freq_bins, n_time_frames) = harmonic_spec.dim();
    let mut harmonic_mask = Array2::zeros((n_freq_bins, n_time_frames));
    let mut percussive_mask = Array2::zeros((n_freq_bins, n_time_frames));

    let epsilon = crate::to_precision(1e-10); // Small constant to avoid division by zero

    for freq_idx in 0..n_freq_bins {
        for time_idx in 0..n_time_frames {
            let h_val = harmonic_spec[[freq_idx, time_idx]];
            let p_val = percussive_spec[[freq_idx, time_idx]];

            let total = h_val + p_val + epsilon;

            if mask_softness == F::zero() {
                // Hard masking: binary decision
                if h_val >= p_val {
                    harmonic_mask[[freq_idx, time_idx]] = F::one();
                    percussive_mask[[freq_idx, time_idx]] = F::zero();
                } else {
                    harmonic_mask[[freq_idx, time_idx]] = F::zero();
                    percussive_mask[[freq_idx, time_idx]] = F::one();
                }
            } else {
                // Soft masking: weighted by relative power with softness factor
                let h_ratio = h_val / total;
                let p_ratio = p_val / total;

                // Apply softness: interpolate between binary and proportional masks
                let h_soft = mask_softness * h_ratio
                    + (F::one() - mask_softness)
                        * if h_val >= p_val { F::one() } else { F::zero() };
                let p_soft = mask_softness * p_ratio
                    + (F::one() - mask_softness) * if p_val > h_val { F::one() } else { F::zero() };

                harmonic_mask[[freq_idx, time_idx]] = h_soft;
                percussive_mask[[freq_idx, time_idx]] = p_soft;
            }
        }
    }

    (harmonic_mask, percussive_mask)
}

/// Apply a real-valued mask to a complex STFT spectrogram.
///
/// Multiplies each complex coefficient by the corresponding mask value.
fn apply_mask_to_stft<F: RealFloat>(
    stft: &Array2<Complex<F>>,
    mask: &Array2<F>,
) -> Array2<Complex<F>> {
    let (n_freq_bins, n_time_frames) = stft.dim();
    let mut masked_stft = Array2::zeros((n_freq_bins, n_time_frames));

    for freq_idx in 0..n_freq_bins {
        for time_idx in 0..n_time_frames {
            let original = stft[[freq_idx, time_idx]];
            let mask_val = mask[[freq_idx, time_idx]];
            masked_stft[[freq_idx, time_idx]] = original * mask_val;
        }
    }

    masked_stft
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::generation::sine_wave;

    #[test]
    fn test_median_filter_1d() {
        let signal = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let filtered = median_filter_1d(&signal, 3);

        // Expected: edge handling + median filtering
        assert_eq!(filtered.len(), signal.len());

        // Middle value should be median of [5, 3, 2] = 3
        assert!((filtered[2] - 3.0f64).abs() < 1e-10);
    }

    #[test]
    fn test_hpss_basic_functionality() {
        // Create a simple test signal
        let sample_rate = crate::sample_rate!(8000); // Small for faster test
        let duration = std::time::Duration::from_millis(200); // Long enough for 1024 window

        // Generate a simple sine wave
        let sine_audio = sine_wave::<f32, f32>(440.0, duration, sample_rate.get(), 0.5);
        let config = HpssConfig::<f32>::realtime(); // Faster for testing

        // Perform HPSS separation
        let (harmonic, percussive) = sine_audio.hpss(&config).unwrap();

        // Verify output properties (allow small length differences due to STFT/ISTFT processing)
        let original_length = sine_audio.samples_per_channel();
        assert!(harmonic.samples_per_channel() > 0);
        assert!(percussive.samples_per_channel() > 0);
        // Length should be close to original (within a reasonable range for STFT processing)
        let length_diff = (harmonic.samples_per_channel() as i32 - original_length as i32).abs();
        assert!(
            length_diff < 1000,
            "Length difference too large: {}",
            length_diff
        );
        assert_eq!(harmonic.sample_rate, sine_audio.sample_rate);
        assert_eq!(percussive.sample_rate, sine_audio.sample_rate);
    }

    #[test]
    fn test_hpss_config_validation() {
        let mut config = HpssConfig::<f32>::new();
        let sample_rate = 44100.0;

        // Valid configuration should pass
        assert!(config.validate(sample_rate).is_ok());

        // Invalid window size (not power of 2)
        config.win_size = 1000;
        assert!(config.validate(sample_rate).is_err());

        // Reset and test hop size > win size
        config = HpssConfig::new();
        config.hop_size = config.win_size + 1;
        assert!(config.validate(sample_rate).is_err());

        // Reset and test invalid mask softness
        config = HpssConfig::new();
        config.mask_softness = 1.5;
        assert!(config.validate(sample_rate).is_err());
    }

    #[test]
    fn test_separation_masks() {
        let harmonic_spec = Array2::from_shape_vec(
            (2, 3),
            vec![
                1.0, 2.0, 3.0, // freq bin 0
                0.5, 1.0, 1.5, // freq bin 1
            ],
        )
        .unwrap();

        let percussive_spec = Array2::from_shape_vec(
            (2, 3),
            vec![
                0.5, 0.5, 0.5, // freq bin 0
                2.0, 1.5, 1.0, // freq bin 1
            ],
        )
        .unwrap();

        // Test hard masking (softness = 0)
        let (h_mask, p_mask) = generate_separation_masks(&harmonic_spec, &percussive_spec, 0.0);

        // Verify masks are binary and complementary
        for i in 0..2 {
            for j in 0..3 {
                let h_val = h_mask[[i, j]];
                let p_val = p_mask[[i, j]];
                assert!(h_val == 0.0 || h_val == 1.0);
                assert!(p_val == 0.0 || p_val == 1.0);
                assert!((h_val + p_val - 1.0f32).abs() < 1e-10 || (h_val + p_val).abs() < 1e-10);
            }
        }
    }
}
