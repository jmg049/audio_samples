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

use std::num::{NonZeroU32, NonZeroUsize};

use crate::operations::traits::{AudioDecomposition, AudioTransforms};
use crate::traits::StandardSample;
use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ParameterError,
};

use ndarray::{Array2, s};
use num_complex::Complex;
use spectrograms::{StftParams, StftParamsBuilder, istft as spectrograms_istft};

/// Configuration for Harmonic/Percussive Source Separation (HPSS).
///
/// HPSS separates audio into harmonic and percussive components using
/// STFT magnitude median filtering. Harmonic components are enhanced
/// by median filtering along the time axis, while percussive components
/// are enhanced by median filtering along the frequency axis.
#[derive(Debug, Clone, PartialEq)]
pub struct HpssConfig {
    /// STFT FFT size in samples
    pub stft_params: StftParams,
    /// Harmonic median filter kernel size (along time axis)
    /// Larger values strengthen harmonic separation but may blur transients
    pub median_filter_harmonic: usize,
    /// Percussive median filter kernel size (along frequency axis)
    /// Larger values strengthen percussive separation but may blur tonal content
    pub median_filter_percussive: usize,
    /// Soft masking parameter (0.0 = hard mask, 1.0 = completely soft)
    /// Soft masking provides smoother component separation but less isolation
    pub mask_softness: f64,
}

impl HpssConfig {
    /// Create a new HPSS configuration with default settings.
    #[inline]
    #[must_use]
    pub const fn new(
        stft_params: StftParams,
        median_filter_harmonic: usize,
        median_filter_percussive: usize,
        mask_softness: f64,
    ) -> Self {
        Self {
            stft_params,
            median_filter_harmonic,
            median_filter_percussive,
            mask_softness,
        }
    }

    /// Create configuration optimized for musical content.
    ///
    /// Uses larger filters for stronger separation, suitable for complex musical material:
    /// - Larger harmonic kernel for better tonal separation
    /// - Larger percussive kernel for cleaner transient isolation
    /// - Softer masking for more musical results
    #[inline]
    #[must_use]
    pub fn musical() -> Self {
        let stft_params = StftParamsBuilder::default()
            .n_fft(crate::nzu!(2048))
            .hop_size(crate::nzu!(512))
            .build()
            .expect("All parameters are set and valid according to the builder");
        Self {
            stft_params,
            median_filter_harmonic: 31,
            median_filter_percussive: 31,
            mask_softness: 0.5,
        }
    }

    /// Create configuration optimized for percussive content.
    ///
    /// Uses asymmetric filters favoring percussive separation:
    /// - Moderate harmonic filtering
    /// - Strong percussive filtering
    /// - Harder masking for cleaner drum isolation
    #[inline]
    #[must_use]
    pub fn percussive() -> Self {
        let stft_params = StftParamsBuilder::default()
            .n_fft(crate::nzu!(2048))
            .hop_size(crate::nzu!(256))
            .build()
            .expect("All parameters are set and valid according to the builder");
        Self {
            stft_params,
            median_filter_harmonic: 17,
            median_filter_percussive: 51, // Stronger percussive filtering
            mask_softness: 0.1,           // Harder masking
        }
    }

    /// Create configuration optimized for harmonic content.
    ///
    /// Uses asymmetric filters favoring harmonic separation:
    /// - Strong harmonic filtering
    /// - Moderate percussive filtering
    /// - Harder masking for cleaner tonal isolation
    #[inline]
    #[must_use]
    pub fn harmonic() -> Self {
        let stft_params = StftParamsBuilder::default()
            .n_fft(crate::nzu!(4096))
            .hop_size(crate::nzu!(512))
            .build()
            .expect("All parameters are set and valid according to the builder");

        Self {
            stft_params,
            median_filter_harmonic: 51, // Stronger harmonic filtering
            median_filter_percussive: 17,
            mask_softness: 0.1, // Harder masking
        }
    }

    /// Create configuration for real-time processing.
    ///
    /// Uses smaller window and filter sizes for lower latency:
    /// - Smaller window for reduced latency
    /// - Smaller hop size for responsiveness
    /// - Smaller filters for faster processing
    #[inline]
    #[must_use]
    pub fn realtime() -> Self {
        let stft_params = StftParamsBuilder::default()
            .n_fft(crate::nzu!(1024))
            .hop_size(crate::nzu!(256))
            .build()
            .expect("All parameters are set and valid according to the builder");

        Self {
            stft_params,
            median_filter_harmonic: 11,
            median_filter_percussive: 11,
            mask_softness: 0.3,
        }
    }

    /// Set STFT parameters.
    ///
    /// # Arguments
    /// * `n_fft` - FFT size in samples (should be power of 2)
    /// * `hop_size` - Hop size in samples
    #[inline]
    pub fn set_stft_params(&mut self, n_fft: NonZeroUsize, hop_size: NonZeroUsize) {
        self.stft_params = StftParamsBuilder::default()
            .n_fft(n_fft)
            .hop_size(hop_size)
            .build()
            .expect("All parameters are set and valid according to the builder");
    }

    /// Set median filter sizes.
    ///
    /// # Arguments
    /// * `harmonic` - Harmonic filter size (odd numbers recommended)
    /// * `percussive` - Percussive filter size (odd numbers recommended)
    #[inline]
    pub const fn set_filter_sizes(&mut self, harmonic: usize, percussive: usize) {
        self.median_filter_harmonic = harmonic;
        self.median_filter_percussive = percussive;
    }

    /// Set mask softness parameter.
    ///
    /// # Arguments
    /// * `softness` - Softness value (0.0 = hard mask, 1.0 = completely soft)
    #[inline]
    pub const fn set_mask_softness(&mut self, softness: f64) {
        self.mask_softness = softness.clamp(0.0, 1.0);
    }

    /// Validate HPSS configuration.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Result indicating whether the configuration is valid
    #[inline]
    pub fn validate(&self, sample_rate: f64) -> AudioSampleResult<()> {
        // Validate window size
        if !self.stft_params.n_fft().is_power_of_two() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "win_size",
                "Window size must be a positive power of 2",
            )));
        }

        // Validate hop size
        if self.stft_params.hop_size() > self.stft_params.n_fft() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "hop_size",
                "Hop size must be positive and not larger than window size",
            )));
        }

        // Validate median filter sizes
        if self.median_filter_harmonic == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "median_filter_harmonic",
                "Harmonic median filter size must be greater than 0",
            )));
        }

        if self.median_filter_percussive == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "median_filter_percussive",
                "Percussive median filter size must be greater than 0",
            )));
        }

        // Validate mask softness
        if self.mask_softness < 0.0 || self.mask_softness > 1.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "mask_softness",
                "Mask softness must be between 0.0 and 1.0",
            )));
        }

        // Check reasonable parameter ranges
        if self.stft_params.n_fft() > crate::nzu!(163840) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "win_size",
                "Window size should not exceed 16384 samples for practical processing",
            )));
        }

        if self.median_filter_harmonic > 101 || self.median_filter_percussive > 101 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "median_filter_size",
                "Median filter sizes should not exceed 101 for practical processing",
            )));
        }

        // Validate minimum frequency resolution
        let freq_resolution = self.freq_resolution(sample_rate);
        if freq_resolution > 50.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "win_size",
                format!(
                    "Window too small, frequency resolution ({freq_resolution:.1} Hz) is too low"
                ),
            )));
        }

        Ok(())
    }

    /// Calculate the number of frequency bins for this configuration.
    #[inline]
    #[must_use]
    pub const fn num_freq_bins(&self) -> NonZeroUsize {
        self.stft_params
            .n_fft()
            .div_ceil(crate::nzu!(2))
            .checked_add(1)
            .expect("Div 2 plus 1 will get nowhere near the max value")
    }

    /// Calculate the frequency resolution in Hz.
    #[inline]
    #[must_use]
    pub fn freq_resolution(&self, sample_rate: f64) -> f64 {
        sample_rate / self.stft_params.n_fft().get() as f64
    }

    /// Calculate the time resolution in seconds.
    #[inline]
    #[must_use]
    pub fn time_resolution(&self, sample_rate: f64) -> f64 {
        self.stft_params.hop_size().get() as f64 / sample_rate
    }
}

impl Default for HpssConfig {
    fn default() -> Self {
        let stft_params = StftParamsBuilder::default()
            .n_fft(crate::nzu!(2048))
            .hop_size(crate::nzu!(512))
            .build()
            .expect("All parameters are set and valid according to the builder");
        Self {
            stft_params,
            median_filter_harmonic: 17,
            median_filter_percussive: 17,
            mask_softness: 0.3,
        }
    }
}

impl<T> AudioDecomposition for AudioSamples<'_, T>
where
    T: StandardSample,
    Self: AudioTypeConversion<Sample = T>,
{
    fn hpss(
        &self,
        config: &HpssConfig,
    ) -> AudioSampleResult<(
        AudioSamples<'static, Self::Sample>,
        AudioSamples<'static, Self::Sample>,
    )> {
        // Validate configuration
        config.validate(self.sample_rate_hz())?;

        // Check minimum signal length
        let min_length = config.stft_params.n_fft();
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
fn perform_hpss<T>(
    audio: &AudioSamples<'_, T>,
    config: &HpssConfig,
) -> AudioSampleResult<(AudioSamples<'static, T>, AudioSamples<'static, T>)>
where
    T: StandardSample,
{
    // Step 1: Compute STFT of the input signal
    let stft_result = audio.stft(&config.stft_params)?;
    // Step 2: Compute magnitude spectrogram
    let magnitude_spec = stft_result.norm();

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
    let n_fft = stft_result.params.n_fft();
    let hop_size = stft_result.params.hop_size();
    let window = stft_result.params.window();
    let centre = stft_result.params.centre();
    // safety: sample_rate was validated during the forward STFT
    let sample_rate = unsafe { NonZeroU32::new_unchecked(stft_result.sample_rate as u32) };

    let harmonic_samples =
        spectrograms_istft(&harmonic_stft, n_fft, hop_size, window.clone(), centre)?;
    let harmonic_audio: AudioSamples<'static, T> =
        AudioSamples::from_mono_vec::<f64>(harmonic_samples, sample_rate);

    let percussive_samples = spectrograms_istft(&percussive_stft, n_fft, hop_size, window, centre)?;
    let percussive_audio: AudioSamples<'static, T> =
        AudioSamples::from_mono_vec::<f64>(percussive_samples, sample_rate);

    Ok((harmonic_audio, percussive_audio))
}

///
/// Harmonic components tend to be stable over time, so median filtering along
/// the time axis preserves sustained tonal content while suppressing transients.
fn median_filter_time_axis(spectrogram: &Array2<f64>, kernel_size: usize) -> Array2<f64> {
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
fn median_filter_freq_axis(spectrogram: &Array2<f64>, kernel_size: usize) -> Array2<f64> {
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

// TODO: Change to NonEmptySlices and NonZeroUsize
/// Simple 1D median filter implementation.
///
/// Uses a sliding window approach with border handling via reflection.
fn median_filter_1d(signal: &[f64], kernel_size: usize) -> Vec<f64> {
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
            f64::midpoint(window[kernel_size / 2 - 1], window[kernel_size / 2])
        };

        filtered.push(median);
    }

    filtered
}

/// Generate separation masks from harmonic and percussive spectrograms.
///
/// Creates binary or soft masks based on the relative strengths of harmonic
/// and percussive components at each time-frequency bin.
fn generate_separation_masks(
    harmonic_spec: &Array2<f64>,
    percussive_spec: &Array2<f64>,
    mask_softness: f64,
) -> (Array2<f64>, Array2<f64>) {
    let (n_freq_bins, n_time_frames) = harmonic_spec.dim();
    let mut harmonic_mask = Array2::zeros((n_freq_bins, n_time_frames));
    let mut percussive_mask = Array2::zeros((n_freq_bins, n_time_frames));

    let epsilon = 1e-10; // Small constant to avoid division by zero

    for freq_idx in 0..n_freq_bins {
        for time_idx in 0..n_time_frames {
            let h_val = harmonic_spec[[freq_idx, time_idx]];
            let p_val = percussive_spec[[freq_idx, time_idx]];

            let total = h_val + p_val + epsilon;

            if mask_softness == 0.0 {
                // Hard masking: binary decision
                if h_val >= p_val {
                    harmonic_mask[[freq_idx, time_idx]] = 1.0;
                    percussive_mask[[freq_idx, time_idx]] = 0.0;
                } else {
                    harmonic_mask[[freq_idx, time_idx]] = 0.0;
                    percussive_mask[[freq_idx, time_idx]] = 1.0;
                }
            } else {
                // Soft masking: weighted by relative power with softness factor
                let h_ratio = h_val / total;
                let p_ratio = p_val / total;

                // Apply softness: interpolate between binary and proportional masks
                let h_soft = mask_softness.mul_add(
                    h_ratio,
                    (1.0 - mask_softness) * if h_val >= p_val { 1.0 } else { 0.0 },
                );
                let p_soft = mask_softness.mul_add(
                    p_ratio,
                    (1.0 - mask_softness) * if p_val > h_val { 1.0 } else { 0.0 },
                );

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
fn apply_mask_to_stft<A: AsRef<Array2<Complex<f64>>>>(
    stft: A,
    mask: &Array2<f64>,
) -> Array2<Complex<f64>> {
    let stft = stft.as_ref();
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
        let sine_audio = sine_wave::<f32>(440.0, duration, sample_rate, 0.5);
        let config = HpssConfig::realtime(); // Faster for testing

        // Perform HPSS separation
        let (harmonic, percussive) = sine_audio.hpss(&config).unwrap();

        // Verify output properties (allow small length differences due to STFT/ISTFT processing)
        let original_length = sine_audio.samples_per_channel().get();
        assert!(harmonic.samples_per_channel().get() > 0);
        assert!(percussive.samples_per_channel().get() > 0);
        // Length should be close to original (within a reasonable range for STFT processing)
        let length_diff =
            (harmonic.samples_per_channel().get() as i32 - original_length as i32).abs();
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
        let config = HpssConfig::default();
        let sample_rate = 44100.0;

        // Valid configuration should pass
        assert!(config.validate(sample_rate).is_ok());

        // Invalid window size (not power of 2)
        let config = HpssConfig {
            stft_params: spectrograms::StftParamsBuilder::default()
                .n_fft(crate::nzu!(1000))
                .hop_size(crate::nzu!(256))
                .build()
                .unwrap(),
            ..HpssConfig::default()
        };
        assert!(config.validate(sample_rate).is_err());

        // Reset and test hop size > win size
        let stft_params_result = spectrograms::StftParamsBuilder::default()
            .n_fft(crate::nzu!(1024))
            .hop_size(crate::nzu!(2048))
            .build();

        // Should fail at builder level due to hop_size > n_fft constraint
        assert!(stft_params_result.is_err());

        // Reset and test invalid mask softness
        let config = HpssConfig::default();
        let config = HpssConfig {
            mask_softness: 1.5,
            ..config
        };
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
                assert!((h_val + p_val - 1.0f64).abs() < 1e-10 || (h_val + p_val).abs() < 1e-10);
            }
        }
    }
}
