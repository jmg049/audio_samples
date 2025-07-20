//! Onset detection algorithms for identifying note onsets and transient events.
//!
//! This module implements comprehensive onset detection algorithms that build upon
//! the Constant-Q Transform (CQT) foundation to provide accurate detection of
//! musical onsets, note attacks, and transient events. The algorithms are designed
//! for rhythm analysis, beat tracking, and music information retrieval applications.
//!
//! ## Mathematical Foundation
//!
//! ### Energy-Based Onset Detection
//!
//! Energy-based onset detection measures changes in spectral energy between
//! consecutive frames:
//!
//! ```text
//! ∆E[n] = Σ(max(0, |X[k,n]|² - |X[k,n-1]|²)) for all frequency bins k
//! ```
//!
//! Where `X[k,n]` is the CQT coefficient at frequency bin `k` and time frame `n`.
//! Only positive energy changes are considered, as they typically indicate onsets.
//!
//! ### Spectral Flux Methods
//!
//! Spectral flux measures the rate of change of the magnitude spectrum:
//!
//! - **Energy flux**: `SF[n] = Σ(max(0, |X[k,n]|² - |X[k,n-1]|²))`
//! - **Magnitude flux**: `SF[n] = Σ(max(0, |X[k,n]| - |X[k,n-1]|))`
//! - **Complex flux**: Uses both magnitude and phase information
//!
//! ### Complex Domain Onset Detection
//!
//! Complex domain methods use both magnitude and phase information:
//!
//! **Phase deviation**: `PD[k,n] = |φ[k,n] - φ[k,n-1] - 2πf[k]H/fs|`
//!
//! **Combined onset detection**: `O[n] = w_m * M[n] + w_p * P[n]`
//!
//! Where:
//! - `φ[k,n]` is the phase of the complex CQT coefficient
//! - `f[k]` is the center frequency of bin `k`
//! - `H` is the hop size in samples
//! - `fs` is the sample rate
//! - `w_m` and `w_p` are magnitude and phase weights
//!
//! ## Implementation Features
//!
//! - **Multiple algorithms**: Energy-based, spectral flux, and complex domain methods
//! - **Adaptive thresholding**: Dynamic threshold adjustment based on local characteristics
//! - **Peak picking**: Temporal constraints to prevent multiple detections
//! - **Signal enhancement**: Pre-emphasis and filtering for improved detection
//! - **Configurable parameters**: Extensive configuration options for different use cases
//!
//! ## References
//!
//! - Bello, J.P., et al. "A tutorial on onset detection in music signals." IEEE TSALP 2005.
//! - Böck, S., et al. "Evaluating the online capabilities of onset detection methods." ISMIR 2012.
//! - Dixon, S. "Onset detection revisited." DAFx 2006.
//! - Duxbury, C., et al. "Complex domain onset detection for musical signals." DAFx 2003.

use super::peak_picking::pick_peaks;
use super::traits::AudioTransforms;
use super::types::{ComplexOnsetConfig, OnsetConfig, SpectralFluxConfig, SpectralFluxMethod};
use crate::{AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, ConvertTo, I24};
use ndarray::Array2;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rustfft::num_complex::Complex;
use std::f64::consts::PI;

impl<T: AudioSample + ToPrimitive + FromPrimitive + Float> AudioSamples<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: ConvertTo<f64>,
{
    /// Detects note onsets using energy-based spectral analysis.
    ///
    /// This method implements energy-based onset detection using the Constant-Q Transform
    /// to analyze changes in spectral energy between consecutive frames. It is particularly
    /// effective for percussive and transient events.
    ///
    /// # Mathematical Theory
    ///
    /// Energy-based onset detection computes the onset detection function (ODF) as:
    /// ```text
    /// ODF[n] = Σ(max(0, |X[k,n]|² - |X[k,n-1]|²)) for all frequency bins k
    /// ```
    ///
    /// The algorithm:
    /// 1. Compute CQT magnitude spectrogram with onset-optimized configuration
    /// 2. Calculate energy differences between consecutive frames
    /// 3. Sum positive energy changes across all frequency bins
    /// 4. Apply peak picking with adaptive thresholding
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration parameters for onset detection
    ///
    /// # Returns
    ///
    /// Vector of onset times in seconds
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use audio_samples::{AudioSamples, operations::*};
    /// use audio_samples::operations::types::OnsetConfig;
    ///
    /// let config = OnsetConfig::percussive();
    /// let onset_times = audio.detect_onsets(&config)?;
    /// println!("Detected {} onsets", onset_times.len());
    /// ```
    pub fn detect_onsets(&self, config: &OnsetConfig) -> AudioSampleResult<Vec<f64>> {
        let sample_rate = self.sample_rate() as f64;
        config.validate(sample_rate).map_err(|e| {
            AudioSampleError::InvalidParameter(format!("Invalid onset config: {}", e))
        })?;

        // Compute onset detection function
        let (_time_frames, odf) = self.onset_detection_function(config)?;

        // Apply peak picking to get onset times
        let peak_indices = pick_peaks(&odf, &config.peak_picking)?;

        // Convert peak indices to onset times
        let onset_times = peak_indices
            .iter()
            .map(|&idx| config.frame_to_seconds(idx, sample_rate))
            .collect();

        Ok(onset_times)
    }

    /// Computes the energy-based onset detection function.
    ///
    /// This method computes the onset detection function (ODF) using energy-based
    /// spectral flux analysis without performing peak picking. The ODF represents
    /// the likelihood of an onset occurring at each time frame.
    ///
    /// # Mathematical Theory
    ///
    /// The energy-based ODF is computed as:
    /// ```text
    /// ODF[n] = Σ(max(0, |X[k,n]|² - |X[k,n-1]|²)) for all frequency bins k
    /// ```
    ///
    /// Optional enhancements include:
    /// - **Pre-emphasis**: Emphasizes high-frequency transients
    /// - **Adaptive thresholding**: Dynamic threshold based on local energy
    /// - **Logarithmic compression**: Reduces dynamic range for better detection
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration parameters for onset detection
    ///
    /// # Returns
    ///
    /// Tuple of (time_frames, onset_detection_function) where:
    /// - `time_frames`: Vector of time values in seconds for each frame
    /// - `onset_detection_function`: Vector of ODF values (non-negative)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use audio_samples::{AudioSamples, operations::*};
    /// use audio_samples::operations::types::OnsetConfig;
    ///
    /// let config = OnsetConfig::musical();
    /// let (times, odf) = audio.onset_detection_function(&config)?;
    ///
    /// // Find the frame with maximum onset strength
    /// let max_idx = odf.iter().position(|&x| x == odf.iter().fold(0.0, |a, &b| a.max(b))).unwrap();
    /// println!("Strongest onset at {:.3}s with strength {:.3}", times[max_idx], odf[max_idx]);
    /// ```
    pub fn onset_detection_function(
        &self,
        config: &OnsetConfig,
    ) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
        let sample_rate = self.sample_rate() as f64;
        config.validate(sample_rate).map_err(|e| {
            AudioSampleError::InvalidParameter(format!("Invalid onset config: {}", e))
        })?;

        // Compute CQT magnitude spectrogram
        let window_size = config.effective_window_size(sample_rate);
        let magnitude_spectrogram = self.cqt_magnitude_spectrogram(
            &config.cqt_config,
            config.hop_size,
            Some(window_size),
            true, // Use power (|X|²) for energy-based detection
        )?;

        let (num_bins, num_frames) = magnitude_spectrogram.dim();
        if num_frames < 2 {
            // For very short signals, return empty onset detection
            return Ok((vec![0.0], vec![0.0]));
        }

        // Compute energy differences between consecutive frames
        let mut odf = Vec::with_capacity(num_frames);

        // First frame has no predecessor, so ODF = 0
        odf.push(0.0);

        for frame_idx in 1..num_frames {
            let mut energy_diff = 0.0;

            for bin_idx in 0..num_bins {
                let current_energy = magnitude_spectrogram[[bin_idx, frame_idx]];
                let prev_energy = magnitude_spectrogram[[bin_idx, frame_idx - 1]];

                // Only consider positive energy changes
                let diff = current_energy - prev_energy;
                if diff > 0.0 {
                    energy_diff += diff;
                }
            }

            // Apply pre-emphasis if configured
            if config.pre_emphasis > 0.0 {
                energy_diff *= 1.0 + config.pre_emphasis;
            }

            odf.push(energy_diff);
        }

        // Apply adaptive thresholding if enabled
        if config.adaptive_threshold {
            let median_filtered = apply_median_filter(&odf, config.median_filter_length)?;
            for (i, &median_val) in median_filtered.iter().enumerate() {
                let threshold = median_val * config.adaptive_threshold_multiplier;
                if odf[i] < threshold {
                    odf[i] = 0.0;
                }
            }
        }

        // Generate time frames
        let time_frames: Vec<f64> = (0..num_frames)
            .map(|i| config.frame_to_seconds(i, sample_rate))
            .collect();

        Ok((time_frames, odf))
    }

    /// Computes the spectral flux using different methods.
    ///
    /// Spectral flux measures the rate of change of the magnitude spectrum
    /// between consecutive frames. Different methods provide different
    /// characteristics for onset detection.
    ///
    /// # Mathematical Theory
    ///
    /// Different spectral flux methods:
    ///
    /// - **Energy**: `SF[n] = Σ(max(0, |X[k,n]|² - |X[k,n-1]|²))`
    /// - **Magnitude**: `SF[n] = Σ(max(0, |X[k,n]| - |X[k,n-1]|))`
    /// - **Complex**: Uses both magnitude and phase information
    ///
    /// # Arguments
    ///
    /// * `config` - CQT configuration for spectral analysis
    /// * `hop_size` - Hop size for frame-based analysis
    /// * `method` - Spectral flux method to use
    ///
    /// # Returns
    ///
    /// Tuple of (time_frames, spectral_flux_values)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = CqtConfig::onset_detection();
    /// let (times, flux) = audio.spectral_flux(&config, 512, SpectralFluxMethod::Energy)?;
    ///
    /// // Analyze spectral flux characteristics
    /// let mean_flux = flux.iter().sum::<f64>() / flux.len() as f64;
    /// println!("Mean spectral flux: {:.3}", mean_flux);
    /// ```
    pub fn spectral_flux(
        &self,
        config: &super::types::CqtConfig,
        hop_size: usize,
        method: SpectralFluxMethod,
    ) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
        let sample_rate = self.sample_rate() as f64;
        config.validate(sample_rate).map_err(|e| {
            AudioSampleError::InvalidParameter(format!("Invalid CQT config: {}", e))
        })?;

        if hop_size == 0 {
            return Err(AudioSampleError::InvalidParameter(
                "Hop size must be greater than 0".to_string(),
            ));
        }

        match method {
            SpectralFluxMethod::Energy => {
                let magnitude_spectrogram = self.cqt_magnitude_spectrogram(
                    config, hop_size, None, true, // Use power for energy-based flux
                )?;
                compute_energy_flux(&magnitude_spectrogram, hop_size, sample_rate)
            }
            SpectralFluxMethod::Magnitude => {
                let magnitude_spectrogram = self.cqt_magnitude_spectrogram(
                    config, hop_size, None, false, // Use magnitude for magnitude-based flux
                )?;
                compute_magnitude_flux(&magnitude_spectrogram, hop_size, sample_rate)
            }
            SpectralFluxMethod::Complex => {
                let complex_spectrogram = self.cqt_spectrogram(config, hop_size, None)?;
                compute_complex_flux(&complex_spectrogram, hop_size, sample_rate)
            }
            SpectralFluxMethod::RectifiedComplex => {
                let complex_spectrogram = self.cqt_spectrogram(config, hop_size, None)?;
                compute_rectified_complex_flux(&complex_spectrogram, hop_size, sample_rate)
            }
        }
    }

    /// Performs complex domain onset detection using both magnitude and phase information.
    ///
    /// This method uses the Constant-Q Transform (CQT) to extract both magnitude and phase
    /// features from the audio signal, then combines them to detect note onsets with higher
    /// accuracy than magnitude-only methods.
    ///
    /// # Mathematical Theory
    ///
    /// Complex domain onset detection combines:
    ///
    /// **Magnitude component**: `M[k,n] = |X[k,n]| - |X[k,n-1]|`
    /// **Phase component**: `P[k,n] = |φ[k,n] - φ[k,n-1] - 2πf[k]H/fs|`
    /// **Combined**: `O[n] = w_m * Σ(max(0, M[k,n])) + w_p * Σ(max(0, P[k,n]))`
    ///
    /// # Arguments
    ///
    /// * `config` - Complex onset detection configuration
    ///
    /// # Returns
    ///
    /// Vector of onset times in seconds
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = ComplexOnsetConfig::musical();
    /// let onset_times = audio.complex_onset_detection(&config)?;
    /// println!("Detected {} onsets", onset_times.len());
    /// ```
    pub fn complex_onset_detection(
        &self,
        config: &ComplexOnsetConfig,
    ) -> AudioSampleResult<Vec<f64>> {
        let sample_rate = self.sample_rate() as f64;
        config.validate(sample_rate).map_err(|e| {
            AudioSampleError::InvalidParameter(format!("Invalid complex onset config: {}", e))
        })?;

        // Compute onset detection function
        let odf = self.onset_detection_function_complex(config)?;

        // Apply peak picking to get onset times
        let peak_indices = pick_peaks(&odf, &config.peak_picking)?;

        // Convert peak indices to onset times
        let onset_times = peak_indices
            .iter()
            .map(|&idx| (idx * config.hop_size) as f64 / sample_rate)
            .collect();

        Ok(onset_times)
    }

    /// Computes the combined onset detection function from magnitude and phase information.
    ///
    /// This function combines magnitude difference and phase deviation matrices
    /// to create a robust onset detection function that leverages both spectral
    /// amplitude and phase coherence information.
    ///
    /// # Arguments
    ///
    /// * `config` - Complex onset detection configuration
    ///
    /// # Returns
    ///
    /// Vector of onset detection function values
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = ComplexOnsetConfig::new();
    /// let odf = audio.onset_detection_function_complex(&config)?;
    /// let max_value = odf.iter().fold(0.0, |a, &b| a.max(b));
    /// ```
    pub fn onset_detection_function_complex(
        &self,
        config: &ComplexOnsetConfig,
    ) -> AudioSampleResult<Vec<f64>> {
        let sample_rate = self.sample_rate() as f64;
        config.validate(sample_rate).map_err(|e| {
            AudioSampleError::InvalidParameter(format!("Invalid complex onset config: {}", e))
        })?;

        // Compute magnitude difference matrix
        let magnitude_diff = self.magnitude_difference_matrix(config)?;

        // Compute phase deviation matrix
        let phase_deviation = self.phase_deviation_matrix(config)?;

        let (num_bins, num_frames) = magnitude_diff.dim();
        let mut odf = Vec::with_capacity(num_frames);

        // Combine magnitude and phase information
        for frame_idx in 0..num_frames {
            let mut magnitude_sum = 0.0;
            let mut phase_sum = 0.0;

            for bin_idx in 0..num_bins {
                let mag_diff = magnitude_diff[[bin_idx, frame_idx]];
                let phase_dev = phase_deviation[[bin_idx, frame_idx]];

                // Apply rectification if configured
                let mag_contribution = if config.magnitude_rectify {
                    mag_diff.max(0.0)
                } else {
                    mag_diff.abs()
                };

                let phase_contribution = if config.phase_rectify {
                    phase_dev.max(0.0)
                } else {
                    phase_dev.abs()
                };

                magnitude_sum += mag_contribution;
                phase_sum += phase_contribution;
            }

            // Weighted combination of magnitude and phase components
            let combined_value =
                config.magnitude_weight * magnitude_sum + config.phase_weight * phase_sum;

            // Apply logarithmic compression if configured
            let compressed_value = if config.log_compression > 0.0 {
                (1.0 + config.log_compression * combined_value).ln()
            } else {
                combined_value
            };

            odf.push(compressed_value);
        }

        Ok(odf)
    }

    /// Computes the phase deviation matrix for onset detection analysis.
    ///
    /// Phase deviation measures the amount by which the phase of each frequency bin
    /// deviates from the expected phase evolution based on the bin's center frequency.
    ///
    /// # Mathematical Theory
    ///
    /// Phase deviation: `PD[k,n] = |φ[k,n] - φ[k,n-1] - 2πf[k]H/fs|`
    ///
    /// Where:
    /// - `φ[k,n]` is the instantaneous phase
    /// - `f[k]` is the center frequency of bin k
    /// - `H` is the hop size
    /// - `fs` is the sample rate
    ///
    /// # Arguments
    ///
    /// * `config` - Complex onset detection configuration
    ///
    /// # Returns
    ///
    /// 2D array with dimensions (num_bins, num_frames) containing phase deviation values
    pub fn phase_deviation_matrix(
        &self,
        config: &ComplexOnsetConfig,
    ) -> AudioSampleResult<Array2<f64>> {
        let sample_rate = self.sample_rate() as f64;
        config.validate(sample_rate).map_err(|e| {
            AudioSampleError::InvalidParameter(format!("Invalid complex onset config: {}", e))
        })?;

        // Compute complex CQT spectrogram
        let window_size = config.window_size.unwrap_or_else(|| {
            let min_period = sample_rate / config.cqt_config.fmin;
            (min_period * 4.0) as usize
        });

        let complex_spectrogram =
            self.cqt_spectrogram(&config.cqt_config, config.hop_size, Some(window_size))?;

        let (num_bins, num_frames) = complex_spectrogram.dim();
        let mut phase_deviation = Array2::zeros((num_bins, num_frames));

        // Calculate expected phase advance for each bin
        let mut expected_phase_advance = Vec::with_capacity(num_bins);
        for bin_idx in 0..num_bins {
            let center_freq = config.cqt_config.bin_frequency(bin_idx);
            let phase_advance = 2.0 * PI * center_freq * config.hop_size as f64 / sample_rate;
            expected_phase_advance.push(phase_advance);
        }

        // Compute phase deviation for each bin and frame
        for bin_idx in 0..num_bins {
            // First frame has no predecessor
            phase_deviation[[bin_idx, 0]] = 0.0;

            for frame_idx in 1..num_frames {
                let current_phase = complex_spectrogram[[bin_idx, frame_idx]].arg();
                let prev_phase = complex_spectrogram[[bin_idx, frame_idx - 1]].arg();

                // Calculate actual phase difference
                let mut phase_diff = current_phase - prev_phase;

                // Unwrap phase difference
                while phase_diff > PI {
                    phase_diff -= 2.0 * PI;
                }
                while phase_diff < -PI {
                    phase_diff += 2.0 * PI;
                }

                // Calculate phase deviation from expected
                let deviation = (phase_diff - expected_phase_advance[bin_idx]).abs();
                phase_deviation[[bin_idx, frame_idx]] = deviation;
            }
        }

        Ok(phase_deviation)
    }

    /// Computes the magnitude difference matrix for onset detection analysis.
    ///
    /// Magnitude difference measures the change in spectral magnitude between
    /// consecutive frames. Positive changes are often associated with note onsets.
    ///
    /// # Mathematical Theory
    ///
    /// Magnitude difference: `MD[k,n] = |X[k,n]| - |X[k,n-1]|`
    ///
    /// # Arguments
    ///
    /// * `config` - Complex onset detection configuration
    ///
    /// # Returns
    ///
    /// 2D array with dimensions (num_bins, num_frames) containing magnitude difference values
    pub fn magnitude_difference_matrix(
        &self,
        config: &ComplexOnsetConfig,
    ) -> AudioSampleResult<Array2<f64>> {
        let sample_rate = self.sample_rate() as f64;
        config.validate(sample_rate).map_err(|e| {
            AudioSampleError::InvalidParameter(format!("Invalid complex onset config: {}", e))
        })?;

        // Compute magnitude spectrogram
        let window_size = config.window_size.unwrap_or_else(|| {
            let min_period = sample_rate / config.cqt_config.fmin;
            (min_period * 4.0) as usize
        });

        let magnitude_spectrogram = self.cqt_magnitude_spectrogram(
            &config.cqt_config,
            config.hop_size,
            Some(window_size),
            false, // Use magnitude, not power
        )?;

        let (num_bins, num_frames) = magnitude_spectrogram.dim();
        let mut magnitude_diff = Array2::zeros((num_bins, num_frames));

        // Compute magnitude differences
        for bin_idx in 0..num_bins {
            // First frame has no predecessor
            magnitude_diff[[bin_idx, 0]] = 0.0;

            for frame_idx in 1..num_frames {
                let current_magnitude = magnitude_spectrogram[[bin_idx, frame_idx]];
                let prev_magnitude = magnitude_spectrogram[[bin_idx, frame_idx - 1]];

                let diff = current_magnitude - prev_magnitude;
                magnitude_diff[[bin_idx, frame_idx]] = diff;
            }
        }

        Ok(magnitude_diff)
    }

    /// Detects onsets using spectral flux analysis with configurable methods.
    ///
    /// This method provides a unified interface for spectral flux-based onset detection
    /// with support for different flux calculation methods and comprehensive configuration
    /// options for peak picking and signal processing.
    ///
    /// # Arguments
    ///
    /// * `config` - Spectral flux configuration parameters
    ///
    /// # Returns
    ///
    /// Vector of onset times in seconds
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = SpectralFluxConfig::percussive();
    /// let onset_times = audio.detect_onsets_spectral_flux(&config)?;
    /// println!("Detected {} onsets", onset_times.len());
    /// ```
    pub fn detect_onsets_spectral_flux(
        &self,
        config: &SpectralFluxConfig,
    ) -> AudioSampleResult<Vec<f64>> {
        let sample_rate = self.sample_rate() as f64;
        config.validate(sample_rate).map_err(|e| {
            AudioSampleError::InvalidParameter(format!("Invalid spectral flux config: {}", e))
        })?;

        // Compute spectral flux
        let (_time_frames, mut flux) =
            self.spectral_flux(&config.cqt_config, config.hop_size, config.flux_method)?;

        // Apply rectification if configured
        if config.rectify {
            flux = flux.into_iter().map(|x| x.max(0.0)).collect();
        }

        // Apply logarithmic compression if configured
        if config.log_compression > 0.0 {
            flux = flux
                .into_iter()
                .map(|x| (1.0 + config.log_compression * x).ln())
                .collect();
        }

        // Apply peak picking to get onset times
        let peak_indices = pick_peaks(&flux, &config.peak_picking)?;

        // Convert peak indices to onset times
        let onset_times = peak_indices
            .iter()
            .map(|&idx| (idx * config.hop_size) as f64 / sample_rate)
            .collect();

        Ok(onset_times)
    }
}

/// Compute energy-based spectral flux from magnitude spectrogram.
fn compute_energy_flux(
    magnitude_spectrogram: &Array2<f64>,
    hop_size: usize,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    let (num_bins, num_frames) = magnitude_spectrogram.dim();
    let mut flux = Vec::with_capacity(num_frames);

    // First frame has no predecessor
    flux.push(0.0);

    for frame_idx in 1..num_frames {
        let mut energy_diff = 0.0;

        for bin_idx in 0..num_bins {
            let current_energy = magnitude_spectrogram[[bin_idx, frame_idx]];
            let prev_energy = magnitude_spectrogram[[bin_idx, frame_idx - 1]];

            let diff = current_energy - prev_energy;
            if diff > 0.0 {
                energy_diff += diff;
            }
        }

        flux.push(energy_diff);
    }

    let time_frames: Vec<f64> = (0..num_frames)
        .map(|i| (i * hop_size) as f64 / sample_rate)
        .collect();

    Ok((time_frames, flux))
}

/// Compute magnitude-based spectral flux from magnitude spectrogram.
fn compute_magnitude_flux(
    magnitude_spectrogram: &Array2<f64>,
    hop_size: usize,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    let (num_bins, num_frames) = magnitude_spectrogram.dim();
    let mut flux = Vec::with_capacity(num_frames);

    // First frame has no predecessor
    flux.push(0.0);

    for frame_idx in 1..num_frames {
        let mut magnitude_diff = 0.0;

        for bin_idx in 0..num_bins {
            let current_magnitude = magnitude_spectrogram[[bin_idx, frame_idx]];
            let prev_magnitude = magnitude_spectrogram[[bin_idx, frame_idx - 1]];

            let diff = current_magnitude - prev_magnitude;
            if diff > 0.0 {
                magnitude_diff += diff;
            }
        }

        flux.push(magnitude_diff);
    }

    let time_frames: Vec<f64> = (0..num_frames)
        .map(|i| (i * hop_size) as f64 / sample_rate)
        .collect();

    Ok((time_frames, flux))
}

/// Compute complex domain spectral flux using both magnitude and phase information.
fn compute_complex_flux(
    complex_spectrogram: &Array2<Complex<f64>>,
    hop_size: usize,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    let (num_bins, num_frames) = complex_spectrogram.dim();
    let mut flux = Vec::with_capacity(num_frames);

    // First frame has no predecessor
    flux.push(0.0);

    for frame_idx in 1..num_frames {
        let mut complex_diff = 0.0;

        for bin_idx in 0..num_bins {
            let current = complex_spectrogram[[bin_idx, frame_idx]];
            let prev = complex_spectrogram[[bin_idx, frame_idx - 1]];

            // Complex domain difference
            let diff = current - prev;
            complex_diff += diff.norm();
        }

        flux.push(complex_diff);
    }

    let time_frames: Vec<f64> = (0..num_frames)
        .map(|i| (i * hop_size) as f64 / sample_rate)
        .collect();

    Ok((time_frames, flux))
}

/// Compute rectified complex domain spectral flux.
fn compute_rectified_complex_flux(
    complex_spectrogram: &Array2<Complex<f64>>,
    hop_size: usize,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    let (num_bins, num_frames) = complex_spectrogram.dim();
    let mut flux = Vec::with_capacity(num_frames);

    // First frame has no predecessor
    flux.push(0.0);

    for frame_idx in 1..num_frames {
        let mut rectified_diff = 0.0;

        for bin_idx in 0..num_bins {
            let current = complex_spectrogram[[bin_idx, frame_idx]];
            let prev = complex_spectrogram[[bin_idx, frame_idx - 1]];

            // Magnitude difference (rectified)
            let magnitude_diff = current.norm() - prev.norm();
            if magnitude_diff > 0.0 {
                rectified_diff += magnitude_diff;
            }
        }

        flux.push(rectified_diff);
    }

    let time_frames: Vec<f64> = (0..num_frames)
        .map(|i| (i * hop_size) as f64 / sample_rate)
        .collect();

    Ok((time_frames, flux))
}

/// Apply median filter to a signal.
fn apply_median_filter(signal: &[f64], filter_length: usize) -> AudioSampleResult<Vec<f64>> {
    if filter_length == 0 || filter_length % 2 == 0 {
        return Err(AudioSampleError::InvalidParameter(
            "Median filter length must be odd and greater than 0".to_string(),
        ));
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
        let start = i.saturating_sub(half_length);
        let end = (i + half_length + 1).min(signal.len());

        let mut window: Vec<f64> = signal[start..end].to_vec();
        window.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = window[window.len() / 2];
        filtered.push(median);
    }

    Ok(filtered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::types::{ComplexOnsetConfig, OnsetConfig, SpectralFluxConfig};
    use ndarray::Array1;
    use std::f64::consts::PI;

    fn generate_test_signal(length: usize, sample_rate: f64) -> Vec<f32> {
        let mut signal = Vec::with_capacity(length);

        // Generate a signal with some onsets
        for i in 0..length {
            let t = i as f64 / sample_rate;

            // Add some onsets at regular intervals
            let onset_strength = if (t * 4.0).fract() < 0.1 { 2.0 } else { 0.5 };

            let value = onset_strength * (2.0 * PI * 440.0 * t).sin();
            signal.push(value as f32);
        }

        signal
    }

    #[test]
    fn test_energy_based_onset_detection() {
        let sample_rate = 44100;
        let signal = generate_test_signal(sample_rate, sample_rate as f64);
        let audio = AudioSamples::new_mono(Array1::from_vec(signal), sample_rate as u32);

        let config = OnsetConfig::percussive();
        let result = audio.detect_onsets(&config);

        assert!(result.is_ok());
        let onset_times = result.unwrap();

        // Should detect some onsets
        assert!(!onset_times.is_empty());

        // All onset times should be within signal duration
        let signal_duration = sample_rate as f64 / sample_rate as f64;
        for &onset_time in &onset_times {
            assert!(onset_time >= 0.0 && onset_time <= signal_duration);
        }
    }

    #[test]
    fn test_onset_detection_function() {
        let sample_rate = 44100;
        let signal = generate_test_signal(sample_rate / 2, sample_rate as f64);
        let audio = AudioSamples::new_mono(Array1::from_vec(signal), sample_rate as u32);

        let config = OnsetConfig::musical();
        let result = audio.onset_detection_function(&config);

        assert!(result.is_ok());
        let (time_frames, odf) = result.unwrap();

        // Should have matching lengths
        assert_eq!(time_frames.len(), odf.len());

        // Should have positive values
        assert!(odf.iter().any(|&x| x > 0.0));

        // Time frames should be monotonically increasing
        for i in 1..time_frames.len() {
            assert!(time_frames[i] > time_frames[i - 1]);
        }
    }

    #[test]
    fn test_spectral_flux_methods() {
        let sample_rate = 44100;
        let signal = generate_test_signal(sample_rate / 4, sample_rate as f64);
        let audio = AudioSamples::new_mono(Array1::from_vec(signal), sample_rate as u32);

        let config = super::super::types::CqtConfig::onset_detection();
        let hop_size = 512;

        let methods = vec![
            SpectralFluxMethod::Energy,
            SpectralFluxMethod::Magnitude,
            SpectralFluxMethod::Complex,
            SpectralFluxMethod::RectifiedComplex,
        ];

        for method in methods {
            let result = audio.spectral_flux(&config, hop_size, method);
            assert!(result.is_ok());

            let (time_frames, flux) = result.unwrap();
            assert_eq!(time_frames.len(), flux.len());
            assert!(!flux.is_empty());
        }
    }

    #[test]
    fn test_complex_onset_detection() {
        let sample_rate = 44100;
        let signal = generate_test_signal(sample_rate / 2, sample_rate as f64);
        let audio = AudioSamples::new_mono(Array1::from_vec(signal), sample_rate as u32);

        let config = ComplexOnsetConfig::musical();
        let result = audio.complex_onset_detection(&config);

        assert!(result.is_ok());
        let onset_times = result.unwrap();

        // Should detect some onsets
        assert!(!onset_times.is_empty());

        // All onset times should be valid
        for &onset_time in &onset_times {
            assert!(onset_time >= 0.0);
        }
    }

    #[test]
    fn test_phase_deviation_matrix() {
        let sample_rate = 44100;
        let signal = generate_test_signal(sample_rate / 4, sample_rate as f64);
        let audio = AudioSamples::new_mono(Array1::from_vec(signal), sample_rate as u32);

        let config = ComplexOnsetConfig::new();
        let result = audio.phase_deviation_matrix(&config);

        assert!(result.is_ok());
        let phase_deviation = result.unwrap();

        let (num_bins, num_frames) = phase_deviation.dim();
        assert!(num_bins > 0);
        assert!(num_frames > 0);

        // First frame should have zero phase deviation
        for bin_idx in 0..num_bins {
            assert_eq!(phase_deviation[[bin_idx, 0]], 0.0);
        }

        // Should have some non-zero values
        assert!(phase_deviation.iter().any(|&x| x > 0.0));
    }

    #[test]
    fn test_magnitude_difference_matrix() {
        let sample_rate = 44100;
        let signal = generate_test_signal(sample_rate / 4, sample_rate as f64);
        let audio = AudioSamples::new_mono(Array1::from_vec(signal), sample_rate as u32);

        let config = ComplexOnsetConfig::new();
        let result = audio.magnitude_difference_matrix(&config);

        assert!(result.is_ok());
        let magnitude_diff = result.unwrap();

        let (num_bins, num_frames) = magnitude_diff.dim();
        assert!(num_bins > 0);
        assert!(num_frames > 0);

        // First frame should have zero magnitude difference
        for bin_idx in 0..num_bins {
            assert_eq!(magnitude_diff[[bin_idx, 0]], 0.0);
        }

        // Should have some non-zero values
        assert!(magnitude_diff.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_spectral_flux_onset_detection() {
        let sample_rate = 44100;
        let signal = generate_test_signal(sample_rate / 2, sample_rate as f64);
        let audio = AudioSamples::new_mono(Array1::from_vec(signal), sample_rate as u32);

        let config = SpectralFluxConfig::percussive();
        let result = audio.detect_onsets_spectral_flux(&config);

        assert!(result.is_ok());
        let onset_times = result.unwrap();

        // Should detect some onsets
        assert!(!onset_times.is_empty());

        // All onset times should be valid
        for &onset_time in &onset_times {
            assert!(onset_time >= 0.0);
        }
    }

    #[test]
    fn test_config_validation() {
        // Test invalid hop size
        let mut config = OnsetConfig::new();
        config.hop_size = 0;
        assert!(config.validate(44100.0).is_err());

        // Test invalid threshold
        config = OnsetConfig::new();
        config.threshold = 1.5;
        assert!(config.validate(44100.0).is_err());

        // Test invalid minimum onset interval
        config = OnsetConfig::new();
        config.min_onset_interval = -0.1;
        assert!(config.validate(44100.0).is_err());

        // Test valid configuration
        config = OnsetConfig::new();
        assert!(config.validate(44100.0).is_ok());
    }

    #[test]
    fn test_preset_configurations() {
        let sample_rate = 44100;
        let signal = generate_test_signal(sample_rate / 4, sample_rate as f64);
        let audio = AudioSamples::new_mono(Array1::from_vec(signal), sample_rate as u32);

        // Test OnsetConfig presets
        let onset_configs = vec![
            OnsetConfig::new(),
            OnsetConfig::percussive(),
            OnsetConfig::musical(),
            OnsetConfig::speech(),
        ];

        for config in onset_configs {
            let result = audio.detect_onsets(&config);
            assert!(result.is_ok());
        }

        // Test ComplexOnsetConfig presets
        let complex_configs = vec![
            ComplexOnsetConfig::new(),
            ComplexOnsetConfig::percussive(),
            ComplexOnsetConfig::musical(),
            ComplexOnsetConfig::speech(),
        ];

        for config in complex_configs {
            let result = audio.complex_onset_detection(&config);
            assert!(result.is_ok());
        }

        // Test SpectralFluxConfig presets
        let flux_configs = vec![
            SpectralFluxConfig::new(),
            SpectralFluxConfig::percussive(),
            SpectralFluxConfig::musical(),
            SpectralFluxConfig::complex(),
        ];

        for config in flux_configs {
            let result = audio.detect_onsets_spectral_flux(&config);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_edge_cases() {
        // Test with very short signal
        let sample_rate = 44100;
        let short_signal = vec![1.0f32, 2.0, 3.0];
        let audio = AudioSamples::new_mono(Array1::from_vec(short_signal), sample_rate);

        let config = OnsetConfig::new();
        let result = audio.detect_onsets(&config);

        // Should handle short signals gracefully
        if result.is_err() {
            eprintln!("Error: {:?}", result.as_ref().err().unwrap());
        }
        assert!(result.is_ok());
        let onset_times = result.unwrap();
        assert!(onset_times.len() <= 1); // May or may not detect onsets in very short signals

        // Test with empty signal
        let empty_signal: Vec<f64> = vec![];
        let audio = AudioSamples::new_mono(Array1::from_vec(empty_signal), sample_rate);

        let result = audio.detect_onsets(&config);
        assert!(result.is_err()); // Should error on empty signal
    }

    #[test]
    fn test_median_filter() {
        let signal = vec![1.0, 5.0, 2.0, 8.0, 3.0, 1.0, 4.0];
        let filtered = apply_median_filter(&signal, 3).unwrap();

        assert_eq!(filtered.len(), signal.len());

        // Median filter should reduce outliers
        assert!(filtered[1] < signal[1]); // 5.0 should be reduced
        assert!(filtered[3] < signal[3]); // 8.0 should be reduced
    }

    #[test]
    fn test_flux_computation_functions() {
        let sample_rate = 44100.0;
        let hop_size = 512;

        // Create a simple test spectrogram
        let spectrogram =
            Array2::from_shape_fn(
                (10, 5),
                |(i, j)| {
                    if j > 0 { (i + j) as f64 * 0.1 } else { 0.0 }
                },
            );

        let (times, flux) = compute_energy_flux(&spectrogram, hop_size, sample_rate).unwrap();

        assert_eq!(times.len(), 5);
        assert_eq!(flux.len(), 5);
        assert_eq!(flux[0], 0.0); // First frame should be zero
        assert!(flux.iter().skip(1).all(|&x| x >= 0.0)); // All other values should be non-negative
    }
}
