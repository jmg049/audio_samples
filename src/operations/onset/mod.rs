//! Onset detection algorithms.
//!
//! This module provides high-level orchestration for onset detection pipelines.
//! All heavy DSP kernels live in submodules and operate on concrete numeric types.

pub mod complex;
pub mod filters;
pub mod flux;
pub mod kernels;

use std::{num::NonZeroUsize, str::FromStr};

use ndarray::Array2;
use non_empty_slice::{NonEmptyVec, non_empty_vec};
use spectrograms::{CqtParams, SpectrogramParams, StftParams};

use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, AudioTransforms, AudioTypeConversion,
    ParameterError, StandardSample,
    operations::{
        onset::{
            complex::{combine_complex_odf, magnitude_difference, phase_deviation},
            filters::{log_compress_inplace, median_filter, rectify_inplace},
            flux::{complex_flux, energy_flux, magnitude_flux, rectified_complex_flux},
            kernels::{apply_adaptive_threshold, energy_odf},
        },
        peak_picking::pick_peaks,
        traits::AudioOnsetDetection,
        types::PeakPickingConfig,
    },
};

#[derive(Debug, Clone, PartialEq)]
pub struct OnsetDetectionConfig {
    pub cqt_params: CqtParams,
    pub hop_size: NonZeroUsize,
    pub window_size: Option<NonZeroUsize>,
    pub threshold: f64,
    pub min_onset_interval_secs: f64,
    pub pre_emphasis: f64,
    pub adaptive_threshold: bool,
    pub median_filter_length: NonZeroUsize,
    pub adaptive_threshold_multiplier: f64,
    pub peak_picking: PeakPickingConfig,
}

impl OnsetDetectionConfig {
    #[inline]
    pub const fn new(
        cqt_params: CqtParams,
        hop_size: NonZeroUsize,
        window_size: Option<NonZeroUsize>,
        threshold: f64,
        min_onset_interval_secs: f64,
        pre_emphasis: f64,
        adaptive_threshold: bool,
        median_filter_length: NonZeroUsize,
        adaptive_threshold_multiplier: f64,
        peak_picking: PeakPickingConfig,
    ) -> Self {
        Self {
            cqt_params,
            hop_size,
            window_size,
            threshold,
            min_onset_interval_secs,
            pre_emphasis,
            adaptive_threshold,
            median_filter_length,
            adaptive_threshold_multiplier,
            peak_picking,
        }
    }

    pub fn effective_window_size(&self, sample_rate: f64) -> NonZeroUsize {
        self.window_size.unwrap_or_else(|| {
            // Auto-calculate based on lowest frequency (4 periods for good resolution)
            let min_period = sample_rate / self.cqt_params.f_min;
            // safety: sample_rate is positive and f_min is positive, so min_period is positive
            unsafe { NonZeroUsize::new_unchecked((min_period * 4.0) as usize) }
        })
    }

    /// Convert onset time from frames to seconds.
    ///
    /// # Arguments
    /// * `frame_index` - Frame index from onset detection
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Time in seconds
    pub fn frame_to_seconds(&self, frame_index: usize, sample_rate: f64) -> f64 {
        (frame_index as f64 * self.hop_size.get() as f64) / sample_rate
    }

    /// Create configuration optimized for percussive onset detection.
    ///
    /// Optimized for detecting drum hits and other percussive events:
    /// - Higher threshold for cleaner detection
    /// - Shorter minimum interval for rapid percussion
    /// - Pre-emphasis to highlight transients
    #[inline]
    pub fn percussive() -> Self {
        let cqt_config = CqtParams::percussive();
        let hop_size = crate::nzu!(256);
        let window = None;
        let threshold = 0.5;
        let min_onset_interval = 0.03; // 30ms
        let pre_emphasis = 0.3;
        let adaptive_threshold = true;
        let median_filter_length = crate::nzu!(3);
        let adaptive_threshold_multiplier = 2.0;
        let peak_picking = PeakPickingConfig::drums();
        Self::new(
            cqt_config,
            hop_size,
            window,
            threshold,
            min_onset_interval,
            pre_emphasis,
            adaptive_threshold,
            median_filter_length,
            adaptive_threshold_multiplier,
            peak_picking,
        )
    }

    /// Create configuration optimized for musical onset detection.
    ///
    /// Optimized for detecting note onsets in musical instruments:
    /// - Moderate threshold for good sensitivity
    /// - Longer minimum interval for typical musical phrasing
    /// - Less pre-emphasis for tonal content
    ///
    /// # Returns
    ///
    /// `OnsetDetectionConfig` instance configured for musical onset detection.
    #[inline]
    pub fn musical() -> Self {
        let cqt_config = CqtParams::musical();
        let hop_size = crate::nzu!(512);
        let window = None;
        let threshold = 0.25;
        let min_onset_interval = 0.1; // 100ms
        let pre_emphasis = 0.1;
        let adaptive_threshold = true;
        let median_filter_length = crate::nzu!(7);
        let adaptive_threshold_multiplier = 1.2;
        let peak_picking = PeakPickingConfig::music();
        Self::new(
            cqt_config,
            hop_size,
            window,
            threshold,
            min_onset_interval,
            pre_emphasis,
            adaptive_threshold,
            median_filter_length,
            adaptive_threshold_multiplier,
            peak_picking,
        )
    }

    /// Create configuration optimized for speech onset detection.
    ///
    /// Optimized for detecting word/syllable onsets in speech:
    /// - Low threshold for speech dynamics
    /// - Moderate minimum interval for speech rate
    /// - Minimal pre-emphasis for speech clarity
    ///
    /// # Returns
    ///
    /// `OnsetDetectionConfig` instance configured for speech onset detection.
    #[inline]
    pub fn speech() -> Self {
        let cqt_config = CqtParams::onset_detection();
        let hop_size = crate::nzu!(256);
        let window = None;
        let threshold = 0.2;
        let min_onset_interval = 0.08; // 80ms
        let pre_emphasis = 0.05;
        let adaptive_threshold = true;
        let median_filter_length = crate::nzu!(9);
        let adaptive_threshold_multiplier = 1.1;
        let peak_picking = PeakPickingConfig::speech();
        Self::new(
            cqt_config,
            hop_size,
            window,
            threshold,
            min_onset_interval,
            pre_emphasis,
            adaptive_threshold,
            median_filter_length,
            adaptive_threshold_multiplier,
            peak_picking,
        )
    }
}

impl Default for OnsetDetectionConfig {
    fn default() -> Self {
        Self::new(
            CqtParams::onset_detection(),
            crate::nzu!(512),
            None,
            0.3,
            0.1,
            0.1,
            true,
            crate::nzu!(5),
            1.5,
            PeakPickingConfig::default(),
        )
    }
}

/// Spectral flux method variants for onset detection.
///
/// Different spectral flux methods provide different characteristics for
/// detecting different types of onsets and musical events.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpectralFluxMethod {
    /// Simple energy-based flux: sum of positive energy differences
    /// ∆E\[n\] = Σ(max(0, |X\[k,n\]|² - |X\[k,n-1\]|²)) for all frequency bins k
    /// Good for general onset detection, especially percussive events
    Energy,
    /// Magnitude-based flux: sum of positive magnitude differences
    /// ∆M\[n\] = Σ(max(0, |X\[k,n\]| - |X\[k,n-1\]|)) for all frequency bins k
    /// More sensitive to subtle onsets, good for tonal instruments
    Magnitude,
    /// Complex domain flux: uses phase information
    /// Takes into account both magnitude and phase changes
    /// More robust to noise but computationally intensive
    Complex,
    /// Rectified complex domain flux: removes negative phase contributions
    /// Balances sensitivity with robustness
    RectifiedComplex,
}

impl FromStr for SpectralFluxMethod {
    type Err = AudioSampleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "energy" => Ok(SpectralFluxMethod::Energy),
            "magnitude" => Ok(SpectralFluxMethod::Magnitude),
            "complex" => Ok(SpectralFluxMethod::Complex),
            "rectified_complex" => Ok(SpectralFluxMethod::RectifiedComplex),
            _ => Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "spectral_flux_method",
                s,
            ))),
        }
    }
}

/// Configuration for spectral flux onset detection.
///
/// Spectral flux measures the rate of change of the magnitude spectrum
/// between consecutive frames, providing effective onset detection for
/// both percussive and tonal instruments.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralFluxConfig {
    /// CQT configuration for spectral analysis
    pub cqt_params: CqtParams,
    /// Hop size for frame-based analysis in samples
    pub hop_size: NonZeroUsize,
    /// Window size for CQT analysis in samples (None = auto-calculate)
    pub window_size: Option<NonZeroUsize>,
    /// Spectral flux method to use
    pub flux_method: SpectralFluxMethod,
    /// Peak picking configuration for onset detection
    pub peak_picking: PeakPickingConfig,
    /// Apply rectification to spectral flux (keep only positive values)
    pub rectify: bool,
    /// Logarithmic compression factor for spectral flux
    /// flux_compressed = log(1 + C * flux) where C is this parameter
    pub log_compression: f64,
}

impl SpectralFluxConfig {
    /// Create a new spectral flux configuration with default settings.
    pub fn new(
        cqt_params: CqtParams,
        hop_size: NonZeroUsize,
        window_size: Option<NonZeroUsize>,
        flux_method: SpectralFluxMethod,
        peak_picking: PeakPickingConfig,
        rectify: bool,
        log_compression: f64,
    ) -> Self {
        Self {
            cqt_params,
            hop_size,
            window_size,
            flux_method,
            peak_picking,
            rectify,
            log_compression,
        }
    }

    /// Create configuration optimized for percussive onset detection.
    pub fn percussive() -> Self {
        Self {
            cqt_params: CqtParams::percussive(),
            hop_size: crate::nzu!(256),
            window_size: None,
            flux_method: SpectralFluxMethod::Energy,
            peak_picking: PeakPickingConfig::drums(),
            rectify: true,
            log_compression: 1000.0,
        }
    }

    /// Create configuration optimized for musical onset detection.
    pub fn musical() -> Self {
        Self {
            cqt_params: CqtParams::onset_detection(),
            hop_size: crate::nzu!(512),
            window_size: None,
            flux_method: SpectralFluxMethod::Magnitude,
            peak_picking: PeakPickingConfig::music(),
            rectify: true,
            log_compression: 100.0,
        }
    }

    /// Create configuration optimized for complex domain onset detection.
    pub fn complex() -> Self {
        Self {
            cqt_params: CqtParams::onset_detection(),
            hop_size: crate::nzu!(512),
            window_size: None,
            flux_method: SpectralFluxMethod::Complex,
            peak_picking: PeakPickingConfig::default(),
            rectify: false,
            log_compression: 100.0,
        }
    }

    /// Validate the spectral flux configuration.
    pub fn validate(&self) -> AudioSampleResult<()> {
        self.peak_picking.validate()?;
        if self.log_compression < 0.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Log compression factor must be non-negative",
            )));
        }

        Ok(())
    }
}

/// Configuration for complex domain onset detection.
///
/// Complex domain onset detection uses both magnitude and phase information
/// from the CQT to provide more accurate onset detection than magnitude-only
/// methods, especially for polyphonic music and complex timbres.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexOnsetConfig {
    /// CQT configuration for spectral analysis
    pub cqt_config: CqtParams,
    /// Hop size for frame-based analysis in samples
    pub hop_size: NonZeroUsize,
    /// Window size for CQT analysis in samples (None = auto-calculate)
    pub window_size: Option<NonZeroUsize>,
    /// Peak picking configuration for onset detection
    pub peak_picking: PeakPickingConfig,
    /// Weight for magnitude-based detection (F::zero()-F::one())
    pub magnitude_weight: f64,
    /// Weight for phase-based detection (F::zero()-F::one())
    pub phase_weight: f64,
    /// Apply magnitude rectification (keep only positive changes)
    pub magnitude_rectify: bool,
    /// Apply phase rectification (keep only positive phase deviations)
    pub phase_rectify: bool,
    /// Logarithmic compression factor for combined onset function
    pub log_compression: f64,
}

impl ComplexOnsetConfig {
    /// Create a new complex onset configuration with default settings.
    pub fn new() -> Self {
        Self {
            cqt_config: CqtParams::onset_detection(),
            hop_size: crate::nzu!(512),
            window_size: None,
            peak_picking: PeakPickingConfig::default(),
            magnitude_weight: 0.7,
            phase_weight: 0.3,
            magnitude_rectify: true,
            phase_rectify: true,
            log_compression: 100.0,
        }
    }

    /// Create configuration optimized for percussive onset detection.
    pub fn percussive() -> Self {
        Self {
            cqt_config: CqtParams::onset_detection(),
            hop_size: crate::nzu!(256),
            window_size: None,
            peak_picking: PeakPickingConfig::drums(),
            magnitude_weight: 0.8,
            phase_weight: 0.2,
            magnitude_rectify: true,
            phase_rectify: true,
            log_compression: 1000.0,
        }
    }

    /// Create configuration optimized for musical onset detection.
    pub fn musical() -> Self {
        Self {
            cqt_config: CqtParams::onset_detection(),
            hop_size: crate::nzu!(512),
            window_size: None,
            peak_picking: PeakPickingConfig::music(),
            magnitude_weight: 0.6,
            phase_weight: 0.4,
            magnitude_rectify: true,
            phase_rectify: true,
            log_compression: 100.0,
        }
    }

    /// Create configuration optimized for speech onset detection.
    pub fn speech() -> Self {
        Self {
            cqt_config: CqtParams::onset_detection(),
            hop_size: crate::nzu!(256),
            window_size: None,
            peak_picking: PeakPickingConfig::speech(),
            magnitude_weight: 0.5,
            phase_weight: 0.5,
            magnitude_rectify: true,
            phase_rectify: false,
            log_compression: 50.0,
        }
    }

    /// Set the magnitude and phase weights.
    pub fn set_weights(&mut self, magnitude_weight: f64, phase_weight: f64) {
        self.magnitude_weight = magnitude_weight.clamp(0.0, 1.0);
        self.phase_weight = phase_weight.clamp(0.0, 1.0);
    }

    /// Validate the complex onset configuration.
    pub fn validate(&self) -> AudioSampleResult<()> {
        self.peak_picking.validate()?;

        if self.magnitude_weight < 0.0 || self.magnitude_weight > 1.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Magnitude weight must be between 0.0 and 1.0",
            )));
        }

        if self.phase_weight < 0.0 || self.phase_weight > 1.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Phase weight must be between 0.0 and 1.0",
            )));
        }

        // Both weights cannot be zero
        if self.magnitude_weight == 0.0 && self.phase_weight == 0.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "At least one of magnitude or phase weight must be greater than 0",
            )));
        }

        if self.log_compression < 0.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Log compression factor must be non-negative",
            )));
        }

        Ok(())
    }
}

impl<T> AudioOnsetDetection for AudioSamples<'_, T>
where
    T: StandardSample,
    Self: AudioTypeConversion<Sample = T>,
{
    fn detect_onsets(&self, config: &OnsetDetectionConfig) -> AudioSampleResult<Vec<f64>> {
        let sample_rate = self.sample_rate_hz();
        let (_times, odf) = self.onset_detection_function(config)?;

        let peaks = pick_peaks(&odf, &config.peak_picking)?;

        Ok(peaks
            .into_iter()
            .map(|idx| config.frame_to_seconds(idx, sample_rate))
            .collect())
    }

    fn onset_detection_function(
        &self,
        config: &OnsetDetectionConfig,
    ) -> AudioSampleResult<(NonEmptyVec<f64>, NonEmptyVec<f64>)> {
        let sample_rate = self.sample_rate_hz();

        let window_size = config.effective_window_size(sample_rate);
        let cqt_params = &config.cqt_params;
        let stft_params = StftParams::builder()
            .n_fft(window_size)
            .hop_size(config.hop_size)
            .window(cqt_params.window.clone())
            .centre(true)
            .build()?;
        let spectrogram_params = SpectrogramParams::new(stft_params, sample_rate)?;
        let mag = self.cqt_magnitude_spectrogram(&spectrogram_params, &cqt_params)?;

        if mag.dim().1 < 2 {
            return Ok((non_empty_vec![0.0], non_empty_vec![0.0]));
        }

        let mut odf = energy_odf(&mag);

        if config.adaptive_threshold {
            let median = median_filter(&odf, config.median_filter_length)?;
            apply_adaptive_threshold(&mut odf, &median, config.adaptive_threshold_multiplier);
        }

        // let time_frames = generate_time_axis(mag.dim().1, config.hop_size, sample_rate);
        let time_frames = mag.times().to_non_empty_vec();
        Ok((time_frames, odf))
    }

    fn detect_onsets_spectral_flux(
        &self,
        config: &SpectralFluxConfig,
    ) -> AudioSampleResult<Vec<f64>> {
        let sample_rate = self.sample_rate_hz();
        let window_size = config.window_size.unwrap_or_else(|| {
            let min_period = sample_rate / config.cqt_params.f_min;
            // safety: sample_rate is positive and f_min is positive, so min_period is positive
            unsafe { NonZeroUsize::new_unchecked((min_period * 4.0) as usize) }
        });
        let (_times, mut flux) = self.spectral_flux(
            &config.cqt_params,
            window_size,
            config.hop_size,
            config.flux_method,
        )?;

        if config.rectify {
            rectify_inplace(&mut flux);
        }

        if config.log_compression > 0.0 {
            log_compress_inplace(&mut flux, config.log_compression);
        }

        let peaks = pick_peaks(&flux, &config.peak_picking)?;

        Ok(peaks
            .into_iter()
            .map(|idx| (idx as f64 * config.hop_size.get() as f64) / sample_rate)
            .collect())
    }

    fn spectral_flux(
        &self,
        config: &CqtParams,
        window_size: NonZeroUsize,
        hop_size: NonZeroUsize,
        method: SpectralFluxMethod,
    ) -> AudioSampleResult<(NonEmptyVec<f64>, NonEmptyVec<f64>)> {
        let sample_rate = self.sample_rate_hz();

        let stft_params = StftParams::builder()
            .n_fft(window_size)
            .hop_size(hop_size)
            .window(config.window.clone())
            .centre(true)
            .build()?;
        let spectrogram_params = SpectrogramParams::new(stft_params, sample_rate)?;

        let (times, flux) = match method {
            SpectralFluxMethod::Energy => {
                let mag = self.cqt_magnitude_spectrogram(&spectrogram_params, config)?;
                (mag.times().to_non_empty_vec(), energy_flux(&mag))
            }
            SpectralFluxMethod::Magnitude => {
                let mag = self.cqt_magnitude_spectrogram(&spectrogram_params, config)?;
                (mag.times().to_non_empty_vec(), magnitude_flux(&mag))
            }
            SpectralFluxMethod::Complex => {
                let cqt_result = self.constant_q_transform(config, hop_size)?;
                let n_frames = cqt_result.n_frames().get();
                // safety: cqt() guarantees n_frames >= 1
                let times = unsafe {
                    NonEmptyVec::new_unchecked(
                        (0..n_frames)
                            .map(|i| i as f64 * hop_size.get() as f64 / sample_rate)
                            .collect(),
                    )
                };
                (times, complex_flux(&cqt_result.data))
            }
            SpectralFluxMethod::RectifiedComplex => {
                let cqt_result = self.constant_q_transform(config, hop_size)?;
                let n_frames = cqt_result.n_frames().get();
                // safety: cqt() guarantees n_frames >= 1
                let times = unsafe {
                    NonEmptyVec::new_unchecked(
                        (0..n_frames)
                            .map(|i| i as f64 * hop_size.get() as f64 / sample_rate)
                            .collect(),
                    )
                };
                (times, rectified_complex_flux(&cqt_result.data))
            }
        };
        Ok((times, flux))
    }

    /// Complex domain onset detection.
    ///
    /// # Arguments
    ///
    /// * `onset_config` - Configuration for complex onset detection
    ///
    /// # Returns
    ///
    /// List of onset times in seconds
    fn complex_onset_detection(
        &self,
        onset_config: &ComplexOnsetConfig,
    ) -> AudioSampleResult<Vec<f64>> {
        let sample_rate = self.sample_rate_hz();
        let odf = self.onset_detection_function_complex(onset_config)?;
        let peaks = pick_peaks(&odf, &onset_config.peak_picking)?;

        Ok(peaks
            .into_iter()
            .map(|idx| (idx as f64 * onset_config.hop_size.get() as f64) / sample_rate)
            .collect())
    }

    fn onset_detection_function_complex(
        &self,
        onset_config: &ComplexOnsetConfig,
    ) -> AudioSampleResult<NonEmptyVec<f64>> {
        let mag_diff = self.magnitude_difference_matrix(onset_config)?;
        let phase_dev = self.phase_deviation_matrix(onset_config)?;

        Ok(combine_complex_odf(&mag_diff, &phase_dev, onset_config))
    }

    fn magnitude_difference_matrix(
        &self,
        config: &ComplexOnsetConfig,
    ) -> AudioSampleResult<Array2<f64>> {
        let cqt_result = self.constant_q_transform(&config.cqt_config, config.hop_size)?;
        let mag = cqt_result.to_magnitude();
        Ok(magnitude_difference(mag.view()))
    }

    fn phase_deviation_matrix(
        &self,
        config: &ComplexOnsetConfig,
    ) -> AudioSampleResult<ndarray::Array2<f64>> {
        let sample_rate = self.sample_rate_hz();
        let cqt_result = self.constant_q_transform(&config.cqt_config, config.hop_size)?;
        Ok(phase_deviation(cqt_result.data.view(), config, sample_rate))
    }
}
