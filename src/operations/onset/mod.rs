//! Onset detection for identifying transient events in audio signals.

//!

//! This module provides algorithms for detecting onsets—the beginning of musical notes,
//! percussive hits, or other transient events—in audio signals. It offers multiple detection
//! methods based on spectral flux, energy changes, and complex domain analysis.

//! Onset detection is fundamental to music information retrieval tasks such as beat tracking,
//! tempo estimation, rhythm analysis, and automatic transcription. Different detection methods
//! are optimized for different audio content (percussive vs. tonal, speech vs. music).

//! The module provides three main configuration types:
//! - [`OnsetDetectionConfig`] — Energy-based onset detection using CQT magnitude
//! - [`SpectralFluxConfig`] — Spectral flux methods (energy, magnitude, complex, rectified)
//! - [`ComplexOnsetConfig`] — Combined magnitude and phase deviation analysis
//!
//! Each configuration offers presets for common use cases (percussive, musical, speech).
//!
//! ## Submodules
//!
//! - [`complex`](crate::operations::AudioDecomposition::complex) — Complex domain onset detection functions (magnitude + phase)
//! - [`filters`] — Signal processing utilities (median filter, rectification, compression)
//! - [`flux`] — Spectral flux computation kernels
//! - [`kernels`] — Low-level DSP kernels for onset detection
//!
//! # Example
//!
//! ```rust,ignore
//! use audio_samples::{AudioSamples, sample_rate};
//! use audio_samples::operations::traits::AudioOnsetDetection;
//! use audio_samples::operations::onset::OnsetDetectionConfig;
//!
//! let audio = AudioSamples::new_mono(&[0.0f32; 44100], sample_rate!(44100))?;
//! let config = OnsetDetectionConfig::percussive();
//! let onset_times = audio.detect_onsets(&config)?;
//! ```

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

/// Configuration parameters for energy-based onset detection using CQT magnitude spectrograms.
///
/// # Purpose
///
/// Encapsulates all settings required for energy-based onset detection, including CQT parameters,
/// hop size, thresholding strategy, and peak picking configuration. This method detects onsets by
/// analyzing sudden increases in spectral energy across frequency bands.
///
/// # Intended Usage
///
/// Construct via preset methods ([`percussive()`], [`musical()`], [`speech()`]) for common use
/// cases, or use [`new()`] for custom configurations. Pass to [`AudioOnsetDetection::detect_onsets`]
/// to detect onset times.
///
/// # Invariants
///
/// - `hop_size` must be positive (enforced by NonZeroUsize)
/// - `threshold` and `min_onset_interval_secs` should be positive for meaningful detection
/// - `pre_emphasis` is typically in range [0.0, 1.0] but not strictly enforced
/// - `median_filter_length` must be odd for symmetric filtering (not enforced, but recommended)
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct OnsetDetectionConfig {
    /// CQT (Constant-Q Transform) parameters controlling frequency resolution and range.
    pub cqt_params: CqtParams,
    /// Hop size in samples between successive analysis frames. Smaller values provide finer
    /// temporal resolution at the cost of more computation.
    pub hop_size: NonZeroUsize,
    /// Window size in samples for CQT analysis. If `None`, auto-calculated based on the lowest
    /// frequency (4 periods for good resolution).
    pub window_size: Option<NonZeroUsize>,
    /// Global threshold for onset detection function. Higher values reduce false positives but
    /// may miss subtle onsets. Typical range: 0.2-0.8.
    pub threshold: f64,
    /// Minimum time interval in seconds between consecutive onsets. Prevents detecting multiple
    /// onsets within short time spans (e.g., for rapid percussion or polyphonic music).
    pub min_onset_interval_secs: f64,
    /// Pre-emphasis factor for highlighting transient content. Higher values accentuate high
    /// frequencies. Typical range: 0.0-0.5.
    pub pre_emphasis: f64,
    /// If `true`, applies adaptive thresholding using median filtering to account for varying
    /// background energy levels.
    pub adaptive_threshold: bool,
    /// Length of median filter kernel for adaptive thresholding. Odd values recommended for
    /// symmetric filtering. Typical values: 3-11.
    pub median_filter_length: NonZeroUsize,
    /// Multiplier for adaptive threshold. The detection threshold is set to
    /// `median * multiplier`. Typical range: 1.0-2.0.
    pub adaptive_threshold_multiplier: f64,
    /// Peak picking configuration for identifying onset peaks in the detection function.
    pub peak_picking: PeakPickingConfig,
}

impl OnsetDetectionConfig {
    /// Creates a new onset detection configuration with explicit parameters.
    ///
    /// For common use cases, prefer the preset methods ([`percussive()`], [`musical()`],
    /// [`speech()`]) which provide sensible defaults. Use this constructor for fine-grained
    /// control over all parameters.
    ///
    /// # Arguments
    ///
    /// * `cqt_params` — CQT configuration (frequency range, bins, window type)
    /// * `hop_size` — Hop size in samples between analysis frames
    /// * `window_size` — Optional window size (None = auto-calculate from lowest frequency)
    /// * `threshold` — Global detection threshold (typical: 0.2-0.8)
    /// * `min_onset_interval_secs` — Minimum time between onsets in seconds
    /// * `pre_emphasis` — Pre-emphasis factor for transient highlighting (0.0-0.5)
    /// * `adaptive_threshold` — Enable adaptive thresholding via median filtering
    /// * `median_filter_length` — Median filter kernel length (odd values recommended)
    /// * `adaptive_threshold_multiplier` — Multiplier for adaptive threshold (1.0-2.0)
    /// * `peak_picking` — Peak picking configuration
    ///
    /// # Returns
    ///
    /// A new [`OnsetDetectionConfig`] instance.
    #[inline]
    #[must_use]
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

    /// Returns the effective window size for CQT analysis.
    ///
    /// If `window_size` is explicitly set, returns that value. Otherwise, auto-calculates based
    /// on the lowest frequency in the CQT configuration (4 periods for good resolution).
    ///
    /// # Arguments
    ///
    /// * `sample_rate` — Audio sample rate in Hz
    ///
    /// # Returns
    ///
    /// Window size in samples.
    #[inline]
    #[must_use]
    pub fn effective_window_size(&self, sample_rate: f64) -> NonZeroUsize {
        self.window_size.unwrap_or_else(|| {
            // Auto-calculate based on lowest frequency (4 periods for good resolution)
            let min_period = sample_rate / self.cqt_params.f_min;
            // safety: sample_rate is positive and f_min is positive, so min_period is positive
            unsafe { NonZeroUsize::new_unchecked((min_period * 4.0) as usize) }
        })
    }

    /// Converts onset frame index to time in seconds.
    ///
    /// # Arguments
    ///
    /// * `frame_index` — Frame index from onset detection (0-based)
    /// * `sample_rate` — Audio sample rate in Hz
    ///
    /// # Returns
    ///
    /// Time in seconds corresponding to the frame index.
    #[inline]
    #[must_use]
    pub fn frame_to_seconds(&self, frame_index: usize, sample_rate: f64) -> f64 {
        (frame_index as f64 * self.hop_size.get() as f64) / sample_rate
    }

    /// Creates a configuration optimized for percussive onset detection.
    ///
    /// Optimized for detecting drum hits, percussion, and other sharp transient events.
    /// Uses higher threshold, shorter minimum interval, and pre-emphasis to highlight transients.
    ///
    /// # Configuration Details
    ///
    /// - CQT: Percussive preset (higher frequency resolution)
    /// - Hop size: 256 samples
    /// - Threshold: 0.5 (high, for clean detection)
    /// - Min interval: 30ms (short, for rapid percussion)
    /// - Pre-emphasis: 0.3 (moderate, to highlight transients)
    /// - Adaptive threshold: Enabled (multiplier: 2.0)
    ///
    /// # Returns
    ///
    /// A [`OnsetDetectionConfig`] instance optimized for percussive content.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use audio_samples::operations::onset::OnsetDetectionConfig;
    ///
    /// let config = OnsetDetectionConfig::percussive();
    /// ```
    #[inline]
    #[must_use]
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

    /// Creates a configuration optimized for musical onset detection.
    ///
    /// Optimized for detecting note onsets in tonal musical instruments (piano, guitar, strings).
    /// Uses moderate threshold, longer minimum interval for typical musical phrasing, and less
    /// pre-emphasis to preserve tonal content.
    ///
    /// # Configuration Details
    ///
    /// - CQT: Musical preset (balanced frequency coverage)
    /// - Hop size: 512 samples
    /// - Threshold: 0.25 (moderate, for sensitivity)
    /// - Min interval: 100ms (longer, for musical phrasing)
    /// - Pre-emphasis: 0.1 (low, to preserve tonal content)
    /// - Adaptive threshold: Enabled (multiplier: 1.2)
    ///
    /// # Returns
    ///
    /// A [`OnsetDetectionConfig`] instance optimized for musical content.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use audio_samples::operations::onset::OnsetDetectionConfig;
    ///
    /// let config = OnsetDetectionConfig::musical();
    /// ```
    #[inline]
    #[must_use]
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

    /// Creates a configuration optimized for speech onset detection.
    ///
    /// Optimized for detecting word and syllable onsets in speech. Uses lower threshold to
    /// capture subtle speech dynamics, moderate minimum interval for typical speech rate, and
    /// minimal pre-emphasis to preserve speech clarity.
    ///
    /// # Configuration Details
    ///
    /// - CQT: Onset detection preset (balanced frequency coverage)
    /// - Hop size: 256 samples
    /// - Threshold: 0.2 (low, for speech dynamics)
    /// - Min interval: 80ms (moderate, for speech rate)
    /// - Pre-emphasis: 0.05 (minimal, for speech clarity)
    /// - Adaptive threshold: Enabled (multiplier: 1.1)
    ///
    /// # Returns
    ///
    /// A [`OnsetDetectionConfig`] instance optimized for speech content.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use audio_samples::operations::onset::OnsetDetectionConfig;
    ///
    /// let config = OnsetDetectionConfig::speech();
    /// ```
    #[inline]
    #[must_use]
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
    #[inline]
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
/// # Purpose
///
/// Defines different approaches for computing spectral flux—the rate of change in a signal's
/// frequency spectrum over time. Each method offers different trade-offs between computational
/// cost, sensitivity, and robustness to noise.
///
/// # Intended Usage
///
/// Select a method based on the audio content type and detection requirements:
/// - [`Energy`](Self::Energy) for percussive and transient-heavy content
/// - [`Magnitude`](Self::Magnitude) for tonal instruments and subtle onsets
/// - [`Complex`](Self::Complex) for polyphonic music requiring phase information
/// - [`RectifiedComplex`](Self::RectifiedComplex) for balanced sensitivity and robustness
///
/// # Invariants
///
/// - All methods produce non-negative flux values (after rectification)
/// - The first frame always has zero flux (no previous frame to compare)
/// - Computational cost increases: Energy < Magnitude < RectifiedComplex < Complex
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpectralFluxMethod {
    /// Energy-based flux using squared magnitude differences.
    ///
    /// Computes the sum of positive energy differences: Σ max(0, |X[k,n]|² - |X[k,n-1]|²)
    /// across all frequency bins. Emphasizes sharp energy increases, making it ideal for
    /// percussive events like drum hits.
    Energy,
    /// Magnitude-based flux using absolute magnitude differences.
    ///
    /// Computes the sum of positive magnitude differences: Σ max(0, |X[k,n]| - |X[k,n-1]|)
    /// across all frequency bins. More sensitive to subtle spectral changes than energy-based
    /// methods, suitable for tonal instruments with gradual attacks.
    Magnitude,
    /// Complex domain flux incorporating both magnitude and phase changes.
    ///
    /// Computes the Euclidean distance between consecutive complex spectra: Σ |X[k,n] - X[k,n-1]|.
    /// Provides the most complete spectral change information but is computationally expensive.
    /// Best for polyphonic music with complex timbres.
    Complex,
    /// Rectified complex domain flux emphasizing positive magnitude changes.
    ///
    /// Similar to complex flux but focuses on magnitude increases while considering phase.
    /// Balances the robustness of complex methods with the interpretability of magnitude-based
    /// approaches. Good general-purpose choice for musical content.
    RectifiedComplex,
}

impl FromStr for SpectralFluxMethod {
    type Err = AudioSampleError;

    #[inline]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "energy" => Ok(Self::Energy),
            "magnitude" => Ok(Self::Magnitude),
            "complex" => Ok(Self::Complex),
            "rectified_complex" => Ok(Self::RectifiedComplex),
            _ => Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "spectral_flux_method",
                s,
            ))),
        }
    }
}

/// Configuration for spectral flux-based onset detection.
///
/// # Purpose
///
/// Encapsulates settings for onset detection using spectral flux—the rate of spectral change
/// between consecutive frames. This approach is effective for detecting both percussive attacks
/// and tonal note onsets across a wide range of musical content.
///
/// # Intended Usage
///
/// Construct via preset methods ([`percussive()`], [`musical()`], [`complex()`]) for common
/// scenarios, or use [`new()`] for custom configurations. Pass to
/// [`AudioOnsetDetection::detect_onsets_spectral_flux`] to detect onset times.
///
/// # Invariants
///
/// - `hop_size` must be positive (enforced by NonZeroUsize)
/// - `log_compression` must be non-negative for valid compression
/// - `flux_method` determines computational cost and detection characteristics
/// - `rectify` should typically be `true` to focus on spectral increases
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct SpectralFluxConfig {
    /// CQT parameters controlling frequency resolution and range for spectral analysis.
    pub cqt_params: CqtParams,
    /// Hop size in samples between successive analysis frames. Smaller values provide finer
    /// temporal resolution at the cost of more computation.
    pub hop_size: NonZeroUsize,
    /// Window size in samples for CQT analysis. If `None`, auto-calculated based on the lowest
    /// frequency (4 periods for good resolution).
    pub window_size: Option<NonZeroUsize>,
    /// Spectral flux computation method. Different methods offer different trade-offs between
    /// sensitivity, robustness, and computational cost.
    pub flux_method: SpectralFluxMethod,
    /// Peak picking configuration for identifying onset peaks in the flux signal.
    pub peak_picking: PeakPickingConfig,
    /// If `true`, rectifies the flux signal (keeps only positive values), emphasizing spectral
    /// increases over decreases. Recommended for most onset detection tasks.
    pub rectify: bool,
    /// Logarithmic compression factor applied to flux values. Higher values reduce dynamic range
    /// and emphasize subtle onsets. The compression formula is: log(1 + C * flux) where C is
    /// this parameter. Set to 0.0 to disable compression.
    pub log_compression: f64,
}

impl SpectralFluxConfig {
    /// Creates a new spectral flux configuration with explicit parameters.
    ///
    /// For common use cases, prefer the preset methods ([`percussive()`], [`musical()`],
    /// [`complex()`]) which provide sensible defaults. Use this constructor for fine-grained
    /// control over all parameters.
    ///
    /// # Arguments
    ///
    /// * `cqt_params` — CQT configuration (frequency range, bins, window type)
    /// * `hop_size` — Hop size in samples between analysis frames
    /// * `window_size` — Optional window size (None = auto-calculate from lowest frequency)
    /// * `flux_method` — Spectral flux computation method
    /// * `peak_picking` — Peak picking configuration
    /// * `rectify` — Enable rectification (keep only positive flux values)
    /// * `log_compression` — Logarithmic compression factor (0.0 = disabled)
    ///
    /// # Returns
    ///
    /// A new [`SpectralFluxConfig`] instance.
    #[inline]
    #[must_use]
    pub const fn new(
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

    /// Creates a configuration optimized for percussive onset detection.
    ///
    /// Uses energy-based flux with high compression for detecting drum hits and other sharp
    /// transient events. Rectification is enabled to focus on energy increases.
    ///
    /// # Returns
    ///
    /// A [`SpectralFluxConfig`] instance optimized for percussive content.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use audio_samples::operations::onset::SpectralFluxConfig;
    ///
    /// let config = SpectralFluxConfig::percussive();
    /// ```
    #[inline]
    #[must_use]
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

    /// Creates a configuration optimized for musical onset detection.
    ///
    /// Uses magnitude-based flux with moderate compression for detecting note onsets in tonal
    /// instruments. Provides good sensitivity to subtle attacks while maintaining robustness.
    ///
    /// # Returns
    ///
    /// A [`SpectralFluxConfig`] instance optimized for musical content.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use audio_samples::operations::onset::SpectralFluxConfig;
    ///
    /// let config = SpectralFluxConfig::musical();
    /// ```
    #[inline]
    #[must_use]
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

    /// Creates a configuration optimized for complex domain onset detection.
    ///
    /// Uses complex flux (magnitude + phase) without rectification for maximum sensitivity to
    /// spectral changes. Best for polyphonic music with complex timbres.
    ///
    /// # Returns
    ///
    /// A [`SpectralFluxConfig`] instance optimized for complex domain analysis.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use audio_samples::operations::onset::SpectralFluxConfig;
    ///
    /// let config = SpectralFluxConfig::complex();
    /// ```
    #[inline]
    #[must_use]
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

    /// Validates the spectral flux configuration for correctness.
    ///
    /// Checks that all parameters are within valid ranges. Should be called before using the
    /// configuration for onset detection if parameters were manually modified.
    ///
    /// # Returns
    ///
    /// `Ok(())` if all parameters are valid, otherwise returns an error describing the first
    /// validation failure encountered.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Parameter] if `log_compression` is negative
    /// - [crate::AudioSampleError::Parameter] if `peak_picking` configuration is invalid
    #[inline]
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

/// Configuration for complex domain onset detection combining magnitude and phase analysis.
///
/// # Purpose
///
/// Encapsulates settings for onset detection using both magnitude differences and phase
/// deviations from the Complex CQT. This dual approach provides superior accuracy compared to
/// magnitude-only methods, especially for polyphonic music with complex timbres where phase
/// information reveals subtle spectral changes.
///
/// # Intended Usage
///
/// Construct via preset methods ([`percussive()`], [`musical()`], [`speech()`]) for common
/// scenarios, or use [`new()`] for custom configurations. Pass to
/// [`AudioOnsetDetection::complex_onset_detection`] to detect onset times. Adjust
/// `magnitude_weight` and `phase_weight` to balance the contribution of each component.
///
/// # Invariants
///
/// - `hop_size` must be positive (enforced by NonZeroUsize)
/// - `magnitude_weight` and `phase_weight` must be in range [0.0, 1.0]
/// - At least one weight must be greater than 0.0
/// - `log_compression` must be non-negative
/// - Weights do not need to sum to 1.0 (linear combination, not convex)
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ComplexOnsetConfig {
    /// CQT parameters controlling frequency resolution and range for complex spectral analysis.
    pub cqt_config: CqtParams,
    /// Hop size in samples between successive analysis frames. Smaller values provide finer
    /// temporal resolution at the cost of more computation.
    pub hop_size: NonZeroUsize,
    /// Window size in samples for CQT analysis. If `None`, auto-calculated based on the lowest
    /// frequency (4 periods for good resolution).
    pub window_size: Option<NonZeroUsize>,
    /// Peak picking configuration for identifying onset peaks in the combined detection function.
    pub peak_picking: PeakPickingConfig,
    /// Weight for magnitude-based onset detection component. Higher values emphasize energy
    /// changes. Must be in range [0.0, 1.0].
    pub magnitude_weight: f64,
    /// Weight for phase-based onset detection component. Higher values emphasize phase deviations
    /// which can capture subtle timbral changes. Must be in range [0.0, 1.0].
    pub phase_weight: f64,
    /// If `true`, rectifies magnitude differences (keeps only positive changes), emphasizing
    /// spectral increases over decreases.
    pub magnitude_rectify: bool,
    /// If `true`, rectifies phase deviations (keeps only positive deviations), focusing on
    /// increases in phase instability.
    pub phase_rectify: bool,
    /// Logarithmic compression factor applied to the combined onset function. Higher values
    /// reduce dynamic range and emphasize subtle onsets. Set to 0.0 to disable compression.
    pub log_compression: f64,
}

impl ComplexOnsetConfig {
    /// Creates a new complex onset configuration with balanced default settings.
    ///
    /// Uses moderate weights for both magnitude (0.7) and phase (0.3) components with
    /// rectification enabled for both. Suitable for general-purpose musical onset detection.
    ///
    /// # Returns
    ///
    /// A [`ComplexOnsetConfig`] instance with balanced defaults.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use audio_samples::operations::onset::ComplexOnsetConfig;
    ///
    /// let config = ComplexOnsetConfig::new();
    /// ```
    #[inline]
    #[must_use]
    pub const fn new(
        cqt_config: CqtParams,
        hop_size: NonZeroUsize,
        window_size: Option<NonZeroUsize>,
        peak_picking: PeakPickingConfig,
        magnitude_weight: f64,
        phase_weight: f64,
        magnitude_rectify: bool,
        phase_rectify: bool,
        log_compression: f64,
    ) -> Self {
        Self {
            cqt_config,
            hop_size,
            window_size,
            peak_picking,
            magnitude_weight,
            phase_weight,
            magnitude_rectify,
            phase_rectify,
            log_compression,
        }
    }

    /// Creates a configuration optimized for percussive onset detection.
    ///
    /// Emphasizes magnitude changes (0.8) over phase (0.2) for detecting sharp transients like
    /// drum hits. Uses high compression and shorter hop size for temporal precision.
    ///
    /// # Returns
    ///
    /// A [`ComplexOnsetConfig`] instance optimized for percussive content.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use audio_samples::operations::onset::ComplexOnsetConfig;
    ///
    /// let config = ComplexOnsetConfig::percussive();
    /// ```
    #[inline]
    #[must_use]
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

    /// Creates a configuration optimized for musical onset detection.
    ///
    /// Balances magnitude (0.6) and phase (0.4) contributions for detecting note onsets in tonal
    /// instruments. Phase information helps capture subtle attacks and pitch changes.
    ///
    /// # Returns
    ///
    /// A [`ComplexOnsetConfig`] instance optimized for musical content.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use audio_samples::operations::onset::ComplexOnsetConfig;
    ///
    /// let config = ComplexOnsetConfig::musical();
    /// ```
    #[inline]
    #[must_use]
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

    /// Creates a configuration optimized for speech onset detection.
    ///
    /// Uses equal weights for magnitude and phase (0.5 each) to capture both energy changes and
    /// formant transitions. Phase rectification is disabled to preserve bidirectional changes.
    ///
    /// # Returns
    ///
    /// A [`ComplexOnsetConfig`] instance optimized for speech content.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use audio_samples::operations::onset::ComplexOnsetConfig;
    ///
    /// let config = ComplexOnsetConfig::speech();
    /// ```
    #[inline]
    #[must_use]
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

    /// Sets the magnitude and phase weights, clamping to valid range [0.0, 1.0].
    ///
    /// Allows dynamic adjustment of the balance between magnitude-based and phase-based onset
    /// detection. Values are automatically clamped to ensure validity.
    ///
    /// # Arguments
    ///
    /// * `magnitude_weight` — Weight for magnitude component (will be clamped to [0.0, 1.0])
    /// * `phase_weight` — Weight for phase component (will be clamped to [0.0, 1.0])
    #[inline]
    pub const fn set_weights(&mut self, magnitude_weight: f64, phase_weight: f64) {
        self.magnitude_weight = magnitude_weight.clamp(0.0, 1.0);
        self.phase_weight = phase_weight.clamp(0.0, 1.0);
    }

    /// Validates the complex onset configuration for correctness.
    ///
    /// Checks that all parameters are within valid ranges. Should be called before using the
    /// configuration for onset detection if parameters were manually modified.
    ///
    /// # Returns
    ///
    /// `Ok(())` if all parameters are valid, otherwise returns an error describing the first
    /// validation failure encountered.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Parameter] if `magnitude_weight` is not in [0.0, 1.0]
    /// - [crate::AudioSampleError::Parameter] if `phase_weight` is not in [0.0, 1.0]
    /// - [crate::AudioSampleError::Parameter] if both weights are zero
    /// - [crate::AudioSampleError::Parameter] if `log_compression` is negative
    /// - [crate::AudioSampleError::Parameter] if `peak_picking` configuration is invalid
    #[inline]
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

impl Default for ComplexOnsetConfig {
    #[inline]
    fn default() -> Self {
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
}

impl<T> AudioOnsetDetection for AudioSamples<'_, T>
where
    T: StandardSample,
    Self: AudioTypeConversion<Sample = T>,
{
    /// Detects onset times using energy-based CQT analysis.
    ///
    /// Computes an onset detection function based on spectral energy changes in the CQT magnitude
    /// spectrogram, then applies peak picking to identify onset times.
    ///
    /// # Arguments
    ///
    /// * `config` — Energy-based onset detection configuration
    ///
    /// # Returns
    ///
    /// Vector of onset times in seconds, sorted in ascending order.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Parameter] if configuration is invalid
    /// - [crate::AudioSampleError::Transform] if CQT computation fails
    /// - [crate::AudioSampleError::PeakPicking] if peak picking fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioOnsetDetection;
    /// use audio_samples::operations::onset::OnsetDetectionConfig;
    ///
    /// let audio = AudioSamples::new_mono(&[0.0f32; 44100], sample_rate!(44100))?;
    /// let config = OnsetDetectionConfig::percussive();
    /// let onset_times = audio.detect_onsets(&config)?;
    /// ```
    #[inline]
    fn detect_onsets(&self, config: &OnsetDetectionConfig) -> AudioSampleResult<Vec<f64>> {
        let sample_rate = self.sample_rate_hz();
        let (_times, odf) = self.onset_detection_function(config)?;

        let peaks = pick_peaks(&odf, &config.peak_picking)?;

        Ok(peaks
            .into_iter()
            .map(|idx| config.frame_to_seconds(idx, sample_rate))
            .collect())
    }

    /// Computes the energy-based onset detection function.
    ///
    /// Returns both time frames and the corresponding onset detection function values, which
    /// measure spectral energy changes over time. Higher values indicate stronger onsets.
    ///
    /// # Arguments
    ///
    /// * `config` — Energy-based onset detection configuration
    ///
    /// # Returns
    ///
    /// A tuple of `(time_frames, odf_values)` where:
    /// - `time_frames` — Time in seconds for each frame
    /// - `odf_values` — Onset detection function values for each frame
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Parameter] if configuration is invalid
    /// - [crate::AudioSampleError::Transform] if CQT computation fails
    #[inline]
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
        let mag = self.cqt_magnitude_spectrogram(&spectrogram_params, cqt_params)?;

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

    /// Detects onset times using spectral flux analysis.
    ///
    /// Computes spectral flux (rate of spectral change) using the configured method, applies
    /// optional rectification and compression, then uses peak picking to identify onset times.
    ///
    /// # Arguments
    ///
    /// * `config` — Spectral flux configuration
    ///
    /// # Returns
    ///
    /// Vector of onset times in seconds, sorted in ascending order.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Parameter] if configuration is invalid
    /// - [crate::AudioSampleError::Transform] if CQT computation fails
    /// - [crate::AudioSampleError::PeakPicking] if peak picking fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioOnsetDetection;
    /// use audio_samples::operations::onset::SpectralFluxConfig;
    ///
    /// let audio = AudioSamples::new_mono(&[0.0f32; 44100], sample_rate!(44100))?;
    /// let config = SpectralFluxConfig::musical();
    /// let onset_times = audio.detect_onsets_spectral_flux(&config)?;
    /// ```
    #[inline]
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

    /// Computes spectral flux using the specified method.
    ///
    /// Returns both time frames and flux values, which measure the rate of spectral change.
    /// Different methods (Energy, Magnitude, Complex, RectifiedComplex) offer different
    /// sensitivities and computational costs.
    ///
    /// # Arguments
    ///
    /// * `config` — CQT parameters
    /// * `window_size` — Window size in samples for CQT analysis
    /// * `hop_size` — Hop size in samples between frames
    /// * `method` — Spectral flux computation method
    ///
    /// # Returns
    ///
    /// A tuple of `(time_frames, flux_values)` where:
    /// - `time_frames` — Time in seconds for each frame
    /// - `flux_values` — Spectral flux values for each frame
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Transform] if CQT computation fails
    #[inline]
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

    /// Detects onset times using complex domain analysis (magnitude + phase).
    ///
    /// Combines magnitude difference and phase deviation into a weighted onset detection function,
    /// then applies peak picking to identify onset times. More accurate than magnitude-only
    /// methods for polyphonic and complex timbral content.
    ///
    /// # Arguments
    ///
    /// * `onset_config` — Complex onset detection configuration
    ///
    /// # Returns
    ///
    /// Vector of onset times in seconds, sorted in ascending order.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Parameter] if configuration is invalid
    /// - [crate::AudioSampleError::Transform] if CQT computation fails
    /// - [crate::AudioSampleError::PeakPicking] if peak picking fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioOnsetDetection;
    /// use audio_samples::operations::onset::ComplexOnsetConfig;
    ///
    /// let audio = AudioSamples::new_mono(&[0.0f32; 44100], sample_rate!(44100))?;
    /// let config = ComplexOnsetConfig::musical();
    /// let onset_times = audio.complex_onset_detection(&config)?;
    /// ```
    #[inline]
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

    /// Computes the complex domain onset detection function.
    ///
    /// Combines magnitude differences and phase deviations using the configured weights to
    /// produce a single onset detection function. Higher values indicate stronger onsets.
    ///
    /// # Arguments
    ///
    /// * `onset_config` — Complex onset detection configuration
    ///
    /// # Returns
    ///
    /// Onset detection function values for each frame.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Transform] if CQT computation fails
    #[inline]
    fn onset_detection_function_complex(
        &self,
        onset_config: &ComplexOnsetConfig,
    ) -> AudioSampleResult<NonEmptyVec<f64>> {
        let mag_diff = self.magnitude_difference_matrix(onset_config)?;
        let phase_dev = self.phase_deviation_matrix(onset_config)?;

        Ok(combine_complex_odf(&mag_diff, &phase_dev, onset_config))
    }

    /// Computes the magnitude difference matrix from CQT.
    ///
    /// Calculates frame-to-frame magnitude differences in the CQT domain. Each element
    /// represents the change in magnitude for a specific frequency bin between consecutive
    /// frames.
    ///
    /// # Arguments
    ///
    /// * `config` — Complex onset detection configuration
    ///
    /// # Returns
    ///
    /// 2D array of magnitude differences with shape `(frequency_bins, frames)`.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Transform] if CQT computation fails
    #[inline]
    fn magnitude_difference_matrix(
        &self,
        config: &ComplexOnsetConfig,
    ) -> AudioSampleResult<Array2<f64>> {
        let cqt_result = self.constant_q_transform(&config.cqt_config, config.hop_size)?;
        let mag = cqt_result.to_magnitude();
        Ok(magnitude_difference(mag.view()))
    }

    /// Computes the phase deviation matrix from CQT.
    ///
    /// Calculates the deviation between observed phase changes and expected phase progression
    /// for each frequency bin. Large deviations indicate transient events or timbral changes.
    ///
    /// # Arguments
    ///
    /// * `config` — Complex onset detection configuration
    ///
    /// # Returns
    ///
    /// 2D array of phase deviations with shape `(frequency_bins, frames)`.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Transform] if CQT computation fails
    #[inline]
    fn phase_deviation_matrix(
        &self,
        config: &ComplexOnsetConfig,
    ) -> AudioSampleResult<ndarray::Array2<f64>> {
        let sample_rate = self.sample_rate_hz();
        let cqt_result = self.constant_q_transform(&config.cqt_config, config.hop_size)?;
        Ok(phase_deviation(cqt_result.data.view(), config, sample_rate))
    }
}
