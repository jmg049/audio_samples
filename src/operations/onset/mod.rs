//! Onset detection for identifying transient events in audio signals.
//!
//! # What
//!
//! Algorithms that locate *onsets* - the beginnings of notes, percussive hits or
//! other transients - and return their times in seconds. Each detector has two
//! stages: an **onset detection function** (ODF), one value per analysis frame
//! that rises sharply at an onset, followed by **peak picking**
//! ([`pick_peaks`]).
//!
//! # Why
//!
//! Onset detection underlies beat tracking, tempo estimation, rhythm analysis
//! and transcription. An onset changes the short-time spectrum abruptly (new
//! energy appears across many bins), so frame-to-frame spectral differences
//! make a good ODF.
//!
//! # How
//!
//! All detectors use a constant-Q transform (CQT) with hop H samples
//! (frame t is centred at `t·H/f_s` seconds). Write X\[b,t\] for the CQT value
//! of bin b in frame t. The ODFs below are computed exactly as written
//! (all sums over bins b, `ODF[0] = 0`):
//!
//! - [`AudioOnsetDetection::onset_detection_function`] ("energy" ODF) and
//!   [`SpectralFluxMethod::Energy`]: `Σ max(0, |X[b,t]| − |X[b,t−1]|)`
//!   (no squaring);
//! - [`SpectralFluxMethod::Magnitude`]: `Σ abs(|X[b,t]| − |X[b,t−1]|)`
//!   (not rectified: decreases count too);
//! - [`SpectralFluxMethod::Complex`]: `Σ |X[b,t] − X[b,t−1]|` (complex
//!   difference, so phase changes count);
//! - [`SpectralFluxMethod::RectifiedComplex`]: `Σ max(0, |X[b,t]| − |X[b,t−1]|)`
//!   (phase discarded; equal to the energy flux);
//! - [`AudioOnsetDetection::onset_detection_function_complex`]:
//!   `w_m·Σ R(|X[b,t]| − |X[b,t−1]|) + w_p·Σ |princarg(Δφ[b,t] − 2π·f_b·H/f_s)|`,
//!   optionally log-compressed.
//!
//! The half-wave-rectified magnitude difference is the standard *spectral flux*:
//! only increases in level count, so note offsets do not create onsets. Note
//! that the "Energy" variants do **not** square the magnitudes; pass a power
//! spectrogram to [`flux::energy_flux`] if a true energy flux is wanted. In the
//! complex method, R is rectification or absolute value and the phase term
//! measures how far each bin's phase advance departs from that of its centre
//! frequency f_b (see [`complex::phase_deviation`]); it is not the
//! second-difference phase deviation of Bello et al. (2005).
//!
//! Post-processing: the energy ODF optionally zeroes values below
//! `adaptive_threshold_multiplier × running median` (length
//! `median_filter_length` frames); the spectral-flux ODF is optionally
//! rectified and log-compressed (`ln(1 + C·x)`). Peak picking then applies
//! [`PeakPickingConfig`]: pre-emphasis, an optional running median,
//! normalisation, an adaptive threshold, local-maximum detection and a
//! minimum separation, **all measured in frames**. Onset frame indices are
//! converted to seconds as `t·H/f_s`.
//!
//! # Limitations
//!
//! - The default [`PeakPickingConfig`] sizes (threshold window 1024 frames,
//!   separation 512 frames, length-3 median) were chosen as if they were audio
//!   samples. On an ODF they span seconds, and the median removes one-frame
//!   peaks, so `detect_onsets(&OnsetDetectionConfig::default())` can return no
//!   onsets for a clean click track. Shorten the window and separation and
//!   disable the median for impulsive material, as in the example below.
//! - The CQT response of the external `spectrograms` crate to identical clicks
//!   varies strongly from click to click (up to ~100×, alternating), so
//!   weaker clicks can fall below any threshold and tempo estimates built on
//!   the ODF can show octave errors.
//! - Reported times are frame centres and may precede the true onset by a
//!   fraction of the CQT window (about 30 ms in the example below).
//!
//! The module provides three main configuration types:
//! - [`OnsetDetectionConfig`] - Energy-based onset detection using CQT magnitude
//! - [`SpectralFluxConfig`] - Spectral flux methods (energy, magnitude, complex, rectified)
//! - [`ComplexOnsetConfig`] - Combined magnitude and phase deviation analysis
//!
//! Each configuration offers presets for common use cases (percussive, musical, speech).
//!
//! ## Submodules
//!
//! - [`complex`] - Complex domain onset detection functions (magnitude + phase)
//! - [`filters`] - Signal processing utilities (median filter, rectification, compression)
//! - [`flux`] - Spectral flux computation kernels
//! - [`kernels`] - Low-level DSP kernels for onset detection
//!
//! # Example
//!
//! ```
//! use audio_samples::{AudioSamples, sample_rate};
//! use audio_samples::operations::traits::AudioOnsetDetection;
//! use audio_samples::operations::onset::OnsetDetectionConfig;
//! use ndarray::Array1;
//! use std::num::NonZeroUsize;
//!
//! // Four short noise bursts ("clicks") at 0.25, 0.75, 1.25 and 1.75 s.
//! let sr = 44_100usize;
//! let mut x = vec![0.0f64; 2 * sr];
//! let mut seed = 0x2545_F491_4F6C_DD1Du64;
//! for k in 0..4 {
//!     let start = (0.25 + 0.5 * k as f64) * sr as f64;
//!     for i in 0..256 {
//!         seed ^= seed << 13;
//!         seed ^= seed >> 7;
//!         seed ^= seed << 17;
//!         let r = (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
//!         x[start as usize + i] = 0.9 * (-(i as f64) / 24.0).exp() * r;
//!     }
//! }
//! let audio = AudioSamples::new_mono(Array1::from_vec(x), sample_rate!(44100)).unwrap();
//!
//! // Peak-picking lengths are in frames (hop 512 → 11.6 ms per frame).
//! let mut config = OnsetDetectionConfig::default();
//! config.peak_picking.median_filter = false; // keep one-frame peaks
//! config.peak_picking.pre_emphasis = false;
//! config.peak_picking.adaptive_threshold.window_size = 11; // ≈ 0.13 s
//! config.peak_picking.min_peak_separation = NonZeroUsize::new(5).unwrap(); // ≈ 58 ms
//!
//! let onsets = audio.detect_onsets(&config).unwrap();
//! assert_eq!(onsets.len(), 4);
//! for (k, t) in onsets.iter().enumerate() {
//!     let truth = 0.25 + 0.5 * k as f64;
//!     assert!((t - truth).abs() < 0.05, "onset {t:.3} s vs {truth} s");
//! }
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
    EnumParseError, ParameterError, StandardSample,
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
/// Construct via preset methods ([`percussive()`](Self::percussive), [`musical()`](Self::musical),
/// [`speech()`](Self::speech)) for common use
/// cases, or use [`new()`](Self::new) for custom configurations. Pass to [`AudioOnsetDetection::detect_onsets`]
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
    /// Global threshold for the onset detection function. **Currently unused**: thresholding
    /// is done by `adaptive_threshold*` and [`PeakPickingConfig::adaptive_threshold`].
    pub threshold: f64,
    /// Minimum time interval in seconds between consecutive onsets. **Currently unused**: the
    /// minimum spacing is [`PeakPickingConfig::min_peak_separation`] (in frames).
    pub min_onset_interval_secs: f64,
    /// Pre-emphasis factor. **Currently unused**: pre-emphasis of the onset function is set by
    /// [`PeakPickingConfig::pre_emphasis`] / `pre_emphasis_coeff`.
    pub pre_emphasis: f64,
    /// If `true`, ODF values below `adaptive_threshold_multiplier × running median` are set to
    /// zero before peak picking (values above it are left unchanged).
    pub adaptive_threshold: bool,
    /// Length, in frames, of the running median used by `adaptive_threshold`; must be odd
    /// (an even length makes [`AudioOnsetDetection::onset_detection_function`] return an
    /// error). Typical values: 3-11.
    pub median_filter_length: NonZeroUsize,
    /// Multiplier m for the adaptive threshold `m × median[t]`. Typical range: 1.0-2.0.
    pub adaptive_threshold_multiplier: f64,
    /// Peak picking configuration for identifying onset peaks in the detection function.
    /// Its lengths are in frames (see [`PeakPickingConfig`]).
    pub peak_picking: PeakPickingConfig,
}

impl OnsetDetectionConfig {
    /// Creates a new onset detection configuration with explicit parameters.
    ///
    /// For common use cases, prefer the preset methods ([`percussive()`](Self::percussive),
    /// [`musical()`](Self::musical),
    /// [`speech()`](Self::speech)) which provide sensible defaults. Use this constructor for fine-grained
    /// control over all parameters.
    ///
    /// # Arguments
    ///
    /// * `cqt_params` - CQT configuration (frequency range, bins, window type)
    /// * `hop_size` - Hop size in samples between analysis frames
    /// * `window_size` - Optional window size (None = auto-calculate from lowest frequency)
    /// * `threshold` - Global detection threshold (typical: 0.2-0.8)
    /// * `min_onset_interval_secs` - Minimum time between onsets in seconds
    /// * `pre_emphasis` - Pre-emphasis factor for transient highlighting (0.0-0.5)
    /// * `adaptive_threshold` - Enable adaptive thresholding via median filtering
    /// * `median_filter_length` - Median filter kernel length (odd values recommended)
    /// * `adaptive_threshold_multiplier` - Multiplier for adaptive threshold (1.0-2.0)
    /// * `peak_picking` - Peak picking configuration
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
    /// * `sample_rate` - Audio sample rate in Hz
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
    /// * `frame_index` - Frame index from onset detection (0-based)
    /// * `sample_rate` - Audio sample rate in Hz
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
/// Defines different approaches for computing spectral flux-the rate of change in a signal's
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
/// - All methods produce non-negative flux values
/// - The first frame always has zero flux (no previous frame to compare)
/// - `Energy` and `RectifiedComplex` give identical values on the same CQT;
///   none of the variants squares the magnitudes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpectralFluxMethod {
    /// Half-wave-rectified magnitude flux (named "energy", but not squared).
    ///
    /// Computes `Σ_k max(0, |X[k,n]| − |X[k,n−1]|)` on the CQT magnitude across all
    /// frequency bins: only level increases count. This is the standard spectral flux.
    Energy,
    /// Unrectified absolute magnitude difference.
    ///
    /// Computes `Σ_k abs(|X[k,n]| − |X[k,n−1]|)` across all frequency bins. Level
    /// decreases (note offsets) count as much as increases, and the `rectify` option has
    /// no effect because the values are already non-negative.
    Magnitude,
    /// Complex-domain difference incorporating both magnitude and phase changes.
    ///
    /// Computes `Σ_k |X[k,n] − X[k,n−1]|` on the complex CQT (sum of per-bin moduli).
    /// Phase advance alone makes this non-zero, even for a steady tone.
    Complex,
    /// Half-wave-rectified magnitude flux computed from the complex CQT.
    ///
    /// Computes `Σ_k max(0, |X[k,n]| − |X[k,n−1]|)`. Phase is discarded, so the result is
    /// identical to [`Energy`](Self::Energy).
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
            _ => Err(EnumParseError::new(
                "SpectralFluxMethod",
                s,
                &["energy", "magnitude", "complex", "rectified_complex"],
            )
            .into()),
        }
    }
}

/// Configuration for spectral flux-based onset detection.
///
/// # Purpose
///
/// Encapsulates settings for onset detection using spectral flux-the rate of spectral change
/// between consecutive frames. This approach is effective for detecting both percussive attacks
/// and tonal note onsets across a wide range of musical content.
///
/// # Intended Usage
///
/// Construct via preset methods ([`percussive()`](Self::percussive), [`musical()`](Self::musical),
/// [`complex()`](Self::complex)) for common
/// scenarios, or use [`new()`](Self::new) for custom configurations. Pass to
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
    /// For common use cases, prefer the preset methods ([`percussive()`](Self::percussive),
    /// [`musical()`](Self::musical),
    /// [`complex()`](Self::complex)) which provide sensible defaults. Use this constructor for fine-grained
    /// control over all parameters.
    ///
    /// # Arguments
    ///
    /// * `cqt_params` - CQT configuration (frequency range, bins, window type)
    /// * `hop_size` - Hop size in samples between analysis frames
    /// * `window_size` - Optional window size (None = auto-calculate from lowest frequency)
    /// * `flux_method` - Spectral flux computation method
    /// * `peak_picking` - Peak picking configuration
    /// * `rectify` - Enable rectification (keep only positive flux values)
    /// * `log_compression` - Logarithmic compression factor (0.0 = disabled)
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
/// Construct via preset methods ([`percussive()`](Self::percussive), [`musical()`](Self::musical),
/// [`speech()`](Self::speech)) for common
/// scenarios, or use [`new()`](Self::new) for custom configurations. Pass to
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
    /// If `true`, phase deviations are rectified (`max(0, ·)`), otherwise their absolute value
    /// is used. The phase deviations are already absolute values, so this flag currently has
    /// no effect.
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
    /// formant transitions. `phase_rectify` is `false` (which has no effect, see the field docs).
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
    /// * `magnitude_weight` - Weight for magnitude component (will be clamped to [0.0, 1.0])
    /// * `phase_weight` - Weight for phase component (will be clamped to [0.0, 1.0])
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
    /// Detects onset times using the CQT "energy" onset detection function.
    ///
    /// Computes [`onset_detection_function`](AudioOnsetDetection::onset_detection_function)
    /// (half-wave-rectified CQT magnitude flux, optionally median-thresholded), then applies
    /// [`pick_peaks`] with `config.peak_picking` and converts frame indices to seconds as
    /// `t·hop_size/f_s`. With the default peak-picking lengths (see the
    /// [module docs](self#limitations)) few or no onsets may be found; shorten
    /// `config.peak_picking` lengths for real use.
    ///
    /// # Arguments
    ///
    /// * `config` - Energy-based onset detection configuration
    ///
    /// # Returns
    ///
    /// Vector of onset times in seconds, sorted in ascending order.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Parameter] if configuration is invalid
    /// - [crate::AudioSampleError::Processing] if CQT computation fails
    /// - [crate::AudioSampleError::Processing] if peak picking fails
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

    /// Computes the "energy" onset detection function from the CQT magnitude.
    ///
    /// `odf[t] = Σ_b max(0, |X[b,t]| − |X[b,t−1]|)` (magnitudes are not squared), `odf[0] = 0`.
    /// If `config.adaptive_threshold` is set, values below
    /// `config.adaptive_threshold_multiplier × median[t]` (running median of
    /// `config.median_filter_length` frames) are set to zero. Higher values indicate stronger
    /// onsets.
    ///
    /// # Arguments
    ///
    /// * `config` - Energy-based onset detection configuration
    ///
    /// # Returns
    ///
    /// A tuple of `(time_frames, odf_values)` where:
    /// - `time_frames` - Time in seconds for each frame
    /// - `odf_values` - Onset detection function values for each frame
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Parameter] if configuration is invalid
    /// - [crate::AudioSampleError::Processing] if CQT computation fails
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
    /// * `config` - Spectral flux configuration
    ///
    /// # Returns
    ///
    /// Vector of onset times in seconds, sorted in ascending order.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Parameter] if configuration is invalid
    /// - [crate::AudioSampleError::Processing] if CQT computation fails
    /// - [crate::AudioSampleError::Processing] if peak picking fails
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
    /// * `config` - CQT parameters
    /// * `window_size` - Window size in samples for CQT analysis
    /// * `hop_size` - Hop size in samples between frames
    /// * `method` - Spectral flux computation method
    ///
    /// # Returns
    ///
    /// A tuple of `(time_frames, flux_values)` where:
    /// - `time_frames` - Time in seconds for each frame
    /// - `flux_values` - Spectral flux values for each frame
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Processing] if CQT computation fails
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
    /// Combines the CQT magnitude difference and the phase deviation from each bin's centre
    /// frequency into a weighted onset detection function
    /// (`w_m·Σ R(Δ|X|) + w_p·Σ |phase deviation|`, optionally log-compressed), then applies
    /// peak picking (lengths in frames) to identify onset times.
    ///
    /// # Arguments
    ///
    /// * `onset_config` - Complex onset detection configuration
    ///
    /// # Returns
    ///
    /// Vector of onset times in seconds, sorted in ascending order.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Parameter] if configuration is invalid
    /// - [crate::AudioSampleError::Processing] if CQT computation fails
    /// - [crate::AudioSampleError::Processing] if peak picking fails
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
    /// Each value is `w_m·Σ_b R(|X[b,t]| − |X[b,t−1]|) + w_p·Σ_b dev[b,t]`, where R is
    /// rectification or absolute value and `dev` is the wrapped phase deviation (see
    /// [`complex::phase_deviation`]), optionally passed through `ln(1 + C·x)`. Higher values
    /// indicate stronger onsets.
    ///
    /// # Arguments
    ///
    /// * `onset_config` - Complex onset detection configuration
    ///
    /// # Returns
    ///
    /// Onset detection function values for each frame.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Processing] if CQT computation fails
    #[inline]
    fn onset_detection_function_complex(
        &self,
        onset_config: &ComplexOnsetConfig,
    ) -> AudioSampleResult<NonEmptyVec<f64>> {
        // Compute the CQT once and share it between the magnitude-difference
        // and phase-deviation computations. `magnitude_difference_matrix` and
        // `phase_deviation_matrix` each compute their own CQT when called
        // standalone; here we avoid that duplicated transform.
        let cqt_result =
            self.constant_q_transform(&onset_config.cqt_config, onset_config.hop_size)?;
        let sample_rate = self.sample_rate_hz();

        let mag = cqt_result.to_magnitude();
        let mag_diff = magnitude_difference(mag.view());
        let phase_dev = phase_deviation(cqt_result.data.view(), onset_config, sample_rate);

        Ok(combine_complex_odf(&mag_diff, &phase_dev, onset_config))
    }

    /// Computes the magnitude difference matrix from CQT.
    ///
    /// Each entry `[b, t]` is the signed CQT magnitude difference `|X[b,t]| − |X[b,t−1]|`
    /// (column 0 is zero).
    ///
    /// # Arguments
    ///
    /// * `config` - Complex onset detection configuration
    ///
    /// # Returns
    ///
    /// 2D array of magnitude differences with shape `(frequency_bins, frames)`.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Processing] if CQT computation fails
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
    /// Each entry `[b, t]` is `|princarg(φ[b,t] − φ[b,t−1] − 2π·f_b·H/f_s)|` in radians, the
    /// deviation of the measured phase advance from that of a sinusoid at the bin centre
    /// frequency f_b (see [`complex::phase_deviation`]). Column 0 is zero.
    ///
    /// # Arguments
    ///
    /// * `config` - Complex onset detection configuration
    ///
    /// # Returns
    ///
    /// 2D array of phase deviations with shape `(frequency_bins, frames)`.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Processing] if CQT computation fails
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

#[cfg(test)]
mod cqt_dedup_tests {
    use super::*;
    use crate::AudioSamples;
    use crate::operations::traits::AudioOnsetDetection;
    use crate::sample_rate;
    use ndarray::Array1;

    /// Item 2 parity: computing the CQT once (the optimized combined path)
    /// must produce a bit-identical ODF to the previous composition that
    /// computed the CQT twice (once per matrix). The CQT is deterministic, so
    /// `combine_complex_odf` over the standalone matrices is the reference.
    #[test]
    fn test_complex_odf_single_cqt_bit_identical() {
        // A multi-tone signal long enough to yield several CQT frames.
        let sr = 44100usize;
        let n = 16384;
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr as f64;
                0.6 * (2.0 * std::f64::consts::PI * 220.0 * t).sin()
                    + 0.3 * (2.0 * std::f64::consts::PI * 880.0 * t).sin()
            })
            .collect();
        let audio = AudioSamples::new_mono(Array1::from_vec(data), sample_rate!(44100)).unwrap();

        let config = ComplexOnsetConfig::default();

        // Reference: standalone matrices (each computes its own CQT), combined.
        let mag_diff = audio.magnitude_difference_matrix(&config).unwrap();
        let phase_dev = audio.phase_deviation_matrix(&config).unwrap();
        let expected = combine_complex_odf(&mag_diff, &phase_dev, &config);

        // Optimized: combined path computes the CQT once internally.
        let actual = audio.onset_detection_function_complex(&config).unwrap();

        let exp = expected.as_slice();
        let act = actual.as_slice();
        assert_eq!(exp.len(), act.len(), "ODF length mismatch");
        for (i, (e, a)) in exp.iter().zip(act.iter()).enumerate() {
            assert_eq!(
                e.to_bits(),
                a.to_bits(),
                "ODF frame {i} differs: {e} vs {a}"
            );
        }
    }

    /// Regression: `combine_complex_odf` sliced a column of the row-major
    /// `(bins, frames)` arrays and expected it to be contiguous, so
    /// `complex_onset_detection` hit `unreachable!` for every preset as soon as
    /// the CQT produced more than one frame (any signal longer than ~0.37 s at
    /// 44.1 kHz).
    #[test]
    fn complex_onset_detection_does_not_panic_on_long_signals() {
        let sr = 22050usize;
        let n = 3 * sr / 2; // 1.5 s
        let onsets = [0.2, 0.6, 1.0];
        let data: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr as f64;
                onsets
                    .iter()
                    .filter(|&&t0| t >= t0)
                    .map(|&t0| {
                        let dt = t - t0;
                        (-dt / 0.1).exp() * (2.0 * std::f64::consts::PI * 330.0 * dt).sin()
                    })
                    .sum::<f64>()
                    * 0.3
            })
            .collect();
        let audio = AudioSamples::new_mono(Array1::from_vec(data), sample_rate!(22050)).unwrap();
        for config in [
            ComplexOnsetConfig::default(),
            ComplexOnsetConfig::percussive(),
            ComplexOnsetConfig::musical(),
            ComplexOnsetConfig::speech(),
        ] {
            let odf = audio.onset_detection_function_complex(&config).unwrap();
            assert!(odf.len().get() > 1, "expected several frames");
            assert!(odf.iter().all(|v| v.is_finite()));
            let onsets = audio.complex_onset_detection(&config).unwrap();
            assert!(onsets.iter().all(|&t| (0.0..=1.5).contains(&t)));
        }
    }

    /// Regression: with the old `PeakPickingConfig::default()` (1024-frame delta
    /// window, 3-point median filter, 512-frame separation) `detect_onsets` found
    /// 1 of the 10 notes of this sequence. The new defaults find all of them,
    /// each within 50 ms, without spurious detections.
    #[test]
    fn detect_onsets_default_finds_every_note() {
        let sr: f64 = 22050.0;
        let onsets: [f64; 10] = [0.25, 0.75, 1.1, 1.6, 2.3, 2.8, 3.5, 4.0, 4.6, 5.2];
        let freqs = [
            262.0, 330.0, 392.0, 523.0, 440.0, 349.0, 294.0, 494.0, 587.0, 220.0,
        ];
        let n = (6.0 * sr) as usize;
        let mut data = vec![0.0f64; n];
        for (&t0, &f) in onsets.iter().zip(&freqs) {
            let start = (t0 * sr).round() as usize;
            for (j, v) in data[start..]
                .iter_mut()
                .take((1.5 * sr) as usize)
                .enumerate()
            {
                let t = j as f64 / sr;
                let env = (t / 0.005).min(1.0) * (-t / 0.12).exp();
                let w = 2.0 * std::f64::consts::PI * f * t;
                *v += 0.3 * env * (w.sin() + 0.5 * (2.0 * w).sin() + 0.25 * (3.0 * w).sin());
            }
        }
        let audio = AudioSamples::new_mono(Array1::from_vec(data), sample_rate!(22050)).unwrap();
        let detected = audio
            .detect_onsets(&OnsetDetectionConfig::default())
            .unwrap();
        assert_eq!(detected.len(), onsets.len(), "detected {detected:?}");
        for (d, t) in detected.iter().zip(&onsets) {
            assert!((d - t).abs() <= 0.05, "detected {detected:?}");
        }
    }
}
