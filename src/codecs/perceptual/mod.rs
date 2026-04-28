//! Psychoacoustic analysis types and the core analysis pipeline.
//!
//! ## What
//!
//! This module provides the data types and entry-point function for the
//! psychoacoustic analysis pipeline used by perceptual audio codecs. It models
//! how the human auditory system masks quieter sounds with louder ones, allowing
//! a codec to allocate more bits where they are perceptually important and fewer
//! where the ear cannot detect the difference.
//!
//! ## Why
//!
//! Perceptual codecs (MP3, AAC, Vorbis, Opus) achieve high compression by
//! quantizing spectral coefficients below the masking threshold more coarsely.
//! This module exposes the analysis layer — computing masking thresholds and
//! signal-to-mask ratios — independently of any specific bit-allocation or
//! entropy-coding scheme.
//!
//! ## How
//!
//! Use [`BandLayout::bark`] or [`BandLayout::mel`] to build a perceptual band
//! partition, construct a [`PsychoacousticConfig`] with masking parameters, then
//! call [`analyse_signal`] (or the [`AudioPerceptualAnalysis`] trait method) on
//! an [`AudioSamples`] value:
//!
//! ```rust
//! # #[cfg(feature = "psychoacoustic")] {
//! use audio_samples::{
//!     AudioPerceptualAnalysis, BandLayout, PsychoacousticConfig,
//!     sine_wave, sample_rate,
//! };
//! use spectrograms::WindowType;
//! use non_empty_slice::NonEmptySlice;
//! use std::num::NonZeroUsize;
//! use std::time::Duration;
//!
//! let signal = sine_wave::<f32>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.5);
//! let n_bins = NonZeroUsize::new(1024).unwrap();
//! let layout = BandLayout::bark(NonZeroUsize::new(24).unwrap(), 44100.0, n_bins);
//!
//! let weights = vec![1.0f32; 24];
//! let weights_slice = NonEmptySlice::from_slice(&weights).unwrap();
//! let config = PsychoacousticConfig::new(
//!     -60.0, 14.5, 5.5, 25.0, 6.0, weights_slice, 1e-10,
//! );
//!
//! let result = signal
//!     .analyse_psychoacoustic(WindowType::Hanning, &layout, &config)
//!     .unwrap();
//! assert_eq!(result.band_metrics.len().get(), 24);
//! # }
//! ```

use std::num::{NonZeroU32, NonZeroUsize};

use non_empty_slice::{NonEmptySlice, NonEmptyVec};
use spectrograms::{MdctParams, WindowType};

use crate::traits::AudioTypeConversion;
use crate::{AudioSampleError, AudioSampleResult, AudioSamples, ParameterError, StandardSample};

pub mod bands;
pub mod codec;
pub mod masking;
pub mod quantization;
pub mod stereo;
mod traits;

pub use codec::{AudioCodec, EncodedSegment, PerceptualCodec, PerceptualEncodedAudio};
pub use masking::{apply_temporal_masking, detect_transient_windows};
pub use stereo::{StereoPerceptualCodec, StereoPerceptualEncodedAudio};
pub use traits::AudioPerceptualAnalysis;

// ── Core types ────────────────────────────────────────────────────────────────

/// A single frequency band defined by its spectral-bin range, centre frequency,
/// and normalised perceptual position.
///
/// Bands are the fundamental unit of a [`BandLayout`]. Each band covers
/// `[start_bin, end_bin)` in the caller's spectral-bin array (MDCT or FFT).
///
/// ## Invariants
///
/// `end_bin > start_bin` is enforced by [`Band::try_new`]; the unsafe
/// [`Band::new`] constructor requires the caller to uphold this.
#[derive(Debug, Clone, PartialEq)]
pub struct Band {
    /// Index of the first spectral bin belonging to this band (inclusive).
    pub start_bin: usize,
    /// Index one past the last spectral bin belonging to this band (exclusive).
    pub end_bin: usize,
    /// Centre frequency of the band in hertz.
    pub centre_frequency: f32,
    /// Normalised position in [0, 1] on the underlying perceptual scale
    /// (Bark, Mel, or custom).
    pub perceptual_position: f32,
}

impl Band {
    /// Creates a `Band`, returning an error if `end_bin <= start_bin`.
    ///
    /// # Arguments
    /// - `start_bin` – First bin index (inclusive).
    /// - `end_bin` – One past the last bin index (exclusive). Must be `> start_bin`.
    /// - `centre_frequency` – Centre frequency in Hz.
    /// - `perceptual_position` – Normalised position in [0, 1].
    ///
    /// # Errors
    /// Returns [`AudioSampleError::Parameter`] if `end_bin <= start_bin`.
    #[inline]
    #[must_use]
    pub fn try_new(
        start_bin: usize,
        end_bin: usize,
        centre_frequency: f32,
        perceptual_position: f32,
    ) -> AudioSampleResult<Self> {
        if end_bin <= start_bin {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "end_bin",
                format!(
                    "end_bin ({end_bin}) must be greater than start_bin ({start_bin})"
                ),
            )));
        }
        Ok(Self {
            start_bin,
            end_bin,
            centre_frequency,
            perceptual_position,
        })
    }

    /// Creates a `Band` without checking the `end_bin > start_bin` invariant.
    ///
    /// # Safety
    ///
    /// The caller must ensure `end_bin > start_bin`. Violating this invariant
    /// will cause incorrect (potentially zero-width) band aggregation in the
    /// masking model.
    #[inline]
    #[must_use]
    pub const unsafe fn new(
        start_bin: usize,
        end_bin: usize,
        centre_frequency: f32,
        perceptual_position: f32,
    ) -> Self {
        Self {
            start_bin,
            end_bin,
            centre_frequency,
            perceptual_position,
        }
    }

    /// Returns the width of the band in spectral bins.
    #[inline]
    #[must_use]
    pub const fn width(&self) -> usize {
        self.end_bin.saturating_sub(self.start_bin)
    }
}

/// A non-empty ordered collection of [`Band`]s that together partition the
/// audible spectrum.
///
/// Construct using [`BandLayout::new`] for custom layouts, or the preset
/// constructors [`BandLayout::bark`] and [`BandLayout::mel`] for standard
/// perceptual scales (see [`bands`]).
#[derive(Debug, Clone, PartialEq)]
pub struct BandLayout {
    /// The ordered list of bands.
    pub bands: NonEmptyVec<Band>,
}

impl BandLayout {
    /// Creates a `BandLayout` from a non-empty slice of bands.
    ///
    /// # Arguments
    /// - `bands` – At least one [`Band`]. Bands should be ordered by
    ///   `start_bin` and cover non-overlapping frequency ranges.
    #[inline]
    #[must_use]
    pub fn new(bands: &NonEmptySlice<Band>) -> Self {
        Self {
            bands: bands.to_non_empty_vec(),
        }
    }

    /// Returns the number of bands.
    #[inline]
    #[must_use]
    pub fn len(&self) -> NonZeroUsize {
        self.bands.len()
    }

    /// Returns a view over all bands.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &NonEmptySlice<Band> {
        self.bands.as_non_empty_slice()
    }
}

/// Per-band psychoacoustic metrics computed by [`analyse_signal`].
///
/// Stores the energy, masking threshold, signal-to-mask ratio, importance, and
/// allowed quantization noise for a single frequency band. Used by codec bit
/// allocators to decide how finely to quantize each band's MDCT coefficients.
#[derive(Debug, Clone, PartialEq)]
pub struct BandMetric {
    /// The frequency band these metrics apply to.
    pub band: Band,
    /// Aggregated band energy in dB.
    pub energy: f32,
    /// Masking threshold for the band in dB (ATH + spread contributions).
    pub masking_threshold: f32,
    /// Signal-to-mask ratio in dB (`energy − masking_threshold`).
    pub signal_to_mask_ratio: f32,
    /// Perceptual importance: `weight × max(SMR, 0)`. Higher values warrant
    /// more bits.
    pub importance: f32,
    /// Quantization noise budget in dB: `masking_threshold − max(SMR, 0)`.
    pub allowed_noise: f32,
}

impl BandMetric {
    /// Creates a `BandMetric` from pre-computed field values.
    #[inline]
    #[must_use]
    pub const fn new(
        band: Band,
        energy: f32,
        masking_threshold: f32,
        signal_to_mask_ratio: f32,
        importance: f32,
        allowed_noise: f32,
    ) -> Self {
        Self {
            band,
            energy,
            masking_threshold,
            signal_to_mask_ratio,
            importance,
            allowed_noise,
        }
    }
}

/// A non-empty collection of [`BandMetric`] values, one per band.
#[derive(Debug, Clone, PartialEq)]
pub struct BandMetrics {
    /// The per-band metrics.
    pub metrics: NonEmptyVec<BandMetric>,
}

impl BandMetrics {
    /// Creates a `BandMetrics` from a non-empty slice of metrics.
    #[inline]
    #[must_use]
    pub fn new(metrics: &NonEmptySlice<BandMetric>) -> Self {
        Self {
            metrics: metrics.to_non_empty_vec(),
        }
    }

    /// Returns the number of band metrics.
    #[inline]
    #[must_use]
    pub fn len(&self) -> NonZeroUsize {
        self.metrics.len()
    }

    /// Returns a view over all band metrics.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &NonEmptySlice<BandMetric> {
        self.metrics.as_non_empty_slice()
    }
}

/// Configuration for the psychoacoustic masking model.
///
/// ## Purpose
///
/// Holds the tunable parameters that control how the masking model computes
/// thresholds. Different codec profiles or quality levels may use different
/// values.
///
/// ## Intended Usage
///
/// Pass to [`analyse_signal`] or [`AudioPerceptualAnalysis::analyse_psychoacoustic`]
/// together with a [`BandLayout`]. The number of entries in `perceptual_weights`
/// must equal `band_layout.len()`; validate with
/// [`PsychoacousticConfig::is_compatible_with`] before use.
///
/// ## Invariants
///
/// `perceptual_weights` is non-empty and all weights are typically in (0, 1].
/// `epsilon > 0` prevents log10(0) in energy computation.
#[derive(Debug, Clone, PartialEq)]
pub struct PsychoacousticConfig {
    /// Absolute noise floor in dB — bands below this level are considered silent.
    pub noise_floor: f32,
    /// Tonal masking gain in dB: how far below a tonal (sinusoidal) masker the
    /// masking threshold sits at zero Bark distance. MPEG-1 value: 14.5 dB.
    pub masking_gain: f32,
    /// Noise masking gain in dB: the equivalent offset for broadband noise maskers.
    ///
    /// Noise maskers are less effective than tonal maskers, so this value is
    /// smaller — a smaller offset means the threshold is closer to (not as far
    /// below) the masker level. MPEG-1 value: 5.5 dB.
    pub noise_masking_gain: f32,
    /// Upward masking spread slope in dB/Bark (masker raises threshold of
    /// higher-frequency bands). Typical MPEG-1 value: ~25.
    pub upward_spread: f32,
    /// Downward masking spread slope in dB/Bark (masker raises threshold of
    /// lower-frequency bands). Typical MPEG-1 value: ~6.
    pub downward_spread: f32,
    /// Per-band importance weights used to scale the SMR into an `importance`
    /// score. Length must equal the number of bands in the associated
    /// [`BandLayout`].
    pub perceptual_weights: NonEmptyVec<f32>,
    /// Small positive value added to band energy before taking log, preventing
    /// `−∞` dB for silent bands. Typical value: `1e-10`.
    pub epsilon: f32,
}

impl PsychoacousticConfig {
    /// Creates a `PsychoacousticConfig` from explicit parameters.
    ///
    /// # Arguments
    /// - `noise_floor` – Absolute noise floor in dB.
    /// - `masking_gain` – Tonal masker gain in dB (MPEG-1: 14.5 dB).
    /// - `noise_masking_gain` – Noise masker gain in dB (MPEG-1: 5.5 dB).
    /// - `upward_spread` – Upward spread slope in dB/Bark.
    /// - `downward_spread` – Downward spread slope in dB/Bark.
    /// - `perceptual_weights` – Per-band importance weights (length must match
    ///   the target [`BandLayout`]).
    /// - `epsilon` – Noise floor added before log; must be > 0.
    #[inline]
    #[must_use]
    pub fn new(
        noise_floor: f32,
        masking_gain: f32,
        noise_masking_gain: f32,
        upward_spread: f32,
        downward_spread: f32,
        perceptual_weights: &NonEmptySlice<f32>,
        epsilon: f32,
    ) -> Self {
        Self {
            noise_floor,
            masking_gain,
            noise_masking_gain,
            upward_spread,
            downward_spread,
            perceptual_weights: perceptual_weights.to_non_empty_vec(),
            epsilon,
        }
    }

    /// Returns the number of frequency bands this config is designed for.
    #[inline]
    #[must_use]
    pub fn band_count(&self) -> NonZeroUsize {
        self.perceptual_weights.len()
    }

    /// Returns `true` if this config's band count matches the given layout.
    #[inline]
    #[must_use]
    pub fn is_compatible_with(&self, band_layout: &BandLayout) -> bool {
        self.band_count() == band_layout.len()
    }

    // ── Preset constructors ───────────────────────────────────────────────────

    /// MPEG-1 psychoacoustic model 1 parameters.
    ///
    /// Uses the standard MPEG-1 (ISO 11172-3 Annex D) values: −60 dB noise floor,
    /// 14.5 dB masking gain, 25 dB/Bark upward spread, 6 dB/Bark downward spread.
    /// Suitable for moderate-quality perceptual coding.
    ///
    /// # Arguments
    /// - `perceptual_weights` – Per-band importance weights. Length must match the
    ///   target [`BandLayout`]. Use [`PsychoacousticConfig::uniform_weights`] if
    ///   you have no prior band preferences.
    #[inline]
    #[must_use]
    pub fn mpeg1(perceptual_weights: &NonEmptySlice<f32>) -> Self {
        Self::new(-60.0, 14.5, 5.5, 25.0, 6.0, perceptual_weights, 1e-10)
    }

    /// Conservative masking profile — lower noise tolerance, higher quality.
    ///
    /// Uses a lower masking gain (10 dB) and tighter spread slopes. Fewer bins
    /// will be below the masking threshold, so less aggressive quantization is
    /// implied. Use when audio quality is paramount.
    ///
    /// # Arguments
    /// - `perceptual_weights` – Per-band importance weights.
    #[inline]
    #[must_use]
    pub fn conservative(perceptual_weights: &NonEmptySlice<f32>) -> Self {
        Self::new(-80.0, 10.0, 3.0, 20.0, 4.0, perceptual_weights, 1e-10)
    }

    /// Aggressive masking profile — higher noise tolerance, lower bitrate.
    ///
    /// Uses a higher masking gain (18 dB) and wider spread slopes. More bins
    /// are considered masked, allowing coarser quantization. Use when bitrate
    /// reduction is the priority.
    ///
    /// # Arguments
    /// - `perceptual_weights` – Per-band importance weights.
    #[inline]
    #[must_use]
    pub fn aggressive(perceptual_weights: &NonEmptySlice<f32>) -> Self {
        Self::new(-40.0, 18.0, 7.0, 30.0, 8.0, perceptual_weights, 1e-10)
    }

    /// Creates a `NonEmptyVec<f32>` of uniform weights (all 1.0) for `n_bands` bands.
    ///
    /// Use as the `perceptual_weights` argument when you have no prior band-importance
    /// preferences and want all bands treated equally.
    ///
    /// # Arguments
    /// - `n_bands` – Number of bands. Must match the target [`BandLayout::len()`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "psychoacoustic")] {
    /// use audio_samples::{BandLayout, PsychoacousticConfig};
    /// use std::num::NonZeroUsize;
    ///
    /// let n_bands = NonZeroUsize::new(24).unwrap();
    /// let weights = PsychoacousticConfig::uniform_weights(n_bands);
    /// let layout = BandLayout::bark(n_bands, 44100.0, NonZeroUsize::new(1024).unwrap());
    /// let config = PsychoacousticConfig::mpeg1(weights.as_non_empty_slice());
    /// assert!(config.is_compatible_with(&layout));
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub fn uniform_weights(n_bands: NonZeroUsize) -> NonEmptyVec<f32> {
        let w = vec![1.0_f32; n_bands.get()];
        NonEmptyVec::new(w).expect("n_bands >= 1")
    }
}

/// Output of the psychoacoustic analysis pipeline.
///
/// ## Purpose
///
/// Bundles everything a downstream codec or bit-allocator needs: the raw
/// spectral coefficients to quantize, the per-bin energies used to build the
/// masking model, the per-band metrics summarising audibility and allowed noise,
/// and enough metadata to reconstruct the signal via [`reconstruct_signal`].
///
/// ## Fields
///
/// - `coefficients` – Flattened MDCT coefficients stored in row-major (C) order
///   with shape `(n_coefficients, n_frames)`. Index as `k * n_frames + f` for
///   bin `k`, frame `f`. These are the values a codec would quantize.
/// - `bin_energies` – Average power per MDCT bin across all analysed frames.
/// - `band_metrics` – One [`BandMetric`] per band.
/// - `n_coefficients` – Number of MDCT bins per frame (`window_size / 2`).
/// - `n_frames` – Number of analysis frames.
/// - `original_length` – Original signal length in samples, needed for accurate IMDCT.
/// - `sample_rate` – Sample rate of the analysed signal.
/// - `mdct_params` – The MDCT parameters used for analysis; required by [`reconstruct_signal`].
#[derive(Debug, Clone)]
pub struct PerceptualAnalysisResult {
    /// Flattened MDCT coefficients, shape `(n_coefficients, n_frames)`, row-major.
    pub coefficients: NonEmptyVec<f32>,
    /// Average power per MDCT bin (linear scale) across all frames.
    pub bin_energies: NonEmptyVec<f32>,
    /// Per-band psychoacoustic metrics.
    pub band_metrics: BandMetrics,
    /// Number of MDCT coefficients (bins) per frame.
    pub n_coefficients: NonZeroUsize,
    /// Number of MDCT frames.
    pub n_frames: NonZeroUsize,
    /// Original signal length in samples.
    pub original_length: usize,
    /// Sample rate of the analysed signal.
    pub sample_rate: NonZeroU32,
    /// MDCT parameters used during analysis (needed for reconstruction).
    pub mdct_params: MdctParams,
}

impl PerceptualAnalysisResult {
    /// Creates a `PerceptualAnalysisResult` from pre-computed fields.
    #[inline]
    #[must_use]
    pub fn new(
        coefficients: NonEmptyVec<f32>,
        bin_energies: NonEmptyVec<f32>,
        band_metrics: BandMetrics,
        n_coefficients: NonZeroUsize,
        n_frames: NonZeroUsize,
        original_length: usize,
        sample_rate: NonZeroU32,
        mdct_params: MdctParams,
    ) -> Self {
        Self {
            coefficients,
            bin_energies,
            band_metrics,
            n_coefficients,
            n_frames,
            original_length,
            sample_rate,
            mdct_params,
        }
    }
}

// ── Analysis entry point ──────────────────────────────────────────────────────

/// Runs the psychoacoustic analysis pipeline on `signal`.
///
/// This is the free-function form of
/// [`AudioPerceptualAnalysis::analyse_psychoacoustic`]. Prefer the trait method
/// when working with [`AudioSamples`] values directly; use this function when
/// you need to call it from outside a trait impl context.
///
/// # Arguments
/// - `signal` – The audio signal to analyse. Must be mono.
/// - `window` – Window function for the MDCT. [`WindowType::Hanning`] is a good
///   default; only the sine window (`MdctParams::sine_window`) gives perfect
///   reconstruction.
/// - `band_layout` – Perceptual band partitioning (see [`BandLayout::bark`],
///   [`BandLayout::mel`]).
/// - `config` – Psychoacoustic model configuration. The number of perceptual
///   weights must equal `band_layout.len()`.
///
/// # Errors
/// - [`AudioSampleError::Parameter`] if `signal` is not mono, is shorter than
///   4 samples, or `config` is incompatible with `band_layout`.
/// - [`AudioSampleError::Spectrogram`] if the MDCT computation fails.
pub fn analyse_signal<T>(
    signal: &AudioSamples<T>,
    window: WindowType,
    band_layout: &BandLayout,
    config: &PsychoacousticConfig,
) -> AudioSampleResult<PerceptualAnalysisResult>
where
    T: StandardSample,
{
    analyse_signal_with_window_size(signal, window, None, band_layout, config)
}

/// Runs the psychoacoustic analysis pipeline with an explicit MDCT window size.
///
/// Like [`analyse_signal`] but lets the caller pin the MDCT window size instead
/// of having it auto-selected. This is the entry point used by [`PerceptualCodec`]
/// when [`PerceptualCodec::window_size`] is `Some`.
///
/// # Arguments
/// - `signal` – The audio signal to analyse. Must be mono.
/// - `window` – Window function for the MDCT.
/// - `window_size` – If `Some`, the MDCT window size to use. Must be even and
///   at least 4. If `None`, an automatic size of `min(2048, signal_length)` is used.
/// - `band_layout` – Perceptual band partitioning.
/// - `config` – Psychoacoustic model configuration.
///
/// # Errors
/// Same as [`analyse_signal`], plus [`AudioSampleError::Parameter`] if `window_size`
/// is odd or less than 4.
pub fn analyse_signal_with_window_size<T>(
    signal: &AudioSamples<T>,
    window: WindowType,
    window_size: Option<NonZeroUsize>,
    band_layout: &BandLayout,
    config: &PsychoacousticConfig,
) -> AudioSampleResult<PerceptualAnalysisResult>
where
    T: StandardSample,
{
    if !config.is_compatible_with(band_layout) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "config",
            format!(
                "PsychoacousticConfig has {} weights but BandLayout has {} bands",
                config.band_count(),
                band_layout.len(),
            ),
        )));
    }

    if signal.num_channels().get() != 1 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "signal",
            "psychoacoustic analysis requires mono input; mix down or extract a channel first",
        )));
    }

    let n_samples = signal.samples_per_channel().get();

    let window_size_val = if let Some(ws) = window_size {
        let ws = ws.get();
        if ws < 4 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "window_size",
                "window size must be at least 4",
            )));
        }
        if ws % 2 != 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "window_size",
                "window size must be even",
            )));
        }
        ws
    } else {
        // Auto: largest even value in [4, min(2048, n_samples)].
        let raw = 2048_usize.min(n_samples);
        if raw % 2 == 0 { raw } else { raw - 1 }
    };

    if window_size_val < 4 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "signal",
            format!("signal too short for psychoacoustic analysis: {n_samples} samples (minimum 4)"),
        )));
    }

    let window_size = NonZeroUsize::new(window_size_val).expect("validated >= 4");
    let hop_size = NonZeroUsize::new(window_size_val / 2).expect("window_size >= 4");
    let mdct_params = MdctParams::new(window_size, hop_size, window)?;
    let n_bins = mdct_params.n_coefficients();

    let sample_rate = signal.sample_rate();
    let original_length = signal.samples_per_channel().get();

    // Convert to f32 and extract the mono channel.
    let signal_f32 = signal.to_format::<f32>();
    let channel = signal_f32
        .channels()
        .next()
        .expect("validated mono above");
    let samples: &[f32] = channel
        .as_slice()
        .expect("mono channel is always contiguous");

    // SAFETY: signal has at least 4 samples (validated above), so non-empty.
    let samples_ne = unsafe { NonEmptySlice::new_unchecked(samples) };
    let mdct_matrix = spectrograms::mdct_f32(samples_ne, &mdct_params)?;
    // mdct_matrix shape: (n_bins, n_frames)

    let n_frames_raw = mdct_matrix.ncols();

    // Flatten coefficients in row-major (C) order: index as k * n_frames + f.
    let coefficients_vec: Vec<f32> = mdct_matrix.iter().copied().collect();
    let coefficients = NonEmptyVec::new(coefficients_vec)
        .expect("MDCT matrix is non-empty for valid input");

    // Average power per bin across frames (linear scale).
    let bin_energies_vec: Vec<f32> = (0..n_bins)
        .map(|k| {
            let sum: f32 = (0..n_frames_raw).map(|f| mdct_matrix[(k, f)].powi(2)).sum();
            sum / n_frames_raw as f32
        })
        .collect();
    let bin_energies = NonEmptyVec::new(bin_energies_vec)
        .expect("n_bins >= 1 for window_size >= 4");

    let band_metrics = masking::compute_band_metrics(
        bin_energies.as_non_empty_slice().as_slice(),
        band_layout,
        config,
        n_bins,
    );

    let n_coefficients = NonZeroUsize::new(n_bins).expect("n_bins >= 1 for window_size >= 4");
    let n_frames = NonZeroUsize::new(n_frames_raw).expect("at least one MDCT frame");

    Ok(PerceptualAnalysisResult::new(
        coefficients,
        bin_energies,
        band_metrics,
        n_coefficients,
        n_frames,
        original_length,
        sample_rate,
        mdct_params,
    ))
}

// ── Reconstruction ────────────────────────────────────────────────────────────

/// Reconstructs an audio signal from (possibly quantized) MDCT coefficients.
///
/// This is the inverse path of [`analyse_signal`]: given coefficients in the
/// same layout as [`PerceptualAnalysisResult::coefficients`], it runs the IMDCT
/// with overlap-add and wraps the result in an [`AudioSamples`] value.
///
/// The function is codec-agnostic — it works on any `f32` coefficient array,
/// whether the values were losslessly preserved from [`analyse_signal`] or
/// replaced with dequantized integers from a codec's decode step.
///
/// # Arguments
/// - `coefficients` – Flattened MDCT coefficients, row-major with shape
///   `(n_coefficients, n_frames)`. Index as `k * n_frames + f`.
/// - `n_coefficients` – Number of MDCT bins per frame.
/// - `n_frames` – Number of analysis frames.
/// - `params` – MDCT parameters used during analysis (see
///   [`PerceptualAnalysisResult::mdct_params`]).
/// - `original_length` – If provided, the output signal is truncated to this
///   many samples, matching the original signal length precisely.
/// - `sample_rate` – Sample rate for the returned [`AudioSamples`].
///
/// # Errors
/// - [`AudioSampleError::Parameter`] if `coefficients.len() != n_coefficients × n_frames`
///   or if the IMDCT produces an empty output.
/// - [`AudioSampleError::Spectrogram`] if the IMDCT computation fails.
pub fn reconstruct_signal(
    coefficients: &NonEmptyVec<f32>,
    n_coefficients: NonZeroUsize,
    n_frames: NonZeroUsize,
    params: &MdctParams,
    original_length: Option<usize>,
    sample_rate: NonZeroU32,
) -> AudioSampleResult<AudioSamples<'static, f32>> {
    use ndarray::Array2;

    let nc = n_coefficients.get();
    let nf = n_frames.get();
    let expected = nc * nf;

    if coefficients.len().get() != expected {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "coefficients",
            format!(
                "expected {nc} × {nf} = {expected} elements, got {}",
                coefficients.len()
            ),
        )));
    }

    // Rebuild Array2<f32> with shape (n_coefficients, n_frames) from the
    // row-major flat buffer: coef_matrix[(k, f)] = coefficients[k * nf + f].
    let coef_vec: Vec<f32> = coefficients.iter().copied().collect();
    let coef_matrix = Array2::from_shape_vec((nc, nf), coef_vec)?;

    let samples = spectrograms::imdct_f32(&coef_matrix, params, original_length)?;

    let samples_ne = NonEmptyVec::new(samples).map_err(|_| AudioSampleError::EmptyData)?;

    Ok(AudioSamples::from_mono_vec(samples_ne, sample_rate))
}
