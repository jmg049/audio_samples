//! librosa-compatible frame-wise audio features: spectral shape descriptors, RMS,
//! zero-crossing rate, PCEN, frequency weighting curves and the reassigned
//! spectrogram.
//!
//! ## What
//!
//! This module provides the [`AudioFeatures`] trait and the free functions it is
//! built from. Together they reproduce librosa 1.0's per-frame feature extractors:
//!
//! | librosa | here (on audio) | here (on a precomputed spectrogram / array) |
//! |---|---|---|
//! | `feature.spectral_centroid` | [`AudioFeatures::spectral_centroid_frames`] | [`spectral_centroid_from_spectrogram`] |
//! | `feature.spectral_bandwidth` | [`AudioFeatures::spectral_bandwidth_frames`] | [`spectral_bandwidth_from_spectrogram`] |
//! | `feature.spectral_rolloff` | [`AudioFeatures::spectral_rolloff_frames`] | [`spectral_rolloff_from_spectrogram`] |
//! | `feature.spectral_flatness` | [`AudioFeatures::spectral_flatness_frames`] | [`spectral_flatness_from_spectrogram`] |
//! | `feature.spectral_contrast` | [`AudioFeatures::spectral_contrast_frames`] | [`spectral_contrast_from_spectrogram`] |
//! | `feature.poly_features` | [`AudioFeatures::poly_features`] | [`poly_features_from_spectrogram`] |
//! | `feature.rms` | [`AudioFeatures::rms_frames`] | [`rms_from_spectrogram`] |
//! | `feature.zero_crossing_rate` | [`AudioFeatures::zero_crossing_rate_frames`] | – |
//! | `reassigned_spectrogram` | [`AudioFeatures::reassigned_spectrogram`] | – |
//! | `pcen` | – | [`pcen`], [`pcen_with_state`] |
//! | `A_weighting` … `Z_weighting`, `frequency_weighting`, `multi_frequency_weighting` | – | [`a_weighting`] … [`z_weighting`], [`frequency_weighting`], [`multi_frequency_weighting`] |
//! | `perceptual_weighting` | – | [`perceptual_weighting`] |
//!
//! The frequency-weighting curves live here (rather than in
//! [`crate::utils::audio_math`]) because their main consumer,
//! [`perceptual_weighting`], needs the dB conversion of
//! [`crate::operations::spectrum`]; they are plain functions of frequency.
//!
//! ## Why
//!
//! The scalar spectral descriptors of [`AudioStatistics`](crate::AudioStatistics)
//! summarise a whole signal with one number (and use their own weighting
//! conventions), and [`AudioEnvelopes`](crate::operations::traits::AudioEnvelopes) frames signals
//! differently from librosa. Machine-learning pipelines and code ported from
//! Python need librosa's per-frame features with its exact framing, centring and
//! formulas. Every config has a `librosa()` constructor that returns librosa
//! 1.0's defaults, so `X::librosa()` gives librosa-equivalent output (to
//! floating-point round-off). `Default` currently equals `librosa()` for every
//! config except [`PcenConfig`], which has no default: its hop length must equal
//! the analysed spectrogram's ([`PcenConfig::new`]).
//!
//! ## How
//!
//! Bring [`AudioFeatures`] into scope and call it on any [`AudioSamples`]. Outputs
//! are arrays shaped `(channels, n_features, n_frames)` exactly like librosa's
//! `(..., n, t)`; per-frame scalars have `n_features == 1`. When several
//! features share one STFT, compute the magnitude spectrogram once with
//! [`AudioSpectrum::spectrogram_matrix`](crate::operations::spectrum::AudioSpectrum::spectrogram_matrix)
//! and call the `*_from_spectrogram`
//! functions (librosa's `S=` argument).
//!
//! The output precision follows the input: the element type is the sample
//! type's [`AudioSample::Float`](crate::AudioSample::Float). `u8`, `i16`, `I24`
//! and `f32` audio gives `f32` arrays; `i32` and `f64` audio gives `f64`. The
//! whole computation runs in that type. The free functions on arrays are
//! generic over the element type ([`AudioFloat`]: `f32` or `f64`); their
//! frequency arguments and config parameters stay `f64` and are rounded to the
//! element type once. The frequency-weighting curves ([`a_weighting`], ...) are
//! tables of frequency and return `f64`.
//!
//! ```rust
//! use audio_samples::operations::features::{
//!     AudioFeatures, SpectralCentroidConfig, SpectralRolloffConfig,
//!     spectral_rolloff_from_spectrogram,
//! };
//! use audio_samples::operations::spectrum::{AudioSpectrum, StftConfig};
//! use audio_samples::utils::audio_math::fft_frequencies;
//! use audio_samples::{sample_rate, sine_wave};
//! use std::time::Duration;
//!
//! let audio = sine_wave::<f32>(1000.0, Duration::from_secs(1), sample_rate!(22050), 0.5);
//!
//! // librosa.feature.spectral_centroid(y=y, sr=22050): (1, 1, 44). f32 audio
//! // gives f32 output.
//! let centroid = audio.spectral_centroid_frames(&SpectralCentroidConfig::default()).unwrap();
//! let _: &ndarray::Array3<f32> = &centroid;
//! assert_eq!(centroid.dim(), (1, 1, 44));
//! assert!((centroid[[0, 0, 20]] - 1000.0).abs() < 20.0);
//!
//! // Reuse one magnitude spectrogram for several features (librosa's `S=`).
//! let s = audio.spectrogram_matrix(&StftConfig::default(), 1.0).unwrap();
//! let freqs = fft_frequencies(2048, 22050.0);
//! let rolloff = spectral_rolloff_from_spectrogram(
//!     s.view(), &freqs, &SpectralRolloffConfig::default(),
//! ).unwrap();
//! assert_eq!(rolloff.dim(), (1, 1, 44));
//! ```

use std::num::NonZeroUsize;

use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};
use num_complex::Complex;

use crate::audio_float::finite_float_rows;
use crate::operations::filterbanks::{Window, padded_window};
use crate::operations::spectrum::{
    DbConfig, StftConfig, StftPadMode, frame_square_sums, power_to_db, spectrogram_rows, stft,
};
use crate::utils::layout::{col, for_each_col_tile, for_each_row_tile, row};
use crate::utils::vmath::{ln_in_place, log10_f64};
use crate::{
    AudioFloat, AudioSampleError, AudioSampleResult, AudioSamples, ParameterError, StandardSample,
};

// =============================================================================
// Shared helpers
// =============================================================================

fn param_err(parameter: &str, reason: impl ToString) -> AudioSampleError {
    AudioSampleError::Parameter(ParameterError::invalid_value(parameter, reason))
}

fn config_err(operation: &str, reason: impl ToString) -> AudioSampleError {
    AudioSampleError::Parameter(ParameterError::invalid_configuration(operation, reason))
}

fn range_err(parameter: &str, value: f64, min: &str, max: &str, reason: &str) -> AudioSampleError {
    AudioSampleError::Parameter(ParameterError::out_of_range(
        parameter, value, min, max, reason,
    ))
}

/// `numpy.fft.rfftfreq(n_fft, 1 / sr)`, rounded exactly as numpy does
/// (`k * (1 / (n_fft * (1 / sr)))`), i.e. librosa's `fft_frequencies`.
fn rfft_frequencies(n_fft: usize, sample_rate: f64) -> Vec<f64> {
    let val = 1.0 / (n_fft as f64 * (1.0 / sample_rate));
    (0..=n_fft / 2).map(|k| k as f64 * val).collect()
}

fn validate_sample_rate(sample_rate: f64) -> AudioSampleResult<()> {
    if sample_rate.is_finite() && sample_rate > 0.0 {
        Ok(())
    } else {
        Err(range_err(
            "sample_rate",
            sample_rate,
            "> 0",
            "finite",
            "the sample rate must be finite and positive",
        ))
    }
}

/// librosa rejects negative (and, through `util.normalize`, non-finite) spectra in
/// the spectral shape descriptors.
fn check_spectrogram<F: AudioFloat>(what: &str, s: &ArrayView3<'_, F>) -> AudioSampleResult<()> {
    let (c, b, t) = s.dim();
    if c == 0 || b == 0 || t == 0 {
        return Err(config_err(
            what,
            format!("the spectrogram must be non-empty, got shape ({c}, {b}, {t})"),
        ));
    }
    if !all_finite_non_negative(s) {
        return Err(param_err(
            "spectrogram",
            format!("{what} is only defined for finite, non-negative spectrogram values"),
        ));
    }
    Ok(())
}

/// Whether every value of `s` is finite and `>= 0`, visited in memory order.
fn all_finite_non_negative<F: AudioFloat>(s: &ArrayView3<'_, F>) -> bool {
    // `0 <= v <= MAX` is false for negative values, infinities and NaN. The
    // chunks are checked without early exit inside, so the test vectorises.
    let ok = |v: &[F]| {
        v.chunks(256).all(|c| {
            c.iter()
                .fold(true, |a, &v| a & (v >= F::zero()) & (v <= F::max_value()))
        })
    };
    if let Some(all) = s.as_slice_memory_order() {
        return ok(all);
    }
    s.outer_iter().all(|ch| {
        let mut good = true;
        for_each_row_tile(ch, |_, tile| {
            for k in 0..tile.nrows() {
                good = good && ok(row(&tile, k));
            }
        });
        good
    })
}

/// Whether every value of `s` is finite, visited in memory order.
fn all_finite<F: AudioFloat>(s: &ArrayView3<'_, F>) -> bool {
    let ok = |v: &[F]| {
        v.chunks(256)
            .all(|c| c.iter().fold(true, |a, &v| a & (v.abs() <= F::max_value())))
    };
    if let Some(all) = s.as_slice_memory_order() {
        return ok(all);
    }
    s.outer_iter().all(|ch| {
        let mut good = true;
        for_each_row_tile(ch, |_, tile| {
            for k in 0..tile.nrows() {
                good = good && ok(row(&tile, k));
            }
        });
        good
    })
}

/// The `(1, frames)` output row of channel `c` as a slice.
fn out_row<F: AudioFloat>(out: &mut Array3<F>, c: usize) -> &mut [F] {
    out.index_axis_mut(Axis(0), c)
        .into_slice()
        .expect("a fresh (channels, 1, frames) array")
}

fn check_freq(what: &str, freq: &[f64], n_bins: usize) -> AudioSampleResult<()> {
    if freq.len() != n_bins {
        return Err(config_err(
            what,
            format!(
                "freq has {} entries but the spectrogram has {n_bins} frequency bins",
                freq.len()
            ),
        ));
    }
    Ok(())
}

/// `x^p` with numpy's exact fast paths for the common exponents.
fn pow_np<F: AudioFloat>(x: F, p: f64) -> F {
    if p == 1.0 {
        x
    } else if p == 2.0 {
        x * x
    } else if p == 0.5 {
        x.sqrt()
    } else {
        x.powf(F::of(p))
    }
}

/// Source index into a signal of length `len` for padded position `i`, following
/// `numpy.pad` semantics (`None` = constant zero).
fn pad_source_index(i: isize, len: usize, mode: StftPadMode) -> Option<usize> {
    let n = len as isize;
    if (0..n).contains(&i) {
        return Some(i as usize);
    }
    match mode {
        StftPadMode::Constant => None,
        StftPadMode::Edge => Some(if i < 0 { 0 } else { len - 1 }),
        StftPadMode::Reflect => {
            if len == 1 {
                return Some(0);
            }
            let period = 2 * (n - 1);
            let m = i.rem_euclid(period);
            Some(if m < n { m } else { period - m } as usize)
        }
        StftPadMode::Symmetric => {
            let period = 2 * n;
            let m = i.rem_euclid(period);
            Some(if m < n { m } else { period - 1 - m } as usize)
        }
    }
}

/// Number of frames `librosa.util.frame` produces, or an error if the (padded)
/// signal is shorter than one frame.
fn frame_count(len: usize, frame_length: usize, hop: usize) -> AudioSampleResult<usize> {
    if len < frame_length {
        return Err(param_err(
            "frame_length",
            format!("input is too short (n={len}) for frame_length={frame_length}"),
        ));
    }
    Ok(1 + (len - frame_length) / hop)
}

// =============================================================================
// Spectral shape descriptors: configs
// =============================================================================

/// Configuration of the frame-wise spectral centroid, mirroring
/// `librosa.feature.spectral_centroid`.
///
/// ## Purpose
///
/// Holds the STFT used to compute the magnitude spectrogram. The centroid itself
/// has no further parameters.
///
/// ## Intended usage
///
/// Pass to [`AudioFeatures::spectral_centroid_frames`]. [`Default`] is librosa's
/// default (`n_fft=2048`, hop 512, periodic Hann, centred, zero padding).
///
/// ## Invariants
///
/// Checked by [`SpectralCentroidConfig::validate`]: the STFT configuration is valid.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::features::SpectralCentroidConfig;
/// use audio_samples::operations::spectrum::StftConfig;
///
/// let cfg = SpectralCentroidConfig::new().with_stft(StftConfig::new().with_n_fft(nzu!(1024)));
/// assert_eq!(cfg.stft.n_fft.get(), 1024);
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SpectralCentroidConfig {
    /// STFT of the magnitude spectrogram (`|stft|`, power 1, as librosa).
    pub stft: StftConfig,
}

impl SpectralCentroidConfig {
    /// Returns the default configuration (same as [`Default`], which equals
    /// librosa 1.0's defaults).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// librosa 1.0's `feature.spectral_centroid` defaults. Currently the same as [`Default`];
    /// use this constructor where agreement with librosa is the requirement.
    #[must_use]
    pub fn librosa() -> Self {
        Self {
            stft: StftConfig::librosa(),
        }
    }

    /// Sets the STFT configuration.
    #[must_use]
    pub fn with_stft(mut self, stft: StftConfig) -> Self {
        self.stft = stft;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the STFT configuration is invalid.
    pub fn validate(&self) -> AudioSampleResult<()> {
        self.stft.validate()
    }
}

/// Configuration of the frame-wise spectral bandwidth, mirroring
/// `librosa.feature.spectral_bandwidth`.
///
/// ## Purpose
///
/// The bandwidth is the order-`p` deviation of the spectrum about its centroid:
/// `(sum_k S[k] |f[k] - centroid|^p)^(1/p)`, with `S` normalised to unit sum per
/// frame when `norm` is set.
///
/// ## Intended usage
///
/// Pass to [`AudioFeatures::spectral_bandwidth_frames`]. [`Default`] is librosa's
/// default (`p = 2`, `norm = true`). To supply a precomputed centroid (librosa's
/// `centroid=`), use [`spectral_bandwidth_from_spectrogram`].
///
/// ## Invariants
///
/// Checked by [`SpectralBandwidthConfig::validate`]: `p` finite and non-zero, STFT
/// valid.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::SpectralBandwidthConfig;
///
/// let cfg = SpectralBandwidthConfig::new().with_p(1.0).with_norm(false);
/// assert_eq!(cfg.p, 1.0);
/// assert!(!cfg.norm);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralBandwidthConfig {
    /// STFT of the magnitude spectrogram.
    pub stft: StftConfig,
    /// Order of the deviation. `2` (librosa default) gives a standard-deviation-like
    /// spread in Hz; `1` the mean absolute deviation. Finite and non-zero.
    pub p: f64,
    /// Normalise each frame to unit sum before weighting (librosa default `true`).
    /// Without normalisation the result scales with the signal level.
    pub norm: bool,
}

impl Default for SpectralBandwidthConfig {
    fn default() -> Self {
        Self {
            stft: StftConfig::default(),
            p: 2.0,
            norm: true,
        }
    }
}

impl SpectralBandwidthConfig {
    /// Returns the default configuration (same as [`Default`], which equals
    /// librosa 1.0's defaults).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// librosa 1.0's `feature.spectral_bandwidth` defaults. Currently the same as [`Default`];
    /// use this constructor where agreement with librosa is the requirement.
    #[must_use]
    pub fn librosa() -> Self {
        Self {
            stft: StftConfig::librosa(),
            ..Self::default()
        }
    }

    /// Sets the STFT configuration.
    #[must_use]
    pub fn with_stft(mut self, stft: StftConfig) -> Self {
        self.stft = stft;
        self
    }

    /// Sets the order `p`.
    #[must_use]
    pub fn with_p(mut self, p: f64) -> Self {
        self.p = p;
        self
    }

    /// Sets whether frames are normalised to unit sum.
    #[must_use]
    pub fn with_norm(mut self, norm: bool) -> Self {
        self.norm = norm;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `p` is zero or not finite, or the STFT configuration is
    /// invalid.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if !(self.p.is_finite() && self.p != 0.0) {
            return Err(param_err("p", "p must be finite and non-zero"));
        }
        self.stft.validate()
    }
}

/// Configuration of the frame-wise spectral rolloff, mirroring
/// `librosa.feature.spectral_rolloff`.
///
/// ## Purpose
///
/// The rolloff is the lowest bin frequency below which `roll_percent` of a frame's
/// total magnitude lies.
///
/// ## Intended usage
///
/// Pass to [`AudioFeatures::spectral_rolloff_frames`]. [`Default`] is librosa's
/// default (`roll_percent = 0.85`). Values near 1 approximate the maximum
/// frequency, values near 0 the minimum.
///
/// ## Invariants
///
/// Checked by [`SpectralRolloffConfig::validate`]: `0 < roll_percent < 1`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::SpectralRolloffConfig;
///
/// assert_eq!(SpectralRolloffConfig::default().roll_percent, 0.85);
/// assert!(SpectralRolloffConfig::new().with_roll_percent(1.0).validate().is_err());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralRolloffConfig {
    /// STFT of the magnitude spectrogram.
    pub stft: StftConfig,
    /// Fraction of the total magnitude, in the open interval `(0, 1)`.
    pub roll_percent: f64,
}

impl Default for SpectralRolloffConfig {
    fn default() -> Self {
        Self {
            stft: StftConfig::default(),
            roll_percent: 0.85,
        }
    }
}

impl SpectralRolloffConfig {
    /// Returns the default configuration (same as [`Default`], which equals
    /// librosa 1.0's defaults).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// librosa 1.0's `feature.spectral_rolloff` defaults. Currently the same as [`Default`];
    /// use this constructor where agreement with librosa is the requirement.
    #[must_use]
    pub fn librosa() -> Self {
        Self {
            stft: StftConfig::librosa(),
            ..Self::default()
        }
    }

    /// Sets the STFT configuration.
    #[must_use]
    pub fn with_stft(mut self, stft: StftConfig) -> Self {
        self.stft = stft;
        self
    }

    /// Sets the roll percentage.
    #[must_use]
    pub fn with_roll_percent(mut self, roll_percent: f64) -> Self {
        self.roll_percent = roll_percent;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `roll_percent` is not in `(0, 1)` or the STFT
    /// configuration is invalid.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if !(self.roll_percent > 0.0 && self.roll_percent < 1.0) {
            return Err(range_err(
                "roll_percent",
                self.roll_percent,
                "0 (exclusive)",
                "1 (exclusive)",
                "roll_percent must lie in the range (0, 1)",
            ));
        }
        self.stft.validate()
    }
}

/// Configuration of the frame-wise spectral flatness, mirroring
/// `librosa.feature.spectral_flatness`.
///
/// ## Purpose
///
/// Flatness (Wiener entropy) is the ratio of the geometric to the arithmetic mean
/// of `max(amin, S^power)` per frame: close to 1 for noise, close to 0 for tones.
///
/// ## Intended usage
///
/// Pass to [`AudioFeatures::spectral_flatness_frames`]. [`Default`] is librosa's
/// default (`amin = 1e-10`, `power = 2`, i.e. on the power spectrum).
///
/// ## Invariants
///
/// Checked by [`SpectralFlatnessConfig::validate`]: `amin` finite and positive,
/// `power` finite and positive.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::SpectralFlatnessConfig;
///
/// let cfg = SpectralFlatnessConfig::new().with_power(1.0);
/// assert_eq!(cfg.amin, 1e-10);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralFlatnessConfig {
    /// STFT of the magnitude spectrogram.
    pub stft: StftConfig,
    /// Floor applied to `S^power` before the logarithm (avoids `log(0)`); it sets
    /// the flatness of silent frames to 1. Positive.
    pub amin: f64,
    /// Exponent applied to the magnitude spectrogram: `2` (default) for power,
    /// `1` for magnitude. Positive.
    pub power: f64,
}

impl Default for SpectralFlatnessConfig {
    fn default() -> Self {
        Self {
            stft: StftConfig::default(),
            amin: 1e-10,
            power: 2.0,
        }
    }
}

impl SpectralFlatnessConfig {
    /// Returns the default configuration (same as [`Default`], which equals
    /// librosa 1.0's defaults).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// librosa 1.0's `feature.spectral_flatness` defaults. Currently the same as [`Default`];
    /// use this constructor where agreement with librosa is the requirement.
    #[must_use]
    pub fn librosa() -> Self {
        Self {
            stft: StftConfig::librosa(),
            ..Self::default()
        }
    }

    /// Sets the STFT configuration.
    #[must_use]
    pub fn with_stft(mut self, stft: StftConfig) -> Self {
        self.stft = stft;
        self
    }

    /// Sets `amin`.
    #[must_use]
    pub fn with_amin(mut self, amin: f64) -> Self {
        self.amin = amin;
        self
    }

    /// Sets the exponent.
    #[must_use]
    pub fn with_power(mut self, power: f64) -> Self {
        self.power = power;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `amin` or `power` is not finite and positive, or the STFT
    /// configuration is invalid.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if !(self.amin.is_finite() && self.amin > 0.0) {
            return Err(range_err(
                "amin",
                self.amin,
                "> 0",
                "finite",
                "amin must be strictly positive",
            ));
        }
        if !(self.power.is_finite() && self.power > 0.0) {
            return Err(range_err(
                "power",
                self.power,
                "> 0",
                "finite",
                "power must be strictly positive",
            ));
        }
        self.stft.validate()
    }
}

/// Configuration of the frame-wise spectral contrast, mirroring
/// `librosa.feature.spectral_contrast`.
///
/// ## Purpose
///
/// Spectral contrast is the per-band difference between spectral peaks and valleys
/// (Jiang et al. 2002). The spectrum is divided into a band below `fmin` and
/// `n_bands` octave bands from `fmin` upwards (the last one extended to Nyquist);
/// in each band the mean of the top and bottom `quantile` fraction of magnitudes
/// are compared.
///
/// ## Intended usage
///
/// Pass to [`AudioFeatures::spectral_contrast_frames`]. [`Default`] is librosa's
/// default (`fmin = 200`, `n_bands = 6`, `quantile = 0.02`, dB contrast).
///
/// ## Invariants
///
/// Checked by [`SpectralContrastConfig::validate`]: `fmin` finite and positive,
/// `0 < quantile < 1`, and every band edge `fmin * 2^k` (`k < n_bands`) below
/// Nyquist.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::features::SpectralContrastConfig;
///
/// let cfg = SpectralContrastConfig::new().with_n_bands(nzu!(4)).with_linear(true);
/// assert!(cfg.validate(22050.0).is_ok());
/// assert!(SpectralContrastConfig::new().with_fmin(4000.0).validate(22050.0).is_err());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralContrastConfig {
    /// STFT of the magnitude spectrogram.
    pub stft: StftConfig,
    /// Upper edge of the lowest band and lower edge of the first octave band, in Hz.
    /// Positive; `fmin * 2^(n_bands - 1)` must be below Nyquist.
    pub fmin: f64,
    /// Number of octave bands above `fmin`; the output has `n_bands + 1` rows.
    pub n_bands: NonZeroUsize,
    /// Fraction of each band's bins averaged into its peak and valley, in `(0, 1)`.
    /// At least one bin is always used.
    pub quantile: f64,
    /// If `true`, return `peak - valley` in linear magnitude; if `false` (default),
    /// `power_to_db(peak) - power_to_db(valley)`.
    pub linear: bool,
}

impl Default for SpectralContrastConfig {
    fn default() -> Self {
        Self {
            stft: StftConfig::default(),
            fmin: 200.0,
            n_bands: crate::nzu!(6),
            quantile: 0.02,
            linear: false,
        }
    }
}

impl SpectralContrastConfig {
    /// Returns the default configuration (same as [`Default`], which equals
    /// librosa 1.0's defaults).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// librosa 1.0's `feature.spectral_contrast` defaults. Currently the same as [`Default`];
    /// use this constructor where agreement with librosa is the requirement.
    #[must_use]
    pub fn librosa() -> Self {
        Self {
            stft: StftConfig::librosa(),
            ..Self::default()
        }
    }

    /// Sets the STFT configuration.
    #[must_use]
    pub fn with_stft(mut self, stft: StftConfig) -> Self {
        self.stft = stft;
        self
    }

    /// Sets `fmin`.
    #[must_use]
    pub fn with_fmin(mut self, fmin: f64) -> Self {
        self.fmin = fmin;
        self
    }

    /// Sets the number of octave bands.
    #[must_use]
    pub fn with_n_bands(mut self, n_bands: NonZeroUsize) -> Self {
        self.n_bands = n_bands;
        self
    }

    /// Sets the quantile.
    #[must_use]
    pub fn with_quantile(mut self, quantile: f64) -> Self {
        self.quantile = quantile;
        self
    }

    /// Sets linear (instead of dB) contrast.
    #[must_use]
    pub fn with_linear(mut self, linear: bool) -> Self {
        self.linear = linear;
        self
    }

    /// Checks the feature parameters (not the STFT) for a signal sampled at
    /// `sample_rate` Hz.
    fn validate_params(&self, sample_rate: f64) -> AudioSampleResult<()> {
        validate_sample_rate(sample_rate)?;
        if !(self.fmin.is_finite() && self.fmin > 0.0) {
            return Err(range_err(
                "fmin",
                self.fmin,
                "> 0",
                "finite",
                "fmin must be a positive number",
            ));
        }
        if !(self.quantile > 0.0 && self.quantile < 1.0) {
            return Err(range_err(
                "quantile",
                self.quantile,
                "0 (exclusive)",
                "1 (exclusive)",
                "quantile must lie in the range (0, 1)",
            ));
        }
        let top = self.fmin * 2f64.powi((self.n_bands.get() - 1).min(i32::MAX as usize) as i32);
        if top.is_nan() || top >= 0.5 * sample_rate {
            return Err(config_err(
                "spectral_contrast",
                format!(
                    "frequency band edge {top} Hz exceeds Nyquist ({} Hz); reduce either fmin or n_bands",
                    0.5 * sample_rate
                ),
            ));
        }
        Ok(())
    }

    /// Checks the configuration for a signal sampled at `sample_rate` Hz.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `fmin` is not finite and positive, `quantile` is not in
    /// `(0, 1)`, a band edge reaches Nyquist, `sample_rate` is not positive, or the
    /// STFT configuration is invalid.
    pub fn validate(&self, sample_rate: f64) -> AudioSampleResult<()> {
        self.validate_params(sample_rate)?;
        self.stft.validate()
    }
}

/// Configuration of the per-frame polynomial fit, mirroring
/// `librosa.feature.poly_features`.
///
/// ## Purpose
///
/// Fits a polynomial of degree `order` to each magnitude-spectrum frame as a
/// function of frequency (least squares, like `numpy.polyfit`).
///
/// ## Intended usage
///
/// Pass to [`AudioFeatures::poly_features`]. [`Default`] is librosa's default
/// (`order = 1`: slope and intercept).
///
/// ## Invariants
///
/// Checked at use: `order + 1` must not exceed the number of frequency bins.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::PolyFeaturesConfig;
///
/// assert_eq!(PolyFeaturesConfig::new().with_order(2).order, 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PolyFeaturesConfig {
    /// STFT of the magnitude spectrogram.
    pub stft: StftConfig,
    /// Polynomial degree; the output has `order + 1` coefficient rows, highest
    /// degree first. `0` fits a constant (the frame mean).
    pub order: usize,
}

impl Default for PolyFeaturesConfig {
    fn default() -> Self {
        Self {
            stft: StftConfig::default(),
            order: 1,
        }
    }
}

impl PolyFeaturesConfig {
    /// Returns the default configuration (same as [`Default`], which equals
    /// librosa 1.0's defaults).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// librosa 1.0's `feature.poly_features` defaults. Currently the same as [`Default`];
    /// use this constructor where agreement with librosa is the requirement.
    #[must_use]
    pub fn librosa() -> Self {
        Self {
            stft: StftConfig::librosa(),
            ..Self::default()
        }
    }

    /// Sets the STFT configuration.
    #[must_use]
    pub fn with_stft(mut self, stft: StftConfig) -> Self {
        self.stft = stft;
        self
    }

    /// Sets the polynomial order.
    #[must_use]
    pub fn with_order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `order + 1` exceeds the number of STFT bins, or the STFT
    /// configuration is invalid.
    pub fn validate(&self) -> AudioSampleResult<()> {
        self.stft.validate()?;
        if self.order >= self.stft.n_bins() {
            return Err(config_err(
                "poly_features",
                format!(
                    "order {} needs at least {} frequency bins, the STFT has {}",
                    self.order,
                    self.order + 1,
                    self.stft.n_bins()
                ),
            ));
        }
        Ok(())
    }
}

// =============================================================================
// Spectral shape descriptors: functions on a spectrogram
// =============================================================================

/// Computes the spectral centroid of every frame of a magnitude spectrogram,
/// equivalent to `librosa.feature.spectral_centroid(S=s, freq=freq)`.
///
/// Each frame is normalised to unit sum and the centroid is
/// `sum_k f[k] * S[k] / sum_k S[k]`. Frames whose sum is below the smallest
/// positive normal value of the element type are left unnormalised (librosa
/// `util.normalize`), so silent frames give 0.
///
/// # Arguments
///
/// * `s` - `(channels, bins, frames)` magnitude (or power) spectrogram of element
///   type `F` (`f32` or `f64`), finite and non-negative, e.g. from
///   [`AudioSpectrum::spectrogram_matrix`](crate::operations::spectrum::AudioSpectrum::spectrogram_matrix)
///   with power 1.
/// * `freq` - Centre frequency of each bin in Hz (length `bins`), e.g.
///   [`fft_frequencies(n_fft, sr)`](crate::utils::audio_math::fft_frequencies);
///   rounded to `F` once.
///
/// # Returns
///
/// A `(channels, 1, frames)` array of centroids in Hz, of element type `F`.
///
/// # Errors
///
/// [`ParameterError`] if `s` is empty, has negative or non-finite values, or
/// `freq.len()` differs from the bin count.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::spectral_centroid_from_spectrogram;
/// use ndarray::Array3;
///
/// let mut s = Array3::<f64>::zeros((1, 5, 2));
/// s[[0, 1, 0]] = 1.0;
/// s[[0, 3, 0]] = 1.0;
/// let c = spectral_centroid_from_spectrogram(s.view(), &[0.0, 100.0, 200.0, 300.0, 400.0]).unwrap();
/// assert_eq!(c[[0, 0, 0]], 200.0);
/// assert_eq!(c[[0, 0, 1]], 0.0); // silent frame
/// ```
///
/// # References
///
/// Klapuri and Davy, *Signal Processing Methods for Music Transcription*, 2007;
/// librosa `feature.spectral_centroid`.
pub fn spectral_centroid_from_spectrogram<F: AudioFloat>(
    s: ArrayView3<'_, F>,
    freq: &[f64],
) -> AudioSampleResult<Array3<F>> {
    check_spectrogram("spectral_centroid", &s)?;
    check_freq("spectral_centroid", freq, s.dim().1)?;
    Ok(centroid_unchecked(&s, freq))
}

/// `sum(S, axis=bins)` of a tile (rows contiguous, see [`for_each_row_tile`])
/// with `util.normalize`'s rule: sums below the smallest positive float are
/// replaced by 1. Each frame is summed over the bins in ascending order.
fn l1_lengths_tile<F: AudioFloat>(tile: &ArrayView2<'_, F>, len: &mut [F]) {
    len.fill(F::zero());
    for k in 0..tile.nrows() {
        for (l, &v) in len.iter_mut().zip(row(tile, k)) {
            *l += v.abs();
        }
    }
    for l in len.iter_mut() {
        if *l < F::min_positive_value() {
            *l = F::one();
        }
    }
}

/// `freq` rounded to `F`.
fn freq_as<F: AudioFloat>(freq: &[f64]) -> Vec<F> {
    freq.iter().map(|&f| F::of(f)).collect()
}

fn centroid_unchecked<F: AudioFloat>(s: &ArrayView3<'_, F>, freq: &[f64]) -> Array3<F> {
    let freq = freq_as::<F>(freq);
    let (n_ch, _, n_frames) = s.dim();
    let mut out = Array3::<F>::zeros((n_ch, 1, n_frames));
    let mut len = vec![F::zero(); n_frames];
    for c in 0..n_ch {
        let o = out_row(&mut out, c);
        for_each_row_tile(s.index_axis(Axis(0), c), |t0, tile| {
            let w = tile.ncols();
            let len = &mut len[..w];
            l1_lengths_tile(&tile, len);
            let o = &mut o[t0..t0 + w];
            for (k, &fk) in freq.iter().enumerate() {
                for ((o, &v), &l) in o.iter_mut().zip(row(&tile, k)).zip(len.iter()) {
                    *o += fk * (v / l);
                }
            }
        });
    }
    out
}

/// [`spectral_centroid_from_spectrogram`] with a frequency for every bin of every
/// frame, equivalent to `librosa.feature.spectral_centroid(S=s, freq=freq)` with a
/// time-varying `freq` (reassigned frequencies, for example).
///
/// # Arguments
///
/// * `s` - `(channels, bins, frames)` finite, non-negative spectrogram of element type
///   `F` (`f32` or `f64`).
/// * `freq` - Frequencies in Hz, `(channels, bins, frames)` or `(1, bins, frames)` for
///   one set shared by all channels. A NaN gives a NaN centroid, as in librosa.
///
/// # Returns
///
/// A `(channels, 1, frames)` array of centroids in Hz, of element type `F`.
///
/// # Errors
///
/// [`ParameterError`] if the spectrogram is invalid or `freq` has another shape.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::spectral_centroid_from_spectrogram_with_freqs;
/// use ndarray::Array3;
///
/// let s = Array3::<f64>::ones((1, 2, 2));
/// let f = Array3::from_shape_vec((1, 2, 2), vec![100.0, 200.0, 300.0, 400.0]).unwrap();
/// let c = spectral_centroid_from_spectrogram_with_freqs(s.view(), f.view()).unwrap();
/// assert_eq!(c[[0, 0, 0]], 200.0);
/// assert_eq!(c[[0, 0, 1]], 300.0);
/// ```
pub fn spectral_centroid_from_spectrogram_with_freqs<F: AudioFloat>(
    s: ArrayView3<'_, F>,
    freq: ArrayView3<'_, F>,
) -> AudioSampleResult<Array3<F>> {
    check_spectrogram("spectral_centroid", &s)?;
    check_freq_matrix("spectral_centroid", &freq, &s)?;
    Ok(centroid_varying(&s, &freq))
}

fn check_freq_matrix<F: AudioFloat>(
    what: &str,
    freq: &ArrayView3<'_, F>,
    s: &ArrayView3<'_, F>,
) -> AudioSampleResult<()> {
    let (n_ch, n_bins, n_frames) = s.dim();
    if freq.dim() != (n_ch, n_bins, n_frames) && freq.dim() != (1, n_bins, n_frames) {
        return Err(config_err(
            what,
            format!(
                "freq has shape {:?}, expected ({n_ch}, {n_bins}, {n_frames}) or (1, {n_bins}, {n_frames})",
                freq.dim()
            ),
        ));
    }
    Ok(())
}

/// The frequencies of channel `c` in bin `k` for frames `t0..t0 + len`, in a
/// standard-layout `freq`.
fn freq_row<'a, F: AudioFloat>(
    freq: &'a ndarray::ArrayBase<ndarray::CowRepr<'_, F>, ndarray::Ix3>,
    c: usize,
    k: usize,
    t0: usize,
    len: usize,
) -> &'a [F] {
    let (n_ch, n_bins, n_frames) = freq.dim();
    let start = (if n_ch == 1 { 0 } else { c } * n_bins + k) * n_frames + t0;
    &freq.as_slice().expect("standard layout")[start..start + len]
}

fn centroid_varying<F: AudioFloat>(s: &ArrayView3<'_, F>, freq: &ArrayView3<'_, F>) -> Array3<F> {
    let freq = freq.as_standard_layout();
    let (n_ch, n_bins, n_frames) = s.dim();
    let mut out = Array3::<F>::zeros((n_ch, 1, n_frames));
    let mut len = vec![F::zero(); n_frames];
    for c in 0..n_ch {
        let o = out_row(&mut out, c);
        for_each_row_tile(s.index_axis(Axis(0), c), |t0, tile| {
            let w = tile.ncols();
            let len = &mut len[..w];
            l1_lengths_tile(&tile, len);
            let o = &mut o[t0..t0 + w];
            for k in 0..n_bins {
                let fk = freq_row(&freq, c, k, t0, w);
                for (((o, &v), &l), &f) in o.iter_mut().zip(row(&tile, k)).zip(len.iter()).zip(fk) {
                    *o += f * (v / l);
                }
            }
        });
    }
    out
}

/// Computes the order-`p` spectral bandwidth of every frame, equivalent to
/// `librosa.feature.spectral_bandwidth(S=s, freq=freq, centroid=centroid, p=p, norm=norm)`.
///
/// `bandwidth = (sum_k S[k] |f[k] - centroid|^p)^(1/p)`, with `S` normalised to unit
/// sum per frame when `config.norm` is set.
///
/// # Arguments
///
/// * `s` - `(channels, bins, frames)` finite, non-negative spectrogram of element
///   type `F` (`f32` or `f64`).
/// * `freq` - Bin centre frequencies in Hz (length `bins`).
/// * `centroid` - Optional precomputed `(channels, 1, frames)` centroid; `None`
///   computes it with [`spectral_centroid_from_spectrogram`].
/// * `config` - Only `p` and `norm` are used (`config.stft` is ignored).
///
/// # Returns
///
/// A `(channels, 1, frames)` array in Hz, of element type `F`.
///
/// # Errors
///
/// [`ParameterError`] if the spectrogram is invalid, `freq` or `centroid` has the
/// wrong shape, or `p` is zero or not finite.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::{
///     SpectralBandwidthConfig, spectral_bandwidth_from_spectrogram,
/// };
/// use ndarray::Array3;
///
/// let mut s = Array3::<f64>::zeros((1, 3, 1));
/// s[[0, 0, 0]] = 1.0;
/// s[[0, 2, 0]] = 1.0;
/// let bw = spectral_bandwidth_from_spectrogram(
///     s.view(), &[0.0, 50.0, 100.0], None, &SpectralBandwidthConfig::default(),
/// ).unwrap();
/// assert!((bw[[0, 0, 0]] - 50.0).abs() < 1e-12);
/// ```
pub fn spectral_bandwidth_from_spectrogram<F: AudioFloat>(
    s: ArrayView3<'_, F>,
    freq: &[f64],
    centroid: Option<ArrayView3<'_, F>>,
    config: &SpectralBandwidthConfig,
) -> AudioSampleResult<Array3<F>> {
    if !(config.p.is_finite() && config.p != 0.0) {
        return Err(param_err("p", "p must be finite and non-zero"));
    }
    check_spectrogram("spectral_bandwidth", &s)?;
    let (n_ch, n_bins, n_frames) = s.dim();
    check_freq("spectral_bandwidth", freq, n_bins)?;
    let computed;
    let centroid = match centroid {
        Some(c) => {
            if c.dim() != (n_ch, 1, n_frames) {
                return Err(config_err(
                    "spectral_bandwidth",
                    format!(
                        "centroid has shape {:?}, expected ({n_ch}, 1, {n_frames})",
                        c.dim()
                    ),
                ));
            }
            c
        }
        None => {
            computed = centroid_unchecked(&s, freq);
            computed.view()
        }
    };
    let p = config.p;
    let freq = freq_as::<F>(freq);
    let mut out = Array3::<F>::zeros((n_ch, 1, n_frames));
    let mut len = vec![F::one(); n_frames];
    let mut cent = vec![F::zero(); n_frames];
    for c in 0..n_ch {
        let o = out_row(&mut out, c);
        for_each_row_tile(s.index_axis(Axis(0), c), |t0, tile| {
            let w = tile.ncols();
            let len = &mut len[..w];
            if config.norm {
                l1_lengths_tile(&tile, len);
            }
            let cent = &mut cent[..w];
            for (j, ce) in cent.iter_mut().enumerate() {
                *ce = centroid[[c, 0, t0 + j]];
            }
            let o = &mut o[t0..t0 + w];
            for (k, &fk) in freq.iter().enumerate() {
                let r = row(&tile, k);
                let terms = o.iter_mut().zip(r).zip(cent.iter()).zip(len.iter());
                // The same operations as `w * pow_np(|centroid - f|, p)`, with
                // the exponent's case chosen once per row.
                macro_rules! acc {
                    ($pow:expr) => {
                        for (((o, &v), &ce), &l) in terms {
                            let dev = (ce - fk).abs();
                            let wv = if config.norm { v / l } else { v };
                            *o += wv * $pow(dev);
                        }
                    };
                }
                if p == 1.0 {
                    acc!(|d: F| d)
                } else if p == 2.0 {
                    acc!(|d: F| d * d)
                } else if p == 0.5 {
                    acc!(|d: F| d.sqrt())
                } else {
                    acc!(|d: F| d.powf(F::of(p)))
                }
            }
        });
    }
    let inv = 1.0 / p;
    out.mapv_inplace(|v| pow_np(v, inv));
    Ok(out)
}

/// [`spectral_bandwidth_from_spectrogram`] with a frequency for every bin of every
/// frame, equivalent to `librosa.feature.spectral_bandwidth(S=s, freq=freq, ...)` with
/// a time-varying `freq`.
///
/// # Arguments
///
/// * `s` - `(channels, bins, frames)` finite, non-negative spectrogram of element type
///   `F` (`f32` or `f64`).
/// * `freq` - Frequencies in Hz, `(channels, bins, frames)` or `(1, bins, frames)`.
/// * `centroid` - Optional precomputed `(channels, 1, frames)` centroid; `None`
///   computes it with [`spectral_centroid_from_spectrogram_with_freqs`].
/// * `config` - Only `p` and `norm` are used.
///
/// # Returns
///
/// A `(channels, 1, frames)` array in Hz, of element type `F`.
///
/// # Errors
///
/// [`ParameterError`] if the spectrogram is invalid, `freq` or `centroid` has the wrong
/// shape, or `p` is zero or not finite.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::{
///     SpectralBandwidthConfig, spectral_bandwidth_from_spectrogram_with_freqs,
/// };
/// use ndarray::Array3;
///
/// let s = Array3::<f64>::ones((1, 2, 1));
/// let f = Array3::from_shape_vec((1, 2, 1), vec![0.0, 100.0]).unwrap();
/// let bw = spectral_bandwidth_from_spectrogram_with_freqs(
///     s.view(), f.view(), None, &SpectralBandwidthConfig::default(),
/// ).unwrap();
/// assert!((bw[[0, 0, 0]] - 50.0).abs() < 1e-12);
/// ```
pub fn spectral_bandwidth_from_spectrogram_with_freqs<F: AudioFloat>(
    s: ArrayView3<'_, F>,
    freq: ArrayView3<'_, F>,
    centroid: Option<ArrayView3<'_, F>>,
    config: &SpectralBandwidthConfig,
) -> AudioSampleResult<Array3<F>> {
    if !(config.p.is_finite() && config.p != 0.0) {
        return Err(param_err("p", "p must be finite and non-zero"));
    }
    check_spectrogram("spectral_bandwidth", &s)?;
    let (n_ch, n_bins, n_frames) = s.dim();
    check_freq_matrix("spectral_bandwidth", &freq, &s)?;
    let computed;
    let centroid = match centroid {
        Some(c) => {
            if c.dim() != (n_ch, 1, n_frames) {
                return Err(config_err(
                    "spectral_bandwidth",
                    format!(
                        "centroid has shape {:?}, expected ({n_ch}, 1, {n_frames})",
                        c.dim()
                    ),
                ));
            }
            c
        }
        None => {
            computed = centroid_varying(&s, &freq);
            computed.view()
        }
    };
    let p = config.p;
    let freq = freq.as_standard_layout();
    let mut out = Array3::<F>::zeros((n_ch, 1, n_frames));
    let mut len = vec![F::one(); n_frames];
    let mut cent = vec![F::zero(); n_frames];
    for c in 0..n_ch {
        let o = out_row(&mut out, c);
        for_each_row_tile(s.index_axis(Axis(0), c), |t0, tile| {
            let w = tile.ncols();
            let len = &mut len[..w];
            if config.norm {
                l1_lengths_tile(&tile, len);
            }
            let cent = &mut cent[..w];
            for (j, ce) in cent.iter_mut().enumerate() {
                *ce = centroid[[c, 0, t0 + j]];
            }
            let o = &mut o[t0..t0 + w];
            for k in 0..n_bins {
                let fk = freq_row(&freq, c, k, t0, w);
                let terms = o
                    .iter_mut()
                    .zip(row(&tile, k))
                    .zip(cent.iter())
                    .zip(len.iter())
                    .zip(fk);
                // The same operations as `w * pow_np(|centroid - f|, p)`, with
                // the exponent's case chosen once per row.
                macro_rules! acc {
                    ($pow:expr) => {
                        for ((((o, &v), &ce), &l), &f) in terms {
                            let dev = (f - ce).abs();
                            let wv = if config.norm { v / l } else { v };
                            *o += wv * $pow(dev);
                        }
                    };
                }
                if p == 1.0 {
                    acc!(|d: F| d)
                } else if p == 2.0 {
                    acc!(|d: F| d * d)
                } else if p == 0.5 {
                    acc!(|d: F| d.sqrt())
                } else {
                    acc!(|d: F| d.powf(F::of(p)))
                }
            }
        });
    }
    let inv = 1.0 / p;
    out.mapv_inplace(|v| pow_np(v, inv));
    Ok(out)
}

/// Computes the spectral rolloff frequency of every frame, equivalent to
/// `librosa.feature.spectral_rolloff(S=s, freq=freq, roll_percent=...)`.
///
/// The rolloff is the smallest `freq[k]` among bins whose cumulative magnitude
/// `sum_{j<=k} S[j]` reaches `roll_percent` of the frame total. Silent frames give
/// `freq[0]`.
///
/// # Arguments
///
/// * `s` - `(channels, bins, frames)` finite, non-negative spectrogram of element
///   type `F` (`f32` or `f64`).
/// * `freq` - Bin centre frequencies in Hz (length `bins`).
/// * `config` - Only `roll_percent` is used (`config.stft` is ignored).
///
/// # Returns
///
/// A `(channels, 1, frames)` array in Hz, of element type `F`.
///
/// # Errors
///
/// [`ParameterError`] if `roll_percent` is not in `(0, 1)`, the spectrogram is
/// invalid, or `freq` has the wrong length.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::{
///     SpectralRolloffConfig, spectral_rolloff_from_spectrogram,
/// };
/// use ndarray::Array3;
///
/// let s = Array3::from_shape_vec((1, 4, 1), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
/// let r = spectral_rolloff_from_spectrogram(
///     s.view(), &[0.0, 10.0, 20.0, 30.0], &SpectralRolloffConfig::new().with_roll_percent(0.5),
/// ).unwrap();
/// assert_eq!(r[[0, 0, 0]], 10.0);
/// ```
pub fn spectral_rolloff_from_spectrogram<F: AudioFloat>(
    s: ArrayView3<'_, F>,
    freq: &[f64],
    config: &SpectralRolloffConfig,
) -> AudioSampleResult<Array3<F>> {
    let rp = config.roll_percent;
    if !(rp > 0.0 && rp < 1.0) {
        return Err(range_err(
            "roll_percent",
            rp,
            "0 (exclusive)",
            "1 (exclusive)",
            "roll_percent must lie in the range (0, 1)",
        ));
    }
    check_spectrogram("spectral_rolloff", &s)?;
    let (n_ch, n_bins, n_frames) = s.dim();
    check_freq("spectral_rolloff", freq, n_bins)?;
    let freq = freq_as::<F>(freq);
    let rp = F::of(rp);
    let mut out = Array3::<F>::zeros((n_ch, 1, n_frames));
    let mut acc = vec![F::zero(); n_frames];
    let mut threshold = vec![F::zero(); n_frames];
    for c in 0..n_ch {
        let o = out_row(&mut out, c);
        for_each_row_tile(s.index_axis(Axis(0), c), |t0, tile| {
            let w = tile.ncols();
            let (acc, threshold) = (&mut acc[..w], &mut threshold[..w]);
            // Pass 1: the frame totals (cumulative sums in ascending bin order).
            acc.fill(F::zero());
            for k in 0..n_bins {
                for (a, &v) in acc.iter_mut().zip(row(&tile, k)) {
                    *a += v;
                }
            }
            for (th, &a) in threshold.iter_mut().zip(acc.iter()) {
                *th = rp * a;
            }
            // Pass 2: the same cumulative sums again; the result is the minimum
            // of the frequencies (in ascending bin order, starting from NaN)
            // whose cumulative sum reaches the threshold.
            acc.fill(F::zero());
            let o = &mut o[t0..t0 + w];
            o.fill(F::nan());
            for (k, &fk) in freq.iter().enumerate() {
                for (((a, o), &v), &th) in acc
                    .iter_mut()
                    .zip(o.iter_mut())
                    .zip(row(&tile, k))
                    .zip(threshold.iter())
                {
                    *a += v;
                    if *a >= th {
                        *o = o.min(fk);
                    }
                }
            }
        });
    }
    Ok(out)
}

/// [`spectral_rolloff_from_spectrogram`] with a frequency for every bin of every frame,
/// equivalent to `librosa.feature.spectral_rolloff(S=s, freq=freq, roll_percent=...)`
/// with a time-varying `freq`.
///
/// The rolloff is the smallest frequency among bins whose cumulative magnitude reaches
/// `roll_percent` of the frame total; NaN frequencies are skipped, as `numpy.nanmin`
/// skips them.
///
/// # Arguments
///
/// * `s` - `(channels, bins, frames)` finite, non-negative spectrogram of element type
///   `F` (`f32` or `f64`).
/// * `freq` - Frequencies in Hz, `(channels, bins, frames)` or `(1, bins, frames)`.
/// * `config` - Only `roll_percent` is used.
///
/// # Returns
///
/// A `(channels, 1, frames)` array in Hz, of element type `F`.
///
/// # Errors
///
/// [`ParameterError`] if `roll_percent` is not in `(0, 1)`, the spectrogram is invalid,
/// or `freq` has another shape.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::{
///     SpectralRolloffConfig, spectral_rolloff_from_spectrogram_with_freqs,
/// };
/// use ndarray::Array3;
///
/// let s = Array3::from_shape_vec((1, 4, 1), vec![1.0, 1.0, 1.0, 1.0]).unwrap();
/// let f = Array3::from_shape_vec((1, 4, 1), vec![0.0, 15.0, 20.0, 30.0]).unwrap();
/// let r = spectral_rolloff_from_spectrogram_with_freqs(
///     s.view(), f.view(), &SpectralRolloffConfig::new().with_roll_percent(0.5),
/// ).unwrap();
/// assert_eq!(r[[0, 0, 0]], 15.0);
/// ```
pub fn spectral_rolloff_from_spectrogram_with_freqs<F: AudioFloat>(
    s: ArrayView3<'_, F>,
    freq: ArrayView3<'_, F>,
    config: &SpectralRolloffConfig,
) -> AudioSampleResult<Array3<F>> {
    let rp = config.roll_percent;
    if !(rp > 0.0 && rp < 1.0) {
        return Err(range_err(
            "roll_percent",
            rp,
            "0 (exclusive)",
            "1 (exclusive)",
            "roll_percent must lie in the range (0, 1)",
        ));
    }
    check_spectrogram("spectral_rolloff", &s)?;
    let (n_ch, n_bins, n_frames) = s.dim();
    check_freq_matrix("spectral_rolloff", &freq, &s)?;
    let freq = freq.as_standard_layout();
    let rp = F::of(rp);
    let mut out = Array3::<F>::zeros((n_ch, 1, n_frames));
    let mut acc = vec![F::zero(); n_frames];
    let mut threshold = vec![F::zero(); n_frames];
    for c in 0..n_ch {
        let o = out_row(&mut out, c);
        for_each_row_tile(s.index_axis(Axis(0), c), |t0, tile| {
            let w = tile.ncols();
            let (acc, threshold) = (&mut acc[..w], &mut threshold[..w]);
            acc.fill(F::zero());
            for k in 0..n_bins {
                for (a, &v) in acc.iter_mut().zip(row(&tile, k)) {
                    *a += v;
                }
            }
            for (th, &a) in threshold.iter_mut().zip(acc.iter()) {
                *th = rp * a;
            }
            acc.fill(F::zero());
            let o = &mut o[t0..t0 + w];
            o.fill(F::nan());
            for k in 0..n_bins {
                let fk = freq_row(&freq, c, k, t0, w);
                for (((a, o), &v), (&th, &f)) in acc
                    .iter_mut()
                    .zip(o.iter_mut())
                    .zip(row(&tile, k))
                    .zip(threshold.iter().zip(fk))
                {
                    *a += v;
                    if *a >= th {
                        *o = o.min(f);
                    }
                }
            }
        });
    }
    Ok(out)
}

/// Computes the spectral flatness of every frame, equivalent to
/// `librosa.feature.spectral_flatness(S=s, amin=..., power=...)`.
///
/// With `P = max(amin, S^power)`, flatness is `exp(mean(ln P)) / mean(P)` per frame.
///
/// # Arguments
///
/// * `s` - `(channels, bins, frames)` finite, non-negative **magnitude**
///   spectrogram of element type `F` (`f32` or `f64`); the exponent is applied
///   here.
/// * `config` - Only `amin` and `power` are used (`config.stft` is ignored).
///
/// # Returns
///
/// A `(channels, 1, frames)` array with values in `(0, 1]`, of element type `F`.
///
/// # Errors
///
/// [`ParameterError`] if `amin` or `power` is not finite and positive, or the
/// spectrogram is invalid.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::{
///     SpectralFlatnessConfig, spectral_flatness_from_spectrogram,
/// };
/// use ndarray::Array3;
///
/// let flat = Array3::<f64>::ones((1, 8, 1));
/// let f = spectral_flatness_from_spectrogram(flat.view(), &SpectralFlatnessConfig::default()).unwrap();
/// assert!((f[[0, 0, 0]] - 1.0).abs() < 1e-12);
/// ```
///
/// # References
///
/// S. Dubnov, "Generalization of spectral flatness measure for non-gaussian linear
/// processes," *IEEE Signal Processing Letters* 11(8), 2004.
pub fn spectral_flatness_from_spectrogram<F: AudioFloat>(
    s: ArrayView3<'_, F>,
    config: &SpectralFlatnessConfig,
) -> AudioSampleResult<Array3<F>> {
    SpectralFlatnessConfig {
        stft: StftConfig::default(),
        ..config.clone()
    }
    .validate()?;
    check_spectrogram("spectral_flatness", &s)?;
    let (n_ch, n_bins, n_frames) = s.dim();
    let mut out = Array3::<F>::zeros((n_ch, 1, n_frames));
    let n = F::of_usize(n_bins);
    let amin = F::of(config.amin);
    let power = config.power;
    let mut log_sum = vec![F::zero(); n_frames];
    let mut sum = vec![F::zero(); n_frames];
    let mut p_row = vec![F::zero(); n_frames];
    for c in 0..n_ch {
        let sc = s.index_axis(Axis(0), c);
        let o = out_row(&mut out, c);
        if n_frames > 1 && sc.strides()[0] == 1 {
            flatness_columns(sc, o, amin, power, n);
            continue;
        }
        for_each_row_tile(sc, |t0, tile| {
            let w = tile.ncols();
            let (log_sum, sum, p_row) = (&mut log_sum[..w], &mut sum[..w], &mut p_row[..w]);
            log_sum.fill(F::zero());
            sum.fill(F::zero());
            for k in 0..n_bins {
                let r = row(&tile, k);
                if power == 2.0 {
                    for (p, &v) in p_row.iter_mut().zip(r) {
                        *p = amin.max(v * v);
                    }
                } else {
                    for (p, &v) in p_row.iter_mut().zip(r) {
                        *p = amin.max(pow_np(v, power));
                    }
                }
                for (su, &p) in sum.iter_mut().zip(p_row.iter()) {
                    *su += p;
                }
                ln_in_place(p_row);
                for (ls, &lp) in log_sum.iter_mut().zip(p_row.iter()) {
                    *ls += lp;
                }
            }
            for ((o, &ls), &su) in o[t0..t0 + w].iter_mut().zip(log_sum.iter()).zip(sum.iter()) {
                *o = (ls / n).exp() / (su / n);
            }
        });
    }
    Ok(out)
}

/// [`spectral_flatness_from_spectrogram`] for one channel whose frames are
/// contiguous (Fortran order): the values of 8 frames at a time are mapped
/// and their logarithms taken over each contiguous column, then the 8
/// frames are summed over the bins in lockstep (ascending bin order, the
/// same operations as the row-wise path).
fn flatness_columns<F: AudioFloat>(sc: ArrayView2<'_, F>, o: &mut [F], amin: F, power: f64, n: F) {
    const G: usize = 8;
    let (n_bins, n_frames) = sc.dim();
    let mut p = vec![F::zero(); G * n_bins];
    let mut t = 0;
    while t < n_frames {
        let g = G.min(n_frames - t);
        for j in 0..g {
            let column = sc.column(t + j);
            let column = column.to_slice().expect("unit row stride");
            let pj = &mut p[j * n_bins..(j + 1) * n_bins];
            if power == 2.0 {
                for (p, &v) in pj.iter_mut().zip(column) {
                    *p = amin.max(v * v);
                }
            } else {
                for (p, &v) in pj.iter_mut().zip(column) {
                    *p = amin.max(pow_np(v, power));
                }
            }
        }
        let mut su = [F::zero(); G];
        lockstep_sums(&p, n_bins, g, &mut su);
        ln_in_place(&mut p[..g * n_bins]);
        let mut ls = [F::zero(); G];
        lockstep_sums(&p, n_bins, g, &mut ls);
        for j in 0..g {
            o[t + j] = (ls[j] / n).exp() / (su[j] / n);
        }
        t += g;
    }
}

/// `acc[j] += cols[j][k]` for `k` ascending, for the `g <= 8` columns of
/// `n` values stored one after another in `p` (the columns advance in
/// lockstep, eight independent sums).
fn lockstep_sums<F: AudioFloat>(p: &[F], n: usize, g: usize, acc: &mut [F; 8]) {
    let cols: [&[F]; 8] = std::array::from_fn(|j| {
        let j = j.min(g - 1);
        &p[j * n..(j + 1) * n]
    });
    let mut a = [F::zero(); 8];
    // `k` indexes the eight columns at once.
    #[allow(clippy::needless_range_loop)]
    for k in 0..n {
        for j in 0..8 {
            a[j] += cols[j][k];
        }
    }
    acc[..g].copy_from_slice(&a[..g]);
}

/// Computes the spectral contrast of every frame, equivalent to
/// `librosa.feature.spectral_contrast(S=s, sr=sample_rate, freq=freq, ...)`.
///
/// Band `0` covers `[0, fmin]` and band `k >= 1` covers `[fmin 2^(k-1), fmin 2^k]`
/// (inclusive), each also taking the bin just below its lower edge; the top band
/// extends to the last bin, and every other band drops its highest selected bin
/// (librosa's band rules, reproduced exactly). In each band the bins are sorted
/// per frame; the valley is the mean of the lowest `max(1, rint(quantile * n))`
/// values and the peak the mean of the highest as many, where `n` is the number of
/// selected bins before the highest is dropped.
///
/// # Arguments
///
/// * `s` - `(channels, bins, frames)` magnitude spectrogram (finite, non-negative)
///   of element type `F` (`f32` or `f64`).
/// * `freq` - Bin centre frequencies in Hz (length `bins`); the band selection is
///   done in `f64`.
/// * `sample_rate` - Sampling rate in Hz (for the Nyquist check on the band edges).
/// * `config` - `fmin`, `n_bands`, `quantile`, `linear` (`config.stft` is ignored).
///
/// # Returns
///
/// A `(channels, n_bands + 1, frames)` array of element type `F`: `peak - valley` if `linear`, else
/// `power_to_db(peak) - power_to_db(valley)` with librosa's default dB settings
/// (`ref = 1`, `amin = 1e-10`, `top_db = 80` per channel).
///
/// # Errors
///
/// [`ParameterError`] if the parameters are invalid (see
/// [`SpectralContrastConfig::validate`]), the spectrogram is invalid, `freq` has the
/// wrong length, or a band contains no frequency bin.
///
/// # Differences from librosa
///
/// A band without any bin (too short an `n_fft` for `fmin`) is an error here;
/// librosa raises `IndexError` or returns `NaN` with a warning, depending on the band.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::{
///     SpectralContrastConfig, spectral_contrast_from_spectrogram,
/// };
/// use audio_samples::utils::audio_math::fft_frequencies;
/// use ndarray::Array3;
///
/// let s = Array3::<f64>::ones((1, 1025, 3));
/// let freqs = fft_frequencies(2048, 22050.0);
/// let c = spectral_contrast_from_spectrogram(
///     s.view(), &freqs, 22050.0, &SpectralContrastConfig::default(),
/// ).unwrap();
/// assert_eq!(c.dim(), (1, 7, 3));
/// assert!(c.iter().all(|&v| v.abs() < 1e-12)); // a flat spectrum has no contrast
/// ```
///
/// # References
///
/// D.-N. Jiang, L. Lu, H.-J. Zhang, J.-H. Tao and L.-H. Cai, "Music type
/// classification by spectral contrast feature," *Proc. ICME*, 2002.
pub fn spectral_contrast_from_spectrogram<F: AudioFloat>(
    s: ArrayView3<'_, F>,
    freq: &[f64],
    sample_rate: f64,
    config: &SpectralContrastConfig,
) -> AudioSampleResult<Array3<F>> {
    config.validate_params(sample_rate)?;
    check_spectrogram("spectral_contrast", &s)?;
    let (n_ch, n_bins, n_frames) = s.dim();
    check_freq("spectral_contrast", freq, n_bins)?;
    let n_bands = config.n_bands.get();

    // Band edges: [0, fmin, 2 fmin, ..., fmin 2^n_bands].
    let mut octa = vec![0.0; n_bands + 2];
    for (k, o) in octa.iter_mut().enumerate().skip(1) {
        *o = config.fmin * 2f64.powi((k - 1) as i32);
    }

    let mut valley = Array3::<F>::zeros((n_ch, n_bands + 1, n_frames));
    let mut peak = Array3::<F>::zeros((n_ch, n_bands + 1, n_frames));
    let mut bands: Vec<(Vec<usize>, usize)> = Vec::with_capacity(n_bands + 1);
    for k in 0..=n_bands {
        let (f_low, f_high) = (octa[k], octa[k + 1]);
        let mut band: Vec<bool> = freq.iter().map(|&f| f >= f_low && f <= f_high).collect();
        let first = band.iter().position(|&b| b);
        let last = band.iter().rposition(|&b| b);
        if k > 0 {
            // librosa: current_band[idx[0] - 1] = True (Python negative indexing).
            let i = first.ok_or_else(|| empty_band(k, f_low, f_high))?;
            band[(i + n_bins - 1) % n_bins] = true;
        }
        if k == n_bands {
            let i = last.ok_or_else(|| empty_band(k, f_low, f_high))?;
            for b in &mut band[i + 1..] {
                *b = true;
            }
        }
        let mut sel: Vec<usize> = (0..n_bins).filter(|&i| band[i]).collect();
        let count = sel.len();
        if k < n_bands {
            sel.pop();
        }
        if sel.is_empty() {
            return Err(empty_band(k, f_low, f_high));
        }
        let q = (config.quantile * count as f64).round_ties_even().max(1.0) as usize;
        let m = q.min(sel.len());
        bands.push((sel, m));
    }
    let (mut lo, mut hi, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for c in 0..n_ch {
        for_each_row_tile(s.index_axis(Axis(0), c), |t0, tile| {
            let w = tile.ncols();
            for (k, (sel, m)) in bands.iter().enumerate() {
                let m = *m;
                extreme_sums_rows(&tile, sel, m, &mut lo, &mut hi, &mut v);
                let inv_m = F::of_usize(m);
                for j in 0..w {
                    // Sums in ascending order from +0, as a sorted band.
                    let (mut l, mut h) = (F::zero(), F::zero());
                    for q in 0..m {
                        l += lo[q * w + j];
                        h += hi[q * w + j];
                    }
                    valley[[c, k, t0 + j]] = l / inv_m;
                    peak[[c, k, t0 + j]] = h / inv_m;
                }
            }
        });
    }
    if config.linear {
        Ok(peak - valley)
    } else {
        // librosa's spectral_contrast converts peak and valley with power_to_db's
        // defaults (80 dB floor); the floor is part of the feature's definition.
        let db = DbConfig::librosa_power();
        Ok(power_to_db(&peak, &db)? - power_to_db(&valley, &db)?)
    }
}

/// For every frame (column) of `tile` (rows contiguous), the `m` smallest
/// and the `m` largest values over the rows `sel`, each in ascending order:
/// `lo[q * w + j]` and `hi[q * w + j]` (`q < m`) for frame `j` of `w`.
///
/// Each new value is inserted with a chain of compare-exchanges (`min` /
/// `max`), the same for every frame, so the loops run across frames without
/// branches. The values are non-negative or `-0.0` and never NaN; the kept
/// values equal those of a full sort, except that `-0.0` and `+0.0` count as
/// equal, which does not change a sum that starts at `+0`.
fn extreme_sums_rows<F: AudioFloat>(
    tile: &ArrayView2<'_, F>,
    sel: &[usize],
    m: usize,
    lo: &mut Vec<F>,
    hi: &mut Vec<F>,
    v: &mut Vec<F>,
) {
    let w = tile.ncols();
    lo.clear();
    lo.resize(m * w, F::infinity());
    hi.clear();
    hi.resize(m * w, F::neg_infinity());
    v.resize(w, F::zero());
    let v = &mut v[..w];
    for &i in sel {
        let r = row(tile, i);
        // Insert into the m smallest (ascending), dropping the largest.
        v.copy_from_slice(r);
        for q in 0..m {
            exchange(&mut lo[q * w..(q + 1) * w], v);
        }
        // Insert into the m largest (ascending), dropping the smallest.
        v.copy_from_slice(r);
        for q in (0..m).rev() {
            exchange(v, &mut hi[q * w..(q + 1) * w]);
        }
    }
}

/// Compare-exchange of two equal-length slices: `small[j]` becomes the
/// smaller and `large[j]` the larger of the two values (no NaN). Not
/// inlined, so that the two slices are known not to overlap and the loop
/// vectorises without a run-time overlap check.
#[inline(never)]
fn exchange<F: AudioFloat>(small: &mut [F], large: &mut [F]) {
    for (s, l) in small.iter_mut().zip(large.iter_mut()) {
        let (a, b) = (*s, *l);
        let lt = b < a;
        *s = if lt { b } else { a };
        *l = if lt { a } else { b };
    }
}

fn empty_band(k: usize, f_low: f64, f_high: f64) -> AudioSampleError {
    config_err(
        "spectral_contrast",
        format!(
            "band {k} ({f_low} Hz to {f_high} Hz) contains no frequency bins; increase n_fft or fmin"
        ),
    )
}

/// Fits a degree-`order` polynomial to every spectrogram frame, equivalent to
/// `librosa.feature.poly_features(S=s, freq=freq, order=order)` (`numpy.polyfit`
/// per frame).
///
/// The Vandermonde matrix of `freq` (columns scaled to unit norm, as `numpy.polyfit`
/// does) is factorised once with Householder QR and every frame is solved in the
/// least-squares sense. The factorisation depends only on `freq` and is computed
/// in `f64`; it is rounded to the element type `F` once, and the per-frame solves
/// run in `F`.
///
/// # Arguments
///
/// * `s` - `(channels, bins, frames)` spectrogram (any finite values) of element
///   type `F` (`f32` or `f64`).
/// * `freq` - Bin centre frequencies in Hz (length `bins`).
/// * `config` - Only `order` is used (`config.stft` is ignored).
///
/// # Returns
///
/// A `(channels, order + 1, frames)` array of coefficients of element type `F`,
/// highest degree first (row 0 is the `freq^order` coefficient, the last row the
/// constant term).
///
/// # Errors
///
/// [`ParameterError`] if `freq` has the wrong length, `s` is empty or non-finite, or
/// the fit is rank-deficient (fewer distinct frequencies than coefficients).
///
/// # Differences from librosa
///
/// `numpy.polyfit` solves with an SVD and returns a minimum-norm solution with a
/// `RankWarning` for rank-deficient fits; this function returns an error instead.
/// For full-rank fits the solutions agree to round-off.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::{PolyFeaturesConfig, poly_features_from_spectrogram};
/// use ndarray::Array3;
///
/// let freq = [0.0, 1.0, 2.0, 3.0];
/// let s = Array3::from_shape_fn((1, 4, 1), |(_, k, _)| 2.0 * freq[k] + 1.0);
/// let p = poly_features_from_spectrogram(s.view(), &freq, &PolyFeaturesConfig::default()).unwrap();
/// assert!((p[[0, 0, 0]] - 2.0).abs() < 1e-12); // slope
/// assert!((p[[0, 1, 0]] - 1.0).abs() < 1e-12); // intercept
/// ```
pub fn poly_features_from_spectrogram<F: AudioFloat>(
    s: ArrayView3<'_, F>,
    freq: &[f64],
    config: &PolyFeaturesConfig,
) -> AudioSampleResult<Array3<F>> {
    let (n_ch, n_bins, n_frames) = s.dim();
    if n_ch == 0 || n_bins == 0 || n_frames == 0 {
        return Err(config_err(
            "poly_features",
            "the spectrogram must be non-empty",
        ));
    }
    check_freq("poly_features", freq, n_bins)?;
    if !all_finite(&s) || freq.iter().any(|v| !v.is_finite()) {
        return Err(param_err(
            "spectrogram",
            "poly_features needs finite spectrogram values and frequencies",
        ));
    }
    let m = poly_terms(config.order, n_bins)?;
    let fit = PolyFit::<F>::new(freq, m)?;
    let mut out = Array3::<F>::zeros((n_ch, m, n_frames));
    let mut y = vec![F::zero(); n_bins];
    for c in 0..n_ch {
        for_each_col_tile(s.index_axis(Axis(0), c), |t0, tile| {
            for tt in 0..tile.ncols() {
                y.copy_from_slice(col(&tile, tt));
                fit.solve(&mut y, &mut out, c, t0 + tt);
            }
        });
    }
    Ok(out)
}

/// [`poly_features_from_spectrogram`] with a frequency for every bin of every frame,
/// equivalent to `librosa.feature.poly_features(S=s, freq=freq, order=...)` with a
/// time-varying `freq`. Each frame has its own fit.
///
/// # Arguments
///
/// * `s` - `(channels, bins, frames)` spectrogram (any finite values) of element type
///   `F` (`f32` or `f64`).
/// * `freq` - Frequencies in Hz, `(channels, bins, frames)` or `(1, bins, frames)`.
/// * `config` - Only `order` is used.
///
/// # Returns
///
/// A `(channels, order + 1, frames)` array of coefficients of element type `F`, highest
/// degree first.
///
/// # Errors
///
/// [`ParameterError`] if `freq` has another shape, `s` or `freq` is empty or not
/// finite, or the fit of a frame is rank-deficient.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::{
///     PolyFeaturesConfig, poly_features_from_spectrogram_with_freqs,
/// };
/// use ndarray::Array3;
///
/// let s = Array3::from_shape_vec((1, 3, 2), vec![1.0_f64, 1.0, 3.0, 2.0, 5.0, 3.0]).unwrap();
/// let f = Array3::from_shape_vec((1, 3, 2), vec![0.0, 0.0, 1.0, 10.0, 2.0, 20.0]).unwrap();
/// let p = poly_features_from_spectrogram_with_freqs(s.view(), f.view(), &PolyFeaturesConfig::default()).unwrap();
/// assert!((p[[0, 0, 0]] - 2.0).abs() < 1e-12); // frame 0: slope 2
/// assert!((p[[0, 0, 1]] - 0.1).abs() < 1e-12); // frame 1: slope 0.1
/// ```
pub fn poly_features_from_spectrogram_with_freqs<F: AudioFloat>(
    s: ArrayView3<'_, F>,
    freq: ArrayView3<'_, F>,
    config: &PolyFeaturesConfig,
) -> AudioSampleResult<Array3<F>> {
    let (n_ch, n_bins, n_frames) = s.dim();
    if n_ch == 0 || n_bins == 0 || n_frames == 0 {
        return Err(config_err(
            "poly_features",
            "the spectrogram must be non-empty",
        ));
    }
    check_freq_matrix("poly_features", &freq, &s)?;
    if !all_finite(&s) || !all_finite(&freq) {
        return Err(param_err(
            "spectrogram",
            "poly_features needs finite spectrogram values and frequencies",
        ));
    }
    let m = poly_terms(config.order, n_bins)?;
    let mut out = Array3::<F>::zeros((n_ch, m, n_frames));
    let mut y = vec![F::zero(); n_bins];
    let mut x = vec![0.0; n_bins];
    let shared = freq.dim().0 == 1;
    let mut failure = Ok(());
    for c in 0..n_ch {
        let fc = freq.index_axis(Axis(0), if shared { 0 } else { c });
        for_each_col_tile(s.index_axis(Axis(0), c), |t0, tile| {
            for tt in 0..tile.ncols() {
                if failure.is_err() {
                    return;
                }
                let t = t0 + tt;
                for (xi, f) in x.iter_mut().zip(fc.column(t)) {
                    *xi = f.as_f64();
                }
                match PolyFit::<F>::new(&x, m) {
                    Ok(fit) => {
                        y.copy_from_slice(col(&tile, tt));
                        fit.solve(&mut y, &mut out, c, t);
                    }
                    Err(e) => failure = Err(e),
                }
            }
        });
    }
    failure.map(|()| out)
}

/// The number of polynomial coefficients for `order`, at most `n_bins`.
fn poly_terms(order: usize, n_bins: usize) -> AudioSampleResult<usize> {
    let m = order + 1;
    if m > n_bins {
        return Err(config_err(
            "poly_features",
            format!("order {order} needs at least {m} frequency bins, got {n_bins}"),
        ));
    }
    Ok(m)
}

/// The Householder QR factorisation of the Vandermonde matrix of the frequencies,
/// computed in `f64` and rounded to `F` once.
struct PolyFit<F> {
    vs: Vec<Vec<F>>,
    betas: Vec<F>,
    a: Array2<F>,
    scale: Vec<F>,
}

impl<F: AudioFloat> PolyFit<F> {
    fn new(freq: &[f64], m: usize) -> AudioSampleResult<Self> {
        let n_bins = freq.len();
        // Vandermonde matrix (numpy.vander: column j holds x^(order - j), powers by
        // repeated multiplication), columns scaled to unit Euclidean norm.
        let mut a = Array2::<f64>::zeros((n_bins, m));
        for (i, &x) in freq.iter().enumerate() {
            let mut v = 1.0;
            for j in (0..m).rev() {
                a[[i, j]] = v;
                v *= x;
            }
        }
        let mut scale = vec![0.0; m];
        for i in 0..n_bins {
            for (j, sc) in scale.iter_mut().enumerate() {
                *sc += a[[i, j]] * a[[i, j]];
            }
        }
        for sc in &mut scale {
            *sc = sc.sqrt();
            if *sc == 0.0 {
                return Err(rank_deficient());
            }
        }
        for i in 0..n_bins {
            for j in 0..m {
                a[[i, j]] /= scale[j];
            }
        }

        // Householder QR of `a`: reflectors stored in `vs`, R in the upper triangle.
        let mut vs: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut betas = Vec::with_capacity(m);
        for j in 0..m {
            let norm = (j..n_bins)
                .map(|i| a[[i, j]] * a[[i, j]])
                .sum::<f64>()
                .sqrt();
            let alpha = if a[[j, j]] > 0.0 { -norm } else { norm };
            let mut v: Vec<f64> = (j..n_bins).map(|i| a[[i, j]]).collect();
            v[0] -= alpha;
            let vtv: f64 = v.iter().map(|x| x * x).sum();
            let beta = if vtv > 0.0 { 2.0 / vtv } else { 0.0 };
            for col in j..m {
                let dot: f64 = (j..n_bins).map(|i| v[i - j] * a[[i, col]]).sum();
                for i in j..n_bins {
                    a[[i, col]] -= beta * dot * v[i - j];
                }
            }
            vs.push(v);
            betas.push(beta);
        }
        // Rank check with numpy.polyfit's default rcond = len(x) * eps.
        let rmax = (0..m).map(|j| a[[j, j]].abs()).fold(0.0, f64::max);
        let rcond = n_bins as f64 * f64::EPSILON;
        if (0..m).any(|j| a[[j, j]].abs() <= rcond * rmax) {
            return Err(rank_deficient());
        }

        // The factorisation, rounded to F once.
        Ok(Self {
            vs: vs
                .iter()
                .map(|v| v.iter().map(|&x| F::of(x)).collect())
                .collect(),
            betas: betas.iter().map(|&b| F::of(b)).collect(),
            a: a.mapv(F::of),
            scale: scale.iter().map(|&x| F::of(x)).collect(),
        })
    }

    /// Writes the coefficients of the frame `y` (overwritten) to `out[[c, .., t]]`.
    #[inline]
    fn solve(&self, y: &mut [F], out: &mut Array3<F>, c: usize, t: usize) {
        let m = self.scale.len();
        for (j, (v, &beta)) in self.vs.iter().zip(&self.betas).enumerate() {
            let dot: F = v.iter().zip(&y[j..]).map(|(&a, &b)| a * b).sum();
            for (yi, &vi) in y[j..].iter_mut().zip(v) {
                *yi -= beta * dot * vi;
            }
        }
        // Back substitution R x = (Q^T y)[..m].
        for j in (0..m).rev() {
            let mut acc = y[j];
            for l in j + 1..m {
                acc -= self.a[[j, l]] * out[[c, l, t]];
            }
            out[[c, j, t]] = acc / self.a[[j, j]];
        }
        for j in 0..m {
            out[[c, j, t]] /= self.scale[j];
        }
    }
}

fn rank_deficient() -> AudioSampleError {
    config_err(
        "poly_features",
        "the polynomial fit is rank-deficient (too few distinct frequencies for the order)",
    )
}

// =============================================================================
// RMS and zero-crossing rate
// =============================================================================

/// Configuration of the frame-wise RMS energy, mirroring `librosa.feature.rms`.
///
/// ## Purpose
///
/// RMS of each frame of the time-domain signal: `sqrt(mean(x^2))`.
///
/// ## Intended usage
///
/// Pass to [`AudioFeatures::rms_frames`]. [`Default`] is librosa 1.0's default
/// (`frame_length = 2048`, `hop_length = 512`, centred, zero padding), so frames
/// line up with those of the default STFT.
///
/// ## Invariants
///
/// None beyond the non-zero lengths enforced by the types.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::features::RmsConfig;
///
/// let cfg = RmsConfig::new().with_frame_length(nzu!(1024)).with_center(false);
/// assert_eq!(cfg.hop_length.get(), 512);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RmsConfig {
    /// Samples per frame; longer frames smooth the envelope.
    pub frame_length: NonZeroUsize,
    /// Samples between successive frames.
    pub hop_length: NonZeroUsize,
    /// If `true` (default), pad `frame_length / 2` samples at both ends so frame `t`
    /// is centred on sample `t * hop_length`.
    pub center: bool,
    /// Padding used when `center` is set (librosa 1.0 default: zeros).
    pub pad_mode: StftPadMode,
}

impl Default for RmsConfig {
    fn default() -> Self {
        Self {
            frame_length: crate::nzu!(2048),
            hop_length: crate::nzu!(512),
            center: true,
            pad_mode: StftPadMode::Constant,
        }
    }
}

impl RmsConfig {
    /// Returns the default configuration (same as [`Default`], which equals
    /// librosa 1.0's defaults).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// librosa 1.0's `feature.rms` defaults. Currently the same as [`Default`];
    /// use this constructor where agreement with librosa is the requirement.
    #[must_use]
    pub fn librosa() -> Self {
        Self::default()
    }

    /// Sets the frame length.
    #[must_use]
    pub fn with_frame_length(mut self, frame_length: NonZeroUsize) -> Self {
        self.frame_length = frame_length;
        self
    }

    /// Sets the hop length.
    #[must_use]
    pub fn with_hop_length(mut self, hop_length: NonZeroUsize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Sets centring.
    #[must_use]
    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Sets the padding mode.
    #[must_use]
    pub fn with_pad_mode(mut self, pad_mode: StftPadMode) -> Self {
        self.pad_mode = pad_mode;
        self
    }

    /// Checks the configuration (always valid; provided for API uniformity).
    ///
    /// # Errors
    ///
    /// None currently.
    pub fn validate(&self) -> AudioSampleResult<()> {
        Ok(())
    }
}

/// Frame-wise RMS of every row of `signal`; see [`AudioFeatures::rms_frames`].
fn rms_rows<F: AudioFloat>(
    signal: ArrayView2<'_, F>,
    config: &RmsConfig,
) -> AudioSampleResult<Array3<F>> {
    let (n_ch, len) = signal.dim();
    let fl = config.frame_length.get();
    let hop = config.hop_length.get();
    let pad = if config.center { fl / 2 } else { 0 };
    let n_frames = frame_count(len + 2 * pad, fl, hop)?;
    let mut out = Array3::<F>::zeros((n_ch, 1, n_frames));
    let mut row_buf = Vec::new();
    for (c, row) in signal.outer_iter().enumerate() {
        let x: &[F] = if let Some(s) = row.as_slice() {
            s
        } else {
            row_buf.clear();
            row_buf.extend(row.iter().copied());
            &row_buf
        };
        let sums = frame_square_sums(x, fl, hop, pad, config.pad_mode, n_frames);
        for (o, &sum) in out.slice_mut(ndarray::s![c, 0, ..]).iter_mut().zip(&sums) {
            *o = (sum / F::of_usize(fl)).sqrt();
        }
    }
    Ok(out)
}

/// Computes frame-wise RMS from a spectrogram, equivalent to
/// `librosa.feature.rms(S=s, frame_length=frame_length)`.
///
/// By Parseval's theorem, `rms = sqrt(2 sum_k w_k |S[k]|^2 / frame_length^2)` with
/// `w_k = 1/2` for the DC bin and (for even `frame_length`) the Nyquist bin and 1
/// otherwise. This equals the time-domain RMS of each windowed frame.
///
/// # Arguments
///
/// * `s` - `(channels, frame_length / 2 + 1, frames)` STFT magnitudes of element
///   type `F` (`f32` or `f64`).
/// * `frame_length` - The FFT length used to compute `s`.
///
/// # Returns
///
/// A `(channels, 1, frames)` array of element type `F`.
///
/// # Errors
///
/// [`ParameterError`] if the bin count is not `frame_length / 2 + 1`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::features::rms_from_spectrogram;
/// use ndarray::Array3;
///
/// // A DC frame of ones (length 8) has DFT bin 0 = 8.
/// let mut s = Array3::<f64>::zeros((1, 5, 1));
/// s[[0, 0, 0]] = 8.0;
/// let r = rms_from_spectrogram(s.view(), nzu!(8)).unwrap();
/// assert!((r[[0, 0, 0]] - 1.0).abs() < 1e-12);
/// ```
pub fn rms_from_spectrogram<F: AudioFloat>(
    s: ArrayView3<'_, F>,
    frame_length: NonZeroUsize,
) -> AudioSampleResult<Array3<F>> {
    let (n_ch, n_bins, n_frames) = s.dim();
    let fl = frame_length.get();
    if n_bins != fl / 2 + 1 {
        return Err(config_err(
            "rms",
            format!(
                "the spectrogram has {n_bins} bins, so frame_length is expected to be {} or {}; found {fl}",
                (2 * n_bins).saturating_sub(2),
                (2 * n_bins).saturating_sub(1)
            ),
        ));
    }
    let mut out = Array3::<F>::zeros((n_ch, 1, n_frames));
    let (half, two, fl_f) = (F::of(0.5), F::of(2.0), F::of_usize(fl));
    let edge = |k: usize| k == 0 || (fl.is_multiple_of(2) && k == n_bins - 1);
    let finish = |a: F| (two * a / (fl_f * fl_f)).sqrt();
    let mut acc = vec![F::zero(); n_frames];
    for c in 0..n_ch {
        let sc = s.index_axis(Axis(0), c);
        let o = out_row(&mut out, c);
        if n_frames > 1 && sc.strides()[0] == 1 {
            // Contiguous frames (Fortran order): 8 frames at a time, each
            // summed over the bins in ascending order as below.
            const G: usize = 8;
            let mut t = 0;
            while t + G <= n_frames {
                let cols: [&[F]; G] =
                    std::array::from_fn(|j| sc.column(t + j).to_slice().expect("unit row stride"));
                let mut a = [F::zero(); G];
                // `k` indexes the 8 columns at once.
                #[allow(clippy::needless_range_loop)]
                for k in 0..n_bins {
                    if edge(k) {
                        for j in 0..G {
                            let mut x = cols[j][k] * cols[j][k];
                            x *= half;
                            a[j] += x;
                        }
                    } else {
                        for j in 0..G {
                            a[j] += cols[j][k] * cols[j][k];
                        }
                    }
                }
                for j in 0..G {
                    o[t + j] = finish(a[j]);
                }
                t += G;
            }
            for (tt, o) in o.iter_mut().enumerate().skip(t) {
                let column = sc.column(tt);
                let column = column.to_slice().expect("unit row stride");
                let mut a = F::zero();
                for (k, &v) in column.iter().enumerate() {
                    let mut x = v * v;
                    if edge(k) {
                        x *= half;
                    }
                    a += x;
                }
                *o = finish(a);
            }
            continue;
        }
        for_each_row_tile(sc, |t0, tile| {
            let w = tile.ncols();
            let acc = &mut acc[..w];
            acc.fill(F::zero());
            for k in 0..n_bins {
                let r = row(&tile, k);
                if edge(k) {
                    for (a, &v) in acc.iter_mut().zip(r) {
                        let mut x = v * v;
                        x *= half;
                        *a += x;
                    }
                } else {
                    for (a, &v) in acc.iter_mut().zip(r) {
                        *a += v * v;
                    }
                }
            }
            for (o, &a) in o[t0..t0 + w].iter_mut().zip(acc.iter()) {
                *o = finish(a);
            }
        });
    }
    Ok(out)
}

/// Scaling of the zero-crossing threshold (librosa's `ref_magnitude`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ZeroCrossingReference {
    /// The threshold is absolute (librosa `ref_magnitude=None`, the default).
    #[default]
    None,
    /// The threshold is multiplied by this value.
    Value(f64),
    /// The threshold is multiplied by the largest absolute sample value of the
    /// framed signal over all channels (librosa `ref_magnitude=numpy.max`).
    Max,
}

/// Configuration of the frame-wise zero-crossing rate, mirroring
/// `librosa.feature.zero_crossing_rate`.
///
/// ## Purpose
///
/// The fraction of sign changes per frame. Samples with `|x| <= threshold` count as
/// zero, which keeps low-level noise from producing crossings.
///
/// ## Intended usage
///
/// Pass to [`AudioFeatures::zero_crossing_rate_frames`]. [`Default`] is librosa's
/// default: `frame_length = 2048`, `hop_length = 512`, centred with **edge**
/// padding (librosa always edge-pads here), `threshold = 1e-10`, no reference,
/// `pad = false`, `zero_pos = true`.
///
/// ## Invariants
///
/// Checked by [`ZeroCrossingRateConfig::validate`]: `threshold` finite and
/// non-negative; a `Value` reference finite.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::ZeroCrossingRateConfig;
///
/// let cfg = ZeroCrossingRateConfig::new().with_zero_pos(false);
/// assert_eq!(cfg.threshold, 1e-10);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ZeroCrossingRateConfig {
    /// Samples per frame.
    pub frame_length: NonZeroUsize,
    /// Samples between successive frames.
    pub hop_length: NonZeroUsize,
    /// If `true` (default), edge-pad `frame_length / 2` samples at both ends so frame
    /// `t` is centred on sample `t * hop_length`.
    pub center: bool,
    /// Samples with `|x| <= threshold * reference` are treated as zero. Non-negative.
    pub threshold: f64,
    /// Scaling of `threshold` (librosa `ref_magnitude`).
    pub ref_magnitude: ZeroCrossingReference,
    /// If `true`, the first sample of every frame counts as a crossing (librosa's
    /// `pad` keyword; `false` in `zero_crossing_rate`).
    pub pad: bool,
    /// If `true` (default), zero counts as positive (sign bit comparison); if
    /// `false`, zero is its own sign, so moving to or from zero is a crossing.
    pub zero_pos: bool,
}

impl Default for ZeroCrossingRateConfig {
    fn default() -> Self {
        Self {
            frame_length: crate::nzu!(2048),
            hop_length: crate::nzu!(512),
            center: true,
            threshold: 1e-10,
            ref_magnitude: ZeroCrossingReference::None,
            pad: false,
            zero_pos: true,
        }
    }
}

impl ZeroCrossingRateConfig {
    /// Returns the default configuration (same as [`Default`], which equals
    /// librosa 1.0's defaults).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// librosa 1.0's `feature.zero_crossing_rate` defaults. Currently the same
    /// as [`Default`]; use this constructor where agreement with librosa is the
    /// requirement.
    #[must_use]
    pub fn librosa() -> Self {
        Self::default()
    }

    /// Sets the frame length.
    #[must_use]
    pub fn with_frame_length(mut self, frame_length: NonZeroUsize) -> Self {
        self.frame_length = frame_length;
        self
    }

    /// Sets the hop length.
    #[must_use]
    pub fn with_hop_length(mut self, hop_length: NonZeroUsize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Sets centring.
    #[must_use]
    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Sets the threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets the threshold reference.
    #[must_use]
    pub fn with_ref_magnitude(mut self, ref_magnitude: ZeroCrossingReference) -> Self {
        self.ref_magnitude = ref_magnitude;
        self
    }

    /// Sets whether the first sample of each frame counts as a crossing.
    #[must_use]
    pub fn with_pad(mut self, pad: bool) -> Self {
        self.pad = pad;
        self
    }

    /// Sets whether zero counts as positive.
    #[must_use]
    pub fn with_zero_pos(mut self, zero_pos: bool) -> Self {
        self.zero_pos = zero_pos;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `threshold` is negative or not finite, or a `Value`
    /// reference is not finite.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if !(self.threshold.is_finite() && self.threshold >= 0.0) {
            return Err(range_err(
                "threshold",
                self.threshold,
                "0",
                "finite",
                "threshold must be finite and non-negative",
            ));
        }
        if let ZeroCrossingReference::Value(v) = self.ref_magnitude {
            if !v.is_finite() {
                return Err(param_err("ref_magnitude", "the reference must be finite"));
            }
        }
        Ok(())
    }
}

/// numpy's `sign` (NaN propagates).
fn np_sign<F: AudioFloat>(x: F) -> F {
    if x > F::zero() {
        F::one()
    } else if x < F::zero() {
        -F::one()
    } else if x == F::zero() {
        F::zero()
    } else {
        F::nan()
    }
}

fn zcr_rows<F: AudioFloat>(
    signal: ArrayView2<'_, F>,
    config: &ZeroCrossingRateConfig,
) -> AudioSampleResult<Array3<F>> {
    config.validate()?;
    let (n_ch, len) = signal.dim();
    let fl = config.frame_length.get();
    let hop = config.hop_length.get();
    let pad = if config.center { fl / 2 } else { 0 };
    let n_frames = frame_count(len + 2 * pad, fl, hop)?;
    // The edge padding repeats x[0] and x[len - 1], so the padded frames
    // contain no crossings inside the padding, and the samples of the padded
    // signal the frames cover are the samples of x[..hi] (plus copies of its
    // end samples).
    let covered = (n_frames - 1) * hop + fl;
    let hi = len.min(covered - pad);
    let signal = signal.as_standard_layout();
    let rows: Vec<&[F]> = signal
        .outer_iter()
        .map(|row| row.to_slice().expect("rows of a standard-layout array"))
        .collect();
    let threshold = match config.ref_magnitude {
        ZeroCrossingReference::None => F::of(config.threshold),
        ZeroCrossingReference::Value(v) => F::of(config.threshold * v),
        ZeroCrossingReference::Max => {
            let max = rows
                .iter()
                .flat_map(|x| x[..hi].iter())
                .fold(F::neg_infinity(), |m, v| m.max(v.abs()));
            F::of(config.threshold) * max
        }
    };
    let clean = |v: F| {
        if -threshold <= v && v <= threshold {
            F::zero()
        } else {
            v
        }
    };
    let zero_pos = config.zero_pos;
    let crossing = |prev: F, cur: F| -> bool {
        let (x1, x0) = (clean(prev), clean(cur));
        if zero_pos {
            x0.is_sign_negative() != x1.is_sign_negative()
        } else {
            np_sign(x0) != np_sign(x1)
        }
    };
    // Frame t covers the pairs (xp[i - 1], xp[i]) for i in t*hop + 1 ..
    // t*hop + fl, i.e. the pairs (x[m - 1], x[m]) of the signal for m in
    // clamp(t*hop + 1 - pad) .. clamp(t*hop + fl - pad). With D(j) the number
    // of crossings among the pairs m < j, the frame count is
    // D(end_t) - D(start_t); D is evaluated at the (sorted) frame bounds in
    // one pass over the signal.
    let clamp = |i: usize| i.saturating_sub(pad).min(len);
    let mut bounds: Vec<usize> = Vec::with_capacity(2 * n_frames);
    for t in 0..n_frames {
        bounds.push(clamp(t * hop + 1));
        bounds.push(clamp(t * hop + fl));
    }
    let mut sorted = bounds.clone();
    sorted.sort_unstable();
    sorted.dedup();
    let mut d_at = vec![0_usize; sorted.len()];
    let mut out = Array3::<F>::zeros((n_ch, 1, n_frames));
    for (c, x) in rows.iter().enumerate() {
        let mut total = 0_usize;
        let mut prev = 1_usize; // pairs start at m = 1
        for (d, &j) in d_at.iter_mut().zip(&sorted) {
            if j > prev {
                total += x[prev - 1..j]
                    .windows(2)
                    .map(|w| usize::from(crossing(w[0], w[1])))
                    .sum::<usize>();
                prev = j;
            }
            *d = total;
        }
        let d_of = |j: usize| {
            d_at[sorted
                .binary_search(&j)
                .expect("bound is in the sorted list")]
        };
        for t in 0..n_frames {
            let count = usize::from(config.pad) + d_of(bounds[2 * t + 1]) - d_of(bounds[2 * t]);
            out[[c, 0, t]] = F::of_usize(count) / F::of_usize(fl);
        }
    }
    Ok(out)
}

// =============================================================================
// PCEN
// =============================================================================

/// Configuration of per-channel energy normalisation, mirroring `librosa.pcen`.
///
/// ## Purpose
///
/// PCEN replaces log compression of a (mel) spectrogram with automatic gain
/// control followed by root compression:
/// `PCEN = (S / (eps + M)^gain + bias)^power - bias^power`, where `M` is `S`
/// smoothed over time by a one-pole low-pass filter. It suppresses stationary
/// background and emphasises onsets.
///
/// ## Intended usage
///
/// Pass to [`pcen`] / [`pcen_with_state`] with the sample rate of the analysed
/// audio. There is no `Default`: [`PcenConfig::new`] takes the hop length,
/// which must be the hop of the spectrogram being normalised (it converts
/// `time_constant` from seconds to frames). The other parameters take librosa's
/// defaults (`gain = 0.98`, `bias = 2`, `power = 0.5`, `time_constant = 0.4` s,
/// `eps = 1e-6`, `max_size = 1`). [`PcenConfig::librosa`] also sets librosa's
/// default `hop_length = 512`. librosa recommends scaling integer-PCM-range
/// input, e.g. `S * 2^31` for a mel spectrogram of `[-1, 1]` audio.
///
/// ## Invariants
///
/// Checked by [`PcenConfig::validate`]: `gain`, `bias`, `power` finite and
/// non-negative; `eps`, `time_constant` finite and positive; `b` (explicit or
/// derived) in `[0, 1]`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::features::PcenConfig;
///
/// let cfg = PcenConfig::new(nzu!(512)).with_power(0.25);
/// // b from the time constant: 0.4 s at 22050 Hz / 512 hop.
/// let b = cfg.smoothing_coefficient(22050.0).unwrap();
/// assert!((b - 0.0573).abs() < 1e-3);
/// assert_eq!(PcenConfig::librosa().hop_length.get(), 512);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PcenConfig {
    /// Hop length of the spectrogram in samples; with the sample rate it converts
    /// `time_constant` to frames. Required ([`PcenConfig::new`]); librosa's
    /// default is 512.
    pub hop_length: NonZeroUsize,
    /// Gain-normalisation exponent `alpha` (typically 0.7 to 1). Non-negative.
    pub gain: f64,
    /// Bias `delta` added before compression. Non-negative.
    pub bias: f64,
    /// Compression exponent `r` (`0` selects `log1p` compression). Non-negative.
    pub power: f64,
    /// Time constant of the smoothing filter in seconds. Positive.
    pub time_constant: f64,
    /// Small positive constant avoiding division by zero.
    pub eps: f64,
    /// Explicit smoothing coefficient in `[0, 1]`; `None` derives it from
    /// `time_constant` (see [`PcenConfig::smoothing_coefficient`]).
    pub b: Option<f64>,
    /// Width (in frequency bins) of a maximum filter applied to `S` before
    /// smoothing (`1` disables it), as in librosa's `max_size`.
    pub max_size: NonZeroUsize,
}

impl PcenConfig {
    /// PCEN for a spectrogram with hop length `hop_length`, with librosa's
    /// defaults for the other parameters (`gain = 0.98`, `bias = 2`,
    /// `power = 0.5`, `time_constant = 0.4` s, `eps = 1e-6`, `b` derived from
    /// the time constant, `max_size = 1`).
    #[must_use]
    pub fn new(hop_length: NonZeroUsize) -> Self {
        Self {
            hop_length,
            gain: 0.98,
            bias: 2.0,
            power: 0.5,
            time_constant: 0.4,
            eps: 1e-6,
            b: None,
            max_size: crate::nzu!(1),
        }
    }

    /// librosa 1.0's `pcen` defaults, including `hop_length = 512` (the hop of
    /// librosa's default STFT).
    #[must_use]
    pub fn librosa() -> Self {
        Self::new(crate::nzu!(512))
    }

    /// Sets the hop length.
    #[must_use]
    pub fn with_hop_length(mut self, hop_length: NonZeroUsize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Sets the gain exponent.
    #[must_use]
    pub fn with_gain(mut self, gain: f64) -> Self {
        self.gain = gain;
        self
    }

    /// Sets the bias.
    #[must_use]
    pub fn with_bias(mut self, bias: f64) -> Self {
        self.bias = bias;
        self
    }

    /// Sets the compression exponent.
    #[must_use]
    pub fn with_power(mut self, power: f64) -> Self {
        self.power = power;
        self
    }

    /// Sets the time constant in seconds.
    #[must_use]
    pub fn with_time_constant(mut self, time_constant: f64) -> Self {
        self.time_constant = time_constant;
        self
    }

    /// Sets `eps`.
    #[must_use]
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Sets an explicit smoothing coefficient.
    #[must_use]
    pub fn with_b(mut self, b: Option<f64>) -> Self {
        self.b = b;
        self
    }

    /// Sets the frequency max-filter width.
    #[must_use]
    pub fn with_max_size(mut self, max_size: NonZeroUsize) -> Self {
        self.max_size = max_size;
        self
    }

    /// The smoothing coefficient `b` used for a spectrogram at `sample_rate` Hz:
    /// the explicit `b`, or `(sqrt(1 + 4 T^2) - 1) / (2 T^2)` with
    /// `T = time_constant * sample_rate / hop_length` frames (librosa's choice, which
    /// approximates the filter's full width at half maximum). Also validates the
    /// configuration.
    ///
    /// # Errors
    ///
    /// As [`PcenConfig::validate`].
    pub fn smoothing_coefficient(&self, sample_rate: f64) -> AudioSampleResult<f64> {
        validate_sample_rate(sample_rate)?;
        for (name, v) in [
            ("power", self.power),
            ("gain", self.gain),
            ("bias", self.bias),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(range_err(
                    name,
                    v,
                    "0",
                    "finite",
                    "must be finite and non-negative",
                ));
            }
        }
        for (name, v) in [("eps", self.eps), ("time_constant", self.time_constant)] {
            if !(v.is_finite() && v > 0.0) {
                return Err(range_err(
                    name,
                    v,
                    "> 0",
                    "finite",
                    "must be strictly positive",
                ));
            }
        }
        let b = match self.b {
            Some(b) => b,
            None => {
                let t = self.time_constant * sample_rate / self.hop_length.get() as f64;
                ((1.0 + 4.0 * t * t).sqrt() - 1.0) / (2.0 * t * t)
            }
        };
        if !(0.0..=1.0).contains(&b) {
            return Err(range_err("b", b, "0", "1", "b must be between 0 and 1"));
        }
        Ok(b)
    }

    /// Checks the configuration for a spectrogram at `sample_rate` Hz.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if any parameter is out of range (see the type-level
    /// invariants) or `sample_rate` is not positive.
    pub fn validate(&self, sample_rate: f64) -> AudioSampleResult<()> {
        self.smoothing_coefficient(sample_rate).map(|_| ())
    }
}

/// Applies per-channel energy normalisation to a spectrogram, equivalent to
/// `librosa.pcen(S, sr=sample_rate, hop_length=..., ...)` with time on the last axis.
///
/// See [`pcen_with_state`] for the algorithm, a custom reference and streaming.
///
/// # Arguments
///
/// * `s` - `(channels, bins, frames)` non-negative spectrogram (typically a mel
///   power spectrogram) of element type `F` (`f32` or `f64`).
/// * `sample_rate` - Sample rate of the analysed audio in Hz.
/// * `config` - PCEN parameters ([`PcenConfig::new`] with the spectrogram's hop
///   length, or [`PcenConfig::librosa`]).
///
/// # Returns
///
/// The PCEN-normalised spectrogram, same shape and element type as `s`.
///
/// # Errors
///
/// [`ParameterError`] if the configuration is invalid.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::{PcenConfig, pcen};
/// use ndarray::Array3;
///
/// let s = Array3::<f64>::from_elem((1, 4, 10), 1e3);
/// let p = pcen(s.view(), 22050.0, &PcenConfig::librosa()).unwrap();
/// assert_eq!(p.dim(), (1, 4, 10));
/// assert!(p.iter().all(|v| v.is_finite() && *v >= 0.0));
/// ```
///
/// # References
///
/// Y. Wang, P. Getreuer, T. Hughes, R. F. Lyon and R. A. Saurous, "Trainable
/// frontend for robust and far-field keyword spotting," *Proc. ICASSP*, 2017;
/// V. Lostanlen et al., "Per-channel energy normalization: Why and how,"
/// *IEEE Signal Processing Letters* 26(1), 2019.
pub fn pcen<F: AudioFloat>(
    s: ArrayView3<'_, F>,
    sample_rate: f64,
    config: &PcenConfig,
) -> AudioSampleResult<Array3<F>> {
    pcen_with_state(s, sample_rate, config, None, None).map(|(out, _)| out)
}

/// Applies PCEN with an optional reference spectrogram and filter state, equivalent
/// to `librosa.pcen(S, ..., ref=reference, zi=zi, return_zf=True)` (time on the
/// last axis, maximum filter over frequency).
///
/// 1. `ref = reference`, or `S` maximum-filtered over `max_size` bins (SciPy
///    `maximum_filter1d`, mode `reflect`), or `S` itself when `max_size == 1`.
/// 2. `M[t] = b ref[t] + (1 - b) M[t - 1]` (SciPy `lfilter([b], [1, b - 1])`); the
///    initial state defaults to `lfilter_zi`, i.e. the steady state for a unit input
///    (`M[-1] = 1`, not scaled by the first frame, as librosa).
/// 3. `G = exp(-gain (ln eps + ln1p(M / eps)))` = `(eps + M)^-gain`.
/// 4. `power == 0`: `ln1p(S G)`; `bias == 0`: `exp(power (ln S + ln G))`;
///    otherwise `bias^power expm1(power ln1p(S G / bias))`.
///
/// The smoothing filter and the compression run in the element type `F` of
/// `s` (`f32` or `f64`); `b` and the other parameters are computed in `f64`
/// and rounded to `F` once.
///
/// # Arguments
///
/// * `s` - `(channels, bins, frames)` spectrogram.
/// * `sample_rate` - Sample rate of the analysed audio in Hz.
/// * `config` - PCEN parameters.
/// * `reference` - Optional `(channels, bins, frames)` array to smooth instead of `s`
///   (librosa `ref`); overrides `max_size`.
/// * `zi` - Optional `(channels, bins)` initial filter state (librosa `zi`, e.g. the
///   `zf` of the previous block when streaming).
///
/// # Returns
///
/// `(pcen, zf)`: the normalised spectrogram and the `(channels, bins)` final filter
/// state, both of element type `F`.
///
/// # Errors
///
/// [`ParameterError`] if the configuration is invalid or `reference` / `zi` have the
/// wrong shape.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::features::{PcenConfig, pcen, pcen_with_state};
/// use ndarray::{Array3, s};
///
/// let x = Array3::from_shape_fn((1, 3, 20), |(_, k, t)| 1.0 + (k + t) as f64);
/// let cfg = PcenConfig::new(nzu!(512));
/// let whole = pcen(x.view(), 22050.0, &cfg).unwrap();
/// // Streaming in two blocks gives the same result.
/// let (a, zf) = pcen_with_state(x.slice(s![.., .., ..8]), 22050.0, &cfg, None, None).unwrap();
/// let (b, _) = pcen_with_state(x.slice(s![.., .., 8..]), 22050.0, &cfg, None, Some(zf.view())).unwrap();
/// assert!((a[[0, 1, 7]] - whole[[0, 1, 7]]).abs() < 1e-12);
/// assert!((b[[0, 1, 0]] - whole[[0, 1, 8]]).abs() < 1e-12);
/// ```
pub fn pcen_with_state<F: AudioFloat>(
    s: ArrayView3<'_, F>,
    sample_rate: f64,
    config: &PcenConfig,
    reference: Option<ArrayView3<'_, F>>,
    zi: Option<ArrayView2<'_, F>>,
) -> AudioSampleResult<(Array3<F>, Array2<F>)> {
    let b64 = config.smoothing_coefficient(sample_rate)?;
    let b = F::of(b64);
    let (n_ch, n_bins, n_frames) = s.dim();
    let filtered;
    let reference = match reference {
        Some(r) => {
            if r.dim() != s.dim() {
                return Err(config_err(
                    "pcen",
                    format!("ref has shape {:?}, expected {:?}", r.dim(), s.dim()),
                ));
            }
            r
        }
        None if config.max_size.get() > 1 => {
            filtered = maximum_filter_bins(&s, config.max_size.get());
            filtered.view()
        }
        None => s,
    };
    let mut state = match zi {
        Some(z) => {
            if z.dim() != (n_ch, n_bins) {
                return Err(config_err(
                    "pcen",
                    format!("zi has shape {:?}, expected ({n_ch}, {n_bins})", z.dim()),
                ));
            }
            z.to_owned()
        }
        None => {
            // scipy.signal.lfilter_zi([b], [1, b - 1]) evaluated as SciPy does
            // (in f64, as SciPy does for the float64 coefficients).
            let a1 = b64 - 1.0;
            let y_inf = b64 / (1.0 + a1);
            Array2::from_elem((n_ch, n_bins), F::of(0.0 - y_inf * a1))
        }
    };
    let a1 = F::of(b64 - 1.0);
    let ln_eps = F::of(config.eps.ln());
    let bias_pow = F::of(config.bias.powf(config.power));
    let (gain, eps, power, bias) = (
        F::of(config.gain),
        F::of(config.eps),
        F::of(config.power),
        F::of(config.bias),
    );
    let mut out = Array3::<F>::zeros((n_ch, n_bins, n_frames));
    let mut x_buf = Vec::new();
    for c in 0..n_ch {
        let rc = reference.index_axis(Axis(0), c);
        let rc = rc.as_standard_layout();
        let rc = rc.as_slice().expect("standard layout");
        let mut oc = out.index_axis_mut(Axis(0), c);
        let oc = oc.as_slice_mut().expect("a fresh standard-layout array");
        // Direct-form II transposed, as scipy.signal.lfilter: out = M. Four
        // bins at a time, so that their recurrences overlap.
        let mut k = 0;
        while k < n_bins {
            let g = (n_bins - k).min(4);
            let mut z = [F::zero(); 4];
            for (j, zj) in z.iter_mut().enumerate().take(g) {
                *zj = state[[c, k + j]];
            }
            for t in 0..n_frames {
                for j in 0..g {
                    let rv = rc[(k + j) * n_frames + t];
                    let m = b * rv + z[j];
                    z[j] = F::zero() * rv - a1 * m;
                    oc[(k + j) * n_frames + t] = m;
                }
            }
            for (j, &zj) in z.iter().enumerate().take(g) {
                state[[c, k + j]] = zj;
            }
            k += g;
        }
        for (k, o) in oc.chunks_exact_mut(n_frames.max(1)).enumerate() {
            let x = contiguous_or_copy(s.slice(ndarray::s![c, k, ..]), &mut x_buf);
            // One function per loop over the row, with the operations of a
            // per-element evaluation.
            // o = G = exp(-gain (ln eps + ln1p(M / eps)))
            for o in o.iter_mut() {
                *o = (*o / eps).ln_1p();
            }
            for o in o.iter_mut() {
                *o = (-gain * (ln_eps + *o)).exp();
            }
            if config.power == 0.0 {
                for (o, &xv) in o.iter_mut().zip(x) {
                    *o = (xv * *o).ln_1p();
                }
            } else if config.bias == 0.0 {
                for (o, &xv) in o.iter_mut().zip(x) {
                    *o = (power * (xv.ln() + o.ln())).exp();
                }
            } else {
                for (o, &xv) in o.iter_mut().zip(x) {
                    *o = (xv * *o / bias).ln_1p();
                }
                for o in o.iter_mut() {
                    *o = bias_pow * (power * *o).exp_m1();
                }
            }
        }
    }
    Ok((out, state))
}

/// The values of a 1-D view as a slice: the view itself when contiguous,
/// otherwise copied into `buf`.
fn contiguous_or_copy<'a, F: AudioFloat>(
    v: ndarray::ArrayView1<'a, F>,
    buf: &'a mut Vec<F>,
) -> &'a [F] {
    match v.to_slice() {
        Some(sl) => sl,
        None => {
            buf.clear();
            buf.extend(v.iter().copied());
            buf
        }
    }
}

/// `scipy.ndimage.maximum_filter1d(S, size, axis=bins)` with mode `reflect`
/// (numpy `symmetric`): output bin `i` is the maximum over bins
/// `i - size / 2 ..= i + (size - 1) / 2`.
fn maximum_filter_bins<F: AudioFloat>(s: &ArrayView3<'_, F>, size: usize) -> Array3<F> {
    let (n_ch, n_bins, n_frames) = s.dim();
    let lo = (size / 2) as isize;
    // A window of 2 * n_bins or more covers a whole symmetric-padding period
    // and so sees every value of the column (NaN included): scanning one
    // period gives the same result and bounds the work for a huge `max_size`.
    let span = size.min(2 * n_bins) as isize;
    let mut out = Array3::<F>::zeros((n_ch, n_bins, n_frames));
    for c in 0..n_ch {
        for k in 0..n_bins {
            for t in 0..n_frames {
                let mut m = F::neg_infinity();
                for j in 0..span {
                    let src = pad_source_index(k as isize - lo + j, n_bins, StftPadMode::Symmetric)
                        .expect("symmetric padding always has a source");
                    let v = s[[c, src, t]];
                    if v > m || v.is_nan() {
                        m = v;
                    }
                }
                out[[c, k, t]] = m;
            }
        }
    }
    out
}

// =============================================================================
// Frequency weighting
// =============================================================================

/// An acoustic frequency-weighting curve (IEC 61672 A/B/C, IEC 537 D, or flat Z).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FrequencyWeighting {
    /// A-weighting: approximates the ear's sensitivity at low levels (librosa default).
    #[default]
    A,
    /// B-weighting (medium levels; obsolete in standards).
    B,
    /// C-weighting: nearly flat, for high levels and peak measurements.
    C,
    /// D-weighting: for aircraft noise.
    D,
    /// Z-weighting: flat (0 dB at all frequencies).
    Z,
}

/// `log10(x)` with the vectorised kernel ([`log10_f64`]).
fn log10_one(x: f64) -> f64 {
    let mut v = [x];
    log10_f64(&mut v);
    v[0]
}

/// Evaluates a weighting curve that is a combination of logarithms:
/// `combine([log10(a_k(f^2))])` for every frequency, then the `min_db` floor.
/// Works in blocks of 256 frequencies, each logarithm term taken over the
/// block with the vectorised [`log10_f64`].
fn weighting_by_logs<const K: usize>(
    frequencies: &[f64],
    min_db: Option<f64>,
    args: impl Fn(f64) -> [f64; K],
    combine: impl Fn(&[f64; K]) -> f64,
) -> Vec<f64> {
    const B: usize = 256;
    let mut out = Vec::with_capacity(frequencies.len());
    let mut t = [[0.0f64; B]; K];
    for chunk in frequencies.chunks(B) {
        let m = chunk.len();
        for (j, &f) in chunk.iter().enumerate() {
            let a = args(f * f);
            for k in 0..K {
                t[k][j] = a[k];
            }
        }
        for row in t.iter_mut() {
            log10_f64(&mut row[..m]);
        }
        // `j` indexes the K term rows at once.
        #[allow(clippy::needless_range_loop)]
        for j in 0..m {
            let l: [f64; K] = std::array::from_fn(|k| t[k][j]);
            out.push(apply_min_db(combine(&l), min_db));
        }
    }
    out
}

fn apply_min_db(w: f64, min_db: Option<f64>) -> f64 {
    match min_db {
        // numpy.maximum propagates NaN.
        Some(m) if !w.is_nan() => w.max(m),
        _ => w,
    }
}

/// Computes the A-weighting of a set of frequencies, equivalent to
/// `librosa.A_weighting(frequencies, min_db=min_db)`.
///
/// `A(f) = 2.0 + 20 log10(12194.217^2 f^4 / ((f^2 + 20.598997^2)
/// (f^2 + 12194.217^2) sqrt(f^2 + 107.65265^2) sqrt(f^2 + 737.86223^2)))`.
///
/// # Arguments
///
/// * `frequencies` - Frequencies in Hz.
/// * `min_db` - Lower clip in dB (librosa default `Some(-80.0)`); `None` leaves
///   `-inf` at 0 Hz.
///
/// # Returns
///
/// The weighting in dB for each frequency (0 dB near 1 kHz).
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::a_weighting;
///
/// let w = a_weighting(&[0.0, 1000.0], Some(-80.0));
/// assert_eq!(w[0], -80.0);
/// assert!(w[1].abs() < 0.01);
/// ```
///
/// # References
///
/// IEC 61672-1:2013, *Electroacoustics - Sound level meters*; librosa `A_weighting`.
#[must_use]
pub fn a_weighting(frequencies: &[f64], min_db: Option<f64>) -> Vec<f64> {
    let c = [
        12194.217f64 * 12194.217,
        20.598997 * 20.598997,
        107.65265 * 107.65265,
        737.86223 * 737.86223,
    ];
    let lc0 = log10_one(c[0]);
    weighting_by_logs(
        frequencies,
        min_db,
        |f_sq| [f_sq, f_sq + c[0], f_sq + c[1], f_sq + c[2], f_sq + c[3]],
        |l| 2.0 + 20.0 * (lc0 + 2.0 * l[0] - l[1] - l[2] - 0.5 * l[3] - 0.5 * l[4]),
    )
}

/// Computes the B-weighting of a set of frequencies, equivalent to
/// `librosa.B_weighting(frequencies, min_db=min_db)`.
///
/// # Arguments
///
/// * `frequencies` - Frequencies in Hz.
/// * `min_db` - Lower clip in dB (librosa default `Some(-80.0)`).
///
/// # Returns
///
/// The weighting in dB for each frequency.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::b_weighting;
///
/// assert!(b_weighting(&[1000.0], Some(-80.0))[0].abs() < 0.01);
/// ```
///
/// # References
///
/// IEC 60651 (B curve); librosa `B_weighting`.
#[must_use]
pub fn b_weighting(frequencies: &[f64], min_db: Option<f64>) -> Vec<f64> {
    let c = [
        12194.217f64 * 12194.217,
        20.598997 * 20.598997,
        158.48932 * 158.48932,
    ];
    let lc0 = log10_one(c[0]);
    weighting_by_logs(
        frequencies,
        min_db,
        |f_sq| [f_sq, f_sq + c[0], f_sq + c[1], f_sq + c[2]],
        |l| 0.17 + 20.0 * (lc0 + 1.5 * l[0] - l[1] - l[2] - 0.5 * l[3]),
    )
}

/// Computes the C-weighting of a set of frequencies, equivalent to
/// `librosa.C_weighting(frequencies, min_db=min_db)`.
///
/// # Arguments
///
/// * `frequencies` - Frequencies in Hz.
/// * `min_db` - Lower clip in dB (librosa default `Some(-80.0)`).
///
/// # Returns
///
/// The weighting in dB for each frequency.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::c_weighting;
///
/// assert!(c_weighting(&[1000.0], Some(-80.0))[0].abs() < 0.01);
/// ```
///
/// # References
///
/// IEC 61672-1:2013; librosa `C_weighting`.
#[must_use]
pub fn c_weighting(frequencies: &[f64], min_db: Option<f64>) -> Vec<f64> {
    let c = [12194.217f64 * 12194.217, 20.598997 * 20.598997];
    let lc0 = log10_one(c[0]);
    weighting_by_logs(
        frequencies,
        min_db,
        |f_sq| [f_sq, f_sq + c[0], f_sq + c[1]],
        |l| 0.062 + 20.0 * (lc0 + l[0] - l[1] - l[2]),
    )
}

/// Computes the D-weighting of a set of frequencies, equivalent to
/// `librosa.D_weighting(frequencies, min_db=min_db)`.
///
/// # Arguments
///
/// * `frequencies` - Frequencies in Hz.
/// * `min_db` - Lower clip in dB (librosa default `Some(-80.0)`).
///
/// # Returns
///
/// The weighting in dB for each frequency.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::d_weighting;
///
/// assert!(d_weighting(&[1000.0], Some(-80.0))[0].abs() < 0.1);
/// ```
///
/// # References
///
/// IEC 537:1976; librosa `D_weighting`.
#[must_use]
pub fn d_weighting(frequencies: &[f64], min_db: Option<f64>) -> Vec<f64> {
    let sq = |x: f64| x * x;
    let c = [
        sq(8.3046305e-3),
        sq(1018.7),
        sq(1039.6),
        sq(3136.5),
        sq(3424.0),
        sq(282.7),
        sq(1160.0),
    ];
    let lc0 = log10_one(c[0]);
    weighting_by_logs(
        frequencies,
        min_db,
        |f_sq| {
            [
                f_sq,
                sq(c[1] - f_sq) + c[2] * f_sq,
                sq(c[3] - f_sq) + c[4] * f_sq,
                c[5] + f_sq,
                c[6] + f_sq,
            ]
        },
        |l| 20.0 * (0.5 * l[0] - lc0 + 0.5 * (l[1] - l[2] - l[3] - l[4])),
    )
}

/// The flat Z-weighting (0 dB everywhere, raised to `min_db` if that is positive),
/// equivalent to `librosa.Z_weighting(frequencies, min_db=min_db)`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::z_weighting;
///
/// assert_eq!(z_weighting(&[0.0, 100.0], None), vec![0.0, 0.0]);
/// ```
#[must_use]
pub fn z_weighting(frequencies: &[f64], min_db: Option<f64>) -> Vec<f64> {
    vec![apply_min_db(0.0, min_db); frequencies.len()]
}

/// Computes a frequency-weighting curve, equivalent to
/// `librosa.frequency_weighting(frequencies, kind=kind, min_db=min_db)`.
///
/// # Arguments
///
/// * `frequencies` - Frequencies in Hz.
/// * `kind` - The weighting curve.
/// * `min_db` - Lower clip in dB. librosa's defaults are `Some(-80.0)` for A-D and
///   `None` for Z (equivalent for Z, which is 0 dB).
///
/// # Returns
///
/// The weighting in dB for each frequency.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::{FrequencyWeighting, a_weighting, frequency_weighting};
///
/// let f = [100.0, 1000.0];
/// assert_eq!(frequency_weighting(&f, FrequencyWeighting::A, Some(-80.0)), a_weighting(&f, Some(-80.0)));
/// ```
#[must_use]
pub fn frequency_weighting(
    frequencies: &[f64],
    kind: FrequencyWeighting,
    min_db: Option<f64>,
) -> Vec<f64> {
    match kind {
        FrequencyWeighting::A => a_weighting(frequencies, min_db),
        FrequencyWeighting::B => b_weighting(frequencies, min_db),
        FrequencyWeighting::C => c_weighting(frequencies, min_db),
        FrequencyWeighting::D => d_weighting(frequencies, min_db),
        FrequencyWeighting::Z => z_weighting(frequencies, min_db),
    }
}

/// Computes several weighting curves at once, equivalent to
/// `librosa.multi_frequency_weighting(frequencies, kinds=kinds, min_db=min_db)`.
///
/// # Arguments
///
/// * `frequencies` - Frequencies in Hz.
/// * `kinds` - The curves, one output row each (librosa default `[Z, A, C]`).
/// * `min_db` - Lower clip in dB applied to every curve.
///
/// # Returns
///
/// A `(kinds.len(), frequencies.len())` array in dB.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::{FrequencyWeighting, multi_frequency_weighting};
/// use FrequencyWeighting::{A, C, Z};
///
/// let w = multi_frequency_weighting(&[50.0, 500.0, 5000.0], &[Z, A, C], Some(-80.0));
/// assert_eq!(w.dim(), (3, 3));
/// assert_eq!(w[[0, 1]], 0.0);
/// ```
#[must_use]
pub fn multi_frequency_weighting(
    frequencies: &[f64],
    kinds: &[FrequencyWeighting],
    min_db: Option<f64>,
) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((kinds.len(), frequencies.len()));
    for (row, &kind) in kinds.iter().enumerate() {
        for (o, w) in
            out.row_mut(row)
                .iter_mut()
                .zip(frequency_weighting(frequencies, kind, min_db))
        {
            *o = w;
        }
    }
    out
}

/// Converts a power spectrogram to perceptually weighted dB, equivalent to
/// `librosa.perceptual_weighting(S, frequencies, kind=kind, **db_kwargs)`.
///
/// The result is `frequency_weighting(f, kind)[k] + power_to_db(S, db)[k, t]`, with
/// the weighting clipped at librosa's default `-80` dB and the `top_db` floor applied
/// by `power_to_db` before the weighting is added.
///
/// # Arguments
///
/// * `s` - `(channels, bins, frames)` power spectrogram (e.g. `|STFT|^2`, a CQT
///   power, ...) of element type `F` (`f32` or `f64`).
/// * `frequencies` - Centre frequency of each bin in Hz (length `bins`).
/// * `kind` - Weighting curve (librosa default A).
/// * `db` - dB conversion settings. [`DbConfig::librosa_power`] gives librosa's
///   `power_to_db` defaults (`ref = 1`, `amin = 1e-10`, `top_db = 80`);
///   [`DbConfig::default`] applies no `top_db` floor.
///
/// # Returns
///
/// A `(channels, bins, frames)` array in dB, of element type `F`. The weighting
/// curve is computed in `f64` and rounded to `F` once.
///
/// # Errors
///
/// [`ParameterError`] if `frequencies.len()` differs from the bin count, or the dB
/// configuration is invalid.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::{FrequencyWeighting, perceptual_weighting};
/// use audio_samples::operations::spectrum::DbConfig;
/// use ndarray::Array3;
///
/// let s = Array3::<f64>::ones((1, 3, 2));
/// let w = perceptual_weighting(s.view(), &[100.0, 1000.0, 5000.0], FrequencyWeighting::A, &DbConfig::default())
///     .unwrap();
/// assert!(w[[0, 0, 0]] < -15.0); // 100 Hz is strongly attenuated by A-weighting
/// ```
pub fn perceptual_weighting<F: AudioFloat>(
    s: ArrayView3<'_, F>,
    frequencies: &[f64],
    kind: FrequencyWeighting,
    db: &DbConfig,
) -> AudioSampleResult<Array3<F>> {
    let n_bins = s.dim().1;
    check_freq("perceptual_weighting", frequencies, n_bins)?;
    let offset = freq_as::<F>(&frequency_weighting(frequencies, kind, Some(-80.0)));
    let mut out = power_to_db(&s, db)?;
    for mut ch in out.outer_iter_mut() {
        for (mut row, &o) in ch.outer_iter_mut().zip(&offset) {
            row.mapv_inplace(|v| o + v);
        }
    }
    Ok(out)
}

// =============================================================================
// Reassigned spectrogram
// =============================================================================

/// Power threshold below which reassigned estimates are discarded (librosa's
/// `ref_power`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReassignRefPower {
    /// A fixed power threshold (librosa default `1e-6`); `0` disables thresholding.
    Value(f64),
    /// The maximum power `|S|^2` over the whole spectrogram (librosa
    /// `ref_power=numpy.max`).
    Max,
}

/// Configuration of the time-frequency reassigned spectrogram, mirroring
/// `librosa.reassigned_spectrogram`.
///
/// ## Purpose
///
/// Reassignment moves each STFT cell's energy to its local centre of gravity in
/// time and frequency, sharpening the representation of tones and transients.
///
/// ## Intended usage
///
/// Pass to [`AudioFeatures::reassigned_spectrogram`]. [`Default`] is librosa's
/// default: the default STFT (`n_fft = 2048`, hop `win_length / 4`, periodic Hann,
/// centred, zero padding), both reassignments, `ref_power = 1e-6`,
/// `fill_nan = false`, `clip = true`.
///
/// ## Invariants
///
/// Checked by [`ReassignedSpectrogramConfig::validate`]: at least one of
/// `reassign_frequencies` / `reassign_times`; a `Value` threshold finite and
/// non-negative.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::features::ReassignedSpectrogramConfig;
///
/// let cfg = ReassignedSpectrogramConfig::new().with_fill_nan(true);
/// assert!(cfg.validate().is_ok());
/// assert!(cfg.with_reassign_frequencies(false).with_reassign_times(false).validate().is_err());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ReassignedSpectrogramConfig {
    /// STFT parameters.
    pub stft: StftConfig,
    /// Reassign frequencies (instantaneous frequency); otherwise bin centres.
    pub reassign_frequencies: bool,
    /// Reassign times (group delay); otherwise frame centres.
    pub reassign_times: bool,
    /// Cells with power `|S|^2` below this threshold get `NaN` estimates.
    pub ref_power: ReassignRefPower,
    /// Replace `NaN` estimates (thresholded or undefined) with the bin frequency /
    /// frame time.
    pub fill_nan: bool,
    /// Clip frequencies to `[0, sr / 2]` and times to `[0, duration]`.
    pub clip: bool,
}

impl Default for ReassignedSpectrogramConfig {
    fn default() -> Self {
        Self {
            stft: StftConfig::default(),
            reassign_frequencies: true,
            reassign_times: true,
            ref_power: ReassignRefPower::Value(1e-6),
            fill_nan: false,
            clip: true,
        }
    }
}

impl ReassignedSpectrogramConfig {
    /// Returns the default configuration (same as [`Default`], which equals
    /// librosa 1.0's defaults).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// librosa 1.0's `reassigned_spectrogram` defaults. Currently the same as
    /// [`Default`]; use this constructor where agreement with librosa is the
    /// requirement.
    #[must_use]
    pub fn librosa() -> Self {
        Self {
            stft: StftConfig::librosa(),
            ..Self::default()
        }
    }

    /// Sets the STFT configuration.
    #[must_use]
    pub fn with_stft(mut self, stft: StftConfig) -> Self {
        self.stft = stft;
        self
    }

    /// Sets frequency reassignment.
    #[must_use]
    pub fn with_reassign_frequencies(mut self, on: bool) -> Self {
        self.reassign_frequencies = on;
        self
    }

    /// Sets time reassignment.
    #[must_use]
    pub fn with_reassign_times(mut self, on: bool) -> Self {
        self.reassign_times = on;
        self
    }

    /// Sets the power threshold.
    #[must_use]
    pub fn with_ref_power(mut self, ref_power: ReassignRefPower) -> Self {
        self.ref_power = ref_power;
        self
    }

    /// Sets `NaN` filling.
    #[must_use]
    pub fn with_fill_nan(mut self, fill_nan: bool) -> Self {
        self.fill_nan = fill_nan;
        self
    }

    /// Sets clipping.
    #[must_use]
    pub fn with_clip(mut self, clip: bool) -> Self {
        self.clip = clip;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if neither reassignment is enabled, a `Value` threshold is
    /// negative or not finite, or the STFT configuration is invalid.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if !self.reassign_frequencies && !self.reassign_times {
            return Err(config_err(
                "reassigned_spectrogram",
                "reassign_frequencies or reassign_times must be true",
            ));
        }
        if let ReassignRefPower::Value(v) = self.ref_power {
            if !(v.is_finite() && v >= 0.0) {
                return Err(range_err(
                    "ref_power",
                    v,
                    "0",
                    "finite",
                    "ref_power must be non-negative",
                ));
            }
        }
        self.stft.validate()
    }
}

/// Output of [`AudioFeatures::reassigned_spectrogram`]: per-cell reassigned
/// frequencies and times plus the STFT magnitudes, each `(channels, bins, frames)`
/// of element type `F` (the audio's [`AudioFeatures::Float`]).
///
/// Thresholded or undefined cells hold `NaN` unless `fill_nan` was set.
#[derive(Debug, Clone, PartialEq)]
pub struct ReassignedSpectrogram<F = f64> {
    /// Reassigned frequency of each cell in Hz.
    pub freqs: Array3<F>,
    /// Reassigned time of each cell in seconds.
    pub times: Array3<F>,
    /// STFT magnitude `|S|` of each cell.
    pub mags: Array3<F>,
}

/// Complex division with numpy's algorithm (Smith's method, and numpy's handling of
/// a zero divisor), so `0 / 0` and `x / 0` give the same NaN / inf as librosa.
fn np_cdiv<F: AudioFloat>(a: Complex<F>, b: Complex<F>) -> Complex<F> {
    let (ar, ai, br, bi) = (a.re, a.im, b.re, b.im);
    let abs_br = br.abs();
    let abs_bi = bi.abs();
    if abs_br >= abs_bi {
        if abs_br == F::zero() && abs_bi == F::zero() {
            Complex::new(ar / abs_br, ai / abs_bi)
        } else {
            let rat = bi / br;
            let scl = F::one() / (br + bi * rat);
            Complex::new((ar + ai * rat) * scl, (ai - ar * rat) * scl)
        }
    } else {
        let rat = br / bi;
        let scl = F::one() / (bi + br * rat);
        Complex::new((ar * rat + ai) * scl, (ai * rat - ar) * scl)
    }
}

fn reassigned_rows<F: AudioFloat>(
    y: ArrayView2<'_, F>,
    sample_rate: f64,
    config: &ReassignedSpectrogramConfig,
) -> AudioSampleResult<ReassignedSpectrogram<F>> {
    config.validate()?;
    validate_sample_rate(sample_rate)?;
    let n_fft = config.stft.n_fft.get();
    let hop = config.stft.effective_hop_length()?.get();
    let window = padded_window(
        &config.stft.window,
        config.stft.win_length,
        config.stft.n_fft,
    )?;
    let s_h = stft(y, &config.stft)?;
    let (n_ch, n_bins, n_frames) = s_h.dim();
    let custom = |w: Vec<f64>| StftConfig {
        window: Window::Custom(w),
        win_length: None,
        ..config.stft.clone()
    };

    // Bin frequencies and frame times are computed in f64 and rounded to F once.
    let bin_freqs = freq_as::<F>(&rfft_frequencies(n_fft, sample_rate));
    let offset = if config.stft.center { 0 } else { n_fft / 2 };
    let frame_times: Vec<F> = (0..n_frames)
        .map(|t| F::of((t * hop + offset) as f64 / sample_rate))
        .collect();
    let (sr, nyquist) = (F::of(sample_rate), F::of(sample_rate / 2.0));

    let mags = s_h.mapv(|z| z.norm());
    let ref_p = match config.ref_power {
        ReassignRefPower::Value(v) => F::of(v),
        ReassignRefPower::Max => mags.iter().fold(F::neg_infinity(), |m, &a| m.max(a * a)),
    };
    let thresh = ref_p.sqrt();

    let freqs = if config.reassign_frequencies {
        // Cyclic gradient of the (padded) window: (w[n + 1] - w[n - 1]) / 2.
        let w = window.as_slice().expect("contiguous window");
        let deriv: Vec<f64> = (0..n_fft)
            .map(|n| (w[(n + 1) % n_fft] - w[(n + n_fft - 1) % n_fft]) / 2.0)
            .collect();
        let s_dh = stft(y, &custom(deriv))?;
        let scale = F::of(0.5 * sample_rate / std::f64::consts::PI);
        let mut f = Array3::<F>::zeros((n_ch, n_bins, n_frames));
        ndarray::Zip::indexed(&mut f)
            .and(&s_dh)
            .and(&s_h)
            .and(&mags)
            .for_each(|(_, k, _), v, &dh, &h, &m| {
                let correction = -np_cdiv(dh, h).im;
                let mut x = bin_freqs[k] + correction * scale;
                if ref_p > F::zero() && m < thresh {
                    x = F::nan();
                }
                if config.fill_nan && x.is_nan() {
                    x = bin_freqs[k];
                }
                if config.clip {
                    x = x.clamp(F::zero(), nyquist);
                }
                *v = x;
            });
        f
    } else {
        Array3::from_shape_fn((n_ch, n_bins, n_frames), |(_, k, _)| bin_freqs[k])
    };

    let times = if config.reassign_times {
        let half = (n_fft / 2) as f64;
        let w = window.as_slice().expect("contiguous window");
        let weighted: Vec<f64> = (0..n_fft)
            .map(|n| {
                let tn = if n_fft % 2 == 1 {
                    n as f64 - half
                } else {
                    0.5 - half + n as f64
                };
                w[n] * tn
            })
            .collect();
        let s_th = stft(y, &custom(weighted))?;
        let duration = F::of(y.ncols() as f64 / sample_rate);
        let mut tt = Array3::<F>::zeros((n_ch, n_bins, n_frames));
        ndarray::Zip::indexed(&mut tt)
            .and(&s_th)
            .and(&s_h)
            .and(&mags)
            .for_each(|(_, _, t), v, &th, &h, &m| {
                let correction = np_cdiv(th, h).re;
                let mut x = frame_times[t] + correction / sr;
                if ref_p > F::zero() && m < thresh {
                    x = F::nan();
                }
                if config.fill_nan && x.is_nan() {
                    x = frame_times[t];
                }
                if config.clip {
                    x = x.clamp(F::zero(), duration);
                }
                *v = x;
            });
        tt
    } else {
        Array3::from_shape_fn((n_ch, n_bins, n_frames), |(_, _, t)| frame_times[t])
    };

    Ok(ReassignedSpectrogram { freqs, times, mags })
}

// =============================================================================
// The AudioFeatures trait
// =============================================================================

/// librosa-compatible frame-wise features on [`AudioSamples`].
///
/// Every method works on all channels: outputs are shaped
/// `(channels, n_features, n_frames)` (mono gives one channel). Outputs have the
/// element type [`Float`](Self::Float): `f32` for `u8`, `i16`, `I24` and `f32`
/// audio, `f64` for `i32` and `f64` audio. The computation runs in that type.
/// Each config's `librosa()` constructor reproduces librosa 1.0's defaults. See
/// the [module documentation](self) for the librosa correspondence and the
/// `*_from_spectrogram` forms that reuse a precomputed spectrogram.
///
/// The whole-signal scalar descriptors of [`AudioStatistics`](crate::AudioStatistics)
/// (`spectral_centroid`, `zero_crossing_rate`, ...) are unchanged; they summarise a
/// signal with one number and are different features.
pub trait AudioFeatures {
    /// The float type of the outputs and of the computation: the sample type's
    /// [`AudioSample::Float`](crate::AudioSample::Float).
    type Float: AudioFloat;

    /// Computes the spectral centroid of every frame, equivalent to
    /// `librosa.feature.spectral_centroid(y=y, sr=sr, ...)`.
    ///
    /// # Arguments
    ///
    /// * `config` - STFT parameters ([`SpectralCentroidConfig::default`] = librosa).
    ///
    /// # Returns
    ///
    /// A `(channels, 1, frames)` array in Hz; see
    /// [`spectral_centroid_from_spectrogram`].
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
    ///
    /// [`ParameterError`] if the configuration is invalid or the signal is too short
    /// for an uncentred STFT.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::features::{AudioFeatures, SpectralCentroidConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let x = sine_wave::<f64>(2000.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
    /// let c = x.spectral_centroid_frames(&SpectralCentroidConfig::default()).unwrap();
    /// assert_eq!(c.dim(), (1, 1, 22));
    /// assert!((c[[0, 0, 10]] - 2000.0).abs() < 20.0);
    /// ```
    fn spectral_centroid_frames(
        &self,
        config: &SpectralCentroidConfig,
    ) -> AudioSampleResult<Array3<Self::Float>>;

    /// Computes the order-`p` spectral bandwidth of every frame, equivalent to
    /// `librosa.feature.spectral_bandwidth(y=y, sr=sr, p=p, norm=norm, ...)`.
    ///
    /// # Arguments
    ///
    /// * `config` - STFT, `p` and `norm` ([`SpectralBandwidthConfig::default`] =
    ///   librosa).
    ///
    /// # Returns
    ///
    /// A `(channels, 1, frames)` array in Hz; see
    /// [`spectral_bandwidth_from_spectrogram`].
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
    ///
    /// [`ParameterError`] if the configuration is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::features::{AudioFeatures, SpectralBandwidthConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let x = sine_wave::<f64>(2000.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
    /// let bw = x.spectral_bandwidth_frames(&SpectralBandwidthConfig::default()).unwrap();
    /// assert!(bw[[0, 0, 10]] < 100.0); // a pure tone is narrow
    /// ```
    fn spectral_bandwidth_frames(
        &self,
        config: &SpectralBandwidthConfig,
    ) -> AudioSampleResult<Array3<Self::Float>>;

    /// Computes the spectral rolloff of every frame, equivalent to
    /// `librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=..., ...)`.
    ///
    /// # Arguments
    ///
    /// * `config` - STFT and `roll_percent` ([`SpectralRolloffConfig::default`] =
    ///   librosa).
    ///
    /// # Returns
    ///
    /// A `(channels, 1, frames)` array in Hz; see
    /// [`spectral_rolloff_from_spectrogram`].
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
    ///
    /// [`ParameterError`] if the configuration is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::features::{AudioFeatures, SpectralRolloffConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let x = sine_wave::<f64>(1000.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
    /// let r = x.spectral_rolloff_frames(&SpectralRolloffConfig::default()).unwrap();
    /// assert!((r[[0, 0, 10]] - 1000.0).abs() < 30.0);
    /// ```
    fn spectral_rolloff_frames(
        &self,
        config: &SpectralRolloffConfig,
    ) -> AudioSampleResult<Array3<Self::Float>>;

    /// Computes the spectral flatness of every frame, equivalent to
    /// `librosa.feature.spectral_flatness(y=y, amin=..., power=..., ...)`.
    ///
    /// # Arguments
    ///
    /// * `config` - STFT, `amin` and `power` ([`SpectralFlatnessConfig::default`] =
    ///   librosa).
    ///
    /// # Returns
    ///
    /// A `(channels, 1, frames)` array in `(0, 1]`; see
    /// [`spectral_flatness_from_spectrogram`].
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
    ///
    /// [`ParameterError`] if the configuration is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::features::{AudioFeatures, SpectralFlatnessConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let x = sine_wave::<f64>(1000.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
    /// let f = x.spectral_flatness_frames(&SpectralFlatnessConfig::default()).unwrap();
    /// assert!(f[[0, 0, 10]] < 0.01); // tonal
    /// ```
    fn spectral_flatness_frames(
        &self,
        config: &SpectralFlatnessConfig,
    ) -> AudioSampleResult<Array3<Self::Float>>;

    /// Computes the spectral contrast of every frame, equivalent to
    /// `librosa.feature.spectral_contrast(y=y, sr=sr, ...)`.
    ///
    /// # Arguments
    ///
    /// * `config` - STFT and band parameters ([`SpectralContrastConfig::default`] =
    ///   librosa).
    ///
    /// # Returns
    ///
    /// A `(channels, n_bands + 1, frames)` array; see
    /// [`spectral_contrast_from_spectrogram`].
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
    ///
    /// [`ParameterError`] if the configuration is invalid for the sample rate or a
    /// band contains no bins.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::features::{AudioFeatures, SpectralContrastConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let x = sine_wave::<f64>(1000.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
    /// let c = x.spectral_contrast_frames(&SpectralContrastConfig::default()).unwrap();
    /// assert_eq!(c.dim(), (1, 7, 22));
    /// ```
    fn spectral_contrast_frames(
        &self,
        config: &SpectralContrastConfig,
    ) -> AudioSampleResult<Array3<Self::Float>>;

    /// Fits a polynomial to every magnitude-spectrum frame, equivalent to
    /// `librosa.feature.poly_features(y=y, sr=sr, order=..., ...)`.
    ///
    /// # Arguments
    ///
    /// * `config` - STFT and `order` ([`PolyFeaturesConfig::default`] = librosa).
    ///
    /// # Returns
    ///
    /// A `(channels, order + 1, frames)` array of coefficients, highest degree first;
    /// see [`poly_features_from_spectrogram`].
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
    ///
    /// [`ParameterError`] if the configuration is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::features::{AudioFeatures, PolyFeaturesConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let x = sine_wave::<f64>(1000.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
    /// let p = x.poly_features(&PolyFeaturesConfig::new().with_order(2)).unwrap();
    /// assert_eq!(p.dim(), (1, 3, 22));
    /// ```
    fn poly_features(&self, config: &PolyFeaturesConfig) -> AudioSampleResult<Array3<Self::Float>>;

    /// Computes the RMS of every frame, equivalent to
    /// `librosa.feature.rms(y=y, frame_length=..., hop_length=..., center=..., pad_mode=...)`.
    ///
    /// # Arguments
    ///
    /// * `config` - Framing ([`RmsConfig::default`] = librosa 1.0).
    ///
    /// # Returns
    ///
    /// A `(channels, 1, frames)` array with
    /// `frames = 1 + (len + 2 (frame_length / 2) - frame_length) / hop_length` when
    /// centred, `1 + (len - frame_length) / hop_length` otherwise.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
    ///
    /// [`ParameterError`] if the (padded) signal is shorter than `frame_length`.
    ///
    /// # Differences from librosa
    ///
    /// librosa computes the mean square in `float32` by default
    /// (`dtype=np.float32`) and returns `float32`, whatever the input type. Here
    /// the computation runs in [`Float`](Self::Float): `f32` audio matches
    /// librosa's default to `f32` round-off, and `f64` audio matches librosa with
    /// `dtype=np.float64`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::features::{AudioFeatures, RmsConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let x = sine_wave::<f64>(1000.0, Duration::from_secs(1), sample_rate!(22050), 1.0);
    /// let r = x.rms_frames(&RmsConfig::default()).unwrap();
    /// assert_eq!(r.dim(), (1, 1, 44));
    /// assert!((r[[0, 0, 20]] - 0.5f64.sqrt()).abs() < 1e-3);
    /// ```
    fn rms_frames(&self, config: &RmsConfig) -> AudioSampleResult<Array3<Self::Float>>;

    /// Computes the zero-crossing rate of every frame, equivalent to
    /// `librosa.feature.zero_crossing_rate(y, frame_length=..., hop_length=..., center=..., ...)`.
    ///
    /// Each frame's rate is the number of sign changes between consecutive samples
    /// (plus one if `pad`) divided by `frame_length`.
    ///
    /// # Arguments
    ///
    /// * `config` - Framing and crossing rules ([`ZeroCrossingRateConfig::default`] =
    ///   librosa).
    ///
    /// # Returns
    ///
    /// A `(channels, 1, frames)` array of rates in `[0, 1)` (crossings per sample).
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
    ///
    /// [`ParameterError`] if the configuration is invalid or the (padded) signal is
    /// shorter than `frame_length`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::features::{AudioFeatures, ZeroCrossingRateConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// // 1 kHz at 22.05 kHz: 2 crossings per 22.05 samples.
    /// let x = sine_wave::<f64>(1000.0, Duration::from_secs(1), sample_rate!(22050), 0.5);
    /// let z = x.zero_crossing_rate_frames(&ZeroCrossingRateConfig::default()).unwrap();
    /// assert!((z[[0, 0, 20]] - 2000.0 / 22050.0).abs() < 2e-3);
    /// ```
    fn zero_crossing_rate_frames(
        &self,
        config: &ZeroCrossingRateConfig,
    ) -> AudioSampleResult<Array3<Self::Float>>;

    /// Computes the time-frequency reassigned spectrogram, equivalent to
    /// `librosa.reassigned_spectrogram(y=y, sr=sr, ...)`.
    ///
    /// Frequencies are corrected with an STFT using the window derivative
    /// (`f = f_k - Im(S_dh / S_h) sr / (2 pi)`), times with an STFT using the
    /// time-weighted window (`t = t_frame + Re(S_th / S_h) / sr`) (Flandrin, Auger
    /// and Chassande-Mottin 2002, eqs. 5.20 and 5.23).
    ///
    /// # Arguments
    ///
    /// * `config` - STFT and reassignment options
    ///   ([`ReassignedSpectrogramConfig::default`] = librosa).
    ///
    /// # Returns
    ///
    /// A [`ReassignedSpectrogram`] with `(channels, bins, frames)` frequencies,
    /// times and magnitudes.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
    ///
    /// [`ParameterError`] if the configuration is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::features::{AudioFeatures, ReassignedSpectrogramConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let x = sine_wave::<f64>(1001.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
    /// let r = x.reassigned_spectrogram(&ReassignedSpectrogramConfig::default()).unwrap();
    /// // The bin nearest 1001 Hz (bin 93 = 1001.3 Hz) is reassigned to ~1001 Hz.
    /// assert!((r.freqs[[0, 93, 10]] - 1001.0).abs() < 1.0);
    /// ```
    ///
    /// # References
    ///
    /// P. Flandrin, F. Auger and E. Chassande-Mottin, "Time-frequency reassignment:
    /// from principles to algorithms," in *Applications in Time-Frequency Signal
    /// Processing*, CRC Press, 2002; librosa `reassigned_spectrogram`.
    fn reassigned_spectrogram(
        &self,
        config: &ReassignedSpectrogramConfig,
    ) -> AudioSampleResult<ReassignedSpectrogram<Self::Float>>;
}

impl<T> AudioFeatures for AudioSamples<'_, T>
where
    T: StandardSample,
{
    type Float = T::Float;

    fn spectral_centroid_frames(
        &self,
        config: &SpectralCentroidConfig,
    ) -> AudioSampleResult<Array3<T::Float>> {
        config.validate()?;
        let s = spectrogram_rows(finite_float_rows(self)?.view(), &config.stft, 1.0)?;
        let freq = rfft_frequencies(config.stft.n_fft.get(), self.sample_rate_hz());
        spectral_centroid_from_spectrogram(s.view(), &freq)
    }

    fn spectral_bandwidth_frames(
        &self,
        config: &SpectralBandwidthConfig,
    ) -> AudioSampleResult<Array3<T::Float>> {
        config.validate()?;
        let s = spectrogram_rows(finite_float_rows(self)?.view(), &config.stft, 1.0)?;
        let freq = rfft_frequencies(config.stft.n_fft.get(), self.sample_rate_hz());
        spectral_bandwidth_from_spectrogram(s.view(), &freq, None, config)
    }

    fn spectral_rolloff_frames(
        &self,
        config: &SpectralRolloffConfig,
    ) -> AudioSampleResult<Array3<T::Float>> {
        config.validate()?;
        let s = spectrogram_rows(finite_float_rows(self)?.view(), &config.stft, 1.0)?;
        let freq = rfft_frequencies(config.stft.n_fft.get(), self.sample_rate_hz());
        spectral_rolloff_from_spectrogram(s.view(), &freq, config)
    }

    fn spectral_flatness_frames(
        &self,
        config: &SpectralFlatnessConfig,
    ) -> AudioSampleResult<Array3<T::Float>> {
        config.validate()?;
        let s = spectrogram_rows(finite_float_rows(self)?.view(), &config.stft, 1.0)?;
        spectral_flatness_from_spectrogram(s.view(), config)
    }

    fn spectral_contrast_frames(
        &self,
        config: &SpectralContrastConfig,
    ) -> AudioSampleResult<Array3<T::Float>> {
        let sr = self.sample_rate_hz();
        config.validate(sr)?;
        let s = spectrogram_rows(finite_float_rows(self)?.view(), &config.stft, 1.0)?;
        let freq = rfft_frequencies(config.stft.n_fft.get(), sr);
        spectral_contrast_from_spectrogram(s.view(), &freq, sr, config)
    }

    fn poly_features(&self, config: &PolyFeaturesConfig) -> AudioSampleResult<Array3<T::Float>> {
        config.validate()?;
        let s = spectrogram_rows(finite_float_rows(self)?.view(), &config.stft, 1.0)?;
        let freq = rfft_frequencies(config.stft.n_fft.get(), self.sample_rate_hz());
        poly_features_from_spectrogram(s.view(), &freq, config)
    }

    fn rms_frames(&self, config: &RmsConfig) -> AudioSampleResult<Array3<T::Float>> {
        config.validate()?;
        rms_rows(finite_float_rows(self)?.view(), config)
    }

    fn zero_crossing_rate_frames(
        &self,
        config: &ZeroCrossingRateConfig,
    ) -> AudioSampleResult<Array3<T::Float>> {
        zcr_rows(finite_float_rows(self)?.view(), config)
    }

    fn reassigned_spectrogram(
        &self,
        config: &ReassignedSpectrogramConfig,
    ) -> AudioSampleResult<ReassignedSpectrogram<T::Float>> {
        reassigned_rows(
            finite_float_rows(self)?.view(),
            self.sample_rate_hz(),
            config,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioTypeConversion;
    use crate::operations::spectrum::AudioSpectrum;
    use crate::sample_rate;
    use ndarray::{Array1, array, s};

    /// librosa's zero_crossing_rate computed literally: edge-pad, frame,
    /// count the crossings of every frame.
    fn zcr_reference(x: &[f64], cfg: &ZeroCrossingRateConfig) -> Vec<f64> {
        let fl = cfg.frame_length.get();
        let hop = cfg.hop_length.get();
        let pad = if cfg.center { fl / 2 } else { 0 };
        let xp = crate::operations::spectrum::pad_signal(x, pad, StftPadMode::Edge);
        let n_frames = 1 + (xp.len() - fl) / hop;
        let covered = (n_frames - 1) * hop + fl;
        let thr = match cfg.ref_magnitude {
            ZeroCrossingReference::None => cfg.threshold,
            ZeroCrossingReference::Value(v) => cfg.threshold * v,
            ZeroCrossingReference::Max => {
                cfg.threshold
                    * xp[..covered]
                        .iter()
                        .fold(f64::NEG_INFINITY, |m, v| m.max(v.abs()))
            }
        };
        let clean = |v: f64| if -thr <= v && v <= thr { 0.0 } else { v };
        (0..n_frames)
            .map(|t| {
                let f = &xp[t * hop..t * hop + fl];
                let mut n = usize::from(cfg.pad);
                for w in f.windows(2) {
                    let (a, b) = (clean(w[0]), clean(w[1]));
                    n += usize::from(if cfg.zero_pos {
                        a.is_sign_negative() != b.is_sign_negative()
                    } else {
                        np_sign(a) != np_sign(b)
                    });
                }
                n as f64 / fl as f64
            })
            .collect()
    }

    #[test]
    fn zcr_matches_the_framed_count() {
        let mut x = noise(3000, 5);
        x[..200].fill(0.0);
        x[1000..1300].fill(0.0);
        for (fl, hop) in [(2048, 512), (100, 30), (64, 64), (16, 50)] {
            for center in [true, false] {
                for (zero_pos, pad) in [(true, false), (false, true)] {
                    for r in [
                        ZeroCrossingReference::None,
                        ZeroCrossingReference::Max,
                        ZeroCrossingReference::Value(2.0),
                    ] {
                        let cfg = ZeroCrossingRateConfig::new()
                            .with_frame_length(NonZeroUsize::new(fl).unwrap())
                            .with_hop_length(NonZeroUsize::new(hop).unwrap())
                            .with_center(center)
                            .with_threshold(0.1)
                            .with_ref_magnitude(r)
                            .with_pad(pad)
                            .with_zero_pos(zero_pos);
                        let rows = Array2::from_shape_vec((1, x.len()), x.clone()).unwrap();
                        let got = zcr_rows(rows.view(), &cfg).unwrap();
                        let want = zcr_reference(&x, &cfg);
                        assert_eq!(got.iter().copied().collect::<Vec<_>>(), want);
                    }
                }
            }
        }
    }
    #[test]
    fn huge_max_filter_equals_one_period() {
        let s = Array3::from_shape_fn((1, 6, 4), |(_, k, t)| ((k * 7 + t * 3) % 13) as f64);
        let one_period = maximum_filter_bins(&s.view(), 12);
        for size in [13, 500, 1 << 40] {
            assert_eq!(
                maximum_filter_bins(&s.view(), size),
                one_period,
                "size {size}"
            );
        }
        for t in 0..4 {
            let col_max = (0..6)
                .map(|k| s[[0, k, t]])
                .fold(f64::NEG_INFINITY, f64::max);
            assert!((0..6).all(|k| one_period[[0, k, t]] == col_max));
        }
    }

    fn mono(x: Vec<f64>) -> AudioSamples<'static, f64> {
        AudioSamples::new_mono(Array1::from(x), sample_rate!(22050)).unwrap()
    }

    fn noise(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn shapes_mono_and_stereo() {
        let x = noise(5000, 1);
        let m = mono(x.clone());
        let mut data = Array2::zeros((2, 5000));
        for i in 0..5000 {
            data[[0, i]] = x[i];
            data[[1, i]] = 0.5 * x[i];
        }
        let st = AudioSamples::new_multi_channel(data, sample_rate!(22050)).unwrap();
        let frames = 1 + 5000 / 512;
        assert_eq!(
            m.spectral_centroid_frames(&SpectralCentroidConfig::default())
                .unwrap()
                .dim(),
            (1, 1, frames)
        );
        let c = st
            .spectral_contrast_frames(&SpectralContrastConfig::default())
            .unwrap();
        assert_eq!(c.dim(), (2, 7, frames));
        assert_eq!(
            st.rms_frames(&RmsConfig::default()).unwrap().dim(),
            (2, 1, frames)
        );
        let z = st
            .zero_crossing_rate_frames(&ZeroCrossingRateConfig::default())
            .unwrap();
        assert_eq!(z.dim(), (2, 1, frames));
        // Scaling a channel does not change its ZCR, centroid or flatness.
        assert_eq!(z.index_axis(Axis(0), 0), z.index_axis(Axis(0), 1));
        let cen = st
            .spectral_centroid_frames(&SpectralCentroidConfig::default())
            .unwrap();
        for t in 0..frames {
            assert!((cen[[0, 0, t]] - cen[[1, 0, t]]).abs() < 1e-9 * cen[[0, 0, t]]);
        }
        // RMS scales linearly.
        let r = st.rms_frames(&RmsConfig::default()).unwrap();
        for t in 0..frames {
            assert!((r[[0, 0, t]] - 2.0 * r[[1, 0, t]]).abs() < 1e-12);
        }
        assert_eq!(
            m.poly_features(&PolyFeaturesConfig::new().with_order(3))
                .unwrap()
                .dim(),
            (1, 4, frames)
        );
    }

    #[test]
    fn rms_matches_spectrogram_rms_for_rectangular_window() {
        // Parseval: RMS of a frame equals rms_from_spectrogram of its DFT.
        let x = noise(4096, 3);
        let audio = mono(x);
        let cfg = StftConfig::new()
            .with_window(Window::Rectangular)
            .with_hop_length(Some(nzu!(512)));
        let s = audio.spectrogram_matrix(&cfg, 1.0).unwrap();
        let a = rms_from_spectrogram(s.view(), nzu!(2048)).unwrap();
        let b = audio.rms_frames(&RmsConfig::default()).unwrap();
        for (p, q) in a.iter().zip(b.iter()) {
            assert!((p - q).abs() < 1e-12, "{p} vs {q}");
        }
        assert!(rms_from_spectrogram(s.view(), nzu!(1000)).is_err());
    }

    #[test]
    fn zero_crossing_rules() {
        // librosa zero_crossings examples: [1, 1e-12, -1e-12, 1e-12, -1] has one crossing.
        let x = vec![1.0, 1e-12, -1e-12, 1e-12, -1.0];
        let cfg = ZeroCrossingRateConfig::new()
            .with_frame_length(nzu!(5))
            .with_hop_length(nzu!(1))
            .with_center(false);
        let z = zcr_rows(Array2::from_shape_vec((1, 5), x).unwrap().view(), &cfg).unwrap();
        assert_eq!(z[[0, 0, 0]], 1.0 / 5.0);
        // zero_pos = false: 1 -> 0 -> 1 counts twice.
        let y = Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 1.0]).unwrap();
        let c3 = cfg.with_frame_length(nzu!(3));
        assert_eq!(zcr_rows(y.view(), &c3).unwrap()[[0, 0, 0]], 0.0);
        assert_eq!(
            zcr_rows(y.view(), &c3.clone().with_zero_pos(false)).unwrap()[[0, 0, 0]],
            2.0 / 3.0
        );
        assert_eq!(
            zcr_rows(y.view(), &c3.clone().with_pad(true)).unwrap()[[0, 0, 0]],
            1.0 / 3.0
        );
        // Threshold relative to the maximum.
        let w = Array2::from_shape_vec((1, 3), vec![1.0, -0.1, 1.0]).unwrap();
        let rel = c3
            .with_threshold(0.2)
            .with_ref_magnitude(ZeroCrossingReference::Max);
        assert_eq!(zcr_rows(w.view(), &rel).unwrap()[[0, 0, 0]], 0.0);
    }

    #[test]
    fn short_input_errors() {
        let m = mono(vec![0.1; 100]);
        let rms_cfg = RmsConfig::default().with_center(false);
        assert!(m.rms_frames(&rms_cfg).is_err());
        assert!(
            m.zero_crossing_rate_frames(&ZeroCrossingRateConfig::default().with_center(false))
                .is_err()
        );
        assert!(m.rms_frames(&RmsConfig::default()).is_ok());
    }

    #[test]
    fn invalid_parameters_error() {
        let m = mono(noise(4096, 5));
        assert!(
            m.spectral_rolloff_frames(&SpectralRolloffConfig::new().with_roll_percent(0.0))
                .is_err()
        );
        assert!(
            m.spectral_flatness_frames(&SpectralFlatnessConfig::new().with_amin(0.0))
                .is_err()
        );
        assert!(
            m.spectral_bandwidth_frames(&SpectralBandwidthConfig::new().with_p(0.0))
                .is_err()
        );
        assert!(
            m.spectral_contrast_frames(&SpectralContrastConfig::new().with_quantile(1.0))
                .is_err()
        );
        assert!(
            m.spectral_contrast_frames(&SpectralContrastConfig::new().with_n_bands(nzu!(7)))
                .is_err()
        );
        // n_fft too small: band 0 has a single bin, which librosa turns into NaN.
        let tiny = SpectralContrastConfig::new().with_stft(StftConfig::new().with_n_fft(nzu!(64)));
        assert!(m.spectral_contrast_frames(&tiny).is_err());
        assert!(
            m.poly_features(
                &PolyFeaturesConfig::new()
                    .with_order(9)
                    .with_stft(StftConfig::new().with_n_fft(nzu!(8)))
            )
            .is_err()
        );
        assert!(
            pcen(
                Array3::<f64>::zeros((1, 2, 3)).view(),
                22050.0,
                &PcenConfig::librosa().with_eps(0.0)
            )
            .is_err()
        );
        assert!(
            pcen(
                Array3::<f64>::zeros((1, 2, 3)).view(),
                22050.0,
                &PcenConfig::librosa().with_b(Some(1.5))
            )
            .is_err()
        );
        let neg = Array3::from_elem((1, 3, 2), -1.0);
        assert!(spectral_centroid_from_spectrogram(neg.view(), &[0.0, 1.0, 2.0]).is_err());
        assert!(
            spectral_centroid_from_spectrogram(Array3::<f64>::ones((1, 3, 2)).view(), &[0.0, 1.0])
                .is_err()
        );
        assert!(
            m.reassigned_spectrogram(
                &ReassignedSpectrogramConfig::new()
                    .with_reassign_frequencies(false)
                    .with_reassign_times(false)
            )
            .is_err()
        );
    }

    #[test]
    fn rolloff_and_flatness_hand_values() {
        let s = array![[[0.0_f64, 1.0], [0.0, 3.0], [0.0, 0.0], [0.0, 4.0]]];
        let freq = [0.0, 10.0, 20.0, 30.0];
        let r =
            spectral_rolloff_from_spectrogram(s.view(), &freq, &SpectralRolloffConfig::default())
                .unwrap();
        assert_eq!(r[[0, 0, 0]], 0.0); // silent frame
        assert_eq!(r[[0, 0, 1]], 30.0); // cumsum 1, 4, 4, 8 >= 6.8 first at bin 3
        let half = SpectralRolloffConfig::new().with_roll_percent(0.5);
        assert_eq!(
            spectral_rolloff_from_spectrogram(s.view(), &freq, &half).unwrap()[[0, 0, 1]],
            10.0
        );
        let f = spectral_flatness_from_spectrogram(s.view(), &SpectralFlatnessConfig::default())
            .unwrap();
        assert!((f[[0, 0, 0]] - 1.0).abs() < 1e-12); // all at amin
    }

    #[test]
    fn poly_features_recovers_polynomial() {
        let freq: Vec<f64> = (0..50).map(|k| k as f64 * 21.5).collect();
        let s = Array3::from_shape_fn((2, 50, 3), |(c, k, t)| {
            let f = freq[k];
            (c as f64 + 1.0) * (1e-5 * f * f - 0.02 * f + t as f64)
        });
        let p = poly_features_from_spectrogram(
            s.view(),
            &freq,
            &PolyFeaturesConfig::new().with_order(2),
        )
        .unwrap();
        for c in 0..2 {
            let g = c as f64 + 1.0;
            for t in 0..3 {
                assert!((p[[c, 0, t]] - g * 1e-5).abs() < 1e-15);
                assert!((p[[c, 1, t]] + g * 0.02).abs() < 1e-12);
                assert!((p[[c, 2, t]] - g * t as f64).abs() < 1e-9);
            }
        }
        // Duplicate frequencies: rank deficient.
        assert!(
            poly_features_from_spectrogram(
                Array3::<f64>::ones((1, 3, 1)).view(),
                &[1.0, 1.0, 1.0],
                &PolyFeaturesConfig::default()
            )
            .is_err()
        );
    }

    #[test]
    fn pcen_constant_input_and_streaming() {
        // Steady state: M = S everywhere when S is the unit input matching zi.
        let s = Array3::<f64>::ones((1, 2, 30));
        let cfg = PcenConfig::librosa();
        let p = pcen(s.view(), 22050.0, &cfg).unwrap();
        let g = (1e-6f64 + 1.0).powf(-0.98);
        let expected = 2f64.sqrt() * ((1.0 + g / 2.0).sqrt() - 1.0);
        for v in &p {
            assert!((v - expected).abs() < 1e-12);
        }
        let x = Array3::from_shape_fn((2, 3, 20), |(c, k, t)| (c + k * t) as f64 + 0.5);
        let whole = pcen(x.view(), 16000.0, &cfg.clone().with_max_size(nzu!(2))).unwrap();
        let cfg2 = cfg.with_max_size(nzu!(2));
        let (a, zf) =
            pcen_with_state(x.slice(s![.., .., ..7]), 16000.0, &cfg2, None, None).unwrap();
        let (b, _) = pcen_with_state(
            x.slice(s![.., .., 7..]),
            16000.0,
            &cfg2,
            None,
            Some(zf.view()),
        )
        .unwrap();
        for c in 0..2 {
            for k in 0..3 {
                for t in 0..20 {
                    let v = if t < 7 {
                        a[[c, k, t]]
                    } else {
                        b[[c, k, t - 7]]
                    };
                    assert!((v - whole[[c, k, t]]).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn maximum_filter_matches_scipy() {
        // scipy.ndimage.maximum_filter1d([5, 1, 2, 9, 3, 0, 4, 7], size) (checked in SciPy 1.18).
        let x = Array3::from_shape_vec((1, 8, 1), vec![5.0, 1.0, 2.0, 9.0, 3.0, 0.0, 4.0, 7.0])
            .unwrap();
        let expect = [
            (2, [5.0, 5.0, 2.0, 9.0, 9.0, 3.0, 4.0, 7.0]),
            (3, [5.0, 5.0, 9.0, 9.0, 9.0, 4.0, 7.0, 7.0]),
            (4, [5.0, 5.0, 9.0, 9.0, 9.0, 9.0, 7.0, 7.0]),
            (5, [5.0, 9.0, 9.0, 9.0, 9.0, 9.0, 7.0, 7.0]),
        ];
        for (size, e) in expect {
            let m = maximum_filter_bins(&x.view(), size);
            assert_eq!(
                m.iter().copied().collect::<Vec<_>>(),
                e.to_vec(),
                "size {size}"
            );
        }
    }

    #[test]
    fn weightings_basic_values() {
        let f = [0.0, 1000.0];
        assert_eq!(a_weighting(&f, None)[0], f64::NEG_INFINITY);
        for kind in [
            FrequencyWeighting::A,
            FrequencyWeighting::B,
            FrequencyWeighting::C,
            FrequencyWeighting::D,
        ] {
            let w = frequency_weighting(&f, kind, Some(-80.0));
            assert_eq!(w[0], -80.0);
            assert!(w[1].abs() < 0.1, "{kind:?}: {}", w[1]);
        }
        assert_eq!(z_weighting(&f, Some(3.0)), vec![3.0, 3.0]);
        assert!(apply_min_db(f64::NAN, Some(-80.0)).is_nan());
    }

    #[test]
    fn reassigned_on_pure_tone() {
        let sr = 22050.0;
        let x: Vec<f64> = (0..8192)
            .map(|n| (2.0 * std::f64::consts::PI * 1234.5 * n as f64 / sr).sin())
            .collect();
        let r = mono(x)
            .reassigned_spectrogram(&ReassignedSpectrogramConfig::default())
            .unwrap();
        let k = (1234.5f64 / (sr / 2048.0)).round() as usize;
        for t in 4..12 {
            assert!(
                (r.freqs[[0, k, t]] - 1234.5).abs() < 0.05,
                "{}",
                r.freqs[[0, k, t]]
            );
            let tf = (t * 512) as f64 / sr;
            assert!((r.times[[0, k, t]] - tf).abs() < 1e-4);
        }
        let filled = mono(vec![0.0; 4096])
            .reassigned_spectrogram(&ReassignedSpectrogramConfig::new().with_fill_nan(true))
            .unwrap();
        assert!(filled.freqs.iter().all(|v| v.is_finite()));
        let raw = mono(vec![0.0; 4096])
            .reassigned_spectrogram(&ReassignedSpectrogramConfig::default())
            .unwrap();
        assert!(raw.freqs.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn output_type_follows_sample_type_and_f32_agrees_with_f64() {
        let x: Vec<f64> = noise(8192, 9)
            .iter()
            .enumerate()
            .map(|(i, v)| 0.4 * (0.07 * i as f64).sin() + 0.3 * v)
            .collect();
        let a64 = mono(x);
        let a16 = a64.to_format::<i16>();
        let r = a16.to_format::<f64>();
        let c32: Array3<f32> = a16
            .spectral_centroid_frames(&SpectralCentroidConfig::default())
            .unwrap();
        let c64 = r
            .spectral_centroid_frames(&SpectralCentroidConfig::default())
            .unwrap();
        let rms32: Array3<f32> = a16.rms_frames(&RmsConfig::default()).unwrap();
        let rms64 = r.rms_frames(&RmsConfig::default()).unwrap();
        let reas: ReassignedSpectrogram<f32> = a16
            .reassigned_spectrogram(&ReassignedSpectrogramConfig::default())
            .unwrap();
        assert_eq!(reas.mags.dim().1, 1025);
        let i32_out: Array3<f64> = a64
            .to_format::<i32>()
            .rms_frames(&RmsConfig::default())
            .unwrap();
        assert_eq!(i32_out.dim(), rms64.dim());
        // f32 round-off of the FFT (about log2(2048) eps) and of the sums:
        // measured centroid 1.2e-6, rms 7.3e-7, pcen 5.6e-7 of the peak; bound
        // 4e-6 (34 f32 eps).
        let rel = |a: &Array3<f32>, b: &Array3<f64>| {
            let peak = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            a.iter()
                .zip(b.iter())
                .fold(0.0f64, |m, (&p, &q)| m.max((f64::from(p) - q).abs()))
                / peak
        };
        eprintln!(
            "f32 vs f64: centroid {:.3e}, rms {:.3e}",
            rel(&c32, &c64),
            rel(&rms32, &rms64)
        );
        assert!(rel(&c32, &c64) < 4e-6, "{:e}", rel(&c32, &c64));
        assert!(rel(&rms32, &rms64) < 4e-6, "{:e}", rel(&rms32, &rms64));
        let p32 = pcen(
            rms32.view(),
            22050.0,
            &PcenConfig::new(nzu!(512)).with_time_constant(0.06),
        )
        .unwrap();
        let p64 = pcen(
            rms64.view(),
            22050.0,
            &PcenConfig::new(nzu!(512)).with_time_constant(0.06),
        )
        .unwrap();
        eprintln!("f32 vs f64: pcen {:.3e}", rel(&p32, &p64));
        assert!(rel(&p32, &p64) < 4e-6, "{:e}", rel(&p32, &p64));
    }
}
