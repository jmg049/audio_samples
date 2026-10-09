//! librosa-compatible frame-wise pitch tracking (YIN, pYIN) and harmonic
//! spectrogram helpers (salience, harmonic interpolation).
//!
//! ## What
//!
//! This module provides the [`AudioPitchTracking`] trait and the free functions
//! it is built from. They reproduce librosa 1.0's pitch estimators and its
//! `librosa.core.harmonic` helpers:
//!
//! | librosa | here |
//! |---|---|
//! | `librosa.yin` | [`AudioPitchTracking::yin`] ([`yin`]) |
//! | `librosa.pyin` | [`AudioPitchTracking::pyin`] ([`pyin`]) |
//! | `librosa.salience` | [`salience`] |
//! | `librosa.interp_harmonics` | [`interp_harmonics`] |
//! | `librosa.f0_harmonics` | [`f0_harmonics`] |
//!
//! ## Why
//!
//! [`AudioPitchAnalysis`](crate::AudioPitchAnalysis) offers single-buffer pitch
//! detection and a pitch contour with its own conventions (frames start at sample 0,
//! unvoiced frames are `None`, mono only). Code ported from Python, and results
//! compared with published librosa pitch tracks, need librosa's exact
//! conventions: centred frames, the cumulative-mean-normalised difference
//! computed through an FFT autocorrelation, parabolic refinement, the
//! global-minimum fallback of YIN, and the probabilistic HMM decoding of pYIN.
//! Every config struct here defaults to librosa's defaults.
//!
//! ## How
//!
//! Bring [`AudioPitchTracking`] into scope and call [`yin`](AudioPitchTracking::yin)
//! or [`pyin`](AudioPitchTracking::pyin) on any [`AudioSamples`]. Outputs are
//! `f64` arrays shaped `(channels, n_frames)`; frame `t` is centred on sample
//! `t * hop_length` when `center` is `true`. The harmonic helpers are free
//! functions on `(channels, n_bins, n_frames)` magnitude spectrograms, such as
//! the output of
//! [`AudioSpectrum::spectrogram_matrix`](crate::operations::spectrum::AudioSpectrum::spectrogram_matrix).
//!
//! ```rust
//! use audio_samples::operations::pitch_tracking::{AudioPitchTracking, PyinConfig, YinConfig};
//! use audio_samples::{sample_rate, sine_wave};
//! use std::time::Duration;
//!
//! let audio = sine_wave::<f32>(220.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
//!
//! // librosa.yin(y, fmin=65.4, fmax=2093.0, sr=22050): one estimate per frame.
//! let f0 = audio.yin(&YinConfig::default()).unwrap();
//! assert_eq!(f0.dim(), (1, 22));
//! assert!((f0[[0, 11]] - 220.0).abs() < 1.0);
//!
//! // librosa.pyin(...): f0, voiced flags and voicing probabilities.
//! let r = audio.pyin(&PyinConfig::default()).unwrap();
//! assert!(r.voiced_flag[[0, 11]]);
//! assert!((r.f0[[0, 11]] - 220.0).abs() < 2.0);
//! ```
//!
//! # References
//!
//! - A. de Cheveigné and H. Kawahara, "YIN, a fundamental frequency estimator for
//!   speech and music", J. Acoust. Soc. Am. 111(4), 2002.
//! - M. Mauch and S. Dixon, "pYIN: A fundamental frequency estimator using
//!   probabilistic threshold distributions", ICASSP 2014.

use std::cell::RefCell;
use std::num::NonZeroUsize;

use ndarray::{Array2, Array3, Array4, ArrayView1, ArrayView2, ArrayView3, Axis};
use realfft::RealFftPlanner;

use crate::operations::sequence::{TransitionWindow, transition_local, transition_loop, viterbi};
use crate::operations::spectrum::{StftPadMode, pad_signal};
use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ParameterError,
    StandardSample,
};

thread_local! {
    /// Per-thread real-FFT planner for the autocorrelations (plans are cached by length).
    static PLANNER: RefCell<RealFftPlanner<f64>> = RefCell::new(RealFftPlanner::new());
}

/// `numpy.finfo(float64).tiny`, librosa's `util.tiny` guard in the CMND denominator.
const TINY: f64 = f64::MIN_POSITIVE;

/// librosa's recommended minimum frequency, `note_to_hz('C2')`.
const C2_HZ: f64 = 65.406_391_325_149_66;
/// librosa's recommended maximum frequency, `note_to_hz('C7')`.
const C7_HZ: f64 = 2_093.004_522_404_789;

fn param_err(parameter: &str, reason: impl ToString) -> AudioSampleError {
    AudioSampleError::Parameter(ParameterError::invalid_value(parameter, reason))
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration of [`AudioPitchTracking::yin`], mirroring `librosa.yin`.
///
/// ## Purpose
///
/// Describes the frequency search range, the framing and the trough threshold
/// of the YIN estimator.
///
/// ## Intended usage
///
/// Pass to [`AudioPitchTracking::yin`] or [`yin`]. `Default` reproduces librosa's
/// defaults (`frame_length = 2048`, `hop_length = frame_length / 4`,
/// `trough_threshold = 0.1`, centred frames, zero padding). librosa has no
/// default for `fmin` / `fmax`; here they default to librosa's recommended range
/// `note_to_hz('C2')` to `note_to_hz('C7')` (about 65.4 Hz to 2093 Hz).
///
/// ## Invariants
///
/// Checked by [`YinConfig::validate`]: `0 < fmin < fmax <= sample_rate / 2`, at
/// least one period of `fmin` fits in a frame (`sample_rate / fmin <
/// frame_length - 1`), the hop is at least one sample, and the threshold is finite.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::pitch_tracking::YinConfig;
///
/// let cfg = YinConfig::new()
///     .with_fmin(80.0)
///     .with_fmax(1000.0)
///     .with_frame_length(nzu!(1024))
///     .with_trough_threshold(0.2);
/// assert_eq!(cfg.effective_hop_length().unwrap().get(), 256);
/// assert!(cfg.validate(22050.0).is_ok());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct YinConfig {
    /// Lowest frequency searched, in Hz (`> 0`). Sets the longest period
    /// `ceil(sample_rate / fmin)` (capped at `frame_length - 1`).
    pub fmin: f64,
    /// Highest frequency searched, in Hz (`fmin < fmax <= sample_rate / 2`). Sets
    /// the shortest period `floor(sample_rate / fmax)`.
    pub fmax: f64,
    /// Samples per analysis frame. librosa default: 2048.
    pub frame_length: NonZeroUsize,
    /// Samples between successive estimates. `None` means `frame_length / 4`.
    pub hop_length: Option<NonZeroUsize>,
    /// Absolute threshold on the cumulative mean normalised difference: the first
    /// trough below it gives the period. librosa default: 0.1.
    pub trough_threshold: f64,
    /// If `true`, the signal is padded by `frame_length / 2` samples at both ends so
    /// that frame `t` is centred on sample `t * hop_length`. If `false`, frame `t`
    /// starts at sample `t * hop_length` and the signal must be at least
    /// `frame_length` samples long.
    pub center: bool,
    /// Padding used when `center` is `true`. librosa default: zeros.
    pub pad_mode: StftPadMode,
}

impl Default for YinConfig {
    fn default() -> Self {
        Self {
            fmin: C2_HZ,
            fmax: C7_HZ,
            frame_length: crate::nzu!(2048),
            hop_length: None,
            trough_threshold: 0.1,
            center: true,
            pad_mode: StftPadMode::Constant,
        }
    }
}

impl YinConfig {
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the lowest searched frequency in Hz.
    #[must_use]
    pub fn with_fmin(mut self, fmin: f64) -> Self {
        self.fmin = fmin;
        self
    }

    /// Sets the highest searched frequency in Hz.
    #[must_use]
    pub fn with_fmax(mut self, fmax: f64) -> Self {
        self.fmax = fmax;
        self
    }

    /// Sets the frame length in samples.
    #[must_use]
    pub fn with_frame_length(mut self, frame_length: NonZeroUsize) -> Self {
        self.frame_length = frame_length;
        self
    }

    /// Sets the hop length (`None` = `frame_length / 4`).
    #[must_use]
    pub fn with_hop_length(mut self, hop_length: Option<NonZeroUsize>) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Sets the trough threshold.
    #[must_use]
    pub fn with_trough_threshold(mut self, trough_threshold: f64) -> Self {
        self.trough_threshold = trough_threshold;
        self
    }

    /// Sets whether frames are centred.
    #[must_use]
    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Sets the padding mode used when frames are centred.
    #[must_use]
    pub fn with_pad_mode(mut self, pad_mode: StftPadMode) -> Self {
        self.pad_mode = pad_mode;
        self
    }

    /// The hop length in samples (`hop_length`, or `frame_length / 4` when `None`).
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `hop_length` is `None` and `frame_length / 4` is 0.
    pub fn effective_hop_length(&self) -> AudioSampleResult<NonZeroUsize> {
        effective_hop(self.hop_length, self.frame_length)
    }

    /// Checks the configuration for a signal sampled at `sample_rate` Hz.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the frequency range is invalid (see the type-level
    /// invariants), the hop is 0, or `trough_threshold` is not finite.
    pub fn validate(&self, sample_rate: f64) -> AudioSampleResult<()> {
        check_yin_params(sample_rate, self.fmin, self.fmax, self.frame_length)?;
        self.effective_hop_length()?;
        if !self.trough_threshold.is_finite() {
            return Err(param_err("trough_threshold", "must be finite"));
        }
        Ok(())
    }
}

/// What [`AudioPitchTracking::pyin`] reports as `f0` for unvoiced frames
/// (librosa's `fill_na`).
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::pitch_tracking::UnvoicedFill;
///
/// assert_eq!(UnvoicedFill::default(), UnvoicedFill::Nan);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum UnvoicedFill {
    /// `f64::NAN` (librosa's default `fill_na=np.nan`).
    #[default]
    Nan,
    /// The decoded best-guess pitch of the unvoiced state (librosa's `fill_na=None`).
    BestGuess,
    /// A fixed value, e.g. `0.0` (librosa's `fill_na=<float>`).
    Value(f64),
}

/// Configuration of [`AudioPitchTracking::pyin`], mirroring `librosa.pyin`.
///
/// ## Purpose
///
/// Describes the YIN front end (frequency range, framing), the probabilistic
/// threshold and trough priors, the pitch-bin resolution and the hidden Markov
/// model used to decode a smooth, voiced/unvoiced pitch track.
///
/// ## Intended usage
///
/// Pass to [`AudioPitchTracking::pyin`] or [`pyin`]. `Default` reproduces
/// librosa 1.0's defaults (`frame_length = 2048`, `hop_length = frame_length / 4`,
/// 100 thresholds, Beta(2, 18) threshold prior, Boltzmann parameter 2, 0.1
/// semitone resolution, 35.92 octaves/s maximum transition rate, switch and
/// no-trough probabilities 0.01, NaN for unvoiced frames, centred frames with zero
/// padding, `transition_min_prob = 1e-4`). As for [`YinConfig`], `fmin` / `fmax`
/// default to librosa's recommended C2 to C7 range.
///
/// ## Invariants
///
/// Checked by [`PyinConfig::validate`]: the YIN invariants of [`YinConfig`];
/// positive, finite Beta and Boltzmann parameters; `0 < resolution < 1`;
/// `max_transition_rate > 0`; `0 <= switch_prob <= 1` and `0 <= no_trough_prob <= 1`;
/// a non-negative `transition_min_prob`; and at least two pitch bins.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::pitch_tracking::{PyinConfig, UnvoicedFill};
///
/// let cfg = PyinConfig::new()
///     .with_fmin(80.0)
///     .with_fmax(800.0)
///     .with_fill(UnvoicedFill::Value(0.0));
/// assert!(cfg.validate(22050.0).is_ok());
/// // 0.1-semitone bins over 80..800 Hz.
/// assert_eq!(cfg.n_pitch_bins(), 399);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PyinConfig {
    /// Lowest frequency searched, in Hz (`> 0`); also the frequency of pitch bin 0.
    pub fmin: f64,
    /// Highest frequency searched, in Hz (`fmin < fmax <= sample_rate / 2`).
    pub fmax: f64,
    /// Samples per analysis frame. librosa default: 2048.
    pub frame_length: NonZeroUsize,
    /// Samples between successive estimates. `None` means `frame_length / 4`.
    pub hop_length: Option<NonZeroUsize>,
    /// Number of thresholds the Beta prior is discretised into. librosa default: 100.
    pub n_thresholds: NonZeroUsize,
    /// Shape parameters `(alpha, beta)` of the Beta prior over thresholds.
    /// librosa default: `(2, 18)` (mean threshold 0.1).
    pub beta_parameters: (f64, f64),
    /// Shape parameter of the Boltzmann prior over trough rank; larger values put
    /// more mass on shorter periods. librosa default: 2.
    pub boltzmann_parameter: f64,
    /// Pitch-bin width in semitones, in `(0, 1)`. librosa default: 0.1.
    pub resolution: f64,
    /// Maximum pitch change between frames, in octaves per second. librosa
    /// default: 35.92.
    pub max_transition_rate: f64,
    /// Probability of switching between voiced and unvoiced between frames.
    /// librosa default: 0.01.
    pub switch_prob: f64,
    /// Probability mass given to the global minimum when no trough is below a
    /// threshold. librosa default: 0.01.
    pub no_trough_prob: f64,
    /// Value reported in `f0` for unvoiced frames. librosa default: NaN.
    pub fill: UnvoicedFill,
    /// If `true`, frames are centred on `t * hop_length` (see [`YinConfig::center`]).
    pub center: bool,
    /// Padding used when `center` is `true`. librosa default: zeros.
    pub pad_mode: StftPadMode,
    /// Viterbi pruning: only transitions with probability `>= p` are searched.
    /// `None` (or `Some(0.0)`) decodes exactly. librosa default: `Some(1e-4)`.
    pub transition_min_prob: Option<f64>,
}

impl Default for PyinConfig {
    fn default() -> Self {
        Self {
            fmin: C2_HZ,
            fmax: C7_HZ,
            frame_length: crate::nzu!(2048),
            hop_length: None,
            n_thresholds: crate::nzu!(100),
            beta_parameters: (2.0, 18.0),
            boltzmann_parameter: 2.0,
            resolution: 0.1,
            max_transition_rate: 35.92,
            switch_prob: 0.01,
            no_trough_prob: 0.01,
            fill: UnvoicedFill::Nan,
            center: true,
            pad_mode: StftPadMode::Constant,
            transition_min_prob: Some(1e-4),
        }
    }
}

impl PyinConfig {
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the lowest searched frequency in Hz.
    #[must_use]
    pub fn with_fmin(mut self, fmin: f64) -> Self {
        self.fmin = fmin;
        self
    }

    /// Sets the highest searched frequency in Hz.
    #[must_use]
    pub fn with_fmax(mut self, fmax: f64) -> Self {
        self.fmax = fmax;
        self
    }

    /// Sets the frame length in samples.
    #[must_use]
    pub fn with_frame_length(mut self, frame_length: NonZeroUsize) -> Self {
        self.frame_length = frame_length;
        self
    }

    /// Sets the hop length (`None` = `frame_length / 4`).
    #[must_use]
    pub fn with_hop_length(mut self, hop_length: Option<NonZeroUsize>) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Sets the number of thresholds.
    #[must_use]
    pub fn with_n_thresholds(mut self, n_thresholds: NonZeroUsize) -> Self {
        self.n_thresholds = n_thresholds;
        self
    }

    /// Sets the Beta prior's shape parameters.
    #[must_use]
    pub fn with_beta_parameters(mut self, alpha: f64, beta: f64) -> Self {
        self.beta_parameters = (alpha, beta);
        self
    }

    /// Sets the Boltzmann parameter.
    #[must_use]
    pub fn with_boltzmann_parameter(mut self, boltzmann_parameter: f64) -> Self {
        self.boltzmann_parameter = boltzmann_parameter;
        self
    }

    /// Sets the pitch-bin resolution in semitones.
    #[must_use]
    pub fn with_resolution(mut self, resolution: f64) -> Self {
        self.resolution = resolution;
        self
    }

    /// Sets the maximum transition rate in octaves per second.
    #[must_use]
    pub fn with_max_transition_rate(mut self, max_transition_rate: f64) -> Self {
        self.max_transition_rate = max_transition_rate;
        self
    }

    /// Sets the voiced/unvoiced switch probability.
    #[must_use]
    pub fn with_switch_prob(mut self, switch_prob: f64) -> Self {
        self.switch_prob = switch_prob;
        self
    }

    /// Sets the no-trough probability.
    #[must_use]
    pub fn with_no_trough_prob(mut self, no_trough_prob: f64) -> Self {
        self.no_trough_prob = no_trough_prob;
        self
    }

    /// Sets the value reported for unvoiced frames.
    #[must_use]
    pub fn with_fill(mut self, fill: UnvoicedFill) -> Self {
        self.fill = fill;
        self
    }

    /// Sets whether frames are centred.
    #[must_use]
    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Sets the padding mode used when frames are centred.
    #[must_use]
    pub fn with_pad_mode(mut self, pad_mode: StftPadMode) -> Self {
        self.pad_mode = pad_mode;
        self
    }

    /// Sets the Viterbi pruning threshold (`None` = exact decoding).
    #[must_use]
    pub fn with_transition_min_prob(mut self, transition_min_prob: Option<f64>) -> Self {
        self.transition_min_prob = transition_min_prob;
        self
    }

    /// The hop length in samples (`hop_length`, or `frame_length / 4` when `None`).
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `hop_length` is `None` and `frame_length / 4` is 0.
    pub fn effective_hop_length(&self) -> AudioSampleResult<NonZeroUsize> {
        effective_hop(self.hop_length, self.frame_length)
    }

    /// Pitch bins per semitone, `ceil(1 / resolution)`.
    #[must_use]
    pub fn bins_per_semitone(&self) -> usize {
        (1.0 / self.resolution).ceil() as usize
    }

    /// Number of pitch bins, `floor(12 * bins_per_semitone * log2(fmax / fmin)) + 1`.
    /// The HMM has twice as many states (voiced and unvoiced copies).
    #[must_use]
    pub fn n_pitch_bins(&self) -> usize {
        let bps = self.bins_per_semitone() as f64;
        ((12.0 * bps) * (self.fmax / self.fmin).log2()).floor() as usize + 1
    }

    /// Centre frequency of every pitch bin, `fmin * 2^(k / (12 * bins_per_semitone))`.
    #[must_use]
    pub fn pitch_bin_frequencies(&self) -> Vec<f64> {
        let denom = (12 * self.bins_per_semitone()) as f64;
        (0..self.n_pitch_bins())
            .map(|k| self.fmin * 2f64.powf(k as f64 / denom))
            .collect()
    }

    /// Checks the configuration for a signal sampled at `sample_rate` Hz.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if any type-level invariant is violated.
    pub fn validate(&self, sample_rate: f64) -> AudioSampleResult<()> {
        check_yin_params(sample_rate, self.fmin, self.fmax, self.frame_length)?;
        self.effective_hop_length()?;
        let (a, b) = self.beta_parameters;
        if !(a.is_finite() && a > 0.0 && b.is_finite() && b > 0.0) {
            return Err(param_err(
                "beta_parameters",
                "both Beta shape parameters must be finite and > 0",
            ));
        }
        if !(self.boltzmann_parameter.is_finite() && self.boltzmann_parameter > 0.0) {
            return Err(param_err("boltzmann_parameter", "must be finite and > 0"));
        }
        if !(self.resolution > 0.0 && self.resolution < 1.0) {
            return Err(ParameterError::out_of_range(
                "resolution",
                self.resolution,
                0.0,
                1.0,
                "the pitch-bin resolution must be in (0, 1) semitones",
            )
            .into());
        }
        if !(self.max_transition_rate.is_finite() && self.max_transition_rate > 0.0) {
            return Err(param_err("max_transition_rate", "must be finite and > 0"));
        }
        for (name, p) in [
            ("switch_prob", self.switch_prob),
            ("no_trough_prob", self.no_trough_prob),
        ] {
            if !(0.0..=1.0).contains(&p) {
                return Err(ParameterError::out_of_range(
                    name,
                    p,
                    0.0,
                    1.0,
                    "must be a probability",
                )
                .into());
            }
        }
        if matches!(self.fill, UnvoicedFill::Value(v) if v.is_infinite()) {
            return Err(param_err("fill", "the fill value must not be infinite"));
        }
        if matches!(self.transition_min_prob, Some(p) if !(p.is_finite() && p >= 0.0)) {
            return Err(param_err(
                "transition_min_prob",
                "must be None or a finite non-negative probability",
            ));
        }
        if self.n_pitch_bins() < 2 {
            return Err(param_err(
                "fmax",
                "the frequency range must span at least two pitch bins",
            ));
        }
        Ok(())
    }
}

/// Output of [`AudioPitchTracking::pyin`]: one value per channel and frame.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::pitch_tracking::{AudioPitchTracking, PyinConfig};
/// use audio_samples::{sample_rate, sine_wave};
/// use std::time::Duration;
///
/// let audio = sine_wave::<f64>(330.0, Duration::from_millis(300), sample_rate!(22050), 0.5);
/// let r = audio.pyin(&PyinConfig::default()).unwrap();
/// assert_eq!(r.f0.dim(), r.voiced_flag.dim());
/// assert_eq!(r.f0.dim(), r.voiced_prob.dim());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PyinResult {
    /// `(channels, n_frames)` fundamental frequency in Hz. Unvoiced frames hold
    /// the value selected by [`PyinConfig::fill`] (NaN by default).
    pub f0: Array2<f64>,
    /// `(channels, n_frames)`: `true` where the decoded state is voiced.
    pub voiced_flag: Array2<bool>,
    /// `(channels, n_frames)` probability in `[0, 1]` that the frame is voiced
    /// (before HMM smoothing).
    pub voiced_prob: Array2<f64>,
}

fn effective_hop(
    hop_length: Option<NonZeroUsize>,
    frame_length: NonZeroUsize,
) -> AudioSampleResult<NonZeroUsize> {
    match hop_length {
        Some(h) => Ok(h),
        None => NonZeroUsize::new(frame_length.get() / 4).ok_or_else(|| {
            param_err(
                "hop_length",
                "frame_length / 4 is 0; set hop_length explicitly for frames shorter than 4 samples",
            )
        }),
    }
}

/// librosa's `__check_yin_params`, plus finiteness and a non-degenerate lag range.
fn check_yin_params(
    sr: f64,
    fmin: f64,
    fmax: f64,
    frame_length: NonZeroUsize,
) -> AudioSampleResult<()> {
    if !(sr.is_finite() && sr > 0.0) {
        return Err(param_err("sample_rate", "must be finite and > 0"));
    }
    if !(fmin.is_finite() && fmax.is_finite()) {
        return Err(param_err("fmin", "fmin and fmax must be finite"));
    }
    if fmax > sr / 2.0 {
        return Err(ParameterError::out_of_range(
            "fmax",
            fmax,
            fmin,
            sr / 2.0,
            "fmax cannot exceed the Nyquist frequency",
        )
        .into());
    }
    if fmin >= fmax {
        return Err(param_err(
            "fmin",
            format!("fmin={fmin:.3} must be less than fmax={fmax:.3}"),
        ));
    }
    if fmin <= 0.0 {
        return Err(param_err("fmin", "must be strictly positive"));
    }
    let fl = frame_length.get();
    if sr / fmin >= fl as f64 - 1.0 {
        return Err(param_err(
            "fmin",
            format!(
                "fmin={fmin:.3} is too small for frame_length={fl} and sr={sr}: increase fmin to {:.3} \
                 or frame_length to {}",
                sr / (fl as f64 - 1.0),
                (sr / fmin).ceil() as usize + 1
            ),
        ));
    }
    let (min_p, max_p) = periods(sr, fmin, fmax, fl);
    if max_p <= min_p {
        return Err(param_err(
            "fmax",
            "the period range floor(sr/fmax)..=ceil(sr/fmin) must contain at least two lags",
        ));
    }
    Ok(())
}

/// `(min_period, max_period)` as in librosa.
fn periods(sr: f64, fmin: f64, fmax: f64, frame_length: usize) -> (usize, usize) {
    let min_period = (sr / fmax).floor() as usize;
    let max_period = ((sr / fmin).ceil() as usize).min(frame_length - 1);
    (min_period, max_period)
}

// =============================================================================
// YIN core (librosa.core.pitch private helpers)
// =============================================================================

/// `scipy.fft.next_fast_len(n, real=True)`: the smallest 5-smooth integer `>= n`.
fn next_fast_len_real(n: usize) -> usize {
    if n <= 6 {
        return n.max(1);
    }
    let mut best = usize::MAX;
    let mut p5 = 1usize;
    while p5 < best {
        let mut p35 = p5;
        while p35 < best {
            let mut m = p35;
            while m < n {
                m *= 2;
            }
            best = best.min(m);
            if p35 >= n {
                break;
            }
            p35 *= 3;
        }
        if p5 >= n {
            break;
        }
        p5 *= 5;
    }
    best
}

/// Pads (if centred) and frames one channel; returns the frames as rows.
fn frames_of(
    y: &[f64],
    frame_length: usize,
    hop: usize,
    center: bool,
    pad_mode: StftPadMode,
) -> AudioSampleResult<Vec<Vec<f64>>> {
    let padded;
    let y = if center {
        padded = pad_signal(y, frame_length / 2, pad_mode);
        &padded[..]
    } else {
        y
    };
    if y.len() < frame_length {
        return Err(param_err(
            "frame_length",
            format!(
                "input is too short (n={}) for frame_length={frame_length}",
                y.len()
            ),
        ));
    }
    let n_frames = 1 + (y.len() - frame_length) / hop;
    Ok((0..n_frames)
        .map(|t| y[t * hop..t * hop + frame_length].to_vec())
        .collect())
}

/// librosa's `_cumulative_mean_normalized_difference` for one frame:
/// returns `cmnd[tau - min_period]` for `tau in min_period..=max_period`.
fn cmnd_frame(
    frame: &[f64],
    min_period: usize,
    max_period: usize,
    fft: &FftScratch,
    work: &mut Vec<f64>,
    spec: &mut [realfft::num_complex::Complex<f64>],
) -> Vec<f64> {
    let n_pad = fft.n_pad;
    // Autocorrelation via the power spectrum (audio.autocorrelate, real input).
    work.clear();
    work.extend_from_slice(frame);
    work.resize(n_pad, 0.0);
    fft.forward
        .process(work, spec)
        .expect("buffer lengths match the plan");
    for z in spec.iter_mut() {
        *z = realfft::num_complex::Complex::new(z.re * z.re + z.im * z.im, 0.0);
    }
    fft.inverse
        .process(spec, work)
        .expect("buffer lengths match the plan");
    let scale = 1.0 / n_pad as f64;
    let acf = |k: usize| work[k] * scale;

    // Energy cumsum c[m] = sum_{i<=m} y_i^2, with c[0] replaced by 0 before use
    // (librosa zeroes row 0 of the shared buffer first).
    let mut csum = vec![0.0; max_period];
    let mut acc = 0.0;
    for (m, c) in csum.iter_mut().enumerate() {
        acc += frame[m] * frame[m];
        *c = acc;
    }
    if let Some(c0) = csum.first_mut() {
        *c0 = 0.0;
    }
    let acf0 = acf(0);
    // d[k] = 2 (acf0 - acf_k) - c[k-1], k = 1..=max_period.
    let mut d = vec![0.0; max_period + 1];
    for k in 1..=max_period {
        d[k] = 2.0 * (acf0 - acf(k)) - csum[k - 1];
    }
    // Cumulative mean of d[1..=k] divided by k.
    let mut out = Vec::with_capacity(max_period - min_period + 1);
    let mut run = 0.0;
    for (k, &dk) in d.iter().enumerate().skip(1) {
        run += dk;
        if k >= min_period {
            let mean = run / k as f64;
            out.push(dk / (mean + TINY));
        }
    }
    out
}

/// librosa's `_parabolic_interpolation` (the `_pi_stencil` rule) for one frame.
fn parabolic_shifts(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut s = vec![0.0; n];
    for i in 1..n.saturating_sub(1) {
        let a = x[i + 1] + x[i - 1] - 2.0 * x[i];
        let b = (x[i + 1] - x[i - 1]) / 2.0;
        s[i] = if b.abs() >= a.abs() { 0.0 } else { -b / a };
    }
    s
}

/// librosa's `util.localmin` along a frame, with YIN's first-element rule
/// (`is_trough[0] = x[0] < x[1]`). Requires `x.len() >= 2`.
fn troughs(x: &[f64]) -> Vec<bool> {
    let n = x.len();
    let mut t = vec![false; n];
    for i in 1..n - 1 {
        t[i] = x[i] < x[i - 1] && x[i] <= x[i + 1];
    }
    t[n - 1] = x[n - 1] < x[n - 2];
    t[0] = x[0] < x[1];
    t
}

/// First index of the minimum (`numpy.argmin`, NaN wins as in NumPy).
fn argmin_first(x: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if x[best].is_nan() {
            break;
        }
        if v.is_nan() || v < x[best] {
            best = i;
        }
    }
    best
}

struct FftScratch {
    n_pad: usize,
    forward: std::sync::Arc<dyn realfft::RealToComplex<f64>>,
    inverse: std::sync::Arc<dyn realfft::ComplexToReal<f64>>,
}

impl FftScratch {
    fn new(frame_length: usize) -> Self {
        let n_pad = next_fast_len_real(2 * frame_length - 1);
        let (forward, inverse) = PLANNER.with(|p| {
            let mut p = p.borrow_mut();
            (p.plan_fft_forward(n_pad), p.plan_fft_inverse(n_pad))
        });
        Self {
            n_pad,
            forward,
            inverse,
        }
    }
}

/// Per-channel CMND and parabolic shifts for every frame.
struct YinFrames {
    min_period: usize,
    /// `cmnd[t][tau - min_period]`.
    cmnd: Vec<Vec<f64>>,
    /// `shifts[t][tau - min_period]`.
    shifts: Vec<Vec<f64>>,
}

#[allow(clippy::too_many_arguments)]
fn yin_frames(
    y: &[f64],
    sr: f64,
    fmin: f64,
    fmax: f64,
    frame_length: usize,
    hop: usize,
    center: bool,
    pad_mode: StftPadMode,
) -> AudioSampleResult<YinFrames> {
    if y.iter().any(|v| !v.is_finite()) {
        return Err(param_err("audio", "audio buffer is not finite everywhere"));
    }
    let frames = frames_of(y, frame_length, hop, center, pad_mode)?;
    let (min_period, max_period) = periods(sr, fmin, fmax, frame_length);
    let fft = FftScratch::new(frame_length);
    let mut work = fft.forward.make_input_vec();
    let mut spec = fft.forward.make_output_vec();
    let cmnd: Vec<Vec<f64>> = frames
        .iter()
        .map(|f| cmnd_frame(f, min_period, max_period, &fft, &mut work, &mut spec))
        .collect();
    let shifts = cmnd.iter().map(|c| parabolic_shifts(c)).collect();
    Ok(YinFrames {
        min_period,
        cmnd,
        shifts,
    })
}

// =============================================================================
// Probability helpers for pYIN
// =============================================================================

/// `numpy.linspace(0, 1, n + 1)`.
fn unit_linspace(n: usize) -> Vec<f64> {
    let step = 1.0 / n as f64;
    let mut v: Vec<f64> = (0..=n).map(|i| i as f64 * step).collect();
    v[n] = 1.0;
    v
}

/// Natural log of the gamma function (Lanczos, g = 7, n = 9), for `x > 0`.
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: Gamma(x) Gamma(1 - x) = pi / sin(pi x).
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Continued fraction for the incomplete beta function (modified Lentz).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    const EPS: f64 = 1e-16;
    const FPMIN: f64 = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..10_000 {
        let m = f64::from(m);
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularised incomplete beta function `I_x(a, b)` (`scipy.stats.beta.cdf`).
///
/// Integer shape parameters (the pYIN default `(2, 18)`) use the exact binomial
/// expansion `I_x(a, b) = sum_{j >= a} C(n, j) x^j (1 - x)^(n - j)`, `n = a + b - 1`,
/// summing whichever tail is smaller so that the result keeps its relative
/// accuracy near 0 and its absolute accuracy (a few ulps) near 1.
/// Other parameters use the Lentz continued fraction (relative error ~1e-14).
fn beta_cdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let integer = a.fract() == 0.0 && b.fract() == 0.0 && a + b <= 200.0;
    if integer {
        let n = (a + b - 1.0) as i32;
        let ka = a as i32;
        let y = 1.0 - x;
        let mut binom = 1.0; // C(n, j)
        let (mut lower, mut upper) = (0.0, 0.0);
        for j in 0..=n {
            if j > 0 {
                binom = binom * f64::from(n - j + 1) / f64::from(j);
            }
            let t = binom * x.powi(j) * y.powi(n - j);
            if j < ka {
                lower += t;
            } else {
                upper += t;
            }
        }
        return if upper <= 0.5 { upper } else { 1.0 - lower };
    }
    let ln_front = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (-x).ln_1p();
    let front = ln_front.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(a, b, x) / a
    } else {
        1.0 - front * beta_cf(b, a, 1.0 - x) / b
    }
}

/// `scipy.stats.boltzmann.pmf(k, lambda, n)`.
fn boltzmann_pmf(k: usize, lambda: f64, n: usize) -> f64 {
    let fact = (1.0 - (-lambda).exp()) / (1.0 - (-lambda * n as f64).exp());
    fact * (-lambda * k as f64).exp()
}

/// NumPy's pairwise summation (`numpy.sum` of a contiguous 1-D `float64` array).
fn np_sum(a: &[f64]) -> f64 {
    fn pairwise(a: &[f64]) -> f64 {
        let n = a.len();
        if n < 8 {
            let mut res = 0.0;
            for &v in a {
                res += v;
            }
            res
        } else if n <= 128 {
            let mut r = [0.0; 8];
            r.copy_from_slice(&a[..8]);
            let mut i = 8;
            while i < n - (n % 8) {
                for j in 0..8 {
                    r[j] += a[i + j];
                }
                i += 8;
            }
            let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            for &v in &a[i..] {
                res += v;
            }
            res
        } else {
            let mut n2 = n / 2;
            n2 -= n2 % 8;
            pairwise(&a[..n2]) + pairwise(&a[n2..])
        }
    }
    pairwise(a)
}

/// librosa's `__pyin_helper` for one frame: `(trough lag index, probability)` for
/// every trough with non-zero probability, in increasing lag order.
fn pyin_frame_candidates(
    cmnd: &[f64],
    thresholds: &[f64],
    beta_probs: &[f64],
    boltzmann: f64,
    no_trough_prob: f64,
) -> Vec<(usize, f64)> {
    let is_trough = troughs(cmnd);
    let idx: Vec<usize> = (0..cmnd.len()).filter(|&i| is_trough[i]).collect();
    if idx.is_empty() {
        return Vec::new();
    }
    let heights: Vec<f64> = idx.iter().map(|&i| cmnd[i]).collect();
    let n_thr = beta_probs.len();
    // below[i][j]: trough i is below thresholds[j + 1].
    let below: Vec<Vec<bool>> = heights
        .iter()
        .map(|&h| thresholds[1..].iter().map(|&t| h < t).collect())
        .collect();
    let mut n_troughs = vec![0usize; n_thr];
    for row in &below {
        for (j, &b) in row.iter().enumerate() {
            n_troughs[j] += usize::from(b);
        }
    }
    let mut position = vec![0usize; n_thr];
    let mut probs = vec![0.0; idx.len()];
    for (i, row) in below.iter().enumerate() {
        let mut p = 0.0;
        for j in 0..n_thr {
            if row[j] {
                let prior = boltzmann_pmf(position[j], boltzmann, n_troughs[j]);
                position[j] += 1;
                p += prior * beta_probs[j];
            }
        }
        probs[i] = p;
    }
    let gmin = argmin_first(&heights);
    let n_not_below = below[gmin].iter().filter(|&&b| !b).count();
    probs[gmin] += no_trough_prob * np_sum(&beta_probs[..n_not_below]);
    idx.into_iter()
        .zip(probs)
        .filter(|&(_, p)| p != 0.0)
        .collect()
}

// =============================================================================
// Array-level YIN / pYIN
// =============================================================================

fn check_signal(signal: &ArrayView2<f64>) -> AudioSampleResult<()> {
    if signal.nrows() == 0 {
        return Err(param_err("signal", "must have at least one channel"));
    }
    Ok(())
}

/// Frame-wise YIN fundamental-frequency estimation of every row of `signal`,
/// equivalent to `librosa.yin`.
///
/// This is the array-level function behind [`AudioPitchTracking::yin`].
///
/// # Arguments
///
/// * `signal` - `(channels, samples)` array of finite samples.
/// * `sample_rate` - Sample rate in Hz.
/// * `config` - YIN parameters.
///
/// # Returns
///
/// A `(channels, n_frames)` array of f0 estimates in Hz, with `n_frames =
/// 1 + samples / hop` when centred and `1 + (samples - frame_length) / hop`
/// otherwise. Every frame gets a value: when no trough is below the threshold,
/// the global minimum of the difference function is used (as in librosa).
///
/// # Errors
///
/// [`ParameterError`] if the configuration is invalid for `sample_rate`, the
/// signal is not finite, or (uncentred) shorter than `frame_length`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::pitch_tracking::{YinConfig, yin};
/// use ndarray::Array2;
///
/// let sr = 22050.0;
/// let y = Array2::from_shape_fn((1, 8192), |(_, n)| {
///     (2.0 * std::f64::consts::PI * 440.0 * n as f64 / sr).sin()
/// });
/// let f0 = yin(y.view(), sr, &YinConfig::default()).unwrap();
/// assert_eq!(f0.dim(), (1, 17));
/// assert!((f0[[0, 8]] - 440.0).abs() < 0.5);
/// ```
pub fn yin(
    signal: ArrayView2<f64>,
    sample_rate: f64,
    config: &YinConfig,
) -> AudioSampleResult<Array2<f64>> {
    config.validate(sample_rate)?;
    check_signal(&signal)?;
    let hop = config.effective_hop_length()?.get();
    let fl = config.frame_length.get();
    let mut rows = Vec::with_capacity(signal.nrows());
    for row in signal.rows() {
        let y = row.to_vec();
        let yf = yin_frames(
            &y,
            sample_rate,
            config.fmin,
            config.fmax,
            fl,
            hop,
            config.center,
            config.pad_mode,
        )?;
        let f0: Vec<f64> = yf
            .cmnd
            .iter()
            .zip(&yf.shifts)
            .map(|(c, s)| {
                let tr = troughs(c);
                let i = (0..c.len())
                    .find(|&i| tr[i] && c[i] < config.trough_threshold)
                    .unwrap_or_else(|| argmin_first(c));
                sample_rate / (yf.min_period as f64 + i as f64 + s[i])
            })
            .collect();
        rows.push(f0);
    }
    stack_rows(rows)
}

fn stack_rows<T: Clone>(rows: Vec<Vec<T>>) -> AudioSampleResult<Array2<T>> {
    let n = rows.first().map_or(0, Vec::len);
    let c = rows.len();
    Array2::from_shape_vec((c, n), rows.into_iter().flatten().collect())
        .map_err(|e| param_err("signal", e))
}

/// Probabilistic YIN (pYIN) of every row of `signal`, equivalent to
/// `librosa.pyin`.
///
/// This is the array-level function behind [`AudioPitchTracking::pyin`]. Each
/// frame's YIN troughs become f0 candidates weighted by a Beta prior over
/// thresholds and a Boltzmann prior over trough rank; the candidates are mapped
/// to `resolution`-semitone pitch bins and decoded by Viterbi over an HMM with
/// voiced and unvoiced copies of every bin, a local (triangular) pitch transition
/// and a voiced/unvoiced switch probability.
///
/// # Arguments
///
/// * `signal` - `(channels, samples)` array of finite samples.
/// * `sample_rate` - Sample rate in Hz.
/// * `config` - pYIN parameters.
///
/// # Returns
///
/// A [`PyinResult`] whose arrays are `(channels, n_frames)` (framing as in [`yin`]).
///
/// # Errors
///
/// [`ParameterError`] if the configuration is invalid for `sample_rate`, the
/// signal is not finite, or (uncentred) shorter than `frame_length`; the
/// Viterbi decoder's errors are propagated (e.g. a `transition_min_prob` that
/// leaves a state without predecessors).
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::pitch_tracking::{PyinConfig, pyin};
/// use ndarray::Array2;
///
/// let sr = 22050.0;
/// let y = Array2::from_shape_fn((1, 8192), |(_, n)| {
///     0.5 * (2.0 * std::f64::consts::PI * 196.0 * n as f64 / sr).sin()
/// });
/// let r = pyin(y.view(), sr, &PyinConfig::default()).unwrap();
/// assert!(r.voiced_flag[[0, 8]]);
/// // f0 is quantised to 0.1-semitone bins.
/// assert!((r.f0[[0, 8]] / 196.0).log2().abs() * 1200.0 < 6.0);
/// ```
pub fn pyin(
    signal: ArrayView2<f64>,
    sample_rate: f64,
    config: &PyinConfig,
) -> AudioSampleResult<PyinResult> {
    config.validate(sample_rate)?;
    check_signal(&signal)?;
    let hop = config.effective_hop_length()?.get();
    let fl = config.frame_length.get();

    let n_thr = config.n_thresholds.get();
    let thresholds = unit_linspace(n_thr);
    let (a, b) = config.beta_parameters;
    let cdf: Vec<f64> = thresholds.iter().map(|&t| beta_cdf(t, a, b)).collect();
    let beta_probs: Vec<f64> = cdf.windows(2).map(|w| w[1] - w[0]).collect();

    let bps = config.bins_per_semitone();
    let n_bins = config.n_pitch_bins();
    let bins_per_octave = (12 * bps) as f64;
    let freqs = config.pitch_bin_frequencies();

    // Transition matrix: kron(switch, local).
    let max_semitones =
        (config.max_transition_rate * 12.0 * hop as f64 / sample_rate).round_ties_even() as usize;
    let width = max_semitones * bps + 1;
    let local = transition_local(n_bins, &[width], TransitionWindow::Triangle, false)?;
    let switch = transition_loop(2, &[1.0 - config.switch_prob])?;
    let n_states = 2 * n_bins;
    let mut transition = Array2::<f64>::zeros((n_states, n_states));
    for bi in 0..2 {
        for bj in 0..2 {
            let s = switch[[bi, bj]];
            let mut block = transition.slice_mut(ndarray::s![
                bi * n_bins..(bi + 1) * n_bins,
                bj * n_bins..(bj + 1) * n_bins
            ]);
            block.zip_mut_with(&local, |t, &l| *t = s * l);
        }
    }

    let n_ch = signal.nrows();
    let mut f0_rows = Vec::with_capacity(n_ch);
    let mut flag_rows = Vec::with_capacity(n_ch);
    let mut prob_rows = Vec::with_capacity(n_ch);
    for row in signal.rows() {
        let y = row.to_vec();
        let yf = yin_frames(
            &y,
            sample_rate,
            config.fmin,
            config.fmax,
            fl,
            hop,
            config.center,
            config.pad_mode,
        )?;
        let n_frames = yf.cmnd.len();
        let mut obs = Array2::<f64>::zeros((n_states, n_frames));
        let mut voiced_prob = vec![0.0; n_frames];
        for (t, (c, s)) in yf.cmnd.iter().zip(&yf.shifts).enumerate() {
            for (i, p) in pyin_frame_candidates(
                c,
                &thresholds,
                &beta_probs,
                config.boltzmann_parameter,
                config.no_trough_prob,
            ) {
                let period = yf.min_period as f64 + i as f64 + s[i];
                let f0 = sample_rate / period;
                let bin = (bins_per_octave * (f0 / config.fmin).log2()).round_ties_even();
                let bin = bin.clamp(0.0, n_bins as f64) as usize;
                // Bin n_bins (above fmax after refinement) lands in the first unvoiced
                // row in librosa and is overwritten below, i.e. dropped.
                if bin < n_bins {
                    obs[[bin, t]] = p;
                }
            }
            let mut sum = 0.0;
            for k in 0..n_bins {
                sum += obs[[k, t]];
            }
            let vp = sum.clamp(0.0, 1.0);
            voiced_prob[t] = vp;
            let unvoiced = (1.0 - vp) / n_bins as f64;
            for k in n_bins..n_states {
                obs[[k, t]] = unvoiced;
            }
        }
        let path = viterbi(
            obs.view(),
            transition.view(),
            None,
            config.transition_min_prob,
        )?;
        let mut f0 = Vec::with_capacity(n_frames);
        let mut flag = Vec::with_capacity(n_frames);
        for &state in &path.states {
            let voiced = state < n_bins;
            let guess = freqs[state % n_bins];
            f0.push(match (voiced, config.fill) {
                (true, _) | (false, UnvoicedFill::BestGuess) => guess,
                (false, UnvoicedFill::Nan) => f64::NAN,
                (false, UnvoicedFill::Value(v)) => v,
            });
            flag.push(voiced);
        }
        f0_rows.push(f0);
        flag_rows.push(flag);
        prob_rows.push(voiced_prob);
    }
    Ok(PyinResult {
        f0: stack_rows(f0_rows)?,
        voiced_flag: stack_rows(flag_rows)?,
        voiced_prob: stack_rows(prob_rows)?,
    })
}

// =============================================================================
// Harmonic helpers (librosa.core.harmonic)
// =============================================================================

/// The frequency of every element of a spectrogram, for [`interp_harmonics`],
/// [`salience`] and [`f0_harmonics`] (librosa's `freqs` argument).
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::pitch_tracking::FrequencyGrid;
/// use ndarray::array;
///
/// let f = array![0.0, 100.0, 200.0];
/// let grid = FrequencyGrid::Fixed(f.view());
/// assert!(matches!(grid, FrequencyGrid::Fixed(_)));
/// ```
#[derive(Debug, Clone, Copy)]
pub enum FrequencyGrid<'a> {
    /// One frequency per bin shared by all frames (e.g. `fft_frequencies`),
    /// length `n_bins`.
    Fixed(ArrayView1<'a, f64>),
    /// A frequency per element, shaped like the spectrogram
    /// `(channels, n_bins, n_frames)` (e.g. a reassigned spectrogram).
    TimeVarying(ArrayView3<'a, f64>),
}

impl FrequencyGrid<'_> {
    fn check(&self, shape: (usize, usize, usize)) -> AudioSampleResult<()> {
        match self {
            Self::Fixed(f) if f.len() == shape.1 => Ok(()),
            Self::TimeVarying(f) if f.dim() == shape => Ok(()),
            Self::Fixed(f) => Err(param_err(
                "freqs",
                format!(
                    "length {} is incompatible with {} frequency bins",
                    f.len(),
                    shape.1
                ),
            )),
            Self::TimeVarying(f) => Err(param_err(
                "freqs",
                format!(
                    "shape {:?} is incompatible with input shape {shape:?}",
                    f.dim()
                ),
            )),
        }
    }

    /// The grid of channel `c`, frame `t`.
    fn column(&self, c: usize, t: usize) -> Vec<f64> {
        match self {
            Self::Fixed(f) => f.to_vec(),
            Self::TimeVarying(f) => f.slice(ndarray::s![c, .., t]).to_vec(),
        }
    }
}

/// `numpy.argsort(x, kind="mergesort")` (stable, NaN last).
fn stable_argsort(x: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&i, &j| x[i].total_cmp(&x[j]).then(std::cmp::Ordering::Equal));
    // total_cmp orders -NaN first; NumPy puts every NaN last.
    let (nan, rest): (Vec<usize>, Vec<usize>) = idx.into_iter().partition(|&i| x[i].is_nan());
    rest.into_iter().chain(nan).collect()
}

/// A sorted linear interpolant, `scipy.interpolate.interp1d(kind="linear",
/// bounds_error=False, fill_value=fill)` on one column.
struct Interp1d {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Interp1d {
    fn new(x: &[f64], y: &[f64]) -> Self {
        let order = stable_argsort(x);
        Self {
            x: order.iter().map(|&i| x[i]).collect(),
            y: order.iter().map(|&i| y[i]).collect(),
        }
    }

    fn out_of_bounds(&self, v: f64) -> bool {
        v < self.x[0] || v > self.x[self.x.len() - 1]
    }

    /// `numpy.interp` (the path SciPy takes for 1-D float64 data).
    fn eval_np(&self, v: f64, fill: f64) -> f64 {
        let (x, y) = (&self.x, &self.y);
        let n = x.len();
        if v.is_nan() {
            return v;
        }
        if self.out_of_bounds(v) {
            return fill;
        }
        // j with x[j] <= v < x[j + 1].
        let j = x.partition_point(|&xi| xi <= v).saturating_sub(1);
        if j >= n - 1 || x[j] == v {
            return y[j];
        }
        let slope = (y[j + 1] - y[j]) / (x[j + 1] - x[j]);
        let mut r = slope * (v - x[j]) + y[j];
        if r.is_nan() {
            r = slope * (v - x[j + 1]) + y[j + 1];
            if r.is_nan() && y[j] == y[j + 1] {
                r = y[j];
            }
        }
        r
    }

    /// SciPy's `_call_linear` (the path taken for 2-D data).
    fn eval_scipy(&self, v: f64, fill: f64) -> f64 {
        let (x, y) = (&self.x, &self.y);
        let n = x.len();
        if self.out_of_bounds(v) {
            return fill;
        }
        // searchsorted(side="left"), NaN sorting last; then clip to [1, n - 1].
        let idx = if v.is_nan() {
            n
        } else {
            x.partition_point(|&xi| xi < v)
        };
        let hi = idx.clamp(1, n - 1);
        let lo = hi - 1;
        let (xl, xh) = (x[lo], x[hi]);
        ((v - xl) / (xh - xl)) * y[hi] + ((xh - v) / (xh - xl)) * y[lo]
    }
}

fn check_harmonics(harmonics: &[f64]) -> AudioSampleResult<()> {
    if harmonics.is_empty() {
        return Err(param_err("harmonics", "must contain at least one harmonic"));
    }
    if harmonics.iter().any(|h| !h.is_finite()) {
        return Err(param_err("harmonics", "harmonics must be finite"));
    }
    Ok(())
}

/// Energy of a spectrogram at harmonics of every frequency, equivalent to
/// `librosa.interp_harmonics(x, freqs=freqs, harmonics=harmonics, kind="linear",
/// fill_value=fill_value, axis=-2)`.
///
/// `out[c, h, k, t]` is the linear interpolation of `x[c, :, t]` (as a function
/// of frequency) at `harmonics[h] * freqs[k]`; targets outside the frequency
/// range get `fill_value`. Harmonics below 1 give sub-harmonics.
///
/// # Arguments
///
/// * `x` - `(channels, n_bins, n_frames)` energy representation (e.g. STFT magnitudes).
/// * `freqs` - The frequency of each bin, fixed or time-varying.
/// * `harmonics` - Harmonic multipliers (non-empty, finite).
/// * `fill_value` - Value for targets outside the frequency range (librosa default 0).
///
/// # Returns
///
/// A `(channels, harmonics.len(), n_bins, n_frames)` array.
///
/// # Errors
///
/// [`ParameterError`] if `freqs` does not match `x`, there are fewer than two
/// bins, or `harmonics` is empty or non-finite.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::pitch_tracking::{FrequencyGrid, interp_harmonics};
/// use ndarray::{Array3, array};
///
/// let freqs = array![0.0, 100.0, 200.0, 300.0];
/// // One channel, one frame: energy rises linearly with frequency.
/// let x = Array3::from_shape_vec((1, 4, 1), vec![0.0, 1.0, 2.0, 3.0]).unwrap();
/// let h = interp_harmonics(x.view(), FrequencyGrid::Fixed(freqs.view()), &[0.5, 2.0], 0.0).unwrap();
/// assert_eq!(h.dim(), (1, 2, 4, 1));
/// assert_eq!(h[[0, 0, 1, 0]], 0.5); // x at 50 Hz
/// assert_eq!(h[[0, 1, 1, 0]], 2.0); // x at 200 Hz
/// assert_eq!(h[[0, 1, 2, 0]], 0.0); // 400 Hz is out of range
/// ```
pub fn interp_harmonics(
    x: ArrayView3<f64>,
    freqs: FrequencyGrid<'_>,
    harmonics: &[f64],
    fill_value: f64,
) -> AudioSampleResult<Array4<f64>> {
    let (n_ch, n_bins, n_frames) = x.dim();
    freqs.check(x.dim())?;
    check_harmonics(harmonics)?;
    if n_bins < 2 {
        return Err(param_err("x", "need at least two frequency bins"));
    }
    let n_h = harmonics.len();
    let mut out = Array4::<f64>::zeros((n_ch, n_h, n_bins, n_frames));
    for c in 0..n_ch {
        for t in 0..n_frames {
            let grid = freqs.column(c, t);
            let col = x.slice(ndarray::s![c, .., t]).to_vec();
            let interp = Interp1d::new(&grid, &col);
            for (hi, &h) in harmonics.iter().enumerate() {
                for (k, &f) in grid.iter().enumerate() {
                    let target = h * f;
                    out[[c, hi, k, t]] = match freqs {
                        // 2-D data in SciPy: de Boor's formula.
                        FrequencyGrid::Fixed(_) => interp.eval_scipy(target, fill_value),
                        // Per-frame 1-D data: numpy.interp.
                        FrequencyGrid::TimeVarying(_) => interp.eval_np(target, fill_value),
                    };
                }
            }
        }
    }
    Ok(out)
}

/// How [`salience`] combines the harmonics (librosa's `aggregate`).
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::pitch_tracking::SalienceAggregate;
///
/// assert_eq!(SalienceAggregate::default(), SalienceAggregate::Average);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SalienceAggregate {
    /// Weighted average with [`SalienceConfig::weights`] (`numpy.average`, librosa's default).
    #[default]
    Average,
    /// Unweighted sum (`numpy.sum`).
    Sum,
    /// Maximum (`numpy.max`; NaN propagates).
    Max,
    /// Minimum (`numpy.min`; NaN propagates).
    Min,
    /// Median (`numpy.median`).
    Median,
}

/// Options of [`salience`], mirroring `librosa.salience`'s keyword arguments.
///
/// ## Purpose
///
/// Selects the harmonic weights, how harmonics are aggregated and whether the
/// result is restricted to spectral peaks.
///
/// ## Intended usage
///
/// `Default` reproduces librosa's defaults: uniform weights, weighted average,
/// output only at peaks (local maxima along frequency) and NaN elsewhere.
///
/// ## Invariants
///
/// Checked by [`SalienceConfig::validate`]: `weights`, if given, has one finite
/// entry per harmonic and (for [`SalienceAggregate::Average`]) a non-zero sum.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::pitch_tracking::{SalienceAggregate, SalienceConfig};
///
/// let cfg = SalienceConfig::new()
///     .with_weights(Some(vec![1.0, 0.5, 0.33, 0.25]))
///     .with_fill_value(0.0);
/// assert!(cfg.validate(4).is_ok());
/// assert!(cfg.validate(3).is_err());
/// assert_eq!(cfg.aggregate, SalienceAggregate::Average);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SalienceConfig {
    /// Weight of each harmonic in [`SalienceAggregate::Average`]; `None` = uniform.
    /// Ignored by the other aggregates (as in librosa).
    pub weights: Option<Vec<f64>>,
    /// How the harmonics are combined.
    pub aggregate: SalienceAggregate,
    /// If `true`, only frequencies that are strict local maxima of the input along
    /// frequency keep their salience; all others get `fill_value`.
    pub filter_peaks: bool,
    /// Value of non-peak elements when `filter_peaks` is `true`. librosa default: NaN.
    pub fill_value: f64,
}

impl Default for SalienceConfig {
    fn default() -> Self {
        Self {
            weights: None,
            aggregate: SalienceAggregate::Average,
            filter_peaks: true,
            fill_value: f64::NAN,
        }
    }
}

impl SalienceConfig {
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the harmonic weights (`None` = uniform).
    #[must_use]
    pub fn with_weights(mut self, weights: Option<Vec<f64>>) -> Self {
        self.weights = weights;
        self
    }

    /// Sets the aggregation.
    #[must_use]
    pub fn with_aggregate(mut self, aggregate: SalienceAggregate) -> Self {
        self.aggregate = aggregate;
        self
    }

    /// Sets whether the output is restricted to peaks.
    #[must_use]
    pub fn with_filter_peaks(mut self, filter_peaks: bool) -> Self {
        self.filter_peaks = filter_peaks;
        self
    }

    /// Sets the fill value of non-peak elements.
    #[must_use]
    pub fn with_fill_value(mut self, fill_value: f64) -> Self {
        self.fill_value = fill_value;
        self
    }

    /// Checks the configuration for `n_harmonics` harmonics.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `weights` has the wrong length, a non-finite entry,
    /// or (with [`SalienceAggregate::Average`]) sums to zero.
    pub fn validate(&self, n_harmonics: usize) -> AudioSampleResult<()> {
        if let Some(w) = &self.weights {
            if w.len() != n_harmonics {
                return Err(param_err(
                    "weights",
                    format!("length {} does not match {n_harmonics} harmonics", w.len()),
                ));
            }
            if w.iter().any(|v| !v.is_finite()) {
                return Err(param_err("weights", "weights must be finite"));
            }
            if self.aggregate == SalienceAggregate::Average && w.iter().sum::<f64>() == 0.0 {
                return Err(param_err(
                    "weights",
                    "weights sum to zero and cannot be normalised",
                ));
            }
        }
        Ok(())
    }
}

fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

fn nan_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

fn median(v: &mut [f64]) -> f64 {
    if v.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Harmonic salience of a time-frequency representation, equivalent to
/// `librosa.salience(S, freqs=freqs, harmonics=harmonics, weights=...,
/// aggregate=..., filter_peaks=..., fill_value=..., kind="linear", axis=-2)`.
///
/// For every bin, the energy at its harmonics (from [`interp_harmonics`] with
/// fill 0) is aggregated; with `filter_peaks`, only bins that are strict local
/// maxima of `s` along frequency (`scipy.signal.argrelmax`) keep the value.
///
/// # Arguments
///
/// * `s` - `(channels, n_bins, n_frames)` non-negative magnitudes.
/// * `freqs` - The frequency of each bin, fixed or time-varying.
/// * `harmonics` - Harmonic multipliers; `1` is `s` itself.
/// * `config` - Weights, aggregation and peak filtering.
///
/// # Returns
///
/// A `(channels, n_bins, n_frames)` salience array.
///
/// # Errors
///
/// [`ParameterError`] for the input errors of [`interp_harmonics`] or an
/// invalid `config`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::pitch_tracking::{FrequencyGrid, SalienceConfig, salience};
/// use ndarray::{Array3, array};
///
/// let freqs = array![0.0, 100.0, 200.0, 300.0, 400.0, 500.0];
/// // A 100 Hz tone with harmonics: peaks at 100, 200 and 300 Hz.
/// let s = Array3::from_shape_vec((1, 6, 1), vec![0.0, 1.0, 0.5, 0.6, 0.1, 0.0]).unwrap();
/// let cfg = SalienceConfig::new().with_fill_value(0.0);
/// let sal = salience(s.view(), FrequencyGrid::Fixed(freqs.view()), &[1.0, 2.0, 3.0], &cfg).unwrap();
/// assert_eq!(sal.dim(), (1, 6, 1));
/// assert!((sal[[0, 1, 0]] - (1.0 + 0.5 + 0.6) / 3.0).abs() < 1e-15);
/// assert_eq!(sal[[0, 2, 0]], 0.0); // not a peak
/// ```
pub fn salience(
    s: ArrayView3<f64>,
    freqs: FrequencyGrid<'_>,
    harmonics: &[f64],
    config: &SalienceConfig,
) -> AudioSampleResult<Array3<f64>> {
    check_harmonics(harmonics)?;
    config.validate(harmonics.len())?;
    let s_harm = interp_harmonics(s, freqs, harmonics, 0.0)?;
    let (n_ch, n_bins, n_frames) = s.dim();
    let n_h = harmonics.len();
    let weights: Vec<f64> = config.weights.clone().unwrap_or_else(|| vec![1.0; n_h]);
    let mut scl = 0.0;
    for &w in &weights {
        scl += w;
    }
    let mut out = Array3::<f64>::zeros((n_ch, n_bins, n_frames));
    let mut buf = vec![0.0; n_h];
    for c in 0..n_ch {
        for k in 0..n_bins {
            for t in 0..n_frames {
                for (h, b) in buf.iter_mut().enumerate() {
                    *b = s_harm[[c, h, k, t]];
                }
                out[[c, k, t]] = match config.aggregate {
                    SalienceAggregate::Average => {
                        let mut acc = 0.0;
                        for (v, w) in buf.iter().zip(&weights) {
                            acc += v * w;
                        }
                        acc / scl
                    }
                    SalienceAggregate::Sum => {
                        let mut acc = 0.0;
                        for v in &buf {
                            acc += v;
                        }
                        acc
                    }
                    SalienceAggregate::Max => buf[1..].iter().fold(buf[0], |m, &v| nan_max(m, v)),
                    SalienceAggregate::Min => buf[1..].iter().fold(buf[0], |m, &v| nan_min(m, v)),
                    SalienceAggregate::Median => median(&mut buf),
                };
            }
        }
    }
    if config.filter_peaks {
        for c in 0..n_ch {
            for t in 0..n_frames {
                for k in 0..n_bins {
                    let peak = k > 0
                        && k + 1 < n_bins
                        && s[[c, k, t]] > s[[c, k - 1, t]]
                        && s[[c, k, t]] > s[[c, k + 1, t]];
                    if !peak {
                        out[[c, k, t]] = config.fill_value;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Energy at harmonics of a time-varying fundamental, equivalent to
/// `librosa.f0_harmonics(x, f0=f0, freqs=freqs, harmonics=harmonics,
/// kind="linear", fill_value=fill_value, axis=-2)`.
///
/// `out[c, h, t]` is the linear interpolation of `x[c, :, t]` at
/// `harmonics[h] * f0[c, t]`. Non-finite grid frequencies are ignored; targets
/// outside the grid, and NaN results (e.g. a NaN `f0` from an unvoiced
/// [`pyin`] frame), get `fill_value`; infinite results are clamped to
/// `±f64::MAX` (`numpy.nan_to_num`).
///
/// # Arguments
///
/// * `x` - `(channels, n_bins, n_frames)` energy representation.
/// * `f0` - `(channels, n_frames)` fundamental frequency per frame.
/// * `freqs` - The frequency of each bin, fixed or time-varying.
/// * `harmonics` - Harmonic multipliers (non-empty, finite).
/// * `fill_value` - Value for out-of-range or undefined targets (librosa default 0).
///
/// # Returns
///
/// A `(channels, harmonics.len(), n_frames)` array.
///
/// # Errors
///
/// [`ParameterError`] if shapes are incompatible, `harmonics` is empty or
/// non-finite, or a frame has no finite grid frequency.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::pitch_tracking::{FrequencyGrid, f0_harmonics};
/// use ndarray::{Array2, Array3, array};
///
/// let freqs = array![0.0, 100.0, 200.0, 300.0, 400.0];
/// let x = Array3::from_shape_vec((1, 5, 2), vec![0., 0., 1., 1., 2., 2., 3., 3., 4., 4.]).unwrap();
/// let f0 = Array2::from_shape_vec((1, 2), vec![150.0, f64::NAN]).unwrap();
/// let h = f0_harmonics(x.view(), f0.view(), FrequencyGrid::Fixed(freqs.view()), &[1.0, 2.0], 0.0)
///     .unwrap();
/// assert_eq!(h.dim(), (1, 2, 2));
/// assert_eq!(h[[0, 0, 0]], 1.5); // x at 150 Hz
/// assert_eq!(h[[0, 1, 0]], 3.0); // x at 300 Hz
/// assert_eq!(h[[0, 0, 1]], 0.0); // unvoiced frame
/// ```
pub fn f0_harmonics(
    x: ArrayView3<f64>,
    f0: ArrayView2<f64>,
    freqs: FrequencyGrid<'_>,
    harmonics: &[f64],
    fill_value: f64,
) -> AudioSampleResult<Array3<f64>> {
    let (n_ch, _, n_frames) = x.dim();
    freqs.check(x.dim())?;
    check_harmonics(harmonics)?;
    if f0.dim() != (n_ch, n_frames) {
        return Err(param_err(
            "f0",
            format!(
                "shape {:?} must be (channels, n_frames) = ({n_ch}, {n_frames})",
                f0.dim()
            ),
        ));
    }
    let n_h = harmonics.len();
    let mut out = Array3::<f64>::zeros((n_ch, n_h, n_frames));
    for c in 0..n_ch {
        for t in 0..n_frames {
            let grid = freqs.column(c, t);
            let col = x.slice(ndarray::s![c, .., t]);
            let (gx, gy): (Vec<f64>, Vec<f64>) = grid
                .iter()
                .zip(col.iter())
                .filter(|(f, _)| f.is_finite())
                .map(|(&f, &v)| (f, v))
                .unzip();
            if gx.is_empty() {
                return Err(param_err(
                    "freqs",
                    format!("frame {t} of channel {c} has no finite frequency"),
                ));
            }
            let interp = Interp1d::new(&gx, &gy);
            for (hi, &h) in harmonics.iter().enumerate() {
                let v = interp.eval_np(h * f0[[c, t]], fill_value);
                out[[c, hi, t]] = if v.is_nan() {
                    fill_value
                } else if v == f64::INFINITY {
                    f64::MAX
                } else if v == f64::NEG_INFINITY {
                    f64::MIN
                } else {
                    v
                };
            }
        }
    }
    Ok(out)
}

// =============================================================================
// Trait
// =============================================================================

/// librosa-compatible frame-wise pitch tracking (`librosa.yin`, `librosa.pyin`).
///
/// # Purpose
///
/// Estimates a fundamental-frequency track per channel with librosa's exact
/// framing and decision rules, so results can be compared with (or substituted
/// for) librosa's.
///
/// # Intended Usage
///
/// Use [`yin`](Self::yin) for a fast, always-defined f0 per frame, and
/// [`pyin`](Self::pyin) when voicing decisions and a smooth track matter (speech,
/// singing, monophonic instruments). For single-buffer detection with `Option`
/// results see [`AudioPitchAnalysis`](crate::AudioPitchAnalysis).
///
/// # Invariants
///
/// Outputs are `(channels, n_frames)`; channels are processed independently.
/// Samples are converted to `f64` with the crate's audio-aware conversion.
pub trait AudioPitchTracking {
    /// YIN f0 estimation per frame and channel, equivalent to `librosa.yin`.
    ///
    /// # Arguments
    ///
    /// * `config` - YIN parameters ([`YinConfig::default`] = librosa defaults with
    ///   `fmin = C2`, `fmax = C7`).
    ///
    /// # Returns
    ///
    /// A `(channels, n_frames)` array of f0 values in Hz; see [`yin`].
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the configuration is invalid for the sample rate or
    /// (uncentred) the signal is shorter than `frame_length`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::pitch_tracking::{AudioPitchTracking, YinConfig};
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::Array2;
    ///
    /// let sr = 22050.0;
    /// let data = Array2::from_shape_fn((2, 11025), |(c, n)| {
    ///     let f = if c == 0 { 220.0 } else { 330.0 };
    ///     (2.0 * std::f64::consts::PI * f * n as f64 / sr).sin() as f32
    /// });
    /// let audio = AudioSamples::new_multi_channel(data, sample_rate!(22050)).unwrap();
    /// let f0 = audio.yin(&YinConfig::default()).unwrap();
    /// assert_eq!(f0.dim(), (2, 22));
    /// assert!((f0[[0, 10]] - 220.0).abs() < 0.5);
    /// assert!((f0[[1, 10]] - 330.0).abs() < 0.5);
    /// ```
    fn yin(&self, config: &YinConfig) -> AudioSampleResult<Array2<f64>>;

    /// Probabilistic YIN per frame and channel, equivalent to `librosa.pyin`.
    ///
    /// # Arguments
    ///
    /// * `config` - pYIN parameters ([`PyinConfig::default`] = librosa defaults
    ///   with `fmin = C2`, `fmax = C7`).
    ///
    /// # Returns
    ///
    /// The f0 track, voiced flags and voicing probabilities, each
    /// `(channels, n_frames)`; see [`pyin`].
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the configuration is invalid for the sample rate or
    /// (uncentred) the signal is shorter than `frame_length`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::pitch_tracking::{AudioPitchTracking, PyinConfig};
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::Array1;
    ///
    /// // 0.25 s of silence, then 0.25 s of a 261.6 Hz tone.
    /// let sr = 22050.0;
    /// let y = Array1::from_shape_fn(11025, |n| {
    ///     if n < 5512 { 0.0 } else { 0.5 * (2.0 * std::f64::consts::PI * 261.6 * n as f64 / sr).sin() }
    /// });
    /// let audio = AudioSamples::new_mono(y, sample_rate!(22050)).unwrap();
    /// let r = audio.pyin(&PyinConfig::default()).unwrap();
    /// assert!(!r.voiced_flag[[0, 2]] && r.f0[[0, 2]].is_nan());
    /// assert!(r.voiced_flag[[0, 18]]);
    /// ```
    fn pyin(&self, config: &PyinConfig) -> AudioSampleResult<PyinResult>;
}

/// The audio as a `(channels, samples)` `f64` array.
fn audio_f64_rows<T: StandardSample>(audio: &AudioSamples<'_, T>) -> Array2<f64> {
    let w = audio.to_format::<f64>();
    if let Some(m) = w.as_mono() {
        m.as_view().insert_axis(Axis(0)).to_owned()
    } else if let Some(m) = w.as_multi_channel() {
        m.as_view().to_owned()
    } else {
        unreachable!("audio is either mono or multi-channel")
    }
}

impl<T> AudioPitchTracking for AudioSamples<'_, T>
where
    T: StandardSample,
{
    fn yin(&self, config: &YinConfig) -> AudioSampleResult<Array2<f64>> {
        yin(audio_f64_rows(self).view(), self.sample_rate_hz(), config)
    }

    fn pyin(&self, config: &PyinConfig) -> AudioSampleResult<PyinResult> {
        pyin(audio_f64_rows(self).view(), self.sample_rate_hz(), config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample_rate;
    use ndarray::{Array1, array};

    fn tone(f: f64, n: usize, sr: f64) -> Array2<f64> {
        Array2::from_shape_fn((1, n), |(_, i)| {
            0.5 * (2.0 * std::f64::consts::PI * f * i as f64 / sr).sin()
        })
    }

    #[test]
    fn next_fast_len_matches_scipy() {
        // scipy.fft.next_fast_len(n, real=True)
        for (n, e) in [
            (1, 1),
            (7, 8),
            (4095, 4096),
            (2047, 2048),
            (1001, 1024),
            (1201, 1215),
            (31, 32),
            (11, 12),
        ] {
            assert_eq!(next_fast_len_real(n), e, "n = {n}");
        }
    }

    #[test]
    fn beta_cdf_closed_forms() {
        // Beta(1, 1) is uniform; Beta(2, 1) has cdf x^2; Beta(1, 2) has 1 - (1-x)^2.
        for &x in &[0.0, 0.1, 0.37, 0.5, 0.9, 1.0] {
            assert!((beta_cdf(x, 1.0, 1.0) - x).abs() < 1e-15);
            assert!((beta_cdf(x, 2.0, 1.0) - x * x).abs() < 1e-15);
            assert!((beta_cdf(x, 1.0, 2.0) - (1.0 - (1.0 - x) * (1.0 - x))).abs() < 1e-15);
            // The continued-fraction path agrees with the exact sum.
            let cf = |x: f64, a: f64, b: f64| {
                let lf =
                    ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (-x).ln_1p();
                if x <= 0.0 || x >= 1.0 {
                    beta_cdf(x, a, b)
                } else if x < (a + 1.0) / (a + b + 2.0) {
                    lf.exp() * beta_cf(a, b, x) / a
                } else {
                    1.0 - lf.exp() * beta_cf(b, a, 1.0 - x) / b
                }
            };
            assert!((cf(x, 2.0, 18.0) - beta_cdf(x, 2.0, 18.0)).abs() < 1e-13);
        }
        // Symmetry I_x(a, b) = 1 - I_{1-x}(b, a) on the non-integer path.
        let v = beta_cdf(0.3, 1.5, 12.5) + beta_cdf(0.7, 12.5, 1.5);
        assert!((v - 1.0).abs() < 1e-13);
    }

    #[test]
    fn ln_gamma_values() {
        assert!(ln_gamma(1.0).abs() < 1e-14);
        assert!(ln_gamma(2.0).abs() < 1e-14);
        assert!((ln_gamma(5.0) - 24f64.ln()).abs() < 1e-13);
        assert!((ln_gamma(0.5) - std::f64::consts::PI.sqrt().ln()).abs() < 1e-14);
    }

    #[test]
    fn boltzmann_pmf_sums_to_one() {
        let s: f64 = (0..5).map(|k| boltzmann_pmf(k, 2.0, 5)).sum();
        assert!((s - 1.0).abs() < 1e-15);
    }

    #[test]
    fn np_sum_matches_sequential_for_short_and_is_accurate_for_long() {
        let a: Vec<f64> = (0..5).map(f64::from).collect();
        assert_eq!(np_sum(&a), 10.0);
        let b: Vec<f64> = (0..300).map(|i| 0.1 * f64::from(i)).collect();
        assert!((np_sum(&b) - 4485.0).abs() < 1e-10);
    }

    #[test]
    fn parabolic_shift_finds_vertex() {
        // Parabola with its vertex at 2.25.
        let x: Vec<f64> = (0..5).map(|i| (f64::from(i) - 2.25).powi(2)).collect();
        let s = parabolic_shifts(&x);
        assert_eq!(s[0], 0.0);
        assert_eq!(s[4], 0.0);
        assert!((s[2] - 0.25).abs() < 1e-14);
    }

    #[test]
    fn yin_shapes_and_accuracy() {
        let sr = 22050.0;
        let y = tone(440.0, 22050, sr);
        let f0 = yin(y.view(), sr, &YinConfig::default()).unwrap();
        assert_eq!(f0.dim(), (1, 44));
        for t in 4..40 {
            // YIN's parabolic refinement is within a few cents of the true pitch.
            assert!(
                (f0[[0, t]] - 440.0).abs() < 1.0,
                "frame {t}: {}",
                f0[[0, t]]
            );
        }
        let nc = YinConfig::default().with_center(false);
        let f0 = yin(y.view(), sr, &nc).unwrap();
        assert_eq!(f0.dim(), (1, 1 + (22050 - 2048) / 512));
    }

    #[test]
    fn yin_silence_falls_back_to_shortest_period() {
        let sr = 22050.0;
        let y = Array2::<f64>::zeros((1, 4096));
        let f0 = yin(y.view(), sr, &YinConfig::default()).unwrap();
        let min_period = (sr / C7_HZ).floor();
        assert!(f0.iter().all(|&v| v == sr / min_period));
    }

    #[test]
    fn yin_rejects_bad_parameters() {
        let sr = 22050.0;
        let y = tone(440.0, 4096, sr);
        let bad = [
            YinConfig::default().with_fmax(20000.0),
            YinConfig::default().with_fmin(3000.0),
            YinConfig::default().with_fmin(0.0),
            YinConfig::default().with_fmin(10.0),
            YinConfig::default().with_trough_threshold(f64::NAN),
            YinConfig::default().with_frame_length(crate::nzu!(2)),
        ];
        for cfg in bad {
            assert!(yin(y.view(), sr, &cfg).is_err(), "{cfg:?}");
        }
        // Uncentred and too short.
        let short = tone(440.0, 1000, sr);
        assert!(yin(short.view(), sr, &YinConfig::default().with_center(false)).is_err());
        // Non-finite input.
        let mut nan = tone(440.0, 4096, sr);
        nan[[0, 10]] = f64::NAN;
        assert!(yin(nan.view(), sr, &YinConfig::default()).is_err());
    }

    #[test]
    fn pyin_voicing_and_fill() {
        let sr = 22050.0;
        let mut y = tone(261.63, 11025, sr);
        y.slice_mut(ndarray::s![0, ..5512]).fill(0.0);
        let r = pyin(y.view(), sr, &PyinConfig::default()).unwrap();
        assert_eq!(r.f0.dim(), (1, 22));
        assert!(!r.voiced_flag[[0, 1]]);
        assert!(r.f0[[0, 1]].is_nan());
        assert!(r.voiced_flag[[0, 18]]);
        assert!(((r.f0[[0, 18]] / 261.63).log2() * 1200.0).abs() < 6.0);
        assert!(r.voiced_prob.iter().all(|p| (0.0..=1.0).contains(p)));

        let r0 = pyin(
            y.view(),
            sr,
            &PyinConfig::default().with_fill(UnvoicedFill::Value(0.0)),
        )
        .unwrap();
        assert_eq!(r0.f0[[0, 1]], 0.0);
        let rg = pyin(
            y.view(),
            sr,
            &PyinConfig::default().with_fill(UnvoicedFill::BestGuess),
        )
        .unwrap();
        assert!(rg.f0.iter().all(|v| v.is_finite()));
        assert_eq!(rg.voiced_flag, r.voiced_flag);
    }

    #[test]
    fn pyin_rejects_bad_parameters() {
        let sr = 22050.0;
        let y = tone(440.0, 4096, sr);
        let bad = [
            PyinConfig::default().with_beta_parameters(0.0, 18.0),
            PyinConfig::default().with_boltzmann_parameter(-1.0),
            PyinConfig::default().with_resolution(1.0),
            PyinConfig::default().with_max_transition_rate(0.0),
            PyinConfig::default().with_switch_prob(1.5),
            PyinConfig::default().with_no_trough_prob(-0.1),
            PyinConfig::default().with_transition_min_prob(Some(-1.0)),
            PyinConfig::default().with_fill(UnvoicedFill::Value(f64::INFINITY)),
            PyinConfig::default().with_fmax(30000.0),
        ];
        for cfg in bad {
            assert!(pyin(y.view(), sr, &cfg).is_err(), "{cfg:?}");
        }
    }

    #[test]
    fn trait_matches_free_functions_and_handles_stereo() {
        let a = tone(220.0, 6000, 22050.0);
        let b = tone(330.0, 6000, 22050.0);
        let mut data = Array2::<f32>::zeros((2, 6000));
        for i in 0..6000 {
            data[[0, i]] = a[[0, i]] as f32;
            data[[1, i]] = b[[0, i]] as f32;
        }
        let audio = AudioSamples::new_multi_channel(data.clone(), sample_rate!(22050)).unwrap();
        let f0 = audio.yin(&YinConfig::default()).unwrap();
        let rows = data.mapv(f64::from);
        assert_eq!(
            f0,
            yin(rows.view(), 22050.0, &YinConfig::default()).unwrap()
        );
        let single = yin(
            rows.slice(ndarray::s![1..2, ..]),
            22050.0,
            &YinConfig::default(),
        )
        .unwrap();
        assert_eq!(f0.row(1), single.row(0));
        let r = audio.pyin(&PyinConfig::default()).unwrap();
        assert_eq!(r.f0.nrows(), 2);
    }

    #[test]
    fn interp_and_salience_basics() {
        let freqs = array![0.0, 100.0, 200.0, 300.0, 400.0];
        let x = Array3::from_shape_vec((1, 5, 1), vec![0.0, 1.0, 0.2, 0.8, 0.1]).unwrap();
        let h =
            interp_harmonics(x.view(), FrequencyGrid::Fixed(freqs.view()), &[1.0], 0.0).unwrap();
        assert_eq!(h.index_axis(Axis(1), 0), x);
        // Time-varying grid equal to the fixed one gives the same result.
        let tv = Array3::from_shape_fn((1, 5, 1), |(_, k, _)| freqs[k]);
        let h2 = interp_harmonics(
            x.view(),
            FrequencyGrid::TimeVarying(tv.view()),
            &[0.5, 1.5],
            -1.0,
        )
        .unwrap();
        let h1 = interp_harmonics(
            x.view(),
            FrequencyGrid::Fixed(freqs.view()),
            &[0.5, 1.5],
            -1.0,
        )
        .unwrap();
        for (a, b) in h1.iter().zip(&h2) {
            assert!((a - b).abs() < 1e-15);
        }
        assert_eq!(h1[[0, 1, 4, 0]], -1.0);

        let sal = salience(
            x.view(),
            FrequencyGrid::Fixed(freqs.view()),
            &[1.0, 2.0],
            &SalienceConfig::default(),
        )
        .unwrap();
        assert!(sal[[0, 0, 0]].is_nan() && sal[[0, 4, 0]].is_nan() && sal[[0, 2, 0]].is_nan());
        assert!((sal[[0, 1, 0]] - 0.6).abs() < 1e-15);
        assert!((sal[[0, 3, 0]] - 0.4).abs() < 1e-15);
    }

    #[test]
    fn harmonic_helpers_reject_bad_shapes() {
        let freqs = array![0.0, 100.0, 200.0];
        let x = Array3::<f64>::zeros((1, 4, 2));
        let g = FrequencyGrid::Fixed(freqs.view());
        assert!(interp_harmonics(x.view(), g, &[1.0], 0.0).is_err());
        assert!(salience(x.view(), g, &[1.0], &SalienceConfig::default()).is_err());
        let f4 = array![0.0, 1.0, 2.0, 3.0];
        let g4 = FrequencyGrid::Fixed(f4.view());
        assert!(interp_harmonics(x.view(), g4, &[], 0.0).is_err());
        let f0 = Array2::<f64>::zeros((1, 3));
        assert!(f0_harmonics(x.view(), f0.view(), g4, &[1.0], 0.0).is_err());
        let cfg = SalienceConfig::default().with_weights(Some(vec![1.0, -1.0]));
        assert!(salience(x.view(), g4, &[1.0, 2.0], &cfg).is_err());
        let all_nan = Array1::from_elem(4, f64::NAN);
        let f0 = Array2::<f64>::zeros((1, 2));
        assert!(
            f0_harmonics(
                x.view(),
                f0.view(),
                FrequencyGrid::Fixed(all_nan.view()),
                &[1.0],
                0.0
            )
            .is_err()
        );
    }

    #[test]
    fn salience_aggregates() {
        let freqs = array![0.0, 100.0, 200.0, 300.0, 400.0];
        let x = Array3::from_shape_vec((1, 5, 1), vec![0.0, 1.0, 0.2, 0.8, 0.1]).unwrap();
        let g = FrequencyGrid::Fixed(freqs.view());
        let cfg = |a| {
            SalienceConfig::default()
                .with_aggregate(a)
                .with_filter_peaks(false)
        };
        let h = [1.0, 2.0, 3.0];
        // bin 1 (100 Hz): harmonics at 100, 200, 300 Hz -> 1.0, 0.2, 0.8.
        let v = |a| salience(x.view(), g, &h, &cfg(a)).unwrap()[[0, 1, 0]];
        assert!((v(SalienceAggregate::Sum) - 2.0).abs() < 1e-15);
        assert_eq!(v(SalienceAggregate::Max), 1.0);
        assert_eq!(v(SalienceAggregate::Min), 0.2);
        assert_eq!(v(SalienceAggregate::Median), 0.8);
    }
}
