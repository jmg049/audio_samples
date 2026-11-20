//! Constant-Q and variable-Q transforms, their inverse, and the chroma /
//! tonnetz features built on them.
//!
//! ## What
//!
//! A constant-Q transform (CQT) analyses a signal with a bank of filters whose
//! centre frequencies are geometrically spaced (a fixed number of bins per octave)
//! and whose bandwidths are proportional to their centre frequencies, so every
//! filter has the same quality factor `Q`. Musical pitches map onto bins directly,
//! which makes the CQT the natural time-frequency representation for tonal
//! analysis. The variable-Q transform (VQT) widens the low-frequency filters by an
//! offset `gamma` (in Hz) for better time resolution there.
//!
//! This module provides the [`AudioConstantQ`] trait and the free functions it is
//! built from:
//!
//! | Operation | Items |
//! |---|---|
//! | Constant-Q and variable-Q transforms | [`AudioConstantQ::cqt`] / [`AudioConstantQ::vqt`] |
//! | Pseudo and hybrid CQT | [`AudioConstantQ::pseudo_cqt`] / [`AudioConstantQ::hybrid_cqt`] |
//! | Inverse CQT and Griffin-Lim reconstruction | [`AudioConstantQ::icqt`] / [`AudioConstantQ::griffinlim_cqt`] |
//! | Constant-Q chroma, CENS and variable-Q chroma | [`AudioConstantQ::chroma_cqt`] / [`AudioConstantQ::chroma_cens`] / [`AudioConstantQ::chroma_vqt`] |
//! | Tonal centroids | [`AudioConstantQ::tonnetz`] ([`tonnetz_from_chroma`]) |
//! | Constant-Q to chroma mapping | [`cq_to_chroma`] |
//! | Wavelet filters and their lengths | [`wavelet`] / [`wavelet_lengths`] |
//!
//! ## Why
//!
//! [`AudioTransforms::constant_q_transform`](crate::AudioTransforms::constant_q_transform)
//! delegates to the `spectrograms` crate, whose CQT uses a different kernel design,
//! normalisation and time alignment (its values differ from this module's by
//! several percent and it returns fewer frames). This module implements the
//! recursive, per-octave CQT with wavelet filters, the form chroma and tonnetz
//! features are usually computed from. Every config struct has a `librosa()`
//! preset. `Default` is the same except for [`ChromaCqtConfig`] (and the
//! [`ChromaCensConfig`] and [`TonnetzConfig`] built on it), whose default fixes
//! the tuning at A440 instead of estimating it from the signal.
//!
//! ## How
//!
//! Bring [`AudioConstantQ`] into scope and call it on any [`AudioSamples`]. Outputs
//! are `(channels, n_bins, n_frames)` arrays (complex for the transforms, real for
//! magnitudes and features); mono input gives one channel. Frame `t` is centred on
//! sample `t * hop_length`, as in every other frame-wise feature of the crate.
//!
//! The output precision follows the input: the element type is the sample
//! type's [`AudioSample::Float`](crate::AudioSample::Float). `u8`, `i16`, `I24`
//! and `f32` audio gives `f32` (or `Complex<f32>`); `i32` and `f64` audio gives
//! `f64`. The framing, FFTs and filter products run in that type. The filter
//! bases (wavelets, their spectra, the chroma mapping) are computed in `f64` and
//! rounded to the output type once. [`AudioConstantQ::icqt`] and
//! [`AudioConstantQ::griffinlim_cqt`] return audio of the receiver's sample type.
//!
//! The per-octave resampling uses [`crate::resample`]'s filters in the same
//! float type, reading the signal in place and writing its output directly.
//!
//! ```rust
//! use audio_samples::operations::constant_q::{AudioConstantQ, ChromaCqtConfig, CqtConfig};
//! use audio_samples::{AudioSamples, sample_rate, sine_wave};
//! use std::time::Duration;
//!
//! let audio = sine_wave::<f64>(440.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
//!
//! // Default CQT: 84 bins from C1, 22 frames of 512 samples.
//! let c = audio.cqt(&CqtConfig::default()).unwrap();
//! assert_eq!(c.dim(), (1, 84, 22));
//! // The strongest bin is A4 (MIDI 69, bin 69 - 24 = 45 above C1).
//! let mags: Vec<f64> = (0..84).map(|k| c[[0, k, 10]].norm()).collect();
//! let peak = mags.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
//! assert_eq!(peak, 45);
//!
//! // Chroma at A440 tuning: pitch class A (index 9).
//! let chroma = audio.chroma_cqt(&ChromaCqtConfig::default().with_tuning(Some(0.0))).unwrap();
//! assert_eq!(chroma.dim(), (1, 12, 22));
//! assert!((chroma[[0, 9, 10]] - 1.0).abs() < 1e-12);
//! ```
//!
//! ## Resampling between octaves
//!
//! The recursive (per-octave) algorithm halves the sample rate between octaves
//! with the crate's own windowed-sinc resampler ([`crate::resample`]) at the
//! quality selected by [`CqtConfig::resampling_quality`] (default `High`). The
//! top octave is computed at the input rate; the decimated octaves carry the
//! resampler's error, about 1e-4 of the peak magnitude. Frequencies other than
//! equal-tempered bins (just intonation, for example) are taken by
//! [`vqt_with_intervals`].

use std::num::NonZeroUsize;

use ndarray::{Array2, Array3, ArrayView2, ArrayView3, CowArray, Ix2, s};
use num_complex::Complex;

use crate::audio_float::{finite_float_rows, float_rows_to_audio};
use crate::operations::filterbanks::{Window, get_window, window_bandwidth};
use crate::operations::spectrum::{
    StftConfig, StftPadMode, forward_plan, istft, padded_frame_into,
};
use crate::operations::types::ResamplingQuality;
use crate::repr::SampleRate;
use crate::utils::audio_math::cqt_frequencies;
use crate::{
    AudioFloat, AudioSampleError, AudioSampleResult, AudioSamples, ParameterError, ProcessingError,
    StandardSample,
};

// =============================================================================
// Small helpers
// =============================================================================

fn param_err(parameter: &str, reason: impl ToString) -> AudioSampleError {
    AudioSampleError::Parameter(ParameterError::invalid_value(parameter, reason))
}

fn config_err(operation: &str, reason: impl ToString) -> AudioSampleError {
    AudioSampleError::Parameter(ParameterError::invalid_configuration(operation, reason))
}

/// The frequency of C1 in Hz, the default `fmin`.
fn c1_hz() -> f64 {
    440.0 * 2.0f64.powf((24.0 - 69.0) / 12.0)
}

/// MIDI note number of `f` Hz, evaluated as `12 (log2 f - log2 440) + 69`.
fn hz_to_midi_librosa(f: f64) -> f64 {
    12.0 * (f.log2() - 440.0f64.log2()) + 69.0
}

/// `n_bins` equal-tempered frequencies from `fmin`, `bpo` per octave.
fn equal_interval_freqs(n_bins: usize, fmin: f64, bpo: usize) -> Vec<f64> {
    (0..n_bins)
        .map(|k| {
            let octave = 2.0f64.powf((k / bpo) as f64);
            let ratio = 2.0f64.powf((k % bpo) as f64 / bpo as f64);
            octave * ratio * fmin
        })
        .collect()
}

/// Equal-temperament relative bandwidth for a single bin.
fn et_relative_bw(bpo: usize) -> f64 {
    let r = 2.0f64.powf(1.0 / bpo as f64);
    (r * r - 1.0) / (r * r + 1.0)
}

/// Per-bin relative bandwidth `alpha` from the local spacing of (at least two,
/// ascending) frequencies.
fn relative_bandwidth(freqs: &[f64]) -> Vec<f64> {
    let n = freqs.len();
    let logf: Vec<f64> = freqs.iter().map(|f| f.log2()).collect();
    let mut bpo = vec![0.0; n];
    bpo[0] = 1.0 / (logf[1] - logf[0]);
    bpo[n - 1] = 1.0 / (logf[n - 1] - logf[n - 2]);
    for k in 1..n - 1 {
        bpo[k] = 2.0 / (logf[k + 1] - logf[k - 1]);
    }
    bpo.iter()
        .map(|&b| {
            let p = 2.0f64.powf(2.0 / b);
            (p - 1.0) / (p + 1.0)
        })
        .collect()
}

/// `alpha` as the transforms use it (single bins use the equal-tempered value).
fn transform_alpha(freqs: &[f64], bpo: usize) -> Vec<f64> {
    if freqs.len() == 1 {
        vec![et_relative_bw(bpo)]
    } else {
        relative_bandwidth(freqs)
    }
}

/// Equivalent noise bandwidth used for the wavelet filters: a fixed table value for
/// named windows; parameterised windows use the formula at `n = 1000`.
fn librosa_window_bandwidth(window: &Window) -> AudioSampleResult<f64> {
    Ok(match window {
        Window::Rectangular => 1.0,
        Window::Hann => 1.500_183_105_468_75,
        Window::Hamming => 1.362_945_532_035_034_8,
        Window::Blackman => 1.726_968_155_426_232_6,
        Window::BlackmanHarris => 2.004_597_528_358_501_4,
        Window::Bartlett => 1.333_496_133_491_280_5,
        Window::Triangular => 1.333_170_652_355_585,
        Window::Kaiser { .. } | Window::Gaussian { .. } => {
            window_bandwidth(window, crate::nzu!(1000))?
        }
        Window::Custom(_) => {
            return Err(param_err(
                "window",
                "a custom (fixed-length) window cannot be used for wavelet filters of varying length; use a named window",
            ));
        }
    })
}

/// Per-bin bandwidth offset: `gamma`, or the ERB-based `24.7 * alpha / 0.108` when `None`.
fn gamma_values(alpha: &[f64], gamma: Option<f64>) -> Vec<f64> {
    match gamma {
        Some(g) => vec![g; alpha.len()],
        None => alpha.iter().map(|a| a * 24.7 / 0.108).collect(),
    }
}

/// The `norm`-length of a vector, by which it is normalised (no-op below the
/// smallest positive normal number).
fn norm_length(mags: impl Iterator<Item = f64> + Clone, norm: f64) -> f64 {
    if norm == f64::INFINITY {
        mags.fold(0.0, f64::max)
    } else if norm == f64::NEG_INFINITY {
        mags.fold(f64::INFINITY, f64::min)
    } else if norm == 0.0 {
        mags.filter(|&m| m > 0.0).count() as f64
    } else {
        mags.map(|m| m.powf(norm)).sum::<f64>().powf(1.0 / norm)
    }
}

fn validate_norm(parameter: &str, norm: Option<f64>) -> AudioSampleResult<()> {
    match norm {
        Some(p) if p.is_nan() || (p.is_finite() && p < 0.0) => Err(param_err(
            parameter,
            format!("norm={p} must be inf, -inf, 0, a positive number, or None"),
        )),
        _ => Ok(()),
    }
}

fn require_finite_positive(parameter: &str, v: f64) -> AudioSampleResult<()> {
    if v.is_finite() && v > 0.0 {
        Ok(())
    } else {
        Err(param_err(
            parameter,
            format!("{parameter}={v} must be finite and positive"),
        ))
    }
}

// =============================================================================
// Wavelet filters
// =============================================================================

/// Shape of the wavelet filters of a constant-Q / variable-Q filterbank.
///
/// ## Purpose
///
/// Collects the parameters that determine each filter's length and shape, given its
/// centre frequency: the analysis window, the filter-length scale, the VQT bandwidth
/// offset `gamma`, the per-filter normalisation and whether the filters are
/// zero-padded to a power of two.
///
/// ## Intended usage
///
/// Pass it to [`wavelet_lengths`] or [`wavelet`] to inspect or reuse the filters
/// that [`AudioConstantQ::cqt`] / [`AudioConstantQ::vqt`] use internally.
/// `WaveletConfig::default()` is a Hann window, filter scale 1, constant Q,
/// L1 normalisation and power-of-two padding.
///
/// ## Invariants
///
/// [`WaveletConfig::validate`] requires a finite positive `filter_scale`, a finite
/// non-negative `gamma`, a valid `norm` and a named (non-custom) window.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::constant_q::WaveletConfig;
///
/// let cfg = WaveletConfig::default().with_gamma(None).with_filter_scale(0.5);
/// assert!(cfg.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct WaveletConfig {
    /// Window applied to each complex exponential. Default: Hann. Must be a
    /// named window ([`Window::Custom`] is rejected: the filters have different
    /// lengths).
    pub window: Window,
    /// Scales every filter length. Values below 1 give shorter filters (better time
    /// resolution, wider bandwidth). Finite and positive; default 1.
    pub filter_scale: f64,
    /// Bandwidth offset in Hz added to every filter (variable-Q). `Some(0.0)` gives
    /// constant Q (the default); `None` uses the ERB-derived offset
    /// `24.7 * alpha / 0.108`. Finite and non-negative.
    pub gamma: Option<f64>,
    /// Normalisation of each filter: `Some(p)` divides by the `p`-norm (`p > 0`,
    /// `inf` = maximum, `-inf` = minimum, `0` = number of non-zeros); `None` leaves
    /// the filters unnormalised. Default `Some(1.0)`.
    pub norm: Option<f64>,
    /// If `true` (the default) all filters are centred in a buffer whose
    /// length is the next power of two of the longest filter; otherwise in a buffer
    /// of `ceil(max length)` samples.
    pub pad_fft: bool,
}

impl Default for WaveletConfig {
    fn default() -> Self {
        Self {
            window: Window::Hann,
            filter_scale: 1.0,
            gamma: Some(0.0),
            norm: Some(1.0),
            pad_fft: true,
        }
    }
}

impl WaveletConfig {
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Settings matching librosa 1.0's defaults; currently the same as [`Default`].
    #[must_use]
    pub fn librosa() -> Self {
        Self::default()
    }

    /// Sets the window.
    #[must_use]
    pub fn with_window(mut self, window: Window) -> Self {
        self.window = window;
        self
    }

    /// Sets the filter-length scale.
    #[must_use]
    pub fn with_filter_scale(mut self, filter_scale: f64) -> Self {
        self.filter_scale = filter_scale;
        self
    }

    /// Sets the bandwidth offset (`None` = ERB-derived).
    #[must_use]
    pub fn with_gamma(mut self, gamma: Option<f64>) -> Self {
        self.gamma = gamma;
        self
    }

    /// Sets the filter normalisation.
    #[must_use]
    pub fn with_norm(mut self, norm: Option<f64>) -> Self {
        self.norm = norm;
        self
    }

    /// Sets whether filters are padded to a power of two.
    #[must_use]
    pub fn with_pad_fft(mut self, pad_fft: bool) -> Self {
        self.pad_fft = pad_fft;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `filter_scale` is not finite and positive, `gamma` is
    /// negative or not finite, `norm` is NaN or a negative finite number, or the
    /// window is [`Window::Custom`].
    pub fn validate(&self) -> AudioSampleResult<()> {
        require_finite_positive("filter_scale", self.filter_scale)?;
        if let Some(g) = self.gamma
            && !(g.is_finite() && g >= 0.0)
        {
            return Err(param_err(
                "gamma",
                format!("gamma={g} must be finite and non-negative"),
            ));
        }
        validate_norm("norm", self.norm)?;
        librosa_window_bandwidth(&self.window).map(|_| ())
    }
}

/// Checks the centre frequencies. Bandwidths derived from the spacing need two
/// or more; given bandwidths (`alpha`) need one.
fn validate_freqs(freqs: &[f64], alpha_given: bool) -> AudioSampleResult<()> {
    if freqs.is_empty() {
        return Err(param_err("freqs", "at least one frequency is required"));
    }
    if freqs.len() < 2 && !alpha_given {
        return Err(param_err(
            "freqs",
            "2 or more frequencies are required to compute bandwidths; give alpha for a single frequency",
        ));
    }
    if freqs.iter().any(|&f| !(f.is_finite() && f > 0.0)) {
        return Err(param_err(
            "freqs",
            "frequencies must be finite and strictly positive",
        ));
    }
    if freqs.windows(2).any(|w| w[0] > w[1]) {
        return Err(param_err("freqs", "frequencies must be in ascending order"));
    }
    Ok(())
}

/// Filter lengths and the highest filter cut-off, given `alpha` (internal form).
fn wavelet_lengths_impl(
    freqs: &[f64],
    sr: f64,
    window_bw: f64,
    filter_scale: f64,
    gamma: Option<f64>,
    alpha: &[f64],
) -> (Vec<f64>, f64) {
    let g = gamma_values(alpha, gamma);
    let mut cutoff = f64::NEG_INFINITY;
    let mut lengths = Vec::with_capacity(freqs.len());
    for ((&f, &a), &gk) in freqs.iter().zip(alpha).zip(&g) {
        let q = filter_scale / a;
        cutoff = cutoff.max(f * (1.0 + 0.5 * window_bw / q) + 0.5 * gk);
        lengths.push(q * sr / (f + gk / a));
    }
    (lengths, cutoff)
}

/// Computes the length of each wavelet filter of a constant-Q / variable-Q
/// filterbank.
///
/// A filter centred at `f` with relative bandwidth `alpha` (from the spacing of the
/// frequencies) has quality factor `Q = filter_scale / alpha` and length
/// `Q * sample_rate / (f + gamma / alpha)` samples. The cut-off is the largest
/// `f * (1 + 0.5 * B / Q) + 0.5 * gamma` over all filters, where `B` is the
/// window's equivalent noise bandwidth in bins.
///
/// # Arguments
///
/// * `freqs` - Centre frequencies in Hz: at least two, positive, ascending.
/// * `sample_rate` - Sample rate in Hz (finite, positive).
/// * `config` - Window, filter scale and `gamma` (`norm` and `pad_fft` are not used).
///
/// # Returns
///
/// `(lengths, cutoff)`: one fractional length in samples per frequency, and the
/// highest cut-off frequency in Hz (compare it with the Nyquist frequency).
///
/// # Errors
///
/// [`ParameterError`] if fewer than two frequencies are given, any is not positive,
/// they are not ascending, the sample rate is not positive, or the configuration is
/// invalid.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::constant_q::{WaveletConfig, wavelet_lengths};
///
/// let freqs: Vec<f64> = (0..12).map(|k| 440.0 * 2f64.powf(k as f64 / 12.0)).collect();
/// let (lengths, cutoff) = wavelet_lengths(&freqs, 22050.0, &WaveletConfig::default()).unwrap();
/// // Q = 1 / alpha with alpha = (2^(2/12) - 1) / (2^(2/12) + 1), i.e. Q = 17.33.
/// let r = 2f64.powf(2.0 / 12.0);
/// let q = (r + 1.0) / (r - 1.0);
/// assert!((lengths[0] - q * 22050.0 / 440.0).abs() < 1e-9);
/// assert!(cutoff > freqs[11] && cutoff < 22050.0 / 2.0);
/// ```
///
/// # References
///
/// C. Schörkhuber and A. Klapuri, "Constant-Q transform toolbox for music
/// processing," SMC 2010.
pub fn wavelet_lengths(
    freqs: &[f64],
    sample_rate: f64,
    config: &WaveletConfig,
) -> AudioSampleResult<(Vec<f64>, f64)> {
    let mut lengths = vec![0.0; freqs.len()];
    let cutoff = wavelet_lengths_into(freqs, sample_rate, config, &mut lengths)?;
    Ok((lengths, cutoff))
}

/// [`wavelet_lengths`] writing the lengths into `lengths` (one per
/// frequency) and returning the cut-off, without allocating.
///
/// # Errors
///
/// As [`wavelet_lengths`], and [`ParameterError`] if `lengths` does not have
/// one element per frequency.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::constant_q::{WaveletConfig, wavelet_lengths, wavelet_lengths_into};
///
/// let freqs: Vec<f64> = (0..12).map(|k| 440.0 * 2f64.powf(k as f64 / 12.0)).collect();
/// let cfg = WaveletConfig::default();
/// let mut lengths = [0.0; 12];
/// let cutoff = wavelet_lengths_into(&freqs, 22050.0, &cfg, &mut lengths).unwrap();
/// assert_eq!((lengths.to_vec(), cutoff), wavelet_lengths(&freqs, 22050.0, &cfg).unwrap());
/// ```
pub fn wavelet_lengths_into(
    freqs: &[f64],
    sample_rate: f64,
    config: &WaveletConfig,
    lengths: &mut [f64],
) -> AudioSampleResult<f64> {
    config.validate()?;
    validate_freqs(freqs, false)?;
    require_finite_positive("sample_rate", sample_rate)?;
    let window_bw = librosa_window_bandwidth(&config.window)?;
    if lengths.len() != freqs.len() {
        return Err(param_err(
            "lengths",
            format!(
                "needs one element per frequency ({}), got {}",
                freqs.len(),
                lengths.len()
            ),
        ));
    }
    // The same arithmetic as `wavelet_lengths_impl` with `relative_bandwidth`
    // and `gamma_values`, one bin at a time.
    let n = freqs.len();
    let log2f = |k: usize| freqs[k].log2();
    let mut cutoff = f64::NEG_INFINITY;
    for (k, (&f, len)) in freqs.iter().zip(lengths.iter_mut()).enumerate() {
        let bpo = if k == 0 {
            1.0 / (log2f(1) - log2f(0))
        } else if k == n - 1 {
            1.0 / (log2f(n - 1) - log2f(n - 2))
        } else {
            2.0 / (log2f(k + 1) - log2f(k - 1))
        };
        let p = 2.0f64.powf(2.0 / bpo);
        let a = (p - 1.0) / (p + 1.0);
        let g = config.gamma.unwrap_or(a * 24.7 / 0.108);
        let q = config.filter_scale / a;
        cutoff = cutoff.max(f * (1.0 + 0.5 * window_bw / q) + 0.5 * g);
        *len = q * sample_rate / (f + g / a);
    }
    Ok(cutoff)
}

/// Builds the time-domain filters (internal form with explicit `alpha`): each row is
/// a windowed, normalised complex exponential centred in a buffer of `width` samples.
fn wavelet_impl(
    freqs: &[f64],
    sr: f64,
    window: &Window,
    lengths: &[f64],
    norm: Option<f64>,
    pad_fft: bool,
) -> AudioSampleResult<Array2<Complex<f64>>> {
    let max_len = lengths.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let width = if pad_fft {
        2.0f64.powf(max_len.log2().ceil())
    } else {
        max_len.ceil()
    };
    crate::utils::check_derived_len(
        "sample_rate",
        "the wavelet basis",
        width * freqs.len() as f64,
        crate::utils::MAX_SIGNAL_ELEMENTS,
    )?;
    let width = width as usize;
    let mut out = Array2::<Complex<f64>>::zeros((freqs.len(), width.max(1)));
    for ((&ilen, &freq), mut row) in lengths.iter().zip(freqs).zip(out.outer_iter_mut()) {
        // Sample offsets floor(-L/2) .. floor(L/2) (exclusive).
        let start = (-ilen / 2.0).floor();
        let stop = (ilen / 2.0).floor();
        let n = (stop - start).max(0.0) as usize;
        let win = get_window(window, n, true)?;
        let mut sig: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let t = start + i as f64;
                let angle = t * 2.0 * std::f64::consts::PI * freq / sr;
                Complex::new(angle.cos(), angle.sin()) * win[i]
            })
            .collect();
        if let Some(p) = norm {
            let len = norm_length(sig.iter().map(|z| z.norm()), p);
            if len >= f64::MIN_POSITIVE {
                for z in &mut sig {
                    *z /= len;
                }
            }
        }
        let lpad = (row.len() - n) / 2;
        for (dst, z) in row.iter_mut().skip(lpad).zip(sig) {
            *dst = z;
        }
    }
    Ok(out)
}

/// Builds the wavelet filters of a constant-Q / variable-Q filterbank.
///
/// Filter `k` is a complex exponential at `freqs[k]` Hz, of the length given by
/// [`wavelet_lengths`], multiplied by the (periodic) window, normalised by
/// `config.norm`, and centred in a common zero-padded buffer.
///
/// # Arguments
///
/// * `freqs` - Centre frequencies in Hz: at least two, positive, ascending.
/// * `sample_rate` - Sample rate in Hz (finite, positive).
/// * `config` - Filter parameters.
///
/// # Returns
///
/// `(filters, lengths)`: a `(freqs.len(), width)` complex array, where `width` is
/// the next power of two of the longest filter (`pad_fft`) or its ceiling, and the
/// fractional filter lengths.
///
/// # Errors
///
/// As for [`wavelet_lengths`]; also a [`ParameterError`] if the filter basis would
/// exceed 2^31 elements (an extreme `sample_rate` makes the filters that long).
///
/// The filters are `Complex<f64>`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::constant_q::{WaveletConfig, wavelet};
///
/// let freqs = [440.0, 466.16, 493.88];
/// let (filters, lengths) = wavelet(&freqs, 22050.0, &WaveletConfig::default()).unwrap();
/// assert_eq!(filters.nrows(), 3);
/// assert!(filters.ncols().is_power_of_two() && filters.ncols() as f64 >= lengths[0]);
/// // L1-normalised.
/// let l1: f64 = filters.row(1).iter().map(|z| z.norm()).sum();
/// assert!((l1 - 1.0).abs() < 1e-12);
/// ```
pub fn wavelet(
    freqs: &[f64],
    sample_rate: f64,
    config: &WaveletConfig,
) -> AudioSampleResult<(Array2<Complex<f64>>, Vec<f64>)> {
    let (lengths, _) = wavelet_lengths(freqs, sample_rate, config)?;
    let filters = wavelet_impl(
        freqs,
        sample_rate,
        &config.window,
        &lengths,
        config.norm,
        config.pad_fft,
    )?;
    Ok((filters, lengths))
}

/// Smooths each row of a [`cq_to_chroma`] map by centred convolution with `window` along the
/// constant-Q bins, keeping their number.
///
/// # Arguments
/// - `map`: the map; rows are chroma bins and columns constant-Q bins.
/// - `window`: the weights.
///
/// # Examples
/// ```rust
/// use audio_samples::operations::constant_q::smooth_chroma_map;
/// use ndarray::array;
///
/// let mut map = array![[0.0, 1.0, 0.0]];
/// smooth_chroma_map(&mut map, &[0.25, 0.5, 0.25]);
/// assert_eq!(map, array![[0.25, 0.5, 0.25]]);
/// ```
pub fn smooth_chroma_map(map: &mut Array2<f64>, window: &[f64]) {
    if window.is_empty() {
        return;
    }
    let start = (window.len() - 1) / 2;
    let mut source = vec![0.0; map.ncols()];
    for mut row in map.rows_mut() {
        for (o, &v) in source.iter_mut().zip(row.iter()) {
            *o = v;
        }
        for (j, slot) in row.iter_mut().enumerate() {
            let position = j + start;
            *slot = window
                .iter()
                .take(position + 1)
                .enumerate()
                .filter_map(|(i, &weight)| source.get(position - i).map(|&v| weight * v))
                .sum();
        }
    }
}

/// [`wavelet_lengths`] with the relative bandwidth of each filter given.
///
/// The bandwidths do not depend on the spacing of the frequencies, so one frequency is
/// enough.
///
/// # Arguments
/// - `freqs`: centre frequencies, one or more, ascending and positive.
/// - `sample_rate`: in Hz.
/// - `config`: the filter settings.
/// - `alpha`: the relative bandwidth per frequency; finite and positive.
///
/// # Returns
/// The lengths and the highest cut-off, as for [`wavelet_lengths`].
///
/// # Errors
/// As [`wavelet_lengths`] (except that one frequency is accepted), and [`ParameterError`]
/// for an invalid `alpha` or an `alpha` without one value per frequency.
///
/// # Examples
/// ```rust
/// use audio_samples::operations::constant_q::{WaveletConfig, wavelet_lengths_with_alpha};
///
/// let cfg = WaveletConfig::default();
/// let (lengths, _) = wavelet_lengths_with_alpha(&[440.0, 880.0], 22050.0, &cfg, &[0.5, 0.5]).unwrap();
/// assert!((lengths[0] - 2.0 * lengths[1]).abs() < 1e-9);
/// // A single frequency: Q = 1 / 0.5, length Q * 22050 / 440.
/// let (single, _) = wavelet_lengths_with_alpha(&[440.0], 22050.0, &cfg, &[0.5]).unwrap();
/// assert!((single[0] - 2.0 * 22050.0 / 440.0).abs() < 1e-9);
/// ```
pub fn wavelet_lengths_with_alpha(
    freqs: &[f64],
    sample_rate: f64,
    config: &WaveletConfig,
    alpha: &[f64],
) -> AudioSampleResult<(Vec<f64>, f64)> {
    let mut lengths = vec![0.0; freqs.len()];
    let cutoff = wavelet_lengths_with_alpha_into(freqs, sample_rate, config, alpha, &mut lengths)?;
    Ok((lengths, cutoff))
}

/// [`wavelet_lengths_with_alpha`] writing the lengths into `lengths` (one per frequency) and
/// returning the cut-off, without allocating.
///
/// # Errors
/// As [`wavelet_lengths_with_alpha`], and [`ParameterError`] if `lengths` does not have one
/// element per frequency.
///
/// # Examples
/// ```rust
/// use audio_samples::operations::constant_q::{WaveletConfig, wavelet_lengths_with_alpha_into};
///
/// let cfg = WaveletConfig::default();
/// let mut lengths = [0.0; 2];
/// wavelet_lengths_with_alpha_into(&[440.0, 880.0], 22050.0, &cfg, &[0.5, 0.5], &mut lengths).unwrap();
/// assert!((lengths[0] - 2.0 * lengths[1]).abs() < 1e-9);
/// ```
pub fn wavelet_lengths_with_alpha_into(
    freqs: &[f64],
    sample_rate: f64,
    config: &WaveletConfig,
    alpha: &[f64],
    lengths: &mut [f64],
) -> AudioSampleResult<f64> {
    config.validate()?;
    validate_freqs(freqs, true)?;
    require_finite_positive("sample_rate", sample_rate)?;
    if alpha.len() != freqs.len() {
        return Err(param_err(
            "alpha",
            format!(
                "needs one value per frequency ({}), got {}",
                freqs.len(),
                alpha.len()
            ),
        ));
    }
    if lengths.len() != freqs.len() {
        return Err(param_err(
            "lengths",
            format!(
                "needs one element per frequency ({}), got {}",
                freqs.len(),
                lengths.len()
            ),
        ));
    }
    for &a in alpha {
        require_finite_positive("alpha", a)?;
    }
    let window_bw = librosa_window_bandwidth(&config.window)?;
    let mut cutoff = f64::NEG_INFINITY;
    for ((len, &f), &a) in lengths.iter_mut().zip(freqs).zip(alpha) {
        let g = config.gamma.unwrap_or(a * 24.7 / 0.108);
        let q = config.filter_scale / a;
        cutoff = cutoff.max(f * (1.0 + 0.5 * window_bw / q) + 0.5 * g);
        *len = q * sample_rate / (f + g / a);
    }
    Ok(cutoff)
}

/// [`wavelet`] with the relative bandwidth of each filter given.
///
/// # Arguments
/// As [`wavelet_lengths_with_alpha`].
///
/// # Returns
/// The filters and their lengths, as for [`wavelet`].
///
/// # Errors
/// As [`wavelet_lengths_with_alpha`].
///
/// # Examples
/// ```rust
/// use audio_samples::operations::constant_q::{WaveletConfig, wavelet_with_alpha};
///
/// let cfg = WaveletConfig::default();
/// let (filters, lengths) = wavelet_with_alpha(&[440.0, 880.0], 22050.0, &cfg, &[0.5, 0.5]).unwrap();
/// assert_eq!(filters.nrows(), lengths.len());
/// ```
pub fn wavelet_with_alpha(
    freqs: &[f64],
    sample_rate: f64,
    config: &WaveletConfig,
    alpha: &[f64],
) -> AudioSampleResult<(Array2<Complex<f64>>, Vec<f64>)> {
    let (lengths, _) = wavelet_lengths_with_alpha(freqs, sample_rate, config, alpha)?;
    let filters = wavelet_impl(
        freqs,
        sample_rate,
        &config.window,
        &lengths,
        config.norm,
        config.pad_fft,
    )?;
    Ok((filters, lengths))
}

// =============================================================================
// Sparse FFT basis
// =============================================================================

/// One sparse row of a frequency-domain basis with `Complex<F>` values.
#[derive(Debug, Clone)]
struct SparseRow<F> {
    idx: Vec<usize>,
    val: Vec<Complex<F>>,
}

impl<F: AudioFloat> SparseRow<F> {
    #[inline]
    fn dot(&self, x: &[Complex<F>]) -> Complex<F> {
        let mut acc = Complex::new(F::zero(), F::zero());
        for (&j, &v) in self.idx.iter().zip(&self.val) {
            acc += v * x[j];
        }
        acc
    }

    #[inline]
    fn dot_abs(&self, x: &[F]) -> F {
        let mut acc = F::zero();
        for (&j, &v) in self.idx.iter().zip(&self.val) {
            acc += v.re * x[j];
        }
        acc
    }
}

/// A sparse frequency-domain filter basis (rows = filters, columns = rFFT bins).
/// The basis is built in `f64` and rounded to the analysis type with
/// [`FftBasis::cast`].
#[derive(Debug, Clone)]
struct FftBasis<F> {
    n_fft: usize,
    rows: Vec<SparseRow<F>>,
}

impl FftBasis<f64> {
    /// The basis with every value rounded to `F` (the identity for `f64`).
    fn cast<F: AudioFloat>(&self) -> FftBasis<F> {
        FftBasis {
            n_fft: self.n_fft,
            rows: self
                .rows
                .iter()
                .map(|r| SparseRow {
                    idx: r.idx.clone(),
                    val: r
                        .val
                        .iter()
                        .map(|z| Complex::new(F::of(z.re), F::of(z.im)))
                        .collect(),
                })
                .collect(),
        }
    }
}

/// Sparsifies one row: keeps the largest magnitudes holding
/// `1 - quantile` of the row's L1 mass.
fn sparsify_row(x: &[Complex<f64>], quantile: f64) -> SparseRow<f64> {
    let mags: Vec<f64> = x.iter().map(|z| z.norm()).collect();
    let norm: f64 = mags.iter().sum();
    // Magnitudes are non-negative and not NaN, so their bit patterns sort in
    // the same order as the values (and as `f64::total_cmp`); an unstable
    // integer sort gives the same sorted sequence as a stable float sort.
    let mut sorted_bits: Vec<u64> = mags.iter().map(|m| m.to_bits()).collect();
    sorted_bits.sort_unstable();
    let sorted: Vec<f64> = sorted_bits.into_iter().map(f64::from_bits).collect();
    let mut cum = 0.0;
    let mut thresh_idx = 0;
    for (i, &m) in sorted.iter().enumerate() {
        cum += m / norm;
        if cum >= quantile {
            thresh_idx = i;
            break;
        }
    }
    let thresh = sorted[thresh_idx];
    let mut row = SparseRow {
        idx: Vec::new(),
        val: Vec::new(),
    };
    for (j, (&m, &z)) in mags.iter().zip(x).enumerate() {
        if m >= thresh && z != Complex::new(0.0, 0.0) {
            row.idx.push(j);
            row.val.push(z);
        }
    }
    row
}

/// Frequency-domain filter basis: wavelets scaled by `length / n_fft`, FFT'd,
/// truncated to the non-negative frequencies and sparsified. `min_hop` enlarges
/// `n_fft` to at least `2^(1 + ceil(log2(hop)))` (pseudo-CQT). The basis is a
/// filter table and is computed in `f64` for every analysis type.
#[allow(clippy::too_many_arguments)]
fn vqt_filter_fft(
    sr: f64,
    freqs: &[f64],
    alpha: &[f64],
    lengths: &[f64],
    window: &Window,
    norm: Option<f64>,
    sparsity: f64,
    min_hop: Option<usize>,
    gain: f64,
) -> AudioSampleResult<FftBasis<f64>> {
    let basis = wavelet_impl(freqs, sr, window, lengths, norm, true)?;
    debug_assert_eq!(alpha.len(), freqs.len());
    let mut n_fft = basis.ncols();
    if let Some(hop) = min_hop {
        let min_n = 2.0f64.powf(1.0 + (hop as f64).log2().ceil());
        if (n_fft as f64) < min_n {
            n_fft = min_n as usize;
        }
    }
    let fft = <f64 as AudioFloat>::with_fft_planner(|p| p.plan_fft_forward(n_fft));
    let mut buf = vec![Complex::new(0.0, 0.0); n_fft];
    let mut rows = Vec::with_capacity(freqs.len());
    for (row, &len) in basis.outer_iter().zip(lengths) {
        let scale = len / n_fft as f64;
        buf.fill(Complex::new(0.0, 0.0));
        for (dst, &z) in buf.iter_mut().zip(row.iter()) {
            *dst = z * scale;
        }
        fft.process(&mut buf);
        let mut sp = sparsify_row(&buf[..n_fft / 2 + 1], sparsity);
        if gain != 1.0 {
            for v in &mut sp.val {
                *v *= gain;
            }
        }
        rows.push(sp);
    }
    Ok(FftBasis { n_fft, rows })
}

/// Calls `f(channel, frame, spectrum)` for every centred frame of every channel,
/// with `window` (`None` = rectangular) and `n_fft / 2` samples of `pad_mode`
/// padding at both ends.
fn for_each_frame<F: AudioFloat>(
    y: ArrayView2<'_, F>,
    n_fft: usize,
    hop: usize,
    pad_mode: StftPadMode,
    window: Option<&[F]>,
    max_frames: usize,
    mut f: impl FnMut(usize, usize, &[Complex<F>]),
) -> AudioSampleResult<()> {
    let plan = forward_plan::<F>(n_fft);
    let mut scratch = plan.make_scratch_vec();
    let mut frame_in = vec![F::zero(); n_fft];
    let mut spec = plan.make_output_vec();
    let pad = n_fft / 2;
    let mut row_buf = Vec::new();
    for (c, row) in y.outer_iter().enumerate() {
        let x: &[F] = if let Some(s) = row.as_slice() {
            s
        } else {
            row_buf.clear();
            row_buf.extend(row.iter().copied());
            &row_buf
        };
        let padded_len = x.len() + 2 * pad;
        if padded_len < n_fft {
            return Err(param_err(
                "signal",
                "the signal is too short for the constant-Q analysis",
            ));
        }
        let n_frames = (1 + (padded_len - n_fft) / hop).min(max_frames);
        for t in 0..n_frames {
            let start = (t * hop) as isize - pad as isize;
            padded_frame_into(x, start, window, pad_mode, &mut frame_in);
            plan.process_with_scratch(&mut frame_in, &mut spec, &mut scratch)
                .map_err(|e| {
                    AudioSampleError::Processing(ProcessingError::algorithm_failure(
                        "constant-Q FFT",
                        e.to_string(),
                    ))
                })?;
            f(c, t, &spec);
        }
    }
    Ok(())
}

/// Number of centred frames of a signal of `len` samples.
fn n_centred_frames(len: usize, n_fft: usize, hop: usize) -> usize {
    1 + (len + 2 * (n_fft / 2)).saturating_sub(n_fft) / hop
}

// =============================================================================
// Resampling helpers
// =============================================================================

/// Resamples every row from `in_sr` to `out_sr` (integers; only the ratio matters),
/// fixing the length to `ceil(n * out / in)` and multiplying by `gain`.
///
/// The rows are resampled with [`crate::resample`]'s filters in the element
/// type `F`, read from the view and written straight into the result (no
/// copy of the input, no `f64` round trip).
fn resample_rows<F: AudioFloat>(
    y: ArrayView2<'_, F>,
    in_sr: u32,
    out_sr: u32,
    quality: ResamplingQuality,
    gain: f64,
) -> AudioSampleResult<Array2<F>> {
    let gain = F::of(gain);
    if in_sr == out_sr {
        return Ok(y.mapv(|v| v * gain));
    }
    let mut rows = crate::resampling::resample_float_rows(y, in_sr, out_sr, quality, None)?;
    rows.mapv_inplace(|v| v * gain);
    Ok(rows)
}

// =============================================================================
// Configuration
// =============================================================================

/// Parameters of a constant-Q transform.
///
/// ## Purpose
///
/// Describes the filterbank (lowest frequency, number of bins, bins per octave,
/// tuning, filter shape and normalisation) and the framing (hop length, padding) of
/// [`AudioConstantQ::cqt`], and of the related transforms
/// ([`pseudo_cqt`](AudioConstantQ::pseudo_cqt), [`hybrid_cqt`](AudioConstantQ::hybrid_cqt),
/// [`icqt`](AudioConstantQ::icqt)); [`VqtConfig`] adds the variable-Q offset.
///
/// ## Intended usage
///
/// `CqtConfig::default()` gives 84 bins (7 octaves) from C1 (32.70 Hz), 12 bins
/// per octave, hop 512. Adjust fields with the `with_*` builders.
///
/// ## Invariants
///
/// [`CqtConfig::validate`] checks every field; the transforms call it and also check
/// that the highest filter fits below the Nyquist frequency.
///
/// ## Assumptions
///
/// The recursive algorithm halves the sample rate (and hop) once per octave while
/// the hop is even, so a hop divisible by `2^(n_octaves - 1)` (e.g. 512 for 7
/// octaves) is the efficient choice.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::constant_q::CqtConfig;
///
/// let cfg = CqtConfig::default()
///     .with_fmin(Some(65.0))
///     .with_n_bins(Some(nzu!(48)))
///     .with_bins_per_octave(nzu!(24));
/// assert!(cfg.validate().is_ok());
/// assert_eq!(cfg.hop_length.get(), 512);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CqtConfig {
    /// Samples between successive frames; frame `t` is centred on sample
    /// `t * hop_length`. Default 512.
    pub hop_length: NonZeroUsize,
    /// Centre frequency of the lowest bin in Hz (before tuning). `None` (the
    /// default) means C1 (32.70 Hz). Must be finite, positive and below the Nyquist
    /// frequency.
    pub fmin: Option<f64>,
    /// Number of bins. `None` uses as many bins as fit below the Nyquist frequency.
    /// Default `Some(84)` (7 octaves of 12 bins).
    pub n_bins: Option<NonZeroUsize>,
    /// Bins per octave: the frequency resolution. 12 gives semitones, 36 thirds of a
    /// semitone. Default 12.
    pub bins_per_octave: NonZeroUsize,
    /// Tuning offset in fractions of a bin: every centre frequency is multiplied by
    /// `2^(tuning / bins_per_octave)`. `None` estimates it from the signal. Default
    /// `Some(0.0)`.
    pub tuning: Option<f64>,
    /// Scales every filter length; below 1 shortens the filters (better time
    /// resolution, wider bins). Finite and positive. Default 1.
    pub filter_scale: f64,
    /// Normalisation of each filter before the transform (see [`WaveletConfig::norm`]).
    /// Default `Some(1.0)` (L1).
    pub norm: Option<f64>,
    /// Fraction of each frequency-domain filter's L1 mass that may be discarded to
    /// make the basis sparse (faster). In `[0, 1)`; 0 keeps every coefficient.
    /// Default 0.01.
    pub sparsity: f64,
    /// Window of the wavelet filters. Default Hann. Must be a named window.
    pub window: Window,
    /// If `true` (the default), each bin is divided by the square root of its
    /// filter length (an orthonormal-like scaling: a full-scale sinusoid gives a
    /// magnitude near `sqrt(L) / 2`). If `false`, the raw filter response.
    pub scale: bool,
    /// How the signal is extended at both ends for centred frames. Default
    /// [`StftPadMode::Constant`] (zeros).
    pub pad_mode: StftPadMode,
    /// Resampler quality for the per-octave decimation. Default
    /// [`ResamplingQuality::High`].
    pub resampling_quality: ResamplingQuality,
}

impl Default for CqtConfig {
    fn default() -> Self {
        Self {
            hop_length: crate::nzu!(512),
            fmin: None,
            n_bins: Some(crate::nzu!(84)),
            bins_per_octave: crate::nzu!(12),
            tuning: Some(0.0),
            filter_scale: 1.0,
            norm: Some(1.0),
            sparsity: 0.01,
            window: Window::Hann,
            scale: true,
            pad_mode: StftPadMode::Constant,
            resampling_quality: ResamplingQuality::High,
        }
    }
}

impl CqtConfig {
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Settings matching librosa 1.0's defaults; currently the same as [`Default`].
    #[must_use]
    pub fn librosa() -> Self {
        Self::default()
    }

    /// Sets the hop length.
    #[must_use]
    pub fn with_hop_length(mut self, hop_length: NonZeroUsize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Sets the lowest frequency (`None` = C1).
    #[must_use]
    pub fn with_fmin(mut self, fmin: Option<f64>) -> Self {
        self.fmin = fmin;
        self
    }

    /// Sets the number of bins (`None` = as many as fit below Nyquist).
    #[must_use]
    pub fn with_n_bins(mut self, n_bins: Option<NonZeroUsize>) -> Self {
        self.n_bins = n_bins;
        self
    }

    /// Sets the number of bins per octave.
    #[must_use]
    pub fn with_bins_per_octave(mut self, bins_per_octave: NonZeroUsize) -> Self {
        self.bins_per_octave = bins_per_octave;
        self
    }

    /// Sets the tuning offset (`None` = estimate from the signal).
    #[must_use]
    pub fn with_tuning(mut self, tuning: Option<f64>) -> Self {
        self.tuning = tuning;
        self
    }

    /// Sets the filter-length scale.
    #[must_use]
    pub fn with_filter_scale(mut self, filter_scale: f64) -> Self {
        self.filter_scale = filter_scale;
        self
    }

    /// Sets the filter normalisation.
    #[must_use]
    pub fn with_norm(mut self, norm: Option<f64>) -> Self {
        self.norm = norm;
        self
    }

    /// Sets the basis sparsity.
    #[must_use]
    pub fn with_sparsity(mut self, sparsity: f64) -> Self {
        self.sparsity = sparsity;
        self
    }

    /// Sets the filter window.
    #[must_use]
    pub fn with_window(mut self, window: Window) -> Self {
        self.window = window;
        self
    }

    /// Sets whether the output is scaled by `1 / sqrt(filter length)`.
    #[must_use]
    pub fn with_scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Sets the padding mode.
    #[must_use]
    pub fn with_pad_mode(mut self, pad_mode: StftPadMode) -> Self {
        self.pad_mode = pad_mode;
        self
    }

    /// Sets the resampler quality.
    #[must_use]
    pub fn with_resampling_quality(mut self, quality: ResamplingQuality) -> Self {
        self.resampling_quality = quality;
        self
    }

    /// Checks the configuration (the Nyquist check needs the sample rate and is done
    /// by the transforms).
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `fmin` is not finite and positive, `tuning` is not
    /// finite, `filter_scale` is not finite and positive, `sparsity` is outside
    /// `[0, 1)`, `norm` is invalid, or the window is [`Window::Custom`].
    pub fn validate(&self) -> AudioSampleResult<()> {
        if let Some(f) = self.fmin {
            require_finite_positive("fmin", f)?;
        }
        if let Some(t) = self.tuning
            && !t.is_finite()
        {
            return Err(param_err("tuning", format!("tuning={t} must be finite")));
        }
        require_finite_positive("filter_scale", self.filter_scale)?;
        if !(0.0..1.0).contains(&self.sparsity) {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "sparsity",
                self.sparsity,
                0.0,
                1.0,
                "sparsity must be in [0, 1)",
            )));
        }
        validate_norm("norm", self.norm)?;
        librosa_window_bandwidth(&self.window).map(|_| ())
    }

    fn fmin_or_c1(&self) -> f64 {
        self.fmin.unwrap_or_else(c1_hz)
    }
}

/// Parameters of a variable-Q transform.
///
/// ## Purpose
///
/// A [`CqtConfig`] plus the bandwidth offset `gamma`: filter `k` has bandwidth
/// `alpha * f_k + gamma`, so low-frequency filters are shorter than in a CQT.
///
/// ## Intended usage
///
/// `VqtConfig::default()` is the default [`CqtConfig`] with `gamma = None`, which
/// derives `gamma` from the equivalent rectangular bandwidth (ERB) of auditory
/// filters. `gamma = Some(0.0)` gives the CQT.
///
/// ## Invariants
///
/// `gamma`, when given, is finite and non-negative ([`VqtConfig::validate`]).
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::constant_q::VqtConfig;
///
/// let cfg = VqtConfig::default();
/// assert_eq!(cfg.gamma, None);
/// assert!(cfg.with_gamma(Some(5.0)).validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct VqtConfig {
    /// Filterbank and framing parameters.
    pub cqt: CqtConfig,
    /// Bandwidth offset in Hz. `None` (the default) uses the ERB-derived
    /// `24.7 * alpha / 0.108`; `Some(0.0)` is the constant-Q transform.
    pub gamma: Option<f64>,
}

impl VqtConfig {
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Settings matching librosa 1.0's defaults; currently the same as [`Default`].
    #[must_use]
    pub fn librosa() -> Self {
        Self {
            cqt: CqtConfig::librosa(),
            gamma: None,
        }
    }

    /// Sets the CQT parameters.
    #[must_use]
    pub fn with_cqt(mut self, cqt: CqtConfig) -> Self {
        self.cqt = cqt;
        self
    }

    /// Sets the bandwidth offset (`None` = ERB-derived).
    #[must_use]
    pub fn with_gamma(mut self, gamma: Option<f64>) -> Self {
        self.gamma = gamma;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// As [`CqtConfig::validate`], and [`ParameterError`] if `gamma` is negative or
    /// not finite.
    pub fn validate(&self) -> AudioSampleResult<()> {
        self.cqt.validate()?;
        if let Some(g) = self.gamma
            && !(g.is_finite() && g >= 0.0)
        {
            return Err(param_err(
                "gamma",
                format!("gamma={g} must be finite and non-negative"),
            ));
        }
        Ok(())
    }
}

// =============================================================================
// The transforms (array level)
// =============================================================================

/// Frequencies, bandwidths and lengths of a transform's filterbank.
struct Filterbank {
    freqs: Vec<f64>,
    alpha: Vec<f64>,
    cutoff: f64,
    window_bw: f64,
}

#[derive(Clone, Copy, PartialEq)]
enum FreqLayout<'a> {
    /// Equal-tempered frequencies from `fmin` (vqt / cqt).
    Interval,
    /// Constant-Q bin frequencies (pseudo / hybrid / icqt).
    Cqt,
    /// The sorted octaves of these ratios to `fmin`, in ascending order.
    Ratios(&'a [f64]),
}

/// Drops the bins whose filters would exceed Nyquist.
fn clip_freqs(
    freqs: Vec<f64>,
    window_bw: f64,
    filter_scale: f64,
    gamma: Option<f64>,
    sr: f64,
) -> AudioSampleResult<Vec<f64>> {
    let n = freqs.len();
    let mut keep = 0;
    if n >= 2 {
        let logf: Vec<f64> = freqs.iter().map(|f| f.log2()).collect();
        let mut running = f64::NEG_INFINITY;
        for k in 0..n {
            let bpo = if k == 0 {
                1.0 / (logf[1] - logf[0])
            } else {
                1.0 / (logf[k] - logf[k - 1])
            };
            let p = 2.0f64.powf(2.0 / bpo);
            let alpha = (p - 1.0) / (p + 1.0);
            let g = gamma.unwrap_or(alpha * 24.7 / 0.108);
            let q = filter_scale / alpha;
            running = running.max(freqs[k] * (1.0 + 0.5 * window_bw / q) + 0.5 * g);
            if running < sr / 2.0 {
                keep = k + 1;
            }
        }
    }
    if keep < 1 {
        return Err(param_err(
            "fmin",
            format!(
                "unable to construct a wavelet basis for fmin={:.2} Hz and sample rate {sr:.2} Hz",
                freqs.first().copied().unwrap_or(f64::NAN)
            ),
        ));
    }
    Ok(freqs[..keep].to_vec())
}

/// The tuned filterbank of a transform; errors if the filters exceed Nyquist.
fn filterbank(
    cfg: &CqtConfig,
    sr: f64,
    tuning: f64,
    gamma: Option<f64>,
    layout: FreqLayout<'_>,
) -> AudioSampleResult<Filterbank> {
    let bpo = cfg.bins_per_octave.get();
    let fmin = cfg.fmin_or_c1() * 2.0f64.powf(tuning / bpo as f64);
    if fmin >= sr / 2.0 {
        return Err(param_err(
            "fmin",
            format!(
                "fmin={fmin} must be less than the Nyquist frequency {}",
                sr / 2.0
            ),
        ));
    }
    let window_bw = librosa_window_bandwidth(&cfg.window)?;
    let (n_bins, auto) = match cfg.n_bins {
        Some(n) => (n.get(), false),
        None => (
            (bpo as f64 * (sr.log2() - fmin.log2())).ceil() as usize,
            true,
        ),
    };
    let mut freqs = match layout {
        FreqLayout::Interval => equal_interval_freqs(n_bins, fmin, bpo),
        FreqLayout::Cqt => cqt_frequencies(n_bins, fmin, cfg.bins_per_octave, 0.0),
        FreqLayout::Ratios(ratios) => {
            let mut all: Vec<f64> = (0..n_bins)
                .map(|k| fmin * 2f64.powi((k / bpo) as i32) * ratios[k % bpo])
                .collect();
            all.sort_by(f64::total_cmp);
            all
        }
    };
    if auto {
        freqs = clip_freqs(freqs, window_bw, cfg.filter_scale, gamma, sr)?;
    }
    let alpha = transform_alpha(&freqs, bpo);
    let (_, cutoff) = wavelet_lengths_impl(&freqs, sr, window_bw, cfg.filter_scale, gamma, &alpha);
    if cutoff > sr / 2.0 {
        let top = freqs.last().copied().unwrap_or(f64::NAN);
        return Err(config_err(
            "constant-Q transform",
            format!(
                "the filter of the top bin ({top:.2} Hz) has cut-off {cutoff:.2} Hz, above the Nyquist frequency {:.2} Hz; reduce n_bins or fmin",
                sr / 2.0
            ),
        ));
    }
    Ok(Filterbank {
        freqs,
        alpha,
        cutoff,
        window_bw,
    })
}

fn num_two_factors(mut x: usize) -> usize {
    if x == 0 {
        return 0;
    }
    let mut n = 0;
    while x.is_multiple_of(2) {
        n += 1;
        x /= 2;
    }
    n
}

/// Stacks per-octave responses (top octave first) into
/// `n_bins` rows, trimmed to the shortest frame count.
fn trim_stack<A: Clone + num_traits::Zero>(resp: &[Array3<A>], n_bins: usize) -> Array3<A> {
    let channels = resp[0].dim().0;
    let max_col = resp.iter().map(|r| r.dim().2).min().unwrap_or(0);
    let mut out = Array3::<A>::zeros((channels, n_bins, max_col));
    let mut end = n_bins;
    for r in resp {
        let n_oct = r.dim().1;
        if end < n_oct {
            out.slice_mut(s![.., ..end, ..])
                .assign(&r.slice(s![.., n_oct - end.., ..max_col]));
            end = 0;
        } else {
            out.slice_mut(s![.., end - n_oct..end, ..])
                .assign(&r.slice(s![.., .., ..max_col]));
            end -= n_oct;
        }
    }
    out
}

/// Complex filter response: rectangular-window centred STFT, then the
/// sparse basis product.
fn cqt_response<F: AudioFloat>(
    y: ArrayView2<'_, F>,
    basis: &FftBasis<F>,
    hop: usize,
    pad_mode: StftPadMode,
) -> AudioSampleResult<Array3<Complex<F>>> {
    let frames = n_centred_frames(y.ncols(), basis.n_fft, hop);
    let mut out = Array3::<Complex<F>>::zeros((y.nrows(), basis.rows.len(), frames));
    for_each_frame(y, basis.n_fft, hop, pad_mode, None, frames, |c, t, spec| {
        for (k, row) in basis.rows.iter().enumerate() {
            out[[c, k, t]] = row.dot(spec);
        }
    })?;
    Ok(out)
}

/// The variable-Q transform of `(channels, samples)` rows (with the tuning
/// resolved), computed in `F`.
fn vqt_rows<F: AudioFloat>(
    y: CowArray<'_, F, Ix2>,
    sr: f64,
    cfg: &CqtConfig,
    tuning: f64,
    gamma: Option<f64>,
) -> AudioSampleResult<Array3<Complex<F>>> {
    vqt_rows_in(y, sr, cfg, tuning, gamma, FreqLayout::Interval)
}

/// [`vqt_rows`] for a given layout of the centre frequencies.
fn vqt_rows_in<F: AudioFloat>(
    y: CowArray<'_, F, Ix2>,
    sr: f64,
    cfg: &CqtConfig,
    tuning: f64,
    gamma: Option<f64>,
    layout: FreqLayout<'_>,
) -> AudioSampleResult<Array3<Complex<F>>> {
    cfg.validate()?;
    if y.ncols() == 0 {
        return Err(param_err("signal", "cannot transform an empty signal"));
    }
    let fb = filterbank(cfg, sr, tuning, gamma, layout)?;
    let n_bins = fb.freqs.len();
    let bpo = cfg.bins_per_octave.get();
    let n_octaves = n_bins.div_ceil(bpo);
    let n_filters = bpo.min(n_bins);
    let quality = cfg.resampling_quality;

    // Early downsampling: halve the rate first while the highest cut-off and the
    // hop allow it.
    let nyquist = sr / 2.0;
    let count1 = (((nyquist / fb.cutoff).log2().ceil() as i64) - 1 - 1).max(0) as usize;
    let count2 =
        (num_two_factors(cfg.hop_length.get()) as i64 - n_octaves as i64 + 1).max(0) as usize;
    let downsample_count = count1.min(count2);
    let mut y = y;
    let mut sr = sr;
    let mut hop = cfg.hop_length.get();
    if downsample_count > 0 {
        let factor = 1usize << downsample_count;
        hop /= factor;
        if y.ncols() < factor {
            return Err(param_err(
                "signal",
                format!(
                    "input signal length={} is too short for a {n_octaves}-octave CQT",
                    y.ncols()
                ),
            ));
        }
        let mut gain = (factor as f64).sqrt();
        if !cfg.scale {
            gain *= (factor as f64).sqrt();
        }
        y = CowArray::from(resample_rows(y.view(), factor as u32, 1, quality, gain)?);
        sr /= factor as f64;
    }

    let mut responses = Vec::with_capacity(n_octaves);
    let (mut my_y, mut my_sr, mut my_hop) = (y, sr, hop);
    for i in 0..n_octaves {
        let hi = n_bins - n_filters * i;
        let lo = n_bins.saturating_sub(n_filters * (i + 1));
        let freqs_oct = &fb.freqs[lo..hi];
        let alpha_oct = &fb.alpha[lo..hi];
        let (lengths_oct, _) = wavelet_lengths_impl(
            freqs_oct,
            my_sr,
            fb.window_bw,
            cfg.filter_scale,
            gamma,
            alpha_oct,
        );
        let basis = vqt_filter_fft(
            my_sr,
            freqs_oct,
            alpha_oct,
            &lengths_oct,
            &cfg.window,
            cfg.norm,
            cfg.sparsity,
            None,
            (sr / my_sr).sqrt(),
        )?
        .cast::<F>();
        responses.push(cqt_response(my_y.view(), &basis, my_hop, cfg.pad_mode)?);
        if i + 1 < n_octaves {
            let f_max_next = fb.freqs[lo - 1];
            if my_hop % 2 == 0 && f_max_next <= my_sr / 5.0 {
                my_hop /= 2;
                my_sr /= 2.0;
                my_y = CowArray::from(resample_rows(
                    my_y.view(),
                    2,
                    1,
                    quality,
                    std::f64::consts::SQRT_2,
                )?);
            }
        }
    }
    let mut v = trim_stack(&responses, n_bins);
    if cfg.scale {
        let (lengths, _) = wavelet_lengths_impl(
            &fb.freqs,
            sr,
            fb.window_bw,
            cfg.filter_scale,
            gamma,
            &fb.alpha,
        );
        for (k, &len) in lengths.iter().enumerate() {
            let d = F::of(len.sqrt());
            v.slice_mut(s![.., k, ..]).mapv_inplace(|z| z / d);
        }
    }
    Ok(v)
}

/// A variable-Q transform whose bins follow a tuning system rather than equal temperament.
///
/// One octave of `ratios` repeats over the bins, each octave twice the frequency of the
/// last. The number of ratios replaces `bins_per_octave` of the configuration.
///
/// # Arguments
/// - `audio`: the signal.
/// - `config`: the transform.
/// - `ratios`: the frequencies of one octave relative to `fmin`; finite and positive.
///
/// # Returns
/// The response, shape `(channels, n_bins, frames)`.
///
/// # Errors
/// [`ParameterError`] if `ratios` is empty or holds a value that is not finite and positive,
/// or for the conditions of [`AudioConstantQ::vqt`].
///
/// # Examples
/// ```rust
/// use audio_samples::operations::constant_q::{CqtConfig, VqtConfig, vqt_with_intervals};
/// use audio_samples::{sample_rate, sine_wave};
/// use std::num::NonZeroUsize;
/// use std::time::Duration;
///
/// let audio = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(22050), 0.5);
/// let ratios = [1.0, 9.0 / 8.0, 5.0 / 4.0, 4.0 / 3.0, 3.0 / 2.0, 5.0 / 3.0, 15.0 / 8.0];
/// let cqt = CqtConfig::new().with_fmin(Some(110.0)).with_n_bins(NonZeroUsize::new(21));
/// let config = VqtConfig::new().with_cqt(cqt);
/// let v = vqt_with_intervals(&audio, &config, &ratios).unwrap();
/// assert_eq!(v.dim().1, 21);
/// ```
pub fn vqt_with_intervals<T: StandardSample>(
    audio: &AudioSamples<'_, T>,
    config: &VqtConfig,
    ratios: &[f64],
) -> AudioSampleResult<Array3<Complex<T::Float>>> {
    let Some(bins_per_octave) = NonZeroUsize::new(ratios.len()) else {
        return Err(param_err("ratios", "must not be empty"));
    };
    if ratios.iter().any(|r| !r.is_finite() || *r <= 0.0) {
        return Err(param_err("ratios", "must be finite and positive"));
    }
    let mut cfg = config.cqt.clone();
    cfg.bins_per_octave = bins_per_octave;
    let scoped = VqtConfig {
        cqt: cfg,
        gamma: config.gamma,
    };
    scoped.validate()?;
    let (y, sr) = (finite_float_rows(audio)?, audio.sample_rate_hz());
    let tuning = resolve_tuning(y.view(), sr, scoped.cqt.tuning, bins_per_octave)?;
    vqt_rows_in(
        y,
        sr,
        &scoped.cqt,
        tuning,
        scoped.gamma,
        FreqLayout::Ratios(ratios),
    )
}

/// Pseudo-CQT of rows (tuning resolved), computed in `F`.
fn pseudo_cqt_rows<F: AudioFloat>(
    y: ArrayView2<'_, F>,
    sr: f64,
    cfg: &CqtConfig,
    tuning: f64,
) -> AudioSampleResult<Array3<F>> {
    cfg.validate()?;
    if y.ncols() == 0 {
        return Err(param_err("signal", "cannot transform an empty signal"));
    }
    let fb = filterbank(cfg, sr, tuning, Some(0.0), FreqLayout::Cqt)?;
    let (lengths, _) = wavelet_lengths_impl(
        &fb.freqs,
        sr,
        fb.window_bw,
        cfg.filter_scale,
        Some(0.0),
        &fb.alpha,
    );
    let mut basis = vqt_filter_fft(
        sr,
        &fb.freqs,
        &fb.alpha,
        &lengths,
        &cfg.window,
        cfg.norm,
        cfg.sparsity,
        Some(cfg.hop_length.get()),
        1.0,
    )?;
    for row in &mut basis.rows {
        for v in &mut row.val {
            *v = Complex::new(v.norm(), 0.0);
        }
    }
    let basis = basis.cast::<F>();
    let n_fft = basis.n_fft;
    let hop = cfg.hop_length.get();
    let window: Vec<F> = get_window(&Window::Hann, n_fft, true)?
        .iter()
        .map(|&w| F::of(w))
        .collect();
    let frames = n_centred_frames(y.ncols(), n_fft, hop);
    let mut out = Array3::<F>::zeros((y.nrows(), fb.freqs.len(), frames));
    let mut mag = vec![F::zero(); n_fft / 2 + 1];
    for_each_frame(
        y,
        n_fft,
        hop,
        cfg.pad_mode,
        Some(&window),
        frames,
        |c, t, spec| {
            for (m, z) in mag.iter_mut().zip(spec) {
                *m = z.norm();
            }
            for (k, row) in basis.rows.iter().enumerate() {
                out[[c, k, t]] = row.dot_abs(&mag);
            }
        },
    )?;
    for (k, &len) in lengths.iter().enumerate() {
        let factor = F::of(if cfg.scale {
            1.0 / (n_fft as f64).sqrt()
        } else {
            (len / n_fft as f64).sqrt()
        });
        out.slice_mut(s![.., k, ..]).mapv_inplace(|v| v * factor);
    }
    Ok(out)
}

/// Hybrid CQT of rows (tuning resolved), computed in `F`.
fn hybrid_cqt_rows<F: AudioFloat>(
    y: CowArray<'_, F, Ix2>,
    sr: f64,
    cfg: &CqtConfig,
    tuning: f64,
) -> AudioSampleResult<Array3<F>> {
    cfg.validate()?;
    let fb = filterbank(cfg, sr, tuning, Some(0.0), FreqLayout::Cqt)?;
    let (lengths, _) = wavelet_lengths_impl(
        &fb.freqs,
        sr,
        fb.window_bw,
        cfg.filter_scale,
        Some(0.0),
        &fb.alpha,
    );
    let hop = cfg.hop_length.get();
    let pseudo: Vec<bool> = lengths
        .iter()
        .map(|l| 2.0f64.powf(l.log2().ceil()) < 2.0 * hop as f64)
        .collect();
    let n_pseudo = pseudo.iter().filter(|&&p| p).count();
    let n_bins = fb.freqs.len();
    let n_full = n_bins - n_pseudo;
    let mut resp = Vec::with_capacity(2);
    if n_pseudo > 0 {
        let fmin_pseudo = fb
            .freqs
            .iter()
            .zip(&pseudo)
            .filter(|(_, p)| **p)
            .map(|(f, _)| *f)
            .fold(f64::INFINITY, f64::min);
        let sub = cfg
            .clone()
            .with_fmin(Some(fmin_pseudo))
            .with_n_bins(NonZeroUsize::new(n_pseudo))
            .with_tuning(Some(0.0));
        resp.push(pseudo_cqt_rows(y.view(), sr, &sub, 0.0)?);
    }
    if n_full > 0 {
        let fmin = cfg.fmin_or_c1() * 2.0f64.powf(tuning / cfg.bins_per_octave.get() as f64);
        let sub = cfg
            .clone()
            .with_fmin(Some(fmin))
            .with_n_bins(NonZeroUsize::new(n_full))
            .with_tuning(Some(0.0));
        resp.push(vqt_rows(y, sr, &sub, 0.0, Some(0.0))?.mapv(|z| z.norm()));
    }
    Ok(trim_stack(&resp, n_bins))
}

/// Inverse CQT of a `(channels, n_bins, frames)` matrix, computed in `F`.
fn icqt_rows<F: AudioFloat>(
    c: ArrayView3<'_, Complex<F>>,
    sr: f64,
    cfg: &CqtConfig,
    length: Option<usize>,
) -> AudioSampleResult<Array2<F>> {
    cfg.validate()?;
    let tuning = cfg.tuning.ok_or_else(|| {
        param_err(
            "tuning",
            "icqt needs the tuning used for the forward transform (tuning = None is not allowed)",
        )
    })?;
    let (channels, n_bins, total_frames) = c.dim();
    if channels == 0 || n_bins == 0 || total_frames == 0 {
        return Err(param_err("C", "the constant-Q matrix is empty"));
    }
    let bpo = cfg.bins_per_octave.get();
    let fmin = cfg.fmin_or_c1() * 2.0f64.powf(tuning / bpo as f64);
    let n_octaves = n_bins.div_ceil(bpo);
    let freqs = cqt_frequencies(n_bins, fmin, cfg.bins_per_octave, 0.0);
    let alpha = transform_alpha(&freqs, bpo);
    let window_bw = librosa_window_bandwidth(&cfg.window)?;
    let (lengths, _) =
        wavelet_lengths_impl(&freqs, sr, window_bw, cfg.filter_scale, Some(0.0), &alpha);
    let hop = cfg.hop_length.get();
    let mut n_frames = total_frames;
    if let Some(len) = length {
        let max_len = lengths.iter().copied().fold(0.0, f64::max);
        n_frames = n_frames.min(((len as f64 + max_len) / hop as f64).ceil() as usize);
    }
    let c = c.slice(s![.., .., ..n_frames]);

    let mut srs = vec![sr];
    let mut hops = vec![hop];
    for _ in 0..n_octaves.saturating_sub(1) {
        if hops[0] % 2 == 0 {
            srs.insert(0, srs[0] * 0.5);
            hops.insert(0, hops[0] / 2);
        } else {
            srs.insert(0, srs[0]);
            hops.insert(0, hops[0]);
        }
    }

    let mut y: Option<Array2<F>> = None;
    for (i, (&my_sr, &my_hop)) in srs.iter().zip(&hops).enumerate() {
        let lo = bpo * i;
        let n_filters = bpo.min(n_bins - lo);
        let hi = lo + n_filters;
        let (lengths_oct, _) = wavelet_lengths_impl(
            &freqs[lo..hi],
            my_sr,
            window_bw,
            cfg.filter_scale,
            Some(0.0),
            &alpha[lo..hi],
        );
        let basis = vqt_filter_fft(
            my_sr,
            &freqs[lo..hi],
            &alpha[lo..hi],
            &lengths_oct,
            &cfg.window,
            cfg.norm,
            cfg.sparsity,
            None,
            1.0,
        )?;
        let n_fft = basis.n_fft;
        let n_freq = n_fft / 2 + 1;
        // Per-filter weights: 1 / ||filter||^2 * n_fft / L (and sqrt(L) when scaled).
        let weights: Vec<F> = basis
            .rows
            .iter()
            .zip(&lengths[lo..hi])
            .map(|(row, &len)| {
                let power: f64 = row.val.iter().map(Complex::norm_sqr).sum();
                let fp = 1.0 / power * (n_fft as f64 / len);
                F::of(if cfg.scale { len.sqrt() * fp } else { fp })
            })
            .collect();
        let basis = basis.cast::<F>();
        let mut d = Array3::<Complex<F>>::zeros((channels, n_freq, n_frames));
        let mut coefs = vec![Complex::new(F::zero(), F::zero()); n_frames];
        for ch in 0..channels {
            for (k, (row, &w)) in basis.rows.iter().zip(&weights).enumerate() {
                for (coef, &v) in coefs.iter_mut().zip(c.slice(s![ch, lo + k, ..])) {
                    *coef = v * w;
                }
                // Frame by frame along each bin's row of `d` (each bin's sum
                // still adds the filters in order).
                for (&j, &v) in row.idx.iter().zip(&row.val) {
                    let v = v.conj();
                    for (dst, &coef) in d.slice_mut(s![ch, j, ..]).iter_mut().zip(&coefs) {
                        *dst += v * coef;
                    }
                }
            }
        }
        let stft_cfg = StftConfig::new()
            .with_n_fft(NonZeroUsize::new(n_fft).ok_or_else(|| param_err("n_fft", "zero"))?)
            .with_hop_length(NonZeroUsize::new(my_hop))
            .with_window(Window::Rectangular);
        let y_oct = istft(d.view(), &stft_cfg, None)?;
        let factor = (sr / my_sr).floor() as u32;
        let y_oct = resample_rows(y_oct.view(), 1, factor.max(1), cfg.resampling_quality, 1.0)?;
        match &mut y {
            None => y = Some(y_oct),
            Some(acc) => {
                let n = acc.ncols().min(y_oct.ncols());
                let mut head = acc.slice_mut(s![.., ..n]);
                head += &y_oct.slice(s![.., ..n]);
            }
        }
    }
    let mut y = y.expect("at least one octave");
    if let Some(len) = length {
        let mut fixed = Array2::<F>::zeros((channels, len));
        let n = len.min(y.ncols());
        fixed.slice_mut(s![.., ..n]).assign(&y.slice(s![.., ..n]));
        y = fixed;
    }
    Ok(y)
}

// =============================================================================
// Chroma
// =============================================================================

/// Builds the matrix that folds constant-Q bins into chroma (pitch-class) bins.
///
/// Each group of `bins_per_octave / n_chroma` adjacent CQT bins (centred on a pitch
/// class) is summed into one chroma bin, in every octave; the rows are rotated so
/// that row 0 is C (`base_c = true`) or A (`base_c = false`), given that the first
/// CQT bin is at `fmin`.
///
/// # Arguments
///
/// * `n_input` - Number of CQT bins.
/// * `bins_per_octave` - CQT bins per octave; a multiple of `n_chroma`.
/// * `n_chroma` - Number of chroma bins (usually 12).
/// * `fmin` - Centre frequency of the first CQT bin in Hz (before tuning); `None`
///   means C1.
/// * `base_c` - If `true` (the usual choice) chroma row 0 is C, otherwise A.
///
/// # Returns
///
/// A `(n_chroma, n_input)` matrix of zeros and ones.
///
/// # Errors
///
/// [`ParameterError`] if `bins_per_octave` is not a multiple of `n_chroma`,
/// `fmin` is not finite and positive, or the matrix would exceed 2^28 elements.
///
/// The matrix is `f64`. [`smooth_chroma_map`] convolves it with a smoothing
/// window.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::constant_q::cq_to_chroma;
///
/// let m = cq_to_chroma(84, nzu!(12), nzu!(12), None, true).unwrap();
/// assert_eq!(m.dim(), (12, 84));
/// // CQT bin 0 (C1) and bin 12 (C2) both map to chroma C.
/// assert_eq!(m[[0, 0]], 1.0);
/// assert_eq!(m[[0, 12]], 1.0);
/// assert_eq!(m.sum(), 84.0);
/// ```
pub fn cq_to_chroma(
    n_input: usize,
    bins_per_octave: NonZeroUsize,
    n_chroma: NonZeroUsize,
    fmin: Option<f64>,
    base_c: bool,
) -> AudioSampleResult<Array2<f64>> {
    let bpo = bins_per_octave.get();
    let nc = n_chroma.get();
    if !bpo.is_multiple_of(nc) {
        return Err(config_err(
            "cq_to_chroma",
            format!("bins_per_octave={bpo} must be an integer multiple of n_chroma={nc}"),
        ));
    }
    let fmin = fmin.unwrap_or_else(c1_hz);
    require_finite_positive("fmin", fmin)?;
    crate::utils::check_derived_len(
        "n_input",
        "a chroma mapping",
        nc as f64 * n_input as f64,
        crate::utils::MAX_TABLE_ELEMENTS,
    )?;
    let n_merge = bpo / nc;
    let shift = n_merge / 2;
    let midi_0 = hz_to_midi_librosa(fmin).rem_euclid(12.0);
    let roll = if base_c { midi_0 } else { midi_0 - 9.0 };
    let roll = (roll * (nc as f64 / 12.0)).round_ties_even() as i64;
    let mut out = Array2::<f64>::zeros((nc, n_input));
    for j in 0..n_input {
        // Column j of the tiled, column-rolled matrix has its one in row
        // ((j % bpo + shift) % bpo) / n_merge; then rows are rolled by `roll`.
        let src_row = ((j % bpo + shift) % bpo) / n_merge;
        let row = (src_row as i64 + roll).rem_euclid(nc as i64) as usize;
        out[[row, j]] = 1.0;
    }
    Ok(out)
}

/// Normalises every frame of a `(channels, features, frames)` array along the
/// feature axis.
#[cfg(feature = "feature-extraction")]
fn normalize_features<F: AudioFloat>(
    x: &mut Array3<F>,
    norm: Option<f64>,
) -> AudioSampleResult<()> {
    use crate::operations::tonal::{NormalizeConfig, normalize_array};
    if norm.is_some() {
        *x = normalize_array(
            x,
            &NormalizeConfig::new().with_norm(norm).with_axis(Some(1)),
        )?;
    }
    Ok(())
}

/// `chroma[c, :, t] = M · X[c, :, t]` for a `(n_chroma, n_bins)` matrix `M`
/// (computed in `f64`, rounded to `F` once).
#[cfg(feature = "feature-extraction")]
fn project<F: AudioFloat>(m: &Array2<f64>, x: &Array3<F>) -> Array3<F> {
    let m = m.mapv(F::of);
    let (channels, _, frames) = x.dim();
    let mut out = Array3::<F>::zeros((channels, m.nrows(), frames));
    for c in 0..channels {
        out.index_axis_mut(ndarray::Axis(0), c)
            .assign(&m.dot(&x.index_axis(ndarray::Axis(0), c)));
    }
    out
}

/// Which constant-Q transform a chroma feature is computed from.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::constant_q::CqtMode;
///
/// assert_eq!(CqtMode::default(), CqtMode::Full);
/// ```
#[cfg(feature = "feature-extraction")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CqtMode {
    /// The full recursive CQT ([`AudioConstantQ::cqt`]); the default.
    #[default]
    Full,
    /// The hybrid CQT ([`AudioConstantQ::hybrid_cqt`]).
    Hybrid,
}

/// Parameters of a constant-Q chromagram.
///
/// ## Purpose
///
/// A chromagram folds the energy of every octave onto pitch classes. This config
/// sets the CQT it is computed from (hop, `fmin`, octaves, bins per octave, tuning,
/// full or hybrid) and the post-processing (threshold, per-frame normalisation).
///
/// ## Intended usage
///
/// `ChromaCqtConfig::default()`: 12 chroma bins from a 7-octave CQT with 36 bins
/// per octave, each frame scaled to a maximum of 1, and A440 tuning
/// (`tuning = Some(0.0)`). [`ChromaCqtConfig::librosa`] differs only in
/// estimating the tuning from the signal (`tuning = None`).
///
/// ## Invariants
///
/// `bins_per_octave` must be a multiple of `n_chroma` ([`ChromaCqtConfig::validate`]).
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::constant_q::ChromaCqtConfig;
///
/// let cfg = ChromaCqtConfig::default().with_tuning(Some(0.0));
/// assert!(cfg.validate().is_ok());
/// assert_eq!(cfg.n_octaves.get(), 7);
/// ```
#[cfg(feature = "feature-extraction")]
#[derive(Debug, Clone, PartialEq)]
pub struct ChromaCqtConfig {
    /// Samples between frames. Default 512.
    pub hop_length: NonZeroUsize,
    /// Lowest CQT frequency in Hz (before tuning); `None` = C1. Also fixes which
    /// CQT bins map to which pitch class.
    pub fmin: Option<f64>,
    /// Per-frame normalisation of the chroma vector (see [`WaveletConfig::norm`] for
    /// the values). Default `Some(inf)` (maximum = 1).
    pub norm: Option<f64>,
    /// Chroma values below this are set to 0 before normalisation; `None` disables
    /// it. Default `Some(0.0)`.
    pub threshold: Option<f64>,
    /// Tuning offset in fractions of a CQT bin. Default `Some(0.0)` (A440).
    /// `None` estimates it from the signal ([`ChromaCqtConfig::librosa`]).
    pub tuning: Option<f64>,
    /// Number of chroma bins. Default 12.
    pub n_chroma: NonZeroUsize,
    /// Number of CQT octaves analysed. Default 7.
    pub n_octaves: NonZeroUsize,
    /// CQT bins per octave; `None` = `n_chroma`. Must be a multiple of `n_chroma`.
    /// Default `Some(36)`.
    pub bins_per_octave: Option<NonZeroUsize>,
    /// Full or hybrid CQT. Default [`CqtMode::Full`].
    pub cqt_mode: CqtMode,
}

#[cfg(feature = "feature-extraction")]
impl Default for ChromaCqtConfig {
    fn default() -> Self {
        Self {
            hop_length: crate::nzu!(512),
            fmin: None,
            norm: Some(f64::INFINITY),
            threshold: Some(0.0),
            tuning: Some(0.0),
            n_chroma: crate::nzu!(12),
            n_octaves: crate::nzu!(7),
            bins_per_octave: Some(crate::nzu!(36)),
            cqt_mode: CqtMode::Full,
        }
    }
}

#[cfg(feature = "feature-extraction")]
impl ChromaCqtConfig {
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Settings matching librosa 1.0's defaults: [`Default`] with `tuning = None`
    /// (the tuning is estimated from the signal).
    #[must_use]
    pub fn librosa() -> Self {
        Self {
            tuning: None,
            ..Self::default()
        }
    }

    /// Sets the hop length.
    #[must_use]
    pub fn with_hop_length(mut self, hop_length: NonZeroUsize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Sets the lowest CQT frequency (`None` = C1).
    #[must_use]
    pub fn with_fmin(mut self, fmin: Option<f64>) -> Self {
        self.fmin = fmin;
        self
    }

    /// Sets the per-frame normalisation.
    #[must_use]
    pub fn with_norm(mut self, norm: Option<f64>) -> Self {
        self.norm = norm;
        self
    }

    /// Sets the threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: Option<f64>) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets the tuning (`None` = estimate).
    #[must_use]
    pub fn with_tuning(mut self, tuning: Option<f64>) -> Self {
        self.tuning = tuning;
        self
    }

    /// Sets the number of chroma bins.
    #[must_use]
    pub fn with_n_chroma(mut self, n_chroma: NonZeroUsize) -> Self {
        self.n_chroma = n_chroma;
        self
    }

    /// Sets the number of octaves.
    #[must_use]
    pub fn with_n_octaves(mut self, n_octaves: NonZeroUsize) -> Self {
        self.n_octaves = n_octaves;
        self
    }

    /// Sets the CQT bins per octave (`None` = `n_chroma`).
    #[must_use]
    pub fn with_bins_per_octave(mut self, bins_per_octave: Option<NonZeroUsize>) -> Self {
        self.bins_per_octave = bins_per_octave;
        self
    }

    /// Sets the CQT mode.
    #[must_use]
    pub fn with_cqt_mode(mut self, cqt_mode: CqtMode) -> Self {
        self.cqt_mode = cqt_mode;
        self
    }

    /// The effective CQT bins per octave.
    fn bpo(&self) -> NonZeroUsize {
        self.bins_per_octave.unwrap_or(self.n_chroma)
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the bins per octave are not a multiple of `n_chroma`,
    /// `fmin` or `tuning` are not finite (and positive, for `fmin`), `threshold` is
    /// NaN, or `norm` is invalid.
    pub fn validate(&self) -> AudioSampleResult<()> {
        let bpo = self.bpo().get();
        if !bpo.is_multiple_of(self.n_chroma.get()) {
            return Err(config_err(
                "chroma_cqt",
                format!(
                    "bins_per_octave={bpo} must be an integer multiple of n_chroma={}",
                    self.n_chroma
                ),
            ));
        }
        if let Some(t) = self.threshold
            && t.is_nan()
        {
            return Err(param_err("threshold", "threshold must not be NaN"));
        }
        validate_norm("norm", self.norm)?;
        self.cqt_config().validate()
    }

    /// The CQT configuration chroma_cqt uses ([`CqtConfig::default`] otherwise).
    fn cqt_config(&self) -> CqtConfig {
        CqtConfig::default()
            .with_hop_length(self.hop_length)
            .with_fmin(self.fmin)
            .with_n_bins(NonZeroUsize::new(self.n_octaves.get() * self.bpo().get()))
            .with_bins_per_octave(self.bpo())
            .with_tuning(self.tuning)
    }
}

/// Parameters of chroma energy normalised statistics (CENS).
///
/// ## Purpose
///
/// CENS features are a coarse, smoothed chroma that is robust to dynamics, timbre
/// and articulation; they are used for audio matching and retrieval.
///
/// ## Intended usage
///
/// `ChromaCensConfig::default()` and [`ChromaCensConfig::librosa`] differ only in
/// the A440 tuning of [`ChromaCqtConfig::default`]. The chroma is computed from
/// `chroma` (its `norm` and `threshold` are not used: CENS starts from the
/// unnormalised chroma), L1-normalised, quantised, smoothed over time and
/// normalised by `norm`.
///
/// ## Invariants
///
/// As [`ChromaCqtConfig`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::constant_q::ChromaCensConfig;
///
/// let cfg = ChromaCensConfig::default();
/// assert_eq!(cfg.win_len_smooth.map(|w| w.get()), Some(41));
/// assert_eq!(cfg.norm, Some(2.0));
/// ```
#[cfg(feature = "feature-extraction")]
#[derive(Debug, Clone, PartialEq)]
pub struct ChromaCensConfig {
    /// The chroma (CQT) parameters; `norm` and `threshold` are ignored.
    pub chroma: ChromaCqtConfig,
    /// Final per-frame normalisation. Default `Some(2.0)` (unit Euclidean
    /// norm).
    pub norm: Option<f64>,
    /// Length of the temporal smoothing window in frames (the window has
    /// `win_len_smooth + 2` points whose end points are zero). `None` disables the
    /// smoothing. Default `Some(41)`.
    pub win_len_smooth: Option<NonZeroUsize>,
    /// Shape of the smoothing window (computed symmetric). Default Hann.
    pub smoothing_window: Window,
}

#[cfg(feature = "feature-extraction")]
impl Default for ChromaCensConfig {
    fn default() -> Self {
        Self {
            chroma: ChromaCqtConfig::default(),
            norm: Some(2.0),
            win_len_smooth: Some(crate::nzu!(41)),
            smoothing_window: Window::Hann,
        }
    }
}

#[cfg(feature = "feature-extraction")]
impl ChromaCensConfig {
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Settings matching librosa 1.0's defaults: [`Default`] with
    /// [`ChromaCqtConfig::librosa`] (the tuning is estimated from the signal).
    #[must_use]
    pub fn librosa() -> Self {
        Self {
            chroma: ChromaCqtConfig::librosa(),
            ..Self::default()
        }
    }

    /// Sets the chroma parameters.
    #[must_use]
    pub fn with_chroma(mut self, chroma: ChromaCqtConfig) -> Self {
        self.chroma = chroma;
        self
    }

    /// Sets the final normalisation.
    #[must_use]
    pub fn with_norm(mut self, norm: Option<f64>) -> Self {
        self.norm = norm;
        self
    }

    /// Sets the smoothing length (`None` = no smoothing).
    #[must_use]
    pub fn with_win_len_smooth(mut self, win_len_smooth: Option<NonZeroUsize>) -> Self {
        self.win_len_smooth = win_len_smooth;
        self
    }

    /// Sets the smoothing window.
    #[must_use]
    pub fn with_smoothing_window(mut self, window: Window) -> Self {
        self.smoothing_window = window;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// As [`ChromaCqtConfig::validate`], and [`ParameterError`] if `norm` is invalid
    /// or the smoothing window is invalid.
    pub fn validate(&self) -> AudioSampleResult<()> {
        self.chroma.validate()?;
        validate_norm("norm", self.norm)?;
        if let Some(w) = self.win_len_smooth {
            get_window(&self.smoothing_window, w.get() + 2, false)?;
        }
        Ok(())
    }
}

/// Parameters of a variable-Q chromagram over equal-tempered bins.
///
/// ## Purpose
///
/// A chromagram with one chroma bin per VQT bin of an octave
/// (`n_chroma = bins_per_octave`), computed from the VQT.
///
/// ## Intended usage
///
/// `ChromaVqtConfig::default()`: 12 bins per octave over 7 octaves from C1, hop
/// 512, each frame scaled to a maximum of 1, and `gamma = Some(0.0)` (so the
/// underlying transform is a CQT).
///
/// ## Invariants
///
/// `gamma` is finite and non-negative ([`ChromaVqtConfig::validate`]).
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::constant_q::ChromaVqtConfig;
///
/// let cfg = ChromaVqtConfig::default().with_gamma(Some(10.0));
/// assert!(cfg.validate().is_ok());
/// ```
#[cfg(feature = "feature-extraction")]
#[derive(Debug, Clone, PartialEq)]
pub struct ChromaVqtConfig {
    /// Samples between frames. Default 512.
    pub hop_length: NonZeroUsize,
    /// Lowest VQT frequency in Hz; `None` = C1.
    pub fmin: Option<f64>,
    /// Per-frame normalisation (see [`WaveletConfig::norm`]). Default
    /// `Some(inf)`.
    pub norm: Option<f64>,
    /// Values below this are set to 0 before normalisation; `None` disables it.
    /// Default `Some(0.0)`.
    pub threshold: Option<f64>,
    /// Number of VQT octaves. Default 7.
    pub n_octaves: NonZeroUsize,
    /// VQT bins per octave (and chroma bins). Default 12.
    pub bins_per_octave: NonZeroUsize,
    /// VQT bandwidth offset in Hz (`None` = ERB-derived). Default `Some(0.0)`.
    pub gamma: Option<f64>,
}

#[cfg(feature = "feature-extraction")]
impl Default for ChromaVqtConfig {
    fn default() -> Self {
        Self {
            hop_length: crate::nzu!(512),
            fmin: None,
            norm: Some(f64::INFINITY),
            threshold: Some(0.0),
            n_octaves: crate::nzu!(7),
            bins_per_octave: crate::nzu!(12),
            gamma: Some(0.0),
        }
    }
}

#[cfg(feature = "feature-extraction")]
impl ChromaVqtConfig {
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Settings matching librosa 1.0's defaults; currently the same as [`Default`].
    #[must_use]
    pub fn librosa() -> Self {
        Self::default()
    }

    /// Sets the hop length.
    #[must_use]
    pub fn with_hop_length(mut self, hop_length: NonZeroUsize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Sets the lowest frequency (`None` = C1).
    #[must_use]
    pub fn with_fmin(mut self, fmin: Option<f64>) -> Self {
        self.fmin = fmin;
        self
    }

    /// Sets the per-frame normalisation.
    #[must_use]
    pub fn with_norm(mut self, norm: Option<f64>) -> Self {
        self.norm = norm;
        self
    }

    /// Sets the threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: Option<f64>) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets the number of octaves.
    #[must_use]
    pub fn with_n_octaves(mut self, n_octaves: NonZeroUsize) -> Self {
        self.n_octaves = n_octaves;
        self
    }

    /// Sets the bins per octave.
    #[must_use]
    pub fn with_bins_per_octave(mut self, bins_per_octave: NonZeroUsize) -> Self {
        self.bins_per_octave = bins_per_octave;
        self
    }

    /// Sets the bandwidth offset (`None` = ERB-derived).
    #[must_use]
    pub fn with_gamma(mut self, gamma: Option<f64>) -> Self {
        self.gamma = gamma;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `fmin` is not finite and positive, `gamma` is negative
    /// or not finite, `threshold` is NaN or `norm` is invalid.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if let Some(t) = self.threshold
            && t.is_nan()
        {
            return Err(param_err("threshold", "threshold must not be NaN"));
        }
        validate_norm("norm", self.norm)?;
        self.vqt_config().validate()
    }

    fn vqt_config(&self) -> VqtConfig {
        VqtConfig::default()
            .with_cqt(
                CqtConfig::default()
                    .with_hop_length(self.hop_length)
                    .with_fmin(self.fmin)
                    .with_n_bins(NonZeroUsize::new(
                        self.n_octaves.get() * self.bins_per_octave.get(),
                    ))
                    .with_bins_per_octave(self.bins_per_octave),
            )
            .with_gamma(self.gamma)
    }
}

/// Parameters of the tonal centroid features.
///
/// ## Purpose
///
/// The tonnetz projects each frame's chroma onto three circles (fifths, minor
/// thirds, major thirds), giving a 6-dimensional harmonic descriptor.
///
/// ## Intended usage
///
/// `TonnetzConfig::default()` computes the chroma with [`ChromaCqtConfig::default`]
/// (A440 tuning); [`TonnetzConfig::librosa`] with [`ChromaCqtConfig::librosa`]
/// (estimated tuning). To use another chromagram (e.g. an STFT chroma), call
/// [`tonnetz_from_chroma`] directly.
///
/// ## Invariants
///
/// As [`ChromaCqtConfig`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::constant_q::{ChromaCqtConfig, TonnetzConfig};
///
/// let cfg = TonnetzConfig::default().with_chroma(ChromaCqtConfig::default().with_tuning(Some(0.0)));
/// assert!(cfg.validate().is_ok());
/// ```
#[cfg(feature = "feature-extraction")]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TonnetzConfig {
    /// The chromagram parameters.
    pub chroma: ChromaCqtConfig,
}

#[cfg(feature = "feature-extraction")]
impl TonnetzConfig {
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Settings matching librosa 1.0's defaults: the chroma uses
    /// [`ChromaCqtConfig::librosa`] (the tuning is estimated from the signal).
    #[must_use]
    pub fn librosa() -> Self {
        Self {
            chroma: ChromaCqtConfig::librosa(),
        }
    }

    /// Sets the chroma parameters.
    #[must_use]
    pub fn with_chroma(mut self, chroma: ChromaCqtConfig) -> Self {
        self.chroma = chroma;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// As [`ChromaCqtConfig::validate`].
    pub fn validate(&self) -> AudioSampleResult<()> {
        self.chroma.validate()
    }
}

/// Computes chroma energy normalised statistics from an unnormalised chromagram.
///
/// Each frame is L1-normalised and quantised (each of the thresholds 0.4, 0.2, 0.1
/// and 0.05 that a value exceeds adds 0.25), the result is smoothed over time with
/// a normalised `win_len_smooth + 2`-point symmetric window (zero beyond the ends),
/// and each frame is normalised by `norm`.
///
/// # Arguments
///
/// * `chroma` - `(channels, n_chroma, frames)` non-negative chromagram of element
///   type `F` (`f32` or `f64`).
/// * `norm` - Final per-frame normalisation (see [`WaveletConfig::norm`]).
/// * `win_len_smooth` - Smoothing length in frames; `None` disables smoothing.
/// * `smoothing_window` - Window shape (computed in `f64`, rounded to `F`).
///
/// # Returns
///
/// The CENS features, same shape and element type as `chroma`.
///
/// # Errors
///
/// [`ParameterError`] if `norm` or the window is invalid;
/// [`ProcessingError`] if `chroma` contains non-finite values.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::filterbanks::Window;
/// use audio_samples::operations::constant_q::chroma_cens_from_chroma;
/// use ndarray::Array3;
///
/// let chroma = Array3::from_shape_fn((1, 12, 50), |(_, k, _)| if k == 9 { 1.0_f64 } else { 0.01 });
/// let cens = chroma_cens_from_chroma(&chroma, Some(2.0), Some(nzu!(41)), &Window::Hann).unwrap();
/// assert!((cens[[0, 9, 25]] - 1.0).abs() < 1e-12);
/// ```
///
/// # References
///
/// M. Müller, F. Kurth and M. Clausen, "Audio matching via chroma-based statistical
/// features," ISMIR 2005.
#[cfg(feature = "feature-extraction")]
pub fn chroma_cens_from_chroma<F: AudioFloat, S: ndarray::Data<Elem = F>>(
    chroma: &ndarray::ArrayBase<S, ndarray::Ix3>,
    norm: Option<f64>,
    win_len_smooth: Option<NonZeroUsize>,
    smoothing_window: &Window,
) -> AudioSampleResult<Array3<F>> {
    validate_norm("norm", norm)?;
    let mut x = chroma.to_owned();
    normalize_features(&mut x, Some(1.0))?;
    let steps = [0.4, 0.2, 0.1, 0.05].map(F::of);
    let quarter = F::of(0.25);
    x.mapv_inplace(|v| F::of_usize(steps.iter().filter(|&&s| v > s).count()) * quarter);
    if let Some(wl) = win_len_smooth {
        let m = wl.get() + 2;
        let mut win = get_window(smoothing_window, m, false)?.to_vec();
        let sum: f64 = win.iter().sum();
        for w in &mut win {
            *w /= sum;
        }
        let win: Vec<F> = win.iter().map(|&w| F::of(w)).collect();
        let (channels, bins, frames) = x.dim();
        let centre = (m / 2) as isize;
        let mut smoothed = Array3::<F>::zeros(x.dim());
        for c in 0..channels {
            for k in 0..bins {
                for t in 0..frames {
                    let mut acc = F::zero();
                    for (j, &w) in win.iter().enumerate() {
                        let src = t as isize + centre - j as isize;
                        if (0..frames as isize).contains(&src) {
                            acc += w * x[[c, k, src as usize]];
                        }
                    }
                    smoothed[[c, k, t]] = acc;
                }
            }
        }
        x = smoothed;
    }
    normalize_features(&mut x, norm)?;
    Ok(x)
}

/// Computes the tonal centroid features (tonnetz) of a chromagram.
///
/// Each frame's chroma vector is L1-normalised and projected onto the circle of
/// fifths (rows 0-1), the circle of minor thirds (rows 2-3) and the circle of major
/// thirds (rows 4-5), with radii 1, 1 and 0.5.
///
/// # Arguments
///
/// * `chroma` - `(channels, n_chroma, frames)` chromagram of element type `F`
///   (`f32` or `f64`; any `n_chroma`, pitch classes are spread evenly over 12
///   semitones).
///
/// # Returns
///
/// A `(channels, 6, frames)` array of element type `F`. The projection matrix is
/// computed in `f64` and rounded to `F` once.
///
/// # Errors
///
/// [`ProcessingError`] if `chroma` contains non-finite values.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::constant_q::tonnetz_from_chroma;
/// use ndarray::Array3;
///
/// // A pure C: angle 0 on every circle (fifths: sin row 0 = 0, cos row 1 = 1).
/// let mut chroma = Array3::<f64>::zeros((1, 12, 1));
/// chroma[[0, 0, 0]] = 1.0;
/// let t = tonnetz_from_chroma(&chroma).unwrap();
/// assert!(t[[0, 0, 0]].abs() < 1e-12 && (t[[0, 1, 0]] - 1.0).abs() < 1e-12);
/// ```
///
/// # References
///
/// C. Harte, M. Sandler and M. Gasser, "Detecting harmonic change in musical
/// audio," ACM AMCMM 2006.
#[cfg(feature = "feature-extraction")]
pub fn tonnetz_from_chroma<F: AudioFloat, S: ndarray::Data<Elem = F>>(
    chroma: &ndarray::ArrayBase<S, ndarray::Ix3>,
) -> AudioSampleResult<Array3<F>> {
    let n_chroma = chroma.dim().1;
    let mut x = chroma.to_owned();
    normalize_features(&mut x, Some(1.0))?;
    let scale = [
        7.0 / 6.0,
        7.0 / 6.0,
        3.0 / 2.0,
        3.0 / 2.0,
        2.0 / 3.0,
        2.0 / 3.0,
    ];
    let radius = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5];
    let step = 12.0 / n_chroma as f64;
    let mut phi = Array2::<f64>::zeros((6, n_chroma));
    for (p, (&sc, &r)) in scale.iter().zip(&radius).enumerate() {
        for k in 0..n_chroma {
            let mut v = sc * (k as f64 * step);
            if p % 2 == 0 {
                v -= 0.5;
            }
            phi[[p, k]] = r * (std::f64::consts::PI * v).cos();
        }
    }
    Ok(project(&phi, &x))
}

#[cfg(feature = "feature-extraction")]
fn apply_threshold<F: AudioFloat>(x: &mut Array3<F>, threshold: Option<f64>) {
    if let Some(th) = threshold {
        let th = F::of(th);
        x.mapv_inplace(|v| if v < th { F::zero() } else { v });
    }
}

// =============================================================================
// Trait
// =============================================================================

/// The iterations of [`AudioConstantQ::griffinlim_cqt`] from the unit phases `angles`.
fn griffinlim_cqt_from<T: StandardSample>(
    magnitude: &Array3<T::Float>,
    mut angles: Array3<Complex<T::Float>>,
    config: &CqtConfig,
    sample_rate: SampleRate,
    n_iter: usize,
    momentum: f64,
    length: Option<NonZeroUsize>,
) -> AudioSampleResult<AudioSamples<'static, T>> {
    if !(momentum.is_finite() && momentum >= 0.0) {
        return Err(param_err(
            "momentum",
            format!("momentum={momentum} must be finite and non-negative"),
        ));
    }
    let tuning = config.tuning.ok_or_else(|| {
        param_err(
            "tuning",
            "griffinlim_cqt needs an explicit tuning (not None)",
        )
    })?;
    let sr = f64::from(sample_rate.get());
    let len = length.map(NonZeroUsize::get);
    let n_bins = NonZeroUsize::new(magnitude.dim().1)
        .ok_or_else(|| param_err("magnitude", "the magnitude has no bins"))?;
    let fwd = config.clone().with_n_bins(Some(n_bins));
    let mut rebuilt: Option<Array3<Complex<T::Float>>> = None;
    let with_phase = |angles: &Array3<Complex<T::Float>>| {
        ndarray::Zip::from(magnitude)
            .and(angles)
            .map_collect(|&m, &a| a * m)
    };
    for _ in 0..n_iter {
        let inverse = icqt_rows(with_phase(&angles).view(), sr, config, len)?;
        let next = vqt_rows(CowArray::from(inverse), sr, &fwd, tuning, Some(0.0))?;
        if next.dim() != magnitude.dim() {
            return Err(config_err(
                "griffinlim_cqt",
                format!(
                    "the CQT of the reconstruction has shape {:?}, the magnitude {:?}; pass the original length",
                    next.dim(),
                    magnitude.dim()
                ),
            ));
        }
        let k = T::Float::of(momentum / (1.0 + momentum));
        match &rebuilt {
            Some(prev) => ndarray::Zip::from(&mut angles)
                .and(&next)
                .and(prev)
                .for_each(|a, &r, &p| *a = r - p * k),
            None => angles.assign(&next),
        }
        angles.mapv_inplace(|a| {
            a / (a.norm() + <T::Float as num_traits::Float>::min_positive_value())
        });
        rebuilt = Some(next);
    }
    let y = icqt_rows(with_phase(&angles).view(), sr, config, len)?;
    float_rows_to_audio::<T>(y, sample_rate, "griffinlim_cqt")
}

/// Griffin-Lim phase retrieval from a constant-Q magnitude with random initial phases.
///
/// # Arguments
///
/// * `magnitude` - `(channels, n_bins, frames)` non-negative CQT magnitude.
/// * `config` - The forward parameters (`tuning` must be `Some`).
/// * `sample_rate` - Sample rate of the signal.
/// * `n_iter` - Number of iterations (typically 32).
/// * `momentum` - Momentum (typically 0.99; must be non-negative).
/// * `length` - Output length per channel, if known.
/// * `seed` - `Some(seed)` makes the result reproducible; `None` draws a fresh seed
///   on every call.
///
/// # Returns
///
/// The reconstructed audio, of sample type `T`. The iterations run in `T::Float`.
///
/// # Errors
///
/// As [`AudioConstantQ::griffinlim_cqt`].
///
/// The phases come from this crate's RNG, as for
/// [`griffinlim`](crate::operations::effects::griffinlim) with random phases.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::constant_q::{AudioConstantQ, CqtConfig, griffinlim_cqt_random};
/// use audio_samples::{sample_rate, sine_wave};
/// use std::time::Duration;
///
/// let audio = sine_wave::<f64>(440.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
/// let cfg = CqtConfig::default();
/// let mag = audio.cqt(&cfg).unwrap().mapv(|z| z.norm());
/// let y = griffinlim_cqt_random::<f64>(&mag, &cfg, sample_rate!(22050), 4, 0.99, Some(nzu!(11025)), Some(0))
///     .unwrap();
/// assert_eq!(y.samples_per_channel().get(), 11025);
/// ```
#[cfg(feature = "random-generation")]
pub fn griffinlim_cqt_random<T: StandardSample>(
    magnitude: &Array3<T::Float>,
    config: &CqtConfig,
    sample_rate: SampleRate,
    n_iter: usize,
    momentum: f64,
    length: Option<NonZeroUsize>,
    seed: Option<u64>,
) -> AudioSampleResult<AudioSamples<'static, T>> {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed.unwrap_or_else(rand::random::<u64>));
    let angles = Array3::from_shape_simple_fn(magnitude.dim(), || {
        let phi = 2.0 * std::f64::consts::PI * rng.random::<f64>();
        Complex::new(T::Float::of(phi.cos()), T::Float::of(phi.sin()))
    });
    griffinlim_cqt_from::<T>(
        magnitude,
        angles,
        config,
        sample_rate,
        n_iter,
        momentum,
        length,
    )
}

/// Constant-Q analysis of [`AudioSamples`]: CQT, VQT, pseudo and
/// hybrid CQT, inverse CQT, and the chroma / tonnetz features built on the CQT.
///
/// All analysis outputs are shaped `(channels, n_bins, n_frames)` with
/// `n_frames = 1 + len / hop_length` (frames centred on multiples of the hop).
/// Outputs have the element type [`Float`](Self::Float) (or
/// `Complex<Self::Float>`): `f32` for `u8`, `i16`, `I24` and `f32` audio, `f64`
/// for `i32` and `f64` audio. The computation, including the per-octave
/// resampling, runs in that type. The
/// reconstructions ([`icqt`](Self::icqt), [`griffinlim_cqt`](Self::griffinlim_cqt))
/// return audio of the sample type [`Sample`](Self::Sample).
pub trait AudioConstantQ {
    /// The float type of the outputs and of the computation: the sample type's
    /// [`AudioSample::Float`](crate::AudioSample::Float).
    type Float: AudioFloat;

    /// The sample type of the audio. [`icqt`](Self::icqt) and
    /// [`griffinlim_cqt`](Self::griffinlim_cqt) return audio of this type.
    type Sample: StandardSample;

    /// Computes the constant-Q transform.
    ///
    /// Bin `k` is centred at `fmin * 2^((k + tuning) / bins_per_octave)` Hz and
    /// filters the signal with a windowed complex exponential of `Q * sr / f_k`
    /// samples (`Q = filter_scale / alpha`, `alpha` the relative bandwidth of the
    /// bin spacing). The octaves are computed top-down; after each octave the
    /// signal is low-pass filtered and decimated by 2 (with the hop halved), so
    /// every octave uses the same short FFT basis.
    ///
    /// # Arguments
    ///
    /// * `config` - Transform parameters (usually [`CqtConfig::default`]).
    ///
    /// # Returns
    ///
    /// A complex `(channels, n_bins, 1 + len / hop_length)` array.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample.
    ///
    /// - [`ParameterError`] if the configuration is invalid, `fmin` (after tuning)
    ///   is not below the Nyquist frequency, the highest filter's cut-off exceeds
    ///   the Nyquist frequency (reduce `n_bins`), or the signal is too short.
    /// - With `config.tuning = None` the tuning is estimated as
    ///   [`estimate_tuning`](crate::operations::tonal::estimate_tuning_from_signal)
    ///   does; that needs the `feature-extraction` feature, and without it `None`
    ///   is a [`ParameterError`]. Errors of the estimate propagate.
    ///
    /// The per-octave decimation uses the crate's resampler (see the
    /// [module documentation](self#resampling-between-octaves)).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::constant_q::{AudioConstantQ, CqtConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f32>(261.63, Duration::from_secs(1), sample_rate!(22050), 1.0);
    /// let c = audio.cqt(&CqtConfig::default()).unwrap();
    /// assert_eq!(c.dim(), (1, 84, 44));
    /// // Middle C is bin 36; a unit sinusoid gives about sqrt(L) / 2 with scale = true.
    /// let m = c[[0, 36, 22]].norm();
    /// assert!(m > 10.0 && m < 20.0);
    /// ```
    ///
    /// # References
    ///
    /// C. Schörkhuber and A. Klapuri, "Constant-Q transform toolbox for music
    /// processing," SMC 2010.
    fn cqt(&self, config: &CqtConfig) -> AudioSampleResult<Array3<Complex<Self::Float>>>;

    /// Computes the variable-Q transform over equal-tempered bins.
    ///
    /// As [`cqt`](Self::cqt), but filter `k` has bandwidth `alpha * f_k + gamma`
    /// (shorter low-frequency filters). `gamma = Some(0.0)` is the CQT.
    ///
    /// # Arguments
    ///
    /// * `config` - Transform parameters ([`VqtConfig::default`] uses an
    ///   ERB-derived `gamma`).
    ///
    /// # Returns
    ///
    /// A complex `(channels, n_bins, 1 + len / hop_length)` array.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample.
    ///
    /// As [`cqt`](Self::cqt), and [`ParameterError`] for an invalid `gamma`.
    ///
    /// The bins are equal-tempered; [`vqt_with_intervals`] takes other frequency
    /// ratios. Resampling as in [`cqt`](Self::cqt).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::constant_q::{AudioConstantQ, VqtConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(110.0, Duration::from_secs(1), sample_rate!(22050), 0.5);
    /// let v = audio.vqt(&VqtConfig::default()).unwrap();
    /// assert_eq!(v.dim(), (1, 84, 44));
    /// ```
    ///
    /// # References
    ///
    /// V. Lostanlen, S. Sridhar et al., "Time-frequency scattering accurately models
    /// auditory similarities between instrumental playing techniques," EURASIP JASMP
    /// 2021 (the ERB-based `gamma`).
    fn vqt(&self, config: &VqtConfig) -> AudioSampleResult<Array3<Complex<Self::Float>>>;

    /// Computes the pseudo constant-Q transform (magnitudes).
    ///
    /// A single Hann-window STFT whose magnitude is multiplied by the magnitudes of
    /// the CQT filters' spectra: no octave recursion and no resampling. It is a
    /// good approximation of `|CQT|` where the filters are short (high frequencies)
    /// and is used by [`hybrid_cqt`](Self::hybrid_cqt) there.
    ///
    /// # Arguments
    ///
    /// * `config` - Transform parameters (`resampling_quality` is not used).
    ///
    /// # Returns
    ///
    /// A real, non-negative `(channels, n_bins, 1 + len / hop_length)` array.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample.
    ///
    /// As [`cqt`](Self::cqt).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::nzu;
    /// use audio_samples::operations::constant_q::{AudioConstantQ, CqtConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(1760.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
    /// let cfg = CqtConfig::default().with_fmin(Some(880.0)).with_n_bins(Some(nzu!(24)));
    /// let p = audio.pseudo_cqt(&cfg).unwrap();
    /// assert_eq!(p.dim(), (1, 24, 22));
    /// ```
    fn pseudo_cqt(&self, config: &CqtConfig) -> AudioSampleResult<Array3<Self::Float>>;

    /// Computes the hybrid constant-Q transform (magnitudes).
    ///
    /// Bins whose filters (rounded up to a power of two) are shorter than twice the
    /// hop use the [`pseudo_cqt`](Self::pseudo_cqt); the remaining (low) bins use
    /// the full [`cqt`](Self::cqt) magnitude.
    ///
    /// # Arguments
    ///
    /// * `config` - Transform parameters.
    ///
    /// # Returns
    ///
    /// A real, non-negative `(channels, n_bins, frames)` array.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample.
    ///
    /// As [`cqt`](Self::cqt).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::constant_q::{AudioConstantQ, CqtConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
    /// let h = audio.hybrid_cqt(&CqtConfig::default()).unwrap();
    /// assert_eq!(h.dim(), (1, 84, 22));
    /// ```
    fn hybrid_cqt(&self, config: &CqtConfig) -> AudioSampleResult<Array3<Self::Float>>;

    /// Reconstructs audio from a constant-Q transform.
    ///
    /// Each octave's bins are mapped back to an STFT through the conjugate
    /// transpose of its filter basis (weighted by the inverse filter energy),
    /// inverted with a rectangular-window ISTFT, upsampled to the original rate and
    /// summed. This is an approximate inverse: for a CQT computed with the same
    /// parameters it recovers the signal's band-limited content between `fmin` and
    /// the top bin. The reconstruction runs in [`Float`](Self::Float) and is
    /// converted to [`Sample`](Self::Sample) with the crate's audio-aware
    /// conversion (integer types round and saturate).
    ///
    /// # Arguments
    ///
    /// * `c` - `(channels, n_bins, frames)` complex CQT.
    /// * `config` - The forward parameters (`n_bins` and `pad_mode` are not used;
    ///   `tuning` must be `Some`).
    /// * `sample_rate` - Sample rate of the signal the CQT was computed from.
    /// * `length` - If `Some(n)`, the output has exactly `n` samples per channel.
    ///
    /// # Returns
    ///
    /// The reconstructed audio, of sample type [`Sample`](Self::Sample).
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the configuration is invalid, `tuning` is `None`, or
    /// `c` is empty.
    ///
    /// Upsampling uses the crate's resampler.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::nzu;
    /// use audio_samples::operations::constant_q::{AudioConstantQ, CqtConfig};
    /// use audio_samples::{AudioSamples, sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_secs(1), sample_rate!(22050), 0.5);
    /// let cfg = CqtConfig::default();
    /// let c = audio.cqt(&cfg).unwrap();
    /// let y = AudioSamples::<f64>::icqt(&c, &cfg, sample_rate!(22050), Some(nzu!(22050))).unwrap();
    /// assert_eq!(y.samples_per_channel().get(), 22050);
    /// ```
    fn icqt(
        c: &Array3<Complex<Self::Float>>,
        config: &CqtConfig,
        sample_rate: SampleRate,
        length: Option<NonZeroUsize>,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// Reconstructs audio from a constant-Q magnitude by Griffin-Lim phase
    /// retrieval.
    ///
    /// Starting from zero phase, alternates [`icqt`](Self::icqt) and
    /// [`cqt`](Self::cqt), keeping the given magnitude and the new phase, with the
    /// "fast Griffin-Lim" momentum update.
    ///
    /// # Arguments
    ///
    /// * `magnitude` - `(channels, n_bins, frames)` non-negative CQT magnitude.
    /// * `config` - The forward parameters (`tuning` must be `Some`).
    /// * `sample_rate` - Sample rate of the signal.
    /// * `n_iter` - Number of iterations (typically 32).
    /// * `momentum` - Momentum (typically 0.99; must be non-negative).
    /// * `length` - Output length per channel, if known.
    ///
    /// # Returns
    ///
    /// The reconstructed audio, of sample type [`Sample`](Self::Sample). The
    /// iterations run in [`Float`](Self::Float).
    ///
    /// # Errors
    ///
    /// As [`icqt`](Self::icqt) and [`cqt`](Self::cqt); [`ParameterError`] if
    /// `momentum` is negative or not finite.
    ///
    /// This is the deterministic zero-phase start; [`griffinlim_cqt_random`] starts
    /// from random phases.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::nzu;
    /// use audio_samples::operations::constant_q::{AudioConstantQ, CqtConfig};
    /// use audio_samples::{AudioSamples, sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
    /// let cfg = CqtConfig::default();
    /// let mag = audio.cqt(&cfg).unwrap().mapv(|z| z.norm());
    /// let y = AudioSamples::<f64>::griffinlim_cqt(&mag, &cfg, sample_rate!(22050), 4, 0.99, Some(nzu!(11025)))
    ///     .unwrap();
    /// assert_eq!(y.samples_per_channel().get(), 11025);
    /// ```
    ///
    /// # References
    ///
    /// N. Perraudin, P. Balazs and P. L. Søndergaard, "A fast Griffin-Lim
    /// algorithm," WASPAA 2013.
    fn griffinlim_cqt(
        magnitude: &Array3<Self::Float>,
        config: &CqtConfig,
        sample_rate: SampleRate,
        n_iter: usize,
        momentum: f64,
        length: Option<NonZeroUsize>,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// Computes a constant-Q chromagram.
    ///
    /// The CQT magnitude (`n_octaves * bins_per_octave` bins from `fmin`) is folded
    /// onto `n_chroma` pitch classes with [`cq_to_chroma`], values below `threshold`
    /// are zeroed, and every frame is normalised by `norm`.
    ///
    /// # Arguments
    ///
    /// * `config` - Parameters ([`ChromaCqtConfig::default`], or
    ///   [`ChromaCqtConfig::librosa`], which estimates the tuning).
    ///
    /// # Returns
    ///
    /// A `(channels, n_chroma, frames)` array; row 0 is C.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample.
    ///
    /// As [`cqt`](Self::cqt), and [`ParameterError`] if `bins_per_octave` is not a
    /// multiple of `n_chroma`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::constant_q::{AudioConstantQ, ChromaCqtConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(392.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
    /// let chroma = audio.chroma_cqt(&ChromaCqtConfig::default().with_tuning(Some(0.0))).unwrap();
    /// assert_eq!(chroma.dim(), (1, 12, 22));
    /// assert!((chroma[[0, 7, 11]] - 1.0).abs() < 1e-12); // G
    /// ```
    #[cfg(feature = "feature-extraction")]
    fn chroma_cqt(&self, config: &ChromaCqtConfig) -> AudioSampleResult<Array3<Self::Float>>;

    /// Computes chroma energy normalised statistics (CENS).
    ///
    /// See [`chroma_cens_from_chroma`] for the post-processing.
    ///
    /// # Arguments
    ///
    /// * `config` - Parameters ([`ChromaCensConfig::default`], or
    ///   [`ChromaCensConfig::librosa`], which estimates the tuning).
    ///
    /// # Returns
    ///
    /// A `(channels, n_chroma, frames)` array.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample.
    ///
    /// As [`chroma_cqt`](Self::chroma_cqt).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::constant_q::{AudioConstantQ, ChromaCensConfig, ChromaCqtConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
    /// let cfg = ChromaCensConfig::default().with_chroma(ChromaCqtConfig::default().with_tuning(Some(0.0)));
    /// let cens = audio.chroma_cens(&cfg).unwrap();
    /// assert_eq!(cens.dim(), (1, 12, 22));
    /// ```
    #[cfg(feature = "feature-extraction")]
    fn chroma_cens(&self, config: &ChromaCensConfig) -> AudioSampleResult<Array3<Self::Float>>;

    /// Computes a variable-Q chromagram over equal-tempered bins.
    ///
    /// # Arguments
    ///
    /// * `config` - Parameters (usually [`ChromaVqtConfig::default`]).
    ///
    /// # Returns
    ///
    /// A `(channels, bins_per_octave, frames)` array.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample.
    ///
    /// As [`vqt`](Self::vqt).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::constant_q::{AudioConstantQ, ChromaVqtConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
    /// let chroma = audio.chroma_vqt(&ChromaVqtConfig::default()).unwrap();
    /// assert_eq!(chroma.dim(), (1, 12, 22));
    /// ```
    #[cfg(feature = "feature-extraction")]
    fn chroma_vqt(&self, config: &ChromaVqtConfig) -> AudioSampleResult<Array3<Self::Float>>;

    /// Computes the tonal centroid features.
    ///
    /// The chromagram is computed with [`chroma_cqt`](Self::chroma_cqt) and projected
    /// with [`tonnetz_from_chroma`].
    ///
    /// # Arguments
    ///
    /// * `config` - Parameters ([`TonnetzConfig::default`], or
    ///   [`TonnetzConfig::librosa`], which estimates the tuning).
    ///
    /// # Returns
    ///
    /// A `(channels, 6, frames)` array.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample.
    ///
    /// As [`chroma_cqt`](Self::chroma_cqt).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::constant_q::{AudioConstantQ, ChromaCqtConfig, TonnetzConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
    /// let cfg = TonnetzConfig::default().with_chroma(ChromaCqtConfig::default().with_tuning(Some(0.0)));
    /// let t = audio.tonnetz(&cfg).unwrap();
    /// assert_eq!(t.dim(), (1, 6, 22));
    /// ```
    #[cfg(feature = "feature-extraction")]
    fn tonnetz(&self, config: &TonnetzConfig) -> AudioSampleResult<Array3<Self::Float>>;
}

/// Resolves `tuning = None` with the tuning estimate of `operations::tonal` (feature
/// `feature-extraction`; without it `None` is an error).
fn resolve_tuning<F: AudioFloat>(
    y: ArrayView2<'_, F>,
    sr: f64,
    tuning: Option<f64>,
    bins_per_octave: NonZeroUsize,
) -> AudioSampleResult<f64> {
    if let Some(t) = tuning {
        return Ok(t);
    }
    #[cfg(feature = "feature-extraction")]
    {
        use crate::operations::tonal::{TuningConfig, estimate_tuning_from_signal};
        estimate_tuning_from_signal(
            y,
            sr,
            &TuningConfig::default().with_bins_per_octave(bins_per_octave),
        )
    }
    #[cfg(not(feature = "feature-extraction"))]
    {
        let _ = (y, sr, bins_per_octave);
        Err(param_err(
            "tuning",
            "estimating the tuning (tuning = None) requires the `feature-extraction` feature; pass Some(tuning)",
        ))
    }
}

#[cfg(feature = "feature-extraction")]
fn cqt_magnitude<T: StandardSample>(
    audio: &AudioSamples<'_, T>,
    cfg: &CqtConfig,
    mode: CqtMode,
) -> AudioSampleResult<Array3<T::Float>> {
    match mode {
        CqtMode::Full => Ok(audio.cqt(cfg)?.mapv(|z| z.norm())),
        CqtMode::Hybrid => audio.hybrid_cqt(cfg),
    }
}

/// Constant-Q chroma with an explicit `norm` / `threshold` override (CENS).
#[cfg(feature = "feature-extraction")]
fn chroma_cqt_impl<T: StandardSample>(
    audio: &AudioSamples<'_, T>,
    config: &ChromaCqtConfig,
    norm: Option<f64>,
    threshold: Option<f64>,
) -> AudioSampleResult<Array3<T::Float>> {
    config.validate()?;
    let c = cqt_magnitude(audio, &config.cqt_config(), config.cqt_mode)?;
    let m = cq_to_chroma(c.dim().1, config.bpo(), config.n_chroma, config.fmin, true)?;
    let mut chroma = project(&m, &c);
    apply_threshold(&mut chroma, threshold);
    normalize_features(&mut chroma, norm)?;
    Ok(chroma)
}

impl<T> AudioConstantQ for AudioSamples<'_, T>
where
    T: StandardSample,
{
    type Float = T::Float;
    type Sample = T;

    fn cqt(&self, config: &CqtConfig) -> AudioSampleResult<Array3<Complex<T::Float>>> {
        config.validate()?;
        let (y, sr) = (finite_float_rows(self)?, self.sample_rate_hz());
        let tuning = resolve_tuning(y.view(), sr, config.tuning, config.bins_per_octave)?;
        vqt_rows(y, sr, config, tuning, Some(0.0))
    }

    fn vqt(&self, config: &VqtConfig) -> AudioSampleResult<Array3<Complex<T::Float>>> {
        config.validate()?;
        let (y, sr) = (finite_float_rows(self)?, self.sample_rate_hz());
        let tuning = resolve_tuning(y.view(), sr, config.cqt.tuning, config.cqt.bins_per_octave)?;
        vqt_rows(y, sr, &config.cqt, tuning, config.gamma)
    }

    fn pseudo_cqt(&self, config: &CqtConfig) -> AudioSampleResult<Array3<T::Float>> {
        config.validate()?;
        let (y, sr) = (finite_float_rows(self)?, self.sample_rate_hz());
        let tuning = resolve_tuning(y.view(), sr, config.tuning, config.bins_per_octave)?;
        pseudo_cqt_rows(y.view(), sr, config, tuning)
    }

    fn hybrid_cqt(&self, config: &CqtConfig) -> AudioSampleResult<Array3<T::Float>> {
        config.validate()?;
        let (y, sr) = (finite_float_rows(self)?, self.sample_rate_hz());
        let tuning = resolve_tuning(y.view(), sr, config.tuning, config.bins_per_octave)?;
        hybrid_cqt_rows(y, sr, config, tuning)
    }

    fn icqt(
        c: &Array3<Complex<T::Float>>,
        config: &CqtConfig,
        sample_rate: SampleRate,
        length: Option<NonZeroUsize>,
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        let y = icqt_rows(
            c.view(),
            f64::from(sample_rate.get()),
            config,
            length.map(NonZeroUsize::get),
        )?;
        float_rows_to_audio::<T>(y, sample_rate, "icqt")
    }

    fn griffinlim_cqt(
        magnitude: &Array3<T::Float>,
        config: &CqtConfig,
        sample_rate: SampleRate,
        n_iter: usize,
        momentum: f64,
        length: Option<NonZeroUsize>,
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        let angles = Array3::<Complex<T::Float>>::from_elem(
            magnitude.dim(),
            Complex::new(T::Float::of(1.0), T::Float::of(0.0)),
        );
        griffinlim_cqt_from::<T>(
            magnitude,
            angles,
            config,
            sample_rate,
            n_iter,
            momentum,
            length,
        )
    }

    #[cfg(feature = "feature-extraction")]
    fn chroma_cqt(&self, config: &ChromaCqtConfig) -> AudioSampleResult<Array3<T::Float>> {
        chroma_cqt_impl(self, config, config.norm, config.threshold)
    }

    #[cfg(feature = "feature-extraction")]
    fn chroma_cens(&self, config: &ChromaCensConfig) -> AudioSampleResult<Array3<T::Float>> {
        config.validate()?;
        let chroma = chroma_cqt_impl(self, &config.chroma, None, Some(0.0))?;
        chroma_cens_from_chroma(
            &chroma,
            config.norm,
            config.win_len_smooth,
            &config.smoothing_window,
        )
    }

    #[cfg(feature = "feature-extraction")]
    fn chroma_vqt(&self, config: &ChromaVqtConfig) -> AudioSampleResult<Array3<T::Float>> {
        config.validate()?;
        let vcfg = config.vqt_config();
        let v = self.vqt(&vcfg)?.mapv(|z| z.norm());
        let m = cq_to_chroma(
            v.dim().1,
            config.bins_per_octave,
            config.bins_per_octave,
            config.fmin,
            true,
        )?;
        let mut chroma = project(&m, &v);
        apply_threshold(&mut chroma, config.threshold);
        normalize_features(&mut chroma, config.norm)?;
        Ok(chroma)
    }

    #[cfg(feature = "feature-extraction")]
    fn tonnetz(&self, config: &TonnetzConfig) -> AudioSampleResult<Array3<T::Float>> {
        let chroma = self.chroma_cqt(&config.chroma)?;
        tonnetz_from_chroma(&chroma)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioTypeConversion;
    use crate::audio_float::audio_float_rows as audio_rows;
    use crate::sample_rate;
    use ndarray::Array1;
    use ndarray::Axis;

    #[test]
    fn extreme_sizes_are_parameter_errors() {
        assert!(matches!(
            cq_to_chroma(1 << 40, crate::nzu!(12), crate::nzu!(12), None, true),
            Err(AudioSampleError::Parameter(_))
        ));
        let freqs = [100.0, 200.0, 400.0];
        assert!(matches!(
            wavelet(&freqs, 1e15, &WaveletConfig::default()),
            Err(AudioSampleError::Parameter(_))
        ));
        assert!(wavelet(&freqs, 22050.0, &WaveletConfig::default()).is_ok());
    }

    fn tone(freq: f64, n: usize) -> AudioSamples<'static, f64> {
        let x = Array1::from_iter(
            (0..n).map(|i| 0.5 * (2.0 * std::f64::consts::PI * freq * i as f64 / 22050.0).sin()),
        );
        AudioSamples::new_mono(x, sample_rate!(22050)).unwrap()
    }

    #[test]
    fn shapes_mono_stereo_and_short_input() {
        let a = tone(440.0, 5000);
        let c = a.cqt(&CqtConfig::default()).unwrap();
        assert_eq!(c.dim(), (1, 84, 1 + 5000 / 512));
        let st = Array2::from_shape_fn((2, 3000), |(ch, i)| ((i * (ch + 1)) as f64 * 0.05).sin());
        let st = AudioSamples::new_multi_channel(st, sample_rate!(22050)).unwrap();
        assert_eq!(st.cqt(&CqtConfig::default()).unwrap().dim(), (2, 84, 6));
        assert_eq!(
            st.pseudo_cqt(&CqtConfig::default()).unwrap().dim(),
            (2, 84, 6)
        );
        assert_eq!(
            st.hybrid_cqt(&CqtConfig::default()).unwrap().dim(),
            (2, 84, 6)
        );
        // Very short input still gives one frame per hop.
        let short = tone(440.0, 100);
        assert_eq!(short.cqt(&CqtConfig::default()).unwrap().dim(), (1, 84, 1));
    }

    #[test]
    fn peak_bin_and_stereo_channels_are_independent() {
        let a = tone(440.0, 11025);
        let c = a.cqt(&CqtConfig::default()).unwrap();
        let col: Vec<f64> = (0..84).map(|k| c[[0, k, 10]].norm()).collect();
        let peak = (0..84).max_by(|&i, &j| col[i].total_cmp(&col[j])).unwrap();
        assert_eq!(peak, 45);
        let x = audio_rows(&a);
        let mut st = Array2::zeros((2, x.ncols()));
        st.row_mut(1).assign(&x.row(0));
        let st = AudioSamples::new_multi_channel(st, sample_rate!(22050)).unwrap();
        let cs = st.cqt(&CqtConfig::default()).unwrap();
        assert!(cs.index_axis(Axis(0), 0).iter().all(|z| z.norm() == 0.0));
        let d = (&cs.index_axis(Axis(0), 1) - &c.index_axis(Axis(0), 0))
            .iter()
            .fold(0.0f64, |m, z| m.max(z.norm()));
        assert!(d < 1e-12);
    }

    #[test]
    fn invalid_parameters_are_errors() {
        let a = tone(440.0, 4096);
        // Top bins above Nyquist.
        let cfg = CqtConfig::default().with_n_bins(Some(nzu!(120)));
        assert!(a.cqt(&cfg).is_err());
        assert!(
            a.cqt(&CqtConfig::default().with_fmin(Some(20000.0)))
                .is_err()
        );
        assert!(a.cqt(&CqtConfig::default().with_sparsity(1.0)).is_err());
        assert!(a.cqt(&CqtConfig::default().with_filter_scale(0.0)).is_err());
        assert!(a.cqt(&CqtConfig::default().with_norm(Some(-1.0))).is_err());
        assert!(
            a.cqt(&CqtConfig::default().with_window(Window::Custom(vec![1.0; 4])))
                .is_err()
        );
        assert!(a.vqt(&VqtConfig::default().with_gamma(Some(-1.0))).is_err());
        assert!(cq_to_chroma(84, nzu!(12), nzu!(5), None, true).is_err());
        #[cfg(feature = "feature-extraction")]
        {
            let cfg = ChromaCqtConfig::default().with_n_chroma(nzu!(7));
            assert!(a.chroma_cqt(&cfg).is_err());
        }
        assert!(wavelet_lengths(&[440.0], 22050.0, &WaveletConfig::default()).is_err());
        assert!(wavelet_lengths(&[880.0, 440.0], 22050.0, &WaveletConfig::default()).is_err());
    }

    #[test]
    fn automatic_n_bins_stays_below_nyquist() {
        let a = tone(440.0, 4096);
        let c = a.cqt(&CqtConfig::default().with_n_bins(None)).unwrap();
        // ceil(12 * log2(22050 / C1)) = 113 candidate bins; the filters of bins
        // 101.. would reach past 11025 Hz.
        assert_eq!(c.dim().1, 101);
    }

    #[test]
    fn icqt_inverts_cqt_inside_the_band() {
        let n = 22050;
        let a = tone(440.0, n);
        let cfg = CqtConfig::default();
        let c = a.cqt(&cfg).unwrap();
        let y =
            AudioSamples::<f64>::icqt(&c, &cfg, sample_rate!(22050), NonZeroUsize::new(n)).unwrap();
        let y = audio_rows(&y).row(0).to_owned();
        let x = audio_rows(&a).row(0).to_owned();
        let (mut err, mut sig) = (0.0, 0.0);
        for i in 4096..n - 4096 {
            err += (y[i] - x[i]).powi(2);
            sig += x[i] * x[i];
        }
        let snr = 10.0 * (sig / err).log10();
        // The inverse CQT is not exact (sparse, non-tight basis): on this signal
        // the round trip gives 7.65 dB (gain 1.084).
        assert!((snr - 7.65).abs() < 0.1, "icqt round-trip SNR {snr:.2} dB");
    }

    #[test]
    fn griffinlim_cqt_runs_and_keeps_length() {
        let a = tone(440.0, 8192);
        let cfg = CqtConfig::default();
        let mag = a.cqt(&cfg).unwrap().mapv(|z| z.norm());
        let y = AudioSamples::<f64>::griffinlim_cqt(
            &mag,
            &cfg,
            sample_rate!(22050),
            2,
            0.99,
            NonZeroUsize::new(8192),
        )
        .unwrap();
        assert_eq!(y.samples_per_channel().get(), 8192);
        assert!(
            AudioSamples::<f64>::griffinlim_cqt(&mag, &cfg, sample_rate!(22050), 1, -1.0, None)
                .is_err()
        );
    }

    #[test]
    fn wavelets_are_normalised_and_centred() {
        let freqs = [110.0, 220.0, 440.0];
        let (w, lengths) = wavelet(&freqs, 22050.0, &WaveletConfig::default()).unwrap();
        assert!(w.ncols().is_power_of_two() && w.ncols() as f64 >= lengths[0]);
        for row in w.outer_iter() {
            let l1: f64 = row.iter().map(|z| z.norm()).sum();
            assert!((l1 - 1.0).abs() < 1e-12);
        }
        // The longest filter is centred: its energy is symmetric about the middle.
        let row = w.row(0);
        let half = w.ncols() / 2;
        let left: f64 = row.iter().take(half).map(|z| z.norm_sqr()).sum();
        let right: f64 = row.iter().skip(half).map(|z| z.norm_sqr()).sum();
        assert!((left - right).abs() / (left + right) < 0.01);
    }

    #[test]
    fn one_frequency_needs_given_bandwidths() {
        let cfg = WaveletConfig::default();
        // Without alpha the bandwidth comes from the spacing, which one frequency lacks.
        assert!(wavelet_lengths(&[440.0], 22050.0, &cfg).is_err());
        assert!(wavelet(&[440.0], 22050.0, &cfg).is_err());
        assert!(wavelet_lengths_with_alpha(&[], 22050.0, &cfg, &[]).is_err());
        // With alpha a single filter is the first row of a longer bank with the same alpha.
        let (one, len_one) = wavelet_with_alpha(&[440.0], 22050.0, &cfg, &[0.1]).unwrap();
        let (two, len_two) =
            wavelet_with_alpha(&[440.0, 880.0], 22050.0, &cfg, &[0.1, 0.1]).unwrap();
        assert_eq!(len_one[0], len_two[0]);
        assert_eq!(one.row(0), two.row(0));
        let (lengths, cutoff) =
            wavelet_lengths_with_alpha(&[440.0], 22050.0, &cfg, &[0.1]).unwrap();
        assert!((lengths[0] - 10.0 * 22050.0 / 440.0).abs() < 1e-9);
        assert!(cutoff > 440.0);
    }

    #[test]
    #[cfg(feature = "feature-extraction")]
    fn chroma_features_shapes_and_tuning_estimate() {
        let a = tone(440.0, 11025);
        let chroma = a.chroma_cqt(&ChromaCqtConfig::default()).unwrap();
        assert_eq!(chroma.dim(), (1, 12, 22));
        assert!((chroma[[0, 9, 11]] - 1.0).abs() < 1e-12);
        let cens = a.chroma_cens(&ChromaCensConfig::default()).unwrap();
        let l2: f64 = (0..12).map(|k| cens[[0, k, 11]].powi(2)).sum();
        assert!((l2 - 1.0).abs() < 1e-12);
        assert_eq!(
            a.tonnetz(&TonnetzConfig::default()).unwrap().dim(),
            (1, 6, 22)
        );
        assert_eq!(
            a.chroma_vqt(&ChromaVqtConfig::default()).unwrap().dim(),
            (1, 12, 22)
        );
    }

    #[test]
    fn cq_to_chroma_folds_octaves() {
        let m = cq_to_chroma(36, nzu!(36), nzu!(12), None, true).unwrap();
        // Three bins per chroma, the centre bin on the pitch class: C gets 35, 0, 1.
        assert_eq!(m.row(0).iter().filter(|&&v| v == 1.0).count(), 3);
        assert_eq!((m[[0, 35]], m[[0, 0]], m[[0, 1]]), (1.0, 1.0, 1.0));
    }

    #[test]
    fn output_types_follow_the_sample_type() {
        let a = tone(440.0, 8192);
        let a16 = a.to_format::<i16>();
        let a32 = a.to_format::<i32>();
        let c16: Array3<Complex<f32>> = a16.cqt(&CqtConfig::default()).unwrap();
        let c32: Array3<Complex<f64>> = a32.cqt(&CqtConfig::default()).unwrap();
        assert_eq!(c16.dim(), c32.dim());
        let p: Array3<f32> = a16.pseudo_cqt(&CqtConfig::default()).unwrap();
        assert_eq!(p.dim(), c32.dim());
        #[cfg(feature = "feature-extraction")]
        {
            let ch: Array3<f32> = a16.chroma_cqt(&ChromaCqtConfig::default()).unwrap();
            assert_eq!(ch.dim(), (1, 12, c16.dim().2));
        }
    }

    #[test]
    fn f32_cqt_agrees_with_f64_cqt() {
        // The same i16 samples analysed in f32 and in f64, so the difference is
        // f32 round-off of the FFTs, filter products and resampling: measured
        // 4.8e-7 of the peak. Bound 2e-6 (17 f32 eps).
        let a16 = tone(440.0, 22050).to_format::<i16>();
        let c32 = a16.cqt(&CqtConfig::default()).unwrap();
        let c64 = a16.to_format::<f64>().cqt(&CqtConfig::default()).unwrap();
        let peak = c64.iter().fold(0.0f64, |m, z| m.max(z.norm()));
        let err = c32.iter().zip(c64.iter()).fold(0.0f64, |m, (p, q)| {
            m.max((Complex::new(f64::from(p.re), f64::from(p.im)) - q).norm())
        });
        eprintln!("f32 vs f64 cqt: {:.3e} of the peak", err / peak);
        assert!(err / peak < 2e-6, "{:e}", err / peak);
    }

    #[test]
    fn icqt_and_griffinlim_return_the_receiver_sample_type() {
        // i16 in, i16 out, close to the f64 path re-quantised to i16.
        let n = 22050;
        let a16 = tone(440.0, n).to_format::<i16>();
        let cfg = CqtConfig::default();
        let c = a16.cqt(&cfg).unwrap();
        let y: AudioSamples<'static, i16> =
            AudioSamples::<i16>::icqt(&c, &cfg, sample_rate!(22050), NonZeroUsize::new(n)).unwrap();
        let c64 = a16.to_format::<f64>().cqt(&cfg).unwrap();
        let y64 = AudioSamples::<f64>::icqt(&c64, &cfg, sample_rate!(22050), NonZeroUsize::new(n))
            .unwrap()
            .to_format::<i16>();
        let max_diff = |p: &AudioSamples<'static, i16>, q: &AudioSamples<'static, i16>| {
            let (p, q) = (
                p.as_mono().unwrap().as_view(),
                q.as_mono().unwrap().as_view(),
            );
            assert_eq!(p.len(), q.len());
            p.iter()
                .zip(q.iter())
                .map(|(&a, &b)| (i32::from(a) - i32::from(b)).abs())
                .max()
                .unwrap()
        };
        // The f32 and f64 reconstructions differ by f32 round-off of a signal of
        // amplitude ~0.5 (about 1e-7 of full scale, below one LSB = 3e-5); a
        // value near a rounding boundary can still round to the neighbouring
        // code, so the bound is one LSB (measured 1).
        eprintln!("icqt i16 vs f64 re-quantised: {} LSB", max_diff(&y, &y64));
        assert!(max_diff(&y, &y64) <= 1);

        let mag = c.mapv(|z| z.norm());
        let g: AudioSamples<'static, i16> = AudioSamples::<i16>::griffinlim_cqt(
            &mag,
            &cfg,
            sample_rate!(22050),
            2,
            0.99,
            NonZeroUsize::new(n),
        )
        .unwrap();
        let g64 = AudioSamples::<f64>::griffinlim_cqt(
            &c64.mapv(|z| z.norm()),
            &cfg,
            sample_rate!(22050),
            2,
            0.99,
            NonZeroUsize::new(n),
        )
        .unwrap()
        .to_format::<i16>();
        assert_eq!(g.samples_per_channel().get(), n);
        // Two iterations, each through icqt and cqt in f32: the phase estimates
        // differ by f32 round-off. Measured 1 LSB; bound 2 LSB.
        eprintln!(
            "griffinlim_cqt i16 vs f64 re-quantised: {} LSB",
            max_diff(&g, &g64)
        );
        assert!(max_diff(&g, &g64) <= 2, "{}", max_diff(&g, &g64));
    }
}
