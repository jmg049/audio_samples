//! librosa-compatible rhythm analysis: onset strength, onset detection, tempograms,
//! tempo estimation, dynamic-programming beat tracking and predominant local pulse.
//!
//! ## What
//!
//! This module provides the [`AudioRhythm`] trait and the free functions it is built
//! from. Together they reproduce librosa 1.0's rhythm pipeline:
//!
//! | librosa | here |
//! |---|---|
//! | `librosa.onset.onset_strength` / `onset_strength_multi` | [`AudioRhythm::onset_strength`] / [`AudioRhythm::onset_strength_multi`] ([`onset_strength_from_db`], [`onset_strength_multi_from_db`]) |
//! | `librosa.util.peak_pick` | [`peak_pick`] |
//! | `librosa.onset.onset_detect` | [`AudioRhythm::onset_detect`] ([`onset_detect_from_envelope`]) |
//! | `librosa.onset.onset_backtrack` | [`onset_backtrack`] |
//! | `librosa.tempo_frequencies` / `fourier_tempo_frequencies` | [`tempo_frequencies`] / [`fourier_tempo_frequencies`] |
//! | `librosa.feature.tempogram` / `fourier_tempogram` | [`AudioRhythm::tempogram`] / [`AudioRhythm::fourier_tempogram`] ([`tempogram_from_envelope`], [`fourier_tempogram_from_envelope`]) |
//! | `librosa.feature.tempo` | [`AudioRhythm::tempo`] ([`tempo_from_envelope`], [`tempo_from_tempogram`]) |
//! | `librosa.beat.beat_track` | [`AudioRhythm::beat_track`] ([`beat_track_from_envelope`], [`beat_track_with_bpm`]) |
//! | `librosa.beat.plp` | [`AudioRhythm::plp`] ([`plp_from_envelope`]) |
//! | `librosa.feature.tempogram_ratio` | [`AudioRhythm::tempogram_ratio`] ([`tempogram_ratio_from_tempogram`]) |
//! | `librosa.feature.hybrid_tempogram` | [`AudioRhythm::hybrid_tempogram`] ([`hybrid_tempogram_from_envelope`]) |
//! | `librosa.feature.metrogram` | [`metrogram`] |
//!
//! ## Why
//!
//! The older [`AudioOnsetDetection`](crate::AudioOnsetDetection) and
//! [`AudioBeatTracking`](crate::operations::AudioBeatTracking) traits use their own detection
//! functions, peak picker and a fixed-tempo beat grid; their results are not
//! comparable with librosa's. Code ported from Python, and results compared with
//! published librosa features, need librosa's exact algorithms. Every config here
//! defaults to librosa's defaults, so `X::default()` reproduces librosa's output.
//!
//! ## How
//!
//! Bring [`AudioRhythm`] into scope and call it on any [`AudioSamples`]. Frame-wise
//! outputs are `(channels, ..., n_frames)` arrays (one row block per channel); event
//! outputs ([`AudioRhythm::onset_detect`], [`AudioRhythm::beat_track`]) need mono
//! audio and return frame indices in a [`FrameEvents`], which converts them to
//! samples or seconds. The `*_from_*` free functions accept a precomputed envelope or
//! spectrogram, like librosa's `onset_envelope=` / `S=` arguments.
//!
//! ```rust
//! use audio_samples::operations::rhythm::{AudioRhythm, BeatTrackConfig, OnsetDetectConfig};
//! use audio_samples::{AudioSamples, sample_rate};
//! use ndarray::Array1;
//!
//! // A click every 0.5 s (120 BPM) for 4 s.
//! let sr = 22050;
//! let mut y = Array1::<f32>::zeros(4 * sr);
//! for k in 0..8 {
//!     for n in 0..200 {
//!         y[k * sr / 2 + n] = (-(n as f32) / 30.0).exp();
//!     }
//! }
//! let audio = AudioSamples::new_mono(y, sample_rate!(22050)).unwrap();
//!
//! let onsets = audio.onset_detect(&OnsetDetectConfig::default()).unwrap();
//! assert!(onsets.len() >= 7);
//!
//! let beats = audio.beat_track(&BeatTrackConfig::default()).unwrap();
//! assert!((beats.tempo - 120.0).abs() < 5.0);
//! ```
//!
//! ## Differences from librosa
//!
//! - The mel basis behind [`OnsetStrengthConfig`] is computed in `f64` (librosa's
//!   default basis is `float32`, a relative rounding of ~6e-8 in the weights).
//! - Event detectors take 1-D (mono) input; librosa only broadcasts them in its
//!   dense (`sparse=False`) mode.
//! - Inputs that crash or read out of bounds in librosa (empty envelopes,
//!   `onset_backtrack` with no events, tempos too fast for one frame per beat) give
//!   an empty result or a [`ParameterError`] here.

use std::num::NonZeroUsize;

use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};
use num_complex::Complex;
use realfft::RealFftPlanner;

use crate::operations::filterbanks::{Window, get_window};
// TRANSITIONAL(f64): `finite_audio_rows` / `melspectrogram_rows` on f64 rows.
use crate::operations::spectrum::{
    DbConfig, MelSpectrogramConfig, StftConfig, StftPadMode, finite_audio_rows, istft,
    melspectrogram_rows, power_to_db, stft,
};
use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, ChannelRequirement, LayoutError,
    ParameterError, ProcessingError, StandardSample,
};

/// A log-density over tempo (BPM) used as a tempo prior (librosa passes
/// `prior.logpdf` of a `scipy.stats` distribution).
///
/// Wraps a plain function so configs stay `Clone` and comparable (two priors are
/// equal when they wrap the same function).
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::rhythm::TempoLogPrior;
///
/// let prior = TempoLogPrior(|bpm| -0.5 * ((bpm - 100.0) / 10.0).powi(2));
/// assert_eq!((prior.0)(100.0), 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TempoLogPrior(pub fn(f64) -> f64);

impl PartialEq for TempoLogPrior {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::fn_addr_eq(self.0, other.0)
    }
}

fn param_err(parameter: &str, reason: impl ToString) -> AudioSampleError {
    AudioSampleError::Parameter(ParameterError::invalid_value(parameter, reason))
}

fn fft_err(e: impl ToString) -> AudioSampleError {
    AudioSampleError::Processing(ProcessingError::algorithm_failure("rhythm", e.to_string()))
}

fn require_positive(name: &str, v: f64) -> AudioSampleResult<()> {
    if v.is_finite() && v > 0.0 {
        Ok(())
    } else {
        Err(AudioSampleError::Parameter(ParameterError::out_of_range(
            name,
            v,
            "> 0",
            "finite",
            format!("{name} must be finite and strictly positive"),
        )))
    }
}

// =============================================================================
// numpy-compatible reductions
// =============================================================================

/// numpy's pairwise summation of a contiguous array (`numpy.sum` / `numpy.mean`
/// along the last axis), reproduced so that sums round exactly like numpy's.
fn pairwise_sum(a: &[f64]) -> f64 {
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
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

fn sequential_sum(a: &[f64]) -> f64 {
    a.iter().fold(0.0, |acc, &v| acc + v)
}

/// `numpy.max` (NaN propagates). `a` must be non-empty.
fn np_max(a: &[f64]) -> f64 {
    let mut m = a[0];
    for &v in &a[1..] {
        if m.is_nan() {
            return m;
        }
        if v > m || v.is_nan() {
            m = v;
        }
    }
    m
}

/// `numpy.median` of the values in `buf` (sorted in place). `buf` must be non-empty.
fn median_in_place(buf: &mut [f64]) -> f64 {
    buf.sort_unstable_by(f64::total_cmp);
    let n = buf.len();
    if n % 2 == 1 {
        buf[n / 2]
    } else {
        (buf[n / 2 - 1] + buf[n / 2]) / 2.0
    }
}

/// Python's float floor division `a // b` for finite `a >= 0`, `b > 0`.
fn py_floordiv(a: f64, b: f64) -> f64 {
    let m = a % b;
    let div = (a - m) / b;
    let mut f = div.floor();
    if div - f > 0.5 {
        f += 1.0;
    }
    f
}

/// Smallest 5-smooth integer `>= n` (scipy `next_fast_len(n, real=True)`).
fn next_fast_len(n: usize) -> usize {
    let mut m = n.max(1);
    loop {
        let mut r = m;
        for p in [2, 3, 5] {
            while r.is_multiple_of(p) {
                r /= p;
            }
        }
        if r == 1 {
            return m;
        }
        m += 1;
    }
}

/// Divides `x` by its `norm` (librosa `util.normalize` with `threshold=None`,
/// `fill=None`: vectors whose norm is below the smallest positive `f64` are left as is).
fn normalize_in_place(x: &mut [f64], norm: Option<f64>) {
    let Some(p) = norm else { return };
    if x.is_empty() {
        return;
    }
    let length = if p == f64::INFINITY {
        x.iter().fold(f64::NEG_INFINITY, |m, v| m.max(v.abs()))
    } else if p == f64::NEG_INFINITY {
        x.iter().fold(f64::INFINITY, |m, v| m.min(v.abs()))
    } else if p == 0.0 {
        x.iter().filter(|v| v.abs() > 0.0).count() as f64
    } else {
        let s = sequential_sum(
            &x.iter()
                .map(|v| if p == 2.0 { v * v } else { v.abs().powf(p) })
                .collect::<Vec<_>>(),
        );
        s.powf(1.0 / p)
    };
    let length = if length < f64::MIN_POSITIVE {
        1.0
    } else {
        length
    };
    for v in x.iter_mut() {
        *v /= length;
    }
}

fn validate_norm(norm: Option<f64>) -> AudioSampleResult<()> {
    match norm {
        Some(p) if p.is_nan() || (p.is_finite() && p < 0.0) => Err(param_err(
            "norm",
            format!("unsupported norm {p}; use None, 0, a positive number or +/-infinity"),
        )),
        _ => Ok(()),
    }
}

// =============================================================================
// Aggregation
// =============================================================================

/// How values are combined across frequency bins (onset strength) or frames (tempo),
/// librosa's `aggregate` argument.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::rhythm::Aggregate;
///
/// assert_eq!(Aggregate::default(), Aggregate::Mean);
/// let first: fn(&[f64]) -> f64 = |v| v[0];
/// assert_eq!(Aggregate::Custom(first), Aggregate::Custom(first));
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub enum Aggregate {
    /// Arithmetic mean (`numpy.mean`). librosa's default for onset strength and tempo.
    #[default]
    Mean,
    /// Median (`numpy.median`). librosa's beat tracker and PLP use it for the onset
    /// envelope.
    Median,
    /// Maximum (`numpy.max`).
    Max,
    /// Minimum (`numpy.min`).
    Min,
    /// Sum (`numpy.sum`). librosa's default for [`metrogram`].
    Sum,
    /// A user function of the (non-empty) values.
    Custom(fn(&[f64]) -> f64),
}

impl PartialEq for Aggregate {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Mean, Self::Mean)
            | (Self::Median, Self::Median)
            | (Self::Max, Self::Max)
            | (Self::Min, Self::Min)
            | (Self::Sum, Self::Sum) => true,
            (Self::Custom(a), Self::Custom(b)) => std::ptr::fn_addr_eq(*a, *b),
            _ => false,
        }
    }
}

impl Aggregate {
    /// Reduces non-empty `values`; `scratch` is reused for the median. `pairwise`
    /// selects numpy's summation order for a contiguous axis (otherwise sequential).
    fn reduce(self, values: &[f64], scratch: &mut Vec<f64>, pairwise: bool) -> f64 {
        match self {
            Self::Mean => {
                let s = if pairwise {
                    pairwise_sum(values)
                } else {
                    sequential_sum(values)
                };
                s / values.len() as f64
            }
            Self::Sum => {
                if pairwise {
                    pairwise_sum(values)
                } else {
                    sequential_sum(values)
                }
            }
            Self::Median => {
                scratch.clear();
                scratch.extend_from_slice(values);
                median_in_place(scratch)
            }
            Self::Max => np_max(values),
            Self::Min => {
                let mut m = values[0];
                for &v in &values[1..] {
                    if m.is_nan() {
                        break;
                    }
                    if v < m || v.is_nan() {
                        m = v;
                    }
                }
                m
            }
            Self::Custom(f) => f(values),
        }
    }
}

// =============================================================================
// Onset strength
// =============================================================================

/// Configuration of the onset strength envelope, mirroring
/// `librosa.onset.onset_strength` / `onset_strength_multi`.
///
/// ## Purpose
///
/// The onset strength at frame `t` is the aggregated positive increase of the
/// log-power mel spectrogram from frame `t - lag` (spectral flux on a dB scale).
///
/// ## Intended usage
///
/// Pass to [`AudioRhythm::onset_strength`] and embed in the other rhythm configs.
/// `Default` reproduces librosa: 128-band Slaney mel power spectrogram (`n_fft = 2048`,
/// `hop_length = 512`, centred, zero padding), `power_to_db` with `ref = 1`,
/// `amin = 1e-10`, `top_db = 80`, `lag = 1`, `max_size = 1`, no detrending, centred
/// output and mean aggregation.
///
/// ## Invariants
///
/// Checked by [`OnsetStrengthConfig::validate`]: the mel and dB configs are valid.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::rhythm::{Aggregate, OnsetStrengthConfig};
///
/// let cfg = OnsetStrengthConfig::new().with_lag(nzu!(2)).with_aggregate(Some(Aggregate::Median));
/// assert_eq!(cfg.hop_length().unwrap().get(), 512);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct OnsetStrengthConfig {
    /// Mel spectrogram parameters (librosa's `n_fft`, `hop_length` and mel `**kwargs`).
    /// The STFT `n_fft` and hop also set the centring shift of the envelope.
    pub mel: MelSpectrogramConfig,
    /// dB conversion applied to the mel spectrogram. Default:
    /// [`DbConfig::librosa_power`], librosa's 80 dB floor below the maximum. The
    /// floor limits the flux that near-silent passages contribute to the envelope;
    /// set `top_db` to `None` to remove it.
    pub db: DbConfig,
    /// Time lag, in frames, of the difference. Larger lags respond to slower rises.
    /// librosa default: 1.
    pub lag: NonZeroUsize,
    /// Size of the maximum filter applied across frequency to the reference
    /// spectrogram (SuperFlux vibrato suppression). 1 disables it. librosa default: 1.
    pub max_size: NonZeroUsize,
    /// If `true`, the envelope is high-pass filtered by
    /// `y[n] = x[n] - x[n-1] + 0.99 y[n-1]` to remove its DC component.
    /// librosa default: `false`.
    pub detrend: bool,
    /// If `true`, the envelope is shifted by `n_fft / (2 * hop)` frames to compensate
    /// for centred STFT frames and trimmed to the spectrogram length. librosa default:
    /// `true`.
    pub center: bool,
    /// How bins are combined into one value per frame (or per sub-band in
    /// [`AudioRhythm::onset_strength_multi`]). `None` keeps every bin (librosa
    /// `aggregate=False`, only valid for the multi-band form). librosa default:
    /// `Some(Mean)`.
    pub aggregate: Option<Aggregate>,
}

impl Default for OnsetStrengthConfig {
    fn default() -> Self {
        Self {
            mel: MelSpectrogramConfig::default(),
            // The 80 dB floor of librosa's onset_strength is kept: it limits the
            // flux contributed by near-silent passages (see the field docs).
            db: DbConfig::librosa_power(),
            lag: crate::nzu!(1),
            max_size: crate::nzu!(1),
            detrend: false,
            center: true,
            aggregate: Some(Aggregate::Mean),
        }
    }
}

impl OnsetStrengthConfig {
    /// Returns the librosa-default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the mel spectrogram parameters.
    #[must_use]
    pub fn with_mel(mut self, mel: MelSpectrogramConfig) -> Self {
        self.mel = mel;
        self
    }

    /// Sets the dB conversion.
    #[must_use]
    pub fn with_db(mut self, db: DbConfig) -> Self {
        self.db = db;
        self
    }

    /// Sets the lag in frames.
    #[must_use]
    pub fn with_lag(mut self, lag: NonZeroUsize) -> Self {
        self.lag = lag;
        self
    }

    /// Sets the frequency maximum-filter size.
    #[must_use]
    pub fn with_max_size(mut self, max_size: NonZeroUsize) -> Self {
        self.max_size = max_size;
        self
    }

    /// Sets detrending.
    #[must_use]
    pub fn with_detrend(mut self, detrend: bool) -> Self {
        self.detrend = detrend;
        self
    }

    /// Sets centring.
    #[must_use]
    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Sets the aggregation.
    #[must_use]
    pub fn with_aggregate(mut self, aggregate: Option<Aggregate>) -> Self {
        self.aggregate = aggregate;
        self
    }

    /// The hop length of the envelope (the mel STFT hop).
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the STFT hop is unset and `win_length / 4` is 0.
    pub fn hop_length(&self) -> AudioSampleResult<NonZeroUsize> {
        self.mel.stft.effective_hop_length()
    }

    /// Checks the configuration for a given sample rate.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the mel spectrogram or dB configuration is invalid.
    pub fn validate(&self, sample_rate: f64) -> AudioSampleResult<()> {
        self.mel.validate(sample_rate)?;
        self.db.validate()
    }
}

/// Sub-band boundaries normalised as librosa's `index_to_slice(pad=False)`.
fn band_ranges(bounds: &[usize], n_bins: usize) -> AudioSampleResult<Vec<(usize, usize)>> {
    let mut v: Vec<usize> = bounds.iter().copied().filter(|&b| b <= n_bins).collect();
    v.sort_unstable();
    v.dedup();
    if v.len() < 2 {
        return Err(param_err(
            "channels",
            format!("need at least two distinct sub-band boundaries in [0, {n_bins}]"),
        ));
    }
    Ok(v.windows(2).map(|w| (w[0], w[1])).collect())
}

/// `scipy.ndimage.maximum_filter1d(s, size, axis=0)` with the default reflect boundary.
fn maximum_filter_freq(s: ArrayView2<'_, f64>, size: usize) -> Array2<f64> {
    let (n_bins, n_t) = s.dim();
    let n = n_bins as isize;
    let reflect = |i: isize| -> usize {
        let period = 2 * n;
        let m = i.rem_euclid(period);
        (if m < n { m } else { period - 1 - m }) as usize
    };
    let half = (size / 2) as isize;
    // A window of 2 * n_bins or more covers a whole reflection period, so it
    // sees every value of the column: scanning one period gives the same
    // maximum and keeps a huge `max_size` from looping for hours.
    let span = size.min(2 * n_bins) as isize;
    Array2::from_shape_fn((n_bins, n_t), |(i, t)| {
        let start = i as isize - half;
        let mut m = s[[reflect(start), t]];
        for j in 1..span {
            let v = s[[reflect(start + j), t]];
            if v > m {
                m = v;
            }
        }
        m
    })
}

/// Computes multi-band onset strength from a dB spectrogram, equivalent to
/// `librosa.onset.onset_strength_multi(S=s_db, ref=reference, channels=...)`.
///
/// # Arguments
///
/// * `s_db` - `(channels, n_bins, n_frames)` log-power spectrogram (e.g.
///   [`power_to_db`] of a mel spectrogram). It is used as given.
/// * `reference` - Optional reference spectrogram of the same shape that frame
///   `t - lag` is compared against (librosa `ref`). `None` uses `s_db`, maximum-filtered
///   across frequency when `config.max_size > 1`.
/// * `config` - `lag`, `max_size`, `detrend`, `center`, `aggregate`, and the STFT
///   `n_fft` / hop that set the centring shift. The mel and dB fields are not used.
/// * `channels` - Sub-band boundaries in bins (librosa `channels` as integers):
///   values above `n_bins` are dropped, the rest sorted and de-duplicated, and each
///   pair of consecutive boundaries is one band. `None` means one band of all bins.
///
/// # Returns
///
/// `(channels, n_bands, n_frames_out)` where `n_frames_out = n_frames` when centred
/// and `n_frames` otherwise too (the lag is compensated by zero padding at the start).
/// With `aggregate = None` the middle axis has one row per bin and `channels` is
/// ignored.
///
/// # Errors
///
/// [`ParameterError`] if the input is empty, `reference` has a different shape,
/// fewer than two valid boundaries are given, or the hop is invalid.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::rhythm::{OnsetStrengthConfig, onset_strength_multi_from_db};
/// use ndarray::Array3;
///
/// let s = Array3::from_shape_fn((1, 8, 10), |(_, b, t)| if t >= 5 { b as f64 } else { 0.0 });
/// let env = onset_strength_multi_from_db(s.view(), None, &OnsetStrengthConfig::default(), Some(&[0, 4, 8]))
///     .unwrap();
/// assert_eq!(env.dim(), (1, 2, 10));
/// // The rise at frame 5 (mean of bins 4..8) lands at frame 5 - lag + (lag + 2048 / 1024) = 7.
/// assert_eq!(env[[0, 1, 7]], 5.5);
/// ```
pub fn onset_strength_multi_from_db(
    s_db: ArrayView3<'_, f64>,
    reference: Option<ArrayView3<'_, f64>>,
    config: &OnsetStrengthConfig,
    channels: Option<&[usize]>,
) -> AudioSampleResult<Array3<f64>> {
    let (n_ch, n_bins, n_t) = s_db.dim();
    if n_ch == 0 || n_bins == 0 || n_t == 0 {
        return Err(param_err("S", "the spectrogram must not be empty"));
    }
    if let Some(r) = &reference {
        if r.dim() != s_db.dim() {
            return Err(param_err(
                "reference",
                format!(
                    "reference shape {:?} must match the spectrogram shape {:?}",
                    r.dim(),
                    s_db.dim()
                ),
            ));
        }
    }
    let lag = config.lag.get();
    let max_size = config.max_size.get();
    let n_fft = config.mel.stft.n_fft.get();
    let hop = config.hop_length()?.get();
    let bands = match channels {
        None => vec![(0, n_bins)],
        Some(b) => band_ranges(b, n_bins)?,
    };
    let n_diff = n_t.saturating_sub(lag);
    let pad = lag + if config.center { n_fft / (2 * hop) } else { 0 };
    let full_len = n_diff + pad;
    let out_len = if config.center {
        full_len.min(n_t)
    } else {
        full_len
    };
    let n_rows = if config.aggregate.is_some() {
        bands.len()
    } else {
        n_bins
    };
    let mut out = Array3::<f64>::zeros((n_ch, n_rows, out_len));
    let mut diff = vec![0.0; n_bins];
    let mut scratch = Vec::with_capacity(n_bins);
    for c in 0..n_ch {
        let sc = s_db.index_axis(Axis(0), c);
        let filtered: Array2<f64>;
        let rc = match &reference {
            Some(r) => r.index_axis(Axis(0), c),
            None if max_size == 1 => sc,
            None => {
                filtered = maximum_filter_freq(sc, max_size);
                filtered.view()
            }
        };
        let mut oc = out.index_axis_mut(Axis(0), c);
        for t in 0..n_diff {
            let o = t + pad;
            if o >= out_len {
                break;
            }
            for (b, d) in diff.iter_mut().enumerate() {
                let v = sc[[b, t + lag]] - rc[[b, t]];
                // numpy.maximum(0, v): NaN propagates.
                *d = if v < 0.0 { 0.0 } else { v };
            }
            match config.aggregate {
                Some(agg) => {
                    for (i, &(lo, hi)) in bands.iter().enumerate() {
                        oc[[i, o]] = agg.reduce(&diff[lo..hi], &mut scratch, false);
                    }
                }
                None => {
                    for (b, &d) in diff.iter().enumerate() {
                        oc[[b, o]] = d;
                    }
                }
            }
        }
        if config.detrend {
            // scipy.signal.lfilter([1, -1], [1, -0.99]) (transposed direct form II).
            for mut row in oc.outer_iter_mut() {
                let mut z = 0.0;
                for v in row.iter_mut() {
                    let x = *v;
                    let y = z + x;
                    z = (-x) - (-0.99 * y);
                    *v = y;
                }
            }
        }
    }
    Ok(out)
}

/// Computes the onset strength envelope from a dB spectrogram, equivalent to
/// `librosa.onset.onset_strength(S=s_db)`.
///
/// Same as [`onset_strength_multi_from_db`] with one band covering all bins.
///
/// # Arguments
///
/// * `s_db` - `(channels, n_bins, n_frames)` log-power spectrogram.
/// * `config` - See [`onset_strength_multi_from_db`]; `aggregate` must be `Some`.
///
/// # Returns
///
/// A `(channels, n_frames)` envelope.
///
/// # Errors
///
/// [`ParameterError`] if `config.aggregate` is `None` or the input is invalid.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::rhythm::{OnsetStrengthConfig, onset_strength_from_db};
/// use ndarray::Array3;
///
/// let s = Array3::from_shape_fn((2, 4, 6), |(c, _, t)| (c * t) as f64);
/// let env = onset_strength_from_db(s.view(), &OnsetStrengthConfig::default()).unwrap();
/// assert_eq!(env.dim(), (2, 6));
/// assert_eq!(env[[1, 5]], 1.0);
/// ```
pub fn onset_strength_from_db(
    s_db: ArrayView3<'_, f64>,
    config: &OnsetStrengthConfig,
) -> AudioSampleResult<Array2<f64>> {
    if config.aggregate.is_none() {
        return Err(param_err(
            "aggregate",
            "aggregate cannot be None when computing full-spectrum onset strength",
        ));
    }
    let multi = onset_strength_multi_from_db(s_db, None, config, None)?;
    Ok(multi.index_axis_move(Axis(1), 0))
}

// =============================================================================
// Peak picking, onset detection and backtracking
// =============================================================================

/// Peak-selection strategy of [`peak_pick`] (librosa `method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PeakPickMethod {
    /// Scan left to right, skipping `wait` frames after each peak (librosa `"greedy"`,
    /// the default).
    #[default]
    Greedy,
    /// Dynamic programming maximising the number of peaks subject to `wait`
    /// (librosa `"dp_count"`).
    DpCount,
    /// Dynamic programming maximising the summed peak values subject to `wait`
    /// (librosa `"dp_value"`).
    DpValue,
}

/// Parameters of librosa's peak picker ([`peak_pick`], `librosa.util.peak_pick`).
///
/// ## Purpose
///
/// A sample `n` is a peak if it equals the maximum of `x[n - pre_max .. n + post_max]`,
/// is at least `delta` above the mean of `x[n - pre_avg .. n + post_avg]`, and (for
/// the greedy method) is more than `wait` samples after the previous peak.
///
/// ## Intended usage
///
/// Use [`PeakPickConfig::for_onsets`] to get the parameters `librosa.onset.onset_detect`
/// derives from the frame rate (30 ms, 0 ms, 100 ms, 100 ms windows, 30 ms wait,
/// `delta = 0.07`); `Default` is that set at 22050 Hz with a hop of 512. Unrelated to
/// [`PeakPickingConfig`](crate::operations::types::PeakPickingConfig), which drives the
/// older non-librosa peak picker.
///
/// ## Invariants
///
/// Checked by [`PeakPickConfig::validate`]: `post_max >= 1`, `post_avg >= 1`, `delta`
/// finite and non-negative.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::rhythm::PeakPickConfig;
///
/// let p = PeakPickConfig::for_onsets(22050.0, nzu!(512));
/// assert_eq!((p.pre_max, p.post_max, p.pre_avg, p.post_avg, p.wait), (1, 1, 4, 5, 1));
/// assert_eq!(p, PeakPickConfig::default());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PeakPickConfig {
    /// Samples before `n` included in the maximum window.
    pub pre_max: usize,
    /// Samples after `n` (exclusive end offset, so `1` means only `n` itself) in the
    /// maximum window. At least 1.
    pub post_max: usize,
    /// Samples before `n` included in the mean window.
    pub pre_avg: usize,
    /// Exclusive end offset of the mean window after `n`. At least 1.
    pub post_avg: usize,
    /// Required margin above the local mean. Non-negative.
    pub delta: f64,
    /// Minimum number of samples between peaks.
    pub wait: usize,
    /// Selection strategy.
    pub method: PeakPickMethod,
}

impl Default for PeakPickConfig {
    fn default() -> Self {
        Self::for_onsets(22050.0, crate::nzu!(512))
    }
}

impl PeakPickConfig {
    /// Returns the default configuration (same as [`Default`]): the
    /// [`for_onsets`](Self::for_onsets) parameters at 22050 Hz with a hop of 512.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The parameters `librosa.onset.onset_detect` uses for a frame rate of
    /// `sample_rate / hop_length`: `pre_max = 0.03 sr // hop`, `post_max = 1`,
    /// `pre_avg = 0.1 sr // hop`, `post_avg = 0.1 sr // hop + 1`, `wait = 0.03 sr // hop`,
    /// `delta = 0.07` (floor divisions on floats, as in Python).
    #[must_use]
    pub fn for_onsets(sample_rate: f64, hop_length: NonZeroUsize) -> Self {
        let h = hop_length.get() as f64;
        let fd = |secs: f64| py_floordiv(secs * sample_rate, h).max(0.0) as usize;
        Self {
            pre_max: fd(0.03),
            post_max: fd(0.0) + 1,
            pre_avg: fd(0.10),
            post_avg: fd(0.10) + 1,
            delta: 0.07,
            wait: fd(0.03),
            method: PeakPickMethod::Greedy,
        }
    }

    /// Sets `delta`.
    #[must_use]
    pub fn with_delta(mut self, delta: f64) -> Self {
        self.delta = delta;
        self
    }

    /// Sets `wait`.
    #[must_use]
    pub fn with_wait(mut self, wait: usize) -> Self {
        self.wait = wait;
        self
    }

    /// Sets the selection method.
    #[must_use]
    pub fn with_method(mut self, method: PeakPickMethod) -> Self {
        self.method = method;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `post_max` or `post_avg` is 0, or `delta` is negative or
    /// not finite.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if self.post_max == 0 {
            return Err(param_err("post_max", "post_max must be positive"));
        }
        if self.post_avg == 0 {
            return Err(param_err("post_avg", "post_avg must be positive"));
        }
        if !(self.delta.is_finite() && self.delta >= 0.0) {
            return Err(param_err("delta", "delta must be finite and non-negative"));
        }
        Ok(())
    }
}

/// Picks peaks in a signal, equivalent to `librosa.util.peak_pick` (sparse output).
///
/// Sample 0 is special-cased as in librosa: it only has to be the maximum of
/// `x[..post_max]` and at least `delta` above the mean of `x[..post_avg]`. Means are
/// accumulated left to right, matching librosa's numba kernels bit for bit.
///
/// # Arguments
///
/// * `x` - The signal (e.g. a normalised onset envelope). Must be finite.
/// * `config` - Window sizes, threshold, wait and method.
///
/// # Returns
///
/// Ascending indices of the peaks (empty for empty input).
///
/// # Errors
///
/// [`ParameterError`] if the configuration is invalid or `x` has non-finite values.
/// Window sizes and `wait` larger than the signal (up to `usize::MAX`) are
/// valid and simply cover the whole signal.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::rhythm::{PeakPickConfig, peak_pick};
///
/// let x = [0.0, 1.0, 0.0, 0.0, 0.9, 0.0, 0.2, 0.0];
/// let cfg = PeakPickConfig { pre_max: 1, post_max: 2, pre_avg: 2, post_avg: 3, delta: 0.1, wait: 0,
///     ..PeakPickConfig::default() };
/// // 0.2 at index 6 is a local maximum but not 0.1 above its local mean (0.275).
/// assert_eq!(peak_pick(&x, &cfg).unwrap(), vec![1, 4]);
/// ```
///
/// # References
///
/// S. Böck, F. Krebs and M. Schedl, "Evaluating the online capabilities of onset
/// detection methods," ISMIR 2012; librosa `util.peak_pick`.
pub fn peak_pick(x: &[f64], config: &PeakPickConfig) -> AudioSampleResult<Vec<usize>> {
    config.validate()?;
    if x.iter().any(|v| !v.is_finite()) {
        return Err(param_err("x", "peak_pick input must be finite"));
    }
    let n = x.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let PeakPickConfig {
        pre_max,
        post_max,
        pre_avg,
        post_avg,
        delta,
        wait,
        method,
    } = *config;
    let mut peaks = Vec::new();
    match method {
        PeakPickMethod::Greedy => {
            let mean = |lo: usize, hi: usize| sequential_sum(&x[lo..hi]) / (hi - lo) as f64;
            let first =
                x[0] >= np_max(&x[..post_max.min(n)]) && x[0] >= mean(0, post_avg.min(n)) + delta;
            let mut i = if first {
                peaks.push(0);
                wait.saturating_add(1)
            } else {
                1
            };
            while i < n {
                let maxn = np_max(&x[i.saturating_sub(pre_max)..i.saturating_add(post_max).min(n)]);
                if x[i] != maxn {
                    i += 1;
                    continue;
                }
                let avgn = mean(i.saturating_sub(pre_avg), i.saturating_add(post_avg).min(n));
                // Inputs are finite, so this is librosa's `not x[n] >= avg + delta`.
                if x[i] < avgn + delta {
                    i += 1;
                    continue;
                }
                peaks.push(i);
                i = i.saturating_add(wait).saturating_add(1);
            }
        }
        PeakPickMethod::DpCount | PeakPickMethod::DpValue => {
            let count = method == PeakPickMethod::DpCount;
            let mut values = vec![0.0; n + 1];
            let mut pointers = vec![0isize; n + 1];
            let mut taken = vec![false; n + 1];
            let mut cumulate = Vec::with_capacity(n);
            let mut acc = 0.0;
            for &v in x {
                acc += v;
                cumulate.push(acc);
            }
            pointers[n] = -1;
            for i in (0..n).rev() {
                values[i] = values[i + 1];
                pointers[i] = (i + 1) as isize;
                let maxn = np_max(&x[i.saturating_sub(pre_max)..i.saturating_add(post_max).min(n)]);
                if x[i] < maxn {
                    continue;
                }
                let idx_prev = i.saturating_sub(pre_avg);
                let idx_post = i.saturating_add(post_avg).min(n);
                let avgn = if idx_prev == 0 {
                    cumulate[idx_post - 1] / idx_post as f64
                } else {
                    (cumulate[idx_post - 1] - cumulate[idx_prev - 1]) / (idx_post - idx_prev) as f64
                };
                let v = if count { 1.0 } else { x[i] };
                let next = n.min(i.saturating_add(wait).saturating_add(1));
                if x[i] >= avgn + delta && values[next] + v > values[i + 1] {
                    values[i] = values[next] + v;
                    pointers[i] = next as isize;
                    taken[i] = true;
                }
            }
            let mut i = 0usize;
            while pointers[i] >= 0 {
                if taken[i] {
                    peaks.push(i);
                }
                i = pointers[i] as usize;
            }
        }
    }
    Ok(peaks)
}

/// Moves each event back to the nearest preceding local minimum of `energy`,
/// equivalent to `librosa.onset.onset_backtrack`.
///
/// The candidate minima are frame 0 and every `i` with
/// `energy[i] <= energy[i - 1]` and `energy[i] < energy[i + 1]`; each event maps to the
/// largest candidate `<=` the event.
///
/// # Arguments
///
/// * `events` - Event frame indices (e.g. from [`peak_pick`]).
/// * `energy` - The function to search for minima (an onset envelope or RMS energy).
///
/// # Returns
///
/// One backtracked frame per event, in the order of `events`.
///
/// # Differences from librosa
///
/// An empty `events` list returns an empty vector (librosa raises).
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::rhythm::onset_backtrack;
///
/// let energy = [3.0, 2.0, 1.0, 4.0, 6.0, 2.0, 5.0, 9.0];
/// assert_eq!(onset_backtrack(&[4, 7, 0], &energy), vec![2, 5, 0]);
/// ```
#[must_use]
pub fn onset_backtrack(events: &[usize], energy: &[f64]) -> Vec<usize> {
    let mut minima = vec![0usize];
    for i in 1..energy.len().saturating_sub(1) {
        if energy[i] <= energy[i - 1] && energy[i] < energy[i + 1] && i != 0 {
            minima.push(i);
        }
    }
    events
        .iter()
        .map(|&e| {
            let k = minima.partition_point(|&m| m <= e);
            minima[k.saturating_sub(1)]
        })
        .collect()
}

/// Frame indices of detected events together with the frame geometry needed to
/// express them in samples or seconds (librosa's `units` argument).
///
/// ## Purpose
///
/// Returned by [`AudioRhythm::onset_detect`] and [`AudioRhythm::beat_track`].
///
/// ## Invariants
///
/// `frames` is sorted ascending.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::rhythm::FrameEvents;
///
/// let ev = FrameEvents { frames: vec![0, 43], sample_rate: 22050.0, hop_length: nzu!(512) };
/// assert_eq!(ev.samples(), vec![0, 22016]);
/// assert!((ev.times()[1] - 0.998_458).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FrameEvents {
    /// Event positions in frames (librosa `units="frames"`).
    pub frames: Vec<usize>,
    /// Sampling rate in Hz of the analysed signal.
    pub sample_rate: f64,
    /// Samples between frames.
    pub hop_length: NonZeroUsize,
}

impl FrameEvents {
    /// Event positions in samples, `frame * hop_length` (librosa `units="samples"`).
    #[must_use]
    pub fn samples(&self) -> Vec<usize> {
        self.frames
            .iter()
            .map(|&f| f * self.hop_length.get())
            .collect()
    }

    /// Event positions in seconds, `frame * hop_length / sample_rate`
    /// (librosa `units="time"`).
    #[must_use]
    pub fn times(&self) -> Vec<f64> {
        self.samples()
            .into_iter()
            .map(|s| s as f64 / self.sample_rate)
            .collect()
    }

    /// Number of events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// `true` if there are no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

/// Configuration of onset detection, mirroring `librosa.onset.onset_detect`.
///
/// ## Purpose
///
/// Combines the onset envelope, its normalisation, the peak picker and optional
/// backtracking.
///
/// ## Intended usage
///
/// Pass to [`AudioRhythm::onset_detect`] or [`onset_detect_from_envelope`].
/// `Default` reproduces librosa: default onset strength, normalisation to `[0, 1]`,
/// peak-picking parameters derived from the frame rate, no backtracking.
///
/// ## Invariants
///
/// Checked by [`OnsetDetectConfig::validate`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::rhythm::OnsetDetectConfig;
///
/// let cfg = OnsetDetectConfig::new().with_backtrack(true);
/// assert!(cfg.normalize && cfg.peak_pick.is_none());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct OnsetDetectConfig {
    /// Onset envelope parameters (also define the hop length).
    pub onset: OnsetStrengthConfig,
    /// Peak-picking parameters. `None` derives librosa's defaults from the frame rate
    /// with [`PeakPickConfig::for_onsets`].
    pub peak_pick: Option<PeakPickConfig>,
    /// If `true` (librosa's default), the envelope is shifted and scaled to `[0, 1]`
    /// before peak picking, so `delta` is relative to its range.
    pub normalize: bool,
    /// If `true`, onsets are moved back to the preceding local minimum of the
    /// (normalised) envelope with [`onset_backtrack`]. librosa default: `false`.
    pub backtrack: bool,
}

impl Default for OnsetDetectConfig {
    fn default() -> Self {
        Self {
            onset: OnsetStrengthConfig::default(),
            peak_pick: None,
            normalize: true,
            backtrack: false,
        }
    }
}

impl OnsetDetectConfig {
    /// Returns the librosa-default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the onset envelope parameters.
    #[must_use]
    pub fn with_onset(mut self, onset: OnsetStrengthConfig) -> Self {
        self.onset = onset;
        self
    }

    /// Sets explicit peak-picking parameters.
    #[must_use]
    pub fn with_peak_pick(mut self, peak_pick: Option<PeakPickConfig>) -> Self {
        self.peak_pick = peak_pick;
        self
    }

    /// Sets envelope normalisation.
    #[must_use]
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Sets backtracking.
    #[must_use]
    pub fn with_backtrack(mut self, backtrack: bool) -> Self {
        self.backtrack = backtrack;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the onset or peak-picking configuration is invalid.
    pub fn validate(&self, sample_rate: f64) -> AudioSampleResult<()> {
        self.onset.validate(sample_rate)?;
        if let Some(p) = &self.peak_pick {
            p.validate()?;
        }
        Ok(())
    }
}

/// Detects onsets in a precomputed onset envelope, equivalent to
/// `librosa.onset.onset_detect(onset_envelope=envelope, energy=energy)`.
///
/// # Arguments
///
/// * `envelope` - Onset strength envelope (one channel).
/// * `sample_rate` - Sampling rate of the analysed audio in Hz (for the default
///   peak-picking parameters).
/// * `hop_length` - Hop of the envelope in samples.
/// * `config` - Normalisation, peak picking and backtracking (`config.onset` is not
///   used).
/// * `energy` - Function to backtrack on; `None` uses the (normalised) envelope.
///
/// # Returns
///
/// The onset frames. Empty if the envelope is all zeros or has non-finite values.
///
/// # Errors
///
/// [`ParameterError`] if the peak-picking parameters are invalid or `sample_rate` is
/// not positive.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::rhythm::{OnsetDetectConfig, onset_detect_from_envelope};
///
/// let mut env = vec![0.0; 40];
/// env[10] = 1.0;
/// env[25] = 0.8;
/// let onsets = onset_detect_from_envelope(&env, 22050.0, nzu!(512), &OnsetDetectConfig::default(), None)
///     .unwrap();
/// assert_eq!(onsets, vec![10, 25]);
/// ```
pub fn onset_detect_from_envelope(
    envelope: &[f64],
    sample_rate: f64,
    hop_length: NonZeroUsize,
    config: &OnsetDetectConfig,
    energy: Option<&[f64]>,
) -> AudioSampleResult<Vec<usize>> {
    require_positive("sample_rate", sample_rate)?;
    if let Some(p) = &config.peak_pick {
        p.validate()?;
    }
    if envelope.is_empty() {
        return Ok(Vec::new());
    }
    let mut env = envelope.to_vec();
    if config.normalize {
        let min = env.iter().copied().fold(f64::INFINITY, f64::min);
        for v in &mut env {
            *v -= min;
        }
        let max = np_max(&env) + f64::MIN_POSITIVE;
        for v in &mut env {
            *v /= max;
        }
    }
    if !env.iter().any(|&v| v != 0.0) || env.iter().any(|v| !v.is_finite()) {
        return Ok(Vec::new());
    }
    let pp = config
        .peak_pick
        .clone()
        .unwrap_or_else(|| PeakPickConfig::for_onsets(sample_rate, hop_length));
    let onsets = peak_pick(&env, &pp)?;
    if config.backtrack {
        Ok(onset_backtrack(&onsets, energy.unwrap_or(&env)))
    } else {
        Ok(onsets)
    }
}

// =============================================================================
// Tempo frequencies and tempograms
// =============================================================================

/// Tempo in BPM of each lag bin of an autocorrelation tempogram, equivalent to
/// `librosa.tempo_frequencies`.
///
/// Bin 0 (zero lag) is `+inf`; bin `k` is `60 * sample_rate / (hop_length * k)`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::rhythm::tempo_frequencies;
///
/// let f = tempo_frequencies(4, nzu!(512), 22050.0);
/// assert!(f[0].is_infinite());
/// assert!((f[1] - 2583.984375).abs() < 1e-9);
/// ```
#[must_use]
pub fn tempo_frequencies(n_bins: usize, hop_length: NonZeroUsize, sample_rate: f64) -> Vec<f64> {
    let num = 60.0 * sample_rate;
    let h = hop_length.get() as f64;
    (0..n_bins)
        .map(|k| {
            if k == 0 {
                f64::INFINITY
            } else {
                num / (h * k as f64)
            }
        })
        .collect()
}

/// Tempo in BPM of each bin of a Fourier tempogram, equivalent to
/// `librosa.fourier_tempo_frequencies`: `win_length / 2 + 1` values
/// `k * (60 * sample_rate / hop_length) / win_length`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::rhythm::fourier_tempo_frequencies;
///
/// let f = fourier_tempo_frequencies(22050.0, nzu!(512), nzu!(384));
/// assert_eq!(f.len(), 193);
/// assert!((f[192] - 1291.9921875).abs() < 1e-9);
/// ```
#[must_use]
pub fn fourier_tempo_frequencies(
    sample_rate: f64,
    hop_length: NonZeroUsize,
    win_length: NonZeroUsize,
) -> Vec<f64> {
    let rate = sample_rate * 60.0 / hop_length.get() as f64;
    let n = win_length.get();
    // numpy.fft.rfftfreq(n, d=1/rate)
    let val = 1.0 / (n as f64 * (1.0 / rate));
    (0..=n / 2).map(|k| k as f64 * val).collect()
}

/// Configuration of the autocorrelation tempogram, mirroring
/// `librosa.feature.tempogram`.
///
/// ## Purpose
///
/// The tempogram is the local autocorrelation of the onset envelope: column `t` is
/// the autocorrelation of a `win_length`-frame window around frame `t`, so bin `k`
/// measures periodicity at a lag of `k` frames (tempo [`tempo_frequencies`]).
///
/// ## Intended usage
///
/// Pass to [`AudioRhythm::tempogram`] or [`tempogram_from_envelope`]. `Default`
/// reproduces librosa: default onset strength, `win_length = 384` frames (~8.9 s at
/// 22050 Hz / 512), centred windows, periodic Hann window, max-normalised columns.
///
/// ## Invariants
///
/// Checked by [`TempogramConfig::validate`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::rhythm::TempogramConfig;
///
/// let cfg = TempogramConfig::new().with_win_length(nzu!(128)).with_norm(None);
/// assert_eq!(cfg.win_length.get(), 128);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TempogramConfig {
    /// Onset envelope parameters (also define the hop length).
    pub onset: OnsetStrengthConfig,
    /// Autocorrelation window length in frames; also the number of lag bins. Longer
    /// windows resolve slower tempi and smooth over time. librosa default: 384.
    pub win_length: NonZeroUsize,
    /// If `true`, windows are centred on their frame (the envelope is padded with
    /// linear ramps to zero). If `false`, the envelope must be at least
    /// `win_length` frames long. librosa default: `true`.
    pub center: bool,
    /// Window applied before autocorrelation (periodic form). librosa default: Hann.
    pub window: Window,
    /// Column normalisation (librosa `util.normalize` norm): `Some(inf)` divides by the
    /// maximum magnitude, `Some(p)` by the `p`-norm, `Some(0)` by the count of
    /// non-zeros, `None` disables it. librosa default: `Some(inf)`.
    pub norm: Option<f64>,
}

impl Default for TempogramConfig {
    fn default() -> Self {
        Self {
            onset: OnsetStrengthConfig::default(),
            win_length: crate::nzu!(384),
            center: true,
            window: Window::Hann,
            norm: Some(f64::INFINITY),
        }
    }
}

impl TempogramConfig {
    /// Returns the librosa-default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the onset envelope parameters.
    #[must_use]
    pub fn with_onset(mut self, onset: OnsetStrengthConfig) -> Self {
        self.onset = onset;
        self
    }

    /// Sets the window length.
    #[must_use]
    pub fn with_win_length(mut self, win_length: NonZeroUsize) -> Self {
        self.win_length = win_length;
        self
    }

    /// Sets centring.
    #[must_use]
    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Sets the window.
    #[must_use]
    pub fn with_window(mut self, window: Window) -> Self {
        self.window = window;
        self
    }

    /// Sets the column normalisation.
    #[must_use]
    pub fn with_norm(mut self, norm: Option<f64>) -> Self {
        self.norm = norm;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the onset configuration, window or norm is invalid.
    pub fn validate(&self, sample_rate: f64) -> AudioSampleResult<()> {
        self.onset.validate(sample_rate)?;
        validate_norm(self.norm)?;
        get_window(&self.window, self.win_length.get(), true).map(|_| ())
    }
}

/// numpy `pad(mode="linear_ramp", end_values=0)` by `w` at both ends.
fn linear_ramp_pad(x: &[f64], w: usize, out: &mut Vec<f64>) {
    out.clear();
    let n = x.len();
    let (first, last) = (x[0], x[n - 1]);
    let wf = w as f64;
    let (s0, s1) = (first / wf, last / wf);
    out.extend((0..w).map(|i| i as f64 * s0));
    out.extend_from_slice(x);
    out.extend((0..w).map(|j| (w - 1 - j) as f64 * s1));
}

/// Computes the autocorrelation tempogram of onset envelopes, equivalent to
/// `librosa.feature.tempogram(onset_envelope=envelope)`.
///
/// # Arguments
///
/// * `envelope` - `(channels, n_frames)` onset envelopes.
/// * `config` - Window length, centring, window and norm (`config.onset` is not used).
///
/// # Returns
///
/// A `(channels, win_length, n_frames)` array (`n_frames - win_length + 1` frames when
/// not centred).
///
/// # Errors
///
/// [`ParameterError`] if the envelope is empty or shorter than the window (uncentred),
/// or the window / norm is invalid.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::rhythm::{TempogramConfig, tempogram_from_envelope};
/// use ndarray::Array2;
///
/// let env = Array2::from_shape_fn((1, 200), |(_, t)| if t % 10 == 0 { 1.0 } else { 0.0 });
/// let tg = tempogram_from_envelope(env.view(), &TempogramConfig::new().with_win_length(nzu!(64)))
///     .unwrap();
/// assert_eq!(tg.dim(), (1, 64, 200));
/// // Strong periodicity at a lag of 10 frames in the middle of the signal.
/// assert!(tg[[0, 10, 100]] > tg[[0, 5, 100]]);
/// ```
///
/// # References
///
/// P. Grosche, M. Müller and F. Kurth, "Cyclic tempogram - a mid-level tempo
/// representation for music signals," ICASSP 2010; librosa `feature.tempogram`.
pub fn tempogram_from_envelope(
    envelope: ArrayView2<'_, f64>,
    config: &TempogramConfig,
) -> AudioSampleResult<Array3<f64>> {
    validate_norm(config.norm)?;
    let win = config.win_length.get();
    let (n_ch, n) = envelope.dim();
    if n_ch == 0 || n == 0 {
        return Err(param_err("onset_envelope", "the onset envelope is empty"));
    }
    // Output is (channels, win, <= n frames); the padded input is n + win.
    crate::utils::check_derived_len(
        "win_length",
        "the tempogram",
        n_ch as f64 * win as f64 * n as f64,
        crate::utils::MAX_SIGNAL_ELEMENTS,
    )?;
    let window = get_window(&config.window, win, true)?;
    let half = win / 2;
    let padded_len = if config.center { n + 2 * half } else { n };
    if padded_len < win {
        return Err(param_err(
            "win_length",
            format!("input is too short (n={n}) for win_length={win}"),
        ));
    }
    let mut n_frames = padded_len - win + 1;
    if config.center {
        n_frames = n_frames.min(n);
    }
    let n_pad = next_fast_len(2 * win - 1);
    let mut planner = RealFftPlanner::<f64>::new();
    let fwd = planner.plan_fft_forward(n_pad);
    let inv = planner.plan_fft_inverse(n_pad);
    let mut time_buf = fwd.make_input_vec();
    let mut spec = fwd.make_output_vec();
    let mut s_fwd = fwd.make_scratch_vec();
    let mut s_inv = inv.make_scratch_vec();
    let mut row_buf = Vec::with_capacity(n);
    let mut padded = Vec::with_capacity(padded_len);
    let scale = 1.0 / n_pad as f64;
    let mut tf = Array2::<f64>::zeros((n_frames, win));
    let mut out = Array3::<f64>::zeros((n_ch, win, n_frames));
    for (c, row) in envelope.outer_iter().enumerate() {
        row_buf.clear();
        row_buf.extend(row.iter().copied());
        let x: &[f64] = if config.center {
            linear_ramp_pad(&row_buf, half, &mut padded);
            &padded
        } else {
            &row_buf
        };
        for (t, mut dst) in tf.outer_iter_mut().enumerate() {
            for (k, slot) in time_buf.iter_mut().enumerate() {
                *slot = if k < win { x[t + k] * window[k] } else { 0.0 };
            }
            fwd.process_with_scratch(&mut time_buf, &mut spec, &mut s_fwd)
                .map_err(fft_err)?;
            for z in spec.iter_mut() {
                *z = Complex::new(z.re * z.re + z.im * z.im, 0.0);
            }
            inv.process_with_scratch(&mut spec, &mut time_buf, &mut s_inv)
                .map_err(fft_err)?;
            let dst = dst.as_slice_mut().expect("rows of a standard-layout array");
            for (d, &v) in dst.iter_mut().zip(&time_buf[..win]) {
                *d = v * scale;
            }
            normalize_in_place(dst, config.norm);
        }
        out.index_axis_mut(Axis(0), c).assign(&tf.t());
    }
    Ok(out)
}

/// Configuration of the Fourier tempogram, mirroring `librosa.feature.fourier_tempogram`.
///
/// ## Purpose
///
/// The Fourier tempogram is the STFT of the onset envelope (hop 1 frame): bin `k`
/// measures periodicity at tempo [`fourier_tempo_frequencies`]`[k]`, with phase.
///
/// ## Intended usage
///
/// Pass to [`AudioRhythm::fourier_tempogram`] or [`fourier_tempogram_from_envelope`].
/// `Default` reproduces librosa: default onset strength, `win_length = 384`, centred
/// (zero padding), periodic Hann window.
///
/// ## Invariants
///
/// Checked by [`FourierTempogramConfig::validate`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::rhythm::FourierTempogramConfig;
///
/// assert_eq!(FourierTempogramConfig::default().win_length.get(), 384);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FourierTempogramConfig {
    /// Onset envelope parameters (also define the hop length).
    pub onset: OnsetStrengthConfig,
    /// STFT length in frames; the tempogram has `win_length / 2 + 1` bins.
    /// librosa default: 384.
    pub win_length: NonZeroUsize,
    /// If `true`, frames are centred (zero padding). librosa default: `true`.
    pub center: bool,
    /// Analysis window. librosa default: Hann.
    pub window: Window,
}

impl Default for FourierTempogramConfig {
    fn default() -> Self {
        Self {
            onset: OnsetStrengthConfig::default(),
            win_length: crate::nzu!(384),
            center: true,
            window: Window::Hann,
        }
    }
}

impl FourierTempogramConfig {
    /// Returns the librosa-default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the onset envelope parameters.
    #[must_use]
    pub fn with_onset(mut self, onset: OnsetStrengthConfig) -> Self {
        self.onset = onset;
        self
    }

    /// Sets the window length.
    #[must_use]
    pub fn with_win_length(mut self, win_length: NonZeroUsize) -> Self {
        self.win_length = win_length;
        self
    }

    /// Sets centring.
    #[must_use]
    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Sets the window.
    #[must_use]
    pub fn with_window(mut self, window: Window) -> Self {
        self.window = window;
        self
    }

    fn stft_config(&self) -> StftConfig {
        StftConfig::new()
            .with_n_fft(self.win_length)
            .with_hop_length(Some(crate::nzu!(1)))
            .with_window(self.window.clone())
            .with_center(self.center)
            .with_pad_mode(StftPadMode::Constant)
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the onset configuration or window is invalid.
    pub fn validate(&self, sample_rate: f64) -> AudioSampleResult<()> {
        self.onset.validate(sample_rate)?;
        self.stft_config().validate()
    }
}

/// Computes the Fourier tempogram of onset envelopes, equivalent to
/// `librosa.feature.fourier_tempogram(onset_envelope=envelope)`.
///
/// # Arguments
///
/// * `envelope` - `(channels, n_frames)` onset envelopes.
/// * `config` - Window length, centring and window (`config.onset` is not used).
///
/// # Returns
///
/// A complex `(channels, win_length / 2 + 1, n_frames + 1)` array when centred
/// (`1 + n_frames - win_length` frames otherwise), as [`stft`] with a hop of 1.
///
/// # Errors
///
/// [`ParameterError`] if the envelope is empty or (uncentred) shorter than the window.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::rhythm::{FourierTempogramConfig, fourier_tempogram_from_envelope};
/// use ndarray::Array2;
///
/// let env = Array2::from_shape_fn((1, 100), |(_, t)| (t % 8 == 0) as u8 as f64);
/// let cfg = FourierTempogramConfig::new().with_win_length(nzu!(64));
/// let ft = fourier_tempogram_from_envelope(env.view(), &cfg).unwrap();
/// assert_eq!(ft.dim(), (1, 33, 101));
/// ```
pub fn fourier_tempogram_from_envelope(
    envelope: ArrayView2<'_, f64>,
    config: &FourierTempogramConfig,
) -> AudioSampleResult<Array3<Complex<f64>>> {
    stft(envelope, &config.stft_config())
}

// =============================================================================
// Tempo
// =============================================================================

/// Configuration of tempo estimation, mirroring `librosa.feature.tempo`.
///
/// ## Purpose
///
/// The tempo is the lag bin maximising `log1p(1e6 * tempogram) + log prior`, where the
/// default prior is log-normal around `start_bpm` with `std_bpm` octaves of spread.
///
/// ## Intended usage
///
/// Pass to [`AudioRhythm::tempo`], [`tempo_from_envelope`] or [`tempo_from_tempogram`].
/// `Default` reproduces librosa: default onset strength, `start_bpm = 120`,
/// `std_bpm = 1`, an 8 s autocorrelation window, tempi of 320 BPM and above excluded,
/// and a global (time-averaged) estimate.
///
/// ## Invariants
///
/// Checked by [`TempoConfig::validate`]: `start_bpm`, `std_bpm`, `ac_size` and
/// `max_tempo` are finite and positive.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::rhythm::TempoConfig;
///
/// // Time-varying tempo (librosa aggregate=None).
/// let cfg = TempoConfig::new().with_aggregate(None).with_start_bpm(90.0);
/// assert_eq!(cfg.max_tempo, Some(320.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TempoConfig {
    /// Onset envelope parameters (also define the hop length).
    pub onset: OnsetStrengthConfig,
    /// Centre of the default log-normal tempo prior, in BPM. librosa default: 120.
    pub start_bpm: f64,
    /// Standard deviation of the default prior in octaves (log2 BPM). Smaller values
    /// pull the estimate towards `start_bpm`. librosa default: 1.
    pub std_bpm: f64,
    /// Length of the autocorrelation window in seconds (`win_length` of the tempogram
    /// is `floor(floor(ac_size * sr) / hop)` frames). librosa default: 8.
    pub ac_size: f64,
    /// Tempi at or above this value (BPM) are excluded. `None` allows any tempo.
    /// librosa default: `Some(320)`.
    pub max_tempo: Option<f64>,
    /// How the tempogram is combined over time. `None` estimates one tempo per frame.
    /// librosa default: `Some(Mean)`.
    pub aggregate: Option<Aggregate>,
    /// Custom log-prior over BPM replacing the log-normal default (librosa
    /// `prior.logpdf`). librosa default: `None`.
    pub prior: Option<TempoLogPrior>,
}

impl Default for TempoConfig {
    fn default() -> Self {
        Self {
            onset: OnsetStrengthConfig::default(),
            start_bpm: 120.0,
            std_bpm: 1.0,
            ac_size: 8.0,
            max_tempo: Some(320.0),
            aggregate: Some(Aggregate::Mean),
            prior: None,
        }
    }
}

impl TempoConfig {
    /// Returns the librosa-default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the onset envelope parameters.
    #[must_use]
    pub fn with_onset(mut self, onset: OnsetStrengthConfig) -> Self {
        self.onset = onset;
        self
    }

    /// Sets the prior centre in BPM.
    #[must_use]
    pub fn with_start_bpm(mut self, start_bpm: f64) -> Self {
        self.start_bpm = start_bpm;
        self
    }

    /// Sets the prior spread in octaves.
    #[must_use]
    pub fn with_std_bpm(mut self, std_bpm: f64) -> Self {
        self.std_bpm = std_bpm;
        self
    }

    /// Sets the autocorrelation window in seconds.
    #[must_use]
    pub fn with_ac_size(mut self, ac_size: f64) -> Self {
        self.ac_size = ac_size;
        self
    }

    /// Sets the maximum tempo.
    #[must_use]
    pub fn with_max_tempo(mut self, max_tempo: Option<f64>) -> Self {
        self.max_tempo = max_tempo;
        self
    }

    /// Sets the time aggregation (`None` = time-varying).
    #[must_use]
    pub fn with_aggregate(mut self, aggregate: Option<Aggregate>) -> Self {
        self.aggregate = aggregate;
        self
    }

    /// Sets a custom log-prior.
    #[must_use]
    pub fn with_prior(mut self, prior: Option<TempoLogPrior>) -> Self {
        self.prior = prior;
        self
    }

    fn validate_tempo(&self) -> AudioSampleResult<()> {
        require_positive("start_bpm", self.start_bpm)?;
        require_positive("std_bpm", self.std_bpm)?;
        require_positive("ac_size", self.ac_size)?;
        if let Some(m) = self.max_tempo {
            require_positive("max_tempo", m)?;
        }
        Ok(())
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the onset configuration is invalid or a tempo parameter is
    /// not finite and positive.
    pub fn validate(&self, sample_rate: f64) -> AudioSampleResult<()> {
        self.onset.validate(sample_rate)?;
        self.validate_tempo()
    }
}

/// Estimates tempo from a precomputed tempogram, equivalent to
/// `librosa.feature.tempo(tg=tempogram)`.
///
/// # Arguments
///
/// * `tempogram` - `(channels, win_length, n_frames)` autocorrelation tempogram; its
///   bin count sets the BPM grid.
/// * `sample_rate`, `hop_length` - Frame geometry of the tempogram.
/// * `config` - Prior, maximum tempo and aggregation (`onset` and `ac_size` unused).
///
/// # Returns
///
/// `(channels, 1)` tempi in BPM, or `(channels, n_frames)` when `config.aggregate` is
/// `None`. Values are on the [`tempo_frequencies`] grid.
///
/// # Errors
///
/// [`ParameterError`] if a tempo parameter is invalid or the tempogram is empty.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::rhythm::{TempoConfig, tempo_from_tempogram};
/// use ndarray::Array3;
///
/// // A tempogram peaking at lag 22 (about 117 BPM at 22050 Hz / 512).
/// let tg = Array3::from_shape_fn((1, 100, 3), |(_, k, _)| if k == 22 { 1.0 } else { 0.1 });
/// let bpm = tempo_from_tempogram(tg.view(), 22050.0, nzu!(512), &TempoConfig::default()).unwrap();
/// assert!((bpm[[0, 0]] - 117.45383522727273).abs() < 1e-9);
/// ```
pub fn tempo_from_tempogram(
    tempogram: ArrayView3<'_, f64>,
    sample_rate: f64,
    hop_length: NonZeroUsize,
    config: &TempoConfig,
) -> AudioSampleResult<Array2<f64>> {
    config.validate_tempo()?;
    require_positive("sample_rate", sample_rate)?;
    let (n_ch, win, n_t) = tempogram.dim();
    if n_ch == 0 || win == 0 || n_t == 0 {
        return Err(param_err("tg", "the tempogram is empty"));
    }
    let bpms = tempo_frequencies(win, hop_length, sample_rate);
    let mut logprior: Vec<f64> = bpms
        .iter()
        .map(|&b| match config.prior {
            Some(p) => (p.0)(b),
            None => {
                let z = (b.log2() - config.start_bpm.log2()) / config.std_bpm;
                -0.5 * (z * z)
            }
        })
        .collect();
    if let Some(m) = config.max_tempo {
        let max_idx = bpms.iter().position(|&b| b < m).unwrap_or(0);
        for v in &mut logprior[..max_idx] {
            *v = f64::NEG_INFINITY;
        }
    }
    let n_out = if config.aggregate.is_some() { 1 } else { n_t };
    let mut out = Array2::<f64>::zeros((n_ch, n_out));
    let mut col = vec![0.0; win];
    let mut series = Vec::with_capacity(n_t);
    let mut scratch = Vec::with_capacity(n_t);
    for c in 0..n_ch {
        let tc = tempogram.index_axis(Axis(0), c);
        for t in 0..n_out {
            for (k, v) in col.iter_mut().enumerate() {
                *v = match config.aggregate {
                    Some(agg) => {
                        series.clear();
                        series.extend(tc.row(k).iter().copied());
                        agg.reduce(&series, &mut scratch, true)
                    }
                    None => tc[[k, t]],
                };
            }
            // numpy.argmax: first maximum, a NaN wins immediately.
            let mut best = 0usize;
            let mut best_v = f64::NAN;
            for (k, (&g, &lp)) in col.iter().zip(&logprior).enumerate() {
                let v = (1e6 * g).ln_1p() + lp;
                if v.is_nan() {
                    best = k;
                    break;
                }
                if k == 0 || v > best_v {
                    best = k;
                    best_v = v;
                }
            }
            out[[c, t]] = bpms[best];
        }
    }
    Ok(out)
}

/// Estimates tempo from onset envelopes, equivalent to
/// `librosa.feature.tempo(onset_envelope=envelope)`.
///
/// Computes a [`tempogram_from_envelope`] with a window of `ac_size` seconds and
/// passes it to [`tempo_from_tempogram`].
///
/// # Arguments
///
/// * `envelope` - `(channels, n_frames)` onset envelopes.
/// * `sample_rate`, `hop_length` - Frame geometry of the envelope.
/// * `config` - Tempo parameters (`config.onset` is not used).
///
/// # Returns
///
/// `(channels, 1)` tempi (or `(channels, n_frames)` for time-varying estimation).
///
/// # Errors
///
/// [`ParameterError`] if a parameter is invalid, `ac_size` is shorter than one frame,
/// the envelope is empty, or the tempogram would exceed 2^31 elements.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::rhythm::{TempoConfig, tempo_from_envelope};
/// use ndarray::Array2;
///
/// // One pulse every 22 frames: ~117 BPM at 22050 Hz / 512.
/// let env = Array2::from_shape_fn((1, 400), |(_, t)| if t % 22 == 0 { 1.0 } else { 0.0 });
/// let bpm = tempo_from_envelope(env.view(), 22050.0, nzu!(512), &TempoConfig::default()).unwrap();
/// assert!((bpm[[0, 0]] - 117.45).abs() < 0.01);
/// ```
pub fn tempo_from_envelope(
    envelope: ArrayView2<'_, f64>,
    sample_rate: f64,
    hop_length: NonZeroUsize,
    config: &TempoConfig,
) -> AudioSampleResult<Array2<f64>> {
    config.validate_tempo()?;
    require_positive("sample_rate", sample_rate)?;
    // librosa.time_to_frames: int(t * sr) // hop
    let samples = (config.ac_size * sample_rate).trunc();
    let win = (samples / hop_length.get() as f64).floor();
    let (n_ch, n) = envelope.dim();
    crate::utils::check_derived_len(
        "ac_size",
        "the autocorrelation tempogram",
        n_ch as f64 * win * n as f64,
        crate::utils::MAX_SIGNAL_ELEMENTS,
    )?;
    let win = win as usize;
    let win_length = NonZeroUsize::new(win).ok_or_else(|| {
        param_err(
            "ac_size",
            "ac_size is shorter than one frame; the autocorrelation window is empty",
        )
    })?;
    let tg_cfg = TempogramConfig {
        win_length,
        ..TempogramConfig::default()
    };
    let tg = tempogram_from_envelope(envelope, &tg_cfg)?;
    tempo_from_tempogram(tg.view(), sample_rate, hop_length, config)
}

// =============================================================================
// Beat tracking
// =============================================================================

/// Configuration of the dynamic-programming beat tracker, mirroring
/// `librosa.beat.beat_track`.
///
/// ## Purpose
///
/// Beats are the path through the onset envelope that maximises onset strength while
/// keeping inter-beat intervals close to the period of the estimated (or given) tempo.
///
/// ## Intended usage
///
/// Pass to [`AudioRhythm::beat_track`] or [`beat_track_from_envelope`]. `Default`
/// reproduces librosa: onset strength with **median** aggregation, tempo estimated
/// with `start_bpm = 120`, `tightness = 100`, weak leading/trailing beats trimmed.
///
/// ## Invariants
///
/// Checked by [`BeatTrackConfig::validate`]: `start_bpm`, `tightness` and `bpm` are
/// finite and positive.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::rhythm::{Aggregate, BeatTrackConfig};
///
/// let cfg = BeatTrackConfig::new().with_bpm(Some(100.0)).with_trim(false);
/// assert_eq!(cfg.onset.aggregate, Some(Aggregate::Median));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BeatTrackConfig {
    /// Onset envelope parameters (also define the hop length). librosa default:
    /// default onset strength with `aggregate = Median`.
    pub onset: OnsetStrengthConfig,
    /// Prior centre of the tempo estimate in BPM. librosa default: 120.
    pub start_bpm: f64,
    /// How strictly beats follow the tempo; larger values penalise deviations from the
    /// beat period more. librosa default: 100. (librosa passes it to its kernel as
    /// `float32`; the value is rounded the same way here.)
    pub tightness: f64,
    /// If `true`, beats at the start and end whose local score is below half the RMS
    /// of the smoothed beat scores are removed. librosa default: `true`.
    pub trim: bool,
    /// Tempo in BPM to track at. `None` estimates it as [`TempoConfig`] does with
    /// `start_bpm` and `prior`. librosa default: `None`.
    pub bpm: Option<f64>,
    /// Custom log-prior for the tempo estimate. librosa default: `None`.
    pub prior: Option<TempoLogPrior>,
}

impl Default for BeatTrackConfig {
    fn default() -> Self {
        Self {
            onset: OnsetStrengthConfig::default().with_aggregate(Some(Aggregate::Median)),
            start_bpm: 120.0,
            tightness: 100.0,
            trim: true,
            bpm: None,
            prior: None,
        }
    }
}

impl BeatTrackConfig {
    /// Returns the librosa-default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the onset envelope parameters.
    #[must_use]
    pub fn with_onset(mut self, onset: OnsetStrengthConfig) -> Self {
        self.onset = onset;
        self
    }

    /// Sets the prior centre in BPM.
    #[must_use]
    pub fn with_start_bpm(mut self, start_bpm: f64) -> Self {
        self.start_bpm = start_bpm;
        self
    }

    /// Sets the tightness.
    #[must_use]
    pub fn with_tightness(mut self, tightness: f64) -> Self {
        self.tightness = tightness;
        self
    }

    /// Sets trimming.
    #[must_use]
    pub fn with_trim(mut self, trim: bool) -> Self {
        self.trim = trim;
        self
    }

    /// Sets a fixed tempo.
    #[must_use]
    pub fn with_bpm(mut self, bpm: Option<f64>) -> Self {
        self.bpm = bpm;
        self
    }

    /// Sets a custom log-prior.
    #[must_use]
    pub fn with_prior(mut self, prior: Option<TempoLogPrior>) -> Self {
        self.prior = prior;
        self
    }

    fn validate_tracking(&self) -> AudioSampleResult<()> {
        require_positive("start_bpm", self.start_bpm)?;
        require_positive("tightness", self.tightness)?;
        if let Some(b) = self.bpm {
            require_positive("bpm", b)?;
        }
        Ok(())
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the onset configuration is invalid or a tracking parameter
    /// is not finite and positive.
    pub fn validate(&self, sample_rate: f64) -> AudioSampleResult<()> {
        self.onset.validate(sample_rate)?;
        self.validate_tracking()
    }
}

/// Result of [`AudioRhythm::beat_track`].
///
/// ## Purpose
///
/// The global tempo used by the tracker and the beat positions.
///
/// ## Invariants
///
/// `tempo` is `0.0` and `beats` is empty when the onset envelope is all zeros.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::rhythm::{BeatTrack, FrameEvents};
///
/// let bt = BeatTrack { tempo: 120.0, beats: FrameEvents { frames: vec![], sample_rate: 22050.0,
///     hop_length: nzu!(512) } };
/// assert!(bt.beats.is_empty());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BeatTrack {
    /// Tempo in BPM (estimated, or the configured `bpm`).
    pub tempo: f64,
    /// Beat positions.
    pub beats: FrameEvents,
}

/// Sample standard deviation (`ddof=1`) computed as numpy does.
fn std_ddof1(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = pairwise_sum(x) / n;
    let sq: Vec<f64> = x
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .collect();
    (pairwise_sum(&sq) / (n - 1.0)).sqrt()
}

/// librosa `__beat_local_score`: a same-mode Gaussian smoothing of the envelope whose
/// width follows the (possibly time-varying) beat period.
fn beat_local_score(onsets: &[f64], fpb: &[f64], out: &mut [f64]) {
    let n = onsets.len();
    let mut window: Vec<f64> = Vec::new();
    let mut cached = f64::NAN;
    for i in 0..n {
        let f = if fpb.len() == 1 { fpb[0] } else { fpb[i] };
        if f != cached {
            cached = f;
            let k_len = 2 * (f as usize) + 1;
            window.clear();
            window.extend((0..k_len).map(|k| {
                let z = ((k as f64 - f) * 32.0) / f;
                (-0.5 * (z * z)).exp()
            }));
        }
        let k_len = window.len();
        let half = k_len / 2;
        let lo = (i + half + 1).saturating_sub(n);
        let hi = (i + half).min(k_len);
        let mut acc = 0.0;
        for (k, &w) in window.iter().enumerate().take(hi).skip(lo) {
            acc += w * onsets[i + half - k];
        }
        out[i] = acc;
    }
}

/// librosa `__beat_track_dp`: cumulative score and back-links.
fn beat_track_dp(
    localscore: &[f64],
    fpb: &[f64],
    tightness: f64,
    backlink: &mut [isize],
    cumscore: &mut [f64],
) {
    let score_thresh = 0.01 * np_max(localscore);
    let mut first_beat = true;
    backlink[0] = -1;
    cumscore[0] = localscore[0];
    let tv = fpb.len() > 1;
    for (i, &score_i) in localscore.iter().enumerate() {
        let f = fpb[if tv { i } else { 0 }];
        let ln_f = f.ln();
        let start = i as isize - (f / 2.0).round_ties_even() as isize;
        let stop = (i as f64 - 2.0 * f - 1.0) as isize;
        let mut best = f64::NEG_INFINITY;
        let mut loc_best: isize = -1;
        let mut loc = start;
        while loc > stop {
            if loc < 0 {
                break;
            }
            let d = ((i as isize - loc) as f64).ln() - ln_f;
            let score = cumscore[loc as usize] - tightness * (d * d);
            if score > best {
                best = score;
                loc_best = loc;
            }
            loc -= 1;
        }
        cumscore[i] = if loc_best >= 0 {
            score_i + best
        } else {
            score_i
        };
        if first_beat && score_i < score_thresh {
            backlink[i] = -1;
        } else {
            backlink[i] = loc_best;
            first_beat = false;
        }
    }
}

/// librosa `__last_beat`: the last local maximum of the cumulative score that reaches
/// half the median of all its local maxima.
fn last_beat(cumscore: &[f64]) -> usize {
    let n = cumscore.len();
    let mut is_max = vec![false; n];
    if n >= 2 {
        for i in 1..n - 1 {
            is_max[i] = cumscore[i] > cumscore[i - 1] && cumscore[i] >= cumscore[i + 1];
        }
        is_max[n - 1] = cumscore[n - 1] > cumscore[n - 2];
    }
    let mut maxima: Vec<f64> = cumscore
        .iter()
        .zip(&is_max)
        .filter_map(|(&c, &m)| m.then_some(c))
        .collect();
    let threshold = if maxima.is_empty() {
        0.0
    } else {
        0.5 * median_in_place(&mut maxima)
    };
    (0..n)
        .rev()
        .find(|&k| is_max[k] && cumscore[k] >= threshold)
        .unwrap_or(n - 1)
}

/// librosa `__trim_beats`.
fn trim_beats(localscore: &[f64], beats: &mut [bool], trim: bool) {
    let n = localscore.len();
    let threshold = if trim {
        let bl: Vec<f64> = localscore
            .iter()
            .zip(beats.iter())
            .filter_map(|(&s, &b)| b.then_some(s))
            .collect();
        let b = bl.len();
        let w = [0.0, 0.5, 1.0, 0.5, 0.0];
        // Full convolution, sliced [2 : n + 2] as librosa does.
        let end = (n + 2).min(b + 4);
        let sq: Vec<f64> = (2..end)
            .map(|j| {
                let mut acc = 0.0;
                for (k, &wk) in w.iter().enumerate() {
                    if j >= k && j - k < b {
                        acc += wk * bl[j - k];
                    }
                }
                acc * acc
            })
            .collect();
        if sq.is_empty() {
            0.0
        } else {
            0.5 * (pairwise_sum(&sq) / sq.len() as f64).sqrt()
        }
    } else {
        0.0
    };
    let mut k = 0;
    while k < n && localscore[k] <= threshold {
        beats[k] = false;
        k += 1;
    }
    let mut k = n;
    while k > 0 && localscore[k - 1] <= threshold {
        beats[k - 1] = false;
        k -= 1;
    }
}

/// Tracks beats in an onset envelope at a given tempo (or tempo curve) with Ellis's
/// dynamic programming, equivalent to librosa's internal `__beat_tracker` (what
/// `librosa.beat.beat_track(onset_envelope=..., bpm=...)` runs).
///
/// Steps: normalise the envelope by its standard deviation; smooth it with a Gaussian
/// of width `period / 32` (the local score); accumulate the best score of a beat at
/// each frame given a predecessor `period / 2 .. 2 period` frames earlier, penalised by
/// `tightness * (ln(interval) - ln(period))^2`; backtrack from the last strong
/// cumulative-score peak; optionally trim weak beats at both ends.
///
/// # Arguments
///
/// * `envelope` - Onset strength envelope.
/// * `bpm` - Tempo in BPM: one value, or one per frame of `envelope` (time-varying).
/// * `frame_rate` - Envelope frames per second (`sample_rate / hop_length`).
/// * `tightness` - Tempo-consistency weight (rounded to `f32` like librosa).
/// * `trim` - Remove weak leading/trailing beats.
///
/// # Returns
///
/// Ascending beat frames (empty for an empty envelope).
///
/// # Errors
///
/// [`ParameterError`] if `bpm` is empty, has the wrong length or non-positive values,
/// a tempo is so fast that its period rounds to less than one frame, `tightness` or
/// `frame_rate` is not finite and positive, or a beat period is so long (an extreme
/// `frame_rate`) that the local-score window would exceed 2^31 frames.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::rhythm::beat_track_with_bpm;
///
/// let env: Vec<f64> = (0..200).map(|t| if t % 20 == 5 { 1.0 } else { 0.0 }).collect();
/// let beats = beat_track_with_bpm(&env, &[129.2], 22050.0 / 512.0, 100.0, true).unwrap();
/// // Same frames as librosa.beat.beat_track(onset_envelope=env, bpm=129.2).
/// assert_eq!(beats, vec![5, 25, 45, 65, 85, 105, 125, 145, 165, 182]);
/// ```
///
/// # References
///
/// D. P. W. Ellis, "Beat tracking by dynamic programming," *J. New Music Research*
/// 36(1), 2007; librosa `beat.beat_track`.
pub fn beat_track_with_bpm(
    envelope: &[f64],
    bpm: &[f64],
    frame_rate: f64,
    tightness: f64,
    trim: bool,
) -> AudioSampleResult<Vec<usize>> {
    require_positive("tightness", tightness)?;
    require_positive("frame_rate", frame_rate)?;
    let n = envelope.len();
    if bpm.is_empty() || (bpm.len() != 1 && bpm.len() != n) {
        return Err(param_err(
            "bpm",
            format!(
                "bpm must have 1 or {n} values (one per envelope frame), got {}",
                bpm.len()
            ),
        ));
    }
    if bpm.iter().any(|&b| !(b.is_finite() && b > 0.0)) {
        return Err(param_err("bpm", "bpm must be finite and strictly positive"));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let fpb: Vec<f64> = bpm
        .iter()
        .map(|&b| (frame_rate * 60.0 / b).round_ties_even())
        .collect();
    if fpb.iter().any(|&f| f < 1.0) {
        return Err(param_err(
            "bpm",
            "tempo too fast for the frame rate: fewer than one frame per beat",
        ));
    }
    // The local-score window spans 2 * frames_per_beat + 1 frames.
    let max_fpb = fpb.iter().copied().fold(0.0, f64::max);
    crate::utils::check_derived_len(
        "frame_rate",
        "the beat-tracking window",
        2.0 * max_fpb + 1.0,
        crate::utils::MAX_SIGNAL_ELEMENTS,
    )?;
    let norm = std_ddof1(envelope) + f64::MIN_POSITIVE;
    let onsets: Vec<f64> = envelope.iter().map(|&v| v / norm).collect();
    let mut localscore = vec![0.0; n];
    beat_local_score(&onsets, &fpb, &mut localscore);
    let mut backlink = vec![-1isize; n];
    let mut cumscore = vec![0.0; n];
    beat_track_dp(
        &localscore,
        &fpb,
        f64::from(tightness as f32),
        &mut backlink,
        &mut cumscore,
    );
    let tail = last_beat(&cumscore);
    let mut beats = vec![false; n];
    let mut k = tail as isize;
    while k >= 0 {
        beats[k as usize] = true;
        k = backlink[k as usize];
    }
    trim_beats(&localscore, &mut beats, trim);
    Ok(beats
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| b.then_some(i))
        .collect())
}

/// Estimates the tempo (unless given) and tracks beats in an onset envelope,
/// equivalent to `librosa.beat.beat_track(onset_envelope=envelope)`.
///
/// # Arguments
///
/// * `envelope` - Onset strength envelope (one channel).
/// * `sample_rate`, `hop_length` - Frame geometry of the envelope.
/// * `config` - Tracking parameters (`config.onset` is not used).
///
/// # Returns
///
/// The tempo and the beat frames. An all-zero envelope gives tempo `0.0` and no
/// beats (as librosa).
///
/// # Errors
///
/// [`ParameterError`] if a parameter is invalid (see [`beat_track_with_bpm`]).
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::rhythm::{BeatTrackConfig, beat_track_from_envelope};
///
/// let env: Vec<f64> = (0..300).map(|t| if t % 22 == 3 { 1.0 } else { 0.05 }).collect();
/// let bt = beat_track_from_envelope(&env, 22050.0, nzu!(512), &BeatTrackConfig::default()).unwrap();
/// assert!((bt.tempo - 117.45).abs() < 0.01);
/// assert!(bt.beats.frames.iter().all(|b| b % 22 == 3));
/// ```
pub fn beat_track_from_envelope(
    envelope: &[f64],
    sample_rate: f64,
    hop_length: NonZeroUsize,
    config: &BeatTrackConfig,
) -> AudioSampleResult<BeatTrack> {
    config.validate_tracking()?;
    require_positive("sample_rate", sample_rate)?;
    let empty = |tempo| BeatTrack {
        tempo,
        beats: FrameEvents {
            frames: Vec::new(),
            sample_rate,
            hop_length,
        },
    };
    if !envelope.iter().any(|&v| v != 0.0) {
        return Ok(empty(0.0));
    }
    let tempo = match config.bpm {
        Some(b) => b,
        None => {
            let env = ndarray::ArrayView2::from_shape((1, envelope.len()), envelope)
                .expect("a slice is a valid 1-row array");
            let cfg = TempoConfig {
                start_bpm: config.start_bpm,
                prior: config.prior,
                ..TempoConfig::default()
            };
            tempo_from_envelope(env, sample_rate, hop_length, &cfg)?[[0, 0]]
        }
    };
    let frame_rate = sample_rate / hop_length.get() as f64;
    let frames = beat_track_with_bpm(
        envelope,
        &[tempo],
        frame_rate,
        config.tightness,
        config.trim,
    )?;
    Ok(BeatTrack {
        tempo,
        beats: FrameEvents {
            frames,
            sample_rate,
            hop_length,
        },
    })
}

// =============================================================================
// Predominant local pulse
// =============================================================================

/// Configuration of predominant local pulse (PLP) estimation, mirroring
/// `librosa.beat.plp`.
///
/// ## Purpose
///
/// PLP keeps, in every frame of the Fourier tempogram, only the strongest tempo
/// component (with its phase) and resynthesises those sinusoids by overlap-add, giving
/// a pulse curve whose peaks are beat candidates that may change tempo over time.
///
/// ## Intended usage
///
/// Pass to [`AudioRhythm::plp`] or [`plp_from_envelope`]. `Default` reproduces
/// librosa: onset strength with median aggregation, `win_length = 384`, tempi limited
/// to 30-300 BPM, no prior.
///
/// ## Invariants
///
/// Checked by [`PlpConfig::validate`]: `tempo_min < tempo_max` when both are set.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::rhythm::PlpConfig;
///
/// let cfg = PlpConfig::new().with_tempo_range(Some(60.0), Some(200.0));
/// assert_eq!(cfg.tempo_max, Some(200.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PlpConfig {
    /// Onset envelope parameters (also define the hop length). librosa default:
    /// default onset strength with `aggregate = Median`.
    pub onset: OnsetStrengthConfig,
    /// Fourier tempogram window in frames. librosa default: 384.
    pub win_length: NonZeroUsize,
    /// Tempo bins below this BPM are discarded. librosa default: `Some(30)`.
    pub tempo_min: Option<f64>,
    /// Tempo bins above this BPM are discarded. librosa default: `Some(300)`.
    pub tempo_max: Option<f64>,
    /// Log-prior over BPM added to the log-magnitude before picking the peak.
    /// librosa default: `None`.
    pub prior: Option<TempoLogPrior>,
}

impl Default for PlpConfig {
    fn default() -> Self {
        Self {
            onset: OnsetStrengthConfig::default().with_aggregate(Some(Aggregate::Median)),
            win_length: crate::nzu!(384),
            tempo_min: Some(30.0),
            tempo_max: Some(300.0),
            prior: None,
        }
    }
}

impl PlpConfig {
    /// Returns the librosa-default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the onset envelope parameters.
    #[must_use]
    pub fn with_onset(mut self, onset: OnsetStrengthConfig) -> Self {
        self.onset = onset;
        self
    }

    /// Sets the window length.
    #[must_use]
    pub fn with_win_length(mut self, win_length: NonZeroUsize) -> Self {
        self.win_length = win_length;
        self
    }

    /// Sets the admissible tempo range.
    #[must_use]
    pub fn with_tempo_range(mut self, tempo_min: Option<f64>, tempo_max: Option<f64>) -> Self {
        self.tempo_min = tempo_min;
        self.tempo_max = tempo_max;
        self
    }

    /// Sets a log-prior.
    #[must_use]
    pub fn with_prior(mut self, prior: Option<TempoLogPrior>) -> Self {
        self.prior = prior;
        self
    }

    fn validate_range(&self) -> AudioSampleResult<()> {
        if let (Some(lo), Some(hi)) = (self.tempo_min, self.tempo_max) {
            if hi <= lo {
                return Err(param_err(
                    "tempo_max",
                    format!("tempo_max={hi} must be larger than tempo_min={lo}"),
                ));
            }
        }
        Ok(())
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the onset configuration is invalid or
    /// `tempo_max <= tempo_min`.
    pub fn validate(&self, sample_rate: f64) -> AudioSampleResult<()> {
        self.onset.validate(sample_rate)?;
        self.validate_range()
    }
}

/// numpy's `maximum` reduction over complex values (lexicographic order).
fn complex_max_lex(values: impl Iterator<Item = Complex<f64>>) -> Complex<f64> {
    let mut it = values;
    let mut m = it.next().unwrap_or_default();
    for v in it {
        let keep = m.re.is_nan()
            || m.im.is_nan()
            || (m.re > v.re && !v.im.is_nan())
            || (m.re == v.re && m.im >= v.im);
        if !keep {
            m = v;
        }
    }
    m
}

/// Computes the predominant local pulse of onset envelopes, equivalent to
/// `librosa.beat.plp(onset_envelope=envelope)`.
///
/// # Arguments
///
/// * `envelope` - `(channels, n_frames)` onset envelopes.
/// * `sample_rate`, `hop_length` - Frame geometry of the envelope.
/// * `config` - Window, tempo range and prior (`config.onset` is not used).
///
/// # Returns
///
/// A `(channels, n_frames)` pulse curve, non-negative and scaled to a maximum of 1 per
/// channel.
///
/// # Errors
///
/// [`ParameterError`] if `tempo_max <= tempo_min` or the envelope is empty.
///
/// # Differences from librosa
///
/// None in the algorithm, including librosa's normalisation by the magnitude of the
/// complex *maximum* (numpy's lexicographic order) of each tempogram frame.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::rhythm::{PlpConfig, plp_from_envelope};
/// use ndarray::Array2;
///
/// let env = Array2::from_shape_fn((1, 400), |(_, t)| if t % 20 == 0 { 1.0 } else { 0.0 });
/// let pulse = plp_from_envelope(env.view(), 22050.0, nzu!(512), &PlpConfig::default()).unwrap();
/// assert_eq!(pulse.dim(), (1, 400));
/// assert!(pulse[[0, 200]] > pulse[[0, 210]]);
/// ```
///
/// # References
///
/// P. Grosche and M. Müller, "Extracting predominant local pulse information from
/// music recordings," *IEEE Trans. ASLP* 19(6), 2011; librosa `beat.plp`.
pub fn plp_from_envelope(
    envelope: ArrayView2<'_, f64>,
    sample_rate: f64,
    hop_length: NonZeroUsize,
    config: &PlpConfig,
) -> AudioSampleResult<Array2<f64>> {
    config.validate_range()?;
    require_positive("sample_rate", sample_rate)?;
    let n = envelope.ncols();
    if n == 0 || envelope.nrows() == 0 {
        return Err(param_err("onset_envelope", "the onset envelope is empty"));
    }
    let ft_cfg = FourierTempogramConfig {
        win_length: config.win_length,
        ..FourierTempogramConfig::default()
    };
    let mut ft = fourier_tempogram_from_envelope(envelope, &ft_cfg)?;
    let freqs = fourier_tempo_frequencies(sample_rate, hop_length, config.win_length);
    let prior: Vec<f64> = freqs
        .iter()
        .map(|&f| config.prior.map_or(0.0, |p| (p.0)(f)))
        .collect();
    let (n_ch, n_bins, n_frames) = ft.dim();
    let mut mag = vec![0.0; n_bins];
    let tiny_sqrt = f64::MIN_POSITIVE.sqrt();
    for c in 0..n_ch {
        for t in 0..n_frames {
            let mut col = ft.slice_mut(ndarray::s![c, .., t]);
            for (k, z) in col.iter_mut().enumerate() {
                let f = freqs[k];
                if config.tempo_min.is_some_and(|lo| f < lo)
                    || config.tempo_max.is_some_and(|hi| f > hi)
                {
                    *z = Complex::new(0.0, 0.0);
                }
            }
            for (k, z) in col.iter().enumerate() {
                mag[k] = if config.prior.is_some() {
                    (1e6 * z.norm()).ln_1p() + prior[k]
                } else {
                    (1e6 * z.norm()).ln_1p()
                };
            }
            let peak = np_max(&mag);
            for (k, z) in col.iter_mut().enumerate() {
                if mag[k] < peak {
                    *z = Complex::new(0.0, 0.0);
                }
            }
            let denom = tiny_sqrt + complex_max_lex(col.iter().copied()).norm();
            for z in col.iter_mut() {
                *z = Complex::new(z.re / denom, z.im / denom);
            }
        }
    }
    let mut pulse = istft(ft.view(), &ft_cfg.stft_config(), Some(n))?;
    for mut row in pulse.outer_iter_mut() {
        for v in row.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        normalize_in_place(
            row.as_slice_mut().expect("rows of a standard-layout array"),
            Some(f64::INFINITY),
        );
    }
    Ok(pulse)
}

// =============================================================================
// Tempogram ratio, hybrid tempogram and metrogram
// =============================================================================

/// Linear interpolation of `(x, y)` at `xq` with scipy `interp1d(kind="linear",
/// bounds_error=False, fill_value=fill)` semantics. `x` must be sorted ascending and
/// have at least two points.
fn interp_linear(x: &[f64], y: &[f64], xq: f64, fill: f64) -> f64 {
    let n = x.len();
    if xq < x[0] || xq > x[n - 1] {
        return fill;
    }
    let idx = x.partition_point(|&v| v < xq).clamp(1, n - 1);
    let (lo, hi) = (idx - 1, idx);
    let slope = (y[hi] - y[lo]) / (x[hi] - x[lo]);
    slope * (xq - x[lo]) + y[lo]
}

/// Stable ascending sort permutation (numpy `argsort(kind="mergesort")`).
fn argsort_stable(x: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    idx
}

/// librosa's default metrical factors for [`TempogramRatioConfig`] (Prockup et al.
/// 2015): `[4, 8/3, 3, 2, 4/3, 3/2, 1, 2/3, 3/4, 1/2, 1/3, 3/8, 1/4]`.
#[must_use]
pub fn default_tempogram_ratio_factors() -> Vec<f64> {
    vec![
        4.0,
        8.0 / 3.0,
        3.0,
        2.0,
        4.0 / 3.0,
        3.0 / 2.0,
        1.0,
        2.0 / 3.0,
        3.0 / 4.0,
        1.0 / 2.0,
        1.0 / 3.0,
        3.0 / 8.0,
        1.0 / 4.0,
    ]
}

/// Configuration of the tempogram ratio feature, mirroring
/// `librosa.feature.tempogram_ratio`.
///
/// ## Purpose
///
/// The tempogram ratio samples the tempogram at fixed multiples (`factors`) of the
/// per-frame tempo estimate, describing the metrical structure (e.g. strong energy at
/// half or double tempo) independently of the tempo itself.
///
/// ## Intended usage
///
/// Pass to [`AudioRhythm::tempogram_ratio`] or [`tempogram_ratio_from_tempogram`].
/// `Default` reproduces librosa: default tempogram, per-frame tempo with the default
/// prior (`start_bpm = 120`, `std_bpm = 1`, `max_tempo = 320`), Prockup's 13 factors,
/// linear interpolation filled with 0, no time aggregation.
///
/// ## Invariants
///
/// Checked by [`TempogramRatioConfig::validate`]: the tempogram and tempo configs are
/// valid, `factors` is non-empty and finite.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::rhythm::TempogramRatioConfig;
///
/// assert_eq!(TempogramRatioConfig::default().factors.len(), 13);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TempogramRatioConfig {
    /// Tempogram parameters (also the onset envelope parameters).
    pub tempogram: TempogramConfig,
    /// Per-frame tempo parameters: `start_bpm`, `std_bpm`, `max_tempo` and `prior` are
    /// used; `onset`, `ac_size` and `aggregate` are ignored (the tempo is estimated per
    /// frame from the tempogram itself, as librosa does).
    pub tempo: TempoConfig,
    /// Multiples of the tempo at which the tempogram is sampled.
    pub factors: Vec<f64>,
    /// Optional aggregation over time. librosa default: `None`.
    pub aggregate: Option<Aggregate>,
    /// Value for tempi outside the tempogram's BPM range. librosa default: 0.
    pub fill_value: f64,
}

impl Default for TempogramRatioConfig {
    fn default() -> Self {
        Self {
            tempogram: TempogramConfig::default(),
            tempo: TempoConfig::default(),
            factors: default_tempogram_ratio_factors(),
            aggregate: None,
            fill_value: 0.0,
        }
    }
}

impl TempogramRatioConfig {
    /// Returns the librosa-default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the factors.
    #[must_use]
    pub fn with_factors(mut self, factors: Vec<f64>) -> Self {
        self.factors = factors;
        self
    }

    /// Sets the time aggregation.
    #[must_use]
    pub fn with_aggregate(mut self, aggregate: Option<Aggregate>) -> Self {
        self.aggregate = aggregate;
        self
    }

    fn validate_ratio(&self) -> AudioSampleResult<()> {
        self.tempo.validate_tempo()?;
        if self.factors.is_empty() || self.factors.iter().any(|f| !f.is_finite()) {
            return Err(param_err("factors", "factors must be non-empty and finite"));
        }
        Ok(())
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the tempogram or tempo configuration is invalid or
    /// `factors` is empty or not finite.
    pub fn validate(&self, sample_rate: f64) -> AudioSampleResult<()> {
        self.tempogram.validate(sample_rate)?;
        self.validate_ratio()
    }
}

/// Computes the tempogram ratio from a tempogram, equivalent to
/// `librosa.feature.tempogram_ratio(tg=tempogram, bpm=bpm)`.
///
/// # Arguments
///
/// * `tempogram` - `(channels, win_length, n_frames)` autocorrelation tempogram; its
///   BPM grid is [`tempo_frequencies`] (the infinite zero-lag bin is not used).
/// * `sample_rate`, `hop_length` - Frame geometry.
/// * `config` - Factors, aggregation, fill value and tempo-estimation parameters.
/// * `bpm` - Optional `(channels, n_frames)` per-frame tempo; `None` estimates it with
///   [`tempo_from_tempogram`] (`aggregate = None`).
///
/// # Returns
///
/// `(channels, n_factors, n_frames)`, or `(channels, n_factors, 1)` when
/// `config.aggregate` is set (librosa drops the time axis instead).
///
/// # Errors
///
/// [`ParameterError`] if a parameter is invalid, the tempogram has fewer than three
/// lag bins, or `bpm` has the wrong shape.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::rhythm::{
///     TempogramConfig, TempogramRatioConfig, tempogram_from_envelope, tempogram_ratio_from_tempogram,
/// };
/// use ndarray::Array2;
///
/// let env = Array2::from_shape_fn((1, 300), |(_, t)| if t % 22 == 0 { 1.0 } else { 0.0 });
/// let tg = tempogram_from_envelope(env.view(), &TempogramConfig::default()).unwrap();
/// let r = tempogram_ratio_from_tempogram(tg.view(), 22050.0, nzu!(512), &TempogramRatioConfig::default(), None)
///     .unwrap();
/// assert_eq!(r.dim(), (1, 13, 300));
/// ```
///
/// # References
///
/// M. Prockup, A. F. Ehmann, F. Gouyon, E. M. Schmidt and Y. E. Kim, "Modeling
/// musical rhythm at scale with the music genome project," WASPAA 2015; librosa
/// `feature.tempogram_ratio`.
pub fn tempogram_ratio_from_tempogram(
    tempogram: ArrayView3<'_, f64>,
    sample_rate: f64,
    hop_length: NonZeroUsize,
    config: &TempogramRatioConfig,
    bpm: Option<ArrayView2<'_, f64>>,
) -> AudioSampleResult<Array3<f64>> {
    config.validate_ratio()?;
    let (n_ch, win, n_t) = tempogram.dim();
    if win < 3 || n_t == 0 || n_ch == 0 {
        return Err(param_err(
            "tg",
            "the tempogram needs at least 3 lag bins and one frame",
        ));
    }
    let estimated;
    let bpm = match bpm {
        Some(b) => {
            if b.dim() != (n_ch, n_t) {
                return Err(param_err(
                    "bpm",
                    format!(
                        "bpm shape {:?} must be (channels, frames) = {:?}",
                        b.dim(),
                        (n_ch, n_t)
                    ),
                ));
            }
            b
        }
        None => {
            let cfg = TempoConfig {
                aggregate: None,
                ..config.tempo.clone()
            };
            estimated = tempo_from_tempogram(tempogram, sample_rate, hop_length, &cfg)?;
            estimated.view()
        }
    };
    let freqs = tempo_frequencies(win, hop_length, sample_rate);
    // Finite bins (k >= 1) sorted ascending.
    let order: Vec<usize> = {
        let finite: Vec<usize> = (0..win).filter(|&k| freqs[k].is_finite()).collect();
        let vals: Vec<f64> = finite.iter().map(|&k| freqs[k]).collect();
        argsort_stable(&vals)
            .into_iter()
            .map(|i| finite[i])
            .collect()
    };
    let xs: Vec<f64> = order.iter().map(|&k| freqs[k]).collect();
    let n_h = config.factors.len();
    let mut tgr = Array3::<f64>::zeros((n_ch, n_h, n_t));
    let mut ys = vec![0.0; order.len()];
    for c in 0..n_ch {
        for t in 0..n_t {
            for (y, &k) in ys.iter_mut().zip(&order) {
                *y = tempogram[[c, k, t]];
            }
            let b = bpm[[c, t]];
            for (h, &factor) in config.factors.iter().enumerate() {
                tgr[[c, h, t]] = interp_linear(&xs, &ys, b * factor, config.fill_value);
            }
        }
    }
    match config.aggregate {
        None => Ok(tgr),
        Some(agg) => {
            let mut out = Array3::<f64>::zeros((n_ch, n_h, 1));
            let mut series = Vec::with_capacity(n_t);
            let mut scratch = Vec::new();
            for c in 0..n_ch {
                for h in 0..n_h {
                    series.clear();
                    series.extend(tgr.slice(ndarray::s![c, h, ..]).iter().copied());
                    out[[c, h, 0]] = agg.reduce(&series, &mut scratch, true);
                }
            }
            Ok(out)
        }
    }
}

/// Configuration of the hybrid tempogram, mirroring `librosa.feature.hybrid_tempogram`.
///
/// ## Purpose
///
/// The hybrid tempogram is the geometric mean of the Fourier tempogram magnitude and
/// the autocorrelation tempogram (linearly interpolated onto the Fourier tempo grid):
/// the Fourier tempogram emphasises tempo harmonics, the autocorrelation tempogram
/// subharmonics, and their product suppresses both.
///
/// ## Intended usage
///
/// Pass to [`AudioRhythm::hybrid_tempogram`] or [`hybrid_tempogram_from_envelope`].
/// `Default` reproduces librosa: default onset strength, `win_length = 384`, centred,
/// Hann window, interpolation filled with 0 outside the autocorrelation range.
///
/// ## Invariants
///
/// Checked by [`HybridTempogramConfig::validate`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::rhythm::HybridTempogramConfig;
///
/// assert!(HybridTempogramConfig::default().center);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HybridTempogramConfig {
    /// Onset envelope parameters (also define the hop length).
    pub onset: OnsetStrengthConfig,
    /// Window length in frames of both tempograms. librosa default: 384.
    pub win_length: NonZeroUsize,
    /// Centred frames for both tempograms. librosa default: `true`.
    pub center: bool,
    /// Window of both tempograms. librosa default: Hann.
    pub window: Window,
}

impl Default for HybridTempogramConfig {
    fn default() -> Self {
        Self {
            onset: OnsetStrengthConfig::default(),
            win_length: crate::nzu!(384),
            center: true,
            window: Window::Hann,
        }
    }
}

impl HybridTempogramConfig {
    /// Returns the librosa-default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the window length.
    #[must_use]
    pub fn with_win_length(mut self, win_length: NonZeroUsize) -> Self {
        self.win_length = win_length;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the onset configuration or window is invalid.
    pub fn validate(&self, sample_rate: f64) -> AudioSampleResult<()> {
        self.onset.validate(sample_rate)?;
        get_window(&self.window, self.win_length.get(), true).map(|_| ())
    }
}

/// Computes the hybrid tempogram of onset envelopes, equivalent to
/// `librosa.feature.hybrid_tempogram(onset_envelope=envelope)`.
///
/// # Arguments
///
/// * `envelope` - `(channels, n_frames)` onset envelopes.
/// * `sample_rate`, `hop_length` - Frame geometry.
/// * `config` - Window length, centring and window (`config.onset` is not used).
///
/// # Returns
///
/// `(channels, win_length / 2 + 1, n)` on the [`fourier_tempo_frequencies`] grid,
/// where `n` is the smaller frame count of the two tempograms.
///
/// # Errors
///
/// [`ParameterError`] if the envelope is empty or too short, `win_length < 3`, or the
/// window is invalid.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::rhythm::{HybridTempogramConfig, hybrid_tempogram_from_envelope};
/// use ndarray::Array2;
///
/// let env = Array2::from_shape_fn((1, 300), |(_, t)| if t % 22 == 0 { 1.0 } else { 0.0 });
/// let h = hybrid_tempogram_from_envelope(env.view(), 22050.0, nzu!(512), &HybridTempogramConfig::default())
///     .unwrap();
/// assert_eq!(h.dim(), (1, 193, 300));
/// ```
pub fn hybrid_tempogram_from_envelope(
    envelope: ArrayView2<'_, f64>,
    sample_rate: f64,
    hop_length: NonZeroUsize,
    config: &HybridTempogramConfig,
) -> AudioSampleResult<Array3<f64>> {
    require_positive("sample_rate", sample_rate)?;
    let win = config.win_length;
    if win.get() < 3 {
        return Err(param_err("win_length", "win_length must be at least 3"));
    }
    let ft = fourier_tempogram_from_envelope(
        envelope,
        &FourierTempogramConfig {
            win_length: win,
            center: config.center,
            window: config.window.clone(),
            ..FourierTempogramConfig::default()
        },
    )?;
    let tg = tempogram_from_envelope(
        envelope,
        &TempogramConfig {
            win_length: win,
            center: config.center,
            window: config.window.clone(),
            ..TempogramConfig::default()
        },
    )?;
    let freqs = fourier_tempo_frequencies(sample_rate, hop_length, win);
    let lags = tempo_frequencies(win.get(), hop_length, sample_rate);
    let finite: Vec<usize> = (1..win.get()).collect();
    let vals: Vec<f64> = finite.iter().map(|&k| lags[k]).collect();
    let order: Vec<usize> = argsort_stable(&vals)
        .into_iter()
        .map(|i| finite[i])
        .collect();
    let xs: Vec<f64> = order.iter().map(|&k| lags[k]).collect();
    let (n_ch, n_bins, n_f) = ft.dim();
    let n = n_f.min(tg.dim().2);
    let mut out = Array3::<f64>::zeros((n_ch, n_bins, n));
    let mut ys = vec![0.0; order.len()];
    for c in 0..n_ch {
        for t in 0..n {
            for (y, &k) in ys.iter_mut().zip(&order) {
                *y = tg[[c, k, t]];
            }
            for (k, &f) in freqs.iter().enumerate() {
                let a = interp_linear(&xs, &ys, f, 0.0);
                let p = ft[[c, k, t]].norm() * a.abs();
                out[[c, k, t]] = (if p < 0.0 { 0.0 } else { p }).sqrt();
            }
        }
    }
    Ok(out)
}

/// Computes a metrogram (metrical-subharmonic salience) from a tempogram, equivalent
/// to `librosa.feature.metrogram(tg=tempogram, freqs=freqs, factors=factors,
/// aggregate=...)`.
///
/// For each factor `h`, the tempogram is linearly interpolated at `h * freqs` (filled
/// with `fill_value` outside the grid), multiplied by the tempogram, and aggregated
/// over the tempo axis.
///
/// # Arguments
///
/// * `tempogram` - `(channels, n_freqs, n_frames)` tempogram (magnitude).
/// * `freqs` - Tempo (BPM) of each tempogram bin, any order, at least two values. An
///   infinite value (the zero-lag bin of [`tempo_frequencies`]) yields NaN results,
///   as in librosa; drop it first.
/// * `factors` - Subharmonic factors; `None` uses librosa's `[1/3, 1/4, 1/5, 1/7]`.
/// * `aggregate` - Reduction over the tempo axis; librosa's default is
///   [`Aggregate::Sum`].
/// * `fill_value` - Value outside the frequency grid. librosa default: 0.
///
/// # Returns
///
/// `(channels, n_factors, n_frames)`.
///
/// # Errors
///
/// [`ParameterError`] if `freqs` does not match the tempogram or has fewer than two
/// values, or `factors` is empty.
///
/// # Differences from librosa
///
/// librosa's `aggregate=None` (returning the 4-D per-bin product) is not offered.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::rhythm::{Aggregate, metrogram};
/// use ndarray::Array3;
///
/// let freqs: Vec<f64> = (1..=60).map(|k| 5.0 * k as f64).collect();
/// let tg = Array3::from_shape_fn((1, 60, 4), |(_, k, _)| if k == 23 || k == 5 { 1.0 } else { 0.0 });
/// let m = metrogram(tg.view(), &freqs, None, Aggregate::Sum, 0.0).unwrap();
/// assert_eq!(m.dim(), (1, 4, 4));
/// // 120 BPM (bin 23) has its quarter, 30 BPM (bin 5), in the tempogram.
/// assert!(m[[0, 1, 0]] > 0.9);
/// ```
pub fn metrogram(
    tempogram: ArrayView3<'_, f64>,
    freqs: &[f64],
    factors: Option<&[f64]>,
    aggregate: Aggregate,
    fill_value: f64,
) -> AudioSampleResult<Array3<f64>> {
    let default_factors = [1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0, 1.0 / 7.0];
    let factors = factors.unwrap_or(&default_factors);
    let (n_ch, n_f, n_t) = tempogram.dim();
    if freqs.len() != n_f || n_f < 2 {
        return Err(param_err(
            "freqs",
            format!(
                "freqs must have one value per tempogram bin ({n_f}, at least 2), got {}",
                freqs.len()
            ),
        ));
    }
    if factors.is_empty() {
        return Err(param_err("factors", "factors must not be empty"));
    }
    let order = argsort_stable(freqs);
    let xs: Vec<f64> = order.iter().map(|&k| freqs[k]).collect();
    let mut ys = vec![0.0; n_f];
    let mut prod = vec![0.0; n_f];
    let mut scratch = Vec::new();
    let mut out = Array3::<f64>::zeros((n_ch, factors.len(), n_t));
    for c in 0..n_ch {
        for t in 0..n_t {
            for (y, &k) in ys.iter_mut().zip(&order) {
                *y = tempogram[[c, k, t]];
            }
            for (h, &factor) in factors.iter().enumerate() {
                for (f, p) in prod.iter_mut().enumerate() {
                    *p = interp_linear(&xs, &ys, factor * freqs[f], fill_value)
                        * tempogram[[c, f, t]];
                }
                out[[c, h, t]] = aggregate.reduce(&prod, &mut scratch, false);
            }
        }
    }
    Ok(out)
}

// =============================================================================
// Trait
// =============================================================================

/// librosa-compatible rhythm analysis of audio: onset strength and detection,
/// tempograms, tempo, beat tracking and predominant local pulse.
///
/// # Purpose
///
/// Reproduces `librosa.onset`, `librosa.feature.{tempogram, fourier_tempogram, tempo}`
/// and `librosa.beat` on [`AudioSamples`]. Every config's `Default` is librosa's
/// default, so results match librosa 1.0 for the same signal.
///
/// # Intended usage
///
/// Frame-wise analyses broadcast over channels and return
/// `(channels, ..., n_frames)` arrays. Event detectors ([`onset_detect`],
/// [`beat_track`]) need mono input. To reuse an onset envelope across analyses,
/// compute it once with [`onset_strength`] and call the `*_from_envelope` free
/// functions of [`crate::operations::rhythm`].
///
/// # Invariants
///
/// Input samples are converted to `f64` with the crate's audio-aware conversion;
/// all outputs are `f64`. The hop length of every output is the onset config's mel
/// STFT hop (512 by default).
///
/// [`onset_detect`]: AudioRhythm::onset_detect
/// [`beat_track`]: AudioRhythm::beat_track
/// [`onset_strength`]: AudioRhythm::onset_strength
pub trait AudioRhythm {
    /// Computes the onset strength envelope, equivalent to
    /// `librosa.onset.onset_strength(y=y, sr=sr)`.
    ///
    /// # Arguments
    ///
    /// * `config` - Envelope parameters ([`OnsetStrengthConfig::default`] = librosa).
    ///
    /// # Returns
    ///
    /// A `(channels, n_frames)` envelope with `n_frames = 1 + len / hop` (centred).
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
    ///
    /// [`ParameterError`] if the configuration is invalid or `aggregate` is `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::rhythm::{AudioRhythm, OnsetStrengthConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(22050), 0.5);
    /// let env = audio.onset_strength(&OnsetStrengthConfig::default()).unwrap();
    /// assert_eq!(env.dim(), (1, 44));
    /// ```
    fn onset_strength(&self, config: &OnsetStrengthConfig) -> AudioSampleResult<Array2<f64>>;

    /// Computes onset strength per frequency sub-band, equivalent to
    /// `librosa.onset.onset_strength_multi(y=y, sr=sr, channels=channels)`.
    ///
    /// # Arguments
    ///
    /// * `config` - Envelope parameters. `aggregate = None` returns one row per mel bin.
    /// * `channels` - Sub-band boundaries in mel bins (see
    ///   [`onset_strength_multi_from_db`]); `None` gives one band.
    ///
    /// # Returns
    ///
    /// A `(channels, n_bands, n_frames)` array.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
    ///
    /// [`ParameterError`] if the configuration or boundaries are invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::rhythm::{AudioRhythm, OnsetStrengthConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(22050), 0.5);
    /// let env = audio
    ///     .onset_strength_multi(&OnsetStrengthConfig::default(), Some(&[0, 32, 64, 128]))
    ///     .unwrap();
    /// assert_eq!(env.dim(), (1, 3, 44));
    /// ```
    fn onset_strength_multi(
        &self,
        config: &OnsetStrengthConfig,
        channels: Option<&[usize]>,
    ) -> AudioSampleResult<Array3<f64>>;

    /// Detects onsets, equivalent to `librosa.onset.onset_detect(y=y, sr=sr)`.
    ///
    /// # Arguments
    ///
    /// * `config` - Envelope, normalisation, peak-picking and backtracking parameters.
    ///
    /// # Returns
    ///
    /// The onset frames (use [`FrameEvents::times`] / [`FrameEvents::samples`] for
    /// librosa's other `units`).
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
    ///
    /// - [`LayoutError::ChannelCountUnsupported`] if the audio is not mono.
    /// - [`ParameterError`] if the configuration is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::rhythm::{AudioRhythm, OnsetDetectConfig};
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let mut y = Array1::<f32>::zeros(22050);
    /// for n in 11025..11225 {
    ///     y[n] = 0.5;
    /// }
    /// let audio = AudioSamples::new_mono(y, sample_rate!(22050)).unwrap();
    /// let onsets = audio.onset_detect(&OnsetDetectConfig::default()).unwrap();
    /// assert!(onsets.times().iter().any(|t| (t - 0.5).abs() < 0.05));
    /// ```
    fn onset_detect(&self, config: &OnsetDetectConfig) -> AudioSampleResult<FrameEvents>;

    /// Computes the autocorrelation tempogram, equivalent to
    /// `librosa.feature.tempogram(y=y, sr=sr)`.
    ///
    /// # Returns
    ///
    /// A `(channels, win_length, n_frames)` array.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
    ///
    /// [`ParameterError`] if the configuration is invalid or the signal is too short
    /// for an uncentred window.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::rhythm::{AudioRhythm, TempogramConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(22050), 0.5);
    /// let tg = audio.tempogram(&TempogramConfig::default()).unwrap();
    /// assert_eq!(tg.dim(), (1, 384, 44));
    /// ```
    fn tempogram(&self, config: &TempogramConfig) -> AudioSampleResult<Array3<f64>>;

    /// Computes the Fourier tempogram, equivalent to
    /// `librosa.feature.fourier_tempogram(y=y, sr=sr)`.
    ///
    /// # Returns
    ///
    /// A complex `(channels, win_length / 2 + 1, n_frames + 1)` array (centred).
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
    /// use audio_samples::operations::rhythm::{AudioRhythm, FourierTempogramConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(22050), 0.5);
    /// let ft = audio.fourier_tempogram(&FourierTempogramConfig::default()).unwrap();
    /// assert_eq!(ft.dim(), (1, 193, 45));
    /// ```
    fn fourier_tempogram(
        &self,
        config: &FourierTempogramConfig,
    ) -> AudioSampleResult<Array3<Complex<f64>>>;

    /// Estimates the tempo, equivalent to `librosa.feature.tempo(y=y, sr=sr)`.
    ///
    /// # Returns
    ///
    /// `(channels, 1)` tempi in BPM, or `(channels, n_frames)` with
    /// `config.aggregate = None`.
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
    /// use audio_samples::operations::rhythm::{AudioRhythm, TempoConfig};
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let sr = 22050;
    /// let mut y = Array1::<f32>::zeros(6 * sr);
    /// for k in 0..12 {
    ///     for n in 0..100 {
    ///         y[k * sr / 2 + n] = 0.8;
    ///     }
    /// }
    /// let audio = AudioSamples::new_mono(y, sample_rate!(22050)).unwrap();
    /// let bpm = audio.tempo(&TempoConfig::default()).unwrap();
    /// assert!((bpm[[0, 0]] - 120.0).abs() < 5.0);
    /// ```
    fn tempo(&self, config: &TempoConfig) -> AudioSampleResult<Array2<f64>>;

    /// Tracks beats, equivalent to `librosa.beat.beat_track(y=y, sr=sr)`.
    ///
    /// # Returns
    ///
    /// The tempo and beat frames ([`BeatTrack`]).
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
    ///
    /// - [`LayoutError::ChannelCountUnsupported`] if the audio is not mono.
    /// - [`ParameterError`] if the configuration is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::rhythm::{AudioRhythm, BeatTrackConfig};
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let sr = 22050;
    /// let mut y = Array1::<f32>::zeros(6 * sr);
    /// for k in 0..12 {
    ///     for n in 0..100 {
    ///         y[k * sr / 2 + n] = 0.8;
    ///     }
    /// }
    /// let audio = AudioSamples::new_mono(y, sample_rate!(22050)).unwrap();
    /// let bt = audio.beat_track(&BeatTrackConfig::default()).unwrap();
    /// assert!(bt.beats.len() >= 8);
    /// ```
    fn beat_track(&self, config: &BeatTrackConfig) -> AudioSampleResult<BeatTrack>;

    /// Computes the predominant local pulse, equivalent to
    /// `librosa.beat.plp(y=y, sr=sr)`.
    ///
    /// # Returns
    ///
    /// A `(channels, n_frames)` pulse curve in `[0, 1]`.
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
    /// use audio_samples::operations::rhythm::{AudioRhythm, PlpConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(22050), 0.5);
    /// let pulse = audio.plp(&PlpConfig::default()).unwrap();
    /// assert_eq!(pulse.dim(), (1, 44));
    /// ```
    fn plp(&self, config: &PlpConfig) -> AudioSampleResult<Array2<f64>>;

    /// Computes the tempogram ratio, equivalent to
    /// `librosa.feature.tempogram_ratio(y=y, sr=sr)`.
    ///
    /// # Returns
    ///
    /// `(channels, n_factors, n_frames)` (or `(channels, n_factors, 1)` aggregated).
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
    /// use audio_samples::operations::rhythm::{AudioRhythm, TempogramRatioConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(22050), 0.5);
    /// let r = audio.tempogram_ratio(&TempogramRatioConfig::default()).unwrap();
    /// assert_eq!(r.dim(), (1, 13, 44));
    /// ```
    fn tempogram_ratio(&self, config: &TempogramRatioConfig) -> AudioSampleResult<Array3<f64>>;

    /// Computes the hybrid tempogram, equivalent to
    /// `librosa.feature.hybrid_tempogram(y=y, sr=sr)`.
    ///
    /// # Returns
    ///
    /// `(channels, win_length / 2 + 1, n_frames)`.
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
    /// use audio_samples::operations::rhythm::{AudioRhythm, HybridTempogramConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(22050), 0.5);
    /// let h = audio.hybrid_tempogram(&HybridTempogramConfig::default()).unwrap();
    /// assert_eq!(h.dim(), (1, 193, 44));
    /// ```
    fn hybrid_tempogram(&self, config: &HybridTempogramConfig) -> AudioSampleResult<Array3<f64>>;
}

fn require_mono<T: StandardSample>(audio: &AudioSamples<'_, T>, op: &str) -> AudioSampleResult<()> {
    let n = audio.num_channels().get();
    if n == 1 {
        Ok(())
    } else {
        Err(AudioSampleError::Layout(
            LayoutError::channel_count_unsupported(op, ChannelRequirement::Mono, n),
        ))
    }
}

fn envelope_of<T: StandardSample>(
    audio: &AudioSamples<'_, T>,
    config: &OnsetStrengthConfig,
) -> AudioSampleResult<Array2<f64>> {
    config.validate(audio.sample_rate_hz())?;
    // TRANSITIONAL(f64): f64 rows until this module computes in T::Float.
    let mel = melspectrogram_rows(
        finite_audio_rows(audio)?.view(),
        audio.sample_rate_hz(),
        &config.mel,
    )?;
    let s_db = power_to_db(&mel, &config.db)?;
    onset_strength_from_db(s_db.view(), config)
}

impl<T> AudioRhythm for AudioSamples<'_, T>
where
    T: StandardSample,
{
    fn onset_strength(&self, config: &OnsetStrengthConfig) -> AudioSampleResult<Array2<f64>> {
        envelope_of(self, config)
    }

    fn onset_strength_multi(
        &self,
        config: &OnsetStrengthConfig,
        channels: Option<&[usize]>,
    ) -> AudioSampleResult<Array3<f64>> {
        config.validate(self.sample_rate_hz())?;
        // TRANSITIONAL(f64): f64 rows until this module computes in T::Float.
        let mel = melspectrogram_rows(
            finite_audio_rows(self)?.view(),
            self.sample_rate_hz(),
            &config.mel,
        )?;
        let s_db = power_to_db(&mel, &config.db)?;
        onset_strength_multi_from_db(s_db.view(), None, config, channels)
    }

    fn onset_detect(&self, config: &OnsetDetectConfig) -> AudioSampleResult<FrameEvents> {
        require_mono(self, "onset_detect")?;
        config.validate(self.sample_rate_hz())?;
        let env = envelope_of(self, &config.onset)?;
        let hop = config.onset.hop_length()?;
        let sr = self.sample_rate_hz();
        let row = env.row(0).to_vec();
        let frames = onset_detect_from_envelope(&row, sr, hop, config, None)?;
        Ok(FrameEvents {
            frames,
            sample_rate: sr,
            hop_length: hop,
        })
    }

    fn tempogram(&self, config: &TempogramConfig) -> AudioSampleResult<Array3<f64>> {
        config.validate(self.sample_rate_hz())?;
        let env = envelope_of(self, &config.onset)?;
        tempogram_from_envelope(env.view(), config)
    }

    fn fourier_tempogram(
        &self,
        config: &FourierTempogramConfig,
    ) -> AudioSampleResult<Array3<Complex<f64>>> {
        config.validate(self.sample_rate_hz())?;
        let env = envelope_of(self, &config.onset)?;
        fourier_tempogram_from_envelope(env.view(), config)
    }

    fn tempo(&self, config: &TempoConfig) -> AudioSampleResult<Array2<f64>> {
        config.validate(self.sample_rate_hz())?;
        let env = envelope_of(self, &config.onset)?;
        tempo_from_envelope(
            env.view(),
            self.sample_rate_hz(),
            config.onset.hop_length()?,
            config,
        )
    }

    fn beat_track(&self, config: &BeatTrackConfig) -> AudioSampleResult<BeatTrack> {
        require_mono(self, "beat_track")?;
        config.validate(self.sample_rate_hz())?;
        let env = envelope_of(self, &config.onset)?;
        let row = env.row(0).to_vec();
        beat_track_from_envelope(
            &row,
            self.sample_rate_hz(),
            config.onset.hop_length()?,
            config,
        )
    }

    fn plp(&self, config: &PlpConfig) -> AudioSampleResult<Array2<f64>> {
        config.validate(self.sample_rate_hz())?;
        let env = envelope_of(self, &config.onset)?;
        plp_from_envelope(
            env.view(),
            self.sample_rate_hz(),
            config.onset.hop_length()?,
            config,
        )
    }

    fn tempogram_ratio(&self, config: &TempogramRatioConfig) -> AudioSampleResult<Array3<f64>> {
        config.validate(self.sample_rate_hz())?;
        let tg = self.tempogram(&config.tempogram)?;
        tempogram_ratio_from_tempogram(
            tg.view(),
            self.sample_rate_hz(),
            config.tempogram.onset.hop_length()?,
            config,
            None,
        )
    }

    fn hybrid_tempogram(&self, config: &HybridTempogramConfig) -> AudioSampleResult<Array3<f64>> {
        config.validate(self.sample_rate_hz())?;
        let env = envelope_of(self, &config.onset)?;
        hybrid_tempogram_from_envelope(
            env.view(),
            self.sample_rate_hz(),
            config.onset.hop_length()?,
            config,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample_rate;
    use ndarray::{Array1, array};

    #[test]
    fn huge_max_filter_equals_one_period() {
        let s = Array2::from_shape_fn((7, 5), |(i, t)| ((i * 3 + t * 5) % 11) as f64);
        let col_max: Vec<f64> = s
            .columns()
            .into_iter()
            .map(|c| c.iter().copied().fold(f64::NEG_INFINITY, f64::max))
            .collect();
        for size in [14, 15, 1000, 1 << 40] {
            let m = maximum_filter_freq(s.view(), size);
            for ((_, t), &v) in m.indexed_iter() {
                assert_eq!(v, col_max[t], "size {size}");
            }
        }
    }

    #[test]
    fn beat_track_with_huge_frame_rate_is_parameter_error() {
        let env: Vec<f64> = (0..300)
            .map(|t| if t % 22 == 3 { 1.0 } else { 0.05 })
            .collect();
        assert!(matches!(
            beat_track_with_bpm(&env, &[120.0], 1e12, 100.0, true),
            Err(AudioSampleError::Parameter(_))
        ));
        assert!(beat_track_with_bpm(&env, &[120.0], 43.0, 100.0, true).is_ok());
    }

    fn clicks(sr: usize, secs: usize, period: usize) -> AudioSamples<'static, f64> {
        let mut y = Array1::<f64>::zeros(secs * sr);
        let mut k = sr / 4;
        while k + 100 < y.len() {
            for n in 0..100 {
                y[k + n] = 0.8 * (-(n as f64) / 20.0).exp();
            }
            k += period;
        }
        AudioSamples::new_mono(y, sample_rate!(22050)).unwrap()
    }

    #[test]
    fn pairwise_sum_matches_naive_closely() {
        let x: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.37).sin()).collect();
        assert!((pairwise_sum(&x) - sequential_sum(&x)).abs() < 1e-12);
        assert_eq!(pairwise_sum(&[1.0, 2.0, 3.0]), 6.0);
    }

    #[test]
    fn helpers() {
        assert_eq!(next_fast_len(767), 768);
        assert_eq!(next_fast_len(7), 8);
        assert_eq!(py_floordiv(661.5, 512.0), 1.0);
        assert_eq!(py_floordiv(2205.0, 512.0), 4.0);
        assert_eq!(median_in_place(&mut [3.0, 1.0, 2.0, 4.0]), 2.5);
        let mut v = [3.0, -4.0];
        normalize_in_place(&mut v, Some(f64::INFINITY));
        assert_eq!(v, [0.75, -1.0]);
        let mut z = [0.0, 0.0];
        normalize_in_place(&mut z, Some(2.0));
        assert_eq!(z, [0.0, 0.0]);
    }

    #[test]
    fn onset_strength_shapes_mono_stereo() {
        let a = clicks(22050, 2, 11025);
        let env = a.onset_strength(&OnsetStrengthConfig::default()).unwrap();
        assert_eq!(env.dim(), (1, 87));
        assert!(env.iter().all(|v| *v >= 0.0));
        let data = ndarray::stack![
            Axis(0),
            a.as_mono().unwrap().as_view(),
            a.as_mono().unwrap().as_view()
        ];
        let st = AudioSamples::new_multi_channel(data, sample_rate!(22050)).unwrap();
        let env2 = st.onset_strength(&OnsetStrengthConfig::default()).unwrap();
        assert_eq!(env2.dim(), (2, 87));
        assert_eq!(env2.row(0), env.row(0));
        assert_eq!(env2.row(1), env.row(0));
        let nocenter = a
            .onset_strength(&OnsetStrengthConfig::default().with_center(false))
            .unwrap();
        assert_eq!(nocenter.dim(), (1, 87));
    }

    #[test]
    fn onset_strength_errors() {
        let a = clicks(22050, 1, 11025);
        let cfg = OnsetStrengthConfig::default().with_aggregate(None);
        assert!(a.onset_strength(&cfg).is_err());
        assert_eq!(
            a.onset_strength_multi(&cfg, None).unwrap().dim(),
            (1, 128, 44)
        );
        assert!(
            a.onset_strength_multi(&OnsetStrengthConfig::default(), Some(&[5]))
                .is_err()
        );
        let s = Array3::<f64>::zeros((1, 4, 5));
        let r = Array3::<f64>::zeros((1, 4, 6));
        assert!(
            onset_strength_multi_from_db(
                s.view(),
                Some(r.view()),
                &OnsetStrengthConfig::default(),
                None
            )
            .is_err()
        );
    }

    #[test]
    fn short_envelope_longer_lag() {
        let s = Array3::<f64>::ones((1, 3, 2));
        let cfg = OnsetStrengthConfig::default().with_lag(nzu!(5));
        let env = onset_strength_from_db(s.view(), &cfg).unwrap();
        assert_eq!(env.dim(), (1, 2));
        let env = onset_strength_from_db(s.view(), &cfg.with_center(false)).unwrap();
        assert_eq!(env.dim(), (1, 5));
    }

    #[test]
    fn peak_pick_edge_cases() {
        let cfg = PeakPickConfig::default();
        assert!(peak_pick(&[], &cfg).unwrap().is_empty());
        // 1.0 < mean(1.0) + 0.07, so a single sample is only a peak with delta = 0.
        assert!(peak_pick(&[1.0], &cfg).unwrap().is_empty());
        assert_eq!(
            peak_pick(&[1.0], &cfg.clone().with_delta(0.0)).unwrap(),
            vec![0]
        );
        assert!(peak_pick(&[f64::NAN], &cfg).is_err());
        let bad = PeakPickConfig { post_max: 0, ..cfg };
        assert!(peak_pick(&[1.0], &bad).is_err());
        let x = [0.0, 1.0, 0.0, 1.0, 0.0];
        let dp = cfg.with_method(PeakPickMethod::DpCount).with_wait(0);
        assert_eq!(peak_pick(&x, &dp).unwrap(), vec![1, 3]);
    }

    #[test]
    fn backtrack_edges() {
        assert!(onset_backtrack(&[], &[1.0, 2.0]).is_empty());
        assert_eq!(onset_backtrack(&[3], &[]), vec![0]);
    }

    #[test]
    fn event_detectors_require_mono() {
        let st =
            AudioSamples::new_multi_channel(Array2::<f64>::zeros((2, 4096)), sample_rate!(22050))
                .unwrap();
        let e = st.onset_detect(&OnsetDetectConfig::default()).unwrap_err();
        assert!(matches!(
            e,
            AudioSampleError::Layout(LayoutError::ChannelCountUnsupported { .. })
        ));
        assert!(st.beat_track(&BeatTrackConfig::default()).is_err());
        assert_eq!(st.tempo(&TempoConfig::default()).unwrap().dim(), (2, 1));
        assert_eq!(st.plp(&PlpConfig::default()).unwrap().dim(), (2, 9));
    }

    #[test]
    fn silence_gives_no_events() {
        let a = AudioSamples::new_mono(Array1::<f64>::zeros(22050), sample_rate!(22050)).unwrap();
        assert!(
            a.onset_detect(&OnsetDetectConfig::default())
                .unwrap()
                .is_empty()
        );
        let bt = a.beat_track(&BeatTrackConfig::default()).unwrap();
        assert_eq!(bt.tempo, 0.0);
        assert!(bt.beats.is_empty());
    }

    #[test]
    fn clicks_tempo_and_beats() {
        // 120 BPM clicks.
        let a = clicks(22050, 6, 11025);
        let bpm = a.tempo(&TempoConfig::default()).unwrap()[[0, 0]];
        assert!((bpm - 120.0).abs() < 4.0, "{bpm}");
        let bt = a.beat_track(&BeatTrackConfig::default()).unwrap();
        let frames = &bt.beats.frames;
        assert!(frames.len() >= 8);
        for w in frames.windows(2) {
            assert!((20..=23).contains(&(w[1] - w[0])), "{frames:?}");
        }
        let onsets = a.onset_detect(&OnsetDetectConfig::default()).unwrap();
        assert!(onsets.len() >= 10);
        let tv = a
            .tempo(&TempoConfig::default().with_aggregate(None))
            .unwrap();
        assert_eq!(
            tv.ncols(),
            a.onset_strength(&OnsetStrengthConfig::default())
                .unwrap()
                .ncols()
        );
    }

    #[test]
    fn beat_tracker_parameter_errors() {
        let env = [1.0, 0.0, 1.0, 0.0];
        assert!(beat_track_with_bpm(&env, &[], 43.0, 100.0, true).is_err());
        assert!(beat_track_with_bpm(&env, &[120.0, 120.0], 43.0, 100.0, true).is_err());
        assert!(beat_track_with_bpm(&env, &[-1.0], 43.0, 100.0, true).is_err());
        assert!(beat_track_with_bpm(&env, &[120.0], 43.0, 0.0, true).is_err());
        assert!(beat_track_with_bpm(&env, &[1e6], 43.0, 100.0, true).is_err());
        assert!(
            beat_track_with_bpm(&[], &[120.0], 43.0, 100.0, true)
                .unwrap()
                .is_empty()
        );
        // Single frame: no panic.
        let _ = beat_track_with_bpm(&[1.0], &[120.0], 43.0, 100.0, true).unwrap();
    }

    #[test]
    fn tempogram_uncentred_and_too_short() {
        let env = Array2::from_shape_fn((1, 50), |(_, t)| (t % 5 == 0) as u8 as f64);
        let cfg = TempogramConfig::new()
            .with_win_length(nzu!(20))
            .with_center(false);
        assert_eq!(
            tempogram_from_envelope(env.view(), &cfg).unwrap().dim(),
            (1, 20, 31)
        );
        let cfg = cfg.with_win_length(nzu!(60));
        assert!(tempogram_from_envelope(env.view(), &cfg).is_err());
        assert!(
            tempogram_from_envelope(env.view(), &TempogramConfig::new().with_norm(Some(-1.0)))
                .is_err()
        );
    }

    #[test]
    fn tempo_and_plp_validation() {
        let env = array![[1.0, 0.0, 1.0]];
        let cfg = TempoConfig::new().with_start_bpm(0.0);
        assert!(tempo_from_envelope(env.view(), 22050.0, nzu!(512), &cfg).is_err());
        let cfg = TempoConfig::new().with_ac_size(0.001);
        assert!(tempo_from_envelope(env.view(), 22050.0, nzu!(512), &cfg).is_err());
        let p = PlpConfig::new().with_tempo_range(Some(200.0), Some(100.0));
        assert!(plp_from_envelope(env.view(), 22050.0, nzu!(512), &p).is_err());
    }

    #[test]
    fn interpolation_and_variants() {
        let x = [1.0, 2.0, 4.0];
        let y = [10.0, 20.0, 0.0];
        assert_eq!(interp_linear(&x, &y, 1.5, -1.0), 15.0);
        assert_eq!(interp_linear(&x, &y, 3.0, -1.0), 10.0);
        assert_eq!(interp_linear(&x, &y, 0.5, -1.0), -1.0);
        assert_eq!(interp_linear(&x, &y, 4.5, -1.0), -1.0);
        let tg = Array3::<f64>::ones((1, 3, 2));
        assert!(metrogram(tg.view(), &[1.0, 2.0], None, Aggregate::Sum, 0.0).is_err());
        assert!(metrogram(tg.view(), &[1.0, 2.0, 3.0], Some(&[]), Aggregate::Sum, 0.0).is_err());
        let cfg = TempogramRatioConfig::default().with_factors(vec![]);
        assert!(tempogram_ratio_from_tempogram(tg.view(), 22050.0, nzu!(512), &cfg, None).is_err());
        let bad_bpm = Array2::<f64>::ones((1, 5));
        assert!(
            tempogram_ratio_from_tempogram(
                tg.view(),
                22050.0,
                nzu!(512),
                &TempogramRatioConfig::default(),
                Some(bad_bpm.view())
            )
            .is_err()
        );
        let env = Array2::<f64>::ones((1, 10));
        let h = HybridTempogramConfig::new().with_win_length(nzu!(2));
        assert!(hybrid_tempogram_from_envelope(env.view(), 22050.0, nzu!(512), &h).is_err());
    }

    #[test]
    fn complex_lexicographic_max() {
        let v = [
            Complex::new(1.0, 0.0),
            Complex::new(-2.0, 5.0),
            Complex::new(3.0, -1.0),
            Complex::new(3.0, 2.0),
        ];
        assert_eq!(complex_max_lex(v.into_iter()), Complex::new(3.0, 2.0));
        let v = [Complex::new(0.0, 0.0), Complex::new(-2.0, 5.0)];
        assert_eq!(complex_max_lex(v.into_iter()), Complex::new(0.0, 0.0));
    }

    /// Regression: a huge `ac_size` (or tempogram `win_length`) allocated a
    /// `(win, n_frames)` tempogram of hundreds of GB and aborted.
    #[test]
    fn tempo_rejects_huge_autocorrelation_window() {
        let env = Array2::<f64>::ones((1, 400));
        let cfg = TempoConfig::new().with_ac_size(1e9);
        let r = tempo_from_envelope(env.view(), 22050.0, crate::nzu!(512), &cfg);
        assert!(matches!(r, Err(AudioSampleError::Parameter(_))));
        let tg = TempogramConfig::new().with_win_length(NonZeroUsize::new(1 << 40).unwrap());
        assert!(tempogram_from_envelope(env.view(), &tg).is_err());
    }

    /// Regression: `i + post_max`, `wait + 1` and `i += wait + 1` overflowed
    /// for huge values (`post_max = usize::MAX` panicked with an out-of-bounds
    /// index; `wait = usize::MAX` wrapped to `i += 0` and looped, pushing peaks
    /// until an allocation failure).
    #[test]
    fn peak_pick_huge_windows_do_not_overflow() {
        let x = [0.0, 1.0, 0.0, 0.5, 0.0, 2.0, 0.0];
        for method in [
            PeakPickMethod::Greedy,
            PeakPickMethod::DpCount,
            PeakPickMethod::DpValue,
        ] {
            let base = PeakPickConfig {
                pre_max: 1,
                post_max: 2,
                pre_avg: 1,
                post_avg: 2,
                delta: 0.0,
                wait: 0,
                method,
            };
            let huge_max = PeakPickConfig {
                pre_max: usize::MAX,
                post_max: usize::MAX,
                ..base.clone()
            };
            // Only the global maximum survives a whole-signal max window.
            assert_eq!(peak_pick(&x, &huge_max).unwrap(), vec![5], "{method:?}");
            let huge_avg = PeakPickConfig {
                pre_avg: usize::MAX,
                post_avg: usize::MAX,
                ..base.clone()
            };
            assert!(peak_pick(&x, &huge_avg).is_ok());
            let huge_wait = PeakPickConfig {
                wait: usize::MAX,
                ..base.clone()
            };
            // At most one peak with an unbounded wait.
            assert!(peak_pick(&x, &huge_wait).unwrap().len() <= 1, "{method:?}");
        }
        assert_eq!(PeakPickConfig::new(), PeakPickConfig::default());
    }
}
