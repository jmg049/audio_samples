//! Tonal analysis: chroma features, tuning estimation and spectral pitch
//! tracking.
//!
//! ## What
//!
//! This module provides the [`AudioTonal`] trait and the free functions it is
//! built from, an STFT-based tonal pipeline:
//!
//! | Operation | Items |
//! |---|---|
//! | Chroma from a power spectrogram | [`AudioTonal::chroma_stft`] ([`chroma_stft_from_spectrogram`]) |
//! | Chroma filterbank | [`chroma_filterbank`] |
//! | Pitch tracking by parabolic peak interpolation | [`AudioTonal::piptrack`] ([`piptrack_from_spectrogram`]) |
//! | Tuning estimation | [`AudioTonal::estimate_tuning`] ([`estimate_tuning_from_signal`], [`estimate_tuning_from_spectrogram`]) |
//! | Tuning of a set of frequencies | [`pitch_tuning`] |
//! | Normalisation along an axis | [`normalize_array`] |
//!
//! Tuning and pitch-notation helpers that do not need a spectrogram
//! (`hz_to_octs`, `A4_to_tuning`, `cqt_frequencies`, `key_to_notes`, ...) live
//! in [`crate::utils::audio_math`].
//!
//! ## Why
//!
//! [`AudioTransforms::chromagram`](crate::AudioTransforms::chromagram) delegates
//! to the `spectrograms` crate, whose chroma uses a different filterbank, a
//! magnitude spectrogram, L2 normalisation and no tuning estimation, so its
//! profiles are much flatter. The chroma here uses a power spectrogram, a
//! Gaussian filterbank with octave weighting, and max-norm frames, which key
//! detection, chord recognition and cover-song features expect. Every config
//! struct has a `librosa()` constructor. `Default` is the same except for
//! [`ChromaStftConfig`], whose default fixes the tuning at A440 instead of
//! estimating it.
//!
//! ## How
//!
//! Bring [`AudioTonal`] into scope and call it on any [`AudioSamples`]. Frame-wise
//! outputs are arrays shaped `(channels, n_features, n_frames)`; mono input
//! gives one channel. Tuning estimates pool all channels.
//!
//! The output precision follows the input: the element type is the sample
//! type's [`AudioSample::Float`](crate::AudioSample::Float). `u8`, `i16`, `I24`
//! and `f32` audio gives `f32` arrays; `i32` and `f64` audio gives `f64`. The
//! whole computation runs in that type. The chroma filterbank is computed in
//! `f64` and rounded to the output type once. Tuning estimates are `f64`
//! scalars. The free functions on arrays ([`normalize_array`],
//! [`chroma_stft_from_spectrogram`], [`piptrack_from_spectrogram`], ...) are
//! generic over the element type ([`AudioFloat`]: `f32` or `f64`).
//!
//! ```rust
//! use audio_samples::operations::tonal::{AudioTonal, ChromaStftConfig, TuningConfig};
//! use audio_samples::{AudioSamples, sample_rate, sine_wave};
//! use std::time::Duration;
//!
//! // A4 (440 Hz): the chroma peak is pitch class 9 (A, counting from C).
//! let audio = sine_wave::<f32>(440.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
//! let chroma = audio.chroma_stft(&ChromaStftConfig::default()).unwrap();
//! assert_eq!(chroma.dim(), (1, 12, 22));
//! assert_eq!(chroma[[0, 9, 10]], 1.0);
//!
//! // A pure A440 is in tune.
//! let tuning = audio.estimate_tuning(&TuningConfig::default()).unwrap();
//! assert!(tuning.abs() < 0.02);
//! ```

use std::num::NonZeroUsize;

use ndarray::{
    Array, Array2, Array3, ArrayBase, ArrayView2, ArrayView3, ArrayViewMut1, Axis, Data, Dimension,
};

use crate::audio_float::finite_float_rows;
use crate::operations::spectrum::{StftConfig, spectrogram_rows, stft_map};
use crate::utils::audio_math::hz_to_octs;
use crate::{
    AudioFloat, AudioSampleError, AudioSampleResult, AudioSamples, ParameterError, StandardSample,
};

fn param_err(parameter: &str, reason: impl ToString) -> AudioSampleError {
    AudioSampleError::Parameter(ParameterError::invalid_value(parameter, reason))
}

fn validate_sample_rate(sample_rate: f64) -> AudioSampleResult<()> {
    if sample_rate.is_finite() && sample_rate > 0.0 {
        Ok(())
    } else {
        Err(AudioSampleError::Parameter(ParameterError::out_of_range(
            "sample_rate",
            sample_rate,
            "> 0",
            "finite",
            "the sample rate must be finite and positive",
        )))
    }
}

/// Floating-point remainder whose result has the sign of the divisor.
fn np_remainder(a: f64, b: f64) -> f64 {
    let m = a % b;
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) { m + b } else { m }
    } else {
        0.0f64.copysign(b)
    }
}

// =============================================================================
// normalize
// =============================================================================

/// Parameters of [`normalize_array`].
///
/// Normalisation rescales each slice of an array along `axis` (each column of a
/// chromagram, each filter of a filterbank, ...) so that its norm is 1. It makes
/// features comparable across frames of different loudness. `Default` is
/// max-norm along axis 0, threshold = smallest
/// positive normal value of the array's element type, quiet slices left
/// unchanged.
///
/// ## Invariants
///
/// Checked by [`NormalizeConfig::validate`]: `norm` is `None`, `+inf`, `-inf`,
/// `0` or a finite `p > 0`; `threshold` (if set) is positive; `fill` is not
/// `Some(true)` with `norm == 0`; `axis` (if set) is in range for the array.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::tonal::NormalizeConfig;
///
/// // L2-normalise rows of a (rows, cols) array, zeroing rows with tiny norm.
/// let cfg = NormalizeConfig::new()
///     .with_norm(Some(2.0))
///     .with_axis(Some(1))
///     .with_threshold(Some(1e-8))
///     .with_fill(Some(false));
/// assert!(cfg.validate(2).is_ok());
/// assert!(cfg.validate(1).is_err()); // axis 1 does not exist in a 1-D array
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizeConfig {
    /// Which norm each slice is scaled to 1 in. `Some(f64::INFINITY)`
    /// (default): maximum absolute value. `Some(f64::NEG_INFINITY)`: minimum
    /// absolute value. `Some(0.0)`: number of non-zero entries. `Some(p)` with
    /// finite `p > 0`: the l_p norm (`Some(1.0)` sum of magnitudes, `Some(2.0)`
    /// Euclidean). `None`: no normalisation (the input is returned unchanged).
    pub norm: Option<f64>,
    /// Axis along which the norm is computed; negative values count from the end
    /// (`-1` is the last axis). `Some(0)` (default) normalises each
    /// column of a 2-D array. `None` normalises the whole array as one slice.
    pub axis: Option<isize>,
    /// Slices whose norm is below this value are treated as silent and handled
    /// according to `fill`. `None` (default) means the smallest positive normal
    /// value of the array's element type (`MIN_POSITIVE`: about `2.2e-308` for
    /// `f64`, `1.2e-38` for `f32`), so only all-zero slices count. Must be
    /// positive when set.
    pub threshold: Option<f64>,
    /// What happens to slices below `threshold`: `None` (default) leaves them
    /// unchanged, `Some(false)` sets them to zero, `Some(true)` fills them with
    /// the constant whose norm is 1 (1 for the max/min norms,
    /// `n^(-1/p)` for an l_p norm over `n` elements).
    pub fill: Option<bool>,
}

impl Default for NormalizeConfig {
    fn default() -> Self {
        Self {
            norm: Some(f64::INFINITY),
            axis: Some(0),
            threshold: None,
            fill: None,
        }
    }
}

impl NormalizeConfig {
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

    /// Sets the norm (`None` disables normalisation).
    #[must_use]
    pub fn with_norm(mut self, norm: Option<f64>) -> Self {
        self.norm = norm;
        self
    }

    /// Sets the axis (`None` = the whole array).
    #[must_use]
    pub fn with_axis(mut self, axis: Option<isize>) -> Self {
        self.axis = axis;
        self
    }

    /// Sets the small-norm threshold (`None` = smallest positive normal value
    /// of the element type).
    #[must_use]
    pub fn with_threshold(mut self, threshold: Option<f64>) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets how slices below the threshold are handled.
    #[must_use]
    pub fn with_fill(mut self, fill: Option<bool>) -> Self {
        self.fill = fill;
        self
    }

    /// Checks the configuration against an array of `ndim` dimensions and
    /// resolves the axis.
    ///
    /// # Returns
    ///
    /// The non-negative axis index, or `None` when the whole array is one slice.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `norm` is negative (other than `-inf`) or NaN, the
    /// threshold is not positive, `fill` is `Some(true)` with `norm == 0`, or
    /// `axis` is out of range for `ndim`.
    pub fn validate(&self, ndim: usize) -> AudioSampleResult<Option<usize>> {
        if let Some(p) = self.norm {
            let ok = p.is_infinite() || (p.is_finite() && p >= 0.0);
            if !ok {
                return Err(param_err(
                    "norm",
                    format!("unsupported norm {p}: use inf, -inf, 0 or a finite p > 0"),
                ));
            }
            if p == 0.0 && self.fill == Some(true) {
                return Err(param_err(
                    "fill",
                    "cannot normalize with norm=0 and fill=true",
                ));
            }
        }
        if let Some(t) = self.threshold
            && (t.is_nan() || t <= 0.0)
        {
            return Err(param_err(
                "threshold",
                format!("threshold={t} must be strictly positive"),
            ));
        }
        match self.axis {
            None => Ok(None),
            Some(a) => {
                let n = ndim as isize;
                let resolved = if a < 0 { a + n } else { a };
                if (0..n).contains(&resolved) {
                    Ok(Some(resolved as usize))
                } else {
                    Err(param_err(
                        "axis",
                        format!("axis {a} is out of range for a {ndim}-dimensional array"),
                    ))
                }
            }
        }
    }
}

/// The norm of one slice, from the magnitudes.
fn slice_norm<F: AudioFloat>(values: impl Iterator<Item = F>, norm: f64) -> F {
    if norm == f64::INFINITY {
        values.fold(F::neg_infinity(), |m, x| m.max(x.abs()))
    } else if norm == f64::NEG_INFINITY {
        values.fold(F::infinity(), |m, x| m.min(x.abs()))
    } else if norm == 0.0 {
        F::of_usize(values.filter(|x| x.abs() > F::zero()).count())
    } else if norm == 2.0 {
        values.map(|x| x * x).sum::<F>().sqrt()
    } else if norm == 1.0 {
        values.map(|x| x.abs()).sum::<F>()
    } else {
        values
            .map(|x| x.abs().powf(F::of(norm)))
            .sum::<F>()
            .powf(F::of(1.0 / norm))
    }
}

/// Normalises one slice in place.
fn normalize_slice<F: AudioFloat>(
    mut lane: ArrayViewMut1<'_, F>,
    norm: f64,
    threshold: F,
    fill: Option<bool>,
) {
    let length = slice_norm(lane.iter().copied(), norm);
    if length < threshold {
        match fill {
            None => {}
            Some(false) => lane.mapv_inplace(|x| x / F::infinity()),
            Some(true) => {
                let fill_norm = if norm.is_infinite() {
                    F::one()
                } else {
                    F::of_usize(lane.len()).powf(F::of(-1.0 / norm))
                };
                lane.fill(fill_norm);
            }
        }
    } else {
        lane.mapv_inplace(|x| x / length);
    }
}

/// [`normalize_slice`] on every lane along `axis` when the lanes are not
/// contiguous in memory (for example the columns of a C-order matrix): the
/// norms are accumulated one `axis` index at a time over all lanes at once, so
/// memory is read in order. Each lane's norm adds its elements in the same
/// order as [`slice_norm`] (with the same starting value), so the results are
/// identical. Not used for `norm = 0`.
fn normalize_strided_lanes<F: AudioFloat>(
    mut out: ndarray::ArrayViewMutD<'_, F>,
    axis: Axis,
    norm: f64,
    threshold: F,
    fill: Option<bool>,
) {
    use ndarray::Zip;
    fn accumulate<F: AudioFloat>(
        out: &ndarray::ArrayViewMutD<'_, F>,
        axis: Axis,
        init: F,
        f: impl Fn(F, F) -> F,
    ) -> ndarray::ArrayD<F> {
        let mut acc = Array::from_elem(out.index_axis(axis, 0).raw_dim(), init);
        for sub in out.axis_iter(axis) {
            Zip::from(&mut acc)
                .and(&sub)
                .for_each(|a, &x| *a = f(*a, x));
        }
        acc
    }
    // `Sum` starts from this value (-0.0 for floats); use the same.
    let sum0: F = std::iter::empty::<F>().sum();
    let lengths = if norm == f64::INFINITY {
        accumulate(&out, axis, F::neg_infinity(), |m, x| m.max(x.abs()))
    } else if norm == f64::NEG_INFINITY {
        accumulate(&out, axis, F::infinity(), |m, x| m.min(x.abs()))
    } else if norm == 2.0 {
        accumulate(&out, axis, sum0, |s, x| s + x * x).mapv_into(|s| s.sqrt())
    } else if norm == 1.0 {
        accumulate(&out, axis, sum0, |s, x| s + x.abs())
    } else {
        let p = F::of(norm);
        let inv = F::of(1.0 / norm);
        accumulate(&out, axis, sum0, |s, x| s + x.abs().powf(p)).mapv_into(|s| s.powf(inv))
    };
    let fill_norm = if norm.is_infinite() {
        F::one()
    } else {
        F::of_usize(out.len_of(axis)).powf(F::of(-1.0 / norm))
    };
    for mut sub in out.axis_iter_mut(axis) {
        Zip::from(&mut sub).and(&lengths).for_each(|x, &length| {
            if length < threshold {
                match fill {
                    None => {}
                    Some(false) => *x /= F::infinity(),
                    Some(true) => *x = fill_norm,
                }
            } else {
                *x /= length;
            }
        });
    }
}

/// Scales every slice of an array along an axis to unit norm.
///
/// Each slice along `config.axis` (for example each column of a
/// `(features, frames)` matrix with `axis = Some(0)`) is divided by its norm.
/// Slices whose norm is below `config.threshold` are left unchanged, zeroed, or
/// filled with a constant, according to `config.fill`.
///
/// # Arguments
///
/// * `data` - Array of any dimensionality and element type `F` (`f32` or
///   `f64`). Must contain only finite values.
/// * `config` - Norm, axis, threshold and fill policy (see
///   [`NormalizeConfig::default`]). The norm exponent and threshold are rounded to `F`.
///
/// # Returns
///
/// A new array of the same shape and element type. With `norm = None` it is a
/// copy of `data`.
///
/// # Errors
///
/// [`ParameterError`] if the configuration is invalid for `data`
/// ([`NormalizeConfig::validate`]) or `data` contains NaN or infinite values.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::tonal::{NormalizeConfig, normalize_array};
/// use ndarray::array;
///
/// let s = array![[-1.0, 0.0, 2.0], [3.0, 0.0, 1.0]];
/// // Max-norm per column (the default).
/// let n = normalize_array(&s, &NormalizeConfig::default()).unwrap();
/// assert_eq!(n, array![[-1.0 / 3.0, 0.0, 1.0], [1.0, 0.0, 0.5]]);
/// // L1 per row; the all-zero middle column is untouched by default.
/// let r = normalize_array(&s, &NormalizeConfig::new().with_norm(Some(1.0)).with_axis(Some(-1))).unwrap();
/// assert_eq!(r, array![[-1.0 / 3.0, 0.0, 2.0 / 3.0], [0.75, 0.0, 0.25]]);
/// ```
pub fn normalize_array<F, S, D>(
    data: &ArrayBase<S, D>,
    config: &NormalizeConfig,
) -> AudioSampleResult<Array<F, D>>
where
    F: AudioFloat,
    S: Data<Elem = F>,
    D: Dimension,
{
    let axis = config.validate(data.ndim())?;
    let non_finite = match data.as_slice_memory_order() {
        Some(flat) => flat.iter().any(|x| !x.is_finite()),
        None => data.iter().any(|x| !x.is_finite()),
    };
    if non_finite {
        return Err(param_err("data", "input must be finite"));
    }
    let mut out = data.to_owned();
    let Some(norm) = config.norm else {
        return Ok(out);
    };
    if out.is_empty() {
        return Ok(out);
    }
    let threshold = config
        .threshold
        .map_or_else(F::min_positive_value, |t| F::of(t));
    match axis {
        Some(ax) if out.stride_of(Axis(ax)).unsigned_abs() != 1 && norm != 0.0 => {
            normalize_strided_lanes(
                out.view_mut().into_dyn(),
                Axis(ax),
                norm,
                threshold,
                config.fill,
            );
        }
        Some(ax) => {
            for lane in out.lanes_mut(Axis(ax)) {
                normalize_slice(lane, norm, threshold, config.fill);
            }
        }
        None => {
            // `to_owned` always yields a contiguous array, so the memory-order
            // slice exists; the norm and scaling do not depend on element order.
            if let Some(flat) = out.as_slice_memory_order_mut() {
                normalize_slice(ArrayViewMut1::from(flat), norm, threshold, config.fill);
            }
        }
    }
    Ok(out)
}

// =============================================================================
// Chroma filterbank
// =============================================================================

/// Parameters of [`chroma_filterbank`].
///
/// The filterbank maps the `1 + n_fft / 2` bins of an STFT onto `n_chroma`
/// pitch classes. Each FFT bin contributes to the pitch classes near its
/// frequency through a Gaussian bump whose width follows the bin's spacing in
/// chroma units, and an optional Gaussian "octave dominance" weight emphasises
/// mid-range octaves. `Default` is 12 chroma bins,
/// A440 tuning, dominance centred 5 octaves above A0 (880 Hz) with a half-width
/// of 2 octaves, L2-normalised columns, and C as row 0.
///
/// ## Invariants
///
/// Checked by [`ChromaFilterbankConfig::validate`]: `tuning` and `ctroct` are
/// finite, `octwidth` (if set) is finite and positive, and `norm` is a valid
/// [`NormalizeConfig::norm`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::tonal::ChromaFilterbankConfig;
///
/// // Quarter-tone chroma with flat octave weighting.
/// let cfg = ChromaFilterbankConfig::new()
///     .with_n_chroma(nzu!(24))
///     .with_octwidth(None);
/// assert!(cfg.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ChromaFilterbankConfig {
    /// Number of chroma bins (pitch classes) per octave. 12 gives semitones,
    /// 24 quarter-tones. Default: 12.
    pub n_chroma: NonZeroUsize,
    /// Tuning deviation from A440 in fractions of a chroma bin; shifts every
    /// pitch-class centre (e.g. from
    /// [`estimate_tuning_from_spectrogram`]). Default: 0.0.
    pub tuning: f64,
    /// Centre of the octave-dominance window, in octaves above A0 (27.5 Hz).
    /// Default: 5.0 (880 Hz).
    pub ctroct: f64,
    /// Gaussian half-width of the octave-dominance window in octaves. `None`
    /// weights all octaves equally. Default: `Some(2.0)`.
    pub octwidth: Option<f64>,
    /// Norm used to normalise each FFT-bin column of the filterbank before the
    /// octave weighting (see [`NormalizeConfig::norm`]); `None` leaves columns
    /// unnormalised. Default: `Some(2.0)`.
    pub norm: Option<f64>,
    /// If `true` row 0 is C; if `false` row 0 is A. Default: `true`.
    pub base_c: bool,
}

impl Default for ChromaFilterbankConfig {
    fn default() -> Self {
        Self {
            n_chroma: crate::nzu!(12),
            tuning: 0.0,
            ctroct: 5.0,
            octwidth: Some(2.0),
            norm: Some(2.0),
            base_c: true,
        }
    }
}

impl ChromaFilterbankConfig {
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

    /// Sets the number of chroma bins.
    #[must_use]
    pub fn with_n_chroma(mut self, n_chroma: NonZeroUsize) -> Self {
        self.n_chroma = n_chroma;
        self
    }

    /// Sets the tuning deviation in fractions of a chroma bin.
    #[must_use]
    pub fn with_tuning(mut self, tuning: f64) -> Self {
        self.tuning = tuning;
        self
    }

    /// Sets the centre of the octave-dominance window (octaves above A0).
    #[must_use]
    pub fn with_ctroct(mut self, ctroct: f64) -> Self {
        self.ctroct = ctroct;
        self
    }

    /// Sets the octave-dominance half-width (`None` = flat weighting).
    #[must_use]
    pub fn with_octwidth(mut self, octwidth: Option<f64>) -> Self {
        self.octwidth = octwidth;
        self
    }

    /// Sets the per-column norm (`None` = no normalisation).
    #[must_use]
    pub fn with_norm(mut self, norm: Option<f64>) -> Self {
        self.norm = norm;
        self
    }

    /// Sets whether row 0 is C (`true`) or A (`false`).
    #[must_use]
    pub fn with_base_c(mut self, base_c: bool) -> Self {
        self.base_c = base_c;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `tuning` or `ctroct` is not finite, `octwidth` is
    /// not finite and positive, or `norm` is not a supported norm.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if !self.tuning.is_finite() {
            return Err(param_err("tuning", "tuning must be finite"));
        }
        if !self.ctroct.is_finite() {
            return Err(param_err("ctroct", "ctroct must be finite"));
        }
        if let Some(w) = self.octwidth
            && !(w.is_finite() && w > 0.0)
        {
            return Err(param_err(
                "octwidth",
                format!("octwidth={w} must be finite and positive"),
            ));
        }
        NormalizeConfig::new()
            .with_norm(self.norm)
            .validate(2)
            .map(|_| ())
    }
}

/// Builds a chroma filterbank that projects STFT bins onto pitch classes.
///
/// FFT bin `k > 0` sits at `n_chroma * hz_to_octs(k * sr / n_fft, tuning)`
/// chroma units; the DC bin is placed 1.5 octaves below bin 1. Each bin gives
/// every pitch class `c` the weight `exp(-0.5 * (2 D / w)^2)`, where `D` is the
/// circular distance from the bin to `c` (in chroma bins) and `w` the bin's
/// width in chroma bins (at least 1). Columns are then normalised with `norm`,
/// multiplied by the octave-dominance Gaussian and rolled so row 0 is C.
///
/// # Arguments
///
/// * `sample_rate` - Sampling rate in Hz. Must be finite and positive.
/// * `n_fft` - FFT length; the filterbank has `1 + n_fft / 2` columns. Must be at least 2.
/// * `config` - Filterbank parameters (see [`ChromaFilterbankConfig::default`]).
///
/// # Returns
///
/// An `(n_chroma, 1 + n_fft / 2)` matrix; multiply it with a `(1 + n_fft / 2, frames)`
/// spectrogram to get a raw chromagram.
///
/// # Errors
///
/// [`ParameterError`] if the configuration is invalid, the sample rate is not
/// finite and positive, `n_fft < 2`, or the `n_chroma x (1 + n_fft / 2)` matrix
/// would exceed 2^28 elements.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::tonal::{ChromaFilterbankConfig, chroma_filterbank};
///
/// let fb = chroma_filterbank(22050.0, nzu!(2048), &ChromaFilterbankConfig::default()).unwrap();
/// assert_eq!(fb.dim(), (12, 1025));
/// // The column of the bin nearest 440 Hz peaks at pitch class A (row 9).
/// let col = fb.column(41); // 41 * 22050 / 2048 = 441.4 Hz
/// let argmax = (0..12usize).max_by(|&a, &b| col[a].total_cmp(&col[b])).unwrap();
/// assert_eq!(argmax, 9);
/// ```
///
/// # Precision
///
/// The basis is computed and returned in `f64`. [`chroma_stft_from_spectrogram`]
/// rounds the basis to the spectrogram's element type once.
///
/// # References
///
/// D. P. W. Ellis, "Chroma feature analysis and synthesis", 2007,
/// <https://www.ee.columbia.edu/~dpwe/resources/matlab/chroma-ansyn/>.
pub fn chroma_filterbank(
    sample_rate: f64,
    n_fft: NonZeroUsize,
    config: &ChromaFilterbankConfig,
) -> AudioSampleResult<Array2<f64>> {
    config.validate()?;
    validate_sample_rate(sample_rate)?;
    let n_fft = n_fft.get();
    if n_fft < 2 {
        return Err(param_err("n_fft", "the chroma filterbank needs n_fft >= 2"));
    }
    let n_chroma = config.n_chroma.get();
    let nc = n_chroma as f64;
    let n_out = 1 + n_fft / 2;
    crate::utils::check_derived_len(
        "n_chroma",
        "a chroma filterbank",
        nc * n_out as f64,
        crate::utils::MAX_TABLE_ELEMENTS,
    )?;

    // FFT-bin positions in chroma units for bins 0..=n_out (one extra bin for the
    // width of the last kept column), at k * sr / n_fft.
    let step = sample_rate / n_fft as f64;
    let n_pos = (n_out + 1).min(n_fft);
    let mut frqbins = vec![0.0; n_pos];
    for (k, f) in frqbins.iter_mut().enumerate().skip(1) {
        *f = nc * hz_to_octs(k as f64 * step, config.tuning, config.n_chroma);
    }
    frqbins[0] = frqbins[1] - 1.5 * nc;
    let binwidth: Vec<f64> = (0..n_out)
        .map(|k| {
            if k + 1 < n_pos {
                (frqbins[k + 1] - frqbins[k]).max(1.0)
            } else {
                1.0
            }
        })
        .collect();

    let n_chroma2 = (nc / 2.0).round_ties_even();
    let mut wts = Array2::<f64>::zeros((n_chroma, n_out));
    for c in 0..n_chroma {
        for k in 0..n_out {
            let d = frqbins[k] - c as f64;
            let d = np_remainder(d + n_chroma2 + 10.0 * nc, nc) - n_chroma2;
            let z = 2.0 * d / binwidth[k];
            wts[[c, k]] = (-0.5 * (z * z)).exp();
        }
    }

    let mut wts = normalize_array(
        &wts,
        &NormalizeConfig::new()
            .with_norm(config.norm)
            .with_axis(Some(0)),
    )?;

    if let Some(octwidth) = config.octwidth {
        for (k, &fb) in frqbins.iter().enumerate().take(n_out) {
            let z = (fb / nc - config.ctroct) / octwidth;
            let w = (-0.5 * (z * z)).exp();
            wts.column_mut(k).mapv_inplace(|x| x * w);
        }
    }

    if config.base_c {
        // Rotate the rows up by 3 * (n_chroma / 12) so that row 0 is C, not A.
        let shift = (3 * (n_chroma / 12)) % n_chroma;
        if shift != 0 {
            let rolled =
                Array2::from_shape_fn((n_chroma, n_out), |(r, k)| wts[[(r + shift) % n_chroma, k]]);
            wts = rolled;
        }
    }
    Ok(wts)
}

// =============================================================================
// piptrack
// =============================================================================

/// How [`piptrack_from_spectrogram`] decides which spectral peaks count as pitches.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::tonal::PiptrackReference;
///
/// assert_eq!(PiptrackReference::default(), PiptrackReference::FrameMax);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PiptrackReference {
    /// A bin is a candidate when it exceeds `threshold * max(frame)`.
    #[default]
    FrameMax,
    /// A bin is a candidate when it exceeds `threshold * mean(frame)`.
    FrameMean,
    /// A bin is a candidate when it exceeds this absolute value. `threshold` is
    /// **not** applied to a fixed reference.
    Value(f64),
}

/// Parameters of [`AudioTonal::piptrack`].
///
/// piptrack finds, in every STFT frame, the local maxima of the magnitude
/// spectrum that exceed a threshold within a frequency range, and refines each
/// peak's frequency and magnitude by parabolic interpolation. `Default` is a
/// magnitude STFT with `n_fft = 2048`,
/// `hop = n_fft / 4` and a Hann window, `fmin = 150` Hz, `fmax = 4000` Hz,
/// `threshold = 0.1` relative to each frame's maximum.
///
/// ## Invariants
///
/// Checked by [`PiptrackConfig::validate`]: the STFT configuration is valid,
/// `fmin`/`fmax`/`threshold` are not NaN, `threshold` is finite and
/// non-negative and a fixed reference is finite.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::tonal::{PiptrackConfig, PiptrackReference};
///
/// let cfg = PiptrackConfig::new()
///     .with_fmin(80.0)
///     .with_fmax(2000.0)
///     .with_threshold(0.2)
///     .with_reference(PiptrackReference::FrameMean);
/// assert!(cfg.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PiptrackConfig {
    /// STFT parameters of the magnitude spectrogram. When a spectrogram is
    /// supplied directly ([`piptrack_from_spectrogram`]) only `stft.n_fft` is
    /// used, to convert bins to Hz.
    pub stft: StftConfig,
    /// Lowest frequency (Hz) at which peaks are reported; clamped to `>= 0`.
    /// Default: 150.0.
    pub fmin: f64,
    /// Peaks must lie strictly below this frequency (Hz); clamped to Nyquist.
    /// Default: 4000.0.
    pub fmax: f64,
    /// Relative threshold: a bin is a candidate when it exceeds
    /// `threshold * reference`. Default: 0.1.
    pub threshold: f64,
    /// The per-frame reference the threshold is relative to. Default:
    /// [`PiptrackReference::FrameMax`].
    pub reference: PiptrackReference,
}

impl Default for PiptrackConfig {
    fn default() -> Self {
        Self {
            stft: StftConfig::default(),
            fmin: 150.0,
            fmax: 4000.0,
            threshold: 0.1,
            reference: PiptrackReference::FrameMax,
        }
    }
}

impl PiptrackConfig {
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Settings matching librosa 1.0's defaults; currently the same as [`Default`].
    #[must_use]
    pub fn librosa() -> Self {
        Self {
            stft: StftConfig::librosa(),
            ..Self::default()
        }
    }

    /// Sets the STFT parameters.
    #[must_use]
    pub fn with_stft(mut self, stft: StftConfig) -> Self {
        self.stft = stft;
        self
    }

    /// Sets the lowest reported frequency in Hz.
    #[must_use]
    pub fn with_fmin(mut self, fmin: f64) -> Self {
        self.fmin = fmin;
        self
    }

    /// Sets the (exclusive) highest reported frequency in Hz.
    #[must_use]
    pub fn with_fmax(mut self, fmax: f64) -> Self {
        self.fmax = fmax;
        self
    }

    /// Sets the relative threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets the threshold reference.
    #[must_use]
    pub fn with_reference(mut self, reference: PiptrackReference) -> Self {
        self.reference = reference;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the STFT configuration is invalid, `fmin` or
    /// `fmax` is NaN, `threshold` is not finite and non-negative, or a fixed
    /// reference is not finite.
    pub fn validate(&self) -> AudioSampleResult<()> {
        self.stft.validate()?;
        if self.fmin.is_nan() {
            return Err(param_err("fmin", "fmin must not be NaN"));
        }
        if self.fmax.is_nan() {
            return Err(param_err("fmax", "fmax must not be NaN"));
        }
        if !(self.threshold.is_finite() && self.threshold >= 0.0) {
            return Err(param_err(
                "threshold",
                format!(
                    "threshold={} must be finite and non-negative",
                    self.threshold
                ),
            ));
        }
        if let PiptrackReference::Value(v) = self.reference
            && !v.is_finite()
        {
            return Err(param_err("reference", "a fixed reference must be finite"));
        }
        Ok(())
    }
}

/// Pitch candidates of one channel's magnitude spectrogram, written into
/// `pitches` / `mags` (pre-zeroed, same shape as `s`).
fn piptrack_channel<F: AudioFloat>(
    s: ArrayView2<'_, F>,
    sample_rate: f64,
    n_fft: usize,
    config: &PiptrackConfig,
    mut pitches: ndarray::ArrayViewMut2<'_, F>,
    mut mags: ndarray::ArrayViewMut2<'_, F>,
) {
    let (n_bins, n_frames) = s.dim();
    let fmin = config.fmin.max(0.0);
    let fmax = config.fmax.min(sample_rate / 2.0);
    // Bin k is at k * (1 / (n_fft * (1 / sr))) Hz, evaluated in that order.
    let val = 1.0 / (n_fft as f64 * (1.0 / sample_rate));
    let in_range: Vec<bool> = (0..n_bins)
        .map(|k| {
            let f = k as f64 * val;
            fmin <= f && f < fmax
        })
        .collect();
    let threshold = F::of(config.threshold);
    let (two, half) = (F::of(2.0), F::of(0.5));
    let (sr, n_fft_f) = (F::of(sample_rate), F::of_usize(n_fft));
    let mut col = vec![F::zero(); n_bins];
    let mut thresholded = vec![F::zero(); n_bins];
    for t in 0..n_frames {
        for (c, &x) in col.iter_mut().zip(s.column(t)) {
            *c = x.abs();
        }
        let reference = match config.reference {
            PiptrackReference::FrameMax => {
                threshold * col.iter().fold(F::neg_infinity(), |m, &x| m.max(x))
            }
            PiptrackReference::FrameMean => {
                threshold * (col.iter().copied().sum::<F>() / F::of_usize(n_bins))
            }
            PiptrackReference::Value(v) => F::of(v).abs(),
        };
        for (th, &x) in thresholded.iter_mut().zip(&col) {
            *th = if x > reference { x } else { F::zero() };
        }
        for k in 0..n_bins {
            if !in_range[k] {
                continue;
            }
            // Local maximum along frequency: x[k] > x[k-1] and x[k] >= x[k+1];
            // the first bin is never a maximum, the last only needs x[-1] > x[-2].
            let is_max = if k == 0 {
                false
            } else if k == n_bins - 1 {
                thresholded[k] > thresholded[k - 1]
            } else {
                thresholded[k] > thresholded[k - 1] && thresholded[k] >= thresholded[k + 1]
            };
            if !is_max {
                continue;
            }
            // Parabolic interpolation (0 at the edges) and the central-difference
            // gradient (one-sided at the last bin).
            let (shift, avg) = if k == n_bins - 1 {
                (F::zero(), col[k] - col[k - 1])
            } else {
                let a = col[k + 1] + col[k - 1] - two * col[k];
                let b = (col[k + 1] - col[k - 1]) / two;
                let shift = if b.abs() >= a.abs() {
                    F::zero()
                } else {
                    -b / a
                };
                (shift, (col[k + 1] - col[k - 1]) / two)
            };
            pitches[[k, t]] = (F::of_usize(k) + shift) * sr / n_fft_f;
            mags[[k, t]] = col[k] + half * avg * shift;
        }
    }
}

/// Pitch tracking by parabolic interpolation of spectral peaks on a precomputed
/// spectrogram.
///
/// In every frame, bins inside `[fmin, fmax)` that are local maxima (along
/// frequency) of the spectrum after zeroing values not above the threshold are
/// reported. For each such bin `k` a parabola through bins `k-1, k, k+1` gives
/// the shift `-b / a` (0 when the vertex lies more than one bin away), with
/// `a = S[k+1] + S[k-1] - 2 S[k]` and `b = (S[k+1] - S[k-1]) / 2`; the pitch is
/// `(k + shift) * sr / n_fft` and the magnitude `S[k] + 0.5 * gradient * shift`.
///
/// # Arguments
///
/// * `spectrogram` - `(channels, 1 + n_fft / 2, frames)` magnitude (or power)
///   spectrogram of element type `F` (`f32` or `f64`); negative values are
///   rectified.
/// * `sample_rate` - Sampling rate in Hz.
/// * `config` - Thresholds and frequency range; only `config.stft.n_fft` of the
///   STFT parameters is used.
///
/// # Returns
///
/// `(pitches, magnitudes)`, both shaped like `spectrogram` and of element type
/// `F`. Entries are 0 except at detected peaks, where `pitches` holds the
/// interpolated frequency in Hz and `magnitudes` the interpolated magnitude.
///
/// # Errors
///
/// [`ParameterError`] if the configuration or sample rate is invalid, or the
/// bin count of `spectrogram` is not `1 + n_fft / 2` or is smaller than 2.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::spectrum::StftConfig;
/// use audio_samples::operations::tonal::{PiptrackConfig, piptrack_from_spectrogram};
/// use ndarray::Array3;
///
/// // One frame with a peak between bins 10 and 11 (n_fft = 64 -> 33 bins).
/// let mut s = Array3::<f64>::zeros((1, 33, 1));
/// s[[0, 9, 0]] = 0.5;
/// s[[0, 10, 0]] = 1.0;
/// s[[0, 11, 0]] = 0.8;
/// let cfg = PiptrackConfig::new()
///     .with_stft(StftConfig::new().with_n_fft(nzu!(64)))
///     .with_fmin(0.0);
/// let (pitches, mags) = piptrack_from_spectrogram(s.view(), 6400.0, &cfg).unwrap();
/// // Bin spacing is 100 Hz; the parabola's vertex is at bin 10 + 0.15/0.7.
/// assert!((pitches[[0, 10, 0]] - (10.0 + 0.15 / 0.7) * 100.0).abs() < 1e-9);
/// assert!(mags[[0, 10, 0]] > 1.0);
/// assert_eq!(pitches.iter().filter(|&&p| p > 0.0).count(), 1);
/// ```
///
/// # References
///
/// J. O. Smith III, "Quadratic Interpolation of Spectral Peaks", *Spectral Audio
/// Signal Processing*, <https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html>.
pub fn piptrack_from_spectrogram<F: AudioFloat>(
    spectrogram: ArrayView3<'_, F>,
    sample_rate: f64,
    config: &PiptrackConfig,
) -> AudioSampleResult<(Array3<F>, Array3<F>)> {
    config.validate()?;
    validate_sample_rate(sample_rate)?;
    let n_fft = config.stft.n_fft.get();
    check_bins(spectrogram.dim().1, n_fft, "piptrack")?;
    if spectrogram.dim().1 < 2 {
        return Err(param_err(
            "spectrogram",
            "piptrack needs at least 2 frequency bins",
        ));
    }
    let mut pitches = Array3::<F>::zeros(spectrogram.dim());
    let mut mags = Array3::<F>::zeros(spectrogram.dim());
    for (c, s) in spectrogram.outer_iter().enumerate() {
        piptrack_channel(
            s,
            sample_rate,
            n_fft,
            config,
            pitches.index_axis_mut(Axis(0), c),
            mags.index_axis_mut(Axis(0), c),
        );
    }
    Ok((pitches, mags))
}

fn check_bins(n_bins: usize, n_fft: usize, operation: &str) -> AudioSampleResult<()> {
    if n_bins == n_fft / 2 + 1 {
        Ok(())
    } else {
        Err(AudioSampleError::Parameter(
            ParameterError::invalid_configuration(
                operation,
                format!(
                    "spectrogram has {n_bins} frequency bins but n_fft={n_fft} implies {}",
                    n_fft / 2 + 1
                ),
            ),
        ))
    }
}

// =============================================================================
// Tuning estimation
// =============================================================================

/// Estimates the tuning offset of a collection of frequencies relative to A440.
///
/// Each frequency's deviation from the nearest bin of a `bins_per_octave`-tone
/// equal-tempered grid anchored at A440 is computed in `[-0.5, 0.5)` bins; the
/// deviations are histogrammed with `ceil(1 / resolution)` equal bins over
/// `[-0.5, 0.5]` and the left edge of the most populated histogram bin is
/// returned.
///
/// # Arguments
///
/// * `frequencies` - Frequencies in Hz. Non-positive and non-finite values are ignored.
/// * `resolution` - Histogram resolution as a fraction of a bin; `0.01` measures
///   in cents when `bins_per_octave` is 12. Must be in `(0, 1)`.
/// * `bins_per_octave` - Number of bins per octave of the reference grid (usually 12).
///
/// # Returns
///
/// The estimated tuning in fractions of a bin, in `[-0.5, 0.5)`. An empty set
/// (after discarding non-positive values) gives `0.0`.
///
/// # Errors
///
/// [`ParameterError`] if `resolution` is not in `(0, 1)` or so small that the
/// histogram would exceed 2^28 bins.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::tonal::pitch_tuning;
/// use audio_samples::utils::audio_math::cqt_frequencies;
///
/// // Notes 25 cents sharp.
/// let freqs = cqt_frequencies(24, 55.0, nzu!(12), 0.25);
/// let tuning = pitch_tuning(&freqs, 0.01, nzu!(12)).unwrap();
/// assert!((tuning - 0.25).abs() < 1e-9);
/// assert_eq!(pitch_tuning(&[], 0.01, nzu!(12)).unwrap(), 0.0);
/// ```
pub fn pitch_tuning(
    frequencies: &[f64],
    resolution: f64,
    bins_per_octave: NonZeroUsize,
) -> AudioSampleResult<f64> {
    if !(resolution > 0.0 && resolution < 1.0) {
        return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
            "resolution",
            resolution,
            "> 0",
            "< 1",
            "the tuning resolution is a fraction of a bin",
        )));
    }
    let bpo = bins_per_octave.get() as f64;
    // ceil(1 / resolution) + 1 evenly spaced edges from -0.5 to 0.5.
    let n_hist = crate::utils::check_derived_len(
        "resolution",
        "the tuning histogram",
        (1.0 / resolution).ceil() + 1.0,
        crate::utils::MAX_TABLE_ELEMENTS,
    )? - 1;
    let step = 1.0 / n_hist as f64;
    let mut edges: Vec<f64> = (0..=n_hist).map(|i| i as f64 * step + -0.5).collect();
    edges[n_hist] = 0.5;

    let mut counts = vec![0usize; n_hist];
    let mut any = false;
    // hz_to_octs(f, 0.0, 12) = log2(f / (A440 / 16)), with the constant
    // (exactly 27.5) hoisted out of the loop.
    let a_ref = 440.0 * 2.0f64.powf(0.0 / 12.0) / 16.0;
    let n_f = n_hist as f64;
    for &f in frequencies {
        if !(f > 0.0 && f.is_finite()) {
            continue;
        }
        any = true;
        // For finite x, `x - floor(x)` is the fractional part in [0, 1), rounded
        // once when x < 0.
        let x = bpo * (f / a_ref).log2();
        let mut residual = x - x.floor();
        if residual >= 0.5 {
            residual -= 1.0;
        }
        // Histogram bins are [e_i, e_{i+1}), with the last bin closed.
        if residual < edges[0] || residual > edges[n_hist] {
            continue;
        }
        // The bin from the uniform spacing, then moved to where a search over
        // the edges places it.
        let mut i = (((residual + 0.5) * n_f) as usize).min(n_hist - 1);
        while i > 0 && residual < edges[i] {
            i -= 1;
        }
        while i + 1 < n_hist && residual >= edges[i + 1] {
            i += 1;
        }
        counts[i] += 1;
    }
    if !any {
        return Ok(0.0);
    }
    // The first maximum wins ties.
    let (mut best, mut best_count) = (0usize, 0usize);
    for (i, &c) in counts.iter().enumerate() {
        if c > best_count {
            best = i;
            best_count = c;
        }
    }
    Ok(edges[best])
}

/// Parameters of tuning estimation ([`AudioTonal::estimate_tuning`]).
///
/// Tuning estimation tracks spectral peaks with piptrack, keeps those at least
/// as strong as the median peak, and histograms their deviation from the
/// equal-tempered grid with [`pitch_tuning`]. `Default` is the piptrack
/// defaults (`n_fft = 2048`, 150-4000 Hz, threshold 0.1),
/// 0.01-bin resolution, 12 bins per octave.
///
/// ## Invariants
///
/// Checked by [`TuningConfig::validate`]: the piptrack configuration is valid and
/// `resolution` is in `(0, 1)`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::tonal::TuningConfig;
///
/// // Tenth-of-a-cent resolution against a quarter-tone grid.
/// let cfg = TuningConfig::new().with_resolution(0.001).with_bins_per_octave(nzu!(24));
/// assert!(cfg.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TuningConfig {
    /// Peak tracking parameters (STFT, frequency range, threshold).
    pub piptrack: PiptrackConfig,
    /// Resolution of the estimate as a fraction of a bin, in `(0, 1)`. 0.01
    /// gives cents for a 12-tone grid. Default: 0.01.
    pub resolution: f64,
    /// Number of bins per octave of the grid the deviation is measured against;
    /// the estimate is in fractions of one such bin. Default: 12.
    pub bins_per_octave: NonZeroUsize,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            piptrack: PiptrackConfig::default(),
            resolution: 0.01,
            bins_per_octave: crate::nzu!(12),
        }
    }
}

impl TuningConfig {
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Settings matching librosa 1.0's defaults; currently the same as [`Default`].
    #[must_use]
    pub fn librosa() -> Self {
        Self {
            piptrack: PiptrackConfig::librosa(),
            ..Self::default()
        }
    }

    /// Sets the piptrack parameters.
    #[must_use]
    pub fn with_piptrack(mut self, piptrack: PiptrackConfig) -> Self {
        self.piptrack = piptrack;
        self
    }

    /// Sets the resolution (fraction of a bin).
    #[must_use]
    pub fn with_resolution(mut self, resolution: f64) -> Self {
        self.resolution = resolution;
        self
    }

    /// Sets the number of bins per octave.
    #[must_use]
    pub fn with_bins_per_octave(mut self, bins_per_octave: NonZeroUsize) -> Self {
        self.bins_per_octave = bins_per_octave;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the piptrack configuration is invalid or
    /// `resolution` is not in `(0, 1)` (or needs more than 2^28 histogram bins).
    pub fn validate(&self) -> AudioSampleResult<()> {
        self.piptrack.validate()?;
        if !(self.resolution > 0.0 && self.resolution < 1.0) {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "resolution",
                self.resolution,
                "> 0",
                "< 1",
                "the tuning resolution is a fraction of a bin",
            )));
        }
        crate::utils::check_derived_len(
            "resolution",
            "the tuning histogram",
            (1.0 / self.resolution).ceil() + 1.0,
            crate::utils::MAX_TABLE_ELEMENTS,
        )?;
        Ok(())
    }
}

/// Estimates the tuning of a precomputed spectrogram.
///
/// Runs [`piptrack_from_spectrogram`], keeps the detected pitches whose
/// magnitude is at least the median magnitude of all detected pitches, and
/// passes them to [`pitch_tuning`]. All channels are pooled into one estimate.
///
/// # Arguments
///
/// * `spectrogram` - `(channels, 1 + n_fft / 2, frames)` magnitude or power
///   spectrogram of element type `F` (`f32` or `f64`); peak tracking runs in `F`.
/// * `sample_rate` - Sampling rate in Hz.
/// * `config` - Tuning parameters; `config.piptrack.stft.n_fft` must match the
///   spectrogram's bin count.
///
/// # Returns
///
/// The estimated tuning deviation in fractions of a bin, in `[-0.5, 0.5)`; 0.0
/// when no peaks are found. The estimate is a multiple of `config.resolution`
/// and is returned as `f64` for every `F`.
///
/// # Errors
///
/// [`ParameterError`] if the configuration or sample rate is invalid or the
/// spectrogram's bin count does not match `n_fft`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::spectrum::{StftConfig, stft};
/// use audio_samples::operations::tonal::{TuningConfig, estimate_tuning_from_spectrogram};
/// use ndarray::Array2;
///
/// // A tone 30 cents above A440.
/// let f = 440.0 * 2f64.powf(0.3 / 12.0);
/// let x = Array2::from_shape_fn((1, 22050), |(_, n)| (2.0 * std::f64::consts::PI * f * n as f64 / 22050.0).sin());
/// let s = stft(x.view(), &StftConfig::default()).unwrap().mapv(|z| z.norm());
/// let t = estimate_tuning_from_spectrogram(s.view(), 22050.0, &TuningConfig::default()).unwrap();
/// assert!((t - 0.3).abs() < 0.03);
/// ```
pub fn estimate_tuning_from_spectrogram<F: AudioFloat>(
    spectrogram: ArrayView3<'_, F>,
    sample_rate: f64,
    config: &TuningConfig,
) -> AudioSampleResult<f64> {
    config.validate()?;
    let (pitches, mags) = piptrack_from_spectrogram(spectrogram, sample_rate, &config.piptrack)?;
    let mut voiced: Vec<F> = pitches
        .iter()
        .zip(mags.iter())
        .filter(|&(&p, _)| p > F::zero())
        .map(|(_, &m)| m)
        .collect();
    let threshold = median(&mut voiced).unwrap_or_else(F::zero);
    let selected: Vec<f64> = pitches
        .iter()
        .zip(mags.iter())
        .filter(|&(&p, &m)| p > F::zero() && m >= threshold)
        .map(|(&p, _)| p.as_f64())
        .collect();
    pitch_tuning(&selected, config.resolution, config.bins_per_octave)
}

/// Estimates the tuning of raw signal rows.
///
/// Computes the magnitude STFT with `config.piptrack.stft` and calls
/// [`estimate_tuning_from_spectrogram`]. All channels are pooled.
///
/// # Arguments
///
/// * `signal` - `(channels, samples)` signal of element type `F` (`f32` or
///   `f64`); the STFT and peak tracking run in `F`.
/// * `sample_rate` - Sampling rate in Hz.
/// * `config` - Tuning parameters (see [`TuningConfig::default`]).
///
/// # Returns
///
/// The estimated tuning deviation in fractions of a bin, in `[-0.5, 0.5)`, as
/// `f64`.
///
/// # Errors
///
/// [`ParameterError`] if the configuration or sample rate is invalid, or the
/// signal is empty (or shorter than `n_fft` with `center = false`).
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::tonal::{TuningConfig, estimate_tuning_from_signal};
/// use ndarray::Array2;
///
/// let f = 440.0 * 2f64.powf(-0.2 / 12.0); // 20 cents flat
/// let x = Array2::from_shape_fn((1, 22050), |(_, n)| (2.0 * std::f64::consts::PI * f * n as f64 / 22050.0).sin());
/// let t = estimate_tuning_from_signal(x.view(), 22050.0, &TuningConfig::default()).unwrap();
/// assert!((t + 0.2).abs() < 0.03);
/// ```
pub fn estimate_tuning_from_signal<F: AudioFloat>(
    signal: ArrayView2<'_, F>,
    sample_rate: f64,
    config: &TuningConfig,
) -> AudioSampleResult<f64> {
    config.validate()?;
    let s = stft_map(signal, &config.piptrack.stft, |z| z.norm())?;
    estimate_tuning_from_spectrogram(s.view(), sample_rate, config)
}

/// `f64::total_cmp` for `F` (exact: every `F` value converts to `f64` exactly
/// and the conversion preserves the order, including signed zeros).
fn total_cmp<F: AudioFloat>(a: &F, b: &F) -> std::cmp::Ordering {
    a.as_f64().total_cmp(&b.as_f64())
}

/// The median: the middle value, or the mean of the two middle values.
fn median<F: AudioFloat>(values: &mut [F]) -> Option<F> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(total_cmp);
    let n = values.len();
    Some(if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / F::of(2.0)
    })
}

// =============================================================================
// chroma_stft
// =============================================================================

/// Parameters of [`AudioTonal::chroma_stft`].
///
/// A chromagram folds the spectrum into pitch classes, giving a
/// `(n_chroma, frames)` profile of harmonic content that is robust to octave
/// and timbre. `Default`: a power spectrogram with `n_fft = 2048`, `hop = 512`
/// and a Hann window, A440 tuning (`tuning = Some(0.0)`), 12 chroma bins, the
/// default chroma filterbank, and each frame scaled to a maximum of 1.
///
/// [`ChromaStftConfig::librosa`] differs only in `tuning = None`, which
/// estimates the tuning from the signal. A fixed tuning makes the chroma of a
/// signal independent of its peak content; an estimated tuning compensates for
/// recordings that are not tuned to A440.
///
/// ## Invariants
///
/// Checked by [`ChromaStftConfig::validate`]: the STFT configuration is valid,
/// `tuning` (if set) is finite, the filterbank parameters are valid, and `norm`
/// is a supported norm.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::tonal::ChromaStftConfig;
///
/// // A4 = 440 Hz assumed, unnormalised quarter-tone chroma.
/// let cfg = ChromaStftConfig::new()
///     .with_tuning(Some(0.0))
///     .with_norm(None)
///     .with_n_chroma(nzu!(24));
/// assert!(cfg.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ChromaStftConfig {
    /// STFT parameters of the power spectrogram. Default: [`StftConfig::default`]
    /// with `hop_length = Some(512)` (a fixed 512-sample hop regardless of
    /// `n_fft`).
    pub stft: StftConfig,
    /// Per-frame normalisation of the chromagram (see [`NormalizeConfig::norm`]);
    /// `None` returns raw chroma energy. Default: `Some(f64::INFINITY)`
    /// (each frame's maximum is 1; silent frames stay 0).
    pub norm: Option<f64>,
    /// Tuning deviation from A440 in fractions of a chroma bin. Default
    /// `Some(0.0)` (A440). `None` estimates it from the spectrogram with
    /// [`estimate_tuning_from_spectrogram`] (default piptrack parameters,
    /// `bins_per_octave = n_chroma`).
    pub tuning: Option<f64>,
    /// Number of chroma bins. Default: 12.
    pub n_chroma: NonZeroUsize,
    /// Centre of the filterbank's octave-dominance window in octaves above A0.
    /// Default: 5.0.
    pub ctroct: f64,
    /// Half-width of the octave-dominance window in octaves; `None` = flat.
    /// Default: `Some(2.0)`.
    pub octwidth: Option<f64>,
    /// If `true` chroma row 0 is C, otherwise A. Default: `true`.
    pub base_c: bool,
}

impl Default for ChromaStftConfig {
    fn default() -> Self {
        Self {
            stft: StftConfig::default().with_hop_length(Some(crate::nzu!(512))),
            norm: Some(f64::INFINITY),
            tuning: Some(0.0),
            n_chroma: crate::nzu!(12),
            ctroct: 5.0,
            octwidth: Some(2.0),
            base_c: true,
        }
    }
}

impl ChromaStftConfig {
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
            stft: StftConfig::librosa().with_hop_length(Some(crate::nzu!(512))),
            tuning: None,
            ..Self::default()
        }
    }

    /// Sets the STFT parameters.
    #[must_use]
    pub fn with_stft(mut self, stft: StftConfig) -> Self {
        self.stft = stft;
        self
    }

    /// Sets the per-frame norm (`None` = unnormalised).
    #[must_use]
    pub fn with_norm(mut self, norm: Option<f64>) -> Self {
        self.norm = norm;
        self
    }

    /// Sets the tuning (`None` = estimate from the signal).
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

    /// Sets the centre of the octave-dominance window.
    #[must_use]
    pub fn with_ctroct(mut self, ctroct: f64) -> Self {
        self.ctroct = ctroct;
        self
    }

    /// Sets the octave-dominance half-width (`None` = flat).
    #[must_use]
    pub fn with_octwidth(mut self, octwidth: Option<f64>) -> Self {
        self.octwidth = octwidth;
        self
    }

    /// Sets whether row 0 is C.
    #[must_use]
    pub fn with_base_c(mut self, base_c: bool) -> Self {
        self.base_c = base_c;
        self
    }

    /// The chroma filterbank configuration for a given tuning.
    #[must_use]
    pub fn filterbank(&self, tuning: f64) -> ChromaFilterbankConfig {
        ChromaFilterbankConfig {
            n_chroma: self.n_chroma,
            tuning,
            ctroct: self.ctroct,
            octwidth: self.octwidth,
            norm: Some(2.0),
            base_c: self.base_c,
        }
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the STFT configuration is invalid, `tuning` is not
    /// finite, the filterbank parameters are invalid, or `norm` is unsupported.
    pub fn validate(&self) -> AudioSampleResult<()> {
        self.stft.validate()?;
        self.filterbank(self.tuning.unwrap_or(0.0)).validate()?;
        NormalizeConfig::new()
            .with_norm(self.norm)
            .validate(2)
            .map(|_| ())
    }
}

/// Computes a chromagram from a precomputed spectrogram.
///
/// Estimates the tuning when `config.tuning` is `None`, projects each frame
/// onto [`chroma_filterbank`] and normalises each frame with `config.norm`.
///
/// # Arguments
///
/// * `spectrogram` - `(channels, 1 + n_fft / 2, frames)` spectrogram of element
///   type `F` (`f32` or `f64`), normally a power spectrogram; a magnitude
///   spectrogram gives "energy" chroma.
/// * `sample_rate` - Sampling rate in Hz.
/// * `config` - Chroma parameters; only `config.stft.n_fft` of the STFT
///   parameters is used.
///
/// # Returns
///
/// A `(channels, n_chroma, frames)` chromagram of element type `F`. The
/// filterbank is computed in `f64` and rounded to `F` once.
///
/// # Errors
///
/// [`ParameterError`] if the configuration or sample rate is invalid, the
/// spectrogram's bin count does not match `n_fft`, or it contains non-finite values.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::tonal::{ChromaStftConfig, chroma_stft_from_spectrogram};
/// use ndarray::Array3;
///
/// let s = Array3::<f64>::ones((2, 1025, 3));
/// let chroma = chroma_stft_from_spectrogram(s.view(), 22050.0, &ChromaStftConfig::default()).unwrap();
/// assert_eq!(chroma.dim(), (2, 12, 3));
///
/// // An f32 spectrogram gives an f32 chromagram.
/// let s32 = Array3::<f32>::ones((1, 1025, 3));
/// let c32 = chroma_stft_from_spectrogram(s32.view(), 22050.0, &ChromaStftConfig::default()).unwrap();
/// assert_eq!(c32.dim(), (1, 12, 3));
/// ```
///
/// # Precision and FFT length
///
/// - The filterbank is computed in `f64` and rounded to `F` once.
/// - With an odd `n_fft` the configured `n_fft` is used throughout, including
///   the tuning estimate; it is not re-inferred from the number of bins.
///
/// # References
///
/// D. P. W. Ellis, "Chroma feature analysis and synthesis", 2007.
pub fn chroma_stft_from_spectrogram<F: AudioFloat>(
    spectrogram: ArrayView3<'_, F>,
    sample_rate: f64,
    config: &ChromaStftConfig,
) -> AudioSampleResult<Array3<F>> {
    config.validate()?;
    validate_sample_rate(sample_rate)?;
    let n_fft = config.stft.n_fft;
    check_bins(spectrogram.dim().1, n_fft.get(), "chroma_stft")?;
    let tuning = match config.tuning {
        Some(t) => t,
        None => {
            let tc = TuningConfig::new()
                .with_bins_per_octave(config.n_chroma)
                .with_piptrack(
                    PiptrackConfig::new().with_stft(StftConfig::new().with_n_fft(n_fft)),
                );
            estimate_tuning_from_spectrogram(spectrogram, sample_rate, &tc)?
        }
    };
    let basis = chroma_filterbank(sample_rate, n_fft, &config.filterbank(tuning))?.mapv(F::of);
    let (n_channels, _, n_frames) = spectrogram.dim();
    let mut raw = Array3::<F>::zeros((n_channels, basis.nrows(), n_frames));
    for (c, s) in spectrogram.outer_iter().enumerate() {
        raw.index_axis_mut(Axis(0), c).assign(&basis.dot(&s));
    }
    normalize_array(
        &raw,
        &NormalizeConfig::new()
            .with_norm(config.norm)
            .with_axis(Some(-2)),
    )
}

// =============================================================================
// Trait
// =============================================================================

/// Tonal analysis of [`AudioSamples`]: chroma, tuning and spectral pitch
/// tracking.
///
/// Frame-wise outputs are shaped `(channels, n_features, n_frames)` and have
/// the element type [`Float`](Self::Float): `f32` for `u8`, `i16`, `I24` and
/// `f32` audio, `f64` for `i32` and `f64` audio. The computation runs in that
/// type. Tuning estimates are `f64` and pool all channels.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::tonal::{AudioTonal, PiptrackConfig};
/// use audio_samples::{AudioSamples, sample_rate, sine_wave};
/// use std::time::Duration;
///
/// let audio = sine_wave::<f64>(330.0, Duration::from_millis(250), sample_rate!(22050), 0.5);
/// let (pitches, mags) = audio.piptrack(&PiptrackConfig::default()).unwrap();
/// assert_eq!(pitches.dim(), mags.dim());
/// assert_eq!(pitches.dim(), (1, 1025, 11));
/// ```
pub trait AudioTonal {
    /// The float type of the outputs and of the computation: the sample type's
    /// [`AudioSample::Float`](crate::AudioSample::Float).
    type Float: AudioFloat;

    /// Computes a chromagram from the power STFT.
    ///
    /// # Arguments
    ///
    /// * `config` - Chroma parameters ([`ChromaStftConfig::default`], or
    ///   [`ChromaStftConfig::librosa`], which estimates the tuning).
    ///
    /// # Returns
    ///
    /// A `(channels, n_chroma, n_frames)` chromagram of element type
    /// [`Float`](Self::Float); with the default max-norm every non-silent frame
    /// has a maximum of 1.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample.
    ///
    /// [`ParameterError`] if the configuration is invalid or the signal is too
    /// short for an uncentred STFT.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::tonal::{AudioTonal, ChromaStftConfig};
    /// use audio_samples::{AudioSamples, sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// // C4 = 261.63 Hz -> pitch class 0.
    /// let audio = sine_wave::<f32>(261.63, Duration::from_millis(500), sample_rate!(22050), 0.5);
    /// let chroma = audio.chroma_stft(&ChromaStftConfig::default()).unwrap();
    /// let frame = chroma.slice(ndarray::s![0, .., 10]);
    /// let argmax = (0..12usize).max_by(|&a, &b| frame[a].total_cmp(&frame[b])).unwrap();
    /// assert_eq!(argmax, 0);
    /// ```
    ///
    /// # Precision
    ///
    /// See [`chroma_stft_from_spectrogram`] (filterbank computed in `f64`).
    fn chroma_stft(&self, config: &ChromaStftConfig) -> AudioSampleResult<Array3<Self::Float>>;

    /// Tracks spectral peaks with parabolic interpolation.
    ///
    /// # Arguments
    ///
    /// * `config` - Peak-tracking parameters (see [`PiptrackConfig::default`]).
    ///
    /// # Returns
    ///
    /// `(pitches, magnitudes)`, each `(channels, 1 + n_fft / 2, n_frames)` of
    /// element type [`Float`](Self::Float); see [`piptrack_from_spectrogram`].
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample.
    ///
    /// [`ParameterError`] if the configuration is invalid or the signal is too
    /// short for an uncentred STFT.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::tonal::{AudioTonal, PiptrackConfig};
    /// use audio_samples::{AudioSamples, sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
    /// let (pitches, _) = audio.piptrack(&PiptrackConfig::default()).unwrap();
    /// // The strongest peak of a middle frame is within a few Hz of 440.
    /// let p = pitches.slice(ndarray::s![0, .., 10]).iter().copied().fold(0.0, f64::max);
    /// assert!((p - 440.0).abs() < 2.0);
    /// ```
    #[allow(clippy::type_complexity)] // (pitches, magnitudes)
    fn piptrack(
        &self,
        config: &PiptrackConfig,
    ) -> AudioSampleResult<(Array3<Self::Float>, Array3<Self::Float>)>;

    /// Estimates the tuning deviation from A440.
    ///
    /// # Arguments
    ///
    /// * `config` - Tuning parameters (see [`TuningConfig::default`]).
    ///
    /// # Returns
    ///
    /// The deviation in fractions of a bin (`config.bins_per_octave` per
    /// octave), in `[-0.5, 0.5)`, as `f64`; all channels are pooled. The
    /// spectrogram and peak tracking are computed in [`Float`](Self::Float).
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample.
    ///
    /// [`ParameterError`] if the configuration is invalid or the signal is too
    /// short for an uncentred STFT.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::tonal::{AudioTonal, TuningConfig};
    /// use audio_samples::{AudioSamples, sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// // 432 Hz "Verdi tuning" is about 32 cents flat of A440.
    /// let audio = sine_wave::<f32>(432.0, Duration::from_secs(1), sample_rate!(22050), 0.5);
    /// let t = audio.estimate_tuning(&TuningConfig::default()).unwrap();
    /// assert!((t + 0.32).abs() < 0.03);
    /// ```
    fn estimate_tuning(&self, config: &TuningConfig) -> AudioSampleResult<f64>;
}

impl<T> AudioTonal for AudioSamples<'_, T>
where
    T: StandardSample,
{
    type Float = T::Float;

    fn chroma_stft(&self, config: &ChromaStftConfig) -> AudioSampleResult<Array3<T::Float>> {
        config.validate()?;
        let s = spectrogram_rows(finite_float_rows(self)?.view(), &config.stft, 2.0)?;
        chroma_stft_from_spectrogram(s.view(), self.sample_rate_hz(), config)
    }

    fn piptrack(
        &self,
        config: &PiptrackConfig,
    ) -> AudioSampleResult<(Array3<T::Float>, Array3<T::Float>)> {
        config.validate()?;
        let s = spectrogram_rows(finite_float_rows(self)?.view(), &config.stft, 1.0)?;
        piptrack_from_spectrogram(s.view(), self.sample_rate_hz(), config)
    }

    fn estimate_tuning(&self, config: &TuningConfig) -> AudioSampleResult<f64> {
        config.validate()?;
        let s = spectrogram_rows(finite_float_rows(self)?.view(), &config.piptrack.stft, 1.0)?;
        estimate_tuning_from_spectrogram(s.view(), self.sample_rate_hz(), config)
    }
}

#[cfg(test)]
mod tests {

    /// The previous pitch_tuning loop (hz_to_octs, sign-of-divisor remainder,
    /// search over the edges), the reference for the optimised one.
    fn pitch_tuning_reference(frequencies: &[f64], resolution: f64) -> f64 {
        let n_hist = ((1.0 / resolution).ceil() + 1.0) as usize - 1;
        let step = 1.0 / n_hist as f64;
        let mut edges: Vec<f64> = (0..=n_hist).map(|i| i as f64 * step + -0.5).collect();
        edges[n_hist] = 0.5;
        let mut counts = vec![0usize; n_hist];
        let mut any = false;
        for &f in frequencies {
            if !(f > 0.0 && f.is_finite()) {
                continue;
            }
            any = true;
            let mut residual = np_remainder(12.0 * hz_to_octs(f, 0.0, crate::nzu!(12)), 1.0);
            if residual >= 0.5 {
                residual -= 1.0;
            }
            if residual < edges[0] || residual > edges[n_hist] {
                continue;
            }
            let i = edges
                .partition_point(|&e| e <= residual)
                .saturating_sub(1)
                .min(n_hist - 1);
            counts[i] += 1;
        }
        if !any {
            return 0.0;
        }
        let (mut best, mut best_count) = (0usize, 0usize);
        for (i, &c) in counts.iter().enumerate() {
            if c > best_count {
                best = i;
                best_count = c;
            }
        }
        edges[best]
    }

    #[test]
    fn pitch_tuning_matches_the_reference_loop() {
        let mut s = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        for &res in &[0.01, 0.003, 0.1, 0.37] {
            for trial in 0..40 {
                let n = 1 + trial * 37;
                let v: Vec<f64> = (0..n)
                    .map(|i| match i % 7 {
                        0 => 27.5 * 2f64.powf((next() * 120.0).round() / 12.0),
                        1 => -next(),
                        _ => 1e-3 + next() * 2e4,
                    })
                    .collect();
                let got = pitch_tuning(&v, res, crate::nzu!(12)).unwrap();
                assert_eq!(got.to_bits(), pitch_tuning_reference(&v, res).to_bits());
            }
            // Exact semitones (residual exactly 0).
            let v: Vec<f64> = (0..200)
                .map(|k| 27.5 * 2f64.powf(k as f64 / 12.0))
                .collect();
            assert_eq!(
                pitch_tuning(&v, res, crate::nzu!(12)).unwrap().to_bits(),
                pitch_tuning_reference(&v, res).to_bits()
            );
        }
        // x - floor(x) equals the sign-of-divisor remainder of x and 1.
        for _ in 0..100_000 {
            let x = (next() - 0.5) * 10f64.powi((next() * 12.0) as i32 - 4);
            assert_eq!(
                (x - x.floor()).to_bits(),
                np_remainder(x, 1.0).to_bits(),
                "{x}"
            );
        }
    }
    use super::*;
    use crate::AudioTypeConversion;
    use crate::sample_rate;
    use crate::utils::audio_math::{
        a4_to_tuning, cqt_frequencies, key_to_degrees, key_to_notes, octs_to_hz, tuning_to_a4,
    };
    use ndarray::array;

    #[test]
    fn huge_chroma_filterbank_is_parameter_error() {
        let cfg = ChromaFilterbankConfig::new().with_n_chroma(NonZeroUsize::new(1 << 30).unwrap());
        assert!(matches!(
            chroma_filterbank(22050.0, crate::nzu!(2048), &cfg),
            Err(AudioSampleError::Parameter(_))
        ));
    }

    fn tone(f: f64, sr: f64, n: usize) -> Array2<f64> {
        Array2::from_shape_fn((1, n), |(_, i)| {
            (2.0 * std::f64::consts::PI * f * i as f64 / sr).sin()
        })
    }

    #[test]
    fn normalize_norms_and_fill() {
        let s = array![[3.0, 0.0], [-4.0, 0.0]];
        let l2 = normalize_array(&s, &NormalizeConfig::new().with_norm(Some(2.0))).unwrap();
        assert_eq!(l2, array![[0.6, 0.0], [-0.8, 0.0]]);
        let l0 = normalize_array(&s, &NormalizeConfig::new().with_norm(Some(0.0))).unwrap();
        assert_eq!(l0, array![[1.5, 0.0], [-2.0, 0.0]]);
        let min = normalize_array(
            &s,
            &NormalizeConfig::new().with_norm(Some(f64::NEG_INFINITY)),
        )
        .unwrap();
        assert_eq!(min[[0, 0]], 1.0);
        let filled = normalize_array(
            &s,
            &NormalizeConfig::new()
                .with_norm(Some(2.0))
                .with_fill(Some(true)),
        )
        .unwrap();
        assert!((filled[[0, 1]] - 0.5f64.sqrt()).abs() < 1e-15);
        let zeroed = normalize_array(
            &array![[1e-3, 1.0]],
            &NormalizeConfig::new()
                .with_axis(None)
                .with_threshold(Some(2.0))
                .with_fill(Some(false)),
        )
        .unwrap();
        assert_eq!(zeroed, array![[0.0, 0.0]]);
        let whole = normalize_array(
            &array![[1.0, 2.0], [4.0, 3.0]],
            &NormalizeConfig::new().with_axis(None),
        )
        .unwrap();
        assert_eq!(whole, array![[0.25, 0.5], [1.0, 0.75]]);
        let none = normalize_array(&s, &NormalizeConfig::new().with_norm(None)).unwrap();
        assert_eq!(none, s);
    }

    #[test]
    fn normalize_is_bit_identical_in_either_memory_order() {
        // Lanes along axis 0 of a C-order matrix take the strided path; the same
        // data in Fortran order takes the per-lane path. Results must match bit
        // for bit, for every norm and fill policy, including small lanes.
        let c = Array2::from_shape_fn((37, 23), |(i, j)| {
            let v = ((i * 31 + j * 17) % 13) as f64 - 6.0;
            if j == 5 {
                0.0
            } else if j == 7 {
                v * 1e-310
            } else {
                v * 0.37
            }
        });
        let f = {
            let mut f = Array2::<f64>::zeros(ndarray::ShapeBuilder::f(c.raw_dim()));
            f.assign(&c);
            f
        };
        for norm in [f64::INFINITY, f64::NEG_INFINITY, 1.0, 2.0, 3.5, 0.5] {
            for fill in [None, Some(false), Some(true)] {
                for axis in [0isize, 1, -1] {
                    let cfg = NormalizeConfig::new()
                        .with_norm(Some(norm))
                        .with_axis(Some(axis))
                        .with_fill(fill);
                    let a = normalize_array(&c, &cfg).unwrap();
                    let b = normalize_array(&f, &cfg).unwrap();
                    assert!(
                        a.iter()
                            .zip(b.iter())
                            .all(|(x, y)| x.to_bits() == y.to_bits()),
                        "norm {norm} fill {fill:?} axis {axis}"
                    );
                    let a32 = normalize_array(&c.mapv(|v| v as f32), &cfg).unwrap();
                    let b32 = normalize_array(&f.mapv(|v| v as f32), &cfg).unwrap();
                    assert!(
                        a32.iter()
                            .zip(b32.iter())
                            .all(|(x, y)| x.to_bits() == y.to_bits()),
                        "f32 norm {norm} fill {fill:?} axis {axis}"
                    );
                }
            }
        }
    }

    #[test]
    fn normalize_rejects_bad_input() {
        let s = array![[1.0, f64::NAN]];
        assert!(normalize_array(&s, &NormalizeConfig::default()).is_err());
        let s = array![[1.0, 2.0]];
        assert!(normalize_array(&s, &NormalizeConfig::new().with_norm(Some(-1.0))).is_err());
        assert!(normalize_array(&s, &NormalizeConfig::new().with_axis(Some(2))).is_err());
        assert!(normalize_array(&s, &NormalizeConfig::new().with_axis(Some(-3))).is_err());
        assert!(normalize_array(&s, &NormalizeConfig::new().with_threshold(Some(0.0))).is_err());
        assert!(
            normalize_array(
                &s,
                &NormalizeConfig::new()
                    .with_norm(Some(0.0))
                    .with_fill(Some(true))
            )
            .is_err()
        );
    }

    #[test]
    fn chroma_filterbank_shape_columns_and_validation() {
        let fb = chroma_filterbank(
            22050.0,
            nzu!(2048),
            &ChromaFilterbankConfig::new().with_octwidth(None),
        )
        .unwrap();
        assert_eq!(fb.dim(), (12, 1025));
        // Without octave weighting every column has unit L2 norm.
        for col in fb.columns() {
            assert!((col.dot(&col) - 1.0).abs() < 1e-12);
        }
        let fb24 = chroma_filterbank(
            16000.0,
            nzu!(513),
            &ChromaFilterbankConfig::new().with_n_chroma(nzu!(24)),
        )
        .unwrap();
        assert_eq!(fb24.dim(), (24, 257));
        assert!(chroma_filterbank(22050.0, nzu!(1), &ChromaFilterbankConfig::default()).is_err());
        assert!(chroma_filterbank(0.0, nzu!(2048), &ChromaFilterbankConfig::default()).is_err());
        assert!(
            chroma_filterbank(
                22050.0,
                nzu!(2048),
                &ChromaFilterbankConfig::new().with_octwidth(Some(0.0))
            )
            .is_err()
        );
        assert!(
            chroma_filterbank(
                22050.0,
                nzu!(2048),
                &ChromaFilterbankConfig::new().with_tuning(f64::NAN)
            )
            .is_err()
        );
        // n_fft = 2 is the smallest valid size.
        assert_eq!(
            chroma_filterbank(8000.0, nzu!(2), &ChromaFilterbankConfig::default())
                .unwrap()
                .dim(),
            (12, 2)
        );
    }

    #[test]
    fn base_c_rolls_rows_by_three() {
        let c = chroma_filterbank(22050.0, nzu!(1024), &ChromaFilterbankConfig::default()).unwrap();
        let a = chroma_filterbank(
            22050.0,
            nzu!(1024),
            &ChromaFilterbankConfig::new().with_base_c(false),
        )
        .unwrap();
        for r in 0..12 {
            assert_eq!(c.row(r), a.row((r + 3) % 12));
        }
    }

    #[test]
    fn pitch_tuning_recovers_offsets_and_handles_edges() {
        for &t in &[-0.4, -0.1, 0.0, 0.25, 0.37] {
            let f = cqt_frequencies(36, 55.0, nzu!(12), t);
            let est = pitch_tuning(&f, 0.01, nzu!(12)).unwrap();
            assert!((est - t).abs() < 0.0101, "{t} -> {est}");
        }
        assert_eq!(
            pitch_tuning(&[0.0, -3.0, f64::NAN], 0.01, nzu!(12)).unwrap(),
            0.0
        );
        assert!(pitch_tuning(&[440.0], 0.0, nzu!(12)).is_err());
        assert!(pitch_tuning(&[440.0], 1.0, nzu!(12)).is_err());
    }

    #[test]
    fn piptrack_finds_tone_and_respects_range() {
        let x = tone(1000.0, 22050.0, 8192);
        let s = crate::operations::spectrum::stft(x.view(), &StftConfig::default())
            .unwrap()
            .mapv(|z| z.norm());
        let (p, m) =
            piptrack_from_spectrogram(s.view(), 22050.0, &PiptrackConfig::default()).unwrap();
        let t = 8;
        let (k, &best) = m
            .slice(ndarray::s![0, .., t])
            .indexed_iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        assert!(best > 0.0);
        assert!((p[[0, k, t]] - 1000.0).abs() < 2.0);
        // Excluding 1000 Hz removes the peak.
        let cfg = PiptrackConfig::default().with_fmax(900.0);
        let (p2, _) = piptrack_from_spectrogram(s.view(), 22050.0, &cfg).unwrap();
        assert!(p2.slice(ndarray::s![0, .., t]).iter().all(|&v| v < 900.0));
        // Bin-count mismatch is an error.
        let cfg = PiptrackConfig::default().with_stft(StftConfig::new().with_n_fft(nzu!(1024)));
        assert!(piptrack_from_spectrogram(s.view(), 22050.0, &cfg).is_err());
        assert!(
            PiptrackConfig::default()
                .with_threshold(-1.0)
                .validate()
                .is_err()
        );
    }

    #[test]
    fn estimate_tuning_detects_detuning() {
        for &cents in &[-30.0, 0.0, 25.0] {
            let f = 440.0 * 2f64.powf(cents / 1200.0);
            let x = tone(f, 22050.0, 22050);
            let t =
                estimate_tuning_from_signal(x.view(), 22050.0, &TuningConfig::default()).unwrap();
            // Histogram estimate at 0.01-bin resolution.
            assert!((t - cents / 100.0).abs() <= 0.025, "{cents} -> {t}");
        }
        // Silence: no peaks -> 0.
        let z = Array2::<f64>::zeros((1, 4096));
        assert_eq!(
            estimate_tuning_from_signal(z.view(), 22050.0, &TuningConfig::default()).unwrap(),
            0.0
        );
    }

    #[test]
    fn chroma_stft_trait_mono_and_stereo() {
        let sr = sample_rate!(22050);
        let l = tone(392.0, 22050.0, 11025); // G4 -> class 7
        let r = tone(293.66, 22050.0, 11025); // D4 -> class 2
        let mut data = Array2::<f64>::zeros((2, 11025));
        data.row_mut(0).assign(&l.row(0));
        data.row_mut(1).assign(&r.row(0));
        let stereo = AudioSamples::new_multi_channel(data, sr).unwrap();
        let c = stereo.chroma_stft(&ChromaStftConfig::default()).unwrap();
        assert_eq!(c.dim(), (2, 12, 22));
        let argmax = |ch: usize| {
            let f = c.slice(ndarray::s![ch, .., 10]);
            (0..12usize).max_by(|&a, &b| f[a].total_cmp(&f[b])).unwrap()
        };
        assert_eq!(argmax(0), 7);
        assert_eq!(argmax(1), 2);
        for v in c.slice(ndarray::s![.., .., 10]).lanes(Axis(0)) {
            assert!(v.iter().all(|&x| (0.0..=1.0).contains(&x)));
        }
        // Mono with fixed tuning, no norm, 24 bins.
        let mono = AudioSamples::new_mono(l.row(0).to_owned(), sr).unwrap();
        let cfg = ChromaStftConfig::new()
            .with_tuning(Some(0.0))
            .with_norm(None)
            .with_n_chroma(nzu!(24));
        assert_eq!(mono.chroma_stft(&cfg).unwrap().dim(), (1, 24, 22));
        // Silence stays zero under max-norm.
        let silent = AudioSamples::new_mono(ndarray::Array1::<f64>::zeros(4096), sr).unwrap();
        assert!(
            silent
                .chroma_stft(&ChromaStftConfig::default())
                .unwrap()
                .iter()
                .all(|&x| x == 0.0)
        );
        // Invalid config.
        assert!(
            mono.chroma_stft(&ChromaStftConfig::new().with_tuning(Some(f64::INFINITY)))
                .is_err()
        );
    }

    #[test]
    fn short_signal_uncentred_is_error() {
        let a = AudioSamples::new_mono(ndarray::Array1::<f64>::zeros(100), sample_rate!(22050))
            .unwrap();
        let cfg = PiptrackConfig::default().with_stft(StftConfig::default().with_center(false));
        assert!(a.piptrack(&cfg).is_err());
        assert!(
            a.estimate_tuning(&TuningConfig::new().with_piptrack(cfg))
                .is_err()
        );
    }

    #[test]
    fn tuning_helpers_round_trip() {
        assert_eq!(a4_to_tuning(440.0, nzu!(12)), 0.0);
        assert!((tuning_to_a4(a4_to_tuning(443.0, nzu!(36)), nzu!(36)) - 443.0).abs() < 1e-9);
        assert!((octs_to_hz(4.0, 0.5, nzu!(12)) - 440.0 * 2f64.powf(0.5 / 12.0)).abs() < 1e-9);
        assert!(cqt_frequencies(0, 32.7, nzu!(12), 0.0).is_empty());
    }

    #[test]
    fn key_spelling_matches_librosa_examples() {
        let n = |k: &str| key_to_notes(k, true, false).unwrap().join(" ");
        assert_eq!(n("C:maj"), "C C♯ D D♯ E F F♯ G G♯ A A♯ B");
        assert_eq!(n("A:min"), "C C♯ D D♯ E F F♯ G G♯ A A♯ B");
        assert_eq!(n("A♯:min"), "B♯ C♯ D D♯ E E♯ F♯ G G♯ A A♯ B");
        assert_eq!(n("G♯:maj"), "B♯ C♯ D D♯ E E♯ F♯ F𝄪 G♯ A A♯ B");
        assert_eq!(n("F♭:min"), "D𝄫 D♭ E𝄫 E♭ F♭ F G♭ A𝄫 A♭ B𝄫 B♭ C♭");
        assert_eq!(n("G:loc"), "C D♭ D E♭ E F G♭ G A♭ A B♭ B");
        assert_eq!(
            key_to_notes("G:loc", true, true).unwrap().join(" "),
            "C D♭ D♮ E♭ E♮ F G♭ G A♭ A♮ B♭ B♮"
        );
        assert_eq!(
            key_to_notes("A♯:min", false, true).unwrap().join(" "),
            "B# C# Dn D# En E# F# Gn G# An A# Bn"
        );
        assert_eq!(key_to_degrees("A:min").unwrap(), [9, 11, 0, 2, 4, 5, 7]);
        assert_eq!(key_to_degrees("A:aeolian").unwrap(), [9, 11, 0, 2, 4, 5, 7]);
        for bad in [
            "", "C", "C:", "C:Major", "H:maj", "Cx:maj", "C:maj ", "Ç:maj",
        ] {
            assert!(key_to_notes(bad, true, false).is_err(), "{bad:?}");
            assert!(key_to_degrees(bad).is_err(), "{bad:?}");
        }
    }

    /// A resolution that would need a 10^9-bin tuning histogram (8 GB) is an
    /// error; an ordinary resolution is accepted.
    #[test]
    fn tuning_rejects_resolution_needing_a_huge_histogram() {
        assert!(pitch_tuning(&[440.0, 441.0], 1e-9, crate::nzu!(12)).is_err());
        assert!(
            TuningConfig::new()
                .with_resolution(1e-9)
                .validate()
                .is_err()
        );
        assert!(pitch_tuning(&[440.0, 441.0], 1e-4, crate::nzu!(12)).is_ok());
    }

    #[test]
    fn output_type_follows_sample_type_and_f32_agrees_with_f64() {
        let x = tone(392.0, 22050.0, 11025).mapv(|v| 0.5 * v);
        let a64 = AudioSamples::new_mono(x.row(0).to_owned(), sample_rate!(22050)).unwrap();
        let a16 = a64.to_format::<i16>();
        let r = a16.to_format::<f64>();
        let cfg = ChromaStftConfig::default();
        let c32: Array3<f32> = a16.chroma_stft(&cfg).unwrap();
        let c64 = r.chroma_stft(&cfg).unwrap();
        let err = c32
            .iter()
            .zip(c64.iter())
            .fold(0.0f64, |m, (&p, &q)| m.max((f64::from(p) - q).abs()));
        // Max-normalised chroma (peak 1): f32 round-off of the power STFT and
        // the 1025-term projection; measured 6.4e-8, bound 4e-6 (34 f32 eps).
        eprintln!("f32 vs f64 chroma_stft: {err:.3e}");
        assert!(err < 4e-6, "{err:e}");
        let (p32, m32): (Array3<f32>, Array3<f32>) =
            a16.piptrack(&PiptrackConfig::default()).unwrap();
        assert_eq!(p32.dim(), m32.dim());
        let t32 = a16.estimate_tuning(&TuningConfig::default()).unwrap();
        let t64 = r.estimate_tuning(&TuningConfig::default()).unwrap();
        assert!((t32 - t64).abs() <= 0.01 + 1e-12, "{t32} vs {t64}");
    }
}
