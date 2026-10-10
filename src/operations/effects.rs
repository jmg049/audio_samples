//! librosa-compatible audio effects: time stretching, pitch shifting,
//! pre-/de-emphasis, dB-based trimming and splitting, remixing, median-filter
//! harmonic/percussive separation, Griffin-Lim phase reconstruction,
//! nearest-neighbour filtering, LPC and click tracks.
//!
//! ## What
//!
//! The [`AudioEffects`] trait and the free functions it is built from reproduce
//! librosa 1.0's `effects` module and a few related helpers:
//!
//! | librosa | here |
//! |---|---|
//! | `librosa.phase_vocoder(D, rate=...)` / `(D, t_out=...)` | [`phase_vocoder`] / [`phase_vocoder_at`] |
//! | `librosa.effects.time_stretch` | [`AudioEffects::time_stretch`] |
//! | `librosa.effects.pitch_shift` | [`AudioEffects::pitch_shift`] |
//! | `librosa.effects.preemphasis` / `deemphasis` | [`AudioEffects::preemphasis`] / [`AudioEffects::deemphasis`] ([`preemphasis_with_state`], [`deemphasis_with_state`] for `zi` / `zf`) |
//! | `librosa.effects.trim` / `split` | [`AudioEffects::trim_db`] / [`AudioEffects::split_db`] ([`nonsilent_frames`]) |
//! | `librosa.effects.remix` | [`AudioEffects::remix`] |
//! | `librosa.decompose.hpss` | [`hpss_spectrogram`], [`hpss_magnitude`], [`hpss_masks`] |
//! | `librosa.effects.hpss` / `harmonic` / `percussive` | [`AudioEffects::harmonic_percussive`] / [`AudioEffects::harmonic`] / [`AudioEffects::percussive`] |
//! | `librosa.util.softmask` | [`softmask`] |
//! | `librosa.griffinlim` | [`griffinlim`], [`AudioEffects::griffinlim_audio`] |
//! | `librosa.decompose.nn_filter` | [`nn_filter`], [`nn_filter_recurrence`] |
//! | `librosa.lpc` | [`lpc`], [`AudioEffects::lpc`] |
//! | `librosa.clicks` | [`clicks`] |
//!
//! ## Why
//!
//! These are the standard building blocks of librosa-based pipelines (data
//! augmentation with stretch/shift, speech front-ends with pre-emphasis and
//! silence trimming, source separation with HPSS and soft masks). Every config
//! struct defaults to librosa's defaults, so `X::default()` gives librosa-equivalent
//! output, and the spectral steps reuse the librosa-compatible STFT of
//! [`crate::operations::spectrum`].
//!
//! The older [`AudioDecomposition::hpss`](crate::AudioDecomposition::hpss) and
//! [`AudioEditing::trim_silence`](crate::AudioEditing::trim_silence) keep their own
//! (different) semantics; the methods here are the librosa-compatible variants.
//!
//! ## How
//!
//! Bring [`AudioEffects`] into scope and call it on any [`AudioSamples`]. Audio is
//! processed in `f64` (converted with the crate's audio-aware conversion) and
//! returned in the input's sample type; every audio-to-audio operation has an
//! `_in_place` variant. Spectrogram-level helpers take
//! `(channels, n_bins, n_frames)` arrays as produced by
//! [`AudioSpectrum::stft_matrix`](crate::operations::spectrum::AudioSpectrum::stft_matrix).
//!
//! ```rust
//! use audio_samples::operations::effects::{AudioEffects, PitchShiftConfig, SilenceConfig};
//! use audio_samples::operations::spectrum::StftConfig;
//! use audio_samples::{sample_rate, sine_wave};
//! use std::time::Duration;
//!
//! let audio = sine_wave::<f32>(440.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
//! let n = audio.samples_per_channel().get();
//!
//! // librosa.effects.time_stretch(y, rate=2.0): half the length.
//! let fast = audio.time_stretch(2.0, &StftConfig::default()).unwrap();
//! assert_eq!(fast.samples_per_channel().get(), n / 2);
//!
//! // librosa.effects.pitch_shift(y, sr=sr, n_steps=2): same length, higher pitch.
//! let up = audio.pitch_shift(2.0, &PitchShiftConfig::default()).unwrap();
//! assert_eq!(up.samples_per_channel().get(), n);
//!
//! // librosa.effects.split(y): one non-silent interval.
//! let intervals = audio.split_db(&SilenceConfig::default()).unwrap();
//! assert_eq!(intervals.len(), 1);
//! ```

use std::num::NonZeroUsize;

use ndarray::{
    Array, Array1, Array2, Array3, ArrayBase, ArrayView2, ArrayView3, Axis, Data, Dimension,
};
use non_empty_slice::NonEmptyVec;
use num_complex::Complex;

use crate::operations::sequence::{RecurrenceConfig, recurrence_matrix};
use crate::operations::spectrum::{
    DbAxes, DbConfig, DbReference, StftConfig, amplitude_to_db, istft, stft,
};
use crate::operations::types::ResamplingQuality;
use crate::repr::SampleRate;
use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ParameterError,
    StandardSample,
};

fn param_err(parameter: &str, reason: impl ToString) -> AudioSampleError {
    AudioSampleError::Parameter(ParameterError::invalid_value(parameter, reason))
}

fn config_err(operation: &str, reason: impl ToString) -> AudioSampleError {
    AudioSampleError::Parameter(ParameterError::invalid_configuration(operation, reason))
}

// =============================================================================
// AudioSamples <-> f64 rows
// =============================================================================

/// The audio as a `(channels, samples)` `f64` array.
fn audio_rows<T: StandardSample>(audio: &AudioSamples<'_, T>) -> Array2<f64> {
    let w = audio.to_format::<f64>();
    if let Some(m) = w.as_mono() {
        m.as_view().insert_axis(Axis(0)).to_owned()
    } else if let Some(m) = w.as_multi_channel() {
        m.as_view().to_owned()
    } else {
        unreachable!("audio is either mono or multi-channel")
    }
}

/// Builds audio of sample type `T` from `(channels, samples)` rows; one row gives
/// mono audio.
fn rows_to_audio<T: StandardSample>(
    rows: Array2<f64>,
    sample_rate: SampleRate,
    operation: &str,
) -> AudioSampleResult<AudioSamples<'static, T>> {
    let n_channels = rows.nrows();
    // Channel-major data: the array's own buffer when it is in standard layout.
    let flat: Vec<f64> = if rows.is_standard_layout() {
        let (v, offset) = rows.into_raw_vec_and_offset();
        debug_assert_eq!(offset.unwrap_or(0), 0);
        v
    } else {
        rows.iter().copied().collect()
    };
    let data = NonEmptyVec::new(flat).map_err(|_| AudioSampleError::empty_data(operation))?;
    if n_channels == 1 {
        Ok(AudioSamples::from_mono_vec::<f64>(data, sample_rate))
    } else {
        let channels = u32::try_from(n_channels)
            .ok()
            .and_then(std::num::NonZeroU32::new)
            .ok_or_else(|| AudioSampleError::empty_data(operation))?;
        AudioSamples::new_multi_channel_from_vec::<f64>(data, channels, sample_rate)
    }
}

// =============================================================================
// Phase vocoder and time stretching
// =============================================================================

/// Time-stretches an STFT matrix by `rate`, equivalent to
/// `librosa.phase_vocoder(D, rate=rate)` (librosa 1.0 algorithm).
///
/// Output frames are taken at the (fractional) analysis positions
/// `t = 0, rate, 2 rate, ... < n_frames`; see [`phase_vocoder_at`] for the
/// algorithm. `rate > 1` speeds up (fewer frames), `rate < 1` slows down.
///
/// # Arguments
///
/// * `stft_matrix` - `(channels, n_bins, n_frames)` complex STFT.
/// * `rate` - Speed-up factor, finite and positive.
///
/// # Returns
///
/// A `(channels, n_bins, ceil(n_frames / rate))` complex array.
///
/// # Errors
///
/// [`ParameterError`] if `rate` is not finite and positive, or the matrix has no
/// frames, bins or channels.
///
/// # Differences from librosa
///
/// librosa 1.0 deprecated (and ignores) the `hop_length` and `n_fft` arguments;
/// they are not accepted here.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::effects::phase_vocoder;
/// use ndarray::Array3;
/// use num_complex::Complex;
///
/// let d = Array3::from_elem((1, 5, 10), Complex::new(1.0, 0.0));
/// assert_eq!(phase_vocoder(d.view(), 2.0).unwrap().dim(), (1, 5, 5));
/// assert_eq!(phase_vocoder(d.view(), 0.5).unwrap().dim(), (1, 5, 20));
/// ```
///
/// # References
///
/// J. L. Flanagan and R. M. Golden, "Phase vocoder," *Bell System Technical
/// Journal* 45(9), 1966; librosa `core.phase_vocoder`.
pub fn phase_vocoder(
    stft_matrix: ArrayView3<'_, Complex<f64>>,
    rate: f64,
) -> AudioSampleResult<Array3<Complex<f64>>> {
    if !(rate.is_finite() && rate > 0.0) {
        return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
            "rate",
            rate,
            "> 0",
            "finite",
            "the stretch rate must be a positive number",
        )));
    }
    let n_frames = stft_matrix.dim().2;
    // numpy.arange(0.0, n_frames, rate): ceil(n / rate) values i * rate.
    let n_out = (n_frames as f64 / rate).ceil() as usize;
    let t_out: Vec<f64> = (0..n_out).map(|i| i as f64 * rate).collect();
    phase_vocoder_at(stft_matrix, &t_out)
}

/// Evaluates the phase vocoder at arbitrary (fractional) frame positions,
/// equivalent to `librosa.phase_vocoder(D, t_out=t_out)` (librosa 1.0).
///
/// For output frame `j` at position `t_j`, with `i0 = floor(t_j)` and
/// `i1 = min(i0 + 1, n_frames - 1)`:
///
/// * the magnitude is `|D|` linearly interpolated at `t_j` (extrapolated from the
///   last two frames when `t_j > n_frames - 1`, as SciPy's
///   `interp1d(fill_value="extrapolate")`);
/// * the phase starts at `angle(D[:, i0_0])` and accumulates the raw frame-to-frame
///   phase differences `angle(D[:, i1]) - angle(D[:, i0])` of the previous output
///   positions (no unwrapping or bin de-trending, exactly as librosa 1.0).
///
/// # Arguments
///
/// * `stft_matrix` - `(channels, n_bins, n_frames)` complex STFT.
/// * `t_out` - Output positions in frames, each in `[0, n_frames)`. Non-monotone
///   positions are allowed (librosa warns that phase estimation may be unstable).
///
/// # Returns
///
/// A `(channels, n_bins, t_out.len())` complex array.
///
/// # Errors
///
/// [`ParameterError`] if `t_out` is empty or has a value outside `[0, n_frames)`,
/// or the matrix has no frames, bins or channels.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::effects::phase_vocoder_at;
/// use ndarray::Array3;
/// use num_complex::Complex;
///
/// let d = Array3::from_shape_fn((1, 3, 4), |(_, k, t)| Complex::new((k + t) as f64, 1.0));
/// // Evaluating at the frame positions themselves keeps the magnitudes.
/// let y = phase_vocoder_at(d.view(), &[0.0, 1.0, 2.0, 3.0]).unwrap();
/// assert!((y[[0, 2, 3]].norm() - d[[0, 2, 3]].norm()).abs() < 1e-12);
/// ```
pub fn phase_vocoder_at(
    stft_matrix: ArrayView3<'_, Complex<f64>>,
    t_out: &[f64],
) -> AudioSampleResult<Array3<Complex<f64>>> {
    let (n_channels, n_bins, n_frames) = stft_matrix.dim();
    if n_channels == 0 || n_bins == 0 || n_frames == 0 {
        return Err(param_err(
            "stft_matrix",
            "the STFT matrix has no channels, bins or frames",
        ));
    }
    if t_out.is_empty() {
        return Err(param_err(
            "t_out",
            "at least one output position is required",
        ));
    }
    if let Some(&bad) = t_out.iter().find(|&&t| !(t >= 0.0 && t < n_frames as f64)) {
        return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
            "t_out",
            bad,
            0,
            n_frames,
            "t_out values must be in the range [0, n_frames)",
        )));
    }
    let n_out = t_out.len();
    let i0: Vec<usize> = t_out.iter().map(|&t| t.floor() as usize).collect();
    let i1: Vec<usize> = i0.iter().map(|&i| (i + 1).min(n_frames - 1)).collect();
    // scipy interp1d (kind="linear"): segment lo = clip(searchsorted_left(x, t), 1, n - 1) - 1.
    let seg: Vec<(usize, usize)> = t_out
        .iter()
        .map(|&t| {
            if n_frames == 1 {
                return (0, 0);
            }
            let left = if t.fract() == 0.0 {
                t as usize
            } else {
                t.floor() as usize + 1
            };
            let hi = left.clamp(1, n_frames - 1);
            (hi - 1, hi)
        })
        .collect();

    let mut out = Array3::<Complex<f64>>::zeros((n_channels, n_bins, n_out));
    // Phase and magnitude of every input frame, computed once per bin (each is
    // used by up to two outputs per neighbouring frame pair).
    let mut arg = vec![0.0; n_frames];
    let mut mag_in = vec![0.0; n_frames];
    for c in 0..n_channels {
        for k in 0..n_bins {
            let row = stft_matrix.slice(ndarray::s![c, k, ..]);
            for ((a, m), z) in arg.iter_mut().zip(&mut mag_in).zip(row.iter()) {
                *a = z.arg();
                *m = z.norm();
            }
            let mut phase = arg[i0[0]];
            for j in 0..n_out {
                if j > 0 {
                    phase += arg[i1[j - 1]] - arg[i0[j - 1]];
                }
                let (lo, hi) = seg[j];
                let mag = if n_frames == 1 {
                    mag_in[0]
                } else {
                    let (y_lo, y_hi) = (mag_in[lo], mag_in[hi]);
                    (y_hi - y_lo) / ((hi - lo) as f64) * (t_out[j] - lo as f64) + y_lo
                };
                out[[c, k, j]] = Complex::new(phase.cos() * mag, phase.sin() * mag);
            }
        }
    }
    Ok(out)
}

/// `librosa.effects.time_stretch` on `(channels, samples)` rows.
fn time_stretch_rows(
    rows: ArrayView2<'_, f64>,
    rate: f64,
    config: &StftConfig,
) -> AudioSampleResult<Array2<f64>> {
    if !(rate.is_finite() && rate > 0.0) {
        return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
            "rate",
            rate,
            "> 0",
            "finite",
            "the stretch rate must be a positive number",
        )));
    }
    let n = rows.ncols();
    // Python's round() rounds half to even.
    let len_stretch = (n as f64 / rate).round_ties_even();
    if len_stretch < 1.0 {
        return Err(param_err(
            "rate",
            format!("rate={rate} would stretch {n} samples to an empty signal"),
        ));
    }
    let d = stft(rows, config)?;
    let d_stretch = phase_vocoder(d.view(), rate)?;
    istft(d_stretch.view(), config, Some(len_stretch as usize))
}

/// Resamples rows by an arbitrary positive ratio `out_rate / in_rate` with the
/// crate's windowed-sinc resampler.
///
/// The resampler works with integer rates, so the ratio is represented by two
/// integers near 2^31 (relative ratio error below 1e-9).
fn resample_rows_by_ratio(
    rows: Array2<f64>,
    ratio: f64,
    quality: ResamplingQuality,
) -> AudioSampleResult<Array2<f64>> {
    const BASE: f64 = 2_147_483_648.0; // 2^31
    let (in_sr, out_sr) = if ratio <= 1.0 {
        (BASE, (BASE * ratio).round())
    } else {
        ((BASE / ratio).round(), BASE)
    };
    let to_rate = |v: f64| {
        SampleRate::new(v as u32)
            .ok_or_else(|| param_err("ratio", format!("resampling ratio {ratio} is out of range")))
    };
    let (in_sr, out_sr) = (to_rate(in_sr)?, to_rate(out_sr)?);
    let n_channels = rows.nrows();
    let audio: AudioSamples<'static, f64> = rows_to_audio(rows, in_sr, "pitch_shift")?;
    let y = crate::resampling::resample(&audio, out_sr, quality)?;
    let out = audio_rows(&y);
    debug_assert_eq!(out.nrows(), n_channels);
    Ok(out)
}

/// Configuration of [`AudioEffects::pitch_shift`], mirroring the keyword
/// arguments of `librosa.effects.pitch_shift`.
///
/// ## Purpose
///
/// Sets the pitch-step size, the resampler quality and the STFT used by the
/// underlying time stretch.
///
/// ## Intended usage
///
/// `Default` reproduces librosa's defaults (12 bins per octave, n_fft 2048, hop
/// 512, periodic Hann) with [`ResamplingQuality::High`], this crate's closest match
/// to librosa's default `res_type="soxr_hq"`.
///
/// ## Invariants
///
/// Checked by [`PitchShiftConfig::validate`]: the STFT configuration is valid.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::effects::PitchShiftConfig;
///
/// // Quarter-tone steps.
/// let cfg = PitchShiftConfig::new().with_bins_per_octave(nzu!(24));
/// assert_eq!(cfg.bins_per_octave.get(), 24);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PitchShiftConfig {
    /// Steps per octave: `n_steps` is measured in `1 / bins_per_octave` octaves.
    /// librosa default: 12 (semitones).
    pub bins_per_octave: NonZeroUsize,
    /// Quality of the final resampling step. librosa uses soxr (`soxr_hq`); the
    /// default here is [`ResamplingQuality::High`].
    pub quality: ResamplingQuality,
    /// STFT used by the phase-vocoder time stretch. librosa default.
    pub stft: StftConfig,
}

impl Default for PitchShiftConfig {
    fn default() -> Self {
        Self {
            bins_per_octave: crate::nzu!(12),
            quality: ResamplingQuality::High,
            stft: StftConfig::default(),
        }
    }
}

impl PitchShiftConfig {
    /// Returns the librosa-default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of steps per octave.
    #[must_use]
    pub fn with_bins_per_octave(mut self, bins_per_octave: NonZeroUsize) -> Self {
        self.bins_per_octave = bins_per_octave;
        self
    }

    /// Sets the resampling quality.
    #[must_use]
    pub fn with_quality(mut self, quality: ResamplingQuality) -> Self {
        self.quality = quality;
        self
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

/// `librosa.effects.pitch_shift` on `(channels, samples)` rows.
fn pitch_shift_rows(
    rows: ArrayView2<'_, f64>,
    n_steps: f64,
    config: &PitchShiftConfig,
) -> AudioSampleResult<Array2<f64>> {
    config.validate()?;
    if !n_steps.is_finite() {
        return Err(param_err("n_steps", "n_steps must be finite"));
    }
    let n = rows.ncols();
    let rate = 2.0f64.powf(-n_steps / config.bins_per_octave.get() as f64);
    let stretched = time_stretch_rows(rows, rate, &config.stft)?;
    // librosa: resample(orig_sr = sr / rate, target_sr = sr), i.e. by the ratio `rate`.
    let shifted = resample_rows_by_ratio(stretched, rate, config.quality)?;
    Ok(fix_length(shifted.view(), n))
}

/// Truncates or zero-pads every row to `n` samples (`librosa.util.fix_length`).
fn fix_length(rows: ArrayView2<'_, f64>, n: usize) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((rows.nrows(), n));
    let m = rows.ncols().min(n);
    out.slice_mut(ndarray::s![.., ..m])
        .assign(&rows.slice(ndarray::s![.., ..m]));
    out
}

// =============================================================================
// Pre-emphasis / de-emphasis
// =============================================================================

fn check_coef(coef: f64) -> AudioSampleResult<()> {
    if coef.is_finite() {
        Ok(())
    } else {
        Err(param_err("coef", "the filter coefficient must be finite"))
    }
}

/// Pre-emphasis filter `y[n] = x[n] - coef * x[n - 1]` with an explicit filter
/// state, equivalent to `librosa.effects.preemphasis(x, coef=coef, zi=zi,
/// return_zf=True)`.
///
/// The filter is `scipy.signal.lfilter([1, -coef], [1], x, zi=[zi])`, so the first
/// output is `y[0] = x[0] + zi`. Passing the returned final state as `zi` of the
/// next block filters a long signal block by block with the same result as one
/// pass.
///
/// # Arguments
///
/// * `x` - Input samples.
/// * `coef` - Pre-emphasis coefficient (librosa default 0.97). Any finite value;
///   `0` is the identity (apart from `zi`), `1` a first difference.
/// * `zi` - Initial filter state. `None` uses librosa's default `2 x[0] - x[1]`.
///
/// # Returns
///
/// `(y, zf)`: the filtered samples and the final state `-coef * x[n - 1]`.
///
/// # Errors
///
/// [`ParameterError`] if `coef` or `zi` is not finite, `x` is empty, or `zi` is
/// `None` and `x` has fewer than two samples.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::effects::preemphasis_with_state;
///
/// let (y, zf) = preemphasis_with_state(&[1.0, 3.0, 0.5], 0.97, None).unwrap();
/// // zi = 2 * 1 - 3 = -1, so y[0] = 1 + (-1) = 0.
/// assert_eq!(y[0], 0.0);
/// assert!((y[1] - (3.0 - 0.97)).abs() < 1e-15);
/// assert!((zf + 0.97 * 0.5).abs() < 1e-15);
/// ```
pub fn preemphasis_with_state(
    x: &[f64],
    coef: f64,
    zi: Option<f64>,
) -> AudioSampleResult<(Vec<f64>, f64)> {
    check_coef(coef)?;
    let zi = match zi {
        Some(z) if z.is_finite() => z,
        Some(_) => return Err(param_err("zi", "the initial state must be finite")),
        None if x.len() >= 2 => 2.0 * x[0] - x[1],
        None => {
            return Err(param_err(
                "x",
                "the default initial state 2*x[0] - x[1] needs at least two samples",
            ));
        }
    };
    if x.is_empty() {
        return Err(AudioSampleError::empty_data("preemphasis"));
    }
    let mut y = Vec::with_capacity(x.len());
    let mut z = zi;
    for &v in x {
        y.push(v + z);
        z = -coef * v;
    }
    Ok((y, z))
}

/// De-emphasis filter `y[n] = x[n] + coef * y[n - 1]` (the inverse of
/// [`preemphasis_with_state`]), equivalent to `librosa.effects.deemphasis(x,
/// coef=coef, zi=zi, return_zf=True)`.
///
/// With an explicit `zi` this is `scipy.signal.lfilter([1], [1, -coef], x,
/// zi=[zi])`, so `y[0] = x[0] + zi`. With `zi = None` (librosa's default) the
/// filter starts from a zero state and then subtracts
/// `((2 - coef) x[0] - x[1]) / (3 - coef) * coef^n`, which exactly inverts
/// pre-emphasis with its default initial state.
///
/// # Arguments
///
/// * `x` - Input samples.
/// * `coef` - Coefficient (librosa default 0.97); any finite value (`coef = 3` is
///   only allowed with an explicit `zi`).
/// * `zi` - Initial filter state, or `None` for librosa's default.
///
/// # Returns
///
/// `(y, zf)`: the filtered samples and the final filter state `coef * y'[n - 1]`,
/// where `y'` is the output *before* the default-state correction (as librosa
/// returns it).
///
/// # Errors
///
/// [`ParameterError`] if `coef` or `zi` is not finite, `x` is empty, or `zi` is
/// `None` and `x` has fewer than two samples or `coef == 3`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::effects::{deemphasis_with_state, preemphasis_with_state};
///
/// let x = [0.1, -0.4, 0.25, 0.9, -0.3];
/// let (pre, _) = preemphasis_with_state(&x, 0.97, None).unwrap();
/// let (back, _) = deemphasis_with_state(&pre, 0.97, None).unwrap();
/// assert!(back.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-12));
/// ```
pub fn deemphasis_with_state(
    x: &[f64],
    coef: f64,
    zi: Option<f64>,
) -> AudioSampleResult<(Vec<f64>, f64)> {
    check_coef(coef)?;
    if x.is_empty() {
        return Err(AudioSampleError::empty_data("deemphasis"));
    }
    let (z0, correction) = match zi {
        Some(z) if z.is_finite() => (z, None),
        Some(_) => return Err(param_err("zi", "the initial state must be finite")),
        None => {
            if x.len() < 2 {
                return Err(param_err(
                    "x",
                    "the default initial state needs at least two samples",
                ));
            }
            if coef == 3.0 {
                return Err(param_err(
                    "coef",
                    "coef = 3 makes librosa's default initial state undefined; pass zi explicitly",
                ));
            }
            (0.0, Some(((2.0 - coef) * x[0] - x[1]) / (3.0 - coef)))
        }
    };
    let mut y = Vec::with_capacity(x.len());
    let mut z = z0;
    for &v in x {
        let out = v + z;
        y.push(out);
        z = coef * out;
    }
    if let Some(c) = correction {
        let mut p = 1.0;
        for v in &mut y {
            *v -= c * p;
            p *= coef;
        }
    }
    Ok((y, z))
}

// =============================================================================
// dB-based silence detection: trim / split
// =============================================================================

/// How per-channel frame levels are combined into one decision per frame
/// (librosa's `aggregate` callable).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ChannelAggregate {
    /// Loudest channel (`numpy.max`, librosa's default): a frame is non-silent if
    /// any channel is.
    #[default]
    Max,
    /// Quietest channel (`numpy.min`): a frame is non-silent only if every channel is.
    Min,
    /// Mean of the channels' dB levels (`numpy.mean`).
    Mean,
}

/// Configuration of dB-based silence detection ([`AudioEffects::trim_db`],
/// [`AudioEffects::split_db`], [`nonsilent_frames`]), mirroring the arguments of
/// `librosa.effects.trim` / `split`.
///
/// ## Purpose
///
/// A frame is non-silent when its RMS level, in dB relative to `reference`, is
/// above `-top_db`.
///
/// ## Intended usage
///
/// `Default` reproduces librosa's defaults: `top_db = 60`, reference = the loudest
/// frame (`ref=numpy.max`), 2048-sample frames, 512-sample hop, `numpy.max` over
/// channels.
///
/// ## Invariants
///
/// Checked by [`SilenceConfig::validate`]: `top_db` finite, a `Value` reference
/// finite.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::effects::SilenceConfig;
///
/// let cfg = SilenceConfig::new().with_top_db(30.0);
/// assert_eq!(cfg.frame_length.get(), 2048);
/// assert!(cfg.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SilenceConfig {
    /// Threshold in dB below the reference under which a frame is silent.
    /// librosa default: 60.
    pub top_db: f64,
    /// Reference level (0 dB). [`DbReference::Max`] (librosa's default) uses the
    /// loudest frame over all channels; `Value(r)` an absolute RMS amplitude.
    pub reference: DbReference,
    /// Samples per RMS frame (frames are centred, zero padded). librosa default: 2048.
    pub frame_length: NonZeroUsize,
    /// Samples between frames; also the resolution of the returned sample
    /// indices. librosa default: 512.
    pub hop_length: NonZeroUsize,
    /// Combination of the channels' dB levels. librosa default: `Max`.
    pub aggregate: ChannelAggregate,
}

impl Default for SilenceConfig {
    fn default() -> Self {
        Self {
            top_db: 60.0,
            reference: DbReference::Max,
            frame_length: crate::nzu!(2048),
            hop_length: crate::nzu!(512),
            aggregate: ChannelAggregate::Max,
        }
    }
}

impl SilenceConfig {
    /// Returns the librosa-default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the threshold in dB below the reference.
    #[must_use]
    pub fn with_top_db(mut self, top_db: f64) -> Self {
        self.top_db = top_db;
        self
    }

    /// Sets the reference level.
    #[must_use]
    pub fn with_reference(mut self, reference: DbReference) -> Self {
        self.reference = reference;
        self
    }

    /// Sets the RMS frame length.
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

    /// Sets the channel aggregation.
    #[must_use]
    pub fn with_aggregate(mut self, aggregate: ChannelAggregate) -> Self {
        self.aggregate = aggregate;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `top_db` is not finite or a `Value` reference is not finite.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if !self.top_db.is_finite() {
            return Err(param_err("top_db", "top_db must be finite"));
        }
        if let DbReference::Value(r) = self.reference {
            if !r.is_finite() {
                return Err(param_err("reference", "reference value must be finite"));
            }
        }
        Ok(())
    }
}

/// Classifies frames as non-silent, equivalent to librosa's internal
/// `_signal_to_frame_nonsilent` (used by `effects.trim` and `effects.split`).
///
/// Computes the centred, zero-padded frame RMS of every channel
/// (`librosa.feature.rms`), converts it to dB with `amplitude_to_db(ref=...,
/// top_db=None)` (the `Max` reference is the maximum over all channels and
/// frames), combines channels with `config.aggregate` and compares with
/// `-top_db`.
///
/// # Arguments
///
/// * `signal` - `(channels, samples)` array with at least one sample.
/// * `config` - Threshold, reference, framing and channel aggregation.
///
/// # Returns
///
/// One flag per frame (`1 + len / hop` frames); frame `t` covers samples centred
/// on `t * hop`.
///
/// # Errors
///
/// [`ParameterError`] if the configuration is invalid or the signal is empty.
///
/// # Differences from librosa
///
/// librosa computes the RMS in `float32` (`feature.rms` default dtype); this
/// function uses `f64`, so a frame whose level is within about 1e-6 dB of the
/// threshold can be classified differently.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::effects::{SilenceConfig, nonsilent_frames};
/// use ndarray::Array2;
///
/// // Silence, then a loud segment.
/// let x = Array2::from_shape_fn((1, 4096), |(_, n)| if n >= 2048 { 0.5 } else { 0.0 });
/// let cfg = SilenceConfig::new().with_frame_length(nzu!(512)).with_hop_length(nzu!(256));
/// let flags = nonsilent_frames(x.view(), &cfg).unwrap();
/// assert_eq!(flags.len(), 17);
/// assert!(!flags[0] && flags[16]);
/// ```
pub fn nonsilent_frames(
    signal: ArrayView2<'_, f64>,
    config: &SilenceConfig,
) -> AudioSampleResult<Vec<bool>> {
    config.validate()?;
    let (n_channels, len) = signal.dim();
    if len == 0 || n_channels == 0 {
        return Err(AudioSampleError::empty_data("nonsilent_frames"));
    }
    let fl = config.frame_length.get();
    let hop = config.hop_length.get();
    let pad = fl / 2;
    let n_frames = 1 + (len + 2 * pad - fl) / hop;
    let mut rms = Array2::<f64>::zeros((n_channels, n_frames));
    for (c, row) in signal.outer_iter().enumerate() {
        for t in 0..n_frames {
            // Padded frame [t*hop, t*hop + fl) maps to signal [t*hop - pad, ...).
            let start = (t * hop) as isize - pad as isize;
            let lo = start.max(0) as usize;
            let hi = ((start + fl as isize).max(0) as usize).min(len);
            let sum: f64 = if lo < hi {
                row.slice(ndarray::s![lo..hi]).iter().map(|v| v * v).sum()
            } else {
                0.0
            };
            rms[[c, t]] = (sum / fl as f64).sqrt();
        }
    }
    let db_cfg = DbConfig::amplitude()
        .with_reference(config.reference)
        .with_top_db(None)
        .with_axes(DbAxes::All);
    let db = amplitude_to_db(&rms, &db_cfg)?;
    let threshold = -config.top_db;
    Ok((0..n_frames)
        .map(|t| {
            let col = db.column(t);
            let v = match config.aggregate {
                ChannelAggregate::Max => col.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                ChannelAggregate::Min => col.iter().copied().fold(f64::INFINITY, f64::min),
                ChannelAggregate::Mean => col.iter().sum::<f64>() / n_channels as f64,
            };
            v > threshold
        })
        .collect())
}

/// `librosa.effects.trim` bounds: `(start, end)` sample indices (`start == end`
/// when everything is silent).
fn trim_bounds(flags: &[bool], hop: usize, len: usize) -> (usize, usize) {
    let first = flags.iter().position(|&f| f);
    let last = flags.iter().rposition(|&f| f);
    match (first, last) {
        (Some(a), Some(b)) => (a * hop, ((b + 1) * hop).min(len)),
        _ => (0, 0),
    }
}

/// `librosa.effects.split` intervals in samples.
fn split_intervals(flags: &[bool], hop: usize, len: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    if flags.first() == Some(&true) {
        edges.push(0);
    }
    for i in 1..flags.len() {
        if flags[i] != flags[i - 1] {
            edges.push(i);
        }
    }
    if flags.last() == Some(&true) {
        edges.push(flags.len());
    }
    edges
        .chunks_exact(2)
        .map(|p| ((p[0] * hop).min(len), (p[1] * hop).min(len)))
        .collect()
}

// =============================================================================
// Remix
// =============================================================================

/// Indices `i` where `librosa.zero_crossings(y, threshold=1e-10, zero_pos=True,
/// pad=True)` is true (index 0 always included), followed by `len`.
fn zero_crossing_positions(y: &[f64]) -> Vec<usize> {
    const THRESHOLD: f64 = 1e-10;
    let clip = |v: f64| {
        if (-THRESHOLD..=THRESHOLD).contains(&v) {
            0.0
        } else {
            v
        }
    };
    let mut out = vec![0];
    for i in 1..y.len() {
        if clip(y[i]).is_sign_negative() != clip(y[i - 1]).is_sign_negative() {
            out.push(i);
        }
    }
    out.push(y.len());
    out
}

/// Index into `sorted_to` of the event nearest `v`, with librosa's
/// `util.match_events` tie-breaking (left/right both allowed).
fn match_event(v: usize, sorted_to: &[usize]) -> usize {
    let n = sorted_to.len();
    let mut middle = sorted_to.partition_point(|&x| x < v);
    if middle == n {
        middle -= 1;
    }
    let diff = |i: usize| sorted_to[i].abs_diff(v);
    let mid_diff = diff(middle);
    let left = (middle > 0).then(|| middle - 1);
    let right = (middle + 1 < n).then_some(middle + 1);
    let left_diff = left.map_or(0, diff);
    let right_diff = right.map_or(0, diff);
    if let Some(l) = left {
        if left_diff < mid_diff && (right.is_none() || left_diff < right_diff) {
            return l;
        }
    }
    if let Some(r) = right {
        if right_diff < mid_diff {
            return r;
        }
    }
    middle
}

/// `librosa.effects.remix` on `(channels, samples)` rows.
fn remix_rows(
    rows: ArrayView2<'_, f64>,
    intervals: &[(usize, usize)],
    align_zeros: bool,
) -> AudioSampleResult<Array2<f64>> {
    if intervals.is_empty() {
        return Err(param_err("intervals", "at least one interval is required"));
    }
    let (n_channels, len) = rows.dim();
    let zeros = if align_zeros {
        let mono: Vec<f64> = if n_channels == 1 {
            rows.row(0).to_vec()
        } else {
            rows.mean_axis(Axis(0))
                .map(|m| m.to_vec())
                .unwrap_or_default()
        };
        Some(zero_crossing_positions(&mono))
    } else {
        None
    };
    let mut pieces = Vec::with_capacity(intervals.len());
    for &(a, b) in intervals {
        let (a, b) = match &zeros {
            Some(z) => (z[match_event(a, z)], z[match_event(b, z)]),
            None => (a, b),
        };
        // numpy slicing semantics: clipped to the signal, empty when a >= b.
        let (a, b) = (a.min(len), b.min(len));
        if a < b {
            pieces.push(rows.slice(ndarray::s![.., a..b]));
        }
    }
    if pieces.is_empty() {
        return Err(AudioSampleError::empty_data("remix"));
    }
    ndarray::concatenate(Axis(1), &pieces)
        .map_err(|e| AudioSampleError::Layout(crate::LayoutError::shape_error("remix", e)))
}

// =============================================================================
// Soft masks and median-filtering HPSS
// =============================================================================

/// `x^p` with numpy's exact fast paths for the common exponents.
#[inline]
fn pow_fast(x: f64, p: f64) -> f64 {
    if p == 1.0 {
        x
    } else if p == 2.0 {
        x * x
    } else if p == 0.5 {
        x.sqrt()
    } else {
        x.powf(p)
    }
}

/// Soft (Wiener-like) mask of `x` against `x_ref`, equivalent to
/// `librosa.util.softmask(X, X_ref, power=power, split_zeros=split_zeros)`.
///
/// With `Z = max(X, X_ref)` the mask is
/// `(X / Z)^p / ((X / Z)^p + (X_ref / Z)^p)`, i.e. `X^p / (X^p + X_ref^p)`
/// computed without overflow. Where both inputs are (numerically) zero the mask is
/// `0.5` if `split_zeros`, else `0`. With `power = f64::INFINITY` the mask is
/// binary: `1` where `X > X_ref` (ties give `0`).
///
/// # Arguments
///
/// * `x` - Non-negative values the mask selects.
/// * `x_ref` - Non-negative reference values, same shape.
/// * `power` - Exponent, positive (`f64::INFINITY` for a hard mask).
/// * `split_zeros` - Value of the mask where both inputs are zero: `0.5` or `0`.
///
/// # Returns
///
/// An array of the same shape with values in `[0, 1]`.
///
/// # Errors
///
/// [`ParameterError`] if the shapes differ, any input is negative (or NaN), or
/// `power` is not positive.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::effects::softmask;
/// use ndarray::array;
///
/// let x = array![3.0, 0.0, 1.0];
/// let r = array![4.0, 0.0, 1.0];
/// let m = softmask(&x, &r, 2.0, true).unwrap();
/// assert!((m[0] - 9.0 / 25.0).abs() < 1e-15);
/// assert_eq!(m[1], 0.5);
/// assert_eq!(softmask(&x, &r, f64::INFINITY, false).unwrap(), array![0.0, 0.0, 0.0]);
/// ```
pub fn softmask<S1, S2, D>(
    x: &ArrayBase<S1, D>,
    x_ref: &ArrayBase<S2, D>,
    power: f64,
    split_zeros: bool,
) -> AudioSampleResult<Array<f64, D>>
where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    D: Dimension,
{
    if x.shape() != x_ref.shape() {
        return Err(config_err(
            "softmask",
            format!("shape mismatch: {:?} != {:?}", x.shape(), x_ref.shape()),
        ));
    }
    if x.iter().chain(x_ref.iter()).any(|v| v.is_nan() || *v < 0.0) {
        return Err(param_err("x", "X and X_ref must be non-negative"));
    }
    if power.is_nan() || power <= 0.0 {
        return Err(param_err("power", "power must be strictly positive"));
    }
    let mut out = Array::<f64, D>::zeros(x.raw_dim());
    ndarray::Zip::from(&mut out)
        .and(x)
        .and(x_ref)
        .for_each(|m, &a, &b| {
            *m = if power.is_finite() {
                let z = a.max(b);
                if z < f64::MIN_POSITIVE {
                    if split_zeros { 0.5 } else { 0.0 }
                } else {
                    let ma = pow_fast(a / z, power);
                    let mb = pow_fast(b / z, power);
                    ma / (ma + mb)
                }
            } else if a > b {
                1.0
            } else {
                0.0
            };
        });
    Ok(out)
}

/// Median filter of every channel of `s` along `axis` (1 = frequency,
/// 2 = time) with kernel `k`, equivalent to `scipy.ndimage.median_filter` with
/// `mode="reflect"` (half-sample symmetric: `d c b a | a b c d | d c b a`) and
/// size 1 on the other axes. Even kernels select the upper median (SciPy's rank
/// `k / 2`), centred as SciPy does (`k / 2` samples before, `k / 2 - 1` after).
fn median_filter_axis(s: ArrayView3<'_, f64>, axis: usize, k: usize) -> Array3<f64> {
    let mut out = Array3::<f64>::zeros(s.raw_dim());
    let n = s.len_of(Axis(axis));
    let half = (k / 2) as isize;
    let reflect = |i: isize| -> usize {
        let period = 2 * n as isize;
        let m = i.rem_euclid(period);
        if m < n as isize {
            m as usize
        } else {
            (period - 1 - m) as usize
        }
    };
    let idx: Vec<Vec<usize>> = (0..n as isize)
        .map(|i| (0..k as isize).map(|j| reflect(i - half + j)).collect())
        .collect();
    let mut line = vec![0.0; n];
    let mut buf = vec![0.0; k];
    for (src, mut dst) in s
        .lanes(Axis(axis))
        .into_iter()
        .zip(out.lanes_mut(Axis(axis)))
    {
        for (l, v) in line.iter_mut().zip(src.iter()) {
            *l = *v;
        }
        for (i, window) in idx.iter().enumerate() {
            for (b, &j) in buf.iter_mut().zip(window) {
                *b = line[j];
            }
            let (_, m, _) = buf.select_nth_unstable_by(k / 2, f64::total_cmp);
            dst[i] = *m;
        }
    }
    out
}

/// Configuration of median-filtering harmonic/percussive separation on a
/// spectrogram, mirroring `librosa.decompose.hpss`'s arguments.
///
/// ## Purpose
///
/// Sets the median-filter sizes and how the two filtered spectrograms are turned
/// into masks.
///
/// ## Intended usage
///
/// Pass to [`hpss_spectrogram`] / [`hpss_magnitude`] / [`hpss_masks`], or embed in a
/// [`HarmonicPercussiveConfig`]. `Default` reproduces librosa's defaults: kernel
/// size 31 for both filters, power 2, margin 1.
///
/// ## Invariants
///
/// Checked by [`HpssMaskConfig::validate`]: `power > 0` (may be infinite, for hard
/// masks), both margins finite and `>= 1`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::effects::HpssMaskConfig;
///
/// let cfg = HpssMaskConfig::new().with_kernel_size(nzu!(17)).with_margin(2.0);
/// assert_eq!(cfg.kernel_harmonic.get(), 17);
/// assert_eq!(cfg.margin_percussive, 2.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HpssMaskConfig {
    /// Length (in frames) of the median filter along time that estimates the
    /// harmonic component. librosa default: 31.
    pub kernel_harmonic: NonZeroUsize,
    /// Length (in bins) of the median filter along frequency that estimates the
    /// percussive component. librosa default: 31.
    pub kernel_percussive: NonZeroUsize,
    /// Mask exponent ([`softmask`]); `f64::INFINITY` gives binary masks.
    /// librosa default: 2.
    pub power: f64,
    /// Harmonic margin: the harmonic mask is `softmask(H, margin * P)`. Values above 1
    /// make the separation stricter and leave a residual. librosa default: 1.
    pub margin_harmonic: f64,
    /// Percussive margin (as `margin_harmonic`, for `softmask(P, margin * H)`).
    /// librosa default: 1.
    pub margin_percussive: f64,
}

impl Default for HpssMaskConfig {
    fn default() -> Self {
        Self {
            kernel_harmonic: crate::nzu!(31),
            kernel_percussive: crate::nzu!(31),
            power: 2.0,
            margin_harmonic: 1.0,
            margin_percussive: 1.0,
        }
    }
}

impl HpssMaskConfig {
    /// Returns the librosa-default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets both median-filter sizes (librosa's scalar `kernel_size`).
    #[must_use]
    pub fn with_kernel_size(mut self, kernel_size: NonZeroUsize) -> Self {
        self.kernel_harmonic = kernel_size;
        self.kernel_percussive = kernel_size;
        self
    }

    /// Sets the harmonic and percussive filter sizes (librosa's tuple `kernel_size`).
    #[must_use]
    pub fn with_kernel_sizes(mut self, harmonic: NonZeroUsize, percussive: NonZeroUsize) -> Self {
        self.kernel_harmonic = harmonic;
        self.kernel_percussive = percussive;
        self
    }

    /// Sets the mask exponent.
    #[must_use]
    pub fn with_power(mut self, power: f64) -> Self {
        self.power = power;
        self
    }

    /// Sets both margins (librosa's scalar `margin`).
    #[must_use]
    pub fn with_margin(mut self, margin: f64) -> Self {
        self.margin_harmonic = margin;
        self.margin_percussive = margin;
        self
    }

    /// Sets the harmonic and percussive margins (librosa's tuple `margin`).
    #[must_use]
    pub fn with_margins(mut self, harmonic: f64, percussive: f64) -> Self {
        self.margin_harmonic = harmonic;
        self.margin_percussive = percussive;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `power` is not positive or a margin is below 1 or not finite.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if self.power.is_nan() || self.power <= 0.0 {
            return Err(param_err("power", "power must be strictly positive"));
        }
        for (name, m) in [
            ("margin_harmonic", self.margin_harmonic),
            ("margin_percussive", self.margin_percussive),
        ] {
            if !(m.is_finite() && m >= 1.0) {
                return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                    name,
                    m,
                    1,
                    "finite",
                    "margins must be >= 1.0 (a typical range is 1 to 10)",
                )));
            }
        }
        Ok(())
    }
}

/// Harmonic and percussive masks of a magnitude spectrogram, equivalent to
/// `librosa.decompose.hpss(S, ..., mask=True)` for a real `S`.
///
/// `H` is `S` median-filtered along time (`kernel_harmonic` frames), `P` along
/// frequency (`kernel_percussive` bins), both with SciPy's `reflect` boundary;
/// the masks are `softmask(H, margin_h * P)` and `softmask(P, margin_p * H)`, with
/// `split_zeros` only when both margins are 1.
///
/// # Arguments
///
/// * `magnitude` - `(channels, n_bins, n_frames)` non-negative spectrogram.
/// * `config` - Filter sizes, mask power and margins.
///
/// # Returns
///
/// `(mask_harmonic, mask_percussive)`, each the shape of `magnitude`. With margins
/// of 1 and finite power they sum to 1.
///
/// # Errors
///
/// [`ParameterError`] if the configuration is invalid or `magnitude` has a negative
/// value.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::effects::{HpssMaskConfig, hpss_masks};
/// use ndarray::Array3;
///
/// // A horizontal line (harmonic) in an otherwise empty spectrogram.
/// let s = Array3::from_shape_fn((1, 40, 50), |(_, k, _)| if k == 20 { 1.0 } else { 0.0 });
/// let (mh, mp) = hpss_masks(s.view(), &HpssMaskConfig::default()).unwrap();
/// assert_eq!(mh[[0, 20, 25]], 1.0);
/// assert_eq!(mp[[0, 20, 25]], 0.0);
/// ```
///
/// # References
///
/// D. FitzGerald, "Harmonic/percussive separation using median filtering," DAFx
/// 2010; J. Driedger, M. Müller and S. Disch, "Extending harmonic-percussive
/// separation of audio signals," ISMIR 2014 (margins).
pub fn hpss_masks(
    magnitude: ArrayView3<'_, f64>,
    config: &HpssMaskConfig,
) -> AudioSampleResult<(Array3<f64>, Array3<f64>)> {
    config.validate()?;
    if magnitude.iter().any(|v| v.is_nan() || *v < 0.0) {
        return Err(param_err(
            "magnitude",
            "the spectrogram must be non-negative",
        ));
    }
    let harm = median_filter_axis(magnitude, 2, config.kernel_harmonic.get());
    let perc = median_filter_axis(magnitude, 1, config.kernel_percussive.get());
    let split_zeros = config.margin_harmonic == 1.0 && config.margin_percussive == 1.0;
    let mh = softmask(
        &harm,
        &perc.mapv(|v| v * config.margin_harmonic),
        config.power,
        split_zeros,
    )?;
    let mp = softmask(
        &perc,
        &harm.mapv(|v| v * config.margin_percussive),
        config.power,
        split_zeros,
    )?;
    Ok((mh, mp))
}

/// Separates a magnitude (or power) spectrogram into harmonic and percussive
/// parts, equivalent to `librosa.decompose.hpss(S)` for a real `S`.
///
/// # Arguments
///
/// * `magnitude` - `(channels, n_bins, n_frames)` non-negative spectrogram.
/// * `config` - Filter sizes, mask power and margins.
///
/// # Returns
///
/// `(S * mask_harmonic, S * mask_percussive)`; see [`hpss_masks`].
///
/// # Errors
///
/// As [`hpss_masks`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::effects::{HpssMaskConfig, hpss_magnitude};
/// use ndarray::Array3;
///
/// let s = Array3::from_shape_fn((1, 64, 64), |(_, k, t)| if k == 10 || t == 32 { 1.0 } else { 0.0 });
/// let (h, p) = hpss_magnitude(s.view(), &HpssMaskConfig::default()).unwrap();
/// assert!(h[[0, 10, 5]] > 0.9 && p[[0, 40, 32]] > 0.9);
/// ```
pub fn hpss_magnitude(
    magnitude: ArrayView3<'_, f64>,
    config: &HpssMaskConfig,
) -> AudioSampleResult<(Array3<f64>, Array3<f64>)> {
    let (mh, mp) = hpss_masks(magnitude, config)?;
    Ok((&magnitude * &mh, &magnitude * &mp))
}

/// Separates a complex STFT into harmonic and percussive STFTs, equivalent to
/// `librosa.decompose.hpss(D)` for a complex `D`.
///
/// The masks of [`hpss_masks`] are computed on `|D|` and applied to the magnitude;
/// the phase of `D` is kept (`(|D| * mask) * D / |D|`, with phase 1 where `D` is 0).
///
/// # Arguments
///
/// * `stft_matrix` - `(channels, n_bins, n_frames)` complex STFT.
/// * `config` - Filter sizes, mask power and margins.
///
/// # Returns
///
/// `(D_harmonic, D_percussive)`, each the shape of `stft_matrix`.
///
/// # Errors
///
/// [`ParameterError`] if the configuration is invalid.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::effects::{HpssMaskConfig, hpss_spectrogram};
/// use audio_samples::operations::spectrum::{AudioSpectrum, StftConfig};
/// use audio_samples::{sample_rate, sine_wave};
/// use std::time::Duration;
///
/// let x = sine_wave::<f64>(440.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
/// let d = x.stft_matrix(&StftConfig::default()).unwrap();
/// let (h, p) = hpss_spectrogram(d.view(), &HpssMaskConfig::default()).unwrap();
/// let energy = |a: &ndarray::Array3<num_complex::Complex<f64>>| a.iter().map(|z| z.norm_sqr()).sum::<f64>();
/// assert!(energy(&h) > 100.0 * energy(&p)); // a steady tone is harmonic
/// ```
#[allow(clippy::type_complexity)]
pub fn hpss_spectrogram(
    stft_matrix: ArrayView3<'_, Complex<f64>>,
    config: &HpssMaskConfig,
) -> AudioSampleResult<(Array3<Complex<f64>>, Array3<Complex<f64>>)> {
    let mag = stft_matrix.mapv(Complex::norm);
    let (mh, mp) = hpss_masks(mag.view(), config)?;
    let apply = |mask: &Array3<f64>| {
        let mut out = Array3::<Complex<f64>>::zeros(stft_matrix.raw_dim());
        ndarray::Zip::from(&mut out)
            .and(&stft_matrix)
            .and(&mag)
            .and(mask)
            .for_each(|o, d, &s, &m| {
                // librosa.magphase: phase = D / |D|, or 1 where |D| == 0.
                let phase = if s == 0.0 {
                    Complex::new(1.0, d.im)
                } else {
                    Complex::new(d.re / s, d.im / s)
                };
                let v = s * m;
                *o = Complex::new(v * phase.re, v * phase.im);
            });
        out
    };
    Ok((apply(&mh), apply(&mp)))
}

/// Configuration of the audio-level harmonic/percussive separation
/// ([`AudioEffects::harmonic`], [`AudioEffects::percussive`],
/// [`AudioEffects::harmonic_percussive`]), mirroring `librosa.effects.hpss`.
///
/// ## Purpose
///
/// Bundles the STFT used for analysis/resynthesis with the mask parameters.
///
/// ## Intended usage
///
/// `Default` reproduces `librosa.effects.hpss(y)`.
///
/// ## Invariants
///
/// Checked by [`HarmonicPercussiveConfig::validate`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::effects::{HarmonicPercussiveConfig, HpssMaskConfig};
///
/// let cfg = HarmonicPercussiveConfig::new().with_hpss(HpssMaskConfig::new().with_margin(3.0));
/// assert!(cfg.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct HarmonicPercussiveConfig {
    /// STFT for analysis and resynthesis. librosa default.
    pub stft: StftConfig,
    /// Median-filter and mask parameters. librosa default.
    pub hpss: HpssMaskConfig,
}

impl HarmonicPercussiveConfig {
    /// Returns the librosa-default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the STFT configuration.
    #[must_use]
    pub fn with_stft(mut self, stft: StftConfig) -> Self {
        self.stft = stft;
        self
    }

    /// Sets the mask configuration.
    #[must_use]
    pub fn with_hpss(mut self, hpss: HpssMaskConfig) -> Self {
        self.hpss = hpss;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the STFT or mask configuration is invalid.
    pub fn validate(&self) -> AudioSampleResult<()> {
        self.stft.validate()?;
        self.hpss.validate()
    }
}

/// Which components of the separation to resynthesise.
#[derive(Clone, Copy)]
enum HpssPart {
    Harmonic,
    Percussive,
}

/// `librosa.effects.harmonic` / `percussive` on rows.
fn hpss_rows(
    rows: ArrayView2<'_, f64>,
    config: &HarmonicPercussiveConfig,
    parts: &[HpssPart],
) -> AudioSampleResult<Vec<Array2<f64>>> {
    config.validate()?;
    let n = rows.ncols();
    let d = stft(rows, &config.stft)?;
    let (h, p) = hpss_spectrogram(d.view(), &config.hpss)?;
    parts
        .iter()
        .map(|part| {
            let s = match part {
                HpssPart::Harmonic => &h,
                HpssPart::Percussive => &p,
            };
            istft(s.view(), &config.stft, Some(n))
        })
        .collect()
}

// =============================================================================
// Griffin-Lim
// =============================================================================

/// Initial phase estimate of [`griffinlim`] (librosa's `init`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GriffinLimInit {
    /// Uniformly random phases (librosa `init="random"`, the default). `Some(seed)`
    /// makes the result reproducible; `None` draws a fresh seed.
    Random(Option<u64>),
    /// All phases zero (librosa `init=None`): fully deterministic.
    ZeroPhase,
}

impl Default for GriffinLimInit {
    fn default() -> Self {
        Self::Random(None)
    }
}

/// Configuration of Griffin-Lim phase reconstruction, mirroring `librosa.griffinlim`.
///
/// ## Purpose
///
/// Sets the iteration count, the momentum of the fast Griffin-Lim update, the
/// initial phase and the STFT that the magnitude came from.
///
/// ## Intended usage
///
/// `Default` reproduces librosa's defaults (32 iterations, momentum 0.99, random
/// initial phase, default STFT). `stft.n_fft` must match the spectrogram
/// (`n_bins = n_fft / 2 + 1`); librosa infers it from the bin count.
///
/// ## Invariants
///
/// Checked by [`GriffinLimConfig::validate`]: momentum finite and non-negative,
/// `length` (if set) positive, the STFT configuration valid.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::effects::{GriffinLimConfig, GriffinLimInit};
///
/// let cfg = GriffinLimConfig::new().with_n_iter(8).with_init(GriffinLimInit::ZeroPhase);
/// assert_eq!(cfg.momentum, 0.99);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct GriffinLimConfig {
    /// Number of iterations. librosa default: 32.
    pub n_iter: usize,
    /// Momentum of the fast Griffin-Lim update (0 gives the original algorithm).
    /// Values above 1 can be unstable. librosa default: 0.99.
    pub momentum: f64,
    /// Initial phase. librosa default: random.
    pub init: GriffinLimInit,
    /// STFT parameters of the magnitude spectrogram. librosa default.
    pub stft: StftConfig,
    /// Exact output length in samples (`None`: as the inverse STFT gives).
    pub length: Option<usize>,
}

impl Default for GriffinLimConfig {
    fn default() -> Self {
        Self {
            n_iter: 32,
            momentum: 0.99,
            init: GriffinLimInit::default(),
            stft: StftConfig::default(),
            length: None,
        }
    }
}

impl GriffinLimConfig {
    /// Returns the librosa-default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of iterations.
    #[must_use]
    pub fn with_n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    /// Sets the momentum.
    #[must_use]
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Sets the initial phase.
    #[must_use]
    pub fn with_init(mut self, init: GriffinLimInit) -> Self {
        self.init = init;
        self
    }

    /// Sets the STFT configuration.
    #[must_use]
    pub fn with_stft(mut self, stft: StftConfig) -> Self {
        self.stft = stft;
        self
    }

    /// Sets the output length.
    #[must_use]
    pub fn with_length(mut self, length: Option<usize>) -> Self {
        self.length = length;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the momentum is negative or not finite, `length` is
    /// `Some(0)`, or the STFT configuration is invalid.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if !(self.momentum.is_finite() && self.momentum >= 0.0) {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "momentum",
                self.momentum,
                0,
                "finite",
                "momentum must be non-negative",
            )));
        }
        if self.length == Some(0) {
            return Err(param_err("length", "the output length must be positive"));
        }
        self.stft.validate()
    }
}

/// Reconstructs a signal from a magnitude spectrogram with the fast Griffin-Lim
/// algorithm, equivalent to `librosa.griffinlim`.
///
/// Starting from `S * exp(i phi0)`, each iteration inverts the current estimate
/// ([`istft`]), re-analyses it ([`stft`]), applies the momentum step
/// `A = rebuilt - momentum / (1 + momentum) * previous`, and keeps only the phase
/// of `A` with the given magnitude: `S * A / (|A| + tiny)`.
///
/// # Arguments
///
/// * `magnitude` - `(channels, n_fft / 2 + 1, n_frames)` non-negative magnitude
///   spectrogram.
/// * `config` - Iterations, momentum, initial phase, STFT and output length.
///
/// # Returns
///
/// A `(channels, samples)` array.
///
/// # Errors
///
/// [`ParameterError`] if the configuration is invalid, the bin count does not match
/// `config.stft.n_fft`, or the spectrogram has no frames.
///
/// # Differences from librosa
///
/// With [`GriffinLimInit::Random`] the random phases come from this crate's RNG,
/// not NumPy's, so results differ from librosa's for the same seed (both are
/// valid reconstructions). [`GriffinLimInit::ZeroPhase`] is deterministic and
/// matches librosa's `init=None` to round-off. librosa warns for momentum > 1;
/// this function accepts it silently.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::effects::{GriffinLimConfig, GriffinLimInit, griffinlim};
/// use audio_samples::operations::spectrum::{StftConfig, stft};
/// use ndarray::Array2;
///
/// let x = Array2::from_shape_fn((1, 4000), |(_, n)| (n as f64 * 0.07).sin());
/// let cfg = StftConfig::new().with_n_fft(nzu!(512));
/// let mag = stft(x.view(), &cfg).unwrap().mapv(|z| z.norm());
/// let gl = GriffinLimConfig::new()
///     .with_stft(cfg)
///     .with_init(GriffinLimInit::Random(Some(0)))
///     .with_length(Some(4000));
/// let y = griffinlim(mag.view(), &gl).unwrap();
/// assert_eq!(y.dim(), (1, 4000));
/// ```
///
/// # References
///
/// D. W. Griffin and J. S. Lim, "Signal estimation from modified short-time Fourier
/// transform," *IEEE Trans. ASSP* 32(2), 1984; N. Perraudin, P. Balazs and
/// P. L. Søndergaard, "A fast Griffin-Lim algorithm," WASPAA 2013.
pub fn griffinlim(
    magnitude: ArrayView3<'_, f64>,
    config: &GriffinLimConfig,
) -> AudioSampleResult<Array2<f64>> {
    config.validate()?;
    if magnitude.iter().any(|v| !(*v >= 0.0 && v.is_finite())) {
        return Err(param_err(
            "magnitude",
            "the magnitude spectrogram must be finite and non-negative",
        ));
    }
    let mut angles: Array3<Complex<f64>> = match config.init {
        GriffinLimInit::ZeroPhase => magnitude.mapv(|s| Complex::new(s, 0.0)),
        GriffinLimInit::Random(seed) => {
            use rand::rngs::StdRng;
            use rand::{RngExt, SeedableRng};
            let mut rng = StdRng::seed_from_u64(seed.unwrap_or_else(rand::random::<u64>));
            magnitude.mapv(|s| {
                let phi = 2.0 * std::f64::consts::PI * rng.random::<f64>();
                Complex::new(phi.cos() * s, phi.sin() * s)
            })
        }
    };
    let beta = config.momentum / (1.0 + config.momentum);
    let mut tprev: Option<Array3<Complex<f64>>> = None;
    for _ in 0..config.n_iter {
        let inverse = istft(angles.view(), &config.stft, config.length)?;
        let rebuilt = stft(inverse.view(), &config.stft)?;
        if rebuilt.dim() != magnitude.dim() {
            return Err(config_err(
                "griffinlim",
                format!(
                    "re-analysis gives shape {:?} but the magnitude is {:?}; check `length`",
                    rebuilt.dim(),
                    magnitude.dim()
                ),
            ));
        }
        angles.assign(&rebuilt);
        if let Some(prev) = &tprev {
            ndarray::Zip::from(&mut angles)
                .and(prev)
                .for_each(|a, &p| *a -= p * beta);
        }
        ndarray::Zip::from(&mut angles)
            .and(&magnitude)
            .for_each(|a, &s| {
                let d = a.norm() + f64::MIN_POSITIVE;
                *a = Complex::new(a.re / d * s, a.im / d * s);
            });
        tprev = Some(rebuilt);
    }
    istft(angles.view(), &config.stft, config.length)
}

// =============================================================================
// Nearest-neighbour filtering
// =============================================================================

/// How [`nn_filter`] combines a frame's neighbours (librosa's `aggregate`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NnAggregate {
    /// Mean of the neighbours (`numpy.mean`, librosa's default).
    #[default]
    Mean,
    /// Element-wise median of the neighbours (`numpy.median`).
    Median,
    /// Average weighted by the recurrence values (`numpy.average(weights=...)`);
    /// use with an affinity recurrence matrix.
    WeightedAverage,
}

/// Replaces every frame of `data` by an aggregate of its nearest neighbours,
/// equivalent to `librosa.decompose.nn_filter(S, rec=rec, aggregate=...)`.
///
/// For output frame `i` the neighbours are the frames `j` with `rec[j, i] != 0`
/// (librosa's orientation, as produced by
/// [`recurrence_matrix`]); a frame with no neighbours is copied unchanged.
///
/// # Arguments
///
/// * `data` - `(n_features, n_frames)` features (e.g. a chromagram or magnitude
///   spectrogram).
/// * `rec` - `(n_frames, n_frames)` recurrence matrix (connectivity or affinity).
/// * `aggregate` - Mean, median or recurrence-weighted average.
///
/// # Returns
///
/// The filtered `(n_features, n_frames)` array.
///
/// # Errors
///
/// [`ParameterError`] if `rec` is not `(n_frames, n_frames)`, or the weights of a
/// frame sum to zero with [`NnAggregate::WeightedAverage`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::effects::{NnAggregate, nn_filter};
/// use ndarray::{Array2, array};
///
/// let data = array![[1.0, 2.0, 3.0]];
/// // Frame 0's neighbours are frames 1 and 2; the others have none.
/// let mut rec = Array2::zeros((3, 3));
/// rec[[1, 0]] = 1.0;
/// rec[[2, 0]] = 1.0;
/// let y = nn_filter(data.view(), rec.view(), NnAggregate::Mean).unwrap();
/// assert_eq!(y, array![[2.5, 2.0, 3.0]]);
/// ```
///
/// # References
///
/// Z. Rafii and B. Pardo, "Music/voice separation using the similarity matrix,"
/// ISMIR 2012 (REPET-SIM).
pub fn nn_filter(
    data: ArrayView2<'_, f64>,
    rec: ArrayView2<'_, f64>,
    aggregate: NnAggregate,
) -> AudioSampleResult<Array2<f64>> {
    let (n_features, n_frames) = data.dim();
    if rec.dim() != (n_frames, n_frames) {
        return Err(config_err(
            "nn_filter",
            format!(
                "invalid self-similarity matrix shape {:?} for data shape {:?}",
                rec.dim(),
                data.dim()
            ),
        ));
    }
    let mut out = data.to_owned();
    let mut values = Vec::new();
    for i in 0..n_frames {
        let targets: Vec<usize> = (0..n_frames).filter(|&j| rec[[j, i]] != 0.0).collect();
        if targets.is_empty() {
            continue;
        }
        let weight_sum: f64 = targets.iter().map(|&j| rec[[j, i]]).sum();
        if aggregate == NnAggregate::WeightedAverage && weight_sum == 0.0 {
            return Err(param_err(
                "rec",
                format!("the recurrence weights of frame {i} sum to zero"),
            ));
        }
        for f in 0..n_features {
            out[[f, i]] = match aggregate {
                NnAggregate::Mean => {
                    targets.iter().map(|&j| data[[f, j]]).sum::<f64>() / targets.len() as f64
                }
                NnAggregate::WeightedAverage => {
                    targets
                        .iter()
                        .map(|&j| data[[f, j]] * rec[[j, i]])
                        .sum::<f64>()
                        / weight_sum
                }
                NnAggregate::Median => {
                    values.clear();
                    values.extend(targets.iter().map(|&j| data[[f, j]]));
                    values.sort_by(f64::total_cmp);
                    let m = values.len();
                    if m % 2 == 1 {
                        values[m / 2]
                    } else {
                        (values[m / 2 - 1] + values[m / 2]) / 2.0
                    }
                }
            };
        }
    }
    Ok(out)
}

/// [`nn_filter`] with the recurrence matrix computed from `data` itself,
/// equivalent to `librosa.decompose.nn_filter(S, aggregate=..., **kwargs)` with
/// `rec=None` (`kwargs` are the [`RecurrenceConfig`] fields).
///
/// # Arguments
///
/// * `data` - `(n_features, n_frames)` features.
/// * `recurrence` - Neighbourhood parameters; [`RecurrenceConfig::default`] is
///   librosa's default (connectivity, `k = 2 ceil(sqrt(t - 2 width + 1))`, width 1).
/// * `aggregate` - Mean, median or weighted average (use an affinity mode for
///   meaningful weights).
///
/// # Returns
///
/// The filtered `(n_features, n_frames)` array.
///
/// # Errors
///
/// As [`recurrence_matrix`] and [`nn_filter`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::effects::{NnAggregate, nn_filter_recurrence};
/// use audio_samples::operations::sequence::RecurrenceConfig;
/// use ndarray::Array2;
///
/// let data = Array2::from_shape_fn((3, 40), |(f, t)| ((t % 8) as f64 + f as f64).sin());
/// let y = nn_filter_recurrence(data.view(), &RecurrenceConfig::default(), NnAggregate::Median)
///     .unwrap();
/// assert_eq!(y.dim(), (3, 40));
/// ```
pub fn nn_filter_recurrence(
    data: ArrayView2<'_, f64>,
    recurrence: &RecurrenceConfig,
    aggregate: NnAggregate,
) -> AudioSampleResult<Array2<f64>> {
    let rec = recurrence_matrix(data, recurrence)?;
    nn_filter(data, rec.view(), aggregate)
}

// =============================================================================
// LPC (Burg)
// =============================================================================

/// Linear-prediction coefficients by Burg's method, equivalent to
/// `librosa.lpc(y, order=order)`.
///
/// # Arguments
///
/// * `y` - Signal, finite, with more than `order` samples.
/// * `order` - Prediction order.
///
/// # Returns
///
/// `order + 1` coefficients `[1, a_1, ..., a_order]` of the prediction-error
/// filter `A(z) = 1 + a_1 z^-1 + ... + a_order z^-order`.
///
/// # Errors
///
/// [`ParameterError`] if `y` has a non-finite value or no more than `order` samples.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::effects::lpc;
///
/// // A sampled sinusoid obeys y[n] = 2 cos(w) y[n-1] - y[n-2].
/// let w = 0.3f64;
/// let y: Vec<f64> = (0..2000).map(|n| (w * n as f64).sin()).collect();
/// let a = lpc(&y, nzu!(2)).unwrap();
/// assert_eq!(a[0], 1.0);
/// assert!((a[1] + 2.0 * w.cos()).abs() < 1e-3);
/// assert!((a[2] - 1.0).abs() < 1e-3);
/// ```
///
/// # References
///
/// J. P. Burg, "Maximum entropy spectral analysis," PhD thesis, Stanford, 1975;
/// librosa `core.lpc`.
pub fn lpc(y: &[f64], order: NonZeroUsize) -> AudioSampleResult<Vec<f64>> {
    let order = order.get();
    if y.iter().any(|v| !v.is_finite()) {
        return Err(param_err("y", "audio buffer is not finite everywhere"));
    }
    if y.len() <= order {
        return Err(param_err(
            "order",
            format!(
                "order={order} needs more than {order} samples (got {})",
                y.len()
            ),
        ));
    }
    let mut ar = vec![0.0; order + 1];
    let mut ar_prev = vec![0.0; order + 1];
    ar[0] = 1.0;
    ar_prev[0] = 1.0;
    let mut fwd: Vec<f64> = y[1..].to_vec();
    let mut bwd: Vec<f64> = y[..y.len() - 1].to_vec();
    let mut den: f64 = fwd.iter().zip(&bwd).map(|(f, b)| f * f + b * b).sum();
    for i in 0..order {
        let dot: f64 = bwd.iter().zip(&fwd).map(|(b, f)| b * f).sum();
        let k = dot * -2.0 / (den + f64::MIN_POSITIVE);
        std::mem::swap(&mut ar, &mut ar_prev);
        for j in 1..=i + 1 {
            ar[j] = ar_prev[j] + k * ar_prev[i + 1 - j];
        }
        let fwd_new: Vec<f64> = fwd.iter().zip(&bwd).map(|(f, b)| f + k * b).collect();
        let bwd_new: Vec<f64> = bwd.iter().zip(&fwd).map(|(b, f)| b + k * f).collect();
        let q = 1.0 - k * k;
        let (b_last, f_first) = (bwd_new[bwd_new.len() - 1], fwd_new[0]);
        den = q * den - b_last * b_last - f_first * f_first;
        fwd = fwd_new[1..].to_vec();
        bwd = bwd_new[..bwd_new.len() - 1].to_vec();
    }
    Ok(ar)
}

// =============================================================================
// Click tracks
// =============================================================================

/// Where [`clicks`] places its clicks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClickPositions<'a> {
    /// Times in seconds (librosa `times=`); sample index `floor(t * sr)`.
    Times(&'a [f64]),
    /// Frame indices (librosa `frames=`); sample index `frame * hop_length`.
    Frames(&'a [usize]),
}

/// Configuration of [`clicks`], mirroring `librosa.clicks`'s keyword arguments.
///
/// ## Purpose
///
/// Sets the click waveform (a decaying sine burst unless `click` is given), the
/// frame hop for frame positions, and the output length.
///
/// ## Intended usage
///
/// `Default` reproduces librosa's defaults: a 1 kHz, 100 ms click, hop 512, length
/// up to the end of the last click.
///
/// ## Invariants
///
/// Checked by [`ClicksConfig::validate`]: `click_freq` and `click_duration` finite
/// and positive (when no custom click is given), a custom click non-empty and
/// finite, `length` positive.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::effects::ClicksConfig;
///
/// let cfg = ClicksConfig::new().with_click_freq(2000.0).with_click_duration(0.05);
/// assert!(cfg.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ClicksConfig {
    /// Samples per frame for [`ClickPositions::Frames`]. librosa default: 512.
    pub hop_length: NonZeroUsize,
    /// Frequency (Hz) of the default click. librosa default: 1000.
    pub click_freq: f64,
    /// Duration (s) of the default click. librosa default: 0.1.
    pub click_duration: f64,
    /// Custom click waveform used instead of the default one.
    pub click: Option<Vec<f64>>,
    /// Output length in samples; clicks starting at or after it are dropped.
    /// `None`: up to the end of the last click.
    pub length: Option<usize>,
}

impl Default for ClicksConfig {
    fn default() -> Self {
        Self {
            hop_length: crate::nzu!(512),
            click_freq: 1000.0,
            click_duration: 0.1,
            click: None,
            length: None,
        }
    }
}

impl ClicksConfig {
    /// Returns the librosa-default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the hop length used for frame positions.
    #[must_use]
    pub fn with_hop_length(mut self, hop_length: NonZeroUsize) -> Self {
        self.hop_length = hop_length;
        self
    }

    /// Sets the frequency of the default click.
    #[must_use]
    pub fn with_click_freq(mut self, click_freq: f64) -> Self {
        self.click_freq = click_freq;
        self
    }

    /// Sets the duration of the default click.
    #[must_use]
    pub fn with_click_duration(mut self, click_duration: f64) -> Self {
        self.click_duration = click_duration;
        self
    }

    /// Sets a custom click waveform.
    #[must_use]
    pub fn with_click(mut self, click: Option<Vec<f64>>) -> Self {
        self.click = click;
        self
    }

    /// Sets the output length.
    #[must_use]
    pub fn with_length(mut self, length: Option<usize>) -> Self {
        self.length = length;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the default click's frequency or duration is not finite
    /// and positive, a custom click is empty or not finite, or `length` is `Some(0)`.
    pub fn validate(&self) -> AudioSampleResult<()> {
        match &self.click {
            Some(c) => {
                if c.is_empty() || c.iter().any(|v| !v.is_finite()) {
                    return Err(param_err("click", "the click must be non-empty and finite"));
                }
            }
            None => {
                if !(self.click_duration.is_finite() && self.click_duration > 0.0) {
                    return Err(param_err(
                        "click_duration",
                        "click_duration must be strictly positive",
                    ));
                }
                if !(self.click_freq.is_finite() && self.click_freq > 0.0) {
                    return Err(param_err(
                        "click_freq",
                        "click_freq must be strictly positive",
                    ));
                }
            }
        }
        if self.length == Some(0) {
            return Err(param_err("length", "length must be a positive integer"));
        }
        Ok(())
    }
}

/// Synthesises a click track, equivalent to `librosa.clicks`.
///
/// The default click is `2^(-10 u) sin(2 pi f n / sr)` for `n = 0 ..
/// floor(sr * duration)` with `u` going linearly from 0 to 1 (a sine burst decaying
/// by 60 dB). Clicks are added (overlapping clicks sum) and truncated at the end.
///
/// # Arguments
///
/// * `positions` - Click times (seconds) or frame indices.
/// * `sample_rate` - Output sample rate.
/// * `config` - Click waveform, hop and length.
///
/// # Returns
///
/// Mono `f64` audio.
///
/// # Errors
///
/// [`ParameterError`] if the configuration is invalid, a time is negative or not
/// finite, or `positions` is empty while `length` is `None`; an empty-data error if
/// the default click would have no samples.
///
/// # Differences from librosa
///
/// librosa returns `float32`; this function returns `f64` audio (values agree to
/// `float32` rounding). Multichannel custom clicks are not supported.
///
/// # Examples
///
/// ```rust
/// use audio_samples::operations::effects::{ClickPositions, ClicksConfig, clicks};
/// use audio_samples::sample_rate;
///
/// let y = clicks(ClickPositions::Times(&[0.0, 0.5]), sample_rate!(22050), &ClicksConfig::default())
///     .unwrap();
/// // Last click starts at 11025 and lasts 2205 samples.
/// assert_eq!(y.samples_per_channel().get(), 11025 + 2205);
/// ```
pub fn clicks(
    positions: ClickPositions<'_>,
    sample_rate: SampleRate,
    config: &ClicksConfig,
) -> AudioSampleResult<AudioSamples<'static, f64>> {
    config.validate()?;
    let sr = f64::from(sample_rate.get());
    let mut pos: Vec<usize> = match positions {
        ClickPositions::Times(times) => {
            let mut v = Vec::with_capacity(times.len());
            for &t in times {
                if !(t.is_finite() && t >= 0.0) {
                    return Err(param_err(
                        "times",
                        "click times must be finite and non-negative",
                    ));
                }
                v.push((t * sr) as usize);
            }
            v
        }
        ClickPositions::Frames(frames) => frames
            .iter()
            .map(|&f| f * config.hop_length.get())
            .collect(),
    };
    let click: Vec<f64> = match &config.click {
        Some(c) => c.clone(),
        None => {
            let n = (sr * config.click_duration) as usize;
            if n == 0 {
                return Err(AudioSampleError::empty_data("clicks"));
            }
            let w = 2.0 * std::f64::consts::PI * config.click_freq / sr;
            // numpy.logspace(0, -10, n, base=2): 2 ** linspace(0, -10, n).
            let step = if n > 1 { -10.0 / (n - 1) as f64 } else { 0.0 };
            (0..n)
                .map(|i| {
                    let e = if n > 1 && i == n - 1 {
                        -10.0
                    } else {
                        i as f64 * step
                    };
                    2.0f64.powf(e) * (w * i as f64).sin()
                })
                .collect()
        }
    };
    let length = match config.length {
        Some(l) => {
            pos.retain(|&p| p < l);
            l
        }
        None => {
            let max = pos.iter().copied().max().ok_or_else(|| {
                param_err(
                    "positions",
                    "at least one click position is required when length is None",
                )
            })?;
            max + click.len()
        }
    };
    let mut out = vec![0.0; length];
    for &start in &pos {
        let end = (start + click.len()).min(length);
        for (o, c) in out[start..end].iter_mut().zip(&click) {
            *o += c;
        }
    }
    let data = NonEmptyVec::new(out).map_err(|_| AudioSampleError::empty_data("clicks"))?;
    Ok(AudioSamples::from_mono_vec::<f64>(data, sample_rate))
}

// =============================================================================
// The AudioEffects trait
// =============================================================================

/// librosa-compatible audio effects on [`AudioSamples`].
///
/// All methods process every channel (mono gives one channel) in `f64` and return
/// audio of the input's sample type and sample rate. Each audio-to-audio method has
/// an `_in_place` variant that replaces the audio (its length may change). See the
/// [module documentation](self) for the librosa correspondence.
pub trait AudioEffects {
    /// The sample type of the audio.
    type Sample: StandardSample;

    /// Changes the speed without changing the pitch, equivalent to
    /// `librosa.effects.time_stretch(y, rate=rate)`.
    ///
    /// The audio is analysed with `stft`, re-timed with [`phase_vocoder`] and
    /// resynthesised to exactly `round(n / rate)` samples (ties to even, as Python's
    /// `round`).
    ///
    /// # Arguments
    ///
    /// * `rate` - Speed-up factor: `2.0` is twice as fast (half the length), `0.5`
    ///   half as fast. Finite and positive.
    /// * `stft` - STFT parameters ([`StftConfig::default`] = librosa's).
    ///
    /// # Returns
    ///
    /// The stretched audio, `round(n / rate)` samples per channel.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `rate` is not finite and positive, the result would be
    /// empty, or the STFT configuration is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::effects::AudioEffects;
    /// use audio_samples::operations::spectrum::StftConfig;
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let x = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(22050), 0.5);
    /// let slow = x.time_stretch(0.8, &StftConfig::default()).unwrap();
    /// assert_eq!(slow.samples_per_channel().get(), 27562); // round(22050 / 0.8) (ties to even)
    /// ```
    fn time_stretch(
        &self,
        rate: f64,
        stft: &StftConfig,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// In-place [`time_stretch`](Self::time_stretch).
    ///
    /// # Errors
    ///
    /// As [`time_stretch`](Self::time_stretch); the audio is unchanged on error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::effects::AudioEffects;
    /// use audio_samples::operations::spectrum::StftConfig;
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let mut x = sine_wave::<f64>(440.0, Duration::from_secs(1), sample_rate!(22050), 0.5);
    /// x.time_stretch_in_place(2.0, &StftConfig::default()).unwrap();
    /// assert_eq!(x.samples_per_channel().get(), 11025);
    /// ```
    fn time_stretch_in_place(&mut self, rate: f64, stft: &StftConfig) -> AudioSampleResult<()>;

    /// Shifts the pitch by `n_steps` without changing the duration, equivalent to
    /// `librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)`.
    ///
    /// With `rate = 2^(-n_steps / bins_per_octave)` the audio is
    /// [time-stretched](Self::time_stretch) by `rate`, resampled by the factor `rate`
    /// (librosa: from `sr / rate` to `sr`) and cut or zero-padded to the original
    /// length.
    ///
    /// # Arguments
    ///
    /// * `n_steps` - Shift in steps (semitones by default); fractional and negative
    ///   values allowed.
    /// * `config` - Steps per octave, resampler quality and STFT.
    ///
    /// # Returns
    ///
    /// The shifted audio, same length and sample rate.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `n_steps` is not finite, the configuration is invalid,
    /// or the shift is so large that the stretched signal is empty.
    ///
    /// # Differences from librosa
    ///
    /// librosa resamples with soxr (`res_type="soxr_hq"`); this method uses the
    /// crate's windowed-sinc resampler ([`ResamplingQuality::High`] by default),
    /// which approximates the arbitrary ratio `rate` by a ratio of two integers near
    /// 2^31. The time-stretch stage matches librosa to round-off; the final signal
    /// agrees within the resampler difference (about 68 dB SNR on the reference test signal,
    /// see `tests/librosa_reference/effects.rs`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::effects::{AudioEffects, PitchShiftConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let x = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(22050), 0.5);
    /// let y = x.pitch_shift(-3.0, &PitchShiftConfig::default()).unwrap();
    /// assert_eq!(y.samples_per_channel(), x.samples_per_channel());
    /// ```
    fn pitch_shift(
        &self,
        n_steps: f64,
        config: &PitchShiftConfig,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// In-place [`pitch_shift`](Self::pitch_shift).
    ///
    /// # Errors
    ///
    /// As [`pitch_shift`](Self::pitch_shift); the audio is unchanged on error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::effects::{AudioEffects, PitchShiftConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let mut x = sine_wave::<f64>(440.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
    /// x.pitch_shift_in_place(12.0, &PitchShiftConfig::default()).unwrap();
    /// assert_eq!(x.samples_per_channel().get(), 11025);
    /// ```
    fn pitch_shift_in_place(
        &mut self,
        n_steps: f64,
        config: &PitchShiftConfig,
    ) -> AudioSampleResult<()>;

    /// Applies a pre-emphasis filter `y[n] = x[n] - coef * x[n - 1]` to every
    /// channel, equivalent to `librosa.effects.preemphasis(y, coef=coef)` (initial
    /// state `2 x[0] - x[1]`). Use [`preemphasis_with_state`] for explicit
    /// `zi` / `zf`.
    ///
    /// # Arguments
    ///
    /// * `coef` - Coefficient (librosa default 0.97), finite.
    ///
    /// # Returns
    ///
    /// The filtered audio.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `coef` is not finite or a channel has fewer than two
    /// samples.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::effects::AudioEffects;
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::array;
    ///
    /// let x = AudioSamples::new_mono(array![1.0f64, 2.0, 3.0, 4.0], sample_rate!(8000)).unwrap();
    /// let y = x.preemphasis(0.97).unwrap();
    /// let y = y.as_mono().unwrap();
    /// assert!((y[1] - (2.0 - 0.97)).abs() < 1e-15);
    /// ```
    fn preemphasis(&self, coef: f64) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// In-place [`preemphasis`](Self::preemphasis).
    ///
    /// # Errors
    ///
    /// As [`preemphasis`](Self::preemphasis).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::effects::AudioEffects;
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::array;
    ///
    /// let mut x = AudioSamples::new_mono(array![1.0f64, 2.0, 3.0], sample_rate!(8000)).unwrap();
    /// x.preemphasis_in_place(0.5).unwrap();
    /// assert_eq!(x.as_mono().unwrap()[2], 3.0 - 0.5 * 2.0);
    /// ```
    fn preemphasis_in_place(&mut self, coef: f64) -> AudioSampleResult<()>;

    /// Applies the de-emphasis filter `y[n] = x[n] + coef * y[n - 1]` (the inverse
    /// of [`preemphasis`](Self::preemphasis)) to every channel, equivalent to
    /// `librosa.effects.deemphasis(y, coef=coef)`. Use [`deemphasis_with_state`] for
    /// explicit `zi` / `zf`.
    ///
    /// # Arguments
    ///
    /// * `coef` - Coefficient (librosa default 0.97), finite and not 3.
    ///
    /// # Returns
    ///
    /// The filtered audio.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `coef` is not finite or 3, or a channel has fewer than
    /// two samples.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::effects::AudioEffects;
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let x = sine_wave::<f64>(440.0, Duration::from_millis(10), sample_rate!(22050), 0.5);
    /// let y = x.preemphasis(0.97).unwrap().deemphasis(0.97).unwrap();
    /// let (a, b) = (x.as_mono().unwrap(), y.as_mono().unwrap());
    /// assert!(a.iter().zip(b.iter()).all(|(p, q)| (p - q).abs() < 1e-12));
    /// ```
    fn deemphasis(&self, coef: f64) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// In-place [`deemphasis`](Self::deemphasis).
    ///
    /// # Errors
    ///
    /// As [`deemphasis`](Self::deemphasis).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::effects::AudioEffects;
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::array;
    ///
    /// let mut x = AudioSamples::new_mono(array![1.0f64, 2.0, 3.0], sample_rate!(8000)).unwrap();
    /// x.deemphasis_in_place(0.0).unwrap();
    /// // coef = 0: y = x - (2 x0 - x1) / 3 * [1, 0, 0].
    /// assert_eq!(x.as_mono().unwrap()[1], 2.0);
    /// ```
    fn deemphasis_in_place(&mut self, coef: f64) -> AudioSampleResult<()>;

    /// Trims leading and trailing silence, equivalent to `librosa.effects.trim(y)`.
    ///
    /// Frames are classified with [`nonsilent_frames`]; the kept region runs from the
    /// first non-silent frame's start `first * hop` to `min(n, (last + 1) * hop)`.
    ///
    /// # Arguments
    ///
    /// * `config` - Threshold, reference, framing and channel aggregation
    ///   ([`SilenceConfig::default`] = librosa's).
    ///
    /// # Returns
    ///
    /// `(trimmed, (start, end))`: the trimmed audio (all channels) and the kept
    /// sample range `start..end` of the input.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the configuration is invalid; an empty-data error if
    /// every frame is silent (librosa returns an empty array and `[0, 0]`; use
    /// [`split_db`](Self::split_db), which returns no intervals, to test for that).
    ///
    /// # Differences from librosa
    ///
    /// Not to be confused with
    /// [`AudioEditing::trim_silence`](crate::AudioEditing::trim_silence), which
    /// thresholds individual samples in absolute dBFS. See also [`nonsilent_frames`]
    /// (RMS in `f64` instead of librosa's `float32`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::effects::{AudioEffects, SilenceConfig};
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let x = Array1::from_shape_fn(22050, |n| if (8000..14000).contains(&n) { (n as f64 * 0.1).sin() } else { 0.0 });
    /// let audio = AudioSamples::new_mono(x, sample_rate!(22050)).unwrap();
    /// let (trimmed, (start, end)) = audio.trim_db(&SilenceConfig::default()).unwrap();
    /// assert_eq!((start, end), (7168, 15360)); // frame-aligned (hop 512, centred frames)
    /// assert_eq!(trimmed.samples_per_channel().get(), end - start);
    /// ```
    #[allow(clippy::type_complexity)]
    fn trim_db(
        &self,
        config: &SilenceConfig,
    ) -> AudioSampleResult<(AudioSamples<'static, Self::Sample>, (usize, usize))>;

    /// In-place [`trim_db`](Self::trim_db).
    ///
    /// # Returns
    ///
    /// The kept sample range `(start, end)` of the original audio.
    ///
    /// # Errors
    ///
    /// As [`trim_db`](Self::trim_db); the audio is unchanged on error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::effects::{AudioEffects, SilenceConfig};
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let x = Array1::from_shape_fn(22050, |n| if n >= 11025 { 0.5 } else { 0.0 });
    /// let mut audio = AudioSamples::new_mono(x, sample_rate!(22050)).unwrap();
    /// let (start, end) = audio.trim_db_in_place(&SilenceConfig::default()).unwrap();
    /// assert_eq!(end, 22050);
    /// assert_eq!(audio.samples_per_channel().get(), end - start);
    /// ```
    fn trim_db_in_place(&mut self, config: &SilenceConfig) -> AudioSampleResult<(usize, usize)>;

    /// Splits the audio into non-silent intervals, equivalent to
    /// `librosa.effects.split(y)`.
    ///
    /// # Arguments
    ///
    /// * `config` - Threshold, reference, framing and channel aggregation.
    ///
    /// # Returns
    ///
    /// Sample ranges `(start, end)` of consecutive non-silent frames, in order,
    /// frame-aligned and clipped to the signal length. Empty if everything is silent.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the configuration is invalid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::effects::{AudioEffects, SilenceConfig};
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let x = Array1::from_shape_fn(44100, |n| {
    ///     if (5000..10000).contains(&n) || (30000..35000).contains(&n) { 0.5 } else { 0.0 }
    /// });
    /// let audio = AudioSamples::new_mono(x, sample_rate!(22050)).unwrap();
    /// let parts = audio.split_db(&SilenceConfig::default()).unwrap();
    /// assert_eq!(parts.len(), 2);
    /// ```
    fn split_db(&self, config: &SilenceConfig) -> AudioSampleResult<Vec<(usize, usize)>>;

    /// Concatenates the given sample intervals, equivalent to
    /// `librosa.effects.remix(y, intervals, align_zeros=align_zeros)`.
    ///
    /// With `align_zeros`, every interval boundary is moved to the nearest zero
    /// crossing of the (mono downmix of the) signal, where sample 0 and the end of
    /// the signal also count as crossings; this avoids clicks at the joins.
    ///
    /// # Arguments
    ///
    /// * `intervals` - `(start, end)` sample ranges, in output order (may repeat or
    ///   be reordered). Ranges are clipped to the signal; empty ranges are skipped.
    /// * `align_zeros` - Snap boundaries to zero crossings (librosa default `true`).
    ///
    /// # Returns
    ///
    /// The remixed audio (all channels).
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `intervals` is empty; an empty-data error if every
    /// interval is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::effects::AudioEffects;
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::array;
    ///
    /// let x = AudioSamples::new_mono(array![0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0], sample_rate!(8000)).unwrap();
    /// // Swap the two halves.
    /// let y = x.remix(&[(3, 6), (0, 3)], false).unwrap();
    /// assert_eq!(y.as_mono().unwrap().to_vec(), vec![3.0, 4.0, 5.0, 0.0, 1.0, 2.0]);
    /// ```
    fn remix(
        &self,
        intervals: &[(usize, usize)],
        align_zeros: bool,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// In-place [`remix`](Self::remix).
    ///
    /// # Errors
    ///
    /// As [`remix`](Self::remix); the audio is unchanged on error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::effects::AudioEffects;
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::array;
    ///
    /// let mut x = AudioSamples::new_mono(array![0.0f64, 1.0, 2.0, 3.0], sample_rate!(8000)).unwrap();
    /// x.remix_in_place(&[(2, 4), (2, 4)], false).unwrap();
    /// assert_eq!(x.as_mono().unwrap().to_vec(), vec![2.0, 3.0, 2.0, 3.0]);
    /// ```
    fn remix_in_place(
        &mut self,
        intervals: &[(usize, usize)],
        align_zeros: bool,
    ) -> AudioSampleResult<()>;

    /// Extracts the harmonic component, equivalent to `librosa.effects.harmonic(y)`.
    ///
    /// STFT, [`hpss_spectrogram`], inverse STFT trimmed to the input length.
    ///
    /// # Arguments
    ///
    /// * `config` - STFT and mask parameters ([`HarmonicPercussiveConfig::default`]
    ///   = librosa's).
    ///
    /// # Returns
    ///
    /// The harmonic component, same length as the input.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the configuration is invalid.
    ///
    /// # Differences from librosa
    ///
    /// librosa's `effects.hpss`/`harmonic`/`percussive` accept a `window` argument
    /// but do not pass it on (they always use Hann); here `config.stft.window` is
    /// used. This is a different method from
    /// [`AudioDecomposition::hpss`](crate::AudioDecomposition::hpss), which keeps its
    /// own mask and window conventions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::effects::{AudioEffects, HarmonicPercussiveConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let x = sine_wave::<f32>(440.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
    /// let h = x.harmonic(&HarmonicPercussiveConfig::default()).unwrap();
    /// assert_eq!(h.samples_per_channel(), x.samples_per_channel());
    /// ```
    fn harmonic(
        &self,
        config: &HarmonicPercussiveConfig,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// In-place [`harmonic`](Self::harmonic).
    ///
    /// # Errors
    ///
    /// As [`harmonic`](Self::harmonic).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::effects::{AudioEffects, HarmonicPercussiveConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let mut x = sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate!(22050), 0.5);
    /// x.harmonic_in_place(&HarmonicPercussiveConfig::default()).unwrap();
    /// assert_eq!(x.samples_per_channel().get(), 4410);
    /// ```
    fn harmonic_in_place(&mut self, config: &HarmonicPercussiveConfig) -> AudioSampleResult<()>;

    /// Extracts the percussive component, equivalent to
    /// `librosa.effects.percussive(y)`.
    ///
    /// # Arguments
    ///
    /// * `config` - STFT and mask parameters.
    ///
    /// # Returns
    ///
    /// The percussive component, same length as the input.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the configuration is invalid.
    ///
    /// # Differences from librosa
    ///
    /// As [`harmonic`](Self::harmonic).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::effects::{AudioEffects, HarmonicPercussiveConfig};
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::Array1;
    ///
    /// // A click train is percussive.
    /// let x = Array1::from_shape_fn(22050, |n| if n % 4410 == 0 { 1.0 } else { 0.0 });
    /// let audio = AudioSamples::new_mono(x, sample_rate!(22050)).unwrap();
    /// let p = audio.percussive(&HarmonicPercussiveConfig::default()).unwrap();
    /// assert!(p.as_mono().unwrap()[4410] > 0.5);
    /// ```
    fn percussive(
        &self,
        config: &HarmonicPercussiveConfig,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// In-place [`percussive`](Self::percussive).
    ///
    /// # Errors
    ///
    /// As [`percussive`](Self::percussive).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::effects::{AudioEffects, HarmonicPercussiveConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let mut x = sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate!(22050), 0.5);
    /// x.percussive_in_place(&HarmonicPercussiveConfig::default()).unwrap();
    /// assert_eq!(x.samples_per_channel().get(), 4410);
    /// ```
    fn percussive_in_place(&mut self, config: &HarmonicPercussiveConfig) -> AudioSampleResult<()>;

    /// Separates harmonic and percussive components, equivalent to
    /// `librosa.effects.hpss(y)`.
    ///
    /// # Arguments
    ///
    /// * `config` - STFT and mask parameters.
    ///
    /// # Returns
    ///
    /// `(harmonic, percussive)`, each the input's length. With margins of 1 they sum
    /// to the input (up to round-off).
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the configuration is invalid.
    ///
    /// # Differences from librosa
    ///
    /// As [`harmonic`](Self::harmonic).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::effects::{AudioEffects, HarmonicPercussiveConfig};
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let x = sine_wave::<f64>(440.0, Duration::from_millis(300), sample_rate!(22050), 0.5);
    /// let (h, p) = x.harmonic_percussive(&HarmonicPercussiveConfig::default()).unwrap();
    /// let (x, h, p) = (x.as_mono().unwrap(), h.as_mono().unwrap(), p.as_mono().unwrap());
    /// assert!((0..x.len().get()).all(|i| (h[i] + p[i] - x[i]).abs() < 1e-9));
    /// ```
    #[allow(clippy::type_complexity)]
    fn harmonic_percussive(
        &self,
        config: &HarmonicPercussiveConfig,
    ) -> AudioSampleResult<(
        AudioSamples<'static, Self::Sample>,
        AudioSamples<'static, Self::Sample>,
    )>;

    /// LPC coefficients of every channel by Burg's method, equivalent to
    /// `librosa.lpc(y, order=order)`; see [`lpc`].
    ///
    /// # Arguments
    ///
    /// * `order` - Prediction order; each channel needs more than `order` samples.
    ///
    /// # Returns
    ///
    /// A `(channels, order + 1)` array; row `c` is `[1, a_1, ..., a_order]`.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if a channel is too short or not finite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::nzu;
    /// use audio_samples::operations::effects::AudioEffects;
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let x = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(22050), 0.5);
    /// let a = x.lpc(nzu!(2)).unwrap();
    /// // A pure sine is an AR(2) process: a_1 = -2 cos(w), a_2 = 1.
    /// let w = 2.0 * std::f64::consts::PI * 440.0 / 22050.0;
    /// assert!((a[[0, 1]] + 2.0 * w.cos()).abs() < 1e-3);
    /// assert!((a[[0, 2]] - 1.0).abs() < 1e-3);
    /// ```
    fn lpc(&self, order: NonZeroUsize) -> AudioSampleResult<Array2<f64>>;

    /// Reconstructs audio from a magnitude spectrogram with Griffin-Lim, equivalent
    /// to `librosa.griffinlim(S)`; see [`griffinlim`].
    ///
    /// # Arguments
    ///
    /// * `magnitude` - `(channels, n_fft / 2 + 1, n_frames)` magnitude spectrogram.
    /// * `config` - Iterations, momentum, initial phase, STFT, length.
    /// * `sample_rate` - Sample rate of the returned audio.
    ///
    /// # Returns
    ///
    /// Mono `f64` audio for one channel, multi-channel otherwise.
    ///
    /// # Errors
    ///
    /// As [`griffinlim`].
    ///
    /// # Differences from librosa
    ///
    /// As [`griffinlim`] (random initial phases use this crate's RNG).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::operations::effects::{AudioEffects, GriffinLimConfig, GriffinLimInit};
    /// use audio_samples::operations::spectrum::{AudioSpectrum, StftConfig};
    /// use audio_samples::{AudioSamples, sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let x = sine_wave::<f64>(440.0, Duration::from_millis(300), sample_rate!(22050), 0.5);
    /// let mag = x.spectrogram_matrix(&StftConfig::default(), 1.0).unwrap();
    /// let cfg = GriffinLimConfig::new().with_n_iter(4).with_init(GriffinLimInit::Random(Some(1)));
    /// let y = AudioSamples::<f64>::griffinlim_audio(&mag, &cfg, sample_rate!(22050)).unwrap();
    /// assert_eq!(y.sample_rate(), sample_rate!(22050));
    /// ```
    fn griffinlim_audio(
        magnitude: &Array3<f64>,
        config: &GriffinLimConfig,
        sample_rate: SampleRate,
    ) -> AudioSampleResult<AudioSamples<'static, f64>>
    where
        Self: Sized;
}

impl<T> AudioEffects for AudioSamples<'_, T>
where
    T: StandardSample,
{
    type Sample = T;

    fn time_stretch(
        &self,
        rate: f64,
        stft: &StftConfig,
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        let y = time_stretch_rows(audio_rows(self).view(), rate, stft)?;
        rows_to_audio(y, self.sample_rate(), "time_stretch")
    }

    fn time_stretch_in_place(&mut self, rate: f64, stft: &StftConfig) -> AudioSampleResult<()> {
        *self = self.time_stretch(rate, stft)?.into_owned();
        Ok(())
    }

    fn pitch_shift(
        &self,
        n_steps: f64,
        config: &PitchShiftConfig,
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        let y = pitch_shift_rows(audio_rows(self).view(), n_steps, config)?;
        rows_to_audio(y, self.sample_rate(), "pitch_shift")
    }

    fn pitch_shift_in_place(
        &mut self,
        n_steps: f64,
        config: &PitchShiftConfig,
    ) -> AudioSampleResult<()> {
        *self = self.pitch_shift(n_steps, config)?.into_owned();
        Ok(())
    }

    fn preemphasis(&self, coef: f64) -> AudioSampleResult<AudioSamples<'static, T>> {
        let mut rows = audio_rows(self);
        for mut row in rows.outer_iter_mut() {
            let y = match row.as_slice() {
                Some(x) => preemphasis_with_state(x, coef, None)?.0,
                None => preemphasis_with_state(&row.to_vec(), coef, None)?.0,
            };
            row.assign(&Array1::from(y));
        }
        rows_to_audio(rows, self.sample_rate(), "preemphasis")
    }

    fn preemphasis_in_place(&mut self, coef: f64) -> AudioSampleResult<()> {
        *self = self.preemphasis(coef)?.into_owned();
        Ok(())
    }

    fn deemphasis(&self, coef: f64) -> AudioSampleResult<AudioSamples<'static, T>> {
        let mut rows = audio_rows(self);
        for mut row in rows.outer_iter_mut() {
            let y = match row.as_slice() {
                Some(x) => deemphasis_with_state(x, coef, None)?.0,
                None => deemphasis_with_state(&row.to_vec(), coef, None)?.0,
            };
            row.assign(&Array1::from(y));
        }
        rows_to_audio(rows, self.sample_rate(), "deemphasis")
    }

    fn deemphasis_in_place(&mut self, coef: f64) -> AudioSampleResult<()> {
        *self = self.deemphasis(coef)?.into_owned();
        Ok(())
    }

    fn trim_db(
        &self,
        config: &SilenceConfig,
    ) -> AudioSampleResult<(AudioSamples<'static, T>, (usize, usize))> {
        let rows = audio_rows(self);
        let flags = nonsilent_frames(rows.view(), config)?;
        let (start, end) = trim_bounds(&flags, config.hop_length.get(), rows.ncols());
        if start >= end {
            return Err(AudioSampleError::empty_data("trim_db"));
        }
        let kept = rows.slice(ndarray::s![.., start..end]).to_owned();
        Ok((
            rows_to_audio(kept, self.sample_rate(), "trim_db")?,
            (start, end),
        ))
    }

    fn trim_db_in_place(&mut self, config: &SilenceConfig) -> AudioSampleResult<(usize, usize)> {
        let (audio, bounds) = self.trim_db(config)?;
        *self = audio.into_owned();
        Ok(bounds)
    }

    fn split_db(&self, config: &SilenceConfig) -> AudioSampleResult<Vec<(usize, usize)>> {
        let rows = audio_rows(self);
        let flags = nonsilent_frames(rows.view(), config)?;
        Ok(split_intervals(
            &flags,
            config.hop_length.get(),
            rows.ncols(),
        ))
    }

    fn remix(
        &self,
        intervals: &[(usize, usize)],
        align_zeros: bool,
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        let y = remix_rows(audio_rows(self).view(), intervals, align_zeros)?;
        rows_to_audio(y, self.sample_rate(), "remix")
    }

    fn remix_in_place(
        &mut self,
        intervals: &[(usize, usize)],
        align_zeros: bool,
    ) -> AudioSampleResult<()> {
        *self = self.remix(intervals, align_zeros)?.into_owned();
        Ok(())
    }

    fn harmonic(
        &self,
        config: &HarmonicPercussiveConfig,
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        let mut parts = hpss_rows(audio_rows(self).view(), config, &[HpssPart::Harmonic])?;
        rows_to_audio(parts.remove(0), self.sample_rate(), "harmonic")
    }

    fn harmonic_in_place(&mut self, config: &HarmonicPercussiveConfig) -> AudioSampleResult<()> {
        *self = self.harmonic(config)?.into_owned();
        Ok(())
    }

    fn percussive(
        &self,
        config: &HarmonicPercussiveConfig,
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        let mut parts = hpss_rows(audio_rows(self).view(), config, &[HpssPart::Percussive])?;
        rows_to_audio(parts.remove(0), self.sample_rate(), "percussive")
    }

    fn percussive_in_place(&mut self, config: &HarmonicPercussiveConfig) -> AudioSampleResult<()> {
        *self = self.percussive(config)?.into_owned();
        Ok(())
    }

    fn harmonic_percussive(
        &self,
        config: &HarmonicPercussiveConfig,
    ) -> AudioSampleResult<(AudioSamples<'static, T>, AudioSamples<'static, T>)> {
        let mut parts = hpss_rows(
            audio_rows(self).view(),
            config,
            &[HpssPart::Harmonic, HpssPart::Percussive],
        )?;
        let p = parts
            .pop()
            .ok_or_else(|| AudioSampleError::empty_data("hpss"))?;
        let h = parts
            .pop()
            .ok_or_else(|| AudioSampleError::empty_data("hpss"))?;
        Ok((
            rows_to_audio(h, self.sample_rate(), "harmonic_percussive")?,
            rows_to_audio(p, self.sample_rate(), "harmonic_percussive")?,
        ))
    }

    fn lpc(&self, order: NonZeroUsize) -> AudioSampleResult<Array2<f64>> {
        let rows = audio_rows(self);
        let mut out = Array2::<f64>::zeros((rows.nrows(), order.get() + 1));
        for (c, row) in rows.outer_iter().enumerate() {
            let a = lpc(&row.to_vec(), order)?;
            out.row_mut(c).assign(&Array1::from(a));
        }
        Ok(out)
    }

    fn griffinlim_audio(
        magnitude: &Array3<f64>,
        config: &GriffinLimConfig,
        sample_rate: SampleRate,
    ) -> AudioSampleResult<AudioSamples<'static, f64>> {
        let y = griffinlim(magnitude.view(), config)?;
        rows_to_audio(y, sample_rate, "griffinlim")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample_rate;
    use ndarray::array;

    fn noise(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    fn mono(x: Vec<f64>) -> AudioSamples<'static, f64> {
        AudioSamples::new_mono(Array1::from(x), sample_rate!(22050)).unwrap()
    }

    #[test]
    fn phase_vocoder_rate_one_is_identity() {
        let x = Array2::from_shape_vec((1, 4000), noise(4000, 1)).unwrap();
        let d = stft(x.view(), &StftConfig::default()).unwrap();
        let y = phase_vocoder(d.view(), 1.0).unwrap();
        assert_eq!(y.dim(), d.dim());
        for (a, b) in y.iter().zip(d.iter()) {
            assert!((a - b).norm() < 1e-9 * (1.0 + b.norm()));
        }
    }

    #[test]
    fn phase_vocoder_rejects_bad_input() {
        let d = Array3::from_elem((1, 3, 4), Complex::new(1.0, 0.0));
        assert!(phase_vocoder(d.view(), 0.0).is_err());
        assert!(phase_vocoder(d.view(), f64::NAN).is_err());
        assert!(phase_vocoder_at(d.view(), &[4.0]).is_err());
        assert!(phase_vocoder_at(d.view(), &[-0.1]).is_err());
        assert!(phase_vocoder_at(d.view(), &[]).is_err());
        // A single frame is handled (librosa's interp1d cannot).
        let one = Array3::from_elem((1, 3, 1), Complex::new(0.0, 2.0));
        let y = phase_vocoder(one.view(), 0.5).unwrap();
        assert_eq!(y.dim(), (1, 3, 2));
        assert!((y[[0, 0, 1]].norm() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn time_stretch_lengths_and_stereo() {
        let data = Array2::from_shape_fn((2, 10000), |(c, n)| ((n * (c + 1)) as f64 * 0.03).sin());
        let x = AudioSamples::new_multi_channel(data, sample_rate!(22050)).unwrap();
        for (rate, len) in [(2.0, 5000), (0.5, 20000), (1.25, 8000), (3.0, 3333)] {
            let y = x.time_stretch(rate, &StftConfig::default()).unwrap();
            assert_eq!(y.samples_per_channel().get(), len);
            assert_eq!(y.num_channels().get(), 2);
        }
        // round half to even: 5 / 2 = 2.5 -> 2.
        let short = mono(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let cfg = StftConfig::new().with_n_fft(crate::nzu!(4));
        assert_eq!(
            short
                .time_stretch(2.0, &cfg)
                .unwrap()
                .samples_per_channel()
                .get(),
            2
        );
        assert!(short.time_stretch(100.0, &cfg).is_err());
        assert!(short.time_stretch(-1.0, &cfg).is_err());
    }

    #[test]
    fn pitch_shift_moves_a_sine() {
        let sr = 22050.0;
        let x: Vec<f64> = (0..22050)
            .map(|n| (2.0 * std::f64::consts::PI * 440.0 * n as f64 / sr).sin() * 0.5)
            .collect();
        let audio = mono(x);
        let y = audio
            .pitch_shift(12.0, &PitchShiftConfig::default())
            .unwrap();
        assert_eq!(y.samples_per_channel().get(), 22050);
        // Count zero crossings in the middle: an octave up doubles them.
        let y = y.as_mono().unwrap().to_vec();
        let zc = y[5000..17000]
            .windows(2)
            .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
            .count() as f64;
        let expected = 2.0 * 880.0 * 12000.0 / sr;
        assert!(
            (zc - expected).abs() < 4.0,
            "zero crossings {zc} vs {expected}"
        );
        assert!(
            audio
                .pitch_shift(f64::NAN, &PitchShiftConfig::default())
                .is_err()
        );
    }

    #[test]
    fn resample_by_arbitrary_ratio_length() {
        let rows = Array2::from_shape_vec((1, 1000), noise(1000, 3)).unwrap();
        let r = 2.0f64.powf(-2.0 / 12.0);
        let y = resample_rows_by_ratio(rows, r, ResamplingQuality::High).unwrap();
        assert!((y.ncols() as f64 - 1000.0 * r).abs() <= 1.0);
    }

    #[test]
    fn preemphasis_state_matches_lfilter() {
        // Values from scipy / librosa (see the module tests in the reference suite).
        let (y, zf) = preemphasis_with_state(&[1.0, 3.0, 0.5], 0.97, None).unwrap();
        assert_eq!(y[0], 0.0);
        assert!((y[1] - 2.03).abs() < 1e-15);
        assert!((y[2] + 2.41).abs() < 1e-15);
        assert!((zf + 0.485).abs() < 1e-15);
        let (y, zf) = deemphasis_with_state(&[1.0, 3.0, 0.5], 0.97, Some(0.5)).unwrap();
        assert_eq!(y, vec![1.5, 4.455, 4.82135]);
        assert!((zf - 4.676_709_5).abs() < 1e-12);
        // Block processing with zf -> zi equals one pass.
        let x = noise(100, 5);
        let (full, _) = preemphasis_with_state(&x, 0.9, Some(0.1)).unwrap();
        let (a, z) = preemphasis_with_state(&x[..40], 0.9, Some(0.1)).unwrap();
        let (b, _) = preemphasis_with_state(&x[40..], 0.9, Some(z)).unwrap();
        assert_eq!([a, b].concat(), full);
        assert!(preemphasis_with_state(&[1.0], 0.97, None).is_err());
        assert!(preemphasis_with_state(&[], 0.97, Some(0.0)).is_err());
        assert!(deemphasis_with_state(&[1.0, 2.0], 3.0, None).is_err());
        assert!(deemphasis_with_state(&[1.0, 2.0], f64::INFINITY, Some(0.0)).is_err());
    }

    #[test]
    fn emphasis_round_trip_on_audio() {
        let x = mono(noise(500, 7));
        let y = x.preemphasis(0.97).unwrap().deemphasis(0.97).unwrap();
        let (a, b) = (x.as_mono().unwrap(), y.as_mono().unwrap());
        assert!(a.iter().zip(b.iter()).all(|(p, q)| (p - q).abs() < 1e-12));
        let mut z = x.clone();
        z.preemphasis_in_place(0.97).unwrap();
        assert_eq!(z, x.preemphasis(0.97).unwrap());
    }

    #[test]
    fn trim_and_split_on_bursts() {
        let n = 30000;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                if (5000..9000).contains(&i) || (20000..24000).contains(&i) {
                    (i as f64 * 0.2).sin()
                } else {
                    0.0
                }
            })
            .collect();
        let audio = mono(x);
        let cfg = SilenceConfig::default();
        let parts = audio.split_db(&cfg).unwrap();
        assert_eq!(parts.len(), 2);
        let (trimmed, (s, e)) = audio.trim_db(&cfg).unwrap();
        assert_eq!(s, parts[0].0);
        assert_eq!(e, parts[1].1);
        assert_eq!(trimmed.samples_per_channel().get(), e - s);
        // All-zero input: every frame is 0 dB relative to max = 0, so nothing is trimmed.
        let zeros = mono(vec![0.0; 3000]);
        assert_eq!(zeros.split_db(&cfg).unwrap(), vec![(0, 3000)]);
        // Absolute reference far above the signal: everything silent.
        let quiet = SilenceConfig::new().with_reference(DbReference::Value(1e6));
        assert!(audio.split_db(&quiet).unwrap().is_empty());
        assert!(audio.trim_db(&quiet).is_err());
        assert!(
            audio
                .split_db(&SilenceConfig::new().with_top_db(f64::NAN))
                .is_err()
        );
    }

    #[test]
    fn split_intervals_edges() {
        assert_eq!(
            split_intervals(&[true, true, false, true], 10, 35),
            vec![(0, 20), (30, 35)]
        );
        assert_eq!(
            split_intervals(&[false, true, false], 10, 100),
            vec![(10, 20)]
        );
        assert!(split_intervals(&[false, false], 10, 100).is_empty());
        assert_eq!(trim_bounds(&[false, true, true, false], 4, 100), (4, 12));
    }

    #[test]
    fn zero_crossings_and_matching() {
        let y = [0.5, -0.5, 1e-12, 0.3, -0.2];
        // 1e-12 counts as +0 (not a sign change from -0.5? signbit(-0.5)=1, signbit(0)=0: yes).
        assert_eq!(zero_crossing_positions(&y), vec![0, 1, 2, 4, 5]);
        let z = [0, 10, 20, 30];
        assert_eq!(match_event(14, &z), 1);
        assert_eq!(match_event(15, &z), 2); // tie between 10 and 20 -> middle (20)
        assert_eq!(match_event(100, &z), 3);
        assert_eq!(match_event(0, &z), 0);
    }

    #[test]
    fn remix_multichannel_and_errors() {
        let data = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let x = AudioSamples::new_multi_channel(data, sample_rate!(8000)).unwrap();
        let y = x.remix(&[(2, 4), (0, 1)], false).unwrap();
        assert_eq!(
            y.as_multi_channel().unwrap().as_view().to_owned(),
            array![[3.0, 4.0, 1.0], [7.0, 8.0, 5.0]]
        );
        assert!(x.remix(&[], false).is_err());
        assert!(x.remix(&[(3, 3), (9, 12)], false).is_err());
        // align_zeros on a positive-only signal: crossings only at 0 and len.
        let y = x.remix(&[(1, 3)], true).unwrap();
        assert_eq!(y.samples_per_channel().get(), 4);
    }

    #[test]
    fn median_filter_matches_scipy_reflect() {
        // scipy.ndimage.median_filter([5,1,2,3,4,9,0], size=k, mode="reflect").
        let x = Array3::from_shape_vec((1, 1, 7), vec![5.0, 1.0, 2.0, 3.0, 4.0, 9.0, 0.0]).unwrap();
        let f = |k| {
            median_filter_axis(x.view(), 2, k)
                .iter()
                .copied()
                .collect::<Vec<_>>()
        };
        assert_eq!(f(5), vec![2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0]);
        assert_eq!(f(4), vec![5.0, 5.0, 3.0, 3.0, 4.0, 4.0, 4.0]);
        assert_eq!(f(20), vec![4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);
        assert_eq!(f(1), vec![5.0, 1.0, 2.0, 3.0, 4.0, 9.0, 0.0]);
    }

    #[test]
    fn softmask_and_hpss_masks() {
        let x = array![[0.0, 1.0], [2.0, 3.0]];
        let r = array![[0.0, 1.0], [1.0, 0.0]];
        let m = softmask(&x, &r, 1.0, false).unwrap();
        assert_eq!(m, array![[0.0, 0.5], [2.0 / 3.0, 1.0]]);
        assert!(softmask(&x, &array![[0.0, 1.0]], 1.0, false).is_err());
        assert!(softmask(&x, &r.mapv(|v| -v - 1.0), 1.0, false).is_err());
        assert!(softmask(&x, &r, 0.0, false).is_err());

        let s = Array3::from_shape_vec(
            (1, 30, 40),
            noise(1200, 9).iter().map(|v| v.abs()).collect(),
        )
        .unwrap();
        let (mh, mp) = hpss_masks(s.view(), &HpssMaskConfig::default()).unwrap();
        assert!(
            mh.iter()
                .zip(mp.iter())
                .all(|(a, b)| (a + b - 1.0).abs() < 1e-12)
        );
        let (mh2, mp2) = hpss_masks(s.view(), &HpssMaskConfig::new().with_margin(2.0)).unwrap();
        assert!(
            mh2.iter()
                .zip(mp2.iter())
                .all(|(a, b)| a + b <= 1.0 + 1e-12)
        );
        assert!(hpss_masks(s.view(), &HpssMaskConfig::new().with_margin(0.5)).is_err());
        let (hard_h, _) =
            hpss_masks(s.view(), &HpssMaskConfig::new().with_power(f64::INFINITY)).unwrap();
        assert!(hard_h.iter().all(|&v| v == 0.0 || v == 1.0));
    }

    #[test]
    fn harmonic_percussive_reconstructs_stereo() {
        let data = Array2::from_shape_fn((2, 6000), |(c, n)| {
            ((n * (c + 2)) as f64 * 0.05).sin() + if n % 1000 == 0 { 1.0 } else { 0.0 }
        });
        let x = AudioSamples::new_multi_channel(data.clone(), sample_rate!(22050)).unwrap();
        let (h, p) = x
            .harmonic_percussive(&HarmonicPercussiveConfig::default())
            .unwrap();
        let (h, p) = (
            h.as_multi_channel().unwrap().as_view().to_owned(),
            p.as_multi_channel().unwrap().as_view().to_owned(),
        );
        let err = (&h + &p - &data).iter().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(err < 1e-9, "{err}");
        let h2 = x.harmonic(&HarmonicPercussiveConfig::default()).unwrap();
        assert_eq!(h2.as_multi_channel().unwrap().as_view().to_owned(), h);
    }

    #[test]
    fn griffinlim_zero_phase_is_deterministic_and_shaped() {
        let x = Array2::from_shape_vec((1, 3000), noise(3000, 11)).unwrap();
        let cfg = StftConfig::new().with_n_fft(crate::nzu!(256));
        let mag = stft(x.view(), &cfg).unwrap().mapv(Complex::norm);
        let gl = GriffinLimConfig::new()
            .with_stft(cfg)
            .with_n_iter(5)
            .with_init(GriffinLimInit::ZeroPhase)
            .with_length(Some(3000));
        let a = griffinlim(mag.view(), &gl).unwrap();
        let b = griffinlim(mag.view(), &gl).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.dim(), (1, 3000));
        let seeded = gl.clone().with_init(GriffinLimInit::Random(Some(3)));
        assert_eq!(
            griffinlim(mag.view(), &seeded).unwrap(),
            griffinlim(mag.view(), &seeded).unwrap()
        );
        assert!(griffinlim(mag.view(), &gl.clone().with_momentum(-1.0)).is_err());
        let wrong = gl.with_stft(StftConfig::default());
        assert!(griffinlim(mag.view(), &wrong).is_err());
    }

    #[test]
    fn nn_filter_aggregates() {
        let data = array![[1.0, 2.0, 3.0, 10.0]];
        let mut rec = Array2::zeros((4, 4));
        for j in [1, 2, 3] {
            rec[[j, 0]] = j as f64;
        }
        assert_eq!(
            nn_filter(data.view(), rec.view(), NnAggregate::Median).unwrap()[[0, 0]],
            3.0
        );
        let w = nn_filter(data.view(), rec.view(), NnAggregate::WeightedAverage).unwrap();
        assert!((w[[0, 0]] - (2.0 + 6.0 + 30.0) / 6.0).abs() < 1e-12);
        rec[[3, 0]] = 0.0;
        assert_eq!(
            nn_filter(data.view(), rec.view(), NnAggregate::Median).unwrap()[[0, 0]],
            2.5
        );
        assert!(nn_filter(data.view(), Array2::zeros((3, 3)).view(), NnAggregate::Mean).is_err());
    }

    #[test]
    fn lpc_recovers_ar_process() {
        // AR(2): y[n] = 1.5 y[n-1] - 0.7 y[n-2] + e[n].
        let e = noise(20000, 13);
        let mut y = vec![0.0; e.len()];
        for n in 2..y.len() {
            y[n] = 1.5 * y[n - 1] - 0.7 * y[n - 2] + e[n];
        }
        let a = lpc(&y, crate::nzu!(2)).unwrap();
        assert!(
            (a[1] + 1.5).abs() < 0.02 && (a[2] - 0.7).abs() < 0.02,
            "{a:?}"
        );
        assert!(lpc(&[1.0, 2.0], crate::nzu!(2)).is_err());
        assert!(lpc(&[1.0, f64::NAN, 3.0], crate::nzu!(1)).is_err());
        let stereo = AudioSamples::new_multi_channel(
            Array2::from_shape_fn((2, 100), |(c, n)| ((n + c) as f64 * 0.3).sin()),
            sample_rate!(8000),
        )
        .unwrap();
        assert_eq!(stereo.lpc(crate::nzu!(3)).unwrap().dim(), (2, 4));
    }

    #[test]
    fn clicks_layout() {
        let cfg = ClicksConfig::new()
            .with_click(Some(vec![1.0, 0.5]))
            .with_length(Some(10));
        let y = clicks(
            ClickPositions::Frames(&[0, 2, 4, 5]),
            sample_rate!(100),
            &cfg.clone().with_hop_length(crate::nzu!(2)),
        )
        .unwrap();
        // Positions 0, 4, 8 (10 dropped); the click at 8 is truncated after 2 samples.
        assert_eq!(
            y.as_mono().unwrap().to_vec(),
            vec![1.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.5]
        );
        let y = clicks(
            ClickPositions::Times(&[0.0, 0.01]),
            sample_rate!(100),
            &cfg.with_length(None),
        )
        .unwrap();
        assert_eq!(y.as_mono().unwrap().to_vec(), vec![1.0, 1.5, 0.5]);
        let d = clicks(
            ClickPositions::Times(&[0.0]),
            sample_rate!(22050),
            &ClicksConfig::default(),
        )
        .unwrap();
        let d = d.as_mono().unwrap();
        assert_eq!(d.len().get(), 2205);
        assert_eq!(d[0], 0.0);
        assert!(
            clicks(
                ClickPositions::Times(&[]),
                sample_rate!(100),
                &ClicksConfig::default()
            )
            .is_err()
        );
        assert!(
            clicks(
                ClickPositions::Times(&[-1.0]),
                sample_rate!(100),
                &ClicksConfig::default()
            )
            .is_err()
        );
    }

    #[test]
    fn integer_audio_round_trips_type() {
        let x = AudioSamples::new_mono(
            Array1::from_shape_fn(8000, |n| ((n as f64 * 0.05).sin() * 10000.0) as i16),
            sample_rate!(16000),
        )
        .unwrap();
        let y: AudioSamples<'static, i16> = x.time_stretch(2.0, &StftConfig::default()).unwrap();
        assert_eq!(y.samples_per_channel().get(), 4000);
    }
}
