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
//! struct has a `librosa()` constructor with librosa 1.0's defaults. `Default`
//! equals it except where noted: [`SilenceConfig`] has no `Default` (the
//! threshold is required), and [`GriffinLimConfig::default`] seeds its random
//! initial phase. The spectral steps reuse the librosa-compatible STFT of
//! [`crate::operations::spectrum`].
//!
//! The older [`AudioDecomposition::hpss`](crate::AudioDecomposition::hpss) and
//! [`AudioEditing::trim_silence`](crate::AudioEditing::trim_silence) keep their own
//! (different) semantics; the methods here are the librosa-compatible variants.
//!
//! ## How
//!
//! Bring [`AudioEffects`] into scope and call it on any [`AudioSamples`]. Audio is
//! processed in the sample type's float type
//! ([`AudioSample::Float`](crate::AudioSample::Float): `f32` for `u8`, `i16`,
//! `I24` and `f32` audio, `f64` for `i32` and `f64` audio) and returned in the
//! input's sample type, converted with the crate's audio-aware conversion
//! (integer types round and saturate). Every audio-to-audio operation has an
//! `_in_place` variant. Spectrogram-level helpers take
//! `(channels, n_bins, n_frames)` arrays as produced by
//! [`AudioSpectrum::stft_matrix`](crate::operations::spectrum::AudioSpectrum::stft_matrix)
//! and are generic over the element type ([`AudioFloat`]: `f32` or `f64`); they
//! compute in and return that type. Parameters in the config structs stay `f64`
//! and are rounded to the element type once.
//!
//! Two steps run in `f64` for every sample type and round their result to the
//! float type once: the analysis STFT of [`AudioEffects::time_stretch`] and
//! [`AudioEffects::pitch_shift`] (the phase vocoder amplifies the round-off of
//! an `f32` STFT), and Burg's recursion in [`lpc`] (unstable in `f32` on tonal
//! signals). Each holds transient `f64` copies of its input and output. The
//! resampling stage of `pitch_shift` runs in the float type.
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
//! let intervals = audio.split_db(&SilenceConfig::librosa()).unwrap();
//! assert_eq!(intervals.len(), 1);
//!
//! // 16-bit audio is processed in f32 and returned as i16.
//! let pcm = sine_wave::<i16>(440.0, Duration::from_millis(500), sample_rate!(22050), 0.5);
//! let fast: audio_samples::AudioSamples<'static, i16> =
//!     pcm.time_stretch(2.0, &StftConfig::default()).unwrap();
//! assert_eq!(fast.samples_per_channel().get(), n / 2);
//! ```

use crate::utils::{MAX_SIGNAL_ELEMENTS, check_derived_len};
use std::num::NonZeroUsize;

use ndarray::{
    Array, Array1, Array2, Array3, ArrayBase, ArrayView2, ArrayView3, Axis, Data, Dimension,
};
use num_complex::Complex;

use crate::audio_float::{finite_float_rows, float_rows_to_audio};
use crate::operations::sequence::{RecurrenceConfig, recurrence_matrix_weighted};
use crate::operations::spectrum::{
    DbAxes, DbConfig, DbReference, StftConfig, StftPadMode, amplitude_to_db, frame_square_sums,
    istft, stft,
};
use crate::operations::types::ResamplingQuality;
use crate::repr::SampleRate;
use crate::{
    AudioFloat, AudioSampleError, AudioSampleResult, AudioSamples, ParameterError, StandardSample,
};

fn param_err(parameter: &str, reason: impl ToString) -> AudioSampleError {
    AudioSampleError::Parameter(ParameterError::invalid_value(parameter, reason))
}

fn config_err(operation: &str, reason: impl ToString) -> AudioSampleError {
    AudioSampleError::Parameter(ParameterError::invalid_configuration(operation, reason))
}

/// The audio as `(channels, samples)` `f64` rows, converted with the crate's
/// audio-aware conversion, rejecting NaN and infinite samples.
///
/// The vocoder-based operations analyse these rows (see [`time_stretch_rows`]):
/// `f64` samples of integer audio are the exact integer ratios, while the `f32`
/// conversion rounds them, and the phase vocoder amplifies that rounding.
fn finite_f64_rows<T: StandardSample>(
    audio: &AudioSamples<'_, T>,
) -> AudioSampleResult<Array2<f64>> {
    use crate::AudioTypeConversion;
    let audio = audio.to_format::<f64>();
    Ok(finite_float_rows(&audio)?.into_owned())
}

/// `a.total_cmp(b)` for either float type (`f32` to `f64` is exact and keeps
/// the total order, including signed zeros).
#[inline]
fn total_cmp<F: AudioFloat>(a: &F, b: &F) -> std::cmp::Ordering {
    a.as_f64().total_cmp(&b.as_f64())
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
/// A `(channels, n_bins, ceil(n_frames / rate))` complex array of the input's
/// element type (`Complex<f32>` or `Complex<f64>`); the computation runs in that
/// type.
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
pub fn phase_vocoder<F: AudioFloat>(
    stft_matrix: ArrayView3<'_, Complex<F>>,
    rate: f64,
) -> AudioSampleResult<Array3<Complex<F>>> {
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
///   The positions are `f64`; each interpolation weight is rounded to the element
///   type once.
///
/// # Returns
///
/// A `(channels, n_bins, t_out.len())` complex array of the input's element
/// type. The magnitudes, phases and the phase accumulator are computed in that
/// type, as librosa does for `complex64` input.
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
pub fn phase_vocoder_at<F: AudioFloat>(
    stft_matrix: ArrayView3<'_, Complex<F>>,
    t_out: &[f64],
) -> AudioSampleResult<Array3<Complex<F>>> {
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

    let mut out = Array3::<Complex<F>>::zeros((n_channels, n_bins, n_out));
    if std::mem::size_of::<F>() != 8 {
        // f32: angles as librosa's float32 path computes them (the phasor
        // products below would round differently from librosa float32 by
        // more than f32 round-off over many frames).
        vocoder_by_angle(stft_matrix, t_out, &i0, &i1, &seg, &mut out);
        return Ok(out);
    }
    // f64: the phase is carried as a unit phasor instead of an angle: the output is
    // `mag · e^{i phi_j}` with `phi_j = arg D[i0_0] + sum (arg D[i1] - arg D[i0])`,
    // i.e. `u[i0_0] · prod u[i1] · conj(u[i0])` with `u = D / |D|`. That is the
    // same quantity without `atan2`, `sin` and `cos` (the bulk of the cost), and
    // without the round-off of `cos` / `sin` of a large accumulated angle.
    // `u` and `|D|` are computed once per element (see [`for_bin_blocks`]).
    for_bin_blocks(stft_matrix, unit_and_norm, |c, k, u_row, m_row| {
        let mut out_row = out.slice_mut(ndarray::s![c, k, ..]);
        let mut phase = u_row[i0[0]];
        for j in 0..n_out {
            if j > 0 {
                let (a, b) = (u_row[i1[j - 1]], u_row[i0[j - 1]]);
                // phase · a · conj(b)
                let w = Complex::new(a.re * b.re + a.im * b.im, a.im * b.re - a.re * b.im);
                phase = Complex::new(
                    phase.re * w.re - phase.im * w.im,
                    phase.re * w.im + phase.im * w.re,
                );
                // Keep |phase| = 1 against the round-off of the products.
                if j % 64 == 0 {
                    phase = phase / phase.norm();
                }
            }
            let (lo, hi) = seg[j];
            let mag = if n_frames == 1 {
                m_row[0]
            } else {
                let (y_lo, y_hi) = (m_row[lo], m_row[hi]);
                (y_hi - y_lo) / F::of_usize(hi - lo) * F::of(t_out[j] - lo as f64) + y_lo
            };
            out_row[j] = Complex::new(phase.re * mag, phase.im * mag);
        }
    });
    Ok(out)
}

/// [`phase_vocoder_at`]'s loop with the phase as an angle: `atan2` of every
/// input value, the accumulated angle differences, `cos` and `sin` per output.
fn vocoder_by_angle<F: AudioFloat>(
    stft_matrix: ArrayView3<'_, Complex<F>>,
    t_out: &[f64],
    i0: &[usize],
    i1: &[usize],
    seg: &[(usize, usize)],
    out: &mut Array3<Complex<F>>,
) {
    let n_frames = stft_matrix.dim().2;
    let angle_and_norm = |z: Complex<F>| (z.arg(), z.norm());
    for_bin_blocks(stft_matrix, angle_and_norm, |c, k, arg, mag_in| {
        let mut out_row = out.slice_mut(ndarray::s![c, k, ..]);
        let mut phase = arg[i0[0]];
        for j in 0..t_out.len() {
            if j > 0 {
                phase += arg[i1[j - 1]] - arg[i0[j - 1]];
            }
            let (lo, hi) = seg[j];
            let mag = if n_frames == 1 {
                mag_in[0]
            } else {
                let (y_lo, y_hi) = (mag_in[lo], mag_in[hi]);
                (y_hi - y_lo) / F::of_usize(hi - lo) * F::of(t_out[j] - lo as f64) + y_lo
            };
            out_row[j] = Complex::new(phase.cos() * mag, phase.sin() * mag);
        }
    });
}

/// Bins per block in [`for_bin_blocks`].
const VOCODER_BIN_BLOCK: usize = 16;

/// Calls `body(channel, bin, a_row, b_row)` for every bin of the STFT, where
/// `(a_row[t], b_row[t]) = f(D[channel, bin, t])` over all frames.
///
/// The values are computed in blocks of [`VOCODER_BIN_BLOCK`] bins, reading the
/// input in its memory order: for a Fortran-order STFT (librosa's layout, bins
/// contiguous within a frame) each frame's run of block bins is contiguous,
/// for a C-order one each bin's row. The scratch is two `16 × n_frames`
/// arrays.
fn for_bin_blocks<F: AudioFloat, A: Copy + Default, B: Copy + Default>(
    stft_matrix: ArrayView3<'_, Complex<F>>,
    f: impl Fn(Complex<F>) -> (A, B),
    mut body: impl FnMut(usize, usize, &[A], &[B]),
) {
    let (n_channels, n_bins, n_frames) = stft_matrix.dim();
    let mut a = vec![A::default(); VOCODER_BIN_BLOCK * n_frames];
    let mut b = vec![B::default(); VOCODER_BIN_BLOCK * n_frames];
    for c in 0..n_channels {
        let d = stft_matrix.index_axis(Axis(0), c);
        let frame_major = d.strides()[0].unsigned_abs() < d.strides()[1].unsigned_abs();
        for k0 in (0..n_bins).step_by(VOCODER_BIN_BLOCK) {
            let kb = VOCODER_BIN_BLOCK.min(n_bins - k0);
            if frame_major {
                for t in 0..n_frames {
                    for kk in 0..kb {
                        let (x, y) = f(d[[k0 + kk, t]]);
                        a[kk * n_frames + t] = x;
                        b[kk * n_frames + t] = y;
                    }
                }
            } else {
                for kk in 0..kb {
                    let row = d.row(k0 + kk);
                    let (ra, rb) = (&mut a[kk * n_frames..], &mut b[kk * n_frames..]);
                    for ((x, y), z) in ra.iter_mut().zip(rb.iter_mut()).zip(row.iter()) {
                        (*x, *y) = f(*z);
                    }
                }
            }
            for kk in 0..kb {
                let r = kk * n_frames..(kk + 1) * n_frames;
                body(c, k0 + kk, &a[r.clone()], &b[r]);
            }
        }
    }
}

/// The unit phasor `e^{i arg z}` and `|z|` of one STFT value. `|z|` is
/// `sqrt(re² + im²)` where that cannot overflow or lose precision to
/// underflow, else `hypot`; a zero value keeps the phasor of `atan2(im, re)`
/// (`±1`, as the signs of the zeros give).
#[inline]
fn unit_and_norm<F: AudioFloat>(z: Complex<F>) -> (Complex<F>, F) {
    let s = z.re * z.re + z.im * z.im;
    let m = if s > F::min_positive_value() && s < F::max_value() {
        s.sqrt()
    } else {
        z.re.hypot(z.im)
    };
    if m > F::zero() {
        (Complex::new(z.re / m, z.im / m), m)
    } else {
        let a = z.im.atan2(z.re);
        (Complex::new(a.cos(), a.sin()), m)
    }
}

/// `librosa.effects.time_stretch` on `(channels, samples)` rows of `f64`,
/// computed in `F`.
///
/// The analysis STFT is computed in `f64` from the `f64` samples and rounded
/// once to `Complex<F>`; the phase vocoder and the inverse STFT run in `F`.
/// librosa's `float32` path does the same (its `float64` window promotes the
/// frames, so its FFT runs in `float64`). The phase vocoder sums the frame-to-frame
/// phase differences of every bin, including bins whose value is at the
/// round-off level of the transform, and their phases decide the output where a
/// signal later reaches them: an `f32` FFT would put those bins at about 1e-7
/// of the peak and change the output by far more than `f32` round-off.
///
/// Memory: for `F = f32` the `f64` rows and the `f64` complex STFT (twice the
/// size of the `f32` one) are held until the STFT is rounded. For `F = f64` the
/// rounding reuses the STFT's buffer.
fn time_stretch_rows<F: AudioFloat>(
    rows: ArrayView2<'_, f64>,
    rate: f64,
    config: &StftConfig,
) -> AudioSampleResult<Array2<F>> {
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
    // The stretched STFT and the output both scale with n / rate.
    check_derived_len(
        "rate",
        "the time-stretched signal",
        len_stretch * rows.nrows() as f64,
        MAX_SIGNAL_ELEMENTS,
    )?;
    let d = stft(rows, config)?.mapv_into_any(|z| Complex::new(F::of(z.re), F::of(z.im)));
    let d_stretch = phase_vocoder(d.view(), rate)?;
    istft(d_stretch.view(), config, Some(len_stretch as usize))
}

/// Resamples rows by an arbitrary positive ratio `out_rate / in_rate` with the
/// crate's windowed-sinc resampler, fixing the length to `n_out` samples when
/// given (`librosa.util.fix_length`).
///
/// The resampler works with integer rates, so the ratio is represented by two
/// integers near 2^31 (relative ratio error below 1e-9). It filters in the
/// element type `F`.
fn resample_rows_by_ratio<F: AudioFloat>(
    rows: ArrayView2<'_, F>,
    ratio: f64,
    quality: ResamplingQuality,
    n_out: Option<usize>,
) -> AudioSampleResult<Array2<F>> {
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
    crate::resampling::resample_float_rows(rows, in_sr.get(), out_sr.get(), quality, n_out)
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
/// `Default` (and [`PitchShiftConfig::librosa`]) uses librosa's defaults (12 bins
/// per octave, n_fft 2048, hop 512, periodic Hann) with
/// [`ResamplingQuality::High`], this crate's closest match to librosa's default
/// `res_type="soxr_hq"`.
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
    /// Default: 12 (semitones), as librosa.
    pub bins_per_octave: NonZeroUsize,
    /// Quality of the final resampling step. Default:
    /// [`ResamplingQuality::High`]. librosa uses soxr (`soxr_hq`), which this
    /// crate does not include.
    pub quality: ResamplingQuality,
    /// STFT used by the phase-vocoder time stretch. Default:
    /// [`StftConfig::default`], librosa's STFT defaults.
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
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// librosa 1.0's defaults, with [`ResamplingQuality::High`] in place of
    /// librosa's `soxr_hq` resampler. Currently the same as [`Default`]; use this
    /// constructor where agreement with librosa is the requirement.
    #[must_use]
    pub fn librosa() -> Self {
        Self {
            bins_per_octave: crate::nzu!(12),
            quality: ResamplingQuality::High,
            stft: StftConfig::librosa(),
        }
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
fn pitch_shift_rows<F: AudioFloat>(
    rows: ArrayView2<'_, f64>,
    n_steps: f64,
    config: &PitchShiftConfig,
) -> AudioSampleResult<Array2<F>> {
    config.validate()?;
    if !n_steps.is_finite() {
        return Err(param_err("n_steps", "n_steps must be finite"));
    }
    let n = rows.ncols();
    let rate = 2.0f64.powf(-n_steps / config.bins_per_octave.get() as f64);
    let stretched = time_stretch_rows::<F>(rows, rate, &config.stft)?;
    // librosa: resample(orig_sr = sr / rate, target_sr = sr), i.e. by the ratio `rate`.
    resample_rows_by_ratio(stretched.view(), rate, config.quality, Some(n))
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
/// * `x` - Input samples (`f32` or `f64`).
/// * `coef` - Pre-emphasis coefficient (librosa default 0.97). Any finite value;
///   `0` is the identity (apart from `zi`), `1` a first difference. It is rounded
///   to the element type of `x` once.
/// * `zi` - Initial filter state. `None` uses librosa's default `2 x[0] - x[1]`.
///
/// # Returns
///
/// `(y, zf)`: the filtered samples and the final state `-coef * x[n - 1]`, in the
/// element type of `x`; the filter runs in that type.
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
/// let (y, zf) = preemphasis_with_state(&[1.0_f64, 3.0, 0.5], 0.97, None).unwrap();
/// // zi = 2 * 1 - 3 = -1, so y[0] = 1 + (-1) = 0.
/// assert_eq!(y[0], 0.0);
/// assert!((y[1] - (3.0 - 0.97)).abs() < 1e-15);
/// assert!((zf + 0.97 * 0.5).abs() < 1e-15);
/// ```
pub fn preemphasis_with_state<F: AudioFloat>(
    x: &[F],
    coef: f64,
    zi: Option<F>,
) -> AudioSampleResult<(Vec<F>, F)> {
    let mut y = vec![F::zero(); x.len()];
    let zf = preemphasis_into(x, coef, zi, &mut y)?;
    Ok((y, zf))
}

/// [`preemphasis_with_state`] writing the output into `out` (same length as
/// `x`); returns the final state.
fn preemphasis_into<F: AudioFloat>(
    x: &[F],
    coef: f64,
    zi: Option<F>,
    out: &mut [F],
) -> AudioSampleResult<F> {
    check_coef(coef)?;
    let zi = match zi {
        Some(z) if z.is_finite() => z,
        Some(_) => return Err(param_err("zi", "the initial state must be finite")),
        None if x.len() >= 2 => F::of(2.0) * x[0] - x[1],
        None => {
            return Err(param_err(
                "x",
                "the default initial state 2*x[0] - x[1] needs at least two samples",
            ));
        }
    };
    let Some((&last, _)) = x.split_last() else {
        return Err(AudioSampleError::empty_data("preemphasis"));
    };
    assert_eq!(out.len(), x.len(), "output length must match the input");
    let neg_coef = F::of(-coef);
    // y[0] = x[0] + zi, y[n] = x[n] + (-coef * x[n - 1]): no recursion, so the
    // loop vectorises.
    out[0] = x[0] + zi;
    for ((o, &v), &prev) in out[1..].iter_mut().zip(&x[1..]).zip(x) {
        *o = v + neg_coef * prev;
    }
    Ok(neg_coef * last)
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
/// * `x` - Input samples (`f32` or `f64`).
/// * `coef` - Coefficient (librosa default 0.97); any finite value (`coef = 3` is
///   only allowed with an explicit `zi`). It is rounded to the element type of
///   `x` once.
/// * `zi` - Initial filter state, or `None` for librosa's default.
///
/// # Returns
///
/// `(y, zf)`: the filtered samples and the final filter state `coef * y'[n - 1]`,
/// where `y'` is the output *before* the default-state correction (as librosa
/// returns it). Both are in the element type of `x`; the filter runs in that
/// type.
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
/// let x = [0.1_f64, -0.4, 0.25, 0.9, -0.3];
/// let (pre, _) = preemphasis_with_state(&x, 0.97, None).unwrap();
/// let (back, _) = deemphasis_with_state(&pre, 0.97, None).unwrap();
/// assert!(back.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-12));
/// ```
pub fn deemphasis_with_state<F: AudioFloat>(
    x: &[F],
    coef: f64,
    zi: Option<F>,
) -> AudioSampleResult<(Vec<F>, F)> {
    let mut y = vec![F::zero(); x.len()];
    let zf = deemphasis_into(x, coef, zi, &mut y)?;
    Ok((y, zf))
}

/// [`deemphasis_with_state`] writing the output into `out` (same length as
/// `x`); returns the final state.
///
/// The result is the same, operation for operation, as the direct recursion
/// `y = x + coef * y_prev` followed by `y[n] -= c * p_n` with `p_0 = 1`,
/// `p_{n+1} = p_n * coef`. Two exact shortcuts avoid long runs of subnormal
/// arithmetic, which is about a hundred times slower than normal arithmetic
/// on x86:
///
/// - in the recursion, once the output is a non-zero subnormal value `y`
///   with `coef * y == y` (the decay has stopped at its rounding fixed point)
///   and the input is zero, every further output is `y` until the input
///   becomes non-zero;
/// - in the correction, once `p_{n+1} == p_n` the term `c * p_n` is the same
///   constant for every later sample.
fn deemphasis_into<F: AudioFloat>(
    x: &[F],
    coef: f64,
    zi: Option<F>,
    out: &mut [F],
) -> AudioSampleResult<F> {
    check_coef(coef)?;
    if x.is_empty() {
        return Err(AudioSampleError::empty_data("deemphasis"));
    }
    assert_eq!(out.len(), x.len(), "output length must match the input");
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
            (
                F::zero(),
                Some((F::of(2.0 - coef) * x[0] - x[1]) / F::of(3.0 - coef)),
            )
        }
    };
    let coef = F::of(coef);
    let tiny = F::min_positive_value();
    let n = x.len();
    let mut z = z0;
    let mut i = 0;
    while i < n {
        let y = x[i] + z;
        out[i] = y;
        z = coef * y;
        i += 1;
        if y.abs() < tiny && y != F::zero() && z == y {
            // Fixed point of the decay: y = 0 + z = z while the input is zero.
            while i < n && x[i] == F::zero() {
                out[i] = y;
                i += 1;
            }
        }
    }
    if let Some(c) = correction {
        let mut p = F::one();
        let mut k = 0;
        while k < n {
            out[k] -= c * p;
            k += 1;
            let next = p * coef;
            if next == p {
                break;
            }
            p = next;
        }
        if k < n {
            // `p` no longer changes: subtract the constant term.
            let t = c * p;
            for v in &mut out[k..] {
                *v -= t;
            }
        }
    }
    Ok(z)
}

/// Applies `filter(row, out_row)` to every row of `rows`, writing a new
/// `(channels, samples)` array. Non-contiguous rows are copied into a buffer
/// first.
fn filter_rows<F: AudioFloat>(
    rows: ArrayView2<'_, F>,
    mut filter: impl FnMut(&[F], &mut [F]) -> AudioSampleResult<()>,
) -> AudioSampleResult<Array2<F>> {
    let mut out = Array2::<F>::zeros(rows.raw_dim());
    let mut buf = Vec::new();
    for (row, mut dst) in rows.outer_iter().zip(out.outer_iter_mut()) {
        let x: &[F] = if let Some(s) = row.as_slice() {
            s
        } else {
            buf.clear();
            buf.extend(row.iter().copied());
            &buf
        };
        let dst = dst
            .as_slice_mut()
            .expect("rows of a new standard-layout array");
        filter(x, dst)?;
    }
    Ok(out)
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
/// The threshold depends on the material (speech, music, noise floor of the
/// recording), so there is no `Default`: [`SilenceConfig::new`] takes `top_db`
/// and sets the other fields to librosa's defaults (reference = the loudest frame,
/// `ref=numpy.max`; 2048-sample frames; 512-sample hop; `numpy.max` over
/// channels). [`SilenceConfig::librosa`] is librosa's full default, `top_db = 60`.
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
/// let cfg = SilenceConfig::new(30.0);
/// assert_eq!(cfg.frame_length.get(), 2048);
/// assert!(cfg.validate().is_ok());
/// assert_eq!(SilenceConfig::librosa(), SilenceConfig::new(60.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SilenceConfig {
    /// Threshold in dB below the reference under which a frame is silent.
    /// Required ([`SilenceConfig::new`]); librosa's default is 60.
    pub top_db: f64,
    /// Reference level (0 dB). [`DbReference::Max`] (set by [`SilenceConfig::new`],
    /// librosa's default) uses the loudest frame over all channels; `Value(r)` an
    /// absolute RMS amplitude.
    pub reference: DbReference,
    /// Samples per RMS frame (frames are centred, zero padded). Set to 2048 by
    /// [`SilenceConfig::new`], as librosa.
    pub frame_length: NonZeroUsize,
    /// Samples between frames; also the resolution of the returned sample
    /// indices. Set to 512 by [`SilenceConfig::new`], as librosa.
    pub hop_length: NonZeroUsize,
    /// Combination of the channels' dB levels. Set to `Max` by
    /// [`SilenceConfig::new`], as librosa.
    pub aggregate: ChannelAggregate,
}

impl SilenceConfig {
    /// A configuration with threshold `top_db` (dB below the reference) and
    /// librosa's defaults for the other fields: reference = loudest frame, frame
    /// length 2048, hop 512, `Max` over channels.
    #[must_use]
    pub fn new(top_db: f64) -> Self {
        Self {
            top_db,
            reference: DbReference::Max,
            frame_length: crate::nzu!(2048),
            hop_length: crate::nzu!(512),
            aggregate: ChannelAggregate::Max,
        }
    }

    /// librosa 1.0's defaults for `effects.trim` / `effects.split`: `top_db = 60`
    /// and the defaults of [`SilenceConfig::new`].
    #[must_use]
    pub fn librosa() -> Self {
        Self::new(60.0)
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
/// * `signal` - `(channels, samples)` array (`f32` or `f64`) with at least one
///   sample.
/// * `config` - Threshold, reference, framing and channel aggregation.
///
/// # Returns
///
/// One flag per frame (`1 + len / hop` frames); frame `t` covers samples centred
/// on `t * hop`. The RMS and dB levels are computed in the element type of
/// `signal`; `top_db` and a `Value` reference are rounded to it once.
///
/// # Errors
///
/// [`ParameterError`] if the configuration is invalid or the signal is empty.
///
/// # Differences from librosa
///
/// librosa sums the squared samples of a frame in NumPy's order (`numpy.mean`),
/// this function sequentially, so a frame whose level lies within round-off of
/// the threshold can be classified differently.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::effects::{SilenceConfig, nonsilent_frames};
/// use ndarray::Array2;
///
/// // Silence, then a loud segment.
/// let x = Array2::from_shape_fn((1, 4096), |(_, n)| if n >= 2048 { 0.5_f64 } else { 0.0 });
/// let cfg = SilenceConfig::librosa().with_frame_length(nzu!(512)).with_hop_length(nzu!(256));
/// let flags = nonsilent_frames(x.view(), &cfg).unwrap();
/// assert_eq!(flags.len(), 17);
/// assert!(!flags[0] && flags[16]);
/// ```
pub fn nonsilent_frames<F: AudioFloat>(
    signal: ArrayView2<'_, F>,
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
    let mut rms = Array2::<F>::zeros((n_channels, n_frames));
    let fl_f = F::of_usize(fl);
    let signal = signal.as_standard_layout();
    for (c, row) in signal.outer_iter().enumerate() {
        let x = row.to_slice().expect("rows of a standard-layout array");
        // Zero padding: the padded zeros add exact zeros to the sums.
        let sums = frame_square_sums(x, fl, hop, pad, StftPadMode::Constant, n_frames);
        for (r, &sum) in rms.row_mut(c).iter_mut().zip(&sums) {
            *r = (sum / fl_f).sqrt();
        }
    }
    let db_cfg = DbConfig::amplitude()
        .with_reference(config.reference)
        .with_top_db(None)
        .with_axes(DbAxes::All);
    let db = amplitude_to_db(&rms, &db_cfg)?;
    let threshold = F::of(-config.top_db);
    let n_channels_f = F::of_usize(n_channels);
    Ok((0..n_frames)
        .map(|t| {
            let col = db.column(t);
            let v = match config.aggregate {
                ChannelAggregate::Max => col.iter().copied().fold(F::neg_infinity(), F::max),
                ChannelAggregate::Min => col.iter().copied().fold(F::infinity(), F::min),
                ChannelAggregate::Mean => col.iter().copied().sum::<F>() / n_channels_f,
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
fn zero_crossing_positions<F: AudioFloat>(y: &[F]) -> Vec<usize> {
    let threshold = F::of(1e-10);
    let clip = |v: F| {
        if (-threshold..=threshold).contains(&v) {
            F::zero()
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
fn remix_rows<F: AudioFloat>(
    rows: ArrayView2<'_, F>,
    intervals: &[(usize, usize)],
    align_zeros: bool,
) -> AudioSampleResult<Array2<F>> {
    if intervals.is_empty() {
        return Err(param_err("intervals", "at least one interval is required"));
    }
    let (n_channels, len) = rows.dim();
    let zeros = if align_zeros {
        let mono: Vec<F> = if n_channels == 1 {
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
fn pow_fast<F: AudioFloat>(x: F, p: f64, p_f: F) -> F {
    if p == 1.0 {
        x
    } else if p == 2.0 {
        x * x
    } else if p == 0.5 {
        x.sqrt()
    } else {
        x.powf(p_f)
    }
}

/// Soft (Wiener-like) mask of `x` against `x_ref`, equivalent to
/// `librosa.util.softmask(X, X_ref, power=power, split_zeros=split_zeros)`.
///
/// With `Z = max(X, X_ref)` the mask is
/// `(X / Z)^p / ((X / Z)^p + (X_ref / Z)^p)`, i.e. `X^p / (X^p + X_ref^p)`
/// computed without overflow. Where both inputs are (numerically) zero, that is
/// `Z` below the smallest positive normal value of the element type (librosa's
/// `tiny`), the mask is `0.5` if `split_zeros`, else `0`. With
/// `power = f64::INFINITY` the mask is binary: `1` where `X > X_ref` (ties give
/// `0`).
///
/// # Arguments
///
/// * `x` - Non-negative values the mask selects (`f32` or `f64`).
/// * `x_ref` - Non-negative reference values, same shape and element type.
/// * `power` - Exponent, positive (`f64::INFINITY` for a hard mask). It is
///   rounded to the element type once.
/// * `split_zeros` - Value of the mask where both inputs are zero: `0.5` or `0`.
///
/// # Returns
///
/// An array of the same shape and element type with values in `[0, 1]`,
/// computed in that type.
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
/// let x = array![3.0_f64, 0.0, 1.0];
/// let r = array![4.0_f64, 0.0, 1.0];
/// let m = softmask(&x, &r, 2.0, true).unwrap();
/// assert!((m[0] - 9.0 / 25.0).abs() < 1e-15);
/// assert_eq!(m[1], 0.5);
/// assert_eq!(softmask(&x, &r, f64::INFINITY, false).unwrap(), array![0.0, 0.0, 0.0]);
/// ```
pub fn softmask<F, S1, S2, D>(
    x: &ArrayBase<S1, D>,
    x_ref: &ArrayBase<S2, D>,
    power: f64,
    split_zeros: bool,
) -> AudioSampleResult<Array<F, D>>
where
    F: AudioFloat,
    S1: Data<Elem = F>,
    S2: Data<Elem = F>,
    D: Dimension,
{
    if x.shape() != x_ref.shape() {
        return Err(config_err(
            "softmask",
            format!("shape mismatch: {:?} != {:?}", x.shape(), x_ref.shape()),
        ));
    }
    // Read each input in its memory order (NumPy spectrograms are often
    // Fortran-order).
    let bad = |a: ndarray::ArrayView<'_, F, D>| -> bool {
        let neg = |v: &F| v.is_nan() || *v < F::zero();
        match a.as_slice_memory_order() {
            Some(flat) => flat.iter().any(neg),
            None => a.iter().any(neg),
        }
    };
    if bad(x.view()) || bad(x_ref.view()) {
        return Err(param_err("x", "X and X_ref must be non-negative"));
    }
    if power.is_nan() || power <= 0.0 {
        return Err(param_err("power", "power must be strictly positive"));
    }
    let power_f = F::of(power);
    let zero_value = if split_zeros { F::of(0.5) } else { F::zero() };
    // The output takes `x`'s memory order so that the zip below walks all
    // three arrays contiguously for Fortran-order input too.
    let fortran = x.ndim() > 1 && !x.is_standard_layout() && x.t().is_standard_layout();
    let mut out = if fortran {
        Array::<F, D>::zeros(ndarray::ShapeBuilder::f(x.raw_dim()))
    } else {
        Array::<F, D>::zeros(x.raw_dim())
    };
    ndarray::Zip::from(&mut out)
        .and(x)
        .and(x_ref)
        .for_each(|m, &a, &b| {
            *m = if power.is_finite() {
                let z = a.max(b);
                if z < F::min_positive_value() {
                    zero_value
                } else {
                    let ma = pow_fast(a / z, power, power_f);
                    let mb = pow_fast(b / z, power, power_f);
                    ma / (ma + mb)
                }
            } else if a > b {
                F::one()
            } else {
                F::zero()
            };
        });
    Ok(out)
}

/// Median filter of every channel of `s` along `axis` (1 = frequency,
/// 2 = time) with kernel `k`, equivalent to `scipy.ndimage.median_filter` with
/// `mode="reflect"` (half-sample symmetric: `d c b a | a b c d | d c b a`) and
/// size 1 on the other axes. Even kernels select the upper median (SciPy's rank
/// `k / 2`), centred as SciPy does (`k / 2` samples before, `k / 2 - 1` after).
fn median_filter_axis<F: AudioFloat>(s: ArrayView3<'_, F>, axis: usize, k: usize) -> Array3<F> {
    let mut out = Array3::<F>::zeros(s.raw_dim());
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
    let mut line = vec![F::zero(); n];
    let mut buf = vec![F::zero(); k];
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
            let (_, m, _) = buf.select_nth_unstable_by(k / 2, total_cmp);
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
/// [`HarmonicPercussiveConfig`]. `Default` (and [`HpssMaskConfig::librosa`]) uses
/// librosa's defaults: kernel size 31 for both filters, power 2, margin 1.
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
    /// harmonic component. Default: 31, as librosa.
    pub kernel_harmonic: NonZeroUsize,
    /// Length (in bins) of the median filter along frequency that estimates the
    /// percussive component. Default: 31, as librosa.
    pub kernel_percussive: NonZeroUsize,
    /// Mask exponent ([`softmask`]); `f64::INFINITY` gives binary masks.
    /// Default: 2, as librosa.
    pub power: f64,
    /// Harmonic margin: the harmonic mask is `softmask(H, margin * P)`. Values above 1
    /// make the separation stricter and leave a residual. Default: 1, as librosa.
    pub margin_harmonic: f64,
    /// Percussive margin (as `margin_harmonic`, for `softmask(P, margin * H)`).
    /// Default: 1, as librosa.
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
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// librosa 1.0's defaults of `decompose.hpss`. Currently the same as
    /// [`Default`]; use this constructor where agreement with librosa is the
    /// requirement.
    #[must_use]
    pub fn librosa() -> Self {
        Self {
            kernel_harmonic: crate::nzu!(31),
            kernel_percussive: crate::nzu!(31),
            power: 2.0,
            margin_harmonic: 1.0,
            margin_percussive: 1.0,
        }
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
/// * `magnitude` - `(channels, n_bins, n_frames)` non-negative spectrogram
///   (`f32` or `f64`).
/// * `config` - Filter sizes, mask power and margins (rounded to the element
///   type once).
///
/// # Returns
///
/// `(mask_harmonic, mask_percussive)`, each the shape and element type of
/// `magnitude`, computed in that type. With margins of 1 and finite power they
/// sum to 1.
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
/// let s = Array3::from_shape_fn((1, 40, 50), |(_, k, _)| if k == 20 { 1.0_f64 } else { 0.0 });
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
pub fn hpss_masks<F: AudioFloat>(
    magnitude: ArrayView3<'_, F>,
    config: &HpssMaskConfig,
) -> AudioSampleResult<(Array3<F>, Array3<F>)> {
    config.validate()?;
    if magnitude.iter().any(|v| v.is_nan() || *v < F::zero()) {
        return Err(param_err(
            "magnitude",
            "the spectrogram must be non-negative",
        ));
    }
    let harm = median_filter_axis(magnitude, 2, config.kernel_harmonic.get());
    let perc = median_filter_axis(magnitude, 1, config.kernel_percussive.get());
    let split_zeros = config.margin_harmonic == 1.0 && config.margin_percussive == 1.0;
    let (margin_h, margin_p) = (
        F::of(config.margin_harmonic),
        F::of(config.margin_percussive),
    );
    let mh = softmask(
        &harm,
        &perc.mapv(|v| v * margin_h),
        config.power,
        split_zeros,
    )?;
    let mp = softmask(
        &perc,
        &harm.mapv(|v| v * margin_p),
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
/// * `magnitude` - `(channels, n_bins, n_frames)` non-negative spectrogram
///   (`f32` or `f64`).
/// * `config` - Filter sizes, mask power and margins.
///
/// # Returns
///
/// `(S * mask_harmonic, S * mask_percussive)` in the element type of `S`; see
/// [`hpss_masks`].
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
/// let s = Array3::from_shape_fn((1, 64, 64), |(_, k, t)| if k == 10 || t == 32 { 1.0_f64 } else { 0.0 });
/// let (h, p) = hpss_magnitude(s.view(), &HpssMaskConfig::default()).unwrap();
/// assert!(h[[0, 10, 5]] > 0.9 && p[[0, 40, 32]] > 0.9);
/// ```
pub fn hpss_magnitude<F: AudioFloat>(
    magnitude: ArrayView3<'_, F>,
    config: &HpssMaskConfig,
) -> AudioSampleResult<(Array3<F>, Array3<F>)> {
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
/// * `stft_matrix` - `(channels, n_bins, n_frames)` complex STFT
///   (`Complex<f32>` or `Complex<f64>`).
/// * `config` - Filter sizes, mask power and margins.
///
/// # Returns
///
/// `(D_harmonic, D_percussive)`, each the shape and element type of
/// `stft_matrix`, computed in that type.
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
pub fn hpss_spectrogram<F: AudioFloat>(
    stft_matrix: ArrayView3<'_, Complex<F>>,
    config: &HpssMaskConfig,
) -> AudioSampleResult<(Array3<Complex<F>>, Array3<Complex<F>>)> {
    let mag = stft_matrix.mapv(Complex::norm);
    let (mh, mp) = hpss_masks(mag.view(), config)?;
    let apply = |mask: &Array3<F>| {
        let mut out = Array3::<Complex<F>>::zeros(stft_matrix.raw_dim());
        ndarray::Zip::from(&mut out)
            .and(&stft_matrix)
            .and(&mag)
            .and(mask)
            .for_each(|o, d, &s, &m| {
                // librosa.magphase: phase = D / |D|, or 1 where |D| == 0.
                let phase = if s == F::zero() {
                    Complex::new(F::one(), d.im)
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
/// `Default` (and [`HarmonicPercussiveConfig::librosa`]) reproduces
/// `librosa.effects.hpss(y)`.
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
    /// STFT for analysis and resynthesis. Default: [`StftConfig::default`],
    /// librosa's STFT defaults.
    pub stft: StftConfig,
    /// Median-filter and mask parameters. Default: [`HpssMaskConfig::default`],
    /// librosa's defaults.
    pub hpss: HpssMaskConfig,
}

impl HarmonicPercussiveConfig {
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// librosa 1.0's defaults of `effects.hpss`. Currently the same as
    /// [`Default`]; use this constructor where agreement with librosa is the
    /// requirement.
    #[must_use]
    pub fn librosa() -> Self {
        Self {
            stft: StftConfig::librosa(),
            hpss: HpssMaskConfig::librosa(),
        }
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
fn hpss_rows<F: AudioFloat>(
    rows: ArrayView2<'_, F>,
    config: &HarmonicPercussiveConfig,
    parts: &[HpssPart],
) -> AudioSampleResult<Vec<Array2<F>>> {
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
///
/// The default is `Random(Some(0))`: random phases from a fixed seed, so the same
/// input gives the same output on every run. librosa's default draws a fresh
/// seed (`Random(None)`, used by [`GriffinLimConfig::librosa`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GriffinLimInit {
    /// Uniformly random phases (librosa `init="random"`). `Some(seed)` makes the
    /// result reproducible; `None` draws a fresh seed on every call.
    Random(Option<u64>),
    /// All phases zero (librosa `init=None`): fully deterministic.
    ZeroPhase,
}

impl Default for GriffinLimInit {
    fn default() -> Self {
        Self::Random(Some(0))
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
/// `Default` uses librosa's values (32 iterations, momentum 0.99, default STFT)
/// with a seeded random initial phase (`GriffinLimInit::Random(Some(0))`), so
/// results are reproducible. [`GriffinLimConfig::librosa`] draws a fresh seed on
/// every call, as librosa does. `stft.n_fft` must match the spectrogram
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
    /// Number of iterations. Default: 32, as librosa.
    pub n_iter: usize,
    /// Momentum of the fast Griffin-Lim update (0 gives the original algorithm).
    /// Values above 1 can be unstable. Default: 0.99, as librosa.
    pub momentum: f64,
    /// Initial phase. Default: `Random(Some(0))` (seeded); librosa:
    /// `Random(None)` (a fresh seed per call).
    pub init: GriffinLimInit,
    /// STFT parameters of the magnitude spectrogram. Default:
    /// [`StftConfig::default`], librosa's STFT defaults.
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
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// librosa 1.0's defaults of `griffinlim`: as [`Default`], but with
    /// `init = GriffinLimInit::Random(None)`, so every call draws a fresh seed.
    #[must_use]
    pub fn librosa() -> Self {
        Self {
            n_iter: 32,
            momentum: 0.99,
            init: GriffinLimInit::Random(None),
            stft: StftConfig::librosa(),
            length: None,
        }
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
///   spectrogram (`f32` or `f64`).
/// * `config` - Iterations, momentum, initial phase, STFT and output length.
///
/// # Returns
///
/// A `(channels, samples)` array of the element type of `magnitude`. Every
/// iteration runs in that type; random initial phases are drawn in `f64` and
/// rounded once.
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
pub fn griffinlim<F: AudioFloat>(
    magnitude: ArrayView3<'_, F>,
    config: &GriffinLimConfig,
) -> AudioSampleResult<Array2<F>> {
    config.validate()?;
    if magnitude
        .iter()
        .any(|v| !(*v >= F::zero() && v.is_finite()))
    {
        return Err(param_err(
            "magnitude",
            "the magnitude spectrogram must be finite and non-negative",
        ));
    }
    let mut angles: Array3<Complex<F>> = match config.init {
        GriffinLimInit::ZeroPhase => magnitude.mapv(|s| Complex::new(s, F::zero())),
        GriffinLimInit::Random(seed) => {
            use rand::rngs::StdRng;
            use rand::{RngExt, SeedableRng};
            let mut rng = StdRng::seed_from_u64(seed.unwrap_or_else(rand::random::<u64>));
            magnitude.mapv(|s| {
                let phi = 2.0 * std::f64::consts::PI * rng.random::<f64>();
                Complex::new(F::of(phi.cos()) * s, F::of(phi.sin()) * s)
            })
        }
    };
    let beta = F::of(config.momentum / (1.0 + config.momentum));
    let tiny = F::min_positive_value();
    let mut tprev: Option<Array3<Complex<F>>> = None;
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
                let d = a.norm() + tiny;
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
/// [`recurrence_matrix_weighted`]); a frame with no neighbours is copied unchanged.
///
/// # Arguments
///
/// * `data` - `(n_features, n_frames)` features (e.g. a chromagram or magnitude
///   spectrogram), `f32` or `f64`.
/// * `rec` - `(n_frames, n_frames)` recurrence matrix (connectivity or affinity)
///   of the same element type.
/// * `aggregate` - Mean, median or recurrence-weighted average.
///
/// # Returns
///
/// The filtered `(n_features, n_frames)` array, computed in the element type of
/// `data`.
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
/// let data = array![[1.0_f64, 2.0, 3.0]];
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
pub fn nn_filter<F: AudioFloat>(
    data: ArrayView2<'_, F>,
    rec: ArrayView2<'_, F>,
    aggregate: NnAggregate,
) -> AudioSampleResult<Array2<F>> {
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
        let targets: Vec<usize> = (0..n_frames)
            .filter(|&j| rec[[j, i]] != F::zero())
            .collect();
        if targets.is_empty() {
            continue;
        }
        let weight_sum: F = targets.iter().map(|&j| rec[[j, i]]).sum();
        if aggregate == NnAggregate::WeightedAverage && weight_sum == F::zero() {
            return Err(param_err(
                "rec",
                format!("the recurrence weights of frame {i} sum to zero"),
            ));
        }
        for f in 0..n_features {
            out[[f, i]] = match aggregate {
                NnAggregate::Mean => {
                    targets.iter().map(|&j| data[[f, j]]).sum::<F>() / F::of_usize(targets.len())
                }
                NnAggregate::WeightedAverage => {
                    targets
                        .iter()
                        .map(|&j| data[[f, j]] * rec[[j, i]])
                        .sum::<F>()
                        / weight_sum
                }
                NnAggregate::Median => {
                    values.clear();
                    values.extend(targets.iter().map(|&j| data[[f, j]]));
                    values.sort_by(total_cmp);
                    let m = values.len();
                    if m % 2 == 1 {
                        values[m / 2]
                    } else {
                        (values[m / 2 - 1] + values[m / 2]) / F::of(2.0)
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
/// * `data` - `(n_features, n_frames)` features, `f32` or `f64`.
/// * `recurrence` - Neighbourhood parameters; [`RecurrenceConfig::default`] is
///   librosa's default (connectivity, `k = 2 ceil(sqrt(t - 2 width + 1))`, width 1).
/// * `aggregate` - Mean, median or weighted average (use an affinity mode for
///   meaningful weights).
///
/// # Returns
///
/// The filtered `(n_features, n_frames)` array in the element type of `data`.
/// The recurrence matrix is currently computed in `f64` and rounded to that type;
/// the filtering runs in it.
///
/// # Errors
///
/// As [`recurrence_matrix_weighted`] and [`nn_filter`].
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
pub fn nn_filter_recurrence<F: AudioFloat>(
    data: ArrayView2<'_, F>,
    recurrence: &RecurrenceConfig,
    aggregate: NnAggregate,
) -> AudioSampleResult<Array2<F>> {
    let rec = recurrence_matrix_weighted(data, recurrence)?;
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
/// * `y` - Signal (`f32` or `f64`), finite, with more than `order` samples.
/// * `order` - Prediction order.
///
/// # Returns
///
/// `order + 1` coefficients `[1, a_1, ..., a_order]` of the prediction-error
/// filter `A(z) = 1 + a_1 z^-1 + ... + a_order z^-order`, in the element type of
/// `y`. The recursion runs in `f64` (see below).
///
/// # Errors
///
/// [`ParameterError`] if `y` has a non-finite value or no more than `order` samples.
///
/// # Differences from librosa
///
/// The recursion always runs in `f64`; `f32` input is widened (exactly) and the
/// coefficients are rounded to `f32` once, at the cost of one `f64` copy of the
/// signal. librosa runs the recursion in the input's dtype, and its `float32`
/// Burg recursion is unstable on tonal input at high orders: on a test signal
/// of three tones and a chirp its `float32` coefficients differ from its
/// `float64` ones by 1.3e-3 (relative to the largest coefficient) at order 4,
/// 0.14 at order 8 and by more than 100 % from order 10 on. For such input this
/// function deliberately does not reproduce librosa's `float32` result.
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
pub fn lpc<F: AudioFloat>(y: &[F], order: NonZeroUsize) -> AudioSampleResult<Vec<F>> {
    Ok(lpc_f64(y, order)?.into_iter().map(F::of).collect())
}

/// Burg's recursion in `f64` on samples of type `F` (widened as they are
/// read).
///
/// The forward and backward prediction errors live in two buffers updated in
/// place: at iteration `i` the forward errors are `fwd[i..i + m]` and the
/// backward errors `bwd[..m]` (`m = n - 1 - i`), which is the same pairing as
/// dropping the first forward and the last backward error after every
/// update. The first update reads the samples directly, and the dot product
/// of the next iteration is accumulated in the update loop, in the same
/// order as a separate pass.
fn lpc_f64<F: AudioFloat>(y: &[F], order: NonZeroUsize) -> AudioSampleResult<Vec<f64>> {
    let order = order.get();
    if !y.iter().fold(true, |ok, v| ok & v.is_finite()) {
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
    let m0 = y.len() - 1;
    let (minus_two, tiny) = (-2.0, f64::MIN_POSITIVE);
    // Initial errors: fwd = y[1..], bwd = y[..n - 1].
    let mut den: f64 = y[1..]
        .iter()
        .zip(&y[..m0])
        .map(|(&f, &b)| {
            let (f, b) = (f.as_f64(), b.as_f64());
            f * f + b * b
        })
        .sum();
    let mut dot: f64 = y[..m0]
        .iter()
        .zip(&y[1..])
        .map(|(&b, &f)| b.as_f64() * f.as_f64())
        .sum();
    let mut fwd = vec![0.0_f64; m0];
    let mut bwd = vec![0.0_f64; m0];
    for i in 0..order {
        let k = dot * minus_two / (den + tiny);
        std::mem::swap(&mut ar, &mut ar_prev);
        for j in 1..=i + 1 {
            ar[j] = ar_prev[j] + k * ar_prev[i + 1 - j];
        }
        let m = m0 - i;
        let fw = &mut fwd[i..i + m];
        let bw = &mut bwd[..m];
        // Update both error sequences; accumulate the next iteration's dot
        // product sum_t bwd'[t] * fwd'[t + 1] as soon as both factors exist.
        let mut next_dot = -0.0_f64;
        if i == 0 {
            for t in 0..m {
                let (f, b) = (y[t + 1].as_f64(), y[t].as_f64());
                fw[t] = f + k * b;
                bw[t] = b + k * f;
            }
            for t in 1..m {
                next_dot += bw[t - 1] * fw[t];
            }
        } else {
            let (f, b) = (fw[0], bw[0]);
            fw[0] = f + k * b;
            bw[0] = b + k * f;
            for t in 1..m {
                let (f, b) = (fw[t], bw[t]);
                let f_new = f + k * b;
                fw[t] = f_new;
                bw[t] = b + k * f;
                next_dot += bw[t - 1] * f_new;
            }
        }
        let q = 1.0 - k * k;
        let (b_last, f_first) = (bw[m - 1], fw[0]);
        den = q * den - b_last * b_last - f_first * f_first;
        dot = next_dot;
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
    Frames {
        /// The frame indices.
        frames: &'a [usize],
        /// Samples per frame: the hop of the analysis that produced `frames`
        /// (librosa's `hop_length`, default 512).
        hop_length: NonZeroUsize,
    },
}

/// Configuration of [`clicks`], mirroring `librosa.clicks`'s keyword arguments.
///
/// ## Purpose
///
/// Sets the click waveform (a decaying sine burst unless `click` is given) and the
/// output length. The hop of frame positions is part of
/// [`ClickPositions::Frames`].
///
/// ## Intended usage
///
/// `Default` (and [`ClicksConfig::librosa`]) uses librosa's defaults: a 1 kHz,
/// 100 ms click, length up to the end of the last click.
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
    /// Frequency (Hz) of the default click. Default: 1000, as librosa.
    pub click_freq: f64,
    /// Duration (s) of the default click. Default: 0.1, as librosa.
    pub click_duration: f64,
    /// Custom click waveform used instead of the default one. It is rounded to
    /// the output's float type once.
    pub click: Option<Vec<f64>>,
    /// Output length in samples; clicks starting at or after it are dropped.
    /// `None`: up to the end of the last click.
    pub length: Option<usize>,
}

impl Default for ClicksConfig {
    fn default() -> Self {
        Self {
            click_freq: 1000.0,
            click_duration: 0.1,
            click: None,
            length: None,
        }
    }
}

impl ClicksConfig {
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// librosa 1.0's defaults of `clicks`. Currently the same as [`Default`]; use
    /// this constructor where agreement with librosa is the requirement.
    #[must_use]
    pub fn librosa() -> Self {
        Self {
            click_freq: 1000.0,
            click_duration: 0.1,
            click: None,
            length: None,
        }
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
/// The output sample type `T` is chosen by the caller, as for
/// [`sine_wave`](crate::sine_wave). The default click is computed in `f64`
/// and rounded to `T`'s float type ([`AudioSample::Float`](crate::AudioSample::Float))
/// once; the clicks are summed in that type and then converted to `T` with the
/// crate's audio-aware conversion (integer types round and saturate).
///
/// # Arguments
///
/// * `positions` - Click times (seconds), or frame indices with their hop.
/// * `sample_rate` - Output sample rate.
/// * `config` - Click waveform and length.
///
/// # Returns
///
/// Mono audio of sample type `T`.
///
/// # Errors
///
/// [`ParameterError`] if the configuration is invalid, a time is negative or not
/// finite, or `positions` is empty while `length` is `None`, or the track or the
/// click would exceed 2^31 samples; an empty-data error if the default click
/// would have no samples.
///
/// # Differences from librosa
///
/// librosa returns `float32` by default (`dtype=numpy.float32`); `clicks::<f32>`
/// is the equivalent here. Multichannel custom clicks are not supported.
///
/// # Examples
///
/// ```rust
/// use audio_samples::nzu;
/// use audio_samples::operations::effects::{ClickPositions, ClicksConfig, clicks};
/// use audio_samples::sample_rate;
///
/// let y = clicks::<f32>(
///     ClickPositions::Times(&[0.0, 0.5]),
///     sample_rate!(22050),
///     &ClicksConfig::default(),
/// )
/// .unwrap();
/// // Last click starts at 11025 and lasts 2205 samples.
/// assert_eq!(y.samples_per_channel().get(), 11025 + 2205);
///
/// // Frame positions carry their hop: frames 0 and 10 at hop 256.
/// let positions = ClickPositions::Frames { frames: &[0, 10], hop_length: nzu!(256) };
/// let y = clicks::<i16>(positions, sample_rate!(22050), &ClicksConfig::default()).unwrap();
/// assert_eq!(y.samples_per_channel().get(), 2560 + 2205);
/// ```
pub fn clicks<T: StandardSample>(
    positions: ClickPositions<'_>,
    sample_rate: SampleRate,
    config: &ClicksConfig,
) -> AudioSampleResult<AudioSamples<'static, T>> {
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
                v.push(check_derived_len(
                    "times",
                    "a click track",
                    (t * sr).trunc(),
                    MAX_SIGNAL_ELEMENTS,
                )?);
            }
            v
        }
        ClickPositions::Frames { frames, hop_length } => frames
            .iter()
            .map(|&f| {
                check_derived_len(
                    "frames",
                    "a click track",
                    f as f64 * hop_length.get() as f64,
                    MAX_SIGNAL_ELEMENTS,
                )
            })
            .collect::<AudioSampleResult<_>>()?,
    };
    let click: Vec<T::Float> = match &config.click {
        Some(c) => c.iter().map(|&v| T::Float::of(v)).collect(),
        None => {
            let n = check_derived_len(
                "click_duration",
                "the click",
                (sr * config.click_duration).trunc(),
                MAX_SIGNAL_ELEMENTS,
            )?;
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
                    T::Float::of(2.0f64.powf(e) * (w * i as f64).sin())
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
            max.saturating_add(click.len())
        }
    };
    check_derived_len(
        "length",
        "a click track",
        length as f64,
        MAX_SIGNAL_ELEMENTS,
    )?;
    let mut out = vec![T::Float::of(0.0); length];
    // Positions are < length here (filtered above, or length = max + click
    // length); the guard and saturating add keep the slice in bounds anyway.
    for &start in pos.iter().filter(|&&p| p < length) {
        let end = start.saturating_add(click.len()).min(length);
        for (o, &c) in out[start..end].iter_mut().zip(&click) {
            *o += c;
        }
    }
    let rows = Array2::from_shape_vec((1, length), out)
        .map_err(|e| AudioSampleError::Layout(crate::LayoutError::shape_error("clicks", e)))?;
    float_rows_to_audio::<T>(rows, sample_rate, "clicks")
}

// =============================================================================
// The AudioEffects trait
// =============================================================================

/// librosa-compatible audio effects on [`AudioSamples`].
///
/// All methods process every channel (mono gives one channel) in the sample
/// type's float type [`Float`](Self::Float) (`f32` for `u8`, `i16`, `I24` and
/// `f32` audio; `f64` for `i32` and `f64` audio) and return audio of the input's
/// sample type and sample rate, converted with the crate's audio-aware conversion
/// (integer types round and saturate). Each audio-to-audio method has an
/// `_in_place` variant that replaces the audio (its length may change). See the
/// [module documentation](self) for the librosa correspondence.
pub trait AudioEffects {
    /// The float type of the computation and of the array outputs
    /// ([`lpc`](Self::lpc)): the sample type's
    /// [`AudioSample::Float`](crate::AudioSample::Float).
    type Float: AudioFloat;

    /// The sample type of the audio. The audio-to-audio methods and
    /// [`griffinlim_audio`](Self::griffinlim_audio) return audio of this type.
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
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
    ///
    /// [`ParameterError`] if `rate` is not finite and positive, the result would be
    /// empty or exceed 2^31 samples, or the STFT configuration is invalid.
    ///
    /// # Differences from librosa
    ///
    /// Same algorithm; on ordinary signals the output matches librosa to
    /// floating-point round-off. Where the input contains exact digital silence
    /// (e.g. a zero lead-in), STFT bins of ~0 magnitude have ill-conditioned
    /// phases that the phase vocoder accumulates, so round-off is amplified:
    /// agreement drops to ~30 dB SNR there. librosa against itself with a
    /// 1e-15 relative STFT perturbation differs by the same amount (~27 dB).
    ///
    /// # Precision and memory
    ///
    /// For `f32` computation (`u8`, `i16`, `I24`, `f32` audio) the analysis STFT
    /// is computed in `f64` from the samples converted to `f64` and rounded once
    /// to `Complex<f32>`; the phase vocoder and the inverse STFT run in `f32`.
    /// librosa's `float32` path also computes its STFT in `float64`. An `f32`
    /// STFT would put the bins that a signal reaches only later (a chirp, an
    /// onset) at the `f32` round-off level, and the phase vocoder carries their
    /// phase into the output. On the reference chirp-and-tones signal the `f32`
    /// result agrees with librosa's `float32` result to 1.9e-7 of the peak and
    /// with this crate's `f64` result to 2.1e-6; `i16` output is within 1 LSB of
    /// the `f64` result re-quantised.
    ///
    /// The cost: while the STFT is computed, an `f64` copy of the samples and the
    /// `f64` complex STFT (twice the size of the `f32` one) are held; the `f64`
    /// STFT is freed once it is rounded.
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
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
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
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
    ///
    /// [`ParameterError`] if `n_steps` is not finite, the configuration is invalid,
    /// or the shift is so large that the stretched signal is empty.
    ///
    /// # Differences from librosa
    ///
    /// librosa resamples with soxr (`res_type="soxr_hq"`); this method uses the
    /// crate's windowed-sinc resampler ([`ResamplingQuality::High`] by default),
    /// which approximates the arbitrary ratio `rate` by a ratio of two integers near
    /// 2^31. The time-stretch stage matches librosa to round-off (amplified on
    /// input containing exact digital silence, see
    /// [`time_stretch`](Self::time_stretch), including the `f32` case); the final
    /// signal agrees within the resampler difference (about 68 dB SNR on the
    /// reference test signal, see `tests/librosa_reference/effects.rs`).
    ///
    /// # Precision and memory
    ///
    /// The time-stretch stage computes its STFT in `f64` and rounds it once, with
    /// the transient `f64` copies described under
    /// [`time_stretch`](Self::time_stretch). The resampling stage filters the
    /// time-stretched rows in the float type and writes the output (already cut
    /// or padded to the input length) directly. `i16` output is within 1 LSB of
    /// the `f64` result re-quantised on the reference test signals.
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
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
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
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
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
    ///
    /// // 16-bit audio is filtered in f32 and returned as i16.
    /// let pcm = AudioSamples::new_mono(array![1000_i16, 2000, 3000], sample_rate!(8000)).unwrap();
    /// let y: AudioSamples<'static, i16> = pcm.preemphasis(0.5).unwrap();
    /// assert_eq!(y.as_mono().unwrap()[2], 3000 - 1000);
    /// ```
    fn preemphasis(&self, coef: f64) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// In-place [`preemphasis`](Self::preemphasis).
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
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
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
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
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
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
    ///   ([`SilenceConfig::librosa`] = librosa's defaults).
    ///
    /// # Returns
    ///
    /// `(trimmed, (start, end))`: the trimmed audio (all channels) and the kept
    /// sample range `start..end` of the input.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
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
    /// (frame RMS summation order).
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
    /// let (trimmed, (start, end)) = audio.trim_db(&SilenceConfig::librosa()).unwrap();
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
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
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
    /// let (start, end) = audio.trim_db_in_place(&SilenceConfig::new(60.0)).unwrap();
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
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
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
    /// let parts = audio.split_db(&SilenceConfig::librosa()).unwrap();
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
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
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
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
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
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
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
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
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
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
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
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
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
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
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
    /// A `(channels, order + 1)` array of [`Float`](Self::Float); row `c` is
    /// `[1, a_1, ..., a_order]`. The recursion runs in `f64` for every sample
    /// type (see [`lpc`]).
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if the audio contains a NaN or infinite sample (as
    /// librosa's "Audio buffer is not finite everywhere").
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
    fn lpc(&self, order: NonZeroUsize) -> AudioSampleResult<Array2<Self::Float>>;

    /// Reconstructs audio from a magnitude spectrogram with Griffin-Lim, equivalent
    /// to `librosa.griffinlim(S)`; see [`griffinlim`].
    ///
    /// # Arguments
    ///
    /// * `magnitude` - `(channels, n_fft / 2 + 1, n_frames)` magnitude spectrogram
    ///   of [`Float`](Self::Float), as produced by
    ///   [`AudioSpectrum::spectrogram_matrix`](crate::operations::spectrum::AudioSpectrum::spectrogram_matrix)
    ///   on audio of the same sample type.
    /// * `config` - Iterations, momentum, initial phase, STFT, length.
    /// * `sample_rate` - Sample rate of the returned audio.
    ///
    /// # Returns
    ///
    /// Audio of sample type [`Sample`](Self::Sample): mono for one channel,
    /// multi-channel otherwise. The reconstruction runs in
    /// [`Float`](Self::Float) and is converted with the crate's audio-aware
    /// conversion (integer types round and saturate).
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
    ///
    /// // 16-bit audio: an f32 magnitude in, i16 audio out.
    /// let pcm = sine_wave::<i16>(440.0, Duration::from_millis(300), sample_rate!(22050), 0.5);
    /// let mag = pcm.spectrogram_matrix(&StftConfig::default(), 1.0).unwrap();
    /// let y: AudioSamples<'static, i16> =
    ///     AudioSamples::<i16>::griffinlim_audio(&mag, &cfg, sample_rate!(22050)).unwrap();
    /// assert_eq!(y.num_channels().get(), 1);
    /// ```
    fn griffinlim_audio(
        magnitude: &Array3<Self::Float>,
        config: &GriffinLimConfig,
        sample_rate: SampleRate,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>
    where
        Self: Sized;
}

impl<T> AudioEffects for AudioSamples<'_, T>
where
    T: StandardSample,
{
    type Float = T::Float;
    type Sample = T;

    fn time_stretch(
        &self,
        rate: f64,
        stft: &StftConfig,
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        let y = time_stretch_rows::<T::Float>(finite_f64_rows(self)?.view(), rate, stft)?;
        float_rows_to_audio(y, self.sample_rate(), "time_stretch")
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
        let y = pitch_shift_rows::<T::Float>(finite_f64_rows(self)?.view(), n_steps, config)?;
        float_rows_to_audio(y, self.sample_rate(), "pitch_shift")
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
        let rows = finite_float_rows(self)?;
        let y = filter_rows(rows.view(), |x, out| {
            preemphasis_into(x, coef, None, out).map(|_| ())
        })?;
        float_rows_to_audio(y, self.sample_rate(), "preemphasis")
    }

    fn preemphasis_in_place(&mut self, coef: f64) -> AudioSampleResult<()> {
        *self = self.preemphasis(coef)?.into_owned();
        Ok(())
    }

    fn deemphasis(&self, coef: f64) -> AudioSampleResult<AudioSamples<'static, T>> {
        let rows = finite_float_rows(self)?;
        let y = filter_rows(rows.view(), |x, out| {
            deemphasis_into(x, coef, None, out).map(|_| ())
        })?;
        float_rows_to_audio(y, self.sample_rate(), "deemphasis")
    }

    fn deemphasis_in_place(&mut self, coef: f64) -> AudioSampleResult<()> {
        *self = self.deemphasis(coef)?.into_owned();
        Ok(())
    }

    fn trim_db(
        &self,
        config: &SilenceConfig,
    ) -> AudioSampleResult<(AudioSamples<'static, T>, (usize, usize))> {
        let rows = finite_float_rows(self)?;
        let flags = nonsilent_frames(rows.view(), config)?;
        let (start, end) = trim_bounds(&flags, config.hop_length.get(), rows.ncols());
        if start >= end {
            return Err(AudioSampleError::empty_data("trim_db"));
        }
        let kept = rows.slice(ndarray::s![.., start..end]).to_owned();
        Ok((
            float_rows_to_audio(kept, self.sample_rate(), "trim_db")?,
            (start, end),
        ))
    }

    fn trim_db_in_place(&mut self, config: &SilenceConfig) -> AudioSampleResult<(usize, usize)> {
        let (audio, bounds) = self.trim_db(config)?;
        *self = audio.into_owned();
        Ok(bounds)
    }

    fn split_db(&self, config: &SilenceConfig) -> AudioSampleResult<Vec<(usize, usize)>> {
        let rows = finite_float_rows(self)?;
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
        let y = remix_rows(finite_float_rows(self)?.view(), intervals, align_zeros)?;
        float_rows_to_audio(y, self.sample_rate(), "remix")
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
        let mut parts = hpss_rows(
            finite_float_rows(self)?.view(),
            config,
            &[HpssPart::Harmonic],
        )?;
        float_rows_to_audio(parts.remove(0), self.sample_rate(), "harmonic")
    }

    fn harmonic_in_place(&mut self, config: &HarmonicPercussiveConfig) -> AudioSampleResult<()> {
        *self = self.harmonic(config)?.into_owned();
        Ok(())
    }

    fn percussive(
        &self,
        config: &HarmonicPercussiveConfig,
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        let mut parts = hpss_rows(
            finite_float_rows(self)?.view(),
            config,
            &[HpssPart::Percussive],
        )?;
        float_rows_to_audio(parts.remove(0), self.sample_rate(), "percussive")
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
            finite_float_rows(self)?.view(),
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
            float_rows_to_audio(h, self.sample_rate(), "harmonic_percussive")?,
            float_rows_to_audio(p, self.sample_rate(), "harmonic_percussive")?,
        ))
    }

    fn lpc(&self, order: NonZeroUsize) -> AudioSampleResult<Array2<T::Float>> {
        let rows = finite_float_rows(self)?;
        let mut out = Array2::<T::Float>::zeros((rows.nrows(), order.get() + 1));
        let rows = rows.as_standard_layout();
        for (c, row) in rows.outer_iter().enumerate() {
            let a = lpc(
                row.as_slice().expect("rows of a standard-layout array"),
                order,
            )?;
            out.row_mut(c).assign(&Array1::from(a));
        }
        Ok(out)
    }

    fn griffinlim_audio(
        magnitude: &Array3<T::Float>,
        config: &GriffinLimConfig,
        sample_rate: SampleRate,
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        let y = griffinlim(magnitude.view(), config)?;
        float_rows_to_audio(y, sample_rate, "griffinlim")
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

    /// The direct recursion and correction pass of `deemphasis`.
    fn deemphasis_reference<F: AudioFloat>(x: &[F], coef: f64) -> Vec<F> {
        let c = (F::of(2.0 - coef) * x[0] - x[1]) / F::of(3.0 - coef);
        let coef = F::of(coef);
        let mut y = Vec::with_capacity(x.len());
        let mut z = F::zero();
        for &v in x {
            let out = v + z;
            y.push(out);
            z = coef * out;
        }
        let mut p = F::one();
        for v in &mut y {
            *v -= c * p;
            p *= coef;
        }
        y
    }

    #[test]
    fn deemphasis_shortcuts_are_exact_through_subnormal_decay() {
        // A burst followed by a long digital silence: the recursion decays
        // into subnormal values and the correction term underflows.
        let mut x32 = vec![0.0_f32; 40_000];
        let mut x64 = vec![0.0_f64; 40_000];
        for (i, v) in noise(500, 4).iter().enumerate() {
            x32[i + 100] = *v as f32;
            x64[i + 100] = *v;
        }
        x32[0] = 0.25;
        x64[0] = 0.25;
        for coef in [0.97, 0.5, -0.9, 0.0] {
            let (y, _) = deemphasis_with_state(&x32, coef, None).unwrap();
            let want = deemphasis_reference(&x32, coef);
            assert!(
                y.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
                "f32 {coef}"
            );
            let (y, _) = deemphasis_with_state(&x64, coef, None).unwrap();
            let want = deemphasis_reference(&x64, coef);
            assert!(
                y.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
                "f64 {coef}"
            );
        }
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
        let d = Array3::from_elem((1, 3, 4), Complex::new(1.0_f64, 0.0));
        assert!(phase_vocoder(d.view(), 0.0).is_err());
        assert!(phase_vocoder(d.view(), f64::NAN).is_err());
        assert!(phase_vocoder_at(d.view(), &[4.0]).is_err());
        assert!(phase_vocoder_at(d.view(), &[-0.1]).is_err());
        assert!(phase_vocoder_at(d.view(), &[]).is_err());
        // A single frame is handled (librosa's interp1d cannot).
        let one = Array3::from_elem((1, 3, 1), Complex::new(0.0_f64, 2.0));
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
        let y = resample_rows_by_ratio(rows.view(), r, ResamplingQuality::High, None).unwrap();
        assert!((y.ncols() as f64 - 1000.0 * r).abs() <= 1.0);
    }

    #[test]
    fn preemphasis_state_matches_lfilter() {
        // Values from scipy / librosa (see the module tests in the reference suite).
        let (y, zf) = preemphasis_with_state(&[1.0_f64, 3.0, 0.5], 0.97, None).unwrap();
        assert_eq!(y[0], 0.0);
        assert!((y[1] - 2.03).abs() < 1e-15);
        assert!((y[2] + 2.41).abs() < 1e-15);
        assert!((zf + 0.485).abs() < 1e-15);
        let (y, zf) = deemphasis_with_state(&[1.0_f64, 3.0, 0.5], 0.97, Some(0.5)).unwrap();
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
        let cfg = SilenceConfig::librosa();
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
        let quiet = SilenceConfig::librosa().with_reference(DbReference::Value(1e6));
        assert!(audio.split_db(&quiet).unwrap().is_empty());
        assert!(audio.trim_db(&quiet).is_err());
        assert!(audio.split_db(&SilenceConfig::new(f64::NAN)).is_err());
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
        let y = clicks::<f64>(
            ClickPositions::Frames {
                frames: &[0, 2, 4, 5],
                hop_length: crate::nzu!(2),
            },
            sample_rate!(100),
            &cfg,
        )
        .unwrap();
        // Positions 0, 4, 8 (10 dropped); the click at 8 is truncated after 2 samples.
        assert_eq!(
            y.as_mono().unwrap().to_vec(),
            vec![1.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.5]
        );
        let y = clicks::<f64>(
            ClickPositions::Times(&[0.0, 0.01]),
            sample_rate!(100),
            &cfg.with_length(None),
        )
        .unwrap();
        assert_eq!(y.as_mono().unwrap().to_vec(), vec![1.0, 1.5, 0.5]);
        let d = clicks::<f64>(
            ClickPositions::Times(&[0.0]),
            sample_rate!(22050),
            &ClicksConfig::default(),
        )
        .unwrap();
        let d = d.as_mono().unwrap();
        assert_eq!(d.len().get(), 2205);
        assert_eq!(d[0], 0.0);
        assert!(
            clicks::<f64>(
                ClickPositions::Times(&[]),
                sample_rate!(100),
                &ClicksConfig::default()
            )
            .is_err()
        );
        assert!(
            clicks::<f64>(
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

    /// 16-bit test audio: two steady tones and a little noise after a
    /// digital-silence lead-in of 2000 samples.
    fn pcm16() -> AudioSamples<'static, i16> {
        let nz = noise(12000, 21);
        AudioSamples::new_mono(
            Array1::from_shape_fn(12000, |n| {
                if n < 2000 {
                    return 0;
                }
                let t = n as f64 / 16000.0;
                let v = 0.4 * (2.0 * std::f64::consts::PI * 330.0 * t).sin()
                    + 0.2 * (2.0 * std::f64::consts::PI * 1250.0 * t).sin()
                    + 0.01 * nz[n];
                (v * 32768.0).round() as i16
            }),
            sample_rate!(16000),
        )
        .unwrap()
    }

    /// Largest difference in LSB between the i16 result (computed in f32) and
    /// the f64 computation re-quantised to i16.
    fn max_lsb(
        x: &AudioSamples<'static, i16>,
        op: impl Fn(&AudioSamples<'static, i16>) -> AudioSamples<'static, i16>,
        op64: impl Fn(&AudioSamples<'static, f64>) -> AudioSamples<'static, f64>,
    ) -> i32 {
        use crate::AudioTypeConversion;
        let y = op(x);
        let y64 = op64(&x.to_format::<f64>()).to_format::<i16>();
        let (a, b) = (y.as_mono().unwrap(), y64.as_mono().unwrap());
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(&p, &q)| (i32::from(p) - i32::from(q)).abs())
            .max()
            .unwrap()
    }

    #[test]
    fn i16_effects_return_i16_close_to_the_f64_path() {
        let x = pcm16();
        let stft = StftConfig::default();
        let ps = PitchShiftConfig::default();
        let hp = HarmonicPercussiveConfig::default();
        // Measured: 1 LSB for every operation except trim_db (0: it copies
        // samples). The f32 and f64 results differ by round-off well below one
        // LSB (3.1e-5); a 1-LSB difference appears where the two values fall on
        // opposite sides of a rounding boundary.
        let cases: [(&str, i32, i32); 5] = [
            (
                "time_stretch",
                max_lsb(
                    &x,
                    |a| a.time_stretch(1.25, &stft).unwrap(),
                    |a| a.time_stretch(1.25, &stft).unwrap(),
                ),
                1,
            ),
            (
                "pitch_shift",
                max_lsb(
                    &x,
                    |a| a.pitch_shift(2.0, &ps).unwrap(),
                    |a| a.pitch_shift(2.0, &ps).unwrap(),
                ),
                1,
            ),
            (
                "preemphasis",
                max_lsb(
                    &x,
                    |a| a.preemphasis(0.97).unwrap(),
                    |a| a.preemphasis(0.97).unwrap(),
                ),
                1,
            ),
            (
                "harmonic",
                max_lsb(
                    &x,
                    |a| a.harmonic(&hp).unwrap(),
                    |a| a.harmonic(&hp).unwrap(),
                ),
                1,
            ),
            (
                "trim_db",
                max_lsb(
                    &x,
                    |a| a.trim_db(&SilenceConfig::librosa()).unwrap().0,
                    |a| a.trim_db(&SilenceConfig::librosa()).unwrap().0,
                ),
                0,
            ),
        ];
        for (name, lsb, bound) in cases {
            eprintln!("i16 {name}: max {lsb} LSB from the re-quantised f64 path (bound {bound})");
            assert!(lsb <= bound, "i16 {name}: {lsb} LSB > {bound}");
        }
        // A chirp sweeps into bins whose STFT values are at the round-off level
        // before it arrives, and the phase vocoder (librosa 1.0's algorithm)
        // carries the phase of those bins forward. The vocoder operations
        // therefore analyse the exact f64 samples with an f64 STFT and round it
        // to f32 once (as librosa's float32 path does); an f32 STFT of the f32
        // samples gave 139 LSB here. Measured: 1 LSB for time_stretch (rates 1.5
        // and 0.75), pitch_shift (+2, -3) and harmonic. Bound: 1 LSB.
        let chirp = AudioSamples::new_mono(
            Array1::from_shape_fn(22050, |n| {
                let t = n as f64 / 22050.0;
                let v = 0.3 * (2.0 * std::f64::consts::PI * 220.0 * t).sin()
                    + 0.15
                        * (2.0 * std::f64::consts::PI * (100.0 * t + 0.5 * 3900.0 * t * t)).sin();
                (v * 32768.0).round() as i16
            }),
            sample_rate!(22050),
        )
        .unwrap();
        let chirp_cases = [
            (
                "time_stretch 1.5",
                max_lsb(
                    &chirp,
                    |a| a.time_stretch(1.5, &stft).unwrap(),
                    |a| a.time_stretch(1.5, &stft).unwrap(),
                ),
            ),
            (
                "time_stretch 0.75",
                max_lsb(
                    &chirp,
                    |a| a.time_stretch(0.75, &stft).unwrap(),
                    |a| a.time_stretch(0.75, &stft).unwrap(),
                ),
            ),
            (
                "pitch_shift +2",
                max_lsb(
                    &chirp,
                    |a| a.pitch_shift(2.0, &ps).unwrap(),
                    |a| a.pitch_shift(2.0, &ps).unwrap(),
                ),
            ),
            (
                "pitch_shift -3",
                max_lsb(
                    &chirp,
                    |a| a.pitch_shift(-3.0, &ps).unwrap(),
                    |a| a.pitch_shift(-3.0, &ps).unwrap(),
                ),
            ),
            (
                "harmonic",
                max_lsb(
                    &chirp,
                    |a| a.harmonic(&hp).unwrap(),
                    |a| a.harmonic(&hp).unwrap(),
                ),
            ),
        ];
        for (name, lsb) in chirp_cases {
            eprintln!("i16 chirp {name}: max {lsb} LSB from the re-quantised f64 path (bound 1)");
            assert!(lsb <= 1, "i16 chirp {name}: {lsb} LSB > 1");
        }

        // The output type is the input type.
        let _: AudioSamples<'static, i16> = x.percussive(&hp).unwrap();
        let _: AudioSamples<'static, i16> = x.deemphasis(0.97).unwrap();
        let _: AudioSamples<'static, i16> = x.remix(&[(3000, 5000)], true).unwrap();
    }

    /// Regression: extreme click times / durations and stretch rates sized
    /// their outputs at up to 10^17 bytes and aborted on allocation.
    #[test]
    fn extreme_sizes_are_parameter_errors() {
        let sr = sample_rate!(22050);
        let cfg = ClicksConfig::default();
        let is_param = |r: AudioSampleResult<AudioSamples<'static, f64>>| {
            matches!(r, Err(AudioSampleError::Parameter(_)))
        };
        assert!(is_param(clicks(ClickPositions::Times(&[1e12]), sr, &cfg)));
        assert!(is_param(clicks(ClickPositions::Times(&[1e20]), sr, &cfg)));
        assert!(is_param(clicks(
            ClickPositions::Frames {
                frames: &[usize::MAX],
                hop_length: crate::nzu!(512),
            },
            sr,
            &cfg
        )));
        let long = ClicksConfig {
            click_duration: 1e12,
            ..ClicksConfig::default()
        };
        assert!(is_param(clicks(ClickPositions::Times(&[0.0]), sr, &long)));
        let fixed = ClicksConfig {
            length: Some(usize::MAX),
            ..ClicksConfig::default()
        };
        assert!(is_param(clicks(ClickPositions::Times(&[0.0]), sr, &fixed)));
        assert!(clicks::<f64>(ClickPositions::Times(&[1.0]), sr, &cfg).is_ok());

        let audio = AudioSamples::new_mono(ndarray::Array1::from(vec![0.1; 22050]), sr).unwrap();
        assert!(matches!(
            audio.time_stretch(1e-6, &StftConfig::default()),
            Err(AudioSampleError::Parameter(_))
        ));
    }
}
