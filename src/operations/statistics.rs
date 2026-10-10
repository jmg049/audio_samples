//! Statistical analysis operations for [`AudioSamples`].
//!
//! ## What
//!
//! This module implements the [`AudioStatistics`] trait: descriptive statistics
//! (peak, mean, RMS, variance), temporal descriptors (zero crossings,
//! auto- and cross-correlation) and whole-signal spectral descriptors
//! (centroid, rolloff, bandwidth, flatness, crest, slope, contrast). It also
//! provides the free function [`fft_alignment_lag`] for delay estimation.
//!
//! ## Why
//!
//! Statistical measures are a core part of audio analysis. Isolating them into a
//! single trait keeps the [`AudioSamples`] API organised and lets users access
//! only the statistical surface they need. This module is the sole implementor of
//! [`AudioStatistics`] for [`AudioSamples`].
//!
//! ## How
//!
//! All operations are available on any [`AudioSamples<T>`] where `T` is a
//! supported sample type (`u8`, `i16`, `I24`, `i32`, `f32`, `f64`). Import
//! [`AudioStatistics`] and call methods on your audio directly.
//!
//! ### Units: raw sample values vs normalised amplitude
//!
//! - `mean`, `variance`, `std_dev`, `rms`, `autocorrelation` and
//!   `cross_correlation` work on **raw sample values** converted numerically to
//!   `f64` (accumulated in `f64`, so integer types never overflow). For `i16`
//!   input an RMS of `11585.0` means `11585 / 32768 ≈ 0.354` of full scale.
//!   `u8` is offset-binary (silence = 128), so its mean of a silent signal is
//!   `128.0`. Convert first (e.g. [`as_f64`](crate::AudioTypeConversion::as_f64))
//!   if you want values in `[-1, 1]`.
//! - `zero_crossings` / `zero_crossing_rate` and all spectral methods use the
//!   audio-aware conversion to `f64` (integers scaled to `[-1, 1]`, `u8` 128 → 0),
//!   so `u8` zero crossings are counted about mid-scale.
//!
//! ### Time-domain definitions
//!
//! For samples `x[0..N]` (all channels pooled unless stated otherwise):
//!
//! - Mean: `x̄ = (1/N) Σ x[n]`.
//! - RMS: `sqrt((1/N) Σ x[n]²)`. A sine of amplitude `A` has RMS `A/√2`.
//! - Variance: population variance `(1/N) Σ (x[n] − x̄)²` over every sample of
//!   every channel; `std_dev` is its square root.
//! - Crest factor (not a method here; see
//!   [`detect_dynamic_range`](crate::utils::detection::detect_dynamic_range)):
//!   `peak / RMS`, or `20·log10(peak/RMS)` dB. A sine has `√2` (≈ 3.01 dB).
//! - Zero crossings: number of adjacent pairs whose sign changes, where a value
//!   `> 0` is positive and a value `≤ 0` is non-positive (exact zeros count on
//!   the non-positive side). Summed over channels.
//! - Zero-crossing rate: **crossings per second**, `zero_crossings / duration_s`
//!   with `duration_s = samples_per_channel / sample_rate`. A sine at `f` Hz
//!   gives about `2f`. For multi-channel audio the count is summed over channels
//!   while the duration is per channel, so the rate scales with the channel
//!   count. Note: librosa's `zero_crossing_rate` is a per-frame **fraction of
//!   samples** (crossings / frame length), not a rate in Hz.
//! - Autocorrelation (first channel): the unbiased estimate
//!   `r[k] = (1/(N−k)) Σ_{n=0}^{N−1−k} x[n]·x[n+k]`, with no mean removal and no
//!   division by `r[0]`. `r[0]` is the mean square (`A²/2` for a sine). For
//!   periodic signals `r[k]` peaks again at multiples of the period, and at large
//!   `k` (few overlapping samples) the estimate is noisy and may exceed `r[0]`.
//!   Divide by `r[0]` for the normalised autocorrelation `ρ[k] ∈ [−1, 1]`
//!   (approximately; the unbiased estimator is not bounded exactly). Note: the
//!   *biased* estimator `(1/N) Σ …` used by many textbooks and
//!   `numpy.correlate(x, x, "full")/N` is smaller by the factor `(N−k)/N`.
//! - Cross-correlation (first channels): `r_xy[l] = (1/C_l) Σ_{i<C_l} x[i]·y[i+l]`
//!   for `l = 0..=max_lag`, with `C_l = min(N_x, N_y − l)`, `x = self`,
//!   `y = other`. A peak at `l = D` means `other` is `self` delayed by `D`
//!   samples. Negative lags are not produced: swap the arguments to detect the
//!   opposite delay, or use [`fft_alignment_lag`].
//!
//! ### Spectral definitions
//!
//! The spectral methods reduce the input to one channel (see
//! `ChannelReduction` in [`crate::operations::types`]), take a
//! **single FFT of the whole signal** (FFT length = signal length `N`, no
//! framing, rectangular window except `spectral_contrast`, which uses a Hann
//! window) and use the one-sided spectrum `X[k]`, `k = 0..=⌊N/2⌋`, without
//! normalisation. Bin `k` lies at `f_k = k·fs/N` Hz. With power `P_k = |X[k]|²`
//! and magnitude `M_k = |X[k]|`:
//!
//! - Centroid (**power**-weighted): `Σ f_k·P_k / Σ P_k` Hz.
//! - Rolloff: the smallest `f_k` such that `Σ_{j≤k} P_j ≥ p·Σ P_j` (power
//!   cumulative sum, no interpolation between bins).
//! - Bandwidth (**magnitude**-weighted): `sqrt(Σ (f_k − c_M)²·M_k / Σ M_k)` with
//!   its own magnitude-weighted centroid `c_M = Σ f_k·M_k / Σ M_k`. It is
//!   therefore not centred on [`spectral_centroid`](AudioStatistics::spectral_centroid)
//!   (which is power-weighted).
//! - Flatness (Wiener entropy): `exp(mean(ln(P_k + ε))) / mean(P_k)`, `ε = 1e-10`,
//!   clamped to `[0, 1]`.
//! - Crest: `max(M_k) / mean(M_k)`.
//! - Slope: ordinary least-squares slope of `M_k` against `f_k` (magnitude per
//!   Hz; depends on the signal length because the spectrum is unnormalised).
//! - Contrast: see [`spectral_contrast`](AudioStatistics::spectral_contrast);
//!   values are `10·log10` of **magnitudes** (librosa's convention).
//!
//! Note: librosa computes these features per STFT frame (Hann window,
//! `n_fft = 2048`) and weights centroid/bandwidth by **magnitude**
//! (`S = |X|`), so its centroid and rolloff are not directly comparable with
//! the power-weighted, whole-signal values here.
//!
//! ## Example: Time-domain analysis
//!
//! ```rust
//! use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
//! use ndarray::array;
//!
//! let data = array![1.0f32, -1.0, 0.5, -0.5];
//! let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
//!
//! let peak = audio.peak();
//! let rms  = audio.rms();
//! let mean = audio.mean();
//!
//! assert_eq!(peak, 1.0);
//! assert!((rms - 0.625f64.sqrt()).abs() < 1e-12); // sqrt((1 + 1 + 0.25 + 0.25) / 4)
//! assert_eq!(mean, 0.0);
//! ```
//!
//! ## Example: Frequency-domain analysis
//!
//! ```rust
//! # #[cfg(feature = "transforms")] {
//! use audio_samples::{sine_wave, AudioStatistics, sample_rate};
//! use audio_samples::operations::types::ChannelReduction;
//! use std::time::Duration;
//!
//! // 1 s at 44.1 kHz: the 1 kHz tone falls exactly on bin 1000 (fs/N = 1 Hz).
//! let audio = sine_wave::<f64>(1000.0, Duration::from_secs(1), sample_rate!(44100), 0.5);
//!
//! let centroid = audio.spectral_centroid(ChannelReduction::Error).unwrap();
//! let rolloff  = audio.spectral_rolloff(0.85, ChannelReduction::Error).unwrap();
//!
//! assert!((centroid - 1000.0).abs() < 1.0);
//! assert!((rolloff - 1000.0).abs() < 1.0);
//! # }
//! ```
//!
//! ## Error Handling
//!
//! Most operations return plain numeric results. Methods that perform FFT
//! computation or validate input parameters return [`crate::AudioSampleResult`].
//! An [`crate::AudioSampleError::Processing`] is returned for mathematical
//! failures, and [`crate::AudioSampleError::Parameter`] for invalid arguments.
//!
//! ## Feature Flags
//!
//! - `statistics`: enables this module and the time-domain methods.
//! - `transforms`: additionally enables `autocorrelation`, the spectral
//!   methods and [`fft_alignment_lag`] via the [`spectrograms`] crate.
//!
//! ## See Also
//!
//! - [`AudioSamples`]: The core data structure providing access to underlying
//!   sample buffers.
//! - [`AudioStatistics`]: The trait defining all statistical operations.
//!
//! [`AudioStatistics`]: crate::operations::traits::AudioStatistics

use std::num::NonZeroUsize;

use crate::operations::traits::AudioStatistics;
#[cfg(feature = "transforms")]
use crate::operations::types::ChannelReduction;
use crate::repr::AudioData;
use crate::traits::StandardSample;
use crate::{AudioSampleResult, AudioSamples, ParameterError};

#[cfg(feature = "transforms")]
use crate::{AudioSampleError, ProcessingError};
#[cfg(feature = "transforms")]
use num_complex::Complex;

#[cfg(feature = "transforms")]
use non_empty_slice::NonEmptySlice;

use ndarray::Axis;
use non_empty_slice::NonEmptyVec;

#[cfg(feature = "transforms")]
use spectrograms::FftPlanner;

#[cfg(feature = "transforms")]
use std::cell::RefCell;

#[cfg(feature = "transforms")]
thread_local! {
    /// Thread-local [`FftPlanner`] reused across all FFT-based statistics in
    /// this module. `spectrograms::FftPlanner` wraps a real-FFT planner that
    /// memoizes plans by size internally, so reconstructing it per call
    /// discarded that cache. One planner per thread lets repeated same-size
    /// transforms reuse cached plans. The planner is `!Sync`, so `thread_local`
    /// is the correct scope.
    static FFT_PLANNER: RefCell<FftPlanner> = RefCell::new(FftPlanner::new());
}

/// Runs `f` with the thread-local cached [`FftPlanner`] borrowed mutably.
///
/// The borrow is released as soon as `f` returns.
#[cfg(feature = "transforms")]
#[inline]
fn with_fft_planner<R>(f: impl FnOnce(&mut FftPlanner) -> R) -> R {
    FFT_PLANNER.with(|p| f(&mut p.borrow_mut()))
}

/// Reduces a (possibly multi-channel) signal to a single `f64` sample vector
/// according to `reduction`, for the spectral analysis ops that produce one
/// scalar result.
///
/// Samples are converted with audio-aware [`ConvertTo`](crate::traits::ConvertTo)
/// scaling. Returns an error when `reduction` forbids the channel layout (e.g.
/// [`ChannelReduction::Error`] on multi-channel input) or names an out-of-range
/// channel.
#[cfg(feature = "transforms")]
fn reduce_to_mono_f64<T>(
    audio: &AudioSamples<'_, T>,
    reduction: ChannelReduction,
    operation: &str,
) -> AudioSampleResult<Vec<f64>>
where
    T: StandardSample,
{
    match &audio.data() {
        AudioData::Mono(arr) => Ok(arr.iter().map(|&x| x.convert_to()).collect()),
        AudioData::Multi(multi) => {
            let view = multi.as_view();
            let num_channels = view.nrows();
            match reduction {
                ChannelReduction::Error => Err(crate::AudioSampleError::Parameter(
                    ParameterError::invalid_value(
                        "channels",
                        format!(
                            "{operation} is only defined for mono signals; pass a \
                             ChannelReduction other than `Error` to reduce \
                             {num_channels} channels to one"
                        ),
                    ),
                )),
                ChannelReduction::First => {
                    Ok(view.row(0).iter().map(|&x| x.convert_to()).collect())
                }
                ChannelReduction::Channel(idx) => {
                    if idx >= num_channels {
                        return Err(crate::AudioSampleError::Parameter(
                            ParameterError::out_of_range(
                                "channel",
                                idx.to_string(),
                                "0",
                                (num_channels - 1).to_string(),
                                format!(
                                    "{operation}: channel index {idx} out of range for \
                                     {num_channels} channels"
                                ),
                            ),
                        ));
                    }
                    Ok(view.row(idx).iter().map(|&x| x.convert_to()).collect())
                }
                ChannelReduction::Average => {
                    let n = view.ncols();
                    let inv = 1.0 / num_channels as f64;
                    let mut out = vec![0.0f64; n];
                    for row in view.rows() {
                        for (o, &x) in out.iter_mut().zip(row.iter()) {
                            let v: f64 = x.convert_to();
                            *o += v * inv;
                        }
                    }
                    Ok(out)
                }
            }
        }
    }
}

/// The raw autocorrelation sums `r[k] = sum_j x[j] x[j + k]` for
/// `k = 0..=max_lag` (`max_lag < x.len()`), computed in `f64`.
///
/// Only `max_lag + 1` lags are needed, so the signal is processed in blocks
/// (overlap-save): block `b` of `B` samples is correlated with the `B +
/// max_lag` samples starting at the same position, through real FFTs of a
/// fixed size `N >= B + max_lag` (no circular wrap-around for those lags),
/// and the cross spectra of all blocks are summed before one inverse
/// transform. The memory is a few `N`-sized buffers, independent of the
/// signal length, and the transforms are shorter than a whole-signal
/// transform of size `>= 2 len - 1`.
#[cfg(feature = "transforms")]
fn autocorrelation_sums<T: StandardSample>(x: &[T], max_lag: usize) -> Option<Vec<f64>> {
    use crate::AudioFloat;
    let n = x.len();
    // Block transform size: four times the lag span (at least 1024), or a
    // single block covering the whole signal when that is shorter.
    let n_fft = (4 * (max_lag + 1))
        .max(1024)
        .next_power_of_two()
        .min((n + max_lag).next_power_of_two());
    let block = n_fft - max_lag;
    let (fwd, inv) =
        f64::with_real_fft_planner(|p| (p.plan_fft_forward(n_fft), p.plan_fft_inverse(n_fft)));
    let mut a_in = vec![0.0_f64; n_fft];
    let mut s_in = vec![0.0_f64; n_fft];
    let mut a_spec = fwd.make_output_vec();
    let mut s_spec = fwd.make_output_vec();
    let mut acc = vec![Complex::new(0.0_f64, 0.0); n_fft / 2 + 1];
    let mut scratch = fwd.make_scratch_vec();
    for start in (0..n).step_by(block) {
        let a_len = block.min(n - start);
        let s_len = n_fft.min(n - start);
        for (d, &v) in a_in.iter_mut().zip(&x[start..start + a_len]) {
            *d = v.cast_into();
        }
        a_in[a_len..].fill(0.0);
        for (d, &v) in s_in.iter_mut().zip(&x[start..start + s_len]) {
            *d = v.cast_into();
        }
        s_in[s_len..].fill(0.0);
        fwd.process_with_scratch(&mut a_in, &mut a_spec, &mut scratch)
            .ok()?;
        fwd.process_with_scratch(&mut s_in, &mut s_spec, &mut scratch)
            .ok()?;
        for ((c, a), s) in acc.iter_mut().zip(&a_spec).zip(&s_spec) {
            *c += a.conj() * s;
        }
    }
    // The imaginary parts of the DC and Nyquist bins are zero in exact
    // arithmetic; the inverse real FFT requires them to be zero.
    acc[0].im = 0.0;
    if let Some(last) = acc.last_mut() {
        last.im = 0.0;
    }
    let mut out = vec![0.0_f64; n_fft];
    let mut iscratch = inv.make_scratch_vec();
    inv.process_with_scratch(&mut acc, &mut out, &mut iscratch)
        .ok()?;
    let scale = 1.0 / n_fft as f64;
    out.truncate(max_lag + 1);
    for v in &mut out {
        *v *= scale;
    }
    Some(out)
}

impl<T> AudioStatistics for AudioSamples<'_, T>
where
    T: StandardSample,
{
    /// Returns the peak (maximum absolute value) across all samples and channels.
    ///
    /// # Returns
    /// The maximum absolute sample value, in the native sample type `T`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -3.0, 2.5, -1.5];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// assert_eq!(audio.peak(), 3.0);
    /// ```
    #[inline]
    fn peak(&self) -> T {
        // Four independent accumulators let LLVM's SLP vectoriser pack them into
        // a single SIMD register (e.g. 4-wide ymm for f32) without requiring
        // -ffast-math. Falls back to ndarray fold for non-contiguous views.
        let zero = T::default();
        let abs_max_slice = |slice: &[T]| -> T {
            if let Some(result) = T::avx2_abs_max(slice) {
                return result;
            }
            let mut acc = [zero; 4];
            let chunks = slice.chunks_exact(4);
            let rem = chunks.remainder();
            for chunk in chunks {
                for j in 0..4 {
                    let x = chunk[j];
                    let ax = if x < zero { zero - x } else { x };
                    if ax > acc[j] {
                        acc[j] = ax;
                    }
                }
            }
            for &x in rem {
                let ax = if x < zero { zero - x } else { x };
                if ax > acc[0] {
                    acc[0] = ax;
                }
            }
            acc.iter().fold(zero, |a, &b| if b > a { b } else { a })
        };
        let fold_ndarray = |acc: T, &x: &T| {
            let ax = if x < zero { zero - x } else { x };
            if ax > acc { ax } else { acc }
        };
        match &self.data() {
            AudioData::Mono(arr) => match arr.as_slice() {
                Some(s) => abs_max_slice(s),
                None => arr.fold(zero, fold_ndarray),
            },
            AudioData::Multi(arr) => match arr.as_slice() {
                Some(s) => abs_max_slice(s),
                None => arr.fold(zero, fold_ndarray),
            },
        }
    }

    /// Returns the minimum sample value across all channels.
    ///
    /// # Returns
    /// The smallest sample value found, in the native sample type `T`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -3.0, 2.5, -1.5];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// assert_eq!(audio.min_sample(), -3.0);
    /// ```
    #[inline]
    fn min_sample(&self) -> T {
        match &self.data() {
            AudioData::Mono(arr) => {
                // Use ndarray's efficient fold operation for vectorized minimum finding
                arr.fold(arr[0], |acc, &x| if x < acc { x } else { acc })
            }
            AudioData::Multi(arr) => {
                // Vectorized minimum across entire multi-channel array
                arr.fold(arr[[0, 0]], |acc, &x| if x < acc { x } else { acc })
            }
        }
    }

    /// Returns the maximum sample value across all channels.
    ///
    /// # Returns
    /// The largest sample value found, in the native sample type `T`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -3.0, 2.5, -1.5];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// assert_eq!(audio.max_sample(), 2.5);
    /// ```
    #[inline]
    fn max_sample(&self) -> T {
        match &self.data() {
            AudioData::Mono(arr) => {
                // Use ndarray's efficient fold operation for vectorized maximum finding
                arr.fold(arr[0], |acc, &x| if x > acc { x } else { acc })
            }
            AudioData::Multi(arr) => {
                // Vectorized maximum across entire multi-channel array
                arr.fold(arr[[0, 0]], |acc, &x| if x > acc { x } else { acc })
            }
        }
    }

    /// Computes the arithmetic mean of all samples across all channels.
    ///
    /// `x̄ = (1/N) Σ x[n]` over every sample of every channel (`N` = total sample
    /// count), accumulated in `f64` so integer sample types cannot overflow. The
    /// value is in raw sample units: for `i16` it lies in `[-32768, 32767]`, and
    /// for `u8` (offset-binary) a silent signal has mean `128.0`.
    ///
    /// # Returns
    /// The mean value as `f64`, in raw sample units.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 2.0, -2.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// assert_eq!(audio.mean(), 0.0);
    /// ```
    #[inline]
    fn mean(&self) -> f64 {
        // Accumulate in f64: summing in `T` overflows/wraps for integer types.
        let (sum, n) = match &self.data() {
            AudioData::Mono(arr) => (
                arr.iter().map(|&x| -> f64 { x.cast_into() }).sum::<f64>(),
                arr.len().get(),
            ),
            AudioData::Multi(arr) => (
                arr.iter().map(|&x| -> f64 { x.cast_into() }).sum::<f64>(),
                arr.len().get(),
            ),
        };
        sum / n as f64
    }

    /// Returns the value at the temporal midpoint of a mono signal.
    ///
    /// For even-length signals the result is the average of the two samples at
    /// the two central indices. For odd-length signals the single central sample
    /// is returned directly. Samples are selected by index position; the buffer
    /// is not sorted.
    ///
    /// # Returns
    /// `Some(value)` for mono audio, or `None` if the signal is multi-channel.
    ///
    /// # Assumptions
    /// For even-length signals the sum of the two central samples must not
    /// overflow the sample type `T`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, 3.0, 5.0, 7.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Central indices 1 and 2: (3.0 + 5.0) / 2.0 = 4.0
    /// assert_eq!(audio.midpoint_sample(), Some(4.0));
    /// ```
    #[inline]
    fn midpoint_sample(&self) -> Option<f64> {
        let mono = self.as_mono()?;
        let mono_len = mono.len().get();
        Some(if mono_len.is_multiple_of(2) {
            let first_idx = (mono_len / 2) - 1;
            let first_val = self[first_idx];
            let second_val = self[first_idx + 1];
            let sum: f64 = (first_val + second_val).cast_into();

            sum / 2.0
        } else {
            self[self.len().get() / 2].cast_into()
        })
    }

    /// Computes the Root Mean Square (RMS) of all samples across all channels.
    ///
    /// `RMS = sqrt((1/N) Σ x[n]²)` over every sample of every channel, accumulated
    /// in `f64`. It measures the signal's average power and is commonly used for
    /// loudness estimation. A sine of amplitude `A` has RMS `A/√2`.
    ///
    /// Samples are used in raw units: for `i16`, full scale is 32768, and for `u8`
    /// the 128 offset is not removed. Convert with
    /// [`as_f64`](crate::AudioTypeConversion::as_f64) first for an RMS relative to
    /// full scale.
    ///
    /// # Returns
    /// The RMS value as `f64`, in raw sample units.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 1.0, -1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let rms = audio.rms();
    /// assert!((rms - 1.0).abs() < 1e-6);
    /// ```
    #[inline]
    fn rms(&self) -> f64 {
        // Four independent f64 accumulators break the sequential dependency chain
        // so LLVM's SLP vectoriser can pack them into a ymm register (4 f64 wide)
        // without -ffast-math. Falls back to ndarray iter for non-contiguous views.
        let sum_sq_slice = |slice: &[T]| -> f64 {
            let mut acc = [0.0f64; 4];
            let chunks = slice.chunks_exact(4);
            let rem = chunks.remainder();
            for chunk in chunks {
                let f0: f64 = chunk[0].cast_into();
                let f1: f64 = chunk[1].cast_into();
                let f2: f64 = chunk[2].cast_into();
                let f3: f64 = chunk[3].cast_into();
                acc[0] += f0 * f0;
                acc[1] += f1 * f1;
                acc[2] += f2 * f2;
                acc[3] += f3 * f3;
            }
            for &x in rem {
                let f: f64 = x.cast_into();
                acc[0] += f * f;
            }
            acc[0] + acc[1] + acc[2] + acc[3]
        };
        let (sum_sq, n) = match &self.data() {
            AudioData::Mono(arr) => {
                let s = match arr.as_slice() {
                    Some(s) => sum_sq_slice(s),
                    None => arr
                        .iter()
                        .map(|&x| {
                            let f: f64 = x.cast_into();
                            f * f
                        })
                        .sum(),
                };
                (s, arr.len().get())
            }
            AudioData::Multi(arr) => {
                let s = match arr.as_slice() {
                    Some(s) => sum_sq_slice(s),
                    None => arr
                        .iter()
                        .map(|&x| {
                            let f: f64 = x.cast_into();
                            f * f
                        })
                        .sum(),
                };
                (s, arr.len().get())
            }
        };
        (sum_sq / n as f64).sqrt()
    }

    #[inline]
    fn rms_and_peak(&self) -> (f64, T) {
        // Single-pass combined computation with four independent accumulators per
        // quantity. Reading the data once is critical for signals larger than L3
        // cache where two separate calls would double memory bandwidth.
        let zero = T::default();

        let combined_slice = |slice: &[T]| -> (f64, T) {
            // Eight independent accumulators: sq uses 2 ymm f64 registers,
            // pk uses 1 ymm f32 register. Explicit constant-index loads
            // let LLVM keep all 8 values in registers (no re-read from memory)
            // and pack both sq and pk into SIMD via SLP vectorisation.
            let mut sq = [0.0f64; 8];
            let mut pk = [zero; 8];
            let chunks = slice.chunks_exact(8);
            let rem = chunks.remainder();
            for chunk in chunks {
                // Load all 8 elements once; compiler keeps them in registers.
                let x0 = chunk[0];
                let x1 = chunk[1];
                let x2 = chunk[2];
                let x3 = chunk[3];
                let x4 = chunk[4];
                let x5 = chunk[5];
                let x6 = chunk[6];
                let x7 = chunk[7];
                // sq: 8 independent f64 chains → 2 ymm accumulator registers.
                let f0: f64 = x0.cast_into();
                sq[0] += f0 * f0;
                let f1: f64 = x1.cast_into();
                sq[1] += f1 * f1;
                let f2: f64 = x2.cast_into();
                sq[2] += f2 * f2;
                let f3: f64 = x3.cast_into();
                sq[3] += f3 * f3;
                let f4: f64 = x4.cast_into();
                sq[4] += f4 * f4;
                let f5: f64 = x5.cast_into();
                sq[5] += f5 * f5;
                let f6: f64 = x6.cast_into();
                sq[6] += f6 * f6;
                let f7: f64 = x7.cast_into();
                sq[7] += f7 * f7;
                // pk: 8-wide abs-max, SLP-vectorisable to vmaxps + vandps.
                let ax0 = if x0 < zero { zero - x0 } else { x0 };
                if ax0 > pk[0] {
                    pk[0] = ax0;
                }
                let ax1 = if x1 < zero { zero - x1 } else { x1 };
                if ax1 > pk[1] {
                    pk[1] = ax1;
                }
                let ax2 = if x2 < zero { zero - x2 } else { x2 };
                if ax2 > pk[2] {
                    pk[2] = ax2;
                }
                let ax3 = if x3 < zero { zero - x3 } else { x3 };
                if ax3 > pk[3] {
                    pk[3] = ax3;
                }
                let ax4 = if x4 < zero { zero - x4 } else { x4 };
                if ax4 > pk[4] {
                    pk[4] = ax4;
                }
                let ax5 = if x5 < zero { zero - x5 } else { x5 };
                if ax5 > pk[5] {
                    pk[5] = ax5;
                }
                let ax6 = if x6 < zero { zero - x6 } else { x6 };
                if ax6 > pk[6] {
                    pk[6] = ax6;
                }
                let ax7 = if x7 < zero { zero - x7 } else { x7 };
                if ax7 > pk[7] {
                    pk[7] = ax7;
                }
            }
            for &x in rem {
                let f: f64 = x.cast_into();
                sq[0] += f * f;
                let ax = if x < zero { zero - x } else { x };
                if ax > pk[0] {
                    pk[0] = ax;
                }
            }
            let sum_sq = sq[0] + sq[1] + sq[2] + sq[3] + sq[4] + sq[5] + sq[6] + sq[7];
            let peak = pk.iter().fold(zero, |a, &b| if b > a { b } else { a });
            (sum_sq, peak)
        };

        let (sum_sq, peak, n) = match &self.data() {
            AudioData::Mono(arr) => {
                let (sq, pk) = match arr.as_slice() {
                    Some(s) => combined_slice(s),
                    None => {
                        let sq = arr
                            .iter()
                            .map(|&x| {
                                let f: f64 = x.cast_into();
                                f * f
                            })
                            .sum();
                        let pk = arr.fold(zero, |acc, &x| {
                            let ax = if x < zero { zero - x } else { x };
                            if ax > acc { ax } else { acc }
                        });
                        (sq, pk)
                    }
                };
                (sq, pk, arr.len().get())
            }
            AudioData::Multi(arr) => {
                let (sq, pk) = match arr.as_slice() {
                    Some(s) => combined_slice(s),
                    None => {
                        let sq = arr
                            .iter()
                            .map(|&x| {
                                let f: f64 = x.cast_into();
                                f * f
                            })
                            .sum();
                        let pk = arr.fold(zero, |acc, &x| {
                            let ax = if x < zero { zero - x } else { x };
                            if ax > acc { ax } else { acc }
                        });
                        (sq, pk)
                    }
                };
                (sq, pk, arr.len().get())
            }
        };

        ((sum_sq / n as f64).sqrt(), peak)
    }

    /// Computes the population variance of the audio samples.
    ///
    /// `σ² = (1/N) Σ (x[n] − x̄)²` over every sample of every channel (all channels
    /// pooled, `N` = total sample count, `x̄` = [`mean`](AudioStatistics::mean)),
    /// accumulated in `f64` in raw sample units. Divide by `N` (population), not
    /// `N − 1`.
    ///
    /// # Returns
    /// The variance as `f64`, in squared raw sample units.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, 2.0, 3.0, 4.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let variance = audio.variance();
    /// assert!((variance - 1.25).abs() < 1e-6);
    /// ```
    #[inline]
    fn variance(&self) -> f64 {
        // Two-pass population variance over every sample of every channel,
        // accumulated in f64 (no overflow for integer sample types).
        let mean = self.mean();
        let sq_dev = |&x: &T| -> f64 {
            let v: f64 = x.cast_into();
            let d = v - mean;
            d * d
        };
        let (sum_sq, n) = match &self.data() {
            AudioData::Mono(arr) => (arr.iter().map(sq_dev).sum::<f64>(), arr.len().get()),
            AudioData::Multi(arr) => (arr.iter().map(sq_dev).sum::<f64>(), arr.len().get()),
        };
        sum_sq / n as f64
    }

    /// Computes the standard deviation of the audio samples.
    ///
    /// Standard deviation is the square root of [`AudioStatistics::variance`].
    /// It expresses the spread of sample values in the same units as the
    /// original data.
    ///
    /// # Returns
    /// The standard deviation as `f64`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, 2.0, 3.0, 4.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let std_dev = audio.std_dev();
    /// assert!((std_dev - 1.25_f64.sqrt()).abs() < 1e-6);
    /// ```
    #[inline]
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Counts the number of zero crossings in the audio signal.
    ///
    /// A zero crossing is a pair of adjacent samples `(x[n−1], x[n])` where one is
    /// `> 0` and the other is `≤ 0` (an exact zero counts as non-positive). Samples
    /// are first mapped with the audio-aware conversion to `f64`, so `u8`
    /// (offset-binary) is measured about its mid-scale value 128. A sine at `f` Hz
    /// lasting `T` seconds has about `2·f·T` crossings. The count is useful for
    /// estimating the noisiness or dominant frequency of a signal.
    ///
    /// # Returns
    /// The total number of zero crossings, summed over all channels (each channel
    /// is scanned separately). Returns 0 if the audio has fewer than 2 samples per
    /// channel.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 1.0, -1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// assert_eq!(audio.zero_crossings(), 3);
    /// ```
    #[inline]
    fn zero_crossings(&self) -> usize {
        match &self.data() {
            AudioData::Mono(arr) => {
                if arr.len() < nzu!(2) {
                    return 0;
                }

                let mut crossings = 0;
                for i in 1..arr.len().get() {
                    // Audio-aware conversion: u8 mid-scale (128) maps to 0.0, so
                    // crossings are counted about the true zero level.
                    let prev: f64 = arr[i - 1].convert_to();
                    let curr: f64 = arr[i].convert_to();

                    // Check for sign change (zero crossing)
                    if (prev > 0.0 && curr <= 0.0) || (prev <= 0.0 && curr > 0.0) {
                        crossings += 1;
                    }
                }
                crossings
            }
            AudioData::Multi(arr) => {
                // For multi-channel, count zero crossings in each channel and sum them
                let mut total_crossings = 0;

                for channel in arr.axis_iter(Axis(0)) {
                    if channel.len() < 2 {
                        continue;
                    }

                    for i in 1..channel.len() {
                        let prev: f64 = channel[i - 1].convert_to();
                        let curr: f64 = channel[i].convert_to();

                        if (prev > 0.0 && curr <= 0.0) || (prev <= 0.0 && curr > 0.0) {
                            total_crossings += 1;
                        }
                    }
                }
                total_crossings
            }
        }
    }

    /// Computes the zero-crossing rate in crossings per second.
    ///
    /// `ZCR = zero_crossings / duration_s` with
    /// `duration_s = samples_per_channel / sample_rate`. The unit is crossings per
    /// second (Hz), not a fraction of samples: a sine at `f` Hz gives about `2f`.
    /// For multi-channel audio the crossings of all channels are summed while the
    /// duration is per channel, so the rate scales with the channel count.
    ///
    /// This normalises the zero crossing count by the signal duration, providing
    /// a frequency-like measure that is independent of signal length. For
    /// multi-channel audio it is the mean of the per-channel rates (the total
    /// from [`zero_crossings`](Self::zero_crossings) divided by the number of
    /// channels), so a stereo signal with two identical channels has the same
    /// rate as either channel alone.
    /// Note: librosa's `zero_crossing_rate` returns a per-frame fraction
    /// (crossings / frame length). For a mono signal, divide this value by the
    /// sample rate to get the equivalent whole-signal fraction of samples.
    ///
    /// # Returns
    /// The number of zero crossings per second as `f64`, averaged over channels.
    ///
    /// # Differences from librosa
    ///
    /// `librosa.feature.zero_crossing_rate` returns, per channel and per frame,
    /// the fraction of samples at which the sign changes (centred frames of 2048
    /// samples, hop 512). This method is a single whole-signal rate in Hz.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 1.0, -1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // 3 crossings in 4 samples = 4 / 44100 s.
    /// let zcr = audio.zero_crossing_rate();
    /// assert!((zcr - 3.0 * 44100.0 / 4.0).abs() < 1e-9);
    /// ```
    #[inline]
    fn zero_crossing_rate(&self) -> f64 {
        // `zero_crossings` sums over channels; average it so the rate is per channel.
        let crossings_per_channel = self.zero_crossings() as f64 / self.num_channels().get() as f64;
        let duration_seconds = self.duration_seconds(); // guaranteed > 0.0 since we do not allow empty audio
        crossings_per_channel / duration_seconds
    }

    /// Computes the autocorrelation function up to `max_lag` samples.
    ///
    /// Returns the unbiased estimate
    /// `r[k] = (1/(N−k)) Σ_{n=0}^{N−1−k} x[n]·x[n+k]` for `k = 0..=max_lag`, in
    /// squared raw sample units, with no mean removal and no normalisation by
    /// `r[0]`. `r[0]` is the mean square (`A²/2` for a sine of amplitude `A`). The
    /// values do not necessarily decrease with `k`: for a periodic signal `r[k]`
    /// peaks again at every multiple of the period, and at large `k` (few
    /// overlapping samples) the estimate is noisy and can exceed `r[0]`. It is
    /// computed with a zero-padded FFT, so the correlation is linear, not circular.
    ///
    /// For multi-channel audio only the first channel is used.
    ///
    /// Note: unlike the biased estimator `(1/N) Σ …` (e.g.
    /// `numpy.correlate(x, x, "full")[N-1:] / N`), each lag is divided by its own
    /// overlap count `N − k`.
    ///
    /// Reference: [Autocorrelation - Wikipedia](https://en.wikipedia.org/wiki/Autocorrelation)
    ///
    /// # Arguments
    /// - `max_lag` - the maximum lag offset in samples. The effective maximum
    ///   lag is clamped to `signal_length - 1`.
    ///
    /// # Returns
    /// A [`NonEmptyVec`] of correlation values for lags `0` through
    /// `min(max_lag, signal_length - 1)`, or `None` if the FFT computation
    /// fails.
    ///
    /// # Differences from librosa
    ///
    /// `librosa.autocorrelate` (like `numpy.correlate(x, x, "full")[n-1:]`)
    /// returns the raw sums `Σ x[j] · x[j+k]`; multiply `r[k]` by `n - k` to
    /// obtain them. librosa also autocorrelates every channel.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    /// use std::num::NonZeroUsize;
    ///
    /// let data = array![1.0f32, 2.0, 3.0, 4.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let r = audio.autocorrelation(NonZeroUsize::new(3).unwrap()).unwrap();
    /// assert_eq!(r.len(), NonZeroUsize::new(4).unwrap()); // lags 0..=3
    /// // r[0] = (1 + 4 + 9 + 16) / 4, r[1] = (2 + 6 + 12) / 3
    /// assert!((r[0] - 7.5).abs() < 1e-9);
    /// assert!((r[1] - 20.0 / 3.0).abs() < 1e-9);
    /// ```
    #[cfg(feature = "transforms")]
    #[inline]
    fn autocorrelation(&self, max_lag: NonZeroUsize) -> Option<NonEmptyVec<f64>> {
        let row = match &self.data() {
            AudioData::Mono(arr) => arr.as_view(),
            AudioData::Multi(arr) => arr.row(0),
        };
        let n = row.len();
        let effective_max_lag = max_lag.get().min(n - 1);
        let owned;
        let x: &[T] = if let Some(s) = row.as_slice() {
            s
        } else {
            owned = row.to_vec();
            &owned
        };
        let raw = autocorrelation_sums(x, effective_max_lag)?;

        // Unbiased normalisation by the overlap count (n - lag).
        let correlations: Vec<f64> = raw
            .iter()
            .enumerate()
            .map(|(lag, &r)| r / (n - lag) as f64)
            .collect();

        // safety: correlations has effective_max_lag + 1 >= 1 elements
        Some(unsafe { NonEmptyVec::new_unchecked(correlations) })
    }

    /// Computes the cross-correlation with another audio signal for non-negative
    /// lags.
    ///
    /// `r[l] = (1/C_l) Σ_{i=0}^{C_l−1} self[i]·other[i+l]` for `l = 0..=max_lag`,
    /// with `C_l = min(len_self, len_other − l)`. This is an unbiased
    /// (overlap-normalised) estimate in squared raw sample units, with no mean
    /// removal, accumulated in `f64`. A peak at `l = D` means `other` is `self`
    /// delayed by `D` samples. Negative lags are not computed: to detect `other`
    /// leading `self`, swap the arguments or use
    /// [`fft_alignment_lag`](crate::operations::statistics::fft_alignment_lag).
    ///
    /// For multi-channel audio only the first channels of both signals are
    /// correlated.
    ///
    /// # Arguments
    /// - `other` - the second audio signal. Must have the same number of
    ///   channels as `self`.
    /// - `max_lag` - the maximum lag offset in samples. The effective maximum
    ///   lag is clamped to `min(len_self, len_other) - 1`.
    ///
    /// # Returns
    /// A [`NonEmptyVec`] of correlation values for lags `0` through the
    /// effective maximum lag.
    ///
    /// # Errors
    /// Returns an error if the two signals have different numbers of channels,
    /// or if one is mono and the other is multi-channel.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    /// use std::num::NonZeroUsize;
    ///
    /// let data1 = array![1.0f32, 0.0, -1.0, 0.0];
    /// let data2 = array![0.0, 1.0, 0.0, -1.0]; // data1 delayed by one sample
    /// let audio1 = AudioSamples::new_mono(data1, sample_rate!(44100)).unwrap();
    /// let audio2 = AudioSamples::new_mono(data2, sample_rate!(44100)).unwrap();
    /// let max_lag = NonZeroUsize::new(3).unwrap();
    /// let xcorr = audio1.cross_correlation(&audio2, max_lag).unwrap();
    /// assert_eq!(xcorr.len(), NonZeroUsize::new(4).unwrap()); // lags 0..=3
    /// // Peak at lag 1: (1·1 + 0·0 + (−1)·(−1)) / 3
    /// assert!((xcorr[1] - 2.0 / 3.0).abs() < 1e-12);
    /// ```
    #[inline]
    fn cross_correlation(
        &self,
        other: &Self,
        max_lag: NonZeroUsize,
    ) -> AudioSampleResult<NonEmptyVec<f64>> {
        // Verify compatible signals
        if self.num_channels() != other.num_channels() {
            return Err(crate::AudioSampleError::Parameter(
                ParameterError::invalid_value(
                    "channels",
                    "Signals must have the same number of channels for cross-correlation",
                ),
            ));
        }

        match (&self.data(), &other.data()) {
            (AudioData::Mono(arr1), AudioData::Mono(arr2)) => {
                let n1 = arr1.len();
                let n2 = arr2.len();

                let effective_max_lag: usize = max_lag.get().min(n1.get().min(n2.get()) - 1);
                let correlations = Vec::with_capacity(effective_max_lag + 1);
                // safety: we just allocated with capacity effective_max_lag + 1 which is at least 1
                // effective_max_lag is at least 0 because max_lag is NonZeroUsize and n1 and n2 are at least 1
                let mut correlations = unsafe { NonEmptyVec::new_unchecked(correlations) };

                for lag in 0..=effective_max_lag {
                    // Accumulate in f64: products/sums in `T` overflow for integers.
                    let mut correlation = 0.0f64;

                    let count = n1.get().min(n2.get() - lag);

                    for i in 0..count {
                        let s1: f64 = arr1[i].cast_into();
                        let s2: f64 = arr2[i + lag].cast_into();
                        correlation += s1 * s2;
                    }
                    correlation /= count as f64;
                    correlations.push(correlation);
                }

                Ok(correlations)
            }
            (AudioData::Multi(arr1), AudioData::Multi(arr2)) => {
                // For multi-channel, correlate the first channels
                let ch1 = arr1.row(0);
                let ch2 = arr2.row(0);

                let n1 = ch1.len();
                let n2 = ch2.len();
                let effective_max_lag: usize = max_lag.get().min(n1.min(n2) - 1);
                let correlations = Vec::with_capacity(effective_max_lag + 1);
                // safety: we just allocated with capacity effective_max_lag + 1 which is at
                // least 1
                let mut correlations = unsafe { NonEmptyVec::new_unchecked(correlations) };

                for lag in 0..=effective_max_lag {
                    let mut correlation = 0.0f64;
                    let count = n1.min(n2 - lag);

                    for i in 0..count {
                        let s1: f64 = ch1[i].cast_into();
                        let s2: f64 = ch2[i + lag].cast_into();
                        correlation += s1 * s2;
                    }
                    correlation /= count as f64;
                    correlations.push(correlation);
                }

                Ok(correlations)
            }
            _ => {
                // Mixed mono/multi-channel case is a structural mismatch, not a
                // type conversion - report it as a layout incompatibility.
                Err(crate::AudioSampleError::Layout(
                    crate::LayoutError::IncompatibleFormat {
                        operation: "cross_correlation".to_string(),
                        reason: "cannot correlate mono and multi-channel signals; \
                                 convert both inputs to the same channel layout first"
                            .to_string(),
                    },
                ))
            }
        }
    }

    /// Computes the spectral centroid of a mono signal.
    ///
    /// The power-weighted mean frequency `Σ f_k·|X[k]|² / Σ |X[k]|²`, where `X` is
    /// a single FFT of the whole (channel-reduced) signal with FFT length equal to
    /// the signal length `N` and no window (rectangular), `k = 0..=⌊N/2⌋`, and
    /// `f_k = k·fs/N` Hz. It serves as a measure of spectral brightness: higher
    /// values indicate energy concentrated at higher frequencies.
    ///
    /// Note: librosa's `spectral_centroid` is computed per STFT frame and weighted
    /// by magnitude `|X|`, so its values are generally higher than this
    /// power-weighted value and are not directly comparable.
    ///
    /// Reference: [Spectral centroid - Wikipedia](https://en.wikipedia.org/wiki/Spectral_centroid)
    ///
    /// # Arguments
    /// - `reduction` - the [`ChannelReduction`](crate::operations::types::ChannelReduction)
    ///   policy for multi-channel input.
    ///
    /// # Returns
    /// The spectral centroid frequency in Hz. Returns `0.0` when the signal
    /// is silence (zero total spectral energy).
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the signal is multi-channel and
    ///   `reduction` is [`ChannelReduction::Error`](crate::operations::types::ChannelReduction::Error),
    ///   or a [`Channel`](crate::operations::types::ChannelReduction::Channel) index
    ///   is out of bounds.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioStatistics, sample_rate, sine_wave};
    /// use audio_samples::operations::types::ChannelReduction;
    /// use std::time::Duration;
    ///
    /// // 100 ms at 44.1 kHz: bin spacing fs/N = 10 Hz, so 440 Hz lies on bin 44.
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 1.0);
    /// let centroid = audio.spectral_centroid(ChannelReduction::Error).unwrap();
    /// assert!((centroid - 440.0).abs() < 1.0);
    /// ```
    #[cfg(feature = "transforms")]
    #[inline]
    fn spectral_centroid(&self, reduction: ChannelReduction) -> AudioSampleResult<f64> {
        // Reduce to a single f64 channel per the requested policy.
        let working_vec = reduce_to_mono_f64(self, reduction, "spectral_centroid")?;

        // safety: self is guaranteed non-empty by design, so the reduced
        // channel has at least one sample.
        let working_samples = unsafe { NonEmptySlice::new_unchecked(working_vec.as_slice()) };

        // safety: working_vec is non-empty (see above).
        let n = unsafe { NonZeroUsize::new_unchecked(working_vec.len()) };
        let power_spectrum =
            with_fft_planner(|planner| planner.power_spectrum(working_samples, n, None))?;

        // Bin k of an n-point DFT is at k * sr / n (also for odd n).
        let freq_step = bin_spacing_hz(self.sample_rate_hz(), n.get());
        // Compute weighted sum and total energy
        let (weighted_sum, total_energy) = power_spectrum.iter().enumerate().fold(
            (0.0, 0.0),
            |(mut w_sum, mut t_energy), (i, &power)| {
                let frequency = i as f64 * freq_step;
                w_sum += frequency * power;
                t_energy += power;
                (w_sum, t_energy)
            },
        );

        // Compute centroid
        if total_energy > 0.0 {
            Ok(weighted_sum / total_energy)
        } else if total_energy == 0.0 {
            // A silent (zero-energy) signal has no spectral content; 0.0 Hz is
            // the conventional, well-defined result for this degenerate case.
            Ok(0.0)
        } else {
            Err(AudioSampleError::Processing(
                ProcessingError::MathematicalFailure {
                    operation: "spectral_centroid".to_string(),
                    reason: format!(
                        "Total spectral energy is negative, cannot compute centroid: total_energy={total_energy},weighted_sum={weighted_sum}"
                    ),
                },
            ))
        }
    }

    /// Computes the spectral rolloff frequency.
    ///
    /// The smallest bin frequency `f_k = k·fs/N` at which the cumulative power
    /// `Σ_{j≤k} |X[j]|²` reaches `rolloff_percent` of the total power, where `X`
    /// is a single rectangular-window FFT of the whole (channel-reduced) signal with
    /// FFT length equal to the signal length `N`. There is no interpolation
    /// between bins, so the result is a multiple of `fs/N`. It is commonly used to
    /// distinguish harmonic signals from noise-like signals.
    ///
    /// Note: librosa's `spectral_rolloff` accumulates magnitude `|X|` per STFT
    /// frame, so its values differ from this whole-signal, power-based rolloff.
    ///
    /// # Arguments
    /// - `rolloff_percent` - the energy proportion threshold. Must lie in the
    ///   open interval `(0.0, 1.0)`. A typical value is `0.85`.
    /// - `reduction` - the [`ChannelReduction`](crate::operations::types::ChannelReduction)
    ///   policy applied to multi-channel input. Pass
    ///   [`First`](crate::operations::types::ChannelReduction::First) to reproduce the
    ///   historical channel-0 behaviour.
    ///
    /// # Returns
    /// The rolloff frequency in Hz. Returns `0.0` when the signal is silence
    /// (zero total spectral energy).
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `rolloff_percent` is not in
    ///   `(0.0, 1.0)`, if the signal is multi-channel and `reduction` is
    ///   [`ChannelReduction::Error`](crate::operations::types::ChannelReduction::Error),
    ///   or if a [`Channel`](crate::operations::types::ChannelReduction::Channel) index is out of bounds.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioStatistics, sample_rate, sine_wave};
    /// use audio_samples::operations::types::ChannelReduction;
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 1.0);
    /// // All the power is in the 440 Hz bin, so 85 % is reached there.
    /// let rolloff = audio.spectral_rolloff(0.85, ChannelReduction::Error).unwrap();
    /// assert!((rolloff - 440.0).abs() < 1e-6);
    /// ```
    #[cfg(feature = "transforms")]
    #[inline]
    fn spectral_rolloff(
        &self,
        rolloff_percent: f64,
        reduction: ChannelReduction,
    ) -> AudioSampleResult<f64> {
        if rolloff_percent <= 0.0 || rolloff_percent >= 1.0 {
            return Err(crate::AudioSampleError::Parameter(
                ParameterError::out_of_range(
                    "rolloff_percent",
                    rolloff_percent.to_string(),
                    "0.0",
                    "1.0",
                    "rolloff_percent must be between 0.0 and 1.0",
                ),
            ));
        }

        // Reduce to a single f64 channel per the requested policy.
        let input_vec = reduce_to_mono_f64(self, reduction, "spectral_rolloff")?;

        // safety: self is guaranteed non-empty by design, so the reduced
        // channel has at least one sample.
        let n = unsafe { NonZeroUsize::new_unchecked(input_vec.len()) };
        // SAFETY: input_vec is the reduced channel of a non-empty signal, so it
        // has at least one element and is a valid NonEmptyVec.
        let input: NonEmptyVec<f64> = unsafe { NonEmptyVec::new_unchecked(input_vec) };

        let power_spectrum = with_fft_planner(|planner| {
            planner.power_spectrum(input.as_non_empty_slice(), n, None)
        })?;

        // Bin k of an n-point DFT is at k * sr / n (also for odd n).
        let nyquist = self.nyquist();
        let freq_step = bin_spacing_hz(self.sample_rate_hz(), n.get());

        // Compute total energy
        let total_energy: f64 = power_spectrum.iter().fold(0.0, |acc, &x| acc + x);
        if total_energy == 0.0 {
            // A silent (zero-energy) signal has no spectral content; 0.0 Hz is
            // the conventional, well-defined rolloff for this degenerate case.
            return Ok(0.0);
        } else if total_energy < 0.0 {
            return Err(AudioSampleError::Processing(
                ProcessingError::MathematicalFailure {
                    operation: "spectral_rolloff".to_string(),
                    reason: "Total spectral energy is negative, cannot compute rolloff frequency"
                        .to_string(),
                },
            ));
        }

        // Find rolloff frequency
        let target_energy = total_energy * rolloff_percent;
        let mut cumulative_energy = 0.0;

        for (i, &power) in power_spectrum.iter().enumerate() {
            cumulative_energy += power;
            if cumulative_energy >= target_energy {
                let frequency = i as f64 * freq_step;
                return Ok(frequency);
            }
        }

        // If we reach here, return Nyquist frequency
        Ok(nyquist)
    }

    /// Computes the spectral bandwidth (spectral spread).
    ///
    /// The magnitude-weighted standard deviation of frequency,
    /// `sqrt( Σ (f_k − c_M)²·M_k / Σ M_k )`, where `M_k = |X[k]|` is the magnitude
    /// of a single rectangular-window FFT of the whole (channel-reduced) signal,
    /// `f_k = k·fs/N` Hz, and `c_M = Σ f_k·M_k / Σ M_k` is a **magnitude**-weighted
    /// centroid computed internally. Because
    /// [`spectral_centroid`](crate::operations::traits::AudioStatistics::spectral_centroid)
    /// is power-weighted, the spread is not centred on that value. Larger values
    /// indicate energy spread across a wider frequency range; a single on-bin pure
    /// tone yields a value near zero.
    ///
    /// Note: this is librosa's `spectral_bandwidth` formula (`p = 2`, magnitude
    /// weights) applied to one whole-signal FFT rather than to each STFT frame.
    ///
    /// # Arguments
    /// - `reduction` - the [`ChannelReduction`](crate::operations::types::ChannelReduction)
    ///   policy for multi-channel input.
    ///
    /// # Returns
    /// The spectral bandwidth in Hz. Returns `0.0` for silence (zero total
    /// magnitude).
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the signal is multi-channel and
    ///   `reduction` is [`ChannelReduction::Error`](crate::operations::types::ChannelReduction::Error),
    ///   or a [`Channel`](crate::operations::types::ChannelReduction::Channel) index
    ///   is out of bounds.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    #[cfg(feature = "transforms")]
    #[inline]
    fn spectral_bandwidth(&self, reduction: ChannelReduction) -> AudioSampleResult<f64> {
        let working_vec = reduce_to_mono_f64(self, reduction, "spectral_bandwidth")?;
        // safety: self is non-empty by design.
        let n = unsafe { NonZeroUsize::new_unchecked(working_vec.len()) };
        // SAFETY: working_vec is the reduced channel of a non-empty signal, so it
        // has at least one element and is a valid NonEmptyVec.
        let input: NonEmptyVec<f64> = unsafe { NonEmptyVec::new_unchecked(working_vec) };
        let mag = with_fft_planner(|planner| {
            planner.magnitude_spectrum(input.as_non_empty_slice(), n, None)
        })?;

        // Bin k of an n-point DFT is at k * sr / n (also for odd n).
        let freq_step = bin_spacing_hz(self.sample_rate_hz(), n.get());

        // Magnitude-weighted centroid.
        let (weighted_sum, total_mag) =
            mag.iter().enumerate().fold((0.0, 0.0), |(w, t), (i, &m)| {
                let f = i as f64 * freq_step;
                (w + f * m, t + m)
            });
        if total_mag <= 0.0 {
            return Ok(0.0);
        }
        let centroid = weighted_sum / total_mag;

        // Weighted variance about the centroid.
        let var = mag.iter().enumerate().fold(0.0, |acc, (i, &m)| {
            let f = i as f64 * freq_step;
            let d = f - centroid;
            acc + d * d * m
        }) / total_mag;

        Ok(var.max(0.0).sqrt())
    }

    /// Computes the spectral flatness (Wiener entropy).
    ///
    /// The ratio of the geometric mean to the arithmetic mean of the power
    /// spectrum `P_k = |X[k]|²` of a single rectangular-window FFT of the whole
    /// (channel-reduced) signal: `exp(mean(ln(P_k + ε))) / mean(P_k)`, with
    /// `ε = 1e-10` so that empty bins keep the logarithm finite. The result is
    /// clamped to `[0, 1]`: values near `1` indicate noise-like (flat) spectra,
    /// while values near `0` indicate tonal spectra dominated by a few peaks.
    ///
    /// Note: librosa's `spectral_flatness` (default `power=2.0`) uses the same
    /// ratio per STFT frame, with `max(P_k, amin)` instead of `P_k + ε`.
    ///
    /// # Arguments
    /// - `reduction` - the [`ChannelReduction`](crate::operations::types::ChannelReduction)
    ///   policy for multi-channel input.
    ///
    /// # Returns
    /// The spectral flatness in `[0, 1]`. Returns `0.0` for silence.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the signal is multi-channel and
    ///   `reduction` is [`ChannelReduction::Error`](crate::operations::types::ChannelReduction::Error),
    ///   or a [`Channel`](crate::operations::types::ChannelReduction::Channel) index
    ///   is out of bounds.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    #[cfg(feature = "transforms")]
    #[inline]
    fn spectral_flatness(&self, reduction: ChannelReduction) -> AudioSampleResult<f64> {
        let working_vec = reduce_to_mono_f64(self, reduction, "spectral_flatness")?;
        // safety: self is non-empty by design.
        let working_samples = unsafe { NonEmptySlice::new_unchecked(working_vec.as_slice()) };
        // SAFETY: working_vec is the reduced channel of a non-empty signal, so its
        // length is non-zero.
        let n = unsafe { NonZeroUsize::new_unchecked(working_vec.len()) };
        let power = with_fft_planner(|planner| planner.power_spectrum(working_samples, n, None))?;

        // Small epsilon floor relative to the spectrum scale keeps log() finite
        // for empty bins without biasing a genuinely flat spectrum.
        const EPS: f64 = 1e-10;
        let count = power.len().get() as f64;
        let arith_mean = power.iter().fold(0.0, |a, &p| a + p) / count;
        if arith_mean <= 0.0 {
            return Ok(0.0);
        }
        // Geometric mean via mean of logs.
        let log_sum = power.iter().fold(0.0, |a, &p| a + (p + EPS).ln());
        let geo_mean = (log_sum / count).exp();

        Ok((geo_mean / arith_mean).clamp(0.0, 1.0))
    }

    /// Computes the spectral crest factor.
    ///
    /// The ratio of the peak magnitude to the mean magnitude of the spectrum:
    /// `max(M_k) / mean(M_k)`, with `M_k = |X[k]|` the magnitude of a single
    /// rectangular-window FFT of the whole (channel-reduced) signal. High values
    /// indicate a strongly peaked (tonal) spectrum; a flat spectrum approaches `1`.
    ///
    /// # Arguments
    /// - `reduction` - the [`ChannelReduction`](crate::operations::types::ChannelReduction)
    ///   policy for multi-channel input.
    ///
    /// # Returns
    /// The spectral crest factor (≥ 1 for non-silent signals). Returns `0.0` for
    /// silence.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the signal is multi-channel and
    ///   `reduction` is [`ChannelReduction::Error`](crate::operations::types::ChannelReduction::Error),
    ///   or a [`Channel`](crate::operations::types::ChannelReduction::Channel) index
    ///   is out of bounds.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    #[cfg(feature = "transforms")]
    #[inline]
    fn spectral_crest(&self, reduction: ChannelReduction) -> AudioSampleResult<f64> {
        let working_vec = reduce_to_mono_f64(self, reduction, "spectral_crest")?;
        // safety: self is non-empty by design.
        let n = unsafe { NonZeroUsize::new_unchecked(working_vec.len()) };
        // SAFETY: working_vec is the reduced channel of a non-empty signal, so it
        // has at least one element and is a valid NonEmptyVec.
        let input: NonEmptyVec<f64> = unsafe { NonEmptyVec::new_unchecked(working_vec) };
        let mag = with_fft_planner(|planner| {
            planner.magnitude_spectrum(input.as_non_empty_slice(), n, None)
        })?;

        let count = mag.len().get() as f64;
        let (sum, peak) = mag
            .iter()
            .fold((0.0, 0.0_f64), |(s, p), &m| (s + m, p.max(m)));
        let mean = sum / count;
        if mean <= 0.0 {
            return Ok(0.0);
        }
        Ok(peak / mean)
    }

    /// Computes the spectral slope.
    ///
    /// The slope of an ordinary least-squares linear fit of the **linear
    /// magnitude** spectrum `M_k = |X[k]|` against frequency `f_k = k·fs/N` (Hz),
    /// where `X` is a single rectangular-window, unnormalised FFT of the whole
    /// (channel-reduced) signal. Returned in units of magnitude per Hz; because the
    /// FFT is unnormalised, the value scales with the signal length. A negative
    /// slope indicates energy concentrated at low frequencies (e.g. a low-pass or
    /// pink-ish spectrum); a flat (white) spectrum yields a slope near zero.
    ///
    /// # Arguments
    /// - `reduction` - the [`ChannelReduction`](crate::operations::types::ChannelReduction)
    ///   policy for multi-channel input.
    ///
    /// # Returns
    /// The least-squares slope (magnitude per Hz). Returns `0.0` for silence or a
    /// degenerate single-bin spectrum.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the signal is multi-channel and
    ///   `reduction` is [`ChannelReduction::Error`](crate::operations::types::ChannelReduction::Error),
    ///   or a [`Channel`](crate::operations::types::ChannelReduction::Channel) index
    ///   is out of bounds.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    #[cfg(feature = "transforms")]
    #[inline]
    fn spectral_slope(&self, reduction: ChannelReduction) -> AudioSampleResult<f64> {
        let working_vec = reduce_to_mono_f64(self, reduction, "spectral_slope")?;
        // safety: self is non-empty by design.
        let n = unsafe { NonZeroUsize::new_unchecked(working_vec.len()) };
        // SAFETY: working_vec is the reduced channel of a non-empty signal, so it
        // has at least one element and is a valid NonEmptyVec.
        let input: NonEmptyVec<f64> = unsafe { NonEmptyVec::new_unchecked(working_vec) };
        let mag = with_fft_planner(|planner| {
            planner.magnitude_spectrum(input.as_non_empty_slice(), n, None)
        })?;

        let len = mag.len().get();
        if len < 2 {
            return Ok(0.0);
        }
        if mag.iter().all(|&m| m <= 0.0) {
            return Ok(0.0);
        }

        // Bin k of an n-point DFT is at k * sr / n (also for odd n).
        let freq_step = bin_spacing_hz(self.sample_rate_hz(), n.get());

        // OLS slope: Σ (x - x̄)(y - ȳ) / Σ (x - x̄)².
        let count = len as f64;
        let (sum_x, sum_y) = mag
            .iter()
            .enumerate()
            .fold((0.0, 0.0), |(sx, sy), (i, &m)| {
                (sx + i as f64 * freq_step, sy + m)
            });
        let mean_x = sum_x / count;
        let mean_y = sum_y / count;
        let (num, den) = mag
            .iter()
            .enumerate()
            .fold((0.0, 0.0), |(num, den), (i, &m)| {
                let dx = i as f64 * freq_step - mean_x;
                (num + dx * (m - mean_y), den + dx * dx)
            });
        if den <= 0.0 {
            return Ok(0.0);
        }
        Ok(num / den)
    }

    /// Computes spectral contrast across logarithmically spaced sub-bands (after
    /// librosa).
    ///
    /// The Hann-windowed magnitude spectrum `M_k` of the whole (channel-reduced)
    /// signal has `B = ⌊N/2⌋ + 1` bins. Bin 0 (DC) is excluded and the bins
    /// `1..B` are split into `n_bands` bands with geometrically spaced edges
    /// `e_b = ⌊r^b⌋`, `r = B^(1/n_bands)`; the last band ends at the Nyquist bin.
    /// Each band therefore spans `log2(r)` octaves, which is exactly one octave
    /// only when `B = 2^n_bands`. Within a band the values `10·log10(M_k + 1e-10)`
    /// are sorted and the contrast is the mean of the top 2 % minus the mean of the
    /// bottom 2 % (rounded, at least one bin each). High contrast indicates clear
    /// tonal/harmonic structure; low contrast indicates noise-like content.
    ///
    /// Note: like librosa, this applies `10·log10` to **magnitudes**, so each value
    /// is half the level difference in dB (multiply by 2 for a `20·log10`
    /// difference). Unlike librosa, the band edges are fixed in bins from bin 1 to
    /// Nyquist (librosa uses octaves above 200 Hz), and one value per band is
    /// returned for the whole signal rather than per frame.
    ///
    /// # Arguments
    /// - `n_bands` - the number of log-spaced sub-bands.
    /// - `reduction` - the [`ChannelReduction`](crate::operations::types::ChannelReduction)
    ///   policy for multi-channel input.
    ///
    /// # Returns
    /// A vector of `n_bands` contrast values (`10·log10` of a magnitude ratio),
    /// low band to high band. Empty bands yield `0.0`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the signal is multi-channel and
    ///   `reduction` is [`ChannelReduction::Error`](crate::operations::types::ChannelReduction::Error),
    ///   or a [`Channel`](crate::operations::types::ChannelReduction::Channel) index
    ///   is out of bounds.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    #[cfg(feature = "transforms")]
    #[inline]
    fn spectral_contrast(
        &self,
        n_bands: NonZeroUsize,
        reduction: ChannelReduction,
    ) -> AudioSampleResult<Vec<f64>> {
        let working_vec = reduce_to_mono_f64(self, reduction, "spectral_contrast")?;
        // safety: self is non-empty by design.
        let n = unsafe { NonZeroUsize::new_unchecked(working_vec.len()) };
        // SAFETY: working_vec is the reduced channel of a non-empty signal, so it
        // has at least one element and is a valid NonEmptyVec.
        let input: NonEmptyVec<f64> = unsafe { NonEmptyVec::new_unchecked(working_vec) };
        // Hann window before the FFT: reduces spectral leakage so a tonal band's
        // peak/valley gap is not flattened by sidelobes (librosa windows too).
        let mag = with_fft_planner(|planner| {
            planner.magnitude_spectrum(
                input.as_non_empty_slice(),
                n,
                Some(spectrograms::WindowType::Hanning),
            )
        })?;

        let bins = mag.len().get();
        let n_bands = n_bands.get();

        // Octave-spaced band edges over bin indices [1, bins): edge_b grows
        // geometrically so each band spans roughly one octave. Bin 0 (DC) is
        // excluded.
        // First non-DC bin: index 1, clamped into the available bin range so it
        // never exceeds the last valid bin when the spectrum is tiny.
        let lo_bin = 1usize.clamp(1, bins.max(1));
        let hi_bin = bins; // exclusive
        let span = (hi_bin as f64 / lo_bin as f64).max(1.0);
        let ratio = span.powf(1.0 / n_bands as f64);

        // Quantile fraction for peak/valley pooling (librosa default 0.02).
        const QUANTILE: f64 = 0.02;
        // dB floor so log10 of an empty bin stays finite.
        const EPS: f64 = 1e-10;

        let mut out = Vec::with_capacity(n_bands);
        for b in 0..n_bands {
            let start = (lo_bin as f64 * ratio.powi(b as i32)).floor() as usize;
            let mut end = (lo_bin as f64 * ratio.powi(b as i32 + 1)).floor() as usize;
            let start = start.min(bins);
            if b == n_bands - 1 {
                end = bins;
            }
            let end = end.clamp(start, bins);

            if end <= start {
                out.push(0.0);
                continue;
            }

            let mut band: Vec<f64> = mag[start..end]
                .iter()
                .map(|&m| 10.0 * (m + EPS).log10())
                .collect();
            band.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let len = band.len();
            let k = ((len as f64 * QUANTILE).round() as usize).clamp(1, len);
            let valley: f64 = band[..k].iter().sum::<f64>() / k as f64;
            let peak: f64 = band[len - k..].iter().sum::<f64>() / k as f64;
            out.push(peak - valley);
        }

        Ok(out)
    }
}

/// Spacing in Hz between the bins of an `n_fft`-point DFT: `sample_rate / n_fft`.
///
/// Bin `k` lies at `k * sample_rate / n_fft` for even and odd `n_fft` alike
/// (`numpy.fft.rfftfreq`, `librosa.fft_frequencies`). The spectral scalars once
/// used `nyquist / (bins - 1)`, which is only equal for even `n_fft`.
#[cfg(feature = "transforms")]
fn bin_spacing_hz(sample_rate: f64, n_fft: usize) -> f64 {
    sample_rate / n_fft as f64
}

/// Returns the signed lag (in samples) that maximises the cross-correlation
/// between `reference` and `query`.
///
/// The cross-correlation is `c[L] = Σ_n reference[n]·query[n − L]`, computed as
/// `IFFT(R·conj(Q))` with zero padding to `≥ len_ref + len_query − 1` so it is
/// linear, not circular. Lags `L ∈ [−M, M]` are searched, with
/// `M = min(max_lag, len_ref − 1)`. Sign convention:
///
/// - **Positive** `L`: `query` **leads** `reference`, i.e. the reference is the
///   query delayed by `L` samples (`reference[n] ≈ query[n − L]`). Delay `query`
///   by `L` samples to align it.
/// - **Negative** `L`: `query` is **delayed** by `−L` samples relative to
///   `reference` (`query[n] ≈ reference[n + L]`). Advance `query` by `−L`
///   samples to align it.
///
/// Only the first channel of each signal is used. Samples are raw values
/// without mean removal, so a large DC offset (e.g. `u8` audio centred on 128)
/// biases the result towards lag 0: convert or remove the DC offset first.
///
/// # Arguments
/// - `reference` - the reference signal.
/// - `query` - the signal whose delay relative to `reference` is estimated.
/// - `max_lag` - the largest absolute lag searched, in samples.
///
/// # Returns
/// `Some(lag)` in samples (sign convention above), or `None` if the FFT fails.
///
/// # Examples
/// ```
/// use audio_samples::{AudioSamples, fft_alignment_lag, sample_rate};
/// use ndarray::Array1;
///
/// // A chirp has a single sharp correlation peak.
/// let reference: Vec<f64> = (0..64).map(|i| (0.02 * (i * i) as f64).sin()).collect();
/// // query = reference delayed by 5 samples
/// let mut query = vec![0.0; 5];
/// query.extend_from_slice(&reference[..59]);
/// let r = AudioSamples::new_mono(Array1::from(reference), sample_rate!(8000)).unwrap();
/// let q = AudioSamples::new_mono(Array1::from(query), sample_rate!(8000)).unwrap();
/// assert_eq!(fft_alignment_lag(&r, &q, 16), Some(-5));
/// ```
#[cfg(feature = "transforms")]
pub fn fft_alignment_lag<T: StandardSample>(
    reference: &AudioSamples<'_, T>,
    query: &AudioSamples<'_, T>,
    max_lag: usize,
) -> Option<i64> {
    let extract = |audio: &AudioSamples<'_, T>| -> Vec<f64> {
        match &audio.data() {
            AudioData::Mono(arr) => arr.iter().map(|&x| x.cast_into()).collect(),
            AudioData::Multi(arr) => arr.row(0).iter().map(|&x| x.cast_into()).collect(),
        }
    };

    let ref_sig = extract(reference);
    let deg_sig = extract(query);
    let n1 = ref_sig.len();
    let n2 = deg_sig.len();
    if n1 == 0 || n2 == 0 {
        return None;
    }

    let fft_size = (n1 + n2 - 1).next_power_of_two();
    // SAFETY: n1 and n2 are both non-zero (checked above), so n1 + n2 - 1 >= 1
    // and its next_power_of_two() is therefore non-zero.
    let fft_nz = unsafe { NonZeroUsize::new_unchecked(fft_size) };

    let mut ref_buf = ref_sig;
    ref_buf.resize(fft_size, 0.0);
    let mut deg_buf = deg_sig;
    deg_buf.resize(fft_size, 0.0);

    let xcorr = with_fft_planner(|planner| {
        // SAFETY: ref_buf was resized to fft_size (>= 1), so it is non-empty.
        let ref_spec = planner
            .fft(unsafe { NonEmptySlice::new_unchecked(&ref_buf) }, fft_nz)
            .ok()?;
        // SAFETY: deg_buf was resized to fft_size (>= 1), so it is non-empty.
        let deg_spec = planner
            .fft(unsafe { NonEmptySlice::new_unchecked(&deg_buf) }, fft_nz)
            .ok()?;

        let cross: Vec<Complex<f64>> = ref_spec
            .iter()
            .zip(deg_spec.iter())
            .map(|(r, d)| r * d.conj())
            .collect();

        // SAFETY: cross has one element per rfft bin (fft_size/2 + 1 >= 1 since
        // fft_size >= 1), so it is non-empty.
        planner
            .irfft(unsafe { NonEmptySlice::new_unchecked(&cross) }, fft_nz)
            .ok()
    })?;

    let scale = 1.0 / fft_size as f64;
    let search = max_lag.min(n1.saturating_sub(1));
    let mut best_val = f64::NEG_INFINITY;
    let mut best_lag = 0i64;

    for k in 0..=search {
        let v = xcorr[k] * scale;
        if v > best_val {
            best_val = v;
            best_lag = k as i64;
        }
    }
    for k in 1..=search {
        let v = xcorr[fft_size - k] * scale;
        if v > best_val {
            best_val = v;
            best_lag = -(k as i64);
        }
    }

    Some(best_lag)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "transforms")]
    use std::time::Duration;

    use super::*;
    use crate::sample_rate;
    use approx_eq::assert_approx_eq;
    use ndarray::{Array1, array};

    #[test]
    fn test_midpoint_sample_odd_and_even() {
        // Odd length: single central sample at index len/2 (3 / 2 == 1 -> value 3.0).
        let odd = AudioSamples::new_mono(array![1.0f32, 3.0, 5.0], sample_rate!(44100)).unwrap();
        assert_eq!(odd.midpoint_sample(), Some(3.0));

        // Even length: average of the two central samples.
        let even =
            AudioSamples::new_mono(array![1.0f32, 3.0, 5.0, 7.0], sample_rate!(44100)).unwrap();
        // Central indices 1 and 2: (3.0 + 5.0) / 2.0 = 4.0
        assert_eq!(even.midpoint_sample(), Some(4.0));

        // Multi-channel returns None.
        let stereo = AudioSamples::new_multi_channel(
            ndarray::array![[1.0f32, 2.0], [3.0, 4.0]],
            sample_rate!(44100),
        )
        .unwrap();
        assert_eq!(stereo.midpoint_sample(), None);
    }

    #[test]
    fn test_peak_min_max_existing_methods() {
        let data = array![-3.0f32, -1.0, 0.0, 2.0, 4.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        // These should use the existing native implementations
        assert_eq!(audio.peak(), 4.0);
        assert_eq!(audio.min_sample(), -3.0);
        assert_eq!(audio.max_sample(), 4.0);
    }

    #[test]
    fn test_rms_computation() {
        // Simple test case where we can verify RMS manually
        let data = array![1.0f32, -1.0, 1.0, -1.0];
        let audio: AudioSamples<'static, f32> =
            AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let rms = audio.rms(); // RMS of [1, -1, 1, -1] = sqrt((1^2 + 1^2 + 1^2 + 1^2)/4) = sqrt(1) = 1.0
        assert_approx_eq!(rms, 1.0, 1e-6);
    }

    #[test]
    fn test_variance_and_std_dev() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio: AudioSamples<'_, f32> =
            AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let variance = audio.variance();
        let std_dev = audio.std_dev();

        // Mean = 3.0, variance = mean((1-3)^2 + (2-3)^2 + ... + (5-3)^2) = mean(4+1+0+1+4) = 2.0
        assert_approx_eq!(variance, 2.0, 1e-6);
        assert_approx_eq!(std_dev, 2.0_f64.sqrt(), 1e-6);
    }

    #[test]
    fn test_zero_crossings() {
        let data = array![1.0f32, -1.0, 1.0, -1.0, 1.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let crossings = audio.zero_crossings();
        // Crossings occur at: 1->-1, -1->1, 1->-1, -1->1 = 4 crossings
        assert_eq!(crossings, 4);
    }

    #[test]
    fn test_zero_crossing_rate() {
        // Create 1 second of 4 Hz square wave: 1,-1,1,-1 pattern every 1/4 second
        let sample_rate = 44100;
        let duration = 1.0; // 1 second
        let freq = 4.0; // 4 Hz

        let n_samples = (sample_rate as f64 * duration) as usize;
        let mut data = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let t = i as f64 / sample_rate as f64;
            // Square wave using sine wave sign: +1 when sin > 0, -1 when sin <= 0
            let phase = 2.0 * std::f64::consts::PI * freq * t;
            let value = if phase.sin() >= 0.0 { 1.0 } else { -1.0 };
            data.push(value);
        }

        let audio = AudioSamples::new_mono(Array1::from(data), sample_rate!(44100)).unwrap();
        let zcr = audio.zero_crossing_rate();

        // 4 Hz square wave has ~8 zero crossings per second (2 per cycle)
        // Due to discrete sampling, we might get 7-8 crossings
        assert!(
            (zcr - 8.0f64).abs() <= 1.0,
            "Expected ~8 crossings/sec, got {}",
            zcr
        );
    }

    #[test]
    #[cfg(feature = "transforms")]
    fn test_autocorrelation() {
        let data = array![1.0f32, 0.0, -1.0, 0.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let autocorr = audio.autocorrelation(crate::nzu!(2)).unwrap();

        // Should have correlations for lags 0, 1, 2
        assert_eq!(autocorr.len(), crate::nzu!(3));

        // Lag 0 should be the highest (signal correlated with itself)
        assert!(autocorr[0] >= autocorr[1]);
        assert!(autocorr[0] >= autocorr[2]);

        // Exact values: r[k] = (1 / (n - k)) Σ x[j] x[j+k].
        assert!((autocorr[0] - 0.5).abs() < 1e-12); // (1 + 1) / 4
        assert!(autocorr[1].abs() < 1e-12);
        assert!((autocorr[2] + 0.5).abs() < 1e-12); // (1 * -1) / 2
    }

    /// Regression: on multi-channel audio the rate summed the crossings of all
    /// channels but divided by the single-channel duration, so two identical
    /// channels reported twice the mono rate.
    #[test]
    fn zero_crossing_rate_is_averaged_over_channels() {
        let mono_data = array![1.0f32, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0];
        let mono = AudioSamples::new_mono(mono_data.clone(), sample_rate!(8)).unwrap();
        let mut stereo_data = ndarray::Array2::zeros((2, mono_data.len()));
        stereo_data.row_mut(0).assign(&mono_data);
        stereo_data.row_mut(1).assign(&mono_data);
        let stereo = AudioSamples::new_multi_channel(stereo_data, sample_rate!(8)).unwrap();

        // 5 crossings in 1 s.
        assert_eq!(mono.zero_crossings(), 5);
        assert!((mono.zero_crossing_rate() - 5.0).abs() < 1e-12);
        // The count is still the total over channels; the rate is per channel.
        assert_eq!(stereo.zero_crossings(), 10);
        assert!((stereo.zero_crossing_rate() - 5.0).abs() < 1e-12);

        // Different channels: the mean of the per-channel rates.
        let mixed = AudioSamples::new_multi_channel(
            array![[1.0f32, -1.0, 1.0, -1.0], [1.0, 1.0, 1.0, -1.0]],
            sample_rate!(4),
        )
        .unwrap();
        assert!((mixed.zero_crossing_rate() - 2.0).abs() < 1e-12); // (3 + 1) / 2
    }

    /// Regression: the spectral scalars put bin k at `k * nyquist / (bins - 1)`,
    /// which is `k * sr / n` only for even n (odd n gave sr/(n+1) in the centroid
    /// and sr/(n-1) in rolloff, bandwidth and slope). Tones exactly on bins 100
    /// and 200 of an odd-length (n = 2205) signal at 22.05 kHz sit at 1000 and
    /// 2000 Hz.
    #[test]
    #[cfg(feature = "transforms")]
    fn spectral_scalars_use_exact_bin_frequencies_for_odd_length() {
        for n in [2205usize, 4410] {
            let sr = 22050.0;
            let step = sr / n as f64;
            let tone = |k: f64| -> Vec<f64> {
                (0..n)
                    .map(|i| (2.0 * std::f64::consts::PI * k * i as f64 / n as f64).sin())
                    .collect()
            };
            let one =
                AudioSamples::new_mono(Array1::from(tone(100.0)), sample_rate!(22050)).unwrap();
            let centroid = one.spectral_centroid(ChannelReduction::First).unwrap();
            assert!(
                (centroid - 100.0 * step).abs() < 1e-6,
                "n {n}: centroid {centroid}"
            );
            let rolloff = one.spectral_rolloff(0.5, ChannelReduction::First).unwrap();
            assert!(
                (rolloff - 100.0 * step).abs() < 1e-6,
                "n {n}: rolloff {rolloff}"
            );

            let two: Vec<f64> = tone(100.0)
                .iter()
                .zip(tone(200.0))
                .map(|(a, b)| a + b)
                .collect();
            let two = AudioSamples::new_mono(Array1::from(two), sample_rate!(22050)).unwrap();
            let bandwidth = two.spectral_bandwidth(ChannelReduction::First).unwrap();
            assert!(
                (bandwidth - 50.0 * step).abs() < 1e-6,
                "n {n}: bandwidth {bandwidth}"
            );
        }
    }

    /// Regression: the values were divided by the zero-padded FFT size twice (the
    /// planner's inverse FFT already normalises), so they were too small by a
    /// factor of 2048 for n = 1000 and 8192 for n = 4096. `r[k] * (n - k)` must
    /// equal the direct sum of products (`numpy.correlate`, `librosa.autocorrelate`).
    #[test]
    #[cfg(feature = "transforms")]
    fn autocorrelation_matches_direct_sum() {
        for (n, max_lag) in [
            (1000usize, 300),
            (4096, 300),
            (777, 300),
            (20_000, 300),
            (50, 49),
            (3000, 2999),
        ] {
            let x: Vec<f64> = (0..n)
                .map(|i| {
                    let t = i as f64;
                    (0.05 * t).sin() + 0.3 * (0.31 * t + 1.0).cos() + 0.01 * (t % 7.0)
                })
                .collect();
            let audio =
                AudioSamples::new_mono(Array1::from(x.clone()), sample_rate!(22050)).unwrap();
            let r = audio
                .autocorrelation(NonZeroUsize::new(max_lag).unwrap())
                .unwrap();
            assert_eq!(r.len().get(), max_lag + 1);
            let scale = x.iter().map(|v| v * v).sum::<f64>();
            for k in 0..=max_lag {
                let direct: f64 = (0..n - k).map(|j| x[j] * x[j + k]).sum();
                let got = r[k] * (n - k) as f64;
                assert!(
                    (got - direct).abs() <= 1e-10 * scale,
                    "n {n}, lag {k}: {got} vs {direct}"
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "transforms")]
    fn test_cross_correlation() {
        let data1 = array![1.0f32, 0.0, -1.0];
        let data2 = array![1.0f32, 0.0, -1.0]; // Same signal
        let audio1 = AudioSamples::new_mono(data1, sample_rate!(44100)).unwrap();
        let audio2 = AudioSamples::new_mono(data2, sample_rate!(44100)).unwrap();

        let cross_corr = audio1.cross_correlation(&audio2, crate::nzu!(1)).unwrap();

        // Cross-correlation of identical signals should be same as autocorrelation
        let autocorr = audio1.autocorrelation(crate::nzu!(1)).unwrap();
        assert_eq!(cross_corr.len(), autocorr.len());
    }

    #[test]
    fn test_multi_channel_statistics() {
        let data = array![[1.0f32, 2.0], [-1.0, 1.0]]; // 2 channels, 2 samples each
        let audio = AudioSamples::new_multi_channel(data, sample_rate!(44100)).unwrap();

        let rms: f64 = audio.rms();
        let variance = audio.variance();
        let crossings = audio.zero_crossings();

        // Should compute across all samples
        assert!(rms > 0.0);
        assert!(variance >= 0.0);
        assert_eq!(crossings, 1); // One crossing in channel 0: 1.0 -> -1.0
    }

    #[test]
    fn test_edge_cases() {
        // Single sample
        let single_data = array![1.0f32];
        let single_audio = AudioSamples::new_mono(single_data, sample_rate!(44100)).unwrap();

        assert_eq!(single_audio.zero_crossings(), 0);
        assert_eq!(
            single_audio.rms(),
            1.0,
            "RMS of single sample should be the sample itself"
        );
    }

    #[test]
    fn test_empty_audio_rejected() {
        // Empty audio should be rejected at construction time
        let empty_data: Array1<f32> = Array1::from(vec![]);
        let empty_audio = AudioSamples::new_mono(empty_data, sample_rate!(44100));
        assert!(empty_audio.is_err(), "Empty audio should not be created");
    }

    #[test]
    #[cfg(feature = "transforms")]
    fn test_spectral_centroid() {
        // Test with a simple sine wave that should have energy concentrated at a specific frequency
        let sample_rate = sample_rate!(44100);
        let duration = Duration::from_secs_f32(1.0); // 1 second
        let freq = 1000.0; // 1kHz sine wave

        let audio = crate::sine_wave::<f64>(freq, duration, sample_rate, 0.5);

        // Generate 1kHz sine wave
        let centroid = audio
            .spectral_centroid(ChannelReduction::Error)
            .expect("Failed to compute spectral centroid");

        // For a pure sine wave, the spectral centroid should be close to the frequency
        // Allow some tolerance due to FFT discretization and numerical precision
        assert!(
            (centroid - freq).abs() < 50.0,
            "Centroid {} should be close to {}",
            centroid,
            freq
        );
    }

    #[test]
    #[cfg(all(feature = "transforms", feature = "random-generation"))]
    fn test_spectral_rolloff() {
        // Test with white noise - rolloff should be around 85% of Nyquist for 85% rolloff
        let sample_rate = sample_rate!(8000); // Use lower sample rate for faster test
        let duration = Duration::from_secs_f32(1.0);
        // Generate white noise
        let audio = crate::white_noise::<f64>(duration, sample_rate, 0.5, None);

        let rolloff = audio
            .spectral_rolloff(0.85, ChannelReduction::Error)
            .expect("Failed to compute spectral rolloff");
        let nyquist = audio.nyquist();

        // For noise-like signals, rolloff should be somewhere reasonable
        assert!(rolloff > 0.0, "Rolloff should be positive");
        assert!(
            rolloff <= nyquist,
            "Rolloff should not exceed Nyquist frequency"
        );
    }

    #[test]
    #[cfg(feature = "transforms")]
    fn test_spectral_rolloff_validation() {
        let data = array![1.0f32, -1.0, 1.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        // Test invalid rolloff percentages
        assert!(
            audio
                .spectral_rolloff(0.0, ChannelReduction::Error)
                .is_err()
        );
        assert!(
            audio
                .spectral_rolloff(1.0, ChannelReduction::Error)
                .is_err()
        );
        assert!(
            audio
                .spectral_rolloff(-0.1, ChannelReduction::Error)
                .is_err()
        );
        assert!(
            audio
                .spectral_rolloff(1.1, ChannelReduction::Error)
                .is_err()
        );

        // Test valid rolloff percentage
        assert!(
            audio
                .spectral_rolloff(0.85, ChannelReduction::Error)
                .is_ok()
        );
    }

    #[test]
    #[cfg(feature = "transforms")]
    fn test_spectral_centroid_channel_reduction() {
        // Stereo signal: channel 0 is a 1 kHz tone, channel 1 is silence.
        let sample_rate = sample_rate!(44100);
        let duration = Duration::from_secs_f32(0.25);
        let tone = crate::sine_wave::<f64>(1000.0, duration, sample_rate, 0.5);
        let tone_ch = tone.as_mono().expect("mono tone");
        let n = tone_ch.len().get();
        let tone_vec: Vec<f64> = (0..n).map(|i| tone[i]).collect();

        let mut data = ndarray::Array2::<f64>::zeros((2, n));
        for (i, &v) in tone_vec.iter().enumerate() {
            data[[0, i]] = v;
        }
        let stereo = AudioSamples::new_multi_channel(data, sample_rate).unwrap();

        // Error: multi-channel must be rejected by default.
        assert!(stereo.spectral_centroid(ChannelReduction::Error).is_err());

        // First: uses channel 0 (the tone) -> centroid near 1 kHz.
        let first = stereo
            .spectral_centroid(ChannelReduction::First)
            .expect("First channel centroid");
        assert!(
            (first - 1000.0).abs() < 50.0,
            "First-channel centroid {first} should be near 1000 Hz"
        );

        // Channel(0) must match First.
        let ch0 = stereo
            .spectral_centroid(ChannelReduction::Channel(0))
            .expect("Channel(0) centroid");
        assert!((ch0 - first).abs() < 1e-6);

        // Channel(1) is silence -> zero-energy -> 0.0.
        let ch1 = stereo
            .spectral_centroid(ChannelReduction::Channel(1))
            .expect("Channel(1) centroid");
        assert_eq!(ch1, 0.0);

        // Out-of-range channel errors.
        assert!(
            stereo
                .spectral_centroid(ChannelReduction::Channel(2))
                .is_err()
        );

        // Average: mean of the tone and silence -> still tonal, centroid near 1 kHz.
        let avg = stereo
            .spectral_centroid(ChannelReduction::Average)
            .expect("Average centroid");
        assert!(
            (avg - 1000.0).abs() < 50.0,
            "Averaged centroid {avg} should be near 1000 Hz"
        );
    }

    /// Sanity check that the thread-local cached [`FftPlanner`] yields
    /// bit-identical results across repeated calls of the same size - i.e.
    /// reusing the planner did not corrupt or mutate its cached state in a way
    /// that changes the numeric output.
    #[test]
    #[cfg(feature = "transforms")]
    fn test_cached_fft_planner_repeatable() {
        let sample_rate = sample_rate!(44100);
        let duration = Duration::from_secs_f32(0.1);
        let tone = crate::sine_wave::<f64>(1000.0, duration, sample_rate, 0.5);

        // spectral_centroid uses the cached planner (power_spectrum path).
        let c1 = tone.spectral_centroid(ChannelReduction::First).unwrap();
        let c2 = tone.spectral_centroid(ChannelReduction::First).unwrap();
        assert_eq!(c1, c2, "centroid must be identical across cached calls");

        // autocorrelation uses the cached planner (fft + irfft path).
        let a1 = tone.autocorrelation(nzu!(64)).unwrap();
        let a2 = tone.autocorrelation(nzu!(64)).unwrap();
        assert_eq!(
            a1.as_slice(),
            a2.as_slice(),
            "autocorrelation must be identical across cached calls"
        );

        // fft_alignment_lag uses the cached planner (two ffts + irfft path).
        let l1 = fft_alignment_lag(&tone, &tone, 32);
        let l2 = fft_alignment_lag(&tone, &tone, 32);
        assert_eq!(
            l1, l2,
            "alignment lag must be identical across cached calls"
        );
    }

    // ---- New spectral feature validation ----

    /// White noise should have flatness near 1; a pure sine near 0 with high crest.
    #[test]
    #[cfg(all(feature = "transforms", feature = "random-generation"))]
    fn test_spectral_flatness_noise_vs_tone() {
        let sr = sample_rate!(16000);
        let dur = Duration::from_secs_f32(1.0);

        let noise = crate::white_noise::<f64>(dur, sr, 0.5, None);
        let flat_noise = noise.spectral_flatness(ChannelReduction::Error).unwrap();
        assert!(
            flat_noise > 0.5,
            "white-noise flatness {flat_noise} should be > 0.5"
        );

        let tone = crate::sine_wave::<f64>(1000.0, dur, sr, 0.5);
        let flat_tone = tone.spectral_flatness(ChannelReduction::Error).unwrap();
        assert!(
            flat_tone < 0.1,
            "pure-tone flatness {flat_tone} should be < 0.1"
        );

        // A pure tone is strongly peaked -> high crest; noise is comparatively low.
        let crest_tone = tone.spectral_crest(ChannelReduction::Error).unwrap();
        let crest_noise = noise.spectral_crest(ChannelReduction::Error).unwrap();
        assert!(
            crest_tone > crest_noise,
            "tone crest {crest_tone} should exceed noise crest {crest_noise}"
        );
        assert!(crest_tone > 10.0, "tone crest {crest_tone} should be large");
    }

    /// A two-tone signal has larger spectral bandwidth than a single tone.
    #[test]
    #[cfg(feature = "transforms")]
    fn test_spectral_bandwidth_two_tone_vs_one() {
        let sr = sample_rate!(16000);
        let n = 16000usize;
        let single: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr.get() as f64;
                0.5 * (2.0 * std::f64::consts::PI * 1000.0 * t).sin()
            })
            .collect();
        let two: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr.get() as f64;
                0.25 * (2.0 * std::f64::consts::PI * 500.0 * t).sin()
                    + 0.25 * (2.0 * std::f64::consts::PI * 5000.0 * t).sin()
            })
            .collect();

        let single = AudioSamples::new_mono(Array1::from(single), sr).unwrap();
        let two = AudioSamples::new_mono(Array1::from(two), sr).unwrap();

        let bw1 = single.spectral_bandwidth(ChannelReduction::Error).unwrap();
        let bw2 = two.spectral_bandwidth(ChannelReduction::Error).unwrap();
        assert!(
            bw2 > bw1,
            "two-tone bandwidth {bw2} should exceed single-tone bandwidth {bw1}"
        );
    }

    /// A low-pass-weighted (energy concentrated low) spectrum has negative slope;
    /// flat/white is near zero.
    #[test]
    #[cfg(all(feature = "transforms", feature = "random-generation"))]
    fn test_spectral_slope_lowpass_vs_white() {
        let sr = sample_rate!(16000);
        let dur = Duration::from_secs_f32(1.0);

        // Low-frequency tone: magnitude concentrated near 0 Hz -> negative slope.
        let low = crate::sine_wave::<f64>(200.0, dur, sr, 0.5);
        let slope_low = low.spectral_slope(ChannelReduction::Error).unwrap();
        assert!(
            slope_low < 0.0,
            "low-frequency-weighted slope {slope_low} should be negative"
        );

        // White noise: roughly flat magnitude -> slope near zero. Use a generous
        // bound since noise realisations vary.
        let noise = crate::white_noise::<f64>(dur, sr, 0.5, None);
        let slope_white = noise.spectral_slope(ChannelReduction::Error).unwrap();
        assert!(
            slope_white.abs() < slope_low.abs(),
            "white-noise slope {slope_white} should be flatter than low-tone slope {slope_low}"
        );
    }

    /// Spectral contrast returns one value per band and is higher for a tonal
    /// signal than for noise.
    #[test]
    #[cfg(all(feature = "transforms", feature = "random-generation"))]
    fn test_spectral_contrast_shape_and_tonality() {
        let sr = sample_rate!(16000);
        let dur = Duration::from_secs_f32(1.0);

        let tone = crate::sine_wave::<f64>(1000.0, dur, sr, 0.5);
        let contrast = tone
            .spectral_contrast(nzu!(4), ChannelReduction::Error)
            .unwrap();
        assert_eq!(contrast.len(), 4, "one contrast value per band");
        assert!(
            contrast.iter().all(|&c| c.is_finite() && c >= 0.0),
            "contrast values must be finite and non-negative: {contrast:?}"
        );
        // A tone produces a strong peak/valley gap in at least one band.
        let max_tone = contrast.iter().cloned().fold(0.0_f64, f64::max);
        assert!(
            max_tone > 1.0,
            "tonal signal should yield a band contrast > 1 dB, got {contrast:?}"
        );
    }

    /// ChannelReduction policies on a stereo signal for the new features.
    #[test]
    #[cfg(feature = "transforms")]
    fn test_new_features_channel_reduction() {
        let sr = sample_rate!(44100);
        let dur = Duration::from_secs_f32(0.25);
        let tone = crate::sine_wave::<f64>(1000.0, dur, sr, 0.5);
        let tone_ch = tone.as_mono().expect("mono tone");
        let n = tone_ch.len().get();
        let tone_vec: Vec<f64> = (0..n).map(|i| tone[i]).collect();

        // Channel 0 = tone, channel 1 = silence.
        let mut data = ndarray::Array2::<f64>::zeros((2, n));
        for (i, &v) in tone_vec.iter().enumerate() {
            data[[0, i]] = v;
        }
        let stereo = AudioSamples::new_multi_channel(data, sr).unwrap();

        // Error rejects multi-channel.
        assert!(stereo.spectral_bandwidth(ChannelReduction::Error).is_err());
        assert!(stereo.spectral_flatness(ChannelReduction::Error).is_err());
        assert!(stereo.spectral_crest(ChannelReduction::Error).is_err());
        assert!(stereo.spectral_slope(ChannelReduction::Error).is_err());
        assert!(
            stereo
                .spectral_contrast(nzu!(4), ChannelReduction::Error)
                .is_err()
        );

        // First selects the tone channel -> tonal (low flatness).
        let flat_first = stereo.spectral_flatness(ChannelReduction::First).unwrap();
        assert!(
            flat_first < 0.1,
            "First-channel flatness {flat_first} should be tonal"
        );

        // Average mixes tone with silence -> still tonal.
        let flat_avg = stereo.spectral_flatness(ChannelReduction::Average).unwrap();
        assert!(
            flat_avg < 0.2,
            "Average flatness {flat_avg} should be tonal"
        );
    }
}
