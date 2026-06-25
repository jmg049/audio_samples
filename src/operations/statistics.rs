//! Statistical analysis operations for [`AudioSamples`].
//!
//! This module implements the [`AudioStatistics`] trait, providing statistical
//! and signal analysis methods for mono and multi-channel audio data. Operations
//! cover both time-domain statistics and, when the `fft` feature is enabled,
//! frequency-domain descriptors computed via the [`spectrograms`] crate.
//!
//! Statistical measures are a core part of audio analysis. Isolating them into a
//! single trait keeps the [`AudioSamples`] API organised and lets users access
//! only the statistical surface they need. This module is the sole implementor of
//! [`AudioStatistics`] for [`AudioSamples`].
//!
//! All operations are available on any [`AudioSamples<T>`] where `T` is a
//! supported sample type (`u8`, `i16`, `I24`, `i32`, `f32`, `f64`). Import
//! [`AudioStatistics`] and call methods on your audio directly.
//!
//! ### Time-domain statistics
//! - [`peak`](AudioStatistics::peak): Maximum absolute amplitude value.
//! - [`min_sample`](AudioStatistics::min_sample): Minimum sample value.
//! - [`max_sample`](AudioStatistics::max_sample): Maximum sample value.
//! - [`mean`](AudioStatistics::mean): Arithmetic mean of all samples.
//! - [`midpoint_sample`](AudioStatistics::midpoint_sample): Temporal midpoint value of a mono signal (the middle sample by position).
//! - [`rms`](AudioStatistics::rms): Root Mean Square energy.
//! - [`variance`](AudioStatistics::variance): Variance of sample values.
//! - [`std_dev`](AudioStatistics::std_dev): Standard deviation of sample values.
//!
//! ### Temporal signal properties
//! - [`zero_crossings`](AudioStatistics::zero_crossings): Counts sign changes in the waveform.
//! - [`zero_crossing_rate`](AudioStatistics::zero_crossing_rate): Normalised zero-crossing frequency (per second).
//! - [`cross_correlation`](AudioStatistics::cross_correlation): Computes correlation between two signals.
//!
//! ### Frequency-domain features *(requires `fft` feature)*
//! - [`autocorrelation`](AudioStatistics::autocorrelation): Signal self-similarity up to a given lag.
//! - [`spectral_centroid`](AudioStatistics::spectral_centroid): Spectral center of mass (mono only).
//! - [`spectral_rolloff`](AudioStatistics::spectral_rolloff): Frequency below which a given proportion of spectral energy lies.
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
//! println!("Peak: {peak}, RMS: {rms:.4}, Mean: {mean:.4}");
//! ```
//!
//! ## Example: Frequency-domain analysis
//!
//! ```rust,ignore
//! // Requires the "transforms" feature.
//! use audio_samples::{sine_wave, AudioSamples, AudioStatistics, sample_rate};
//! use audio_samples::operations::types::ChannelReduction;
//! use std::time::Duration;
//!
//! let sr       = sample_rate!(44100);
//! let freq     = 1000.0;
//! let duration = Duration::from_secs(1);
//! let audio    = sine_wave::<f64>(freq, duration, sr, 0.5);
//!
//! let centroid = audio.spectral_centroid(ChannelReduction::Error).unwrap();
//! let rolloff  = audio.spectral_rolloff(0.85, ChannelReduction::Error).unwrap();
//!
//! println!("Spectral centroid: {:.2} Hz", centroid);
//! println!("Spectral rolloff (85%): {:.2} Hz", rolloff);
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
//! - `fft`: Enables spectral and autocorrelation analyses via the
//!   [`spectrograms`] crate.
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
use crate::repr::AudioData;
use crate::traits::StandardSample;
#[cfg(feature = "transforms")]
use crate::operations::types::ChannelReduction;
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
    /// # Returns
    /// The mean value as `f64`.
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
        match &self.data() {
            AudioData::Mono(mono_data) => mono_data.mean().cast_into(),
            AudioData::Multi(multi_data) => multi_data.mean().cast_into(),
        }
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
    /// RMS is the square root of the mean of squared sample values. It provides
    /// a measure of the signal's average energy and is commonly used for
    /// loudness estimation.
    ///
    /// # Returns
    /// The RMS value as `f64`.
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
    /// Variance measures the spread of sample values around the mean.
    /// For mono audio this is the standard population variance of all samples.
    /// For multi-channel audio the variance is computed per sample position
    /// across channels and the results are averaged over time.
    ///
    /// # Returns
    /// The variance as `f64`.
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
        match &self.data() {
            AudioData::Mono(mono_data) => mono_data.variance(),
            AudioData::Multi(multi_data) => multi_data
                .variance_axis(Axis(0))
                .mean()
                .expect("Non empty data will produce a mean"),
        }
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
    /// Zero crossings occur when the signal changes sign between adjacent samples.
    /// This metric is useful for pitch detection, signal analysis, and estimating
    /// the noisiness of a signal.
    ///
    /// # Returns
    /// The total number of zero crossings across all channels. Returns 0 if the
    /// audio has fewer than 2 samples.
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
                    let prev: T = arr[i - 1];
                    let curr: T = arr[i];

                    // Check for sign change (zero crossing)
                    if (prev > T::zero() && curr <= T::zero())
                        || (prev <= T::zero() && curr > T::zero())
                    {
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
                        let prev: T = channel[i - 1];
                        let curr: T = channel[i];

                        if (prev > T::zero() && curr <= T::zero())
                            || (prev <= T::zero() && curr > T::zero())
                        {
                            total_crossings += 1;
                        }
                    }
                }
                total_crossings
            }
        }
    }

    /// Computes the zero crossing rate (crossings per second).
    ///
    /// This normalises the zero crossing count by the signal duration, providing
    /// a frequency-like measure that is independent of signal length.
    ///
    /// # Returns
    /// The number of zero crossings per second as `f64`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 1.0, -1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let zcr = audio.zero_crossing_rate();
    /// assert!(zcr > 0.0);
    /// ```
    #[inline]
    fn zero_crossing_rate(&self) -> f64 {
        let crossings = self.zero_crossings() as f64;
        let duration_seconds = self.duration_seconds(); // guaranteed > 0.0 since we do not allow empty audio
        crossings / duration_seconds
    }

    /// Computes the autocorrelation function up to `max_lag` samples.
    ///
    /// Autocorrelation measures the similarity of a signal with a time-shifted
    /// copy of itself. The value at lag 0 is the signal's mean-square value;
    /// subsequent lags decrease as the shift increases.
    ///
    /// For multi-channel audio only the first channel is used.
    ///
    /// Reference: [Autocorrelation — Wikipedia](https://en.wikipedia.org/wiki/Autocorrelation)
    ///
    /// # Arguments
    /// - `max_lag` — the maximum lag offset in samples. The effective maximum
    ///   lag is clamped to `signal_length - 1`.
    ///
    /// # Returns
    /// A [`NonEmptyVec`] of correlation values for lags `0` through
    /// `min(max_lag, signal_length - 1)`, or `None` if the FFT computation
    /// fails.
    ///
    /// # Examples
    /// ```ignore
    /// // Requires the "transforms" feature.
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    /// use std::num::NonZeroUsize;
    ///
    /// let data = array![1.0f32, 0.5, -0.5, -1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let autocorr = audio.autocorrelation(NonZeroUsize::new(3).unwrap()).unwrap();
    /// assert_eq!(autocorr.len(), NonZeroUsize::new(4).unwrap()); // lags 0..=3
    /// ```
    #[cfg(feature = "transforms")]
    #[inline]
    fn autocorrelation(&self, max_lag: NonZeroUsize) -> Option<NonEmptyVec<f64>> {
        let max_lag = max_lag.get();

        // Extract signal as Vec<f64> — mono uses the array directly,
        // multi-channel uses the first channel (row 0).
        let (signal, n) = match &self.data() {
            AudioData::Mono(arr) => {
                let n = arr.len().get();
                let sig: Vec<f64> = arr.iter().map(|&x| x.cast_into()).collect();
                (sig, n)
            }
            AudioData::Multi(arr) => {
                let first_channel = arr.row(0);
                let n = first_channel.len();
                let sig: Vec<f64> = first_channel.iter().map(|&x| x.cast_into()).collect();
                (sig, n)
            }
        };

        let effective_max_lag = max_lag.min(n - 1);

        // Zero-pad to next power of two of (2n − 1) to avoid circular correlation
        let fft_size = (2 * n - 1).next_power_of_two();
        let mut padded: Vec<f64> = Vec::with_capacity(fft_size);
        padded.extend_from_slice(&signal);
        padded.resize(fft_size, 0.0);

        // safety: padded is non-empty (fft_size >= 1)
        let padded_slice = unsafe { NonEmptySlice::new_unchecked(&padded[..]) };
        let fft_size_nz =
            // safety: fft_size is a next_power_of_two of a value >= 1
            unsafe { NonZeroUsize::new_unchecked(fft_size) };

        // Inverse real FFT → NonEmptyVec<f64> of length fft_size
        let raw = with_fft_planner(|planner| {
            // Forward FFT → Array1<Complex<f64>> of length fft_size/2 + 1
            let spectrum = planner.fft(padded_slice, fft_size_nz).ok()?;

            // Power spectrum: |X[k]|² as complex values for irfft
            let power: Vec<Complex<f64>> = spectrum
                .iter()
                .map(|c| Complex {
                    re: c.norm_sqr(),
                    im: 0.0,
                })
                .collect();
            // safety: power has the same length as spectrum (fft_size/2 + 1 >= 1)
            let power_slice = unsafe { NonEmptySlice::new_unchecked(&power[..]) };

            planner.irfft(power_slice, fft_size_nz).ok()
        })?;

        // Normalize: divide by fft_size (IFFT scaling) then by overlap count (n − lag)
        let fft_size_f = fft_size as f64;
        let mut correlations: Vec<f64> = Vec::with_capacity(effective_max_lag + 1);
        for lag in 0..=effective_max_lag {
            let overlap_count = (n - lag) as f64;
            correlations.push(raw[lag] / fft_size_f / overlap_count);
        }

        // safety: correlations has effective_max_lag + 1 >= 1 elements
        Some(unsafe { NonEmptyVec::new_unchecked(correlations) })
    }

    /// Computes cross-correlation with another audio signal.
    ///
    /// Cross-correlation measures the similarity between two signals as a
    /// function of the displacement of one relative to the other. It is useful
    /// for signal alignment, pattern matching, and delay estimation.
    ///
    /// For multi-channel audio only the first channels of both signals are
    /// correlated.
    ///
    /// # Arguments
    /// - `other` — the second audio signal. Must have the same number of
    ///   channels as `self`.
    /// - `max_lag` — the maximum lag offset in samples. The effective maximum
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
    /// let data2 = array![0.0, 1.0, 0.0, -1.0];
    /// let audio1 = AudioSamples::new_mono(data1, sample_rate!(44100)).unwrap();
    /// let audio2 = AudioSamples::new_mono(data2, sample_rate!(44100)).unwrap();
    /// let max_lag = NonZeroUsize::new(3).unwrap();
    /// let xcorr = audio1.cross_correlation(&audio2, max_lag).unwrap();
    /// assert_eq!(xcorr.len(), NonZeroUsize::new(4).unwrap()); // lags 0..=3
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
                    let mut correlation: T = T::zero();

                    let count = n1.get().min(n2.get() - lag);

                    for i in 0..count {
                        let s1 = arr1[i];
                        let s2 = arr2[i + lag];
                        correlation += s1 * s2;
                    }
                    let mut correlation = correlation.cast_into();
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
                let effective_max_lag: usize = max_lag.get().min(n1.min(n2 - 1));
                let correlations = Vec::with_capacity(effective_max_lag + 1);
                // safety: we just allocated with capacity effective_max_lag + 1 which is at
                // least 1
                let mut correlations = unsafe { NonEmptyVec::new_unchecked(correlations) };

                for lag in 0..=effective_max_lag {
                    let mut correlation = T::zero();
                    let count = n1.min(n2 - lag);

                    for i in 0..count {
                        let s1 = ch1[i];
                        let s2 = ch2[i + lag];
                        correlation += s1 * s2;
                    }
                    let mut correlation: f64 = correlation.cast_into();
                    correlation /= count as f64;
                    correlations.push(correlation);
                }

                Ok(correlations)
            }
            _ => {
                // Mixed mono/multi-channel case is a structural mismatch, not a
                // type conversion — report it as a layout incompatibility.
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
    /// The spectral centroid is the frequency-weighted mean of the power
    /// spectrum and serves as a measure of spectral brightness. Higher values
    /// indicate energy concentrated at higher frequencies.
    ///
    /// Reference: [Spectral centroid — Wikipedia](https://en.wikipedia.org/wiki/Spectral_centroid)
    ///
    /// # Returns
    /// The spectral centroid frequency in Hz. Returns `0.0` when the signal
    /// is silence (zero total spectral energy).
    ///
    /// # Arguments
    /// - `reduction` — the [`ChannelReduction`] policy for multi-channel input.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the signal is multi-channel
    ///   and `reduction` is [`ChannelReduction::Error`], or a
    ///   [`ChannelReduction::Channel`] index is out of bounds.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
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
        let fft_output_size = n.div_ceil(nzu!(2)).checked_add(1).expect(
            "n is non-zero since self is non-empty by design and  n/2 is always << usize::MAX, which means n/2 + 1 << usize::MAX and cannot overflow",
        );
        let power_spectrum =
            with_fft_planner(|planner| planner.power_spectrum(working_samples, n, None))?;

        let nyquist = self.nyquist();
        let freq_step = nyquist / (fft_output_size.get() - 1) as f64;
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
            // todo! --- is this correct behaviour? Should we force handling of zero-energy signals?
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
    /// The spectral rolloff is the frequency below which a specified proportion
    /// of the total spectral energy is contained. It is commonly used to
    /// distinguish harmonic signals from noise-like signals.
    ///
    /// # Arguments
    /// - `rolloff_percent` — the energy proportion threshold. Must lie in the
    ///   open interval `(0.0, 1.0)`. A typical value is `0.85`.
    /// - `reduction` — the [`ChannelReduction`] policy for multi-channel input.
    ///   Pass [`ChannelReduction::First`] to reproduce the historical channel-0
    ///   behaviour.
    ///
    /// # Returns
    /// The rolloff frequency in Hz. Returns `0.0` when the signal is silence
    /// (zero total spectral energy).
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `rolloff_percent` is not in
    ///   `(0.0, 1.0)`, if the signal is multi-channel and `reduction` is
    ///   [`ChannelReduction::Error`], or a [`ChannelReduction::Channel`] index is
    ///   out of bounds.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
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
        let input: NonEmptyVec<f64> = unsafe { NonEmptyVec::new_unchecked(input_vec) };

        let power_spectrum = with_fft_planner(|planner| {
            planner.power_spectrum(input.as_non_empty_slice(), n, None)
        })?;

        // Generate frequency bins
        let nyquist = self.nyquist();
        let freq_step = nyquist / (power_spectrum.len().get() as f64 - 1.0);

        // Compute total energy
        let total_energy: f64 = power_spectrum.iter().fold(0.0, |acc, &x| acc + x);
        if total_energy == 0.0 {
            // todo! --- is this correct behaviour? Should we force handling of zero-energy signals?
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

    /// Computes the spectral bandwidth (magnitude-weighted spread about the centroid).
    ///
    /// `sqrt( Σ (f_k − centroid)² · mag_k / Σ mag_k )`. The centroid is computed
    /// from the same magnitude spectrum used here.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] on a forbidden channel layout.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    #[cfg(feature = "transforms")]
    #[inline]
    fn spectral_bandwidth(&self, reduction: ChannelReduction) -> AudioSampleResult<f64> {
        let working_vec = reduce_to_mono_f64(self, reduction, "spectral_bandwidth")?;
        // safety: self is non-empty by design.
        let n = unsafe { NonZeroUsize::new_unchecked(working_vec.len()) };
        let input: NonEmptyVec<f64> = unsafe { NonEmptyVec::new_unchecked(working_vec) };
        let mag = with_fft_planner(|planner| {
            planner.magnitude_spectrum(input.as_non_empty_slice(), n, None)
        })?;

        let nyquist = self.nyquist();
        let freq_step = nyquist / (mag.len().get() as f64 - 1.0);

        // Magnitude-weighted centroid.
        let (weighted_sum, total_mag) =
            mag.iter()
                .enumerate()
                .fold((0.0, 0.0), |(w, t), (i, &m)| {
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

    /// Computes the spectral flatness (Wiener entropy): `geo_mean(power) / arith_mean(power)`.
    ///
    /// The geometric mean is evaluated in the log domain with an epsilon floor.
    /// Result is clamped to `[0, 1]`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] on a forbidden channel layout.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    #[cfg(feature = "transforms")]
    #[inline]
    fn spectral_flatness(&self, reduction: ChannelReduction) -> AudioSampleResult<f64> {
        let working_vec = reduce_to_mono_f64(self, reduction, "spectral_flatness")?;
        // safety: self is non-empty by design.
        let working_samples = unsafe { NonEmptySlice::new_unchecked(working_vec.as_slice()) };
        let n = unsafe { NonZeroUsize::new_unchecked(working_vec.len()) };
        let power =
            with_fft_planner(|planner| planner.power_spectrum(working_samples, n, None))?;

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

    /// Computes the spectral crest factor: `max(mag) / mean(mag)`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] on a forbidden channel layout.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    #[cfg(feature = "transforms")]
    #[inline]
    fn spectral_crest(&self, reduction: ChannelReduction) -> AudioSampleResult<f64> {
        let working_vec = reduce_to_mono_f64(self, reduction, "spectral_crest")?;
        // safety: self is non-empty by design.
        let n = unsafe { NonZeroUsize::new_unchecked(working_vec.len()) };
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

    /// Computes the spectral slope: ordinary least-squares slope of the **linear
    /// magnitude** spectrum versus frequency (magnitude per Hz).
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] on a forbidden channel layout.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    #[cfg(feature = "transforms")]
    #[inline]
    fn spectral_slope(&self, reduction: ChannelReduction) -> AudioSampleResult<f64> {
        let working_vec = reduce_to_mono_f64(self, reduction, "spectral_slope")?;
        // safety: self is non-empty by design.
        let n = unsafe { NonZeroUsize::new_unchecked(working_vec.len()) };
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

        let nyquist = self.nyquist();
        let freq_step = nyquist / (len as f64 - 1.0);

        // OLS slope: Σ (x − x̄)(y − ȳ) / Σ (x − x̄)².
        let count = len as f64;
        let (sum_x, sum_y) = mag.iter().enumerate().fold((0.0, 0.0), |(sx, sy), (i, &m)| {
            (sx + i as f64 * freq_step, sy + m)
        });
        let mean_x = sum_x / count;
        let mean_y = sum_y / count;
        let (num, den) = mag.iter().enumerate().fold((0.0, 0.0), |(num, den), (i, &m)| {
            let dx = i as f64 * freq_step - mean_x;
            (num + dx * (m - mean_y), den + dx * dx)
        });
        if den <= 0.0 {
            return Ok(0.0);
        }
        Ok(num / den)
    }

    /// Computes octave-band spectral contrast (librosa-style): per band, the dB
    /// difference between the mean of the top quantile and the bottom quantile.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] on a forbidden channel layout.
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
        let lo_bin = 1usize.min(bins.saturating_sub(1)).max(1);
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

            let mut band: Vec<f64> = mag[start..end].iter().map(|&m| 10.0 * (m + EPS).log10()).collect();
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

/// Returns the signed lag (in samples) that maximises the FFT cross-correlation
/// between `reference` and `query`, searching ±`max_lag` samples.
///
/// Positive lag → `query` is delayed relative to `reference`; negative → `query` leads.
/// Uses only the first channel of each signal.  Returns `None` if the FFT fails
/// or either signal is empty.
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
    let fft_nz = unsafe { NonZeroUsize::new_unchecked(fft_size) };

    let mut ref_buf = ref_sig;
    ref_buf.resize(fft_size, 0.0);
    let mut deg_buf = deg_sig;
    deg_buf.resize(fft_size, 0.0);

    let xcorr = with_fft_planner(|planner| {
        let ref_spec = planner
            .fft(unsafe { NonEmptySlice::new_unchecked(&ref_buf) }, fft_nz)
            .ok()?;
        let deg_spec = planner
            .fft(unsafe { NonEmptySlice::new_unchecked(&deg_buf) }, fft_nz)
            .ok()?;

        let cross: Vec<Complex<f64>> = ref_spec
            .iter()
            .zip(deg_spec.iter())
            .map(|(r, d)| r * d.conj())
            .collect();

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
        let stereo =
            AudioSamples::new_multi_channel(ndarray::array![[1.0f32, 2.0], [3.0, 4.0]], sample_rate!(44100))
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

        let audio = AudioSamples::new_mono(Array1::from(data).into(), sample_rate!(44100)).unwrap();
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
    }

    #[test]
    #[cfg(feature = "transforms")]
    fn test_cross_correlation() {
        let data1 = array![1.0f32, 0.0, -1.0];
        let data2 = array![1.0f32, 0.0, -1.0]; // Same signal
        let audio1 = AudioSamples::new_mono(data1.into(), sample_rate!(44100)).unwrap();
        let audio2 = AudioSamples::new_mono(data2.into(), sample_rate!(44100)).unwrap();

        let cross_corr = audio1.cross_correlation(&audio2, crate::nzu!(1)).unwrap();

        // Cross-correlation of identical signals should be same as autocorrelation
        let autocorr = audio1.autocorrelation(crate::nzu!(1)).unwrap();
        assert_eq!(cross_corr.len(), autocorr.len());
    }

    #[test]
    fn test_multi_channel_statistics() {
        let data = array![[1.0f32, 2.0], [-1.0, 1.0]]; // 2 channels, 2 samples each
        let audio = AudioSamples::new_multi_channel(data.into(), sample_rate!(44100)).unwrap();

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
        let single_audio = AudioSamples::new_mono(single_data.into(), sample_rate!(44100)).unwrap();

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
        let empty_audio = AudioSamples::new_mono(empty_data.into(), sample_rate!(44100));
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
        assert!(audio.spectral_rolloff(0.0, ChannelReduction::Error).is_err());
        assert!(audio.spectral_rolloff(1.0, ChannelReduction::Error).is_err());
        assert!(audio.spectral_rolloff(-0.1, ChannelReduction::Error).is_err());
        assert!(audio.spectral_rolloff(1.1, ChannelReduction::Error).is_err());

        // Test valid rolloff percentage
        assert!(audio.spectral_rolloff(0.85, ChannelReduction::Error).is_ok());
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
        assert!(stereo.spectral_centroid(ChannelReduction::Channel(2)).is_err());

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
    /// bit-identical results across repeated calls of the same size — i.e.
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
        assert_eq!(l1, l2, "alignment lag must be identical across cached calls");
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

        let single = AudioSamples::new_mono(Array1::from(single).into(), sr).unwrap();
        let two = AudioSamples::new_mono(Array1::from(two).into(), sr).unwrap();

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
        assert!(flat_first < 0.1, "First-channel flatness {flat_first} should be tonal");

        // Average mixes tone with silence -> still tonal.
        let flat_avg = stereo.spectral_flatness(ChannelReduction::Average).unwrap();
        assert!(flat_avg < 0.2, "Average flatness {flat_avg} should be tonal");
    }
}
