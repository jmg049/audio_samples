//! Statistical analysis operations for [`AudioSamples`].
//!
//! ## What
//!
//! This module implements the [`AudioStatistics`] trait, providing statistical
//! and signal analysis methods for mono and multi-channel audio data. Operations
//! cover both time-domain statistics and, when the `fft` feature is enabled,
//! frequency-domain descriptors computed via the [`spectrograms`] crate.
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
//! ### Time-domain statistics
//! - [`peak`](AudioStatistics::peak): Maximum absolute amplitude value.
//! - [`min_sample`](AudioStatistics::min_sample): Minimum sample value.
//! - [`max_sample`](AudioStatistics::max_sample): Maximum sample value.
//! - [`mean`](AudioStatistics::mean): Arithmetic mean of all samples.
//! - [`median`](AudioStatistics::median): Temporal midpoint value of a mono signal.
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
//! use std::time::Duration;
//!
//! let sr       = sample_rate!(44100);
//! let freq     = 1000.0;
//! let duration = Duration::from_secs(1);
//! let audio    = sine_wave::<f64>(freq, duration, sr, 0.5);
//!
//! let centroid = audio.spectral_centroid().unwrap();
//! let rolloff  = audio.spectral_rolloff(0.85).unwrap();
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
use crate::{AudioSampleResult, AudioSamples, ConversionError, ParameterError};

#[cfg(feature = "transforms")]
use crate::{AudioSampleError, AudioTypeConversion, ProcessingError};
#[cfg(feature = "transforms")]
use num_complex::Complex;

#[cfg(feature = "transforms")]
use non_empty_slice::NonEmptySlice;

use ndarray::Axis;
use non_empty_slice::NonEmptyVec;

#[cfg(feature = "transforms")]
use spectrograms::FftPlanner;

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
    fn peak(&self) -> T {
        match &self.data {
            AudioData::Mono(arr) => {
                // Use ndarray's vectorized operations for SIMD-optimized absolute value and max
                let abs_values = arr.mapv(|x| {
                    // Manual absolute value that works with existing trait bounds
                    if x < T::default() {
                        T::default() - x
                    } else {
                        x
                    }
                });

                // Use ndarray's efficient fold operation instead of iterator chains
                abs_values.fold(T::default(), |acc, &x| if x > acc { x } else { acc })
            }
            AudioData::Multi(arr) => {
                // Vectorized absolute value and max across entire multi-channel array
                let abs_values = arr.mapv(|x| {
                    if x < T::default() {
                        T::default() - x
                    } else {
                        x
                    }
                });

                abs_values.fold(T::default(), |acc, &x| if x > acc { x } else { acc })
            }
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
    fn min_sample(&self) -> T {
        match &self.data {
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
    fn max_sample(&self) -> T {
        match &self.data {
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
    fn mean(&self) -> f64 {
        match &self.data {
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
    /// assert_eq!(audio.median(), Some(4.0));
    /// ```
    fn median(&self) -> Option<f64> {
        let mono = match self.as_mono() {
            Some(mono) => mono,
            None => {
                return None;
            }
        };
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
    fn rms(&self) -> f64 {
        self.powf(2.0, None).mean().sqrt()
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
    fn variance(&self) -> f64 {
        match &self.data {
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
    fn zero_crossings(&self) -> usize {
        match &self.data {
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
    fn autocorrelation(&self, max_lag: NonZeroUsize) -> Option<NonEmptyVec<f64>> {
        let max_lag = max_lag.get();

        // Extract signal as Vec<f64> — mono uses the array directly,
        // multi-channel uses the first channel (row 0).
        let (signal, n) = match &self.data {
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

        let mut planner = FftPlanner::new();
        // safety: padded is non-empty (fft_size >= 1)
        let padded_slice = unsafe { NonEmptySlice::new_unchecked(&padded[..]) };
        let fft_size_nz =
            // safety: fft_size is a next_power_of_two of a value >= 1
            unsafe { NonZeroUsize::new_unchecked(fft_size) };

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

        // Inverse real FFT → NonEmptyVec<f64> of length fft_size
        let raw = planner.irfft(power_slice, fft_size_nz).ok()?;

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

        match (&self.data, &other.data) {
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
                // Mixed mono/multi-channel case - not supported
                Err(crate::AudioSampleError::Conversion(
                    ConversionError::audio_conversion(
                        "Mixed",
                        "cross_correlation",
                        "compatible",
                        "Cannot correlate mono and multi-channel signals",
                    ),
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
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the signal is multi-channel.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    ///
    /// # Assumptions
    /// The input signal must be mono. Multi-channel signals must be mixed or
    /// channel-selected before calling this method.
    #[cfg(feature = "transforms")]
    fn spectral_centroid(&self) -> AudioSampleResult<f64> {
        if self.is_multi_channel() {
            return Err(crate::AudioSampleError::Parameter(
                ParameterError::invalid_value(
                    "channels",
                    "spectral_centroid is only defined for mono signals",
                ),
            ));
        }

        // convert to f64 for processing
        let working_samples = self.to_format::<f64>();

        let working_slice = working_samples.as_slice().expect(
            "Mono audio means a 1d array which is contiguous, which means as_slice cannot fail",
        );
        // safety: working_samples is guaranteed non-empty since self is non-empty by design
        let working_samples = unsafe { NonEmptySlice::new_unchecked(working_slice) };

        let mut planner = FftPlanner::new();

        let n = self.len();
        let fft_output_size = n.div_ceil(nzu!(2)).checked_add(1).expect(
            "n is non-zero since self is non-empty by design and  n/2 is always << usize::MAX, which means n/2 + 1 << usize::MAX and cannot overflow",
        );
        let power_spectrum = planner.power_spectrum(working_samples, n, None)?;

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
    /// For multi-channel audio only the first channel is used.
    ///
    /// # Arguments
    /// - `rolloff_percent` — the energy proportion threshold. Must lie in the
    ///   open interval `(0.0, 1.0)`. A typical value is `0.85`.
    ///
    /// # Returns
    /// The rolloff frequency in Hz. Returns `0.0` when the signal is silence
    /// (zero total spectral energy).
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `rolloff_percent` is not in
    ///   `(0.0, 1.0)`.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    #[cfg(feature = "transforms")]
    fn spectral_rolloff(&self, rolloff_percent: f64) -> AudioSampleResult<f64> {
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

        match &self.data {
            AudioData::Mono(arr) => {
                let n = arr.len();
                // Safety: Self is guaranteed non-empty by design, therefore arr is non-empty, therefore to_vec is non-empty
                let input: NonEmptyVec<f64> = unsafe {
                    NonEmptyVec::new_unchecked(
                        arr.mapv(super::super::traits::ConvertTo::convert_to)
                            .to_vec(),
                    )
                };

                // Create FFT backend
                let mut planner = FftPlanner::new();

                let power_spectrum = planner.power_spectrum(input.as_non_empty_slice(), n, None)?;

                // Generate frequency bins
                let nyquist = self.nyquist();
                let freq_step = nyquist / (power_spectrum.len().get() as f64 - 1.0);

                // Compute total energy
                let total_energy: f64 = power_spectrum.iter().fold(0.0, |acc, &x| acc + x);
                if total_energy == 0.0 {
                    // todo! --- is this correct behaviour? Should we force handling of zero-energy signals?
                    return Ok(0.0);
                } else if total_energy < 0.0 {
                    return Err(AudioSampleError::Processing(ProcessingError::MathematicalFailure { operation: "spectral_rolloff".to_string(), reason: "Total spectral energy is negative, cannot compute rolloff frequency".to_string()}));
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
            AudioData::Multi(arr) => {
                // For multi-channel, compute rolloff on the first channel
                // todo! --- figure out integration of multichannel. These methods typically are only an issue since they would either:
                // 1)  Force the return value to be a Vec<f64> of per-channel values
                // 2)  Force the method to accept a weighting/averaging method for channels so as to produce a single value
                // 3)  Simply use the first channel as representative (current implementation)

                let first_channel = arr.row(0);
                let n = first_channel.len();
                // safety: Self is guaranteed non-empty by design, therefore at least one channel exists with at least one sample
                let n = unsafe { NonZeroUsize::new_unchecked(n) };

                // Convert to float type for FFT
                let input: NonEmptyVec<f64> = unsafe {
                    NonEmptyVec::new_unchecked(
                        first_channel.iter().map(|&x| x.convert_to()).collect(),
                    )
                };

                let mut planner = FftPlanner::new();
                // Prepare output buffer
                let power_spectrum = planner.power_spectrum(input.as_non_empty_slice(), n, None)?;
                // Generate frequency bins
                let nyquist = self.nyquist();
                let freq_step = nyquist / (power_spectrum.len().get() as f64 - 1.0);

                // Compute total energy
                let total_energy: f64 = power_spectrum.iter().fold(0.0, |acc, &x| acc + x);
                if total_energy == 0.0 {
                    return Ok(0.0);
                } else if total_energy < 0.0 {
                    return Err(AudioSampleError::Processing(ProcessingError::MathematicalFailure { operation: "spectral_rolloff".to_string(), reason: "Total spectral energy is negative, cannot compute rolloff frequency".to_string()}));
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
        }
    }
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
            .spectral_centroid()
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
            .spectral_rolloff(0.85)
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
        assert!(audio.spectral_rolloff(0.0).is_err());
        assert!(audio.spectral_rolloff(1.0).is_err());
        assert!(audio.spectral_rolloff(-0.1).is_err());
        assert!(audio.spectral_rolloff(1.1).is_err());

        // Test valid rolloff percentage
        assert!(audio.spectral_rolloff(0.85).is_ok());
    }
}
