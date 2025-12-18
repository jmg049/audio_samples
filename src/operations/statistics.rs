//! Statistical analysis operations for [`AudioSamples`].
//!
//! This module implements the [`AudioStatistics`] trait, providing a range of
//! statistical and signal analysis methods for mono and multi-channel audio data.
//!
//! ## Overview
//!
//! The statistical operations exposed here are designed for direct use on
//! [`AudioSamples`] instances. They support all core sample formats
//! (`i16`, `i24`, `i32`, `f32`, `f64`) and handle both mono and multi-channel data
//! transparently. Where possible, computations are performed using
//! numerically stable and vectorised algorithms with minimal branching.
//!
//! When the `fft` feature is enabled, several frequency-domain analyses
//! (e.g. spectral centroid and rolloff) become available, using efficient FFT
//! backends (via `UnifiedFftBackend`) to accelerate large transforms. See the `mkl` feature also.
//!
//! ## Available Operations
//!
//! ### Time-domain statistics
//! - [`peak`](AudioStatistics::peak): Maximum absolute amplitude value.
//! - [`min_sample`](AudioStatistics::min_sample): Minimum sample value.
//! - [`max_sample`](AudioStatistics::max_sample): Maximum sample value.
//! - [`mean`](AudioStatistics::mean): Arithmetic mean of all samples.
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
//! - `autocorrelation`: Computes signal self-similarity up to a lag.
//! - `spectral_centroid`: Computes the "brightness" or spectral center of mass.
//! - `spectral_rolloff`: Finds the frequency below which a given proportion of total energy lies.
//!
//! ## Example: Time-domain analysis
//!
//! ```rust
//! use audio_samples::{AudioSamples, AudioStatistics};
//! use ndarray::array;
//!
//! let data = array![1.0f32, -1.0, 0.5, -0.5];
//! let audio = AudioSamples::new_mono(data, 44100);
//!
//! // Compute common time-domain statistics
//! let peak = audio.peak();
//! let rms = audio.rms();
//! let mean = audio.mean();
//!
//! println!("Peak: {peak}, RMS: {rms:.4}, Mean: {mean:.4}");
//! ```
//!
//! ## Example: Frequency-domain analysis
//!
//! ```rust,ignore
//! // This example requires the "fft" feature
//! use audio_samples::{sine_wave, AudioSamples, AudioStatistics};
//!
//! // Generate a simple 1 kHz sine wave
//! let sample_rate = 44100;
//! let freq = 1000.0;
//! let duration = 1.0; // 1 second
//! let audio: AudioSamples<f32> = sine_wave(freq, duration, sample_rate, 0.5).unwrap();
//!
//! // Compute spectral descriptors (requires "fft" feature)
//! let centroid = audio.spectral_centroid().unwrap();
//! let rolloff = audio.spectral_rolloff(0.85).unwrap();
//!
//! println!("Spectral centroid: {:.2} Hz", centroid);
//! println!("Spectral rolloff (85%): {:.2} Hz", rolloff);
//! ```
//!
//! ## Error Handling
//!
//! Most operations return simple numeric results. However, some methods that
//! require non-empty data (e.g. `variance`, `autocorrelation`, or
//! `spectral_rolloff`) return a [`Result`], allowing graceful handling of
//! empty or invalid inputs. In such cases, an [`crate::AudioSampleError::Processing`]
//! is returned with a descriptive message.
//!
//! ## Implementation Details
//!
//! - Operations use [`ndarray`]’s internal vectorisation to achieve SIMD-level
//!   performance where available.
//! - Statistical computations are automatically promoted to `f64` precision to
//!   minimise rounding errors, then cast back to the original sample type when required.
//! - FFT-based analyses pad input signals to avoid circular correlation artifacts,
//!   and use normalisation consistent with standard DSP practice.
//!
//! ## Feature Flags
//!
//! - `fft`: Enables spectral and autocorrelation analyses using fast FFT backends.
//!
//! ## See Also
//!
//! - [`AudioSamples`]: The core data structure providing access
//!   to underlying sample buffers.
//! - [`AudioStatistics`]: The trait defining
//!   all available statistical operations.
//! - `UnifiedFftBackend`: Unified FFT
//!   abstraction for optimised spectral computation.
//!
//! [`AudioStatistics`]: crate::operations::traits::AudioStatistics

#[cfg(feature = "fft")]
use crate::operations::fft_backends::{FftBackendImpl, UnifiedFftBackend};
use crate::operations::traits::AudioStatistics;
use crate::repr::AudioData;
use crate::{
    AudioSample, AudioSampleResult, AudioSamples, AudioTypeConversion, ConversionError, ConvertTo,
    I24, ParameterError, RealFloat, to_precision,
};

use ndarray::{Array1, Axis};
#[cfg(feature = "fft")]
use num_complex::Complex;
#[cfg(feature = "fft")]
use rustfft::FftPlanner;

impl<'a, T: AudioSample> AudioStatistics<'a, T> for AudioSamples<'a, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'b> AudioSamples<'b, T>: AudioTypeConversion<'b, T>,
{
    /// Returns the peak (maximum absolute value) in the audio samples.
    ///
    /// # Returns
    /// The maximum absolute value found across all samples and channels. Returns the
    /// default value for type T if the audio data is empty.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -3.0, 2.5, -1.5];
    /// let audio = AudioSamples::new_mono(data, 44100);
    /// assert_eq!(audio.peak(), 3.0);
    /// ```
    fn peak(&self) -> T {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() {
                    return T::default();
                }

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
                if arr.is_empty() {
                    return T::default();
                }

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

    /// Returns the minimum value in the audio samples.
    ///
    /// # Returns
    /// The smallest sample value found across all samples and channels. Returns the
    /// default value for type T if the audio data is empty.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -3.0, 2.5, -1.5];
    /// let audio = AudioSamples::new_mono(data, 44100);
    /// assert_eq!(audio.min_sample(), -3.0);
    /// ```
    fn min_sample(&self) -> T {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() {
                    return T::default();
                }
                // Use ndarray's efficient fold operation for vectorized minimum finding
                arr.fold(arr[0], |acc, &x| if x < acc { x } else { acc })
            }
            AudioData::Multi(arr) => {
                if arr.is_empty() {
                    return T::default();
                }
                // Vectorized minimum across entire multi-channel array
                arr.fold(arr[[0, 0]], |acc, &x| if x < acc { x } else { acc })
            }
        }
    }

    /// Returns the maximum value in the audio samples.
    ///
    /// # Returns
    /// The largest sample value found across all samples and channels. Returns the
    /// default value for type T if the audio data is empty.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -3.0, 2.5, -1.5];
    /// let audio = AudioSamples::new_mono(data, 44100);
    /// assert_eq!(audio.max_sample(), 2.5);
    /// ```
    fn max_sample(&self) -> T {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() {
                    return T::default();
                }
                // Use ndarray's efficient fold operation for vectorized maximum finding
                arr.fold(arr[0], |acc, &x| if x > acc { x } else { acc })
            }
            AudioData::Multi(arr) => {
                if arr.is_empty() {
                    return T::default();
                }
                // Vectorized maximum across entire multi-channel array
                arr.fold(arr[[0, 0]], |acc, &x| if x > acc { x } else { acc })
            }
        }
    }

    /// Computes the mean (average) of the audio samples.
    ///
    /// # Returns
    /// `Some(mean)` containing the arithmetic mean of all samples across all channels,
    /// or `None` if the audio data is empty.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 2.0, -2.0];
    /// let audio = AudioSamples::new_mono(data, 44100);
    /// assert_eq!(audio.mean::<f32>(), 0.0);
    /// ```
    fn mean<F: RealFloat>(&self) -> F {
        match &self.data {
            AudioData::Mono(mono_data) => to_precision::<F, _>(
                mono_data
                    .mean()
                    .expect("AudioSamples cannot be empty therefore mean is always available"),
            ),
            AudioData::Multi(multi_data) => to_precision::<F, _>(
                multi_data
                    .mean()
                    .expect("AudioSamples cannot be empty therefore mean is always available"),
            ),
        }
    }

    /// Computes the Root Mean Square (RMS) of the audio samples.
    ///
    /// RMS = sqrt(mean(x^2)) where x is the audio signal. This provides a measure
    /// of the signal's energy content and is commonly used for loudness estimation.
    ///
    /// # Returns
    /// `Some(rms)` containing the RMS value of all samples across all channels,
    /// or `None` if the audio data is empty.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 1.0, -1.0];
    /// let audio = AudioSamples::new_mono(data, 44100);
    /// let rms = audio.rms::<f32>();
    /// assert!((rms - 1.0).abs() < 1e-6);
    /// ```
    fn rms<F: RealFloat>(&self) -> F {
        // A frame iterator **could** be used but would be inefficient here due to the extra
        // indirection and lack of vectorization. Direct ndarray operations are preferred for functions like this.
        match &self.data {
            AudioData::Mono(mono_data) => {
                let mean_square: F = mono_data
                    .mapv(|x| {
                        let x: F = to_precision(x);
                        x * x
                    })
                    .mean()
                    .expect("AudioSamples cannot be empty therefore mean is always available");

                mean_square.sqrt()
            }
            AudioData::Multi(multi_data) => {
                let mean_square: F = multi_data
                    .mapv(|x| {
                        let x: F = to_precision(x);
                        x * x
                    })
                    .mean()
                    .expect("AudioSamples cannot be empty therefore mean is always available");

                mean_square.sqrt()
            }
        }
    }

    /// Computes the statistical variance of the audio samples.
    ///
    /// Variance = mean((x - mean(x))^2). This measures the spread of the data
    /// points around the mean value.
    ///
    /// # Returns
    /// `Some(variance)` containing the variance of all samples across all channels,
    /// or `None` if the audio data is empty.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, 2.0, 3.0, 4.0];
    /// let audio = AudioSamples::new_mono(data, 44100);
    /// let variance = audio.variance::<f32>();
    /// assert!((variance - 1.25).abs() < 1e-6);
    /// ```
    fn variance<F: RealFloat>(&self) -> F {
        match &self.data {
            AudioData::Mono(arr) => {
                let mean: F = self.mean::<F>();

                let deviations: Array1<F> = arr.mapv(|x| {
                    let x: F = to_precision(x);
                    x - mean
                });

                // Vectorized variance computation: (x - mean)^2

                let squared_deviations = &deviations * &deviations; // Element-wise square 

                squared_deviations
                    .mean()
                    .expect("Mean cannot be None as array is non-empty")
            }
            AudioData::Multi(arr) => {
                let mean: F = self.mean::<F>();

                // Vectorized variance computation: (x - mean)^2 across all channels
                let mut deviations = arr.mapv(|x| {
                    let x: F = to_precision(x);
                    x - mean
                });

                deviations.mapv_inplace(|x| x * x);

                deviations
                    .mean()
                    .expect("Mean cannot be None as array is non-empty")
            }
        }
    }

    /// Computes the standard deviation of the audio samples.
    ///
    /// Standard deviation is the square root of variance. It provides a measure
    /// of how spread out the sample values are from the mean, in the same units
    /// as the original data.
    ///
    /// # Returns
    /// `Some(std_dev)` containing the standard deviation of all samples across all channels,
    /// or `None` if the audio data is empty.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, 2.0, 3.0, 4.0];
    /// let audio = AudioSamples::new_mono(data, 44100);
    /// let std_dev = audio.std_dev::<f32>();
    /// assert!((std_dev - (1.25_f32).sqrt()).abs() < 1e-6);
    /// ```
    fn std_dev<F: RealFloat>(&self) -> F {
        self.variance::<F>().sqrt()
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
    /// use audio_samples::{AudioSamples, AudioStatistics};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 1.0, -1.0];
    /// let audio = AudioSamples::new_mono(data, 44100);
    /// assert_eq!(audio.zero_crossings(), 3);
    /// ```
    fn zero_crossings(&self) -> usize {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.len() < 2 {
                    return 0;
                }

                let mut crossings = 0;
                for i in 1..arr.len() {
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
    /// This normalizes the zero crossing count by the signal duration, providing
    /// a frequency-like measure that is independent of signal length.
    ///
    /// # Returns
    /// The number of zero crossings per second, or 0.0 if the signal duration is zero.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 1.0, -1.0];
    /// let audio = AudioSamples::new_mono(data, 44100);
    /// let zcr = audio.zero_crossing_rate::<f32>();
    /// assert!(zcr > 0.0);
    /// ```
    fn zero_crossing_rate<F: RealFloat>(&self) -> F {
        let crossings: F = to_precision::<F, _>(self.zero_crossings());
        let duration_seconds: F = self.duration_seconds();

        if duration_seconds > F::zero() {
            crossings / duration_seconds
        } else {
            F::zero()
        }
    }

    /// Computes the autocorrelation function up to max_lag samples.
    ///
    /// Uses FFT-based correlation for O(n log n) performance instead of O(n²).
    /// Autocorrelation measures how similar a signal is to a delayed version of itself.
    /// Formula: autocorr(x) = IFFT(FFT(x) * conj(FFT(x)))
    ///
    /// # Arguments
    /// * `max_lag` - Maximum lag offset to compute correlations for
    ///
    /// # Returns
    /// `Some(correlations)` containing correlation values for each lag offset from 0 to max_lag,
    /// or `None` if the array is empty or max_lag is zero.
    ///
    /// # Examples
    /// ```ignore
    /// // This example requires the "fft" feature
    /// use audio_samples::{AudioSamples, AudioStatistics};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, 0.5, -0.5, -1.0];
    /// let audio = AudioSamples::new_mono(data, 44100);
    /// let autocorr = audio.autocorrelation::<f32>(3).unwrap();
    /// assert_eq!(autocorr.len(), 4); // 0 to max_lag inclusive
    /// ```
    #[cfg(feature = "fft")]
    fn autocorrelation<F: RealFloat>(&self, max_lag: usize) -> Option<Vec<F>> {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() || max_lag == 0 {
                    return None;
                }

                let n = arr.len();
                let effective_max_lag = max_lag.min(n - 1);

                // For FFT-based correlation, we need to pad to 2*n-1 to avoid circular correlation
                let fft_size = (2 * n - 1).next_power_of_two();

                // Convert to f64 and pad with zeros
                let mut padded_signal: Vec<Complex<F>> = Vec::with_capacity(fft_size);
                for &sample in arr.iter() {
                    let sample = to_precision(sample);
                    padded_signal.push(Complex::new(sample, F::zero()));
                }
                // Pad with zeros
                padded_signal.resize(fft_size, Complex::new(F::zero(), F::zero()));

                // Compute FFT
                let mut planner = FftPlanner::new();
                let fft = planner.plan_fft_forward(fft_size);
                fft.process(&mut padded_signal);

                // Compute power spectrum: FFT(x) * conj(FFT(x))
                for sample in padded_signal.iter_mut() {
                    *sample *= sample.conj();
                }

                // Compute inverse FFT
                let ifft = planner.plan_fft_inverse(fft_size);
                ifft.process(&mut padded_signal);

                // Extract autocorrelation values and normalize
                let mut correlations = Vec::with_capacity(effective_max_lag + 1);
                let fft_size = to_precision(fft_size);

                for (lag, &signal_value) in
                    padded_signal.iter().enumerate().take(effective_max_lag + 1)
                {
                    // IFFT result is scaled by fft_size, and we normalize by number of overlaps
                    let correlation = signal_value.re / fft_size;
                    let overlap_count = n - lag;
                    let normalized_correlation = correlation / to_precision::<F, _>(overlap_count);
                    correlations.push(normalized_correlation);
                }

                Some(correlations)
            }
            AudioData::Multi(arr) => {
                // For multi-channel, compute autocorrelation on the first channel
                if arr.is_empty() || max_lag == 0 {
                    return None;
                }

                let first_channel = arr.row(0);
                let n = first_channel.len();
                let effective_max_lag = max_lag.min(n - 1);

                let fft_size = (2 * n - 1).next_power_of_two();

                // Convert first channel to complex and pad
                let mut padded_signal: Vec<Complex<F>> = Vec::with_capacity(fft_size);
                for &sample in first_channel.iter() {
                    let sample: F = to_precision(sample);
                    padded_signal.push(Complex::new(sample, F::zero()));
                }
                padded_signal.resize(fft_size, Complex::new(F::zero(), F::zero()));

                // FFT-based autocorrelation
                let mut planner = FftPlanner::new();
                let fft = planner.plan_fft_forward(fft_size);
                fft.process(&mut padded_signal);

                // Power spectrum
                for sample in padded_signal.iter_mut() {
                    *sample *= sample.conj();
                }

                // Inverse FFT
                let ifft = planner.plan_fft_inverse(fft_size);
                ifft.process(&mut padded_signal);

                // Extract and normalize correlations
                let mut correlations = Vec::with_capacity(effective_max_lag + 1);
                let fft_size = to_precision(fft_size);

                for (lag, &signal_value) in
                    padded_signal.iter().enumerate().take(effective_max_lag + 1)
                {
                    let correlation = signal_value.re / fft_size;
                    let overlap_count = n - lag;
                    let normalized_correlation = correlation / to_precision::<F, _>(overlap_count);
                    correlations.push(normalized_correlation);
                }

                Some(correlations)
            }
        }
    }

    /// Computes cross-correlation with another audio signal.
    ///
    /// Cross-correlation measures the similarity between two signals as a function
    /// of the displacement of one relative to the other. Useful for signal alignment,
    /// pattern matching, and delay estimation.
    ///
    /// # Arguments
    /// * `other` - The second audio signal to correlate with
    /// * `max_lag` - Maximum lag offset to compute correlations for
    ///
    /// # Returns
    /// `Ok(correlations)` containing correlation values for each lag offset from 0 to max_lag.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Signals have different numbers of channels
    /// - Either signal is empty
    /// - max_lag is zero
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics};
    /// use ndarray::array;
    ///
    /// let data1 = array![1.0f32, 0.0, -1.0, 0.0];
    /// let data2 = array![0.0, 1.0, 0.0, -1.0];
    /// let audio1 = AudioSamples::new_mono(data1, 44100);
    /// let audio2 = AudioSamples::new_mono(data2, 44100);
    /// let xcorr = audio1.cross_correlation::<f32>(&audio2, 3).unwrap();
    /// assert_eq!(xcorr.len(), 4); // 0 to max_lag inclusive
    /// ```
    fn cross_correlation<F: RealFloat>(
        &self,
        other: &Self,
        max_lag: usize,
    ) -> AudioSampleResult<Vec<F>> {
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
                if arr1.is_empty() || arr2.is_empty() || max_lag == 0 {
                    return Err(crate::AudioSampleError::Parameter(
                        ParameterError::invalid_value(
                            "input_data",
                            format!(
                                "One of the signals is empty or max_lag is zero -- arr1 = {} arr2 = {} max_lag = {}",
                                arr1.len(),
                                arr2.len(),
                                max_lag
                            ),
                        ),
                    ));
                }

                let n1 = arr1.len();
                let n2 = arr2.len();
                let effective_max_lag = max_lag.min(n1.min(n2) - 1);
                let mut correlations = Vec::with_capacity(effective_max_lag + 1);

                for lag in 0..=effective_max_lag {
                    let mut correlation: T = T::zero();

                    let count = n1.min(n2 - lag);

                    for i in 0..count {
                        let s1 = arr1[i];
                        let s2 = arr2[i + lag];
                        correlation += s1 * s2;
                    }
                    let mut correlation: F = to_precision::<F, T>(correlation);
                    correlation /= to_precision(count);
                    correlations.push(correlation);
                }

                Ok(correlations)
            }
            (AudioData::Multi(arr1), AudioData::Multi(arr2)) => {
                // For multi-channel, correlate the first channels
                let ch1 = arr1.row(0);
                let ch2 = arr2.row(0);

                if ch1.is_empty() || ch2.is_empty() || max_lag == 0 {
                    return Err(crate::AudioSampleError::Parameter(
                        ParameterError::invalid_value(
                            "input_data",
                            "One of the channels is empty or max_lag is zero",
                        ),
                    ));
                }

                let n1 = ch1.len();
                let n2 = ch2.len();
                let effective_max_lag = max_lag.min(n1.min(n2) - 1);
                let mut correlations = Vec::with_capacity(effective_max_lag + 1);

                for lag in 0..=effective_max_lag {
                    let mut correlation = T::zero();
                    let count = n1.min(n2 - lag);

                    for i in 0..count {
                        let s1 = ch1[i];
                        let s2 = ch2[i + lag];
                        correlation += s1 * s2;
                    }
                    let mut correlation: F = to_precision::<F, T>(correlation);
                    correlation /= to_precision(count);
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

    /// Computes the spectral centroid (brightness measure).
    ///
    /// The spectral centroid represents the "center of mass" of the spectrum
    /// and is often used as a measure of brightness or timbre.
    /// Higher values indicate brighter, more treble-heavy sounds.
    #[cfg(feature = "fft")]
    fn spectral_centroid<F: RealFloat + ConvertTo<T>>(&self) -> AudioSampleResult<F>
    where
        T: ConvertTo<F>,
    {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() {
                    return Ok(F::zero());
                }

                let n = arr.len();
                let duration: F = self.duration_seconds();

                // Create FFT backend
                let mut fft_backend = UnifiedFftBackend::auto_select(duration, n)?;

                // Prepare output buffer
                let output_size = n / 2 + 1;
                let mut fft_output = vec![Complex::new(F::zero(), F::zero()); output_size];

                // Convert to f64 for FFT
                let arr: Vec<F> = arr.mapv(|x| x.convert_to()).to_vec();

                // Compute FFT
                fft_backend.compute_real_fft(&arr, &mut fft_output)?;

                // Compute power spectrum
                let power_spectrum: Vec<F> = {
                    #[cfg(feature = "parallel-processing")]
                    {
                        use rayon::prelude::*;
                        fft_output.par_iter().map(|c| c.norm_sqr()).collect()
                    }
                    #[cfg(not(feature = "parallel-processing"))]
                    {
                        fft_output.iter().map(|c| c.norm_sqr()).collect()
                    }
                };
                // Generate frequency bins
                let sample_rate = to_precision::<F, _>(self.sample_rate.get());
                let nyquist = sample_rate / to_precision::<F, _>(2.0);
                let freq_step = nyquist / to_precision::<F, _>(output_size - 1);

                // Compute weighted sum and total energy
                let mut weighted_sum = F::zero();
                let mut total_energy = F::zero();

                for (i, &power) in power_spectrum.iter().enumerate() {
                    let frequency = to_precision::<F, _>(i) * freq_step;
                    weighted_sum += frequency * power;
                    total_energy += power;
                }

                // Compute centroid
                if total_energy > F::zero() {
                    Ok(weighted_sum / total_energy)
                } else if total_energy == F::zero() {
                    Ok(F::zero())
                } else {
                    Err(AudioSampleError::Processing(
                        ProcessingError::MathematicalFailure {
                            operation: "spectral_centroid".to_string(),
                            reason: "Total spectral energy is negative, cannot compute centroid"
                                .to_string(),
                        },
                    ))
                }
            }
            AudioData::Multi(arr) => {
                // For multi-channel, compute centroid on the first channel
                if arr.is_empty() {
                    return Ok(F::zero());
                }

                let first_channel = arr.row(0);
                let n = first_channel.len();
                let duration: F = self.duration_seconds();

                // Convert to f64 for FFT
                let input: Vec<F> = first_channel.mapv(|x| x.convert_to()).to_vec();

                // Create FFT backend
                let mut fft_backend = UnifiedFftBackend::auto_select(duration, n)?;

                // Prepare output buffer
                let output_size = n / 2 + 1;
                let mut fft_output = vec![Complex::new(F::zero(), F::zero()); output_size];

                // Compute FFT
                fft_backend.compute_real_fft(&input, &mut fft_output)?;

                // Compute power spectrum
                let power_spectrum: Vec<F> = {
                    #[cfg(feature = "parallel-processing")]
                    {
                        use rayon::prelude::*;
                        fft_output.par_iter().map(|c| c.norm_sqr()).collect()
                    }
                    #[cfg(not(feature = "parallel-processing"))]
                    {
                        fft_output.iter().map(|c| c.norm_sqr()).collect()
                    }
                };

                // Generate frequency bins
                let sample_rate = to_precision::<F, _>(self.sample_rate.get());
                let nyquist = sample_rate / to_precision::<F, _>(2.0);
                let freq_step = nyquist / to_precision::<F, _>(output_size - 1);

                // Compute weighted sum and total energy
                let mut weighted_sum = F::zero();
                let mut total_energy = F::zero();

                for (i, &power) in power_spectrum.iter().enumerate() {
                    let frequency = to_precision::<F, _>(i) * freq_step;
                    weighted_sum += frequency * power;
                    total_energy += power;
                }

                // Compute centroid
                if total_energy > F::zero() {
                    Ok(weighted_sum / total_energy)
                } else if total_energy == F::zero() {
                    Ok(F::zero())
                } else {
                    Err(AudioSampleError::Processing(
                        ProcessingError::MathematicalFailure {
                            operation: "spectral_centroid".to_string(),
                            reason: "Total spectral energy is negative, cannot compute centroid"
                                .to_string(),
                        },
                    ))
                }
            }
        }
    }

    /// Computes spectral rolloff frequency.
    ///
    /// The spectral rolloff frequency is the frequency below which a given percentage
    /// (typically 85%) of the total spectral energy is contained.
    /// This measure is useful for distinguishing between harmonic and noise-like sounds.
    #[cfg(feature = "fft")]
    fn spectral_rolloff<F: RealFloat + ConvertTo<T>>(
        &self,
        rolloff_percent: F,
    ) -> AudioSampleResult<F>
    where
        T: ConvertTo<F>,
    {
        if rolloff_percent <= F::zero() || rolloff_percent >= F::one() {
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
                if arr.is_empty() {
                    return Ok(F::zero());
                }

                let n = arr.len();
                let duration: F = self.duration_seconds();

                // Convert to f64 for FFT
                let input: Vec<F> = arr.mapv(|x| x.convert_to()).to_vec();

                // Create FFT backend
                let mut fft_backend = UnifiedFftBackend::auto_select(duration, n)?;

                // Prepare output buffer
                let output_size = n / 2 + 1;
                let mut fft_output = vec![Complex::new(F::zero(), F::zero()); output_size];

                // Compute FFT
                fft_backend.compute_real_fft(&input, &mut fft_output)?;

                // Compute power spectrum
                let power_spectrum: Vec<F> = {
                    #[cfg(feature = "parallel-processing")]
                    {
                        use rayon::prelude::*;
                        fft_output.par_iter().map(|c| c.norm_sqr()).collect()
                    }
                    #[cfg(not(feature = "parallel-processing"))]
                    {
                        fft_output.iter().map(|c| c.norm_sqr()).collect()
                    }
                };
                // Generate frequency bins
                let sample_rate = to_precision::<F, _>(self.sample_rate.get());
                let nyquist = sample_rate / to_precision::<F, _>(2.0);
                let freq_step = nyquist / to_precision::<F, _>(output_size - 1);

                // Compute total energy
                let total_energy: F = power_spectrum.iter().fold(F::zero(), |acc, &x| acc + x);
                if total_energy == F::zero() {
                    return Ok(F::zero());
                } else if total_energy < F::zero() {
                    return Err(AudioSampleError::Processing(ProcessingError::MathematicalFailure { operation: "spectral_rolloff".to_string(), reason: "Total spectral energy is negative, cannot compute rolloff frequency".to_string()}));
                }

                // Find rolloff frequency
                let target_energy = total_energy * rolloff_percent;
                let mut cumulative_energy = F::zero();

                for (i, &power) in power_spectrum.iter().enumerate() {
                    cumulative_energy += power;
                    if cumulative_energy >= target_energy {
                        let frequency = to_precision::<F, _>(i) * freq_step;
                        return Ok(frequency);
                    }
                }

                // If we reach here, return Nyquist frequency
                Ok(nyquist)
            }
            AudioData::Multi(arr) => {
                // For multi-channel, compute rolloff on the first channel
                if arr.is_empty() {
                    return Ok(F::zero());
                }

                let first_channel = arr.row(0);
                let n = first_channel.len();
                let duration: F = self.duration_seconds();

                // Convert to float type for FFT
                let input: Vec<F> = first_channel.iter().map(|&x| x.convert_to()).collect();

                // Create FFT backend
                let mut fft_backend = UnifiedFftBackend::auto_select(duration, n)?;

                // Prepare output buffer
                let output_size = n / 2 + 1;
                let mut fft_output = vec![Complex::new(F::zero(), F::zero()); output_size];

                // Compute FFT
                fft_backend.compute_real_fft(&input, &mut fft_output)?;

                // Compute power spectrum
                let power_spectrum: Vec<F> = {
                    #[cfg(feature = "parallel-processing")]
                    {
                        use rayon::prelude::*;
                        fft_output.par_iter().map(|c| c.norm_sqr()).collect()
                    }
                    #[cfg(not(feature = "parallel-processing"))]
                    {
                        fft_output.iter().map(|c| c.norm_sqr()).collect()
                    }
                };
                // Generate frequency bins
                let sample_rate = to_precision::<F, _>(self.sample_rate.get());
                let nyquist = sample_rate / to_precision::<F, _>(2.0);
                let freq_step = nyquist / to_precision::<F, _>(output_size - 1);

                // Compute total energy
                let total_energy: F = power_spectrum.iter().fold(F::zero(), |acc, &x| acc + x);
                if total_energy == F::zero() {
                    return Ok(F::zero());
                } else if total_energy < F::zero() {
                    return Err(AudioSampleError::Processing(ProcessingError::MathematicalFailure { operation: "spectral_rolloff".to_string(), reason: "Total spectral energy is negative, cannot compute rolloff frequency".to_string()}));
                }

                // Find rolloff frequency
                let target_energy = total_energy * rolloff_percent;
                let mut cumulative_energy = F::zero();

                for (i, &power) in power_spectrum.iter().enumerate() {
                    cumulative_energy += power;
                    if cumulative_energy >= target_energy {
                        let frequency = to_precision::<F, _>(i) * freq_step;
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
    use super::*;
    use crate::sample_rate;
    use approx_eq::assert_approx_eq;
    use ndarray::{Array1, array};

    #[test]
    fn test_peak_min_max_existing_methods() {
        let data = array![-3.0f32, -1.0, 0.0, 2.0, 4.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100));

        // These should use the existing native implementations
        assert_eq!(audio.peak(), 4.0);
        assert_eq!(audio.min_sample(), -3.0);
        assert_eq!(audio.max_sample(), 4.0);
    }

    #[test]
    fn test_rms_computation() {
        // Simple test case where we can verify RMS manually
        let data = array![1.0f32, -1.0, 1.0, -1.0];
        let audio: AudioSamples<'static, f32> = AudioSamples::new_mono(data, sample_rate!(44100));

        let rms = audio.rms::<f64>(); // RMS of [1, -1, 1, -1] = sqrt((1^2 + 1^2 + 1^2 + 1^2)/4) = sqrt(1) = 1.0
        assert_approx_eq!(rms, 1.0, 1e-6);
    }

    #[test]
    fn test_variance_and_std_dev() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio: AudioSamples<'_, f32> = AudioSamples::new_mono(data, sample_rate!(44100));

        let variance = audio.variance::<f64>();
        let std_dev = audio.std_dev::<f64>();

        // Mean = 3.0, variance = mean((1-3)^2 + (2-3)^2 + ... + (5-3)^2) = mean(4+1+0+1+4) = 2.0
        assert_approx_eq!(variance, 2.0, 1e-6);
        assert_approx_eq!(std_dev, 2.0_f64.sqrt(), 1e-6);
    }

    #[test]
    fn test_zero_crossings() {
        let data = array![1.0f32, -1.0, 1.0, -1.0, 1.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100));

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

        let audio = AudioSamples::new_mono(Array1::from(data).into(), sample_rate!(44100));
        let zcr = audio.zero_crossing_rate::<f64>();

        // 4 Hz square wave has ~8 zero crossings per second (2 per cycle)
        // Due to discrete sampling, we might get 7-8 crossings
        assert!(
            (zcr - 8.0f64).abs() <= 1.0,
            "Expected ~8 crossings/sec, got {}",
            zcr
        );
    }

    #[cfg(feature = "fft")]
    #[test]
    fn test_autocorrelation() {
        let data = array![1.0f32, 0.0, -1.0, 0.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100));

        let autocorr = audio.autocorrelation::<f64>(2).unwrap();

        // Should have correlations for lags 0, 1, 2
        assert_eq!(autocorr.len(), 3);

        // Lag 0 should be the highest (signal correlated with itself)
        assert!(autocorr[0] >= autocorr[1]);
        assert!(autocorr[0] >= autocorr[2]);
    }

    #[cfg(feature = "fft")]
    #[test]
    fn test_cross_correlation() {
        let data1 = array![1.0f32, 0.0, -1.0];
        let data2 = array![1.0f32, 0.0, -1.0]; // Same signal
        let audio1 = AudioSamples::new_mono(data1.into(), sample_rate!(44100));
        let audio2 = AudioSamples::new_mono(data2.into(), sample_rate!(44100));

        let cross_corr = audio1.cross_correlation::<f64>(&audio2, 1).unwrap();

        // Cross-correlation of identical signals should be same as autocorrelation
        let autocorr = audio1.autocorrelation::<f64>(1).unwrap();
        assert_eq!(cross_corr.len(), autocorr.len());
    }

    #[test]
    fn test_multi_channel_statistics() {
        let data = array![[1.0f32, 2.0], [-1.0, 1.0]]; // 2 channels, 2 samples each
        let audio = AudioSamples::new_multi_channel(data.into(), sample_rate!(44100));

        let rms: f64 = audio.rms();
        let variance = audio.variance::<f64>();
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
        let single_audio = AudioSamples::new_mono(single_data.into(), sample_rate!(44100));

        assert_eq!(single_audio.zero_crossings(), 0);
        assert_eq!(
            single_audio.rms::<f64>(),
            1.0,
            "RMS of single sample should be the sample itself"
        );
    }

    #[test]
    #[should_panic(expected = "data must not be empty")]
    fn test_empty_audio_rejected() {
        // Empty audio should be rejected at construction time
        let empty_data: ndarray::Array1<f32> = ndarray::Array1::from(vec![]);
        let _empty_audio = AudioSamples::new_mono(empty_data.into(), sample_rate!(44100));
    }

    #[cfg(feature = "fft")]
    #[test]
    fn test_spectral_centroid() {
        // Test with a simple sine wave that should have energy concentrated at a specific frequency
        let sample_rate = 44100;
        let duration = 1.0; // 1 second
        let freq = 1000.0; // 1kHz sine wave
        let n_samples = (sample_rate as f64 * duration) as usize;

        // Generate 1kHz sine wave
        let mut data = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let t = i as f64 / sample_rate as f64;
            let sample = (2.0 * std::f64::consts::PI * freq * t).sin() as f32;
            data.push(sample);
        }

        let audio = AudioSamples::new_mono(ndarray::Array1::from(data).into(), sample_rate!(44100));
        let centroid = audio
            .spectral_centroid::<f64>()
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

    #[cfg(feature = "fft")]
    #[test]
    fn test_spectral_rolloff() {
        // Test with white noise - rolloff should be around 85% of Nyquist for 85% rolloff
        let sample_rate = 8000; // Use lower sample rate for faster test
        let n_samples = 4096;

        // Generate white noise
        let mut data = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            // Simple deterministic "noise" for reproducible tests
            let sample = ((i * 17 + 13) % 101) as f32 / 50.0 - 1.0;
            data.push(sample);
        }

        let audio = AudioSamples::new_mono(ndarray::Array1::from(data).into(), sample_rate!(8000));
        let rolloff = audio
            .spectral_rolloff(0.85)
            .expect("Failed to compute spectral rolloff");
        let nyquist = sample_rate as f64 / 2.0;

        // For noise-like signals, rolloff should be somewhere reasonable
        assert!(rolloff > 0.0, "Rolloff should be positive");
        assert!(
            rolloff <= nyquist,
            "Rolloff should not exceed Nyquist frequency"
        );
    }

    #[cfg(feature = "fft")]
    #[test]
    fn test_spectral_rolloff_validation() {
        let data = array![1.0f32, -1.0, 1.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100));

        // Test invalid rolloff percentages
        assert!(audio.spectral_rolloff(0.0).is_err());
        assert!(audio.spectral_rolloff(1.0).is_err());
        assert!(audio.spectral_rolloff(-0.1).is_err());
        assert!(audio.spectral_rolloff(1.1).is_err());

        // Test valid rolloff percentage
        assert!(audio.spectral_rolloff(0.85).is_ok());
    }

    #[cfg(feature = "fft")]
    #[test]
    #[should_panic(expected = "data must not be empty")]
    fn test_spectral_methods_empty_audio() {
        // Empty audio should be rejected at construction time
        let empty_data: ndarray::Array1<f32> = ndarray::Array1::from(vec![]);
        let _empty_audio = AudioSamples::new_mono(empty_data.into(), sample_rate!(44100));
    }
}
