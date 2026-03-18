//! Signal processing operations for audio samples.
//!
//! This module provides the [`AudioProcessing`] trait implementation for
//! [`AudioSamples`], covering normalization, scaling, filtering, dynamic-range
//! compression, and windowing. All operations use a consuming pattern — each
//! method takes ownership and returns the processed audio — enabling fluent
//! method chaining.
//!
//! Audio signals routinely need amplitude adjustment, spectral shaping, and
//! dynamic-range management before playback or further analysis. A single trait
//! keeps the API discoverable and composable across all supported sample types.
//!
//! Import [`AudioProcessing`] and call methods directly on an [`AudioSamples`]
//! value. Infallible methods (e.g. [`AudioProcessing::scale`]) return `Self`
//! directly, allowing seamless chaining into fallible calls which return
//! [`AudioSampleResult`].
//!
//! ```
//! use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
//! use ndarray::array;
//!
//! let audio = AudioSamples::new_mono(array![2.0f32, -3.0, 1.5], sample_rate!(44100))
//!     .unwrap()
//!     .scale(0.5)              // infallible — returns Self
//!     .clip(-1.0, 1.0)        // fallible — returns Result<Self>
//!     .unwrap();
//! assert!(audio[0] <= 1.0);
//! ```

use std::num::NonZeroUsize;

#[cfg(feature = "resampling")]
use crate::operations::types::ResamplingQuality;
use crate::operations::types::{NormalizationConfig, NormalizationMethod};
use crate::repr::AudioData;

#[cfg(feature = "resampling")]
use crate::repr::SampleRate;

use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, ConvertTo, LayoutError,
    ParameterError, StandardSample,
    operations::traits::{AudioProcessing, AudioStatistics},
};

use ndarray::{Array2, Axis};
use non_empty_slice::NonEmptySlice;
use num_traits::FloatConst;

impl<T> AudioProcessing for AudioSamples<'_, T>
where
    T: StandardSample,
{
    /// Normalizes audio samples using the specified configuration.
    ///
    /// The normalization method determines both the algorithm and parameters:
    /// - `NormalizationConfig::min_max(min, max)` — scale to the `[min, max]` range
    /// - `NormalizationConfig::peak(target)` — scale so the peak equals `target`
    /// - `NormalizationConfig::mean()` — subtract the mean (center around zero)
    /// - `NormalizationConfig::median()` — subtract the median (mono only)
    /// - `NormalizationConfig::zscore()` — transform to zero mean, unit variance
    ///
    /// # Arguments
    /// - `config` - Normalization configuration. Use the associated constructors
    ///   on [`NormalizationConfig`] to build the desired method and parameters.
    ///
    /// # Returns
    /// The normalized audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `min >= max` (MinMax), or if
    ///   the input is multi-channel (Median).
    ///
    /// # Panics
    /// Panics if the configuration fields required by the selected method are
    /// `None`. Use the [`NormalizationConfig`] constructors to avoid this.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, AudioStatistics, NormalizationConfig, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -3.0, 2.0, -1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .normalize(NormalizationConfig::peak(1.0))
    ///     .unwrap();
    /// assert!(audio.peak() <= 1.0);
    /// ```
    #[inline]
    fn normalize(mut self, config: NormalizationConfig<Self::Sample>) -> AudioSampleResult<Self> {
        match config.method {
            NormalizationMethod::MinMax => {
                let min = config.min.unwrap_or(T::MIN);
                let max = config.max.unwrap_or(T::MAX);
                // Validate input range
                if min >= max {
                    return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                        "normalization_range",
                        format!("min ({min:?}) >= max ({max:?})"),
                        format!("{min:?}"),
                        format!("{max:?}"),
                        "min value must be less than max value for normalization",
                    )));
                }
                // Min-Max normalization: scale to [min, max] range
                let current_min = self.min_sample();
                let current_max = self.max_sample();

                // Avoid division by zero
                if current_min == current_max {
                    // All values are the same, set to middle of target range
                    let middle = min + (max - min) / Self::Sample::cast_from(2.0f64);
                    match &mut self.data {
                        AudioData::Mono(arr) => arr.fill(middle),
                        AudioData::Multi(arr) => arr.fill(middle),
                    }
                    return Ok(self);
                }

                let current_range = current_max - current_min;
                let target_range = max - min;
                let scale_factor = target_range / current_range;

                match &mut self.data {
                    AudioData::Mono(arr) => {
                        arr.mapv_inplace(|x| min + (x - current_min) * scale_factor);
                    }
                    AudioData::Multi(arr) => {
                        arr.mapv_inplace(|x| min + (x - current_min) * scale_factor);
                    }
                }
            }

            NormalizationMethod::Peak => {
                let target = config.target.unwrap_or({
                    // Default target is the maximum representable value for the sample type
                    Self::Sample::MAX
                });
                // Peak normalization: scale by peak value to target level
                let peak: Self::Sample = self.peak();
                if peak == Self::Sample::zero() {
                    return Ok(self); // No scaling needed for zero signal
                }

                let target_f64: f64 = target.convert_to();
                let target_peak = target_f64.abs();
                let peak_f64: f64 = peak.convert_to();
                let scale_factor = target_peak / peak_f64;
                let factor_t: T = T::cast_from(scale_factor);

                match &mut self.data {
                    AudioData::Mono(arr) => arr.mapv_inplace(|x| x * factor_t),
                    AudioData::Multi(arr) => arr.mapv_inplace(|x| x * factor_t),
                }
            }

            NormalizationMethod::Mean => {
                // Mean normalization: subtract mean to center around zero
                let mean: f64 = self.mean();

                match &mut self.data {
                    AudioData::Mono(arr) => {
                        arr.mapv_inplace(|x| {
                            let x: f64 = x.cast_into();
                            let diff = x - mean;
                            Self::Sample::cast_from(diff)
                        });
                    }
                    AudioData::Multi(arr) => {
                        arr.mapv_inplace(|x| {
                            let x: f64 = x.cast_into();
                            let diff = x - mean;
                            Self::Sample::cast_from(diff)
                        });
                    }
                }
            }

            NormalizationMethod::Median => {
                // Median normalization: subtract median to center around zero
                let median: f64 = self.median().ok_or_else(|| {
                    AudioSampleError::Parameter(ParameterError::InvalidValue {
                        parameter: "self".to_string(),
                        reason: "Self is not mono".to_string(),
                    })
                })?;

                match &mut self.data {
                    AudioData::Mono(arr) => {
                        arr.mapv_inplace(|x| {
                            let x: f64 = x.cast_into();
                            let diff = x - median;
                            Self::Sample::cast_from(diff)
                        });
                    }
                    AudioData::Multi(arr) => {
                        arr.mapv_inplace(|x| {
                            let x: f64 = x.cast_into();
                            let diff = x - median;
                            Self::Sample::cast_from(diff)
                        });
                    }
                }
            }

            NormalizationMethod::ZScore => {
                // Z-Score normalization: zero mean, unit variance
                let mean: f64 = self.mean();
                let std_dev: f64 = self.std_dev();

                if std_dev == 0.0f64 {
                    // All values are the same, just subtract mean
                    match &mut self.data {
                        AudioData::Mono(arr) => {
                            arr.mapv_inplace(|x| {
                                let x: f64 = x.cast_into();
                                let diff = x - mean;
                                Self::Sample::cast_from(diff)
                            });
                        }
                        AudioData::Multi(arr) => {
                            arr.mapv_inplace(|x| {
                                let x: f64 = x.cast_into();
                                let diff = x - mean;
                                Self::Sample::cast_from(diff)
                            });
                        }
                    }
                } else {
                    match &mut self.data {
                        AudioData::Mono(arr) => {
                            arr.mapv_inplace(|x| {
                                let x: f64 = x.cast_into();
                                let diff = x - mean;
                                Self::Sample::cast_from(diff / std_dev)
                            });
                        }
                        AudioData::Multi(arr) => {
                            arr.mapv_inplace(|x| {
                                let x: f64 = x.cast_into();
                                let diff = x - mean;
                                Self::Sample::cast_from(diff / std_dev)
                            });
                        }
                    }
                }
            }
        }

        Ok(self)
    }

    /// Scales all audio samples by a constant factor.
    ///
    /// This is equivalent to adjusting the volume or amplitude of the signal.
    /// A factor of 1.0 leaves the signal unchanged; values > 1.0 amplify and
    /// values < 1.0 attenuate.
    ///
    /// # Arguments
    /// - `factor` - The scaling factor to apply to all samples.
    ///
    /// # Returns
    /// The scaled audio samples.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 0.5];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .scale(2.0);
    /// assert_eq!(audio[0], 2.0);
    /// assert_eq!(audio[1], -2.0);
    /// ```
    #[inline]
    fn scale(mut self, factor: f64) -> Self {
        let factor_t: T = T::cast_from(factor);
        match &mut self.data {
            AudioData::Mono(arr) => arr.mapv_inplace(|x| x * factor_t),
            AudioData::Multi(arr) => arr.mapv_inplace(|x| x * factor_t),
        }
        self
    }

    /// Removes DC offset by subtracting the mean value.
    ///
    /// This centers the audio around zero and removes any constant bias that
    /// may have been introduced during recording or processing.
    ///
    /// # Returns
    /// The audio samples with the DC offset removed.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![2.0f32, 3.0, 4.0, 5.0]; // Has DC offset
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .remove_dc_offset()
    ///     .unwrap();
    /// let mean: f64 = audio.mean();
    /// assert!(mean.abs() < 1e-6); // Mean is now ~0
    /// ```
    #[inline]
    fn remove_dc_offset(mut self) -> AudioSampleResult<Self> {
        let mean: f64 = self.mean();

        match &mut self.data {
            AudioData::Mono(arr) => {
                arr.mapv_inplace(|x| {
                    let x: f64 = x.cast_into();
                    let diff = x - mean;
                    Self::Sample::cast_from(diff)
                });
            }
            AudioData::Multi(arr) => {
                arr.mapv_inplace(|x| {
                    let x: f64 = x.cast_into();
                    let diff = x - mean;
                    Self::Sample::cast_from(diff)
                });
            }
        }
        Ok(self)
    }

    /// Clips audio samples to the specified range.
    ///
    /// Any samples outside `[min_val, max_val]` are clamped to the nearest
    /// boundary. Useful for preventing digital clipping before output.
    ///
    /// # Arguments
    /// - `min_val` - Minimum allowed sample value.
    /// - `max_val` - Maximum allowed sample value.
    ///
    /// # Returns
    /// The clipped audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `min_val > max_val`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![2.0f32, -3.0, 1.5, -0.5];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .clip(-1.0, 1.0)
    ///     .unwrap();
    /// assert_eq!(audio[0], 1.0);   // clamped to max
    /// assert_eq!(audio[1], -1.0);  // clamped to min
    /// assert_eq!(audio[3], -0.5);  // within range, unchanged
    /// ```
    #[inline]
    fn clip(mut self, min_val: Self::Sample, max_val: Self::Sample) -> AudioSampleResult<Self> {
        if min_val > max_val {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "clipping_range",
                format!("min ({min_val:?}) > max ({max_val:?})"),
                format!("{min_val:?}"),
                format!("{max_val:?}"),
                "min value must be less than or equal to max value for clipping",
            )));
        }

        match &mut self.data {
            AudioData::Mono(arr) => arr.mapv_inplace(|x| {
                if x < min_val { min_val } else if x > max_val { max_val } else { x }
            }),
            AudioData::Multi(arr) => arr.mapv_inplace(|x| {
                if x < min_val { min_val } else if x > max_val { max_val } else { x }
            }),
        }
        Ok(self)
    }

    /// Applies a windowing function to the audio samples.
    ///
    /// Multiplies each sample by the corresponding window coefficient
    /// element-wise. Windowing is commonly used before FFT operations to
    /// reduce spectral leakage. Applied independently to each channel.
    ///
    /// # Arguments
    /// - `window` - Window coefficients. Length must equal the number of
    ///   samples in the audio.
    ///
    /// # Returns
    /// The windowed audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the window length does not
    ///   match the audio length.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    /// use non_empty_slice::NonEmptySlice;
    ///
    /// let data = array![1.0f32, 1.0, 1.0, 1.0];
    /// let window = [1.0f32, 0.5, 0.5, 1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .apply_window(NonEmptySlice::new(&window).unwrap())
    ///     .unwrap();
    /// assert_eq!(audio[0], 1.0);
    /// assert_eq!(audio[1], 0.5);
    /// ```
    #[inline]
    fn apply_window(mut self, window: &NonEmptySlice<Self::Sample>) -> AudioSampleResult<Self> {
        match &mut self.data {
            AudioData::Mono(arr) => {
                if window.len() != arr.len() {
                    return Err(AudioSampleError::Layout(LayoutError::dimension_mismatch(
                        format!("window length ({})", window.len()),
                        format!("audio length ({})", arr.len()),
                        "apply_window",
                    )));
                }

                for (sample, &win_coeff) in arr.iter_mut().zip(window.iter()) {
                    *sample *= win_coeff;
                }
            }
            AudioData::Multi(arr) => {
                let num_samples = arr.ncols();
                if window.len() != num_samples {
                    return Err(AudioSampleError::Layout(LayoutError::dimension_mismatch(
                        format!("window length ({})", window.len()),
                        format!("audio length ({num_samples})"),
                        "apply_window",
                    )));
                }

                // Apply window to each channel
                for mut channel in arr.axis_iter_mut(Axis(0)) {
                    for (sample, &win_coeff) in channel.iter_mut().zip(window.iter()) {
                        *sample *= win_coeff;
                    }
                }
            }
        }
        Ok(self)
    }

    /// Applies a digital filter to the audio samples using direct FIR convolution.
    ///
    /// Each output sample is the dot product of the filter coefficients with the
    /// corresponding segment of the input signal. The resulting audio is shorter
    /// than the original by `filter_coeffs.len() - 1` samples.
    ///
    /// # Arguments
    /// - `filter_coeffs` - FIR filter coefficients. A single-element slice
    ///   `[1.0]` leaves the signal unchanged.
    ///
    /// # Returns
    /// The filtered audio samples, with length reduced by `filter_coeffs.len() - 1`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the audio is shorter than
    ///   the filter.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    /// use non_empty_slice::NonEmptySlice;
    ///
    /// let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
    /// let coeffs = [0.5f32, 0.5]; // 2-sample moving average
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .apply_filter(NonEmptySlice::new(&coeffs).unwrap())
    ///     .unwrap();
    /// assert_eq!(audio[0], 1.5); // 1*0.5 + 2*0.5
    /// assert_eq!(audio[1], 2.5); // 2*0.5 + 3*0.5
    /// ```
    #[inline]
    fn apply_filter(
        mut self,
        filter_coeffs: &NonEmptySlice<Self::Sample>,
    ) -> AudioSampleResult<Self> {
        match &mut self.data {
            AudioData::Mono(arr) => {
                if arr.len() < filter_coeffs.len() {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "audio_length",
                        "Audio length must be at least as long as filter length",
                    )));
                }

                let filter_len = filter_coeffs.len();

                let output_len = arr.len().get() - filter_len.get() + 1;
                // safety: output_len > 0 because arr.len() >= filter_len
                let output_len = unsafe { NonZeroUsize::new_unchecked(output_len) };

                // Perform convolution using array views (no vector allocation)
                // Create output buffer to avoid overwriting input during convolution
                let mut output = ndarray::Array1::zeros(output_len.get());

                // Perform convolution using array views (no vector allocation)
                for i in 0..output_len.get() {
                    let mut sum = Self::Sample::zero();
                    for j in 0..filter_len.get() {
                        sum += arr[i + j] * filter_coeffs[j];
                    }
                    output[i] = sum;
                }
                // Replace the original data with the filtered output
                *arr = output.try_into()?;
            }
            AudioData::Multi(arr) => {
                if arr.ncols() < filter_coeffs.len() {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "audio_length",
                        "Audio length must be at least as long as filter length",
                    )));
                }

                let filter_len = filter_coeffs.len();
                let output_len = arr.ncols().get() - filter_len.get() + 1;
                let num_channels = arr.nrows();

                // Create output buffer
                let mut output = Array2::zeros((num_channels.get(), output_len));

                // Apply filter to each channel using views (no vector allocation)
                for (ch, mut output_channel) in output.axis_iter_mut(Axis(0)).enumerate() {
                    let input_channel = arr.row(ch);

                    for i in 0..output_len {
                        let mut sum = Self::Sample::zero();
                        for j in 0..filter_len.get() {
                            sum += input_channel[i + j] * filter_coeffs[j];
                        }
                        output_channel[i] = sum;
                    }
                }

                *arr = output.try_into()?;
            }
        }
        Ok(self)
    }

    /// Applies μ-law compression to the audio samples.
    ///
    /// Compresses the dynamic range of the signal using a μ-law nonlinear
    /// transfer function. Higher `mu` values produce stronger compression.
    ///
    /// # Arguments
    /// - `mu` - Compression parameter (typically 255 for standard μ-law).
    ///
    /// # Returns
    /// The compressed audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if sample-type conversion fails.
    ///
    /// # See Also
    /// - [μ-law algorithm (Wikipedia)](https://en.wikipedia.org/wiki/G.711)
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![0.5f32, -0.5];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .mu_compress(255.0)
    ///     .unwrap();
    /// assert!(audio[0] > 0.0); // Positive input stays positive
    /// assert!(audio[1] < 0.0); // Negative input stays negative
    /// ```
    #[inline]
    fn mu_compress(self, mu: Self::Sample) -> AudioSampleResult<Self> {
        let mu_f64: f64 = mu.convert_to();
        let mu_plus_one: f64 = mu_f64 + 1.0;

        self.apply_with_error(|x: Self::Sample| {
            let x: f64 = x.convert_to();
            let sign = if x >= 0.0 { 1.0 } else { -1.0 };
            let abs_x = x.abs();
            let compressed = sign * mu_f64.mul_add(abs_x, mu_plus_one.ln()).ln() / mu_plus_one.ln();
            Ok(Self::Sample::convert_from(compressed))
        })
    }

    /// Applies μ-law expansion (decompression) to the audio samples.
    ///
    /// This inverts μ-law compression. The `mu` parameter must match the value
    /// used during compression for correct reconstruction.
    ///
    /// # Arguments
    /// - `mu` - Expansion parameter. Must match the value used for compression.
    ///
    /// # Returns
    /// The expanded audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if sample-type conversion fails.
    ///
    /// # See Also
    /// - [μ-law algorithm (Wikipedia)](https://en.wikipedia.org/wiki/G.711)
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![0.0f32, 0.5, -0.5];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .mu_expand(255.0)
    ///     .unwrap();
    /// assert_eq!(audio[0], 0.0); // Zero maps to zero
    /// assert!(audio[1] > 0.0);   // Sign is preserved
    /// assert!(audio[2] < 0.0);   // Sign is preserved
    /// ```
    #[inline]
    fn mu_expand(self, mu: Self::Sample) -> AudioSampleResult<Self> {
        let mu: f64 = mu.convert_to();
        let mu_plus_one = mu + 1.0;

        self.apply_with_error(|x: Self::Sample| {
            let x_f64: f64 = x.convert_to();
            let sign = if x_f64 >= 0.0 { 1.0 } else { -1.0 };
            let abs_x = x_f64.abs();
            let expanded = sign * (mu_plus_one.powf(abs_x) - 1.0) / mu;
            Ok(Self::Sample::convert_from(expanded))
        })
    }

    /// Applies a first-order low-pass filter with the specified cutoff frequency.
    ///
    /// Uses a single-pole IIR filter to attenuate frequencies above the cutoff.
    /// The filter operates independently on each channel.
    ///
    /// # Arguments
    /// - `cutoff_hz` - Cutoff frequency in Hz. Must be less than the Nyquist
    ///   frequency (half the sample rate).
    ///
    /// # Returns
    /// The filtered audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `cutoff_hz` is ≥ the
    ///   Nyquist frequency.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .low_pass_filter(1000.0)
    ///     .unwrap();
    /// // High-frequency content is attenuated
    /// assert!(audio[1].abs() < 1.0);
    /// ```
    #[inline]
    fn low_pass_filter(mut self, cutoff_hz: f64) -> AudioSampleResult<Self> {
        // Simple implementation using a basic low-pass filter design
        let sample_rate = self.sample_rate_hz();
        let normalized_cutoff = cutoff_hz / sample_rate;

        if normalized_cutoff >= 0.5 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "cutoff_hz",
                "Cutoff frequency must be less than Nyquist frequency",
            )));
        }

        // Simple single-pole low-pass filter coefficient
        let alpha = 2.0 * f64::PI() * normalized_cutoff;
        let one_minus_alpha = 1.0 - alpha;

        match &mut self.data {
            AudioData::Mono(arr) => {
                let mut prev_output: f64 = arr[0].convert_to();
                for sample in arr.iter_mut() {
                    let s: f64 = (*sample).convert_to();
                    let s = alpha * s + one_minus_alpha * prev_output;
                    prev_output = s;
                    *sample = s.convert_to();
                }
            }
            AudioData::Multi(arr) => {
                for mut channel in arr.axis_iter_mut(Axis(0)) {
                    let mut prev_output: f64 = channel[0].convert_to();
                    for sample in &mut channel {
                        let s: f64 = (*sample).convert_to();
                        let s = alpha * s + one_minus_alpha * prev_output;
                        prev_output = s;
                        *sample = s.convert_to();
                    }
                }
            }
        }
        Ok(self)
    }

    /// Applies a first-order high-pass filter with the specified cutoff frequency.
    ///
    /// Uses an RC high-pass filter model to attenuate frequencies below the
    /// cutoff. The filter operates independently on each channel.
    ///
    /// # Arguments
    /// - `cutoff_hz` - Cutoff frequency in Hz. Must be less than the Nyquist
    ///   frequency (half the sample rate).
    ///
    /// # Returns
    /// The filtered audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `cutoff_hz` is ≥ the
    ///   Nyquist frequency.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, 1.0, 1.0, 1.0]; // Constant (DC) signal
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .high_pass_filter(100.0)
    ///     .unwrap();
    /// // A constant signal is fully removed by a high-pass filter
    /// assert_eq!(audio[0], 0.0);
    /// assert_eq!(audio[3], 0.0);
    /// ```
    #[inline]
    fn high_pass_filter(mut self, cutoff_hz: f64) -> AudioSampleResult<Self> {
        // Simple implementation using a basic high-pass filter design
        let sample_rate = self.sample_rate_hz();
        let normalized_cutoff = cutoff_hz / sample_rate;

        if normalized_cutoff >= 0.5 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "cutoff_hz",
                "Cutoff frequency must be less than Nyquist frequency",
            )));
        }

        // Simple high-pass filter using RC circuit model
        let rc = 1.0 / (2.0 * f64::PI() * cutoff_hz);
        let dt = 1.0 / sample_rate;
        let alpha = T::cast_from(rc / (rc + dt));

        match &mut self.data {
            AudioData::Mono(arr) => {
                if arr.len().get() > 1 {
                    let mut prev_input = arr[0];
                    let mut prev_output = Self::Sample::zero();

                    for sample in arr.iter_mut() {
                        let current = *sample;
                        *sample = alpha * (prev_output + current - prev_input);
                        prev_input = current;
                        prev_output = *sample;
                    }
                }
            }
            AudioData::Multi(arr) => {
                for mut channel in arr.axis_iter_mut(Axis(0)) {
                    if channel.len() > 1 {
                        let mut prev_input = channel[0];
                        let mut prev_output = Self::Sample::zero();

                        for sample in &mut channel {
                            let current = *sample;
                            *sample = alpha * (prev_output + current - prev_input);
                            prev_input = current;
                            prev_output = *sample;
                        }
                    }
                }
            }
        }
        Ok(self)
    }

    /// Applies a band-pass filter that passes frequencies between the two cutoffs.
    ///
    /// Implemented by cascading a high-pass filter at `low_cutoff_hz` followed
    /// by a low-pass filter at `high_cutoff_hz`.
    ///
    /// # Arguments
    /// - `low_cutoff_hz` - Lower cutoff frequency in Hz.
    /// - `high_cutoff_hz` - Upper cutoff frequency in Hz. Must be greater than
    ///   `low_cutoff_hz` and less than the Nyquist frequency.
    ///
    /// # Returns
    /// The filtered audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `low_cutoff_hz >= high_cutoff_hz`,
    ///   or if either cutoff exceeds the Nyquist frequency.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .band_pass_filter(100.0, 5000.0)
    ///     .unwrap();
    /// assert!(audio[0].is_finite());
    /// ```
    #[inline]
    fn band_pass_filter(self, low_cutoff_hz: f64, high_cutoff_hz: f64) -> AudioSampleResult<Self> {
        if low_cutoff_hz >= high_cutoff_hz {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "frequency_range",
                "Low frequency must be less than high frequency",
            )));
        }

        // Apply high-pass filter first, then low-pass (now using consuming pattern)
        let audio = self.high_pass_filter(low_cutoff_hz)?;
        audio.low_pass_filter(high_cutoff_hz)
    }

    /// Resamples audio to a new sample rate using high-quality resampling.
    ///
    /// Delegates to the `rubato` resampling library. The `quality` parameter
    /// controls the trade-off between speed and output fidelity.
    ///
    /// # Arguments
    /// - `target_sample_rate` - Desired output sample rate.
    /// - `quality` - Resampling quality preset (see [`ResamplingQuality`]).
    ///
    /// # Returns
    /// A new [`AudioSamples`] instance at the target sample rate.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the target sample rate or
    ///   quality parameters are invalid.
    /// - [`crate::AudioSampleError::Layout`] if the input audio is empty.
    #[cfg(feature = "resampling")]
    #[inline]
    fn resample(
        &self,
        target_sample_rate: SampleRate,
        quality: ResamplingQuality,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>> {
        use crate::resample;

        resample::<Self::Sample>(self, target_sample_rate, quality)
    }

    /// Resamples audio by a specific ratio.
    ///
    /// The output length is scaled by `ratio` relative to the input. A ratio
    /// of 2.0 doubles the sample count; 0.5 halves it.
    ///
    /// Delegates to the `rubato` resampling library. The `quality` parameter
    /// controls the trade-off between speed and output fidelity.
    ///
    /// # Arguments
    /// - `ratio` - Resampling ratio (`output_rate / input_rate`). Must be > 0.
    /// - `quality` - Resampling quality preset (see [`ResamplingQuality`]).
    ///
    /// # Returns
    /// A new [`AudioSamples`] instance resampled by the given ratio.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `ratio` is ≤ 0 or the
    ///   quality parameters are invalid.
    /// - [`crate::AudioSampleError::Layout`] if the input audio is empty.
    #[cfg(feature = "resampling")]
    #[inline]
    fn resample_by_ratio(
        &self,
        ratio: f64,
        quality: ResamplingQuality,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>> {
        use crate::resample_by_ratio;

        resample_by_ratio(self, ratio, quality)
    }
}

// Helper methods for the AudioProcessing implementation
impl<T> AudioSamples<'_, T>
where
    T: StandardSample,
{
    /// Applies a fallible function to all samples in the audio data.
    /// This is a helper method for operations that can fail.
    ///
    /// # Examples
    /// ```rust,ignore
    /// // Apply a conversion that might fail
    /// let audio = audio.apply_with_error(|sample| {
    ///     if sample > max_value {
    ///         Err(AudioSampleError::Parameter(/* error */))
    ///     } else {
    ///         Ok(sample * 2.0)
    ///     }
    /// })?;
    /// ```
    ///
    /// # Errors
    ///
    /// - if the provided function errors for some reason
    #[inline]
    pub fn apply_with_error<F>(mut self, f: F) -> AudioSampleResult<Self>
    where
        F: Fn(T) -> AudioSampleResult<T>,
    {
        match &mut self.data {
            AudioData::Mono(arr) => {
                for x in arr.iter_mut() {
                    *x = f(*x)?;
                }
                Ok(self)
            }
            AudioData::Multi(arr) => {
                for x in arr.iter_mut() {
                    *x = f(*x)?;
                }
                Ok(self)
            }
        }
    }

    /// Applies a fallible function that uses an accumulator pattern.
    /// Useful for operations that need to maintain state across samples.
    ///
    /// # Examples
    /// ```rust,ignore
    /// let mut state = 0.0;
    /// audio.try_fold(|acc, sample| {
    ///     *acc += sample.abs();
    ///     if *acc > threshold {
    ///         Err(AudioSampleError::Parameter(/* error */))
    ///     } else {
    ///         Ok(sample)
    ///     }
    /// })?;
    /// ```
    ///
    ///  # Errors
    ///
    /// - if the provided function errors for some reason
    #[inline]
    pub fn try_fold<Acc, F>(&mut self, mut acc: Acc, mut f: F) -> AudioSampleResult<Acc>
    where
        F: FnMut(&mut Acc, T) -> AudioSampleResult<T>,
    {
        match &mut self.data {
            AudioData::Mono(arr) => {
                for x in arr.iter_mut() {
                    *x = f(&mut acc, *x)?;
                }
                Ok(acc)
            }
            AudioData::Multi(arr) => {
                for x in arr.iter_mut() {
                    *x = f(&mut acc, *x)?;
                }
                Ok(acc)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample_rate;
    use approx_eq::assert_approx_eq;
    use ndarray::array;

    use crate::AudioProcessing;

    #[test]
    fn test_normalize_min_max() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio: AudioSamples<f32> = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let audio = audio
            .normalize(NormalizationConfig::min_max(-1.0, 1.0))
            .unwrap();

        assert_approx_eq!(audio.min_sample() as f64, -1.0);
        assert_approx_eq!(audio.max_sample() as f64, 1.0);
    }

    #[test]
    fn test_normalize_peak() {
        let data = array![-2.0f32, 1.0, 3.0, -1.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let audio = audio.normalize(NormalizationConfig::peak(1.0)).unwrap();

        assert_approx_eq!(audio.peak() as f64, 1.0);
    }

    #[test]
    fn test_scale() {
        let data = array![1.0f32, 2.0, 3.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let audio = audio.scale(2.0);

        let expected = array![2.0f32, 4.0, 6.0];
        match &audio.data {
            AudioData::Mono(arr) => {
                for (actual, expected) in arr.iter().zip(expected.iter()) {
                    assert_approx_eq!(*actual as f64, *expected as f64, 1e-6);
                }
            }
            _ => panic!("Expected mono data"),
        }
    }

    #[test]
    fn test_remove_dc_offset() {
        let data = array![3.0f32, 4.0, 5.0]; // Mean = 4.0
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let audio = audio.remove_dc_offset().unwrap();

        let mean: f64 = audio.mean();
        assert_approx_eq!(mean as f64, 0.0, 1e-6);
    }

    #[test]
    fn test_clip() {
        let data = array![-3.0f32, -1.0, 0.0, 1.0, 3.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let audio = audio.clip(-2.0, 2.0).unwrap();

        let expected = array![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        match &audio.data {
            AudioData::Mono(arr) => {
                for (actual, expected) in arr.iter().zip(expected.iter()) {
                    assert_approx_eq!(*actual as f64, *expected as f64, 1e-6);
                }
            }
            _ => panic!("Expected mono data"),
        }
    }

    #[test]
    fn test_apply_window() {
        let data = array![1.0f32, 1.0, 1.0, 1.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
        let window = [0.5f32, 1.0, 1.0, 0.5];
        let window_slice = NonEmptySlice::new(&window).unwrap();
        let audio = audio.apply_window(window_slice).unwrap();

        let expected = array![0.5f32, 1.0, 1.0, 0.5];
        match &audio.data {
            AudioData::Mono(arr) => {
                for (actual, expected) in arr.iter().zip(expected.iter()) {
                    assert_approx_eq!(*actual as f64, *expected as f64, 1e-6);
                }
            }
            _ => panic!("Expected mono data"),
        }
    }

    #[test]
    fn test_multi_channel_normalize() {
        let data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let audio: AudioSamples<f32> =
            AudioSamples::new_multi_channel(data.into(), sample_rate!(44100)).unwrap();

        let audio = audio
            .normalize(NormalizationConfig::min_max(-1.0, 1.0))
            .unwrap();

        assert_approx_eq!(audio.min_sample() as f64, -1.0);
        assert_approx_eq!(audio.max_sample() as f64, 1.0);
    }

    #[test]
    fn test_normalize_zscore() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let audio = audio.normalize(NormalizationConfig::zscore()).unwrap();

        // After Z-score normalization, mean should be ~0 and std dev should be ~1
        let mean: f64 = audio.mean();
        let std_dev: f64 = audio.std_dev();
        assert_approx_eq!(mean as f64, 0.0, 1e-6);
        assert_approx_eq!(std_dev as f64, 1.0, 1e-6);
    }

    #[test]
    fn test_direct_chaining() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        // Test chaining multiple operations using consuming pattern
        let audio = audio.scale(2.0).clip(-5.0, 5.0).unwrap();

        // Verify the result
        let expected = array![2.0f32, 4.0, 5.0, 5.0, 5.0]; // After scaling and clipping
        let result_data = audio.as_mono().unwrap();
        for (actual, expected) in result_data.iter().zip(expected.iter()) {
            assert_approx_eq!(*actual as f64, *expected as f64, 1e-6);
        }
    }

    #[test]
    fn test_chaining_error_handling() {
        let data = array![1.0f32, 2.0, 3.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        // Test that invalid operations return errors
        let result = audio.normalize(NormalizationConfig::min_max(2.0, 1.0)); // Invalid range (min > max)

        assert!(result.is_err());
    }

    #[test]
    fn test_multi_channel_chaining() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let audio = AudioSamples::new_multi_channel(data.into(), sample_rate!(44100)).unwrap();

        let expected = array![[0.5f32, 1.0], [1.5, 2.0]];

        // Test chaining with multi-channel audio
        let audio = audio.scale(0.5);

        // Check the result using the public API instead of accessing data directly
        let result_data = audio.as_multi_channel().unwrap();
        for (actual_row, expected_row) in result_data
            .axis_iter(ndarray::Axis(0))
            .zip(expected.axis_iter(ndarray::Axis(0)))
        {
            for (actual, expected) in actual_row.iter().zip(expected_row.iter()) {
                assert_approx_eq!(*actual as f64, *expected as f64, 1e-6);
            }
        }
    }
}
