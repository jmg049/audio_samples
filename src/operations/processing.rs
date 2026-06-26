//! Signal processing operations for audio samples.
//!
//! This module provides the [`AudioProcessing`] trait implementation for
//! [`AudioSamples`], covering normalization, scaling, filtering, dynamic-range
//! compression, and windowing. Each transforming operation has an in-place
//! primitive (`op_in_place(&mut self, …)`) and a non-mutating twin
//! (`op(&self, …)`, provided by the trait default) that clones then mutates —
//! enabling both zero-allocation in-place pipelines and fluent method chaining.
//!
//! Audio signals routinely need amplitude adjustment, spectral shaping, and
//! dynamic-range management before playback or further analysis. A single trait
//! keeps the API discoverable and composable across all supported sample types.
//!
//! Import [`AudioProcessing`] and call methods directly on an [`AudioSamples`]
//! value. The infallible [`AudioProcessing::scale`] returns `Self` directly,
//! allowing seamless chaining into fallible calls which return
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

#[cfg(feature = "resampling")]
use crate::operations::types::ResamplingQuality;
use crate::operations::types::{NormalizationConfig, NormalizationMethod};
use crate::repr::AudioData;

#[cfg(feature = "resampling")]
use crate::repr::SampleRate;

use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, ConvertTo, LayoutError, ParameterError,
    StandardSample,
    operations::traits::{AudioProcessing, AudioStatistics},
};

use ndarray::{Array2, Axis};
use non_empty_slice::NonEmptySlice;
use num_traits::FloatConst;

/// Valid-region FIR convolution over flat slices, matching [`AudioProcessing::apply_filter`]'s
/// convention: `output[i] = Σ_j input[i + j] · coeffs[j]` for `i in 0..output_len`.
///
/// Working on contiguous `&[T]` slices (rather than enum-dispatched element
/// indexing) lets the compiler autovectorize the inner dot product.
///
/// `output_len` must equal `input.len() - coeffs.len() + 1`.
#[inline]
fn fir_valid_convolve<T>(input: &[T], coeffs: &[T], output_len: usize) -> Vec<T>
where
    T: StandardSample,
{
    let filter_len = coeffs.len();
    let mut output = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let window = &input[i..i + filter_len];
        let mut sum = T::zero();
        for (s, c) in window.iter().zip(coeffs.iter()) {
            sum += *s * *c;
        }
        output.push(sum);
    }
    output
}

/// Tap count at or above which FFT convolution beats direct convolution for
/// whole-signal FIR filtering. Below this the O(N·M) direct loop wins because
/// the FFT's constant overhead dominates; above it the O(N·log N) FFT path pulls
/// far ahead (e.g. ~120x at 4096 taps on a 1 s signal).
#[cfg(feature = "transforms")]
const FIR_FFT_MIN_TAPS: usize = 256;

/// Choose the fastest valid-region FIR convolution for the given filter length.
///
/// With `transforms` enabled, long filters route through FFT convolution
/// (`spectrograms::fft_convolve`); short filters and builds without `transforms`
/// use the direct slice loop. Both produce identical results up to floating-point
/// rounding and preserve `apply_filter`'s valid-region / correlation convention.
#[cfg(feature = "transforms")]
fn fir_valid_convolve_dispatch<T>(
    input: &[T],
    coeffs: &[T],
    output_len: usize,
) -> AudioSampleResult<Vec<T>>
where
    T: StandardSample,
{
    if coeffs.len() >= FIR_FFT_MIN_TAPS {
        fir_valid_convolve_fft(input, coeffs, output_len)
    } else {
        Ok(fir_valid_convolve(input, coeffs, output_len))
    }
}

/// Direct-only dispatch used when the `transforms` feature (and thus the FFT
/// backend) is not available.
#[cfg(not(feature = "transforms"))]
fn fir_valid_convolve_dispatch<T>(
    input: &[T],
    coeffs: &[T],
    output_len: usize,
) -> AudioSampleResult<Vec<T>>
where
    T: StandardSample,
{
    Ok(fir_valid_convolve(input, coeffs, output_len))
}

/// Valid-region FIR convolution via FFT, matching the direct path's convention
/// `output[i] = Σ_j input[i + j] · coeffs[j]`.
///
/// FFT convolution computes true convolution `(x * h)`, so we convolve `x` with
/// the **reversed** coefficients and then take the central "valid" region, which
/// reproduces the correlation convention exactly. Computation is in `f64` for
/// headroom; results match the direct loop to floating-point tolerance.
#[cfg(feature = "transforms")]
fn fir_valid_convolve_fft<T>(
    input: &[T],
    coeffs: &[T],
    output_len: usize,
) -> AudioSampleResult<Vec<T>>
where
    T: StandardSample,
{
    use non_empty_slice::NonEmptySlice;

    let x: Vec<f64> = input.iter().map(|&s| -> f64 { s.convert_to() }).collect();
    let h_rev: Vec<f64> = coeffs.iter().rev().map(|&c| -> f64 { c.convert_to() }).collect();

    let x_ne = NonEmptySlice::new(x.as_slice()).ok_or_else(|| {
        AudioSampleError::Parameter(ParameterError::invalid_value("audio", "empty signal"))
    })?;
    let h_ne = NonEmptySlice::new(h_rev.as_slice()).ok_or_else(|| {
        AudioSampleError::Parameter(ParameterError::invalid_value("filter", "empty filter"))
    })?;

    // Full linear convolution, length x.len() + coeffs.len() - 1.
    let full = crate::operations::fft_convolution::fft_convolve(x_ne, h_ne)?.into_vec();

    // Valid region starts at coeffs.len() - 1 (see module note above).
    let start = coeffs.len() - 1;
    let out: Vec<T> = full[start..start + output_len]
        .iter()
        .map(|&v| -> T { v.convert_to() })
        .collect();
    Ok(out)
}

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
    fn normalize_in_place(
        &mut self,
        config: NormalizationConfig<Self::Sample>,
    ) -> AudioSampleResult<()> {
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
                    match self.data_mut() {
                        AudioData::Mono(arr) => arr.fill(middle),
                        AudioData::Multi(arr) => arr.fill(middle),
                    }
                    return Ok(());
                }

                let current_range = current_max - current_min;
                let target_range = max - min;
                let scale_factor = target_range / current_range;

                match self.data_mut() {
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
                    return Ok(()); // No scaling needed for zero signal
                }

                let target_f64: f64 = target.convert_to();
                let target_peak = target_f64.abs();
                let peak_f64: f64 = peak.convert_to();
                let scale_factor = target_peak / peak_f64;
                let factor_t: T = T::cast_from(scale_factor);

                match self.data_mut() {
                    AudioData::Mono(arr) => arr.mapv_inplace(|x| x * factor_t),
                    AudioData::Multi(arr) => arr.mapv_inplace(|x| x * factor_t),
                }
            }

            NormalizationMethod::Mean => {
                // Mean normalization: subtract mean to center around zero
                let mean: f64 = self.mean();

                match self.data_mut() {
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
                let median: f64 = self.midpoint_sample().ok_or_else(|| {
                    AudioSampleError::Parameter(ParameterError::InvalidValue {
                        parameter: "self".to_string(),
                        reason: "Self is not mono".to_string(),
                    })
                })?;

                match self.data_mut() {
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
                    match self.data_mut() {
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
                    match self.data_mut() {
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

        Ok(())
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
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.scale_in_place(2.0);
    /// assert_eq!(audio[0], 2.0);
    /// assert_eq!(audio[1], -2.0);
    /// ```
    #[inline]
    fn scale_in_place(&mut self, factor: f64) {
        let factor_t: T = T::cast_from(factor);
        match self.data_mut() {
            AudioData::Mono(arr) => arr.mapv_inplace(|x| x * factor_t),
            AudioData::Multi(arr) => arr.mapv_inplace(|x| x * factor_t),
        }
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
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.remove_dc_offset_in_place().unwrap();
    /// let mean: f64 = audio.mean();
    /// assert!(mean.abs() < 1e-6); // Mean is now ~0
    /// ```
    #[inline]
    fn remove_dc_offset_in_place(&mut self) -> AudioSampleResult<()> {
        let mean: f64 = self.mean();

        match self.data_mut() {
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
        Ok(())
    }

    /// Clips all samples in-place to `[min_val, max_val]`.
    ///
    /// Any samples outside `[min_val, max_val]` are clamped to the nearest
    /// boundary. Useful for preventing digital clipping before output. This is
    /// the in-place primitive; [`clip`](AudioProcessing::clip) is the
    /// non-mutating twin.
    ///
    /// # Arguments
    /// - `min_val` — lower bound; samples below this value are set to `min_val`.
    /// - `max_val` — upper bound; samples above this value are set to `max_val`.
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
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.clip_in_place(-1.0f32, 1.0f32).unwrap();
    /// assert_eq!(audio[0], 1.0);
    /// assert_eq!(audio[1], -1.0);
    /// assert_eq!(audio[3], -0.5);
    /// ```
    #[inline]
    fn clip_in_place(
        &mut self,
        min_val: Self::Sample,
        max_val: Self::Sample,
    ) -> AudioSampleResult<()> {
        if min_val > max_val {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "clipping_range",
                format!("min ({min_val:?}) > max ({max_val:?})"),
                format!("{min_val:?}"),
                format!("{max_val:?}"),
                "min value must be less than or equal to max value for clipping",
            )));
        }

        match self.data_mut() {
            AudioData::Mono(arr) => arr.mapv_inplace(|x| x.clamp_to(min_val, max_val)),
            AudioData::Multi(arr) => arr.mapv_inplace(|x| x.clamp_to(min_val, max_val)),
        }
        Ok(())
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
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.apply_window_in_place(NonEmptySlice::new(&window).unwrap()).unwrap();
    /// assert_eq!(audio[0], 1.0);
    /// assert_eq!(audio[1], 0.5);
    /// ```
    #[inline]
    fn apply_window_in_place(
        &mut self,
        window: &NonEmptySlice<Self::Sample>,
    ) -> AudioSampleResult<()> {
        match self.data_mut() {
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
        Ok(())
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
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.apply_filter_in_place(NonEmptySlice::new(&coeffs).unwrap()).unwrap();
    /// assert_eq!(audio[0], 1.5); // 1*0.5 + 2*0.5
    /// assert_eq!(audio[1], 2.5); // 2*0.5 + 3*0.5
    /// ```
    #[inline]
    fn apply_filter_in_place(
        &mut self,
        filter_coeffs: &NonEmptySlice<Self::Sample>,
    ) -> AudioSampleResult<()> {
        // Filter coefficients as a flat slice (used by every dot product below).
        let coeffs: &[Self::Sample] = filter_coeffs.as_ref();
        let filter_len = coeffs.len();

        match self.data_mut() {
            AudioData::Mono(arr) => {
                if arr.len().get() < filter_len {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "audio_length",
                        "Audio length must be at least as long as filter length",
                    )));
                }

                let output_len = arr.len().get() - filter_len + 1;

                // Operate on a flat contiguous slice. The previous version indexed
                // `arr[i + j]`, which dispatched through the `MonoData` enum on every
                // single sample access; a raw slice lets the inner dot product
                // autovectorize and removes that per-element overhead entirely.
                let owned_fallback;
                let input: &[Self::Sample] = match arr.as_slice() {
                    Some(s) => s,
                    None => {
                        owned_fallback = arr.to_vec();
                        &owned_fallback
                    }
                };

                let output = fir_valid_convolve_dispatch(input, coeffs, output_len)?;
                *arr = ndarray::Array1::from_vec(output).try_into()?;
            }
            AudioData::Multi(arr) => {
                if arr.ncols().get() < filter_len {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "audio_length",
                        "Audio length must be at least as long as filter length",
                    )));
                }

                let output_len = arr.ncols().get() - filter_len + 1;
                let num_channels = arr.nrows().get();

                let mut output = Array2::zeros((num_channels, output_len));

                for (ch, mut output_channel) in output.axis_iter_mut(Axis(0)).enumerate() {
                    let row = arr.index_axis(Axis(0), ch);
                    let owned_fallback;
                    let input: &[Self::Sample] = match row.as_slice() {
                        Some(s) => s,
                        None => {
                            owned_fallback = row.to_vec();
                            &owned_fallback
                        }
                    };
                    let filtered = fir_valid_convolve_dispatch(input, coeffs, output_len)?;
                    for (dst, src) in output_channel.iter_mut().zip(filtered) {
                        *dst = src;
                    }
                }

                *arr = output.try_into()?;
            }
        }
        Ok(())
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
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.mu_compress_in_place(255.0).unwrap();
    /// assert!(audio[0] > 0.0); // Positive input stays positive
    /// assert!(audio[1] < 0.0); // Negative input stays negative
    /// ```
    #[inline]
    fn mu_compress_in_place(&mut self, mu: Self::Sample) -> AudioSampleResult<()> {
        let mu_f64: f64 = mu.convert_to();
        let mu_plus_one: f64 = mu_f64 + 1.0;

        self.apply_with_error_in_place(|x: Self::Sample| {
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
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.mu_expand_in_place(255.0).unwrap();
    /// assert_eq!(audio[0], 0.0); // Zero maps to zero
    /// assert!(audio[1] > 0.0);   // Sign is preserved
    /// assert!(audio[2] < 0.0);   // Sign is preserved
    /// ```
    #[inline]
    fn mu_expand_in_place(&mut self, mu: Self::Sample) -> AudioSampleResult<()> {
        let mu: f64 = mu.convert_to();
        let mu_plus_one = mu + 1.0;

        self.apply_with_error_in_place(|x: Self::Sample| {
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
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.low_pass_filter_in_place(1000.0).unwrap();
    /// // High-frequency content is attenuated
    /// assert!(audio[1].abs() < 1.0);
    /// ```
    #[inline]
    fn low_pass_filter_in_place(&mut self, cutoff_hz: f64) -> AudioSampleResult<()> {
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

        match self.data_mut() {
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
        Ok(())
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
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.high_pass_filter_in_place(100.0).unwrap();
    /// // A constant signal is fully removed by a high-pass filter
    /// assert_eq!(audio[0], 0.0);
    /// assert_eq!(audio[3], 0.0);
    /// ```
    #[inline]
    fn high_pass_filter_in_place(&mut self, cutoff_hz: f64) -> AudioSampleResult<()> {
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

        match self.data_mut() {
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
        Ok(())
    }

    /// Applies a band-pass filter that passes frequencies between the two cutoffs.
    ///
    /// Implemented by cascading a high-pass filter at `low_cutoff_hz` followed
    /// by a low-pass filter at `high_cutoff_hz`, both applied in place.
    ///
    /// # Arguments
    /// - `low_cutoff_hz` - Lower cutoff frequency in Hz.
    /// - `high_cutoff_hz` - Upper cutoff frequency in Hz. Must be greater than
    ///   `low_cutoff_hz` and less than the Nyquist frequency.
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
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.band_pass_filter_in_place(100.0, 5000.0).unwrap();
    /// assert!(audio[0].is_finite());
    /// ```
    #[inline]
    fn band_pass_filter_in_place(
        &mut self,
        low_cutoff_hz: f64,
        high_cutoff_hz: f64,
    ) -> AudioSampleResult<()> {
        if low_cutoff_hz >= high_cutoff_hz {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "frequency_range",
                "Low frequency must be less than high frequency",
            )));
        }

        // Apply high-pass filter first, then low-pass.
        self.high_pass_filter_in_place(low_cutoff_hz)?;
        self.low_pass_filter_in_place(high_cutoff_hz)
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

    /// Resamples audio to a new sample rate in place, replacing the internal
    /// buffer with the resampled result.
    ///
    /// Explicit in-place twin of [`resample`](AudioProcessing::resample); see that
    /// method for details. Computes the resampled buffer and assigns it back to
    /// `self`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the target sample rate or
    ///   quality parameters are invalid.
    /// - [`crate::AudioSampleError::Layout`] if the input audio is empty.
    #[cfg(feature = "resampling")]
    #[inline]
    fn resample_in_place(
        &mut self,
        target_sample_rate: SampleRate,
        quality: ResamplingQuality,
    ) -> AudioSampleResult<()> {
        *self = self.resample(target_sample_rate, quality)?.into_owned();
        Ok(())
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

    /// Resamples audio by a specific ratio in place, replacing the internal
    /// buffer with the resampled result.
    ///
    /// Explicit in-place twin of
    /// [`resample_by_ratio`](AudioProcessing::resample_by_ratio); see that method
    /// for details. Computes the resampled buffer and assigns it back to `self`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `ratio` is ≤ 0 or the
    ///   quality parameters are invalid.
    /// - [`crate::AudioSampleError::Layout`] if the input audio is empty.
    #[cfg(feature = "resampling")]
    #[inline]
    fn resample_by_ratio_in_place(
        &mut self,
        ratio: f64,
        quality: ResamplingQuality,
    ) -> AudioSampleResult<()> {
        *self = self.resample_by_ratio(ratio, quality)?.into_owned();
        Ok(())
    }
}

// Helper methods for the AudioProcessing implementation
impl<T> AudioSamples<'_, T>
where
    T: StandardSample,
{
    /// Applies a fallible function to all samples in the audio data in place.
    /// This is a helper method for operations that can fail.
    ///
    /// # Examples
    /// ```rust,ignore
    /// // Apply a conversion that might fail
    /// audio.apply_with_error_in_place(|sample| {
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
    pub fn apply_with_error_in_place<F>(&mut self, f: F) -> AudioSampleResult<()>
    where
        F: Fn(T) -> AudioSampleResult<T>,
    {
        match self.data_mut() {
            AudioData::Mono(arr) => {
                for x in arr.iter_mut() {
                    *x = f(*x)?;
                }
            }
            AudioData::Multi(arr) => {
                for x in arr.iter_mut() {
                    *x = f(*x)?;
                }
            }
        }
        Ok(())
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
        match self.data_mut() {
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

    /// The FFT fast path (filter >= FIR_FFT_MIN_TAPS) must match the direct
    /// slice convolution to floating-point tolerance, preserving apply_filter's
    /// valid-region / correlation convention.
    #[cfg(feature = "transforms")]
    #[test]
    fn apply_filter_fft_matches_direct_long_filter() {
        let n = 4000usize;
        let m = 512usize; // >= FIR_FFT_MIN_TAPS, so this exercises the FFT path
        assert!(m >= FIR_FFT_MIN_TAPS);

        let input: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.013).sin() + 0.4 * (i as f32 * 0.07).cos())
            .collect();
        let coeffs: Vec<f32> = (0..m)
            .map(|k| ((k as f32 * 0.05).sin()) / m as f32)
            .collect();

        // Reference: direct slice convolution (same convention).
        let output_len = n - m + 1;
        let reference = fir_valid_convolve(&input, &coeffs, output_len);

        // Public API path (routes to FFT because m >= threshold).
        let audio: AudioSamples<f32> =
            AudioSamples::new_mono(ndarray::Array1::from_vec(input), sample_rate!(44100)).unwrap();
        let filtered = audio
            .apply_filter(NonEmptySlice::new(&coeffs).unwrap())
            .unwrap();

        assert_eq!(filtered.samples_per_channel().get(), output_len);
        for (i, &want) in reference.iter().enumerate() {
            let got = filtered[i];
            assert!(
                (got - want).abs() < 1e-3,
                "sample {i}: fft {got} vs direct {want}"
            );
        }
    }

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
        match audio.data() {
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
        match audio.data() {
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
        match audio.data() {
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
            AudioSamples::new_multi_channel(data, sample_rate!(44100)).unwrap();

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
        let audio = AudioSamples::new_multi_channel(data, sample_rate!(44100)).unwrap();

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

    /// Helper: assert two mono signals are element-wise approximately equal.
    fn assert_mono_eq(a: &AudioSamples<f32>, b: &AudioSamples<f32>) {
        let a = a.as_mono().unwrap();
        let b = b.as_mono().unwrap();
        assert_eq!(a.len(), b.len(), "lengths differ");
        for (x, y) in a.iter().zip(b.iter()) {
            assert_approx_eq!(*x as f64, *y as f64, 1e-6);
        }
    }

    /// `normalize` (non-mutating) and `normalize_in_place` must produce identical
    /// results, and the non-mutating variant must leave the original untouched.
    #[test]
    fn test_normalize_dual_variant_equivalence() {
        let data = array![1.0f32, -3.0, 2.0, -1.0, 0.5];
        let original = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        // Non-mutating variant.
        let borrowed = original
            .normalize(NormalizationConfig::peak(1.0))
            .unwrap();

        // The original must be unchanged.
        assert_mono_eq(
            &original,
            &AudioSamples::new_mono(
                array![1.0f32, -3.0, 2.0, -1.0, 0.5],
                sample_rate!(44100),
            )
            .unwrap(),
        );

        // In-place variant on a clone.
        let mut mutated = original.clone();
        mutated
            .normalize_in_place(NormalizationConfig::peak(1.0))
            .unwrap();

        // Both variants must agree.
        assert_mono_eq(&borrowed, &mutated);
    }

    /// `scale` (non-mutating, infallible) and `scale_in_place` must produce
    /// identical results, and the non-mutating variant must leave the original
    /// untouched.
    #[test]
    fn test_scale_dual_variant_equivalence() {
        let original =
            AudioSamples::new_mono(array![1.0f32, -2.0, 3.0, -4.0], sample_rate!(44100)).unwrap();

        let borrowed = original.scale(0.25);

        // Original unchanged.
        assert_mono_eq(
            &original,
            &AudioSamples::new_mono(array![1.0f32, -2.0, 3.0, -4.0], sample_rate!(44100)).unwrap(),
        );

        let mut mutated = original.clone();
        mutated.scale_in_place(0.25);

        assert_mono_eq(&borrowed, &mutated);
    }

    /// `clip` (non-mutating) and `clip_in_place` must produce identical results,
    /// and the non-mutating variant must leave the original untouched.
    #[test]
    fn test_clip_dual_variant_equivalence() {
        let original =
            AudioSamples::new_mono(array![2.0f32, -3.0, 1.5, -0.5], sample_rate!(44100)).unwrap();

        let borrowed = original.clip(-1.0, 1.0).unwrap();

        // Original unchanged.
        assert_mono_eq(
            &original,
            &AudioSamples::new_mono(array![2.0f32, -3.0, 1.5, -0.5], sample_rate!(44100)).unwrap(),
        );

        let mut mutated = original.clone();
        mutated.clip_in_place(-1.0, 1.0).unwrap();

        assert_mono_eq(&borrowed, &mutated);
    }
}
