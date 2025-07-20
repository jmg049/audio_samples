//! Signal processing operations for AudioSamples.
//!
//! This module implements the AudioProcessing trait, providing comprehensive
//! signal processing operations including normalization, filtering, compression,
//! and envelope operations using efficient ndarray operations.

use super::types::NormalizationMethod;
use crate::repr::AudioData;
use crate::{
    AudioProcessing, AudioSample, AudioSampleError, AudioSampleResult, AudioSamples,
    AudioStatistics, AudioTypeConversion, CastFrom, CastInto, ConvertTo, I24,
};
use ndarray::Axis;

impl<T: AudioSample> AudioProcessing<T> for AudioSamples<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
    T: CastInto<i16>
        + CastInto<I24>
        + CastInto<i32>
        + CastInto<f32>
        + CastInto<f64>
        + CastFrom<i16>
        + CastFrom<I24>
        + CastFrom<i32>
        + CastFrom<f32>
        + CastFrom<f64>,
{
    /// Normalizes audio samples using the specified method and range.
    ///
    /// This method modifies the audio samples in-place to fit within the target range
    /// using different normalization strategies.
    fn normalize(&mut self, min: T, max: T, method: NormalizationMethod) -> AudioSampleResult<()> {
        // Validate input range
        if min >= max {
            return Err(AudioSampleError::InvalidRange(format!(
                "Invalid normalization range: min ({:?}) >= max ({:?})",
                min, max
            )));
        }

        match method {
            NormalizationMethod::MinMax => {
                // Min-Max normalization: scale to [min, max] range
                let current_min = self.min_native();
                let current_max = self.max_native();

                // Avoid division by zero
                if current_min == current_max {
                    // All values are the same, set to middle of target range
                    let middle = min + (max - min) / T::cast_from(2.0f64);
                    match &mut self.data {
                        AudioData::Mono(arr) => arr.fill(middle),
                        AudioData::MultiChannel(arr) => arr.fill(middle),
                    }
                    return Ok(());
                }

                let current_range = current_max - current_min;
                let target_range = max - min;
                let scale_factor = target_range / current_range;

                match &mut self.data {
                    AudioData::Mono(arr) => {
                        arr.mapv_inplace(|x| min + (x - current_min) * scale_factor);
                    }
                    AudioData::MultiChannel(arr) => {
                        arr.mapv_inplace(|x| min + (x - current_min) * scale_factor);
                    }
                }
            }

            NormalizationMethod::Peak => {
                // Peak normalization: scale by peak value
                let peak = self.peak();
                if peak == T::zero() {
                    return Ok(()); // No scaling needed for zero signal
                }
                let min_f64: f64 = min.convert_to().unwrap_or(0.0f64).abs();
                let max_f64: f64 = max.convert_to().unwrap_or(0.0f64).abs();

                let target_peak: f64 = min_f64.max(max_f64);
                let scale_factor = target_peak / peak.convert_to().unwrap_or(1.0f64);

                match &mut self.data {
                    AudioData::Mono(arr) => {
                        arr.mapv_inplace(|x| {
                            <f64 as ConvertTo<T>>::convert_to(
                                &(x.convert_to().unwrap_or(0.0) * scale_factor),
                            )
                            .unwrap_or(T::zero())
                        });
                    }
                    AudioData::MultiChannel(arr) => {
                        arr.mapv_inplace(|x| {
                            <f64 as ConvertTo<T>>::convert_to(
                                &(x.convert_to().unwrap_or(0.0) * scale_factor),
                            )
                            .unwrap_or(T::zero())
                        });
                    }
                }
            }

            NormalizationMethod::Mean => {
                // Mean normalization: subtract mean to center around zero
                let mean = self.compute_mean()?;

                match &mut self.data {
                    AudioData::Mono(arr) => {
                        arr.mapv_inplace(|x| x - mean);
                    }
                    AudioData::MultiChannel(arr) => {
                        arr.mapv_inplace(|x| x - mean);
                    }
                }
            }

            NormalizationMethod::Median => {
                // Median normalization: subtract median to center around zero
                let median = self.compute_median()?;

                match &mut self.data {
                    AudioData::Mono(arr) => {
                        arr.mapv_inplace(|x| x - median);
                    }
                    AudioData::MultiChannel(arr) => {
                        arr.mapv_inplace(|x| x - median);
                    }
                }
            }

            NormalizationMethod::ZScore => {
                // Z-Score normalization: zero mean, unit variance
                let mean = self.compute_mean()?;
                let std_dev = self.compute_std_dev()?;

                if std_dev == T::zero() {
                    // All values are the same, just subtract mean
                    match &mut self.data {
                        AudioData::Mono(arr) => {
                            arr.mapv_inplace(|x| x - mean);
                        }
                        AudioData::MultiChannel(arr) => {
                            arr.mapv_inplace(|x| x - mean);
                        }
                    }
                } else {
                    match &mut self.data {
                        AudioData::Mono(arr) => {
                            arr.mapv_inplace(|x| (x - mean) / std_dev);
                        }
                        AudioData::MultiChannel(arr) => {
                            arr.mapv_inplace(|x| (x - mean) / std_dev);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Scales all audio samples by a constant factor.
    ///
    /// This is equivalent to adjusting the volume/amplitude of the signal.
    fn scale(&mut self, factor: T) -> AudioSampleResult<()> {
        match &mut self.data {
            AudioData::Mono(arr) => {
                arr.mapv_inplace(|x| x * factor);
            }
            AudioData::MultiChannel(arr) => {
                arr.mapv_inplace(|x| x * factor);
            }
        }
        Ok(())
    }

    /// Removes DC offset by subtracting the mean value.
    ///
    /// This centers the audio around zero and removes any constant bias.
    fn remove_dc_offset(&mut self) -> AudioSampleResult<()> {
        let mean = self.compute_mean()?;

        match &mut self.data {
            AudioData::Mono(arr) => {
                arr.mapv_inplace(|x| x - mean);
            }
            AudioData::MultiChannel(arr) => {
                arr.mapv_inplace(|x| x - mean);
            }
        }
        Ok(())
    }

    /// Clips audio samples to the specified range.
    ///
    /// Any samples outside the range will be limited to the range boundaries.
    fn clip(&mut self, min_val: T, max_val: T) -> AudioSampleResult<()> {
        if min_val > max_val {
            return Err(AudioSampleError::InvalidRange(format!(
                "Invalid clipping range: min ({:?}) > max ({:?})",
                min_val, max_val
            )));
        }

        match &mut self.data {
            AudioData::Mono(arr) => {
                arr.mapv_inplace(|x| {
                    if x < min_val {
                        min_val
                    } else if x > max_val {
                        max_val
                    } else {
                        x
                    }
                });
            }
            AudioData::MultiChannel(arr) => {
                arr.mapv_inplace(|x| {
                    if x < min_val {
                        min_val
                    } else if x > max_val {
                        max_val
                    } else {
                        x
                    }
                });
            }
        }
        Ok(())
    }

    /// Applies a windowing function to the audio samples.
    ///
    /// The window length must match the number of samples in the audio.
    fn apply_window(&mut self, window: &[T]) -> AudioSampleResult<()> {
        match &mut self.data {
            AudioData::Mono(arr) => {
                if window.len() != arr.len() {
                    return Err(AudioSampleError::DimensionMismatch(format!(
                        "Window length ({}) doesn't match audio length ({})",
                        window.len(),
                        arr.len()
                    )));
                }

                for (sample, &win_coeff) in arr.iter_mut().zip(window.iter()) {
                    *sample = *sample * win_coeff;
                }
            }
            AudioData::MultiChannel(arr) => {
                let num_samples = arr.ncols();
                if window.len() != num_samples {
                    return Err(AudioSampleError::DimensionMismatch(format!(
                        "Window length ({}) doesn't match audio length ({})",
                        window.len(),
                        num_samples
                    )));
                }

                // Apply window to each channel
                for mut channel in arr.axis_iter_mut(Axis(0)) {
                    for (sample, &win_coeff) in channel.iter_mut().zip(window.iter()) {
                        *sample = *sample * win_coeff;
                    }
                }
            }
        }
        Ok(())
    }

    /// Applies a digital filter to the audio samples using convolution.
    ///
    /// This implements basic FIR filtering using direct convolution.
    fn apply_filter(&mut self, filter_coeffs: &[T]) -> AudioSampleResult<()> {
        if filter_coeffs.is_empty() {
            return Err(AudioSampleError::InvalidParameter(
                "Filter coefficients cannot be empty".to_string(),
            ));
        }

        match &mut self.data {
            AudioData::Mono(arr) => {
                if arr.len() < filter_coeffs.len() {
                    return Err(AudioSampleError::InvalidParameter(
                        "Audio length must be at least as long as filter length".to_string(),
                    ));
                }

                let input = arr.to_vec(); // Copy original data
                let filter_len = filter_coeffs.len();
                let output_len = input.len() - filter_len + 1;

                // Perform convolution
                for i in 0..output_len {
                    let mut sum = T::zero();
                    for j in 0..filter_len {
                        sum = sum + input[i + j] * filter_coeffs[j];
                    }
                    arr[i] = sum;
                }

                // Truncate array to output length
                *arr = arr.slice(ndarray::s![..output_len]).to_owned();
            }
            AudioData::MultiChannel(arr) => {
                if arr.ncols() < filter_coeffs.len() {
                    return Err(AudioSampleError::InvalidParameter(
                        "Audio length must be at least as long as filter length".to_string(),
                    ));
                }

                let filter_len = filter_coeffs.len();
                let output_len = arr.ncols() - filter_len + 1;
                let num_channels = arr.nrows();

                // Create output buffer
                let mut output = ndarray::Array2::zeros((num_channels, output_len));

                // Apply filter to each channel
                for (ch, mut output_channel) in output.axis_iter_mut(Axis(0)).enumerate() {
                    let input_channel = arr.row(ch);

                    for i in 0..output_len {
                        let mut sum = T::zero();
                        for j in 0..filter_len {
                            sum = sum + input_channel[i + j] * filter_coeffs[j];
                        }
                        output_channel[i] = sum;
                    }
                }

                *arr = output;
            }
        }
        Ok(())
    }

    /// Applies μ-law compression to the audio samples.
    fn mu_compress(&mut self, mu: T) -> AudioSampleResult<()> {
        let mu_f64: f64 = mu.convert_to()?;
        let mu_plus_one: f64 = mu_f64 + 1.0;

        let mu_fn = |x: T| {
            let x: f64 = match x.convert_to() {
                Ok(val) => val,
                Err(e) => {
                    return Err(AudioSampleError::ConversionError(
                        e.to_string(),
                        "f64".to_string(),
                        std::any::type_name::<T>().to_string(),
                        "mu_compress".to_string(),
                    ));
                }
            };
            let sign = if x >= 0.0 { 1.0 } else { -1.0 };
            let abs_x = x.abs();
            let compressed = sign * (mu_plus_one.ln() + mu_f64 * abs_x).ln() / mu_plus_one.ln();
            T::convert_from(compressed)
        };

        match &mut self.data {
            AudioData::Mono(arr) => {
                for x in arr.iter_mut() {
                    *x = mu_fn(*x)?;
                }
                Ok(())
            }
            AudioData::MultiChannel(arr) => {
                for x in arr.iter_mut() {
                    *x = mu_fn(*x)?;
                }
                Ok(())
            }
        }
    }

    /// Applies μ-law expansion (decompression) to the audio samples.
    ///
    /// Fist the mu value is converted to f64, then the expansion is applied.
    /// Second, the result is converted back to T.
    fn mu_expand(&mut self, mu: T) -> AudioSampleResult<()> {
        let mu: f64 = mu.convert_to()?;
        let mu_plus_one = mu + 1.0;

        match &mut self.data {
            AudioData::Mono(arr) => {
                for x in arr.iter_mut() {
                    let x_f64: f64 = x.convert_to()?;
                    let sign = if x_f64 >= 0.0 { 1.0 } else { -1.0 };
                    let abs_x = x_f64.abs();
                    let expanded = sign * (mu_plus_one.powf(abs_x) - 1.0) / mu;
                    *x = T::convert_from(expanded)?;
                }
            }
            AudioData::MultiChannel(arr) => {
                for x in arr.iter_mut() {
                    let x_f64: f64 = x.convert_to()?;
                    let sign = if x_f64 >= 0.0 { 1.0 } else { -1.0 };
                    let abs_x = x_f64.abs();
                    let expanded = sign * (mu_plus_one.powf(abs_x) - 1.0) / mu;
                    *x = T::convert_from(expanded)?;
                }
            }
        }
        Ok(())
    }

    /// Applies a low-pass filter with the specified cutoff frequency.
    fn low_pass_filter(&mut self, cutoff_hz: f64) -> AudioSampleResult<()> {
        // Simple implementation using a basic low-pass filter design
        let sample_rate = self.sample_rate() as f64;
        let normalized_cutoff = cutoff_hz / sample_rate;

        if normalized_cutoff >= 0.5 {
            return Err(AudioSampleError::InvalidParameter(
                "Cutoff frequency must be less than Nyquist frequency".to_string(),
            ));
        }

        // Simple single-pole low-pass filter coefficient
        let alpha = 2.0 * std::f64::consts::PI * normalized_cutoff;
        let one_minus_alpha = 1.0 - alpha;

        match &mut self.data {
            AudioData::Mono(arr) => {
                if !arr.is_empty() {
                    let mut prev_output: f64 = arr[0].convert_to()?;
                    for sample in arr.iter_mut() {
                        let s: f64 = sample.convert_to()?;
                        let s: f64 = alpha * s + one_minus_alpha * prev_output;
                        prev_output = s;
                        *sample = T::convert_from(s)?;
                    }
                }
            }
            AudioData::MultiChannel(arr) => {
                for mut channel in arr.axis_iter_mut(Axis(0)) {
                    if !channel.is_empty() {
                        let mut prev_output = channel[0].convert_to()?;
                        for sample in channel.iter_mut() {
                            let s: f64 = sample.convert_to()?;
                            let s: f64 = alpha * s + one_minus_alpha * prev_output;
                            prev_output = s;
                            *sample = T::convert_from(s)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Applies a high-pass filter with the specified cutoff frequency.
    fn high_pass_filter(&mut self, cutoff_hz: f64) -> AudioSampleResult<()> {
        // Simple implementation using a basic high-pass filter design
        let sample_rate = self.sample_rate() as f64;
        let normalized_cutoff = cutoff_hz / sample_rate;

        if normalized_cutoff >= 0.5 {
            return Err(AudioSampleError::InvalidParameter(
                "Cutoff frequency must be less than Nyquist frequency".to_string(),
            ));
        }

        // Simple high-pass filter using RC circuit model
        let rc = 1.0 / (2.0 * std::f64::consts::PI * cutoff_hz);
        let dt = 1.0 / sample_rate;
        let alpha = rc / (rc + dt);

        match &mut self.data {
            AudioData::Mono(arr) => {
                if arr.len() > 1 {
                    let mut prev_input = arr[0];
                    let mut prev_output = T::zero();

                    for sample in arr.iter_mut() {
                        let current = *sample;
                        *sample = T::cast_from(alpha) * (prev_output + current - prev_input);
                        prev_input = current;
                        prev_output = *sample;
                    }
                }
            }
            AudioData::MultiChannel(arr) => {
                for mut channel in arr.axis_iter_mut(Axis(0)) {
                    if channel.len() > 1 {
                        let mut prev_input = channel[0];
                        let mut prev_output = T::zero();

                        for sample in channel.iter_mut() {
                            let current = *sample;
                            *sample = T::cast_from(alpha) * (prev_output + current - prev_input);
                            prev_input = current;
                            prev_output = *sample;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Applies a band-pass filter between low and high frequencies.
    fn band_pass_filter(&mut self, low_hz: f64, high_hz: f64) -> AudioSampleResult<()> {
        if low_hz >= high_hz {
            return Err(AudioSampleError::InvalidParameter(
                "Low frequency must be less than high frequency".to_string(),
            ));
        }

        // Apply high-pass filter first, then low-pass
        self.high_pass_filter(low_hz)?;
        self.low_pass_filter(high_hz)?;

        Ok(())
    }

    /// Resamples audio to a new sample rate using high-quality algorithms.
    fn resample(
        &self,
        target_sample_rate: usize,
        quality: super::types::ResamplingQuality,
    ) -> AudioSampleResult<Self>
    where
        Self: Sized,
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
        AudioSamples<T>: AudioTypeConversion<T>,
    {
        crate::resampling::resample(self, target_sample_rate, quality)
    }

    /// Resamples audio by a specific ratio.
    fn resample_by_ratio(
        &self,
        ratio: f64,
        quality: super::types::ResamplingQuality,
    ) -> AudioSampleResult<Self>
    where
        Self: Sized,
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
        AudioSamples<T>: AudioTypeConversion<T>,
    {
        crate::resampling::resample_by_ratio(self, ratio, quality)
    }
}

// Helper methods for the AudioProcessing implementation
impl<T: AudioSample> AudioSamples<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    /// Computes the mean value of all samples.
    fn compute_mean(&self) -> AudioSampleResult<T> {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() {
                    return Ok(T::zero());
                }
                let sum = arr.sum();
                let sum: f64 = sum.convert_to()?; // Convert to f64 for division
                // Convert the final result back to T.
                // Under the AudioSamples assumptions, which means we will have already verified that T has a valid value for the given sample type,
                // T in any case is :
                // (a) convertible to/from f64 (and any other supported type), and
                // (b) therefore equivalent, an audio sample represented as f64 is equivalent to the same sample represented as i16 (WITHIN SUPPORTED RANGE)
                Ok((sum / arr.len() as f64).convert_to()?)
            }
            AudioData::MultiChannel(arr) => {
                if arr.is_empty() {
                    return Ok(T::zero());
                }
                let sum = arr.sum();
                let sum: f64 = sum.convert_to()?; // Convert to f64 for division
                // Convert the final result back to T.
                // Under the AudioSamples assumptions, which means we will have already verified that T has a valid value for the given sample type,
                // T in any case is :
                // (a) convertible to/from f64 (and any other supported type), and
                // (b) therefore equivalent, an audio sample represented as f64 is equivalent to the same sample represented as i16 (WITHIN SUPPORTED RANGE)
                Ok((sum / arr.len() as f64).convert_to()?)
            }
        }
    }

    /// Computes the median value of all samples.
    fn compute_median(&self) -> AudioSampleResult<T>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
        AudioSamples<T>: AudioTypeConversion<T>,
    {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() {
                    return Ok(T::zero());
                }
                let mut values: Vec<T> = arr.clone().into_raw_vec_and_offset().0;
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let median = if values.len() % 2 == 0 {
                    let mid = values.len() / 2;
                    (<T as ConvertTo<f64>>::convert_to(&(values[mid - 1] + values[mid]))? / 2.0f64)
                        .convert_to()?
                } else {
                    values[values.len() / 2]
                };

                Ok(median)
            }
            AudioData::MultiChannel(arr) => {
                if arr.is_empty() {
                    return Ok(T::zero());
                }
                let mut values: Vec<T> = arr.clone().into_raw_vec_and_offset().0;
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let median = if values.len() % 2 == 0 {
                    let mid = values.len() / 2;
                    (<T as ConvertTo<f64>>::convert_to(&(values[mid - 1] + values[mid]))? / 2.0f64)
                        .convert_to()?
                } else {
                    values[values.len() / 2]
                };

                Ok(T::convert_from(median).unwrap_or(T::zero()))
            }
        }
    }

    /// Computes the standard deviation of all samples.
    fn compute_std_dev(&self) -> AudioSampleResult<T>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
        AudioSamples<T>: AudioTypeConversion<T>,
    {
        let mean = self.compute_mean()?;

        match &self.data {
            AudioData::Mono(arr) => {
                if arr.len() <= 1 {
                    return Ok(T::zero());
                }

                let variance_sum = arr
                    .mapv(|x| {
                        let val = x;
                        let diff = val - mean;
                        diff * diff
                    })
                    .sum();

                let variance_sum: f64 = variance_sum.convert_to()?;
                let variance: f64 = variance_sum / arr.len() as f64;
                let variance_sqrt = variance.sqrt();
                let variance: T = T::convert_from(variance_sqrt)?;
                Ok(variance)
            }
            AudioData::MultiChannel(arr) => {
                if arr.len() <= 1 {
                    return Ok(T::zero());
                }

                let variance_sum = arr
                    .mapv(|x| {
                        let diff = x - mean;
                        diff * diff
                    })
                    .sum();

                let variance_sum: f64 = variance_sum.convert_to()?;
                let variance: f64 = variance_sum / arr.len() as f64;
                let variance_sqrt = variance.sqrt();
                let variance: T = T::convert_from(variance_sqrt)?;
                Ok(variance)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;
    use ndarray::array;

    use crate::AudioProcessing;

    #[test]
    fn test_normalize_min_max() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut audio = AudioSamples::new_mono(data, 44100);

        audio
            .normalize(-1.0, 1.0, NormalizationMethod::MinMax)
            .unwrap();

        assert_approx_eq!(audio.min() as f64, -1.0);
        assert_approx_eq!(audio.max() as f64, 1.0);
    }

    #[test]
    fn test_normalize_peak() {
        let data = array![-2.0f32, 1.0, 3.0, -1.0];
        let mut audio = AudioSamples::new_mono(data, 44100);

        audio
            .normalize(-1.0, 1.0, NormalizationMethod::Peak)
            .unwrap();

        assert_approx_eq!(audio.peak() as f64, 1.0);
    }

    #[test]
    fn test_scale() {
        let data = array![1.0f32, 2.0, 3.0];
        let mut audio = AudioSamples::new_mono(data, 44100);

        audio.scale(2.0).unwrap();

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
        let mut audio = AudioSamples::new_mono(data, 44100);

        audio.remove_dc_offset().unwrap();

        let mean = audio.compute_mean().expect("Mean calculation failed");
        assert_approx_eq!(mean as f64, 0.0, 1e-6);
    }

    #[test]
    fn test_clip() {
        let data = array![-3.0f32, -1.0, 0.0, 1.0, 3.0];
        let mut audio = AudioSamples::new_mono(data, 44100);

        audio.clip(-2.0, 2.0).unwrap();

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
        let mut audio = AudioSamples::new_mono(data, 44100);
        let window = [0.5f32, 1.0, 1.0, 0.5];

        audio.apply_window(&window).unwrap();

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
        let mut audio = AudioSamples::new_multi_channel(data, 44100);

        audio
            .normalize(-1.0, 1.0, NormalizationMethod::MinMax)
            .unwrap();

        assert_approx_eq!(audio.min() as f64, -1.0);
        assert_approx_eq!(audio.max() as f64, 1.0);
    }

    #[test]
    fn test_normalize_zscore() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut audio = AudioSamples::new_mono(data, 44100);

        audio
            .normalize(-1.0, 1.0, NormalizationMethod::ZScore)
            .unwrap();

        // After Z-score normalization, mean should be ~0 and std dev should be ~1
        let mean = audio.compute_mean().expect("Mean calculation failed");
        let std_dev = audio.compute_std_dev().expect("Std dev calculation failed");
        assert_approx_eq!(mean as f64, 0.0, 1e-6);
        assert_approx_eq!(std_dev as f64, 1.0, 1e-6);
    }
}
