//! Signal processing operations for AudioSamples.
//!
//! This module implements the AudioProcessing trait, providing comprehensive
//! signal processing operations including normalization, filtering, compression,
//! and envelope operations using efficient ndarray operations.

use super::traits::{AudioProcessing, AudioStatistics};
use super::types::NormalizationMethod;
use crate::repr::AudioData;
use crate::{AudioSample, AudioSampleError, AudioSampleResult, AudioSamples};
use ndarray::Axis;
use num_traits::{Float, FromPrimitive, ToPrimitive, Zero};

impl<T: AudioSample + ToPrimitive + FromPrimitive + Zero + Float> AudioProcessing<T>
    for AudioSamples<T>
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
                    let middle = min + (max - min) / T::from(2.0).unwrap();
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

                let target_peak = T::max(min.abs(), max.abs());
                let scale_factor = target_peak / peak;

                match &mut self.data {
                    AudioData::Mono(arr) => {
                        arr.mapv_inplace(|x| x * scale_factor);
                    }
                    AudioData::MultiChannel(arr) => {
                        arr.mapv_inplace(|x| x * scale_factor);
                    }
                }
            }

            NormalizationMethod::Mean => {
                // Mean normalization: subtract mean to center around zero
                let mean = self.compute_mean();

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
                let median = self.compute_median();

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
                let mean = self.compute_mean();
                let std_dev = self.compute_std_dev();

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
        let mean = self.compute_mean();

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
        let mu_f64 = mu.to_f64().unwrap_or(255.0);
        let mu_plus_one = mu_f64 + 1.0;

        match &mut self.data {
            AudioData::Mono(arr) => {
                arr.mapv_inplace(|x| {
                    let x_f64 = x.to_f64().unwrap_or(0.0);
                    let sign = if x_f64 >= 0.0 { 1.0 } else { -1.0 };
                    let abs_x = x_f64.abs();
                    let compressed =
                        sign * (mu_plus_one.ln() + mu_f64 * abs_x).ln() / mu_plus_one.ln();
                    T::from_f64(compressed).unwrap_or(T::zero())
                });
            }
            AudioData::MultiChannel(arr) => {
                arr.mapv_inplace(|x| {
                    let x_f64 = x.to_f64().unwrap_or(0.0);
                    let sign = if x_f64 >= 0.0 { 1.0 } else { -1.0 };
                    let abs_x = x_f64.abs();
                    let compressed =
                        sign * (mu_plus_one.ln() + mu_f64 * abs_x).ln() / mu_plus_one.ln();
                    T::from_f64(compressed).unwrap_or(T::zero())
                });
            }
        }
        Ok(())
    }

    /// Applies μ-law expansion (decompression) to the audio samples.
    fn mu_expand(&mut self, mu: T) -> AudioSampleResult<()> {
        let mu_f64 = mu.to_f64().unwrap_or(255.0);
        let mu_plus_one = mu_f64 + 1.0;

        match &mut self.data {
            AudioData::Mono(arr) => {
                arr.mapv_inplace(|x| {
                    let x_f64 = x.to_f64().unwrap_or(0.0);
                    let sign = if x_f64 >= 0.0 { 1.0 } else { -1.0 };
                    let abs_x = x_f64.abs();
                    let expanded = sign * (mu_plus_one.powf(abs_x) - 1.0) / mu_f64;
                    T::from_f64(expanded).unwrap_or(T::zero())
                });
            }
            AudioData::MultiChannel(arr) => {
                arr.mapv_inplace(|x| {
                    let x_f64 = x.to_f64().unwrap_or(0.0);
                    let sign = if x_f64 >= 0.0 { 1.0 } else { -1.0 };
                    let abs_x = x_f64.abs();
                    let expanded = sign * (mu_plus_one.powf(abs_x) - 1.0) / mu_f64;
                    T::from_f64(expanded).unwrap_or(T::zero())
                });
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
        let alpha = T::from_f64(2.0 * std::f64::consts::PI * normalized_cutoff).unwrap();
        let one_minus_alpha = T::from_f64(1.0).unwrap() - alpha;

        match &mut self.data {
            AudioData::Mono(arr) => {
                if !arr.is_empty() {
                    let mut prev_output = arr[0];
                    for sample in arr.iter_mut() {
                        *sample = alpha * *sample + one_minus_alpha * prev_output;
                        prev_output = *sample;
                    }
                }
            }
            AudioData::MultiChannel(arr) => {
                for mut channel in arr.axis_iter_mut(Axis(0)) {
                    if !channel.is_empty() {
                        let mut prev_output = channel[0];
                        for sample in channel.iter_mut() {
                            *sample = alpha * *sample + one_minus_alpha * prev_output;
                            prev_output = *sample;
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
        let alpha = T::from_f64(rc / (rc + dt)).unwrap();

        match &mut self.data {
            AudioData::Mono(arr) => {
                if arr.len() > 1 {
                    let mut prev_input = arr[0];
                    let mut prev_output = T::zero();

                    for sample in arr.iter_mut() {
                        let current = *sample;
                        *sample = alpha * (prev_output + current - prev_input);
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
        T: num_traits::Float
            + num_traits::FromPrimitive
            + num_traits::ToPrimitive
            + crate::ConvertTo<f64>,
        f64: crate::ConvertTo<T>,
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
        T: num_traits::Float
            + num_traits::FromPrimitive
            + num_traits::ToPrimitive
            + crate::ConvertTo<f64>,
        f64: crate::ConvertTo<T>,
    {
        crate::resampling::resample_by_ratio(self, ratio, quality)
    }
}

// Helper methods for the AudioProcessing implementation
impl<T: AudioSample + ToPrimitive + FromPrimitive + Zero + Float> AudioSamples<T> {
    /// Computes the mean value of all samples.
    fn compute_mean(&self) -> T {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() {
                    return T::zero();
                }
                let sum = arr.mapv(|x| x.to_f64().unwrap_or(0.0)).sum();
                T::from_f64(sum / arr.len() as f64).unwrap_or(T::zero())
            }
            AudioData::MultiChannel(arr) => {
                if arr.is_empty() {
                    return T::zero();
                }
                let sum = arr.mapv(|x| x.to_f64().unwrap_or(0.0)).sum();
                T::from_f64(sum / arr.len() as f64).unwrap_or(T::zero())
            }
        }
    }

    /// Computes the median value of all samples.
    fn compute_median(&self) -> T {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() {
                    return T::zero();
                }
                let mut values: Vec<f64> = arr
                    .mapv(|x| x.to_f64().unwrap_or(0.0))
                    .into_raw_vec_and_offset()
                    .0;
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let median = if values.len() % 2 == 0 {
                    let mid = values.len() / 2;
                    (values[mid - 1] + values[mid]) / 2.0
                } else {
                    values[values.len() / 2]
                };

                T::from_f64(median).unwrap_or(T::zero())
            }
            AudioData::MultiChannel(arr) => {
                if arr.is_empty() {
                    return T::zero();
                }
                let mut values: Vec<f64> = arr
                    .mapv(|x| x.to_f64().unwrap_or(0.0))
                    .into_raw_vec_and_offset()
                    .0;
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let median = if values.len() % 2 == 0 {
                    let mid = values.len() / 2;
                    (values[mid - 1] + values[mid]) / 2.0
                } else {
                    values[values.len() / 2]
                };

                T::from_f64(median).unwrap_or(T::zero())
            }
        }
    }

    /// Computes the standard deviation of all samples.
    fn compute_std_dev(&self) -> T {
        let mean = self.compute_mean().to_f64().unwrap_or(0.0);

        match &self.data {
            AudioData::Mono(arr) => {
                if arr.len() <= 1 {
                    return T::zero();
                }

                let variance_sum = arr
                    .mapv(|x| {
                        let val = x.to_f64().unwrap_or(0.0);
                        let diff = val - mean;
                        diff * diff
                    })
                    .sum();

                let variance = variance_sum / arr.len() as f64;
                T::from_f64(variance.sqrt()).unwrap_or(T::zero())
            }
            AudioData::MultiChannel(arr) => {
                if arr.len() <= 1 {
                    return T::zero();
                }

                let variance_sum = arr
                    .mapv(|x| {
                        let val = x.to_f64().unwrap_or(0.0);
                        let diff = val - mean;
                        diff * diff
                    })
                    .sum();

                let variance = variance_sum / arr.len() as f64;
                T::from_f64(variance.sqrt()).unwrap_or(T::zero())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;
    use ndarray::array;

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

        let mean = audio.compute_mean();
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
        let mean = audio.compute_mean();
        let std_dev = audio.compute_std_dev();
        assert_approx_eq!(mean as f64, 0.0, 1e-6);
        assert_approx_eq!(std_dev as f64, 1.0, 1e-6);
    }
}
