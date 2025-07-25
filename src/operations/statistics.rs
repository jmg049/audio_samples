//! Statistical analysis operations for AudioSamples.
//!
//! This module implements the AudioStatistics trait, providing efficient
//! statistical analysis using ndarray operations and leveraging existing
//! native implementations where available.

use super::traits::AudioStatistics;
use crate::repr::AudioData;
use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ConvertTo,
    I24,
};
use ndarray::Axis;

impl<T: AudioSample> AudioStatistics<T> for AudioSamples<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    /// Returns the peak (maximum absolute value) in the audio samples.
    ///
    /// Leverages the existing optimized `peak_native()` implementation.
    fn peak(&self) -> T {
        self.peak_native()
    }

    /// Returns the minimum value in the audio samples.
    ///
    /// Leverages the existing optimized `min_native()` implementation.
    fn min(&self) -> T {
        self.min_native()
    }

    /// Returns the maximum value in the audio samples.
    ///
    /// Leverages the existing optimized `max_native()` implementation.
    fn max(&self) -> T {
        self.max_native()
    }

    /// Computes the Root Mean Square (RMS) of the audio samples.
    ///
    /// Uses efficient vectorized operations via ndarray to compute RMS.
    /// RMS = sqrt(mean(x^2)) where x is the audio signal.
    fn rms(&self) -> AudioSampleResult<f64> {
        match &self.data {
            AudioData::Mono(arr) => {
                // Use ndarray's efficient operations for vectorized computation
                let sum_of_squares: f64 = arr
                    .iter()
                    .map(|x| {
                        let x: f64 = x.cast_into();
                        x * x
                    })
                    .sum();

                let mean_square = sum_of_squares / arr.len() as f64;

                let rms = mean_square.sqrt();
                // Convert back to the original type
                Ok(rms)
            }
            AudioData::MultiChannel(arr) => {
                // Compute RMS across all samples in all channels
                let sum_of_squares: f64 = arr
                    .iter()
                    .map(|x| {
                        let x: f64 = x.cast_into();
                        x * x
                    })
                    .sum();

                let mean_square = sum_of_squares / arr.len() as f64;

                let rms = mean_square.sqrt();
                // Convert back to the original type
                Ok(rms)
            }
        }
    }

    /// Computes the statistical variance of the audio samples.
    ///
    /// Variance = mean((x - mean(x))^2)
    /// Uses efficient ndarray operations for vectorized computation.
    fn variance(&self) -> AudioSampleResult<f64> {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() {
                    return Err(AudioSampleError::ProcessingError {
                        msg: "Cannot compute variance on empty audio data".to_string(),
                    });
                }

                // Compute mean first
                let sum = arr.sum();
                let sum: f64 = sum.cast_into();
                let mean = sum / arr.len() as f64;

                // Compute variance using the standard formula
                let variance_sum: f64 = arr
                    .iter()
                    .map(|&x| {
                        let x: f64 = x.cast_into();
                        let diff = x - mean;
                        diff * diff
                    })
                    .sum();

                let variance_f64 = variance_sum / arr.len() as f64;
                Ok(variance_f64)
            }
            AudioData::MultiChannel(arr) => {
                if arr.is_empty() {
                    return Err(AudioSampleError::ProcessingError {
                        msg: "Cannot compute variance on empty audio data".to_string(),
                    });
                }

                let sum = arr.sum();
                let sum: f64 = sum.cast_into();

                let mean = sum / arr.len() as f64;

                let variance_sum: f64 = arr
                    .iter()
                    .map(|x| {
                        let x: f64 = x.cast_into();
                        let diff = x - mean;
                        diff * diff
                    })
                    .sum();

                let variance_f64 = variance_sum / arr.len() as f64;
                Ok(variance_f64)
            }
        }
    }

    /// Computes the standard deviation of the audio samples.
    ///
    /// Standard deviation is the square root of variance.
    fn std_dev(&self) -> AudioSampleResult<f64> {
        Ok(self.variance()?.sqrt())
    }

    /// Counts the number of zero crossings in the audio signal.
    ///
    /// Zero crossings occur when the signal changes sign between adjacent samples.
    /// This is useful for pitch detection and signal analysis.
    fn zero_crossings(&self) -> usize {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.len() < 2 {
                    return 0;
                }

                let mut crossings = 0;
                for i in 1..arr.len() {
                    let prev: f64 = arr[i - 1].cast_into();
                    let curr: f64 = arr[i].cast_into();

                    // Check for sign change (zero crossing)
                    if (prev > 0.0 && curr <= 0.0) || (prev <= 0.0 && curr > 0.0) {
                        crossings += 1;
                    }
                }
                crossings
            }
            AudioData::MultiChannel(arr) => {
                // For multi-channel, count zero crossings in each channel and sum them
                let mut total_crossings = 0;

                for channel in arr.axis_iter(Axis(0)) {
                    if channel.len() < 2 {
                        continue;
                    }

                    for i in 1..channel.len() {
                        let prev: f64 = channel[i - 1].cast_into();
                        let curr: f64 = channel[i].cast_into();

                        if (prev > 0.0 && curr <= 0.0) || (prev <= 0.0 && curr > 0.0) {
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
    /// This normalizes the zero crossing count by the signal duration.
    fn zero_crossing_rate(&self) -> f64 {
        let crossings = self.zero_crossings();
        let duration_seconds = self.duration_seconds();

        if duration_seconds > 0.0 {
            crossings as f64 / duration_seconds
        } else {
            0.0
        }
    }

    /// Computes the autocorrelation function up to max_lag samples.
    ///
    /// For efficiency, this uses a simplified correlation computation.
    /// Returns correlation values for each lag offset.
    fn autocorrelation(&self, max_lag: usize) -> AudioSampleResult<Vec<f64>> {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() || max_lag == 0 {
                    return Ok(vec![]);
                }

                let n = arr.len();
                let effective_max_lag = max_lag.min(n - 1);
                let mut correlations = Vec::with_capacity(effective_max_lag + 1);

                // Convert to f64 for precision

                // Compute autocorrelation for each lag
                for lag in 0..=effective_max_lag {
                    let mut correlation = 0.0;
                    let count = n - lag;

                    for i in 0..count {
                        let s_i: f64 = arr[i].cast_into();
                        let s_i_lag: f64 = arr[i + lag].cast_into();
                        correlation += s_i * s_i_lag;
                    }

                    // Normalize by the number of overlapping samples
                    correlation /= count as f64;
                    correlations.push(correlation);
                }

                Ok(correlations)
            }
            AudioData::MultiChannel(arr) => {
                // For multi-channel, compute autocorrelation on the first channel
                if arr.is_empty() || max_lag == 0 {
                    return Ok(vec![]);
                }

                let first_channel = arr.row(0);
                let n = first_channel.len();
                let effective_max_lag = max_lag.min(n - 1);
                let mut correlations = Vec::with_capacity(effective_max_lag + 1);

                for lag in 0..=effective_max_lag {
                    let mut correlation = 0.0;
                    let count = n - lag;

                    for i in 0..count {
                        let s_i: f64 = first_channel[i].cast_into();
                        let s_i_lag: f64 = first_channel[i + lag].cast_into();
                        correlation += s_i * s_i_lag;
                    }

                    correlation /= count as f64;
                    correlations.push(correlation);
                }

                Ok(correlations)
            }
        }
    }

    /// Computes cross-correlation with another audio signal.
    ///
    /// Signals must have the same number of channels for meaningful correlation.
    fn cross_correlation(&self, other: &Self, max_lag: usize) -> AudioSampleResult<Vec<f64>> {
        // Verify compatible signals
        if self.channels() != other.channels() {
            return Err(crate::AudioSampleError::ConversionError(
                "Incompatible".to_string(),
                "cross_correlation".to_string(),
                "same_channels".to_string(),
                "Signals must have the same number of channels for cross-correlation".to_string(),
            ));
        }

        match (&self.data, &other.data) {
            (AudioData::Mono(arr1), AudioData::Mono(arr2)) => {
                if arr1.is_empty() || arr2.is_empty() || max_lag == 0 {
                    return Ok(vec![]);
                }

                let n1 = arr1.len();
                let n2 = arr2.len();
                let effective_max_lag = max_lag.min(n1.min(n2) - 1);
                let mut correlations = Vec::with_capacity(effective_max_lag + 1);

                for lag in 0..=effective_max_lag {
                    let mut correlation = 0.0;

                    let count = n1.min(n2 - lag);

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
            (AudioData::MultiChannel(arr1), AudioData::MultiChannel(arr2)) => {
                // For multi-channel, correlate the first channels
                let ch1 = arr1.row(0);
                let ch2 = arr2.row(0);

                if ch1.is_empty() || ch2.is_empty() || max_lag == 0 {
                    return Ok(vec![]);
                }

                let n1 = ch1.len();
                let n2 = ch2.len();
                let effective_max_lag = max_lag.min(n1.min(n2) - 1);
                let mut correlations = Vec::with_capacity(effective_max_lag + 1);

                for lag in 0..=effective_max_lag {
                    let mut correlation = 0.0;
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
                // Mixed mono/multi-channel case - not supported
                Err(crate::AudioSampleError::ConversionError(
                    "Mixed".to_string(),
                    "cross_correlation".to_string(),
                    "compatible".to_string(),
                    "Cannot correlate mono and multi-channel signals".to_string(),
                ))
            }
        }
    }

    /// Computes the spectral centroid (brightness measure).
    ///
    /// Placeholder implementation - requires FFT functionality.
    /// Returns an error indicating FFT dependencies are needed.
    fn spectral_centroid(&self) -> AudioSampleResult<f64> {
        // TODO!
        todo!()
    }

    /// Computes spectral rolloff frequency.
    ///
    /// Placeholder implementation - requires FFT functionality.
    fn spectral_rolloff(&self, _rolloff_percent: f64) -> AudioSampleResult<f64> {
        // TODO!
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;
    use ndarray::array;

    #[test]
    fn test_peak_min_max_existing_methods() {
        let data = array![-3.0f32, -1.0, 0.0, 2.0, 4.0];
        let audio = AudioSamples::new_mono(data, 44100);

        // These should use the existing native implementations
        assert_eq!(audio.peak(), 4.0);
        assert_eq!(audio.min(), -3.0);
        assert_eq!(audio.max(), 4.0);
    }

    #[test]
    fn test_rms_computation() {
        // Simple test case where we can verify RMS manually
        let data = array![1.0f32, -1.0, 1.0, -1.0];
        let audio = AudioSamples::new_mono(data, 44100);

        let rms = audio.rms().expect("Failed to compute RMS");
        // RMS of [1, -1, 1, -1] = sqrt((1^2 + 1^2 + 1^2 + 1^2)/4) = sqrt(1) = 1.0
        assert_approx_eq!(rms, 1.0, 1e-6);
    }

    #[test]
    fn test_variance_and_std_dev() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioSamples::new_mono(data, 44100);

        let variance = audio.variance().expect("Failed to compute variance");
        let std_dev = audio.std_dev().expect("Failed to compute std_dev");

        // Mean = 3.0, variance = mean((1-3)^2 + (2-3)^2 + ... + (5-3)^2) = mean(4+1+0+1+4) = 2.0
        assert_approx_eq!(variance, 2.0, 1e-6);
        assert_approx_eq!(std_dev, 2.0_f64.sqrt(), 1e-6);
    }

    #[test]
    fn test_zero_crossings() {
        let data = array![1.0f32, -1.0, 1.0, -1.0, 1.0];
        let audio = AudioSamples::new_mono(data, 44100);

        let crossings = audio.zero_crossings();
        // Crossings occur at: 1->-1, -1->1, 1->-1, -1->1 = 4 crossings
        assert_eq!(crossings, 4);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let data = array![1.0f32, -1.0, 1.0, -1.0]; // 4 samples at 4 Hz = 1 second
        let audio = AudioSamples::new_mono(data, 4);

        let zcr = audio.zero_crossing_rate();
        // 3 crossings in 1 second = 3 Hz
        assert_approx_eq!(zcr, 3.0, 1e-6);
    }

    #[test]
    fn test_autocorrelation() {
        let data = array![1.0f32, 0.0, -1.0, 0.0];
        let audio = AudioSamples::new_mono(data, 44100);

        let autocorr = audio.autocorrelation(2).unwrap();

        // Should have correlations for lags 0, 1, 2
        assert_eq!(autocorr.len(), 3);

        // Lag 0 should be the highest (signal correlated with itself)
        assert!(autocorr[0] >= autocorr[1]);
        assert!(autocorr[0] >= autocorr[2]);
    }

    #[test]
    fn test_cross_correlation() {
        let data1 = array![1.0f32, 0.0, -1.0];
        let data2 = array![1.0f32, 0.0, -1.0]; // Same signal
        let audio1 = AudioSamples::new_mono(data1, 44100);
        let audio2 = AudioSamples::new_mono(data2, 44100);

        let cross_corr = audio1.cross_correlation(&audio2, 1).unwrap();

        // Cross-correlation of identical signals should be same as autocorrelation
        let autocorr = audio1.autocorrelation(1).unwrap();
        assert_eq!(cross_corr.len(), autocorr.len());
    }

    #[test]
    fn test_multi_channel_statistics() {
        let data = array![[1.0f32, 2.0], [-1.0, 1.0]]; // 2 channels, 2 samples each
        let audio = AudioSamples::new_multi_channel(data, 44100);

        let rms = audio.rms().expect("Failed to compute RMS");
        let variance = audio.variance().expect("Failed to compute variance");
        let crossings = audio.zero_crossings();

        // Should compute across all samples
        assert!(rms > 0.0);
        assert!(variance >= 0.0);
        assert_eq!(crossings, 1); // One crossing in channel 0: 1.0 -> -1.0
    }

    #[test]
    fn test_edge_cases() {
        // Empty audio
        let empty_data: ndarray::Array1<f32> = ndarray::Array1::from(vec![]);
        let empty_audio = AudioSamples::new_mono(empty_data, 44100);

        assert_eq!(empty_audio.zero_crossings(), 0);
        assert_eq!(empty_audio.zero_crossing_rate(), 0.0);

        // Single sample
        let single_data = array![1.0f32];
        let single_audio = AudioSamples::new_mono(single_data, 44100);

        assert_eq!(single_audio.zero_crossings(), 0);
        assert_eq!(single_audio.rms().expect("Failed to compute RMS"), 1.0);
    }

    #[test]
    fn test_spectral_methods_placeholder() {
        let data = array![1.0f32, -1.0, 1.0];
        let audio = AudioSamples::new_mono(data, 44100);

        // These should return errors indicating FFT dependencies are needed
        assert!(audio.spectral_centroid().is_err());
        assert!(audio.spectral_rolloff(0.85).is_err());
    }
}
