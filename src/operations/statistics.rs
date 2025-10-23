//! Statistical analysis operations for AudioSamples.
//!
//! This module implements the AudioStatistics trait, providing efficient
//! statistical analysis using ndarray operations and leveraging existing
//! native implementations where available.

use crate::operations::fft_backends::{FftBackendImpl, UnifiedFftBackend};
use crate::operations::traits::AudioStatistics;
use crate::repr::AudioData;
use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ConvertTo,
    I24,
};
use ndarray::Axis;
use num_complex::Complex;
use rustfft::FftPlanner;

impl<T: AudioSample> AudioStatistics<T> for AudioSamples<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'b> AudioSamples<T>: AudioTypeConversion<T>,
{
    /// Returns the peak (maximum absolute value) in the audio samples.
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
            AudioData::MultiChannel(arr) => {
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
    fn min_sample(&self) -> T {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() {
                    return T::default();
                }
                // Use ndarray's efficient fold operation for vectorized minimum finding
                arr.fold(arr[0], |acc, &x| if x < acc { x } else { acc })
            }
            AudioData::MultiChannel(arr) => {
                if arr.is_empty() {
                    return T::default();
                }
                // Vectorized minimum across entire multi-channel array
                arr.fold(arr[[0, 0]], |acc, &x| if x < acc { x } else { acc })
            }
        }
    }

    /// Returns the maximum value in the audio samples.
    fn max_sample(&self) -> T {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() {
                    return T::default();
                }
                // Use ndarray's efficient fold operation for vectorized maximum finding
                arr.fold(arr[0], |acc, &x| if x > acc { x } else { acc })
            }
            AudioData::MultiChannel(arr) => {
                if arr.is_empty() {
                    return T::default();
                }
                // Vectorized maximum across entire multi-channel array
                arr.fold(arr[[0, 0]], |acc, &x| if x > acc { x } else { acc })
            }
        }
    }

    /// Computes the mean (average) of the audio samples.
    fn mean(&self) -> T {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() {
                    return T::zero();
                }

                let mean = arr
                    .mapv(|x| {
                        let x: f64 = x.cast_into();
                        x
                    })
                    .sum()
                    / arr.len() as f64;

                T::cast_from(mean)
            }
            AudioData::MultiChannel(arr) => {
                if arr.is_empty() {
                    return T::zero();
                }

                let mean = arr
                    .mapv(|x| {
                        let x: f64 = x.cast_into();
                        x
                    })
                    .sum()
                    / arr.len() as f64;

                T::cast_from(mean)
            }
        }
    }

    /// Computes the Root Mean Square (RMS) of the audio samples.
    ///
    /// Uses efficient vectorized operations via ndarray for SIMD-optimized computation.
    /// RMS = sqrt(mean(x^2)) where x is the audio signal.
    fn rms(&self) -> f64 {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() {
                    return 0.0;
                }

                // Use ndarray's vectorized mathematical operations for SIMD optimization
                // This leverages BLAS/LAPACK backends when available
                let arr_f64 = arr.mapv(|x| {
                    let x: f64 = x.cast_into();
                    x
                });

                // Vectorized square operation - enables SIMD
                let squares = &arr_f64 * &arr_f64;

                // Fast vectorized mean computation
                let mean_square = squares.mean().unwrap_or(0.0);

                mean_square.sqrt()
            }
            AudioData::MultiChannel(arr) => {
                if arr.is_empty() {
                    return 0.0;
                }

                // Convert to f64 using vectorized operations
                let arr_f64 = arr.mapv(|x| {
                    let x: f64 = x.cast_into();
                    x
                });

                // Vectorized element-wise squaring - enables SIMD across all channels
                let squares = &arr_f64 * &arr_f64;

                // Fast vectorized mean across entire array
                let mean_square = squares.mean().unwrap_or(0.0);

                mean_square.sqrt()
            }
        }
    }

    /// Computes the statistical variance of the audio samples.
    ///
    /// Variance = mean((x - mean(x))^2)
    /// Uses efficient vectorized ndarray operations for SIMD-optimized computation.
    fn variance(&self) -> AudioSampleResult<f64> {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() {
                    return Err(AudioSampleError::ProcessingError {
                        msg: "Cannot compute variance on empty audio data".to_string(),
                    });
                }

                // Convert to f64 using vectorized operations
                let arr_f64 = arr.mapv(|x| {
                    let x: f64 = x.cast_into();
                    x
                });

                // Vectorized mean computation
                let mean = arr_f64.mean().unwrap_or(0.0);

                // Vectorized variance computation: (x - mean)^2
                let deviations = &arr_f64 - mean; // Broadcasting subtraction 
                let squared_deviations = &deviations * &deviations; // Element-wise square 

                let variance = squared_deviations.mean().unwrap_or(0.0);
                Ok(variance)
            }
            AudioData::MultiChannel(arr) => {
                if arr.is_empty() {
                    return Err(AudioSampleError::ProcessingError {
                        msg: "Cannot compute variance on empty audio data".to_string(),
                    });
                }

                // Convert to f64 using vectorized operations across all channels
                let arr_f64 = arr.mapv(|x| {
                    let x: f64 = x.cast_into();
                    x
                });

                // Vectorized mean computation across entire array
                let mean = arr_f64.mean().unwrap_or(0.0);

                // Vectorized variance computation: (x - mean)^2 across all channels
                let deviations = &arr_f64 - mean; // Broadcasting subtraction
                let squared_deviations = &deviations * &deviations; // Element-wise square
                let variance = squared_deviations.mean().unwrap_or(0.0);
                Ok(variance)
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
    /// Uses FFT-based correlation for O(n log n) performance instead of O(n²).
    /// Formula: autocorr(x) = IFFT(FFT(x) * conj(FFT(x)))
    /// Returns correlation values for each lag offset.
    fn autocorrelation(&self, max_lag: usize) -> AudioSampleResult<Vec<f64>> {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() || max_lag == 0 {
                    return Ok(vec![]);
                }

                let n = arr.len();
                let effective_max_lag = max_lag.min(n - 1);

                // For FFT-based correlation, we need to pad to 2*n-1 to avoid circular correlation
                let fft_size = (2 * n - 1).next_power_of_two();

                // Convert to f64 and pad with zeros
                let mut padded_signal: Vec<Complex<f64>> = Vec::with_capacity(fft_size);
                for &sample in arr.iter() {
                    let sample: f64 = sample.cast_into();
                    padded_signal.push(Complex::new(sample, 0.0));
                }
                // Pad with zeros
                padded_signal.resize(fft_size, Complex::new(0.0, 0.0));

                // Compute FFT
                let mut planner = FftPlanner::new();
                let fft = planner.plan_fft_forward(fft_size);
                fft.process(&mut padded_signal);

                // Compute power spectrum: FFT(x) * conj(FFT(x))
                for sample in padded_signal.iter_mut() {
                    *sample = *sample * sample.conj();
                }

                // Compute inverse FFT
                let ifft = planner.plan_fft_inverse(fft_size);
                ifft.process(&mut padded_signal);

                // Extract autocorrelation values and normalize
                let mut correlations = Vec::with_capacity(effective_max_lag + 1);
                let fft_size_f64 = fft_size as f64;

                for lag in 0..=effective_max_lag {
                    // IFFT result is scaled by fft_size, and we normalize by number of overlaps
                    let correlation = padded_signal[lag].re / fft_size_f64;
                    let overlap_count = n - lag;
                    let normalized_correlation = correlation / overlap_count as f64;
                    correlations.push(normalized_correlation);
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

                let fft_size = (2 * n - 1).next_power_of_two();

                // Convert first channel to complex and pad
                let mut padded_signal: Vec<Complex<f64>> = Vec::with_capacity(fft_size);
                for &sample in first_channel.iter() {
                    let sample: f64 = sample.cast_into();
                    padded_signal.push(Complex::new(sample, 0.0));
                }
                padded_signal.resize(fft_size, Complex::new(0.0, 0.0));

                // FFT-based autocorrelation
                let mut planner = FftPlanner::new();
                let fft = planner.plan_fft_forward(fft_size);
                fft.process(&mut padded_signal);

                // Power spectrum
                for sample in padded_signal.iter_mut() {
                    *sample = *sample * sample.conj();
                }

                // Inverse FFT
                let ifft = planner.plan_fft_inverse(fft_size);
                ifft.process(&mut padded_signal);

                // Extract and normalize correlations
                let mut correlations = Vec::with_capacity(effective_max_lag + 1);
                let fft_size_f64 = fft_size as f64;

                for lag in 0..=effective_max_lag {
                    let correlation = padded_signal[lag].re / fft_size_f64;
                    let overlap_count = n - lag;
                    let normalized_correlation = correlation / overlap_count as f64;
                    correlations.push(normalized_correlation);
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
        if self.num_channels() != other.num_channels() {
            return Err(crate::AudioSampleError::InvalidInput {
                msg: "Signals must have the same number of channels for cross-correlation"
                    .to_string(),
            });
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
    /// The spectral centroid represents the "center of mass" of the spectrum
    /// and is often used as a measure of brightness or timbre.
    /// Higher values indicate brighter, more treble-heavy sounds.
    fn spectral_centroid(&self) -> AudioSampleResult<f64> {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() {
                    return Ok(0.0);
                }

                let n = arr.len();
                let duration = self.duration_seconds();

                // Convert to f64 for FFT
                let input: Vec<f64> = arr.iter().map(|&x| x.cast_into()).collect();

                // Create FFT backend
                let mut fft_backend = UnifiedFftBackend::auto_select(duration, n)?;

                // Prepare output buffer
                let output_size = n / 2 + 1;
                let mut fft_output = vec![Complex::new(0.0, 0.0); output_size];

                // Compute FFT
                fft_backend.compute_real_fft(&input, &mut fft_output)?;

                // Compute power spectrum
                let power_spectrum: Vec<f64> = fft_output.iter().map(|c| c.norm_sqr()).collect();

                // Generate frequency bins
                let sample_rate = self.sample_rate as f64;
                let nyquist = sample_rate / 2.0;
                let freq_step = nyquist / (output_size - 1) as f64;

                // Compute weighted sum and total energy
                let mut weighted_sum = 0.0;
                let mut total_energy = 0.0;

                for (i, &power) in power_spectrum.iter().enumerate() {
                    let frequency = i as f64 * freq_step;
                    weighted_sum += frequency * power;
                    total_energy += power;
                }

                // Compute centroid
                if total_energy > 0.0 {
                    Ok(weighted_sum / total_energy)
                } else {
                    Ok(0.0)
                }
            }
            AudioData::MultiChannel(arr) => {
                // For multi-channel, compute centroid on the first channel
                if arr.is_empty() {
                    return Ok(0.0);
                }

                let first_channel = arr.row(0);
                let n = first_channel.len();
                let duration = self.duration_seconds();

                // Convert to f64 for FFT
                let input: Vec<f64> = first_channel.iter().map(|&x| x.cast_into()).collect();

                // Create FFT backend
                let mut fft_backend = UnifiedFftBackend::auto_select(duration, n)?;

                // Prepare output buffer
                let output_size = n / 2 + 1;
                let mut fft_output = vec![Complex::new(0.0, 0.0); output_size];

                // Compute FFT
                fft_backend.compute_real_fft(&input, &mut fft_output)?;

                // Compute power spectrum
                let power_spectrum: Vec<f64> = fft_output.iter().map(|c| c.norm_sqr()).collect();

                // Generate frequency bins
                let sample_rate = self.sample_rate as f64;
                let nyquist = sample_rate / 2.0;
                let freq_step = nyquist / (output_size - 1) as f64;

                // Compute weighted sum and total energy
                let mut weighted_sum = 0.0;
                let mut total_energy = 0.0;

                for (i, &power) in power_spectrum.iter().enumerate() {
                    let frequency = i as f64 * freq_step;
                    weighted_sum += frequency * power;
                    total_energy += power;
                }

                // Compute centroid
                if total_energy > 0.0 {
                    Ok(weighted_sum / total_energy)
                } else {
                    Ok(0.0)
                }
            }
        }
    }

    /// Computes spectral rolloff frequency.
    ///
    /// The spectral rolloff frequency is the frequency below which a given percentage
    /// (typically 85%) of the total spectral energy is contained.
    /// This measure is useful for distinguishing between harmonic and noise-like sounds.
    fn spectral_rolloff(&self, rolloff_percent: f64) -> AudioSampleResult<f64> {
        if rolloff_percent <= 0.0 || rolloff_percent >= 1.0 {
            return Err(AudioSampleError::ProcessingError {
                msg: format!(
                    "rolloff_percent must be between 0.0 and 1.0, got {}",
                    rolloff_percent
                ),
            });
        }

        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() {
                    return Ok(0.0);
                }

                let n = arr.len();
                let duration = self.duration_seconds();

                // Convert to f64 for FFT
                let input: Vec<f64> = arr.iter().map(|&x| x.cast_into()).collect();

                // Create FFT backend
                let mut fft_backend = UnifiedFftBackend::auto_select(duration, n)?;

                // Prepare output buffer
                let output_size = n / 2 + 1;
                let mut fft_output = vec![Complex::new(0.0, 0.0); output_size];

                // Compute FFT
                fft_backend.compute_real_fft(&input, &mut fft_output)?;

                // Compute power spectrum
                let power_spectrum: Vec<f64> = fft_output.iter().map(|c| c.norm_sqr()).collect();

                // Generate frequency bins
                let sample_rate = self.sample_rate as f64;
                let nyquist = sample_rate / 2.0;
                let freq_step = nyquist / (output_size - 1) as f64;

                // Compute total energy
                let total_energy: f64 = power_spectrum.iter().sum();
                if total_energy <= 0.0 {
                    return Ok(0.0);
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
            AudioData::MultiChannel(arr) => {
                // For multi-channel, compute rolloff on the first channel
                if arr.is_empty() {
                    return Ok(0.0);
                }

                let first_channel = arr.row(0);
                let n = first_channel.len();
                let duration = self.duration_seconds();

                // Convert to f64 for FFT
                let input: Vec<f64> = first_channel.iter().map(|&x| x.cast_into()).collect();

                // Create FFT backend
                let mut fft_backend = UnifiedFftBackend::auto_select(duration, n)?;

                // Prepare output buffer
                let output_size = n / 2 + 1;
                let mut fft_output = vec![Complex::new(0.0, 0.0); output_size];

                // Compute FFT
                fft_backend.compute_real_fft(&input, &mut fft_output)?;

                // Compute power spectrum
                let power_spectrum: Vec<f64> = fft_output.iter().map(|c| c.norm_sqr()).collect();

                // Generate frequency bins
                let sample_rate = self.sample_rate as f64;
                let nyquist = sample_rate / 2.0;
                let freq_step = nyquist / (output_size - 1) as f64;

                // Compute total energy
                let total_energy: f64 = power_spectrum.iter().sum();
                if total_energy <= 0.0 {
                    return Ok(0.0);
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
    use super::*;
    use approx_eq::assert_approx_eq;
    use ndarray::array;

    #[test]
    fn test_peak_min_max_existing_methods() {
        let data = array![-3.0f32, -1.0, 0.0, 2.0, 4.0];
        let audio = AudioSamples::new_mono(data.into(), 44100);

        // These should use the existing native implementations
        assert_eq!(audio.peak(), 4.0);
        assert_eq!(audio.min_sample(), -3.0);
        assert_eq!(audio.max_sample(), 4.0);
    }

    #[test]
    fn test_rms_computation() {
        // Simple test case where we can verify RMS manually
        let data = array![1.0f32, -1.0, 1.0, -1.0];
        let audio = AudioSamples::new_mono(data.into(), 44100);

        let rms = audio.rms();
        // RMS of [1, -1, 1, -1] = sqrt((1^2 + 1^2 + 1^2 + 1^2)/4) = sqrt(1) = 1.0
        assert_approx_eq!(rms, 1.0, 1e-6);
    }

    #[test]
    fn test_variance_and_std_dev() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioSamples::new_mono(data.into(), 44100);

        let variance = audio.variance().expect("Failed to compute variance");
        let std_dev = audio.std_dev().expect("Failed to compute std_dev");

        // Mean = 3.0, variance = mean((1-3)^2 + (2-3)^2 + ... + (5-3)^2) = mean(4+1+0+1+4) = 2.0
        assert_approx_eq!(variance, 2.0, 1e-6);
        assert_approx_eq!(std_dev, 2.0_f64.sqrt(), 1e-6);
    }

    #[test]
    fn test_zero_crossings() {
        let data = array![1.0f32, -1.0, 1.0, -1.0, 1.0];
        let audio = AudioSamples::new_mono(data.into(), 44100);

        let crossings = audio.zero_crossings();
        // Crossings occur at: 1->-1, -1->1, 1->-1, -1->1 = 4 crossings
        assert_eq!(crossings, 4);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let data = array![1.0f32, -1.0, 1.0, -1.0]; // 4 samples at 4 Hz = 1 second
        let audio = AudioSamples::new_mono(data.into(), 44100);

        let zcr = audio.zero_crossing_rate();
        // 3 crossings in 1 second = 3 Hz
        assert_approx_eq!(zcr, 3.0, 1e-6);
    }

    #[test]
    fn test_autocorrelation() {
        let data = array![1.0f32, 0.0, -1.0, 0.0];
        let audio = AudioSamples::new_mono(data.into(), 44100);

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
        let audio1 = AudioSamples::new_mono(data1.into(), 44100);
        let audio2 = AudioSamples::new_mono(data2.into(), 44100);

        let cross_corr = audio1.cross_correlation(&audio2, 1).unwrap();

        // Cross-correlation of identical signals should be same as autocorrelation
        let autocorr = audio1.autocorrelation(1).unwrap();
        assert_eq!(cross_corr.len(), autocorr.len());
    }

    #[test]
    fn test_multi_channel_statistics() {
        let data = array![[1.0f32, 2.0], [-1.0, 1.0]]; // 2 channels, 2 samples each
        let audio = AudioSamples::new_multi_channel(data.into(), 44100);

        let rms = audio.rms();
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
        let empty_audio = AudioSamples::new_mono(empty_data.into(), 44100);

        assert_eq!(empty_audio.zero_crossings(), 0);
        assert_eq!(empty_audio.zero_crossing_rate(), 0.0);

        // Single sample
        let single_data = array![1.0f32];
        let single_audio = AudioSamples::new_mono(single_data.into(), 44100);

        assert_eq!(single_audio.zero_crossings(), 0);
        assert_eq!(
            single_audio.rms(),
            1.0,
            "RMS of single sample should be the sample itself"
        );
    }

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

        let audio = AudioSamples::new_mono(ndarray::Array1::from(data).into(), sample_rate);
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

        let audio = AudioSamples::new_mono(ndarray::Array1::from(data).into(), sample_rate);
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

    #[test]
    fn test_spectral_rolloff_validation() {
        let data = array![1.0f32, -1.0, 1.0];
        let audio = AudioSamples::new_mono(data.into(), 44100);

        // Test invalid rolloff percentages
        assert!(audio.spectral_rolloff(0.0).is_err());
        assert!(audio.spectral_rolloff(1.0).is_err());
        assert!(audio.spectral_rolloff(-0.1).is_err());
        assert!(audio.spectral_rolloff(1.1).is_err());

        // Test valid rolloff percentage
        assert!(audio.spectral_rolloff(0.85).is_ok());
    }

    #[test]
    fn test_spectral_methods_empty_audio() {
        let empty_data: ndarray::Array1<f32> = ndarray::Array1::from(vec![]);
        let empty_audio = AudioSamples::new_mono(empty_data.into(), 44100);

        assert_eq!(empty_audio.spectral_centroid().unwrap(), 0.0);
        assert_eq!(empty_audio.spectral_rolloff(0.85).unwrap(), 0.0);
    }
}
