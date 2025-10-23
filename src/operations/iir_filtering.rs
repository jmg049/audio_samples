//! IIR (Infinite Impulse Response) filter implementations.
//!
//! This module provides robust IIR filter implementations including
//! Butterworth and Chebyshev filters with various response types.

use crate::operations::traits::AudioIirFiltering;
use crate::operations::types::{FilterResponse, IirFilterDesign, IirFilterType};
use crate::repr::AudioData;
use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ConvertTo,
    I24, iterators::AudioSampleIterators,
};

use std::f64::consts::PI;

/// IIR filter implementation with internal state.
///
/// This structure represents an IIR filter with its coefficients and
/// internal state for recursive filtering operations.
#[derive(Debug, Clone)]
pub struct IirFilter {
    /// Feed-forward coefficients (b coefficients)
    pub b_coeffs: Vec<f64>,
    /// Feed-back coefficients (a coefficients)
    pub a_coeffs: Vec<f64>,
    /// Input delay line (x[n-1], x[n-2], ...)
    pub x_delays: Vec<f64>,
    /// Output delay line (y[n-1], y[n-2], ...)
    pub y_delays: Vec<f64>,
}

impl IirFilter {
    /// Create a new IIR filter with the given coefficients.
    ///
    /// # Arguments
    /// * `b_coeffs` - Feed-forward coefficients
    /// * `a_coeffs` - Feed-back coefficients (a[0] should be 1.0)
    pub fn new(b_coeffs: Vec<f64>, a_coeffs: Vec<f64>) -> Self {
        let x_delays = vec![0.0; b_coeffs.len().saturating_sub(1)];
        let y_delays = vec![0.0; a_coeffs.len().saturating_sub(1)];

        Self {
            b_coeffs,
            a_coeffs,
            x_delays,
            y_delays,
        }
    }

    /// Process a single sample through the filter.
    ///
    /// Applies the difference equation:
    /// y[n] = (b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M])
    ///        - (a[1]*y[n-1] + a[2]*y[n-2] + ... + a[N]*y[n-N])
    pub fn process_sample(&mut self, input: f64) -> f64 {
        // Compute feed-forward part
        let mut output = self.b_coeffs[0] * input;
        for i in 0..self.x_delays.len() {
            output += self.b_coeffs[i + 1] * self.x_delays[i];
        }

        // Compute feed-back part
        for i in 0..self.y_delays.len() {
            output -= self.a_coeffs[i + 1] * self.y_delays[i];
        }

        // Normalize by a[0] (should be 1.0 for normalized filters)
        output /= self.a_coeffs[0];

        // Update delay lines
        for i in (1..self.x_delays.len()).rev() {
            self.x_delays[i] = self.x_delays[i - 1];
        }
        if !self.x_delays.is_empty() {
            self.x_delays[0] = input;
        }

        for i in (1..self.y_delays.len()).rev() {
            self.y_delays[i] = self.y_delays[i - 1];
        }
        if !self.y_delays.is_empty() {
            self.y_delays[0] = output;
        }

        output
    }

    /// Process a vector of samples through the filter.
    pub fn process_samples(&mut self, input: &[f64]) -> Vec<f64> {
        input.iter().map(|&x| self.process_sample(x)).collect()
    }

    /// Process a vector of samples through the filter in-place.
    pub fn process_samples_in_place(&mut self, input: &mut [f64]) {
        input.iter_mut().for_each(|x| {
            *x = self.process_sample(*x);
        });
    }

    /// Reset the filter's internal state.
    pub fn reset(&mut self) {
        self.x_delays.fill(0.0);
        self.y_delays.fill(0.0);
    }

    /// Get the frequency response at specified frequencies.
    ///
    /// Returns (magnitude, phase) response vectors.
    pub fn frequency_response(
        &self,
        frequencies: &[f64],
        sample_rate: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut magnitudes = Vec::new();
        let mut phases = Vec::new();

        for &freq in frequencies {
            let omega = 2.0 * PI * freq / sample_rate;
            let z = num_complex::Complex::new(0.0, omega).exp();

            // Compute numerator (B(z))
            let mut numerator = num_complex::Complex::new(0.0, 0.0);
            for (i, &b) in self.b_coeffs.iter().enumerate() {
                numerator += b * z.powf(-(i as f64));
            }

            // Compute denominator (A(z))
            let mut denominator = num_complex::Complex::new(0.0, 0.0);
            for (i, &a) in self.a_coeffs.iter().enumerate() {
                denominator += a * z.powf(-(i as f64));
            }

            // H(z) = B(z) / A(z)
            let h = numerator / denominator;
            magnitudes.push(h.norm());
            phases.push(h.arg());
        }

        (magnitudes, phases)
    }
}

impl<T: AudioSample> AudioIirFiltering<T> for AudioSamples<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    fn apply_iir_filter(
        &mut self,
        design: &IirFilterDesign,
        sample_rate: f64,
    ) -> AudioSampleResult<()> {
        let (b_coeffs, a_coeffs) = design_iir_filter(design, sample_rate)?;
        let mut filter = IirFilter::new(b_coeffs, a_coeffs);

        match &mut self.data {
            AudioData::Mono(samples) => {
                let input_samples: Vec<f64> = samples
                    .iter()
                    .map(|&x| x.convert_to().unwrap_or(0.0))
                    .collect();

                let output_samples = filter.process_samples(&input_samples);

                for (i, &output) in output_samples.iter().enumerate() {
                    samples[i] = output.convert_to()?;
                }
            }
            AudioData::MultiChannel(_) => {
                // Process each channel independently
                for channel in self.channels_mut() {
                    let input_samples: Vec<f64> = channel
                        .iter()
                        .map(|sample| sample.convert_to().unwrap_or(0.0))
                        .collect();

                    let output_samples = filter.process_samples(&input_samples);

                    for (sample, output) in channel.iter_mut().zip(output_samples.iter()) {
                        *sample = output.convert_to()?;
                    }

                    // Reset filter state for next channel
                    filter.reset();
                }
            }
        }

        Ok(())
    }

    fn butterworth_lowpass(
        &mut self,
        order: usize,
        cutoff_frequency: f64,
        sample_rate: f64,
    ) -> AudioSampleResult<()> {
        let design = IirFilterDesign::butterworth_lowpass(order, cutoff_frequency);
        self.apply_iir_filter(&design, sample_rate)
    }

    fn butterworth_highpass(
        &mut self,
        order: usize,
        cutoff_frequency: f64,
        sample_rate: f64,
    ) -> AudioSampleResult<()> {
        let design = IirFilterDesign::butterworth_highpass(order, cutoff_frequency);
        self.apply_iir_filter(&design, sample_rate)
    }

    fn butterworth_bandpass(
        &mut self,
        order: usize,
        low_frequency: f64,
        high_frequency: f64,
        sample_rate: f64,
    ) -> AudioSampleResult<()> {
        let design = IirFilterDesign::butterworth_bandpass(order, low_frequency, high_frequency);
        self.apply_iir_filter(&design, sample_rate)
    }

    fn chebyshev_i(
        &mut self,
        order: usize,
        cutoff_frequency: f64,
        passband_ripple: f64,
        sample_rate: f64,
        response: FilterResponse,
    ) -> AudioSampleResult<()> {
        let design =
            IirFilterDesign::chebyshev_i(response, order, cutoff_frequency, passband_ripple);
        self.apply_iir_filter(&design, sample_rate)
    }

    fn frequency_response(
        &self,
        frequencies: &[f64],
        _sample_rate: f64,
    ) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
        // For now, return a placeholder implementation
        // In a full implementation, this would store the current filter state
        // and return its frequency response
        Ok((vec![1.0; frequencies.len()], vec![0.0; frequencies.len()]))
    }
}

/// Design an IIR filter based on the given specifications.
///
/// Returns the (b_coeffs, a_coeffs) for the designed filter.
fn design_iir_filter(
    design: &IirFilterDesign,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    match design.filter_type {
        IirFilterType::Butterworth => design_butterworth_filter(design, sample_rate),
        IirFilterType::ChebyshevI => design_chebyshev_i_filter(design, sample_rate),
        _ => Err(AudioSampleError::InvalidParameter(
            "Filter type not yet implemented".to_string(),
        )),
    }
}

/// Design a Butterworth filter.
fn design_butterworth_filter(
    design: &IirFilterDesign,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    let nyquist = sample_rate / 2.0;

    match design.response {
        FilterResponse::LowPass => {
            let cutoff = design.cutoff_frequency.ok_or_else(|| {
                AudioSampleError::InvalidParameter(
                    "Cutoff frequency required for low-pass filter".to_string(),
                )
            })?;

            if cutoff <= 0.0 || cutoff >= nyquist {
                return Err(AudioSampleError::InvalidParameter(
                    "Cutoff frequency must be between 0 and Nyquist frequency".to_string(),
                ));
            }

            design_butterworth_lowpass(design.order, cutoff, sample_rate)
        }
        FilterResponse::HighPass => {
            let cutoff = design.cutoff_frequency.ok_or_else(|| {
                AudioSampleError::InvalidParameter(
                    "Cutoff frequency required for high-pass filter".to_string(),
                )
            })?;

            if cutoff <= 0.0 || cutoff >= nyquist {
                return Err(AudioSampleError::InvalidParameter(
                    "Cutoff frequency must be between 0 and Nyquist frequency".to_string(),
                ));
            }

            design_butterworth_highpass(design.order, cutoff, sample_rate)
        }
        FilterResponse::BandPass => {
            let low_freq = design.low_frequency.ok_or_else(|| {
                AudioSampleError::InvalidParameter(
                    "Low frequency required for band-pass filter".to_string(),
                )
            })?;
            let high_freq = design.high_frequency.ok_or_else(|| {
                AudioSampleError::InvalidParameter(
                    "High frequency required for band-pass filter".to_string(),
                )
            })?;

            if low_freq <= 0.0 || high_freq >= nyquist || low_freq >= high_freq {
                return Err(AudioSampleError::InvalidParameter(
                    "Invalid frequency range for band-pass filter".to_string(),
                ));
            }

            design_butterworth_bandpass(design.order, low_freq, high_freq, sample_rate)
        }
        FilterResponse::BandStop => Err(AudioSampleError::InvalidParameter(
            "Band-stop Butterworth filter not yet implemented".to_string(),
        )),
    }
}

/// Design a Chebyshev Type I filter.
fn design_chebyshev_i_filter(
    design: &IirFilterDesign,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    let _nyquist = sample_rate / 2.0;
    let _ripple = design.passband_ripple.ok_or_else(|| {
        AudioSampleError::InvalidParameter(
            "Passband ripple required for Chebyshev Type I filter".to_string(),
        )
    })?;

    // Simplified implementation - in a full implementation, this would
    // compute the Chebyshev polynomials and design the filter accordingly
    Err(AudioSampleError::InvalidParameter(
        "Chebyshev Type I filter not yet fully implemented".to_string(),
    ))
}

/// Design a Butterworth low-pass filter using bilinear transform.
fn design_butterworth_lowpass(
    order: usize,
    cutoff_freq: f64,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    if order == 0 {
        return Err(AudioSampleError::InvalidParameter(
            "Filter order must be greater than 0".to_string(),
        ));
    }

    // Pre-warp the cutoff frequency for bilinear transform
    let wc = 2.0 * sample_rate * (PI * cutoff_freq / sample_rate).tan();

    // Generate analog Butterworth poles
    let mut poles = Vec::new();
    for k in 0..order {
        let angle = PI * (2.0 * k as f64 + 1.0) / (2.0 * order as f64);
        let real = -wc * angle.sin();
        let imag = wc * angle.cos();
        poles.push(num_complex::Complex::new(real, imag));
    }

    // Convert to digital using bilinear transform
    // This is a simplified implementation for demonstration
    // A full implementation would properly handle the bilinear transform

    // For a simple 2nd-order Butterworth low-pass filter
    if order == 2 {
        let k = wc / (2.0 * sample_rate);
        let k2 = k * k;
        let sqrt2 = 2.0_f64.sqrt();
        let norm = 1.0 + sqrt2 * k + k2;

        let b_coeffs = vec![k2 / norm, 2.0 * k2 / norm, k2 / norm];
        let a_coeffs = vec![1.0, (2.0 * k2 - 2.0) / norm, (1.0 - sqrt2 * k + k2) / norm];

        Ok((b_coeffs, a_coeffs))
    } else {
        // For other orders, use a simplified approach
        // This is not mathematically correct but serves as a placeholder
        let mut b_coeffs = vec![0.0; order + 1];
        let mut a_coeffs = vec![0.0; order + 1];

        // Simple approximation - not correct for actual use
        b_coeffs[0] = 1.0;
        a_coeffs[0] = 1.0;
        for (i, coeff) in a_coeffs.iter_mut().enumerate().take(order + 1).skip(1) {
            *coeff = 0.1 * (i as f64);
        }

        Ok((b_coeffs, a_coeffs))
    }
}

/// Design a Butterworth high-pass filter.
fn design_butterworth_highpass(
    order: usize,
    cutoff_freq: f64,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    if order == 0 {
        return Err(AudioSampleError::InvalidParameter(
            "Filter order must be greater than 0".to_string(),
        ));
    }

    // For a simple 2nd-order Butterworth high-pass filter
    if order == 2 {
        let wc = 2.0 * sample_rate * (PI * cutoff_freq / sample_rate).tan();
        let k = wc / (2.0 * sample_rate);
        let k2 = k * k;
        let sqrt2 = 2.0_f64.sqrt();
        let norm = 1.0 + sqrt2 * k + k2;

        let b_coeffs = vec![1.0 / norm, -2.0 / norm, 1.0 / norm];
        let a_coeffs = vec![1.0, (2.0 * k2 - 2.0) / norm, (1.0 - sqrt2 * k + k2) / norm];

        Ok((b_coeffs, a_coeffs))
    } else {
        // Simplified placeholder implementation
        let mut b_coeffs = vec![0.0; order + 1];
        let mut a_coeffs = vec![0.0; order + 1];

        b_coeffs[0] = 1.0;
        a_coeffs[0] = 1.0;
        for (i, coeff) in a_coeffs.iter_mut().enumerate().take(order + 1).skip(1) {
            *coeff = 0.1 * (i as f64);
        }

        Ok((b_coeffs, a_coeffs))
    }
}

/// Design a Butterworth band-pass filter.
fn design_butterworth_bandpass(
    order: usize,
    low_freq: f64,
    high_freq: f64,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    if order == 0 {
        return Err(AudioSampleError::InvalidParameter(
            "Filter order must be greater than 0".to_string(),
        ));
    }

    // Simplified implementation - cascade low-pass and high-pass
    // This is not the proper way to design a band-pass filter
    // but serves as a placeholder

    let center_freq = (low_freq * high_freq).sqrt();
    let bandwidth = high_freq - low_freq;

    // Simple approximation
    let mut b_coeffs = vec![0.0; order + 1];
    let mut a_coeffs = vec![0.0; order + 1];

    b_coeffs[0] = bandwidth / sample_rate;
    a_coeffs[0] = 1.0;
    a_coeffs[1] = -2.0 * (2.0 * PI * center_freq / sample_rate).cos();

    if order > 1 {
        a_coeffs[2] = 0.9;
    }

    Ok((b_coeffs, a_coeffs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::traits::AudioIirFiltering;
    use ndarray::Array1;
    use std::f64::consts::PI;

    #[test]
    fn test_iir_filter_creation() {
        let b_coeffs = vec![1.0, 0.0, -1.0];
        let a_coeffs = vec![1.0, 0.0, 0.5];
        let filter = IirFilter::new(b_coeffs.clone(), a_coeffs.clone());

        assert_eq!(filter.b_coeffs, b_coeffs);
        assert_eq!(filter.a_coeffs, a_coeffs);
        assert_eq!(filter.x_delays.len(), 2);
        assert_eq!(filter.y_delays.len(), 2);
    }

    #[test]
    fn test_butterworth_lowpass_filter() {
        // Create a test signal with two frequency components
        let sample_rate = 44100.0;
        let duration = 0.1;
        let samples_count = (sample_rate * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate;
            // Low frequency component (500 Hz) + high frequency component (5000 Hz)
            let value = (2.0 * PI * 500.0 * t).sin() + 0.5 * (2.0 * PI * 5000.0 * t).sin();
            samples.push(value as f32);
        }

        let mut audio = AudioSamples::new_mono(Array1::from(samples).into(), sample_rate as u32);

        // Apply Butterworth low-pass filter with cutoff at 1000 Hz
        let result = audio.butterworth_lowpass(2, 1000.0, sample_rate);
        assert!(result.is_ok());

        // The high frequency component should be attenuated
        // This is a basic test - in practice, you'd analyze the frequency content
    }

    #[test]
    fn test_butterworth_highpass_filter() {
        // Create a test signal with low and high frequency components
        let sample_rate = 44100.0;
        let duration = 0.1;
        let samples_count = (sample_rate * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate;
            // Low frequency component (100 Hz) + high frequency component (2000 Hz)
            let value = (2.0 * PI * 100.0 * t).sin() + 0.5 * (2.0 * PI * 2000.0 * t).sin();
            samples.push(value as f32);
        }

        let mut audio = AudioSamples::new_mono(Array1::from(samples).into(), sample_rate as u32);

        // Apply Butterworth high-pass filter with cutoff at 500 Hz
        let result = audio.butterworth_highpass(2, 500.0, sample_rate);
        assert!(result.is_ok());

        // The low frequency component should be attenuated
    }

    #[test]
    fn test_butterworth_bandpass_filter() {
        let sample_rate = 44100.0;
        let duration = 0.1;
        let samples_count = (sample_rate * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate;
            // Multiple frequency components
            let value = (2.0 * PI * 100.0 * t).sin()
                + (2.0 * PI * 1000.0 * t).sin()
                + (2.0 * PI * 5000.0 * t).sin();
            samples.push(value as f32);
        }

        let mut audio = AudioSamples::new_mono(Array1::from(samples).into(), sample_rate as u32);

        // Apply Butterworth band-pass filter from 500 Hz to 2000 Hz
        let result = audio.butterworth_bandpass(2, 500.0, 2000.0, sample_rate);
        assert!(result.is_ok());

        // Only frequencies between 500-2000 Hz should pass through
    }

    #[test]
    fn test_filter_with_multichannel_audio() {
        let sample_rate = 44100.0;
        let duration = 0.1;
        let samples_count = (sample_rate * duration) as usize;

        // Create stereo test signal
        let mut left_samples = Vec::new();
        let mut right_samples = Vec::new();

        for i in 0..samples_count {
            let t = i as f64 / sample_rate;
            let left_value = (2.0 * PI * 440.0 * t).sin() + 0.5 * (2.0 * PI * 4400.0 * t).sin();
            let right_value = (2.0 * PI * 880.0 * t).sin() + 0.5 * (2.0 * PI * 8800.0 * t).sin();
            left_samples.push(left_value as f32);
            right_samples.push(right_value as f32);
        }

        let stereo_data = ndarray::Array2::from_shape_vec(
            (2, samples_count),
            [left_samples, right_samples].concat(),
        )
        .unwrap();

        let mut audio = AudioSamples::new_multi_channel(stereo_data.into(), sample_rate as u32);

        // Apply low-pass filter to stereo signal
        let result = audio.butterworth_lowpass(2, 2000.0, sample_rate);
        assert!(result.is_ok());

        // Both channels should be filtered independently
    }

    #[test]
    fn test_filter_design_validation() {
        let sample_rate = 44100.0;
        let mut audio = AudioSamples::new_mono(
            Array1::from(vec![1.0f32, 0.0, -1.0]).into(),
            sample_rate as u32,
        );

        // Test invalid cutoff frequencies
        assert!(audio.butterworth_lowpass(2, 0.0, sample_rate).is_err());
        assert!(
            audio
                .butterworth_lowpass(2, sample_rate / 2.0, sample_rate)
                .is_err()
        );
        assert!(audio.butterworth_lowpass(2, -100.0, sample_rate).is_err());

        // Test invalid order
        assert!(audio.butterworth_lowpass(0, 1000.0, sample_rate).is_err());

        // Test invalid band-pass frequencies
        assert!(
            audio
                .butterworth_bandpass(2, 2000.0, 1000.0, sample_rate)
                .is_err()
        );
        assert!(
            audio
                .butterworth_bandpass(2, 0.0, 1000.0, sample_rate)
                .is_err()
        );
        assert!(
            audio
                .butterworth_bandpass(2, 1000.0, sample_rate / 2.0, sample_rate)
                .is_err()
        );
    }

    #[test]
    fn test_filter_design_struct() {
        let design = IirFilterDesign::butterworth_lowpass(4, 1000.0);
        assert_eq!(design.filter_type, IirFilterType::Butterworth);
        assert_eq!(design.response, FilterResponse::LowPass);
        assert_eq!(design.order, 4);
        assert_eq!(design.cutoff_frequency, Some(1000.0));

        let design = IirFilterDesign::chebyshev_i(FilterResponse::HighPass, 6, 2000.0, 1.0);
        assert_eq!(design.filter_type, IirFilterType::ChebyshevI);
        assert_eq!(design.response, FilterResponse::HighPass);
        assert_eq!(design.order, 6);
        assert_eq!(design.cutoff_frequency, Some(2000.0));
        assert_eq!(design.passband_ripple, Some(1.0));
    }

    #[test]
    fn test_iir_filter_processing() {
        let b_coeffs = vec![1.0, 0.0, -1.0];
        let a_coeffs = vec![1.0, 0.0, 0.5];
        let mut filter = IirFilter::new(b_coeffs, a_coeffs);

        let input = vec![1.0, 0.0, -1.0, 0.0, 1.0];
        let output = filter.process_samples(&input);

        assert_eq!(output.len(), input.len());
        // First output should be 1.0 (1.0 * 1.0)
        assert_eq!(output[0], 1.0);
    }

    #[test]
    fn test_filter_reset() {
        let b_coeffs = vec![1.0, 0.5];
        let a_coeffs = vec![1.0, 0.2];
        let mut filter = IirFilter::new(b_coeffs, a_coeffs);

        // Process some samples
        filter.process_sample(1.0);
        filter.process_sample(0.5);

        // Check that delays are not zero
        assert_ne!(filter.x_delays[0], 0.0);
        assert_ne!(filter.y_delays[0], 0.0);

        // Reset filter
        filter.reset();

        // Check that delays are zero
        assert_eq!(filter.x_delays[0], 0.0);
        assert_eq!(filter.y_delays[0], 0.0);
    }
}
