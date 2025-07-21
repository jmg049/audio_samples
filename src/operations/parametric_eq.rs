//! Parametric equalizer implementation.
//!
//! This module provides parametric EQ functionality with support for
//! multiple band types including peaks, shelves, and filters.

use super::iir_filtering::IirFilter;
use super::traits::AudioParametricEq;
use super::types::{EqBand, EqBandType, ParametricEq};
use crate::repr::AudioData;
use crate::{
    AudioChannelOps, AudioSample, AudioSampleError, AudioSampleResult, AudioSamples,
    AudioTypeConversion, ConvertTo, I24,
};
use std::f64::consts::PI;

impl<T: AudioSample> AudioParametricEq<T> for AudioSamples<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
    AudioSamples<T>: AudioChannelOps<T>,
{
    fn apply_parametric_eq(
        &mut self,
        eq: &ParametricEq,
        sample_rate: f64,
    ) -> AudioSampleResult<()> {
        if eq.is_bypassed() {
            return Ok(());
        }

        // Validate the EQ configuration
        eq.validate(sample_rate)
            .map_err(AudioSampleError::InvalidParameter)?;

        // Apply each enabled band in sequence
        for band in &eq.bands {
            if band.is_enabled() {
                self.apply_eq_band(band, sample_rate)?;
            }
        }

        // Apply output gain if present
        if eq.output_gain_db != 0.0 {
            let output_gain_linear = db_to_linear(eq.output_gain_db);
            self.apply_linear_gain(output_gain_linear)?;
        }

        Ok(())
    }

    fn apply_eq_band(&mut self, band: &EqBand, sample_rate: f64) -> AudioSampleResult<()> {
        if !band.is_enabled() {
            return Ok(());
        }

        // Validate band parameters
        band.validate(sample_rate)
            .map_err(AudioSampleError::InvalidParameter)?;

        // Design the filter based on band type
        let (b_coeffs, a_coeffs) = design_eq_band_filter(band, sample_rate)?;
        let mut filter = IirFilter::new(b_coeffs, a_coeffs);

        // Apply filter to audio data
        match &mut self.data {
            AudioData::Mono(_) => {
                let mut working_samples = self.as_type::<f64>()?;
                let mono_self = match self.as_mono_mut() {
                    Some(working) => working,
                    None => {
                        return Err(AudioSampleError::ArrayLayoutError {
                            message: "Mono samples must be contiguous".to_string(),
                        });
                    }
                };

                let working_samples =
                    working_samples
                        .as_mono_mut()
                        .ok_or(AudioSampleError::OptionError {
                            message: "Failed to get mono data. Underlying data is not mono."
                                .to_string(),
                        })?;

                let working_samples =
                    working_samples
                        .as_slice_mut()
                        .ok_or(AudioSampleError::ArrayLayoutError {
                            message: "Mono samples must be contiguous".to_string(),
                        })?;

                filter.process_samples_in_place(working_samples);

                for (i, output) in working_samples.iter_mut().enumerate() {
                    mono_self[i] = T::convert_from(*output)?;
                }
            }
            AudioData::MultiChannel(samples) => {
                let num_channels = samples.nrows();
                // Process each channel independently
                for channel in 0..num_channels {
                    let mut working_samples = self.as_type::<f64>()?;

                    let multi_self = match self.as_multi_channel_mut() {
                        Some(working) => working,
                        None => {
                            return Err(AudioSampleError::ArrayLayoutError {
                                message: "Multi-channel samples must be contiguous".to_string(),
                            });
                        }
                    };
                    let working_samples = working_samples.as_multi_channel_mut().ok_or(AudioSampleError::OptionError { message: "Failed to get multi-channel data. Underlying data is not multi-channel.".to_string() })?;
                    let working_samples = working_samples.as_slice_mut().ok_or(
                        AudioSampleError::ArrayLayoutError {
                            message: "Multi-channel samples must be contiguous".to_string(),
                        },
                    )?;

                    filter.process_samples_in_place(working_samples);

                    for (i, output) in working_samples.iter().enumerate() {
                        multi_self[[channel, i]] = output.convert_to()?;
                    }

                    // Reset filter state for next channel
                    filter.reset();
                }
            }
        }

        Ok(())
    }

    fn apply_peak_filter(
        &mut self,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
        sample_rate: f64,
    ) -> AudioSampleResult<()> {
        let band = EqBand::peak(frequency, gain_db, q_factor);
        self.apply_eq_band(&band, sample_rate)
    }

    fn apply_low_shelf(
        &mut self,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
        sample_rate: f64,
    ) -> AudioSampleResult<()> {
        let band = EqBand::low_shelf(frequency, gain_db, q_factor);
        self.apply_eq_band(&band, sample_rate)
    }

    fn apply_high_shelf(
        &mut self,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
        sample_rate: f64,
    ) -> AudioSampleResult<()> {
        let band = EqBand::high_shelf(frequency, gain_db, q_factor);
        self.apply_eq_band(&band, sample_rate)
    }

    fn apply_three_band_eq(
        &mut self,
        low_freq: f64,
        low_gain: f64,
        mid_freq: f64,
        mid_gain: f64,
        mid_q: f64,
        high_freq: f64,
        high_gain: f64,
        sample_rate: f64,
    ) -> AudioSampleResult<()> {
        let eq = ParametricEq::three_band(
            low_freq, low_gain, mid_freq, mid_gain, mid_q, high_freq, high_gain,
        );
        self.apply_parametric_eq(&eq, sample_rate)
    }

    fn eq_frequency_response(
        &self,
        eq: &ParametricEq,
        frequencies: &[f64],
        sample_rate: f64,
    ) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
        let mut combined_magnitude = vec![1.0; frequencies.len()];
        let mut combined_phase = vec![0.0; frequencies.len()];

        for band in &eq.bands {
            if !band.is_enabled() {
                continue;
            }

            let (b_coeffs, a_coeffs) = design_eq_band_filter(band, sample_rate)?;
            let filter = IirFilter::new(b_coeffs, a_coeffs);
            let (magnitude, phase) = filter.frequency_response(frequencies, sample_rate);

            // Combine responses (multiply magnitudes, add phases)
            for i in 0..frequencies.len() {
                combined_magnitude[i] *= magnitude[i];
                combined_phase[i] += phase[i];
            }
        }

        // Apply output gain
        if eq.output_gain_db != 0.0 {
            let output_gain_linear = db_to_linear(eq.output_gain_db);
            for magnitude in &mut combined_magnitude {
                *magnitude *= output_gain_linear;
            }
        }

        Ok((combined_magnitude, combined_phase))
    }
}

/// Design a filter for a parametric EQ band.
fn design_eq_band_filter(
    band: &EqBand,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    match band.band_type {
        EqBandType::Peak => {
            design_peak_filter(band.frequency, band.gain_db, band.q_factor, sample_rate)
        }
        EqBandType::LowShelf => {
            design_low_shelf_filter(band.frequency, band.gain_db, band.q_factor, sample_rate)
        }
        EqBandType::HighShelf => {
            design_high_shelf_filter(band.frequency, band.gain_db, band.q_factor, sample_rate)
        }
        EqBandType::LowPass => design_lowpass_filter(band.frequency, band.q_factor, sample_rate),
        EqBandType::HighPass => design_highpass_filter(band.frequency, band.q_factor, sample_rate),
        EqBandType::BandPass => design_bandpass_filter(band.frequency, band.q_factor, sample_rate),
        EqBandType::BandStop => design_bandstop_filter(band.frequency, band.q_factor, sample_rate),
    }
}

/// Design a peak/notch filter using the RBJ (Robert Bristow-Johnson) cookbook formulas.
fn design_peak_filter(
    frequency: f64,
    gain_db: f64,
    q_factor: f64,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    let a = 10.0_f64.powf(gain_db / 40.0); // sqrt of linear gain
    let omega = 2.0 * PI * frequency / sample_rate;
    let sin_omega = omega.sin();
    let cos_omega = omega.cos();
    let alpha = sin_omega / (2.0 * q_factor);

    // RBJ peak filter coefficients
    let b0 = 1.0 + alpha * a;
    let b1 = -2.0 * cos_omega;
    let b2 = 1.0 - alpha * a;
    let a0 = 1.0 + alpha / a;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha / a;

    // Normalize by a0
    let b_coeffs = vec![b0 / a0, b1 / a0, b2 / a0];
    let a_coeffs = vec![1.0, a1 / a0, a2 / a0];

    Ok((b_coeffs, a_coeffs))
}

/// Design a low shelf filter using the RBJ cookbook formulas.
fn design_low_shelf_filter(
    frequency: f64,
    gain_db: f64,
    q_factor: f64,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    let a = 10.0_f64.powf(gain_db / 40.0); // sqrt of linear gain
    let omega = 2.0 * PI * frequency / sample_rate;
    let sin_omega = omega.sin();
    let cos_omega = omega.cos();
    let alpha = sin_omega / (2.0 * q_factor);
    let sqrt_2a = (2.0 * a).sqrt();

    // RBJ low shelf filter coefficients
    let b0 = a * ((a + 1.0) - (a - 1.0) * cos_omega + sqrt_2a * alpha);
    let b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_omega);
    let b2 = a * ((a + 1.0) - (a - 1.0) * cos_omega - sqrt_2a * alpha);
    let a0 = (a + 1.0) + (a - 1.0) * cos_omega + sqrt_2a * alpha;
    let a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_omega);
    let a2 = (a + 1.0) + (a - 1.0) * cos_omega - sqrt_2a * alpha;

    // Normalize by a0
    let b_coeffs = vec![b0 / a0, b1 / a0, b2 / a0];
    let a_coeffs = vec![1.0, a1 / a0, a2 / a0];

    Ok((b_coeffs, a_coeffs))
}

/// Design a high shelf filter using the RBJ cookbook formulas.
fn design_high_shelf_filter(
    frequency: f64,
    gain_db: f64,
    q_factor: f64,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    let a = 10.0_f64.powf(gain_db / 40.0); // sqrt of linear gain
    let omega = 2.0 * PI * frequency / sample_rate;
    let sin_omega = omega.sin();
    let cos_omega = omega.cos();
    let alpha = sin_omega / (2.0 * q_factor);
    let sqrt_2a = (2.0 * a).sqrt();

    // RBJ high shelf filter coefficients
    let b0 = a * ((a + 1.0) + (a - 1.0) * cos_omega + sqrt_2a * alpha);
    let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_omega);
    let b2 = a * ((a + 1.0) + (a - 1.0) * cos_omega - sqrt_2a * alpha);
    let a0 = (a + 1.0) - (a - 1.0) * cos_omega + sqrt_2a * alpha;
    let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_omega);
    let a2 = (a + 1.0) - (a - 1.0) * cos_omega - sqrt_2a * alpha;

    // Normalize by a0
    let b_coeffs = vec![b0 / a0, b1 / a0, b2 / a0];
    let a_coeffs = vec![1.0, a1 / a0, a2 / a0];

    Ok((b_coeffs, a_coeffs))
}

/// Design a low-pass filter using the RBJ cookbook formulas.
fn design_lowpass_filter(
    frequency: f64,
    q_factor: f64,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    let omega = 2.0 * PI * frequency / sample_rate;
    let sin_omega = omega.sin();
    let cos_omega = omega.cos();
    let alpha = sin_omega / (2.0 * q_factor);

    // RBJ low-pass filter coefficients
    let b0 = (1.0 - cos_omega) / 2.0;
    let b1 = 1.0 - cos_omega;
    let b2 = (1.0 - cos_omega) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha;

    // Normalize by a0
    let b_coeffs = vec![b0 / a0, b1 / a0, b2 / a0];
    let a_coeffs = vec![1.0, a1 / a0, a2 / a0];

    Ok((b_coeffs, a_coeffs))
}

/// Design a high-pass filter using the RBJ cookbook formulas.
fn design_highpass_filter(
    frequency: f64,
    q_factor: f64,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    let omega = 2.0 * PI * frequency / sample_rate;
    let sin_omega = omega.sin();
    let cos_omega = omega.cos();
    let alpha = sin_omega / (2.0 * q_factor);

    // RBJ high-pass filter coefficients
    let b0 = (1.0 + cos_omega) / 2.0;
    let b1 = -(1.0 + cos_omega);
    let b2 = (1.0 + cos_omega) / 2.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha;

    // Normalize by a0
    let b_coeffs = vec![b0 / a0, b1 / a0, b2 / a0];
    let a_coeffs = vec![1.0, a1 / a0, a2 / a0];

    Ok((b_coeffs, a_coeffs))
}

/// Design a band-pass filter using the RBJ cookbook formulas.
fn design_bandpass_filter(
    frequency: f64,
    q_factor: f64,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    let omega = 2.0 * PI * frequency / sample_rate;
    let sin_omega = omega.sin();
    let cos_omega = omega.cos();
    let alpha = sin_omega / (2.0 * q_factor);

    // RBJ band-pass filter coefficients
    let b0 = alpha;
    let b1 = 0.0;
    let b2 = -alpha;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha;

    // Normalize by a0
    let b_coeffs = vec![b0 / a0, b1 / a0, b2 / a0];
    let a_coeffs = vec![1.0, a1 / a0, a2 / a0];

    Ok((b_coeffs, a_coeffs))
}

/// Design a band-stop (notch) filter using the RBJ cookbook formulas.
fn design_bandstop_filter(
    frequency: f64,
    q_factor: f64,
    sample_rate: f64,
) -> AudioSampleResult<(Vec<f64>, Vec<f64>)> {
    let omega = 2.0 * PI * frequency / sample_rate;
    let sin_omega = omega.sin();
    let cos_omega = omega.cos();
    let alpha = sin_omega / (2.0 * q_factor);

    // RBJ band-stop filter coefficients
    let b0 = 1.0;
    let b1 = -2.0 * cos_omega;
    let b2 = 1.0;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_omega;
    let a2 = 1.0 - alpha;

    // Normalize by a0
    let b_coeffs = vec![b0 / a0, b1 / a0, b2 / a0];
    let a_coeffs = vec![1.0, a1 / a0, a2 / a0];

    Ok((b_coeffs, a_coeffs))
}

/// Convert dB to linear gain.
pub fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Convert linear gain to dB.
pub fn linear_to_db(linear: f64) -> f64 {
    20.0 * linear.log10()
}

impl<T: AudioSample> AudioSamples<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    /// Apply a linear gain to all samples.
    fn apply_linear_gain(&mut self, gain: f64) -> AudioSampleResult<()> {
        match &mut self.data {
            AudioData::Mono(samples) => {
                for sample in samples.iter_mut() {
                    let value: f64 = sample.convert_to()?;
                    let scaled = value * gain;
                    *sample = scaled.convert_to()?;
                }
            }
            AudioData::MultiChannel(samples) => {
                for sample in samples.iter_mut() {
                    let value: f64 = sample.convert_to()?;
                    let scaled = value * gain;
                    *sample = scaled.convert_to()?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::AudioParametricEq;
    use ndarray::Array1;
    use std::f64::consts::PI;

    #[test]
    fn test_peak_filter() {
        // Create a test signal with multiple frequency components
        let sample_rate = 44100.0;
        let duration = 0.1;
        let samples_count = (sample_rate * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate;
            // Mix of frequencies: 440Hz, 880Hz, 1760Hz
            let value = (2.0 * PI * 440.0 * t).sin()
                + (2.0 * PI * 880.0 * t).sin()
                + (2.0 * PI * 1760.0 * t).sin();
            samples.push(value as f32);
        }

        let mut audio = AudioSamples::new_mono(Array1::from(samples), sample_rate as u32);

        // Apply peak filter at 880Hz with +6dB gain
        let result = audio.apply_peak_filter(880.0, 6.0, 2.0, sample_rate);
        assert!(result.is_ok());

        // The 880Hz component should be boosted
    }

    #[test]
    fn test_low_shelf_filter() {
        let sample_rate = 44100.0;
        let duration = 0.1;
        let samples_count = (sample_rate * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate;
            // Mix of low and high frequencies
            let value = (2.0 * PI * 100.0 * t).sin()
                + (2.0 * PI * 1000.0 * t).sin()
                + (2.0 * PI * 5000.0 * t).sin();
            samples.push(value as f32);
        }

        let mut audio = AudioSamples::new_mono(Array1::from(samples), sample_rate as u32);

        // Apply low shelf filter at 500Hz with -3dB gain
        let result = audio.apply_low_shelf(500.0, -3.0, 0.707, sample_rate);
        assert!(result.is_ok());

        // Low frequencies should be attenuated
    }

    #[test]
    fn test_high_shelf_filter() {
        let sample_rate = 44100.0;
        let duration = 0.1;
        let samples_count = (sample_rate * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate;
            // Mix of low and high frequencies
            let value = (2.0 * PI * 100.0 * t).sin()
                + (2.0 * PI * 1000.0 * t).sin()
                + (2.0 * PI * 5000.0 * t).sin();
            samples.push(value as f32);
        }

        let mut audio = AudioSamples::new_mono(Array1::from(samples), sample_rate as u32);

        // Apply high shelf filter at 2000Hz with +4dB gain
        let result = audio.apply_high_shelf(2000.0, 4.0, 0.707, sample_rate);
        assert!(result.is_ok());

        // High frequencies should be boosted
    }

    #[test]
    fn test_three_band_eq() {
        let sample_rate = 44100.0;
        let duration = 0.1;
        let samples_count = (sample_rate * duration) as usize;

        let mut samples = Vec::new();
        for i in 0..samples_count {
            let t = i as f64 / sample_rate;
            // Wide frequency range
            let value = (2.0 * PI * 100.0 * t).sin()
                + (2.0 * PI * 1000.0 * t).sin()
                + (2.0 * PI * 5000.0 * t).sin();
            samples.push(value as f32);
        }

        let mut audio = AudioSamples::new_mono(Array1::from(samples), sample_rate as u32);

        // Apply 3-band EQ: low shelf at 200Hz (-2dB), mid peak at 1kHz (+3dB), high shelf at 4kHz (+1dB)
        let result =
            audio.apply_three_band_eq(200.0, -2.0, 1000.0, 3.0, 2.0, 4000.0, 1.0, sample_rate);
        assert!(result.is_ok());

        // Should apply all three bands
    }

    #[test]
    fn test_parametric_eq_configuration() {
        let sample_rate = 44100.0;
        let mut audio =
            AudioSamples::new_mono(Array1::from(vec![1.0f32, 0.0, -1.0]), sample_rate as u32);

        let mut eq = ParametricEq::new();
        eq.add_band(EqBand::peak(1000.0, 3.0, 2.0));
        eq.add_band(EqBand::low_shelf(100.0, -2.0, 0.707));
        eq.set_output_gain(1.0);

        let result = audio.apply_parametric_eq(&eq, sample_rate);
        assert!(result.is_ok());

        // Check EQ configuration
        assert_eq!(eq.band_count(), 2);
        assert_eq!(eq.output_gain_db, 1.0);
        assert!(!eq.is_bypassed());
    }

    #[test]
    fn test_eq_band_validation() {
        let sample_rate = 44100.0;

        // Test valid band
        let valid_band = EqBand::peak(1000.0, 3.0, 2.0);
        assert!(valid_band.validate(sample_rate).is_ok());

        // Test invalid frequency (too high)
        let invalid_band = EqBand::peak(sample_rate, 3.0, 2.0);
        assert!(invalid_band.validate(sample_rate).is_err());

        // Test invalid Q factor
        let invalid_band = EqBand::peak(1000.0, 3.0, 0.0);
        assert!(invalid_band.validate(sample_rate).is_err());

        // Test extreme gain
        let extreme_band = EqBand::peak(1000.0, 50.0, 2.0);
        assert!(extreme_band.validate(sample_rate).is_err());
    }

    #[test]
    fn test_eq_band_enable_disable() {
        let mut band = EqBand::peak(1000.0, 3.0, 2.0);

        assert!(band.is_enabled());

        band.set_enabled(false);
        assert!(!band.is_enabled());

        band.set_enabled(true);
        assert!(band.is_enabled());
    }

    #[test]
    fn test_parametric_eq_bypass() {
        let sample_rate = 44100.0;
        let mut audio =
            AudioSamples::new_mono(Array1::from(vec![1.0f32, 0.5, -0.5]), sample_rate as u32);
        let original_samples = audio.data.clone();

        let mut eq = ParametricEq::new();
        eq.add_band(EqBand::peak(1000.0, 10.0, 2.0)); // Large gain
        eq.set_bypassed(true);

        let result = audio.apply_parametric_eq(&eq, sample_rate);
        assert!(result.is_ok());

        // Audio should be unchanged when bypassed
        match (&audio.data, &original_samples) {
            (AudioData::Mono(new), AudioData::Mono(orig)) => {
                assert_eq!(new, orig);
            }
            _ => panic!("Expected mono audio"),
        }
    }

    #[test]
    fn test_db_linear_conversion() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-10);
        assert!((db_to_linear(20.0) - 10.0).abs() < 1e-10);
        assert!((db_to_linear(-20.0) - 0.1).abs() < 1e-10);

        assert!((linear_to_db(1.0) - 0.0).abs() < 1e-10);
        assert!((linear_to_db(10.0) - 20.0).abs() < 1e-10);
        assert!((linear_to_db(0.1) - (-20.0)).abs() < 1e-10);
    }

    #[test]
    fn test_five_band_eq() {
        let eq = ParametricEq::five_band();
        assert_eq!(eq.band_count(), 5);

        // Check default frequencies
        assert_eq!(eq.get_band(0).unwrap().frequency, 100.0);
        assert_eq!(eq.get_band(1).unwrap().frequency, 300.0);
        assert_eq!(eq.get_band(2).unwrap().frequency, 1000.0);
        assert_eq!(eq.get_band(3).unwrap().frequency, 3000.0);
        assert_eq!(eq.get_band(4).unwrap().frequency, 8000.0);
    }
}
