//! Audio signal generation utilities.
//!
//! This module provides functions for generating various types of audio signals
//! for testing, synthesis, and audio processing applications.

use crate::operations::traits::AudioTypeConversion;
use crate::{AudioSample, AudioSampleResult, AudioSamples, ConvertTo, I24};
use ndarray::Array1;
use std::f64::consts::PI;

/// Generates a sine wave with the specified parameters.
///
/// # Arguments
/// * `frequency` - Frequency of the sine wave in Hz
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the sine wave (0.0 to 1.0)
///
/// # Returns
/// An AudioSamples instance containing the generated sine wave
pub fn sine_wave<T: AudioSample>(
    frequency: f64,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
) -> AudioSampleResult<AudioSamples<T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    let num_samples = (duration * sample_rate as f64) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    let sample_rate_f64 = sample_rate as f64;

    for i in 0..num_samples {
        let t = i as f64 / sample_rate_f64;
        let sample = amplitude * (2.0 * PI * frequency * t).sin();
        samples.push(sample.convert_to()?);
    }
    let array = Array1::from_vec(samples);
    Ok(AudioSamples::new_mono(array, sample_rate))
}

/// Generates a cosine wave with the specified parameters.
///
/// # Arguments
/// * `frequency` - Frequency of the cosine wave in Hz
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the cosine wave (0.0 to 1.0)
///
/// # Returns
/// An AudioSamples instance containing the generated cosine wave
pub fn cosine_wave<T: AudioSample>(
    frequency: f64,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
) -> AudioSampleResult<AudioSamples<T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    let num_samples = (duration * sample_rate as f64) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    let sample_rate_f64 = sample_rate as f64;

    for i in 0..num_samples {
        let t = i as f64 / sample_rate_f64;
        let sample = amplitude * (2.0 * PI * frequency * t).cos();
        samples.push(sample.convert_to()?);
    }

    let array = Array1::from_vec(samples);
    Ok(AudioSamples::new_mono(array, sample_rate))
}

/// Generates white noise with the specified parameters.
///
/// White noise has equal energy across all frequencies within the Nyquist range.
///
/// # Arguments
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the noise (0.0 to 1.0)
///
/// # Returns
/// An AudioSamples instance containing the generated white noise
pub fn white_noise<T: AudioSample>(
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
) -> AudioSampleResult<AudioSamples<T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    let num_samples = (duration * sample_rate as f64) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        // Generate random number between -1.0 and 1.0
        let random_value = (rand::random::<f64>() - 0.5) * 2.0;
        let sample = amplitude * random_value;
        samples.push(sample.convert_to()?);
    }

    let array = Array1::from_vec(samples);
    Ok(AudioSamples::new_mono(array, sample_rate))
}

/// Generates pink noise with the specified parameters.
///
/// Pink noise has equal energy per octave, with power decreasing at -3dB per octave.
///
/// # Arguments
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the noise (0.0 to 1.0)
///
/// # Returns
/// An AudioSamples instance containing the generated pink noise
pub fn pink_noise<T: AudioSample>(
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
) -> AudioSampleResult<AudioSamples<T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    let num_samples = (duration * sample_rate as f64) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    // Pink noise generation using Paul Kellett's method
    let mut b = [0.0; 7];

    for _ in 0..num_samples {
        let white = (rand::random::<f64>() - 0.5) * 2.0;

        b[0] = 0.99886 * b[0] + white * 0.0555179;
        b[1] = 0.99332 * b[1] + white * 0.0750759;
        b[2] = 0.96900 * b[2] + white * 0.1538520;
        b[3] = 0.86650 * b[3] + white * 0.3104856;
        b[4] = 0.55000 * b[4] + white * 0.5329522;
        b[5] = -0.7616 * b[5] - white * 0.0168980;

        let pink = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + white * 0.5362;
        b[6] = white * 0.115926;

        let sample = amplitude * pink * 0.11; // Scale to reasonable amplitude
        samples.push(sample.convert_to()?);
    }

    let array = Array1::from_vec(samples);
    Ok(AudioSamples::new_mono(array, sample_rate))
}

/// Generates a square wave with the specified parameters.
///
/// # Arguments
/// * `frequency` - Frequency of the square wave in Hz
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the square wave (0.0 to 1.0)
///
/// # Returns
/// An AudioSamples instance containing the generated square wave
pub fn square_wave<T: AudioSample>(
    frequency: f64,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
) -> AudioSampleResult<AudioSamples<T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    let num_samples = (duration * sample_rate as f64) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    let sample_rate_f64 = sample_rate as f64;
    let period = sample_rate_f64 / frequency;

    for i in 0..num_samples {
        let phase = (i as f64 % period) / period;
        let sample = if phase < 0.5 { amplitude } else { -amplitude };
        samples.push(sample.convert_to()?);
    }

    let array = Array1::from_vec(samples);
    Ok(AudioSamples::new_mono(array, sample_rate))
}

/// Generates a sawtooth wave with the specified parameters.
///
/// # Arguments
/// * `frequency` - Frequency of the sawtooth wave in Hz
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the sawtooth wave (0.0 to 1.0)
///
/// # Returns
/// An AudioSamples instance containing the generated sawtooth wave
pub fn sawtooth_wave<T: AudioSample>(
    frequency: f64,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
) -> AudioSampleResult<AudioSamples<T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    let num_samples = (duration * sample_rate as f64) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    let sample_rate_f64 = sample_rate as f64;
    let period = sample_rate_f64 / frequency;

    for i in 0..num_samples {
        let phase = (i as f64 % period) / period;
        let sample = amplitude * (2.0 * phase - 1.0);
        samples.push(sample.convert_to()?);
    }

    let array = Array1::from_vec(samples);
    Ok(AudioSamples::new_mono(array, sample_rate))
}

/// Generates a triangle wave with the specified parameters.
///
/// # Arguments
/// * `frequency` - Frequency of the triangle wave in Hz
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the triangle wave (0.0 to 1.0)
///
/// # Returns
/// An AudioSamples instance containing the generated triangle wave
pub fn triangle_wave<T: AudioSample>(
    frequency: f64,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
) -> AudioSampleResult<AudioSamples<T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    let num_samples = (duration * sample_rate as f64) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    let sample_rate_f64 = sample_rate as f64;
    let period = sample_rate_f64 / frequency;

    for i in 0..num_samples {
        let phase = (i as f64 % period) / period;
        let sample = if phase < 0.5 {
            amplitude * (4.0 * phase - 1.0)
        } else {
            amplitude * (3.0 - 4.0 * phase)
        };
        samples.push(sample.convert_to()?);
    }

    let array = Array1::from_vec(samples);
    Ok(AudioSamples::new_mono(array, sample_rate))
}

/// Generates a chirp (frequency sweep) signal.
///
/// # Arguments
/// * `start_freq` - Starting frequency in Hz
/// * `end_freq` - Ending frequency in Hz
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the chirp (0.0 to 1.0)
///
/// # Returns
/// An AudioSamples instance containing the generated chirp signal
pub fn chirp<T: AudioSample>(
    start_freq: f64,
    end_freq: f64,
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
) -> AudioSampleResult<AudioSamples<T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    let num_samples = (duration * sample_rate as f64) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    let sample_rate_f64 = sample_rate as f64;
    let freq_rate = (end_freq - start_freq) / duration;

    for i in 0..num_samples {
        let t = i as f64 / sample_rate_f64;
        // let frequency = start_freq + freq_rate * t;
        let phase = 2.0 * PI * (start_freq * t + 0.5 * freq_rate * t * t);
        let sample = amplitude * phase.sin();
        samples.push(sample.convert_to()?);
    }

    let array = Array1::from_vec(samples);
    Ok(AudioSamples::new_mono(array, sample_rate))
}

/// Generates an impulse (delta function) signal.
///
/// # Arguments
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the impulse (0.0 to 1.0)
/// * `position` - Position of the impulse in seconds (0.0 to duration)
///
/// # Returns
/// An AudioSamples instance containing the generated impulse signal
pub fn impulse<T: AudioSample>(
    duration: f64,
    sample_rate: u32,
    amplitude: f64,
    position: f64,
) -> AudioSampleResult<AudioSamples<T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    let num_samples = (duration * sample_rate as f64) as usize;
    let mut samples = vec![0.0.convert_to()?; num_samples];

    let impulse_sample = (position * sample_rate as f64) as usize;

    if impulse_sample < num_samples {
        samples[impulse_sample] = amplitude.convert_to()?;
    }

    let array = Array1::from_vec(samples);
    Ok(AudioSamples::new_mono(array, sample_rate))
}

/// Generates silence (zeros) with the specified duration.
///
/// # Arguments
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// An AudioSamples instance containing silence
pub fn silence<T: AudioSample>(
    duration: f64,
    sample_rate: u32,
) -> AudioSampleResult<AudioSamples<T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    let num_samples = (duration * sample_rate as f64) as usize;
    let samples = vec![0.0.convert_to()?; num_samples];

    let array = Array1::from_vec(samples);
    Ok(AudioSamples::new_mono(array, sample_rate))
}

// Note: We need to add rand as a dependency for noise generation
// For now, let's use a simple pseudorandom number generator
mod rand {
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEED: AtomicU64 = AtomicU64::new(1);

    pub fn random<T>() -> T
    where
        T: From<f64>,
    {
        // Simple linear congruential generator
        let current = SEED.load(Ordering::Relaxed);
        let next = current.wrapping_mul(1103515245).wrapping_add(12345);
        SEED.store(next, Ordering::Relaxed);

        let normalized = (next % 1000000) as f64 / 1000000.0;
        T::from(normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_sine_wave_generation() {
        let audio = sine_wave::<f32>(440.0, 1.0, 44100, 1.0).unwrap();

        assert_eq!(audio.sample_rate(), 44100);
        assert_eq!(audio.channels(), 1);
        assert_eq!(audio.samples_per_channel(), 44100);

        // Check that the peak is approximately 1.0
        let peak = audio.peak_native();
        assert!(peak > 0.9 && peak <= 1.0);
    }

    #[test]
    fn test_white_noise_generation() {
        let audio = white_noise::<f32>(1.0, 44100, 1.0).unwrap();

        assert_eq!(audio.sample_rate(), 44100);
        assert_eq!(audio.channels(), 1);
        assert_eq!(audio.samples_per_channel(), 44100);

        // Check that we have some variation in the signal
        let mono = audio.as_mono().unwrap();
        let min_val = mono
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max_val = mono
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        assert!(max_val > min_val); // Should have variation
    }

    #[test]
    fn test_square_wave_generation() {
        let audio = square_wave::<f32>(1.0, 1.0, 10, 1.0).unwrap(); // 1 Hz, 10 samples/sec

        assert_eq!(audio.sample_rate(), 10);
        assert_eq!(audio.samples_per_channel(), 10);

        // Check that values are either 1.0 or -1.0 (approximately)
        let mono = audio.as_mono().unwrap();
        for &sample in mono.iter() {
            assert!(sample.abs() > 0.9); // Should be close to ±1.0
        }
    }

    #[test]
    fn test_impulse_generation() {
        let audio = impulse::<f32>(1.0, 10, 1.0, 0.5).unwrap(); // Impulse at 0.5 seconds

        assert_eq!(audio.sample_rate(), 10);
        assert_eq!(audio.samples_per_channel(), 10);

        let mono = audio.as_mono().unwrap();

        // Check that only one sample is non-zero
        let non_zero_count = mono.iter().filter(|&&x| x != 0.0).count();
        assert_eq!(non_zero_count, 1);

        // Check that the impulse is at the right position (sample 5)
        assert_approx_eq!(mono[5].into(), 1.0, 1e-6);
    }

    #[test]
    fn test_silence_generation() {
        let audio = silence::<f32>(1.0, 44100).unwrap();

        assert_eq!(audio.sample_rate(), 44100);
        assert_eq!(audio.samples_per_channel(), 44100);

        let mono = audio.as_mono().unwrap();

        // Check that all samples are zero
        for &sample in mono.iter() {
            assert_eq!(sample, 0.0);
        }
    }

    #[test]
    fn test_chirp_generation() {
        let audio = chirp::<f32>(100.0, 1000.0, 1.0, 44100, 1.0).unwrap();

        assert_eq!(audio.sample_rate(), 44100);
        assert_eq!(audio.samples_per_channel(), 44100);

        // Check that the peak is approximately 1.0
        let peak = audio.peak_native();
        assert!(peak > 0.9 && peak <= 1.0);
    }
}
