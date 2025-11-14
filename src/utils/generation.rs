//! Audio signal generation utilities.
//!
//! This module provides functions for generating various types of audio signals
//! for testing, synthesis, and audio processing applications.

use crate::{
    AudioEditing, AudioSample, AudioSampleResult, AudioSamples, ConvertTo, I24, RealFloat,
    to_precision,
};
use ndarray::Array1;

/// Builder for creating mono audio samples through method chaining.
pub struct MonoSampleBuilder<T: AudioSample> {
    samples: Vec<AudioSamples<'static, T>>,
}

impl<T> MonoSampleBuilder<T>
where
    T: AudioSample,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    /// Creates a new empty builder.
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Adds an existing sample to the builder.
    pub fn add_sample(mut self, sample: AudioSamples<'static, T>) -> Self {
        self.samples.push(sample);
        self
    }

    /// Adds a sine wave to the audio builder.
    pub fn sine_wave<F>(self, frequency: F, duration: F, sample_rate: u32, amplitude: F) -> Self
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
    {
        self.add_sample(super::sine_wave::<T, F>(
            frequency,
            duration,
            sample_rate,
            amplitude,
        ))
    }

    /// Adds a cosine wave to the audio builder.
    pub fn cosine_wave<F>(self, frequency: F, duration: F, sample_rate: u32, amplitude: F) -> Self
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
    {
        self.add_sample(super::cosine_wave::<T, F>(
            frequency,
            duration,
            sample_rate,
            amplitude,
        ))
    }

    /// Adds white noise to the audio builder.
    pub fn white_noise<F>(self, duration: F, sample_rate: u32, amplitude: F) -> Self
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
    {
        self.add_sample(super::white_noise::<T, F>(duration, sample_rate, amplitude))
    }

    /// Adds pink noise to the audio builder.
    pub fn pink_noise<F>(self, duration: F, sample_rate: u32, amplitude: F) -> Self
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
    {
        self.add_sample(super::pink_noise::<T, F>(duration, sample_rate, amplitude))
    }

    /// Adds a sawtooth wave to the audio builder.
    pub fn sawtooth_wave<F>(self, frequency: F, duration: F, sample_rate: u32, amplitude: F) -> Self
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
    {
        self.add_sample(super::sawtooth_wave::<T, F>(
            frequency,
            duration,
            sample_rate,
            amplitude,
        ))
    }

    /// Adds a square wave to the audio builder.
    pub fn square_wave<F>(self, frequency: F, duration: F, sample_rate: u32, amplitude: F) -> Self
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
    {
        self.add_sample(super::square_wave::<T, F>(
            frequency,
            duration,
            sample_rate,
            amplitude,
        ))
    }

    /// Adds a triangle wave to the audio builder.
    pub fn triangle_wave<F>(self, frequency: F, duration: F, sample_rate: u32, amplitude: F) -> Self
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
    {
        self.add_sample(super::triangle_wave::<T, F>(
            frequency,
            duration,
            sample_rate,
            amplitude,
        ))
    }

    /// Adds silence to the audio builder.
    pub fn silence<F>(self, duration: F, sample_rate: u32) -> Self
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
    {
        self.add_sample(super::silence::<T, F>(duration, sample_rate))
    }

    /// Finalises the builder and returns a single concatenated audio sample.
    pub fn build(self) -> AudioSampleResult<AudioSamples<'static, T>> {
        AudioEditing::concatenate(&self.samples)
    }
}

/// Generates a sine wave with the specified parameters.
///
/// # Arguments
/// * `frequency` - Frequency of the sine wave in Hz
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the sine wave (0.0 to 1.0)
///
/// # Returns
/// An AudioSamples instance containing the generated sine wave. If the conversion fails (it shouldn't), default values are used.
pub fn sine_wave<T, F>(
    frequency: F,
    duration: F,
    sample_rate: u32,
    amplitude: F,
) -> AudioSamples<'static, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    F: RealFloat + ConvertTo<T>,
{
    let sample_rate = to_precision::<F, _>(sample_rate);
    let num_samples = (duration * sample_rate).to_usize().unwrap_or(0);
    let mut samples = Vec::with_capacity(num_samples);
    let two = to_precision::<F, _>(2.0);
    let pi = F::PI();
    let two_pi_frrq = two * pi * frequency;

    for i in 0..num_samples {
        let t = to_precision::<F, _>(i) / sample_rate;
        let sample = amplitude * (two_pi_frrq * t).sin();
        samples.push(sample.convert_to().unwrap_or_default());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array.into(), sample_rate.to_u32().unwrap_or(44100))
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
/// An AudioSamples instance containing the generated cosine wave. If the conversion fails (it shouldn't), default values are used.
pub fn cosine_wave<T, F>(
    frequency: F,
    duration: F,
    sample_rate: u32,
    amplitude: F,
) -> AudioSamples<'static, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    F: RealFloat + ConvertTo<T>,
{
    let sample_rate = to_precision::<F, _>(sample_rate);
    let num_samples = (duration * sample_rate).to_usize().unwrap_or(0);
    let mut samples = Vec::with_capacity(num_samples);
    let pi = F::PI();
    let two_pi_freq = to_precision::<F, _>(2.0) * pi * frequency;
    for i in 0..num_samples {
        let t = to_precision::<F, _>(i) / sample_rate;
        let sample = amplitude * (two_pi_freq * t).cos();
        samples.push(sample.convert_to().unwrap_or_default());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array.into(), sample_rate.to_u32().unwrap_or(44100))
}

/// Generates brown noise with the specified parameters.
///
/// Brown noise, also known as red noise, has a power density that decreases 6 dB per octave with increasing frequency.
/// This results in a deeper sound compared to white and pink noise.
///
/// # Arguments
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the noise (0.0 to 1.0)
///
/// # Returns
/// `Ok(brown_noise)` containing an Array1 instance with the generated brown noise.
///
/// # Examples
/// ```
/// use audio_samples::brown_noise;
///
/// let noise = brown_noise::<f32, f32>(1.0, 44100, 0.5).unwrap();
/// assert!(noise.len() > 0);
/// ```
pub fn brown_noise<T, F>(
    duration: F,
    sample_rate: u32,
    amplitude: F,
) -> AudioSampleResult<Array1<T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    F: RealFloat + ConvertTo<T>,
{
    let num_samples = (duration * to_precision::<F, _>(sample_rate))
        .to_usize()
        .unwrap_or(0);

    let mut samples = Vec::with_capacity(num_samples);
    let mut brown_state = 0.0;

    for _ in 0..num_samples {
        let white = (rand::random::<f64>() - 0.5) * 2.0;
        brown_state += (brown_state + white * 0.02_f64).clamp(-1.0, 1.0);
        let b_state: F = to_precision::<F, _>(brown_state);
        let sample = amplitude * b_state;
        samples.push(sample.convert_to().unwrap_or_default());
    }

    Ok(Array1::from_vec(samples))
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
/// An AudioSamples instance containing the generated white noise. If the conversion fails (it shouldn't), default values are used.
pub fn white_noise<'b, T, F>(duration: F, sample_rate: u32, amplitude: F) -> AudioSamples<'b, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    F: RealFloat + ConvertTo<T>,
{
    let num_samples = (duration * to_precision::<F, _>(sample_rate))
        .to_usize()
        .unwrap_or(0);
    let mut samples = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        // Generate random number between -1.0 and 1.0
        let random_value = (rand::random::<f64>() - 0.5) * 2.0;
        let random_value: F = to_precision::<F, _>(random_value);
        let sample = amplitude * random_value;
        samples.push(sample.convert_to().unwrap_or_default());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array.into(), sample_rate)
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
/// An AudioSamples instance containing the generated pink noise. If the conversion fails (it shouldn't), default values are used.
pub fn pink_noise<'b, T, F>(duration: F, sample_rate: u32, amplitude: F) -> AudioSamples<'b, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    F: RealFloat + ConvertTo<T>,
{
    let num_samples = (duration * to_precision::<F, _>(sample_rate))
        .to_usize()
        .unwrap_or(0);
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
        let pink: F = to_precision::<F, _>(pink * 0.11);

        let sample = amplitude * pink;
        samples.push(sample.convert_to().unwrap_or_default());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array.into(), sample_rate)
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
/// An AudioSamples instance containing the generated square wave. If the conversion fails (it shouldn't), default values are used.
pub fn square_wave<'b, T, F>(
    frequency: F,
    duration: F,
    sample_rate: u32,
    amplitude: F,
) -> AudioSamples<'static, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    F: RealFloat + ConvertTo<T>,
{
    let sample_rate = to_precision::<F, _>(sample_rate);
    let num_samples = (duration * sample_rate).to_usize().unwrap_or(0);
    let mut samples = Vec::with_capacity(num_samples);

    let period = sample_rate / frequency;

    for i in 0..num_samples {
        let phase = (to_precision::<F, _>(i) % period) / period;
        let sample = if phase < to_precision::<F, _>(0.5) {
            amplitude
        } else {
            -amplitude
        };
        samples.push(sample.convert_to().unwrap_or_default());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array.into(), sample_rate.to_u32().unwrap_or(44100))
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
/// An AudioSamples instance containing the generated sawtooth wave. If the conversion fails (it shouldn't), default values are used.
pub fn sawtooth_wave<T, F>(
    frequency: F,
    duration: F,
    sample_rate: u32,
    amplitude: F,
) -> AudioSamples<'static, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    F: RealFloat + ConvertTo<T>,
{
    let sample_rate = to_precision::<F, _>(sample_rate);
    let num_samples = (duration * sample_rate).to_usize().unwrap_or(0);
    let mut samples = Vec::with_capacity(num_samples);

    let period = sample_rate / frequency;
    let two = to_precision::<F, _>(2.0);

    for i in 0..num_samples {
        let phase = (to_precision::<F, _>(i) % period) / period;
        let sample = amplitude * (two * phase - F::one());
        samples.push(sample.convert_to().unwrap_or_default());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array.into(), sample_rate.to_u32().unwrap_or(44100))
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
/// An AudioSamples instance containing the generated triangle wave. If the conversion fails (it shouldn't), default values are used.
pub fn triangle_wave<'b, T, F>(
    frequency: F,
    duration: F,
    sample_rate: u32,
    amplitude: F,
) -> AudioSamples<'b, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    F: RealFloat + ConvertTo<T>,
{
    let sample_rate = to_precision::<F, _>(sample_rate);
    let num_samples = (duration * sample_rate).to_usize().unwrap_or(0);
    let mut samples = Vec::with_capacity(num_samples);

    let period = sample_rate / frequency;

    for i in 0..num_samples {
        let phase = (to_precision::<F, _>(i) % period) / period;
        let sample = if phase < to_precision::<F, _>(0.5) {
            amplitude * (to_precision::<F, _>(4.0) * phase - F::one())
        } else {
            amplitude * (to_precision::<F, _>(3.0) - to_precision::<F, _>(4.0) * phase)
        };
        samples.push(sample.convert_to().unwrap_or_default());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array.into(), sample_rate.to_u32().unwrap_or(44100))
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
/// An AudioSamples instance containing the generated chirp signal. If the conversion fails (it shouldn't), default values are used.
pub fn chirp<'b, T, F>(
    start_freq: F,
    end_freq: F,
    duration: F,
    sample_rate: u32,
    amplitude: F,
) -> AudioSamples<'b, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    F: RealFloat + ConvertTo<T>,
{
    let num_samples = (duration * to_precision::<F, _>(sample_rate))
        .to_usize()
        .unwrap_or(0);
    let mut samples = Vec::with_capacity(num_samples);
    let two_pi = to_precision::<F, _>(2.0) * F::PI();
    let freq_rate = (end_freq - start_freq) / duration;
    let sample_rate = to_precision::<F, _>(sample_rate);
    for i in 0..num_samples {
        let t = to_precision::<F, _>(i) / sample_rate;
        // let frequency = start_freq + freq_rate * t;
        let phase = two_pi * (start_freq * t + to_precision::<F, _>(0.5) * freq_rate * t * t);
        let sample = amplitude * phase.sin();
        samples.push(sample.convert_to().unwrap_or_default());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array.into(), sample_rate.to_u32().unwrap_or(44100))
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
/// An AudioSamples instance containing the generated impulse signal. If the conversion fails (it shouldn't), default values are used.
pub fn impulse<T, F>(
    duration: F,
    sample_rate: u32,
    amplitude: F,
    position: F,
) -> AudioSamples<'static, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    F: RealFloat + ConvertTo<T>,
{
    let num_samples = (duration * to_precision::<F, _>(sample_rate))
        .to_usize()
        .unwrap_or(0);
    let mut samples = vec![0.0.convert_to().unwrap_or_default(); num_samples];

    let impulse_sample = (position * to_precision::<F, _>(sample_rate))
        .to_usize()
        .unwrap_or(0);

    if impulse_sample < num_samples {
        samples[impulse_sample] = amplitude.convert_to().unwrap_or_default();
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array.into(), sample_rate)
}

/// Generates silence (zeros) with the specified duration.
///
/// # Arguments
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// An AudioSamples instance containing silence.
pub fn silence<T, F>(duration: F, sample_rate: u32) -> AudioSamples<'static, T>
where
    T: AudioSample,
    F: RealFloat,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let num_samples = (duration * to_precision::<F, _>(sample_rate))
        .to_usize()
        .unwrap_or(0);

    let samples = vec![T::zero(); num_samples];

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array.into(), sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::traits::AudioStatistics;
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_sine_wave_generation() {
        let audio = sine_wave::<f32, f32>(440.0, 1.0, 44100, 1.0);

        assert_eq!(audio.sample_rate(), 44100);
        assert_eq!(audio.num_channels(), 1);
        assert_eq!(audio.samples_per_channel(), 44100);

        // Check that the peak is approximately 1.0
        let peak = audio.peak();
        assert!(peak > 0.9 && peak <= 1.0);
    }

    #[test]
    fn test_white_noise_generation() {
        let audio = white_noise::<f32, f64>(1.0f64, 44100, 1.0);

        assert_eq!(audio.sample_rate(), 44100);
        assert_eq!(audio.num_channels(), 1);
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
        let audio = square_wave::<f32, f64>(1.0, 1.0, 10, 1.0); // 1 Hz, 10 samples/sec

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
        let audio = impulse::<f32, f64>(1.0, 10, 1.0, 0.5); // Impulse at 0.5 seconds

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
        let audio = silence::<f32, f64>(1.0, 44100);

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
        let audio = chirp::<f32, f64>(100.0, 1000.0, 1.0, 44100, 1.0);

        assert_eq!(audio.sample_rate(), 44100);
        assert_eq!(audio.samples_per_channel(), 44100);

        // Check that the peak is approximately 1.0
        let peak = audio.peak();
        assert!(peak > 0.9 && peak <= 1.0);
    }
}
