//! Audio signal generation utilities.
//!
//! This module provides functions for generating various types of audio signals
//! for testing, synthesis, and audio processing applications.
//!
//! Some helpers are feature-gated:
//! - `MonoSampleBuilder` requires `feature = "editing"`.
//! - Noise generators require `feature = "random-generation"`.

use std::num::NonZeroU32;
use std::time::Duration;

use crate::{
    AudioSample, AudioSampleResult, AudioSamples, ConvertTo, I24, RealFloat, to_precision,
};

#[cfg(feature = "editing")]
use crate::AudioEditing;

use ndarray::Array1;
use num_traits::FloatConst;

#[cfg(feature = "random-generation")]
use rand::distr::StandardUniform;

/// Builder for creating mono audio samples through method chaining.
#[cfg(feature = "editing")]
#[derive(Debug, Clone, Default)]
pub struct MonoSampleBuilder<'a, T: AudioSample> {
    samples: Vec<AudioSamples<'a, T>>,
}

#[cfg(feature = "editing")]
impl<'a, T> MonoSampleBuilder<'a, T>
where
    T: AudioSample,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    /// Creates a new empty builder.
    pub const fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Adds an existing sample to the builder.
    pub fn add_sample(mut self, sample: AudioSamples<'_, T>) -> Self {
        self.samples.push(sample.into_owned());
        self
    }

    /// Adds a sine wave to the audio builder.
    pub fn sine_wave<F>(
        self,
        frequency: F,
        duration: Duration,
        sample_rate: u32,
        amplitude: F,
    ) -> Self
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
    {
        self.add_sample(sine_wave::<T, F>(
            frequency,
            duration,
            sample_rate,
            amplitude,
        ))
    }

    /// Adds a cosine wave to the audio builder.
    pub fn cosine_wave<F>(
        self,
        frequency: F,
        duration: Duration,
        sample_rate: u32,
        amplitude: F,
    ) -> Self
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
    {
        self.add_sample(cosine_wave::<T, F>(
            frequency,
            duration,
            sample_rate,
            amplitude,
        ))
    }

    /// Adds white noise to the audio builder.
    #[cfg(feature = "random-generation")]
    pub fn white_noise<F>(self, duration: Duration, sample_rate: u32, amplitude: F) -> Self
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
        StandardUniform: rand::distr::Distribution<F>,
    {
        self.add_sample(white_noise::<T, F>(duration, sample_rate, amplitude, None))
    }

    /// Adds pink noise to the audio builder.
    #[cfg(feature = "random-generation")]
    pub fn pink_noise<F>(self, duration: Duration, sample_rate: u32, amplitude: F) -> Self
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
        StandardUniform: rand::distr::Distribution<F>,
    {
        self.add_sample(pink_noise::<T, F>(duration, sample_rate, amplitude))
    }

    /// Adds a sawtooth wave to the audio builder.
    pub fn sawtooth_wave<F>(
        self,
        frequency: F,
        duration: Duration,
        sample_rate: u32,
        amplitude: F,
    ) -> Self
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
    {
        self.add_sample(sawtooth_wave::<T, F>(
            frequency,
            duration,
            sample_rate,
            amplitude,
        ))
    }

    /// Adds a square wave to the audio builder.
    pub fn square_wave<F>(
        self,
        frequency: F,
        duration: Duration,
        sample_rate: u32,
        amplitude: F,
    ) -> Self
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
    {
        self.add_sample(square_wave::<T, F>(
            frequency,
            duration,
            sample_rate,
            amplitude,
        ))
    }

    /// Adds a triangle wave to the audio builder.
    pub fn triangle_wave<F>(
        self,
        frequency: F,
        duration: Duration,
        sample_rate: u32,
        amplitude: F,
    ) -> Self
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
    {
        self.add_sample(triangle_wave::<T, F>(
            frequency,
            duration,
            sample_rate,
            amplitude,
        ))
    }

    /// Adds silence to the audio builder.
    pub fn silence<F>(self, duration: Duration, sample_rate: u32) -> Self
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
    {
        self.add_sample(silence::<T, F>(duration, sample_rate))
    }

    /// Finalises the builder and returns a single concatenated audio sample.
    ///
    /// # Errors
    /// Returns an error if the buffered segments are not compatible for concatenation
    /// (e.g. mismatched sample rates or channel configuration).
    pub fn build(self) -> AudioSampleResult<AudioSamples<'static, T>> {
        AudioSamples::concatenate_owned(self.samples)
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
/// An [`AudioSamples`] instance containing the generated sine wave.
///
/// # Panics
/// - If `sample_rate` is 0.
/// - If the computed number of samples cannot be represented as `usize`.
/// - If `duration` results in zero samples (empty audio is rejected at construction time).
pub fn sine_wave<T, F>(
    frequency: F,
    duration: Duration,
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
    let sample_rate_f = to_precision::<F, _>(sample_rate);
    let num_samples = (to_precision::<F, _>(duration.as_secs_f32()) * sample_rate_f).to_usize().expect("Duration and sample_rate are non-negative numbers so their product will be non-negative and therefore a valid usize");

    let mut samples = Vec::with_capacity(num_samples);
    let two = to_precision::<F, _>(2.0);
    let pi = <F as FloatConst>::PI();
    let two_pi_frrq = two * pi * frequency;

    for i in 0..num_samples {
        let t = to_precision::<F, _>(i) / sample_rate_f;
        let sample = amplitude * num_traits::Float::sin(two_pi_frrq * t);
        samples.push(sample.convert_to());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(
        array,
        NonZeroU32::new(sample_rate).expect("sample_rate must be non-zero"),
    )
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
/// An [`AudioSamples`] instance containing the generated cosine wave.
///
/// # Panics
/// - If `sample_rate` is 0.
/// - If the computed number of samples cannot be represented as `usize`.
/// - If `duration` results in zero samples (empty audio is rejected at construction time).
pub fn cosine_wave<T, F>(
    frequency: F,
    duration: Duration,
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
    let sample_rate_f = to_precision::<F, _>(sample_rate);
    let num_samples = (to_precision::<F, _>(duration.as_secs_f32()) * sample_rate_f).to_usize().expect("Duration and sample_rate are non-negative numbers so their product will be non-negative and therefore a valid usize");
    let mut samples = Vec::with_capacity(num_samples);
    let pi = <F as FloatConst>::PI();
    let two_pi_freq = to_precision::<F, _>(2.0) * pi * frequency;
    for i in 0..num_samples {
        let t = to_precision::<F, _>(i) / sample_rate_f;
        let sample = amplitude * num_traits::Float::cos(two_pi_freq * t);
        samples.push(sample.convert_to());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(
        array,
        NonZeroU32::new(sample_rate).expect("sample_rate must be non-zero"),
    )
}

/// Generates brown noise with the specified parameters.
///
/// Brown noise, also known as red noise, has a power density that decreases 6 dB per octave with increasing frequency.
/// This results in a deeper sound compared to white and pink noise.
///
/// # Arguments
/// * `duration` - Duration of the signal
/// * `sample_rate` - Sample rate in Hz
/// * `step` - Step size of the Brownian walk (larger values are “rougher”)
/// * `amplitude` - Amplitude of the noise (0.0 to 1.0)
///
/// # Returns
/// `Ok(...)` containing the generated brown noise.
///
/// # Panics
/// - If `sample_rate` is 0.
/// - If the computed number of samples cannot be represented as `usize`.
/// - If `duration` results in zero samples (empty audio is rejected at construction time).
///
/// # Examples
/// ```no_run
/// use std::time::Duration;
///
/// use audio_samples::brown_noise;
///
/// let noise = brown_noise::<f32, f64>(Duration::from_secs_f32(1.0), 44_100, 0.02, 0.5).unwrap();
/// assert_eq!(noise.samples_per_channel(), 44_100);
/// ```
#[cfg(feature = "random-generation")]
#[cfg(feature = "random-generation")]
pub fn brown_noise<T, F>(
    duration: Duration,
    sample_rate: u32,
    step: F,
    amplitude: F,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    F: RealFloat + ConvertTo<T>,
    StandardUniform: rand::distr::Distribution<F>,
{
    let num_samples = (to_precision::<F, _>(duration.as_secs_f32()) * to_precision::<F, _>(sample_rate))
        .to_usize().expect("Duration and sample_rate are non-negative numbers so their product will be non-negative and therefore a valid usize");

    let mut samples = Vec::with_capacity(num_samples);
    let mut brown_state = F::zero();

    for _ in 0..num_samples {
        let white = (rand::random::<F>() - to_precision::<F, _>(0.5)) * to_precision::<F, _>(2.0);
        brown_state += white * step;
        brown_state = brown_state.clamp(-F::one(), F::one());

        let b_state: F = to_precision::<F, _>(brown_state);
        let sample = amplitude * b_state;
        samples.push(sample.convert_to());
    }

    let arr = Array1::from_vec(samples);
    Ok(AudioSamples::new_mono(
        arr,
        NonZeroU32::new(sample_rate).expect("sample_rate must be non-zero"),
    ))
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
/// An [`AudioSamples`] instance containing the generated white noise.
///
/// # Panics
/// - If `sample_rate` is 0.
/// - If the computed number of samples cannot be represented as `usize`.
/// - If `duration` results in zero samples (empty audio is rejected at construction time).
#[cfg(feature = "random-generation")]
pub fn white_noise<T, F>(
    duration: Duration,
    sample_rate: u32,
    amplitude: F,
    seed: Option<u64>,
) -> AudioSamples<'static, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    F: RealFloat + ConvertTo<T>,
    StandardUniform: rand::distr::Distribution<F>,
{
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let num_samples = (to_precision::<F, _>(duration.as_secs_f32()) * to_precision::<F, _>(sample_rate))
        .to_usize().expect("Duration and sample_rate are non-negative numbers so their product will be non-negative and therefore a valid usize");
    let mut samples = Vec::with_capacity(num_samples);

    let two = to_precision::<F, _>(2.0);
    let half = to_precision::<F, _>(0.5);

    if let Some(seed) = seed {
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..num_samples {
            let random_value = (rng.random::<F>() - half) * two;
            let random_value: F = to_precision::<F, _>(random_value);
            let sample = amplitude * random_value;
            samples.push(sample.convert_to());
        }
    } else {
        for _ in 0..num_samples {
            let random_value = (rand::random::<F>() - half) * two;
            let random_value: F = to_precision::<F, _>(random_value);
            let sample = amplitude * random_value;
            samples.push(sample.convert_to());
        }
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(
        array,
        NonZeroU32::new(sample_rate).expect("sample_rate must be non-zero"),
    )
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
/// An [`AudioSamples`] instance containing the generated pink noise.
///
/// # Panics
/// - If `sample_rate` is 0.
/// - If the computed number of samples cannot be represented as `usize`.
/// - If `duration` results in zero samples (empty audio is rejected at construction time).
#[cfg(feature = "random-generation")]
pub fn pink_noise<T, F>(
    duration: Duration,
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
    StandardUniform: rand::distr::Distribution<F>,
{
    let num_samples = (to_precision::<F, _>(duration.as_secs_f32()) * to_precision::<F, _>(sample_rate))
        .to_usize().expect("Duration and sample_rate are non-negative numbers so their product will be non-negative and therefore a valid usize");
    let mut samples = Vec::with_capacity(num_samples);

    // Pink noise generation using Paul Kellett's method
    let mut b = [F::zero(); 7];
    let two = to_precision::<F, _>(2.0);
    let half = to_precision::<F, _>(0.5);

    let a0 = to_precision::<F, _>(0.99886);
    let b0 = to_precision::<F, _>(0.0555179);

    let a1 = to_precision::<F, _>(0.99332);
    let b1 = to_precision::<F, _>(0.0750759);
    let a2 = to_precision::<F, _>(0.96900);
    let b2 = to_precision::<F, _>(0.1538520);
    let a3 = to_precision::<F, _>(0.86650);
    let b3 = to_precision::<F, _>(0.3104856);
    let a4 = to_precision::<F, _>(0.55000);
    let b4 = to_precision::<F, _>(0.5329522);
    let a5 = to_precision::<F, _>(-0.7616);
    let b5 = to_precision::<F, _>(-0.0168980);

    let b6 = to_precision::<F, _>(0.115926);
    let pink_calc_multiplier1 = to_precision::<F, _>(0.5362);
    let pink_calc_multiplier2 = to_precision::<F, _>(0.11);

    for _ in 0..num_samples {
        let white = (rand::random::<F>() - half) * two;
        b[0] = a0 * b[0] + white * b0;
        b[1] = a1 * b[1] + white * b1;
        b[2] = a2 * b[2] + white * b2;
        b[3] = a3 * b[3] + white * b3;
        b[4] = a4 * b[4] + white * b4;
        b[5] = a5 * b[5] + white * b5;

        let pink = b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + white * pink_calc_multiplier1;
        b[6] = white * b6;
        let pink: F = to_precision::<F, _>(pink * pink_calc_multiplier2);

        let sample = amplitude * pink;
        samples.push(sample.convert_to());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(
        array,
        NonZeroU32::new(sample_rate).expect("sample_rate must be non-zero"),
    )
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
/// An [`AudioSamples`] instance containing the generated square wave.
///
/// # Panics
/// - If `sample_rate` is 0.
/// - If the computed number of samples cannot be represented as `usize`.
/// - If `duration` results in zero samples (empty audio is rejected at construction time).
pub fn square_wave<T, F>(
    frequency: F,
    duration: Duration,
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
    let num_samples = (to_precision::<F, _>(duration.as_secs_f32()) * to_precision::<F, _>(sample_rate))
        .to_usize().expect("Duration and sample_rate are non-negative numbers so their product will be non-negative and therefore a valid usize");
    let mut samples = Vec::with_capacity(num_samples);

    let two_pi = to_precision::<F, _>(2.0 * std::f64::consts::PI);
    let freq = frequency;
    let sample_rate_f = to_precision::<F, _>(sample_rate);

    for i in 0..num_samples {
        let t = to_precision::<F, _>(i) / sample_rate_f;
        let arg = two_pi * freq * t;
        let sin_val = num_traits::Float::sin(arg);
        let sample = if sin_val >= F::zero() {
            amplitude
        } else {
            -amplitude
        };

        samples.push(sample.convert_to());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(
        array,
        NonZeroU32::new(sample_rate).expect("sample_rate must be non-zero"),
    )
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
/// An [`AudioSamples`] instance containing the generated sawtooth wave.
///
/// # Panics
/// - If `sample_rate` is 0.
/// - If the computed number of samples cannot be represented as `usize`.
/// - If `duration` results in zero samples (empty audio is rejected at construction time).
pub fn sawtooth_wave<T, F>(
    frequency: F,
    duration: Duration,
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
    let sr_f = to_precision::<F, _>(sample_rate);
    let _freq_f = to_precision::<F, _>(frequency);

    let num_samples = (to_precision::<F, _>(duration.as_secs_f32())* sr_f).to_usize().expect("Duration and sample_rate are non-negative numbers so their product will be non-negative and therefore a valid usize");
    let mut samples = Vec::with_capacity(num_samples);

    let two_pi = to_precision::<F, _>(2.0 * std::f64::consts::PI);
    let freq = frequency;

    for i in 0..num_samples {
        let t = to_precision::<F, _>(i) / to_precision::<F, _>(sample_rate);
        let arg = two_pi * freq * t;
        // sawtooth: 2 * ((t / (2*pi) - 1) % 1) - 1 for width=1.0
        let phase = arg / two_pi;
        let frac = (phase - F::one()) % F::one();
        // Handle negative modulo
        let frac = if frac < F::zero() {
            frac + F::one()
        } else {
            frac
        };
        let sample = amplitude * (frac * to_precision::<F, _>(2.0) - F::one());
        samples.push(sample.convert_to());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(
        array,
        NonZeroU32::new(sample_rate).expect("sample_rate must be non-zero"),
    )
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
/// An [`AudioSamples`] instance containing the generated triangle wave.
///
/// # Panics
/// - If `sample_rate` is 0.
/// - If the computed number of samples cannot be represented as `usize`.
/// - If `duration` results in zero samples (empty audio is rejected at construction time).
pub fn triangle_wave<T, F>(
    frequency: F,
    duration: Duration,
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
    let sr_f = to_precision::<F, _>(sample_rate);
    let freq_f = to_precision::<F, _>(frequency);

    let num_samples = (to_precision::<F,_>(duration.as_secs_f32())* sr_f).to_usize().expect("Duration and sample_rate are non-negative numbers so their product will be non-negative and therefore a valid usize");
    let mut samples = Vec::with_capacity(num_samples);

    let mut phase = F::zero();
    let phase_inc = freq_f / sr_f;

    for _ in 0..num_samples {
        // triangle: ramp up then down
        let sample = if phase < to_precision::<F, _>(0.5) {
            // rising edge: -1 -> +1
            amplitude * (to_precision::<F, _>(4.0) * phase - F::one())
        } else {
            // falling edge: +1 -> -1
            amplitude * (to_precision::<F, _>(3.0) - to_precision::<F, _>(4.0) * phase)
        };

        samples.push(sample.convert_to());

        phase += phase_inc;
        if phase >= F::one() {
            phase -= F::one();
        }
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(
        array,
        NonZeroU32::new(sample_rate).expect("sample_rate must be non-zero"),
    )
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
/// An [`AudioSamples`] instance containing the generated chirp signal.
///
/// # Panics
/// - If `sample_rate` is 0.
/// - If the computed number of samples cannot be represented as `usize`.
/// - If `duration` results in zero samples (empty audio is rejected at construction time).
pub fn chirp<T, F>(
    start_freq: F,
    end_freq: F,
    duration: Duration,
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
    let num_samples = (to_precision::<F, _>(duration.as_secs_f32()) * to_precision::<F, _>(sample_rate))
        .to_usize().expect("Duration is non-negative and sample_rate is also non-negative, so multiplication should be non-negative");
    let mut samples = Vec::with_capacity(num_samples);

    let sr_f = to_precision::<F, _>(sample_rate);
    let f0 = to_precision::<F, _>(start_freq);
    let f1 = to_precision::<F, _>(end_freq);
    let duration_f = to_precision::<F, _>(duration.as_secs_f32());
    let k = (f1 - f0) / duration_f; // linear frequency slope
    let mut phase = F::zero(); // radians, unwrapped

    let two_pi = to_precision::<F, _>(2.0) * <F as FloatConst>::PI();
    for i in 0..num_samples {
        let t = to_precision::<F, _>(i) / sr_f;

        // Instantaneous frequency: f(t) = f0 + k t
        let freq = f0 + k * t;

        // Phase increment = 2π f(t) / sample_rate
        let phase_inc = two_pi * freq / sr_f;

        // Update phase
        phase += phase_inc;

        // Compute sample
        let value = amplitude * num_traits::Float::sin(phase);
        samples.push(value.convert_to());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(
        array,
        NonZeroU32::new(sample_rate).expect("sample_rate must be non-zero"),
    )
}

/// Generates an impulse (delta function) signal.
///
/// # Arguments
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the impulse (0.0 to 1.0)
/// * `position` - Position of the impulse in seconds
///
/// # Returns
/// An [`AudioSamples`] instance containing the generated impulse signal.
///
/// If `position * sample_rate` is not representable as `usize`, the impulse is placed at the
/// start of the signal. If the computed sample index is out of bounds, the output is silence.
///
/// # Panics
/// - If `sample_rate` is 0.
/// - If the computed number of samples cannot be represented as `usize`.
/// - If `duration` results in zero samples (empty audio is rejected at construction time).
pub fn impulse<T, F>(
    duration: Duration,
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
    let num_samples = (to_precision::<F, _>(duration.as_secs_f32())* to_precision::<F, _>(sample_rate))
        .to_usize()
        .expect("Duration and sample_rate are non-negative numbers so their product will be non-negative and therefore a valid usize");
    let mut samples: Vec<T> = vec![0.0.convert_to(); num_samples];

    let impulse_sample = (position * to_precision::<F, _>(sample_rate))
        .to_usize()
        .unwrap_or(0);

    if impulse_sample < num_samples {
        samples[impulse_sample] = amplitude.convert_to();
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(
        array,
        NonZeroU32::new(sample_rate).expect("sample_rate must be non-zero"),
    )
}

/// A component of a compound tone, specifying frequency and relative amplitude.
#[derive(Debug, Clone, Copy)]
pub struct ToneComponent<F> {
    /// Frequency in Hz
    pub frequency: F,
    /// Relative amplitude (typically 0.0 to 1.0)
    pub amplitude: F,
}

impl<F: RealFloat> ToneComponent<F> {
    /// Creates a new tone component.
    pub const fn new(frequency: F, amplitude: F) -> Self {
        Self {
            frequency,
            amplitude,
        }
    }
}

/// Generates a compound tone from multiple frequency components.
///
/// This is useful for creating signals with harmonics or multiple simultaneous tones.
///
/// # Arguments
/// * `components` - Slice of frequency/amplitude pairs
/// * `duration` - Duration of the signal
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// An [`AudioSamples`] instance containing the summed tones.
///
/// # Panics
/// - If `sample_rate` is 0.
/// - If the computed number of samples cannot be represented as `usize`.
/// - If `duration` results in zero samples (empty audio is rejected at construction time).
/// - If `components` is empty.
///
/// # Examples
/// ```rust,no_run
/// use audio_samples::utils::generation::{compound_tone, ToneComponent};
/// use std::time::Duration;
///
/// // Create 440 Hz with harmonics
/// let components = [
///     ToneComponent::new(440.0, 1.0),    // fundamental
///     ToneComponent::new(880.0, 0.5),    // 2nd harmonic
///     ToneComponent::new(1320.0, 0.25),  // 3rd harmonic
/// ];
/// let audio = compound_tone::<f64, f64>(&components, Duration::from_secs(1), 44100);
/// ```
pub fn compound_tone<T, F>(
    components: &[ToneComponent<F>],
    duration: Duration,
    sample_rate: u32,
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
    assert!(!components.is_empty(), "components must not be empty");

    let sr_f = to_precision::<F, _>(sample_rate);
    let num_samples = (to_precision::<F, _>(duration.as_secs_f32()) * sr_f)
        .to_usize()
        .expect("Duration and sample_rate produce valid sample count");

    let two_pi = to_precision::<F, _>(2.0) * <F as FloatConst>::PI();
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = to_precision::<F, _>(i) / sr_f;
        let mut sum = F::zero();
        for comp in components {
            sum += comp.amplitude * num_traits::Float::sin(two_pi * comp.frequency * t);
        }
        samples.push(sum.convert_to());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(
        array,
        NonZeroU32::new(sample_rate).expect("sample_rate must be non-zero"),
    )
}

/// Generates an amplitude-modulated (AM) signal.
///
/// The carrier signal is modulated by a low-frequency envelope.
///
/// # Arguments
/// * `carrier_freq` - Frequency of the carrier signal in Hz
/// * `modulator_freq` - Frequency of the modulating envelope in Hz
/// * `modulation_depth` - Depth of modulation (0.0 = no modulation, 1.0 = full modulation)
/// * `duration` - Duration of the signal
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Overall amplitude
///
/// # Returns
/// An [`AudioSamples`] instance containing the AM signal.
///
/// # Panics
/// - If `sample_rate` is 0.
/// - If the computed number of samples cannot be represented as `usize`.
/// - If `duration` results in zero samples.
///
/// # Examples
/// ```rust,no_run
/// use audio_samples::utils::generation::am_signal;
/// use std::time::Duration;
///
/// // 440 Hz carrier modulated at 2 Hz
/// let audio = am_signal::<f64, f64>(440.0, 2.0, 0.5, Duration::from_secs(1), 44100, 0.8);
/// ```
pub fn am_signal<T, F>(
    carrier_freq: F,
    modulator_freq: F,
    modulation_depth: F,
    duration: Duration,
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
    let sr_f = to_precision::<F, _>(sample_rate);
    let num_samples = (to_precision::<F, _>(duration.as_secs_f32()) * sr_f)
        .to_usize()
        .expect("Duration and sample_rate produce valid sample count");

    let two_pi = to_precision::<F, _>(2.0) * <F as FloatConst>::PI();
    let one = F::one();
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = to_precision::<F, _>(i) / sr_f;
        // Envelope: (1 - depth) + depth * (0.5 + 0.5 * sin(2π * mod_freq * t))
        // This gives envelope range from (1-depth) to 1.0
        let envelope = (one - modulation_depth)
            + modulation_depth * num_traits::Float::sin(two_pi * modulator_freq * t);
        let carrier = num_traits::Float::sin(two_pi * carrier_freq * t);
        let sample = amplitude * envelope * carrier;
        samples.push(sample.convert_to());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(
        array,
        NonZeroU32::new(sample_rate).expect("sample_rate must be non-zero"),
    )
}

/// Generates a signal with periodic exponential decay bursts.
///
/// This creates percussive-like transients useful for testing onset detection.
///
/// # Arguments
/// * `burst_rate` - Number of bursts per second
/// * `decay_rate` - Exponential decay rate (higher = faster decay)
/// * `duration` - Duration of the signal
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Peak amplitude of bursts
///
/// # Returns
/// An [`AudioSamples`] instance containing the burst signal.
///
/// # Panics
/// - If `sample_rate` is 0.
/// - If the computed number of samples cannot be represented as `usize`.
/// - If `duration` results in zero samples.
///
/// # Examples
/// ```rust,no_run
/// use audio_samples::utils::generation::exponential_bursts;
/// use std::time::Duration;
///
/// // 2 bursts per second with fast decay
/// let audio = exponential_bursts::<f64, f64>(2.0, 30.0, Duration::from_secs(3), 44100, 0.8);
/// ```
#[cfg(feature = "random-generation")]
pub fn exponential_bursts<T, F>(
    burst_rate: F,
    decay_rate: F,
    duration: Duration,
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
    StandardUniform: rand::distr::Distribution<F>,
{
    let sr_f = to_precision::<F, _>(sample_rate);
    let num_samples = (to_precision::<F, _>(duration.as_secs_f32()) * sr_f)
        .to_usize()
        .expect("Duration and sample_rate produce valid sample count");

    let two_pi = to_precision::<F, _>(2.0) * <F as FloatConst>::PI();
    let burst_period_threshold = to_precision::<F, _>(0.1); // 10% of period is active
    let noise_mix = to_precision::<F, _>(0.7);
    let tone_mix = to_precision::<F, _>(0.3);
    let tone_freq = to_precision::<F, _>(200.0);

    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = to_precision::<F, _>(i) / sr_f;
        let burst_phase = (burst_rate * t) % F::one();

        let envelope = if burst_phase < burst_period_threshold {
            num_traits::Float::exp(-burst_phase * decay_rate)
        } else {
            F::zero()
        };

        // Mix of noise and tone for percussive character
        let noise = (rand::random::<F>() - to_precision::<F, _>(0.5)) * to_precision::<F, _>(2.0);
        let tone = num_traits::Float::sin(two_pi * tone_freq * t);
        let sample = amplitude * envelope * (noise_mix * noise + tone_mix * tone);
        samples.push(sample.convert_to());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(
        array,
        NonZeroU32::new(sample_rate).expect("sample_rate must be non-zero"),
    )
}

/// Generates silence (zeros) with the specified duration.
///
/// # Arguments
/// * `duration` - Duration of the signal in seconds
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// An [`AudioSamples`] instance containing silence.
///
/// # Panics
/// - If `sample_rate` is 0.
/// - If the computed number of samples cannot be represented as `usize`.
/// - If `duration` results in zero samples (empty audio is rejected at construction time).
pub fn silence<T, F>(duration: Duration, sample_rate: u32) -> AudioSamples<'static, T>
where
    T: AudioSample,
    F: RealFloat,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    let num_samples = (to_precision::<F, _>(duration.as_secs_f32()) * to_precision::<F, _>(sample_rate))
        .to_usize().expect("Duration and sample_rate are non-negative numbers so their product will be non-negative and therefore a valid usize");
    let samples = vec![T::zero(); num_samples];

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(
        array,
        NonZeroU32::new(sample_rate).expect("sample_rate must be non-zero"),
    )
}

// ============================================================================
// Multi-channel generation helpers
// ============================================================================

/// Generates a stereo sine wave by duplicating mono to both channels.
///
/// This is a convenience function that generates a mono sine wave and
/// duplicates it to stereo (left and right channels identical).
///
/// # Arguments
/// * `frequency` - Frequency of the sine wave in Hz
/// * `duration` - Duration of the signal
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the sine wave (0.0 to 1.0)
///
/// # Returns
/// A stereo [`AudioSamples`] with the sine wave on both channels.
///
/// # Panics
/// Same as [`sine_wave`].
///
/// # Examples
/// ```rust,no_run
/// use audio_samples::utils::generation::stereo_sine_wave;
/// use std::time::Duration;
///
/// let stereo = stereo_sine_wave::<f32, f64>(440.0, Duration::from_secs(1), 44100, 0.8);
/// assert_eq!(stereo.num_channels(), 2);
/// ```
#[cfg(feature = "channels")]
pub fn stereo_sine_wave<T, F>(
    frequency: F,
    duration: Duration,
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
    use crate::operations::AudioChannelOps;
    let mono = sine_wave::<T, F>(frequency, duration, sample_rate, amplitude);
    mono.duplicate_to_channels(2)
        .expect("duplicating mono to stereo should not fail")
}

/// Generates a stereo chirp signal by duplicating mono to both channels.
///
/// # Arguments
/// * `start_freq` - Starting frequency in Hz
/// * `end_freq` - Ending frequency in Hz
/// * `duration` - Duration of the signal
/// * `sample_rate` - Sample rate in Hz
/// * `amplitude` - Amplitude of the chirp (0.0 to 1.0)
///
/// # Returns
/// A stereo [`AudioSamples`] with the chirp on both channels.
///
/// # Examples
/// ```rust,no_run
/// use audio_samples::utils::generation::stereo_chirp;
/// use std::time::Duration;
///
/// let stereo = stereo_chirp::<f32, f64>(100.0, 1000.0, Duration::from_secs(1), 44100, 0.8);
/// assert_eq!(stereo.num_channels(), 2);
/// ```
#[cfg(feature = "channels")]
pub fn stereo_chirp<T, F>(
    start_freq: F,
    end_freq: F,
    duration: Duration,
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
    use crate::operations::AudioChannelOps;
    let mono = chirp::<T, F>(start_freq, end_freq, duration, sample_rate, amplitude);
    mono.duplicate_to_channels(2)
        .expect("duplicating mono to stereo should not fail")
}

/// Generates stereo silence.
///
/// # Arguments
/// * `duration` - Duration of the signal
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// A stereo [`AudioSamples`] containing silence on both channels.
///
/// # Panics
///
/// - If `sample_rate` is 0.
/// - If the computed number of samples cannot be represented as `usize`.
/// - If `duration` results in zero samples (empty audio is rejected at construction time).
///
/// # Examples
///
/// ```rust,no_run
/// use audio_samples::utils::generation::stereo_silence;
/// use std::time::Duration;
///
/// let silent_stereo = stereo_silence::<f32, f64>(Duration::from_secs(2), 44100);
/// assert_eq!(silent_stereo.num_channels(), 2);
/// ```
#[cfg(feature = "channels")]
pub fn stereo_silence<T, F>(duration: Duration, sample_rate: u32) -> AudioSamples<'static, T>
where
    T: AudioSample,
    F: RealFloat,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    use crate::operations::AudioChannelOps;
    let mono = silence::<T, F>(duration, sample_rate);
    mono.duplicate_to_channels(2)
        .expect("duplicating mono to stereo should not fail")
}

/// Generates a multi-channel compound tone.
///
/// Creates a compound tone from multiple frequency components and duplicates
/// it to the specified number of channels.
///
/// # Arguments
/// * `components` - Slice of frequency/amplitude pairs
/// * `duration` - Duration of the signal
/// * `sample_rate` - Sample rate in Hz
/// * `n_channels` - Number of output channels
///
/// # Returns
/// A multi-channel [`AudioSamples`] with identical content on all channels.
///
/// # Panics
///
/// - If `sample_rate` is 0.
/// - If the computed number of samples cannot be represented as `usize`.
///
/// # Examples
/// ```rust,no_run
/// use audio_samples::utils::generation::{multichannel_compound_tone, ToneComponent};
/// use std::time::Duration;
///
/// let components = [
///     ToneComponent::new(440.0, 1.0),
///     ToneComponent::new(880.0, 0.5),
/// ];
/// // Create 5.1 surround sound test tone
/// let surround = multichannel_compound_tone::<f64, f64>(&components, Duration::from_secs(1), 44100, 6);
/// assert_eq!(surround.num_channels(), 6);
/// ```
#[cfg(feature = "channels")]
pub fn multichannel_compound_tone<T, F>(
    components: &[ToneComponent<F>],
    duration: Duration,
    sample_rate: u32,
    n_channels: usize,
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
    use crate::operations::AudioChannelOps;
    let mono = compound_tone::<T, F>(components, duration, sample_rate);
    mono.duplicate_to_channels(n_channels)
        .expect("duplicating mono to n_channels should not fail")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::traits::AudioStatistics;
    use crate::sample_rate;
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_sine_wave_generation() {
        let audio = sine_wave::<f32, f32>(440.0, Duration::from_secs_f32(1.0), 44100, 1.0);

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio.num_channels(), 1);
        assert_eq!(audio.samples_per_channel(), 44100);

        // Check that the peak is approximately 1.0
        let peak = audio.peak();
        assert!(peak > 0.9 && peak <= 1.0);
    }

    #[test]
    #[cfg(feature = "random-generation")]
    fn test_white_noise_generation() {
        let audio = white_noise::<f32, f64>(Duration::from_secs_f32(1.0), 44100, 1.0, None);

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
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
        let audio = square_wave::<f32, f64>(1.0, Duration::from_secs_f32(1.0), 10, 1.0); // 1 Hz, 10 samples/sec

        assert_eq!(audio.sample_rate(), sample_rate!(10));
        assert_eq!(audio.samples_per_channel(), 10);

        // Check that values are either 1.0 or -1.0 (approximately)
        let mono = audio.as_mono().unwrap();
        for &sample in mono.iter() {
            assert!(sample.abs() > 0.9); // Should be close to ±1.0
        }
    }

    #[test]
    fn test_impulse_generation() {
        let audio = impulse::<f32, f64>(Duration::from_secs_f32(1.0), 10, 1.0, 0.5); // Impulse at 0.5 seconds

        assert_eq!(audio.sample_rate(), sample_rate!(10));
        assert_eq!(audio.samples_per_channel(), 10);

        let mono = audio.as_mono().unwrap();

        // Check that only one sample is non-zero
        let non_zero_count = mono.iter().filter(|&&x| x != 0.0).count();
        assert_eq!(non_zero_count, 1);

        // Check that the impulse is at the right position (sample 5)
        assert_approx_eq!(mono[5].into(), 1.0, 1e-6);
        assert!((mono[5] as f64 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_silence_generation() {
        let audio = silence::<f32, f64>(Duration::from_secs_f32(1.0), 44100);

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio.samples_per_channel(), 44100);

        let mono = audio.as_mono().unwrap();

        // Check that all samples are zero
        for &sample in mono.iter() {
            assert_eq!(sample, 0.0);
        }
    }

    #[test]
    fn test_chirp_generation() {
        let audio = chirp::<f32, f64>(100.0, 1000.0, Duration::from_secs_f32(1.0), 44100, 1.0);

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio.samples_per_channel(), 44100);

        // Check that the peak is approximately 1.0
        let peak = audio.peak();
        assert!(peak > 0.9 && peak <= 1.0);
    }

    #[test]
    fn test_compound_tone_generation() {
        let components = [
            ToneComponent::new(440.0, 1.0),
            ToneComponent::new(880.0, 0.5),
            ToneComponent::new(1320.0, 0.25),
        ];
        let audio = compound_tone::<f32, f64>(&components, Duration::from_secs_f32(1.0), 44100);

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio.samples_per_channel(), 44100);
        assert_eq!(audio.num_channels(), 1);

        // The signal should have variation (not silent)
        let mono = audio.as_mono().unwrap();
        let min_val = mono.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = mono.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_val > min_val);
    }

    #[test]
    fn test_am_signal_generation() {
        let audio =
            am_signal::<f32, f64>(440.0, 2.0, 0.5, Duration::from_secs_f32(1.0), 44100, 0.8);

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio.samples_per_channel(), 44100);
        assert_eq!(audio.num_channels(), 1);

        // Peak should be around 0.8 (the amplitude)
        let peak = audio.peak();
        assert!(peak > 0.7 && peak <= 0.85);
    }

    #[test]
    #[cfg(feature = "random-generation")]
    fn test_exponential_bursts_generation() {
        let audio =
            exponential_bursts::<f32, f64>(2.0, 30.0, Duration::from_secs_f32(1.0), 44100, 0.8);

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio.samples_per_channel(), 44100);
        assert_eq!(audio.num_channels(), 1);

        // Should have some variation (bursts)
        let mono = audio.as_mono().unwrap();
        let min_val = mono.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = mono.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_val > min_val);
    }

    #[test]
    #[cfg(feature = "channels")]
    fn test_stereo_sine_wave_generation() {
        let audio = stereo_sine_wave::<f32, f64>(440.0, Duration::from_secs_f32(1.0), 44100, 0.8);

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio.samples_per_channel(), 44100);
        assert_eq!(audio.num_channels(), 2);

        // Both channels should be identical - check via underlying array
        let multi = audio.as_multi_channel().unwrap();
        for s_idx in 0..audio.samples_per_channel() {
            assert_eq!(multi[(0, s_idx)], multi[(1, s_idx)]);
        }
    }

    #[test]
    #[cfg(feature = "channels")]
    fn test_multichannel_compound_tone_generation() {
        let components = [
            ToneComponent::new(440.0, 1.0),
            ToneComponent::new(880.0, 0.5),
        ];
        let audio = multichannel_compound_tone::<f32, f64>(
            &components,
            Duration::from_secs_f32(0.1),
            44100,
            6, // 5.1 surround
        );

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio.num_channels(), 6);

        // All 6 channels should be identical - check via underlying array
        let multi = audio.as_multi_channel().unwrap();
        for s_idx in 0..audio.samples_per_channel() {
            let first_val = multi[(0, s_idx)];
            for ch_idx in 1..6 {
                assert_eq!(multi[(ch_idx, s_idx)], first_val);
            }
        }
    }
}
