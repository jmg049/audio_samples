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

#[cfg(feature = "editing")]
use crate::AudioSampleResult;

use crate::{AudioSamples, ConvertTo, StandardSample};
#[cfg(feature = "editing")]
use non_empty_slice::NonEmptyVec;

#[cfg(feature = "editing")]
use crate::{AudioEditing, repr::SampleRate};

use ndarray::Array1;
use non_empty_slice::NonEmptySlice;
use num_traits::FloatConst;

/// Builder for creating mono audio samples through method chaining.
#[cfg(feature = "editing")]
#[derive(Debug, Clone)]
pub struct MonoSampleBuilder<'a, T>
where
    T: StandardSample,
{
    samples: Vec<AudioSamples<'a, T>>,
    sample_rate: SampleRate,
}

#[cfg(feature = "editing")]
impl<T> MonoSampleBuilder<'_, T>
where
    T: StandardSample,
{
    /// Creates a new empty builder.
    #[inline]
    #[must_use]
    pub const fn new(sample_rate: SampleRate) -> Self {
        Self {
            samples: Vec::new(),
            sample_rate,
        }
    }

    /// Adds an existing sample to the builder.
    #[inline]
    #[must_use]
    pub fn add_sample(mut self, sample: AudioSamples<'_, T>) -> Self {
        self.samples.push(sample.into_owned());
        self
    }

    /// Adds a sine wave to the audio builder.
    #[inline]
    #[must_use]
    pub fn sine_wave(self, frequency: f64, duration: Duration, amplitude: f64) -> Self {
        let sr = self.sample_rate;
        self.add_sample(sine_wave::<T>(frequency, duration, sr, amplitude))
    }

    /// Adds a cosine wave to the audio builder.
    #[inline]
    #[must_use]
    pub fn cosine_wave(self, frequency: f64, duration: Duration, amplitude: f64) -> Self {
        let sr = self.sample_rate;
        self.add_sample(cosine_wave::<T>(frequency, duration, sr, amplitude))
    }

    /// Adds white noise to the audio builder.
    #[cfg(feature = "random-generation")]
    #[inline]
    #[must_use]
    pub fn white_noise(self, duration: Duration, amplitude: f64, seed: Option<u64>) -> Self {
        let sr = self.sample_rate;
        self.add_sample(white_noise::<T>(duration, sr, amplitude, seed))
    }

    /// Adds pink noise to the audio builder.
    #[cfg(feature = "random-generation")]
    #[inline]
    #[must_use]
    pub fn pink_noise(self, duration: Duration, amplitude: f64, seed: Option<u64>) -> Self {
        let sr = self.sample_rate;
        self.add_sample(pink_noise::<T>(duration, sr, amplitude, seed))
    }

    /// Adds a sawtooth wave to the audio builder.
    #[inline]
    #[must_use]
    pub fn sawtooth_wave(self, frequency: f64, duration: Duration, amplitude: f64) -> Self {
        let sr = self.sample_rate;
        self.add_sample(sawtooth_wave::<T>(frequency, duration, sr, amplitude))
    }

    /// Adds a square wave to the audio builder.
    #[inline]
    #[must_use]
    pub fn square_wave(self, frequency: f64, duration: Duration, amplitude: f64) -> Self {
        let sr = self.sample_rate;
        self.add_sample(square_wave::<T>(frequency, duration, sr, amplitude))
    }

    /// Adds a triangle wave to the audio builder.
    #[inline]
    #[must_use]
    pub fn triangle_wave(self, frequency: f64, duration: Duration, amplitude: f64) -> Self {
        let sr = self.sample_rate;
        self.add_sample(triangle_wave::<T>(frequency, duration, sr, amplitude))
    }

    /// Adds silence to the audio builder.
    #[inline]
    #[must_use]
    pub fn silence(self, duration: Duration) -> Self {
        let sr = self.sample_rate;
        self.add_sample(silence::<T>(duration, sr))
    }

    /// Finalises the builder and returns a single concatenated audio sample.
    ///
    /// # Errors
    /// Returns an error if no samples were added to the builder.
    #[inline]
    pub fn build(self) -> AudioSampleResult<AudioSamples<'static, T>> {
        let samples =
            NonEmptyVec::new(self.samples).map_err(|_| crate::AudioSampleError::EmptyData)?;
        AudioSamples::concatenate_owned(samples)
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
#[inline]
#[must_use]
pub fn sine_wave<T>(
    frequency: f64,
    duration: Duration,
    sample_rate: NonZeroU32,
    amplitude: f64,
) -> AudioSamples<'static, T>
where
    T: StandardSample,
{
    let sample_rate_f = f64::from(sample_rate.get());
    let num_samples = (duration.as_secs_f64() * sample_rate_f) as usize;
    let mut samples = Vec::with_capacity(num_samples);
    let two = 2.0;
    let pi = f64::PI();
    let two_pi_frrq = two * pi * frequency;

    for i in 0..num_samples {
        let t = i as f64 / sample_rate_f;
        let sample = amplitude * (two_pi_frrq * t).sin();
        samples.push(sample.convert_to());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array, sample_rate).expect("Guaranteed valid mono audio")
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

#[inline]
#[must_use]
pub fn cosine_wave<T>(
    frequency: f64,
    duration: Duration,
    sample_rate: NonZeroU32,
    amplitude: f64,
) -> AudioSamples<'static, T>
where
    T: StandardSample,
{
    let sample_rate_f = f64::from(sample_rate.get());
    let num_samples = (duration.as_secs_f64() * sample_rate_f) as usize;
    let mut samples = Vec::with_capacity(num_samples);
    let pi = f64::PI();
    let two_pi_freq = 2.0 * pi * frequency;
    for i in 0..num_samples {
        let t = i as f64 / sample_rate_f;
        let sample = amplitude * (two_pi_freq * t).cos();
        samples.push(sample.convert_to());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array, sample_rate).expect("Guaranteed valid mono audio")
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
///
/// # Examples
/// ```no_run
/// use std::time::Duration;
///
/// use audio_samples::brown_noise;
///
/// let noise = brown_noise::<f64>(Duration::from_secs_f32(1.0), 44_100, 0.02, 0.5).unwrap();
/// assert_eq!(noise.samples_per_channel.get(), 44_100);
/// ```
#[cfg(feature = "random-generation")]
#[inline]
pub fn brown_noise<T>(
    duration: Duration,
    sample_rate: NonZeroU32,
    step: f64,
    amplitude: f64,
    seed: Option<u64>,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    T: StandardSample,
{
    let num_samples = (duration.as_secs_f64() * f64::from(sample_rate.get())) as usize;
    let mut samples = Vec::with_capacity(num_samples);
    let mut brown_state = 0.0f64;

    if let Some(seed) = seed {
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);
        let white: f64 = (rng.random::<f64>() - 0.5) * 2.0;
        brown_state += white * step;
        brown_state = brown_state.clamp(-1.0, 1.0);

        let b_state: f64 = brown_state;
        let sample = amplitude * b_state;
        samples.push(sample.convert_to());
    } else {
        for _ in 0..num_samples {
            let white: f64 = (rand::random::<f64>() - 0.5) * 2.0;
            brown_state += white * step;
            brown_state = brown_state.clamp(-1.0, 1.0);

            let b_state: f64 = brown_state;
            let sample = amplitude * b_state;
            samples.push(sample.convert_to());
        }
    }

    let arr = Array1::from_vec(samples);
    Ok(AudioSamples::new_mono(arr, sample_rate).expect("Guaranteed valid mono audio"))
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

#[cfg(feature = "random-generation")]
#[inline]
#[must_use]
pub fn white_noise<T>(
    duration: Duration,
    sample_rate: NonZeroU32,
    amplitude: f64,
    seed: Option<u64>,
) -> AudioSamples<'static, T>
where
    T: StandardSample,
{
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    let num_samples = ((duration.as_secs_f64()) * f64::from(sample_rate.get())) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    let two = 2.0;
    let half = 0.5;

    if let Some(seed) = seed {
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..num_samples {
            let random_value = (rng.random::<f64>() - half) * two;
            let random_value: f64 = random_value;
            let sample = amplitude * random_value;
            samples.push(sample.convert_to());
        }
    } else {
        for _ in 0..num_samples {
            let random_value = (rand::random::<f64>() - half) * two;
            let random_value: f64 = random_value;
            let sample = amplitude * random_value;
            samples.push(sample.convert_to());
        }
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array, sample_rate).expect("Guaranteed valid mono audio")
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

#[cfg(feature = "random-generation")]
#[inline]
#[must_use]
pub fn pink_noise<T>(
    duration: Duration,
    sample_rate: NonZeroU32,
    amplitude: f64,
    seed: Option<u64>,
) -> AudioSamples<'static, T>
where
    T: StandardSample,
{
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    let num_samples = ((duration.as_secs_f64()) * f64::from(sample_rate.get())) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    // Pink noise generation using Paul Kellett's method
    let mut b = [0.0; 7];
    let two = 2.0;
    let half = 0.5;

    let a0 = 0.99886;
    let b0 = 0.0555179;

    let a1 = 0.99332;
    let b1 = 0.0750759;
    let a2 = 0.96900;
    let b2 = 0.1538520;
    let a3 = 0.86650;
    let b3 = 0.3104856;
    let a4 = 0.55000;
    let b4 = 0.5329522;
    let a5 = -0.7616;
    let b5 = -0.0168980;

    let b6 = 0.115926;
    let pink_calc_multiplier1 = 0.5362;
    let pink_calc_multiplier2 = 0.11;

    if let Some(seed) = seed {
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..num_samples {

            let white = (rng.random::<f64>() - half) * two;
            b[0] = a0 * b[0] + white * b0;
            b[1] = a1 * b[1] + white * b1;
            b[2] = a2 * b[2] + white * b2;
            b[3] = a3 * b[3] + white * b3;
            b[4] = a4 * b[4] + white * b4;
            b[5] = a5 * b[5] + white * b5;

            let pink =
                b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + white * pink_calc_multiplier1;
            b[6] = white * b6;
            let pink: f64 = pink * pink_calc_multiplier2;

            let sample = amplitude * pink;
            samples.push(sample.convert_to());
        }
    } else {
        for _ in 0..num_samples {
            let white = (rand::random::<f64>() - half) * two;
            b[0] = a0 * b[0] + white * b0;
            b[1] = a1 * b[1] + white * b1;
            b[2] = a2 * b[2] + white * b2;
            b[3] = a3 * b[3] + white * b3;
            b[4] = a4 * b[4] + white * b4;
            b[5] = a5 * b[5] + white * b5;

            let pink =
                b[0] + b[1] + b[2] + b[3] + b[4] + b[5] + b[6] + white * pink_calc_multiplier1;
            b[6] = white * b6;
            let pink: f64 = pink * pink_calc_multiplier2;

            let sample = amplitude * pink;
            samples.push(sample.convert_to());
        }
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array, sample_rate).expect("Guaranteed valid mono audio")
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

#[inline]
#[must_use]
pub fn square_wave<T>(
    frequency: f64,
    duration: Duration,
    sample_rate: NonZeroU32,
    amplitude: f64,
) -> AudioSamples<'static, T>
where
    T: StandardSample,
{
    let num_samples = ((duration.as_secs_f64()) * f64::from(sample_rate.get())) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    let two_pi = 2.0 * std::f64::consts::PI;
    let freq = frequency;
    let sample_rate_f = f64::from(sample_rate.get());

    for i in 0..num_samples {
        let t = i as f64 / sample_rate_f;
        let arg = two_pi * freq * t;
        let sin_val = arg.sin();
        let sample = if sin_val >= 0.0 {
            amplitude
        } else {
            -amplitude
        };

        samples.push(sample.convert_to());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array, sample_rate).expect("Guaranteed valid mono audio")
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

#[inline]
#[must_use]
pub fn sawtooth_wave<T>(
    frequency: f64,
    duration: Duration,
    sample_rate: NonZeroU32,
    amplitude: f64,
) -> AudioSamples<'static, T>
where
    T: StandardSample,
{
    let sr_f = f64::from(sample_rate.get());

    let num_samples = (duration.as_secs_f64() * sr_f) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    let two_pi = 2.0 * std::f64::consts::PI;
    let freq = frequency;

    for i in 0..num_samples {
        let t = i as f64 / sr_f;
        let arg = two_pi * freq * t;
        // sawtooth: 2 * ((t / (2*pi) - 1) % 1) - 1 for width=1.0
        let phase = arg / two_pi;
        let frac = (phase - 1.0) % 1.0;
        // Handle negative modulo
        let frac = if frac < 0.0 { frac + 1.0 } else { frac };
        let sample = amplitude * (frac * 2.0 - 1.0);
        samples.push(sample.convert_to());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array, sample_rate).expect("Guaranteed valid mono audio")
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

#[inline]
#[must_use]
pub fn triangle_wave<T>(
    frequency: f64,
    duration: Duration,
    sample_rate: NonZeroU32,
    amplitude: f64,
) -> AudioSamples<'static, T>
where
    T: StandardSample,
{
    let sr_f = f64::from(sample_rate.get());
    let freq_f = frequency;

    let num_samples = (duration.as_secs_f64() * sr_f) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    let mut phase = 0.0;
    let phase_inc = freq_f / sr_f;

    for _ in 0..num_samples {
        // triangle: ramp up then down
        let sample = if phase < 0.5 {
            // rising edge: -1 -> +1
            amplitude * 4.0f64.mul_add(phase, -1.0)
        } else {
            // falling edge: +1 -> -1
            amplitude * 4.0f64.mul_add(-phase, 3.0)
        };

        samples.push(sample.convert_to());

        phase += phase_inc;
        if phase >= 1.0 {
            phase -= 1.0;
        }
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array, sample_rate).expect("Guaranteed valid mono audio")
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

#[inline]
#[must_use]
pub fn chirp<T>(
    start_freq: f64,
    end_freq: f64,
    duration: Duration,
    sample_rate: NonZeroU32,
    amplitude: f64,
) -> AudioSamples<'static, T>
where
    T: StandardSample,
{
    let num_samples = ((duration.as_secs_f64()) * f64::from(sample_rate.get())) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    let sr_f = f64::from(sample_rate.get());
    let f0 = start_freq;
    let f1 = end_freq;
    let duration_f = duration.as_secs_f64();
    let k = (f1 - f0) / duration_f; // linear frequency slope

    let two_pi = 2.0 * f64::PI();

    for i in 0..num_samples {
        let t = i as f64 / sr_f;

        // Analytic phase: φ(t) = 2π (f0 t + 0.5 k t^2)
        let phase = two_pi * f0.mul_add(t, 0.5 * k * t * t);

        let value = amplitude * phase.sin();
        samples.push(value.convert_to());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array, sample_rate).expect("guaranteed valid audio")
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

#[inline]
#[must_use]
pub fn impulse<T>(
    duration: Duration,
    sample_rate: NonZeroU32,
    amplitude: f64,
    position: f64,
) -> AudioSamples<'static, T>
where
    T: StandardSample,
{
    let num_samples = ((duration.as_secs_f64()) * f64::from(sample_rate.get())) as usize;
    let mut samples: Vec<T> = vec![0.0.convert_to(); num_samples];

    let impulse_sample = (position * f64::from(sample_rate.get())) as usize;
    if impulse_sample < num_samples {
        samples[impulse_sample] = amplitude.convert_to();
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array, sample_rate).expect("guaranteed valid audio")
}
/// A component of a compound tone, specifying frequency and relative amplitude.
#[derive(Debug, Clone, Copy)]
pub struct ToneComponent {
    /// Frequency in Hz
    pub frequency: f64,
    /// Relative amplitude (typically 0.0 to 1.0)
    pub amplitude: f64,
}

impl ToneComponent {
    /// Creates a new tone component.
    #[inline]
    #[must_use]
    pub const fn new(frequency: f64, amplitude: f64) -> Self {
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
/// let audio = compound_tone::<f64>(&components, Duration::from_secs(1), 44100);
/// ```
#[inline]
#[must_use]
pub fn compound_tone<T>(
    components: &NonEmptySlice<ToneComponent>,
    duration: Duration,
    sample_rate: NonZeroU32,
) -> AudioSamples<'static, T>
where
    T: StandardSample,
{
    let sr_f = f64::from(sample_rate.get());
    let num_samples = ((duration.as_secs_f64()) * sr_f) as usize;
    let two_pi = 2.0 * f64::PI();
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f64 / sr_f;
        let mut sum = 0.0;
        for comp in components {
            sum += comp.amplitude * two_pi * comp.frequency * t.sin();
        }
        samples.push(sum.convert_to());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array, sample_rate).expect("guaranteed valid audio")
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
/// let audio = am_signal::<f64>(440.0, 2.0, 0.5, Duration::from_secs(1), 44100, 0.8);
/// ```
#[inline]
#[must_use]
pub fn am_signal<T>(
    carrier_freq: f64,
    modulator_freq: f64,
    modulation_depth: f64,
    duration: Duration,
    sample_rate: NonZeroU32,
    amplitude: f64,
) -> AudioSamples<'static, T>
where
    T: StandardSample,
{
    let sr_f = f64::from(sample_rate.get());
    let num_samples = ((duration.as_secs_f64()) * sr_f) as usize;

    let two_pi = 2.0 * f64::PI();
    let one = 1.0;
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f64 / sr_f;
        // Envelope: (1 - depth) + depth * (0.5 + 0.5 * sin(2π * mod_freq * t))
        // This gives envelope range from (1-depth) to 1.0
        let envelope =
            modulation_depth.mul_add((two_pi * modulator_freq * t).sin(), one - modulation_depth);
        let carrier = (two_pi * carrier_freq * t).sin();
        let sample = amplitude * envelope * carrier;
        samples.push(sample.convert_to());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array, sample_rate).expect("guaranteed valid audio")
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
/// let audio = exponential_bursts::<f64>(2.0, 30.0, Duration::from_secs(3), 44100, 0.8);
/// ```
#[cfg(feature = "random-generation")]
#[inline]
#[must_use]
pub fn exponential_bursts<T>(
    burst_rate: f64,
    decay_rate: f64,
    duration: Duration,
    sample_rate: NonZeroU32,
    amplitude: f64,
) -> AudioSamples<'static, T>
where
    T: StandardSample,
{
    let sr_f = f64::from(sample_rate.get());
    let num_samples = ((duration.as_secs_f64()) * sr_f) as usize;

    let two_pi = 2.0 * f64::PI();
    let burst_period_threshold = 0.1; // 10% of period is active
    let noise_mix = 0.7;
    let tone_mix = 0.3;
    let tone_freq = 200.0;

    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f64 / sr_f;
        let burst_phase = (burst_rate * t) % 1.0;

        let envelope = if burst_phase < burst_period_threshold {
            (-burst_phase * decay_rate).exp()
        } else {
            0.0
        };

        // Mix of noise and tone for percussive character
        let noise = (rand::random::<f64>() - 0.5) * 2.0;
        let tone = (two_pi * tone_freq * t).sin();
        let sample = amplitude * envelope * (noise_mix * noise + tone_mix * tone);
        samples.push(sample.convert_to());
    }

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array, sample_rate).expect("guaranteed valid audio")
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

#[inline]
#[must_use]
pub fn silence<T>(duration: Duration, sample_rate: NonZeroU32) -> AudioSamples<'static, T>
where
    T: StandardSample,
{
    let num_samples = ((duration.as_secs_f64()) * f64::from(sample_rate.get())) as usize;
    let samples = vec![T::zero(); num_samples];

    let array = Array1::from_vec(samples);
    AudioSamples::new_mono(array, sample_rate).expect("guaranteed valid audio")
}

// ============================================================================
// Multi-channel generation helpers
// ============================================================================

#[cfg(feature = "channels")]
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
/// let stereo = stereo_sine_wave::<f64>(440.0, Duration::from_secs(1), 44100, 0.8);
/// assert_eq!(stereo.num_channels().get(), 2);
/// ```
#[inline]
#[must_use]
pub fn stereo_sine_wave<T>(
    frequency: f64,
    duration: Duration,
    sample_rate: NonZeroU32,
    amplitude: f64,
) -> AudioSamples<'static, T>
where
    T: StandardSample,
{
    use crate::operations::AudioChannelOps;
    let mono = sine_wave::<T>(frequency, duration, sample_rate, amplitude);
    mono.duplicate_to_channels(2)
        .expect("duplicating mono to stereo should not fail")
}

#[cfg(feature = "channels")]
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
/// let stereo = stereo_chirp::<f64>(100.0, 1000.0, Duration::from_secs(1), 44100, 0.8);
/// assert_eq!(stereo.num_channels().get(), 2);
/// ```
#[inline]
#[must_use]
pub fn stereo_chirp<T>(
    start_freq: f64,
    end_freq: f64,
    duration: Duration,
    sample_rate: NonZeroU32,
    amplitude: f64,
) -> AudioSamples<'static, T>
where
    T: StandardSample,
{
    use crate::operations::AudioChannelOps;
    let mono = chirp::<T>(start_freq, end_freq, duration, sample_rate, amplitude);
    mono.duplicate_to_channels(2)
        .expect("duplicating mono to stereo should not fail")
}

#[cfg(feature = "channels")]
/// Generates stereo silence.
///
/// # Arguments
/// * `duration` - Duration of the signal
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// A stereo [`AudioSamples`] containing silence on both channels.
///
///
/// # Examples
///
/// ```rust,no_run
/// use audio_samples::utils::generation::stereo_silence;
/// use std::time::Duration;
///
/// let silent_stereo = stereo_silence::<f64>(Duration::from_secs(2), 44100);
/// assert_eq!(silent_stereo.num_channels().get(), 2);
/// ```
#[inline]
#[must_use]
pub fn stereo_silence<T>(duration: Duration, sample_rate: NonZeroU32) -> AudioSamples<'static, T>
where
    T: StandardSample,
{
    use crate::operations::AudioChannelOps;
    let mono = silence::<T>(duration, sample_rate);
    mono.duplicate_to_channels(2)
        .expect("duplicating mono to stereo should not fail")
}

#[cfg(feature = "channels")]
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
/// assert_eq!(surround.num_channels().get(), 6);
/// ```
#[inline]
#[must_use]
pub fn multichannel_compound_tone<T>(
    components: &NonEmptySlice<ToneComponent>,
    duration: Duration,
    sample_rate: NonZeroU32,
    n_channels: usize,
) -> AudioSamples<'static, T>
where
    T: StandardSample,
{
    use crate::operations::AudioChannelOps;
    let mono = compound_tone::<T>(components, duration, sample_rate);
    mono.duplicate_to_channels(n_channels)
        .expect("duplicating mono to n_channels should not fail")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "statistics")]
    use crate::operations::traits::AudioStatistics;
    use crate::sample_rate;
    use approx_eq::assert_approx_eq;

    #[test]
    fn test_sine_wave_generation() {
        let audio = sine_wave::<f32>(
            440.0,
            Duration::from_secs_f32(1.0),
            sample_rate!(44100),
            1.0,
        );

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio.num_channels().get(), 1);
        assert_eq!(audio.samples_per_channel().get(), 44100);

        #[cfg(feature = "statistics")]
        {
            // Check that the peak is approximately 1.0
            let peak = audio.peak();
            assert!(peak > 0.9 && peak <= 1.0);
        }
    }

    #[test]
    #[cfg(feature = "random-generation")]
    fn test_white_noise_generation() {
        let audio =
            white_noise::<f64>(Duration::from_secs_f32(1.0), sample_rate!(44100), 1.0, None);

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio.num_channels().get(), 1);
        assert_eq!(audio.samples_per_channel().get(), 44100);

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
        let audio = square_wave::<f64>(1.0, Duration::from_secs_f32(1.0), sample_rate!(10), 1.0); // 1 Hz, 10 samples/sec

        assert_eq!(audio.sample_rate(), sample_rate!(10));
        assert_eq!(audio.samples_per_channel().get(), 10);

        // Check that values are either 1.0 or -1.0 (approximately)
        let mono = audio.as_mono().unwrap();
        for &sample in mono.iter() {
            assert!(sample.abs() > 0.9); // Should be close to ±1.0
        }
    }

    #[test]
    fn test_impulse_generation() {
        let audio = impulse::<f64>(Duration::from_secs_f32(1.0), sample_rate!(10), 1.0, 0.5); // Impulse at 0.5 seconds

        assert_eq!(audio.sample_rate(), sample_rate!(10));
        assert_eq!(audio.samples_per_channel().get(), 10);

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
        let audio = silence::<f64>(Duration::from_secs_f32(1.0), sample_rate!(44100));

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio.samples_per_channel().get(), 44100);

        let mono = audio.as_mono().unwrap();

        // Check that all samples are zero
        for &sample in mono.iter() {
            assert_eq!(sample, 0.0);
        }
    }

    #[test]
    fn test_chirp_generation() {
        let audio = chirp::<f64>(
            100.0,
            1000.0,
            Duration::from_secs_f32(1.0),
            sample_rate!(44100),
            1.0,
        );

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio.samples_per_channel().get(), 44100);

        #[cfg(feature = "statistics")]
        {
            // Check that the peak is approximately 1.0
            let peak = audio.peak();
            assert!(peak > 0.9 && peak <= 1.0);
        }
    }
    #[test]
    fn test_compound_tone_generation() {
        let components = [
            ToneComponent::new(440.0, 1.0),
            ToneComponent::new(880.0, 0.5),
            ToneComponent::new(1320.0, 0.25),
        ];
        let components = NonEmptySlice::from_slice(&components).unwrap();
        let audio = compound_tone::<f64>(
            &components,
            Duration::from_secs_f32(1.0),
            sample_rate!(44100),
        );

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio.samples_per_channel().get(), 44100);
        assert_eq!(audio.num_channels().get(), 1);

        // The signal should have variation (not silent)
        let mono = audio.as_mono().unwrap();
        let min_val = mono.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = mono.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_val > min_val);
    }

    #[test]
    fn test_am_signal_generation() {
        let audio = am_signal::<f64>(
            440.0,
            2.0,
            0.5,
            Duration::from_secs_f32(1.0),
            sample_rate!(44100),
            0.8,
        );

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio.samples_per_channel().get(), 44100);
        assert_eq!(audio.num_channels().get(), 1);

        // Peak should be around 0.8 (the amplitude)
        #[cfg(feature = "statistics")]
        {
            let peak = audio.peak();
            assert!(peak > 0.7 && peak <= 0.85, "Peak was {}", peak);
        }
    }

    #[test]
    #[cfg(feature = "random-generation")]
    fn test_exponential_bursts_generation() {
        let audio = exponential_bursts::<f64>(
            2.0,
            30.0,
            Duration::from_secs_f32(1.0),
            sample_rate!(44100),
            0.8,
        );

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio.samples_per_channel().get(), 44100);
        assert_eq!(audio.num_channels().get(), 1);

        // Should have some variation (bursts)
        let mono = audio.as_mono().unwrap();
        let min_val = mono.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = mono.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_val > min_val);
    }

    #[cfg(all(feature = "editing", feature = "channels"))]
    #[test]
    fn test_stereo_sine_wave_generation() {
        let audio = stereo_sine_wave::<f64>(
            440.0,
            Duration::from_secs_f32(1.0),
            sample_rate!(44100),
            0.8,
        );

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio.samples_per_channel().get(), 44100);
        assert_eq!(audio.num_channels().get(), 2);

        // Both channels should be identical - check via underlying array
        let multi = audio.as_multi_channel().unwrap();
        for s_idx in 0..audio.samples_per_channel().get() {
            assert_eq!(multi[(0, s_idx)], multi[(1, s_idx)]);
        }
    }

    #[cfg(all(feature = "editing", feature = "channels"))]
    #[test]
    fn test_multichannel_compound_tone_generation() {
        let components = [
            ToneComponent::new(440.0, 1.0),
            ToneComponent::new(880.0, 0.5),
        ];
        let components = NonEmptySlice::from_slice(&components).unwrap();
        let audio = multichannel_compound_tone::<f64>(
            &components,
            Duration::from_secs_f32(0.1),
            sample_rate!(44100),
            6, // 5.1 surround
        );

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio.num_channels().get(), 6);

        // All 6 channels should be identical - check via underlying array
        let multi = audio.as_multi_channel().unwrap();
        for s_idx in 0..audio.samples_per_channel().get() {
            let first_val = multi[(0, s_idx)];
            for ch_idx in 1..6 {
                assert_eq!(multi[(ch_idx, s_idx)], first_val);
            }
        }
    }
}
