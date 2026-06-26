//! Audio signal generation utilities.

//!

//! Functions for synthesising deterministic and stochastic audio signals: waveforms
//! (sine, cosine, square, sawtooth, triangle), frequency sweeps (chirp), noise (white,
//! pink, brown), silence, impulses, compound multi-harmonic tones, and amplitude-modulated
//! signals.
//!
//! Multi-channel convenience wrappers (`stereo_sine_wave`, `stereo_chirp`, etc.) are
//! provided when the `channels` feature is enabled. A `MonoSampleBuilder` for constructing
//! concatenated sequences through method chaining is available when `editing` is enabled.

//! Generated signals are the backbone of audio unit tests, benchmarks, and DSP prototyping.
//! By centralising generators in this module, dependent code can produce reproducible test
//! data without reaching for external audio files.

//! All generators accept a `NonZeroU32` sample rate. Use the `sample_rate!` macro to
//! construct one from an integer literal:
//!
//! ```rust
//! use audio_samples::utils::generation::sine_wave;
//! use audio_samples::sample_rate;
//! use std::time::Duration;
//!
//! let tone = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(44100), 1.0);
//! assert_eq!(tone.samples_per_channel().get(), 44100);
//! ```
//!
//! Some helpers are feature-gated:
//! - `MonoSampleBuilder` requires `feature = "editing"`.
//! - Noise generators (`white_noise`, `pink_noise`, `brown_noise`, `exponential_bursts`) require `feature = "random-generation"`.
//! - `stereo_*` and `multichannel_compound_tone` require `feature = "channels"`.

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
///
/// `MonoSampleBuilder` accumulates a sequence of mono audio segments and concatenates them
/// into a single [`AudioSamples`] buffer on [`build`](MonoSampleBuilder::build). All segments
/// use the same sample rate provided at construction.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::generation::MonoSampleBuilder;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let audio = MonoSampleBuilder::<f32>::new(sample_rate!(44100))
///     .sine_wave(440.0, Duration::from_millis(100), 1.0)
///     .silence(Duration::from_millis(50))
///     .sine_wave(880.0, Duration::from_millis(100), 0.5)
///     .build()
///     .unwrap();
///
/// assert_eq!(audio.num_channels().get(), 1);
/// ```
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
    ///
    /// # Arguments
    ///
    /// - `sample_rate` – The sample rate used for all segments added to this builder.
    #[inline]
    #[must_use]
    pub const fn new(sample_rate: SampleRate) -> Self {
        Self {
            samples: Vec::new(),
            sample_rate,
        }
    }

    /// Appends an existing [`AudioSamples`] segment to the builder.
    ///
    /// The segment is converted to an owned buffer before being stored.
    #[inline]
    #[must_use]
    pub fn add_sample(mut self, sample: AudioSamples<'_, T>) -> Self {
        self.samples.push(sample.into_owned());
        self
    }

    /// Appends a sine wave segment to the builder.
    ///
    /// # Arguments
    ///
    /// - `frequency` – Frequency in Hz.
    /// - `duration` – Duration of the segment.
    /// - `amplitude` – Peak amplitude.
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
        let samples = NonEmptyVec::new(self.samples)
            .map_err(|_| crate::AudioSampleError::empty_data("MonoSampleBuilder::build"))?;
        AudioSamples::concatenate_owned(samples)
    }
}

/// Generates a sine wave with the specified parameters.
///
/// Produces `floor(duration * sample_rate)` samples of `amplitude * sin(2π · frequency · t)`,
/// where `t` is the sample time in seconds. The output is a mono [`AudioSamples`] buffer.
///
/// # Arguments
///
/// - `frequency` – Frequency of the sine wave in Hz.
/// - `duration` – Duration of the generated signal.
/// - `sample_rate` – Sampling rate in Hz.
/// - `amplitude` – Peak amplitude. A value of `1.0` reaches the full-scale positive peak.
///
/// # Returns
///
/// A mono [`AudioSamples`] buffer containing the sine wave at the given sample rate.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::generation::sine_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let tone = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(44100), 1.0);
/// assert_eq!(tone.sample_rate().get(), 44100);
/// assert_eq!(tone.samples_per_channel().get(), 44100);
/// assert_eq!(tone.num_channels().get(), 1);
/// ```
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
    // safety: we just created the data so is non-empty
    unsafe { AudioSamples::new_mono_unchecked(array, sample_rate) }
}

/// Generates a cosine wave with the specified parameters.
///
/// Produces `floor(duration * sample_rate)` samples of `amplitude * cos(2π · frequency · t)`.
/// The output is a mono [`AudioSamples`] buffer.
///
/// # Arguments
///
/// - `frequency` – Frequency of the cosine wave in Hz.
/// - `duration` – Duration of the generated signal.
/// - `sample_rate` – Sampling rate in Hz.
/// - `amplitude` – Peak amplitude.
///
/// # Returns
///
/// A mono [`AudioSamples`] buffer containing the cosine wave at the given sample rate.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::generation::cosine_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let tone = cosine_wave::<f32>(440.0, Duration::from_millis(10), sample_rate!(44100), 0.5);
/// assert_eq!(tone.num_channels().get(), 1);
/// ```
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
    // safety: we just created the data so is non-empty
    unsafe { AudioSamples::new_mono_unchecked(array, sample_rate) }
}

/// Generates brown noise (Brownian / red noise) with the specified parameters.
///
/// Brown noise has a power spectral density that rolls off at −6 dB per octave, producing
/// a deeper, lower-frequency character than white or pink noise. It is synthesised via a
/// random walk where each sample is the previous value plus a uniform random step, clamped
/// to `[−1, 1]`.
///
/// # Arguments
///
/// - `duration` – Duration of the generated signal.
/// - `sample_rate` – Sampling rate in Hz.
/// - `step` – Step size of the Brownian walk. Larger values produce rougher, more volatile
///   noise; smaller values produce smoother, more slowly drifting noise. A value around
///   `0.02` is a common starting point.
/// - `amplitude` – Overall amplitude scale applied to each output sample.
/// - `seed` – Optional RNG seed for reproducibility. Pass `None` for non-deterministic output.
///
/// # Returns
///
/// `Ok(audio)` containing the generated brown noise as a mono [`AudioSamples`] buffer.
///
/// # Errors
///
/// Currently infallible; the `Result` wrapper is reserved for future validation.
///
/// # Examples
///
/// ```rust,no_run
/// use audio_samples::utils::generation::brown_noise;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let noise = brown_noise::<f64>(Duration::from_secs(1), sample_rate!(44100), 0.02, 0.5, None);
/// assert_eq!(noise.samples_per_channel().get(), 44100);
/// ```
#[cfg(feature = "random-generation")]
#[inline]
#[must_use]
pub fn brown_noise<T>(
    duration: Duration,
    sample_rate: NonZeroU32,
    step: f64,
    amplitude: f64,
    seed: Option<u64>,
) -> AudioSamples<'static, T>
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
        for _ in 0..num_samples {
            let white: f64 = (rng.random::<f64>() - 0.5) * 2.0;
            brown_state += white * step;
            brown_state = brown_state.clamp(-1.0, 1.0);

            let b_state: f64 = brown_state;
            let sample = amplitude * b_state;
            samples.push(sample.convert_to());
        }
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
    // safety: we just created the data so is non-empty
    unsafe { AudioSamples::new_mono_unchecked(arr, sample_rate) }
}

/// Generates white noise with the specified parameters.
///
/// White noise has equal expected energy across all frequencies within the Nyquist range.
/// Each sample is drawn independently from a uniform distribution scaled to the requested
/// amplitude.
///
/// # Arguments
///
/// - `duration` – Duration of the generated signal.
/// - `sample_rate` – Sampling rate in Hz.
/// - `amplitude` – Peak amplitude of the noise envelope (0.0 to 1.0).
/// - `seed` – Optional RNG seed for reproducibility. Pass `None` for non-deterministic output.
///
/// # Returns
///
/// A mono [`AudioSamples`] buffer containing the generated white noise.
///
/// # Examples
///
/// ```rust,no_run
/// use audio_samples::utils::generation::white_noise;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let noise = white_noise::<f32>(Duration::from_millis(100), sample_rate!(44100), 0.5, None);
/// assert_eq!(noise.num_channels().get(), 1);
/// ```
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
    // safety: we just created the data so is non-empty
    unsafe { AudioSamples::new_mono_unchecked(array, sample_rate) }
}

/// Generates pink noise (1/f noise) with the specified parameters.
///
/// Pink noise has equal energy per octave, meaning its power spectral density decreases at
/// −3 dB per octave. It is generated using Paul Kellett's IIR approximation method, which
/// shapes white noise through a bank of first-order filters.
///
/// # Arguments
///
/// - `duration` – Duration of the generated signal.
/// - `sample_rate` – Sampling rate in Hz.
/// - `amplitude` – Overall amplitude scale applied to each output sample.
/// - `seed` – Optional RNG seed for reproducibility. Pass `None` for non-deterministic output.
///
/// # Returns
///
/// A mono [`AudioSamples`] buffer containing the generated pink noise.
///
/// # Examples
///
/// ```rust,no_run
/// use audio_samples::utils::generation::pink_noise;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let noise = pink_noise::<f32>(Duration::from_millis(100), sample_rate!(44100), 0.5, None);
/// assert_eq!(noise.num_channels().get(), 1);
/// ```
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

    let a0 = 0.998_86;
    let b0 = 0.055_517_9;

    let a1 = 0.993_32;
    let b1 = 0.075_075_9;
    let a2 = 0.969_00;
    let b2 = 0.153_852_0;
    let a3 = 0.866_50;
    let b3 = 0.310_485_6;
    let a4 = 0.550_00;
    let b4 = 0.532_952_2;
    let a5 = -0.761_6;
    let b5 = -0.016_898_0;

    let b6 = 0.115_926;
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
    // safety: we just created the data so is non-empty
    unsafe { AudioSamples::new_mono_unchecked(array, sample_rate) }
}

/// Generates a square wave with the specified parameters.
///
/// Each sample is `+amplitude` when `sin(2π · frequency · t) ≥ 0` and `−amplitude`
/// otherwise, producing an ideal (non-band-limited) square wave. The duty cycle is fixed
/// at 50%.
///
/// # Arguments
///
/// - `frequency` – Frequency of the square wave in Hz.
/// - `duration` – Duration of the generated signal.
/// - `sample_rate` – Sampling rate in Hz.
/// - `amplitude` – Peak amplitude.
///
/// # Returns
///
/// A mono [`AudioSamples`] buffer containing the generated square wave.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::generation::square_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let wave = square_wave::<f32>(440.0, Duration::from_millis(10), sample_rate!(44100), 1.0);
/// assert_eq!(wave.num_channels().get(), 1);
/// ```
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
    // safety: we just created the data so is non-empty
    unsafe { AudioSamples::new_mono_unchecked(array, sample_rate) }
}

/// Generates a sawtooth wave with the specified parameters.
///
/// Produces a rising sawtooth waveform that ramps linearly from `−amplitude` to
/// `+amplitude` over each period and then resets. The wave is non-band-limited (no
/// anti-aliasing is applied).
///
/// # Arguments
///
/// - `frequency` – Frequency of the sawtooth wave in Hz.
/// - `duration` – Duration of the generated signal.
/// - `sample_rate` – Sampling rate in Hz.
/// - `amplitude` – Peak amplitude.
///
/// # Returns
///
/// A mono [`AudioSamples`] buffer containing the generated sawtooth wave.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::generation::sawtooth_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let wave = sawtooth_wave::<f32>(440.0, Duration::from_millis(10), sample_rate!(44100), 1.0);
/// assert_eq!(wave.num_channels().get(), 1);
/// ```
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
    // safety: we just created the data so is non-empty
    unsafe { AudioSamples::new_mono_unchecked(array, sample_rate) }
}

/// Generates a triangle wave with the specified parameters.
///
/// Produces a piecewise-linear waveform that rises from `−amplitude` to `+amplitude` over
/// the first half-period and falls back to `−amplitude` over the second half-period. The
/// wave is non-band-limited.
///
/// # Arguments
///
/// - `frequency` – Frequency of the triangle wave in Hz.
/// - `duration` – Duration of the generated signal.
/// - `sample_rate` – Sampling rate in Hz.
/// - `amplitude` – Peak amplitude.
///
/// # Returns
///
/// A mono [`AudioSamples`] buffer containing the generated triangle wave.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::generation::triangle_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let wave = triangle_wave::<f32>(440.0, Duration::from_millis(10), sample_rate!(44100), 1.0);
/// assert_eq!(wave.num_channels().get(), 1);
/// ```
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
    // safety: we just created the data so is non-empty
    unsafe { AudioSamples::new_mono_unchecked(array, sample_rate) }
}

/// Generates a linear chirp (frequency sweep) signal.
///
/// Produces a sinusoidal signal whose instantaneous frequency increases linearly from
/// `start_freq` to `end_freq` over the given duration. The analytic phase is
/// `2π (f₀ t + 0.5 k t²)` where `k = (f₁ − f₀) / duration`.
///
/// # Arguments
///
/// - `start_freq` – Starting frequency in Hz.
/// - `end_freq` – Ending frequency in Hz.
/// - `duration` – Duration of the generated signal.
/// - `sample_rate` – Sampling rate in Hz.
/// - `amplitude` – Peak amplitude.
///
/// # Returns
///
/// A mono [`AudioSamples`] buffer containing the generated chirp signal.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::generation::chirp;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let sweep = chirp::<f32>(100.0, 8000.0, Duration::from_millis(100), sample_rate!(44100), 1.0);
/// assert_eq!(sweep.num_channels().get(), 1);
/// ```
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
    // safety: we just created the data so is non-empty
    unsafe { AudioSamples::new_mono_unchecked(array, sample_rate) }
}

/// Generates an exponential (logarithmic) chirp — a frequency sweep whose instantaneous
/// frequency grows geometrically rather than linearly.
///
/// Unlike [`chirp`], which sweeps frequency linearly, this generator sweeps frequency
/// exponentially so that equal time intervals correspond to equal frequency *ratios*
/// (octaves). With `k = (end_freq / start_freq)^(1/T)` the instantaneous frequency is
/// `f(t) = start_freq · kᵗ` and the analytic phase is
/// `φ(t) = 2π · start_freq · ((kᵗ − 1) / ln k)`. When `start_freq == end_freq` (so `k == 1`)
/// the signal degenerates to a pure tone at that frequency.
///
/// # Arguments
///
/// - `start_freq` – Starting frequency in Hz. Must be `> 0`.
/// - `end_freq` – Ending frequency in Hz. Must be `> 0`.
/// - `duration` – Duration of the generated signal.
/// - `sample_rate` – Sampling rate in Hz.
/// - `amplitude` – Peak amplitude.
///
/// # Returns
///
/// A mono [`AudioSamples`] buffer containing the generated exponential chirp.
///
/// # Panics
///
/// Panics if `start_freq` or `end_freq` is not strictly positive.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::generation::exponential_chirp;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let sweep = exponential_chirp::<f32>(100.0, 8000.0, Duration::from_millis(100), sample_rate!(44100), 1.0);
/// assert_eq!(sweep.num_channels().get(), 1);
/// ```
#[inline]
#[must_use]
pub fn exponential_chirp<T>(
    start_freq: f64,
    end_freq: f64,
    duration: Duration,
    sample_rate: NonZeroU32,
    amplitude: f64,
) -> AudioSamples<'static, T>
where
    T: StandardSample,
{
    assert!(
        start_freq > 0.0 && end_freq > 0.0,
        "exponential_chirp requires start_freq > 0 and end_freq > 0"
    );

    let sr_f = f64::from(sample_rate.get());
    let num_samples = (duration.as_secs_f64() * sr_f) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    let duration_f = duration.as_secs_f64();
    let two_pi = 2.0 * f64::PI();
    // k is the per-second geometric frequency growth factor.
    let k = (end_freq / start_freq).powf(1.0 / duration_f);
    let ln_k = k.ln();

    for i in 0..num_samples {
        let t = i as f64 / sr_f;
        // Analytic phase. For k == 1 (start == end) ln_k == 0 and the limit of
        // (k^t - 1)/ln k as ln_k -> 0 is t, i.e. a pure tone at start_freq.
        let phase = if ln_k == 0.0 {
            two_pi * start_freq * t
        } else {
            two_pi * start_freq * ((k.powf(t) - 1.0) / ln_k)
        };
        let value = amplitude * phase.sin();
        samples.push(value.convert_to());
    }

    let array = Array1::from_vec(samples);
    // safety: we just created the data so is non-empty
    unsafe { AudioSamples::new_mono_unchecked(array, sample_rate) }
}

/// Generates a band-limited (alias-free) square wave via additive synthesis.
///
/// Whereas [`square_wave`] hard-clips a sine and therefore contains harmonics above the
/// Nyquist frequency that fold back as aliasing, this generator sums only the odd harmonics
/// that lie strictly below Nyquist, producing an exactly band-limited signal:
/// `(4/π) · Σ sin(2π(2k−1)f t)/(2k−1)` for every odd harmonic `(2k−1)·frequency < Nyquist`.
///
/// # Arguments
///
/// - `frequency` – Fundamental frequency of the square wave in Hz.
/// - `duration` – Duration of the generated signal.
/// - `sample_rate` – Sampling rate in Hz.
/// - `amplitude` – Peak amplitude of the underlying ideal square wave.
///
/// # Returns
///
/// A mono [`AudioSamples`] buffer containing the band-limited square wave.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::generation::square_wave_bandlimited;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let wave = square_wave_bandlimited::<f32>(440.0, Duration::from_millis(10), sample_rate!(44100), 1.0);
/// assert_eq!(wave.num_channels().get(), 1);
/// ```
#[inline]
#[must_use]
pub fn square_wave_bandlimited<T>(
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

    let two_pi = 2.0 * f64::PI();
    let nyquist = sr_f / 2.0;
    let scale = amplitude * 4.0 / f64::PI();

    // Collect odd harmonic numbers (1, 3, 5, ...) whose frequency is below Nyquist.
    let mut harmonics: Vec<f64> = Vec::new();
    let mut h = 1.0;
    while h * frequency < nyquist {
        harmonics.push(h);
        h += 2.0;
    }

    for i in 0..num_samples {
        let t = i as f64 / sr_f;
        let mut sum = 0.0;
        for &n in &harmonics {
            sum += (two_pi * n * frequency * t).sin() / n;
        }
        let sample = scale * sum;
        samples.push(sample.convert_to());
    }

    let array = Array1::from_vec(samples);
    // safety: we just created the data so is non-empty
    unsafe { AudioSamples::new_mono_unchecked(array, sample_rate) }
}

/// Generates a band-limited (alias-free) sawtooth wave via additive synthesis.
///
/// Whereas [`sawtooth_wave`] ramps linearly and therefore aliases, this generator sums only
/// the harmonics below Nyquist, producing an exactly band-limited signal:
/// `(2/π) · Σ (−1)^(k+1) sin(2π k f t)/k` for every harmonic `k·frequency < Nyquist`.
///
/// # Arguments
///
/// - `frequency` – Fundamental frequency of the sawtooth wave in Hz.
/// - `duration` – Duration of the generated signal.
/// - `sample_rate` – Sampling rate in Hz.
/// - `amplitude` – Peak amplitude of the underlying ideal sawtooth wave.
///
/// # Returns
///
/// A mono [`AudioSamples`] buffer containing the band-limited sawtooth wave.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::generation::sawtooth_wave_bandlimited;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let wave = sawtooth_wave_bandlimited::<f32>(440.0, Duration::from_millis(10), sample_rate!(44100), 1.0);
/// assert_eq!(wave.num_channels().get(), 1);
/// ```
#[inline]
#[must_use]
pub fn sawtooth_wave_bandlimited<T>(
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

    let two_pi = 2.0 * f64::PI();
    let nyquist = sr_f / 2.0;
    let scale = amplitude * 2.0 / f64::PI();

    // Collect all harmonic numbers (1, 2, 3, ...) whose frequency is below Nyquist,
    // paired with their alternating sign (-1)^(k+1).
    let mut harmonics: Vec<(f64, f64)> = Vec::new();
    let mut k = 1.0;
    while k * frequency < nyquist {
        let sign = if (k as i64) % 2 == 1 { 1.0 } else { -1.0 };
        harmonics.push((k, sign));
        k += 1.0;
    }

    for i in 0..num_samples {
        let t = i as f64 / sr_f;
        let mut sum = 0.0;
        for &(n, sign) in &harmonics {
            sum += sign * (two_pi * n * frequency * t).sin() / n;
        }
        let sample = scale * sum;
        samples.push(sample.convert_to());
    }

    let array = Array1::from_vec(samples);
    // safety: we just created the data so is non-empty
    unsafe { AudioSamples::new_mono_unchecked(array, sample_rate) }
}

/// Generates a band-limited (alias-free) triangle wave via additive synthesis.
///
/// Whereas [`triangle_wave`] is piecewise-linear and therefore aliases, this generator sums
/// only the odd harmonics below Nyquist, producing an exactly band-limited signal:
/// `(8/π²) · Σ (−1)^k sin(2π(2k+1)f t)/(2k+1)²` for `k = 0, 1, 2, …` while
/// `(2k+1)·frequency < Nyquist`.
///
/// # Arguments
///
/// - `frequency` – Fundamental frequency of the triangle wave in Hz.
/// - `duration` – Duration of the generated signal.
/// - `sample_rate` – Sampling rate in Hz.
/// - `amplitude` – Peak amplitude of the underlying ideal triangle wave.
///
/// # Returns
///
/// A mono [`AudioSamples`] buffer containing the band-limited triangle wave.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::generation::triangle_wave_bandlimited;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let wave = triangle_wave_bandlimited::<f32>(440.0, Duration::from_millis(10), sample_rate!(44100), 1.0);
/// assert_eq!(wave.num_channels().get(), 1);
/// ```
#[inline]
#[must_use]
pub fn triangle_wave_bandlimited<T>(
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

    let two_pi = 2.0 * f64::PI();
    let nyquist = sr_f / 2.0;
    let scale = amplitude * 8.0 / (f64::PI() * f64::PI());

    // For k = 0, 1, 2, ... the harmonic number is (2k+1); include while below Nyquist.
    // The sign alternates as (-1)^k.
    let mut harmonics: Vec<(f64, f64)> = Vec::new();
    let mut k: i64 = 0;
    loop {
        let n = (2 * k + 1) as f64;
        if n * frequency >= nyquist {
            break;
        }
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        harmonics.push((n, sign));
        k += 1;
    }

    for i in 0..num_samples {
        let t = i as f64 / sr_f;
        let mut sum = 0.0;
        for &(n, sign) in &harmonics {
            sum += sign * (two_pi * n * frequency * t).sin() / (n * n);
        }
        let sample = scale * sum;
        samples.push(sample.convert_to());
    }

    let array = Array1::from_vec(samples);
    // safety: we just created the data so is non-empty
    unsafe { AudioSamples::new_mono_unchecked(array, sample_rate) }
}

/// Generates a frequency-modulated (FM) signal.
///
/// A sinusoidal carrier whose instantaneous phase is modulated by a sinusoidal modulator:
/// `amplitude · sin(2π·carrier·t + modulation_index·sin(2π·modulator·t))`. With
/// `modulation_index == 0` this reduces exactly to a pure sine at `carrier_freq`.
///
/// # Arguments
///
/// - `carrier_freq` – Carrier frequency in Hz.
/// - `modulator_freq` – Modulator frequency in Hz.
/// - `modulation_index` – Modulation index (peak phase deviation in radians).
/// - `duration` – Duration of the generated signal.
/// - `sample_rate` – Sampling rate in Hz.
/// - `amplitude` – Peak amplitude.
///
/// # Returns
///
/// A mono [`AudioSamples`] buffer containing the FM signal.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::generation::fm_signal;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// // 440 Hz carrier modulated at 110 Hz with index 2.0.
/// let audio = fm_signal::<f64>(440.0, 110.0, 2.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
/// assert_eq!(audio.num_channels().get(), 1);
/// ```
#[inline]
#[must_use]
pub fn fm_signal<T>(
    carrier_freq: f64,
    modulator_freq: f64,
    modulation_index: f64,
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

    let two_pi = 2.0 * f64::PI();
    let two_pi_carrier = two_pi * carrier_freq;
    let two_pi_modulator = two_pi * modulator_freq;

    for i in 0..num_samples {
        let t = i as f64 / sr_f;
        let modulation = modulation_index * (two_pi_modulator * t).sin();
        let phase = two_pi_carrier.mul_add(t, modulation);
        let sample = amplitude * phase.sin();
        samples.push(sample.convert_to());
    }

    let array = Array1::from_vec(samples);
    // safety: we just created the data so is non-empty
    unsafe { AudioSamples::new_mono_unchecked(array, sample_rate) }
}

/// Generates a unit impulse (Dirac delta approximation) signal.
///
/// Produces a buffer of zeros with a single non-zero sample at `floor(position * sample_rate)`.
/// If the computed sample index is out of bounds, the entire output is silence.
///
/// # Arguments
///
/// - `duration` – Duration of the generated signal.
/// - `sample_rate` – Sampling rate in Hz.
/// - `amplitude` – Value of the impulse sample.
/// - `position` – Time offset of the impulse in seconds.
///
/// # Returns
///
/// A mono [`AudioSamples`] buffer containing the impulse at the specified position.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::generation::impulse;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// // 1-second buffer, impulse at t = 0.5 s (sample 22050 at 44100 Hz).
/// let imp = impulse::<f64>(Duration::from_secs(1), sample_rate!(44100), 1.0, 0.5);
/// let mono = imp.as_mono().unwrap();
/// assert!((mono[22050] - 1.0).abs() < 1e-12);
/// ```
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
    // safety: we just created the data so is non-empty
    unsafe { AudioSamples::new_mono_unchecked(array, sample_rate) }
}
/// A component of a compound tone, specifying frequency and relative amplitude.
///
/// Used as input to [`compound_tone`] and [`multichannel_compound_tone`] to define a
/// harmonic series or an arbitrary set of simultaneous sinusoids.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ToneComponent {
    /// Frequency in Hz.
    pub frequency: f64,
    /// Relative amplitude (typically 0.0 to 1.0).
    pub amplitude: f64,
}

impl ToneComponent {
    /// Creates a new tone component.
    ///
    /// # Arguments
    ///
    /// - `frequency` – Frequency of this component in Hz.
    /// - `amplitude` – Relative amplitude of this component.
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
///
/// ```rust
/// use audio_samples::utils::generation::{compound_tone, ToneComponent};
/// use audio_samples::sample_rate;
/// use non_empty_slice::NonEmptySlice;
/// use std::time::Duration;
///
/// // 440 Hz fundamental with two harmonics.
/// let raw = [
///     ToneComponent::new(440.0, 1.0),   // fundamental
///     ToneComponent::new(880.0, 0.5),   // 2nd harmonic
///     ToneComponent::new(1320.0, 0.25), // 3rd harmonic
/// ];
/// let components = NonEmptySlice::from_slice(&raw).unwrap();
/// let audio = compound_tone::<f64>(components, Duration::from_millis(100), sample_rate!(44100));
/// assert_eq!(audio.num_channels().get(), 1);
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
            sum += comp.amplitude * (two_pi * comp.frequency * t).sin();
        }
        samples.push(sum.convert_to());
    }

    let array = Array1::from_vec(samples);
    // safety: we just created the data so is non-empty
    unsafe { AudioSamples::new_mono_unchecked(array, sample_rate) }
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
///
/// ```rust
/// use audio_samples::utils::generation::am_signal;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// // 440 Hz carrier modulated at 2 Hz with 50% depth.
/// let audio = am_signal::<f64>(440.0, 2.0, 0.5, Duration::from_millis(100), sample_rate!(44100), 0.8);
/// assert_eq!(audio.num_channels().get(), 1);
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
    // safety: we just created the data so is non-empty
    unsafe { AudioSamples::new_mono_unchecked(array, sample_rate) }
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
///
/// ```rust,no_run
/// use audio_samples::utils::generation::exponential_bursts;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// // 2 bursts per second with fast decay.
/// let audio = exponential_bursts::<f64>(2.0, 30.0, Duration::from_secs(3), sample_rate!(44100), 0.8);
/// assert_eq!(audio.num_channels().get(), 1);
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
    // safety: we know array is non-empty
    unsafe { AudioSamples::new_mono_unchecked(array, sample_rate) }
}

/// Generates a silence buffer (all-zero samples) with the specified duration.
///
/// # Arguments
///
/// - `duration` – Duration of the generated signal.
/// - `sample_rate` – Sampling rate in Hz.
///
/// # Returns
///
/// A mono [`AudioSamples`] buffer containing zero-valued samples for the entire duration.
///
/// # Panics
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::generation::silence;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let silent = silence::<f32>(Duration::from_millis(100), sample_rate!(44100));
/// assert_eq!(silent.num_channels().get(), 1);
/// assert_eq!(silent.samples_per_channel().get(), 4410);
/// ```
#[inline]
#[must_use]
pub fn silence<T>(duration: Duration, sample_rate: NonZeroU32) -> AudioSamples<'static, T>
where
    T: StandardSample,
{
    let num_samples = ((duration.as_secs_f64()) * f64::from(sample_rate.get())) as usize;
    let samples = vec![T::zero(); num_samples];

    let array = Array1::from_vec(samples);
    // safety: we know array is non-empty
    unsafe { AudioSamples::new_mono_unchecked(array, sample_rate) }
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
///
/// ```rust,no_run
/// use audio_samples::utils::generation::stereo_sine_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let stereo = stereo_sine_wave::<f64>(440.0, Duration::from_secs(1), sample_rate!(44100), 0.8);
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
/// # Panics
///
/// If ndarray cannot stack the channels for some reason
/// # Examples
///
/// ```rust,no_run
/// use audio_samples::utils::generation::stereo_chirp;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let stereo = stereo_chirp::<f64>(100.0, 1000.0, Duration::from_secs(1), sample_rate!(44100), 0.8);
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
/// # Panics
///
/// If ndarray cannot stack the channels for some reason
///
/// # Examples
///
/// ```rust,no_run
/// use audio_samples::utils::generation::stereo_silence;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let silent_stereo = stereo_silence::<f64>(Duration::from_secs(2), sample_rate!(44100));
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
///
/// ```rust,no_run
/// use audio_samples::utils::generation::{multichannel_compound_tone, ToneComponent};
/// use audio_samples::sample_rate;
/// use non_empty_slice::NonEmptySlice;
/// use std::time::Duration;
///
/// let raw = [ToneComponent::new(440.0, 1.0), ToneComponent::new(880.0, 0.5)];
/// let components = NonEmptySlice::from_slice(&raw).unwrap();
/// // 5.1 surround sound test tone.
/// let surround = multichannel_compound_tone::<f64>(components, Duration::from_secs(1), sample_rate!(44100), 6);
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
        assert_approx_eq!(mono[5], 1.0, 1e-6);
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
            components,
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

    // Regression test for BUG 1: operator-precedence bug in compound_tone made
    // `.sin()` bind only to `t`, producing amplitude*2pi*f*sin(t) instead of a
    // tone at `comp.frequency`. A single 1000 Hz component over 1 s should cross
    // zero ~2*f times. The buggy expression produces a ramped sin(t) (near DC),
    // crossing zero only ~2 times over a second.
    #[test]
    fn test_compound_tone_frequency_zero_crossings() {
        let freq = 1000.0;
        let sr = 44100u32;
        let components = [ToneComponent::new(freq, 1.0)];
        let components = NonEmptySlice::from_slice(&components).unwrap();
        let audio = compound_tone::<f64>(
            components,
            Duration::from_secs_f64(1.0),
            sample_rate!(44100),
        );

        let mono = audio.as_mono().unwrap();
        let samples: Vec<f64> = mono.iter().cloned().collect();

        let mut crossings = 0usize;
        for w in samples.windows(2) {
            if (w[0] <= 0.0 && w[1] > 0.0) || (w[0] >= 0.0 && w[1] < 0.0) {
                crossings += 1;
            }
        }

        // Expected ~2*f zero crossings per second.
        let expected = 2.0 * freq;
        let _ = sr;
        assert!(
            (crossings as f64 - expected).abs() < expected * 0.05,
            "expected ~{expected} zero crossings for a {freq} Hz tone, got {crossings}"
        );
    }

    // Regression test for BUG 2: the seeded branch of brown_noise had no
    // generation loop and pushed exactly one sample. It must now return a full
    // buffer, be deterministic for a fixed seed, and differ across seeds.
    #[test]
    #[cfg(feature = "random-generation")]
    fn test_brown_noise_seeded_length_and_determinism() {
        let dur = Duration::from_secs_f64(0.1);
        let sr = sample_rate!(44100);
        let num_samples = (0.1 * 44100.0) as usize;

        let a = brown_noise::<f64>(dur, sr, 0.02, 0.5, Some(42));
        let b = brown_noise::<f64>(dur, sr, 0.02, 0.5, Some(42));
        let c = brown_noise::<f64>(dur, sr, 0.02, 0.5, Some(7));

        // Correct length (not a 1-sample buffer).
        assert_eq!(a.samples_per_channel().get(), num_samples);

        let av: Vec<f64> = a.as_mono().unwrap().iter().cloned().collect();
        let bv: Vec<f64> = b.as_mono().unwrap().iter().cloned().collect();
        let cv: Vec<f64> = c.as_mono().unwrap().iter().cloned().collect();

        // Deterministic: same seed -> identical output.
        assert_eq!(av, bv);
        // Different seed -> different output.
        assert_ne!(av, cv);
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

    // Counts signed zero crossings over a window and converts to an estimated
    // instantaneous frequency (crossings per second / 2).
    fn estimate_freq_over_window(samples: &[f64], sr: f64) -> f64 {
        let mut crossings = 0usize;
        for w in samples.windows(2) {
            if (w[0] <= 0.0 && w[1] > 0.0) || (w[0] >= 0.0 && w[1] < 0.0) {
                crossings += 1;
            }
        }
        let window_secs = samples.len() as f64 / sr;
        (crossings as f64 / window_secs) / 2.0
    }

    #[test]
    fn test_exponential_chirp_endpoints() {
        let start = 200.0;
        let end = 4000.0;
        let sr = 44100u32;
        let audio = exponential_chirp::<f64>(
            start,
            end,
            Duration::from_secs_f64(1.0),
            sample_rate!(44100),
            1.0,
        );

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio.samples_per_channel().get(), 44100);
        assert_eq!(audio.num_channels().get(), 1);

        let mono = audio.as_mono().unwrap();
        let s: Vec<f64> = mono.iter().cloned().collect();

        // Local zero-crossing rate over a short window at each end. Use a window
        // large enough to contain several cycles at the relevant frequency.
        let win = 4410; // 0.1 s
        let near_start = estimate_freq_over_window(&s[..win], sr as f64);
        let near_end = estimate_freq_over_window(&s[s.len() - win..], sr as f64);

        // Frequency near t=0 should be close to start_freq, and near t=T close to
        // end_freq. The window averages over a slice so we allow generous tolerance.
        assert!(
            (near_start - start).abs() < start * 0.30,
            "start: expected ~{start}, measured {near_start}"
        );
        assert!(
            (near_end - end).abs() < end * 0.30,
            "end: expected ~{end}, measured {near_end}"
        );
        // Exponential sweep must be monotonically rising: end >> start.
        assert!(near_end > near_start * 4.0);
    }

    #[test]
    fn test_exponential_chirp_equal_freqs_is_tone() {
        // start == end => k == 1 => pure tone at that frequency.
        let f = 1000.0;
        let sr = 48000u32;
        let audio = exponential_chirp::<f64>(
            f,
            f,
            Duration::from_secs_f64(1.0),
            sample_rate!(48000),
            1.0,
        );
        let mono = audio.as_mono().unwrap();
        let s: Vec<f64> = mono.iter().cloned().collect();
        let est = estimate_freq_over_window(&s, sr as f64);
        assert!(
            (est - f).abs() < f * 0.02,
            "degenerate exponential_chirp should be a {f} Hz tone, got {est}"
        );
    }

    #[test]
    fn test_square_wave_bandlimited() {
        let f = 1000.0;
        let sr = 44100u32;
        let audio = square_wave_bandlimited::<f64>(
            f,
            Duration::from_secs_f64(1.0),
            sample_rate!(44100),
            1.0,
        );
        assert_eq!(audio.samples_per_channel().get(), 44100);
        let mono = audio.as_mono().unwrap();
        let s: Vec<f64> = mono.iter().cloned().collect();

        // Fundamental frequency via zero-crossing rate ~ 2f crossings per second.
        let est = estimate_freq_over_window(&s, sr as f64);
        assert!(
            (est - f).abs() < f * 0.02,
            "expected fundamental ~{f} Hz, got {est}"
        );

        // Highest odd harmonic used must be strictly below Nyquist.
        let nyquist = sr as f64 / 2.0;
        let mut max_h = 1.0;
        let mut h = 1.0;
        while h * f < nyquist {
            max_h = h;
            h += 2.0;
        }
        assert!(max_h * f < nyquist, "max harmonic must be < Nyquist");
        // For 1000 Hz at 44100, highest odd harmonic < 22050 is 21*1000=21000.
        assert_eq!(max_h, 21.0);

        // Spot-check a sample against the closed-form harmonic sum.
        let idx = 137usize;
        let t = idx as f64 / sr as f64;
        let mut expected = 0.0;
        let mut n = 1.0;
        while n * f < nyquist {
            expected += (2.0 * std::f64::consts::PI * n * f * t).sin() / n;
            n += 2.0;
        }
        expected *= 4.0 / std::f64::consts::PI;
        assert_approx_eq!(s[idx], expected, 1e-9);
    }

    #[test]
    fn test_sawtooth_wave_bandlimited() {
        let f = 1000.0;
        let sr = 44100u32;
        let audio = sawtooth_wave_bandlimited::<f64>(
            f,
            Duration::from_secs_f64(1.0),
            sample_rate!(44100),
            1.0,
        );
        let mono = audio.as_mono().unwrap();
        let s: Vec<f64> = mono.iter().cloned().collect();

        let est = estimate_freq_over_window(&s, sr as f64);
        assert!(
            (est - f).abs() < f * 0.05,
            "expected fundamental ~{f} Hz, got {est}"
        );

        let nyquist = sr as f64 / 2.0;
        // Highest harmonic (all integers) used < Nyquist.
        let mut max_h = 1.0;
        let mut h = 1.0;
        while h * f < nyquist {
            max_h = h;
            h += 1.0;
        }
        assert!(max_h * f < nyquist);
        // 22*1000 = 22000 < 22050 for the saw (all harmonics).
        assert_eq!(max_h, 22.0);

        // Closed-form spot check.
        let idx = 251usize;
        let t = idx as f64 / sr as f64;
        let mut expected = 0.0;
        let mut k = 1.0;
        while k * f < nyquist {
            let sign = if (k as i64) % 2 == 1 { 1.0 } else { -1.0 };
            expected += sign * (2.0 * std::f64::consts::PI * k * f * t).sin() / k;
            k += 1.0;
        }
        expected *= 2.0 / std::f64::consts::PI;
        assert_approx_eq!(s[idx], expected, 1e-9);
    }

    #[test]
    fn test_triangle_wave_bandlimited() {
        let f = 1000.0;
        let sr = 44100u32;
        let audio = triangle_wave_bandlimited::<f64>(
            f,
            Duration::from_secs_f64(1.0),
            sample_rate!(44100),
            1.0,
        );
        let mono = audio.as_mono().unwrap();
        let s: Vec<f64> = mono.iter().cloned().collect();

        let est = estimate_freq_over_window(&s, sr as f64);
        assert!(
            (est - f).abs() < f * 0.02,
            "expected fundamental ~{f} Hz, got {est}"
        );

        let nyquist = sr as f64 / 2.0;
        // Triangle uses odd harmonics (2k+1); highest must be < Nyquist.
        let mut max_n = 1.0;
        let mut k: i64 = 0;
        loop {
            let n = (2 * k + 1) as f64;
            if n * f >= nyquist {
                break;
            }
            max_n = n;
            k += 1;
        }
        assert!(max_n * f < nyquist);
        assert_eq!(max_n, 21.0);

        // Closed-form spot check.
        let idx = 89usize;
        let t = idx as f64 / sr as f64;
        let mut expected = 0.0;
        let mut kk: i64 = 0;
        loop {
            let n = (2 * kk + 1) as f64;
            if n * f >= nyquist {
                break;
            }
            let sign = if kk % 2 == 0 { 1.0 } else { -1.0 };
            expected += sign * (2.0 * std::f64::consts::PI * n * f * t).sin() / (n * n);
            kk += 1;
        }
        expected *= 8.0 / (std::f64::consts::PI * std::f64::consts::PI);
        assert_approx_eq!(s[idx], expected, 1e-9);
    }

    #[cfg(feature = "transforms")]
    #[test]
    fn test_bandlimited_no_energy_above_nyquist() {
        use crate::operations::traits::AudioTransforms;
        use std::num::NonZeroUsize;
        // A band-limited oscillator must, by construction, contain no harmonic at
        // or above Nyquist. We verify the spectrum has negligible energy in the
        // top bins (which only fill if aliasing occurred).
        let f = 1000.0;
        let sr = 44100u32;
        // Choose signal length == n_fft (no zero-padding => no windowing leakage)
        // and a fundamental whose harmonics land exactly on FFT bins: with
        // n_fft = 4410 the bin spacing is 10 Hz, and 1000 Hz (+ all odd
        // multiples) lands exactly on a bin. Any energy in high bins would then
        // only come from aliasing, which a band-limited oscillator must not have.
        let n_fft = 4410usize;
        let audio = square_wave_bandlimited::<f64>(
            f,
            Duration::from_secs_f64(0.1),
            sample_rate!(44100),
            1.0,
        );
        assert_eq!(audio.samples_per_channel().get(), n_fft);
        let spectrum = audio.fft(NonZeroUsize::new(n_fft).unwrap()).unwrap();
        // One row per channel; mono => row 0.
        let mags: Vec<f64> = spectrum.row(0).iter().map(|c| c.norm()).collect();
        // Bins map linearly to [0, sr). Highest harmonic used is 21 kHz; bins
        // above ~21.5 kHz must be essentially empty.
        let bin_hz = sr as f64 / n_fft as f64;
        let nyquist_bin = n_fft / 2;
        let total: f64 = mags[..nyquist_bin].iter().sum();
        let high_cutoff_bin = (21500.0 / bin_hz) as usize;
        let high_energy: f64 = mags[high_cutoff_bin..nyquist_bin].iter().sum();
        assert!(
            high_energy < total * 1e-6,
            "band-limited square wave has unexpected high-frequency energy: {high_energy} of {total}"
        );
    }

    #[test]
    fn test_fm_signal_zero_index_is_sine() {
        let carrier = 440.0;
        let audio = fm_signal::<f64>(
            carrier,
            110.0,
            0.0, // no modulation
            Duration::from_secs_f64(0.05),
            sample_rate!(44100),
            0.8,
        );
        let reference = sine_wave::<f64>(
            carrier,
            Duration::from_secs_f64(0.05),
            sample_rate!(44100),
            0.8,
        );
        let a: Vec<f64> = audio.as_mono().unwrap().iter().cloned().collect();
        let b: Vec<f64> = reference.as_mono().unwrap().iter().cloned().collect();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_approx_eq!(*x, *y, 1e-12);
        }
    }

    #[test]
    fn test_fm_signal_peak_bounded() {
        let amplitude = 0.8;
        let audio = fm_signal::<f64>(
            440.0,
            110.0,
            5.0, // heavy modulation
            Duration::from_secs_f64(0.2),
            sample_rate!(44100),
            amplitude,
        );
        let mono = audio.as_mono().unwrap();
        let peak = mono.iter().cloned().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(
            peak <= amplitude + 1e-9,
            "FM peak {peak} should not exceed amplitude {amplitude}"
        );
        assert_eq!(audio.num_channels().get(), 1);
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
            components,
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
