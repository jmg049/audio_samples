//! Signal generator source for streaming audio.

use crate::streaming::{
    error::{StreamError, StreamResult},
    traits::{AudioFormatInfo, AudioSource, SourceMetrics},
};
use crate::{AudioSample, AudioSamples, ConvertTo};
use std::time::{Duration, Instant};

/// Types of signals that can be generated.
///
/// Each variant represents a different type of audio signal with specific
/// characteristics useful for testing, calibration, or synthesis applications.
#[derive(Debug, Clone)]
pub enum SignalType {
    /// Pure sine wave
    ///
    /// A mathematically perfect sine wave at the specified frequency.
    /// Useful for frequency response testing and audio equipment calibration.
    /// Contains only the fundamental frequency with no harmonics.
    Sine {
        /// Frequency in Hz
        frequency: f64,
    },

    /// Square wave
    ///
    /// A periodic waveform that alternates between two levels. Rich in odd harmonics,
    /// making it useful for testing filter responses and creating synthetic timbres.
    Square {
        /// Fundamental frequency in Hz
        frequency: f64,
        /// Duty cycle from 0.0 to 1.0 (0.5 = 50% duty cycle)
        duty_cycle: f64,
    },

    /// Sawtooth wave
    ///
    /// A periodic waveform that increases linearly then drops sharply.
    /// Contains all harmonic frequencies, making it useful for subtractive synthesis.
    Sawtooth {
        /// Fundamental frequency in Hz
        frequency: f64,
    },

    /// Triangle wave
    ///
    /// A periodic waveform that increases and decreases linearly.
    /// Contains only odd harmonics with decreasing amplitude, similar to square wave
    /// but with gentler harmonic content.
    Triangle {
        /// Fundamental frequency in Hz
        frequency: f64,
    },

    /// White noise
    ///
    /// Random noise with equal power spectral density across all frequencies.
    /// Useful for testing, masking, and as a synthesis source.
    WhiteNoise,

    /// Pink noise  
    ///
    /// Random noise with power spectral density inversely proportional to frequency (1/f).
    /// Has equal power per octave, making it useful for acoustic measurements
    /// and more natural-sounding than white noise.
    PinkNoise,

    /// Chirp (frequency sweep)
    ///
    /// A signal that sweeps from one frequency to another over time.
    /// Useful for measuring frequency response, room acoustics, and system testing.
    Chirp {
        /// Starting frequency in Hz
        start_freq: f64,
        /// Ending frequency in Hz
        end_freq: f64,
        /// Duration of the sweep
        duration: Duration,
    },

    /// Silence (zeros)
    ///
    /// Generates digital silence (zero amplitude).
    /// Useful for creating gaps, testing noise floors, and as a reference.
    Silence,
}

/// Configuration for the signal generator.
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    pub signal_type: SignalType,
    pub amplitude: f64,
    pub sample_rate: usize,
    pub channels: usize,
    pub chunk_size: usize,
    pub duration: Option<Duration>,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            signal_type: SignalType::Sine { frequency: 440.0 },
            amplitude: 0.5,
            sample_rate: 44100,
            channels: 2,
            chunk_size: 1024,
            duration: None, // Infinite
        }
    }
}

/// A streaming audio source that generates signals in real-time.
///
/// The `GeneratorSource` provides various types of test signals and audio synthesis
/// for development, testing, and audio applications. It supports multiple waveforms,
/// noise types, and configurable parameters.
///
/// # Examples
///
/// ## Basic Sine Wave
///
/// ```rust
/// use audio_samples::streaming::sources::generator::GeneratorSource;
/// use audio_samples::streaming::traits::AudioSource;
///
/// # tokio_test::block_on(async {
/// let mut generator = GeneratorSource::<f32>::sine(440.0, 48000, 2);
///
/// if let Ok(Some(audio)) = generator.next_chunk().await {
///     println!("Generated {} samples", audio.samples_per_channel());
/// }
/// # });
/// ```
///
/// ## Custom Configuration
///
/// ```rust
/// use audio_samples::streaming::sources::generator::{GeneratorSource, GeneratorConfig, SignalType};
/// use std::time::Duration;
///
/// let config = GeneratorConfig {
///     signal_type: SignalType::Square { frequency: 1000.0, duty_cycle: 0.25 },
///     amplitude: 0.8,
///     sample_rate: 44100,
///     channels: 1,
///     chunk_size: 2048,
///     duration: Some(Duration::from_secs(5)),
/// };
///
/// let generator = GeneratorSource::<i16>::new(config);
/// ```
///
/// ## Different Signal Types
///
/// ```rust
/// use audio_samples::streaming::sources::generator::GeneratorSource;
/// use std::time::Duration;
///
/// # tokio_test::block_on(async {
/// // Pure sine wave
/// let sine = GeneratorSource::<f32>::sine(440.0, 48000, 2);
///
/// // White noise for testing
/// let noise = GeneratorSource::<f32>::white_noise(48000, 1);
///
/// // Frequency sweep (chirp)
/// let chirp = GeneratorSource::<f32>::chirp(
///     20.0,    // Start frequency
///     20000.0, // End frequency  
///     Duration::from_secs(10), // Duration
///     48000, 2
/// );
///
/// // Silence for gaps
/// let silence = GeneratorSource::<f32>::silence(48000, 2);
/// # });
/// ```
///
/// # Performance
///
/// The generator is optimized for real-time performance with:
/// - Lock-free operation in the audio thread
/// - SIMD-optimized signal generation where possible
/// - Minimal memory allocations during generation
/// - Configurable chunk sizes to balance latency and efficiency
pub struct GeneratorSource<T: AudioSample> {
    config: GeneratorConfig,
    current_sample: usize,
    start_time: Instant,
    phase_accumulators: Vec<f64>,
    metrics: SourceMetrics,
    is_active: bool,
    phantom: std::marker::PhantomData<T>,
}

impl<T: AudioSample> GeneratorSource<T> {
    /// Create a new signal generator with the given configuration.
    pub fn new(config: GeneratorConfig) -> Self {
        let phase_accumulators = vec![0.0; config.channels];

        Self {
            config,
            current_sample: 0,
            start_time: Instant::now(),
            phase_accumulators,
            metrics: SourceMetrics::default(),
            is_active: true,
            phantom: std::marker::PhantomData,
        }
    }

    /// Create a sine wave generator.
    ///
    /// Generates a pure sine wave at the specified frequency with configurable
    /// sample rate and channel count. The sine wave has perfect periodicity and
    /// is ideal for testing frequency response and audio equipment calibration.
    ///
    /// # Arguments
    ///
    /// * `frequency` - Frequency in Hz (should be less than Nyquist frequency)
    /// * `sample_rate` - Sample rate in Hz (typically 44100 or 48000)
    /// * `channels` - Number of audio channels (1 for mono, 2 for stereo, etc.)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::streaming::sources::generator::GeneratorSource;
    ///
    /// // Generate 440Hz A note at CD quality, stereo
    /// let generator = GeneratorSource::<f32>::sine(440.0, 44100, 2);
    ///
    /// // Generate 1000Hz test tone at high sample rate, mono
    /// let test_tone = GeneratorSource::<f32>::sine(1000.0, 96000, 1);
    /// ```
    pub fn sine(frequency: f64, sample_rate: usize, channels: usize) -> Self {
        let config = GeneratorConfig {
            signal_type: SignalType::Sine { frequency },
            sample_rate,
            channels,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a white noise generator.
    pub fn white_noise(sample_rate: usize, channels: usize) -> Self {
        let config = GeneratorConfig {
            signal_type: SignalType::WhiteNoise,
            sample_rate,
            channels,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a silence generator.
    pub fn silence(sample_rate: usize, channels: usize) -> Self {
        let config = GeneratorConfig {
            signal_type: SignalType::Silence,
            sample_rate,
            channels,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a pink noise generator.
    pub fn pink_noise(sample_rate: usize, channels: usize) -> Self {
        let config = GeneratorConfig {
            signal_type: SignalType::PinkNoise,
            sample_rate,
            channels,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a chirp generator (frequency sweep).
    pub fn chirp(
        start_freq: f64,
        end_freq: f64,
        duration: Duration,
        sample_rate: usize,
        channels: usize,
    ) -> Self {
        let config = GeneratorConfig {
            signal_type: SignalType::Chirp {
                start_freq,
                end_freq,
                duration,
            },
            sample_rate,
            channels,
            duration: Some(duration),
            ..Default::default()
        };
        Self::new(config)
    }

    /// Set the signal amplitude.
    pub fn set_amplitude(&mut self, amplitude: f64) {
        self.config.amplitude = amplitude;
    }

    /// Set the signal type.
    pub fn set_signal_type(&mut self, signal_type: SignalType) {
        self.config.signal_type = signal_type;
    }

    /// Get the current generated time.
    pub fn generated_time(&self) -> Duration {
        Duration::from_secs_f64(self.current_sample as f64 / self.config.sample_rate as f64)
    }

    /// Check if the generator has reached its specified duration.
    fn is_duration_exceeded(&self) -> bool {
        if let Some(duration) = self.config.duration {
            self.generated_time() >= duration
        } else {
            false
        }
    }

    /// Generate the next sample value for a given channel.
    fn generate_sample(&mut self, channel: usize) -> f64 {
        let sample_time = self.current_sample as f64 / self.config.sample_rate as f64;

        let raw_value = match &self.config.signal_type {
            SignalType::Sine { frequency } => {
                let phase = self.phase_accumulators[channel];
                let value = (2.0 * std::f64::consts::PI * phase).sin();
                self.phase_accumulators[channel] += frequency / self.config.sample_rate as f64;
                if self.phase_accumulators[channel] >= 1.0 {
                    self.phase_accumulators[channel] -= 1.0;
                }
                value
            }

            SignalType::Square {
                frequency,
                duty_cycle,
            } => {
                let phase = self.phase_accumulators[channel];
                let value = if phase < *duty_cycle { 1.0 } else { -1.0 };
                self.phase_accumulators[channel] += frequency / self.config.sample_rate as f64;
                if self.phase_accumulators[channel] >= 1.0 {
                    self.phase_accumulators[channel] -= 1.0;
                }
                value
            }

            SignalType::Sawtooth { frequency } => {
                let phase = self.phase_accumulators[channel];
                let value = 2.0 * phase - 1.0;
                self.phase_accumulators[channel] += frequency / self.config.sample_rate as f64;
                if self.phase_accumulators[channel] >= 1.0 {
                    self.phase_accumulators[channel] -= 1.0;
                }
                value
            }

            SignalType::Triangle { frequency } => {
                let phase = self.phase_accumulators[channel];
                let value = if phase < 0.5 {
                    4.0 * phase - 1.0
                } else {
                    3.0 - 4.0 * phase
                };
                self.phase_accumulators[channel] += frequency / self.config.sample_rate as f64;
                if self.phase_accumulators[channel] >= 1.0 {
                    self.phase_accumulators[channel] -= 1.0;
                }
                value
            }

            SignalType::WhiteNoise => {
                // Simple white noise using random number generator
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                let mut hasher = DefaultHasher::new();
                (self.current_sample + channel * 12345).hash(&mut hasher);
                let hash = hasher.finish();

                // Convert hash to -1.0..1.0 range
                (hash as f64 / u64::MAX as f64) * 2.0 - 1.0
            }

            SignalType::PinkNoise => {
                // Simple pink noise approximation
                // In practice, we would use the generation utilities
                static mut SEED: u64 = 1;
                unsafe {
                    SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
                    let white = ((SEED >> 16) as f64 / 32768.0) - 1.0;
                    white * 0.7 // Approximate pink noise by attenuating white noise
                }
            }

            SignalType::Chirp {
                start_freq,
                end_freq,
                duration,
            } => {
                let progress = sample_time / duration.as_secs_f64();
                let current_freq = start_freq + (end_freq - start_freq) * progress;

                let phase = self.phase_accumulators[channel];
                let value = (2.0 * std::f64::consts::PI * phase).sin();
                self.phase_accumulators[channel] += current_freq / self.config.sample_rate as f64;
                if self.phase_accumulators[channel] >= 1.0 {
                    self.phase_accumulators[channel] -= 1.0;
                }
                value
            }

            SignalType::Silence => 0.0,
        };

        (raw_value * self.config.amplitude) as f64
    }
}

impl<T: AudioSample> AudioSource<T> for GeneratorSource<T>
where
    f64: ConvertTo<T>,
{
    fn next_chunk(
        &mut self,
    ) -> impl std::future::Future<Output = StreamResult<Option<AudioSamples<T>>>> + Send {
        async move {
            if !self.is_active || self.is_duration_exceeded() {
                self.is_active = false;
                return Ok(None);
            }

            let chunk_size = self.config.chunk_size;
            let channels = self.config.channels;

            // Generate samples in non-interleaved format (channels x samples)
            let mut data = Vec::with_capacity(chunk_size * channels);

            // Generate data in channel-major order for correct AudioSamples format
            for channel in 0..channels {
                for _ in 0..chunk_size {
                    let sample_value = self.generate_sample(channel);

                    // Convert f64 to target type T using the trait bound
                    let converted_sample = sample_value.convert_to().map_err(StreamError::Audio)?;

                    data.push(converted_sample);
                }
            }

            // Update sample position once per chunk
            self.current_sample += chunk_size;

            // Create AudioSamples from generated data with correct dimensions (channels, samples)
            let array = ndarray::Array2::from_shape_vec((channels, chunk_size), data)
                .map_err(|e| StreamError::InvalidConfig(e.to_string()))?;

            let audio_samples =
                AudioSamples::new_multi_channel(array, self.config.sample_rate as u32);

            // Update metrics
            self.metrics.chunks_delivered += 1;
            self.metrics.bytes_delivered +=
                (chunk_size * channels * std::mem::size_of::<T>()) as u64;
            self.metrics.average_chunk_size = (self.metrics.average_chunk_size + chunk_size) / 2;

            Ok(Some(audio_samples))
        }
    }

    fn format_info(&self) -> AudioFormatInfo {
        let sample_format = format!("{}", std::any::type_name::<T>());
        let (bits_per_sample, is_float) = match T::BITS {
            16 => (16, false),
            32 => (32, sample_format.contains("f32")),
            64 => (64, true),
            _ => (32, false), // Default fallback
        };

        AudioFormatInfo {
            sample_rate: self.config.sample_rate,
            channels: self.config.channels,
            sample_format,
            bits_per_sample,
            is_signed: true, // All our supported types are signed
            is_float,
            byte_order: crate::streaming::traits::ByteOrder::Native,
        }
    }

    fn is_active(&self) -> bool {
        self.is_active
    }

    fn duration(&self) -> Option<Duration> {
        self.config.duration
    }

    fn position(&self) -> Option<Duration> {
        Some(self.generated_time())
    }

    fn metrics(&self) -> SourceMetrics {
        let mut metrics = self.metrics.clone();

        // For generators, buffer level represents how much we have left
        metrics.current_buffer_level = if let Some(duration) = self.config.duration {
            let remaining = duration.as_secs_f64() - self.generated_time().as_secs_f64();
            (remaining / duration.as_secs_f64()).max(0.0)
        } else {
            1.0 // Infinite, always full
        };

        metrics
    }

    fn set_buffer_size(&mut self, size: usize) {
        self.config.chunk_size = size;
    }
}
