//! Signal generator source for streaming audio.

use crate::streaming::{
    error::{StreamError, StreamResult},
    traits::{AudioFormatInfo, AudioSource, SourceMetrics},
};
use crate::utils::generation::*; // Will use our existing generation utilities
use crate::{AudioSample, AudioSampleError, AudioSamples, ConvertTo};
use std::time::{Duration, Instant};

/// Types of signals that can be generated.
#[derive(Debug, Clone)]
pub enum SignalType {
    /// Pure sine wave
    Sine { frequency: f64 },
    /// Square wave
    Square { frequency: f64, duty_cycle: f64 },
    /// Sawtooth wave
    Sawtooth { frequency: f64 },
    /// Triangle wave
    Triangle { frequency: f64 },
    /// White noise
    WhiteNoise,
    /// Pink noise  
    PinkNoise,
    /// Chirp (frequency sweep)
    Chirp {
        start_freq: f64,
        end_freq: f64,
        duration: Duration,
    },
    /// Silence (zeros)
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
pub struct GeneratorSource<T: AudioSample> {
    config: GeneratorConfig,
    current_sample: usize,
    start_time: Instant,
    phase_accumulators: Vec<f64>,
    pink_noise_state: Option<PinkNoiseState>,
    metrics: SourceMetrics,
    is_active: bool,
    phantom: std::marker::PhantomData<T>,
}

impl<T: AudioSample> GeneratorSource<T> {
    /// Create a new signal generator with the given configuration.
    pub fn new(config: GeneratorConfig) -> Self {
        let mut phase_accumulators = vec![0.0; config.channels];
        let pink_noise_state = match config.signal_type {
            SignalType::PinkNoise => Some(PinkNoiseState::new()),
            _ => None,
        };

        Self {
            config,
            current_sample: 0,
            start_time: Instant::now(),
            phase_accumulators,
            pink_noise_state,
            metrics: SourceMetrics::default(),
            is_active: true,
            phantom: std::marker::PhantomData,
        }
    }

    /// Create a sine wave generator.
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

        // Reset state if needed
        if matches!(signal_type, SignalType::PinkNoise) && self.pink_noise_state.is_none() {
            self.pink_noise_state = Some(PinkNoiseState::new());
        }
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
                if let Some(ref mut state) = self.pink_noise_state {
                    state.next_sample()
                } else {
                    0.0
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

        raw_value * self.config.amplitude
    }
}

impl<T: AudioSample> AudioSource<T> for GeneratorSource<T> {
    async fn next_chunk(&mut self) -> StreamResult<Option<AudioSamples<T>>> {
        if !self.is_active || self.is_duration_exceeded() {
            self.is_active = false;
            return Ok(None);
        }

        let chunk_size = self.config.chunk_size;
        let channels = self.config.channels;

        // Generate samples
        let mut data = Vec::with_capacity(chunk_size * channels);

        for _ in 0..chunk_size {
            for channel in 0..channels {
                let sample_value = self.generate_sample(channel);

                // Convert f64 to target type T
                let converted_sample = match T::BITS {
                    16 => {
                        let i16_val = (sample_value.clamp(-1.0, 1.0) * i16::MAX as f64) as i16;
                        i16_val.convert_to::<T>().map_err(StreamError::Audio)?
                    }
                    32 => {
                        if T::BITS == 32 && std::any::type_name::<T>().contains("f32") {
                            // It's f32
                            let f32_val = sample_value as f32;
                            T::cast_from(f32_val as f32)
                        } else {
                            // It's i32
                            let i32_val = (sample_value.clamp(-1.0, 1.0) * i32::MAX as f64) as i32;
                            i32_val.convert_to::<T>().map_err(StreamError::Audio)?
                        }
                    }
                    64 => {
                        // f64
                        sample_value.convert_to::<T>().map_err(StreamError::Audio)?
                    }
                    _ => {
                        return Err(StreamError::InvalidConfig(format!(
                            "Unsupported sample type bit width: {}",
                            T::BITS
                        )));
                    }
                };

                data.push(converted_sample);
            }
            self.current_sample += 1;
        }

        // Create AudioSamples from generated data
        let array = ndarray::Array2::from_shape_vec((chunk_size, channels), data)
            .map_err(|e| StreamError::InvalidConfig(e.to_string()))?;

        let audio_samples =
            AudioSamples::new(array, self.config.sample_rate).map_err(StreamError::Audio)?;

        // Update metrics
        self.metrics.chunks_delivered += 1;
        self.metrics.bytes_delivered += (chunk_size * channels * std::mem::size_of::<T>()) as u64;
        self.metrics.average_chunk_size = (self.metrics.average_chunk_size + chunk_size) / 2;

        Ok(Some(audio_samples))
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

/// State for pink noise generation using the Voss algorithm.
struct PinkNoiseState {
    generators: [f64; 16],
    counter: usize,
}

impl PinkNoiseState {
    fn new() -> Self {
        Self {
            generators: [0.0; 16],
            counter: 0,
        }
    }

    fn next_sample(&mut self) -> f64 {
        // Simple pink noise implementation
        // This could be improved with a proper Voss algorithm

        let mut sum = 0.0;
        let mut mask = 1;

        for i in 0..16 {
            if (self.counter & mask) != 0 {
                // Update this generator with white noise
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                let mut hasher = DefaultHasher::new();
                (self.counter * (i + 1) * 12345).hash(&mut hasher);
                let hash = hasher.finish();
                self.generators[i] = (hash as f64 / u64::MAX as f64) * 2.0 - 1.0;
            }
            sum += self.generators[i];
            mask <<= 1;
        }

        self.counter = (self.counter + 1) % 65536;
        sum / 16.0 // Normalize
    }
}
