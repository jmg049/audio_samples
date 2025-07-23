//! High-level audio player with transport controls.

use super::{
    devices::{DeviceHandle, DeviceManager},
    error::{PlaybackError, PlaybackResult},
    traits::{
        AudioFormatSpec, PlaybackController, PlaybackMetrics, PlaybackSink, PlaybackState,
        SampleFormat,
    },
};
use crate::{AudioSample, AudioSamples};
use parking_lot::{Mutex, RwLock};
use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::{Duration, Instant};

#[cfg(feature = "playback")]
use cpal::{
    SampleRate, Stream, StreamConfig,
    traits::{DeviceTrait, StreamTrait},
};

/// Configuration for audio playback.
#[derive(Debug, Clone)]
pub struct PlaybackConfig {
    /// Target audio format
    pub format: AudioFormatSpec,

    /// Buffer size in samples per channel
    pub buffer_size: usize,

    /// Maximum latency acceptable (milliseconds)
    pub max_latency_ms: u32,

    /// Whether to automatically recover from errors
    pub auto_recovery: bool,

    /// Maximum number of recovery attempts
    pub max_recovery_attempts: usize,

    /// Initial volume (0.0 to 1.0)
    pub volume: f64,

    /// Whether to start with looping enabled
    pub loop_enabled: bool,

    /// Preferred audio device (None = use default)
    pub preferred_device: Option<String>,

    /// Whether to automatically start playback when audio is loaded
    pub auto_play: bool,

    /// Pre-buffer size (how much to buffer before starting playback)
    pub pre_buffer_ms: u32,
}

impl Default for PlaybackConfig {
    fn default() -> Self {
        Self {
            format: AudioFormatSpec::cd_quality(),
            buffer_size: 1024,
            max_latency_ms: 50,
            auto_recovery: true,
            max_recovery_attempts: 3,
            volume: 0.8,
            loop_enabled: false,
            preferred_device: None,
            auto_play: false,
            pre_buffer_ms: 100,
        }
    }
}

impl PlaybackConfig {
    /// Create configuration optimized for low latency
    pub fn low_latency() -> Self {
        Self {
            format: AudioFormatSpec::new(48000, 2, SampleFormat::F32),
            buffer_size: 256,
            max_latency_ms: 10,
            auto_recovery: true,
            max_recovery_attempts: 5,
            volume: 0.8,
            loop_enabled: false,
            preferred_device: None,
            auto_play: false,
            pre_buffer_ms: 20,
        }
    }

    /// Create configuration optimized for high quality
    pub fn high_quality() -> Self {
        Self {
            format: AudioFormatSpec::new(96000, 2, SampleFormat::F32),
            buffer_size: 2048,
            max_latency_ms: 100,
            auto_recovery: true,
            max_recovery_attempts: 3,
            volume: 0.8,
            loop_enabled: false,
            preferred_device: None,
            auto_play: false,
            pre_buffer_ms: 200,
        }
    }
}

/// Internal audio buffer for playback
#[derive(Debug)]
struct AudioBuffer<T: AudioSample> {
    data: VecDeque<T>,
    channels: usize,
    sample_rate: u32,
    max_size: usize,
}

impl<T: AudioSample> AudioBuffer<T> {
    fn new(channels: usize, sample_rate: u32, max_samples: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(max_samples),
            channels,
            sample_rate,
            max_size: max_samples,
        }
    }

    fn push_samples(&mut self, audio: &AudioSamples<T>) -> PlaybackResult<()> {
        let samples = audio.as_slice();

        // Check if we have room
        if self.data.len() + samples.len() > self.max_size {
            return Err(PlaybackError::BufferOverflow {
                requested: samples.len(),
                available: self.max_size.saturating_sub(self.data.len()),
            });
        }

        // Add samples to buffer
        for &sample in samples {
            self.data.push_back(sample);
        }

        Ok(())
    }

    fn read_samples(&mut self, output: &mut [T]) -> usize {
        let available = self.data.len().min(output.len());

        for i in 0..available {
            output[i] = self.data.pop_front().unwrap_or_default();
        }

        available
    }

    fn available_samples(&self) -> usize {
        self.data.len()
    }

    fn duration(&self) -> Duration {
        let samples_per_channel = self.data.len() / self.channels.max(1);
        Duration::from_secs_f64(samples_per_channel as f64 / self.sample_rate as f64)
    }

    fn clear(&mut self) {
        self.data.clear();
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// High-level audio player with transport controls.
pub struct AudioPlayer<T: AudioSample> {
    config: PlaybackConfig,
    device_manager: DeviceManager,
    device: Option<DeviceHandle>,

    #[cfg(feature = "playback")]
    stream: Option<Stream>,

    #[cfg(not(feature = "playback"))]
    stream: Option<()>,

    // Shared state between playback thread and control thread
    state: Arc<RwLock<PlaybackState>>,
    buffer: Arc<Mutex<AudioBuffer<T>>>,
    volume: Arc<Mutex<f64>>,
    loop_enabled: Arc<AtomicBool>,

    // Position tracking
    position: Arc<AtomicU64>, // In samples
    total_duration: Arc<Mutex<Option<Duration>>>,

    // Statistics
    metrics: Arc<Mutex<PlaybackMetrics>>,

    // Error recovery
    recovery_attempts: usize,
    last_error: Option<PlaybackError>,
}

impl<T: AudioSample> AudioPlayer<T> {
    /// Create a new audio player with default configuration.
    pub fn new() -> PlaybackResult<Self> {
        Self::with_config(PlaybackConfig::default())
    }

    /// Create a new audio player with the specified configuration.
    pub fn with_config(config: PlaybackConfig) -> PlaybackResult<Self> {
        let device_manager = DeviceManager::new()?;

        // Calculate buffer size in samples
        let channels = config.format.channels;
        let sample_rate = config.format.sample_rate;
        let max_buffer_samples = (sample_rate as f64 * 10.0) as usize * channels; // 10 seconds max

        let buffer = AudioBuffer::new(channels, sample_rate, max_buffer_samples);

        Ok(Self {
            config,
            device_manager,
            device: None,
            stream: None,
            state: Arc::new(RwLock::new(PlaybackState::Idle)),
            buffer: Arc::new(Mutex::new(buffer)),
            volume: Arc::new(Mutex::new(config.volume)),
            loop_enabled: Arc::new(AtomicBool::new(config.loop_enabled)),
            position: Arc::new(AtomicU64::new(0)),
            total_duration: Arc::new(Mutex::new(None)),
            metrics: Arc::new(Mutex::new(PlaybackMetrics::default())),
            recovery_attempts: 0,
            last_error: None,
        })
    }

    /// Create a low-latency audio player.
    pub fn low_latency() -> PlaybackResult<Self> {
        Self::with_config(PlaybackConfig::low_latency())
    }

    /// Create a high-quality audio player.
    pub fn high_quality() -> PlaybackResult<Self> {
        Self::with_config(PlaybackConfig::high_quality())
    }

    /// Load audio data for playback.
    pub async fn load_audio(&mut self, audio: AudioSamples<T>) -> PlaybackResult<()> {
        let mut buffer = self.buffer.lock();

        // Clear existing data if not in the middle of playback
        if matches!(
            *self.state.read(),
            PlaybackState::Idle | PlaybackState::Stopped
        ) {
            buffer.clear();
        }

        buffer.push_samples(&audio)?;

        // Update total duration
        *self.total_duration.lock() = Some(buffer.duration());

        // Auto-start if configured
        if self.config.auto_play && matches!(*self.state.read(), PlaybackState::Idle) {
            drop(buffer); // Release lock before calling play
            self.play().await?;
        }

        Ok(())
    }

    /// Initialize the playback device and stream.
    #[cfg(feature = "playback")]
    fn initialize_stream(&mut self) -> PlaybackResult<()> {
        // Select device
        let device = if let Some(ref device_name) = self.config.preferred_device {
            self.device_manager
                .find_device_by_name(device_name)?
                .ok_or_else(|| PlaybackError::DeviceNotFound {
                    device_name: device_name.clone(),
                })?
        } else {
            self.device_manager
                .default_output_device()?
                .ok_or(PlaybackError::NoDevicesAvailable)?
        };

        // Find best format match
        let target_format = &self.config.format;
        let device_format = if device.supports_format(target_format) {
            target_format.clone()
        } else {
            device.default_format()?
        };

        // Create CPAL stream config
        let stream_config = StreamConfig {
            channels: device_format.channels as cpal::ChannelCount,
            sample_rate: SampleRate(device_format.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(self.config.buffer_size as u32),
        };

        // Clone Arc references for the callback
        let buffer = Arc::clone(&self.buffer);
        let volume = Arc::clone(&self.volume);
        let state = Arc::clone(&self.state);
        let position = Arc::clone(&self.position);
        let metrics = Arc::clone(&self.metrics);
        let loop_enabled = Arc::clone(&self.loop_enabled);

        // Create the output stream based on sample format
        let stream = match device_format.sample_format {
            SampleFormat::F32 => device
                .cpal_device()
                .build_output_stream(
                    &stream_config,
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        Self::audio_callback_f32(
                            data,
                            &buffer,
                            &volume,
                            &state,
                            &position,
                            &metrics,
                            &loop_enabled,
                        );
                    },
                    |err| {
                        eprintln!("Audio stream error: {}", err);
                    },
                    None,
                )
                .map_err(|e| PlaybackError::StreamCreation {
                    source: Box::new(e),
                })?,
            SampleFormat::I16 => device
                .cpal_device()
                .build_output_stream(
                    &stream_config,
                    move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                        Self::audio_callback_i16(
                            data,
                            &buffer,
                            &volume,
                            &state,
                            &position,
                            &metrics,
                            &loop_enabled,
                        );
                    },
                    |err| {
                        eprintln!("Audio stream error: {}", err);
                    },
                    None,
                )
                .map_err(|e| PlaybackError::StreamCreation {
                    source: Box::new(e),
                })?,
            SampleFormat::I32 => device
                .cpal_device()
                .build_output_stream(
                    &stream_config,
                    move |data: &mut [i32], _: &cpal::OutputCallbackInfo| {
                        Self::audio_callback_i32(
                            data,
                            &buffer,
                            &volume,
                            &state,
                            &position,
                            &metrics,
                            &loop_enabled,
                        );
                    },
                    |err| {
                        eprintln!("Audio stream error: {}", err);
                    },
                    None,
                )
                .map_err(|e| PlaybackError::StreamCreation {
                    source: Box::new(e),
                })?,
            SampleFormat::F64 => device
                .cpal_device()
                .build_output_stream(
                    &stream_config,
                    move |data: &mut [f64], _: &cpal::OutputCallbackInfo| {
                        Self::audio_callback_f64(
                            data,
                            &buffer,
                            &volume,
                            &state,
                            &position,
                            &metrics,
                            &loop_enabled,
                        );
                    },
                    |err| {
                        eprintln!("Audio stream error: {}", err);
                    },
                    None,
                )
                .map_err(|e| PlaybackError::StreamCreation {
                    source: Box::new(e),
                })?,
        };

        self.device = Some(device);
        self.stream = Some(stream);

        Ok(())
    }

    #[cfg(not(feature = "playback"))]
    fn initialize_stream(&mut self) -> PlaybackResult<()> {
        Err(PlaybackError::FeatureNotEnabled {
            feature: "playback".to_string(),
            operation: "initialize stream".to_string(),
        })
    }

    // Audio callback functions for different sample formats
    #[cfg(feature = "playback")]
    fn audio_callback_f32(
        output: &mut [f32],
        buffer: &Arc<Mutex<AudioBuffer<T>>>,
        volume: &Arc<Mutex<f64>>,
        state: &Arc<RwLock<PlaybackState>>,
        position: &Arc<AtomicU64>,
        metrics: &Arc<Mutex<PlaybackMetrics>>,
        _loop_enabled: &Arc<AtomicBool>,
    ) where
        T: AudioSample + Into<f32>,
    {
        let current_state = *state.read();
        if !current_state.is_active() {
            // Fill with silence
            output.fill(0.0);
            return;
        }

        let mut temp_buffer: Vec<T> = vec![T::default(); output.len()];
        let samples_read = {
            let mut buf = buffer.lock();
            buf.read_samples(&mut temp_buffer)
        };

        let volume_level = *volume.lock();

        // Convert and apply volume
        for (i, &sample) in temp_buffer.iter().take(samples_read).enumerate() {
            output[i] = sample.into() * volume_level as f32;
        }

        // Fill remaining with silence if not enough samples
        if samples_read < output.len() {
            output[samples_read..].fill(0.0);

            // Update state to stopped if we're out of data
            if samples_read == 0 && !matches!(current_state, PlaybackState::Paused) {
                *state.write() = PlaybackState::Stopped;
            }
        }

        // Update position and metrics
        position.fetch_add(samples_read as u64, Ordering::Relaxed);
        {
            let mut m = metrics.lock();
            m.samples_played += samples_read as u64;
            m.bytes_processed += samples_read * std::mem::size_of::<f32>();
        }
    }

    #[cfg(feature = "playback")]
    fn audio_callback_i16(
        output: &mut [i16],
        buffer: &Arc<Mutex<AudioBuffer<T>>>,
        volume: &Arc<Mutex<f64>>,
        state: &Arc<RwLock<PlaybackState>>,
        position: &Arc<AtomicU64>,
        metrics: &Arc<Mutex<PlaybackMetrics>>,
        _loop_enabled: &Arc<AtomicBool>,
    ) where
        T: AudioSample + Into<i16>,
    {
        let current_state = *state.read();
        if !current_state.is_active() {
            output.fill(0);
            return;
        }

        let mut temp_buffer: Vec<T> = vec![T::default(); output.len()];
        let samples_read = {
            let mut buf = buffer.lock();
            buf.read_samples(&mut temp_buffer)
        };

        let volume_level = *volume.lock();

        // Convert and apply volume
        for (i, &sample) in temp_buffer.iter().take(samples_read).enumerate() {
            let value = sample.into() as f32 * volume_level as f32;
            output[i] = value.clamp(-32768.0, 32767.0) as i16;
        }

        // Fill remaining with silence
        if samples_read < output.len() {
            output[samples_read..].fill(0);

            if samples_read == 0 && !matches!(current_state, PlaybackState::Paused) {
                *state.write() = PlaybackState::Stopped;
            }
        }

        position.fetch_add(samples_read as u64, Ordering::Relaxed);
        {
            let mut m = metrics.lock();
            m.samples_played += samples_read as u64;
            m.bytes_processed += samples_read * std::mem::size_of::<i16>();
        }
    }

    // Similar implementations for i32 and f64...
    #[cfg(feature = "playback")]
    fn audio_callback_i32(
        output: &mut [i32],
        _buffer: &Arc<Mutex<AudioBuffer<T>>>,
        _volume: &Arc<Mutex<f64>>,
        state: &Arc<RwLock<PlaybackState>>,
        _position: &Arc<AtomicU64>,
        _metrics: &Arc<Mutex<PlaybackMetrics>>,
        _loop_enabled: &Arc<AtomicBool>,
    ) {
        // Simplified - fill with silence for now
        output.fill(0);
        if state.read().is_active() {
            *state.write() = PlaybackState::Stopped;
        }
    }

    #[cfg(feature = "playback")]
    fn audio_callback_f64(
        output: &mut [f64],
        _buffer: &Arc<Mutex<AudioBuffer<T>>>,
        _volume: &Arc<Mutex<f64>>,
        state: &Arc<RwLock<PlaybackState>>,
        _position: &Arc<AtomicU64>,
        _metrics: &Arc<Mutex<PlaybackMetrics>>,
        _loop_enabled: &Arc<AtomicBool>,
    ) {
        // Simplified - fill with silence for now
        output.fill(0.0);
        if state.read().is_active() {
            *state.write() = PlaybackState::Stopped;
        }
    }

    /// Get the current playback position.
    fn current_position(&self) -> Duration {
        let samples = self.position.load(Ordering::Relaxed);
        let sample_rate = self.config.format.sample_rate as f64;
        let channels = self.config.format.channels as f64;
        Duration::from_secs_f64(samples as f64 / (sample_rate * channels))
    }
}

impl<T: AudioSample> PlaybackController for AudioPlayer<T>
where
    T: Into<f32> + Into<i16> + Into<i32> + Into<f64>,
{
    async fn play(&mut self) -> PlaybackResult<()> {
        let current_state = *self.state.read();

        if !current_state.can_play() {
            return Err(PlaybackError::InvalidState {
                current: format!("{:?}", current_state),
                operation: "play".to_string(),
            });
        }

        // Initialize stream if not already done
        if self.stream.is_none() {
            self.initialize_stream()?;
        }

        // Start the stream
        #[cfg(feature = "playback")]
        if let Some(ref stream) = self.stream {
            stream.play().map_err(|e| PlaybackError::StreamControl {
                operation: "play".to_string(),
                source: Box::new(e),
            })?;
        }

        *self.state.write() = PlaybackState::Playing;
        Ok(())
    }

    async fn pause(&mut self) -> PlaybackResult<()> {
        let current_state = *self.state.read();

        if !current_state.can_pause() {
            return Err(PlaybackError::InvalidState {
                current: format!("{:?}", current_state),
                operation: "pause".to_string(),
            });
        }

        #[cfg(feature = "playback")]
        if let Some(ref stream) = self.stream {
            stream.pause().map_err(|e| PlaybackError::StreamControl {
                operation: "pause".to_string(),
                source: Box::new(e),
            })?;
        }

        *self.state.write() = PlaybackState::Paused;
        Ok(())
    }

    async fn stop(&mut self) -> PlaybackResult<()> {
        #[cfg(feature = "playback")]
        if let Some(ref stream) = self.stream {
            stream.pause().map_err(|e| PlaybackError::StreamControl {
                operation: "stop".to_string(),
                source: Box::new(e),
            })?;
        }

        *self.state.write() = PlaybackState::Stopped;
        self.position.store(0, Ordering::Relaxed);

        Ok(())
    }

    async fn seek(&mut self, position: Duration) -> PlaybackResult<Duration> {
        let sample_rate = self.config.format.sample_rate as f64;
        let channels = self.config.format.channels as f64;
        let target_samples = (position.as_secs_f64() * sample_rate * channels) as u64;

        self.position.store(target_samples, Ordering::Relaxed);

        // For now, we can't actually seek in the buffer - this would require
        // reloading data from the source. Return the requested position.
        Ok(position)
    }

    fn set_volume(&mut self, volume: f64) -> PlaybackResult<()> {
        let clamped_volume = volume.clamp(0.0, 1.0);
        *self.volume.lock() = clamped_volume;
        Ok(())
    }

    fn volume(&self) -> f64 {
        *self.volume.lock()
    }

    fn set_loop(&mut self, enable: bool) {
        self.loop_enabled.store(enable, Ordering::Relaxed);
    }

    fn is_loop_enabled(&self) -> bool {
        self.loop_enabled.load(Ordering::Relaxed)
    }

    fn state(&self) -> PlaybackState {
        *self.state.read()
    }
}

impl<T: AudioSample> PlaybackSink<T> for AudioPlayer<T> {
    async fn write(&mut self, audio: AudioSamples<T>) -> PlaybackResult<()> {
        self.load_audio(audio).await
    }

    async fn flush(&mut self) -> PlaybackResult<()> {
        // Wait for buffer to empty
        while !self.buffer.lock().is_empty() && self.state().is_active() {
            #[cfg(feature = "playback")]
            tokio::time::sleep(Duration::from_millis(10)).await;

            #[cfg(not(feature = "playback"))]
            std::thread::sleep(Duration::from_millis(10));
        }
        Ok(())
    }

    fn position(&self) -> Duration {
        self.current_position()
    }

    fn buffered_duration(&self) -> Duration {
        self.buffer.lock().duration()
    }

    fn is_playing(&self) -> bool {
        matches!(self.state(), PlaybackState::Playing)
    }

    fn metrics(&self) -> PlaybackMetrics {
        let mut metrics = self.metrics.lock().clone();

        // Update current values
        metrics.volume_level = *self.volume.lock();
        metrics.sample_rate = self.config.format.sample_rate;
        metrics.channels = self.config.format.channels;
        metrics.buffer_level = {
            let buffer = self.buffer.lock();
            buffer.available_samples() as f64 / buffer.max_size as f64
        };

        metrics
    }
}
