//! Core traits for audio playback functionality.

use super::error::{PlaybackError, PlaybackResult};
use crate::{AudioSample, AudioSamples};
use std::time::Duration;

#[cfg(feature = "playback")]
use cpal::{StreamConfig, SupportedStreamConfig};

/// Represents an audio output device capability.
pub trait AudioDevice: Send + Sync {
    /// Get device information
    fn device_info(&self) -> DeviceInfo;

    /// Get supported audio formats
    fn supported_formats(&self) -> PlaybackResult<Vec<AudioFormatSpec>>;

    /// Get the default format for this device
    fn default_format(&self) -> PlaybackResult<AudioFormatSpec>;

    /// Check if a specific format is supported
    fn supports_format(&self, format: &AudioFormatSpec) -> bool;

    /// Get the device's preferred buffer size
    fn preferred_buffer_size(&self) -> Option<usize>;

    /// Get minimum and maximum supported buffer sizes
    fn buffer_size_range(&self) -> (usize, usize);

    /// Get the device's output latency
    fn output_latency(&self) -> Duration;

    /// Check if the device is currently available
    fn is_available(&self) -> bool;
}

/// Information about an audio device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub id: String,
    pub name: String,
    pub device_type: DeviceType,
    pub is_default: bool,
    pub max_channels: usize,
    pub supported_sample_rates: Vec<u32>,
    pub native_format: Option<AudioFormatSpec>,
}

/// Type of audio device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceType {
    /// Built-in speakers or headphones
    BuiltIn,
    /// USB audio device
    Usb,
    /// Bluetooth audio device
    Bluetooth,
    /// Professional audio interface
    AudioInterface,
    /// Virtual audio device
    Virtual,
    /// Unknown device type
    Unknown,
}

/// Specification for audio format.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioFormatSpec {
    pub sample_rate: u32,
    pub channels: usize,
    pub sample_format: SampleFormat,
    pub buffer_size: Option<usize>,
}

/// Audio sample format specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    I16,
    I32,
    F32,
    F64,
}

impl SampleFormat {
    /// Get the size in bytes for this sample format
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::I16 => 2,
            Self::I32 => 4,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// Check if this is a floating point format
    pub fn is_float(&self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }
}

impl AudioFormatSpec {
    /// Create a new format specification
    pub fn new(sample_rate: u32, channels: usize, sample_format: SampleFormat) -> Self {
        Self {
            sample_rate,
            channels,
            sample_format,
            buffer_size: None,
        }
    }

    /// Create CD-quality audio format (44.1kHz, stereo, 16-bit)
    pub fn cd_quality() -> Self {
        Self::new(44100, 2, SampleFormat::I16)
    }

    /// Create high-quality audio format (48kHz, stereo, 32-bit float)
    pub fn high_quality() -> Self {
        Self::new(48000, 2, SampleFormat::F32)
    }

    /// Set buffer size
    pub fn with_buffer_size(mut self, buffer_size: usize) -> Self {
        self.buffer_size = Some(buffer_size);
        self
    }

    /// Check compatibility with another format
    pub fn is_compatible(&self, other: &Self) -> bool {
        self.sample_rate == other.sample_rate
            && self.channels == other.channels
            && self.sample_format == other.sample_format
    }

    /// Get the frame size (all channels for one sample) in bytes
    pub fn frame_size_bytes(&self) -> usize {
        self.channels * self.sample_format.size_bytes()
    }
}

/// A sink that can receive and play audio data.
pub trait PlaybackSink<T: AudioSample>: Send + Sync {
    /// Write audio data to the sink
    async fn write(&mut self, audio: AudioSamples<T>) -> PlaybackResult<()>;

    /// Flush any pending audio data
    async fn flush(&mut self) -> PlaybackResult<()>;

    /// Get the current playback position
    fn position(&self) -> Duration;

    /// Get the amount of audio data currently buffered
    fn buffered_duration(&self) -> Duration;

    /// Check if the sink is currently playing
    fn is_playing(&self) -> bool;

    /// Get sink-specific metrics
    fn metrics(&self) -> PlaybackMetrics;
}

/// Controls playback transport (play, pause, stop, etc.).
pub trait PlaybackController: Send + Sync {
    /// Start or resume playback
    async fn play(&mut self) -> PlaybackResult<()>;

    /// Pause playback
    async fn pause(&mut self) -> PlaybackResult<()>;

    /// Stop playback and reset position
    async fn stop(&mut self) -> PlaybackResult<()>;

    /// Seek to a specific position
    async fn seek(&mut self, position: Duration) -> PlaybackResult<Duration>;

    /// Set playback volume (0.0 to 1.0)
    fn set_volume(&mut self, volume: f64) -> PlaybackResult<()>;

    /// Get current volume
    fn volume(&self) -> f64;

    /// Set whether playback should loop
    fn set_loop(&mut self, enable: bool);

    /// Check if looping is enabled
    fn is_loop_enabled(&self) -> bool;

    /// Get current playback state
    fn state(&self) -> PlaybackState;
}

/// Current state of playback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackState {
    /// Not initialized or stopped
    Idle,
    /// Currently playing
    Playing,
    /// Paused
    Paused,
    /// Stopped (can be resumed from current position)
    Stopped,
    /// Buffering data
    Buffering,
    /// Error state
    Error,
}

impl PlaybackState {
    /// Check if playback is active
    pub fn is_active(self) -> bool {
        matches!(self, Self::Playing | Self::Buffering)
    }

    /// Check if playback can be started/resumed
    pub fn can_play(self) -> bool {
        matches!(self, Self::Idle | Self::Paused | Self::Stopped)
    }

    /// Check if playback can be paused
    pub fn can_pause(self) -> bool {
        matches!(self, Self::Playing | Self::Buffering)
    }
}

/// Metrics for monitoring playback performance.
#[derive(Debug, Clone, Default)]
pub struct PlaybackMetrics {
    /// Total samples played
    pub samples_played: u64,

    /// Total bytes processed
    pub bytes_processed: u64,

    /// Number of buffer underruns
    pub underruns: u64,

    /// Number of buffer overruns
    pub overruns: u64,

    /// Current buffer fill level (0.0 to 1.0)
    pub buffer_level: f64,

    /// Average latency in milliseconds
    pub average_latency_ms: f64,

    /// Current volume level
    pub volume_level: f64,

    /// Sample rate being used
    pub sample_rate: u32,

    /// Number of channels being played
    pub channels: usize,
}

impl PlaybackMetrics {
    /// Calculate playback rate in samples per second
    pub fn playback_rate(&self, duration: Duration) -> f64 {
        if duration.as_secs_f64() > 0.0 {
            self.samples_played as f64 / duration.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Calculate data rate in bytes per second
    pub fn data_rate(&self, duration: Duration) -> f64 {
        if duration.as_secs_f64() > 0.0 {
            self.bytes_processed as f64 / duration.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get error rate based on underruns and overruns
    pub fn error_rate(&self) -> f64 {
        let total_operations = self.samples_played;
        if total_operations > 0 {
            (self.underruns + self.overruns) as f64 / total_operations as f64
        } else {
            0.0
        }
    }
}

/// Configuration for playback behavior.
#[derive(Debug, Clone)]
pub struct PlaybackConfig {
    /// Target audio format
    pub format: AudioFormatSpec,

    /// Buffer size in samples
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
        }
    }

    /// Set target format
    pub fn with_format(mut self, format: AudioFormatSpec) -> Self {
        self.format = format;
        self
    }

    /// Set buffer size
    pub fn with_buffer_size(mut self, buffer_size: usize) -> Self {
        self.buffer_size = buffer_size;
        self
    }

    /// Set maximum latency
    pub fn with_max_latency(mut self, latency_ms: u32) -> Self {
        self.max_latency_ms = latency_ms;
        self
    }

    /// Set initial volume
    pub fn with_volume(mut self, volume: f64) -> Self {
        self.volume = volume.clamp(0.0, 1.0);
        self
    }

    /// Set preferred device
    pub fn with_device(mut self, device_name: impl Into<String>) -> Self {
        self.preferred_device = Some(device_name.into());
        self
    }
}
