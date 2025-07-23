//! Core traits for streaming audio processing.

use super::error::{StreamError, StreamResult};
use crate::{AudioSample, AudioSampleResult, AudioSamples};
use std::time::Duration;

#[cfg(feature = "streaming")]
use futures::Stream;

#[cfg(feature = "streaming")]
use tokio::sync::mpsc;

/// Represents an audio source that can provide streaming audio data.
///
/// Audio sources are responsible for providing audio chunks in real-time,
/// whether from network streams, files, or generated content.
pub trait AudioSource<T: AudioSample>: Send + Sync {
    /// Get the next chunk of audio data.
    ///
    /// Returns `None` when the stream ends naturally, or an error
    /// if something goes wrong that could potentially be recovered from.
    async fn next_chunk(&mut self) -> StreamResult<Option<AudioSamples<T>>>;

    /// Get information about the audio format this source provides.
    fn format_info(&self) -> AudioFormatInfo;

    /// Check if the source is still active/available.
    fn is_active(&self) -> bool;

    /// Attempt to seek to a specific position (if supported).
    ///
    /// Returns `Ok(actual_position)` if seeking succeeded, where
    /// `actual_position` may differ from the requested position.
    /// Returns an error if seeking is not supported or fails.
    async fn seek(&mut self, position: Duration) -> StreamResult<Duration> {
        let _ = position;
        Err(StreamError::InvalidConfig(
            "Seeking not supported".to_string(),
        ))
    }

    /// Get the estimated duration of the stream (if known).
    fn duration(&self) -> Option<Duration> {
        None
    }

    /// Get the current position in the stream (if supported).
    fn position(&self) -> Option<Duration> {
        None
    }

    /// Get streaming metrics for this source.
    fn metrics(&self) -> SourceMetrics {
        SourceMetrics::default()
    }

    /// Configure buffer size hint for optimal performance.
    fn set_buffer_size(&mut self, size: usize) {
        let _ = size; // Default implementation ignores buffer size
    }
}

/// Information about audio format and stream characteristics.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioFormatInfo {
    pub sample_rate: usize,
    pub channels: usize,
    pub sample_format: String,
    pub bits_per_sample: u8,
    pub is_signed: bool,
    pub is_float: bool,
    pub byte_order: ByteOrder,
}

impl AudioFormatInfo {
    /// Create format info for f32 samples
    pub fn f32(sample_rate: usize, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            sample_format: "f32".to_string(),
            bits_per_sample: 32,
            is_signed: true,
            is_float: true,
            byte_order: ByteOrder::Native,
        }
    }

    /// Create format info for i16 samples  
    pub fn i16(sample_rate: usize, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            sample_format: "i16".to_string(),
            bits_per_sample: 16,
            is_signed: true,
            is_float: false,
            byte_order: ByteOrder::Native,
        }
    }

    /// Create format info for i32 samples
    pub fn i32(sample_rate: usize, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            sample_format: "i32".to_string(),
            bits_per_sample: 32,
            is_signed: true,
            is_float: false,
            byte_order: ByteOrder::Native,
        }
    }

    /// Check if this format is compatible with another
    pub fn is_compatible(&self, other: &Self) -> bool {
        self.sample_rate == other.sample_rate
            && self.channels == other.channels
            && self.sample_format == other.sample_format
    }
}

/// Byte order for multi-byte samples
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
    Native,
}

/// Processes streaming audio data in real-time.
///
/// Stream processors can modify audio data as it flows through
/// the streaming pipeline, enabling real-time effects and analysis.
pub trait StreamProcessor<T: AudioSample>: Send + Sync {
    /// Process a chunk of audio data.
    ///
    /// The processor may modify the input in-place or return
    /// a new AudioSamples instance. It may also buffer data
    /// across chunks if needed.
    async fn process_chunk(
        &mut self,
        input: AudioSamples<T>,
    ) -> StreamResult<Option<AudioSamples<T>>>;

    /// Get information about any latency this processor adds.
    fn latency_samples(&self) -> usize {
        0
    }

    /// Flush any internal buffers and return remaining data.
    async fn flush(&mut self) -> StreamResult<Option<AudioSamples<T>>> {
        Ok(None)
    }

    /// Reset the processor state.
    fn reset(&mut self) {}
}

/// Consumes streaming audio data.
///
/// Stream sinks are the final destination for audio data,
/// typically sending it to audio devices, files, or network endpoints.
pub trait StreamSink<T: AudioSample>: Send + Sync {
    /// Send a chunk of audio data to the sink.
    async fn send_chunk(&mut self, chunk: AudioSamples<T>) -> StreamResult<()>;

    /// Flush any pending data and ensure it's sent.
    async fn flush(&mut self) -> StreamResult<()>;

    /// Get the current playback/output latency.
    fn output_latency(&self) -> Duration {
        Duration::from_millis(0)
    }

    /// Get sink-specific metrics.
    fn metrics(&self) -> SinkMetrics {
        SinkMetrics::default()
    }
}

/// Metrics for monitoring audio sources.
#[derive(Debug, Clone, Default)]
pub struct SourceMetrics {
    pub chunks_delivered: u64,
    pub bytes_delivered: u64,
    pub chunks_dropped: u64,
    pub average_chunk_size: usize,
    pub current_buffer_level: f64, // 0.0 to 1.0
    pub underruns: u64,
    pub overruns: u64,
}

/// Metrics for monitoring stream sinks.
#[derive(Debug, Clone, Default)]
pub struct SinkMetrics {
    pub chunks_received: u64,
    pub bytes_received: u64,
    pub chunks_dropped: u64,
    pub average_latency_ms: f64,
    pub buffer_fullness: f64, // 0.0 to 1.0
}

/// Configuration for stream behavior.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Target buffer size in samples per channel
    pub buffer_size: usize,

    /// Maximum allowed buffer size before dropping data
    pub max_buffer_size: usize,

    /// Minimum buffer level before triggering underrun recovery
    pub min_buffer_level: f64,

    /// Maximum time to wait for data before timing out
    pub read_timeout: Duration,

    /// Whether to automatically recover from errors
    pub auto_recovery: bool,

    /// Maximum number of recovery attempts
    pub max_recovery_attempts: usize,

    /// Preferred audio format (if source supports multiple)
    pub preferred_format: Option<AudioFormatInfo>,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_size: 1024,
            max_buffer_size: 8192,
            min_buffer_level: 0.25,
            read_timeout: Duration::from_millis(100),
            auto_recovery: true,
            max_recovery_attempts: 3,
            preferred_format: None,
        }
    }
}

impl StreamConfig {
    /// Create configuration optimized for low-latency applications
    pub fn low_latency() -> Self {
        Self {
            buffer_size: 256,
            max_buffer_size: 1024,
            min_buffer_level: 0.1,
            read_timeout: Duration::from_millis(10),
            auto_recovery: true,
            max_recovery_attempts: 5,
            preferred_format: None,
        }
    }

    /// Create configuration optimized for high-quality streaming
    pub fn high_quality() -> Self {
        Self {
            buffer_size: 4096,
            max_buffer_size: 16384,
            min_buffer_level: 0.5,
            read_timeout: Duration::from_millis(500),
            auto_recovery: true,
            max_recovery_attempts: 3,
            preferred_format: None,
        }
    }

    /// Create configuration for network streaming with error tolerance
    pub fn network_streaming() -> Self {
        Self {
            buffer_size: 2048,
            max_buffer_size: 8192,
            min_buffer_level: 0.3,
            read_timeout: Duration::from_millis(200),
            auto_recovery: true,
            max_recovery_attempts: 10,
            preferred_format: None,
        }
    }
}
