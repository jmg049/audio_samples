//! Main streaming infrastructure and audio stream management.

use super::{
    buffers::{BufferConfig, CircularBuffer},
    error::{StreamError, StreamResult},
    traits::{AudioFormatInfo, AudioSource, StreamConfig},
};
use crate::{AudioSample, AudioSamples};
use std::time::{Duration, Instant};

#[cfg(feature = "streaming")]

/// Represents the current state of an audio stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamState {
    /// Stream is not yet initialized
    Idle,
    /// Stream is starting up
    Starting,
    /// Stream is actively running
    Running,
    /// Stream is paused
    Paused,
    /// Stream is stopping
    Stopping,
    /// Stream has stopped
    Stopped,
    /// Stream encountered an error
    Error(String),
}

impl StreamState {
    /// Check if the stream is in an active state
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Running)
    }

    /// Check if the stream can be started
    pub fn can_start(&self) -> bool {
        matches!(self, Self::Idle | Self::Stopped)
    }

    /// Check if the stream can be paused
    pub fn can_pause(&self) -> bool {
        matches!(self, Self::Running)
    }

    /// Check if the stream is in an error state
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }
}

/// A complete audio streaming pipeline.
///
/// This struct manages the flow of audio data from source through
/// optional processors to an optional sink, with buffering and
/// error recovery capabilities.
pub struct AudioStream<T: AudioSample> {
    config: StreamConfig,
    state: StreamState,
    buffer: CircularBuffer<T>,
    format_info: Option<AudioFormatInfo>,

    // Stream statistics
    stats: StreamStats,
    last_chunk_time: Option<Instant>,

    // Error recovery
    error_count: usize,
    last_error: Option<StreamError>,
}

impl<T: AudioSample> AudioStream<T> {
    /// Create a new audio stream with the given configuration.
    pub fn new(config: StreamConfig) -> Self {
        let buffer_config = BufferConfig {
            max_chunks: config.buffer_size / 256, // Estimate chunks from samples
            max_samples: config.max_buffer_size,
            target_level: 0.5,
            min_level: config.min_buffer_level,
            max_level: 0.9,
            drop_on_overflow: true,
            pre_allocate: true,
        };

        Self {
            config,
            state: StreamState::Idle,
            buffer: CircularBuffer::new(buffer_config),
            format_info: None,
            stats: StreamStats::default(),
            last_chunk_time: None,
            error_count: 0,
            last_error: None,
        }
    }

    /// Create a stream with default configuration.
    pub fn with_default_config() -> Self {
        Self::new(StreamConfig::default())
    }

    /// Create a stream optimized for low latency.
    pub fn low_latency() -> Self {
        Self::new(StreamConfig::low_latency())
    }

    /// Create a stream optimized for network streaming.
    pub fn network_streaming() -> Self {
        Self::new(StreamConfig::network_streaming())
    }

    /// Get the current stream state.
    pub fn state(&self) -> &StreamState {
        &self.state
    }

    /// Get the stream format information.
    pub fn format_info(&self) -> Option<&AudioFormatInfo> {
        self.format_info.as_ref()
    }

    /// Set the format information for this stream.
    pub fn set_format_info(&mut self, format: AudioFormatInfo) {
        self.format_info = Some(format);
    }

    /// Start the stream with the given source.
    pub async fn start_with_source<S>(&mut self, source: S) -> StreamResult<()>
    where
        S: AudioSource<T> + 'static,
    {
        if !self.state.can_start() {
            return Err(StreamError::InvalidConfig(format!(
                "Cannot start stream in state {:?}",
                self.state
            )));
        }

        self.state = StreamState::Starting;

        // Get format info from source
        let format = source.format_info();
        self.format_info = Some(format.clone());

        // Verify format compatibility if we have preferences
        if let Some(ref preferred) = self.config.preferred_format {
            if !format.is_compatible(preferred) {
                self.state = StreamState::Error("Format mismatch".to_string());
                return Err(StreamError::format_mismatch(
                    format!("{:?}", preferred),
                    format!("{:?}", format),
                ));
            }
        }

        // Start streaming
        self.state = StreamState::Running;
        self.stats.start_time = Instant::now();

        Ok(())
    }

    /// Read the next chunk from the internal buffer.
    pub fn read_chunk(&mut self) -> Option<AudioSamples<T>> {
        let chunk = self.buffer.pop();

        if chunk.is_some() {
            self.stats.chunks_read += 1;
            self.last_chunk_time = Some(Instant::now());
        } else if self.buffer.is_underrun() {
            self.stats.underruns += 1;
        }

        chunk
    }

    /// Write a chunk to the internal buffer.
    pub fn write_chunk(&mut self, chunk: AudioSamples<T>) -> StreamResult<()> {
        let result = self.buffer.push(chunk);

        match result {
            Ok(()) => {
                self.stats.chunks_written += 1;
                self.stats.bytes_processed +=
                    (self.buffer.sample_count() * std::mem::size_of::<T>()) as u64;
            }
            Err(ref e)
                if matches!(
                    e,
                    StreamError::Buffer {
                        operation: "overrun",
                        ..
                    }
                ) =>
            {
                self.stats.overruns += 1;
            }
            _ => {}
        }

        result
    }

    /// Pause the stream.
    pub fn pause(&mut self) -> StreamResult<()> {
        if !self.state.can_pause() {
            return Err(StreamError::InvalidConfig(format!(
                "Cannot pause stream in state {:?}",
                self.state
            )));
        }

        self.state = StreamState::Paused;
        Ok(())
    }

    /// Resume a paused stream.
    pub fn resume(&mut self) -> StreamResult<()> {
        if !matches!(self.state, StreamState::Paused) {
            return Err(StreamError::InvalidConfig(format!(
                "Cannot resume stream in state {:?}",
                self.state
            )));
        }

        self.state = StreamState::Running;
        Ok(())
    }

    /// Stop the stream.
    pub fn stop(&mut self) -> StreamResult<()> {
        if matches!(self.state, StreamState::Stopped | StreamState::Idle) {
            return Ok(());
        }

        self.state = StreamState::Stopping;
        self.buffer.clear();
        self.state = StreamState::Stopped;

        Ok(())
    }

    /// Get current buffer level (0.0 to 1.0).
    pub fn buffer_level(&self) -> f64 {
        self.buffer.level()
    }

    /// Check if the buffer is currently experiencing underrun.
    pub fn is_underrun(&self) -> bool {
        self.buffer.is_underrun()
    }

    /// Check if the buffer is currently experiencing overrun.
    pub fn is_overrun(&self) -> bool {
        self.buffer.is_overrun()
    }

    /// Get stream statistics.
    pub fn stats(&self) -> &StreamStats {
        &self.stats
    }

    /// Reset stream statistics.
    pub fn reset_stats(&mut self) {
        self.stats = StreamStats::default();
        self.stats.start_time = Instant::now();
    }

    /// Get the estimated latency of the stream.
    pub fn estimated_latency(&self) -> Duration {
        let buffer_samples = self.buffer.sample_count();
        let sample_rate = self
            .format_info
            .as_ref()
            .map(|f| f.sample_rate)
            .unwrap_or(44100);

        let buffer_duration_ms = (buffer_samples as f64 / sample_rate as f64) * 1000.0;
        Duration::from_millis(buffer_duration_ms as u64)
    }

    /// Handle errors and attempt recovery if configured.
    pub fn handle_error(&mut self, error: StreamError) -> bool {
        self.last_error = Some(error.clone());
        self.error_count += 1;
        self.stats.errors += 1;

        if self.config.auto_recovery
            && error.is_recoverable()
            && self.error_count <= self.config.max_recovery_attempts
        {
            // Clear buffer and attempt to continue
            self.buffer.clear();
            self.stats.recovery_attempts += 1;
            true // Signal that recovery was attempted
        } else {
            self.state = StreamState::Error(error.to_string());
            false // Signal that recovery failed or wasn't attempted
        }
    }

    /// Get the last error that occurred.
    pub fn last_error(&self) -> Option<&StreamError> {
        self.last_error.as_ref()
    }

    /// Clear any error state and reset error counter.
    pub fn clear_error(&mut self) {
        self.last_error = None;
        self.error_count = 0;
        if matches!(self.state, StreamState::Error(_)) {
            self.state = StreamState::Idle;
        }
    }
}

/// Statistics for monitoring stream performance.
#[derive(Debug, Clone)]
pub struct StreamStats {
    pub start_time: Instant,
    pub chunks_written: u64,
    pub chunks_read: u64,
    pub bytes_processed: u64,
    pub underruns: u64,
    pub overruns: u64,
    pub errors: u64,
    pub recovery_attempts: u64,
    pub average_latency_ms: f64,
}

impl Default for StreamStats {
    fn default() -> Self {
        Self {
            start_time: Instant::now(),
            chunks_written: 0,
            chunks_read: 0,
            bytes_processed: 0,
            underruns: 0,
            overruns: 0,
            errors: 0,
            recovery_attempts: 0,
            average_latency_ms: 0.0,
        }
    }
}

impl StreamStats {
    /// Get the total runtime of the stream.
    pub fn runtime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get the average data rate in bytes per second.
    pub fn data_rate_bps(&self) -> f64 {
        let runtime_secs = self.runtime().as_secs_f64();
        if runtime_secs > 0.0 {
            self.bytes_processed as f64 / runtime_secs
        } else {
            0.0
        }
    }

    /// Get the chunk processing rate.
    pub fn chunk_rate(&self) -> f64 {
        let runtime_secs = self.runtime().as_secs_f64();
        if runtime_secs > 0.0 {
            (self.chunks_read + self.chunks_written) as f64 / runtime_secs
        } else {
            0.0
        }
    }

    /// Get the error rate (0.0 to 1.0).
    pub fn error_rate(&self) -> f64 {
        let total_operations = self.chunks_read + self.chunks_written;
        if total_operations > 0 {
            self.errors as f64 / total_operations as f64
        } else {
            0.0
        }
    }
}
