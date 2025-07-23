//! Buffer management for streaming audio.

use super::error::{StreamError, StreamResult};
use crate::{AudioSample, AudioSamples};
use std::collections::VecDeque;

#[cfg(feature = "streaming")]
use crossbeam::queue::{ArrayQueue, SegQueue};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Configuration for streaming buffers.
#[derive(Debug, Clone)]
pub struct BufferConfig {
    /// Maximum number of chunks to buffer
    pub max_chunks: usize,

    /// Maximum total samples to buffer
    pub max_samples: usize,

    /// Target buffer level (0.0 to 1.0)
    pub target_level: f64,

    /// Minimum buffer level before underrun (0.0 to 1.0)
    pub min_level: f64,

    /// Maximum buffer level before overrun (0.0 to 1.0)
    pub max_level: f64,

    /// Whether to drop old data on overflow
    pub drop_on_overflow: bool,

    /// Pre-allocate buffer space
    pub pre_allocate: bool,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            max_chunks: 32,
            max_samples: 16384,
            target_level: 0.5,
            min_level: 0.25,
            max_level: 0.9,
            drop_on_overflow: true,
            pre_allocate: true,
        }
    }
}

/// A lock-free circular buffer optimized for streaming audio.
///
/// This buffer is designed for single-producer, single-consumer scenarios
/// and provides low-latency access with minimal allocations.
pub struct CircularBuffer<T: AudioSample> {
    config: BufferConfig,
    buffer: VecDeque<AudioSamples<T>>,
    total_samples: usize,
    underrun_count: u64,
    overrun_count: u64,
    last_access: Instant,
}

impl<T: AudioSample> CircularBuffer<T> {
    /// Create a new circular buffer with the given configuration.
    pub fn new(config: BufferConfig) -> Self {
        let capacity = if config.pre_allocate {
            config.max_chunks
        } else {
            0
        };

        Self {
            config,
            buffer: VecDeque::with_capacity(capacity),
            total_samples: 0,
            underrun_count: 0,
            overrun_count: 0,
            last_access: Instant::now(),
        }
    }

    /// Push a new audio chunk into the buffer.
    pub fn push(&mut self, chunk: AudioSamples<T>) -> StreamResult<()> {
        let chunk_samples = chunk.samples_per_channel() * chunk.channels();

        // Check if adding this chunk would exceed limits
        if self.buffer.len() >= self.config.max_chunks
            || self.total_samples + chunk_samples > self.config.max_samples
        {
            if self.config.drop_on_overflow {
                // Drop oldest chunk to make room
                if let Some(dropped) = self.buffer.pop_front() {
                    self.total_samples -= dropped.samples_per_channel() * dropped.channels();
                }
                self.overrun_count += 1;
            } else {
                return Err(StreamError::buffer_overrun(format!(
                    "Buffer full: {} chunks, {} samples",
                    self.buffer.len(),
                    self.total_samples
                )));
            }
        }

        self.total_samples += chunk_samples;
        self.buffer.push_back(chunk);
        self.last_access = Instant::now();

        Ok(())
    }

    /// Pop the next audio chunk from the buffer.
    pub fn pop(&mut self) -> Option<AudioSamples<T>> {
        if let Some(chunk) = self.buffer.pop_front() {
            let chunk_samples = chunk.samples_per_channel() * chunk.channels();
            self.total_samples -= chunk_samples;
            self.last_access = Instant::now();
            Some(chunk)
        } else {
            self.underrun_count += 1;
            None
        }
    }

    /// Peek at the next chunk without removing it.
    pub fn peek(&self) -> Option<&AudioSamples<T>> {
        self.buffer.front()
    }

    /// Get the current buffer level (0.0 to 1.0).
    pub fn level(&self) -> f64 {
        self.total_samples as f64 / self.config.max_samples as f64
    }

    /// Check if the buffer is at or below the minimum level.
    pub fn is_underrun(&self) -> bool {
        self.level() <= self.config.min_level
    }

    /// Check if the buffer is at or above the maximum level.
    pub fn is_overrun(&self) -> bool {
        self.level() >= self.config.max_level
    }

    /// Get the number of chunks currently buffered.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get the total number of samples currently buffered.
    pub fn sample_count(&self) -> usize {
        self.total_samples
    }

    /// Get buffer statistics.
    pub fn stats(&self) -> BufferStats {
        BufferStats {
            current_chunks: self.buffer.len(),
            current_samples: self.total_samples,
            max_chunks: self.config.max_chunks,
            max_samples: self.config.max_samples,
            level: self.level(),
            underrun_count: self.underrun_count,
            overrun_count: self.overrun_count,
            last_access: self.last_access,
        }
    }

    /// Clear all buffered data.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.total_samples = 0;
        self.last_access = Instant::now();
    }

    /// Resize the buffer capacity.
    pub fn resize(&mut self, new_config: BufferConfig) {
        self.config = new_config;

        // If we're now over capacity, drop oldest chunks
        while self.buffer.len() > self.config.max_chunks
            || self.total_samples > self.config.max_samples
        {
            if let Some(dropped) = self.buffer.pop_front() {
                self.total_samples -= dropped.samples_per_channel() * dropped.channels();
                self.overrun_count += 1;
            } else {
                break;
            }
        }
    }
}

/// Statistics about buffer performance.
#[derive(Debug, Clone)]
pub struct BufferStats {
    pub current_chunks: usize,
    pub current_samples: usize,
    pub max_chunks: usize,
    pub max_samples: usize,
    pub level: f64,
    pub underrun_count: u64,
    pub overrun_count: u64,
    pub last_access: Instant,
}

/// A thread-safe streaming buffer for multi-producer, multi-consumer scenarios.
///
/// This buffer uses lock-free queues when possible and falls back to
/// mutex-protected structures when necessary.
#[cfg(feature = "streaming")]
pub struct StreamBuffer<T: AudioSample> {
    queue: Arc<SegQueue<AudioSamples<T>>>,
    config: BufferConfig,
    stats: Arc<Mutex<BufferStats>>,
}

#[cfg(feature = "streaming")]
impl<T: AudioSample> StreamBuffer<T> {
    /// Create a new thread-safe stream buffer.
    pub fn new(config: BufferConfig) -> Self {
        let stats = BufferStats {
            current_chunks: 0,
            current_samples: 0,
            max_chunks: config.max_chunks,
            max_samples: config.max_samples,
            level: 0.0,
            underrun_count: 0,
            overrun_count: 0,
            last_access: Instant::now(),
        };

        Self {
            queue: Arc::new(SegQueue::new()),
            config,
            stats: Arc::new(Mutex::new(stats)),
        }
    }

    /// Push a chunk into the buffer (non-blocking).
    pub fn try_push(&self, chunk: AudioSamples<T>) -> StreamResult<()> {
        let chunk_samples = chunk.samples_per_channel() * chunk.channels();

        // Check current buffer level
        {
            let mut stats = self.stats.lock().unwrap();

            if stats.current_chunks >= self.config.max_chunks
                || stats.current_samples + chunk_samples > self.config.max_samples
            {
                if !self.config.drop_on_overflow {
                    return Err(StreamError::buffer_overrun(format!(
                        "Buffer full: {} chunks, {} samples",
                        stats.current_chunks, stats.current_samples
                    )));
                }

                // Try to drop an item to make room
                if let Some(dropped) = self.queue.pop() {
                    let dropped_samples = dropped.samples_per_channel() * dropped.channels();
                    stats.current_chunks -= 1;
                    stats.current_samples -= dropped_samples;
                    stats.overrun_count += 1;
                }
            }

            stats.current_chunks += 1;
            stats.current_samples += chunk_samples;
            stats.level = stats.current_samples as f64 / self.config.max_samples as f64;
            stats.last_access = Instant::now();
        }

        self.queue.push(chunk);
        Ok(())
    }

    /// Pop a chunk from the buffer (non-blocking).
    pub fn try_pop(&self) -> Option<AudioSamples<T>> {
        if let Some(chunk) = self.queue.pop() {
            let chunk_samples = chunk.samples_per_channel() * chunk.channels();

            {
                let mut stats = self.stats.lock().unwrap();
                stats.current_chunks = stats.current_chunks.saturating_sub(1);
                stats.current_samples = stats.current_samples.saturating_sub(chunk_samples);
                stats.level = stats.current_samples as f64 / self.config.max_samples as f64;
                stats.last_access = Instant::now();
            }

            Some(chunk)
        } else {
            // Record underrun
            {
                let mut stats = self.stats.lock().unwrap();
                stats.underrun_count += 1;
            }
            None
        }
    }

    /// Get current buffer statistics.
    pub fn stats(&self) -> BufferStats {
        self.stats.lock().unwrap().clone()
    }

    /// Check if buffer is below minimum level.
    pub fn is_underrun(&self) -> bool {
        let stats = self.stats.lock().unwrap();
        stats.level <= self.config.min_level
    }

    /// Check if buffer is above maximum level.
    pub fn is_overrun(&self) -> bool {
        let stats = self.stats.lock().unwrap();
        stats.level >= self.config.max_level
    }

    /// Get the current buffer level (0.0 to 1.0).
    pub fn level(&self) -> f64 {
        self.stats.lock().unwrap().level
    }
}

/// Adaptive buffer that automatically adjusts its size based on stream characteristics.
pub struct AdaptiveBuffer<T: AudioSample> {
    buffer: CircularBuffer<T>,
    base_config: BufferConfig,
    adaptation_history: VecDeque<f64>,
    last_adaptation: Instant,
    adaptation_interval: Duration,
}

impl<T: AudioSample> AdaptiveBuffer<T> {
    /// Create a new adaptive buffer.
    pub fn new(base_config: BufferConfig) -> Self {
        let buffer = CircularBuffer::new(base_config.clone());

        Self {
            buffer,
            base_config,
            adaptation_history: VecDeque::with_capacity(10),
            last_adaptation: Instant::now(),
            adaptation_interval: Duration::from_millis(500),
        }
    }

    /// Push data and potentially adapt buffer size.
    pub fn push(&mut self, chunk: AudioSamples<T>) -> StreamResult<()> {
        let result = self.buffer.push(chunk);
        self.maybe_adapt();
        result
    }

    /// Pop data from the buffer.
    pub fn pop(&mut self) -> Option<AudioSamples<T>> {
        let result = self.buffer.pop();
        self.maybe_adapt();
        result
    }

    fn maybe_adapt(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_adaptation) < self.adaptation_interval {
            return;
        }

        let current_level = self.buffer.level();
        self.adaptation_history.push_back(current_level);

        if self.adaptation_history.len() > 10 {
            self.adaptation_history.pop_front();
        }

        // Only adapt if we have enough history
        if self.adaptation_history.len() >= 5 {
            let avg_level: f64 =
                self.adaptation_history.iter().sum::<f64>() / self.adaptation_history.len() as f64;

            let mut new_config = self.base_config.clone();

            if avg_level < 0.2 {
                // Consistently low, reduce buffer size
                new_config.max_samples = (new_config.max_samples * 3 / 4).max(1024);
                new_config.max_chunks = (new_config.max_chunks * 3 / 4).max(4);
            } else if avg_level > 0.8 {
                // Consistently high, increase buffer size
                new_config.max_samples = (new_config.max_samples * 5 / 4).min(65536);
                new_config.max_chunks = (new_config.max_chunks * 5 / 4).min(128);
            }

            if new_config.max_samples != self.base_config.max_samples {
                self.buffer.resize(new_config.clone());
                self.base_config = new_config;
                self.adaptation_history.clear();
            }
        }

        self.last_adaptation = now;
    }

    /// Get buffer statistics including adaptation info.
    pub fn stats(&self) -> AdaptiveBufferStats {
        let base_stats = self.buffer.stats();
        AdaptiveBufferStats {
            base_stats,
            adaptations_made: self.adaptation_history.len(),
            current_target_size: self.base_config.max_samples,
            average_level: self.adaptation_history.iter().sum::<f64>()
                / self.adaptation_history.len().max(1) as f64,
        }
    }
}

/// Statistics for adaptive buffers.
#[derive(Debug, Clone)]
pub struct AdaptiveBufferStats {
    pub base_stats: BufferStats,
    pub adaptations_made: usize,
    pub current_target_size: usize,
    pub average_level: f64,
}
