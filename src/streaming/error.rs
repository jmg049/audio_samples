//! Error types for streaming operations.

use crate::AudioSampleError;
use std::fmt;

/// Streaming-specific error types.
#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    /// Audio sample conversion or processing error
    #[error("Audio processing error: {0}")]
    Audio(#[from] AudioSampleError),

    /// Network connectivity issues
    #[error("Network error: {0}")]
    Network(#[from] std::io::Error),

    /// Buffer underrun or overrun
    #[error("Buffer {operation} - {details}")]
    Buffer {
        operation: &'static str,
        details: String,
    },

    /// Stream format mismatch
    #[error("Format mismatch: expected {expected}, got {actual}")]
    FormatMismatch { expected: String, actual: String },

    /// Stream configuration errors
    #[error("Invalid stream configuration: {0}")]
    InvalidConfig(String),

    /// Stream synchronization issues
    #[error("Synchronization error: {0}")]
    Sync(String),

    /// Stream ended unexpectedly
    #[error("Stream ended unexpectedly: {0}")]
    UnexpectedEnd(String),

    /// Timeout during stream operation
    #[error("Operation timed out after {duration_ms}ms")]
    Timeout { duration_ms: u64 },

    /// Protocol-specific errors
    #[error("Protocol error: {protocol} - {details}")]
    Protocol {
        protocol: &'static str,
        details: String,
    },

    /// Resource allocation errors
    #[error("Resource allocation failed: {resource} - {reason}")]
    ResourceAllocation {
        resource: &'static str,
        reason: String,
    },
}

impl StreamError {
    /// Create a buffer underrun error
    pub fn buffer_underrun(details: impl Into<String>) -> Self {
        Self::Buffer {
            operation: "underrun",
            details: details.into(),
        }
    }

    /// Create a buffer overrun error
    pub fn buffer_overrun(details: impl Into<String>) -> Self {
        Self::Buffer {
            operation: "overrun",
            details: details.into(),
        }
    }

    /// Create a format mismatch error
    pub fn format_mismatch(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        Self::FormatMismatch {
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Create a timeout error
    pub fn timeout(duration_ms: u64) -> Self {
        Self::Timeout { duration_ms }
    }

    /// Create a protocol error
    pub fn protocol(protocol: &'static str, details: impl Into<String>) -> Self {
        Self::Protocol {
            protocol,
            details: details.into(),
        }
    }

    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Network(_) => true,
            Self::Buffer { .. } => true,
            Self::Timeout { .. } => true,
            Self::Protocol { .. } => true,
            Self::Sync(_) => true,
            _ => false,
        }
    }

    /// Check if this is a fatal error that should terminate the stream
    pub fn is_fatal(&self) -> bool {
        !self.is_recoverable()
    }
}

/// Result type for streaming operations
pub type StreamResult<T> = Result<T, StreamError>;

/// Helper macro for creating stream errors
#[macro_export]
macro_rules! stream_error {
    ($kind:ident, $($args:tt)*) => {
        $crate::streaming::StreamError::$kind(format!($($args)*))
    };
}

/// Metrics for stream error tracking
#[derive(Debug, Clone, Default)]
pub struct StreamErrorMetrics {
    pub total_errors: u64,
    pub network_errors: u64,
    pub buffer_errors: u64,
    pub format_errors: u64,
    pub timeout_errors: u64,
    pub recovered_errors: u64,
    pub fatal_errors: u64,
}

impl StreamErrorMetrics {
    /// Record a new error
    pub fn record_error(&mut self, error: &StreamError) {
        self.total_errors += 1;

        match error {
            StreamError::Network(_) => self.network_errors += 1,
            StreamError::Buffer { .. } => self.buffer_errors += 1,
            StreamError::FormatMismatch { .. } => self.format_errors += 1,
            StreamError::Timeout { .. } => self.timeout_errors += 1,
            _ => {}
        }

        if error.is_recoverable() {
            self.recovered_errors += 1;
        } else {
            self.fatal_errors += 1;
        }
    }

    /// Reset all counters
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Get error rate (0.0 to 1.0)
    pub fn error_rate(&self) -> f64 {
        if self.total_errors == 0 {
            0.0
        } else {
            self.fatal_errors as f64 / self.total_errors as f64
        }
    }
}
