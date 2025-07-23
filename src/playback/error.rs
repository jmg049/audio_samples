//! Error types for audio playback operations.

use crate::AudioSampleError;
use std::fmt;

/// Playback-specific error types.
#[derive(Debug, thiserror::Error)]
pub enum PlaybackError {
    /// Audio sample conversion or processing error
    #[error("Audio processing error: {0}")]
    Audio(#[from] AudioSampleError),

    /// Device-related errors
    #[error("Device error: {0}")]
    Device(String),

    /// Audio format not supported by device
    #[error("Unsupported format: {expected} not supported, available formats: {available:?}")]
    UnsupportedFormat {
        expected: String,
        available: Vec<String>,
    },

    /// Device not found or unavailable
    #[error("Device not found: {device_name}")]
    DeviceNotFound { device_name: String },

    /// Device configuration issues
    #[error("Device configuration error: {0}")]
    DeviceConfig(String),

    /// Buffer underrun or overrun during playback
    #[error("Buffer {operation}: {details}")]
    Buffer {
        operation: &'static str,
        details: String,
    },

    /// Stream interruption or failure
    #[error("Stream error: {0}")]
    Stream(String),

    /// Playback synchronization issues
    #[error("Synchronization error: {0}")]
    Sync(String),

    /// Invalid playback state transition
    #[error("Invalid state transition from {from} to {to}")]
    InvalidState { from: String, to: String },

    /// Resource allocation failures
    #[error("Resource allocation failed: {resource} - {reason}")]
    ResourceAllocation {
        resource: &'static str,
        reason: String,
    },

    /// Backend-specific errors (CPAL, etc.)
    #[error("Backend error: {backend} - {details}")]
    Backend {
        backend: &'static str,
        details: String,
    },

    /// Latency constraints cannot be met
    #[error(
        "Latency constraint violation: requested {requested_ms}ms, minimum possible {minimum_ms}ms"
    )]
    LatencyConstraint { requested_ms: u32, minimum_ms: u32 },

    /// Volume or gain out of valid range
    #[error("Volume out of range: {volume}, valid range: {min} to {max}")]
    VolumeOutOfRange { volume: f64, min: f64, max: f64 },
}

impl PlaybackError {
    /// Create a device error
    pub fn device(details: impl Into<String>) -> Self {
        Self::Device(details.into())
    }

    /// Create a device not found error
    pub fn device_not_found(device_name: impl Into<String>) -> Self {
        Self::DeviceNotFound {
            device_name: device_name.into(),
        }
    }

    /// Create an unsupported format error
    pub fn unsupported_format(expected: impl Into<String>, available: Vec<String>) -> Self {
        Self::UnsupportedFormat {
            expected: expected.into(),
            available,
        }
    }

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

    /// Create a backend error
    pub fn backend(backend: &'static str, details: impl Into<String>) -> Self {
        Self::Backend {
            backend,
            details: details.into(),
        }
    }

    /// Create a latency constraint error
    pub fn latency_constraint(requested_ms: u32, minimum_ms: u32) -> Self {
        Self::LatencyConstraint {
            requested_ms,
            minimum_ms,
        }
    }

    /// Create a volume out of range error
    pub fn volume_out_of_range(volume: f64, min: f64, max: f64) -> Self {
        Self::VolumeOutOfRange { volume, min, max }
    }

    /// Create an invalid state transition error
    pub fn invalid_state(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self::InvalidState {
            from: from.into(),
            to: to.into(),
        }
    }

    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Buffer { .. } => true,
            Self::Stream(_) => true,
            Self::Backend { .. } => true,
            Self::DeviceConfig(_) => true,
            _ => false,
        }
    }

    /// Check if this error indicates a device problem
    pub fn is_device_error(&self) -> bool {
        matches!(
            self,
            Self::Device(_) | Self::DeviceNotFound { .. } | Self::DeviceConfig(_)
        )
    }

    /// Check if this error is related to audio format issues
    pub fn is_format_error(&self) -> bool {
        matches!(self, Self::UnsupportedFormat { .. })
    }
}

/// Result type for playback operations
pub type PlaybackResult<T> = Result<T, PlaybackError>;

/// Helper macro for creating playback errors
#[macro_export]
macro_rules! playback_error {
    ($kind:ident, $($args:tt)*) => {
        $crate::playback::PlaybackError::$kind(format!($($args)*))
    };
}

/// Metrics for tracking playback errors
#[derive(Debug, Clone, Default)]
pub struct PlaybackErrorMetrics {
    pub total_errors: u64,
    pub device_errors: u64,
    pub format_errors: u64,
    pub buffer_errors: u64,
    pub stream_errors: u64,
    pub backend_errors: u64,
    pub recovered_errors: u64,
    pub fatal_errors: u64,
}

impl PlaybackErrorMetrics {
    /// Record a new error
    pub fn record_error(&mut self, error: &PlaybackError) {
        self.total_errors += 1;

        match error {
            PlaybackError::Device(_) | PlaybackError::DeviceNotFound { .. } => {
                self.device_errors += 1;
            }
            PlaybackError::UnsupportedFormat { .. } => {
                self.format_errors += 1;
            }
            PlaybackError::Buffer { .. } => {
                self.buffer_errors += 1;
            }
            PlaybackError::Stream(_) => {
                self.stream_errors += 1;
            }
            PlaybackError::Backend { .. } => {
                self.backend_errors += 1;
            }
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

    /// Get the overall error rate (0.0 to 1.0)
    pub fn error_rate(&self) -> f64 {
        if self.total_errors == 0 {
            0.0
        } else {
            self.fatal_errors as f64 / self.total_errors as f64
        }
    }

    /// Get device error rate specifically
    pub fn device_error_rate(&self) -> f64 {
        if self.total_errors == 0 {
            0.0
        } else {
            self.device_errors as f64 / self.total_errors as f64
        }
    }
}

/// Convert CPAL errors to PlaybackError
#[cfg(feature = "playback")]
impl From<cpal::BuildStreamError> for PlaybackError {
    fn from(err: cpal::BuildStreamError) -> Self {
        Self::backend("cpal", format!("Failed to build stream: {}", err))
    }
}

#[cfg(feature = "playback")]
impl From<cpal::PlayStreamError> for PlaybackError {
    fn from(err: cpal::PlayStreamError) -> Self {
        Self::backend("cpal", format!("Failed to play stream: {}", err))
    }
}

#[cfg(feature = "playback")]
impl From<cpal::PauseStreamError> for PlaybackError {
    fn from(err: cpal::PauseStreamError) -> Self {
        Self::backend("cpal", format!("Failed to pause stream: {}", err))
    }
}

#[cfg(feature = "playback")]
impl From<cpal::DefaultStreamConfigError> for PlaybackError {
    fn from(err: cpal::DefaultStreamConfigError) -> Self {
        Self::backend(
            "cpal",
            format!("Failed to get default stream config: {}", err),
        )
    }
}

#[cfg(feature = "playback")]
impl From<cpal::SupportedStreamConfigsError> for PlaybackError {
    fn from(err: cpal::SupportedStreamConfigsError) -> Self {
        Self::backend(
            "cpal",
            format!("Failed to get supported stream configs: {}", err),
        )
    }
}
