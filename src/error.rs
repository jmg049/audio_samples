//! Error types and result utilities for audio sample operations.

use thiserror::Error;

/// Convenience type alias for results that may contain AudioSampleError
pub type AudioSampleResult<T> = Result<T, AudioSampleError>;

/// Error types that can occur during audio sample operations.
#[derive(Error, Debug)]
pub enum AudioSampleError {
    /// Error that occurs when converting between audio sample types fails.
    /// 
    /// Contains the sample value, source type, target type, and reason for failure.
    /// This typically happens when values are out of range for the target type.
    #[error("Audio sample conversion error: Failed to convert {0} from type {1} to type {2}:\nReason: {3}")]
    ConversionError(String, String, String, String)
}