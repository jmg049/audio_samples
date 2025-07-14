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
    #[error(
        "Audio sample conversion error: Failed to convert {0} from type {1} to type {2}:\nReason: {3}"
    )]
    ConversionError(String, String, String, String),

    /// Error that occurs when an invalid range is provided.
    ///
    /// This typically happens when min >= max in normalization or clipping operations.
    #[error("Invalid range error: {0}")]
    InvalidRange(String),

    /// Error that occurs when invalid parameters are provided to an operation.
    ///
    /// This includes cases like empty filter coefficients, negative durations, etc.
    #[error("Invalid parameter error: {0}")]
    InvalidParameter(String),

    /// Error that occurs when array dimensions don't match expected values.
    ///
    /// This happens when window length doesn't match audio length, etc.
    #[error("Dimension mismatch error: {0}")]
    DimensionMismatch(String),

    /// Error that occurs when invalid input is provided to an operation.
    ///
    /// This includes cases like empty audio data, invalid file formats, etc.
    #[error("Invalid input error: {msg}")]
    InvalidInput { msg: String },

    /// Error that occurs during audio processing operations.
    ///
    /// This includes resampling errors, filter failures, etc.
    #[error("Processing error: {msg}")]
    ProcessingError { msg: String },

    /// Error that occurs when a feature is not enabled.
    ///
    /// This happens when trying to use optional functionality that wasn't compiled in.
    #[error("Feature '{feature}' is not enabled. Please enable the feature and recompile.")]
    FeatureNotEnabled { feature: String },
}
