//! Error types and result utilities for audio sample operations.

use thiserror::Error;

/// Convenience type alias for results that may contain AudioSampleError
pub type AudioSampleResult<T> = Result<T, AudioSampleError>;

/// Error types that can occur during audio sample operations.
#[derive(Error, Debug, Clone)]
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
    InvalidInput {
        /// Error message.
        msg: String,
    },

    /// Error that occurs during audio processing operations.
    ///
    /// This includes resampling errors, filter failures, etc.
    #[error("Processing error: {msg}")]
    ProcessingError {
        /// Error message.
        msg: String,
    },

    /// Error that occurs when a feature is not enabled.
    ///
    /// This happens when trying to use optional functionality that wasn't compiled in.
    #[error("Feature '{feature}' is not enabled. Please enable the feature and recompile.")]
    FeatureNotEnabled {
        /// Feature name.
        feature: String,
    },

    /// Error that occurs when trying to get an ndarray as a slice (slice mut).
    ///
    /// This typically happens when the ndarray is not contiguous and not in standard order
    #[error("Array layout error: {message}")]
    ArrayLayoutError {
        /// Error message.
        message: String,
    },

    /// Error that can be used to indicate that the error originates from an Option type.
    ///
    #[error("Option error: {message}")]
    OptionError {
        /// Error message.
        message: String,
    },

    /// Error that occurs when trying to modify borrowed audio data.
    ///
    /// This happens when attempting to perform in-place operations on audio data that is borrowed.
    #[error("Cannot modify borrowed audio data: {message}")]
    BorrowedDataError {
        /// Error message.
        message: String,
    },

    /// Generic internal error for unexpected situations.
    ///
    /// Contains a message describing the internal error.
    #[error("Internal error: {0}")]
    InternalError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_types() {
        let conversion_error = AudioSampleError::ConversionError(
            "32768".to_string(),
            "i16".to_string(),
            "i8".to_string(),
            "Value out of range".to_string(),
        );

        assert!(format!("{}", conversion_error).contains("Failed to convert 32768"));

        let invalid_range = AudioSampleError::InvalidRange("min >= max".to_string());
        assert!(format!("{}", invalid_range).contains("Invalid range"));
    }

    #[test]
    fn test_result_type_alias() {
        let ok_result: AudioSampleResult<i32> = Ok(42);
        assert_eq!(ok_result.unwrap(), 42);

        let err_result: AudioSampleResult<i32> =
            Err(AudioSampleError::InternalError("test".to_string()));
        assert!(err_result.is_err());
    }

    #[test]
    fn test_standard_result_operations() -> AudioSampleResult<()> {
        let result: AudioSampleResult<i32> = Ok(42);
        let doubled = result.map(|x| x * 2)?;
        assert_eq!(doubled, 84);

        Ok(())
    }
}
