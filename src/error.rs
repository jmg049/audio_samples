//! Error types and result utilities for audio sample operations.

use thiserror::Error;

/// Convenience type alias for results that may contain AudioSampleError
pub type AudioSampleResult<T> = Result<T, AudioSampleError>;

/// Chainable result type for improved error handling ergonomics.
///
/// This wrapper around `AudioSampleResult<T>` provides fluent methods for
/// error handling and method chaining, making it easier to work with fallible
/// audio operations in a functional style.
#[derive(Debug)]
pub struct ChainableResult<T> {
    inner: AudioSampleResult<T>,
}

impl<T> ChainableResult<T> {
    /// Creates a new ChainableResult from a successful value.
    pub fn ok(value: T) -> Self {
        Self { inner: Ok(value) }
    }

    /// Creates a new ChainableResult from an error.
    pub fn err(error: AudioSampleError) -> Self {
        Self { inner: Err(error) }
    }

    /// Maps the contained value using the provided function if Ok, leaves error unchanged.
    pub fn map<U, F>(self, f: F) -> ChainableResult<U>
    where
        F: FnOnce(T) -> U,
    {
        ChainableResult {
            inner: self.inner.map(f),
        }
    }

    /// Maps the contained value using a fallible function if Ok.
    pub fn and_then<U, F>(self, f: F) -> ChainableResult<U>
    where
        F: FnOnce(T) -> AudioSampleResult<U>,
    {
        ChainableResult {
            inner: self.inner.and_then(f),
        }
    }

    /// Maps the contained value using a fallible function that returns ChainableResult.
    pub fn chain<U, F>(self, f: F) -> ChainableResult<U>
    where
        F: FnOnce(T) -> ChainableResult<U>,
    {
        match self.inner {
            Ok(value) => f(value),
            Err(error) => ChainableResult::err(error),
        }
    }

    /// Maps the error using the provided function if Err, leaves Ok unchanged.
    pub fn map_err<F>(self, f: F) -> Self
    where
        F: FnOnce(AudioSampleError) -> AudioSampleError,
    {
        Self {
            inner: self.inner.map_err(f),
        }
    }

    /// Provides a default value in case of error.
    pub fn unwrap_or(self, default: T) -> T {
        self.inner.unwrap_or(default)
    }

    /// Provides a default value computed from the error in case of Err.
    pub fn unwrap_or_else<F>(self, f: F) -> T
    where
        F: FnOnce(AudioSampleError) -> T,
    {
        self.inner.unwrap_or_else(f)
    }

    /// Converts into the underlying AudioSampleResult.
    pub fn into_result(self) -> AudioSampleResult<T> {
        self.inner
    }

    /// Returns true if the result is Ok.
    pub fn is_ok(&self) -> bool {
        self.inner.is_ok()
    }

    /// Returns true if the result is Err.
    pub fn is_err(&self) -> bool {
        self.inner.is_err()
    }

    /// Logs the error using tracing if Err, returns self for chaining.
    pub fn log_on_error(self, context: &str) -> Self {
        if let Err(ref error) = self.inner {
            tracing::error!("{}: {}", context, error);
        }
        self
    }

    /// Executes a closure with the contained value if Ok, returns self for chaining.
    pub fn inspect<F>(self, f: F) -> Self
    where
        F: FnOnce(&T),
    {
        if let Ok(ref value) = self.inner {
            f(value);
        }
        self
    }

    /// Executes a closure with the contained error if Err, returns self for chaining.
    pub fn inspect_err<F>(self, f: F) -> Self
    where
        F: FnOnce(&AudioSampleError),
    {
        if let Err(ref error) = self.inner {
            f(error);
        }
        self
    }
}

impl<T> From<AudioSampleResult<T>> for ChainableResult<T> {
    fn from(result: AudioSampleResult<T>) -> Self {
        Self { inner: result }
    }
}

impl<T> From<ChainableResult<T>> for AudioSampleResult<T> {
    fn from(chainable: ChainableResult<T>) -> Self {
        chainable.inner
    }
}

/// Trait for converting AudioSampleResult into ChainableResult.
pub trait IntoChainable<T> {
    fn into_chainable(self) -> ChainableResult<T>;
}

impl<T> IntoChainable<T> for AudioSampleResult<T> {
    fn into_chainable(self) -> ChainableResult<T> {
        ChainableResult::from(self)
    }
}

impl IntoChainable<()> for () {
    fn into_chainable(self) -> ChainableResult<()> {
        ChainableResult::ok(())
    }
}

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

    /// Error that occurs when trying to get an ndarray as a slice (slice mut).
    ///
    /// This typically happens when the ndarray is not contiguous and not in standard order
    #[error("Array layout error: {message}")]
    ArrayLayoutError { message: String },

    /// Error that can be used to indicate that the error originates from an Option type.
    ///
    #[error("Option error: {message}")]
    OptionError { message: String },

    /// Error that occurs when trying to modify borrowed audio data.
    ///
    /// This happens when attempting to perform in-place operations on audio data that is borrowed.
    #[error("Cannot modify borrowed audio data: {message}")]
    BorrowedDataError { message: String },

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
    fn test_chainable_result_ok() {
        let result = ChainableResult::ok(42).map(|x| x * 2).map(|x| x + 10);

        assert!(result.is_ok());
        assert_eq!(result.into_result().unwrap(), 94);
    }

    #[test]
    fn test_chainable_result_err() {
        let result = ChainableResult::err(AudioSampleError::InternalError("test".to_string()))
            .map(|x: i32| x * 2)
            .map(|x| x + 10);

        assert!(result.is_err());
        assert!(matches!(
            result.into_result().unwrap_err(),
            AudioSampleError::InternalError(_)
        ));
    }

    #[test]
    fn test_chainable_result_and_then() {
        let success = ChainableResult::ok(10)
            .and_then(|x| Ok(x * 2))
            .and_then(|x| Ok(x + 5));

        assert_eq!(success.into_result().unwrap(), 25);

        let failure: ChainableResult<i32> = ChainableResult::ok(10)
            .and_then(|x| Ok(x * 2))
            .and_then(|_| Err(AudioSampleError::InternalError("failed".to_string())));

        assert!(failure.is_err());
    }

    #[test]
    fn test_chainable_result_chain() {
        let result = ChainableResult::ok(5)
            .chain(|x| ChainableResult::ok(x * 3))
            .chain(|x| ChainableResult::ok(x + 1));

        assert_eq!(result.into_result().unwrap(), 16);
    }

    #[test]
    fn test_chainable_result_logging() {
        let mut logged = false;

        let _result =
            ChainableResult::err(AudioSampleError::InternalError("test error".to_string()))
                .inspect_err(|_| logged = true)
                .map(|x: i32| x + 1);

        assert!(logged);
    }

    #[test]
    fn test_into_chainable_trait() {
        let ok_result: AudioSampleResult<i32> = Ok(42);
        let chainable = ok_result.into_chainable();
        assert_eq!(chainable.into_result().unwrap(), 42);

        let err_result: AudioSampleResult<i32> =
            Err(AudioSampleError::InternalError("test".to_string()));
        let chainable = err_result.into_chainable();
        assert!(chainable.is_err());
    }

    #[test]
    fn test_chainable_result_unwrap_or() {
        let ok_result = ChainableResult::ok(42);
        assert_eq!(ok_result.unwrap_or(0), 42);

        let err_result = ChainableResult::err(AudioSampleError::InternalError("test".to_string()));
        assert_eq!(err_result.unwrap_or(100), 100);
    }

    #[test]
    fn test_chainable_result_functional_style() {
        // Demonstrate fluent functional-style error handling
        let process_audio = |value: i32| -> ChainableResult<String> {
            ChainableResult::ok(value)
                .map(|x| x * 2) // Scale
                .and_then(|x| {
                    // Validate range
                    if x > 100 {
                        Err(AudioSampleError::InvalidRange(
                            "Value too large".to_string(),
                        ))
                    } else {
                        Ok(x)
                    }
                })
                .map(|x| format!("Processed: {}", x)) // Convert to string
                .inspect(|s| println!("Success: {}", s)) // Log success
                .log_on_error("Audio processing failed") // Log any errors
        };

        let success = process_audio(25);
        assert!(success.is_ok());
        assert_eq!(success.into_result().unwrap(), "Processed: 50");

        let failure = process_audio(75); // Will be 150 after scaling, > 100
        assert!(failure.is_err());
    }
}
