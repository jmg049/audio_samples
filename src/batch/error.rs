//! Error types for batch processing operations.

use crate::AudioSampleError;
use thiserror::Error;

/// Result type for batch operations.
pub type BatchResult<T> = Result<T, BatchError>;

/// Errors that can occur during batch processing.
#[derive(Error, Debug)]
pub enum BatchError {
    /// An error occurred while processing a specific item in the batch.
    #[error("Error processing item {index}: {source}")]
    ItemError {
        /// The index of the item that failed.
        index: usize,
        /// The underlying error.
        source: AudioSampleError,
    },

    /// Multiple errors occurred during batch processing.
    #[error("Multiple errors occurred during batch processing: {count} errors")]
    MultipleErrors {
        /// The number of errors that occurred.
        count: usize,
        /// The individual errors with their indices.
        errors: Vec<(usize, AudioSampleError)>,
    },

    /// Invalid batch configuration.
    #[error("Invalid batch configuration: {message}")]
    InvalidConfiguration { message: String },

    /// I/O error during batch processing.
    #[error("I/O error during batch processing: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },

    /// Parallel processing error.
    #[cfg(feature = "parallel-processing")]
    #[error("Parallel processing error: {message}")]
    ParallelError { message: String },

    /// Progress tracking error.
    #[cfg(feature = "progress-tracking")]
    #[error("Progress tracking error: {message}")]
    ProgressError { message: String },
}

impl BatchError {
    /// Create a new item error.
    pub fn item_error(index: usize, source: AudioSampleError) -> Self {
        Self::ItemError { index, source }
    }

    /// Create a new multiple errors error.
    pub fn multiple_errors(errors: Vec<(usize, AudioSampleError)>) -> Self {
        let count = errors.len();
        Self::MultipleErrors { count, errors }
    }

    /// Create a new invalid configuration error.
    pub fn invalid_configuration(message: impl Into<String>) -> Self {
        Self::InvalidConfiguration {
            message: message.into(),
        }
    }

    /// Create a new parallel processing error.
    #[cfg(feature = "parallel-processing")]
    pub fn parallel_error(message: impl Into<String>) -> Self {
        Self::ParallelError {
            message: message.into(),
        }
    }

    /// Create a new progress tracking error.
    #[cfg(feature = "progress-tracking")]
    pub fn progress_error(message: impl Into<String>) -> Self {
        Self::ProgressError {
            message: message.into(),
        }
    }
}

/// Configuration for error handling in batch operations.
#[derive(Debug, Clone)]
pub enum ErrorHandling {
    /// Stop processing on the first error encountered.
    StopOnFirstError,
    /// Continue processing and collect all errors.
    CollectErrors,
    /// Continue processing but ignore errors.
    IgnoreErrors,
}

impl Default for ErrorHandling {
    fn default() -> Self {
        Self::StopOnFirstError
    }
}
