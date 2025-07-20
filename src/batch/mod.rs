//! Batch processing functionality for audio samples.
//!
//! This module provides tools for efficiently processing multiple audio samples
//! in batch operations, with support for parallel processing and progress tracking.
//!
//! # Features
//!
//! - **Parallel Processing**: Leverage multiple CPU cores for faster processing
//! - **Progress Tracking**: Monitor progress with real-time updates
//! - **Error Handling**: Graceful handling of partial failures
//! - **Flexible Operations**: Composable batch operations
//!
//! # Example
//!
//! ```rust,ignore
//! use audio_samples::batch::*;
//!
//! let results = BatchBuilder::new()
//!     .operation(BatchNormalize::peak())
//!     .operation(BatchFilter::lowpass(1000.0))
//!     .parallel(true)
//!     .process(audio_files)?;
//! ```

#[cfg(feature = "batch-processing")]
pub mod traits;

#[cfg(feature = "batch-processing")]
pub mod builder;

#[cfg(feature = "batch-processing")]
pub mod operations;

#[cfg(feature = "batch-processing")]
pub mod error;

#[cfg(all(feature = "batch-processing", feature = "progress-tracking"))]
pub mod progress;

#[cfg(all(feature = "batch-processing", feature = "parallel-processing"))]
pub mod parallel;

// TODO: Implement batch I/O operations
// #[cfg(feature = "batch-processing")]
// pub mod io;

// Re-export main types for convenience
#[cfg(feature = "batch-processing")]
pub use builder::BatchBuilder;

#[cfg(feature = "batch-processing")]
pub use traits::{BatchOperation, BatchProcessor};

#[cfg(feature = "batch-processing")]
pub use operations::*;

#[cfg(feature = "batch-processing")]
pub use error::{BatchError, BatchResult};

#[cfg(all(feature = "batch-processing", feature = "progress-tracking"))]
pub use progress::{ProgressInfo, ProgressTracker};
