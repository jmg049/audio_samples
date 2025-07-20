//! Core traits for batch processing operations.
//!
//! This module defines the fundamental traits that enable batch processing
//! of audio samples with composable operations.

use super::error::BatchResult;
use crate::{AudioSample, AudioSampleResult, AudioSamples};

/// Core trait for batch processing operations.
///
/// This trait defines the interface for processing multiple audio samples
/// in a batch operation. Implementations can choose to process sequentially
/// or in parallel based on their specific needs.
pub trait BatchProcessor<T: AudioSample> {
    /// Process a batch of audio samples.
    ///
    /// This method takes a vector of audio samples and processes them
    /// according to the specific implementation's logic.
    ///
    /// # Arguments
    /// * `items` - Vector of audio samples to process
    ///
    /// # Returns
    /// A BatchResult containing the processed audio samples or an error
    ///
    /// # Example
    /// ```rust,ignore
    /// use audio_samples::batch::*;
    ///
    /// let processor = MyBatchProcessor::new();
    /// let results = processor.process_batch(audio_samples)?;
    /// ```
    fn process_batch(&mut self, items: Vec<AudioSamples<T>>) -> BatchResult<Vec<AudioSamples<T>>>;
}

/// Trait for individual batch operations that can be composed.
///
/// This trait defines a single operation that can be applied to an audio sample
/// as part of a batch processing pipeline. Operations implementing this trait
/// can be combined and reused across different batch processing contexts.
pub trait BatchOperation<T: AudioSample>: Send + Sync {
    /// Apply this operation to a single audio sample.
    ///
    /// This method modifies the audio sample in-place according to the
    /// operation's specific logic.
    ///
    /// # Arguments
    /// * `item` - The audio sample to modify
    ///
    /// # Returns
    /// A result indicating success or failure
    ///
    /// # Example
    /// ```rust,ignore
    /// use audio_samples::batch::*;
    ///
    /// let operation = BatchNormalize::new();
    /// operation.apply_to_item(&mut audio_sample)?;
    /// ```
    fn apply_to_item(&self, item: &mut AudioSamples<T>) -> AudioSampleResult<()>;

    /// Get a description of this operation for logging/debugging.
    ///
    /// This method returns a human-readable description of the operation
    /// that can be used for progress tracking and debugging.
    ///
    /// # Returns
    /// A string describing the operation
    fn description(&self) -> &str;

    /// Check if this operation can be applied in parallel.
    ///
    /// Some operations may have dependencies or side effects that prevent
    /// them from being applied in parallel. This method allows operations
    /// to specify their parallelization capabilities.
    ///
    /// # Returns
    /// true if the operation can be applied in parallel, false otherwise
    fn can_parallelize(&self) -> bool {
        true // Most operations can be parallelized by default
    }

    /// Estimate the computational cost of this operation.
    ///
    /// This method returns a relative cost estimate that can be used by
    /// batch processors to balance workloads and optimize scheduling.
    ///
    /// # Returns
    /// A cost estimate (higher values indicate more expensive operations)
    fn cost_estimate(&self) -> f64 {
        1.0 // Default cost
    }
}

/// Trait for batch operations that can be configured.
///
/// This trait extends BatchOperation to provide configuration capabilities
/// for operations that need parameters or settings.
pub trait ConfigurableBatchOperation<T: AudioSample, C>: BatchOperation<T> {
    /// Create a new instance with the given configuration.
    ///
    /// # Arguments
    /// * `config` - Configuration parameters for the operation
    ///
    /// # Returns
    /// A new instance of the operation with the specified configuration
    fn with_config(config: C) -> Self;

    /// Get the current configuration.
    ///
    /// # Returns
    /// The current configuration of the operation
    fn config(&self) -> &C;

    /// Update the configuration.
    ///
    /// # Arguments
    /// * `config` - New configuration parameters
    fn set_config(&mut self, config: C);
}

/// Trait for batch operations that can validate their inputs.
///
/// This trait allows operations to check if they can be applied to a given
/// audio sample before processing, enabling early error detection.
pub trait ValidatingBatchOperation<T: AudioSample>: BatchOperation<T> {
    /// Check if this operation can be applied to the given audio sample.
    ///
    /// # Arguments
    /// * `item` - The audio sample to validate
    ///
    /// # Returns
    /// Ok(()) if the operation can be applied, Err otherwise
    fn validate(&self, item: &AudioSamples<T>) -> AudioSampleResult<()>;
}

/// Trait for batch operations that can provide progress information.
///
/// This trait allows operations to report their progress during processing,
/// which can be useful for long-running operations.
pub trait ProgressReportingBatchOperation<T: AudioSample>: BatchOperation<T> {
    /// Apply the operation with progress reporting.
    ///
    /// # Arguments
    /// * `item` - The audio sample to modify
    /// * `progress_callback` - Callback function to report progress
    ///
    /// # Returns
    /// A result indicating success or failure
    fn apply_with_progress<F>(
        &self,
        item: &mut AudioSamples<T>,
        progress_callback: F,
    ) -> AudioSampleResult<()>
    where
        F: Fn(f64) + Send + Sync;
}

/// Trait for batch operations that can be chained together.
///
/// This trait allows operations to be composed into more complex processing
/// pipelines while maintaining type safety and performance.
pub trait ChainableBatchOperation<T: AudioSample>: BatchOperation<T> + Sized {
    /// Chain this operation with another operation.
    ///
    /// # Arguments
    /// * `other` - The operation to chain after this one
    ///
    /// # Returns
    /// A new operation that applies both operations in sequence
    fn chain<O: BatchOperation<T>>(self, other: O) -> ChainedBatchOperation<T, Self, O> {
        ChainedBatchOperation::new(self, other)
    }
}

/// A batch operation that applies two operations in sequence.
///
/// This struct represents the composition of two batch operations,
/// applying them in the order they were chained.
pub struct ChainedBatchOperation<T: AudioSample, A: BatchOperation<T>, B: BatchOperation<T>> {
    first: A,
    second: B,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: AudioSample, A: BatchOperation<T>, B: BatchOperation<T>> ChainedBatchOperation<T, A, B> {
    /// Create a new chained operation.
    ///
    /// # Arguments
    /// * `first` - The first operation to apply
    /// * `second` - The second operation to apply
    ///
    /// # Returns
    /// A new chained operation
    pub fn new(first: A, second: B) -> Self {
        Self {
            first,
            second,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: AudioSample, A: BatchOperation<T>, B: BatchOperation<T>> BatchOperation<T>
    for ChainedBatchOperation<T, A, B>
{
    fn apply_to_item(&self, item: &mut AudioSamples<T>) -> AudioSampleResult<()> {
        self.first.apply_to_item(item)?;
        self.second.apply_to_item(item)?;
        Ok(())
    }

    fn description(&self) -> &str {
        "Chained Operation"
    }

    fn can_parallelize(&self) -> bool {
        self.first.can_parallelize() && self.second.can_parallelize()
    }

    fn cost_estimate(&self) -> f64 {
        self.first.cost_estimate() + self.second.cost_estimate()
    }
}

// Blanket implementation for all BatchOperation types
impl<T: AudioSample, O: BatchOperation<T>> ChainableBatchOperation<T> for O {}

/// Trait for batch operations that can be conditionally applied.
///
/// This trait allows operations to be applied only when certain conditions
/// are met, enabling more flexible processing pipelines.
pub trait ConditionalBatchOperation<T: AudioSample>: BatchOperation<T> {
    /// Check if the operation should be applied to the given audio sample.
    ///
    /// # Arguments
    /// * `item` - The audio sample to check
    ///
    /// # Returns
    /// true if the operation should be applied, false otherwise
    fn should_apply(&self, item: &AudioSamples<T>) -> bool;

    /// Apply the operation conditionally.
    ///
    /// # Arguments
    /// * `item` - The audio sample to potentially modify
    ///
    /// # Returns
    /// A result indicating success or failure
    fn apply_conditionally(&self, item: &mut AudioSamples<T>) -> AudioSampleResult<()> {
        if self.should_apply(item) {
            self.apply_to_item(item)
        } else {
            Ok(())
        }
    }
}

/// Trait for batch operations that can be applied in parallel.
///
/// This trait provides a default implementation for parallel processing
/// that can be overridden for operations that need custom parallel logic.
#[cfg(feature = "parallel-processing")]
pub trait ParallelBatchOperation<T: AudioSample>: BatchOperation<T> + Send + Sync {
    /// Apply the operation to multiple items in parallel.
    ///
    /// # Arguments
    /// * `items` - Vector of audio samples to process in parallel
    ///
    /// # Returns
    /// A result indicating success or failure
    fn apply_parallel(&self, items: &mut [AudioSamples<T>]) -> BatchResult<()> {
        use rayon::prelude::*;

        let results: Vec<AudioSampleResult<()>> = items
            .par_iter_mut()
            .map(|item| self.apply_to_item(item))
            .collect();

        // Collect any errors
        let errors: Vec<_> = results
            .into_iter()
            .enumerate()
            .filter_map(|(i, result)| result.err().map(|e| (i, e)))
            .collect();

        if errors.is_empty() {
            Ok(())
        } else if errors.len() == 1 {
            let (index, error) = errors.into_iter().next().unwrap();
            Err(super::error::BatchError::item_error(index, error))
        } else {
            Err(super::error::BatchError::multiple_errors(errors))
        }
    }
}

// Blanket implementation for all BatchOperation types when parallel processing is enabled
#[cfg(feature = "parallel-processing")]
impl<T: AudioSample, O: BatchOperation<T> + Send + Sync> ParallelBatchOperation<T> for O {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioSamples;
    use ndarray::array;

    // Test implementation of BatchOperation
    struct TestOperation {
        scale: f32,
    }

    impl TestOperation {
        fn new(scale: f32) -> Self {
            Self { scale }
        }
    }

    impl BatchOperation<f32> for TestOperation {
        fn apply_to_item(&self, item: &mut AudioSamples<f32>) -> AudioSampleResult<()> {
            item.apply(|sample| sample * self.scale)
        }

        fn description(&self) -> &str {
            "Test scaling operation"
        }

        fn cost_estimate(&self) -> f64 {
            0.1 // Very cheap operation
        }
    }

    #[test]
    fn test_batch_operation_apply() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut audio = AudioSamples::new_mono(data, 44100);

        let operation = TestOperation::new(2.0);
        operation.apply_to_item(&mut audio).unwrap();

        let expected = array![2.0f32, 4.0, 6.0, 8.0, 10.0];
        assert_eq!(audio.as_mono().unwrap(), &expected);
    }

    #[test]
    fn test_chained_operations() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut audio = AudioSamples::new_mono(data, 44100);

        let operation1 = TestOperation::new(2.0);
        let operation2 = TestOperation::new(3.0);
        let chained = operation1.chain(operation2);

        chained.apply_to_item(&mut audio).unwrap();

        let expected = array![6.0f32, 12.0, 18.0, 24.0, 30.0]; // 1*2*3, 2*2*3, etc.
        assert_eq!(audio.as_mono().unwrap(), &expected);
    }

    #[test]
    fn test_operation_properties() {
        let operation = TestOperation::new(1.5);

        assert_eq!(operation.description(), "Test scaling operation");
        assert_eq!(operation.can_parallelize(), true);
        assert_eq!(operation.cost_estimate(), 0.1);
    }

    #[test]
    fn test_chained_operation_properties() {
        let operation1 = TestOperation::new(2.0);
        let operation2 = TestOperation::new(3.0);
        let chained = operation1.chain(operation2);

        assert_eq!(chained.description(), "Chained Operation");
        assert_eq!(chained.can_parallelize(), true);
        assert_eq!(chained.cost_estimate(), 0.2); // 0.1 + 0.1
    }

    #[cfg(feature = "parallel-processing")]
    #[test]
    fn test_parallel_operation() {
        let data1 = array![1.0f32, 2.0, 3.0];
        let data2 = array![4.0f32, 5.0, 6.0];
        let mut audio1 = AudioSamples::new_mono(data1, 44100);
        let mut audio2 = AudioSamples::new_mono(data2, 44100);

        let operation = TestOperation::new(2.0);
        let mut items = vec![audio1, audio2];

        operation.apply_parallel(&mut items).unwrap();

        let expected1 = array![2.0f32, 4.0, 6.0];
        let expected2 = array![8.0f32, 10.0, 12.0];

        assert_eq!(items[0].as_mono().unwrap(), &expected1);
        assert_eq!(items[1].as_mono().unwrap(), &expected2);
    }
}
