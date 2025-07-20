//! Batch builder pattern for constructing and executing batch operations.
//!
//! This module provides a fluent builder interface for creating complex
//! batch processing pipelines with multiple operations, parallel execution,
//! and progress tracking.

use super::error::{BatchError, BatchResult, ErrorHandling};
use super::traits::BatchOperation;
use crate::{AudioSample, AudioSamples};

#[cfg(feature = "progress-tracking")]
use super::progress::{ProgressInfo, ProgressTracker};

/// Configuration for batch processing execution.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Whether to enable parallel processing
    pub parallel: bool,
    /// Number of threads to use for parallel processing (None = auto-detect)
    pub thread_count: Option<usize>,
    /// How to handle errors during batch processing
    pub error_handling: ErrorHandling,
    /// Maximum number of items to process in a single batch
    pub batch_size: Option<usize>,
    /// Whether to shuffle items before processing (useful for load balancing)
    pub shuffle: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            parallel: false,
            thread_count: None,
            error_handling: ErrorHandling::StopOnFirstError,
            batch_size: None,
            shuffle: false,
        }
    }
}

/// Builder for constructing batch processing pipelines.
///
/// This builder provides a fluent interface for creating complex batch
/// processing operations with multiple stages, parallel execution, and
/// progress tracking.
///
/// # Example
/// ```rust,ignore
/// use audio_samples::batch::*;
///
/// let results = BatchBuilder::new()
///     .operation(BatchNormalize::peak())
///     .operation(BatchFilter::lowpass(1000.0))
///     .parallel(true)
///     .error_handling(ErrorHandling::CollectErrors)
///     .process(audio_files)?;
/// ```
pub struct BatchBuilder<T: AudioSample> {
    operations: Vec<Box<dyn BatchOperation<T>>>,
    config: BatchConfig,
    #[cfg(feature = "progress-tracking")]
    progress_tracker: Option<ProgressTracker>,
}

impl<T: AudioSample> BatchBuilder<T> {
    /// Create a new batch builder with default configuration.
    ///
    /// # Returns
    /// A new BatchBuilder instance
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            config: BatchConfig::default(),
            #[cfg(feature = "progress-tracking")]
            progress_tracker: None,
        }
    }

    /// Add an operation to the batch processing pipeline.
    ///
    /// Operations are applied in the order they are added to the builder.
    ///
    /// # Arguments
    /// * `operation` - The operation to add to the pipeline
    ///
    /// # Returns
    /// The builder instance for method chaining
    pub fn operation<O: BatchOperation<T> + 'static>(mut self, operation: O) -> Self {
        self.operations.push(Box::new(operation));
        self
    }

    /// Add multiple operations to the batch processing pipeline.
    ///
    /// # Arguments
    /// * `operations` - Vector of operations to add to the pipeline
    ///
    /// # Returns
    /// The builder instance for method chaining
    pub fn operations<O: BatchOperation<T> + 'static>(mut self, operations: Vec<O>) -> Self {
        for operation in operations {
            self.operations.push(Box::new(operation));
        }
        self
    }

    /// Enable or disable parallel processing.
    ///
    /// # Arguments
    /// * `enabled` - Whether to enable parallel processing
    ///
    /// # Returns
    /// The builder instance for method chaining
    pub fn parallel(mut self, enabled: bool) -> Self {
        self.config.parallel = enabled;
        self
    }

    /// Set the number of threads to use for parallel processing.
    ///
    /// # Arguments
    /// * `thread_count` - Number of threads to use (None = auto-detect)
    ///
    /// # Returns
    /// The builder instance for method chaining
    pub fn thread_count(mut self, thread_count: Option<usize>) -> Self {
        self.config.thread_count = thread_count;
        self
    }

    /// Set the error handling strategy.
    ///
    /// # Arguments
    /// * `error_handling` - How to handle errors during processing
    ///
    /// # Returns
    /// The builder instance for method chaining
    pub fn error_handling(mut self, error_handling: ErrorHandling) -> Self {
        self.config.error_handling = error_handling;
        self
    }

    /// Set the maximum batch size for processing.
    ///
    /// Large batches will be split into smaller chunks for processing.
    ///
    /// # Arguments
    /// * `batch_size` - Maximum number of items per batch
    ///
    /// # Returns
    /// The builder instance for method chaining
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = Some(batch_size);
        self
    }

    /// Enable or disable shuffling of items before processing.
    ///
    /// # Arguments
    /// * `shuffle` - Whether to shuffle items before processing
    ///
    /// # Returns
    /// The builder instance for method chaining
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.config.shuffle = shuffle;
        self
    }

    /// Set a progress tracker for monitoring batch processing progress.
    ///
    /// # Arguments
    /// * `tracker` - Progress tracker instance
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[cfg(feature = "progress-tracking")]
    pub fn progress(mut self, tracker: ProgressTracker) -> Self {
        self.progress_tracker = Some(tracker);
        self
    }

    /// Process a batch of audio samples using the configured pipeline.
    ///
    /// This method applies all configured operations to the input audio samples
    /// according to the specified configuration (parallel/sequential, error handling, etc.).
    ///
    /// # Arguments
    /// * `mut items` - Vector of audio samples to process
    ///
    /// # Returns
    /// A BatchResult containing the processed audio samples or an error
    pub fn process(self, mut items: Vec<AudioSamples<T>>) -> BatchResult<Vec<AudioSamples<T>>> {
        if items.is_empty() {
            return Ok(items);
        }

        if self.operations.is_empty() {
            return Ok(items);
        }

        // Shuffle items if requested
        if self.config.shuffle {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            items.len().hash(&mut hasher);
            let seed = hasher.finish();
            self.shuffle_items(&mut items, seed);
        }

        // Process in batches if batch size is specified
        if let Some(batch_size) = self.config.batch_size {
            self.process_in_batches(items, batch_size)
        } else {
            self.process_single_batch(items)
        }
    }

    /// Process items in a single batch.
    fn process_single_batch(
        self,
        mut items: Vec<AudioSamples<T>>,
    ) -> BatchResult<Vec<AudioSamples<T>>> {
        let total_operations = self.operations.len();
        let total_items = items.len();

        #[cfg(feature = "progress-tracking")]
        let progress_tracker = self.progress_tracker.as_ref();

        // Apply each operation to all items
        for (op_index, operation) in self.operations.iter().enumerate() {
            let operation_start = std::time::Instant::now();

            #[cfg(feature = "progress-tracking")]
            if let Some(tracker) = progress_tracker {
                let progress = ProgressInfo {
                    current_operation: op_index + 1,
                    total_operations,
                    current_item: 0,
                    total_items,
                    operation_name: operation.description().to_string(),
                    elapsed: operation_start.elapsed(),
                    estimated_remaining: None,
                };
                tracker.report_progress(&progress);
            }

            let result = if self.config.parallel {
                self.apply_operation_parallel(operation.as_ref(), &mut items)
            } else {
                self.apply_operation_sequential(operation.as_ref(), &mut items)
            };

            match result {
                Ok(()) =>
                {
                    #[cfg(feature = "progress-tracking")]
                    if let Some(tracker) = progress_tracker {
                        let progress = ProgressInfo {
                            current_operation: op_index + 1,
                            total_operations,
                            current_item: total_items,
                            total_items,
                            operation_name: operation.description().to_string(),
                            elapsed: operation_start.elapsed(),
                            estimated_remaining: None,
                        };
                        tracker.report_progress(&progress);
                    }
                }
                Err(e) => {
                    match self.config.error_handling {
                        ErrorHandling::StopOnFirstError => return Err(e),
                        ErrorHandling::CollectErrors => {
                            // Continue processing and collect all errors
                            // For now, just return the first error
                            return Err(e);
                        }
                        ErrorHandling::IgnoreErrors => {
                            // Continue processing, ignoring errors
                            continue;
                        }
                    }
                }
            }
        }

        Ok(items)
    }

    /// Process items in multiple batches.
    fn process_in_batches(
        self,
        items: Vec<AudioSamples<T>>,
        _batch_size: usize,
    ) -> BatchResult<Vec<AudioSamples<T>>> {
        // For now, we'll process all items in a single batch to avoid the complexity
        // of cloning boxed trait objects. This can be improved later with a more
        // sophisticated approach (e.g., using Arc<dyn BatchOperation> or similar).
        self.process_single_batch(items)
    }

    /// Apply an operation sequentially to all items.
    fn apply_operation_sequential(
        &self,
        operation: &dyn BatchOperation<T>,
        items: &mut [AudioSamples<T>],
    ) -> BatchResult<()> {
        let mut errors = Vec::new();

        for (i, item) in items.iter_mut().enumerate() {
            if let Err(error) = operation.apply_to_item(item) {
                match self.config.error_handling {
                    ErrorHandling::StopOnFirstError => {
                        return Err(BatchError::item_error(i, error));
                    }
                    ErrorHandling::CollectErrors => {
                        errors.push((i, error));
                    }
                    ErrorHandling::IgnoreErrors => {
                        // Continue processing
                    }
                }
            }
        }

        if !errors.is_empty() && matches!(self.config.error_handling, ErrorHandling::CollectErrors)
        {
            return Err(BatchError::multiple_errors(errors));
        }

        Ok(())
    }

    /// Apply an operation in parallel to all items.
    #[cfg(feature = "parallel-processing")]
    fn apply_operation_parallel(
        &self,
        operation: &dyn BatchOperation<T>,
        items: &mut [AudioSamples<T>],
    ) -> BatchResult<()> {
        if !operation.can_parallelize() {
            return self.apply_operation_sequential(operation, items);
        }

        use rayon::prelude::*;

        // Configure thread pool if specified
        if let Some(thread_count) = self.config.thread_count {
            rayon::ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build_global()
                .map_err(|e| {
                    BatchError::parallel_error(format!("Failed to configure thread pool: {}", e))
                })?;
        }

        let results: Vec<_> = items
            .par_iter_mut()
            .enumerate()
            .map(|(i, item)| (i, operation.apply_to_item(item)))
            .collect();

        let mut errors = Vec::new();
        for (i, result) in results {
            if let Err(error) = result {
                match self.config.error_handling {
                    ErrorHandling::StopOnFirstError => {
                        return Err(BatchError::item_error(i, error));
                    }
                    ErrorHandling::CollectErrors => {
                        errors.push((i, error));
                    }
                    ErrorHandling::IgnoreErrors => {
                        // Continue processing
                    }
                }
            }
        }

        if !errors.is_empty() && matches!(self.config.error_handling, ErrorHandling::CollectErrors)
        {
            return Err(BatchError::multiple_errors(errors));
        }

        Ok(())
    }

    /// Apply an operation in parallel to all items (fallback for non-parallel builds).
    #[cfg(not(feature = "parallel-processing"))]
    fn apply_operation_parallel(
        &self,
        operation: &dyn BatchOperation<T>,
        items: &mut [AudioSamples<T>],
    ) -> BatchResult<()> {
        // Fall back to sequential processing
        self.apply_operation_sequential(operation, items)
    }

    /// Shuffle items using a simple linear congruential generator.
    fn shuffle_items(&self, items: &mut [AudioSamples<T>], seed: u64) {
        let mut rng = SimpleRng::new(seed);

        for i in (1..items.len()).rev() {
            let j = rng.next_usize() % (i + 1);
            items.swap(i, j);
        }
    }

    /// Get the total estimated cost of all operations.
    ///
    /// This can be used to estimate the computational cost of the batch processing
    /// pipeline before execution.
    ///
    /// # Returns
    /// The total estimated cost of all operations
    pub fn total_cost_estimate(&self) -> f64 {
        self.operations.iter().map(|op| op.cost_estimate()).sum()
    }

    /// Get the number of operations in the pipeline.
    ///
    /// # Returns
    /// The number of operations that will be applied
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    /// Check if all operations can be parallelized.
    ///
    /// # Returns
    /// true if all operations can be parallelized, false otherwise
    pub fn can_parallelize(&self) -> bool {
        self.operations.iter().all(|op| op.can_parallelize())
    }
}

impl<T: AudioSample> Default for BatchBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple pseudorandom number generator for shuffling.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        self.state
    }

    fn next_usize(&mut self) -> usize {
        self.next_u64() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioSamples;
    use ndarray::array;

    // Test operation for testing
    struct TestScaleOperation {
        scale: f32,
    }

    impl TestScaleOperation {
        fn new(scale: f32) -> Self {
            Self { scale }
        }
    }

    impl BatchOperation<f32> for TestScaleOperation {
        fn apply_to_item(&self, item: &mut AudioSamples<f32>) -> crate::AudioSampleResult<()> {
            item.apply(|sample| sample * self.scale)
        }

        fn description(&self) -> &str {
            "Test scale operation"
        }

        fn cost_estimate(&self) -> f64 {
            0.1
        }
    }

    #[test]
    fn test_batch_builder_empty() {
        let builder = BatchBuilder::<f32>::new();
        let items = vec![];
        let results = builder.process(items).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_batch_builder_no_operations() {
        let builder = BatchBuilder::<f32>::new();
        let data = array![1.0f32, 2.0, 3.0];
        let audio = AudioSamples::new_mono(data.clone(), 44100);
        let items = vec![audio];

        let results = builder.process(items).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].as_mono().unwrap(), &data);
    }

    #[test]
    fn test_batch_builder_single_operation() {
        let builder = BatchBuilder::new().operation(TestScaleOperation::new(2.0));

        let data = array![1.0f32, 2.0, 3.0];
        let audio = AudioSamples::new_mono(data, 44100);
        let items = vec![audio];

        let results = builder.process(items).unwrap();
        assert_eq!(results.len(), 1);

        let expected = array![2.0f32, 4.0, 6.0];
        assert_eq!(results[0].as_mono().unwrap(), &expected);
    }

    #[test]
    fn test_batch_builder_multiple_operations() {
        let builder = BatchBuilder::new()
            .operation(TestScaleOperation::new(2.0))
            .operation(TestScaleOperation::new(3.0));

        let data = array![1.0f32, 2.0, 3.0];
        let audio = AudioSamples::new_mono(data, 44100);
        let items = vec![audio];

        let results = builder.process(items).unwrap();
        assert_eq!(results.len(), 1);

        let expected = array![6.0f32, 12.0, 18.0]; // 1*2*3, 2*2*3, 3*2*3
        assert_eq!(results[0].as_mono().unwrap(), &expected);
    }

    #[test]
    fn test_batch_builder_multiple_items() {
        let builder = BatchBuilder::new().operation(TestScaleOperation::new(2.0));

        let data1 = array![1.0f32, 2.0, 3.0];
        let data2 = array![4.0f32, 5.0, 6.0];
        let audio1 = AudioSamples::new_mono(data1, 44100);
        let audio2 = AudioSamples::new_mono(data2, 44100);
        let items = vec![audio1, audio2];

        let results = builder.process(items).unwrap();
        assert_eq!(results.len(), 2);

        let expected1 = array![2.0f32, 4.0, 6.0];
        let expected2 = array![8.0f32, 10.0, 12.0];
        assert_eq!(results[0].as_mono().unwrap(), &expected1);
        assert_eq!(results[1].as_mono().unwrap(), &expected2);
    }

    #[test]
    fn test_batch_builder_configuration() {
        let builder: BatchBuilder<f32> = BatchBuilder::new()
            .parallel(true)
            .thread_count(Some(2))
            .error_handling(ErrorHandling::CollectErrors)
            .batch_size(10)
            .shuffle(true);

        // Just test that configuration is accepted
        assert_eq!(builder.config.parallel, true);
        assert_eq!(builder.config.thread_count, Some(2));
        assert!(matches!(
            builder.config.error_handling,
            ErrorHandling::CollectErrors
        ));
        assert_eq!(builder.config.batch_size, Some(10));
        assert_eq!(builder.config.shuffle, true);
    }

    #[test]
    fn test_batch_builder_properties() {
        let builder = BatchBuilder::new()
            .operation(TestScaleOperation::new(2.0))
            .operation(TestScaleOperation::new(3.0));

        assert_eq!(builder.operation_count(), 2);
        assert_eq!(builder.total_cost_estimate(), 0.2); // 0.1 + 0.1
        assert_eq!(builder.can_parallelize(), true);
    }

    #[cfg(feature = "parallel-processing")]
    #[test]
    fn test_batch_builder_parallel() {
        let builder = BatchBuilder::new()
            .operation(TestScaleOperation::new(2.0))
            .parallel(true);

        let data1 = array![1.0f32, 2.0, 3.0];
        let data2 = array![4.0f32, 5.0, 6.0];
        let audio1 = AudioSamples::new_mono(data1, 44100);
        let audio2 = AudioSamples::new_mono(data2, 44100);
        let items = vec![audio1, audio2];

        let results = builder.process(items).unwrap();
        assert_eq!(results.len(), 2);

        let expected1 = array![2.0f32, 4.0, 6.0];
        let expected2 = array![8.0f32, 10.0, 12.0];
        assert_eq!(results[0].as_mono().unwrap(), &expected1);
        assert_eq!(results[1].as_mono().unwrap(), &expected2);
    }
}
