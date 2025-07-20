//! Parallel processing utilities for batch operations.
//!
//! This module provides parallel processing capabilities for batch operations,
//! leveraging rayon for efficient multi-threaded audio processing.

use super::{BatchError, BatchOperation, BatchResult};
use crate::{AudioSample, AudioSamples};
use rayon::prelude::*;

/// Parallel batch processor for applying operations to multiple audio samples concurrently.
pub struct ParallelProcessor;

impl ParallelProcessor {
    /// Process multiple audio samples in parallel using the provided operation.
    ///
    /// # Arguments
    /// * `items` - Vector of audio samples to process
    /// * `operation` - Batch operation to apply to each sample
    ///
    /// # Returns
    /// Vector of processed audio samples or error if processing fails
    pub fn process_parallel<T, Op>(
        items: Vec<AudioSamples<T>>,
        operation: &Op,
    ) -> BatchResult<Vec<AudioSamples<T>>>
    where
        T: AudioSample + Send + Sync,
        Op: BatchOperation<T> + Sync,
    {
        let results: Result<Vec<_>, _> = items
            .into_par_iter()
            .map(|mut item| operation.apply_to_item(&mut item).map(|_| item))
            .collect();

        match results {
            Ok(processed) => Ok(processed),
            Err(e) => Err(BatchError::parallel_error(format!(
                "Parallel processing failed: {}",
                e
            ))),
        }
    }

    /// Process items in parallel chunks with a specified chunk size.
    ///
    /// This is useful for controlling memory usage and balancing parallelism
    /// with memory constraints.
    ///
    /// # Arguments
    /// * `items` - Vector of audio samples to process
    /// * `operation` - Batch operation to apply to each sample
    /// * `chunk_size` - Size of chunks to process in parallel
    ///
    /// # Returns
    /// Vector of processed audio samples or error if processing fails
    pub fn process_chunked<T, Op>(
        items: Vec<AudioSamples<T>>,
        operation: &Op,
        chunk_size: usize,
    ) -> BatchResult<Vec<AudioSamples<T>>>
    where
        T: AudioSample + Send + Sync,
        Op: BatchOperation<T> + Sync,
    {
        let mut results = Vec::with_capacity(items.len());

        for chunk in items.chunks(chunk_size) {
            let chunk_results: Result<Vec<_>, _> = chunk
                .par_iter()
                .map(|item| {
                    let mut item_copy = item.clone();
                    operation.apply_to_item(&mut item_copy).map(|_| item_copy)
                })
                .collect();

            match chunk_results {
                Ok(mut processed) => results.append(&mut processed),
                Err(e) => {
                    return Err(BatchError::parallel_error(format!(
                        "Chunked parallel processing failed: {}",
                        e
                    )));
                }
            }
        }

        Ok(results)
    }

    /// Process items with a custom parallel iterator configuration.
    ///
    /// This allows fine-grained control over the parallel processing behavior.
    ///
    /// # Arguments
    /// * `items` - Vector of audio samples to process
    /// * `operation` - Batch operation to apply to each sample
    /// * `thread_count` - Number of threads to use (None for default)
    ///
    /// # Returns
    /// Vector of processed audio samples or error if processing fails
    pub fn process_with_threads<T, Op>(
        items: Vec<AudioSamples<T>>,
        operation: &Op,
        thread_count: Option<usize>,
    ) -> BatchResult<Vec<AudioSamples<T>>>
    where
        T: AudioSample + Send + Sync,
        Op: BatchOperation<T> + Sync,
    {
        let pool = if let Some(threads) = thread_count {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| {
                    BatchError::parallel_error(format!("Thread pool creation failed: {}", e))
                })?
        } else {
            rayon::ThreadPoolBuilder::new().build().map_err(|e| {
                BatchError::parallel_error(format!("Thread pool creation failed: {}", e))
            })?
        };

        let results = pool.install(|| {
            items
                .into_par_iter()
                .map(|mut item| operation.apply_to_item(&mut item).map(|_| item))
                .collect::<Result<Vec<_>, _>>()
        });

        match results {
            Ok(processed) => Ok(processed),
            Err(e) => Err(BatchError::parallel_error(format!(
                "Thread pool processing failed: {}",
                e
            ))),
        }
    }
}

/// Utility function to determine optimal chunk size based on available memory and CPU cores.
///
/// # Arguments
/// * `total_items` - Total number of items to process
/// * `estimated_item_size` - Estimated memory size per item in bytes
/// * `max_memory_mb` - Maximum memory to use in megabytes
///
/// # Returns
/// Recommended chunk size
pub fn optimal_chunk_size(
    total_items: usize,
    estimated_item_size: usize,
    max_memory_mb: usize,
) -> usize {
    let cpu_cores = num_cpus::get();
    let max_memory_bytes = max_memory_mb * 1024 * 1024;

    // Calculate chunk size based on memory constraints
    let memory_based_chunk_size = max_memory_bytes / (estimated_item_size * cpu_cores);

    // Calculate chunk size based on work distribution
    let work_based_chunk_size = (total_items + cpu_cores - 1) / cpu_cores;

    // Use the smaller of the two, but ensure it's at least 1
    std::cmp::max(
        1,
        std::cmp::min(memory_based_chunk_size, work_based_chunk_size),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::operations::BatchScale;
    use ndarray::Array1;

    #[test]
    fn test_parallel_processing() {
        let items = vec![
            AudioSamples::new_mono(Array1::from_vec(vec![1.0f32, 2.0, 3.0]), 44100),
            AudioSamples::new_mono(Array1::from_vec(vec![4.0f32, 5.0, 6.0]), 44100),
        ];

        let operation = BatchScale::new(2.0);
        let results = ParallelProcessor::process_parallel(items, &operation).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].as_mono().unwrap()[0], 2.0);
        assert_eq!(results[1].as_mono().unwrap()[0], 8.0);
    }

    #[test]
    fn test_chunked_processing() {
        let items = vec![
            AudioSamples::new_mono(Array1::from_vec(vec![1.0f32, 2.0]), 44100),
            AudioSamples::new_mono(Array1::from_vec(vec![3.0f32, 4.0]), 44100),
            AudioSamples::new_mono(Array1::from_vec(vec![5.0f32, 6.0]), 44100),
        ];

        let operation = BatchScale::new(0.5);
        let results = ParallelProcessor::process_chunked(items, &operation, 2).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].as_mono().unwrap()[0], 0.5);
        assert_eq!(results[2].as_mono().unwrap()[0], 2.5);
    }

    #[test]
    fn test_optimal_chunk_size() {
        let chunk_size = optimal_chunk_size(1000, 1024, 100);
        assert!(chunk_size > 0);
        assert!(chunk_size <= 1000);
    }
}
