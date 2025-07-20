//! Progress tracking for batch operations.
//!
//! This module provides progress tracking capabilities for batch processing
//! operations, allowing users to monitor the progress of long-running tasks.

use std::sync::Arc;
use std::time::{Duration, Instant};

#[cfg(feature = "progress-tracking")]
use indicatif::{ProgressBar, ProgressStyle};

/// Information about the current progress of a batch operation.
#[derive(Debug, Clone)]
pub struct ProgressInfo {
    /// Current operation being executed (1-based)
    pub current_operation: usize,
    /// Total number of operations in the pipeline
    pub total_operations: usize,
    /// Current item being processed (0-based)
    pub current_item: usize,
    /// Total number of items to process
    pub total_items: usize,
    /// Name of the current operation
    pub operation_name: String,
    /// Time elapsed since the start of the current operation
    pub elapsed: Duration,
    /// Estimated time remaining for the current operation
    pub estimated_remaining: Option<Duration>,
}

impl ProgressInfo {
    /// Get the overall progress as a percentage (0.0 to 1.0).
    pub fn overall_progress(&self) -> f64 {
        if self.total_operations == 0 || self.total_items == 0 {
            return 0.0;
        }

        let operation_progress = (self.current_operation - 1) as f64 / self.total_operations as f64;
        let item_progress = self.current_item as f64 / self.total_items as f64;
        let current_operation_progress = item_progress / self.total_operations as f64;

        operation_progress + current_operation_progress
    }

    /// Get the current operation progress as a percentage (0.0 to 1.0).
    pub fn operation_progress(&self) -> f64 {
        if self.total_items == 0 {
            return 0.0;
        }

        self.current_item as f64 / self.total_items as f64
    }

    /// Get the estimated total time for completion.
    pub fn estimated_total_time(&self) -> Option<Duration> {
        if self.current_item == 0 || self.total_items == 0 {
            return None;
        }

        let items_per_second = self.current_item as f64 / self.elapsed.as_secs_f64();
        if items_per_second <= 0.0 {
            return None;
        }

        let total_seconds = self.total_items as f64 / items_per_second;
        Some(Duration::from_secs_f64(total_seconds))
    }
}

/// Trait for reporting progress during batch operations.
pub trait ProgressReporter: Send + Sync {
    /// Report progress information.
    fn report_progress(&self, info: &ProgressInfo);

    /// Report that the operation has started.
    fn start(&self, total_operations: usize, total_items: usize) {
        let info = ProgressInfo {
            current_operation: 1,
            total_operations,
            current_item: 0,
            total_items,
            operation_name: "Starting".to_string(),
            elapsed: Duration::from_secs(0),
            estimated_remaining: None,
        };
        self.report_progress(&info);
    }

    /// Report that the operation has completed.
    fn finish(&self, elapsed: Duration) {
        // Default implementation does nothing
        let _ = elapsed;
    }
}

/// Console-based progress reporter.
#[derive(Debug)]
pub struct ConsoleProgressReporter {
    verbose: bool,
}

impl ConsoleProgressReporter {
    /// Create a new console progress reporter.
    pub fn new(verbose: bool) -> Self {
        Self { verbose }
    }
}

impl ProgressReporter for ConsoleProgressReporter {
    fn report_progress(&self, info: &ProgressInfo) {
        if self.verbose {
            println!(
                "Operation {}/{}: {} - Item {}/{} ({:.1}% complete, {:.1}s elapsed)",
                info.current_operation,
                info.total_operations,
                info.operation_name,
                info.current_item,
                info.total_items,
                info.overall_progress() * 100.0,
                info.elapsed.as_secs_f64()
            );
        } else {
            print!(
                "\rProcessing: {:.1}% complete",
                info.overall_progress() * 100.0
            );
            use std::io::{self, Write};
            io::stdout().flush().unwrap();
        }
    }

    fn finish(&self, elapsed: Duration) {
        if self.verbose {
            println!(
                "Batch processing completed in {:.2}s",
                elapsed.as_secs_f64()
            );
        } else {
            println!(
                "\nBatch processing completed in {:.2}s",
                elapsed.as_secs_f64()
            );
        }
    }
}

/// Progress bar-based progress reporter using indicatif.
#[cfg(feature = "progress-tracking")]
#[derive(Debug)]
pub struct ProgressBarReporter {
    bar: ProgressBar,
    start_time: Instant,
}

#[cfg(feature = "progress-tracking")]
impl ProgressBarReporter {
    /// Create a new progress bar reporter.
    pub fn new(total_items: usize) -> Self {
        let bar = ProgressBar::new(total_items as u64);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg} ({eta})")
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("#>-")
        );

        Self {
            bar,
            start_time: Instant::now(),
        }
    }

    /// Create a new progress bar reporter with custom style.
    pub fn with_style(total_items: usize, style: ProgressStyle) -> Self {
        let bar = ProgressBar::new(total_items as u64);
        bar.set_style(style);

        Self {
            bar,
            start_time: Instant::now(),
        }
    }
}

#[cfg(feature = "progress-tracking")]
impl ProgressReporter for ProgressBarReporter {
    fn report_progress(&self, info: &ProgressInfo) {
        self.bar.set_position(info.current_item as u64);
        self.bar.set_message(format!(
            "{} (op {}/{})",
            info.operation_name, info.current_operation, info.total_operations
        ));
    }

    fn start(&self, _total_operations: usize, _total_items: usize) {
        // Progress bar is already initialized
    }

    fn finish(&self, _elapsed: Duration) {
        self.bar.finish_with_message("Batch processing completed");
    }
}

/// Callback-based progress reporter.
#[derive(Debug)]
pub struct CallbackProgressReporter<F> {
    callback: F,
}

impl<F> CallbackProgressReporter<F>
where
    F: Fn(&ProgressInfo) + Send + Sync,
{
    /// Create a new callback progress reporter.
    pub fn new(callback: F) -> Self {
        Self { callback }
    }
}

impl<F> ProgressReporter for CallbackProgressReporter<F>
where
    F: Fn(&ProgressInfo) + Send + Sync,
{
    fn report_progress(&self, info: &ProgressInfo) {
        (self.callback)(info);
    }
}

/// Multi-reporter that sends progress to multiple reporters.
#[derive(Debug)]
pub struct MultiProgressReporter {
    reporters: Vec<Box<dyn ProgressReporter>>,
}

impl MultiProgressReporter {
    /// Create a new multi-reporter.
    pub fn new() -> Self {
        Self {
            reporters: Vec::new(),
        }
    }

    /// Add a reporter to the multi-reporter.
    pub fn add_reporter<R: ProgressReporter + 'static>(mut self, reporter: R) -> Self {
        self.reporters.push(Box::new(reporter));
        self
    }
}

impl Default for MultiProgressReporter {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressReporter for MultiProgressReporter {
    fn report_progress(&self, info: &ProgressInfo) {
        for reporter in &self.reporters {
            reporter.report_progress(info);
        }
    }

    fn start(&self, total_operations: usize, total_items: usize) {
        for reporter in &self.reporters {
            reporter.start(total_operations, total_items);
        }
    }

    fn finish(&self, elapsed: Duration) {
        for reporter in &self.reporters {
            reporter.finish(elapsed);
        }
    }
}

/// Progress tracker that manages progress reporting for batch operations.
pub struct ProgressTracker {
    reporter: Arc<dyn ProgressReporter>,
    start_time: Option<Instant>,
}

impl ProgressTracker {
    /// Create a new progress tracker with a custom reporter.
    pub fn new<R: ProgressReporter + 'static>(reporter: R) -> Self {
        Self {
            reporter: Arc::new(reporter),
            start_time: None,
        }
    }

    /// Create a console progress tracker.
    pub fn console(verbose: bool) -> Self {
        Self::new(ConsoleProgressReporter::new(verbose))
    }

    /// Create a progress bar tracker.
    #[cfg(feature = "progress-tracking")]
    pub fn progress_bar(total_items: usize) -> Self {
        Self::new(ProgressBarReporter::new(total_items))
    }

    /// Create a callback progress tracker.
    pub fn callback<F>(callback: F) -> Self
    where
        F: Fn(&ProgressInfo) + Send + Sync + 'static,
    {
        Self::new(CallbackProgressReporter::new(callback))
    }

    /// Create a multi-reporter progress tracker.
    pub fn multi() -> MultiProgressReporter {
        MultiProgressReporter::new()
    }

    /// Start tracking progress.
    pub fn start(&mut self, total_operations: usize, total_items: usize) {
        self.start_time = Some(Instant::now());
        self.reporter.start(total_operations, total_items);
    }

    /// Report progress.
    pub fn report_progress(&self, info: &ProgressInfo) {
        self.reporter.report_progress(info);
    }

    /// Finish tracking progress.
    pub fn finish(&self) {
        if let Some(start_time) = self.start_time {
            self.reporter.finish(start_time.elapsed());
        }
    }
}

impl std::fmt::Debug for ProgressTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProgressTracker")
            .field("start_time", &self.start_time)
            .finish()
    }
}

/// Null progress reporter that does nothing.
#[derive(Debug)]
pub struct NullProgressReporter;

impl ProgressReporter for NullProgressReporter {
    fn report_progress(&self, _info: &ProgressInfo) {
        // Do nothing
    }
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self::new(NullProgressReporter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_progress_info_calculation() {
        let info = ProgressInfo {
            current_operation: 2,
            total_operations: 4,
            current_item: 50,
            total_items: 100,
            operation_name: "Test".to_string(),
            elapsed: Duration::from_secs(10),
            estimated_remaining: None,
        };

        assert_eq!(info.operation_progress(), 0.5);
        assert_eq!(info.overall_progress(), 0.375); // (1/4) + (0.5/4) = 0.375
    }

    #[test]
    fn test_console_reporter() {
        let reporter = ConsoleProgressReporter::new(true);
        let info = ProgressInfo {
            current_operation: 1,
            total_operations: 2,
            current_item: 25,
            total_items: 100,
            operation_name: "Test Operation".to_string(),
            elapsed: Duration::from_secs(5),
            estimated_remaining: None,
        };

        // This test just ensures the reporter doesn't panic
        reporter.report_progress(&info);
        reporter.finish(Duration::from_secs(10));
    }

    #[test]
    fn test_callback_reporter() {
        let received_info = Arc::new(Mutex::new(None));
        let received_info_clone = Arc::clone(&received_info);

        let reporter = CallbackProgressReporter::new(move |info: &ProgressInfo| {
            *received_info_clone.lock().unwrap() = Some(info.clone());
        });

        let info = ProgressInfo {
            current_operation: 1,
            total_operations: 1,
            current_item: 10,
            total_items: 20,
            operation_name: "Test".to_string(),
            elapsed: Duration::from_secs(1),
            estimated_remaining: None,
        };

        reporter.report_progress(&info);

        let received = received_info.lock().unwrap();
        assert!(received.is_some());
        let received = received.as_ref().unwrap();
        assert_eq!(received.current_item, 10);
        assert_eq!(received.total_items, 20);
    }

    #[test]
    fn test_multi_reporter() {
        let counter1 = Arc::new(Mutex::new(0));
        let counter2 = Arc::new(Mutex::new(0));

        let counter1_clone = Arc::clone(&counter1);
        let counter2_clone = Arc::clone(&counter2);

        let multi = MultiProgressReporter::new()
            .add_reporter(CallbackProgressReporter::new(move |_| {
                *counter1_clone.lock().unwrap() += 1;
            }))
            .add_reporter(CallbackProgressReporter::new(move |_| {
                *counter2_clone.lock().unwrap() += 1;
            }));

        let info = ProgressInfo {
            current_operation: 1,
            total_operations: 1,
            current_item: 5,
            total_items: 10,
            operation_name: "Test".to_string(),
            elapsed: Duration::from_secs(1),
            estimated_remaining: None,
        };

        multi.report_progress(&info);

        assert_eq!(*counter1.lock().unwrap(), 1);
        assert_eq!(*counter2.lock().unwrap(), 1);
    }

    #[test]
    fn test_progress_tracker() {
        let mut tracker = ProgressTracker::console(false);
        tracker.start(2, 10);

        let info = ProgressInfo {
            current_operation: 1,
            total_operations: 2,
            current_item: 5,
            total_items: 10,
            operation_name: "Test".to_string(),
            elapsed: Duration::from_secs(1),
            estimated_remaining: None,
        };

        tracker.report_progress(&info);
        tracker.finish();
    }

    #[test]
    fn test_null_reporter() {
        let reporter = NullProgressReporter;
        let info = ProgressInfo {
            current_operation: 1,
            total_operations: 1,
            current_item: 1,
            total_items: 1,
            operation_name: "Test".to_string(),
            elapsed: Duration::from_secs(1),
            estimated_remaining: None,
        };

        // Should not panic
        reporter.report_progress(&info);
        reporter.start(1, 1);
        reporter.finish(Duration::from_secs(1));
    }
}
