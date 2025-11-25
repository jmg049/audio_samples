//! Hierarchical error types and result utilities for audio sample operations.
//!
//! This module provides an error hierarchy designed around the core
//! failure modes of audio processing operations. The hierarchy separates concerns
//! into distinct categories to enable precise error handling and clear debugging.
//!
//! # Error Hierarchy
//!
//! ```text
//! AudioSampleError
//! ├── Conversion(ConversionError)     - Type conversion and casting failures
//! ├── Parameter(ParameterError)       - Invalid parameters and configuration
//! ├── Layout(LayoutError)            - Memory layout and array structure issues
//! ├── Processing(ProcessingError)     - Audio processing operation failures
//! └── Feature(FeatureError)          - Missing feature dependencies
//! ```
//!
//! # Usage Examples
//!
//! ```rust
//! use audio_samples::{AudioSampleError, ConversionError, ParameterError};
//!
//! // Handle specific error types
//! match audio_result {
//!     Err(AudioSampleError::Conversion(conv_err)) => {
//!         eprintln!("Failed to convert {} to {}", conv_err.source_type, conv_err.target_type);
//!     }
//!     Err(AudioSampleError::Parameter(param_err)) => {
//!         eprintln!("Invalid parameter '{}': {}", param_err.parameter, param_err.reason);
//!     }
//!     Ok(value) => { /* success */ }
//!     _ => { /* other errors */ }
//! }
//! ```

use thiserror::Error;

/// Convenience type alias for results that may contain AudioSampleError
pub type AudioSampleResult<T> = Result<T, AudioSampleError>;

/// Main error type for audio sample operations.
///
/// This is the root error type that encompasses all possible failures
/// in audio processing operations. It's designed as a hierarchy to allow
/// both generic error handling and specific error inspection.
///
/// # Migration Note
///
/// This enum now uses a hierarchical structure but maintains backward
/// compatibility through deprecated variants that forward to the new system.
#[derive(Error, Debug, Clone)]
pub enum AudioSampleError {
    /// Errors related to type conversions and casting operations.
    #[error(transparent)]
    Conversion(#[from] ConversionError),

    /// Errors related to invalid parameters or configuration.
    #[error(transparent)]
    Parameter(#[from] ParameterError),

    /// Errors related to memory layout, array structure, or data organization.
    #[error(transparent)]
    Layout(#[from] LayoutError),

    /// Errors that occur during audio processing operations.
    #[error(transparent)]
    Processing(#[from] ProcessingError),

    /// Errors related to missing or disabled features.
    #[error(transparent)]
    Feature(#[from] FeatureError),

    #[cfg(feature = "plotting")]
    /// Errors related to plotting operations.
    /// This variant is only available when the "plotting" feature is enabled.
    #[error(transparent)]
    Plotting(#[from] PlottingError),

    #[cfg(feature = "serialization")]
    /// Errors related to serialization and deserialization operations.
    /// This variant is only available when the "serialization" feature is enabled.
    #[error(transparent)]
    Serialization(#[from] SerializationError),

}

/// Errors that occur during type conversion and casting operations.
///
/// This covers both audio-aware conversions (e.g., i16 ↔ f32 with normalization)
/// and raw numeric casting operations between sample types.
#[derive(Error, Debug, Clone)]
pub enum ConversionError {
    /// Failed to convert between audio sample types with audio-aware scaling.
    #[error("Failed to convert sample value {value} from {source_type} to {target_type}: {reason}")]
    AudioConversion {
        /// The sample value that failed to convert (as string for display).
        value: String,
        /// Source sample type name.
        source_type: String,
        /// Target sample type name.
        target_type: String,
        /// Detailed reason for the conversion failure.
        reason: String,
    },

    /// Failed to perform raw numeric casting between types.
    #[error("Failed to cast value {value} from {source_type} to {target_type}: {reason}")]
    NumericCast {
        /// The numeric value that failed to cast (as string for display).
        value: String,
        /// Source numeric type name.
        source_type: String,
        /// Target numeric type name.
        target_type: String,
        /// Detailed reason for the casting failure.
        reason: String,
    },

    /// The conversion operation is not supported between the specified types.
    #[error("Conversion from {source_type} to {target_type} is not supported")]
    UnsupportedConversion {
        /// Source type name.
        source_type: String,
        /// Target type name.
        target_type: String,
    },
}

/// Errors related to invalid parameters, ranges, or configuration values.
///
/// This covers validation failures for user-provided parameters to audio
/// processing operations, including range violations and invalid configurations.
#[derive(Error, Debug, Clone)]
pub enum ParameterError {
    /// A parameter value is outside the valid range for the operation.
    #[error(
        "Parameter '{parameter}' value {value} is outside valid range [{min}, {max}]: {reason}"
    )]
    OutOfRange {
        /// Name of the parameter that was out of range.
        parameter: String,
        /// The invalid value (as string for display).
        value: String,
        /// Minimum valid value (as string for display).
        min: String,
        /// Maximum valid value (as string for display).
        max: String,
        /// Additional context about why this range is required.
        reason: String,
    },

    /// A parameter has an invalid value that doesn't meet operation requirements.
    #[error("Invalid value for parameter '{parameter}': {reason}")]
    InvalidValue {
        /// Name of the parameter with invalid value.
        parameter: String,
        /// Detailed explanation of why the value is invalid.
        reason: String,
    },

    /// Required parameters are missing or empty.
    #[error("Required parameter '{parameter}' is missing or empty")]
    Missing {
        /// Name of the missing parameter.
        parameter: String,
    },

    /// A configuration object contains conflicting or invalid settings.
    #[error("Invalid configuration for {operation}: {reason}")]
    InvalidConfiguration {
        /// The operation or component being configured.
        operation: String,
        /// Detailed explanation of the configuration problem.
        reason: String,
    },
}

/// Errors related to memory layout, array structure, and data organization.
///
/// This covers failures related to array contiguity, dimension mismatches,
/// borrowed data mutation attempts, and other structural issues.
#[derive(Error, Debug, Clone)]
pub enum LayoutError {
    /// Array data is not contiguous in memory when contiguous layout is required.
    #[error(
        "Array layout error: {operation} requires contiguous memory layout, but array is {layout_type}"
    )]
    NonContiguous {
        /// The operation that requires contiguous layout.
        operation: String,
        /// Description of the actual layout (e.g., "strided", "non-standard order").
        layout_type: String,
    },

    /// Array dimensions don't match expected values for the operation.
    #[error(
        "Dimension mismatch: expected {expected_dims}, got {actual_dims} for operation '{operation}'"
    )]
    DimensionMismatch {
        /// Description of expected dimensions.
        expected_dims: String,
        /// Description of actual dimensions.
        actual_dims: String,
        /// The operation that has dimension requirements.
        operation: String,
    },

    /// Attempted to modify borrowed audio data that should be immutable.
    #[error("Cannot modify borrowed audio data in {operation}: {reason}")]
    BorrowedDataMutation {
        /// The operation that attempted to modify borrowed data.
        operation: String,
        /// Explanation of why modification is not allowed.
        reason: String,
    },

    /// Input data is empty when the operation requires non-empty data.
    #[error("Empty input data provided to {operation}")]
    EmptyData {
        /// The operation that received empty data.
        operation: String,
    },

    /// Data size or format is incompatible with the operation requirements.
    #[error("Incompatible data format for {operation}: {reason}")]
    IncompatibleFormat {
        /// The operation with format requirements.
        operation: String,
        /// Detailed explanation of the format incompatibility.
        reason: String,
    },
}

/// Errors that occur during audio processing operations.
///
/// This covers failures in DSP algorithms, mathematical operations,
/// and other processing-specific issues that don't fit into other categories.
#[derive(Error, Debug, Clone)]
pub enum ProcessingError {
    /// A mathematical operation failed due to invalid input or numerical issues.
    #[error("Mathematical operation '{operation}' failed: {reason}")]
    MathematicalFailure {
        /// The mathematical operation that failed.
        operation: String,
        /// Detailed explanation of the mathematical failure.
        reason: String,
    },

    /// An audio processing algorithm encountered an error during execution.
    #[error("Audio processing algorithm '{algorithm}' failed: {reason}")]
    AlgorithmFailure {
        /// The processing algorithm that failed.
        algorithm: String,
        /// Detailed explanation of the algorithm failure.
        reason: String,
    },

    /// The operation failed due to insufficient data or resources.
    #[error("Insufficient data for {operation}: {reason}")]
    InsufficientData {
        /// The operation that requires more data.
        operation: String,
        /// Explanation of the data requirements.
        reason: String,
    },

    /// A convergence-based algorithm failed to converge within limits.
    #[error("Algorithm '{algorithm}' failed to converge after {iterations} iterations")]
    ConvergenceFailure {
        /// The algorithm that failed to converge.
        algorithm: String,
        /// Number of iterations attempted.
        iterations: u32,
    },

    /// An external dependency or resource required for processing is unavailable.
    #[error("External dependency '{dependency}' required for {operation} is unavailable: {reason}")]
    ExternalDependency {
        /// The external dependency that's unavailable.
        dependency: String,
        /// The operation that requires the dependency.
        operation: String,
        /// Reason why the dependency is unavailable.
        reason: String,
    },
}

/// Errors related to missing or disabled cargo features.
///
/// This covers cases where optional functionality was used but the required
/// cargo feature wasn't enabled at compile time.
#[derive(Error, Debug, Clone)]
pub enum FeatureError {
    /// A cargo feature is required but not enabled.
    #[error(
        "Feature '{feature}' is required for {operation} but not enabled. Enable with: --features {feature}"
    )]
    NotEnabled {
        /// The cargo feature that needs to be enabled.
        feature: String,
        /// The operation that requires this feature.
        operation: String,
    },

    /// Multiple features are required but some are missing.
    #[error(
        "Operation '{operation}' requires features [{required_features}], but only [{enabled_features}] are enabled"
    )]
    MultipleRequired {
        /// The operation requiring multiple features.
        operation: String,
        /// Comma-separated list of required features.
        required_features: String,
        /// Comma-separated list of currently enabled features.
        enabled_features: String,
    },

    /// A feature is enabled but its dependencies are not properly configured.
    #[error("Feature '{feature}' is enabled but misconfigured: {reason}")]
    Misconfigured {
        /// The misconfigured feature.
        feature: String,
        /// Explanation of the configuration issue.
        reason: String,
    },
}

#[cfg(feature = "plotting")]
/// Errors related to plotting operations.
#[derive(Error, Clone, Debug)]
pub enum PlottingError {
    
    /// Failed to create a plot due to invalid data or configuration.
    #[error("Failed to create plot: {reason}")]
    PlotCreation {
        /// Detailed reason for the plot creation failure.
        reason: String,
    },

    /// Rendering the plot to a file or display failed.
    #[error("Failed to render plot: {reason}")]
    PlotRendering {
        /// Detailed reason for the rendering failure.
        reason: String,
    },

    #[error("html_view error: {0}")]
    HtmlViewError(#[from] html_view::ViewerError)

}

#[cfg(feature = "serialization")]
/// Errors related to serialization and deserialization operations.
///
/// This covers failures in data format conversion, I/O operations,
/// and format-specific encoding/decoding issues.
#[derive(Error, Debug, Clone)]
pub enum SerializationError {
    /// Failed to serialize audio data to the specified format.
    #[error("Failed to serialize to {format}: {reason}")]
    SerializationFailed {
        /// The target format that failed
        format: String,
        /// Detailed reason for the serialization failure
        reason: String,
    },

    /// Failed to deserialize audio data from the specified format.
    #[error("Failed to deserialize from {format}: {reason}")]
    DeserializationFailed {
        /// The source format that failed to deserialize
        format: String,
        /// Detailed reason for the deserialization failure
        reason: String,
    },

    /// Input/output error during file operations.
    #[error("IO error during {operation}: {reason}")]
    IoError {
        /// The I/O operation that failed
        operation: String,
        /// Detailed explanation of the I/O failure
        reason: String,
    },

    /// The requested format is not supported for the operation.
    #[error("Unsupported format {format} for {operation}")]
    UnsupportedFormat {
        /// The unsupported format
        format: String,
        /// The operation that doesn't support this format
        operation: String,
    },

    /// Failed to automatically detect the file format.
    #[error("Format detection failed: {reason}")]
    FormatDetectionFailed {
        /// Reason why format detection failed
        reason: String,
    },

    /// Data validation failed after serialization round-trip.
    #[error("Data validation failed after round-trip: {reason}")]
    ValidationFailed {
        /// Details about what validation failed
        reason: String,
    },

    /// The file header or metadata is invalid or corrupted.
    #[error("Invalid or corrupted {component}: {reason}")]
    InvalidHeader {
        /// The component that has invalid data (e.g., "header", "metadata")
        component: String,
        /// Details about the invalid data
        reason: String,
    },

    /// A required dependency for the format is not available.
    #[error("Missing dependency '{dependency}' for {format} format: {reason}")]
    MissingDependency {
        /// The missing dependency
        dependency: String,
        /// The format that requires this dependency
        format: String,
        /// Additional context about the dependency requirement
        reason: String,
    },
}

// Convenience constructors for common error patterns
impl ConversionError {
    /// Create a new audio conversion error.
    pub fn audio_conversion<V, S, T, R>(value: V, source_type: S, target_type: T, reason: R) -> Self
    where
        V: ToString,
        S: ToString,
        T: ToString,
        R: ToString,
    {
        Self::AudioConversion {
            value: value.to_string(),
            source_type: source_type.to_string(),
            target_type: target_type.to_string(),
            reason: reason.to_string(),
        }
    }

    /// Create a new numeric cast error.
    pub fn numeric_cast<V, S, T, R>(value: V, source_type: S, target_type: T, reason: R) -> Self
    where
        V: ToString,
        S: ToString,
        T: ToString,
        R: ToString,
    {
        Self::NumericCast {
            value: value.to_string(),
            source_type: source_type.to_string(),
            target_type: target_type.to_string(),
            reason: reason.to_string(),
        }
    }
}

impl ParameterError {
    /// Create a new out-of-range parameter error.
    pub fn out_of_range<P, V, Min, Max, R>(
        parameter: P,
        value: V,
        min: Min,
        max: Max,
        reason: R,
    ) -> Self
    where
        P: ToString,
        V: ToString,
        Min: ToString,
        Max: ToString,
        R: ToString,
    {
        Self::OutOfRange {
            parameter: parameter.to_string(),
            value: value.to_string(),
            min: min.to_string(),
            max: max.to_string(),
            reason: reason.to_string(),
        }
    }

    /// Create a new invalid value parameter error.
    pub fn invalid_value<P, R>(parameter: P, reason: R) -> Self
    where
        P: ToString,
        R: ToString,
    {
        Self::InvalidValue {
            parameter: parameter.to_string(),
            reason: reason.to_string(),
        }
    }
}

impl LayoutError {
    /// Create a new borrowed data mutation error.
    pub fn borrowed_mutation<O, R>(operation: O, reason: R) -> Self
    where
        O: ToString,
        R: ToString,
    {
        Self::BorrowedDataMutation {
            operation: operation.to_string(),
            reason: reason.to_string(),
        }
    }

    /// Create a new dimension mismatch error.
    pub fn dimension_mismatch<E, A, O>(expected: E, actual: A, operation: O) -> Self
    where
        E: ToString,
        A: ToString,
        O: ToString,
    {
        Self::DimensionMismatch {
            expected_dims: expected.to_string(),
            actual_dims: actual.to_string(),
            operation: operation.to_string(),
        }
    }
}

impl ProcessingError {
    /// Create a new algorithm failure error.
    pub fn algorithm_failure<A, R>(algorithm: A, reason: R) -> Self
    where
        A: ToString,
        R: ToString,
    {
        Self::AlgorithmFailure {
            algorithm: algorithm.to_string(),
            reason: reason.to_string(),
        }
    }
}

impl FeatureError {
    /// Create a new feature not enabled error.
    pub fn not_enabled<F, O>(feature: F, operation: O) -> Self
    where
        F: ToString,
        O: ToString,
    {
        Self::NotEnabled {
            feature: feature.to_string(),
            operation: operation.to_string(),
        }
    }
}

// Conversion traits for easier error creation
impl From<&str> for ParameterError {
    fn from(msg: &str) -> Self {
        Self::InvalidValue {
            parameter: "unknown".to_string(),
            reason: msg.to_string(),
        }
    }
}

impl From<String> for ParameterError {
    fn from(msg: String) -> Self {
        Self::InvalidValue {
            parameter: "unknown".to_string(),
            reason: msg,
        }
    }
}

impl From<&str> for ProcessingError {
    fn from(msg: &str) -> Self {
        Self::AlgorithmFailure {
            algorithm: "unknown".to_string(),
            reason: msg.to_string(),
        }
    }
}

impl From<String> for ProcessingError {
    fn from(msg: String) -> Self {
        Self::AlgorithmFailure {
            algorithm: "unknown".to_string(),
            reason: msg,
        }
    }
}

#[cfg(feature = "serialization")]
impl SerializationError {
    /// Create a new serialization failed error.
    pub fn serialization_failed<F, R>(format: F, reason: R) -> Self
    where
        F: ToString,
        R: ToString,
    {
        Self::SerializationFailed {
            format: format.to_string(),
            reason: reason.to_string(),
        }
    }

    /// Create a new deserialization failed error.
    pub fn deserialization_failed<F, R>(format: F, reason: R) -> Self
    where
        F: ToString,
        R: ToString,
    {
        Self::DeserializationFailed {
            format: format.to_string(),
            reason: reason.to_string(),
        }
    }

    /// Create a new I/O error.
    pub fn io_error<O, R>(operation: O, reason: R) -> Self
    where
        O: ToString,
        R: ToString,
    {
        Self::IoError {
            operation: operation.to_string(),
            reason: reason.to_string(),
        }
    }

    /// Create a new unsupported format error.
    pub fn unsupported_format<F, O>(format: F, operation: O) -> Self
    where
        F: ToString,
        O: ToString,
    {
        Self::UnsupportedFormat {
            format: format.to_string(),
            operation: operation.to_string(),
        }
    }

    /// Create a new format detection failed error.
    pub fn format_detection_failed<R>(reason: R) -> Self
    where
        R: ToString,
    {
        Self::FormatDetectionFailed {
            reason: reason.to_string(),
        }
    }

    /// Create a new validation failed error.
    pub fn validation_failed<R>(reason: R) -> Self
    where
        R: ToString,
    {
        Self::ValidationFailed {
            reason: reason.to_string(),
        }
    }

    /// Create a new invalid header error.
    pub fn invalid_header<C, R>(component: C, reason: R) -> Self
    where
        C: ToString,
        R: ToString,
    {
        Self::InvalidHeader {
            component: component.to_string(),
            reason: reason.to_string(),
        }
    }

    /// Create a new missing dependency error.
    pub fn missing_dependency<D, F, R>(dependency: D, format: F, reason: R) -> Self
    where
        D: ToString,
        F: ToString,
        R: ToString,
    {
        Self::MissingDependency {
            dependency: dependency.to_string(),
            format: format.to_string(),
            reason: reason.to_string(),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_hierarchy() {
        let conversion_err =
            ConversionError::audio_conversion("32768", "i16", "i8", "Out of range");
        let audio_err = AudioSampleError::Conversion(conversion_err);

        assert!(matches!(audio_err, AudioSampleError::Conversion(_)));
        assert!(format!("{}", audio_err).contains("Failed to convert sample value 32768"));
    }

    #[test]
    fn test_parameter_error() {
        let param_err = ParameterError::out_of_range(
            "cutoff_hz",
            "25000",
            "20",
            "22050",
            "Exceeds Nyquist limit",
        );
        assert!(format!("{}", param_err).contains("cutoff_hz"));
        assert!(format!("{}", param_err).contains("25000"));
        assert!(format!("{}", param_err).contains("22050"));
    }

    #[test]
    fn test_layout_error() {
        let layout_err = LayoutError::borrowed_mutation("slice_mut", "Data is borrowed immutably");
        assert!(format!("{}", layout_err).contains("slice_mut"));
        assert!(format!("{}", layout_err).contains("borrowed"));
    }

    #[test]
    fn test_processing_error() {
        let proc_err = ProcessingError::algorithm_failure("FFT", "Input size must be power of 2");
        assert!(format!("{}", proc_err).contains("FFT"));
        assert!(format!("{}", proc_err).contains("power of 2"));
    }

    #[test]
    fn test_feature_error() {
        let feat_err = FeatureError::not_enabled("fft", "spectral analysis");
        assert!(format!("{}", feat_err).contains("fft"));
        assert!(format!("{}", feat_err).contains("spectral analysis"));
    }

    #[test]
    fn test_result_type_alias() {
        let ok_result: AudioSampleResult<i32> = Ok(42);
        assert_eq!(ok_result.unwrap(), 42);

        let err_result: AudioSampleResult<i32> = Err(AudioSampleError::Parameter(
            ParameterError::invalid_value("test_param", "Invalid for testing"),
        ));
        assert!(err_result.is_err());
    }

    #[test]
    fn test_error_chain_display() {
        let conversion_err = ConversionError::audio_conversion(
            "1.5",
            "f32",
            "i16",
            "Value out of signed 16-bit range",
        );
        let audio_err = AudioSampleError::Conversion(conversion_err);

        let error_string = format!("{}", audio_err);
        assert!(error_string.contains("Failed to convert sample value 1.5"));
        assert!(error_string.contains("f32"));
        assert!(error_string.contains("i16"));
    }

    #[test]
    fn test_error_pattern_matching() {
        let errors = vec![
            AudioSampleError::Conversion(ConversionError::audio_conversion(
                "val", "src", "dst", "reason",
            )),
            AudioSampleError::Parameter(ParameterError::invalid_value("param", "reason")),
            AudioSampleError::Layout(LayoutError::borrowed_mutation("op", "reason")),
            AudioSampleError::Processing(ProcessingError::algorithm_failure("alg", "reason")),
            AudioSampleError::Feature(FeatureError::not_enabled("feat", "op")),
        ];

        for error in errors {
            #[cfg(feature = "plotting")]
            {
                match error {
                    AudioSampleError::Conversion(_) => { /* handle conversion */ }
                    AudioSampleError::Parameter(_) => { /* handle parameter */ }
                    AudioSampleError::Layout(_) => { /* handle layout */ }
                    AudioSampleError::Processing(_) => { /* handle processing */ }
                    AudioSampleError::Feature(_) => { /* handle feature */ } // All deprecated variants have been migrated to new error types
                    AudioSampleError::Plotting(_) => { /* handle plotting */ }
                    #[cfg(feature = "serialization")]
                    AudioSampleError::Serialization(_) => { /* handle serialization */ }
                }
            }
            #[cfg(all(not(feature = "plotting"), not(feature = "serialization")))]
            {
                match error {
                    AudioSampleError::Conversion(_) => { /* handle conversion */ }
                    AudioSampleError::Parameter(_) => { /* handle parameter */ }
                    AudioSampleError::Layout(_) => { /* handle layout */ }
                    AudioSampleError::Processing(_) => { /* handle processing */ }
                    AudioSampleError::Feature(_) => { /* handle feature */ } // All deprecated variants have been migrated to new error types
                }
            }
            #[cfg(all(not(feature = "plotting"), feature = "serialization"))]
            {
                match error {
                    AudioSampleError::Conversion(_) => { /* handle conversion */ }
                    AudioSampleError::Parameter(_) => { /* handle parameter */ }
                    AudioSampleError::Layout(_) => { /* handle layout */ }
                    AudioSampleError::Processing(_) => { /* handle processing */ }
                    AudioSampleError::Feature(_) => { /* handle feature */ } // All deprecated variants have been migrated to new error types
                    AudioSampleError::Serialization(_) => { /* handle serialization */ }
                }
            }
            #[cfg(all(feature = "plotting", not(feature = "serialization")))]
            {
                match error {
                    AudioSampleError::Conversion(_) => { /* handle conversion */ }
                    AudioSampleError::Parameter(_) => { /* handle parameter */ }
                    AudioSampleError::Layout(_) => { /* handle layout */ }
                    AudioSampleError::Processing(_) => { /* handle processing */ }
                    AudioSampleError::Feature(_) => { /* handle feature */ } // All deprecated variants have been migrated to new error types
                    AudioSampleError::Plotting(_) => { /* handle plotting */ }
                }
            }
        }
    }
}