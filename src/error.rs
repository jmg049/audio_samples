//! Hierarchical error types and result utilities for audio sample operations.

//! This module defines [`AudioSampleError`], the root error type for the entire
//! crate, together with five specialised sub-error enums that each cover a
//! distinct failure domain:
//!
//! - [`ConversionError`] — type conversion and casting failures
//! - [`ParameterError`] — invalid parameters and configuration values
//! - [`LayoutError`] — memory layout and array-structure issues
//! - [`ProcessingError`] — DSP algorithm and arithmetic failures
//! - [`FeatureError`] — optional cargo features that are missing or misconfigured
//!
//! ```text
//! AudioSampleError
//! ├── Conversion(ConversionError)     – type conversion and casting failures
//! ├── Parameter(ParameterError)       – invalid parameters and configuration
//! ├── Layout(LayoutError)             – memory layout and array structure issues
//! ├── Processing(ProcessingError)     – audio processing operation failures
//! └── Feature(FeatureError)           – missing feature dependencies
//! ```

//! Grouping errors by domain makes it possible to handle broad categories with a
//! single match arm while still allowing callers to inspect specific failure
//! causes when needed. The sub-enum design keeps `AudioSampleError` `#[non_exhaustive]`
//! so that new variants can be added in future minor versions without breaking
//! downstream code.

//! Import the root type and any sub-types you intend to match on, then propagate
//! with `?` or inspect with a `match` expression:
//!
//! ```rust
//! use audio_samples::{
//!     AudioSampleError, AudioSampleResult, ConversionError, LayoutError, ParameterError,
//! };
//!
//! let audio_result: AudioSampleResult<()> = Err(AudioSampleError::Parameter(
//!     ParameterError::invalid_value("cutoff_hz", "must be > 0"),
//! ));
//!
//! match audio_result {
//!     Err(AudioSampleError::Conversion(ConversionError::AudioConversion {
//!         source_type,
//!         target_type,
//!         reason,
//!         ..
//!     })) => {
//!         eprintln!("Failed to convert {source_type} → {target_type}: {reason}");
//!     }
//!     Err(AudioSampleError::Parameter(ParameterError::InvalidValue { parameter, reason })) => {
//!         eprintln!("Invalid parameter '{parameter}': {reason}");
//!     }
//!     Err(AudioSampleError::Layout(LayoutError::NonContiguous { operation, .. })) => {
//!         eprintln!("Operation '{operation}' requires contiguous audio");
//!     }
//!     Ok(()) => {
//!         // success
//!     }
//!     Err(other) => {
//!         eprintln!("{other}");
//!     }
//! }
//! ```

use thiserror::Error;

/// Convenience alias for any `Result` that may fail with [`AudioSampleError`].
///
/// Every fallible public API in this crate returns `AudioSampleResult<T>`.
/// Results can be propagated with `?` or matched on [`AudioSampleError`] variants
/// directly.
pub type AudioSampleResult<T> = Result<T, AudioSampleError>;

/// Root error type for all audio sample operations.
///
/// # Purpose
///
/// `AudioSampleError` is the single error type returned by every fallible
/// public API in this crate. It groups failures into five sub-domains so that
/// callers can handle broad categories with a single match arm while retaining
/// the ability to inspect specific variants when required.
///
/// # Intended Usage
///
/// Prefer matching on the inner sub-type rather than the outer variant wherever
/// the specific failure domain matters. Propagate errors out of functions using
/// the `?` operator together with the [`AudioSampleResult`] return type.
///
/// # Invariants
///
/// The enum is `#[non_exhaustive]`, meaning new variants may be added in future
/// minor versions. Always include a catch-all arm when matching exhaustively.
///
/// # Feature-gated variants
///
/// The `Spectrogram` variant is only present when the `transforms` feature is
/// enabled. It wraps errors originating from the `spectrograms` crate.
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
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

    /// Errors related to parsing of any of the various audio_samples support types.
    #[error("Failed to parse {type_name}. Context: {context}")]
    Parse {
        /// Type name of the type we failed to parse
        type_name: String,
        /// User provided context
        context: String,
    },

    #[cfg(feature = "transforms")]
    /// Errors originating from the `spectrograms` crate (requires `feature = "transforms"`).
    #[error(transparent)]
    Spectrogram(#[from] spectrograms::SpectrogramError),

    #[error("Empty audio data provided where non-empty data is required")]
    /// Error indicating that audio data is empty when non-empty data is required.
    EmptyData,

    /// Error indicating mismatch between total samples and channel count.
    ///
    /// Occurs when the total number of samples is not evenly divisible by the number of
    /// channels, indicating malformed or corrupted audio data.
    #[error("Invalid number of samples with respect to the number of channels")]
    InvalidNumberOfSamples {
        /// Total number of samples
        total_samples: usize,
        /// Number of channels
        channels: u32,
    },

    /// Error indicating a formatting failure during display or debug output.
    #[error("Fmt error occurred: {0}")]
    Fmt(#[from] std::fmt::Error),

    /// Error indicating an unsupported operation or configuration.
    ///
    /// Used when a requested operation is valid in principle but not implemented for the
    /// given combination of inputs (e.g., certain multi-channel operations, unsupported
    /// sample types, or platform-specific limitations).
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

impl AudioSampleError {
    /// Creates an [crate::AudioSampleError::Unsupported] with the given message.
    ///
    /// Use this when a caller requests an operation that is valid in principle
    /// but not implemented for the given combination of inputs (e.g. multi-channel
    /// STFT).
    ///
    /// # Arguments
    ///
    /// - `msg` — Human-readable description of the unsupported operation.
    ///
    /// # Returns
    ///
    /// `AudioSampleError::Unsupported(msg.to_string())`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::AudioSampleError;
    ///
    /// let err = AudioSampleError::unsupported("multi-channel MFCC is not supported");
    /// assert!(matches!(err, AudioSampleError::Unsupported(_)));
    /// ```
    #[inline]
    pub fn unsupported<S>(msg: S) -> Self
    where
        S: ToString,
    {
        Self::Unsupported(msg.to_string())
    }

    /// Creates an [`AudioSampleError::InvalidNumberOfSamples`] error.
    ///
    /// Use this when `total_samples` cannot be distributed evenly across
    /// `channels`, or when the combination is otherwise incoherent for the
    /// requested layout.
    ///
    /// # Arguments
    ///
    /// - `total_samples` — The total number of interleaved samples provided.
    /// - `channels` — The number of audio channels requested.
    ///
    /// # Returns
    ///
    /// `AudioSampleError::InvalidNumberOfSamples { total_samples, channels }`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::AudioSampleError;
    ///
    /// // 5 samples cannot be divided evenly into 2 channels
    /// let err = AudioSampleError::invalid_number_of_samples(5, 2);
    /// assert!(matches!(err, AudioSampleError::InvalidNumberOfSamples { .. }));
    /// ```
    #[inline]
    #[must_use]
    pub const fn invalid_number_of_samples(total_samples: usize, channels: u32) -> Self {
        Self::InvalidNumberOfSamples {
            total_samples,
            channels,
        }
    }

    /// Creates an [`AudioSampleError::Parse`] error for the type `T`.
    ///
    /// The type name of `T` is captured automatically via
    /// [`std::any::type_name`] and embedded in the returned error for
    /// diagnostic purposes.
    ///
    /// # Arguments
    ///
    /// - `msg` — Human-readable description of the parse failure.
    ///
    /// # Type Parameters
    ///
    /// - `T` — The type that could not be parsed. Its name is recorded in the
    ///   `type_name` field of the returned error.
    ///
    /// # Returns
    ///
    /// `AudioSampleError::Parse { type_name: std::any::type_name::<T>(), context: msg }`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::AudioSampleError;
    ///
    /// let err = AudioSampleError::parse::<f32, _>("expected a finite float");
    /// assert!(matches!(err, AudioSampleError::Parse { .. }));
    /// ```
    #[inline]
    pub fn parse<T, S>(msg: S) -> Self
    where
        S: ToString,
    {
        Self::Parse {
            type_name: std::any::type_name::<T>().to_string(),
            context: msg.to_string(),
        }
    }

    /// Creates an [crate::AudioSampleError::Layout] wrapping a
    /// [`LayoutError::ShapeError`].
    ///
    /// Convenience constructor for shape-related layout errors where the
    /// specific operation name is not available at the call site. The
    /// `operation` field is set to `"unknown"`.
    ///
    /// # Arguments
    ///
    /// - `msg` — Human-readable description of the shape problem.
    ///
    /// # Returns
    ///
    /// `AudioSampleError::Layout(LayoutError::ShapeError { operation: "unknown", info: msg })`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::{AudioSampleError, LayoutError};
    ///
    /// let err = AudioSampleError::layout("expected 2-D array, got 3-D");
    /// assert!(matches!(err, AudioSampleError::Layout(LayoutError::ShapeError { .. })));
    /// ```
    #[inline]
    pub fn layout<S>(msg: S) -> Self
    where
        S: ToString,
    {
        Self::Layout(LayoutError::ShapeError {
            operation: "unknown".to_string(),
            info: msg.to_string(),
        })
    }
}

/// Errors that occur during type conversion and casting operations.
///
/// # Purpose
///
/// Covers both audio-aware conversions (e.g. `i16 ↔ f32` with normalisation)
/// and raw numeric casts between sample types. The distinction between
/// [`AudioConversion`][ConversionError::AudioConversion] and
/// [`NumericCast`][ConversionError::NumericCast] reflects whether the
/// operation understood audio semantics.
///
/// # Intended Usage
///
/// Returned by `to_format`, `to_type`, `cast_as`, and `cast_to`. Prefer the
/// audio-aware variants (`to_format`, `to_type`) over raw casts unless you
/// deliberately need non-normalised bit patterns.
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
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
/// # Purpose
///
/// Covers validation failures for caller-supplied parameters to audio
/// processing operations, including range violations and conflicting
/// configuration fields.
///
/// # Intended Usage
///
/// Return these errors at the boundary between user input and internal logic.
/// Use [`OutOfRange`][ParameterError::OutOfRange] when a numeric bound is
/// violated, [`InvalidValue`][ParameterError::InvalidValue] for other
/// semantic constraints, and
/// [`InvalidConfiguration`][ParameterError::InvalidConfiguration] when multiple
/// fields interact in an unsupported way.
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
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
/// # Purpose
///
/// Covers failures related to array contiguity, dimension mismatches,
/// borrowed data mutation attempts, and other structural issues that arise
/// from the ndarray-backed storage of audio samples.
///
/// # Intended Usage
///
/// Return these errors when an operation's structural preconditions are not
/// met — for example, when a mono-only function receives multi-channel input,
/// or when an in-place operation is attempted on borrowed (non-owned) data.
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
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

    /// Data size or format is incompatible with the operation requirements.
    #[error("Incompatible data format for {operation}: {reason}")]
    IncompatibleFormat {
        /// The operation with format requirements.
        operation: String,
        /// Detailed explanation of the format incompatibility.
        reason: String,
    },

    /// A shape-related error occurred during array operations.
    #[error("Shape error in {operation}: {info}")]
    ShapeError {
        /// The operation that encountered the shape error.
        operation: String,
        /// Detailed information about the shape error.
        info: String,
    },

    /// Error indicating an invalid operation on audio data.
    ///
    /// Occurs when attempting an operation that violates preconditions, such as applying
    /// a function to incompatible audio layouts, mismatched dimensions, or invalid state.
    #[error("Invalid operation on {0}:\nReason: {1}")]
    InvalidOperation(String, String),
}

impl From<ndarray::ShapeError> for AudioSampleError {
    #[inline]
    fn from(err: ndarray::ShapeError) -> Self {
        Self::Layout(LayoutError::ShapeError {
            operation: "ndarray operation".to_string(),
            info: err.to_string(),
        })
    }
}

impl LayoutError {
    /// Creates a [`LayoutError::InvalidOperation`] error.
    ///
    /// Use this when a valid operation is called in a context where it cannot
    /// proceed — for example, requesting an STFT on multi-channel audio.
    ///
    /// # Arguments
    ///
    /// - `operation` — Name of the operation that was called in an invalid context.
    /// - `reason` — Human-readable explanation of why the operation cannot proceed
    ///   (e.g. `"STFT requires mono audio"`).
    ///
    /// # Returns
    ///
    /// `LayoutError::InvalidOperation(operation.to_string(), reason.to_string())`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::LayoutError;
    ///
    /// let err = LayoutError::invalid_operation("stft", "requires mono audio");
    /// assert!(matches!(err, LayoutError::InvalidOperation(_, _)));
    /// ```
    #[inline]
    pub fn invalid_operation<S>(operation: S, reason: S) -> Self
    where
        S: ToString,
    {
        Self::InvalidOperation(operation.to_string(), reason.to_string())
    }
}

/// Errors that occur during audio processing operations.
///
/// # Purpose
///
/// Covers failures in DSP algorithms, mathematical operations, and other
/// processing-specific issues that don't belong in the conversion, parameter,
/// or layout domains.
///
/// # Intended Usage
///
/// Return these errors when a processing algorithm cannot complete successfully.
/// Use [`MathematicalFailure`][ProcessingError::MathematicalFailure] for
/// arithmetic issues, [`AlgorithmFailure`][ProcessingError::AlgorithmFailure]
/// for general algorithm errors, and
/// [`ConvergenceFailure`][ProcessingError::ConvergenceFailure] when an
/// iterative algorithm exhausts its iteration budget.
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
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
/// # Purpose
///
/// Covers cases where optional functionality was invoked but the required
/// cargo feature was not enabled at compile time.
///
/// # Intended Usage
///
/// Return [`NotEnabled`][FeatureError::NotEnabled] from feature-gated code
/// paths that cannot be reached at runtime without the relevant feature flag.
/// This is distinct from a compile error — it allows conditional code paths
/// to provide a clear runtime message rather than silently doing nothing.
#[derive(Error, Debug, Clone)]
#[non_exhaustive]
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

impl ConversionError {
    /// Creates a [`ConversionError::AudioConversion`] error.
    ///
    /// Use this when an audio-aware type conversion fails — for example, when a
    /// normalised float value cannot be represented in the target integer type.
    ///
    /// # Arguments
    ///
    /// - `value` — The sample value that failed to convert (converted to string for display).
    /// - `source_type` — Name of the source sample type (e.g. `"i16"`).
    /// - `target_type` — Name of the target sample type (e.g. `"f32"`).
    /// - `reason` — Human-readable explanation of why the conversion failed.
    ///
    /// # Returns
    ///
    /// A `ConversionError::AudioConversion` variant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::{AudioSampleError, ConversionError};
    ///
    /// let err = ConversionError::audio_conversion("32768", "i16", "i8", "value out of range");
    /// assert!(matches!(err, ConversionError::AudioConversion { .. }));
    ///
    /// // Wraps into the root error type via the From impl.
    /// let audio_err: AudioSampleError = err.into();
    /// assert!(matches!(audio_err, AudioSampleError::Conversion(_)));
    /// ```
    #[inline]
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

    /// Creates a [`ConversionError::NumericCast`] error.
    ///
    /// Use this when a raw numeric cast fails — for example, when casting a
    /// large `f64` value to `u8` would overflow or produce a meaningless bit
    /// pattern.
    ///
    /// # Arguments
    ///
    /// - `value` — The numeric value that failed to cast (converted to string for display).
    /// - `source_type` — Name of the source numeric type (e.g. `"f64"`).
    /// - `target_type` — Name of the target numeric type (e.g. `"u8"`).
    /// - `reason` — Human-readable explanation of why the cast failed.
    ///
    /// # Returns
    ///
    /// A `ConversionError::NumericCast` variant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::ConversionError;
    ///
    /// let err = ConversionError::numeric_cast("300.0", "f64", "u8", "value exceeds 255");
    /// assert!(matches!(err, ConversionError::NumericCast { .. }));
    /// ```
    #[inline]
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
    /// Creates a [`ParameterError::OutOfRange`] error.
    ///
    /// Use this when a caller supplies a numeric value that falls outside the
    /// bounds required by an operation (e.g. a cutoff frequency above the
    /// Nyquist limit).
    ///
    /// # Arguments
    ///
    /// - `parameter` — Name of the parameter that is out of range.
    /// - `value` — The invalid value supplied by the caller (converted to string for display).
    /// - `min` — The minimum valid value, inclusive (converted to string for display).
    /// - `max` — The maximum valid value, inclusive (converted to string for display).
    /// - `reason` — Human-readable explanation of why this range is required.
    ///
    /// # Returns
    ///
    /// A `ParameterError::OutOfRange` variant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::ParameterError;
    ///
    /// let err = ParameterError::out_of_range("cutoff_hz", "25000", "20", "22050", "exceeds Nyquist");
    /// assert!(matches!(err, ParameterError::OutOfRange { .. }));
    /// ```
    #[inline]
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

    /// Creates a [`ParameterError::InvalidValue`] error.
    ///
    /// Use this when a parameter value is semantically invalid in a way that
    /// cannot be described by a numeric range — for example, a window size that
    /// is not a power of two, or a filter order that is too high for stability.
    ///
    /// # Arguments
    ///
    /// - `parameter` — Name of the parameter with the invalid value.
    /// - `reason` — Human-readable explanation of why the value is invalid.
    ///
    /// # Returns
    ///
    /// A `ParameterError::InvalidValue` variant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::ParameterError;
    ///
    /// let err = ParameterError::invalid_value("window_size", "must be a power of two");
    /// assert!(matches!(err, ParameterError::InvalidValue { .. }));
    /// ```
    #[inline]
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
    /// Creates a [`LayoutError::BorrowedDataMutation`] error.
    ///
    /// Use this when an in-place or mutating operation is attempted on audio
    /// data that is held via an immutable borrow and therefore cannot be
    /// modified.
    ///
    /// # Arguments
    ///
    /// - `operation` — Name of the operation that attempted to modify borrowed data.
    /// - `reason` — Human-readable explanation of why modification is not allowed.
    ///
    /// # Returns
    ///
    /// A `LayoutError::BorrowedDataMutation` variant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::LayoutError;
    ///
    /// let err = LayoutError::borrowed_mutation("normalize_in_place", "audio data is borrowed");
    /// assert!(matches!(err, LayoutError::BorrowedDataMutation { .. }));
    /// ```
    #[inline]
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

    /// Creates a [`LayoutError::DimensionMismatch`] error.
    ///
    /// Use this when an array or buffer has the wrong number of dimensions or
    /// an unexpected shape for the operation being performed.
    ///
    /// # Arguments
    ///
    /// - `expected` — Description of the expected dimensions (e.g. `"(1, 1024)"`).
    /// - `actual` — Description of the actual dimensions received (e.g. `"(2, 1024)"`).
    /// - `operation` — Name of the operation that has the dimension requirement.
    ///
    /// # Returns
    ///
    /// A `LayoutError::DimensionMismatch` variant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::LayoutError;
    ///
    /// let err = LayoutError::dimension_mismatch("(1, 1024)", "(2, 1024)", "mono_fft");
    /// assert!(matches!(err, LayoutError::DimensionMismatch { .. }));
    /// ```
    #[inline]
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
    /// Creates a [`ProcessingError::AlgorithmFailure`] error.
    ///
    /// Use this when a DSP or signal-processing algorithm cannot complete
    /// successfully for reasons intrinsic to the algorithm itself, such as
    /// filter instability or a failed FFT plan.
    ///
    /// # Arguments
    ///
    /// - `algorithm` — Name of the processing algorithm that failed.
    /// - `reason` — Human-readable explanation of what went wrong.
    ///
    /// # Returns
    ///
    /// A `ProcessingError::AlgorithmFailure` variant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::ProcessingError;
    ///
    /// let err = ProcessingError::algorithm_failure("butterworth", "filter became unstable");
    /// assert!(matches!(err, ProcessingError::AlgorithmFailure { .. }));
    /// ```
    #[inline]
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
    /// Creates a [`FeatureError::NotEnabled`] error.
    ///
    /// Use this to report that an optional cargo feature required by the
    /// requested operation is not enabled in the current build.
    ///
    /// # Arguments
    ///
    /// - `feature` — The cargo feature name that must be enabled (e.g. `"transforms"`).
    /// - `operation` — Name of the operation that requires this feature.
    ///
    /// # Returns
    ///
    /// A `FeatureError::NotEnabled` variant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::FeatureError;
    ///
    /// let err = FeatureError::not_enabled("transforms", "stft");
    /// assert!(matches!(err, FeatureError::NotEnabled { .. }));
    /// ```
    #[inline]
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
    #[inline]
    fn from(msg: &str) -> Self {
        Self::InvalidValue {
            parameter: "unknown".to_string(),
            reason: msg.to_string(),
        }
    }
}

impl From<String> for ParameterError {
    #[inline]
    fn from(msg: String) -> Self {
        Self::InvalidValue {
            parameter: "unknown".to_string(),
            reason: msg,
        }
    }
}

impl From<&str> for ProcessingError {
    #[inline]
    fn from(msg: &str) -> Self {
        Self::AlgorithmFailure {
            algorithm: "unknown".to_string(),
            reason: msg.to_string(),
        }
    }
}

impl From<String> for ProcessingError {
    #[inline]
    fn from(msg: String) -> Self {
        Self::AlgorithmFailure {
            algorithm: "unknown".to_string(),
            reason: msg,
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
        let proc_err =
            ProcessingError::algorithm_failure("transforms", "Input size must be power of 2");
        assert!(format!("{}", proc_err).contains("transforms"));
        assert!(format!("{}", proc_err).contains("power of 2"));
    }

    #[test]
    fn test_feature_error() {
        let feat_err = FeatureError::not_enabled("transforms", "spectral analysis");
        assert!(format!("{}", feat_err).contains("transforms"));
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
}
