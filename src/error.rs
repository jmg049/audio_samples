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

//! # Diagnostics
//!
//! Every error type implements [`miette::Diagnostic`] in addition to
//! [`std::error::Error`]. Each variant carries a stable, namespaced **code**
//! (`audio_samples::<domain>::<variant>`) for programmatic matching and an
//! actionable **help** hint telling the caller how to recover. Text-parsing
//! failures additionally carry a [`SourceSpan`](miette::SourceSpan) pointing a
//! caret at the offending character (see [`NoteParseError`] and
//! [`EnumParseError`]).
//!
//! The rich caret-underline rendering is provided by miette's `fancy` renderer,
//! gated behind this crate's `fancy` feature so library consumers are not forced
//! to pull graphical dependencies. Library users get plain [`Display`] output
//! unless they opt in; the `examples/` and the `educational` module enable
//! `fancy` so the pretty output is visible out of the box.

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
//!         from,
//!         to,
//!         reason,
//!         ..
//!     })) => {
//!         eprintln!("Failed to convert {from} → {to}: {reason}");
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

use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

use crate::repr::SampleType;

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
#[derive(Error, Debug, Clone, Diagnostic)]
#[non_exhaustive]
pub enum AudioSampleError {
    /// Errors related to type conversions and casting operations.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Conversion(#[from] ConversionError),

    /// Errors related to invalid parameters or configuration.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Parameter(#[from] ParameterError),

    /// Errors related to memory layout, array structure, or data organization.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Layout(#[from] LayoutError),

    /// Errors that occur during audio processing operations.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Processing(#[from] ProcessingError),

    /// Errors related to missing or disabled features.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Feature(#[from] FeatureError),

    /// Failed to parse a value from user-supplied text (generic fallback).
    ///
    /// Prefer the span-carrying [`NoteParse`](Self::NoteParse) /
    /// [`EnumParse`](Self::EnumParse) variants where the offending character can
    /// be located; this variant is retained for parses that cannot produce a
    /// span.
    #[error("Failed to parse {type_name}. Context: {context}")]
    #[diagnostic(
        code(audio_samples::parse::failed),
        help("check the input format described in the context message")
    )]
    Parse {
        /// Type name of the type we failed to parse
        type_name: String,
        /// User provided context
        context: String,
    },

    /// A note name could not be parsed (carries a caret pointing at the fault).
    #[error(transparent)]
    #[diagnostic(transparent)]
    NoteParse(#[from] NoteParseError),

    /// An enum value could not be parsed from a string (carries a caret).
    #[error(transparent)]
    #[diagnostic(transparent)]
    EnumParse(#[from] EnumParseError),

    #[cfg(feature = "transforms")]
    /// Errors originating from the `spectrograms` crate (requires `feature = "transforms"`).
    #[error(transparent)]
    #[diagnostic(code(audio_samples::transforms::spectrogram))]
    Spectrogram(#[from] spectrograms::SpectrogramError),

    /// Audio data was empty where non-empty data is required.
    ///
    /// Names the operation that produced or received no samples so the caller
    /// knows which step to inspect.
    #[error("Empty audio data in {operation}, where non-empty data is required")]
    #[diagnostic(
        code(audio_samples::empty_data),
        help("ensure `{operation}` is given at least one sample")
    )]
    EmptyData {
        /// The operation that produced or received no samples.
        operation: String,
    },

    /// Error indicating mismatch between total samples and channel count.
    ///
    /// Occurs when the total number of samples is not evenly divisible by the number of
    /// channels, indicating malformed or corrupted audio data.
    #[error(
        "Invalid number of samples ({total_samples}) for {channels} channels: not evenly divisible"
    )]
    #[diagnostic(
        code(audio_samples::invalid_number_of_samples),
        help("provide a sample count that is an exact multiple of the channel count")
    )]
    InvalidNumberOfSamples {
        /// Total number of samples
        total_samples: usize,
        /// Number of channels
        channels: u32,
    },

    /// Error indicating a formatting failure during display or debug output.
    #[error("Fmt error occurred: {0}")]
    #[diagnostic(code(audio_samples::fmt))]
    Fmt(#[from] std::fmt::Error),

    /// An I/O operation failed.
    ///
    /// Captures the [`std::io::ErrorKind`] and message of the originating
    /// `io::Error` (which is not `Clone`) so the error remains cloneable while
    /// preserving the salient detail.
    #[error("I/O error during {operation}: {message}")]
    #[diagnostic(
        code(audio_samples::io),
        help("check file paths, permissions, and that any external viewer is available")
    )]
    Io {
        /// The operation that performed I/O.
        operation: String,
        /// The kind of I/O failure.
        kind: std::io::ErrorKind,
        /// The original error message.
        message: String,
    },

    /// Error indicating an unsupported operation or configuration.
    ///
    /// Used when a requested operation is valid in principle but not implemented for the
    /// given combination of inputs (e.g., certain multi-channel operations, unsupported
    /// sample types, or platform-specific limitations).
    #[error("Unsupported operation '{operation}': {reason}")]
    #[diagnostic(
        code(audio_samples::unsupported),
        help(
            "this configuration is not implemented; see the operation's docs for supported inputs"
        )
    )]
    Unsupported {
        /// The operation that was requested.
        operation: String,
        /// Why the requested configuration is unsupported.
        reason: String,
    },
}

impl AudioSampleError {
    /// Creates an [`AudioSampleError::Unsupported`] error.
    ///
    /// Use this when a caller requests an operation that is valid in principle
    /// but not implemented for the given combination of inputs (e.g. multi-channel
    /// STFT).
    ///
    /// # Arguments
    ///
    /// - `operation` — Name of the operation that was requested.
    /// - `reason` — Why this particular configuration is unsupported.
    ///
    /// # Returns
    ///
    /// `AudioSampleError::Unsupported { operation, reason }`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::AudioSampleError;
    ///
    /// let err = AudioSampleError::unsupported("mfcc", "multi-channel input is not supported");
    /// assert!(matches!(err, AudioSampleError::Unsupported { .. }));
    /// ```
    #[inline]
    pub fn unsupported<O, R>(operation: O, reason: R) -> Self
    where
        O: ToString,
        R: ToString,
    {
        Self::Unsupported {
            operation: operation.to_string(),
            reason: reason.to_string(),
        }
    }

    /// Creates an [`AudioSampleError::EmptyData`] error.
    ///
    /// Use this when an operation requires at least one sample but received an
    /// empty buffer. Naming the operation lets the caller pinpoint which step
    /// produced or consumed no data.
    ///
    /// # Arguments
    ///
    /// - `operation` — Name of the operation that produced or received no samples.
    ///
    /// # Returns
    ///
    /// `AudioSampleError::EmptyData { operation }`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::AudioSampleError;
    ///
    /// let err = AudioSampleError::empty_data("resample");
    /// assert!(matches!(err, AudioSampleError::EmptyData { .. }));
    /// ```
    #[inline]
    pub fn empty_data<O>(operation: O) -> Self
    where
        O: ToString,
    {
        Self::EmptyData {
            operation: operation.to_string(),
        }
    }

    /// Creates an [`AudioSampleError::Io`] error from a [`std::io::Error`].
    ///
    /// Captures the error's [`kind`](std::io::ErrorKind) and message so the
    /// resulting `AudioSampleError` remains `Clone` (an `io::Error` is not).
    ///
    /// # Arguments
    ///
    /// - `operation` — Name of the operation that performed I/O.
    /// - `err` — The originating I/O error.
    ///
    /// # Returns
    ///
    /// `AudioSampleError::Io { operation, kind, message }`.
    #[inline]
    pub fn io<O>(operation: O, err: &std::io::Error) -> Self
    where
        O: ToString,
    {
        Self::Io {
            operation: operation.to_string(),
            kind: err.kind(),
            message: err.to_string(),
        }
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
    /// For parses where the offending character can be located, prefer
    /// constructing a [`NoteParseError`] or [`EnumParseError`] (which carry a
    /// [`SourceSpan`]) instead.
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
}

/// A note name (scientific pitch notation) could not be parsed.
///
/// Carries the offending input and a [`SourceSpan`] pointing a caret at the
/// character that could not be interpreted, so the `fancy` renderer can underline
/// exactly what went wrong.
#[derive(Error, Debug, Clone, Diagnostic)]
#[error("invalid note name: {kind}")]
#[diagnostic(
    code(audio_samples::parse::note_name),
    help("expected scientific pitch notation like `A4`, `C#3`, `Bb2`")
)]
pub struct NoteParseError {
    /// The original text that failed to parse.
    #[source_code]
    pub input: String,
    /// The span within `input` that could not be interpreted.
    #[label("{kind}")]
    pub span: SourceSpan,
    /// A short description of what was wrong (e.g. `"unrecognised note"`).
    pub kind: String,
}

impl NoteParseError {
    /// Creates a [`NoteParseError`] pointing at `span` within `input`.
    ///
    /// # Arguments
    ///
    /// - `input` — The full text that was being parsed.
    /// - `span` — Byte offset and length of the offending region.
    /// - `kind` — Short description of the fault.
    #[inline]
    pub fn new<I, K>(input: I, span: impl Into<SourceSpan>, kind: K) -> Self
    where
        I: ToString,
        K: ToString,
    {
        Self {
            input: input.to_string(),
            span: span.into(),
            kind: kind.to_string(),
        }
    }
}

/// A string could not be parsed into one of a closed set of enum variants.
///
/// Carries the offending input, a [`SourceSpan`] underlining the unrecognised
/// token, and a help line listing the valid alternatives.
#[derive(Error, Debug, Clone, Diagnostic)]
#[error("unrecognised value for {type_name}")]
#[diagnostic(code(audio_samples::parse::enum_value))]
pub struct EnumParseError {
    /// The original text that failed to parse.
    #[source_code]
    pub input: String,
    /// The span within `input` that was not recognised.
    #[label("not one of the accepted values")]
    pub span: SourceSpan,
    /// The name of the enum that failed to parse.
    pub type_name: String,
    /// Help text listing the valid alternatives.
    #[help]
    pub help: String,
}

impl EnumParseError {
    /// Creates an [`EnumParseError`] for `input`, listing the `expected` values.
    ///
    /// The span underlines the whole input (the unrecognised token), and the
    /// help line is generated from `expected`.
    ///
    /// # Arguments
    ///
    /// - `type_name` — Human-readable name of the enum being parsed.
    /// - `input` — The text that did not match any variant.
    /// - `expected` — The accepted string values.
    #[inline]
    pub fn new<N, I>(type_name: N, input: I, expected: &[&str]) -> Self
    where
        N: ToString,
        I: AsRef<str>,
    {
        let input = input.as_ref().to_string();
        let len = input.len();
        Self {
            span: (0, len).into(),
            help: format!("expected one of: {}", expected.join(", ")),
            type_name: type_name.to_string(),
            input,
        }
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
/// The source and target are carried as [`SampleType`] values rather than free
/// text, so callers can match on the exact pair and a type name can never be
/// mistyped.
///
/// # Intended Usage
///
/// Returned by `to_format`, `to_type`, `cast_as`, and `cast_to`. Prefer the
/// audio-aware variants (`to_format`, `to_type`) over raw casts unless you
/// deliberately need non-normalised bit patterns.
#[derive(Error, Debug, Clone, Diagnostic)]
#[non_exhaustive]
pub enum ConversionError {
    /// Failed to convert between audio sample types with audio-aware scaling.
    #[error("Failed to convert sample value {value} from {from} to {to}: {reason}")]
    #[diagnostic(
        code(audio_samples::conversion::audio_conversion),
        url(docsrs),
        help(
            "use `to_format`/`to_type` for audio-aware conversion; the value may be out of the target range"
        )
    )]
    AudioConversion {
        /// The sample value that failed to convert (as string for display).
        value: String,
        /// Source sample type.
        from: SampleType,
        /// Target sample type.
        to: SampleType,
        /// Detailed reason for the conversion failure.
        reason: String,
    },

    /// Failed to perform raw numeric casting between types.
    #[error("Failed to cast value {value} from {from} to {to}: {reason}")]
    #[diagnostic(
        code(audio_samples::conversion::numeric_cast),
        url(docsrs),
        help("raw casts do not normalise; use `to_format`/`to_type` for audio-aware conversion")
    )]
    NumericCast {
        /// The numeric value that failed to cast (as string for display).
        value: String,
        /// Source sample type.
        from: SampleType,
        /// Target sample type.
        to: SampleType,
        /// Detailed reason for the casting failure.
        reason: String,
    },

    /// The conversion operation is not supported between the specified types.
    #[error("Conversion from {from} to {to} is not supported")]
    #[diagnostic(
        code(audio_samples::conversion::unsupported),
        url(docsrs),
        help(
            "use `to_format`/`to_type` for audio-aware conversion, or `cast_as`/`cast_to` for raw bit-casts"
        )
    )]
    UnsupportedConversion {
        /// Source sample type.
        from: SampleType,
        /// Target sample type.
        to: SampleType,
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
#[derive(Error, Debug, Clone, Diagnostic)]
#[non_exhaustive]
pub enum ParameterError {
    /// A parameter value is outside the valid range for the operation.
    #[error(
        "Parameter '{parameter}' value {value} is outside valid range [{min}, {max}]: {reason}"
    )]
    #[diagnostic(
        code(audio_samples::parameter::out_of_range),
        url(docsrs),
        help("pass a value for `{parameter}` in [{min}, {max}]; got {value}")
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
    #[diagnostic(
        code(audio_samples::parameter::invalid_value),
        url(docsrs),
        help("check the documented constraints for `{parameter}`")
    )]
    InvalidValue {
        /// Name of the parameter with invalid value.
        parameter: String,
        /// Detailed explanation of why the value is invalid.
        reason: String,
    },

    /// Required parameters are missing or empty.
    #[error("Required parameter '{parameter}' is missing or empty")]
    #[diagnostic(
        code(audio_samples::parameter::missing),
        url(docsrs),
        help("supply a value for `{parameter}`")
    )]
    Missing {
        /// Name of the missing parameter.
        parameter: String,
    },

    /// A configuration object contains conflicting or invalid settings.
    #[error("Invalid configuration for {operation}: {reason}")]
    #[diagnostic(
        code(audio_samples::parameter::invalid_configuration),
        url(docsrs),
        help("review the configuration fields for `{operation}`; some settings conflict")
    )]
    InvalidConfiguration {
        /// The operation or component being configured.
        operation: String,
        /// Detailed explanation of the configuration problem.
        reason: String,
    },
}

/// Describes how many channels an operation requires.
///
/// Used by [`LayoutError::ChannelCountUnsupported`] to express a channel-count
/// precondition in a structured, matchable way rather than as prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChannelRequirement {
    /// Exactly one channel (mono).
    Mono,
    /// Exactly two channels (stereo).
    Stereo,
    /// Exactly `n` channels.
    Exactly(u32),
    /// At least `n` channels.
    AtLeast(u32),
}

impl std::fmt::Display for ChannelRequirement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mono => write!(f, "mono (1 channel)"),
            Self::Stereo => write!(f, "stereo (2 channels)"),
            Self::Exactly(n) => write!(f, "exactly {n} channels"),
            Self::AtLeast(n) => write!(f, "at least {n} channels"),
        }
    }
}

/// Errors related to memory layout, array structure, and data organization.
///
/// # Purpose
///
/// Covers failures related to array contiguity, dimension mismatches,
/// borrowed data mutation attempts, channel-count preconditions, and other
/// structural issues that arise from the ndarray-backed storage of audio
/// samples.
///
/// # Intended Usage
///
/// Return these errors when an operation's structural preconditions are not
/// met — for example, when a mono-only function receives multi-channel input
/// (use [`ChannelCountUnsupported`][LayoutError::ChannelCountUnsupported]), or
/// when an in-place operation is attempted on borrowed (non-owned) data.
#[derive(Error, Debug, Clone, Diagnostic)]
#[non_exhaustive]
pub enum LayoutError {
    /// Array data is not contiguous in memory when contiguous layout is required.
    #[error(
        "Array layout error: {operation} requires contiguous memory layout, but array is {layout_type}"
    )]
    #[diagnostic(
        code(audio_samples::layout::non_contiguous),
        url(docsrs),
        help("clone or copy the data into a contiguous buffer before `{operation}`")
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
    #[diagnostic(
        code(audio_samples::layout::dimension_mismatch),
        url(docsrs),
        help("reshape the input to {expected_dims} before `{operation}`")
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
    #[diagnostic(
        code(audio_samples::layout::borrowed_data_mutation),
        url(docsrs),
        help("clone the audio into an owned value before `{operation}`")
    )]
    BorrowedDataMutation {
        /// The operation that attempted to modify borrowed data.
        operation: String,
        /// Explanation of why modification is not allowed.
        reason: String,
    },

    /// Data size or format is incompatible with the operation requirements.
    #[error("Incompatible data format for {operation}: {reason}")]
    #[diagnostic(
        code(audio_samples::layout::incompatible_format),
        url(docsrs),
        help("convert the data to the format `{operation}` expects")
    )]
    IncompatibleFormat {
        /// The operation with format requirements.
        operation: String,
        /// Detailed explanation of the format incompatibility.
        reason: String,
    },

    /// An operation requires a specific channel count that the input does not have.
    ///
    /// This is the structured home for the recurring "X is only supported for
    /// mono audio" precondition. Callers can match on `required`/`actual` rather
    /// than parsing prose.
    #[error("{operation} requires {required}, but the input has {actual} channel(s)")]
    #[diagnostic(code(audio_samples::layout::channel_count_unsupported), url(docsrs))]
    ChannelCountUnsupported {
        /// The operation with the channel-count precondition.
        operation: String,
        /// The channel layout the operation requires.
        required: ChannelRequirement,
        /// The number of channels actually supplied.
        actual: u32,
        /// Actionable recovery hint (depends on `required`).
        #[help]
        help: String,
    },

    /// A shape-related error occurred during array operations.
    ///
    /// Carries the originating [`ndarray::ShapeError`] as a source so the full
    /// cause chain is preserved rather than flattened to a string.
    #[error("Shape error in {operation}")]
    #[diagnostic(
        code(audio_samples::layout::shape_error),
        url(docsrs),
        help("check that the array dimensions are compatible with `{operation}`")
    )]
    ShapeError {
        /// The operation that encountered the shape error.
        operation: String,
        /// The underlying ndarray shape error.
        #[source]
        source: ndarray::ShapeError,
    },

    /// Error indicating an invalid operation on audio data.
    ///
    /// Occurs when attempting an operation that violates preconditions that are
    /// not better expressed by the more specific variants above.
    #[error("Invalid operation on {0}:\nReason: {1}")]
    #[diagnostic(
        code(audio_samples::layout::invalid_operation),
        url(docsrs),
        help("see the operation's documentation for its preconditions")
    )]
    InvalidOperation(String, String),
}

impl From<ndarray::ShapeError> for AudioSampleError {
    #[inline]
    fn from(err: ndarray::ShapeError) -> Self {
        Self::Layout(LayoutError::ShapeError {
            operation: "ndarray operation".to_string(),
            source: err,
        })
    }
}

impl LayoutError {
    /// Creates a [`LayoutError::ShapeError`] preserving the originating error.
    ///
    /// # Arguments
    ///
    /// - `operation` — Name of the operation that encountered the shape error.
    /// - `source` — The underlying [`ndarray::ShapeError`].
    ///
    /// # Returns
    ///
    /// A `LayoutError::ShapeError` carrying `source` as its cause.
    #[inline]
    pub fn shape_error<O>(operation: O, source: ndarray::ShapeError) -> Self
    where
        O: ToString,
    {
        Self::ShapeError {
            operation: operation.to_string(),
            source,
        }
    }

    /// Creates a [`LayoutError::ChannelCountUnsupported`] error.
    ///
    /// This is the canonical home for channel-count preconditions such as the
    /// recurring "only supported for mono audio" failure.
    ///
    /// # Arguments
    ///
    /// - `operation` — Name of the operation with the precondition.
    /// - `required` — The channel layout the operation requires.
    /// - `actual` — The number of channels actually supplied.
    ///
    /// # Returns
    ///
    /// A `LayoutError::ChannelCountUnsupported` variant with a help hint derived
    /// from `required`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::{ChannelRequirement, LayoutError};
    ///
    /// let err = LayoutError::channel_count_unsupported("stft", ChannelRequirement::Mono, 2);
    /// assert!(matches!(err, LayoutError::ChannelCountUnsupported { .. }));
    /// ```
    #[inline]
    pub fn channel_count_unsupported<O>(
        operation: O,
        required: ChannelRequirement,
        actual: u32,
    ) -> Self
    where
        O: ToString,
    {
        let help = match required {
            ChannelRequirement::Mono => "convert to mono with `.to_mono()` first".to_string(),
            ChannelRequirement::Stereo => "convert to stereo with `.to_stereo()` first".to_string(),
            other => format!("supply audio with {other}"),
        };
        Self::ChannelCountUnsupported {
            operation: operation.to_string(),
            required,
            actual,
            help,
        }
    }

    /// Creates a [`LayoutError::InvalidOperation`] error.
    ///
    /// Use this when a valid operation is called in a context where it cannot
    /// proceed and no more specific variant applies.
    ///
    /// # Arguments
    ///
    /// - `operation` — Name of the operation that was called in an invalid context.
    /// - `reason` — Human-readable explanation of why the operation cannot proceed.
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
#[derive(Error, Debug, Clone, Diagnostic)]
#[non_exhaustive]
pub enum ProcessingError {
    /// A mathematical operation failed due to invalid input or numerical issues.
    #[error("Mathematical operation '{operation}' failed: {reason}")]
    #[diagnostic(
        code(audio_samples::processing::mathematical_failure),
        url(docsrs),
        help("check the input values for `{operation}` (NaN, infinity, or division by zero)")
    )]
    MathematicalFailure {
        /// The mathematical operation that failed.
        operation: String,
        /// Detailed explanation of the mathematical failure.
        reason: String,
    },

    /// An audio processing algorithm encountered an error during execution.
    #[error("Audio processing algorithm '{algorithm}' failed: {reason}")]
    #[diagnostic(
        code(audio_samples::processing::algorithm_failure),
        url(docsrs),
        help("see the documentation for `{algorithm}` for its preconditions")
    )]
    AlgorithmFailure {
        /// The processing algorithm that failed.
        algorithm: String,
        /// Detailed explanation of the algorithm failure.
        reason: String,
    },

    /// The operation failed due to insufficient data or resources.
    #[error("Insufficient data for {operation}: {reason}")]
    #[diagnostic(
        code(audio_samples::processing::insufficient_data),
        url(docsrs),
        help("provide more samples to `{operation}`")
    )]
    InsufficientData {
        /// The operation that requires more data.
        operation: String,
        /// Explanation of the data requirements.
        reason: String,
    },

    /// A convergence-based algorithm failed to converge within limits.
    #[error("Algorithm '{algorithm}' failed to converge after {iterations} iterations")]
    #[diagnostic(
        code(audio_samples::processing::convergence_failure),
        url(docsrs),
        help("raise the iteration limit or relax the convergence tolerance for `{algorithm}`")
    )]
    ConvergenceFailure {
        /// The algorithm that failed to converge.
        algorithm: String,
        /// Number of iterations attempted.
        iterations: u32,
    },

    /// An external dependency or resource required for processing is unavailable.
    #[error("External dependency '{dependency}' required for {operation} is unavailable: {reason}")]
    #[diagnostic(
        code(audio_samples::processing::external_dependency),
        url(docsrs),
        help("ensure the `{dependency}` backend is configured correctly for `{operation}`")
    )]
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
#[derive(Error, Debug, Clone, Diagnostic)]
#[non_exhaustive]
pub enum FeatureError {
    /// A cargo feature is required but not enabled.
    #[error("Feature '{feature}' is required for {operation} but not enabled")]
    #[diagnostic(
        code(audio_samples::feature::not_enabled),
        url(docsrs),
        help("rebuild with `--features {feature}`")
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
    #[diagnostic(
        code(audio_samples::feature::multiple_required),
        url(docsrs),
        help("rebuild with `--features {required_features}`")
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
    #[diagnostic(
        code(audio_samples::feature::misconfigured),
        url(docsrs),
        help("review the build configuration for the `{feature}` feature")
    )]
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
    /// - `from` — The source [`SampleType`].
    /// - `to` — The target [`SampleType`].
    /// - `reason` — Human-readable explanation of why the conversion failed.
    ///
    /// # Returns
    ///
    /// A `ConversionError::AudioConversion` variant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::{AudioSampleError, ConversionError, SampleType};
    ///
    /// let err = ConversionError::audio_conversion("32768", SampleType::I16, SampleType::U8, "value out of range");
    /// assert!(matches!(err, ConversionError::AudioConversion { .. }));
    ///
    /// // Wraps into the root error type via the From impl.
    /// let audio_err: AudioSampleError = err.into();
    /// assert!(matches!(audio_err, AudioSampleError::Conversion(_)));
    /// ```
    #[inline]
    pub fn audio_conversion<V, R>(value: V, from: SampleType, to: SampleType, reason: R) -> Self
    where
        V: ToString,
        R: ToString,
    {
        Self::AudioConversion {
            value: value.to_string(),
            from,
            to,
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
    /// - `from` — The source [`SampleType`].
    /// - `to` — The target [`SampleType`].
    /// - `reason` — Human-readable explanation of why the cast failed.
    ///
    /// # Returns
    ///
    /// A `ConversionError::NumericCast` variant.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::{ConversionError, SampleType};
    ///
    /// let err = ConversionError::numeric_cast("300.0", SampleType::F64, SampleType::U8, "value exceeds 255");
    /// assert!(matches!(err, ConversionError::NumericCast { .. }));
    /// ```
    #[inline]
    pub fn numeric_cast<V, R>(value: V, from: SampleType, to: SampleType, reason: R) -> Self
    where
        V: ToString,
        R: ToString,
    {
        Self::NumericCast {
            value: value.to_string(),
            from,
            to,
            reason: reason.to_string(),
        }
    }

    /// Creates a [`ConversionError::UnsupportedConversion`] error.
    ///
    /// # Arguments
    ///
    /// - `from` — The source [`SampleType`].
    /// - `to` — The target [`SampleType`].
    ///
    /// # Returns
    ///
    /// A `ConversionError::UnsupportedConversion` variant.
    #[inline]
    #[must_use]
    pub const fn unsupported_conversion(from: SampleType, to: SampleType) -> Self {
        Self::UnsupportedConversion { from, to }
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

    /// Creates a [`ParameterError::Missing`] error.
    ///
    /// # Arguments
    ///
    /// - `parameter` — Name of the missing or empty parameter.
    ///
    /// # Returns
    ///
    /// A `ParameterError::Missing` variant.
    #[inline]
    pub fn missing<P>(parameter: P) -> Self
    where
        P: ToString,
    {
        Self::Missing {
            parameter: parameter.to_string(),
        }
    }

    /// Creates a [`ParameterError::InvalidConfiguration`] error.
    ///
    /// # Arguments
    ///
    /// - `operation` — The operation or component being configured.
    /// - `reason` — Why the configuration is invalid.
    ///
    /// # Returns
    ///
    /// A `ParameterError::InvalidConfiguration` variant.
    #[inline]
    pub fn invalid_configuration<O, R>(operation: O, reason: R) -> Self
    where
        O: ToString,
        R: ToString,
    {
        Self::InvalidConfiguration {
            operation: operation.to_string(),
            reason: reason.to_string(),
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

    /// Creates a [`ProcessingError::MathematicalFailure`] error.
    ///
    /// # Arguments
    ///
    /// - `operation` — The mathematical operation that failed.
    /// - `reason` — Why it failed (e.g. NaN, division by zero).
    ///
    /// # Returns
    ///
    /// A `ProcessingError::MathematicalFailure` variant.
    #[inline]
    pub fn mathematical_failure<O, R>(operation: O, reason: R) -> Self
    where
        O: ToString,
        R: ToString,
    {
        Self::MathematicalFailure {
            operation: operation.to_string(),
            reason: reason.to_string(),
        }
    }

    /// Creates a [`ProcessingError::ExternalDependency`] error.
    ///
    /// Use this when an external backend (e.g. the `rubato` resampler) reports a
    /// failure. The originating error's message is captured in `reason`.
    ///
    /// # Arguments
    ///
    /// - `dependency` — The external dependency that failed (e.g. `"rubato"`).
    /// - `operation` — The operation that required the dependency.
    /// - `reason` — The failure detail (often the foreign error's message).
    ///
    /// # Returns
    ///
    /// A `ProcessingError::ExternalDependency` variant.
    #[inline]
    pub fn external_dependency<D, O, R>(dependency: D, operation: O, reason: R) -> Self
    where
        D: ToString,
        O: ToString,
        R: ToString,
    {
        Self::ExternalDependency {
            dependency: dependency.to_string(),
            operation: operation.to_string(),
            reason: reason.to_string(),
        }
    }

    /// Creates a [`ProcessingError::InsufficientData`] error.
    ///
    /// # Arguments
    ///
    /// - `operation` — The operation that needs more data.
    /// - `reason` — Explanation of the data requirement.
    ///
    /// # Returns
    ///
    /// A `ProcessingError::InsufficientData` variant.
    #[inline]
    pub fn insufficient_data<O, R>(operation: O, reason: R) -> Self
    where
        O: ToString,
        R: ToString,
    {
        Self::InsufficientData {
            operation: operation.to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use miette::Diagnostic;

    #[test]
    fn test_error_hierarchy() {
        let conversion_err = ConversionError::audio_conversion(
            "32768",
            SampleType::I16,
            SampleType::U8,
            "Out of range",
        );
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
            SampleType::F32,
            SampleType::I16,
            "Value out of signed 16-bit range",
        );
        let audio_err = AudioSampleError::Conversion(conversion_err);

        let error_string = format!("{}", audio_err);
        assert!(error_string.contains("Failed to convert sample value 1.5"));
        assert!(error_string.contains("f32"));
        assert!(error_string.contains("i16"));
    }

    #[test]
    fn test_diagnostic_codes() {
        // Codes propagate transparently through the root wrapper.
        let err = AudioSampleError::Parameter(ParameterError::out_of_range(
            "cutoff_hz",
            "25000",
            "20",
            "22050",
            "exceeds Nyquist",
        ));
        assert_eq!(
            err.code().unwrap().to_string(),
            "audio_samples::parameter::out_of_range"
        );

        let layout = AudioSampleError::Layout(LayoutError::channel_count_unsupported(
            "stft",
            ChannelRequirement::Mono,
            2,
        ));
        assert_eq!(
            layout.code().unwrap().to_string(),
            "audio_samples::layout::channel_count_unsupported"
        );
    }

    #[test]
    fn test_diagnostic_help() {
        let err =
            ParameterError::out_of_range("cutoff_hz", "25000", "20", "22050", "exceeds Nyquist");
        let help = err.help().unwrap().to_string();
        assert!(help.contains("cutoff_hz"));
        assert!(help.contains("22050"));

        let layout = LayoutError::channel_count_unsupported("stft", ChannelRequirement::Mono, 2);
        let help = layout.help().unwrap().to_string();
        assert!(help.contains("to_mono"));
    }

    #[test]
    fn test_enum_parse_error_span() {
        let err = EnumParseError::new("PadSide", "lefty", &["left", "right"]);
        // span covers the whole unrecognised token
        assert_eq!(err.span.offset(), 0);
        assert_eq!(err.span.len(), "lefty".len());
        let help = err.help().unwrap().to_string();
        assert!(help.contains("left"));
        assert!(help.contains("right"));
    }
}
