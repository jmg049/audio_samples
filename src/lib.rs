#![cfg_attr(feature = "simd", feature(portable_simd))]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![deny(missing_docs)]
// Additional strictness beyond default groups
#![warn(clippy::unwrap_used)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![warn(clippy::exhaustive_enums)]
#![warn(clippy::exhaustive_structs)]
#![warn(clippy::panic_in_result_fn)]
#![warn(clippy::unnecessary_wraps)]
// Intentional allowances
#![allow(clippy::too_many_arguments)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::tuple_array_conversions)]
#![allow(clippy::unsafe_derive_deserialize)]
#![allow(clippy::multiple_unsafe_ops_per_block)]
#![allow(clippy::doc_markdown)]
#![allow(unused_unsafe)]

//! # AudioSamples
//!
//! A typed audio processing library for Rust that treats audio as a first-class,
//! invariant-preserving object rather than an unstructured numeric buffer. is this library?
//!
//! `audio_samples` provides a single central type, [`AudioSamples<T>`], that pairs raw
//! PCM data (backed by [`ndarray`](https://docs.rs/ndarray)) with essential metadata:
//! sample rate, channel count, and memory layout. Every audio processing operation in the
//! library is defined as a trait method on this type, ensuring that metadata travels with
//! the data throughout a processing pipeline. does this library exist?
//!
//! Low-level audio APIs in Rust typically expose bare slices or `Vec<f32>` buffers,
//! leaving metadata management to the caller. This encourages subtle bugs such as
//! mismatched sample rates after resampling, or interleaved/non-interleaved confusion
//! when passing buffers between components. `audio_samples` eliminates these error
//! classes by encoding invariants directly into the type. should it be used?
//!
//! Start by creating an [`AudioSamples<T>`] from an ndarray or from one of the
//! built-in signal generators (see [`utils::generation`]), then chain trait methods
//! for any further processing. Feature flags keep the dependency footprint small –
//! only enable what your application needs.
//!
//! ## Installation
//!
//! ```bash
//! cargo add audio_samples
//! ```
//!
//! ## Features
//!
//! The library uses a modular feature flag system. Only enable what you need.
//!
//! | Feature | Description |
//! |---|---|
//! | `statistics` | Peak, RMS, variance, zero-crossings, and other statistical measures |
//! | `processing` | Normalise, scale, clip, DC offset removal, and other sample-level operations (implies `statistics`) |
//! | `editing` | Time-domain editing: trim, pad, reverse, concatenate, fade (implies `statistics`, `random-generation`) |
//! | `channels` | Channel operations: mono↔stereo conversion, channel extraction, interleave/deinterleave |
//! | `transforms` | Spectrogram and frequency-domain transforms via the `spectrograms` crate |
//! | `iir-filtering` | IIR filter design and application (Butterworth, Chebyshev I) |
//! | `parametric-eq` | Multi-band parametric equaliser (implies `iir-filtering`) |
//! | `dynamic-range` | Compression, limiting, and expansion with side-chain support |
//! | `envelopes` | Amplitude, RMS, and analytical envelope followers (implies `dynamic-range`, `editing`, `random-generation`) |
//! | `peak-picking` | Onset strength curve peak picking |
//! | `onset-detection` | Onset detection (implies `transforms`, `peak-picking`, `processing`) |
//! | `beat-tracking` | Beat tracking and tempo estimation |
//! | `decomposition` | Audio source decomposition (implies `onset-detection`) |
//! | `pitch-analysis` | YIN and autocorrelation pitch detection (implies `transforms`) |
//! | `vad` | Voice activity detection |
//! | `psychoacoustic` | Psychoacoustic analysis: Bark/Mel band layouts, ATH, masking thresholds, SMR (implies `transforms`) |
//! | `resampling` | High-quality resampling via the `rubato` crate |
//! | `plotting` | Signal plotting via `plotly` |
//! | `fixed-size-audio` | Stack-allocated fixed-size audio buffers |
//! | `random-generation` | Noise generators and stochastic signal sources (implies `rand`) |
//! | `full` | Enables all of the above |
//!
//! See `Cargo.toml` for the complete dependency graph.
//!
//! ## Error Handling
//!
//! All fallible operations return [`AudioSampleResult<T>`], which is an alias for
//! `Result<T, AudioSampleError>`. Errors are structured so that the variant indicates
//! the category of failure and the inner type provides detail.
//!
//! ```rust
//! use audio_samples::{AudioSampleError, AudioSampleResult, ParameterError};
//!
//! let audio_result: AudioSampleResult<()> = Err(AudioSampleError::Parameter(
//!     ParameterError::invalid_value("window_size", "must be > 0"),
//! ));
//!
//! match audio_result {
//!     Ok(()) => {}
//!     Err(AudioSampleError::Conversion(err)) => eprintln!("Conversion failed: {err}"),
//!     Err(AudioSampleError::Parameter(err)) => eprintln!("Invalid parameter: {err}"),
//!     Err(other_err) => eprintln!("Other error: {other_err}"),
//! }
//! ```
//!
//! ## Full Examples
//!
//! Examples live in the repository (see `examples/`) and in the crate-level docs on
//! <https://docs.rs/audio_samples>.
//!
//! ## Quick Start
//!
//! ### Creating Audio Data
//!
//! ```rust
//! use audio_samples::{AudioSamples, sample_rate};
//! use ndarray::array;
//!
//! // Create mono audio
//! let data = array![0.1f32, 0.5, -0.3, 0.8, -0.2];
//! let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
//!
//! // Create stereo audio (channels × samples)
//! let stereo_data = array![
//!     [0.1f32, 0.5, -0.3],  // Left channel
//!     [0.8f32, -0.2, 0.4]   // Right channel
//! ];
//! let stereo_audio = AudioSamples::new_multi_channel(stereo_data, sample_rate!(44100)).unwrap();
//! ```
//!
//! ### Basic Statistics
//!
//! Requires the `statistics` feature.
//!
//! ```rust,ignore
//! use audio_samples::{AudioStatistics, sine_wave, sample_rate};
//! use std::time::Duration;
//!
//! // Generate a 440 Hz sine wave at 44.1 kHz sample rate, amplitude 1.0
//! let audio = sine_wave::<f32>(440.0, Duration::from_secs_f32(1.0), sample_rate!(44100), 1.0);
//!
//! let peak           = audio.peak();
//! let min            = audio.min_sample();
//! let max            = audio.max_sample();
//! let mean           = audio.mean();
//! let rms            = audio.rms().unwrap();
//! let variance       = audio.variance().unwrap();
//! let zero_crossings = audio.zero_crossings();
//! ```
//!
//! ### Processing Operations
//!
//! Requires the `processing` feature.
//!
//! ```rust,ignore
//! use audio_samples::{AudioProcessing, NormalizationConfig, AudioSamples, sample_rate};
//! use ndarray::array;
//!
//! let data = array![0.1f32, 0.5, -0.3, 0.8, -0.2];
//! let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
//!
//! // Method chaining: each operation consumes and returns Self
//! let audio = audio
//!     .normalize(NormalizationConfig::peak(1.0))
//!     .unwrap()
//!     .scale(0.5)
//!     .remove_dc_offset()
//!     .unwrap();
//! ```
//!
//! ### Type Conversions
//!
//! ```rust
//! use audio_samples::{AudioSamples, AudioTypeConversion, sample_rate};
//! use ndarray::array;
//!
//! let audio_f32 = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0], sample_rate!(44100)).unwrap();
//! let audio_i16 = audio_f32.clone().to_type::<i16>();
//! let audio_f64 = audio_f32.to_type::<f64>();
//! ```
//!
//! ### Iterating Over Audio Data
//!
//! ```rust
//! use audio_samples::{AudioSampleIterators, AudioSamples, sample_rate};
//! use ndarray::array;
//!
//! let audio = AudioSamples::new_mono(
//!     array![1.0f32, 2.0, 3.0, 4.0],
//!     sample_rate!(44100),
//! ).unwrap();
//!
//! // Iterate by frames (one sample per channel per time step)
//! for frame in audio.frames() {
//!     println!("Frame: {:?}", frame);
//! }
//!
//! // Iterate by channels
//! for channel in audio.channels() {
//!     println!("Channel: {:?}", channel);
//! }
//! ```
//!
//! ### Psychoacoustic Analysis
//!
//! Requires the `psychoacoustic` feature.
//!
//! ```rust,ignore
//! use audio_samples::{
//!     AudioPerceptualAnalysis, BandLayout, PsychoacousticConfig, sine_wave, sample_rate,
//! };
//! use non_empty_slice::NonEmptySlice;
//! use spectrograms::WindowType;
//! use std::num::NonZeroUsize;
//! use std::time::Duration;
//!
//! let signal = sine_wave::<f32>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.8);
//!
//! // 24 Bark critical bands mapped onto 1024 MDCT bins.
//! let layout = BandLayout::bark(
//!     NonZeroUsize::new(24).unwrap(),
//!     44100.0,
//!     NonZeroUsize::new(1024).unwrap(),
//! );
//!
//! let weights = vec![1.0_f32; 24];
//! let config = PsychoacousticConfig::new(
//!     -60.0, 14.5, 5.5, 25.0, 6.0,
//!     NonEmptySlice::from_slice(&weights).unwrap(),
//!     1e-10,
//! );
//!
//! let result = signal
//!     .analyse_psychoacoustic(WindowType::Hanning, &layout, &config)
//!     .unwrap();
//!
//! // Bands with positive SMR are audible above the masking threshold.
//! for metric in result.band_metrics.as_slice().iter() {
//!     if metric.signal_to_mask_ratio > 0.0 {
//!         println!(
//!             "{:.0} Hz — SMR {:.1} dB (importance {:.2})",
//!             metric.band.centre_frequency,
//!             metric.signal_to_mask_ratio,
//!             metric.importance,
//!         );
//!     }
//! }
//! ```
//!
//! ## Core Type System
//!
//! ### Supported Sample Types
//!
//! | Type | Width | Notes |
//! |---|---|---|
//! | `u8`  | 8-bit unsigned  | Mid-scale (128) represents silence |
//! | `i16` | 16-bit signed   | CD-quality audio |
//! | [`I24`](i24::I24) | 24-bit signed | From the `i24` crate |
//! | `i32` | 32-bit signed   | High-dynamic-range integer audio |
//! | `f32` | 32-bit float    | Most DSP operations use this type |
//! | `f64` | 64-bit float    | High-precision processing |
//!
//! ### Type System Traits
//!
//! - **[`AudioSample`]** – Core trait all sample types implement. Provides constants
//!   (`MAX`, `MIN`, `BITS`, `BYTES`, `LABEL`) and low-level byte operations.
//!
//! - **[`ConvertTo<T>`]** / **[`ConvertFrom<T>`]** – Audio-aware conversions with
//!   correct scaling between bit depths (e.g. `i16 → f32` normalises to `[-1.0, 1.0]`).
//!
//! - **[`CastFrom<T>`]** / **[`CastInto<T>`]** – Raw numeric casts without scaling.
//!   Use these when you need the raw integer value as a float for computation,
//!   not as an audio amplitude.
//!
//! - **[`AudioTypeConversion`]** – High-level conversion methods on [`AudioSamples<T>`]
//!   such as `as_f32()`, `as_i16()`, and `as_type::<U>()`.
//!
//! ## Signal Generation
//!
//! The [`utils::generation`] module provides functions for creating test and
//! reference signals: [`sine_wave`], [`cosine_wave`], [`square_wave`],
//! [`triangle_wave`], [`sawtooth_wave`], [`chirp`], [`impulse`], [`silence`],
//! [`compound_tone`], and [`am_signal`].
//!
//! ```rust
//! use audio_samples::{sine_wave, sample_rate};
//! use audio_samples::utils::comparison;
//! use std::time::Duration;
//!
//! let a = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(44100), 1.0);
//! let b = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(44100), 1.0);
//! let corr: f64 = comparison::correlation(&a, &b).unwrap();
//! assert!(corr > 0.99);
//! ```
//!
//! ## Documentation
//!
//! Full API documentation is available at [docs.rs/audio_samples](https://docs.rs/audio_samples).
//!
//! ## License
//!
//! MIT
//!
//! ## Contributing
//!
//! Contributions are welcome. Please open an issue or pull request on
//! [GitHub](https://github.com/jmg049/audio_samples).

// General todos in audio_samples:
// - Define / decide on a policy for parallel processing
// - Improve the error system, it has breadth currently, but not depth or aethetics. Rust set a new standard for error handling (i.e. nice errors that tell you exactly what went wrong, where and how to possible fix things.) which other libraries in other languages have tried to emulate.
// - Better constants and better use of them -- e.g. the LEFT, RIGHT and SUPPORTED_DTYPES are rarely used

/// Creates a `NonZeroUsize` from a compile-time constant with zero-check.
///
/// This macro creates a `NonZeroUsize` with a compile-time assertion that the value is
/// non-zero, making it safe to use `new_unchecked` internally. If the value is zero, the
/// code will fail to compile.
///
/// # Example
///
/// ```
/// # use audio_samples::nzu;
/// let size = nzu!(1024);
/// assert_eq!(size.get(), 1024);
/// ```
///
/// This will fail to compile:
/// ```compile_fail
/// # use audio_samples::nzu;
/// let invalid = nzu!(0); // Compile error: assertion failed
/// ```
#[macro_export]
macro_rules! nzu {
    ($rate:expr) => {{
        const RATE: usize = $rate;
        const { assert!(RATE > 0, "non zero usize must be greater than 0") };
        // SAFETY: We just asserted RATE > 0 at compile time
        unsafe { ::core::num::NonZeroUsize::new_unchecked(RATE) }
    }};
}

/// Codec infrastructure: the [`AudioCodec`] trait, `encode`/`decode` free functions,
/// and the [`PerceptualCodec`] implementation (requires `feature = "psychoacoustic"`).
#[cfg(feature = "psychoacoustic")]
pub mod codecs;
pub mod conversions;
pub mod iterators;
pub mod operations;
#[cfg(feature = "resampling")]
pub mod resampling;
/// Core trait definitions for audio sample types and operations.
pub mod traits;
pub mod utils;

mod error;
/// Fixed-size audio buffer types whose geometry is encoded in the type system.
#[cfg(feature = "fixed-size-audio")]
pub mod fixed_audio;
mod repr;

pub mod simd_conversions;

/// Educational / explainable layer: step-by-step operation explanations with
/// before/after waveforms and term-maths formula rendering.
///
/// Enable with `--features educational`. See [`educational::open_explanation_document`]
/// for the primary entry point.
#[cfg(feature = "educational")]
pub mod educational;

pub use crate::error::{
    AudioSampleError, AudioSampleResult, ConversionError, FeatureError, LayoutError,
    ParameterError, ProcessingError,
};
pub use crate::repr::AudioData;
pub use crate::repr::StereoAudioSamples;
pub use crate::repr::{AudioSamples, SampleType};
pub use crate::traits::{
    AudioSample, AudioTypeConversion, CastFrom, CastInto, ConvertFrom, ConvertTo, StandardSample,
};

pub use crate::iterators::{AudioSampleIterators, ChannelIterator, FrameIterator, PaddingMode};

#[cfg(feature = "editing")]
pub use crate::iterators::WindowIterator;

pub use crate::utils::generation::{
    ToneComponent, am_signal, chirp, compound_tone, cosine_wave, impulse, sawtooth_wave, silence,
    sine_wave, square_wave, triangle_wave,
};

#[cfg(feature = "channels")]
pub use crate::utils::generation::{
    multichannel_compound_tone, stereo_chirp, stereo_silence, stereo_sine_wave,
};

#[cfg(feature = "statistics")]
pub use crate::operations::AudioStatistics;

#[cfg(feature = "processing")]
pub use crate::operations::AudioProcessing;

#[cfg(any(feature = "processing", feature = "peak-picking"))]
pub use crate::operations::types::{NormalizationConfig, NormalizationMethod};

#[cfg(feature = "channels")]
pub use crate::operations::AudioChannelOps;

#[cfg(feature = "editing")]
pub use crate::operations::AudioEditing;

#[cfg(feature = "transforms")]
pub use crate::operations::AudioTransforms;

#[cfg(feature = "fixed-size-audio")]
pub use crate::fixed_audio::{FixedSizeAudioSamples, FixedSizeMultiChannelAudioSamples};

#[cfg(feature = "onset-detection")]
pub use crate::operations::traits::AudioOnsetDetection;

#[cfg(feature = "decomposition")]
pub use crate::operations::traits::AudioDecomposition;

#[cfg(feature = "dynamic-range")]
pub use crate::operations::traits::AudioDynamicRange;

#[cfg(feature = "iir-filtering")]
pub use crate::operations::traits::AudioIirFiltering;

#[cfg(feature = "parametric-eq")]
pub use crate::operations::traits::AudioParametricEq;

#[cfg(feature = "pitch-analysis")]
pub use crate::operations::traits::AudioPitchAnalysis;

#[cfg(feature = "vad")]
pub use crate::operations::traits::AudioVoiceActivityDetection;

#[cfg(feature = "psychoacoustic")]
pub use crate::codecs::perceptual::{
    AudioCodec, AudioPerceptualAnalysis, Band, BandLayout, BandMetric, BandMetrics, EncodedSegment,
    PerceptualAnalysisResult, PerceptualCodec, PerceptualEncodedAudio, PsychoacousticConfig,
    StereoPerceptualCodec, StereoPerceptualEncodedAudio, analyse_signal_with_window_size,
    apply_temporal_masking, detect_transient_windows, reconstruct_signal,
};

#[cfg(feature = "psychoacoustic")]
pub use crate::codecs::perceptual::bands::scale_band_layout;

#[cfg(feature = "psychoacoustic")]
pub use crate::codecs::perceptual::masking::{
    MaskerType, absolute_threshold_of_hearing, classify_masker_types, compute_smr,
    spreading_attenuation,
};

#[cfg(feature = "psychoacoustic")]
pub use crate::codecs::perceptual::stereo::{mid_side_decode, mid_side_encode};

#[cfg(feature = "psychoacoustic")]
pub use crate::codecs::perceptual::quantization::{
    BandAllocation, BitAllocationResult, allocate_bits, dequantize, dequantize_band, quantize,
    quantize_band, step_size_from_allowed_noise,
};

#[cfg(feature = "psychoacoustic")]
pub use crate::codecs::perceptual::codec::{decode as codec_decode, encode as codec_encode};

#[cfg(feature = "random-generation")]
pub use crate::utils::generation::{brown_noise, pink_noise, white_noise};

#[cfg(feature = "educational")]
pub use explainable::{ExplainMode, Explainable, Explaining, Explanation};

#[cfg(all(feature = "educational", feature = "processing"))]
pub use crate::operations::traits::AudioProcessingExt;

pub use i24::{I24, PackedStruct}; // Re-export I24 type that has the AudioSample implementation

use ndarray::{Array1, Array2};
#[cfg(feature = "resampling")]
pub use resampling::{resample, resample_by_ratio};

/// A tagged result that is either a 1-D mono array or a 2-D multi-channel array.
///
/// ## Purpose
///
/// Several operations on [`AudioSamples<T>`] produce output whose dimensionality
/// mirrors the input: a mono input yields a 1-D result; a multi-channel input yields
/// a 2-D result. `NdResult` captures both possibilities in a single return type,
/// letting callers branch on the actual dimensionality at runtime rather than
/// discarding the channel structure.
///
/// ## Intended Usage
///
/// Pattern-match on the variants directly, or use [`into_array1`](NdResult::into_array1) /
/// [`into_array2`](NdResult::into_array2) when you already know the expected shape.
/// Prefer the safe helpers over the `_unchecked` variants unless performance is
/// critical and the shape has already been verified.
///
/// ## Invariants
///
/// - The [`Mono`](NdResult::Mono) variant always holds a 1-D array with at least one element.
/// - The [`MultiChannel`](NdResult::MultiChannel) variant always holds a 2-D array with at
///   least one channel and at least one sample per channel.
///
/// ## Assumptions
///
/// Callers must not construct `NdResult` directly with empty arrays.
/// All library functions that return `NdResult` uphold the non-empty invariants above.
#[non_exhaustive]
pub enum NdResult<T>
where
    T: StandardSample,
{
    /// A single-channel (mono) result stored as a 1-D ndarray.
    Mono(Array1<T>),
    /// A multi-channel result stored as a 2-D ndarray (channels × samples).
    MultiChannel(Array2<T>),
}

impl<T> NdResult<T>
where
    T: StandardSample,
{
    /// Extracts the inner 1-D array if this result is [`Mono`](NdResult::Mono).
    ///
    /// # Returns
    ///
    /// `Some(array)` when the variant is `Mono`; `None` when it is `MultiChannel`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::NdResult;
    /// use ndarray::array;
    ///
    /// let result: NdResult<f32> = NdResult::Mono(array![0.1, 0.2, 0.3]);
    /// assert!(result.into_array1().is_some());
    ///
    /// let result: NdResult<f32> = NdResult::MultiChannel(ndarray::array![[0.1, 0.2], [0.3, 0.4]]);
    /// assert!(result.into_array1().is_none());
    /// ```
    #[inline]
    #[must_use]
    pub fn into_array1(self) -> Option<Array1<T>> {
        match self {
            Self::Mono(arr) => Some(arr),
            Self::MultiChannel(_) => None,
        }
    }

    /// Extracts the inner 1-D array without checking the variant.
    ///
    /// # Safety
    ///
    /// The caller must ensure that this `NdResult` is the [`Mono`](NdResult::Mono)
    /// variant. Calling this on a [`MultiChannel`](NdResult::MultiChannel) value is
    /// a programmer error and will panic in debug builds. There is no undefined
    /// behaviour, but the panic is not recoverable.
    ///
    /// # Returns
    ///
    /// The inner `Array1<T>`.
    ///
    /// # Panics
    ///
    /// Panics if the variant is `MultiChannel`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::NdResult;
    /// use ndarray::array;
    ///
    /// let result: NdResult<f32> = NdResult::Mono(array![0.1, 0.2, 0.3]);
    /// // SAFETY: we just constructed a Mono variant above
    /// let arr = unsafe { result.into_array1_unchecked() };
    /// assert_eq!(arr.len(), 3);
    /// ```
    #[inline]
    #[must_use]
    pub unsafe fn into_array1_unchecked(self) -> Array1<T> {
        match self {
            Self::Mono(arr) => arr,
            Self::MultiChannel(_) => panic!("Cannot convert MultiChannel to Array1"),
        }
    }

    /// Extracts the inner 2-D array if this result is [`MultiChannel`](NdResult::MultiChannel).
    ///
    /// # Returns
    ///
    /// `Some(array)` when the variant is `MultiChannel`; `None` when it is `Mono`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::NdResult;
    /// use ndarray::array;
    ///
    /// let result: NdResult<f32> = NdResult::MultiChannel(array![[0.1, 0.2], [0.3, 0.4]]);
    /// assert!(result.into_array2().is_some());
    ///
    /// let result: NdResult<f32> = NdResult::Mono(array![0.1, 0.2, 0.3]);
    /// assert!(result.into_array2().is_none());
    /// ```
    #[inline]
    #[must_use]
    pub fn into_array2(self) -> Option<Array2<T>> {
        match self {
            Self::Mono(_) => None,
            Self::MultiChannel(arr) => Some(arr),
        }
    }

    /// Extracts the inner 2-D array without checking the variant.
    ///
    /// # Safety
    ///
    /// The caller must ensure that this `NdResult` is the
    /// [`MultiChannel`](NdResult::MultiChannel) variant. Calling this on a
    /// [`Mono`](NdResult::Mono) value is a programmer error and will panic in debug
    /// builds. There is no undefined behaviour, but the panic is not recoverable.
    ///
    /// # Returns
    ///
    /// The inner `Array2<T>`.
    ///
    /// # Panics
    ///
    /// Panics if the variant is `Mono`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::NdResult;
    /// use ndarray::array;
    ///
    /// let result: NdResult<f32> = NdResult::MultiChannel(array![[0.1, 0.2], [0.3, 0.4]]);
    /// // SAFETY: we just constructed a MultiChannel variant above
    /// let arr = unsafe { result.into_array2_unchecked() };
    /// assert_eq!(arr.nrows(), 2);
    /// ```
    #[inline]
    #[must_use]
    pub unsafe fn into_array2_unchecked(self) -> Array2<T> {
        match self {
            Self::Mono(_) => panic!("Cannot convert Mono to Array2"),
            Self::MultiChannel(arr) => arr,
        }
    }
}
