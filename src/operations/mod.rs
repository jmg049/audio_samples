//! Audio processing operations and transformations.
//!
//! This module provides a comprehensive set of audio processing capabilities
//! organized into focused, composable traits. Each trait handles a specific
//! aspect of audio processing to maintain clean separation of concerns.
//!
//! ## Module Organization
//!
//! - [`traits`] - Core trait definitions
//! - [`statistics`] - Statistical analysis operations
//! - [`processing`] - Signal processing operations
//! - [`transforms`] - FFT / spectral analysis (requires `spectral-analysis`)
//! - [`fft_backends`] - FFT backend selection (requires `fft`)
//! - [`editing`] - Time-domain editing operations
//! - [`channels`] - Channel manipulation operations
//! - [`types`] - Supporting types and enums
//!
//! Additional submodules are feature-gated, e.g. `plotting` (requires `plotting`) and
//! `serialization` (requires `serialization`).
//!
//! ## Quick Start
//!
//! ```rust
//! use audio_samples::AudioSamples;
//! use audio_samples::operations::types::NormalizationMethod;
//! use audio_samples::operations::traits::{AudioProcessing, AudioStatistics};
//! use ndarray::array;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
//! let audio = AudioSamples::new_mono(data, 44100);
//!
//! // Statistical analysis
//! let peak = audio.peak(); // Returns f32 directly
//! let rms = audio.rms::<f32>();
//! let _ = (peak, rms);
//!
//! // Signal processing
//! let mut audio_copy = audio.clone();
//! // Requires the `processing` feature.
//! audio_copy
//!     .normalize(-1.0, 1.0, NormalizationMethod::MinMax)
//!     .unwrap();
//!
//! // Type conversion
//! // Requires `AudioTypeConversion` (re-exported at the crate root).
//! let audio_i16 = audio.as_type::<i16>().unwrap();
//! let _ = audio_i16;
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "beat-detection")]
/// Beat tracking and tempo analysis operations.
pub mod beats;
pub mod channels;
pub mod dynamic_range;
pub mod editing;
#[cfg(feature = "fft")]
pub mod fft_backends;
#[cfg(feature = "hpss")]
pub mod hpss;
pub mod iir_filtering;
#[cfg(feature = "spectral-analysis")]
pub mod onset_detection;
pub mod parametric_eq;
pub mod peak_picking;
#[cfg(feature = "spectral-analysis")]
pub mod pitch_analysis;
#[cfg(feature = "plotting")]
pub mod plotting;
pub mod processing;
#[cfg(feature = "serialization")]
pub mod serialization;
pub mod statistics;
pub mod traits;
#[cfg(feature = "spectral-analysis")]
pub mod transforms;
pub mod types;
#[cfg(feature = "statistics")]
pub mod vad;

// Re-export main traits for convenience
#[cfg(feature = "statistics")]
pub use traits::AudioStatistics;

#[cfg(feature = "statistics")]
pub use traits::AudioVoiceActivityDetection;

#[cfg(feature = "processing")]
pub use traits::AudioProcessing;

#[cfg(feature = "editing")]
pub use traits::AudioEditing;

#[cfg(feature = "channels")]
pub use traits::AudioChannelOps;

#[cfg(feature = "core-ops")]
pub use traits::{AudioDynamicRange, AudioIirFiltering, AudioParametricEq};

#[cfg(feature = "spectral-analysis")]
pub use traits::AudioTransforms;

#[cfg(feature = "plotting")]
pub use traits::AudioPlottingUtils;

#[cfg(feature = "serialization")]
pub use traits::AudioSamplesSerialise;

#[cfg(feature = "hpss")]
pub use traits::AudioDecomposition;

// Re-export builder types
pub use processing::ProcessingBuilder;

// Re-export plotting types and functions (composable API)
#[cfg(feature = "plotting")]
pub use plotting::{
    AudioPlotBuilders,
    BeatMarkers,
    BeatPlotConfig,
    // Styling and configuration
    ColorPalette,
    LayoutConfig,
    LineStyle,
    LineStyleType,
    MarkerShape,
    MarkerStyle,
    OnsetConfig,
    OnsetMarkers,
    PitchContour,
    PitchDetectionMethod,

    PlotBounds,
    // Core plotting API
    PlotComposer,
    PlotElement,
    PlotHandle,
    PlotMetadata,
    PlotResult,

    PlotTheme,
    // High-level Plotting trait and config types
    Plotting,
    PowerSpectrumPlot,
    SpectrogramConfig,
    SpectrogramPlot,
    SpectrogramPlotConfig,
    SpectrumPlotConfig,
    // Plot elements
    WaveformPlot,
    WaveformPlotConfig,
    channel_label,
};

// Re-export supporting types
pub use types::{
    ChannelConversionMethod, CqtConfig, MonoConversionMethod, NormalizationMethod,
    PeakPickingConfig, ResamplingQuality, SpectralFluxConfig, SpectralFluxMethod,
    StereoConversionMethod,
};

#[cfg(feature = "hpss")]
pub use types::HpssConfig;

#[cfg(feature = "serialization")]
pub use types::{Endianness, SerializationConfig, SerializationFormat, TextDelimiter};

#[cfg(feature = "beat-detection")]
pub use beats::*;

pub use channels::deinterleave;
