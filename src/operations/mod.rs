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
//! - [`transforms`] - FFT and spectral analysis
//! - [`editing`] - Time-domain editing operations
//! - [`channels`] - Channel manipulation operations
//! - [`types`] - Supporting types and enums
//!
//! ## Quick Start
//!
//! ```rust
//! use audio_samples::{AudioSamples, operations::*};
//! use ndarray::array;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
//! let audio = AudioSamples::new_mono(data, 44100);
//!
//! // Statistical analysis
//! let peak = audio.peak(); // Returns f32 directly
//! let rms = audio.rms()?;  // Returns Result<f64>
//!
//! // Signal processing
//! let mut audio_copy = audio.clone();
//! audio_copy.normalize(-1.0, 1.0, NormalizationMethod::MinMax)?;
//!
//! // Type conversion
//! let audio_i16 = audio.as_type::<i16>()?;
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
pub mod statistics;
pub mod traits;
#[cfg(feature = "spectral-analysis")]
pub mod transforms;
pub mod types;

// Re-export main traits for convenience
pub use traits::{
    AudioChannelOps, AudioDynamicRange, AudioEditing, AudioIirFiltering, AudioParametricEq,
    AudioProcessing, AudioSamplesOperations, AudioStatistics,
};

#[cfg(feature = "spectral-analysis")]
pub use traits::AudioTransforms;

#[cfg(feature = "plotting")]
pub use traits::AudioPlottingUtils;

// Re-export builder types
pub use processing::ProcessingBuilder;

// Re-export plotting types and functions (composable API)
#[cfg(feature = "plotting")]
pub use plotting::{
    AudioPlotBuilders,
    BeatConfig,
    BeatMarkers,
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
    PlotMetadata,
    PlotResult,

    PlotTheme,
    PowerSpectrumPlot,
    SpectrogramConfig,
    SpectrogramPlot,
    // Plot elements
    WaveformPlot,
};

// Re-export supporting types
pub use types::{
    ChannelConversionMethod, CqtConfig, MonoConversionMethod, NormalizationMethod,
    PeakPickingConfig, ResamplingQuality, SpectralFluxConfig, SpectralFluxMethod,
    StereoConversionMethod,
};

#[cfg(feature = "beat-detection")]
pub use beats::*;
