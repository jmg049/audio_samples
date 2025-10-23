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
//! - [`conversions`] - Type conversion operations
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

pub mod beats;
pub mod channels;
pub mod dynamic_range;
pub mod editing;
pub mod fft_backends;
pub mod iir_filtering;
pub mod onset_detection;
pub mod parametric_eq;
pub mod peak_picking;
pub mod pitch_analysis;
pub mod plotting;
pub mod processing;
pub mod statistics;
pub mod traits;
pub mod transforms;
pub mod types;

// Re-export main traits for convenience
pub use traits::{
    AudioChannelOps, AudioDynamicRange, AudioEditing, AudioIirFiltering, AudioParametricEq,
    AudioPlottingUtils, AudioProcessing, AudioSamplesOperations, AudioStatistics, AudioTransforms,
};

// Re-export builder types
pub use processing::ProcessingBuilder;

// Re-export plotting types and functions
pub use plotting::{
    ComparisonPlotOptions, PlotResult, SpectrogramPlotOptions, WaveformPlotOptions,
    plot_comparison, plot_difference, plot_spectrogram, plot_waveform, time_ticks_seconds,
};

// Re-export supporting types
pub use types::{
    ChannelConversionMethod, CqtConfig, MonoConversionMethod, NormalizationMethod, OnsetConfig,
    PeakPickingConfig, ResamplingQuality, SpectralFluxConfig, SpectralFluxMethod,
    StereoConversionMethod,
};

pub use beats::*;
