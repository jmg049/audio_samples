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
//! - [`transforms`] - FFT / spectral analysis
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
//! let audio = AudioSamples::new_mono(data, 44100).unwrap();
//!
//! // Statistical analysis
//! let peak = audio.peak(); // Returns f32 directly
//! let rms = audio.rms::<f32>();
//! let _ = (peak, rms);
//!
//! // Signal processing
//! let audio_copy = audio.clone();
//! // Requires the `processing` feature.
//! let audio_copy = audio_copy
//!     .normalize(NormalizationConfig::min_max(-1.0, 1.0))
//!     .unwrap();
//! # let _ = audio_copy;
//!
//! // Type conversion
//! // Requires `AudioTypeConversion` (re-exported at the crate root).
//! let audio_i16 = audio.as_type::<i16>().unwrap();
//! let _ = audio_i16;
//! # Ok(())
//! # }
//! ```
pub mod traits;
pub mod types;

#[cfg(feature = "beat-tracking")]
pub mod beat;

#[cfg(feature = "channels")]
pub mod channels;

#[cfg(feature = "dynamic-range")]
pub mod dynamic_range;

#[cfg(feature = "editing")]
pub mod editing;

#[cfg(feature = "decomposition")]
pub mod hpss;

#[cfg(feature = "iir-filtering")]
pub mod iir_filtering;

#[cfg(feature = "parametric-eq")]
pub mod parametric_eq;

#[cfg(feature = "peak-picking")]
pub mod peak_picking;

#[cfg(feature = "pitch-analysis")]
pub mod pitch_analysis;

#[cfg(feature = "processing")]
pub mod processing;

#[cfg(feature = "statistics")]
pub mod statistics;

#[cfg(feature = "transforms")]
pub mod transforms;

#[cfg(feature = "vad")]
pub mod vad;

#[cfg(feature = "onset-detection")]
pub mod onset;

#[cfg(feature = "plotting")]
pub mod plotting;

#[cfg(feature = "envelopes")]
pub mod envelopes;

// Re-export main traits for convenience
#[cfg(feature = "statistics")]
pub use traits::AudioStatistics;

#[cfg(feature = "vad")]
pub use traits::AudioVoiceActivityDetection;

#[cfg(feature = "processing")]
pub use traits::AudioProcessing;

#[cfg(feature = "channels")]
pub use traits::AudioChannelOps;

#[cfg(feature = "editing")]
pub use traits::AudioEditing;

#[cfg(feature = "iir-filtering")]
pub use traits::AudioIirFiltering;

#[cfg(feature = "parametric-eq")]
pub use traits::AudioParametricEq;

#[cfg(feature = "peak-picking")]
pub use crate::operations::peak_picking::*;

#[cfg(feature = "pitch-analysis")]
pub use traits::AudioPitchAnalysis;

#[cfg(feature = "dynamic-range")]
pub use traits::AudioDynamicRange;

#[cfg(feature = "transforms")]
pub use traits::AudioTransforms;

#[cfg(feature = "decomposition")]
pub use traits::AudioDecomposition;

#[cfg(feature = "beat-tracking")]
pub use traits::AudioBeatTracking;

#[cfg(feature = "beat-tracking")]
pub use beat::*;

#[cfg(feature = "onset-detection")]
pub use onset::*;

#[cfg(feature = "decomposition")]
pub use hpss::*;

#[cfg(feature = "plotting")]
pub use traits::AudioPlotting;

#[cfg(feature = "plotting")]
pub use plotting::waveform::{WaveformPlot, WaveformPlotParams, create_waveform_plot};

#[cfg(feature = "envelopes")]
pub use traits::AudioEnvelopes;

#[cfg(feature = "plotting")]
pub use plotting::spectrograms::{
    SpectrogramPlot, SpectrogramPlotParams, SpectrogramType, create_spectrogram_plot,
};

#[cfg(feature = "plotting")]
pub use plotting::composite::{CompositeLayout, CompositePlot, PlotComponent};

#[cfg(feature = "plotting")]
pub use plotting::spectrum::{
    MagnitudeSpectrumParams, MagnitudeSpectrumPlot, create_magnitude_spectrum_plot,
};

#[cfg(feature = "plotting")]
pub use plotting::{ChannelManagementStrategy, Layout, PlotParams, PlotUtils};

#[cfg(feature = "plotting")]
pub use plotting::dsp_overlays;

#[cfg(feature = "resampling")]
pub use types::ResamplingQuality;
