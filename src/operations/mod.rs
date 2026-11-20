//! Audio processing operations and transformations.
//!
//! This module provides the audio processing operations, organized into
//! composable traits. Each trait covers one area of audio processing.
//!
//! ## Module Organization
//!
//! - [`traits`] - Core trait definitions
#![cfg_attr(
    feature = "statistics",
    doc = r#" - [`statistics`] - Statistical analysis operations"#
)]
#![cfg_attr(
    not(feature = "statistics"),
    doc = r#" - `statistics` - Statistical analysis operations (requires the `statistics` feature)"#
)]
#![cfg_attr(
    feature = "processing",
    doc = r#" - [`processing`] - Signal processing operations"#
)]
#![cfg_attr(
    not(feature = "processing"),
    doc = r#" - `processing` - Signal processing operations (requires the `processing` feature)"#
)]
#![cfg_attr(
    feature = "transforms",
    doc = r#" - [`transforms`] - FFT / spectral analysis"#
)]
#![cfg_attr(
    not(feature = "transforms"),
    doc = r#" - `transforms` - FFT / spectral analysis (requires the `transforms` feature)"#
)]
#![cfg_attr(
    feature = "transforms",
    doc = r#" - [`spectrum`] - STFT, mel spectrogram, MFCC, dB, delta"#
)]
#![cfg_attr(
    not(feature = "transforms"),
    doc = r#" - `spectrum` - STFT, mel spectrogram, MFCC, dB, delta (requires the `transforms` feature)"#
)]
#![cfg_attr(
    feature = "effects",
    doc = r#" - [`effects`] - time stretch, pitch shift, emphasis, dB trim/split, HPSS"#
)]
#![cfg_attr(
    not(feature = "effects"),
    doc = r#" - `effects` - time stretch, pitch shift, emphasis, dB trim/split, HPSS (requires the `effects` feature)"#
)]
#![cfg_attr(
    all(feature = "transforms", feature = "resampling"),
    doc = r#" - [`constant_q`] - CQT / VQT / iCQT, CQT chroma and tonnetz"#
)]
#![cfg_attr(
    not(all(feature = "transforms", feature = "resampling")),
    doc = r#" - `constant_q` - CQT / VQT / iCQT, CQT chroma and tonnetz (requires the `transforms` and `resampling` features)"#
)]
#![cfg_attr(
    feature = "transforms",
    doc = r#" - [`filterbanks`] - window functions and mel filterbanks"#
)]
#![cfg_attr(
    not(feature = "transforms"),
    doc = r#" - `filterbanks` - window functions and mel filterbanks (requires the `transforms` feature)"#
)]
#![cfg_attr(
    feature = "editing",
    doc = r#" - [`editing`] - Time-domain editing operations"#
)]
#![cfg_attr(
    not(feature = "editing"),
    doc = r#" - `editing` - Time-domain editing operations (requires the `editing` feature)"#
)]
#![cfg_attr(
    feature = "channels",
    doc = r#" - [`channels`] - Channel manipulation operations"#
)]
#![cfg_attr(
    not(feature = "channels"),
    doc = r#" - `channels` - Channel manipulation operations (requires the `channels` feature)"#
)]
//! - [`types`] - Supporting types and enums
//!
#![cfg_attr(
    feature = "plotting",
    doc = r#" Additional submodules are feature-gated, e.g. [`plotting`] (requires `plotting`) and"#
)]
#![cfg_attr(
    not(feature = "plotting"),
    doc = r#" Additional submodules are feature-gated, e.g. `plotting` (requires `plotting`) and"#
)]
#![cfg_attr(feature = "vad", doc = r#" [`vad`] (requires `vad`)."#)]
#![cfg_attr(not(feature = "vad"), doc = r#" `vad` (requires `vad`)."#)]
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! // Full example requires the `statistics` and `processing` features.
//! use audio_samples::{AudioSamples, AudioTypeConversion, sample_rate};
//! use audio_samples::operations::types::{NormalizationConfig, NormalizationMethod};
//! use audio_samples::operations::traits::{AudioProcessing, AudioStatistics};
//! use ndarray::array;
//!
//! let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
//! let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
//!
//! // Statistical analysis (requires `statistics` feature)
//! let peak = audio.peak();
//! let rms = audio.rms();
//! let _ = (peak, rms);
//!
//! // Signal processing (requires `processing` feature)
//! let config = NormalizationConfig::min_max(-1.0, 1.0);
//! let normalized = audio.normalize(config).unwrap();
//!
//! // Type conversion (always available)
//! let audio_i16 = normalized.to_type::<i16>();
//! let _ = audio_i16;
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
pub(crate) mod iir_design;
#[cfg(feature = "iir-filtering")]
pub mod iir_filtering;

#[cfg(feature = "parametric-eq")]
pub mod parametric_eq;

#[cfg(feature = "peak-picking")]
pub mod peak_picking;

#[cfg(feature = "pitch-analysis")]
pub mod pitch_analysis;

#[cfg(feature = "pitch-analysis")]
pub mod pitch_tracking;

#[cfg(feature = "processing")]
pub mod processing;

#[cfg(feature = "statistics")]
pub mod statistics;

#[cfg(feature = "transforms")]
pub mod transforms;

#[cfg(feature = "transforms")]
pub mod filterbanks;

#[cfg(feature = "transforms")]
pub mod spectrum;

#[cfg(all(feature = "transforms", feature = "resampling"))]
pub mod constant_q;

#[cfg(feature = "vad")]
pub mod vad;

#[cfg(feature = "onset-detection")]
pub mod onset;

#[cfg(feature = "plotting")]
pub mod plotting;

#[cfg(feature = "envelopes")]
pub mod envelopes;

#[cfg(feature = "sequence")]
pub mod sequence;

#[cfg(feature = "linalg")]
pub mod linalg;

#[cfg(feature = "feature-inversion")]
pub mod inverse;

#[cfg(feature = "factorization")]
pub mod factorization;

#[cfg(feature = "pitch-filterbank")]
pub mod pitch_filterbank;

#[cfg(feature = "mellin")]
pub mod mellin;

#[cfg(feature = "segmentation")]
pub mod segment;

#[cfg(feature = "feature-extraction")]
pub mod tonal;

#[cfg(feature = "feature-extraction")]
pub mod features;
#[cfg(feature = "feature-extraction")]
pub use traits::AudioFeatures;

#[cfg(feature = "effects")]
pub mod effects;

#[cfg(feature = "beat-tracking")]
pub mod rhythm;

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

#[cfg(feature = "pitch-analysis")]
pub use traits::AudioPitchTracking;

#[cfg(feature = "dynamic-range")]
pub use traits::AudioDynamicRange;

#[cfg(feature = "transforms")]
pub use traits::AudioTransforms;

#[cfg(feature = "transforms")]
pub use traits::AudioSpectrum;

#[cfg(feature = "feature-extraction")]
pub use traits::AudioTonal;

#[cfg(all(feature = "transforms", feature = "resampling"))]
pub use traits::AudioConstantQ;
#[cfg(feature = "effects")]
pub use traits::AudioEffects;

#[cfg(feature = "decomposition")]
pub use traits::AudioDecomposition;

#[cfg(feature = "beat-tracking")]
pub use traits::AudioBeatTracking;

#[cfg(feature = "beat-tracking")]
pub use beat::*;

#[cfg(feature = "beat-tracking")]
pub use traits::AudioRhythm;

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
pub use plotting::phase_spectrum::{
    PhaseSpectrumParams, PhaseSpectrumPlot, create_phase_spectrum_plot,
};

#[cfg(feature = "plotting")]
pub use plotting::lissajous::{LissajousParams, LissajousPlot, create_lissajous_plot};

#[cfg(feature = "plotting")]
pub use plotting::{ChannelManagementStrategy, Layout, PlotParams, PlotUtils};

#[cfg(feature = "plotting")]
pub use plotting::dsp_overlays;

#[cfg(feature = "resampling")]
pub use types::ResamplingQuality;
