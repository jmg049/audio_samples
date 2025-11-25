//! Composable plotting system for audio visualization.
//!
//! This module provides a flexible, composable system for creating audio visualizations
//! using Plotly. The system is designed around the concept of plot elements
//! that can be combined in various layouts and rendered to different backends.
//!
//! # High-Level API: `Plotting` trait
//!
//! The recommended way to plot audio is via the `Plotting` trait, which automatically
//! creates per-channel visualizations stacked vertically:
//!
//! ```rust,ignore
//! use audio_samples::{AudioSamples, operations::plotting::Plotting};
//!
//! let audio = AudioSamples::new_stereo(/* ... */);
//!
//! // Plot waveform - one subplot per channel
//! audio.plot_waveform::<f64>(None)?.show(true)?;
//!
//! // Plot spectrogram with custom config
//! let config = SpectrogramPlotConfig::high_resolution();
//! audio.plot_spectrogram::<f64>(Some(config))?.show(true)?;
//! ```
//!
//! # Low-Level API: `AudioPlotBuilders` trait
//!
//! For fine-grained control over individual plot elements, use `AudioPlotBuilders`:
//!
//! ```rust,ignore
//! use audio_samples::{AudioSamples, operations::plotting::*};
//!
//! let audio = AudioSamples::new_mono(/* ... */);
//!
//! // Create individual plot elements
//! let waveform = audio.waveform_plot::<f64>(None)?;
//! let spectrum = audio.power_spectrum_plot::<f64>(None, None, None, None, None)?;
//!
//! // Compose manually
//! PlotComposer::new()
//!     .add_element(waveform)
//!     .add_element(spectrum)
//!     .with_layout(LayoutConfig::VerticalStack)
//!     .with_title("Audio Analysis")
//!     .show(true)?;
//! ```
//!
//! # Architecture
//!
//! The plotting system consists of several key components:
//!
//! - **`Plotting` trait**: High-level API returning `PlotComposer` with per-channel plots
//! - **`PlotElement` trait**: Core abstraction for all plot components
//! - **Plot Elements**: Concrete implementations (waveforms, spectrograms, overlays)
//! - **`PlotComposer`**: Composition and layout system
//! - **`AudioPlotBuilders`**: Low-level builders for individual elements

pub mod builders;
pub mod composer;
pub mod core;
pub mod elements;
pub mod plotting_trait;

// Re-export the main types for easy access
pub use builders::*;
pub use composer::*;
pub use core::*;
pub use elements::*;
pub use plotting_trait::*;
