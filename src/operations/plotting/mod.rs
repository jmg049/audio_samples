//! Composable plotting system for audio visualization.
//!
//! This module provides a flexible, composable system for creating audio visualizations
//! using the plotters crate. The system is designed around the concept of plot elements
//! that can be combined in various layouts and rendered to different backends.
//!
//! # Quick Start
//!
//! ```rust
//! use audio_samples::{AudioSamples, operations::plotting::*};
//! use ndarray::Array1;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create some test audio
//! let data = Array1::from_vec(vec![0.1f32, 0.2, 0.3, 0.2, 0.1]);
//! let audio = AudioSamples::new_mono(data, 44100);
//!
//! // Create individual plot elements
//! let waveform = audio.waveform_plot(None);
//! let spectrum = audio.power_spectrum_plot(None)?;
//!
//! // Compose and render
//! let plot = PlotComposer::new()
//!     .add_element(waveform)
//!     .add_element(spectrum)
//!     .with_layout(LayoutConfig::VerticalStack)
//!     .with_title("Audio Analysis".to_string());
//!
//! plot.render_to_file("analysis.png")?;
//! # Ok(())
//! # }
//! ```
//!
//! # Architecture
//!
//! The plotting system consists of several key components:
//!
//! - **PlotElement trait**: Core abstraction for all plot components
//! - **Plot Elements**: Concrete implementations (waveforms, spectrograms, overlays)
//! - **PlotComposer**: Composition and layout system
//! - **Builders**: Integration with AudioSamples for easy plot creation

pub mod builders;
pub mod composer;
pub mod core;
pub mod elements;

// Re-export the main types for easy access
pub use builders::*;
pub use composer::*;
pub use core::*;
pub use elements::*;
