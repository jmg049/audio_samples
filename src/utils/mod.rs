//! Utility functions and helpers for audio processing.
//!
//! This module provides a comprehensive collection of utility functions that make common
//! audio processing tasks more convenient, intuitive, and robust. These utilities complement
//! the core audio processing operations with specialized tools for analysis, generation,
//! and comparison tasks.
//!
//! # Module Organization
//!
//! The utilities are organized into focused submodules, each addressing specific aspects
//! of audio processing:
//!
//! - [`comparison`] - Audio comparison, similarity metrics, and correlation analysis
//! - [`detection`] - Format detection, audio analysis, and content identification  
//! - [`generation`] - Signal generation, test tones, and synthetic audio creation
//!
//! # Design Philosophy
//!
//! ## Convenience Over Performance
//! While the core [`operations`](crate::operations) module prioritizes performance and
//! flexibility, these utilities prioritize ease of use and common-case optimization.
//!
//! ## Sensible Defaults
//! All utility functions provide reasonable defaults for their parameters, allowing
//! quick experimentation and prototyping without extensive configuration.
//!
//! ## Composability
//! Utilities are designed to work well together and with the core audio processing
//! pipeline, enabling complex workflows through simple function composition.
//!
//! # Quick Start Examples
//!
//! ## Audio Generation
//!
//! ```rust,ignore
//! use audio_samples::utils::generation::*;
//!
//! // Generate test tones
//! let sine_wave = generate_sine(440.0, 44100, 1.0); // 440Hz, 1 second
//! let white_noise = generate_white_noise(44100, 2.0); // 2 seconds
//! let chirp = generate_chirp(220.0, 880.0, 44100, 3.0); // Linear sweep
//!
//! // Generate complex signals
//! let multi_tone = generate_multi_tone(&[440.0, 880.0, 1320.0], 44100, 1.0);
//! ```
//!
//! ## Audio Analysis
//!
//! ```rust,ignore
//! use audio_samples::utils::detection::*;
//!
//! // Detect audio properties
//! let sample_rate = detect_sample_rate(&audio_data)?;
//! let is_stereo = detect_channel_layout(&audio_data)?;
//! let dynamic_range = estimate_dynamic_range(&audio_data)?;
//!
//! // Content analysis
//! let has_speech = detect_speech_activity(&audio_data)?;
//! let silence_ratio = detect_silence_ratio(&audio_data, threshold)?;
//! ```
//!
//! ## Audio Comparison
//!
//! ```rust,ignore
//! use audio_samples::utils::comparison::*;
//!
//! // Compare audio signals
//! let similarity = compute_similarity(&audio1, &audio2)?;
//! let correlation = cross_correlate(&audio1, &audio2)?;
//! let snr = signal_to_noise_ratio(&signal, &noise)?;
//!
//! // Quality metrics
//! let thd = total_harmonic_distortion(&audio)?;
//! let perceptual_distance = perceptual_audio_distance(&reference, &test)?;
//! ```
//!
//! # Integration with Core Operations
//!
//! Utilities are designed to integrate seamlessly with the core operations:
//!
//! ```rust,ignore
//! use audio_samples::{AudioSamples, operations::*, utils::*};
//!
//! // Generate test signal
//! let test_audio = generation::generate_sine(1000.0, 44100, 1.0);
//!
//! // Process with core operations
//! let mut processed = test_audio.normalize(-1.0, 1.0, NormalizationMethod::Peak)?;
//! processed.apply_window(WindowType::Hann)?;
//!
//! // Analyze with utilities
//! let quality_metrics = comparison::analyze_quality(&processed)?;
//! let content_type = detection::classify_audio_content(&processed)?;
//! ```
//!
//! # Common Use Cases
//!
//! ## Audio Testing and Validation
//! - Generate known test signals for algorithm validation
//! - Compare processed audio against reference signals
//! - Measure audio quality and distortion metrics
//!
//! ## Content Analysis
//! - Detect audio format and properties from unknown sources
//! - Classify audio content (speech, music, silence, etc.)
//! - Extract perceptual features for machine learning
//!
//! ## Signal Generation
//! - Create test tones for calibration and measurement
//! - Generate synthetic audio for training datasets
//! - Produce known signals for algorithm development
//!
//! ## Performance Benchmarking
//! - Generate standardized test inputs
//! - Measure processing quality objectively
//! - Compare different processing algorithms
//!
//! See individual submodules for detailed function documentation and examples.

pub mod comparison;
pub mod detection;
pub mod generation;

// Re-export common utilities
pub use comparison::*;
pub use detection::*;
pub use generation::*;

// Re-export existing utils from the parent module
// Note: Cannot glob-import a module into itself, so we'll define the compatibility exports individually
// pub use super::utils_old::*;
