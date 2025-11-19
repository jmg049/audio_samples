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

use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, I24, ParameterError, RealFloat, to_precision,
};

/// Converts a byte slice into a single audio sample of type T.
pub fn audio_sample_from_bytes<T: AudioSample>(bytes: &[u8]) -> AudioSampleResult<T> {
    let sample_size = std::mem::size_of::<T>();
    if bytes.len() != sample_size {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "input",
            format!(
                "Data length {} does not match sample size {}",
                bytes.len(),
                sample_size
            ),
        )));
    }

    match sample_size {
        1 => {
            let array: [u8; 1] = bytes.try_into().map_err(|_| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "input",
                    "Failed to convert bytes to array",
                ))
            })?;
            Ok(unsafe { std::mem::transmute_copy(&array) })
        }
        2 => {
            let array: [u8; 2] = bytes.try_into().map_err(|_| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "input",
                    "Failed to convert bytes to array",
                ))
            })?;
            Ok(unsafe { std::mem::transmute_copy(&array) })
        }
        3 => {
            let array: [u8; 3] = bytes.try_into().map_err(|_| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "input",
                    "Failed to convert bytes to array",
                ))
            })?;
            let i24 = I24::from_le_bytes(array);
            Ok(T::cast_from(i24))
        }
        4 => {
            let array: [u8; 4] = bytes.try_into().map_err(|_| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "input",
                    "Failed to convert bytes to array",
                ))
            })?;
            Ok(unsafe { std::mem::transmute_copy(&array) })
        }
        8 => {
            let array: [u8; 8] = bytes.try_into().map_err(|_| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "input",
                    "Failed to convert bytes to array",
                ))
            })?;
            Ok(unsafe { std::mem::transmute_copy(&array) })
        }
        _ => Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "input",
            format!("Unsupported sample size: {}", sample_size),
        ))),
    }
}

/// Convert bytes to aligned samples, creating a Vec when alignment is required
pub fn bytes_to_samples_aligned<T: AudioSample>(bytes: &[u8]) -> AudioSampleResult<Vec<T>> {
    let sample_size = std::mem::size_of::<T>();
    if !bytes.len().is_multiple_of(sample_size) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "input",
            "Data size is not a multiple of sample size",
        )));
    }

    let samples = if T::BITS != 24 {
        let num_samples = bytes.len() / sample_size;
        let ptr = bytes.as_ptr();
        let alignment = std::mem::align_of::<T>();

        if (ptr as usize).is_multiple_of(alignment) {
            // Safe to cast directly if aligned
            let slice = unsafe { std::slice::from_raw_parts(ptr as *const T, num_samples) };
            slice.to_vec()
        } else {
            // Need to copy data to ensure alignment
            let mut result = Vec::<T>::with_capacity(num_samples);
            unsafe {
                let mut byte_ptr = ptr;
                for _ in 0..num_samples {
                    // Read sample bytes into an aligned buffer
                    let mut sample_bytes = [0u8; 8]; // Max size for any AudioSample
                    std::ptr::copy_nonoverlapping(byte_ptr, sample_bytes.as_mut_ptr(), sample_size);
                    let sample = std::ptr::read(sample_bytes.as_ptr() as *const T);
                    result.push(sample);
                    byte_ptr = byte_ptr.add(sample_size);
                }
            }
            result
        }
    } else {
        // Handle I24 case
        let i24_samples = I24::read_i24s_le_slice(bytes).ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "input",
                "Invalid I24 data alignment or size",
            ))
        })?;

        let mut result = Vec::<T>::with_capacity(i24_samples.len());
        for i24_sample in i24_samples {
            // Safe conversion since T should be I24 when T::BITS == 24
            let sample = unsafe { std::ptr::read(i24_sample as *const I24 as *const T) };
            result.push(sample);
        }
        result
    };

    Ok(samples)
}

/// Convert bytes to samples with alignment checking (unsafe but fast when aligned)
///
/// # Safety
/// The caller must ensure that:
/// - `bytes` points to valid, properly aligned memory for type `T`
/// - The memory region represents valid samples of type `T`
/// - The byte slice length is a multiple of `size_of::<T>()`
/// - The data is not mutated elsewhere while the returned slice is borrowed
///
/// Undefined behaviour results if these conditions are not met.
pub unsafe fn bytes_to_samples_unchecked<T: AudioSample>(bytes: &[u8]) -> AudioSampleResult<&[T]> {
    let sample_size = std::mem::size_of::<T>();
    if !bytes.len().is_multiple_of(sample_size) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "input",
            "Data size is not a multiple of sample size",
        )));
    }

    let slice = if T::BITS != 24 {
        let num_samples = bytes.len() / sample_size;
        let ptr = bytes.as_ptr();
        let alignment = std::mem::align_of::<T>();

        // Check alignment before proceeding
        if !(ptr as usize).is_multiple_of(alignment) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "data_alignment",
                format!(
                    "Data is not properly aligned for type {} (requires {}-byte alignment)",
                    std::any::type_name::<T>(),
                    alignment
                ),
            )));
        }

        unsafe { std::slice::from_raw_parts(ptr as *const T, num_samples) }
    } else {
        let samples = I24::read_i24s_le_slice(bytes).ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "input",
                "Invalid I24 data alignment or size",
            ))
        })?;

        // For I24, we can safely cast since I24 has the same memory layout
        let num_samples = samples.len();
        let ptr = samples.as_ptr() as *const T;
        unsafe { std::slice::from_raw_parts(ptr, num_samples) }
    };

    Ok(slice)
}

/// Helper function to convert seconds to samples
/// Converts time in seconds to number of samples at given sample rate
///
/// # Arguments
/// - `seconds`: Duration in seconds
/// - `sample_rate`: Sampling frequency in Hz
///
/// # Returns
/// Number of samples representing the specified duration
///
/// # Panics
/// Panics if the computed sample count cannot be converted to `usize`,
/// typically when the result would overflow or is infinite/NaN.
pub fn seconds_to_samples<F: RealFloat>(seconds: F, sample_rate: u32) -> usize {
    (seconds * to_precision::<F, _>(sample_rate))
        .to_usize()
        .expect("Invalid sample rate")
}

/// Converts a number of samples to duration in seconds.
pub fn samples_to_seconds<F: RealFloat>(num_samples: usize, sample_rate: u32) -> F {
    to_precision::<F, _>(num_samples) / to_precision::<F, _>(sample_rate)
}
