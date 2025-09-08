//! Tests for streaming functionality.
//!
//! This module contains comprehensive tests for all streaming components including
//! sources, processors, buffers, and error handling.

use crate::{AudioSample, AudioSamples, ConvertTo};

// Include test modules from sibling files
mod buffer_tests;
mod error_tests;
mod generator_tests;
mod stream_tests;

/// Helper function to create test audio data
pub(crate) fn create_test_audio_data<T: AudioSample>(
    samples: usize,
    channels: usize,
    sample_rate: u32,
) -> AudioSamples<T>
where
    f64: crate::ConvertTo<T>,
{
    let mut data = Vec::with_capacity(samples * channels);
    for i in 0..samples {
        for _ch in 0..channels {
            let sample_value = (i as f64 / samples as f64).sin();
            data.push(sample_value.convert_to().unwrap());
        }
    }

    let array = ndarray::Array2::from_shape_vec((samples, channels), data)
        .expect("Failed to create test array");
    AudioSamples::new_multi_channel(array, sample_rate)
}

/// Helper function to validate audio samples format
pub(crate) fn validate_audio_format<T: AudioSample>(
    audio: &AudioSamples<T>,
    expected_samples: usize,
    expected_channels: usize,
    expected_sample_rate: u32,
) {
    assert_eq!(
        audio.samples_per_channel(),
        expected_samples,
        "Sample count mismatch"
    );
    assert_eq!(
        audio.num_channels(),
        expected_channels,
        "Channel count mismatch"
    );
    assert_eq!(
        audio.sample_rate(),
        expected_sample_rate,
        "Sample rate mismatch"
    );
}
