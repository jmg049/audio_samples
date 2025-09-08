//! Tests for streaming buffer implementations.

use super::super::buffers::*;
use crate::AudioSamples;
use ndarray::Array1;

#[test]
fn test_circular_buffer_basic_operations() {
    let config = BufferConfig::default();
    let mut buffer = CircularBuffer::<f32>::new(config);

    // Test initial state
    assert_eq!(buffer.len(), 0);
    assert!(buffer.is_empty());

    // Create test audio chunks
    let test_data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let audio_chunk = AudioSamples::new_mono(test_data, 44100);

    // Test pushing data
    let result = buffer.push(audio_chunk);
    assert!(result.is_ok());
    assert_eq!(buffer.len(), 1);
    assert!(!buffer.is_empty());

    // Test popping data
    let popped = buffer.pop();
    assert!(popped.is_some());
    if let Some(chunk) = popped {
        assert_eq!(chunk.samples_per_channel(), 5);
        assert_eq!(chunk.sample_rate(), 44100);
    }
    assert_eq!(buffer.len(), 0);
}

#[test]
fn test_circular_buffer_multiple_chunks() {
    let config = BufferConfig {
        max_chunks: 3,
        max_samples: 1000,
        ..BufferConfig::default()
    };
    let mut buffer = CircularBuffer::<i16>::new(config);

    // Add multiple chunks
    for i in 1..=3 {
        let data = Array1::from_vec(vec![i as i16 * 10, i as i16 * 20]);
        let chunk = AudioSamples::new_mono(data, 44100);
        let result = buffer.push(chunk);
        assert!(result.is_ok());
    }

    assert_eq!(buffer.len(), 3);

    // Try to add one more (should trigger overflow handling)
    let overflow_data = Array1::from_vec(vec![40, 50]);
    let overflow_chunk = AudioSamples::new_mono(overflow_data, 44100);
    let result = buffer.push(overflow_chunk);
    assert!(result.is_ok()); // Should succeed due to drop_on_overflow

    // Should still have 3 chunks (oldest was dropped)
    assert_eq!(buffer.len(), 3);
}

#[test]
fn test_circular_buffer_underrun_detection() {
    let config = BufferConfig {
        max_chunks: 10,
        max_samples: 1000,
        min_level: 0.25,
        ..BufferConfig::default()
    };
    let mut buffer = CircularBuffer::<f32>::new(config);

    // Buffer should be at underrun level when empty
    assert!(buffer.is_underrun());

    // Add some data
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let chunk = AudioSamples::new_mono(data, 44100);
    let _ = buffer.push(chunk);

    // Should still be at underrun (small amount of data)
    assert!(buffer.is_underrun());
}

#[test]
fn test_buffer_clear() {
    let config = BufferConfig::default();
    let mut buffer = CircularBuffer::<f64>::new(config);

    // Add data
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let chunk = AudioSamples::new_mono(data, 44100);
    let _ = buffer.push(chunk);
    assert_eq!(buffer.len(), 1);

    // Clear buffer
    buffer.clear();
    assert!(buffer.is_empty());
    assert_eq!(buffer.len(), 0);
}

#[cfg(feature = "streaming")]
#[test]
fn test_stream_buffer_basic() {
    let config = BufferConfig {
        max_chunks: 10,
        max_samples: 1024,
        ..BufferConfig::default()
    };
    let buffer = StreamBuffer::<f32>::new(config);

    // Test initial state
    assert!(buffer.is_underrun()); // Empty buffer is underrun

    // Create test chunk
    let data = Array1::from_vec(vec![0.5, 0.6, 0.7]);
    let chunk = AudioSamples::new_mono(data, 44100);

    // Test push operation
    let result = buffer.try_push(chunk);
    assert!(result.is_ok());

    // Test pop operation
    let popped = buffer.try_pop();
    assert!(popped.is_some());
    if let Some(chunk) = popped {
        assert_eq!(chunk.samples_per_channel(), 3);
        assert_eq!(chunk.sample_rate(), 44100);
    }
}

#[test]
fn test_buffer_config_validation() {
    let config = BufferConfig {
        max_chunks: 32,
        max_samples: 16384,
        target_level: 0.5,
        min_level: 0.25,
        max_level: 0.9,
        drop_on_overflow: true,
        pre_allocate: true,
    };

    // Basic validation that config can be created with correct fields
    assert_eq!(config.max_chunks, 32);
    assert_eq!(config.max_samples, 16384);
    assert!(config.min_level < config.target_level);
    assert!(config.target_level < config.max_level);
    assert!(config.drop_on_overflow);
    assert!(config.pre_allocate);
}

#[test]
fn test_adaptive_buffer_basic() {
    let base_config = BufferConfig::default();
    let mut buffer = AdaptiveBuffer::<f32>::new(base_config);

    // Test initial state - should be able to push/pop
    let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let chunk = AudioSamples::new_mono(data, 44100);

    let result = buffer.push(chunk);
    assert!(result.is_ok());

    let popped = buffer.pop();
    assert!(popped.is_some());
}
