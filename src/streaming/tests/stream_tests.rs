//! Tests for core streaming functionality.

use super::super::sources::generator::GeneratorSource;
use super::super::*;
use std::time::Duration;

#[tokio::test]
async fn test_stream_config_basic() {
    let config = StreamConfig::default();

    // Test that default config can be created
    assert!(config.buffer_size > 0);
}

#[test]
fn test_stream_config_creation() {
    let config = StreamConfig {
        buffer_size: 2048,
        max_buffer_size: 8192,
        min_buffer_level: 0.25,
        read_timeout: Duration::from_millis(100),
        auto_recovery: true,
        max_recovery_attempts: 5,
        preferred_format: None,
    };

    assert_eq!(config.buffer_size, 2048);
    assert_eq!(config.max_buffer_size, 8192);
    assert_eq!(config.min_buffer_level, 0.25);
    assert!(config.auto_recovery);
    assert_eq!(config.max_recovery_attempts, 5);
}

#[tokio::test]
async fn test_generator_integration() {
    // Test that generator can be created and produce data
    let mut generator = GeneratorSource::<f32>::sine(440.0, 48000, 2);

    assert!(generator.is_active());

    let chunk = generator.next_chunk().await.unwrap();
    assert!(chunk.is_some());

    let audio = chunk.unwrap();
    assert_eq!(audio.num_channels(), 2);
    assert_eq!(audio.sample_rate(), 48000);
}

#[tokio::test]
async fn test_multiple_generator_types() {
    // Test different generator types
    let generators = vec![
        GeneratorSource::<f32>::sine(440.0, 44100, 1),
        GeneratorSource::<f32>::white_noise(44100, 1),
        GeneratorSource::<f32>::silence(44100, 1),
    ];

    for mut generator in generators {
        assert!(generator.is_active());
        let chunk = generator.next_chunk().await.unwrap();
        assert!(chunk.is_some());
    }
}
