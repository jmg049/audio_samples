//! Tests for signal generator streaming sources.

use super::super::sources::generator::*;
use super::super::traits::*;
use super::*;
use std::time::Duration;

#[tokio::test]
async fn test_sine_wave_generator_basic() {
    let mut generator = GeneratorSource::<f32>::sine(440.0, 48000, 2);

    // Test initial state
    assert!(generator.is_active());

    // Generate some samples
    let chunk = generator.next_chunk().await.unwrap();
    assert!(chunk.is_some());

    let audio_data = chunk.unwrap();
    validate_audio_format(&audio_data, 1024, 2, 48000); // Default chunk size
}

#[tokio::test]
async fn test_white_noise_generator_basic() {
    let mut generator = GeneratorSource::<f32>::white_noise(44100, 1);

    assert!(generator.is_active());

    let chunk = generator.next_chunk().await.unwrap();
    assert!(chunk.is_some());

    let audio_data = chunk.unwrap();
    validate_audio_format(&audio_data, 1024, 1, 44100);
}

#[tokio::test]
async fn test_silence_generator_basic() {
    let mut generator = GeneratorSource::<f32>::silence(44100, 2);

    let chunk = generator.next_chunk().await.unwrap();
    assert!(chunk.is_some());

    let audio_data = chunk.unwrap();
    validate_audio_format(&audio_data, 1024, 2, 44100);
}

#[tokio::test]
async fn test_generator_with_duration() {
    let config = GeneratorConfig {
        signal_type: SignalType::Sine { frequency: 1000.0 },
        amplitude: 0.5,
        sample_rate: 48000,
        channels: 1,
        chunk_size: 1024,
        duration: Some(Duration::from_millis(100)), // Very short duration
    };

    let mut generator = GeneratorSource::<f32>::new(config);

    let mut chunk_count = 0;

    loop {
        match generator.next_chunk().await.unwrap() {
            Some(_audio) => {
                chunk_count += 1;
                assert!(
                    chunk_count < 10,
                    "Generator should stop after short duration"
                );
            }
            None => break,
        }
    }

    // Should have generated at least one chunk before stopping
    assert!(chunk_count > 0);
    assert!(
        !generator.is_active(),
        "Generator should be inactive after duration"
    );
}

#[test]
fn test_generator_config_defaults() {
    let config = GeneratorConfig::default();

    assert!(matches!(
        config.signal_type,
        SignalType::Sine { frequency: 440.0 }
    ));
    assert_eq!(config.amplitude, 0.5);
    assert_eq!(config.sample_rate, 44100);
    assert_eq!(config.channels, 2);
    assert_eq!(config.chunk_size, 1024);
    assert_eq!(config.duration, None);
}

#[tokio::test]
async fn test_different_signal_types() {
    // Test that different signal types can be created without errors
    let sine = GeneratorSource::<f32>::sine(440.0, 44100, 1);
    assert!(sine.is_active());

    let noise = GeneratorSource::<f32>::white_noise(44100, 1);
    assert!(noise.is_active());

    let silence = GeneratorSource::<f32>::silence(44100, 1);
    assert!(silence.is_active());
}

#[tokio::test]
async fn test_generator_format_info() {
    let generator = GeneratorSource::<f32>::sine(440.0, 48000, 2);
    let format_info = generator.format_info();

    assert_eq!(format_info.sample_rate, 48000);
    assert_eq!(format_info.channels, 2);
    // Just verify format info can be retrieved without error
}
