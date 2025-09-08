//! Demonstration of streaming audio playback capabilities.
//!
//! This example shows how to connect streaming audio sources to real-time playback,
//! bridging the gap between the streaming and playback modules.

use audio_samples::streaming::sources::generator::GeneratorSource;
use audio_samples::streaming_playback::StreamingPlayback;
use std::time::Duration;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎵 Audio Samples - Streaming Playback Demo");

    // Create a streaming playback instance
    let mut playback = StreamingPlayback::<f32>::new()?;

    println!("✅ Created streaming playback instance");

    // Demo 1: Play a sine wave
    println!("\n🎼 Demo 1: Playing 440Hz sine wave for 3 seconds...");
    playback.play_sine_wave(440.0, 44100, 2).await?;

    // Let it play for a few seconds
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Pause playback
    println!("⏸️  Pausing playback...");
    playback.pause()?;

    tokio::time::sleep(Duration::from_secs(1)).await;

    // Resume playback
    println!("▶️  Resuming playback...");
    playback.resume()?;

    tokio::time::sleep(Duration::from_secs(2)).await;

    // Stop the sine wave
    println!("⏹️  Stopping sine wave...");
    playback.stop().await?;

    // Demo 2: Play white noise
    println!("\n🌊 Demo 2: Playing white noise for 2 seconds...");
    playback.play_white_noise(44100, 2).await?;

    // Demonstrate volume control
    tokio::time::sleep(Duration::from_secs(1)).await;
    println!("🔉 Reducing volume to 30%...");
    playback.set_volume(0.3);

    tokio::time::sleep(Duration::from_secs(1)).await;
    println!("🔊 Increasing volume to 80%...");
    playback.set_volume(0.8);

    tokio::time::sleep(Duration::from_secs(1)).await;
    playback.stop().await?;

    // Demo 3: Custom generator source
    println!("\n Demo 3: Playing custom generator source...");
    let generator = GeneratorSource::<f32>::chirp(
        220.0,                  // Start frequency: 220Hz (A3)
        880.0,                  // End frequency: 880Hz (A5)
        Duration::from_secs(5), // Duration: 5 seconds
        44100,                  // Sample rate
        1,                      // Mono
    );

    playback.play_source(generator).await?;

    // Monitor buffer level while playing
    for i in 0..50 {
        let buffer_level = playback.buffer_level();
        let state = playback.state();
        println!(
            "Buffer level: {:.1}% | State: {:?}",
            buffer_level * 100.0,
            state
        );

        if !playback.is_playing() {
            break;
        }

        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    playback.stop().await?;

    println!("\n✨ Demo completed successfully!");
    println!(
        "📊 Streaming playback integration allows real-time audio streaming to output devices."
    );
    println!("🔗 This bridges streaming sources (generators, network, files) with audio hardware.");

    Ok(())
}
