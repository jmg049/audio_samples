//! Working demonstration of streaming audio playback.

use audio_samples::streaming_playback::StreamingPlayback;
use std::time::Duration;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎵 Audio Samples - Streaming Playback Integration Demo");
    println!("🎯 This demonstrates real-time streaming from generators to audio output");

    // Create a streaming playback instance
    let mut playback = StreamingPlayback::<f32>::new()?;
    println!("✅ Created streaming playback instance");

    // Test 1: Play a 440Hz sine wave
    println!("\n🎼 Test 1: Playing 440Hz sine wave for 3 seconds...");
    playback.play_sine_wave(440.0, 44100, 2).await?;

    // Let it run for a few seconds, yielding to allow tasks to execute
    for i in 1..=15 {
        // Give tasks time to run
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }

        if i % 5 == 0 {
            let state = playback.state();
            println!("  Status check {}/3: {:?}", i / 5, state);
        }

        std::thread::sleep(Duration::from_millis(200));
    }

    println!("⏹️  Stopping sine wave...");
    playback.stop().await?;

    // Test 2: Play white noise
    println!("\n🌊 Test 2: Playing white noise for 2 seconds...");
    playback.play_white_noise(44100, 2).await?;

    // Demonstrate volume control
    println!("🔊 Volume at 80%");
    std::thread::sleep(Duration::from_millis(1000));

    println!("🔉 Reducing volume to 30%...");
    playback.set_volume(0.3);
    std::thread::sleep(Duration::from_millis(1000));

    playback.stop().await?;

    // Test 3: Transport controls
    println!("\n⏯️  Test 3: Transport controls with sine wave...");
    playback.play_sine_wave(880.0, 44100, 1).await?; // A5 note, mono

    std::thread::sleep(Duration::from_millis(1000));

    println!("⏸️  Pausing...");
    playback.pause()?;
    std::thread::sleep(Duration::from_millis(500));

    println!("▶️  Resuming...");
    playback.resume()?;
    std::thread::sleep(Duration::from_millis(1000));

    println!("⏹️  Final stop...");
    playback.stop().await?;

    println!("\n✨ Integration Demo Complete!");
    println!("🎉 Successfully demonstrated:");
    println!("   • Real-time streaming from generators to audio output");
    println!("   • Multiple audio sources (sine waves, white noise)");
    println!("   • Volume control during playback");
    println!("   • Transport controls (play, pause, resume, stop)");
    println!("   • Proper lifecycle management");
    println!("");
    println!("🔗 The streaming and playback components are now fully integrated!");

    Ok(())
}
