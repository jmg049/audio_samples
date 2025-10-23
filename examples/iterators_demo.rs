use audio_samples::{AudioSampleIterators, AudioSamples, PaddingMode};
use ndarray::array;

fn main() {
    // Create some sample audio data
    let stereo_audio = AudioSamples::new_multi_channel(
        array![
            [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],    // Left channel
            [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0]  // Right channel
        ],
        44100,
    );

    println!(
        "Audio: {} channels, {} samples per channel",
        stereo_audio.num_channels(),
        stereo_audio.samples_per_channel()
    );

    // Frame iterator - iterate over frames (one sample from each channel)
    println!("\n--- Frame Iterator ---");
    for (i, frame) in stereo_audio.frames().enumerate() {
        println!("Frame {}: {:?}", i, frame);
    }

    // Channel iterator - iterate over complete channels
    println!("\n--- Channel Iterator ---");
    for (i, channel) in stereo_audio.channels().enumerate() {
        println!("Channel {}: {:?}", i, channel);
    }

    // Window iterator with no overlap
    println!("\n--- Window Iterator (no overlap) ---");
    for (i, window) in stereo_audio.windows(4, 4).enumerate() {
        println!("Window {}: {:?}", i, window);
    }

    // Window iterator with overlap
    println!("\n--- Window Iterator (50% overlap) ---");
    for (i, window) in stereo_audio.windows(4, 2).enumerate() {
        println!("Window {}: {:?}", i, window);
    }

    // Window iterator with different padding modes
    let mono_audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0], 44100);

    println!("\n--- Padding Modes (5 samples, window=3, hop=2) ---");

    println!("Zero padding:");
    for (i, window) in mono_audio
        .windows(3, 2)
        .with_padding_mode(PaddingMode::Zero)
        .enumerate()
    {
        println!("  Window {}: {:?}", i, window);
    }

    println!("No padding:");
    for (i, window) in mono_audio
        .windows(3, 2)
        .with_padding_mode(PaddingMode::None)
        .enumerate()
    {
        println!("  Window {}: {:?}", i, window);
    }

    println!("Skip incomplete:");
    for (i, window) in mono_audio
        .windows(3, 2)
        .with_padding_mode(PaddingMode::Skip)
        .enumerate()
    {
        println!("  Window {}: {:?}", i, window);
    }

    // Demonstrate multiple iterators from the same audio
    println!("\n--- Multiple Iterators ---");
    let frames = mono_audio.frames();
    let channels = mono_audio.channels();
    let windows = mono_audio.windows(2, 1);

    println!("Frame count: {}", frames.len());
    println!("Channel count: {}", channels.len());
    println!("Window count: {}", windows.len());
}
