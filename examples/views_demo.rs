//! Demonstration of zero-copy view types for efficient audio access.

use audio_samples::{AudioSamples, AudioStatistics, AudioView};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Views Demo - Zero-Copy Audio Access");

    // Create some sample audio data
    let data = array![0.1f32, 0.5, -0.3, 0.8, -0.9, 0.2, -0.4, 0.7];
    let audio = AudioSamples::new_mono(data, 44100);

    println!(
        "Original audio: {} samples at {} Hz",
        audio.samples_per_channel(),
        audio.sample_rate()
    );

    // Create a read-only view (zero-copy)
    let view: AudioView<f32> = audio.as_view();
    println!("View stats:");
    println!("  Channels: {}", view.num_channels());
    println!("  Duration: {:.3}s", view.duration_seconds());
    println!("  Peak: {:.3}", view.peak());

    // Views can be passed to functions without cloning the data
    analyze_audio(&view);

    // Demonstrate zero-copy type conversion for compatible layouts
    let i32_data = array![1065353216i32, 1073741824i32, 1077936128i32]; // bit patterns for 1.0f32, 2.0f32, 3.0f32
    let i32_audio = AudioSamples::new_mono(i32_data, 44100);

    println!("\nZero-copy type conversion demo:");
    println!(
        "Original i32 audio: {} samples",
        i32_audio.samples_per_channel()
    );

    // Try to create a zero-copy view as f32 (same 32-bit layout)
    if let Some(f32_view) = i32_audio.try_as_type_view::<f32>() {
        println!("Successfully created f32 view from i32 data");
        println!("f32 view peak: {:.3}", f32_view.peak());
        println!("This required zero memory allocation!");
    } else {
        println!("Could not create zero-copy view (incompatible layouts)");
    }

    // Demonstrate mutable views
    let mut mutable_audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0], 44100);
    println!("\nMutable view demo:");
    println!("Before: peak = {:.3}", mutable_audio.peak());

    {
        let mut_view = mutable_audio.as_view_mut();
        // Note: We would need to implement processing operations for views
        // to demonstrate in-place modifications here
        println!(
            "Created mutable view with {} samples",
            mut_view.samples_per_channel()
        );
    }

    println!("Views demo completed successfully!");
    Ok(())
}

/// Example function that takes a view for read-only analysis.
/// This demonstrates how views enable zero-copy parameter passing.
fn analyze_audio<T: audio_samples::AudioSample>(view: &AudioView<T>) {
    println!("Analyzing audio view:");
    println!("  Type: {}", std::any::type_name::<T>());
    println!("  Total samples: {}", view.total_samples());
    println!("  Layout: {:?}", view.layout());

    if view.is_mono() {
        println!("  Format: Mono");
    } else {
        println!("  Format: Multi-channel ({} channels)", view.num_channels());
    }
}
