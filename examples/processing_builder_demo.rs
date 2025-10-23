//! Demonstration of the ProcessingBuilder fluent API for chaining audio operations.

use audio_samples::{AudioSamples, AudioStatistics, operations::types::NormalizationMethod};
use ndarray::array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ProcessingBuilder Demo - Fluent Audio Processing API");

    // Create some sample audio data
    let data = array![0.1f32, 0.5, -0.3, 0.8, -0.9, 0.2, -0.4, 0.7];
    let mut audio = AudioSamples::new_mono(data, 44100);

    println!("Original audio stats:");
    println!("  Peak: {:.3}", audio.peak());
    println!("  Min: {:.3}", audio.min_sample());
    println!("  Max: {:.3}", audio.max_sample());

    // Use the fluent builder API to chain multiple processing operations
    audio
        .processing()
        .normalize(-1.0, 1.0, NormalizationMethod::Peak) // Normalize to peak
        .scale(0.8) // Reduce volume to 80%
        .clip(-0.5, 0.5) // Apply soft clipping
        .remove_dc_offset() // Remove any DC bias
        .apply()?;

    println!("\nAfter processing chain:");
    println!("  Peak: {:.3}", audio.peak());
    println!("  Min: {:.3}", audio.min_sample());
    println!("  Max: {:.3}", audio.max_sample());

    // Demonstrate error handling in the builder
    let mut audio2 = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0], 44100);

    let result = audio2
        .processing()
        .normalize(1.0, -1.0, NormalizationMethod::Peak) // Invalid range - should fail
        .apply();

    match result {
        Ok(()) => println!("Unexpected success!"),
        Err(e) => println!("Expected error caught: {}", e),
    }

    println!("\nProcessingBuilder demo completed successfully!");
    Ok(())
}
