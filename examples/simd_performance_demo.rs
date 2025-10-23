//! Performance demonstration of SIMD-optimized audio conversions.

use audio_samples::{AudioSamples, AudioTypeConversion};
use ndarray::Array1;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SIMD Performance Demo");
    println!("=====================");

    // Create a large audio buffer for benchmarking
    let sample_count = 1_000_000; // 1M samples
    println!("Testing with {} samples", sample_count);

    // Generate test data: sine wave
    let mut test_data = Vec::with_capacity(sample_count);
    for i in 0..sample_count {
        let sample = (i as f32 * 0.001).sin() * 0.8; // 0.8 amplitude sine wave
        test_data.push(sample);
    }

    let audio_f32 = AudioSamples::new_mono(Array1::from(test_data).into(), 44100);

    println!("\nTesting f32 to i16 conversion performance:");

    // Warm-up run
    let _ = audio_f32.as_type::<i16>()?;

    // Benchmark the conversion
    let start = Instant::now();
    let _result = audio_f32.as_type::<i16>()?;
    let duration = start.elapsed();

    println!(
        "Conversion completed in: {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );
    println!(
        "Throughput: {:.2} million samples/second",
        sample_count as f64 / duration.as_secs_f64() / 1_000_000.0
    );

    // Also test the reverse conversion
    let audio_i16 = audio_f32.as_type::<i16>()?;

    println!("\nTesting i16 to f32 conversion performance:");

    let start = Instant::now();
    let _result = audio_i16.as_type::<f32>()?;
    let duration = start.elapsed();

    println!(
        "Conversion completed in: {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );
    println!(
        "Throughput: {:.2} million samples/second",
        sample_count as f64 / duration.as_secs_f64() / 1_000_000.0
    );

    // Test multi-channel performance
    println!("\nTesting multi-channel performance (stereo):");
    let stereo_data = ndarray::Array2::from_shape_fn((2, sample_count / 2), |(ch, i)| {
        let sample = (i as f32 * 0.001).sin() * 0.8;
        if ch == 0 { sample } else { sample * 0.5 } // Different amplitude per channel
    });

    let stereo_audio = AudioSamples::new_multi_channel(stereo_data.into(), 44100);

    let start = Instant::now();
    let _result = stereo_audio.as_type::<i16>()?;
    let duration = start.elapsed();

    println!(
        "Stereo conversion completed in: {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );
    println!(
        "Throughput: {:.2} million samples/second",
        sample_count as f64 / duration.as_secs_f64() / 1_000_000.0
    );

    #[cfg(feature = "simd")]
    println!(
        "\n✓ SIMD optimizations enabled - you should see improved performance!\nIf not make sure you are also compiling with the appropriate RUSTFLAGS for your CPU architecture."
    );

    #[cfg(not(feature = "simd"))]
    println!(
        "\n⚠ SIMD optimizations disabled - rebuild with --features simd for better performance"
    );

    println!("\nPerformance demo completed!");
    Ok(())
}
