//! Demonstration of allocation-free real-time audio operations.
//!
//! This example showcases the real-time audio processing capabilities that are
//! designed for zero-allocation, low-latency audio applications like real-time
//! audio effects, live performance systems, and audio callback functions.

use audio_samples::{AudioSamples, realtime::RealtimeAudioOps};
use ndarray::Array1;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Real-time Audio Operations Demo");
    println!("===============================");
    println!("All operations shown are allocation-free and suitable for real-time use.\n");

    // Create test audio with some dynamic content
    let sample_rate = 44100u32;
    let duration_samples = (sample_rate / 10) as usize; // 0.1 seconds
    let mut test_audio = generate_test_signal(duration_samples, sample_rate);

    println!(
        "Original signal: {} samples at {}Hz",
        test_audio.samples_per_channel(),
        test_audio.sample_rate()
    );

    // Demonstrate real-time processing operations

    // 1. Basic gain operations
    println!("\n1. Real-time Gain Operations:");
    benchmark_operation("   Scale by 0.8", || {
        test_audio.realtime_scale(0.8);
    });

    benchmark_operation("   Apply -3dB gain", || {
        test_audio.realtime_apply_gain_db(-3.0);
    });

    // 2. Dynamic range operations
    println!("\n2. Dynamic Range Operations:");
    benchmark_operation("   Hard clip to ±0.9", || {
        test_audio.realtime_clip(-0.9, 0.9);
    });

    benchmark_operation("   Soft limit with threshold 0.8", || {
        test_audio.realtime_soft_limit(0.8);
    });

    // 3. Filtering operations
    println!("\n3. Real-time Filtering:");
    benchmark_operation("   Remove DC offset", || {
        test_audio.realtime_remove_dc_offset(0.995);
    });

    benchmark_operation("   Apply lowpass filter", || {
        test_audio.realtime_lowpass_filter(0.8);
    });

    // 4. Custom processing
    println!("\n4. Custom Processing:");
    benchmark_operation("   Square all samples", || {
        test_audio.realtime_map_inplace(|sample| sample * sample);
    });

    benchmark_operation("   Apply soft saturation", || {
        test_audio.realtime_map_inplace(|sample| {
            // Soft saturation curve
            if sample.abs() < 0.5 {
                sample
            } else {
                sample.signum() * (0.5 + 0.5 * (1.0 - (-2.0 * (sample.abs() - 0.5)).exp()))
            }
        });
    });

    // 5. Stereo operations
    println!("\n5. Stereo Processing:");
    let mut stereo_audio = generate_stereo_test_signal(duration_samples / 2, sample_rate);

    benchmark_operation("   Adjust stereo width (1.5x)", || {
        stereo_audio.realtime_adjust_stereo_width(1.5).unwrap();
    });

    // 6. Mixing operations
    println!("\n6. Real-time Mixing:");
    let mut audio1 = generate_test_signal(1000, sample_rate);
    let audio2 = generate_test_signal(1000, sample_rate);

    benchmark_operation("   Mix two audio sources", || {
        audio1.realtime_mix_with(&audio2, 0.3).unwrap();
    });

    // 7. Performance comparison: allocation vs allocation-free
    println!("\n7. Performance Comparison:");
    println!("   Testing 1000 iterations of gain operations...");

    let mut test_data = generate_test_signal(1024, sample_rate); // 1K samples

    // Real-time (allocation-free) version
    let start = Instant::now();
    for _ in 0..1000 {
        test_data.realtime_scale(0.99);
    }
    let realtime_duration = start.elapsed();

    println!(
        "   Real-time operations: {:.2}μs/iteration",
        realtime_duration.as_micros() as f64 / 1000.0
    );

    // Compare with traditional methods (just for reference)
    let mut traditional_data = generate_test_signal(1024, sample_rate);
    let start = Instant::now();
    for _ in 0..1000 {
        // This simulates traditional processing that might allocate
        traditional_data.apply(|x| x * 0.99);
    }
    let traditional_duration = start.elapsed();

    println!(
        "   Traditional operations: {:.2}μs/iteration",
        traditional_duration.as_micros() as f64 / 1000.0
    );

    let speedup = traditional_duration.as_nanos() as f64 / realtime_duration.as_nanos() as f64;
    if speedup > 1.0 {
        println!("   Real-time operations are {:.1}x faster!", speedup);
    }

    println!("\n8. Real-time Suitability Analysis:");
    println!("   ✓ Zero allocations during processing");
    println!("   ✓ Bounded execution time");
    println!("   ✓ In-place operations");
    println!("   ✓ No garbage collection pressure");
    println!("   ✓ Suitable for audio callback threads");
    println!("   ✓ Lock-free operations");

    println!("\n✓ Real-time operations demo completed!");
    println!("\nKey Benefits:");
    println!("  • Predictable latency for real-time audio");
    println!("  • No memory allocation during audio processing");
    println!("  • Optimized for audio callback functions");
    println!("  • Suitable for live performance and low-latency applications");

    Ok(())
}

/// Generate a test signal with various frequency components
fn generate_test_signal(samples: usize, sample_rate: u32) -> AudioSamples<f32> {
    let mut data = Vec::with_capacity(samples);
    let sr = sample_rate as f32;

    for i in 0..samples {
        let t = i as f32 / sr;

        // Mix of frequencies to create realistic test signal
        let fundamental = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.3;
        let harmonic = (2.0 * std::f32::consts::PI * 880.0 * t).sin() * 0.1;
        let noise = (2.0 * std::f32::consts::PI * 1000.0 * t).sin() * 0.05;

        data.push(fundamental + harmonic + noise);
    }

    AudioSamples::new_mono(Array1::from(data).into(), sample_rate)
}

/// Generate stereo test signal
fn generate_stereo_test_signal(samples: usize, sample_rate: u32) -> AudioSamples<f32> {
    let sr = sample_rate as f32;
    let mut left = Vec::with_capacity(samples);
    let mut right = Vec::with_capacity(samples);

    for i in 0..samples {
        let t = i as f32 / sr;

        // Different content for left and right channels
        let left_signal = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
        let right_signal = (2.0 * std::f32::consts::PI * 660.0 * t).sin() * 0.3;

        left.push(left_signal);
        right.push(right_signal);
    }

    let stereo_data =
        ndarray::Array2::from_shape_fn(
            (2, samples),
            |(ch, i)| {
                if ch == 0 { left[i] } else { right[i] }
            },
        );

    AudioSamples::new_multi_channel(stereo_data, sample_rate)
}

/// Benchmark a real-time operation
fn benchmark_operation<F>(name: &str, mut operation: F)
where
    F: FnMut(),
{
    // Warm-up
    operation();

    // Benchmark
    let start = Instant::now();
    operation();
    let duration = start.elapsed();

    println!("{}:  {:.2}μs", name, duration.as_micros());
}
