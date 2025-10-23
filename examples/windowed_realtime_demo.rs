//! Demonstration of optimized windowed operations for real-time performance.
//!
//! This example showcases high-performance windowed audio processing techniques
//! designed for real-time applications like STFT analysis, spectral effects,
//! and overlapping window-based audio processing.

use audio_samples::{
    AudioSamples,
    realtime::{RealtimeAudioOps, RealtimeState, WindowType},
};
use ndarray::Array1;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Optimized Windowed Operations Demo");
    println!("=================================");
    println!("Showcasing real-time windowed processing for audio applications.\n");

    // Create test signals
    let sample_rate = 44100u32;
    let signal_length = sample_rate as usize; // 1 second
    let test_signal = generate_complex_signal(signal_length, sample_rate);

    println!(
        "Generated test signal: {} samples at {}Hz",
        signal_length, sample_rate
    );

    // Configuration for windowed processing
    let window_size = 512;
    let hop_size = 256; // 50% overlap
    let num_iterations = 100;

    println!(
        "Window configuration: {} samples, {} hop size ({}% overlap)",
        window_size,
        hop_size,
        (1.0 - hop_size as f32 / window_size as f32) * 100.0
    );

    // 1. Test different window functions
    println!("\n1. Window Function Performance:");
    test_window_functions(window_size);

    // 2. Pre-allocated vs. traditional windowed processing
    println!("\n2. Performance Comparison:");
    compare_windowed_processing(&test_signal, window_size, hop_size, num_iterations)?;

    // 3. Real-time overlapping window processing
    println!("\n3. Real-time Overlapping Processing:");
    test_realtime_overlapping(&test_signal, window_size, hop_size)?;

    // 4. Block-based processing simulation (audio callback style)
    println!("\n4. Audio Callback Simulation:");
    simulate_audio_callback(&test_signal, window_size, hop_size)?;

    // 5. Spectral processing simulation
    println!("\n5. Spectral Processing Simulation:");
    simulate_spectral_processing(&test_signal, window_size, hop_size)?;

    println!("\n✓ Windowed operations demo completed!");
    println!("\nKey Optimizations Demonstrated:");
    println!("  • Pre-allocated buffers for zero-allocation processing");
    println!("  • Efficient overlap-add algorithms");
    println!("  • Optimized window function generation");
    println!("  • Real-time suitable buffer management");
    println!("  • Minimal memory allocation during processing");

    Ok(())
}

/// Generate a complex test signal with multiple frequency components
fn generate_complex_signal(length: usize, sample_rate: u32) -> AudioSamples<f32> {
    let mut data = Vec::with_capacity(length);
    let sr = sample_rate as f32;

    for i in 0..length {
        let t = i as f32 / sr;

        // Create a complex signal with multiple components
        let fundamental = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.3;
        let harmonic2 = (2.0 * std::f32::consts::PI * 880.0 * t).sin() * 0.15;
        let harmonic3 = (2.0 * std::f32::consts::PI * 1320.0 * t).sin() * 0.1;

        // Add some modulation
        let modulation = (2.0 * std::f32::consts::PI * 5.0 * t).sin() * 0.1 + 1.0;
        let noise = (2.0 * std::f32::consts::PI * 3000.0 * t).sin() * 0.02;

        let sample = (fundamental + harmonic2 + harmonic3) * modulation + noise;
        data.push(sample);
    }

    AudioSamples::new_mono(Array1::from(data), sample_rate)
}

/// Test different window function generation performance
fn test_window_functions(window_size: usize) {
    let window_types = [
        ("Rectangular", WindowType::Rectangular),
        ("Hanning", WindowType::Hanning),
        ("Hamming", WindowType::Hamming),
        ("Blackman", WindowType::Blackman),
    ];

    for (name, window_type) in &window_types {
        let start = Instant::now();
        let _window = RealtimeState::<f32>::generate_window_function(window_size, *window_type);
        let duration = start.elapsed();

        println!("   {}: {:.2}μs generation time", name, duration.as_micros());
    }
}

/// Compare pre-allocated vs traditional windowed processing
fn compare_windowed_processing(
    signal: &AudioSamples<f32>,
    window_size: usize,
    hop_size: usize,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Test 1: Optimized pre-allocated processing
    let mut optimized_signal = signal.clone();
    let mut state = RealtimeState::new(1);
    state.setup_windowed_processing(window_size, hop_size);

    let start = Instant::now();
    for _ in 0..iterations {
        optimized_signal.realtime_overlapping_process(&mut state, |window| {
            // Simple gain processing
            for sample in window.iter_mut() {
                *sample *= 0.95;
            }
        })?;
    }
    let optimized_duration = start.elapsed();

    // Test 2: Traditional processing (for comparison)
    let mut traditional_signal = signal.clone();
    let start = Instant::now();
    for _ in 0..iterations {
        traditional_signal.apply_windowed(window_size, hop_size, |input, _output| {
            // Simple gain processing
            input.iter().map(|&x| x * 0.95).collect()
        })?;
    }
    let traditional_duration = start.elapsed();

    println!(
        "   Optimized (pre-allocated): {:.2}ms total, {:.2}μs/iteration",
        optimized_duration.as_millis(),
        optimized_duration.as_micros() as f64 / iterations as f64
    );

    println!(
        "   Traditional (allocating): {:.2}ms total, {:.2}μs/iteration",
        traditional_duration.as_millis(),
        traditional_duration.as_micros() as f64 / iterations as f64
    );

    let speedup = traditional_duration.as_nanos() as f64 / optimized_duration.as_nanos() as f64;
    if speedup > 1.0 {
        println!("   → Optimized version is {:.1}x faster!", speedup);
    }

    Ok(())
}

/// Test real-time overlapping window processing
fn test_realtime_overlapping(
    signal: &AudioSamples<f32>,
    window_size: usize,
    hop_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut test_signal = signal.clone();
    let mut state = RealtimeState::new(1);
    state.setup_windowed_processing(window_size, hop_size);

    // Simulate spectral gain processing
    let start = Instant::now();
    test_signal.realtime_overlapping_process(&mut state, |window| {
        // Apply frequency-dependent gain (simulate EQ)
        let window_len = window.len();
        for (i, sample) in window.iter_mut().enumerate() {
            let freq_factor = i as f32 / window_len as f32;
            let gain = 1.0 - 0.3 * freq_factor; // Reduce high frequencies
            *sample *= gain;
        }
    })?;
    let duration = start.elapsed();

    let samples_processed = test_signal.samples_per_channel();
    let throughput = samples_processed as f64 / duration.as_secs_f64() / 1_000_000.0;

    println!(
        "   Processed {} samples in {:.2}ms",
        samples_processed,
        duration.as_millis()
    );
    println!("   Throughput: {:.1} million samples/second", throughput);

    Ok(())
}

/// Simulate real-time audio callback processing
fn simulate_audio_callback(
    signal: &AudioSamples<f32>,
    window_size: usize,
    hop_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut state = RealtimeState::new(1);
    state.setup_windowed_processing(window_size, hop_size);

    let block_size = 64; // Typical audio interface block size
    let signal_data = signal.as_mono().unwrap();
    let mut total_latency = 0u128;
    let mut blocks_processed = 0;

    // Simulate processing audio blocks as they arrive
    let mut offset = 0;
    while offset + block_size <= signal_data.len() {
        let chunk = &signal_data.slice(ndarray::s![offset..offset + block_size]);
        let start = Instant::now();

        // Process this block (simulate real-time windowed effect)
        let chunk_slice = chunk.as_slice().unwrap();
        let mut dummy_audio = AudioSamples::new_mono(Array1::from(chunk_slice.to_vec()), 44100);
        let _output = dummy_audio.realtime_windowed_process(chunk_slice, &mut state, |window| {
            // Simulate a real-time effect (e.g., dynamic compression)
            window
                .iter()
                .map(|&x| {
                    if x.abs() > 0.5 {
                        x.signum() * (0.5 + 0.5 * (1.0 - (-2.0f32 * (x.abs() - 0.5)).exp()))
                    } else {
                        x
                    }
                })
                .collect()
        })?;

        let block_latency = start.elapsed();
        total_latency += block_latency.as_nanos();
        blocks_processed += 1;
        offset += block_size;
    }

    let avg_latency = total_latency as f64 / blocks_processed as f64 / 1000.0; // Convert to microseconds
    println!(
        "   Processed {} blocks of {} samples each",
        blocks_processed, block_size
    );
    println!(
        "   Average processing latency: {:.2}μs per block",
        avg_latency
    );
    println!(
        "   Real-time factor: {:.1}x (lower is better)",
        avg_latency / (block_size as f64 / 44100.0 * 1_000_000.0)
    );

    Ok(())
}

/// Simulate spectral processing with windowed operations
fn simulate_spectral_processing(
    signal: &AudioSamples<f32>,
    window_size: usize,
    hop_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut test_signal = signal.clone();
    let mut state = RealtimeState::new(1);
    state.setup_windowed_processing(window_size, hop_size);

    let start = Instant::now();

    // Simulate complex spectral processing
    test_signal.realtime_overlapping_process(&mut state, |window| {
        // Simulate spectral processing steps:

        // 1. Apply window function (already done by the framework)

        // 2. Simulate spectral analysis
        let mut spectral_data = vec![0.0f32; window.len()];
        for (i, &sample) in window.iter().enumerate() {
            // Simple frequency domain simulation (not real FFT for demo)
            spectral_data[i] = sample;
        }

        // 3. Spectral modification (e.g., noise reduction, EQ)
        let spectral_len = spectral_data.len();
        for (i, value) in spectral_data.iter_mut().enumerate() {
            let freq_ratio = i as f32 / spectral_len as f32;

            // Simulate frequency-selective processing
            if freq_ratio > 0.8 {
                // Reduce high frequency noise
                *value *= 0.3;
            } else if freq_ratio < 0.1 {
                // Reduce low frequency rumble
                *value *= 0.7;
            }
        }

        // 4. Copy back to time domain (overlap-add handles the rest)
        for (i, sample) in window.iter_mut().enumerate() {
            *sample = spectral_data[i];
        }
    })?;

    let duration = start.elapsed();
    let samples_processed = test_signal.samples_per_channel();

    println!(
        "   Spectral processing: {} samples in {:.2}ms",
        samples_processed,
        duration.as_millis()
    );
    println!(
        "   Processing rate: {:.1} million samples/second",
        samples_processed as f64 / duration.as_secs_f64() / 1_000_000.0
    );

    Ok(())
}
