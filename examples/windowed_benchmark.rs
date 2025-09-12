use audio_samples::AudioSamples;
use ndarray::Array1;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Benchmarking windowed operations: legacy vs optimized");
    
    // Create test signal
    let sample_rate = 44100;
    let duration_seconds = 5;
    let num_samples = sample_rate * duration_seconds;
    
    let data: Vec<f32> = (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5)
        .collect();
    
    let window_size = 1024;
    let hop_size = 512;
    
    println!("Testing with {} samples, window_size={}, hop_size={}", 
             num_samples, window_size, hop_size);
    
    // Test legacy version (allocates per window)
    let legacy_time = {
        let arr = Array1::from_vec(data.clone());
        let mut audio = AudioSamples::new_mono(arr, sample_rate as u32);
        
        let start = Instant::now();
        
        // Simple gain operation with legacy API
        audio.apply_windowed(window_size, hop_size, |window| {
            window.iter().map(|&x| x * 0.8).collect()
        })?;
        
        let duration = start.elapsed();
        println!("Legacy windowed operation: {:?}", duration);
        duration
    };
    
    // Test optimized version (pre-allocated buffers)
    let optimized_time = {
        let arr = Array1::from_vec(data.clone());
        let mut audio = AudioSamples::new_mono(arr, sample_rate as u32);
        
        let start = Instant::now();
        
        // Same gain operation with optimized API
        audio.apply_windowed_inplace(window_size, hop_size, |input, output| {
            for (i, &sample) in input.iter().enumerate() {
                output[i] = sample * 0.8;
            }
        })?;
        
        let duration = start.elapsed();
        println!("Optimized windowed operation: {:?}", duration);
        duration
    };
    
    if legacy_time > optimized_time {
        let speedup = legacy_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
        println!("Speedup: {:.1}x faster", speedup);
    }
    
    // Test memory allocation behavior
    println!("\nMemory allocation analysis:");
    let num_windows = (num_samples - window_size) / hop_size + 1;
    println!("Number of windows: {}", num_windows);
    println!("Legacy version allocates: {} vectors of {} samples each", 
             num_windows, window_size);
    println!("Optimized version allocates: 3 vectors total (result, overlap_count, window_buffer)");
    println!("Memory allocation reduction: ~{}x", num_windows / 3);
    
    Ok(())
}