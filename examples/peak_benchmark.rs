use audio_samples::{AudioSamples, operations::AudioStatistics};
use ndarray::Array1;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Benchmarking vectorized peak/min/max operations");
    
    // Test different signal sizes
    let test_sizes = vec![44100, 441000, 4410000]; // 1s, 10s, 100s at 44.1kHz
    
    for &size in &test_sizes {
        // Generate test signal with full dynamic range
        let data: Vec<f32> = (0..size)
            .map(|i| {
                let t = i as f32 / 44100.0;
                // Mix of positive and negative values to test absolute value optimization
                0.8 * (2.0 * std::f32::consts::PI * 440.0 * t).sin() + 
                0.2 * (2.0 * std::f32::consts::PI * 1760.0 * t).cos() - 0.1
            })
            .collect();
        
        let arr = Array1::from_vec(data);
        let audio = AudioSamples::new_mono(arr, 44100);
        
        println!("\nTesting with {} samples ({:.1}s of audio)", size, size as f32 / 44100.0);
        
        // Warm up
        for _ in 0..10 {
            let _ = audio.peak();
            let _ = audio.min_native();
            let _ = audio.max_native();
        }
        
        // Benchmark peak finding
        let iterations = if size >= 1_000_000 { 100 } else { 1000 };
        
        let start = Instant::now();
        for _ in 0..iterations {
            let _peak = audio.peak();
        }
        let peak_time = start.elapsed();
        
        // Benchmark min finding
        let start = Instant::now();
        for _ in 0..iterations {
            let _min = audio.min_native();
        }
        let min_time = start.elapsed();
        
        // Benchmark max finding  
        let start = Instant::now();
        for _ in 0..iterations {
            let _max = audio.max_native();
        }
        let max_time = start.elapsed();
        
        let peak_avg_us = peak_time.as_micros() / iterations as u128;
        let min_avg_us = min_time.as_micros() / iterations as u128;
        let max_avg_us = max_time.as_micros() / iterations as u128;
        
        println!("Peak finding: {} µs/call", peak_avg_us);
        println!("Min finding:  {} µs/call", min_avg_us);
        println!("Max finding:  {} µs/call", max_avg_us);
        
        // Calculate throughput
        let peak_throughput = (size as f64 * iterations as f64) / peak_time.as_secs_f64() / 1_000_000.0;
        println!("Peak throughput: {:.0} million samples/second", peak_throughput);
        
        // Show improvement over naive iteration (estimated)
        let estimated_naive_ops = size * 3; // comparison + abs + max per sample
        let actual_vectorized_ops = size; // single vectorized pass
        let theoretical_speedup = estimated_naive_ops as f64 / actual_vectorized_ops as f64;
        
        println!("Theoretical improvement over scalar: {:.1}x", theoretical_speedup);
    }
    
    Ok(())
}