use audio_samples::{AudioSamples, operations::AudioStatistics};
use ndarray::Array1;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Benchmarking FFT-based autocorrelation performance");
    
    // Test different signal sizes to show scaling behavior
    let test_sizes = vec![1024, 4096, 16384, 44100]; // Up to 1 second at 44.1kHz
    
    for &size in &test_sizes {
        // Generate test signal (sine wave with some noise)
        let data: Vec<f32> = (0..size)
            .map(|i| {
                let t = i as f32 / 44100.0;
                0.8 * (2.0 * std::f32::consts::PI * 440.0 * t).sin() + 
                0.2 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
            })
            .collect();
        
        let arr = Array1::from_vec(data);
        let audio = AudioSamples::new_mono(arr, 44100);
        
        // Test with reasonable max_lag (20% of signal length)
        let max_lag = size / 5;
        
        println!("\nTesting with {} samples, max_lag = {}", size, max_lag);
        
        // Warm up
        for _ in 0..3 {
            let _ = audio.autocorrelation(max_lag)?;
        }
        
        // Benchmark
        let iterations = if size <= 4096 { 100 } else { 20 }; // Fewer iterations for larger sizes
        let start = Instant::now();
        
        for _ in 0..iterations {
            let _autocorr = audio.autocorrelation(max_lag)?;
        }
        
        let duration = start.elapsed();
        let avg_time_ms = duration.as_millis() / iterations as u128;
        
        println!("Average autocorrelation time: {} ms", avg_time_ms);
        
        // Estimate what O(n²) would have taken
        let o_n_squared_ops = size * max_lag;
        let estimated_scalar_time_ms = o_n_squared_ops / 1_000_000; // Very rough estimate
        
        println!("Estimated O(n²) time: ~{} ms (rough estimate)", estimated_scalar_time_ms);
        
        if estimated_scalar_time_ms > 0 {
            let speedup = estimated_scalar_time_ms as f64 / avg_time_ms as f64;
            println!("Estimated speedup: {:.1}x", speedup);
        }
        
        // Calculate actual complexity for this size
        let fft_ops = size * (size as f64).log2() as usize;
        println!("FFT complexity O(n log n): {} operations vs O(n²): {} operations", 
                fft_ops, o_n_squared_ops);
    }
    
    Ok(())
}