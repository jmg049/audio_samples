use audio_samples::{AudioSamples, operations::AudioStatistics};
use ndarray::Array1;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a large audio buffer to test performance
    let sample_rate = 44100;
    let duration_seconds = 10;
    let num_samples = sample_rate * duration_seconds;
    
    // Generate test data (sine wave)
    let data: Vec<f32> = (0..num_samples)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin() * 0.5)
        .collect();
    
    let arr = Array1::from_vec(data);
    let audio = AudioSamples::new_mono(arr, sample_rate as u32);
    
    println!("Benchmarking RMS computation on {} samples ({} seconds of audio)", num_samples, duration_seconds);
    
    // Warm up
    for _ in 0..5 {
        let _ = audio.rms()?;
    }
    
    // Benchmark RMS
    let num_iterations = 1000;
    let start = Instant::now();
    
    for _ in 0..num_iterations {
        let _rms = audio.rms()?;
    }
    
    let duration = start.elapsed();
    let avg_time_us = duration.as_micros() / num_iterations as u128;
    
    println!("Average RMS computation time: {} microseconds", avg_time_us);
    println!("RMS throughput: {:.1} million samples/second", 
             (num_samples as f64 * num_iterations as f64) / duration.as_secs_f64() / 1_000_000.0);
    
    // Also test variance
    let start = Instant::now();
    
    for _ in 0..num_iterations {
        let _var = audio.variance()?;
    }
    
    let duration = start.elapsed();
    let avg_time_us = duration.as_micros() / num_iterations as u128;
    
    println!("Average variance computation time: {} microseconds", avg_time_us);
    println!("Variance throughput: {:.1} million samples/second", 
             (num_samples as f64 * num_iterations as f64) / duration.as_secs_f64() / 1_000_000.0);
    
    Ok(())
}