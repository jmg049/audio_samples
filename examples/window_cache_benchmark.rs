use audio_samples::{AudioSamples, operations::AudioTransforms, operations::types::WindowType};
use ndarray::Array1;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Benchmarking window function caching performance");
    
    // Test different window types and sizes
    let window_types = vec![
        WindowType::Hanning,
        WindowType::Hamming,
        WindowType::Blackman,
        WindowType::Kaiser { beta: 8.6 },
        WindowType::Gaussian { std: 0.4 },
    ];
    
    let window_sizes = vec![512, 1024, 2048, 4096];
    let hop_sizes = vec![256, 512, 1024, 2048];
    
    for window_type in &window_types {
        for (&window_size, &hop_size) in window_sizes.iter().zip(hop_sizes.iter()) {
            println!("\nTesting {:?} window, size={}, hop={}", window_type, window_size, hop_size);
            
            // Create test signal - enough for multiple STFT calls
            let samples_per_channel = 44100 * 2; // 2 seconds
            let data: Vec<f32> = (0..samples_per_channel)
                .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
                .collect();
            
            let arr = Array1::from_vec(data);
            let audio = AudioSamples::new_mono(arr, 44100);
            
            // First call - will compute and cache the window
            let first_call_time = {
                let start = Instant::now();
                let _ = audio.stft(window_size, hop_size, *window_type)?;
                start.elapsed()
            };
            
            // Second call - should use cached window
            let cached_call_time = {
                let start = Instant::now();
                let _ = audio.stft(window_size, hop_size, *window_type)?;
                start.elapsed()
            };
            
            // Multiple cached calls to get average
            let num_cached_calls = 10;
            let multi_cached_time = {
                let start = Instant::now();
                for _ in 0..num_cached_calls {
                    let _ = audio.stft(window_size, hop_size, *window_type)?;
                }
                start.elapsed()
            };
            
            let avg_cached_time = multi_cached_time / num_cached_calls as u32;
            
            println!("First call (compute + cache): {:8.2} µs", first_call_time.as_micros());
            println!("Second call (cached):         {:8.2} µs", cached_call_time.as_micros());
            println!("Average cached call:          {:8.2} µs", avg_cached_time.as_micros());
            
            if cached_call_time < first_call_time {
                let speedup = first_call_time.as_nanos() as f64 / cached_call_time.as_nanos() as f64;
                println!("Cache speedup:                {:.2}x", speedup);
            }
            
            // Show window generation overhead saved
            let num_frames = if samples_per_channel >= window_size {
                (samples_per_channel - window_size) / hop_size + 1
            } else {
                0
            };
            
            println!("Frames per STFT:              {}", num_frames);
            println!("Window reuses per STFT:       {}", num_frames);
            
            // Theoretical benefit: window computation is done once vs once per frame (without caching)
            // In practice, the window is reused across frames in the current implementation,
            // but caching benefits multiple STFT calls with same parameters
        }
    }
    
    println!("\n=== Cache Statistics ===");
    println!("Window functions are now cached globally for:");
    println!("- Same window type and size combinations");
    println!("- Eliminates repeated trigonometric calculations");
    println!("- Particularly beneficial for repeated STFT operations");
    println!("- Cache limited to 100 entries to prevent unbounded growth");
    
    Ok(())
}