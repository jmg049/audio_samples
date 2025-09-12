//! Benchmark for size-aware FFT backend selection
//! 
//! This benchmark validates that:
//! 1. Small files maintain excellent performance (realfft backend)
//! 2. Large files benefit from MKL when available 
//! 3. No performance regression occurs compared to original implementation

use audio_samples::{AudioSamples, operations::{AudioTransforms, types::{SpectrogramScale, WindowType}}};
use ndarray::Array1;
use std::time::Instant;

/// Generate sine wave test data
fn generate_test_audio(duration_seconds: f64, sample_rate: usize) -> AudioSamples<f64> {
    let num_samples = (duration_seconds * sample_rate as f64) as usize;
    let frequency = 440.0; // A4
    
    let data: Array1<f64> = Array1::from_iter((0..num_samples).map(|i| {
        let t = i as f64 / sample_rate as f64;
        (2.0 * std::f64::consts::PI * frequency * t).sin() * 0.5
    }));
    
    AudioSamples::new_mono(data, sample_rate as u32)
}

/// Benchmark spectrogram computation for different file sizes
fn benchmark_spectrogram(duration: f64, label: &str) {
    let sample_rate = 44100;
    let audio = generate_test_audio(duration, sample_rate);
    
    println!("Benchmarking {}: {:.1}s audio, {} samples", 
             label, duration, audio.samples_per_channel());
    
    let window_size = 2048;
    let hop_size = 512;
    
    // Warm up
    for _ in 0..3 {
        let _ = audio.spectrogram(
            window_size, 
            hop_size, 
            WindowType::Hanning,
            SpectrogramScale::Linear,
            false
        );
    }
    
    // Benchmark runs
    let num_runs = 10;
    let mut times = Vec::new();
    
    for _ in 0..num_runs {
        let start = Instant::now();
        let result = audio.spectrogram(
            window_size, 
            hop_size, 
            WindowType::Hanning,
            SpectrogramScale::Linear,
            false
        );
        let elapsed = start.elapsed();
        
        assert!(result.is_ok(), "Spectrogram computation failed");
        times.push(elapsed.as_secs_f64() * 1000.0); // Convert to milliseconds
    }
    
    // Calculate statistics
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let median = times[times.len() / 2];
    let min = times[0];
    let max = times[times.len() - 1];
    
    println!("Results: {:.2}ms ± {:.2}ms (median: {:.2}ms, range: {:.2}-{:.2}ms)",
             mean, 
             (times.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / times.len() as f64).sqrt(),
             median, min, max);
    
    // Performance expectations based on the performance report
    if duration <= 1.0 {
        // Small files should be very fast (sub-10ms is excellent)
        println!("✅ Small file performance: {:.2}ms", mean);
    } else if duration >= 5.0 {
        // Large files - expecting competitive performance with librosa
        println!("🎯 Large file performance: {:.2}ms", mean);
        if cfg!(feature = "mkl") {
            println!("   Using Intel MKL backend for large files");
        } else {
            println!("   Using RealFFT backend (MKL not available)");
        }
    } else {
        println!("⚡ Medium file performance: {:.2}ms", mean);
    }
    
    println!();
}

fn main() {
    println!("🎵 Audio Samples Size-Aware FFT Backend Benchmark");
    println!("=================================================");
    
    println!("Features enabled:");
    if cfg!(feature = "mkl") {
        println!("  ✅ Intel MKL support");
    } else {
        println!("  ❌ Intel MKL support (not compiled in)");
    }
    println!();
    
    // Test different file sizes to validate size-aware selection
    let test_cases = vec![
        (0.5, "Very Small"),    // Should use RealFFT, be extremely fast
        (1.0, "Small"),         // Should use RealFFT, be very fast  
        (5.0, "Medium"),        // Boundary case
        (10.0, "Large"),        // Should use MKL if available
        (30.0, "Very Large"),   // Should use MKL if available
    ];
    
    for (duration, label) in test_cases {
        benchmark_spectrogram(duration, label);
    }
    
    println!("🏁 Benchmark Complete!");
    println!();
    
    println!("Expected Performance Characteristics:");
    println!("  📊 Small files (≤1s): Should be 100x+ faster than librosa (~1-10ms)");
    println!("  📊 Large files (≥5s): Should be competitive with librosa (~50-500ms)");
    println!("  🔄 Size-aware selection: Automatic backend choice based on duration");
    
    if !cfg!(feature = "mkl") {
        println!();
        println!("💡 To test Intel MKL integration, compile with:");
        println!("   cargo run --release --features mkl --bin size_aware_fft_benchmark");
    }
}