//! Basic Audio Processing Example
//!
//! This example demonstrates core audio processing operations including:
//! - Statistical analysis (RMS, peak, variance)
//! - Normalization and dynamic range control
//! - Basic signal processing operations
//! - Multi-channel audio handling
//!
//! Run with: `cargo run --example basic_audio_processing`

use audio_samples::{
    operations::*,
    AudioSamples, 
    AudioSampleResult,
};
use ndarray::Array1;

fn main() -> AudioSampleResult<()> {
    println!("🎵 Basic Audio Processing Example");
    println!("=================================\n");

    // Create sample audio data - a simple sine-like wave
    let sample_rate = 44100;
    let duration_samples = 1000;
    
    // Generate test data: mix of sine-like pattern with some noise
    let mut audio_data = Vec::with_capacity(duration_samples);
    for i in 0..duration_samples {
        let t = i as f32 / sample_rate as f32;
        let sine_wave = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        let noise = (i % 17) as f32 / 100.0 - 0.08; // Small amount of "noise"
        audio_data.push(sine_wave * 0.7 + noise);
    }
    
    let audio = AudioSamples::new_mono(
        Array1::from(audio_data), 
        sample_rate as u32
    );

    println!("📊 Audio Information:");
    println!("   Sample rate: {} Hz", audio.sample_rate());
    println!("   Channels: {}", audio.num_channels());
    println!("   Samples per channel: {}", audio.samples_per_channel());
    println!("   Duration: {:.3} seconds", audio.duration_seconds());
    println!();

    // Statistical Analysis
    println!("📈 Statistical Analysis:");
    
    // Peak analysis
    let peak = audio.peak();
    println!("   Peak amplitude: {:.6}", peak);
    
    // RMS (Root Mean Square) - indicates the "energy" of the signal
    let rms = audio.rms()?;
    println!("   RMS level: {:.6}", rms);
    
    // Variance and standard deviation
    let variance = audio.variance()?;
    let std_dev = audio.std_dev()?;
    println!("   Variance: {:.6}", variance);
    println!("   Standard deviation: {:.6}", std_dev);
    
    // Zero crossing analysis - useful for pitch detection
    let zero_crossings = audio.zero_crossings();
    let zcr = audio.zero_crossing_rate();
    println!("   Zero crossings: {}", zero_crossings);
    println!("   Zero crossing rate: {:.2} Hz", zcr);
    println!();

    // Signal Processing Operations
    println!("🔧 Signal Processing:");
    
    // Normalization - scale audio to use full dynamic range
    let mut audio_copy = audio.clone();
    audio_copy.normalize(-1.0, 1.0, NormalizationMethod::MinMax)?;
    
    let normalized_peak = audio_copy.peak();
    let normalized_rms = audio_copy.rms()?;
    println!("   After normalization:");
    println!("     Peak: {:.6}", normalized_peak);
    println!("     RMS: {:.6}", normalized_rms);
    println!();

    // Multi-channel Audio Example
    println!("🎛️ Multi-channel Audio Example:");
    
    // Create stereo audio with different content in each channel
    let samples_per_channel = 500;
    let mut stereo_data = ndarray::Array2::<f32>::zeros((2, samples_per_channel));
    
    // Left channel: 440 Hz sine wave
    for i in 0..samples_per_channel {
        let t = i as f32 / sample_rate as f32;
        stereo_data[[0, i]] = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
    }
    
    // Right channel: 880 Hz sine wave (one octave higher)
    for i in 0..samples_per_channel {
        let t = i as f32 / sample_rate as f32;
        stereo_data[[1, i]] = (2.0 * std::f32::consts::PI * 880.0 * t).sin() * 0.3;
    }
    
    let stereo_audio = AudioSamples::new_multi_channel(stereo_data, sample_rate as u32);
    
    println!("   Stereo audio created:");
    println!("     Channels: {}", stereo_audio.num_channels());
    println!("     Samples per channel: {}", stereo_audio.samples_per_channel());
    
    let stereo_peak = stereo_audio.peak();
    let stereo_rms = stereo_audio.rms()?;
    println!("     Combined peak: {:.6}", stereo_peak);
    println!("     Combined RMS: {:.6}", stereo_rms);
    println!();

    // Demonstrate type conversion
    println!("🔄 Type Conversion Example:");
    println!("   Original audio type: f32");
    
    // Convert to different sample types
    let audio_i16 = audio.as_type::<i16>()?;
    let audio_i32 = audio.as_type::<i32>()?;
    
    println!("   Converted to i16 - Peak: {}", audio_i16.peak());
    println!("   Converted to i32 - Peak: {}", audio_i32.peak());
    
    // Show that statistical relationships are preserved
    let i16_rms = audio_i16.rms()?;
    let i32_rms = audio_i32.rms()?;
    println!("   i16 RMS: {:.6}", i16_rms);
    println!("   i32 RMS: {:.6}", i32_rms);
    println!("   (Note: RMS values differ due to different scaling but proportions are preserved)");
    println!();

    // Advanced Statistical Analysis
    println!("🧮 Advanced Analysis:");
    
    // Autocorrelation for pitch detection (first few lags)
    let autocorr = audio.autocorrelation(50)?;
    println!("   Autocorrelation computed for {} lags", autocorr.len());
    
    // Find the highest correlation after lag 0 (which is always 1.0)
    if autocorr.len() > 1 {
        let max_corr_idx = (1..autocorr.len().min(20))
            .max_by(|&a, &b| autocorr[a].partial_cmp(&autocorr[b]).unwrap())
            .unwrap();
        
        println!("   Highest correlation at lag {}: {:.4}", max_corr_idx, autocorr[max_corr_idx]);
        
        // Estimate fundamental frequency from autocorrelation
        let estimated_freq = sample_rate as f32 / max_corr_idx as f32;
        println!("   Estimated fundamental frequency: {:.1} Hz", estimated_freq);
    }
    println!();

    println!("✅ Audio processing complete!");
    println!("\nThis example demonstrated:");
    println!("  • Statistical analysis of audio signals");
    println!("  • Normalization and dynamic range control");
    println!("  • Multi-channel audio handling");
    println!("  • Type-safe sample format conversions");
    println!("  • Advanced analysis techniques like autocorrelation");
    
    Ok(())
}