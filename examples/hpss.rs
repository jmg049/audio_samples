//! Harmonic/Percussive Source Separation (HPSS) example.
//!
//! This example demonstrates how to use HPSS to separate audio signals into
//! harmonic and percussive components. It creates a synthetic signal combining
//! a sine wave (harmonic) and impulse train (percussive), then separates them.

#[cfg(feature = "hpss")]
use std::time::Duration;

#[cfg(feature = "hpss")]
use audio_samples::{
    AudioEditing, AudioSampleResult, AudioSamples, AudioStatistics,
    operations::{AudioDecomposition, HpssConfig},
    utils::generation::{impulse, sine_wave},
};

#[cfg(not(feature = "hpss"))]
fn main() {
    eprintln!("This example requires the 'hpss' feature.");
    eprintln!("Run with: cargo run --example hpss --features hpss");
}

#[cfg(feature = "hpss")]
fn main() -> AudioSampleResult<()> {
    println!("🎵 Harmonic/Percussive Source Separation (HPSS) Example");
    println!("=========================================");

    // Audio parameters
    let sample_rate = 44100;
    let duration = Duration::from_secs(3);
    let samples_per_channel = duration.as_secs_f64() * sample_rate as f64;

    println!("\n📊 Signal Parameters:");
    println!("  • Sample rate: {} Hz", sample_rate);
    println!("  • Duration: {} seconds", duration.as_secs_f64());
    println!("  • Samples: {}", samples_per_channel as usize);

    // Create harmonic component (sine wave)
    println!("\n🎼 Creating harmonic component (sine wave)...");
    let harmonic_freq = 440.0; // A4
    let harmonic_amplitude = 0.6;
    let harmonic_signal =
        sine_wave::<f32, f32>(harmonic_freq, duration, sample_rate, harmonic_amplitude);

    println!("  • Frequency: {} Hz", harmonic_freq);
    println!("  • Amplitude: {}", harmonic_amplitude);
    println!("  • RMS: {:.3}", harmonic_signal.rms::<f32>());

    // Create percussive component (impulse train)
    println!("\n🥁 Creating percussive component (impulse train)...");
    let impulse_interval = 0.5; // Every 0.5 seconds
    let impulse_amplitude = 0.8;

    // Generate multiple impulses
    let mut impulses = Vec::new();
    let mut t = 0.5; // Start at 0.5 seconds
    while t < duration.as_secs_f64() - 0.1 {
        let impulse_signal = impulse::<f32, f64>(duration, sample_rate, impulse_amplitude, t);
        impulses.push(impulse_signal);
        t += impulse_interval;
    }

    let percussive_signal = AudioSamples::mix::<f32>(&impulses, None)?;

    println!("  • Interval: {} seconds", impulse_interval);
    println!("  • Amplitude: {}", impulse_amplitude);
    println!("  • Number of impulses: {}", impulses.len());
    println!("  • RMS: {:.3}", percussive_signal.rms::<f32>());

    // Mix harmonic and percussive components
    println!("\n🎛️  Mixing components...");
    let mixed_signal = AudioSamples::mix::<f32>(&[harmonic_signal, percussive_signal], None)?;

    println!("  • Mixed RMS: {:.3}", mixed_signal.rms::<f32>());
    println!("  • Mixed peak: {:.3}", mixed_signal.peak());

    // Test different HPSS configurations
    test_hpss_configuration(&mixed_signal, "Default", &HpssConfig::new())?;
    test_hpss_configuration(&mixed_signal, "Musical", &HpssConfig::musical())?;
    test_hpss_configuration(&mixed_signal, "Percussive", &HpssConfig::percussive())?;
    test_hpss_configuration(&mixed_signal, "Harmonic", &HpssConfig::harmonic())?;
    test_hpss_configuration(&mixed_signal, "Real-time", &HpssConfig::realtime())?;

    println!("\n✅ HPSS example completed successfully!");
    println!("\n💡 Tips for better separation:");
    println!("  • Use larger median filters for stronger separation");
    println!("  • Adjust mask softness for smoother/sharper separation");
    println!("  • Use different configurations based on audio content");
    println!("  • Consider the trade-off between separation quality and processing time");

    Ok(())
}

#[cfg(feature = "hpss")]
fn test_hpss_configuration(
    audio: &AudioSamples<'_, f32>,
    config_name: &str,
    config: &HpssConfig<f32>,
) -> AudioSampleResult<()> {
    use std::time::Instant;

    println!("\n🔍 Testing {} configuration:", config_name);
    println!("  • Window size: {} samples", config.win_size);
    println!("  • Hop size: {} samples", config.hop_size);
    println!("  • Harmonic filter: {}", config.median_filter_harmonic);
    println!("  • Percussive filter: {}", config.median_filter_percussive);
    println!("  • Mask softness: {:.2}", config.mask_softness);

    // Perform HPSS separation with timing
    let start_time = Instant::now();
    let (harmonic_separated, percussive_separated) = audio.hpss(config)?;
    let processing_time = start_time.elapsed();

    // Analyze separation results
    let harmonic_rms = harmonic_separated.rms::<f32>();
    let percussive_rms = percussive_separated.rms::<f32>();
    let harmonic_peak = harmonic_separated.peak();
    let percussive_peak = percussive_separated.peak();

    // Calculate energy distribution
    let total_energy = harmonic_rms.powi(2) + percussive_rms.powi(2);
    let harmonic_energy_ratio = harmonic_rms.powi(2) / total_energy * 100.0;
    let percussive_energy_ratio = percussive_rms.powi(2) / total_energy * 100.0;

    println!("\n  📈 Separation Results:");
    println!("    Harmonic component:");
    println!("      - RMS: {:.3}", harmonic_rms);
    println!("      - Peak: {:.3}", harmonic_peak);
    println!("      - Energy ratio: {:.1}%", harmonic_energy_ratio);
    println!("    Percussive component:");
    println!("      - RMS: {:.3}", percussive_rms);
    println!("      - Peak: {:.3}", percussive_peak);
    println!("      - Energy ratio: {:.1}%", percussive_energy_ratio);

    println!(
        "  ⏱️  Processing time: {:.2} ms",
        processing_time.as_secs_f64() * 1000.0
    );

    // Quality assessment
    assess_separation_quality(harmonic_energy_ratio, percussive_energy_ratio);

    Ok(())
}

#[cfg(feature = "hpss")]
fn assess_separation_quality(harmonic_ratio: f32, percussive_ratio: f32) {
    println!("  🎯 Quality Assessment:");

    // For our synthetic signal, we expect roughly balanced energy since both
    // harmonic (sine) and percussive (impulses) components have similar amplitudes
    let balance_score = 100.0 - (harmonic_ratio - percussive_ratio).abs();

    if balance_score > 80.0 {
        println!("    ✅ Excellent separation (balanced energy distribution)");
    } else if balance_score > 60.0 {
        println!("    ✅ Good separation");
    } else if balance_score > 40.0 {
        println!("    ⚠️  Fair separation");
    } else {
        println!("    ❌ Poor separation (highly unbalanced)");
    }

    // Check for reasonable energy levels
    if harmonic_ratio < 10.0 {
        println!("    ⚠️  Very low harmonic energy - might need adjustment");
    }
    if percussive_ratio < 10.0 {
        println!("    ⚠️  Very low percussive energy - might need adjustment");
    }

    println!("    📊 Balance score: {:.1}/100", balance_score);
}

#[cfg(test)]
#[cfg(feature = "hpss")]
mod tests {
    use super::*;

    #[test]
    fn test_hpss_example() {
        // Run the main example logic as a test
        main().expect("HPSS example should complete without error");
    }

    #[test]
    fn test_synthetic_separation() {
        let sample_rate = 8000; // Smaller for faster test
        let duration = Duration::from_millis(100);

        // Create simple test signals
        let harmonic = sine_wave::<f32, f32>(440.0, duration, sample_rate, 0.5);
        let percussive = impulse::<f32, f32>(duration, sample_rate, 0.5, 0.05);
        let mixed = AudioSamples::mix::<f32>(&[harmonic, percussive], None).unwrap();

        // Test separation
        let config = HpssConfig::realtime(); // Faster for testing
        let (h_sep, p_sep) = mixed.hpss(&config).unwrap();

        // Basic sanity checks
        assert_eq!(h_sep.samples_per_channel(), mixed.samples_per_channel());
        assert_eq!(p_sep.samples_per_channel(), mixed.samples_per_channel());
        assert_eq!(h_sep.sample_rate, mixed.sample_rate);
        assert_eq!(p_sep.sample_rate, mixed.sample_rate);

        // Energy should be preserved (approximately)
        let original_rms = mixed.rms::<f32>();
        let separated_rms = (h_sep.rms::<f32>().powi(2) + p_sep.rms::<f32>().powi(2)).sqrt();
        let energy_ratio = separated_rms / original_rms;

        // Energy should be mostly preserved (within 20% due to processing artifacts)
        assert!(
            energy_ratio > 0.8 && energy_ratio < 1.2,
            "Energy not preserved: {} -> {}",
            original_rms,
            separated_rms
        );
    }
}
