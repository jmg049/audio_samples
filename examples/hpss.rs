//! Harmonic/Percussive Source Separation (HPSS) example.
//!
//! This example demonstrates how to use HPSS to separate audio signals into
//! harmonic and percussive components. It creates a synthetic signal combining
//! a sine wave (harmonic) and impulse train (percussive), then separates them.

#[cfg(not(all(feature = "decomposition", feature = "statistics", feature = "editing")))]
fn main() {
    eprintln!("This example requires the 'decomposition', 'statistics', and 'editing' features.");
    std::process::exit(1);
}

#[cfg(all(feature = "decomposition", feature = "statistics", feature = "editing"))]
fn main() -> audio_samples::AudioSampleResult<()> {
    use non_empty_slice::NonEmptyVec;
    use std::time::Duration;

    use audio_samples::{
        AudioEditing, AudioSamples, AudioStatistics,
        utils::generation::{impulse, sine_wave},
    };
    println!("Harmonic/Percussive Source Separation (HPSS) Example");
    println!("=========================================");

    // Audio parameters
    let sample_rate = core::num::NonZeroU32::new(44100).unwrap();
    let duration = Duration::from_secs(3);
    let samples_per_channel = duration.as_secs_f64() * sample_rate.get() as f64;

    println!("\nSignal Parameters:");
    println!("  • Sample rate: {} Hz", sample_rate.get());
    println!("  • Duration: {} seconds", duration.as_secs_f64());
    println!("  • Samples: {}", samples_per_channel as usize);

    // Create harmonic component (sine wave)
    println!("\nCreating harmonic component (sine wave)...");
    let harmonic_freq = 440.0; // A4
    let harmonic_amplitude = 0.6;
    let harmonic_signal =
        sine_wave::<f32>(harmonic_freq, duration, sample_rate, harmonic_amplitude);

    println!("  • Frequency: {} Hz", harmonic_freq);
    println!("  • Amplitude: {}", harmonic_amplitude);
    println!("  • RMS: {:.3}", harmonic_signal.rms());

    // Create percussive component (impulse train)
    println!("\nCreating percussive component (impulse train)...");
    let impulse_interval = 0.5; // Every 0.5 seconds
    let impulse_amplitude = 0.8;

    // Generate multiple impulses
    let mut impulses = Vec::new();
    let mut t = 0.5; // Start at 0.5 seconds
    while t < duration.as_secs_f64() - 0.1 {
        let impulse_signal = impulse::<f32>(duration, sample_rate, impulse_amplitude, t);
        impulses.push(impulse_signal);
        t += impulse_interval;
    }
    let impulses = NonEmptyVec::new(impulses).unwrap();
    let percussive_signal = AudioSamples::mix(&impulses, None)?;

    println!("  • Interval: {} seconds", impulse_interval);
    println!("  • Amplitude: {}", impulse_amplitude);
    println!("  • Number of impulses: {}", impulses.len());
    println!("  • RMS: {:.3}", percussive_signal.rms());

    // Mix harmonic and percussive components
    println!("\nMixing components...");
    let sources = NonEmptyVec::new(vec![harmonic_signal, percussive_signal]).unwrap();
    let mixed_signal = AudioSamples::mix(&sources, None)?;

    println!("  • Mixed RMS: {:.3}", mixed_signal.rms());
    println!("  • Mixed peak: {:.3}", mixed_signal.peak());

    Ok(())
}

#[cfg(test)]
#[cfg(all(feature = "decomposition", feature = "statistics", feature = "editing"))]
mod tests {
    use audio_samples::sample_rate;

    use super::*;

    #[test]
    fn test_hpss_example() {
        // Run the main example logic as a test
        main().expect("HPSS example should complete without error");
    }

    #[test]
    fn test_synthetic_separation() {
        let sample_rate = sample_rate!(8000); // Smaller for faster test
        let duration = Duration::from_millis(100);

        // Create simple test signals
        let harmonic = sine_wave::<f32, f32>(440.0, duration, sample_rate, 0.5);
        let percussive = impulse::<f32, f32>(duration, sample_rate, 0.5, 0.05);

        let sources = NonEmptySlice::new(&[harmonic, percussive]).unwrap();
        let mixed = AudioSamples::mix::<f32>(&sources, None).unwrap();

        // Test separation
        let config = HpssConfig::realtime(); // Faster for testing
        let (h_sep, p_sep) = mixed.hpss(&config).unwrap();

        // Basic sanity checks
        assert_eq!(h_sep.samples_per_channel(), mixed.samples_per_channel());
        assert_eq!(p_sep.samples_per_channel(), mixed.samples_per_channel());
        assert_eq!(h_sep.sample_rate, mixed.sample_rate);
        assert_eq!(p_sep.sample_rate, mixed.sample_rate);

        // Energy should be preserved (approximately)
        let original_rms = mixed.rms();
        let separated_rms = (h_sep.rms().powi(2) + p_sep.rms().powi(2)).sqrt();
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
