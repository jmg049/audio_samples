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

    // Capture component properties before they are moved into the mix.
    let harmonic_rms = harmonic_signal.rms();
    let percussive_rms = percussive_signal.rms();
    let component_len = harmonic_signal.samples_per_channel();

    // Mix harmonic and percussive components
    println!("\nMixing components...");
    let sources = NonEmptyVec::new(vec![harmonic_signal, percussive_signal]).unwrap();
    let mixed_signal = AudioSamples::mix(&sources, None)?;

    println!("  • Mixed RMS: {:.3}", mixed_signal.rms());
    println!("  • Mixed peak: {:.3}", mixed_signal.peak());

    // --- Self-verification -------------------------------------------------
    // We placed an impulse every 0.5 s starting at 0.5 s over a 3 s signal.
    assert_eq!(
        impulses.len().get(),
        5,
        "expected 5 impulses across the 3 s signal"
    );
    assert!(
        harmonic_rms > 0.0 && percussive_rms > 0.0,
        "both components must be non-silent"
    );
    // Mixing two non-silent sources produces a non-silent result.
    assert!(mixed_signal.rms() > 0.0, "mixed signal must be non-silent");
    assert_eq!(
        mixed_signal.samples_per_channel(),
        component_len,
        "mix preserves the per-channel sample count"
    );

    Ok(())
}

#[cfg(test)]
#[cfg(all(feature = "decomposition", feature = "statistics", feature = "editing"))]
mod tests {
    use audio_samples::operations::AudioDecomposition;
    use audio_samples::operations::hpss::HpssConfig;
    use audio_samples::utils::generation::{impulse, sine_wave};
    use audio_samples::{AudioEditing, AudioSamples, AudioStatistics, sample_rate};
    use non_empty_slice::NonEmptySlice;
    use std::time::Duration;

    #[test]
    fn test_hpss_example() {
        // Run the main example logic as a test
        super::main().expect("HPSS example should complete without error");
    }

    #[test]
    fn test_synthetic_separation() {
        let sample_rate = sample_rate!(8000); // Smaller for faster test
        // Keep this comfortably above the HPSS window size (1024 samples):
        // 8 kHz * 0.5 s = 4000 samples.
        let duration = Duration::from_millis(500);

        // Create simple test signals
        let harmonic = sine_wave::<f32>(440.0, duration, sample_rate, 0.5);
        let percussive = impulse::<f32>(duration, sample_rate, 0.5, 0.05);

        let sources = [harmonic, percussive];
        let sources = NonEmptySlice::from_slice(&sources).unwrap();
        let mixed = AudioSamples::mix(sources, None).unwrap();

        // Test separation
        let config = HpssConfig::realtime(); // Faster for testing
        let (h_sep, p_sep) = mixed.hpss(&config).unwrap();

        // Basic sanity checks. HPSS works on whole STFT frames, so the output
        // length may be trimmed to a frame boundary (it will not exceed the
        // input length), but both components share the same length and rate.
        assert_eq!(
            h_sep.samples_per_channel(),
            p_sep.samples_per_channel(),
            "harmonic and percussive parts must have equal length"
        );
        assert!(
            h_sep.samples_per_channel().get() <= mixed.samples_per_channel().get(),
            "separated length must not exceed the input length"
        );
        assert!(
            h_sep.samples_per_channel().get() * 100 >= mixed.samples_per_channel().get() * 90,
            "separated length should retain at least ~90% of the input"
        );
        assert_eq!(h_sep.sample_rate(), mixed.sample_rate());
        assert_eq!(p_sep.sample_rate(), mixed.sample_rate());

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
