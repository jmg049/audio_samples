use audio_samples::{
    AudioEditing, AudioSampleResult, AudioSamples, AudioTransforms,
    operations::types::ChromaConfig, sine_wave,
};
use std::time::Duration;

pub fn main() -> AudioSampleResult<()> {
    println!("Chromagram Feature Demonstration");
    println!("=====================================");

    let sample_rate_hz = 44_100u32;

    // Create a simple chord progression: C major chord (C-E-G)
    println!("\n1. Creating C major chord (C-E-G) audio signal...");
    let c_note = sine_wave::<f64, f64>(261.63, Duration::from_millis(1000), sample_rate_hz, 0.3); // C4
    let e_note = sine_wave::<f64, f64>(329.63, Duration::from_millis(1000), sample_rate_hz, 0.3); // E4
    let g_note = sine_wave::<f64, f64>(392.00, Duration::from_millis(1000), sample_rate_hz, 0.3); // G4

    // Mix the notes together using AudioSamples::mix
    let chord = AudioSamples::mix::<f64>(&[c_note, e_note, g_note], None)?;

    println!(
        "   - Audio length: {:.2}s",
        chord.samples_per_channel() as f64 / sample_rate_hz as f64
    );
    println!("   - Sample rate: {} Hz", sample_rate_hz);

    // Example 1: Basic chromagram with default settings
    println!("\n2. Basic chromagram with default STFT method:");
    let config_default = ChromaConfig::<f64>::new();
    let chroma_default = chord.chromagram(&config_default)?;
    println!("   - Method: STFT");
    println!(
        "   - Shape: {:?} (chroma_bins × time_frames)",
        chroma_default.dim()
    );
    println!("   - Window size: {} samples", config_default.window_size);
    println!("   - Hop size: {} samples", config_default.hop_size);

    // Display some chroma values for the first frame
    println!("   - First frame chroma values:");
    for i in 0..config_default.n_chroma {
        let note_names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        println!("     {}: {:.3}", note_names[i], chroma_default[[i, 0]]);
    }

    // Example 2: STFT-based chromagram with custom settings
    println!("\n3. Custom STFT chromagram (high resolution):");
    let config_stft = ChromaConfig::<f64>::high_resolution();
    let chroma_stft = chord.chromagram(&config_stft)?;
    println!("   - Method: STFT");
    println!("   - Shape: {:?}", chroma_stft.dim());
    println!("   - Window size: {} samples", config_stft.window_size);
    println!("   - Hop size: {} samples", config_stft.hop_size);

    // Example 3: CQT-based chromagram
    println!("\n4. CQT-based chromagram:");
    let config_cqt = ChromaConfig::<f64>::cqt();
    let chroma_cqt = chord.chromagram(&config_cqt)?;
    println!("   - Method: CQT");
    println!("   - Shape: {:?}", chroma_cqt.dim());
    println!("   - Better frequency resolution for lower frequencies");

    // Example 4: Real-time configuration
    println!("\n5. Real-time chromagram (low latency):");
    let config_realtime = ChromaConfig::<f64>::realtime();
    let chroma_realtime = chord.chromagram(&config_realtime)?;
    println!("   - Method: STFT (low latency)");
    println!("   - Shape: {:?}", chroma_realtime.dim());
    println!("   - Window size: {} samples", config_realtime.window_size);
    println!("   - Hop size: {} samples", config_realtime.hop_size);

    // Example 5: Custom configuration with builder pattern
    println!("\n6. Custom configuration using builder pattern:");
    let config_custom = ChromaConfig::<f64>::new()
        .with_n_chroma(24) // 24-tone equal temperament
        .with_window_size(1024)
        .with_hop_size(512)
        .with_norm(false); // No normalization

    let chroma_custom = chord.chromagram(&config_custom)?;
    println!("   - Chroma bins: {}", config_custom.n_chroma);
    println!("   - Shape: {:?}", chroma_custom.dim());
    println!("   - Normalization: disabled");

    // Example 6: Backward compatibility test with old chroma method
    println!("\n7. Backward compatibility test:");
    let chroma_old = chord.chroma::<f64>(12)?;
    println!("   - Old method shape: {:?}", chroma_old.dim());
    println!("   - Compatible with existing code: ✓");

    // Compare methods
    println!("\n8. Method comparison summary:");
    println!("   ┌─────────────┬─────────────┬──────────────┐");
    println!("   │ Method      │ Time Frames │ Frequency    │");
    println!("   │             │             │ Resolution   │");
    println!("   ├─────────────┼─────────────┼──────────────┤");
    println!(
        "   │ STFT        │ {:>11} │ Linear       │",
        chroma_stft.dim().1
    );
    println!(
        "   │ CQT         │ {:>11} │ Logarithmic  │",
        chroma_cqt.dim().1
    );
    println!(
        "   │ Real-time   │ {:>11} │ Low latency  │",
        chroma_realtime.dim().1
    );
    println!("   └─────────────┴─────────────┴──────────────┘");

    // Example 7: Testing with different musical content
    println!("\n9. Testing with A minor chord (A-C-E):");
    let a_note = sine_wave::<f64, f64>(220.00, Duration::from_millis(1000), sample_rate_hz, 0.3); // A3
    let c_note2 = sine_wave::<f64, f64>(261.63, Duration::from_millis(1000), sample_rate_hz, 0.3); // C4
    let e_note2 = sine_wave::<f64, f64>(329.63, Duration::from_millis(1000), sample_rate_hz, 0.3); // E4

    let a_minor = AudioSamples::mix::<f64>(&[a_note, c_note2, e_note2], None)?;

    let chroma_a_minor = a_minor.chromagram(&ChromaConfig::<f64>::new())?;
    println!("   - A minor chord shape: {:?}", chroma_a_minor.dim());

    println!("   - First frame chroma values:");
    for i in 0..12 {
        let note_names = [
            "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
        ];
        println!("     {}: {:.3}", note_names[i], chroma_a_minor[[i, 0]]);
    }

    Ok(())
}
