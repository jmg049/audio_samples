#[cfg(not(all(feature = "plotting", feature = "statistics")))]
fn main() {
    eprintln!("error: This example requires the `plotting` and `statistics` features.");
    std::process::exit(1);
}

#[cfg(all(feature = "plotting", feature = "statistics"))]
fn main() -> audio_samples::AudioSampleResult<()> {
    use audio_samples::{AudioSamples, chirp, sample_rate};
    use audio_samples::operations::plotting::dsp_overlays;
    use audio_samples::operations::plotting::spectrograms::SpectrogramPlotParams;
    use audio_samples::operations::{AudioPlotting, PlotUtils};
    use std::time::Duration;

    println!("=== Spectral Overlays Example ===\n");

    let audio: AudioSamples<'_, f32> = chirp::<f32>(
        200.0,                        // start frequency
        2000.0,                       // end frequency
        Duration::from_secs_f32(4.0), // duration
        sample_rate!(44100),
        0.5, // amplitude
    );
    println!("Generated chirp signal: 200 Hz → 2000 Hz sweep over 4 seconds");

    // Window parameters for spectral analysis
    let window_size = audio_samples::nzu!(2048); // ~46ms at 44.1kHz
    let hop_size = audio_samples::nzu!(512); // ~12ms hop

    println!("\nSpectral analysis parameters:");
    println!(
        "  Window size: {} samples (~{:.1}ms)",
        window_size,
        window_size.get() as f64 / 44100.0 * 1000.0
    );
    println!(
        "  Hop size: {} samples (~{:.1}ms)",
        hop_size,
        hop_size.get() as f64 / 44100.0 * 1000.0
    );

    // Example 1: Spectrogram with spectral centroid overlay
    println!("\n--- Spectral Centroid Overlay ---");
    let (centroid_times, centroid_values) =
        dsp_overlays::compute_windowed_spectral_centroid(&audio.borrow(), window_size, hop_size);

    println!(
        "Computed spectral centroid for {} windows",
        centroid_times.len()
    );
    println!(
        "Centroid range: {:.0} Hz to {:.0} Hz",
        centroid_values
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min),
        centroid_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );

    let spectrogram_plot = audio
        .plot_spectrogram(&SpectrogramPlotParams::mel_db())?
        .add_spectral_centroid(centroid_times.clone(), centroid_values.clone(), None);

    spectrogram_plot.save("output/spectral_overlay_centroid.html")?;
    println!("\nSaved spectrogram with centroid to: output/spectral_overlay_centroid.html");
    println!("  White line with cyan markers = Spectral Centroid");

    // Example 2: Spectrogram with spectral rolloff overlay (85%)
    println!("\n--- Spectral Rolloff Overlay ---");
    let rolloff_percent = 0.85; // 85% energy cutoff
    let (rolloff_times, rolloff_values) = dsp_overlays::compute_windowed_spectral_rolloff(
        &audio.borrow(),
        window_size,
        hop_size,
        rolloff_percent,
    );

    println!(
        "Computed spectral rolloff ({}%) for {} windows",
        (rolloff_percent * 100.0) as u32,
        rolloff_times.len()
    );
    println!(
        "Rolloff range: {:.0} Hz to {:.0} Hz",
        rolloff_values.iter().cloned().fold(f64::INFINITY, f64::min),
        rolloff_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );

    let rolloff_plot = audio
        .plot_spectrogram(&SpectrogramPlotParams::mel_db())?
        .add_spectral_rolloff(rolloff_times.clone(), rolloff_values.clone(), None);

    rolloff_plot.save("output/spectral_overlay_rolloff.html")?;
    println!("\nSaved spectrogram with rolloff to: output/spectral_overlay_rolloff.html");
    println!("  White line with cyan markers = Spectral Rolloff (85%)");

    // Example 3: Combined - both centroid and rolloff
    println!("\n--- Combined Spectral Features ---");

    let combined_plot = audio
        .plot_spectrogram(&SpectrogramPlotParams::mel_db())?
        .add_spectral_centroid(centroid_times, centroid_values, Some("Centroid"))
        .add_spectral_rolloff(rolloff_times, rolloff_values, Some("Rolloff 85%"));

    combined_plot.save("output/spectral_overlay_combined.html")?;
    println!("Saved combined spectrogram to: output/spectral_overlay_combined.html");
    println!("  Shows both Centroid and Rolloff tracks with legend");

    // Example 4: Different rolloff percentages comparison
    println!("\n--- Multiple Rolloff Percentages ---");

    let (rolloff_50_times, rolloff_50_values) = dsp_overlays::compute_windowed_spectral_rolloff(
        &audio.borrow(),
        window_size,
        hop_size,
        0.50,
    );
    let (rolloff_85_times, rolloff_85_values) = dsp_overlays::compute_windowed_spectral_rolloff(
        &audio.borrow(),
        window_size,
        hop_size,
        0.85,
    );
    let (rolloff_95_times, rolloff_95_values) = dsp_overlays::compute_windowed_spectral_rolloff(
        &audio.borrow(),
        window_size,
        hop_size,
        0.95,
    );
    let comparison_plot = audio
        .plot_spectrogram(&SpectrogramPlotParams::mel_db())?
        .add_spectral_rolloff(rolloff_50_times, rolloff_50_values, Some("Rolloff 50%"))
        .add_spectral_rolloff(rolloff_85_times, rolloff_85_values, Some("Rolloff 85%"))
        .add_spectral_rolloff(rolloff_95_times, rolloff_95_values, Some("Rolloff 95%"));

    comparison_plot.save("output/spectral_overlay_rolloff_comparison.html")?;
    Ok(())
}
