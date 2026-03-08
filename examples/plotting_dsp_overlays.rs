//! Example demonstrating DSP overlay functionality on waveform plots
//!
//! This example shows how to:
//! 1. Create a waveform plot
//! 2. Compute DSP features using the dsp_overlays module
//! 3. Add RMS envelope, peak envelope, ZCR, and energy overlays
//!
//! Run with:
//! ```bash
//! cargo run --example plotting_dsp_overlays --features plotting,statistics
//! ```

#[cfg(not(all(feature = "plotting", feature = "statistics")))]
fn main() {
    eprintln!("error: This example requires the `plotting` and `statistics` features.");
    std::process::exit(1);
}

#[cfg(all(feature = "plotting", feature = "statistics"))]
fn create_test_audio() -> audio_samples::AudioSamples<'static, f32> {
    use audio_samples::{sample_rate, utils::generation::am_signal};
    use std::time::Duration;
    let carrier_freq = 440.0; // A4 note
    let mod_freq = 2.0; // 2 Hz modulation for slow amplitude
    let mod_depth = 1.0; // Full modulation depth
    let duration = Duration::from_secs_f32(2.0); // 2 seconds
    let sample_rate = sample_rate!(44100);

    am_signal::<f32>(
        carrier_freq,
        mod_freq,
        mod_depth,
        duration,
        sample_rate,
        0.5, // amplitude
    )
}

#[cfg(all(feature = "plotting", feature = "statistics"))]
fn main() -> audio_samples::AudioSampleResult<()> {
    use audio_samples::operations::plotting::dsp_overlays;
    use audio_samples::operations::plotting::waveform::WaveformPlotParams;
    use audio_samples::operations::{AudioPlotting, PlotUtils};
    use audio_samples::{nzu, sample_rate};
    println!("=== DSP Overlays Example ===\n");

    let sample_rate = sample_rate!(44100);
    let duration = 2.0;
    let n_samples = (sample_rate.get() as f64 * duration) as usize;
    let audio = create_test_audio();
    let sample_rate_hz = audio.sample_rate_hz();

    println!("Generated {} samples at {} Hz", n_samples, sample_rate_hz);
    println!("Duration: {:.2} seconds\n", duration);

    // Configuration for DSP overlays
    let window_size = nzu!(2048); // ~46ms at 44.1kHz
    let hop_size = nzu!(512); // ~12ms hop for smooth overlay

    println!("DSP overlay parameters:");
    println!(
        "  Window size: {} samples (~{:.1}ms)",
        window_size,
        window_size.get() as f64 / sample_rate_hz * 1000.0
    );
    println!(
        "  Hop size: {} samples (~{:.1}ms)\n",
        hop_size,
        hop_size.get() as f64 / sample_rate_hz * 1000.0
    );

    // Example 1: Waveform with RMS envelope
    println!("Creating waveform with RMS envelope overlay...");
    let (rms_times, rms_values) =
        dsp_overlays::compute_windowed_rms(&audio.borrow(), window_size, hop_size);

    let plot_rms = audio
        .plot_waveform(&WaveformPlotParams::default())?
        .add_rms_envelope(rms_times, rms_values, Some("red"), Some(2.5));

    plot_rms.save("output/dsp_overlay_rms.html")?;
    println!("  Saved to: output/dsp_overlay_rms.html");

    // Example 2: Waveform with Peak envelope
    println!("\nCreating waveform with Peak envelope overlay...");
    let (peak_times, peak_values) =
        dsp_overlays::compute_windowed_peak(&audio.borrow(), window_size, hop_size);

    let plot_peak = audio
        .plot_waveform(&WaveformPlotParams::default())?
        .add_peak_envelope(peak_times, peak_values, Some("orange"), Some(2.5));

    plot_peak.save("output/dsp_overlay_peak.html")?;
    println!("  Saved to: output/dsp_overlay_peak.html");

    // Example 3: Waveform with Zero-Crossing Rate overlay
    println!("\nCreating waveform with ZCR overlay...");
    let (zcr_times, zcr_values) =
        dsp_overlays::compute_windowed_zcr(&audio.borrow(), window_size, hop_size);

    let plot_zcr = audio
        .plot_waveform(&WaveformPlotParams::default())?
        .add_zcr_overlay(zcr_times, zcr_values, Some("blue"), Some(2.0));

    plot_zcr.save("output/dsp_overlay_zcr.html")?;
    println!("  Saved to: output/dsp_overlay_zcr.html");
    println!("  Note: ZCR uses secondary y-axis (right side)");

    // Example 4: Waveform with Energy overlay
    println!("\nCreating waveform with Energy overlay...");
    let (energy_times, energy_values) =
        dsp_overlays::compute_windowed_energy(&audio.borrow(), window_size, hop_size);

    let plot_energy = audio
        .plot_waveform(&WaveformPlotParams::default())?
        .add_energy_overlay(energy_times, energy_values, Some("green"), Some(2.5));

    plot_energy.save("output/dsp_overlay_energy.html")?;
    println!("  Saved to: output/dsp_overlay_energy.html");
    println!("  Note: Energy uses secondary y-axis (right side)");

    // Example 5: Waveform with multiple overlays combined
    println!("\nCreating waveform with multiple overlays...");
    let (rms_times, rms_values) =
        dsp_overlays::compute_windowed_rms(&audio.borrow(), window_size, hop_size);
    let (peak_times, peak_values) =
        dsp_overlays::compute_windowed_peak(&audio.borrow(), window_size, hop_size);

    let plot_combined = audio
        .plot_waveform(&WaveformPlotParams::default())?
        .add_rms_envelope(rms_times, rms_values, Some("red"), Some(2.0))
        .add_peak_envelope(peak_times, peak_values, Some("orange"), Some(1.5));

    plot_combined.save("output/dsp_overlay_combined.html")?;
    println!("  Saved to: output/dsp_overlay_combined.html");

    // Example 6: Different window sizes comparison
    println!("\nCreating comparison with different window sizes...");

    // Small window (more temporal detail, noisier)
    let (rms_times_small, rms_values_small) =
        dsp_overlays::compute_windowed_rms(&audio.borrow(), nzu!(512), nzu!(128));

    // Large window (smoother, less temporal detail)
    let (rms_times_large, rms_values_large) =
        dsp_overlays::compute_windowed_rms(&audio.borrow(), nzu!(4096), nzu!(1024));
    let plot_comparison = audio
        .plot_waveform(&WaveformPlotParams::default())?
        .add_rms_envelope(rms_times_small, rms_values_small, Some("red"), Some(1.0))
        .add_rms_envelope(
            rms_times_large,
            rms_values_large,
            Some("darkred"),
            Some(3.0),
        );

    plot_comparison.save("output/dsp_overlay_window_comparison.html")?;
    println!("  Saved to: output/dsp_overlay_window_comparison.html");

    Ok(())
}
