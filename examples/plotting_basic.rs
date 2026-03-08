#[cfg(not(all(feature = "plotting", feature = "transforms")))]
fn main() {
    eprintln!("error: This example requires the `plotting` and `transforms` features.");
    std::process::exit(1);
}

#[cfg(all(feature = "plotting", feature = "transforms"))]
fn main() -> audio_samples::AudioSampleResult<()> {
    use audio_samples::operations::{
        AudioPlotting, PlotParams, WaveformPlotParams,
        plotting::{PlotUtils, spectrograms::SpectrogramPlotParams},
    };
    use audio_samples::{AudioStatistics, sample_rate};

    println!("Creating test audio signal...");
    let duration = std::time::Duration::from_secs(1);
    let audio = audio_samples::stereo_sine_wave::<f64>(440.0, duration, sample_rate!(44100), 0.5);

    println!(
        "Audio stats: {} channels, {} samples, peak = {}",
        audio.num_channels(),
        audio.len(),
        audio.peak()
    );

    // Test waveform plot
    println!("\nCreating waveform plot...");
    let waveform_params = WaveformPlotParams::new(
        &PlotParams::new(
            Some("Test Waveform".to_string()),
            Some("Time (s)".to_string()),
            Some("Amplitude".to_string()),
            None,
            true,
            Some("Channels".to_string()),
            None,
            true,
        ),
        Some(
            audio_samples::operations::plotting::ChannelManagementStrategy::Separate(
                audio_samples::operations::plotting::Layout::Vertical,
            ),
        ),
        None,
        None,
        None,
        false,
        None,
    );

    let waveform_plot = audio
        .plot_waveform(&waveform_params)
        .expect("Failed to create waveform plot");
    waveform_plot
        .save("test_waveform.html")
        .expect("Failed to save waveform plot");
    println!("Saved waveform plot to test_waveform.html");

    // Test spectrogram plot
    println!("\nCreating spectrogram plot...");
    let spec_params = SpectrogramPlotParams::mel_db();
    let spec_plot = audio
        .plot_spectrogram(&spec_params)
        .expect("Failed to create spectrogram plot");
    spec_plot
        .save("outputs/test_spectrogram.html")
        .expect("Failed to save spectrogram plot");
    println!("Saved spectrogram plot to outputs/test_spectrogram.html");
    Ok(())
}
