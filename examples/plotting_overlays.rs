#[cfg(not(all(feature = "plotting", feature = "transforms")))]
fn main() {
    eprintln!("error: This example requires the `plotting` and `transforms` features.");
    std::process::exit(1);
}

#[cfg(all(feature = "plotting", feature = "transforms"))]
fn main() -> audio_samples::AudioSampleResult<()> {
    {
        use audio_samples::operations::{
            AudioPlotting, AudioStatistics, ChannelManagementStrategy, Layout, PlotParams,
            PlotUtils, SpectrogramPlotParams, WaveformPlotParams,
        };
        use audio_samples::sample_rate;
        println!("Creating test audio signal...");
        let duration = std::time::Duration::from_secs(2);
        let audio = audio_samples::sine_wave::<f64>(440.0, duration, sample_rate!(44100), 0.5);

        println!(
            "Audio stats: {} channels, {} samples, peak = {}",
            audio.num_channels(),
            audio.len(),
            audio.peak()
        );

        // Test waveform plot with overlays
        println!("\nCreating waveform plot with overlays...");
        let waveform_params = WaveformPlotParams::new(
            &PlotParams::new(
                Some("Waveform with Event Markers".to_string()),
                Some("Time (s)".to_string()),
                Some("Amplitude".to_string()),
                None,
                true,
                Some("Events".to_string()),
                None,
                true,
            ),
            Some(ChannelManagementStrategy::Separate(Layout::Vertical)),
            None,
            None,
            None,
            false,
            None,
        );

        let waveform_plot = audio
            .plot_waveform(&waveform_params)
            .expect("Failed to create waveform plot")
            .add_vline(0.5, Some("Event 1"), Some("red"))
            .add_vline(1.0, Some("Event 2"), Some("blue"))
            .add_vline(1.5, Some("Event 3"), Some("green"))
            .add_hline(0.0, Some("Zero"), Some("black"))
            .add_marker(0.25, 0.4, Some("Peak"), None)
            .add_shaded_region(0.8, 1.2, Some("rgba(255,200,0,0.2)"), None);

        let waveform_html = waveform_plot
            .html()
            .expect("Failed to render waveform plot");
        assert!(
            !waveform_html.is_empty() && waveform_html.contains("plotly"),
            "waveform overlay HTML should be a non-empty Plotly document"
        );
        println!(
            "Rendered waveform plot with overlays ({} bytes)",
            waveform_html.len()
        );

        // Test spectrogram plot with overlays
        println!("\nCreating spectrogram plot with overlays...");
        let spec_params = SpectrogramPlotParams::mel_db();
        let spec_plot = audio
            .plot_spectrogram(&spec_params)
            .expect("Failed to create spectrogram plot")
            .add_vline(0.5, Some("Onset"))
            .add_vline(1.5, Some("Offset"))
            .add_hline(440.0, Some("F0 (A4)"));

        let spec_html = spec_plot.html().expect("Failed to render spectrogram plot");
        assert!(
            !spec_html.is_empty() && spec_html.contains("plotly"),
            "spectrogram overlay HTML should be a non-empty Plotly document"
        );
        println!(
            "Rendered spectrogram plot with overlays ({} bytes)",
            spec_html.len()
        );

        // Test pitch contour overlay
        println!("\nCreating spectrogram with pitch contour...");

        // Simulate a simple pitch contour (rising then falling)
        let n_points = 20;
        let times: Vec<f64> = (0..n_points)
            .map(|i| i as f64 * 2.0 / n_points as f64)
            .collect();
        let freqs: Vec<f64> = times
            .iter()
            .map(|&t| {
                // Sine wave modulation of frequency around 440 Hz
                440.0 + 100.0 * (t * std::f64::consts::PI).sin()
            })
            .collect();

        let spec_with_contour = audio
            .plot_spectrogram(&spec_params)
            .expect("Failed to create spectrogram plot")
            .overlay_contour(&times, &freqs, Some("F0 Contour"));

        let contour_html = spec_with_contour
            .html()
            .expect("Failed to render spectrogram with contour");
        assert!(
            !contour_html.is_empty() && contour_html.contains("plotly"),
            "spectrogram-with-contour HTML should be a non-empty Plotly document"
        );
        println!(
            "Rendered spectrogram with pitch contour ({} bytes)",
            contour_html.len()
        );
        Ok(())
    }
}
