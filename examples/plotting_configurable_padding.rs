#[cfg(not(all(feature = "plotting", feature = "transforms")))]
fn main() {
    eprintln!("error: This example requires the `plotting` and `transforms` features.");
    std::process::exit(1);
}

#[cfg(all(feature = "plotting", feature = "transforms"))]
fn main() -> audio_samples::AudioSampleResult<()> {
    use std::time::Duration;

    use audio_samples::operations::AudioPlotting;
    use audio_samples::operations::plotting::PlotUtils;
    use audio_samples::operations::plotting::spectrograms::SpectrogramPlotParams;
    use audio_samples::{sample_rate, sine_wave};
    println!("Testing configurable frequency range padding...\n");

    // Generate a 440 Hz tone
    let audio = sine_wave::<f32>(
        440.0,
        Duration::from_secs_f32(1.0),
        sample_rate!(44100),
        0.5,
    );

    // Test with different padding values
    let padding_options = vec![
        (0.0, "0% (tight fit)"),
        (0.1, "10% (default)"),
        (0.3, "30% (more context)"),
        (0.5, "50% (lots of context)"),
    ];

    for (padding, description) in padding_options {
        println!("Creating spectrogram with {} padding...", description);

        let mut params = SpectrogramPlotParams::mel_db();
        params.freq_range_padding = Some(padding);

        let plot = audio.plot_spectrogram(&params)?;
        let filename = format!("outputs/test_padding_{:.0}pct.html", padding * 100.0);
        plot.save(&filename)?;

        println!("  Saved: {}", filename);
    }
    Ok(())
}
