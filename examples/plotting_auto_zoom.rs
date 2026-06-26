#[cfg(not(feature = "plotting"))]
fn main() {
    eprintln!("error: This example requires the `plotting` feature.");
    std::process::exit(1);
}

#[cfg(feature = "plotting")]
fn main() -> audio_samples::AudioSampleResult<()> {
    use std::time::Duration;

    use audio_samples::operations::AudioPlotting;
    use audio_samples::operations::plotting::PlotUtils;
    use audio_samples::operations::plotting::spectrograms::SpectrogramPlotParams;
    use audio_samples::{sample_rate, sine_wave};
    println!("Testing auto-zoom feature...");

    let audio = sine_wave::<f32>(
        440.0,
        Duration::from_secs_f32(1.0),
        sample_rate!(44100),
        0.5,
    );

    // Test 1: Auto-zoom enabled (default)
    println!("Creating spectrogram with auto-zoom enabled...");
    let params_auto = SpectrogramPlotParams::mel_db();
    let plot_auto = audio.plot_spectrogram(&params_auto)?;
    let html_auto = plot_auto.html()?;
    println!("Rendered auto-zoom spectrogram ({} bytes)", html_auto.len());

    // Test 2: Auto-zoom disabled (show full range)
    println!("Creating spectrogram with auto-zoom disabled...");
    let mut params_manual = SpectrogramPlotParams::mel_db();
    params_manual.auto_zoom_freq = false;
    let plot_manual = audio.plot_spectrogram(&params_manual)?;
    let html_manual = plot_manual.html()?;
    println!("Rendered fixed-range spectrogram ({} bytes)", html_manual.len());

    // Test 3: Manual frequency range override
    println!("Creating spectrogram with manual frequency range...");
    let mut params_override = SpectrogramPlotParams::mel_db();
    params_override.freq_range = Some((200.0, 1000.0));
    let plot_override = audio.plot_spectrogram(&params_override)?;
    let html_override = plot_override.html()?;
    println!("Rendered manual-range spectrogram ({} bytes)", html_override.len());

    // All three render paths must produce non-empty HTML documents.
    for (name, html) in [
        ("auto", &html_auto),
        ("manual", &html_manual),
        ("override", &html_override),
    ] {
        assert!(
            !html.is_empty() && html.contains("plotly"),
            "{name} spectrogram HTML should be a non-empty Plotly document"
        );
    }

    Ok(())
}
