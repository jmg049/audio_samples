#[cfg(not(all(feature = "plotting", feature = "transforms")))]
fn main() {
    eprintln!("error: This example requires the `plotting` and `transforms` features.");
    std::process::exit(1);
}

#[cfg(all(feature = "plotting", feature = "transforms"))]
fn main() -> audio_samples::AudioSampleResult<()> {
    use std::time::Duration;

    use audio_samples::operations::AudioPlotting;
    use audio_samples::operations::plotting::{
        PlotUtils,
        composite::{CompositeLayout, CompositePlot},
        spectrograms::SpectrogramPlotParams,
        waveform::WaveformPlotParams,
    };
    use audio_samples::{chirp, sample_rate};

    println!("Testing CompositePlot with waveform + spectrogram...");
    // Generate a chirp signal (sweeping frequency from 200 Hz to 2000 Hz)
    let duration = Duration::from_secs_f32(2.0); // seconds
    let audio = chirp::<f32>(200.0, 2000.0, duration, sample_rate!(44100), 0.5);

    // Create waveform plot
    println!("Creating waveform plot...");
    let waveform_params = WaveformPlotParams::default();
    let waveform = audio.plot_waveform(&waveform_params)?;

    // Create spectrogram plot
    println!("Creating spectrogram plot...");
    let spec_params = SpectrogramPlotParams::mel_db();
    let spectrogram = audio.plot_spectrogram(&spec_params)?;

    // Test 1: Vertical composition (most common)
    println!("Creating vertical composite...");
    let vertical = CompositePlot::new()
        .add_plot(waveform)
        .add_plot(spectrogram)
        .layout(CompositeLayout::Vertical)
        .build()?;

    vertical.save("outputs/test_composite_vertical.html")?;
    println!("Saved: outputs/test_composite_vertical.html");

    // Test 2: Horizontal composition
    // Create new plots since we consumed the originals
    println!("Creating horizontal composite...");
    let waveform2 = audio.plot_waveform(&waveform_params)?;
    let spectrogram2 = audio.plot_spectrogram(&spec_params)?;

    let horizontal = CompositePlot::new()
        .add_plot(waveform2)
        .add_plot(spectrogram2)
        .layout(CompositeLayout::Horizontal)
        .build()?;

    horizontal.save("outputs/test_composite_horizontal.html")?;
    println!("Saved: outputs/test_composite_horizontal.html");

    // Test 3: Show vertical composite
    #[cfg(feature = "html_view")]
    {
        println!("Opening vertical composite in html_view...");
        vertical.show()?;
    }

    #[cfg(not(feature = "html_view"))]
    {
        println!("Note: Enable 'html_view' feature to open in html_view");
    }

    Ok(())
}
