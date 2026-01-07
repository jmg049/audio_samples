#[cfg(not(feature = "plotting"))]
fn main() {
    eprintln!("error: This example requires the `plotting` feature.");
    std::process::exit(1);
}

#[cfg(feature = "plotting")]
fn main() -> audio_samples::AudioSampleResult<()> {
    use audio_samples::operations::AudioPlotting;
    use audio_samples::operations::plotting::{PlotUtils, spectrum::MagnitudeSpectrumParams};
    use audio_samples::utils::MonoSampleBuilder;
    use audio_samples::sample_rate;
    use std::time::Duration;

    println!("Testing MagnitudeSpectrumPlot...");

    // Generate a signal with multiple frequency components
    // 440 Hz (A4), 880 Hz (A5), 1320 Hz (E6)
    let duration = Duration::from_secs_f32(1.0);
    let audio = MonoSampleBuilder::<f32>::new(sample_rate!(44100))
        .sine_wave(440.0, duration, 0.5)
        .sine_wave(880.0, duration, 0.5)
        .sine_wave(1320.0, duration, 0.5)
        .build()?;

    // Test 1: dB scale spectrum (default)
    println!("Creating magnitude spectrum (dB scale)...");
    let spec_db = audio.plot_magnitude_spectrum(&MagnitudeSpectrumParams::db())?;
    spec_db.save("outputs/test_spectrum_db.html")?;
    println!("Saved: outputs/test_spectrum_db.html");

    // Test 2: Linear scale spectrum
    println!("Creating magnitude spectrum (linear scale)...");
    let spec_linear = audio.plot_magnitude_spectrum(&MagnitudeSpectrumParams::linear())?;
    spec_linear.save("outputs/test_spectrum_linear.html")?;
    println!("Saved: outputs/test_spectrum_linear.html");

    // Test 3: Zoom into frequency range of interest (0-2000 Hz)
    println!("Creating zoomed magnitude spectrum...");
    let mut params_zoomed = MagnitudeSpectrumParams::db();
    params_zoomed.freq_range = Some((0.0, 2000.0));
    let spec_zoomed = audio.plot_magnitude_spectrum(&params_zoomed)?;
    spec_zoomed.save("outputs/test_spectrum_zoomed.html")?;
    println!("Saved: outputs/test_spectrum_zoomed.html");

    // Test 4: Show in browser
    #[cfg(feature = "html_view")]
    {
        println!("Opening dB spectrum in browser...");
        spec_db.show()?;
    }

    #[cfg(not(feature = "html_view"))]
    {
        println!("Note: Enable 'html_view' feature to open in browser");
    }

    println!("All magnitude spectrum tests passed!");
    Ok(())
}
