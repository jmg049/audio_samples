#[cfg(not(all(feature = "plotting", feature = "envelopes")))]
fn main() {
    eprintln!("error: This example requires the `plotting` and `envelopes` features.");
    std::process::exit(1);
}

#[cfg(all(feature = "plotting", feature = "envelopes"))]
fn main() -> audio_samples::AudioSampleResult<()> {
    use std::time::Duration;

    use audio_samples::operations::plotting::{
        PlotParams, PlotUtils,
        composite::{CompositeLayout, CompositePlot},
        spectrum::MagnitudeSpectrumParams,
        waveform::WaveformPlotParams,
    };

    use audio_samples::operations::{AudioEnvelopes, AudioPlotting};
    use audio_samples::{sample_rate, utils::generation::am_signal};
    use ndarray::Array1;
    println!("Testing advanced plot composition with overlays...");

    // Generate amplitude-modulated signal (440 Hz carrier, 2 Hz modulation)
    let audio = am_signal::<f32>(
        440.0,
        2.0,
        1.0,
        Duration::from_secs_f32(2.0),
        sample_rate!(44100),
        0.5,
    );

    // Compute RMS envelope
    println!("Computing RMS envelope...");
    let window_size = audio_samples::nzu!(2048); // ~46ms at 44.1kHz
    let hop_size = audio_samples::nzu!(512); // ~12ms hop
    // let (rms_times, rms_values) = compute_rms_envelope(&audio, window_size, hop_size, 44100);
    let rms_envelope = audio.rms_envelope(window_size, hop_size).unwrap();
    let envelope: Array1<f32> = rms_envelope.into_array1().expect("We know audio is 1D");

    let rms_times: Vec<f64> = (0..envelope.len())
        .map(|i| (i * hop_size.get()) as f64 / audio.sample_rate_hz())
        .collect();
    let rms_values: Vec<f64> = envelope.iter().map(|&v| v as f64).collect();

    // Create waveform plot
    println!("Creating waveform plot with overlays...");
    let waveform_params = WaveformPlotParams {
        plot_params: PlotParams {
            title: Some("Amplitude-Modulated Signal".to_string()),
            x_label: Some("Time (s)".to_string()),
            y_label: Some("Amplitude".to_string()),
            ..Default::default()
        },
        ch_mgmt_strategy: None,
        color: Some("blue".to_string()),
        line_style: None,
        line_width: Some(1.0),
        markers: false,
        save_path: None,
    };

    let waveform = audio
        .plot_waveform(&waveform_params)?
        .add_vline(0.5, Some("0.5s Mark"), Some("gray"))
        .add_vline(1.0, Some("1.0s Mark"), Some("gray"))
        .add_vline(1.5, Some("1.5s Mark"), Some("gray"))
        .add_rms_envelope(rms_times, rms_values, Some("red"), Some(2.5))
        .add_shaded_region(0.75, 1.25, Some("rgba(255,200,0,0.2)"), None);

    // Create magnitude spectrum plot
    println!("Creating magnitude spectrum plot...");
    let mut spectrum_params = MagnitudeSpectrumParams::db();
    spectrum_params.plot_params.title = Some("Frequency Spectrum".to_string());
    spectrum_params.freq_range = Some((0.0, 2000.0)); // Focus on 0-2kHz
    let spectrum = audio.plot_magnitude_spectrum(&spectrum_params)?;

    // Create vertical composition
    println!("Creating composite plot...");
    let composite = CompositePlot::new()
        .add_plot(waveform)
        .add_plot(spectrum)
        .layout(CompositeLayout::Vertical)
        .build()?;

    composite.save("outputs/test_composition_advanced.html")?;
    println!("Saved: outputs/test_composition_advanced.html");

    #[cfg(feature = "html_view")]
    {
        println!("Opening composite plot in browser...");
        composite.show()?;
    }

    #[cfg(not(feature = "html_view"))]
    {
        println!("Note: Enable 'html_view' feature to open in browser");
    }

    Ok(())
}
