//! Plotting functionality demonstration for audio_samples
//!
//! This example demonstrates the various plotting capabilities available
//! in the audio_samples crate for visualizing audio data and analysis results.
//!
//! Run with: cargo run --example plotting --features full
//!
//! Available demos (pass as command line argument):
//!   waveform       - Basic waveform visualization
//!   spectrum       - Power spectrum analysis
//!   spectrogram    - Time-frequency spectrogram
//!   multi-channel  - Per-channel plotting for stereo audio
//!   fft-analysis   - FFT-based plots (complex spectrum, phase, peaks)
//!   detection      - Onset, beat, and pitch detection overlays
//!   composed       - Multiple plot elements combined
//!   all            - Run all demos sequentially

use std::time::Duration;

use std::path::Path;

#[cfg(feature = "random-generation")]
use audio_samples::utils::generation::exponential_bursts;
use audio_samples::{
    AudioProcessing, AudioSampleResult, AudioSamples, NormalizationMethod,
    operations::{
        AudioPlotBuilders, ColorPalette, LayoutConfig, LineStyle, MarkerStyle, PlotComposer,
        Plotting, WaveformPlotConfig, plotting::PeakDetectionConfig,
    },
    to_precision,
    utils::generation::{ToneComponent, am_signal, chirp, compound_tone, sine_wave},
};
use ndarray::Array2;
use std::env;
use std::num::NonZeroU32;

/// Output directory for saved plots
const OUTPUT_DIR: &str = "target/plots";

/// Save a plot composer to both PNG and HTML formats
fn save_plot<F: audio_samples::RealFloat>(
    composer: &audio_samples::operations::PlotComposer<F>,
    name: &str,
) -> AudioSampleResult<()> {
    let output_dir = Path::new(OUTPUT_DIR);

    // Save HTML (always available)
    let html_path = output_dir.join(format!("{}.html", name));
    composer.render_to_html(&html_path, true)?;
    println!("  Saved: {}", html_path.display());

    // Save PNG (requires static-plots feature)
    #[cfg(feature = "static-plots")]
    {
        let png_path = output_dir.join(format!("{}.png", name));
        match composer.render_to_png(&png_path, 1600, 1200) {
            Ok(()) => println!("  Saved: {}", png_path.display()),
            Err(e) => eprintln!("  Warning: PNG export failed: {} (HTML still saved)", e),
        }
    }

    #[cfg(not(feature = "static-plots"))]
    println!("  (PNG export requires 'static-plots' feature)");

    Ok(())
}

fn main() -> AudioSampleResult<()> {
    let args: Vec<String> = env::args().collect();

    // Check for --no-show flag
    let no_show = args.iter().any(|a| a == "--no-show" || a == "--save-only");
    let show = !no_show;

    // Get demo name (skip flags)
    let demo = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .map(|s| s.as_str())
        .unwrap_or("all");

    println!("Audio Samples Plotting Demo");
    println!("===========================");
    println!("Output directory: {}", OUTPUT_DIR);
    if no_show {
        println!("Mode: save-only (plots will not be displayed)");
    }
    println!();

    match demo {
        "waveform" => demo_waveform(show)?,
        "spectrum" => demo_spectrum(show)?,
        "spectrogram" => demo_spectrogram(show)?,
        "multi-channel" => demo_multi_channel(show)?,
        "fft-analysis" => demo_fft_analysis(show)?,
        "detection" => demo_detection(show)?,
        "composed" => demo_composed(show)?,
        "all" => {
            demo_waveform(show)?;
            demo_spectrum(show)?;
            demo_spectrogram(show)?;
            demo_multi_channel(show)?;
            demo_fft_analysis(show)?;
            // demo_detection(show)?;
            demo_composed(show)?;
        }
        _ => {
            println!("Unknown demo: {}", demo);
            println!("\nAvailable demos:");
            println!("  waveform       - Basic waveform visualization");
            println!("  spectrum       - Power spectrum analysis");
            println!("  spectrogram    - Time-frequency spectrogram");
            println!("  multi-channel  - Per-channel plotting for stereo audio");
            println!("  fft-analysis   - FFT-based plots (complex spectrum, phase, peaks)");
            println!("  detection      - Onset, beat, and pitch detection overlays");
            println!("  composed       - Multiple plot elements combined");
            println!("  all            - Run all demos sequentially");
            println!("\nFlags:");
            println!("  --no-show      - Save plots without displaying them");
            println!("  --save-only    - Alias for --no-show");
        }
    }

    Ok(())
}

// ============================================================================
// Signal Generation Helpers (using crate utilities)
// ============================================================================

/// Generate a test signal with 440 Hz fundamental and harmonics using AM modulation
fn generate_test_signal(duration_secs: f64, sample_rate: u32) -> AudioSamples<'static, f64> {
    // Use AM signal: 440 Hz carrier with 2 Hz modulation for visual interest
    am_signal::<f64, f64>(
        440.0, // carrier frequency
        2.0,   // modulator frequency
        0.5,   // modulation depth
        Duration::from_secs_f64(duration_secs),
        sample_rate,
        0.8, // amplitude
    )
}

/// Generate a multi-tone test signal with harmonics using compound_tone
fn generate_harmonic_signal(duration_secs: f64, sample_rate: u32) -> AudioSamples<'static, f64> {
    let components = [
        ToneComponent::new(440.0, 1.0),   // fundamental
        ToneComponent::new(880.0, 0.5),   // 2nd harmonic
        ToneComponent::new(1320.0, 0.25), // 3rd harmonic
    ];
    compound_tone::<f64, f64>(
        &components,
        Duration::from_secs_f64(duration_secs),
        sample_rate,
    )
}

/// Generate a stereo test signal with different content per channel
/// Uses compound_tone for each channel, then combines them
fn generate_stereo_signal(duration_secs: f64, sample_rate: u32) -> AudioSamples<'static, f64> {
    // Left channel: lower frequency content (220 Hz + 330 Hz)
    let left_components = [
        ToneComponent::new(220.0, 0.6),
        ToneComponent::new(330.0, 0.3),
    ];
    let left = compound_tone::<f64, f64>(
        &left_components,
        Duration::from_secs_f64(duration_secs),
        sample_rate,
    );

    // Right channel: higher frequency content (880 Hz + 1100 Hz)
    let right_components = [
        ToneComponent::new(880.0, 0.6),
        ToneComponent::new(1100.0, 0.3),
    ];
    let right = compound_tone::<f64, f64>(
        &right_components,
        Duration::from_secs_f64(duration_secs),
        sample_rate,
    );

    // Combine into stereo (channels x samples)
    let num_samples = left.samples_per_channel();
    let mut data = Array2::zeros((2, num_samples));

    let left_data = left.as_mono().expect("left is mono");
    let right_data = right.as_mono().expect("right is mono");

    for i in 0..num_samples {
        data[[0, i]] = left_data[i];
        data[[1, i]] = right_data[i];
    }

    let mut audio = AudioSamples::new_multi_channel(data, NonZeroU32::new(sample_rate).unwrap());
    audio
        .normalize(-1.0, 1.0, NormalizationMethod::MinMax)
        .unwrap();
    audio
}

/// Generate a signal with transients for onset detection
/// Uses the crate's exponential_bursts function
#[cfg(feature = "random-generation")]
fn generate_percussive_signal(duration_secs: f64, sample_rate: u32) -> AudioSamples<'static, f64> {
    exponential_bursts::<f64, f64>(
        2.0,  // burst_rate: 2 bursts per second
        30.0, // decay_rate: fast decay
        Duration::from_secs_f64(duration_secs),
        sample_rate,
        0.8, // amplitude
    )
}

/// Fallback percussive signal when random-generation is not available
/// Uses a simple sine burst pattern
#[cfg(not(feature = "random-generation"))]
fn generate_percussive_signal(duration_secs: f64, sample_rate: u32) -> AudioSamples<'static, f64> {
    // Without random-generation, use a deterministic burst pattern
    am_signal::<f64, f64>(
        200.0, // carrier frequency (low tone)
        2.0,   // burst rate
        0.95,  // high modulation depth for burst effect
        Duration::from_secs_f64(duration_secs),
        sample_rate,
        0.8,
    )
}

// ============================================================================
// Demo Functions
// ============================================================================

fn demo_waveform(show: bool) -> AudioSampleResult<()> {
    println!("Demo: Waveform Visualization");
    println!("----------------------------");
    println!("Displaying a 440 Hz sine wave with harmonics.\n");

    // Use harmonic signal for cleaner waveform visualization
    let audio = generate_harmonic_signal(0.1, 44100); // 100ms for clear visualization

    // Simple waveform plot using AudioPlotBuilders trait
    let waveform = audio.waveform_plot::<f64>(None)?;

    let composer = PlotComposer::<f64>::new()
        .add_element(waveform)
        .with_title("Waveform - 440 Hz with Harmonics");

    // Save to files
    save_plot(&composer, "waveform")?;

    // Display (blocking by default)
    if show {
        composer.show(true)?;
    }

    println!("Waveform demo complete.\n");
    Ok(())
}

fn demo_spectrum(show: bool) -> AudioSampleResult<()> {
    println!("Demo: Power Spectrum Analysis");
    println!("-----------------------------");
    println!("Analyzing frequency content of a multi-tone signal.\n");

    // Use harmonic signal - should show clear peaks at 440, 880, 1320 Hz
    let audio = generate_harmonic_signal(1.0, 44100);

    // Create power spectrum plot with custom parameters
    let spectrum = audio.power_spectrum_plot::<f64>(
        Some(4096),          // n_fft - higher for better frequency resolution
        None,                // window - use default (Hanning)
        Some(true),          // dB scale
        Some((0.0, 2000.0)), // frequency range to display
        Some(LineStyle {
            color: "#2ca02c".to_string(),
            width: to_precision(2.0),
            style: audio_samples::operations::LineStyleType::Solid,
        }),
    )?;

    let composer = PlotComposer::<f64>::new()
        .add_element(spectrum)
        .with_title("Power Spectrum - Frequency Content Analysis");

    // Save to files
    save_plot(&composer, "spectrum")?;

    if show {
        composer.show(true)?;
    }

    println!("Spectrum demo complete.\n");
    Ok(())
}

fn demo_spectrogram(show: bool) -> AudioSampleResult<()> {
    println!("Demo: Spectrogram (Time-Frequency Analysis)");
    println!("-------------------------------------------");
    println!("Visualizing how frequency content changes over time.\n");

    // Generate a chirp signal (frequency sweep) using the crate's chirp function
    let sample_rate = 44100u32;
    let duration = Duration::from_secs(2);

    let audio = chirp::<f64, f64>(
        100.0,  // start frequency
        2000.0, // end frequency
        duration,
        sample_rate,
        0.8, // amplitude
    );

    // Create spectrogram with custom settings
    let spectrogram = audio.spectrogram_plot::<f64>(
        Some(2048), // n_fft
        Some(512),  // hop_length
        None,       // window
        Some(ColorPalette::Viridis),
        Some((-80.0, 0.0)), // dB range
        Some(false),        // log frequency scale
    )?;

    let composer = PlotComposer::<f64>::new()
        .add_element(spectrogram)
        .with_title("Spectrogram - Linear Frequency Chirp (100 Hz to 2000 Hz)");

    // Save to files
    save_plot(&composer, "spectrogram")?;

    if show {
        composer.show(true)?;
    }

    println!("Spectrogram demo complete.\n");
    Ok(())
}

fn demo_multi_channel(show: bool) -> AudioSampleResult<()> {
    println!("Demo: Multi-Channel (Stereo) Plotting");
    println!("-------------------------------------");
    println!("Displaying left and right channels with different frequency content.\n");

    let audio = generate_stereo_signal(0.05, 44100); // 50ms

    // Use the new Plotting trait for per-channel visualization
    let config = WaveformPlotConfig {
        style: LineStyle::default(),
        shared_time_axis: true,
    };

    let composer = audio
        .plot_waveform::<f64>(Some(config))?
        .with_title("Stereo Waveform - Left (220 Hz) vs Right (880 Hz)");

    // Save to files
    save_plot(&composer, "multi_channel")?;

    if show {
        composer.show(true)?;
    }

    println!("Multi-channel demo complete.\n");
    Ok(())
}

fn demo_fft_analysis(show: bool) -> AudioSampleResult<()> {
    println!("Demo: FFT-Based Analysis");
    println!("------------------------");
    println!("Complex spectrum, phase spectrum, and peak detection.\n");

    let audio = generate_test_signal(1.0, 44100);

    // Create multiple FFT-based plots
    let complex_spectrum = audio.complex_spectrum_plot::<f64>(None)?;
    let phase_spectrum = audio.phase_spectrum_plot::<f64>(None, None)?;

    // Peak frequency detection with custom config
    let peak_config = PeakDetectionConfig {
        min_height: Some(-40.0),
        min_prominence: Some(6.0),
        min_distance: None,
        max_peaks: Some(10),
        frequency_range: Some((100.0, 3000.0)),
    };
    let peak_plot = audio.peak_frequencies_plot::<f64>(
        Some(peak_config),
        None,
        Some(MarkerStyle {
            color: "#d62728".to_string(),
            size: to_precision(12.0),
            shape: audio_samples::operations::MarkerShape::Circle,
            fill: true,
        }),
    )?;

    // Compose all three plots in a vertical stack
    let composer = PlotComposer::<f64>::new()
        .with_layout(LayoutConfig::VerticalStack)
        .add_element(complex_spectrum)
        .add_element(phase_spectrum)
        .add_element(peak_plot)
        .with_title("FFT Analysis Suite");

    // Save to files
    save_plot(&composer, "fft_analysis")?;

    if show {
        composer.show(true)?;
    }

    println!("FFT analysis demo complete.\n");
    Ok(())
}

fn demo_detection(show: bool) -> AudioSampleResult<()> {
    println!("Demo: Detection Algorithms");
    println!("--------------------------");
    println!("Onset detection, beat tracking, and pitch contour.\n");
    println!("Note: Detection quality depends on enabled features.\n");

    let audio = generate_percussive_signal(3.0, 44100);

    // Create waveform as base
    let waveform = audio.waveform_plot::<f64>(Some(LineStyle {
        color: "#1f77b4".to_string(),
        width: to_precision(1.0),
        style: audio_samples::operations::LineStyleType::Solid,
    }))?;

    // Create onset markers overlay
    let onsets = audio.onset_markers::<f64>(
        Some(MarkerStyle {
            color: "#d62728".to_string(),
            size: to_precision(10.0),
            shape: audio_samples::operations::MarkerShape::Triangle,
            fill: true,
        }),
        None,
        Some(true),
        Some(0.1),
    )?;

    // Create beat markers overlay
    let beats = audio.beat_markers::<f64>(
        Some(MarkerStyle {
            color: "#2ca02c".to_string(),
            size: to_precision(12.0),
            shape: audio_samples::operations::MarkerShape::Diamond,
            fill: true,
        }),
        None,
        Some(true),
    )?;

    // Compose waveform with detection overlays
    let composer = PlotComposer::<f64>::new()
        .add_element(waveform)
        .add_element(onsets)
        .add_element(beats)
        .with_title("Waveform with Onset and Beat Detection");

    // Save to files
    save_plot(&composer, "detection_onset_beat")?;

    if show {
        composer.show(true)?;
    }

    // Also show pitch contour for a tonal signal
    println!("Now showing pitch contour for a tonal signal...\n");

    // Use a pure sine wave for cleaner pitch detection
    let tonal_audio = sine_wave::<f64, f64>(440.0, Duration::from_secs(1), 44100, 0.8);
    let pitch = tonal_audio.pitch_contour::<f64>(
        Some(LineStyle {
            color: "#ff7f0e".to_string(),
            width: to_precision(2.0),
            style: audio_samples::operations::LineStyleType::Solid,
        }),
        Some(false),
        Some((400.0, 500.0)), // Narrow range around 440 Hz
        None,
    )?;

    let pitch_composer = PlotComposer::<f64>::new()
        .add_element(pitch)
        .with_title("Pitch Contour - 440 Hz Fundamental");

    // Save to files
    save_plot(&pitch_composer, "detection_pitch")?;

    if show {
        pitch_composer.show(true)?;
    }

    println!("Detection demo complete.\n");
    Ok(())
}

fn demo_composed(show: bool) -> AudioSampleResult<()> {
    println!("Demo: Composed Multi-Plot Layout");
    println!("--------------------------------");
    println!("Combining multiple visualizations in a single view.\n");

    // Use AM signal for interesting time-varying content in the dashboard
    let audio = generate_test_signal(1.0, 44100);

    // Create various plot elements
    let waveform = audio.waveform_plot::<f64>(None)?;
    let spectrum = audio.power_spectrum_plot::<f64>(
        Some(4096),
        None,
        Some(true),
        Some((0.0, 3000.0)),
        None,
    )?;
    let spectrogram = audio.spectrogram_plot::<f64>(
        Some(1024),
        Some(256),
        None,
        Some(ColorPalette::Plasma),
        None,
        None,
    )?;

    // Compose in a grid layout
    let composer = PlotComposer::<f64>::new()
        .with_layout(LayoutConfig::Grid { rows: 2, cols: 2 })
        .add_element(waveform)
        .add_element(spectrum)
        .add_element(spectrogram)
        .with_title("Audio Analysis Dashboard");

    // Save to files
    save_plot(&composer, "composed_dashboard")?;

    if show {
        composer.show(true)?;
    }

    println!("Composed plot demo complete.\n");
    Ok(())
}
