//! Plotting functionality demonstration
//!
//! This example shows how to use the plotting capabilities of the audio_samples library
//! to create high-quality visualizations of audio data using the plotters crate.

use audio_samples::{AudioSamples, operations::*};
use ndarray::Array1;
use std::f32::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Audio Samples Plotting Demo");
    println!("===========================");

    // Create a test sine wave signal
    let sample_rate = 44100;
    let duration = 1.0; // 1 second
    let frequency = 440.0; // A4 note
    let samples = (sample_rate as f64 * duration) as usize;

    let sine_wave: Array1<f32> = Array1::from_shape_fn(samples, |i| {
        let t = i as f32 / sample_rate as f32;
        (2.0 * PI * frequency * t).sin() * 0.5
    });

    let audio = AudioSamples::new_mono(sine_wave, sample_rate);

    println!(
        "Created sine wave: {} Hz, {} samples, {} seconds",
        frequency, samples, duration
    );

    // Test basic waveform plotting
    println!("Plotting basic waveform...");
    audio.quick_plot("sine_wave.png")?;
    println!("✓ Saved waveform plot to sine_wave.png");

    // Test plotting with custom title
    println!("Plotting waveform with custom title...");
    audio.quick_plot_with_title("sine_wave_titled.png", "440Hz Sine Wave")?;
    println!("✓ Saved titled waveform plot to sine_wave_titled.png");

    // Create a second signal for comparison (different frequency)
    let frequency2 = 880.0; // A5 note (octave higher)
    let sine_wave2: Array1<f32> = Array1::from_shape_fn(samples, |i| {
        let t = i as f32 / sample_rate as f32;
        (2.0 * PI * frequency2 * t).sin() * 0.3
    });

    let audio2 = AudioSamples::new_mono(sine_wave2, sample_rate);

    // Test comparison plotting
    println!("Plotting comparison between two signals...");
    audio.compare_with(&audio2, "comparison.png")?;
    println!("✓ Saved comparison plot to comparison.png");

    // Test difference plotting
    println!("Plotting difference between signals...");
    audio.plot_diff_with(&audio2, "difference.png")?;
    println!("✓ Saved difference plot to difference.png");

    // Test plotting with custom options
    println!("Plotting with custom options...");
    let mut options = WaveformPlotOptions::default();
    options.title = "Custom Styled Sine Wave".to_string();
    options.figsize = (1200, 800);
    options.wave_color = plotters::prelude::GREEN;
    options.background_color = plotters::prelude::RGBColor(240, 240, 240);
    options.save_path = Some("custom_sine.png".to_string());

    plot_waveform(&audio, options)?;
    println!("✓ Saved custom styled plot to custom_sine.png");

    // Test plotting multiple signals with custom comparison
    println!("Plotting multiple signals comparison...");
    let audio_samples = vec![&audio, &audio2];
    let labels = vec!["440 Hz".to_string(), "880 Hz".to_string()];
    let mut comp_options = ComparisonPlotOptions::default();
    comp_options.title = "Frequency Comparison".to_string();
    comp_options.save_path = Some("frequency_comparison.png".to_string());
    comp_options.colors = vec![plotters::prelude::BLUE, plotters::prelude::RED];

    plot_comparison(&audio_samples, &labels, comp_options)?;
    println!("✓ Saved frequency comparison to frequency_comparison.png");

    println!("\nAll plotting tests completed successfully!");
    println!("Generated files:");
    println!("  - sine_wave.png");
    println!("  - sine_wave_titled.png");
    println!("  - comparison.png");
    println!("  - difference.png");
    println!("  - custom_sine.png");
    println!("  - frequency_comparison.png");

    Ok(())
}
