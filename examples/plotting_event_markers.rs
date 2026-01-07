//! Example demonstrating onset and beat marker overlays on waveform plots
//!
//! This example shows how to:
//! 1. Detect onsets (note attacks, percussion hits)
//! 2. Detect beats (tempo tracking)
//! 3. Visualize them as vertical markers on waveform plots
//!
//! Run with:
//! ```bash
//! cargo run --example plotting_event_markers --features plotting,onset-detection,beat-tracking,statistics
//! ```

#[cfg(all(
    feature = "plotting",
    feature = "onset-detection",
    feature = "beat-tracking",
    feature = "statistics"
))]
fn create_percussive_audio() -> audio_samples::AudioSamples<'static, f32> {
    // Generate a percussive pattern: kick drum on beats 1 and 3, snare on 2 and 4
    // 120 BPM = 0.5 seconds per beat

    use std::num::NonZero;
    let sample_rate = audio_samples::sample_rate!(44100);
    let duration = 4.0; // 4 seconds = 8 beats at 120 BPM
    let n_samples = (sample_rate.get() as f64 * duration) as usize;
    let n_samples = NonZero::new(n_samples).unwrap();
    let beat_interval = 0.5; // 120 BPM

    let mut samples = audio_samples::AudioSamples::zeros_mono(n_samples, sample_rate);
    let sample_rate_hz = samples.sample_rate_hz();
    // Add kick drum hits (low frequency transients) on beats 1, 3, 5, 7
    for beat in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] {
        let start_sample = (beat * beat_interval * sample_rate_hz) as usize;

        // Kick drum: short low-frequency burst with exponential decay
        for i in 0..2000 {
            if start_sample + i >= n_samples.get() {
                break;
            }
            let t = i as f64 / sample_rate_hz;
            let decay = (-t * 50.0).exp();
            let kick = (2.0 * std::f64::consts::PI * 60.0 * t).sin(); // 60 Hz
            samples[start_sample + i] += (kick * decay * 0.8) as f32;
        }
    }

    // Add snare hits (high frequency transients) on beats 2, 4, 6, 8
    for beat in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5] {
        let start_sample = (beat * beat_interval * sample_rate_hz) as usize;

        // Snare: noise burst with exponential decay
        for i in 0..1500 {
            if start_sample + i >= n_samples.get() {
                break;
            }
            let t = i as f64 / sample_rate_hz;
            let decay = (-t * 80.0).exp();
            // Mix of sine waves for brightness
            let snare = (2.0 * std::f64::consts::PI * 200.0 * t).sin()
                + (2.0 * std::f64::consts::PI * 350.0 * t).sin() * 0.5;
            samples[start_sample + i] += (snare * decay * 0.6) as f32;
        }
    }
    samples
}

#[cfg(not(all(
    feature = "plotting",
    feature = "onset-detection",
    feature = "beat-tracking",
    feature = "statistics"
)))]
fn main() {
    eprintln!(
        "error: This example requires the `plotting`, `onset-detection`, `beat-tracking`, and `statistics` features."
    );
    std::process::exit(1);
}

#[cfg(all(
    feature = "plotting",
    feature = "onset-detection",
    feature = "beat-tracking",
    feature = "statistics"
))]
fn main() -> audio_samples::AudioSampleResult<()> {
    use audio_samples::operations::beat::BeatTrackingConfig;
    use audio_samples::operations::onset::OnsetDetectionConfig;
    use audio_samples::operations::plotting::waveform::WaveformPlotParams;
    use audio_samples::operations::{AudioBeatTracking, AudioPlotting, PlotUtils};
    use audio_samples::{AudioOnsetDetection, AudioSamples};
    println!("=== Event Markers Example ===\n");

    let audio: &'static AudioSamples<'static, f32> = Box::leak(Box::new(create_percussive_audio()));
    println!("Generated percussive audio: 4 seconds at 120 BPM");

    // Example 1: Onset Detection with markers
    println!("\n--- Onset Detection ---");
    let onset_config = OnsetDetectionConfig::percussive();
    let onsets = audio.detect_onsets(&onset_config)?;

    println!("Detected {} onsets:", onsets.len());
    for (idx, onset_time) in onsets.iter().enumerate() {
        println!("  Onset {}: {:.3}s", idx + 1, onset_time);
    }

    let plot_onsets = audio
        .plot_waveform(&WaveformPlotParams::default())?
        .add_onset_markers(onsets.clone(), Some("red"), false);

    plot_onsets.save("output/event_markers_onsets.html")?;
    println!("\nSaved onset markers to: output/event_markers_onsets.html");

    // Example 2: Onset Detection with time labels
    let plot_onsets_labeled = audio
        .plot_waveform(&WaveformPlotParams::default())?
        .add_onset_markers(onsets, Some("red"), true);

    plot_onsets_labeled.save("output/event_markers_onsets_labeled.html")?;
    println!("Saved labeled onset markers to: output/event_markers_onsets_labeled.html");

    // Example 3: Beat Tracking with markers
    println!("\n--- Beat Tracking ---");
    // Beat tracking uses onset detection internally
    let beat_config = BeatTrackingConfig::new(
        120.0,     // Expected tempo in BPM (can be rough estimate)
        Some(0.1), // Tolerance for tempo matching
        onset_config.clone(),
    );
    let beat_data = audio.detect_beats(&beat_config)?;

    println!("Detected tempo: {:.1} BPM", beat_data.tempo_bpm);
    println!("Detected {} beats:", beat_data.beat_times.len());
    for (idx, beat_time) in beat_data.beat_times.iter().take(8).enumerate() {
        println!("  Beat {}: {:.3}s", idx + 1, beat_time);
    }

    let plot_beats = audio
        .plot_waveform(&WaveformPlotParams::default())?
        .add_beat_markers(beat_data.beat_times.clone(), Some("blue"), false);

    plot_beats.save("output/event_markers_beats.html")?;
    println!("\nSaved beat markers to: output/event_markers_beats.html");

    // Example 4: Beat tracking with beat numbers
    let plot_beats_numbered = audio
        .plot_waveform(&WaveformPlotParams::default())?
        .add_beat_markers(beat_data.beat_times.clone(), Some("blue"), true);

    plot_beats_numbered.save("output/event_markers_beats_numbered.html")?;
    println!("Saved numbered beat markers to: output/event_markers_beats_numbered.html");

    // Example 5: Combined - both onsets and beats
    println!("\n--- Combined Visualization ---");

    // Redetect since we moved onsets earlier
    let onsets_fresh = audio.detect_onsets(&onset_config)?;

    let plot_combined = audio
        .plot_waveform(&WaveformPlotParams::default())?
        .add_onset_markers(onsets_fresh, Some("rgba(255,0,0,0.5)"), false)
        .add_beat_markers(beat_data.beat_times, Some("rgba(0,0,255,0.7)"), true);

    plot_combined.save("output/event_markers_combined.html")?;
    println!("Saved combined markers to: output/event_markers_combined.html");
    println!("  Red = Onsets (note attacks)");
    println!("  Blue = Beats (tempo grid)");

    Ok(())
}
