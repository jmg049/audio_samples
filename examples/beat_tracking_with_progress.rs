//! Example demonstrating beat tracking with progress reporting.
//!
//! This example shows how to use the new progress tracking functionality
//! in the beat detection system to monitor processing progress.

use audio_samples::{AudioSamples, operations::beats::BeatConfig};
use ndarray::Array1;
use std::sync::{Arc, Mutex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a simple test audio signal with clear beats
    let sample_rate = 44100;
    let duration = 5.0; // 5 seconds
    let samples_per_channel = (sample_rate as f64 * duration) as usize;

    // Generate a simple beat pattern at 120 BPM
    let bpm = 120.0;
    let beat_interval = sample_rate as f64 / (bpm / 60.0); // samples per beat

    let mut audio_data = vec![0.0f32; samples_per_channel];

    // Add synthetic beats (kick drum sounds)
    for beat_num in 0..((duration * bpm / 60.0) as usize) {
        let beat_start = (beat_num as f64 * beat_interval) as usize;

        // Add a simple synthetic kick drum (exponentially decaying sine wave)
        for i in 0..1000.min(samples_per_channel - beat_start) {
            let t = i as f64 / sample_rate as f64;
            let amplitude = 0.5 * (-t * 50.0).exp(); // Exponential decay
            let frequency = 60.0; // Low frequency for kick drum
            let sample = amplitude * (2.0 * std::f64::consts::PI * frequency * t).sin();

            if beat_start + i < audio_data.len() {
                audio_data[beat_start + i] += sample as f32;
            }
        }
    }

    // Create AudioSamples from the generated data
    let audio = AudioSamples::new_mono(Array1::from_vec(audio_data), sample_rate);

    println!(
        "Generated {} seconds of test audio at {} BPM",
        duration, bpm
    );
    println!("Starting beat detection with progress tracking...\n");

    // Track progress using a simple counter
    let progress_counter = Arc::new(Mutex::new((0, 0, String::new())));
    let progress_counter_clone = progress_counter.clone();

    let progress_callback = move |current: usize, total: usize, phase: &str| {
        let mut counter = progress_counter_clone.lock().unwrap();
        *counter = (current, total, phase.to_string());

        let percentage = if total > 0 {
            (current * 100) / total
        } else {
            0
        };

        println!("\r[{}%] {} ({}/{})", percentage, phase, current, total);
    };

    // Configure beat detection
    let config = BeatConfig::new(bpm).with_tolerance(0.1); // 10% tolerance

    // Run beat detection with progress tracking
    let beat_tracker = audio.detect_beats_with_progress(
        &config,
        Some(0.5), // log compression
        Some(&progress_callback),
    )?;

    println!("\n\nBeat detection completed!");
    println!("Expected tempo: {} BPM", bpm);
    println!("Detected tempo: {} BPM", beat_tracker.tempo_bpm);
    println!(
        "Number of beats detected: {}",
        beat_tracker.beat_times.len()
    );

    // Show first few detected beat times
    println!("\nFirst 10 detected beat times (seconds):");
    for (i, &beat_time) in beat_tracker.beat_times.iter().take(10).enumerate() {
        println!("  Beat {}: {:.3}s", i + 1, beat_time);
    }

    // Calculate expected vs actual beat times for accuracy assessment
    let expected_beat_times: Vec<f64> = (0..((duration * bpm / 60.0) as usize))
        .map(|i| i as f64 * 60.0 / bpm)
        .collect();

    println!("\nAccuracy assessment:");
    println!(
        "Expected {} beats, detected {} beats",
        expected_beat_times.len(),
        beat_tracker.beat_times.len()
    );

    // Find average timing error for detected beats
    if !beat_tracker.beat_times.is_empty() && !expected_beat_times.is_empty() {
        let mut total_error = 0.0;
        let mut matched_beats = 0;

        for &detected_time in &beat_tracker.beat_times {
            // Find the closest expected beat time
            if let Some(&closest_expected) = expected_beat_times.iter().min_by(|&&a, &&b| {
                (a - detected_time)
                    .abs()
                    .partial_cmp(&(b - detected_time).abs())
                    .unwrap()
            }) {
                let error = (detected_time - closest_expected).abs();
                if error < 0.1 {
                    // Only count if within 100ms
                    total_error += error;
                    matched_beats += 1;
                }
            }
        }

        if matched_beats > 0 {
            let avg_error = total_error / matched_beats as f64;
            println!(
                "Average timing error: {:.3}s ({} matched beats)",
                avg_error, matched_beats
            );
        }
    }

    Ok(())
}
