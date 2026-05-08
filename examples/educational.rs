//! Demonstrates the `educational` feature: step-by-step operation explanations
//! rendered as a polished HTML document.
//!
//! Run with:
//!   cargo run --example educational --features educational,processing
//!
//! Add `--features plotting` to include before/after waveform panels in each card.

use audio_samples::educational;
use audio_samples::operations::types::NormalizationConfig;
use audio_samples::utils::generation::sine_wave;
use audio_samples::{
    AudioProcessingExt, AudioSamples, AudioStatistics, ExplainMode, Explainable, sample_rate,
};
use ndarray::Array1;
use std::time::Duration;

fn main() {
    // Generate a 440 Hz sine wave and add a DC bias of 0.35 so the
    // remove_dc_offset step produces a clearly visible waveform shift
    // (asymmetric → centred) rather than a no-op on a perfectly balanced signal.
    let sine = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(2000), 0.8);
    let dc_bias = 0.35_f32;
    let biased: Vec<f32> = sine
        .as_slice()
        .expect("sine_wave returns contiguous mono audio")
        .iter()
        .map(|&s| s + dc_bias)
        .collect();
    let audio =
        AudioSamples::<'static, f32>::new_mono(Array1::from_vec(biased), sample_rate!(2000))
            .expect("valid mono buffer");

    let (result, explanations) = audio
        .explaining(ExplainMode::Both)
        .remove_dc_offset()
        .normalize(NormalizationConfig::peak(1.0))
        .scale(0.5)
        .clip(-0.3_f32, 0.3_f32)
        .explain();

    println!("Operations complete. Final peak: {:.4}", result.peak());
    println!("Opening explanation document in browser…");

    educational::open_explanation_document(&explanations, "Audio Processing Walkthrough")
        .expect("failed to open explanation document");
}
