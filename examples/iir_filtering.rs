#[cfg(feature = "core-ops")]
use std::time::Duration;

#[cfg(feature = "core-ops")]
use audio_samples::{AudioSampleResult, AudioStatistics, sine_wave};

#[cfg(feature = "core-ops")]
use audio_samples::operations::{
    AudioIirFiltering,
    types::{FilterResponse, IirFilterDesign},
};

#[cfg(not(feature = "core-ops"))]
fn main() {
    eprintln!("This example requires the 'core-ops' feature.");
}

#[cfg(feature = "core-ops")]
pub fn main() -> AudioSampleResult<()> {
    use audio_samples::AudioEditing;

    let sample_rate_hz = 44_100u32;
    let sr = sample_rate_hz as f64;

    // Two-tone signal so filtering is visible.
    let low = sine_wave::<f64, f64>(300.0, Duration::from_secs(1), sample_rate_hz, 0.7);
    let high = sine_wave::<f64, f64>(6_000.0, Duration::from_secs(1), sample_rate_hz, 0.3);
    let mut audio = audio_samples::AudioSamples::mix::<f64>(&[low, high], None)?;
    println!(
        "Input: peak={:.4} rms={:.4}",
        audio.peak(),
        audio.rms::<f64>()
    );

    // Apply a Butterworth low-pass.
    audio.butterworth_lowpass(4, 1_000.0, sr)?;
    println!(
        "Low-pass: peak={:.4} rms={:.4}",
        audio.peak(),
        audio.rms::<f64>()
    );

    // Frequency response (after filter is configured).
    let freqs: Vec<f64> = (0..10).map(|i| i as f64 * 500.0).collect();
    let (mag, _phase) = audio.frequency_response(&freqs, sr)?;
    println!("Response @ {:?} Hz => mag {:?}", freqs, mag);

    // Chebyshev example via full design.
    let design = IirFilterDesign::chebyshev_i(FilterResponse::HighPass, 6, 2000.0, 1.0);
    audio.apply_iir_filter(&design, sr)?;
    println!(
        "Chebyshev HP: peak={:.4} rms={:.4}",
        audio.peak(),
        audio.rms::<f64>()
    );

    Ok(())
}
