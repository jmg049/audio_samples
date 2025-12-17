#[cfg(feature = "core-ops")]
use std::time::Duration;

#[cfg(feature = "core-ops")]
use audio_samples::{AudioSampleResult, AudioStatistics, ToneComponent, compound_tone};

#[cfg(feature = "core-ops")]
use audio_samples::operations::{
    AudioParametricEq,
    types::{EqBand, ParametricEq},
};

#[cfg(not(feature = "core-ops"))]
fn main() {
    eprintln!("This example requires the 'core-ops' feature.");
}

#[cfg(feature = "core-ops")]
pub fn main() -> AudioSampleResult<()> {
    let sample_rate_hz = 44_100u32;
    let sr = sample_rate_hz as f64;

    // A harmonic tone (rich spectrum) makes EQ effects visible.
    let components = [
        ToneComponent::new(220.0, 1.0),
        ToneComponent::new(440.0, 0.6),
        ToneComponent::new(880.0, 0.3),
        ToneComponent::new(1760.0, 0.15),
    ];
    let mut audio = compound_tone::<f64, f64>(&components, Duration::from_secs(1), sample_rate_hz);
    println!(
        "Input: peak={:.4} rms={:.4}",
        audio.peak(),
        audio.rms::<f64>()
    );

    // Build an EQ: low shelf cut, mid peak boost, high shelf boost.
    let eq = ParametricEq::three_band(120.0, -3.0, 1_000.0, 4.0, 1.2, 8_000.0, 2.0);
    audio.apply_parametric_eq(&eq, sr)?;
    println!(
        "EQ applied: peak={:.4} rms={:.4}",
        audio.peak(),
        audio.rms::<f64>()
    );

    // Single-band convenience.
    audio.apply_eq_band(&EqBand::peak(500.0, -2.0, 1.0), sr)?;
    println!(
        "+ Notch 500Hz: peak={:.4} rms={:.4}",
        audio.peak(),
        audio.rms::<f64>()
    );

    // Response for inspection.
    let freqs = [
        60.0, 120.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0, 12_000.0,
    ];
    let (mag, _phase) = audio.eq_frequency_response(&eq, &freqs, sr)?;
    println!("EQ magnitude response @ {:?} Hz => {:?}", freqs, mag);

    Ok(())
}
