#[cfg(not(feature = "parametric-eq"))]
pub fn main() {
    eprintln!("error: This example requires the `parametric-eq` feature.");
    std::process::exit(1);
}

#[cfg(feature = "parametric-eq")]
pub fn main() -> audio_samples::AudioSampleResult<()> {
    use std::time::Duration;

    use audio_samples::{AudioStatistics, ToneComponent, compound_tone};

    use audio_samples::operations::{
        AudioParametricEq,
        types::{EqBand, ParametricEq},
    };
    let sample_rate_hz = core::num::NonZeroU32::new(44_100).unwrap();

    // A harmonic tone (rich spectrum) makes EQ effects visible.
    let components = [
        ToneComponent::new(220.0, 1.0),
        ToneComponent::new(440.0, 0.6),
        ToneComponent::new(880.0, 0.3),
        ToneComponent::new(1760.0, 0.15),
    ];
    let components_slice: &[_] = &components;
    let non_empty = non_empty_slice::NonEmptySlice::from_slice(components_slice).unwrap();
    let mut audio = compound_tone::<f64>(&non_empty, Duration::from_secs(1), sample_rate_hz);
    println!("Input: peak={:.4} rms={:.4}", audio.peak(), audio.rms());

    // Build an EQ: low shelf cut, mid peak boost, high shelf boost.
    let eq = ParametricEq::three_band(120.0, -3.0, 1_000.0, 4.0, 1.2, 8_000.0, 2.0);
    audio.apply_parametric_eq(&eq)?;
    println!(
        "EQ applied: peak={:.4} rms={:.4}",
        audio.peak(),
        audio.rms()
    );

    // Single-band convenience.
    audio.apply_eq_band(&EqBand::peak(500.0, -2.0, 1.0))?;
    println!(
        "+ Notch 500Hz: peak={:.4} rms={:.4}",
        audio.peak(),
        audio.rms()
    );

    // Response for inspection.
    let freqs = [
        60.0, 120.0, 250.0, 500.0, 1_000.0, 2_000.0, 4_000.0, 8_000.0, 12_000.0,
    ];
    let (mag, _phase) = audio.eq_frequency_response(&eq, &freqs)?;
    println!("EQ magnitude response @ {:?} Hz => {:?}", freqs, mag);

    Ok(())
}
