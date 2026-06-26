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
    audio.apply_parametric_eq_in_place(&eq)?;
    println!(
        "EQ applied: peak={:.4} rms={:.4}",
        audio.peak(),
        audio.rms()
    );

    // Single-band convenience.
    audio.apply_eq_band_in_place(&EqBand::peak(500.0, -2.0, 1.0))?;
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

    // --- Self-verification -------------------------------------------------
    // freqs = [60, 120, 250, 500, 1000, 2000, 4000, 8000, 12000].
    // The 1 kHz band has a +4 dB peak boost; 120 Hz has a -3 dB low-shelf cut.
    // So the response at 1 kHz (index 4) must exceed the response at 120 Hz (index 1).
    assert_eq!(mag.len(), freqs.len(), "one magnitude per probed frequency");
    assert!(
        mag[4] > mag[1],
        "boosted 1 kHz band should exceed cut 120 Hz band: {} !> {}",
        mag[4],
        mag[1]
    );
    assert!(audio.rms() > 0.0, "EQ output must remain non-silent");

    Ok(())
}
