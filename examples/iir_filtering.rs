#[cfg(not(all(feature = "iir-filtering", feature = "editing", feature = "statistics")))]
pub fn main() {
    eprintln!(
        "error: This example requires the `iir-filtering`, `statistics`, and `editing` features."
    );
    std::process::exit(1);
}
#[cfg(all(feature = "iir-filtering", feature = "editing", feature = "statistics"))]
pub fn main() -> audio_samples::AudioSampleResult<()> {
    use audio_samples::AudioEditing;
    use std::time::Duration;

    use audio_samples::{AudioStatistics, sine_wave};

    use audio_samples::operations::{
        AudioIirFiltering,
        types::{FilterResponse, IirFilterDesign},
    };
    let sample_rate_hz = core::num::NonZeroU32::new(44_100).unwrap();

    // Two-tone signal so filtering is visible.
    let low = sine_wave::<f64>(300.0, Duration::from_secs(1), sample_rate_hz, 0.7);
    let high = sine_wave::<f64>(6_000.0, Duration::from_secs(1), sample_rate_hz, 0.3);
    let mut audio = audio_samples::AudioSamples::mix(
        non_empty_slice::NonEmptySlice::from_slice(&[low, high]).unwrap(),
        None,
    )?;
    let rms_before = audio.rms();
    println!("Input: peak={:.4} rms={:.4}", audio.peak(), audio.rms());

    // Apply a Butterworth low-pass.
    audio.butterworth_lowpass_in_place(core::num::NonZeroUsize::new(4).unwrap(), 1_000.0)?;
    println!("Low-pass: peak={:.4} rms={:.4}", audio.peak(), audio.rms());
    // The 6 kHz tone is far above the 1 kHz cutoff, so it must be attenuated.
    assert!(
        audio.rms() < rms_before,
        "low-pass should reduce energy of the two-tone signal: {} !< {}",
        audio.rms(),
        rms_before
    );

    // Frequency response of the design that was just applied.
    let lowpass =
        IirFilterDesign::butterworth_lowpass(core::num::NonZeroUsize::new(4).unwrap(), 1_000.0);
    let freqs: Vec<f64> = (0..10).map(|i| i as f64 * 500.0).collect();
    let (mag, _phase) = lowpass.frequency_response(&freqs, 44_100.0)?;
    println!("Response @ {:?} Hz => mag {:?}", freqs, mag);
    // Unity gain at DC, exactly -3.01 dB at the cutoff, and monotonically
    // falling above it (Butterworth).
    assert!((mag[0] - 1.0).abs() < 1e-12, "DC gain must be 1");
    assert!(
        (mag[2] - core::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12,
        "|H(1 kHz)| must be 1/sqrt(2)"
    );
    assert!(mag.windows(2).all(|w| w[1] <= w[0]), "monotonic roll-off");

    // Chebyshev example via full design.
    let design = IirFilterDesign::chebyshev_i(
        FilterResponse::HighPass,
        core::num::NonZeroUsize::new(6).unwrap(),
        2000.0,
        1.0,
    );
    audio.apply_iir_filter_in_place(&design)?;
    println!(
        "Chebyshev HP: peak={:.4} rms={:.4}",
        audio.peak(),
        audio.rms()
    );

    Ok(())
}
