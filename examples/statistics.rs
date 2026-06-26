#[cfg(not(feature = "statistics"))]
fn main() {
    eprintln!("error: This example requires the `statistics` feature.");
    std::process::exit(1);
}

#[cfg(feature = "statistics")]
pub fn main() -> audio_samples::AudioSampleResult<()> {
    use std::time::Duration;

    use audio_samples::{AudioStatistics, ToneComponent};
    let components = [
        ToneComponent::new(440.0, 1.0),   // fundamental
        ToneComponent::new(880.0, 0.5),   // 2nd harmonic
        ToneComponent::new(1320.0, 0.25), // 3rd harmonic
    ];
    let audio = audio_samples::compound_tone::<f32>(
        non_empty_slice::NonEmptySlice::from_slice(&components).unwrap(),
        Duration::from_secs(2),
        core::num::NonZeroU32::new(44100).unwrap(),
    );

    let (num_channels, samples_per_channel, duration_seconds, sample_rate) = audio.info();
    println!("Compound Tone wave info:");
    println!("|-Number of channels: {}", num_channels);
    println!("|-Samples per channel: {}", samples_per_channel);
    println!("|-Duration (seconds): {:.2}", duration_seconds);
    println!("|-Sample rate: {} Hz", sample_rate);

    println!("-----\n");

    println!("== Statistics ==");
    println!("Mean: {}", audio.mean());
    println!("RMS: {}", audio.rms());
    println!("Min: {}", audio.min_sample());
    println!("Max: {}", audio.max_sample());
    println!("Variance: {}", audio.variance());
    println!("Standard Deviation: {}", audio.std_dev());
    println!("Peak: {}", audio.peak());
    println!("Zero Crossings: {}", audio.zero_crossings());
    println!("Zero Crossing Rate: {}", audio.zero_crossing_rate());
    #[cfg(feature = "transforms")]
    {
        if let Some(ac) = audio.autocorrelation(audio_samples::nzu!(1)) {
            println!("Autocorrelation (lag 1): {:?}", ac);
        }
        use audio_samples::operations::types::ChannelReduction;
        println!(
            "Spectral-centroid: {}",
            audio.spectral_centroid(ChannelReduction::Error)?
        );
        println!(
            "Spectral-rolloff (0.85): {}",
            audio.spectral_rolloff(0.85, ChannelReduction::Error)?
        );
    }

    // Create a new 880 Hz tone by shifting the frequency of the original audio
    let components = [
        ToneComponent::new(880.0, 1.0),   // fundamental
        ToneComponent::new(1760.0, 0.5),  // 2nd harmonic
        ToneComponent::new(2640.0, 0.25), // 3rd harmonic
    ];
    let other_audio = audio_samples::compound_tone::<f32>(
        non_empty_slice::NonEmptySlice::from_slice(&components).unwrap(),
        Duration::from_secs(2),
        core::num::NonZeroU32::new(44100).unwrap(),
    );
    let (num_channels, samples_per_channel, duration_seconds, sample_rate) = other_audio.info();
    println!("\nCompound Tone wave info:");
    println!("|-Number of channels: {}", num_channels);
    println!("|-Samples per channel: {}", samples_per_channel);
    println!("|-Duration (seconds): {:.2}", duration_seconds);
    println!("|-Sample rate: {} Hz", sample_rate);

    let xcorr = audio.cross_correlation(&other_audio, core::num::NonZeroUsize::new(1).unwrap())?;
    println!("\nCross-correlation with 880 Hz tone: {:?}", xcorr);

    // --- Self-verification -------------------------------------------------
    // A symmetric tone centred on zero has ~zero mean and a peak near its
    // largest harmonic-sum amplitude (1.0 + 0.5 + 0.25 = 1.75, normalised by
    // the generator so |peak| <= ~1.0 region; we just assert sane bounds).
    assert!(audio.mean().abs() < 1e-2, "compound tone should be ~zero-mean, got {}", audio.mean());
    assert!(audio.rms() > 0.0, "rms of a non-silent signal must be positive");
    assert!(
        f64::from(audio.peak()) > audio.rms(),
        "peak must exceed rms for a tonal signal"
    );
    assert!(
        audio.max_sample() >= audio.min_sample(),
        "max must be >= min"
    );
    // A 440 Hz fundamental crosses zero ~twice per cycle, so the zero-crossing
    // rate (crossings per second) is on the order of ~880; assert it is positive
    // and in a sane sub-Nyquist range.
    assert!(
        audio.zero_crossing_rate() > 0.0 && audio.zero_crossing_rate() < 44_100.0,
        "zero-crossing rate out of expected range: {}",
        audio.zero_crossing_rate()
    );
    assert!(
        xcorr.as_slice().iter().all(|v| v.is_finite()),
        "cross-correlation values must be finite"
    );

    Ok(())
}
