#[cfg(not(all(feature = "dynamic-range", feature = "statistics")))]
pub fn main() {
    eprintln!("error: This example requires the `dynamic-range` and `statistics` features.");
    std::process::exit(1);
}

#[cfg(all(feature = "dynamic-range", feature = "statistics"))]
pub fn main() -> audio_samples::AudioSampleResult<()> {
    use audio_samples::AudioStatistics;
    use audio_samples::operations::{
        AudioDynamicRange,
        types::{CompressorConfig, LimiterConfig},
    };
    use audio_samples::utils::generation::am_signal;
    use std::time::Duration;

    let sample_rate_hz = core::num::NonZeroU32::new(44_100).unwrap();

    // AM signal: varying envelope makes compression/limiting visible.
    let mut audio = am_signal::<f64>(
        440.0,
        2.0,
        0.9,
        Duration::from_secs(2),
        sample_rate_hz,
        0.95,
    );
    println!("Input: peak={:.4} rms={:.4}", audio.peak(), audio.rms());

    let config = CompressorConfig::vocal();
    let curve_input = [-40.0f64, -30.0, -20.0, -10.0, 0.0];
    let curve = audio.get_compression_curve(
        &config,
        non_empty_slice::NonEmptySlice::from_slice(&curve_input).unwrap(),
    )?;
    println!("Compressor curve (dB out): {:?}", curve);

    let gr = audio.get_gain_reduction(&config)?;
    let max_gr = gr.into_iter().fold(0.0f64, f64::max);
    println!("Max gain reduction: {:.2} dB", max_gr);

    audio.apply_compressor(&config)?;
    println!(
        "Compressed: peak={:.4} rms={:.4}",
        audio.peak(),
        audio.rms()
    );

    let limiter = LimiterConfig::mastering();
    audio.apply_limiter(&limiter)?;
    println!("Limited: peak={:.4} rms={:.4}", audio.peak(), audio.rms());

    Ok(())
}
