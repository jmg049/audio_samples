#[cfg(feature = "core-ops")]
use std::time::Duration;

#[cfg(feature = "core-ops")]
use audio_samples::{AudioSampleResult, AudioStatistics};

#[cfg(feature = "core-ops")]
use audio_samples::operations::{
    AudioDynamicRange,
    types::{CompressorConfig, LimiterConfig},
};

#[cfg(feature = "core-ops")]
use audio_samples::utils::generation::am_signal;

#[cfg(not(feature = "core-ops"))]
fn main() {
    eprintln!("This example requires the 'core-ops' feature.");
}

#[cfg(feature = "core-ops")]
pub fn main() -> AudioSampleResult<()> {
    let sample_rate_hz = 44_100u32;
    let sr = sample_rate_hz as f64;

    // AM signal: varying envelope makes compression/limiting visible.
    let mut audio = am_signal::<f64, f64>(
        440.0,
        2.0,
        0.9,
        Duration::from_secs(2),
        sample_rate_hz,
        0.95,
    );
    println!(
        "Input: peak={:.4} rms={:.4}",
        audio.peak(),
        audio.rms::<f64>()
    );

    let config = CompressorConfig::vocal();
    let curve = audio.get_compression_curve(&config, &[-40.0, -30.0, -20.0, -10.0, 0.0], sr)?;
    println!("Compressor curve (dB out): {:?}", curve);

    let gr = audio.get_gain_reduction(&config, sr)?;
    let max_gr = gr.into_iter().fold(0.0f64, f64::max);
    println!("Max gain reduction: {:.2} dB", max_gr);

    audio.apply_compressor(&config, sr)?;
    println!(
        "Compressed: peak={:.4} rms={:.4}",
        audio.peak(),
        audio.rms::<f64>()
    );

    let limiter = LimiterConfig::mastering();
    audio.apply_limiter(&limiter, sr)?;
    println!(
        "Limited: peak={:.4} rms={:.4}",
        audio.peak(),
        audio.rms::<f64>()
    );

    Ok(())
}
