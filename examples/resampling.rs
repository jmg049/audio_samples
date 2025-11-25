#[cfg(feature = "resampling")]
use std::time::Duration;

#[cfg(feature = "resampling")]
use audio_samples::{AudioProcessing, AudioSampleResult, AudioSamples, AudioStatistics, sine_wave};

#[cfg(feature = "resampling")]
use audio_samples::operations::ResamplingQuality;

#[cfg(not(feature = "resampling"))]
fn main() {
    eprintln!("This example requires the 'resampling' feature.");
}

#[cfg(feature = "resampling")]
pub fn main() -> AudioSampleResult<()> {
    let src_sr = 44_100u32;
    let dst_sr = 16_000usize;

    let audio: AudioSamples<'static, f64> =
        sine_wave::<f64, f64>(440.0, Duration::from_secs(1), src_sr, 0.8);
    println!(
        "Input: sr={}Hz len={} peak={:.4}",
        audio.sample_rate().get(),
        audio.samples_per_channel(),
        audio.peak()
    );

    let resampled = audio.resample::<f64>(dst_sr, ResamplingQuality::High)?;
    println!(
        "Resampled: sr={}Hz len={} peak={:.4}",
        resampled.sample_rate().get(),
        resampled.samples_per_channel(),
        resampled.peak()
    );

    let half = audio.resample_by_ratio::<f64>(0.5, ResamplingQuality::Medium)?;
    println!(
        "Ratio 0.5: sr={}Hz len={} peak={:.4}",
        half.sample_rate().get(),
        half.samples_per_channel(),
        half.peak()
    );

    Ok(())
}
