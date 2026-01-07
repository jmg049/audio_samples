#[cfg(feature = "resampling")]
use std::time::Duration;

#[cfg(feature = "resampling")]
use audio_samples::{
    AudioProcessing, AudioSampleResult, AudioSamples, AudioStatistics,
    operations::types::ResamplingQuality, sine_wave,
};

#[cfg(not(feature = "resampling"))]
fn main() {
    eprintln!("This example requires the 'resampling' feature.");
}

#[cfg(feature = "resampling")]
#[inline]
pub fn main() -> AudioSampleResult<()> {
    let src_sr = core::num::NonZeroU32::new(44_100).unwrap();
    let dst_sr = core::num::NonZeroU32::new(16_000).unwrap();

    let audio: AudioSamples<'static, f64> =
        sine_wave::<f64>(440.0, Duration::from_secs(1), src_sr, 0.8);
    println!(
        "Input: sr={}Hz len={} peak={:.4}",
        audio.sample_rate().get(),
        audio.samples_per_channel().get(),
        audio.peak()
    );

    let resampled = audio.resample(dst_sr, ResamplingQuality::High)?;
    println!(
        "Resampled: sr={}Hz len={} peak={:.4}",
        resampled.sample_rate().get(),
        resampled.samples_per_channel().get(),
        resampled.peak()
    );

    let half = audio.resample_by_ratio(0.5, ResamplingQuality::Medium)?;
    println!(
        "Ratio 0.5: sr={}Hz len={} peak={:.4}",
        half.sample_rate().get(),
        half.samples_per_channel().get(),
        half.peak()
    );

    Ok(())
}
