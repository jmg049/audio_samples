#[cfg(all(feature = "resampling", feature = "statistics", feature = "processing"))]
use std::time::Duration;

#[cfg(all(feature = "resampling", feature = "statistics", feature = "processing"))]
use audio_samples::{
    AudioProcessing, AudioSampleResult, AudioSamples, AudioStatistics,
    operations::types::ResamplingQuality, sine_wave,
};

#[cfg(not(all(feature = "resampling", feature = "statistics", feature = "processing")))]
fn main() {
    eprintln!("This example requires the 'resampling', 'statistics', and 'processing' features.");
    eprintln!("Run with: cargo run --example resampling --features full");
}

#[cfg(all(feature = "resampling", feature = "statistics", feature = "processing"))]
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

    // --- Self-verification -------------------------------------------------
    // Resampling 44.1 kHz -> 16 kHz must set the new rate exactly and shrink
    // the sample count by roughly the rate ratio (~0.3628).
    assert_eq!(
        resampled.sample_rate().get(),
        16_000,
        "resample must set the target sample rate"
    );
    let expected = audio.samples_per_channel().get() as f64 * (16_000.0 / 44_100.0);
    let got = resampled.samples_per_channel().get() as f64;
    assert!(
        (got - expected).abs() / expected < 0.02,
        "resampled length {got} should be ~{expected:.0} (within 2%)"
    );
    // Ratio 0.5 roughly halves the length.
    let half_expected = audio.samples_per_channel().get() as f64 * 0.5;
    let half_got = half.samples_per_channel().get() as f64;
    assert!(
        (half_got - half_expected).abs() / half_expected < 0.02,
        "ratio-0.5 length {half_got} should be ~{half_expected:.0} (within 2%)"
    );

    Ok(())
}
