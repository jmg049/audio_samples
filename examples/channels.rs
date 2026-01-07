#[cfg(not(all(feature = "channels", feature = "statistics")))]
pub fn main() {
    eprintln!("error: This example requires the `channels` and `statistics` features.");
    std::process::exit(1);
}

#[cfg(all(feature = "channels", feature = "statistics"))]
pub fn main() -> audio_samples::AudioSampleResult<()> {
    use audio_samples::{
        AudioChannelOps, AudioSamples, AudioStatistics,
        operations::types::{MonoConversionMethod, StereoConversionMethod},
        sine_wave,
    };
    use non_empty_slice::NonEmptySlice;
    use std::time::Duration;

    let sample_rate_hz = core::num::NonZeroU32::new(44_100).unwrap();

    // Start with mono.
    let mono: AudioSamples<'static, f64> =
        sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate_hz, 0.8);
    println!(
        "Mono: channels={} peak={:.4}",
        mono.num_channels(),
        mono.peak()
    );

    // Mono -> stereo.
    let mut stereo = mono.to_stereo(StereoConversionMethod::Duplicate)?;
    println!(
        "Stereo: channels={} peak={:.4}",
        stereo.num_channels(),
        stereo.peak()
    );

    // Pan and balance.
    stereo.pan(-0.5f64)?;
    stereo.balance(0.2f64)?;

    // Extract and swap channels.
    let left = stereo.extract_channel(0)?;
    let right = stereo.extract_channel(1)?;

    println!(
        "Extracted: left_peak={:.4} right_peak={:.4}",
        left.peak(),
        right.peak()
    );

    stereo.swap_channels(0, 1)?;

    // Apply a function to a single channel (invert channel 1).
    stereo.apply_to_channel(1, |x| -x)?;

    // Stereo -> mono.
    let mono_avg = stereo.to_mono(MonoConversionMethod::Average)?;
    println!(
        "Back to mono (avg): channels={} peak={:.4}",
        mono_avg.num_channels(),
        mono_avg.peak()
    );

    // Interleave/deinterleave.
    let ch0 = stereo.borrow_channel(0)?;
    let ch1 = stereo.borrow_channel(1)?;
    let interleaved = AudioSamples::interleave_channels(NonEmptySlice::new(&[ch0, ch1]).unwrap())?;
    let deinterleaved = interleaved.deinterleave_channels()?;
    println!("Interleave/deinterleave: {} channels", deinterleaved.len());

    Ok(())
}
