#[cfg(feature = "channels")]
use std::time::Duration;

#[cfg(feature = "channels")]
use audio_samples::{AudioChannelOps, AudioSampleResult, AudioSamples, AudioStatistics, sine_wave};

#[cfg(feature = "channels")]
use audio_samples::operations::{MonoConversionMethod, StereoConversionMethod};

#[cfg(not(feature = "channels"))]
fn main() {
    eprintln!("This example requires the 'channels' feature.");
}

#[cfg(feature = "channels")]
pub fn main() -> AudioSampleResult<()> {
    let sample_rate_hz = 44_100u32;

    // Start with mono.
    let mono: AudioSamples<'static, f64> =
        sine_wave::<f64, f64>(440.0, Duration::from_millis(200), sample_rate_hz, 0.8);
    println!(
        "Mono: channels={} peak={:.4}",
        mono.num_channels(),
        mono.peak()
    );

    // Mono -> stereo.
    let mut stereo = mono.to_stereo(StereoConversionMethod::<f64>::Duplicate)?;
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
    let mono_avg = stereo.to_mono::<f64>(MonoConversionMethod::Average)?;
    println!(
        "Back to mono (avg): channels={} peak={:.4}",
        mono_avg.num_channels(),
        mono_avg.peak()
    );

    // Interleave/deinterleave.
    let ch0 = stereo.borrow_channel(0)?;
    let ch1 = stereo.borrow_channel(1)?;
    let interleaved = AudioSamples::interleave_channels(&[ch0, ch1])?;
    let deinterleaved = interleaved.deinterleave_channels()?;
    println!("Interleave/deinterleave: {} channels", deinterleaved.len());

    Ok(())
}
