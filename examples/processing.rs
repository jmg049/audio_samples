#[cfg(feature = "processing")]
use std::time::Duration;

#[cfg(feature = "processing")]
use audio_samples::{
    AudioProcessing, AudioSampleResult, AudioSamples, AudioStatistics, NormalizationConfig,
    sine_wave,
};

#[cfg(not(feature = "processing"))]
fn main() {
    eprintln!("This example requires the 'processing' feature.");
}

#[cfg(feature = "processing")]
#[inline]
pub fn main() -> AudioSampleResult<()> {
    use audio_samples::{AudioEditing, sample_rate};

    let sample_rate_hz = sample_rate!(44100);

    // Start with a clean 440 Hz sine.
    let audio: AudioSamples<'static, f64> =
        sine_wave(440.0, Duration::from_secs(1), sample_rate_hz, 0.8);
    println!("Input:  peak={:.4}  rms={:.4}", audio.peak(), audio.rms());

    // Demonstrate method chaining with the new consuming API
    // Simple gain.
    let audio = audio.scale(0.5);
    println!("Scaled: peak={:.4}  rms={:.4}", audio.peak(), audio.rms());

    // Remove DC offset (no-op for a pure sine, but demonstrates API).
    let audio = audio.remove_dc_offset()?;

    // Normalization using NormalizationConfig.
    let audio = audio.normalize(NormalizationConfig::peak(1.0))?;
    println!(
        "Normalized (Peak): peak={:.4}  rms={:.4}",
        audio.peak(),
        audio.rms()
    );

    // Clip to a tighter range.
    let audio = audio.clip(-0.7, 0.7)?;
    println!("Clipped: peak={:.4}  rms={:.4}", audio.peak(), audio.rms());

    // Windowing (Hann).

    let num_samples = audio.samples_per_channel().get() as usize;
    let window = spectrograms::hanning_window(core::num::NonZeroUsize::new(num_samples).unwrap());
    let audio = audio.apply_window(&window)?;
    println!("Windowed: peak={:.4}  rms={:.4}", audio.peak(), audio.rms());

    // FIR filtering (moving average).
    let taps = vec![1.0 / 9.0; 9];
    let audio = audio.apply_filter(&non_empty_slice::NonEmptySlice::from_slice(&taps).unwrap())?;
    println!(
        "Filtered (moving avg): len={}  peak={:.4}  rms={:.4}",
        audio.samples_per_channel(),
        audio.peak(),
        audio.rms()
    );

    // µ-law round-trip - demonstrating chaining.
    let ulaw = audio.clone().mu_compress(255.0)?.mu_expand(255.0)?;
    println!(
        "µ-law round-trip: peak={:.4}  rms={:.4}",
        ulaw.peak(),
        ulaw.rms()
    );

    // Simple IIR-ish (internal) filters based on cutoff frequency.
    let lp = sine_wave::<f64>(440.0, Duration::from_secs(1), sample_rate_hz, 0.6);
    // Add a higher tone so the low-pass effect is visible in stats.
    let hi = sine_wave::<f64>(5_000.0, Duration::from_secs(1), sample_rate_hz, 0.2);
    let mixed = AudioSamples::mix(
        &non_empty_slice::NonEmptySlice::from_slice(&[lp, hi]).unwrap(),
        None,
    )?;

    println!("Two-tone: peak={:.4}  rms={:.4}", mixed.peak(), mixed.rms());
    let lp = mixed.low_pass_filter(1_000.0)?;
    println!("Low-pass 1kHz: peak={:.4}  rms={:.4}", lp.peak(), lp.rms());

    Ok(())
}
