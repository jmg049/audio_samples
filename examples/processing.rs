#[cfg(feature = "processing")]
use std::f64::consts::PI;
#[cfg(feature = "processing")]
use std::time::Duration;

#[cfg(feature = "processing")]
use audio_samples::{
    AudioProcessing, AudioSampleResult, AudioSamples, AudioStatistics, NormalizationMethod,
    sine_wave,
};

#[cfg(not(feature = "processing"))]
fn main() {
    eprintln!("This example requires the 'processing' feature.");
}

#[cfg(feature = "processing")]
fn hann_window(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }

    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * (i as f64) / ((n - 1) as f64)).cos())
        .collect()
}

#[cfg(feature = "processing")]
pub fn main() -> AudioSampleResult<()> {
    use audio_samples::AudioEditing;

    let sample_rate_hz = 44_100u32;

    // Start with a clean 440 Hz sine.
    let mut audio: AudioSamples<'static, f64> =
        sine_wave::<f64, f64>(440.0, Duration::from_secs(1), sample_rate_hz, 0.8);
    println!(
        "Input:  peak={:.4}  rms={:.4}",
        audio.peak(),
        audio.rms::<f64>()
    );

    // Simple gain.
    audio.scale(0.5);
    println!(
        "Scaled: peak={:.4}  rms={:.4}",
        audio.peak(),
        audio.rms::<f64>()
    );

    // Remove DC offset (no-op for a pure sine, but demonstrates API).
    audio.remove_dc_offset()?;

    // Normalization.
    audio.normalize(-1.0, 1.0, NormalizationMethod::Peak)?;
    println!(
        "Normalized (Peak): peak={:.4}  rms={:.4}",
        audio.peak(),
        audio.rms::<f64>()
    );

    // Clip to a tighter range.
    audio.clip(-0.7, 0.7)?;
    println!(
        "Clipped: peak={:.4}  rms={:.4}",
        audio.peak(),
        audio.rms::<f64>()
    );

    // Windowing (Hann).
    let window = hann_window(audio.samples_per_channel());
    audio.apply_window(&window)?;
    println!(
        "Windowed: peak={:.4}  rms={:.4}",
        audio.peak(),
        audio.rms::<f64>()
    );

    // FIR filtering (moving average).
    let taps = vec![1.0 / 9.0; 9];
    audio.apply_filter(&taps)?;
    println!(
        "Filtered (moving avg): len={}  peak={:.4}  rms={:.4}",
        audio.samples_per_channel(),
        audio.peak(),
        audio.rms::<f64>()
    );

    // µ-law round-trip.
    let mut ulaw = audio.clone();
    ulaw.mu_compress(255.0)?;
    ulaw.mu_expand(255.0)?;
    println!(
        "µ-law round-trip: peak={:.4}  rms={:.4}",
        ulaw.peak(),
        ulaw.rms::<f64>()
    );

    // Simple IIR-ish (internal) filters based on cutoff frequency.
    let mut lp = sine_wave::<f64, f64>(440.0, Duration::from_secs(1), sample_rate_hz, 0.6);
    // Add a higher tone so the low-pass effect is visible in stats.
    let hi = sine_wave::<f64, f64>(5_000.0, Duration::from_secs(1), sample_rate_hz, 0.2);
    let mixed = AudioSamples::mix::<f64>(&[lp.clone(), hi], None)?;
    lp = mixed;

    println!(
        "Two-tone: peak={:.4}  rms={:.4}",
        lp.peak(),
        lp.rms::<f64>()
    );
    lp.low_pass_filter::<f64>(1_000.0)?;
    println!(
        "Low-pass 1kHz: peak={:.4}  rms={:.4}",
        lp.peak(),
        lp.rms::<f64>()
    );

    Ok(())
}
