#[cfg(feature = "spectral-analysis")]
use std::time::Duration;

#[cfg(feature = "spectral-analysis")]
use audio_samples::{AudioSampleResult, AudioSamples, impulse};

#[cfg(feature = "spectral-analysis")]
use audio_samples::operations::types::OnsetConfig;

#[cfg(not(feature = "spectral-analysis"))]
fn main() {
    eprintln!("This example requires the 'spectral-analysis' feature.");
}

#[cfg(feature = "spectral-analysis")]
pub fn main() -> AudioSampleResult<()> {
    use audio_samples::AudioEditing;

    let sample_rate_hz = 44_100u32;
    let duration = Duration::from_secs(4);

    // Synthetic click track: one impulse every 0.5s (120 BPM).
    let mut clicks: Vec<AudioSamples<'static, f64>> = Vec::new();
    for i in 0..8 {
        let t = i as f64 * 0.5;
        clicks.push(impulse::<f64, f64>(duration, sample_rate_hz, 1.0, t));
    }
    let audio = AudioSamples::mix::<f64>(&clicks, None)?;

    let config = OnsetConfig::<f64>::percussive();
    let onsets = audio.detect_onsets(&config)?;
    println!("Detected onsets (s): {:?}", onsets);

    let env = audio.onset_strength_envelope::<f64>(&config, None)?;
    println!("Onset envelope: len={}", env.len());

    Ok(())
}
