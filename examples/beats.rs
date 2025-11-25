#[cfg(feature = "beat-detection")]
use std::time::Duration;

#[cfg(feature = "beat-detection")]
use audio_samples::{AudioSampleResult, AudioSamples, impulse};

#[cfg(feature = "beat-detection")]
use audio_samples::operations::BeatConfig;

#[cfg(not(feature = "beat-detection"))]
fn main() {
    eprintln!("This example requires the 'beat-detection' feature.");
}

#[cfg(feature = "beat-detection")]
pub fn main() -> AudioSampleResult<()> {
    use audio_samples::AudioEditing;

    let sample_rate_hz = 44_100u32;
    let duration = Duration::from_secs(4);

    // Click track at 120 BPM (0.5s between beats).
    let mut clicks: Vec<AudioSamples<'static, f64>> = Vec::new();
    for i in 0..8 {
        let t = i as f64 * 0.5;
        clicks.push(impulse::<f64, f64>(duration, sample_rate_hz, 1.0, t));
    }
    let audio = AudioSamples::mix::<f64>(&clicks, None)?;

    let config = BeatConfig::new(120.0f64).with_tolerance(0.1);
    let tracker = audio.detect_beats(&config, None)?;
    println!("Tempo (configured): {:.1} BPM", tracker.tempo_bpm);
    println!("Beats (s): {:?}", tracker.beat_times);

    Ok(())
}
