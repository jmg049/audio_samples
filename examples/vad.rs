//! Voice Activity Detection (VAD) example.
//!
//! Run:
//! `cargo run --example vad` (default features include `full`)

#[cfg(not(feature = "statistics"))]
compile_error!("This example requires the `statistics` feature.");

#[cfg(feature = "statistics")]
fn run() -> audio_samples::AudioSampleResult<()> {
    use audio_samples::operations::traits::AudioVoiceActivityDetection;
    use audio_samples::operations::types::{VadChannelPolicy, VadConfig, VadMethod};
    use audio_samples::utils::detection::{detect_speech_regions, detect_voice_activity_mask};
    use audio_samples::utils::generation::{silence, sine_wave};
    use audio_samples::{AudioEditing, AudioSamples, sample_rate};
    use std::time::Duration;

    // Construct a simple signal: 0.25s silence + 0.5s tone + 0.25s silence.
    let sr = sample_rate!(44100);
    let sr_u32 = sr.get();

    let s1: AudioSamples<f32> = silence::<f32, f32>(Duration::from_secs_f32(0.25), sr_u32);
    let tone: AudioSamples<f32> =
        sine_wave::<f32, f32>(220.0, Duration::from_secs_f32(0.5), sr_u32, 0.5);
    let s2: AudioSamples<f32> = silence::<f32, f32>(Duration::from_secs_f32(0.25), sr_u32);

    // Concatenate segments using the built-in API.
    let audio = AudioSamples::concatenate_owned(vec![s1, tone, s2])?;

    // Configure VAD.
    let cfg = VadConfig::<f32> {
        method: VadMethod::Combined,
        frame_size: 1024,
        hop_size: 512,
        pad_end: false,
        channel_policy: VadChannelPolicy::AverageToMono,
        energy_threshold_db: audio_samples::to_precision(-40.0),
        ..VadConfig::new()
    };

    // Trait API.
    let mask = audio.voice_activity_mask(&cfg)?;
    let regions = audio.speech_regions(&cfg)?;

    println!("frames: {}", mask.len());
    println!("speech frames: {}", mask.iter().filter(|&&b| b).count());
    println!("speech regions (samples): {regions:?}");

    // Utils wrappers (same behavior).
    let mask2 = detect_voice_activity_mask(&audio, &cfg)?;
    let regions2 = detect_speech_regions(&audio, &cfg)?;

    assert_eq!(mask, mask2);
    assert_eq!(regions, regions2);

    Ok(())
}

#[cfg(feature = "statistics")]
fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
