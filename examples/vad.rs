//! Voice Activity Detection (VAD) example.
//!
//! Run:
//! `cargo run --example vad` (default features include `full`)

#[cfg(not(feature = "vad"))]
fn main() {
    eprintln!("error: This example requires the `vad` feature.");
    std::process::exit(1);
}

#[cfg(feature = "vad")]
fn main() -> audio_samples::AudioSampleResult<()> {
    use audio_samples::{
        AudioEditing, AudioSamples, AudioVoiceActivityDetection, nzu,
        operations::types::{VadChannelPolicy, VadConfig, VadMethod},
        sample_rate,
        utils::generation::{silence, sine_wave},
    };
    use std::time::Duration;

    // Construct a simple signal: 0.25s silence + 0.5s tone + 0.25s silence.
    let sr = sample_rate!(44100);
    let sr_u32 = core::num::NonZeroU32::new(sr.get()).unwrap();

    let s1: AudioSamples<f32> = silence::<f32>(Duration::from_secs_f32(0.25), sr_u32);
    let tone: AudioSamples<f32> =
        sine_wave::<f32>(220.0, Duration::from_secs_f32(0.5), sr_u32, 0.5);
    let s2: AudioSamples<f32> = silence::<f32>(Duration::from_secs_f32(0.25), sr_u32);

    // Concatenate segments using the built-in API.
    use non_empty_slice::NonEmptyVec;
    let vec = vec![s1, tone, s2];
    let non_empty = NonEmptyVec::try_from(vec)
        .map_err(|_| audio_samples::AudioSampleError::layout("Empty audio"))?;
    let audio = AudioSamples::concatenate_owned(non_empty)?;

    // Configure VAD.
    let cfg = VadConfig::default()
        .with_method(VadMethod::Combined)
        .with_frame_size(nzu!(1024))?
        .with_hop_size(nzu!(512))?
        .with_channel_policy(VadChannelPolicy::AverageToMono)
        .with_energy_threshold_db(-40.0);

    // Trait API.
    let mask = audio.voice_activity_mask(&cfg)?;
    let regions = audio.speech_regions(&cfg)?;

    println!("frames: {}", mask.len());
    println!("speech frames: {}", mask.iter().filter(|&&b| b).count());
    println!("speech regions (samples): {regions:?}");

    Ok(())
}
