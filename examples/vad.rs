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
    let non_empty = NonEmptyVec::try_from(vec).map_err(|_| {
        audio_samples::AudioSampleError::empty_data("vad example: concatenate segments")
    })?;
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

    let speech_frames = mask.iter().filter(|&&b| b).count();
    println!("frames: {}", mask.len());
    println!("speech frames: {}", speech_frames);
    println!("speech regions (samples): {regions:?}");

    // --- Self-verification -------------------------------------------------
    // The 1 s signal at 44.1 kHz framed with a 512-sample hop yields a
    // predictable, non-zero frame count, and the per-frame mask and derived
    // regions must be self-consistent.
    let total_frames = mask.len().get();
    assert!(total_frames > 50, "expected ~87 analysis frames, got {total_frames}");
    assert!(
        speech_frames <= total_frames,
        "speech frames cannot exceed total frames"
    );
    // Every reported region must lie within the signal and be non-empty.
    let n_samples = audio.samples_per_channel().get();
    for (start, end) in &regions {
        assert!(start < end, "region must be non-empty: ({start}, {end})");
        assert!(*end <= n_samples, "region end {end} exceeds signal length {n_samples}");
    }

    Ok(())
}
