//! Integration tests: channels + processing + editing pipeline.
//!
//! Verifies that mono/stereo conversions, channel duplication, and extraction
//! hold when the channels, processing, and editing features are active together.
//!
//! Run with:
//! ```bash
//! cargo test --test channel_pipeline --no-default-features --features channels,processing,editing
//! ```

use std::time::Duration;

use audio_samples::{
    AudioChannelOps, AudioEditing, AudioProcessing, AudioSamples, AudioStatistics,
    operations::types::{MonoConversionMethod, StereoConversionMethod},
    sample_rate, sine_wave,
};

fn sine_f32(freq: f64, ms: u64, amp: f64) -> AudioSamples<'static, f32> {
    sine_wave::<f32>(freq, Duration::from_millis(ms), sample_rate!(44100), amp)
}

#[test]
fn duplicate_produces_correct_channel_count() {
    let mono = sine_f32(440.0, 50, 0.5);
    for n in 1usize..=4 {
        let multi = mono.duplicate_to_channels(n).unwrap();
        assert_eq!(
            multi.num_channels().get(),
            n as u32,
            "duplicate_to_channels({n}) returned wrong channel count"
        );
    }
}

#[test]
fn duplicate_preserves_sample_count() {
    let mono = sine_f32(440.0, 50, 0.5);
    let n = mono.samples_per_channel().get();
    let stereo = mono.duplicate_to_channels(2).unwrap();
    assert_eq!(
        stereo.samples_per_channel().get(),
        n,
        "duplicate_to_channels changed sample count from {n}"
    );
}

#[test]
fn duplicate_both_channels_match_original() {
    let mono = sine_f32(440.0, 50, 0.5);
    let original: Vec<f32> = mono.as_slice().unwrap().to_vec();
    let stereo = mono.duplicate_to_channels(2).unwrap();

    let ch0 = stereo.extract_channel(0).unwrap();
    let ch1 = stereo.extract_channel(1).unwrap();

    for (i, (&orig, &c0)) in original.iter().zip(ch0.as_slice().unwrap()).enumerate() {
        assert_eq!(orig, c0, "ch0 sample {i}: orig={orig}, got={c0}");
    }
    for (i, (&orig, &c1)) in original.iter().zip(ch1.as_slice().unwrap()).enumerate() {
        assert_eq!(orig, c1, "ch1 sample {i}: orig={orig}, got={c1}");
    }
}

#[test]
fn extract_channel_is_mono() {
    let stereo = sine_f32(440.0, 50, 0.5).duplicate_to_channels(2).unwrap();
    let ch = stereo.extract_channel(0).unwrap();
    assert!(ch.is_mono(), "extracted channel should be mono");
}

#[test]
fn extract_channel_preserves_sample_rate() {
    let sr = sample_rate!(22050);
    let audio = sine_wave::<f32>(440.0, Duration::from_millis(50), sr, 0.5);
    let stereo = audio.duplicate_to_channels(2).unwrap();
    let ch = stereo.extract_channel(0).unwrap();
    assert_eq!(ch.sample_rate(), sr, "extract_channel changed sample rate");
}

#[test]
fn to_stereo_produces_two_channels() {
    let mono = sine_f32(440.0, 50, 0.5);
    let stereo = mono.to_stereo(StereoConversionMethod::Duplicate).unwrap();
    assert_eq!(stereo.num_channels().get(), 2);
}

#[test]
fn to_stereo_then_to_mono_preserves_first_sample() {
    let mono = sine_f32(440.0, 50, 0.5);
    let first_sample = mono.as_slice().unwrap()[0];
    let n = mono.samples_per_channel().get();

    let stereo = mono.to_stereo(StereoConversionMethod::Duplicate).unwrap();
    let back = stereo.to_mono(MonoConversionMethod::Average).unwrap();

    assert!(back.is_mono());
    assert_eq!(
        back.samples_per_channel().get(),
        n,
        "round-trip changed sample count"
    );
    let round_trip_first = back.as_slice().unwrap()[0];
    assert!(
        (round_trip_first - first_sample).abs() < 1e-5_f32,
        "round-trip first sample changed: {first_sample} → {round_trip_first}"
    );
}

#[test]
fn to_stereo_preserves_sample_rate() {
    let sr = sample_rate!(48000);
    let audio = sine_wave::<f32>(440.0, Duration::from_millis(50), sr, 0.5);
    let stereo = audio.to_stereo(StereoConversionMethod::Duplicate).unwrap();
    assert_eq!(stereo.sample_rate(), sr);
}

#[test]
fn scale_then_duplicate_peak_consistent() {
    // scale() takes f64.
    let mono = sine_f32(440.0, 100, 1.0).scale(0.5);
    let peak_mono: f32 = mono.peak();
    let stereo = mono.duplicate_to_channels(2).unwrap();
    let ch0 = stereo.extract_channel(0).unwrap();
    let peak_ch0: f32 = ch0.peak();
    assert!(
        (peak_ch0 - peak_mono).abs() < 1e-5_f32,
        "peak changed after duplicate: mono={peak_mono}, ch0={peak_ch0}"
    );
}

#[test]
fn trim_then_duplicate_preserves_trimmed_length() {
    let audio = sine_f32(440.0, 1000, 0.5);
    let trimmed = audio.trim(0.25, 0.75).unwrap();
    let trimmed_len = trimmed.samples_per_channel().get();
    let stereo = trimmed.duplicate_to_channels(2).unwrap();
    assert_eq!(
        stereo.samples_per_channel().get(),
        trimmed_len,
        "duplicate changed length after trim"
    );
}
