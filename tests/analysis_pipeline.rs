//! Integration tests: pitch-analysis + vad pipeline.
//!
//! Verifies that pitch detection and voice activity detection produce
//! correct results on known signals when both features are active together.
//!
//! Run with:
//! ```bash
//! cargo test --test analysis_pipeline --no-default-features --features pitch-analysis,vad
//! ```

use std::time::Duration;

use audio_samples::{
    AudioPitchAnalysis, AudioSamples, AudioVoiceActivityDetection, operations::types::VadConfig,
    sample_rate, silence, sine_wave,
};

fn sine_f64(freq: f64, ms: u64, amp: f64) -> AudioSamples<'static, f64> {
    sine_wave::<f64>(freq, Duration::from_millis(ms), sample_rate!(44100), amp)
}

fn sine_f32(freq: f64, ms: u64, amp: f64) -> AudioSamples<'static, f32> {
    sine_wave::<f32>(freq, Duration::from_millis(ms), sample_rate!(44100), amp)
}

#[test]
fn yin_detects_440hz() {
    let hz = 440.0_f64;
    let audio = sine_f64(hz, 200, 1.0);
    let detected = audio.detect_pitch_yin(0.1, 80.0, 1000.0).unwrap();
    assert!(
        detected.is_some(),
        "YIN returned None for a clean 440 Hz sine"
    );
    let detected_hz = detected.unwrap();
    assert!(
        (detected_hz - hz).abs() < 15.0,
        "YIN detected {detected_hz:.1} Hz, expected {hz} Hz (tolerance ±15 Hz)"
    );
}

#[test]
fn yin_detects_880hz() {
    let hz = 880.0_f64;
    let audio = sine_f64(hz, 200, 1.0);
    let detected = audio.detect_pitch_yin(0.1, 200.0, 2000.0).unwrap();
    assert!(
        detected.is_some(),
        "YIN returned None for a clean 880 Hz sine"
    );
    let detected_hz = detected.unwrap();
    assert!(
        (detected_hz - hz).abs() < 20.0,
        "YIN detected {detected_hz:.1} Hz, expected {hz} Hz (tolerance ±20 Hz)"
    );
}

#[test]
fn yin_returns_error_for_invalid_parameters() {
    let audio = sine_f64(440.0, 100, 0.5);
    // min_frequency <= 0 should be an error.
    assert!(
        audio.detect_pitch_yin(0.1, 0.0, 1000.0).is_err(),
        "YIN should error when min_frequency is 0"
    );
    // max_frequency <= min_frequency should be an error.
    assert!(
        audio.detect_pitch_yin(0.1, 500.0, 200.0).is_err(),
        "YIN should error when max_frequency < min_frequency"
    );
}

#[test]
fn vad_detects_sine_as_active() {
    // 440 Hz sine at −9 dBFS (amplitude 0.5 ≈ −6 dBFS) is well above the
    // default energy threshold of −40 dBFS.
    let audio = sine_f32(440.0, 200, 0.5);
    let config = VadConfig::energy_only();
    let mask = audio.voice_activity_mask(&config).unwrap();
    assert!(
        mask.iter().any(|&v| v),
        "VAD classified all frames of a 440 Hz sine as silence"
    );
}

#[test]
fn vad_silence_is_inactive() {
    let audio = silence::<f32>(Duration::from_millis(200), sample_rate!(44100));
    let config = VadConfig::energy_only();
    let mask = audio.voice_activity_mask(&config).unwrap();
    assert!(
        mask.iter().all(|&v| !v),
        "VAD classified silence frames as speech"
    );
}

#[test]
fn speech_regions_non_empty_for_sine() {
    let audio = sine_f32(440.0, 300, 0.5);
    let config = VadConfig::energy_only();
    let regions = audio.speech_regions(&config).unwrap();
    assert!(
        !regions.is_empty(),
        "speech_regions returned no regions for a 440 Hz sine"
    );
}

#[test]
fn speech_regions_empty_for_silence() {
    let audio = silence::<f32>(Duration::from_millis(300), sample_rate!(44100));
    let config = VadConfig::energy_only();
    let regions = audio.speech_regions(&config).unwrap();
    assert!(
        regions.is_empty(),
        "speech_regions returned regions for pure silence: {regions:?}"
    );
}

#[test]
fn voiced_signal_detected_by_both_yin_and_vad() {
    let hz = 440.0_f64;
    let audio_f64 = sine_f64(hz, 200, 0.8);
    let audio_f32 = sine_f32(hz, 200, 0.8);

    // YIN on f64 signal
    let pitch = audio_f64.detect_pitch_yin(0.1, 80.0, 1000.0).unwrap();
    assert!(pitch.is_some(), "YIN found no pitch in voiced signal");

    // VAD on f32 signal
    let config = VadConfig::energy_only();
    let mask = audio_f32.voice_activity_mask(&config).unwrap();
    assert!(
        mask.iter().any(|&v| v),
        "VAD found no active frames in voiced signal"
    );
}
