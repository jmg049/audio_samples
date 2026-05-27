//! Integration tests: statistics + processing + editing + transforms pipeline.
//!
//! Verifies that operations from multiple modules compose correctly: signal
//! generation → normalization → scaling → trimming → FFT, checking that
//! metadata and amplitude invariants hold across the full chain.
//!
//! Run with:
//! ```bash
//! cargo test --test processing_pipeline --no-default-features --features statistics,processing,editing,transforms
//! ```

use std::num::NonZeroUsize;
use std::time::Duration;

use audio_samples::{
    AudioEditing, AudioProcessing, AudioSamples, AudioStatistics, AudioTransforms,
    NormalizationConfig, sample_rate, sine_wave,
};

fn sine_f32(freq: f64, ms: u64, amp: f64) -> AudioSamples<'static, f32> {
    sine_wave::<f32>(freq, Duration::from_millis(ms), sample_rate!(44100), amp)
}

fn sine_f64(freq: f64, ms: u64, amp: f64) -> AudioSamples<'static, f64> {
    sine_wave::<f64>(freq, Duration::from_millis(ms), sample_rate!(44100), amp)
}

#[test]
fn normalize_to_peak_one() {
    let audio = sine_f32(440.0, 100, 0.3)
        .normalize(NormalizationConfig::peak(1.0_f32))
        .unwrap();
    let peak: f32 = audio.peak();
    assert!(
        (peak - 1.0_f32).abs() < 1e-5_f32,
        "peak was {peak}, expected 1.0"
    );
}

#[test]
fn normalize_then_scale_halves_peak() {
    // scale() takes f64; 0.5 is inferred as f64.
    let audio = sine_f32(440.0, 100, 0.3)
        .normalize(NormalizationConfig::peak(1.0_f32))
        .unwrap()
        .scale(0.5);
    let peak: f32 = audio.peak();
    assert!(
        (peak - 0.5_f32).abs() < 1e-5_f32,
        "peak was {peak}, expected ~0.5"
    );
}

#[test]
fn scale_preserves_sample_count() {
    let audio = sine_f32(440.0, 100, 1.0);
    let n_before = audio.samples_per_channel().get();
    let scaled = audio.scale(0.5);
    assert_eq!(
        scaled.samples_per_channel().get(),
        n_before,
        "scale changed sample count"
    );
}

#[test]
fn trim_to_half_duration() {
    let audio = sine_f32(440.0, 1000, 0.5); // 1 s → 44100 samples
    let full_len = audio.samples_per_channel().get();
    let trimmed = audio.trim(0.25, 0.75).unwrap(); // 0.5 s window
    let trimmed_len = trimmed.samples_per_channel().get();
    let expected = full_len / 2;
    let diff = (trimmed_len as i64 - expected as i64).unsigned_abs() as usize;
    assert!(
        diff <= 2,
        "trimmed length {trimmed_len}, expected ~{expected} (diff {diff})"
    );
}

#[test]
fn trim_preserves_sample_rate() {
    let sr = sample_rate!(44100);
    let audio = sine_f32(440.0, 1000, 0.5);
    let trimmed = audio.trim(0.1, 0.9).unwrap();
    assert_eq!(trimmed.sample_rate(), sr, "trim changed sample rate");
}

#[test]
fn statistics_after_normalize_scale() {
    let audio = sine_f32(440.0, 100, 0.5)
        .normalize(NormalizationConfig::peak(1.0_f32))
        .unwrap()
        .scale(0.5);
    let peak: f32 = audio.peak();
    let rms = audio.rms();
    assert!(
        peak <= 0.5_f32 + 1e-5_f32,
        "peak {peak} exceeds 0.5 after scale(0.5)"
    );
    assert!(rms > 0.0, "rms should be positive for a sine wave");
    assert!(
        rms <= f64::from(peak) + 1e-6,
        "rms {rms} should not exceed peak {peak}"
    );
}

#[test]
fn rms_increases_after_normalization_upward() {
    // Normalize a quiet signal to peak 1.0 → rms should increase.
    let original = sine_f32(440.0, 200, 0.3);
    let rms_before = original.rms();
    let normalized = original
        .normalize(NormalizationConfig::peak(1.0_f32))
        .unwrap();
    let rms_after = normalized.rms();
    assert!(
        rms_after > rms_before,
        "rms should increase when normalizing 0.3-amplitude sine to peak 1.0 \
         (before={rms_before:.6}, after={rms_after:.6})"
    );
}

#[test]
fn fft_peak_bin_matches_signal_frequency() {
    let freq_hz = 440.0_f64;
    // n_fft must be >= signal length. 50ms at 44100 Hz = 2205 samples < 4096.
    let n_fft = NonZeroUsize::new(4096).unwrap();
    let audio = sine_f64(freq_hz, 50, 1.0);
    let spectrum = audio.fft(n_fft).unwrap();

    // spectrum is Array2<Complex<f64>>, one row per channel, n_fft columns.
    let half = n_fft.get() / 2;
    let peak_bin = spectrum
        .row(0)
        .iter()
        .take(half)
        .enumerate()
        .max_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let freq_resolution = 44100.0 / n_fft.get() as f64;
    let peak_freq = peak_bin as f64 * freq_resolution;

    assert!(
        (peak_freq - freq_hz).abs() < 20.0,
        "FFT peak at {peak_freq:.1} Hz, expected {freq_hz} Hz \
         (bin {peak_bin}, res {freq_resolution:.2} Hz/bin)"
    );
}

#[test]
fn fft_shape_matches_channel_count_and_n_fft() {
    // 10ms at 44100 Hz = 441 samples < 1024 → zero-padded to n_fft.
    let n_fft = NonZeroUsize::new(1024).unwrap();
    let audio = sine_f64(440.0, 10, 1.0);
    let spectrum = audio.fft(n_fft).unwrap();
    // The FFT implementation returns a one-sided spectrum: n_fft/2 + 1 bins.
    assert_eq!(spectrum.shape()[0], 1, "expected 1 row for mono input");
    assert_eq!(
        spectrum.shape()[1],
        n_fft.get() / 2 + 1,
        "expected {n} columns (n_fft/2+1)",
        n = n_fft.get() / 2 + 1,
    );
}

#[test]
fn scale_does_not_shift_spectral_peak() {
    // 20ms at 44100 Hz = 882 samples < 1024.
    let n_fft = NonZeroUsize::new(1024).unwrap();
    let half = n_fft.get() / 2;

    let audio_a = sine_f64(440.0, 20, 1.0);
    let audio_b = sine_f64(440.0, 20, 0.5);

    let spec_a = audio_a.fft(n_fft).unwrap();
    let spec_b = audio_b.fft(n_fft).unwrap();

    let peak_bin = |spec: &ndarray::Array2<num_complex::Complex<f64>>| -> usize {
        spec.row(0)
            .iter()
            .take(half)
            .enumerate()
            .max_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    };

    assert_eq!(
        peak_bin(&spec_a),
        peak_bin(&spec_b),
        "amplitude scaling shifted the spectral peak bin"
    );
}
