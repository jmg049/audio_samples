//! Integration tests: iir-filtering + dynamic-range + parametric-eq chain.
//!
//! Verifies that IIR filtering, dynamic-range compression, and parametric EQ
//! compose correctly and produce the expected amplitude effects.
//!
//! Run with:
//! ```bash
//! cargo test --test effects_chain --no-default-features --features iir-filtering,dynamic-range,parametric-eq
//! ```

use std::num::NonZeroUsize;
use std::time::Duration;

use audio_samples::{
    AudioDynamicRange, AudioIirFiltering, AudioParametricEq, AudioSamples,
    operations::types::{CompressorConfig, EqBand, ParametricEq},
    sample_rate, sine_wave,
};

fn sine_f32(freq: f64, ms: u64, amp: f64) -> AudioSamples<'static, f32> {
    sine_wave::<f32>(freq, Duration::from_millis(ms), sample_rate!(44100), amp)
}

fn rms(audio: &AudioSamples<'_, f32>) -> f64 {
    let slice = audio.as_slice().expect("contiguous buffer");
    let sum_sq: f64 = slice.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    (sum_sq / slice.len() as f64).sqrt()
}

fn peak_abs(audio: &AudioSamples<'_, f32>) -> f32 {
    audio
        .as_slice()
        .expect("contiguous buffer")
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn butterworth_lowpass_passes_below_cutoff() {
    // 440 Hz well below 2000 Hz cutoff: RMS should be mostly preserved.
    let mut audio = sine_f32(440.0, 500, 0.8);
    let rms_before = rms(&audio);
    audio
        .butterworth_lowpass_in_place(NonZeroUsize::new(2).unwrap(), 2000.0)
        .unwrap();
    let rms_after = rms(&audio);
    assert!(
        rms_after > rms_before * 0.7,
        "lowpass at 2 kHz attenuated 440 Hz too much: before={rms_before:.4}, after={rms_after:.4}"
    );
}

#[test]
fn butterworth_lowpass_attenuates_above_cutoff() {
    // 8000 Hz well above 1000 Hz cutoff: RMS should be significantly reduced.
    let mut audio = sine_f32(8000.0, 500, 0.8);
    let rms_before = rms(&audio);
    audio
        .butterworth_lowpass_in_place(NonZeroUsize::new(4).unwrap(), 1000.0)
        .unwrap();
    let rms_after = rms(&audio);
    assert!(
        rms_after < rms_before * 0.3,
        "lowpass at 1 kHz did not attenuate 8 kHz enough: \
         before={rms_before:.4}, after={rms_after:.4} (ratio {:.2})",
        rms_after / rms_before
    );
}

#[test]
fn butterworth_lowpass_does_not_produce_nan_or_inf() {
    let mut audio = sine_f32(1000.0, 200, 0.9);
    audio
        .butterworth_lowpass_in_place(NonZeroUsize::new(4).unwrap(), 500.0)
        .unwrap();
    let all_finite = audio.as_slice().unwrap().iter().all(|x| x.is_finite());
    assert!(all_finite, "butterworth_lowpass produced NaN or Inf");
}

#[test]
fn compressor_modifies_signal() {
    // The vocal() preset includes make-up gain, so the peak may rise, but the
    // compressor must demonstrably change the signal (non-trivial processing).
    let original = sine_f32(440.0, 300, 0.9);
    let rms_original = rms(&original);

    let mut compressed = sine_f32(440.0, 300, 0.9);
    compressed
        .apply_compressor_in_place(&CompressorConfig::vocal())
        .unwrap();
    let rms_after = rms(&compressed);

    assert!(
        (rms_after - rms_original).abs() > 0.01,
        "compressor (with make-up gain) did not modify RMS: \
         original={rms_original:.4}, after={rms_after:.4}"
    );
}

#[test]
fn compressor_output_is_finite() {
    let mut audio = sine_f32(440.0, 200, 0.8);
    audio.apply_compressor_in_place(&CompressorConfig::vocal()).unwrap();
    let all_finite = audio.as_slice().unwrap().iter().all(|x| x.is_finite());
    assert!(all_finite, "compressor output contains NaN or Inf");
}

#[test]
fn eq_peak_boost_increases_rms() {
    let mut audio = sine_f32(1000.0, 300, 0.5);
    let rms_before = rms(&audio);

    let mut eq = ParametricEq::new();
    eq.add_band(EqBand::peak(1000.0, 6.0, 2.0)); // +6 dB at 1 kHz
    audio.apply_parametric_eq_in_place(&eq).unwrap();

    let rms_after = rms(&audio);
    assert!(
        rms_after > rms_before,
        "EQ +6 dB boost did not increase RMS: before={rms_before:.4}, after={rms_after:.4}"
    );
}

#[test]
fn eq_output_is_finite() {
    let mut audio = sine_f32(440.0, 200, 0.7);
    let mut eq = ParametricEq::new();
    eq.add_band(EqBand::peak(1000.0, 3.0, 2.0));
    eq.add_band(EqBand::low_shelf(100.0, -2.0, 0.707));
    audio.apply_parametric_eq_in_place(&eq).unwrap();
    let all_finite = audio.as_slice().unwrap().iter().all(|x| x.is_finite());
    assert!(all_finite, "parametric EQ output contains NaN or Inf");
}

#[test]
fn full_chain_lowpass_compress_eq_is_finite() {
    let mut audio = sine_f32(440.0, 500, 0.8);

    audio
        .butterworth_lowpass_in_place(NonZeroUsize::new(2).unwrap(), 2000.0)
        .unwrap();
    audio.apply_compressor_in_place(&CompressorConfig::vocal()).unwrap();

    let mut eq = ParametricEq::new();
    eq.add_band(EqBand::peak(800.0, 2.0, 1.5));
    audio.apply_parametric_eq_in_place(&eq).unwrap();

    let all_finite = audio.as_slice().unwrap().iter().all(|x| x.is_finite());
    assert!(all_finite, "full effects chain produced NaN or Inf");
    let peak = peak_abs(&audio);
    assert!(peak > 0.0, "full effects chain produced silence");
}
