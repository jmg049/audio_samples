//! Integration tests for the `utils` module: `audio_math`, `comparison`, and
//! `detection`.
//!
//! These exercise the largely-ungated utility surface with known-value checks,
//! round-trips, and the documented note-parsing case sensitivity. The
//! log-spectral-distance check is gated behind `transforms` (it runs an FFT),
//! and is therefore compiled in only when that feature is present.
//!
//! Run with:
//! ```bash
//! cargo test --test utils_integration --features transforms
//! ```

use std::time::Duration;

use audio_samples::utils::audio_math::{
    cents_to_ratio, frequency_to_note, hz_to_midi, midi_to_hz, midi_to_note, note_to_frequency,
    note_to_midi, ratio_to_cents,
};
use audio_samples::utils::audio_math::{ms_to_samples, samples_to_time, seconds_to_samples};
use audio_samples::utils::comparison::{
    correlation, correlation_per_channel, mse, mse_per_channel, psnr, segmental_snr, snr,
};
use audio_samples::utils::detection::{detect_dynamic_range, detect_fundamental_frequency};
use audio_samples::utils::generation::sine_wave;
use audio_samples::{AudioSampleError, AudioSamples, sample_rate};
use ndarray::Array1;

fn sine_f64(freq: f64, ms: u64, amp: f64) -> AudioSamples<'static, f64> {
    sine_wave::<f64>(freq, Duration::from_millis(ms), sample_rate!(44100), amp)
}

fn mono_f64(data: Vec<f64>) -> AudioSamples<'static, f64> {
    AudioSamples::new_mono(Array1::from(data), sample_rate!(44100)).unwrap()
}

// =============================================================================
// audio_math — note / MIDI / frequency conversions
// =============================================================================

#[test]
fn note_to_midi_known_values() {
    assert_eq!(note_to_midi("A4").unwrap(), 69);
    assert_eq!(note_to_midi("C4").unwrap(), 60);
    assert_eq!(note_to_midi("C0").unwrap(), 12);
}

#[test]
fn enharmonic_spellings_match() {
    assert_eq!(note_to_midi("C#4").unwrap(), 61);
    assert_eq!(note_to_midi("Db4").unwrap(), 61);
    assert_eq!(note_to_midi("C#4").unwrap(), note_to_midi("Db4").unwrap());
}

#[test]
fn midi_note_round_trip() {
    // midi_to_note normalises to sharps, so only sharp/natural names round-trip.
    for &(name, midi) in &[("A4", 69u8), ("C4", 60), ("C#4", 61), ("G2", 43)] {
        assert_eq!(note_to_midi(name).unwrap(), midi);
        assert_eq!(midi_to_note(midi).unwrap(), name);
    }
}

#[test]
fn note_parsing_is_case_sensitive() {
    // The reviewer-flagged behaviour: lowercase note letters are NOT accepted
    // and must error rather than silently mapping to the uppercase note.
    assert!(note_to_midi("a4").is_err(), "lowercase 'a4' should error");
    assert!(note_to_midi("c4").is_err(), "lowercase 'c4' should error");
    assert!(note_to_midi("db4").is_err(), "lowercase 'db4' should error");

    // It is specifically a *parse* error, not a range error.
    match note_to_midi("a4").unwrap_err() {
        AudioSampleError::NoteParse(_) => {}
        other => panic!("expected NoteParse for 'a4', got {other:?}"),
    }

    // And it does not coincidentally equal the uppercase result.
    assert!(note_to_midi("a4").is_err() && note_to_midi("A4").is_ok());
}

#[test]
fn note_to_frequency_a4_is_440() {
    let a4 = note_to_frequency("A4").unwrap();
    assert!((a4 - 440.0).abs() < 1e-6, "A4 was {a4}, expected 440");
}

#[test]
fn frequency_to_note_round_trip() {
    let (note, cents) = frequency_to_note(440.0).unwrap();
    assert_eq!(note, "A4");
    assert!(cents.abs() < 1e-6, "440 Hz should be 0 cents, was {cents}");

    // Non-positive frequency is rejected.
    assert!(frequency_to_note(0.0).is_err());
}

#[test]
fn hz_midi_round_trip() {
    assert!((hz_to_midi(440.0) - 69.0).abs() < 1e-9);
    assert!((midi_to_hz(69.0) - 440.0).abs() < 1e-9);
    // Octave up = +12 semitones = double frequency.
    assert!((midi_to_hz(81.0) - 880.0).abs() < 1e-6);
}

#[test]
fn cents_ratio_round_trip() {
    // 1200 cents = one octave = ratio 2.0.
    assert!((cents_to_ratio(1200.0) - 2.0).abs() < 1e-9);
    assert!((ratio_to_cents(2.0) - 1200.0).abs() < 1e-6);
    let r = cents_to_ratio(37.5);
    assert!((ratio_to_cents(r) - 37.5).abs() < 1e-6);
}

// =============================================================================
// audio_math — time <-> samples
// =============================================================================

#[test]
fn seconds_samples_conversions() {
    assert_eq!(seconds_to_samples(1.0, 44_100.0), 44_100);
    assert_eq!(seconds_to_samples(0.5, 44_100.0), 22_050);
    assert_eq!(ms_to_samples(10.0, 44_100.0), 441);

    // samples_to_time is the inverse of seconds_to_samples on whole seconds.
    let t = samples_to_time(44_100, 44_100.0);
    assert!((t - 1.0).abs() < 1e-12, "samples_to_time gave {t}");

    // Round-trip a duration.
    let n = seconds_to_samples(0.25, 48_000.0);
    let back = samples_to_time(n, 48_000.0);
    assert!((back - 0.25).abs() < 1e-9, "round-trip gave {back}");
}

// =============================================================================
// comparison — known-value checks
// =============================================================================

#[test]
fn correlation_identical_is_one() {
    let a = sine_f64(440.0, 100, 1.0);
    assert!((correlation(&a, &a).unwrap() - 1.0).abs() < 1e-9);
}

#[test]
fn correlation_negated_is_minus_one() {
    let a = mono_f64(vec![0.1, -0.4, 0.8, -0.2, 0.5, -0.9]);
    let neg = mono_f64(vec![-0.1, 0.4, -0.8, 0.2, -0.5, 0.9]);
    let c = correlation(&a, &neg).unwrap();
    assert!((c + 1.0).abs() < 1e-9, "negated correlation was {c}");
}

#[test]
fn mse_identical_is_zero() {
    let a = sine_f64(440.0, 100, 0.7);
    assert!(mse(&a, &a).unwrap() < 1e-12);
}

#[test]
fn mse_known_constant_error() {
    // Constant offset of 0.1 → MSE = 0.01.
    let a = mono_f64(vec![0.0; 64]);
    let b = mono_f64(vec![0.1; 64]);
    let m = mse(&a, &b).unwrap();
    assert!((m - 0.01).abs() < 1e-9, "mse was {m}, expected 0.01");
}

#[test]
fn snr_identical_signal_is_infinite() {
    // noise == signal subtracted from itself elsewhere; here noise power is the
    // signal itself, so SNR of a signal against zero noise is +inf.
    let signal = sine_f64(440.0, 100, 0.8);
    let zero = mono_f64(vec![0.0; signal.as_slice().unwrap().len()]);
    let s = snr(&signal, &zero).unwrap();
    assert!(s.is_infinite() && s > 0.0, "snr with zero noise was {s}");
}

#[test]
fn psnr_known_value_20db() {
    // Reference peak 1.0, constant error 0.1 → MSE 0.01 → PSNR = 10*log10(1/0.01)
    // = 20 dB.
    let reference = mono_f64({
        let mut v = vec![0.0; 64];
        v[0] = 1.0; // establishes peak == 1.0
        v
    });
    let test = mono_f64({
        let mut v = vec![0.1; 64];
        v[0] = 1.1; // same 0.1 error on the peak sample
        v
    });
    let p = psnr(&reference, &test).unwrap();
    assert!((p - 20.0).abs() < 0.2, "psnr was {p}, expected ~20 dB");
}

#[test]
fn psnr_identical_is_infinite() {
    let a = sine_f64(440.0, 100, 0.5);
    assert!(psnr(&a, &a).unwrap().is_infinite());
}

#[test]
fn segmental_snr_high_for_clean_signal() {
    // Signal energy >> noise energy → high (clamped) segmental SNR.
    let signal = sine_f64(440.0, 200, 0.9);
    let noise = sine_f64(440.0, 200, 0.001);
    let seg = segmental_snr(&signal, &noise, core::num::NonZeroUsize::new(256).unwrap()).unwrap();
    assert!(seg > 20.0, "segmental SNR was {seg}, expected high");
}

#[test]
fn per_channel_matches_aggregate_for_mono() {
    let a = mono_f64(vec![0.1, -0.4, 0.8, -0.2, 0.5, -0.9]);
    let b = mono_f64(vec![0.2, -0.3, 0.7, -0.1, 0.6, -0.8]);

    let corr_pc = correlation_per_channel(&a, &b).unwrap();
    assert_eq!(corr_pc.len(), 1);
    assert!((corr_pc[0] - correlation(&a, &b).unwrap()).abs() < 1e-12);

    let mse_pc = mse_per_channel(&a, &b).unwrap();
    assert_eq!(mse_pc.len(), 1);
    assert!((mse_pc[0] - mse(&a, &b).unwrap()).abs() < 1e-12);
}

#[test]
fn comparison_rejects_mismatched_lengths() {
    let a = mono_f64(vec![0.1, 0.2, 0.3, 0.4]);
    let b = mono_f64(vec![0.1, 0.2]);
    assert!(correlation(&a, &b).is_err());
    assert!(mse(&a, &b).is_err());
}

// =============================================================================
// detection — known-value checks
// =============================================================================

#[test]
fn dynamic_range_of_full_amplitude_sine() {
    // A sine's crest factor is 20*log10(peak/rms) = 20*log10(sqrt(2)) ≈ 3.01 dB.
    let audio = sine_f64(440.0, 500, 1.0);
    let (peak, rms, dr_db) = detect_dynamic_range(&audio).unwrap();
    assert!((peak - 1.0).abs() < 1e-3, "peak was {peak}");
    assert!(rms > 0.0, "rms should be positive");
    assert!(
        (dr_db - 3.0103).abs() < 0.05,
        "dynamic range was {dr_db} dB, expected ~3.01 dB"
    );
}

#[test]
fn dynamic_range_known_small_vector() {
    let audio = mono_f64(vec![0.1, 0.5, 1.0, 0.2, 0.8]);
    let (peak, rms, dr_db) = detect_dynamic_range(&audio).unwrap();
    assert!((peak - 1.0).abs() < 1e-12, "peak was {peak}");
    assert!(rms > 0.0 && rms < 1.0);
    assert!(dr_db > 0.0, "dynamic range should be positive, was {dr_db}");
}

#[test]
fn detect_fundamental_on_known_tone() {
    // detect_fundamental_frequency uses autocorrelation, which on a pure sine
    // can lock onto a subharmonic (the crate's own tests accept this), so we
    // assert it finds *something* and that it's the fundamental or a low-order
    // subharmonic of 440 Hz.
    let audio = sine_f64(440.0, 1000, 1.0);
    let detected = detect_fundamental_frequency(&audio).unwrap();
    assert!(detected.is_some(), "no fundamental found for 440 Hz tone");
    let hz = detected.unwrap();
    let ok_440 = (hz - 440.0).abs() < 15.0;
    let ok_220 = (hz - 220.0).abs() < 15.0;
    let ok_110 = (hz - 110.0).abs() < 15.0;
    assert!(
        ok_440 || ok_220 || ok_110,
        "detected {hz} Hz, expected 440 (or a low subharmonic 220/110)"
    );
}

// =============================================================================
// comparison — log-spectral distance (needs transforms; runs an FFT)
// =============================================================================

#[cfg(feature = "transforms")]
#[test]
fn log_spectral_distance_identical_is_zero() {
    use audio_samples::utils::comparison::log_spectral_distance;
    let a = sine_f64(440.0, 100, 0.8);
    let lsd = log_spectral_distance(&a, &a).unwrap();
    assert!(lsd.abs() < 1e-6, "LSD of identical signals was {lsd}");
}
