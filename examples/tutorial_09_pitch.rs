//! Tutorial 09 lab: pitch estimation by autocorrelation and YIN.
//!
//! Accompanies the guide chapter `audio_samples::guide::c09_pitch`
//! (`documentation/guide/09_pitch.md`).
//!
//! What it does:
//!
//! 1. Measures the integer-lag resolution of autocorrelation pitch
//!    (`detect_pitch_autocorr`) and compares it with f0^2/f_s.
//! 2. Writes YIN out (difference function, cumulative-mean-normalised
//!    difference, absolute threshold, parabolic interpolation) and checks it
//!    against `detect_pitch_yin`.
//! 3. Shows the missing fundamental (the pitch is found although no energy
//!    sits at f0) and an octave error on a signal with weak odd harmonics.
//! 4. Tracks a pitch glide with `track_pitch` and shows what happens when
//!    the frame is shorter than two periods of the lowest pitch.
//! 5. Prints the explanations `topics::yin` and `topics::autocorrelation_pitch`.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example tutorial_09_pitch --features educational,full_no_plotting
//! ```

use std::f64::consts::PI;
use std::num::NonZeroUsize;
use std::time::Duration;

use audio_samples::educational::{WarningLevel, topics};
use audio_samples::operations::types::PitchDetectionMethod;
use audio_samples::{
    AudioPitchAnalysis, AudioSamples, AudioTransforms, chirp, sample_rate, sine_wave,
};
use ndarray::Array1;

const FS: f64 = 16_000.0;

fn mono(x: Vec<f64>) -> AudioSamples<'static, f64> {
    AudioSamples::new_mono(Array1::from(x), sample_rate!(16000)).expect("non-empty")
}

/// Sum of harmonics `(h, amplitude)` of `f0`, `n` samples, cosine phase.
fn harmonic_complex(f0: f64, harmonics: &[(u32, f64)], n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / FS;
            harmonics
                .iter()
                .map(|&(h, a)| a * (2.0 * PI * f64::from(h) * f0 * t).cos())
                .sum()
        })
        .collect()
}

/// YIN written out (de Cheveigne and Kawahara 2002, steps 2-5). Returns
/// (period in samples, d' values for lags 0..=max_tau + 1).
fn my_yin(x: &[f64], min_tau: usize, max_tau: usize, threshold: f64) -> (Option<f64>, Vec<f64>) {
    let n = x.len();
    let w = n - max_tau - 1; // fixed integration window
    let d: Vec<f64> = (0..=max_tau + 1)
        .map(|tau| (0..w).map(|j| (x[j] - x[j + tau]).powi(2)).sum())
        .collect();
    let mut cmnd = vec![1.0; max_tau + 2];
    let mut running = 0.0;
    for tau in 1..=max_tau + 1 {
        running += d[tau];
        if running > 0.0 {
            cmnd[tau] = d[tau] * tau as f64 / running;
        }
    }
    let Some(mut tau) = (min_tau.max(1)..=max_tau).find(|&t| cmnd[t] < threshold) else {
        return (None, cmnd);
    };
    while tau < max_tau && cmnd[tau + 1] < cmnd[tau] {
        tau += 1;
    }
    let (a, b, c) = (cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]);
    let den = a - 2.0 * b + c;
    let shift = if den.abs() > f64::EPSILON {
        (0.5 * (a - c) / den).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    (Some(tau as f64 + shift), cmnd)
}

fn main() {
    println!("== Tutorial 09: pitch by autocorrelation and YIN ==\n");

    // ------------------------------------------------------------ 1. Resolution
    println!("1. Integer-lag resolution of autocorrelation (f_s = {FS} Hz, 50 ms sine frames)");
    println!("   f0 (Hz) | period (samples) | grid step | f0^2/f_s | autocorr (Hz) | YIN (Hz)");
    // (f0, autocorrelation error, YIN error) in Hz, as quoted in the guide.
    let quoted = [
        (110.0, 0.34, 0.001),
        (220.0, -0.82, 0.014),
        (440.0, 4.44, 0.108),
        (880.0, 8.89, 1.283),
    ];
    for (f0, ac_err, yin_err) in quoted {
        let x = sine_wave::<f64>(f0, Duration::from_millis(50), sample_rate!(16000), 0.8);
        let ac = x
            .detect_pitch_autocorr(80.0, 1000.0)
            .expect("valid range")
            .expect("voiced");
        let yin = x
            .detect_pitch_yin(0.1, 80.0, 1000.0)
            .expect("valid range")
            .expect("voiced");
        let tau = FS / f0;
        let step = FS / tau.floor() - FS / tau.ceil();
        println!(
            "   {f0:7.0} | {tau:16.2} | {step:9.2} | {:8.2} | {ac:13.3} | {yin:8.3}",
            f0 * f0 / FS
        );
        // Autocorrelation returns f_s / integer lag, so it is never further
        // than half a grid step from the truth, and the grid step is close to
        // f0^2/f_s (within 4 % here).
        assert!((FS / ac).fract() == 0.0);
        assert!((ac - f0).abs() <= step / 2.0 + 1e-9);
        assert!((step - f0 * f0 / FS).abs() < 0.04 * step);
        // YIN interpolates: much finer than the grid.
        assert!((yin - f0).abs() < step / 10.0);
        assert!(((ac - f0) - ac_err).abs() < 0.005);
        assert!(((yin - f0) - yin_err).abs() < 0.0005);
    }
    let x440 = sine_wave::<f64>(440.0, Duration::from_millis(50), sample_rate!(16000), 0.8);
    let ac440 = x440.detect_pitch_autocorr(80.0, 1000.0).unwrap().unwrap();
    let yin440 = x440.detect_pitch_yin(0.1, 80.0, 1000.0).unwrap().unwrap();
    assert!((ac440 - 16_000.0 / 36.0).abs() < 1e-9); // 444.44 Hz, lag 36
    assert!((yin440 - 440.108).abs() < 5e-4);
    println!("   440 Hz: autocorr picks lag 36 -> {ac440:.2} Hz; YIN gives {yin440:.3} Hz");

    // ------------------------------------------------------------ 2. YIN
    let min_tau = (FS / 1000.0) as usize;
    let max_tau = (FS / 80.0) as usize;
    let (tau, cmnd) = my_yin(x440.as_slice().expect("mono"), min_tau, max_tau, 0.1);
    let mine = FS / tau.expect("voiced");
    println!("\n2. YIN written out (lags {min_tau}..{max_tau}, theta = 0.1)");
    println!(
        "   d'(34..=38) = [{}]",
        cmnd[34..=38]
            .iter()
            .map(|v| format!("{v:.4}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let first = (min_tau..=max_tau).find(|&t| cmnd[t] < 0.1).expect("dip");
    println!(
        "   first lag below theta: {first}; bottom of the dip: 36; interpolated period {:.4} samples",
        tau.unwrap()
    );
    assert_eq!(first, 34);
    assert_eq!(tau.unwrap().round(), 36.0);
    println!("   ours {mine:.6} Hz, detect_pitch_yin {yin440:.6} Hz");
    // Same arithmetic in the same order: identical up to rounding.
    assert!((mine - yin440).abs() < 1e-9);

    // ------------------------------------------------------------ 3. Octaves
    let n = 1600; // 0.1 s = 20 periods of 200 Hz, so FFT bins fall on harmonics
    let missing = harmonic_complex(
        200.0,
        &[(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0)],
        n,
    );
    let a = mono(missing);
    let spectrum = a.rfft(NonZeroUsize::new(n).expect("n > 0")).expect("fft");
    let (at200, at400) = (spectrum[[0, 20]], spectrum[[0, 40]]);
    let yin_missing = a
        .detect_pitch_yin(0.1, 80.0, 500.0)
        .unwrap()
        .expect("voiced");
    let ac_missing = a
        .detect_pitch_autocorr(80.0, 500.0)
        .unwrap()
        .expect("voiced");
    println!("\n3. Missing fundamental: harmonics 2..6 of 200 Hz, equal amplitudes");
    println!("   |X| at 200 Hz = {at200:.1e}, at 400 Hz = {at400:.0}: no energy at f0");
    println!(
        "   YIN {yin_missing:.2} Hz, autocorrelation {ac_missing:.2} Hz: both hear the 200 Hz period"
    );
    assert!(at200 < 1e-9 * at400);
    assert!((yin_missing - 200.0).abs() < 0.1);
    assert!((ac_missing - 200.0).abs() < 1e-9);

    let weak = harmonic_complex(
        200.0,
        &[(1, 0.1), (2, 1.0), (3, 0.1), (4, 1.0), (5, 0.1), (6, 1.0)],
        n,
    );
    let (_, c) = my_yin(&weak, 32, 200, 0.1);
    let b = mono(weak);
    let yin_01 = b
        .detect_pitch_yin(0.1, 80.0, 500.0)
        .unwrap()
        .expect("voiced");
    let yin_001 = b
        .detect_pitch_yin(0.01, 80.0, 500.0)
        .unwrap()
        .expect("voiced");
    let ac_weak = b
        .detect_pitch_autocorr(80.0, 500.0)
        .unwrap()
        .expect("voiced");
    println!(
        "   Weak odd harmonics (h = 1, 3, 5 at 0.1; h = 2, 4, 6 at 1.0), true period 80 samples:"
    );
    println!(
        "   d'(40) = {:.4} (half period), d'(80) = {:.1e} (true period)",
        c[40], c[80]
    );
    println!(
        "   YIN theta = 0.1 -> {yin_01:.2} Hz (octave error); theta = 0.01 -> {yin_001:.2} Hz"
    );
    println!("   autocorrelation -> {ac_weak:.2} Hz (octave error)");
    assert!((c[40] - 0.0198).abs() < 5e-5);
    assert!(c[80] < 1e-6);
    assert!((yin_01 - 400.0).abs() < 1.0);
    assert!((yin_001 - 200.0).abs() < 0.1);
    assert!((ac_weak - 400.0).abs() < 1e-9);

    // ------------------------------------------------------------ 4. Tracking
    println!("\n4. track_pitch on a 1 s linear glide 100 -> 300 Hz (YIN, theta 0.1, 80-400 Hz)");
    let glide = chirp::<f64>(
        100.0,
        300.0,
        Duration::from_secs(1),
        sample_rate!(16000),
        0.8,
    );
    let need = topics::yin(0.1, 80.0, 400.0, FS, None)
        .quantity("Minimum frame length")
        .expect("quantity")
        .value as usize;
    assert_eq!(need, 402);
    println!(
        "   frames must hold more than 2 floor(f_s/f_min) + 1 = {} samples ({need} or more)",
        need - 1
    );
    for (win, hop) in [(1024usize, 256usize), (512, 128), (256, 64)] {
        let track = glide
            .track_pitch(
                NonZeroUsize::new(win).expect("win"),
                NonZeroUsize::new(hop).expect("hop"),
                PitchDetectionMethod::Yin,
                0.1,
                80.0,
                400.0,
            )
            .expect("valid parameters");
        let mut max_err: f64 = 0.0;
        let mut full = 0;
        let mut full_voiced = 0;
        for frame in track.frames() {
            // PitchFrame::time is the frame START; compare with the glide's
            // frequency at the frame centre.
            if frame.time + win as f64 / FS > 1.0 {
                continue; // truncated tail frame
            }
            full += 1;
            if let Some(f) = frame.frequency {
                full_voiced += 1;
                let truth = 100.0 + 200.0 * (frame.time + win as f64 / 2.0 / FS);
                max_err = max_err.max((f - truth).abs());
            }
        }
        println!(
            "   N = {win:4} ({:2.0} ms), hop {hop:3}: {full_voiced:3} of {full:3} full frames voiced, max |error| {max_err:.2} Hz",
            1000.0 * win as f64 / FS
        );
        if win >= need {
            assert_eq!(full_voiced, full);
            assert!(max_err < 1.0, "glide error {max_err}");
        } else {
            assert_eq!(full_voiced, 0); // too short to hold two 80 Hz periods
        }
        let doc = topics::yin(0.1, 80.0, 400.0, FS, Some(win));
        assert_eq!(doc.has_warning(WarningLevel::Problem), win < need);
    }

    // ------------------------------------------------------------ 5. Explain
    let doc = topics::autocorrelation_pitch(80.0, 1000.0, FS);
    let step_hi = doc.quantity("Pitch step at f_max").expect("quantity").value;
    println!(
        "\n5. topics::autocorrelation_pitch(80, 1000, {FS}): pitch step at 1 kHz = {step_hi:.1} Hz"
    );
    assert!((step_hi - (1000.0 - FS / 17.0)).abs() < 1e-9);
    assert!(doc.has_warning_containing("octave errors"));
    println!("\n{doc}");
    println!("{}", topics::yin(0.1, 80.0, 400.0, FS, Some(256)));
}
