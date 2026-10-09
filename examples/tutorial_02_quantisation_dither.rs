//! Tutorial 02: Quantisation and dither.
//!
//! Accompanies the guide chapter `audio_samples::guide::c02_quantisation_dither`
//! (`documentation/guide/02_quantisation_dither.md`).
//!
//! What it teaches:
//! 1. the crate's (asymmetric) integer/float convention and the resulting step
//!    sizes;
//! 2. the signal-to-quantisation-noise ratio (SQNR) of a full-scale sine,
//!    measured at several word lengths with a plain-Rust requantiser and at
//!    16 bits through the crate's `i16` conversion, against `6.02 b + 1.76` dB;
//! 3. triangular (TPDF) dither, implemented here because the crate has none,
//!    turning signal-dependent distortion of a quiet sine into benign noise.
//!
//! Every number printed is also checked with `assert!`.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example tutorial_02_quantisation_dither --features educational,full_no_plotting
//! ```

use std::num::NonZeroUsize;
use std::time::Duration;

use audio_samples::educational::topics;
use audio_samples::utils::comparison::snr;
use audio_samples::{
    AudioSamples, AudioTransforms, AudioTypeConversion, sample_rate, sine_wave, white_noise,
};
use ndarray::Array1;

/// Requantise `x` to `b` bits with the crate's convention: negative values
/// scale by 2^(b-1), non-negative ones by 2^(b-1) - 1, round half away from
/// zero, then map back to float the same way.
fn requantise(x: f64, b: u32) -> f64 {
    let half = 2f64.powi(b as i32 - 1);
    let x = x.clamp(-1.0, 1.0);
    if x < 0.0 {
        (x * half).round() / half
    } else {
        (x * (half - 1.0)).round() / (half - 1.0)
    }
}

fn power(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64
}

fn mono(v: Vec<f64>, fs: std::num::NonZeroU32) -> AudioSamples<'static, f64> {
    AudioSamples::new_mono(Array1::from(v), fs).expect("non-empty")
}

fn main() {
    println!("=== Tutorial 02: quantisation and dither ===\n");

    // ------------------------------------------------------------------
    // 1. The convention.
    // ------------------------------------------------------------------
    println!("1. Integer <-> float convention (i16)");
    let edge = mono(vec![-1.0, -0.5, 0.0, 0.5, 1.0], sample_rate!(16000));
    let codes = edge.to_format::<i16>();
    let codes = codes.as_slice().expect("contiguous").to_vec();
    println!("   [-1, -0.5, 0, 0.5, 1] -> {codes:?}");
    assert_eq!(codes, vec![-32768, -16384, 0, 16384, 32767]);
    let d_neg = 1.0 / 32768.0;
    let d_pos = 1.0 / 32767.0;
    println!("   step below zero 1/32768 = {d_neg:.6e}, above zero 1/32767 = {d_pos:.6e}");
    let doc16 = topics::quantisation(16);
    assert_eq!(
        doc16
            .quantity("Step size (negative half)")
            .expect("q")
            .value,
        d_neg
    );
    assert!(
        (doc16
            .quantity("Step size (positive half)")
            .expect("q")
            .value
            - d_pos)
            .abs()
            < 1e-18
    );

    // ------------------------------------------------------------------
    // 2. SQNR of a full-scale sine.
    // ------------------------------------------------------------------
    println!("\n2. SQNR of a full-scale 997 Hz sine at 48 kHz (1 s)");
    let fs = sample_rate!(48000);
    let x_audio = sine_wave::<f64>(997.0, Duration::from_secs(1), fs, 1.0);
    let x = x_audio.as_slice().expect("contiguous").to_vec();
    let p_signal = power(&x);
    assert!((p_signal - 0.5).abs() < 1e-3);
    println!("   bits  measured  6.02b+1.76   (dB)");
    for b in [4u32, 8, 12, 16] {
        let err: Vec<f64> = x.iter().map(|&v| requantise(v, b) - v).collect();
        let measured = 10.0 * (p_signal / power(&err)).log10();
        let rule = 6.02 * f64::from(b) + 1.76;
        println!("   {b:>4}  {measured:>8.2}  {rule:>10.2}");
        // The uniform-error model behind the rule is only approximate at 4
        // bits (measured: 0.22 dB off); from 12 bits up it holds to 0.05 dB.
        let tol = if b >= 12 { 0.05 } else { 0.25 };
        assert!((measured - rule).abs() < tol, "b={b}: {measured} vs {rule}");
    }

    // The same at 16 bits through the crate's i16 conversion.
    let back = x_audio.to_format::<i16>().to_format::<f64>();
    let back_s = back.as_slice().expect("contiguous");
    let max_dev = x
        .iter()
        .zip(back_s)
        .map(|(&v, &q)| (q - requantise(v, 16)).abs())
        .fold(0.0, f64::max);
    assert!(
        max_dev < 1e-15,
        "crate i16 round trip == plain-Rust requantiser"
    );
    let noise: Vec<f64> = back_s.iter().zip(&x).map(|(q, v)| q - v).collect();
    let sqnr_crate = snr(&x_audio, &mono(noise, fs)).expect("snr");
    let sqnr_explain = doc16.quantity("SQNR (full-scale sine)").expect("q").value;
    println!(
        "   crate i16 round trip: SQNR = {sqnr_crate:.2} dB (topics::quantisation(16) says {sqnr_explain:.2} dB)"
    );
    assert!((sqnr_crate - sqnr_explain).abs() < 0.05);
    assert!((sqnr_explain - 98.09).abs() < 0.01);

    // ------------------------------------------------------------------
    // 3. TPDF dither on a quiet sine, through the crate's i16 conversion.
    // ------------------------------------------------------------------
    println!("\n3. A 997 Hz sine of amplitude 2.5 LSB at 16 kHz, quantised to i16");
    let fs = sample_rate!(16000);
    let lsb = 1.0 / 32768.0;
    let quiet = sine_wave::<f64>(997.0, Duration::from_secs(1), fs, 2.5 * lsb);
    let q_s = quiet.as_slice().expect("contiguous").to_vec();
    let n = q_s.len();
    assert_eq!(n, 16_000);

    // TPDF dither = sum of two independent uniform variables on [-Δ/2, Δ/2).
    // The crate's white_noise is uniform on [-A, A), so A = Δ/2.
    let u1 = white_noise::<f64>(Duration::from_secs(1), fs, lsb / 2.0, Some(1));
    let u2 = white_noise::<f64>(Duration::from_secs(1), fs, lsb / 2.0, Some(2));
    let dither: Vec<f64> = u1
        .as_slice()
        .expect("contiguous")
        .iter()
        .zip(u2.as_slice().expect("contiguous"))
        .map(|(a, b)| a + b)
        .collect();
    // TPDF variance is 2 · Δ²/12 = Δ²/6.
    let dither_ratio = power(&dither) / (lsb * lsb / 6.0);
    assert!((dither_ratio - 1.0).abs() < 0.03);

    let quantise_i16 = |v: Vec<f64>| -> Vec<f64> {
        mono(v, fs)
            .to_format::<i16>()
            .to_format::<f64>()
            .as_slice()
            .expect("contiguous")
            .to_vec()
    };
    let plain = quantise_i16(q_s.clone());
    let dithered = quantise_i16(q_s.iter().zip(&dither).map(|(a, d)| a + d).collect());
    let e_plain: Vec<f64> = plain.iter().zip(&q_s).map(|(y, x)| y - x).collect();
    let e_dith: Vec<f64> = dithered.iter().zip(&q_s).map(|(y, x)| y - x).collect();

    // Total error power relative to Δ²/12.
    let r_plain = power(&e_plain) / (lsb * lsb / 12.0);
    let r_dith = power(&e_dith) / (lsb * lsb / 12.0);
    println!("   error power / (Δ²/12): undithered {r_plain:.2}, TPDF-dithered {r_dith:.2}");
    // With TPDF dither the error power is Δ²/12 + Δ²/6 = Δ²/4, i.e. 3x.
    assert!((r_dith - 3.0).abs() < 0.1);

    // Distortion: the error spectrum at the 3rd harmonic (2991 Hz, bin 2991
    // of a 16000-point DFT) relative to the error's average power per bin.
    let spur_db = |e: &[f64]| {
        let a = mono(e.to_vec(), fs);
        let mag = a
            .rfft(NonZeroUsize::new(n).expect("n > 0"))
            .expect("rfft")
            .row(0)
            .to_vec();
        let mean_pow = mag.iter().map(|m| m * m).sum::<f64>() / mag.len() as f64;
        let h3 = mag[3 * 997];
        10.0 * (h3 * h3 / mean_pow).log10()
    };
    let spur_plain = spur_db(&e_plain);
    let spur_dith = spur_db(&e_dith);
    println!(
        "   3rd-harmonic bin of the error re mean bin power: undithered {spur_plain:+.1} dB, dithered {spur_dith:+.1} dB"
    );
    assert!(spur_plain > 30.0);
    assert!(spur_dith < 3.0);

    // Correlation between error and signal: the error of plain rounding is a
    // deterministic function of the input; with TPDF dither its first two
    // moments are independent of the input.
    let corr = |e: &[f64]| {
        let num: f64 = e.iter().zip(&q_s).map(|(a, b)| a * b).sum();
        num / (power(e).sqrt() * power(&q_s).sqrt() * n as f64)
    };
    let c_plain = corr(&e_plain);
    let c_dith = corr(&e_dith);
    println!("   correlation(error, signal): undithered {c_plain:+.3}, dithered {c_dith:+.3}");
    assert!(c_plain < -0.5);
    assert!(c_dith.abs() < 0.03);

    // ------------------------------------------------------------------
    // 4. Ask the library to explain.
    // ------------------------------------------------------------------
    println!("\n4. topics::quantisation(16)\n");
    let rule = doc16.quantity("Rule of thumb").expect("q").value;
    let dithered_sqnr = doc16.quantity("SQNR with TPDF dither").expect("q").value;
    assert!((sqnr_explain - dithered_sqnr - 10.0 * 3f64.log10()).abs() < 1e-9);
    assert!((rule - 98.09).abs() < 0.01);
    println!("{doc16}");
    println!("All assertions passed.");
}
