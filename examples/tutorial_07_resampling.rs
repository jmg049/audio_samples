//! Tutorial 07: Resampling.
//!
//! Accompanies the guide chapter `audio_samples::guide::c07_resampling`.
//!
//! What this lab shows, with every number checked by an `assert!`:
//!
//! 1. Downsampling 48 kHz -> 16 kHz by naive decimation (keep every third
//!    sample, written by hand) turns a 10 kHz tone into a full-level 6 kHz
//!    alias (|10 000 - 16 000| = 6 000 Hz).
//! 2. Every `ResamplingQuality` tier low-pass filters first and suppresses
//!    that alias; the lab measures the alias rejection in dB, and the
//!    pass-band gain at 1 kHz.
//! 3. The output is time-aligned (no delay): an impulse at input sample 3000
//!    appears at output sample 1000, and the 1 kHz tone has no phase shift.
//! 4. A plain-Rust windowed-sinc decimator (low-pass, then keep every third
//!    sample) reproduces the library's behaviour: an alias rejection between
//!    `Fast` and `Medium`, and the same pass-band output as `High`.
//! 5. Upsampling 16 kHz -> 48 kHz with linear interpolation (`Fast`) droops
//!    the pass band and leaves spectral images; the sinc tiers do not.
//!
//! Run with:
//! `cargo run --release --example tutorial_07_resampling --features educational,full_no_plotting`

use std::f64::consts::PI;
use std::time::Duration;

use audio_samples::educational::topics;
use audio_samples::operations::types::ResamplingQuality;
use audio_samples::{AudioSampleResult, AudioSamples, resample, sample_rate, sine_wave};

/// Amplitude of the component at `f` Hz, from one DFT bin over `x`.
/// Exact (no leakage) when `x` holds a whole number of periods of every
/// component, which the lab arranges.
fn tone_amplitude(x: &[f64], f: f64, fs: f64) -> f64 {
    let (mut re, mut im) = (0.0, 0.0);
    for (n, &v) in x.iter().enumerate() {
        let ph = 2.0 * PI * f * n as f64 / fs;
        re += v * ph.cos();
        im -= v * ph.sin();
    }
    2.0 * (re * re + im * im).sqrt() / x.len() as f64
}

/// Phase (radians) of the component at `f` Hz, from one DFT bin.
fn tone_phase(x: &[f64], f: f64, fs: f64) -> f64 {
    let (mut re, mut im) = (0.0, 0.0);
    for (n, &v) in x.iter().enumerate() {
        let ph = 2.0 * PI * f * n as f64 / fs;
        re += v * ph.cos();
        im -= v * ph.sin();
    }
    im.atan2(re)
}

/// 0.5 sin(2 pi f n / fs) for the 4000 output samples.
fn ideal_tone(f: f64, fs: f64) -> Vec<f64> {
    (0..4_000)
        .map(|n| 0.5 * (2.0 * PI * f * n as f64 / fs).sin())
        .collect()
}

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-300).log10()
}

fn samples(a: &AudioSamples<'_, f64>) -> Vec<f64> {
    a.as_slice().expect("mono, contiguous").to_vec()
}

/// Windowed-sinc low-pass FIR, `2 * half + 1` taps, cutoff `fc` (fraction of fs).
fn sinc_lowpass(fc: f64, half: usize) -> Vec<f64> {
    let m = 2 * half;
    (0..=m)
        .map(|i| {
            let t = i as f64 - half as f64;
            let sinc = if t == 0.0 {
                2.0 * fc
            } else {
                (2.0 * PI * fc * t).sin() / (PI * t)
            };
            // Blackman window.
            let w = 0.42 - 0.5 * (2.0 * PI * i as f64 / m as f64).cos()
                + 0.08 * (4.0 * PI * i as f64 / m as f64).cos();
            sinc * w
        })
        .collect()
}

/// Decimate by `q`: centred FIR low-pass, then keep every q-th sample.
fn my_decimate(x: &[f64], q: usize, h: &[f64]) -> Vec<f64> {
    let half = h.len() / 2;
    (0..x.len().div_ceil(q))
        .map(|m| {
            let n = m * q;
            h.iter()
                .enumerate()
                .filter_map(|(k, &hk)| {
                    let idx = n as isize + half as isize - k as isize;
                    (idx >= 0 && (idx as usize) < x.len()).then(|| hk * x[idx as usize])
                })
                .sum()
        })
        .collect()
}

fn main() -> AudioSampleResult<()> {
    println!("== Tutorial 07: Resampling ==\n");
    let (fs_in, fs_out) = (48_000.0, 16_000.0);
    let dur = Duration::from_millis(250);

    // Test signal: a 1 kHz tone (kept) plus a 10 kHz tone (must be removed).
    let keep = sine_wave::<f64>(1_000.0, dur, sample_rate!(48000), 0.5);
    let drop = sine_wave::<f64>(10_000.0, dur, sample_rate!(48000), 0.5);
    let x: Vec<f64> = samples(&keep)
        .iter()
        .zip(samples(&drop))
        .map(|(a, b)| a + b)
        .collect();
    let input = AudioSamples::from_mono_vec(
        non_empty_slice::NonEmptyVec::new(x.clone()).unwrap(),
        sample_rate!(48000),
    );

    let doc = topics::resampling(fs_in, fs_out);
    println!("{doc}");
    assert!((doc.quantity("Ratio").unwrap().value - 1.0 / 3.0).abs() < 1e-12);
    // Medium's pass band ends at 0.90 x 8 kHz.
    assert!((doc.quantity("Sinc pass-band edge").unwrap().value - 7_200.0).abs() < 1e-9);
    assert!(doc.has_warning_containing("Downsampling"));

    // ---------------------------------------------------------------- 1, 2
    println!("1-2. 48 kHz -> 16 kHz, input = 0.5 sin(2pi 1k t) + 0.5 sin(2pi 10k t)");
    println!("     the 10 kHz tone is above the new Nyquist (8 kHz) and folds to 6 kHz");
    // Analyse the middle 100 ms of the output: 1600 samples hold whole periods
    // of both 1 kHz and 6 kHz, so the single-bin DFT has no leakage.
    let mid = 1_200..2_800;

    // Naive decimation, by hand: keep every third sample.
    let naive: Vec<f64> = x.iter().step_by(3).copied().collect();
    let naive_rej = db(tone_amplitude(&naive[mid.clone()], 6_000.0, fs_out) / 0.5);
    println!("      naive: 6 kHz alias at {naive_rej:+.1} dB re the 10 kHz input");
    // The alias is at full level (0 dB) ...
    assert!(naive_rej.abs() < 1e-9);
    // ... and is exactly a 6 kHz sine: sin(2 pi 10k m / 16k) = -sin(2 pi 6k m / 16k).
    for (m, v) in naive.iter().enumerate().take(100) {
        let six = -0.5 * (2.0 * PI * 6_000.0 * m as f64 / fs_out).sin();
        let one = 0.5 * (2.0 * PI * 1_000.0 * m as f64 / fs_out).sin();
        assert!((v - (one + six)).abs() < 1e-12);
    }

    let mut measured = Vec::new();
    for q in [
        ResamplingQuality::Fast,
        ResamplingQuality::Medium,
        ResamplingQuality::High,
    ] {
        let y = samples(&resample(&input, sample_rate!(16000), q)?);
        assert_eq!(y.len(), 4_000); // ceil(12000 / 3)
        let g1 = db(tone_amplitude(&y[mid.clone()], 1_000.0, fs_out) / 0.5);
        let rej = db(tone_amplitude(&y[mid.clone()], 6_000.0, fs_out) / 0.5);
        println!(
            "     {q:>6}: 1 kHz gain {g1:+.4} dB, 6 kHz alias at {rej:+.1} dB re the 10 kHz input"
        );
        measured.push((y, g1, rej));
    }
    // Every tier keeps the pass band (better than 0.001 dB) and rejects the
    // alias; the resampler computes in f64, so the levels are set by the
    // filters (0.1 dB tolerance).
    for (_, g1, _) in &measured {
        assert!(g1.abs() < 1e-3);
    }
    assert!((measured[0].2 + 90.8).abs() < 0.1); // Fast: short 70 dB design filter
    assert!((measured[1].2 + 116.7).abs() < 0.1); // Medium
    assert!((measured[2].2 + 145.5).abs() < 0.1); // High
    println!();

    // ---------------------------------------------------------------- 3
    println!("3. Time alignment");
    let mut imp = vec![0.0; 12_000];
    imp[3_000] = 1.0;
    let imp = AudioSamples::from_mono_vec(
        non_empty_slice::NonEmptyVec::new(imp).unwrap(),
        sample_rate!(48000),
    );
    for q in [
        ResamplingQuality::Fast,
        ResamplingQuality::Medium,
        ResamplingQuality::High,
    ] {
        let y = samples(&resample(&imp, sample_rate!(16000), q)?);
        let peak = (0..y.len())
            .max_by(|&a, &b| y[a].abs().total_cmp(&y[b].abs()))
            .unwrap();
        println!("     {q:>6}: impulse at input 3000 -> output peak at {peak} (3000 / 3 = 1000)");
        assert_eq!(peak, 1_000);
    }
    // Sub-sample check with the 1 kHz tone's phase: each output sample is the
    // band-limited reconstruction at the exact input position 3m, so the
    // shift is zero (to rounding error) for every tier.
    let ideal_phase = tone_phase(&ideal_tone(1_000.0, fs_out)[mid.clone()], 1_000.0, fs_out);
    for (i, q) in ["Fast", "Medium", "High"].iter().enumerate() {
        let ph = tone_phase(&measured[i].0[mid.clone()], 1_000.0, fs_out);
        let mut dphi = ph - ideal_phase;
        dphi = (dphi + PI).rem_euclid(2.0 * PI) - PI;
        let tau = dphi / (2.0 * PI * 1_000.0 / fs_out); // output frames the output leads by
        println!("     {q:>6}: shift of the 1 kHz tone {tau:+.1e} output frames");
        assert!(tau.abs() < 1e-9);
    }
    println!();

    // ---------------------------------------------------------------- 4
    println!("4. How it works: windowed-sinc decimator by 3, written out");
    let h = sinc_lowpass(0.95 * 8_000.0 / fs_in, 96);
    let mine = my_decimate(&x, 3, &h);
    let g1 = db(tone_amplitude(&mine[mid.clone()], 1_000.0, fs_out) / 0.5);
    let rej = db(tone_amplitude(&mine[mid.clone()], 6_000.0, fs_out) / 0.5);
    println!(
        "     {} taps, Blackman window, cutoff 7.6 kHz: 1 kHz gain {g1:+.4} dB, alias {rej:+.1} dB",
        h.len()
    );
    // 107.5 dB: between the library's Fast (90.8) and Medium (116.7).
    assert!(g1.abs() < 1e-3 && (rej + 107.5).abs() < 0.1);
    // Both this decimator and the library are centred (zero delay), so they
    // can be compared with the ideal sine and with each other directly.
    let lib = &measured[2].0;
    let ideal = ideal_tone(1_000.0, fs_out);
    let lib_err = mid
        .clone()
        .map(|n| (lib[n] - ideal[n]).abs())
        .fold(0.0, f64::max);
    let my_err = mid
        .clone()
        .map(|n| (mine[n] - ideal[n]).abs())
        .fold(0.0, f64::max);
    let diff = mid.map(|n| (mine[n] - lib[n]).abs()).fold(0.0, f64::max);
    println!("     max |library High - 1 kHz sine| = {lib_err:.1e}");
    println!("     max |by hand - 1 kHz sine|      = {my_err:.1e}");
    println!("     max |by hand - library High|    = {diff:.1e}");
    // Library High: 3.6e-8 (pass-band ripple and the residual alias). By hand:
    // 2.3e-6 (the Blackman window's pass-band ripple), which also bounds the
    // difference between the two.
    assert!(lib_err < 1e-7 && my_err < 3e-6 && diff < 3e-6);
    println!();

    // ---------------------------------------------------------------- 5
    println!("5. Upsampling 16 kHz -> 48 kHz: a 6 kHz tone (images at 16k +- 6k = 10k, 22k)");
    let lo = sine_wave::<f64>(6_000.0, dur, sample_rate!(16000), 0.5);
    let mid48 = 3_600..8_400; // 100 ms: whole periods of 6 kHz and 10 kHz at 48 kHz
    // Linear interpolation by 3 = zero-stuffing then the triangular FIR
    // [1, 2, 3, 2, 1] / 3, whose gain is (3 + 4 cos w + 2 cos 2w) / 3.
    let tri = |f: f64| {
        let w = 2.0 * PI * f / fs_in;
        (3.0 + 4.0 * w.cos() + 2.0 * (2.0 * w).cos()) / 3.0 / 3.0
    };
    for q in [
        ResamplingQuality::Fast,
        ResamplingQuality::Medium,
        ResamplingQuality::High,
    ] {
        let y = samples(&resample(&lo, sample_rate!(48000), q)?);
        let g6 = db(tone_amplitude(&y[mid48.clone()], 6_000.0, fs_in) / 0.5);
        let g10 = db(tone_amplitude(&y[mid48.clone()], 10_000.0, fs_in) / 0.5);
        println!("     {q:>6}: 6 kHz gain {g6:+.2} dB, image at 10 kHz {g10:+.1} dB");
        match q {
            ResamplingQuality::Fast => {
                // Theory: -3.77 dB droop and a -11.8 dB image, met to 1e-6 dB
                // (the resampler computes in f64).
                assert!((g6 - db(tri(6_000.0).abs())).abs() < 1e-6);
                assert!((g10 - db(tri(10_000.0).abs())).abs() < 1e-6);
                assert!((g6 + 3.77).abs() < 0.01 && (g10 + 11.8).abs() < 0.05);
            }
            // The sinc tiers pass 6 kHz unchanged and attenuate the image to -116.7 dB
            // (Medium) and -145.5 dB (High).
            ResamplingQuality::Medium => assert!(g6.abs() < 1e-4 && (g10 + 116.7).abs() < 0.1),
            _ => assert!(g6.abs() < 1e-4 && (g10 + 145.5).abs() < 0.1),
        }
    }
    println!("\nAll checks passed.");
    Ok(())
}
