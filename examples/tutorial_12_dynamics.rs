//! Tutorial 12: Dynamics: compressors and limiters.
//!
//! Accompanies the guide chapter `audio_samples::guide::c12_dynamics`.
//!
//! What this lab shows, with every number checked by an `assert!`:
//!
//! 1. The static gain computer in the dB domain: hard knee and the quadratic
//!    soft knee (continuous slope, reduction (1 - 1/R) W / 8 at the threshold).
//! 2. Attack and release are one-pole time constants: on a step the envelope
//!    reaches 63.2 % after tau and its 10-90 % rise time is tau ln 9 = 2.2 tau,
//!    measured from the library's gain-reduction trace.
//! 3. A plain-Rust compressor (envelope follower + gain computer + gain)
//!    reproduces `apply_compressor` sample for sample.
//! 4. A limiter without lookahead lets the start of a step through; with a
//!    lookahead of five attack time constants the overshoot is 0.06 dB, as
//!    predicted from the envelope formula.
//! 5. `CompressorConfig::explain` / `LimiterConfig::explain` report the same
//!    numbers.
//!
//! Run with:
//! `cargo run --release --example tutorial_12_dynamics --features educational,full_no_plotting`

use audio_samples::educational::ExplainFor;
use audio_samples::operations::dynamic_range::{
    EnvelopeFollower, calculate_compression_gain, calculate_limiting_gain,
};
use audio_samples::operations::types::{DynamicRangeMethod, KneeType};
use audio_samples::{
    AudioDynamicRange, AudioSampleResult, AudioSamples, CompressorConfig, LimiterConfig,
    sample_rate,
};
use non_empty_slice::{NonEmptySlice, NonEmptyVec};

const FS: f64 = 48_000.0;

fn db(x: f64) -> f64 {
    20.0 * x.log10()
}

/// Piecewise-constant test signal: (level, duration in ms) segments.
fn steps(segments: &[(f64, f64)]) -> AudioSamples<'static, f64> {
    let x: Vec<f64> = segments
        .iter()
        .flat_map(|&(v, ms)| std::iter::repeat_n(v, (ms * FS / 1000.0) as usize))
        .collect();
    AudioSamples::from_mono_vec(NonEmptyVec::new(x).unwrap(), sample_rate!(48000))
}

/// Static curve written out (Giannoulis, Massberg & Reiss 2012).
fn my_reduction(level: f64, t: f64, r: f64, w: f64) -> f64 {
    let o = level - t;
    let slope = 1.0 - 1.0 / r;
    if w > 0.0 && 2.0 * o.abs() <= w {
        slope * (o + w / 2.0).powi(2) / (2.0 * w)
    } else if o > 0.0 {
        slope * o
    } else {
        0.0
    }
}

fn compressor(threshold_db: f64, ratio: f64, knee: KneeType, width: f64) -> CompressorConfig {
    let mut c = CompressorConfig::new();
    c.threshold_db = threshold_db;
    c.ratio = ratio;
    c.knee_type = knee;
    c.knee_width_db = width;
    c.attack_ms = 5.0;
    c.release_ms = 50.0;
    c.detection_method = DynamicRangeMethod::Peak;
    c.makeup_gain_db = 0.0;
    c.lookahead_ms = 0.0;
    c
}

fn main() -> AudioSampleResult<()> {
    println!("== Tutorial 12: dynamics ==\n");

    // ---------------------------------------------------------------- 1
    println!("1. Static curve, T = -20 dBFS, R = 4:1");
    println!("   L (dBFS)   hard: Y (dBFS)   soft W = 10 dB: Y (dBFS)");
    for l in [-40.0, -25.0, -22.0, -20.0, -18.0, -15.0, -8.0, 0.0] {
        let hard = l - calculate_compression_gain(l, -20.0, 4.0, KneeType::Hard, 0.0);
        let soft = l - calculate_compression_gain(l, -20.0, 4.0, KneeType::Soft, 10.0);
        println!("   {l:>7.1}    {hard:>10.3}       {soft:>10.3}");
        assert!((l - hard - my_reduction(l, -20.0, 4.0, 0.0)).abs() < 1e-12);
        assert!((l - soft - my_reduction(l, -20.0, 4.0, 10.0)).abs() < 1e-12);
    }
    // Hard knee: -8 dBFS in -> -20 + 12/4 = -17 dBFS out.
    assert!(
        (calculate_compression_gain(-8.0, -20.0, 4.0, KneeType::Hard, 0.0) - 9.0).abs() < 1e-12
    );
    // Soft knee at the threshold: (1 - 1/R) W / 8 = 0.9375 dB.
    assert!(
        (calculate_compression_gain(-20.0, -20.0, 4.0, KneeType::Soft, 10.0) - 0.9375).abs()
            < 1e-12
    );
    // Continuous slope at the knee edges: 1 below (reduction slope 0), 1/R above.
    let d = 1e-6;
    let slope = |l: f64| {
        let y = |l: f64| l - calculate_compression_gain(l, -20.0, 4.0, KneeType::Soft, 10.0);
        (y(l + d) - y(l - d)) / (2.0 * d)
    };
    assert!((slope(-25.0) - 1.0).abs() < 1e-5 && (slope(-15.0) - 0.25).abs() < 1e-5);
    // The trait method returns the same curve as output levels (+ makeup gain).
    let audio = steps(&[(0.5, 1.0)]);
    let levels = [-30.0, -20.0, -8.0];
    let curve = audio.get_compression_curve(
        &compressor(-20.0, 4.0, KneeType::Hard, 0.0),
        NonEmptySlice::new(&levels).unwrap(),
    )?;
    println!("   get_compression_curve(-30, -20, -8) = {curve:?}");
    assert_eq!(curve, vec![-30.0, -20.0, -17.0]);
    // Limiter curve (R -> infinity): output never above the ceiling.
    for l in [-3.0, -1.0, 0.0, 6.0] {
        let y = l - calculate_limiting_gain(l, -1.0, KneeType::Soft, 2.0);
        assert!(y <= -1.0 + 1e-12);
    }
    println!();

    // ---------------------------------------------------------------- 2
    println!(
        "2. Ballistics on a step: -40 dBFS -> 0 dBFS -> -40 dBFS (attack 5 ms, release 50 ms)"
    );
    let cfg = compressor(-20.0, 4.0, KneeType::Hard, 0.0);
    let x = steps(&[(0.01, 100.0), (1.0, 200.0), (0.01, 300.0)]);
    let gr = x.get_gain_reduction(&cfg)?;
    // Invert the hard-knee curve above the threshold to recover the envelope:
    // r = (1 - 1/R)(L - T)  =>  L = T + r / (1 - 1/R).
    let env = |n: usize| 10f64.powf((-20.0 + gr[n] / 0.75) / 20.0);
    let on = 4_800; // step up at 100 ms
    let first_at = |level: f64| {
        (on..on + 4_800)
            .find(|&n| gr[n] > 0.0 && env(n) >= level)
            .unwrap()
    };
    let (lo, hi) = (0.01, 1.0);
    let n63 = first_at(lo + (1.0 - (-1f64).exp()) * (hi - lo));
    let n10 = first_at(lo + 0.1 * (hi - lo));
    let n90 = first_at(lo + 0.9 * (hi - lo));
    // Samples are counted from the first sample of the new level, which already
    // contributes one step of the recursion: e[on + k] reaches the level after k + 1 updates.
    let t63 = (n63 - on + 1) as f64 / FS * 1000.0;
    let rise = (n90 - n10) as f64 / FS * 1000.0;
    println!("   63.2 % of the step after {t63:.3} ms (tau_A = 5 ms)");
    println!(
        "   10-90 % rise time {rise:.3} ms (tau_A ln 9 = {:.3} ms)",
        5.0 * 9f64.ln()
    );
    // One sample is 0.021 ms, so the measured times are within 0.03 ms.
    assert!((t63 - 5.0).abs() < 0.03);
    assert!((rise - 5.0 * 9f64.ln()).abs() < 0.03);
    // Steady state: envelope = 1 -> reduction 0.75 * 20 = 15 dB.
    assert!((gr[on + 4_000] - 15.0).abs() < 1e-6);

    // Release, measured directly on the EnvelopeFollower (the compressor's
    // reduction is zero below the threshold, so it cannot show the whole decay).
    let mut f = EnvelopeFollower::new(5.0, 50.0, FS, DynamicRangeMethod::Peak);
    let xs = x.as_slice().unwrap();
    let e: Vec<f64> = xs.iter().map(|&v| f.process(v)).collect();
    let off = 14_400; // step down at 300 ms
    let target = lo + (-1f64).exp() * (e[off - 1] - lo);
    let n37 = (off..xs.len()).find(|&n| e[n] <= target).unwrap();
    let t_rel = (n37 - off + 1) as f64 / FS * 1000.0;
    println!("   release: down to 36.8 % of the step after {t_rel:.3} ms (tau_R = 50 ms)");
    assert!((t_rel - 50.0).abs() < 0.03);
    println!();

    // ---------------------------------------------------------------- 3
    println!("3. How it works: follower + gain computer + gain, by hand");
    let soft = compressor(-20.0, 4.0, KneeType::Soft, 6.0);
    let y_lib = x.apply_compressor(&soft)?;
    let (aa, ar) = ((-1.0 / (0.005 * FS)).exp(), (-1.0 / (0.050 * FS)).exp());
    let mut env_state = 0.0f64;
    let y_mine: Vec<f64> = xs
        .iter()
        .map(|&v| {
            let d = v.abs();
            let a = if d > env_state { aa } else { ar };
            env_state = a * env_state + (1.0 - a) * d;
            let level = if env_state > 0.0 {
                db(env_state)
            } else {
                -80.0
            };
            v * 10f64.powf(-my_reduction(level, -20.0, 4.0, 6.0) / 20.0)
        })
        .collect();
    let err = y_mine
        .iter()
        .zip(y_lib.as_slice().unwrap())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("   max |mine - apply_compressor| = {err:.2e}");
    // Identical operations in the same order: only last-bit differences in exp/log.
    assert!(err < 1e-12);
    println!();

    // ---------------------------------------------------------------- 4
    println!("4. Limiter: ceiling -6 dBFS, attack 1 ms, on a step from silence to 0 dBFS");
    let burst = steps(&[(0.0, 20.0), (1.0, 50.0)]);
    let start = 960;
    for lookahead_ms in [0.0, 5.0] {
        let lim = LimiterConfig::new(
            -6.0,
            1.0,
            100.0,
            KneeType::Hard,
            0.0,
            DynamicRangeMethod::Peak,
            lookahead_ms,
            false,
        );
        let y = burst.apply_limiter(&lim)?;
        let ys = y.as_slice().unwrap();
        let peak = ys.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let overshoot = db(peak) + 6.0;
        println!(
            "   lookahead {lookahead_ms} ms: output peak {:.4} dBFS (overshoot {overshoot:.4} dB)",
            db(peak)
        );
        if lookahead_ms == 0.0 {
            // The first sample of the step passes unreduced: the envelope is
            // still 1 - a = 0.02 (-33.7 dBFS), below the ceiling.
            assert!((ys[start] - 1.0).abs() < 1e-12);
            assert!((overshoot - 6.0).abs() < 1e-9);
        } else {
            // With N_L = 240 samples the reduction applied to the first sample is
            // the one computed 240 samples later: envelope 1 - a^241.
            let a: f64 = (-1.0 / (0.001 * FS)).exp();
            let predicted = -db(1.0 - a.powi(241));
            println!("   predicted overshoot -20 log10(1 - a^241) = {predicted:.4} dB");
            assert!((overshoot - predicted).abs() < 1e-9);
            assert!((predicted - 0.0575).abs() < 1e-3);
        }
    }
    println!();

    // ---------------------------------------------------------------- 5
    println!("5. Explain\n");
    let doc = x.explain(&cfg);
    println!("{doc}");
    let rise = doc.quantity("Attack 10-90% rise time").unwrap().value;
    assert!((rise - 5.0 * 9f64.ln()).abs() < 1e-9); // 10.99 ms
    assert!(
        (doc.quantity("Attack coefficient").unwrap().value - (-1.0 / 240.0f64).exp()).abs() < 1e-12
    );
    assert!((doc.quantity("Gain reduction at 0 dBFS").unwrap().value - 15.0).abs() < 1e-9);
    assert!(doc.has_warning_containing("No lookahead"));
    let lim_doc = x.explain(&LimiterConfig::transparent());
    assert!(
        lim_doc
            .quantity("Gain reduction for a 0 dBFS envelope")
            .is_some()
    );

    println!("All checks passed.");
    Ok(())
}
