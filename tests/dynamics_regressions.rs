#![cfg(all(feature = "dynamic-range", feature = "envelopes"))]
//! Regression tests for dynamics processors and envelope followers.
//!
//! Every expected value comes from the textbook definition of the operation,
//! not from the library itself.

use std::f64::consts::PI;

use audio_samples::operations::dynamic_range::{
    EnvelopeFollower, calculate_compression_gain, calculate_limiting_gain,
};
use audio_samples::operations::traits::AudioEnvelopes;
use audio_samples::operations::types::{
    CompressorConfig, DynamicRangeMethod, GateConfig, KneeType, LimiterConfig,
};
use audio_samples::{AudioDynamicRange, AudioSamples, NdResult, sample_rate};
use ndarray::Array1;
use non_empty_slice::NonEmptySlice;

const FS: f64 = 48_000.0;

fn sine(freq: f64, n: usize, amp: f64) -> Vec<f64> {
    (0..n)
        .map(|i| amp * (2.0 * PI * freq * i as f64 / FS).sin())
        .collect()
}

fn mono(data: Vec<f64>) -> AudioSamples<'static, f64> {
    AudioSamples::new_mono(Array1::from_vec(data), sample_rate!(48000)).unwrap()
}

fn peak(x: &[f64]) -> f64 {
    x.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

fn db(x: f64) -> f64 {
    20.0 * x.log10()
}

/// A one-pole follower with time constant τ follows a unit step as
/// 1 − e^{−t/τ}: 0.632 after τ, > 0.99 after 5τ.
#[test]
fn attack_step_response_reaches_63_percent_after_tau() {
    let attack_ms = 1.0;
    let tau_samples = (attack_ms * 1e-3 * FS) as usize; // 48
    let mut f = EnvelopeFollower::new(attack_ms, 100.0, FS, DynamicRangeMethod::Peak);
    let env: Vec<f64> = (0..10 * tau_samples).map(|_| f.process(1.0)).collect();

    let at_tau = env[tau_samples - 1];
    assert!(
        (at_tau - (1.0 - (-1.0f64).exp())).abs() < 0.01,
        "env(τ_A) = {at_tau}, expected 0.632"
    );
    let at_5tau = env[5 * tau_samples - 1];
    assert!(at_5tau > 0.99, "env(5τ_A) = {at_5tau}, expected > 0.99");
}

/// After a downward step the envelope decays as e^{−t/τ_R}: 0.368 after τ_R.
#[test]
fn release_step_response_reaches_37_percent_after_tau() {
    let release_ms = 20.0;
    let tau_samples = (release_ms * 1e-3 * FS) as usize; // 960
    let mut f = EnvelopeFollower::new(0.1, release_ms, FS, DynamicRangeMethod::Peak);
    for _ in 0..2000 {
        f.process(1.0);
    }
    let mut last = 0.0;
    for _ in 0..tau_samples {
        last = f.process(0.0);
    }
    assert!(
        (last - (-1.0f64).exp()).abs() < 0.01,
        "env(τ_R) after release = {last}, expected 0.368"
    );
}

/// A peak follower with a short attack and long release holds close to the
/// peak amplitude of a steady sine.
#[test]
fn peak_follower_holds_sine_peak() {
    let mut f = EnvelopeFollower::new(0.1, 100.0, FS, DynamicRangeMethod::Peak);
    let x = sine(1000.0, 4800, 1.0);
    let env: Vec<f64> = x.iter().map(|&v| f.process(v)).collect();
    let tail_min = env[3840..].iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        tail_min > 0.95,
        "steady envelope min {tail_min}, expected > 0.95"
    );
}

/// Hard-knee compressor, peak detection, steady 0 dBFS sine, T = −20 dB, R = 4:
/// the static curve predicts an output level of T + (0 − T)/R = −15 dBFS.
#[test]
fn compressor_steady_state_matches_static_curve() {
    let mut cfg = CompressorConfig::new();
    cfg.threshold_db = -20.0;
    cfg.ratio = 4.0;
    cfg.attack_ms = 0.1;
    cfg.release_ms = 200.0;
    cfg.knee_type = KneeType::Hard;
    cfg.detection_method = DynamicRangeMethod::Peak;
    cfg.makeup_gain_db = 0.0;
    cfg.lookahead_ms = 0.0;

    let mut audio = mono(sine(1000.0, 24_000, 1.0));
    audio.apply_compressor_in_place(&cfg).unwrap();
    let out = audio.as_slice().unwrap();
    let steady = db(peak(&out[19_200..]));
    assert!(
        (steady - (-15.0)).abs() < 0.3,
        "steady output peak {steady} dBFS, expected -15 dBFS"
    );
}

fn burst_compressor(lookahead_ms: f64) -> Vec<f64> {
    let mut cfg = CompressorConfig::new();
    cfg.threshold_db = -20.0;
    cfg.ratio = 20.0;
    cfg.attack_ms = 0.5;
    cfg.release_ms = 100.0;
    cfg.knee_type = KneeType::Hard;
    cfg.detection_method = DynamicRangeMethod::Peak;
    cfg.makeup_gain_db = 0.0;
    cfg.lookahead_ms = lookahead_ms;

    let mut x = vec![0.0; 4800];
    x.extend(std::iter::repeat_n(0.9, 4800));
    let mut audio = mono(x);
    audio.apply_compressor_in_place(&cfg).unwrap();
    audio.as_slice().unwrap().to_vec()
}

/// With lookahead ≥ 5 τ_A the gain reduction is in place when the burst
/// arrives, the output stays time-aligned with the input, and the result
/// differs from the zero-lookahead output.
#[test]
fn lookahead_attenuates_burst_onset_without_shifting_it() {
    let onset = 4800;
    let no_la = burst_compressor(0.0);
    let la = burst_compressor(5.0); // 5 ms = 10 τ_A

    // Without lookahead the first burst sample passes almost unchanged.
    assert!(no_la[onset] > 0.8, "no-lookahead onset {}", no_la[onset]);

    // With lookahead it is already attenuated to within 1 dB of the static
    // target T + (L − T)/R for L = 20·log10(0.9).
    let level = db(0.9);
    let target = -20.0 + (level + 20.0) / 20.0;
    let onset_db = db(la[onset]);
    assert!(
        (onset_db - target).abs() < 1.0,
        "lookahead onset {onset_db} dB, static target {target} dB"
    );

    // Latency is compensated: silence stays silent up to the onset.
    assert!(la[..onset].iter().all(|&v| v == 0.0));
    assert!(la[onset] > 0.0);
}

fn limiter_steady_peak_db(cfg: &LimiterConfig, x: Vec<f64>, skip: usize) -> f64 {
    let mut audio = mono(x);
    audio.apply_limiter_in_place(cfg).unwrap();
    db(peak(&audio.as_slice().unwrap()[skip..]))
}

/// For a signal whose peak detector input is constant (a ±1 square wave,
/// |x| = 1), the envelope converges to 1 and the hard-knee limiter output
/// settles exactly at the ceiling: 20·log10(g·1) = C.
#[test]
fn limiter_settles_at_ceiling_for_constant_level() {
    let cfg = LimiterConfig::broadcast(); // hard knee, C = -1 dBFS
    let square: Vec<f64> = (0..48_000)
        .map(|i| if (i / 24) % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let out = limiter_steady_peak_db(&cfg, square, 4800);
    assert!(
        (out - cfg.ceiling_db).abs() < 0.01,
        "steady peak {out} dBFS, ceiling {} dBFS",
        cfg.ceiling_db
    );
}

/// A 0 dBFS, 1 kHz sine through the `transparent` and `mastering` presets
/// comes out at or below the ceiling after the attack transient (sample
/// peaks), and not pinned at 0.00 dBFS by a stuck envelope follower.
#[test]
fn limiter_presets_hold_ceiling_for_full_scale_sine() {
    for (name, cfg) in [
        ("transparent", LimiterConfig::transparent()),
        ("mastering", LimiterConfig::mastering()),
    ] {
        let out = limiter_steady_peak_db(&cfg, sine(1000.0, 48_000, 1.0), 4800);
        assert!(
            out <= cfg.ceiling_db + 0.05,
            "{name}: steady peak {out} dBFS, ceiling {} dBFS",
            cfg.ceiling_db
        );
        assert!(
            out > cfg.ceiling_db - 1.5,
            "{name}: over-limited, steady peak {out} dBFS"
        );
    }
    // `broadcast` (attack 0.5 ms) is limiting but its smoothed envelope sags
    // between the sine peaks, so it overshoots by a fraction of a dB.
    let cfg = LimiterConfig::broadcast();
    let out = limiter_steady_peak_db(&cfg, sine(1000.0, 48_000, 1.0), 4800);
    assert!(out < -0.5, "broadcast: steady peak {out} dBFS");
}

/// Quadratic soft knee (Giannoulis, Massberg & Reiss 2012), T = −20, R = 4,
/// W = 12: y(T − W/2) = T − W/2, y(T) = T + (1/R − 1)·W/8,
/// y(T + W/2) = T + W/(2R); the curve is monotone and C¹.
#[test]
fn soft_knee_static_curve_matches_quadratic_knee() {
    let (t, r, w) = (-20.0, 4.0, 12.0);
    let y = |x: f64| x - calculate_compression_gain(x, t, r, KneeType::Soft, w);

    assert!((y(t - w / 2.0) - (t - w / 2.0)).abs() < 1e-12);
    assert!((y(t) - (t + (1.0 / r - 1.0) * w / 8.0)).abs() < 1e-12); // -21.125
    assert!((y(t + w / 2.0) - (t + w / (2.0 * r))).abs() < 1e-12); // -18.5
    assert!((y(0.0) - (t + (0.0 - t) / r)).abs() < 1e-12); // -15

    // Monotone non-decreasing on a fine grid.
    let xs: Vec<f64> = (0..=6000).map(|i| -60.0 + i as f64 * 0.01).collect();
    for pair in xs.windows(2) {
        assert!(
            y(pair[1]) >= y(pair[0]) - 1e-12,
            "non-monotone at {}",
            pair[0]
        );
    }

    // Slope is 1 at the lower edge and 1/R at the upper edge (C¹).
    let h = 1e-6;
    let slope = |x: f64| (y(x + h) - y(x - h)) / (2.0 * h);
    assert!((slope(t - w / 2.0) - 1.0).abs() < 1e-4);
    assert!((slope(t + w / 2.0) - 1.0 / r).abs() < 1e-4);

    // The trait method gives the same static curve.
    let mut cfg = CompressorConfig::new();
    cfg.threshold_db = t;
    cfg.ratio = r;
    cfg.knee_type = KneeType::Soft;
    cfg.knee_width_db = w;
    cfg.makeup_gain_db = 0.0;
    let levels = [-26.0, -20.0, -14.0];
    let curve = mono(vec![0.0; 4])
        .get_compression_curve(&cfg, NonEmptySlice::new(&levels).unwrap())
        .unwrap();
    assert!((curve[0] + 26.0).abs() < 1e-12);
    assert!((curve[1] + 21.125).abs() < 1e-12);
    assert!((curve[2] + 18.5).abs() < 1e-12);
}

/// The soft-knee limiter static curve never exceeds the ceiling and reaches
/// it exactly at C + W/2.
#[test]
fn soft_limiter_static_curve_never_exceeds_ceiling() {
    let (c, w) = (-1.0, 6.0);
    let y = |x: f64| x - calculate_limiting_gain(x, c, KneeType::Soft, w);
    let max = (0..=4000)
        .map(|i| y(-20.0 + i as f64 * 0.01))
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max <= c + 1e-12,
        "max static output {max} dB > ceiling {c} dB"
    );
    assert!((y(c + w / 2.0) - c).abs() < 1e-12);
    assert!((y(c - w / 2.0) - (c - w / 2.0)).abs() < 1e-12);
}

/// A gate with ratio R is a 1:R downward expander: a steady −60 dBFS tone with
/// T = −40 dBFS, R = 10 is reduced by (T − L)(R − 1) = 180 dB.
#[test]
fn gate_ratio_is_downward_expansion_ratio() {
    let cfg = GateConfig::with_params(-40.0, 10.0, 0.1, 100.0);
    let mut audio = mono(sine(1000.0, 24_000, 1e-3));
    audio.apply_gate_in_place(&cfg).unwrap();
    let out_db = db(peak(&audio.as_slice().unwrap()[19_200..]));
    let attenuation = -60.0 - out_db;
    assert!(
        attenuation >= 100.0,
        "gate attenuation only {attenuation} dB"
    );
}

/// `attack_decay_envelope(method = Rms)` must use a 10 ms RMS window even when
/// the follower was built for peak detection: a unit sine has RMS 1/√2.
#[test]
fn attack_decay_rms_method_uses_rms_window() {
    let follower = EnvelopeFollower::new(1.0, 10.0, FS, DynamicRangeMethod::Peak);
    let audio = mono(sine(1000.0, 48_000, 1.0));
    let (attack, _decay) = audio.attack_decay_envelope(&follower, DynamicRangeMethod::Rms);
    let NdResult::Mono(env) = attack else {
        panic!("expected mono result")
    };
    let last = env[env.len() - 1];
    assert!(
        (last - std::f64::consts::FRAC_1_SQRT_2).abs() < 0.01,
        "RMS envelope {last}, expected 0.7071"
    );
}
