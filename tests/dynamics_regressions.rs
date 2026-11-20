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
    let mut cfg = LimiterConfig::broadcast(); // hard knee, C = -1 dBFS
    // Sample-peak detection: the square's band-limited reconstruction
    // overshoots its samples, so a true-peak detector reads more than 1.
    cfg.isp_limiting = false;
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

// ---------------------------------------------------------------------------
// Side chain and true-peak detection
// ---------------------------------------------------------------------------

fn rms(x: &[f64]) -> f64 {
    (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
}

fn samples(a: &AudioSamples<'static, f64>) -> Vec<f64> {
    a.as_slice().unwrap().to_vec()
}

/// A compressor with a fast attack and release, so steady tones settle within
/// a few milliseconds, and a hard knee at −20 dBFS, ratio 10.
fn steady_compressor() -> CompressorConfig {
    let mut c = CompressorConfig::new();
    c.threshold_db = -20.0;
    c.ratio = 10.0;
    c.attack_ms = 1.0;
    c.release_ms = 5.0;
    c.knee_type = KneeType::Hard;
    c.knee_width_db = 0.0;
    c.makeup_gain_db = 0.0;
    c.lookahead_ms = 0.0;
    c.detection_method = DynamicRangeMethod::Peak;
    c
}

/// Output level in dB relative to the input over the second half of a tone.
fn gain_db(input: &[f64], output: &[f64]) -> f64 {
    let half = input.len() / 2;
    db(rms(&output[half..]) / rms(&input[half..]))
}

/// The high-pass stops low frequencies from triggering gain reduction: a loud
/// 50 Hz tone is compressed with an unfiltered side chain and passes almost
/// unchanged with a 1 kHz high-pass (2nd order, about −52 dB at 50 Hz). A tone
/// in the pass band is compressed either way.
#[test]
fn side_chain_high_pass_keeps_bass_from_triggering() {
    let run = |freq: f64, hp: Option<f64>| {
        let x = sine(freq, 24_000, 0.5); // −6 dBFS, 14 dB over the threshold
        let mut cfg = steady_compressor();
        cfg.side_chain.enable();
        if let Some(f) = hp {
            cfg.side_chain.set_high_pass(f);
        }
        let y = mono(x.clone()).apply_compressor(&cfg).unwrap();
        gain_db(&x, &samples(&y))
    };
    assert!(run(50.0, None) < -10.0, "unfiltered: {}", run(50.0, None));
    assert!(
        run(50.0, Some(1_000.0)) > -0.1,
        "high-passed: {}",
        run(50.0, Some(1_000.0))
    );
    assert!(run(5_000.0, Some(1_000.0)) < -10.0);
}

/// The low-pass stops high frequencies from triggering gain reduction.
#[test]
fn side_chain_low_pass_keeps_treble_from_triggering() {
    let run = |freq: f64, lp: Option<f64>| {
        let x = sine(freq, 24_000, 0.5);
        let mut cfg = steady_compressor();
        cfg.side_chain.enable();
        if let Some(f) = lp {
            cfg.side_chain.set_low_pass(f);
        }
        let y = mono(x.clone()).apply_compressor(&cfg).unwrap();
        gain_db(&x, &samples(&y))
    };
    assert!(run(10_000.0, None) < -10.0);
    assert!(
        run(10_000.0, Some(300.0)) > -0.1,
        "low-passed: {}",
        run(10_000.0, Some(300.0))
    );
    assert!(run(100.0, Some(300.0)) < -10.0);
}

/// Pre-emphasis raises the detector level of high frequencies by up to
/// `pre_emphasis_db` and leaves low frequencies alone. The shelf gives about
/// +11.6 dB at 15 kHz for +12 dB, so a 15 kHz tone at −24 dBFS (under the
/// −20 dBFS threshold) is reduced about as much as one at −12 dBFS without
/// pre-emphasis; a 100 Hz tone at −24 dBFS is not reduced; and −12 dB of
/// pre-emphasis takes the −12 dBFS tone under the threshold.
#[test]
fn side_chain_pre_emphasis_weights_high_frequencies() {
    let run = |freq: f64, amp_db: f64, pre: f64| {
        let x = sine(freq, 24_000, 10f64.powf(amp_db / 20.0));
        let mut cfg = steady_compressor();
        cfg.side_chain.enable();
        cfg.side_chain.pre_emphasis_db = pre;
        let y = mono(x.clone()).apply_compressor(&cfg).unwrap();
        gain_db(&x, &samples(&y))
    };
    assert_eq!(run(15_000.0, -24.0, 0.0), 0.0);
    let loud = run(15_000.0, -12.0, 0.0);
    assert!(loud < -4.5, "no pre-emphasis at -12 dBFS: {loud}");
    let boosted = run(15_000.0, -24.0, 12.0);
    assert!(
        (boosted - loud).abs() < 0.5,
        "boosted {boosted}, loud {loud}"
    );
    assert_eq!(run(100.0, -24.0, 12.0), 0.0);
    assert_eq!(run(15_000.0, -12.0, -12.0), 0.0);
}

/// `external_mix` blends the main and the side-chain signal in the detector:
/// 1 ducks a quiet main signal under a loud side-chain signal, 0 detects on
/// the main signal alone (the same as the plain compressor), and 0.5 reduces
/// less than 1.
#[test]
fn side_chain_external_mix_blends_detector_sources() {
    let main = sine(440.0, 24_000, 0.05); // −26 dBFS, under the threshold
    let key = sine(1_000.0, 24_000, 0.9); // loud key signal
    let run = |mix: f64| {
        let mut cfg = steady_compressor();
        cfg.side_chain.enable();
        cfg.side_chain.external_mix = mix;
        let y = mono(main.clone())
            .apply_compressor_sidechain(&cfg, &mono(key.clone()))
            .unwrap();
        gain_db(&main, &samples(&y))
    };
    let full = run(1.0);
    let half = run(0.5);
    assert!(full < -10.0, "mix 1: {full}");
    assert!(half < -3.0 && half > full, "mix 0.5: {half}, mix 1: {full}");
    assert_eq!(run(0.0), 0.0);
    // Mix 0 is the plain compressor with the side chain's filters.
    let mut cfg = steady_compressor();
    cfg.side_chain.enable();
    cfg.side_chain.external_mix = 0.0;
    let loud = sine(440.0, 24_000, 0.5);
    let a = mono(loud.clone())
        .apply_compressor_sidechain(&cfg, &mono(key.clone()))
        .unwrap();
    let b = mono(loud).apply_compressor(&cfg).unwrap();
    assert_eq!(samples(&a), samples(&b));
    // The default mix of an enabled side chain is the external signal alone.
    assert_eq!(CompressorConfig::new().side_chain.external_mix, 1.0);
}

/// A mono side-chain signal ducks every channel of a stereo signal; a stereo
/// one drives the channels one by one; other channel counts are an error.
#[test]
fn side_chain_channel_combinations() {
    let n = 24_000;
    let quiet = sine(440.0, n, 0.05);
    let key = sine(1_000.0, n, 0.9);
    let mut stereo = ndarray::Array2::<f64>::zeros((2, n));
    for (i, v) in quiet.iter().enumerate() {
        stereo[[0, i]] = *v;
        stereo[[1, i]] = *v;
    }
    let main = AudioSamples::new_multi_channel(stereo.clone(), sample_rate!(48000)).unwrap();
    let mut cfg = steady_compressor();
    cfg.side_chain.enable();

    let out = main
        .apply_compressor_sidechain(&cfg, &mono(key.clone()))
        .unwrap();
    for c in 0..2 {
        let row: Vec<f64> = out.as_multi_channel().unwrap().as_view().row(c).to_vec();
        assert!(gain_db(&quiet, &row) < -10.0, "channel {c}");
    }

    // Key on the left only: the left channel is ducked, the right is not.
    let mut keys = ndarray::Array2::<f64>::zeros((2, n));
    for (i, v) in key.iter().enumerate() {
        keys[[0, i]] = *v;
    }
    let keys = AudioSamples::new_multi_channel(keys, sample_rate!(48000)).unwrap();
    let out = main.apply_compressor_sidechain(&cfg, &keys).unwrap();
    let rows = out.as_multi_channel().unwrap().as_view();
    assert!(gain_db(&quiet, &rows.row(0).to_vec()) < -10.0);
    assert_eq!(gain_db(&quiet, &rows.row(1).to_vec()), 0.0);

    let three =
        AudioSamples::new_multi_channel(ndarray::Array2::<f64>::zeros((3, n)), sample_rate!(48000))
            .unwrap();
    assert!(main.apply_compressor_sidechain(&cfg, &three).is_err());
    assert!(mono(quiet).apply_compressor_sidechain(&cfg, &keys).is_err());
}

/// True-peak limiting: a full-scale sine at f_s/4 sampled 45° off its peaks has
/// sample peaks at −3.01 dBFS but a true peak of 0 dBTP. With a −1 dB ceiling,
/// sample-peak detection leaves it unchanged; true-peak detection brings the
/// true peak down to the ceiling (the samples to about −4 dBFS).
#[test]
fn isp_limiting_limits_inter_sample_peaks() {
    let n = 24_000;
    let x: Vec<f64> = (0..n)
        .map(|i| (PI / 2.0 * i as f64 + PI / 4.0).sin())
        .collect();
    let mut cfg = LimiterConfig::default();
    cfg.ceiling_db = -1.0;
    cfg.knee_type = KneeType::Hard;
    cfg.knee_width_db = 0.0;
    cfg.attack_ms = 0.1;
    cfg.release_ms = 50.0;
    cfg.lookahead_ms = 2.0;
    cfg.detection_method = DynamicRangeMethod::Peak;

    cfg.isp_limiting = false;
    let sample_peak = samples(&mono(x.clone()).apply_limiter(&cfg).unwrap());
    assert_eq!(sample_peak, x);

    cfg.isp_limiting = true;
    let y = samples(&mono(x).apply_limiter(&cfg).unwrap());
    let steady = &y[n / 2..];
    // An f_s/4 sine's true peak is √2 times its sample peak.
    let true_peak_db = db(peak(steady) * std::f64::consts::SQRT_2);
    assert!(
        (true_peak_db - cfg.ceiling_db).abs() < 0.1,
        "true peak {true_peak_db} dBTP, ceiling {}",
        cfg.ceiling_db
    );
}

/// With true-peak detection the side-chain limiter reads the true peaks of the
/// side-chain signal.
#[test]
fn isp_limiting_reads_the_side_chain_signal() {
    let n = 24_000;
    let key: Vec<f64> = (0..n)
        .map(|i| (PI / 2.0 * i as f64 + PI / 4.0).sin())
        .collect();
    let main = sine(440.0, n, 0.5);
    let mut cfg = LimiterConfig::default();
    cfg.ceiling_db = -2.0;
    cfg.knee_type = KneeType::Hard;
    cfg.knee_width_db = 0.0;
    cfg.side_chain.enable();
    cfg.isp_limiting = false;
    // Key sample peaks at −3 dBFS: under the ceiling, no reduction.
    let a = mono(main.clone())
        .apply_limiter_sidechain(&cfg, &mono(key.clone()))
        .unwrap();
    assert_eq!(samples(&a), main);
    cfg.isp_limiting = true;
    // Key true peak 0 dBTP: 2 dB over the ceiling.
    let b = mono(main.clone())
        .apply_limiter_sidechain(&cfg, &mono(key))
        .unwrap();
    let g = gain_db(&main, &samples(&b));
    assert!((g + 2.0).abs() < 0.1, "gain {g} dB");
}

/// Side-chain parameters are validated: NaN frequencies and mixes, and
/// pre-emphasis outside ±40 dB, are rejected.
#[test]
fn side_chain_validation() {
    let x = mono(sine(440.0, 4_800, 0.5));
    let bad = |edit: &dyn Fn(&mut CompressorConfig)| {
        let mut cfg = steady_compressor();
        cfg.side_chain.enable();
        edit(&mut cfg);
        x.apply_compressor(&cfg).is_err()
    };
    assert!(bad(&|c| c.side_chain.high_pass_freq = Some(f64::NAN)));
    assert!(bad(&|c| c.side_chain.low_pass_freq = Some(24_000.0)));
    assert!(bad(&|c| c.side_chain.external_mix = f64::NAN));
    assert!(bad(&|c| c.side_chain.pre_emphasis_db = 41.0));
    assert!(bad(&|c| c.side_chain.pre_emphasis_db = f64::INFINITY));
    assert!(!bad(&|c| c.side_chain.pre_emphasis_db = -40.0));
}
