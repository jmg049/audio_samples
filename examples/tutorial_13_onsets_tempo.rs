//! Tutorial 13 lab: onset detection and tempo estimation.
//!
//! Accompanies the guide chapter `audio_samples::guide::c13_onsets_tempo`
//! (`documentation/guide/13_onsets_tempo.md`).
//!
//! What it does:
//!
//! 1. Builds a click track (decaying noise bursts every 0.5 s) and runs
//!    `detect_onsets` with the default configuration (five of the eight
//!    bursts: the 0.1 threshold floor drops the three weakest) and with the
//!    floor lowered to 0.01 (all eight).
//! 2. Switches peak-picking lengths sized as if they counted audio samples
//!    (as in earlier releases) on one at a time, to show which setting loses
//!    which onsets.
//! 3. Re-derives the onset detection function from the CQT magnitude and the
//!    peak picking from its definition, and converts frames to seconds.
//! 4. Estimates tempo for click tracks at several tempi, re-derives
//!    `estimate_tempo` (autocorrelation of the onset envelope with a
//!    log-normal prior) and shows the metrical-level errors at 80 BPM
//!    (half the tempo, an octave) and 174 BPM (a third of the tempo).
//! 5. Prints the explanation of the default `OnsetDetectionConfig`.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --example tutorial_13_onsets_tempo --features educational,full_no_plotting
//! ```

use std::num::NonZeroUsize;

use audio_samples::educational::{Explain, ExplainContext};
use audio_samples::operations::AudioBeatTracking;
use audio_samples::operations::beat::{BeatTrackingConfig, onset_strength_envelope};
use audio_samples::operations::onset::OnsetDetectionConfig;
use audio_samples::{AudioOnsetDetection, AudioSamples, AudioTransforms, sample_rate};
use ndarray::Array1;
use spectrograms::{SpectrogramParams, StftParams};

const SR: f64 = 44_100.0;
const HOP: f64 = 512.0;

/// Exponentially decaying noise bursts ("clicks") starting at `times` (s).
fn click_track(times: &[f64], seconds: f64) -> AudioSamples<'static, f64> {
    let mut x = vec![0.0f64; (seconds * SR) as usize];
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    for &t in times {
        let start = (t * SR) as usize;
        for i in 0..256 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let r = (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            if start + i < x.len() {
                x[start + i] = 0.9 * (-(i as f64) / 24.0).exp() * r;
            }
        }
    }
    AudioSamples::new_mono(Array1::from(x), sample_rate!(44100)).expect("non-empty")
}

/// The default configuration with the threshold floor lowered from 0.1 to
/// 0.01, so that weak clicks are kept.
fn sensitive() -> OnsetDetectionConfig {
    let mut c = OnsetDetectionConfig::default();
    c.peak_picking.adaptive_threshold.min_threshold = 0.01;
    c
}

/// Pre-emphasis y[i] = x[i] - c x[i-1] (first value unchanged), then
/// normalisation by the maximum absolute value: the ODF the threshold is applied to.
fn pre_emphasised(odf: &[f64], c: f64) -> Vec<f64> {
    let y: Vec<f64> = (0..odf.len())
        .map(|i| {
            if i == 0 {
                odf[0]
            } else {
                c.mul_add(-odf[i - 1], odf[i])
            }
        })
        .collect();
    let max = y.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    y.iter().map(|v| v / max).collect()
}

/// Peak picking as implemented, for a config with pre-emphasis and no median:
/// pre-emphasise and peak-normalise, threshold = clamp(local max over +-w/2 -
/// delta, floor, 1), strict local maxima above it, then strongest-first
/// minimum separation.
fn my_pick_peaks(
    odf: &[f64],
    pre_coeff: f64,
    window: usize,
    delta: f64,
    floor: f64,
    sep: usize,
) -> Vec<usize> {
    let o = pre_emphasised(odf, pre_coeff);
    let half = window.min(o.len()).max(3) / 2;
    let mut cand: Vec<(usize, f64)> = (1..o.len() - 1)
        .filter(|&i| {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(o.len());
            let local_max = o[lo..hi].iter().fold(f64::NEG_INFINITY, |m, v| m.max(*v));
            let thr = (local_max - delta).clamp(floor, 1.0);
            o[i] > o[i - 1] && o[i] > o[i + 1] && o[i] > thr
        })
        .map(|i| (i, o[i]))
        .collect();
    cand.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut kept: Vec<usize> = Vec::new();
    for (i, _) in cand {
        if kept.iter().all(|&k| k.abs_diff(i) >= sep) {
            kept.push(i);
        }
    }
    kept.sort_unstable();
    kept
}

/// estimate_tempo written out: autocorrelation of the mean-removed onset
/// envelope over lags for 40-240 BPM, weighted by a log-normal prior at 120 BPM.
/// Returns (bpm, lag, per-lag raw autocorrelation).
fn my_tempo(env: &[f64]) -> (f64, usize, Vec<f64>) {
    let frame_rate = SR / HOP;
    let mean = env.iter().sum::<f64>() / env.len() as f64;
    let c: Vec<f64> = env.iter().map(|v| v - mean).collect();
    let min_lag = ((60.0 * frame_rate / 240.0).floor() as usize).max(1);
    let max_lag = ((60.0 * frame_rate / 40.0).ceil() as usize).min(c.len() - 1);
    let mut ac = vec![0.0; max_lag + 1];
    let (mut best, mut best_score) = (min_lag, f64::NEG_INFINITY);
    for lag in min_lag..=max_lag {
        ac[lag] = (lag..c.len()).map(|i| c[i] * c[i - lag]).sum::<f64>() / (c.len() - lag) as f64;
        if ac[lag] <= 0.0 {
            continue;
        }
        let z = (60.0 * frame_rate / lag as f64).log2() - 120f64.log2();
        let score = ac[lag] * (-0.5 * z * z).exp();
        if score > best_score {
            best_score = score;
            best = lag;
        }
    }
    (60.0 * frame_rate / best as f64, best, ac)
}

fn main() {
    println!("== Tutorial 13: onsets and tempo ==\n");

    // ------------------------------------------------------------ 1. Onsets
    let truth: Vec<f64> = (0..8).map(|k| 0.25 + 0.5 * f64::from(k)).collect();
    let audio = click_track(&truth, 4.25);
    println!("1. Click track: 8 noise bursts at 0.25, 0.75, ..., 3.75 s (44.1 kHz)");
    let default_onsets = audio
        .detect_onsets(&OnsetDetectionConfig::default())
        .expect("detect");
    println!("   default configuration: {} onsets", default_onsets.len());
    // The three weakest bursts (the first three) fall below the 0.1 floor;
    // bursts 4 to 8 are found, each 29-35 ms early (see below).
    assert_eq!(default_onsets.len(), 5);
    for (t, want) in default_onsets.iter().zip(&truth[3..]) {
        assert!((t - want) < 0.0 && (t - want) > -0.035);
    }
    let onsets = audio.detect_onsets(&sensitive()).expect("detect");
    println!("   threshold floor 0.01:  {} onsets", onsets.len());
    assert_eq!(onsets.len(), 8);
    for (t, want) in onsets.iter().zip(&truth) {
        println!(
            "     {t:.4} s (true {want:.2} s, {:+.1} ms)",
            1000.0 * (t - want)
        );
        // Frame centres: the 72.7 ms window contains the burst before the
        // frame centre reaches it, so the reported time is early.
        assert!((t - want) < 0.0 && (t - want) > -0.035);
    }
    // 0.5 s is 43.07 frames; frames are 43 apart, so the lead grows by 0.77 ms per burst.
    assert!((0.5 * SR / HOP - 43.07).abs() < 0.005);
    assert!((1000.0 * (0.5 - 43.0 * HOP / SR) - 0.77).abs() < 0.005);
    assert!((43.0 * HOP / SR - 0.4992).abs() < 5e-5);
    let lead = 1000.0 * (onsets[0] - truth[0]);
    assert!((lead + 29.4).abs() < 0.05);
    // The threshold's input: the pre-emphasised, normalised ODF at each burst.
    let (_, odf_default) = audio
        .onset_detection_function(&OnsetDetectionConfig::default())
        .expect("odf");
    let seen = pre_emphasised(&odf_default, 0.97);
    let burst_peaks: Vec<f64> = truth
        .iter()
        .map(|t| {
            let f = (t * SR / HOP).round() as usize;
            (f - 6..=f).map(|i| seen[i]).fold(f64::MIN, f64::max)
        })
        .collect();
    let rounded: Vec<f64> = burst_peaks
        .iter()
        .map(|h| (h * 1000.0).round() / 1000.0)
        .collect();
    println!("   normalised peaks per burst (the floor is 0.1): {rounded:?}");
    assert_eq!(
        rounded,
        vec![0.030, 0.025, 0.087, 0.172, 0.288, 0.315, 1.0, 0.904]
    );
    assert_eq!(burst_peaks.iter().filter(|&&h| h < 0.1).count(), 3);

    // ------------------------------------------------------------ 2. Settings
    println!("\n2. Peak-picking lengths sized in samples instead of frames (one change at a time)");
    type Change = fn(&mut OnsetDetectionConfig);
    let cases: [(&str, Change, usize); 4] = [
        (
            "floor 0.01 + median_filter (length 3)",
            |c| c.peak_picking.median_filter = true,
            0,
        ),
        (
            "floor 0.01 + threshold window 1024 frames",
            |c| c.peak_picking.adaptive_threshold.window_size = 1024,
            1,
        ),
        (
            "floor 0.01 + min separation 512 frames",
            |c| c.peak_picking.min_peak_separation = NonZeroUsize::new(512).expect("nz"),
            3,
        ),
        (
            "floor 0.01 without pre-emphasis",
            |c| c.peak_picking.pre_emphasis = false,
            8,
        ),
    ];
    for (name, change, expected) in cases {
        let mut c = sensitive();
        change(&mut c);
        let n = audio.detect_onsets(&c).expect("detect").len();
        println!("   {name:42} -> {n} onsets");
        assert_eq!(n, expected, "{name}");
    }

    // ------------------------------------------------------------ 3. How it works
    println!("\n3. The onset detection function and peak picking, re-derived");
    let mut raw_cfg = OnsetDetectionConfig::default();
    raw_cfg.adaptive_threshold = false;
    let (_, odf_raw) = audio.onset_detection_function(&raw_cfg).expect("odf");
    let win = raw_cfg.effective_window_size(SR);
    let stft = StftParams::builder()
        .n_fft(win)
        .hop_size(raw_cfg.hop_size)
        .window(raw_cfg.cqt_params.window.clone())
        .centre(true)
        .build()
        .expect("stft");
    let sp = SpectrogramParams::new(stft, SR).expect("params");
    let mag = audio
        .cqt_magnitude_spectrogram(&sp, &raw_cfg.cqt_params)
        .expect("cqt");
    let m = mag.data();
    let mut worst = 0.0f64;
    for t in 1..m.ncols() {
        let flux: f64 = (0..m.nrows())
            .map(|b| (m[[b, t]] - m[[b, t - 1]]).max(0.0))
            .sum();
        worst = worst.max((flux - odf_raw[t]).abs());
    }
    println!(
        "   CQT: {} bins x {} frames, window {win} samples; sum of rectified increases vs library ODF: max diff {worst:.1e}",
        m.nrows(),
        m.ncols()
    );
    assert!(worst < 1e-9);

    let peaks_at: Vec<usize> = truth
        .iter()
        .map(|t| {
            let f = (t * SR / HOP).round() as usize;
            (f - 6..=f)
                .max_by(|&a, &b| odf_raw[a].total_cmp(&odf_raw[b]))
                .expect("frames")
        })
        .collect();
    let heights: Vec<f64> = peaks_at.iter().map(|&i| odf_raw[i]).collect();
    println!(
        "   ODF peak frames {peaks_at:?}, heights {:?}",
        heights
            .iter()
            .map(|h| (h * 10.0).round() / 10.0)
            .collect::<Vec<_>>()
    );
    let spread = heights.iter().fold(0.0f64, |m, v| m.max(*v))
        / heights.iter().fold(f64::INFINITY, |m, v| m.min(*v));
    println!("   identical bursts, but the strongest ODF peak is {spread:.0} times the weakest");
    assert!(spread > 10.0);
    // Values quoted in the guide.
    let rounded: Vec<f64> = heights.iter().map(|h| (h * 10.0).round() / 10.0).collect();
    assert_eq!(rounded, vec![0.3, 0.3, 1.0, 2.0, 3.3, 3.6, 11.5, 10.4]);
    assert_eq!(spread.round(), 40.0);
    assert_eq!(m.ncols(), 367);
    let reduced = m.ncols() / 4; // separation 512 is reduced to len/4 frames
    assert_eq!(reduced, 91);
    assert!((reduced as f64 * HOP / SR - 1.06).abs() < 0.005);

    // Same ODF for both configurations (they differ only in the floor).
    for (floor, expected) in [(0.1, &default_onsets), (0.01, &onsets)] {
        let frames = my_pick_peaks(&odf_default, 0.97, 9, 0.07, floor, 3);
        let seconds: Vec<f64> = frames.iter().map(|&i| i as f64 * HOP / SR).collect();
        println!("   our peak picking, floor {floor}: frames {frames:?}");
        assert_eq!(&seconds, expected);
    }
    println!("   t = i H / f_s reproduces detect_onsets exactly");

    // ------------------------------------------------------------ 4. Tempo
    println!("\n4. Tempo by autocorrelation of the onset envelope (hop 512, 86.13 frames/s)");
    println!("   true BPM | estimate | lag (frames) | estimate / true");
    let mut results = Vec::new();
    for bpm in [60.0, 80.0, 100.0, 120.0, 140.0, 174.0] {
        let ibi = 60.0 / bpm;
        let n = (8.0 / ibi) as usize;
        let times: Vec<f64> = (0..n).map(|k| 0.1 + ibi * k as f64).collect();
        let track = click_track(&times, 8.5);
        let cfg = BeatTrackingConfig::new(120.0, None, OnsetDetectionConfig::default());
        let est = track.estimate_tempo(&cfg).expect("tempo");
        let env = onset_strength_envelope(&track, &cfg.onset_config, None).expect("envelope");
        let (mine, lag, ac) = my_tempo(&env);
        println!("   {bpm:8.0} | {est:8.2} | {lag:12} | {:.3}", est / bpm);
        assert!((mine - est).abs() < 1e-9, "re-derived {mine} vs {est}");
        results.push((bpm, est, ac));
    }
    let est = |b: f64| results.iter().find(|r| r.0 == b).expect("tempo").1;
    assert!((est(120.0) - 120.19).abs() < 0.005);
    assert!((est(80.0) - 40.06).abs() < 0.005); // half tempo
    assert!((est(174.0) - 58.07).abs() < 0.005); // a third
    for (b, e) in [(60.0, 60.09), (100.0, 99.38), (140.0, 139.67)] {
        assert!((est(b) - e).abs() < 0.005);
    }
    let ac80 = &results.iter().find(|r| r.0 == 80.0).expect("80").2;
    println!(
        "   80 BPM: autocorrelation at lag 65 (79.5 BPM) {:.5}, at lag 129 (40.1 BPM) {:.5}",
        ac80[65], ac80[129]
    );
    assert!(ac80[129] > 10.0 * ac80[65]);
    assert!((ac80[129] - 0.00240).abs() < 5e-6 && (ac80[65] - 0.00012).abs() < 5e-6);
    assert!((ac80[129] / ac80[65]).round() == 20.0);
    let prior = |lag: f64| {
        let z = (60.0 * SR / HOP / lag).log2() - 120f64.log2();
        (-0.5 * z * z).exp()
    };
    println!(
        "   prior weights: lag 65 -> {:.2}, lag 129 -> {:.2}",
        prior(65.0),
        prior(129.0)
    );
    assert!((prior(65.0) - 0.84).abs() < 0.005 && (prior(129.0) - 0.29).abs() < 0.005);

    let times80: Vec<f64> = (0..10).map(|k| 0.1 + 0.75 * f64::from(k)).collect();
    let (_, odf80) = click_track(&times80, 8.5)
        .onset_detection_function(&OnsetDetectionConfig::default())
        .expect("odf");
    let per_click: Vec<f64> = times80
        .iter()
        .map(|t| {
            let f = (t * SR / HOP).round() as usize;
            (f.saturating_sub(6)..=f)
                .map(|i| odf80[i])
                .fold(0.0, f64::max)
        })
        .collect();
    println!(
        "   80 BPM onset function at each click: {:?}",
        per_click
            .iter()
            .map(|v| (v * 100.0).round() / 100.0)
            .collect::<Vec<_>>()
    );
    println!("   every other click is close to zero, so the envelope repeats every two beats");
    let rounded: Vec<f64> = per_click
        .iter()
        .map(|v| (v * 100.0).round() / 100.0)
        .collect();
    assert_eq!(
        rounded,
        vec![0.88, 0.0, 2.6, 0.08, 13.42, 0.71, 0.0, 3.27, 0.03, 8.69]
    );

    // ------------------------------------------------------------ 5. Explain
    let doc =
        OnsetDetectionConfig::default().explain_with(&ExplainContext::new().with_sample_rate(SR));
    let w = doc.quantity("Window size").expect("q").value;
    let res = doc.quantity("Time resolution").expect("q").value;
    let sep = doc.quantity("Minimum peak separation").expect("q").value;
    assert_eq!(w, 3207.0);
    assert!((res - 11.61).abs() < 0.005);
    assert!((sep - 34.83).abs() < 0.005); // 3 frames
    println!("\n5. The default configuration explained:\n\n{doc}");
}
