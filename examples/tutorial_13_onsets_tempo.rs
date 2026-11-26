//! Tutorial 13 lab: onset detection and tempo estimation.
//!
//! Accompanies the guide chapter `audio_samples::guide::c13_onsets_tempo`
//! (`documentation/guide/13_onsets_tempo.md`).
//!
//! What it does:
//!
//! 1. Builds a click track (decaying noise bursts every 0.5 s, the first three
//!    26 dB quieter than the rest) and runs `detect_onsets` with the default
//!    configuration (five of the eight bursts: the 0.1 threshold floor drops
//!    the three quiet ones) and with the floor lowered to 0.01 (all eight).
//! 2. Switches on, one at a time, peak-picking lengths sized as if they
//!    counted audio samples rather than frames, to show which setting loses
//!    which onsets.
//! 3. Re-derives the onset detection function from the CQT magnitude and the
//!    peak picking from its definition, and converts frames to seconds.
//! 4. Estimates tempo for click tracks at several tempi, re-derives
//!    `estimate_tempo` (autocorrelation of the onset envelope with a
//!    log-normal prior) and shows a metrical-level error: at 80 BPM with
//!    quiet off-beats the estimate is half the tempo (an octave).
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

/// Exponentially decaying noise bursts ("clicks") starting at `times` (s), with
/// peak amplitudes `amps`.
fn click_track(times: &[f64], amps: &[f64], seconds: f64) -> AudioSamples<'static, f64> {
    let mut x = vec![0.0f64; (seconds * SR) as usize];
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    for (&t, &amp) in times.iter().zip(amps) {
        let start = (t * SR) as usize;
        for i in 0..256 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let r = (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            if start + i < x.len() {
                x[start + i] = amp * (-(i as f64) / 24.0).exp() * r;
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

/// Normalisation by the maximum absolute value: the ODF the threshold is applied to.
fn normalised(odf: &[f64]) -> Vec<f64> {
    let max = odf.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    odf.iter().map(|v| v / max).collect()
}

/// Peak picking as implemented, for a config without pre-emphasis or median:
/// peak-normalise, threshold = clamp(local max over +-w/2 - delta, floor, 1),
/// strict local maxima above it, then strongest-first minimum separation.
fn my_pick_peaks(odf: &[f64], window: usize, delta: f64, floor: f64, sep: usize) -> Vec<usize> {
    let o = normalised(odf);
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
    // The first three bursts are 0.045 instead of 0.9: 26 dB quieter.
    let amps = [0.045, 0.045, 0.045, 0.9, 0.9, 0.9, 0.9, 0.9];
    assert!((20.0 * (0.045f64 / 0.9).log10() + 26.02).abs() < 0.005);
    let audio = click_track(&truth, &amps, 4.25);
    println!(
        "1. Click track: 8 noise bursts at 0.25, 0.75, ..., 3.75 s (44.1 kHz), the first three 26 dB quieter"
    );
    let default_onsets = audio
        .detect_onsets(&OnsetDetectionConfig::default())
        .expect("detect");
    println!("   default configuration: {} onsets", default_onsets.len());
    // The three quiet bursts fall below the 0.1 floor; bursts 4 to 8 are found.
    assert_eq!(default_onsets.len(), 5);
    for (t, want) in default_onsets.iter().zip(&truth[3..]) {
        assert!((t - want).abs() < HOP / SR);
    }
    let onsets = audio.detect_onsets(&sensitive()).expect("detect");
    println!("   threshold floor 0.01:  {} onsets", onsets.len());
    assert_eq!(onsets.len(), 8);
    let errors_ms: Vec<f64> = onsets
        .iter()
        .zip(&truth)
        .map(|(t, want)| (10_000.0 * (t - want)).round() / 10.0)
        .collect();
    for (t, (want, e)) in onsets.iter().zip(truth.iter().zip(&errors_ms)) {
        println!("     {t:.4} s (true {want:.2} s, {e:+.1} ms)");
    }
    // Every reported time is a frame centre within one hop (11.61 ms) of its burst.
    assert!(errors_ms.iter().all(|e| e.abs() < 1000.0 * HOP / SR));
    assert_eq!(errors_ms, vec![-6.2, -7.0, -7.7, 3.1, 2.3, 1.6, 0.8, 0.0]);
    // 0.5 s is 43.07 frames; frames are 43 apart, so while the peak stays in the frame
    // before a burst the lead grows by 0.77 ms per burst.
    assert!((0.5 * SR / HOP - 43.07).abs() < 0.005);
    assert!((1000.0 * (0.5 - 43.0 * HOP / SR) - 0.77).abs() < 0.005);
    // The threshold's input: the normalised ODF at each burst.
    let (_, odf_default) = audio
        .onset_detection_function(&OnsetDetectionConfig::default())
        .expect("odf");
    let seen = normalised(&odf_default);
    let burst_peaks: Vec<f64> = truth
        .iter()
        .map(|t| {
            let f = (t * SR / HOP).round() as usize;
            (f - 2..=f + 1).map(|i| seen[i]).fold(f64::MIN, f64::max)
        })
        .collect();
    let rounded: Vec<f64> = burst_peaks
        .iter()
        .map(|h| (h * 1000.0).round() / 1000.0)
        .collect();
    println!("   normalised peaks per burst (the floor is 0.1): {rounded:?}");
    assert_eq!(
        rounded,
        vec![0.036, 0.033, 0.024, 0.333, 0.757, 0.794, 1.0, 0.884]
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
            "floor 0.01 + pre-emphasis 0.97",
            |c| c.peak_picking.pre_emphasis = true,
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
            (f - 2..=f + 1)
                .max_by(|&a, &b| odf_raw[a].total_cmp(&odf_raw[b]))
                .expect("frames")
        })
        .collect();
    let heights: Vec<f64> = peaks_at.iter().map(|&i| odf_raw[i]).collect();
    println!(
        "   ODF peak frames {peaks_at:?}, heights {:?}",
        heights
            .iter()
            .map(|h| (h * 100.0).round() / 100.0)
            .collect::<Vec<_>>()
    );
    // The five loud bursts have the same amplitude but differ in their noise and
    // in where they fall relative to the frame grid.
    let loud = &heights[3..];
    let spread = loud.iter().fold(0.0f64, |m, v| m.max(*v))
        / loud.iter().fold(f64::INFINITY, |m, v| m.min(*v));
    println!(
        "   the strongest of the five loud bursts' ODF peaks is {spread:.1} times the weakest"
    );
    let rounded: Vec<f64> = heights
        .iter()
        .map(|h| (h * 100.0).round() / 100.0)
        .collect();
    assert_eq!(
        rounded,
        vec![0.21, 0.19, 0.14, 1.94, 4.41, 4.62, 5.82, 5.15]
    );
    assert_eq!((spread * 10.0).round(), 30.0);
    assert_eq!(m.ncols(), 367);
    let reduced = m.ncols() / 4; // separation 512 is reduced to len/4 frames
    assert_eq!(reduced, 91);
    assert!((reduced as f64 * HOP / SR - 1.06).abs() < 0.005);

    // Same ODF for both configurations (they differ only in the floor).
    for (floor, expected) in [(0.1, &default_onsets), (0.01, &onsets)] {
        let frames = my_pick_peaks(&odf_default, 9, 0.07, floor, 3);
        let seconds: Vec<f64> = frames.iter().map(|&i| i as f64 * HOP / SR).collect();
        println!("   our peak picking, floor {floor}: frames {frames:?}");
        assert_eq!(&seconds, expected);
    }
    println!("   t = i H / f_s reproduces detect_onsets exactly");

    // ------------------------------------------------------------ 4. Tempo
    println!("\n4. Tempo by autocorrelation of the onset envelope (hop 512, 86.13 frames/s)");
    println!("   true BPM | estimate | lag (frames) | estimate / true");
    let cfg = BeatTrackingConfig::new(120.0, None, OnsetDetectionConfig::default());
    // Clicks every 60/bpm s from 0.1 s; every other click (the off-beats) at `off` of
    // the on-beat amplitude 0.9.
    let tempo_track = |bpm: f64, off: f64| {
        let ibi = 60.0 / bpm;
        let n = (8.0 / ibi) as usize;
        let times: Vec<f64> = (0..n).map(|k| 0.1 + ibi * k as f64).collect();
        let amps: Vec<f64> = (0..n)
            .map(|k| if k % 2 == 0 { 0.9 } else { 0.9 * off })
            .collect();
        (click_track(&times, &amps, 8.5), times)
    };
    let mut results = Vec::new();
    for (bpm, off) in [
        (60.0, 1.0),
        (80.0, 1.0),
        (100.0, 1.0),
        (120.0, 1.0),
        (140.0, 1.0),
        (174.0, 1.0),
        (80.0, 0.2),
    ] {
        let (track, _) = tempo_track(bpm, off);
        let est = track.estimate_tempo(&cfg).expect("tempo");
        let env = onset_strength_envelope(&track, &cfg.onset_config, None).expect("envelope");
        let (mine, lag, ac) = my_tempo(&env);
        let label = if off < 1.0 { " (quiet off-beats)" } else { "" };
        println!(
            "   {bpm:8.0} | {est:8.2} | {lag:12} | {:.3}{label}",
            est / bpm
        );
        assert!((mine - est).abs() < 1e-9, "re-derived {mine} vs {est}");
        results.push((bpm, off, est, ac));
    }
    let est = |b: f64, off: f64| {
        results
            .iter()
            .find(|r| r.0 == b && r.1 == off)
            .expect("tempo")
            .2
    };
    for (b, e) in [
        (60.0, 60.09),
        (80.0, 79.51),
        (100.0, 99.38),
        (120.0, 120.19),
        (140.0, 139.67),
        (174.0, 172.27),
    ] {
        assert!((est(b, 1.0) - e).abs() < 0.005);
    }
    assert!((est(80.0, 0.2) - 40.06).abs() < 0.005); // half tempo
    let ac80 = &results
        .iter()
        .find(|r| r.0 == 80.0 && r.1 == 0.2)
        .expect("80")
        .3;
    println!(
        "   80 BPM, quiet off-beats: autocorrelation at lag 65 (79.5 BPM) {:.5}, at lag 129 (40.1 BPM) {:.5}",
        ac80[65], ac80[129]
    );
    assert!(ac80[129] > 0.0 && ac80[65] > 0.0);
    assert!((ac80[129] - 0.00607).abs() < 5e-6 && (ac80[65] - 0.00181).abs() < 5e-6);
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
    assert!(ac80[129] * prior(129.0) > ac80[65] * prior(65.0));
    assert!((ac80[129] / ac80[65] - 3.35).abs() < 0.005);

    let (accented, times80) = tempo_track(80.0, 0.2);
    let (_, odf80) = accented
        .onset_detection_function(&OnsetDetectionConfig::default())
        .expect("odf");
    let per_click: Vec<f64> = times80
        .iter()
        .map(|t| {
            let f = (t * SR / HOP).round() as usize;
            (f.saturating_sub(2)..=f + 1)
                .map(|i| odf80[i])
                .fold(0.0, f64::max)
        })
        .collect();
    let rounded: Vec<f64> = per_click
        .iter()
        .map(|v| (v * 100.0).round() / 100.0)
        .collect();
    println!("   80 BPM onset function at each click: {rounded:?}");
    println!("   the off-beats are much weaker, so the envelope repeats every two beats");
    assert_eq!(
        rounded,
        vec![3.83, 1.32, 3.61, 0.49, 5.77, 0.38, 5.9, 0.79, 2.82, 0.73]
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
