//! Figures for chapter 13, onsets and tempo.

use audio_samples::operations::AudioBeatTracking;
use audio_samples::operations::beat::{BeatTrackingConfig, onset_strength_envelope};
use audio_samples::operations::onset::OnsetDetectionConfig;
use audio_samples::{AudioOnsetDetection, AudioSamples, sample_rate};
use guide_figures::*;
use ndarray::Array1;

const FS: f64 = 44_100.0;
const HOP: f64 = 512.0;

fn main() -> Result {
    let mut figs = Figures::new("ch13")?;
    onsets(&mut figs)?;
    tempo(&mut figs)?;
    figs.finish()
}

/// The chapter's decaying noise burst (xorshift noise, 256 samples), peak amplitude `amp`.
fn burst(x: &mut [f64], start: usize, amp: f64, seed: &mut u64) {
    for i in 0..256 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        let r = (*seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
        x[start + i] = amp * (-(i as f64) / 24.0).exp() * r;
    }
}

/// The four-burst click track of "With the library".
fn four_bursts() -> Result<AudioSamples<'static, f64>> {
    let sr = 44_100usize;
    let mut x = vec![0.0f64; 2 * sr];
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    for k in 0..4 {
        let start = ((0.25 + 0.5 * k as f64) * sr as f64) as usize;
        burst(&mut x, start, 0.9, &mut seed);
    }
    Ok(AudioSamples::new_mono(
        Array1::from(x),
        sample_rate!(44100),
    )?)
}

/// The click track of the tempo example: every other click at `off` times 0.9.
fn clicks(bpm: f64, off: f64, seconds: f64) -> Result<AudioSamples<'static, f64>> {
    let mut x = vec![0.0f64; (seconds * FS) as usize];
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let (mut t, mut k) = (0.1, 0);
    while t < seconds - 0.1 {
        let amp = if k % 2 == 0 { 0.9 } else { 0.9 * off };
        burst(&mut x, (t * FS) as usize, amp, &mut seed);
        t += 60.0 / bpm;
        k += 1;
    }
    Ok(AudioSamples::new_mono(
        Array1::from(x),
        sample_rate!(44100),
    )?)
}

/// Waveform, normalised ODF, threshold and detected onsets.
fn onsets(figs: &mut Figures) -> Result {
    let audio = four_bursts()?;
    let config = OnsetDetectionConfig::default();
    let pp = &config.peak_picking;
    check(
        "default peak picking: window 9, separation 3",
        (
            pp.adaptive_threshold.window_size,
            pp.min_peak_separation.get(),
        ) == (9, 3),
    );
    check(
        "default delta 0.07, floor 0.1",
        (
            pp.adaptive_threshold.delta,
            pp.adaptive_threshold.min_threshold,
        ) == (0.07, 0.1),
    );

    let detected = audio.detect_onsets(&config)?;
    let frames: Vec<f64> = detected.iter().map(|t| (t * FS / HOP).round()).collect();
    check(
        "onset frames 21, 64, 107, 150",
        frames == vec![21.0, 64.0, 107.0, 150.0],
    );
    for (k, t) in detected.iter().enumerate() {
        let lead = 0.25 + 0.5 * k as f64 - t;
        check(
            format!("lead {lead} in 6.0..8.6 ms"),
            lead > 0.006 && lead < 0.0086,
        );
    }

    // The chapter's re-implementation of the default peak picking, on the ODF
    // detect_onsets uses: pre-emphasis, normalisation, threshold.
    let (_, odf) = audio.onset_detection_function(&config)?;
    let odf: Vec<f64> = odf.iter().copied().collect();
    let pre: Vec<f64> = (0..odf.len())
        .map(|i| {
            if i == 0 {
                odf[0]
            } else {
                0.97f64.mul_add(-odf[i - 1], odf[i])
            }
        })
        .collect();
    let max = pre.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let o: Vec<f64> = pre.iter().map(|v| v / max).collect();
    let (half, delta, floor, sep) = (9 / 2, 0.07, 0.1, 3);
    let theta: Vec<f64> = (0..o.len())
        .map(|i| {
            let local_max = o[i.saturating_sub(half)..(i + half + 1).min(o.len())]
                .iter()
                .fold(f64::NEG_INFINITY, |a, v| a.max(*v));
            (local_max - delta).clamp(floor, 1.0)
        })
        .collect();
    let mut candidates: Vec<(usize, f64)> = (1..o.len() - 1)
        .filter(|&i| o[i] > o[i - 1] && o[i] > o[i + 1] && o[i] > theta[i])
        .map(|i| (i, o[i]))
        .collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut peaks: Vec<usize> = Vec::new();
    for (i, _) in candidates {
        if peaks.iter().all(|&p| p.abs_diff(i) >= sep) {
            peaks.push(i);
        }
    }
    peaks.sort_unstable();
    let ours: Vec<f64> = peaks.iter().map(|&i| i as f64 * HOP / FS).collect();
    check(
        "re-implemented peak picking equals detect_onsets",
        ours == detected,
    );

    // Waveform as a block peak envelope (50-sample blocks) to keep the SVG small.
    let x = audio.as_slice().ok_or("mono slice")?;
    let block = 50;
    let mut upper = Vec::new();
    let mut lower = Vec::new();
    for (b, chunk) in x.chunks(block).enumerate() {
        let t = (b * block) as f64 / FS;
        upper.push((t, chunk.iter().fold(0.0f64, |m, v| m.max(*v))));
        lower.push((t, chunk.iter().fold(0.0f64, |m, v| m.min(*v))));
    }
    let frame_t = |i: usize| i as f64 * HOP / FS;
    let odf_pts: Vec<(f64, f64)> = o
        .iter()
        .enumerate()
        .map(|(i, &v)| (frame_t(i), v))
        .collect();
    let theta_pts: Vec<(f64, f64)> = theta
        .iter()
        .enumerate()
        .map(|(i, &v)| (frame_t(i), v))
        .collect();
    let peak_pts: Vec<(f64, f64)> = peaks.iter().map(|&i| (frame_t(i), o[i])).collect();

    figs.svg(
        "ch13-onsets",
        "Top: the four-burst click track (bursts start at 0.25, 0.75, 1.25 and 1.75 s). \
         Bottom: the pre-emphasised, normalised onset detection function with the default \
         threshold, and the four onsets detect_onsets reports at frames 21, 64, 107 and 150, \
         6 to 9 ms before each burst.",
        (720, 560),
        |area| {
            let panels = rows(area, 2);
            let mut top = Chart::new("Time (s)", "Amplitude", (0.0, 2.0), (-1.0, 1.0))
                .add(Series::line("", upper.clone(), BLUE))
                .add(Series::line("", lower.clone(), BLUE));
            for k in 0..4 {
                top = top.vrule(0.25 + 0.5 * f64::from(k), "", GREY);
            }
            top.draw(&panels[0])?;
            let mut bottom = Chart::new("Time (s)", "Normalised ODF", (0.0, 2.0), (-0.4, 1.15))
                .add(Series::line(
                    "onset detection function",
                    odf_pts.clone(),
                    BLUE,
                ))
                .add(Series::new(
                    "threshold",
                    theta_pts.clone(),
                    ORANGE,
                    Style::Dashed(2),
                ))
                .add(Series::new(
                    "detected onsets",
                    peak_pts.clone(),
                    VERMILION,
                    Style::Dots(5),
                ))
                .legend(LegendPos::UpperLeft);
            for k in 0..4 {
                bottom = bottom.vrule(0.25 + 0.5 * f64::from(k), "", GREY);
            }
            bottom.draw(&panels[1])
        },
    )
}

/// Autocorrelation of the onset strength envelope, the tempo prior and the
/// chosen lag, at 120 BPM and at 80 BPM with quiet off-beats.
fn tempo(figs: &mut Figures) -> Result {
    let frame_rate = FS / HOP;
    let config = BeatTrackingConfig::new(120.0, None, OnsetDetectionConfig::default());
    let mut curves = Vec::new();
    for (bpm, off, expected_lag) in [(120.0, 1.0, 43usize), (80.0, 0.2, 129usize)] {
        let audio = clicks(bpm, off, 8.5)?;
        let lib = audio.estimate_tempo(&config)?;
        let env = onset_strength_envelope(&audio, &config.onset_config, None)?;
        let env: Vec<f64> = env.iter().copied().collect();

        // The chapter's autocorrelation and log-normal prior.
        let mean = env.iter().sum::<f64>() / env.len() as f64;
        let c: Vec<f64> = env.iter().map(|v| v - mean).collect();
        let (lo, hi) = (
            (60.0 * frame_rate / 240.0) as usize,
            (60.0 * frame_rate / 40.0).ceil() as usize,
        );
        let mut best = (f64::NEG_INFINITY, lo);
        let mut acf = Vec::new();
        let mut score = Vec::new();
        for lag in lo..=hi.min(c.len() - 1) {
            let a = (lag..c.len()).map(|i| c[i] * c[i - lag]).sum::<f64>() / (c.len() - lag) as f64;
            let b = 60.0 * frame_rate / lag as f64;
            let z = (b / 120.0).log2();
            let s = a * (-0.5 * z * z).exp();
            acf.push((b, a));
            score.push((b, s));
            if a > 0.0 && s > best.0 {
                best = (s, lag);
            }
        }
        let ours = 60.0 * frame_rate / best.1 as f64;
        assert_close("tempo estimate vs estimate_tempo", ours, lib, 1e-9);
        check(
            format!("lag {} == {expected_lag}", best.1),
            best.1 == expected_lag,
        );
        curves.push((bpm, off, lib, acf, score));
    }
    assert_close("lag 43 in BPM", 60.0 * FS / (HOP * 43.0), 120.19, 0.005);
    assert_close("lag 129 in BPM", 60.0 * FS / (HOP * 129.0), 40.06, 0.005);

    // The prior itself (the chapter's formula), for reference.
    let prior: Vec<(f64, f64)> = (40..=240)
        .map(|b| {
            let z = (f64::from(b) / 120.0).log2();
            (f64::from(b), (-0.5 * z * z).exp())
        })
        .collect();
    figs.svg(
        "ch13-tempo",
        "Autocorrelation of the onset strength envelope against tempo, and the same \
         weighted by the log-normal prior centred on 120 BPM (dashed); both curves are scaled to a maximum of 1. \
         For 120 BPM clicks estimate_tempo picks lag 43 (120.19 BPM); for 80 BPM clicks whose \
         off-beats have a fifth of the amplitude it picks lag 129 (40.06 BPM), half the tempo.",
        (720, 600),
        |area| {
            let panels = rows(area, 2);
            for (p, (bpm, off, lib, acf, score)) in panels.iter().zip(&curves) {
                let norm = |v: &[(f64, f64)]| -> Vec<(f64, f64)> {
                    let m = v.iter().fold(0.0f64, |m, p| m.max(p.1));
                    v.iter().map(|&(b, a)| (b, a / m)).collect()
                };
                Chart::new("Tempo (BPM)", "Scaled value", (38.0, 242.0), (-0.6, 1.1))
                    .title(&if *off < 1.0 {
                        format!("{bpm} BPM click track, off-beats at {off} of the amplitude")
                    } else {
                        format!("{bpm} BPM click track")
                    })
                    .add(Series::line("autocorrelation a(lag)", norm(acf), BLUE))
                    .add(Series::line("weighted by the prior", norm(score), VERMILION))
                    .add(Series::new("prior w (tempo)", prior.clone(), GREY, Style::Dashed(1)))
                    .vrule(*bpm, &format!("true tempo {bpm}"), GREY)
                    .vrule(*lib, &format!("estimate {lib:.2}"), VERMILION)
                    .legend(LegendPos::MiddleRight)
                    .draw(p)?;
            }
            Ok(())
        },
    )
}
