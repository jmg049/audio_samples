//! Figures for chapter 09, pitch: autocorrelation and YIN.

use std::f64::consts::PI;
use std::time::Duration;

use audio_samples::operations::types::PitchDetectionMethod;
use audio_samples::{AudioPitchAnalysis, AudioSamples, chirp, nzu, sample_rate, sine_wave};
use guide_figures::*;
use ndarray::Array1;

const FS: f64 = 16_000.0;

fn main() -> Result {
    let mut figs = Figures::new("ch09")?;
    autocorrelation(&mut figs)?;
    yin(&mut figs)?;
    octave(&mut figs)?;
    track(&mut figs)?;
    figs.finish()
}

/// d(tau) over all N - tau pairs and the CMND d'(tau), for tau = 0..=max_tau,
/// written out exactly as in the chapter's "How it works".
fn cmnd(x: &[f64], max_tau: usize) -> Vec<f64> {
    let n = x.len();
    let d: Vec<f64> = (0..=max_tau)
        .map(|tau| (0..n - tau).map(|j| (x[j] - x[j + tau]).powi(2)).sum())
        .collect();
    let mut c = vec![1.0; max_tau + 1];
    let mut running = 0.0;
    for tau in 1..=max_tau {
        running += d[tau];
        c[tau] = d[tau] * tau as f64 / running;
    }
    c
}

fn tone_440() -> AudioSamples<'static, f64> {
    sine_wave::<f64>(440.0, Duration::from_millis(50), sample_rate!(16000), 0.8)
}

/// r(tau) of the 440 Hz frame, with the lag the library picks.
fn autocorrelation(figs: &mut Figures) -> Result {
    let audio = tone_440();
    let x = audio.as_slice().ok_or("mono slice")?.to_vec();
    let n = x.len();
    let (min_tau, max_tau) = ((FS / 1000.0) as usize, (FS / 80.0) as usize);
    assert_eq!((min_tau, max_tau), (16, 200));
    let r = |tau: usize| (0..n - tau).map(|i| x[i] * x[i + tau]).sum::<f64>();
    let best = (min_tau..=max_tau)
        .max_by(|&a, &b| r(a).total_cmp(&r(b)))
        .ok_or("lag range")?;
    assert_eq!(best, 36);
    let ac = audio
        .detect_pitch_autocorr(80.0, 1000.0)?
        .ok_or("autocorrelation found no pitch")?;
    assert_close("library autocorrelation pitch", ac, FS / 36.0, 0.0);
    let r0 = r(0);
    let pts: Vec<(f64, f64)> = (0..=max_tau).map(|t| (t as f64, r(t) / r0)).collect();
    let peak = (best as f64, r(best) / r0);

    figs.svg(
        "ch09-autocorr",
        "Normalised autocorrelation r(tau)/r(0) of the 50 ms, 440 Hz tone at 16 kHz. Over the \
         search range for 80 to 1000 Hz (lags 16 to 200, right of the grey line) the largest \
         value is at lag 36, so detect_pitch_autocorr reports 16000/36 = 444.44 Hz; the true \
         period is 36.36 samples.",
        SIZE,
        |area| {
            Chart::new(
                "Lag tau (samples)",
                "r(tau) / r(0)",
                (0.0, 200.0),
                (-1.05, 1.15),
            )
            .add(Series::line("", pts.clone(), BLUE))
            .add(Series::new("", vec![peak], VERMILION, Style::Dots(6)))
            .vrule(16.0, "tau min = 16", GREY)
            .note(peak.0 + 3.0, peak.1 + 0.1, "lag 36: 444.44 Hz", VERMILION)
            .draw(area)
        },
    )
}

/// CMND d'(tau) of the 440 Hz frame: threshold, first lag below it, the bottom
/// of the dip and the interpolated period.
fn yin(figs: &mut Figures) -> Result {
    let audio = tone_440();
    let x = audio.as_slice().ok_or("mono slice")?.to_vec();
    let (min_tau, max_tau) = (16usize, 200usize);
    let c = cmnd(&x, max_tau);
    let theta = 0.1;
    let first = (min_tau..=max_tau)
        .find(|&t| c[t] < theta)
        .ok_or("no lag below theta")?;
    assert_eq!(first, 34);
    assert_close("d'(34)", c[34], 0.0756, 1e-4);
    assert_close("d'(36)", c[36], 0.0019, 1e-4);
    let mut tau = first;
    while tau < max_tau && c[tau + 1] < c[tau] {
        tau += 1;
    }
    assert_eq!(tau, 36);
    let a = c[tau + 1] + c[tau - 1] - 2.0 * c[tau];
    let b = (c[tau + 1] - c[tau - 1]) / 2.0;
    let shift = -b / a;
    assert_close("parabolic shift", shift, 0.3555, 1e-4);
    let tau_hat = tau as f64 + shift;
    let lib = audio
        .detect_pitch_yin(theta, 80.0, 1000.0)?
        .ok_or("YIN found no pitch")?;
    assert_close(
        "library YIN agrees with the written-out YIN",
        FS / tau_hat,
        lib,
        1e-9,
    );
    assert_close("library YIN pitch", lib, 440.0986, 5e-5);

    let full: Vec<(f64, f64)> = c.iter().enumerate().map(|(t, &v)| (t as f64, v)).collect();
    let top = (c.iter().fold(0.0f64, |m, &v| m.max(v)) * 1.1 * 10.0).ceil() / 10.0;
    let zoom: Vec<(f64, f64)> = (32..=40).map(|t| (t as f64, c[t])).collect();
    // The parabola through d'(35), d'(36), d'(37) (the chapter's step 4).
    let parabola: Vec<(f64, f64)> = (0..=100)
        .map(|i| {
            let u = -1.2 + 2.4 * f64::from(i) / 100.0;
            (tau as f64 + u, c[tau] + b * u + 0.5 * a * u * u)
        })
        .collect();
    let vertex = (tau_hat, c[tau] + b * shift + 0.5 * a * shift * shift);

    figs.svg(
        "ch09-yin",
        "YIN on the 440 Hz tone. Top: the cumulative-mean-normalised difference d'(tau) \
         starts at 1 and dips at multiples of the period. Bottom: the first lag below the \
         threshold 0.1 is 34; the search moves on to the bottom of the dip at 36, and a \
         parabola through lags 35 to 37 puts the period at 36.36 samples, 440.10 Hz \
         (detect_pitch_yin).",
        (720, 600),
        |area| {
            let p = rows(area, 2);
            Chart::new("Lag tau (samples)", "d'(tau)", (0.0, 200.0), (0.0, top))
                .add(Series::line("", full.clone(), BLUE))
                .hrule(theta, "threshold 0.1", VERMILION)
                .vrule(16.0, "tau min", GREY)
                .draw(&p[0])?;
            Chart::new("Lag tau (samples)", "d'(tau)", (33.0, 39.5), (0.0, 0.14))
                .add(Series::new("d'(tau)", zoom.clone(), BLUE, Style::Dots(4)))
                .add(Series::new("", zoom.clone(), BLUE, Style::Line(1)))
                .add(Series::new(
                    "parabola through 35, 36, 37",
                    parabola.clone(),
                    GREEN,
                    Style::Dashed(2),
                ))
                .add(Series::new(
                    "interpolated period 36.36",
                    vec![vertex],
                    VERMILION,
                    Style::Crosses(7),
                ))
                .hrule(theta, "threshold 0.1", VERMILION)
                .vrule(34.0, "first below threshold", GREY)
                .ticks(8, 7)
                .legend(LegendPos::UpperMiddle)
                .draw(&p[1])
        },
    )
}

/// The weak-odd-harmonics signal: a dip at the half period below 0.1.
fn octave(figs: &mut Figures) -> Result {
    let n = 1600;
    let amps = [
        (1.0, 0.1),
        (2.0, 1.0),
        (3.0, 0.1),
        (4.0, 1.0),
        (5.0, 0.1),
        (6.0, 1.0),
    ];
    let x: Vec<f64> = (0..n)
        .map(|i| {
            amps.iter()
                .map(|&(h, a): &(f64, f64)| a * (2.0 * PI * h * 200.0 * i as f64 / FS).cos())
                .sum()
        })
        .collect();
    let c = cmnd(&x, 200);
    assert_close("d'(40)", c[40], 0.0196, 5e-5);
    check(
        "lag 40 is the bottom of the dip",
        (32..40).all(|t| c[t] > c[40]),
    );
    let audio = AudioSamples::new_mono(Array1::from(x), sample_rate!(16000))?;
    let f = |theta: f64| -> Result<f64> {
        Ok(audio
            .detect_pitch_yin(theta, 80.0, 500.0)?
            .ok_or("YIN found no pitch")?)
    };
    assert_close("YIN at theta 0.1 (octave error)", f(0.1)?, 400.0, 1.0);
    assert_close("YIN at theta 0.01", f(0.01)?, 200.0, 0.1);
    assert_close("YIN at theta 0.021", f(0.021)?, 400.0, 1.0);
    assert_close("YIN at theta 0.019", f(0.019)?, 200.0, 0.1);
    let ac = audio
        .detect_pitch_autocorr(80.0, 500.0)?
        .ok_or("autocorrelation found no pitch")?;
    assert_close("autocorrelation also doubles the pitch", ac, 400.0, 0.0);
    let pts: Vec<(f64, f64)> = c.iter().enumerate().map(|(t, &v)| (t as f64, v)).collect();
    let top = (c.iter().fold(0.0f64, |m, &v| m.max(v)) * 1.1 * 10.0).ceil() / 10.0;

    figs.svg(
        "ch09-octave",
        "d'(tau) of the 200 Hz signal whose odd harmonics are 20 dB weaker than its even ones. \
         The half-period dip at lag 40 reaches 0.0196, below the default threshold 0.1, so \
         detect_pitch_yin reports 400 Hz; with a threshold of 0.019 or 0.01 it continues to the \
         true period, lag 80 (200 Hz).",
        SIZE,
        |area| {
            Chart::new("Lag tau (samples)", "d'(tau)", (0.0, 200.0), (0.0, top))
                .add(Series::line("", pts.clone(), BLUE))
                .hrule(0.1, "threshold 0.1", VERMILION)
                .vrule(40.0, "lag 40: 400 Hz", VERMILION)
                .vrule(80.0, "lag 80: 200 Hz", GREEN)
                .draw(area)
        },
    )
}

/// track_pitch on the 100 to 200 Hz glide.
fn track(figs: &mut Figures) -> Result {
    let glide = chirp::<f64>(
        100.0,
        200.0,
        Duration::from_millis(500),
        sample_rate!(16000),
        0.8,
    );
    let tr = glide.track_pitch(
        nzu!(512),
        nzu!(128),
        PitchDetectionMethod::Yin,
        0.1,
        80.0,
        400.0,
    )?;
    let mut worst: f64 = 0.0;
    let mut est = Vec::new();
    let mut errs = Vec::new();
    for frame in tr.frames().iter().filter(|f| f.time + 512.0 / FS <= 0.5) {
        let centre = frame.time + 256.0 / FS;
        let truth = 100.0 + 200.0 * centre;
        let f = frame.frequency.ok_or("unvoiced full frame")?;
        worst = worst.max((f - truth).abs());
        est.push((1000.0 * centre, f));
        errs.push((1000.0 * centre, f - truth));
    }
    assert_close("worst glide error (Hz)", worst, 0.12, 0.005);
    let truth: Vec<(f64, f64)> = vec![(0.0, 100.0), (500.0, 200.0)];

    figs.svg(
        "ch09-track",
        "track_pitch with YIN, 512-sample frames every 128 samples, on a 0.5 s linear glide \
         from 100 to 200 Hz. Top: each estimate at its frame centre. Bottom: estimate minus the \
         glide's frequency at the frame centre; the largest deviation is 0.12 Hz.",
        (720, 560),
        |area| {
            let p = rows(area, 2);
            Chart::new("Time (ms)", "Frequency (Hz)", (0.0, 500.0), (90.0, 210.0))
                .add(Series::new(
                    "true frequency 100 + 200 t",
                    truth.clone(),
                    GREY,
                    Style::Line(2),
                ))
                .add(Series::new(
                    "track_pitch (YIN)",
                    est.clone(),
                    VERMILION,
                    Style::Dots(4),
                ))
                .legend(LegendPos::UpperLeft)
                .draw(&p[0])?;
            Chart::new("Time (ms)", "Error (Hz)", (0.0, 500.0), (-0.2, 0.2))
                .add(Series::new("", errs.clone(), VERMILION, Style::Stems))
                .draw(&p[1])
        },
    )
}
