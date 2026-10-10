//! Figures for chapter 05, convolution and FIR filters.

use std::f64::consts::PI;
use std::num::{NonZeroU32, NonZeroUsize};

use audio_samples::educational::topics;
use audio_samples::{AudioProcessing, AudioSamples, AudioTransforms};
use guide_figures::*;
use non_empty_slice::NonEmptySlice;
use num_complex::Complex;
use spectrograms::{WindowType, make_window};

fn main() -> Result {
    let mut figs = Figures::new("ch05")?;
    let lp = windowed_sinc(63, 2000.0, 16_000.0);
    impulse_response(&mut figs, &lp)?;
    responses(&mut figs, &lp)?;
    convolution(&mut figs, &lp)?;
    figs.finish()
}

fn mono(v: Vec<f64>, fs: u32) -> Result<AudioSamples<'static, f64>> {
    let sr = NonZeroU32::new(fs).ok_or("sample rate")?;
    Ok(AudioSamples::new_mono(ndarray::Array1::from(v), sr)?)
}

/// The chapter's 63-tap Hamming windowed-sinc low-pass (the crate has no FIR
/// designer, so the chapter designs it by hand).
fn windowed_sinc(m: usize, fc: f64, fs: f64) -> Vec<f64> {
    let w = make_window::<f64>(WindowType::Hamming, NonZeroUsize::new(m).expect("m > 0"));
    let wc = 2.0 * fc / fs;
    let centre = (m - 1) as f64 / 2.0;
    let mut h: Vec<f64> = (0..m)
        .map(|n| {
            let t = n as f64 - centre;
            let sinc = if t == 0.0 {
                1.0
            } else {
                (PI * wc * t).sin() / (PI * wc * t)
            };
            wc * sinc * w[n]
        })
        .collect();
    let dc: f64 = h.iter().sum();
    h.iter_mut().for_each(|v| *v /= dc);
    h
}

/// The chapter's `measured_gain`: filter one second of a sine with
/// `apply_filter` and project the output on sin and cos.
fn measured_gain(h: &[f64], f: f64, fs: f64) -> Result<f64> {
    let len = fs as usize + h.len() - 1;
    let x: Vec<f64> = (0..len)
        .map(|n| (2.0 * PI * f * n as f64 / fs).sin())
        .collect();
    let y = mono(x, fs as u32)?.apply_filter(NonEmptySlice::new(h).ok_or("taps")?)?;
    let (mut s, mut c) = (0.0, 0.0);
    for (i, &v) in y.as_slice().ok_or("mono slice")?.iter().enumerate() {
        let phi = 2.0 * PI * f * i as f64 / fs;
        s += v * phi.sin();
        c += v * phi.cos();
    }
    Ok(2.0 / fs * (s * s + c * c).sqrt())
}

/// The taps of the low-pass, obtained as the library's response to an impulse.
fn impulse_response(figs: &mut Figures, lp: &[f64]) -> Result {
    let mut delta = vec![0.0; 64];
    delta[0] = 1.0;
    let ir = mono(delta, 16_000)?.convolve(&mono(lp.to_vec(), 16_000)?)?;
    let ir = ir.as_slice().ok_or("mono slice")?[..63].to_vec();
    for (a, b) in ir.iter().zip(lp) {
        assert_close("impulse response equals the taps", *a, *b, 1e-12);
    }
    let doc = topics::fir(lp, 16_000.0);
    let delay = doc.quantity("Group delay").ok_or("group delay")?.value;
    assert_close("group delay of the 63-tap low-pass", delay, 31.0, 0.0);
    check(
        "63-tap low-pass is type I",
        doc.has_warning_containing("type I"),
    );

    let pts = indexed(ir.iter().copied(), 1.0);
    figs.svg(
        "ch05-impulse-response",
        "Impulse response of the chapter's 63-tap Hamming windowed-sinc low-pass \
         (2 kHz cutoff at 16 kHz), obtained by convolving a unit impulse with convolve. \
         The taps are symmetric about tap 31, the group delay that topics::fir reports.",
        SIZE,
        |area| {
            Chart::new("Tap index m", "h[m]", (-1.0, 63.0), (-0.07, 0.28))
                .add(Series::new("", pts.clone(), BLUE, Style::Stems))
                .vrule(31.0, "centre: 31 samples of delay", VERMILION)
                .draw(area)
        },
    )
}

/// Magnitude responses measured by filtering sines with `apply_filter`.
fn responses(figs: &mut Figures, lp: &[f64]) -> Result {
    // Windowed-sinc low-pass at 16 kHz, every 40 Hz.
    let fs = 16_000.0;
    let mut lp_db = Vec::new();
    for k in 0..200 {
        let f = 20.0 + 40.0 * f64::from(k);
        lp_db.push((f, db(measured_gain(lp, f, fs)?, -120.0)));
    }
    // The chapter's checks, on the same filter and the same measurement.
    let dtft = |f: f64| -> f64 {
        let om = 2.0 * PI * f / fs;
        lp.iter()
            .enumerate()
            .map(|(j, &c)| Complex::from_polar(c, -om * j as f64))
            .sum::<Complex<f64>>()
            .norm()
    };
    for f in [500.0, 1000.0, 1500.0] {
        let g = measured_gain(lp, f, fs)?;
        check(
            format!("pass band flat at {f} Hz"),
            (20.0 * g.log10()).abs() < 0.05,
        );
        assert_close("measured vs DTFT", g, dtft(f), 1e-9);
    }
    let at_fc = measured_gain(lp, 2000.0, fs)?;
    assert_close("gain at the 2 kHz cutoff", at_fc, 0.5005, 1e-4);
    for f in [3000.0, 4000.0, 6000.0] {
        check(
            format!("stop band below -59 dB at {f} Hz"),
            20.0 * measured_gain(lp, f, fs)?.log10() < -59.0,
        );
    }
    let lp_theory: Vec<(f64, f64)> = (0..=800)
        .map(|i| {
            let f = 10.0 * f64::from(i);
            (f, db(dtft(f), -120.0))
        })
        .collect();

    // 8-tap moving average at 8 kHz.
    let ma = vec![1.0 / 8.0; 8];
    let fs_ma = 8_000.0;
    for (f, expected) in [
        (250.0, 0.901764),
        (500.0, 0.640729),
        (750.0, 0.304489),
        (1500.0, 0.224994),
    ] {
        assert_close(
            &format!("moving-average gain at {f} Hz"),
            measured_gain(&ma, f, fs_ma)?,
            expected,
            1e-6,
        );
    }
    for f in [1000.0, 2000.0, 3000.0] {
        check(
            format!("moving-average null at {f} Hz"),
            measured_gain(&ma, f, fs_ma)? < 1e-9,
        );
    }
    let mut ma_meas = Vec::new();
    for k in 0..40 {
        let f = 50.0 + 100.0 * f64::from(k);
        ma_meas.push((f, measured_gain(&ma, f, fs_ma)?));
    }
    // The Dirichlet kernel the chapter derives, as a reference curve.
    let dirichlet: Vec<(f64, f64)> = (1..=800)
        .map(|i| {
            let f = 5.0 * f64::from(i);
            let u = PI * f / fs_ma;
            (f, ((8.0 * u).sin() / (8.0 * u.sin())).abs())
        })
        .collect();

    figs.svg(
        "ch05-magnitude-response",
        "Magnitude responses measured by filtering sines with apply_filter (dots) against \
         the responses computed from the taps (lines). Top: the 63-tap low-pass, -6 dB at its \
         2 kHz cutoff and below -59 dB from 3 kHz. Bottom: the 8-tap moving average at 8 kHz, \
         which follows the Dirichlet kernel with nulls at multiples of 1 kHz.",
        (720, 600),
        |area| {
            let p = rows(area, 2);
            Chart::new("Frequency (Hz)", "Gain (dB)", (0.0, 8000.0), (-110.0, 8.0))
                .title("63-tap windowed-sinc low-pass, fs = 16 kHz")
                .x_format(khz)
                .add(Series::line("DTFT of the taps", lp_theory.clone(), BLUE))
                .add(Series::new(
                    "measured (apply_filter)",
                    lp_db.clone(),
                    VERMILION,
                    Style::Dots(2),
                ))
                .vrule(2000.0, "cutoff 2 kHz", GREY)
                .hrule(-6.02, "", GREY)
                .note(2150.0, -1.0, "-6 dB at the cutoff", GREY)
                .draw(&p[0])?;
            Chart::new("Frequency (Hz)", "Gain", (0.0, 4000.0), (0.0, 1.08))
                .title("8-tap moving average, fs = 8 kHz")
                .x_format(khz)
                .add(Series::line("Dirichlet kernel", dirichlet.clone(), BLUE))
                .add(Series::new(
                    "measured (apply_filter)",
                    ma_meas.clone(),
                    VERMILION,
                    Style::Dots(3),
                ))
                .draw(&p[1])
        },
    )
}

/// Full and valid convolution of a smooth pulse with the low-pass.
fn convolution(figs: &mut Figures, lp: &[f64]) -> Result {
    let pulse: Vec<f64> = (0..400)
        .map(|n| (-((n as f64 - 150.0) / 12.0).powi(2) / 2.0).exp())
        .collect();
    let full = mono(pulse.clone(), 16_000)?.convolve(&mono(lp.to_vec(), 16_000)?)?;
    let full = full.as_slice().ok_or("mono slice")?.to_vec();
    let valid = mono(pulse.clone(), 16_000)?.apply_filter(NonEmptySlice::new(lp).ok_or("taps")?)?;
    let valid = valid.as_slice().ok_or("mono slice")?.to_vec();
    assert_eq!(full.len(), 400 + 63 - 1, "full length N + M - 1");
    assert_eq!(valid.len(), 400 - 63 + 1, "valid length N - M + 1");
    for (a, b) in full[62..400].iter().zip(&valid) {
        assert_close("valid part of the full convolution", *a, *b, 1e-12);
    }
    let centroid = |v: &[f64]| {
        v.iter().enumerate().map(|(n, a)| n as f64 * a).sum::<f64>() / v.iter().sum::<f64>()
    };
    let (c_in, c_out) = (centroid(&pulse), centroid(&full));
    assert_close("centroid shift (M - 1)/2", c_out - c_in, 31.0, 1e-9);

    let x = indexed(pulse.iter().copied(), 1.0);
    let y = indexed(full.iter().copied(), 1.0);
    // apply_filter's output i belongs to time i + M - 1.
    let v: Vec<(f64, f64)> = valid
        .iter()
        .enumerate()
        .map(|(i, &a)| ((i + 62) as f64, a))
        .collect();
    figs.svg(
        "ch05-convolution",
        "A Gaussian pulse filtered by the 63-tap low-pass. The full convolution (convolve) \
         has the same shape, delayed by exactly (M - 1)/2 = 31 samples; apply_filter returns \
         its valid part, which starts at index M - 1 = 62.",
        SIZE,
        |area| {
            Chart::new("Sample index n", "Amplitude", (0.0, 462.0), (-0.1, 1.25))
                .add(Series::line("input pulse", x.clone(), GREY))
                .add(Series::line(
                    "convolve (full, 462 samples)",
                    y.clone(),
                    BLUE,
                ))
                .add(Series::new(
                    "apply_filter (valid, 338 samples)",
                    v.clone(),
                    VERMILION,
                    Style::Dashed(2),
                ))
                .vrule(c_in, "centroid 150", GREY)
                .vrule(c_out, "centroid 181", BLUE)
                .draw(area)
        },
    )
}
