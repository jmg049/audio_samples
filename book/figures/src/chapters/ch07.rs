//! Figures for chapter 07, resampling.

use std::f64::consts::PI;
use std::time::Duration;

use audio_samples::operations::types::ResamplingQuality;
use audio_samples::{AudioSamples, AudioTransforms, nzu, resample, sample_rate, sine_wave};
use guide_figures::*;
use non_empty_slice::NonEmptyVec;

const TIERS: [(ResamplingQuality, &str, plotters::style::RGBColor); 3] = [
    (ResamplingQuality::Fast, "Fast", VERMILION),
    (ResamplingQuality::Medium, "Medium", GREEN),
    (ResamplingQuality::High, "High", BLUE),
];

fn main() -> Result {
    let mut figs = Figures::new("ch07")?;
    tier_responses(&mut figs)?;
    alias_spectra(&mut figs)?;
    upsampling_images(&mut figs)?;
    figs.finish()
}

/// The chapter's one-bin DFT at `f` Hz.
fn bin(x: &[f64], f: f64, fs: f64) -> (f64, f64) {
    x.iter().enumerate().fold((0.0, 0.0), |(re, im), (n, &v)| {
        let ph = 2.0 * PI * f * n as f64 / fs;
        (re + v * ph.cos(), im - v * ph.sin())
    })
}
fn amplitude(x: &[f64], f: f64, fs: f64) -> f64 {
    let (re, im) = bin(x, f, fs);
    2.0 * (re * re + im * im).sqrt() / x.len() as f64
}
fn phase(x: &[f64], f: f64, fs: f64) -> f64 {
    let (re, im) = bin(x, f, fs);
    im.atan2(re)
}
fn to_db(v: f64) -> f64 {
    20.0 * v.log10()
}
fn fold(f: f64, fs: f64) -> f64 {
    (f - (f / fs).round() * fs).abs()
}

/// Level of a 48 kHz tone after conversion to 16 kHz, measured at the
/// frequency it appears at, for every tier: the pass band below 8 kHz and the
/// alias rejection above.
fn tier_responses(figs: &mut Figures) -> Result {
    let dur = Duration::from_millis(250);
    let mid = 1_200..2_800;
    let mut curves: Vec<Vec<(f64, f64)>> = vec![Vec::new(); 3];
    for k in 1..240 {
        let f = 100.0 * f64::from(k);
        let fa = fold(f, 16_000.0);
        if fa == 0.0 || fa == 8_000.0 {
            continue; // a tone at DC or Nyquist has no well-defined one-bin level
        }
        let x = sine_wave::<f64>(f, dur, sample_rate!(48000), 0.5);
        for (i, (q, _, _)) in TIERS.iter().enumerate() {
            let y = resample(&x, sample_rate!(16000), *q)?;
            let y = y.as_slice().ok_or("mono slice")?;
            let level = to_db(amplitude(&y[mid.clone()], fa, 16_000.0) / 0.5);
            curves[i].push((f, level.max(-200.0)));
        }
    }
    // The chapter's 10 kHz rejection, measured here on a lone 10 kHz tone.
    let at_10k = |c: &[(f64, f64)]| c.iter().find(|p| p.0 == 10_000.0).map_or(f64::NAN, |p| p.1);
    for (c, want) in curves.iter().zip([-90.8, -116.7, -145.5]) {
        assert_close("rejection of a 10 kHz tone (dB)", at_10k(c), want, 0.1);
    }

    figs.svg(
        "ch07-tier-responses",
        "Level after converting a 48 kHz tone to 16 kHz with resample, measured at the \
         frequency at which it appears, for each ResamplingQuality. Below 8 kHz this is the pass \
         band (Fast rolls off first); above 8 kHz it is the residue of the alias, 90.8, 116.7 \
         and 145.5 dB down at 10 kHz.",
        (720, 440),
        |area| {
            let mut c = Chart::new(
                "Input frequency (Hz)",
                "Level (dB re input)",
                (0.0, 24_000.0),
                (-200.0, 10.0),
            )
            .x_format(khz)
            .vrule(8_000.0, "output Nyquist 8 kHz", GREY)
            .legend(LegendPos::UpperRight);
            for ((_, name, color), pts) in TIERS.iter().zip(&curves) {
                c = c.add(Series::line(name, pts.clone(), *color));
            }
            c.draw(area)
        },
    )
}

/// Spectra of the 1 kHz + 10 kHz mix after naive decimation and after each tier.
fn alias_spectra(figs: &mut Figures) -> Result {
    let dur = Duration::from_millis(250);
    let a = sine_wave::<f64>(1_000.0, dur, sample_rate!(48000), 0.5);
    let b = sine_wave::<f64>(10_000.0, dur, sample_rate!(48000), 0.5);
    let x: Vec<f64> = a
        .as_slice()
        .ok_or("mono slice")?
        .iter()
        .zip(b.as_slice().ok_or("mono slice")?)
        .map(|(p, q)| p + q)
        .collect();
    let input: AudioSamples<f64> = AudioSamples::from_mono_vec(
        NonEmptyVec::new(x.clone()).map_err(|_| "empty")?,
        sample_rate!(48000),
    );
    let mid = 1_200..2_800;
    let ideal: Vec<f64> = (0..4_000)
        .map(|n| 0.5 * (2.0 * PI * 1_000.0 * n as f64 / 16_000.0).sin())
        .collect();
    let ideal_phase = phase(&ideal[mid.clone()], 1_000.0, 16_000.0);

    // Spectrum of the middle 1600 samples (10 Hz bins) with the library's rfft,
    // in dB relative to the 0.5 input amplitude.
    let floor = -200.0;
    let spectrum = |seg: &[f64]| -> Result<Vec<(f64, f64)>> {
        let s: AudioSamples<f64> = AudioSamples::from_mono_vec(
            NonEmptyVec::new(seg.to_vec()).map_err(|_| "empty")?,
            sample_rate!(16000),
        );
        let mag = s.rfft(nzu!(1600))?.row(0).to_vec();
        Ok(mag
            .iter()
            .enumerate()
            .map(|(k, &m)| (10.0 * k as f64, db(2.0 * m / 1600.0 / 0.5, floor)))
            .collect())
    };

    let naive: Vec<f64> = x.iter().step_by(3).copied().collect();
    let naive_alias = to_db(amplitude(&naive[mid.clone()], 6_000.0, 16_000.0) / 0.5);
    assert_close("naive alias level (dB)", naive_alias, 0.0, 1e-9);
    let naive_spec = spectrum(&naive[mid.clone()])?;
    assert_close(
        "naive alias in the rfft bin (dB)",
        naive_spec[600].1,
        0.0,
        1e-9,
    );

    let mut tier_specs = Vec::new();
    for ((q, name, _), want) in TIERS.iter().zip([-90.8, -116.7, -145.5]) {
        let y = resample(&input, sample_rate!(16000), *q)?;
        assert_eq!(y.samples_per_channel().get(), 4_000, "ceil(12000 / 3)");
        let y = y.as_slice().ok_or("mono slice")?;
        let seg = &y[mid.clone()];
        let pass_db = to_db(amplitude(seg, 1_000.0, 16_000.0) / 0.5);
        let alias_db = to_db(amplitude(seg, 6_000.0, 16_000.0) / 0.5);
        let shift = (phase(seg, 1_000.0, 16_000.0) - ideal_phase) / (2.0 * PI / 16.0);
        check(
            format!("{name} passes 1 kHz unchanged"),
            pass_db.abs() < 1e-3,
        );
        assert_close(&format!("{name} alias rejection (dB)"), alias_db, want, 0.1);
        check(format!("{name} adds no delay"), shift.abs() < 1e-9);
        let spec = spectrum(seg)?;
        assert_close("alias in the rfft bin (dB)", spec[600].1, alias_db, 1e-6);
        tier_specs.push(spec);
    }

    figs.svg(
        "ch07-alias-suppression",
        "A 1 kHz plus 10 kHz mix converted from 48 kHz to 16 kHz, spectra of the middle \
         100 ms. Keeping every third sample (top) moves the 10 kHz tone to 6 kHz at full \
         level; every tier of resample (bottom) removes it, leaving 90.8 dB (Fast), 116.7 dB \
         (Medium) and 145.5 dB (High) below the input.",
        (720, 600),
        |area| {
            let p = rows(area, 2);
            Chart::new(
                "Frequency (Hz)",
                "Level (dB re input)",
                (0.0, 8_000.0),
                (floor, 15.0),
            )
            .title("Naive decimation (every third sample)")
            .x_format(khz)
            .add(Series::line("", naive_spec.clone(), INK))
            .note(6_150.0, -6.0, "10 kHz alias at 6 kHz, 0 dB", INK)
            .draw(&p[0])?;
            let mut c = Chart::new(
                "Frequency (Hz)",
                "Level (dB re input)",
                (0.0, 8_000.0),
                (floor, 15.0),
            )
            .title("resample, three quality tiers")
            .x_format(khz)
            .legend(LegendPos::UpperRight);
            for (((_, name, color), s), want) in
                TIERS.iter().zip(&tier_specs).zip([-90.8, -116.7, -145.5])
            {
                c = c.add(Series::line(name, s.clone(), *color)).note(
                    6_150.0,
                    want + 8.0,
                    &format!("{name}: {want} dB"),
                    *color,
                );
            }
            c.draw(&p[1])
        },
    )
}

/// Upsampling 16 kHz to 48 kHz: linear interpolation (Fast) against the
/// sinc tiers.
fn upsampling_images(figs: &mut Figures) -> Result {
    let fs = 48_000.0;
    // The chapter's hand-derived gain of linear interpolation, in dB.
    let g = |f: f64| {
        let w = 2.0 * PI * f / fs;
        20.0 * ((3.0 + 4.0 * w.cos() + 2.0 * (2.0 * w).cos()) / 9.0)
            .abs()
            .log10()
    };
    let level = |y: &[f64], f: f64| to_db(amplitude(y, f, fs) / 0.5);
    let up = |f: f64, q: ResamplingQuality| -> Result<Vec<f64>> {
        let x = sine_wave::<f64>(f, Duration::from_millis(250), sample_rate!(16000), 0.5);
        let y = resample(&x, sample_rate!(48000), q)?;
        Ok(y.as_slice().ok_or("mono slice")?[3_600..8_400].to_vec())
    };

    // The chapter's numbers.
    let y6 = up(6_000.0, ResamplingQuality::Fast)?;
    assert_close(
        "Fast, 6 kHz tone (dB)",
        level(&y6, 6_000.0),
        g(6_000.0),
        1e-6,
    );
    assert_close(
        "Fast, 10 kHz image (dB)",
        level(&y6, 10_000.0),
        g(10_000.0),
        1e-6,
    );
    assert_close("G(6 kHz) (dB)", g(6_000.0), -3.77, 0.01);
    assert_close("G(10 kHz) (dB)", g(10_000.0), -11.8, 0.05);
    let y2 = up(2_000.0, ResamplingQuality::Fast)?;
    assert_close("Fast droop at 2 kHz (dB)", level(&y2, 2_000.0), -0.40, 0.01);
    for (q, image_db) in [
        (ResamplingQuality::Medium, -114.8),
        (ResamplingQuality::High, -145.2),
    ] {
        let y = up(6_000.0, q)?;
        check(
            "sinc tier passes 6 kHz unchanged",
            level(&y, 6_000.0).abs() < 1e-4,
        );
        assert_close(
            "sinc tier, 10 kHz image (dB)",
            level(&y, 10_000.0),
            image_db,
            0.1,
        );
    }

    // Sweep: tone level and first image (16 kHz - f) for every tier.
    let mut tone_fast = Vec::new();
    let mut images: Vec<Vec<(f64, f64)>> = vec![Vec::new(); 3];
    for k in 1..80 {
        let f = 100.0 * f64::from(k);
        for (i, (q, _, _)) in TIERS.iter().enumerate() {
            let y = up(f, *q)?;
            if i == 0 {
                tone_fast.push((f, level(&y, f)));
            }
            images[i].push((16_000.0 - f, level(&y, 16_000.0 - f).max(-200.0)));
        }
    }
    let curve: Vec<(f64, f64)> = (1..=2399)
        .map(|i| {
            let f = 10.0 * f64::from(i);
            (f, g(f).max(-200.0))
        })
        .collect();

    figs.svg(
        "ch07-upsampling-images",
        "Upsampling a tone from 16 kHz to 48 kHz. Fast (linear interpolation) follows the \
         chapter's G(f) (line): the tone droops (dots below 8 kHz) and its image at 16 kHz \
         minus f is only partly removed. Medium and High remove the image to below -116 dB.",
        (720, 440),
        |area| {
            let mut c = Chart::new(
                "Output frequency (Hz)",
                "Level (dB re input)",
                (0.0, 16_000.0),
                (-200.0, 10.0),
            )
            .x_format(khz)
            .add(Series::new(
                "G(f), linear interpolation",
                curve.clone(),
                INK,
                Style::Line(1),
            ))
            .add(Series::new(
                "Fast, tone",
                tone_fast.clone(),
                VERMILION,
                Style::Dots(3),
            ))
            .vrule(8_000.0, "input Nyquist 8 kHz", GREY)
            .legend(LegendPos::LowerLeft);
            for (i, ((_, name, color), pts)) in TIERS.iter().zip(&images).enumerate() {
                // Fast's image lies on G(f): dots, so the curve stays visible.
                let style = if i == 0 {
                    Style::Rings(3)
                } else {
                    Style::Line(2)
                };
                c = c.add(Series::new(
                    &format!("{name}, image at 16 kHz - f"),
                    pts.clone(),
                    *color,
                    style,
                ));
            }
            c.draw(area)
        },
    )
}
