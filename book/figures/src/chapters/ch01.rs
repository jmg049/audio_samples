//! Figures for chapter 01, sampling and aliasing.

use std::f64::consts::PI;
use std::time::Duration;

use audio_samples::educational::topics;
use audio_samples::utils::generation::{square_wave, square_wave_bandlimited};
use audio_samples::{AudioTransforms, nzu, sample_rate, sine_wave};
use guide_figures::*;

fn main() -> Result {
    let mut figs = Figures::new("ch01")?;
    samples(&mut figs)?;
    folding(&mut figs)?;
    square_spectra(&mut figs)?;
    figs.finish()
}

/// An 11 kHz sine sampled at 16 kHz gives the samples of a 5 kHz sine with
/// inverted sign.
fn samples(figs: &mut Figures) -> Result {
    let fs = 16_000.0;
    // The library's samples: 11 kHz, amplitude 1, at 16 kHz.
    let x = sine_wave::<f64>(
        11_000.0,
        Duration::from_millis(10),
        sample_rate!(16000),
        1.0,
    );
    let x = x.as_slice().ok_or("mono slice")?.to_vec();
    // The chapter's identity: x[n] = -sin(2 pi 5000 n / fs), to 1e-9.
    for (n, &v) in x.iter().enumerate() {
        let b = -(2.0 * PI * 5_000.0 * n as f64 / fs).sin();
        assert_close("11 kHz sample vs -sin(5 kHz)", v, b, 1e-9);
    }
    let alias = topics::aliasing(11_000.0, fs)
        .quantity("Alias frequency")
        .ok_or("alias quantity")?
        .value;
    assert_close("alias of 11 kHz at 16 kHz", alias, 5_000.0, 0.0);

    let n_show = 9; // samples 0..=8: 0.5 ms
    let t_ms = |n: f64| 1000.0 * n / fs;
    let t_end = t_ms((n_show - 1) as f64);
    // Reference curves the chapter derives by hand (continuous time).
    let dense = |f: f64, sign: f64| -> Vec<(f64, f64)> {
        (0..=800)
            .map(|i| {
                let t = t_end * f64::from(i) / 800.0;
                (t, sign * (2.0 * PI * f * t / 1000.0).sin())
            })
            .collect()
    };
    let pts: Vec<(f64, f64)> = (0..n_show).map(|n| (t_ms(n as f64), x[n])).collect();

    figs.svg(
        "ch01-samples",
        "An 11 kHz sine (blue) sampled at 16 kHz. The samples (dots, from sine_wave) \
         lie exactly on a 5 kHz sine with inverted sign (orange): after sampling the two \
         tones cannot be told apart.",
        SIZE,
        |area| {
            Chart::new(
                "Time (ms)",
                "Amplitude (full scale)",
                (0.0, t_end),
                (-1.25, 1.45),
            )
            .add(Series::line("11 kHz tone", dense(11_000.0, 1.0), BLUE))
            .add(Series::new(
                "-sin, 5 kHz alias",
                dense(5_000.0, -1.0),
                ORANGE,
                Style::Dashed(2),
            ))
            .add(Series::new("samples at 16 kHz", pts, INK, Style::Dots(5)))
            .legend(LegendPos::UpperRight)
            .draw(area)
        },
    )
}

/// Alias frequency against input frequency at 16 kHz, from the library.
fn folding(figs: &mut Figures) -> Result {
    let fs = 16_000.0;
    let mut lib = Vec::new();
    for i in 0..=200 {
        let f = 250.0 * f64::from(i);
        let a = topics::aliasing(f, fs)
            .quantity("Alias frequency")
            .ok_or("alias quantity")?
            .value;
        // The chapter checks this sweep against the folding formula exactly.
        let fold = (f - (f / fs).round() * fs).abs();
        assert_close("folding formula", a, fold, 0.0);
        check("alias below Nyquist", a <= 8_000.0);
        lib.push((f / 1000.0, a / 1000.0));
    }
    let eleven = lib[44];
    assert_close("alias of 11 kHz", eleven.1, 5.0, 0.0);

    figs.svg(
        "ch01-folding",
        "Alias frequency of a tone sampled at 16 kHz, from topics::aliasing every 250 Hz. \
         The frequency axis folds at multiples of the 8 kHz Nyquist frequency; an 11 kHz \
         tone appears at 5 kHz.",
        SIZE,
        |area| {
            let mut c = Chart::new(
                "Input frequency (kHz)",
                "Apparent frequency (kHz)",
                (0.0, 50.0),
                (0.0, 9.5),
            )
            .add(Series::line("", lib.clone(), BLUE))
            .add(Series::new("", vec![eleven], VERMILION, Style::Dots(6)))
            .hrule(8.0, "Nyquist frequency 8 kHz", GREY)
            .note(
                eleven.0 + 0.8,
                eleven.1 - 0.1,
                "11 kHz appears at 5 kHz",
                VERMILION,
            );
            for k in 1..=3 {
                let f = 16.0 * f64::from(k);
                c = c.vrule(
                    f,
                    &format!("{}fs", if k == 1 { String::new() } else { k.to_string() }),
                    GREY,
                );
            }
            c.draw(area)
        },
    )
}

/// Spectra of the naive and band-limited 700 Hz square waves at 16 kHz.
fn square_spectra(figs: &mut Figures) -> Result {
    let one_s = Duration::from_secs(1);
    let naive = square_wave::<f64>(700.0, one_s, sample_rate!(16000), 1.0);
    let clean = square_wave_bandlimited::<f64>(700.0, one_s, sample_rate!(16000), 1.0);
    let s_naive = naive.rfft(nzu!(16000))?.row(0).to_vec(); // 1 Hz bins
    let s_clean = clean.rfft(nzu!(16000))?.row(0).to_vec();

    let odd_harmonic = |k: usize| k.is_multiple_of(700) && (k / 700) % 2 == 1;
    let stray_db = |s: &[f64]| {
        let total: f64 = s.iter().map(|m| m * m).sum();
        let stray: f64 = s
            .iter()
            .enumerate()
            .filter(|(k, _)| !odd_harmonic(*k))
            .map(|(_, m)| m * m)
            .sum();
        10.0 * (stray / total).log10()
    };
    assert_close(
        "aliased power of the naive square wave (dB)",
        stray_db(&s_naive),
        -14.77,
        0.01,
    );
    check(
        "band-limited square wave has no aliased power",
        stray_db(&s_clean) < -200.0,
    );
    let rel_db = 20.0 * (s_naive[6900] / s_naive[700]).log10();
    assert_close(
        "13th harmonic folded to 6900 Hz (dB)",
        rel_db,
        20.0 * (1.0f64 / 13.0).log10(),
        0.3,
    );
    check(
        "naive square wave has energy at 1300 Hz",
        s_naive[1300] / s_naive[700] > 1e-3,
    );

    let floor = -100.0;
    let rel = |s: &[f64]| -> Vec<(f64, f64)> {
        let f0 = s[700];
        s.iter()
            .enumerate()
            .map(|(k, &m)| (k as f64, db(m / f0, floor)))
            .collect()
    };
    let (rn, rc) = (rel(&s_naive), rel(&s_clean));
    figs.svg(
        "ch01-square-spectra",
        "Spectra of a 700 Hz square wave at 16 kHz, relative to the fundamental. The naive \
         wave (top) has aliased harmonics between the true odd harmonics, such as the 13th \
         folded to 6900 Hz and the 21st to 1300 Hz; the band-limited wave (bottom) has only \
         the odd harmonics up to 7700 Hz.",
        (720, 520),
        |area| {
            let panels = rows(area, 2);
            Chart::new(
                "Frequency (Hz)",
                "Level (dB re fundamental)",
                (0.0, 8000.0),
                (floor, 8.0),
            )
            .title("square_wave (naive)")
            .add(Series::line("", rn.clone(), VERMILION))
            .x_format(khz)
            .note(6900.0 + 60.0, -18.0, "13th, folded", INK)
            .note(1300.0 + 60.0, -30.0, "21st, folded", INK)
            .draw(&panels[0])?;
            Chart::new(
                "Frequency (Hz)",
                "Level (dB re fundamental)",
                (0.0, 8000.0),
                (floor, 8.0),
            )
            .title("square_wave_bandlimited")
            .add(Series::line("", rc.clone(), BLUE))
            .x_format(khz)
            .draw(&panels[1])
        },
    )
}
