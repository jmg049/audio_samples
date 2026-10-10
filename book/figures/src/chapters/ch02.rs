//! Figures for chapter 02, quantisation and dither.

use std::time::Duration;

use audio_samples::{
    AudioSamples, AudioTransforms, AudioTypeConversion, nzu, sample_rate, sine_wave, white_noise,
};
use guide_figures::*;

fn main() -> Result {
    let mut figs = Figures::new("ch02")?;
    let q = Quiet::compute()?;
    staircase(&mut figs, &q)?;
    dither_spectra(&mut figs, &q)?;
    figs.finish()
}

/// The chapter's quiet case: a 997 Hz sine of amplitude 2.5 LSB at 16 kHz,
/// rounded to i16 with and without TPDF dither.
struct Quiet {
    lsb: f64,
    xs: Vec<f64>,
    plain: Vec<f64>,
    e_plain: Vec<f64>,
    e_dith: Vec<f64>,
}

impl Quiet {
    fn compute() -> Result<Self> {
        let fs = sample_rate!(16000);
        let one_s = Duration::from_secs(1);
        let lsb = 1.0 / 32768.0;
        let x = sine_wave::<f64>(997.0, one_s, fs, 2.5 * lsb);
        let xs = x.as_slice().ok_or("mono slice")?.to_vec();

        let u1 = white_noise::<f64>(one_s, fs, lsb / 2.0, Some(1));
        let u2 = white_noise::<f64>(one_s, fs, lsb / 2.0, Some(2));
        let dither: Vec<f64> = u1
            .as_slice()
            .ok_or("mono slice")?
            .iter()
            .zip(u2.as_slice().ok_or("mono slice")?)
            .map(|(a, b)| a + b)
            .collect();

        let to_i16_and_back = |v: Vec<f64>| -> Result<Vec<f64>> {
            Ok(AudioSamples::new_mono(ndarray::Array1::from(v), fs)?
                .to_format::<i16>()
                .to_format::<f64>()
                .as_slice()
                .ok_or("mono slice")?
                .to_vec())
        };
        let plain = to_i16_and_back(xs.clone())?;
        let dith = to_i16_and_back(xs.iter().zip(&dither).map(|(a, d)| a + d).collect())?;
        let e_plain: Vec<f64> = plain.iter().zip(&xs).map(|(y, v)| y - v).collect();
        let e_dith: Vec<f64> = dith.iter().zip(&xs).map(|(y, v)| y - v).collect();

        // The chapter's numbers: error powers relative to Δ²/12, and the
        // correlation between error and signal.
        let power = |v: &[f64]| v.iter().map(|a| a * a).sum::<f64>() / v.len() as f64;
        assert_close(
            "dithered error power / (Δ²/12)",
            power(&e_dith) / (lsb * lsb / 12.0),
            3.0,
            0.1,
        );
        assert_close(
            "undithered error power / (Δ²/12)",
            power(&e_plain) / (lsb * lsb / 12.0),
            1.23,
            0.01,
        );
        let corr = |e: &[f64]| {
            let num: f64 = e.iter().zip(&xs).map(|(a, b)| a * b).sum();
            num / ((power(e) * power(&xs)).sqrt() * xs.len() as f64)
        };
        check(
            "undithered error correlates with the signal",
            corr(&e_plain) < -0.5,
        );
        check(
            "dithered error is uncorrelated with the signal",
            corr(&e_dith).abs() < 0.03,
        );
        Ok(Self {
            lsb,
            xs,
            plain,
            e_plain,
            e_dith,
        })
    }
}

/// The quantised quiet sine and its rounding error, in LSB.
fn staircase(figs: &mut Figures, q: &Quiet) -> Result {
    let n_show = 49; // three periods of 997 Hz at 16 kHz, about 3 ms
    let t_ms = |n: usize| n as f64 / 16.0;
    let in_lsb = |v: &[f64]| -> Vec<(f64, f64)> {
        v.iter()
            .take(n_show)
            .enumerate()
            .map(|(n, &a)| (t_ms(n), a / q.lsb))
            .collect()
    };
    let x = in_lsb(&q.xs);
    let y = in_lsb(&q.plain);
    let e = in_lsb(&q.e_plain);
    let t_end = t_ms(n_show - 1);

    figs.svg(
        "ch02-staircase",
        "Top: a 997 Hz sine of amplitude 2.5 LSB at 16 kHz (blue) and its value after \
         to_format::<i16>() (orange steps). Bottom: the rounding error, which stays within \
         half a step but repeats with the signal.",
        (720, 560),
        |area| {
            let panels = rows(area, 2);
            Chart::new("Time (ms)", "Amplitude (LSB)", (0.0, t_end), (-3.5, 3.5))
                .add(Series::new(
                    "input samples",
                    x.clone(),
                    BLUE,
                    Style::Line(2),
                ))
                .add(Series::new(
                    "quantised (i16)",
                    y.clone(),
                    ORANGE,
                    Style::Steps(2),
                ))
                .add(Series::new("", y.clone(), ORANGE, Style::Dots(3)))
                .ticks(10, 8)
                .legend(LegendPos::UpperRight)
                .draw(&panels[0])?;
            Chart::new("Time (ms)", "Error (LSB)", (0.0, t_end), (-0.75, 0.75))
                .add(Series::new("", e.clone(), VERMILION, Style::Stems))
                .hrule(0.5, "+half step", GREY)
                .hrule(-0.5, "-half step", GREY)
                .ticks(10, 7)
                .draw(&panels[1])
        },
    )
}

/// Error spectra without and with TPDF dither, relative to the mean bin power.
fn dither_spectra(figs: &mut Figures, q: &Quiet) -> Result {
    let fs = sample_rate!(16000);
    // The chapter's measure: each bin's power against the mean bin power.
    let rel_db = |e: &[f64]| -> Result<Vec<f64>> {
        let a = AudioSamples::new_mono(ndarray::Array1::from(e.to_vec()), fs)?;
        let m = a.rfft(nzu!(16000))?.row(0).to_vec(); // 1 Hz bins
        let mean = m.iter().map(|v| v * v).sum::<f64>() / m.len() as f64;
        Ok(m.iter()
            .map(|v| 10.0 * (v * v / mean).max(1e-6).log10())
            .collect())
    };
    let plain = rel_db(&q.e_plain)?;
    let dith = rel_db(&q.e_dith)?;
    check("undithered 2991 Hz spur > 30 dB", plain[2991] > 30.0);
    check("dithered 2991 Hz bin < 3 dB", dith[2991] < 3.0);

    let pts = |v: &[f64]| -> Vec<(f64, f64)> {
        v.iter().enumerate().map(|(k, &d)| (k as f64, d)).collect()
    };
    let (pp, pd) = (pts(&plain), pts(&dith));
    figs.svg(
        "ch02-dither-spectra",
        "Spectra of the quantisation error of the 2.5 LSB sine, each bin relative to the \
         mean bin power. Without dither (top) the error is concentrated at harmonics of \
         997 Hz, such as the 3rd at 2991 Hz; with TPDF dither (bottom) it is flat noise.",
        (720, 560),
        |area| {
            let panels = rows(area, 2);
            Chart::new(
                "Frequency (Hz)",
                "Power (dB re mean bin)",
                (0.0, 8000.0),
                (-40.0, 45.0),
            )
            .title("No dither")
            .add(Series::line("", pp.clone(), VERMILION))
            .vrule(2991.0, "3rd harmonic, 2991 Hz", GREY)
            .x_format(khz)
            .draw(&panels[0])?;
            Chart::new(
                "Frequency (Hz)",
                "Power (dB re mean bin)",
                (0.0, 8000.0),
                (-40.0, 45.0),
            )
            .title("TPDF dither")
            .add(Series::line("", pd.clone(), BLUE))
            .vrule(2991.0, "2991 Hz", GREY)
            .x_format(khz)
            .draw(&panels[1])
        },
    )
}
