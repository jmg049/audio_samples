//! Figures for chapter 08, LPC, the source-filter model and formants.

use std::f64::consts::PI;
use std::num::NonZeroUsize;

use audio_samples::educational::topics;
use audio_samples::{AudioSamples, AudioTransforms, lpc_analysis, lpc_residual, sample_rate};
use guide_figures::*;
use ndarray::Array1;
use num_complex::Complex64;

const FS: f64 = 16_000.0;
const FORMANTS: [(f64, f64); 3] = [(700.0, 80.0), (1220.0, 90.0), (2600.0, 120.0)];

fn main() -> Result {
    let mut figs = Figures::new("ch08")?;
    let v = Vowel::analyse()?;
    envelope(&mut figs, &v)?;
    poles(&mut figs, &v)?;
    figs.finish()
}

/// The chapter's vowel frame and its order-18 LPC analysis.
struct Vowel {
    frame: Vec<f32>,
    a_true: Vec<f64>,
    a_lpc: Vec<f64>,
    roots: Vec<Complex64>,
    /// Narrow resonances (F, B) in Hz, sorted by frequency.
    formants: Vec<(f64, f64)>,
}

impl Vowel {
    fn analyse() -> Result<Self> {
        // A(z) as the product of one second-order section per formant
        // (the chapter's hand-built reference filter).
        let mut a_true = vec![1.0f64];
        for &(f, b) in &FORMANTS {
            let (r, theta) = ((-PI * b / FS).exp(), 2.0 * PI * f / FS);
            let sec = [1.0, -2.0 * r * theta.cos(), r * r];
            let mut out = vec![0.0; a_true.len() + 2];
            for (i, ai) in a_true.iter().enumerate() {
                for (j, sj) in sec.iter().enumerate() {
                    out[i + j] += ai * sj;
                }
            }
            a_true = out;
        }
        // Source: one impulse every 133 samples, filtered by 1/A(z).
        let n = 2400;
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let e = if i % 133 == 0 { 1.0 } else { 0.0 };
            y[i] = e
                - (1..7)
                    .filter(|&k| k <= i)
                    .map(|k| a_true[k] * y[i - k])
                    .sum::<f64>();
        }
        let frame: Vec<f32> = y[1600..2112].iter().map(|&v| v as f32).collect();

        let p = topics::lpc(18, FS)
            .quantity("Rule-of-thumb order")
            .ok_or("order quantity")?
            .value as usize;
        assert_eq!(p, 18);
        let lpc = lpc_analysis(&frame, p);
        check("18 coefficients", lpc.coeffs.len() == 18);
        let e = lpc_residual(&frame, &lpc);
        let energy = |v: &[f32]| v.iter().map(|&s| f64::from(s).powi(2)).sum::<f64>();
        let gain_db = 10.0 * (energy(&frame) / energy(&e)).log10();
        assert_close("prediction gain (dB)", gain_db, 13.8, 0.05);

        let mut a_lpc = vec![1.0];
        a_lpc.extend(lpc.coeffs.iter().map(|&v| f64::from(v)));
        let roots = poly_roots(&a_lpc);
        check(
            "all poles inside the unit circle",
            roots.iter().all(|z| z.norm() < 1.0),
        );
        let mut formants: Vec<(f64, f64)> = roots
            .iter()
            .filter(|z| z.im > 0.0)
            .map(|z| (z.arg() * FS / (2.0 * PI), -z.norm().ln() * FS / PI))
            .filter(|&(_, b)| b < 400.0)
            .collect();
        formants.sort_by(|x, y| x.0.total_cmp(&y.0));
        check("three narrow resonances", formants.len() == 3);
        let errors: Vec<f64> = formants
            .iter()
            .zip(&FORMANTS)
            .map(|(e, t)| (e.0 - t.0).round())
            .collect();
        check(
            format!("formant errors {errors:?} = [11, -9, -3]"),
            errors == [11.0, -9.0, -3.0],
        );
        Ok(Self {
            frame,
            a_true,
            a_lpc,
            roots,
            formants,
        })
    }
}

/// Roots of z^p + c[1] z^(p-1) + ... + c[p] (c[0] = 1), Durand-Kerner, as in
/// the chapter.
fn poly_roots(c: &[f64]) -> Vec<Complex64> {
    let p = c.len() - 1;
    let eval = |z: Complex64| {
        c.iter()
            .fold(Complex64::new(0.0, 0.0), |acc, &ci| acc * z + ci)
    };
    let mut z: Vec<Complex64> = (0..p)
        .map(|i| Complex64::from_polar(0.9, 2.0 * PI * i as f64 / p as f64 + 0.4))
        .collect();
    for _ in 0..5000 {
        let mut biggest = 0.0f64;
        for i in 0..p {
            let den = (0..p)
                .filter(|&j| j != i)
                .fold(Complex64::new(1.0, 0.0), |d, j| d * (z[i] - z[j]));
            let step = eval(z[i]) / den;
            z[i] -= step;
            biggest = biggest.max(step.norm());
        }
        if biggest < 1e-14 {
            break;
        }
    }
    z
}

/// 20 log10 |1/A(e^{jw})| at `f` Hz.
fn envelope_db(a: &[f64], f: f64) -> f64 {
    let w = 2.0 * PI * f / FS;
    let s = a
        .iter()
        .enumerate()
        .fold(Complex64::new(0.0, 0.0), |acc, (k, &ak)| {
            acc + ak * Complex64::from_polar(1.0, -w * k as f64)
        });
    -20.0 * s.norm().log10()
}

/// FFT of the Hamming-windowed frame with the LPC envelope, as in the lab.
fn envelope(figs: &mut Figures, v: &Vowel) -> Result {
    let n = v.frame.len();
    let windowed: Vec<f64> = v
        .frame
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let w = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();
            w * f64::from(x)
        })
        .collect();
    let audio = AudioSamples::new_mono(Array1::from(windowed), sample_rate!(16000))?;
    let n_fft = 1024;
    let mag = audio.rfft(NonZeroUsize::new(n_fft).ok_or("n_fft")?)?;
    let df = FS / n_fft as f64;
    // Harmonic peaks (h f0, h = 1..33) minus the envelope: constant offset
    // plus a scatter of 1.06 dB (the lab's and the chapter's number).
    let harmonic_offsets = |a: &[f64]| -> Vec<f64> {
        (1..=33)
            .map(|h| {
                let f = f64::from(h) * 120.0;
                let k = (f / df).round() as usize;
                let kk = (k - 2..=k + 2)
                    .max_by(|&i, &j| mag[[0, i]].total_cmp(&mag[[0, j]]))
                    .unwrap_or(k);
                20.0 * mag[[0, kk]].log10() - envelope_db(a, f)
            })
            .collect()
    };
    let mean_of = |d: &[f64]| d.iter().sum::<f64>() / d.len() as f64;
    let diffs = harmonic_offsets(&v.a_lpc);
    let mean = mean_of(&diffs);
    // The true filter is drawn with its own offset, found the same way.
    let true_offset = mean_of(&harmonic_offsets(&v.a_true));
    let sd = (diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / diffs.len() as f64).sqrt();
    assert_close("harmonic scatter about the envelope (dB)", sd, 1.06, 0.005);

    let fft: Vec<(f64, f64)> = (0..mag.ncols())
        .map(|k| (k as f64 * df, 20.0 * mag[[0, k]].max(1e-12).log10()))
        .collect();
    let curve = |a: &[f64], offset: f64| -> Vec<(f64, f64)> {
        (0..=800)
            .map(|i| {
                let f = 10.0 * f64::from(i);
                (f, envelope_db(a, f) + offset)
            })
            .collect()
    };
    let (lpc_env, true_env) = (curve(&v.a_lpc, mean), curve(&v.a_true, true_offset));
    let top = fft.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let y_max = (top / 10.0).ceil() * 10.0 + 20.0;
    let y_min = y_max - 130.0;

    figs.svg(
        "ch08-envelope",
        "Spectrum of the Hamming-windowed vowel frame (grey) with the order-18 LPC envelope \
         from lpc_analysis (blue) and the true vocal-tract response (orange, dashed), each \
         shifted to sit on the harmonic peaks. Green lines mark the formants found from the \
         roots of A(z): 711, 1211 and 2597 Hz against 700, 1220 and 2600 Hz.",
        (720, 420),
        |area| {
            let mut c = Chart::new(
                "Frequency (Hz)",
                "Level (dB)",
                (0.0, 8000.0),
                (y_min, y_max),
            )
            .x_format(khz)
            .add(Series::new(
                "FFT of the frame",
                fft.clone(),
                GREY,
                Style::Line(1),
            ))
            .add(Series::line("LPC envelope, p = 18", lpc_env.clone(), BLUE))
            .add(Series::new(
                "true filter 1/A(z)",
                true_env.clone(),
                VERMILION,
                Style::Dashed(2),
            ))
            .legend(LegendPos::UpperRight);
            for &(f, _) in &v.formants {
                c = c.vrule(f, &format!("{f:.0} Hz"), GREEN);
            }
            c.draw(area)
        },
    )
}

/// The 18 LPC poles in the z-plane, with the narrow ones (formants) marked.
fn poles(figs: &mut Figures, v: &Vowel) -> Result {
    let narrow = |z: &Complex64| -z.norm().ln() * FS / PI < 400.0;
    let pts = |keep: bool| -> Vec<(f64, f64)> {
        v.roots
            .iter()
            .filter(|z| narrow(z) == keep)
            .map(|z| (z.re, z.im))
            .collect()
    };
    let circle: Vec<(f64, f64)> = (0..=360)
        .map(|i| {
            let t = 2.0 * PI * f64::from(i) / 360.0;
            (t.cos(), t.sin())
        })
        .collect();
    check("18 poles", v.roots.len() == 18);
    figs.svg(
        "ch08-poles",
        "The 18 roots of the LPC polynomial A(z) in the z-plane. The six narrow poles \
         (bandwidth under 400 Hz, orange) close to the unit circle are the three formants \
         and their conjugates; the broad poles (blue) shape the overall spectral slope.",
        (520, 500),
        |area| {
            Chart::new("Real part", "Imaginary part", (-1.15, 1.15), (-1.15, 1.15))
                .add(Series::new(
                    "unit circle",
                    circle.clone(),
                    GREY,
                    Style::Line(1),
                ))
                .add(Series::new(
                    "broad poles",
                    pts(false),
                    BLUE,
                    Style::Crosses(6),
                ))
                .add(Series::new(
                    "formant poles (B < 400 Hz)",
                    pts(true),
                    VERMILION,
                    Style::Crosses(6),
                ))
                .ticks(9, 9)
                .legend(LegendPos::LowerLeft)
                .draw(area)
        },
    )
}
