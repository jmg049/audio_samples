//! Figures for chapter 06, IIR filters.

use std::f64::consts::PI;
use std::time::Duration;

use audio_samples::educational::ExplainFor;
use audio_samples::operations::types::{EqBand, FilterResponse, IirFilterDesign, ParametricEq};
use audio_samples::{AudioParametricEq, AudioSamples, nzu, sample_rate, sine_wave};
use guide_figures::*;
use num_complex::Complex64;

const FS: f64 = 16_000.0;
const FC: f64 = 1_000.0;

fn main() -> Result {
    let mut figs = Figures::new("ch06")?;
    magnitude(&mut figs)?;
    phase(&mut figs)?;
    pole_zero(&mut figs)?;
    notch(&mut figs)?;
    figs.finish()
}

fn to_db(m: &[f64]) -> Vec<f64> {
    m.iter().map(|&v| db(v, -200.0)).collect()
}

/// Magnitude responses of the chapter's Butterworth and Chebyshev designs.
fn magnitude(figs: &mut Figures) -> Result {
    let bw4 = IirFilterDesign::butterworth_lowpass(nzu!(4), FC);
    let bw8 = IirFilterDesign::butterworth_lowpass(nzu!(8), FC);
    let cheb = IirFilterDesign::chebyshev_i(FilterResponse::LowPass, nzu!(4), FC, 1.0);

    // The chapter's asserted numbers.
    let probes = [0.0, FC, 2.0 * FC, 4.0 * FC];
    let d4 = to_db(&bw4.frequency_response(&probes, FS)?.0);
    assert_close("Butterworth 4 at DC (dB)", d4[0], 0.0, 1e-9);
    assert_close("Butterworth 4 at fc (dB)", d4[1], -3.0103, 1e-4);
    assert_close("Butterworth 4 at 2 kHz (dB)", d4[2], -25.5, 0.05);
    assert_close("Butterworth 4 at 4 kHz (dB)", d4[3], -56.1, 0.05);
    let d8 = to_db(&bw8.frequency_response(&probes, FS)?.0);
    assert_close("Butterworth 8 at fc (dB)", d8[1], -3.0103, 1e-4);
    assert_close("Butterworth 8 at 2 kHz (dB)", d8[2], -51.0, 0.05);
    let dc = to_db(&cheb.frequency_response(&probes, FS)?.0);
    assert_close("Chebyshev I at DC (dB)", dc[0], -1.0, 1e-6);
    assert_close("Chebyshev I at fc (dB)", dc[1], -1.0, 1e-6);
    assert_close("Chebyshev I at 2 kHz (dB)", dc[2], -35.5, 0.05);

    let grid: Vec<f64> = (0..=1600).map(|i| 5.0 * f64::from(i)).collect();
    let curve = |d: &IirFilterDesign| -> Result<Vec<(f64, f64)>> {
        let (m, _) = d.frequency_response(&grid, FS)?;
        Ok(zip(grid.iter().copied(), to_db(&m)))
    };
    let (c4, c8, cc) = (curve(&bw4)?, curve(&bw8)?, curve(&cheb)?);
    let zoom = |c: &[(f64, f64)]| -> Vec<(f64, f64)> {
        c.iter().copied().filter(|p| p.0 <= 1500.0).collect()
    };

    figs.svg(
        "ch06-magnitude",
        "Magnitude responses from IirFilterDesign::frequency_response at 16 kHz, cutoff \
         1 kHz. Both Butterworth filters are -3.01 dB at the cutoff; at 2 kHz the 4th order \
         is 25.5 dB down, the 8th order 51.0 dB and the 4th-order Chebyshev I (1 dB ripple) \
         35.5 dB. The lower panel enlarges the pass band, where the Chebyshev ripples between \
         0 and -1 dB.",
        (720, 640),
        |area| {
            let p = rows(area, 2);
            Chart::new("Frequency (Hz)", "Gain (dB)", (0.0, 8000.0), (-100.0, 5.0))
                .x_format(khz)
                .add(Series::line("Butterworth, order 4", c4.clone(), BLUE))
                .add(Series::line("Butterworth, order 8", c8.clone(), GREEN))
                .add(Series::line(
                    "Chebyshev I, order 4, 1 dB",
                    cc.clone(),
                    VERMILION,
                ))
                .vrule(FC, "cutoff 1 kHz", GREY)
                .draw(&p[0])?;
            Chart::new("Frequency (Hz)", "Gain (dB)", (0.0, 1500.0), (-6.0, 0.5))
                .title("Pass band")
                .add(Series::line("Butterworth, order 4", zoom(&c4), BLUE))
                .add(Series::line("Butterworth, order 8", zoom(&c8), GREEN))
                .add(Series::line(
                    "Chebyshev I, order 4, 1 dB",
                    zoom(&cc),
                    VERMILION,
                ))
                .vrule(FC, "", GREY)
                .hrule(-3.0103, "-3.01 dB", GREY)
                .legend(LegendPos::LowerLeft)
                .draw(&p[1])
        },
    )
}

/// Unwraps a phase sequence in radians.
fn unwrap(ph: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(ph.len());
    let mut offset = 0.0;
    for (i, &p) in ph.iter().enumerate() {
        if i > 0 {
            let d = p + offset - out[i - 1];
            if d > PI {
                offset -= 2.0 * PI;
            } else if d < -PI {
                offset += 2.0 * PI;
            }
        }
        out.push(p + offset);
    }
    out
}

/// Phase response and group delay of the 4th-order Butterworth.
fn phase(figs: &mut Figures) -> Result {
    let design = IirFilterDesign::butterworth_lowpass(nzu!(4), FC);
    let grid: Vec<f64> = (0..=1600).map(|i| 5.0 * f64::from(i)).collect();
    let (_, ph) = design.frequency_response(&grid, FS)?;
    let deg: Vec<f64> = unwrap(&ph).iter().map(|p| p.to_degrees()).collect();
    let at_fc = deg[200];
    assert_close("phase at the cutoff (degrees)", at_fc, -180.0, 1e-6);

    // The chapter's group delay: central difference on the design's response.
    let d = 0.01;
    let group_delay = |f: f64| -> Result<f64> {
        let (m, p) = design.frequency_response(&[f - d, f + d], FS)?;
        let h = |i: usize| Complex64::from_polar(m[i], p[i]);
        Ok(-(h(1) / h(0)).arg() / (2.0 * PI * 2.0 * d / FS))
    };
    assert_close(
        "group delay at 1 kHz (samples)",
        group_delay(FC)?,
        9.66,
        0.01,
    );
    let mut gd = Vec::new();
    for i in 1..400 {
        let f = 10.0 * f64::from(i);
        gd.push((f, group_delay(f)?));
    }
    // Phase delay -arg H / omega, in samples: 8 at the cutoff.
    let pd: Vec<(f64, f64)> = grid
        .iter()
        .zip(&unwrap(&ph))
        .skip(1)
        .filter(|(f, _)| **f <= 4000.0)
        .map(|(&f, &p)| (f, -p / (2.0 * PI * f / FS)))
        .collect();
    assert_close("phase delay at 1 kHz (samples)", pd[199].1, 8.0, 1e-6);

    let phase_pts = zip(grid.iter().copied(), deg);
    figs.svg(
        "ch06-phase",
        "Phase response (top) and delay (bottom) of the 4th-order Butterworth low-pass, \
         1 kHz at 16 kHz, from IirFilterDesign::frequency_response. At the cutoff the phase \
         is -180 degrees, a phase delay of 8 samples, while the group delay there is 9.66 \
         samples: IIR phase is not linear in frequency.",
        (720, 600),
        |area| {
            let p = rows(area, 2);
            Chart::new(
                "Frequency (Hz)",
                "Phase (degrees)",
                (0.0, 8000.0),
                (-380.0, 20.0),
            )
            .x_format(khz)
            .add(Series::line("", phase_pts.clone(), BLUE))
            .vrule(FC, "cutoff: -180 degrees", VERMILION)
            .draw(&p[0])?;
            Chart::new(
                "Frequency (Hz)",
                "Delay (samples)",
                (0.0, 4000.0),
                (0.0, 12.0),
            )
            .x_format(khz)
            .add(Series::line("group delay", gd.clone(), BLUE))
            .add(Series::new(
                "phase delay",
                pd.clone(),
                ORANGE,
                Style::Dashed(2),
            ))
            .vrule(FC, "", GREY)
            .note(FC + 60.0, 11.4, "9.66 (group), 8 (phase)", INK)
            .draw(&p[1])
        },
    )
}

/// Roots of c0 z^2 + c1 z + c2.
fn quad_roots(c0: f64, c1: f64, c2: f64) -> [Complex64; 2] {
    let d = Complex64::new(c1 * c1 - 4.0 * c0 * c2, 0.0).sqrt();
    [(-c1 + d) / (2.0 * c0), (-c1 - d) / (2.0 * c0)]
}

/// Poles and zeros of the 4th-order Butterworth, from its SOS rows.
fn pole_zero(figs: &mut Figures) -> Result {
    let design = IirFilterDesign::butterworth_lowpass(nzu!(4), FC);
    let sos = design.sos_coefficients(FS)?;
    assert_eq!(sos.len(), 2, "N = 4 gives two biquads");
    let poles: Vec<Complex64> = sos
        .iter()
        .flat_map(|s| quad_roots(s[3], s[4], s[5]))
        .collect();
    let zeros: Vec<Complex64> = sos
        .iter()
        .flat_map(|s| quad_roots(s[0], s[1], s[2]))
        .collect();
    for z in &zeros {
        // A double root: sqrt of a rounding-level discriminant, hence 1e-6.
        assert_close(
            "zero at z = -1",
            (z - Complex64::new(-1.0, 0.0)).norm(),
            0.0,
            1e-6,
        );
    }
    let r_max = poles.iter().map(|p| p.norm()).fold(0.0, f64::max);
    check("stable: all poles inside the unit circle", r_max < 1.0);
    // The explainer's number, through a signal as in the chapter.
    let audio = sine_wave::<f64>(440.0, Duration::from_millis(50), sample_rate!(16000), 0.5);
    let doc = audio.explain(&design);
    let lib_r = doc
        .quantity("Largest pole radius")
        .ok_or("pole radius")?
        .value;
    assert_close("largest pole radius", lib_r, 0.862856, 1e-6);
    assert_close("largest pole radius from the SOS rows", r_max, lib_r, 1e-6);

    let circle: Vec<(f64, f64)> = (0..=360)
        .map(|i| {
            let t = f64::from(i).to_radians();
            (t.cos(), t.sin())
        })
        .collect();
    let pp: Vec<(f64, f64)> = poles.iter().map(|p| (p.re, p.im)).collect();
    let zz = vec![(-1.0, 0.0)];
    let top = poles
        .iter()
        .copied()
        .max_by(|a, b| a.im.total_cmp(&b.im))
        .ok_or("poles")?;
    figs.svg(
        "ch06-pole-zero",
        "Poles (crosses) and zeros (circle) of the 4th-order Butterworth low-pass, 1 kHz at \
         16 kHz, computed from its sos_coefficients. All four zeros are at z = -1 (Nyquist); \
         the poles lie inside the unit circle, the largest at radius 0.8629.",
        (560, 560),
        |area| {
            Chart::new("Real part", "Imaginary part", (-1.3, 1.3), (-1.3, 1.3))
                .ticks(7, 7)
                .add(Series::new(
                    "unit circle",
                    circle.clone(),
                    GREY,
                    Style::Line(1),
                ))
                .add(Series::new(
                    "poles",
                    pp.clone(),
                    VERMILION,
                    Style::Crosses(7),
                ))
                .add(Series::new(
                    "zeros (4 at z = -1)",
                    zz.clone(),
                    BLUE,
                    Style::Rings(7),
                ))
                .note(top.re + 0.06, top.im + 0.12, "r = 0.8629", VERMILION)
                .legend(LegendPos::UpperLeft)
                .draw(area)
        },
    )
}

/// The 50 Hz hum notch: response from the library and measured levels.
fn notch(figs: &mut Figures) -> Result {
    let (f0, q) = (50.0, 10.0);
    // The chapter's cookbook coefficients and -3 dB bandwidth.
    let w0 = 2.0 * PI * f0 / FS;
    let alpha = w0.sin() / (2.0 * q);
    let (b, a) = (
        [1.0, -2.0 * w0.cos(), 1.0],
        [1.0 + alpha, -2.0 * w0.cos(), 1.0 - alpha],
    );
    let gain = |f: f64| {
        let z1 = Complex64::from_polar(1.0, -2.0 * PI * f / FS);
        ((b[0] + b[1] * z1 + b[2] * z1 * z1) / (a[0] + a[1] * z1 + a[2] * z1 * z1)).norm()
    };
    check("exact zero at f0", gain(f0) < 1e-9);
    let edge = |mut lo: f64, mut hi: f64, rising: bool| {
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if (gain(mid).powi(2) < 0.5) == rising {
                lo = mid
            } else {
                hi = mid
            }
        }
        0.5 * (lo + hi)
    };
    let (lo_edge, hi_edge) = (edge(1.0, f0, false), edge(f0, 2.0 * f0, true));
    assert_close(
        "notch -3 dB bandwidth (Hz)",
        hi_edge - lo_edge,
        f0 / q,
        0.01,
    );

    // Hum plus tone through apply_eq_band, measured after settling.
    let dur = Duration::from_secs(2);
    let hum = sine_wave::<f64>(50.0, dur, sample_rate!(16000), 0.5);
    let tone = sine_wave::<f64>(1_000.0, dur, sample_rate!(16000), 0.5);
    let mix: Vec<f64> = hum
        .as_slice()
        .ok_or("mono slice")?
        .iter()
        .zip(tone.as_slice().ok_or("mono slice")?)
        .map(|(p, q)| p + q)
        .collect();
    let mix = AudioSamples::new_mono(ndarray::Array1::from(mix), sample_rate!(16000))?;
    let band = EqBand::band_stop(f0, q);
    let out = mix.apply_eq_band(&band)?;
    let tail = &out.as_slice().ok_or("mono slice")?[16_000..];
    let amplitude = |f: f64| {
        let (re, im) = tail
            .iter()
            .enumerate()
            .fold((0.0, 0.0), |(re, im), (n, &v)| {
                let ph = 2.0 * PI * f * n as f64 / FS;
                (re + v * ph.cos(), im - v * ph.sin())
            });
        2.0 * (re * re + im * im).sqrt() / tail.len() as f64
    };
    let hum_db = 20.0 * (amplitude(50.0) / 0.5).log10();
    let tone_db = 20.0 * (amplitude(1_000.0) / 0.5).log10();
    check("hum more than 150 dB down", hum_db < -150.0);
    check("tone unchanged to 0.001 dB", tone_db.abs() < 1e-3);

    // The library's response of the same band.
    let mut eq = ParametricEq::new();
    eq.add_band(band);
    // Log-spaced from 10 Hz to 2 kHz, plus a fine grid through the notch.
    let mut grid: Vec<f64> = (0..=1000)
        .map(|i| 10.0 * 200f64.powf(f64::from(i) / 1000.0))
        .chain((0..=400).map(|i| 45.0 + 0.025 * f64::from(i)))
        .collect();
    grid.sort_by(f64::total_cmp);
    let (m, _) = mix.eq_frequency_response(&eq, &grid)?;
    for (&f, &g) in grid.iter().zip(&m) {
        assert_close("library response vs cookbook formula", g, gain(f), 1e-9);
    }
    let floor = -60.0;
    let curve = zip(grid.iter().copied(), m.iter().map(|&g| db(g, floor)));
    let measured = vec![(50.0, floor), (1_000.0, tone_db)];

    figs.svg(
        "ch06-notch",
        "The 50 Hz notch (EqBand::band_stop, Q = 10) at 16 kHz: response from \
         eq_frequency_response, -3 dB edges 5 Hz apart. Dots: levels measured on hum plus a \
         1 kHz tone filtered with apply_eq_band; the hum is 160 dB down (below the axis) and \
         the tone is unchanged.",
        SIZE,
        |area| {
            Chart::new("Frequency (Hz)", "Gain (dB)", (10.0, 2000.0), (floor, 5.0))
                .log_x()
                .add(Series::line(
                    "response (eq_frequency_response)",
                    curve.clone(),
                    BLUE,
                ))
                .add(Series::new(
                    "measured (apply_eq_band)",
                    measured.clone(),
                    VERMILION,
                    Style::Dots(5),
                ))
                .hrule(-3.0103, "", GREY)
                .note(12.0, -4.5, "-3 dB", GREY)
                .note(56.0, -52.0, "hum: -160 dB", VERMILION)
                .legend(LegendPos::LowerRight)
                .draw(area)
        },
    )
}
