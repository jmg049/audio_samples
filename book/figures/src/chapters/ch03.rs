//! Figures for chapter 03, the DFT, windows and leakage.

use std::num::NonZeroUsize;
use std::time::Duration;

use audio_samples::educational::{Explain, ExplainContext};
use audio_samples::{AudioProcessing, AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
use guide_figures::*;
use spectrograms::{WindowType, make_window};

fn main() -> Result {
    let mut figs = Figures::new("ch03")?;
    leakage(&mut figs)?;
    windows(&mut figs)?;
    zero_padding(&mut figs)?;
    figs.finish()
}

fn nz(n: usize) -> Result<NonZeroUsize> {
    Ok(NonZeroUsize::new(n).ok_or("zero length")?)
}

/// The chapter's measurement: a 512-sample tone at 16 kHz, windowed, rfft.
fn spectrum(f: f64, wt: WindowType) -> Result<Vec<f64>> {
    let w = make_window::<f64>(wt, nzu!(512));
    Ok(
        sine_wave::<f64>(f, Duration::from_millis(32), sample_rate!(16000), 1.0)
            .apply_window_f64(&w)?
            .rfft(nzu!(512))?
            .row(0)
            .to_vec(),
    )
}

fn peak(s: &[f64]) -> f64 {
    s.iter().copied().fold(0.0, f64::max)
}

fn to_db(r: f64) -> f64 {
    20.0 * r.log10()
}

/// Bin-centred (1000 Hz) and worst-case (1015.625 Hz) tones, rectangular and
/// Hann windows, N = 512 at 16 kHz.
fn leakage(figs: &mut Figures) -> Result {
    // (window, title, peak drop in dB, leakage at bin 52 in dB): the chapter's numbers.
    let cases = [
        (
            WindowType::Rectangular,
            "Rectangular (no window)",
            3.86,
            -34.0,
        ),
        (WindowType::Hanning, "Hann", 1.42, -86.0),
    ];
    let floor = -140.0;
    let mut panels_data = Vec::new();
    for (wt, title, drop, leak) in cases {
        let on = spectrum(1000.0, wt.clone())?;
        let off = spectrum(1015.625, wt)?;
        assert_close(
            &format!("{title}: off-bin peak drop (dB)"),
            to_db(peak(&on) / peak(&off)),
            drop,
            0.01,
        );
        assert_close(
            &format!("{title}: bin 52 below the off-bin peak (dB)"),
            to_db(off[52] / peak(&off)),
            leak,
            0.1,
        );
        let reference = peak(&on);
        let pts = |s: &[f64]| -> Vec<(f64, f64)> {
            s.iter()
                .enumerate()
                .map(|(k, &m)| (k as f64 * 31.25, db(m / reference, floor)))
                .collect()
        };
        panels_data.push((title, pts(&on), pts(&off)));
    }

    figs.svg(
        "ch03-leakage",
        "Spectra of a 512-sample tone at 16 kHz (bins 31.25 Hz apart), relative to the \
         bin-centred peak. At 1000 Hz (bin 32, blue) the tone falls on one bin; at \
         1015.625 Hz (bin 32.5, orange) it leaks into every bin. At bin 52 the leakage is \
         34 dB below the peak without a window and 86 dB below with a Hann window.",
        (720, 600),
        |area| {
            let panels = rows(area, 2);
            for (panel, (title, on, off)) in panels.iter().zip(&panels_data) {
                Chart::new(
                    "Frequency (Hz)",
                    "Level (dB re bin-centred peak)",
                    (0.0, 4000.0),
                    (floor, 10.0),
                )
                .title(title)
                .add(Series::line("1000 Hz, bin-centred", on.clone(), BLUE))
                .add(Series::line(
                    "1015.625 Hz, half a bin off",
                    off.clone(),
                    ORANGE,
                ))
                .vrule(52.0 * 31.25, "bin 52", GREY)
                .x_format(khz)
                .legend(LegendPos::UpperRight)
                .draw(panel)?;
            }
            Ok(())
        },
    )
}

/// Window shapes and their spectra, N = 512, with the explainer's figures.
fn windows(figs: &mut Figures) -> Result {
    let pad = 16; // zero-pad 16x to draw the continuous window spectrum
    let n = 512;
    // (window, name, coherent gain, ENBW, peak side-lobe level): the chapter's table.
    let table = [
        (WindowType::Rectangular, "Rectangular", 1.0, 1.0, -13.26),
        (WindowType::Hanning, "Hann", 0.4990, 1.5029, -31.47),
        (WindowType::Hamming, "Hamming", 0.5391, 1.3647, -42.67),
        (WindowType::Blackman, "Blackman", 0.4192, 1.7301, -58.11),
    ];
    let floor = -120.0;
    let mut shapes = Vec::new();
    let mut spectra = Vec::new();
    for ((wt, name, cg, enbw, psl), color) in table.into_iter().zip(PALETTE) {
        let doc = wt.explain_with(
            &ExplainContext::new()
                .with_n_fft(512)
                .with_sample_rate(16_000.0),
        );
        let q = |k: &str| -> Result<f64> { Ok(doc.quantity(k).ok_or("missing quantity")?.value) };
        assert_close(
            &format!("{name} coherent gain"),
            q("Coherent gain")?,
            cg,
            1e-4,
        );
        assert_close(
            &format!("{name} ENBW"),
            q("Equivalent noise bandwidth")?,
            enbw,
            1e-4,
        );
        assert_close(
            &format!("{name} PSL"),
            q("Peak side-lobe level")?,
            psl,
            0.01,
        );

        let w = make_window::<f64>(wt, nz(n)?);
        let wv: Vec<f64> = w.iter().copied().collect();
        shapes.push(Series::line(
            name,
            wv.iter().enumerate().map(|(i, &v)| (i as f64, v)).collect(),
            color,
        ));
        let m = AudioSamples::new_mono(ndarray::Array1::from(wv), sample_rate!(16000))?
            .rfft(nz(n * pad)?)?
            .row(0)
            .to_vec();
        let m0 = m[0];
        let rel: Vec<f64> = m.iter().map(|&v| db(v / m0, floor)).collect();
        // The plotted curve shows the same peak side-lobe level: the highest
        // local maximum after the first null.
        let first_null = (1..rel.len() - 1)
            .find(|&i| rel[i] <= rel[i - 1] && rel[i] <= rel[i + 1])
            .ok_or("no null")?;
        let measured = rel[first_null..].iter().copied().fold(f64::MIN, f64::max);
        assert_close(
            &format!("{name} PSL read off the plotted spectrum"),
            measured,
            psl,
            0.1,
        );
        let bins = 12 * pad;
        spectra.push(Series::line(
            &format!("{name} (PSL {psl} dB)"),
            rel.iter()
                .take(bins + 1)
                .enumerate()
                .map(|(i, &v)| (i as f64 / pad as f64, v))
                .collect(),
            color,
        ));
    }

    figs.svg(
        "ch03-windows",
        "Top: the four windows for N = 512. Bottom: their spectra (zero-padded 16 times), \
         relative to the peak, against the offset from the tone in bins. Lower side lobes \
         come with a wider main lobe. The peak side-lobe levels in the legend are those \
         the explainer reports.",
        (720, 680),
        |area| {
            let (top, bottom) = area.split_vertically(260);
            let mut c = Chart::new("Sample index n", "w[n]", (0.0, 511.0), (0.0, 1.08))
                .legend(LegendPos::UpperRight)
                .ticks(9, 5);
            for s in &shapes {
                c = c.add(s.clone());
            }
            c.draw(&top)?;
            let mut c = Chart::new(
                "Offset from the tone (bins)",
                "Level (dB re peak)",
                (0.0, 12.0),
                (floor, 5.0),
            )
            .legend(LegendPos::UpperRight)
            .ticks(13, 7);
            for s in &spectra {
                c = c.add(s.clone());
            }
            c.draw(&bottom)
        },
    )
}

/// Two tones 20 Hz apart: zero padding against a longer signal.
fn zero_padding(figs: &mut Figures) -> Result {
    // The chapter's analysis, returning the spectrum as well as the peak count.
    let analyse = |dur: Duration, n_fft: usize| -> Result<(usize, Vec<(f64, f64)>)> {
        let a = sine_wave::<f64>(1000.0, dur, sample_rate!(16000), 1.0);
        let b = sine_wave::<f64>(1020.0, dur, sample_rate!(16000), 1.0);
        let v: Vec<f64> = a
            .as_slice()
            .ok_or("mono slice")?
            .iter()
            .zip(b.as_slice().ok_or("mono slice")?)
            .map(|(p, q)| p + q)
            .collect();
        let w = make_window::<f64>(WindowType::Hanning, nz(v.len())?);
        let m = AudioSamples::new_mono(ndarray::Array1::from(v), sample_rate!(16000))?
            .apply_window_f64(&w)?
            .rfft(nz(n_fft)?)?
            .row(0)
            .to_vec();
        let bin = 16_000.0 / n_fft as f64;
        let (lo, hi) = ((900.0 / bin) as usize, (1120.0 / bin) as usize);
        let top = m[lo..hi].iter().copied().fold(0.0, f64::max);
        let peaks = (lo..hi)
            .filter(|&i| m[i] > m[i - 1] && m[i] >= m[i + 1] && m[i] > 0.5 * top)
            .count();
        let pts = (lo..=hi).map(|i| (i as f64 * bin, m[i] / top)).collect();
        Ok((peaks, pts))
    };
    let (p1, s1) = analyse(Duration::from_millis(32), 512)?;
    let (p2, s2) = analyse(Duration::from_millis(32), 8192)?;
    let (p3, s3) = analyse(Duration::from_millis(256), 8192)?;
    assert_eq!((p1, p2, p3), (1, 1, 2), "peak counts the chapter asserts");

    figs.svg(
        "ch03-zero-padding",
        "Two tones at 1000 and 1020 Hz, Hann window, each spectrum scaled to its maximum. \
         512 samples give one peak, and zero-padding them to 8192 points only draws the \
         same peak more smoothly; 4096 samples resolve both tones.",
        SIZE,
        |area| {
            Chart::new(
                "Frequency (Hz)",
                "Magnitude (re maximum)",
                (900.0, 1120.0),
                (0.0, 1.12),
            )
            .add(Series::new(
                "512 samples, N = 512",
                s1.clone(),
                GREY,
                Style::Dots(4),
            ))
            .add(Series::new(
                "512 samples zero-padded, N = 8192",
                s2.clone(),
                ORANGE,
                Style::Dashed(2),
            ))
            .add(Series::line("4096 samples, N = 8192", s3.clone(), BLUE))
            .vrule(1000.0, "", GREY)
            .vrule(1020.0, "", GREY)
            .legend(LegendPos::UpperLeft)
            .draw(area)
        },
    )
}
