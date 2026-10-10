//! Figures for chapter 04, STFT and spectrograms.

use std::num::NonZeroUsize;
use std::time::Duration;

use audio_samples::utils::generation::{chirp, impulse};
use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
use guide_figures::*;
use spectrograms::{LogParams, MelParams, SpectrogramParams, StftParams, WindowType};

fn main() -> Result {
    let mut figs = Figures::new("ch04")?;
    chirp_spectrograms(&mut figs)?;
    tradeoff(&mut figs)?;
    figs.finish()
}

/// Rows of an ndarray matrix as nested vectors (rows = frequency, bottom up).
fn rows_of(a: &ndarray::Array2<f64>) -> Vec<Vec<f64>> {
    a.rows().into_iter().map(|r| r.to_vec()).collect()
}

/// A linear-frequency dB spectrogram and a mel spectrogram of a chirp, with
/// the chapter's parameters (N = 512, H = 128, Hann, centred, floor -80 dB).
fn chirp_spectrograms(figs: &mut Figures) -> Result {
    let x = chirp::<f64>(
        100.0,
        7900.0,
        Duration::from_secs(2),
        sample_rate!(16000),
        0.5,
    );
    let stft = StftParams::new(nzu!(512), nzu!(128), WindowType::Hanning, true)?;
    let sp = SpectrogramParams::new(stft, 16_000.0)?;
    let floor = LogParams::new(-80.0)?;
    let lin = x.linear_db_spectrogram(&sp, &floor)?;
    let mel = x.mel_power_spectrogram(&sp, &MelParams::new(nzu!(40), 0.0, 8000.0)?)?;
    assert_eq!(
        (lin.data().nrows(), mel.data().nrows()),
        (257, 40),
        "bins and mel bands as in the chapter"
    );
    // 2 s = 32 000 samples: 1 + (32 000 + 512 - 512) / 128 = 251 frames.
    assert_eq!(lin.data().ncols(), 251, "frame count formula");
    check(
        "the dB spectrogram is clamped at the -80 dB floor",
        lin.data().iter().all(|&v| v >= -80.0 - 1e-9),
    );

    let lin_rows = rows_of(lin.data());
    let peak = lin.data().iter().copied().fold(f64::MIN, f64::max);
    let mel_db: Vec<Vec<f64>> = rows_of(mel.data())
        .into_iter()
        .map(|r| r.iter().map(|&p| 10.0 * p.max(1e-8).log10()).collect())
        .collect();
    let mel_peak = mel_db.iter().flatten().copied().fold(f64::MIN, f64::max);
    let t_end = 250.0 * 128.0 / 16_000.0;

    figs.svg(
        "ch04-chirp",
        "A 2 s linear chirp from 100 Hz to 7.9 kHz at 16 kHz (amplitude 0.5). Top: \
         linear_db_spectrogram with N = 512, H = 128 and a -80 dB floor, in dB re 1. \
         Bottom: mel_power_spectrogram with 40 bands, in dB; the mel scale spends more \
         bands on low frequencies, so the line bends.",
        (720, 640),
        |area| {
            let panels = rows(area, 2);
            Heatmap {
                values: &lin_rows,
                x: (0.0, t_end),
                y: (0.0, 8000.0),
                z: (-80.0, peak.ceil()),
                x_label: "Time (s)".into(),
                y_label: "Frequency (Hz)".into(),
                z_label: "dB re 1".into(),
                title: Some("Linear frequency, 257 bins".into()),
                y_fmt: Some(khz),
                overlays: Vec::new(),
            }
            .draw(&panels[0])?;
            Heatmap {
                values: &mel_db,
                x: (0.0, t_end),
                y: (0.0, 40.0),
                z: (mel_peak.ceil() - 80.0, mel_peak.ceil()),
                x_label: "Time (s)".into(),
                y_label: "Mel band".into(),
                z_label: "dB re 1".into(),
                title: Some("Mel, 40 bands from 0 to 8 kHz".into()),
                y_fmt: None,
                overlays: Vec::new(),
            }
            .draw(&panels[1])
        },
    )
}

/// The chapter's two test signals: two tones 50 Hz apart, and a click.
fn signals() -> Result<(AudioSamples<'static, f64>, AudioSamples<'static, f64>)> {
    let half_s = Duration::from_millis(500);
    let a = sine_wave::<f64>(1000.0, half_s, sample_rate!(16000), 0.5);
    let b = sine_wave::<f64>(1050.0, half_s, sample_rate!(16000), 0.5);
    let tones: Vec<f64> = a
        .as_slice()
        .ok_or("mono slice")?
        .iter()
        .zip(b.as_slice().ok_or("mono slice")?)
        .map(|(p, q)| p + q)
        .collect();
    let tones = AudioSamples::new_mono(ndarray::Array1::from(tones), sample_rate!(16000))?;
    let click = impulse::<f64>(half_s, sample_rate!(16000), 1.0, 0.25);
    Ok((tones, click))
}

fn params(n: usize) -> Result<StftParams> {
    Ok(StftParams::new(
        NonZeroUsize::new(n).ok_or("n")?,
        nzu!(32),
        WindowType::Hanning,
        true,
    )?)
}

/// The chapter's measurement of the trade-off, with the curves it reads.
fn tradeoff(figs: &mut Figures) -> Result {
    let (tones, click) = signals()?;
    // Returns (peak count, spread in ms, summed-magnitude curve, click energy curve).
    type Curves = (usize, f64, Vec<(f64, f64)>, Vec<(f64, f64)>);
    let analyse = |n: usize| -> Result<Curves> {
        let p = params(n)?;
        let s = tones.stft(&p)?;
        let avg: Vec<f64> = (0..s.data.nrows())
            .map(|k| s.data.row(k).iter().map(|c| c.norm()).sum())
            .collect();
        let bin = 16_000.0 / n as f64;
        let (lo, hi) = ((800.0 / bin) as usize, (1250.0 / bin) as usize);
        let top = avg[lo..hi].iter().copied().fold(0.0, f64::max);
        let peaks = (lo..hi)
            .filter(|&i| avg[i] > avg[i - 1] && avg[i] >= avg[i + 1] && avg[i] > 0.5 * top)
            .count();
        let c = click.stft(&p)?;
        let e: Vec<f64> = (0..c.data.ncols())
            .map(|j| c.data.column(j).iter().map(|v| v.norm_sqr()).sum())
            .collect();
        let emax = e.iter().copied().fold(0.0, f64::max);
        let spread_ms = e.iter().filter(|&&v| v > 0.5 * emax).count() as f64 * 2.0;
        let freq = (lo..=hi).map(|i| (i as f64 * bin, avg[i] / top)).collect();
        // Frame j is centred on sample 32 j, i.e. at 2 j ms.
        let time = e
            .iter()
            .enumerate()
            .map(|(j, &v)| (j as f64 * 2.0, v / emax))
            .collect();
        Ok((peaks, spread_ms, freq, time))
    };
    let (peaks_short, spread_short, f_short, t_short) = analyse(256)?;
    let (peaks_long, spread_long, f_long, t_long) = analyse(2048)?;
    assert_eq!((peaks_short, peaks_long), (1, 2), "peak counts");
    assert_eq!(
        (spread_short, spread_long),
        (6.0, 46.0),
        "click spreads in ms"
    );

    figs.svg(
        "ch04-tradeoff",
        "The chapter's trade-off measurement, hop 2 ms. Top: summed STFT magnitude of two \
         tones at 1000 and 1050 Hz; only the 128 ms window (N = 2048) separates them. \
         Bottom: frame energy of a click at 250 ms; it stays above half its maximum for \
         6 ms with the 16 ms window (N = 256) and 46 ms with the 128 ms window.",
        (720, 600),
        |area| {
            let panels = rows(area, 2);
            Chart::new(
                "Frequency (Hz)",
                "Summed magnitude (re max)",
                (800.0, 1250.0),
                (0.0, 1.12),
            )
            .add(Series::line("N = 256 (16 ms)", f_short.clone(), ORANGE))
            .add(Series::line("N = 2048 (128 ms)", f_long.clone(), BLUE))
            .vrule(1000.0, "", GREY)
            .vrule(1050.0, "", GREY)
            .legend(LegendPos::UpperRight)
            .draw(&panels[0])?;
            Chart::new(
                "Frame centre (ms)",
                "Click energy (re max)",
                (180.0, 320.0),
                (0.0, 1.12),
            )
            .add(Series::new(
                "N = 256 (16 ms)",
                t_short.clone(),
                ORANGE,
                Style::Line(2),
            ))
            .add(Series::new(
                "N = 2048 (128 ms)",
                t_long.clone(),
                BLUE,
                Style::Line(2),
            ))
            .hrule(0.5, "half maximum", GREY)
            .legend(LegendPos::UpperRight)
            .draw(&panels[1])
        },
    )
}
