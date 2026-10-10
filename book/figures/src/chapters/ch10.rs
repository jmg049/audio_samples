//! Figures for chapter 10, cepstrum, mel and MFCC.

use std::f64::consts::PI;
use std::num::NonZeroUsize;
use std::time::Duration;

use audio_samples::operations::filterbanks::{
    MelFilterNorm, MelFilterbankConfig, MelScale, mel_filterbank,
};
use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
use guide_figures::*;
use ndarray::Array1;
use spectrograms::{MelParams, MfccParams, SpectrogramParams, StftParams, WindowType};

const FS: f64 = 16_000.0;

fn main() -> Result {
    let mut figs = Figures::new("ch10")?;
    cepstrum(&mut figs)?;
    filterbank(&mut figs)?;
    mfcc(&mut figs)?;
    figs.finish()
}

/// The chapter's vowel: impulse train (period 100 samples) through the three
/// formant resonators of chapter 8.
fn vowel(period: usize, len: usize) -> Vec<f64> {
    let mut a = vec![1.0f64];
    for &(f, b) in &[(700.0, 80.0), (1220.0, 90.0), (2600.0f64, 120.0f64)] {
        let (r, th) = ((-PI * b / FS).exp(), 2.0 * PI * f / FS);
        let s = [1.0, -2.0 * r * th.cos(), r * r];
        let mut o = vec![0.0; a.len() + 2];
        for (i, ai) in a.iter().enumerate() {
            for (j, sj) in s.iter().enumerate() {
                o[i + j] += ai * sj;
            }
        }
        a = o;
    }
    let mut y = vec![0.0f64; len];
    for i in 0..len {
        let e = if i % period == 0 { 1.0 } else { 0.0 };
        y[i] = e
            - (1..a.len())
                .filter(|&k| k <= i)
                .map(|k| a[k] * y[i - k])
                .sum::<f64>();
    }
    y
}

/// Log spectrum and real cepstrum of the Hann-windowed vowel frame.
fn cepstrum(figs: &mut Figures) -> Result {
    let y = vowel(100, 2600);
    let n = 1024;
    let frame: Vec<f64> = (0..n)
        .map(|i| (0.5 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()) * y[1500 + i])
        .collect();
    let audio = AudioSamples::new_mono(Array1::from(frame), sample_rate!(16000))?;
    let x = audio.fft(NonZeroUsize::new(n).ok_or("n")?)?;
    let logmag: Vec<f64> = (0..=n / 2)
        .map(|k| x[[0, k]].norm().max(1e-12).ln())
        .collect();
    let ceps = |q: usize| {
        let mut s = logmag[0] + logmag[n / 2] * (PI * q as f64).cos();
        for (k, &lk) in logmag.iter().enumerate().take(n / 2).skip(1) {
            s += 2.0 * lk * (2.0 * PI * (k * q) as f64 / n as f64).cos();
        }
        s / n as f64
    };
    let c: Vec<f64> = (0..=n / 2).map(ceps).collect();
    let (q_lo, q_hi) = ((FS / 400.0) as usize, (FS / 60.0) as usize);
    assert_eq!((q_lo, q_hi), (40, 266));
    let q_peak = (q_lo..=q_hi)
        .max_by(|&p, &q| c[p].total_cmp(&c[q]))
        .ok_or("range")?;
    assert_eq!(q_peak, 100);
    assert_close("cepstral f0", FS / q_peak as f64, 160.0, 0.0);
    assert_close("c[1]", c[1], 2.32, 0.005);
    assert_close("c[2]", c[2], 0.47, 0.005);
    assert_close("peak height", c[q_peak], 1.55, 0.005);
    let second = (q_lo..=q_hi)
        .filter(|&q| q.abs_diff(q_peak) > 5)
        .map(|q| c[q])
        .fold(f64::NEG_INFINITY, f64::max);
    assert_close(
        "next highest value in the search range",
        second,
        0.38,
        0.005,
    );

    let spec: Vec<(f64, f64)> = (0..=n / 2)
        .map(|k| {
            (
                k as f64 * FS / n as f64,
                20.0 * x[[0, k]].norm().max(1e-12).log10(),
            )
        })
        .collect();
    let top = spec.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
    let y_max = (top / 10.0).ceil() * 10.0 + 10.0;
    let cq: Vec<(f64, f64)> = (1..=300).map(|q| (q as f64, c[q])).collect();

    figs.svg(
        "ch10-cepstrum",
        "Top: spectrum of the Hann-windowed vowel frame (1024 samples, f0 = 160 Hz): \
         harmonics every 160 Hz under the formant envelope. Bottom: its real cepstrum. The \
         envelope is at low quefrencies (c[1] = 2.32); in the search range for 60 to 400 Hz \
         (lags 40 to 266) the largest value, 1.55, is at q = 100 samples, so f0 = 160 Hz.",
        (720, 600),
        |area| {
            let p = rows(area, 2);
            Chart::new(
                "Frequency (Hz)",
                "Level (dB)",
                (0.0, 8000.0),
                (y_max - 100.0, y_max),
            )
            .x_format(khz)
            .add(Series::new("", spec.clone(), BLUE, Style::Line(1)))
            .draw(&p[0])?;
            Chart::new("Quefrency q (samples)", "c[q]", (0.0, 300.0), (-0.6, 2.5))
                .add(Series::line("", cq.clone(), BLUE))
                .add(Series::new(
                    "",
                    vec![(q_peak as f64, c[q_peak])],
                    VERMILION,
                    Style::Dots(6),
                ))
                .vrule(40.0, "q = 40 (400 Hz)", GREY)
                .vrule(266.0, "q = 266 (60 Hz)", GREY)
                .note(
                    q_peak as f64 + 4.0,
                    c[q_peak] + 0.25,
                    "q = 100: f0 = 160 Hz",
                    VERMILION,
                )
                .draw(&p[1])
        },
    )
}

fn tone() -> AudioSamples<'static, f64> {
    sine_wave::<f64>(440.0, Duration::from_millis(250), sample_rate!(16000), 0.5)
}

fn stft() -> Result<StftParams> {
    Ok(StftParams::new(
        nzu!(512),
        nzu!(160),
        WindowType::Hanning,
        true,
    )?)
}

/// The 40 Slaney-scale triangles of the library's mel spectrogram.
fn filterbank(figs: &mut Figures) -> Result {
    let audio = tone();
    let sp = SpectrogramParams::new(stft()?, FS)?;
    let mel_params = MelParams::new(nzu!(40), 0.0, FS / 2.0)?;
    let lin = audio.linear_power_spectrogram(&sp)?;
    let mel = audio.mel_power_spectrogram(&sp, &mel_params)?;
    let (lin, mel) = (lin.data(), mel.data());
    assert_eq!(mel.dim(), (40, 26));

    let cfg = MelFilterbankConfig::new()
        .with_n_mels(nzu!(40))
        .with_fmin(0.0)
        .with_fmax(Some(FS / 2.0))
        .with_scale(MelScale::Slaney)
        .with_norm(MelFilterNorm::None);
    let fb = mel_filterbank(FS, nzu!(512), &cfg)?;
    assert_eq!(fb.dim(), (40, 257));
    // These triangles are the ones inside mel_power_spectrogram: applied to
    // the library's power spectrogram they give its mel spectrogram.
    let mut worst = 0.0f64;
    for t in 0..mel.ncols() {
        for i in 0..40 {
            let p: f64 = (0..lin.nrows()).map(|k| fb[[i, k]] * lin[[k, t]]).sum();
            worst = worst.max((p - mel[[i, t]]).abs() / mel[[i, t]].max(1e-300));
        }
    }
    check(
        format!("filterbank reproduces the mel spectrogram ({worst:e})"),
        worst < 1e-10,
    );

    // The loudest band of frame 13, as in the chapter.
    let loudest = (0..40)
        .max_by(|&a, &b| mel[[a, 13]].total_cmp(&mel[[b, 13]]))
        .ok_or("bands")?;
    assert_eq!(loudest, 5);
    let centre =
        (loudest + 1) as f64 * (200.0 / 3.0) * (15.0 + 27.0 * 8f64.ln() / 6.4f64.ln()) / 41.0;
    assert_close("centre of band 5 (Hz)", centre, 441.4, 0.05);

    let df = FS / 512.0;
    let series: Vec<Series> = (0..40)
        .map(|i| {
            let pts: Vec<(f64, f64)> = (0..fb.ncols())
                .filter(|&k| {
                    // Keep the support of the triangle plus one zero on each side.
                    let near = |j: usize| j < fb.ncols() && fb[[i, j]] > 0.0;
                    near(k) || (k > 0 && near(k - 1)) || near(k + 1)
                })
                .map(|k| (k as f64 * df, fb[[i, k]]))
                .collect();
            let color = if i == loudest {
                VERMILION
            } else if i % 2 == 0 {
                BLUE
            } else {
                SKY
            };
            Series::new(
                "",
                pts,
                color,
                Style::Line(if i == loudest { 3 } else { 1 }),
            )
        })
        .collect();

    figs.svg(
        "ch10-filterbank",
        "The 40 triangular mel filters (Slaney scale, 0 Hz to 8 kHz, peak 1, no \
         normalisation) from mel_filterbank, evaluated at the 257 bins of a 512-point FFT at 16 kHz. \
         They are evenly spaced below 1 kHz and widen logarithmically above. Band 5 \
         (orange, centred at 441.4 Hz) is the loudest for the 440 Hz tone.",
        SIZE,
        |area| {
            let mut c = Chart::new("Frequency (Hz)", "Filter weight", (0.0, 8000.0), (0.0, 1.1))
                .x_format(khz);
            for s in &series {
                c = c.add(s.clone());
            }
            c.vrule(440.0, "440 Hz", VERMILION).draw(area)
        },
    )
}

/// Log mel energies of one frame and its 13 MFCCs, with and without lifter.
fn mfcc(figs: &mut Figures) -> Result {
    let audio = tone();
    let stft = stft()?;
    let m = 40;
    let mel = audio.mel_power_spectrogram(
        &SpectrogramParams::new(stft.clone(), FS)?,
        &MelParams::new(nzu!(40), 0.0, FS / 2.0)?,
    )?;
    let params = MfccParams::speech_standard();
    let lib = audio.mfcc(&stft, nzu!(40), &params)?;
    let lib_plain = audio.mfcc(&stft, nzu!(40), &params.with_lifter(0))?;
    assert_eq!(lib.data.dim(), (13, 26));

    // The chapter's re-derivation: dB floor, unnormalised DCT-II, lifter 22.
    let dct = |l: &[f64], k: usize| -> f64 {
        l.iter()
            .enumerate()
            .map(|(i, li)| li * (PI * k as f64 * (i as f64 + 0.5) / m as f64).cos())
            .sum()
    };
    let mut worst = 0.0f64;
    for t in 0..mel.data().ncols() {
        let l: Vec<f64> = (0..m)
            .map(|i| 10.0 * mel.data()[[i, t]].max(1e-8).log10())
            .collect();
        for k in 0..13 {
            let lifter = 1.0 + 11.0 * (PI * k as f64 / 22.0).sin();
            worst = worst.max((dct(&l, k) * lifter - lib.data[[k, t]]).abs());
            check(
                "unliftered MFCC",
                (dct(&l, k) - lib_plain.data[[k, t]]).abs() < 1e-8,
            );
        }
    }
    check(format!("MFCC re-derivation ({worst:e})"), worst < 1e-8);
    let largest = lib.data.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    assert_close("largest liftered MFCC", largest, 2298.06, 0.01);

    let frame = 13;
    let l: Vec<(f64, f64)> = (0..m)
        .map(|i| (i as f64, 10.0 * mel.data()[[i, frame]].max(1e-8).log10()))
        .collect();
    let lift: Vec<(f64, f64)> = (0..13).map(|k| (k as f64, lib.data[[k, frame]])).collect();
    let plain: Vec<(f64, f64)> = (0..13)
        .map(|k| (k as f64 + 0.25, lib_plain.data[[k, frame]]))
        .collect();
    let span = |v: &[(f64, f64)]| {
        v.iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), p| {
                (a.min(p.1), b.max(p.1))
            })
    };
    let (l_lo, l_hi) = span(&l);
    let (c_lo, c_hi) = span(&[lift.as_slice(), plain.as_slice()].concat());
    let pad = 0.1 * (c_hi - c_lo);

    figs.svg(
        "ch10-mfcc",
        "One frame (frame 13, 130 ms) of the 440 Hz tone. Top: log mel energies in dB, \
         floored at -80 dB, with the peak in band 5. Bottom: the 13 MFCCs from \
         AudioTransforms::mfcc (unnormalised DCT-II of the top panel), with the default \
         lifter 22 and with the lifter off.",
        (720, 600),
        |area| {
            let p = rows(area, 2);
            Chart::new(
                "Mel band",
                "Log mel energy (dB)",
                (-0.5, 39.5),
                (l_lo - 5.0, l_hi + 10.0),
            )
            .add(Series::new("", l.clone(), BLUE, Style::Line(1)))
            .add(Series::new("", l.clone(), BLUE, Style::Dots(3)))
            .draw(&p[0])?;
            Chart::new(
                "Coefficient k",
                "MFCC value",
                (-0.5, 12.75),
                (c_lo - pad, c_hi + pad),
            )
            .ticks(13, 8)
            .add(Series::new(
                "lifter 22 (default)",
                lift.clone(),
                VERMILION,
                Style::Stems,
            ))
            .add(Series::new("lifter off", plain.clone(), BLUE, Style::Stems))
            .legend(LegendPos::UpperRight)
            .draw(&p[1])
        },
    )
}
