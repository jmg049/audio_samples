//! Figures for chapter 14, psychoacoustics and codecs.

use std::f64::consts::PI;
use std::num::NonZeroUsize;
use std::time::Duration;

use audio_samples::codecs::perceptual::bands::{hz_to_bark, hz_to_erb};
use audio_samples::codecs::{decode, encode};
use audio_samples::{
    AudioProcessing, AudioSamples, BandLayout, MaskerType, PerceptualCodec, PsychoacousticConfig,
    absolute_threshold_of_hearing, sample_rate, sine_wave, spreading_attenuation,
};
use guide_figures::*;
use non_empty_slice::NonEmptyVec;
use spectrograms::WindowType;

fn main() -> Result {
    let mut figs = Figures::new("ch14")?;
    threshold(&mut figs)?;
    scales(&mut figs)?;
    masking(&mut figs)?;
    mu_law(&mut figs)?;
    codec(&mut figs)?;
    figs.finish()
}

/// Log-spaced frequencies from `lo` to `hi` Hz.
fn log_freqs(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| lo * (hi / lo).powf(i as f64 / (n - 1) as f64))
        .collect()
}

fn hz_ticks(v: f64) -> String {
    khz(v)
}

/// Terhardt's threshold in quiet, from the library.
fn threshold(figs: &mut Figures) -> Result {
    let ath = |f: f64| f64::from(absolute_threshold_of_hearing(f as f32));
    assert_close("threshold at 100 Hz", ath(100.0), 22.95, 0.01);
    assert_close("threshold at 1 kHz", ath(1_000.0), 3.37, 0.01);
    assert_close("threshold at 3.3 kHz", ath(3_300.0), -4.98, 0.01);
    assert_close("threshold at 50 Hz", ath(50.0), 40.0, 0.1);
    // The chapter's check of the library against the formula, every 10 Hz.
    let terhardt = |f: f64| {
        let k = f / 1000.0;
        3.64 * k.powf(-0.8) - 6.5 * (-0.6 * (k - 3.3).powi(2)).exp() + 1e-3 * k.powi(4)
    };
    for f in (20..=16_000).step_by(10) {
        let f = f64::from(f);
        assert_close(
            "library vs Terhardt",
            ath(f),
            terhardt(f).clamp(-20.0, 60.0),
            1e-3,
        );
    }
    let (f_min, t_min) = (20..=16_000)
        .map(|f| (f, ath(f64::from(f))))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .ok_or("empty")?;
    assert_close("frequency of the minimum", f64::from(f_min), 3_320.0, 20.0);
    assert_close("minimum threshold", t_min, -4.98, 0.01);

    let curve: Vec<(f64, f64)> = log_freqs(20.0, 16_000.0, 600)
        .into_iter()
        .map(|f| (f, ath(f)))
        .collect();
    let marks = vec![
        (100.0, ath(100.0)),
        (1_000.0, ath(1_000.0)),
        (f64::from(f_min), t_min),
    ];
    figs.svg(
        "ch14-threshold",
        "Threshold of hearing in quiet (Terhardt), from absolute_threshold_of_hearing: \
         22.95 dB SPL at 100 Hz, 3.37 dB SPL at 1 kHz and a minimum of -4.98 dB SPL near \
         3.3 kHz. Tones below the curve are inaudible. The library clamps the formula to \
         60 dB SPL, hence the flat segment below 30 Hz.",
        SIZE,
        |area| {
            Chart::new(
                "Frequency (Hz)",
                "Threshold (dB SPL)",
                (20.0, 16_000.0),
                (-10.0, 65.0),
            )
            .log_x()
            .x_format(hz_ticks)
            .add(Series::line("", curve.clone(), BLUE))
            .add(Series::new("", marks.clone(), VERMILION, Style::Dots(5)))
            .note(110.0, ath(100.0) + 6.0, "22.95 dB at 100 Hz", VERMILION)
            .note(1_100.0, ath(1_000.0) + 6.0, "3.37 dB at 1 kHz", VERMILION)
            .note(2_000.0, t_min - 1.0, "-4.98 dB near 3.3 kHz", VERMILION)
            .note(400.0, 40.0, "inaudible below the curve", GREY)
            .draw(area)
        },
    )
}

/// The Bark scale and the auditory bandwidths.
fn scales(figs: &mut Figures) -> Result {
    let bark = |f: f64| f64::from(hz_to_bark(f as f32));
    let erb = |f: f64| f64::from(hz_to_erb(f as f32));
    assert_close("1 kHz in Bark", bark(1_000.0), 8.527, 1e-3);
    assert_close("4 kHz in Bark", bark(4_000.0), 17.46, 0.01);
    assert_close(
        "Bark distance 1 to 4 kHz",
        bark(4_000.0) - bark(1_000.0),
        8.93,
        0.01,
    );
    assert_close("ERB at 1 kHz", erb(1_000.0), 132.639, 1e-3);
    assert_close("ERB at 4 kHz", erb(4_000.0), 456.456, 1e-2);

    let fs = log_freqs(20.0, 16_000.0, 400);
    let z: Vec<(f64, f64)> = fs.iter().map(|&f| (f, bark(f))).collect();
    let erb_pts: Vec<(f64, f64)> = fs.iter().map(|&f| (f, erb(f))).collect();
    // One critical band is one Bark: its width in Hz is 1 / (dz/df), with the
    // derivative taken numerically from the library's hz_to_bark.
    // The library clamps the Bark value at 0 below about 50 Hz, so the
    // bandwidth starts at 60 Hz; the step is 1 % for f32 precision.
    let cb: Vec<(f64, f64)> = fs
        .iter()
        .filter(|&&f| f >= 60.0)
        .map(|&f| {
            let h = f * 1e-2;
            (f, 2.0 * h / (bark(f + h) - bark(f - h)))
        })
        .collect();
    let marks = vec![(1_000.0, bark(1_000.0)), (4_000.0, bark(4_000.0))];

    figs.svg(
        "ch14-scales",
        "Top: the Bark scale (hz_to_bark, Traunmüller); 1 kHz is 8.53 Bark and 4 kHz is \
         17.46 Bark. Bottom: the critical bandwidth (the width of one Bark, from hz_to_bark) \
         and the Glasberg-Moore equivalent rectangular bandwidth (hz_to_erb).",
        (720, 600),
        |area| {
            let panels = rows(area, 2);
            Chart::new(
                "Frequency (Hz)",
                "Critical-band rate (Bark)",
                (20.0, 16_000.0),
                (-1.0, 25.0),
            )
            .log_x()
            .x_format(hz_ticks)
            .add(Series::line("", z.clone(), BLUE))
            .add(Series::new("", marks.clone(), VERMILION, Style::Dots(5)))
            .note(1_100.0, 7.6, "1 kHz: 8.53 Bark", VERMILION)
            .note(4_400.0, 16.6, "4 kHz: 17.46 Bark", VERMILION)
            .draw(&panels[0])?;
            Chart::new(
                "Frequency (Hz)",
                "Bandwidth (Hz)",
                (20.0, 16_000.0),
                (20.0, 10_000.0),
            )
            .log_x()
            .log_y()
            .x_format(hz_ticks)
            .add(Series::line(
                "critical bandwidth (1 Bark)",
                cb.clone(),
                BLUE,
            ))
            .add(Series::line("ERB (hz_to_erb)", erb_pts.clone(), VERMILION))
            .legend(LegendPos::UpperLeft)
            .draw(&panels[1])
        },
    )
}

/// Masked threshold around a masker: the library's mpeg1 preset against the
/// Schroeder et al. spreading function.
fn masking(figs: &mut Figures) -> Result {
    let w = PsychoacousticConfig::uniform_weights(NonZeroUsize::new(24).ok_or("24")?);
    let cfg = PsychoacousticConfig::mpeg1(w.as_non_empty_slice());
    let att = |dz: f64, t: MaskerType| {
        f64::from(spreading_attenuation(10.0, (10.0 + dz) as f32, t, &cfg))
    };
    assert_close(
        "tonal, two Bark above",
        att(2.0, MaskerType::Tonal),
        64.5,
        1e-4,
    );
    assert_close(
        "tonal, two Bark below",
        att(-2.0, MaskerType::Tonal),
        26.5,
        1e-4,
    );
    assert_close(
        "noise, at the masker",
        att(0.0, MaskerType::Noise),
        5.5,
        1e-6,
    );

    // Schroeder et al. (1979), derived by hand in the chapter.
    let s = |dz: f64| 15.81 + 7.5 * (dz + 0.474) - 17.5 * (1.0 + (dz + 0.474).powi(2)).sqrt();
    assert_close(
        "Schroeder upper slope",
        (s(10.0) - s(8.0)) / 2.0,
        -10.0,
        0.3,
    );
    assert_close(
        "Schroeder lower slope",
        (s(-8.0) - s(-10.0)) / 2.0,
        25.0,
        0.3,
    );

    let dzs: Vec<f64> = (0..=240)
        .map(|i| -4.0 + 8.0 * f64::from(i) / 240.0)
        .collect();
    let tonal: Vec<(f64, f64)> = dzs
        .iter()
        .map(|&d| (d, -att(d, MaskerType::Tonal)))
        .collect();
    let noise: Vec<(f64, f64)> = dzs
        .iter()
        .map(|&d| (d, -att(d, MaskerType::Noise)))
        .collect();
    let schroeder: Vec<(f64, f64)> = dzs.iter().map(|&d| (d, s(d) - 14.5)).collect();

    figs.svg(
        "ch14-masking",
        "Masked threshold relative to the masker level against the distance from the \
         masker in Bark. The library's mpeg1 preset (spreading_attenuation) falls by \
         25 dB/Bark above the masker and 6 dB/Bark below it; the Schroeder et al. spreading \
         function (shifted by the 14.5 dB tonal offset) falls steeply below and gently above.",
        SIZE,
        |area| {
            Chart::new(
                "Distance from the masker (Bark, maskee minus masker)",
                "Threshold re masker (dB)",
                (-4.0, 4.0),
                (-120.0, 5.0),
            )
            .add(Series::line("library, tonal masker", tonal.clone(), BLUE))
            .add(Series::line("library, noise masker", noise.clone(), GREEN))
            .add(Series::new(
                "Schroeder et al., tonal offset",
                schroeder.clone(),
                VERMILION,
                Style::Dashed(2),
            ))
            .add(Series::new(
                "",
                vec![(2.0, -64.5), (-2.0, -26.5)],
                INK,
                Style::Dots(4),
            ))
            .note(1.15, -62.0, "-64.5 dB", INK)
            .note(-1.9, -29.5, "-26.5 dB", INK)
            .legend(LegendPos::LowerMiddle)
            .draw(area)
        },
    )
}

// G.711 mu-law, written out from the standard as in the chapter.
fn ulaw_encode(x: i32) -> u8 {
    let (mag, sign) = if x < 0 { (-x, 0x80u8) } else { (x, 0x00u8) };
    let biased = (mag.min(8_159) + 33).min(8_191);
    let seg = (0..8).find(|&s| biased < (64 << s)).unwrap_or(7);
    let mant = ((biased >> (seg + 1)) & 0x0F) as u8;
    !(sign | ((seg as u8) << 4) | mant)
}

fn ulaw_decode(code: u8) -> i32 {
    let c = !code;
    let (seg, mant) = (i32::from((c >> 4) & 7), i32::from(c & 0x0F));
    let mag = (((mant << 1) + 33) << seg) - 33;
    if c & 0x80 != 0 { -mag } else { mag }
}

/// The mu-law curve (library) with the G.711 codes, and SNR against level.
fn mu_law(figs: &mut Figures) -> Result {
    // The library's continuous curve on [0, 1].
    let xs: Vec<f64> = (0..=400).map(|i| f64::from(i) / 400.0).collect();
    let audio: AudioSamples<f64> = AudioSamples::from_mono_vec(
        NonEmptyVec::new(xs.clone()).map_err(|_| "empty")?,
        sample_rate!(8000),
    );
    let compressed = audio.mu_compress(255.0)?;
    let ys = compressed.as_slice().ok_or("mono slice")?;
    let mu: f64 = 255.0;
    let f = |x: f64| x.signum() * (1.0 + mu * x.abs()).ln() / (1.0 + mu).ln();
    for (x, y) in xs.iter().zip(ys) {
        assert_close("mu_compress vs F(x)", *y, f(*x), 1e-12);
    }
    check("F(0) = 0", ys[0] == 0.0);
    assert_close("F(1) = 1", ys[400], 1.0, 1e-12);
    let curve: Vec<(f64, f64)> = xs.iter().copied().zip(ys.iter().copied()).collect();

    // G.711 code index (0..127) / 127, against the input magnitude.
    check("G.711 positive zero", ulaw_encode(0) == 0xFF);
    check(
        "G.711 largest magnitude",
        ulaw_decode(ulaw_encode(8_159)) == 8_031,
    );
    let mut worst = 0.0f64;
    let mut codes = Vec::new();
    for x in (0..=8_159).step_by(8) {
        let idx = f64::from(!ulaw_encode(x) & 0x7F);
        if x > 0 {
            worst = worst.max((idx - 127.0 * f(f64::from(x) / 8_159.0)).abs());
        }
        codes.push((f64::from(x) / 8_159.0, idx / 127.0));
    }
    check("G.711 within 2 codes of 127 F(x)", worst < 2.0);

    // SNR of a sine against its peak level, as in the chapter.
    let snr = |level_db: f64, q: &dyn Fn(f64) -> f64| {
        let a = 10f64.powf(level_db / 20.0);
        let (mut s, mut e) = (0.0, 0.0);
        for n in 0..8_000 {
            let x = a * (2.0 * PI * 0.1234567 * f64::from(n)).sin();
            s += x * x;
            e += (x - q(x)).powi(2);
        }
        10.0 * (s / e).log10()
    };
    let ulaw = |x: f64| f64::from(ulaw_decode(ulaw_encode((x * 8_159.0).round() as i32))) / 8_159.0;
    let lin8 = |x: f64| (x * 127.0).round() / 127.0;
    let levels = [0.0, -10.0, -20.0, -30.0, -40.0];
    let su: Vec<f64> = levels.iter().map(|&l| snr(l, &ulaw)).collect();
    let sl: Vec<f64> = levels.iter().map(|&l| snr(l, &lin8)).collect();
    check(
        "mu-law SNR within 34..40.5 dB",
        su.iter().all(|s| (34.0..40.5).contains(s)),
    );
    assert_close("8-bit linear at full scale", sl[0], 49.9, 0.5);
    assert_close("8-bit linear loses 40 dB", sl[0] - sl[4], 40.0, 1.5);
    check("mu-law beats 8-bit linear at -40 dB", su[4] > sl[4] + 20.0);
    let sweep: Vec<f64> = (0..=50).map(|i| -f64::from(i)).collect();
    let su_curve: Vec<(f64, f64)> = sweep.iter().map(|&l| (l, snr(l, &ulaw))).collect();
    let sl_curve: Vec<(f64, f64)> = sweep.iter().map(|&l| (l, snr(l, &lin8))).collect();

    figs.svg(
        "ch14-mulaw",
        "Top: the mu-law curve (mu_compress, mu = 255) and the G.711 segmented code index \
         scaled to 1 (steps), which stays within 2 codes of the curve. Bottom: SNR of a sine \
         against its peak level; G.711 stays between 34 and 40.5 dB over 40 dB of level, \
         while 8-bit linear PCM falls from 49.9 dB by about 1 dB per dB.",
        (720, 640),
        |area| {
            let panels = rows(area, 2);
            Chart::new(
                "Input magnitude |x| (full scale)",
                "Compressed value",
                (0.0, 1.0),
                (0.0, 1.05),
            )
            .add(Series::line("mu_compress, mu = 255", curve.clone(), BLUE))
            .add(Series::new(
                "G.711 code index / 127",
                codes.clone(),
                VERMILION,
                Style::Steps(1),
            ))
            .legend(LegendPos::LowerRight)
            .draw(&panels[0])?;
            Chart::new(
                "Sine peak level (dBFS peak)",
                "SNR (dB)",
                (-50.0, 0.0),
                (0.0, 55.0),
            )
            .add(Series::line("G.711 mu-law, 8 bits", su_curve.clone(), BLUE))
            .add(Series::line(
                "linear PCM, 8 bits",
                sl_curve.clone(),
                VERMILION,
            ))
            .legend(LegendPos::LowerRight)
            .draw(&panels[1])
        },
    )
}

/// Decoded SNR of the perceptual codec against the bit budget.
fn codec(figs: &mut Figures) -> Result {
    let x = sine_wave::<f32>(
        1_000.0,
        Duration::from_millis(500),
        sample_rate!(44100),
        0.5,
    );
    let snr_db = |budget: u32| -> Result<f64> {
        let n_bands = NonZeroUsize::new(24).ok_or("24")?;
        let layout = BandLayout::bark(n_bands, 44_100.0, NonZeroUsize::new(1024).ok_or("1024")?);
        let w = PsychoacousticConfig::uniform_weights(n_bands);
        let codec = PerceptualCodec::with_window_size(
            layout,
            PsychoacousticConfig::mpeg1(w.as_non_empty_slice()),
            WindowType::Hanning,
            budget,
            1,
            NonZeroUsize::new(2048).ok_or("2048")?,
        );
        let y = decode::<PerceptualCodec, f32>(encode(&x, codec)?)?;
        let (a, b) = (x.as_slice().ok_or("slice")?, y.as_slice().ok_or("slice")?);
        let s: f64 = a.iter().map(|&v| f64::from(v).powi(2)).sum();
        let e: f64 = a
            .iter()
            .zip(b)
            .map(|(&p, &q)| f64::from(p - q).powi(2))
            .sum();
        Ok(10.0 * (s / e).log10())
    };
    let budgets = [
        8_000u32, 16_000, 24_000, 32_000, 48_000, 64_000, 96_000, 128_000, 192_000, 256_000,
    ];
    let mut pts = Vec::new();
    for &b in &budgets {
        pts.push((f64::from(b) / 1000.0, snr_db(b)?));
    }
    let low = snr_db(32_000)?;
    let high = snr_db(128_000)?;
    assert_close("SNR at budget 32000", low, 42.1, 1.0);
    assert_close("SNR at budget 128000", high, 50.6, 1.0);
    check("more bits, higher SNR", high > low);
    let y_max = pts.iter().fold(0.0f64, |m, p| m.max(p.1)) + 8.0;
    let y_min = (pts.iter().fold(f64::INFINITY, |m, p| m.min(p.1)) - 8.0).max(0.0);

    figs.svg(
        "ch14-codec-snr",
        "SNR of half a second of a 1 kHz tone after encoding and decoding with the \
         PerceptualCodec (24 Bark bands, 2048-sample MDCT, mpeg1 masking preset), against \
         the bit budget: about 42 dB at 32000 and 51 dB at 128000.",
        SIZE,
        |area| {
            Chart::new(
                "Bit budget (thousands)",
                "SNR of the decoded signal (dB)",
                (6.0, 300.0),
                (y_min, y_max),
            )
            .log_x()
            .ticks(4, 8)
            .add(Series::line("", pts.clone(), BLUE))
            .add(Series::new("", pts.clone(), BLUE, Style::Dots(4)))
            .add(Series::new(
                "",
                vec![(32.0, low), (128.0, high)],
                VERMILION,
                Style::Dots(6),
            ))
            .note(34.0, low - 2.5, &format!("{low:.1} dB at 32000"), VERMILION)
            .note(
                134.0,
                high - 2.5,
                &format!("{high:.1} dB at 128000"),
                VERMILION,
            )
            .draw(area)
        },
    )
}
