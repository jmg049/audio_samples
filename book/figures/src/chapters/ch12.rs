//! Figures for chapter 12, dynamics: compressors and limiters.

use audio_samples::operations::dynamic_range::{
    calculate_compression_gain, calculate_limiting_gain,
};
use audio_samples::operations::types::{DynamicRangeMethod, KneeType};
use audio_samples::{
    AudioDynamicRange, AudioSamples, CompressorConfig, LimiterConfig, sample_rate,
};
use guide_figures::*;
use non_empty_slice::{NonEmptySlice, NonEmptyVec};

const FS: f64 = 48_000.0;

fn main() -> Result {
    let mut figs = Figures::new("ch12")?;
    static_curve(&mut figs)?;
    attack(&mut figs)?;
    compressor_step(&mut figs)?;
    lookahead(&mut figs)?;
    figs.finish()
}

fn mono(x: Vec<f64>) -> Result<AudioSamples<'static, f64>> {
    Ok(AudioSamples::from_mono_vec(
        NonEmptyVec::new(x).map_err(|_| "empty signal")?,
        sample_rate!(48000),
    ))
}

/// Static curves: hard and soft knee compressor, limiter.
fn static_curve(figs: &mut Figures) -> Result {
    let (t, ratio, w) = (-20.0, 4.0, 10.0);
    // The chapter's checks of the gain computer.
    let hard = |l: f64| calculate_compression_gain(l, t, ratio, KneeType::Hard, 0.0);
    let soft = |l: f64| calculate_compression_gain(l, t, ratio, KneeType::Soft, w);
    assert_close("hard knee reduction at -8 dBFS", hard(-8.0), 9.0, 1e-12);
    assert_close("soft knee reduction at T - W/2", soft(-25.0), 0.0, 0.0);
    assert_close("soft knee reduction at T", soft(-20.0), 0.9375, 1e-12);
    assert_close("soft knee reduction at T + W/2", soft(-15.0), 3.75, 1e-12);

    // The same hard-knee curve through the trait, as output levels.
    let mut cfg = CompressorConfig::new();
    cfg.threshold_db = t;
    cfg.ratio = ratio;
    cfg.knee_type = KneeType::Hard;
    cfg.knee_width_db = 0.0;
    cfg.makeup_gain_db = 0.0;
    let audio = mono(vec![0.5; 16])?;
    let levels: Vec<f64> = (0..=600).map(|i| -60.0 + 0.1 * f64::from(i)).collect();
    let curve = audio.get_compression_curve(&cfg, NonEmptySlice::new(&levels).ok_or("levels")?)?;
    let check3 = audio.get_compression_curve(
        &cfg,
        NonEmptySlice::new(&[-30.0, -20.0, -8.0]).ok_or("levels")?,
    )?;
    check(
        format!("get_compression_curve {check3:?} is [-30, -20, -17]"),
        check3 == vec![-30.0, -20.0, -17.0],
    );

    let ceiling = -6.0;
    let limiter = |l: f64| l - calculate_limiting_gain(l, ceiling, KneeType::Hard, 0.0);
    assert_close("limiter output at 0 dBFS", limiter(0.0), ceiling, 1e-9);

    let hard_pts: Vec<(f64, f64)> = levels.iter().copied().zip(curve).collect();
    let soft_pts: Vec<(f64, f64)> = levels.iter().map(|&l| (l, l - soft(l))).collect();
    let lim_pts: Vec<(f64, f64)> = levels.iter().map(|&l| (l, limiter(l))).collect();
    let unity = vec![(-60.0, -60.0), (0.0, 0.0)];

    figs.svg(
        "ch12-static-curve",
        "Static curves from calculate_compression_gain and get_compression_curve: threshold \
         -20 dBFS and ratio 4:1 with a hard knee and with a 10 dB soft knee (the dashed lines at \
         -25 and -15 dBFS mark its edges), and a limiter with a -6 dBFS ceiling. Above the \
         threshold the output rises 1 dB for every 4 dB of input.",
        SIZE,
        |area| {
            Chart::new(
                "Input level (dBFS)",
                "Output level (dBFS)",
                (-60.0, 0.0),
                (-60.0, 0.0),
            )
            .ticks(12, 12)
            .add(Series::new(
                "1:1 (no processing)",
                unity.clone(),
                GREY,
                Style::Dashed(1),
            ))
            .add(Series::line(
                "compressor, hard knee",
                hard_pts.clone(),
                BLUE,
            ))
            .add(Series::new(
                "compressor, 10 dB soft knee",
                soft_pts.clone(),
                VERMILION,
                Style::Dashed(2),
            ))
            .add(Series::line(
                "limiter, ceiling -6 dBFS",
                lim_pts.clone(),
                GREEN,
            ))
            .vrule(-25.0, "", GREY)
            .vrule(-15.0, "", GREY)
            .vrule(t, "threshold", GREY)
            .legend(LegendPos::LowerRight)
            .draw(area)
        },
    )
}

/// Attack of the hard-knee compressor on a step from -40 to 0 dBFS.
fn attack(figs: &mut Figures) -> Result {
    let x: Vec<f64> = std::iter::repeat_n(0.01, 4_800)
        .chain(std::iter::repeat_n(1.0, 9_600))
        .collect();
    let audio = mono(x)?;
    let mut cfg = CompressorConfig::new();
    (
        cfg.threshold_db,
        cfg.ratio,
        cfg.knee_type,
        cfg.knee_width_db,
    ) = (-20.0, 4.0, KneeType::Hard, 0.0);
    (cfg.attack_ms, cfg.release_ms, cfg.detection_method) = (5.0, 50.0, DynamicRangeMethod::Peak);
    let gr = audio.get_gain_reduction(&cfg)?;
    // Inverting the hard-knee curve recovers the envelope while gr > 0.
    let env = |n: usize| 10f64.powf((-20.0 + gr[n] / 0.75) / 20.0);
    let reach = |level: f64| (4_800..).find(|&n| gr[n] > 0.0 && env(n) >= level);
    let (lo, hi) = (0.01, 1.0);
    let n63 = reach(lo + (1.0 - (-1f64).exp()) * (hi - lo)).ok_or("63 %")?;
    let n10 = reach(lo + 0.1 * (hi - lo)).ok_or("10 %")?;
    let n90 = reach(lo + 0.9 * (hi - lo)).ok_or("90 %")?;
    let t63_ms = (n63 - 4_800 + 1) as f64 / FS * 1e3;
    let rise_ms = (n90 - n10) as f64 / FS * 1e3;
    assert_close("time to 63.2 % (ms)", t63_ms, 5.0, 0.03);
    assert_close("10-90 % rise time (ms)", rise_ms, 5.0 * 9f64.ln(), 0.03);
    assert_close("settled gain reduction (dB)", gr[14_000], 15.0, 1e-6);

    let ms = |n: usize| n as f64 / FS * 1e3;
    let (n0, n1) = (4_560, 6_240); // 95 to 130 ms
    let env_pts: Vec<(f64, f64)> = (n0..n1)
        .map(|n| (ms(n), if gr[n] > 0.0 { env(n) } else { f64::NAN }))
        .filter(|p| p.1.is_finite())
        .collect();
    let gr_pts: Vec<(f64, f64)> = (n0..n1).map(|n| (ms(n), gr[n])).collect();
    let input: Vec<(f64, f64)> = vec![(ms(n0), lo), (100.0, lo), (100.0, hi), (ms(n1), hi)];
    let t63 = 100.0 + t63_ms;
    let (t10, t90) = (ms(n10), ms(n90));

    figs.svg(
        "ch12-attack",
        "Attack of a peak-detecting compressor (5 ms attack, threshold -20 dBFS, 4:1) on a \
         step from -40 to 0 dBFS at 100 ms. Top: the envelope, recovered from \
         get_gain_reduction by inverting the static curve while the envelope is above the \
         threshold; it reaches 63.2 % of the step after 5 ms and rises from 10 % to 90 % in \
         11 ms. Bottom: the gain reduction, settling at 15 dB.",
        (720, 560),
        |area| {
            let p = rows(area, 2);
            Chart::new("Time (ms)", "Amplitude", (95.0, 130.0), (0.0, 1.1))
                .ticks(8, 6)
                .add(Series::new("input", input.clone(), GREY, Style::Dashed(2)))
                .add(Series::line("envelope", env_pts.clone(), BLUE))
                .add(Series::new(
                    "10 %, 63.2 %, 90 % points",
                    vec![(t10, env(n10)), (t63, env(n63)), (t90, env(n90))],
                    VERMILION,
                    Style::Dots(5),
                ))
                .vrule(t63, "63.2 % at 105 ms", VERMILION)
                .legend(LegendPos::LowerRight)
                .draw(&p[0])?;
            Chart::new(
                "Time (ms)",
                "Gain reduction (dB)",
                (95.0, 130.0),
                (0.0, 17.0),
            )
            .ticks(8, 6)
            .add(Series::line("", gr_pts.clone(), BLUE))
            .hrule(15.0, "15 dB: 0 dBFS is 20 dB over, at 4:1", GREY)
            .draw(&p[1])
        },
    )
}

/// The soft-knee compressor on a step up and back down.
fn compressor_step(figs: &mut Figures) -> Result {
    let x: Vec<f64> = [(0.01, 4_800), (1.0, 9_600), (0.01, 14_400)]
        .iter()
        .flat_map(|&(v, n)| std::iter::repeat_n(v, n))
        .collect();
    let audio = mono(x.clone())?;
    let (t, ratio, w) = (-20.0, 4.0, 6.0);
    let mut cfg = CompressorConfig::new();
    (
        cfg.threshold_db,
        cfg.ratio,
        cfg.knee_type,
        cfg.knee_width_db,
    ) = (t, ratio, KneeType::Soft, w);
    (cfg.attack_ms, cfg.release_ms, cfg.detection_method) = (5.0, 50.0, DynamicRangeMethod::Peak);
    (cfg.makeup_gain_db, cfg.lookahead_ms) = (0.0, 0.0);
    let lib = audio.apply_compressor(&cfg)?;
    let lib = lib.as_slice().ok_or("mono")?.to_vec();
    let gr = audio.get_gain_reduction(&cfg)?;

    // The chapter's sample-by-sample reimplementation; it must agree to 1e-12.
    let (a_att, a_rel) = ((-1.0 / (0.005 * FS)).exp(), (-1.0 / (0.050 * FS)).exp());
    let reduction = |l: f64| {
        let o = l - t;
        let slope = 1.0 - 1.0 / ratio;
        if 2.0 * o.abs() <= w {
            slope * (o + w / 2.0).powi(2) / (2.0 * w)
        } else {
            slope * o.max(0.0)
        }
    };
    let mut e = 0.0f64;
    let mut envelope = Vec::with_capacity(x.len());
    let mine: Vec<f64> = x
        .iter()
        .map(|&v| {
            let d = v.abs();
            let a = if d > e { a_att } else { a_rel };
            e = a * e + (1.0 - a) * d;
            envelope.push(e);
            let level = if e > 0.0 { 20.0 * e.log10() } else { -80.0 };
            v * 10f64.powf(-reduction(level) / 20.0)
        })
        .collect();
    let err = mine
        .iter()
        .zip(&lib)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max);
    check(
        format!("apply_compressor matches the recursion ({err})"),
        err < 1e-12,
    );
    assert_close("settled gain reduction (dB)", gr[14_000], 15.0, 1e-6);

    // Plot every 8th sample (3600 points per series).
    let ms = |n: usize| n as f64 / FS * 1e3;
    let lvl = |v: f64| db(v.abs(), -70.0);
    let pick = |f: &dyn Fn(usize) -> f64| -> Vec<(f64, f64)> {
        (0..x.len()).step_by(8).map(|n| (ms(n), f(n))).collect()
    };
    let input = pick(&|n| lvl(x[n]));
    let output = pick(&|n| lvl(lib[n]));
    let env_db = pick(&|n| lvl(envelope[n]));
    let gr_pts = pick(&|n| gr[n]);

    figs.svg(
        "ch12-compressor-step",
        "A soft-knee compressor (threshold -20 dBFS, 4:1, 6 dB knee, 5 ms attack, 50 ms \
         release) on a step from -40 to 0 dBFS and back. Top: input, envelope and output \
         level of apply_compressor. Bottom: gain reduction from get_gain_reduction. The \
         reduction builds up within a few ms and decays over the 50 ms release, so the output \
         dips below the input level after the step down.",
        (720, 560),
        |area| {
            let p = rows(area, 2);
            Chart::new("Time (ms)", "Level (dBFS)", (0.0, 600.0), (-60.0, 5.0))
                .ticks(12, 6)
                .add(Series::new("input", input.clone(), GREY, Style::Line(2)))
                .add(Series::new(
                    "envelope",
                    env_db.clone(),
                    ORANGE,
                    Style::Dashed(2),
                ))
                .add(Series::line("output", output.clone(), BLUE))
                .hrule(t, "threshold -20 dBFS", GREY)
                .legend(LegendPos::UpperRight)
                .draw(&p[0])?;
            Chart::new(
                "Time (ms)",
                "Gain reduction (dB)",
                (0.0, 600.0),
                (0.0, 17.0),
            )
            .ticks(12, 6)
            .add(Series::line("", gr_pts.clone(), VERMILION))
            .draw(&p[1])
        },
    )
}

/// The limiter with and without lookahead on a step from silence to 0 dBFS.
fn lookahead(figs: &mut Figures) -> Result {
    let x: Vec<f64> = std::iter::repeat_n(0.0, 960)
        .chain(std::iter::repeat_n(1.0, 2_400))
        .collect();
    let audio = mono(x.clone())?;
    let run = |lookahead_ms: f64| -> Result<Vec<f64>> {
        let lim = LimiterConfig::new(
            -6.0,
            1.0,
            100.0,
            KneeType::Hard,
            0.0,
            DynamicRangeMethod::Peak,
            lookahead_ms,
            false,
        );
        let y = audio.apply_limiter(&lim)?;
        Ok(y.as_slice().ok_or("mono")?.to_vec())
    };
    let peak_db = |y: &[f64]| 20.0 * y.iter().fold(0.0f64, |m, v| m.max(v.abs())).log10();
    let y0 = run(0.0)?;
    let y5 = run(5.0)?;
    assert_close("peak without lookahead (dBFS)", peak_db(&y0), 0.0, 1e-9);
    let a: f64 = (-1.0 / (0.001 * FS)).exp();
    let predicted = -20.0 * (1.0 - a.powi(241)).log10();
    assert_close("predicted overshoot (dB)", predicted, 0.0575, 1e-3);
    assert_close(
        "peak with 5 ms lookahead (dBFS)",
        peak_db(&y5),
        -6.0 + predicted,
        1e-9,
    );

    let ms = |n: usize| n as f64 / FS * 1e3;
    let (n0, n1) = (720, 1_680); // 15 to 35 ms
    let series = |y: &[f64]| -> Vec<(f64, f64)> { (n0..n1).map(|n| (ms(n), y[n])).collect() };
    let ceiling = 10f64.powf(-6.0 / 20.0);

    figs.svg(
        "ch12-lookahead",
        "A limiter (ceiling -6 dBFS, 1 ms attack) on a step from silence to 0 dBFS at 20 ms, \
         from apply_limiter. Without lookahead the start of the step passes at full level, \
         6 dB over the ceiling; with 5 ms of lookahead the gain is already reduced when the \
         step arrives and the overshoot is 0.06 dB.",
        SIZE,
        |area| {
            Chart::new("Time (ms)", "Amplitude", (15.0, 35.0), (0.0, 1.1))
                .ticks(10, 6)
                .add(Series::new("input", series(&x), GREY, Style::Dashed(2)))
                .add(Series::line("no lookahead", series(&y0), VERMILION))
                .add(Series::line("5 ms lookahead", series(&y5), BLUE))
                .hrule(ceiling, "ceiling -6 dBFS", INK)
                .legend(LegendPos::UpperRight)
                .draw(area)
        },
    )
}
