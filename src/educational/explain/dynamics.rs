//! Explanations for the compressor and limiter configurations.
//!
//! The static curves are tabulated with the crate's own gain computers
//! ([`calculate_compression_gain`] and [`calculate_limiting_gain`]), and the
//! smoothing coefficients use the envelope follower's convention, so the
//! numbers match what `apply_compressor` / `apply_limiter` do.

use crate::operations::dynamic_range::{calculate_compression_gain, calculate_limiting_gain};
use crate::operations::types::{CompressorConfig, DynamicRangeMethod, KneeType, LimiterConfig};

use super::doc::{ExplainDoc, Table, Topic, WarningLevel, fmt_num};
use super::{Explain, ExplainContext, note_missing_sample_rate};

/// One-pole smoothing coefficient for a time constant in milliseconds, as in
/// `EnvelopeFollower::new` (0 for a zero time constant).
fn smoothing_coeff(ms: f64, fs: f64) -> f64 {
    if ms > 0.0 {
        (-1.0 / (ms * 0.001 * fs)).exp()
    } else {
        0.0
    }
}

fn detector_formula(method: DynamicRangeMethod) -> (&'static str, &'static str) {
    match method {
        DynamicRangeMethod::Peak => (r"x_d[n] = |x[n]|", "Peak detector"),
        DynamicRangeMethod::Rms => (
            r"x_d[n] = \sqrt{\frac{1}{N_{rms}}\sum_{i=0}^{N_{rms}-1} x^2[n-i]}",
            "RMS detector over the last N_rms = 10 ms of samples (fewer at the very start)",
        ),
        _ => (
            r"x_d[n] = \max\left(|x[n]|, \sqrt{\frac{1}{N_{rms}}\sum_{i=0}^{N_{rms}-1} x^2[n-i]}\right)",
            "Hybrid detector: the larger of peak and RMS over N_rms = 10 ms",
        ),
    }
}

/// Formulas, symbols and quantities shared by compressor and limiter.
fn add_ballistics(
    doc: &mut ExplainDoc,
    attack_ms: f64,
    release_ms: f64,
    lookahead_ms: f64,
    method: DynamicRangeMethod,
    fs: Option<f64>,
) {
    let (det, det_caption) = detector_formula(method);
    doc.add_formula(det, det_caption)
        .add_formula(
            r"e[n] = \alpha \, e[n-1] + (1 - \alpha) \, x_d[n], \quad \alpha = \begin{cases} \alpha_A & x_d[n] > e[n-1] \\ \alpha_R & x_d[n] \le e[n-1] \end{cases}",
            "Envelope follower: attack coefficient while the detector is above the envelope, release otherwise",
        )
        .add_formula(r"\alpha = e^{-\frac{1}{\tau f_s}}", "One-pole coefficient for a time constant τ (seconds); after τ the envelope has covered 63.2% of a step")
        .add_formula(r"L[n] = 20 \log_{10} e[n]", "Envelope level in dBFS (-80 dBFS when e = 0)")
        .add_formula(
            r"\overline{GR}[n] = \max_{0 \le k \le D} GR[n + k], \quad D = \mathrm{round}(t_{la} f_s)",
            "Lookahead: each sample receives the largest reduction computed over the next D samples, so the gain starts falling before a peak arrives (the output is re-aligned with the input)",
        );
    doc.add_symbol_value(r"\tau_A", "attack time", fmt_num(attack_ms), "ms")
        .add_symbol_value(r"\tau_R", "release time", fmt_num(release_ms), "ms");
    if let Some(fs) = fs {
        let (aa, ar) = (
            smoothing_coeff(attack_ms, fs),
            smoothing_coeff(release_ms, fs),
        );
        doc.add_quantity(
            "Attack coefficient",
            r"\alpha_A",
            aa,
            "",
            r"e^{-1/(\tau_A f_s)}",
        )
        .add_quantity(
            "Release coefficient",
            r"\alpha_R",
            ar,
            "",
            r"e^{-1/(\tau_R f_s)}",
        )
        .add_quantity(
            "Attack 10-90% rise time",
            r"\tau_A \ln 9",
            attack_ms * 9f64.ln(),
            "ms",
            r"\tau_A \ln 9",
        )
        .add_quantity(
            "Release 90-10% fall time",
            r"\tau_R \ln 9",
            release_ms * 9f64.ln(),
            "ms",
            r"\tau_R \ln 9",
        )
        .add_quantity(
            "Lookahead",
            "D",
            (lookahead_ms * 0.001 * fs).round(),
            "samples",
            r"\mathrm{round}(t_{la} f_s / 1000)",
        );
        if method != DynamicRangeMethod::Peak {
            doc.add_quantity(
                "RMS window",
                r"N_{rms}",
                (0.01 * fs).max(1.0).floor(),
                "samples",
                r"\lfloor 0.01 f_s \rfloor",
            );
        }
    } else {
        note_missing_sample_rate(doc);
    }
    doc.add_symbol_value(r"t_{la}", "lookahead time", fmt_num(lookahead_ms), "ms");
    if lookahead_ms <= 0.0 {
        doc.add_warning(
            WarningLevel::Info,
            "No lookahead: the gain reacts only after the envelope has risen, so the start of each transient passes partly unreduced (for roughly the attack time).",
            "A lookahead of about the attack time lets the gain anticipate peaks.",
        );
    }
    if release_ms < attack_ms {
        doc.add_warning(
            WarningLevel::Info,
            "The release is shorter than the attack, which is unusual and tends to cause audible pumping.",
            "",
        );
    }
}

/// Input levels at which to tabulate a static curve around `threshold`.
fn probe_levels(threshold: f64, knee: f64) -> Vec<f64> {
    let mut v = vec![
        threshold - 12.0,
        threshold - 6.0,
        threshold,
        threshold + 6.0,
        threshold + 12.0,
        threshold + 24.0,
        0.0,
    ];
    if knee > 0.0 {
        v.push(threshold - knee / 2.0);
        v.push(threshold + knee / 2.0);
    }
    v.retain(|l| *l <= 0.0 && *l >= -120.0);
    v.sort_by(f64::total_cmp);
    v.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    v
}

impl Explain for CompressorConfig {
    /// Uses `ctx.sample_rate_hz` for the smoothing coefficients and validation.
    fn explain_with(&self, ctx: &ExplainContext) -> ExplainDoc {
        let (t, r, w) = (self.threshold_db, self.ratio, self.knee_width_db);
        let soft = self.knee_type == KneeType::Soft && w > 0.0;
        let mut doc = ExplainDoc::new(Topic::Dynamics, "Compressor");
        doc.add_formula(r"o = L - T", "Overshoot above the threshold (dB)")
            .add_formula(
                r"GR = \begin{cases} 0 & o \le 0 \\ o \left(1 - \frac{1}{R}\right) & o > 0 \end{cases}",
                "Hard-knee gain reduction (dB): above T the output rises only 1/R dB per input dB",
            );
        if soft {
            doc.add_formula(
                r"GR = \left(1 - \frac{1}{R}\right) \frac{(o + W/2)^2}{2W} \quad (|o| \le W/2)",
                "Soft knee (Giannoulis, Massberg and Reiss 2012), centred on T: reduction starts W/2 below the threshold and joins the hard-knee line W/2 above it",
            );
        }
        doc.add_formula(r"y[n] = x[n] \cdot 10^{\frac{M - \overline{GR}[n]}{20}}", "Output: the held gain reduction (see Lookahead below) and the make-up gain applied to the sample");
        doc.add_symbol_value("T", "threshold", fmt_num(t), "dBFS")
            .add_symbol_value("R", "ratio", fmt_num(r), ": 1")
            .add_symbol_value("M", "make-up gain", fmt_num(self.makeup_gain_db), "dB")
            .add_symbol_value(
                "W",
                "knee width",
                if soft {
                    fmt_num(w)
                } else {
                    "0 (hard knee)".into()
                },
                "dB",
            );
        add_ballistics(
            &mut doc,
            self.attack_ms,
            self.release_ms,
            self.lookahead_ms,
            self.detection_method,
            ctx.valid_sample_rate(),
        );

        let gr = |l: f64| calculate_compression_gain(l, t, r, self.knee_type, w);
        let rows: Vec<Vec<String>> = probe_levels(t, if soft { w } else { 0.0 })
            .into_iter()
            .map(|l| {
                let g = gr(l);
                vec![fmt_num(l), fmt_num(g), fmt_num(l - g + self.makeup_gain_db)]
            })
            .collect();
        doc.add_table(Table {
            caption: "Static curve (calculate_compression_gain)".into(),
            headers: vec![
                "Input level (dBFS)".into(),
                "Gain reduction (dB)".into(),
                "Output level (dBFS)".into(),
            ],
            rows,
        });
        let full_scale_out = -gr(0.0) + self.makeup_gain_db;
        doc.add_quantity(
            "Gain reduction at 0 dBFS",
            "GR(0)",
            gr(0.0),
            "dB",
            r"\text{static curve at } L = 0",
        )
        .add_quantity(
            "Output for a 0 dBFS input",
            r"M - GR(0)",
            full_scale_out,
            "dBFS",
            r"M - GR(0)",
        )
        .add_quantity("Slope above the knee", r"1/R", 1.0 / r, "dB/dB", r"1/R");
        doc.set_summary(format!(
            "Above {} dBFS the compressor lets the level rise only 1 dB per {} dB of input{}; the gain follows the \
             signal with a {} ms attack and {} ms release, then {} dB of make-up gain is added.",
            fmt_num(t),
            fmt_num(r),
            if soft { format!(" (soft knee from {} to {} dBFS)", fmt_num(t - w / 2.0), fmt_num(t + w / 2.0)) } else { String::new() },
            fmt_num(self.attack_ms),
            fmt_num(self.release_ms),
            fmt_num(self.makeup_gain_db)
        ));
        doc.add_trade_off("A fast attack catches transients but dulls them; a slow attack lets them through and keeps punch.")
            .add_trade_off("A fast release restores level quickly but can pump; a slow release is smoother but holds down the following material.")
            .add_trade_off("RMS detection follows loudness; peak detection follows the waveform and reacts to short spikes.");
        if r <= 1.0 {
            doc.add_warning(
                WarningLevel::Info,
                "With a ratio of 1 the compressor never reduces gain.",
                "",
            );
        }
        if full_scale_out > 0.0 {
            doc.add_warning(
                WarningLevel::Caution,
                format!("A full-scale input comes out at {} dBFS after make-up gain, which will clip in fixed-point formats.", fmt_num(full_scale_out)),
                "Reduce the make-up gain or follow the compressor with a limiter.",
            );
        }
        if let Some(fs) = ctx.valid_sample_rate() {
            if let Err(e) = self.validate(fs) {
                doc.add_warning(
                    WarningLevel::Problem,
                    format!("Invalid configuration: {e}"),
                    "",
                );
            }
        }
        doc.add_see_also(
            "Apply",
            "audio_samples::AudioDynamicRange::apply_compressor",
        )
        .add_see_also(
            "Apply (Python)",
            "audio_samples.AudioSamples.apply_compressor",
        )
        .add_see_also(
            "Gain computer",
            "audio_samples::operations::dynamic_range::calculate_compression_gain",
        )
        .add_see_also("Python", "audio_samples.CompressorConfig.explain");
        doc
    }
}

impl Explain for LimiterConfig {
    /// Uses `ctx.sample_rate_hz` for the smoothing coefficients and validation.
    fn explain_with(&self, ctx: &ExplainContext) -> ExplainDoc {
        let (c, w) = (self.ceiling_db, self.knee_width_db);
        let soft = self.knee_type == KneeType::Soft && w > 0.0;
        let mut doc = ExplainDoc::new(Topic::Dynamics, "Limiter");
        doc.add_formula(r"o = L - C", "Overshoot above the ceiling (dB)")
            .add_formula(
                r"GR = \max(0, o)",
                "Hard-knee limiting: everything above the ceiling is removed (infinite ratio)",
            );
        if soft {
            doc.add_formula(
                r"GR = \frac{(o + W/2)^2}{2W} \quad (|o| \le W/2)",
                "Soft knee centred on the ceiling: reduction starts W/2 below C and equals the overshoot from W/2 above it",
            );
        }
        doc.add_formula(
            r"y[n] = x[n] \cdot 10^{-\frac{\overline{GR}[n]}{20}}",
            "Output: the held gain reduction applied to the sample (no make-up gain)",
        );
        doc.add_symbol_value("C", "ceiling", fmt_num(c), "dBFS")
            .add_symbol_value(
                "W",
                "knee width",
                if soft {
                    fmt_num(w)
                } else {
                    "0 (hard knee)".into()
                },
                "dB",
            );
        add_ballistics(
            &mut doc,
            self.attack_ms,
            self.release_ms,
            self.lookahead_ms,
            self.detection_method,
            ctx.valid_sample_rate(),
        );
        let gr = |l: f64| calculate_limiting_gain(l, c, self.knee_type, w);
        let rows = probe_levels(c, if soft { w } else { 0.0 })
            .into_iter()
            .chain([c + 3.0, c + 6.0])
            .map(|l| vec![fmt_num(l), fmt_num(gr(l)), fmt_num(l - gr(l))])
            .collect();
        doc.add_table(Table {
            caption: "Static curve (calculate_limiting_gain)".into(),
            headers: vec![
                "Envelope level (dBFS)".into(),
                "Gain reduction (dB)".into(),
                "Output level (dBFS)".into(),
            ],
            rows,
        });
        doc.add_quantity(
            "Gain reduction for a 0 dBFS envelope",
            "GR(0)",
            gr(0.0),
            "dB",
            r"\text{static curve at } L = 0",
        );
        doc.set_summary(format!(
            "Gain is reduced so that the smoothed envelope does not exceed {} dBFS ({} ms attack, {} ms release).",
            fmt_num(c),
            fmt_num(self.attack_ms),
            fmt_num(self.release_ms)
        ));
        doc.add_trade_off("A shorter attack holds peaks closer to the ceiling but distorts low frequencies more.")
            .add_trade_off("Because the gain follows a smoothed envelope, individual samples can exceed the ceiling briefly unless the lookahead covers the attack; a final clipper guarantees the ceiling.");
        if self.isp_limiting {
            doc.add_warning(
                WarningLevel::Info,
                "isp_limiting is set, but apply_limiter does not read it: no oversampled (inter-sample) peak detection is performed.",
                "Leave headroom (e.g. a ceiling of -1 dBFS) if the output will be resampled or encoded.",
            );
        }
        if let Some(fs) = ctx.valid_sample_rate() {
            if let Err(e) = self.validate(fs) {
                doc.add_warning(
                    WarningLevel::Problem,
                    format!("Invalid configuration: {e}"),
                    "",
                );
            }
        }
        doc.add_see_also("Apply", "audio_samples::AudioDynamicRange::apply_limiter")
            .add_see_also("Apply (Python)", "audio_samples.AudioSamples.apply_limiter")
            .add_see_also(
                "Gain computer",
                "audio_samples::operations::dynamic_range::calculate_limiting_gain",
            )
            .add_see_also("Python", "audio_samples.LimiterConfig.explain");
        doc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::traits::AudioDynamicRange;
    use crate::{AudioSamples, sample_rate};

    #[test]
    fn compressor_curve_matches_closed_form() {
        let mut c = CompressorConfig::new();
        c.threshold_db = -20.0;
        c.ratio = 4.0;
        c.knee_type = KneeType::Hard;
        c.makeup_gain_db = 0.0;
        let d = c.explain_with(&ExplainContext::new().with_sample_rate(48_000.0));
        // 20 dB over a 4:1 threshold: 15 dB of reduction.
        assert_eq!(d.quantity("Gain reduction at 0 dBFS").unwrap().value, 15.0);
        // α = exp(-1/(τ fs)).
        let aa = d.quantity("Attack coefficient").unwrap().value;
        assert!((aa - (-1.0 / (c.attack_ms * 1e-3 * 48_000.0)).exp()).abs() < 1e-15);
        let mut loud = c;
        loud.makeup_gain_db = 20.0;
        assert!(loud.explain().has_warning(WarningLevel::Caution));
    }

    #[test]
    fn knee_follows_the_documented_quadratic() {
        let (t, r, w): (f64, f64, f64) = (-20.0, 4.0, 10.0);
        let slope = 1.0 - 1.0 / r;
        for o in [-6.0, -5.0, -2.0, 0.0, 2.0, 5.0] {
            let expected = if o < -w / 2.0 {
                0.0
            } else if o <= w / 2.0 {
                slope * (o + w / 2.0).powi(2) / (2.0 * w)
            } else {
                slope * o
            };
            let got = calculate_compression_gain(t + o, t, r, KneeType::Soft, w);
            assert!((got - expected).abs() < 1e-12, "o={o}: {got} vs {expected}");
        }
        // Above the knee the soft and hard curves coincide.
        let soft = calculate_compression_gain(t + 8.0, t, r, KneeType::Soft, w);
        assert!((soft - slope * 8.0).abs() < 1e-12);
        // Limiter: slope 1.
        let g = calculate_limiting_gain(-1.0, -1.0, KneeType::Soft, 4.0);
        assert!((g - 4.0 / 8.0).abs() < 1e-12);
    }

    /// The explanation states that each sample receives the largest reduction
    /// of the next D samples, so adding lookahead can only lower the output
    /// magnitude, sample by sample.
    #[test]
    fn lookahead_only_increases_reduction() {
        let fs = 48_000u32;
        let data: Vec<f64> = (0..4800)
            .map(|i| {
                let burst = if (1000..1600).contains(&i) { 0.9 } else { 0.05 };
                burst * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / f64::from(fs)).sin()
            })
            .collect();
        let audio = AudioSamples::<f64>::from_mono_vec(
            non_empty_slice::NonEmptyVec::new(data).unwrap(),
            sample_rate!(48000),
        );
        let mut a = CompressorConfig::new();
        a.lookahead_ms = 0.0;
        a.makeup_gain_db = 0.0;
        let mut b = a;
        b.lookahead_ms = 5.0;
        let ya = audio.apply_compressor(&a).unwrap();
        let yb = audio.apply_compressor(&b).unwrap();
        let (sa, sb) = (ya.as_slice().unwrap(), yb.as_slice().unwrap());
        assert!(sa.iter().zip(sb).all(|(x, y)| y.abs() <= x.abs() + 1e-12));
        assert!(
            sa.iter().zip(sb).any(|(x, y)| y.abs() < x.abs() - 1e-6),
            "lookahead had no effect"
        );
        assert!(a.explain().has_warning_containing("No lookahead"));
        assert!(!b.explain().has_warning_containing("No lookahead"));
    }

    #[test]
    fn limiter_table_and_isp_note() {
        let l = LimiterConfig::mastering();
        let d = l.explain_with(&ExplainContext::new().with_sample_rate(44_100.0));
        let expected = calculate_limiting_gain(0.0, l.ceiling_db, l.knee_type, l.knee_width_db);
        assert_eq!(
            d.quantity("Gain reduction for a 0 dBFS envelope")
                .unwrap()
                .value,
            expected
        );
        assert!(expected > 0.0);
        assert_eq!(d.has_warning_containing("isp_limiting"), l.isp_limiting);
    }
}
