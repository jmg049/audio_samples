//! Explanations for IIR filter designs and parametric-EQ bands.
//!
//! Both explainers design the filter with the crate's own code
//! ([`IirFilterDesign::sos_coefficients`] and the RBJ biquad designer used by
//! the parametric EQ), list the resulting poles and zeros, and measure the
//! response with the crate's own frequency-response routines.

use num_complex::Complex;

use crate::operations::types::{BesselNorm, FilterResponse, IirFilterDesign, IirFilterType};

use super::doc::{ExplainDoc, Table, Topic, WarningLevel, fmt_num};
use super::numeric::{fmt_complex, mag_db, section_roots};
use super::{Explain, ExplainContext, note_missing_sample_rate};

/// Poles and zeros of one SOS row `[b0, b1, b2, a0, a1, a2]`.
fn section_poles_zeros(s: &[f64; 6]) -> (Vec<Complex<f64>>, Vec<Complex<f64>>) {
    (
        section_roots(s[0], s[1], s[2]),
        section_roots(s[3], s[4], s[5]),
    )
}

fn fmt_roots(r: &[Complex<f64>]) -> String {
    if r.is_empty() {
        "none".into()
    } else {
        r.iter()
            .map(|z| fmt_complex(*z))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Adds a table of poles, zeros and pole radii, returning the largest radius.
fn add_pole_zero_table(doc: &mut ExplainDoc, sos: &[[f64; 6]]) -> f64 {
    let mut max_r = 0.0f64;
    let rows = sos
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let (zeros, poles) = section_poles_zeros(s);
            let r = poles.iter().map(|p| p.norm()).fold(0.0, f64::max);
            max_r = max_r.max(r);
            vec![
                (i + 1).to_string(),
                fmt_roots(&zeros),
                fmt_roots(&poles),
                fmt_num(r),
            ]
        })
        .collect();
    doc.add_table(Table {
        caption: "Second-order sections: zeros and poles in the z-plane".into(),
        headers: vec![
            "Section".into(),
            "Zeros".into(),
            "Poles".into(),
            "max |p|".into(),
        ],
        rows,
    });
    max_r
}

/// Frequencies at which to tabulate a response around `centre`, kept inside
/// `(0, nyquist)`.
fn probe_frequencies(centres: &[f64], nyquist: f64) -> Vec<f64> {
    let mut f: Vec<f64> = vec![0.0];
    for &c in centres {
        for k in [0.25, 0.5, 1.0, 2.0, 4.0] {
            f.push(c * k);
        }
    }
    f.push(nyquist * 0.999);
    f.retain(|&x| x >= 0.0 && x < nyquist);
    f.sort_by(f64::total_cmp);
    f.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    f
}

fn response_table(freqs: &[f64], mags: &[f64]) -> Table {
    Table {
        caption: "Measured magnitude response".into(),
        headers: vec!["f (Hz)".into(), "|H|".into(), "|H| (dB)".into()],
        rows: freqs
            .iter()
            .zip(mags)
            .map(|(f, m)| {
                vec![
                    fmt_num(*f),
                    fmt_num(*m),
                    if *m > 0.0 {
                        fmt_num(mag_db(*m))
                    } else {
                        "-inf".into()
                    },
                ]
            })
            .collect(),
    }
}

fn family_name(t: IirFilterType) -> &'static str {
    match t {
        IirFilterType::Butterworth => "Butterworth",
        IirFilterType::ChebyshevI => "Chebyshev type I",
        IirFilterType::ChebyshevII => "Chebyshev type II",
        IirFilterType::Elliptic => "Elliptic (Cauer)",
        IirFilterType::Bessel => "Bessel-Thomson",
    }
}

fn response_name(r: FilterResponse) -> &'static str {
    match r {
        FilterResponse::LowPass => "low-pass",
        FilterResponse::HighPass => "high-pass",
        FilterResponse::BandPass => "band-pass",
        FilterResponse::BandStop => "band-stop",
    }
}

impl Explain for IirFilterDesign {
    /// Designs the filter at `ctx.sample_rate_hz` (required for numbers).
    fn explain_with(&self, ctx: &ExplainContext) -> ExplainDoc {
        let family = family_name(self.filter_type);
        let response = response_name(self.response);
        let order = self.order.get();
        let mut doc = ExplainDoc::new(
            Topic::IirFilter,
            format!("{family} {response} filter (order {order})"),
        );

        // Analogue prototype.
        match self.filter_type {
            IirFilterType::Butterworth => {
                doc.add_formula(
                    r"|H_a(j\Omega)|^2 = \frac{1}{1 + \left(\frac{\Omega}{\Omega_c}\right)^{2N}}",
                    "Butterworth prototype: maximally flat pass band, -3.01 dB at the cutoff",
                );
            }
            IirFilterType::ChebyshevI => {
                doc.add_formula(
                    r"|H_a(j\Omega)|^2 = \frac{1}{1 + \varepsilon^2 T_N^2\left(\frac{\Omega}{\Omega_c}\right)}, \quad \varepsilon = \sqrt{10^{r_p/10} - 1}",
                    "Chebyshev I prototype: equiripple pass band between 0 and -r_p dB",
                );
            }
            IirFilterType::ChebyshevII => {
                doc.add_formula(
                    r"|H_a(j\Omega)|^2 = \frac{1}{1 + \frac{1}{\varepsilon^2 T_N^2\left(\frac{\Omega_c}{\Omega}\right)}}, \quad \varepsilon = \frac{1}{\sqrt{10^{r_s/10} - 1}}",
                    "Chebyshev II prototype: flat pass band, equiripple stop band at least r_s dB down beyond the critical frequency",
                );
            }
            IirFilterType::Elliptic => {
                doc.add_formula(
                    r"|H_a(j\Omega)|^2 = \frac{1}{1 + \varepsilon^2 R_N^2\left(\xi, \frac{\Omega}{\Omega_c}\right)}",
                    "Elliptic prototype (R_N: elliptic rational function): equiripple in both bands, steepest transition for a given order",
                );
            }
            IirFilterType::Bessel => {
                doc.add_formula(
                    r"H_a(s) = \frac{\theta_N(0)}{\theta_N\left(\frac{s}{\omega_0}\right)}",
                    "Bessel prototype (θ_N: reverse Bessel polynomial): maximally flat group delay",
                );
            }
        }
        doc.add_formula(
            r"s = K \frac{1 - z^{-1}}{1 + z^{-1}}, \quad \Omega_c = K \tan\left(\frac{\pi f_c}{f_s}\right)",
            "Bilinear transform with pre-warping, so the digital filter has its critical frequency exactly at f_c (the code uses SciPy's normalisation K = 4; any K gives the same H(z))",
        );
        match self.response {
            FilterResponse::HighPass => {
                doc.add_formula(
                    r"s \rightarrow \frac{\Omega_c}{s}",
                    "Low-pass to high-pass transformation of the prototype",
                );
            }
            FilterResponse::BandPass => {
                doc.add_formula(
                    r"s \rightarrow \frac{s^2 + \Omega_0^2}{s B}, \quad \Omega_0 = \sqrt{\Omega_1 \Omega_2}, \quad B = \Omega_2 - \Omega_1",
                    "Low-pass to band-pass transformation (doubles the number of poles)",
                );
            }
            FilterResponse::BandStop => {
                doc.add_formula(
                    r"s \rightarrow \frac{s B}{s^2 + \Omega_0^2}, \quad \Omega_0 = \sqrt{\Omega_1 \Omega_2}, \quad B = \Omega_2 - \Omega_1",
                    "Low-pass to band-stop transformation (doubles the number of poles)",
                );
            }
            _ => {}
        }
        doc.add_formula(
            r"H(z) = \prod_{i=1}^{S} \frac{b_{0i} + b_{1i} z^{-1} + b_{2i} z^{-2}}{1 + a_{1i} z^{-1} + a_{2i} z^{-2}}",
            "Realised as a cascade of second-order sections (SOS), which is far less sensitive to coefficient rounding than one high-order polynomial",
        );
        doc.add_symbol_value("N", "prototype order", order.to_string(), "")
            .add_symbol("K", "bilinear-transform constant");
        let edges: Vec<f64> = match self.response {
            FilterResponse::LowPass | FilterResponse::HighPass => {
                self.cutoff_frequency.into_iter().collect()
            }
            _ => self
                .low_frequency
                .into_iter()
                .chain(self.high_frequency)
                .collect(),
        };
        match edges.as_slice() {
            [fc] => {
                doc.add_symbol_value("f_c", "critical (cutoff) frequency", fmt_num(*fc), "Hz");
            }
            [lo, hi] => {
                doc.add_symbol_value("f_1", "lower band edge", fmt_num(*lo), "Hz")
                    .add_symbol_value("f_2", "upper band edge", fmt_num(*hi), "Hz");
            }
            _ => {}
        }
        if let Some(rp) = self.passband_ripple {
            doc.add_symbol_value("r_p", "pass-band ripple", fmt_num(rp), "dB");
        }
        if let Some(rs) = self.stopband_attenuation {
            doc.add_symbol_value("r_s", "stop-band attenuation", fmt_num(rs), "dB");
        }
        doc.add_see_also(
            "Design as SOS",
            "audio_samples::operations::types::IirFilterDesign::sos_coefficients",
        )
        .add_see_also(
            "Apply",
            "audio_samples::AudioIirFiltering::apply_iir_filter",
        )
        .add_see_also("Python", "audio_samples.IirFilterDesign.explain");
        doc.add_trade_off("Higher order gives a steeper transition band but more phase distortion near the cutoff and poles closer to the unit circle.")
            .add_trade_off("Butterworth is flattest, Chebyshev and elliptic trade ripple for steepness, Bessel keeps the group delay flat (least ringing) at the cost of a gentle roll-off.")
            .add_trade_off("IIR filters have non-linear phase; filtfilt (forward-backward) removes it at the cost of doubling the attenuation and being non-causal.");

        let Some(fs) = ctx.valid_sample_rate() else {
            doc.set_summary(format!(
                "An order-{order} {family} {response} filter, designed from an analogue prototype with the bilinear transform. \
                 A sample rate is needed to compute coefficients."
            ));
            note_missing_sample_rate(&mut doc);
            return doc;
        };
        let nyq = fs / 2.0;
        doc.add_symbol_value("f_s", "sample rate", fmt_num(fs), "Hz");

        let sos = match self.sos_coefficients(fs) {
            Ok(s) => s,
            Err(e) => {
                doc.set_summary(format!(
                    "This {family} {response} design cannot be realised at {} Hz.",
                    fmt_num(fs)
                ));
                doc.add_warning(
                    WarningLevel::Problem,
                    format!("Design fails: {e}"),
                    "Fix the parameter named in the message.",
                );
                return doc;
            }
        };
        let n_poles: usize = sos.iter().map(|s| section_poles_zeros(s).1.len()).sum();
        doc.add_quantity(
            "Sections",
            "S",
            sos.len() as f64,
            "",
            r"\lceil \text{poles} / 2 \rceil",
        )
        .add_quantity(
            "Poles",
            r"N_p",
            n_poles as f64,
            "",
            if edges.len() == 2 { "2N" } else { "N" },
        );
        for (label, f) in edges.iter().enumerate().map(|(i, f)| {
            (
                if edges.len() == 1 {
                    "f_c"
                } else if i == 0 {
                    "f_1"
                } else {
                    "f_2"
                },
                f,
            )
        }) {
            doc.add_quantity(
                format!("Pre-warped {label}"),
                format!(r"\tfrac{{f_s}}{{\pi}} \tan(\pi {label}/f_s)"),
                fs / std::f64::consts::PI * (std::f64::consts::PI * f / fs).tan(),
                "Hz",
                r"\frac{f_s}{\pi}\tan(\pi f / f_s)",
            );
        }
        let max_r = add_pole_zero_table(&mut doc, &sos);
        doc.add_quantity(
            "Largest pole radius",
            r"\max |p|",
            max_r,
            "",
            r"\max_i |p_i|",
        )
        .add_quantity(
            "Stability margin",
            r"1 - \max |p|",
            1.0 - max_r,
            "",
            r"1 - \max_i |p_i|",
        );

        // Measured gains, using the crate's frequency-response code.
        let expected = match (self.filter_type, self.bessel_norm) {
            (IirFilterType::Butterworth, _) | (IirFilterType::Bessel, BesselNorm::Magnitude) => {
                Some(-10.0 * 2f64.log10())
            }
            (IirFilterType::ChebyshevI | IirFilterType::Elliptic, _) => {
                self.passband_ripple.map(|r| -r)
            }
            (IirFilterType::ChebyshevII, _) => self.stopband_attenuation.map(|r| -r),
            _ => None,
        };
        if let Ok((mags, _)) = self.frequency_response(&edges, fs) {
            for (i, m) in mags.iter().enumerate() {
                let name = match (edges.len(), i) {
                    (1, _) => "Measured gain at the cutoff".to_string(),
                    (_, 0) => "Measured gain at the lower edge".to_string(),
                    _ => "Measured gain at the upper edge".to_string(),
                };
                doc.add_quantity(
                    name,
                    r"20\log_{10}|H(e^{j 2\pi f/f_s})|",
                    mag_db(*m),
                    "dB",
                    r"\text{sosfreqz at the edge}",
                );
                if let Some(exp) = expected {
                    if (mag_db(*m) - exp).abs() > 0.05 {
                        doc.add_warning(
                            WarningLevel::Caution,
                            format!(
                                "Measured gain at {} Hz is {} dB; the {family} design rule predicts {} dB.",
                                fmt_num(edges[i]),
                                fmt_num(mag_db(*m)),
                                fmt_num(exp)
                            ),
                            "",
                        );
                    }
                }
            }
        }
        if let Some(exp) = expected {
            doc.add_quantity(
                "Design gain at the critical frequency",
                r"G_c",
                exp,
                "dB",
                match self.filter_type {
                    IirFilterType::ChebyshevI | IirFilterType::Elliptic => r"-r_p",
                    IirFilterType::ChebyshevII => r"-r_s",
                    _ => r"10\log_{10}(1/2)",
                },
            );
        } else if self.filter_type == IirFilterType::Bessel {
            doc.add_warning(
                WarningLevel::Info,
                format!("With Bessel normalisation {:?} the critical frequency fixes the phase or delay, not the gain.", self.bessel_norm),
                "Use BesselNorm::Magnitude for -3 dB at the cutoff.",
            );
        }
        let probes = probe_frequencies(&edges, nyq);
        if let Ok((mags, _)) = self.frequency_response(&probes, fs) {
            doc.add_table(response_table(&probes, &mags));
        }

        doc.set_summary(format!(
            "An order-{order} {family} {response} filter at {} Hz, realised as {} second-order section(s) with {n_poles} poles. \
             All poles lie inside the unit circle (largest radius {}), so the filter is {}.",
            fmt_num(fs),
            sos.len(),
            fmt_num(max_r),
            if max_r < 1.0 { "stable" } else { "UNSTABLE" }
        ));
        if max_r >= 1.0 {
            doc.add_warning(
                WarningLevel::Problem,
                "A pole lies on or outside the unit circle: the filter is unstable.",
                "",
            );
        } else if 1.0 - max_r < 1e-3 {
            doc.add_warning(
                WarningLevel::Caution,
                format!(
                    "Poles sit within {} of the unit circle (normalised edge {} of f_s, order {order}); the response is very sensitive to rounding and rings for a long time.",
                    fmt_num(1.0 - max_r),
                    fmt_num(edges.iter().copied().fold(f64::INFINITY, f64::min) / fs)
                ),
                "Lower the order, or decimate before filtering so the cutoff is a larger fraction of the sample rate.",
            );
        }
        if let Some(&f_hi) = edges.iter().max_by(|a, b| a.total_cmp(b)) {
            if f_hi > 0.4 * fs && f_hi < nyq {
                let warped = fs / std::f64::consts::PI * (std::f64::consts::PI * f_hi / fs).tan();
                doc.add_warning(
                    WarningLevel::Info,
                    format!(
                        "{} Hz is close to Nyquist: pre-warping maps it to an analogue {} Hz, so the digital response shape departs noticeably from the analogue prototype.",
                        fmt_num(f_hi),
                        fmt_num(warped)
                    ),
                    "",
                );
            }
        }
        doc
    }
}

#[cfg(feature = "parametric-eq")]
mod eq {
    use super::*;
    use crate::IirFilter;
    use crate::operations::parametric_eq::design_eq_band_filter;
    use crate::operations::types::{EqBand, EqBandType};

    fn coefficient_formulas(t: EqBandType) -> Vec<(&'static str, &'static str)> {
        match t {
            EqBandType::Peak => vec![
                (
                    r"b_0 = 1 + \alpha A, \quad b_1 = -2\cos\omega_0, \quad b_2 = 1 - \alpha A",
                    "Numerator (peaking EQ)",
                ),
                (
                    r"a_0 = 1 + \frac{\alpha}{A}, \quad a_1 = -2\cos\omega_0, \quad a_2 = 1 - \frac{\alpha}{A}",
                    "Denominator (peaking EQ)",
                ),
            ],
            EqBandType::LowShelf => vec![
                (
                    r"b_0 = A\left[(A+1) - (A-1)\cos\omega_0 + 2\sqrt{A}\alpha\right], \quad b_1 = 2A\left[(A-1) - (A+1)\cos\omega_0\right], \quad b_2 = A\left[(A+1) - (A-1)\cos\omega_0 - 2\sqrt{A}\alpha\right]",
                    "Numerator (low shelf)",
                ),
                (
                    r"a_0 = (A+1) + (A-1)\cos\omega_0 + 2\sqrt{A}\alpha, \quad a_1 = -2\left[(A-1) + (A+1)\cos\omega_0\right], \quad a_2 = (A+1) + (A-1)\cos\omega_0 - 2\sqrt{A}\alpha",
                    "Denominator (low shelf)",
                ),
            ],
            EqBandType::HighShelf => vec![
                (
                    r"b_0 = A\left[(A+1) + (A-1)\cos\omega_0 + 2\sqrt{A}\alpha\right], \quad b_1 = -2A\left[(A-1) + (A+1)\cos\omega_0\right], \quad b_2 = A\left[(A+1) + (A-1)\cos\omega_0 - 2\sqrt{A}\alpha\right]",
                    "Numerator (high shelf)",
                ),
                (
                    r"a_0 = (A+1) - (A-1)\cos\omega_0 + 2\sqrt{A}\alpha, \quad a_1 = 2\left[(A-1) - (A+1)\cos\omega_0\right], \quad a_2 = (A+1) - (A-1)\cos\omega_0 - 2\sqrt{A}\alpha",
                    "Denominator (high shelf)",
                ),
            ],
            EqBandType::LowPass => vec![(
                r"b_0 = b_2 = \frac{1 - \cos\omega_0}{2}, \quad b_1 = 1 - \cos\omega_0, \quad a_0 = 1 + \alpha, \quad a_1 = -2\cos\omega_0, \quad a_2 = 1 - \alpha",
                "Coefficients (low-pass)",
            )],
            EqBandType::HighPass => vec![(
                r"b_0 = b_2 = \frac{1 + \cos\omega_0}{2}, \quad b_1 = -(1 + \cos\omega_0), \quad a_0 = 1 + \alpha, \quad a_1 = -2\cos\omega_0, \quad a_2 = 1 - \alpha",
                "Coefficients (high-pass)",
            )],
            EqBandType::BandPass => vec![(
                r"b_0 = \alpha, \quad b_1 = 0, \quad b_2 = -\alpha, \quad a_0 = 1 + \alpha, \quad a_1 = -2\cos\omega_0, \quad a_2 = 1 - \alpha",
                "Coefficients (band-pass, 0 dB peak gain)",
            )],
            EqBandType::BandStop => vec![(
                r"b_0 = 1, \quad b_1 = -2\cos\omega_0, \quad b_2 = 1, \quad a_0 = 1 + \alpha, \quad a_1 = -2\cos\omega_0, \quad a_2 = 1 - \alpha",
                "Coefficients (band-stop / notch)",
            )],
        }
    }

    impl Explain for EqBand {
        /// Designs the band at `ctx.sample_rate_hz` (required for numbers).
        fn explain_with(&self, ctx: &ExplainContext) -> ExplainDoc {
            let t = self.band_type;
            let mut doc = ExplainDoc::new(Topic::Equalisation, format!("Parametric EQ band ({t})"));
            doc.add_formula(
                r"A = 10^{G/40}, \quad \omega_0 = \frac{2\pi f_0}{f_s}, \quad \alpha = \frac{\sin\omega_0}{2Q}",
                "RBJ Audio-EQ-Cookbook intermediates, as implemented (shelves use Q directly, not the shelf-slope S)",
            );
            for (latex, caption) in coefficient_formulas(t) {
                doc.add_formula(latex, caption);
            }
            doc.add_formula(
                r"H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}",
                "Biquad transfer function (the code divides every coefficient by a_0)",
            );
            doc.add_symbol_value(
                "f_0",
                "centre / corner frequency",
                fmt_num(self.frequency),
                "Hz",
            )
            .add_symbol_value("G", "gain", fmt_num(self.gain_db), "dB")
            .add_symbol_value("Q", "quality factor", fmt_num(self.q_factor), "");
            doc.add_see_also(
                "Apply one band",
                "audio_samples::AudioParametricEq::apply_eq_band",
            )
            .add_see_also("Python", "audio_samples.EqBand.explain");
            doc.add_trade_off("Higher Q narrows the band affected; lower Q gives broad, gentle tonal changes.")
                .add_trade_off("Boosting by G dB raises the peak level by G dB; leave headroom or follow with a limiter.");
            if !self.enabled {
                doc.add_warning(
                    WarningLevel::Info,
                    "This band is disabled; the EQ skips it.",
                    "",
                );
            }
            if matches!(
                t,
                EqBandType::LowPass
                    | EqBandType::HighPass
                    | EqBandType::BandPass
                    | EqBandType::BandStop
            ) && self.gain_db != 0.0
            {
                doc.add_warning(
                    WarningLevel::Info,
                    format!(
                        "gain_db = {} is ignored for {t} bands.",
                        fmt_num(self.gain_db)
                    ),
                    "",
                );
            }
            let Some(fs) = ctx.valid_sample_rate() else {
                doc.set_summary(format!(
                    "A {t} biquad at {} Hz designed with the RBJ cookbook formulas.",
                    fmt_num(self.frequency)
                ));
                note_missing_sample_rate(&mut doc);
                return doc;
            };
            doc.add_symbol_value("f_s", "sample rate", fmt_num(fs), "Hz");
            if let Err(e) = self.validate(fs) {
                doc.add_warning(WarningLevel::Problem, format!("Invalid band: {e}"), "");
                doc.set_summary(format!(
                    "This {t} band cannot be applied at {} Hz.",
                    fmt_num(fs)
                ));
                return doc;
            }
            let a_lin = 10f64.powf(self.gain_db / 40.0);
            let w0 = 2.0 * std::f64::consts::PI * self.frequency / fs;
            let alpha = w0.sin() / (2.0 * self.q_factor);
            doc.add_quantity("Amplitude parameter", "A", a_lin, "", r"10^{G/40}")
                .add_quantity(
                    "Normalised frequency",
                    r"\omega_0",
                    w0,
                    "rad/sample",
                    r"2\pi f_0 / f_s",
                )
                .add_quantity(
                    "Bandwidth parameter",
                    r"\alpha",
                    alpha,
                    "",
                    r"\sin\omega_0 / (2Q)",
                );
            if t == EqBandType::Peak {
                let bw =
                    2.0 / std::f64::consts::LN_2 * (1.0 / (2.0 * self.q_factor)).asinh() * w0.sin()
                        / w0;
                doc.add_quantity("Bandwidth", "BW", bw, "octaves", r"\frac{2}{\ln 2}\,\mathrm{asinh}\left(\frac{1}{2Q}\right)\frac{\sin\omega_0}{\omega_0}");
            }
            let (b, a) = design_eq_band_filter(self, fs);
            let sos = [[b[0], b[1], b[2], a[0], a[1], a[2]]];
            let max_r = add_pole_zero_table(&mut doc, &sos);
            doc.add_quantity(
                "Largest pole radius",
                r"\max |p|",
                max_r,
                "",
                r"\max_i |p_i|",
            );
            let filter = IirFilter::new(b, a);
            let (centre, _) = filter.frequency_response(&[self.frequency, 0.0, fs / 2.0], fs);
            if centre[0] > 0.0 {
                doc.add_quantity(
                    "Measured gain at f0",
                    r"|H(e^{j\omega_0})|",
                    mag_db(centre[0]),
                    "dB",
                    r"20\log_{10}|H(e^{j\omega_0})|",
                );
            }
            if centre[1] > 0.0 {
                doc.add_quantity(
                    "Measured gain at DC",
                    "|H(1)|",
                    mag_db(centre[1]),
                    "dB",
                    r"20\log_{10}|H(1)|",
                );
            }
            if centre[2] > 0.0 {
                doc.add_quantity(
                    "Measured gain at Nyquist",
                    "|H(-1)|",
                    mag_db(centre[2]),
                    "dB",
                    r"20\log_{10}|H(-1)|",
                );
            }
            let probes = probe_frequencies(&[self.frequency], fs / 2.0);
            let (mags, _) = filter.frequency_response(&probes, fs);
            doc.add_table(response_table(&probes, &mags));
            let expectation = match t {
                EqBandType::Peak => format!("exactly {} dB at f0", fmt_num(self.gain_db)),
                EqBandType::LowShelf | EqBandType::HighShelf => format!(
                    "{} dB (half the shelf gain) at f0",
                    fmt_num(self.gain_db / 2.0)
                ),
                EqBandType::LowPass | EqBandType::HighPass => format!(
                    "{} dB (= 20 log10 Q) at f0",
                    fmt_num(20.0 * self.q_factor.log10())
                ),
                EqBandType::BandPass => "0 dB at f0".to_string(),
                EqBandType::BandStop => "a null at f0".to_string(),
            };
            doc.set_summary(format!(
                "A {t} biquad at {} Hz (Q = {}) designed with the RBJ cookbook formulas; by design it has {expectation}. \
                 Its poles lie at radius {} (stable).",
                fmt_num(self.frequency),
                fmt_num(self.q_factor),
                fmt_num(max_r)
            ));
            if self.frequency > 0.25 * fs {
                doc.add_warning(
                    WarningLevel::Info,
                    format!(
                        "f0 is {} of the sample rate; the bilinear transform compresses the response towards Nyquist (\"cramping\"), so the band is narrower above f0 than below.",
                        fmt_num(self.frequency / fs)
                    ),
                    "",
                );
            }
            doc
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn butterworth_is_minus_3db_at_cutoff_and_stable() {
        let d = IirFilterDesign::butterworth_lowpass(nzu!(4), 1_000.0)
            .explain_with(&ExplainContext::new().with_sample_rate(48_000.0));
        let g = d.quantity("Measured gain at the cutoff").unwrap().value;
        assert!((g + 3.0103).abs() < 1e-3, "{g}");
        assert_eq!(d.quantity("Poles").unwrap().value, 4.0);
        assert_eq!(d.quantity("Sections").unwrap().value, 2.0);
        let r = d.quantity("Largest pole radius").unwrap().value;
        assert!(r < 1.0 && r > 0.8);
        assert!(!d.has_warning(WarningLevel::Caution));
        assert!(d.to_text().contains("stable"));
    }

    #[test]
    fn pole_radius_matches_independent_bilinear_calculation() {
        // First-order Butterworth low-pass: analogue pole -Ωc, Ωc = K tan(π fc/fs),
        // bilinear with K: z = (K + s)/(K - s) => radius (K - Ωc)/(K + Ωc) with K = 1.
        let (fc, fs) = (1_000.0, 48_000.0);
        let t = (std::f64::consts::PI * fc / fs).tan();
        let expected = (1.0 - t) / (1.0 + t);
        let d = IirFilterDesign::butterworth_lowpass(nzu!(1), fc)
            .explain_with(&ExplainContext::new().with_sample_rate(fs));
        assert!((d.quantity("Largest pole radius").unwrap().value - expected).abs() < 1e-12);
    }

    #[test]
    fn missing_rate_and_invalid_cutoff() {
        let d = IirFilterDesign::butterworth_lowpass(nzu!(2), 1_000.0).explain();
        assert!(d.has_warning(WarningLevel::Info));
        assert!(d.quantity("Poles").is_none());
        let d = IirFilterDesign::butterworth_lowpass(nzu!(2), 30_000.0)
            .explain_with(&ExplainContext::new().with_sample_rate(48_000.0));
        assert!(d.has_warning(WarningLevel::Problem));
    }

    #[test]
    fn near_unit_circle_warning() {
        let d = IirFilterDesign::butterworth_lowpass(nzu!(8), 20.0)
            .explain_with(&ExplainContext::new().with_sample_rate(96_000.0));
        assert!(d.has_warning_containing("unit circle"), "{:?}", d.warnings);
    }

    #[test]
    fn chebyshev_gains_follow_the_design_rule() {
        let mut d = IirFilterDesign::butterworth_lowpass(nzu!(5), 2_000.0);
        d.filter_type = IirFilterType::ChebyshevI;
        d.passband_ripple = Some(1.0);
        let doc = d.explain_with(&ExplainContext::new().with_sample_rate(48_000.0));
        assert!((doc.quantity("Measured gain at the cutoff").unwrap().value + 1.0).abs() < 1e-6);
        assert!(!doc.has_warning(WarningLevel::Caution));
        let mut d2 = d;
        d2.filter_type = IirFilterType::ChebyshevII;
        d2.stopband_attenuation = Some(40.0);
        let doc = d2.explain_with(&ExplainContext::new().with_sample_rate(48_000.0));
        assert!((doc.quantity("Measured gain at the cutoff").unwrap().value + 40.0).abs() < 1e-6);
    }

    #[test]
    fn band_pass_has_two_edges() {
        let d = IirFilterDesign::butterworth_bandpass(nzu!(2), 300.0, 3_000.0)
            .explain_with(&ExplainContext::new().with_sample_rate(16_000.0));
        assert_eq!(d.quantity("Poles").unwrap().value, 4.0);
        for name in [
            "Measured gain at the lower edge",
            "Measured gain at the upper edge",
        ] {
            assert!((d.quantity(name).unwrap().value + 3.0103).abs() < 1e-3);
        }
    }

    #[cfg(feature = "parametric-eq")]
    #[test]
    fn eq_band_gains() {
        use crate::operations::types::EqBand;
        let ctx = ExplainContext::new().with_sample_rate(48_000.0);
        let d = EqBand::peak(1_000.0, 6.0, 1.0).explain_with(&ctx);
        assert!((d.quantity("Measured gain at f0").unwrap().value - 6.0).abs() < 1e-9);
        let d = EqBand::low_shelf(200.0, 6.0, 0.707).explain_with(&ctx);
        assert!((d.quantity("Measured gain at f0").unwrap().value - 3.0).abs() < 1e-6);
        assert!((d.quantity("Measured gain at DC").unwrap().value - 6.0).abs() < 1e-6);
        let d = EqBand::peak(30_000.0, 6.0, 1.0).explain_with(&ctx);
        assert!(d.has_warning(WarningLevel::Problem));
        let mut off = EqBand::peak(1_000.0, 6.0, 1.0);
        off.enabled = false;
        assert!(off.explain_with(&ctx).has_warning_containing("disabled"));
    }
}
