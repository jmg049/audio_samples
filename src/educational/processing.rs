//! `AudioProcessingExplainText` implementation for `AudioSamples<'static, T>`.
//!
//! Each explain_text function returns a string in the format:
//!
//! ```text
//! [operation: Name]
//! [formula: LaTeX source]
//! term-maths rendering of the formula
//!
//! prose paragraphs
//! ```
//!
//! The HTML renderer parses the operation name and LaTeX for KaTeX rendering.
//! Terminal output (via `Explanation::surface`) shows the rendering and prose
//! directly. Every formula is chosen so that `term-maths` renders it without
//! leaving raw `\commands` behind (no `\operatorname`, no `/` inside
//! arguments, no `cases` environments), and every claim describes what the
//! code in `src/operations` actually does.
//!
//! The explanation functions only see the signal before and after the call,
//! never the arguments. Where a parameter matters (the normalisation method,
//! a gain, the clip bounds) it is inferred from the two snapshots, and the
//! text says that it was inferred.

use crate::operations::traits::{AudioProcessing, AudioProcessingExplainText, AudioStatistics};
use crate::repr::{AudioData, AudioSamples};
use crate::traits::StandardSample;

// ─── Shared helpers ─────────────────────────────────────────────────────────

/// Builds the `[operation: …]` / `[formula: …]` header plus the terminal
/// rendering of `latex`, followed by the blank line that separates it from
/// the prose.
fn header(op: &str, latex: &str) -> String {
    let rendered = term_maths::render(latex).to_string();
    // The HTML parser skips the rendering up to the first blank line, so the
    // rendering itself must not contain one.
    let rendered = rendered
        .lines()
        .filter(|l| !l.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    format!("[operation: {op}]\n[formula: {latex}]\n{rendered}\n\n")
}

/// Rewrites `[operation: Name]` to `[operation: Name (in-place)]` so the
/// rendered call chain names the `_in_place` method that was actually called.
fn in_place(text: String) -> String {
    match text
        .strip_prefix("[operation: ")
        .and_then(|rest| rest.find("]\n").map(|end| (rest, end)))
    {
        Some((rest, end)) if !rest[..end].ends_with("(in-place)") => {
            format!(
                "[operation: {} (in-place)]\n{}",
                &rest[..end],
                &rest[end + 2..]
            )
        }
        _ => text,
    }
}

/// Renames the `[operation: …]` header of `text` from `from` to `to`, so the
/// `f64`-coefficient variants share the explanation of their sample-typed
/// counterparts while the rendered call chain names the method that was
/// called.
fn rename_op(text: String, from: &str, to: &str) -> String {
    let old = format!("[operation: {from}]");
    match text.strip_prefix(&old) {
        Some(rest) => format!("[operation: {to}]{rest}"),
        None => text,
    }
}

/// All samples of all channels, as raw `f64` values (no rescaling of
/// integer types), in the buffer's logical order.
fn raw_values<T: StandardSample>(a: &AudioSamples<'static, T>) -> Vec<f64> {
    match a.data() {
        AudioData::Mono(arr) => arr.iter().map(|&x| -> f64 { x.cast_into() }).collect(),
        AudioData::Multi(arr) => arr.iter().map(|&x| -> f64 { x.cast_into() }).collect(),
    }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// Population standard deviation (the convention of `AudioStatistics::std_dev`).
fn std_pop(v: &[f64]) -> f64 {
    let m = mean(v);
    if v.is_empty() {
        0.0
    } else {
        (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt()
    }
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_unstable_by(f64::total_cmp);
    let n = s.len();
    if n == 0 {
        0.0
    } else if n.is_multiple_of(2) {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    } else {
        s[n / 2]
    }
}

fn rms(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
    }
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

/// Tolerance for "these two snapshots are related exactly": a small relative
/// term for floating-point rounding plus one unit when the data are integers
/// (integer sample types round or truncate every result).
fn tolerance(x: &[f64], y: &[f64]) -> f64 {
    let scale = max_abs(x).max(max_abs(y)).max(1e-12);
    let integral = x.iter().chain(y).all(|v| v.fract() == 0.0) && scale > 1.0;
    1e-5 * scale + if integral { 1.0 } else { 0.0 }
}

/// Least-squares fit `y ≈ a + b·x`, returning `(a, b, largest residual)`.
fn fit_affine(x: &[f64], y: &[f64]) -> Option<(f64, f64, f64)> {
    if x.len() != y.len() || x.is_empty() {
        return None;
    }
    let (mx, my) = (mean(x), mean(y));
    let sxx: f64 = x.iter().map(|v| (v - mx) * (v - mx)).sum();
    if sxx <= 0.0 {
        return None;
    }
    let sxy: f64 = x.iter().zip(y).map(|(u, v)| (u - mx) * (v - my)).sum();
    let b = sxy / sxx;
    let a = my - b * mx;
    let resid = x
        .iter()
        .zip(y)
        .fold(0.0f64, |m, (u, v)| m.max((v - (a + b * u)).abs()));
    Some((a, b, resid))
}

/// One line comparing the RMS level before and after, in raw sample units.
fn rms_line(before: &[f64], after: &[f64]) -> String {
    let (rb, ra) = (rms(before), rms(after));
    if rb > 0.0 && ra > 0.0 {
        format!(
            "RMS level:  {rb:.4}  \u{2192}  {ra:.4}  ({:+.2} dB)",
            20.0 * (ra / rb).log10()
        )
    } else {
        format!("RMS level:  {rb:.4}  \u{2192}  {ra:.4}")
    }
}

/// Standard continuous μ-law curve on the normalised scale.
fn mu_law_standard(x: f64, mu: f64) -> f64 {
    x.signum() * mu.mul_add(x.abs(), 1.0).ln() / (1.0 + mu).ln()
}

/// Probes this build's `mu_compress` / `mu_expand` with μ = 255 on a few
/// known inputs and returns `(compress(0), compress(0.5), max deviation of
/// compress from the standard curve, expand(compress(0.5)))`.
fn mu_law_probe() -> Option<(f64, f64, f64, f64)> {
    let xs = [0.0f64, 0.001, 0.1, 0.5, 1.0];
    let probe = AudioSamples::new_mono(
        ndarray::Array1::from_vec(xs.to_vec()),
        crate::sample_rate!(8000),
    )
    .ok()?;
    let compressed = probe.mu_compress(255.0).ok()?;
    let c = raw_values(&compressed);
    let dev = xs.iter().zip(&c).fold(0.0f64, |m, (&x, &y)| {
        m.max((y - mu_law_standard(x, 255.0)).abs())
    });
    let expanded = compressed.mu_expand(255.0).ok()?;
    let e = raw_values(&expanded);
    Some((c[0], c[3], dev, e[3]))
}

// ─── Normalisation: infer the method from the two snapshots ─────────────────

enum NormKind {
    Unchanged,
    Peak {
        gain: f64,
        peak_after: f64,
    },
    Mean {
        c: f64,
    },
    Median {
        c: f64,
    },
    Offset {
        c: f64,
    },
    ZScore {
        mu: f64,
        sigma: f64,
    },
    MinMax {
        lo: f64,
        hi: f64,
        x_min: f64,
        x_max: f64,
    },
    Unknown,
}

fn infer_normalisation(x: &[f64], y: &[f64]) -> NormKind {
    let tol = tolerance(x, y);
    let Some((a, b, resid)) = fit_affine(x, y) else {
        return NormKind::Unknown;
    };
    if resid > tol {
        return NormKind::Unknown;
    }
    if (b - 1.0).abs() <= 1e-4 {
        if a.abs() <= tol {
            return NormKind::Unchanged;
        }
        let c = -a;
        if (c - mean(x)).abs() <= tol {
            return NormKind::Mean { c };
        }
        if (c - median(x)).abs() <= tol {
            return NormKind::Median { c };
        }
        return NormKind::Offset { c };
    }
    let (sigma, mu) = (std_pop(x), mean(x));
    if sigma > 0.0
        && mean(y).abs() <= tol
        && (std_pop(y) - 1.0).abs() <= 1e-3
        && (b * sigma - 1.0).abs() <= 1e-3
    {
        return NormKind::ZScore { mu, sigma };
    }
    if b > 0.0 && a.abs() <= tol {
        return NormKind::Peak {
            gain: b,
            peak_after: max_abs(y),
        };
    }
    if b > 0.0 {
        let lo = y.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let x_min = x.iter().copied().fold(f64::INFINITY, f64::min);
        let x_max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        return NormKind::MinMax {
            lo,
            hi,
            x_min,
            x_max,
        };
    }
    NormKind::Unknown
}

const NORM_INFERENCE_NOTE: &str = "The explanation only sees the signal before and after the call, \
     not the `NormalizationConfig`, so the method above was inferred from how the samples changed. \
     All five `NormalizationMethod`s (Peak, MinMax, Mean, Median, ZScore) are affine maps \
     x\u{2032} = a + b\u{00b7}x applied to every sample of every channel with one pair (a, b); \
     Mean, Median and ZScore work on raw sample values, Peak on the normalised [\u{2212}1, 1] scale.";

impl<T: StandardSample> AudioProcessingExplainText for AudioSamples<'static, T> {
    fn explain_text_normalize(before: &Self, after: &Self) -> String {
        let x = raw_values(before);
        let y = raw_values(after);
        let (latex, body) = match infer_normalisation(&x, &y) {
            NormKind::Peak { gain, peak_after } => (
                r"x'[n] = A \cdot \frac{x[n]}{\max_m |x[m]|}",
                format!(
                    "Peak normalisation (`NormalizationMethod::Peak`): every sample is multiplied by \
                     the same gain A / max|x|, so the largest absolute sample becomes the target peak A \
                     and all amplitude ratios are preserved (the waveform shape does not change). For \
                     integer sample types the gain is applied on the normalised [\u{2212}1, 1] scale and \
                     the result is rounded. Peak normalisation says nothing about loudness: two signals \
                     with the same peak can differ in RMS level by many dB.\n\
                     \n\
                     Inferred gain:  {gain:.6}  ({:+.2} dB)     |     Peak after (A):  {peak_after:.4}",
                    20.0 * gain.log10()
                ),
            ),
            NormKind::MinMax {
                lo,
                hi,
                x_min,
                x_max,
            } => (
                r"x'[n] = a + (b - a) \frac{x[n] - x_{min}}{x_{max} - x_{min}}",
                format!(
                    "Min-max normalisation (`NormalizationMethod::MinMax`): an affine map that sends the \
                     smallest sample x_min to a and the largest x_max to b, with every other sample placed \
                     proportionally in between. Unlike peak normalisation it also shifts the signal, so a \
                     zero-mean waveform is generally no longer zero-mean afterwards.\n\
                     \n\
                     Input range [{x_min:.4}, {x_max:.4}]  \u{2192}  output range [a, b] = [{lo:.4}, {hi:.4}]"
                ),
            ),
            NormKind::Mean { c } => (
                r"x'[n] = x[n] - \bar{x}",
                format!(
                    "Mean normalisation (`NormalizationMethod::Mean`): the arithmetic mean of all samples \
                     (all channels together) is subtracted, re-centring the signal on zero. The amplitude \
                     scale is unchanged.\n\
                     \n\
                     Mean removed:  {c:.6}"
                ),
            ),
            NormKind::Median { c } => (
                r"x'[n] = x[n] - \mathrm{med}(x)",
                format!(
                    "Median normalisation (`NormalizationMethod::Median`, mono only): the median of the \
                     sample values (the middle of the sorted values; the mean of the two middle values \
                     for an even count) is subtracted. The median is less affected than the mean by a few \
                     large outliers. The amplitude scale is unchanged.\n\
                     \n\
                     Median removed:  {c:.6}"
                ),
            ),
            NormKind::Offset { c } => (
                r"x'[n] = x[n] - c",
                format!(
                    "The samples were shifted by a constant offset c without any change of scale, as \
                     done by the Mean and Median methods; c does not match either statistic of the input \
                     exactly (for integer sample types the result is rounded).\n\
                     \n\
                     Offset removed:  {c:.6}"
                ),
            ),
            NormKind::ZScore { mu, sigma } => (
                r"x'[n] = \frac{x[n] - \bar{x}}{\sigma_x}",
                format!(
                    "Z-score normalisation (`NormalizationMethod::ZScore`): subtract the mean x\u{0304} and \
                     divide by the population standard deviation \u{03c3}\u{2093} (both over all samples of \
                     all channels), giving zero mean and unit standard deviation. Sample values are then \
                     no longer bounded by [\u{2212}1, 1]: peaks of 3\u{2013}4 are normal, so the result is \
                     a feature for analysis rather than playable audio.\n\
                     \n\
                     Mean x\u{0304}:  {mu:.6}     |     \u{03c3}\u{2093}:  {sigma:.6}"
                ),
            ),
            NormKind::Unchanged => (
                r"x'[n] = a + b \, x[n]",
                "The samples did not change: the signal already satisfied the requested normalisation \
                 (for example its peak already equalled the target, or its mean was already zero)."
                    .to_string(),
            ),
            NormKind::Unknown => (
                r"x'[n] = a + b \, x[n]",
                "Normalisation maps every sample through one affine function x\u{2032} = a + b\u{00b7}x \
                 whose coefficients depend on the method: Peak (a = 0, b = A / max|x|), MinMax (input \
                 range onto [a, b]), Mean or Median (b = 1, a = \u{2212}mean or \u{2212}median) and \
                 ZScore (a = \u{2212}mean/\u{03c3}, b = 1/\u{03c3}). The method used here could not be \
                 identified from the samples alone (for example because integer rounding hid the exact \
                 relation)."
                    .to_string(),
            ),
        };
        let peak_before = before.peak();
        let peak_after = after.peak();
        format!(
            "{}{body}\n\n{NORM_INFERENCE_NOTE}\n\nPeak amplitude:  {peak_before:.4}  \u{2192}  {peak_after:.4}",
            header("Normalize", latex)
        )
    }

    fn explain_text_scale(before: &Self, after: &Self) -> String {
        let latex = r"x'[n] = g \cdot x[n]";
        let x = raw_values(before);
        let y = raw_values(after);
        let sxx: f64 = x.iter().map(|v| v * v).sum();
        let gain_line = if sxx > 0.0 {
            let g = x.iter().zip(&y).map(|(u, v)| u * v).sum::<f64>() / sxx;
            let resid = x
                .iter()
                .zip(&y)
                .fold(0.0f64, |m, (u, v)| m.max((v - g * u).abs()));
            if resid <= tolerance(&x, &y) && g != 0.0 {
                format!(
                    "Inferred gain g:  {g:.6}  ({:+.2} dB{})\n",
                    20.0 * g.abs().log10(),
                    if g < 0.0 { ", polarity inverted" } else { "" }
                )
            } else {
                "Gain g:  not recoverable exactly from the samples (integer rounding or saturation)\n"
                    .to_string()
            }
        } else {
            String::new()
        };
        let peak_before = before.peak();
        let peak_after = after.peak();
        format!(
            "{}Linear amplitude scaling multiplies every sample of every channel by the same constant g. \
             The level changes by 20\u{00b7}log\u{2081}\u{2080}|g| dB: |g| < 1 attenuates, |g| > 1 \
             amplifies and a negative g also inverts the polarity. Float results are not clipped, so \
             |g| > 1 can push samples beyond \u{00b1}1; for integer sample types the gain is applied on \
             the normalised [\u{2212}1, 1] scale, then the result is rounded to the nearest integer and \
             saturated at the type's range. Unlike peak normalisation, g is chosen explicitly rather \
             than derived from the signal, which makes it a repeatable gain change.\n\
             \n\
             {gain_line}Peak amplitude:  {peak_before:.4}  \u{2192}  {peak_after:.4}",
            header("Scale", latex)
        )
    }

    fn explain_text_apply_window(before: &Self, after: &Self) -> String {
        let latex = r"x'[n] = w[n] \, x[n], \quad n = 0, \dots, N - 1";
        let n = before.samples_per_channel().get();
        let x = raw_values(before);
        let y = raw_values(after);
        format!(
            "{}Windowing multiplies sample n by the coefficient w[n]. The window has exactly N \
             coefficients, one per sample, and the same window is applied to every channel; the whole \
             buffer is treated as one frame. A tapering window (Hann, Hamming, Blackman, ...) brings \
             the signal smoothly towards zero at the frame edges. Without it the DFT sees the abrupt \
             start and end of the frame as part of the signal and spreads energy into far-away bins \
             (spectral leakage). The window shape trades main-lobe width (frequency resolution) \
             against side-lobe level (leakage suppression): Hann is a common compromise, Blackman has \
             lower side lobes and a wider main lobe. Windowing also lowers the level: a Hann window \
             scales a steady sinusoid's DFT peak by its coherent gain of 0.5.\n\
             \n\
             Window length N:  {n} samples     |     {}",
            header("Apply Window", latex),
            rms_line(&x, &y)
        )
    }

    fn explain_text_apply_filter(before: &Self, after: &Self) -> String {
        let latex = r"y[i] = \sum_{j=0}^{M-1} h[j] \, x[i + M - 1 - j]";
        let before_len = before.samples_per_channel().get();
        let after_len = after.samples_per_channel().get();
        let taps = (before_len + 1).saturating_sub(after_len);
        format!(
            "{}FIR (finite impulse response) filtering convolves each channel with the M coefficients \
             h[0], ..., h[M\u{2212}1]: every output sample is a weighted sum of M consecutive input \
             samples. Only the fully overlapping (\"valid\") part of the convolution is returned, so an \
             N-sample input gives N\u{2212}M+1 output samples, the same as numpy.convolve(x, h, \
             mode='valid'). Output index i corresponds to input time i+M\u{2212}1; the result is not \
             delay-compensated. Long filters (160 taps or more, with the `transforms` feature) are \
             computed by FFT convolution with the same result.\n\
             \n\
             FIR filters have no feedback, so they are always stable. Their phase is linear (a constant \
             delay of (M\u{2212}1)/2 samples at every frequency, so no phase distortion) only when the \
             coefficients are symmetric, h[j] = h[M\u{2212}1\u{2212}j], or antisymmetric, \
             h[j] = \u{2212}h[M\u{2212}1\u{2212}j]. An arbitrary set of taps, for example [1, 0.5], is not \
             linear-phase: its group delay varies with frequency.\n\
             \n\
             Filter length M:  {taps}     |     Length:  {before_len}  \u{2192}  {after_len} samples \
             (N \u{2212} M + 1)",
            header("Apply FIR Filter", latex)
        )
    }

    fn explain_text_apply_window_f64(before: &Self, after: &Self) -> String {
        rename_op(
            Self::explain_text_apply_window(before, after),
            "Apply Window",
            "Apply Window (f64)",
        )
    }

    fn explain_text_apply_filter_f64(before: &Self, after: &Self) -> String {
        rename_op(
            Self::explain_text_apply_filter(before, after),
            "Apply FIR Filter",
            "Apply FIR Filter (f64)",
        )
    }

    fn explain_text_mu_compress(before: &Self, after: &Self) -> String {
        let latex = r"y = \mathrm{sgn}(x) \frac{\ln(1 + \mu |x|)}{\ln(1 + \mu)}";
        let caveat = match mu_law_probe() {
            Some((c0, c_half, dev, _)) if dev > 1e-9 => format!(
                "\n\nImplementation note: the formula above is the intended definition, but this \
                 build's `mu_compress` does not match it exactly. With \u{03bc} = 255 it maps 0 to \
                 {c0:.4} (the formula gives 0) and 0.5 to {c_half:.4} (the formula gives {:.4}); the \
                 largest deviation on a probe of [0, 1] is {dev:.4}. This is a known bug that is being \
                 fixed upstream. Until then, do not compare its output numerically with the formula.",
                mu_law_standard(0.5, 255.0)
            ),
            _ => String::new(),
        };
        let peak_before = before.peak();
        let peak_after = after.peak();
        format!(
            "{}\u{03bc}-law companding maps each sample x (on the normalised [\u{2212}1, 1] scale of \
             the sample type) through a logarithmic curve: small amplitudes are boosted and large ones \
             compressed, while 0, +1 and \u{2212}1 are fixed points. Equal ratios of input amplitude \
             then become roughly equal steps of output, so a uniform quantiser applied afterwards gives \
             roughly constant signal-to-noise ratio across levels. \u{03bc} sets the strength of the \
             curve; \u{03bc} = 255 is the North American and Japanese telephony value.\n\
             \n\
             ITU-T G.711 \u{03bc}-law codes 14-bit linear PCM into 8-bit codes using an 8-segment \
             piecewise-linear approximation of this curve (the European A-law variant codes 13-bit \
             PCM). This operation applies only the continuous curve: it does not quantise to 8 bits \
             and does not use the segmented approximation.{caveat}\n\
             \n\
             Peak amplitude:  {peak_before:.4}  \u{2192}  {peak_after:.4}",
            header("\u{03bc}-law Compress", latex)
        )
    }

    fn explain_text_mu_expand(before: &Self, after: &Self) -> String {
        let latex = r"x = \mathrm{sgn}(y) \frac{(1 + \mu)^{|y|} - 1}{\mu}";
        let caveat = match mu_law_probe() {
            Some((_, _, dev, round_trip)) if dev > 1e-9 => format!(
                "\n\nImplementation note: `mu_expand` implements this formula, but this build's \
                 `mu_compress` deviates from the standard compression curve (a known bug being fixed \
                 upstream), so expanding a signal compressed by this library does not give back the \
                 original: with \u{03bc} = 255, 0.5 comes back as {round_trip:.4}."
            ),
            _ => String::new(),
        };
        let peak_before = before.peak();
        let peak_after = after.peak();
        format!(
            "{}\u{03bc}-law expansion applies the inverse of the standard continuous \u{03bc}-law \
             compression curve, on the normalised [\u{2212}1, 1] scale: it maps a companded value y \
             back to a linear amplitude x. The \u{03bc} used here must equal the \u{03bc} used for \
             compression; a mismatch bends every level by a different amount. Together the compressor \
             (at the sender) and the expander (at the receiver) form a compander.{caveat}\n\
             \n\
             Peak amplitude:  {peak_before:.4}  \u{2192}  {peak_after:.4}",
            header("\u{03bc}-law Expand", latex)
        )
    }

    fn explain_text_low_pass_filter(before: &Self, after: &Self) -> String {
        let latex = r"|H(f)|^2 = \frac{1}{1 + \left(\frac{\Omega}{\Omega_c}\right)^{2}}, \quad \Omega = \tan\frac{\pi f}{f_s}";
        let x = raw_values(before);
        let y = raw_values(after);
        format!(
            "{}`low_pass_filter` is a first-order Butterworth low-pass designed with the bilinear \
             transform (identical to scipy.signal.butter(1, f_c, fs=f_s)), not an ideal brick wall: \
             an ideal filter that passes everything below f_c and nothing above is not realisable. \
             Here f_c is the cutoff, f_s the sample rate and \u{03a9}_c = tan(\u{03c0}f_c/f_s). The gain is \
             1 at 0 Hz, exactly \u{2212}3 dB (|H|\u{00b2} = 1/2) at f_c and 0 at the Nyquist frequency \
             f_s/2; well below Nyquist the response falls by about 6 dB per octave above f_c. With \
             p = (1 \u{2212} \u{03a9}_c)/(1 + \u{03a9}_c) the filter is \
             y[n] = p\u{00b7}y[n\u{2212}1] + (1\u{2212}p)/2\u{00b7}(x[n] + x[n\u{2212}1]), run on each channel \
             from zero initial state. It is causal, so it also delays and phase-shifts the signal.\n\
             \n\
             A single pole gives only a gentle slope, so this filter smooths: it reduces hiss and other \
             high-frequency content and, applied to a rectified signal, extracts the slowly varying \
             amplitude envelope. It is far too gentle to serve as an anti-aliasing filter before \
             downsampling; use `AudioIirFiltering::butterworth_lowpass` with a higher order, or a \
             sinc resampler, for that.\n\
             \n\
             {}",
            header("Low-Pass Filter", latex),
            rms_line(&x, &y)
        )
    }

    fn explain_text_high_pass_filter(before: &Self, after: &Self) -> String {
        let latex = r"|H(f)|^2 = \frac{1}{1 + \left(\frac{\Omega_c}{\Omega}\right)^{2}}, \quad \Omega = \tan\frac{\pi f}{f_s}";
        let x = raw_values(before);
        let y = raw_values(after);
        format!(
            "{}`high_pass_filter` is a first-order Butterworth high-pass designed with the bilinear \
             transform (identical to scipy.signal.butter(1, f_c, 'highpass', fs=f_s)), not an ideal \
             brick wall. With \u{03a9}_c = tan(\u{03c0}f_c/f_s) the gain is 0 at 0 Hz (a zero at z = 1), \
             exactly \u{2212}3 dB at the cutoff f_c and 1 at the Nyquist frequency; below f_c it falls \
             by about 6 dB per octave. With p = (1 \u{2212} \u{03a9}_c)/(1 + \u{03a9}_c) the filter is \
             y[n] = p\u{00b7}y[n\u{2212}1] + (1+p)/2\u{00b7}(x[n] \u{2212} x[n\u{2212}1]), run on each \
             channel from zero initial state.\n\
             \n\
             Typical uses are removing rumble, handling noise and wind below the band of interest, and \
             DC blocking: because of its zero at z = 1 the filter removes a constant offset exactly for \
             any cutoff. The cutoff only sets how fast the offset decays, with a time constant of about \
             1/(2\u{03c0}f_c) (about 16 ms for f_c = 10 Hz), so a low cutoff of 5\u{2013}20 Hz is the usual \
             DC blocker.\n\
             \n\
             {}",
            header("High-Pass Filter", latex),
            rms_line(&x, &y)
        )
    }

    fn explain_text_band_pass_filter(before: &Self, after: &Self) -> String {
        let latex = r"|H(f)|^2 = \frac{1}{1 + \left(\frac{\Omega^2 - \Omega_l \Omega_h}{\Omega (\Omega_h - \Omega_l)}\right)^{2}}, \quad \Omega = \tan\frac{\pi f}{f_s}";
        let x = raw_values(before);
        let y = raw_values(after);
        format!(
            "{}`band_pass_filter` is the band-pass transform of a first-order Butterworth prototype \
             (one second-order section, identical to scipy.signal.butter(1, [f_l, f_h], 'bandpass', \
             fs=f_s)), with \u{03a9}_l = tan(\u{03c0}f_l/f_s) and \u{03a9}_h = tan(\u{03c0}f_h/f_s). The gain is \
             1 at the centre frequency where \u{03a9}\u{00b2} = \u{03a9}_l\u{03a9}_h (the geometric centre of the \
             pre-warped band), exactly \u{2212}3 dB at both edges f_l and f_h, and 0 at 0 Hz and at \
             Nyquist. Outside the band the response falls by only about 6 dB per octave, so a lot of \
             out-of-band energy remains: one octave outside an edge the attenuation is only about 7\u{2013}8 dB. \
             The response is similar to, but not identical with, a first-order high-pass at f_l \
             cascaded with a first-order low-pass at f_h.\n\
             \n\
             It suits gentle band limiting, such as emphasising the speech band before level \
             detection. To isolate a narrow band, use `AudioIirFiltering::butterworth_bandpass` with a \
             higher order.\n\
             \n\
             {}",
            header("Band-Pass Filter", latex),
            rms_line(&x, &y)
        )
    }

    fn explain_text_remove_dc_offset(before: &Self, after: &Self) -> String {
        let latex = r"x'[n] = x[n] - \bar{x}";
        let dc = before.mean();
        let dc_after = after.mean();
        format!(
            "{}A DC offset is a non-zero mean that shifts the whole waveform above or below zero. It \
             is inaudible but wastes headroom (the waveform clips earlier on one side), biases level \
             measurements and later processing, and causes clicks where buffers with different \
             offsets are joined. `remove_dc_offset` subtracts the arithmetic mean x\u{0304} of all \
             samples of all channels (one value for the whole buffer, in raw sample units) from every \
             sample. In the DFT of the buffer only bin 0 (0 Hz) changes. A single mean cannot follow an \
             offset that drifts over time, nor different offsets per channel; a high-pass filter with a \
             low cutoff (`high_pass_filter`) handles those cases.\n\
             \n\
             Mean removed:  {dc:.6}     |     Mean after:  {dc_after:.6}",
            header("Remove DC Offset", latex)
        )
    }

    fn explain_text_clip(before: &Self, after: &Self) -> String {
        clip_text("Clip", r"x'[n] = \min(\max(x[n], a), b)", before, after)
    }

    fn explain_text_normalize_in_place(before: &Self, after: &Self) -> String {
        in_place(Self::explain_text_normalize(before, after))
    }

    fn explain_text_scale_in_place(before: &Self, after: &Self) -> String {
        in_place(Self::explain_text_scale(before, after))
    }

    fn explain_text_apply_window_in_place(before: &Self, after: &Self) -> String {
        in_place(Self::explain_text_apply_window(before, after))
    }

    fn explain_text_apply_filter_in_place(before: &Self, after: &Self) -> String {
        in_place(Self::explain_text_apply_filter(before, after))
    }

    fn explain_text_apply_window_f64_in_place(before: &Self, after: &Self) -> String {
        in_place(Self::explain_text_apply_window_f64(before, after))
    }

    fn explain_text_apply_filter_f64_in_place(before: &Self, after: &Self) -> String {
        in_place(Self::explain_text_apply_filter_f64(before, after))
    }

    fn explain_text_mu_compress_in_place(before: &Self, after: &Self) -> String {
        in_place(Self::explain_text_mu_compress(before, after))
    }

    fn explain_text_mu_expand_in_place(before: &Self, after: &Self) -> String {
        in_place(Self::explain_text_mu_expand(before, after))
    }

    fn explain_text_low_pass_filter_in_place(before: &Self, after: &Self) -> String {
        in_place(Self::explain_text_low_pass_filter(before, after))
    }

    fn explain_text_high_pass_filter_in_place(before: &Self, after: &Self) -> String {
        in_place(Self::explain_text_high_pass_filter(before, after))
    }

    fn explain_text_band_pass_filter_in_place(before: &Self, after: &Self) -> String {
        in_place(Self::explain_text_band_pass_filter(before, after))
    }

    fn explain_text_remove_dc_offset_in_place(before: &Self, after: &Self) -> String {
        in_place(Self::explain_text_remove_dc_offset(before, after))
    }

    fn explain_text_clip_in_place(before: &Self, after: &Self) -> String {
        clip_text(
            "Clip (in-place)",
            r"x[n] \leftarrow \min(\max(x[n], a), b)",
            before,
            after,
        )
    }

    // Resampling lives behind the `resampling` feature; the #[explainable] macro
    // only generates these required methods when that feature is enabled, so the
    // impls are gated to match.
    #[cfg(feature = "resampling")]
    fn explain_text_resample_in_place(before: &Self, after: &Self) -> String {
        in_place(explain_text_resample(before, after))
    }

    #[cfg(feature = "resampling")]
    fn explain_text_resample_by_ratio_in_place(before: &Self, after: &Self) -> String {
        in_place(explain_text_resample(before, after).replacen(
            "[operation: Resample]",
            "[operation: Resample by Ratio]",
            1,
        ))
    }
}

/// Shared text for `clip` and `clip_in_place`.
fn clip_text<T: StandardSample>(
    op: &str,
    latex: &str,
    before: &AudioSamples<'static, T>,
    after: &AudioSamples<'static, T>,
) -> String {
    let x = raw_values(before);
    let y = raw_values(after);
    let changed = x.iter().zip(&y).filter(|(u, v)| u != v).count();
    let total = x.len().max(1);
    let lo = y.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let in_place_note = if op.ends_with("(in-place)") {
        " The in-place form overwrites the buffer instead of allocating a new one; the result is \
         identical."
    } else {
        ""
    };
    format!(
        "{}Hard clipping clamps every sample to the interval [a, b]: samples below a become a, \
         samples above b become b, the rest pass unchanged.{in_place_note} It is a memoryless \
         non-linearity. A symmetrically clipped sine gains odd harmonics whose level rises with the \
         amount of overdrive (not in proportion to the fraction of clipped samples), and the sharp \
         corners spread energy to high frequencies, which can alias above the Nyquist frequency. \
         Tape and valve saturation are imitated with smooth soft-clipping curves (such as tanh), not \
         with a hard clip; hard clipping is mainly a protective last stage that keeps samples inside \
         the representable range.\n\
         \n\
         Samples clamped:  {changed} of {} ({:.2} %)     |     Output range:  [{lo:.4}, {hi:.4}]",
        header(op, latex),
        x.len(),
        100.0 * changed as f64 / total as f64
    )
}

// ─── Free-function explanations for operations outside `AudioProcessing` ────────
//
// Dynamics (compressor / limiter), parametric EQ, and resampling live on their
// own traits (`AudioDynamicRange`, `AudioParametricEq`) or as free functions
// (`resample`), so the `#[explainable]` macro does not generate
// `explain_text_*` methods for them. They follow the keyed-by-op-name pattern
// instead: each returns the same `[operation: …]\n[formula: …]` string the
// `AudioProcessingExplainText` methods produce, and their display names are
// wired into `op_name_to_method` so the rendered call chain resolves them.

/// Explain dynamic-range compression (threshold / ratio / knee / attack / release).
pub(crate) fn explain_text_compressor<T: StandardSample>(
    before: &AudioSamples<'static, T>,
    after: &AudioSamples<'static, T>,
) -> String {
    let latex = r"r = \left(1 - \frac{1}{R}\right) \max(L - T, 0), \quad y[n] = x[n] \cdot 10^{\frac{M - r[n]}{20}}";
    let peak_before = before.peak();
    let peak_after = after.peak();
    format!(
        "{}A compressor reduces dynamic range by turning the signal down when its level is above a \
         threshold. Per channel, an envelope follower turns the detector value d[n] into a smoothed \
         level e[n] (`detection_method`: Peak uses |x[n]|, Rms the RMS over the last 10 ms, Hybrid \
         the larger of the two); L = 20\u{00b7}log\u{2081}\u{2080} e[n] is its level in dBFS. The \
         static gain computer maps L to a gain reduction r in dB: with threshold T (`threshold_db`) \
         and ratio R (`ratio`), every dB above T comes out as only 1/R dB (R = 4: 4 dB in, 1 dB \
         out). The output is the input times 10^((M \u{2212} r)/20), where M is `makeup_gain_db`. \
         Gain is computed from the smoothed level, so it changes smoothly rather than following each \
         waveform cycle.\n\
         \n\
         The formula above is the hard knee. With `KneeType::Soft` and knee width W \
         (`knee_width_db`) the quadratic knee of Giannoulis, Massberg and Reiss (2012) is used: it \
         is centred on T, spans [T \u{2212} W/2, T + W/2] and there gives \
         r = (1 \u{2212} 1/R)\u{00b7}(L \u{2212} T + W/2)\u{00b2}/(2W), so the slope eases from 1 to 1/R \
         and r = (1 \u{2212} 1/R)\u{00b7}W/8 at L = T.\n\
         \n\
         `attack_ms` and `release_ms` are one-pole time constants \u{03c4}: the envelope uses \
         e[n] = a\u{00b7}e[n\u{2212}1] + (1 \u{2212} a)\u{00b7}d[n] with a = exp(\u{2212}1/(\u{03c4}\u{00b7}f_s)), \
         taking the attack coefficient while the detector value rises above the envelope and the \
         release coefficient otherwise. After a step the envelope covers 63 % of the change in \u{03c4}, \
         and its 10\u{2013}90 % rise time is \u{03c4}\u{00b7}ln 9 \u{2248} 2.2\u{03c4}; it never reaches \
         \"full\" reduction in finite time. A short attack catches transients, a long one lets them \
         through; a short release can cause audible pumping.\n\
         \n\
         With `lookahead_ms` > 0 the detector sees the undelayed signal and the gain applied to \
         sample n is the largest reduction over the next N_L = round(lookahead_ms\u{00b7}f_s/1000) \
         samples (a max-hold), so reduction is already in place when a peak arrives. Because the \
         whole buffer is processed at once, the N_L-sample delay is compensated: output and input \
         stay time-aligned.\n\
         \n\
         Peak amplitude:  {peak_before:.4}  \u{2192}  {peak_after:.4}",
        header("Compressor", latex)
    )
}

/// Explain peak limiting (compression with infinite ratio, driven by a smoothed envelope).
pub(crate) fn explain_text_limiter<T: StandardSample>(
    before: &AudioSamples<'static, T>,
    after: &AudioSamples<'static, T>,
) -> String {
    let latex = r"g[n] = \min\left(1, \frac{c}{e[n]}\right), \quad c = 10^{\frac{C}{20}}";
    let peak_before = before.peak();
    let peak_after = after.peak();
    let peak_after_f: f64 = peak_after.convert_to();
    let peak_db = if peak_after_f > 0.0 {
        format!("{:.2} dBFS", 20.0 * peak_after_f.log10())
    } else {
        "\u{2212}\u{221e} dBFS".to_string()
    };
    format!(
        "{}A limiter is a compressor with an infinite ratio: whenever the envelope level e[n] \
         exceeds the ceiling c (`ceiling_db` = C in dBFS, linear c = 10^(C/20)), the gain g[n] is \
         chosen so that the envelope would sit exactly at the ceiling; y[n] = g[n]\u{00b7}x[n] on each \
         channel independently. The formula shows the hard knee; with a soft knee of width W the \
         reduction (L \u{2212} C + W/2)\u{00b2}/(2W) dB starts at C \u{2212} W/2. There is no makeup gain.\n\
         \n\
         e[n] is the same smoothed envelope as in the compressor (one-pole attack and release time \
         constants `attack_ms` and `release_ms`: 63 % of a step after \u{03c4}, 10\u{2013}90 % rise in \
         about 2.2\u{03c4}). Because the gain follows this smoothed level rather than each sample, the \
         ceiling is exact only once the envelope has settled: fast transients and the peaks of \
         low-frequency waveforms can exceed it, so this is not a brick-wall limiter and \
         |y[n]| \u{2264} c is not guaranteed. `lookahead_ms` works as in the compressor (the gain \
         reduction is max-held over the lookahead window, with the delay compensated offline), which \
         reduces but does not remove overshoot. `isp_limiting` is not implemented and currently has \
         no effect.\n\
         \n\
         Peak amplitude:  {peak_before:.4}  \u{2192}  {peak_after:.4}  ({peak_db} after)",
        header("Limiter", latex)
    )
}

/// Explain a single parametric-EQ band (RBJ cookbook biquad).
pub(crate) fn explain_text_eq_band<T: StandardSample>(
    before: &AudioSamples<'static, T>,
    after: &AudioSamples<'static, T>,
) -> String {
    let latex = r"H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{1 + a_1 z^{-1} + a_2 z^{-2}}";
    let x = raw_values(before);
    let y = raw_values(after);
    format!(
        "{}A parametric-EQ band is one second-order (biquad) IIR section with the transfer function \
         above (coefficients normalised so that a\u{2080} = 1), run on every channel from zero initial \
         state. The coefficients come from the RBJ Audio EQ Cookbook formulas, using \
         \u{03c9}\u{2080} = 2\u{03c0}f\u{2080}/f_s for the `frequency` f\u{2080}, A = 10^(G/40) for `gain_db` G and \
         \u{03b1} = sin \u{03c9}\u{2080}/(2Q) for `q_factor` Q. A higher Q gives a narrower band.\n\
         \n\
         The `band_type` selects the shape. `Peak` (bell) changes the level by G dB at f\u{2080} and \
         leaves frequencies far from f\u{2080} unchanged; it corrects resonances or carves space for an \
         instrument. `LowShelf` and `HighShelf` change everything below or above the corner by G dB, \
         reaching G/2 dB at f\u{2080}; there Q sets the steepness of the transition (Q = 1/\u{221a}2 is \
         the steepest without a bump). `LowPass`, `HighPass`, `BandPass` and `BandStop` are \
         second-order RBJ filters that ignore `gain_db`. Like all IIR filters a band also shifts the \
         phase near f\u{2080}. Several bands cascaded build up an arbitrary response.\n\
         \n\
         {}",
        header("EQ Band", latex),
        rms_line(&x, &y)
    )
}

/// Explain sample-rate conversion (resampling).
pub(crate) fn explain_text_resample<T: StandardSample>(
    before: &AudioSamples<'static, T>,
    after: &AudioSamples<'static, T>,
) -> String {
    let latex =
        r"L_{out} = \lceil L_{in} \cdot \rho \rceil, \quad \rho = \frac{f_{s,out}}{f_{s,in}}";
    let before_len = before.samples_per_channel().get();
    let after_len = after.samples_per_channel().get();
    let fs_in = before.sample_rate_hz();
    let fs_out = after.sample_rate_hz();
    format!(
        "{}Resampling converts a signal from one sample rate to another (for example from 44.1 kHz \
         to 48 kHz) by evaluating the band-limited waveform that the samples represent at the new \
         sampling instants. An N-sample input gives \u{2308}N\u{00b7}\u{03c1}\u{2309} output samples, so the \
         duration and pitch are preserved; only the number of samples per second changes.\n\
         \n\
         The new grid can only represent frequencies below its Nyquist frequency f_s,out/2. When \
         downsampling, input content between f_s,out/2 and f_s,in/2 must be removed first, or it \
         folds back (aliases) into the band as spurious tones that cannot be undone. The \
         `ResamplingQuality` tier decides how this is done. Every tier except `Fast` upsampling \
         evaluates a Kaiser-windowed sinc low-pass, centred on the exact input position of each \
         output sample, so it interpolates and anti-aliases in one step: `Fast` (the default) \
         uses a short filter with its pass band to 0.80 of the lower Nyquist frequency and 70 dB \
         of stop-band attenuation, `Medium` 0.90 and 100 dB, `High` 0.913 and 125 dB (close to \
         librosa's default, soxr_hq). `Fast` upsampling is linear interpolation, which dulls high \
         frequencies and leaves spectral images. The filter is symmetric and centred, so the \
         output is time-aligned with the input (no delay) and the tail is flushed; computation \
         is in f64. Use `Medium` or `High` for analysis or listening.\n\
         \n\
         Sample rate:  {fs_in} Hz  \u{2192}  {fs_out} Hz     |     Length:  {before_len}  \u{2192}  \
         {after_len} samples",
        header("Resample", latex)
    )
}
