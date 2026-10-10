//! Explanations of concepts that have no configuration type of their own.
//!
//! ## What
//!
//! Each function takes the handful of numbers that define a concept (a sample
//! rate, a bit depth, an FFT size, a set of FIR taps, …) and returns an
//! [`ExplainDoc`] with the formulas, the numbers they produce for those
//! inputs, and warnings.
//!
//! ## Why
//!
//! Many syllabus topics (sampling, aliasing, quantisation, the DFT, pitch
//! estimation, linear prediction) are not configured through a struct in this
//! crate, but students still benefit from seeing the formula evaluated for
//! their numbers.
//!
//! ## How
//!
//! Call the function and print or render the result. Invalid inputs (a
//! negative sample rate, zero bits) produce a [`WarningLevel::Problem`]
//! warning instead of a panic or an error.
//!
//! ```
//! use audio_samples::educational::topics;
//!
//! let doc = topics::quantisation(16);
//! let sqnr = doc.quantity("SQNR (full-scale sine)").unwrap().value;
//! assert!((sqnr - 98.09).abs() < 0.01);
//! println!("{}", doc.to_markdown());
//! ```

use super::doc::{ExplainDoc, Table, Topic, WarningLevel, fmt_num};
use super::numeric::{dtft_magnitude, mag_db};

/// Returns `true` (and records a problem) if `fs` is not a usable rate.
fn reject_rate(doc: &mut ExplainDoc, fs: f64) -> bool {
    if fs.is_finite() && fs > 0.0 {
        false
    } else {
        doc.add_warning(
            WarningLevel::Problem,
            format!("The sample rate must be a positive, finite number of hertz (got {fs})."),
            "",
        );
        true
    }
}

/// Folds `f` (Hz) into the baseband `[0, fs/2]`.
fn fold(f: f64, fs: f64) -> f64 {
    (f - fs * (f / fs).round()).abs()
}

/// Explains uniform sampling at `sample_rate_hz`: the sampling period, the
/// Nyquist frequency and the sampling theorem.
///
/// # Arguments
///
/// * `sample_rate_hz` - sampling rate in Hz.
///
/// # Returns
///
/// An [`ExplainDoc`] with the sampling period and Nyquist frequency. A
/// non-positive or non-finite rate yields a [`WarningLevel::Problem`].
///
/// # Examples
///
/// ```
/// use audio_samples::educational::topics;
///
/// let doc = topics::sampling(16_000.0);
/// assert_eq!(doc.quantity("Nyquist frequency").unwrap().value, 8_000.0);
/// assert_eq!(doc.quantity("Sampling period").unwrap().value, 0.0625); // ms
/// ```
#[must_use]
pub fn sampling(sample_rate_hz: f64) -> ExplainDoc {
    let fs = sample_rate_hz;
    let mut doc = ExplainDoc::new(Topic::Sampling, "Sampling");
    doc.add_formula(
        r"x[n] = x_a(nT), \quad T = \frac{1}{f_s}",
        "A sample is the analogue signal read every T seconds",
    )
    .add_formula(
        r"f_N = \frac{f_s}{2}",
        "Nyquist frequency: the highest representable frequency",
    )
    .add_formula(
        r"x_a(t) = \sum_{n} x[n] \, \mathrm{sinc}\left(\frac{t - nT}{T}\right)",
        "Ideal (Whittaker-Shannon) reconstruction, exact when x_a has no energy at or above f_N",
    );
    doc.add_see_also("Aliasing", "audio_samples::educational::topics::aliasing")
        .add_see_also("Aliasing (Python)", "audio_samples.explain.aliasing")
        .add_see_also(
            "Sample rate of a signal",
            "audio_samples::AudioSamples::sample_rate",
        )
        .add_see_also(
            "Sample rate of a signal (Python)",
            "audio_samples.AudioSamples.sample_rate",
        )
        .add_see_also("Python", "audio_samples.explain.sampling");
    if reject_rate(&mut doc, fs) {
        doc.set_summary(
            "Sampling turns a continuous signal into a sequence of numbers taken at a fixed rate.",
        );
        return doc;
    }
    let t_ms = 1000.0 / fs;
    let nyq = fs / 2.0;
    doc.set_summary(format!(
        "Sampling at {} Hz takes one sample every {} ms. Frequencies below {} Hz are represented \
         unambiguously; anything above folds back (aliases) into that range.",
        fmt_num(fs),
        fmt_num(t_ms),
        fmt_num(nyq)
    ));
    doc.add_symbol_value("f_s", "sample rate", fmt_num(fs), "Hz")
        .add_symbol_value("T", "sampling period", fmt_num(t_ms), "ms")
        .add_symbol_value("f_N", "Nyquist frequency", fmt_num(nyq), "Hz");
    doc.add_quantity("Sampling period", "T", t_ms, "ms", r"T = 1000 / f_s")
        .add_quantity("Nyquist frequency", "f_N", nyq, "Hz", r"f_N = f_s / 2")
        .add_quantity(
            "Samples per millisecond",
            r"f_s / 1000",
            fs / 1000.0,
            "",
            r"f_s / 1000",
        );
    doc.add_trade_off(
        "A higher sample rate represents more bandwidth but costs proportionally more storage and computation.",
    )
    .add_trade_off(
        "Real converters need an analogue anti-aliasing filter whose transition band lies below f_N, so usable bandwidth is somewhat less than f_s/2.",
    );
    if nyq < 20_000.0 {
        doc.add_warning(
            WarningLevel::Info,
            format!(
                "The Nyquist frequency ({} Hz) is below 20 kHz, the nominal upper limit of hearing; content between {} Hz and 20 kHz cannot be represented.",
                fmt_num(nyq),
                fmt_num(nyq)
            ),
            "Fine for speech; use 44.1 kHz or 48 kHz for full-band music.",
        );
    }
    doc
}

/// Explains where a tone of `signal_hz` appears after sampling at
/// `sample_rate_hz`, computing the folded (alias) frequency.
///
/// # Arguments
///
/// * `signal_hz` - frequency of the analogue tone in Hz (non-negative).
/// * `sample_rate_hz` - sampling rate in Hz.
///
/// # Returns
///
/// An [`ExplainDoc`] whose `"Alias frequency"` quantity is
/// `|f - k f_s|` with `k = round(f / f_s)`. A
/// [`WarningLevel::Caution`] is raised when the tone is above the Nyquist
/// frequency.
///
/// # Examples
///
/// ```
/// use audio_samples::educational::{WarningLevel, topics};
///
/// let doc = topics::aliasing(7_000.0, 8_000.0);
/// assert_eq!(doc.quantity("Alias frequency").unwrap().value, 1_000.0);
/// assert!(doc.has_warning(WarningLevel::Caution));
/// ```
#[must_use]
pub fn aliasing(signal_hz: f64, sample_rate_hz: f64) -> ExplainDoc {
    let (f, fs) = (signal_hz, sample_rate_hz);
    let mut doc = ExplainDoc::new(Topic::Sampling, "Aliasing");
    doc.add_formula(
        r"f_a = \left| f - k f_s \right|, \quad k = \mathrm{round}\left(\frac{f}{f_s}\right)",
        "Folded (alias) frequency: where the tone appears after sampling",
    )
    .add_formula(
        r"\cos(2\pi f n T) = \cos(2\pi (f + m f_s) n T), \quad m \in \mathbb{Z}",
        "Tones that differ by a multiple of f_s give identical samples",
    );
    doc.add_see_also("Sampling", "audio_samples::educational::topics::sampling")
        .add_see_also("Sampling (Python)", "audio_samples.explain.sampling")
        .add_see_also(
            "Resampling",
            "audio_samples::educational::topics::resampling",
        )
        .add_see_also("Resampling (Python)", "audio_samples.explain.resampling");
    if reject_rate(&mut doc, fs) {
        return doc;
    }
    if !(f.is_finite() && f >= 0.0) {
        doc.add_warning(
            WarningLevel::Problem,
            format!("The tone frequency must be a non-negative, finite number of hertz (got {f})."),
            "",
        );
        return doc;
    }
    let nyq = fs / 2.0;
    let k = (f / fs).round();
    let fa = fold(f, fs);
    doc.set_summary(format!(
        "A tone at {} Hz, sampled at {} Hz, produces the same samples as a {} Hz tone. \
         Only frequencies up to the Nyquist frequency ({} Hz) are represented faithfully.",
        fmt_num(f),
        fmt_num(fs),
        fmt_num(fa),
        fmt_num(nyq)
    ));
    doc.add_symbol_value("f", "tone frequency", fmt_num(f), "Hz")
        .add_symbol_value("f_s", "sample rate", fmt_num(fs), "Hz")
        .add_symbol_value("k", "nearest multiple of f_s", fmt_num(k), "");
    doc.add_quantity("Nyquist frequency", "f_N", nyq, "Hz", r"f_s / 2")
        .add_quantity(
            "Normalised frequency",
            r"f / f_s",
            f / fs,
            "cycles/sample",
            r"f / f_s",
        )
        .add_quantity(
            "Alias frequency",
            "f_a",
            fa,
            "Hz",
            r"\left| f - k f_s \right|",
        );

    let mut rows = Vec::new();
    for m in 0..3 {
        let base = f64::from(m) * fs;
        for cand in [base + fa, base + fs - fa] {
            if rows.len() < 6 && !rows.iter().any(|r: &Vec<String>| r[0] == fmt_num(cand)) {
                rows.push(vec![fmt_num(cand), fmt_num(fold(cand, fs))]);
            }
        }
    }
    doc.add_table(Table {
        caption: "Analogue frequencies that produce identical samples".into(),
        headers: vec!["Analogue tone (Hz)".into(), "Appears at (Hz)".into()],
        rows,
    });
    doc.add_trade_off("Aliasing cannot be undone after sampling: the alias and a genuine tone at f_a are indistinguishable.");

    if f > nyq {
        doc.add_warning(
            WarningLevel::Caution,
            format!(
                "{} Hz is above the Nyquist frequency ({} Hz) and will alias to {} Hz.",
                fmt_num(f),
                fmt_num(nyq),
                fmt_num(fa)
            ),
            format!(
                "Low-pass filter below {} Hz before sampling or downsampling, or sample above {} Hz.",
                fmt_num(nyq),
                fmt_num(2.0 * f)
            ),
        );
    } else if f == nyq {
        doc.add_warning(
            WarningLevel::Info,
            "The tone is exactly at the Nyquist frequency: its samples alternate in sign and their amplitude depends on the phase (a sine gives all zeros).",
            "",
        );
    }
    doc
}

/// Explains uniform quantisation to `bits` bits using this crate's
/// integer/float convention.
///
/// The crate maps integers to floats asymmetrically: negative codes are
/// divided by 2^(b-1) and non-negative codes by 2^(b-1) - 1, so that both -1.0
/// and +1.0 are reachable and the round trip is exact. The step size is
/// therefore 2^-(b-1) below zero and 1/(2^(b-1) - 1) above; the textbook
/// SQNR formula uses the former.
///
/// # Arguments
///
/// * `bits` - word length in bits (1 to 64).
///
/// # Returns
///
/// An [`ExplainDoc`] with the number of levels, step sizes, noise power and
/// the SQNR of a full-scale sine (≈ 6.02 b + 1.76 dB).
///
/// # Examples
///
/// ```
/// use audio_samples::educational::topics;
///
/// let doc = topics::quantisation(16);
/// assert_eq!(doc.quantity("Quantisation levels").unwrap().value, 65_536.0);
/// assert!((doc.quantity("SQNR (full-scale sine)").unwrap().value - 98.09).abs() < 0.01);
/// ```
#[must_use]
pub fn quantisation(bits: u32) -> ExplainDoc {
    let mut doc = ExplainDoc::new(Topic::Quantisation, "Quantisation");
    doc.add_formula(
        r"q = \begin{cases} \mathrm{round}(x \cdot 2^{b-1}) & x < 0 \\ \mathrm{round}(x \cdot (2^{b-1} - 1)) & x \geq 0 \end{cases}",
        "Float to integer as implemented (x clamped to [-1, 1], round half away from zero)",
    )
    .add_formula(r"\Delta = \frac{2}{2^{b}} = 2^{1-b}", "Step size over the full scale [-1, 1)")
    .add_formula(r"\sigma_e^2 = \frac{\Delta^2}{12}", "Noise power of rounding error uniformly distributed in one step")
    .add_formula(
        r"\mathrm{SQNR} = 10 \log_{10} \frac{6}{\Delta^2} \approx 6.02\,b + 1.76 \ \mathrm{dB}",
        "Signal-to-quantisation-noise ratio of a full-scale sine: signal power 1/2 over noise power Δ²/12",
    );
    doc.add_see_also(
        "Type conversion",
        "audio_samples::AudioTypeConversion::to_format",
    )
    .add_see_also(
        "Type conversion (Python)",
        "audio_samples.AudioSamples.to_format",
    )
    .add_see_also("Python", "audio_samples.explain.quantisation");
    if bits == 0 || bits > 64 {
        doc.add_warning(
            WarningLevel::Problem,
            format!("The word length must be between 1 and 64 bits (got {bits})."),
            "",
        );
        return doc;
    }
    let b = f64::from(bits);
    let levels = 2f64.powf(b);
    let half = 2f64.powf(b - 1.0);
    let step = 2.0 / levels;
    let step_pos = if half > 1.0 {
        1.0 / (half - 1.0)
    } else {
        f64::INFINITY
    };
    let noise_power = step * step / 12.0;
    let sqnr = 10.0 * (0.5 / noise_power).log10();
    let rule = 20.0 * 2f64.log10() * b + 10.0 * 1.5f64.log10();
    let dither_penalty = 10.0 * 3f64.log10();

    doc.set_summary(format!(
        "{bits}-bit quantisation maps each sample to one of {} levels, {} apart on the [-1, 1] scale. \
         For a full-scale sine the rounding noise is {} dB below the signal.",
        fmt_num(levels),
        fmt_num(step),
        fmt_num(sqnr)
    ));
    doc.add_symbol_value("b", "word length", bits.to_string(), "bits")
        .add_symbol_value(r"\Delta", "quantisation step", fmt_num(step), "")
        .add_symbol("x", "sample value on the normalised [-1, 1] scale")
        .add_symbol("q", "integer code");
    doc.add_quantity("Quantisation levels", "L", levels, "", r"2^{b}")
        .add_quantity("Step size (negative half)", r"\Delta", step, "", r"2^{1-b}")
        .add_quantity(
            "Step size (positive half)",
            r"\Delta_+",
            step_pos,
            "",
            r"1 / (2^{b-1} - 1)",
        )
        .add_quantity(
            "Noise power",
            r"\sigma_e^2",
            noise_power,
            "",
            r"\Delta^2 / 12",
        )
        .add_quantity(
            "Noise level",
            r"10 \log_{10} \sigma_e^2",
            10.0 * noise_power.log10(),
            "dB re 1 (power of a full-scale DC level)",
            r"10 \log_{10}(\Delta^2/12)",
        )
        .add_quantity(
            "Noise level (AES17 dBFS)",
            r"10 \log_{10} (2 \sigma_e^2)",
            10.0 * (noise_power / 0.5).log10(),
            "dBFS (AES17: full-scale sine = 0 dBFS)",
            r"10 \log_{10}(\Delta^2/12) + 10 \log_{10} 2",
        )
        .add_quantity(
            "SQNR (full-scale sine)",
            "SQNR",
            sqnr,
            "dB",
            r"10 \log_{10}((1/2)/(\Delta^2/12))",
        )
        .add_quantity(
            "Rule of thumb",
            r"20 \log_{10}(2) \, b + 10 \log_{10}(1.5)",
            rule,
            "dB",
            r"\text{exact form of } 6.02\,b + 1.76 \text{ (the rounded constants give 0.01 dB less at 16 bits)}",
        )
        .add_quantity(
            "SQNR with TPDF dither",
            r"\mathrm{SQNR}_d",
            sqnr - dither_penalty,
            "dB",
            r"\mathrm{SQNR} - 10 \log_{10} 3",
        );
    doc.add_trade_off("Each extra bit halves the step size and raises the SQNR by about 6.02 dB.")
        .add_trade_off(
            "Triangular (TPDF) dither of ±1 step adds noise of power Δ²/6, raising total noise to Δ²/4 (about 4.77 dB worse) but making the error independent of the signal, which removes distortion on quiet signals.",
        )
        .add_trade_off(
            "This crate's float-to-integer conversion rounds without dither; add dither yourself before requantising quiet material.",
        )
        .add_trade_off("The SQNR formula assumes a full-scale signal; a signal peaking 20 dB below full scale loses 20 dB of SQNR.");
    let crate_type = match bits {
        8 => Some("u8 (offset binary, centred on 128)"),
        16 => Some("i16"),
        24 => Some("I24"),
        32 => Some("i32"),
        _ => None,
    };
    if let Some(t) = crate_type {
        doc.add_warning(
            WarningLevel::Info,
            format!("{bits} bits is the word length of this crate's {t} sample type."),
            "",
        );
    }
    doc
}

/// Explains the discrete Fourier transform of length `n_fft` at
/// `sample_rate_hz`: bin spacing, bin frequencies, the Nyquist bin and the
/// symmetry of real-input spectra.
///
/// # Arguments
///
/// * `n_fft` - transform length in samples (at least 1).
/// * `sample_rate_hz` - sampling rate in Hz.
///
/// # Returns
///
/// An [`ExplainDoc`] with the bin spacing, the number of non-redundant bins
/// and a table of selected bin frequencies.
///
/// # Examples
///
/// ```
/// use audio_samples::educational::topics;
///
/// let doc = topics::dft(1024, 48_000.0);
/// assert_eq!(doc.quantity("Bin spacing").unwrap().value, 46.875);
/// assert_eq!(doc.quantity("Non-redundant bins").unwrap().value, 513.0);
/// ```
#[must_use]
pub fn dft(n_fft: usize, sample_rate_hz: f64) -> ExplainDoc {
    let mut doc = ExplainDoc::new(Topic::Dft, "Discrete Fourier transform (DFT)");
    doc.add_formula(
        r"X[k] = \sum_{n=0}^{N-1} x[n] \, e^{-j \frac{2\pi}{N} k n}",
        "DFT of one block of N samples",
    )
    .add_formula(
        r"f_k = \frac{k f_s}{N}, \quad \Delta f = \frac{f_s}{N}",
        "Frequency of bin k and bin spacing",
    )
    .add_formula(
        r"X[N-k] = \overline{X[k]}",
        "Real input: the upper half mirrors the lower half (complex conjugate)",
    );
    doc.add_see_also("FFT of a signal", "audio_samples::AudioTransforms::fft")
        .add_see_also("FFT of a signal (Python)", "audio_samples.AudioSamples.fft")
        .add_see_also("STFT", "spectrograms::StftParams (explain it)");
    if n_fft == 0 {
        doc.add_warning(
            WarningLevel::Problem,
            "The DFT length must be at least 1.",
            "",
        );
        return doc;
    }
    if reject_rate(&mut doc, sample_rate_hz) {
        return doc;
    }
    let (n, fs) = (n_fft as f64, sample_rate_hz);
    let df = fs / n;
    let unique = n_fft / 2 + 1;
    doc.set_summary(format!(
        "A {n_fft}-point DFT at {} Hz analyses {} ms of signal into bins {} Hz apart. \
         For real input only bins 0..={} are independent; the rest are mirror images.",
        fmt_num(fs),
        fmt_num(1000.0 * n / fs),
        fmt_num(df),
        unique - 1
    ));
    doc.add_symbol_value("N", "DFT length", n_fft.to_string(), "samples")
        .add_symbol_value("f_s", "sample rate", fmt_num(fs), "Hz")
        .add_symbol("k", "bin index")
        .add_symbol_value(r"\Delta f", "bin spacing", fmt_num(df), "Hz");
    doc.add_quantity("Bin spacing", r"\Delta f", df, "Hz", r"f_s / N")
        .add_quantity(
            "Block duration",
            r"N / f_s",
            1000.0 * n / fs,
            "ms",
            r"1000 N / f_s",
        )
        .add_quantity(
            "Non-redundant bins",
            r"\lfloor N/2 \rfloor + 1",
            unique as f64,
            "",
            r"\lfloor N/2 \rfloor + 1",
        )
        .add_quantity(
            "Highest bin frequency",
            r"f_{\lfloor N/2 \rfloor}",
            (unique - 1) as f64 * df,
            "Hz",
            r"\lfloor N/2 \rfloor f_s / N",
        );
    let mut bins: Vec<usize> = vec![0, 1, 2, n_fft / 4, n_fft / 2];
    bins.retain(|&k| k < n_fft);
    bins.dedup();
    let rows = bins
        .iter()
        .map(|&k| {
            let note = if k == 0 {
                "DC (mean value), always real"
            } else if 2 * k == n_fft {
                "Nyquist bin, always real"
            } else {
                ""
            };
            vec![k.to_string(), fmt_num(k as f64 * df), note.to_string()]
        })
        .collect();
    doc.add_table(Table {
        caption: "Selected bins".into(),
        headers: vec!["k".into(), "f_k (Hz)".into(), "Note".into()],
        rows,
    });
    doc.add_trade_off("Finer bin spacing needs a longer block (Δf = f_s/N), which blurs changes in time.")
        .add_trade_off("Zero-padding interpolates the spectrum on a finer grid but does not improve the ability to separate close tones; only a longer signal does.");
    if n_fft % 2 == 1 {
        doc.add_warning(
            WarningLevel::Info,
            format!(
                "N = {n_fft} is odd, so there is no bin exactly at the Nyquist frequency; the highest bin is at {} Hz.",
                fmt_num((unique - 1) as f64 * df)
            ),
            "",
        );
    }
    if !n_fft.is_power_of_two() {
        doc.add_warning(
            WarningLevel::Info,
            format!("N = {n_fft} is not a power of two; mixed-radix FFTs handle it, but prime-heavy sizes are slower."),
            format!("The nearest power of two is {}.", n_fft.next_power_of_two()),
        );
    }
    doc
}

/// Explains a window function of length `n` as generated by the crate's
/// STFT code, with figures measured from its coefficients.
///
/// Reports coherent gain, equivalent noise bandwidth (bins and Hz),
/// scalloping loss, peak side-lobe level and the -3 dB / -6 dB main-lobe
/// widths (measured on a zero-padded FFT), and whether the window is
/// symmetric or periodic.
///
/// # Arguments
///
/// * `window` - the window type.
/// * `n` - window length in samples (at least 1).
/// * `sample_rate_hz` - sample rate used to convert bins to Hz.
///
/// # Returns
///
/// An [`ExplainDoc`] for the window.
///
/// # Examples
///
/// ```
/// use audio_samples::educational::topics;
/// use spectrograms::WindowType;
///
/// let doc = topics::window(&WindowType::Hanning, 1024, 48_000.0);
/// let enbw = doc.quantity("Equivalent noise bandwidth").unwrap().value;
/// assert!((enbw - 1.5).abs() < 0.01);
/// ```
#[cfg(feature = "transforms")]
#[must_use]
pub fn window(window: &spectrograms::WindowType, n: usize, sample_rate_hz: f64) -> ExplainDoc {
    let fs = (sample_rate_hz.is_finite() && sample_rate_hz > 0.0).then_some(sample_rate_hz);
    let mut doc = super::spectral::explain_window(window, n, fs);
    if fs.is_none() {
        reject_rate(&mut doc, sample_rate_hz);
    }
    doc
}

/// Lag range, minimum frame length and pitch quantisation shared by the two
/// time-domain pitch explainers.
fn add_lag_range(
    doc: &mut ExplainDoc,
    min_hz: f64,
    max_hz: f64,
    fs: f64,
    integer_lags: bool,
) -> Option<usize> {
    if !(min_hz.is_finite() && max_hz.is_finite() && min_hz > 0.0 && max_hz > min_hz) {
        doc.add_warning(
            WarningLevel::Problem,
            format!(
                "The frequency range must satisfy 0 < f_min < f_max (got {min_hz} to {max_hz} Hz)."
            ),
            "",
        );
        return None;
    }
    // As implemented: lags are truncated to whole samples.
    let tau_min = (fs / max_hz) as usize;
    let tau_max = (fs / min_hz) as usize;
    let min_len = 2 * (tau_max + 1);
    doc.add_symbol_value(
        r"\tau_{min}",
        "shortest lag searched",
        tau_min.to_string(),
        "samples",
    )
    .add_symbol_value(
        r"\tau_{max}",
        "longest lag searched",
        tau_max.to_string(),
        "samples",
    );
    doc.add_quantity(
        "Shortest lag",
        r"\tau_{min}",
        tau_min as f64,
        "samples",
        r"\lfloor f_s / f_{max} \rfloor",
    )
    .add_quantity(
        "Longest lag",
        r"\tau_{max}",
        tau_max as f64,
        "samples",
        r"\lfloor f_s / f_{min} \rfloor",
    )
    .add_quantity(
        "Minimum frame length",
        r"N_{min}",
        min_len as f64,
        "samples",
        r"2(\tau_{max} + 1)",
    )
    .add_quantity(
        "Minimum frame duration",
        r"N_{min} / f_s",
        1000.0 * min_len as f64 / fs,
        "ms",
        r"1000 N_{min} / f_s",
    );
    if integer_lags && tau_min >= 1 {
        let cents = |tau: f64| 1200.0 * ((tau + 1.0) / tau).log2();
        doc.add_quantity(
            "Pitch step at f_max",
            r"\Delta f_{hi}",
            fs / tau_min as f64 - fs / (tau_min as f64 + 1.0),
            "Hz",
            r"f_s/\tau_{min} - f_s/(\tau_{min}+1)",
        )
        .add_quantity(
            "Pitch step at f_max (cents)",
            r"\Delta c_{hi}",
            cents(tau_min as f64),
            "cents",
            r"1200 \log_2((\tau+1)/\tau)",
        )
        .add_quantity(
            "Pitch step at f_min (cents)",
            r"\Delta c_{lo}",
            cents(tau_max as f64),
            "cents",
            r"1200 \log_2((\tau+1)/\tau)",
        );
    }
    if tau_min == 0 {
        doc.add_warning(
            WarningLevel::Problem,
            format!(
                "f_max = {} Hz is above the sample rate, so the shortest lag rounds to 0 samples.",
                fmt_num(max_hz)
            ),
            "Lower f_max below the sample rate (and ideally below f_s/2).",
        );
    }
    Some(min_len)
}

/// Explains the YIN fundamental-frequency estimator as implemented by
/// `audio_samples::operations::pitch_analysis::yin_pitch_detection` (feature
/// `pitch-analysis`).
///
/// # Arguments
///
/// * `threshold` - CMND threshold in `[0, 1]`.
/// * `min_hz` - lowest f0 to search for, in Hz.
/// * `max_hz` - highest f0 to search for, in Hz.
/// * `sample_rate_hz` - sample rate in Hz.
/// * `frame_len` - analysis frame length in samples, if known; enables the
///   "frame too short" check.
///
/// # Returns
///
/// An [`ExplainDoc`] with the lag range and the minimum frame length.
///
/// # Examples
///
/// ```
/// use audio_samples::educational::{WarningLevel, topics};
///
/// // Searching down to 50 Hz at 16 kHz needs frames of at least 642 samples.
/// let doc = topics::yin(0.15, 50.0, 500.0, 16_000.0, Some(1024));
/// assert_eq!(doc.quantity("Minimum frame length").unwrap().value, 642.0);
/// assert!(!doc.has_warning(WarningLevel::Problem));
/// let doc = topics::yin(0.15, 20.0, 500.0, 16_000.0, Some(1024));
/// assert!(doc.has_warning(WarningLevel::Problem));
/// ```
#[must_use]
pub fn yin(
    threshold: f64,
    min_hz: f64,
    max_hz: f64,
    sample_rate_hz: f64,
    frame_len: Option<usize>,
) -> ExplainDoc {
    let fs = sample_rate_hz;
    let mut doc = ExplainDoc::new(Topic::Pitch, "YIN pitch estimation");
    doc.add_formula(
        r"d(\tau) = \sum_{j=0}^{N-1-\tau} \left( x[j] - x[j+\tau] \right)^2",
        "Difference function over all N − τ overlapping pairs, for τ = 1 … τ_max (the paper and librosa use a fixed window W instead, which biases gliding pitch towards the frame's start)",
    )
    .add_formula(
        r"d'(\tau) = \frac{d(\tau)}{\frac{1}{\tau} \sum_{i=1}^{\tau} d(i)}, \quad d'(0) = 1",
        "Cumulative-mean-normalised difference (CMND); the running mean always starts at τ = 1",
    )
    .add_formula(
        r"\tau_0 = \min \{ \tau \ge \tau_{min} : d'(\tau) < \theta \}",
        "Absolute threshold: first lag below θ, then step forward while d' keeps falling (bottom of the dip); no dip means unvoiced (no estimate)",
    )
    .add_formula(
        r"\hat{\tau} = \tau_0 + \frac{1}{2} \frac{d'(\tau_0 - 1) - d'(\tau_0 + 1)}{d'(\tau_0 - 1) - 2 d'(\tau_0) + d'(\tau_0 + 1)}, \quad \hat{f}_0 = \frac{f_s}{\hat{\tau}}",
        "Parabolic interpolation of d' gives a fractional period. This follows librosa; de Cheveigné & Kawahara (2002, JASA 111(4)) step 5 interpolates the raw difference function d(τ) instead, and their step 6 (best local estimate) is not implemented",
    );
    doc.add_symbol("x", "analysis frame (N samples)")
        .add_symbol(r"\tau", "lag in samples")
        .add_symbol_value(r"\theta", "CMND threshold", fmt_num(threshold), "");
    doc.add_see_also(
        "Single estimate",
        "audio_samples::AudioPitchAnalysis::detect_pitch_yin",
    )
    .add_see_also(
        "Single estimate (Python)",
        "audio_samples.AudioSamples.detect_pitch_yin",
    )
    .add_see_also(
        "Low-level routine",
        "audio_samples::operations::pitch_analysis::yin_pitch_detection",
    )
    .add_see_also("Python", "audio_samples.explain.yin");
    if !(0.0..=1.0).contains(&threshold) {
        doc.add_warning(
            WarningLevel::Problem,
            format!("The threshold must lie in [0, 1] (got {threshold}); detect_pitch_yin returns an error."),
            "Typical values are 0.1 to 0.2.",
        );
    }
    if reject_rate(&mut doc, fs) {
        return doc;
    }
    let Some(min_len) = add_lag_range(&mut doc, min_hz, max_hz, fs, false) else {
        return doc;
    };
    doc.set_summary(format!(
        "YIN searches lags {} to {} samples ({} Hz to {} Hz at {} Hz) for the first dip of the normalised \
         difference function below θ = {} and refines it by parabolic interpolation of d' (as librosa does). Frames need at least {min_len} samples.",
        (fs / max_hz) as usize,
        (fs / min_hz) as usize,
        fmt_num(max_hz),
        fmt_num(min_hz),
        fmt_num(fs),
        fmt_num(threshold)
    ));
    doc.add_trade_off("A lower threshold rejects more unvoiced or noisy frames but misses weakly periodic ones.")
        .add_trade_off("Parabolic interpolation refines the period below one sample, so the estimate is not limited to f_s/integer values.")
        .add_trade_off("Extending f_min downwards lengthens the frame the method needs (N ≥ 2(f_s/f_min + 1)), costing time resolution.");
    if let Some(n) = frame_len {
        if n < min_len {
            doc.add_warning(
                WarningLevel::Problem,
                format!(
                    "A {n}-sample frame is shorter than the {min_len} samples needed to search down to {} Hz; the estimator returns no pitch.",
                    fmt_num(min_hz)
                ),
                format!("Use frames of at least {min_len} samples or raise f_min."),
            );
        }
    }
    if threshold > 0.5 && threshold <= 1.0 {
        doc.add_warning(
            WarningLevel::Caution,
            format!(
                "θ = {} is permissive; noise often dips below it, giving spurious estimates.",
                fmt_num(threshold)
            ),
            "Typical values are 0.1 to 0.2.",
        );
    }
    doc
}

/// Explains autocorrelation pitch estimation as implemented by
/// `audio_samples::operations::pitch_analysis::autocorr_pitch_detection`
/// (feature `pitch-analysis`).
///
/// # Arguments
///
/// * `min_hz` - lowest f0 to search for, in Hz.
/// * `max_hz` - highest f0 to search for, in Hz.
/// * `sample_rate_hz` - sample rate in Hz.
///
/// # Returns
///
/// An [`ExplainDoc`] with the lag range, the minimum frame length and the
/// pitch resolution of integer lags.
///
/// # Examples
///
/// ```
/// use audio_samples::educational::topics;
///
/// let doc = topics::autocorrelation_pitch(80.0, 400.0, 16_000.0);
/// assert_eq!(doc.quantity("Longest lag").unwrap().value, 200.0);
/// ```
#[must_use]
pub fn autocorrelation_pitch(min_hz: f64, max_hz: f64, sample_rate_hz: f64) -> ExplainDoc {
    let fs = sample_rate_hz;
    let mut doc = ExplainDoc::new(Topic::Pitch, "Autocorrelation pitch estimation");
    doc.add_formula(
        r"r(\tau) = \sum_{i=0}^{N-1-\tau} x[i] \, x[i+\tau]",
        "Unnormalised autocorrelation over the whole frame",
    )
    .add_formula(
        r"\hat{\tau} = \arg\max_{\tau_{min} \le \tau \le \tau_{max}} r(\tau), \quad \hat{f}_0 = \frac{f_s}{\hat{\tau}}",
        "The lag with the largest positive correlation is taken as the period",
    );
    doc.add_symbol("x", "analysis frame (N samples)")
        .add_symbol(r"\tau", "lag in samples");
    doc.add_see_also(
        "Single estimate",
        "audio_samples::AudioPitchAnalysis::detect_pitch_autocorr",
    )
    .add_see_also(
        "Single estimate (Python)",
        "audio_samples.AudioSamples.detect_pitch_autocorr",
    )
    .add_see_also("YIN", "audio_samples::educational::topics::yin")
    .add_see_also("YIN (Python)", "audio_samples.explain.yin");
    if reject_rate(&mut doc, fs) {
        return doc;
    }
    let Some(min_len) = add_lag_range(&mut doc, min_hz, max_hz, fs, true) else {
        return doc;
    };
    doc.set_summary(format!(
        "The estimator correlates the frame with lagged copies of itself for lags {} to {} samples and reports \
         f_s divided by the best lag. Frames need at least {min_len} samples.",
        (fs / max_hz) as usize,
        (fs / min_hz) as usize
    ));
    doc.add_trade_off("The sum has N-τ terms, so r(τ) is biased towards short lags; this suppresses sub-octave errors but favours harmonics when the fundamental is weak.")
        .add_trade_off("Without normalisation the method is fast but more sensitive to loudness changes and noise than YIN.");
    if max_hz / min_hz >= 2.0 {
        doc.add_warning(
            WarningLevel::Info,
            format!(
                "The search range spans {} octaves, so octave errors (picking 2τ₀ or τ₀/2) are possible.",
                fmt_num((max_hz / min_hz).log2())
            ),
            "Narrow the range when the expected pitch is known.",
        );
    }
    doc
}

/// Explains linear predictive coding (LPC) of order `order` at
/// `sample_rate_hz`, using the conventions of the crate's
/// `audio_samples::lpc_analysis` and `audio_samples::levinson_durbin`
/// (available with the `opus-codec` feature).
///
/// # Arguments
///
/// * `order` - predictor order p (at least 1).
/// * `sample_rate_hz` - sample rate in Hz.
///
/// # Returns
///
/// An [`ExplainDoc`] with the autocorrelation-method and Levinson-Durbin
/// formulas, the rule-of-thumb order f_s/1000 + 2 and the number of
/// resonances the model can represent.
///
/// # Examples
///
/// ```
/// use audio_samples::educational::topics;
///
/// let doc = topics::lpc(18, 16_000.0);
/// assert_eq!(doc.quantity("Rule-of-thumb order").unwrap().value, 18.0);
/// ```
#[must_use]
pub fn lpc(order: usize, sample_rate_hz: f64) -> ExplainDoc {
    let fs = sample_rate_hz;
    let mut doc = ExplainDoc::new(Topic::Lpc, "Linear predictive coding (LPC)");
    doc.add_formula(
        r"e[n] = x[n] + \sum_{k=1}^{p} a_k \, x[n-k], \quad A(z) = 1 + \sum_{k=1}^{p} a_k z^{-k}",
        "Prediction error (residual) and inverse filter; note the plus sign: the crate stores a_k with this convention",
    )
    .add_formula(
        r"R[m] = \sum_{n=0}^{N-1-m} w[n]x[n] \, w[n+m]x[n+m]",
        "Autocorrelation of the Hamming-windowed frame (biased estimate); R[0] is multiplied by 1 + 1e-5 for numerical stability",
    )
    .add_formula(
        r"k_m = -\frac{R[m] + \sum_{i=1}^{m-1} a_i^{(m-1)} R[m-i]}{E_{m-1}}",
        "Levinson-Durbin: reflection coefficient at step m (E_0 = R[0])",
    )
    .add_formula(
        r"a_i^{(m)} = a_i^{(m-1)} + k_m \, a_{m-i}^{(m-1)}, \quad a_m^{(m)} = k_m, \quad E_m = (1 - k_m^2) E_{m-1}",
        "Levinson-Durbin: coefficient and prediction-error updates",
    )
    .add_formula(r"H(z) = \frac{G}{A(z)}", "All-pole model of the vocal tract");
    doc.add_symbol_value("p", "predictor order", order.to_string(), "")
        .add_symbol("a_k", "predictor coefficients")
        .add_symbol(
            "k_m",
            "reflection (PARCOR) coefficients, |k_m| < 1 for a stable model",
        )
        .add_symbol("E_m", "prediction-error energy after m steps");
    doc.add_see_also(
        "Analysis",
        "audio_samples::lpc_analysis (feature opus-codec)",
    )
    .add_see_also(
        "Recursion",
        "audio_samples::levinson_durbin (feature opus-codec)",
    );
    if order == 0 {
        doc.add_warning(
            WarningLevel::Problem,
            "The LPC order must be at least 1.",
            "",
        );
        return doc;
    }
    if reject_rate(&mut doc, fs) {
        return doc;
    }
    let rule = fs / 1000.0 + 2.0;
    let resonances = order / 2;
    doc.set_summary(format!(
        "An order-{order} predictor models each sample as a weighted sum of the previous {order}, which \
         corresponds to an all-pole filter with up to {resonances} resonances (formants). At {} Hz the usual \
         rule of thumb suggests about {} coefficients.",
        fmt_num(fs),
        fmt_num(rule.round())
    ));
    doc.add_quantity(
        "Rule-of-thumb order",
        r"p_{rule}",
        rule,
        "",
        r"f_s / 1000 + 2",
    )
    .add_quantity(
        "Resonances representable",
        r"\lfloor p/2 \rfloor",
        resonances as f64,
        "",
        r"\lfloor p/2 \rfloor",
    )
    .add_quantity(
        "Minimum frame length",
        r"2p",
        2.0 * order as f64,
        "samples",
        r"2p",
    );
    #[cfg(feature = "opus-codec")]
    doc.add_quantity(
        "SILK reference order",
        r"p_{SILK}",
        crate::SILK_LPC_ORDER as f64,
        "",
        r"\mathrm{SILK\_LPC\_ORDER}",
    );
    doc.add_trade_off("The rule of thumb allows one pole pair per kHz of bandwidth (one formant per kHz) plus two poles for the glottal and radiation spectral tilt.")
        .add_trade_off("Too low an order merges formants; too high an order starts modelling individual harmonics of the voice instead of the envelope.");
    if (order as f64) < rule - 2.0 {
        doc.add_warning(
            WarningLevel::Caution,
            format!(
                "p = {order} is well below the rule of thumb ({}); the model has too few poles to capture every formant below {} Hz.",
                fmt_num(rule.round()),
                fmt_num(fs / 2.0)
            ),
            format!("Try p ≈ {}.", fmt_num(rule.round())),
        );
    } else if (order as f64) > 2.0 * rule {
        doc.add_warning(
            WarningLevel::Info,
            format!("p = {order} is more than twice the rule of thumb ({}); the envelope may follow harmonics.", fmt_num(rule.round())),
            "",
        );
    }
    doc.add_warning(
        WarningLevel::Info,
        "lpc_analysis (Rust only; not in the Python bindings) silently lowers the order to half the frame length for very short frames.",
        format!("Use frames of at least {} samples.", 2 * order),
    );
    doc
}

/// Explains the real cepstrum of an `n_fft`-point spectrum at
/// `sample_rate_hz`: quefrency resolution and the f0 range a cepstral peak
/// can reveal.
///
/// The crate has no stand-alone cepstrum function; its MFCCs are a
/// mel-warped cepstrum (DCT-II of log mel energies).
///
/// # Arguments
///
/// * `n_fft` - FFT length in samples (at least 2).
/// * `sample_rate_hz` - sample rate in Hz.
///
/// # Returns
///
/// An [`ExplainDoc`] with the quefrency resolution, the longest usable
/// quefrency and the corresponding lowest detectable f0.
///
/// # Examples
///
/// ```
/// use audio_samples::educational::topics;
///
/// let doc = topics::cepstrum(1024, 16_000.0);
/// assert_eq!(doc.quantity("Lowest detectable f0").unwrap().value, 31.25);
/// ```
#[must_use]
pub fn cepstrum(n_fft: usize, sample_rate_hz: f64) -> ExplainDoc {
    let fs = sample_rate_hz;
    let mut doc = ExplainDoc::new(Topic::Cepstrum, "Real cepstrum");
    doc.add_formula(
        r"c[q] = \frac{1}{N} \sum_{k=0}^{N-1} \log \left| X[k] \right| \, e^{j \frac{2\pi}{N} k q}",
        "Real cepstrum: inverse DFT of the log-magnitude spectrum",
    )
    .add_formula(
        r"\hat{f}_0 = \frac{f_s}{q_{peak}}",
        "A voiced frame shows a peak at the quefrency of its period",
    )
    .add_formula(
        r"\log |X| = \log |E| + \log |H|",
        "Log turns the source-filter product into a sum, separable by quefrency (liftering)",
    );
    doc.add_symbol("q", "quefrency index (samples)")
        .add_symbol("X[k]", "DFT of the windowed frame");
    doc.add_see_also(
        "Mel cepstrum (MFCC)",
        "spectrograms::MfccParams (explain it)",
    )
    .add_see_also("MFCCs of a signal", "audio_samples::AudioTransforms::mfcc")
    .add_see_also(
        "MFCCs of a signal (Python)",
        "audio_samples.AudioSamples.mfcc",
    );
    if n_fft < 2 {
        doc.add_warning(
            WarningLevel::Problem,
            "The FFT length must be at least 2.",
            "",
        );
        return doc;
    }
    if reject_rate(&mut doc, fs) {
        return doc;
    }
    let n = n_fft as f64;
    let q_res = 1000.0 / fs;
    let q_max = 1000.0 * (n / 2.0) / fs;
    let f0_min = 2.0 * fs / n;
    doc.set_summary(format!(
        "With N = {n_fft} at {} Hz the cepstrum resolves quefrency in steps of {} ms up to {} ms, so it can \
         reveal fundamental frequencies above {} Hz.",
        fmt_num(fs),
        fmt_num(q_res),
        fmt_num(q_max),
        fmt_num(f0_min)
    ));
    doc.add_quantity("Quefrency resolution", r"1/f_s", q_res, "ms", r"1000 / f_s")
        .add_quantity(
            "Longest quefrency",
            r"q_{max}",
            q_max,
            "ms",
            r"1000 (N/2) / f_s",
        )
        .add_quantity(
            "Lowest detectable f0",
            r"f_{0,min}",
            f0_min,
            "Hz",
            r"2 f_s / N",
        );
    doc.add_trade_off("A longer frame reaches lower f0 but averages over more time, which blurs pitch changes.")
        .add_trade_off("In practice a few periods are needed in the frame for a clear harmonic structure, so the usable f0 floor is higher than 2f_s/N.");
    doc.add_warning(
        WarningLevel::Info,
        "This crate has no stand-alone cepstrum routine; MFCCs are its cepstral feature.",
        "Compute log|FFT| and an inverse FFT yourself to experiment with the real cepstrum.",
    );
    doc
}

/// Explains sample-rate conversion from `from_hz` to `to_hz`.
///
/// # Arguments
///
/// * `from_hz` - input sample rate in Hz.
/// * `to_hz` - output sample rate in Hz.
///
/// # Returns
///
/// An [`ExplainDoc`] with the conversion ratio, both Nyquist frequencies and
/// (with the `resampling` feature) the anti-aliasing settings of each
/// `ResamplingQuality`.
///
/// # Examples
///
/// ```
/// use audio_samples::educational::{WarningLevel, topics};
///
/// let doc = topics::resampling(48_000.0, 16_000.0);
/// assert!((doc.quantity("Ratio").unwrap().value - 1.0 / 3.0).abs() < 1e-12);
/// assert!(doc.has_warning(WarningLevel::Caution)); // downsampling
/// ```
#[must_use]
pub fn resampling(from_hz: f64, to_hz: f64) -> ExplainDoc {
    let mut doc = ExplainDoc::new(Topic::Resampling, "Resampling");
    doc.add_formula(r"r = \frac{f_{out}}{f_{in}}", "Conversion ratio")
        .add_formula(
            r"f_{c} \le \frac{\min(f_{in}, f_{out})}{2}",
            "Anti-aliasing / anti-imaging cut-off must not exceed the lower Nyquist frequency",
        )
        .add_formula(
            r"y[m] = \sum_{n} x[n] \, h\left(\frac{m}{r} - n\right)",
            "Band-limited interpolation with a windowed-sinc kernel h (every tier except Fast upsampling, which interpolates linearly)",
        );
    doc.add_see_also("Offline", "audio_samples::resample")
        .add_see_also("Offline (Python)", "audio_samples.AudioSamples.resample")
        .add_see_also("Streaming", "audio_samples::StreamingResampler")
        .add_see_also("Aliasing", "audio_samples::educational::topics::aliasing")
        .add_see_also("Aliasing (Python)", "audio_samples.explain.aliasing");
    if reject_rate(&mut doc, from_hz) || reject_rate(&mut doc, to_hz) {
        return doc;
    }
    let r = to_hz / from_hz;
    let (nyq_in, nyq_out) = (from_hz / 2.0, to_hz / 2.0);
    doc.set_summary(format!(
        "Converting {} Hz to {} Hz (ratio {}) {}. The output can represent frequencies up to {} Hz.",
        fmt_num(from_hz),
        fmt_num(to_hz),
        fmt_num(r),
        if r < 1.0 {
            "discards bandwidth, so content above the new Nyquist frequency must be filtered out first"
        } else if r > 1.0 {
            "adds no new information; the band above the old Nyquist frequency stays empty"
        } else {
            "is the identity"
        },
        fmt_num(nyq_out)
    ));
    doc.add_symbol_value(r"f_{in}", "input rate", fmt_num(from_hz), "Hz")
        .add_symbol_value(r"f_{out}", "output rate", fmt_num(to_hz), "Hz");
    doc.add_quantity("Ratio", "r", r, "", r"f_{out} / f_{in}")
        .add_quantity("Input Nyquist", r"f_{in}/2", nyq_in, "Hz", r"f_{in} / 2")
        .add_quantity(
            "Output Nyquist",
            r"f_{out}/2",
            nyq_out,
            "Hz",
            r"f_{out} / 2",
        );

    #[cfg(feature = "resampling")]
    {
        use crate::operations::types::ResamplingQuality;
        let lower_nyq = nyq_in.min(nyq_out);
        // Whole-hertz rates only reach this point when both are integers; the
        // tier choice depends only on the direction of the conversion.
        let (in_sr, out_sr) = (from_hz.round() as u32, to_hz.round() as u32);
        let spec = |q| crate::resampling::quality_filter_spec(q, in_sr, out_sr);
        let describe = |q| match spec(q) {
            Some((pass, att)) => (
                format!("Kaiser-windowed sinc, pass-band to {pass} x Nyquist, {att} dB stop-band"),
                fmt_num(pass * lower_nyq),
            ),
            None => (
                "linear interpolation, no anti-imaging filter".to_string(),
                "none".to_string(),
            ),
        };
        if let Some((pass, _)) = spec(ResamplingQuality::Medium) {
            doc.add_quantity(
                "Sinc pass-band edge",
                r"f_c",
                pass * lower_nyq,
                "Hz",
                r"\text{pass} \cdot \min(f_{in}, f_{out})/2",
            );
        }
        let rows = [
            ("Fast", ResamplingQuality::Fast),
            ("Medium", ResamplingQuality::Medium),
            ("High", ResamplingQuality::High),
        ]
        .into_iter()
        .map(|(name, q)| {
            let (method, edge) = describe(q);
            vec![name.into(), method, edge]
        })
        .collect();
        doc.add_table(Table {
            caption: "Quality presets (ResamplingQuality)".into(),
            headers: vec![
                "Quality".into(),
                "Method".into(),
                "Pass-band edge (Hz)".into(),
            ],
            rows,
        });
    }
    doc.add_trade_off("Longer sinc kernels give a sharper transition band and better stop-band rejection at a higher computational cost per sample.")
        .add_trade_off("The pass-band edge is slightly below the lower Nyquist frequency to leave room for the transition band.");
    if r < 1.0 {
        doc.add_warning(
            WarningLevel::Caution,
            format!(
                "Downsampling: anything between {} Hz and {} Hz must be removed before decimation or it folds back (e.g. {} Hz would appear at {} Hz).",
                fmt_num(nyq_out),
                fmt_num(nyq_in),
                fmt_num(nyq_in),
                fmt_num(fold(nyq_in, to_hz))
            ),
            "Every ResamplingQuality low-passes before decimating (Fast with a short 70 dB filter); naive decimation (keeping every N-th sample) does not and aliases.",
        );
    }
    if !(from_hz.fract() == 0.0 && to_hz.fract() == 0.0) {
        doc.add_warning(
            WarningLevel::Info,
            "Non-integer sample rates cannot be stored in AudioSamples (rates are whole hertz).",
            "",
        );
    }
    doc
}

/// Explains an FIR filter given its taps: linear-phase type, group delay,
/// DC and Nyquist gain, and the magnitude response at a few frequencies.
///
/// The crate applies FIR filters with
/// [`apply_filter`](crate::AudioProcessing::apply_filter), which computes the
/// "valid" part of the convolution: the output is `M - 1` samples shorter
/// than the input and output sample `i` equals the causal filter output at
/// time `i + M - 1`.
///
/// # Arguments
///
/// * `coeffs` - filter taps `h[0..M]`.
/// * `sample_rate_hz` - sample rate in Hz.
///
/// # Returns
///
/// An [`ExplainDoc`] with symmetry classification, group delay and a
/// response table computed by evaluating the DTFT of the taps.
///
/// # Examples
///
/// ```
/// use audio_samples::educational::topics;
///
/// // 5-tap moving average: symmetric, so linear phase with a 2-sample delay.
/// let doc = topics::fir(&[0.2; 5], 48_000.0);
/// assert_eq!(doc.quantity("Group delay").unwrap().value, 2.0);
/// assert!((doc.quantity("DC gain").unwrap().value - 1.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn fir(coeffs: &[f64], sample_rate_hz: f64) -> ExplainDoc {
    let fs = sample_rate_hz;
    let mut doc = ExplainDoc::new(Topic::FirFilter, "FIR filter");
    doc.add_formula(r"y[n] = \sum_{j=0}^{M-1} h[j] \, x[n-j]", "Convolution of the input with the M taps")
        .add_formula(r"H(e^{j\omega}) = \sum_{j=0}^{M-1} h[j] \, e^{-j \omega j}, \quad \omega = \frac{2\pi f}{f_s}", "Frequency response")
        .add_formula(r"h[j] = \pm h[M-1-j] \Rightarrow \tau_g = \frac{M-1}{2}", "Symmetric or antisymmetric taps give linear phase with constant group delay");
    doc.add_see_also("Apply", "audio_samples::AudioProcessing::apply_filter")
        .add_see_also("Apply (Python)", "audio_samples.AudioSamples.apply_filter")
        .add_see_also("Python", "audio_samples.explain.fir");
    if coeffs.is_empty() || coeffs.iter().any(|c| !c.is_finite()) {
        doc.add_warning(
            WarningLevel::Problem,
            "The filter needs at least one finite tap.",
            "",
        );
        return doc;
    }
    if reject_rate(&mut doc, fs) {
        return doc;
    }
    let m = coeffs.len();
    let peak = coeffs.iter().fold(0.0f64, |a, c| a.max(c.abs()));
    let tol = 1e-12 * peak.max(f64::MIN_POSITIVE);
    let symmetric = (0..m / 2).all(|i| (coeffs[i] - coeffs[m - 1 - i]).abs() <= tol);
    let antisymmetric = (0..m.div_ceil(2)).all(|i| (coeffs[i] + coeffs[m - 1 - i]).abs() <= tol);
    let fir_type = match (symmetric, antisymmetric, m % 2 == 1) {
        (true, _, true) => Some(("I", "symmetric, odd length")),
        (true, _, false) => Some((
            "II",
            "symmetric, even length: forced zero at Nyquist, cannot be high-pass",
        )),
        (false, true, true) => Some((
            "III",
            "antisymmetric, odd length: forced zeros at DC and Nyquist",
        )),
        (false, true, false) => Some((
            "IV",
            "antisymmetric, even length: forced zero at DC, cannot be low-pass",
        )),
        _ => None,
    };
    let dc: f64 = coeffs.iter().sum();
    let nyq_gain: f64 = coeffs
        .iter()
        .enumerate()
        .map(|(n, c)| if n % 2 == 0 { *c } else { -*c })
        .sum();
    let delay = (m as f64 - 1.0) / 2.0;

    doc.set_summary(format!(
        "{m} taps at {} Hz. {} apply_filter (Rust AudioProcessing::apply_filter, Python AudioSamples.apply_filter) returns the fully overlapping part of the convolution, \
         {} samples shorter than the input.",
        fmt_num(fs),
        match fir_type {
            Some((t, _)) => format!(
                "The taps are {} (type {t}), so the phase is linear and every frequency is delayed by {} samples ({} ms).",
                if symmetric { "symmetric" } else { "antisymmetric" },
                fmt_num(delay),
                fmt_num(1000.0 * delay / fs)
            ),
            None => "The taps are neither symmetric nor antisymmetric, so the phase is not linear and different frequencies are delayed differently.".to_string(),
        },
        m - 1
    ));
    doc.add_symbol_value("M", "number of taps", m.to_string(), "")
        .add_symbol("h[j]", "filter taps")
        .add_symbol_value("f_s", "sample rate", fmt_num(fs), "Hz");
    doc.add_quantity("Taps", "M", m as f64, "", "M")
        .add_quantity("DC gain", "H(1)", dc, "", r"\sum_j h[j]")
        .add_quantity("Nyquist gain", "H(-1)", nyq_gain, "", r"\sum_j (-1)^j h[j]");
    if dc.abs() > 0.0 {
        doc.add_quantity(
            "DC gain (dB)",
            r"20\log_{10}|H(1)|",
            mag_db(dc.abs()),
            "dB",
            r"20 \log_{10} |\sum_j h[j]|",
        );
    }
    if fir_type.is_some() {
        doc.add_quantity("Group delay", r"\tau_g", delay, "samples", r"(M-1)/2")
            .add_quantity(
                "Group delay (time)",
                r"\tau_g / f_s",
                1000.0 * delay / fs,
                "ms",
                r"1000 (M-1) / (2 f_s)",
            );
    } else if dc.abs() > tol {
        let gd0: f64 = coeffs
            .iter()
            .enumerate()
            .map(|(n, c)| n as f64 * c)
            .sum::<f64>()
            / dc;
        doc.add_quantity(
            "Group delay at DC",
            r"\tau_g(0)",
            gd0,
            "samples",
            r"\sum_j j\,h[j] / \sum_j h[j]",
        );
    }
    let rows = [0.0, 0.125, 0.25, 0.375, 0.5]
        .iter()
        .map(|&frac| {
            let mag = dtft_magnitude(coeffs, frac);
            vec![
                fmt_num(frac * fs),
                fmt_num(mag),
                if mag > 0.0 {
                    fmt_num(mag_db(mag))
                } else {
                    "-inf".to_string()
                },
            ]
        })
        .collect();
    doc.add_table(Table {
        caption: "Measured magnitude response".into(),
        headers: vec!["f (Hz)".into(), "|H|".into(), "|H| (dB)".into()],
        rows,
    });
    doc.add_trade_off("Linear phase preserves waveform shape (all frequencies delayed equally) at the cost of (M-1)/2 samples of latency.")
        .add_trade_off("A sharper transition band needs more taps, which lengthens the delay and the computation.");
    if let Some((t, note)) = fir_type {
        doc.add_warning(
            WarningLevel::Info,
            format!("Linear-phase type {t}: {note}."),
            "",
        );
    }
    #[cfg(feature = "transforms")]
    if m >= crate::operations::processing::FIR_FFT_MIN_TAPS {
        doc.add_warning(
            WarningLevel::Info,
            format!(
                "With {m} taps apply_filter uses FFT convolution (at or above {} taps); the result is the same up to rounding.",
                crate::operations::processing::FIR_FFT_MIN_TAPS
            ),
            "",
        );
    }
    doc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alias_of_7k_at_8k_is_1k() {
        let d = aliasing(7_000.0, 8_000.0);
        assert_eq!(d.quantity("Alias frequency").unwrap().value, 1_000.0);
        assert!(d.has_warning(WarningLevel::Caution));
        // Below Nyquist: no warning, alias equals the tone.
        let d = aliasing(3_000.0, 8_000.0);
        assert_eq!(d.quantity("Alias frequency").unwrap().value, 3_000.0);
        assert!(d.warnings.is_empty());
        // 11 kHz at 8 kHz folds to 3 kHz.
        assert_eq!(
            aliasing(11_000.0, 8_000.0)
                .quantity("Alias frequency")
                .unwrap()
                .value,
            3_000.0
        );
        assert!(aliasing(1.0, -5.0).has_warning(WarningLevel::Problem));
    }

    #[test]
    fn quantisation_matches_closed_forms() {
        let d = quantisation(16);
        let sqnr = d.quantity("SQNR (full-scale sine)").unwrap().value;
        // Independent: 6.0206 b + 1.7609.
        assert!((sqnr - (6.020_599_913 * 16.0 + 1.760_912_591)).abs() < 1e-6);
        assert!((sqnr - 98.09).abs() < 0.01);
        // Noise level: dB re 1 (full-scale DC) and AES17 dBFS (full-scale sine = 0 dBFS).
        let re1 = d.quantity("Noise level").unwrap();
        assert!((re1.value + 101.10).abs() < 0.01, "{}", re1.value);
        assert!(re1.unit.contains("dB re 1") && !re1.unit.contains("dBFS"));
        let aes = d.quantity("Noise level (AES17 dBFS)").unwrap().value;
        assert!((aes + 98.09).abs() < 0.01, "{aes}");
        assert!((aes + sqnr).abs() < 1e-9);
        // The "rule of thumb" line carries the exact expression it evaluates.
        let rule = d.quantity("Rule of thumb").unwrap();
        assert!(rule.symbol.contains(r"20 \log_{10}(2)"));
        assert!((rule.value - (20.0 * 2f64.log10() * 16.0 + 10.0 * 1.5f64.log10())).abs() < 1e-12);
        assert_eq!(
            d.quantity("Step size (negative half)").unwrap().value,
            1.0 / 32768.0
        );
        assert_eq!(
            d.quantity("Step size (positive half)").unwrap().value,
            1.0 / 32767.0
        );
        assert!(
            (d.quantity("Noise power").unwrap().value - (1.0 / 32768.0f64).powi(2) / 12.0).abs()
                < 1e-24
        );
        assert!(quantisation(0).has_warning(WarningLevel::Problem));
    }

    #[test]
    fn crate_conversion_uses_the_documented_steps() {
        use crate::traits::ConvertTo;
        // The explanation's positive and negative step sizes are exactly the
        // scale factors the crate's i16 conversion uses.
        let d = quantisation(16);
        let pos = d.quantity("Step size (positive half)").unwrap().value;
        let neg = d.quantity("Step size (negative half)").unwrap().value;
        let one: i16 = 1;
        let minus_one: i16 = -1;
        let p: f64 = one.convert_to();
        let n: f64 = minus_one.convert_to();
        assert!((p - pos).abs() < 1e-15);
        assert!((n + neg).abs() < 1e-15);
    }

    #[test]
    fn dft_bins() {
        let d = dft(1000, 8000.0);
        assert_eq!(d.quantity("Bin spacing").unwrap().value, 8.0);
        assert_eq!(d.quantity("Non-redundant bins").unwrap().value, 501.0);
        assert!(d.has_warning_containing("not a power of two"));
        let d = dft(7, 7000.0);
        assert!(d.has_warning_containing("is odd"));
        assert_eq!(d.quantity("Highest bin frequency").unwrap().value, 3000.0);
    }

    #[test]
    fn sampling_numbers() {
        let d = sampling(44_100.0);
        assert_eq!(d.quantity("Nyquist frequency").unwrap().value, 22_050.0);
        assert!(d.warnings.is_empty());
        assert!(sampling(16_000.0).has_warning(WarningLevel::Info));
        assert!(sampling(0.0).has_warning(WarningLevel::Problem));
    }

    #[cfg(feature = "pitch-analysis")]
    #[test]
    fn yin_frame_length_check_matches_the_implementation() {
        use crate::operations::pitch_analysis::yin_pitch_detection;
        use non_empty_slice::NonEmptySlice;
        use std::f64::consts::PI;
        let fs = 8000.0;
        let d = yin(0.15, 100.0, 400.0, fs, None);
        let min_len = d.quantity("Minimum frame length").unwrap().value as usize;
        let tau_min = d.quantity("Shortest lag").unwrap().value as usize;
        let tau_max = d.quantity("Longest lag").unwrap().value as usize;
        let tone: Vec<f64> = (0..min_len)
            .map(|i| (2.0 * PI * 200.0 * i as f64 / fs).sin())
            .collect();
        // Exactly the minimum length works; one sample less does not.
        assert!(
            yin_pitch_detection(NonEmptySlice::new(&tone).unwrap(), tau_min, tau_max, 0.15)
                .is_some()
        );
        assert!(
            yin_pitch_detection(
                NonEmptySlice::new(&tone[..min_len - 1]).unwrap(),
                tau_min,
                tau_max,
                0.15
            )
            .is_none()
        );
        assert!(yin(0.15, 100.0, 400.0, fs, Some(min_len - 1)).has_warning(WarningLevel::Problem));
        assert!(!yin(0.15, 100.0, 400.0, fs, Some(min_len)).has_warning(WarningLevel::Problem));
        assert!(yin(1.5, 100.0, 400.0, fs, None).has_warning(WarningLevel::Problem));
    }

    #[test]
    fn lpc_rule_of_thumb() {
        let d = lpc(10, 16_000.0);
        assert_eq!(d.quantity("Rule-of-thumb order").unwrap().value, 18.0);
        assert!(d.has_warning(WarningLevel::Caution));
        assert!(!lpc(18, 16_000.0).has_warning(WarningLevel::Caution));
    }

    #[test]
    fn fir_classification_and_delay() {
        let d = fir(&[1.0, 2.0, 3.0, 2.0, 1.0], 1000.0);
        assert_eq!(d.quantity("Group delay").unwrap().value, 2.0);
        assert!(d.has_warning_containing("type I:"));
        let d = fir(&[1.0, -1.0], 1000.0);
        assert!(d.has_warning_containing("type IV"));
        assert_eq!(d.quantity("DC gain").unwrap().value, 0.0);
        let d = fir(&[1.0, 0.5], 1000.0);
        assert!(d.quantity("Group delay").is_none());
        assert!((d.quantity("Group delay at DC").unwrap().value - 1.0 / 3.0).abs() < 1e-12);
        assert!(fir(&[], 1000.0).has_warning(WarningLevel::Problem));
    }

    #[test]
    fn resampling_ratio_and_warning() {
        let d = resampling(44_100.0, 88_200.0);
        assert_eq!(d.quantity("Ratio").unwrap().value, 2.0);
        assert!(!d.has_warning(WarningLevel::Caution));
        let d = resampling(48_000.0, 16_000.0);
        assert!(d.has_warning(WarningLevel::Caution));
        #[cfg(feature = "resampling")]
        assert!((d.quantity("Sinc pass-band edge").unwrap().value - 0.90 * 8000.0).abs() < 1e-3);
    }

    #[test]
    fn cepstrum_numbers() {
        let d = cepstrum(512, 8000.0);
        assert_eq!(d.quantity("Lowest detectable f0").unwrap().value, 31.25);
        assert_eq!(d.quantity("Quefrency resolution").unwrap().value, 0.125);
    }
}
