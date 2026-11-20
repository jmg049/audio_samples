//! Explanations for voice-activity detection and onset detection settings.

use super::doc::{ExplainDoc, Topic, WarningLevel, fmt_num};
use super::{Explain, ExplainContext, note_missing_sample_rate};

#[cfg(feature = "vad")]
mod vad {
    use super::*;
    use crate::operations::types::{VadConfig, VadMethod};

    /// Number of analysis frames, reproducing the VAD frame planner.
    pub(crate) fn vad_frame_count(
        n_samples: usize,
        frame: usize,
        hop: usize,
        pad_end: bool,
    ) -> usize {
        if pad_end {
            n_samples.div_ceil(hop)
        } else if n_samples >= frame {
            (n_samples - frame) / hop + 1
        } else {
            0
        }
    }

    impl Explain for VadConfig {
        fn explain_with(&self, ctx: &ExplainContext) -> ExplainDoc {
            let (f, h) = (self.frame_size.get(), self.hop_size.get());
            let mut doc = ExplainDoc::new(Topic::Vad, "Voice activity detection");
            doc.add_formula(
                r"r = \sqrt{\frac{1}{N_d} \sum_{i=0}^{F'-1} x^2[mH + i]}",
                "Frame RMS; F' is the available frame length and N_d = F when pad_end is set (the missing tail counts as zeros), else F'",
            )
            .add_formula(
                r"z = \frac{\#\{\text{sign changes, zeros skipped}\}}{F - 1}",
                "Zero-crossing rate in crossings per sample; a sinusoid at f gives z ≈ 2f/f_s",
            )
            .add_formula(r"r \ge 10^{\frac{T}{20}}", "Energy decision against the threshold T (dBFS)");
            match self.method {
                VadMethod::ZeroCrossing => {
                    doc.add_formula(
                        r"z_{min} \le z \le z_{max}",
                        "Zero-crossing decision (energy is ignored)",
                    );
                }
                VadMethod::Combined => {
                    doc.add_formula(
                        r"r \ge 10^{\frac{T}{20}} \ \wedge \ z_{min} \le z \le z_{max}",
                        "Combined decision: both must hold",
                    );
                }
                _ => {}
            }
            doc.add_formula(
                r"\tilde{v}[m] = \left[ 2 \sum_{j=m-S+1}^{m} v[j] \ge \min(m+1, S) \right]",
                "Causal majority vote over the last S frames (ties count as speech); then hangover, minimum-speech and minimum-silence run rules",
            );
            doc.add_symbol_value("F", "frame size", f.to_string(), "samples")
                .add_symbol_value("H", "hop size", h.to_string(), "samples")
                .add_symbol_value(
                    "T",
                    "energy threshold",
                    fmt_num(self.energy_threshold_db),
                    "dBFS",
                )
                .add_symbol_value(
                    r"z_{min}, z_{max}",
                    "ZCR bounds",
                    format!("{}, {}", fmt_num(self.zcr_min), fmt_num(self.zcr_max)),
                    "crossings/sample",
                )
                .add_symbol_value(
                    "S",
                    "smoothing window",
                    self.smooth_frames.get().to_string(),
                    "frames",
                );
            let thr = 10f64.powf(self.energy_threshold_db / 20.0);
            doc.add_quantity("Threshold RMS", r"10^{T/20}", thr, "", r"10^{T/20}")
                .add_quantity(
                    "Overlap",
                    r"1 - H/F",
                    100.0 * (1.0 - h as f64 / f as f64),
                    "%",
                    r"100(1 - H/F)",
                );
            let fs = ctx.valid_sample_rate();
            if let Some(fs) = fs {
                let ms = |frames: f64| 1000.0 * frames * h as f64 / fs;
                doc.add_quantity(
                    "Frame duration",
                    r"F / f_s",
                    1000.0 * f as f64 / fs,
                    "ms",
                    r"1000 F / f_s",
                )
                .add_quantity(
                    "Hop duration",
                    r"H / f_s",
                    1000.0 * h as f64 / fs,
                    "ms",
                    r"1000 H / f_s",
                )
                .add_quantity(
                    "Hangover",
                    r"H_o H / f_s",
                    ms(self.hangover_frames.get() as f64),
                    "ms",
                    r"\text{hangover} \cdot H / f_s",
                )
                .add_quantity(
                    "Shortest speech region kept",
                    r"\text{min}_s H / f_s",
                    ms(self.min_speech_frames as f64),
                    "ms",
                    r"\text{min\_speech} \cdot H / f_s",
                )
                .add_quantity(
                    "Shortest silence gap kept",
                    r"\text{min}_g H / f_s",
                    ms(self.min_silence_frames as f64),
                    "ms",
                    r"\text{min\_silence} \cdot H / f_s",
                )
                .add_quantity(
                    "ZCR lower bound as a tone",
                    r"z_{min} f_s / 2",
                    self.zcr_min * fs / 2.0,
                    "Hz",
                    r"z_{min} f_s / 2",
                )
                .add_quantity(
                    "ZCR upper bound as a tone",
                    r"z_{max} f_s / 2",
                    self.zcr_max * fs / 2.0,
                    "Hz",
                    r"z_{max} f_s / 2",
                );
                if self.method == VadMethod::Spectral {
                    doc.add_quantity(
                        "Spectral resolution",
                        r"f_s / F",
                        fs / f as f64,
                        "Hz",
                        r"f_s / F",
                    );
                    if self.speech_band_high_hz > fs / 2.0 {
                        doc.add_warning(
                            WarningLevel::Problem,
                            format!(
                                "The speech band ends at {} Hz, above the Nyquist frequency {} Hz, so voice_activity_mask returns an error.",
                                fmt_num(self.speech_band_high_hz),
                                fmt_num(fs / 2.0)
                            ),
                            "Lower speech_band_high_hz.",
                        );
                    }
                }
                let frame_ms = 1000.0 * f as f64 / fs;
                if frame_ms > 50.0 {
                    doc.add_warning(
                        WarningLevel::Info,
                        format!("Frames are {} ms long; speech VAD usually uses 10 to 30 ms frames, so short pauses will be missed.", fmt_num(frame_ms)),
                        format!("At {} Hz a 25 ms frame is {} samples.", fmt_num(fs), (0.025 * fs).round()),
                    );
                }
            } else {
                note_missing_sample_rate(&mut doc);
            }
            if let Some(n) = ctx.n_samples {
                let frames = vad_frame_count(n, f, h, self.pad_end);
                doc.add_quantity(
                    "Frames",
                    "M",
                    frames as f64,
                    "",
                    if self.pad_end {
                        r"\lceil L / H \rceil"
                    } else {
                        r"\lfloor (L - F)/H \rfloor + 1"
                    },
                );
                if frames == 0 {
                    doc.add_warning(
                        WarningLevel::Problem,
                        format!("The signal ({n} samples) is shorter than one frame and pad_end is false, so there is no frame to classify."),
                        "Set pad_end = true or use a shorter frame.",
                    );
                }
            }
            doc.set_summary(format!(
                "Each {f}-sample frame (hop {h}) is classified as speech by {}, then the decisions are smoothed by a {}-frame majority vote, \
                 extended by {} hangover frame(s), and runs shorter than {} speech / {} silence frames are removed.",
                match self.method {
                    VadMethod::Energy => format!("its RMS level (≥ {} dBFS)", fmt_num(self.energy_threshold_db)),
                    VadMethod::ZeroCrossing => "its zero-crossing rate".to_string(),
                    VadMethod::Combined => "both RMS level and zero-crossing rate".to_string(),
                    VadMethod::Spectral => format!(
                        "its RMS level (≥ {} dBFS) and the share of its energy between {} and {} Hz (≥ {})",
                        fmt_num(self.energy_threshold_db),
                        fmt_num(self.speech_band_low_hz),
                        fmt_num(self.speech_band_high_hz),
                        fmt_num(self.spectral_ratio_threshold)
                    ),
                },
                self.smooth_frames.get(),
                self.hangover_frames.get(),
                self.min_speech_frames,
                self.min_silence_frames
            ));
            doc.add_trade_off("A lower threshold detects quieter speech but also classifies more background noise as speech.")
                .add_trade_off("Smoothing, hangover and minimum-run rules remove rapid switching between speech and silence but delay and stretch region boundaries by a few frames.");
            if let Err(e) = self.validate() {
                doc.add_warning(
                    WarningLevel::Problem,
                    format!("Invalid configuration: {e}"),
                    "",
                );
            }
            if self.smooth_frames.get() > 1 {
                doc.add_warning(
                    WarningLevel::Info,
                    format!("The majority vote uses only past frames, so speech onsets are reported about {} frame(s) late.", (self.smooth_frames.get() - 1) / 2),
                    "",
                );
            }
            doc.add_see_also(
                "Mask",
                "audio_samples::AudioVoiceActivityDetection::voice_activity_mask",
            )
            .add_see_also(
                "Mask (Python)",
                "audio_samples.AudioSamples.voice_activity_mask",
            )
            .add_see_also(
                "Regions",
                "audio_samples::AudioVoiceActivityDetection::speech_regions",
            )
            .add_see_also(
                "Regions (Python)",
                "audio_samples.AudioSamples.speech_regions",
            )
            .add_see_also("Python", "audio_samples.VadConfig.explain");
            doc
        }
    }
}

#[cfg(feature = "onset-detection")]
mod onset {
    use super::*;
    use crate::operations::onset::OnsetDetectionConfig;

    impl Explain for OnsetDetectionConfig {
        fn explain_with(&self, ctx: &ExplainContext) -> ExplainDoc {
            let h = self.hop_size.get();
            let cqt = &self.cqt_params;
            let n_bins = cqt.bins_per_octave.get() * cqt.n_octaves.get();
            // Highest bin centre: f_min 2^{(BO-1)/B}, the last term of the bin formula below
            // (the same values as `CqtParams::frequencies`).
            let f_top =
                cqt.f_min * 2f64.powf((n_bins - 1) as f64 / cqt.bins_per_octave.get() as f64);
            let q = cqt.q_factor;
            let mut doc = ExplainDoc::new(Topic::Onsets, "Onset detection (CQT energy flux)");
            doc.add_formula(
                r"N_w = \left\lfloor \frac{4 f_s}{f_{min}} \right\rfloor",
                "Analysis window when window_size is not set: four periods of the lowest CQT frequency",
            )
            .add_formula(
                r"f_b = f_{min} \, 2^{\frac{b}{B}}, \quad b = 0, \dots, B O - 1",
                "Constant-Q bin frequencies (B bins per octave over O octaves)",
            )
            .add_formula(
                r"o[t] = \sum_{b} \max\left(0, |X_b[t]| - |X_b[t-1]|\right), \quad o[0] = 0",
                "Onset strength: half-wave rectified increase of CQT magnitude",
            );
            if self.adaptive_threshold {
                doc.add_formula(
                    r"o[t] \leftarrow 0 \ \text{if} \ o[t] < \mu \cdot \mathrm{median}_{L}(o)[t]",
                    "Adaptive threshold against a running median of length L",
                );
            }
            doc.add_formula(
                r"t_{onset} = \frac{i H}{f_s}",
                "Peak index (in frames) to seconds",
            );
            doc.add_symbol_value("H", "hop size", h.to_string(), "samples")
                .add_symbol_value(r"f_{min}", "lowest CQT frequency", fmt_num(cqt.f_min), "Hz")
                .add_symbol_value(
                    "B",
                    "bins per octave",
                    cqt.bins_per_octave.get().to_string(),
                    "",
                )
                .add_symbol_value("O", "octaves", cqt.n_octaves.get().to_string(), "")
                .add_symbol_value("Q", "CQT quality factor f_b / bandwidth", fmt_num(q), "");
            if self.adaptive_threshold {
                doc.add_symbol_value(
                    r"\mu",
                    "adaptive multiplier",
                    fmt_num(self.adaptive_threshold_multiplier),
                    "",
                )
                .add_symbol_value(
                    "L",
                    "median length",
                    self.median_filter_length.get().to_string(),
                    "frames",
                );
            }
            doc.add_quantity("CQT bins", "B O", n_bins as f64, "", "B O")
                .add_quantity(
                    "Highest CQT frequency",
                    r"f_{min} 2^{(BO-1)/B}",
                    f_top,
                    "Hz",
                    r"f_{min} 2^{(BO-1)/B}",
                );
            let sep = self.peak_picking.min_peak_separation.get();
            match ctx.valid_sample_rate() {
                Some(fs) => {
                    let n_w = self.effective_window_size(fs).get();
                    let sep_s = sep as f64 * h as f64 / fs;
                    doc.add_quantity(
                        "Window size",
                        r"N_w",
                        n_w as f64,
                        "samples",
                        if self.window_size.is_some() {
                            "window_size"
                        } else {
                            r"\lfloor 4 f_s / f_{min} \rfloor"
                        },
                    )
                    .add_quantity(
                        "Window duration",
                        r"N_w / f_s",
                        1000.0 * n_w as f64 / fs,
                        "ms",
                        r"1000 N_w / f_s",
                    )
                    .add_quantity(
                        "Time resolution",
                        r"H / f_s",
                        1000.0 * h as f64 / fs,
                        "ms",
                        r"1000 H / f_s",
                    )
                    .add_quantity(
                        "Minimum peak separation",
                        r"s H / f_s",
                        1000.0 * sep_s,
                        "ms",
                        r"1000 s H / f_s",
                    );
                    // Kernel length per bin is round(Q f_s / f_b), capped at the analysis window
                    // (the `spectrograms` CQT kernels are generated with n_fft = N_w).
                    let ideal_lowest = q * fs / cqt.f_min;
                    doc.add_quantity(
                        "Ideal kernel length at f_min",
                        r"Q f_s / f_{min}",
                        ideal_lowest,
                        "samples",
                        r"Q f_s / f_{min}",
                    );
                    let n_truncated = (0..n_bins)
                        .map(|b| cqt.f_min * 2f64.powf(b as f64 / cqt.bins_per_octave.get() as f64))
                        .filter(|&f_b| f_b < fs / 2.0 && (q * fs / f_b).round() > n_w as f64)
                        .count();
                    doc.add_quantity(
                        "Truncated low bins",
                        r"\mathrm{count}(Q f_s / f_b > N_w)",
                        n_truncated as f64,
                        "bins",
                        r"\text{count of bins whose kernel exceeds } N_w",
                    );
                    if n_truncated > 0 {
                        doc.add_warning(
                            WarningLevel::Info,
                            format!(
                                "The lowest {n_truncated} bin(s) (below about {} Hz) need kernels longer than the {n_w}-sample window (up to {} samples at f_min with Q = {}); their kernels are cut to {n_w} samples, so those bins are wider than constant-Q.",
                                fmt_num(q * fs / n_w as f64),
                                fmt_num(ideal_lowest.round()),
                                fmt_num(q)
                            ),
                            "Set a longer window_size, or a lower Q, if low-frequency resolution matters.",
                        );
                    }
                    if f_top >= fs / 2.0 {
                        doc.add_warning(
                            WarningLevel::Caution,
                            format!("The highest CQT bin centre ({} Hz) is at or above the Nyquist frequency ({} Hz); the CQT drops every bin from the first one at or above Nyquist, so it returns fewer than B O bins.", fmt_num(f_top), fmt_num(fs / 2.0)),
                            "Reduce n_octaves or raise the sample rate.",
                        );
                    }
                    doc.add_warning(
                        WarningLevel::Caution,
                        format!(
                            "peak_picking.min_peak_separation = {sep} counts onset-function frames, i.e. {} ms here (not {} ms, as it would be if it counted audio samples).",
                            fmt_num(1000.0 * sep_s),
                            fmt_num(1000.0 * sep as f64 / fs)
                        ),
                        "Choose min_peak_separation = (desired seconds) × f_s / hop_size.",
                    );
                    doc.set_summary(format!(
                        "A {}-sample ({} ms) window is hopped every {h} samples ({} ms); {n_bins} constant-Q bins (Q = {}) from {} Hz to {} Hz feed a rectified \
                         spectral-increase onset function whose peaks are reported as onset times.",
                        n_w,
                        fmt_num(1000.0 * n_w as f64 / fs),
                        fmt_num(1000.0 * h as f64 / fs),
                        fmt_num(q),
                        fmt_num(cqt.f_min),
                        fmt_num(f_top)
                    ));
                }
                None => {
                    doc.set_summary(format!(
                        "{n_bins} constant-Q bins (Q = {}) from {} Hz to {} Hz feed a rectified spectral-increase onset function computed every {h} samples; its peaks are the onsets.",
                        fmt_num(q),
                        fmt_num(cqt.f_min),
                        fmt_num(f_top)
                    ));
                    note_missing_sample_rate(&mut doc);
                }
            }
            doc.add_trade_off("A smaller hop locates onsets more precisely in time at more computation.")
                .add_trade_off("A long analysis window (needed for low CQT frequencies) spreads fast onsets over the window length.");
            if self.adaptive_threshold && self.median_filter_length.get().is_multiple_of(2) {
                doc.add_warning(
                    WarningLevel::Problem,
                    format!("median_filter_length = {} is even; the median filter rejects it and onset detection returns an error.", self.median_filter_length.get()),
                    "Use an odd length.",
                );
            }
            doc.add_see_also(
                "Detect",
                "audio_samples::AudioOnsetDetection::detect_onsets",
            )
            .add_see_also(
                "Detect (Python)",
                "audio_samples.AudioSamples.detect_onsets",
            )
            .add_see_also(
                "Onset function",
                "audio_samples::AudioOnsetDetection::onset_detection_function",
            )
            .add_see_also(
                "Onset function (Python)",
                "audio_samples.AudioSamples.onset_detection_function",
            )
            .add_see_also(
                "Python",
                "audio_samples.explain.explain(OnsetDetectionConfig)",
            );
            doc
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "vad")]
    #[test]
    fn vad_frame_count_matches_mask_length() {
        use crate::operations::traits::AudioVoiceActivityDetection;
        use crate::operations::types::VadConfig;
        use crate::{sample_rate, sine_wave};
        use std::time::Duration;
        for pad_end in [true, false] {
            let cfg = VadConfig::default().with_pad_end(pad_end);
            let audio =
                sine_wave::<f32>(440.0, Duration::from_millis(250), sample_rate!(16000), 0.5);
            let mask = audio.voice_activity_mask(&cfg).unwrap();
            let doc = cfg.explain_with(&ExplainContext::for_audio(&audio));
            assert_eq!(
                doc.quantity("Frames").unwrap().value as usize,
                mask.len().get(),
                "pad_end={pad_end}"
            );
            assert_eq!(doc.quantity("Frame duration").unwrap().value, 64.0);
            assert!(doc.has_warning_containing("speech VAD usually uses"));
        }
        let short = VadConfig::default().with_pad_end(false);
        assert!(
            short
                .explain_with(&ExplainContext::new().with_n_samples(100))
                .has_warning(WarningLevel::Problem)
        );
        let spectral =
            VadConfig::default().with_method(crate::operations::types::VadMethod::Spectral);
        let doc = spectral.explain_with(&ExplainContext::new().with_sample_rate(16_000.0));
        assert!(!doc.has_warning(WarningLevel::Problem));
        assert_eq!(doc.quantity("Spectral resolution").unwrap().value, 15.625);
        // 8 kHz audio cannot have a band up to 4.5 kHz.
        let mut above = spectral;
        above.speech_band_high_hz = 4_500.0;
        assert!(
            above
                .explain_with(&ExplainContext::new().with_sample_rate(8_000.0))
                .has_warning(WarningLevel::Problem)
        );
    }

    #[cfg(feature = "onset-detection")]
    #[test]
    fn onset_window_and_separation() {
        use crate::operations::onset::OnsetDetectionConfig;
        let c = OnsetDetectionConfig::percussive(crate::sample_rate!(44100));
        let d = c.explain_with(&ExplainContext::new().with_sample_rate(44_100.0));
        assert_eq!(
            d.quantity("Window size").unwrap().value as usize,
            c.effective_window_size(44_100.0).get()
        );
        let sep_ms = d.quantity("Minimum peak separation").unwrap().value;
        let expected = 1000.0 * c.peak_picking.min_peak_separation.get() as f64 * 256.0 / 44_100.0;
        assert!((sep_ms - expected).abs() < 1e-9);
    }

    #[cfg(feature = "onset-detection")]
    #[test]
    fn onset_cqt_top_frequency_q_and_truncation_match_parameters() {
        use crate::operations::onset::OnsetDetectionConfig;
        let c = OnsetDetectionConfig::percussive(crate::sample_rate!(16000));
        let fs = 16_000.0;
        let d = c.explain_with(&ExplainContext::new().with_sample_rate(fs));
        // Top centre is the last element of the spectrograms crate's own bin list.
        let freqs = c.cqt_params.frequencies();
        let last = *freqs.iter().last().unwrap();
        let top = d.quantity("Highest CQT frequency").unwrap().value;
        assert!((top - last).abs() < 1e-9, "{top} vs {last}");
        // 12 bins/oct, 7 octaves from 32.7 Hz: 32.7 * 2^(83/12) = 3950.7 Hz.
        assert!((top - 3950.7).abs() < 0.1, "{top}");
        assert!(d.symbols.iter().any(|s| s.symbol == "Q"
            && s.value.as_deref() == Some(fmt_num(c.cqt_params.q_factor).as_str())));
        // Independent count: bins with round(Q fs / f_b) > N_w.
        let n_w = c.effective_window_size(fs).get() as f64;
        let q = c.cqt_params.q_factor;
        let expected = freqs
            .iter()
            .filter(|&&f| (q * fs / f).round() > n_w)
            .count();
        assert_eq!(expected, 25);
        assert_eq!(
            d.quantity("Truncated low bins").unwrap().value as usize,
            expected
        );
        assert!(d.has_warning_containing("cut to"));
        assert!(d.summary.contains("to 3950.68 Hz"), "{}", d.summary);
    }
}
