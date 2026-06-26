//! `AudioProcessingExplainText` implementation for `AudioSamples<'static, T>`.
//!
//! Each explain_text function returns a string in the format:
//!   [operation: <Name>]\n[formula: <latex>]\n<unicode art>\n\n<prose>
//!
//! The HTML renderer parses the operation name and latex for KaTeX rendering.
//! Terminal output (via Explanation::surface) shows the unicode art + prose directly.

use crate::operations::traits::{AudioProcessingExplainText, AudioStatistics};
use crate::repr::AudioSamples;
use crate::traits::StandardSample;

impl<T: StandardSample> AudioProcessingExplainText for AudioSamples<'static, T> {
    fn explain_text_normalize(before: &Self, after: &Self) -> String {
        let latex = r"x'[n] = \frac{x[n]}{\max_n |x[n]|}";
        let formula = term_maths::render(latex);
        let peak_before = before.peak();
        let peak_after = after.peak();
        format!(
            "[operation: Normalize]\n[formula: {latex}]\n{formula}\n\n\
             Peak normalization divides every sample by the maximum absolute value, \
             scaling the waveform so its loudest moment reaches exactly the target ceiling. \
             All relative amplitude relationships are preserved — the operation is linear and \
             lossless in terms of waveform shape. It is the standard first step before spectral \
             analysis (to eliminate amplitude bias across signals) and before playback mastering \
             (to maximize loudness without clipping).\n\
             \n\
             Peak amplitude:  {peak_before:.4}  →  {peak_after:.4}"
        )
    }

    fn explain_text_scale(before: &Self, after: &Self) -> String {
        let latex = r"x'[n] = \alpha \cdot x[n]";
        let formula = term_maths::render(latex);
        let peak_before = before.peak();
        let peak_after = after.peak();
        format!(
            "[operation: Scale]\n[formula: {latex}]\n{formula}\n\n\
             Linear amplitude scaling multiplies every sample by the constant \u{03b1}, shifting \
             the entire dynamic range uniformly. A factor below 1.0 attenuates the signal; above \
             1.0 amplifies it; a negative factor flips polarity. Unlike peak normalization, the \
             scale factor is chosen explicitly rather than derived from the signal content, making \
             it the correct choice when you need a repeatable, deterministic gain change.\n\
             \n\
             Peak amplitude:  {peak_before:.4}  →  {peak_after:.4}"
        )
    }

    fn explain_text_apply_window(before: &Self, after: &Self) -> String {
        let latex = r"x'[n] = x[n] \cdot w[n]";
        let formula = term_maths::render(latex);
        let n = before.samples_per_channel().get();
        let _ = after;
        format!(
            "[operation: Apply Window]\n[formula: {latex}]\n{formula}\n\n\
             Windowing multiplies each sample by a smoothly tapering coefficient w[n], reducing \
             amplitude toward zero at the frame edges. Without a window, the abrupt boundary \
             transitions of a rectangular frame leak energy across the entire spectrum — an \
             artefact known as spectral leakage — which corrupts FFT magnitude estimates. The \
             choice of window shape trades main-lobe width (frequency resolution) against \
             side-lobe suppression (dynamic range). Common choices: Hanning balances both; \
             Blackman offers high suppression; flat-top is used for accurate amplitude measurement.\n\
             \n\
             Window length:  {n} samples"
        )
    }

    fn explain_text_apply_filter(before: &Self, after: &Self) -> String {
        let latex = r"y[n] = \sum_{k=0}^{K-1} b_k \cdot x[n-k]";
        let formula = term_maths::render(latex);
        let before_len = before.samples_per_channel().get();
        let after_len = after.samples_per_channel().get();
        let order = before_len.saturating_sub(after_len);
        format!(
            "[operation: Apply FIR Filter]\n[formula: {latex}]\n{formula}\n\n\
             FIR (Finite Impulse Response) convolution computes each output sample as the dot \
             product of the filter coefficient vector b with the corresponding window of input \
             samples. FIR filters are unconditionally stable, have a linear phase response \
             (no phase distortion), and their frequency behaviour is fully determined by the \
             coefficient vector — making them straightforward to design and interpret. The \
             output length decreases by the filter order K\u{2212}1 due to boundary handling \
             (the transient region at each end is discarded in valid-convolution mode).\n\
             \n\
             Filter order K\u{2212}1:  {order}     |     Length:  {before_len}  →  {after_len} samples"
        )
    }

    fn explain_text_mu_compress(before: &Self, after: &Self) -> String {
        let latex = r"y = \operatorname{sgn}(x)\cdot\frac{\ln(1 + \mu|x|)}{\ln(1 + \mu)}";
        let formula = term_maths::render(latex);
        let peak_before = before.peak();
        let peak_after = after.peak();
        format!(
            "[operation: \u{03bc}-law Compress]\n[formula: {latex}]\n{formula}\n\n\
             \u{03bc}-law companding applies a logarithmic transfer curve that compresses large \
             amplitudes and expands small ones, reducing the effective dynamic range of the signal. \
             It exploits the logarithmic sensitivity of human hearing: equal perceived loudness \
             steps correspond to multiplicative amplitude steps rather than additive ones. \
             Companding is the compression stage of G.711 telephony encoding, where it allows \
             a 13-bit quality signal to be transmitted over an 8-bit PCM channel. The parameter \
             \u{03bc} (typically 255 for North America/Japan) controls the degree of compression.\n\
             \n\
             Peak amplitude:  {peak_before:.4}  →  {peak_after:.4}"
        )
    }

    fn explain_text_mu_expand(before: &Self, after: &Self) -> String {
        let latex = r"y = \operatorname{sgn}(x)\cdot\frac{(1+\mu)^{|x|} - 1}{\mu}";
        let formula = term_maths::render(latex);
        let peak_before = before.peak();
        let peak_after = after.peak();
        format!(
            "[operation: \u{03bc}-law Expand]\n[formula: {latex}]\n{formula}\n\n\
             \u{03bc}-law expansion is the exact inverse of \u{03bc}-law compression. Applying it \
             to a compressed signal restores the original amplitude ratios, recovering the full \
             dynamic range. In a real telephony system the expander at the receiver is paired \
             precisely with the compressor at the transmitter — any mismatch in the \u{03bc} \
             parameter between the two ends introduces distortion proportional to the error. \
             Together the compress\u{2013}expand pair is called a compander.\n\
             \n\
             Peak amplitude:  {peak_before:.4}  →  {peak_after:.4}"
        )
    }

    fn explain_text_low_pass_filter(before: &Self, after: &Self) -> String {
        let latex = r"H(f) = \begin{cases} 1 & f \le f_c \\ 0 & f > f_c \end{cases}";
        let formula = term_maths::render(latex);
        let _ = (before, after);
        format!(
            "[operation: Low-Pass Filter]\n[formula: {latex}]\n{formula}\n\n\
             A low-pass filter attenuates frequency components above the cutoff f\u{2081}, \
             leaving lower frequencies untouched. In the time domain this is equivalent to \
             smoothing: rapid sample-to-sample fluctuations (high frequencies) are averaged away \
             while the slowly-varying envelope is preserved. Primary uses: removing high-frequency \
             noise, preparing a signal for downsampling (anti-aliasing filter to prevent foldover), \
             and extracting the pitch envelope from a voiced speech signal."
        )
    }

    fn explain_text_high_pass_filter(before: &Self, after: &Self) -> String {
        let latex = r"H(f) = \begin{cases} 0 & f < f_c \\ 1 & f \ge f_c \end{cases}";
        let formula = term_maths::render(latex);
        let _ = (before, after);
        format!(
            "[operation: High-Pass Filter]\n[formula: {latex}]\n{formula}\n\n\
             A high-pass filter attenuates frequency components below the cutoff f\u{2082}, \
             emphasising rapid changes (transients, consonants, attacks) while suppressing the \
             slowly-varying baseline. Primary uses: removing low-frequency rumble and \
             microphone-stand vibration, eliminating DC offset as a first-order degenerate case \
             (f\u{2082} \u{2192} 0), and separating percussive content from the low-frequency \
             sustain of a mix."
        )
    }

    fn explain_text_band_pass_filter(before: &Self, after: &Self) -> String {
        let latex =
            r"H(f) = \begin{cases} 1 & f_l \le f \le f_h \\ 0 & \text{otherwise} \end{cases}";
        let formula = term_maths::render(latex);
        let _ = (before, after);
        format!(
            "[operation: Band-Pass Filter]\n[formula: {latex}]\n{formula}\n\n\
             A band-pass filter retains only the energy in the frequency window [f\u{2097}, f\u{2095}], \
             attenuating both the sub-band below f\u{2097} and the supra-band above f\u{2095}. It is \
             conceptually equivalent to cascading a high-pass at f\u{2097} and a low-pass at f\u{2095}. \
             Primary uses: isolating a specific instrument or voice formant from a complex mix, \
             limiting bandwidth before amplitude modulation (to avoid sideband overlap), and \
             extracting a narrowband carrier for frequency analysis."
        )
    }

    fn explain_text_remove_dc_offset(before: &Self, after: &Self) -> String {
        let latex = r"x'[n] = x[n] - \bar{x}";
        let formula = term_maths::render(latex);
        let dc = before.mean();
        let _ = after;
        format!(
            "[operation: Remove DC Offset]\n[formula: {latex}]\n{formula}\n\n\
             A DC offset is a non-zero signal mean that shifts the entire waveform above or below \
             the zero axis. It is inaudible at audio frequencies but causes several practical \
             problems: it wastes amplitude headroom (the offset is subtracted from the maximum \
             swing before clipping), it biases subsequent filtering and level-detection operations, \
             and it produces a click artefact when buffers with mismatched offsets are concatenated. \
             Subtracting the sample mean re-centres the signal at zero without altering any \
             frequency content above 0 Hz.\n\
             \n\
             Mean removed:  {dc:.6}"
        )
    }

    fn explain_text_clip(before: &Self, after: &Self) -> String {
        let latex = r"x'[n] = \operatorname{clip}(x[n],\, \min,\, \max)";
        let formula = term_maths::render(latex);
        let peak_before = before.peak();
        let peak_after = after.peak();
        format!(
            "[operation: Clip]\n[formula: {latex}]\n{formula}\n\n\
             Hard clipping enforces a ceiling and floor on the waveform: any sample outside \
             [\u{2009}min,\u{2009}max\u{2009}] is clamped to the nearest boundary. The operation \
             is nonlinear — the clipped portions introduce harmonic distortion proportional to \
             the fraction of the signal that exceeds the threshold. Used intentionally to simulate \
             tape saturation or valve amplifier overdrive; used defensively at signal boundaries \
             to prevent digital-to-analogue overflow. The more the signal exceeds the threshold, \
             the richer the harmonic content added.\n\
             \n\
             Peak amplitude:  {peak_before:.4}  →  {peak_after:.4}"
        )
    }

    fn explain_text_normalize_in_place(before: &Self, after: &Self) -> String {
        Self::explain_text_normalize(before, after)
    }

    fn explain_text_scale_in_place(before: &Self, after: &Self) -> String {
        Self::explain_text_scale(before, after)
    }

    fn explain_text_apply_window_in_place(before: &Self, after: &Self) -> String {
        Self::explain_text_apply_window(before, after)
    }

    fn explain_text_apply_filter_in_place(before: &Self, after: &Self) -> String {
        Self::explain_text_apply_filter(before, after)
    }

    fn explain_text_mu_compress_in_place(before: &Self, after: &Self) -> String {
        Self::explain_text_mu_compress(before, after)
    }

    fn explain_text_mu_expand_in_place(before: &Self, after: &Self) -> String {
        Self::explain_text_mu_expand(before, after)
    }

    fn explain_text_low_pass_filter_in_place(before: &Self, after: &Self) -> String {
        Self::explain_text_low_pass_filter(before, after)
    }

    fn explain_text_high_pass_filter_in_place(before: &Self, after: &Self) -> String {
        Self::explain_text_high_pass_filter(before, after)
    }

    fn explain_text_band_pass_filter_in_place(before: &Self, after: &Self) -> String {
        Self::explain_text_band_pass_filter(before, after)
    }

    fn explain_text_remove_dc_offset_in_place(before: &Self, after: &Self) -> String {
        Self::explain_text_remove_dc_offset(before, after)
    }

    fn explain_text_clip_in_place(before: &Self, after: &Self) -> String {
        let latex = r"x[n] \mathrel{{:}{=}} \operatorname{clip}(x[n],\, \min,\, \max)";
        let formula = term_maths::render(latex);
        let peak_before = before.peak();
        let peak_after = after.peak();
        format!(
            "[operation: Clip (in-place)]\n[formula: {latex}]\n{formula}\n\n\
             In-place hard clipping applies the same ceiling\u{2013}floor clamping as the \
             allocating variant but mutates the buffer directly rather than producing a new \
             allocation. The mathematical effect is identical; the distinction is purely in \
             memory layout. Use the in-place form when working with large buffers where \
             avoiding a copy matters, or when the original unclipped samples are no longer needed.\n\
             \n\
             Peak amplitude:  {peak_before:.4}  →  {peak_after:.4}"
        )
    }
}

// ─── Free-function explanations for operations outside `AudioProcessing` ────────
//
// Dynamics (compressor / limiter / gate / expander), parametric EQ, and
// resampling live on their own traits (`AudioDynamicRange`, `AudioParametricEq`)
// or as free functions (`resample`), so the `#[explainable]` macro does not
// generate `explain_text_*` methods for them. They follow the keyed-by-op-name
// pattern instead: each returns the same `[operation: …]\n[formula: …]` string
// the `AudioProcessingExplainText` methods produce, and their display names are
// wired into `op_name_to_method` so the rendered call chain resolves them to a
// concrete method rather than the generic `"operation"` fallback.

/// Explain dynamic-range compression (threshold / ratio / attack / release).
pub(crate) fn explain_text_compressor<T: StandardSample>(
    before: &AudioSamples<'static, T>,
    after: &AudioSamples<'static, T>,
) -> String {
    let latex = r"g(L) = \begin{cases} 0 & L \le T \\ \left(\tfrac{1}{R}-1\right)(L-T) & L > T \end{cases}";
    let formula = term_maths::render(latex);
    let peak_before = before.peak();
    let peak_after = after.peak();
    format!(
        "[operation: Compressor]\n[formula: {latex}]\n{formula}\n\n\
         A compressor reduces dynamic range by attenuating signal above a level threshold \
         (`threshold_db`, in dBFS). For every decibel the input level L rises past the \
         threshold T, the output rises by only 1/R dB, where R is the compression `ratio` \
         (R\u{2009}=\u{2009}4 means 4 dB in produces 1 dB out). The gain-reduction curve g(L) above \
         is applied to a level estimate produced by the `detection_method` (peak or RMS), \
         optionally smoothed by a soft `knee_width_db` around the threshold so the onset of \
         compression is gradual rather than abrupt.\n\
         \n\
         The attack and release time constants govern how fast gain reduction tracks the \
         signal envelope: `attack_ms` is the time to clamp down on a level that has crossed \
         the threshold (a short attack catches transients; a long one lets them through), \
         while `release_ms` is the time to return to unity gain once the level falls back \
         below it (too short causes audible pumping). A non-zero `lookahead_ms` buffers the \
         input so reduction can begin before a peak arrives, and `makeup_gain_db` restores the \
         loudness lost to gain reduction.\n\
         \n\
         Peak amplitude:  {peak_before:.4}  →  {peak_after:.4}"
    )
}

/// Explain peak limiting (a brick-wall ceiling — compression with infinite ratio).
pub(crate) fn explain_text_limiter<T: StandardSample>(
    before: &AudioSamples<'static, T>,
    after: &AudioSamples<'static, T>,
) -> String {
    let latex = r"|y[n]| \le 10^{C/20}, \qquad R \to \infty";
    let formula = term_maths::render(latex);
    let peak_before = before.peak();
    let peak_after = after.peak();
    format!(
        "[operation: Limiter]\n[formula: {latex}]\n{formula}\n\n\
         A limiter is the limiting case of a compressor with an effectively infinite ratio: \
         no output sample is allowed to exceed the ceiling `ceiling_db` (in dBFS), shown above \
         as the linear bound 10^(C/20). Where a compressor gently leans on level above its \
         threshold, a limiter forms a brick wall — any approach to the ceiling is met with \
         exactly enough gain reduction to hold the output at or below it. This makes it the \
         standard final stage of a mastering chain, guaranteeing the signal never clips on the \
         output converter.\n\
         \n\
         A very short `attack_ms` (sub-millisecond) is what enables true peak control, while \
         `release_ms` sets how quickly full gain is restored afterwards. A non-zero \
         `lookahead_ms` lets reduction start before the peak lands, trading latency for cleaner, \
         distortion-free limiting; enabling `isp_limiting` additionally guards against \
         inter-sample peaks that would exceed 0 dBFS only after digital-to-analogue \
         reconstruction.\n\
         \n\
         Peak amplitude:  {peak_before:.4}  →  {peak_after:.4}"
    )
}

/// Explain a single parametric-EQ band (peak / shelf biquad).
pub(crate) fn explain_text_eq_band<T: StandardSample>(
    before: &AudioSamples<'static, T>,
    after: &AudioSamples<'static, T>,
) -> String {
    let latex = r"H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{a_0 + a_1 z^{-1} + a_2 z^{-2}}";
    let formula = term_maths::render(latex);
    let _ = (before, after);
    format!(
        "[operation: EQ Band]\n[formula: {latex}]\n{formula}\n\n\
         A parametric-EQ band is a second-order (biquad) IIR filter whose transfer function \
         H(z) is shown above. Its five coefficients are derived from three musically meaningful \
         parameters via the RBJ cookbook formulas: the centre `frequency` (in Hz) sets where the \
         band acts, `gain_db` sets how much it boosts (positive) or cuts (negative) there, and \
         the `q_factor` sets how wide the affected region is — a high Q is narrow and surgical, \
         a low Q is broad and gentle.\n\
         \n\
         The `band_type` selects the response shape. A `Peak` (bell) band boosts or cuts a region \
         centred on the frequency, leaving everything either side untouched — ideal for \
         correcting a resonance or carving space for an instrument. A `LowShelf` or `HighShelf` \
         instead lifts or lowers everything below or above its corner frequency, like a tilt \
         control; here Q sets the slope of the transition. Because each band is a single biquad, \
         several can be cascaded to build an arbitrary frequency response one band at a time."
    )
}

/// Explain sample-rate conversion (resampling).
pub(crate) fn explain_text_resample<T: StandardSample>(
    before: &AudioSamples<'static, T>,
    after: &AudioSamples<'static, T>,
) -> String {
    let latex = r"L_{out} = L_{in} \cdot \frac{f_{s,\,out}}{f_{s,\,in}}";
    let formula = term_maths::render(latex);
    let before_len = before.samples_per_channel().get();
    let after_len = after.samples_per_channel().get();
    format!(
        "[operation: Resample]\n[formula: {latex}]\n{formula}\n\n\
         Resampling converts a signal from one sample rate to another — for example from a \
         44.1 kHz CD master to the 48 kHz used in video — by reconstructing the underlying \
         continuous waveform and re-sampling it on the new time grid. The output length scales \
         by the conversion ratio f_s,out / f_s,in, so the perceived pitch and duration are \
         preserved; only the number of samples representing each second changes.\n\
         \n\
         The hard part is avoiding aliasing. When downsampling, any energy above the new Nyquist \
         frequency (half the target rate) must be removed first with an anti-aliasing low-pass \
         filter, or it folds back into the audible band as spurious tones that cannot be undone. \
         When upsampling, an interpolation filter reconstructs the new samples without \
         introducing imaging artefacts. High-quality resampling matters because the choice of \
         filter is a direct trade-off: a steep, long filter (`ResamplingQuality::High`) preserves \
         the full bandwidth with minimal aliasing at higher CPU cost, whereas a short, cheap one \
         (`ResamplingQuality::Fast`, linear interpolation) is fast but rolls off highs and leaks \
         some alias energy.\n\
         \n\
         Length:  {before_len}  →  {after_len} samples"
    )
}
