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
