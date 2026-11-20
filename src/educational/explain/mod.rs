//! Explanations of configurations and concepts, computed from your settings.
//!
//! This module turns a configuration value (an
#![cfg_attr(feature = "transforms", doc = r#" [`StftParams`],"#)]
#![cfg_attr(not(feature = "transforms"), doc = r#" `StftParams`,"#)]
//! an [`IirFilterDesign`], a
#![cfg_attr(feature = "dynamic-range", doc = r#" [`CompressorConfig`],"#)]
#![cfg_attr(not(feature = "dynamic-range"), doc = r#" `CompressorConfig`,"#)]
//! …) or a bare concept
//! ("what happens to a 7 kHz tone sampled at 8 kHz?") into an
//! [`ExplainDoc`]: a short summary, the formulas the code
//! evaluates, the symbols in those formulas with their values, numbers derived
//! from your settings (frequency resolution, group delay, pole radii, …),
//! trade-offs, and warnings about suspicious settings.
//!
//! Signal-processing parameters are easy to get subtly wrong: a window that is
//! too short to resolve a low voice, a mel filterbank with empty bands, a
//! filter whose cutoff is at the Nyquist frequency. The
//! [`explaining`](explainable::Explainable::explaining) chain answers "what did
//! this pipeline do to my signal" *after* processing; this module answers "what
//! will this configuration do, and why" *before* anything runs. Every number is
//! computed (window figures are measured from the actual window coefficients,
//! filter gains from the crate's own frequency-response code), and where the
//! implementation deviates from a textbook convention the explanation says
//! what the code does.
//!
//! * Types that implement [`Explain`] can be explained on their own
//!   (`params.explain()`) or with extra context
//!   (`params.explain_with(&ExplainContext::new().with_sample_rate(16_000.0))`).
//! * Any [`AudioSamples`] can explain a configuration in the context of that
//!   signal through [`ExplainFor`]: `audio.explain(&params)` uses the signal's
//!   sample rate, length and channel count.
//! * Concepts without a configuration type live in [`topics`].
//!
//! Render an [`ExplainDoc`] with [`to_text`](ExplainDoc::to_text) (also its
//! `Display`), [`to_markdown`](ExplainDoc::to_markdown) (Jupyter, mdBook) or
//! [`to_html`](ExplainDoc::to_html).
//!
//! ```
//! use audio_samples::educational::{Explain, ExplainContext, WarningLevel, topics};
//!
//! // A concept: the alias of a 7 kHz tone sampled at 8 kHz.
//! let doc = topics::aliasing(7_000.0, 8_000.0);
//! assert_eq!(doc.quantity("Alias frequency").unwrap().value, 1_000.0);
//! println!("{doc}");
//!
//! // A configuration, with context.
//! # #[cfg(feature = "transforms")] {
//! use audio_samples::nzu;
//! use spectrograms::{StftParams, WindowType};
//! let stft = StftParams::new(nzu!(256), nzu!(128), WindowType::Hanning, true).unwrap();
//! let ctx = ExplainContext::new().with_sample_rate(16_000.0).with_f0_range(80.0, 400.0);
//! let doc = stft.explain_with(&ctx);
//! assert_eq!(doc.quantity("Frequency resolution").unwrap().value, 62.5);
//! assert!(doc.has_warning(WarningLevel::Caution)); // 16 ms is shorter than 2 periods of 80 Hz
//! # }
//! ```
//!
#![cfg_attr(
    feature = "transforms",
    doc = r#" [`StftParams`]: spectrograms::StftParams"#
)]
//! [`IirFilterDesign`]: crate::operations::types::IirFilterDesign
#![cfg_attr(
    feature = "dynamic-range",
    doc = r#" [`CompressorConfig`]: crate::CompressorConfig"#
)]

mod doc;
mod numeric;
pub mod topics;

#[cfg(feature = "transforms")]
mod spectral;

#[cfg(feature = "iir-filtering")]
mod filters;

#[cfg(feature = "dynamic-range")]
mod dynamics;

#[cfg(any(feature = "vad", feature = "onset-detection"))]
mod detection;

pub use doc::{
    ExplainDoc, Formula, Quantity, SeeAlso, Symbol, Table, Topic, Warning, WarningLevel,
};

use crate::repr::AudioSamples;
use crate::traits::StandardSample;

/// Optional facts about the signal and the analysis that an explainer may use
/// to turn symbols into numbers and to issue warnings.
///
/// ## Purpose
/// Configuration types rarely carry everything needed to explain them: an
#[cfg_attr(
    feature = "transforms",
    doc = r#" [`StftParams`](spectrograms::StftParams) has no sample rate, a mel"#
)]
#[cfg_attr(
    not(feature = "transforms"),
    doc = r#" `StftParams` has no sample rate, a mel"#
)]
/// filterbank does not know the FFT size. The context supplies those facts.
///
/// ## Intended Usage
/// Build with [`ExplainContext::new`] and the `with_*` methods, or from a
/// signal with [`ExplainContext::for_audio`] (which [`ExplainFor`] does for
/// you). Every field is optional; when a fact is missing, explainers print
/// symbolic formulas and an informational note instead of numbers.
///
/// ## Invariants
/// None enforced; explainers validate the values they use and report
/// unusable ones as warnings.
///
/// ## Assumptions
/// `n_samples` counts samples per channel.
#[derive(Debug, Default, Clone, PartialEq)]
#[non_exhaustive]
pub struct ExplainContext {
    /// Sample rate in Hz.
    pub sample_rate_hz: Option<f64>,
    /// Signal length in samples per channel.
    pub n_samples: Option<usize>,
    /// Number of channels.
    pub n_channels: Option<usize>,
    /// Expected fundamental-frequency range `(lowest, highest)` in Hz.
    /// Enables "window shorter than a pitch period" style checks.
    pub f0_range_hz: Option<(f64, f64)>,
    /// FFT size the configuration will be used with (for mel filterbanks).
    pub n_fft: Option<usize>,
    /// Number of mel bands (for MFCC configurations).
    pub n_mels: Option<usize>,
}

impl ExplainContext {
    /// Creates an empty context.
    ///
    /// # Returns
    ///
    /// A context with every field `None`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a context describing `audio`: its sample rate, length per
    /// channel and channel count.
    ///
    /// # Arguments
    ///
    /// * `audio` - the signal the configuration will be applied to.
    ///
    /// # Returns
    ///
    /// A context with `sample_rate_hz`, `n_samples` and `n_channels` set.
    ///
    /// # Examples
    ///
    /// ```
    /// use audio_samples::educational::ExplainContext;
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f32>(440.0, Duration::from_millis(100), sample_rate!(16000), 0.5);
    /// let ctx = ExplainContext::for_audio(&audio);
    /// assert_eq!(ctx.sample_rate_hz, Some(16_000.0));
    /// assert_eq!(ctx.n_samples, Some(1600));
    /// assert_eq!(ctx.n_channels, Some(1));
    /// ```
    #[must_use]
    pub fn for_audio<T: StandardSample>(audio: &AudioSamples<'_, T>) -> Self {
        Self {
            sample_rate_hz: Some(audio.sample_rate_hz()),
            n_samples: Some(audio.samples_per_channel().get()),
            n_channels: Some(audio.num_channels().get() as usize),
            ..Self::default()
        }
    }

    /// Sets the sample rate.
    ///
    /// # Arguments
    ///
    /// * `hz` - sample rate in Hz.
    #[must_use]
    pub const fn with_sample_rate(mut self, hz: f64) -> Self {
        self.sample_rate_hz = Some(hz);
        self
    }

    /// Sets the signal length (samples per channel).
    ///
    /// # Arguments
    ///
    /// * `n` - samples per channel.
    #[must_use]
    pub const fn with_n_samples(mut self, n: usize) -> Self {
        self.n_samples = Some(n);
        self
    }

    /// Sets the number of channels.
    ///
    /// # Arguments
    ///
    /// * `n` - channel count.
    #[must_use]
    pub const fn with_n_channels(mut self, n: usize) -> Self {
        self.n_channels = Some(n);
        self
    }

    /// Sets the expected fundamental-frequency range.
    ///
    /// # Arguments
    ///
    /// * `low_hz` - lowest expected f0 in Hz.
    /// * `high_hz` - highest expected f0 in Hz.
    #[must_use]
    pub const fn with_f0_range(mut self, low_hz: f64, high_hz: f64) -> Self {
        self.f0_range_hz = Some((low_hz, high_hz));
        self
    }

    /// Sets the FFT size the configuration will be combined with.
    ///
    /// # Arguments
    ///
    /// * `n_fft` - FFT length in samples.
    #[must_use]
    pub const fn with_n_fft(mut self, n_fft: usize) -> Self {
        self.n_fft = Some(n_fft);
        self
    }

    /// Sets the number of mel bands (used when explaining MFCC settings).
    ///
    /// # Arguments
    ///
    /// * `n_mels` - number of mel bands.
    #[must_use]
    pub const fn with_n_mels(mut self, n_mels: usize) -> Self {
        self.n_mels = Some(n_mels);
        self
    }

    /// The sample rate if it is present, finite and positive.
    pub(crate) fn valid_sample_rate(&self) -> Option<f64> {
        self.sample_rate_hz.filter(|f| f.is_finite() && *f > 0.0)
    }
}

/// A value that can describe what it will do.
///
/// Implemented for the crate's configuration types (behind their cargo
/// features) and open for implementation by downstream types.
///
/// # Examples
///
/// ```
/// use audio_samples::educational::{Explain, ExplainContext, ExplainDoc, Topic};
///
/// struct Gain(f64);
///
/// impl Explain for Gain {
///     fn explain_with(&self, _ctx: &ExplainContext) -> ExplainDoc {
///         let mut doc = ExplainDoc::new(Topic::Dynamics, "Fixed gain");
///         doc.set_summary("Multiplies every sample by a constant.");
///         doc.add_formula(r"y[n] = g\,x[n]", "Output sample");
///         doc.add_quantity("Gain", "G", 20.0 * self.0.log10(), "dB", r"20\log_{10} g");
///         doc
///     }
/// }
///
/// assert!((Gain(0.5).explain().quantity("Gain").unwrap().value + 6.0206).abs() < 1e-3);
/// ```
pub trait Explain {
    /// Explains `self` using whatever facts `ctx` provides.
    ///
    /// # Arguments
    ///
    /// * `ctx` - optional facts about the signal and analysis.
    ///
    /// # Returns
    ///
    /// The explanation. Invalid settings are reported as
    /// [`WarningLevel::Problem`] warnings rather than errors.
    fn explain_with(&self, ctx: &ExplainContext) -> ExplainDoc;

    /// Explains `self` without context.
    ///
    /// # Returns
    ///
    /// The explanation; quantities that need a sample rate are omitted.
    fn explain(&self) -> ExplainDoc {
        self.explain_with(&ExplainContext::default())
    }
}

/// Explain a configuration in the context of a particular signal.
///
/// Implemented for [`AudioSamples`]: `audio.explain(&params)` is shorthand for
/// `params.explain_with(&ExplainContext::for_audio(&audio))`.
///
/// # Examples
///
/// ```
/// use audio_samples::educational::ExplainFor;
/// use audio_samples::operations::types::IirFilterDesign;
/// use audio_samples::{nzu, sample_rate, sine_wave};
/// use std::time::Duration;
///
/// let audio = sine_wave::<f32>(440.0, Duration::from_millis(50), sample_rate!(48000), 0.5);
/// let doc = audio.explain(&IirFilterDesign::butterworth_lowpass(nzu!(4), 1_000.0));
/// let gain = doc.quantity("Measured gain at the cutoff").unwrap().value;
/// assert!((gain + 3.0103).abs() < 1e-3);
/// ```
pub trait ExplainFor {
    /// Explains `item` using the facts of `self` as context.
    ///
    /// # Arguments
    ///
    /// * `item` - any explainable configuration.
    ///
    /// # Returns
    ///
    /// The explanation.
    fn explain<E: Explain + ?Sized>(&self, item: &E) -> ExplainDoc;
}

impl<T: StandardSample> ExplainFor for AudioSamples<'_, T> {
    fn explain<E: Explain + ?Sized>(&self, item: &E) -> ExplainDoc {
        item.explain_with(&ExplainContext::for_audio(self))
    }
}

/// Adds the standard "no sample rate" note.
pub(crate) fn note_missing_sample_rate(doc: &mut ExplainDoc) {
    doc.add_warning(
        WarningLevel::Info,
        "No sample rate given, so quantities are shown in samples only.",
        "Use ExplainContext::with_sample_rate, or explain through a signal with audio.explain(&config).",
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{sample_rate, sine_wave};
    use std::time::Duration;

    #[test]
    fn context_for_audio_reads_signal_facts() {
        let audio = sine_wave::<f64>(100.0, Duration::from_millis(10), sample_rate!(8000), 0.1);
        let ctx = ExplainContext::for_audio(&audio);
        assert_eq!(ctx.sample_rate_hz, Some(8000.0));
        assert_eq!(ctx.n_samples, Some(80));
        assert_eq!(ctx.n_channels, Some(1));
        assert_eq!(ctx.valid_sample_rate(), Some(8000.0));
        assert_eq!(
            ExplainContext::new()
                .with_sample_rate(-1.0)
                .valid_sample_rate(),
            None
        );
    }
}
