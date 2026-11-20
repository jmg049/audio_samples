//! The audio operations: one trait per area, implemented for
//! [`AudioSamples`](crate::AudioSamples), with the modules that implement them and
//! their configuration types.
//!
//! The crate root explains the conventions every module follows (the
//! [precision rule](crate#precision), the [array layout](crate#array-layout) and
//! [configuration](crate#configuration)) and has a
//! [task table](crate#where-to-find-things). This page adds the table below, for
//! tasks that more than one API can do. Bring a trait into scope, from here, from
//! [`traits`] or from the crate root, and call its methods:
//!
//! ```rust
//! # #[cfg(all(feature = "statistics", feature = "processing"))] {
//! use audio_samples::operations::types::NormalizationConfig;
//! use audio_samples::operations::{AudioProcessing, AudioStatistics};
//! use audio_samples::{sample_rate, sine_wave};
//! use std::time::Duration;
//!
//! let audio = sine_wave::<f32>(440.0, Duration::from_millis(100), sample_rate!(8000), 0.25);
//! let louder = audio.normalize(NormalizationConfig::peak(1.0)).unwrap();
//! assert!((louder.peak() - 1.0).abs() < 1e-6);
//! assert!((audio.peak() - 0.25).abs() < 1e-6); // the original is unchanged
//! # }
//! ```
//!
//! ## Choosing between overlapping APIs
//!
//! Some tasks have two or more APIs. They follow different conventions, so their
//! results differ in detail; pick by the conventions you need, and do not mix
//! results from both in one pipeline.
//!
//! | Task | Use | Alternative | How they differ |
//! |---|---|---|---|
#![cfg_attr(
    feature = "transforms",
    doc = r#" | STFT, mel spectrogram, MFCC | [`AudioSpectrum`] | [`AudioTransforms`] | `AudioSpectrum` takes any number of channels, uses a periodic window and configurable padding, gives dB relative to a reference with an optional floor (`top_db`) and an orthonormal DCT, and computes in the [analysis float](crate#precision). `AudioTransforms` is mono, returns the [`spectrograms`] crate's types, uses a symmetric window, zero padding, an absolute dB floor and an unnormalised DCT-II with a lifter, and computes in `f64`. It also has log-frequency and gammatone spectrograms and a Welch PSD. |"#
)]
#![cfg_attr(
    not(feature = "transforms"),
    doc = r#" | STFT, mel spectrogram, MFCC | `AudioSpectrum` | `AudioTransforms` | `AudioSpectrum` takes any number of channels, uses a periodic window and configurable padding, gives dB relative to a reference with an optional floor (`top_db`) and an orthonormal DCT, and computes in the analysis float. `AudioTransforms` is mono, returns the `spectrograms` crate's types, uses a symmetric window, zero padding, an absolute dB floor and an unnormalised DCT-II with a lifter, and computes in `f64`. It also has log-frequency and gammatone spectrograms and a Welch PSD. |"#
)]
#![cfg_attr(
    all(feature = "transforms", feature = "resampling"),
    doc = r#" | Constant-Q transform | [`AudioConstantQ::cqt`] | [`AudioTransforms::constant_q_transform`] | `cqt` is the recursive per-octave transform with wavelet filters that the chroma and tonnetz features use. The alternative uses the `spectrograms` crate's kernel; its values differ by several percent and it returns fewer frames. |"#
)]
#![cfg_attr(
    not(all(feature = "transforms", feature = "resampling")),
    doc = r#" | Constant-Q transform | `AudioConstantQ::cqt` | `AudioTransforms::constant_q_transform` | `cqt` is the recursive per-octave transform with wavelet filters that the chroma and tonnetz features use. The alternative uses the `spectrograms` crate's kernel; its values differ by several percent and it returns fewer frames. |"#
)]
#![cfg_attr(
    all(feature = "feature-extraction", feature = "resampling"),
    doc = r#" | Chroma | [`AudioTonal::chroma_stft`], [`AudioConstantQ::chroma_cqt`] | [`AudioTransforms::chromagram`] | The first two use a power spectrogram, an octave-weighted Gaussian filterbank, max-normalised frames and tuning estimation. `chromagram` uses a magnitude spectrogram, L2 normalisation and no tuning estimation, and gives flatter profiles. |"#
)]
#![cfg_attr(
    not(all(feature = "feature-extraction", feature = "resampling")),
    doc = r#" | Chroma | `AudioTonal::chroma_stft`, `AudioConstantQ::chroma_cqt` | `AudioTransforms::chromagram` | The first two use a power spectrogram, an octave-weighted Gaussian filterbank, max-normalised frames and tuning estimation. `chromagram` uses a magnitude spectrogram, L2 normalisation and no tuning estimation, and gives flatter profiles. |"#
)]
#![cfg_attr(
    all(feature = "feature-extraction", feature = "statistics"),
    doc = r#" | Spectral centroid, rolloff, flatness | [`AudioFeatures`] | [`AudioStatistics`] | `AudioFeatures` gives one value per frame (`*_frames`); `AudioStatistics` one value for the whole signal. |"#
)]
#![cfg_attr(
    not(all(feature = "feature-extraction", feature = "statistics")),
    doc = r#" | Spectral centroid, rolloff, flatness | `AudioFeatures` | `AudioStatistics` | `AudioFeatures` gives one value per frame (`*_frames`); `AudioStatistics` one value for the whole signal. |"#
)]
#![cfg_attr(
    all(
        feature = "feature-extraction",
        feature = "statistics",
        feature = "envelopes"
    ),
    doc = r#" | RMS | [`AudioFeatures::rms_frames`] | [`AudioStatistics::rms`], [`AudioEnvelopes::rms_envelope`] | `rms_frames` is per centred frame, in units of full scale. `rms` is one value for the signal and `rms_envelope` uses frames that start at `k × hop`; both use the raw stored values. |"#
)]
#![cfg_attr(
    not(all(
        feature = "feature-extraction",
        feature = "statistics",
        feature = "envelopes"
    )),
    doc = r#" | RMS | `AudioFeatures::rms_frames` | `AudioStatistics::rms`, `AudioEnvelopes::rms_envelope` | `rms_frames` is per centred frame, in units of full scale. `rms` is one value for the signal and `rms_envelope` uses frames that start at `k × hop`; both use the raw stored values. |"#
)]
#![cfg_attr(
    all(feature = "pitch-analysis", feature = "feature-extraction"),
    doc = r#" | Pitch | [`AudioPitchTracking::yin`], [`AudioPitchTracking::pyin`] | [`AudioPitchAnalysis`], [`AudioTonal::piptrack`], [`detect_fundamental_frequency`](crate::utils::detection::detect_fundamental_frequency) | `AudioPitchTracking` returns one estimate per centred frame and channel, and pYIN adds voicing. `AudioPitchAnalysis` estimates one pitch per buffer, or a mono contour whose frames start at sample 0 with unvoiced frames as `None`. `piptrack` gives candidate pitches per frequency bin; `detect_fundamental_frequency` one autocorrelation estimate for the whole signal. |"#
)]
#![cfg_attr(
    not(all(feature = "pitch-analysis", feature = "feature-extraction")),
    doc = r#" | Pitch | `AudioPitchTracking::yin`, `AudioPitchTracking::pyin` | `AudioPitchAnalysis`, `AudioTonal::piptrack`, `detect_fundamental_frequency` | `AudioPitchTracking` returns one estimate per centred frame and channel, and pYIN adds voicing. `AudioPitchAnalysis` estimates one pitch per buffer, or a mono contour whose frames start at sample 0 with unvoiced frames as `None`. `piptrack` gives candidate pitches per frequency bin; `detect_fundamental_frequency` one autocorrelation estimate for the whole signal. |"#
)]
#![cfg_attr(
    feature = "beat-tracking",
    doc = r#" | Onsets | [`AudioRhythm::onset_detect`] | [`AudioOnsetDetection`](crate::AudioOnsetDetection) | `onset_detect` peak-picks a spectral-flux onset strength computed from a log-power mel spectrogram and returns frame indices ([`FrameEvents`](rhythm::FrameEvents)). `AudioOnsetDetection` uses constant-Q detection functions and returns times in seconds. |"#
)]
#![cfg_attr(
    not(feature = "beat-tracking"),
    doc = r#" | Onsets | `AudioRhythm::onset_detect` | `AudioOnsetDetection` | `onset_detect` peak-picks a spectral-flux onset strength computed from a log-power mel spectrogram and returns frame indices (`FrameEvents`). `AudioOnsetDetection` uses constant-Q detection functions and returns times in seconds. |"#
)]
#![cfg_attr(
    feature = "beat-tracking",
    doc = r#" | Tempo and beats | [`AudioRhythm::tempo`], [`AudioRhythm::beat_track`] | [`AudioBeatTracking`] | `AudioRhythm` estimates tempo from an autocorrelation tempogram with a log-normal prior and places beats by dynamic programming, following the onsets. `AudioBeatTracking` places a rigid grid at one tempo, anchored at the strongest onset. |"#
)]
#![cfg_attr(
    not(feature = "beat-tracking"),
    doc = r#" | Tempo and beats | `AudioRhythm::tempo`, `AudioRhythm::beat_track` | `AudioBeatTracking` | `AudioRhythm` estimates tempo from an autocorrelation tempogram with a log-normal prior and places beats by dynamic programming, following the onsets. `AudioBeatTracking` places a rigid grid at one tempo, anchored at the strongest onset. |"#
)]
#![cfg_attr(
    all(feature = "effects", feature = "decomposition"),
    doc = r#" | Harmonic-percussive separation | [`AudioEffects::harmonic_percussive`] | [`AudioDecomposition::hpss`] | `harmonic_percussive` takes any number of channels and uses soft masks with margins and reflect padding, and returns audio of the input's length. `hpss` is mono, with its own masks (`mask_softness`) and a symmetric window. |"#
)]
#![cfg_attr(
    not(all(feature = "effects", feature = "decomposition")),
    doc = r#" | Harmonic-percussive separation | `AudioEffects::harmonic_percussive` | `AudioDecomposition::hpss` | `harmonic_percussive` takes any number of channels and uses soft masks with margins and reflect padding, and returns audio of the input's length. `hpss` is mono, with its own masks (`mask_softness`) and a symmetric window. |"#
)]
#![cfg_attr(
    feature = "effects",
    doc = r#" | Silence trimming | [`AudioEffects::trim_db`], [`AudioEffects::split_db`] | [`AudioEditing::trim_silence`] | `trim_db` compares frame RMS with the loudest frame (`top_db` below it). `trim_silence` compares each sample with an absolute threshold in dB re full scale. |"#
)]
#![cfg_attr(
    not(feature = "effects"),
    doc = r#" | Silence trimming | `AudioEffects::trim_db`, `AudioEffects::split_db` | `AudioEditing::trim_silence` | `trim_db` compares frame RMS with the loudest frame (`top_db` below it). `trim_silence` compares each sample with an absolute threshold in dB re full scale. |"#
)]
#![cfg_attr(
    feature = "transforms",
    doc = r#" | Mel frequencies | [`hz_to_mel_with_scale`](crate::utils::audio_math::hz_to_mel_with_scale), [`mel_frequencies_with_scale`](crate::utils::audio_math::mel_frequencies_with_scale) | [`hz_to_mel`](crate::utils::audio_math::hz_to_mel), [`mel_frequencies`](crate::utils::audio_math::mel_frequencies) | The mel spectrograms use the Slaney scale by default ([`MelScale`](crate::MelScale)). `hz_to_mel` and `mel_frequencies` use the HTK formula only, so they do not label those bands. |"#
)]
#![cfg_attr(
    not(feature = "transforms"),
    doc = r#" | Mel frequencies | `hz_to_mel_with_scale`, `mel_frequencies_with_scale` | `hz_to_mel`, `mel_frequencies` | The mel spectrograms use the Slaney scale by default (`MelScale`). `hz_to_mel` and `mel_frequencies` use the HTK formula only, so they do not label those bands. |"#
)]
#![cfg_attr(
    feature = "transforms",
    doc = r#" | Decibels | [`spectrum::amplitude_to_db`], [`spectrum::power_to_db`] | [`utils::audio_math::amplitude_to_db`](crate::utils::audio_math::amplitude_to_db), [`utils::audio_math::power_to_db`](crate::utils::audio_math::power_to_db) | The `spectrum` functions convert arrays with a reference, a lower clamp and an optional `top_db`. The `utils` functions convert one value with reference 1 and return −80 dB only for non-positive input. `utils` re-exports them under the same names. |"#
)]
#![cfg_attr(
    not(feature = "transforms"),
    doc = r#" | Decibels | `spectrum::amplitude_to_db`, `spectrum::power_to_db` | `utils::audio_math::amplitude_to_db`, `utils::audio_math::power_to_db` | The `spectrum` functions convert arrays with a reference, a lower clamp and an optional `top_db`. The `utils` functions convert one value with reference 1 and return −80 dB only for non-positive input. `utils` re-exports them under the same names. |"#
)]
#![cfg_attr(
    all(feature = "resampling", feature = "processing"),
    doc = r#" | Resampling | [`resample`](crate::resample), [`AudioProcessing::resample`] | [`StreamingResampler`](crate::StreamingResampler) | `resample` converts a whole buffer in the analysis float. `StreamingResampler` converts a stream chunk by chunk, in `f64`. |"#
)]
#![cfg_attr(
    not(all(feature = "resampling", feature = "processing")),
    doc = r#" | Resampling | `resample`, `AudioProcessing::resample` | `StreamingResampler` | `resample` converts a whole buffer in the analysis float. `StreamingResampler` converts a stream chunk by chunk, in `f64`. |"#
)]
pub mod traits;
pub mod types;

#[cfg(feature = "beat-tracking")]
pub mod beat;

#[cfg(feature = "channels")]
pub mod channels;

#[cfg(feature = "dynamic-range")]
pub mod dynamic_range;

#[cfg(feature = "editing")]
pub mod editing;

#[cfg(feature = "decomposition")]
pub mod hpss;

#[cfg(feature = "iir-filtering")]
pub(crate) mod iir_design;
#[cfg(feature = "iir-filtering")]
pub mod iir_filtering;

#[cfg(feature = "parametric-eq")]
pub mod parametric_eq;

#[cfg(feature = "peak-picking")]
pub mod peak_picking;

#[cfg(feature = "pitch-analysis")]
pub mod pitch_analysis;

#[cfg(feature = "pitch-analysis")]
pub mod pitch_tracking;

#[cfg(feature = "processing")]
pub mod processing;

#[cfg(feature = "statistics")]
pub mod statistics;

#[cfg(feature = "transforms")]
pub mod transforms;

#[cfg(feature = "transforms")]
pub mod filterbanks;

#[cfg(feature = "transforms")]
pub mod spectrum;

#[cfg(all(feature = "transforms", feature = "resampling"))]
pub mod constant_q;

#[cfg(feature = "vad")]
pub mod vad;

#[cfg(feature = "onset-detection")]
pub mod onset;

#[cfg(feature = "plotting")]
pub mod plotting;

#[cfg(feature = "envelopes")]
pub mod envelopes;

#[cfg(feature = "sequence")]
pub mod sequence;

#[cfg(feature = "linalg")]
pub mod linalg;

#[cfg(feature = "feature-inversion")]
pub mod inverse;

#[cfg(feature = "factorization")]
pub mod factorization;

#[cfg(feature = "pitch-filterbank")]
pub mod pitch_filterbank;

#[cfg(feature = "mellin")]
pub mod mellin;

#[cfg(feature = "segmentation")]
pub mod segment;

#[cfg(feature = "feature-extraction")]
pub mod tonal;

#[cfg(feature = "feature-extraction")]
pub mod features;
#[cfg(feature = "feature-extraction")]
pub use traits::AudioFeatures;

#[cfg(feature = "effects")]
pub mod effects;

#[cfg(feature = "beat-tracking")]
pub mod rhythm;

// Re-export main traits for convenience
#[cfg(feature = "statistics")]
pub use traits::AudioStatistics;

#[cfg(feature = "vad")]
pub use traits::AudioVoiceActivityDetection;

#[cfg(feature = "processing")]
pub use traits::AudioProcessing;

#[cfg(feature = "channels")]
pub use traits::AudioChannelOps;

#[cfg(feature = "editing")]
pub use traits::AudioEditing;

#[cfg(feature = "iir-filtering")]
pub use traits::AudioIirFiltering;

#[cfg(feature = "parametric-eq")]
pub use traits::AudioParametricEq;

#[cfg(feature = "peak-picking")]
pub use crate::operations::peak_picking::*;

#[cfg(feature = "pitch-analysis")]
pub use traits::AudioPitchAnalysis;

#[cfg(feature = "pitch-analysis")]
pub use traits::AudioPitchTracking;

#[cfg(feature = "dynamic-range")]
pub use traits::AudioDynamicRange;

#[cfg(feature = "transforms")]
pub use traits::AudioTransforms;

#[cfg(feature = "transforms")]
pub use traits::AudioSpectrum;

#[cfg(feature = "feature-extraction")]
pub use traits::AudioTonal;

#[cfg(all(feature = "transforms", feature = "resampling"))]
pub use traits::AudioConstantQ;
#[cfg(feature = "effects")]
pub use traits::AudioEffects;

#[cfg(feature = "decomposition")]
pub use traits::AudioDecomposition;

#[cfg(feature = "beat-tracking")]
pub use traits::AudioBeatTracking;

#[cfg(feature = "beat-tracking")]
pub use beat::*;

#[cfg(feature = "beat-tracking")]
pub use traits::AudioRhythm;

#[cfg(feature = "onset-detection")]
pub use onset::*;

#[cfg(feature = "decomposition")]
pub use hpss::*;

#[cfg(feature = "plotting")]
pub use traits::AudioPlotting;

#[cfg(feature = "plotting")]
pub use plotting::waveform::{WaveformPlot, WaveformPlotParams, create_waveform_plot};

#[cfg(feature = "envelopes")]
pub use traits::AudioEnvelopes;

#[cfg(feature = "plotting")]
pub use plotting::spectrograms::{
    SpectrogramPlot, SpectrogramPlotParams, SpectrogramType, create_spectrogram_plot,
};

#[cfg(feature = "plotting")]
pub use plotting::composite::{CompositeLayout, CompositePlot, PlotComponent};

#[cfg(feature = "plotting")]
pub use plotting::spectrum::{
    MagnitudeSpectrumParams, MagnitudeSpectrumPlot, create_magnitude_spectrum_plot,
};

#[cfg(feature = "plotting")]
pub use plotting::phase_spectrum::{
    PhaseSpectrumParams, PhaseSpectrumPlot, create_phase_spectrum_plot,
};

#[cfg(feature = "plotting")]
pub use plotting::lissajous::{LissajousParams, LissajousPlot, create_lissajous_plot};

#[cfg(feature = "plotting")]
pub use plotting::{ChannelManagementStrategy, Layout, PlotParams, PlotUtils};

#[cfg(feature = "plotting")]
pub use plotting::dsp_overlays;

#[cfg(feature = "resampling")]
pub use types::ResamplingQuality;
