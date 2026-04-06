#![allow(unused_imports)]
//! Configuration types and support enumerations for audio processing operations.
//!
//! This module defines every configuration struct, option enum, and helper type
//! used by the audio processing traits. Types are grouped by the feature gate that
//! enables them: editing, processing, channels, VAD, resampling, transforms, pitch
//! analysis, IIR filtering, parametric EQ, dynamic range, and peak picking.
//!
//! Centralising configuration types in one place ensures consistent naming, shared
//! validation logic, and a single source of truth for parameter constraints. Callers
//! construct a typed configuration value and pass it to the relevant processing
//! method, rather than threading raw primitive arguments through call chains.
//!
//! Each struct that accepts numeric bounds exposes a `validate` method that checks
//! invariants before the value is submitted to a processing call. Using preset
//! constructors (e.g. [`CompressorConfig::vocal`], [`VadConfig::energy_only`]) is
//! the quickest path to a sensible default.
//!
//! Import the types you need from [`audio_samples::operations::types`](crate::operations::types) or via the
//! crate-level re-export. Construct a configuration using one of the provided
//! constructors or preset methods, optionally adjust fields, then pass it to the
//! corresponding processing function.
//!
//! ```
//! # #[cfg(feature = "processing")]
//! # {
//! use audio_samples::operations::types::NormalizationConfig;
//!
//! // Use a preset — peak normalise to 1.0
//! let config = NormalizationConfig::<f64>::peak_normalized();
//! assert_eq!(config.target, Some(1.0));
//! # }
//! ```

// General todos in types.rs
// - Serialisation + Deserialisation for types
// - Better derives -- more consistency
// - More "helper" trait implementations like From/Into et al.

#[cfg(feature = "processing")]
use crate::traits::StandardSample;

use crate::{AudioSampleError, AudioSampleResult, ParameterError};
use core::fmt::Display;
use std::num::NonZeroUsize;
use std::str::FromStr;

/// Which end of a buffer to pad when extending audio length.
///
/// Used by padding operations that insert silence or a repeated value at one
/// end of the sample data. Defaults to [`Right`][PadSide::Right] (append).
#[cfg(feature = "editing")]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PadSide {
    /// Prepend padding before the first sample.
    Left,
    /// Append padding after the last sample.
    #[default]
    Right,
}

#[cfg(feature = "editing")]
impl FromStr for PadSide {
    type Err = crate::AudioSampleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "left" => Ok(Self::Left),
            "right" => Ok(Self::Right),
            _ => Err(AudioSampleError::Parameter(ParameterError::InvalidValue {
                parameter: s.to_string(),
                reason: "Expected 'left' or 'right'".to_string(),
            })),
        }
    }
}

/// Algorithm used to scale or centre audio sample values.
///
/// Different methods suit different goals. Choose the method that matches your
/// intended output range and tolerance for outliers.
#[cfg(any(feature = "processing", feature = "peak-picking"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum NormalizationMethod {
    /// Min-Max normalisation — scales all values into a caller-specified `[min, max]` range.
    ///
    /// Good for general level adjustment and ensuring samples stay within a
    /// target amplitude bound. Requires a valid `min` and `max` in the accompanying
    /// [`NormalizationConfig`].
    #[default]
    MinMax,

    /// Peak normalisation — scales by the maximum absolute value to reach a target peak.
    ///
    /// Preserves the dynamic shape of the signal while bringing the loudest sample
    /// to a specified level. Requires a valid `target` in the accompanying
    /// [`NormalizationConfig`].
    Peak,

    /// Mean normalisation — subtracts the mean so the output is centred at zero.
    ///
    /// Removes DC offset without altering relative amplitudes. Equivalent to
    /// zero-mean centering; does not rescale the range.
    Mean,

    /// Median normalisation — subtracts the median so the output is centred at zero.
    ///
    /// More robust than mean normalisation when the signal contains large outlier
    /// values, as the median is not skewed by extreme samples.
    Median,

    /// Z-score normalisation — transforms to zero mean and unit variance.
    ///
    /// Useful for statistical comparisons and machine-learning preprocessing where
    /// equal contribution from all features is required.
    ZScore,
}

#[cfg(feature = "processing")]
impl FromStr for NormalizationMethod {
    type Err = AudioSampleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalised = s.trim().to_ascii_lowercase();

        match normalised.as_str() {
            "min_max" | "min-max" | "minmax" => Ok(Self::MinMax),
            "peak" => Ok(Self::Peak),
            "mean" => Ok(Self::Mean),
            "median" => Ok(Self::Median),
            "zscore" | "z_score" | "z-score" => Ok(Self::ZScore),
            _ => Err(AudioSampleError::parse::<Self, _>(format!(
                "Failed to parse {}. Got {}, must be one of {:?}",
                std::any::type_name::<Self>(),
                s,
                [
                    "min_max", "min-max", "minmax", "peak", "mean", "median", "zscore", "z_score",
                    "z-score",
                ]
            ))),
        }
    }
}

/// Parameters governing a single call to `normalize`.
///
/// ## Purpose
///
/// Bundles the [`NormalizationMethod`] with the numeric bounds it requires into
/// one type-safe value. This avoids ambiguous argument lists and makes it clear
/// which fields are active for a given method.
///
/// ## Intended Usage
///
/// Construct with one of the provided constructors (`peak`, `min_max`, `mean`,
/// `median`, `zscore`) or use the `f64`-specific presets (`peak_normalized`,
/// `range_normalized`). Pass the resulting value to
/// [`AudioProcessing::normalize`][crate::operations::AudioProcessing::normalize].
///
/// ## Invariants
///
/// - When `method` is [`NormalizationMethod::MinMax`]: `min` and `max` must
///   both be `Some`, and `min < max`.
/// - When `method` is [`NormalizationMethod::Peak`]: `target` must be `Some`
///   and positive.
/// - For `Mean`, `Median`, and `ZScore`: `min`, `max`, and `target` are unused;
///   their values are ignored during processing.
///
/// ## Assumptions
///
/// `T` implements [`StandardSample`][crate::traits::StandardSample], which bounds
/// it to the numeric types supported throughout the crate.
#[cfg(feature = "processing")]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct NormalizationConfig<T>
where
    T: StandardSample,
{
    /// The normalisation algorithm to apply.
    pub method: NormalizationMethod,

    /// Lower bound for [`NormalizationMethod::MinMax`].
    ///
    /// Must be `Some` and less than `max` when `method` is `MinMax`.
    /// Ignored for all other methods.
    pub min: Option<T>,

    /// Upper bound for [`NormalizationMethod::MinMax`].
    ///
    /// Must be `Some` and greater than `min` when `method` is `MinMax`.
    /// Ignored for all other methods.
    pub max: Option<T>,

    /// Target peak amplitude for [`NormalizationMethod::Peak`].
    ///
    /// Must be `Some` and positive when `method` is `Peak`.
    /// Ignored for all other methods.
    pub target: Option<T>,
}

#[cfg(feature = "processing")]
impl<T> NormalizationConfig<T>
where
    T: StandardSample,
{
    /// Creates a peak normalisation configuration targeting `target`.
    ///
    /// # Arguments
    ///
    /// – `target` – the amplitude level that the loudest sample will be scaled to.
    ///   Must be positive for meaningful output.
    ///
    /// # Returns
    ///
    /// A [`NormalizationConfig`] with `method = Peak` and `target = Some(target)`.
    #[inline]
    pub const fn peak(target: T) -> Self {
        Self {
            method: NormalizationMethod::Peak,
            min: None,
            max: None,
            target: Some(target),
        }
    }

    /// Creates a min-max normalisation configuration scaling to `[min, max]`.
    ///
    /// # Arguments
    ///
    /// – `min` – the lower bound of the target amplitude range. Must be less than `max`.\
    /// – `max` – the upper bound of the target amplitude range. Must be greater than `min`.
    ///
    /// # Returns
    ///
    /// A [`NormalizationConfig`] with `method = MinMax`, `min = Some(min)`, and
    /// `max = Some(max)`.
    #[inline]
    pub const fn min_max(min: T, max: T) -> Self {
        Self {
            method: NormalizationMethod::MinMax,
            min: Some(min),
            max: Some(max),
            target: None,
        }
    }

    /// Creates a mean normalisation configuration (zero-mean centering).
    ///
    /// # Returns
    ///
    /// A [`NormalizationConfig`] with `method = Mean`. The `min`, `max`, and `target`
    /// fields are unused and set to `None`.
    #[inline]
    #[must_use]
    pub const fn mean() -> Self {
        Self {
            method: NormalizationMethod::Mean,
            min: None,
            max: None,
            target: None,
        }
    }

    /// Creates a median normalisation configuration (median centering).
    ///
    /// # Returns
    ///
    /// A [`NormalizationConfig`] with `method = Median`. The `min`, `max`, and `target`
    /// fields are unused and set to `None`.
    #[inline]
    #[must_use]
    pub const fn median() -> Self {
        Self {
            method: NormalizationMethod::Median,
            min: None,
            max: None,
            target: None,
        }
    }

    /// Creates a z-score normalisation configuration (zero mean, unit variance).
    ///
    /// # Returns
    ///
    /// A [`NormalizationConfig`] with `method = ZScore`. The `min`, `max`, and `target`
    /// fields are unused and set to `None`.
    #[inline]
    #[must_use]
    pub const fn zscore() -> Self {
        Self {
            method: NormalizationMethod::ZScore,
            min: None,
            max: None,
            target: None,
        }
    }
}

#[cfg(feature = "processing")]
impl NormalizationConfig<f64> {
    /// Peak normalisation that scales to a maximum absolute amplitude of 1.0.
    ///
    /// # Returns
    ///
    /// Equivalent to `NormalizationConfig::<f64>::peak(1.0)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "processing")]
    /// # {
    /// use audio_samples::operations::types::NormalizationConfig;
    ///
    /// let config = NormalizationConfig::<f64>::peak_normalized();
    /// assert_eq!(config.target, Some(1.0));
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub const fn peak_normalized() -> Self {
        Self::peak(1.0)
    }

    /// Min-Max normalisation into the `[-1.0, 1.0]` range.
    ///
    /// # Returns
    ///
    /// Equivalent to `NormalizationConfig::<f64>::min_max(-1.0, 1.0)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[cfg(feature = "processing")]
    /// # {
    /// use audio_samples::operations::types::NormalizationConfig;
    ///
    /// let config = NormalizationConfig::<f64>::range_normalized();
    /// assert_eq!(config.min, Some(-1.0));
    /// assert_eq!(config.max, Some(1.0));
    /// # }
    /// ```
    #[inline]
    #[must_use]
    pub const fn range_normalized() -> Self {
        Self::min_max(-1.0, 1.0)
    }
}

/// Shape of the gain envelope applied during a fade-in or fade-out.
///
/// Each variant describes how the gain multiplier changes from 0.0 to 1.0 over
/// the fade region. The default is [`Linear`][FadeCurve::Linear].
#[cfg(feature = "editing")]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FadeCurve {
    /// Gain changes at a constant rate from start to end.
    ///
    /// Simple and predictable, but can sound abrupt at the endpoints because
    /// there is no gradual onset or release.
    #[default]
    Linear,

    /// Gain changes rapidly near the start and slowly near the end.
    ///
    /// Produces a fast attack or quick onset when fading in, and a slow tail
    /// when fading out. Perceptually emphasises the early portion of the fade.
    Exponential,

    /// Gain changes slowly near the start and rapidly near the end.
    ///
    /// Produces a gradual onset and a sharp cutoff. The perceptual complement
    /// of [`Exponential`][FadeCurve::Exponential].
    Logarithmic,

    /// S-shaped gain curve with zero first derivative at both endpoints.
    ///
    /// Starts and ends smoothly, resulting in the most perceptually natural
    /// transition. Implemented as a cubic Hermite interpolation (smoothstep).
    SmoothStep,
}

#[cfg(feature = "editing")]
impl FromStr for FadeCurve {
    type Err = AudioSampleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err(AudioSampleError::parse::<Self, _>(
                "Input must not be empty. Must be one of ['linear', 'exp', 'exponential', 'log', 'logarithmic', 'smooth', 'smoothstep']",
            ));
        }

        let normalised = s.trim();
        match normalised.to_lowercase().as_str() {
            "linear" => Ok(Self::Linear),
            "exp" | "exponential" => Ok(Self::Exponential),
            "log" | "logarithmic" => Ok(Self::Logarithmic),
            "smooth" | "smoothstep" => Ok(Self::SmoothStep),
            _ => Err(AudioSampleError::parse::<Self, _>(format!(
                "Failed to parse {normalised}. Must be one of ['linear', 'exp', 'exponential', 'log', 'logarithmic', 'smooth', 'smoothstep']"
            ))),
        }
    }
}

/// Algorithm for collapsing a multi-channel signal to a single mono channel.
///
/// Select the variant that best matches the intended perceptual result.
/// [`Average`][MonoConversionMethod::Average] is the default and suits general
/// downmix scenarios. Use [`Weighted`][MonoConversionMethod::Weighted] when
/// channels carry unequal acoustic importance.
#[cfg(feature = "channels")]
#[derive(Default, Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum MonoConversionMethod {
    /// Mix all channels to mono with equal weight.
    ///
    /// Each output sample is the arithmetic mean across all input channels.
    #[default]
    Average,

    /// Use only the left channel (channel index 0).
    ///
    /// Appropriate for stereo input when only left content is needed.
    Left,

    /// Use only the right channel (channel index 1).
    ///
    /// Appropriate for stereo input when only right content is needed.
    Right,

    /// Mix channels using per-channel gain weights.
    ///
    /// The inner `Vec<f64>` must contain one weight per channel. Weights need
    /// not sum to 1.0; they are applied as linear gain multipliers and the
    /// resulting values are summed. Mismatched length behaviour is
    /// implementation-defined.
    Weighted(Vec<f64>),

    /// Use the center channel when present; fall back to averaging L/R otherwise.
    ///
    /// For 5.1 and similar multichannel formats where a discrete center channel
    /// (typically index 2) carries dialogue or lead content.
    Center,
}

/// Algorithm for expanding a mono signal into a two-channel stereo pair.
///
/// Defaults to [`Duplicate`][StereoConversionMethod::Duplicate], which produces
/// a centred mono-compatible stereo signal.
#[cfg(feature = "channels")]
#[derive(Default, Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum StereoConversionMethod {
    /// Copy the mono signal identically to both left and right channels.
    ///
    /// The result is a centred, mono-compatible stereo signal.
    #[default]
    Duplicate,

    /// Pan the mono signal to a specific stereo position.
    ///
    /// The inner `f64` encodes the pan position: `-1.0` = hard left,
    /// `0.0` = centre, `1.0` = hard right. Values outside `[-1.0, 1.0]`
    /// have implementation-defined behaviour.
    Pan(f64),

    /// Route the mono signal to the left channel; fill the right with silence.
    Left,

    /// Route the mono signal to the right channel; fill the left with silence.
    Right,
}

/// Algorithm used by the Voice Activity Detector to classify frames.
///
/// Each method makes different trade-offs between simplicity, computational cost,
/// and robustness to noise. [`Energy`][VadMethod::Energy] is the default and
/// suits clean or lightly noisy speech. For challenging conditions, prefer
/// [`Combined`][VadMethod::Combined] or [`Spectral`][VadMethod::Spectral].
#[cfg(feature = "vad")]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum VadMethod {
    /// RMS energy threshold detection.
    ///
    /// A frame is classified as speech when its RMS level exceeds
    /// [`VadConfig::energy_threshold_db`]. Fast and reliable for clean audio;
    /// sensitive to background noise.
    #[default]
    Energy,

    /// Zero-crossing rate (ZCR) detection.
    ///
    /// Uses the per-frame rate of sign changes as a proxy for voiced activity.
    /// Speech typically produces ZCR in the range defined by
    /// [`VadConfig::zcr_min`] and [`VadConfig::zcr_max`].
    ZeroCrossing,

    /// Combined energy and zero-crossing rate detection.
    ///
    /// Both the energy threshold and the ZCR window must be satisfied for a
    /// frame to be classified as speech. Reduces false positives from either
    /// method alone.
    Combined,

    /// Spectral energy ratio detection.
    ///
    /// Computes the fraction of spectral energy within the speech band
    /// (`speech_band_low_hz`..`speech_band_high_hz`) and compares it to
    /// [`VadConfig::spectral_ratio_threshold`]. More computationally intensive
    /// but more discriminative in noisy environments.
    Spectral,
}

#[cfg(feature = "vad")]
impl FromStr for VadMethod {
    type Err = AudioSampleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalised = s.trim().to_ascii_lowercase();

        match normalised.as_str() {
            "energy" | "rms" => Ok(Self::Energy),
            "zero_crossing" | "zero-crossing" | "zcr" => Ok(Self::ZeroCrossing),
            "combined" | "energy_zcr" | "energy-zcr" => Ok(Self::Combined),
            "spectral" | "spectrum" => Ok(Self::Spectral),
            _ => Err(AudioSampleError::parse::<Self, _>(format!(
                "Failed to parse {}. Got {}, must be one of {:?}",
                std::any::type_name::<Self>(),
                s,
                [
                    "energy",
                    "rms",
                    "zero_crossing",
                    "zero-crossing",
                    "zcr",
                    "combined",
                    "energy_zcr",
                    "energy-zcr",
                    "spectral",
                    "spectrum",
                ]
            ))),
        }
    }
}

/// How to derive a single activity decision from a multi-channel signal.
///
/// Applies only when the input has more than one channel. For mono input this
/// setting has no effect. Defaults to
/// [`AverageToMono`][VadChannelPolicy::AverageToMono].
#[cfg(feature = "vad")]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum VadChannelPolicy {
    /// Average all channels to mono before running VAD.
    ///
    /// Cheapest option; produces one activity mask independent of channel
    /// count.
    #[default]
    AverageToMono,

    /// Run VAD on every channel independently; output `true` if any is active.
    ///
    /// Use when voice may appear on any single channel (e.g. a microphone
    /// array where only some elements pick up speech).
    AnyChannel,

    /// Run VAD on every channel independently; output `true` only if all are active.
    ///
    /// Use when corroborated activity across all channels is required before
    /// classifying a frame as speech.
    AllChannels,

    /// Run VAD on only the specified channel index.
    ///
    /// The inner `usize` is the zero-based channel index. Behaviour when the
    /// index exceeds the actual channel count is implementation-defined.
    Channel(usize),
}

/// Full configuration for Voice Activity Detection.
///
/// ## Purpose
///
/// Controls every aspect of the frame-based VAD pipeline: the detection algorithm,
/// framing parameters, per-method thresholds, temporal smoothing, and multi-channel
/// handling.
///
/// ## Intended Usage
///
/// Use [`VadConfig::energy_only`] for a quick, energy-threshold VAD; use the
/// [`Default`] impl for a balanced starting point; or build a fully custom
/// configuration with [`VadConfig::new`]. Call [`validate`][VadConfig::validate]
/// before passing to a VAD function to catch invalid parameter combinations.
///
/// ## Invariants
///
/// - `hop_size` must be ≤ `frame_size`.
/// - `0.0 ≤ zcr_min ≤ zcr_max ≤ 1.0`.
/// - `0.0 < speech_band_low_hz < speech_band_high_hz`.
/// - `0.0 ≤ spectral_ratio_threshold ≤ 1.0`.
///
/// ## Assumptions
///
/// Threshold fields that are not relevant to the chosen [`VadMethod`] are
/// silently ignored during processing (e.g. `zcr_*` fields have no effect when
/// `method` is [`VadMethod::Energy`]).
#[cfg(feature = "vad")]
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct VadConfig {
    /// VAD method to use.
    pub method: VadMethod,
    /// Frame size in samples.
    pub frame_size: NonZeroUsize,
    /// Hop size in samples (frame step).
    pub hop_size: NonZeroUsize,
    /// Whether to include a final partial frame (zero-padded).
    pub pad_end: bool,
    /// Policy for multi-channel audio.
    pub channel_policy: VadChannelPolicy,

    /// Energy threshold in dBFS (RMS). Typical values: `-60.0` (very sensitive) to `-30.0`.
    pub energy_threshold_db: f64,
    /// Minimum acceptable zero crossing rate (ZCR), expressed as crossings per sample in `[0, 1]`.
    pub zcr_min: f64,
    /// Maximum acceptable zero crossing rate (ZCR), expressed as crossings per sample in `[0, 1]`.
    pub zcr_max: f64,

    /// Minimum number of consecutive speech frames to keep a speech region.
    pub min_speech_frames: usize,
    /// Minimum number of consecutive non-speech frames to keep a silence region.
    /// Shorter silence gaps are filled as speech.
    pub min_silence_frames: usize,
    /// Hangover in frames: keep speech active for this many frames after energy drops.
    pub hangover_frames: NonZeroUsize,
    /// Majority-vote smoothing window in frames (1 = no smoothing).
    pub smooth_frames: NonZeroUsize,

    /// Lower bound of the speech band in Hz (used by `VadMethod::Spectral`).
    pub speech_band_low_hz: f64,
    /// Upper bound of the speech band in Hz (used by `VadMethod::Spectral`).
    pub speech_band_high_hz: f64,
    /// Threshold on speech-band energy ratio (used by `VadMethod::Spectral`).
    pub spectral_ratio_threshold: f64,
}

#[cfg(feature = "vad")]
impl VadConfig {
    /// Creates a VAD configuration with every parameter specified explicitly.
    ///
    /// Prefer the [`Default`] impl or [`energy_only`][VadConfig::energy_only] for
    /// typical use cases. Use this constructor only when full control over every
    /// parameter is required.
    ///
    /// # Arguments
    ///
    /// – `method` – detection algorithm.\
    /// – `frame_size` – analysis window length in samples.\
    /// – `hop_size` – step between consecutive frames; must be ≤ `frame_size`.\
    /// – `pad_end` – whether to include a final zero-padded partial frame.\
    /// – `channel_policy` – how to derive activity from multi-channel input.\
    /// – `energy_threshold_db` – RMS threshold in dBFS; typical range `[-60.0, -20.0]`.\
    /// – `zcr_min` – minimum zero-crossing rate for voiced activity; in `[0.0, 1.0]`.\
    /// – `zcr_max` – maximum zero-crossing rate for voiced activity; in `[zcr_min, 1.0]`.\
    /// – `min_speech_frames` – minimum consecutive speech frames to confirm a region.\
    /// – `min_silence_frames` – minimum consecutive silence frames to close a region.\
    /// – `hangover_frames` – frames to remain "active" after energy drops below threshold.\
    /// – `smooth_frames` – majority-vote window size; `1` disables smoothing.\
    /// – `speech_band_low_hz` – lower boundary of the speech band (Spectral method).\
    /// – `speech_band_high_hz` – upper boundary of the speech band (Spectral method).\
    /// – `spectral_ratio_threshold` – minimum in-band energy fraction; in `[0.0, 1.0]`.
    ///
    /// # Returns
    ///
    /// A [`VadConfig`] with all fields set as provided. Use
    /// [`validate`][VadConfig::validate] to check constraints before use.
    #[inline]
    #[must_use]
    pub const fn new(
        method: VadMethod,
        frame_size: NonZeroUsize,
        hop_size: NonZeroUsize,
        pad_end: bool,
        channel_policy: VadChannelPolicy,
        energy_threshold_db: f64,
        zcr_min: f64,
        zcr_max: f64,
        min_speech_frames: usize,
        min_silence_frames: usize,
        hangover_frames: NonZeroUsize,
        smooth_frames: NonZeroUsize,
        speech_band_low_hz: f64,
        speech_band_high_hz: f64,
        spectral_ratio_threshold: f64,
    ) -> Self {
        Self {
            method,
            frame_size,
            hop_size,
            pad_end,
            channel_policy,
            energy_threshold_db,
            zcr_min,
            zcr_max,
            min_speech_frames,
            min_silence_frames,
            hangover_frames,
            smooth_frames,
            speech_band_low_hz,
            speech_band_high_hz,
            spectral_ratio_threshold,
        }
    }

    /// Creates a VAD configuration using only energy-based (RMS) detection.
    ///
    /// All parameters are taken from the [`Default`] impl; only `method` is
    /// overridden to [`VadMethod::Energy`]. This is the simplest and fastest
    /// configuration and works well for clean or lightly noisy speech.
    ///
    /// # Returns
    ///
    /// A [`VadConfig`] identical to `Default::default()` except that `method` is
    /// `VadMethod::Energy`.
    #[inline]
    #[must_use]
    pub fn energy_only() -> Self {
        Self {
            method: VadMethod::Energy,
            ..Default::default()
        }
    }

    /// Validates all parameter constraints.
    ///
    /// # Returns
    ///
    /// `Ok(self)` if all constraints are satisfied.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter][crate::AudioSampleError] describing the
    /// first violated constraint:
    ///
    /// - `hop_size` ≤ `frame_size`
    /// - `0.0 ≤ zcr_min ≤ zcr_max ≤ 1.0`
    /// - `0.0 < speech_band_low_hz < speech_band_high_hz`
    /// - `0.0 ≤ spectral_ratio_threshold ≤ 1.0`
    #[inline]
    pub fn validate(self) -> AudioSampleResult<Self> {
        if self.hop_size > self.frame_size {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "hop_size",
                "must be <= frame_size",
            )));
        }
        if self.zcr_min < 0.0 || self.zcr_max > 1.0 || self.zcr_min > self.zcr_max {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "zcr_*",
                "expected 0 <= zcr_min <= zcr_max <= 1",
            )));
        }
        if self.speech_band_low_hz <= 0.0
            || self.speech_band_high_hz <= 0.0
            || self.speech_band_low_hz >= self.speech_band_high_hz
        {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "speech_band_*",
                "expected 0 < low_hz < high_hz",
            )));
        }
        if self.spectral_ratio_threshold < 0.0 || self.spectral_ratio_threshold > 1.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "spectral_ratio_threshold",
                "expected 0 <= threshold <= 1",
            )));
        }

        Ok(self)
    }

    /// Returns a copy of `self` with the specified `method`.
    ///
    /// Other fields are unchanged.
    #[inline]
    #[must_use]
    pub const fn with_method(self, method: VadMethod) -> Self {
        Self { method, ..self }
    }

    /// Returns a copy of `self` with the specified `frame_size`.
    ///
    /// Other fields are unchanged.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter][crate::AudioSampleError] if
    /// `frame_size` is less than the current `hop_size`.
    #[inline]
    pub fn with_frame_size(self, frame_size: NonZeroUsize) -> AudioSampleResult<Self> {
        let updated = Self { frame_size, ..self };
        if updated.hop_size > updated.frame_size {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "frame_size",
                "must be >= hop_size",
            )));
        }
        Ok(updated)
    }

    /// Returns a copy of `self` with the specified `hop_size`.
    ///
    /// Other fields are unchanged.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter][crate::AudioSampleError] if
    /// `hop_size` is greater than the current `frame_size`.
    #[inline]
    pub fn with_hop_size(self, hop_size: NonZeroUsize) -> AudioSampleResult<Self> {
        let updated = Self { hop_size, ..self };
        if updated.hop_size > updated.frame_size {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "hop_size",
                "must be <= frame_size",
            )));
        }
        Ok(updated)
    }

    /// Returns a copy of `self` with the specified `pad_end`.
    ///
    /// Other fields are unchanged.
    #[inline]
    #[must_use]
    pub const fn with_pad_end(self, pad_end: bool) -> Self {
        Self { pad_end, ..self }
    }

    /// Returns a copy of `self` with the specified `channel_policy`.
    ///
    /// Other fields are unchanged.
    #[inline]
    #[must_use]
    pub const fn with_channel_policy(self, channel_policy: VadChannelPolicy) -> Self {
        Self {
            channel_policy,
            ..self
        }
    }

    /// Returns a copy of `self` with the specified `energy_threshold_db`.
    ///
    /// Other fields are unchanged.
    #[inline]
    #[must_use]
    pub const fn with_energy_threshold_db(self, energy_threshold_db: f64) -> Self {
        Self {
            energy_threshold_db,
            ..self
        }
    }

    /// Returns a copy of `self` with the specified `zcr_min`.
    ///
    /// Other fields are unchanged.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter][crate::AudioSampleError] if
    /// `zcr_min` is not in the range `[0.0, 1.0]` or is greater than the current `zcr_max`.
    #[inline]
    pub fn with_zcr_min(self, zcr_min: f64) -> AudioSampleResult<Self> {
        let updated = Self { zcr_min, ..self };
        if updated.zcr_min < 0.0 || updated.zcr_min > 1.0 || updated.zcr_min > updated.zcr_max {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "zcr_min",
                "expected 0 <= zcr_min <= zcr_max <= 1",
            )));
        }
        Ok(updated)
    }

    /// Returns a copy of `self` with the specified `zcr_max`.
    ///
    /// Other fields are unchanged.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter][crate::AudioSampleError] if
    /// `zcr_max` is not in the range `[0.0, 1.0]` or is less than the current `zcr_min`.
    #[inline]
    pub fn with_zcr_max(self, zcr_max: f64) -> AudioSampleResult<Self> {
        let updated = Self { zcr_max, ..self };
        if updated.zcr_max < 0.0 || updated.zcr_max > 1.0 || updated.zcr_min > updated.zcr_max {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "zcr_max",
                "expected 0 <= zcr_min <= zcr_max <= 1",
            )));
        }
        Ok(updated)
    }

    /// Returns a copy of `self` with the specified `min_speech_frames`.
    ///
    /// Other fields are unchanged.
    #[inline]
    #[must_use]
    pub const fn with_min_speech_frames(self, min_speech_frames: usize) -> Self {
        Self {
            min_speech_frames,
            ..self
        }
    }

    /// Returns a copy of `self` with the specified `min_silence_frames`.
    ///
    /// Other fields are unchanged.
    #[inline]
    #[must_use]
    pub const fn with_min_silence_frames(self, min_silence_frames: usize) -> Self {
        Self {
            min_silence_frames,
            ..self
        }
    }

    /// Returns a copy of `self` with the specified `hangover_frames`.
    ///
    /// Other fields are unchanged.
    #[inline]
    #[must_use]
    pub const fn with_hangover_frames(self, hangover_frames: NonZeroUsize) -> Self {
        Self {
            hangover_frames,
            ..self
        }
    }

    /// Returns a copy of `self` with the specified `smooth_frames`.
    ///
    /// Other fields are unchanged.
    #[inline]
    #[must_use]
    pub const fn with_smooth_frames(self, smooth_frames: NonZeroUsize) -> Self {
        Self {
            smooth_frames,
            ..self
        }
    }

    /// Returns a copy of `self` with the specified `speech_band_low_hz`.
    ///
    /// Other fields are unchanged.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter][crate::AudioSampleError] if
    /// `speech_band_low_hz` is not positive or is not less than the current `speech_band_high_hz`.
    #[inline]
    pub fn with_speech_band_low_hz(self, speech_band_low_hz: f64) -> AudioSampleResult<Self> {
        let updated = Self {
            speech_band_low_hz,
            ..self
        };
        if updated.speech_band_low_hz <= 0.0
            || updated.speech_band_low_hz >= updated.speech_band_high_hz
        {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "speech_band_low_hz",
                "expected 0 < low_hz < high_hz",
            )));
        }
        Ok(updated)
    }

    /// Returns a copy of `self` with the specified `speech_band_high_hz`.
    ///
    /// Other fields are unchanged.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter][crate::AudioSampleError] if
    /// `speech_band_high_hz` is not positive or is not greater than the current `speech_band_low_hz`.
    #[inline]
    pub fn with_speech_band_high_hz(self, speech_band_high_hz: f64) -> AudioSampleResult<Self> {
        let updated = Self {
            speech_band_high_hz,
            ..self
        };
        if updated.speech_band_high_hz <= 0.0
            || updated.speech_band_low_hz >= updated.speech_band_high_hz
        {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "speech_band_high_hz",
                "expected 0 < low_hz < high_hz",
            )));
        }
        Ok(updated)
    }

    /// Returns a copy of `self` with the specified `spectral_ratio_threshold`.
    ///
    /// Other fields are unchanged.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter][crate::AudioSampleError] if
    /// `spectral_ratio_threshold` is not in the range `[0.0, 1.0]`.
    #[inline]
    pub fn with_spectral_ratio_threshold(
        self,
        spectral_ratio_threshold: f64,
    ) -> AudioSampleResult<Self> {
        let updated = Self {
            spectral_ratio_threshold,
            ..self
        };
        if updated.spectral_ratio_threshold < 0.0 || updated.spectral_ratio_threshold > 1.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "spectral_ratio_threshold",
                "expected 0 <= threshold <= 1",
            )));
        }
        Ok(updated)
    }
}

#[cfg(feature = "vad")]
impl Default for VadConfig {
    fn default() -> Self {
        Self::new(
            VadMethod::Energy,
            crate::nzu!(1024),
            crate::nzu!(512),
            true,
            VadChannelPolicy::AverageToMono,
            -40.0,          // energy threshold in dBFS
            0.02,           // zcr_min
            0.3,            // zcr_max
            3,              // min_speech_frames
            5,              // min_silence_frames
            crate::nzu!(2), // hangover_frames
            crate::nzu!(5), // smooth_frames
            300.0,          // speech_band_low_hz
            3400.0,         // speech_band_high_hz
            0.6,            // spectral_ratio_threshold
        )
    }
}

/// Discrete quality levels for resampling operations.
///
/// Higher quality levels trade increased computational cost and latency
/// for improved spectral fidelity, reduced aliasing, and more stable phase
/// behaviour. Lower quality levels prioritise throughput and low latency.
#[cfg(feature = "resampling")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[non_exhaustive]
pub enum ResamplingQuality {
    /// Fast but lower quality resampling.
    #[default]
    Fast,
    /// Balanced speed and quality.
    Medium,
    /// Highest quality but slower resampling.
    High,
}

#[cfg(feature = "resampling")]
impl Display for ResamplingQuality {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::Fast => "fast",
            Self::Medium => "medium",
            Self::High => "high",
        };
        f.write_str(s)
    }
}

#[cfg(feature = "resampling")]
impl FromStr for ResamplingQuality {
    type Err = AudioSampleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalised = s.trim().to_ascii_lowercase();

        match normalised.as_str() {
            "fast" | "low" => Ok(Self::Fast),
            "medium" | "med" | "balanced" => Ok(Self::Medium),
            "high" | "best" => Ok(Self::High),
            _ => Err(AudioSampleError::parse::<Self, _>(format!(
                "Failed to parse {}. Got {}, must be one of {:?}",
                std::any::type_name::<Self>(),
                s,
                ["fast", "low", "medium", "med", "balanced", "high", "best"]
            ))),
        }
    }
}

#[cfg(feature = "resampling")]
impl TryFrom<&str> for ResamplingQuality {
    type Error = AudioSampleError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        value.parse()
    }
}

/// Scaling methods for spectrogram magnitude and frequency representations.
///
/// Different scaling approaches expose different structure in spectral content
/// and are appropriate for different analysis and visualisation tasks.
#[cfg(feature = "transforms")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum SpectrogramScale {
    /// Linear power scale.
    ///
    /// Preserves absolute magnitude relationships and is most appropriate
    /// for quantitative analysis and energy measurements.
    #[default]
    Linear,

    /// Logarithmic (decibel) magnitude scale.
    ///
    /// Compresses dynamic range to improve visualisation of low-energy
    /// components alongside strong spectral peaks.
    ///
    /// Note: The exact reference level and floor behaviour are implementation
    /// details and should not be relied upon for semantic correctness.
    Log,

    /// Mel-frequency scale.
    ///
    /// Applies a perceptually motivated nonlinear mapping of frequency
    /// designed to better approximate human auditory resolution.
    Mel,
}

#[cfg(feature = "transforms")]
impl Display for SpectrogramScale {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::Linear => "linear",
            Self::Log => "log",
            Self::Mel => "mel",
        };
        f.write_str(s)
    }
}

#[cfg(feature = "transforms")]
impl FromStr for SpectrogramScale {
    type Err = AudioSampleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalised = s.trim().to_ascii_lowercase();

        match normalised.as_str() {
            // Linear
            "linear" | "lin" => Ok(Self::Linear),

            // Logarithmic / dB
            "log" | "logarithmic" | "db" | "decibel" => Ok(Self::Log),

            // Mel
            "mel" | "mel-scale" | "melscale" => Ok(Self::Mel),

            _ => Err(AudioSampleError::parse::<Self, _>(format!(
                "Failed to parse {}. Got {}, must be one of {:?}",
                std::any::type_name::<Self>(),
                s,
                ["linear", "lin", "log", "db", "decibel", "mel", "mel-scale",]
            ))),
        }
    }
}

#[cfg(feature = "transforms")]
impl TryFrom<&str> for SpectrogramScale {
    type Error = AudioSampleError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        value.parse()
    }
}

/// Pitch detection algorithm selection.
///
/// Different algorithms trade off accuracy, robustness to noise and
/// inharmonicity, latency, and computational cost when estimating the
/// fundamental frequency of a signal.
#[cfg(feature = "pitch-analysis")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum PitchDetectionMethod {
    /// YIN pitch detection algorithm.
    ///
    /// Provides robust and accurate fundamental frequency estimation for
    /// both speech and musical signals, at moderate computational cost.
    #[default]
    Yin,

    /// Autocorrelation-based pitch detection.
    ///
    /// Simple and fast, but sensitive to noise and octave errors for
    /// complex or weakly periodic signals.
    Autocorrelation,

    /// Cepstral pitch detection.
    ///
    /// Operates in the frequency domain and performs well for voiced speech,
    /// but can degrade for dense harmonic or noisy spectra.
    Cepstrum,

    /// Harmonic Product Spectrum (HPS).
    ///
    /// Emphasises harmonic structure and is well-suited to musical signals
    /// with strong harmonic content.
    HarmonicProduct,
}

#[cfg(feature = "pitch-analysis")]
impl Display for PitchDetectionMethod {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::Yin => "yin",
            Self::Autocorrelation => "autocorrelation",
            Self::Cepstrum => "cepstrum",
            Self::HarmonicProduct => "harmonic_product",
        };
        f.write_str(s)
    }
}

#[cfg(feature = "pitch-analysis")]
impl FromStr for PitchDetectionMethod {
    type Err = AudioSampleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalised = s.trim().to_ascii_lowercase();

        match normalised.as_str() {
            // YIN
            "yin" => Ok(Self::Yin),

            // Autocorrelation
            "autocorrelation" | "auto" | "acf" => Ok(Self::Autocorrelation),

            // Cepstrum
            "cepstrum" | "cep" => Ok(Self::Cepstrum),

            // Harmonic Product Spectrum
            "harmonic_product" | "harmonic-product" | "hps" | "harmonic" => {
                Ok(Self::HarmonicProduct)
            }

            _ => Err(AudioSampleError::parse::<Self, _>(format!(
                "Failed to parse {}. Got {}, must be one of {:?}",
                std::any::type_name::<Self>(),
                s,
                [
                    "yin",
                    "autocorrelation",
                    "acf",
                    "cepstrum",
                    "cep",
                    "harmonic_product",
                    "hps",
                ]
            ))),
        }
    }
}

#[cfg(feature = "pitch-analysis")]
impl TryFrom<&str> for PitchDetectionMethod {
    type Error = AudioSampleError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        value.parse()
    }
}

/// IIR filter family selection for digital signal processing.
///
/// IIR (Infinite Impulse Response) filters provide efficient recursive
/// implementations with feedback, typically achieving sharper transition
/// bands than FIR filters for a given computational budget.
#[cfg(feature = "iir-filtering")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum IirFilterType {
    /// Butterworth filter.
    ///
    /// Exhibits a maximally flat passband with monotonic magnitude response
    /// and no ripple. Provides smooth behaviour and predictable phase
    /// characteristics at the cost of wider transition bands.
    #[default]
    Butterworth,

    /// Chebyshev Type I filter.
    ///
    /// Introduces controlled ripple in the passband to achieve a sharper
    /// transition region than Butterworth designs.
    ChebyshevI,

    /// Chebyshev Type II filter.
    ///
    /// Introduces ripple in the stopband while preserving a monotonic
    /// passband response, allowing sharper transitions than Butterworth.
    ChebyshevII,

    /// Elliptic (Cauer) filter.
    ///
    /// Introduces ripple in both passband and stopband, yielding the
    /// steepest transition region for a given filter order.
    Elliptic,
}

#[cfg(feature = "iir-filtering")]
impl Display for IirFilterType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::Butterworth => "butterworth",
            Self::ChebyshevI => "chebyshev1",
            Self::ChebyshevII => "chebyshev2",
            Self::Elliptic => "elliptic",
        };
        f.write_str(s)
    }
}

#[cfg(feature = "iir-filtering")]
impl FromStr for IirFilterType {
    type Err = AudioSampleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalised = s.trim().to_ascii_lowercase();

        match normalised.as_str() {
            // Butterworth
            "butterworth" | "butter" | "bw" => Ok(Self::Butterworth),
            "chebyshev1" | "cheby1" | "chebyshev_i" | "chebyshev-i" => Ok(Self::ChebyshevI),
            "chebyshev2" | "cheby2" | "chebyshev_ii" | "chebyshev-ii" => Ok(Self::ChebyshevII),
            "elliptic" | "cauer" | "ellip" => Ok(Self::Elliptic),
            _ => Err(AudioSampleError::parse::<Self, _>(format!(
                "Failed to parse {}. Got {}, must be one of {:?}",
                std::any::type_name::<Self>(),
                s,
                [
                    "butterworth",
                    "chebyshev1",
                    "chebyshev2",
                    "elliptic",
                    "cauer",
                ]
            ))),
        }
    }
}

#[cfg(feature = "iir-filtering")]
impl TryFrom<&str> for IirFilterType {
    type Error = AudioSampleError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        value.parse()
    }
}

/// Filter response characteristics.
///
/// Defines the qualitative frequency response shape of a filter.
#[cfg(feature = "iir-filtering")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum FilterResponse {
    /// Low-pass filter.
    ///
    /// Attenuates frequencies above the cutoff frequency while preserving
    /// lower-frequency components.
    #[default]
    LowPass,

    /// High-pass filter.
    ///
    /// Attenuates frequencies below the cutoff frequency while preserving
    /// higher-frequency components.
    HighPass,

    /// Band-pass filter.
    ///
    /// Preserves frequencies within a specified band while attenuating
    /// frequencies outside that range.
    BandPass,

    /// Band-stop (notch) filter.
    ///
    /// Attenuates frequencies within a specified band while preserving
    /// frequencies outside that range.
    BandStop,
}

#[cfg(feature = "iir-filtering")]
impl Display for FilterResponse {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::LowPass => "lowpass",
            Self::HighPass => "highpass",
            Self::BandPass => "bandpass",
            Self::BandStop => "bandstop",
        };
        f.write_str(s)
    }
}

#[cfg(feature = "iir-filtering")]
impl FromStr for FilterResponse {
    type Err = AudioSampleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalised = s.trim().to_ascii_lowercase();

        match normalised.as_str() {
            "lowpass" | "low-pass" | "lp" => Ok(Self::LowPass),
            "highpass" | "high-pass" | "hp" => Ok(Self::HighPass),
            "bandpass" | "band-pass" | "bp" => Ok(Self::BandPass),
            "bandstop" | "band-stop" | "bs" | "notch" => Ok(Self::BandStop),

            _ => Err(AudioSampleError::parse::<Self, _>(format!(
                "Failed to parse {}. Got {}, must be one of {:?}",
                std::any::type_name::<Self>(),
                s,
                [
                    "lowpass", "highpass", "bandpass", "bandstop", "lp", "hp", "bp", "bs",
                ]
            ))),
        }
    }
}

#[cfg(feature = "iir-filtering")]
impl TryFrom<&str> for FilterResponse {
    type Error = AudioSampleError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        value.parse()
    }
}

/// Design specification for an IIR digital filter.
///
/// ## Purpose
///
/// Encapsulates all parameters needed to specify a single IIR filter stage:
/// the filter family, response shape, order, cutoff frequencies, and ripple
/// or attenuation tolerances.
///
/// ## Intended Usage
///
/// Use one of the provided constructors (`butterworth_lowpass`, etc.) to obtain
/// a valid design for the most common cases. Pass the result to
/// [`AudioIirFiltering::apply_iir_filter`][crate::operations::AudioIirFiltering::apply_iir_filter].
///
/// ## Invariants
///
/// - For `LowPass` / `HighPass` responses: `cutoff_frequency` must be `Some`.
/// - For `BandPass` / `BandStop` responses: `low_frequency` and
///   `high_frequency` must both be `Some`, with `low_frequency < high_frequency`.
/// - For `ChebyshevI` and `Elliptic`: `passband_ripple` must be `Some` and
///   positive.
/// - For `ChebyshevII` and `Elliptic`: `stopband_attenuation` must be `Some`
///   and positive.
/// - All frequencies must be positive and below the Nyquist frequency of the
///   target sample rate.
///
/// ## Assumptions
///
/// Validation of frequency values against the sample rate occurs at filter
/// application time, not at construction.
#[cfg(feature = "iir-filtering")]
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct IirFilterDesign {
    /// The filter family (Butterworth, Chebyshev I/II, Elliptic).
    pub filter_type: IirFilterType,

    /// The frequency response shape (low-pass, high-pass, band-pass, band-stop).
    pub response: FilterResponse,

    /// Filter order — the number of poles.
    ///
    /// Higher order yields a steeper transition band at the cost of more
    /// computation and potentially less stable phase behaviour.
    pub order: NonZeroUsize,

    /// Cutoff frequency in Hz for `LowPass` and `HighPass` responses.
    ///
    /// Must be `Some` and in `(0, nyquist)` for those response types; otherwise
    /// `None`.
    pub cutoff_frequency: Option<f64>,

    /// Lower edge frequency in Hz for `BandPass` and `BandStop` responses.
    ///
    /// Must be `Some` and less than `high_frequency` for those response types;
    /// otherwise `None`.
    pub low_frequency: Option<f64>,

    /// Upper edge frequency in Hz for `BandPass` and `BandStop` responses.
    ///
    /// Must be `Some` and greater than `low_frequency` for those response types;
    /// otherwise `None`.
    pub high_frequency: Option<f64>,

    /// Passband ripple in dB for `ChebyshevI` and `Elliptic` filter types.
    ///
    /// Controls how much gain variation is permitted within the passband.
    /// Larger values yield a steeper transition but more audible ripple.
    /// Must be positive. `None` for filter types that do not use this parameter.
    pub passband_ripple: Option<f64>,

    /// Stopband attenuation in dB for `ChebyshevII` and `Elliptic` filter types.
    ///
    /// Minimum attenuation guaranteed in the stopband. Larger values yield
    /// stronger rejection but require a higher filter order to achieve.
    /// Must be positive. `None` for filter types that do not use this parameter.
    pub stopband_attenuation: Option<f64>,
}

#[cfg(feature = "iir-filtering")]
impl IirFilterDesign {
    /// Creates a Butterworth low-pass filter design.
    ///
    /// # Arguments
    ///
    /// – `order` – filter order; higher values yield a steeper roll-off.\
    /// – `cutoff_frequency` – –3 dB point in Hz; must be in `(0, nyquist)`.
    ///
    /// # Returns
    ///
    /// An [`IirFilterDesign`] with `filter_type = Butterworth` and
    /// `response = LowPass`.
    #[inline]
    #[must_use]
    pub const fn butterworth_lowpass(order: NonZeroUsize, cutoff_frequency: f64) -> Self {
        Self {
            filter_type: IirFilterType::Butterworth,
            response: FilterResponse::LowPass,
            order,
            cutoff_frequency: Some(cutoff_frequency),
            low_frequency: None,
            high_frequency: None,
            passband_ripple: None,
            stopband_attenuation: None,
        }
    }

    /// Creates a Butterworth high-pass filter design.
    ///
    /// # Arguments
    ///
    /// – `order` – filter order; higher values yield a steeper roll-off.\
    /// – `cutoff_frequency` – –3 dB point in Hz; must be in `(0, nyquist)`.
    ///
    /// # Returns
    ///
    /// An [`IirFilterDesign`] with `filter_type = Butterworth` and
    /// `response = HighPass`.
    #[inline]
    #[must_use]
    pub const fn butterworth_highpass(order: NonZeroUsize, cutoff_frequency: f64) -> Self {
        Self {
            filter_type: IirFilterType::Butterworth,
            response: FilterResponse::HighPass,
            order,
            cutoff_frequency: Some(cutoff_frequency),
            low_frequency: None,
            high_frequency: None,
            passband_ripple: None,
            stopband_attenuation: None,
        }
    }

    /// Creates a Butterworth band-pass filter design.
    ///
    /// # Arguments
    ///
    /// – `order` – filter order per edge; total order is `2 × order` for a band-pass.\
    /// – `low_frequency` – lower –3 dB edge in Hz; must be positive and less than `high_frequency`.\
    /// – `high_frequency` – upper –3 dB edge in Hz; must be less than the Nyquist frequency.
    ///
    /// # Returns
    ///
    /// An [`IirFilterDesign`] with `filter_type = Butterworth` and `response = BandPass`.
    #[inline]
    #[must_use]
    pub const fn butterworth_bandpass(
        order: NonZeroUsize,
        low_frequency: f64,
        high_frequency: f64,
    ) -> Self {
        Self {
            filter_type: IirFilterType::Butterworth,
            response: FilterResponse::BandPass,
            order,
            cutoff_frequency: None,
            low_frequency: Some(low_frequency),
            high_frequency: Some(high_frequency),
            passband_ripple: None,
            stopband_attenuation: None,
        }
    }

    /// Creates a Chebyshev Type I filter design.
    ///
    /// Chebyshev Type I filters achieve a steeper transition than Butterworth
    /// designs of the same order by allowing controlled ripple in the passband.
    ///
    /// # Arguments
    ///
    /// – `response` – frequency response shape (`LowPass` or `HighPass`).\
    /// – `order` – filter order; higher values yield a steeper roll-off.\
    /// – `cutoff_frequency` – passband edge frequency in Hz.\
    /// – `passband_ripple` – peak-to-peak ripple in dB within the passband;
    ///   must be positive. Typical values are 0.5–3.0 dB.
    ///
    /// # Returns
    ///
    /// An [`IirFilterDesign`] with `filter_type = ChebyshevI`.
    #[inline]
    #[must_use]
    pub const fn chebyshev_i(
        response: FilterResponse,
        order: NonZeroUsize,
        cutoff_frequency: f64,
        passband_ripple: f64,
    ) -> Self {
        Self {
            filter_type: IirFilterType::ChebyshevI,
            response,
            order,
            cutoff_frequency: Some(cutoff_frequency),
            low_frequency: None,
            high_frequency: None,
            passband_ripple: Some(passband_ripple),
            stopband_attenuation: None,
        }
    }
}

/// Parametric equaliser band types.
///
/// Each band type defines how gain is applied across the frequency spectrum
/// relative to a centre or cutoff frequency.
#[cfg(feature = "parametric-eq")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum EqBandType {
    /// Peaking (bell) filter.
    ///
    /// Boosts or attenuates a narrow frequency region centred at the target
    /// frequency. Positive gain produces a peak; negative gain produces a notch.
    #[default]
    Peak,

    /// Low-shelf filter.
    ///
    /// Applies a broadband boost or cut to frequencies below the corner
    /// frequency.
    LowShelf,

    /// High-shelf filter.
    ///
    /// Applies a broadband boost or cut to frequencies above the corner
    /// frequency.
    HighShelf,

    /// Low-pass filter.
    ///
    /// Attenuates frequencies above the cutoff frequency.
    LowPass,

    /// High-pass filter.
    ///
    /// Attenuates frequencies below the cutoff frequency.
    HighPass,

    /// Band-pass filter.
    ///
    /// Preserves frequencies within a specified band while attenuating
    /// frequencies outside that range.
    BandPass,

    /// Band-stop (notch) filter.
    ///
    /// Attenuates frequencies within a specified band while preserving
    /// frequencies outside that range.
    BandStop,
}

#[cfg(feature = "parametric-eq")]
impl Display for EqBandType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::Peak => "peak",
            Self::LowShelf => "low_shelf",
            Self::HighShelf => "high_shelf",
            Self::LowPass => "lowpass",
            Self::HighPass => "highpass",
            Self::BandPass => "bandpass",
            Self::BandStop => "bandstop",
        };
        f.write_str(s)
    }
}

#[cfg(feature = "parametric-eq")]
impl FromStr for EqBandType {
    type Err = AudioSampleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalised = s.trim().to_ascii_lowercase();

        match normalised.as_str() {
            "peak" | "bell" | "notch" => Ok(Self::Peak),
            "low_shelf" | "low-shelf" | "lowshelf" | "ls" => Ok(Self::LowShelf),
            "high_shelf" | "high-shelf" | "highshelf" | "hs" => Ok(Self::HighShelf),
            "lowpass" | "low-pass" | "lp" => Ok(Self::LowPass),
            "highpass" | "high-pass" | "hp" => Ok(Self::HighPass),
            "bandpass" | "band-pass" | "bp" => Ok(Self::BandPass),
            "bandstop" | "band-stop" | "bs" => Ok(Self::BandStop),
            _ => Err(AudioSampleError::parse::<Self, _>(format!(
                "Failed to parse {}. Got {}, must be one of {:?}",
                std::any::type_name::<Self>(),
                s,
                [
                    "peak",
                    "bell",
                    "low_shelf",
                    "high_shelf",
                    "lowpass",
                    "highpass",
                    "bandpass",
                    "bandstop",
                ]
            ))),
        }
    }
}

#[cfg(feature = "parametric-eq")]
impl TryFrom<&str> for EqBandType {
    type Error = AudioSampleError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        value.parse()
    }
}

/// A single band in a parametric equaliser.
///
/// ## Purpose
///
/// Describes one frequency-shaping stage: its type (peak, shelf, pass filter),
/// the target frequency, the amount of gain or attenuation, and the bandwidth.
///
/// ## Intended Usage
///
/// Construct with one of the type-specific helpers (`peak`, `low_shelf`, etc.)
/// and add to a [`ParametricEq`] instance. Individual bands can be toggled
/// without removing them via [`set_enabled`][EqBand::set_enabled]. Call
/// [`validate`][EqBand::validate] with the target sample rate to check that
/// frequency and gain values are within acceptable limits.
///
/// ## Invariants
///
/// - `frequency` must be in `(0, nyquist)` where `nyquist = sample_rate / 2`.
/// - `q_factor` must be positive.
/// - `gain_db` must be within `[-40.0, 40.0]` dB.
///
/// ## Assumptions
///
/// The `gain_db` field is ignored by `LowPass` and `HighPass` band types;
/// those bands always set `gain_db = 0.0` at construction.
#[cfg(feature = "parametric-eq")]
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct EqBand {
    /// Shape of the frequency response for this band.
    pub band_type: EqBandType,

    /// Centre frequency for peak/notch bands, or corner frequency for shelves
    /// and pass filters, in Hz.
    pub frequency: f64,

    /// Gain in dB applied within the active region.
    ///
    /// Positive values boost; negative values cut. Ignored by `LowPass` and
    /// `HighPass` band types (always 0.0 for those). Valid range: `[-40.0, 40.0]`.
    pub gain_db: f64,

    /// Quality factor controlling bandwidth.
    ///
    /// Higher Q → narrower affected frequency region; lower Q → wider region.
    /// For shelf filters this controls the shelf slope. Must be positive.
    pub q_factor: f64,

    /// Whether this band is currently active.
    ///
    /// When `false`, the band is bypassed and contributes no gain change.
    pub enabled: bool,
}

#[cfg(feature = "parametric-eq")]
impl EqBand {
    /// Creates a peak (bell) EQ band.
    ///
    /// Boosts or attenuates a frequency region centred at `frequency`. Positive
    /// `gain_db` creates a peak; negative `gain_db` creates a dip.
    ///
    /// # Arguments
    ///
    /// – `frequency` – centre frequency in Hz; must be in `(0, nyquist)`.\
    /// – `gain_db` – gain at the centre frequency; positive = boost, negative = cut.\
    /// – `q_factor` – bandwidth control; higher Q = narrower bell.
    ///
    /// # Returns
    ///
    /// An enabled [`EqBand`] with `band_type = EqBandType::Peak`.
    #[inline]
    #[must_use]
    pub const fn peak(frequency: f64, gain_db: f64, q_factor: f64) -> Self {
        Self {
            band_type: EqBandType::Peak,
            frequency,
            gain_db,
            q_factor,
            enabled: true,
        }
    }

    /// Creates a low-shelf EQ band.
    ///
    /// Applies a broadband boost or cut to all frequencies below `frequency`.
    ///
    /// # Arguments
    ///
    /// – `frequency` – corner (transition) frequency in Hz.\
    /// – `gain_db` – gain applied below the corner; positive = boost, negative = cut.\
    /// – `q_factor` – shelf slope control; 0.707 gives a Butterworth-style shelf.
    ///
    /// # Returns
    ///
    /// An enabled [`EqBand`] with `band_type = EqBandType::LowShelf`.
    #[inline]
    #[must_use]
    pub const fn low_shelf(frequency: f64, gain_db: f64, q_factor: f64) -> Self {
        Self {
            band_type: EqBandType::LowShelf,
            frequency,
            gain_db,
            q_factor,
            enabled: true,
        }
    }

    /// Creates a high-shelf EQ band.
    ///
    /// Applies a broadband boost or cut to all frequencies above `frequency`.
    ///
    /// # Arguments
    ///
    /// – `frequency` – corner (transition) frequency in Hz.\
    /// – `gain_db` – gain applied above the corner; positive = boost, negative = cut.\
    /// – `q_factor` – shelf slope control; 0.707 gives a Butterworth-style shelf.
    ///
    /// # Returns
    ///
    /// An enabled [`EqBand`] with `band_type = EqBandType::HighShelf`.
    #[inline]
    #[must_use]
    pub const fn high_shelf(frequency: f64, gain_db: f64, q_factor: f64) -> Self {
        Self {
            band_type: EqBandType::HighShelf,
            frequency,
            gain_db,
            q_factor,
            enabled: true,
        }
    }

    /// Creates a low-pass filter band.
    ///
    /// Attenuates frequencies above `frequency`. No gain is applied; the band
    /// acts as a filter rather than a boost/cut stage.
    ///
    /// # Arguments
    ///
    /// – `frequency` – cutoff frequency in Hz.\
    /// – `q_factor` – filter resonance; 0.707 gives a Butterworth (maximally flat) response.
    ///
    /// # Returns
    ///
    /// An enabled [`EqBand`] with `band_type = EqBandType::LowPass` and `gain_db = 0.0`.
    #[inline]
    #[must_use]
    pub const fn low_pass(frequency: f64, q_factor: f64) -> Self {
        Self {
            band_type: EqBandType::LowPass,
            frequency,
            gain_db: 0.0,
            q_factor,
            enabled: true,
        }
    }

    /// Creates a high-pass filter band.
    ///
    /// Attenuates frequencies below `frequency`. No gain is applied; the band
    /// acts as a filter rather than a boost/cut stage.
    ///
    /// # Arguments
    ///
    /// – `frequency` – cutoff frequency in Hz.\
    /// – `q_factor` – filter resonance; 0.707 gives a Butterworth (maximally flat) response.
    ///
    /// # Returns
    ///
    /// An enabled [`EqBand`] with `band_type = EqBandType::HighPass` and `gain_db = 0.0`.
    #[inline]
    #[must_use]
    pub const fn high_pass(frequency: f64, q_factor: f64) -> Self {
        Self {
            band_type: EqBandType::HighPass,
            frequency,
            gain_db: 0.0,
            q_factor,
            enabled: true,
        }
    }

    /// Enables or disables this band.
    ///
    /// Disabled bands are bypassed by the EQ processor without being removed.
    ///
    /// # Arguments
    ///
    /// – `enabled` – `true` to activate this band; `false` to bypass it.
    #[inline]
    pub const fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Returns `true` if this band is active.
    #[inline]
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Validates that the band parameters are consistent with `sample_rate`.
    ///
    /// # Arguments
    ///
    /// – `sample_rate` – the target sample rate in Hz; used to derive the Nyquist limit.
    ///
    /// # Returns
    ///
    /// `Ok(self)` if all constraints are satisfied.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter][crate::AudioSampleError] if:
    ///
    /// - `frequency` is not in `(0, nyquist)`
    /// - `q_factor` is not positive
    /// - `gain_db` has absolute value greater than 40 dB
    #[inline]
    pub fn validate(self, sample_rate: f64) -> AudioSampleResult<Self> {
        let nyquist = sample_rate / 2.0;

        if self.frequency <= 0.0 || self.frequency >= nyquist {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "frequency",
                format!("{} Hz", self.frequency),
                "0",
                format!("{nyquist}"),
                "Frequency must be between 0 and Nyquist frequency",
            )));
        }

        if self.q_factor <= 0.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "q_factor",
                "Q factor must be positive",
            )));
        }

        // Check reasonable gain limits
        if self.gain_db.abs() > 40.0 {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "gain_db",
                format!("{} dB", self.gain_db),
                "-40",
                "40",
                "Gain must be within reasonable range",
            )));
        }

        Ok(self)
    }
}

/// A multi-band parametric equaliser configuration.
///
/// ## Purpose
///
/// Holds an ordered collection of [`EqBand`] stages along with a global output
/// gain and a bypass flag. Bands are processed in the order they appear in
/// `bands`.
///
/// ## Intended Usage
///
/// Build the EQ using [`new`][ParametricEq::new] and [`add_band`][ParametricEq::add_band],
/// or use the [`three_band`][ParametricEq::three_band] / [`five_band`][ParametricEq::five_band]
/// presets as a starting point. Pass the result to
/// [`AudioParametricEq::apply_parametric_eq`][crate::operations::AudioParametricEq::apply_parametric_eq].
/// Call [`validate`][ParametricEq::validate] with the target sample rate to check
/// all band parameters before processing.
///
/// ## Invariants
///
/// - Each band in `bands` must satisfy the [`EqBand`] invariants (validated by
///   [`validate`][ParametricEq::validate]).
/// - `output_gain_db` is an unclamped linear gain applied after all bands.
///
/// ## Assumptions
///
/// Bands are applied sequentially; the order in `bands` affects the result for
/// non-minimum-phase chains.
#[cfg(feature = "parametric-eq")]
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ParametricEq {
    /// Ordered sequence of EQ bands to apply.
    pub bands: Vec<EqBand>,

    /// Output gain in dB applied after all bands.
    ///
    /// Positive values boost the overall output; negative values attenuate it.
    /// Set to `0.0` for unity gain.
    pub output_gain_db: f64,

    /// When `true`, all band processing is skipped and the input passes through unchanged.
    pub bypassed: bool,
}

#[cfg(feature = "parametric-eq")]
impl ParametricEq {
    /// Creates an empty parametric EQ with no bands and unity output gain.
    ///
    /// # Returns
    ///
    /// A [`ParametricEq`] with an empty `bands` vector, `output_gain_db = 0.0`,
    /// and `bypassed = false`.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            bands: Vec::new(),
            output_gain_db: 0.0,
            bypassed: false,
        }
    }

    /// Appends a band to the end of the processing chain.
    ///
    /// # Arguments
    ///
    /// – `band` – the [`EqBand`] to append.
    #[inline]
    pub fn add_band(&mut self, band: EqBand) {
        self.bands.push(band);
    }

    /// Removes the band at `index` and returns it, or `None` if out of range.
    ///
    /// # Arguments
    ///
    /// – `index` – zero-based position in `bands`.
    ///
    /// # Returns
    ///
    /// `Some(band)` if `index < band_count()`, otherwise `None`.
    #[inline]
    pub fn remove_band(&mut self, index: usize) -> Option<EqBand> {
        if index < self.bands.len() {
            Some(self.bands.remove(index))
        } else {
            None
        }
    }

    /// Returns a shared reference to the band at `index`, or `None` if out of range.
    ///
    /// # Arguments
    ///
    /// – `index` – zero-based position in `bands`.
    #[inline]
    #[must_use]
    pub fn get_band(&self, index: usize) -> Option<&EqBand> {
        self.bands.get(index)
    }

    /// Returns an exclusive reference to the band at `index`, or `None` if out of range.
    ///
    /// # Arguments
    ///
    /// – `index` – zero-based position in `bands`.
    #[inline]
    pub fn get_band_mut(&mut self, index: usize) -> Option<&mut EqBand> {
        self.bands.get_mut(index)
    }

    /// Returns the number of bands currently in the EQ.
    #[inline]
    #[must_use]
    pub const fn band_count(&self) -> usize {
        self.bands.len()
    }

    /// Sets the overall output gain applied after all bands.
    ///
    /// # Arguments
    ///
    /// – `gain_db` – gain in dB; `0.0` is unity, positive values boost, negative values attenuate.
    #[inline]
    pub const fn set_output_gain(&mut self, gain_db: f64) {
        self.output_gain_db = gain_db;
    }

    /// Sets whether the entire EQ is bypassed.
    ///
    /// When bypassed, the input signal passes through without modification.
    ///
    /// # Arguments
    ///
    /// – `bypassed` – `true` to bypass; `false` to enable processing.
    #[inline]
    pub const fn set_bypassed(&mut self, bypassed: bool) {
        self.bypassed = bypassed;
    }

    /// Returns `true` if the EQ is currently bypassed.
    #[inline]
    #[must_use]
    pub const fn is_bypassed(&self) -> bool {
        self.bypassed
    }

    /// Validates all bands against `sample_rate`.
    ///
    /// # Arguments
    ///
    /// – `sample_rate` – the target sample rate in Hz.
    ///
    /// # Returns
    ///
    /// `Ok(self)` if every band passes [`EqBand::validate`].
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter][crate::AudioSampleError] identifying the
    /// index and error of the first failing band.
    #[inline]
    pub fn validate(self, sample_rate: f64) -> AudioSampleResult<Self> {
        for (i, band) in self.bands.iter().enumerate() {
            match band.validate(sample_rate) {
                Ok(_) => {}
                Err(er) => {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "band",
                        format!("Band {}/{} validation error: {}", i, self.bands.len(), er),
                    )));
                }
            }
        }
        Ok(self)
    }

    /// Creates a three-band EQ preset: low shelf, mid peak, high shelf.
    ///
    /// # Arguments
    ///
    /// – `low_freq` – corner frequency of the low shelf in Hz.\
    /// – `low_gain` – gain of the low shelf in dB.\
    /// – `mid_freq` – centre frequency of the mid peak in Hz.\
    /// – `mid_gain` – gain of the mid peak in dB.\
    /// – `mid_q` – Q factor of the mid peak band.\
    /// – `high_freq` – corner frequency of the high shelf in Hz.\
    /// – `high_gain` – gain of the high shelf in dB.
    ///
    /// # Returns
    ///
    /// A [`ParametricEq`] with three bands: low shelf at Q 0.707, mid peak, high shelf at Q 0.707.
    #[inline]
    #[must_use]
    pub fn three_band(
        low_freq: f64,
        low_gain: f64,
        mid_freq: f64,
        mid_gain: f64,
        mid_q: f64,
        high_freq: f64,
        high_gain: f64,
    ) -> Self {
        let mut eq = Self::new();
        eq.add_band(EqBand::low_shelf(low_freq, low_gain, 0.707));
        eq.add_band(EqBand::peak(mid_freq, mid_gain, mid_q));
        eq.add_band(EqBand::high_shelf(high_freq, high_gain, 0.707));
        eq
    }

    /// Creates a five-band EQ preset with all gains at 0 dB.
    ///
    /// Bands: 100 Hz low shelf, 300 Hz peak, 1 kHz peak, 3 kHz peak, 8 kHz high shelf.
    /// All gains are initialised to 0.0 dB; adjust individual bands to taste.
    ///
    /// # Returns
    ///
    /// A [`ParametricEq`] with five bands at unity gain.
    #[inline]
    #[must_use]
    pub fn five_band() -> Self {
        let mut eq = Self::new();
        eq.add_band(EqBand::low_shelf(100.0, 0.0, 0.707));
        eq.add_band(EqBand::peak(300.0, 0.0, 1.0));
        eq.add_band(EqBand::peak(1000.0, 0.0, 1.0));
        eq.add_band(EqBand::peak(3000.0, 0.0, 1.0));
        eq.add_band(EqBand::high_shelf(8000.0, 0.0, 0.707));
        eq
    }
}

#[cfg(feature = "parametric-eq")]
impl Default for ParametricEq {
    fn default() -> Self {
        Self::new()
    }
}

/// Knee characteristic for dynamic range processing.
///
/// Controls how smoothly gain reduction transitions as the signal crosses
/// the threshold.
#[cfg(any(feature = "parametric-eq", feature = "dynamic-range"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum KneeType {
    /// Hard knee.
    ///
    /// Applies an abrupt transition at the threshold, yielding precise
    /// dynamics control at the potential cost of audible artefacts.
    Hard,

    /// Soft knee.
    ///
    /// Applies a gradual transition around the threshold, producing smoother
    /// and more perceptually natural behaviour.
    #[default]
    Soft,
}

#[cfg(any(feature = "parametric-eq", feature = "dynamic-range"))]
impl Display for KneeType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::Hard => "hard",
            Self::Soft => "soft",
        };
        f.write_str(s)
    }
}

#[cfg(any(feature = "parametric-eq", feature = "dynamic-range"))]
impl FromStr for KneeType {
    type Err = AudioSampleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalised = s.trim().to_ascii_lowercase();

        match normalised.as_str() {
            "hard" | "hard-knee" | "hardknee" => Ok(Self::Hard),
            "soft" | "soft-knee" | "softknee" => Ok(Self::Soft),
            _ => Err(AudioSampleError::parse::<Self, _>(format!(
                "Failed to parse {}. Got {}, must be one of {:?}",
                std::any::type_name::<Self>(),
                s,
                [
                    "hard",
                    "hard-knee",
                    "hardknee",
                    "soft",
                    "soft-knee",
                    "softknee"
                ]
            ))),
        }
    }
}

#[cfg(any(feature = "parametric-eq", feature = "dynamic-range"))]
impl TryFrom<&str> for KneeType {
    type Error = AudioSampleError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        value.parse()
    }
}

/// Detection method for dynamic range processing.
///
/// Determines how signal level is estimated for driving gain reduction.
#[cfg(feature = "dynamic-range")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum DynamicRangeMethod {
    /// RMS-based level detection.
    ///
    /// Estimates average signal power over time, producing smoother and more
    /// perceptually stable gain control.
    #[default]
    Rms,

    /// Peak-based level detection.
    ///
    /// Responds to instantaneous signal peaks, providing tight peak control
    /// with increased sensitivity to transients.
    Peak,

    /// Hybrid level detection.
    ///
    /// Combines RMS and peak estimation to balance smoothness and transient
    /// control.
    Hybrid,
}

#[cfg(feature = "dynamic-range")]
impl Display for DynamicRangeMethod {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::Rms => "rms",
            Self::Peak => "peak",
            Self::Hybrid => "hybrid",
        };
        f.write_str(s)
    }
}

#[cfg(feature = "dynamic-range")]
impl FromStr for DynamicRangeMethod {
    type Err = AudioSampleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalised = s.trim().to_ascii_lowercase();

        match normalised.as_str() {
            "rms" | "average" | "avg" => Ok(Self::Rms),
            "peak" | "pk" => Ok(Self::Peak),
            "hybrid" | "mixed" | "combo" => Ok(Self::Hybrid),
            _ => Err(AudioSampleError::parse::<Self, _>(format!(
                "Failed to parse {}. Got {}, must be one of {:?}",
                std::any::type_name::<Self>(),
                s,
                [
                    "rms", "average", "avg", "peak", "pk", "hybrid", "mixed", "combo"
                ]
            ))),
        }
    }
}

#[cfg(feature = "dynamic-range")]
impl TryFrom<&str> for DynamicRangeMethod {
    type Error = AudioSampleError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        value.parse()
    }
}

/// Side-chain configuration for dynamic range processors.
///
/// ## Purpose
///
/// Controls whether gain reduction is driven by the main signal (internal) or
/// by an external control signal, and applies optional frequency shaping to the
/// control path.
///
/// ## Intended Usage
///
/// Attach to [`CompressorConfig::side_chain`] or [`LimiterConfig::side_chain`].
/// Use [`disabled`][SideChainConfig::disabled] (the default) for standard operation.
/// Use [`enabled`][SideChainConfig::enabled] or call [`enable`][SideChainConfig::enable]
/// when side-chain triggering is required. Apply a high-pass filter to the side-chain
/// via [`set_high_pass`][SideChainConfig::set_high_pass] to reduce low-frequency pumping.
///
/// ## Invariants
///
/// - `high_pass_freq`, if `Some`, must be in `(0, nyquist)`.
/// - `low_pass_freq`, if `Some`, must be in `(0, nyquist)`.
/// - If both are `Some`, `high_pass_freq < low_pass_freq`.
/// - `external_mix` must be in `[0.0, 1.0]`.
#[cfg(feature = "dynamic-range")]
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct SideChainConfig {
    /// Whether side-chain processing is active.
    ///
    /// When `false` the processor uses its own output as the gain-reduction signal.
    pub enabled: bool,

    /// Optional high-pass filter on the side-chain signal, in Hz.
    ///
    /// Filtering out low frequencies prevents bass content from triggering
    /// excessive gain reduction ("pumping"). Must be in `(0, nyquist)` if `Some`.
    pub high_pass_freq: Option<f64>,

    /// Optional low-pass filter on the side-chain signal, in Hz.
    ///
    /// Limits the frequency range that can trigger gain reduction.
    /// Must be in `(0, nyquist)` and greater than `high_pass_freq` if both are `Some`.
    pub low_pass_freq: Option<f64>,

    /// Pre-emphasis gain applied to the side-chain signal in dB.
    ///
    /// Positive values make specific frequencies trigger gain reduction more
    /// easily. `0.0` disables pre-emphasis.
    pub pre_emphasis_db: f64,

    /// Mix between the internal signal (`0.0`) and external side-chain (`1.0`).
    ///
    /// Must be in `[0.0, 1.0]`. Values between 0 and 1 blend both sources.
    pub external_mix: f64,
}

#[cfg(feature = "dynamic-range")]
impl SideChainConfig {
    /// Creates a disabled side-chain configuration (the default).
    ///
    /// The processor uses its own output as the gain-detection signal.
    ///
    /// # Returns
    ///
    /// A [`SideChainConfig`] with `enabled = false` and no filter frequencies.
    #[inline]
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            high_pass_freq: None,
            low_pass_freq: None,
            pre_emphasis_db: 0.0,
            external_mix: 0.0,
        }
    }

    /// Creates an enabled side-chain configuration with a default 100 Hz high-pass filter.
    ///
    /// The 100 Hz high-pass filter reduces bass-driven pumping artefacts.
    ///
    /// # Returns
    ///
    /// A [`SideChainConfig`] with `enabled = true`, `high_pass_freq = Some(100.0)`,
    /// and `external_mix = 1.0`.
    #[inline]
    #[must_use]
    pub const fn enabled() -> Self {
        Self {
            enabled: true,
            high_pass_freq: Some(100.0),
            low_pass_freq: None,
            pre_emphasis_db: 0.0,
            external_mix: 1.0,
        }
    }

    /// Activates side-chain processing.
    #[inline]
    pub const fn enable(&mut self) {
        self.enabled = true;
    }

    /// Deactivates side-chain processing.
    #[inline]
    pub const fn disable(&mut self) {
        self.enabled = false;
    }

    /// Sets the high-pass filter frequency applied to the side-chain signal.
    ///
    /// # Arguments
    ///
    /// – `freq` – cut-off frequency in Hz; must be in `(0, nyquist)`.
    #[inline]
    pub const fn set_high_pass(&mut self, freq: f64) {
        self.high_pass_freq = Some(freq);
    }

    /// Sets the low-pass filter frequency applied to the side-chain signal.
    ///
    /// # Arguments
    ///
    /// – `freq` – cut-off frequency in Hz; must be in `(0, nyquist)` and greater than
    ///   `high_pass_freq` if that is also set.
    #[inline]
    pub const fn set_low_pass(&mut self, freq: f64) {
        self.low_pass_freq = Some(freq);
    }

    /// Validates all side-chain parameters against `sample_rate`.
    ///
    /// # Arguments
    ///
    /// – `sample_rate` – the target sample rate in Hz.
    ///
    /// # Returns
    ///
    /// `Ok(self)` if all constraints are satisfied.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter][crate::AudioSampleError] if any frequency
    /// is outside `(0, nyquist)`, if `high_pass_freq ≥ low_pass_freq`, or if
    /// `external_mix` is outside `[0.0, 1.0]`.
    #[inline]
    pub fn validate(self, sample_rate: f64) -> AudioSampleResult<Self> {
        if let Some(hp_freq) = self.high_pass_freq
            && (hp_freq <= 0.0 || hp_freq >= sample_rate / 2.0)
        {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "high_pass_freq",
                "High-pass frequency must be between 0 and Nyquist frequency",
            )));
        }

        if let Some(lp_freq) = self.low_pass_freq
            && (lp_freq <= 0.0 || lp_freq >= sample_rate / 2.0)
        {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "low_pass_freq",
                "Low-pass frequency must be between 0 and Nyquist frequency",
            )));
        }

        if let (Some(hp), Some(lp)) = (self.high_pass_freq, self.low_pass_freq)
            && (hp >= lp)
        {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "high_pass_freq",
                "High-pass frequency must be less than low-pass frequency",
            )));
        }

        if self.external_mix < 0.0 || self.external_mix > 1.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "external_mix",
                "External mix must be between0.0 and1.0",
            )));
        }

        Ok(self)
    }
}

#[cfg(feature = "dynamic-range")]
impl Default for SideChainConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Configuration for a dynamic range compressor.
///
/// ## Purpose
///
/// Specifies all parameters governing the compressor's gain-reduction behaviour:
/// threshold, ratio, envelope times, knee shape, level detection, side-chain,
/// and lookahead.
///
/// ## Intended Usage
///
/// Use [`CompressorConfig::new`] (which delegates to [`Default`]) or one of the
/// presets (`vocal`, `drum`, `bus`) as a starting point. Call
/// [`validate`][CompressorConfig::validate] to check parameter bounds before
/// passing to a compressor function.
///
/// ## Invariants
///
/// - `threshold_db` must be ≤ 0.
/// - `ratio` must be ≥ 1.0.
/// - `attack_ms` must be in `[0.01, 1000.0]` ms.
/// - `release_ms` must be in `[1.0, 10000.0]` ms.
/// - `|makeup_gain_db|` must be ≤ 40 dB.
/// - `knee_width_db` must be in `[0.0, 20.0]` dB.
/// - `lookahead_ms` must be in `[0.0, 20.0]` ms.
/// - `side_chain` must satisfy its own invariants.
#[cfg(feature = "dynamic-range")]
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct CompressorConfig {
    /// Level above which gain reduction begins, in dBFS.
    ///
    /// Must be ≤ 0. Typical range: `[-40.0, 0.0]`.
    pub threshold_db: f64,

    /// Ratio of input level change to output level change above the threshold.
    ///
    /// `1.0` = no compression; `∞` (very large) = limiting. Must be ≥ 1.0.
    /// Typical values: `2.0` (light) to `10.0` (heavy).
    pub ratio: f64,

    /// Time for the compressor to reach full gain reduction after a threshold
    /// crossing, in milliseconds.
    ///
    /// Shorter attack catches transients; longer attack lets them through.
    /// Valid range: `[0.01, 1000.0]` ms.
    pub attack_ms: f64,

    /// Time for the compressor to return to unity gain after the signal drops
    /// below the threshold, in milliseconds.
    ///
    /// Shorter release may produce pumping artefacts; longer release sounds
    /// more transparent. Valid range: `[1.0, 10000.0]` ms.
    pub release_ms: f64,

    /// Gain added after compression to compensate for level reduction, in dB.
    ///
    /// Positive values restore loudness lost during gain reduction.
    /// Valid range: `[-40.0, 40.0]` dB.
    pub makeup_gain_db: f64,

    /// Transition shape at the threshold.
    pub knee_type: KneeType,

    /// Width of the soft-knee transition region around the threshold, in dB.
    ///
    /// Only meaningful when `knee_type = KneeType::Soft`. `0.0` is equivalent
    /// to a hard knee. Valid range: `[0.0, 20.0]` dB.
    pub knee_width_db: f64,

    /// Algorithm used to estimate the signal level for gain-reduction decisions.
    pub detection_method: DynamicRangeMethod,

    /// Side-chain routing and filtering configuration.
    pub side_chain: SideChainConfig,

    /// Lookahead delay in milliseconds.
    ///
    /// When greater than `0.0`, the compressor buffers the input and begins
    /// gain reduction before a peak is reached, improving transparency.
    /// Introduces latency equal to `lookahead_ms`. Valid range: `[0.0, 20.0]` ms.
    pub lookahead_ms: f64,
}

#[cfg(feature = "dynamic-range")]
impl CompressorConfig {
    /// Creates a compressor configuration using default values.
    ///
    /// Equivalent to [`CompressorConfig::default()`]: threshold −12 dBFS,
    /// ratio 4:1, attack 5 ms, release 50 ms, soft knee, no makeup gain.
    ///
    /// # Returns
    ///
    /// A [`CompressorConfig`] suitable as a general-purpose starting point.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A preset optimised for lead vocals.
    ///
    /// Moderate ratio and soft knee for transparent vocal control.
    /// Threshold −18 dBFS, ratio 3:1, attack 2 ms, release 100 ms.
    ///
    /// # Returns
    ///
    /// A [`CompressorConfig`] tuned for typical vocal dynamics.
    #[inline]
    #[must_use]
    pub const fn vocal() -> Self {
        Self {
            threshold_db: -18.0,
            ratio: 3.0,
            attack_ms: 2.0,
            release_ms: 100.0,
            makeup_gain_db: 3.0,
            knee_type: KneeType::Soft,
            knee_width_db: 4.0,
            detection_method: DynamicRangeMethod::Rms,
            side_chain: SideChainConfig::disabled(),
            lookahead_ms: 0.0,
        }
    }

    /// A preset optimised for drums and percussive material.
    ///
    /// High ratio, fast attack and release, hard knee for punchy transient
    /// control. Threshold −8 dBFS, ratio 6:1, attack 0.1 ms, release 20 ms.
    ///
    /// # Returns
    ///
    /// A [`CompressorConfig`] tuned for percussive signals.
    #[inline]
    #[must_use]
    pub const fn drum() -> Self {
        Self {
            threshold_db: -8.0,
            ratio: 6.0,
            attack_ms: 0.1,
            release_ms: 20.0,
            makeup_gain_db: 2.0,
            knee_type: KneeType::Hard,
            knee_width_db: 0.5,
            detection_method: DynamicRangeMethod::Peak,
            side_chain: SideChainConfig::disabled(),
            lookahead_ms: 0.0,
        }
    }

    /// A preset for mix-bus (glue) compression.
    ///
    /// Low ratio, slow attack and release, wide soft knee for transparent
    /// cohesion across a full mix. Threshold −20 dBFS, ratio 2:1, attack 10 ms,
    /// release 200 ms.
    ///
    /// # Returns
    ///
    /// A [`CompressorConfig`] tuned for bus compression.
    #[inline]
    #[must_use]
    pub const fn bus() -> Self {
        Self {
            threshold_db: -20.0,
            ratio: 2.0,
            attack_ms: 10.0,
            release_ms: 200.0,
            makeup_gain_db: 1.0,
            knee_type: KneeType::Soft,
            knee_width_db: 6.0,
            detection_method: DynamicRangeMethod::Rms,
            side_chain: SideChainConfig::disabled(),
            lookahead_ms: 0.0,
        }
    }

    /// Validates all compressor parameters.
    ///
    /// # Arguments
    ///
    /// – `sample_rate` – target sample rate in Hz; forwarded to side-chain validation.
    ///
    /// # Returns
    ///
    /// `Ok(self)` if all parameter bounds are satisfied.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter][crate::AudioSampleError] for the first
    /// violated constraint (threshold, ratio, attack, release, makeup gain, knee
    /// width, lookahead, or side-chain).
    #[inline]
    pub fn validate(self, sample_rate: f64) -> AudioSampleResult<Self> {
        if self.threshold_db > 0.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "threshold_db",
                "Threshold should be negative (below 0 dB)",
            )));
        }

        if self.ratio < 1.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "ratio",
                "Ratio must be1.0 or greater",
            )));
        }

        if self.attack_ms < 0.01 || self.attack_ms > 1000.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Attack time must be between 0.01 and 1000 ms",
            )));
        }

        if self.release_ms < 1.0 || self.release_ms > 10000.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Release time must be between1.0 and 10000 ms",
            )));
        }

        if self.makeup_gain_db.abs() > 40.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Makeup gain must be between -40.0 and +40.0 dB",
            )));
        }

        if self.knee_width_db < 0.0 || self.knee_width_db > 20.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Knee width must be between0.0 and 20.0 dB",
            )));
        }

        if self.lookahead_ms < 0.0 || self.lookahead_ms > 20.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Lookahead time must be between0.0 and 20.0 ms",
            )));
        }

        self.side_chain.validate(sample_rate)?;
        Ok(self)
    }
}

#[cfg(feature = "dynamic-range")]
impl Default for CompressorConfig {
    fn default() -> Self {
        Self {
            threshold_db: -12.0,
            ratio: 4.0,
            attack_ms: 5.0,
            release_ms: 50.0,
            makeup_gain_db: 0.0,
            knee_type: KneeType::Soft,
            knee_width_db: 2.0,
            detection_method: DynamicRangeMethod::Rms,
            side_chain: SideChainConfig::disabled(),
            lookahead_ms: 0.0,
        }
    }
}

/// Configuration for a true-peak limiter.
///
/// ## Purpose
///
/// Specifies all parameters controlling the limiter's ceiling enforcement:
/// the maximum output level, envelope times, knee shape, detection method,
/// side-chain routing, lookahead, and optional inter-sample peak (ISP) limiting.
///
/// ## Intended Usage
///
/// Use one of the presets (`transparent`, `mastering`, `broadcast`) or build
/// from scratch with [`LimiterConfig::new`]. Call
/// [`validate`][LimiterConfig::validate] before passing to a limiter function.
///
/// ## Invariants
///
/// - `ceiling_db` must be ≤ 0.
/// - `attack_ms` must be in `[0.001, 100.0]` ms.
/// - `release_ms` must be in `[1.0, 10000.0]` ms.
/// - `knee_width_db` must be in `[0.0, 10.0]` dB.
/// - `lookahead_ms` must be in `[0.0, 20.0]` ms.
/// - `side_chain` must satisfy its own invariants.
#[cfg(feature = "dynamic-range")]
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct LimiterConfig {
    /// Maximum allowable output level in dBFS.
    ///
    /// No output sample will exceed this level when limiting is active.
    /// Must be ≤ 0. Typical range: `[-3.0, -0.1]`.
    pub ceiling_db: f64,

    /// Time for the limiter to react to a signal approaching the ceiling, in ms.
    ///
    /// Very short attack (< 1 ms) enables brick-wall peak control.
    /// Valid range: `[0.001, 100.0]` ms.
    pub attack_ms: f64,

    /// Time to return to unity gain after the signal recedes, in ms.
    ///
    /// Valid range: `[1.0, 10000.0]` ms.
    pub release_ms: f64,

    /// Transition shape at the ceiling.
    pub knee_type: KneeType,

    /// Width of the soft-knee transition region in dB.
    ///
    /// Only meaningful when `knee_type = KneeType::Soft`. Valid range: `[0.0, 10.0]` dB.
    pub knee_width_db: f64,

    /// Algorithm used to estimate peak or RMS level for gain-reduction decisions.
    pub detection_method: DynamicRangeMethod,

    /// Side-chain routing and filtering configuration.
    pub side_chain: SideChainConfig,

    /// Lookahead delay in milliseconds.
    ///
    /// Buffers audio and begins gain reduction ahead of approaching peaks,
    /// enabling cleaner limiting at the cost of added latency.
    /// Valid range: `[0.0, 20.0]` ms.
    pub lookahead_ms: f64,

    /// Whether inter-sample peak (ISP) limiting is enabled.
    ///
    /// When `true`, the limiter accounts for peaks that may exceed 0 dBFS when
    /// the digital signal is reconstructed via a DAC, preventing audible
    /// clipping on playback. Recommended for mastering and broadcast delivery.
    pub isp_limiting: bool,
}

#[cfg(feature = "dynamic-range")]
impl LimiterConfig {
    /// Creates a limiter configuration with every parameter specified explicitly.
    ///
    /// # Arguments
    ///
    /// – `ceiling_db` – maximum output level in dBFS; must be ≤ 0.\
    /// – `attack_ms` – attack time in milliseconds; must be in `[0.001, 100.0]`.\
    /// – `release_ms` – release time in milliseconds; must be in `[1.0, 10000.0]`.\
    /// – `knee_type` – transition shape at the ceiling.\
    /// – `knee_width_db` – soft-knee width in dB; in `[0.0, 10.0]`.\
    /// – `detection_method` – level estimation algorithm.\
    /// – `lookahead_ms` – lookahead delay in milliseconds; in `[0.0, 20.0]`.\
    /// – `isp_filtering` – whether inter-sample peak protection is enabled.
    ///
    /// # Returns
    ///
    /// A [`LimiterConfig`] with `side_chain` set to `SideChainConfig::disabled()`.
    /// Use [`validate`][LimiterConfig::validate] to verify constraints.
    #[inline]
    #[must_use]
    pub const fn new(
        ceiling_db: f64,
        attack_ms: f64,
        release_ms: f64,
        knee_type: KneeType,
        knee_width_db: f64,
        detection_method: DynamicRangeMethod,
        lookahead_ms: f64,
        isp_filtering: bool,
    ) -> Self {
        Self {
            ceiling_db,
            attack_ms,
            release_ms,
            knee_type,
            knee_width_db,
            detection_method,
            side_chain: SideChainConfig::disabled(),
            lookahead_ms,
            isp_limiting: isp_filtering,
        }
    }

    /// A preset for low-audibility peak limiting.
    ///
    /// Very fast attack and soft knee for transparent operation at −0.1 dBFS.
    /// Suitable for final-stage limiting where audible artefacts are undesirable.
    ///
    /// # Returns
    ///
    /// A [`LimiterConfig`] with ceiling −0.1 dBFS, attack 0.1 ms, release 100 ms,
    /// 5 ms lookahead, and ISP limiting enabled.
    #[inline]
    #[must_use]
    pub const fn transparent() -> Self {
        Self {
            ceiling_db: -0.1,
            attack_ms: 0.1,
            release_ms: 100.0,
            knee_type: KneeType::Soft,
            knee_width_db: 2.0,
            detection_method: DynamicRangeMethod::Peak,
            side_chain: SideChainConfig::disabled(),
            lookahead_ms: 5.0,
            isp_limiting: true,
        }
    }

    /// A preset for mastering-grade limiting.
    ///
    /// Hybrid detection and a longer lookahead for optimal loudness management
    /// at −0.3 dBFS. ISP limiting enabled for streaming and physical delivery.
    ///
    /// # Returns
    ///
    /// A [`LimiterConfig`] with ceiling −0.3 dBFS, attack 1 ms, release 200 ms,
    /// 10 ms lookahead, and ISP limiting enabled.
    #[inline]
    #[must_use]
    pub const fn mastering() -> Self {
        Self {
            ceiling_db: -0.3,
            attack_ms: 1.0,
            release_ms: 200.0,
            knee_type: KneeType::Soft,
            knee_width_db: 3.0,
            detection_method: DynamicRangeMethod::Hybrid,
            side_chain: SideChainConfig::disabled(),
            lookahead_ms: 10.0,
            isp_limiting: true,
        }
    }

    /// A preset for broadcast delivery compliance.
    ///
    /// Hard knee and fast attack enforce a strict −1.0 dBFS ceiling suitable
    /// for loudness-normalised delivery formats. ISP limiting enabled.
    ///
    /// # Returns
    ///
    /// A [`LimiterConfig`] with ceiling −1.0 dBFS, attack 0.5 ms, release 50 ms,
    /// 2 ms lookahead, and ISP limiting enabled.
    #[inline]
    #[must_use]
    pub const fn broadcast() -> Self {
        Self {
            ceiling_db: -1.0,
            attack_ms: 0.5,
            release_ms: 50.0,
            knee_type: KneeType::Hard,
            knee_width_db: 0.5,
            detection_method: DynamicRangeMethod::Peak,
            side_chain: SideChainConfig::disabled(),
            lookahead_ms: 2.0,
            isp_limiting: true,
        }
    }

    /// Validates all limiter parameters.
    ///
    /// # Arguments
    ///
    /// – `sample_rate` – target sample rate in Hz; forwarded to side-chain validation.
    ///
    /// # Returns
    ///
    /// `Ok(self)` if all constraints hold.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter][crate::AudioSampleError] for the first
    /// violated constraint (ceiling, attack, release, knee width, lookahead, or
    /// side-chain).
    #[inline]
    pub fn validate(self, sample_rate: f64) -> AudioSampleResult<Self> {
        if self.ceiling_db > 0.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Ceiling should be negative (below 0 dB)",
            )));
        }

        if self.attack_ms < 0.001 || self.attack_ms > 100.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Attack time must be between 0.001 and 100 ms",
            )));
        }

        if self.release_ms < 1.0 || self.release_ms > 10000.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Release time must be between1.0 and 10000 ms",
            )));
        }

        if self.knee_width_db < 0.0 || self.knee_width_db > 10.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Knee width must be between0.0 and 10.0 dB",
            )));
        }

        if self.lookahead_ms < 0.0 || self.lookahead_ms > 20.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Lookahead time must be between0.0 and 20.0 ms",
            )));
        }

        self.side_chain.validate(sample_rate)?;
        Ok(self)
    }
}

#[cfg(feature = "dynamic-range")]
impl Default for LimiterConfig {
    fn default() -> Self {
        Self {
            ceiling_db: -0.1,
            attack_ms: 0.5,
            release_ms: 50.0,
            knee_type: KneeType::Soft,
            knee_width_db: 1.0,
            detection_method: DynamicRangeMethod::Peak,
            side_chain: SideChainConfig::disabled(),
            lookahead_ms: 2.0,
            isp_limiting: true,
        }
    }
}

/// Adaptive thresholding strategy for peak picking.
///
/// Determines how dynamic detection thresholds are estimated from the
/// onset strength function over time.
#[cfg(feature = "peak-picking")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum AdaptiveThresholdMethod {
    /// Delta-based adaptive threshold.
    ///
    /// Tracks local maxima and applies a fixed offset to determine the
    /// detection threshold. Responds quickly to rapid changes but can be
    /// sensitive to noise and transient outliers.
    Delta,

    /// Percentile-based adaptive threshold.
    ///
    /// Estimates the threshold from rolling distribution statistics of the
    /// onset strength function, yielding increased robustness at the cost
    /// of slower adaptation.
    #[default]
    Percentile,

    /// Combined adaptive threshold.
    ///
    /// Combines delta-based and percentile-based thresholds to balance
    /// responsiveness and robustness across a wide range of signals.
    Combined,
}

#[cfg(feature = "peak-picking")]
impl Display for AdaptiveThresholdMethod {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::Delta => "delta",
            Self::Percentile => "percentile",
            Self::Combined => "combined",
        };
        f.write_str(s)
    }
}

#[cfg(feature = "peak-picking")]
impl FromStr for AdaptiveThresholdMethod {
    type Err = AudioSampleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalised = s.trim().to_ascii_lowercase();

        match normalised.as_str() {
            "delta" | "offset" => Ok(Self::Delta),
            "percentile" | "quantile" | "pct" => Ok(Self::Percentile),
            "combined" | "hybrid" | "mixed" => Ok(Self::Combined),
            _ => Err(AudioSampleError::parse::<Self, _>(format!(
                "Failed to parse {}. Got {}, must be one of {:?}",
                std::any::type_name::<Self>(),
                s,
                [
                    "delta",
                    "offset",
                    "percentile",
                    "quantile",
                    "pct",
                    "combined",
                    "hybrid",
                    "mixed"
                ]
            ))),
        }
    }
}

#[cfg(feature = "peak-picking")]
impl TryFrom<&str> for AdaptiveThresholdMethod {
    type Error = AudioSampleError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        value.parse()
    }
}

/// Parameters for adaptive threshold estimation in peak picking.
///
/// ## Purpose
///
/// Controls how a dynamic detection threshold is derived from the local
/// statistics of the onset strength function. A well-tuned adaptive threshold
/// adapts to varying signal conditions and reduces both false positives and
/// missed detections compared to a fixed threshold.
///
/// ## Intended Usage
///
/// Use one of the type-specific constructors (`delta`, `percentile`, `combined`)
/// rather than `new` for most scenarios. Attach the result to
/// [`PeakPickingConfig::adaptive_threshold`]. Call
/// [`validate`][AdaptiveThresholdConfig::validate] to check parameter bounds.
///
/// ## Invariants
///
/// - `delta` must be ≥ 0.0.
/// - `percentile` must be in `[0.0, 1.0]`.
/// - `window_size` must be > 0.
/// - `min_threshold` must be ≥ 0.0.
/// - `max_threshold` must be > `min_threshold`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg(feature = "peak-picking")]
#[non_exhaustive]
pub struct AdaptiveThresholdConfig {
    /// Algorithm used to derive the adaptive threshold.
    pub method: AdaptiveThresholdMethod,

    /// Fixed offset added to local maxima for delta-based thresholding.
    ///
    /// Larger values raise the threshold and reduce false positives at the
    /// cost of potentially missing weak onsets. Must be ≥ 0.0.
    /// Typical range: `[0.01, 0.1]`.
    pub delta: f64,

    /// Percentile of the local distribution used as the threshold.
    ///
    /// Higher percentiles (closer to 1.0) yield a more conservative threshold
    /// and fewer detections. Must be in `[0.0, 1.0]`.
    pub percentile: f64,

    /// Length of the local analysis window in samples.
    ///
    /// Larger windows produce more stable thresholds but respond more slowly
    /// to changes in signal dynamics. Must be > 0.
    pub window_size: usize,

    /// Floor below which the threshold will never drop.
    ///
    /// Prevents over-sensitivity when the onset strength is very low.
    /// Must be ≥ 0.0 and less than `max_threshold`.
    pub min_threshold: f64,

    /// Ceiling above which the threshold will never rise.
    ///
    /// Prevents under-sensitivity when the onset strength is very high.
    /// Must be > `min_threshold`.
    pub max_threshold: f64,
}

#[cfg(feature = "peak-picking")]
impl AdaptiveThresholdConfig {
    /// Creates an adaptive threshold configuration with all fields specified.
    ///
    /// Prefer the type-specific constructors (`delta`, `percentile`, `combined`)
    /// for typical usage.
    ///
    /// # Arguments
    ///
    /// – `method` – threshold estimation algorithm.\
    /// – `delta` – offset for delta-based estimation; must be ≥ 0.0.\
    /// – `percentile` – percentile for distribution-based estimation; in `[0.0, 1.0]`.\
    /// – `window_size` – local analysis window in samples; must be > 0.\
    /// – `min_threshold` – lower clamp on the computed threshold; must be ≥ 0.0.\
    /// – `max_threshold` – upper clamp on the computed threshold; must be > `min_threshold`.
    ///
    /// # Returns
    ///
    /// An [`AdaptiveThresholdConfig`] with the specified values. Use
    /// [`validate`][AdaptiveThresholdConfig::validate] to verify constraints.
    #[inline]
    #[must_use]
    pub const fn new(
        method: AdaptiveThresholdMethod,
        delta: f64,
        percentile: f64,
        window_size: usize,
        min_threshold: f64,
        max_threshold: f64,
    ) -> Self {
        Self {
            method,
            delta,
            percentile,
            window_size,
            min_threshold,
            max_threshold,
        }
    }

    /// Creates a delta-based adaptive threshold configuration.
    ///
    /// # Arguments
    ///
    /// – `delta` – fixed offset added to local maxima; must be ≥ 0.0.\
    /// – `window_size` – local analysis window in samples; must be > 0.
    ///
    /// # Returns
    ///
    /// An [`AdaptiveThresholdConfig`] with `method = Delta`, `percentile = 0.9`,
    /// and threshold bounds `[0.01, 1.0]`.
    #[inline]
    #[must_use]
    pub const fn delta(delta: f64, window_size: usize) -> Self {
        Self {
            method: AdaptiveThresholdMethod::Delta,
            delta,
            percentile: 0.9,
            window_size,
            min_threshold: 0.01,
            max_threshold: 1.0,
        }
    }

    /// Creates a percentile-based adaptive threshold configuration.
    ///
    /// # Arguments
    ///
    /// – `percentile` – distribution percentile used as the threshold; must be in `[0.0, 1.0]`.\
    /// – `window_size` – local analysis window in samples; must be > 0.
    ///
    /// # Returns
    ///
    /// An [`AdaptiveThresholdConfig`] with `method = Percentile`, `delta = 0.05`,
    /// and threshold bounds `[0.01, 1.0]`.
    #[inline]
    #[must_use]
    pub const fn percentile(percentile: f64, window_size: usize) -> Self {
        Self {
            method: AdaptiveThresholdMethod::Percentile,
            delta: 0.05,
            percentile,
            window_size,
            min_threshold: 0.01,
            max_threshold: 1.0,
        }
    }

    /// Creates a combined delta + percentile adaptive threshold configuration.
    ///
    /// # Arguments
    ///
    /// – `delta` – delta offset component; must be ≥ 0.0.\
    /// – `percentile` – percentile component; must be in `[0.0, 1.0]`.\
    /// – `window_size` – local analysis window in samples; must be > 0.
    ///
    /// # Returns
    ///
    /// An [`AdaptiveThresholdConfig`] with `method = Combined` and threshold
    /// bounds `[0.01, 1.0]`.
    #[inline]
    #[must_use]
    pub const fn combined(delta: f64, percentile: f64, window_size: usize) -> Self {
        Self {
            method: AdaptiveThresholdMethod::Combined,
            delta,
            percentile,
            window_size,
            min_threshold: 0.01,
            max_threshold: 1.0,
        }
    }

    /// Sets the lower clamp on the computed threshold.
    ///
    /// # Arguments
    ///
    /// – `min_threshold` – new minimum; must be ≥ 0.0 and < `max_threshold`.
    #[inline]
    pub const fn set_min_threshold(&mut self, min_threshold: f64) {
        self.min_threshold = min_threshold;
    }

    /// Sets the upper clamp on the computed threshold.
    ///
    /// # Arguments
    ///
    /// – `max_threshold` – new maximum; must be > `min_threshold`.
    #[inline]
    pub const fn set_max_threshold(&mut self, max_threshold: f64) {
        self.max_threshold = max_threshold;
    }

    /// Validates all parameter constraints.
    ///
    /// # Returns
    ///
    /// `Ok(self)` if all constraints hold.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter][crate::AudioSampleError] for the first
    /// violated constraint: `delta < 0`, `percentile` outside `[0, 1]`,
    /// `window_size == 0`, `min_threshold < 0`, or `max_threshold ≤ min_threshold`.
    #[inline]
    pub fn validate(self) -> AudioSampleResult<Self> {
        if self.delta < 0.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Delta must be non-negative",
            )));
        }

        if self.percentile < 0.0 || self.percentile > 1.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Percentile must be between0.0 and1.0",
            )));
        }

        if self.window_size == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Window size must be greater than 0",
            )));
        }

        if self.min_threshold < 0.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Minimum threshold must be non-negative",
            )));
        }

        if self.max_threshold <= self.min_threshold {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Maximum threshold must be greater than minimum threshold",
            )));
        }

        Ok(self)
    }
}

#[cfg(feature = "peak-picking")]
impl Default for AdaptiveThresholdConfig {
    fn default() -> Self {
        Self {
            method: AdaptiveThresholdMethod::Delta,
            delta: 0.05,
            percentile: 0.9,
            window_size: 1024,
            min_threshold: 0.01,
            max_threshold: 1.0,
        }
    }
}

/// Full configuration for onset peak picking.
///
/// ## Purpose
///
/// Governs every stage of the peak-picking pipeline: optional pre-emphasis
/// filtering, optional median smoothing, optional onset-strength normalisation,
/// adaptive threshold estimation, and minimum-separation enforcement.
///
/// ## Intended Usage
///
/// Use one of the content-specific presets (`music`, `speech`, `drums`) or
/// build from scratch with [`PeakPickingConfig::new`]. Attach to the relevant
/// peak-picking call. Call [`validate`][PeakPickingConfig::validate] to check
/// constraints before processing.
///
/// ## Invariants
///
/// - `pre_emphasis_coeff` must be in `[0.0, 1.0]`.
/// - `median_filter_length` must be an odd positive integer.
/// - `adaptive_threshold` must satisfy its own invariants.
///
/// ## Assumptions
///
/// Disabling `pre_emphasis`, `median_filter`, and `normalize_onset_strength`
/// is safe and reduces processing cost at the expense of robustness. These
/// flags are always respected regardless of the chosen preset.
#[cfg(feature = "peak-picking")]
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PeakPickingConfig {
    /// Parameters for adaptive threshold estimation.
    pub adaptive_threshold: AdaptiveThresholdConfig,

    /// Minimum distance between two consecutive detected peaks, in samples.
    ///
    /// Prevents multiple detections from a single onset event. At 44.1 kHz,
    /// 512 samples ≈ 11.6 ms.
    pub min_peak_separation: NonZeroUsize,

    /// Whether to apply pre-emphasis (high-pass) filtering before detection.
    ///
    /// Enhances transient content by attenuating slowly-varying components.
    pub pre_emphasis: bool,

    /// First-order high-pass coefficient for pre-emphasis, in `[0.0, 1.0]`.
    ///
    /// Higher values produce stronger emphasis on rapid changes. Only used
    /// when `pre_emphasis = true`.
    pub pre_emphasis_coeff: f64,

    /// Whether to apply a median filter to the onset strength signal.
    ///
    /// Smooths impulsive noise while preserving genuine peak structure.
    pub median_filter: bool,

    /// Kernel length of the median filter; must be an odd positive integer.
    ///
    /// Larger kernels smooth more aggressively but may merge nearby peaks.
    /// Only used when `median_filter = true`.
    pub median_filter_length: NonZeroUsize,

    /// Whether to normalise the onset strength before thresholding.
    ///
    /// Normalisation makes detection thresholds consistent across signals with
    /// different overall levels.
    pub normalize_onset_strength: bool,

    /// Normalisation method applied to onset strength when
    /// `normalize_onset_strength = true`.
    pub normalization_method: NormalizationMethod,
}

#[cfg(feature = "peak-picking")]
impl PeakPickingConfig {
    /// Creates a peak-picking configuration with all fields specified.
    ///
    /// Prefer the content-specific presets (`music`, `speech`, `drums`) for
    /// common scenarios.
    ///
    /// # Arguments
    ///
    /// – `adaptive_threshold` – adaptive threshold parameters.\
    /// – `min_peak_separation` – minimum samples between adjacent peaks.\
    /// – `pre_emphasis` – whether to apply pre-emphasis filtering.\
    /// – `pre_emphasis_coeff` – first-order high-pass coefficient; in `[0.0, 1.0]`.\
    /// – `median_filter` – whether to smooth with a median filter.\
    /// – `median_filter_length` – kernel size; must be an odd positive integer.\
    /// – `normalize_onset_strength` – whether to normalise onset strength.\
    /// – `normalization_method` – normalisation algorithm when `normalize_onset_strength = true`.
    ///
    /// # Returns
    ///
    /// A [`PeakPickingConfig`] with the given values. Use
    /// [`validate`][PeakPickingConfig::validate] to verify constraints.
    #[inline]
    #[must_use]
    pub const fn new(
        adaptive_threshold: AdaptiveThresholdConfig,
        min_peak_separation: NonZeroUsize,
        pre_emphasis: bool,
        pre_emphasis_coeff: f64,
        median_filter: bool,
        median_filter_length: NonZeroUsize,
        normalize_onset_strength: bool,
        normalization_method: NormalizationMethod,
    ) -> Self {
        Self {
            adaptive_threshold,
            min_peak_separation,
            pre_emphasis,
            pre_emphasis_coeff,
            median_filter,
            median_filter_length,
            normalize_onset_strength,
            normalization_method,
        }
    }

    /// A preset optimised for musical onset detection.
    ///
    /// Combined adaptive thresholding, long minimum separation, strong
    /// pre-emphasis, and median filtering for robustness on typical music.
    ///
    /// # Returns
    ///
    /// A [`PeakPickingConfig`] with combined thresholding (delta 0.03, percentile 0.85,
    /// window 2048), min separation 1024 samples, pre-emphasis 0.95, and median filter 5.
    ///
    /// # Panics
    ///
    /// Never panics; hardcoded nonzero values are used.
    #[inline]
    #[must_use]
    pub const fn music() -> Self {
        Self {
            adaptive_threshold: AdaptiveThresholdConfig::combined(0.03, 0.85, 2048),
            min_peak_separation: NonZeroUsize::new(1024).expect("Must be non-zero"),
            pre_emphasis: true,
            pre_emphasis_coeff: 0.95,
            median_filter: true,
            median_filter_length: NonZeroUsize::new(5).expect("Must be non-zero"),
            normalize_onset_strength: true,
            normalization_method: NormalizationMethod::Peak,
        }
    }

    /// A preset optimised for speech onset detection.
    ///
    /// Delta-based thresholding with short minimum separation and a small
    /// median filter to preserve rapid speech transitions.
    ///
    /// # Returns
    ///
    /// A [`PeakPickingConfig`] with delta thresholding (delta 0.07, window 1024),
    /// min separation 256 samples, pre-emphasis 0.98, and median filter 3.
    ///
    /// # Panics
    ///
    /// Never panics; hardcoded nonzero values are used.
    #[inline]
    #[must_use]
    pub const fn speech() -> Self {
        Self {
            adaptive_threshold: AdaptiveThresholdConfig::delta(0.07, 1024),
            min_peak_separation: NonZeroUsize::new(256).expect("Must be non-zero"),
            pre_emphasis: true,
            pre_emphasis_coeff: 0.98,
            median_filter: true,
            median_filter_length: NonZeroUsize::new(3).expect("Must be non-zero"),
            normalize_onset_strength: true,
            normalization_method: NormalizationMethod::Peak,
        }
    }

    /// A preset optimised for drum and percussive onset detection.
    ///
    /// Percentile-based thresholding with very short minimum separation and
    /// no median filtering to preserve sharp transient edges.
    ///
    /// # Returns
    ///
    /// A [`PeakPickingConfig`] with percentile thresholding (0.95, window 512),
    /// min separation 128 samples, pre-emphasis 0.93, and median filter disabled.
    ///
    /// # Panics
    ///
    /// Never panics; hardcoded nonzero values are used.
    #[inline]
    #[must_use]
    pub const fn drums() -> Self {
        Self {
            adaptive_threshold: AdaptiveThresholdConfig::percentile(0.95, 512),
            min_peak_separation: NonZeroUsize::new(128).expect("Must be non-zero"),
            pre_emphasis: true,
            pre_emphasis_coeff: 0.93,
            median_filter: false,
            median_filter_length: NonZeroUsize::new(3).expect("Must be non-zero"),
            normalize_onset_strength: true,
            normalization_method: NormalizationMethod::Peak,
        }
    }

    /// Sets the minimum distance between consecutive detected peaks.
    ///
    /// # Arguments
    ///
    /// – `samples` – minimum separation in samples.
    #[inline]
    pub const fn set_min_peak_separation(&mut self, samples: NonZeroUsize) {
        self.min_peak_separation = samples;
    }

    /// Sets the minimum distance between consecutive peaks, given in milliseconds.
    ///
    /// # Arguments
    ///
    /// – `ms` – minimum separation in milliseconds.\
    /// – `sample_rate` – sample rate in Hz used to convert milliseconds to samples.
    ///
    /// # Panics
    ///
    /// Panics if `ms * sample_rate / 1000.0` rounds to zero, which would produce
    /// an invalid `NonZeroUsize`.
    #[inline]
    pub fn set_min_peak_separation_ms(&mut self, ms: f64, sample_rate: f64) {
        self.min_peak_separation =
            NonZeroUsize::new((ms * sample_rate / 1000.0) as usize).expect("Must be non-zero");
    }

    /// Configures pre-emphasis filtering.
    ///
    /// # Arguments
    ///
    /// – `enabled` – whether to apply pre-emphasis before peak picking.\
    /// – `coeff` – first-order high-pass coefficient; must be in `[0.0, 1.0]`.
    #[inline]
    pub const fn set_pre_emphasis(&mut self, enabled: bool, coeff: f64) {
        self.pre_emphasis = enabled;
        self.pre_emphasis_coeff = coeff;
    }

    /// Configures median smoothing.
    ///
    /// # Arguments
    ///
    /// – `enabled` – whether to apply median filtering to the onset strength.\
    /// – `length` – kernel size; must be an odd positive integer.
    #[inline]
    pub const fn set_median_filter(&mut self, enabled: bool, length: NonZeroUsize) {
        self.median_filter = enabled;
        self.median_filter_length = length;
    }

    /// Validates all parameter constraints.
    ///
    /// # Returns
    ///
    /// `Ok(self)` if all constraints hold.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter][crate::AudioSampleError] if
    /// `pre_emphasis_coeff` is outside `[0.0, 1.0]`, `median_filter_length`
    /// is even, or `adaptive_threshold` fails its own validation.
    #[inline]
    pub fn validate(self) -> AudioSampleResult<Self> {
        self.adaptive_threshold.validate()?;

        if self.pre_emphasis_coeff < 0.0 || self.pre_emphasis_coeff > 1.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Pre-emphasis coefficient must be between0.0 and1.0",
            )));
        }

        if self.median_filter_length.get().is_multiple_of(2) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Median filter length must be a positive odd integer",
            )));
        }

        Ok(self)
    }
}

#[cfg(feature = "peak-picking")]
impl Default for PeakPickingConfig {
    fn default() -> Self {
        Self {
            adaptive_threshold: AdaptiveThresholdConfig::default(),
            min_peak_separation: NonZeroUsize::new(512).expect("Must be non-zero"),
            pre_emphasis: true,
            pre_emphasis_coeff: 0.97,
            median_filter: true,
            median_filter_length: NonZeroUsize::new(3).expect("Must be non-zero"),
            normalize_onset_strength: true,
            normalization_method: NormalizationMethod::Peak,
        }
    }
}

/// Noise colour classification for audio perturbation and synthesis.
///
/// Different noise colours exhibit distinct spectral energy distributions,
/// influencing perceived brightness, smoothness, and temporal structure.
#[cfg(feature = "editing")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum NoiseColor {
    /// White noise.
    ///
    /// Exhibits approximately uniform spectral energy density across the
    /// frequency spectrum, resulting in a bright and broadband character.
    #[default]
    White,

    /// Pink noise.
    ///
    /// Exhibits decreasing spectral energy with increasing frequency,
    /// producing a perceptually balanced spectrum across octaves.
    Pink,

    /// Brown (red) noise.
    ///
    /// Exhibits strongly attenuated high-frequency content, yielding a
    /// smoother and more correlated temporal structure.
    Brown,
}

#[cfg(feature = "editing")]
impl Display for NoiseColor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            Self::White => "white",
            Self::Pink => "pink",
            Self::Brown => "brown",
        };
        f.write_str(s)
    }
}

#[cfg(feature = "editing")]
impl FromStr for NoiseColor {
    type Err = AudioSampleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let normalised = s.trim().to_ascii_lowercase();

        match normalised.as_str() {
            "white" | "white_noise" | "whitenoise" => Ok(Self::White),
            "pink" | "pink_noise" | "pinknoise" => Ok(Self::Pink),
            "brown" | "brown_noise" | "brownian" | "red" | "red_noise" => Ok(Self::Brown),

            _ => Err(AudioSampleError::parse::<Self, _>(format!(
                "Failed to parse {}. Got {}, must be one of {:?}",
                std::any::type_name::<Self>(),
                s,
                [
                    "white",
                    "white_noise",
                    "whitenoise",
                    "pink",
                    "pink_noise",
                    "pinknoise",
                    "brown",
                    "brownian",
                    "red",
                    "red_noise"
                ]
            ))),
        }
    }
}

#[cfg(feature = "editing")]
impl TryFrom<&str> for NoiseColor {
    type Error = AudioSampleError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        value.parse()
    }
}

/// Type of perturbation applied to audio during data augmentation.
///
/// Each variant encodes all parameters needed for one perturbation operation.
/// Use the associated constructor functions (`gaussian_noise`, `random_gain`,
/// etc.) rather than constructing variants directly, as they set sensible
/// defaults for optional fields.
#[cfg(feature = "editing")]
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum PerturbationMethod {
    /// Adds coloured Gaussian noise to reach a target signal-to-noise ratio.
    ///
    /// Noise is scaled so the SNR between the unperturbed signal's RMS level
    /// and the added noise matches `target_snr_db`. Valid range: `[-60, 60]` dB.
    GaussianNoise {
        /// Target signal-to-noise ratio in dB. Valid range: `[-60.0, 60.0]`.
        target_snr_db: f64,
        /// Spectral colour of the added noise.
        noise_color: NoiseColor,
    },

    /// Applies a uniformly distributed random gain within a dB range.
    ///
    /// A gain value is drawn from the uniform distribution over
    /// `[min_gain_db, max_gain_db]` and applied equally to all channels.
    RandomGain {
        /// Lower bound of the gain range in dB. Must be < `max_gain_db` and ≥ −40.0.
        min_gain_db: f64,
        /// Upper bound of the gain range in dB. Must be > `min_gain_db` and ≤ 20.0.
        max_gain_db: f64,
    },

    /// Applies a high-pass filter to attenuate low-frequency content.
    ///
    /// Simulates microphone rumble filtering or band-limited recording conditions.
    HighPassFilter {
        /// Cutoff frequency in Hz. Must be in `(0, nyquist)`.
        cutoff_hz: f64,
        /// Filter slope in dB per octave. `None` uses the default 6 dB/octave.
        /// When `Some`, must be in `[0.0, 48.0]` dB/octave.
        slope_db_per_octave: Option<f64>,
    },

    /// Applies a low-pass filter to attenuate high-frequency content.
    ///
    /// Simulates telephone bandwidth or other band-limited environments.
    LowPassFilter {
        /// Cutoff frequency in Hz. Must be in `(0, nyquist)`.
        cutoff_hz: f64,
        /// Filter slope in dB per octave. `None` uses the default 6 dB/octave.
        /// When `Some`, must be in `[0.0, 48.0]` dB/octave.
        slope_db_per_octave: Option<f64>,
    },

    /// Shifts the fundamental frequency by a fixed number of semitones.
    ///
    /// Uses a phase vocoder algorithm to preserve audio duration while shifting
    /// pitch. Valid range: `[-12.0, 12.0]` semitones.
    PitchShift {
        /// Pitch shift in semitones. Positive = higher; negative = lower.
        /// Valid range: `[-12.0, 12.0]`.
        semitones: f64,
        /// Whether to preserve spectral envelope (formants) during pitch shifting.
        ///
        /// When `true`, extracts and reapplies the spectral envelope to maintain
        /// natural vocal timbre. Useful for vocal pitch shifts to avoid "chipmunk"
        /// or "monster" effects. For non-vocal sources, this can typically be `false`.
        preserve_formants: bool,
    },
}

#[cfg(feature = "editing")]
impl PerturbationMethod {
    /// Creates a Gaussian noise perturbation targeting `target_snr_db`.
    ///
    /// # Arguments
    ///
    /// – `target_snr_db` – desired SNR in dB; valid range `[-60.0, 60.0]`.\
    /// – `noise_color` – spectral colour of the added noise.
    ///
    /// # Returns
    ///
    /// A [`PerturbationMethod::GaussianNoise`] variant.
    #[inline]
    #[must_use]
    pub const fn gaussian_noise(target_snr_db: f64, noise_color: NoiseColor) -> Self {
        Self::GaussianNoise {
            target_snr_db,
            noise_color,
        }
    }

    /// Creates a random gain perturbation within `[min_gain_db, max_gain_db]`.
    ///
    /// # Arguments
    ///
    /// – `min_gain_db` – lower bound in dB; must be ≥ −40.0 and < `max_gain_db`.\
    /// – `max_gain_db` – upper bound in dB; must be ≤ 20.0 and > `min_gain_db`.
    ///
    /// # Returns
    ///
    /// A [`PerturbationMethod::RandomGain`] variant.
    #[inline]
    #[must_use]
    pub const fn random_gain(min_gain_db: f64, max_gain_db: f64) -> Self {
        Self::RandomGain {
            min_gain_db,
            max_gain_db,
        }
    }

    /// Creates a high-pass filter perturbation with the default 6 dB/octave slope.
    ///
    /// # Arguments
    ///
    /// – `cutoff_hz` – cutoff frequency in Hz; must be in `(0, nyquist)`.
    ///
    /// # Returns
    ///
    /// A [`PerturbationMethod::HighPassFilter`] variant with `slope_db_per_octave = None`.
    #[inline]
    #[must_use]
    pub const fn high_pass_filter(cutoff_hz: f64) -> Self {
        Self::HighPassFilter {
            cutoff_hz,
            slope_db_per_octave: None,
        }
    }

    /// Creates a high-pass filter perturbation with an explicit filter slope.
    ///
    /// # Arguments
    ///
    /// – `cutoff_hz` – cutoff frequency in Hz; must be in `(0, nyquist)`.\
    /// – `slope_db_per_octave` – roll-off rate; must be in `[0.0, 48.0]` dB/octave.
    ///
    /// # Returns
    ///
    /// A [`PerturbationMethod::HighPassFilter`] variant with the given slope.
    #[inline]
    #[must_use]
    pub const fn high_pass_filter_with_slope(cutoff_hz: f64, slope_db_per_octave: f64) -> Self {
        Self::HighPassFilter {
            cutoff_hz,
            slope_db_per_octave: Some(slope_db_per_octave),
        }
    }

    /// Creates a low-pass filter perturbation.
    ///
    /// # Arguments
    ///
    /// – `cutoff_hz` – cutoff frequency in Hz; must be in `(0, nyquist)`.\
    /// – `slope_db_per_octave` – roll-off rate. `None` uses the default 6 dB/octave;
    ///   when `Some`, must be in `[0.0, 48.0]` dB/octave.
    ///
    /// # Returns
    ///
    /// A [`PerturbationMethod::LowPassFilter`] variant.
    #[inline]
    #[must_use]
    pub const fn low_pass_filter(cutoff_hz: f64, slope_db_per_octave: Option<f64>) -> Self {
        Self::LowPassFilter {
            cutoff_hz,
            slope_db_per_octave,
        }
    }

    /// Creates a pitch-shift perturbation using phase vocoder algorithm.
    ///
    /// Preserves audio duration while shifting pitch. Uses STFT-based phase vocoder
    /// for high-quality results. When formant preservation is enabled, maintains
    /// vocal timbre by preserving spectral envelope.
    ///
    /// # Arguments
    ///
    /// – `semitones` – shift amount; positive = higher, negative = lower. Valid range `[-12.0, 12.0]`.\
    /// – `preserve_formants` – whether to preserve spectral envelope (formants); recommended for vocal content.
    ///
    /// # Returns
    ///
    /// A [`PerturbationMethod::PitchShift`] variant.
    #[inline]
    #[must_use]
    pub const fn pitch_shift(semitones: f64, preserve_formants: bool) -> Self {
        Self::PitchShift {
            semitones,
            preserve_formants,
        }
    }

    /// Validates all perturbation parameters against `sample_rate`.
    ///
    /// # Arguments
    ///
    /// – `sample_rate` – sample rate in Hz; used to compute the Nyquist limit for
    ///   frequency bounds validation.
    ///
    /// # Returns
    ///
    /// `Ok(self)` if all constraints are satisfied.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter][crate::AudioSampleError] describing the
    /// first violated constraint.
    #[inline]
    pub fn validate(self, sample_rate: f64) -> AudioSampleResult<Self> {
        match self {
            Self::GaussianNoise { target_snr_db, .. } => {
                if !(-60.0..=60.0).contains(&target_snr_db) {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "parameter",
                        "Target SNR should be between -60 and 60 dB",
                    )));
                }
            }
            Self::RandomGain {
                min_gain_db,
                max_gain_db,
            } => {
                if min_gain_db >= max_gain_db {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "parameter",
                        "Minimum gain must be less than maximum gain",
                    )));
                }
                if min_gain_db < -40.0 || max_gain_db > 20.0 {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "parameter",
                        "Gain values should be between -40 dB and +20 dB",
                    )));
                }
            }
            Self::HighPassFilter {
                cutoff_hz,
                slope_db_per_octave,
            }
            | Self::LowPassFilter {
                cutoff_hz,
                slope_db_per_octave,
            } => {
                let nyquist = sample_rate / 2.0;
                if cutoff_hz <= 0.0 || cutoff_hz >= nyquist {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "parameter",
                        format!("Cutoff frequency must be between 0 and Nyquist ({nyquist:.1} Hz)"),
                    )));
                }
                if let Some(slope) = slope_db_per_octave
                    && !(0.0..=48.0).contains(&slope)
                {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "parameter",
                        "Slope must be between 0 and 48 dB/octave",
                    )));
                }
            }
            Self::PitchShift { semitones, .. } => {
                if semitones.abs() > 12.0 {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "parameter",
                        "Pitch shift should be between -12 and +12 semitones",
                    )));
                }
            }
        }
        Ok(self)
    }
}

/// Configuration for a single audio perturbation operation.
///
/// ## Purpose
///
/// Pairs a [`PerturbationMethod`] with an optional random seed to allow both
/// stochastic and fully reproducible perturbation pipelines.
///
/// ## Intended Usage
///
/// Use [`PerturbationConfig::new`] for randomised perturbation or
/// [`PerturbationConfig::with_seed`] when reproducibility is required.
/// Call [`validate`][PerturbationConfig::validate] with the target sample rate
/// before passing to a perturbation function.
///
/// ## Invariants
///
/// The `method` field must satisfy its own parameter constraints (checked by
/// [`validate`][PerturbationConfig::validate]).
///
/// ## Assumptions
///
/// When `seed` is `None`, the implementation uses the thread-local random
/// number generator; results will vary between runs.
#[cfg(feature = "editing")]
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PerturbationConfig {
    /// The perturbation algorithm and its parameters.
    pub method: PerturbationMethod,

    /// Optional seed for the random number generator.
    ///
    /// When `Some`, the perturbation is fully reproducible. When `None`,
    /// the thread-local RNG is used and results will differ between runs.
    pub seed: Option<u64>,
}

#[cfg(feature = "editing")]
impl PerturbationConfig {
    /// Creates a perturbation configuration using the thread-local RNG.
    ///
    /// # Arguments
    ///
    /// – `method` – the perturbation algorithm and its parameters.
    ///
    /// # Returns
    ///
    /// A [`PerturbationConfig`] with `seed = None`.
    #[inline]
    #[must_use]
    pub const fn new(method: PerturbationMethod) -> Self {
        Self { method, seed: None }
    }

    /// Creates a perturbation configuration with a fixed random seed.
    ///
    /// Using the same `method` and `seed` on identical input guarantees
    /// identical output.
    ///
    /// # Arguments
    ///
    /// – `method` – the perturbation algorithm and its parameters.\
    /// – `seed` – value used to seed the random number generator.
    ///
    /// # Returns
    ///
    /// A [`PerturbationConfig`] with `seed = Some(seed)`.
    #[inline]
    #[must_use]
    pub const fn with_seed(method: PerturbationMethod, seed: u64) -> Self {
        Self {
            method,
            seed: Some(seed),
        }
    }

    /// Validates that the perturbation method parameters are consistent with `sample_rate`.
    ///
    /// # Arguments
    ///
    /// – `sample_rate` – sample rate in Hz; used for frequency bounds checking.
    ///
    /// # Returns
    ///
    /// `Ok(self)` if validation passes; otherwise an
    ///
    /// # Errors
    ///
    /// Returns an [crate::AudioSampleError::Parameter][crate::AudioSampleError] describing the first
    /// [crate::AudioSampleError::Parameter][crate::AudioSampleError].
    #[inline]
    pub fn validate(self, sample_rate: f64) -> AudioSampleResult<Self> {
        self.method.validate(sample_rate)?;
        Ok(self)
    }
}

#[cfg(feature = "editing")]
impl Default for PerturbationConfig {
    fn default() -> Self {
        Self::new(PerturbationMethod::GaussianNoise {
            target_snr_db: 20.0,
            noise_color: NoiseColor::White,
        })
    }
}


/// Noise shaping algorithm applied during dithering.
///
/// Selects the spectral distribution of the dither noise added by
/// [`AudioDithering::dither`][crate::operations::traits::AudioDithering::dither].
/// All variants use triangular-probability-density-function (TPDF) dither as the
/// noise source; the variant controls how that noise is spectrally shaped before it
/// is mixed into the signal.
///
/// ## Intended Usage
///
/// Pass a `NoiseShape` value to
/// [`AudioDithering::dither`][crate::operations::traits::AudioDithering::dither]
/// to choose between flat or perceptually optimized noise placement.
#[cfg(feature = "dithering")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NoiseShape {
    /// Flat (white) TPDF dithering.
    ///
    /// Adds triangular noise with a flat power spectral density across all
    /// frequencies.  This is the simplest and most widely supported dither
    /// variant and is appropriate when downstream processing will handle
    /// perceptual weighting, or when the extra computation of noise shaping
    /// is undesirable.
    Flat,

    /// F-weighted noise shaping.
    ///
    /// Applies a first-order high-pass filter to the TPDF noise so that
    /// noise energy is redistributed towards higher frequencies where the
    /// human ear is less sensitive.  The result is perceptually quieter
    /// dither at the cost of increased noise power near Nyquist.
    ///
    /// This variant approximates the behaviour of the classic Wannamaker
    /// F-weighting filter using a single-pole recursive shaper.
    FWeighted,
}
