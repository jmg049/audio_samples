#![allow(unused_imports)]
//! Supporting types and enums for audio operations.
//!
//! This module contains all the configuration types, enums, and helper structures
//! used by the audio processing traits.

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

/// Pad side enum
#[cfg(feature = "editing")]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadSide {
    /// Pad on the left side
    Left,
    /// Pad on the right side
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

/// Methods for normalizing audio sample values.
///
/// Different normalization methods are appropriate for different audio processing scenarios.
#[cfg(any(feature = "processing", feature = "peak-picking"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NormalizationMethod {
    /// Min-Max normalization: scales values to a specified range [min, max].
    /// Best for general audio level adjustment and ensuring samples fit within a target range.
    #[default]
    MinMax,
    /// Peak normalization: scales by the maximum absolute value to a target level.
    /// Preserves dynamic range while preventing clipping.
    Peak,
    /// Mean normalization: centers data around zero by subtracting the mean.
    /// Good for removing DC offset while preserving relative amplitudes.
    Mean,
    /// Median normalization: centers data around zero using the median.
    /// More robust to outliers than mean normalization.
    Median,
    /// Z-Score normalization: transforms to zero mean and unit variance.
    /// Useful for statistical analysis and machine learning preprocessing.
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

/// Configuration for audio normalization operations.
///
/// Combines a normalization method with its parameters, providing type-safe
/// configuration for the normalize() operation.
#[cfg(feature = "processing")]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalizationConfig<T>
where
    T: StandardSample,
{
    /// The normalization method to apply
    pub method: NormalizationMethod,
    /// Target minimum value (for MinMax method)
    pub min: Option<T>,
    /// Target maximum value (for MinMax method)
    pub max: Option<T>,
    /// Target peak level (for Peak method)
    pub target: Option<T>,
}

#[cfg(feature = "processing")]
impl<T> NormalizationConfig<T>
where
    T: StandardSample,
{
    /// Create a Peak normalization configuration.
    #[inline]
    pub const fn peak(target: T) -> Self {
        Self {
            method: NormalizationMethod::Peak,
            min: None,
            max: None,
            target: Some(target),
        }
    }

    /// Create a MinMax normalization configuration.
    #[inline]
    pub const fn min_max(min: T, max: T) -> Self {
        Self {
            method: NormalizationMethod::MinMax,
            min: Some(min),
            max: Some(max),
            target: None,
        }
    }

    /// Create a Mean normalization configuration.
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

    /// Create a Median normalization configuration.
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

    /// Create a ZScore normalization configuration.
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
    /// Peak normalization to 1.0
    #[inline]
    #[must_use]
    pub const fn peak_normalized() -> Self {
        Self::peak(1.0)
    }

    /// Min-Max normalization to [-1.0, 1.0] range
    #[inline]
    #[must_use]
    pub const fn range_normalized() -> Self {
        Self::min_max(-1.0, 1.0)
    }
}

/// Fade curve shapes for envelope operations.
///
/// Different curves provide different perceptual characteristics for fades.
#[cfg(feature = "editing")]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum FadeCurve {
    /// Linear fade - constant rate of change.
    #[default]
    Linear,
    /// Exponential fade - faster change at the beginning.
    Exponential,
    /// Logarithmic fade - faster change at the end.
    Logarithmic,
    /// Smooth step fade - S-curve with smooth transitions.
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

/// Methods for converting multi-channel audio to mono.
#[cfg(feature = "channels")]
#[derive(Default, Debug, Clone, PartialEq)]
pub enum MonoConversionMethod {
    /// Average all channels equally.
    #[default]
    Average,
    /// Use left channel only (for stereo input).
    Left,
    /// Use right channel only (for stereo input).
    Right,
    /// Use weighted average with custom weights per channel.
    Weighted(Vec<f64>),
    /// Use center channel if available, otherwise average L/R.
    Center,
}

/// Methods for converting mono audio to stereo.
#[cfg(feature = "channels")]
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub enum StereoConversionMethod {
    /// Duplicate mono signal to both left and right channels.
    #[default]
    Duplicate,
    /// Pan the mono signal (0.0 = center, -1.0 = left,1.0 = right).
    Pan(f64),
    /// Use as left channel, fill right with silence.
    Left,
    /// Use as right channel, fill left with silence.
    Right,
}

/// Voice Activity Detection (VAD) methods.
#[cfg(feature = "vad")]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadMethod {
    /// Simple energy-based detection using RMS threshold.
    #[default]
    Energy,
    /// Zero crossing rate based detection.
    ZeroCrossing,
    /// Combined energy and zero crossing rate.
    Combined,
    /// Spectral-based detection using spectral features.
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

/// Multi-channel handling policy for Voice Activity Detection (VAD).
///
/// This determines how VAD decisions are produced for multi-channel audio.
#[cfg(feature = "vad")]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadChannelPolicy {
    /// Average all channels to a mono signal and run VAD once.
    #[default]
    AverageToMono,
    /// Run VAD per-channel and mark speech if any channel is active.
    AnyChannel,
    /// Run VAD per-channel and mark speech only if all channels are active.
    AllChannels,
    /// Run VAD on a specific channel index.
    Channel(usize),
}

/// Configuration for Voice Activity Detection (VAD).
///
/// The VAD implementation is frame-based: it produces a boolean decision per frame
/// of length `frame_size` with step `hop_size`.
///
/// Defaults are chosen to work reasonably well for general audio, but you should
/// tune thresholds for your content and sample format.
#[cfg(feature = "vad")]
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// Create a new VAD configuration with specified parameters
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

    /// Create a VAD configuration using only energy-based detection.
    #[inline]
    #[must_use]
    pub fn energy_only() -> Self {
        let mut conf = Self::default();
        conf.method = VadMethod::Energy;
        conf
    }

    /// Validate configuration parameters.
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

/// IIR filter design parameters.
///
/// Comprehensive parameters for designing IIR filters with various
/// characteristics and specifications.
#[cfg(feature = "iir-filtering")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IirFilterDesign {
    /// Type of IIR filter (Butterworth, Chebyshev, etc.)
    pub filter_type: IirFilterType,
    /// Response type (low-pass, high-pass, etc.)
    pub response: FilterResponse,
    /// Filter order (number of poles)
    pub order: NonZeroUsize,
    /// Cutoff frequency in Hz (for low-pass/high-pass)
    pub cutoff_frequency: Option<f64>,
    /// Lower cutoff frequency in Hz (for band-pass/band-stop)
    pub low_frequency: Option<f64>,
    /// Upper cutoff frequency in Hz (for band-pass/band-stop)
    pub high_frequency: Option<f64>,
    /// Passband ripple in dB (for Chebyshev Type I and Elliptic)
    pub passband_ripple: Option<f64>,
    /// Stopband attenuation in dB (for Chebyshev Type II and Elliptic)
    pub stopband_attenuation: Option<f64>,
}

#[cfg(feature = "iir-filtering")]
impl IirFilterDesign {
    /// Create a simple Butterworth low-pass filter design.
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

    /// Create a simple Butterworth high-pass filter design.
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

    /// Create a simple Butterworth band-pass filter design.
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

    /// Create a Chebyshev Type I filter design.
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

/// Parametric EQ band configuration.
///
/// Represents a single band in a parametric equalizer with
/// frequency, gain, and Q (quality factor) parameters.
#[cfg(feature = "parametric-eq")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EqBand {
    /// Type of EQ band (peak, shelf, etc.)
    pub band_type: EqBandType,
    /// Center frequency in Hz (for peak/notch) or corner frequency (for shelves)
    pub frequency: f64,
    /// Gain in dB (positive for boost, negative for cut)
    pub gain_db: f64,
    /// Quality factor (bandwidth control)
    /// Higher Q = narrower bandwidth, Lower Q = wider bandwidth
    pub q_factor: f64,
    /// Whether this band is enabled/active
    pub enabled: bool,
}

#[cfg(feature = "parametric-eq")]
impl EqBand {
    /// Create a new peak/notch EQ band.
    ///
    /// # Arguments
    /// * `frequency` - Center frequency in Hz
    /// * `gain_db` - Gain in dB (positive for boost, negative for cut)
    /// * `q_factor` - Quality factor (bandwidth control)
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

    /// Create a new low shelf EQ band.
    ///
    /// # Arguments
    /// * `frequency` - Corner frequency in Hz
    /// * `gain_db` - Gain in dB (positive for boost, negative for cut)
    /// * `q_factor` - Shelf slope control
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

    /// Create a new high shelf EQ band.
    ///
    /// # Arguments
    /// * `frequency` - Corner frequency in Hz
    /// * `gain_db` - Gain in dB (positive for boost, negative for cut)
    /// * `q_factor` - Shelf slope control
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

    /// Create a new low-pass filter band.
    ///
    /// # Arguments
    /// * `frequency` - Cutoff frequency in Hz
    /// * `q_factor` - Filter resonance (typically 0.707 for Butterworth)
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

    /// Create a new high-pass filter band.
    ///
    /// # Arguments
    /// * `frequency` - Cutoff frequency in Hz
    /// * `q_factor` - Filter resonance (typically 0.707 for Butterworth)
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

    /// Enable or disable this EQ band.
    #[inline]
    pub const fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if this EQ band is enabled.
    #[inline]
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Validate the EQ band parameters.
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

/// Parametric equalizer configuration.
///
/// A complete parametric EQ consisting of multiple bands that can be
/// applied to audio signals for frequency shaping.
#[cfg(feature = "parametric-eq")]
#[derive(Debug, Clone, PartialEq)]
pub struct ParametricEq {
    /// Vector of EQ bands
    pub bands: Vec<EqBand>,
    /// Overall output gain in dB
    pub output_gain_db: f64,
    /// Whether the EQ is bypassed
    pub bypassed: bool,
}

#[cfg(feature = "parametric-eq")]
impl ParametricEq {
    /// Create a new empty parametric EQ.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            bands: Vec::new(),
            output_gain_db: 0.0,
            bypassed: false,
        }
    }

    /// Add an EQ band to the parametric EQ.
    #[inline]
    pub fn add_band(&mut self, band: EqBand) {
        self.bands.push(band);
    }

    /// Remove an EQ band by index.
    #[inline]
    pub fn remove_band(&mut self, index: usize) -> Option<EqBand> {
        if index < self.bands.len() {
            Some(self.bands.remove(index))
        } else {
            None
        }
    }

    /// Get a reference to an EQ band by index.
    #[inline]
    #[must_use]
    pub fn get_band(&self, index: usize) -> Option<&EqBand> {
        self.bands.get(index)
    }

    /// Get a mutable reference to an EQ band by index.
    #[inline]
    pub fn get_band_mut(&mut self, index: usize) -> Option<&mut EqBand> {
        self.bands.get_mut(index)
    }

    /// Get the number of bands in the EQ.
    #[inline]
    #[must_use]
    pub const fn band_count(&self) -> usize {
        self.bands.len()
    }

    /// Set the overall output gain.
    #[inline]
    pub const fn set_output_gain(&mut self, gain_db: f64) {
        self.output_gain_db = gain_db;
    }

    /// Enable or disable the EQ (bypass).
    #[inline]
    pub const fn set_bypassed(&mut self, bypassed: bool) {
        self.bypassed = bypassed;
    }

    /// Check if the EQ is bypassed.
    #[inline]
    #[must_use]
    pub const fn is_bypassed(&self) -> bool {
        self.bypassed
    }

    /// Validate all EQ bands.
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

    /// Create a common 3-band EQ (low shelf, mid peak, high shelf).
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

    /// Create a common 5-band EQ.
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

/// Side-chain configuration for dynamic range processing.
///
/// Allows external control signals to drive the compressor/limiter.
#[cfg(feature = "dynamic-range")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SideChainConfig {
    /// Whether side-chain processing is enabled
    pub enabled: bool,
    /// High-pass filter frequency for side-chain signal (Hz)
    /// Helps reduce low-frequency pumping effects
    pub high_pass_freq: Option<f64>,
    /// Low-pass filter frequency for side-chain signal (Hz)
    /// Focuses compression on specific frequency ranges
    pub low_pass_freq: Option<f64>,
    /// Pre-emphasis for side-chain signal (dB)
    /// Emphasizes specific frequencies in the control signal
    pub pre_emphasis_db: f64,
    /// Mix between internal and external side-chain signal (0.0-1.0)
    ///0.0 = internal only,1.0 = external only
    pub external_mix: f64,
}

#[cfg(feature = "dynamic-range")]
impl SideChainConfig {
    /// Create a new disabled side-chain configuration.
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

    /// Create a new enabled side-chain configuration with default settings.
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

    /// Enable side-chain processing.
    #[inline]
    pub const fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable side-chain processing.
    #[inline]
    pub const fn disable(&mut self) {
        self.enabled = false;
    }

    /// Set high-pass filter frequency for side-chain signal.
    #[inline]
    pub const fn set_high_pass(&mut self, freq: f64) {
        self.high_pass_freq = Some(freq);
    }

    /// Set low-pass filter frequency for side-chain signal.
    #[inline]
    pub const fn set_low_pass(&mut self, freq: f64) {
        self.low_pass_freq = Some(freq);
    }

    /// Validate side-chain configuration.
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

/// Compressor configuration parameters.
///
/// Controls how the compressor responds to signal levels above the threshold.
#[cfg(feature = "dynamic-range")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompressorConfig {
    /// Threshold level in dB (typically -40 to 0 dB)
    /// Signal levels above this will be compressed
    pub threshold_db: f64,
    /// Compression ratio (1.0 = no compression, >1.0 = compression)
    /// Higher values provide more aggressive compression
    pub ratio: f64,
    /// Attack time in milliseconds (0.1 to 100 ms typical)
    /// How quickly the compressor responds to signals above threshold
    pub attack_ms: f64,
    /// Release time in milliseconds (10 to 1000 ms typical)
    /// How quickly the compressor stops compressing when signal drops below threshold
    pub release_ms: f64,
    /// Makeup gain in dB (-20 to +20 dB typical)
    /// Gain applied after compression to restore loudness
    pub makeup_gain_db: f64,
    /// Knee type for compression curve
    pub knee_type: KneeType,
    /// Knee width in dB (0.1 to 10 dB for soft knee)
    /// Controls the transition smoothness around the threshold
    pub knee_width_db: f64,
    /// Detection method for compression
    pub detection_method: DynamicRangeMethod,
    /// Side-chain configuration
    pub side_chain: SideChainConfig,
    /// Lookahead time in milliseconds (0 to 10 ms typical)
    /// Allows the compressor to "see" upcoming peaks
    pub lookahead_ms: f64,
}

#[cfg(feature = "dynamic-range")]
impl CompressorConfig {
    /// Create a new compressor configuration with default settings.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a vocal compressor preset.
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

    /// Create a drum compressor preset.
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

    /// Create a bus compressor preset.
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

    /// Validate compressor configuration.
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

/// Limiter configuration parameters.
///
/// Controls how the limiter prevents signal levels from exceeding the ceiling.
#[cfg(feature = "dynamic-range")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LimiterConfig {
    /// Ceiling level in dB (typically -0.1 to -3.0 dB)
    /// Absolute maximum level that the limiter will allow
    pub ceiling_db: f64,
    /// Attack time in milliseconds (0.01 to 10 ms typical)
    /// How quickly the limiter responds to signals approaching the ceiling
    pub attack_ms: f64,
    /// Release time in milliseconds (10 to 1000 ms typical)
    /// How quickly the limiter stops limiting when signal drops below ceiling
    pub release_ms: f64,
    /// Knee type for limiting curve
    pub knee_type: KneeType,
    /// Knee width in dB (0.1 to 5 dB for soft knee)
    /// Controls the transition smoothness around the ceiling
    pub knee_width_db: f64,
    /// Detection method for limiting
    pub detection_method: DynamicRangeMethod,
    /// Side-chain configuration
    pub side_chain: SideChainConfig,
    /// Lookahead time in milliseconds (0.1 to 10 ms typical)
    /// Allows the limiter to prevent peaks before they occur
    pub lookahead_ms: f64,
    /// Whether to apply ISP (Inter-Sample Peak) limiting
    /// Prevents aliasing and inter-sample peaks in the digital domain
    pub isp_limiting: bool,
}

#[cfg(feature = "dynamic-range")]
impl LimiterConfig {
    /// Create a new limiter configuration with default settings.
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

    /// Create a transparent limiter preset.
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

    /// Create a mastering limiter preset.
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

    /// Create a broadcast limiter preset.
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

    /// Validate limiter configuration.
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

/// Configuration for adaptive thresholding in peak picking.
///
/// Adaptive thresholding dynamically adjusts the detection threshold based on
/// local characteristics of the onset strength function to improve detection
/// accuracy across varying signal conditions.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg(feature = "peak-picking")]
pub struct AdaptiveThresholdConfig {
    /// Method for computing adaptive threshold
    pub method: AdaptiveThresholdMethod,
    /// Delta value for delta-based thresholding (typical range: 0.01-0.1)
    /// Larger values = fewer false positives but may miss weak onsets
    pub delta: f64,
    /// Percentile value for percentile-based thresholding (0.0-1.0)
    /// Higher percentiles = more conservative thresholding
    pub percentile: f64,
    /// Size of local window for adaptive computation (in samples)
    /// Larger windows = more stable but less responsive thresholds
    pub window_size: usize,
    /// Minimum threshold value to prevent over-sensitivity
    /// Ensures threshold never drops below this absolute minimum
    pub min_threshold: f64,
    /// Maximum threshold value to prevent under-sensitivity
    /// Ensures threshold never exceeds this absolute maximum
    pub max_threshold: f64,
}

#[cfg(feature = "peak-picking")]
impl AdaptiveThresholdConfig {
    /// Create a new adaptive threshold configuration with default settings.
    ///
    /// Default configuration suitable for general onset detection:
    /// - Delta method with 0.05 delta value
    /// - Window size of 1024 samples (about 23ms at 44.1kHz)
    /// - Reasonable min/max threshold bounds
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

    /// Create a delta-based adaptive threshold configuration.
    ///
    /// # Arguments
    /// * `delta` - Delta value for threshold computation
    /// * `window_size` - Size of local window in samples
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

    /// Create a percentile-based adaptive threshold configuration.
    ///
    /// # Arguments
    /// * `percentile` - Percentile value (0.0-1.0)
    /// * `window_size` - Size of local window in samples
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

    /// Create a combined adaptive threshold configuration.
    ///
    /// # Arguments
    /// * `delta` - Delta value for delta component
    /// * `percentile` - Percentile value for percentile component
    /// * `window_size` - Size of local window in samples
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

    /// Set the minimum threshold value.
    #[inline]
    pub const fn set_min_threshold(&mut self, min_threshold: f64) {
        self.min_threshold = min_threshold;
    }

    /// Set the maximum threshold value.
    #[inline]
    pub const fn set_max_threshold(&mut self, max_threshold: f64) {
        self.max_threshold = max_threshold;
    }

    /// Validate the adaptive threshold configuration.
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

/// Configuration for peak picking with temporal constraints.
///
/// Peak picking identifies local maxima in the onset strength function that
/// exceed a threshold. Temporal constraints ensure detected peaks are
/// separated by minimum time intervals and can include smoothing.
#[cfg(feature = "peak-picking")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeakPickingConfig {
    /// Adaptive threshold configuration
    pub adaptive_threshold: AdaptiveThresholdConfig,
    /// Minimum time separation between peaks (in samples)
    /// Prevents detecting multiple peaks for the same onset event
    /// Must be greater than zero
    pub min_peak_separation: NonZeroUsize,
    /// Enable pre-emphasis to enhance transient detection
    /// Applies high-pass filtering to emphasize onset characteristics
    pub pre_emphasis: bool,
    /// Pre-emphasis coefficient (0.0-1.0) for high-pass filtering
    /// Higher values = stronger emphasis on transients
    pub pre_emphasis_coeff: f64,
    /// Enable median filtering for onset strength smoothing
    /// Reduces noise while preserving peak structure
    pub median_filter: bool,
    /// Median filter length (must be odd)
    /// Larger values = more smoothing but may blur sharp onsets
    pub median_filter_length: NonZeroUsize,
    /// Normalize onset strength before peak picking
    /// Ensures consistent detection across different signal levels
    pub normalize_onset_strength: bool,
    /// Normalization method for onset strength
    pub normalization_method: NormalizationMethod,
}

#[cfg(feature = "peak-picking")]
impl PeakPickingConfig {
    /// Create a new peak picking configuration with default settings.
    ///
    /// Default configuration optimized for general onset detection:
    /// - Adaptive delta thresholding
    /// - 512 samples minimum separation (about 11ms at 44.1kHz)
    /// - Pre-emphasis enabled with moderate coefficient
    /// - Median filtering enabled with small kernel
    /// - Peak normalization enabled
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

    /// Create a configuration optimized for music onset detection.
    ///
    /// Uses settings that work well for typical musical content:
    /// - Combined adaptive thresholding for robustness
    /// - Longer minimum separation to avoid over-detection
    /// - Strong pre-emphasis for transient enhancement
    /// - Median filtering for noise reduction
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

    /// Create a configuration optimized for speech onset detection.
    ///
    /// Uses settings that work well for speech signals:
    /// - Delta-based thresholding for responsiveness
    /// - Shorter minimum separation for rapid speech
    /// - Moderate pre-emphasis
    /// - Smaller median filter to preserve speech transients
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

    /// Create a configuration optimized for drum onset detection.
    ///
    /// Uses settings that work well for percussive content:
    /// - Percentile-based thresholding for stability
    /// - Very short minimum separation for rapid sequences
    /// - Strong pre-emphasis for transient enhancement
    /// - No median filtering to preserve sharp transients
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

    /// Set the minimum peak separation in samples.
    #[inline]
    pub const fn set_min_peak_separation(&mut self, samples: NonZeroUsize) {
        self.min_peak_separation = samples;
    }

    /// Set the minimum peak separation in milliseconds.
    ///
    /// # Panics
    ///
    /// Panics if the millisecond to sample conversion results in a value that cannot be converted to usize.
    #[inline]
    pub fn set_min_peak_separation_ms(&mut self, ms: f64, sample_rate: f64) {
        self.min_peak_separation =
            NonZeroUsize::new((ms * sample_rate / 1000.0) as usize).expect("Must be non-zero");
    }

    /// Enable or disable pre-emphasis.
    #[inline]
    pub const fn set_pre_emphasis(&mut self, enabled: bool, coeff: f64) {
        self.pre_emphasis = enabled;
        self.pre_emphasis_coeff = coeff;
    }

    /// Enable or disable median filtering.
    #[inline]
    pub const fn set_median_filter(&mut self, enabled: bool, length: NonZeroUsize) {
        self.median_filter = enabled;
        self.median_filter_length = length;
    }

    /// Validate the peak picking configuration.
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

/// Perturbation methods for audio data augmentation.
///
/// Each variant defines a specific type of perturbation that can be applied
/// to audio samples for data augmentation, robustness testing, or creative effects.
#[cfg(feature = "editing")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PerturbationMethod {
    /// Gaussian noise injection with specified signal-to-noise ratio.
    ///
    /// Adds colored Gaussian noise to achieve the target SNR relative to
    /// the input signal's RMS level.
    ///
    /// # Arguments
    /// * `target_snr_db` - Target signal-to-noise ratio in dB
    /// * `noise_color` - Color/spectrum of the noise to add
    GaussianNoise {
        /// Target signal-to-noise ratio in dB
        target_snr_db: f64,
        /// Color/spectrum of the noise to add
        noise_color: NoiseColor,
    },
    /// Random gain adjustment within specified dB range.
    ///
    /// Applies uniform random gain scaling to all channels.
    /// Positive values boost, negative values attenuate.
    ///
    /// # Arguments
    /// * `min_gain_db` - Minimum gain in dB
    /// * `max_gain_db` - Maximum gain in dB
    RandomGain {
        /// Minimum gain in dB
        min_gain_db: f64,
        /// Maximum gain in dB
        max_gain_db: f64,
    },
    /// High-pass filtering to remove low-frequency content.
    ///
    /// Applies a high-pass filter to simulate microphone rumble removal
    /// or other high-pass effects commonly found in audio processing chains.
    ///
    /// # Arguments
    /// * `cutoff_hz` - Cutoff frequency in Hz
    /// * `slope_db_per_octave` - Filter slope (None = default 6dB/octave)
    HighPassFilter {
        /// Cutoff frequency in Hz
        cutoff_hz: f64,
        /// Filter slope in dB per octave (None = default 6dB/octave)
        slope_db_per_octave: Option<f64>,
    },
    /// Low-pass filtering to remove high-frequency content.
    ///
    /// Applies a low-pass filter to simulate telephone bandwidth
    /// or other low-pass effects commonly found in audio processing chains.
    ///
    /// # Arguments
    /// * `cutoff_hz` - Cutoff frequency in Hz
    /// * `slope_db_per_octave` - Filter slope (None = default 6dB/octave)
    LowPassFilter {
        /// Cutoff frequency in Hz
        cutoff_hz: f64,
        /// Filter slope in dB per octave (None = default 6dB/octave)
        slope_db_per_octave: Option<f64>,
    },
    /// Pitch shifting for data augmentation.
    ///
    /// Shifts the pitch of the audio signal by the specified number of semitones
    /// while attempting to maintain the original duration.
    ///
    /// # Arguments
    /// * `semitones` - Pitch shift in semitones (positive = higher, negative = lower)
    /// * `preserve_formants` - Whether to attempt formant preservation (basic implementation)
    PitchShift {
        /// Pitch shift in semitones (positive = higher, negative = lower)
        semitones: f64,
        /// Whether to attempt formant preservation (basic implementation)
        preserve_formants: bool,
    },
}

#[cfg(feature = "editing")]
impl PerturbationMethod {
    /// Create a Gaussian noise perturbation configuration.
    ///
    /// # Arguments
    /// * `target_snr_db` - Target signal-to-noise ratio in dB
    /// * `noise_color` - Color/spectrum of the noise
    #[inline]
    #[must_use]
    pub const fn gaussian_noise(target_snr_db: f64, noise_color: NoiseColor) -> Self {
        Self::GaussianNoise {
            target_snr_db,
            noise_color,
        }
    }

    /// Create a random gain perturbation configuration.
    ///
    /// # Arguments
    /// * `min_gain_db` - Minimum gain in dB
    /// * `max_gain_db` - Maximum gain in dB
    #[inline]
    #[must_use]
    pub const fn random_gain(min_gain_db: f64, max_gain_db: f64) -> Self {
        Self::RandomGain {
            min_gain_db,
            max_gain_db,
        }
    }

    /// Create a high-pass filter perturbation configuration.
    ///
    /// # Arguments
    /// * `cutoff_hz` - Cutoff frequency in Hz
    #[inline]
    #[must_use]
    pub const fn high_pass_filter(cutoff_hz: f64) -> Self {
        Self::HighPassFilter {
            cutoff_hz,
            slope_db_per_octave: None,
        }
    }

    /// Create a high-pass filter perturbation with custom slope.
    ///
    /// # Arguments
    /// * `cutoff_hz` - Cutoff frequency in Hz
    /// * `slope_db_per_octave` - Filter slope in dB per octave
    #[inline]
    #[must_use]
    pub const fn high_pass_filter_with_slope(cutoff_hz: f64, slope_db_per_octave: f64) -> Self {
        Self::HighPassFilter {
            cutoff_hz,
            slope_db_per_octave: Some(slope_db_per_octave),
        }
    }

    /// Create a low-pass filter perturbation configuration.
    ///
    /// # Arguments
    /// * `cutoff_hz` - Cutoff frequency in Hz
    /// * `slope_db_per_octave` - Filter slope in dB per octave
    #[inline]
    #[must_use]
    pub const fn low_pass_filter(cutoff_hz: f64, slope_db_per_octave: Option<f64>) -> Self {
        Self::LowPassFilter {
            cutoff_hz,
            slope_db_per_octave,
        }
    }

    /// Create a pitch shift perturbation configuration.
    ///
    /// # Arguments
    /// * `semitones` - Pitch shift in semitones
    /// * `preserve_formants` - Whether to preserve formants
    #[inline]
    #[must_use]
    pub const fn pitch_shift(semitones: f64, preserve_formants: bool) -> Self {
        Self::PitchShift {
            semitones,
            preserve_formants,
        }
    }

    /// Validate the perturbation method parameters.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz (for frequency validation)
    ///
    /// # Returns
    /// Result indicating whether the parameters are valid
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
            Self::LowPassFilter {
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

/// Configuration for audio perturbation operations.
///
/// This struct defines how audio samples should be perturbed for data augmentation,
/// robustness testing, or creative effects.
#[cfg(feature = "editing")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerturbationConfig {
    /// The perturbation method to apply
    pub method: PerturbationMethod,
    /// Optional random seed for deterministic perturbation
    /// If None, uses thread-local random number generator
    pub seed: Option<u64>,
}

#[cfg(feature = "editing")]
impl PerturbationConfig {
    /// Create a new perturbation configuration.
    ///
    /// # Arguments
    /// * `method` - The perturbation method to apply
    #[inline]
    #[must_use]
    pub const fn new(method: PerturbationMethod) -> Self {
        Self { method, seed: None }
    }

    /// Create a new perturbation configuration with a specific seed.
    ///
    /// # Arguments
    /// * `method` - The perturbation method to apply
    /// * `seed` - Random seed for deterministic results
    #[inline]
    #[must_use]
    pub const fn with_seed(method: PerturbationMethod, seed: u64) -> Self {
        Self {
            method,
            seed: Some(seed),
        }
    }

    /// Validate the perturbation configuration.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Result indicating whether the configuration is valid, containing the validated configuration
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
