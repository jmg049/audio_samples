//! Supporting types and enums for audio operations.
//!
//! This module contains all the configuration types, enums, and helper structures
//! used by the audio processing traits.

/// Methods for normalizing audio sample values.
///
/// Different normalization methods are appropriate for different audio processing scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationMethod {
    /// Min-Max normalization: scales values to a specified range [min, max].
    /// Best for general audio level adjustment and ensuring samples fit within a target range.
    MinMax,
    /// Z-Score normalization: transforms to zero mean and unit variance.
    /// Useful for statistical analysis and machine learning preprocessing.
    ZScore,
    /// Mean normalization: centers data around zero by subtracting the mean.
    /// Good for removing DC offset while preserving relative amplitudes.
    Mean,
    /// Median normalization: centers data around zero using the median.
    /// More robust to outliers than mean normalization.
    Median,
    /// Peak normalization: scales by the maximum absolute value.
    /// Preserves dynamic range while preventing clipping.
    Peak,
}

/// Window functions for spectral analysis and filtering.
///
/// Different window types provide different trade-offs between frequency resolution
/// and spectral leakage in FFT-based analysis.
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum WindowType {
    /// Rectangular window (no windowing) - best frequency resolution but high leakage.
    Rectangular,
    /// Hanning window - good general-purpose window with moderate leakage.
    Hanning,
    /// Hamming window - similar to Hanning but slightly different coefficients.
    Hamming,
    /// Blackman window - low leakage but wider main lobe.
    Blackman,
    /// Kaiser window - parameterizable trade-off between resolution and leakage.
    Kaiser { beta: f64 },
    /// Gaussian window - smooth roll-off with parameterizable width.
    Gaussian { std: f64 },
}

/// Fade curve shapes for envelope operations.
///
/// Different curves provide different perceptual characteristics for fades.
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum FadeCurve {
    /// Linear fade - constant rate of change.
    Linear,
    /// Exponential fade - faster change at the beginning.
    Exponential,
    /// Logarithmic fade - faster change at the end.
    Logarithmic,
    /// Smooth step fade - S-curve with smooth transitions.
    SmoothStep,
    /// Custom fade curve defined by a function.
    Custom(fn(f64) -> f64),
}

/// Methods for converting multi-channel audio to mono.
#[derive(Debug, Clone, PartialEq)]
pub enum MonoConversionMethod {
    /// Average all channels equally.
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StereoConversionMethod {
    /// Duplicate mono signal to both left and right channels.
    Duplicate,
    /// Pan the mono signal (0.0 = center, -1.0 = left, 1.0 = right).
    Pan(f64),
    /// Use as left channel, fill right with silence.
    Left,
    /// Use as right channel, fill left with silence.
    Right,
}

/// Methods for converting between arbitrary channel counts.
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelConversionMethod {
    /// Repeat existing channels cyclically to reach target count.
    Repeat,
    /// Smart conversion: average down for fewer channels, duplicate for more.
    Smart,
    /// Custom mapping matrix where each row defines the weights for an output channel.
    /// Matrix dimensions should be [output_channels x input_channels].
    Custom(Vec<Vec<f64>>),
}

/// Voice Activity Detection (VAD) methods.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VadMethod {
    /// Simple energy-based detection using RMS threshold.
    Energy,
    /// Zero crossing rate based detection.
    ZeroCrossing,
    /// Combined energy and zero crossing rate.
    Combined,
    /// Spectral-based detection using spectral features.
    Spectral,
}

/// Filter types for digital signal processing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterType {
    /// Low-pass filter - allows frequencies below cutoff.
    LowPass,
    /// High-pass filter - allows frequencies above cutoff.
    HighPass,
    /// Band-pass filter - allows frequencies within a range.
    BandPass,
    /// Band-stop filter - blocks frequencies within a range.
    BandStop,
    /// All-pass filter - preserves all frequencies but changes phase.
    AllPass,
}

/// Time units for duration parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeUnit {
    /// Duration in seconds.
    Seconds(f64),
    /// Duration in samples.
    Samples(usize),
    /// Duration in milliseconds.
    Milliseconds(f64),
}

impl TimeUnit {
    /// Convert to number of samples given a sample rate.
    pub fn to_samples(&self, sample_rate: usize) -> usize {
        match self {
            TimeUnit::Seconds(s) => (*s * sample_rate as f64) as usize,
            TimeUnit::Samples(n) => *n,
            TimeUnit::Milliseconds(ms) => (*ms * sample_rate as f64 / 1000.0) as usize,
        }
    }

    /// Convert to seconds given a sample rate.
    pub fn to_seconds(&self, sample_rate: usize) -> f64 {
        match self {
            TimeUnit::Seconds(s) => *s,
            TimeUnit::Samples(n) => *n as f64 / sample_rate as f64,
            TimeUnit::Milliseconds(ms) => *ms / 1000.0,
        }
    }
}

/// Resampling quality settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResamplingQuality {
    /// Fast but lower quality resampling.
    Fast,
    /// Balanced speed and quality.
    Medium,
    /// Highest quality but slower resampling.
    High,
}

/// Pitch shift algorithms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PitchShiftMethod {
    /// Phase vocoder method - good quality, preserves formants.
    PhaseVocoder,
    /// Simple time-domain stretching - fast but lower quality.
    TimeStretch,
    /// PSOLA (Pitch Synchronous Overlap and Add) - good for speech.
    Psola,
}

/// Spectral analysis parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralParams {
    /// FFT window size in samples.
    pub window_size: usize,
    /// Hop size in samples (overlap = window_size - hop_size).
    pub hop_size: usize,
    /// Window function to apply.
    pub window_type: WindowType,
    /// Whether to apply zero-padding.
    pub zero_pad: bool,
}

/// Scaling methods for spectrograms.
///
/// Different scaling approaches provide different perspectives on spectral content
/// and are appropriate for different analysis tasks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpectrogramScale {
    /// Linear power scale - preserves absolute power relationships.
    /// Best for scientific analysis and energy measurements.
    Linear,
    /// Logarithmic (dB) scale - compresses dynamic range for visualization.
    /// Formula: 20 * log10(power) with floor at -80 dB to prevent log(0).
    /// Useful for visualizing weak signals alongside strong ones.
    Log,
    /// Mel frequency scale - perceptually motivated frequency spacing.
    /// Maps linear frequencies to mel scale using: mel = 2595 * log10(1 + f/700).
    /// Commonly used in speech recognition and music information retrieval.
    Mel,
}
