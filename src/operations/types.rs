//! Supporting types and enums for audio operations.
//!
//! This module contains all the configuration types, enums, and helper structures
//! used by the audio processing traits.

use std::str::FromStr;
use std::marker::PhantomData;

use crate::{AudioSampleError, AudioSampleResult, ParameterError, RealFloat, to_precision};

/// Pad side enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadSide {
    /// Pad on the left side
    Left,
    /// Pad on the right side
    Right,
}

impl FromStr for PadSide {
    type Err = crate::AudioSampleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "left" => Ok(PadSide::Left),
            "right" => Ok(PadSide::Right),
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
pub enum WindowType<F: RealFloat> {
    /// Rectangular window (no windowing) - best frequency resolution but high leakage.
    Rectangular,
    /// Hanning window - good general-purpose window with moderate leakage.
    Hanning,
    /// Hamming window - similar to Hanning but slightly different coefficients.
    Hamming,
    /// Blackman window - low leakage but wider main lobe.
    Blackman,
    /// Kaiser window - parameterizable trade-off between resolution and leakage.
    Kaiser {
        /// Beta parameter controlling the trade-off between main lobe width and side lobe level
        beta: F,
    },
    /// Gaussian window - smooth roll-off with parameterizable width.
    Gaussian {
        /// Standard deviation parameter controlling the window width
        std: F,
    },
}

/// Fade curve shapes for envelope operations.
///
/// Different curves provide different perceptual characteristics for fades.
#[derive(Debug, Clone, Copy)]
pub enum FadeCurve<F: RealFloat> {
    /// Linear fade - constant rate of change.
    Linear,
    /// Exponential fade - faster change at the beginning.
    Exponential,
    /// Logarithmic fade - faster change at the end.
    Logarithmic,
    /// Smooth step fade - S-curve with smooth transitions.
    SmoothStep,
    /// Custom fade curve defined by a function.
    Custom(fn(F) -> F),
}

/// Methods for converting multi-channel audio to mono.
#[derive(Debug, Clone, PartialEq)]
pub enum MonoConversionMethod<F: RealFloat> {
    /// Average all channels equally.
    Average,
    /// Use left channel only (for stereo input).
    Left,
    /// Use right channel only (for stereo input).
    Right,
    /// Use weighted average with custom weights per channel.
    Weighted(Vec<F>),
    /// Use center channel if available, otherwise average L/R.
    Center,
}

/// Methods for converting mono audio to stereo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StereoConversionMethod<F: RealFloat> {
    /// Duplicate mono signal to both left and right channels.
    Duplicate,
    /// Pan the mono signal (F::zero() = center, -F::one() = left, F::one() = right).
    Pan(F),
    /// Use as left channel, fill right with silence.
    Left,
    /// Use as right channel, fill left with silence.
    Right,
}

/// Methods for converting between arbitrary channel counts.
#[derive(Debug, Clone, PartialEq)]
pub enum ChannelConversionMethod<F: RealFloat> {
    /// Repeat existing channels cyclically to reach target count.
    Repeat,
    /// Smart conversion: average down for fewer channels, duplicate for more.
    Smart,
    /// Custom mapping matrix where each row defines the weights for an output channel.
    /// Matrix dimensions should be [output_channels x input_channels].
    Custom(Vec<Vec<F>>),
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

/// Multi-channel handling policy for Voice Activity Detection (VAD).
///
/// This determines how VAD decisions are produced for multi-channel audio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VadChannelPolicy {
    /// Average all channels to a mono signal and run VAD once.
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
#[derive(Debug, Clone, PartialEq)]
pub struct VadConfig<F: RealFloat> {
    /// VAD method to use.
    pub method: VadMethod,
    /// Frame size in samples.
    pub frame_size: usize,
    /// Hop size in samples (frame step).
    pub hop_size: usize,
    /// Whether to include a final partial frame (zero-padded).
    pub pad_end: bool,
    /// Policy for multi-channel audio.
    pub channel_policy: VadChannelPolicy,

    /// Energy threshold in dBFS (RMS). Typical values: `-60.0` (very sensitive) to `-30.0`.
    pub energy_threshold_db: F,
    /// Minimum acceptable zero crossing rate (ZCR), expressed as crossings per sample in `[0, 1]`.
    pub zcr_min: F,
    /// Maximum acceptable zero crossing rate (ZCR), expressed as crossings per sample in `[0, 1]`.
    pub zcr_max: F,

    /// Minimum number of consecutive speech frames to keep a speech region.
    pub min_speech_frames: usize,
    /// Minimum number of consecutive non-speech frames to keep a silence region.
    /// Shorter silence gaps are filled as speech.
    pub min_silence_frames: usize,
    /// Hangover in frames: keep speech active for this many frames after energy drops.
    pub hangover_frames: usize,
    /// Majority-vote smoothing window in frames (1 = no smoothing).
    pub smooth_frames: usize,

    /// Lower bound of the speech band in Hz (used by `VadMethod::Spectral`).
    pub speech_band_low_hz: F,
    /// Upper bound of the speech band in Hz (used by `VadMethod::Spectral`).
    pub speech_band_high_hz: F,
    /// Threshold on speech-band energy ratio (used by `VadMethod::Spectral`).
    pub spectral_ratio_threshold: F,
}

impl<F: RealFloat> VadConfig<F> {
    /// Create a new VAD configuration with sensible defaults.
    pub fn new() -> Self {
        Self {
            method: VadMethod::Combined,
            frame_size: 1024,
            hop_size: 512,
            pad_end: false,
            channel_policy: VadChannelPolicy::AverageToMono,
            energy_threshold_db: to_precision(-40.0),
            zcr_min: to_precision(0.005),
            zcr_max: to_precision(0.25),
            min_speech_frames: 3,
            min_silence_frames: 2,
            hangover_frames: 3,
            smooth_frames: 3,
            speech_band_low_hz: to_precision(300.0),
            speech_band_high_hz: to_precision(3400.0),
            spectral_ratio_threshold: to_precision(0.5),
        }
    }

    /// Create an energy-only VAD configuration.
    pub fn energy_only() -> Self {
        Self {
            method: VadMethod::Energy,
            ..Self::new()
        }
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if self.frame_size == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "frame_size",
                "must be > 0",
            )));
        }
        if self.hop_size == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "hop_size",
                "must be > 0",
            )));
        }
        if self.hop_size > self.frame_size {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "hop_size",
                "must be <= frame_size",
            )));
        }
        if self.smooth_frames == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "smooth_frames",
                "must be >= 1",
            )));
        }
        if self.zcr_min < F::zero() || self.zcr_max > F::one() || self.zcr_min > self.zcr_max {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "zcr_*",
                "expected 0 <= zcr_min <= zcr_max <= 1",
            )));
        }
        if self.speech_band_low_hz <= F::zero()
            || self.speech_band_high_hz <= F::zero()
            || self.speech_band_low_hz >= self.speech_band_high_hz
        {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "speech_band_*",
                "expected 0 < low_hz < high_hz",
            )));
        }
        if self.spectral_ratio_threshold < F::zero() || self.spectral_ratio_threshold > F::one() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "spectral_ratio_threshold",
                "expected 0 <= threshold <= 1",
            )));
        }

        Ok(())
    }
}

impl<F: RealFloat> Default for VadConfig<F> {
    fn default() -> Self {
        Self::new()
    }
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
pub struct SpectralParams<F: RealFloat> {
    /// FFT window size in samples.
    pub window_size: usize,
    /// Hop size in samples (overlap = window_size - hop_size).
    pub hop_size: usize,
    /// Window function to apply.
    pub window_type: WindowType<F>,
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

/// Pitch detection algorithms.
///
/// Different algorithms provide different trade-offs between accuracy, robustness,
/// and computational efficiency for fundamental frequency estimation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PitchDetectionMethod {
    /// YIN algorithm - robust, works well with speech and music.
    /// Uses cumulative normalized difference function for accurate pitch detection.
    Yin,
    /// Autocorrelation method - simple and fast.
    /// Works well for clean periodic signals but less robust to noise.
    Autocorrelation,
    /// Cepstrum method - frequency domain approach.
    /// Effective for voiced speech but may struggle with complex timbres.
    Cepstrum,
    /// Harmonic product spectrum - emphasizes harmonics.
    /// Good for musical instruments with clear harmonic structure.
    HarmonicProduct,
}

/// IIR filter types for digital signal processing.
///
/// IIR (Infinite Impulse Response) filters provide efficient implementation
/// of recursive filters with feedback. They can achieve sharper roll-offs
/// than FIR filters with fewer coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IirFilterType {
    /// Butterworth filter - maximally flat passband response.
    /// Provides smooth frequency response with no ripple in passband.
    Butterworth,
    /// Chebyshev Type I - ripple in passband, sharp transition.
    /// Allows some ripple in passband for sharper cutoff.
    ChebyshevI,
    /// Chebyshev Type II - ripple in stopband, sharp transition.
    /// Allows some ripple in stopband for sharper cutoff.
    ChebyshevII,
    /// Elliptic filter - ripple in both passband and stopband.
    /// Provides sharpest transition at cost of ripple in both bands.
    Elliptic,
}

/// Filter response characteristics.
///
/// Defines the frequency response shape of the filter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterResponse {
    /// Low-pass filter - allows frequencies below cutoff.
    LowPass,
    /// High-pass filter - allows frequencies above cutoff.
    HighPass,
    /// Band-pass filter - allows frequencies within a range.
    BandPass,
    /// Band-stop filter - blocks frequencies within a range.
    BandStop,
}

/// IIR filter design parameters.
///
/// Comprehensive parameters for designing IIR filters with various
/// characteristics and specifications.
#[derive(Debug, Clone, PartialEq)]
pub struct IirFilterDesign<F: RealFloat> {
    /// Type of IIR filter (Butterworth, Chebyshev, etc.)
    pub filter_type: IirFilterType,
    /// Response type (low-pass, high-pass, etc.)
    pub response: FilterResponse,
    /// Filter order (number of poles)
    pub order: usize,
    /// Cutoff frequency in Hz (for low-pass/high-pass)
    pub cutoff_frequency: Option<F>,
    /// Lower cutoff frequency in Hz (for band-pass/band-stop)
    pub low_frequency: Option<F>,
    /// Upper cutoff frequency in Hz (for band-pass/band-stop)
    pub high_frequency: Option<F>,
    /// Passband ripple in dB (for Chebyshev Type I and Elliptic)
    pub passband_ripple: Option<F>,
    /// Stopband attenuation in dB (for Chebyshev Type II and Elliptic)
    pub stopband_attenuation: Option<F>,
}

impl<F: RealFloat> IirFilterDesign<F> {
    /// Create a simple Butterworth low-pass filter design.
    pub const fn butterworth_lowpass(order: usize, cutoff_frequency: F) -> Self {
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
    pub const fn butterworth_highpass(order: usize, cutoff_frequency: F) -> Self {
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
    pub const fn butterworth_bandpass(order: usize, low_frequency: F, high_frequency: F) -> Self {
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
    pub const fn chebyshev_i(
        response: FilterResponse,
        order: usize,
        cutoff_frequency: F,
        passband_ripple: F,
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

/// Parametric EQ band types.
///
/// Different band types provide different frequency shaping characteristics
/// for parametric equalization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EqBandType {
    /// Peak/Notch filter - boosts or cuts at a specific frequency.
    /// Gain > 0 creates a peak, gain < 0 creates a notch.
    Peak,
    /// Low shelf - affects frequencies below the corner frequency.
    /// Provides gentle boost or cut in the low frequency range.
    LowShelf,
    /// High shelf - affects frequencies above the corner frequency.
    /// Provides gentle boost or cut in the high frequency range.
    HighShelf,
    /// Low-pass filter - removes frequencies above the cutoff.
    /// Typically used as a protective filter.
    LowPass,
    /// High-pass filter - removes frequencies below the cutoff.
    /// Typically used to remove low-frequency rumble.
    HighPass,
    /// Band-pass filter - allows only frequencies within a range.
    /// Useful for isolating specific frequency ranges.
    BandPass,
    /// Band-stop (notch) filter - removes frequencies within a range.
    /// Useful for removing specific interference frequencies.
    BandStop,
}

/// Parametric EQ band configuration.
///
/// Represents a single band in a parametric equalizer with
/// frequency, gain, and Q (quality factor) parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct EqBand<F: RealFloat> {
    /// Type of EQ band (peak, shelf, etc.)
    pub band_type: EqBandType,
    /// Center frequency in Hz (for peak/notch) or corner frequency (for shelves)
    pub frequency: F,
    /// Gain in dB (positive for boost, negative for cut)
    pub gain_db: F,
    /// Quality factor (bandwidth control)
    /// Higher Q = narrower bandwidth, Lower Q = wider bandwidth
    pub q_factor: F,
    /// Whether this band is enabled/active
    pub enabled: bool,
}

impl<F: RealFloat> EqBand<F> {
    /// Create a new peak/notch EQ band.
    ///
    /// # Arguments
    /// * `frequency` - Center frequency in Hz
    /// * `gain_db` - Gain in dB (positive for boost, negative for cut)
    /// * `q_factor` - Quality factor (bandwidth control)
    pub const fn peak(frequency: F, gain_db: F, q_factor: F) -> Self {
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
    pub const fn low_shelf(frequency: F, gain_db: F, q_factor: F) -> Self {
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
    pub const fn high_shelf(frequency: F, gain_db: F, q_factor: F) -> Self {
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
    pub fn low_pass(frequency: F, q_factor: F) -> Self {
        Self {
            band_type: EqBandType::LowPass,
            frequency,
            gain_db: F::zero(),
            q_factor,
            enabled: true,
        }
    }

    /// Create a new high-pass filter band.
    ///
    /// # Arguments
    /// * `frequency` - Cutoff frequency in Hz
    /// * `q_factor` - Filter resonance (typically 0.707 for Butterworth)
    pub fn high_pass(frequency: F, q_factor: F) -> Self {
        Self {
            band_type: EqBandType::HighPass,
            frequency,
            gain_db: F::zero(),
            q_factor,
            enabled: true,
        }
    }

    /// Enable or disable this EQ band.
    pub const fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if this EQ band is enabled.
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Validate the EQ band parameters.
    pub fn validate(&self, sample_rate: F) -> AudioSampleResult<()> {
        let nyquist = sample_rate / to_precision(2.0);

        if self.frequency <= F::zero() || self.frequency >= nyquist {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "frequency",
                format!("{} Hz", self.frequency),
                "0",
                format!("{}", nyquist),
                "Frequency must be between 0 and Nyquist frequency",
            )));
        }

        if self.q_factor <= F::zero() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "q_factor",
                "Q factor must be positive",
            )));
        }

        // Check reasonable gain limits
        if self.gain_db.abs() > to_precision(40.0) {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "gain_db",
                format!("{} dB", self.gain_db),
                "-40",
                "40",
                "Gain must be within reasonable range",
            )));
        }

        Ok(())
    }
}

/// Parametric equalizer configuration.
///
/// A complete parametric EQ consisting of multiple bands that can be
/// applied to audio signals for frequency shaping.
#[derive(Debug, Clone, PartialEq)]
pub struct ParametricEq<F: RealFloat> {
    /// Vector of EQ bands
    pub bands: Vec<EqBand<F>>,
    /// Overall output gain in dB
    pub output_gain_db: F,
    /// Whether the EQ is bypassed
    pub bypassed: bool,
}

impl<F: RealFloat> ParametricEq<F> {
    /// Create a new empty parametric EQ.
    pub fn new() -> Self {
        Self {
            bands: Vec::new(),
            output_gain_db: F::zero(),
            bypassed: false,
        }
    }

    /// Add an EQ band to the parametric EQ.
    pub fn add_band(&mut self, band: EqBand<F>) {
        self.bands.push(band);
    }

    /// Remove an EQ band by index.
    pub fn remove_band(&mut self, index: usize) -> Option<EqBand<F>> {
        if index < self.bands.len() {
            Some(self.bands.remove(index))
        } else {
            None
        }
    }

    /// Get a reference to an EQ band by index.
    pub fn get_band(&self, index: usize) -> Option<&EqBand<F>> {
        self.bands.get(index)
    }

    /// Get a mutable reference to an EQ band by index.
    pub fn get_band_mut(&mut self, index: usize) -> Option<&mut EqBand<F>> {
        self.bands.get_mut(index)
    }

    /// Get the number of bands in the EQ.
    pub const fn band_count(&self) -> usize {
        self.bands.len()
    }

    /// Set the overall output gain.
    pub const fn set_output_gain(&mut self, gain_db: F) {
        self.output_gain_db = gain_db;
    }

    /// Enable or disable the EQ (bypass).
    pub const fn set_bypassed(&mut self, bypassed: bool) {
        self.bypassed = bypassed;
    }

    /// Check if the EQ is bypassed.
    pub const fn is_bypassed(&self) -> bool {
        self.bypassed
    }

    /// Validate all EQ bands.
    pub fn validate(&self, sample_rate: F) -> AudioSampleResult<()> {
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
        Ok(())
    }

    /// Create a common 3-band EQ (low shelf, mid peak, high shelf).
    pub fn three_band(
        low_freq: F,
        low_gain: F,
        mid_freq: F,
        mid_gain: F,
        mid_q: F,
        high_freq: F,
        high_gain: F,
    ) -> Self {
        let mut eq = Self::new();
        eq.add_band(EqBand::low_shelf(low_freq, low_gain, to_precision(0.707)));
        eq.add_band(EqBand::peak(mid_freq, mid_gain, mid_q));
        eq.add_band(EqBand::high_shelf(
            high_freq,
            high_gain,
            to_precision(0.707),
        ));
        eq
    }

    /// Create a common 5-band EQ.
    pub fn five_band() -> Self {
        let mut eq = Self::new();
        eq.add_band(EqBand::low_shelf(
            to_precision(100.0),
            F::zero(),
            to_precision(0.707),
        ));
        eq.add_band(EqBand::peak(to_precision(300.0), F::zero(), F::one()));
        eq.add_band(EqBand::peak(to_precision(1000.0), F::zero(), F::one()));
        eq.add_band(EqBand::peak(to_precision(3000.0), F::zero(), F::one()));
        eq.add_band(EqBand::high_shelf(
            to_precision(8000.0),
            F::zero(),
            to_precision(0.707),
        ));
        eq
    }
}

impl<F: RealFloat> Default for ParametricEq<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Knee types for dynamic range processing.
///
/// Controls the transition behavior when the signal crosses the threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KneeType {
    /// Hard knee - abrupt transition at threshold.
    /// Provides precise control but may introduce audible artifacts.
    Hard,
    /// Soft knee - gradual transition around threshold.
    /// Provides smoother, more musical compression but less precise control.
    Soft,
}

/// Dynamic range processing methods.
///
/// Different algorithms provide different characteristics and quality levels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DynamicRangeMethod {
    /// RMS-based detection - measures average power over time.
    /// Provides smooth, musical compression suitable for most material.
    Rms,
    /// Peak-based detection - responds to instantaneous peaks.
    /// Provides tight control of peaks but may sound more aggressive.
    Peak,
    /// Hybrid detection - combines RMS and peak detection.
    /// Balances musicality with peak control.
    Hybrid,
}

/// Side-chain configuration for dynamic range processing.
///
/// Allows external control signals to drive the compressor/limiter.
#[derive(Debug, Clone, PartialEq)]
pub struct SideChainConfig<F: RealFloat> {
    /// Whether side-chain processing is enabled
    pub enabled: bool,
    /// High-pass filter frequency for side-chain signal (Hz)
    /// Helps reduce low-frequency pumping effects
    pub high_pass_freq: Option<F>,
    /// Low-pass filter frequency for side-chain signal (Hz)
    /// Focuses compression on specific frequency ranges
    pub low_pass_freq: Option<F>,
    /// Pre-emphasis for side-chain signal (dB)
    /// Emphasizes specific frequencies in the control signal
    pub pre_emphasis_db: F,
    /// Mix between internal and external side-chain signal (F::zero()-F::one())
    /// F::zero() = internal only, F::one() = external only
    pub external_mix: F,
}

impl<F: RealFloat> SideChainConfig<F> {
    /// Create a new disabled side-chain configuration.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            high_pass_freq: None,
            low_pass_freq: None,
            pre_emphasis_db: F::zero(),
            external_mix: F::zero(),
        }
    }

    /// Create a new enabled side-chain configuration with default settings.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            high_pass_freq: Some(to_precision(100.0)),
            low_pass_freq: None,
            pre_emphasis_db: F::zero(),
            external_mix: F::one(),
        }
    }

    /// Enable side-chain processing.
    pub const fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable side-chain processing.
    pub const fn disable(&mut self) {
        self.enabled = false;
    }

    /// Set high-pass filter frequency for side-chain signal.
    pub const fn set_high_pass(&mut self, freq: F) {
        self.high_pass_freq = Some(freq);
    }

    /// Set low-pass filter frequency for side-chain signal.
    pub const fn set_low_pass(&mut self, freq: F) {
        self.low_pass_freq = Some(freq);
    }

    /// Validate side-chain configuration.
    pub fn validate(&self, sample_rate: F) -> AudioSampleResult<()> {
        if let Some(hp_freq) = self.high_pass_freq
            && (hp_freq <= F::zero() || hp_freq >= sample_rate / to_precision(2.0))
        {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "high_pass_freq",
                "High-pass frequency must be between 0 and Nyquist frequency",
            )));
        }

        if let Some(lp_freq) = self.low_pass_freq
            && (lp_freq <= F::zero() || lp_freq >= sample_rate / to_precision(2.0))
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

        if self.external_mix < F::zero() || self.external_mix > F::one() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "external_mix",
                "External mix must be between F::zero() and F::one()",
            )));
        }

        Ok(())
    }
}

/// Compressor configuration parameters.
///
/// Controls how the compressor responds to signal levels above the threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct CompressorConfig<F: RealFloat> {
    /// Threshold level in dB (typically -40 to 0 dB)
    /// Signal levels above this will be compressed
    pub threshold_db: F,
    /// Compression ratio (F::one() = no compression, >F::one() = compression)
    /// Higher values provide more aggressive compression
    pub ratio: F,
    /// Attack time in milliseconds (0.1 to 100 ms typical)
    /// How quickly the compressor responds to signals above threshold
    pub attack_ms: F,
    /// Release time in milliseconds (10 to 1000 ms typical)
    /// How quickly the compressor stops compressing when signal drops below threshold
    pub release_ms: F,
    /// Makeup gain in dB (-20 to +20 dB typical)
    /// Gain applied after compression to restore loudness
    pub makeup_gain_db: F,
    /// Knee type for compression curve
    pub knee_type: KneeType,
    /// Knee width in dB (0.1 to 10 dB for soft knee)
    /// Controls the transition smoothness around the threshold
    pub knee_width_db: F,
    /// Detection method for compression
    pub detection_method: DynamicRangeMethod,
    /// Side-chain configuration
    pub side_chain: SideChainConfig<F>,
    /// Lookahead time in milliseconds (0 to 10 ms typical)
    /// Allows the compressor to "see" upcoming peaks
    pub lookahead_ms: F,
}

impl<F: RealFloat> CompressorConfig<F> {
    /// Create a new compressor configuration with default settings.
    pub fn new() -> Self {
        Self {
            threshold_db: to_precision(-12.0),
            ratio: to_precision(4.0),
            attack_ms: to_precision(5.0),
            release_ms: to_precision(50.0),
            makeup_gain_db: F::zero(),
            knee_type: KneeType::Soft,
            knee_width_db: to_precision(2.0),
            detection_method: DynamicRangeMethod::Rms,
            side_chain: SideChainConfig::disabled(),
            lookahead_ms: F::zero(),
        }
    }

    /// Create a vocal compressor preset.
    pub fn vocal() -> Self {
        Self {
            threshold_db: to_precision(-18.0),
            ratio: to_precision(3.0),
            attack_ms: to_precision(2.0),
            release_ms: to_precision(100.0),
            makeup_gain_db: to_precision(3.0),
            knee_type: KneeType::Soft,
            knee_width_db: to_precision(4.0),
            detection_method: DynamicRangeMethod::Rms,
            side_chain: SideChainConfig::disabled(),
            lookahead_ms: F::zero(),
        }
    }

    /// Create a drum compressor preset.
    pub fn drum() -> Self {
        Self {
            threshold_db: to_precision(-8.0),
            ratio: to_precision(6.0),
            attack_ms: to_precision(0.1),
            release_ms: to_precision(20.0),
            makeup_gain_db: to_precision(2.0),
            knee_type: KneeType::Hard,
            knee_width_db: to_precision(0.5),
            detection_method: DynamicRangeMethod::Peak,
            side_chain: SideChainConfig::disabled(),
            lookahead_ms: F::zero(),
        }
    }

    /// Create a bus compressor preset.
    pub fn bus() -> Self {
        Self {
            threshold_db: to_precision(-20.0),
            ratio: to_precision(2.0),
            attack_ms: to_precision(10.0),
            release_ms: to_precision(200.0),
            makeup_gain_db: F::one(),
            knee_type: KneeType::Soft,
            knee_width_db: to_precision(6.0),
            detection_method: DynamicRangeMethod::Rms,
            side_chain: SideChainConfig::disabled(),
            lookahead_ms: F::zero(),
        }
    }

    /// Validate compressor configuration.
    pub fn validate(&self, sample_rate: F) -> AudioSampleResult<()> {
        if self.threshold_db > F::zero() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "threshold_db",
                "Threshold should be negative (below 0 dB)",
            )));
        }

        if self.ratio < F::one() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "ratio",
                "Ratio must be F::one() or greater",
            )));
        }

        if self.attack_ms < to_precision(0.01) || self.attack_ms > to_precision(1000.0) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Attack time must be between 0.01 and 1000 ms",
            )));
        }

        if self.release_ms < F::one() || self.release_ms > to_precision(10000.0) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Release time must be between F::one() and 10000 ms",
            )));
        }

        if self.makeup_gain_db.abs() > to_precision(40.0) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Makeup gain must be between -40.0 and +40.0 dB",
            )));
        }

        if self.knee_width_db < F::zero() || self.knee_width_db > to_precision(20.0) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Knee width must be between F::zero() and 20.0 dB",
            )));
        }

        if self.lookahead_ms < F::zero() || self.lookahead_ms > to_precision(20.0) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Lookahead time must be between F::zero() and 20.0 ms",
            )));
        }

        self.side_chain.validate(sample_rate)
    }
}

impl<F: RealFloat> Default for CompressorConfig<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Limiter configuration parameters.
///
/// Controls how the limiter prevents signal levels from exceeding the ceiling.
#[derive(Debug, Clone, PartialEq)]
pub struct LimiterConfig<F: RealFloat> {
    /// Ceiling level in dB (typically -0.1 to -3.0 dB)
    /// Absolute maximum level that the limiter will allow
    pub ceiling_db: F,
    /// Attack time in milliseconds (0.01 to 10 ms typical)
    /// How quickly the limiter responds to signals approaching the ceiling
    pub attack_ms: F,
    /// Release time in milliseconds (10 to 1000 ms typical)
    /// How quickly the limiter stops limiting when signal drops below ceiling
    pub release_ms: F,
    /// Knee type for limiting curve
    pub knee_type: KneeType,
    /// Knee width in dB (0.1 to 5 dB for soft knee)
    /// Controls the transition smoothness around the ceiling
    pub knee_width_db: F,
    /// Detection method for limiting
    pub detection_method: DynamicRangeMethod,
    /// Side-chain configuration
    pub side_chain: SideChainConfig<F>,
    /// Lookahead time in milliseconds (0.1 to 10 ms typical)
    /// Allows the limiter to prevent peaks before they occur
    pub lookahead_ms: F,
    /// Whether to apply ISP (Inter-Sample Peak) limiting
    /// Prevents aliasing and inter-sample peaks in the digital domain
    pub isp_limiting: bool,
}

impl<F: RealFloat> LimiterConfig<F> {
    /// Create a new limiter configuration with default settings.
    pub fn new() -> Self {
        Self {
            ceiling_db: to_precision(-0.1),
            attack_ms: to_precision(0.5),
            release_ms: to_precision(50.0),
            knee_type: KneeType::Soft,
            knee_width_db: F::one(),
            detection_method: DynamicRangeMethod::Peak,
            side_chain: SideChainConfig::disabled(),
            lookahead_ms: to_precision(2.0),
            isp_limiting: true,
        }
    }

    /// Create a transparent limiter preset.
    pub fn transparent() -> Self {
        Self {
            ceiling_db: to_precision(-0.1),
            attack_ms: to_precision(0.1),
            release_ms: to_precision(100.0),
            knee_type: KneeType::Soft,
            knee_width_db: to_precision(2.0),
            detection_method: DynamicRangeMethod::Peak,
            side_chain: SideChainConfig::disabled(),
            lookahead_ms: to_precision(5.0),
            isp_limiting: true,
        }
    }

    /// Create a mastering limiter preset.
    pub fn mastering() -> Self {
        Self {
            ceiling_db: to_precision(-0.3),
            attack_ms: F::one(),
            release_ms: to_precision(200.0),
            knee_type: KneeType::Soft,
            knee_width_db: to_precision(3.0),
            detection_method: DynamicRangeMethod::Hybrid,
            side_chain: SideChainConfig::disabled(),
            lookahead_ms: to_precision(10.0),
            isp_limiting: true,
        }
    }

    /// Create a broadcast limiter preset.
    pub fn broadcast() -> Self {
        Self {
            ceiling_db: -F::one(),
            attack_ms: to_precision(0.5),
            release_ms: to_precision(50.0),
            knee_type: KneeType::Hard,
            knee_width_db: to_precision(0.5),
            detection_method: DynamicRangeMethod::Peak,
            side_chain: SideChainConfig::disabled(),
            lookahead_ms: to_precision(2.0),
            isp_limiting: true,
        }
    }

    /// Validate limiter configuration.
    pub fn validate(&self, sample_rate: F) -> AudioSampleResult<()> {
        if self.ceiling_db > F::zero() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Ceiling should be negative (below 0 dB)",
            )));
        }

        if self.attack_ms < to_precision(0.001) || self.attack_ms > to_precision(100.0) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Attack time must be between 0.001 and 100 ms",
            )));
        }

        if self.release_ms < F::one() || self.release_ms > to_precision(10000.0) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Release time must be between F::one() and 10000 ms",
            )));
        }

        if self.knee_width_db < F::zero() || self.knee_width_db > to_precision(10.0) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Knee width must be between F::zero() and 10.0 dB",
            )));
        }

        if self.lookahead_ms < F::zero() || self.lookahead_ms > to_precision(20.0) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Lookahead time must be between F::zero() and 20.0 ms",
            )));
        }

        self.side_chain.validate(sample_rate)
    }
}

impl<F: RealFloat> Default for LimiterConfig<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for Constant-Q Transform (CQT) analysis.
///
/// The CQT provides logarithmic frequency spacing that aligns with musical
/// intervals, making it ideal for music analysis and harmonic detection.
#[derive(Debug, Clone, PartialEq)]
pub struct CqtConfig<F: RealFloat> {
    /// Number of frequency bins per octave (typically 12-24 for musical analysis)
    /// Higher values provide better frequency resolution but increase computation
    pub bins_per_octave: usize,
    /// Minimum frequency in Hz (typically 55 Hz for A1 or 27.5 Hz for A0)
    pub fmin: F,
    /// Maximum frequency in Hz (None = Nyquist frequency)
    /// Should be less than or equal to sample_rate / 2
    pub fmax: Option<F>,
    /// Quality factor controlling frequency resolution vs time resolution
    /// Higher Q = better frequency resolution, lower Q = better time resolution
    pub q_factor: F,
    /// Window function applied to each frequency bin
    pub window_type: WindowType<F>,
    /// Sparsity threshold for kernel optimization (F::zero() to F::one())
    /// Smaller values = more sparse kernels = faster computation
    pub sparsity_threshold: F,
    /// Whether to normalize the CQT output
    pub normalize: bool,
}

impl<F: RealFloat> CqtConfig<F> {
    /// Create a new CQT configuration with default settings.
    ///
    /// Default configuration suitable for general musical analysis:
    /// - 12 bins per octave (chromatic scale)
    /// - 55 Hz minimum frequency (A1)
    /// - Quality factor of F::one()
    /// - Hanning window
    /// - 0.01 sparsity threshold
    pub fn new() -> Self {
        Self {
            bins_per_octave: 12,
            fmin: to_precision(55.0), // A1
            fmax: None,               // Will be set to Nyquist frequency
            q_factor: F::one(),
            window_type: WindowType::Hanning,
            sparsity_threshold: to_precision(0.01),
            normalize: true,
        }
    }

    /// Create a CQT configuration optimized for musical analysis.
    ///
    /// Uses 12 bins per octave for chromatic scale analysis,
    /// starting from C1 (32.7 Hz) for full piano range coverage.
    pub fn musical() -> Self {
        Self {
            bins_per_octave: 12,
            fmin: to_precision(32.7), // C1
            fmax: None,
            q_factor: F::one(),
            window_type: WindowType::Hanning,
            sparsity_threshold: to_precision(0.01),
            normalize: true,
        }
    }

    /// Create a CQT configuration optimized for harmonic analysis.
    ///
    /// Uses 24 bins per octave for quarter-tone resolution,
    /// providing detailed harmonic analysis capabilities.
    pub fn harmonic() -> Self {
        Self {
            bins_per_octave: 24,
            fmin: to_precision(55.0), // A1
            fmax: None,
            q_factor: F::one(),
            window_type: WindowType::Hanning,
            sparsity_threshold: to_precision(0.005),
            normalize: true,
        }
    }

    /// Create a CQT configuration optimized for chord detection.
    ///
    /// Uses settings that balance frequency resolution with computational
    /// efficiency for real-time chord detection applications.
    pub fn chord_detection() -> Self {
        Self {
            bins_per_octave: 12,
            fmin: to_precision(82.4),
            fmax: Some(to_precision(2093.0)),
            q_factor: to_precision(0.8), // Slightly lower Q for faster response
            window_type: WindowType::Hanning,
            sparsity_threshold: to_precision(0.02),
            normalize: true,
        }
    }

    /// Create a CQT configuration optimized for onset detection.
    ///
    /// Uses lower Q factor for better time resolution,
    /// suitable for detecting note onsets and transients.
    pub fn onset_detection() -> Self {
        Self {
            bins_per_octave: 12,
            fmin: to_precision(55.0), // A1
            fmax: None,
            q_factor: to_precision(0.5), // Lower Q for better time resolution
            window_type: WindowType::Hanning,
            sparsity_threshold: to_precision(0.02),
            normalize: true,
        }
    }

    /// Set the frequency range for analysis.
    ///
    /// # Arguments
    /// * `fmin` - Minimum frequency in Hz
    /// * `fmax` - Maximum frequency in Hz (None for Nyquist)
    pub const fn set_frequency_range(&mut self, fmin: F, fmax: Option<F>) {
        self.fmin = fmin;
        self.fmax = fmax;
    }

    /// Set the number of bins per octave.
    ///
    /// # Arguments
    /// * `bins_per_octave` - Number of frequency bins per octave
    pub const fn set_bins_per_octave(&mut self, bins_per_octave: usize) {
        self.bins_per_octave = bins_per_octave;
    }

    /// Set the quality factor.
    ///
    /// # Arguments
    /// * `q_factor` - Quality factor (higher = better frequency resolution)
    pub const fn set_q_factor(&mut self, q_factor: F) {
        self.q_factor = q_factor;
    }

    /// Calculate the actual maximum frequency based on sample rate.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// The effective maximum frequency (either fmax or Nyquist frequency)
    pub fn effective_fmax(&self, sample_rate: F) -> F {
        self.fmax.unwrap_or(sample_rate / to_precision(2.0))
    }

    /// Calculate the total number of CQT bins.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Number of frequency bins in the CQT
    ///
    /// # Panics
    ///
    /// Panics if the octaves calculation results in a value that cannot be converted to usize.
    pub fn num_bins(&self, sample_rate: F) -> usize {
        let fmax = self.effective_fmax(sample_rate);
        let octaves = (fmax / self.fmin).log2();
        (octaves * to_precision::<F, _>(self.bins_per_octave))
            .ceil()
            .to_usize()
            .expect("octaves * self.bins_per_octave is a valid float to cast to")
    }

    /// Calculate the center frequency for a given bin index.
    ///
    /// # Arguments
    /// * `bin_index` - Zero-based bin index
    ///
    /// # Returns
    /// Center frequency in Hz for the specified bin
    pub fn bin_frequency(&self, bin_index: usize) -> F {
        self.fmin
            * to_precision::<F, _>(2.0)
                .powf(to_precision::<F, _>(bin_index) / to_precision::<F, _>(self.bins_per_octave))
    }

    /// Calculate the bandwidth for a given bin index.
    ///
    /// # Arguments
    /// * `bin_index` - Zero-based bin index
    ///
    /// # Returns
    /// Bandwidth in Hz for the specified bin
    pub fn bin_bandwidth(&self, bin_index: usize) -> F {
        let center_freq = self.bin_frequency(bin_index);
        center_freq / self.q_factor
    }

    /// Validate the CQT configuration parameters.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Result indicating whether the configuration is valid
    pub fn validate(&self, sample_rate: F) -> AudioSampleResult<()> {
        if self.bins_per_octave == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Bins per octave must be greater than 0",
            )));
        }

        if self.fmin <= F::zero() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Minimum frequency must be greater than 0",
            )));
        }

        let nyquist = sample_rate / to_precision(2.0);
        if self.fmin >= nyquist {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Minimum frequency must be less than Nyquist frequency",
            )));
        }

        if let Some(fmax) = self.fmax {
            if fmax <= self.fmin {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "Maximum frequency must be greater than minimum frequency",
                )));
            }
            if fmax > nyquist {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "Maximum frequency must be less than or equal to Nyquist frequency",
                )));
            }
        }

        if self.q_factor <= F::zero() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Quality factor must be greater than 0",
            )));
        }

        if self.sparsity_threshold < F::zero() || self.sparsity_threshold > F::one() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Sparsity threshold must be between F::zero() and F::one()",
            )));
        }

        // Check that we have at least one bin
        if self.num_bins(sample_rate) == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "CQT configuration results in zero frequency bins",
            )));
        }

        Ok(())
    }
}

impl<F: RealFloat> Default for CqtConfig<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Adaptive thresholding methods for peak picking.
///
/// Different methods provide different strategies for setting dynamic thresholds
/// based on the onset strength function characteristics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdaptiveThresholdMethod {
    /// Delta-based adaptive threshold using local maximum tracking.
    /// Threshold = local_max - delta, where delta is a constant offset.
    /// More responsive to sudden changes but may be sensitive to noise.
    Delta,
    /// Percentile-based adaptive threshold using rolling statistics.
    /// Threshold = percentile(local_window, percentile_value).
    /// More robust to noise but may be slower to adapt.
    Percentile,
    /// Combined delta and percentile method.
    /// Uses the maximum of both delta and percentile thresholds.
    /// Provides balance between responsiveness and robustness.
    Combined,
}

/// Configuration for adaptive thresholding in peak picking.
///
/// Adaptive thresholding dynamically adjusts the detection threshold based on
/// local characteristics of the onset strength function to improve detection
/// accuracy across varying signal conditions.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveThresholdConfig<F: RealFloat> {
    /// Method for computing adaptive threshold
    pub method: AdaptiveThresholdMethod,
    /// Delta value for delta-based thresholding (typical range: 0.01-0.1)
    /// Larger values = fewer false positives but may miss weak onsets
    pub delta: F,
    /// Percentile value for percentile-based thresholding (F::zero()-F::one())
    /// Higher percentiles = more conservative thresholding
    pub percentile: F,
    /// Size of local window for adaptive computation (in samples)
    /// Larger windows = more stable but less responsive thresholds
    pub window_size: usize,
    /// Minimum threshold value to prevent over-sensitivity
    /// Ensures threshold never drops below this absolute minimum
    pub min_threshold: F,
    /// Maximum threshold value to prevent under-sensitivity
    /// Ensures threshold never exceeds this absolute maximum
    pub max_threshold: F,
}

impl<F: RealFloat> AdaptiveThresholdConfig<F> {
    /// Create a new adaptive threshold configuration with default settings.
    ///
    /// Default configuration suitable for general onset detection:
    /// - Delta method with 0.05 delta value
    /// - Window size of 1024 samples (about 23ms at 44.1kHz)
    /// - Reasonable min/max threshold bounds
    pub fn new() -> Self {
        Self {
            method: AdaptiveThresholdMethod::Delta,
            delta: crate::to_precision(0.05),
            percentile: crate::to_precision(0.9),
            window_size: 1024,
            min_threshold: crate::to_precision(0.01),
            max_threshold: F::one(),
        }
    }

    /// Create a delta-based adaptive threshold configuration.
    ///
    /// # Arguments
    /// * `delta` - Delta value for threshold computation
    /// * `window_size` - Size of local window in samples
    pub fn delta(delta: F, window_size: usize) -> Self {
        Self {
            method: AdaptiveThresholdMethod::Delta,
            delta,
            percentile: crate::to_precision(0.9),
            window_size,
            min_threshold: crate::to_precision(0.01),
            max_threshold: F::one(),
        }
    }

    /// Create a percentile-based adaptive threshold configuration.
    ///
    /// # Arguments
    /// * `percentile` - Percentile value (F::zero()-F::one())
    /// * `window_size` - Size of local window in samples
    pub fn percentile(percentile: F, window_size: usize) -> Self {
        Self {
            method: AdaptiveThresholdMethod::Percentile,
            delta: crate::to_precision(0.05),
            percentile,
            window_size,
            min_threshold: crate::to_precision(0.01),
            max_threshold: F::one(),
        }
    }

    /// Create a combined adaptive threshold configuration.
    ///
    /// # Arguments
    /// * `delta` - Delta value for delta component
    /// * `percentile` - Percentile value for percentile component
    /// * `window_size` - Size of local window in samples
    pub fn combined(delta: F, percentile: F, window_size: usize) -> Self {
        Self {
            method: AdaptiveThresholdMethod::Combined,
            delta,
            percentile,
            window_size,
            min_threshold: crate::to_precision(0.01),
            max_threshold: F::one(),
        }
    }

    /// Set the minimum threshold value.
    pub const fn set_min_threshold(&mut self, min_threshold: F) {
        self.min_threshold = min_threshold;
    }

    /// Set the maximum threshold value.
    pub const fn set_max_threshold(&mut self, max_threshold: F) {
        self.max_threshold = max_threshold;
    }

    /// Validate the adaptive threshold configuration.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if self.delta < F::zero() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Delta must be non-negative",
            )));
        }

        if self.percentile < F::zero() || self.percentile > F::one() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Percentile must be between F::zero() and F::one()",
            )));
        }

        if self.window_size == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Window size must be greater than 0",
            )));
        }

        if self.min_threshold < F::zero() {
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

        Ok(())
    }
}

impl<F: RealFloat> Default for AdaptiveThresholdConfig<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for peak picking with temporal constraints.
///
/// Peak picking identifies local maxima in the onset strength function that
/// exceed a threshold. Temporal constraints ensure detected peaks are
/// separated by minimum time intervals and can include smoothing.
#[derive(Debug, Clone, PartialEq)]
pub struct PeakPickingConfig<F: RealFloat> {
    /// Adaptive threshold configuration
    pub adaptive_threshold: AdaptiveThresholdConfig<F>,
    /// Minimum time separation between peaks (in samples)
    /// Prevents detecting multiple peaks for the same onset event
    pub min_peak_separation: usize,
    /// Enable pre-emphasis to enhance transient detection
    /// Applies high-pass filtering to emphasize onset characteristics
    pub pre_emphasis: bool,
    /// Pre-emphasis coefficient (F::zero()-F::one()) for high-pass filtering
    /// Higher values = stronger emphasis on transients
    pub pre_emphasis_coeff: F,
    /// Enable median filtering for onset strength smoothing
    /// Reduces noise while preserving peak structure
    pub median_filter: bool,
    /// Median filter length (must be odd)
    /// Larger values = more smoothing but may blur sharp onsets
    pub median_filter_length: usize,
    /// Normalize onset strength before peak picking
    /// Ensures consistent detection across different signal levels
    pub normalize_onset_strength: bool,
    /// Normalization method for onset strength
    pub normalization_method: NormalizationMethod,
}

impl<F: RealFloat> PeakPickingConfig<F> {
    /// Create a new peak picking configuration with default settings.
    ///
    /// Default configuration optimized for general onset detection:
    /// - Adaptive delta thresholding
    /// - 512 samples minimum separation (about 11ms at 44.1kHz)
    /// - Pre-emphasis enabled with moderate coefficient
    /// - Median filtering enabled with small kernel
    /// - Peak normalization enabled
    pub fn new() -> Self {
        Self {
            adaptive_threshold: AdaptiveThresholdConfig::new(),
            min_peak_separation: 512,
            pre_emphasis: true,
            pre_emphasis_coeff: to_precision(0.97),
            median_filter: true,
            median_filter_length: 3,
            normalize_onset_strength: true,
            normalization_method: NormalizationMethod::Peak,
        }
    }

    /// Create a configuration optimized for music onset detection.
    ///
    /// Uses settings that work well for typical musical content:
    /// - Combined adaptive thresholding for robustness
    /// - Longer minimum separation to avoid over-detection
    /// - Strong pre-emphasis for transient enhancement
    /// - Median filtering for noise reduction
    pub fn music() -> Self {
        Self {
            adaptive_threshold: AdaptiveThresholdConfig::combined(
                to_precision(0.03),
                to_precision(0.85),
                2048,
            ),
            min_peak_separation: 1024,
            pre_emphasis: true,
            pre_emphasis_coeff: to_precision(0.95),
            median_filter: true,
            median_filter_length: 5,
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
    pub fn speech() -> Self {
        Self {
            adaptive_threshold: AdaptiveThresholdConfig::delta(to_precision(0.07), 1024),
            min_peak_separation: 256,
            pre_emphasis: true,
            pre_emphasis_coeff: to_precision(0.98),
            median_filter: true,
            median_filter_length: 3,
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
    pub fn drums() -> Self {
        Self {
            adaptive_threshold: AdaptiveThresholdConfig::percentile(to_precision(0.95), 512),
            min_peak_separation: 128,
            pre_emphasis: true,
            pre_emphasis_coeff: to_precision(0.93),
            median_filter: false,
            median_filter_length: 3,
            normalize_onset_strength: true,
            normalization_method: NormalizationMethod::Peak,
        }
    }

    /// Set the minimum peak separation in samples.
    pub const fn set_min_peak_separation(&mut self, samples: usize) {
        self.min_peak_separation = samples;
    }

    /// Set the minimum peak separation in milliseconds.
    ///
    /// # Panics
    ///
    /// Panics if the millisecond to sample conversion results in a value that cannot be converted to usize.
    pub fn set_min_peak_separation_ms(&mut self, ms: F, sample_rate: F) {
        self.min_peak_separation = (ms * sample_rate / to_precision(1000.0)).to_usize().expect("Given positive ms and sample rate this value will always be >= 0 which can be cast to a usize");
    }

    /// Enable or disable pre-emphasis.
    pub const fn set_pre_emphasis(&mut self, enabled: bool, coeff: F) {
        self.pre_emphasis = enabled;
        self.pre_emphasis_coeff = coeff;
    }

    /// Enable or disable median filtering.
    pub const fn set_median_filter(&mut self, enabled: bool, length: usize) {
        self.median_filter = enabled;
        self.median_filter_length = length;
    }

    /// Validate the peak picking configuration.
    pub fn validate(&self) -> AudioSampleResult<()> {
        self.adaptive_threshold.validate()?;

        if self.min_peak_separation == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Minimum peak separation must be greater than 0",
            )));
        }

        if self.pre_emphasis_coeff < F::zero() || self.pre_emphasis_coeff > F::one() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Pre-emphasis coefficient must be between F::zero() and F::one()",
            )));
        }

        if self.median_filter_length == 0 || self.median_filter_length.is_multiple_of(2) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Median filter length must be a positive odd integer",
            )));
        }

        Ok(())
    }
}

impl<F: RealFloat> Default for PeakPickingConfig<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for onset detection algorithms.
///
/// Onset detection identifies points in time where new musical events begin,
/// such as note onsets, drum hits, or other transient events. The energy-based
/// method analyzes changes in spectral energy over time to identify onsets.
#[derive(Debug, Clone, PartialEq)]
pub struct OnsetConfig<F: RealFloat> {
    /// CQT configuration for spectral analysis
    /// Uses CqtConfig::onset_detection() by default for optimal time resolution
    pub cqt_config: CqtConfig<F>,
    /// Hop size for frame-based analysis in samples
    /// Smaller values provide better time resolution but increase computation
    pub hop_size: usize,
    /// Window size for CQT analysis in samples (None = auto-calculate)
    /// If None, calculated as 4 periods of the minimum frequency
    pub window_size: Option<usize>,
    /// Threshold for onset detection (F::zero() to F::one())
    /// Higher values = fewer, more confident onsets
    /// Lower values = more onsets, potentially including false positives
    pub threshold: F,
    /// Minimum time between onsets in seconds
    /// Prevents detection of spurious onsets too close together
    pub min_onset_interval: F,
    /// Pre-emphasis factor for spectral flux (F::zero() to F::one())
    /// Emphasizes high-frequency content which often carries onset information
    pub pre_emphasis: F,
    /// Whether to use adaptive thresholding
    /// Adapts threshold based on local energy characteristics
    pub adaptive_threshold: bool,
    /// Median filter length for adaptive thresholding (in frames)
    /// Used to smooth the threshold over time
    pub median_filter_length: usize,
    /// Multiplier for adaptive threshold
    /// threshold = median_value * adaptive_threshold_multiplier
    pub adaptive_threshold_multiplier: F,
    /// Peak picking configuration for onset detection
    pub peak_picking: PeakPickingConfig<F>,
}

impl<F: RealFloat> OnsetConfig<F> {
    /// Create a new onset detection configuration with default settings.
    ///
    /// Default configuration suitable for general onset detection:
    /// - CQT optimized for onset detection (Q=0.5)
    /// - 512 sample hop size (good time resolution)
    /// - Auto-calculated window size
    /// - Moderate threshold (0.3)
    /// - 50ms minimum onset interval
    /// - Adaptive thresholding enabled
    pub fn new() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 512,
            window_size: None,
            threshold: to_precision(0.3),

            min_onset_interval: to_precision(0.05),
            pre_emphasis: to_precision(F::zero()),
            adaptive_threshold: true,
            median_filter_length: 5,
            adaptive_threshold_multiplier: to_precision(1.5),
            peak_picking: PeakPickingConfig::new(),
        }
    }

    /// Create configuration optimized for percussive onset detection.
    ///
    /// Optimized for detecting drum hits and other percussive events:
    /// - Higher threshold for cleaner detection
    /// - Shorter minimum interval for rapid percussion
    /// - Pre-emphasis to highlight transients
    pub fn percussive() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 256, // Higher time resolution for drums
            window_size: None,
            threshold: to_precision(0.5), // Higher threshold for cleaner detection
            min_onset_interval: to_precision(0.03), // 30ms for rapid percussion
            pre_emphasis: to_precision(0.3), // Emphasize high frequencies
            adaptive_threshold: true,
            median_filter_length: 3,
            adaptive_threshold_multiplier: to_precision(2.0),
            peak_picking: PeakPickingConfig::drums(),
        }
    }

    /// Create configuration optimized for musical onset detection.
    ///
    /// Optimized for detecting note onsets in musical instruments:
    /// - Moderate threshold for good sensitivity
    /// - Longer minimum interval for typical musical phrasing
    /// - Less pre-emphasis for tonal content
    pub fn musical() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 512,
            window_size: None,
            threshold: to_precision(0.25), // Lower threshold for musical content
            min_onset_interval: to_precision(0.1), // 100ms for musical phrasing
            pre_emphasis: to_precision(0.1), // Light pre-emphasis
            adaptive_threshold: true,
            median_filter_length: 7,
            adaptive_threshold_multiplier: to_precision(1.2),
            peak_picking: PeakPickingConfig::music(),
        }
    }

    /// Create configuration optimized for speech onset detection.
    ///
    /// Optimized for detecting word/syllable onsets in speech:
    /// - Low threshold for speech dynamics
    /// - Moderate minimum interval for speech rate
    /// - Minimal pre-emphasis for speech clarity
    pub fn speech() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 256, // Good time resolution for speech
            window_size: None,
            threshold: to_precision(0.2), // Low threshold for speech dynamics
            min_onset_interval: to_precision(0.08), // 80ms for speech rate
            pre_emphasis: to_precision(0.05), // Minimal pre-emphasis
            adaptive_threshold: true,
            median_filter_length: 9,
            adaptive_threshold_multiplier: to_precision(1.1),
            peak_picking: PeakPickingConfig::speech(),
        }
    }

    /// Set the hop size for frame-based analysis.
    ///
    /// # Arguments
    /// * `hop_size` - Hop size in samples (must be > 0)
    pub const fn set_hop_size(&mut self, hop_size: usize) {
        self.hop_size = hop_size;
    }

    /// Set the onset detection threshold.
    ///
    /// # Arguments
    /// * `threshold` - Threshold value (F::zero() to F::one())
    pub fn set_threshold(&mut self, threshold: F) {
        self.threshold = threshold.clamp(F::zero(), F::one());
    }

    /// Set the minimum time between onsets.
    ///
    /// # Arguments
    /// * `interval_seconds` - Minimum interval in seconds (must be > 0)
    pub fn set_min_onset_interval(&mut self, interval_seconds: F) {
        self.min_onset_interval = interval_seconds.max(to_precision(0.001)); // At least 1ms
    }

    /// Enable or disable adaptive thresholding.
    ///
    /// # Arguments
    /// * `enabled` - Whether to use adaptive thresholding
    pub const fn set_adaptive_threshold(&mut self, enabled: bool) {
        self.adaptive_threshold = enabled;
    }

    /// Set the pre-emphasis factor for spectral flux.
    ///
    /// # Arguments
    /// * `pre_emphasis` - Pre-emphasis factor (F::zero() to F::one())
    pub fn set_pre_emphasis(&mut self, pre_emphasis: F) {
        self.pre_emphasis = pre_emphasis.clamp(F::zero(), F::one());
    }

    /// Calculate the effective window size for CQT analysis.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Window size in samples
    ///
    /// # Panics
    ///
    /// Panics if the minimum period calculation results in a value that cannot be converted to usize.
    pub fn effective_window_size(&self, sample_rate: F) -> usize {
        self.window_size.unwrap_or_else(|| {
            // Auto-calculate based on lowest frequency (4 periods for good resolution)
            let min_period = sample_rate / self.cqt_config.fmin;
            (min_period * to_precision(4.0))
                .to_usize()
                .expect("Should be castable to usize")
        })
    }

    /// Calculate the time resolution of onset detection.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Time resolution in seconds
    pub fn time_resolution(&self, sample_rate: F) -> F {
        to_precision::<F, _>(self.hop_size) / sample_rate
    }

    /// Convert onset time from frames to seconds.
    ///
    /// # Arguments
    /// * `frame_index` - Frame index from onset detection
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Time in seconds
    pub fn frame_to_seconds(&self, frame_index: usize, sample_rate: F) -> F {
        (to_precision::<F, _>(frame_index) * to_precision::<F, _>(self.hop_size)) / sample_rate
    }

    /// Convert onset time from seconds to frames.
    ///
    /// # Arguments
    /// * `time_seconds` - Time in seconds
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Frame index
    ///
    /// # Panics
    ///
    /// Panics if the time to frame conversion results in a value that cannot be converted to usize.
    pub fn seconds_to_frame(&self, time_seconds: F, sample_rate: F) -> usize {
        ((time_seconds * sample_rate) / to_precision::<F, _>(self.hop_size).round())
            .to_usize()
            .expect("Should be castable to usize")
    }

    /// Validate the onset detection configuration.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Result indicating whether the configuration is valid
    ///
    /// # Panics
    ///
    /// Panics if time resolution conversion to f64 fails.
    pub fn validate(&self, sample_rate: F) -> AudioSampleResult<()> {
        // Validate CQT configuration
        self.cqt_config.validate(sample_rate)?;

        // Validate hop size
        if self.hop_size == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Hop size must be greater than 0",
            )));
        }

        // Validate threshold
        if self.threshold < F::zero() || self.threshold > F::one() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Threshold must be between F::zero() and F::one()",
            )));
        }

        // Validate minimum onset interval
        if self.min_onset_interval <= F::zero() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Minimum onset interval must be greater than 0",
            )));
        }

        // Validate pre-emphasis
        if self.pre_emphasis < F::zero() || self.pre_emphasis > F::one() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Pre-emphasis factor must be between F::zero() and F::one()",
            )));
        }

        // Validate window size if specified
        if let Some(window_size) = self.window_size
            && (window_size == 0)
        {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Window size must be greater than 0",
            )));
        }

        // Validate adaptive threshold parameters
        if self.adaptive_threshold && self.median_filter_length == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Median filter length must be greater than 0",
            )));
        }

        // Check that time resolution is reasonable
        let time_resolution = self.time_resolution(sample_rate);
        if time_resolution > to_precision(0.1) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                format!(
                    "Time resolution ({:.3}s) is too low. Consider reducing hop size.",
                    time_resolution
                        .to_f64()
                        .expect("We know this is at least a f32, so f64 conversion is safe")
                ),
            )));
        }

        Ok(())
    }
}

impl<F: RealFloat> Default for OnsetConfig<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Spectral flux method variants for onset detection.
///
/// Different spectral flux methods provide different characteristics for
/// detecting different types of onsets and musical events.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpectralFluxMethod {
    /// Simple energy-based flux: sum of positive energy differences
    /// ∆E\[n\] = Σ(max(0, |X\[k,n\]|² - |X\[k,n-1\]|²)) for all frequency bins k
    /// Good for general onset detection, especially percussive events
    Energy,
    /// Magnitude-based flux: sum of positive magnitude differences
    /// ∆M\[n\] = Σ(max(0, |X\[k,n\]| - |X\[k,n-1\]|)) for all frequency bins k
    /// More sensitive to subtle onsets, good for tonal instruments
    Magnitude,
    /// Complex domain flux: uses phase information
    /// Takes into account both magnitude and phase changes
    /// More robust to noise but computationally intensive
    Complex,
    /// Rectified complex domain flux: removes negative phase contributions
    /// Balances sensitivity with robustness
    RectifiedComplex,
}

/// Configuration for spectral flux onset detection.
///
/// Spectral flux measures the rate of change of the magnitude spectrum
/// between consecutive frames, providing effective onset detection for
/// both percussive and tonal instruments.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralFluxConfig<F: RealFloat> {
    /// CQT configuration for spectral analysis
    pub cqt_config: CqtConfig<F>,
    /// Hop size for frame-based analysis in samples
    pub hop_size: usize,
    /// Window size for CQT analysis in samples (None = auto-calculate)
    pub window_size: Option<usize>,
    /// Spectral flux method to use
    pub flux_method: SpectralFluxMethod,
    /// Peak picking configuration for onset detection
    pub peak_picking: PeakPickingConfig<F>,
    /// Apply rectification to spectral flux (keep only positive values)
    pub rectify: bool,
    /// Logarithmic compression factor for spectral flux
    /// flux_compressed = log(1 + C * flux) where C is this parameter
    pub log_compression: F,
}

impl<F: RealFloat> SpectralFluxConfig<F> {
    /// Create a new spectral flux configuration with default settings.
    pub fn new() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 512,
            window_size: None,
            flux_method: SpectralFluxMethod::Energy,
            peak_picking: PeakPickingConfig::new(),
            rectify: true,
            log_compression: to_precision(100.0),
        }
    }

    /// Create configuration optimized for percussive onset detection.
    pub fn percussive() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 256,
            window_size: None,
            flux_method: SpectralFluxMethod::Energy,
            peak_picking: PeakPickingConfig::drums(),
            rectify: true,
            log_compression: to_precision(1000.0),
        }
    }

    /// Create configuration optimized for musical onset detection.
    pub fn musical() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 512,
            window_size: None,
            flux_method: SpectralFluxMethod::Magnitude,
            peak_picking: PeakPickingConfig::music(),
            rectify: true,
            log_compression: to_precision(100.0),
        }
    }

    /// Create configuration optimized for complex domain onset detection.
    pub fn complex() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 512,
            window_size: None,
            flux_method: SpectralFluxMethod::Complex,
            peak_picking: PeakPickingConfig::new(),
            rectify: false,
            log_compression: to_precision(100.0),
        }
    }

    /// Validate the spectral flux configuration.
    pub fn validate(&self, sample_rate: F) -> AudioSampleResult<()> {
        self.cqt_config.validate(sample_rate)?;
        self.peak_picking.validate()?;

        if self.hop_size == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Hop size must be greater than 0",
            )));
        }

        if let Some(window_size) = self.window_size
            && (window_size == 0)
        {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Window size must be greater than 0",
            )));
        }

        if self.log_compression < F::zero() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Log compression factor must be non-negative",
            )));
        }

        Ok(())
    }
}

impl<F: RealFloat> Default for SpectralFluxConfig<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for complex domain onset detection.
///
/// Complex domain onset detection uses both magnitude and phase information
/// from the CQT to provide more accurate onset detection than magnitude-only
/// methods, especially for polyphonic music and complex timbres.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexOnsetConfig<F: RealFloat> {
    /// CQT configuration for spectral analysis
    pub cqt_config: CqtConfig<F>,
    /// Hop size for frame-based analysis in samples
    pub hop_size: usize,
    /// Window size for CQT analysis in samples (None = auto-calculate)
    pub window_size: Option<usize>,
    /// Peak picking configuration for onset detection
    pub peak_picking: PeakPickingConfig<F>,
    /// Weight for magnitude-based detection (F::zero()-F::one())
    pub magnitude_weight: F,
    /// Weight for phase-based detection (F::zero()-F::one())
    pub phase_weight: F,
    /// Apply magnitude rectification (keep only positive changes)
    pub magnitude_rectify: bool,
    /// Apply phase rectification (keep only positive phase deviations)
    pub phase_rectify: bool,
    /// Logarithmic compression factor for combined onset function
    pub log_compression: F,
}

impl<F: RealFloat> ComplexOnsetConfig<F> {
    /// Create a new complex onset configuration with default settings.
    pub fn new() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 512,
            window_size: None,
            peak_picking: PeakPickingConfig::new(),
            magnitude_weight: to_precision(0.7),
            phase_weight: to_precision(0.3),
            magnitude_rectify: true,
            phase_rectify: true,
            log_compression: to_precision(100.0),
        }
    }

    /// Create configuration optimized for percussive onset detection.
    pub fn percussive() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 256,
            window_size: None,
            peak_picking: PeakPickingConfig::drums(),
            magnitude_weight: to_precision(0.8),
            phase_weight: to_precision(0.2),
            magnitude_rectify: true,
            phase_rectify: true,
            log_compression: to_precision(1000.0),
        }
    }

    /// Create configuration optimized for musical onset detection.
    pub fn musical() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 512,
            window_size: None,
            peak_picking: PeakPickingConfig::music(),
            magnitude_weight: to_precision(0.6),
            phase_weight: to_precision(0.4),
            magnitude_rectify: true,
            phase_rectify: true,
            log_compression: to_precision(100.0),
        }
    }

    /// Create configuration optimized for speech onset detection.
    pub fn speech() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 256,
            window_size: None,
            peak_picking: PeakPickingConfig::speech(),
            magnitude_weight: to_precision(0.5),
            phase_weight: to_precision(0.5),
            magnitude_rectify: true,
            phase_rectify: false,
            log_compression: to_precision(50.0),
        }
    }

    /// Set the magnitude and phase weights.
    pub fn set_weights(&mut self, magnitude_weight: F, phase_weight: F) {
        self.magnitude_weight = magnitude_weight.clamp(F::zero(), F::one());
        self.phase_weight = phase_weight.clamp(F::zero(), F::one());
    }

    /// Validate the complex onset configuration.
    pub fn validate(&self, sample_rate: F) -> AudioSampleResult<()> {
        self.cqt_config.validate(sample_rate)?;
        self.peak_picking.validate()?;

        if self.hop_size == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Hop size must be greater than 0",
            )));
        }

        if let Some(window_size) = self.window_size
            && (window_size == 0)
        {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Window size must be greater than 0",
            )));
        }

        if self.magnitude_weight < F::zero() || self.magnitude_weight > F::one() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Magnitude weight must be between F::zero() and F::one()",
            )));
        }

        if self.phase_weight < F::zero() || self.phase_weight > F::one() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Phase weight must be between F::zero() and F::one()",
            )));
        }

        // Both weights cannot be zero
        if self.magnitude_weight == F::zero() && self.phase_weight == F::zero() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "At least one of magnitude or phase weight must be greater than 0",
            )));
        }

        if self.log_compression < F::zero() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Log compression factor must be non-negative",
            )));
        }

        Ok(())
    }
}

impl<F: RealFloat> Default for ComplexOnsetConfig<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Noise color types for audio perturbation.
///
/// Different noise colors have different spectral characteristics:
/// - White noise: Equal power across all frequencies
/// - Pink noise: Equal power per octave (1/f spectrum)
/// - Brown noise: Power decreases at -6dB per octave (1/f² spectrum)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseColor {
    /// White noise - equal power across all frequencies
    White,
    /// Pink noise - equal power per octave
    Pink,
    /// Brown (red) noise - 1/f² spectrum
    Brown,
}

/// Perturbation methods for audio data augmentation.
///
/// Each variant defines a specific type of perturbation that can be applied
/// to audio samples for data augmentation, robustness testing, or creative effects.
#[derive(Debug, Clone, PartialEq)]
pub enum PerturbationMethod<F: RealFloat> {
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
        target_snr_db: F,
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
        min_gain_db: F,
        /// Maximum gain in dB
        max_gain_db: F,
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
        cutoff_hz: F,
        /// Filter slope in dB per octave (None = default 6dB/octave)
        slope_db_per_octave: Option<F>,
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
        semitones: F,
        /// Whether to attempt formant preservation (basic implementation)
        preserve_formants: bool,
    },
}

impl<F: RealFloat> PerturbationMethod<F> {
    /// Create a Gaussian noise perturbation configuration.
    ///
    /// # Arguments
    /// * `target_snr_db` - Target signal-to-noise ratio in dB
    /// * `noise_color` - Color/spectrum of the noise
    pub const fn gaussian_noise(target_snr_db: F, noise_color: NoiseColor) -> Self {
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
    pub const fn random_gain(min_gain_db: F, max_gain_db: F) -> Self {
        Self::RandomGain {
            min_gain_db,
            max_gain_db,
        }
    }

    /// Create a high-pass filter perturbation configuration.
    ///
    /// # Arguments
    /// * `cutoff_hz` - Cutoff frequency in Hz
    pub const fn high_pass_filter(cutoff_hz: F) -> Self {
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
    pub const fn high_pass_filter_with_slope(cutoff_hz: F, slope_db_per_octave: F) -> Self {
        Self::HighPassFilter {
            cutoff_hz,
            slope_db_per_octave: Some(slope_db_per_octave),
        }
    }

    /// Create a pitch shift perturbation configuration.
    ///
    /// # Arguments
    /// * `semitones` - Pitch shift in semitones
    /// * `preserve_formants` - Whether to preserve formants
    pub const fn pitch_shift(semitones: F, preserve_formants: bool) -> Self {
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
    pub fn validate(&self, sample_rate: F) -> AudioSampleResult<()> {
        match self {
            Self::GaussianNoise { target_snr_db, .. } => {
                if *target_snr_db < to_precision(-60.0) || *target_snr_db > to_precision(60.0) {
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
                if *min_gain_db < to_precision(-40.0) || *max_gain_db > to_precision(20.0) {
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
                let nyquist = sample_rate / to_precision(2.0);
                if *cutoff_hz <= F::zero() || *cutoff_hz >= nyquist {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "parameter",
                        format!(
                            "Cutoff frequency must be between 0 and Nyquist ({:.1} Hz)",
                            nyquist
                        ),
                    )));
                }
                if let Some(slope) = slope_db_per_octave
                    && (*slope < F::zero() || *slope > to_precision(48.0))
                {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "parameter",
                        "Slope must be between 0 and 48 dB/octave",
                    )));
                }
            }
            Self::PitchShift { semitones, .. } => {
                if semitones.abs() > to_precision(12.0) {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "parameter",
                        "Pitch shift should be between -12 and +12 semitones",
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Configuration for audio perturbation operations.
///
/// This struct defines how audio samples should be perturbed for data augmentation,
/// robustness testing, or creative effects.
#[derive(Debug, Clone, PartialEq)]
pub struct PerturbationConfig<F: RealFloat> {
    /// The perturbation method to apply
    pub method: PerturbationMethod<F>,
    /// Optional random seed for deterministic perturbation
    /// If None, uses thread-local random number generator
    pub seed: Option<u64>,
}

impl<F: RealFloat> PerturbationConfig<F> {
    /// Create a new perturbation configuration.
    ///
    /// # Arguments
    /// * `method` - The perturbation method to apply
    pub const fn new(method: PerturbationMethod<F>) -> Self {
        Self { method, seed: None }
    }

    /// Create a new perturbation configuration with a specific seed.
    ///
    /// # Arguments
    /// * `method` - The perturbation method to apply
    /// * `seed` - Random seed for deterministic results
    pub const fn with_seed(method: PerturbationMethod<F>, seed: u64) -> Self {
        Self {
            method,
            seed: Some(seed),
        }
    }

    /// Set the random seed for deterministic perturbation.
    pub const fn set_seed(&mut self, seed: u64) {
        self.seed = Some(seed);
    }

    /// Clear the random seed to use non-deterministic perturbation.
    pub const fn clear_seed(&mut self) {
        self.seed = None;
    }

    /// Validate the perturbation configuration.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Result indicating whether the configuration is valid
    pub fn validate(&self, sample_rate: F) -> AudioSampleResult<()> {
        self.method.validate(sample_rate)
    }
}

impl<F: RealFloat> Default for PerturbationConfig<F> {
    fn default() -> Self {
        Self::new(PerturbationMethod::GaussianNoise {
            target_snr_db: to_precision(20.0),
            noise_color: NoiseColor::White,
        })
    }
}

/// Supported serialization formats for AudioSamples.
///
/// This enum defines the various file formats that can be used for saving and
/// loading AudioSamples data, focusing on data analysis and interchange formats
/// rather than traditional audio file formats like WAV or MP3.
#[cfg(feature = "serialization")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializationFormat {
    /// Plain text format with configurable delimiter.
    /// Saves samples as human-readable text with metadata header.
    Text {
        /// Character used to separate values
        delimiter: TextDelimiter,
    },
    /// NumPy binary format (.npy) for single arrays.
    /// Compatible with NumPy's save/load functions.
    Numpy,
    /// NumPy compressed format (.npz) for multiple arrays with metadata.
    /// Uses ZIP compression to reduce file size.
    NumpyCompressed {
        /// Compression level (0-9, where 9 is maximum compression)
        compression_level: u32,
    },
    /// JSON format with full metadata preservation.
    /// Human-readable but larger file sizes.
    Json,
    /// CSV (Comma-Separated Values) format with headers.
    /// Compatible with spreadsheet applications and data analysis tools.
    Csv,
    /// Custom binary format with configurable endianness.
    /// Compact and fast but specific to this library.
    Binary {
        /// Byte order for multi-byte values
        endian: Endianness,
    },
    /// MessagePack binary format for efficient serialization.
    /// Compact binary format with wide language support.
    MessagePack,
    /// CBOR (Concise Binary Object Representation) format.
    /// Standardized binary format (RFC 7049).
    Cbor,
}

/// Text delimiters for plain text serialization formats.
#[cfg(feature = "serialization")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextDelimiter {
    /// Space character separator
    Space,
    /// Tab character separator
    Tab,
    /// Comma separator
    Comma,
    /// Newline separator (each sample on a new line)
    Newline,
    /// Custom character separator
    Custom(char),
}

/// Byte order specification for binary formats.
#[cfg(feature = "serialization")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    /// Little-endian byte order (least significant byte first)
    Little,
    /// Big-endian byte order (most significant byte first)
    Big,
    /// Native byte order of the current platform
    Native,
}

/// Configuration for serialization operations.
///
/// This struct provides fine-grained control over how AudioSamples are
/// serialized to and deserialized from various file formats.
#[cfg(feature = "serialization")]
#[derive(Debug, Clone)]
pub struct SerializationConfig<F: RealFloat> {
    /// Format to use for serialization
    pub format: SerializationFormat,
    /// Whether to include metadata (sample rate, channel info, etc.)
    pub include_metadata: bool,
    /// Floating point precision for text-based formats
    /// None uses default precision for the type
    pub precision: Option<usize>,
    /// Compression level for formats that support it (0-9)
    pub compression_level: Option<u32>,
    /// Custom headers/attributes to include in supported formats
    pub custom_headers: Option<std::collections::HashMap<String, String>>,
    /// Whether to normalize data before serialization
    pub normalize_before_save: bool,
    /// Normalization method if normalize_before_save is true
    pub normalization_method: NormalizationMethod,
    /// Whether to validate data integrity after round-trip
    pub validate_roundtrip: bool,
    /// Floating point type for precision calculations
    pub _phantom: std::marker::PhantomData<F>,
}

#[cfg(feature = "serialization")]
impl<F: RealFloat> SerializationConfig<F> {
    /// Create a new serialization configuration with default settings.
    pub const fn new(format: SerializationFormat) -> Self {
        Self {
            format,
            include_metadata: true,
            precision: None,
            compression_level: None,
            custom_headers: None,
            normalize_before_save: false,
            normalization_method: NormalizationMethod::Peak,
            validate_roundtrip: false,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a configuration optimized for CSV export.
    pub const fn csv() -> Self {
        Self::new(SerializationFormat::Csv)
            .with_precision(6)
            .with_metadata(true)
    }

    /// Create a configuration optimized for JSON export with metadata.
    pub const fn json() -> Self {
        Self::new(SerializationFormat::Json)
            .with_precision(8)
            .with_metadata(true)
    }

    /// Create a configuration optimized for NumPy compatibility.
    pub const fn numpy() -> Self {
        Self::new(SerializationFormat::Numpy).with_metadata(false) // NumPy format doesn't support metadata natively
    }

    /// Create a configuration optimized for compressed NumPy format.
    pub const fn numpy_compressed(compression_level: u32) -> Self {
        Self::new(SerializationFormat::NumpyCompressed { compression_level }).with_metadata(true)
    }

    /// Create a configuration optimized for binary format.
    pub const fn binary(endian: Endianness) -> Self {
        Self::new(SerializationFormat::Binary { endian }).with_metadata(true)
    }

    /// Create a configuration optimized for MessagePack format.
    pub const fn messagepack() -> Self {
        Self::new(SerializationFormat::MessagePack).with_metadata(true)
    }

    /// Create a configuration optimized for CBOR format.
    pub const fn cbor() -> Self {
        Self::new(SerializationFormat::Cbor).with_metadata(true)
    }

    /// Set whether to include metadata in the serialized output.
    pub const fn with_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    /// Set the floating point precision for text-based formats.
    pub const fn with_precision(mut self, precision: usize) -> Self {
        self.precision = Some(precision);
        self
    }

    /// Set the compression level for compressible formats.
    pub fn with_compression(mut self, level: u32) -> Self {
        self.compression_level = Some(level.min(9));
        self
    }

    /// Add custom headers to the serialized data.
    pub fn with_custom_headers(
        mut self,
        headers: std::collections::HashMap<String, String>,
    ) -> Self {
        self.custom_headers = Some(headers);
        self
    }

    /// Enable normalization before saving.
    pub const fn with_normalization(mut self, method: NormalizationMethod) -> Self {
        self.normalize_before_save = true;
        self.normalization_method = method;
        self
    }

    /// Enable validation of round-trip serialization accuracy.
    pub const fn with_validation(mut self, validate: bool) -> Self {
        self.validate_roundtrip = validate;
        self
    }

    /// Validate the serialization configuration.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if let Some(precision) = self.precision
            && (precision == 0 || precision > 17)
        {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "precision",
                "Precision must be between 1 and 17 digits",
            )));
        }

        if let Some(compression) = self.compression_level
            && compression > 9
        {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "compression_level",
                "Compression level must be between 0 and 9",
            )));
        }

        // Validate format-specific constraints
        match self.format {
            SerializationFormat::NumpyCompressed { compression_level } => {
                if compression_level > 9 {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "compression_level",
                        "NumPy compression level must be between 0 and 9",
                    )));
                }
            }
            SerializationFormat::Text { delimiter } => {
                if matches!(delimiter, TextDelimiter::Custom('\0')) {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "delimiter",
                        "Null character is not a valid delimiter",
                    )));
                }
            }
            _ => {} // Other formats don't have specific constraints
        }

        Ok(())
    }
}

#[cfg(feature = "serialization")]
impl<F: RealFloat> Default for SerializationConfig<F> {
    fn default() -> Self {
        Self::new(SerializationFormat::Json)
    }
}

/// Configuration for Harmonic/Percussive Source Separation (HPSS).
///
/// HPSS separates audio into harmonic and percussive components using
/// STFT magnitude median filtering. Harmonic components are enhanced
/// by median filtering along the time axis, while percussive components
/// are enhanced by median filtering along the frequency axis.
#[derive(Debug, Clone, PartialEq)]
pub struct HpssConfig<F: RealFloat> {
    /// STFT window size in samples
    /// Larger windows provide better frequency resolution but lower time resolution
    pub win_size: usize,
    /// STFT hop size in samples
    /// Smaller hop sizes provide better time resolution but increase computation
    pub hop_size: usize,
    /// Harmonic median filter kernel size (along time axis)
    /// Larger values strengthen harmonic separation but may blur transients
    pub median_filter_harmonic: usize,
    /// Percussive median filter kernel size (along frequency axis)
    /// Larger values strengthen percussive separation but may blur tonal content
    pub median_filter_percussive: usize,
    /// Soft masking parameter (0.0 = hard mask, 1.0 = completely soft)
    /// Soft masking provides smoother component separation but less isolation
    pub mask_softness: F,
}

impl<F: RealFloat> HpssConfig<F> {
    /// Create a new HPSS configuration with default settings.
    ///
    /// Default configuration suitable for general harmonic/percussive separation:
    /// - 2048 sample window (good frequency resolution)
    /// - 512 sample hop size (good time resolution)
    /// - Harmonic kernel: 17 (enhances sustained tones)
    /// - Percussive kernel: 17 (enhances transients)
    /// - Moderate soft masking (0.3)
    pub fn new() -> Self {
        Self {
            win_size: 2048,
            hop_size: 512,
            median_filter_harmonic: 17,
            median_filter_percussive: 17,
            mask_softness: to_precision(0.3),
        }
    }

    /// Create configuration optimized for musical content.
    ///
    /// Uses larger filters for stronger separation, suitable for complex musical material:
    /// - Larger harmonic kernel for better tonal separation
    /// - Larger percussive kernel for cleaner transient isolation
    /// - Softer masking for more musical results
    pub fn musical() -> Self {
        Self {
            win_size: 2048,
            hop_size: 512,
            median_filter_harmonic: 31,
            median_filter_percussive: 31,
            mask_softness: to_precision(0.5),
        }
    }

    /// Create configuration optimized for percussive content.
    ///
    /// Uses asymmetric filters favoring percussive separation:
    /// - Moderate harmonic filtering
    /// - Strong percussive filtering
    /// - Harder masking for cleaner drum isolation
    pub fn percussive() -> Self {
        Self {
            win_size: 2048,
            hop_size: 256,  // Better time resolution for transients
            median_filter_harmonic: 17,
            median_filter_percussive: 51,  // Stronger percussive filtering
            mask_softness: to_precision(0.1),  // Harder masking
        }
    }

    /// Create configuration optimized for harmonic content.
    ///
    /// Uses asymmetric filters favoring harmonic separation:
    /// - Strong harmonic filtering
    /// - Moderate percussive filtering
    /// - Harder masking for cleaner tonal isolation
    pub fn harmonic() -> Self {
        Self {
            win_size: 4096,  // Better frequency resolution for harmonics
            hop_size: 512,
            median_filter_harmonic: 51,  // Stronger harmonic filtering
            median_filter_percussive: 17,
            mask_softness: to_precision(0.1),  // Harder masking
        }
    }

    /// Create configuration for real-time processing.
    ///
    /// Uses smaller window and filter sizes for lower latency:
    /// - Smaller window for reduced latency
    /// - Smaller hop size for responsiveness
    /// - Smaller filters for faster processing
    pub fn realtime() -> Self {
        Self {
            win_size: 1024,
            hop_size: 256,
            median_filter_harmonic: 11,
            median_filter_percussive: 11,
            mask_softness: to_precision(0.3),
        }
    }

    /// Set STFT parameters.
    ///
    /// # Arguments
    /// * `win_size` - Window size in samples (should be power of 2)
    /// * `hop_size` - Hop size in samples (typically win_size/4)
    pub const fn set_stft_params(&mut self, win_size: usize, hop_size: usize) {
        self.win_size = win_size;
        self.hop_size = hop_size;
    }

    /// Set median filter sizes.
    ///
    /// # Arguments
    /// * `harmonic` - Harmonic filter size (odd numbers recommended)
    /// * `percussive` - Percussive filter size (odd numbers recommended)
    pub const fn set_filter_sizes(&mut self, harmonic: usize, percussive: usize) {
        self.median_filter_harmonic = harmonic;
        self.median_filter_percussive = percussive;
    }

    /// Set mask softness parameter.
    ///
    /// # Arguments
    /// * `softness` - Softness value (0.0 = hard mask, 1.0 = completely soft)
    pub fn set_mask_softness(&mut self, softness: F) {
        self.mask_softness = softness.clamp(F::zero(), F::one());
    }

    /// Validate HPSS configuration.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Result indicating whether the configuration is valid
    pub fn validate(&self, sample_rate: F) -> AudioSampleResult<()> {
        // Validate window size
        if self.win_size == 0 || !self.win_size.is_power_of_two() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "win_size",
                "Window size must be a positive power of 2",
            )));
        }

        // Validate hop size
        if self.hop_size == 0 || self.hop_size > self.win_size {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "hop_size",
                "Hop size must be positive and not larger than window size",
            )));
        }

        // Validate median filter sizes
        if self.median_filter_harmonic == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "median_filter_harmonic",
                "Harmonic median filter size must be greater than 0",
            )));
        }

        if self.median_filter_percussive == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "median_filter_percussive",
                "Percussive median filter size must be greater than 0",
            )));
        }

        // Validate mask softness
        if self.mask_softness < F::zero() || self.mask_softness > F::one() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "mask_softness",
                "Mask softness must be between 0.0 and 1.0",
            )));
        }

        // Check reasonable parameter ranges
        if self.win_size > 16384 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "win_size",
                "Window size should not exceed 16384 samples for practical processing",
            )));
        }

        if self.median_filter_harmonic > 101 || self.median_filter_percussive > 101 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "median_filter_size",
                "Median filter sizes should not exceed 101 for practical processing",
            )));
        }

        // Validate minimum frequency resolution
        let freq_resolution = sample_rate / to_precision::<F, _>(self.win_size);
        if freq_resolution > to_precision(50.0) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "win_size",
                format!("Window too small, frequency resolution ({:.1} Hz) is too low", freq_resolution),
            )));
        }

        Ok(())
    }

    /// Calculate the number of frequency bins for this configuration.
    pub const fn num_freq_bins(&self) -> usize {
        self.win_size / 2 + 1
    }

    /// Calculate the frequency resolution in Hz.
    pub fn freq_resolution(&self, sample_rate: F) -> F {
        sample_rate / to_precision::<F, _>(self.win_size)
    }

    /// Calculate the time resolution in seconds.
    pub fn time_resolution(&self, sample_rate: F) -> F {
        to_precision::<F, _>(self.hop_size) / sample_rate
    }
}

impl<F: RealFloat> Default for HpssConfig<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Method for computing chromagram features.
///
/// Chromagram can be computed using different spectral representations:
/// - STFT: Standard Short-Time Fourier Transform based approach
/// - CQT: Constant-Q Transform based approach (better frequency resolution for low frequencies)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaMethod {
    /// Use Short-Time Fourier Transform for chromagram computation
    Stft,
    /// Use Constant-Q Transform for chromagram computation
    Cqt,
}

impl Default for ChromaMethod {
    fn default() -> Self {
        ChromaMethod::Stft
    }
}

/// Configuration for chromagram (chroma vector) computation.
///
/// Chromagram represents the distribution of energy across pitch classes,
/// providing a harmonic representation that is invariant to octave shifts.
/// Each bin represents a semitone in the 12-tone equal temperament scale.
#[derive(Debug, Clone, PartialEq)]
pub struct ChromaConfig<F: RealFloat> {
    /// Method for computing the underlying spectral representation
    pub method: ChromaMethod,
    /// Number of chroma bins (typically 12 for Western music)
    pub n_chroma: usize,
    /// STFT window size in samples (used when method = Stft)
    /// Larger windows provide better frequency resolution
    pub window_size: usize,
    /// STFT hop size in samples (used when method = Stft)
    /// Smaller hop sizes provide better time resolution
    pub hop_size: usize,
    /// Sample rate override (if None, uses audio's sample rate)
    pub sample_rate: Option<usize>,
    /// Whether to normalize chroma vectors per time frame
    /// When true, each time frame is normalized to sum to 1.0
    pub norm: bool,
    /// Phantom data to bind the float type parameter
    _phantom: PhantomData<F>,
}

impl<F: RealFloat> ChromaConfig<F> {
    /// Create a new chroma configuration with default settings.
    ///
    /// Default configuration suitable for general chromagram analysis:
    /// - STFT method
    /// - 12 chroma bins (standard semitones)
    /// - 2048 sample window (good frequency resolution)
    /// - 512 sample hop size (good time resolution)
    /// - Frame normalization enabled
    pub fn new() -> Self {
        Self {
            method: ChromaMethod::Stft,
            n_chroma: 12,
            window_size: 2048,
            hop_size: 512,
            sample_rate: None,
            norm: true,
            _phantom: PhantomData,
        }
    }

    /// Create configuration optimized for STFT-based chromagram.
    ///
    /// Uses STFT with parameters optimized for harmonic content analysis.
    pub fn stft() -> Self {
        Self {
            method: ChromaMethod::Stft,
            n_chroma: 12,
            window_size: 2048,
            hop_size: 512,
            sample_rate: None,
            norm: true,
            _phantom: PhantomData,
        }
    }

    /// Create configuration optimized for CQT-based chromagram.
    ///
    /// Uses Constant-Q Transform which provides better frequency resolution
    /// for lower frequencies, making it ideal for harmonic analysis.
    pub fn cqt() -> Self {
        Self {
            method: ChromaMethod::Cqt,
            n_chroma: 12,
            window_size: 2048,  // Used for CQT kernel calculation
            hop_size: 512,
            sample_rate: None,
            norm: true,
            _phantom: PhantomData,
        }
    }

    /// Create configuration for high-resolution chromagram analysis.
    ///
    /// Uses larger windows for better frequency resolution.
    pub fn high_resolution() -> Self {
        Self {
            method: ChromaMethod::Stft,
            n_chroma: 12,
            window_size: 4096,
            hop_size: 1024,
            sample_rate: None,
            norm: true,
            _phantom: PhantomData,
        }
    }

    /// Create configuration for real-time chromagram analysis.
    ///
    /// Uses smaller windows for lower latency.
    pub fn realtime() -> Self {
        Self {
            method: ChromaMethod::Stft,
            n_chroma: 12,
            window_size: 1024,
            hop_size: 256,
            sample_rate: None,
            norm: true,
            _phantom: PhantomData,
        }
    }

    /// Set the number of chroma bins.
    pub fn with_n_chroma(mut self, n_chroma: usize) -> Self {
        self.n_chroma = n_chroma;
        self
    }

    /// Set the window size for STFT.
    pub fn with_window_size(mut self, window_size: usize) -> Self {
        self.window_size = window_size;
        self
    }

    /// Set the hop size.
    pub fn with_hop_size(mut self, hop_size: usize) -> Self {
        self.hop_size = hop_size;
        self
    }

    /// Set the sample rate override.
    pub fn with_sample_rate(mut self, sample_rate: usize) -> Self {
        self.sample_rate = Some(sample_rate);
        self
    }

    /// Set the normalization option.
    pub fn with_norm(mut self, norm: bool) -> Self {
        self.norm = norm;
        self
    }

    /// Calculate the number of time frames for given audio length.
    pub fn num_frames(&self, num_samples: usize) -> usize {
        if num_samples <= self.window_size {
            1
        } else {
            (num_samples - self.window_size) / self.hop_size + 1
        }
    }

    /// Calculate the time resolution in seconds.
    pub fn time_resolution(&self, sample_rate: F) -> F {
        to_precision::<F, _>(self.hop_size) / sample_rate
    }
}

impl<F: RealFloat> Default for ChromaConfig<F> {
    fn default() -> Self {
        Self::new()
    }
}
