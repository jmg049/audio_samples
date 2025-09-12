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
pub struct IirFilterDesign {
    /// Type of IIR filter (Butterworth, Chebyshev, etc.)
    pub filter_type: IirFilterType,
    /// Response type (low-pass, high-pass, etc.)
    pub response: FilterResponse,
    /// Filter order (number of poles)
    pub order: usize,
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

impl IirFilterDesign {
    /// Create a simple Butterworth low-pass filter design.
    pub const fn butterworth_lowpass(order: usize, cutoff_frequency: f64) -> Self {
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
    pub const fn butterworth_highpass(order: usize, cutoff_frequency: f64) -> Self {
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
    pub const fn butterworth_bandpass(
        order: usize,
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
    pub const fn chebyshev_i(
        response: FilterResponse,
        order: usize,
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

impl EqBand {
    /// Create a new peak/notch EQ band.
    ///
    /// # Arguments
    /// * `frequency` - Center frequency in Hz
    /// * `gain_db` - Gain in dB (positive for boost, negative for cut)
    /// * `q_factor` - Quality factor (bandwidth control)
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
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if this EQ band is enabled.
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Validate the EQ band parameters.
    pub fn validate(&self, sample_rate: f64) -> Result<(), String> {
        let nyquist = sample_rate / 2.0;

        if self.frequency <= 0.0 || self.frequency >= nyquist {
            return Err(format!(
                "Frequency {} Hz is out of range (0, {})",
                self.frequency, nyquist
            ));
        }

        if self.q_factor <= 0.0 {
            return Err("Q factor must be greater than 0".to_string());
        }

        // Check reasonable gain limits
        if self.gain_db.abs() > 40.0 {
            return Err("Gain should be within ±40 dB for practical use".to_string());
        }

        Ok(())
    }
}

/// Parametric equalizer configuration.
///
/// A complete parametric EQ consisting of multiple bands that can be
/// applied to audio signals for frequency shaping.
#[derive(Debug, Clone, PartialEq)]
pub struct ParametricEq {
    /// Vector of EQ bands
    pub bands: Vec<EqBand>,
    /// Overall output gain in dB
    pub output_gain_db: f64,
    /// Whether the EQ is bypassed
    pub bypassed: bool,
}

impl ParametricEq {
    /// Create a new empty parametric EQ.
    pub const fn new() -> Self {
        Self {
            bands: Vec::new(),
            output_gain_db: 0.0,
            bypassed: false,
        }
    }

    /// Add an EQ band to the parametric EQ.
    pub fn add_band(&mut self, band: EqBand) {
        self.bands.push(band);
    }

    /// Remove an EQ band by index.
    pub fn remove_band(&mut self, index: usize) -> Option<EqBand> {
        if index < self.bands.len() {
            Some(self.bands.remove(index))
        } else {
            None
        }
    }

    /// Get a reference to an EQ band by index.
    pub fn get_band(&self, index: usize) -> Option<&EqBand> {
        self.bands.get(index)
    }

    /// Get a mutable reference to an EQ band by index.
    pub fn get_band_mut(&mut self, index: usize) -> Option<&mut EqBand> {
        self.bands.get_mut(index)
    }

    /// Get the number of bands in the EQ.
    pub fn band_count(&self) -> usize {
        self.bands.len()
    }

    /// Set the overall output gain.
    pub fn set_output_gain(&mut self, gain_db: f64) {
        self.output_gain_db = gain_db;
    }

    /// Enable or disable the EQ (bypass).
    pub fn set_bypassed(&mut self, bypassed: bool) {
        self.bypassed = bypassed;
    }

    /// Check if the EQ is bypassed.
    pub const fn is_bypassed(&self) -> bool {
        self.bypassed
    }

    /// Validate all EQ bands.
    pub fn validate(&self, sample_rate: f64) -> Result<(), String> {
        for (i, band) in self.bands.iter().enumerate() {
            band.validate(sample_rate)
                .map_err(|e| format!("Band {}: {}", i, e))?;
        }
        Ok(())
    }

    /// Create a common 3-band EQ (low shelf, mid peak, high shelf).
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

impl Default for ParametricEq {
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
    /// 0.0 = internal only, 1.0 = external only
    pub external_mix: f64,
}

impl SideChainConfig {
    /// Create a new disabled side-chain configuration.
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
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable side-chain processing.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Set high-pass filter frequency for side-chain signal.
    pub fn set_high_pass(&mut self, freq: f64) {
        self.high_pass_freq = Some(freq);
    }

    /// Set low-pass filter frequency for side-chain signal.
    pub fn set_low_pass(&mut self, freq: f64) {
        self.low_pass_freq = Some(freq);
    }

    /// Validate side-chain configuration.
    pub fn validate(&self, sample_rate: f64) -> Result<(), String> {
        if let Some(hp_freq) = self.high_pass_freq {
            if hp_freq <= 0.0 || hp_freq >= sample_rate / 2.0 {
                return Err(
                    "High-pass frequency must be between 0 and Nyquist frequency".to_string(),
                );
            }
        }

        if let Some(lp_freq) = self.low_pass_freq {
            if lp_freq <= 0.0 || lp_freq >= sample_rate / 2.0 {
                return Err(
                    "Low-pass frequency must be between 0 and Nyquist frequency".to_string()
                );
            }
        }

        if let (Some(hp), Some(lp)) = (self.high_pass_freq, self.low_pass_freq) {
            if hp >= lp {
                return Err("High-pass frequency must be less than low-pass frequency".to_string());
            }
        }

        if self.external_mix < 0.0 || self.external_mix > 1.0 {
            return Err("External mix must be between 0.0 and 1.0".to_string());
        }

        Ok(())
    }
}

/// Compressor configuration parameters.
///
/// Controls how the compressor responds to signal levels above the threshold.
#[derive(Debug, Clone, PartialEq)]
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

impl CompressorConfig {
    /// Create a new compressor configuration with default settings.
    pub const fn new() -> Self {
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

    /// Create a vocal compressor preset.
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
    pub fn validate(&self, sample_rate: f64) -> Result<(), String> {
        if self.threshold_db > 0.0 {
            return Err("Threshold should be negative (below 0 dB)".to_string());
        }

        if self.ratio < 1.0 {
            return Err("Ratio must be >= 1.0".to_string());
        }

        if self.attack_ms < 0.01 || self.attack_ms > 1000.0 {
            return Err("Attack time must be between 0.01 and 1000 ms".to_string());
        }

        if self.release_ms < 1.0 || self.release_ms > 10000.0 {
            return Err("Release time must be between 1.0 and 10000 ms".to_string());
        }

        if self.makeup_gain_db.abs() > 40.0 {
            return Err("Makeup gain should be within ±40 dB".to_string());
        }

        if self.knee_width_db < 0.0 || self.knee_width_db > 20.0 {
            return Err("Knee width must be between 0.0 and 20.0 dB".to_string());
        }

        if self.lookahead_ms < 0.0 || self.lookahead_ms > 20.0 {
            return Err("Lookahead time must be between 0.0 and 20.0 ms".to_string());
        }

        self.side_chain.validate(sample_rate)?;

        Ok(())
    }
}

impl Default for CompressorConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Limiter configuration parameters.
///
/// Controls how the limiter prevents signal levels from exceeding the ceiling.
#[derive(Debug, Clone, PartialEq)]
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

impl LimiterConfig {
    /// Create a new limiter configuration with default settings.
    pub const fn new() -> Self {
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

    /// Create a transparent limiter preset.
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
    pub fn validate(&self, sample_rate: f64) -> Result<(), String> {
        if self.ceiling_db > 0.0 {
            return Err("Ceiling should be negative (below 0 dB)".to_string());
        }

        if self.attack_ms < 0.001 || self.attack_ms > 100.0 {
            return Err("Attack time must be between 0.001 and 100 ms".to_string());
        }

        if self.release_ms < 1.0 || self.release_ms > 10000.0 {
            return Err("Release time must be between 1.0 and 10000 ms".to_string());
        }

        if self.knee_width_db < 0.0 || self.knee_width_db > 10.0 {
            return Err("Knee width must be between 0.0 and 10.0 dB".to_string());
        }

        if self.lookahead_ms < 0.0 || self.lookahead_ms > 20.0 {
            return Err("Lookahead time must be between 0.0 and 20.0 ms".to_string());
        }

        self.side_chain.validate(sample_rate)?;

        Ok(())
    }
}

impl Default for LimiterConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for Constant-Q Transform (CQT) analysis.
///
/// The CQT provides logarithmic frequency spacing that aligns with musical
/// intervals, making it ideal for music analysis and harmonic detection.
#[derive(Debug, Clone, PartialEq)]
pub struct CqtConfig {
    /// Number of frequency bins per octave (typically 12-24 for musical analysis)
    /// Higher values provide better frequency resolution but increase computation
    pub bins_per_octave: usize,
    /// Minimum frequency in Hz (typically 55 Hz for A1 or 27.5 Hz for A0)
    pub fmin: f64,
    /// Maximum frequency in Hz (None = Nyquist frequency)
    /// Should be less than or equal to sample_rate / 2
    pub fmax: Option<f64>,
    /// Quality factor controlling frequency resolution vs time resolution
    /// Higher Q = better frequency resolution, lower Q = better time resolution
    pub q_factor: f64,
    /// Window function applied to each frequency bin
    pub window_type: WindowType,
    /// Sparsity threshold for kernel optimization (0.0 to 1.0)
    /// Smaller values = more sparse kernels = faster computation
    pub sparsity_threshold: f64,
    /// Whether to normalize the CQT output
    pub normalize: bool,
}

impl CqtConfig {
    /// Create a new CQT configuration with default settings.
    ///
    /// Default configuration suitable for general musical analysis:
    /// - 12 bins per octave (chromatic scale)
    /// - 55 Hz minimum frequency (A1)
    /// - Quality factor of 1.0
    /// - Hanning window
    /// - 0.01 sparsity threshold
    pub const fn new() -> Self {
        Self {
            bins_per_octave: 12,
            fmin: 55.0, // A1
            fmax: None, // Will be set to Nyquist frequency
            q_factor: 1.0,
            window_type: WindowType::Hanning,
            sparsity_threshold: 0.01,
            normalize: true,
        }
    }

    /// Create a CQT configuration optimized for musical analysis.
    ///
    /// Uses 12 bins per octave for chromatic scale analysis,
    /// starting from C1 (32.7 Hz) for full piano range coverage.
    pub const fn musical() -> Self {
        Self {
            bins_per_octave: 12,
            fmin: 32.7, // C1
            fmax: None,
            q_factor: 1.0,
            window_type: WindowType::Hanning,
            sparsity_threshold: 0.01,
            normalize: true,
        }
    }

    /// Create a CQT configuration optimized for harmonic analysis.
    ///
    /// Uses 24 bins per octave for quarter-tone resolution,
    /// providing detailed harmonic analysis capabilities.
    pub const fn harmonic() -> Self {
        Self {
            bins_per_octave: 24,
            fmin: 55.0, // A1
            fmax: None,
            q_factor: 1.0,
            window_type: WindowType::Hanning,
            sparsity_threshold: 0.005, // Lower threshold for better precision
            normalize: true,
        }
    }

    /// Create a CQT configuration optimized for chord detection.
    ///
    /// Uses settings that balance frequency resolution with computational
    /// efficiency for real-time chord detection applications.
    pub const fn chord_detection() -> Self {
        Self {
            bins_per_octave: 12,
            fmin: 82.4,         // E2 (lowest guitar string)
            fmax: Some(2093.0), // C7 (high piano range)
            q_factor: 0.8,      // Slightly lower Q for faster response
            window_type: WindowType::Hanning,
            sparsity_threshold: 0.02,
            normalize: true,
        }
    }

    /// Create a CQT configuration optimized for onset detection.
    ///
    /// Uses lower Q factor for better time resolution,
    /// suitable for detecting note onsets and transients.
    pub const fn onset_detection() -> Self {
        Self {
            bins_per_octave: 12,
            fmin: 55.0, // A1
            fmax: None,
            q_factor: 0.5, // Lower Q for better time resolution
            window_type: WindowType::Hanning,
            sparsity_threshold: 0.02,
            normalize: true,
        }
    }

    /// Set the frequency range for analysis.
    ///
    /// # Arguments
    /// * `fmin` - Minimum frequency in Hz
    /// * `fmax` - Maximum frequency in Hz (None for Nyquist)
    pub fn set_frequency_range(&mut self, fmin: f64, fmax: Option<f64>) {
        self.fmin = fmin;
        self.fmax = fmax;
    }

    /// Set the number of bins per octave.
    ///
    /// # Arguments
    /// * `bins_per_octave` - Number of frequency bins per octave
    pub fn set_bins_per_octave(&mut self, bins_per_octave: usize) {
        self.bins_per_octave = bins_per_octave;
    }

    /// Set the quality factor.
    ///
    /// # Arguments
    /// * `q_factor` - Quality factor (higher = better frequency resolution)
    pub fn set_q_factor(&mut self, q_factor: f64) {
        self.q_factor = q_factor;
    }

    /// Calculate the actual maximum frequency based on sample rate.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// The effective maximum frequency (either fmax or Nyquist frequency)
    pub fn effective_fmax(&self, sample_rate: f64) -> f64 {
        self.fmax.unwrap_or(sample_rate / 2.0)
    }

    /// Calculate the total number of CQT bins.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Number of frequency bins in the CQT
    pub fn num_bins(&self, sample_rate: f64) -> usize {
        let fmax = self.effective_fmax(sample_rate);
        let octaves = (fmax / self.fmin).log2();
        (octaves * self.bins_per_octave as f64).ceil() as usize
    }

    /// Calculate the center frequency for a given bin index.
    ///
    /// # Arguments
    /// * `bin_index` - Zero-based bin index
    ///
    /// # Returns
    /// Center frequency in Hz for the specified bin
    pub fn bin_frequency(&self, bin_index: usize) -> f64 {
        self.fmin * 2.0_f64.powf(bin_index as f64 / self.bins_per_octave as f64)
    }

    /// Calculate the bandwidth for a given bin index.
    ///
    /// # Arguments
    /// * `bin_index` - Zero-based bin index
    ///
    /// # Returns
    /// Bandwidth in Hz for the specified bin
    pub fn bin_bandwidth(&self, bin_index: usize) -> f64 {
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
    pub fn validate(&self, sample_rate: f64) -> Result<(), String> {
        if self.bins_per_octave == 0 {
            return Err("Bins per octave must be greater than 0".to_string());
        }

        if self.fmin <= 0.0 {
            return Err("Minimum frequency must be greater than 0".to_string());
        }

        let nyquist = sample_rate / 2.0;
        if self.fmin >= nyquist {
            return Err("Minimum frequency must be less than Nyquist frequency".to_string());
        }

        if let Some(fmax) = self.fmax {
            if fmax <= self.fmin {
                return Err("Maximum frequency must be greater than minimum frequency".to_string());
            }
            if fmax > nyquist {
                return Err("Maximum frequency cannot exceed Nyquist frequency".to_string());
            }
        }

        if self.q_factor <= 0.0 {
            return Err("Quality factor must be greater than 0".to_string());
        }

        if self.sparsity_threshold < 0.0 || self.sparsity_threshold > 1.0 {
            return Err("Sparsity threshold must be between 0.0 and 1.0".to_string());
        }

        // Check that we have at least one bin
        if self.num_bins(sample_rate) == 0 {
            return Err("Configuration results in zero CQT bins".to_string());
        }

        Ok(())
    }
}

impl Default for CqtConfig {
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

impl AdaptiveThresholdConfig {
    /// Create a new adaptive threshold configuration with default settings.
    ///
    /// Default configuration suitable for general onset detection:
    /// - Delta method with 0.05 delta value
    /// - Window size of 1024 samples (about 23ms at 44.1kHz)
    /// - Reasonable min/max threshold bounds
    pub const fn new() -> Self {
        Self {
            method: AdaptiveThresholdMethod::Delta,
            delta: 0.05,
            percentile: 0.9,
            window_size: 1024,
            min_threshold: 0.01,
            max_threshold: 1.0,
        }
    }

    /// Create a delta-based adaptive threshold configuration.
    ///
    /// # Arguments
    /// * `delta` - Delta value for threshold computation
    /// * `window_size` - Size of local window in samples
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
    pub fn set_min_threshold(&mut self, min_threshold: f64) {
        self.min_threshold = min_threshold;
    }

    /// Set the maximum threshold value.
    pub fn set_max_threshold(&mut self, max_threshold: f64) {
        self.max_threshold = max_threshold;
    }

    /// Validate the adaptive threshold configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.delta < 0.0 {
            return Err("Delta must be non-negative".to_string());
        }

        if self.percentile < 0.0 || self.percentile > 1.0 {
            return Err("Percentile must be between 0.0 and 1.0".to_string());
        }

        if self.window_size == 0 {
            return Err("Window size must be greater than 0".to_string());
        }

        if self.min_threshold < 0.0 {
            return Err("Minimum threshold must be non-negative".to_string());
        }

        if self.max_threshold <= self.min_threshold {
            return Err("Maximum threshold must be greater than minimum threshold".to_string());
        }

        Ok(())
    }
}

impl Default for AdaptiveThresholdConfig {
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
pub struct PeakPickingConfig {
    /// Adaptive threshold configuration
    pub adaptive_threshold: AdaptiveThresholdConfig,
    /// Minimum time separation between peaks (in samples)
    /// Prevents detecting multiple peaks for the same onset event
    pub min_peak_separation: usize,
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
    pub median_filter_length: usize,
    /// Normalize onset strength before peak picking
    /// Ensures consistent detection across different signal levels
    pub normalize_onset_strength: bool,
    /// Normalization method for onset strength
    pub normalization_method: NormalizationMethod,
}

impl PeakPickingConfig {
    /// Create a new peak picking configuration with default settings.
    ///
    /// Default configuration optimized for general onset detection:
    /// - Adaptive delta thresholding
    /// - 512 samples minimum separation (about 11ms at 44.1kHz)
    /// - Pre-emphasis enabled with moderate coefficient
    /// - Median filtering enabled with small kernel
    /// - Peak normalization enabled
    pub const fn new() -> Self {
        Self {
            adaptive_threshold: AdaptiveThresholdConfig::new(),
            min_peak_separation: 512,
            pre_emphasis: true,
            pre_emphasis_coeff: 0.97,
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
    pub const fn music() -> Self {
        Self {
            adaptive_threshold: AdaptiveThresholdConfig::combined(0.03, 0.85, 2048),
            min_peak_separation: 1024,
            pre_emphasis: true,
            pre_emphasis_coeff: 0.95,
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
    pub const fn speech() -> Self {
        Self {
            adaptive_threshold: AdaptiveThresholdConfig::delta(0.07, 1024),
            min_peak_separation: 256,
            pre_emphasis: true,
            pre_emphasis_coeff: 0.98,
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
    pub const fn drums() -> Self {
        Self {
            adaptive_threshold: AdaptiveThresholdConfig::percentile(0.95, 512),
            min_peak_separation: 128,
            pre_emphasis: true,
            pre_emphasis_coeff: 0.93,
            median_filter: false,
            median_filter_length: 3,
            normalize_onset_strength: true,
            normalization_method: NormalizationMethod::Peak,
        }
    }

    /// Set the minimum peak separation in samples.
    pub fn set_min_peak_separation(&mut self, samples: usize) {
        self.min_peak_separation = samples;
    }

    /// Set the minimum peak separation in milliseconds.
    pub fn set_min_peak_separation_ms(&mut self, ms: f64, sample_rate: f64) {
        self.min_peak_separation = (ms * sample_rate / 1000.0) as usize;
    }

    /// Enable or disable pre-emphasis.
    pub fn set_pre_emphasis(&mut self, enabled: bool, coeff: f64) {
        self.pre_emphasis = enabled;
        self.pre_emphasis_coeff = coeff;
    }

    /// Enable or disable median filtering.
    pub fn set_median_filter(&mut self, enabled: bool, length: usize) {
        self.median_filter = enabled;
        self.median_filter_length = length;
    }

    /// Validate the peak picking configuration.
    pub fn validate(&self) -> Result<(), String> {
        self.adaptive_threshold.validate()?;

        if self.min_peak_separation == 0 {
            return Err("Minimum peak separation must be greater than 0".to_string());
        }

        if self.pre_emphasis_coeff < 0.0 || self.pre_emphasis_coeff > 1.0 {
            return Err("Pre-emphasis coefficient must be between 0.0 and 1.0".to_string());
        }

        if self.median_filter_length == 0 || self.median_filter_length % 2 == 0 {
            return Err("Median filter length must be odd and greater than 0".to_string());
        }

        Ok(())
    }
}

impl Default for PeakPickingConfig {
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
pub struct OnsetConfig {
    /// CQT configuration for spectral analysis
    /// Uses CqtConfig::onset_detection() by default for optimal time resolution
    pub cqt_config: CqtConfig,
    /// Hop size for frame-based analysis in samples
    /// Smaller values provide better time resolution but increase computation
    pub hop_size: usize,
    /// Window size for CQT analysis in samples (None = auto-calculate)
    /// If None, calculated as 4 periods of the minimum frequency
    pub window_size: Option<usize>,
    /// Threshold for onset detection (0.0 to 1.0)
    /// Higher values = fewer, more confident onsets
    /// Lower values = more onsets, potentially including false positives
    pub threshold: f64,
    /// Minimum time between onsets in seconds
    /// Prevents detection of spurious onsets too close together
    pub min_onset_interval: f64,
    /// Pre-emphasis factor for spectral flux (0.0 to 1.0)
    /// Emphasizes high-frequency content which often carries onset information
    pub pre_emphasis: f64,
    /// Whether to use adaptive thresholding
    /// Adapts threshold based on local energy characteristics
    pub adaptive_threshold: bool,
    /// Median filter length for adaptive thresholding (in frames)
    /// Used to smooth the threshold over time
    pub median_filter_length: usize,
    /// Multiplier for adaptive threshold
    /// threshold = median_value * adaptive_threshold_multiplier
    pub adaptive_threshold_multiplier: f64,
    /// Peak picking configuration for onset detection
    pub peak_picking: PeakPickingConfig,
}

impl OnsetConfig {
    /// Create a new onset detection configuration with default settings.
    ///
    /// Default configuration suitable for general onset detection:
    /// - CQT optimized for onset detection (Q=0.5)
    /// - 512 sample hop size (good time resolution)
    /// - Auto-calculated window size
    /// - Moderate threshold (0.3)
    /// - 50ms minimum onset interval
    /// - Adaptive thresholding enabled
    pub const fn new() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 512,
            window_size: None,
            threshold: 0.3,
            min_onset_interval: 0.05, // 50ms
            pre_emphasis: 0.0,
            adaptive_threshold: true,
            median_filter_length: 5,
            adaptive_threshold_multiplier: 1.5,
            peak_picking: PeakPickingConfig::new(),
        }
    }

    /// Create configuration optimized for percussive onset detection.
    ///
    /// Optimized for detecting drum hits and other percussive events:
    /// - Higher threshold for cleaner detection
    /// - Shorter minimum interval for rapid percussion
    /// - Pre-emphasis to highlight transients
    pub const fn percussive() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 256, // Higher time resolution for drums
            window_size: None,
            threshold: 0.5,           // Higher threshold for cleaner detection
            min_onset_interval: 0.03, // 30ms for rapid percussion
            pre_emphasis: 0.3,        // Emphasize high frequencies
            adaptive_threshold: true,
            median_filter_length: 3,
            adaptive_threshold_multiplier: 2.0,
            peak_picking: PeakPickingConfig::drums(),
        }
    }

    /// Create configuration optimized for musical onset detection.
    ///
    /// Optimized for detecting note onsets in musical instruments:
    /// - Moderate threshold for good sensitivity
    /// - Longer minimum interval for typical musical phrasing
    /// - Less pre-emphasis for tonal content
    pub const fn musical() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 512,
            window_size: None,
            threshold: 0.25,         // Lower threshold for musical content
            min_onset_interval: 0.1, // 100ms for musical phrasing
            pre_emphasis: 0.1,       // Light pre-emphasis
            adaptive_threshold: true,
            median_filter_length: 7,
            adaptive_threshold_multiplier: 1.2,
            peak_picking: PeakPickingConfig::music(),
        }
    }

    /// Create configuration optimized for speech onset detection.
    ///
    /// Optimized for detecting word/syllable onsets in speech:
    /// - Low threshold for speech dynamics
    /// - Moderate minimum interval for speech rate
    /// - Minimal pre-emphasis for speech clarity
    pub const fn speech() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 256, // Good time resolution for speech
            window_size: None,
            threshold: 0.2,           // Low threshold for speech dynamics
            min_onset_interval: 0.08, // 80ms for speech rate
            pre_emphasis: 0.05,       // Minimal pre-emphasis
            adaptive_threshold: true,
            median_filter_length: 9,
            adaptive_threshold_multiplier: 1.1,
            peak_picking: PeakPickingConfig::speech(),
        }
    }

    /// Set the hop size for frame-based analysis.
    ///
    /// # Arguments
    /// * `hop_size` - Hop size in samples (must be > 0)
    pub fn set_hop_size(&mut self, hop_size: usize) {
        self.hop_size = hop_size;
    }

    /// Set the onset detection threshold.
    ///
    /// # Arguments
    /// * `threshold` - Threshold value (0.0 to 1.0)
    pub fn set_threshold(&mut self, threshold: f64) {
        self.threshold = threshold.clamp(0.0, 1.0);
    }

    /// Set the minimum time between onsets.
    ///
    /// # Arguments
    /// * `interval_seconds` - Minimum interval in seconds (must be > 0)
    pub fn set_min_onset_interval(&mut self, interval_seconds: f64) {
        self.min_onset_interval = interval_seconds.max(0.001); // At least 1ms
    }

    /// Enable or disable adaptive thresholding.
    ///
    /// # Arguments
    /// * `enabled` - Whether to use adaptive thresholding
    pub fn set_adaptive_threshold(&mut self, enabled: bool) {
        self.adaptive_threshold = enabled;
    }

    /// Set the pre-emphasis factor for spectral flux.
    ///
    /// # Arguments
    /// * `pre_emphasis` - Pre-emphasis factor (0.0 to 1.0)
    pub fn set_pre_emphasis(&mut self, pre_emphasis: f64) {
        self.pre_emphasis = pre_emphasis.clamp(0.0, 1.0);
    }

    /// Calculate the effective window size for CQT analysis.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Window size in samples
    pub fn effective_window_size(&self, sample_rate: f64) -> usize {
        self.window_size.unwrap_or_else(|| {
            // Auto-calculate based on lowest frequency (4 periods for good resolution)
            let min_period = sample_rate / self.cqt_config.fmin;
            (min_period * 4.0) as usize
        })
    }

    /// Calculate the time resolution of onset detection.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Time resolution in seconds
    pub fn time_resolution(&self, sample_rate: f64) -> f64 {
        self.hop_size as f64 / sample_rate
    }

    /// Convert onset time from frames to seconds.
    ///
    /// # Arguments
    /// * `frame_index` - Frame index from onset detection
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Time in seconds
    pub fn frame_to_seconds(&self, frame_index: usize, sample_rate: f64) -> f64 {
        (frame_index * self.hop_size) as f64 / sample_rate
    }

    /// Convert onset time from seconds to frames.
    ///
    /// # Arguments
    /// * `time_seconds` - Time in seconds
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Frame index
    pub fn seconds_to_frame(&self, time_seconds: f64, sample_rate: f64) -> usize {
        ((time_seconds * sample_rate) / self.hop_size as f64).round() as usize
    }

    /// Validate the onset detection configuration.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Result indicating whether the configuration is valid
    pub fn validate(&self, sample_rate: f64) -> Result<(), String> {
        // Validate CQT configuration
        self.cqt_config.validate(sample_rate)?;

        // Validate hop size
        if self.hop_size == 0 {
            return Err("Hop size must be greater than 0".to_string());
        }

        // Validate threshold
        if self.threshold < 0.0 || self.threshold > 1.0 {
            return Err("Threshold must be between 0.0 and 1.0".to_string());
        }

        // Validate minimum onset interval
        if self.min_onset_interval <= 0.0 {
            return Err("Minimum onset interval must be greater than 0".to_string());
        }

        // Validate pre-emphasis
        if self.pre_emphasis < 0.0 || self.pre_emphasis > 1.0 {
            return Err("Pre-emphasis must be between 0.0 and 1.0".to_string());
        }

        // Validate window size if specified
        if let Some(window_size) = self.window_size {
            if window_size == 0 {
                return Err("Window size must be greater than 0".to_string());
            }
        }

        // Validate adaptive threshold parameters
        if self.adaptive_threshold {
            if self.median_filter_length == 0 {
                return Err("Median filter length must be greater than 0".to_string());
            }
            if self.adaptive_threshold_multiplier <= 0.0 {
                return Err("Adaptive threshold multiplier must be greater than 0".to_string());
            }
        }

        // Check that time resolution is reasonable
        let time_resolution = self.time_resolution(sample_rate);
        if time_resolution > 0.1 {
            return Err(format!(
                "Time resolution ({:.3}s) is too low. Consider reducing hop size.",
                time_resolution
            ));
        }

        Ok(())
    }
}

impl Default for OnsetConfig {
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
    /// ∆E[n] = Σ(max(0, |X[k,n]|² - |X[k,n-1]|²)) for all frequency bins k
    /// Good for general onset detection, especially percussive events
    Energy,
    /// Magnitude-based flux: sum of positive magnitude differences
    /// ∆M[n] = Σ(max(0, |X[k,n]| - |X[k,n-1]|)) for all frequency bins k
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
pub struct SpectralFluxConfig {
    /// CQT configuration for spectral analysis
    pub cqt_config: CqtConfig,
    /// Hop size for frame-based analysis in samples
    pub hop_size: usize,
    /// Window size for CQT analysis in samples (None = auto-calculate)
    pub window_size: Option<usize>,
    /// Spectral flux method to use
    pub flux_method: SpectralFluxMethod,
    /// Peak picking configuration for onset detection
    pub peak_picking: PeakPickingConfig,
    /// Apply rectification to spectral flux (keep only positive values)
    pub rectify: bool,
    /// Logarithmic compression factor for spectral flux
    /// flux_compressed = log(1 + C * flux) where C is this parameter
    pub log_compression: f64,
}

impl SpectralFluxConfig {
    /// Create a new spectral flux configuration with default settings.
    pub const fn new() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 512,
            window_size: None,
            flux_method: SpectralFluxMethod::Energy,
            peak_picking: PeakPickingConfig::new(),
            rectify: true,
            log_compression: 100.0,
        }
    }

    /// Create configuration optimized for percussive onset detection.
    pub const fn percussive() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 256,
            window_size: None,
            flux_method: SpectralFluxMethod::Energy,
            peak_picking: PeakPickingConfig::drums(),
            rectify: true,
            log_compression: 1000.0,
        }
    }

    /// Create configuration optimized for musical onset detection.
    pub const fn musical() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 512,
            window_size: None,
            flux_method: SpectralFluxMethod::Magnitude,
            peak_picking: PeakPickingConfig::music(),
            rectify: true,
            log_compression: 100.0,
        }
    }

    /// Create configuration optimized for complex domain onset detection.
    pub const fn complex() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 512,
            window_size: None,
            flux_method: SpectralFluxMethod::Complex,
            peak_picking: PeakPickingConfig::new(),
            rectify: false,
            log_compression: 100.0,
        }
    }

    /// Validate the spectral flux configuration.
    pub fn validate(&self, sample_rate: f64) -> Result<(), String> {
        self.cqt_config.validate(sample_rate)?;
        self.peak_picking.validate()?;

        if self.hop_size == 0 {
            return Err("Hop size must be greater than 0".to_string());
        }

        if let Some(window_size) = self.window_size {
            if window_size == 0 {
                return Err("Window size must be greater than 0".to_string());
            }
        }

        if self.log_compression < 0.0 {
            return Err("Log compression factor must be non-negative".to_string());
        }

        Ok(())
    }
}

impl Default for SpectralFluxConfig {
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
pub struct ComplexOnsetConfig {
    /// CQT configuration for spectral analysis
    pub cqt_config: CqtConfig,
    /// Hop size for frame-based analysis in samples
    pub hop_size: usize,
    /// Window size for CQT analysis in samples (None = auto-calculate)
    pub window_size: Option<usize>,
    /// Peak picking configuration for onset detection
    pub peak_picking: PeakPickingConfig,
    /// Weight for magnitude-based detection (0.0-1.0)
    pub magnitude_weight: f64,
    /// Weight for phase-based detection (0.0-1.0)
    pub phase_weight: f64,
    /// Apply magnitude rectification (keep only positive changes)
    pub magnitude_rectify: bool,
    /// Apply phase rectification (keep only positive phase deviations)
    pub phase_rectify: bool,
    /// Logarithmic compression factor for combined onset function
    pub log_compression: f64,
}

impl ComplexOnsetConfig {
    /// Create a new complex onset configuration with default settings.
    pub const fn new() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 512,
            window_size: None,
            peak_picking: PeakPickingConfig::new(),
            magnitude_weight: 0.7,
            phase_weight: 0.3,
            magnitude_rectify: true,
            phase_rectify: true,
            log_compression: 100.0,
        }
    }

    /// Create configuration optimized for percussive onset detection.
    pub const fn percussive() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 256,
            window_size: None,
            peak_picking: PeakPickingConfig::drums(),
            magnitude_weight: 0.8,
            phase_weight: 0.2,
            magnitude_rectify: true,
            phase_rectify: true,
            log_compression: 1000.0,
        }
    }

    /// Create configuration optimized for musical onset detection.
    pub const fn musical() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 512,
            window_size: None,
            peak_picking: PeakPickingConfig::music(),
            magnitude_weight: 0.6,
            phase_weight: 0.4,
            magnitude_rectify: true,
            phase_rectify: true,
            log_compression: 100.0,
        }
    }

    /// Create configuration optimized for speech onset detection.
    pub const fn speech() -> Self {
        Self {
            cqt_config: CqtConfig::onset_detection(),
            hop_size: 256,
            window_size: None,
            peak_picking: PeakPickingConfig::speech(),
            magnitude_weight: 0.5,
            phase_weight: 0.5,
            magnitude_rectify: true,
            phase_rectify: false,
            log_compression: 50.0,
        }
    }

    /// Set the magnitude and phase weights.
    pub fn set_weights(&mut self, magnitude_weight: f64, phase_weight: f64) {
        self.magnitude_weight = magnitude_weight.clamp(0.0, 1.0);
        self.phase_weight = phase_weight.clamp(0.0, 1.0);
    }

    /// Validate the complex onset configuration.
    pub fn validate(&self, sample_rate: f64) -> Result<(), String> {
        self.cqt_config.validate(sample_rate)?;
        self.peak_picking.validate()?;

        if self.hop_size == 0 {
            return Err("Hop size must be greater than 0".to_string());
        }

        if let Some(window_size) = self.window_size {
            if window_size == 0 {
                return Err("Window size must be greater than 0".to_string());
            }
        }

        if self.magnitude_weight < 0.0 || self.magnitude_weight > 1.0 {
            return Err("Magnitude weight must be between 0.0 and 1.0".to_string());
        }

        if self.phase_weight < 0.0 || self.phase_weight > 1.0 {
            return Err("Phase weight must be between 0.0 and 1.0".to_string());
        }

        // Both weights cannot be zero
        if self.magnitude_weight == 0.0 && self.phase_weight == 0.0 {
            return Err("Both magnitude and phase weights cannot be zero".to_string());
        }

        if self.log_compression < 0.0 {
            return Err("Log compression factor must be non-negative".to_string());
        }

        Ok(())
    }
}

impl Default for ComplexOnsetConfig {
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
        target_snr_db: f64,
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
        min_gain_db: f64,
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
        cutoff_hz: f64,
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
        semitones: f64,
        preserve_formants: bool,
    },
}

impl PerturbationMethod {
    /// Create a Gaussian noise perturbation configuration.
    ///
    /// # Arguments
    /// * `target_snr_db` - Target signal-to-noise ratio in dB
    /// * `noise_color` - Color/spectrum of the noise
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
    pub const fn high_pass_filter_with_slope(cutoff_hz: f64, slope_db_per_octave: f64) -> Self {
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
    pub fn validate(&self, sample_rate: f64) -> Result<(), String> {
        match self {
            Self::GaussianNoise { target_snr_db, .. } => {
                if *target_snr_db < -60.0 || *target_snr_db > 60.0 {
                    return Err("Target SNR should be between -60 and 60 dB".to_string());
                }
            }
            Self::RandomGain { min_gain_db, max_gain_db } => {
                if min_gain_db >= max_gain_db {
                    return Err("Minimum gain must be less than maximum gain".to_string());
                }
                if min_gain_db < &-40.0 || *max_gain_db > 20.0 {
                    return Err("Gain range should be reasonable (-40 to +20 dB)".to_string());
                }
            }
            Self::HighPassFilter { cutoff_hz, slope_db_per_octave } => {
                let nyquist = sample_rate / 2.0;
                if *cutoff_hz <= 0.0 || *cutoff_hz >= nyquist {
                    return Err(format!(
                        "Cutoff frequency must be between 0 and {} Hz",
                        nyquist
                    ));
                }
                if let Some(slope) = slope_db_per_octave {
                    if *slope < 0.0 || *slope > 48.0 {
                        return Err("Filter slope should be between 0 and 48 dB/octave".to_string());
                    }
                }
            }
            Self::PitchShift { semitones, .. } => {
                if semitones.abs() > 12.0 {
                    return Err("Pitch shift should be within ±12 semitones".to_string());
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
pub struct PerturbationConfig {
    /// The perturbation method to apply
    pub method: PerturbationMethod,
    /// Optional random seed for deterministic perturbation
    /// If None, uses thread-local random number generator
    pub seed: Option<u64>,
}

impl PerturbationConfig {
    /// Create a new perturbation configuration.
    ///
    /// # Arguments
    /// * `method` - The perturbation method to apply
    pub const fn new(method: PerturbationMethod) -> Self {
        Self {
            method,
            seed: None,
        }
    }

    /// Create a new perturbation configuration with a specific seed.
    ///
    /// # Arguments
    /// * `method` - The perturbation method to apply
    /// * `seed` - Random seed for deterministic results
    pub const fn with_seed(method: PerturbationMethod, seed: u64) -> Self {
        Self {
            method,
            seed: Some(seed),
        }
    }

    /// Set the random seed for deterministic perturbation.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = Some(seed);
    }

    /// Clear the random seed to use non-deterministic perturbation.
    pub fn clear_seed(&mut self) {
        self.seed = None;
    }

    /// Validate the perturbation configuration.
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// Result indicating whether the configuration is valid
    pub fn validate(&self, sample_rate: f64) -> Result<(), String> {
        self.method.validate(sample_rate)
    }
}

impl Default for PerturbationConfig {
    fn default() -> Self {
        Self::new(PerturbationMethod::GaussianNoise {
            target_snr_db: 20.0,
            noise_color: NoiseColor::White,
        })
    }
}
