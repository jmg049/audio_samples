//! Common batch operations for audio processing.
//!
//! This module provides implementations of common audio processing operations
//! that can be used in batch processing pipelines.

use super::traits::{BatchOperation, ValidatingBatchOperation};
use crate::operations::types::NormalizationMethod;
use crate::{
    AudioProcessing, AudioSample, AudioSampleResult, AudioSamples, AudioStatistics,
    AudioTypeConversion, CastFrom, CastInto, ConvertTo, I24,
};

/// Batch operation for normalizing audio samples.
///
/// This operation applies normalization to audio samples using various methods
/// such as peak normalization, RMS normalization, or custom range normalization.
#[derive(Debug, Clone)]
pub struct BatchNormalize<T: AudioSample>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
    T: CastInto<i16>
        + CastInto<I24>
        + CastInto<i32>
        + CastInto<f32>
        + CastInto<f64>
        + CastFrom<i16>
        + CastFrom<I24>
        + CastFrom<i32>
        + CastFrom<f32>
        + CastFrom<f64>,
{
    method: NormalizationMethod,
    min_value: T,
    max_value: T,
}

impl<T: AudioSample> BatchNormalize<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
    T: CastInto<i16>
        + CastInto<I24>
        + CastInto<i32>
        + CastInto<f32>
        + CastInto<f64>
        + CastFrom<i16>
        + CastFrom<I24>
        + CastFrom<i32>
        + CastFrom<f32>
        + CastFrom<f64>,
{
    /// Create a new peak normalization operation.
    ///
    /// Peak normalization scales the audio so that the peak amplitude
    /// reaches the specified target value.
    ///
    /// # Arguments
    /// * `target_peak` - Target peak amplitude
    ///
    /// # Returns
    /// A new BatchNormalize operation
    pub fn peak(target_peak: T) -> Self
    where
        T: Default,
    {
        Self {
            method: NormalizationMethod::Peak,
            min_value: T::default(),
            max_value: target_peak,
        }
    }

    /// Create a new RMS normalization operation.
    ///
    /// RMS normalization scales the audio so that the RMS level
    /// reaches the specified target value.
    ///
    /// # Arguments
    /// * `target_rms` - Target RMS level
    ///
    /// # Returns
    /// A new BatchNormalize operation
    pub fn rms(target_rms: T) -> Self
    where
        T: Default,
    {
        Self {
            method: NormalizationMethod::ZScore,
            min_value: T::default(),
            max_value: target_rms,
        }
    }

    /// Create a new range normalization operation.
    ///
    /// Range normalization scales the audio to fit within the specified
    /// minimum and maximum values.
    ///
    /// # Arguments
    /// * `min_value` - Minimum value of the output range
    /// * `max_value` - Maximum value of the output range
    ///
    /// # Returns
    /// A new BatchNormalize operation
    pub fn range(min_value: T, max_value: T) -> Self {
        Self {
            method: NormalizationMethod::MinMax,
            min_value,
            max_value,
        }
    }

    /// Create a new normalization operation with custom method.
    ///
    /// # Arguments
    /// * `method` - Normalization method to use
    /// * `min_value` - Minimum value of the output range
    /// * `max_value` - Maximum value of the output range
    ///
    /// # Returns
    /// A new BatchNormalize operation
    pub fn new(method: NormalizationMethod, min_value: T, max_value: T) -> Self {
        Self {
            method,
            min_value,
            max_value,
        }
    }
}

impl<T: AudioSample> BatchOperation<T> for BatchNormalize<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
    T: CastInto<i16>
        + CastInto<I24>
        + CastInto<i32>
        + CastInto<f32>
        + CastInto<f64>
        + CastFrom<i16>
        + CastFrom<I24>
        + CastFrom<i32>
        + CastFrom<f32>
        + CastFrom<f64>,
{
    fn apply_to_item(&self, item: &mut AudioSamples<T>) -> AudioSampleResult<()> {
        item.normalize(self.min_value, self.max_value, self.method)
    }

    fn description(&self) -> &str {
        match self.method {
            NormalizationMethod::Peak => "Peak normalization",
            NormalizationMethod::ZScore => "Z-Score normalization",
            NormalizationMethod::MinMax => "Range normalization",
            NormalizationMethod::Mean => "Mean normalization",
            NormalizationMethod::Median => "Median normalization",
        }
    }

    fn cost_estimate(&self) -> f64 {
        2.0 // Moderate cost due to statistics computation
    }
}

impl<T: AudioSample> ValidatingBatchOperation<T> for BatchNormalize<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
    T: CastInto<i16>
        + CastInto<I24>
        + CastInto<i32>
        + CastInto<f32>
        + CastInto<f64>
        + CastFrom<i16>
        + CastFrom<I24>
        + CastFrom<i32>
        + CastFrom<f32>
        + CastFrom<f64>,
{
    fn validate(&self, item: &AudioSamples<T>) -> AudioSampleResult<()> {
        if item.samples_per_channel() == 0 {
            return Err(crate::AudioSampleError::InvalidParameter(
                "Cannot normalize empty audio".to_string(),
            ));
        }
        Ok(())
    }
}

/// Batch operation for scaling audio samples by a constant factor.
///
/// This operation multiplies all audio samples by a specified scaling factor,
/// effectively changing the amplitude/volume of the audio.
#[derive(Debug, Clone)]
pub struct BatchScale<T: AudioSample> {
    factor: T,
}

impl<T: AudioSample> BatchScale<T> {
    /// Create a new scaling operation.
    ///
    /// # Arguments
    /// * `factor` - Scaling factor to apply
    ///
    /// # Returns
    /// A new BatchScale operation
    pub fn new(factor: T) -> Self {
        Self { factor }
    }

    /// Create a scaling operation that halves the amplitude.
    ///
    /// # Returns
    /// A new BatchScale operation with factor 0.5
    pub fn half() -> AudioSampleResult<Self> {
        Ok(Self {
            factor: T::cast_from(0.5f64),
        })
    }

    /// Create a scaling operation that doubles the amplitude.
    ///
    /// # Returns
    /// A new BatchScale operation with factor 2.0
    pub fn double() -> AudioSampleResult<Self> {
        Ok(Self {
            factor: T::cast_from(2.0f64),
        })
    }

    /// Create a scaling operation for a specific dB change.
    ///
    /// # Arguments
    /// * `db_change` - Change in decibels (positive = louder, negative = quieter)
    ///
    /// # Returns
    /// A new BatchScale operation
    pub fn from_db(db_change: f64) -> AudioSampleResult<Self> {
        let factor = 10.0_f64.powf(db_change / 20.0);
        Ok(Self {
            factor: T::cast_from(factor),
        })
    }
}

impl<T: AudioSample> BatchOperation<T> for BatchScale<T> {
    fn apply_to_item(&self, item: &mut AudioSamples<T>) -> AudioSampleResult<()> {
        item.apply(|sample| sample * self.factor)
    }

    fn description(&self) -> &str {
        "Amplitude scaling"
    }

    fn cost_estimate(&self) -> f64 {
        0.5 // Low cost operation
    }
}

/// Batch operation for applying filters to audio samples.
///
/// This operation applies various types of filters (low-pass, high-pass, band-pass)
/// to audio samples using the built-in filtering capabilities.
#[derive(Debug, Clone)]
pub struct BatchFilter {
    filter_type: FilterType,
    cutoff_low: f64,
    cutoff_high: Option<f64>,
}

#[derive(Debug, Clone)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
}

impl BatchFilter {
    /// Create a low-pass filter operation.
    ///
    /// # Arguments
    /// * `cutoff_hz` - Cutoff frequency in Hz
    ///
    /// # Returns
    /// A new BatchFilter operation
    pub fn lowpass(cutoff_hz: f64) -> Self {
        Self {
            filter_type: FilterType::LowPass,
            cutoff_low: cutoff_hz,
            cutoff_high: None,
        }
    }

    /// Create a high-pass filter operation.
    ///
    /// # Arguments
    /// * `cutoff_hz` - Cutoff frequency in Hz
    ///
    /// # Returns
    /// A new BatchFilter operation
    pub fn highpass(cutoff_hz: f64) -> Self {
        Self {
            filter_type: FilterType::HighPass,
            cutoff_low: cutoff_hz,
            cutoff_high: None,
        }
    }

    /// Create a band-pass filter operation.
    ///
    /// # Arguments
    /// * `low_hz` - Lower cutoff frequency in Hz
    /// * `high_hz` - Upper cutoff frequency in Hz
    ///
    /// # Returns
    /// A new BatchFilter operation
    pub fn bandpass(low_hz: f64, high_hz: f64) -> Self {
        Self {
            filter_type: FilterType::BandPass,
            cutoff_low: low_hz,
            cutoff_high: Some(high_hz),
        }
    }
}

impl<T: AudioSample> BatchOperation<T> for BatchFilter
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
    T: CastInto<i16>
        + CastInto<I24>
        + CastInto<i32>
        + CastInto<f32>
        + CastInto<f64>
        + CastFrom<i16>
        + CastFrom<I24>
        + CastFrom<i32>
        + CastFrom<f32>
        + CastFrom<f64>,
{
    fn apply_to_item(&self, item: &mut AudioSamples<T>) -> AudioSampleResult<()> {
        use crate::operations::AudioProcessing;
        match self.filter_type {
            FilterType::LowPass => item.low_pass_filter(self.cutoff_low),
            FilterType::HighPass => item.high_pass_filter(self.cutoff_low),
            FilterType::BandPass => {
                let high_cutoff = self.cutoff_high.unwrap_or(self.cutoff_low * 2.0);
                item.band_pass_filter(self.cutoff_low, high_cutoff)
            }
        }
    }

    fn description(&self) -> &str {
        match self.filter_type {
            FilterType::LowPass => "Low-pass filter",
            FilterType::HighPass => "High-pass filter",
            FilterType::BandPass => "Band-pass filter",
        }
    }

    fn cost_estimate(&self) -> f64 {
        5.0 // High cost due to filtering computation
    }
}

impl<T: AudioSample> ValidatingBatchOperation<T> for BatchFilter
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: ConvertTo<f64>,
    T: ConvertTo<f32>,
    T: ConvertTo<i16>,
    T: ConvertTo<I24>,
    T: ConvertTo<i32>,
    AudioSamples<T>: AudioTypeConversion<T>,
    AudioSamples<f32>: AudioTypeConversion<T>,
    T: CastInto<i16>
        + CastInto<I24>
        + CastInto<i32>
        + CastInto<f32>
        + CastInto<f64>
        + CastFrom<i16>
        + CastFrom<I24>
        + CastFrom<i32>
        + CastFrom<f32>
        + CastFrom<f64>,
{
    fn validate(&self, item: &AudioSamples<T>) -> AudioSampleResult<()> {
        let nyquist = item.sample_rate() as f64 / 2.0;

        if self.cutoff_low >= nyquist {
            return Err(crate::AudioSampleError::InvalidParameter(format!(
                "Cutoff frequency {} Hz exceeds Nyquist frequency {} Hz",
                self.cutoff_low, nyquist
            )));
        }

        if let Some(high_cutoff) = self.cutoff_high {
            if high_cutoff >= nyquist {
                return Err(crate::AudioSampleError::InvalidParameter(format!(
                    "High cutoff frequency {} Hz exceeds Nyquist frequency {} Hz",
                    high_cutoff, nyquist
                )));
            }

            if high_cutoff <= self.cutoff_low {
                return Err(crate::AudioSampleError::InvalidParameter(
                    "High cutoff frequency must be greater than low cutoff frequency".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// Batch operation for type conversion.
///
/// This operation converts audio samples from one type to another,
/// handling the conversion safely and preserving audio quality.
#[derive(Debug)]
pub struct BatchConvert<From: AudioSample, To: AudioSample> {
    _phantom: std::marker::PhantomData<(From, To)>,
}

impl<From: AudioSample, To: AudioSample> BatchConvert<From, To> {
    /// Create a new type conversion operation.
    ///
    /// # Returns
    /// A new BatchConvert operation
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<From: AudioSample, To: AudioSample> Default for BatchConvert<From, To> {
    fn default() -> Self {
        Self::new()
    }
}

impl<From: AudioSample, To: AudioSample> BatchOperation<From> for BatchConvert<From, To>
where
    From: ConvertTo<To>,
{
    fn apply_to_item(&self, item: &mut AudioSamples<From>) -> AudioSampleResult<()> {
        // Note: This is a conceptual implementation. In practice, we'd need to
        // modify the trait to support type transformation or use a different approach.
        // For now, this serves as a placeholder for the API design.
        Ok(())
    }

    fn description(&self) -> &str {
        "Type conversion"
    }

    fn cost_estimate(&self) -> f64 {
        1.0 // Moderate cost due to type conversion
    }
}

/// Batch operation for removing DC offset.
///
/// This operation removes any DC bias from audio samples by subtracting
/// the mean value from all samples.
#[derive(Debug, Clone)]
pub struct BatchRemoveDC;

impl BatchRemoveDC {
    /// Create a new DC removal operation.
    ///
    /// # Returns
    /// A new BatchRemoveDC operation
    pub fn new() -> Self {
        Self
    }
}

impl Default for BatchRemoveDC {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: AudioSample> BatchOperation<T> for BatchRemoveDC
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
    AudioSamples<f32>: AudioTypeConversion<T>,
    T: CastInto<i16>
        + CastInto<I24>
        + CastInto<i32>
        + CastInto<f32>
        + CastInto<f64>
        + CastFrom<i16>
        + CastFrom<I24>
        + CastFrom<i32>
        + CastFrom<f32>
        + CastFrom<f64>,
{
    fn apply_to_item(&self, item: &mut AudioSamples<T>) -> AudioSampleResult<()> {
        use crate::operations::AudioProcessing;
        item.remove_dc_offset()
    }

    fn description(&self) -> &str {
        "DC offset removal"
    }

    fn cost_estimate(&self) -> f64 {
        1.0 // Moderate cost due to mean calculation
    }
}

/// Batch operation for applying fade-in/fade-out effects.
///
/// This operation applies fade effects to audio samples using various curve types.
#[derive(Debug, Clone)]
pub struct BatchFade {
    fade_type: FadeType,
    duration_seconds: f64,
    curve: FadeCurve,
}

#[derive(Debug, Clone)]
pub enum FadeType {
    In,
    Out,
}

#[derive(Debug, Clone)]
pub enum FadeCurve {
    Linear,
    Exponential,
    Logarithmic,
}

impl BatchFade {
    /// Create a fade-in operation.
    ///
    /// # Arguments
    /// * `duration_seconds` - Duration of the fade in seconds
    /// * `curve` - Fade curve type
    ///
    /// # Returns
    /// A new BatchFade operation
    pub fn fade_in(duration_seconds: f64, curve: FadeCurve) -> Self {
        Self {
            fade_type: FadeType::In,
            duration_seconds,
            curve,
        }
    }

    /// Create a fade-out operation.
    ///
    /// # Arguments
    /// * `duration_seconds` - Duration of the fade in seconds
    /// * `curve` - Fade curve type
    ///
    /// # Returns
    /// A new BatchFade operation
    pub fn fade_out(duration_seconds: f64, curve: FadeCurve) -> Self {
        Self {
            fade_type: FadeType::Out,
            duration_seconds,
            curve,
        }
    }
}

impl<T: AudioSample> BatchOperation<T> for BatchFade
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: CastInto<i16>
        + CastInto<I24>
        + CastInto<i32>
        + CastInto<f32>
        + CastInto<f64>
        + CastFrom<i16>
        + CastFrom<I24>
        + CastFrom<i32>
        + CastFrom<f32>
        + CastFrom<f64>,
{
    fn apply_to_item(&self, item: &mut AudioSamples<T>) -> AudioSampleResult<()> {
        use crate::operations::AudioEditing;
        match self.fade_type {
            FadeType::In => item.fade_in(self.duration_seconds, self.curve.clone().into()),
            FadeType::Out => item.fade_out(self.duration_seconds, self.curve.clone().into()),
        }
    }

    fn description(&self) -> &str {
        match self.fade_type {
            FadeType::In => "Fade-in",
            FadeType::Out => "Fade-out",
        }
    }

    fn cost_estimate(&self) -> f64 {
        1.5 // Moderate cost due to envelope calculation
    }
}

impl<T: AudioSample> ValidatingBatchOperation<T> for BatchFade
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: CastInto<i16>
        + CastInto<I24>
        + CastInto<i32>
        + CastInto<f32>
        + CastInto<f64>
        + CastFrom<i16>
        + CastFrom<I24>
        + CastFrom<i32>
        + CastFrom<f32>
        + CastFrom<f64>,
{
    fn validate(&self, item: &AudioSamples<T>) -> AudioSampleResult<()> {
        let audio_duration = item.duration_seconds();

        if self.duration_seconds > audio_duration {
            return Err(crate::AudioSampleError::InvalidParameter(format!(
                "Fade duration {} seconds exceeds audio duration {} seconds",
                self.duration_seconds, audio_duration
            )));
        }

        if self.duration_seconds <= 0.0 {
            return Err(crate::AudioSampleError::InvalidParameter(
                "Fade duration must be positive".to_string(),
            ));
        }

        Ok(())
    }
}

// Convert our FadeCurve to the operations FadeCurve
impl From<FadeCurve> for crate::operations::types::FadeCurve {
    fn from(curve: FadeCurve) -> Self {
        match curve {
            FadeCurve::Linear => crate::operations::types::FadeCurve::Linear,
            FadeCurve::Exponential => crate::operations::types::FadeCurve::Exponential,
            FadeCurve::Logarithmic => crate::operations::types::FadeCurve::Logarithmic,
        }
    }
}

/// Batch operation for statistical analysis.
///
/// This operation computes various statistics for audio samples without
/// modifying the audio data itself.
#[derive(Debug, Clone)]
pub struct BatchStatistics {
    compute_rms: bool,
    compute_peak: bool,
    compute_zero_crossings: bool,
    compute_spectral_centroid: bool,
}

impl BatchStatistics {
    /// Create a new statistics operation with all statistics enabled.
    ///
    /// # Returns
    /// A new BatchStatistics operation
    pub fn all() -> Self {
        Self {
            compute_rms: true,
            compute_peak: true,
            compute_zero_crossings: true,
            compute_spectral_centroid: true,
        }
    }

    /// Create a new statistics operation with only basic statistics.
    ///
    /// # Returns
    /// A new BatchStatistics operation
    pub fn basic() -> Self {
        Self {
            compute_rms: true,
            compute_peak: true,
            compute_zero_crossings: false,
            compute_spectral_centroid: false,
        }
    }

    /// Create a new statistics operation with custom configuration.
    ///
    /// # Arguments
    /// * `rms` - Whether to compute RMS
    /// * `peak` - Whether to compute peak
    /// * `zero_crossings` - Whether to compute zero crossings
    /// * `spectral_centroid` - Whether to compute spectral centroid
    ///
    /// # Returns
    /// A new BatchStatistics operation
    pub fn custom(rms: bool, peak: bool, zero_crossings: bool, spectral_centroid: bool) -> Self {
        Self {
            compute_rms: rms,
            compute_peak: peak,
            compute_zero_crossings: zero_crossings,
            compute_spectral_centroid: spectral_centroid,
        }
    }
}

impl<T: AudioSample> BatchOperation<T> for BatchStatistics
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: CastInto<i16>
        + CastInto<I24>
        + CastInto<i32>
        + CastInto<f32>
        + CastInto<f64>
        + CastFrom<i16>
        + CastFrom<I24>
        + CastFrom<i32>
        + CastFrom<f32>
        + CastFrom<f64>,
{
    fn apply_to_item(&self, item: &mut AudioSamples<T>) -> AudioSampleResult<()> {
        // This is a read-only operation, so we don't actually modify the audio
        // In a real implementation, you might want to store the statistics somewhere
        // or use a different trait for analysis operations

        if self.compute_rms {
            let _rms = item.rms();
        }

        if self.compute_peak {
            let _peak = item.peak();
        }

        if self.compute_zero_crossings {
            let _zero_crossings = item.zero_crossings();
        }

        if self.compute_spectral_centroid {
            let _centroid = item.spectral_centroid();
        }

        Ok(())
    }

    fn description(&self) -> &str {
        "Statistical analysis"
    }

    fn cost_estimate(&self) -> f64 {
        let mut cost: f64 = 0.0;

        if self.compute_rms {
            cost += 1.0;
        }

        if self.compute_peak {
            cost += 0.5;
        }

        if self.compute_zero_crossings {
            cost += 1.0;
        }

        if self.compute_spectral_centroid {
            cost += 3.0; // Expensive due to FFT
        }

        cost.max(0.1) // Minimum cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioSamples;
    use approx_eq::assert_approx_eq;
    use ndarray::array;

    #[test]
    fn test_batch_scale() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut audio = AudioSamples::new_mono(data, 44100);

        let operation = BatchScale::new(2.0);
        operation.apply_to_item(&mut audio).unwrap();

        let expected = array![2.0f32, 4.0, 6.0, 8.0, 10.0];
        assert_eq!(audio.as_mono().unwrap(), &expected);
    }

    #[test]
    fn test_batch_scale_from_db() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut audio = AudioSamples::new_mono(data, 44100);

        let operation = BatchScale::from_db(6.0).expect("Batch scale op failed with db change 6.0"); // +6dB ≈ 2x amplitude
        operation.apply_to_item(&mut audio).unwrap();

        // Check that the first sample is approximately doubled
        let result = audio.as_mono().unwrap();
        assert_approx_eq!(result[0] as f64, 2.0, 0.01);
    }

    #[test]
    fn test_batch_scale_convenience_methods() {
        let data = array![2.0f32, 4.0, 6.0];
        let mut audio = AudioSamples::new_mono(data, 44100);

        let operation = BatchScale::half().expect("Batch scale op failed with half factor");
        operation.apply_to_item(&mut audio).unwrap();

        let expected = array![1.0f32, 2.0, 3.0];
        assert_eq!(audio.as_mono().unwrap(), &expected);
    }

    #[test]
    fn test_batch_remove_dc() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0]; // Has DC offset
        let mut audio = AudioSamples::new_mono(data, 44100);

        let operation = BatchRemoveDC::new();
        operation.apply_to_item(&mut audio).unwrap();

        // After DC removal, the mean should be approximately zero
        let result = audio.as_mono().unwrap();
        let mean = result.iter().sum::<f32>() / result.len() as f32;
        assert_approx_eq!(mean as f64, 0.0, 1e-6);
    }

    #[test]
    fn test_batch_filter_validation() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioSamples::new_mono(data, 44100);

        let operation = BatchFilter::lowpass(50000.0); // Above Nyquist
        let result = operation.validate(&audio);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_fade_validation() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioSamples::new_mono(data, 44100);

        let operation = BatchFade::fade_in(10.0, FadeCurve::Linear); // Longer than audio
        let result = operation.validate(&audio);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_statistics() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut audio = AudioSamples::new_mono(data, 44100);

        let operation = BatchStatistics::basic();
        operation.apply_to_item(&mut audio).unwrap();

        // Statistics operation shouldn't modify the audio
        let expected = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(audio.as_mono().unwrap(), &expected);
    }

    #[test]
    fn test_operation_properties() {
        let scale_op: BatchScale<f64> = BatchScale::new(2.0);
        assert_eq!(scale_op.description(), "Amplitude scaling");
        assert_eq!(scale_op.cost_estimate(), 0.5);

        let filter_op: BatchFilter = BatchFilter::lowpass(1000.0);
        let description = <BatchFilter as BatchOperation<f64>>::description(&filter_op);
        assert_eq!(description, "Low-pass filter");
        let cost_estimate = <BatchFilter as BatchOperation<f64>>::cost_estimate(&filter_op);

        assert_approx_eq!(cost_estimate, 5.0);

        let stats_op = BatchStatistics::all();
        let description = <BatchStatistics as BatchOperation<f64>>::description(&stats_op);
        assert_eq!(description, "Statistical analysis");

        let cost_estimate = <BatchStatistics as BatchOperation<f64>>::cost_estimate(&stats_op);
        assert!(cost_estimate > 5.0, "Expected 5.0 got {}", cost_estimate); // Should be expensive due to spectral centroid
    }
}
