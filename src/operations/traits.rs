//! Core trait definitions for audio processing operations.
//!
//! This module defines the focused traits that replace the monolithic
//! `AudioSamplesOperations` trait. Each trait has a single responsibility
//! and can be implemented independently.

use super::types::*;
use crate::{AudioSample, AudioSampleResult, AudioSamples, AudioTypeConversion, ConvertTo, I24};
use ndarray::Array2;
use std::collections::VecDeque;

// Complex numbers using num-complex crate
pub use num_complex::Complex;

/// Statistical analysis operations for audio data.
///
/// This trait provides methods to compute various statistical measures
/// of audio samples, useful for analysis, visualization, and processing decisions.
///
/// All methods return values in the native sample type `T` for consistency
/// with the underlying data representation.
pub trait AudioStatistics<T: AudioSample>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<T>: AudioTypeConversion<T>,
{
    /// Returns the peak (maximum absolute value) in the audio samples.
    ///
    /// This is useful for preventing clipping and measuring signal levels.
    fn peak(&self) -> T;

    /// Alias for peak to match common terminology
    fn amplitude(&self) -> T {
        self.peak()
    }

    /// Returns the minimum value in the audio samples.
    fn min_sample(&self) -> T;

    /// Returns the maximum value in the audio samples.
    fn max_sample(&self) -> T;

    /// Computes the mean (average) of the audio samples.
    fn mean(&self) -> T;

    /// Computes the Root Mean Square (RMS) of the audio samples.
    ///
    /// RMS is useful for measuring average signal power/energy and
    /// provides a perceptually relevant measure of loudness.
    fn rms(&self) -> f64;

    /// Computes the statistical variance of the audio samples.
    ///
    /// Variance measures the spread of sample values around the mean.
    fn variance(&self) -> AudioSampleResult<f64>;

    /// Computes the standard deviation of the audio samples.
    ///
    /// Standard deviation is the square root of variance and provides
    /// a measure of signal variability in the same units as the samples.
    fn std_dev(&self) -> AudioSampleResult<f64>;

    /// Counts the number of zero crossings in the audio signal.
    ///
    /// Zero crossings are useful for pitch detection and signal analysis.
    /// The count represents transitions from positive to negative values or vice versa.
    fn zero_crossings(&self) -> usize;

    /// Computes the zero crossing rate (crossings per second).
    ///
    /// This normalizes the zero crossing count by the signal duration,
    /// making it independent of audio length.
    fn zero_crossing_rate(&self) -> f64;

    /// Computes the autocorrelation function up to max_lag samples.
    ///
    /// Returns a vector of correlation values for each lag offset.
    /// Useful for pitch detection and periodicity analysis.
    ///
    /// # Arguments
    /// * `max_lag` - Maximum lag to compute (in samples)
    fn autocorrelation(&self, max_lag: usize) -> AudioSampleResult<Vec<f64>>;

    /// Computes cross-correlation with another audio signal.
    ///
    /// Returns correlation values for each lag offset between the two signals.
    /// Useful for alignment, synchronization, and similarity analysis.
    ///
    /// # Arguments
    /// * `other` - The other audio signal to correlate with
    /// * `max_lag` - Maximum lag to compute (in samples)
    fn cross_correlation(&self, other: &Self, max_lag: usize) -> AudioSampleResult<Vec<f64>>;

    /// Computes the spectral centroid (brightness measure).
    ///
    /// The spectral centroid represents the "center of mass" of the spectrum
    /// and is often used as a measure of brightness or timbre.
    /// Requires FFT computation internally.
    fn spectral_centroid(&self) -> AudioSampleResult<f64>;

    /// Computes spectral rolloff frequency.
    ///
    /// The rolloff frequency is the frequency below which a specified percentage
    /// of the total spectral energy is contained.
    ///
    /// # Arguments
    /// * `rolloff_percent` - Percentage of energy (0.0 to 1.0, typically 0.85)
    fn spectral_rolloff(&self, rolloff_percent: f64) -> AudioSampleResult<f64>;
}

/// Signal processing operations for audio manipulation.
///
/// This trait provides methods for common audio processing tasks including
/// normalization, filtering, compression, and envelope operations.
///
/// Most methods modify the audio in-place for efficiency and return
/// a Result to indicate success or failure.
pub trait AudioProcessing<T: AudioSample>
where
    Self: Sized,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<T>: AudioTypeConversion<T>,
{
    /// Normalizes audio samples using the specified method and range.
    ///
    /// # Arguments
    /// * `min` - Target minimum value
    /// * `max` - Target maximum value  
    /// * `method` - Normalization method to use
    ///
    /// # Errors
    /// Returns an error if min >= max or if the method cannot be applied.
    fn normalize(&mut self, min: T, max: T, method: NormalizationMethod) -> AudioSampleResult<()>;

    /// Scales all audio samples by a constant factor.
    ///
    /// This is equivalent to adjusting the volume/amplitude of the signal.
    ///
    /// # Arguments
    /// * `factor` - Scaling factor (1.0 = no change, 2.0 = double amplitude)
    fn scale(&mut self, factor: T);

    /// Applies a windowing function to the audio samples.
    ///
    /// The window length must match the number of samples in the audio.
    /// This is commonly used before FFT analysis to reduce spectral leakage.
    ///
    /// # Arguments
    /// * `window` - Window coefficients (must match audio length)
    fn apply_window(&mut self, window: &[T]) -> AudioSampleResult<()>;

    /// Applies a digital filter to the audio samples.
    ///
    /// Uses convolution with the provided filter coefficients.
    /// The filtered result will be shorter than the original by filter_length - 1 samples.
    ///
    /// # Arguments
    /// * `filter_coeffs` - FIR filter coefficients
    fn apply_filter(&mut self, filter_coeffs: &[T]) -> AudioSampleResult<()>;

    /// Applies μ-law compression to the audio samples.
    ///
    /// μ-law compression is commonly used in telecommunications.
    /// Standard μ-law uses μ = 255.
    ///
    /// # Arguments
    /// * `mu` - Compression parameter (typically 255 for standard μ-law)
    fn mu_compress(&mut self, mu: T) -> AudioSampleResult<()>;

    /// Applies μ-law expansion (decompression) to the audio samples.
    ///
    /// This reverses μ-law compression. The μ parameter should match
    /// the value used for compression.
    ///
    /// # Arguments
    /// * `mu` - Expansion parameter (should match compression parameter)
    fn mu_expand(&mut self, mu: T) -> AudioSampleResult<()>;

    /// Applies a low-pass filter with the specified cutoff frequency.
    ///
    /// Frequencies above the cutoff will be attenuated.
    ///
    /// # Arguments
    /// * `cutoff_hz` - Cutoff frequency in Hz
    fn low_pass_filter(&mut self, cutoff_hz: f64) -> AudioSampleResult<()>;

    /// Applies a high-pass filter with the specified cutoff frequency.
    ///
    /// Frequencies below the cutoff will be attenuated.
    ///
    /// # Arguments
    /// * `cutoff_hz` - Cutoff frequency in Hz
    fn high_pass_filter(&mut self, cutoff_hz: f64) -> AudioSampleResult<()>;

    /// Applies a band-pass filter between low and high frequencies.
    ///
    /// Only frequencies within the specified range will pass through.
    ///
    /// # Arguments
    /// * `low_hz` - Lower cutoff frequency in Hz
    /// * `high_hz` - Upper cutoff frequency in Hz
    fn band_pass_filter(&mut self, low_hz: f64, high_hz: f64) -> AudioSampleResult<()>;

    /// Removes DC offset by subtracting the mean value.
    ///
    /// This centers the audio around zero and removes any constant bias.
    fn remove_dc_offset(&mut self) -> AudioSampleResult<()>;

    /// Clips audio samples to the specified range.
    ///
    /// Any samples outside the range will be limited to the range boundaries.
    /// This is useful for preventing clipping in subsequent processing.
    ///
    /// # Arguments
    /// * `min_val` - Minimum allowed value
    /// * `max_val` - Maximum allowed value
    fn clip(&mut self, min_val: T, max_val: T) -> AudioSampleResult<()>;

    /// Resamples audio to a new sample rate using high-quality algorithms.
    ///
    /// This method provides a convenient interface to rubato's resampling capabilities
    /// with different quality/performance trade-offs.
    ///
    /// # Arguments
    /// * `target_sample_rate` - Desired output sample rate in Hz
    /// * `quality` - Quality/performance trade-off setting
    ///
    /// # Returns
    /// A new AudioSamples instance with the target sample rate
    ///
    /// # Errors
    /// Returns an error if:
    /// - The resampling feature is not enabled
    /// - The resampling parameters are invalid
    /// - The input audio is empty
    /// - Rubato encounters an internal error
    fn resample(
        &self,
        target_sample_rate: usize,
        quality: ResamplingQuality,
    ) -> AudioSampleResult<Self>;

    /// Resamples audio by a specific ratio.
    ///
    /// # Arguments
    /// * `ratio` - Resampling ratio (output_rate / input_rate)
    /// * `quality` - Quality/performance trade-off setting
    ///
    /// # Returns
    /// A new AudioSamples instance resampled by the given ratio
    ///
    /// # Errors
    /// Returns an error if:
    /// - The resampling feature is not enabled
    /// - The ratio is invalid (≤ 0)
    /// - The input audio is empty
    /// - Rubato encounters an internal error
    fn resample_by_ratio(&self, ratio: f64, quality: ResamplingQuality) -> AudioSampleResult<Self>;
}

/// Frequency domain analysis and spectral transformations.
///
/// This trait provides methods for FFT-based analysis and spectral processing.
/// Requires external FFT library dependencies.
///
/// Complex numbers are used for frequency domain representations,
/// and ndarray is used for efficient matrix operations on spectrograms.
pub trait AudioTransforms<T: AudioSample>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    /// Computes the Fast Fourier Transform of the audio samples.
    ///
    /// Returns complex frequency domain representation where the real and
    /// imaginary parts represent the magnitude and phase at each frequency bin.
    fn fft(&self) -> AudioSampleResult<Vec<Complex<f64>>>;

    /// Computes the inverse FFT from frequency domain back to time domain.
    ///
    /// # Arguments
    /// * `spectrum` - Complex frequency domain data
    fn ifft(&self, spectrum: &[Complex<f64>]) -> AudioSampleResult<Self>
    where
        Self: Sized;

    /// Computes the Short-Time Fourier Transform (STFT).
    ///
    /// Returns a 2D array where each column represents an FFT frame at a specific time.
    /// This provides both time and frequency information simultaneously.
    ///
    /// # Arguments
    /// * `window_size` - Size of each analysis window in samples
    /// * `hop_size` - Number of samples between successive windows
    /// * `window_type` - Window function to apply to each frame
    fn stft(
        &self,
        window_size: usize,
        hop_size: usize,
        window_type: WindowType,
    ) -> AudioSampleResult<Array2<Complex<f64>>>;

    /// Computes the inverse STFT to reconstruct time domain signal.
    ///
    /// Reconstructs a time-domain signal from its STFT representation.
    ///
    /// # Arguments
    /// * `stft_matrix` - STFT data (frequency bins × time frames)
    /// * `hop_size` - Hop size used in the original STFT
    /// * `window_type` - Window type used in the original STFT
    /// * `sample_rate` - Sample rate for the reconstructed signal
    fn istft(
        stft_matrix: &Array2<Complex<f64>>,
        hop_size: usize,
        window_type: WindowType,
        sample_rate: usize,
        center: bool,
    ) -> AudioSampleResult<Self>
    where
        Self: Sized;

    /// Computes the magnitude spectrogram (|STFT|^2) with scaling options.
    ///
    /// Returns the power spectrum over time, useful for visualization
    /// and analysis of spectral content.
    ///
    /// # Arguments
    /// * `window_size` - Size of each analysis window in samples
    /// * `hop_size` - Number of samples between successive windows
    /// * `window_type` - Window function to apply to each frame
    /// * `scale` - Scaling method to apply (Linear, Log, or Mel)
    /// * `normalize` - Whether to normalize the result
    fn spectrogram(
        &self,
        window_size: usize,
        hop_size: usize,
        window_type: WindowType,
        scale: SpectrogramScale,
        normalize: bool,
    ) -> AudioSampleResult<Array2<f64>>;

    /// Computes mel-scaled spectrogram for perceptual analysis.
    ///
    /// The mel scale better represents human auditory perception
    /// and is commonly used in speech and music analysis.
    ///
    /// # Arguments
    /// * `n_mels` - Number of mel frequency bands
    /// * `fmin` - Minimum frequency in Hz
    /// * `fmax` - Maximum frequency in Hz
    /// * `window_size` - Size of each analysis window in samples
    /// * `hop_size` - Number of samples between successive windows
    fn mel_spectrogram(
        &self,
        n_mels: usize,
        fmin: f64,
        fmax: f64,
        window_size: usize,
        hop_size: usize,
    ) -> AudioSampleResult<Array2<f64>>;

    /// Computes Mel-Frequency Cepstral Coefficients (MFCC).
    ///
    /// MFCCs are commonly used features in speech recognition and audio analysis.
    /// They provide a compact representation of spectral characteristics.
    ///
    /// # Arguments
    /// * `n_mfcc` - Number of MFCC coefficients to compute
    /// * `n_mels` - Number of mel frequency bands (intermediate step)
    /// * `fmin` - Minimum frequency in Hz
    /// * `fmax` - Maximum frequency in Hz
    fn mfcc(
        &self,
        n_mfcc: usize,
        n_mels: usize,
        fmin: f64,
        fmax: f64,
    ) -> AudioSampleResult<Array2<f64>>;

    /// Computes chromagram (pitch class profile).
    ///
    /// Useful for music analysis and chord detection by representing
    /// the energy in each of the 12 pitch classes (C, C#, D, etc.).
    ///
    /// # Arguments
    /// * `n_chroma` - Number of chroma bins (typically 12)
    fn chroma(&self, n_chroma: usize) -> AudioSampleResult<Array2<f64>>;

    /// Computes the power spectral density using Welch's method.
    ///
    /// Returns both the frequency bins and corresponding power values.
    /// Welch's method provides better noise reduction than a single FFT.
    ///
    /// # Arguments
    /// * `window_size` - Size of each segment for averaging
    /// * `overlap` - Overlap between segments (0.0 to 1.0)
    fn power_spectral_density(
        &self,
        window_size: usize,
        overlap: f64,
    ) -> AudioSampleResult<(Vec<f64>, Vec<f64>)>; // (frequencies, psd)

    /// Computes gammatone-filtered spectrogram for auditory modeling.
    ///
    /// Gammatone filters model the human auditory system's frequency selectivity.
    /// They use the ERB (Equivalent Rectangular Bandwidth) scale which approximates
    /// the filtering characteristics of the human cochlea.
    ///
    /// # Mathematical Foundation
    ///
    /// Gammatone filter impulse response:
    /// ```text
    /// g(t) = a * t^(n-1) * exp(-2πERB(f)t) * cos(2πft + φ)
    /// ```
    ///
    /// ERB scale:
    /// ```text
    /// ERB(f) = 24.7 * (4.37*f/1000 + 1)
    /// ```
    ///
    /// # Applications
    ///
    /// - **Auditory modeling**: Simulates cochlear filtering
    /// - **Psychoacoustic analysis**: Perceptually relevant frequency decomposition
    /// - **Hearing research**: Models human auditory processing
    /// - **Audio compression**: Perceptually motivated frequency analysis
    ///
    /// # Arguments
    /// * `n_filters` - Number of gammatone filters (typically 64-128)
    /// * `fmin` - Minimum frequency in Hz (typically 80-100 Hz)
    /// * `fmax` - Maximum frequency in Hz (typically sample_rate/2)
    /// * `window_size` - Size of analysis windows in samples
    /// * `hop_size` - Number of samples between successive windows
    fn gammatone_spectrogram(
        &self,
        n_filters: usize,
        fmin: f64,
        fmax: f64,
        window_size: usize,
        hop_size: usize,
    ) -> AudioSampleResult<Array2<f64>>;

    /// Computes the Constant-Q Transform (CQT) of the audio signal.
    ///
    /// The CQT provides logarithmic frequency resolution that aligns with musical
    /// intervals, making it ideal for music analysis and harmonic detection.
    /// Unlike FFT which has linear frequency spacing, CQT bins are spaced
    /// logarithmically with constant Q factor (Q = f_center / bandwidth).
    ///
    /// # Mathematical Foundation
    ///
    /// The CQT is computed by convolving the signal with a bank of complex
    /// exponential kernels at logarithmically spaced frequencies:
    ///
    /// ```text
    /// CQT[k] = Σ(n=0 to N-1) x[n] * W*[k,n]
    /// ```
    ///
    /// Where:
    /// - `W[k,n]` is the kernel for bin k at sample n
    /// - `f_k = fmin * 2^(k/bins_per_octave)` (logarithmic frequency spacing)
    /// - `Q = f_k / bandwidth_k` (constant across all bins)
    ///
    /// # Applications
    ///
    /// - **Music analysis**: Chord detection, harmonic analysis
    /// - **Pitch tracking**: Fundamental frequency estimation
    /// - **Audio classification**: Genre classification, instrument recognition
    /// - **Sound synthesis**: Spectral modeling and resynthesis
    ///
    /// # Arguments
    /// * `config` - CQT configuration parameters
    ///
    /// # Returns
    /// Complex-valued CQT coefficients as `Array2<Complex<f64>>` with dimensions
    /// `(num_bins, 1)` for single-frame analysis
    ///
    /// # Example
    /// ```rust,ignore
    /// use audio_samples::{AudioSamples, operations::*};
    /// use audio_samples::operations::types::CqtConfig;
    ///
    /// let config = CqtConfig::musical();
    /// let cqt = audio.constant_q_transform(&config)?;
    /// ```
    fn constant_q_transform(
        &self,
        config: &super::types::CqtConfig,
    ) -> AudioSampleResult<Array2<Complex<f64>>>;

    /// Computes the inverse Constant-Q Transform (iCQT) to reconstruct time-domain signal.
    ///
    /// Reconstructs a time-domain signal from its CQT representation using
    /// dual-frame reconstruction or pseudo-inverse methods.
    ///
    /// # Mathematical Foundation
    ///
    /// The iCQT attempts to reconstruct the original signal:
    /// ```text
    /// x[n] = Σ(k=0 to K-1) CQT[k] * W[k,n]
    /// ```
    ///
    /// Due to the non-uniform sampling in the CQT, perfect reconstruction
    /// requires careful consideration of the dual frame or pseudo-inverse.
    ///
    /// # Arguments
    /// * `cqt_matrix` - CQT coefficients from `constant_q_transform`
    /// * `config` - CQT configuration used for the forward transform
    /// * `signal_length` - Length of the original signal in samples
    ///
    /// # Returns
    /// Reconstructed time-domain signal
    ///
    /// # Example
    /// ```rust,ignore
    /// let cqt = audio.constant_q_transform(&config)?;
    /// let reconstructed = AudioSamples::inverse_constant_q_transform(&cqt, &config, original_length)?;
    /// ```
    fn inverse_constant_q_transform(
        cqt_matrix: &Array2<Complex<f64>>,
        config: &super::types::CqtConfig,
        signal_length: usize,
        sample_rate: usize,
    ) -> AudioSampleResult<Self>
    where
        Self: Sized;

    /// Computes the CQT spectrogram for time-frequency analysis.
    ///
    /// Applies the CQT to overlapping windows of the signal to create a
    /// time-frequency representation with logarithmic frequency resolution.
    ///
    /// # Mathematical Foundation
    ///
    /// The CQT spectrogram is computed by applying the CQT to overlapping
    /// windows of the signal:
    /// ```text
    /// CQT_spectrogram[k,t] = CQT(x[t*hop_size:(t*hop_size + window_size)])
    /// ```
    ///
    /// # Applications
    ///
    /// - **Musical visualization**: Spectrograms with note-aligned frequency bins
    /// - **Harmonic tracking**: Following harmonics over time
    /// - **Onset detection**: Detecting note onsets in musical signals
    /// - **Tempo analysis**: Beat tracking and rhythm analysis
    ///
    /// # Arguments
    /// * `config` - CQT configuration parameters
    /// * `hop_size` - Number of samples between successive windows
    /// * `window_size` - Size of analysis windows in samples (None = auto-calculated)
    ///
    /// # Returns
    /// Complex-valued CQT spectrogram as `Array2<Complex<f64>>` with dimensions
    /// `(num_bins, num_frames)` where num_frames depends on signal length and hop_size
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = CqtConfig::chord_detection();
    /// let hop_size = 512;
    /// let cqt_spectrogram = audio.cqt_spectrogram(&config, hop_size, None)?;
    /// ```
    fn cqt_spectrogram(
        &self,
        config: &super::types::CqtConfig,
        hop_size: usize,
        window_size: Option<usize>,
    ) -> AudioSampleResult<Array2<Complex<f64>>>;

    /// Computes the magnitude CQT spectrogram for visualization and analysis.
    ///
    /// Similar to `cqt_spectrogram` but returns only the magnitude (amplitude)
    /// information, which is often sufficient for analysis and visualization.
    ///
    /// # Arguments
    /// * `config` - CQT configuration parameters
    /// * `hop_size` - Number of samples between successive windows
    /// * `window_size` - Size of analysis windows in samples (None = auto-calculated)
    /// * `power` - Whether to return power (magnitude²) instead of magnitude
    ///
    /// # Returns
    /// Real-valued magnitude CQT spectrogram as `Array2<f64>` with dimensions
    /// `(num_bins, num_frames)`
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = CqtConfig::musical();
    /// let hop_size = 512;
    /// let magnitude_spectrogram = audio.cqt_magnitude_spectrogram(&config, hop_size, None, false)?;
    /// ```
    fn cqt_magnitude_spectrogram(
        &self,
        config: &super::types::CqtConfig,
        hop_size: usize,
        window_size: Option<usize>,
        power: bool,
    ) -> AudioSampleResult<Array2<f64>>;
}

/// Pitch detection and fundamental frequency analysis.
///
/// This trait provides methods for detecting the fundamental frequency (pitch)
/// of audio signals using various algorithms. These methods are essential for
/// music analysis, tuning applications, and vocal processing.
pub trait AudioPitchAnalysis<T: AudioSample>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<T>: AudioTypeConversion<T>,
{
    /// Detects the fundamental frequency using the YIN algorithm.
    ///
    /// YIN is a robust pitch detection algorithm that works well with
    /// both musical and vocal signals. It uses autocorrelation with
    /// a cumulative normalized difference function.
    ///
    /// # Mathematical Foundation
    ///
    /// The YIN algorithm uses a difference function:
    /// ```text
    /// d_t(τ) = Σ(x_j - x_{j+τ})²
    /// ```
    ///
    /// And cumulative mean normalized difference:
    /// ```text
    /// d'_t(τ) = d_t(τ) / [(1/τ) * Σ d_t(j)] if τ > 0, else 1
    /// ```
    ///
    /// # Arguments
    /// * `threshold` - Confidence threshold (0.0-1.0, typically 0.1-0.2)
    /// * `min_frequency` - Minimum expected frequency in Hz (typically 80-100)
    /// * `max_frequency` - Maximum expected frequency in Hz (typically 1000-2000)
    ///
    /// # Returns
    /// * `Some(frequency)` - Detected fundamental frequency in Hz
    /// * `None` - No reliable pitch detected
    fn detect_pitch_yin(
        &self,
        threshold: f64,
        min_frequency: f64,
        max_frequency: f64,
    ) -> AudioSampleResult<Option<f64>>;

    /// Detects pitch using autocorrelation method.
    ///
    /// A simpler but less robust method that works well for clean signals.
    /// Uses the peak in the autocorrelation function to estimate periodicity.
    ///
    /// # Arguments
    /// * `min_frequency` - Minimum expected frequency in Hz
    /// * `max_frequency` - Maximum expected frequency in Hz
    ///
    /// # Returns
    /// * `Some(frequency)` - Detected fundamental frequency in Hz
    /// * `None` - No reliable pitch detected
    fn detect_pitch_autocorr(
        &self,
        min_frequency: f64,
        max_frequency: f64,
    ) -> AudioSampleResult<Option<f64>>;

    /// Tracks pitch over time using a sliding window analysis.
    ///
    /// Applies pitch detection to overlapping windows across the signal,
    /// providing a time-varying pitch contour.
    ///
    /// # Arguments
    /// * `window_size` - Size of analysis window in samples
    /// * `hop_size` - Number of samples between successive windows
    /// * `method` - Pitch detection method to use
    /// * `threshold` - Confidence threshold (for YIN method)
    /// * `min_frequency` - Minimum expected frequency in Hz
    /// * `max_frequency` - Maximum expected frequency in Hz
    ///
    /// # Returns
    /// Vector of (time_seconds, frequency_hz) pairs, where frequency
    /// is None if no pitch was detected in that window.
    fn track_pitch(
        &self,
        window_size: usize,
        hop_size: usize,
        method: PitchDetectionMethod,
        threshold: f64,
        min_frequency: f64,
        max_frequency: f64,
    ) -> AudioSampleResult<Vec<(f64, Option<f64>)>>;

    /// Computes the harmonic-to-noise ratio (HNR).
    ///
    /// HNR measures the ratio of periodic (harmonic) to aperiodic (noise)
    /// components in the signal. Higher values indicate cleaner pitch.
    ///
    /// # Arguments
    /// * `fundamental_freq` - Known or estimated fundamental frequency
    /// * `num_harmonics` - Number of harmonics to analyze (typically 5-10)
    ///
    /// # Returns
    /// HNR value in dB. Higher values indicate stronger harmonic structure.
    fn harmonic_to_noise_ratio(
        &self,
        fundamental_freq: f64,
        num_harmonics: usize,
    ) -> AudioSampleResult<f64>;

    /// Performs harmonic analysis by detecting harmonics of a fundamental frequency.
    ///
    /// Returns the magnitudes of detected harmonics relative to the fundamental.
    /// Useful for timbre analysis and synthesis applications.
    ///
    /// # Arguments
    /// * `fundamental_freq` - Fundamental frequency in Hz
    /// * `num_harmonics` - Number of harmonics to analyze
    /// * `tolerance` - Frequency tolerance for harmonic detection (0.0-1.0)
    ///
    /// # Returns
    /// Vector of harmonic magnitudes normalized to the fundamental (index 0).
    fn harmonic_analysis(
        &self,
        fundamental_freq: f64,
        num_harmonics: usize,
        tolerance: f64,
    ) -> AudioSampleResult<Vec<f64>>;

    /// Estimates the key/pitch class of musical audio.
    ///
    /// Uses chromagram analysis to determine the most likely musical key.
    /// Returns both the key and a confidence measure.
    ///
    /// # Arguments
    /// * `window_size` - Size of analysis window for chromagram
    /// * `hop_size` - Hop size for chromagram analysis
    ///
    /// # Returns
    /// Tuple of (key_index, confidence) where key_index is 0-11 for
    /// C, C#, D, D#, E, F, F#, G, G#, A, A#, B and confidence is 0.0-1.0
    fn estimate_key(&self, window_size: usize, hop_size: usize) -> AudioSampleResult<(usize, f64)>;
}

/// IIR (Infinite Impulse Response) filtering operations.
///
/// This trait provides methods for applying IIR filters to audio signals.
/// IIR filters are recursive filters that can achieve sharp roll-offs
/// with fewer coefficients than FIR filters.
pub trait AudioIirFiltering<T: AudioSample>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<T>: AudioTypeConversion<T>,
{
    /// Apply an IIR filter using the specified design parameters.
    ///
    /// Creates and applies an IIR filter based on the provided design
    /// specification. The filter state is maintained internally.
    ///
    /// # Arguments
    /// * `design` - Filter design parameters including type, order, and frequencies
    /// * `sample_rate` - Sample rate of the audio signal
    ///
    /// # Returns
    /// Result containing the filtered audio or an error if the design is invalid
    ///
    /// # Example
    /// ```rust,ignore
    /// use audio_samples::{AudioSamples, operations::*};
    /// use audio_samples::operations::types::IirFilterDesign;
    ///
    /// let design = IirFilterDesign::butterworth_lowpass(4, 1000.0);
    /// let mut audio = AudioSamples::new_mono(samples, 44100);
    /// audio.apply_iir_filter(&design, 44100)?;
    /// ```
    fn apply_iir_filter(
        &mut self,
        design: &super::types::IirFilterDesign,
        sample_rate: f64,
    ) -> AudioSampleResult<()>;

    /// Apply a simple Butterworth low-pass filter.
    ///
    /// Convenience method for applying a Butterworth low-pass filter
    /// without needing to create a full design specification.
    ///
    /// # Arguments
    /// * `order` - Filter order (number of poles)
    /// * `cutoff_frequency` - Cutoff frequency in Hz
    /// * `sample_rate` - Sample rate of the audio signal
    ///
    /// # Returns
    /// Result indicating success or failure
    fn butterworth_lowpass(
        &mut self,
        order: usize,
        cutoff_frequency: f64,
        sample_rate: f64,
    ) -> AudioSampleResult<()>;

    /// Apply a simple Butterworth high-pass filter.
    ///
    /// Convenience method for applying a Butterworth high-pass filter
    /// without needing to create a full design specification.
    ///
    /// # Arguments
    /// * `order` - Filter order (number of poles)
    /// * `cutoff_frequency` - Cutoff frequency in Hz
    /// * `sample_rate` - Sample rate of the audio signal
    ///
    /// # Returns
    /// Result indicating success or failure
    fn butterworth_highpass(
        &mut self,
        order: usize,
        cutoff_frequency: f64,
        sample_rate: f64,
    ) -> AudioSampleResult<()>;

    /// Apply a simple Butterworth band-pass filter.
    ///
    /// Convenience method for applying a Butterworth band-pass filter
    /// without needing to create a full design specification.
    ///
    /// # Arguments
    /// * `order` - Filter order (number of poles)
    /// * `low_frequency` - Lower cutoff frequency in Hz
    /// * `high_frequency` - Upper cutoff frequency in Hz
    /// * `sample_rate` - Sample rate of the audio signal
    ///
    /// # Returns
    /// Result indicating success or failure
    fn butterworth_bandpass(
        &mut self,
        order: usize,
        low_frequency: f64,
        high_frequency: f64,
        sample_rate: f64,
    ) -> AudioSampleResult<()>;

    /// Apply a Chebyshev Type I filter.
    ///
    /// Chebyshev Type I filters have ripple in the passband but provide
    /// sharper roll-off than Butterworth filters of the same order.
    ///
    /// # Arguments
    /// * `order` - Filter order (number of poles)
    /// * `cutoff_frequency` - Cutoff frequency in Hz
    /// * `passband_ripple` - Passband ripple in dB
    /// * `sample_rate` - Sample rate of the audio signal
    /// * `response` - Filter response type (low-pass, high-pass, etc.)
    ///
    /// # Returns
    /// Result indicating success or failure
    fn chebyshev_i(
        &mut self,
        order: usize,
        cutoff_frequency: f64,
        passband_ripple: f64,
        sample_rate: f64,
        response: super::types::FilterResponse,
    ) -> AudioSampleResult<()>;

    /// Get the frequency response of the current filter.
    ///
    /// Computes the magnitude and phase response of the filter at
    /// the specified frequencies. Useful for analyzing filter characteristics.
    ///
    /// # Arguments
    /// * `frequencies` - Frequencies at which to compute the response (in Hz)
    /// * `sample_rate` - Sample rate of the audio signal
    ///
    /// # Returns
    /// Tuple of (magnitude_response, phase_response) vectors
    fn frequency_response(
        &self,
        frequencies: &[f64],
        sample_rate: f64,
    ) -> AudioSampleResult<(Vec<f64>, Vec<f64>)>;
}

/// Parametric equalization operations.
///
/// This trait provides methods for applying parametric EQ to audio signals.
/// Parametric EQ allows precise frequency shaping with adjustable frequency,
/// gain, and Q (bandwidth) parameters for each band.
pub trait AudioParametricEq<T: AudioSample>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<T>: AudioTypeConversion<T> + AudioChannelOps<T>,
{
    /// Apply a parametric EQ to the audio signal.
    ///
    /// Processes the audio through all enabled bands in the parametric EQ
    /// configuration, applying frequency shaping according to each band's
    /// type, frequency, gain, and Q parameters.
    ///
    /// # Arguments
    /// * `eq` - Parametric EQ configuration with multiple bands
    /// * `sample_rate` - Sample rate of the audio signal
    ///
    /// # Returns
    /// Result indicating success or failure
    ///
    /// # Example
    /// ```rust,ignore
    /// use audio_samples::{AudioSamples, operations::*};
    /// use audio_samples::operations::types::{ParametricEq, EqBand};
    ///
    /// let mut eq = ParametricEq::new();
    /// eq.add_band(EqBand::peak(1000.0, 3.0, 2.0)); // +3dB boost at 1kHz
    /// eq.add_band(EqBand::low_shelf(100.0, -2.0, 0.707)); // -2dB cut below 100Hz
    ///
    /// let mut audio = AudioSamples::new_mono(samples, 44100);
    /// audio.apply_parametric_eq(&eq, 44100.0)?;
    /// ```
    fn apply_parametric_eq(
        &mut self,
        eq: &super::types::ParametricEq,
        sample_rate: f64,
    ) -> AudioSampleResult<()>;

    /// Apply a single EQ band to the audio signal.
    ///
    /// Convenience method for applying a single parametric EQ band
    /// without creating a full ParametricEq configuration.
    ///
    /// # Arguments
    /// * `band` - EQ band configuration
    /// * `sample_rate` - Sample rate of the audio signal
    ///
    /// # Returns
    /// Result indicating success or failure
    fn apply_eq_band(
        &mut self,
        band: &super::types::EqBand,
        sample_rate: f64,
    ) -> AudioSampleResult<()>;

    /// Apply a peak/notch filter at the specified frequency.
    ///
    /// Convenience method for applying a peak or notch filter.
    /// Positive gain creates a peak, negative gain creates a notch.
    ///
    /// # Arguments
    /// * `frequency` - Center frequency in Hz
    /// * `gain_db` - Gain in dB (positive for boost, negative for cut)
    /// * `q_factor` - Quality factor (bandwidth control)
    /// * `sample_rate` - Sample rate of the audio signal
    ///
    /// # Returns
    /// Result indicating success or failure
    fn apply_peak_filter(
        &mut self,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
        sample_rate: f64,
    ) -> AudioSampleResult<()>;

    /// Apply a low shelf filter.
    ///
    /// Affects frequencies below the corner frequency with a gentle
    /// boost or cut that levels off at low frequencies.
    ///
    /// # Arguments
    /// * `frequency` - Corner frequency in Hz
    /// * `gain_db` - Gain in dB (positive for boost, negative for cut)
    /// * `q_factor` - Shelf slope control
    /// * `sample_rate` - Sample rate of the audio signal
    ///
    /// # Returns
    /// Result indicating success or failure
    fn apply_low_shelf(
        &mut self,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
        sample_rate: f64,
    ) -> AudioSampleResult<()>;

    /// Apply a high shelf filter.
    ///
    /// Affects frequencies above the corner frequency with a gentle
    /// boost or cut that levels off at high frequencies.
    ///
    /// # Arguments
    /// * `frequency` - Corner frequency in Hz
    /// * `gain_db` - Gain in dB (positive for boost, negative for cut)
    /// * `q_factor` - Shelf slope control
    /// * `sample_rate` - Sample rate of the audio signal
    ///
    /// # Returns
    /// Result indicating success or failure
    fn apply_high_shelf(
        &mut self,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
        sample_rate: f64,
    ) -> AudioSampleResult<()>;

    /// Create and apply a common 3-band EQ (low shelf, mid peak, high shelf).
    ///
    /// Convenience method for applying a typical 3-band EQ configuration
    /// commonly found in audio equipment.
    ///
    /// # Arguments
    /// * `low_freq` - Low shelf corner frequency in Hz
    /// * `low_gain` - Low shelf gain in dB
    /// * `mid_freq` - Mid peak center frequency in Hz
    /// * `mid_gain` - Mid peak gain in dB
    /// * `mid_q` - Mid peak Q factor
    /// * `high_freq` - High shelf corner frequency in Hz
    /// * `high_gain` - High shelf gain in dB
    /// * `sample_rate` - Sample rate of the audio signal
    ///
    /// # Returns
    /// Result indicating success or failure
    fn apply_three_band_eq(
        &mut self,
        low_freq: f64,
        low_gain: f64,
        mid_freq: f64,
        mid_gain: f64,
        mid_q: f64,
        high_freq: f64,
        high_gain: f64,
        sample_rate: f64,
    ) -> AudioSampleResult<()>;

    /// Get the combined frequency response of a parametric EQ.
    ///
    /// Computes the magnitude and phase response of the complete
    /// parametric EQ at the specified frequencies.
    ///
    /// # Arguments
    /// * `eq` - Parametric EQ configuration
    /// * `frequencies` - Frequencies at which to compute the response (in Hz)
    /// * `sample_rate` - Sample rate of the audio signal
    ///
    /// # Returns
    /// Tuple of (magnitude_response, phase_response) vectors
    fn eq_frequency_response(
        &self,
        eq: &super::types::ParametricEq,
        frequencies: &[f64],
        sample_rate: f64,
    ) -> AudioSampleResult<(Vec<f64>, Vec<f64>)>;
}

/// Dynamic range processing operations.
///
/// This trait provides methods for compressing and limiting audio signals
/// to control dynamic range. These operations are essential for professional
/// audio production, mixing, and mastering.
pub trait AudioDynamicRange<T: AudioSample>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<T>: AudioTypeConversion<T>,
{
    /// Apply compression to the audio signal.
    ///
    /// Compresses the dynamic range by reducing the amplitude of signals
    /// that exceed the threshold. The amount of reduction is determined
    /// by the compression ratio.
    ///
    /// # Mathematical Foundation
    ///
    /// For a signal level `x` above the threshold `T` with ratio `R`:
    /// ```text
    /// output_level = T + (x - T) / R
    /// ```
    ///
    /// Attack and release times control the envelope follower:
    /// ```text
    /// envelope[n] = α * envelope[n-1] + (1-α) * |input[n]|
    /// ```
    /// where `α` is calculated from the attack/release time constants.
    ///
    /// # Applications
    ///
    /// - **Vocals**: Smooth out level variations for consistent presence
    /// - **Drums**: Control transients and add punch
    /// - **Mix bus**: Glue mix elements together
    /// - **Mastering**: Control overall dynamics
    ///
    /// # Arguments
    /// * `config` - Compressor configuration parameters
    /// * `sample_rate` - Sample rate of the audio signal
    ///
    /// # Returns
    /// Result indicating success or failure
    ///
    /// # Example
    /// ```rust,ignore
    /// use audio_samples::{AudioSamples, operations::*};
    /// use audio_samples::operations::types::CompressorConfig;
    ///
    /// let config = CompressorConfig::vocal();
    /// let mut audio = AudioSamples::new_mono(samples, 44100);
    /// audio.apply_compressor(&config, 44100.0)?;
    /// ```
    fn apply_compressor(
        &mut self,
        config: &super::types::CompressorConfig,
        sample_rate: f64,
    ) -> AudioSampleResult<()>;

    /// Apply limiting to the audio signal.
    ///
    /// Prevents the signal from exceeding the specified ceiling level.
    /// Limiting is typically used as the final stage in mastering to
    /// prevent clipping and maximize loudness.
    ///
    /// # Mathematical Foundation
    ///
    /// Limiting uses a very high compression ratio (typically ∞:1) above the ceiling:
    /// ```text
    /// output_level = min(ceiling, input_level)
    /// ```
    ///
    /// Lookahead processing delays the audio while analyzing future samples:
    /// ```text
    /// gain_reduction[n] = calculate_gain(input[n + lookahead_samples])
    /// output[n] = input[n] * gain_reduction[n]
    /// ```
    ///
    /// # Applications
    ///
    /// - **Mastering**: Final peak control and loudness maximization
    /// - **Broadcast**: Ensure signal compliance with broadcasting standards
    /// - **Live sound**: Prevent speaker damage and feedback
    /// - **Protection**: Safeguard against unexpected signal peaks
    ///
    /// # Arguments
    /// * `config` - Limiter configuration parameters
    /// * `sample_rate` - Sample rate of the audio signal
    ///
    /// # Returns
    /// Result indicating success or failure
    ///
    /// # Example
    /// ```rust,ignore
    /// use audio_samples::{AudioSamples, operations::*};
    /// use audio_samples::operations::types::LimiterConfig;
    ///
    /// let config = LimiterConfig::mastering();
    /// let mut audio = AudioSamples::new_mono(samples, 44100);
    /// audio.apply_limiter(&config, 44100.0)?;
    /// ```
    fn apply_limiter(
        &mut self,
        config: &super::types::LimiterConfig,
        sample_rate: f64,
    ) -> AudioSampleResult<()>;

    /// Apply compression with external side-chain input.
    ///
    /// The compressor is controlled by an external audio signal instead of
    /// the input signal itself. This enables advanced techniques like
    /// ducking, pumping effects, and frequency-conscious compression.
    ///
    /// # Applications
    ///
    /// - **Ducking**: Reduce music level when vocals are present
    /// - **Pumping**: Rhythmic compression driven by kick drum
    /// - **De-essing**: Frequency-specific compression for harsh sibilants
    /// - **Multiband**: Independent compression of different frequency bands
    ///
    /// # Arguments
    /// * `config` - Compressor configuration parameters
    /// * `sidechain_signal` - External control signal
    /// * `sample_rate` - Sample rate of the audio signal
    ///
    /// # Returns
    /// Result indicating success or failure
    ///
    /// # Example
    /// ```rust,ignore
    /// use audio_samples::{AudioSamples, operations::*};
    /// use audio_samples::operations::types::CompressorConfig;
    ///
    /// let mut config = CompressorConfig::new();
    /// config.side_chain.enable();
    /// let mut audio = AudioSamples::new_mono(samples, 44100);
    /// let sidechain = AudioSamples::new_mono(sidechain_samples, 44100);
    /// audio.apply_compressor_sidechain(&config, &sidechain, 44100.0)?;
    /// ```
    fn apply_compressor_sidechain(
        &mut self,
        config: &super::types::CompressorConfig,
        sidechain_signal: &Self,
        sample_rate: f64,
    ) -> AudioSampleResult<()>
    where
        Self: Sized;

    /// Apply limiting with external side-chain input.
    ///
    /// The limiter is controlled by an external audio signal instead of
    /// the input signal itself. This enables advanced limiting techniques
    /// and frequency-conscious peak control.
    ///
    /// # Applications
    ///
    /// - **Multiband limiting**: Independent limiting of frequency bands
    /// - **Frequency-conscious limiting**: Prevent specific frequencies from dominating
    /// - **Advanced mastering**: Sophisticated peak control techniques
    ///
    /// # Arguments
    /// * `config` - Limiter configuration parameters
    /// * `sidechain_signal` - External control signal
    /// * `sample_rate` - Sample rate of the audio signal
    ///
    /// # Returns
    /// Result indicating success or failure
    fn apply_limiter_sidechain(
        &mut self,
        config: &super::types::LimiterConfig,
        sidechain_signal: &Self,
        sample_rate: f64,
    ) -> AudioSampleResult<()>
    where
        Self: Sized;

    /// Analyze the compression curve characteristics.
    ///
    /// Computes the input-output relationship for the given compressor
    /// configuration at specified input levels. This is useful for
    /// visualizing and analyzing compressor behavior.
    ///
    /// # Arguments
    /// * `config` - Compressor configuration parameters
    /// * `input_levels_db` - Input levels in dB to analyze
    /// * `sample_rate` - Sample rate (affects attack/release timing)
    ///
    /// # Returns
    /// Vector of output levels in dB corresponding to the input levels
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = CompressorConfig::new();
    /// let input_levels = vec![-40.0, -30.0, -20.0, -10.0, 0.0];
    /// let output_levels = audio.get_compression_curve(&config, &input_levels, 44100.0)?;
    /// ```
    fn get_compression_curve(
        &self,
        config: &super::types::CompressorConfig,
        input_levels_db: &[f64],
        sample_rate: f64,
    ) -> AudioSampleResult<Vec<f64>>;

    /// Track gain reduction over time.
    ///
    /// Applies the compressor/limiter and returns the amount of gain reduction
    /// applied at each sample. This is useful for visualization, metering,
    /// and analysis of dynamic range processing.
    ///
    /// # Arguments
    /// * `config` - Compressor configuration parameters
    /// * `sample_rate` - Sample rate of the audio signal
    ///
    /// # Returns
    /// Vector of gain reduction values in dB (positive values indicate reduction)
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = CompressorConfig::new();
    /// let gain_reduction = audio.get_gain_reduction(&config, 44100.0)?;
    /// let max_reduction = gain_reduction.iter().fold(0.0, |a, &b| a.max(b));
    /// ```
    fn get_gain_reduction(
        &self,
        config: &super::types::CompressorConfig,
        sample_rate: f64,
    ) -> AudioSampleResult<Vec<f64>>;

    /// Apply gate processing to the audio signal.
    ///
    /// A gate (noise gate) mutes or attenuates the signal when it falls
    /// below a specified threshold. This is useful for removing background
    /// noise and cleaning up recordings.
    ///
    /// # Mathematical Foundation
    ///
    /// For a signal level `x` below the threshold `T` with ratio `R`:
    /// ```text
    /// output_level = T + (x - T) / R  (where R >> 1, typically ∞)
    /// ```
    ///
    /// # Applications
    ///
    /// - **Noise removal**: Eliminate background noise between phrases
    /// - **Drum gating**: Remove bleed between drum hits
    /// - **Vocal cleanup**: Remove breath noise and room tone
    /// - **Creative effects**: Rhythmic gating for musical effects
    ///
    /// # Arguments
    /// * `threshold_db` - Gate threshold in dB
    /// * `ratio` - Gate ratio (typically very high, e.g., 10:1 or ∞:1)
    /// * `attack_ms` - Attack time in milliseconds
    /// * `release_ms` - Release time in milliseconds
    /// * `sample_rate` - Sample rate of the audio signal
    ///
    /// # Returns
    /// Result indicating success or failure
    fn apply_gate(
        &mut self,
        threshold_db: f64,
        ratio: f64,
        attack_ms: f64,
        release_ms: f64,
        sample_rate: f64,
    ) -> AudioSampleResult<()>;

    /// Apply expansion to the audio signal.
    ///
    /// Expansion increases the dynamic range by increasing the amplitude
    /// difference between loud and quiet parts. It's the opposite of compression.
    ///
    /// # Mathematical Foundation
    ///
    /// For a signal level `x` below the threshold `T` with ratio `R`:
    /// ```text
    /// output_level = T + (x - T) * R  (where R > 1)
    /// ```
    ///
    /// # Applications
    ///
    /// - **Restore dynamics**: Reverse over-compression
    /// - **Enhance transients**: Make drums and percussive elements more punchy
    /// - **Creative effects**: Exaggerate dynamic differences
    /// - **Mix enhancement**: Add life to flat-sounding material
    ///
    /// # Arguments
    /// * `threshold_db` - Expansion threshold in dB
    /// * `ratio` - Expansion ratio (typically 1.5:1 to 4:1)
    /// * `attack_ms` - Attack time in milliseconds
    /// * `release_ms` - Release time in milliseconds
    /// * `sample_rate` - Sample rate of the audio signal
    ///
    /// # Returns
    /// Result indicating success or failure
    fn apply_expander(
        &mut self,
        threshold_db: f64,
        ratio: f64,
        attack_ms: f64,
        release_ms: f64,
        sample_rate: f64,
    ) -> AudioSampleResult<()>;
}

/// Time-domain editing and manipulation operations.
///
/// This trait provides methods for cutting, pasting, mixing, and modifying
/// audio samples in the time domain. Most operations create new AudioSamples
/// instances rather than modifying in-place to preserve the original data.
pub trait AudioEditing<T: AudioSample>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<T>: AudioTypeConversion<T>,
{
    /// Reverses the order of audio samples.
    ///
    /// Creates a new AudioSamples instance with time-reversed content.
    fn reverse(&self) -> AudioSamples<T>
    where
        Self: Sized;

    /// Reverses the order of audio samples in place.
    fn reverse_in_place(&mut self) -> AudioSampleResult<()>
    where
        Self: Sized;

    /// Extracts a segment of audio between start and end times.
    ///
    /// # Arguments
    /// * `start_seconds` - Start time in seconds
    /// * `end_seconds` - End time in seconds
    ///
    /// # Errors
    /// Returns an error if start >= end or if times are out of bounds.
    fn trim(&self, start_seconds: f64, end_seconds: f64) -> AudioSampleResult<AudioSamples<T>>
    where
        Self: Sized;

    /// Adds padding/silence to the beginning and/or end of the audio.
    ///
    /// # Arguments
    /// * `pad_start_seconds` - Duration to pad at the beginning
    /// * `pad_end_seconds` - Duration to pad at the end
    /// * `pad_value` - Value to use for padding (typically zero for silence)
    fn pad(
        &self,
        pad_start_seconds: f64,
        pad_end_seconds: f64,
        pad_value: T,
    ) -> AudioSampleResult<AudioSamples<T>>
    where
        Self: Sized;

    fn pad_to_duration(
        &self,
        target_duration_seconds: f64,
        pad_value: T,
        pad_side: PadSide,
    ) -> AudioSampleResult<AudioSamples<T>>
    where
        Self: Sized;

    /// Splits audio into segments of specified duration.
    ///
    /// The last segment may be shorter if the audio doesn't divide evenly.
    ///
    /// # Arguments
    /// * `segment_duration_seconds` - Duration of each segment
    fn split(&self, segment_duration_seconds: f64) -> AudioSampleResult<Vec<AudioSamples<T>>>
    where
        Self: Sized;

    /// Concatenates multiple audio segments into one.
    ///
    /// All segments must have the same sample rate and channel configuration.
    ///
    /// # Arguments
    /// * `segments` - Audio segments to concatenate in order
    fn concatenate(segments: &[Self]) -> AudioSampleResult<AudioSamples<T>>
    where
        Self: Sized;

    /// Mixes multiple audio sources together.
    ///
    /// Sources must have the same sample rate, channel count, and length.
    /// Optional weights can be provided for each source.
    ///
    /// # Arguments
    /// * `sources` - Audio sources to mix
    /// * `weights` - Optional mixing weights (defaults to equal weighting)
    fn mix(sources: &[Self], weights: Option<&[f64]>) -> AudioSampleResult<AudioSamples<T>>
    where
        Self: Sized;

    /// Applies fade-in envelope over specified duration.
    ///
    /// # Arguments
    /// * `duration_seconds` - Duration of the fade-in
    /// * `curve` - Shape of the fade curve
    fn fade_in(&mut self, duration_seconds: f64, curve: FadeCurve) -> AudioSampleResult<()>;

    /// Applies fade-out envelope over specified duration.
    ///
    /// # Arguments
    /// * `duration_seconds` - Duration of the fade-out
    /// * `curve` - Shape of the fade curve
    fn fade_out(&mut self, duration_seconds: f64, curve: FadeCurve) -> AudioSampleResult<()>;

    /// Repeats the audio signal a specified number of times.
    ///
    /// # Arguments
    /// * `count` - Number of repetitions (total length = original × count)
    fn repeat(&self, count: usize) -> AudioSampleResult<AudioSamples<T>>
    where
        Self: Sized;

    /// Crops audio to remove silence from beginning and end.
    ///
    /// # Arguments
    /// * `threshold` - Amplitude threshold below which samples are considered silence
    ///
    /// # Returns
    /// A new AudioSamples instance with leading and trailing silence removed
    /// # Errors
    /// Returns an error if the operation fails for any reason
    fn trim_silence(&self, threshold: T) -> AudioSampleResult<AudioSamples<T>>
    where
        Self: Sized;

    /// Applies perturbation to audio samples for data augmentation.
    ///
    /// Creates a new AudioSamples instance with the specified perturbation applied.
    /// This method preserves the original audio data while creating a modified copy.
    ///
    /// # Arguments
    /// * `config` - Perturbation configuration specifying method and parameters
    ///
    /// # Returns
    /// A new AudioSamples instance with perturbation applied
    ///
    /// # Errors
    /// Returns an error if the perturbation configuration is invalid or if the
    /// perturbation operation fails (e.g., insufficient memory, invalid parameters).
    ///
    /// # Examples
    /// ```rust,ignore
    /// use audio_samples::{AudioSamples, operations::*};
    /// use audio_samples::operations::types::*;
    ///
    /// let audio = AudioSamples::new_mono(samples, 44100);
    ///
    /// // Add white noise at 20dB SNR
    /// let noise_config = PerturbationConfig::new(
    ///     PerturbationMethod::gaussian_noise(20.0, NoiseColor::White)
    /// );
    /// let noisy_audio = audio.perturb(&noise_config)?;
    ///
    /// // Apply random gain with deterministic seed
    /// let gain_config = PerturbationConfig::with_seed(
    ///     PerturbationMethod::random_gain(-3.0, 3.0),
    ///     12345
    /// );
    /// let gained_audio = audio.perturb(&gain_config)?;
    /// ```
    fn perturb(&self, config: &PerturbationConfig) -> AudioSampleResult<AudioSamples<T>>
    where
        Self: Sized;

    /// Applies perturbation to audio samples in place.
    ///
    /// Modifies the current AudioSamples instance by applying the specified perturbation.
    /// This method is more memory-efficient as it doesn't create a copy.
    ///
    /// # Arguments
    /// * `config` - Perturbation configuration specifying method and parameters
    ///
    /// # Returns
    /// Result indicating success or failure of the perturbation operation
    ///
    /// # Errors
    /// Returns an error if the perturbation configuration is invalid or if the
    /// perturbation operation fails.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use audio_samples::{AudioSamples, operations::*};
    /// use audio_samples::operations::types::*;
    ///
    /// let mut audio = AudioSamples::new_mono(samples, 44100);
    ///
    /// // Apply high-pass filter in place
    /// let filter_config = PerturbationConfig::new(
    ///     PerturbationMethod::high_pass_filter(80.0)
    /// );
    /// audio.perturb_(&filter_config)?;
    ///
    /// // Apply pitch shift with seed
    /// let pitch_config = PerturbationConfig::with_seed(
    ///     PerturbationMethod::pitch_shift(2.0, false),
    ///     54321
    /// );
    /// audio.perturb_(&pitch_config)?;
    /// ```
    fn perturb_(&mut self, config: &PerturbationConfig) -> AudioSampleResult<()>;

    /// Stacks multiple mono audio samples into a multi-channel audio sample.
    fn stack(sources: &[Self]) -> AudioSampleResult<AudioSamples<T>>
    where
        Self: Sized;
}

/// Channel manipulation and spatial audio operations.
///
/// This trait provides methods for converting between different channel
/// configurations and manipulating multi-channel audio.
pub trait AudioChannelOps<T: AudioSample>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<T>: AudioTypeConversion<T>,
{
    /// Converts multi-channel audio to mono using specified method.
    ///
    /// # Arguments
    /// * `method` - Method for combining channels into mono
    fn to_mono(&self, method: MonoConversionMethod) -> AudioSampleResult<Self>
    where
        Self: Sized;

    /// Converts mono audio to stereo using specified method.
    ///
    /// # Arguments
    /// * `method` - Method for creating stereo from mono
    fn to_stereo(&self, method: StereoConversionMethod) -> AudioSampleResult<Self>
    where
        Self: Sized;

    /// Extracts a specific channel from multi-channel audio.
    ///
    /// # Arguments
    /// * `channel_index` - Zero-based index of channel to extract
    fn extract_channel(&self, channel_index: usize) -> AudioSampleResult<Self>
    where
        Self: Sized;

    /// Swaps two channels in multi-channel audio.
    ///
    /// # Arguments
    /// * `channel1` - Index of first channel
    /// * `channel2` - Index of second channel
    fn swap_channels(&mut self, channel1: usize, channel2: usize) -> AudioSampleResult<()>;

    /// Applies pan control to stereo audio.
    ///
    /// # Arguments
    /// * `pan_value` - Pan position (-1.0 = full left, 0.0 = center, 1.0 = full right)
    fn pan(&mut self, pan_value: f64) -> AudioSampleResult<()>;

    /// Adjusts balance between left and right channels.
    ///
    /// # Arguments
    /// * `balance` - Balance adjustment (-1.0 = left only, 0.0 = equal, 1.0 = right only)
    fn balance(&mut self, balance: f64) -> AudioSampleResult<()>;

    fn apply_to_channel<F>(&mut self, channel_index: usize, func: F) -> AudioSampleResult<()>
    where
        F: FnMut(T) -> T,
        Self: Sized;

    fn interleave_channels(channels: &[Self]) -> AudioSampleResult<Self>
    where
        Self: Sized;

    fn deinterleave_channels(&self) -> AudioSampleResult<Vec<Self>>
    where
        Self: Sized;
}

/// Operation application and chaining functionality.
///
/// This trait provides methods for chaining operations together in fluent interfaces,
/// applying batches of operations efficiently, and creating reusable operation pipelines.
/// It enables more ergonomic and performant audio processing workflows.
pub trait AudioOperationApply<T: AudioSample>
where
    Self: Sized,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    /// Apply a generic operation function to the audio samples.
    ///
    /// This method enables functional-style audio processing by accepting
    /// any function that takes a mutable reference to Self and returns a Result.
    /// It provides a foundation for building fluent interfaces and operation chains.
    ///
    /// # Arguments
    /// * `operation` - Function that modifies the audio samples
    ///
    /// # Returns
    /// Self for method chaining, or an error if the operation fails
    ///
    /// # Example
    /// ```rust,ignore
    /// use audio_samples::{AudioSamples, operations::*};
    ///
    /// let result = audio
    ///     .apply(|samples| samples.normalize(T::from_f64(-1.0), T::from_f64(1.0), NormalizationMethod::Peak))?
    ///     .apply(|samples| samples.scale(T::from_f64(0.8)))?
    ///     .apply(|samples| samples.low_pass_filter(1000.0))?;
    /// ```
    fn apply<F>(mut self, operation: F) -> AudioSampleResult<Self>
    where
        F: FnOnce(&mut Self) -> AudioSampleResult<()>,
    {
        operation(&mut self)?;
        Ok(self)
    }

    /// Apply a fallible operation that may return a new instance.
    ///
    /// Some operations naturally return new AudioSamples instances rather than
    /// modifying in place. This method handles such operations while maintaining
    /// the fluent interface pattern.
    ///
    /// # Arguments
    /// * `operation` - Function that takes Self and returns a new instance
    ///
    /// # Returns
    /// New AudioSamples instance or an error if the operation fails
    ///
    /// # Example
    /// ```rust,ignore
    /// let result = audio
    ///     .apply_transform(|samples| samples.resample(48000, ResamplingQuality::High))?
    ///     .apply_transform(|samples| samples.to_mono(MonoConversionMethod::Average))?;
    /// ```
    fn apply_transform<F>(self, operation: F) -> AudioSampleResult<Self>
    where
        F: FnOnce(Self) -> AudioSampleResult<Self>,
    {
        operation(self)
    }

    /// Apply multiple operations in sequence with short-circuit error handling.
    ///
    /// This method applies a vector of operations in order, stopping at the first
    /// error encountered. It's useful for batch processing and creating reusable
    /// operation sequences.
    ///
    /// # Arguments
    /// * `operations` - Vector of operations to apply in sequence
    ///
    /// # Returns
    /// Self after all operations, or the first error encountered
    ///
    /// # Example
    /// ```rust,ignore
    /// let operations: Vec<Box<dyn FnOnce(&mut AudioSamples<f32>) -> AudioSampleResult<()>>> = vec![
    ///     Box::new(|s| s.remove_dc_offset()),
    ///     Box::new(|s| s.normalize(T::from_f64(-1.0), T::from_f64(1.0), NormalizationMethod::RMS)),
    ///     Box::new(|s| s.low_pass_filter(8000.0)),
    /// ];
    /// let result = audio.apply_batch(operations)?;
    /// ```
    fn apply_batch<F>(mut self, operations: Vec<F>) -> AudioSampleResult<Self>
    where
        F: FnOnce(&mut Self) -> AudioSampleResult<()>,
    {
        for operation in operations {
            operation(&mut self)?;
        }
        Ok(self)
    }

    /// Apply operations conditionally based on a predicate.
    ///
    /// This method only applies the operation if the condition evaluates to true.
    /// It's useful for dynamic processing pipelines where operations may be
    /// applied based on audio characteristics or user preferences.
    ///
    /// # Arguments
    /// * `condition` - Predicate that determines whether to apply the operation
    /// * `operation` - Operation to apply if condition is true
    ///
    /// # Returns
    /// Self (possibly modified) for method chaining
    ///
    /// # Example
    /// ```rust,ignore
    /// let result = audio
    ///     .apply_if(|samples| samples.peak() > T::from_f64(0.9), |samples| {
    ///         samples.apply_limiter(&LimiterConfig::default(), 44100.0)
    ///     })?
    ///     .apply_if(|samples| samples.channels() > 2, |samples| {
    ///         samples.to_stereo(StereoConversionMethod::DownmixCenter)
    ///     })?;
    /// ```
    fn apply_if<P, F>(mut self, condition: P, operation: F) -> AudioSampleResult<Self>
    where
        P: FnOnce(&Self) -> bool,
        F: FnOnce(&mut Self) -> AudioSampleResult<()>,
    {
        if condition(&self) {
            operation(&mut self)?;
        }
        Ok(self)
    }

    /// Try to apply an operation, falling back to a default operation on failure.
    ///
    /// This method attempts the primary operation first, and if it fails,
    /// applies the fallback operation instead. It's useful for robust processing
    /// pipelines where alternative processing methods should be used if the
    /// preferred method fails.
    ///
    /// # Arguments
    /// * `primary_op` - Primary operation to attempt
    /// * `fallback_op` - Fallback operation to use if primary fails
    ///
    /// # Returns
    /// Self after applying either primary or fallback operation
    ///
    /// # Example
    /// ```rust,ignore
    /// let result = audio.try_apply_or(
    ///     |samples| samples.resample(96000, ResamplingQuality::High),
    ///     |samples| samples.resample(48000, ResamplingQuality::Medium),
    /// )?;
    /// ```
    fn try_apply_or<F1, F2>(mut self, primary_op: F1, fallback_op: F2) -> AudioSampleResult<Self>
    where
        F1: FnOnce(&mut Self) -> AudioSampleResult<()>,
        F2: FnOnce(&mut Self) -> AudioSampleResult<()>,
    {
        if primary_op(&mut self).is_err() {
            fallback_op(&mut self)?;
        }
        Ok(self)
    }

    /// Apply an operation and collect metrics about its execution.
    ///
    /// This method wraps operation execution with timing and error tracking,
    /// useful for performance monitoring and debugging complex processing chains.
    ///
    /// # Arguments
    /// * `operation` - Operation to apply and measure
    /// * `operation_name` - Human-readable name for the operation
    ///
    /// # Returns
    /// Tuple of (modified_audio, execution_metrics)
    ///
    /// # Example
    /// ```rust,ignore
    /// let (result, metrics) = audio.apply_with_metrics(
    ///     |samples| samples.apply_compressor(&config, 44100.0),
    ///     "compressor"
    /// )?;
    /// println!("Operation {} took {:.2}ms", metrics.name, metrics.duration_ms);
    /// ```
    fn apply_with_metrics<F>(
        mut self,
        operation: F,
        operation_name: &str,
    ) -> AudioSampleResult<(Self, OperationMetrics)>
    where
        F: FnOnce(&mut Self) -> AudioSampleResult<()>,
    {
        let start = std::time::Instant::now();
        let result = operation(&mut self);
        let duration = start.elapsed();

        let metrics = OperationMetrics {
            name: operation_name.to_string(),
            duration_ms: duration.as_secs_f64() * 1000.0,
            success: result.is_ok(),
            error_message: result.as_ref().err().map(|e| e.to_string()),
        };

        result?;
        Ok((self, metrics))
    }

    /// Apply a sequence of operations using a simple closure-based approach.
    ///
    /// This method provides a lightweight alternative to full pipeline functionality
    /// by accepting a closure that can chain multiple operations together.
    ///
    /// # Arguments
    /// * `operations` - Closure that chains operations on the audio samples
    ///
    /// # Returns
    /// Self after applying all operations in the closure
    ///
    /// # Example
    /// ```rust,ignore
    /// let result = audio.apply_sequence(|s| {
    ///     s.apply(|s| s.normalize(T::from_f64(-1.0), T::from_f64(1.0), NormalizationMethod::Peak))?
    ///      .apply(|s| s.scale(T::from_f64(0.8)))?
    ///      .apply(|s| s.low_pass_filter(1000.0))
    /// })?;
    /// ```
    fn apply_sequence<F>(self, operations: F) -> AudioSampleResult<Self>
    where
        F: FnOnce(Self) -> AudioSampleResult<Self>,
    {
        operations(self)
    }
}

/// Metrics collected during operation execution.
///
/// Provides timing and error information for individual operations,
/// useful for performance monitoring and debugging.
#[derive(Debug, Clone)]
pub struct OperationMetrics {
    pub name: String,
    pub duration_ms: f64,
    pub success: bool,
    pub error_message: Option<String>,
}

/// A reusable pipeline of audio processing operations.
///
/// Encapsulates a sequence of operations that can be applied to multiple
/// AudioSamples instances efficiently. Pipelines can be built using a
/// fluent interface and provide built-in error handling and metrics.
pub struct OperationPipeline<T: AudioSample> {
    pub name: String,
    operations: Vec<Box<dyn Fn(&mut AudioSamples<T>) -> AudioSampleResult<()> + Send + Sync>>,
    collect_metrics: bool,
}

impl<T: AudioSample> OperationPipeline<T> {
    /// Create a new empty operation pipeline.
    pub fn new(name: String) -> Self {
        Self {
            name,
            operations: Vec::new(),
            collect_metrics: false,
        }
    }

    /// Add an operation to the pipeline.
    ///
    /// # Arguments
    /// * `operation` - Operation function to add to the pipeline
    pub fn add_operation<F>(mut self, operation: F) -> Self
    where
        F: Fn(&mut AudioSamples<T>) -> AudioSampleResult<()> + Send + Sync + 'static,
    {
        self.operations.push(Box::new(operation));
        self
    }

    /// Enable metrics collection for this pipeline.
    pub fn with_metrics(mut self) -> Self {
        self.collect_metrics = true;
        self
    }

    /// Apply the pipeline to audio samples.
    ///
    /// # Arguments
    /// * `audio` - Audio samples to process
    ///
    /// # Returns
    /// Processed audio samples or error if any operation fails
    pub fn apply<'a>(&'a self, mut audio: AudioSamples<T>) -> AudioSampleResult<AudioSamples<T>> {
        for (i, operation) in self.operations.iter().enumerate() {
            if self.collect_metrics {
                let start = std::time::Instant::now();
                let result = operation(&mut audio);
                let duration = start.elapsed();

                if let Err(e) = &result {
                    eprintln!(
                        "Pipeline '{}' operation {} failed after {:.2}ms: {}",
                        self.name,
                        i,
                        duration.as_secs_f64() * 1000.0,
                        e
                    );
                    return result.map(|_| audio);
                }
            } else {
                operation(&mut audio)?;
            }
        }
        Ok(audio)
    }

    /// Get the number of operations in this pipeline.
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    /// Create a copy of this pipeline with a new name.
    pub fn clone_with_name(&self, new_name: String) -> Self {
        Self {
            name: new_name,
            operations: Vec::new(), // Note: Cannot clone Fn trait objects
            collect_metrics: self.collect_metrics,
        }
    }
}

/// Real-time operation processing functionality.
///
/// This trait provides methods for applying operations in real-time scenarios
/// where low-latency processing is critical, such as live audio processing,
/// streaming, and interactive applications.
pub trait AudioRealtimeOps<T: AudioSample>: AudioOperationApply<T>
where
    Self: Sized,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    /// Apply operations with real-time constraints.
    ///
    /// This method applies operations with timing constraints suitable for
    /// real-time processing. Operations that take too long are skipped or
    /// simplified to maintain consistent throughput.
    ///
    /// # Arguments
    /// * `operations` - Queue of operations to apply
    /// * `max_processing_time_ms` - Maximum time budget per processing cycle
    ///
    /// # Returns
    /// Self after processing, with metrics about completed/skipped operations
    fn apply_realtime<F>(
        mut self,
        mut operations: VecDeque<F>,
        max_processing_time_ms: f64,
    ) -> AudioSampleResult<(Self, RealtimeMetrics)>
    where
        F: FnOnce(&mut Self) -> AudioSampleResult<()>,
    {
        let start_time = std::time::Instant::now();
        let max_duration = std::time::Duration::from_secs_f64(max_processing_time_ms / 1000.0);

        let mut completed_operations = 0;
        let mut skipped_operations = 0;

        while let Some(operation) = operations.pop_front() {
            if start_time.elapsed() >= max_duration {
                skipped_operations = operations.len() + 1;
                break;
            }

            if let Err(_) = operation(&mut self) {
                // In real-time processing, we continue despite errors
                // to maintain audio continuity
                continue;
            }

            completed_operations += 1;
        }

        let metrics = RealtimeMetrics {
            total_processing_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            completed_operations,
            skipped_operations,
            target_time_ms: max_processing_time_ms,
        };

        Ok((self, metrics))
    }

    /// Apply operations with adaptive quality based on available processing time.
    ///
    /// This method implements adaptive processing where operation quality
    /// is automatically adjusted based on available processing time and
    /// system load. High-quality operations are used when time permits,
    /// with graceful degradation under high load.
    ///
    /// # Arguments
    /// * `high_quality_op` - High-quality operation (slower)
    /// * `medium_quality_op` - Medium-quality operation (moderate speed)
    /// * `low_quality_op` - Low-quality operation (fastest)
    /// * `time_budget_ms` - Available processing time
    ///
    /// # Returns
    /// Self after applying the appropriate quality operation
    fn apply_adaptive_quality<F1, F2, F3>(
        mut self,
        high_quality_op: F1,
        medium_quality_op: F2,
        low_quality_op: F3,
        time_budget_ms: f64,
    ) -> AudioSampleResult<Self>
    where
        F1: FnOnce(&mut Self) -> AudioSampleResult<()>,
        F2: FnOnce(&mut Self) -> AudioSampleResult<()>,
        F3: FnOnce(&mut Self) -> AudioSampleResult<()>,
    {
        let _start_time = std::time::Instant::now();

        // Try high quality first if we have generous time budget
        if time_budget_ms > 10.0 {
            if high_quality_op(&mut self).is_ok() {
                return Ok(self);
            }
        }

        // Fall back to medium quality
        if time_budget_ms > 5.0 {
            if medium_quality_op(&mut self).is_ok() {
                return Ok(self);
            }
        }

        // Last resort: low quality operation
        low_quality_op(&mut self)?;
        Ok(self)
    }

    /// Buffer operations for batch processing to improve efficiency.
    ///
    /// This method collects operations in a buffer and applies them in
    /// optimized batches. This can improve cache efficiency and reduce
    /// overhead for certain types of operations.
    ///
    /// # Arguments
    /// * `operation` - Operation to add to the buffer
    /// * `buffer` - Operation buffer (shared between calls)
    /// * `batch_size` - Number of operations to collect before processing
    ///
    /// # Returns
    /// Self, possibly modified if buffer was flushed
    fn buffer_operation<F>(
        self,
        operation: F,
        buffer: &mut Vec<F>,
        batch_size: usize,
    ) -> AudioSampleResult<Self>
    where
        F: FnOnce(&mut Self) -> AudioSampleResult<()>,
    {
        buffer.push(operation);

        if buffer.len() >= batch_size {
            let operations = std::mem::take(buffer);
            return self.apply_batch(operations);
        }

        Ok(self)
    }
}

/// Metrics for real-time audio processing.
#[derive(Debug, Clone)]
pub struct RealtimeMetrics {
    pub total_processing_time_ms: f64,
    pub completed_operations: usize,
    pub skipped_operations: usize,
    pub target_time_ms: f64,
}

impl RealtimeMetrics {
    /// Check if processing met the real-time constraints.
    pub fn is_realtime(&self) -> bool {
        self.total_processing_time_ms <= self.target_time_ms && self.skipped_operations == 0
    }

    /// Get the processing efficiency (0.0 to 1.0).
    pub fn efficiency(&self) -> f64 {
        if self.target_time_ms > 0.0 {
            (self.target_time_ms - self.total_processing_time_ms).max(0.0) / self.target_time_ms
        } else {
            0.0
        }
    }
}

/// Utilities for plotting audio data.
///
/// Actual plotting functionality is implemented separately.
/// Allows the use of common plotting utilities across Rust and Python
pub trait AudioPlottingUtils<T: AudioSample>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<T>: AudioTypeConversion<T>,
{
    /// Generate time axis values for plotting.
    fn time_axis(&self, step: Option<f64>) -> Vec<f64>;
    /// Seconds from 0 to duration with ~target_ticks "nice" spacing (1–2–5).
    fn time_ticks_seconds(&self, target_ticks: usize) -> Vec<f64>;
    fn frequency_axis(&self) -> Vec<T>;
    /// Create a quick analysis plot with waveform and spectrum
    fn quick_plot(&self) -> super::PlotResult<()>
    where
        Self: crate::operations::plotting::AudioPlotBuilders<T>
    {
        let waveform = self.waveform_plot(None);
        let spectrum = self.power_spectrum_plot(None, None, None, None, None)?;

        let plot = crate::operations::plotting::PlotComposer::new()
            .add_element(waveform)
            .add_element(spectrum)
            .with_layout(crate::operations::plotting::LayoutConfig::VerticalStack)
            .with_title("Audio Analysis");

        plot.render_to_file("audio_analysis.png")
    }
}

/// Unified trait that combines all audio processing capabilities.
///
/// This trait extends all the focused traits, providing a single interface
/// for comprehensive audio processing. Use individual traits when you only
/// need specific functionality for better compile times and clearer dependencies.
///
/// This trait is automatically implemented for any type that implements
/// all the constituent traits.
pub trait AudioSamplesOperations<T: AudioSample>:
    AudioStatistics<T>
    + AudioProcessing<T>
    + AudioTransforms<T>
    + AudioDynamicRange<T>
    + AudioEditing<T>
    + AudioChannelOps<T>
    + AudioTypeConversion<T>
    + AudioPitchAnalysis<T>
    + AudioIirFiltering<T>
    + AudioParametricEq<T>
    + AudioOperationApply<T>
    + AudioRealtimeOps<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<T>: AudioTypeConversion<T>,
{
}

// Blanket implementation for the unified trait
impl<T: AudioSample, A> AudioSamplesOperations<T> for A
where
    A: AudioStatistics<T>
        + AudioProcessing<T>
        + AudioTransforms<T>
        + AudioDynamicRange<T>
        + AudioEditing<T>
        + AudioChannelOps<T>
        + AudioTypeConversion<T>
        + AudioPitchAnalysis<T>
        + AudioIirFiltering<T>
        + AudioParametricEq<T>
        + AudioOperationApply<T>
        + AudioRealtimeOps<T>,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<T>: AudioTypeConversion<T>,
{
}
