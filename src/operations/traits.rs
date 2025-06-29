//! Core trait definitions for audio processing operations.
//!
//! This module defines the focused traits that replace the monolithic
//! `AudioSamplesOperations` trait. Each trait has a single responsibility
//! and can be implemented independently.

use super::types::*;
use crate::{AudioSample, AudioSampleResult, AudioSamples, ConvertTo};
use ndarray::Array2;

// Complex numbers using num-complex crate
pub use num_complex::Complex;

/// Statistical analysis operations for audio data.
///
/// This trait provides methods to compute various statistical measures
/// of audio samples, useful for analysis, visualization, and processing decisions.
///
/// All methods return values in the native sample type `T` for consistency
/// with the underlying data representation.
pub trait AudioStatistics<T: AudioSample> {
    /// Returns the peak (maximum absolute value) in the audio samples.
    ///
    /// This is useful for preventing clipping and measuring signal levels.
    fn peak(&self) -> T;

    /// Returns the minimum value in the audio samples.
    fn min(&self) -> T;

    /// Returns the maximum value in the audio samples.
    fn max(&self) -> T;

    /// Computes the Root Mean Square (RMS) of the audio samples.
    ///
    /// RMS is useful for measuring average signal power/energy and
    /// provides a perceptually relevant measure of loudness.
    fn rms(&self) -> T;

    /// Computes the statistical variance of the audio samples.
    ///
    /// Variance measures the spread of sample values around the mean.
    fn variance(&self) -> T;

    /// Computes the standard deviation of the audio samples.
    ///
    /// Standard deviation is the square root of variance and provides
    /// a measure of signal variability in the same units as the samples.
    fn std_dev(&self) -> T;

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
    fn autocorrelation(&self, max_lag: usize) -> AudioSampleResult<Vec<T>>;

    /// Computes cross-correlation with another audio signal.
    ///
    /// Returns correlation values for each lag offset between the two signals.
    /// Useful for alignment, synchronization, and similarity analysis.
    ///
    /// # Arguments
    /// * `other` - The other audio signal to correlate with
    /// * `max_lag` - Maximum lag to compute (in samples)
    fn cross_correlation(&self, other: &Self, max_lag: usize) -> AudioSampleResult<Vec<T>>;

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
pub trait AudioProcessing<T: AudioSample> {
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
    fn scale(&mut self, factor: T) -> AudioSampleResult<()>;

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
}

/// Frequency domain analysis and spectral transformations.
///
/// This trait provides methods for FFT-based analysis and spectral processing.
/// Requires external FFT library dependencies.
///
/// Complex numbers are used for frequency domain representations,
/// and ndarray is used for efficient matrix operations on spectrograms.
pub trait AudioTransforms<T: AudioSample> {
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
}

/// Time-domain editing and manipulation operations.
///
/// This trait provides methods for cutting, pasting, mixing, and modifying
/// audio samples in the time domain. Most operations create new AudioSamples
/// instances rather than modifying in-place to preserve the original data.
pub trait AudioEditing<T: AudioSample> {
    /// Reverses the order of audio samples.
    ///
    /// Creates a new AudioSamples instance with time-reversed content.
    fn reverse(&self) -> Self
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
    fn trim(&self, start_seconds: f64, end_seconds: f64) -> AudioSampleResult<Self>
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
    ) -> AudioSampleResult<Self>
    where
        Self: Sized;

    /// Splits audio into segments of specified duration.
    ///
    /// The last segment may be shorter if the audio doesn't divide evenly.
    ///
    /// # Arguments
    /// * `segment_duration_seconds` - Duration of each segment
    fn split(&self, segment_duration_seconds: f64) -> AudioSampleResult<Vec<Self>>
    where
        Self: Sized;

    /// Concatenates multiple audio segments into one.
    ///
    /// All segments must have the same sample rate and channel configuration.
    ///
    /// # Arguments
    /// * `segments` - Audio segments to concatenate in order
    fn concatenate(segments: &[Self]) -> AudioSampleResult<Self>
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
    fn mix(sources: &[Self], weights: Option<&[f64]>) -> AudioSampleResult<Self>
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
    fn repeat(&self, count: usize) -> AudioSampleResult<Self>
    where
        Self: Sized;

    /// Crops audio to remove silence from beginning and end.
    ///
    /// # Arguments
    /// * `threshold` - Amplitude threshold below which samples are considered silence
    fn trim_silence(&self, threshold: T) -> AudioSampleResult<Self>
    where
        Self: Sized;
}

/// Channel manipulation and spatial audio operations.
///
/// This trait provides methods for converting between different channel
/// configurations and manipulating multi-channel audio.
pub trait AudioChannelOps<T: AudioSample> {
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

    /// Converts audio to specified number of channels.
    ///
    /// # Arguments
    /// * `target_channels` - Desired number of output channels
    /// * `method` - Method for channel conversion
    fn to_channels(
        &self,
        target_channels: usize,
        method: ChannelConversionMethod,
    ) -> AudioSampleResult<Self>
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
}

/// Type conversion operations between different sample formats.
///
/// This trait provides safe conversion between different audio sample types
/// while preserving audio quality and handling potential conversion errors.
/// Leverages the existing ConvertTo trait system for type safety.
pub trait AudioTypeConversion<T: AudioSample> {
    /// Converts to different sample type, borrowing the original.
    ///
    /// Uses the existing ConvertTo trait system for type-safe conversions.
    /// The original AudioSamples instance remains unchanged.
    fn as_type<O: AudioSample>(&self) -> AudioSampleResult<AudioSamples<O>>
    where
        T: ConvertTo<O>;

    /// Converts to different sample type, consuming the original.
    ///
    /// More efficient than as_type when the original is no longer needed.
    fn to_type<O: AudioSample>(self) -> AudioSampleResult<AudioSamples<O>>
    where
        T: ConvertTo<O>;

    /// Converts to the highest precision floating-point format.
    ///
    /// This is useful when maximum precision is needed for processing.
    fn to_f64(&self) -> AudioSampleResult<AudioSamples<f64>>
    where
        T: ConvertTo<f64>;

    /// Converts to single precision floating-point format.
    ///
    /// Good balance between precision and memory usage.
    fn to_f32(&self) -> AudioSampleResult<AudioSamples<f32>>
    where
        T: ConvertTo<f32>;

    /// Converts to 32-bit integer format.
    ///
    /// Highest precision integer format, useful for high-quality processing.
    fn to_i32(&self) -> AudioSampleResult<AudioSamples<i32>>
    where
        T: ConvertTo<i32>;

    /// Converts to 16-bit integer format (most common).
    ///
    /// Standard format for CD audio and many audio files.
    fn to_i16(&self) -> AudioSampleResult<AudioSamples<i16>>
    where
        T: ConvertTo<i16>;
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
    + AudioEditing<T>
    + AudioChannelOps<T>
    + AudioTypeConversion<T>
{
    // No additional methods - just combines all focused traits
}

// Blanket implementation for the unified trait
impl<T: AudioSample, A> AudioSamplesOperations<T> for A where
    A: AudioStatistics<T>
        + AudioProcessing<T>
        + AudioTransforms<T>
        + AudioEditing<T>
        + AudioChannelOps<T>
        + AudioTypeConversion<T>
{
}
