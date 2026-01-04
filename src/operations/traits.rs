//! Core trait definitions for audio processing operations.

#[cfg(feature = "plotting")]
#[cfg(feature = "spectral-analysis")]
use crate::operations::AudioPlotBuilders;
#[cfg(feature = "resampling")]
use crate::operations::ResamplingQuality;
#[cfg(feature = "processing")]
use crate::operations::types::NormalizationMethod;

#[cfg(feature = "spectral-analysis")]
use crate::operations::types::ChromaConfig;
#[cfg(feature = "spectral-analysis")]
use crate::operations::{
    CqtConfig,
    types::{SpectrogramScale, WindowType},
};

use crate::{
    AudioSample, AudioSampleResult, AudioSamples, AudioTypeConversion, CastFrom, CastInto,
    ConvertTo, I24, RealFloat,
    operations::{
        MonoConversionMethod, StereoConversionMethod,
        types::{
            CompressorConfig, EqBand, FadeCurve, FilterResponse, IirFilterDesign, LimiterConfig,
            PadSide, ParametricEq, PerturbationConfig, PitchDetectionMethod,
        },
    },
};

#[cfg(feature = "statistics")]
use crate::operations::types::VadConfig;
#[cfg(feature = "spectral-analysis")]
use ndarray::Array2;

#[cfg(feature = "random-generation")]
use rand::distr::{Distribution, StandardUniform};

#[cfg(feature = "spectral-analysis")]
use rustfft::FftNum;

#[cfg(feature = "plotting")]
#[cfg(feature = "spectral-analysis")]
use std::path::Path;

#[cfg(feature = "spectral-analysis")]
use std::num::NonZeroUsize;

#[cfg(feature = "spectral-analysis")]
use ndarray::Zip;

// Complex numbers using num-complex crate
pub use num_complex::Complex;

// Type aliases for complex types to satisfy clippy::type_complexity
#[cfg(feature = "spectral-analysis")]
type FftInfoResult<F> = AudioSampleResult<(Vec<F>, Vec<F>, Array2<Complex<F>>)>;

/// Statistical analysis operations for audio data.
///
/// This trait provides methods to compute various statistical measures
/// of audio samples, useful for analysis, visualization, and processing decisions.
///
/// All methods return values in the native sample type `T` for consistency
/// with the underlying data representation.
pub trait AudioStatistics<'a, T: AudioSample>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
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
    ///
    /// Always returns a value as audio samples cannot not be empty if properly contructed.
    fn mean<F>(&self) -> F
    where
        F: RealFloat;

    /// Computes the Root Mean Square (RMS) of the audio samples.
    ///
    /// RMS is useful for measuring average signal power/energy and
    /// provides a perceptually relevant measure of loudness.
    ///
    /// Always returns a value as audio samples cannot not be empty if properly contructed.
    fn rms<F>(&self) -> F
    where
        F: RealFloat;

    /// Computes the statistical variance of the audio samples.
    ///
    /// Variance measures the spread of sample values around the mean.
    ///
    /// Always returns a value as audio samples cannot not be empty if properly contructed.
    fn variance<F>(&self) -> F
    where
        F: RealFloat;

    /// Computes the standard deviation of the audio samples.
    ///
    /// Standard deviation is the square root of variance and provides
    /// a measure of signal variability in the same units as the samples.
    ///
    /// Always returns a value as audio samples cannot not be empty if properly contructed.
    fn std_dev<F>(&self) -> F
    where
        F: RealFloat;

    /// Counts the number of zero crossings in the audio signal.
    ///
    /// Zero crossings are useful for pitch detection and signal analysis.
    /// The count represents transitions from positive to negative values or vice versa.
    fn zero_crossings(&self) -> usize;

    /// Computes the zero crossing rate (crossings per second).
    ///
    /// This normalizes the zero crossing count by the signal duration,
    /// making it independent of audio length.
    fn zero_crossing_rate<F>(&self) -> F
    where
        F: RealFloat;

    /// Computes the autocorrelation function up to max_lag samples.
    ///
    /// Returns a vector of correlation values for each lag offset.
    /// Useful for pitch detection and periodicity analysis.
    ///
    /// # Arguments
    /// * `max_lag` - Maximum lag to compute (in samples)
    #[cfg(feature = "fft")]
    fn autocorrelation<F>(&self, max_lag: usize) -> Option<Vec<F>>
    where
        F: RealFloat;

    /// Computes cross-correlation with another audio signal.
    ///
    /// Returns correlation values for each lag offset between the two signals.
    /// Useful for alignment, synchronization, and similarity analysis.
    ///
    /// # Arguments
    /// * `other` - The other audio signal to correlate with
    /// * `max_lag` - Maximum lag to compute (in samples)
    fn cross_correlation<F>(&self, other: &Self, max_lag: usize) -> AudioSampleResult<Vec<F>>
    where
        F: RealFloat;

    /// Computes the spectral centroid (brightness measure).
    ///
    /// The spectral centroid represents the "center of mass" of the spectrum
    /// and is often used as a measure of brightness or timbre.
    /// Requires FFT computation internally.
    #[cfg(feature = "fft")]
    fn spectral_centroid<F: RealFloat + ConvertTo<T>>(&self) -> AudioSampleResult<F>
    where
        T: ConvertTo<F>;

    /// Computes spectral rolloff frequency.
    ///
    /// The rolloff frequency is the frequency below which a specified percentage
    /// of the total spectral energy is contained.
    ///
    /// # Arguments
    /// * `rolloff_percent` - Percentage of energy (0.0 to 1.0, typically 0.85)
    #[cfg(feature = "fft")]
    fn spectral_rolloff<F: RealFloat + ConvertTo<T>>(
        &self,
        rolloff_percent: F,
    ) -> AudioSampleResult<F>
    where
        T: ConvertTo<F>;
}

/// Voice Activity Detection (VAD) operations.
///
/// This trait provides frame-based voice/speech activity detection for audio.
/// It returns a boolean decision per frame (see [`VadConfig`]) and can also
/// derive contiguous speech regions in sample indices.
#[cfg(feature = "statistics")]
pub trait AudioVoiceActivityDetection<'a, T: AudioSample> {
    /// Compute a per-frame speech activity mask.
    ///
    /// The returned vector has one entry per analysis frame.
    fn voice_activity_mask<F: RealFloat>(
        &self,
        config: &VadConfig<F>,
    ) -> AudioSampleResult<Vec<bool>>;

    /// Compute contiguous speech regions as `(start_sample, end_sample)` pairs.
    ///
    /// Indices are in samples-per-channel units (i.e., frame boundaries are derived
    /// from `config.hop_size` / `config.frame_size`). `end_sample` is exclusive.
    fn speech_regions<F: RealFloat>(
        &self,
        config: &VadConfig<F>,
    ) -> AudioSampleResult<Vec<(usize, usize)>>;
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
    for<'a> AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
{
    /// Normalizes audio samples using the specified method and range.
    ///
    /// # Arguments
    /// * `min` - Target minimum value
    /// * `max` - Target maximum value
    /// * `max` - Target maximum value
    /// * `method` - Normalization method to use
    ///
    /// # Errors
    /// Returns an error if min >= max or if the method cannot be applied.
    #[cfg(feature = "processing")]
    #[cfg(feature = "processing")]
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
    fn low_pass_filter<F>(&mut self, cutoff_hz: F) -> AudioSampleResult<()>
    where
        F: RealFloat + ConvertTo<T>,
        T: CastFrom<F> + ConvertTo<F>;

    /// Applies a high-pass filter with the specified cutoff frequency.
    ///
    /// Frequencies below the cutoff will be attenuated.
    ///
    /// # Arguments
    /// * `cutoff_hz` - Cutoff frequency in Hz
    fn high_pass_filter<F>(&mut self, cutoff_hz: F) -> AudioSampleResult<()>
    where
        T: CastFrom<F> + ConvertTo<F>,
        F: RealFloat + ConvertTo<T>;

    /// Applies a band-pass filter between low and high frequencies.
    ///
    /// Only frequencies within the specified range will pass through.
    ///
    /// # Arguments
    /// * `low_cutoff_hz` - Lower cutoff frequency in Hz
    /// * `high_cutoff_hz` - Upper cutoff frequency in Hz
    fn band_pass_filter<F>(&mut self, low_cutoff_hz: F, high_cutoff_hz: F) -> AudioSampleResult<()>
    where
        T: CastFrom<F> + ConvertTo<F>,
        F: RealFloat + ConvertTo<T>;

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
    #[cfg(feature = "resampling")]
    fn resample<F>(
        &self,
        target_sample_rate: usize,
        quality: ResamplingQuality,
    ) -> AudioSampleResult<Self>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;

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
    #[cfg(feature = "resampling")]
    fn resample_by_ratio<F>(&self, ratio: F, quality: ResamplingQuality) -> AudioSampleResult<Self>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;
}

/// Frequency domain analysis and spectral transformations.
///
/// This trait provides methods for FFT-based analysis and spectral processing.
/// Requires external FFT library dependencies.
///
/// Complex numbers are used for frequency domain representations,
/// and ndarray is used for efficient matrix operations on spectrograms.
#[cfg(feature = "spectral-analysis")]
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
    ///
    /// The output is a 2D array with shape (1, num_bins) for single-channel audio.
    /// And (num_channels, num_bins) for multi-channel audio. Each row represents
    /// the FFT of a channel.
    fn fft<F>(&self) -> AudioSampleResult<Array2<Complex<F>>>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>;

    /// Compute FFT with additional information.
    fn fft_info<F>(
        &self,
        n_fft: Option<usize>,
        window: Option<WindowType<F>>,
        normalise: bool,
    ) -> FftInfoResult<F>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>,
        i16: ConvertTo<F>,
        I24: ConvertTo<F>,
        i32: ConvertTo<F>,
        f32: ConvertTo<F>,
        f64: ConvertTo<F>,
        for<'b> AudioSamples<'b, F>: AudioTypeConversion<'b, F>;

    /// Computes the inverse FFT from frequency domain back to time domain.
    ///
    /// # Arguments
    /// * `spectrum` - Complex frequency domain data
    fn ifft<F>(&self, spectrum: &Array2<Complex<F>>) -> AudioSampleResult<Self>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>,
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
    fn stft<F>(
        &self,
        window_size: usize,
        hop_size: usize,
        window_type: WindowType<F>,
    ) -> AudioSampleResult<Array2<Complex<F>>>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>;

    /// Computes the Short-Time Fourier Transform (STFT).
    ///
    /// Returns a tuple containing a 2D array where each column represents an FFT frame at a specific time and a vector of frequency bins.
    /// This provides both time and frequency information simultaneously.
    ///
    /// # Arguments
    /// * `window_size` - Size of each analysis window in samples
    /// * `hop_size` - Number of samples between successive windows
    /// * `window_type` - Window function to apply to each frame
    fn stft_with_freqs<F>(
        &self,
        window_size: usize,
        hop_size: usize,
        window_type: WindowType<F>,
    ) -> AudioSampleResult<(Array2<Complex<F>>, Vec<F>)>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>;

    /// Computes the inverse STFT to reconstruct time domain signal.
    ///
    /// Reconstructs a time-domain signal from its STFT representation.
    ///
    /// # Arguments
    /// * `stft_matrix` - STFT data (frequency bins × time frames)
    /// * `hop_size` - Hop size used in the original STFT
    /// * `window_type` - Window type used in the original STFT
    /// * `sample_rate` - Sample rate for the reconstructed signal
    /// * `center` - Whether the original signal was centered with padding
    ///
    /// # Returns
    ///
    /// Reconstructed time-domain audio samples
    ///
    /// # Errors
    ///
    /// Returns an error if reconstruction fails due to mismatched parameters.
    fn istft<F>(
        stft_matrix: &Array2<Complex<F>>,
        hop_size: usize,
        window_type: WindowType<F>,
        sample_rate: usize,
        center: bool,
    ) -> AudioSampleResult<Self>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>,
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
    fn spectrogram<F>(
        &self,
        window_size: usize,
        hop_size: usize,
        window_type: WindowType<F>,
        scale: SpectrogramScale,
        normalize: bool,
    ) -> AudioSampleResult<Array2<F>>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>;

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
    fn mel_spectrogram<F>(
        &self,
        n_mels: usize,
        fmin: F,
        fmax: F,
        window_size: usize,
        hop_size: usize,
    ) -> AudioSampleResult<Array2<F>>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>;

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
    fn mfcc<F>(
        &self,
        n_mfcc: usize,
        n_mels: usize,
        fmin: F,
        fmax: F,
    ) -> AudioSampleResult<Array2<F>>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>;

    /// Computes chromagram (pitch class profile).
    ///
    /// Useful for music analysis and chord detection by representing
    /// the energy in each of the 12 pitch classes (C, C#, D, etc.).
    ///
    /// # Arguments
    /// * `n_chroma` - Number of chroma bins (typically 12)
    fn chroma<F>(&self, n_chroma: usize) -> AudioSampleResult<Array2<F>>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>;

    /// Computes chromagram features with configurable parameters and methods.
    ///
    /// Chromagram represents the distribution of energy across pitch classes,
    /// providing a harmonic representation that is invariant to octave shifts.
    /// This method provides full control over the computation including choice
    /// between STFT and CQT methods.
    ///
    /// # Arguments
    /// * `cfg` - Configuration specifying method, window parameters, and normalization
    ///
    /// # Returns
    /// A 2D array with shape `(n_chroma, n_frames)` where each column represents
    /// the chroma vector for one time frame.
    ///
    /// # Examples
    /// ```rust,no_run
    /// use audio_samples::{AudioSamples, AudioTransforms};
    /// use audio_samples::operations::types::{ChromaConfig, ChromaMethod};
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let audio = AudioSamples::from_mono(&[0.1, 0.2, 0.3, 0.4], 44100)?;
    ///
    /// // Basic chromagram with STFT
    /// let config = ChromaConfig::stft();
    /// let chroma = audio.chromagram::<f64>(&config)?;
    ///
    /// // High resolution chromagram
    /// let config = ChromaConfig::high_resolution();
    /// let chroma = audio.chromagram::<f64>(&config)?;
    ///
    /// // CQT-based chromagram
    /// let config = ChromaConfig::cqt();
    /// let chroma = audio.chromagram::<f64>(&config)?;
    /// # Ok(())
    /// # }
    /// ```
    fn chromagram<F>(&self, cfg: &ChromaConfig<F>) -> AudioSampleResult<Array2<F>>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>;

    /// Computes the power spectral density using Welch's method.
    ///
    /// Returns both the frequency bins and corresponding power values.
    /// Welch's method provides better noise reduction than a single FFT.
    ///
    /// # Arguments
    /// * `window_size` - Size of each segment for averaging
    /// * `overlap` - Overlap between segments (0.0 to 1.0)
    fn power_spectral_density<F>(
        &self,
        window_size: usize,
        overlap: F,
    ) -> AudioSampleResult<(Vec<F>, Vec<F>)>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>; // (frequencies, psd)

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
    fn gammatone_spectrogram<F>(
        &self,
        n_filters: usize,
        fmin: F,
        fmax: F,
        window_size: usize,
        hop_size: usize,
    ) -> AudioSampleResult<Array2<F>>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>;

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
    fn constant_q_transform<F>(
        &self,
        config: &CqtConfig<F>,
    ) -> AudioSampleResult<Array2<Complex<F>>>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>;

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
    fn inverse_constant_q_transform<F>(
        cqt_matrix: &Array2<Complex<F>>,
        config: &CqtConfig<F>,
        signal_length: usize,
        sample_rate: F,
    ) -> AudioSampleResult<Self>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>,
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
    /// Complex-valued CQT spectrogram as `Array2<Complex<F>>` with dimensions
    /// `(num_bins, num_frames)` where num_frames depends on signal length and hop_size
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = CqtConfig::chord_detection();
    /// let hop_size = 512;
    /// let cqt_spectrogram = audio.cqt_spectrogram(&config, hop_size, None)?;
    /// ```
    fn cqt_spectrogram<F>(
        &self,
        config: &CqtConfig<F>,
        hop_size: usize,
        window_size: Option<usize>,
    ) -> AudioSampleResult<Array2<Complex<F>>>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>;

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
    /// Real-valued magnitude CQT spectrogram as `Array2<F>` with dimensions
    /// `(num_bins, num_frames)`
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = CqtConfig::musical();
    /// let hop_size = 512;
    /// let magnitude_spectrogram = audio.cqt_magnitude_spectrogram(&config, hop_size, None, false)?;
    /// ```
    fn cqt_magnitude_spectrogram<F>(
        &self,
        config: &CqtConfig<F>,
        hop_size: usize,
        window_size: Option<usize>,
        power: bool,
    ) -> AudioSampleResult<Array2<F>>
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>;

    /// Separate a complex-valued spectrogram D into its magnitude (S) and phase (P) components, so that D = S * P.
    fn magphase<F>(
        complex_spect: &Array2<Complex<F>>,
        power: Option<NonZeroUsize>,
    ) -> (Array2<F>, Array2<Complex<F>>)
    where
        F: RealFloat + FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>,
    {
        // Magnitude: elementwise absolute value

        let mut mag = complex_spect.mapv(|x| x.norm());

        // zeros_to_ones: 1.0 where mag == 0, else 0.0
        let zeros_to_ones = mag.mapv(|x| if x == F::zero() { F::one() } else { F::zero() });

        // mag_nonzero = mag + zeros_to_ones
        let mag_nonzero = &mag + &zeros_to_ones;

        // Compute phase = D / mag_nonzero, but handle zeros separately
        let mut phase = complex_spect.clone();

        let power = match power {
            Some(p) => crate::to_precision(p.get() as f64),
            None => F::one(),
        };

        // Perform elementwise division for real and imaginary parts
        Zip::from(&mut phase)
            .and(&mag_nonzero)
            .and(&zeros_to_ones)
            .for_each(|p, &m_nz, &z| {
                let div = Complex {
                    re: p.re / m_nz + z, // add 1.0 if originally zero
                    im: p.im / m_nz,
                };
                *p = div;
            });

        // Raise magnitude to the given power
        mag.mapv_inplace(|x| x.powf(power));

        (mag, phase)
    }
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
    for<'a> AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
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
    fn detect_pitch_yin<F>(
        &self,
        threshold: F,
        min_frequency: F,
        max_frequency: F,
    ) -> AudioSampleResult<Option<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;

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
    fn detect_pitch_autocorr<F>(
        &self,
        min_frequency: F,
        max_frequency: F,
    ) -> AudioSampleResult<Option<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;

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
    fn track_pitch<F>(
        &self,
        window_size: usize,
        hop_size: usize,
        method: PitchDetectionMethod,
        threshold: F,
        min_frequency: F,
        max_frequency: F,
    ) -> AudioSampleResult<Vec<(F, Option<F>)>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;

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
    fn harmonic_to_noise_ratio<F>(
        &self,
        fundamental_freq: F,
        num_harmonics: usize,
    ) -> AudioSampleResult<F>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;

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
    fn harmonic_analysis<F>(
        &self,
        fundamental_freq: F,
        num_harmonics: usize,
        tolerance: F,
    ) -> AudioSampleResult<Vec<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;

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
    fn estimate_key<F>(&self, window_size: usize, hop_size: usize) -> AudioSampleResult<(usize, F)>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;
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
    for<'a> AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
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
    fn apply_iir_filter<F>(
        &mut self,
        design: &IirFilterDesign<F>,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>;

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
    fn butterworth_lowpass<F>(
        &mut self,
        order: usize,
        cutoff_frequency: F,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>;

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
    fn butterworth_highpass<F>(
        &mut self,
        order: usize,
        cutoff_frequency: F,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>;

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
    fn butterworth_bandpass<F>(
        &mut self,
        order: usize,
        low_frequency: F,
        high_frequency: F,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>;

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
    fn chebyshev_i<F>(
        &mut self,
        order: usize,
        cutoff_frequency: F,
        passband_ripple: F,
        sample_rate: F,
        response: FilterResponse,
    ) -> AudioSampleResult<()>
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>;

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
    fn frequency_response<F>(
        &self,
        frequencies: &[F],
        sample_rate: F,
    ) -> AudioSampleResult<(Vec<F>, Vec<F>)>
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>;
}

/// Parametric equalization operations.
///
/// This trait provides methods for applying parametric EQ to audio signals.
/// Parametric EQ allows precise frequency shaping with adjustable frequency,
/// gain, and Q (bandwidth) parameters for each band.
pub trait AudioParametricEq<'a, T: AudioSample>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'b> AudioSamples<'b, T>: AudioTypeConversion<'b, T> + AudioChannelOps<T>,
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
    fn apply_parametric_eq<F>(
        &mut self,
        eq: &ParametricEq<F>,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>;

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
    fn apply_eq_band<F>(&mut self, band: &EqBand<F>, sample_rate: F) -> AudioSampleResult<()>
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>;

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
    fn apply_peak_filter<F>(
        &mut self,
        frequency: F,
        gain_db: F,
        q_factor: F,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>;

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
    fn apply_low_shelf<F>(
        &mut self,
        frequency: F,
        gain_db: F,
        q_factor: F,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>;

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
    fn apply_high_shelf<F>(
        &mut self,
        frequency: F,
        gain_db: F,
        q_factor: F,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>;

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
    fn apply_three_band_eq<F>(
        &mut self,
        low_freq: F,
        low_gain: F,
        mid_freq: F,
        mid_gain: F,
        mid_q: F,
        high_freq: F,
        high_gain: F,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>;

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
    fn eq_frequency_response<F>(
        &self,
        eq: &ParametricEq<F>,
        frequencies: &[F],
        sample_rate: F,
    ) -> AudioSampleResult<(Vec<F>, Vec<F>)>
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>;
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
    for<'a> AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
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
    fn apply_compressor<F>(
        &mut self,
        config: &CompressorConfig<F>,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;

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
    fn apply_limiter<F>(
        &mut self,
        config: &LimiterConfig<F>,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;

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
    fn apply_compressor_sidechain<F>(
        &mut self,
        config: &CompressorConfig<F>,
        sidechain_signal: &Self,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
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
    fn apply_limiter_sidechain<F>(
        &mut self,
        config: &LimiterConfig<F>,
        sidechain_signal: &Self,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
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
    fn get_compression_curve<F>(
        &self,
        config: &CompressorConfig<F>,
        input_levels_db: &[F],
        sample_rate: F,
    ) -> AudioSampleResult<Vec<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;

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
    fn get_gain_reduction<F>(
        &self,
        config: &CompressorConfig<F>,
        sample_rate: F,
    ) -> AudioSampleResult<Vec<F>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;

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
    fn apply_gate<F>(
        &mut self,
        threshold_db: F,
        ratio: F,
        attack_ms: F,
        release_ms: F,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;

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
    fn apply_expander<F>(
        &mut self,
        threshold_db: F,
        ratio: F,
        attack_ms: F,
        release_ms: F,
        sample_rate: F,
    ) -> AudioSampleResult<()>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>;
}

/// Time-domain editing and manipulation operations.
///
/// This trait provides methods for cutting, pasting, mixing, and modifying
/// audio samples in the time domain. Most operations create new AudioSamples
/// instances rather than modifying in-place to preserve the original data.
pub trait AudioEditing<'a, T: AudioSample>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'b> AudioSamples<'b, T>: AudioTypeConversion<'b, T>,
{
    /// Reverses the order of audio samples.
    ///
    /// Creates a new AudioSamples instance with time-reversed content.
    fn reverse<'b>(&self) -> AudioSamples<'b, T>
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
    fn trim<'b, F: RealFloat>(
        &self,
        start_seconds: F,
        end_seconds: F,
    ) -> AudioSampleResult<AudioSamples<'b, T>>;

    /// Adds padding/silence to the beginning and/or end of the audio.
    ///
    /// # Arguments
    /// * `pad_start_seconds` - Duration to pad at the beginning
    /// * `pad_end_seconds` - Duration to pad at the end
    /// * `pad_value` - Value to use for padding (typically zero for silence)
    fn pad<'b, F: RealFloat>(
        &self,
        pad_start_seconds: F,
        pad_end_seconds: F,
        pad_value: T,
    ) -> AudioSampleResult<AudioSamples<'b, T>>;

    /// Pad samples to the right to reach a target number of samples.
    fn pad_samples_right<'b>(
        &self,
        target_num_samples: usize,
        pad_value: T,
    ) -> AudioSampleResult<AudioSamples<'b, T>>;

    /// Pad audio to a target duration.
    fn pad_to_duration<'b, F: RealFloat>(
        &self,
        target_duration_seconds: F,
        pad_value: T,
        pad_side: PadSide,
    ) -> AudioSampleResult<AudioSamples<'b, T>>;

    /// Splits audio into segments of specified duration.
    ///
    /// The last segment may be shorter if the audio doesn't divide evenly.
    ///
    /// # Arguments
    /// * `segment_duration_seconds` - Duration of each segment
    fn split<F: RealFloat>(
        &self,
        segment_duration_seconds: F,
    ) -> AudioSampleResult<Vec<AudioSamples<'static, T>>>;

    /// Concatenates multiple possible borrowed, audio segments into one.
    /// Concatenates multiple possible borrowed, audio segments into one.
    ///
    /// All segments must have the same sample rate and channel configuration.
    ///
    /// # Arguments
    /// * `segments` - Audio segments to concatenate in order
    fn concatenate<'b>(
        segments: &'b [AudioSamples<'b, T>],
    ) -> AudioSampleResult<AudioSamples<'b, T>>
    where
        'b: 'a,
        Self: Sized;

    /// Concatenates multiple owned audio segments into one.
    fn concatenate_owned<'b>(
        segments: Vec<AudioSamples<'_, T>>,
    ) -> AudioSampleResult<AudioSamples<'b, T>>
    where
        'b: 'a,
        Self: Sized;

    /// Mixes multiple audio sources together.
    ///
    /// Sources must have the same sample rate, channel count, and length.
    /// Optional weights can be provided for each source.
    ///
    /// # Arguments
    /// * `sources` - Audio sources to mix
    /// * `weights` - Optional mixing weights (defaults to equal weighting)
    fn mix<F>(
        sources: &[Self],
        weights: Option<&[F]>,
    ) -> AudioSampleResult<AudioSamples<'static, T>>
    where
        F: RealFloat + ConvertTo<T>,
        T: ConvertTo<F>,
        Self: Sized;

    /// Applies fade-in envelope over specified duration.
    ///
    /// # Arguments
    /// * `duration_seconds` - Duration of the fade-in
    /// * `curve` - Shape of the fade curve
    fn fade_in<F>(&mut self, duration_seconds: F, curve: FadeCurve<F>) -> AudioSampleResult<()>
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>;

    /// Applies fade-out envelope over specified duration.
    ///
    /// # Arguments
    /// * `duration_seconds` - Duration of the fade-out
    /// * `curve` - Shape of the fade curve
    fn fade_out<F>(&mut self, duration_seconds: F, curve: FadeCurve<F>) -> AudioSampleResult<()>
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>;

    /// Repeats the audio signal a specified number of times.
    ///
    /// # Arguments
    /// * `count` - Number of repetitions (total length = original × count)
    fn repeat(&self, count: usize) -> AudioSampleResult<AudioSamples<'static, T>>;

    /// Crops audio to remove silence from beginning and end.
    ///
    /// # Arguments
    /// * `threshold_db` - db threshold below which samples are considered silence
    ///
    /// # Returns
    /// A new AudioSamples instance with leading and trailing silence removed
    /// # Errors
    /// Returns an error if the operation fails for any reason
    fn trim_silence<F>(&self, threshold_db: F) -> AudioSampleResult<AudioSamples<'static, T>>
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>;

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
    #[cfg(feature = "random-generation")]
    #[cfg(feature = "random-generation")]
    fn perturb<'b, F>(
        &self,
        config: &PerturbationConfig<F>,
    ) -> AudioSampleResult<AudioSamples<'b, T>>
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
        StandardUniform: Distribution<F>;

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
    #[cfg(feature = "random-generation")]
    #[cfg(feature = "random-generation")]
    fn perturb_<F>(&mut self, config: &PerturbationConfig<F>) -> AudioSampleResult<()>
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
        StandardUniform: Distribution<F>;

    /// Stacks multiple mono audio samples into a multi-channel audio sample.
    fn stack(sources: &[Self]) -> AudioSampleResult<AudioSamples<'static, T>>
    where
        Self: Sized;

    /// Trim silence anywhere in the audio.
    fn trim_all_silence<F>(
        &self,
        threshold_db: F,
        min_silence_duration_seconds: F,
    ) -> AudioSampleResult<AudioSamples<'static, T>>
    where
        T: ConvertTo<F>,
        F: RealFloat + ConvertTo<T>;
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
    for<'a> AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
{
    /// Converts multi-channel audio to mono using specified method.
    ///
    /// # Arguments
    /// * `method` - Method for combining channels into mono
    fn to_mono<F>(
        &self,
        method: MonoConversionMethod<F>,
    ) -> AudioSampleResult<AudioSamples<'static, T>>
    where
        T: CastFrom<F> + ConvertTo<F>,
        F: RealFloat + ConvertTo<T>;

    /// Converts mono audio to stereo using specified method.
    ///
    /// # Arguments
    /// * `method` - Method for creating stereo from mono
    fn to_stereo<F>(
        &self,
        method: StereoConversionMethod<F>,
    ) -> AudioSampleResult<AudioSamples<'static, T>>
    where
        T: CastFrom<F> + ConvertTo<F>,
        F: RealFloat + CastInto<T> + ConvertTo<T>;

    /// Duplicates mono audio to N channels.
    ///
    /// This is a convenience method for creating multi-channel audio from mono
    /// by copying the same signal to all channels.
    ///
    /// # Arguments
    /// * `n_channels` - Number of output channels (must be >= 1)
    ///
    /// # Returns
    /// Multi-channel audio with the mono signal duplicated to all channels.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The input is not mono
    /// - `n_channels` is 0
    ///
    /// # Examples
    /// ```rust,no_run
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::AudioChannelOps;
    /// use ndarray::Array1;
    ///
    /// let mono = AudioSamples::new_mono(Array1::from(vec![0.1f32, 0.2, 0.3]), sample_rate!(44100));
    /// let stereo = mono.duplicate_to_channels(2).unwrap();
    /// assert_eq!(stereo.num_channels(), 2);
    ///
    /// let surround = mono.duplicate_to_channels(6).unwrap(); // 5.1 surround
    /// assert_eq!(surround.num_channels(), 6);
    /// ```
    fn duplicate_to_channels(
        &self,
        n_channels: usize,
    ) -> AudioSampleResult<AudioSamples<'static, T>>;

    /// Extracts a specific channel from multi-channel audio.
    ///
    /// # Arguments
    /// * `channel_index` - Zero-based index of channel to extract
    fn extract_channel(&self, channel_index: usize) -> AudioSampleResult<AudioSamples<'static, T>>;

    /// Borrows a specific channel from multi-channel audio.
    ///
    /// # Arguments
    /// * `channel_index` - Zero-based index of channel to borrow
    ///
    /// # Returns
    /// AudioSamples with a borrowed representation of the specified channel from self.
    fn borrow_channel(&self, channel_index: usize) -> AudioSampleResult<AudioSamples<'_, T>>;

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
    fn pan<F>(&mut self, pan_value: F) -> AudioSampleResult<()>
    where
        T: CastFrom<F> + ConvertTo<F>,
        F: RealFloat + CastInto<T> + ConvertTo<T>;

    /// Adjusts balance between left and right channels.
    ///
    /// # Arguments
    /// * `balance` - Balance adjustment (-1.0 = left only, 0.0 = equal, 1.0 = right only)
    fn balance<F>(&mut self, balance: F) -> AudioSampleResult<()>
    where
        T: CastFrom<F> + ConvertTo<F>,
        F: RealFloat + CastInto<T> + ConvertTo<T>;

    /// Apply a function to a specific channel.
    fn apply_to_channel<F>(&mut self, channel_index: usize, func: F) -> AudioSampleResult<()>
    where
        F: FnMut(T) -> T,
        Self: Sized;

    /// Interleave multiple channels into one audio sample.
    fn interleave_channels(
        channels: &[AudioSamples<'_, T>],
    ) -> AudioSampleResult<AudioSamples<'static, T>>;

    /// Deinterleave audio into separate channel samples.
    fn deinterleave_channels(&self) -> AudioSampleResult<Vec<AudioSamples<'static, T>>>;
}

/// Utilities for plotting audio data.
#[cfg(feature = "plotting")]
pub trait AudioPlottingUtils<T: AudioSample>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
{
    /// Generate time axis values for plotting.
    fn time_axis<F>(&self, step: Option<F>) -> Vec<F>;
    /// Seconds from 0 to duration with ~target_ticks "nice" spacing (1–2–5).
    fn time_ticks_seconds<F>(&self, target_ticks: usize) -> Vec<F>;
    /// Generate frequency axis values for frequency domain plotting.
    fn frequency_axis(&self) -> Vec<T>;
    /// Create a quick analysis plot with waveform and spectrum
    #[cfg(feature = "spectral-analysis")]
    fn quick_plot<F, P: AsRef<Path>>(&self, save_to: Option<P>) -> super::PlotResult<()>
    where
        Self: AudioPlotBuilders<T>,
        F: RealFloat + ConvertTo<T>,
        T: CastFrom<F> + ConvertTo<F>,
    {
        use std::path::PathBuf;

        use crate::operations::{LayoutConfig, PlotComposer};

        let waveform = self.waveform_plot::<F>(None)?;
        let spectrum = self.power_spectrum_plot::<F>(None, None, None, None, None)?;

        let plot = PlotComposer::new()
            .add_element(waveform)
            .add_element(spectrum)
            .with_layout(LayoutConfig::VerticalStack)
            .with_title("Audio Analysis");

        match save_to {
            None => plot.show(true),
            Some(p) => {
                let out_path: PathBuf = p.as_ref().into();
                plot.render_to_file(out_path, true)
            }
        }
    }
}

/// Serialization and deserialization operations for audio samples.
///
/// This trait provides methods for saving and loading AudioSamples to/from
/// various file formats including text-based formats (CSV, JSON, TXT),
/// binary formats (NumPy, MessagePack, CBOR), and compressed formats.
///
/// The focus is on data analysis and interchange formats rather than
/// traditional audio file formats like WAV or MP3. This enables seamless
/// integration with data science workflows, Python NumPy/SciPy, and other
/// audio analysis tools.
///
/// # Format Support
///
/// - **Text formats**: CSV, JSON, plain text with configurable delimiters
/// - **Binary formats**: NumPy (.npy), MessagePack, CBOR, custom binary
/// - **Compressed formats**: NumPy compressed (.npz), gzip compression
/// - **Metadata preservation**: Sample rate, channel information, custom attributes
///
/// # Example Usage
///
/// ```rust,ignore
/// use audio_samples::{AudioSamples, operations::*};
/// use audio_samples::operations::types::{SerializationFormat, SerializationConfig};
///
/// let audio = AudioSamples::new_mono(samples, 44100);
///
/// // Save to JSON with metadata
/// audio.save_to_file("audio_data.json")?;
///
/// // Save to NumPy format
/// let config = SerializationConfig::numpy();
/// audio.save_with_config("audio_data.npy", &config)?;
///
/// // Load from file with automatic format detection
/// let loaded = AudioSamples::load_from_file("audio_data.json")?;
/// ```
#[cfg(feature = "serialization")]
pub trait AudioSamplesSerialise<T: AudioSample>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'b> AudioSamples<'b, T>: AudioTypeConversion<'b, T>,
{
    /// Save audio samples to a file with automatic format detection from extension.
    ///
    /// The file format is automatically determined based on the file extension:
    /// - `.json` → JSON format with metadata
    /// - `.csv` → CSV format with headers
    /// - `.npy` → NumPy binary format
    /// - `.npz` → NumPy compressed format
    /// - `.txt` → Plain text with space delimiters
    /// - `.bin` → Custom binary format
    /// - `.msgpack` → MessagePack format
    /// - `.cbor` → CBOR format
    ///
    /// # Arguments
    /// * `path` - File path with extension indicating desired format
    ///
    /// # Returns
    /// Result indicating success or failure
    ///
    /// # Errors
    /// - `SerializationError::UnsupportedFormat` if extension is not recognized
    /// - `SerializationError::IoError` for file I/O failures
    /// - `SerializationError::SerializationFailed` for format-specific errors
    ///
    /// # Example
    /// ```rust,ignore
    /// audio.save_to_file("data/audio_analysis.json")?;
    /// audio.save_to_file("output.npy")?;
    /// ```
    fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> AudioSampleResult<()>;

    /// Save audio samples with explicit format configuration.
    ///
    /// Provides fine-grained control over the serialization process including
    /// precision, compression, metadata inclusion, and format-specific options.
    ///
    /// # Arguments
    /// * `path` - Output file path
    /// * `config` - Serialization configuration specifying format and options
    ///
    /// # Returns
    /// Result indicating success or failure
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = SerializationConfig::json()
    ///     .with_precision(6)
    ///     .with_metadata(true);
    /// audio.save_with_config("analysis.json", &config)?;
    /// ```
    fn save_with_config<P: AsRef<std::path::Path>>(
        &self,
        path: P,
        config: &crate::operations::types::SerializationConfig<f64>,
    ) -> AudioSampleResult<()>;

    /// Load audio samples from a file with automatic format detection.
    ///
    /// Attempts to detect the file format using:
    /// 1. File extension hints
    /// 2. Magic number/header detection
    /// 3. Content analysis for text formats
    ///
    /// # Arguments
    /// * `path` - Input file path
    ///
    /// # Returns
    /// New AudioSamples instance loaded from the file
    ///
    /// # Errors
    /// - `SerializationError::FormatDetectionFailed` if format cannot be determined
    /// - `SerializationError::IoError` for file I/O failures
    /// - `SerializationError::DeserializationFailed` for parsing errors
    /// - `SerializationError::InvalidHeader` for corrupted files
    ///
    /// # Example
    /// ```rust,ignore
    /// let audio: AudioSamples<f32> = AudioSamples::load_from_file("data.json")?;
    /// ```
    fn load_from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> AudioSampleResult<AudioSamples<'static, T>>;

    /// Load audio samples with explicit format configuration.
    ///
    /// Uses the specified configuration to parse the file, bypassing automatic
    /// format detection. Useful when the automatic detection fails or when
    /// specific parsing options are required.
    ///
    /// # Arguments
    /// * `path` - Input file path
    /// * `config` - Deserialization configuration specifying format and options
    ///
    /// # Returns
    /// New AudioSamples instance loaded from the file
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = SerializationConfig::csv();
    /// let audio = AudioSamples::load_with_config("data.csv", &config)?;
    /// ```
    fn load_with_config<P: AsRef<std::path::Path>>(
        path: P,
        config: &crate::operations::types::SerializationConfig<f64>,
    ) -> AudioSampleResult<AudioSamples<'static, T>>;

    /// Serialize audio samples to bytes in memory.
    ///
    /// Converts the audio data to the specified format and returns the
    /// serialized bytes. Useful for network transmission, caching, or
    /// embedding in other data structures.
    ///
    /// # Arguments
    /// * `format` - Target serialization format
    ///
    /// # Returns
    /// Vector of bytes containing the serialized audio data
    ///
    /// # Example
    /// ```rust,ignore
    /// let json_bytes = audio.serialize_to_bytes(SerializationFormat::Json)?;
    /// let numpy_bytes = audio.serialize_to_bytes(SerializationFormat::Numpy)?;
    /// ```
    fn serialize_to_bytes(
        &self,
        format: crate::operations::types::SerializationFormat,
    ) -> AudioSampleResult<Vec<u8>>;

    /// Deserialize audio samples from bytes in memory.
    ///
    /// Parses audio data from the provided byte array using the specified format.
    /// The inverse operation of `serialize_to_bytes`.
    ///
    /// # Arguments
    /// * `data` - Byte array containing serialized audio data
    /// * `format` - Format of the serialized data
    ///
    /// # Returns
    /// New AudioSamples instance deserialized from the bytes
    ///
    /// # Example
    /// ```rust,ignore
    /// let audio = AudioSamples::deserialize_from_bytes(&bytes, SerializationFormat::Json)?;
    /// ```
    fn deserialize_from_bytes(
        data: &[u8],
        format: crate::operations::types::SerializationFormat,
    ) -> AudioSampleResult<AudioSamples<'static, T>>;

    /// Get an estimate of the serialized size for a format.
    ///
    /// Provides an approximation of how many bytes the audio data will
    /// occupy when serialized to the specified format. Useful for memory
    /// planning and storage estimation.
    ///
    /// # Arguments
    /// * `format` - Target serialization format
    ///
    /// # Returns
    /// Estimated size in bytes
    ///
    /// # Note
    /// The estimate may not be exact, especially for compressed formats
    /// where the final size depends on data compressibility.
    fn estimate_serialized_size(
        &self,
        format: crate::operations::types::SerializationFormat,
    ) -> AudioSampleResult<usize>;

    /// List all supported formats for serialization.
    ///
    /// Returns a vector of all serialization formats that are available
    /// for saving audio data. The availability depends on enabled cargo features.
    ///
    /// # Returns
    /// Vector of supported serialization formats
    fn supported_serialization_formats() -> Vec<crate::operations::types::SerializationFormat>;

    /// List all supported formats for deserialization.
    ///
    /// Returns a vector of all formats that can be loaded. Usually identical
    /// to serialization formats, but may differ if some formats are write-only.
    ///
    /// # Returns
    /// Vector of supported deserialization formats
    fn supported_deserialization_formats() -> Vec<crate::operations::types::SerializationFormat>;

    /// Auto-detect format from file extension or magic bytes.
    ///
    /// Analyzes the file to determine its format using:
    /// 1. File extension mapping
    /// 2. Magic number detection in file headers
    /// 3. Content structure analysis for text formats
    ///
    /// # Arguments
    /// * `path` - File path to analyze
    ///
    /// # Returns
    /// Detected serialization format
    ///
    /// # Errors
    /// - `SerializationError::FormatDetectionFailed` if format cannot be determined
    /// - `SerializationError::IoError` if file cannot be accessed
    ///
    /// # Example
    /// ```rust,ignore
    /// let format = AudioSamples::<f32>::detect_format("data.json")?;
    /// assert_eq!(format, SerializationFormat::Json);
    /// ```
    fn detect_format<P: AsRef<std::path::Path>>(
        path: P,
    ) -> AudioSampleResult<crate::operations::types::SerializationFormat>;

    /// Validate serialization round-trip accuracy.
    ///
    /// Serializes the audio data to the specified format, then deserializes
    /// it back and compares with the original. Useful for testing data integrity
    /// and format compatibility.
    ///
    /// # Arguments
    /// * `format` - Format to test
    /// * `tolerance` - Acceptable difference threshold for floating-point comparison
    ///
    /// # Returns
    /// Result indicating whether round-trip was successful within tolerance
    ///
    /// # Example
    /// ```rust,ignore
    /// audio.validate_round_trip(SerializationFormat::Json, 1e-6)?;
    /// ```
    fn validate_round_trip(
        &self,
        format: crate::operations::types::SerializationFormat,
        tolerance: f64,
    ) -> AudioSampleResult<()>;

    /// Export metadata as a separate JSON file.
    ///
    /// Saves audio metadata (sample rate, channel count, duration, custom attributes)
    /// to a JSON file alongside the main audio data. Useful when using binary
    /// formats that don't support metadata natively.
    ///
    /// # Arguments
    /// * `path` - Output file path for metadata JSON
    ///
    /// # Returns
    /// Result indicating success or failure
    ///
    /// # Example
    /// ```rust,ignore
    /// audio.save_to_file("data.npy")?;
    /// audio.export_metadata("data_metadata.json")?;
    /// ```
    fn export_metadata<P: AsRef<std::path::Path>>(&self, path: P) -> AudioSampleResult<()>;

    /// Import metadata from a JSON file.
    ///
    /// Loads metadata from a JSON file and applies it to the current AudioSamples
    /// instance. Useful when loading binary formats that don't preserve metadata.
    ///
    /// # Arguments
    /// * `path` - Input file path for metadata JSON
    ///
    /// # Returns
    /// Result indicating success or failure
    ///
    /// # Example
    /// ```rust,ignore
    /// let mut audio = AudioSamples::load_from_file("data.npy")?;
    /// audio.import_metadata("data_metadata.json")?;
    /// ```
    fn import_metadata<P: AsRef<std::path::Path>>(&mut self, path: P) -> AudioSampleResult<()>;
}

/// Audio decomposition operations for separating signals into components.
///
/// This trait provides methods for separating audio signals into different
/// components based on their spectral or temporal characteristics. These
/// decomposition techniques are fundamental in music information retrieval,
/// audio analysis, and preprocessing for machine learning applications.
///
/// # Available Decomposition Methods
///
/// - **HPSS (Harmonic/Percussive Source Separation)**: Separates audio into
///   harmonic (tonal, sustained) and percussive (transient, attack) components
///   using STFT magnitude median filtering.
///
/// # Example Usage
///
/// ```rust,ignore
/// use audio_samples::{AudioSamples, operations::AudioDecomposition};
/// use audio_samples::operations::types::HpssConfig;
/// use ndarray::array;
///
/// let audio = AudioSamples::new_mono(samples, 44100);
/// let config = HpssConfig::new();
///
/// // Separate into harmonic and percussive components
/// let (harmonic, percussive) = audio.hpss(&config)?;
///
/// // Process components separately
/// let drums_isolated = percussive.normalize(-1.0, 1.0, NormalizationMethod::Peak)?;
/// let melody_isolated = harmonic.normalize(-1.0, 1.0, NormalizationMethod::Peak)?;
/// ```
#[cfg(feature = "hpss")]
pub trait AudioDecomposition<T: AudioSample>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
{
    /// Separate audio into harmonic and percussive components using HPSS.
    ///
    /// Harmonic/Percussive Source Separation (HPSS) uses Short-Time Fourier Transform
    /// (STFT) magnitude median filtering to separate audio signals based on their
    /// spectral characteristics:
    ///
    /// - **Harmonic components**: Sustained tonal content (vocals, sustained instruments)
    /// - **Percussive components**: Transient content (drums, attacks, onsets)
    ///
    /// The algorithm works by:
    /// 1. Computing the STFT magnitude spectrogram
    /// 2. Applying median filtering along time axis (enhances harmonic content)
    /// 3. Applying median filtering along frequency axis (enhances percussive content)
    /// 4. Creating separation masks based on the filtered spectrograms
    /// 5. Reconstructing time-domain signals using inverse STFT
    ///
    /// # Arguments
    ///
    /// * `config` - HPSS configuration parameters controlling window size,
    ///   hop size, median filter sizes, and mask softness
    ///
    /// # Returns
    ///
    /// A tuple containing `(harmonic_component, percussive_component)` as separate
    /// AudioSamples instances with the same sample rate as the input.
    ///
    /// # Errors
    ///
    /// - `AudioSampleError::Parameter` if configuration parameters are invalid
    /// - `AudioSampleError::Parameter` if signal is too short for the specified window size
    /// - `AudioSampleError::Layout` if STFT/ISTFT operations fail
    ///
    /// # Performance Notes
    ///
    /// - Computational complexity is O(N log N) due to FFT operations
    /// - Memory usage scales with window size and signal length
    /// - Consider using smaller window/hop sizes for real-time applications
    /// - Enable `parallel-processing` feature for multi-threaded acceleration
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use audio_samples::operations::{AudioDecomposition, HpssConfig};
    ///
    /// let config = HpssConfig::musical(); // Optimized for musical content
    /// let (harmonic, percussive) = audio.hpss(&config)?;
    ///
    /// // Harmonic component contains vocals, sustained instruments
    /// // Percussive component contains drums, attacks, transients
    /// ```
    ///
    /// # References
    ///
    /// - Fitzgerald, D. (2010). "Harmonic/percussive separation using median filtering"
    /// - Müller, M. (2015). "Fundamentals of Music Processing", Section 8.4
    fn hpss<F: RealFloat>(
        &self,
        config: &crate::operations::types::HpssConfig<F>,
    ) -> AudioSampleResult<(AudioSamples<'static, T>, AudioSamples<'static, T>)>
    where
        F: FftNum + AudioSample + ConvertTo<T>,
        T: ConvertTo<F>,
        for<'a> AudioSamples<'a, T>: AudioTypeConversion<'a, T> + AudioTransforms<T>;
}
