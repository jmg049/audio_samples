//! Core trait definitions for audio processing operations.
//!
//! This module declares the public trait surface of the `audio_samples` operation layer.
//! Every capability is expressed as a separate, feature-gated trait that is implemented
//! for [`AudioSamples`]:
//!
//! | Trait | Feature | Responsibility |
//! |---|---|---|
//! | [`AudioStatistics`] | `statistics` | Peak, RMS, mean, variance, spectral centroid, … |
//! | [`AudioProcessing`] | `processing` | Normalize, filter, resample, μ-law, … |
//! | [`AudioTransforms`] | `transforms` | FFT, STFT, spectrograms, MFCCs, chromagram, … |
//! | [`AudioEditing`] | `editing` | Trim, pad, fade, concatenate, mix, … |
//! | [`AudioChannelOps`] | `channels` | Mono/stereo conversion, panning, interleave, … |
//! | [`AudioEnvelopes`] | `envelopes` | Amplitude, RMS, attack/decay, analytic envelope, … |
//! | [`AudioDynamicRange`] | `dynamic-range` | Compressor, limiter, gate, expander |
//! | [`AudioIirFiltering`] | `iir-filtering` | Butterworth, Chebyshev IIR filters |
//! | [`AudioParametricEq`] | `parametric-eq` | Multi-band parametric EQ |
//! | [`AudioPitchAnalysis`] | `pitch-analysis` | YIN, autocorrelation, harmonic analysis |
//! | [`AudioVoiceActivityDetection`] | `vad` | Frame-based speech / silence classification |
//! | [`AudioDecomposition`] | `decomposition` | Harmonic–percussive source separation (HPSS) |
//! | [`AudioOnsetDetection`] | `onset-detection` | Onset times, spectral flux, complex ODF |
//! | [`AudioBeatTracking`] | `beat-tracking` | Tempo-aware beat detection |
//! | [`AudioPlotting`](crate::operations::AudioPlotting) | `plotting` | Waveform, spectrogram, magnitude-spectrum plots |
//!
//! Grouping operations into separate traits keeps compile times low — only the code
//! required for the enabled features is compiled — while providing a clean extension
//! point for new capabilities.  Each trait is independently usable: bring only the
//! traits you need into scope without paying for the rest.
//!
//! ## Usage
//!
//! Bring the relevant trait into scope with a `use` statement, then call its methods
//! on any [`AudioSamples`] instance:
//!
//! ```rust
//! use audio_samples::{AudioSamples, sample_rate};
//! use ndarray::array;
//!
//! let audio = AudioSamples::new_mono(
//!     array![1.0f32, -0.5, 0.25, -0.1],
//!     sample_rate!(44100),
//! ).unwrap();
//!
//! assert_eq!(audio.samples_per_channel().get(), 4);
//! assert_eq!(audio.sample_rate().get(), 44100);
//! ```

#[cfg(feature = "envelopes")]
use crate::{NdResult, operations::dynamic_range::EnvelopeFollower};

#[cfg(feature = "envelopes")]
use crate::operations::types::DynamicRangeMethod;

#[cfg(feature = "editing")]
use crate::operations::types::{FadeCurve, PadSide};

#[cfg(feature = "iir-filtering")]
use crate::AudioSample;

#[cfg(all(
    feature = "editing",
    feature = "random-generation",
    feature = "iir-filtering"
))]
use crate::operations::types::PerturbationConfig;

#[cfg(feature = "channels")]
use crate::operations::types::{MonoConversionMethod, StereoConversionMethod};

#[cfg(feature = "processing")]
use crate::operations::types::NormalizationConfig;

#[cfg(all(feature = "processing", feature = "resampling"))]
use crate::{operations::types::ResamplingQuality, repr::SampleRate};

#[cfg(feature = "decomposition")]
use crate::operations::hpss::HpssConfig;

#[cfg(feature = "onset-detection")]
use crate::operations::onset::{
    ComplexOnsetConfig, OnsetDetectionConfig, SpectralFluxConfig, SpectralFluxMethod,
};

#[cfg(feature = "parametric-eq")]
use crate::operations::types::{EqBand, ParametricEq};

#[cfg(feature = "beat-tracking")]
use crate::operations::beat::{BeatTrackingConfig, BeatTrackingData};

#[cfg(feature = "vad")]
use crate::operations::types::VadConfig;

#[cfg(feature = "dynamic-range")]
use crate::operations::types::{CompressorConfig, LimiterConfig};

#[cfg(feature = "pitch-analysis")]
use crate::operations::types::PitchDetectionMethod;

#[cfg(feature = "iir-filtering")]
use crate::operations::types::{FilterResponse, IirFilterDesign};

#[cfg(feature = "transforms")]
use spectrograms::{
    AmpScaleSpec, ChromaParams, Chromagram, CqtDbSpectrogram, CqtMagnitudeSpectrogram, CqtParams,
    CqtPowerSpectrogram, CqtResult, CqtSpectrogram, Decibels, Gammatone, GammatoneDbSpectrogram,
    GammatoneMagnitudeSpectrogram, GammatoneParams, GammatonePowerSpectrogram, LinearDbSpectrogram,
    LinearHz, LinearMagnitudeSpectrogram, LinearPowerSpectrogram, LogHz, LogHzDbSpectrogram,
    LogHzMagnitudeSpectrogram, LogHzParams, LogHzPowerSpectrogram, LogMelSpectrogram, LogParams,
    Magnitude, MelMagnitudeSpectrogram, MelParams, MelPowerSpectrogram, MelSpectrogram, Mfcc,
    MfccParams, Power, Spectrogram, SpectrogramParams, StftParams, StftResult,
};

#[cfg(any(feature = "transforms", feature = "onset-detection"))]
use ndarray::{Array2, Zip};
#[cfg(feature = "pitch-analysis")]
use spectrograms::WindowType;

#[cfg(feature = "plotting")]
use crate::operations::{
    WaveformPlot, WaveformPlotParams,
    plotting::spectrograms::{SpectrogramPlot, SpectrogramPlotParams},
    plotting::spectrum::{MagnitudeSpectrumParams, MagnitudeSpectrumPlot},
};
#[cfg(any(feature = "transforms", feature = "onset-detection"))]
use num_complex::Complex;
// "Unused" imports below are required pretty much as soon as any of the traits are implemented, this is cleaner than a huge cfg(any(...)) block.
#[allow(unused_imports)]
use crate::{AudioSampleResult, AudioSamples, AudioTypeConversion, StandardSample};
#[allow(unused_imports)]
use non_empty_slice::{NonEmptySlice, NonEmptyVec};
#[allow(unused_imports)]
use std::num::NonZeroUsize;

/// Statistical analysis operations for audio data.
///
/// # Purpose
///
/// Provides descriptive statistics, energy metrics, and spectral summary measures
/// computed from the sample buffer.  Operations range from simple extrema (peak,
/// min, max) through signal-level descriptors (RMS, variance, zero-crossing rate)
/// to frequency-weighted features (spectral centroid, spectral rolloff).
///
/// # Intended Usage
///
/// Use this trait for analysis, visualisation, and gating decisions — for example
/// to check loudness before normalisation, to estimate the noise floor before gating,
/// or to drive automatic level-matching in a processing pipeline.
///
/// Frequency-domain methods (`spectral_centroid`, `spectral_rolloff`,
/// `autocorrelation`) require `feature = "transforms"`.
///
/// # Invariants
///
/// All methods operate on the full multi-channel buffer unless the documentation
/// explicitly states otherwise (e.g. `median` is mono-only, `spectral_centroid`
/// and `spectral_rolloff` use only the first channel).
#[cfg(feature = "statistics")]
pub trait AudioStatistics: AudioTypeConversion
where
    Self::Sample: StandardSample,
{
    /// Returns the peak (maximum absolute value) across all samples and channels.
    ///
    /// # Returns
    /// The maximum absolute sample value, in the native sample type `T`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -3.0, 2.5, -1.5];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// assert_eq!(audio.peak(), 3.0);
    /// ```
    fn peak(&self) -> Self::Sample;

    /// Returns the peak (maximum absolute value) across all samples and channels.
    ///
    /// Alias for [`peak`][Self::peak]. Provided to match the conventional term
    /// "amplitude" used in some audio contexts.
    ///
    /// # Returns
    /// The maximum absolute sample value as the native sample type `T`.
    #[inline]
    fn amplitude(&self) -> Self::Sample {
        self.peak()
    }

    /// Returns the minimum sample value across all channels.
    ///
    /// # Returns
    /// The smallest sample value found, in the native sample type `T`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -3.0, 2.5, -1.5];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// assert_eq!(audio.min_sample(), -3.0);
    /// ```
    fn min_sample(&self) -> Self::Sample;

    /// Returns the maximum sample value across all channels.
    ///
    /// # Returns
    /// The largest sample value found, in the native sample type `T`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -3.0, 2.5, -1.5];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// assert_eq!(audio.max_sample(), 2.5);
    /// ```
    fn max_sample(&self) -> Self::Sample;

    /// Computes the arithmetic mean of all samples across all channels.
    ///
    /// # Returns
    /// The mean value as `f64`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 2.0, -2.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// assert_eq!(audio.mean(), 0.0);
    /// ```
    fn mean(&self) -> f64;

    /// Returns the value at the temporal midpoint of a mono signal.
    ///
    /// For even-length signals the result is the average of the two samples at
    /// the two central indices. For odd-length signals the single central sample
    /// is returned directly. Samples are selected by index position; the buffer
    /// is not sorted.
    ///
    /// # Returns
    /// `Some(value)` for mono audio, or `None` if the signal is multi-channel.
    ///
    /// # Assumptions
    /// For even-length signals the sum of the two central samples must not
    /// overflow the sample type `T`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, 3.0, 5.0, 7.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Central indices 1 and 2: (3.0 + 5.0) / 2.0 = 4.0
    /// assert_eq!(audio.median(), Some(4.0));
    /// ```
    fn median(&self) -> Option<f64>;

    /// Computes the Root Mean Square (RMS) of all samples across all channels.
    ///
    /// RMS is the square root of the mean of squared sample values. It provides
    /// a measure of the signal's average energy and is commonly used for
    /// loudness estimation.
    ///
    /// # Returns
    /// The RMS value as `f64`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 1.0, -1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let rms = audio.rms();
    /// assert!((rms - 1.0).abs() < 1e-6);
    /// ```
    fn rms(&self) -> f64;

    /// Computes RMS and peak absolute value in a single pass over the data.
    ///
    /// Equivalent to calling [`rms`](Self::rms) and [`peak`](Self::peak)
    /// separately, but reads the sample buffer only once. For large signals
    /// where the data does not fit in CPU cache this is roughly twice as fast
    /// as two separate calls.
    ///
    /// # Returns
    /// A tuple `(rms, peak)` where `rms` is the root-mean-square as `f64`
    /// and `peak` is the maximum absolute sample value as `T`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 1.0, -1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let (rms, peak) = audio.rms_and_peak();
    /// assert!((rms - 1.0).abs() < 1e-6);
    /// assert_eq!(peak, 1.0f32);
    /// ```
    fn rms_and_peak(&self) -> (f64, Self::Sample) {
        (self.rms(), self.peak())
    }

    /// Computes the population variance of the audio samples.
    ///
    /// Variance measures the spread of sample values around the mean.
    /// For mono audio this is the standard population variance of all samples.
    /// For multi-channel audio the variance is computed per sample position
    /// across channels and the results are averaged over time.
    ///
    /// # Returns
    /// The variance as `f64`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, 2.0, 3.0, 4.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let variance = audio.variance();
    /// assert!((variance - 1.25).abs() < 1e-6);
    /// ```
    fn variance(&self) -> f64;

    /// Computes the standard deviation of the audio samples.
    ///
    /// Standard deviation is the square root of [`AudioStatistics::variance`].
    /// It expresses the spread of sample values in the same units as the
    /// original data.
    ///
    /// # Returns
    /// The standard deviation as `f64`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, 2.0, 3.0, 4.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let std_dev = audio.std_dev();
    /// assert!((std_dev - 1.25_f64.sqrt()).abs() < 1e-6);
    /// ```
    fn std_dev(&self) -> f64;

    /// Counts the number of zero crossings in the audio signal.
    ///
    /// Zero crossings occur when the signal changes sign between adjacent samples.
    /// This metric is useful for pitch detection, signal analysis, and estimating
    /// the noisiness of a signal.
    ///
    /// # Returns
    /// The total number of zero crossings across all channels. Returns 0 if the
    /// audio has fewer than 2 samples.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 1.0, -1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// assert_eq!(audio.zero_crossings(), 3);
    /// ```
    fn zero_crossings(&self) -> usize;

    /// Computes the zero crossing rate (crossings per second).
    ///
    /// This normalises the zero crossing count by the signal duration, providing
    /// a frequency-like measure that is independent of signal length.
    ///
    /// # Returns
    /// The number of zero crossings per second as `f64`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 1.0, -1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let zcr = audio.zero_crossing_rate();
    /// assert!(zcr > 0.0);
    /// ```
    fn zero_crossing_rate(&self) -> f64;

    /// Computes cross-correlation with another audio signal.
    ///
    /// Cross-correlation measures the similarity between two signals as a
    /// function of the displacement of one relative to the other. It is useful
    /// for signal alignment, pattern matching, and delay estimation.
    ///
    /// For multi-channel audio only the first channels of both signals are
    /// correlated.
    ///
    /// # Arguments
    /// - `other` — the second audio signal. Must have the same number of
    ///   channels as `self`.
    /// - `max_lag` — the maximum lag offset in samples. The effective maximum
    ///   lag is clamped to `min(len_self, len_other) - 1`.
    ///
    /// # Returns
    /// A [`NonEmptyVec`] of correlation values for lags `0` through the
    /// effective maximum lag.
    ///
    /// # Errors
    /// Returns an error if the two signals have different numbers of channels,
    /// or if one is mono and the other is multi-channel.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    /// use std::num::NonZeroUsize;
    ///
    /// let data1 = array![1.0f32, 0.0, -1.0, 0.0];
    /// let data2 = array![0.0, 1.0, 0.0, -1.0];
    /// let audio1 = AudioSamples::new_mono(data1, sample_rate!(44100)).unwrap();
    /// let audio2 = AudioSamples::new_mono(data2, sample_rate!(44100)).unwrap();
    /// let max_lag = NonZeroUsize::new(3).unwrap();
    /// let xcorr = audio1.cross_correlation(&audio2, max_lag).unwrap();
    /// assert_eq!(xcorr.len(), NonZeroUsize::new(4).unwrap()); // lags 0..=3
    /// ```
    fn cross_correlation(
        &self,
        other: &Self,
        max_lag: NonZeroUsize,
    ) -> AudioSampleResult<NonEmptyVec<f64>>;

    /// Computes the spectral centroid of a mono signal.
    ///
    /// The spectral centroid is the frequency-weighted mean of the power
    /// spectrum and serves as a measure of spectral brightness. Higher values
    /// indicate energy concentrated at higher frequencies.
    ///
    /// Reference: [Spectral centroid — Wikipedia](https://en.wikipedia.org/wiki/Spectral_centroid)
    ///
    /// # Returns
    /// The spectral centroid frequency in Hz. Returns `0.0` when the signal
    /// is silence (zero total spectral energy).
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the signal is multi-channel.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    ///
    /// # Assumptions
    /// The input signal must be mono. Multi-channel signals must be mixed or
    /// channel-selected before calling this method.
    ///
    /// # Examples
    /// ```no_run
    /// // Requires feature = "transforms"
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 1.0);
    /// let centroid = audio.spectral_centroid().unwrap();
    /// // A 440 Hz sine wave should have a centroid near 440 Hz.
    /// assert!(centroid > 0.0);
    /// ```
    #[cfg(feature = "transforms")]
    fn spectral_centroid(&self) -> AudioSampleResult<f64>;

    /// Computes the autocorrelation function up to `max_lag` samples.
    ///
    /// Autocorrelation measures the similarity of a signal with a time-shifted
    /// copy of itself. The value at lag 0 is the signal's mean-square value;
    /// subsequent lags decrease as the shift increases.
    ///
    /// For multi-channel audio only the first channel is used.
    ///
    /// Reference: [Autocorrelation — Wikipedia](https://en.wikipedia.org/wiki/Autocorrelation)
    ///
    /// # Arguments
    /// - `max_lag` — the maximum lag offset in samples. The effective maximum
    ///   lag is clamped to `signal_length - 1`.
    ///
    /// # Returns
    /// A [`NonEmptyVec`] of correlation values for lags `0` through
    /// `min(max_lag, signal_length - 1)`, or `None` if the FFT computation
    /// fails.
    ///
    /// # Examples
    /// ```no_run
    /// // Requires feature = "transforms"
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    /// use std::num::NonZeroUsize;
    ///
    /// let data = array![1.0f32, 0.5, -0.5, -1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let autocorr = audio.autocorrelation(NonZeroUsize::new(3).unwrap()).unwrap();
    /// assert_eq!(autocorr.len(), NonZeroUsize::new(4).unwrap()); // lags 0..=3
    /// ```
    #[cfg(feature = "transforms")]
    fn autocorrelation(&self, max_lag: NonZeroUsize) -> Option<NonEmptyVec<f64>>;

    /// Computes the spectral rolloff frequency.
    ///
    /// The spectral rolloff is the frequency below which a specified proportion
    /// of the total spectral energy is contained. It is commonly used to
    /// distinguish harmonic signals from noise-like signals.
    ///
    /// For multi-channel audio only the first channel is used.
    ///
    /// # Arguments
    /// - `rolloff_percent` — the energy proportion threshold. Must lie in the
    ///   open interval `(0.0, 1.0)`. A typical value is `0.85`.
    ///
    /// # Returns
    /// The rolloff frequency in Hz. Returns `0.0` when the signal is silence
    /// (zero total spectral energy).
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `rolloff_percent` is not in
    ///   `(0.0, 1.0)`.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    ///
    /// # Examples
    /// ```no_run
    /// // Requires feature = "transforms"
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 1.0);
    /// // 85 % of spectral energy is below the rolloff frequency.
    /// let rolloff = audio.spectral_rolloff(0.85).unwrap();
    /// assert!(rolloff > 0.0);
    /// ```
    #[cfg(feature = "transforms")]
    fn spectral_rolloff(&self, rolloff_percent: f64) -> AudioSampleResult<f64>;
}

/// Voice Activity Detection (VAD) operations.
///
/// # Purpose
///
/// Frame-based speech/silence classification for audio signals.  Produces
/// per-frame boolean masks and contiguous speech region boundaries expressed
/// as sample-index pairs.
///
/// # Intended Usage
///
/// Use this trait to strip silence, isolate voiced segments, or gate downstream
/// processing to active speech regions.  Multi-channel audio is handled via the
/// `channel_policy` field of [`VadConfig`].
///
/// # Invariants
///
/// Both methods delegate to the same internal frame-analysis pipeline, so
/// [`speech_regions`][AudioVoiceActivityDetection::speech_regions] is always
/// consistent with [`voice_activity_mask`][AudioVoiceActivityDetection::voice_activity_mask].
#[cfg(feature = "vad")]
pub trait AudioVoiceActivityDetection: AudioTypeConversion
where
    Self::Sample: StandardSample,
{
    /// Classifies each audio frame as speech (`true`) or silence (`false`).
    ///
    /// Divides the signal into overlapping frames according to `config.frame_size`
    /// and `config.hop_size`, applies the detection method, and post-processes
    /// the raw per-frame decisions through four sequential steps:
    ///
    /// 1. Majority-vote smoothing over `config.smooth_frames`.
    /// 2. Hangover extension — extends active frames forward by `config.hangover_frames`.
    /// 3. Minimum speech run enforcement — speech runs shorter than
    ///    `config.min_speech_frames` are reclassified as silence.
    /// 4. Minimum silence run enforcement — silence gaps shorter than
    ///    `config.min_silence_frames` are reclassified as speech.
    ///
    /// Multi-channel audio is handled according to `config.channel_policy`.
    ///
    /// # Arguments
    ///
    /// - `config` – VAD parameters: frame size, hop size, detection method,
    ///   energy threshold, ZCR bounds, smoothing, hangover, and minimum
    ///   region lengths.
    ///
    /// # Returns
    ///
    /// A `NonEmptyVec<bool>` with one entry per analysis frame. The length
    /// equals the number of frame starts produced by `config.frame_size`,
    /// `config.hop_size`, and `config.pad_end`.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if `config` fails validation
    /// (e.g. `hop_size > frame_size` or invalid ZCR bounds).
    /// Returns [crate::AudioSampleError::Layout] if the audio array is non-contiguous.
    /// Returns [crate::AudioSampleError::Feature] if `config.method` is
    /// `VadMethod::Spectral` (not yet implemented).
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::operations::traits::AudioVoiceActivityDetection;
    /// use audio_samples::operations::types::VadConfig;
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f32>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.5);
    /// let config = VadConfig::energy_only();
    /// let mask = audio.voice_activity_mask(&config).unwrap();
    /// assert!(mask.iter().any(|&v| v));
    /// ```
    fn voice_activity_mask(&self, config: &VadConfig) -> AudioSampleResult<NonEmptyVec<bool>>;

    /// Returns contiguous speech segments as `(start_sample, end_sample)` pairs.
    ///
    /// Internally calls [`voice_activity_mask`] to obtain per-frame decisions,
    /// then converts frame indices to sample-index ranges. Adjacent or
    /// overlapping regions are merged and the result is sorted by
    /// `start_sample`. `end_sample` is exclusive (one past the last sample of
    /// the region).
    ///
    /// [`voice_activity_mask`]: AudioVoiceActivityDetection::voice_activity_mask
    ///
    /// # Arguments
    ///
    /// - `config` – VAD parameters (same as [`voice_activity_mask`]).
    ///
    /// # Returns
    ///
    /// A `Vec<(usize, usize)>` of `(start_sample, end_sample)` pairs sorted
    /// by `start_sample`. Returns an empty `Vec` if no speech frames are
    /// detected.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`voice_activity_mask`].
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::operations::traits::AudioVoiceActivityDetection;
    /// use audio_samples::operations::types::VadConfig;
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f32>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.5);
    /// let regions = audio.speech_regions(&VadConfig::energy_only()).unwrap();
    /// for &(start, end) in &regions {
    ///     assert!(start < end);
    /// }
    /// ```
    fn speech_regions(&self, config: &VadConfig) -> AudioSampleResult<Vec<(usize, usize)>>;
}

/// Signal processing operations for audio manipulation.
///
/// # Purpose
///
/// Provides time-domain signal processing primitives: amplitude scaling,
/// normalisation, windowing, FIR filtering, μ-law compression/expansion,
/// DC removal, hard clipping, and high-quality resampling.
///
/// # Intended Usage
///
/// Apply these methods in a preparation or post-processing pipeline — for example
/// to normalise a recording before feature extraction, to remove DC bias before
/// spectral analysis, or to resample to a target rate before model inference.
/// Most methods consume `self` and return the processed signal, enabling
/// method chaining:
///
/// ```rust
/// use audio_samples::{AudioSamples, AudioProcessing, AudioStatistics,
///                     NormalizationConfig, sample_rate};
/// use ndarray::array;
///
/// let audio = AudioSamples::new_mono(array![2.0f32, -4.0, 1.0], sample_rate!(44100))
///     .unwrap()
///     .normalize(NormalizationConfig::peak(1.0))
///     .unwrap();
/// assert!(audio.peak() <= 1.0 + 1e-6);
/// ```
///
/// Resampling methods (`resample`, `resample_by_ratio`) take `&self` rather than
/// consuming `self`, since they produce a new allocation at a different rate.
///
/// # Invariants
///
/// Methods that accept a sample-rate or frequency parameter validate that the
/// value is within the valid range; they return [crate::AudioSampleError::Parameter]
/// on failure rather than panicking.
#[cfg(feature = "processing")]
pub trait AudioProcessing: AudioTypeConversion
where
    Self::Sample: StandardSample,
{
    /// Normalizes audio samples using the specified configuration.
    ///
    /// The normalization method determines both the algorithm and parameters:
    /// - `NormalizationConfig::min_max(min, max)` — scale to the `[min, max]` range
    /// - `NormalizationConfig::peak(target)` — scale so the peak equals `target`
    /// - `NormalizationConfig::mean()` — subtract the mean (center around zero)
    /// - `NormalizationConfig::median()` — subtract the median (mono only)
    /// - `NormalizationConfig::zscore()` — transform to zero mean, unit variance
    ///
    /// # Arguments
    /// - `config` - Normalization configuration. Use the associated constructors
    ///   on [`NormalizationConfig`] to build the desired method and parameters.
    ///
    /// # Returns
    /// The normalized audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `min >= max` (MinMax), or if
    ///   the input is multi-channel (Median).
    ///
    /// # Panics
    /// Panics if the configuration fields required by the selected method are
    /// `None`. Use the [`NormalizationConfig`] constructors to avoid this.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, AudioStatistics, NormalizationConfig, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -3.0, 2.0, -1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .normalize(NormalizationConfig::peak(1.0))
    ///     .unwrap();
    /// assert!(audio.peak() <= 1.0);
    /// ```
    fn normalize(self, config: NormalizationConfig<Self::Sample>) -> AudioSampleResult<Self>;

    /// Scales all audio samples by a constant factor.
    ///
    /// This is equivalent to adjusting the volume or amplitude of the signal.
    /// A factor of 1.0 leaves the signal unchanged; values > 1.0 amplify and
    /// values < 1.0 attenuate.
    ///
    /// # Arguments
    /// - `factor` - The scaling factor to apply to all samples.
    ///
    /// # Returns
    /// The scaled audio samples.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 0.5];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .scale(2.0);
    /// assert_eq!(audio[0], 2.0);
    /// assert_eq!(audio[1], -2.0);
    /// ```
    #[must_use]
    fn scale(self, factor: f64) -> Self;

    /// Applies a windowing function to the audio samples.
    ///
    /// Multiplies each sample by the corresponding window coefficient
    /// element-wise. Windowing is commonly used before FFT operations to
    /// reduce spectral leakage. Applied independently to each channel.
    ///
    /// # Arguments
    /// - `window` - Window coefficients. Length must equal the number of
    ///   samples in the audio.
    ///
    /// # Returns
    /// The windowed audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the window length does not
    ///   match the audio length.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    /// use non_empty_slice::NonEmptySlice;
    ///
    /// let data = array![1.0f32, 1.0, 1.0, 1.0];
    /// let window = [1.0f32, 0.5, 0.5, 1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .apply_window(NonEmptySlice::new(&window).unwrap())
    ///     .unwrap();
    /// assert_eq!(audio[0], 1.0);
    /// assert_eq!(audio[1], 0.5);
    /// ```
    fn apply_window(self, window: &NonEmptySlice<Self::Sample>) -> AudioSampleResult<Self>;

    /// Applies a digital filter to the audio samples using direct FIR convolution.
    ///
    /// Each output sample is the dot product of the filter coefficients with the
    /// corresponding segment of the input signal. The resulting audio is shorter
    /// than the original by `filter_coeffs.len() - 1` samples.
    ///
    /// # Arguments
    /// - `filter_coeffs` - FIR filter coefficients. A single-element slice
    ///   `[1.0]` leaves the signal unchanged.
    ///
    /// # Returns
    /// The filtered audio samples, with length reduced by `filter_coeffs.len() - 1`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the audio is shorter than
    ///   the filter.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    /// use non_empty_slice::NonEmptySlice;
    ///
    /// let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
    /// let coeffs = [0.5f32, 0.5]; // 2-sample moving average
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .apply_filter(NonEmptySlice::new(&coeffs).unwrap())
    ///     .unwrap();
    /// assert_eq!(audio[0], 1.5); // 1*0.5 + 2*0.5
    /// assert_eq!(audio[1], 2.5); // 2*0.5 + 3*0.5
    /// ```
    fn apply_filter(self, filter_coeffs: &NonEmptySlice<Self::Sample>) -> AudioSampleResult<Self>;

    /// Applies μ-law compression to the audio samples.
    ///
    /// Compresses the dynamic range of the signal using a μ-law nonlinear
    /// transfer function. Higher `mu` values produce stronger compression.
    ///
    /// # Arguments
    /// - `mu` - Compression parameter (typically 255 for standard μ-law).
    ///
    /// # Returns
    /// The compressed audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if sample-type conversion fails.
    ///
    /// # See Also
    /// - [μ-law algorithm (Wikipedia)](https://en.wikipedia.org/wiki/G.711)
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![0.5f32, -0.5];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .mu_compress(255.0)
    ///     .unwrap();
    /// assert!(audio[0] > 0.0); // Positive input stays positive
    /// assert!(audio[1] < 0.0); // Negative input stays negative
    /// ```
    fn mu_compress(self, mu: Self::Sample) -> AudioSampleResult<Self>;

    /// Applies μ-law expansion (decompression) to the audio samples.
    ///
    /// This inverts μ-law compression. The `mu` parameter must match the value
    /// used during compression for correct reconstruction.
    ///
    /// # Arguments
    /// - `mu` - Expansion parameter. Must match the value used for compression.
    ///
    /// # Returns
    /// The expanded audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if sample-type conversion fails.
    ///
    /// # See Also
    /// - [μ-law algorithm (Wikipedia)](https://en.wikipedia.org/wiki/G.711)
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![0.0f32, 0.5, -0.5];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .mu_expand(255.0)
    ///     .unwrap();
    /// assert_eq!(audio[0], 0.0); // Zero maps to zero
    /// assert!(audio[1] > 0.0);   // Sign is preserved
    /// assert!(audio[2] < 0.0);   // Sign is preserved
    /// ```
    fn mu_expand(self, mu: Self::Sample) -> AudioSampleResult<Self>;

    /// Applies a first-order low-pass filter with the specified cutoff frequency.
    ///
    /// Uses a single-pole IIR filter to attenuate frequencies above the cutoff.
    /// The filter operates independently on each channel.
    ///
    /// # Arguments
    /// - `cutoff_hz` - Cutoff frequency in Hz. Must be less than the Nyquist
    ///   frequency (half the sample rate).
    ///
    /// # Returns
    /// The filtered audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `cutoff_hz` is ≥ the
    ///   Nyquist frequency.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .low_pass_filter(1000.0)
    ///     .unwrap();
    /// // High-frequency content is attenuated
    /// assert!(audio[1].abs() < 1.0);
    /// ```
    fn low_pass_filter(self, cutoff_hz: f64) -> AudioSampleResult<Self>;

    /// Applies a first-order high-pass filter with the specified cutoff frequency.
    ///
    /// Uses an RC high-pass filter model to attenuate frequencies below the
    /// cutoff. The filter operates independently on each channel.
    ///
    /// # Arguments
    /// - `cutoff_hz` - Cutoff frequency in Hz. Must be less than the Nyquist
    ///   frequency (half the sample rate).
    ///
    /// # Returns
    /// The filtered audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `cutoff_hz` is ≥ the
    ///   Nyquist frequency.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, 1.0, 1.0, 1.0]; // Constant (DC) signal
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .high_pass_filter(100.0)
    ///     .unwrap();
    /// // A constant signal is fully removed by a high-pass filter
    /// assert_eq!(audio[0], 0.0);
    /// assert_eq!(audio[3], 0.0);
    /// ```
    fn high_pass_filter(self, cutoff_hz: f64) -> AudioSampleResult<Self>;

    /// Applies a band-pass filter that passes frequencies between the two cutoffs.
    ///
    /// Implemented by cascading a high-pass filter at `low_cutoff_hz` followed
    /// by a low-pass filter at `high_cutoff_hz`.
    ///
    /// # Arguments
    /// - `low_cutoff_hz` - Lower cutoff frequency in Hz.
    /// - `high_cutoff_hz` - Upper cutoff frequency in Hz. Must be greater than
    ///   `low_cutoff_hz` and less than the Nyquist frequency.
    ///
    /// # Returns
    /// The filtered audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `low_cutoff_hz >= high_cutoff_hz`,
    ///   or if either cutoff exceeds the Nyquist frequency.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .band_pass_filter(100.0, 5000.0)
    ///     .unwrap();
    /// assert!(audio[0].is_finite());
    /// ```
    fn band_pass_filter(self, low_cutoff_hz: f64, high_cutoff_hz: f64) -> AudioSampleResult<Self>;
    /// Removes DC offset by subtracting the mean value.
    ///
    /// This centers the audio around zero and removes any constant bias that
    /// may have been introduced during recording or processing.
    ///
    /// # Returns
    /// The audio samples with the DC offset removed.
    ///
    /// # Errors
    /// Returns an error if the audio data layout is invalid or if mean computation fails.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![2.0f32, 3.0, 4.0, 5.0]; // Has DC offset
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .remove_dc_offset()
    ///     .unwrap();
    /// let mean: f64 = audio.mean();
    /// assert!(mean.abs() < 1e-6); // Mean is now ~0
    /// ```
    fn remove_dc_offset(self) -> AudioSampleResult<Self>;

    /// Clips audio samples to the specified range.
    ///
    /// Any samples outside `[min_val, max_val]` are clamped to the nearest
    /// boundary. Useful for preventing digital clipping before output.
    ///
    /// # Arguments
    /// - `min_val` - Minimum allowed sample value.
    /// - `max_val` - Maximum allowed sample value.
    ///
    /// # Returns
    /// The clipped audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `min_val > max_val`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![2.0f32, -3.0, 1.5, -0.5];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .clip(-1.0, 1.0)
    ///     .unwrap();
    /// assert_eq!(audio[0], 1.0);   // clamped to max
    /// assert_eq!(audio[1], -1.0);  // clamped to min
    /// assert_eq!(audio[3], -0.5);  // within range, unchanged
    /// ```
    fn clip(self, min_val: Self::Sample, max_val: Self::Sample) -> AudioSampleResult<Self>;

    /// Resamples audio to a new sample rate using high-quality resampling.
    ///
    /// Delegates to the `rubato` resampling library. The `quality` parameter
    /// controls the trade-off between speed and output fidelity.
    ///
    /// # Arguments
    /// - `target_sample_rate` - Desired output sample rate.
    /// - `quality` - Resampling quality preset (see [`ResamplingQuality`]).
    ///
    /// # Returns
    /// A new [`AudioSamples`] instance at the target sample rate.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the target sample rate or
    ///   quality parameters are invalid.
    /// - [`crate::AudioSampleError::Layout`] if the input audio is empty.
    #[cfg(feature = "resampling")]
    fn resample(
        &self,
        target_sample_rate: SampleRate,
        quality: ResamplingQuality,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// Resamples audio by a specific ratio.
    ///
    /// The output length is scaled by `ratio` relative to the input. A ratio
    /// of 2.0 doubles the sample count; 0.5 halves it.
    ///
    /// Delegates to the `rubato` resampling library. The `quality` parameter
    /// controls the trade-off between speed and output fidelity.
    ///
    /// # Arguments
    /// - `ratio` - Resampling ratio (`output_rate / input_rate`). Must be > 0.
    /// - `quality` - Resampling quality preset (see [`ResamplingQuality`]).
    ///
    /// # Returns
    /// A new [`AudioSamples`] instance resampled by the given ratio.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `ratio` is ≤ 0 or the
    ///   quality parameters are invalid.
    /// - [`crate::AudioSampleError::Layout`] if the input audio is empty.
    #[cfg(feature = "resampling")]
    fn resample_by_ratio(
        &self,
        ratio: f64,
        quality: ResamplingQuality,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;
}

/// Frequency-domain analysis and spectral transformation operations.
///
/// # Purpose
///
/// Provides FFT, STFT, and a comprehensive set of spectral representations —
/// linear, log-frequency, mel-scale, MFCC, chromagram, gammatone, and
/// constant-Q — all delegating to the `spectrograms` crate for numerics.
///
/// # Intended Usage
///
/// Use these methods when you need to move from the time domain into a
/// frequency or perceptual representation:
///
/// - Feature extraction for machine-learning models: mel spectrograms, MFCCs, chroma.
/// - Analysis and visualisation: linear spectrograms, power spectral density.
/// - Round-trip processing: STFT → process → ISTFT.
/// - Low-level building blocks: FFT, RFFT, magnitude/phase decomposition.
///
/// The `spectrograms` types returned by spectrogram methods are defined in
/// the `spectrograms` crate and are **not** re-exported from `audio_samples`.
/// Import them directly from `spectrograms`.
///
/// # Invariants
///
/// Methods that accept an FFT length (`n_fft`) zero-pad shorter signals rather
/// than truncating.  Signals *longer* than `n_fft` are rejected with a
/// [crate::AudioSampleError::Parameter] error.
#[cfg(feature = "transforms")]
pub trait AudioTransforms: AudioTypeConversion
where
    Self::Sample: StandardSample,
{
    /// Computes the Fast Fourier Transform of the audio signal.
    ///
    /// Each channel is transformed independently. The output has one row per
    /// channel containing the complex spectral bins.
    ///
    /// # Arguments
    /// - `n_fft` — FFT length in samples. If longer than the signal the input
    ///   is zero-padded internally.
    ///
    /// # Returns
    /// An `Array2<Complex<f64>>` where each row is the FFT of the
    /// corresponding channel.
    ///
    /// # Errors
    /// Returns an error if the FFT computation fails.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, sample_rate};
    /// use ndarray::array;
    /// use std::num::NonZeroUsize;
    ///
    /// let data  = array![1.0f32, 0.0, -1.0, 0.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let spectrum = audio.fft(NonZeroUsize::new(4).unwrap()).unwrap();
    /// assert_eq!(spectrum.shape()[0], 1); // one row per channel
    /// ```
    fn fft(&self, n_fft: NonZeroUsize) -> AudioSampleResult<Array2<Complex<f64>>>;

    /// Computes the magnitude spectrum of the audio signal.
    ///
    /// Equivalent to [`fft`] followed by taking the absolute value of each complex bin.
    /// Each channel is transformed independently.
    ///
    /// # Arguments
    ///
    /// - `n_fft` — FFT length in samples. If longer than the signal the input is
    ///   zero-padded internally.
    ///
    /// # Returns
    ///
    /// An `Array2<f64>` where each row is the magnitude spectrum of the corresponding
    /// channel (one row per channel).
    ///
    /// # Errors
    ///
    /// Returns an error if the FFT computation fails.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// // Use a short signal so n_fft (1024) is larger than the signal length.
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(10), sample_rate!(44100), 0.8);
    /// let mag = audio.rfft(nzu!(1024)).unwrap();
    /// assert_eq!(mag.nrows(), 1); // one row per channel
    /// ```
    ///
    /// [`fft`]: Self::fft
    #[inline]
    fn rfft(&self, n_fft: NonZeroUsize) -> AudioSampleResult<Array2<f64>> {
        let fft_complex = self.fft(n_fft)?;
        Ok(fft_complex.mapv(Complex::norm))
    }

    /// Computes the Short-Time Fourier Transform (STFT) of a mono signal.
    ///
    /// The signal is divided into overlapping, windowed frames and each frame
    /// is transformed to the frequency domain. The returned [`StftResult`]
    /// carries both the complex matrix and the parameters required by
    /// [`istft`] for reconstruction.
    ///
    /// # Arguments
    /// - `params` — STFT configuration (FFT size, hop size, window function,
    ///   and centering behaviour). Field-level constraints are documented on
    ///   [`StftParams`].
    ///
    /// # Returns
    /// An [`StftResult`] containing the complex STFT matrix and metadata.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the signal is multi-channel.
    /// - Errors from the underlying STFT computation.
    ///
    /// [`istft`]: Self::istft
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let params = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let result = audio.stft(&params).unwrap();
    /// assert!(result.data.nrows() > 0); // frequency bins
    /// ```
    fn stft(&self, params: &StftParams) -> AudioSampleResult<StftResult>;

    /// Reconstructs a time-domain signal from an [`StftResult`].
    ///
    /// Uses overlap-add synthesis with the window and hop parameters stored
    /// in the [`StftResult`]. The output is a mono signal at the sample rate
    /// that was recorded during the forward transform.
    ///
    /// # Arguments
    /// - `stft` — the [`StftResult`] produced by a prior call to [`stft`].
    ///
    /// # Returns
    /// A mono [`AudioSamples`] at the original sample rate.
    ///
    /// # Errors
    /// Returns an error if the reconstruction fails (e.g. mismatched
    /// parameters inside the [`StftResult`]).
    ///
    /// [`stft`]: Self::stft
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let params = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let stft_result = audio.stft(&params).unwrap();
    /// let reconstructed = AudioSamples::<f64>::istft(stft_result).unwrap();
    /// assert!(reconstructed.samples_per_channel().get() > 0);
    /// ```
    fn istft(stft: StftResult) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// Computes a linearly-spaced spectrogram.
    ///
    /// Prefer the typed convenience methods —
    /// [`linear_magnitude_spectrogram`](crate::operations::AudioTransforms::linear_magnitude_spectrogram), [`linear_power_spectrogram`](crate::operations::AudioTransforms::linear_power_spectrogram), or
    /// [`linear_db_spectrogram`](crate::operations::AudioTransforms::linear_db_spectrogram) — for the most common amplitude scales.
    ///
    /// # Arguments
    /// - `params` — spectrogram parameters (window, hop, FFT size).
    /// - `db` — required when `AmpScale` is `Decibels`; ignored otherwise.
    ///
    /// # Returns
    /// A `Spectrogram<LinearHz, AmpScale>`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying spectrogram computation.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{Magnitude, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let spect = audio.linear_spectrogram::<Magnitude>(&params, None).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    fn linear_spectrogram<AmpScale>(
        &self,
        params: &SpectrogramParams,
        db: Option<&LogParams>,
    ) -> AudioSampleResult<Spectrogram<LinearHz, AmpScale>>
    where
        AmpScale: AmpScaleSpec;

    /// Shorthand for [`linear_spectrogram`](crate::operations::AudioTransforms::linear_spectrogram) with `Magnitude` amplitude scale.
    ///
    /// # Errors
    /// Returns an error if STFT computation fails or if parameters are invalid.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let spect = audio.linear_magnitude_spectrogram(&params).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn linear_magnitude_spectrogram(
        &self,
        params: &SpectrogramParams,
    ) -> AudioSampleResult<LinearMagnitudeSpectrogram> {
        self.linear_spectrogram::<Magnitude>(params, None)
    }

    /// Shorthand for [`linear_spectrogram`](crate::operations::AudioTransforms::linear_spectrogram) with `Power` amplitude scale.
    ///
    /// # Errors
    /// Returns an error if STFT computation fails or if parameters are invalid.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let spect = audio.linear_power_spectrogram(&params).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn linear_power_spectrogram(
        &self,
        params: &SpectrogramParams,
    ) -> AudioSampleResult<LinearPowerSpectrogram> {
        self.linear_spectrogram::<Power>(params, None)
    }

    /// Shorthand for [`linear_spectrogram`](crate::operations::AudioTransforms::linear_spectrogram) with `Decibels` amplitude scale.
    ///
    /// # Errors
    /// Returns an error if STFT computation fails or if parameters are invalid.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{LogParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let db = LogParams::new(-80.0).unwrap();
    /// let spect = audio.linear_db_spectrogram(&params, &db).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn linear_db_spectrogram(
        &self,
        params: &SpectrogramParams,
        db: &LogParams,
    ) -> AudioSampleResult<LinearDbSpectrogram> {
        self.linear_spectrogram::<Decibels>(params, Some(db))
    }

    /// Computes a log-frequency-spaced spectrogram.
    ///
    /// Prefer the typed convenience methods —
    /// [`loghz_power_spectrogram`](crate::operations::AudioTransforms::loghz_power_spectrogram), [`loghz_magnitude_spectrogram`](crate::operations::AudioTransforms::loghz_magnitude_spectrogram), or
    /// [`loghz_db_spectrogram`](crate::operations::AudioTransforms::loghz_db_spectrogram) — for the most common amplitude scales.
    ///
    /// # Arguments
    /// - `params` — spectrogram parameters (window, hop, FFT size).
    /// - `loghz` — log-Hz frequency-axis configuration (min/max
    ///   frequency, number of bins). Field-level constraints are
    ///   documented on [`LogHzParams`].
    /// - `db` — required when `AmpScale` is `Decibels`; ignored otherwise.
    ///
    /// # Returns
    /// A `Spectrogram<LogHz, AmpScale>`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying spectrogram computation.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{LogHzParams, Magnitude, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let loghz = LogHzParams::new(nzu!(64), 80.0, 8_000.0).unwrap();
    /// let spect = audio.log_frequency_spectrogram::<Magnitude>(&params, &loghz, None).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    fn log_frequency_spectrogram<AmpScale>(
        &self,
        params: &SpectrogramParams,
        loghz: &LogHzParams,
        db: Option<&LogParams>,
    ) -> AudioSampleResult<Spectrogram<LogHz, AmpScale>>
    where
        AmpScale: AmpScaleSpec;

    /// Shorthand for [`log_frequency_spectrogram`](crate::operations::AudioTransforms::log_frequency_spectrogram) with `Power` amplitude scale.
    ///
    /// # Errors
    /// Returns an error if STFT computation or frequency binning fails, or if parameters are invalid.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{LogHzParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let loghz = LogHzParams::new(nzu!(64), 80.0, 8_000.0).unwrap();
    /// let spect = audio.loghz_power_spectrogram(&params, &loghz).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn loghz_power_spectrogram(
        &self,
        params: &SpectrogramParams,
        loghz: &LogHzParams,
    ) -> AudioSampleResult<LogHzPowerSpectrogram> {
        self.log_frequency_spectrogram::<Power>(params, loghz, None)
    }

    /// Shorthand for [`log_frequency_spectrogram`](crate::operations::AudioTransforms::log_frequency_spectrogram) with `Magnitude` amplitude scale.
    ///
    /// # Errors
    /// Returns an error if STFT computation or frequency binning fails, or if parameters are invalid.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{LogHzParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let loghz = LogHzParams::new(nzu!(64), 80.0, 8_000.0).unwrap();
    /// let spect = audio.loghz_magnitude_spectrogram(&params, &loghz).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn loghz_magnitude_spectrogram(
        &self,
        params: &SpectrogramParams,
        loghz: &LogHzParams,
    ) -> AudioSampleResult<LogHzMagnitudeSpectrogram> {
        self.log_frequency_spectrogram::<Magnitude>(params, loghz, None)
    }

    /// Shorthand for [`log_frequency_spectrogram`](crate::operations::AudioTransforms::log_frequency_spectrogram) with `Decibels` amplitude scale.
    ///
    /// # Errors
    /// Returns an error if STFT computation or frequency binning fails, or if parameters are invalid.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{LogHzParams, LogParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let loghz = LogHzParams::new(nzu!(64), 80.0, 8_000.0).unwrap();
    /// let db = LogParams::new(-80.0).unwrap();
    /// let spect = audio.loghz_db_spectrogram(&params, &loghz, &db).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn loghz_db_spectrogram(
        &self,
        params: &SpectrogramParams,
        loghz: &LogHzParams,
        db: &LogParams,
    ) -> AudioSampleResult<LogHzDbSpectrogram> {
        self.log_frequency_spectrogram::<Decibels>(params, loghz, Some(db))
    }

    /// Computes a mel-scaled spectrogram.
    ///
    /// The mel scale approximates human auditory perception by compressing
    /// high frequencies relative to low ones.  Prefer the typed
    /// convenience methods — [`mel_mag_spectrogram`](crate::operations::AudioTransforms::mel_mag_spectrogram),
    /// [`mel_power_spectrogram`](crate::operations::AudioTransforms::mel_power_spectrogram), or [`mel_db_spectrogram`](crate::operations::AudioTransforms::mel_db_spectrogram) — for the
    /// most common amplitude scales.
    ///
    /// # Arguments
    /// - `params` — spectrogram parameters (window, hop, FFT size).
    /// - `mel` — mel filter-bank configuration (number of bands,
    ///   frequency range). Field-level constraints are documented on
    ///   [`MelParams`].
    /// - `db` — required when `AmpScale` is `Decibels`; ignored otherwise.
    ///
    /// # Returns
    /// A [`MelSpectrogram<AmpScale>`].
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying spectrogram computation.
    ///
    /// ## See Also
    /// - [Mel scale — Wikipedia](https://en.wikipedia.org/wiki/Mel_scale)
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{Magnitude, MelParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let mel = MelParams::new(nzu!(40), 0.0, 8_000.0).unwrap();
    /// let spect = audio.mel_spectrogram::<Magnitude>(&params, &mel, None).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    fn mel_spectrogram<AmpScale>(
        &self,
        params: &SpectrogramParams,
        mel: &MelParams,
        db: Option<&LogParams>, // only used when AmpScale = Decibels
    ) -> AudioSampleResult<MelSpectrogram<AmpScale>>
    where
        AmpScale: AmpScaleSpec;

    /// Shorthand for [`mel_spectrogram`](crate::operations::AudioTransforms::mel_spectrogram) with `Magnitude` amplitude scale.
    ///
    /// # Errors
    /// Returns an error if STFT computation or mel filterbank application fails, or if parameters are invalid.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{MelParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let mel = MelParams::new(nzu!(40), 0.0, 8_000.0).unwrap();
    /// let spect = audio.mel_mag_spectrogram(&params, &mel).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn mel_mag_spectrogram(
        &self,
        params: &SpectrogramParams,
        mel: &MelParams,
    ) -> AudioSampleResult<MelMagnitudeSpectrogram> {
        self.mel_spectrogram(params, mel, None)
    }

    /// Shorthand for [`mel_spectrogram`](crate::operations::AudioTransforms::mel_spectrogram) with `Decibels` amplitude scale.
    ///
    /// # Errors
    /// Returns an error if STFT computation or mel filterbank application fails, or if parameters are invalid.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{LogParams, MelParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let mel = MelParams::new(nzu!(40), 0.0, 8_000.0).unwrap();
    /// let db = LogParams::new(-80.0).unwrap();
    /// let spect = audio.mel_db_spectrogram(&params, &mel, &db).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn mel_db_spectrogram(
        &self,
        params: &SpectrogramParams,
        mel: &MelParams,
        db: &LogParams,
    ) -> AudioSampleResult<LogMelSpectrogram> {
        self.mel_spectrogram(params, mel, Some(db))
    }

    /// Shorthand for [`mel_spectrogram`](crate::operations::AudioTransforms::mel_spectrogram) with `Power` amplitude scale.
    ///
    /// # Errors
    /// Returns an error if STFT computation or mel filterbank application fails, or if parameters are invalid.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{MelParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let mel = MelParams::new(nzu!(40), 0.0, 8_000.0).unwrap();
    /// let spect = audio.mel_power_spectrogram(&params, &mel).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn mel_power_spectrogram(
        &self,
        params: &SpectrogramParams,
        mel: &MelParams,
    ) -> AudioSampleResult<MelPowerSpectrogram> {
        self.mel_spectrogram(params, mel, None)
    }

    /// Computes Mel-Frequency Cepstral Coefficients (MFCCs).
    ///
    /// MFCCs are a compact spectral representation widely used in speech
    /// recognition and audio classification.  They are derived by
    /// applying a DCT to log-mel filter-bank energies.
    ///
    /// # Arguments
    /// - `stft_params` — STFT configuration used for the underlying
    ///   spectrogram.
    /// - `n_mels` — number of mel filter-bank bands.
    /// - `mfcc_params` — MFCC-specific configuration (number of
    ///   coefficients, etc.). Field-level constraints are documented on
    ///   [`MfccParams`].
    ///
    /// # Returns
    /// An [`Mfcc`] containing the MFCC matrix.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying computation.
    ///
    /// ## See Also
    /// - [MFCC — Wikipedia](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{MfccParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let mfcc_params = MfccParams::speech_standard();
    /// let result = audio.mfcc(&stft, nzu!(40), &mfcc_params).unwrap();
    /// assert!(result.data.nrows() > 0); // MFCC coefficients
    /// ```
    fn mfcc(
        &self,
        stft_params: &StftParams,
        n_mels: NonZeroUsize,
        mfcc_params: &MfccParams,
    ) -> AudioSampleResult<Mfcc>;

    /// Computes chromagram (pitch-class energy) features.
    ///
    /// A chromagram projects the spectrum onto the twelve pitch classes
    /// (C, C♯, D, … , B), collapsing octave differences.  The result
    /// is useful for harmonic and key detection.
    ///
    /// # Arguments
    /// - `stft_params` — STFT configuration used for the underlying
    ///   spectrogram.
    /// - `cfg` — chromagram configuration (tuning, normalization,
    ///   etc.). Field-level constraints are documented on
    ///   [`ChromaParams`].
    ///
    /// # Returns
    /// A [`Chromagram`].
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying computation.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{ChromaParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let chroma_params = ChromaParams::music_standard();
    /// let result = audio.chromagram(&stft, &chroma_params).unwrap();
    /// assert_eq!(result.data.nrows(), 12); // twelve pitch classes (C through B)
    /// ```
    fn chromagram(
        &self,
        stft_params: &StftParams,
        cfg: &ChromaParams,
    ) -> AudioSampleResult<Chromagram>;

    /// Estimates the power spectral density using Welch's method.
    ///
    /// The signal is split into overlapping segments; each is windowed
    /// with a Hanning window and FFT'd, and the resulting periodograms
    /// are averaged.  The final values are normalised to power per Hz.
    ///
    /// # Arguments
    /// - `window_size` — length of each segment in samples.  Must not
    ///   exceed the signal length.
    /// - `overlap` — fractional overlap between adjacent segments, in
    ///   the range `[0, 1)`.
    ///
    /// # Returns
    /// A pair `(frequencies, psd)` of equal length.  `frequencies[i]` is
    /// the centre frequency of bin `i` in Hz; `psd[i]` is the estimated
    /// power spectral density at that frequency.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the signal is
    ///   multi-channel, if `overlap` is outside `[0, 1)`, or if
    ///   `window_size` exceeds the signal length.
    ///
    /// ## See Also
    /// - [Welch's method — Wikipedia](https://en.wikipedia.org/wiki/Welch%27s_method)
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.8);
    /// let (freqs, psd) = audio.power_spectral_density(nzu!(1024), 0.5).unwrap();
    /// assert_eq!(freqs.len(), psd.len());
    /// assert!(!freqs.is_empty());
    /// ```
    fn power_spectral_density(
        &self,
        window_size: NonZeroUsize,
        overlap: f64,
    ) -> AudioSampleResult<(Vec<f64>, Vec<f64>)>;

    /// Computes a gammatone-filtered spectrogram.
    ///
    /// Gammatone filters model the bandpass response of the human
    /// cochlea.  The filter centre frequencies are spaced according to
    /// the ERB (Equivalent Rectangular Bandwidth) scale.  Prefer the
    /// typed convenience methods —
    /// [`gammatone_magnitude_spectrogram`](crate::operations::AudioTransforms::gammatone_magnitude_spectrogram),
    /// [`gammatone_power_spectrogram`](crate::operations::AudioTransforms::gammatone_power_spectrogram), or
    /// [`gammatone_db_spectrogram`](crate::operations::AudioTransforms::gammatone_db_spectrogram) — for the most common amplitude
    /// scales.
    ///
    /// # Arguments
    /// - `params` — spectrogram parameters (window, hop, FFT size).
    /// - `gammatone_params` — gammatone filter-bank configuration
    ///   (number of bands, frequency range). Field-level constraints are
    ///   documented on [`GammatoneParams`].
    /// - `db` — required when `AmpScale` is `Decibels`; ignored otherwise.
    ///
    /// # Returns
    /// A `Spectrogram<Gammatone, AmpScale>`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying computation.
    ///
    /// ## See Also
    /// - [Gammatone filter — Wikipedia](https://en.wikipedia.org/wiki/Gammatone)
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{GammatoneParams, Magnitude, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let gammatone = GammatoneParams::new(nzu!(32), 80.0, 8_000.0).unwrap();
    /// let spect = audio.gammatone_spectrogram::<Magnitude>(&params, &gammatone, None).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    fn gammatone_spectrogram<AmpScale>(
        &self,
        params: &SpectrogramParams,
        gammatone_params: &GammatoneParams,
        db: Option<&LogParams>,
    ) -> AudioSampleResult<Spectrogram<Gammatone, AmpScale>>
    where
        AmpScale: AmpScaleSpec;

    /// Shorthand for [`gammatone_spectrogram`](crate::operations::AudioTransforms::gammatone_spectrogram) with `Magnitude` amplitude scale.
    ///
    /// # Errors
    /// Returns an error if STFT computation or gammatone filterbank application fails, or if parameters are invalid.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{GammatoneParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let gammatone = GammatoneParams::new(nzu!(32), 80.0, 8_000.0).unwrap();
    /// let spect = audio.gammatone_magnitude_spectrogram(&params, &gammatone).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn gammatone_magnitude_spectrogram(
        &self,
        params: &SpectrogramParams,
        gammatone_params: &GammatoneParams,
    ) -> AudioSampleResult<GammatoneMagnitudeSpectrogram> {
        self.gammatone_spectrogram::<Magnitude>(params, gammatone_params, None)
    }

    /// Shorthand for [`gammatone_spectrogram`](crate::operations::AudioTransforms::gammatone_spectrogram) with `Power` amplitude scale.
    ///
    /// # Errors
    /// Returns an error if STFT computation or gammatone filterbank application fails, or if parameters are invalid.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{GammatoneParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let gammatone = GammatoneParams::new(nzu!(32), 80.0, 8_000.0).unwrap();
    /// let spect = audio.gammatone_power_spectrogram(&params, &gammatone).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn gammatone_power_spectrogram(
        &self,
        params: &SpectrogramParams,
        gammatone_params: &GammatoneParams,
    ) -> AudioSampleResult<GammatonePowerSpectrogram> {
        self.gammatone_spectrogram::<Power>(params, gammatone_params, None)
    }

    /// Shorthand for [`gammatone_spectrogram`](crate::operations::AudioTransforms::gammatone_spectrogram) with `Decibels` amplitude scale.
    ///
    /// # Errors
    /// Returns an error if STFT computation or gammatone filterbank application fails, or if parameters are invalid.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{GammatoneParams, LogParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let gammatone = GammatoneParams::new(nzu!(32), 80.0, 8_000.0).unwrap();
    /// let db = LogParams::new(-80.0).unwrap();
    /// let spect = audio.gammatone_db_spectrogram(&params, &gammatone, &db).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn gammatone_db_spectrogram(
        &self,
        params: &SpectrogramParams,
        gammatone_params: &GammatoneParams,
        db: &LogParams,
    ) -> AudioSampleResult<GammatoneDbSpectrogram> {
        self.gammatone_spectrogram::<Decibels>(params, gammatone_params, Some(db))
    }

    /// Computes the Constant-Q Transform (CQT) of a mono signal.
    ///
    /// The CQT uses a bank of bandpass filters whose centre frequencies
    /// are spaced logarithmically with a constant ratio
    /// Q = f / Δf.  This gives it the same frequency resolution as the
    /// musical scale, making it preferred for pitch and harmonic
    /// analysis.
    ///
    /// # Arguments
    /// - `params` — CQT parameters (frequency range, bins per octave,
    ///   etc.). Field-level constraints are documented on [`CqtParams`].
    /// - `hop_size` — hop length in samples between successive frames.
    ///
    /// # Returns
    /// A [`CqtResult`] containing the CQT matrix.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying CQT computation.
    ///
    /// ## See Also
    /// - [Constant-Q transform — Wikipedia](https://en.wikipedia.org/wiki/Constant-Q_transform)
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::CqtParams;
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.8);
    /// // 12 bins per octave, 7 octaves, starting at 32.7 Hz (C1)
    /// let cqt_params = CqtParams::new(nzu!(12), nzu!(7), 32.7).unwrap();
    /// let result = audio.constant_q_transform(&cqt_params, nzu!(256)).unwrap();
    /// assert!(result.data.nrows() > 0);
    /// ```
    fn constant_q_transform(
        &self,
        params: &CqtParams,
        hop_size: NonZeroUsize,
    ) -> AudioSampleResult<CqtResult>;

    /// Computes a CQT-based spectrogram.
    ///
    /// Applies the CQT to the signal and returns the result as a typed
    /// spectrogram.  Prefer the typed convenience methods —
    /// [`cqt_magnitude_spectrogram`](crate::operations::AudioTransforms::cqt_magnitude_spectrogram), [`cqt_power_spectrogram`](crate::operations::AudioTransforms::cqt_power_spectrogram), or
    /// [`cqt_db_spectrogram`](crate::operations::AudioTransforms::cqt_db_spectrogram) — for the most common amplitude scales.
    ///
    /// # Arguments
    /// - `params` — spectrogram parameters (window, hop, FFT size).
    /// - `cqt` — CQT parameters (frequency range, bins per octave,
    ///   etc.). Field-level constraints are documented on [`CqtParams`].
    /// - `db` — required when `AmpScale` is `Decibels`; ignored otherwise.
    ///
    /// # Returns
    /// A [`CqtSpectrogram<AmpScale>`].
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying computation.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{CqtParams, Magnitude, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let cqt = CqtParams::new(nzu!(12), nzu!(7), 32.7).unwrap();
    /// let spect = audio.cqt_spectrogram::<Magnitude>(&params, &cqt, None).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    fn cqt_spectrogram<AmpScale>(
        &self,
        params: &SpectrogramParams,
        cqt: &CqtParams,
        db: Option<&LogParams>,
    ) -> AudioSampleResult<CqtSpectrogram<AmpScale>>
    where
        AmpScale: AmpScaleSpec;

    /// Shorthand for [`cqt_spectrogram`](crate::operations::AudioTransforms::cqt_spectrogram) with `Magnitude` amplitude scale.
    ///
    /// # Errors
    /// Returns an error if the signal is multi-channel, or if STFT or CQT computation fails.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{CqtParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let cqt = CqtParams::new(nzu!(12), nzu!(7), 32.7).unwrap();
    /// let spect = audio.cqt_magnitude_spectrogram(&params, &cqt).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn cqt_magnitude_spectrogram(
        &self,
        params: &SpectrogramParams,
        cqt: &CqtParams,
    ) -> AudioSampleResult<CqtMagnitudeSpectrogram> {
        self.cqt_spectrogram::<Magnitude>(params, cqt, None)
    }

    /// Shorthand for [`cqt_spectrogram`](crate::operations::AudioTransforms::cqt_spectrogram) with `Power` amplitude scale.
    ///
    /// # Errors
    /// Returns an error if the signal is multi-channel, or if STFT or CQT computation fails.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{CqtParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let cqt = CqtParams::new(nzu!(12), nzu!(7), 32.7).unwrap();
    /// let spect = audio.cqt_power_spectrogram(&params, &cqt).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn cqt_power_spectrogram(
        &self,
        params: &SpectrogramParams,
        cqt: &CqtParams,
    ) -> AudioSampleResult<CqtPowerSpectrogram> {
        self.cqt_spectrogram::<Power>(params, cqt, None)
    }

    /// Shorthand for [`cqt_spectrogram`](crate::operations::AudioTransforms::cqt_spectrogram) with `Decibels` amplitude scale.
    ///
    /// # Errors
    /// Returns an error if the signal is multi-channel, or if STFT or CQT computation fails.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use spectrograms::{CqtParams, LogParams, SpectrogramParams, StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.8);
    /// let stft = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true).unwrap();
    /// let params = SpectrogramParams::new(stft, audio.sample_rate_hz()).unwrap();
    /// let cqt = CqtParams::new(nzu!(12), nzu!(7), 32.7).unwrap();
    /// let db = LogParams::new(-80.0).unwrap();
    /// let spect = audio.cqt_db_spectrogram(&params, &cqt, &db).unwrap();
    /// assert!(spect.data().nrows() > 0);
    /// ```
    #[inline]
    fn cqt_db_spectrogram(
        &self,
        params: &SpectrogramParams,
        cqt: &CqtParams,
        db: &LogParams,
    ) -> AudioSampleResult<CqtDbSpectrogram> {
        self.cqt_spectrogram::<Decibels>(params, cqt, Some(db))
    }

    /// Decomposes a complex spectrogram into magnitude and phase.
    ///
    /// Given a complex matrix `D`, returns `(S, P)` such that
    /// `D = S * P` elementwise, where `S` contains magnitudes raised to
    /// `power` and `P` contains unit-magnitude complex phase factors.
    /// Bins where the magnitude is zero are assigned a phase of `1 + 0i`.
    ///
    /// # Arguments
    /// - `complex_spect` — the complex STFT or FFT matrix.
    /// - `power` — exponent applied to the magnitude values.  `None`
    ///   defaults to 1 (raw magnitude).
    ///
    /// # Returns
    /// `(magnitude, phase)` — the magnitude matrix (real-valued) and
    /// the phase matrix (complex unit-magnitude).
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// // Use a short signal so n_fft (1024) is larger than the signal length.
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(10), sample_rate!(44100), 0.8);
    /// let spectrum = audio.fft(nzu!(1024)).unwrap();
    /// let (mag, phase) = AudioSamples::<f64>::magphase(&spectrum, None);
    /// assert_eq!(mag.shape(), phase.shape());
    /// ```
    #[inline]
    #[must_use]
    fn magphase(
        complex_spect: &Array2<Complex<f64>>,
        power: Option<NonZeroUsize>,
    ) -> (Array2<f64>, Array2<Complex<f64>>) {
        // Magnitude: elementwise absolute value

        let mut mag = complex_spect.mapv(num_complex::Complex::norm);

        // zeros_to_ones: 1.0 where mag == 0, else 0.0
        let zeros_to_ones = mag.mapv(|x| if x == 0.0 { 1.0 } else { 0.0 });

        // mag_nonzero = mag + zeros_to_ones
        let mag_nonzero = &mag + &zeros_to_ones;

        // Compute phase = D / mag_nonzero, but handle zeros separately
        let mut phase = complex_spect.clone();

        let power = power.map_or(1.0, |p| p.get() as f64);

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
/// # Purpose
///
/// Provides fundamental-frequency estimation (YIN and autocorrelation),
/// temporal pitch tracking, harmonic-to-noise ratio measurement, harmonic
/// content analysis, and musical key estimation via chromagram correlation.
///
/// # Intended Usage
///
/// Use this trait for pitch-based analysis of monophonic audio — for example
/// to extract a melody contour from a vocal recording, to measure intonation
/// accuracy, or to classify the musical key of a short clip.
///
/// # Invariants
///
/// All methods require mono audio.  Multi-channel signals must be reduced to
/// mono (e.g. with [`AudioChannelOps::to_mono`]) before use; passing a
/// multi-channel signal returns [crate::AudioSampleError::Unsupported].
#[cfg(feature = "pitch-analysis")]
pub trait AudioPitchAnalysis: AudioTypeConversion
where
    Self::Sample: StandardSample,
{
    /// Detects the fundamental frequency using the YIN pitch detection algorithm.
    ///
    /// YIN computes a cumulative mean normalised difference function (CMND) and
    /// finds the first lag below `threshold`, which corresponds to the fundamental
    /// period. Lower thresholds are stricter and reduce false detections; values
    /// in `[0.1, 0.2]` are typical for musical audio.
    ///
    /// # Arguments
    ///
    /// - `threshold` – Confidence threshold in `[0.0, 1.0]`.
    /// - `min_frequency` – Minimum detectable frequency in Hz (> 0).
    /// - `max_frequency` – Maximum detectable frequency in Hz (> `min_frequency`).
    ///
    /// # Returns
    ///
    /// - `Some(frequency_hz)` – Estimated fundamental frequency.
    /// - `None` – Signal is too short, silent, or no pitch was detected.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Unsupported] for multi-channel audio.
    /// Returns [crate::AudioSampleError::Parameter] if `threshold ∉ [0.0, 1.0]`,
    /// `min_frequency ≤ 0.0`, or `max_frequency ≤ min_frequency`.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::operations::traits::AudioPitchAnalysis;
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let hz = 440.0f64;
    /// let audio = sine_wave::<f64>(hz, Duration::from_millis(100), sample_rate!(44100), 1.0);
    ///
    /// let pitch = audio.detect_pitch_yin(0.1, 80.0, 1000.0).unwrap();
    /// assert!(pitch.is_some());
    /// assert!((pitch.unwrap() - hz).abs() < 10.0);
    /// ```
    fn detect_pitch_yin(
        &self,
        threshold: f64,
        min_frequency: f64,
        max_frequency: f64,
    ) -> AudioSampleResult<Option<f64>>;

    /// Detects the fundamental frequency using autocorrelation.
    ///
    /// Finds the lag with maximum autocorrelation within the range implied by
    /// `[min_frequency, max_frequency]` and converts it to a frequency. This
    /// method is fast and effective for clean, periodic signals but less robust
    /// than YIN on noisy or voiced speech.
    ///
    /// # Arguments
    ///
    /// - `min_frequency` – Minimum detectable frequency in Hz (> 1.0).
    /// - `max_frequency` – Maximum detectable frequency in Hz (> `min_frequency`).
    ///
    /// # Returns
    ///
    /// - `Some(frequency_hz)` – Estimated fundamental frequency.
    /// - `None` – Signal is too short, silent, or unpitched.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Unsupported] for multi-channel audio.
    /// Returns [crate::AudioSampleError::Parameter] if `min_frequency ≤ 1.0` or
    /// `max_frequency ≤ min_frequency`.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::operations::traits::AudioPitchAnalysis;
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let hz = 220.0f64;
    /// let audio = sine_wave::<f32>(hz, Duration::from_millis(100), sample_rate!(44100), 1.0);
    ///
    /// let pitch = audio.detect_pitch_autocorr(80.0, 1000.0).unwrap();
    /// assert!(pitch.is_some());
    /// assert!((pitch.unwrap() - hz).abs() < 15.0);
    /// ```
    fn detect_pitch_autocorr(
        &self,
        min_frequency: f64,
        max_frequency: f64,
    ) -> AudioSampleResult<Option<f64>>;

    /// Tracks pitch over time by applying pitch detection to successive windows.
    ///
    /// The signal is split into overlapping frames of `window_size` samples,
    /// advancing by `hop_size` each step. Each frame is analysed independently
    /// using `method`. Frames shorter than `window_size / 2` at the signal end
    /// are discarded. Only [`PitchDetectionMethod::Yin`] and
    /// [`PitchDetectionMethod::Autocorrelation`] are implemented; other variants
    /// log a warning and return `None` for that frame.
    ///
    /// # Arguments
    ///
    /// - `window_size` – Analysis window length in samples; must be ≤ signal length.
    /// - `hop_size` – Step between successive windows in samples; must be < `window_size`.
    /// - `method` – Pitch detection algorithm to use per frame.
    /// - `threshold` – YIN confidence threshold; ignored for autocorrelation.
    /// - `min_frequency` – Minimum detectable frequency in Hz.
    /// - `max_frequency` – Maximum detectable frequency in Hz.
    ///
    /// # Returns
    ///
    /// A `Vec<(f64, Option<f64>)>` of `(time_seconds, frequency_hz)` pairs in
    /// time order. `frequency_hz` is `None` when no pitch was found in that frame.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Unsupported] for multi-channel audio.
    /// Returns [crate::AudioSampleError::Parameter] if `window_size > signal length`
    /// or `hop_size >= window_size`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::num::NonZeroUsize;
    /// use audio_samples::operations::traits::AudioPitchAnalysis;
    /// use audio_samples::operations::types::PitchDetectionMethod;
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let hz = 440.0f64;
    /// let audio = sine_wave::<f32>(hz, Duration::from_millis(500), sample_rate!(44100), 1.0);
    ///
    /// let track = audio.track_pitch(
    ///     NonZeroUsize::new(2048).unwrap(),
    ///     NonZeroUsize::new(512).unwrap(),
    ///     PitchDetectionMethod::Yin,
    ///     0.1,
    ///     80.0,
    ///     1000.0,
    /// ).unwrap();
    ///
    /// assert!(!track.is_empty());
    /// ```
    fn track_pitch(
        &self,
        window_size: NonZeroUsize,
        hop_size: NonZeroUsize,
        method: PitchDetectionMethod,
        threshold: f64,
        min_frequency: f64,
        max_frequency: f64,
    ) -> AudioSampleResult<Vec<(f64, Option<f64>)>>;

    /// Computes the harmonic-to-noise ratio (HNR) in decibels.
    ///
    /// HNR measures how much of the signal's energy comes from periodic
    /// (harmonic) components versus aperiodic (noise) components. A high HNR
    /// indicates a clean, voiced tone; a low or negative HNR indicates
    /// noise-dominated content.
    ///
    /// # Arguments
    ///
    /// - `fundamental_freq` – Known fundamental frequency in Hz (> 0).
    /// - `num_harmonics` – Number of harmonics to accumulate into harmonic power.
    /// - `n_fft` – FFT size. Defaults to the signal length when `None`.
    /// - `window_type` – Window function applied before FFT. Defaults to Hanning when `None`.
    ///
    /// # Returns
    ///
    /// HNR in dB. Returns `f64::INFINITY` when noise power is zero.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Unsupported] for multi-channel audio.
    /// Returns [crate::AudioSampleError::Parameter] if `fundamental_freq ≤ 0.0`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// use audio_samples::operations::traits::AudioPitchAnalysis;
    /// use audio_samples::{sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let hz = 440.0f64;
    /// let audio = sine_wave::<f32>(hz, Duration::from_millis(100), sample_rate!(44100), 1.0);
    ///
    /// let hnr = audio
    ///     .harmonic_to_noise_ratio(hz, NonZeroUsize::new(5).unwrap(), None, None)
    ///     .unwrap();
    /// assert!(hnr > 0.0, "pure sine should have positive HNR, got {hnr:.1} dB");
    /// ```
    fn harmonic_to_noise_ratio(
        &self,
        fundamental_freq: f64,
        num_harmonics: NonZeroUsize,
        n_fft: Option<NonZeroUsize>,
        window_type: Option<WindowType>,
    ) -> AudioSampleResult<f64>;

    /// Analyses the harmonic content relative to a known fundamental frequency.
    ///
    /// Computes the power spectrum of the signal and extracts the peak power
    /// within a `tolerance`-relative frequency band around each harmonic of
    /// `fundamental_freq`. All magnitudes are normalised so that the fundamental
    /// (index 0) equals 1.0.
    ///
    /// # Arguments
    ///
    /// - `fundamental_freq` – Fundamental frequency in Hz (> 0).
    /// - `num_harmonics` – Number of harmonics to extract, including the fundamental.
    /// - `tolerance` – Fractional bandwidth around each harmonic to search, in `[0.0, 1.0]`.
    /// - `n_fft` – FFT size. Defaults to the signal length when `None`.
    /// - `window_type` – Window function applied before FFT. Defaults to Hanning when `None`.
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` of length `num_harmonics`. Index 0 is always 1.0 after
    /// normalisation.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Unsupported] for multi-channel audio.
    /// Returns [crate::AudioSampleError::Parameter] if `fundamental_freq ≤ 0.0` or
    /// `tolerance ∉ [0.0, 1.0]`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::num::NonZeroUsize;
    /// use audio_samples::operations::traits::AudioPitchAnalysis;
    /// use audio_samples::{sample_rate, sawtooth_wave};
    /// use std::time::Duration;
    ///
    /// let hz = 220.0f64;
    /// let audio = sawtooth_wave::<f32>(hz, Duration::from_millis(500), sample_rate!(44100), 1.0);
    ///
    /// let harmonics = audio
    ///     .harmonic_analysis(hz, NonZeroUsize::new(5).unwrap(), 0.1, None, None)
    ///     .unwrap();
    /// assert_eq!(harmonics.len(), 5);
    /// ```
    fn harmonic_analysis(
        &self,
        fundamental_freq: f64,
        num_harmonics: NonZeroUsize,
        tolerance: f64,
        n_fft: Option<NonZeroUsize>,
        window_type: Option<WindowType>,
    ) -> AudioSampleResult<Vec<f64>>;

    /// Estimates the musical key of the audio using chromagram analysis.
    ///
    /// Computes a chromagram and compares the averaged chroma vector against
    /// Krumhansl-Schmuckler major and minor key profiles via Pearson correlation.
    ///
    /// # Arguments
    ///
    /// - `stft_params` – STFT parameters controlling frame size and hop for
    ///   chromagram computation.
    ///
    /// # Returns
    ///
    /// A `(key_index, confidence)` tuple where:
    /// - `key_index` is in `0..=11` for major keys (C=0, C♯=1, …, B=11) and
    ///   `12..=23` for minor keys (Cm=12, C♯m=13, …, Bm=23).
    /// - `confidence` is in `[0.0, 1.0]`.
    ///
    /// # Errors
    ///
    /// Propagates any error from the `spectrograms` crate during STFT or
    /// chromagram computation.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::num::NonZeroUsize;
    /// use audio_samples::operations::traits::AudioPitchAnalysis;
    /// use audio_samples::{sample_rate, sine_wave};
    /// use spectrograms::{StftParams, WindowType};
    /// use std::time::Duration;
    ///
    /// # let audio = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(44100), 1.0);
    /// let params = StftParams::new(
    ///     NonZeroUsize::new(2048).unwrap(),
    ///     NonZeroUsize::new(512).unwrap(),
    ///     WindowType::Hanning,
    ///     true,
    /// ).unwrap();
    /// let (key, confidence) = audio.estimate_key(&params).unwrap();
    /// assert!(key < 24);
    /// assert!((0.0..=1.0).contains(&confidence));
    /// ```
    fn estimate_key(&self, stft_params: &StftParams) -> AudioSampleResult<(usize, f64)>;
}

/// IIR (Infinite Impulse Response) filtering operations.
///
/// # Purpose
///
/// Applies recursive digital filters — Butterworth and Chebyshev Type I — to
/// audio signals.  IIR filters achieve sharp frequency roll-offs with far fewer
/// coefficients than equivalent FIR designs, making them efficient for tasks
/// such as anti-aliasing, DC removal, and band limiting.
///
/// # Intended Usage
///
/// Use this trait when steeper roll-off or lower latency is more important than
/// linear phase response.  For a gentler, linear-phase roll-off prefer the FIR
/// helpers in [`AudioProcessing`].
///
/// ```rust
/// use audio_samples::{AudioSamples, sample_rate};
/// use audio_samples::operations::traits::AudioIirFiltering;
/// use audio_samples::operations::types::IirFilterDesign;
/// use non_empty_slice::NonEmptyVec;
/// use std::num::NonZeroUsize;
///
/// let samples = NonEmptyVec::new(vec![1.0f32, -0.5, 0.25, -0.1, 0.0, 0.1]).unwrap();
/// let mut audio: AudioSamples<'_, f32> =
///     AudioSamples::from_mono_vec(samples, sample_rate!(44100));
/// audio.butterworth_lowpass(NonZeroUsize::new(2).unwrap(), 5000.0).unwrap();
/// ```
///
/// # Invariants
///
/// Each channel is filtered independently from a clean (zero) initial state.
/// The filter coefficients are derived from the audio's own sample rate at
/// call time; changing the sample rate between calls requires re-applying
/// the filter.
#[cfg(feature = "iir-filtering")]
pub trait AudioIirFiltering: AudioTypeConversion
where
    Self::Sample: AudioSample,
{
    /// Apply an IIR filter using the specified design parameters.
    ///
    /// Designs a filter from `design` (using the audio's own sample
    /// rate), then applies it to every channel independently.
    /// Multi-channel audio resets the filter state between channels
    /// so each channel is filtered from a clean initial state.
    ///
    /// # Arguments
    /// - `design` – Filter specification (type, order, frequencies, ripple).
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if any frequency in `design`
    ///   is out of the valid range (0, Nyquist), or the filter type is
    ///   not yet implemented.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioIirFiltering;
    /// use audio_samples::operations::types::IirFilterDesign;
    /// use non_empty_slice::NonEmptyVec;
    /// use std::num::NonZeroUsize;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5, -1.0, 0.0, 1.0, 0.5, -0.5]).unwrap();
    /// let mut audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let design = IirFilterDesign::butterworth_lowpass(NonZeroUsize::new(2).unwrap(), 1000.0);
    /// assert!(audio.apply_iir_filter(&design).is_ok());
    /// ```
    fn apply_iir_filter(&mut self, design: &IirFilterDesign) -> AudioSampleResult<()>;

    /// Apply a second-order Butterworth low-pass filter.
    ///
    /// Convenience wrapper that constructs an [`IirFilterDesign`] and
    /// delegates to [`Self::apply_iir_filter`].  Only order 2 produces
    /// mathematically correct coefficients; other orders use an
    /// approximate placeholder.
    ///
    /// # Arguments
    /// - `order` – Filter order (use 2 for correct results).
    /// - `cutoff_frequency` – Cutoff frequency in Hz; must be in
    ///   (0, Nyquist).
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if `cutoff_frequency` is
    ///   outside the valid range.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioIirFiltering;
    /// use non_empty_slice::NonEmptyVec;
    /// use std::num::NonZeroUsize;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]).unwrap();
    /// let mut audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// assert!(audio.butterworth_lowpass(NonZeroUsize::new(2).unwrap(), 1000.0).is_ok());
    /// ```
    fn butterworth_lowpass(
        &mut self,
        order: NonZeroUsize,
        cutoff_frequency: f64,
    ) -> AudioSampleResult<()>;
    /// Apply a second-order Butterworth high-pass filter.
    ///
    /// Convenience wrapper that constructs an [`IirFilterDesign`] and
    /// delegates to [`Self::apply_iir_filter`].  Only order 2 produces
    /// mathematically correct coefficients; other orders use an
    /// approximate placeholder.
    ///
    /// # Arguments
    /// - `order` – Filter order (use 2 for correct results).
    /// - `cutoff_frequency` – Cutoff frequency in Hz; must be in
    ///   (0, Nyquist).
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if `cutoff_frequency` is
    ///   outside the valid range.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioIirFiltering;
    /// use non_empty_slice::NonEmptyVec;
    /// use std::num::NonZeroUsize;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
    /// let mut audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// assert!(audio.butterworth_highpass(NonZeroUsize::new(2).unwrap(), 500.0).is_ok());
    /// ```
    fn butterworth_highpass(
        &mut self,
        order: NonZeroUsize,
        cutoff_frequency: f64,
    ) -> AudioSampleResult<()>;

    /// Apply a Butterworth band-pass filter.
    ///
    /// Convenience wrapper that constructs an [`IirFilterDesign`] and
    /// delegates to [`Self::apply_iir_filter`].  The current
    /// implementation uses an approximate placeholder; treat results
    /// as indicative only.
    ///
    /// # Arguments
    /// - `order` – Filter order.
    /// - `low_frequency` – Lower cutoff frequency in Hz; must be > 0.
    /// - `high_frequency` – Upper cutoff frequency in Hz; must be < Nyquist
    ///   and > `low_frequency`.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if the frequency range is
    ///   invalid.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioIirFiltering;
    /// use non_empty_slice::NonEmptyVec;
    /// use std::num::NonZeroUsize;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]).unwrap();
    /// let mut audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// assert!(audio.butterworth_bandpass(NonZeroUsize::new(2).unwrap(), 100.0, 5000.0).is_ok());
    /// ```
    fn butterworth_bandpass(
        &mut self,
        order: NonZeroUsize,
        low_frequency: f64,
        high_frequency: f64,
    ) -> AudioSampleResult<()>;

    /// Apply a Chebyshev Type I filter.
    ///
    /// Chebyshev Type I filters offer sharper roll-off than Butterworth
    /// filters of the same order at the cost of passband ripple.
    ///
    /// > **Note:** This design is not yet implemented.  All calls
    /// > currently return `Err`.
    ///
    /// # Arguments
    /// - `order` – Filter order (number of poles).
    /// - `cutoff_frequency` – Cutoff frequency in Hz.
    /// - `passband_ripple` – Maximum passband ripple in dB.
    /// - `response` – Filter response type (low-pass, high-pass, etc.).
    ///
    /// # Returns
    /// `Ok(())` on success (currently unreachable).
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – always, because the design
    ///   is not yet implemented.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioIirFiltering;
    /// use audio_samples::operations::types::FilterResponse;
    /// use non_empty_slice::NonEmptyVec;
    /// use std::num::NonZeroUsize;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.0, -1.0, 0.0]).unwrap();
    /// let mut audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// audio.chebyshev_i(
    ///     NonZeroUsize::new(4).unwrap(), 1000.0, 0.5, FilterResponse::LowPass,
    /// )?;
    /// # Ok::<(), audio_samples::AudioSampleError>(())
    /// ```
    fn chebyshev_i(
        &mut self,
        order: NonZeroUsize,
        cutoff_frequency: f64,
        passband_ripple: f64,
        response: FilterResponse,
    ) -> AudioSampleResult<()>;

    /// Return the frequency response at the specified frequencies.
    ///
    /// > **Note:** This is a placeholder that returns a flat magnitude-1,
    /// > phase-0 response regardless of the audio content.  A complete
    /// > implementation would store and query the last-applied filter.
    ///
    /// # Arguments
    /// - `frequencies` – Frequencies in Hz at which to compute the response.
    ///
    /// # Returns
    /// `Ok((magnitudes, phases))` where both vectors have the same
    /// length as `frequencies`.  Currently `magnitudes` is all 1.0 and
    /// `phases` is all 0.0.
    ///
    /// # Errors
    /// May return errors from underlying filter analysis (placeholder currently does not error).
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioIirFiltering;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.0, -1.0]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let freqs = [100.0, 500.0, 1000.0];
    /// let (mag, phase) = audio.frequency_response(&freqs).unwrap();
    /// assert_eq!(mag.len(), 3);
    /// assert_eq!(phase, vec![0.0, 0.0, 0.0]);
    /// ```
    fn frequency_response(&self, frequencies: &[f64]) -> AudioSampleResult<(Vec<f64>, Vec<f64>)>;
}

/// Parametric equalization operations.
///
/// # Purpose
///
/// Multi-band parametric EQ with independent control over centre frequency,
/// gain, and bandwidth (Q) per band.  All seven standard band types are
/// supported: peak, low shelf, high shelf, low pass, high pass, band pass,
/// and band stop.
///
/// # Intended Usage
///
/// Use this trait to shape the tonal balance of audio — for example to boost
/// presence in a vocal, to cut rumble below 80 Hz, or to compare frequency
/// responses programmatically.  Build a [`ParametricEq`] from one or more
/// [`EqBand`] entries and apply it with [`apply_parametric_eq`].
///
/// # Invariants
///
/// EQ is applied independently to each channel.  The frequency response is
/// computed from the audio's sample rate at call time; the same `EqBand`
/// parameters produce different digital coefficients at different sample rates.
///
/// [`apply_parametric_eq`]: AudioParametricEq::apply_parametric_eq
#[cfg(feature = "parametric-eq")]
pub trait AudioParametricEq: AudioChannelOps
where
    Self::Sample: StandardSample,
{
    /// Applies a multi-band parametric EQ to the signal.
    ///
    /// Each enabled band in `eq` is applied in sequence using the RBJ biquad filter
    /// coefficients appropriate for its type. An optional output gain is applied after
    /// all bands. If the EQ is bypassed, the signal is returned unchanged.
    ///
    /// # Arguments
    ///
    /// - `eq` – The parametric EQ configuration. All enabled bands are validated before
    ///   processing begins.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if any enabled band fails validation
    /// (e.g. frequency above the Nyquist limit, Q factor ≤ 0, or gain out of range).
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioParametricEq, sample_rate};
    /// use audio_samples::operations::types::{ParametricEq, EqBand};
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_elem(512, 0.5f32);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    ///
    /// let mut eq = ParametricEq::new();
    /// eq.add_band(EqBand::peak(1000.0, 3.0, 2.0));
    /// audio.apply_parametric_eq(&eq).unwrap();
    /// ```
    fn apply_parametric_eq(&mut self, eq: &ParametricEq) -> AudioSampleResult<()>;

    /// Applies a single EQ band filter to the signal.
    ///
    /// Designs and applies one biquad filter for the given [`EqBand`] using RBJ cookbook
    /// formulas. All seven band types are supported: `Peak`, `LowShelf`, `HighShelf`,
    /// `LowPass`, `HighPass`, `BandPass`, and `BandStop`. Multi-channel audio is
    /// processed per channel with filter state reset between channels.
    ///
    /// If the band is disabled, the signal is returned unchanged.
    ///
    /// # Arguments
    ///
    /// - `band` – The EQ band to apply, including its type, frequency, gain, and Q factor.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Parameter] if band validation fails (frequency above Nyquist,
    ///   Q ≤ 0, or gain out of range).
    /// - [crate::AudioSampleError::Layout] if the underlying sample buffer is non-contiguous.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioParametricEq, sample_rate};
    /// use audio_samples::operations::types::EqBand;
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_elem(512, 0.5f32);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Boost at 2 kHz by 4 dB with Q of 1.5
    /// audio.apply_eq_band(&EqBand::peak(2000.0, 4.0, 1.5)).unwrap();
    /// ```
    fn apply_eq_band(&mut self, band: &EqBand) -> AudioSampleResult<()>;

    /// Applies a peak (or notch) filter at the specified centre frequency.
    ///
    /// A peak filter boosts or cuts a band of frequencies centred around `frequency`.
    /// Positive `gain_db` creates a boost (peak); negative creates a cut (notch). The
    /// Q factor controls bandwidth: higher Q values produce a narrower effect.
    ///
    /// # Arguments
    ///
    /// - `frequency` – Centre frequency in Hz. Must be in `(0, Nyquist)`.
    /// - `gain_db` – Gain in dB. Positive boosts, negative cuts.
    /// - `q_factor` – Quality factor controlling bandwidth. Must be > 0.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if `frequency`, `gain_db`, or `q_factor`
    /// fail band validation.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioParametricEq, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_elem(512, 0.5f32);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Boost at 880 Hz by 6 dB with Q of 2.0
    /// audio.apply_peak_filter(880.0, 6.0, 2.0).unwrap();
    /// ```
    fn apply_peak_filter(
        &mut self,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
    ) -> AudioSampleResult<()>;

    /// Applies a low shelf filter that boosts or cuts frequencies below `frequency`.
    ///
    /// All frequencies below the corner frequency receive approximately `gain_db` of
    /// boost or cut, with a transition region controlled by `q_factor`. The shelf levels
    /// off smoothly as frequency approaches zero.
    ///
    /// # Arguments
    ///
    /// - `frequency` – Corner (shelf) frequency in Hz. Must be in `(0, Nyquist)`.
    /// - `gain_db` – Shelf gain in dB. Positive boosts, negative cuts.
    /// - `q_factor` – Shelf slope control. `0.707` gives a maximally flat shelf.
    ///   Must be > 0.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if any parameter fails band validation.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioParametricEq, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_elem(512, 0.5f32);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Cut -3 dB below 200 Hz
    /// audio.apply_low_shelf(200.0, -3.0, 0.707).unwrap();
    /// ```
    fn apply_low_shelf(
        &mut self,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
    ) -> AudioSampleResult<()>;

    /// Applies a high shelf filter that boosts or cuts frequencies above `frequency`.
    ///
    /// All frequencies above the corner frequency receive approximately `gain_db` of
    /// boost or cut, with a transition region controlled by `q_factor`. The shelf levels
    /// off smoothly as frequency approaches the Nyquist limit.
    ///
    /// # Arguments
    ///
    /// - `frequency` – Corner (shelf) frequency in Hz. Must be in `(0, Nyquist)`.
    /// - `gain_db` – Shelf gain in dB. Positive boosts, negative cuts.
    /// - `q_factor` – Shelf slope control. `0.707` gives a maximally flat shelf.
    ///   Must be > 0.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if any parameter fails band validation.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioParametricEq, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_elem(512, 0.5f32);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Boost +4 dB above 8 kHz
    /// audio.apply_high_shelf(8000.0, 4.0, 0.707).unwrap();
    /// ```
    fn apply_high_shelf(
        &mut self,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
    ) -> AudioSampleResult<()>;

    /// Applies a three-band EQ (low shelf, mid peak, high shelf) in a single call.
    ///
    /// Constructs a [`ParametricEq`] with three bands and applies it:
    /// - A low shelf affecting frequencies below `low_freq`.
    /// - A peak filter centred at `mid_freq`.
    /// - A high shelf affecting frequencies above `high_freq`.
    ///
    /// This mirrors the EQ section found on most mixers and channel strips.
    ///
    /// # Arguments
    ///
    /// - `low_freq` – Low shelf corner frequency in Hz.
    /// - `low_gain` – Low shelf gain in dB.
    /// - `mid_freq` – Mid peak centre frequency in Hz.
    /// - `mid_gain` – Mid peak gain in dB.
    /// - `mid_q` – Mid peak Q factor.
    /// - `high_freq` – High shelf corner frequency in Hz.
    /// - `high_gain` – High shelf gain in dB.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if any band fails validation
    /// (e.g. frequency above Nyquist or Q ≤ 0).
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioParametricEq, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_elem(512, 0.5f32);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Low shelf -2 dB at 200 Hz, mid peak +3 dB at 1 kHz (Q=2), high shelf +1 dB at 4 kHz
    /// audio.apply_three_band_eq(200.0, -2.0, 1000.0, 3.0, 2.0, 4000.0, 1.0).unwrap();
    /// ```
    fn apply_three_band_eq(
        &mut self,
        low_freq: f64,
        low_gain: f64,
        mid_freq: f64,
        mid_gain: f64,
        mid_q: f64,
        high_freq: f64,
        high_gain: f64,
    ) -> AudioSampleResult<()>;

    /// Computes the combined magnitude and phase response of a parametric EQ.
    ///
    /// Evaluates each enabled band's biquad filter at every frequency in `frequencies`
    /// and combines the results: magnitudes are multiplied and phases are summed. The
    /// EQ's output gain is applied to the combined magnitude. Disabled bands are skipped.
    ///
    /// This method does not modify the audio signal; it is purely analytical.
    ///
    /// # Arguments
    ///
    /// - `eq` – The parametric EQ whose frequency response to compute.
    /// - `frequencies` – Frequencies in Hz at which to evaluate the response. An empty
    ///   slice returns empty vectors.
    ///
    /// # Returns
    ///
    /// A tuple `(magnitudes, phases)` where:
    /// - `magnitudes` – Linear magnitude response at each frequency (1.0 = unity gain).
    /// - `phases` – Phase response in radians at each frequency.
    ///
    /// Both vectors have the same length as `frequencies`.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if any enabled band fails to design
    /// a filter (e.g. frequency above the Nyquist limit).
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioParametricEq, sample_rate};
    /// use audio_samples::operations::types::{ParametricEq, EqBand};
    /// use ndarray::Array1;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     Array1::from_elem(512, 0.5f32), sample_rate!(44100),
    /// ).unwrap();
    ///
    /// let mut eq = ParametricEq::new();
    /// eq.add_band(EqBand::peak(1000.0, 6.0, 2.0));
    ///
    /// let freqs = [100.0_f64, 500.0, 1000.0, 5000.0];
    /// let (magnitudes, phases) = audio.eq_frequency_response(&eq, &freqs).unwrap();
    /// assert_eq!(magnitudes.len(), 4);
    /// // At the peak frequency, magnitude should be boosted above unity
    /// assert!(magnitudes[2] > 1.0);
    /// ```
    fn eq_frequency_response(
        &self,
        eq: &ParametricEq,
        frequencies: &[f64],
    ) -> AudioSampleResult<(Vec<f64>, Vec<f64>)>;
}

/// Dynamic range control operations for audio signals.
///
/// # Purpose
///
/// The standard toolkit for controlling loudness variation: compression,
/// limiting, noise gating, and expansion.  All four processors share a
/// threshold-based design and support optional lookahead and side-chain inputs.
///
/// # Intended Usage
///
/// Apply these processors as a final stage in a mastering or broadcast chain
/// to ensure consistent loudness, or as a creative effect.  For quick results
/// use the preset constructors (`CompressorConfig::vocal()`,
/// `LimiterConfig::transparent()`, etc.); for precise control build a config
/// struct directly.
///
/// # Invariants
///
/// All operations modify the signal in place and propagate errors rather than
/// panicking.  Each channel is processed independently from the same
/// configuration unless side-chain routing is enabled.
#[cfg(feature = "dynamic-range")]
pub trait AudioDynamicRange: AudioTypeConversion
where
    Self::Sample: StandardSample,
{
    /// Applies compression to reduce the dynamic range of the signal.
    ///
    /// Samples above the threshold are attenuated according to the compression ratio.
    /// Attack and release times control how quickly the compressor responds to level
    /// changes. An optional lookahead delay allows the compressor to anticipate peaks
    /// before they occur. Multi-channel audio is compressed independently per channel.
    ///
    /// # Arguments
    ///
    /// - `config` – Compressor parameters including threshold, ratio, attack, release,
    ///   knee type, makeup gain, lookahead, and detection method. Use preset constructors
    ///   such as [`CompressorConfig::vocal`] or [`CompressorConfig::drum`] for common
    ///   configurations.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if the configuration fails validation
    /// (e.g. threshold above 0 dBFS, ratio below 1.0, or negative time constants).
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::CompressorConfig;
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let config = CompressorConfig::vocal();
    /// audio.apply_compressor(&config).unwrap();
    /// ```
    fn apply_compressor(&mut self, config: &CompressorConfig) -> AudioSampleResult<()>;

    /// Prevents the signal from exceeding a specified ceiling level.
    ///
    /// Applies gain reduction to any samples that breach the ceiling. Lookahead
    /// processing delays the output signal while analysing future samples, allowing gain
    /// reduction to begin before peaks arrive and minimising audible distortion.
    /// Multi-channel audio is limited independently per channel.
    ///
    /// # Arguments
    ///
    /// - `config` – Limiter parameters including ceiling, attack, release, knee type,
    ///   lookahead, and detection method. Use preset constructors such as
    ///   [`LimiterConfig::transparent`] or [`LimiterConfig::mastering`] for common
    ///   configurations.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if the configuration fails validation
    /// (e.g. ceiling above 0 dBFS or invalid time constants).
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::LimiterConfig;
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let config = LimiterConfig::mastering();
    /// audio.apply_limiter(&config).unwrap();
    /// ```
    fn apply_limiter(&mut self, config: &LimiterConfig) -> AudioSampleResult<()>;

    /// Applies compression driven by an external sidechain signal.
    ///
    /// The gain reduction is determined by the level of `sidechain_signal` rather than
    /// the main audio, allowing one signal to control the dynamics of another. A common
    /// use case is ducking: a voice track causes background music to decrease in level
    /// whenever speech is present.
    ///
    /// Only mono-to-mono sidechain processing is currently supported. Multi-channel
    /// combinations return an error.
    ///
    /// # Arguments
    ///
    /// - `config` – Compressor configuration. Sidechain processing must be enabled;
    ///   call `config.side_chain.enable()` before passing the config to this method.
    /// - `sidechain_signal` – External control signal. Must be the same length as the
    ///   main signal.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The main audio is modified in place.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Parameter] if sidechain is not enabled in `config`.
    /// - [crate::AudioSampleError::Parameter] if the main and sidechain signals have
    ///   different lengths.
    /// - [crate::AudioSampleError::Parameter] if either signal is multi-channel (not yet
    ///   supported).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::CompressorConfig;
    /// use ndarray::Array1;
    ///
    /// let mut audio = AudioSamples::new_mono(
    ///     Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]),
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let sidechain = AudioSamples::new_mono(
    ///     Array1::from_vec(vec![0.0f32, 1.0, 0.0, 1.0, 0.0]),
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let mut config = CompressorConfig::new();
    /// config.side_chain.enable();
    /// audio.apply_compressor_sidechain(&config, &sidechain).unwrap();
    /// ```
    fn apply_compressor_sidechain(
        &mut self,
        config: &CompressorConfig,
        sidechain_signal: &Self,
    ) -> AudioSampleResult<()>;

    /// Applies limiting driven by an external sidechain signal.
    ///
    /// The gain reduction ceiling is enforced based on the level of `sidechain_signal`
    /// rather than the main audio. Useful for frequency-selective limiting where a
    /// filtered copy of the signal controls gain reduction.
    ///
    /// Only mono-to-mono sidechain processing is currently supported. Multi-channel
    /// combinations return an error.
    ///
    /// # Arguments
    ///
    /// - `config` – Limiter configuration. Sidechain processing must be enabled;
    ///   call `config.side_chain.enable()` before passing the config to this method.
    /// - `sidechain_signal` – External control signal. Must be the same length as the
    ///   main signal.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The main audio is modified in place.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Parameter] if sidechain is not enabled in `config`.
    /// - [crate::AudioSampleError::Parameter] if the main and sidechain signals have
    ///   different lengths.
    /// - [crate::AudioSampleError::Parameter] if either signal is multi-channel (not yet
    ///   supported).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::LimiterConfig;
    /// use ndarray::Array1;
    ///
    /// let mut audio = AudioSamples::new_mono(
    ///     Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]),
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let sidechain = AudioSamples::new_mono(
    ///     Array1::from_vec(vec![0.0f32, 1.0, 0.0, 1.0, 0.0]),
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let mut config = LimiterConfig::default();
    /// config.side_chain.enable();
    /// audio.apply_limiter_sidechain(&config, &sidechain).unwrap();
    /// ```
    fn apply_limiter_sidechain(
        &mut self,
        config: &LimiterConfig,
        sidechain_signal: &Self,
    ) -> AudioSampleResult<()>;

    /// Computes the static compression input-output curve for given input levels.
    ///
    /// Maps each input level in `input_levels_db` through the compressor's static gain
    /// characteristic (threshold, ratio, knee) plus makeup gain, returning the resulting
    /// output level in dBFS for each input. Does not use time-varying envelope following —
    /// the result depends only on the static transfer function.
    ///
    /// Useful for visualising and verifying compressor behaviour without processing actual
    /// audio samples.
    ///
    /// # Arguments
    ///
    /// - `config` – Compressor configuration parameters.
    /// - `input_levels_db` – Non-empty slice of input levels in dBFS to evaluate.
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` of output levels in dBFS, one per entry in `input_levels_db`,
    /// in the same order.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration parameters are invalid.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::CompressorConfig;
    /// use non_empty_slice::NonEmptySlice;
    /// use ndarray::Array1;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     Array1::from_elem(10, 0.5f32), sample_rate!(44100),
    /// ).unwrap();
    /// let config = CompressorConfig::new();
    /// let levels = [-40.0f64, -20.0, -12.0, 0.0];
    /// let curve = audio.get_compression_curve(
    ///     &config,
    ///     NonEmptySlice::new(&levels).unwrap(),
    /// ).unwrap();
    /// assert_eq!(curve.len(), 4);
    /// // Levels above the threshold are reduced (output < input)
    /// assert!(curve[3] < 0.0);
    /// ```
    fn get_compression_curve(
        &self,
        config: &CompressorConfig,
        input_levels_db: &NonEmptySlice<f64>,
    ) -> AudioSampleResult<Vec<f64>>;

    /// Returns the per-sample gain reduction that would be applied by the compressor.
    ///
    /// Passes the audio through the envelope follower and compression gain calculation,
    /// collecting the gain reduction (in dB) at every sample without modifying the
    /// signal. For multi-channel audio, only the first channel is analysed.
    ///
    /// Useful for metering, visualising, and analysing compressor activity.
    ///
    /// # Arguments
    ///
    /// - `config` – Compressor configuration parameters.
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` of gain reduction values in dB (each value ≥ 0.0), one per sample
    /// in the signal (or first channel for multi-channel audio).
    ///
    /// # Errors
    ///
    /// Returns an error if configuration parameters are invalid or signal processing fails.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::CompressorConfig;
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let config = CompressorConfig::new();
    /// let reductions = audio.get_gain_reduction(&config).unwrap();
    /// assert_eq!(reductions.len(), 5);
    /// assert!(reductions.iter().all(|&r| r >= 0.0));
    /// ```
    fn get_gain_reduction(&self, config: &CompressorConfig) -> AudioSampleResult<Vec<f64>>;

    /// Attenuates the signal when it falls below a threshold (noise gate).
    ///
    /// A gate mutes or reduces quiet passages — typically background noise, room tone,
    /// or bleed — while leaving louder content unaffected. Signals below `threshold_db`
    /// are attenuated by the given ratio; signals above are passed through unchanged.
    /// Peak detection is always used for gate processing.
    ///
    /// # Arguments
    ///
    /// - `threshold_db` – Gate threshold in dBFS. Signals below this level are attenuated.
    /// - `ratio` – Attenuation ratio below the threshold. Higher values produce more
    ///   aggressive gating; values near 1.0 approach unity gain.
    /// - `attack_ms` – Attack time in milliseconds. Controls how quickly the gate opens
    ///   when the signal rises above the threshold.
    /// - `release_ms` – Release time in milliseconds. Controls how quickly the gate
    ///   closes when the signal falls below the threshold.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Currently always returns `Ok`. Future versions may validate parameter ranges.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_vec(vec![0.001f32, 0.8, 0.002, 0.9, 0.001]);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Gate at -20 dBFS with 10:1 ratio
    /// audio.apply_gate(-20.0, 10.0, 1.0, 10.0).unwrap();
    /// ```
    fn apply_gate(
        &mut self,
        threshold_db: f64,
        ratio: f64,
        attack_ms: f64,
        release_ms: f64,
    ) -> AudioSampleResult<()>;

    /// Increases dynamic range by expanding signals below a threshold.
    ///
    /// An expander is the complement of compression: signals below `threshold_db` are
    /// attenuated by an amount that grows with distance from the threshold, making quiet
    /// passages quieter while leaving loud passages unchanged. RMS detection is always
    /// used for expansion.
    ///
    /// # Arguments
    ///
    /// - `threshold_db` – Expansion threshold in dBFS. Signals below this level are
    ///   attenuated.
    /// - `ratio` – Expansion ratio. Values greater than `1.0` produce increasing
    ///   attenuation the further the signal falls below the threshold.
    /// - `attack_ms` – Attack time in milliseconds.
    /// - `release_ms` – Release time in milliseconds.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Currently always returns `Ok`. Future versions may validate parameter ranges.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Expand at -20 dBFS with 2:1 ratio
    /// audio.apply_expander(-20.0, 2.0, 1.0, 10.0).unwrap();
    /// ```
    fn apply_expander(
        &mut self,
        threshold_db: f64,
        ratio: f64,
        attack_ms: f64,
        release_ms: f64,
    ) -> AudioSampleResult<()>;
}

/// Time-domain editing operations for [`AudioSamples`].
///
/// # Purpose
///
/// Provides structural manipulation of audio buffers: reversing, trimming,
/// padding, splitting, concatenating, mixing, fading, repeating, and silence
/// removal.  Also includes data-augmentation via [`perturb`].
///
/// # Intended Usage
///
/// Use this trait to construct or reshape audio segments — for example to
/// trim silence from the edges of a recording, to concatenate a sequence of
/// clips, or to build training data through random perturbation.  Most
/// operations return a new owned [`AudioSamples`]; mutations are performed
/// in place only where the method name ends with `_in_place`.
///
/// # Invariants
///
/// Operations that combine two signals (mix, concatenate, stack) validate that
/// sample rates and channel counts match before proceeding; mismatches return
/// [crate::AudioSampleError::Parameter].
///
/// [`perturb`]: AudioEditing::perturb
#[cfg(feature = "editing")]
pub trait AudioEditing: AudioTypeConversion
where
    Self::Sample: StandardSample,
{
    /// Returns a time-reversed copy of the signal.
    ///
    /// All channels are reversed independently.  The sample rate and
    /// channel count are preserved.
    ///
    /// # Returns
    /// A new [`AudioSamples`] with the sample order reversed.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     array![1.0f32, 2.0, 3.0], sample_rate!(44100),
    /// ).unwrap();
    /// let rev = audio.reverse();
    /// assert_eq!(rev[0], 3.0);
    /// assert_eq!(rev[1], 2.0);
    /// assert_eq!(rev[2], 1.0);
    /// ```
    fn reverse<'b>(&self) -> AudioSamples<'b, Self::Sample>
    where
        Self: Sized;

    /// Reverses the sample order in place.
    ///
    /// All channels are reversed independently.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if an internal multi-channel
    ///   array row is not contiguous.
    fn reverse_in_place(&mut self) -> AudioSampleResult<()>
    where
        Self: Sized;

    /// Extracts a segment of audio between two time boundaries.
    ///
    /// Works on both mono and multi-channel audio.  Time values are
    /// converted to sample indices using the signal's sample rate.
    ///
    /// # Arguments
    /// - `start_seconds` — start of the segment in seconds (`>= 0`).
    /// - `end_seconds` — end of the segment in seconds
    ///   (`> start_seconds`, `<= duration`).
    ///
    /// # Returns
    /// A new [`AudioSamples`] containing the requested segment.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `start_seconds < 0`,
    ///   `end_seconds <= start_seconds`, or `end_seconds` exceeds the
    ///   signal duration.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     Array1::<f32>::zeros(100), sample_rate!(100),
    /// ).unwrap();
    /// let trimmed = audio.trim(0.25, 0.75).unwrap();
    /// assert_eq!(trimmed.samples_per_channel().get(), 50);
    /// ```
    fn trim<'b>(
        &self,
        start_seconds: f64,
        end_seconds: f64,
    ) -> AudioSampleResult<AudioSamples<'b, Self::Sample>>;

    /// Adds silence (or a constant value) at the beginning and/or end.
    ///
    /// Works on both mono and multi-channel audio.
    ///
    /// # Arguments
    /// - `pad_start_seconds` — duration of padding prepended, in seconds.
    ///   Must be `>= 0`.
    /// - `pad_end_seconds` — duration of padding appended, in seconds.
    ///   Must be `>= 0`.
    /// - `pad_value` — the sample value used for the padded region
    ///   (typically the zero value of `T` for silence).
    ///
    /// # Returns
    /// A new [`AudioSamples`] with the requested padding applied.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if either padding
    ///   duration is negative.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     Array1::from_elem(5, 1.0f32), sample_rate!(10),
    /// ).unwrap();
    /// // 0.2 s at 10 Hz = 2 samples each side
    /// let padded = audio.pad(0.2, 0.2, 0.0).unwrap();
    /// assert_eq!(padded.samples_per_channel().get(), 9);
    /// assert_eq!(padded[0], 0.0);
    /// assert_eq!(padded[2], 1.0);
    /// ```
    fn pad<'b>(
        &self,
        pad_start_seconds: f64,
        pad_end_seconds: f64,
        pad_value: Self::Sample,
    ) -> AudioSampleResult<AudioSamples<'b, Self::Sample>>;

    /// Pads with a constant value on the right to reach a target sample count.
    ///
    /// If the signal already has `target_num_samples` or more samples per
    /// channel it is returned unchanged (as an owned clone).
    ///
    /// # Arguments
    /// - `target_num_samples` — desired number of samples per channel.
    /// - `pad_value` — the sample value used for the padded region.
    ///
    /// # Returns
    /// A new [`AudioSamples`] with at least `target_num_samples` samples
    /// per channel.
    ///
    /// # Errors
    /// Propagates any error from the underlying [`pad_to_duration`] call.
    ///
    /// [`pad_to_duration`]: Self::pad_to_duration
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     array![1.0f32, 2.0, 3.0], sample_rate!(44100),
    /// ).unwrap();
    /// let padded = audio.pad_samples_right(6, 0.0).unwrap();
    /// assert_eq!(padded.samples_per_channel().get(), 6);
    /// assert_eq!(padded[3], 0.0); // padded region is silent
    /// ```
    fn pad_samples_right<'b>(
        &self,
        target_num_samples: usize,
        pad_value: Self::Sample,
    ) -> AudioSampleResult<AudioSamples<'b, Self::Sample>>;

    /// Pads the signal to reach a target duration.
    ///
    /// Padding is added on the side specified by `pad_side`.  If the
    /// signal is already at least as long as `target_duration_seconds`
    /// it is returned unchanged (as an owned clone).
    ///
    /// # Arguments
    /// - `target_duration_seconds` — desired length in seconds.
    /// - `pad_value` — the sample value used for the padded region.
    /// - `pad_side` — which end to pad ([`PadSide::Left`] or
    ///   [`PadSide::Right`]).
    ///
    /// # Returns
    /// A new [`AudioSamples`] at least `target_duration_seconds` long.
    ///
    /// # Errors
    /// Propagates any error from the underlying [`pad`] call.
    ///
    /// [`pad`]: Self::pad
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use audio_samples::operations::types::PadSide;
    /// use ndarray::Array1;
    ///
    /// // 100 samples at 100 Hz = 1.0 s
    /// let audio = AudioSamples::new_mono(
    ///     Array1::<f32>::ones(100), sample_rate!(100),
    /// ).unwrap();
    /// // Pad to 1.5 s on the right → 50 extra samples of silence
    /// let padded = audio.pad_to_duration(1.5, 0.0, PadSide::Right).unwrap();
    /// assert_eq!(padded.samples_per_channel().get(), 150);
    /// ```
    fn pad_to_duration<'b>(
        &self,
        target_duration_seconds: f64,
        pad_value: Self::Sample,
        pad_side: PadSide,
    ) -> AudioSampleResult<AudioSamples<'b, Self::Sample>>;

    /// Splits the signal into segments of a fixed duration.
    ///
    /// The last segment may be shorter than `segment_duration_seconds`
    /// if the signal does not divide evenly.  Works on both mono and
    /// multi-channel audio.
    ///
    /// # Arguments
    /// - `segment_duration_seconds` — target length of each segment in
    ///   seconds.  Must be `> 0` and must not exceed the signal length.
    ///
    /// # Returns
    /// A vector of segments in chronological order.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the duration is not
    ///   positive or exceeds the signal length.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     Array1::<f32>::zeros(100), sample_rate!(100),
    /// ).unwrap();
    /// let segments = audio.split(0.25).unwrap();
    /// assert_eq!(segments.len(), 4);
    /// assert_eq!(segments[0].samples_per_channel().get(), 25);
    /// ```
    fn split(
        &self,
        segment_duration_seconds: f64,
    ) -> AudioSampleResult<Vec<AudioSamples<'static, Self::Sample>>>;

    /// Joins multiple audio segments end-to-end.
    ///
    /// All segments must share the same sample rate and channel count.
    /// The output preserves the order of the input slice.
    ///
    /// # Arguments
    /// - `segments` — the segments to join, in order.
    ///
    /// # Returns
    /// A single [`AudioSamples`] containing all segments concatenated.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if any segment has a
    ///   different sample rate or channel count from the first.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::array;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let a = AudioSamples::new_mono(array![1.0f32, 2.0], sample_rate!(44100)).unwrap();
    /// let b = AudioSamples::new_mono(array![3.0f32, 4.0], sample_rate!(44100)).unwrap();
    /// let segments = NonEmptyVec::new(vec![a, b]).unwrap();
    /// let joined = AudioSamples::concatenate(&segments).unwrap();
    /// assert_eq!(joined.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    /// ```
    fn concatenate<'b>(
        segments: &'b NonEmptySlice<AudioSamples<'b, Self::Sample>>,
    ) -> AudioSampleResult<AudioSamples<'b, Self::Sample>>;

    /// Joins multiple owned audio segments end-to-end.
    ///
    /// Identical in behaviour to [`concatenate`] but accepts owned
    /// segments rather than borrowed ones.  All segments must share the
    /// same sample rate and channel count.
    ///
    /// # Arguments
    /// - `segments` — the owned segments to join, in order.
    ///
    /// # Returns
    /// A single [`AudioSamples`] containing all segments concatenated.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if any segment has a
    ///   different sample rate or channel count from the first.
    ///
    /// [`concatenate`]: Self::concatenate
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::array;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let a = AudioSamples::new_mono(array![1.0f32, 2.0], sample_rate!(44100)).unwrap();
    /// let b = AudioSamples::new_mono(array![3.0f32, 4.0], sample_rate!(44100)).unwrap();
    /// let segments = NonEmptyVec::new(vec![a, b]).unwrap();
    /// let joined = AudioSamples::concatenate_owned(segments).unwrap();
    /// assert_eq!(joined.samples_per_channel().get(), 4);
    /// ```
    fn concatenate_owned<'b>(
        segments: NonEmptyVec<AudioSamples<'_, Self::Sample>>,
    ) -> AudioSampleResult<AudioSamples<'b, Self::Sample>>;

    /// Produces a weighted sum of multiple audio sources.
    ///
    /// All sources must have the same sample rate, channel count, and
    /// length.  When `weights` is `None`, equal weighting (1 / N per
    /// source) is applied.
    ///
    /// # Arguments
    /// - `sources` — the audio signals to mix.
    /// - `weights` — optional per-source gain factors.  Must have the
    ///   same length as `sources`.
    ///
    /// # Returns
    /// The mixed signal.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the sources differ in
    ///   sample rate, channel count, or length, or if the weights slice
    ///   length does not match the sources.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::Array1;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let a = AudioSamples::new_mono(Array1::from_elem(4, 1.0f32), sample_rate!(44100)).unwrap();
    /// let b = AudioSamples::new_mono(Array1::from_elem(4, 3.0f32), sample_rate!(44100)).unwrap();
    /// let sources = NonEmptyVec::new(vec![a, b]).unwrap();
    /// let mixed = AudioSamples::mix(&sources, None).unwrap();
    /// // Equal weighting: (1.0 + 3.0) / 2 = 2.0
    /// assert_eq!(mixed[0], 2.0);
    /// ```
    fn mix(
        sources: &NonEmptySlice<Self>,
        weights: Option<&NonEmptySlice<f64>>,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// Applies a fade-in envelope to the beginning of the signal.
    ///
    /// The first `duration_seconds` of every channel are multiplied by
    /// an amplitude ramp from 0 to 1 shaped by `curve`.  If
    /// `duration_seconds` exceeds the signal length the entire signal
    /// is faded.
    ///
    /// # Arguments
    /// - `duration_seconds` — fade length in seconds.  Must be `> 0`.
    /// - `curve` — the envelope shape (see [`FadeCurve`]).
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `duration_seconds`
    ///   is not positive.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use audio_samples::operations::types::FadeCurve;
    /// use ndarray::Array1;
    ///
    /// let mut audio = AudioSamples::new_mono(
    ///     Array1::<f32>::ones(100), sample_rate!(100),
    /// ).unwrap();
    /// audio.fade_in(0.5, FadeCurve::Linear).unwrap();
    /// // First sample is at position 0 → gain 0
    /// assert_eq!(audio.as_slice().unwrap()[0], 0.0);
    /// ```
    fn fade_in(&mut self, duration_seconds: f64, curve: FadeCurve) -> AudioSampleResult<()>;

    /// Applies a fade-out envelope to the end of the signal.
    ///
    /// The last `duration_seconds` of every channel are multiplied by
    /// an amplitude ramp from 1 to 0 shaped by `curve`.  If
    /// `duration_seconds` exceeds the signal length the entire signal
    /// is faded.
    ///
    /// # Arguments
    /// - `duration_seconds` — fade length in seconds.  Must be `> 0`.
    /// - `curve` — the envelope shape (see [`FadeCurve`]).
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `duration_seconds`
    ///   is not positive.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use audio_samples::operations::types::FadeCurve;
    /// use ndarray::Array1;
    ///
    /// let mut audio = AudioSamples::new_mono(
    ///     Array1::<f32>::ones(100), sample_rate!(100),
    /// ).unwrap();
    /// audio.fade_out(0.5, FadeCurve::Linear).unwrap();
    /// // Last sample has position 0 in the fade ramp → gain ≈ 0
    /// let last = *audio.as_slice().unwrap().last().unwrap();
    /// assert!(last < 0.1);
    /// ```
    fn fade_out(&mut self, duration_seconds: f64, curve: FadeCurve) -> AudioSampleResult<()>;

    /// Tiles the signal, repeating it end-to-end.
    ///
    /// Works on both mono and multi-channel audio.  A count of 1
    /// returns an owned clone of the original.
    ///
    /// # Arguments
    /// - `count` — number of repetitions.  Must be `>= 1`.
    ///
    /// # Returns
    /// A new [`AudioSamples`] whose length is `original_length × count`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `count` is 0.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     array![1.0f32, 2.0], sample_rate!(44100),
    /// ).unwrap();
    /// let tiled = audio.repeat(3).unwrap();
    /// assert_eq!(tiled.as_slice().unwrap(), &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    /// ```
    fn repeat(&self, count: usize) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// Removes leading and trailing silence from the signal.
    ///
    /// A sample (or, for multi-channel audio, an entire frame) is
    /// considered silent when its absolute value is at or below the
    /// linear equivalent of `threshold_db`.  If the entire signal is
    /// silent a zero-filled signal of the same length is returned.
    ///
    /// # Arguments
    /// - `threshold_db` — silence threshold in dB (typically negative,
    ///   e.g. `-40.0`).  Samples with amplitude at or below
    ///   `10^(threshold_db / 20)` are treated as silence.
    ///
    /// # Returns
    /// A new [`AudioSamples`] with leading and trailing silence removed.
    ///
    /// # Errors
    /// Cannot fail for valid input; errors are propagated from internal
    /// signal construction only.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::array;
    ///
    /// // 3 silent samples, 2 loud samples, 3 silent samples
    /// let data = array![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(8)).unwrap();
    /// let trimmed = audio.trim_silence(-10.0).unwrap();
    /// assert_eq!(trimmed.samples_per_channel().get(), 2);
    /// ```
    fn trim_silence(
        &self,
        threshold_db: f64,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// Returns a perturbed copy of the signal for data augmentation.
    ///
    /// Delegates to [`perturb_in_place`] on an owned clone.  The original signal
    /// is not modified.  Available only when the `random-generation`
    /// feature is enabled.
    ///
    /// # Arguments
    /// - `config` — perturbation configuration (method, parameters, and
    ///   optional deterministic seed). See [`PerturbationConfig`].
    ///
    /// # Returns
    /// A new [`AudioSamples`] with the perturbation applied.
    ///
    /// # Errors
    /// Propagates any error from validation or from the underlying
    /// perturbation method.
    ///
    /// [`perturb_in_place`]: Self::perturb_in_place
    #[cfg(all(feature = "random-generation", feature = "iir-filtering"))]
    fn perturb<'b>(
        &self,
        config: &PerturbationConfig,
    ) -> AudioSampleResult<AudioSamples<'b, Self::Sample>>;

    /// Applies a perturbation to the signal in place.
    ///
    /// Available only when the `random-generation` feature is enabled.
    /// When a seed is set in `config` the operation is fully
    /// deterministic.
    ///
    /// # Arguments
    /// - `config` — perturbation configuration (method, parameters, and
    ///   optional deterministic seed). See [`PerturbationConfig`].
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - Validation errors from [`PerturbationConfig`] (e.g. cutoff
    ///   above Nyquist).
    /// - Errors from the underlying perturbation method (filtering,
    ///   pitch shift, etc.).
    #[cfg(all(feature = "random-generation", feature = "iir-filtering"))]
    fn perturb_in_place(&mut self, config: &PerturbationConfig) -> AudioSampleResult<()>;

    /// Interleaves multiple mono signals into a single multi-channel signal.
    ///
    /// Each source becomes one channel in the output.  If only one source
    /// is provided it is returned unchanged (as an owned clone).
    ///
    /// # Arguments
    /// - `sources` — mono audio signals of equal length and sample rate.
    ///
    /// # Returns
    /// A multi-channel [`AudioSamples`] with one channel per source.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if any source is
    ///   multi-channel, or if the sources differ in sample rate or length.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::array;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let ch1 = AudioSamples::new_mono(array![1.0f32, 2.0], sample_rate!(44100)).unwrap();
    /// let ch2 = AudioSamples::new_mono(array![3.0f32, 4.0], sample_rate!(44100)).unwrap();
    /// let sources = NonEmptyVec::new(vec![ch1, ch2]).unwrap();
    /// let stereo = AudioSamples::stack(&sources).unwrap();
    /// assert_eq!(stereo.num_channels().get(), 2);
    /// ```
    fn stack(
        sources: &NonEmptySlice<Self>,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// Removes all silence regions throughout the signal.
    ///
    /// The signal is scanned for runs of silent samples (amplitude at
    /// or below `threshold_db`).  Any silence run at least
    /// `min_silence_duration_seconds` long is excised; the remaining
    /// non-silent segments are concatenated in order.
    ///
    /// # Arguments
    /// - `threshold_db` — silence threshold in dB (typically negative).
    /// - `min_silence_duration_seconds` — minimum length of a silence
    ///   region (in seconds) before it is removed.
    ///
    /// # Returns
    /// A new [`AudioSamples`] with qualifying silence regions removed.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the entire signal
    ///   consists of silence, resulting in an empty output.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::array;
    ///
    /// // loud(2) – silence(4) – loud(2) at 10 Hz
    /// let data = array![1.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(10)).unwrap();
    /// // Remove silence runs >= 0.3 s (3 samples at 10 Hz)
    /// let trimmed = audio.trim_all_silence(-10.0, 0.3).unwrap();
    /// assert_eq!(trimmed.samples_per_channel().get(), 4);
    /// ```
    fn trim_all_silence(
        &self,
        threshold_db: f64,
        min_silence_duration_seconds: f64,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;
}

/// Channel manipulation and layout conversion operations.
///
/// # Purpose
///
/// Provides everything needed to change the channel structure of an
/// [`AudioSamples`] buffer: mono/stereo conversion, channel duplication,
/// per-channel extraction, panning, balance adjustment, interleaving, and
/// element-wise channel transforms.
///
/// # Intended Usage
///
/// Use this trait to prepare audio for a pipeline that has different channel
/// requirements — for example downmixing a recording to mono before pitch
/// analysis, or interleaving planar channels before writing to a file.
///
/// # Invariants
///
/// Operations that reference a specific channel index validate that the index
/// is within bounds and return [crate::AudioSampleError::Parameter] if not.
/// Interleaving and de-interleaving are inverse operations: round-tripping
/// through both preserves all sample values.
#[cfg(feature = "channels")]
pub trait AudioChannelOps: AudioTypeConversion
where
    Self::Sample: StandardSample,
{
    /// Convert multi-channel audio to mono using the specified method.
    ///
    /// If the audio is already mono, a clone is returned without
    /// applying the conversion method.
    ///
    /// # Arguments
    /// - `method` – The downmix strategy.  Available variants:
    ///   - `Average` – arithmetic mean of all channels.
    ///   - `Left` – channel 0 only.
    ///   - `Right` – channel 1 only.
    ///   - `Weighted(weights)` – user-supplied per-channel weights.
    ///   - `Center` – surround center channel (≥ 6 ch), or average
    ///     of left and right for stereo.
    ///
    /// # Returns
    /// An owned mono [`AudioSamples`] at the same sample rate.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if `method` is `Weighted`
    ///   and the weights vector length does not match the channel count.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use audio_samples::operations::types::MonoConversionMethod;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5, -1.0]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let stereo = audio.duplicate_to_channels(2).unwrap();
    /// let mono = stereo.to_mono(MonoConversionMethod::Average).unwrap();
    /// assert!(mono.is_mono());
    /// ```
    fn to_mono(
        &self,
        method: MonoConversionMethod,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// Convert audio to a different stereo.
    ///
    /// Behaviour depends on the chosen method:
    /// - `Duplicate` – copies mono audio to both channels; multi-channel
    ///   input is returned unchanged.
    /// - `Pan(value)` – applies equal-power panning to mono input.
    ///   The value is clamped to \[-1, 1\]: −1 is hard left, 0 is
    ///   centre, +1 is hard right.  Multi-channel input is returned
    ///   unchanged.
    /// - `Left` – extracts channel 0 as a mono signal.
    /// - `Right` – extracts channel 1 as a mono signal.
    ///
    /// # Arguments
    /// - `method` – The conversion strategy.
    ///
    /// # Returns
    /// An owned [`AudioSamples`] at the same sample rate.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if `Left` or `Right` is
    ///   chosen and the requested channel does not exist.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use audio_samples::operations::types::StereoConversionMethod;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5, -1.0]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let stereo = audio.to_stereo(StereoConversionMethod::Duplicate).unwrap();
    /// assert_eq!(stereo.num_channels().get(), 2);
    /// ```
    fn to_stereo(
        &self,
        method: StereoConversionMethod,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// Duplicate audio into an n-channel signal.
    ///
    /// For mono input the single channel is replicated into all output
    /// channels.  For multi-channel input only channel 0 is used as the
    /// source; the remaining input channels are ignored.
    ///
    /// # Arguments
    /// - `n_channels` – Target channel count; must be ≥ 1.
    ///
    /// # Returns
    /// An owned [`AudioSamples`] with `n_channels` identical channels
    /// at the same sample rate.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if `n_channels` is 0.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let quad = audio.duplicate_to_channels(4).unwrap();
    /// assert_eq!(quad.num_channels().get(), 4);
    /// ```
    fn duplicate_to_channels(
        &self,
        n_channels: usize,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// Extract a single channel as an owned mono signal.
    ///
    /// If the audio is already mono, a clone is returned (only
    /// `channel_index` 0 is valid in that case).
    ///
    /// # Arguments
    /// - `channel_index` – Zero-based index of the channel to extract;
    ///   must be less than the number of channels.
    ///
    /// # Returns
    /// An owned mono [`AudioSamples`] at the same sample rate.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if `channel_index` is ≥ the
    ///   number of channels.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let stereo = audio.duplicate_to_channels(2).unwrap();
    /// let ch0 = stereo.extract_channel(0).unwrap();
    /// assert!(ch0.is_mono());
    /// ```
    fn extract_channel(
        &self,
        channel_index: u32,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// Borrow a single channel as a zero-copy view.
    ///
    /// The returned [`AudioSamples`] shares memory with `self`; its
    /// lifetime is tied to the borrow of `self`.
    ///
    /// # Arguments
    /// - `channel_index` – Zero-based index of the channel to borrow;
    ///   must be less than the number of channels.
    ///
    /// # Returns
    /// A borrowed mono [`AudioSamples`] at the same sample rate.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if `channel_index` is ≥ the
    ///   number of channels.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let stereo = audio.duplicate_to_channels(2).unwrap();
    /// let ch1 = stereo.borrow_channel(1).unwrap();
    /// assert!(ch1.is_mono());
    /// ```
    fn borrow_channel(
        &self,
        channel_index: u32,
    ) -> AudioSampleResult<AudioSamples<'_, Self::Sample>>;

    /// Swap two channels in place.
    ///
    /// For mono audio the only valid index is 0, making the swap a
    /// no-op.  Passing any other index for mono audio will trigger
    /// the out-of-range error below.
    ///
    /// # Arguments
    /// - `channel1` – Zero-based index of the first channel.
    /// - `channel2` – Zero-based index of the second channel.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if either index is ≥ the
    ///   number of channels.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let mut stereo = audio.duplicate_to_channels(2).unwrap();
    /// assert!(stereo.swap_channels(0, 1).is_ok());
    /// ```
    fn swap_channels(&mut self, channel1: u32, channel2: u32) -> AudioSampleResult<()>;

    /// Apply linear panning to a stereo signal in place.
    ///
    /// The left channel is scaled by `1 − pan_value` and the right
    /// channel by `1 + pan_value`, after clamping `pan_value` to
    /// \[-1, 1\].  A value of 0 leaves both channels unchanged;
    /// −1 silences the right channel; +1 silences the left channel.
    ///
    /// # Arguments
    /// - `pan_value` – Panning position in the range \[-1, 1\].
    ///   Values outside this range are clamped.  −1 is hard left,
    ///   0 is centre, +1 is hard right.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if the audio is mono or
    ///   has a channel count other than 2.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let mut stereo = audio.duplicate_to_channels(2).unwrap();
    /// stereo.pan(0.5).unwrap();
    /// ```
    fn pan(&mut self, pan_value: f64) -> AudioSampleResult<()>;

    /// Adjust the left/right balance of a stereo signal in place.
    ///
    /// The formula applied is identical to [`AudioChannelOps::pan`]:
    /// the left channel is scaled by `1 − balance` and the right
    /// channel by `1 + balance`, after clamping to \[-1, 1\].
    ///
    /// # Arguments
    /// - `balance` – Balance position in the range \[-1, 1\].
    ///   Values outside this range are clamped.  −1 is full left,
    ///   0 is centre, +1 is full right.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if the audio is mono or
    ///   has a channel count other than 2.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let mut stereo = audio.duplicate_to_channels(2).unwrap();
    /// stereo.balance(-0.3).unwrap();
    /// ```
    fn balance(&mut self, balance: f64) -> AudioSampleResult<()>;

    /// Apply a closure to every sample in a single channel.
    ///
    /// For mono audio the closure is applied to the single channel
    /// and `channel_index` is ignored.  For multi-channel audio
    /// `channel_index` is validated and must be less than the number
    /// of channels.
    ///
    /// # Arguments
    /// - `channel_index` – Zero-based index of the target channel.
    ///   Ignored for mono audio.
    /// - `func` – A closure that maps each sample to a new value.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if the audio is
    ///   multi-channel and `channel_index` is ≥ the number of
    ///   channels.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let mut stereo = audio.duplicate_to_channels(2).unwrap();
    /// stereo.apply_to_channel(1, |s| s * 0.5).unwrap();
    /// ```
    fn apply_to_channel<F>(&mut self, channel_index: u32, func: F) -> AudioSampleResult<()>
    where
        F: FnMut(Self::Sample) -> Self::Sample;

    /// Combine multiple mono signals into a single multi-channel signal.
    ///
    /// All input signals must have the same number of samples.  The
    /// first signal becomes channel 0, the second becomes channel 1,
    /// and so on.  The output sample rate is taken from the first
    /// input signal.
    ///
    /// # Arguments
    /// - `channels` – A non-empty slice of mono [`AudioSamples`].
    ///   All elements must share the same sample count.
    ///
    /// # Returns
    /// An owned multi-channel [`AudioSamples`] with one channel per
    /// input signal.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if the input signals do
    ///   not all have the same sample count.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use non_empty_slice::{NonEmptySlice, NonEmptyVec};
    ///
    /// let s0 = NonEmptyVec::new(vec![1.0f32, 0.5]).unwrap();
    /// let s1 = NonEmptyVec::new(vec![-1.0f32, -0.5]).unwrap();
    /// let ch0: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(s0, sample_rate!(44100));
    /// let ch1: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(s1, sample_rate!(44100));
    /// let arr = [ch0, ch1];
    /// let channels = NonEmptySlice::new(&arr).unwrap();
    /// let stereo = <AudioSamples<'_, f32> as AudioChannelOps>::interleave_channels(channels).unwrap();
    /// assert_eq!(stereo.num_channels().get(), 2);
    /// ```
    fn interleave_channels(
        channels: &NonEmptySlice<AudioSamples<'_, Self::Sample>>,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// Split a multi-channel signal into individual mono signals.
    ///
    /// Each output signal contains the samples from a single input
    /// channel.  Channel 0 becomes element 0, channel 1 becomes
    /// element 1, and so on.  For mono input a single-element vector
    /// is returned.
    ///
    /// # Returns
    /// A vector of owned mono [`AudioSamples`], one per input channel.
    ///
    /// # Errors
    /// Returns an error if channel separation fails or memory allocation fails.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let stereo = audio.duplicate_to_channels(2).unwrap();
    /// let channels = stereo.deinterleave_channels().unwrap();
    /// assert_eq!(channels.len(), 2);
    /// assert!(channels[0].is_mono());
    /// ```
    fn deinterleave_channels(&self) -> AudioSampleResult<Vec<AudioSamples<'static, Self::Sample>>>;
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
/// let audio = AudioSamples::new_mono(samples, 44100).unwrap();
/// let config = HpssConfig::new();
///
/// // Separate into harmonic and percussive components
/// let (harmonic, percussive) = audio.hpss(&config)?;
///
/// // Process components separately
/// let drums_isolated = percussive.normalize(-1.0, 1.0, NormalizationMethod::Peak)?;
/// let melody_isolated = harmonic.normalize(-1.0, 1.0, NormalizationMethod::Peak)?;
/// ```
/// Audio source separation using spectral decomposition techniques.
///
/// # Purpose
///
/// Separates an audio signal into its constituent perceptual components.
/// Currently provides Harmonic–Percussive Source Separation (HPSS), which
/// splits a signal into a tonal/sustained layer and a transient/percussive layer.
///
/// # Intended Usage
///
/// Use this trait when downstream processing needs to operate on only one
/// component type — for example to apply pitch correction only to the harmonic
/// layer, or to analyse rhythm independently of melody.
///
/// # Invariants
///
/// The harmonic and percussive components sum to the original signal (within
/// floating-point precision) when soft masking is disabled.  Both outputs carry
/// the same sample rate and channel count as the input.
#[cfg(feature = "decomposition")]
pub trait AudioDecomposition: AudioTransforms
where
    Self::Sample: StandardSample,
{
    /// Separates audio into harmonic and percussive components (HPSS).
    ///
    /// Harmonic–Percussive Source Separation applies median filtering to the
    /// STFT magnitude spectrogram along the time axis to enhance tonal content
    /// and along the frequency axis to enhance transient content.  Binary or
    /// soft masks are derived from the filtered spectrograms and applied to
    /// the original STFT; the resulting masked spectra are inverted to produce
    /// two time-domain signals.
    ///
    /// Reference:
    /// Fitzgerald, D. (2010). "Harmonic/percussive separation using median filtering".
    /// Müller, M. (2015). *Fundamentals of Music Processing*, Section 8.4.
    ///
    /// # Arguments
    ///
    /// - `config` – HPSS parameters: window size, hop size, harmonic and percussive
    ///   median-filter lengths, and mask softness (`HpssConfig::musical()` for a
    ///   good default).
    ///
    /// # Returns
    ///
    /// A `(harmonic, percussive)` tuple.  Both signals have the same sample rate
    /// and channel count as the input.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Parameter] if `config` fields are invalid or the
    ///   signal is shorter than the specified STFT window.
    /// - [crate::AudioSampleError::Layout] if internal STFT or ISTFT operations fail.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use audio_samples::{AudioSamples, sample_rate, sine_wave};
    /// use audio_samples::operations::hpss::HpssConfig;
    /// use audio_samples::operations::traits::AudioDecomposition;
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_secs(1), sample_rate!(44100), 1.0);
    /// let config = HpssConfig::musical();
    /// let (harmonic, percussive) = audio.hpss(&config).unwrap();
    /// assert_eq!(harmonic.sample_rate(), audio.sample_rate());
    /// assert_eq!(percussive.sample_rate(), audio.sample_rate());
    /// ```
    #[allow(clippy::type_complexity)]
    fn hpss(
        &self,
        config: &HpssConfig,
    ) -> AudioSampleResult<(
        AudioSamples<'static, Self::Sample>,
        AudioSamples<'static, Self::Sample>,
    )>;
}

/// Onset detection and spectral analysis operations.
///
/// # Purpose
///
/// Locates the moments where a new musical event begins — note attacks,
/// drum hits, chord changes — and exposes the intermediate spectral
/// representations that drive the detection.
///
/// # Intended Usage
///
/// Use `detect_onsets` or `detect_onsets_spectral_flux` when you need
/// a list of onset timestamps.  Use `onset_detection_function` or
/// `spectral_flux` when you need the raw activation curve for downstream
/// analysis (e.g. beat induction or custom peak-picking).
/// `complex_onset_detection` and its helpers provide a phase-sensitive
/// alternative useful for polyphonic or sustained content.
///
/// # Invariants
///
/// All returned time vectors are sorted in ascending order and expressed in
/// seconds relative to the start of the signal.  The detection-function and
/// timestamp vectors returned by tuple methods always have the same length.
#[cfg(feature = "onset-detection")]
pub trait AudioOnsetDetection: AudioTransforms
where
    Self::Sample: StandardSample,
{
    /// Detects onset times in the audio signal using spectral flux.
    ///
    /// Computes an onset detection function from the signal and applies
    /// peak-picking to find the frames where new events begin.  The
    /// exact pipeline is controlled by `config`.
    ///
    /// # Arguments
    ///
    /// - `config` – Onset detection parameters: STFT settings, flux method,
    ///   peak-picking thresholds, and minimum inter-onset interval.
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` of onset times in seconds, sorted ascending.
    /// Returns an empty `Vec` if no onsets are found.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if `config` fields are invalid.
    /// Returns [crate::AudioSampleError::Processing] if the STFT computation fails.
    fn detect_onsets(&self, config: &OnsetDetectionConfig) -> AudioSampleResult<Vec<f64>>;

    /// Computes the onset detection function and its time axis.
    ///
    /// Returns the raw activation curve before peak-picking, together with
    /// the corresponding frame timestamps.  Useful when you want to inspect
    /// the ODF, apply custom thresholding, or feed it into a beat tracker.
    ///
    /// # Arguments
    ///
    /// - `config` – Onset detection parameters controlling the spectral
    ///   analysis and flux computation.
    ///
    /// # Returns
    ///
    /// A `(odf_values, timestamps)` tuple:
    /// - `odf_values` – One activation value per analysis frame.
    /// - `timestamps` – Corresponding frame centre times in seconds.
    ///   Both vectors have the same length.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if `config` fields are invalid.
    /// Returns [crate::AudioSampleError::Processing] if the STFT computation fails.
    fn onset_detection_function(
        &self,
        config: &OnsetDetectionConfig,
    ) -> AudioSampleResult<(NonEmptyVec<f64>, NonEmptyVec<f64>)>;

    /// Detects onset times using the spectral-flux method.
    ///
    /// An alternative entry point that accepts a [`SpectralFluxConfig`]
    /// directly, exposing finer control over the CQT analysis and flux
    /// accumulation than the higher-level [`detect_onsets`].
    ///
    /// # Arguments
    ///
    /// - `config` – Spectral flux parameters: CQT settings, flux method,
    ///   and peak-picking thresholds.
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` of onset times in seconds, sorted ascending.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if `config` fields are invalid.
    /// Returns [crate::AudioSampleError::Processing] if the CQT computation fails.
    ///
    /// [`detect_onsets`]: AudioOnsetDetection::detect_onsets
    fn detect_onsets_spectral_flux(
        &self,
        config: &SpectralFluxConfig,
    ) -> AudioSampleResult<Vec<f64>>;

    /// Computes the spectral flux curve and its time axis.
    ///
    /// Returns the per-frame positive spectral change using the specified
    /// CQT parameters and flux method, together with the corresponding
    /// timestamps.  The raw flux curve can be used as an onset strength
    /// signal for beat tracking or visualisation.
    ///
    /// # Arguments
    ///
    /// - `config` – CQT analysis parameters (bins per octave, frequency range, etc.).
    /// - `window_size` – Analysis window length in samples.
    /// - `hop_size` – Number of samples to advance between successive windows.
    /// - `method` – Spectral flux variant to compute (e.g. positive flux,
    ///   complex flux, or Wiener entropy).
    ///
    /// # Returns
    ///
    /// A `(flux_values, timestamps)` tuple, both of equal length.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if `window_size` or `hop_size`
    /// are inconsistent with the signal length.
    /// Returns [crate::AudioSampleError::Processing] if the CQT computation fails.
    fn spectral_flux(
        &self,
        config: &CqtParams,
        window_size: NonZeroUsize,
        hop_size: NonZeroUsize,
        method: SpectralFluxMethod,
    ) -> AudioSampleResult<(NonEmptyVec<f64>, NonEmptyVec<f64>)>;

    /// Detects onset times using the complex-domain onset detection function.
    ///
    /// Combines magnitude difference and phase deviation into a single
    /// activation curve that is sensitive to both amplitude changes and phase
    /// discontinuities.  This makes it more robust than spectral-flux alone
    /// for sustained or polyphonic content.
    ///
    /// # Arguments
    ///
    /// - `onset_config` – Complex onset detection parameters: STFT settings
    ///   and peak-picking thresholds.
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` of onset times in seconds, sorted ascending.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if `onset_config` fields are invalid.
    /// Returns [crate::AudioSampleError::Processing] if the STFT computation fails.
    fn complex_onset_detection(
        &self,
        onset_config: &ComplexOnsetConfig,
    ) -> AudioSampleResult<Vec<f64>>;

    /// Computes the complex-domain onset detection function curve.
    ///
    /// Returns the raw per-frame activation values before peak-picking.
    /// Each value combines the magnitude difference and unwrapped phase
    /// deviation for that frame.
    ///
    /// # Arguments
    ///
    /// - `onset_config` – Complex onset detection parameters.
    ///
    /// # Returns
    ///
    /// A `NonEmptyVec<f64>` with one value per analysis frame, in
    /// chronological order.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if `onset_config` is invalid.
    /// Returns [crate::AudioSampleError::Processing] if the STFT computation fails.
    fn onset_detection_function_complex(
        &self,
        onset_config: &ComplexOnsetConfig,
    ) -> AudioSampleResult<NonEmptyVec<f64>>;

    /// Computes the frame-by-frame magnitude difference matrix.
    ///
    /// Each entry `[i, j]` is the absolute difference in spectral magnitude
    /// between consecutive STFT frames at frequency bin `j` and frame `i`.
    /// This is an intermediate building block of the complex ODF.
    ///
    /// # Arguments
    ///
    /// - `config` – Complex onset detection parameters controlling the STFT
    ///   analysis window.
    ///
    /// # Returns
    ///
    /// An `Array2<f64>` with shape `[frames, frequency_bins]`.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Processing] if the STFT computation fails.
    fn magnitude_difference_matrix(
        &self,
        config: &ComplexOnsetConfig,
    ) -> AudioSampleResult<Array2<f64>>;

    /// Computes the frame-by-frame phase deviation matrix.
    ///
    /// Each entry `[i, j]` is the second-order phase difference (phase
    /// deviation from a constant-frequency prediction) at frequency bin `j`
    /// and frame `i`.  This is an intermediate building block of the
    /// complex ODF.
    ///
    /// # Arguments
    ///
    /// - `config` – Complex onset detection parameters controlling the STFT
    ///   analysis window.
    ///
    /// # Returns
    ///
    /// An `Array2<f64>` with shape `[frames, frequency_bins]`.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Processing] if the STFT computation fails.
    fn phase_deviation_matrix(&self, config: &ComplexOnsetConfig)
    -> AudioSampleResult<Array2<f64>>;
}

/// Beat tracking operations for audio signals.
///
/// This trait provides tempo-aware beat detection via
/// [`detect_beats`][AudioBeatTracking::detect_beats].  Configure the
/// detection pipeline with [`BeatTrackingConfig`] and retrieve results
/// as [`BeatTrackingData`].
#[cfg(feature = "beat-tracking")]
pub trait AudioBeatTracking: AudioTransforms
where
    Self::Sample: StandardSample,
{
    /// Detect beat positions in the audio signal at the target tempo.
    ///
    /// Computes an onset strength envelope from the signal using the
    /// onset detection configuration in `config`, then locates beat
    /// frames by walking forward and backward from the global onset
    /// peak in steps of one inter-beat interval.
    ///
    /// # Arguments
    /// - `config` – Beat tracking configuration: target tempo,
    ///   optional timing tolerance, and onset detection parameters.
    ///
    /// # Returns
    /// A [`BeatTrackingData`] containing the target tempo and the
    /// detected beat timestamps in seconds.  Beat times are in
    /// detection order (global peak first, then forward beats, then
    /// backward beats in reverse); sort `beat_times` for
    /// chronological order.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if `config.tempo_bpm` is
    ///   ≤ 0 or if the inter-beat interval is too small relative to
    ///   the hop size.
    ///
    /// # Examples
    /// ```no_run
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::beat::{BeatTrackingConfig, BeatTrackingData};
    /// use audio_samples::operations::traits::AudioBeatTracking;
    ///
    /// # fn example(audio: AudioSamples<'_, f32>, config: BeatTrackingConfig) {
    /// let result = audio.detect_beats(&config).unwrap();
    /// println!("Tempo: {:.1} BPM", result.tempo_bpm);
    /// for &t in &result.beat_times {
    ///     println!("  beat at {:.3} s", t);
    /// }
    /// # }
    /// ```
    fn detect_beats(&self, config: &BeatTrackingConfig) -> AudioSampleResult<BeatTrackingData>;
}

/// Audio visualisation operations.
///
/// # Purpose
///
/// Produces interactive plots of audio signals in three representations:
/// time-domain waveform, time–frequency spectrogram, and single-frame
/// magnitude spectrum.  All plots are rendered via the `plotly` crate.
///
/// # Intended Usage
///
/// Use this trait during exploratory analysis or for generating report
/// artefacts.  Each method returns an opaque plot object that can be
/// rendered to HTML or saved to a file — consult the `WaveformPlot`,
/// `SpectrogramPlot`, and `MagnitudeSpectrumPlot` types for output
/// options.
///
/// # Invariants
///
/// Plot methods do not modify the audio signal.  Errors are only returned
/// when the underlying spectrogram or FFT computation fails (e.g. if the
/// signal is too short for the requested window size).
#[cfg(feature = "plotting")]
pub trait AudioPlotting: AudioTransforms
where
    Self::Sample: StandardSample,
{
    /// Renders a time-domain waveform plot.
    ///
    /// Draws the amplitude of each sample against time, one trace per channel.
    ///
    /// # Arguments
    ///
    /// - `params` – Visual parameters: title, axis labels, colour scheme,
    ///   and time range.
    ///
    /// # Returns
    ///
    /// A [`WaveformPlot`] that can be rendered to HTML or saved to a file.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Processing] if the internal rendering
    /// pipeline fails.
    fn plot_waveform(&self, params: &WaveformPlotParams) -> AudioSampleResult<WaveformPlot>;

    /// Renders a time–frequency spectrogram plot.
    ///
    /// Computes an STFT and displays the resulting magnitude (or log-magnitude)
    /// as a heat-map with time on the x-axis and frequency on the y-axis.
    ///
    /// # Arguments
    ///
    /// - `params` – Spectrogram and visual parameters: STFT window size, hop
    ///   size, frequency range, colour scale, and title.
    ///
    /// # Returns
    ///
    /// A [`SpectrogramPlot`] that can be rendered to HTML or saved to a file.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Processing] if the STFT computation fails
    /// (e.g. signal shorter than the STFT window).
    fn plot_spectrogram(
        &self,
        params: &SpectrogramPlotParams,
    ) -> AudioSampleResult<SpectrogramPlot>;

    /// Renders a magnitude spectrum plot for a single frame.
    ///
    /// Applies an FFT to the audio and displays the per-bin magnitudes on
    /// a frequency axis, optionally on a logarithmic scale.
    ///
    /// # Arguments
    ///
    /// - `params` – Spectrum and visual parameters: FFT size, frequency range,
    ///   amplitude scale, and title.
    ///
    /// # Returns
    ///
    /// A [`MagnitudeSpectrumPlot`] that can be rendered to HTML or saved to a file.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Processing] if the FFT computation fails.
    fn plot_magnitude_spectrum(
        &self,
        params: &MagnitudeSpectrumParams,
    ) -> AudioSampleResult<MagnitudeSpectrumPlot>;
}

/// Amplitude envelope extraction operations.
///
/// # Purpose
///
/// Tracks how the amplitude of an audio signal changes over time.  Five
/// complementary methods cover the most common envelope shapes: per-sample
/// rectification, RMS sliding window, attack/decay follower, analytic
/// (Hilbert-transform) envelope, and moving-average smoothing.
///
/// # Intended Usage
///
/// Use this trait in dynamics processing (side-chain detection), amplitude
/// modulation analysis, or anywhere you need a smooth representation of
/// signal loudness over time.  Choose the method based on smoothing
/// preference and computational budget:
///
/// - `amplitude_envelope` — fastest, no smoothing.
/// - `rms_envelope` / `moving_average_envelope` — configurable smoothing via window.
/// - `analytic_envelope` — smoothest, requires a Hilbert transform.
/// - `attack_decay_envelope` — separates rising and falling energy.
///
/// # Invariants
///
/// All methods return an [`NdResult`] that mirrors the input channel layout:
/// `NdResult::Mono` for mono audio and `NdResult::MultiChannel` for stereo
/// or higher.  The sample count of each output matches the input (per-sample
/// methods) or `⌈input_samples / hop_size⌉` (windowed methods).
#[cfg(feature = "envelopes")]
pub trait AudioEnvelopes: AudioStatistics
where
    Self::Sample: StandardSample,
{
    /// Compute a per-sample rectified amplitude envelope.
    ///
    /// Each output sample is the absolute value of the corresponding
    /// input sample.  The channel layout of the input is preserved.
    ///
    /// # Returns
    /// An [`NdResult`] matching the input channel layout:
    /// - [`NdResult::Mono`] for mono audio.
    /// - [`NdResult::MultiChannel`] for multi-channel audio.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, NdResult, sample_rate};
    /// use audio_samples::operations::traits::AudioEnvelopes;
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     array![1.0f32, -0.5, 0.25],
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let envelope = audio.amplitude_envelope();
    /// if let NdResult::Mono(env) = envelope {
    ///     assert_eq!(env[0], 1.0f32);
    ///     assert_eq!(env[1], 0.5f32); // |-0.5| = 0.5
    /// }
    /// ```
    fn amplitude_envelope(&self) -> NdResult<Self::Sample>;

    /// Compute the root-mean-square (RMS) envelope using a sliding window.
    ///
    /// The signal is divided into overlapping windows of `window_size`
    /// samples, advancing by `hop_size` samples between windows.  The
    /// RMS of each window becomes one output sample.  Partial windows
    /// at the end of the signal are included.
    ///
    /// # Arguments
    /// - `window_size` – Number of samples in each analysis window.
    /// - `hop_size` – Number of samples to advance between successive
    ///   windows.
    ///
    /// # Returns
    /// An [`NdResult`] matching the input channel layout, with one
    /// value per window.  The output length is approximately
    /// `ceil(samples / hop_size)`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, NdResult, sample_rate};
    /// use audio_samples::operations::traits::AudioEnvelopes;
    /// use ndarray::array;
    /// use std::num::NonZeroUsize;
    ///
    /// // A constant ±1 signal has RMS of 1.0 per window
    /// let audio = AudioSamples::new_mono(
    ///     array![1.0f32, -1.0, 1.0, -1.0],
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let w = NonZeroUsize::new(2).unwrap();
    /// let h = NonZeroUsize::new(2).unwrap();
    /// let envelope = audio.rms_envelope(w, h);
    /// if let NdResult::Mono(env) = envelope {
    ///     assert_eq!(env.len(), 2);
    ///     assert!((env[0] - 1.0).abs() < 1e-6);
    /// }
    /// ```
    fn rms_envelope(
        &self,
        window_size: NonZeroUsize,
        hop_size: NonZeroUsize,
    ) -> NdResult<Self::Sample>;

    /// Track amplitude over time with an envelope follower, separating
    /// attack and decay phases.
    ///
    /// For each sample, the [`EnvelopeFollower`] estimates the current
    /// signal level.  Samples where the level is rising contribute to
    /// the *attack* envelope; samples where it is falling contribute
    /// to the *decay* envelope.  Both output envelopes have the same
    /// length as the input.
    ///
    /// # Arguments
    /// - `follower` – A configured [`EnvelopeFollower`] that controls
    ///   attack and release time constants.
    /// - `method` – Detection strategy used to estimate level:
    ///   peak, RMS, or hybrid.
    ///
    /// # Returns
    /// A tuple `(attack, decay)` of [`NdResult`] values, both matching
    /// the input channel layout and length.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, NdResult, sample_rate};
    /// use audio_samples::operations::traits::AudioEnvelopes;
    /// use audio_samples::operations::dynamic_range::EnvelopeFollower;
    /// use audio_samples::operations::types::DynamicRangeMethod;
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     array![0.0f32, 0.0, 0.0],
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let follower = EnvelopeFollower::new(
    ///     1.0, 10.0, 44100.0, DynamicRangeMethod::Peak,
    /// );
    /// let (attack, _decay) = audio
    ///     .attack_decay_envelope(&follower, DynamicRangeMethod::Peak);
    /// if let NdResult::Mono(env) = attack {
    ///     assert!(env.iter().all(|&v| v.abs() < 1e-6));
    /// }
    /// ```
    #[allow(clippy::type_complexity)]
    fn attack_decay_envelope(
        &self,
        follower: &EnvelopeFollower,
        method: DynamicRangeMethod,
    ) -> (NdResult<Self::Sample>, NdResult<Self::Sample>);

    /// Compute the instantaneous amplitude envelope via the analytic signal.
    ///
    /// The analytic signal is obtained by applying a Hilbert transform
    /// to the input.  The instantaneous amplitude at each sample is the
    /// magnitude of the resulting complex signal
    /// (√(xᵣ² + xᵢ²)).  This produces a smooth envelope that tracks
    /// the true amplitude of modulated or band-limited signals more
    /// accurately than simple rectification.
    ///
    /// # Returns
    /// An [`NdResult`] matching the input channel layout, with one
    /// instantaneous amplitude value per input sample.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, NdResult, sample_rate};
    /// use audio_samples::operations::traits::AudioEnvelopes;
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     array![1.0f32, 0.0, -1.0, 0.0],
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let envelope = audio.analytic_envelope();
    /// assert!(matches!(envelope, NdResult::Mono(_)));
    /// ```
    fn analytic_envelope(&self) -> NdResult<Self::Sample>;

    /// Compute a moving-average envelope over the rectified signal.
    ///
    /// The signal is first rectified (absolute value taken per sample),
    /// then a sliding window mean is applied.  The window advances by
    /// `hop_size` samples between successive outputs.  Partial windows
    /// at the end of the signal are included.
    ///
    /// # Arguments
    /// - `window_size` – Number of samples per averaging window.
    /// - `hop_size` – Number of samples to advance between successive
    ///   windows.
    ///
    /// # Returns
    /// An [`NdResult`] matching the input channel layout, with one
    /// mean value per window.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, NdResult, sample_rate};
    /// use audio_samples::operations::traits::AudioEnvelopes;
    /// use ndarray::array;
    /// use std::num::NonZeroUsize;
    ///
    /// // signal [0, 2, 4, 6]: window means are 1.0 and 5.0
    /// let audio = AudioSamples::new_mono(
    ///     array![0.0f32, 2.0, 4.0, 6.0],
    ///     sample_rate!(44100),
    /// ).unwrap();
    /// let w = NonZeroUsize::new(2).unwrap();
    /// let h = NonZeroUsize::new(2).unwrap();
    /// let envelope = audio.moving_average_envelope(w, h);
    /// if let NdResult::Mono(env) = envelope {
    ///     assert_eq!(env.len(), 2);
    ///     assert!((env[0] - 1.0).abs() < 1e-6);
    ///     assert!((env[1] - 5.0).abs() < 1e-6);
    /// }
    /// ```
    fn moving_average_envelope(
        &self,
        window_size: NonZeroUsize,
        hop_size: NonZeroUsize,
    ) -> NdResult<Self::Sample>;
}
