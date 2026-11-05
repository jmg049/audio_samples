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
//! | [`AudioSpectrum`] | `transforms` | librosa-compatible STFT/ISTFT, mel spectrogram, MFCC |
//! | [`AudioTonal`] | `feature-extraction` | librosa-compatible chroma_stft, piptrack, tuning estimation |
//! | [`AudioFeatures`] | `feature-extraction` | librosa-compatible frame-wise spectral features, RMS, ZCR |
//! | `AudioEffects` | `effects` | librosa-compatible time stretch, pitch shift, emphasis, dB trim/split, HPSS |
//! | [`AudioConstantQ`] | `transforms` + `resampling` | librosa-compatible CQT/VQT/iCQT, chroma_cqt/cens/vqt, tonnetz |
//! | [`AudioEditing`] | `editing` | Trim, pad, fade, concatenate, mix, … |
//! | [`AudioChannelOps`] | `channels` | Mono/stereo conversion, panning, interleave, … |
//! | [`AudioEnvelopes`] | `envelopes` | Amplitude, RMS, attack/decay, analytic envelope, … |
//! | [`AudioDynamicRange`] | `dynamic-range` | Compressor, limiter, gate, expander |
//! | [`AudioIirFiltering`] | `iir-filtering` | Butterworth, Chebyshev IIR filters |
//! | [`AudioParametricEq`] | `parametric-eq` | Multi-band parametric EQ |
//! | [`AudioPitchAnalysis`] | `pitch-analysis` | YIN, autocorrelation, harmonic analysis |
//! | [`AudioPitchTracking`] | `pitch-analysis` | librosa-compatible frame-wise YIN and pYIN |
//! | [`AudioVoiceActivityDetection`] | `vad` | Frame-based speech / silence classification |
//! | [`AudioDecomposition`] | `decomposition` | Harmonic-percussive source separation (HPSS) |
//! | [`AudioOnsetDetection`] | `onset-detection` | Onset times, spectral flux, complex ODF |
//! | [`AudioBeatTracking`] | `beat-tracking` | Tempo-aware beat detection |
//! | [`AudioRhythm`] | `beat-tracking` | librosa-compatible onset strength/detection, tempogram, tempo, beat tracking, PLP |
//! | `AudioPlotting` | `plotting` | Waveform, spectrogram, magnitude-spectrum plots |
//!
//! Grouping operations into separate traits means only the code for the enabled
//! features is compiled, and new capabilities can be added as new traits. Each trait
//! is independently usable: bring only the traits you need into scope.
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

// The `explainable` attribute (feature `educational`) generates a chain
// wrapper that calls every `AudioProcessing` method, including the deprecated
// `apply_window` / `apply_window_in_place`. The generated code is emitted into
// this module and cannot carry its own attribute, so the deprecation lint is
// allowed here for that feature only.
#![cfg_attr(feature = "educational", allow(deprecated))]

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
use crate::operations::types::{EqBand, ParametricEq, ThreeBandEqConfig};

#[cfg(feature = "beat-tracking")]
use crate::operations::beat::{BeatTrackingConfig, BeatTrackingData};

#[cfg(feature = "vad")]
use crate::operations::types::VadConfig;

#[cfg(feature = "dynamic-range")]
use crate::operations::types::{CompressorConfig, ExpanderConfig, GateConfig, LimiterConfig};

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
    plotting::lissajous::{LissajousParams, LissajousPlot},
    plotting::phase_spectrum::{PhaseSpectrumParams, PhaseSpectrumPlot},
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
/// Use this trait for analysis, visualisation, and gating decisions - for example
/// to check loudness before normalisation, to estimate the noise floor before gating,
/// or to drive automatic level-matching in a processing pipeline.
///
/// `autocorrelation` and the spectral methods (`spectral_*`) require
/// `feature = "transforms"`. Units and exact formulas (raw vs normalised
/// sample units, power vs magnitude weighting) are defined in the
/// [`statistics`](crate::operations::statistics) module documentation.
///
/// # Invariants
///
/// All methods operate on the full multi-channel buffer unless the documentation
/// explicitly states otherwise (e.g. `midpoint_sample` is mono-only, while
/// `spectral_centroid` and `spectral_rolloff` take an explicit
/// [`ChannelReduction`](crate::operations::types::ChannelReduction) policy for
/// multi-channel input).
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
    /// `x̄ = (1/N) Σ x[n]` over every sample of every channel (`N` = total sample
    /// count), accumulated in `f64` so integer sample types cannot overflow. The
    /// value is in raw sample units: for `i16` it lies in `[-32768, 32767]`, and
    /// for `u8` (offset-binary) a silent signal has mean `128.0`.
    ///
    /// # Returns
    /// The mean value as `f64`, in raw sample units.
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
    /// assert_eq!(audio.midpoint_sample(), Some(4.0));
    /// ```
    fn midpoint_sample(&self) -> Option<f64>;

    /// Computes the Root Mean Square (RMS) of all samples across all channels.
    ///
    /// `RMS = sqrt((1/N) Σ x[n]²)` over every sample of every channel, accumulated
    /// in `f64`. It measures the signal's average power and is commonly used for
    /// loudness estimation. A sine of amplitude `A` has RMS `A/√2`.
    ///
    /// Samples are used in raw units: for `i16`, full scale is 32768, and for `u8`
    /// the 128 offset is not removed. Convert with
    /// [`as_f64`](crate::AudioTypeConversion::as_f64) first for an RMS relative to
    /// full scale.
    ///
    /// # Returns
    /// The RMS value as `f64`, in raw sample units.
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
    /// `σ² = (1/N) Σ (x[n] − x̄)²` over every sample of every channel (all channels
    /// pooled, `N` = total sample count, `x̄` = [`mean`](AudioStatistics::mean)),
    /// accumulated in `f64` in raw sample units. Divide by `N` (population), not
    /// `N − 1`.
    ///
    /// # Returns
    /// The variance as `f64`, in squared raw sample units.
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
    /// A zero crossing is a pair of adjacent samples `(x[n−1], x[n])` where one is
    /// `> 0` and the other is `≤ 0` (an exact zero counts as non-positive). Samples
    /// are first mapped with the audio-aware conversion to `f64`, so `u8`
    /// (offset-binary) is measured about its mid-scale value 128. A sine at `f` Hz
    /// lasting `T` seconds has about `2·f·T` crossings. The count is useful for
    /// estimating the noisiness or dominant frequency of a signal.
    ///
    /// # Returns
    /// The total number of zero crossings, summed over all channels (each channel
    /// is scanned separately). Returns 0 if the audio has fewer than 2 samples per
    /// channel.
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

    /// Computes the zero-crossing rate in crossings per second.
    ///
    /// `ZCR = zero_crossings / (channels · duration_s)` with
    /// `duration_s = samples_per_channel / sample_rate`. The unit is crossings per
    /// second (Hz), not a fraction of samples: a sine at `f` Hz gives about `2f`.
    /// For multi-channel audio it is the mean of the per-channel rates (the total
    /// from [`zero_crossings`](Self::zero_crossings) divided by the number of
    /// channels), so a stereo signal with two identical channels has the same
    /// rate as either channel alone.
    ///
    /// Note: librosa's `zero_crossing_rate` returns a per-frame fraction
    /// (crossings / frame length). For a mono signal, divide this value by the
    /// sample rate to get the equivalent whole-signal fraction of samples.
    ///
    /// # Returns
    /// The number of zero crossings per second as `f64`, averaged over channels.
    ///
    /// # Differences from librosa
    ///
    /// `librosa.feature.zero_crossing_rate` returns, per channel and per frame,
    /// the fraction of samples at which the sign changes (centred frames of 2048
    /// samples, hop 512). This method is a single whole-signal rate in Hz.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 1.0, -1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // 3 crossings in 4 samples = 4 / 44100 s.
    /// let zcr = audio.zero_crossing_rate();
    /// assert!((zcr - 3.0 * 44100.0 / 4.0).abs() < 1e-9);
    /// ```
    fn zero_crossing_rate(&self) -> f64;

    /// Computes the cross-correlation with another audio signal for non-negative
    /// lags.
    ///
    /// `r[l] = (1/C_l) Σ_{i=0}^{C_l−1} self[i]·other[i+l]` for `l = 0..=max_lag`,
    /// with `C_l = min(len_self, len_other − l)`. This is an unbiased
    /// (overlap-normalised) estimate in squared raw sample units, with no mean
    /// removal, accumulated in `f64`. A peak at `l = D` means `other` is `self`
    /// delayed by `D` samples. Negative lags are not computed: to detect `other`
    /// leading `self`, swap the arguments or use
    /// [`fft_alignment_lag`](crate::operations::statistics::fft_alignment_lag).
    ///
    /// For multi-channel audio only the first channels of both signals are
    /// correlated.
    ///
    /// # Arguments
    /// - `other` - the second audio signal. Must have the same number of
    ///   channels as `self`.
    /// - `max_lag` - the maximum lag offset in samples. The effective maximum
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
    /// let data2 = array![0.0, 1.0, 0.0, -1.0]; // data1 delayed by one sample
    /// let audio1 = AudioSamples::new_mono(data1, sample_rate!(44100)).unwrap();
    /// let audio2 = AudioSamples::new_mono(data2, sample_rate!(44100)).unwrap();
    /// let max_lag = NonZeroUsize::new(3).unwrap();
    /// let xcorr = audio1.cross_correlation(&audio2, max_lag).unwrap();
    /// assert_eq!(xcorr.len(), NonZeroUsize::new(4).unwrap()); // lags 0..=3
    /// // Peak at lag 1: (1·1 + 0·0 + (−1)·(−1)) / 3
    /// assert!((xcorr[1] - 2.0 / 3.0).abs() < 1e-12);
    /// ```
    fn cross_correlation(
        &self,
        other: &Self,
        max_lag: NonZeroUsize,
    ) -> AudioSampleResult<NonEmptyVec<f64>>;

    /// Computes the spectral centroid of a mono signal.
    ///
    /// The power-weighted mean frequency `Σ f_k·|X[k]|² / Σ |X[k]|²`, where `X` is
    /// a single FFT of the whole (channel-reduced) signal with FFT length equal to
    /// the signal length `N` and no window (rectangular), `k = 0..=⌊N/2⌋`, and
    /// `f_k = k·fs/N` Hz. It serves as a measure of spectral brightness: higher
    /// values indicate energy concentrated at higher frequencies.
    ///
    /// Note: librosa's `spectral_centroid` is computed per STFT frame and weighted
    /// by magnitude `|X|`, so its values are generally higher than this
    /// power-weighted value and are not directly comparable.
    ///
    /// Reference: [Spectral centroid - Wikipedia](https://en.wikipedia.org/wiki/Spectral_centroid)
    ///
    /// # Arguments
    /// - `reduction` - the [`ChannelReduction`](crate::operations::types::ChannelReduction)
    ///   policy for multi-channel input.
    ///
    /// # Returns
    /// The spectral centroid frequency in Hz. Returns `0.0` when the signal
    /// is silence (zero total spectral energy).
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the signal is multi-channel and
    ///   `reduction` is [`ChannelReduction::Error`](crate::operations::types::ChannelReduction::Error),
    ///   or a [`Channel`](crate::operations::types::ChannelReduction::Channel) index
    ///   is out of bounds.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioStatistics, sample_rate, sine_wave};
    /// use audio_samples::operations::types::ChannelReduction;
    /// use std::time::Duration;
    ///
    /// // 100 ms at 44.1 kHz: bin spacing fs/N = 10 Hz, so 440 Hz lies on bin 44.
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 1.0);
    /// let centroid = audio.spectral_centroid(ChannelReduction::Error).unwrap();
    /// assert!((centroid - 440.0).abs() < 1.0);
    /// ```
    #[cfg(feature = "transforms")]
    fn spectral_centroid(
        &self,
        reduction: crate::operations::types::ChannelReduction,
    ) -> AudioSampleResult<f64>;

    /// Computes the autocorrelation function up to `max_lag` samples.
    ///
    /// Returns the unbiased estimate
    /// `r[k] = (1/(N−k)) Σ_{n=0}^{N−1−k} x[n]·x[n+k]` for `k = 0..=max_lag`, in
    /// squared raw sample units, with no mean removal and no normalisation by
    /// `r[0]`. `r[0]` is the mean square (`A²/2` for a sine of amplitude `A`). The
    /// values do not necessarily decrease with `k`: for a periodic signal `r[k]`
    /// peaks again at every multiple of the period, and at large `k` (few
    /// overlapping samples) the estimate is noisy and can exceed `r[0]`. It is
    /// computed with a zero-padded FFT, so the correlation is linear, not circular.
    ///
    /// For multi-channel audio only the first channel is used.
    ///
    /// Note: unlike the biased estimator `(1/N) Σ …` (e.g.
    /// `numpy.correlate(x, x, "full")[N-1:] / N`), each lag is divided by its own
    /// overlap count `N − k`.
    ///
    /// Reference: [Autocorrelation - Wikipedia](https://en.wikipedia.org/wiki/Autocorrelation)
    ///
    /// # Arguments
    /// - `max_lag` - the maximum lag offset in samples. The effective maximum
    ///   lag is clamped to `signal_length - 1`.
    ///
    /// # Returns
    /// A [`NonEmptyVec`] of correlation values for lags `0` through
    /// `min(max_lag, signal_length - 1)`, or `None` if the FFT computation
    /// fails.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioStatistics, sample_rate};
    /// use ndarray::array;
    /// use std::num::NonZeroUsize;
    ///
    /// let data = array![1.0f32, 2.0, 3.0, 4.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// let r = audio.autocorrelation(NonZeroUsize::new(3).unwrap()).unwrap();
    /// assert_eq!(r.len(), NonZeroUsize::new(4).unwrap()); // lags 0..=3
    /// // r[0] = (1 + 4 + 9 + 16) / 4, r[1] = (2 + 6 + 12) / 3
    /// assert!((r[0] - 7.5).abs() < 1e-9);
    /// assert!((r[1] - 20.0 / 3.0).abs() < 1e-9);
    /// ```
    #[cfg(feature = "transforms")]
    fn autocorrelation(&self, max_lag: NonZeroUsize) -> Option<NonEmptyVec<f64>>;

    /// Computes the spectral rolloff frequency.
    ///
    /// The smallest bin frequency `f_k = k·fs/N` at which the cumulative power
    /// `Σ_{j≤k} |X[j]|²` reaches `rolloff_percent` of the total power, where `X`
    /// is a single rectangular-window FFT of the whole (channel-reduced) signal with
    /// FFT length equal to the signal length `N`. There is no interpolation
    /// between bins, so the result is a multiple of `fs/N`. It is commonly used to
    /// distinguish harmonic signals from noise-like signals.
    ///
    /// Note: librosa's `spectral_rolloff` accumulates magnitude `|X|` per STFT
    /// frame, so its values differ from this whole-signal, power-based rolloff.
    ///
    /// # Arguments
    /// - `rolloff_percent` - the energy proportion threshold. Must lie in the
    ///   open interval `(0.0, 1.0)`. A typical value is `0.85`.
    /// - `reduction` - the [`ChannelReduction`](crate::operations::types::ChannelReduction)
    ///   policy applied to multi-channel input. Pass
    ///   [`First`](crate::operations::types::ChannelReduction::First) to reproduce the
    ///   historical channel-0 behaviour.
    ///
    /// # Returns
    /// The rolloff frequency in Hz. Returns `0.0` when the signal is silence
    /// (zero total spectral energy).
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `rolloff_percent` is not in
    ///   `(0.0, 1.0)`, if the signal is multi-channel and `reduction` is
    ///   [`ChannelReduction::Error`](crate::operations::types::ChannelReduction::Error),
    ///   or if a [`Channel`](crate::operations::types::ChannelReduction::Channel) index is out of bounds.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioStatistics, sample_rate, sine_wave};
    /// use audio_samples::operations::types::ChannelReduction;
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(100), sample_rate!(44100), 1.0);
    /// // All the power is in the 440 Hz bin, so 85 % is reached there.
    /// let rolloff = audio.spectral_rolloff(0.85, ChannelReduction::Error).unwrap();
    /// assert!((rolloff - 440.0).abs() < 1e-6);
    /// ```
    #[cfg(feature = "transforms")]
    fn spectral_rolloff(
        &self,
        rolloff_percent: f64,
        reduction: crate::operations::types::ChannelReduction,
    ) -> AudioSampleResult<f64>;

    /// Computes the spectral bandwidth (spectral spread).
    ///
    /// The magnitude-weighted standard deviation of frequency,
    /// `sqrt( Σ (f_k − c_M)²·M_k / Σ M_k )`, where `M_k = |X[k]|` is the magnitude
    /// of a single rectangular-window FFT of the whole (channel-reduced) signal,
    /// `f_k = k·fs/N` Hz, and `c_M = Σ f_k·M_k / Σ M_k` is a **magnitude**-weighted
    /// centroid computed internally. Because
    /// [`spectral_centroid`](crate::operations::traits::AudioStatistics::spectral_centroid)
    /// is power-weighted, the spread is not centred on that value. Larger values
    /// indicate energy spread across a wider frequency range; a single on-bin pure
    /// tone yields a value near zero.
    ///
    /// Note: this is librosa's `spectral_bandwidth` formula (`p = 2`, magnitude
    /// weights) applied to one whole-signal FFT rather than to each STFT frame.
    ///
    /// # Arguments
    /// - `reduction` - the [`ChannelReduction`](crate::operations::types::ChannelReduction)
    ///   policy for multi-channel input.
    ///
    /// # Returns
    /// The spectral bandwidth in Hz. Returns `0.0` for silence (zero total
    /// magnitude).
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the signal is multi-channel and
    ///   `reduction` is [`ChannelReduction::Error`](crate::operations::types::ChannelReduction::Error),
    ///   or a [`Channel`](crate::operations::types::ChannelReduction::Channel) index
    ///   is out of bounds.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    #[cfg(feature = "transforms")]
    fn spectral_bandwidth(
        &self,
        reduction: crate::operations::types::ChannelReduction,
    ) -> AudioSampleResult<f64>;

    /// Computes the spectral flatness (Wiener entropy).
    ///
    /// The ratio of the geometric mean to the arithmetic mean of the power
    /// spectrum `P_k = |X[k]|²` of a single rectangular-window FFT of the whole
    /// (channel-reduced) signal: `exp(mean(ln(P_k + ε))) / mean(P_k)`, with
    /// `ε = 1e-10` so that empty bins keep the logarithm finite. The result is
    /// clamped to `[0, 1]`: values near `1` indicate noise-like (flat) spectra,
    /// while values near `0` indicate tonal spectra dominated by a few peaks.
    ///
    /// Note: librosa's `spectral_flatness` (default `power=2.0`) uses the same
    /// ratio per STFT frame, with `max(P_k, amin)` instead of `P_k + ε`.
    ///
    /// # Arguments
    /// - `reduction` - the [`ChannelReduction`](crate::operations::types::ChannelReduction)
    ///   policy for multi-channel input.
    ///
    /// # Returns
    /// The spectral flatness in `[0, 1]`. Returns `0.0` for silence.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the signal is multi-channel and
    ///   `reduction` is [`ChannelReduction::Error`](crate::operations::types::ChannelReduction::Error),
    ///   or a [`Channel`](crate::operations::types::ChannelReduction::Channel) index
    ///   is out of bounds.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    #[cfg(feature = "transforms")]
    fn spectral_flatness(
        &self,
        reduction: crate::operations::types::ChannelReduction,
    ) -> AudioSampleResult<f64>;

    /// Computes the spectral crest factor.
    ///
    /// The ratio of the peak magnitude to the mean magnitude of the spectrum:
    /// `max(M_k) / mean(M_k)`, with `M_k = |X[k]|` the magnitude of a single
    /// rectangular-window FFT of the whole (channel-reduced) signal. High values
    /// indicate a strongly peaked (tonal) spectrum; a flat spectrum approaches `1`.
    ///
    /// # Arguments
    /// - `reduction` - the [`ChannelReduction`](crate::operations::types::ChannelReduction)
    ///   policy for multi-channel input.
    ///
    /// # Returns
    /// The spectral crest factor (≥ 1 for non-silent signals). Returns `0.0` for
    /// silence.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the signal is multi-channel and
    ///   `reduction` is [`ChannelReduction::Error`](crate::operations::types::ChannelReduction::Error),
    ///   or a [`Channel`](crate::operations::types::ChannelReduction::Channel) index
    ///   is out of bounds.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    #[cfg(feature = "transforms")]
    fn spectral_crest(
        &self,
        reduction: crate::operations::types::ChannelReduction,
    ) -> AudioSampleResult<f64>;

    /// Computes the spectral slope.
    ///
    /// The slope of an ordinary least-squares linear fit of the **linear
    /// magnitude** spectrum `M_k = |X[k]|` against frequency `f_k = k·fs/N` (Hz),
    /// where `X` is a single rectangular-window, unnormalised FFT of the whole
    /// (channel-reduced) signal. Returned in units of magnitude per Hz; because the
    /// FFT is unnormalised, the value scales with the signal length. A negative
    /// slope indicates energy concentrated at low frequencies (e.g. a low-pass or
    /// pink-ish spectrum); a flat (white) spectrum yields a slope near zero.
    ///
    /// # Arguments
    /// - `reduction` - the [`ChannelReduction`](crate::operations::types::ChannelReduction)
    ///   policy for multi-channel input.
    ///
    /// # Returns
    /// The least-squares slope (magnitude per Hz). Returns `0.0` for silence or a
    /// degenerate single-bin spectrum.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the signal is multi-channel and
    ///   `reduction` is [`ChannelReduction::Error`](crate::operations::types::ChannelReduction::Error),
    ///   or a [`Channel`](crate::operations::types::ChannelReduction::Channel) index
    ///   is out of bounds.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    #[cfg(feature = "transforms")]
    fn spectral_slope(
        &self,
        reduction: crate::operations::types::ChannelReduction,
    ) -> AudioSampleResult<f64>;

    /// Computes spectral contrast across logarithmically spaced sub-bands (after
    /// librosa).
    ///
    /// The Hann-windowed magnitude spectrum `M_k` of the whole (channel-reduced)
    /// signal has `B = ⌊N/2⌋ + 1` bins. Bin 0 (DC) is excluded and the bins
    /// `1..B` are split into `n_bands` bands with geometrically spaced edges
    /// `e_b = ⌊r^b⌋`, `r = B^(1/n_bands)`; the last band ends at the Nyquist bin.
    /// Each band therefore spans `log2(r)` octaves, which is exactly one octave
    /// only when `B = 2^n_bands`. Within a band the values `10·log10(M_k + 1e-10)`
    /// are sorted and the contrast is the mean of the top 2 % minus the mean of the
    /// bottom 2 % (rounded, at least one bin each). High contrast indicates clear
    /// tonal/harmonic structure; low contrast indicates noise-like content.
    ///
    /// Note: like librosa, this applies `10·log10` to **magnitudes**, so each value
    /// is half the level difference in dB (multiply by 2 for a `20·log10`
    /// difference). Unlike librosa, the band edges are fixed in bins from bin 1 to
    /// Nyquist (librosa uses octaves above 200 Hz), and one value per band is
    /// returned for the whole signal rather than per frame.
    ///
    /// # Arguments
    /// - `n_bands` - the number of log-spaced sub-bands.
    /// - `reduction` - the [`ChannelReduction`](crate::operations::types::ChannelReduction)
    ///   policy for multi-channel input.
    ///
    /// # Returns
    /// A vector of `n_bands` contrast values (`10·log10` of a magnitude ratio),
    /// low band to high band. Empty bands yield `0.0`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the signal is multi-channel and
    ///   `reduction` is [`ChannelReduction::Error`](crate::operations::types::ChannelReduction::Error),
    ///   or a [`Channel`](crate::operations::types::ChannelReduction::Channel) index
    ///   is out of bounds.
    /// - [`crate::AudioSampleError::Processing`] if the FFT computation fails.
    #[cfg(feature = "transforms")]
    fn spectral_contrast(
        &self,
        n_bands: core::num::NonZeroUsize,
        reduction: crate::operations::types::ChannelReduction,
    ) -> AudioSampleResult<Vec<f64>>;
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
    /// and `config.hop_size`, computes each frame's RMS level (compared in
    /// dBFS with `config.energy_threshold_db`) and zero-crossing rate (crossings
    /// per sample, compared with `[config.zcr_min, config.zcr_max]`), applies
    /// the detection method, and post-processes the raw per-frame decisions
    /// through four sequential steps:
    ///
    /// 1. Majority-vote smoothing over `config.smooth_frames`.
    /// 2. Hangover extension - extends active frames forward by `config.hangover_frames`.
    /// 3. Minimum speech run enforcement - speech runs shorter than
    ///    `config.min_speech_frames` are reclassified as silence.
    /// 4. Minimum silence run enforcement - silence gaps shorter than
    ///    `config.min_silence_frames` are reclassified as speech.
    ///
    /// The formulas are given in the [`vad`](crate::operations::vad) module
    /// docs. Multi-channel audio is handled according to `config.channel_policy`.
    ///
    /// # Arguments
    ///
    /// - `config` - VAD parameters: frame size, hop size, detection method,
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
    /// (e.g. `hop_size > frame_size` or invalid ZCR bounds), or if `pad_end` is
    /// `false` and the signal is shorter than `frame_size` (no complete frame).
    /// Returns [crate::AudioSampleError::Layout] if the audio array is non-contiguous.
    /// Returns [crate::AudioSampleError::Unsupported] if `config.method` is
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
    /// - `config` - VAD parameters (same as [`voice_activity_mask`]).
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
/// Apply these methods in a preparation or post-processing pipeline - for example
/// to normalise a recording before feature extraction, to remove DC bias before
/// spectral analysis, or to resample to a target rate before model inference.
///
/// Every transforming operation comes in two variants per the 2.0 convention:
/// an in-place primitive (`op_in_place(&mut self, …)`) that mutates the buffer
/// with no extra allocation, and a non-mutating default (`op(&self, …)`) that
/// clones, applies the in-place variant, and returns the new copy. The unmarked
/// name borrows `self` and returns a fresh `Self`, enabling method chaining:
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
/// Resampling methods (`resample`, `resample_by_ratio`) take `&self` and build a
/// new, possibly different-length allocation; their `_in_place` twins replace the
/// internal buffer in place.
///
/// # Invariants
///
/// Methods that accept a sample-rate or frequency parameter validate that the
/// value is within the valid range; they return [crate::AudioSampleError::Parameter]
/// on failure rather than panicking.
#[cfg(feature = "processing")]
#[cfg_attr(feature = "educational", explainable::explainable)]
pub trait AudioProcessing: AudioTypeConversion
where
    Self::Sample: StandardSample,
{
    /// Normalizes audio samples using the specified configuration.
    ///
    /// The normalization method determines both the algorithm and parameters:
    /// - `NormalizationConfig::min_max(a, b)` - affine map onto `[a, b]`:
    ///   `x' = a + (x − x_min)(b − a)/(x_max − x_min)`
    /// - `NormalizationConfig::peak(A)` - `x' = A · x / max|x|`, so the
    ///   largest absolute sample becomes `A`
    /// - `NormalizationConfig::mean()` - `x' = x − x̄` (center around zero)
    /// - `NormalizationConfig::median()` - `x' = x − median(x)`, where the
    ///   median is the middle value of the *sorted* samples (mean of the two
    ///   middle values for an even count); mono only
    /// - `NormalizationConfig::zscore()` - `x' = (x − x̄)/σ` (zero mean, unit
    ///   population standard deviation)
    ///
    /// MinMax and Peak evaluate the map in `f64` and round back to the sample
    /// type, so they work for integer audio (Peak uses the normalised
    /// `[-1, 1]` scale of the sample type, MinMax the raw sample values).
    /// Mean and Median work on raw sample values; for integer types the result
    /// is rounded to nearest (half away from zero) and saturated to the type's
    /// range. ZScore output (dimensionless values of a few units) cannot be
    /// stored in an integer type, so ZScore returns an error for integer
    /// sample types; convert to `f32` or `f64` first.
    ///
    /// # Arguments
    /// - `config` - Normalization configuration. Use the associated constructors
    ///   on [`NormalizationConfig`] to build the desired method and parameters.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `min >= max` (MinMax), or if
    ///   the input is multi-channel (Median).
    /// - [`crate::AudioSampleError::Unsupported`] for ZScore with an integer
    ///   sample type (`u8`, `i16`, `I24`, `i32`).
    ///
    /// # Panics
    /// Panics if the configuration fields required by the selected method are
    /// `None`. Use the [`NormalizationConfig`] constructors to avoid this.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, NormalizationConfig, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -3.0, 2.0, -1.0];
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.normalize_in_place(NormalizationConfig::peak(1.0)).unwrap();
    /// ```
    fn normalize_in_place(
        &mut self,
        config: NormalizationConfig<Self::Sample>,
    ) -> AudioSampleResult<()>;

    /// Normalizes audio samples, returning a new normalized copy and leaving
    /// `self` unchanged.
    ///
    /// Non-mutating twin of [`normalize_in_place`](Self::normalize_in_place).
    /// See that method for the available normalization methods.
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
    /// - [`crate::AudioSampleError::Unsupported`] for ZScore with an integer
    ///   sample type (`u8`, `i16`, `I24`, `i32`).
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
    #[inline]
    fn normalize(&self, config: NormalizationConfig<Self::Sample>) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.normalize_in_place(config)?;
        Ok(out)
    }

    /// Scales all audio samples by a constant factor.
    ///
    /// This is equivalent to adjusting the volume or amplitude of the signal:
    /// `x'[n] = g · x[n]`. A factor of 1.0 leaves the signal unchanged,
    /// `|g| > 1` amplifies, `|g| < 1` attenuates (a gain of `g` changes the
    /// level by `20·log10|g|` dB) and a negative factor also inverts the
    /// polarity.
    ///
    /// For float sample types the product is not clipped. For integer sample
    /// types the gain is applied on the normalised `[-1, 1]` scale in `f64`
    /// and the result is rounded to nearest and saturated at the type's range
    /// (so `scale(0.5)` of `10000i16` is `5000`).
    ///
    /// # Arguments
    /// - `factor` - The scaling factor to apply to all samples.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 0.5];
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.scale_in_place(2.0);
    /// assert_eq!(audio[0], 2.0);
    /// assert_eq!(audio[1], -2.0);
    /// ```
    fn scale_in_place(&mut self, factor: f64);

    /// Scales all audio samples by a constant factor, returning a new scaled copy
    /// and leaving `self` unchanged.
    ///
    /// Non-mutating twin of [`scale_in_place`](Self::scale_in_place).
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
    #[inline]
    fn scale(&self, factor: f64) -> Self
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.scale_in_place(factor);
        out
    }

    /// Multiplies each sample by a window coefficient given in the sample type.
    ///
    /// Deprecated in favour of
    /// [`apply_window_f64_in_place`](Self::apply_window_f64_in_place). The
    /// window has the sample type and the product is computed in the sample
    /// type, so for integer types the only usable coefficients are `0` and
    /// `1`: a fractional coefficient cannot be represented, and a window
    /// scaled to the integer range overflows. For `f32` and `f64` audio the
    /// behaviour is unchanged from 2.0, and `apply_window_f64_in_place` with
    /// the same coefficients gives the same result.
    ///
    /// # Arguments
    /// - `window` - Window coefficients. Length must equal the number of
    ///   samples in the audio.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the window length does not
    ///   match the audio length.
    #[deprecated(
        since = "2.1.0",
        note = "integer sample types cannot hold fractional window coefficients; use apply_window_f64_in_place"
    )]
    fn apply_window_in_place(
        &mut self,
        window: &NonEmptySlice<Self::Sample>,
    ) -> AudioSampleResult<()>;

    /// Multiplies each sample by a window coefficient given in the sample
    /// type, returning a new copy.
    ///
    /// Deprecated in favour of [`apply_window_f64`](Self::apply_window_f64);
    /// see [`apply_window_in_place`](Self::apply_window_in_place) for the
    /// reason.
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
    #[deprecated(
        since = "2.1.0",
        note = "integer sample types cannot hold fractional window coefficients; use apply_window_f64"
    )]
    #[inline]
    fn apply_window(&self, window: &NonEmptySlice<Self::Sample>) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        #[allow(deprecated)] // the deprecated twin delegates to the deprecated primitive
        out.apply_window_in_place(window)?;
        Ok(out)
    }

    /// Applies a window function with `f64` coefficients, in place.
    ///
    /// Computes `x'[n] = w[n] · x[n]` for every channel, with the same window
    /// for each channel. Windowing is commonly used before a DFT to reduce
    /// spectral leakage. Each sample is converted to `f64` with the
    /// audio-aware conversion (integer samples on the normalised `[-1, 1]`
    /// scale), multiplied by its coefficient, and converted back: integer
    /// results are rounded to nearest and saturated at the type's range, and
    /// float results are rounded to the sample type. The sample type of the
    /// audio is preserved (`i16` in, `i16` out).
    ///
    /// # Arguments
    /// - `window` - Window coefficients, one per sample. Length must equal the
    ///   number of samples per channel. Any finite values are accepted;
    ///   coefficients above 1 amplify.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the window length does not
    ///   match the audio length.
    /// - [`crate::AudioSampleError::Unsupported`] from the default
    ///   implementation, which exists only so that implementors outside this
    ///   crate keep compiling; [`AudioSamples`] overrides
    ///   it.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    /// use non_empty_slice::NonEmptySlice;
    ///
    /// // i16 audio, a window with fractional coefficients.
    /// let mut audio = AudioSamples::new_mono(array![16384i16, 16384, 16384, 16384], sample_rate!(8000)).unwrap();
    /// let window = [1.0, 0.5, 0.25, 0.0];
    /// audio.apply_window_f64_in_place(NonEmptySlice::new(&window).unwrap()).unwrap();
    /// assert_eq!(audio[1], 8192);
    /// assert_eq!(audio[2], 4096);
    /// assert_eq!(audio[3], 0);
    /// ```
    #[inline]
    fn apply_window_f64_in_place(&mut self, window: &NonEmptySlice<f64>) -> AudioSampleResult<()> {
        let _ = window;
        Err(crate::AudioSampleError::unsupported(
            "apply_window_f64_in_place",
            "this AudioProcessing implementation does not provide apply_window_f64_in_place",
        ))
    }

    /// Applies a window function with `f64` coefficients, returning a new
    /// windowed copy and leaving `self` unchanged.
    ///
    /// Non-mutating twin of
    /// [`apply_window_f64_in_place`](Self::apply_window_f64_in_place); see that
    /// method for the arithmetic and the handling of integer sample types.
    ///
    /// # Arguments
    /// - `window` - Window coefficients, one per sample. Length must equal the
    ///   number of samples per channel.
    ///
    /// # Returns
    /// The windowed audio, in the same sample type as `self`.
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
    /// let window = [1.0, 0.5, 0.5, 1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .apply_window_f64(NonEmptySlice::new(&window).unwrap())
    ///     .unwrap();
    /// assert_eq!(audio[0], 1.0);
    /// assert_eq!(audio[1], 0.5);
    /// ```
    #[inline]
    fn apply_window_f64(&self, window: &NonEmptySlice<f64>) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.apply_window_f64_in_place(window)?;
        Ok(out)
    }

    /// Applies an FIR filter to the audio samples by linear convolution.
    ///
    /// Computes the fully-overlapping ("valid") part of the convolution of each
    /// channel with `filter_coeffs`, i.e. `y[i] = Σ_j h[j] · x[i + M - 1 - j]`
    /// for a filter of length `M` - identical to
    /// `numpy.convolve(x, h, mode='valid')`, and to
    /// `scipy.signal.lfilter(h, 1, x)[M-1:]`. The resulting audio is shorter
    /// than the original by `filter_coeffs.len() - 1` samples. Long filters
    /// (≥ 160 taps, with the `transforms` feature) use FFT convolution.
    ///
    /// Output sample `i` corresponds to input time `n = i + M − 1`: the
    /// first `M − 1` input samples produce no output of their own, and the
    /// result is not delay-compensated (a symmetric, linear-phase filter still
    /// delays its output by `(M − 1)/2` samples relative to that index).
    /// Note: unlike `AudioTransforms::convolve` (feature `transforms`), which
    /// returns the full convolution of length `N + M − 1`, this returns
    /// only the valid part of length `N − M + 1`.
    ///
    /// Computation is carried out in `f64`. For integer sample types both the
    /// samples and the coefficients are interpreted on the normalised
    /// `[-1, 1]` scale (so an `i16` coefficient of `16384` means ≈ 0.5), and
    /// the output is rounded to nearest and saturated. Integer taps are
    /// therefore limited to `[-1, 1]` and to the resolution of the sample
    /// type (`1/32768` for `i16`). For taps with a magnitude above 1 (for
    /// example `[1.5, -0.5]`) or at full `f64` precision, use
    /// [`apply_filter_f64_in_place`](Self::apply_filter_f64_in_place), which
    /// takes the taps as `f64` for every sample type.
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
    /// let coeffs = [1.0f32, -1.0]; // first difference: y[n] = x[n] - x[n-1]
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.apply_filter_in_place(NonEmptySlice::new(&coeffs).unwrap()).unwrap();
    /// assert_eq!(audio.samples_per_channel().get(), 4);
    /// assert_eq!(audio[0], 1.0); // 2 - 1
    /// assert_eq!(audio[3], 1.0); // 5 - 4
    /// ```
    fn apply_filter_in_place(
        &mut self,
        filter_coeffs: &NonEmptySlice<Self::Sample>,
    ) -> AudioSampleResult<()>;

    /// Applies a digital filter, returning a new filtered copy and leaving `self`
    /// unchanged.
    ///
    /// Non-mutating twin of [`apply_filter_in_place`](Self::apply_filter_in_place).
    /// The returned audio is shorter than the original by `filter_coeffs.len() - 1`
    /// samples. For integer sample types the taps are limited to `[-1, 1]`
    /// and the sample type's resolution; [`apply_filter_f64`](Self::apply_filter_f64)
    /// takes `f64` taps.
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
    /// let coeffs = [1.0f32, -1.0]; // first difference: y[n] = x[n] - x[n-1]
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .apply_filter(NonEmptySlice::new(&coeffs).unwrap())
    ///     .unwrap();
    /// assert_eq!(audio[0], 1.0); // 2 - 1
    /// assert_eq!(audio[3], 1.0); // 5 - 4
    /// ```
    #[inline]
    fn apply_filter(&self, filter_coeffs: &NonEmptySlice<Self::Sample>) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.apply_filter_in_place(filter_coeffs)?;
        Ok(out)
    }

    /// Applies an FIR filter with `f64` taps by linear convolution, in place.
    ///
    /// Same convolution as [`apply_filter_in_place`](Self::apply_filter_in_place):
    /// the valid part `y[i] = Σ_j h[j] · x[i + M - 1 - j]`, of length
    /// `N − M + 1` (`numpy.convolve(x, h, mode='valid')`), using FFT
    /// convolution for 160 taps or more when the `transforms` feature is
    /// enabled. The taps are `f64` for every sample type, so integer audio can
    /// use taps of any magnitude at full precision. Samples are converted to
    /// `f64` with the audio-aware conversion (integer samples on the
    /// normalised `[-1, 1]` scale); the result is converted back to the sample
    /// type, rounded to nearest and saturated for integer types. For `f32`
    /// and `f64` audio the result equals `apply_filter_in_place` with the
    /// same taps.
    ///
    /// # Arguments
    /// - `filter_coeffs` - FIR filter taps `h[0..M]`. `[1.0]` leaves the
    ///   signal unchanged.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the audio is shorter than
    ///   the filter.
    /// - [`crate::AudioSampleError::Unsupported`] from the default
    ///   implementation, which exists only so that implementors outside this
    ///   crate keep compiling; [`AudioSamples`] overrides
    ///   it.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    /// use non_empty_slice::NonEmptySlice;
    ///
    /// // i16 audio with a tap above 1: y[n] = 1.5·x[n] − 0.5·x[n−1].
    /// let mut audio = AudioSamples::new_mono(array![1000i16, 2000, 3000], sample_rate!(8000)).unwrap();
    /// audio.apply_filter_f64_in_place(NonEmptySlice::new(&[1.5, -0.5]).unwrap()).unwrap();
    /// assert_eq!(audio.samples_per_channel().get(), 2);
    /// assert_eq!(audio[0], 2500); // 1.5·2000 − 0.5·1000
    /// assert_eq!(audio[1], 3500); // 1.5·3000 − 0.5·2000
    /// ```
    #[inline]
    fn apply_filter_f64_in_place(
        &mut self,
        filter_coeffs: &NonEmptySlice<f64>,
    ) -> AudioSampleResult<()> {
        let _ = filter_coeffs;
        Err(crate::AudioSampleError::unsupported(
            "apply_filter_f64_in_place",
            "this AudioProcessing implementation does not provide apply_filter_f64_in_place",
        ))
    }

    /// Applies an FIR filter with `f64` taps, returning a new filtered copy
    /// and leaving `self` unchanged.
    ///
    /// Non-mutating twin of
    /// [`apply_filter_f64_in_place`](Self::apply_filter_f64_in_place). The
    /// returned audio is shorter than the original by `filter_coeffs.len() - 1`
    /// samples and has the same sample type.
    ///
    /// # Arguments
    /// - `filter_coeffs` - FIR filter taps `h[0..M]`.
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
    /// let x = AudioSamples::new_mono(array![1.0f64, 2.0, 3.0, 4.0, 5.0], sample_rate!(8000)).unwrap();
    /// let y = x.apply_filter_f64(NonEmptySlice::new(&[1.0, 0.5]).unwrap()).unwrap();
    /// assert_eq!(y.samples_per_channel().get(), 4);
    /// assert_eq!(y[0], 2.5); // 1.0·2 + 0.5·1
    /// ```
    #[inline]
    fn apply_filter_f64(&self, filter_coeffs: &NonEmptySlice<f64>) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.apply_filter_f64_in_place(filter_coeffs)?;
        Ok(out)
    }

    /// Applies μ-law compression to the audio samples.
    ///
    /// Compresses the dynamic range of the signal using the μ-law transfer
    /// function `y = sign(x) · ln(1 + μ|x|) / ln(1 + μ)`, applied to samples on
    /// the normalised `[-1, 1]` scale (integer samples are converted, companded
    /// and converted back). This equals `librosa.mu_compress(x, mu,
    /// quantize=False)`; for librosa's default integer codes
    /// (`quantize=True`) use [`crate::mu_law_encode`]. Higher `mu` values
    /// produce stronger compression.
    ///
    /// # Arguments
    /// - `mu` - Compression parameter (typically 255 for standard μ-law). Its
    ///   raw numeric value is used for every sample type: for `i16` audio
    ///   `mu = 255` means μ = 255, not 255 / 32768.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `mu` is not finite and
    ///   strictly positive.
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
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.mu_compress_in_place(255.0).unwrap();
    /// assert!(audio[0] > 0.0); // Positive input stays positive
    /// assert!(audio[1] < 0.0); // Negative input stays negative
    /// ```
    fn mu_compress_in_place(&mut self, mu: Self::Sample) -> AudioSampleResult<()>;

    /// Applies μ-law compression, returning a new compressed copy and leaving
    /// `self` unchanged.
    ///
    /// Non-mutating twin of [`mu_compress_in_place`](Self::mu_compress_in_place).
    ///
    /// # Arguments
    /// - `mu` - Compression parameter (typically 255 for standard μ-law); its
    ///   raw numeric value is used for every sample type.
    ///
    /// # Returns
    /// The compressed audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `mu` is not finite and
    ///   strictly positive.
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
    #[inline]
    fn mu_compress(&self, mu: Self::Sample) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.mu_compress_in_place(mu)?;
        Ok(out)
    }

    /// Applies μ-law expansion (decompression) to the audio samples.
    ///
    /// Applies `x = sign(y) · ((1 + μ)^|y| - 1) / μ` on the normalised
    /// `[-1, 1]` scale: the exact inverse of
    /// [`mu_compress`](Self::mu_compress), equal to `librosa.mu_expand(y, mu,
    /// quantize=False)` (for integer codes use [`crate::mu_law_decode`]). The
    /// `mu` parameter must match the value used during compression.
    ///
    /// # Arguments
    /// - `mu` - Expansion parameter. Must match the value used for compression.
    ///   Its raw numeric value is used for every sample type.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `mu` is not finite and
    ///   strictly positive.
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
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.mu_expand_in_place(255.0).unwrap();
    /// assert_eq!(audio[0], 0.0); // Zero maps to zero
    /// assert!(audio[1] > 0.0);   // Sign is preserved
    /// assert!(audio[2] < 0.0);   // Sign is preserved
    /// ```
    fn mu_expand_in_place(&mut self, mu: Self::Sample) -> AudioSampleResult<()>;

    /// Applies μ-law expansion, returning a new expanded copy and leaving `self`
    /// unchanged.
    ///
    /// Non-mutating twin of [`mu_expand_in_place`](Self::mu_expand_in_place).
    ///
    /// # Arguments
    /// - `mu` - Expansion parameter. Must match the value used for compression.
    ///   Its raw numeric value is used for every sample type.
    ///
    /// # Returns
    /// The expanded audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `mu` is not finite and
    ///   strictly positive.
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
    #[inline]
    fn mu_expand(&self, mu: Self::Sample) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.mu_expand_in_place(mu)?;
        Ok(out)
    }

    /// Applies a first-order low-pass filter with the specified cutoff frequency.
    ///
    /// The filter is a first-order Butterworth low-pass designed with the
    /// bilinear transform: -3 dB at `cutoff_hz`, unity gain at DC, a zero at
    /// Nyquist and a 6 dB/octave roll-off. It is applied to each channel
    /// independently from zero initial state, and is identical to
    /// `scipy.signal.sosfilt(scipy.signal.butter(1, cutoff_hz, fs=fs,
    /// output='sos'), x)`. For steeper slopes use
    /// [`AudioIirFiltering::butterworth_lowpass`].
    ///
    /// # Arguments
    /// - `cutoff_hz` - Cutoff (-3 dB) frequency in Hz, in `(0, Nyquist)`.
    ///
    /// # Returns
    /// The filtered audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `cutoff_hz` is not in
    ///   `(0, Nyquist)`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.low_pass_filter_in_place(1000.0).unwrap();
    /// // A Nyquist-rate tone is strongly attenuated
    /// assert!(audio[7].abs() < 0.2);
    /// ```
    fn low_pass_filter_in_place(&mut self, cutoff_hz: f64) -> AudioSampleResult<()>;

    /// Applies a first-order low-pass filter, returning a new filtered copy and
    /// leaving `self` unchanged.
    ///
    /// Non-mutating twin of [`low_pass_filter_in_place`](Self::low_pass_filter_in_place)
    /// (first-order Butterworth, identical to `scipy.signal.butter(1, cutoff_hz)`).
    ///
    /// # Arguments
    /// - `cutoff_hz` - Cutoff (-3 dB) frequency in Hz, in `(0, Nyquist)`.
    ///
    /// # Returns
    /// The filtered audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `cutoff_hz` is not in
    ///   `(0, Nyquist)`.
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
    /// // A Nyquist-rate tone is strongly attenuated
    /// assert!(audio[7].abs() < 0.2);
    /// ```
    #[inline]
    fn low_pass_filter(&self, cutoff_hz: f64) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.low_pass_filter_in_place(cutoff_hz)?;
        Ok(out)
    }

    /// Applies a first-order high-pass filter with the specified cutoff frequency.
    ///
    /// The filter is a first-order Butterworth high-pass designed with the
    /// bilinear transform: -3 dB at `cutoff_hz`, a zero at DC, unity gain at
    /// Nyquist and a 6 dB/octave slope. It is applied to each channel
    /// independently from zero initial state, and is identical to
    /// `scipy.signal.sosfilt(scipy.signal.butter(1, cutoff_hz, 'highpass',
    /// fs=fs, output='sos'), x)`. For steeper slopes use
    /// [`AudioIirFiltering::butterworth_highpass`].
    ///
    /// # Arguments
    /// - `cutoff_hz` - Cutoff (-3 dB) frequency in Hz, in `(0, Nyquist)`.
    ///
    /// # Returns
    /// The filtered audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `cutoff_hz` is not in
    ///   `(0, Nyquist)`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::Array1;
    ///
    /// // A DC (constant) signal decays to zero through a high-pass filter.
    /// let data = Array1::from_elem(4410, 1.0f32);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.high_pass_filter_in_place(100.0).unwrap();
    /// assert!(audio[4409].abs() < 1e-3);
    /// ```
    fn high_pass_filter_in_place(&mut self, cutoff_hz: f64) -> AudioSampleResult<()>;

    /// Applies a first-order high-pass filter, returning a new filtered copy and
    /// leaving `self` unchanged.
    ///
    /// Non-mutating twin of [`high_pass_filter_in_place`](Self::high_pass_filter_in_place)
    /// (first-order Butterworth, identical to `scipy.signal.butter(1, cutoff_hz, 'highpass')`).
    ///
    /// # Arguments
    /// - `cutoff_hz` - Cutoff (-3 dB) frequency in Hz, in `(0, Nyquist)`.
    ///
    /// # Returns
    /// The filtered audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `cutoff_hz` is not in
    ///   `(0, Nyquist)`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::Array1;
    ///
    /// // A DC (constant) signal decays to zero through a high-pass filter.
    /// let data = Array1::from_elem(4410, 1.0f32);
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap()
    ///     .high_pass_filter(100.0)
    ///     .unwrap();
    /// assert!(audio[4409].abs() < 1e-3);
    /// ```
    #[inline]
    fn high_pass_filter(&self, cutoff_hz: f64) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.high_pass_filter_in_place(cutoff_hz)?;
        Ok(out)
    }

    /// Applies a band-pass filter that passes frequencies between the two cutoffs.
    ///
    /// The filter is the band-pass transform of a first-order Butterworth
    /// prototype (a single second-order section): unity gain at the geometric
    /// centre of the (pre-warped) band, -3 dB at both edges, and 6 dB/octave
    /// slopes outside the band. It is identical to
    /// `scipy.signal.sosfilt(scipy.signal.butter(1, [low, high], 'bandpass',
    /// fs=fs, output='sos'), x)`. For steeper slopes use
    /// [`AudioIirFiltering::butterworth_bandpass`].
    ///
    /// # Arguments
    /// - `low_cutoff_hz` - Lower -3 dB edge in Hz.
    /// - `high_cutoff_hz` - Upper -3 dB edge in Hz. Must be greater than
    ///   `low_cutoff_hz` and less than the Nyquist frequency.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] unless
    ///   `0 < low_cutoff_hz < high_cutoff_hz < Nyquist`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.band_pass_filter_in_place(100.0, 5000.0).unwrap();
    /// assert!(audio[0].is_finite());
    /// ```
    fn band_pass_filter_in_place(
        &mut self,
        low_cutoff_hz: f64,
        high_cutoff_hz: f64,
    ) -> AudioSampleResult<()>;

    /// Applies a band-pass filter, returning a new filtered copy and leaving
    /// `self` unchanged.
    ///
    /// Non-mutating twin of [`band_pass_filter_in_place`](Self::band_pass_filter_in_place)
    /// (identical to `scipy.signal.butter(1, [low, high], 'bandpass')`).
    ///
    /// # Arguments
    /// - `low_cutoff_hz` - Lower -3 dB edge in Hz.
    /// - `high_cutoff_hz` - Upper -3 dB edge in Hz. Must be greater than
    ///   `low_cutoff_hz` and less than the Nyquist frequency.
    ///
    /// # Returns
    /// The filtered audio samples.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] unless
    ///   `0 < low_cutoff_hz < high_cutoff_hz < Nyquist`.
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
    #[inline]
    fn band_pass_filter(&self, low_cutoff_hz: f64, high_cutoff_hz: f64) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.band_pass_filter_in_place(low_cutoff_hz, high_cutoff_hz)?;
        Ok(out)
    }

    /// Removes DC offset by subtracting the mean value, mutating `self` in place.
    ///
    /// Computes `x'[n] = x[n] − x̄`, where `x̄` is the mean over all samples
    /// (all channels together). This centers the audio around zero and
    /// removes any constant bias that may have been introduced during
    /// recording or processing. Only the 0 Hz bin of a DFT over the whole
    /// buffer changes; for a drifting offset use
    /// [`high_pass_filter`](Self::high_pass_filter) instead.
    ///
    /// The subtraction is done in `f64` on the raw sample values; for integer
    /// sample types the result is rounded to nearest and saturated.
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
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.remove_dc_offset_in_place().unwrap();
    /// let mean: f64 = audio.mean();
    /// assert!(mean.abs() < 1e-6); // Mean is now ~0
    /// ```
    fn remove_dc_offset_in_place(&mut self) -> AudioSampleResult<()>;

    /// Removes DC offset, returning a new copy and leaving `self` unchanged.
    ///
    /// Non-mutating twin of [`remove_dc_offset_in_place`](Self::remove_dc_offset_in_place).
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
    #[inline]
    fn remove_dc_offset(&self) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.remove_dc_offset_in_place()?;
        Ok(out)
    }

    /// Clips all samples in-place to `[min_val, max_val]`.
    ///
    /// Any samples outside `[min_val, max_val]` are clamped to the nearest
    /// boundary. Useful for preventing digital clipping before output. This is
    /// the in-place primitive; [`clip`](Self::clip) is the non-mutating twin.
    ///
    /// # Arguments
    /// - `min_val` - lower bound; samples below this value are set to `min_val`.
    /// - `max_val` - upper bound; samples above this value are set to `max_val`.
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
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.clip_in_place(-1.0f32, 1.0f32).unwrap();
    /// assert_eq!(audio[0], 1.0);
    /// assert_eq!(audio[1], -1.0);
    /// assert_eq!(audio[3], -0.5);
    /// ```
    fn clip_in_place(
        &mut self,
        min_val: Self::Sample,
        max_val: Self::Sample,
    ) -> AudioSampleResult<()>;

    /// Clips audio samples to the specified range, returning a new clipped copy
    /// and leaving `self` unchanged.
    ///
    /// Non-mutating twin of [`clip_in_place`](Self::clip_in_place). Any samples
    /// outside `[min_val, max_val]` are clamped to the nearest boundary.
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
    #[inline]
    fn clip(&self, min_val: Self::Sample, max_val: Self::Sample) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.clip_in_place(min_val, max_val)?;
        Ok(out)
    }

    /// Resamples audio to a new sample rate.
    ///
    /// Uses the crate's windowed-sinc resampler (see [`crate::resampling`]): the
    /// output is time-aligned with the input and has exactly
    /// `ceil(n · target / input)` samples per channel. The `quality` parameter
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

    /// Resamples audio to a new sample rate in place, replacing the internal
    /// buffer with the resampled result.
    ///
    /// Explicit in-place twin of [`resample`](Self::resample). Because resampling
    /// produces a different-length buffer it cannot reuse the existing storage;
    /// this method computes the new buffer via [`resample`](Self::resample) and
    /// assigns it back to `self`.
    ///
    /// # Arguments
    /// - `target_sample_rate` - Desired output sample rate.
    /// - `quality` - Resampling quality preset (see [`ResamplingQuality`]).
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the target sample rate or
    ///   quality parameters are invalid.
    /// - [`crate::AudioSampleError::Layout`] if the input audio is empty.
    #[cfg(feature = "resampling")]
    fn resample_in_place(
        &mut self,
        target_sample_rate: SampleRate,
        quality: ResamplingQuality,
    ) -> AudioSampleResult<()>;

    /// Resamples audio by a specific ratio.
    ///
    /// The output length is scaled by `ratio` relative to the input. A ratio
    /// of 2.0 doubles the sample count; 0.5 halves it.
    ///
    /// Uses the crate's windowed-sinc resampler (see [`crate::resampling`]): the
    /// output is time-aligned with the input and has exactly
    /// `ceil(n · target / input)` samples per channel. The `quality` parameter
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

    /// Resamples audio by a specific ratio in place, replacing the internal
    /// buffer with the resampled result.
    ///
    /// Explicit in-place twin of [`resample_by_ratio`](Self::resample_by_ratio).
    /// Because resampling produces a different-length buffer it cannot reuse the
    /// existing storage; this method computes the new buffer via
    /// [`resample_by_ratio`](Self::resample_by_ratio) and assigns it back to `self`.
    ///
    /// # Arguments
    /// - `ratio` - Resampling ratio (`output_rate / input_rate`). Must be > 0.
    /// - `quality` - Resampling quality preset (see [`ResamplingQuality`]).
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `ratio` is ≤ 0 or the
    ///   quality parameters are invalid.
    /// - [`crate::AudioSampleError::Layout`] if the input audio is empty.
    #[cfg(feature = "resampling")]
    fn resample_by_ratio_in_place(
        &mut self,
        ratio: f64,
        quality: ResamplingQuality,
    ) -> AudioSampleResult<()>;
}

/// Frequency-domain analysis and spectral transformation operations.
///
/// # Purpose
///
/// Provides FFT, STFT, and a range of spectral representations -
/// linear, log-frequency, mel-scale, MFCC, chromagram, gammatone, and
/// constant-Q - all delegating to the `spectrograms` crate for numerics.
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
/// # Conventions
///
/// Forward transforms are unnormalised one-sided DFTs,
/// X\[k\] = Σ_n x\[n\]·e^{−j2πkn/N}, with bin k at k·f_s/N Hz. `WindowType::Hanning`
/// is the **symmetric** Hann window (librosa/SciPy default to the periodic one).
/// Spectrogram dB values are 10·log10(max(power, 10^(floor_db/10))) re 1 (an
/// absolute floor, no `top_db`). Mel bands use the Slaney mel scale applied to
/// power, without filter normalisation by default; MFCCs use an unnormalised
/// DCT-II and an HTK lifter (L = 22) by default. The
/// [`crate::operations::transforms`] module docs give every formula and the
/// differences from librosa.
///
/// # Invariants
///
/// Methods that accept an FFT length (`n_fft`) zero-pad shorter signals rather
/// than truncating.  Signals *longer* than `n_fft` are rejected with an error
/// (propagated from the `spectrograms` crate).
#[cfg(feature = "transforms")]
pub trait AudioTransforms: AudioTypeConversion
where
    Self::Sample: StandardSample,
{
    /// Computes the Fast Fourier Transform of the audio signal.
    ///
    /// Each channel is transformed independently. Each row holds the
    /// non-negative-frequency half of the unnormalised DFT,
    /// X\[k\] = Σ_n x\[n\]·e^{−j2πkn/N} for k = 0 … ⌊N/2⌋ (N = `n_fft`); bin k is at
    /// k·f_s/N Hz (see [`crate::utils::audio_math::fft_frequencies`]). No window
    /// is applied. Like `numpy.fft.rfft`, there is no 1/N scaling.
    ///
    /// # Arguments
    /// - `n_fft` - FFT length in samples. If longer than the signal the input
    ///   is zero-padded internally; a signal longer than `n_fft` is an error.
    ///
    /// # Returns
    /// An `Array2<Complex<f64>>` of shape `(channels, n_fft / 2 + 1)`.
    ///
    /// # Errors
    /// Returns an error if the signal is longer than `n_fft` or the FFT
    /// computation fails.
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

    /// Linear convolution of this signal with `other` via FFT.
    ///
    /// Both signals are converted to `f64` internally. The result is mono and
    /// carries this signal's sample rate, with length
    /// `self.len() + other.len() - 1`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if either signal is multi-channel.
    /// - Errors propagated from the underlying FFT.
    fn convolve(&self, other: &Self) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// Regularised spectral-division deconvolution: recover the system response
    /// `h` from `self = denominator ⊛ h`.
    ///
    /// `regularization` (≥ 0) is scaled by the denominator's peak power to
    /// stabilise division near spectral nulls. Result is mono at this signal's
    /// sample rate.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if either signal is multi-channel.
    /// - Errors propagated from the underlying FFT.
    fn deconvolve(
        &self,
        denominator: &Self,
        regularization: f64,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// Computes the magnitude spectrum of the audio signal.
    ///
    /// Equivalent to [`fft`] followed by taking the absolute value of each complex bin,
    /// |X\[k\]| for k = 0 … ⌊N/2⌋. Each channel is transformed independently.
    ///
    /// Note: unlike `numpy.fft.rfft`, this returns real magnitudes |X\[k\]|, not the
    /// complex one-sided spectrum; use [`fft`] for the complex values.
    ///
    /// # Arguments
    ///
    /// - `n_fft` - FFT length in samples. If longer than the signal the input is
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
    /// For librosa-compatible results (periodic window, multichannel) see [`AudioSpectrum::stft_matrix`].
    ///
    /// Computes X\[k, m\] = Σ_n x_p\[n + mH\]·w\[n\]·e^{−j2πkn/N} (N = `n_fft`,
    /// H = `hop_size`), an unnormalised one-sided DFT of each windowed frame
    /// (⌊N/2⌋ + 1 bins × frames). `WindowType::Hanning` is the **symmetric** Hann
    /// window 0.5 − 0.5·cos(2πn/(N − 1)), not the periodic Hann used by default in
    /// librosa and SciPy. With `centre = true` the signal is zero-padded by ⌊N/2⌋
    /// samples on both sides, so frame m is centred on sample mH. The result equals
    /// `librosa.stft(x, n_fft=N, hop_length=H, window=scipy.signal.get_window('hann',
    /// N, fftbins=False), pad_mode='constant')`. See the
    /// [`crate::operations::transforms`] module docs for the frame count. The
    /// returned [`StftResult`] carries both the complex matrix and the parameters
    /// required by [`istft`] for reconstruction.
    ///
    /// # Arguments
    /// - `params` - STFT configuration (FFT size, hop size, window function,
    ///   and centering behaviour). Field-level constraints are documented on
    ///   [`StftParams`].
    ///
    /// # Returns
    /// An [`StftResult`] whose `data` has shape `(n_fft / 2 + 1, n_frames)`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
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
    /// For librosa-compatible results (periodic window, multichannel) see [`AudioSpectrum::istft_matrix`].
    ///
    /// Synthesis is windowed overlap-add normalised by Σ_m w²[n − mH] (the
    /// least-squares inverse STFT), using the window and hop stored in the
    /// [`StftResult`]; reconstruction is exact wherever that sum is non-zero.
    /// The output is a mono signal at the sample rate recorded during the
    /// forward transform. Its length is (n_frames − 1)·H + N − 2⌊N/2⌋ with
    /// centring, which can be up to `hop − 1` samples shorter than the original;
    /// trim or pad to the original length if needed.
    ///
    /// # Arguments
    /// - `stft` - the [`StftResult`] produced by a prior call to [`stft`].
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
    /// Rows are the STFT bins (bin k at k·f_s/N Hz). `Power` gives |X|², `Magnitude`
    /// gives |X|, and `Decibels` gives 10·log10(max(|X|², 10^(floor_db/10))) re 1
    /// (an absolute floor, not a range below the peak; see the
    /// [`crate::operations::transforms`] module docs).
    /// Prefer the typed convenience methods -
    /// [`linear_magnitude_spectrogram`](crate::operations::AudioTransforms::linear_magnitude_spectrogram), [`linear_power_spectrogram`](crate::operations::AudioTransforms::linear_power_spectrogram), or
    /// [`linear_db_spectrogram`](crate::operations::AudioTransforms::linear_db_spectrogram) - for the most common amplitude scales.
    ///
    /// # Arguments
    /// - `params` - spectrogram parameters (window, hop, FFT size).
    /// - `db` - required when `AmpScale` is `Decibels`; ignored otherwise.
    ///   Its `floor_db` is an absolute power floor in dB re 1.
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
    /// Values are 10·log10(max(P, 10^(floor_db/10))) of the power P, in dB re 1:
    /// `floor_db` is an absolute floor, not a range below the peak (no `top_db`).
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
    /// Prefer the typed convenience methods -
    /// [`loghz_power_spectrogram`](crate::operations::AudioTransforms::loghz_power_spectrogram), [`loghz_magnitude_spectrogram`](crate::operations::AudioTransforms::loghz_magnitude_spectrogram), or
    /// [`loghz_db_spectrogram`](crate::operations::AudioTransforms::loghz_db_spectrogram) - for the most common amplitude scales.
    ///
    /// # Arguments
    /// - `params` - spectrogram parameters (window, hop, FFT size).
    /// - `loghz` - log-Hz frequency-axis configuration (min/max
    ///   frequency, number of bins). Field-level constraints are
    ///   documented on [`LogHzParams`].
    /// - `db` - required when `AmpScale` is `Decibels`; ignored otherwise.
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
    /// Values are 10·log10(max(P, 10^(floor_db/10))) of the power P, in dB re 1:
    /// `floor_db` is an absolute floor, not a range below the peak (no `top_db`).
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
    /// For librosa-compatible results (periodic window, multichannel) see [`AudioSpectrum::melspectrogram`].
    ///
    /// The mel scale approximates human pitch perception by compressing
    /// high frequencies relative to low ones. Bands use the **Slaney** mel scale
    /// (`librosa.hz_to_mel(htk=False)`) with triangular filters in Hz, applied to the
    /// **power** spectrogram |X|². `MelParams::new` applies no filter normalisation
    /// (librosa's default is `norm='slaney'`; use `MelParams::with_norm(..,
    /// MelNorm::Slaney)` to match it). The `Magnitude` output is the square root of
    /// mel power, not a mel-weighted sum of magnitudes; `Decibels` is
    /// 10·log10(max(mel power, 10^(floor_db/10))) re 1. See the
    /// [`crate::operations::transforms`] module docs for the formulas.
    ///
    /// Note: in `spectrograms` 2.1.3 the result's `frequencies()` axis is only
    /// correct when `f_min = 0` and `f_max` = Nyquist; for other ranges compute the
    /// band centres yourself (the inner `n_mels` of `n_mels + 2` points equally
    /// spaced on the Slaney mel scale between `f_min` and `f_max`).
    ///
    /// Prefer the typed
    /// convenience methods - [`mel_mag_spectrogram`](crate::operations::AudioTransforms::mel_mag_spectrogram),
    /// [`mel_power_spectrogram`](crate::operations::AudioTransforms::mel_power_spectrogram), or [`mel_db_spectrogram`](crate::operations::AudioTransforms::mel_db_spectrogram) - for the
    /// most common amplitude scales.
    ///
    /// # Arguments
    /// - `params` - spectrogram parameters (window, hop, FFT size).
    /// - `mel` - mel filter-bank configuration (number of bands,
    ///   frequency range). Field-level constraints are documented on
    ///   [`MelParams`].
    /// - `db` - required when `AmpScale` is `Decibels`; ignored otherwise.
    ///
    /// # Returns
    /// A [`MelSpectrogram<AmpScale>`].
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying spectrogram computation.
    ///
    /// ## See Also
    /// - [Mel scale - Wikipedia](https://en.wikipedia.org/wiki/Mel_scale)
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
    /// Values are 10·log10(max(P, 10^(floor_db/10))) of the power P, in dB re 1:
    /// `floor_db` is an absolute floor, not a range below the peak (no `top_db`).
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
    /// For librosa-compatible results (periodic window, multichannel) see [`AudioSpectrum::mfcc_frames`].
    ///
    /// MFCCs are a compact spectral representation widely used in speech
    /// recognition and audio classification. The pipeline is: power mel
    /// spectrogram (Slaney scale, `n_mels` bands from 0 Hz to f_s/2, no filter
    /// normalisation) → 10·log10(max(·, 10⁻⁸)) → DCT-II **without** orthonormal
    /// scaling, c\[q\] = Σ_b L\[b\]·cos(πq(b + ½)/`n_mels`), keeping the first
    /// `n_mfcc` coefficients → HTK sinusoidal lifter 1 + (L/2)·sin(πq/L) with
    /// **L = 22 by default** (`MfccParams::with_lifter(0)` disables it).
    /// `with_c0(false)` drops c0 and returns `n_mfcc − 1` coefficients.
    ///
    /// Note: values therefore differ from `librosa.feature.mfcc` (orthonormal DCT,
    /// no lifter, `top_db = 80`) by a factor √(`n_mels`/2) for q ≥ 1 (√`n_mels` for
    /// c0) and by the lifter weights.
    ///
    /// # Arguments
    /// - `stft_params` - STFT configuration used for the underlying
    ///   spectrogram.
    /// - `n_mels` - number of mel filter-bank bands.
    /// - `mfcc_params` - MFCC-specific configuration (number of
    ///   coefficients, etc.). Field-level constraints are documented on
    ///   [`MfccParams`].
    ///
    /// # Returns
    /// An [`Mfcc`] whose `data` has shape `(n_mfcc, n_frames)` (or
    /// `(n_mfcc − 1, n_frames)` without c0).
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying computation, including `n_mfcc > n_mels`.
    ///
    /// ## See Also
    /// - [MFCC - Wikipedia](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
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
    /// - `stft_params` - STFT configuration used for the underlying
    ///   spectrogram.
    /// - `cfg` - chromagram configuration (tuning, normalization,
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
    /// The signal is split into segments of N = `window_size` samples with hop
    /// H = max(1, ⌊(1 − `overlap`)·N⌋); incomplete trailing segments are dropped.
    /// Each segment is multiplied by the **symmetric** Hann window w\[n\]
    /// (`WindowType::Hanning`), with no detrending, and transformed with an
    /// N-point DFT X_s\[k\]. The one-sided density estimate is
    ///
    /// ```text
    /// P[k] = c_k · mean_s |X_s[k]|² / (f_s · Σ_n w[n]²),   c_k = 1 at DC and Nyquist, 2 otherwise
    /// ```
    ///
    /// in units² per Hz (units of full scale), at f_k = k·f_s/N. This matches
    /// `scipy.signal.welch(x, fs, window=scipy.signal.windows.hann(N, sym=True),
    /// nperseg=N, noverlap=N − H, detrend=False, scaling='density')`, and
    /// Σ_k P\[k\]·f_s/N ≈ the mean square of the signal (Parseval). Note: SciPy's
    /// defaults differ (periodic Hann, `detrend='constant'`).
    ///
    /// # Arguments
    /// - `window_size` - segment length N in samples. Must be at least 2 and must
    ///   not exceed the signal length.
    /// - `overlap` - fractional overlap between adjacent segments, in
    ///   the range `[0, 1)`.
    ///
    /// # Returns
    /// A [`Psd`](crate::Psd) whose `frequencies()` and `density()` slices both have
    /// length ⌊N/2⌋ + 1. `frequencies[k]` = k·f_s/N in Hz; `density[k]` is P\[k\] in
    /// units²/Hz.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - [`crate::AudioSampleError::Parameter`] if `overlap` is outside `[0, 1)`,
    ///   or if `window_size` is 1 or exceeds the signal length.
    ///
    /// ## See Also
    /// - [Welch's method - Wikipedia](https://en.wikipedia.org/wiki/Welch%27s_method)
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTransforms, nzu, sample_rate, sine_wave};
    /// use std::time::Duration;
    ///
    /// let audio = sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.8);
    /// let psd = audio.power_spectral_density(nzu!(1024), 0.5).unwrap();
    /// assert_eq!(psd.frequencies().len(), psd.density().len());
    /// assert!(!psd.frequencies().is_empty());
    /// ```
    fn power_spectral_density(
        &self,
        window_size: NonZeroUsize,
        overlap: f64,
    ) -> AudioSampleResult<crate::Psd>;

    /// Computes a gammatone-filtered spectrogram.
    ///
    /// Gammatone filters model the bandpass response of the human
    /// cochlea. Note: with the default `ErbSpacing::Linear`, `spectrograms` 2.1.3
    /// spaces the centre frequencies uniformly in ERB **bandwidth**
    /// 24.7·(4.37 f/1000 + 1), which is linear in Hz, **not** on the ERB-number
    /// (Cam) scale 21.4·log10(1 + 0.00437 f) (an upstream bug). For auditory
    /// (ERB-rate) spacing use
    /// `GammatoneParams::new(..)?.with_spacing(ErbSpacing::AppleTr35)` (Slaney /
    /// Patterson spacing, uniform in ln(f + 228.7), equivalent to the
    /// Glasberg & Moore ERB-number scale). Prefer the
    /// typed convenience methods -
    /// [`gammatone_magnitude_spectrogram`](crate::operations::AudioTransforms::gammatone_magnitude_spectrogram),
    /// [`gammatone_power_spectrogram`](crate::operations::AudioTransforms::gammatone_power_spectrogram), or
    /// [`gammatone_db_spectrogram`](crate::operations::AudioTransforms::gammatone_db_spectrogram) - for the most common amplitude
    /// scales.
    ///
    /// # Arguments
    /// - `params` - spectrogram parameters (window, hop, FFT size).
    /// - `gammatone_params` - gammatone filter-bank configuration
    ///   (number of bands, frequency range). Field-level constraints are
    ///   documented on [`GammatoneParams`].
    /// - `db` - required when `AmpScale` is `Decibels`; ignored otherwise.
    ///
    /// # Returns
    /// A `Spectrogram<Gammatone, AmpScale>`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying computation.
    ///
    /// ## See Also
    /// - [Gammatone filter - Wikipedia](https://en.wikipedia.org/wiki/Gammatone)
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
    /// Values are 10·log10(max(P, 10^(floor_db/10))) of the power P, in dB re 1:
    /// `floor_db` is an absolute floor, not a range below the peak (no `top_db`).
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
    /// Bin k has centre frequency f_k = f_min·2^{k/b} (b = bins per octave;
    /// geometric spacing with ratio 2^{1/b}) and a constant quality factor
    /// Q = f_k/Δf_k = 1/(2^{1/b} − 1) (≈ 16.8 for b = 12), so its frequency
    /// resolution follows the musical scale, which suits pitch and harmonic
    /// analysis. The kernel for bin k is about Q·f_s/f_k samples long, so low
    /// bins have long kernels (good frequency, poor time resolution).
    ///
    /// Note (`spectrograms` 2.1.3): frames are `min(signal_len, 16384)` samples
    /// long and are not centred, so a signal shorter than 16384 samples yields a
    /// single frame whatever `hop_size` is; kernels longer than the frame are
    /// truncated, which lowers the effective Q at low frequencies.
    ///
    /// # Arguments
    /// - `params` - CQT parameters (frequency range, bins per octave,
    ///   etc.). Field-level constraints are documented on [`CqtParams`].
    /// - `hop_size` - hop length in samples between successive frames (only
    ///   relevant for signals longer than the 16384-sample frame).
    ///
    /// # Returns
    /// A [`CqtResult`] containing the CQT matrix.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the signal is multi-channel.
    /// - Errors from the underlying CQT computation.
    ///
    /// ## See Also
    /// - [Constant-Q transform - Wikipedia](https://en.wikipedia.org/wiki/Constant-Q_transform)
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
    /// spectrogram.  Prefer the typed convenience methods -
    /// [`cqt_magnitude_spectrogram`](crate::operations::AudioTransforms::cqt_magnitude_spectrogram), [`cqt_power_spectrogram`](crate::operations::AudioTransforms::cqt_power_spectrogram), or
    /// [`cqt_db_spectrogram`](crate::operations::AudioTransforms::cqt_db_spectrogram) - for the most common amplitude scales.
    ///
    /// # Arguments
    /// - `params` - spectrogram parameters (window, hop, FFT size).
    /// - `cqt` - CQT parameters (frequency range, bins per octave,
    ///   etc.). Field-level constraints are documented on [`CqtParams`].
    /// - `db` - required when `AmpScale` is `Decibels`; ignored otherwise.
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
    /// Values are 10·log10(max(P, 10^(floor_db/10))) of the power P, in dB re 1:
    /// `floor_db` is an absolute floor, not a range below the peak (no `top_db`).
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
    /// Given a complex matrix `D`, returns `(S, P)` with S = |D|^p (p = `power`)
    /// and P = D/|D| (unit-magnitude phase factors), so that `D = S^(1/p) · P`
    /// elementwise (`D = S · P` for the default p = 1).
    /// Bins where the magnitude is zero are assigned a phase of `1 + 0i`.
    ///
    /// # Arguments
    /// - `complex_spect` - the complex STFT or FFT matrix.
    /// - `power` - exponent applied to the magnitude values.  `None`
    ///   defaults to 1 (raw magnitude).
    ///
    /// # Returns
    /// `(magnitude, phase)` - the magnitude matrix (real-valued) and
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
/// Use this trait for pitch-based analysis of monophonic audio - for example
/// to extract a melody contour from a vocal recording, to measure intonation
/// accuracy, or to classify the musical key of a short clip.
///
/// # Invariants
///
/// All methods require mono audio.  Multi-channel signals must be reduced to
/// mono (e.g. with [`AudioChannelOps::to_mono`]) before use; passing a
/// multi-channel signal to the pitch and harmonic methods returns
/// [crate::AudioSampleError::Layout]. The YIN and autocorrelation formulas,
/// and the relation between the lag range and the frequency range, are given
/// in the [`pitch_analysis`](crate::operations::pitch_analysis) module docs.
///
/// For librosa-compatible frame-wise pitch tracks (`librosa.yin`, `librosa.pyin`:
/// centred frames, multichannel, voicing probabilities) see
/// [`AudioPitchTracking`].
#[cfg(feature = "pitch-analysis")]
pub trait AudioPitchAnalysis: AudioTypeConversion
where
    Self::Sample: StandardSample,
{
    /// Detects the fundamental frequency using the YIN pitch detection algorithm.
    ///
    /// The whole signal is analysed as one frame. YIN computes the cumulative
    /// mean normalised difference d′(τ), takes the first lag in the search
    /// range whose d′ falls below `threshold`, descends to the bottom of that
    /// dip and refines it by parabolic interpolation, so the period (and hence
    /// the frequency) is not restricted to whole samples. Lower thresholds are
    /// stricter (fewer false detections, more `None`); values in `[0.1, 0.2]`
    /// are typical.
    ///
    /// # Arguments
    ///
    /// - `threshold` - Absolute threshold on d′ (dimensionless), in `[0.0, 1.0]`.
    /// - `min_frequency` - Lowest frequency searched, in Hz (> 0). Sets the
    ///   longest lag `⌊f_s/min_frequency⌋`; the signal must be longer than
    ///   twice that lag.
    /// - `max_frequency` - Highest frequency searched, in Hz (> `min_frequency`).
    ///   Sets the shortest lag `⌊f_s/max_frequency⌋`. Values above Nyquist are
    ///   not rejected.
    ///
    /// # Returns
    ///
    /// - `Some(frequency_hz)` - Estimated fundamental frequency `f_s/τ̂` in Hz.
    /// - `None` - Signal of at most `2·⌊f_s/min_frequency⌋ + 1` samples,
    ///   silent, or no lag with d′ below `threshold` (unvoiced).
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Layout] for multi-channel audio.
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
    /// assert!((pitch.unwrap() - hz).abs() < 1.0);
    /// ```
    fn detect_pitch_yin(
        &self,
        threshold: f64,
        min_frequency: f64,
        max_frequency: f64,
    ) -> AudioSampleResult<Option<f64>>;

    /// Detects the fundamental frequency using autocorrelation.
    ///
    /// The whole signal is analysed as one frame. Finds the integer lag τ̂ that
    /// maximises the raw autocorrelation `r(τ) = Σ x[n]·x[n+τ]` within the lag
    /// range implied by `[min_frequency, max_frequency]` and returns `f_s/τ̂`.
    /// There is no sub-sample interpolation, so the result lies on the grid
    /// `f_s/k` (spacing ≈ f₀²/f_s). Fast and effective for clean, periodic
    /// signals, but less robust than YIN on noisy signals or speech.
    ///
    /// # Arguments
    ///
    /// - `min_frequency` - Lowest frequency searched, in Hz (> 1.0).
    /// - `max_frequency` - Highest frequency searched, in Hz (> `min_frequency`).
    ///
    /// # Returns
    ///
    /// - `Some(frequency_hz)` - Estimated fundamental frequency in Hz.
    /// - `None` - Signal of at most `2·⌊f_s/min_frequency⌋ + 1` samples,
    ///   silent, or no positive autocorrelation in the lag range.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Layout] for multi-channel audio.
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
    /// The signal is split into frames of `window_size` samples starting at
    /// `0, hop_size, 2·hop_size, …` (no padding, no window function). Each frame
    /// is analysed independently with the same algorithm as
    /// [`detect_pitch_yin`](AudioPitchAnalysis::detect_pitch_yin) or
    /// [`detect_pitch_autocorr`](AudioPitchAnalysis::detect_pitch_autocorr).
    /// The final, truncated frames are kept while they are at least
    /// `window_size / 2` samples long. A frame is unvoiced (`None`) when it is
    /// at most `2·⌊f_s/min_frequency⌋ + 1` samples long, so choose
    /// `window_size` > 2 periods of the lowest f₀ of interest. Only
    /// [`PitchDetectionMethod::Yin`] and [`PitchDetectionMethod::Autocorrelation`]
    /// are implemented; other variants return an error.
    ///
    /// # Arguments
    ///
    /// - `window_size` - Analysis frame length in samples; must be ≤ signal length.
    /// - `hop_size` - Step between successive frames in samples; must be < `window_size`.
    /// - `method` - Pitch detection algorithm to use per frame.
    /// - `threshold` - YIN threshold on d′; ignored for autocorrelation.
    /// - `min_frequency` - Lowest frequency searched, in Hz.
    /// - `max_frequency` - Highest frequency searched, in Hz.
    ///
    /// # Returns
    ///
    /// A [`PitchContour`](crate::PitchContour) of per-frame
    /// [`PitchFrame`](crate::PitchFrame)s in time order. `PitchFrame::time` is
    /// the frame **start** time `k·hop_size/f_s` in seconds (not the frame
    /// centre); `PitchFrame::frequency` is `None` when no pitch was found.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Layout] for multi-channel audio.
    /// Returns [crate::AudioSampleError::Parameter] if `window_size > signal length`,
    /// `hop_size >= window_size`, the threshold or frequency range is invalid, or
    /// `method` is [`PitchDetectionMethod::Cepstrum`] or
    /// [`PitchDetectionMethod::HarmonicProduct`] (not implemented).
    ///
    /// # Example
    ///
    /// ```
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
    /// assert!((track.mean_pitch().unwrap() - hz).abs() < 1.0);
    /// ```
    fn track_pitch(
        &self,
        window_size: NonZeroUsize,
        hop_size: NonZeroUsize,
        method: PitchDetectionMethod,
        threshold: f64,
        min_frequency: f64,
        max_frequency: f64,
    ) -> AudioSampleResult<crate::PitchContour>;

    /// Computes a spectral harmonic-to-noise ratio (HNR) in decibels.
    ///
    /// One windowed power spectrum P\[k\] of the whole signal is computed
    /// (bin spacing Δf = f_s/`n_fft` Hz). Bin k (frequency k·Δf, DC included)
    /// counts as *harmonic* when |k·Δf − h·f₀| < Δf for some
    /// h = 1..=`num_harmonics`; every other bin counts as *noise*, and
    /// `HNR = 10·log10(Σ_harmonic P[k] / Σ_noise P[k])` dB.
    ///
    /// Window leakage outside that ±1-bin neighbourhood is counted as noise,
    /// so the estimate is biased low and even a pure tone gives a small finite
    /// value that depends on where the tone falls relative to the bins (≈3 dB
    /// for an on-bin sine with the default Hann window, about 13 dB off-bin).
    /// Compare values computed with the same `n_fft` and window only. This is
    /// not the autocorrelation HNR of Boersma (1993) used by Praat.
    ///
    /// # Arguments
    ///
    /// - `fundamental_freq` - Known fundamental frequency in Hz (> 0).
    /// - `num_harmonics` - Number of harmonics to accumulate into harmonic power.
    /// - `n_fft` - FFT size. Defaults to the signal length when `None`.
    /// - `window_type` - Window function applied before FFT. Defaults to Hanning when `None`.
    ///
    /// # Returns
    ///
    /// HNR in dB. Returns `f64::INFINITY` only when every bin outside the
    /// harmonic neighbourhoods is exactly zero.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Layout] for multi-channel audio.
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
    /// // Finite, not infinite: window leakage is partly counted as noise.
    /// assert!(hnr > 0.0 && hnr.is_finite(), "got {hnr:.1} dB");
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
    /// - `fundamental_freq` - Fundamental frequency in Hz (> 0).
    /// - `num_harmonics` - Number of harmonics to extract, including the fundamental.
    /// - `tolerance` - Fractional bandwidth around each harmonic to search, in `[0.0, 1.0]`.
    /// - `n_fft` - FFT size. Defaults to the signal length when `None`.
    /// - `window_type` - Window function applied before FFT. Defaults to Hanning when `None`.
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` of length `num_harmonics`. Index 0 is always 1.0 after
    /// normalisation.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Layout] for multi-channel audio.
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
    /// Computes a chromagram and averages it over all frames into a 12-element
    /// chroma vector c (index 0 = C). For each of the 24 keys the
    /// Krumhansl-Kessler major or minor profile p (index 0 = tonic) is rotated
    /// so the tonic weight lands on the key's pitch class t,
    /// `p_t[i] = p[(i − t) mod 12]`, and the Pearson correlation ρ(c, p_t) is
    /// computed (Krumhansl-Schmuckler algorithm). The key with the largest ρ is
    /// returned and `confidence = (ρ + 1)/2`.
    ///
    /// # Arguments
    ///
    /// - `stft_params` - STFT parameters controlling frame size and hop for
    ///   chromagram computation.
    ///
    /// # Returns
    ///
    /// A [`Key`](crate::Key) with the estimated `tonic`
    /// ([`PitchClass`](crate::PitchClass)), `mode` ([`Mode`](crate::Mode)),
    /// and a `confidence` in `[0.0, 1.0]`.
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
    /// let key = audio.estimate_key(&params).unwrap();
    /// assert!((0.0..=1.0).contains(&key.confidence));
    /// ```
    fn estimate_key(&self, stft_params: &StftParams) -> AudioSampleResult<crate::Key>;
}

/// IIR (Infinite Impulse Response) filtering operations.
///
/// # Purpose
///
/// Applies recursive digital filters - Butterworth and Chebyshev Type I - to
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
/// audio.butterworth_lowpass_in_place(NonZeroUsize::new(2).unwrap(), 5000.0).unwrap();
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
    /// rate), then applies it to every channel independently, each starting
    /// from zero state. Equivalent to `scipy.signal.sosfilt(sos, x)` with the
    /// SOS coefficients SciPy would design for the same specification.
    ///
    /// # Arguments
    /// - `design` - Filter specification (type, order, frequencies, ripple).
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if any frequency in `design`
    ///   is out of the valid range (0, Nyquist), a required ripple /
    ///   attenuation is missing, or the order exceeds the supported maximum.
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
    /// assert!(audio.apply_iir_filter_in_place(&design).is_ok());
    /// ```
    fn apply_iir_filter_in_place(&mut self, design: &IirFilterDesign) -> AudioSampleResult<()>;

    /// Apply an IIR filter, returning a new filtered copy and leaving `self` unchanged.
    ///
    /// Non-mutating twin of [`apply_iir_filter_in_place`](Self::apply_iir_filter_in_place).
    fn apply_iir_filter(&self, design: &IirFilterDesign) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.apply_iir_filter_in_place(design)?;
        Ok(out)
    }

    /// Apply zero-phase (forward-backward) IIR filtering in place.
    ///
    /// Designs the filter from `design`, then filters each channel forward,
    /// reverses, filters again, and reverses back. The two passes cancel the
    /// phase response (giving exactly **zero phase delay**) and **square** the
    /// magnitude response relative to a single pass. This is the SciPy
    /// `filtfilt` algorithm.
    ///
    /// Matches `scipy.signal.sosfiltfilt(sos, x)` exactly: each channel is
    /// extended at both ends by odd (point-)reflection of
    /// `3 · (2·n_sections + 1)` samples (fewer when sections are first
    /// order), and each pass starts from the filter's steady-state initial
    /// conditions scaled by the first sample, which suppresses start-up
    /// transients at the edges. The padding is stripped afterwards. Signals
    /// shorter than the pad length (which SciPy rejects) are padded with
    /// `len - 1` samples instead. Integer sample types are processed through
    /// an `f64` intermediate.
    ///
    /// Unlike [`apply_iir_filter_in_place`](Self::apply_iir_filter_in_place),
    /// which delays the signal by the filter's group delay, `filtfilt` leaves a
    /// symmetric input symmetric.
    ///
    /// # Arguments
    /// - `design` - Filter specification (type, order, frequencies, ripple).
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if the design is invalid
    ///   (out-of-range frequency, missing ripple/attenuation, order above the
    ///   supported maximum, …).
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate, sine_wave};
    /// use audio_samples::operations::traits::AudioIirFiltering;
    /// use audio_samples::operations::types::IirFilterDesign;
    /// use std::num::NonZeroUsize;
    /// use std::time::Duration;
    ///
    /// // A 100 Hz tone well inside the passband comes back with no delay.
    /// let tone = sine_wave::<f64>(100.0, Duration::from_millis(200), sample_rate!(48000), 0.5);
    /// let design = IirFilterDesign::butterworth_lowpass(NonZeroUsize::new(4).unwrap(), 5000.0);
    /// let out = tone.filtfilt(&design).unwrap();
    /// for i in 0..tone.samples_per_channel().get() {
    ///     assert!((out[i] - tone[i]).abs() < 1e-3);
    /// }
    /// ```
    fn filtfilt_in_place(&mut self, design: &IirFilterDesign) -> AudioSampleResult<()>;

    /// Apply zero-phase (forward-backward) IIR filtering, returning a new copy.
    ///
    /// Non-mutating twin of [`filtfilt_in_place`](Self::filtfilt_in_place).
    fn filtfilt(&self, design: &IirFilterDesign) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.filtfilt_in_place(design)?;
        Ok(out)
    }

    /// Apply a Butterworth low-pass filter.
    ///
    /// Convenience wrapper that constructs an [`IirFilterDesign`] and
    /// delegates to [`Self::apply_iir_filter_in_place`]. Equivalent to
    /// `scipy.signal.sosfilt(scipy.signal.butter(order, cutoff, 'lowpass',
    /// fs=fs, output='sos'), x)`.
    ///
    /// # Arguments
    /// - `order` - Filter order (number of poles).
    /// - `cutoff_frequency` - -3 dB frequency in Hz; must be in (0, Nyquist).
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if `cutoff_frequency` is
    ///   outside the valid range, or if `order` exceeds the supported maximum.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioIirFiltering;
    /// use non_empty_slice::NonEmptyVec;
    /// use std::num::NonZeroUsize;
    ///
    /// // Order-2 Butterworth lowpass
    /// let samples = NonEmptyVec::new(vec![1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]).unwrap();
    /// let mut audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// assert!(audio.butterworth_lowpass_in_place(NonZeroUsize::new(2).unwrap(), 1000.0).is_ok());
    ///
    /// // High-order (8th-order) Butterworth lowpass
    /// let samples2 = NonEmptyVec::new(vec![1.0f32; 200]).unwrap();
    /// let mut audio2: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples2, sample_rate!(44100));
    /// assert!(audio2.butterworth_lowpass_in_place(NonZeroUsize::new(8).unwrap(), 2000.0).is_ok());
    /// ```
    fn butterworth_lowpass_in_place(
        &mut self,
        order: NonZeroUsize,
        cutoff_frequency: f64,
    ) -> AudioSampleResult<()>;

    /// Apply a Butterworth low-pass filter, returning a new copy.
    ///
    /// Non-mutating twin of [`butterworth_lowpass_in_place`](Self::butterworth_lowpass_in_place);
    /// leaves `self` unchanged.
    fn butterworth_lowpass(
        &self,
        order: NonZeroUsize,
        cutoff_frequency: f64,
    ) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.butterworth_lowpass_in_place(order, cutoff_frequency)?;
        Ok(out)
    }

    /// Apply a Butterworth high-pass filter.
    ///
    /// Convenience wrapper that constructs an [`IirFilterDesign`] and
    /// delegates to [`Self::apply_iir_filter_in_place`]. Equivalent to
    /// `scipy.signal.sosfilt(scipy.signal.butter(order, cutoff, 'highpass',
    /// fs=fs, output='sos'), x)`.
    ///
    /// # Arguments
    /// - `order` - Filter order (number of poles).
    /// - `cutoff_frequency` - -3 dB frequency in Hz; must be in (0, Nyquist).
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if `cutoff_frequency` is
    ///   outside the valid range, or if `order` exceeds the supported maximum.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioIirFiltering;
    /// use non_empty_slice::NonEmptyVec;
    /// use std::num::NonZeroUsize;
    ///
    /// // Order-2 Butterworth highpass
    /// let samples = NonEmptyVec::new(vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
    /// let mut audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// assert!(audio.butterworth_highpass_in_place(NonZeroUsize::new(2).unwrap(), 500.0).is_ok());
    ///
    /// // High-order (6th-order) Butterworth highpass
    /// let samples2 = NonEmptyVec::new(vec![1.0f32; 200]).unwrap();
    /// let mut audio2: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples2, sample_rate!(44100));
    /// assert!(audio2.butterworth_highpass_in_place(NonZeroUsize::new(6).unwrap(), 1500.0).is_ok());
    /// ```
    fn butterworth_highpass_in_place(
        &mut self,
        order: NonZeroUsize,
        cutoff_frequency: f64,
    ) -> AudioSampleResult<()>;

    /// Apply a Butterworth high-pass filter, returning a new copy.
    ///
    /// Non-mutating twin of [`butterworth_highpass_in_place`](Self::butterworth_highpass_in_place);
    /// leaves `self` unchanged.
    fn butterworth_highpass(
        &self,
        order: NonZeroUsize,
        cutoff_frequency: f64,
    ) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.butterworth_highpass_in_place(order, cutoff_frequency)?;
        Ok(out)
    }

    /// Apply a Butterworth band-pass filter.
    ///
    /// Convenience wrapper that constructs an [`IirFilterDesign`] and
    /// delegates to [`Self::apply_iir_filter_in_place`]. The band-pass is
    /// derived from an `order`-pole low-pass prototype by the analog
    /// low-pass→band-pass transform, so the digital filter has `2 × order`
    /// poles. Equivalent to `scipy.signal.butter(order, [low, high],
    /// 'bandpass', fs=fs, output='sos')`.
    ///
    /// # Arguments
    /// - `order` - Order of the low-pass prototype.
    /// - `low_frequency` - Lower -3 dB edge in Hz; must be > 0.
    /// - `high_frequency` - Upper -3 dB edge in Hz; must be < Nyquist
    ///   and > `low_frequency`.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if the frequency range is
    ///   invalid, or if `order` exceeds the supported maximum.
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
    /// assert!(audio.butterworth_bandpass_in_place(NonZeroUsize::new(2).unwrap(), 100.0, 5000.0).is_ok());
    ///
    /// let samples2 = NonEmptyVec::new(vec![1.0f32; 200]).unwrap();
    /// let mut audio2: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples2, sample_rate!(44100));
    /// assert!(audio2.butterworth_bandpass_in_place(NonZeroUsize::new(4).unwrap(), 800.0, 1200.0).is_ok());
    /// ```
    fn butterworth_bandpass_in_place(
        &mut self,
        order: NonZeroUsize,
        low_frequency: f64,
        high_frequency: f64,
    ) -> AudioSampleResult<()>;

    /// Apply a Butterworth band-pass filter, returning a new copy.
    ///
    /// Non-mutating twin of [`butterworth_bandpass_in_place`](Self::butterworth_bandpass_in_place);
    /// leaves `self` unchanged.
    fn butterworth_bandpass(
        &self,
        order: NonZeroUsize,
        low_frequency: f64,
        high_frequency: f64,
    ) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.butterworth_bandpass_in_place(order, low_frequency, high_frequency)?;
        Ok(out)
    }

    /// Apply a Chebyshev Type I low-pass or high-pass filter.
    ///
    /// Chebyshev Type I filters offer sharper roll-off than Butterworth
    /// filters of the same order at the cost of passband ripple. The passband
    /// ripples between 0 dB and `-passband_ripple` dB; at
    /// `cutoff_frequency` the response is exactly `-passband_ripple` dB.
    /// Equivalent to `scipy.signal.cheby1(order, passband_ripple, cutoff,
    /// response, fs=fs, output='sos')`.
    ///
    /// For band-pass / band-stop use [`IirFilterDesign::chebyshev_i_band`]
    /// with [`Self::apply_iir_filter_in_place`].
    ///
    /// # Arguments
    /// - `order` - Filter order (number of poles).
    /// - `cutoff_frequency` - Passband edge in Hz; must be in (0, Nyquist).
    /// - `passband_ripple` - Maximum passband ripple in dB; must be positive.
    ///   Typical values: 0.1-1 dB.
    /// - `response` - [`FilterResponse::LowPass`] or [`FilterResponse::HighPass`].
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if `cutoff_frequency` is out of
    ///   range, `passband_ripple` is not positive, the order exceeds the
    ///   supported maximum, or `response` is a band response (which needs two
    ///   edge frequencies).
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioIirFiltering;
    /// use audio_samples::operations::types::FilterResponse;
    /// use non_empty_slice::NonEmptyVec;
    /// use std::num::NonZeroUsize;
    ///
    /// // 4th-order Chebyshev Type I lowpass with 0.5 dB ripple
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0]).unwrap();
    /// let mut audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// assert!(audio.chebyshev_i_in_place(
    ///     NonZeroUsize::new(4).unwrap(),
    ///     1000.0,
    ///     0.5,
    ///     FilterResponse::LowPass,
    /// ).is_ok());
    ///
    /// // 6th-order Chebyshev Type I highpass with 1.0 dB ripple
    /// let samples2 = NonEmptyVec::new(vec![1.0f32; 200]).unwrap();
    /// let mut audio2: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples2, sample_rate!(44100));
    /// assert!(audio2.chebyshev_i_in_place(
    ///     NonZeroUsize::new(6).unwrap(),
    ///     2000.0,
    ///     1.0,
    ///     FilterResponse::HighPass,
    /// ).is_ok());
    /// ```
    fn chebyshev_i_in_place(
        &mut self,
        order: NonZeroUsize,
        cutoff_frequency: f64,
        passband_ripple: f64,
        response: FilterResponse,
    ) -> AudioSampleResult<()>;

    /// Apply a Chebyshev Type I filter, returning a new copy.
    ///
    /// Non-mutating twin of [`chebyshev_i_in_place`](Self::chebyshev_i_in_place);
    /// leaves `self` unchanged.
    fn chebyshev_i(
        &self,
        order: NonZeroUsize,
        cutoff_frequency: f64,
        passband_ripple: f64,
        response: FilterResponse,
    ) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.chebyshev_i_in_place(order, cutoff_frequency, passband_ripple, response)?;
        Ok(out)
    }

    /// Return the frequency response at the specified frequencies.
    ///
    /// Deprecated placeholder: audio samples do not carry a filter, so this
    /// always returns a flat magnitude-1, phase-0 response. Use
    /// [`IirFilterDesign::frequency_response`] or
    /// [`SosFilter::frequency_response`](crate::SosFilter::frequency_response) instead.
    ///
    /// # Arguments
    /// - `frequencies` - Frequencies in Hz at which to compute the response.
    ///
    /// # Returns
    /// `Ok((magnitudes, phases))` where both vectors have the same length as
    /// `frequencies`; `magnitudes` is all 1.0 and `phases` is all 0.0.
    #[deprecated(
        since = "2.0.2",
        note = "always returns a flat response; use IirFilterDesign::frequency_response or SosFilter::frequency_response"
    )]
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
/// Use this trait to shape the tonal balance of audio - for example to boost
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
    /// - `eq` - The parametric EQ configuration. All enabled bands are validated before
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
    /// audio.apply_parametric_eq_in_place(&eq).unwrap();
    /// ```
    fn apply_parametric_eq_in_place(&mut self, eq: &ParametricEq) -> AudioSampleResult<()>;

    /// Applies a multi-band parametric EQ, returning a new copy.
    ///
    /// Non-mutating twin of [`apply_parametric_eq_in_place`](Self::apply_parametric_eq_in_place);
    /// leaves `self` unchanged.
    fn apply_parametric_eq(&self, eq: &ParametricEq) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.apply_parametric_eq_in_place(eq)?;
        Ok(out)
    }

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
    /// - `band` - The EQ band to apply, including its type, frequency, gain, and Q factor.
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
    /// audio.apply_eq_band_in_place(&EqBand::peak(2000.0, 4.0, 1.5)).unwrap();
    /// ```
    fn apply_eq_band_in_place(&mut self, band: &EqBand) -> AudioSampleResult<()>;

    /// Applies a single EQ band filter, returning a new copy.
    ///
    /// Non-mutating twin of [`apply_eq_band_in_place`](Self::apply_eq_band_in_place);
    /// leaves `self` unchanged.
    fn apply_eq_band(&self, band: &EqBand) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.apply_eq_band_in_place(band)?;
        Ok(out)
    }

    /// Applies a peak (bell) filter at the specified centre frequency.
    ///
    /// A peak filter boosts or cuts a band of frequencies centred around `frequency`
    /// (RBJ cookbook peaking EQ; the gain at `frequency` is exactly `gain_db`).
    /// Positive `gain_db` creates a boost (peak); negative creates a cut (dip). For a
    /// true notch (a zero of transmission) use an [`EqBand::band_stop`] band. The
    /// Q factor controls bandwidth: higher Q values produce a narrower effect.
    ///
    /// # Arguments
    ///
    /// - `frequency` - Centre frequency in Hz. Must be in `(0, Nyquist)`.
    /// - `gain_db` - Gain in dB. Positive boosts, negative cuts.
    /// - `q_factor` - Quality factor controlling bandwidth. Must be > 0.
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
    /// audio.apply_peak_filter_in_place(880.0, 6.0, 2.0).unwrap();
    /// ```
    fn apply_peak_filter_in_place(
        &mut self,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
    ) -> AudioSampleResult<()>;

    /// Applies a peak (bell) filter, returning a new copy.
    ///
    /// Non-mutating twin of [`apply_peak_filter_in_place`](Self::apply_peak_filter_in_place);
    /// leaves `self` unchanged.
    fn apply_peak_filter(
        &self,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
    ) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.apply_peak_filter_in_place(frequency, gain_db, q_factor)?;
        Ok(out)
    }

    /// Applies a low shelf filter that boosts or cuts frequencies below `frequency`.
    ///
    /// Frequencies well below the corner frequency receive `gain_db` of boost or
    /// cut, with a transition region controlled by `q_factor`; at the corner
    /// frequency itself the gain is `gain_db / 2` (RBJ cookbook shelf). The shelf
    /// levels off smoothly as frequency approaches zero.
    ///
    /// # Arguments
    ///
    /// - `frequency` - Corner (half-gain) frequency in Hz. Must be in `(0, Nyquist)`.
    /// - `gain_db` - Shelf gain in dB. Positive boosts, negative cuts.
    /// - `q_factor` - Shelf slope control. `0.707` (shelf slope `S = 1`) gives the
    ///   steepest shelf without overshoot. Must be > 0.
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
    /// audio.apply_low_shelf_in_place(200.0, -3.0, 0.707).unwrap();
    /// ```
    fn apply_low_shelf_in_place(
        &mut self,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
    ) -> AudioSampleResult<()>;

    /// Applies a low shelf filter, returning a new copy.
    ///
    /// Non-mutating twin of [`apply_low_shelf_in_place`](Self::apply_low_shelf_in_place);
    /// leaves `self` unchanged.
    fn apply_low_shelf(
        &self,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
    ) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.apply_low_shelf_in_place(frequency, gain_db, q_factor)?;
        Ok(out)
    }

    /// Applies a high shelf filter that boosts or cuts frequencies above `frequency`.
    ///
    /// Frequencies well above the corner frequency receive `gain_db` of boost or
    /// cut, with a transition region controlled by `q_factor`; at the corner
    /// frequency itself the gain is `gain_db / 2` (RBJ cookbook shelf). The shelf
    /// levels off smoothly as frequency approaches the Nyquist limit.
    ///
    /// # Arguments
    ///
    /// - `frequency` - Corner (half-gain) frequency in Hz. Must be in `(0, Nyquist)`.
    /// - `gain_db` - Shelf gain in dB. Positive boosts, negative cuts.
    /// - `q_factor` - Shelf slope control. `0.707` (shelf slope `S = 1`) gives the
    ///   steepest shelf without overshoot. Must be > 0.
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
    /// audio.apply_high_shelf_in_place(8000.0, 4.0, 0.707).unwrap();
    /// ```
    fn apply_high_shelf_in_place(
        &mut self,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
    ) -> AudioSampleResult<()>;

    /// Applies a high shelf filter, returning a new copy.
    ///
    /// Non-mutating twin of [`apply_high_shelf_in_place`](Self::apply_high_shelf_in_place);
    /// leaves `self` unchanged.
    fn apply_high_shelf(
        &self,
        frequency: f64,
        gain_db: f64,
        q_factor: f64,
    ) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.apply_high_shelf_in_place(frequency, gain_db, q_factor)?;
        Ok(out)
    }

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
    /// - `low_freq` - Low shelf corner frequency in Hz.
    /// - `low_gain` - Low shelf gain in dB.
    /// - `config` - Three-band EQ parameters (low/mid/high frequencies and gains,
    ///   plus the mid Q). Use [`ThreeBandEqConfig::new`] or the
    ///   [`ThreeBandEqConfig::flat`] preset.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if the configuration fails
    /// [`ThreeBandEqConfig::validate`] or if any band's frequency exceeds Nyquist.
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioParametricEq, sample_rate};
    /// use audio_samples::operations::types::ThreeBandEqConfig;
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_elem(512, 0.5f32);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Low shelf -2 dB at 200 Hz, mid peak +3 dB at 1 kHz (Q=2), high shelf +1 dB at 4 kHz
    /// let config = ThreeBandEqConfig::new(200.0, -2.0, 1000.0, 3.0, 2.0, 4000.0, 1.0);
    /// audio.apply_three_band_eq_in_place(&config).unwrap();
    /// ```
    fn apply_three_band_eq_in_place(&mut self, config: &ThreeBandEqConfig)
    -> AudioSampleResult<()>;

    /// Applies a three-band EQ, returning a new copy.
    ///
    /// Non-mutating twin of [`apply_three_band_eq_in_place`](Self::apply_three_band_eq_in_place);
    /// leaves `self` unchanged.
    fn apply_three_band_eq(&self, config: &ThreeBandEqConfig) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.apply_three_band_eq_in_place(config)?;
        Ok(out)
    }

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
    /// - `eq` - The parametric EQ whose frequency response to compute.
    /// - `frequencies` - Frequencies in Hz at which to evaluate the response. An empty
    ///   slice returns empty vectors.
    ///
    /// # Returns
    ///
    /// A tuple `(magnitudes, phases)` where:
    /// - `magnitudes` - Linear magnitude response at each frequency (1.0 = unity gain).
    /// - `phases` - Phase response in radians at each frequency.
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
/// Processors that control loudness variation: compression,
/// limiting, noise gating, and expansion.  All four processors share a
/// threshold-based, feed-forward design (envelope follower, static gain curve,
/// gain application); the compressor and limiter also support lookahead and
/// mono side-chain inputs.  The formulas are in the
/// [`dynamic_range`](crate::operations::dynamic_range) module documentation.
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
    /// The detector envelope is mapped through the static curve (hard knee or quadratic
    /// soft knee, threshold `T`, ratio `R`); levels above the threshold are reduced so
    /// that the output level is `T + (L − T)/R`. `attack_ms` and `release_ms` are
    /// one-pole time constants (63 % of a step after one time constant). With
    /// `lookahead_ms > 0` gain reduction starts up to `lookahead_ms` before a peak; the
    /// delay is compensated, so the output stays time-aligned with the input.
    /// Multi-channel audio is compressed independently per channel (no stereo linking).
    /// The formulas are in the [`dynamic_range`](crate::operations::dynamic_range)
    /// module documentation.
    ///
    /// # Arguments
    ///
    /// - `config` - Compressor parameters including threshold, ratio, attack, release,
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
    /// audio.apply_compressor_in_place(&config).unwrap();
    /// ```
    fn apply_compressor_in_place(&mut self, config: &CompressorConfig) -> AudioSampleResult<()>;

    /// Applies compression, returning a new copy.
    ///
    /// Non-mutating twin of [`apply_compressor_in_place`](Self::apply_compressor_in_place);
    /// leaves `self` unchanged.
    fn apply_compressor(&self, config: &CompressorConfig) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.apply_compressor_in_place(config)?;
        Ok(out)
    }

    /// Prevents the signal from exceeding a specified ceiling level.
    ///
    /// Applies gain reduction when the smoothed envelope level exceeds the ceiling (the
    /// compressor curve with an infinite ratio). The detector sees each sample up to
    /// `lookahead_ms` before it is output and the largest reduction over that window is
    /// applied, so gain reduction begins before peaks arrive; the delay is compensated.
    /// This is not a brick-wall limiter: the ceiling is exact only once the envelope has
    /// settled, and transients shorter than a few attack time constants can exceed it.
    /// Only sample peaks are detected (`isp_limiting` currently has no effect).
    /// Multi-channel audio is limited independently per channel. See the
    /// [`dynamic_range`](crate::operations::dynamic_range) module documentation.
    ///
    /// # Arguments
    ///
    /// - `config` - Limiter parameters including ceiling, attack, release, knee type,
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
    /// audio.apply_limiter_in_place(&config).unwrap();
    /// ```
    fn apply_limiter_in_place(&mut self, config: &LimiterConfig) -> AudioSampleResult<()>;

    /// Prevents the signal from exceeding a ceiling, returning a new copy.
    ///
    /// Non-mutating twin of [`apply_limiter_in_place`](Self::apply_limiter_in_place);
    /// leaves `self` unchanged.
    fn apply_limiter(&self, config: &LimiterConfig) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.apply_limiter_in_place(config)?;
        Ok(out)
    }

    /// Applies compression driven by an external sidechain signal.
    ///
    /// The gain reduction is determined by the level of `sidechain_signal` rather than
    /// the main audio, allowing one signal to control the dynamics of another. A common
    /// use case is ducking: a voice track causes background music to decrease in level
    /// whenever speech is present. Lookahead and latency compensation work as in
    /// [`apply_compressor_in_place`](Self::apply_compressor_in_place). The filter
    /// settings in `config.side_chain` are not applied: the side-chain signal drives the
    /// detector unfiltered.
    ///
    /// Only mono-to-mono sidechain processing is currently supported. Multi-channel
    /// combinations return an error.
    ///
    /// # Arguments
    ///
    /// - `config` - Compressor configuration. Sidechain processing must be enabled;
    ///   call `config.side_chain.enable()` before passing the config to this method.
    /// - `sidechain_signal` - External control signal. Must be the same length as the
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
    /// audio.apply_compressor_sidechain_in_place(&config, &sidechain).unwrap();
    /// ```
    fn apply_compressor_sidechain_in_place(
        &mut self,
        config: &CompressorConfig,
        sidechain_signal: &Self,
    ) -> AudioSampleResult<()>;

    /// Applies sidechain compression, returning a new copy.
    ///
    /// Non-mutating twin of
    /// [`apply_compressor_sidechain_in_place`](Self::apply_compressor_sidechain_in_place);
    /// leaves `self` unchanged.
    fn apply_compressor_sidechain(
        &self,
        config: &CompressorConfig,
        sidechain_signal: &Self,
    ) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.apply_compressor_sidechain_in_place(config, sidechain_signal)?;
        Ok(out)
    }

    /// Applies limiting driven by an external sidechain signal.
    ///
    /// The gain reduction ceiling is enforced based on the level of `sidechain_signal`
    /// rather than the main audio. Useful for frequency-selective limiting where a
    /// filtered copy of the signal (filtered by the caller) controls gain reduction.
    /// Lookahead works as in [`apply_limiter_in_place`](Self::apply_limiter_in_place).
    /// The filter settings in `config.side_chain` are not applied.
    ///
    /// Only mono-to-mono sidechain processing is currently supported. Multi-channel
    /// combinations return an error.
    ///
    /// # Arguments
    ///
    /// - `config` - Limiter configuration. Sidechain processing must be enabled;
    ///   call `config.side_chain.enable()` before passing the config to this method.
    /// - `sidechain_signal` - External control signal. Must be the same length as the
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
    /// audio.apply_limiter_sidechain_in_place(&config, &sidechain).unwrap();
    /// ```
    fn apply_limiter_sidechain_in_place(
        &mut self,
        config: &LimiterConfig,
        sidechain_signal: &Self,
    ) -> AudioSampleResult<()>;

    /// Applies sidechain limiting, returning a new copy.
    ///
    /// Non-mutating twin of
    /// [`apply_limiter_sidechain_in_place`](Self::apply_limiter_sidechain_in_place);
    /// leaves `self` unchanged.
    fn apply_limiter_sidechain(
        &self,
        config: &LimiterConfig,
        sidechain_signal: &Self,
    ) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.apply_limiter_sidechain_in_place(config, sidechain_signal)?;
        Ok(out)
    }

    /// Computes the static compression input-output curve for given input levels.
    ///
    /// Maps each input level in `input_levels_db` through the compressor's static gain
    /// characteristic (threshold, ratio, knee) plus makeup gain, returning the resulting
    /// output level in dBFS for each input. Does not use time-varying envelope following -
    /// the result depends only on the static transfer function.
    ///
    /// Useful for visualising and verifying compressor behaviour without processing actual
    /// audio samples.
    ///
    /// # Arguments
    ///
    /// - `config` - Compressor configuration parameters.
    /// - `input_levels_db` - Non-empty slice of input levels in dBFS to evaluate.
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
    /// The values come from the detector and static curve only:
    /// `config.lookahead_ms` is ignored here, so they show the gain reduction
    /// without the lookahead max-hold that `apply_compressor` applies.
    ///
    /// # Arguments
    ///
    /// - `config` - Compressor configuration parameters.
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
    /// A gate mutes or reduces quiet passages - typically background noise, room tone,
    /// or bleed - while leaving louder content unaffected. When the envelope level `L`
    /// is below the threshold `T`, the gain is reduced by `(R − 1)·(T − L)` dB, a 1:R
    /// downward expansion; above the threshold the signal passes unchanged. Peak
    /// detection is always used; there is no hold time, range floor or lookahead.
    ///
    /// # Arguments
    ///
    /// - `config` - Gate parameters (threshold, ratio, attack, release). Use
    ///   [`GateConfig::new`] or the [`GateConfig::noise_gate`] preset.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if the configuration fails
    /// [`GateConfig::validate`] (e.g. ratio ≤ 0).
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::GateConfig;
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_vec(vec![0.001f32, 0.8, 0.002, 0.9, 0.001]);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Gate at -20 dBFS, ratio 10: 1 dB under the threshold becomes 10 dB under
    /// audio.apply_gate_in_place(&GateConfig::with_params(-20.0, 10.0, 1.0, 10.0)).unwrap();
    /// ```
    fn apply_gate_in_place(&mut self, config: &GateConfig) -> AudioSampleResult<()>;

    /// Attenuates the signal below a threshold (noise gate), returning a new copy.
    ///
    /// Non-mutating twin of [`apply_gate_in_place`](Self::apply_gate_in_place);
    /// leaves `self` unchanged.
    fn apply_gate(&self, config: &GateConfig) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.apply_gate_in_place(config)?;
        Ok(out)
    }

    /// Increases dynamic range by expanding signals below a threshold.
    ///
    /// An expander is the complement of compression: signals below `threshold_db` are
    /// attenuated by `(R − 1)·(T − L)` dB, where `L` is the envelope level, `T` the
    /// threshold and `R` the ratio, making quiet passages quieter while leaving loud
    /// passages unchanged. RMS detection (10 ms window) is always used for expansion.
    ///
    /// # Arguments
    ///
    /// - `config` - Expander parameters (threshold, ratio, attack, release). Use
    ///   [`ExpanderConfig::new`] or the [`ExpanderConfig::gentle`] preset.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success. The audio is modified in place.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if the configuration fails
    /// [`ExpanderConfig::validate`] (e.g. ratio ≤ 0).
    ///
    /// # Example
    ///
    /// ```
    /// use audio_samples::{AudioSamples, AudioDynamicRange, sample_rate};
    /// use audio_samples::operations::types::ExpanderConfig;
    /// use ndarray::Array1;
    ///
    /// let data = Array1::from_vec(vec![0.1f32, 0.8, 0.2, 0.9, 0.1]);
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// // Expand at -20 dBFS with 2:1 ratio
    /// audio.apply_expander_in_place(&ExpanderConfig::with_params(-20.0, 2.0, 1.0, 10.0)).unwrap();
    /// ```
    fn apply_expander_in_place(&mut self, config: &ExpanderConfig) -> AudioSampleResult<()>;

    /// Increases dynamic range by expanding below a threshold, returning a new copy.
    ///
    /// Non-mutating twin of [`apply_expander_in_place`](Self::apply_expander_in_place);
    /// leaves `self` unchanged.
    fn apply_expander(&self, config: &ExpanderConfig) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.apply_expander_in_place(config)?;
        Ok(out)
    }
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
/// Use this trait to construct or reshape audio segments - for example to
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
    /// - `start_seconds` - start of the segment in seconds (`>= 0`).
    /// - `end_seconds` - end of the segment in seconds
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

    /// Extracts a time-bounded segment in place, replacing `self`.
    ///
    /// In-place twin of [`trim`](Self::trim). See that method for argument
    /// semantics and error conditions.
    ///
    /// # Errors
    /// See [`trim`](Self::trim).
    fn trim_in_place(&mut self, start_seconds: f64, end_seconds: f64) -> AudioSampleResult<()>;

    /// Adds silence (or a constant value) at the beginning and/or end.
    ///
    /// Works on both mono and multi-channel audio.
    ///
    /// # Arguments
    /// - `pad_start_seconds` - duration of padding prepended, in seconds.
    ///   Must be `>= 0`.
    /// - `pad_end_seconds` - duration of padding appended, in seconds.
    ///   Must be `>= 0`.
    /// - `pad_value` - the sample value used for the padded region
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

    /// Adds padding in place, replacing `self`.
    ///
    /// In-place twin of [`pad`](Self::pad). See that method for argument
    /// semantics and error conditions.
    ///
    /// # Errors
    /// See [`pad`](Self::pad).
    fn pad_in_place(
        &mut self,
        pad_start_seconds: f64,
        pad_end_seconds: f64,
        pad_value: Self::Sample,
    ) -> AudioSampleResult<()>;

    /// Pads with a constant value on the right to reach a target sample count.
    ///
    /// If the signal already has `target_num_samples` or more samples per
    /// channel it is returned unchanged (as an owned clone).
    ///
    /// # Arguments
    /// - `target_num_samples` - desired number of samples per channel.
    /// - `pad_value` - the sample value used for the padded region.
    ///
    /// # Returns
    /// A new [`AudioSamples`] with exactly `max(target_num_samples, current)`
    /// samples per channel.
    ///
    /// # Errors
    /// None in practice (the result is never empty); the `Result` is kept for
    /// API consistency.
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

    /// Pads on the right to a target sample count in place, replacing `self`.
    ///
    /// In-place twin of [`pad_samples_right`](Self::pad_samples_right). See that
    /// method for argument semantics and error conditions.
    ///
    /// # Errors
    /// See [`pad_samples_right`](Self::pad_samples_right).
    fn pad_samples_right_in_place(
        &mut self,
        target_num_samples: usize,
        pad_value: Self::Sample,
    ) -> AudioSampleResult<()>;

    /// Pads the signal to reach a target duration.
    ///
    /// Padding is added on the side specified by `pad_side`.  If the
    /// signal is already at least as long as `target_duration_seconds`
    /// it is returned unchanged (as an owned clone).
    ///
    /// # Arguments
    /// - `target_duration_seconds` - desired length in seconds.
    /// - `pad_value` - the sample value used for the padded region.
    /// - `pad_side` - which end to pad ([`PadSide::Left`] or
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

    /// Pads to a target duration in place, replacing `self`.
    ///
    /// In-place twin of [`pad_to_duration`](Self::pad_to_duration). See that
    /// method for argument semantics and error conditions.
    ///
    /// # Errors
    /// See [`pad_to_duration`](Self::pad_to_duration).
    fn pad_to_duration_in_place(
        &mut self,
        target_duration_seconds: f64,
        pad_value: Self::Sample,
        pad_side: PadSide,
    ) -> AudioSampleResult<()>;

    /// Splits the signal into segments of a fixed duration.
    ///
    /// The last segment may be shorter than `segment_duration_seconds`
    /// if the signal does not divide evenly.  Works on both mono and
    /// multi-channel audio.
    ///
    /// # Arguments
    /// - `segment_duration_seconds` - target length of each segment in
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
    /// - `segments` - the segments to join, in order.
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
    /// - `segments` - the owned segments to join, in order.
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
    /// Computes `y[n] = Σ_k w_k · x_k[n]`. All sources must have the same
    /// sample rate, channel count, and length.  When `weights` is `None`,
    /// equal weighting (1 / N per source) is applied. The sum is not
    /// clipped for float sample types; for integer sample types it is
    /// computed in `f64` on the normalised `[-1, 1]` scale, then rounded and
    /// saturated.
    ///
    /// # Arguments
    /// - `sources` - the audio signals to mix.
    /// - `weights` - optional per-source gain factors.  Must have the
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
    /// The first `n = ⌊duration_seconds · sample_rate⌋` samples of
    /// every channel are multiplied by `g(p)`, where `g` is the curve's gain
    /// function (see [`FadeCurve`]) and `p = i/n` for sample `i = 0, …, n−1`.
    /// The ramp therefore starts at gain 0 and its last sample has gain
    /// `g((n−1)/n)`, one step short of 1; samples after the fade are
    /// unchanged. If `duration_seconds` exceeds the signal length the entire
    /// signal is faded. Integer sample types get fractional gains (applied
    /// on the normalised scale, then rounded).
    ///
    /// # Arguments
    /// - `duration_seconds` - fade length in seconds.  Must be `> 0`.
    /// - `curve` - the envelope shape (see [`FadeCurve`]).
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
    /// audio.fade_in_in_place(0.5, FadeCurve::Linear).unwrap();
    /// // First sample is at position 0 → gain 0
    /// assert_eq!(audio.as_slice().unwrap()[0], 0.0);
    /// ```
    fn fade_in_in_place(
        &mut self,
        duration_seconds: f64,
        curve: FadeCurve,
    ) -> AudioSampleResult<()>;

    /// Applies a fade-in envelope, returning a new faded copy and leaving
    /// `self` unchanged.
    ///
    /// Non-mutating twin of [`fade_in_in_place`](Self::fade_in_in_place).
    ///
    /// # Arguments
    /// - `duration_seconds` - fade length in seconds.  Must be `> 0`.
    /// - `curve` - the envelope shape (see [`FadeCurve`]).
    ///
    /// # Returns
    /// The faded audio samples.
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
    /// let audio = AudioSamples::new_mono(
    ///     Array1::<f32>::ones(100), sample_rate!(100),
    /// ).unwrap()
    ///     .fade_in(0.5, FadeCurve::Linear)
    ///     .unwrap();
    /// // First sample is at position 0 → gain 0
    /// assert_eq!(audio.as_slice().unwrap()[0], 0.0);
    /// ```
    #[inline]
    fn fade_in(&self, duration_seconds: f64, curve: FadeCurve) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.fade_in_in_place(duration_seconds, curve)?;
        Ok(out)
    }

    /// Applies a fade-out envelope to the end of the signal.
    ///
    /// The last `n = ⌊duration_seconds · sample_rate⌋` samples of
    /// every channel are multiplied by `g(p)`, where `g` is the curve's gain
    /// function (see [`FadeCurve`]) and `p = 1 − i/n` for the `i`-th faded
    /// sample (`i = 0, …, n−1`). The ramp starts at gain 1 and its final
    /// sample has gain `g(1/n)`, one step before silence. If
    /// `duration_seconds` exceeds the signal length the entire signal is
    /// faded. Integer sample types get fractional gains (applied on the
    /// normalised scale, then rounded).
    ///
    /// # Arguments
    /// - `duration_seconds` - fade length in seconds.  Must be `> 0`.
    /// - `curve` - the envelope shape (see [`FadeCurve`]).
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
    /// audio.fade_out_in_place(0.5, FadeCurve::Linear).unwrap();
    /// // Last sample has position p = 1/n = 1/50 → linear gain 0.02
    /// let last = *audio.as_slice().unwrap().last().unwrap();
    /// assert!(last < 0.1);
    /// ```
    fn fade_out_in_place(
        &mut self,
        duration_seconds: f64,
        curve: FadeCurve,
    ) -> AudioSampleResult<()>;

    /// Applies a fade-out envelope, returning a new faded copy and leaving
    /// `self` unchanged.
    ///
    /// Non-mutating twin of [`fade_out_in_place`](Self::fade_out_in_place).
    ///
    /// # Arguments
    /// - `duration_seconds` - fade length in seconds.  Must be `> 0`.
    /// - `curve` - the envelope shape (see [`FadeCurve`]).
    ///
    /// # Returns
    /// The faded audio samples.
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
    /// let audio = AudioSamples::new_mono(
    ///     Array1::<f32>::ones(100), sample_rate!(100),
    /// ).unwrap()
    ///     .fade_out(0.5, FadeCurve::Linear)
    ///     .unwrap();
    /// // Last sample has position p = 1/n = 1/50 → linear gain 0.02
    /// let last = *audio.as_slice().unwrap().last().unwrap();
    /// assert!(last < 0.1);
    /// ```
    #[inline]
    fn fade_out(&self, duration_seconds: f64, curve: FadeCurve) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.fade_out_in_place(duration_seconds, curve)?;
        Ok(out)
    }

    /// Tiles the signal, repeating it end-to-end.
    ///
    /// Works on both mono and multi-channel audio.  A count of 1
    /// returns an owned clone of the original.
    ///
    /// # Arguments
    /// - `count` - number of repetitions.  Must be `>= 1`.
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

    /// Tiles the signal in place, replacing `self`.
    ///
    /// In-place twin of [`repeat`](Self::repeat). See that method for argument
    /// semantics and error conditions.
    ///
    /// # Errors
    /// See [`repeat`](Self::repeat).
    fn repeat_in_place(&mut self, count: usize) -> AudioSampleResult<()>;

    /// Removes leading and trailing silence from the signal.
    ///
    /// A sample (or, for multi-channel audio, an entire frame) is
    /// considered silent when its absolute value is at or below the
    /// linear equivalent of `threshold_db`.  If the entire signal is
    /// silent a zero-filled signal of the same length is returned.
    ///
    /// # Arguments
    /// - `threshold_db` - silence threshold in dB (typically negative,
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

    /// Removes leading and trailing silence in place, replacing `self`.
    ///
    /// In-place twin of [`trim_silence`](Self::trim_silence). See that method
    /// for argument semantics and error conditions.
    ///
    /// # Errors
    /// See [`trim_silence`](Self::trim_silence).
    fn trim_silence_in_place(&mut self, threshold_db: f64) -> AudioSampleResult<()>;

    /// Returns a perturbed copy of the signal for data augmentation.
    ///
    /// Delegates to [`perturb_in_place`] on an owned clone.  The original signal
    /// is not modified.  Available only when the `random-generation`
    /// feature is enabled.
    ///
    /// # Arguments
    /// - `config` - perturbation configuration (method, parameters, and
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
    /// - `config` - perturbation configuration (method, parameters, and
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
    /// - `sources` - mono audio signals of equal length and sample rate.
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
    /// or below `threshold_db`; for multi-channel audio, a frame is silent
    /// when every channel is).  Any silence run at least
    /// `min_silence_duration_seconds` long is excised, including leading and
    /// trailing runs; everything else, including shorter silent gaps such as
    /// the near-zero samples around a waveform's zero crossings, is kept and
    /// concatenated in order.
    ///
    /// # Arguments
    /// - `threshold_db` - silence threshold in dB (typically negative).
    /// - `min_silence_duration_seconds` - minimum length of a silence
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
    /// // loud(2) - silence(4) - loud(2) at 10 Hz
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

    /// Removes all qualifying silence regions in place, replacing `self`.
    ///
    /// In-place twin of [`trim_all_silence`](Self::trim_all_silence). See that
    /// method for argument semantics and error conditions.
    ///
    /// # Errors
    /// See [`trim_all_silence`](Self::trim_all_silence).
    fn trim_all_silence_in_place(
        &mut self,
        threshold_db: f64,
        min_silence_duration_seconds: f64,
    ) -> AudioSampleResult<()>;
}

/// Channel manipulation and layout conversion operations.
///
/// # Purpose
///
/// Provides the operations that change the channel structure of an
/// [`AudioSamples`] buffer: mono/stereo conversion, channel duplication,
/// per-channel extraction, panning, balance adjustment, interleaving, and
/// element-wise channel transforms.
///
/// # Intended Usage
///
/// Use this trait to prepare audio for a pipeline that has different channel
/// requirements - for example downmixing a recording to mono before pitch
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
    /// - `method` - The downmix strategy.  Available variants:
    ///   - `Average` - arithmetic mean of all channels (`(L + R)/2` for
    ///     stereo: identical channels keep their level, while a mono source
    ///     panned to the centre with the equal-power `to_stereo(Pan(0.0))`
    ///     comes back 3 dB quieter).
    ///   - `Left` - channel 0 only.
    ///   - `Right` - channel 1 only.
    ///   - `Weighted(weights)` - user-supplied per-channel weights.
    ///   - `Center` - surround center channel (≥ 6 ch), or average
    ///     of left and right for stereo.
    ///
    /// For integer sample types `Average` and `Center` average the raw
    /// sample values and `Weighted` sums on the normalised `[-1, 1]` scale,
    /// both in `f64`; the result is rounded to nearest and saturated at the
    /// type's range.
    ///
    /// # Returns
    /// An owned mono [`AudioSamples`] at the same sample rate.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if `method` is `Weighted`
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

    /// Convert this audio to mono in place, replacing `self`.
    ///
    /// In-place twin of [`to_mono`](Self::to_mono). See that method for the
    /// available downmix strategies and error conditions.
    ///
    /// # Errors
    /// See [`to_mono`](Self::to_mono).
    fn to_mono_in_place(&mut self, method: MonoConversionMethod) -> AudioSampleResult<()>;

    /// Convert audio to a different stereo.
    ///
    /// Behaviour depends on the chosen method:
    /// - `Duplicate` - copies mono audio to both channels; multi-channel
    ///   input is returned unchanged.
    /// - `Pan(value)` - applies equal-power panning to mono input.
    ///   The value is clamped to \[-1, 1\]: -1 is hard left, 0 is
    ///   centre, +1 is hard right.  Multi-channel input is returned
    ///   unchanged. For integer sample types the gains are applied on the
    ///   normalised `[-1, 1]` scale and the result is rounded and saturated.
    /// - `Left` - routes mono input to the left channel and fills the
    ///   right with silence (`[m, 0]`). For multi-channel input, extracts
    ///   channel 0 as a mono signal (as in 2.0; prefer
    ///   [`extract_channel`](AudioChannelOps::extract_channel) for that).
    /// - `Right` - routes mono input to the right channel and fills the
    ///   left with silence (`[0, m]`). For multi-channel input, extracts
    ///   channel 1 as a mono signal.
    ///
    /// # Arguments
    /// - `method` - The conversion strategy.
    ///
    /// # Returns
    /// An owned [`AudioSamples`] at the same sample rate: two channels for
    /// mono input; for multi-channel input, the input unchanged
    /// (`Duplicate`, `Pan`) or one extracted channel (`Left`, `Right`).
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Layout] - only if the internal stereo
    ///   buffer cannot be built (not expected for valid input).
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

    /// Convert this audio to stereo in place, replacing `self`.
    ///
    /// In-place twin of [`to_stereo`](Self::to_stereo). See that method for the
    /// available conversion strategies and error conditions.
    ///
    /// # Errors
    /// See [`to_stereo`](Self::to_stereo).
    fn to_stereo_in_place(&mut self, method: StereoConversionMethod) -> AudioSampleResult<()>;

    /// Duplicate audio into an n-channel signal.
    ///
    /// For mono input the single channel is replicated into all output
    /// channels.  For multi-channel input only channel 0 is used as the
    /// source; the remaining input channels are ignored.
    ///
    /// # Arguments
    /// - `n_channels` - Target channel count; must be ≥ 1.
    ///
    /// # Returns
    /// An owned [`AudioSamples`] with `n_channels` identical channels
    /// at the same sample rate.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if `n_channels` is 0.
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

    /// Duplicate audio into an n-channel signal in place, replacing `self`.
    ///
    /// In-place twin of [`duplicate_to_channels`](Self::duplicate_to_channels).
    /// See that method for argument semantics and error conditions.
    ///
    /// # Errors
    /// See [`duplicate_to_channels`](Self::duplicate_to_channels).
    fn duplicate_to_channels_in_place(&mut self, n_channels: usize) -> AudioSampleResult<()>;

    /// Extract a single channel as an owned mono signal.
    ///
    /// If the audio is already mono, a clone is returned (only
    /// `channel_index` 0 is valid in that case).
    ///
    /// # Arguments
    /// - `channel_index` - Zero-based index of the channel to extract;
    ///   must be less than the number of channels.
    ///
    /// # Returns
    /// An owned mono [`AudioSamples`] at the same sample rate.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if `channel_index` is ≥ the
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
        channel_index: usize,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>>;

    /// Borrow a single channel as a zero-copy view.
    ///
    /// The returned [`AudioSamples`] shares memory with `self`; its
    /// lifetime is tied to the borrow of `self`.
    ///
    /// # Arguments
    /// - `channel_index` - Zero-based index of the channel to borrow;
    ///   must be less than the number of channels.
    ///
    /// # Returns
    /// A borrowed mono [`AudioSamples`] at the same sample rate.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if `channel_index` is ≥ the
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
        channel_index: usize,
    ) -> AudioSampleResult<AudioSamples<'_, Self::Sample>>;

    /// Swap two channels in place.
    ///
    /// For mono audio the only valid index is 0, making the swap a
    /// no-op.  Passing any other index for mono audio will trigger
    /// the out-of-range error below.
    ///
    /// # Arguments
    /// - `channel1` - Zero-based index of the first channel.
    /// - `channel2` - Zero-based index of the second channel.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if either index is ≥ the
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
    /// assert!(stereo.swap_channels_in_place(0, 1).is_ok());
    /// ```
    fn swap_channels_in_place(&mut self, channel1: usize, channel2: usize)
    -> AudioSampleResult<()>;

    /// Swap two channels, returning a new copy and leaving `self` unchanged.
    ///
    /// Non-mutating twin of [`swap_channels_in_place`](Self::swap_channels_in_place).
    ///
    /// # Arguments
    /// - `channel1` - Zero-based index of the first channel.
    /// - `channel2` - Zero-based index of the second channel.
    ///
    /// # Returns
    /// The audio with the two channels swapped.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if either index is ≥ the
    ///   number of channels.
    #[inline]
    fn swap_channels(&self, channel1: usize, channel2: usize) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.swap_channels_in_place(channel1, channel2)?;
        Ok(out)
    }

    /// Apply a linear balance-with-boost to a stereo signal in place.
    ///
    /// The left channel is scaled by `1 - p` and the right channel by
    /// `1 + p`, where `p` is `pan_value` clamped to \[-1, 1\].  A value
    /// of 0 leaves both channels unchanged; -1 silences the right
    /// channel and doubles the left (+6 dB, which can clip); +1 silences
    /// the left channel and doubles the right.  The total power changes
    /// with `p` (it is not a constant-power pan law); for a −3 dB-centre
    /// equal-power pan of a mono source use
    /// `to_stereo(StereoConversionMethod::Pan(p))`.
    ///
    /// For integer sample types the gains are applied on the normalised
    /// `[-1, 1]` scale in `f64`, and the result is rounded to nearest and
    /// saturated at the type's range (for `u8` the gain scales the distance
    /// from the mid-scale value 128).
    ///
    /// # Arguments
    /// - `pan_value` - Panning position in the range \[-1, 1\].
    ///   Values outside this range are clamped.  -1 is hard left,
    ///   0 is centre, +1 is hard right.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if the audio is mono or
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
    /// stereo.pan_in_place(0.5).unwrap();
    /// ```
    fn pan_in_place(&mut self, pan_value: f64) -> AudioSampleResult<()>;

    /// Apply a linear balance-with-boost (see [`pan_in_place`](Self::pan_in_place)),
    /// returning a new copy and leaving `self` unchanged.
    ///
    /// Non-mutating twin of [`pan_in_place`](Self::pan_in_place).
    ///
    /// # Arguments
    /// - `pan_value` - Panning position in the range \[-1, 1\].
    ///   Values outside this range are clamped.
    ///
    /// # Returns
    /// The panned audio samples.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if the audio is mono or
    ///   has a channel count other than 2.
    #[inline]
    fn pan(&self, pan_value: f64) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.pan_in_place(pan_value)?;
        Ok(out)
    }

    /// Adjust the left/right balance of a stereo signal in place.
    ///
    /// The formula applied is identical to [`AudioChannelOps::pan`]:
    /// the left channel is scaled by `1 - balance` and the right
    /// channel by `1 + balance`, after clamping to \[-1, 1\].  The
    /// favoured channel is boosted by up to +6 dB (gain 2), so this is
    /// not a constant-power law.
    ///
    /// For integer sample types the gains are applied on the normalised
    /// `[-1, 1]` scale in `f64`, and the result is rounded to nearest and
    /// saturated at the type's range (for `u8` the gain scales the distance
    /// from the mid-scale value 128).
    ///
    /// # Arguments
    /// - `balance` - Balance position in the range \[-1, 1\].
    ///   Values outside this range are clamped.  -1 is full left,
    ///   0 is centre, +1 is full right.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if the audio is mono or
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
    /// stereo.balance_in_place(-0.3).unwrap();
    /// ```
    fn balance_in_place(&mut self, balance: f64) -> AudioSampleResult<()>;

    /// Adjust left/right balance, returning a new copy and leaving `self`
    /// unchanged.
    ///
    /// Non-mutating twin of [`balance_in_place`](Self::balance_in_place).
    ///
    /// # Arguments
    /// - `balance` - Balance position in the range \[-1, 1\].
    ///   Values outside this range are clamped.
    ///
    /// # Returns
    /// The balance-adjusted audio samples.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if the audio is mono or
    ///   has a channel count other than 2.
    #[inline]
    fn balance(&self, balance: f64) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
    {
        let mut out = self.clone();
        out.balance_in_place(balance)?;
        Ok(out)
    }

    /// Apply a closure to every sample in a single channel.
    ///
    /// For mono audio the closure is applied to the single channel
    /// and `channel_index` is ignored.  For multi-channel audio
    /// `channel_index` is validated and must be less than the number
    /// of channels.
    ///
    /// # Arguments
    /// - `channel_index` - Zero-based index of the target channel.
    ///   Ignored for mono audio.
    /// - `func` - A closure that maps each sample to a new value.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if the audio is
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
    /// stereo.apply_to_channel_in_place(1, |s| s * 0.5).unwrap();
    /// ```
    fn apply_to_channel_in_place<F>(
        &mut self,
        channel_index: usize,
        func: F,
    ) -> AudioSampleResult<()>
    where
        F: FnMut(Self::Sample) -> Self::Sample;

    /// Apply a closure to every sample in a single channel, returning a new
    /// copy and leaving `self` unchanged.
    ///
    /// Non-mutating twin of [`apply_to_channel_in_place`](Self::apply_to_channel_in_place).
    ///
    /// # Arguments
    /// - `channel_index` - Zero-based index of the target channel.
    ///   Ignored for mono audio.
    /// - `func` - A closure that maps each sample to a new value.
    ///
    /// # Returns
    /// The audio with `func` applied to the selected channel.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if the audio is
    ///   multi-channel and `channel_index` is ≥ the number of
    ///   channels.
    #[inline]
    fn apply_to_channel<F>(&self, channel_index: usize, func: F) -> AudioSampleResult<Self>
    where
        Self: Sized + Clone,
        F: FnMut(Self::Sample) -> Self::Sample,
    {
        let mut out = self.clone();
        out.apply_to_channel_in_place(channel_index, func)?;
        Ok(out)
    }

    /// Combine multiple mono signals into a single multi-channel signal.
    ///
    /// All input signals must have the same number of samples.  The
    /// first signal becomes channel 0, the second becomes channel 1,
    /// and so on.  The output sample rate is taken from the first
    /// input signal.
    ///
    /// # Arguments
    /// - `channels` - A non-empty slice of mono [`AudioSamples`].
    ///   All elements must share the same sample count.
    ///
    /// # Returns
    /// An owned multi-channel [`AudioSamples`] with one channel per
    /// input signal.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if the input signals do
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
/// decomposition techniques are used in music information retrieval,
/// audio analysis, and preprocessing for machine learning applications.
///
/// # Available Decomposition Methods
///
/// - **HPSS (Harmonic/Percussive Source Separation)**: Separates audio into
///   harmonic (tonal, sustained) and percussive (transient, attack) components
///   using STFT magnitude median filtering.
///
/// See [`AudioDecomposition::hpss`] for a runnable example and the exact masking rule.
///
/// Audio source separation using spectral decomposition techniques.
///
/// # Purpose
///
/// Separates an audio signal into its constituent perceptual components.
/// Currently provides Harmonic-Percussive Source Separation (HPSS), which
/// splits a signal into a tonal/sustained layer and a transient/percussive layer.
///
/// # Intended Usage
///
/// Use this trait when downstream processing needs to operate on only one
/// component type - for example to apply pitch correction only to the harmonic
/// layer, or to analyse rhythm independently of melody.
///
/// # Invariants
///
/// The harmonic and percussive masks sum to 1 (up to a 10⁻¹⁰ regulariser) in every
/// time-frequency bin where the two median-filtered magnitudes are not both zero, for
/// both hard and soft masking, so the two outputs sum to (almost exactly) the ISTFT
/// reconstruction of the input. Input must be mono; both outputs are mono and
/// carry the input's sample rate. Their length is the ISTFT output length, which can
/// differ slightly from the input length (see [`AudioTransforms::istft`]).
#[cfg(feature = "decomposition")]
pub trait AudioDecomposition: AudioTransforms
where
    Self::Sample: StandardSample,
{
    /// Separates audio into harmonic and percussive components (HPSS).
    ///
    /// Harmonic-Percussive Source Separation applies median filtering to the
    /// STFT magnitude spectrogram |X| along the time axis (length
    /// `median_filter_harmonic`) to get H, and along the frequency axis (length
    /// `median_filter_percussive`) to get P. With softness s = `mask_softness`, the
    /// harmonic mask is M_H = s·H/(H + P) + (1 − s)·\[H ≥ P\] and the percussive mask is
    /// M_P = s·P/(H + P) + (1 − s)·\[P > H\]. Both masks multiply the original complex
    /// STFT, and each masked STFT is inverted with the ISTFT.
    ///
    /// Note: librosa's `decompose.hpss` uses a soft mask on *powers* (H²/(H² + P²))
    /// and `scipy.ndimage.median_filter` with `reflect` edges; this implementation
    /// uses exponent 1 on magnitudes and mirror edges (d c b | a b c d), so results
    /// differ slightly from librosa.
    ///
    /// Reference:
    /// Fitzgerald, D. (2010). "Harmonic/percussive separation using median filtering".
    /// Müller, M. (2015). *Fundamentals of Music Processing*, Section 8.4.
    ///
    /// # Arguments
    ///
    /// - `config` - HPSS parameters: window size, hop size, harmonic and percussive
    ///   median-filter lengths, and mask softness (`HpssConfig::musical()` for a
    ///   good default).
    ///
    /// # Returns
    ///
    /// A `(harmonic, percussive)` tuple of mono signals at the input's sample rate.
    ///
    /// # Errors
    ///
    /// - [crate::AudioSampleError::Parameter] if `config` fields are invalid or the
    ///   signal is shorter than the specified STFT window.
    /// - [crate::AudioSampleError::Layout] if the input is multi-channel.
    /// - Errors propagated from the internal STFT or ISTFT.
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
/// For librosa-compatible onset strength / `onset_detect` see `AudioRhythm`
/// (`operations::rhythm`, feature `beat-tracking`).
///
/// # Purpose
///
/// Locates the moments where a new musical event begins - note attacks,
/// drum hits, chord changes - and exposes the intermediate spectral
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
    /// Detects onset times using the CQT "energy" onset detection function.
    ///
    /// Computes [`onset_detection_function`](AudioOnsetDetection::onset_detection_function)
    /// (half-wave-rectified CQT magnitude flux, optionally median-thresholded),
    /// applies peak picking with `config.peak_picking` (lengths in frames) and
    /// converts frame indices to seconds as `t·hop_size/f_s`. The default
    /// peak picking ignores peaks below 0.1 of the strongest one; see the
    /// [`onset`](crate::operations::onset) module docs for the known
    /// limitations.
    ///
    /// # Arguments
    ///
    /// - `config` - Onset detection parameters: CQT settings, hop size,
    ///   median-based ODF thresholding and peak-picking configuration.
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
    /// Returns the activation curve before peak-picking, together with the
    /// corresponding frame timestamps: `odf[t] = Σ_b max(0, |X[b,t]| − |X[b,t−1]|)`
    /// on the CQT magnitude (not squared), with values below
    /// `adaptive_threshold_multiplier × running median` set to zero when
    /// `config.adaptive_threshold` is set. Useful when you want to inspect
    /// the ODF, apply custom thresholding, or feed it into a beat tracker.
    ///
    /// # Arguments
    ///
    /// - `config` - Onset detection parameters controlling the spectral
    ///   analysis and flux computation.
    ///
    /// # Returns
    ///
    /// A `(timestamps, odf_values)` tuple:
    /// - `timestamps` - Frame centre times in seconds.
    /// - `odf_values` - One activation value per analysis frame.
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
    /// - `config` - Spectral flux parameters: CQT settings, flux method,
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
    /// - `config` - CQT analysis parameters (bins per octave, frequency range, etc.).
    /// - `window_size` - Analysis window length in samples.
    /// - `hop_size` - Number of samples to advance between successive windows.
    /// - `method` - Spectral flux variant to compute (see
    ///   [`SpectralFluxMethod`] for the exact formula of each variant; none
    ///   squares the magnitudes).
    ///
    /// # Returns
    ///
    /// A `(timestamps, flux_values)` tuple, both of equal length.
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
    /// Combines the CQT magnitude difference and the phase deviation from
    /// each bin's centre frequency into a single activation curve
    /// (`w_m·Σ R(Δ|X|) + w_p·Σ |phase deviation|`, optionally log-compressed),
    /// then applies peak picking (lengths in frames).
    ///
    /// # Arguments
    ///
    /// - `onset_config` - Complex onset detection parameters: CQT settings,
    ///   weights and peak-picking thresholds.
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
    /// Each value is `w_m·Σ_b R(|X[b,t]| − |X[b,t−1]|) + w_p·Σ_b dev[b,t]`, where
    /// R is rectification or absolute value and `dev` is the wrapped phase
    /// deviation (see [`phase_deviation_matrix`](AudioOnsetDetection::phase_deviation_matrix)),
    /// optionally passed through `ln(1 + C·x)`.
    ///
    /// # Arguments
    ///
    /// - `onset_config` - Complex onset detection parameters.
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
    /// Each entry `[b, t]` is the signed CQT magnitude difference
    /// `|X[b,t]| − |X[b,t−1]|` at frequency bin `b` and frame `t` (column 0 is
    /// zero). This is an intermediate building block of the complex ODF.
    ///
    /// # Arguments
    ///
    /// - `config` - Complex onset detection parameters controlling the CQT
    ///   analysis.
    ///
    /// # Returns
    ///
    /// An `Array2<f64>` with shape `[frequency_bins, frames]`.
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
    /// Each entry `[b, t]` is `|princarg(φ[b,t] − φ[b,t−1] − 2π·f_b·H/f_s)|`
    /// in radians: the deviation of the measured phase advance from that of a
    /// sinusoid at the bin centre frequency f_b (H = hop size). This is a
    /// first-order measure, not the second-difference phase deviation of
    /// Bello et al. (2005); a stationary sinusoid off the bin centre gives a
    /// constant non-zero value. Column 0 is zero.
    ///
    /// # Arguments
    ///
    /// - `config` - Complex onset detection parameters controlling the CQT
    ///   analysis.
    ///
    /// # Returns
    ///
    /// An `Array2<f64>` with shape `[frequency_bins, frames]`.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Processing] if the STFT computation fails.
    fn phase_deviation_matrix(&self, config: &ComplexOnsetConfig)
    -> AudioSampleResult<Array2<f64>>;
}

/// Beat tracking operations for audio signals.
///
/// For librosa-compatible tempo estimation and dynamic-programming beat tracking see
/// [`AudioRhythm`].
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
    /// peak in steps of one inter-beat interval (expected beat k is at
    /// `peak ± k·IBI` frames with the exact, fractional
    /// `IBI = 60·f_s/(tempo_bpm·hop_size)`, refined to the largest envelope
    /// value within ±`tolerance`). The grid is rigid, so `config.tempo_bpm`
    /// must be accurate: a tempo error of ΔB BPM shifts beat k by about
    /// `k·60·ΔB/B²` seconds.
    ///
    /// # Arguments
    /// - `config` - Beat tracking configuration: target tempo,
    ///   optional timing tolerance, and onset detection parameters.
    ///
    /// # Returns
    /// A [`BeatTrackingData`] containing the target tempo
    /// (`config.tempo_bpm`, echoed: the tempo is not estimated here) and the
    /// detected beat timestamps in seconds, sorted ascending.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if `config.tempo_bpm` is
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

    /// Estimate the dominant tempo of the signal in beats per minute.
    ///
    /// Computes the onset strength envelope (using the onset configuration in
    /// `config`), autocorrelates it, restricts the search to a plausible BPM lag
    /// range (40-240 BPM at the envelope frame rate), weights each lag by a
    /// log-normal prior centred on 120 BPM (σ = 1 octave), and picks the
    /// best-scoring integer lag L, converting it to BPM as
    /// `60·f_s/(hop_size·L)`. Results therefore lie on the integer-lag grid,
    /// and octave errors (×2, ×½, occasionally ×⅓) remain possible: the prior
    /// only biases the choice toward 120 BPM. Unlike
    /// [`detect_beats`][AudioBeatTracking::detect_beats], this does **not** require
    /// the tempo to be known in advance - `config.tempo_bpm` is ignored here.
    ///
    /// # Arguments
    /// - `config` - beat tracking configuration; only `onset_config` is used.
    ///
    /// # Returns
    /// The estimated tempo in BPM.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] - if no plausible tempo lag can be
    ///   found (e.g. the envelope is too short for the 40 BPM lag).
    /// - [crate::AudioSampleError::Processing] - if onset detection fails.
    fn estimate_tempo(&self, config: &BeatTrackingConfig) -> AudioSampleResult<f64>;
}

/// Audio visualisation operations.
///
/// # Purpose
///
/// Produces interactive plots of audio signals in three representations:
/// time-domain waveform, time-frequency spectrogram, and single-frame
/// magnitude spectrum.  All plots are rendered via the `plotly` crate.
///
/// # Intended Usage
///
/// Use this trait during exploratory analysis or for generating report
/// artefacts.  Each method returns an opaque plot object that can be
/// rendered to HTML or saved to a file - consult the `WaveformPlot`,
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
    /// - `params` - Visual parameters: title, axis labels, colour scheme,
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

    /// Renders a time-frequency spectrogram plot.
    ///
    /// Computes an STFT and displays the resulting magnitude (or log-magnitude)
    /// as a heat-map with time on the x-axis and frequency on the y-axis.
    ///
    /// # Arguments
    ///
    /// - `params` - Spectrogram and visual parameters: STFT window size, hop
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
    /// - `params` - Spectrum and visual parameters: FFT size, frequency range,
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

    /// Renders a phase spectrum plot for a single frame.
    ///
    /// Applies an FFT to the audio and displays the per-bin phase angle
    /// (`Complex::arg()`, in radians) on a frequency axis, optionally unwrapped.
    ///
    /// # Arguments
    ///
    /// - `params` - Phase-spectrum and visual parameters: FFT size, unwrapping,
    ///   frequency range, and title.
    ///
    /// # Returns
    ///
    /// A [`PhaseSpectrumPlot`] that can be rendered to HTML or saved to a file.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Processing] if the FFT computation fails,
    /// or [crate::AudioSampleError::Feature] if the `transforms` feature is disabled.
    fn plot_phase_spectrum(
        &self,
        params: &PhaseSpectrumParams,
    ) -> AudioSampleResult<PhaseSpectrumPlot>;

    /// Renders a Lissajous (stereo X-Y) plot.
    ///
    /// Scatters channel 0 (x) against channel 1 (y) for a 2-channel signal - the
    /// standard stereo correlation / phase visualization.
    ///
    /// # Arguments
    ///
    /// - `params` - Visual parameters and an optional point cap.
    ///
    /// # Returns
    ///
    /// A [`LissajousPlot`] that can be rendered to HTML or saved to a file.
    ///
    /// # Errors
    ///
    /// Returns [crate::AudioSampleError::Parameter] if the audio does not have
    /// exactly two channels.
    fn plot_lissajous(&self, params: &LissajousParams) -> AudioSampleResult<LissajousPlot>;
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
/// - `amplitude_envelope` - fastest, no smoothing.
/// - `rms_envelope` / `moving_average_envelope` - configurable smoothing via window.
/// - `analytic_envelope` - smoothest, requires a Hilbert transform.
/// - `attack_decay_envelope` - separates rising and falling energy.
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
    /// - `window_size` - Number of samples in each analysis window.
    /// - `hop_size` - Number of samples to advance between successive
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
    /// A copy of `follower` (reset to zero, detection switched to
    /// `method`) produces the envelope `e[n]` described in
    /// [`EnvelopeFollower`]. The two outputs sample-and-hold it:
    /// `attack[n] = e[n]` when `e[n] ≥ e[n−1]`, otherwise
    /// `attack[n] = attack[n−1]`; `decay[n] = e[n]` when
    /// `e[n] < e[n−1]`, otherwise `decay[n] = decay[n−1]` (both start
    /// at 0). Both output envelopes have the same length as the input.
    /// The time constants are converted with the sample rate given to
    /// [`EnvelopeFollower::new`], not the sample rate of `self`.
    /// Samples are used in their stored units (integer samples are not
    /// rescaled to `[-1, 1]`).
    ///
    /// # Arguments
    /// - `follower` - A configured [`EnvelopeFollower`] that supplies the
    ///   attack and release time constants.
    /// - `method` - Detection strategy used to estimate level: peak
    ///   (`|x|`), RMS over a 10 ms window, or hybrid (the larger of the
    ///   two). It overrides the method the follower was built with.
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
    /// - `window_size` - Number of samples per averaging window.
    /// - `hop_size` - Number of samples to advance between successive
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

#[cfg(feature = "transforms")]
pub use crate::operations::spectrum::AudioSpectrum;

#[cfg(all(feature = "transforms", feature = "resampling"))]
pub use crate::operations::constant_q::AudioConstantQ;
#[cfg(feature = "feature-extraction")]
pub use crate::operations::tonal::AudioTonal;

#[cfg(feature = "feature-extraction")]
pub use crate::operations::features::AudioFeatures;
#[cfg(feature = "pitch-analysis")]
pub use crate::operations::pitch_tracking::AudioPitchTracking;

#[cfg(feature = "effects")]
pub use crate::operations::effects::AudioEffects;

#[cfg(feature = "beat-tracking")]
pub use crate::operations::rhythm::AudioRhythm;
