//! Audio analysis and detection utilities.

//! Lightweight functions for inspecting structural and perceptual properties of audio
//! signals without building full spectrograms or running heavy DSP pipelines:
//!
//! - [`detect_silence_regions`] - time intervals where amplitude falls below a threshold
//! - [`detect_clipping`] - intervals where the signal reaches or exceeds the sample type's full-scale value
//! - [`detect_dynamic_range`] - peak amplitude, RMS amplitude and crest factor (peak-to-RMS ratio) in dB
//! - [`detect_fundamental_frequency`] - autocorrelation-based fundamental frequency estimate
//! - [`estimate_noise_floor`] - crude noise-floor proxy in dBFS: the 10th percentile of `|x|`
//! - [`detect_sample_rate`] *(requires `transforms` feature)* - spectral-cutoff heuristic for detecting original sample rate
//! - [`estimate_frequency_range`] *(requires `transforms` feature)* - lowest and highest frequency with significant energy
//!
//! Detecting silence, clipping, and noise floor are common pre-processing steps in audio
//! pipelines. Centralising them avoids duplication across higher-level operations and
//! ensures consistent threshold semantics throughout the library.
//!
//! All detection functions accept a reference to an [`AudioSamples`] buffer. For mono
//! signals the analysis runs directly on the single channel. For multi-channel signals,
//! channel-averaging or per-channel inspection is applied as described in the individual
//! function documentation.
//!
//! ```rust
//! use audio_samples::utils::detection::detect_silence_regions;
//! use audio_samples::utils::generation::silence;
//! use audio_samples::sample_rate;
//! use std::time::Duration;
//!
//! let audio = silence::<f32>(Duration::from_millis(200), sample_rate!(44100));
//! let regions = detect_silence_regions(&audio, 0.001f32).unwrap();
//! assert!(!regions.is_empty());
//! ```

#[cfg(feature = "transforms")]
use spectrograms::{WindowType, power_spectrum};
#[cfg(feature = "transforms")]
use std::num::{NonZeroU32, NonZeroUsize};

use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, LayoutError,
    ParameterError, traits::StandardSample,
};

use non_empty_slice::{NonEmptySlice, NonEmptyVec};

/// Attempts to detect the original sample rate of an audio signal from spectral content.
///
/// Analyses the Hann-windowed power spectrum of the first channel (FFT length = next power
/// of two ≥ signal length) for sharp high-frequency cutoffs that are characteristic of
/// anti-aliasing filters applied during resampling. When a cutoff is detected near a common
/// sample rate's Nyquist frequency, that rate is returned as the estimated original sample
/// rate (see [`analyze_spectrum_for_cutoff`] for the exact test).
///
/// This is a heuristic function and is not guaranteed to produce accurate results for all
/// inputs. It is most reliable when the signal was previously down-sampled from one of the
/// standard rates listed below.
///
/// Supported candidate rates (Hz): 8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200,
/// 96000.
///
/// # Arguments
///
/// - `audio` - The audio signal to analyse. Only the first channel is used for analysis.
///
/// # Returns
///
/// * `Ok(Some(rate))` - A detected candidate rate whose Nyquist matches a spectral cutoff.
/// * `Ok(None)` - No candidate rate could be identified.
///
/// # Errors
///
/// Returns an error if the underlying spectral computation fails or the sample layout is
/// non-contiguous. Finding no cutoff is not an error (`Ok(None)`).
///
/// # Feature flag
///
/// Requires `feature = "transforms"`.
#[cfg(feature = "transforms")]
#[inline]
pub fn detect_sample_rate<T>(audio: &AudioSamples<T>) -> AudioSampleResult<Option<NonZeroU32>>
where
    T: StandardSample,
{
    let audio_f64 = audio.as_f64();

    // Use the first channel for analysis
    let data = if let Some(mono) = audio_f64.as_mono() {
        mono.as_slice()
            .ok_or_else(|| {
                AudioSampleError::Layout(LayoutError::NonContiguous {
                    operation: "audio processing".to_string(),
                    layout_type: "non-contiguous mono array".to_string(),
                })
            })?
            .to_vec()
    } else {
        // Use the first channel of multi-channel audio
        let multi = audio_f64.as_multi_channel().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_data",
                "Audio must be multi-channel",
            ))
        })?;
        multi
            .row(0)
            .as_slice()
            .ok_or_else(|| {
                AudioSampleError::Layout(LayoutError::NonContiguous {
                    operation: "audio processing".to_string(),
                    layout_type: "non-contiguous multi-channel array".to_string(),
                })
            })?
            .to_vec()
    };

    // Look for high-frequency cutoff patterns that might indicate resampling
    // safety: data is non-empty
    let n_fft = unsafe { NonZeroUsize::new_unchecked(data.len().next_power_of_two()) };
    // safety: data is non-empty (channel was extracted from valid audio)
    let data_slice = unsafe { NonEmptySlice::new_unchecked(&data[..]) };
    let spectrum = power_spectrum(data_slice, n_fft, Some(WindowType::Hanning))
        .map_err(AudioSampleError::from)?;
    let nyquist_freq = audio.nyquist();
    let spectrum_slice = spectrum.as_non_empty_slice();
    // Analyze the spectrum for sharp cutoffs that might indicate resampling.
    // No cutoff found is a normal outcome, reported as `Ok(None)`.
    let Some(detected_rate) = analyze_spectrum_for_cutoff(spectrum_slice, nyquist_freq) else {
        return Ok(None);
    };

    let detected_rate = NonZeroU32::new(detected_rate).ok_or_else(|| {
        AudioSampleError::Parameter(ParameterError::invalid_value(
            "detected_rate",
            "detected sample rate is zero",
        ))
    })?;
    Ok(Some(detected_rate))
}

/// Estimates the fundamental frequency of an audio signal using autocorrelation.
///
/// Computes the normalised autocorrelation `ρ(τ) = r(τ)/r(0)` of the first channel
/// (with the unbiased estimate `r(τ) = (1/(N−τ)) Σ x[n]·x[n+τ]`) over the lags `τ` that
/// correspond to 50 Hz - 2 000 Hz, and returns `fs / τ*` where `τ*` is the **first** local
/// maximum of `ρ` whose height is at least 0.9 times the highest `ρ` in that range. Taking the
/// first such peak, rather than the global maximum, avoids reporting a sub-harmonic, since
/// every multiple of the period gives an almost equally high peak. See
/// [`detect_fundamental_autocorrelation`] for details. The period is an integer number of
/// samples (no interpolation), so the frequency resolution is `f²/fs`: about 4.4 Hz at
/// 440 Hz and 44.1 kHz.
///
/// For multi-channel signals, only the first channel is analysed. This function is a
/// lightweight pitch estimator; for production pitch tracking consider the methods on
/// [`AudioPitchAnalysis`](crate::operations::traits::AudioPitchAnalysis).
///
/// # Arguments
///
/// - `audio` - The audio signal to analyse.
///
/// # Returns
///
/// * `Ok(Some(freq))` - Estimated fundamental frequency in Hz.
/// * `Ok(None)` - No periodic component with `ρ > 0.1`, a silent signal, or a signal not
///   longer than `fs / 50` samples (one period of the lowest frequency).
///
/// # Errors
///
/// Returns an error if the sample layout is non-contiguous and cannot be read as a
/// contiguous slice.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::detection::detect_fundamental_frequency;
/// use audio_samples::utils::generation::sine_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// // 440 Hz at 44.1 kHz has a period of 100.2 samples; the integer lag 100 gives 441 Hz.
/// let audio = sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.1);
/// let f = detect_fundamental_frequency(&audio).unwrap().unwrap();
/// assert!((f - 440.0).abs() < 4.4);
/// ```
#[inline]
pub fn detect_fundamental_frequency<T>(audio: &AudioSamples<T>) -> AudioSampleResult<Option<f64>>
where
    T: StandardSample,
{
    // error if audio is empty
    // if audio is empty, you are doing something wrong
    let audio_f64 = audio.as_f64();

    // Use the first channel for analysis
    let data = if let Some(mono) = audio_f64.as_mono() {
        mono.as_slice()
            .ok_or_else(|| {
                AudioSampleError::Layout(LayoutError::NonContiguous {
                    operation: "audio processing".to_string(),
                    layout_type: "non-contiguous mono array".to_string(),
                })
            })?
            .to_vec()
    } else {
        // Use the first channel of multi-channel audio
        let multi = audio_f64.as_multi_channel().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_data",
                "Audio must be multi-channel",
            ))
        })?;
        multi
            .row(0)
            .as_slice()
            .ok_or_else(|| {
                AudioSampleError::Layout(LayoutError::NonContiguous {
                    operation: "audio processing".to_string(),
                    layout_type: "non-contiguous multi-channel array".to_string(),
                })
            })?
            .to_vec()
    };
    // safety: data is non-empty (channel was extracted from valid audio)
    let data = unsafe { NonEmptyVec::new_unchecked(data) };

    // Use autocorrelation method for fundamental frequency detection
    let fundamental =
        detect_fundamental_autocorrelation(&data, f64::from(audio.sample_rate().get()))?;

    Ok(fundamental)
}

/// Detects contiguous silence regions in an audio signal.
///
/// A sample (or sample column for multi-channel audio) is considered silent when its
/// absolute amplitude falls strictly below `threshold`. Contiguous runs of silent samples
/// are collected into time-domain intervals and returned as `(start_seconds, end_seconds)`
/// tuples.
///
/// For mono signals, each sample is checked independently.\
/// For multi-channel signals, a column is considered silent only when **all** channels are
/// below the threshold at that sample position.
///
/// # Arguments
///
/// - `audio` - The audio signal to analyse.
/// - `threshold` - Absolute amplitude threshold, in raw sample units of `T`. Samples with
///   `|x| < threshold` are silent. The comparison uses raw values, so for `u8`
///   (offset-binary, silence = 128) silence is never detected near 128: convert to a signed
///   or float type first.
///
/// # Returns
///
/// A vector of `(start_time, end_time)` tuples in seconds, one entry per contiguous silent
/// run. Returns an empty vector when no silence is detected.
///
/// # Errors
///
/// Returns an error if the multi-channel layout cannot be accessed.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::detection::detect_silence_regions;
/// use audio_samples::utils::generation::silence;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let audio = silence::<f32>(Duration::from_millis(200), sample_rate!(44100));
/// let regions = detect_silence_regions(&audio, 0.001f32).unwrap();
/// assert!(!regions.is_empty());
/// assert_eq!(regions[0].0, 0.0); // starts at the beginning
/// ```
#[inline]
pub fn detect_silence_regions<T>(
    audio: &AudioSamples<'_, T>,
    threshold: T,
) -> AudioSampleResult<Vec<(f64, f64)>>
where
    T: StandardSample,
{
    let mut silence_regions = Vec::new();
    let mut in_silence = false;
    let mut silence_start = 0;

    let sample_rate = audio.sample_rate().get();
    let samples_per_second = sample_rate;
    let threshold: f64 = threshold.cast_into();

    // Helper function to check if a sample is below threshold (absolute value)
    let is_below_threshold = |sample: T| -> bool {
        let abs_val: f64 = sample.cast_into();
        let abs_val = abs_val.abs();
        abs_val < threshold
    };

    if let Some(mono) = audio.as_mono() {
        for (i, &sample) in mono.iter().enumerate() {
            if is_below_threshold(sample) {
                if !in_silence {
                    silence_start = i;
                    in_silence = true;
                }
            } else if in_silence {
                let start_time = silence_start as f64 / f64::from(samples_per_second);
                let end_time = i as f64 / f64::from(samples_per_second);
                silence_regions.push((start_time, end_time));
                in_silence = false;
            }
        }

        // Handle case where silence extends to the end
        if in_silence {
            let start_time = silence_start as f64 / f64::from(samples_per_second);
            let end_time = mono.len().get() as f64 / f64::from(samples_per_second);
            silence_regions.push((start_time, end_time));
        }
    } else {
        // Multi-channel analysis - consider it silence only if ALL channels are below threshold
        let multi = audio.as_multi_channel().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_data",
                "Audio must be multi-channel",
            ))
        })?;
        for i in 0..multi.ncols().get() {
            let all_below_threshold =
                (0..multi.nrows().get()).all(|ch| is_below_threshold(multi[(ch, i)]));

            if all_below_threshold {
                if !in_silence {
                    silence_start = i;
                    in_silence = true;
                }
            } else if in_silence {
                let start_time = silence_start as f64 / f64::from(samples_per_second);
                let end_time = i as f64 / f64::from(samples_per_second);
                silence_regions.push((start_time, end_time));
                in_silence = false;
            }
        }

        // Handle case where silence extends to the end
        if in_silence {
            let start_time = silence_start as f64 / f64::from(samples_per_second);
            let end_time = multi.ncols().get() as f64 / f64::from(samples_per_second);
            silence_regions.push((start_time, end_time));
        }
    }

    Ok(silence_regions)
}

/// Computes the peak level, RMS level and crest factor of an audio signal.
///
/// Samples are converted to `f64` in `[-1, 1]` (audio-aware conversion, so `u8` 128 maps to
/// 0) and all channels are pooled. It returns the peak `max |x|`, the RMS
/// `sqrt((1/N) Σ x²)` and the crest factor `20·log10(peak / RMS)` in dB. The crest factor
/// is the peak-to-average ratio (3.01 dB for any sine, 0 dB for a square wave); it is
/// **not** the dynamic range of the recording (the ratio between its loudest and quietest
/// passages), despite the function's name.
///
/// # Arguments
///
/// - `audio` - The audio signal to analyse.
///
/// # Returns
///
/// A tuple `(peak_amplitude, rms_amplitude, dynamic_range_db)` where:
///
/// - `peak_amplitude` - Maximum absolute sample value across the entire buffer.
/// - `rms_amplitude` - Root-mean-square amplitude.
/// - `dynamic_range_db` - Crest factor `20·log10(peak / rms)` in dB. Returns `0.0` when
///   `rms_amplitude` is zero.
///
/// # Errors
///
/// Returns an error if the sample layout is non-contiguous and cannot be read as a slice.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::detection::detect_dynamic_range;
/// use audio_samples::utils::generation::sine_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let audio = sine_wave::<f32>(440.0, Duration::from_millis(100), sample_rate!(44100), 1.0);
/// let (peak, rms, crest_db) = detect_dynamic_range(&audio).unwrap();
/// assert!(peak <= 1.0);
/// assert!((rms - 1.0 / 2f64.sqrt()).abs() < 1e-3); // sine RMS = A/√2
/// assert!((crest_db - 3.0103).abs() < 0.01); // crest factor of a sine = √2 ≈ 3.01 dB
/// ```
#[inline]
pub fn detect_dynamic_range<T>(audio: &AudioSamples<T>) -> AudioSampleResult<(f64, f64, f64)>
where
    T: StandardSample,
{
    let audio_f = audio.as_f64();

    let data: Vec<f64> = if let Some(mono) = audio_f.as_mono() {
        mono.as_slice()
            .ok_or_else(|| {
                AudioSampleError::Layout(LayoutError::NonContiguous {
                    operation: "audio processing".to_string(),
                    layout_type: "non-contiguous mono array".to_string(),
                })
            })?
            .to_vec()
    } else {
        // Flatten multi-channel audio
        let multi = audio_f.as_multi_channel().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_data",
                "Audio must be multi-channel",
            ))
        })?;
        multi
            .as_slice()
            .ok_or_else(|| {
                AudioSampleError::Layout(LayoutError::NonContiguous {
                    operation: "audio processing".to_string(),
                    layout_type: "non-contiguous multi-channel array".to_string(),
                })
            })?
            .to_vec()
    };

    if data.is_empty() {
        return Ok((0.0, 0.0, 0.0));
    }

    // Calculate peak amplitude
    let peak_amplitude = data.iter().map(|&x| x.abs()).fold(0.0, f64::max);

    // Calculate RMS amplitude
    let rms_amplitude = (data.iter().map(|x| x * x).sum::<f64>() / data.len() as f64).sqrt();

    // Calculate dynamic range in dB
    let dynamic_range_db = if rms_amplitude > 0.0 {
        20.0 * (peak_amplitude / rms_amplitude).log10()
    } else {
        0.0
    };

    Ok((peak_amplitude, rms_amplitude, dynamic_range_db))
}

/// Detects contiguous clipping regions in an audio signal.
///
/// A sample is considered clipped when its value reaches or exceeds
/// `threshold_ratio × T::MAX` or falls at or below `threshold_ratio × T::MIN`. Contiguous
/// clipped runs are collected into time-domain intervals.
///
/// For mono signals, each sample is checked independently.\
/// For multi-channel signals, a sample column is considered clipped when **any** channel
/// at that position is clipped.
///
/// # Arguments
///
/// - `audio` - The audio signal to analyse.
/// - `threshold_ratio` - Fraction of the sample type's full-scale value that constitutes
///   clipping. A value of `0.99` means samples within 1% of the full-scale limit are
///   considered clipped. Must be in `(0, 1]`.
///
/// # Returns
///
/// A vector of `(start_time, end_time)` tuples in seconds, one entry per contiguous clipped
/// run. Returns an empty vector when no clipping is detected.
///
/// # Errors
///
/// Returns an error if the multi-channel layout cannot be accessed.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::detection::detect_clipping;
/// use audio_samples::utils::generation::sine_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// // A sine wave with amplitude 1.0 in f32 should trigger clipping detection near its peaks.
/// let audio = sine_wave::<f32>(440.0, Duration::from_millis(100), sample_rate!(44100), 1.0);
/// let regions = detect_clipping(&audio, 0.99).unwrap();
/// // The sine wave reaches exactly ±1.0 so at least one clipped region is expected.
/// assert!(!regions.is_empty());
/// ```
#[inline]
pub fn detect_clipping<T>(
    audio: &AudioSamples<T>,
    threshold_ratio: f64,
) -> AudioSampleResult<Vec<(f64, f64)>>
where
    T: StandardSample,
{
    #[inline]
    fn is_clipped<T>(sample: T, upper_threshold: T, lower_threshold: T) -> bool
    where
        T: StandardSample,
    {
        sample >= upper_threshold || sample <= lower_threshold
    }
    let mut clipped_regions = Vec::new();
    let mut in_clipped = false;
    let mut clipped_start = 0;

    let sample_rate = audio.sample_rate().get();

    // Determine clipping thresholds
    let max_val: f64 = T::MAX.cast_into();
    let min_val: f64 = T::MIN.cast_into();

    let upper_threshold: T = T::cast_from(max_val * threshold_ratio);
    let lower_threshold: T = T::cast_from(min_val * threshold_ratio);

    if let Some(mono) = audio.as_mono() {
        for (i, &sample) in mono.iter().enumerate() {
            if is_clipped(sample, upper_threshold, lower_threshold) {
                if !in_clipped {
                    clipped_start = i;
                    in_clipped = true;
                }
            } else if in_clipped {
                let start_time = clipped_start as f64 / f64::from(sample_rate);
                let end_time = i as f64 / f64::from(sample_rate);
                clipped_regions.push((start_time, end_time));
                in_clipped = false;
            }
        }

        // Handle case where clipping extends to the end
        if in_clipped {
            let start_time = clipped_start as f64 / f64::from(sample_rate);
            let end_time = mono.len().get() as f64 / f64::from(sample_rate);
            clipped_regions.push((start_time, end_time));
        }
    } else {
        // Multi-channel analysis - consider it clipped if ANY channel is clipped
        let multi = audio.as_multi_channel().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_data",
                "Audio must be mono or multi-channel",
            ))
        })?;
        for i in 0..multi.ncols().get() {
            let mut any_clipped = false;
            for ch in 0..multi.nrows().get() {
                if is_clipped(multi[(ch, i)], upper_threshold, lower_threshold) {
                    any_clipped = true;
                    break;
                }
            }

            if any_clipped {
                if !in_clipped {
                    clipped_start = i;
                    in_clipped = true;
                }
            } else if in_clipped {
                let start_time = clipped_start as f64 / f64::from(sample_rate);
                let end_time = i as f64 / f64::from(sample_rate);
                clipped_regions.push((start_time, end_time));
                in_clipped = false;
            }
        }

        // Handle case where clipping extends to the end
        if in_clipped {
            let start_time = clipped_start as f64 / f64::from(sample_rate);
            let end_time = multi.ncols().get() as f64 / f64::from(sample_rate);
            clipped_regions.push((start_time, end_time));
        }
    }

    Ok(clipped_regions)
}

/// Estimates the noise floor of an audio signal in dBFS.
///
/// Converts the first channel to `f64` in `[-1, 1]`, sorts the instantaneous magnitudes
/// `|x[n]|` and returns `20·log10` of the value at index `⌊N/10⌋` (the 10th percentile) as a
/// proxy for the noise floor, in dB relative to full scale (1.0).
///
/// This is a crude proxy. The percentile is taken over instantaneous sample magnitudes, not
/// over short-time frame levels, so for tonal or speech signals it reflects the samples near
/// the waveform's zero crossings rather than background noise: a noiseless full-scale sine
/// reports `20·log10(sin(π·0.05)) ≈ −16.1` dBFS. It is meaningful mainly when at least 10 %
/// of the samples are pure background noise.
///
/// # Arguments
///
/// - `audio` - The audio signal to analyse.
///
/// # Returns
///
/// * `Ok(Some(db))` - Estimated noise floor in dBFS (decibels relative to full scale).
/// * `Ok(None)` - The signal has fewer than 10 samples or the 10th-percentile magnitude is
///   exactly zero.
///
/// # Errors
///
/// Returns an error if the sample layout is non-contiguous and cannot be read as a slice.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::detection::estimate_noise_floor;
/// use audio_samples::utils::generation::sine_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// // A noiseless sine still reports a "floor": 20·log10(0.5·sin(π·0.05)) ≈ -22.1 dBFS.
/// let audio = sine_wave::<f32>(440.0, Duration::from_secs(1), sample_rate!(44100), 0.5);
/// let floor_db = estimate_noise_floor::<f32, ()>(&audio).unwrap().unwrap();
/// assert!((floor_db - (-22.1)).abs() < 0.5);
/// ```
#[inline]
pub fn estimate_noise_floor<T, F>(audio: &AudioSamples<T>) -> AudioSampleResult<Option<f64>>
where
    T: StandardSample,
{
    let audio = audio.as_f64();

    let data: Vec<f64> = if let Some(mono) = audio.as_mono() {
        mono.as_slice()
            .ok_or_else(|| {
                AudioSampleError::Layout(LayoutError::NonContiguous {
                    operation: "audio processing".to_string(),
                    layout_type: "non-contiguous mono array".to_string(),
                })
            })?
            .to_vec()
    } else {
        // Use first channel
        let multi = audio.as_multi_channel().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_data",
                "Audio must be multi-channel",
            ))
        })?;
        multi
            .row(0)
            .as_slice()
            .ok_or_else(|| {
                AudioSampleError::Layout(LayoutError::NonContiguous {
                    operation: "audio processing".to_string(),
                    layout_type: "non-contiguous multi-channel array".to_string(),
                })
            })?
            .to_vec()
    };

    if data.is_empty() {
        return Ok(None);
    }

    // Find quietest regions (bottom 10th percentile)
    let mut sorted_abs: Vec<f64> = data.iter().map(|&x| x.abs()).collect();
    sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let percentile_10 = sorted_abs.len() / 10;
    if percentile_10 > 0 {
        let noise_level = sorted_abs[percentile_10];
        if noise_level > 0.0 {
            let noise_floor_db = 20.0 * noise_level.log10();
            return Ok(Some(noise_floor_db));
        }
    }

    Ok(None)
}

/// Estimates the active frequency range of an audio signal.
///
/// Computes the power spectrum of the first channel and returns the lowest and highest
/// frequencies that carry more than 1% of the peak spectral energy. Returns `None` when the
/// signal is shorter than 1024 samples or when no bin exceeds the threshold.
///
/// # Arguments
///
/// - `audio` - The audio signal to analyse. Only the first channel is used.
///
/// # Returns
///
/// * `Ok(Some((low_hz, high_hz)))` - Approximate frequency range with significant energy.
/// * `Ok(None)` - Signal too short (< 1024 samples) or no significant energy found.
///
/// # Errors
///
/// Returns an error if the underlying spectral computation fails or if the sample layout
/// is non-contiguous.
///
/// # Feature flag
///
/// Requires `feature = "transforms"`.
#[cfg(feature = "transforms")]
#[inline]
pub fn estimate_frequency_range<T>(audio: &AudioSamples<T>) -> AudioSampleResult<Option<(f64, f64)>>
where
    T: StandardSample,
{
    let audio = audio.as_f64();

    let data: Vec<f64> = if let Some(mono) = audio.as_mono() {
        mono.as_slice()
            .ok_or_else(|| {
                AudioSampleError::Layout(LayoutError::NonContiguous {
                    operation: "audio processing".to_string(),
                    layout_type: "non-contiguous mono array".to_string(),
                })
            })?
            .to_vec()
    } else {
        // Use first channel
        let multi = audio.as_multi_channel().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_data",
                "Audio must be multi-channel",
            ))
        })?;
        multi
            .row(0)
            .as_slice()
            .ok_or_else(|| {
                AudioSampleError::Layout(LayoutError::NonContiguous {
                    operation: "audio processing".to_string(),
                    layout_type: "non-contiguous multi-channel array".to_string(),
                })
            })?
            .to_vec()
    };

    if data.len() < 1024 {
        return Ok(None); // Too short for meaningful analysis
    }

    // Compute spectrum
    // safety: data is non-empty
    let n_fft = unsafe { NonZeroUsize::new_unchecked(data.len().next_power_of_two()) };
    // safety: data is non-empty (channel was extracted from valid audio)
    let data_slice = unsafe { NonEmptySlice::new_unchecked(&data[..]) };
    let spectrum = power_spectrum(data_slice, n_fft, Some(WindowType::Hanning))
        .map_err(AudioSampleError::from)?;
    let nyquist = audio.nyquist();

    // Find frequency range with significant energy
    let threshold = spectrum.iter().fold(0.0, |acc: f64, &x| acc.max(x)) * 0.01; // 1% of peak

    let mut low_freq = None;
    let mut high_freq = None;

    for (i, &energy) in spectrum.iter().enumerate() {
        if energy > threshold {
            let freq = i as f64 / spectrum.len().get() as f64 * nyquist;
            if low_freq.is_none() {
                low_freq = Some(freq);
            }
            high_freq = Some(freq);
        }
    }

    if let (Some(low), Some(high)) = (low_freq, high_freq) {
        Ok(Some((low, high)))
    } else {
        Ok(None)
    }
}

/// Analyses a power spectrum for spectral cutoff frequencies that indicate prior resampling.
///
/// For each candidate rate `R` (8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200,
/// 96000 Hz) whose Nyquist `R/2` is below `nyquist_freq`, the bin
/// `b = round((R/2) / nyquist_freq · (len − 1))` is located and the summed power of the
/// 5 bins below it (`b−5..b`) is compared with the summed power of the 10 bins from it
/// (`b..b+10`). If the lower sum is more than twice the upper sum, `R` is a candidate; the
/// first (lowest) candidate is returned.
///
/// # Arguments
///
/// - `spectrum` - Non-empty one-sided power spectrum of the signal (FFT magnitude squared),
///   with `len` bins spanning 0 Hz to `nyquist_freq` inclusive (bin `k` at
///   `k·nyquist_freq/(len − 1)`).
/// - `nyquist_freq` - Nyquist frequency of the audio that produced `spectrum` in Hz.
///
/// # Returns
///
/// The first candidate sample rate (in Hz) whose Nyquist frequency shows a significant
/// energy drop, or `None` if no candidate is found.
/// A non-finite `nyquist_freq` yields `None`; spectra with 10 or fewer bins yield `None`.
#[inline]
#[must_use]
pub fn analyze_spectrum_for_cutoff(
    spectrum: &NonEmptySlice<f64>,
    nyquist_freq: f64,
) -> Option<u32> {
    // Look for sharp cutoffs that might indicate resampling
    let len = spectrum.len().get();

    // Check for energy drops that might indicate filtering
    let mut candidates = Vec::new();

    // Common resampling target frequencies
    let common_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000];

    for &rate in &common_rates {
        let target_nyquist = f64::from(rate) / 2.0;
        if target_nyquist < nyquist_freq {
            // Check if there's a significant drop in energy around this frequency
            // The one-sided spectrum has `len` bins spanning 0..=nyquist_freq, so
            // bin k lies at k·nyquist_freq/(len-1).
            let bin_index = (target_nyquist / nyquist_freq * (len - 1) as f64).round() as usize;
            if len > 10 && bin_index < len - 10 {
                let energy_before: f64 = spectrum[bin_index.saturating_sub(5)..bin_index]
                    .iter()
                    .fold(0.0, |acc, &x| acc + x);
                let energy_after = spectrum[bin_index..bin_index + 10]
                    .iter()
                    .fold(0.0, |acc, &x| acc + x);

                if energy_before > energy_after * 2.0 {
                    candidates.push(rate);
                }
            }
        }
    }

    // Return the most likely candidate
    candidates.first().copied()
}

/// Estimates the fundamental frequency from a sample buffer using autocorrelation.
///
/// With `r(τ) = (1/(N−τ)) Σ_{n=0}^{N−1−τ} x[n]·x[n+τ]` (unbiased, no mean removal) and the
/// normalised autocorrelation `ρ(τ) = r(τ)/r(0)`, the integer lags
/// `τ ∈ [⌊fs/2000⌋, min(⌊fs/50⌋, ⌊N/2⌋))` are searched (50 Hz - 2 000 Hz). Let `ρ_max` be
/// the highest `ρ` in that range. If `ρ_max ≤ 0.1` there is no periodic component.
/// Otherwise the result is `fs / τ*`, where `τ*` is the **first** lag that is a local
/// maximum of `ρ` (`ρ(τ) ≥ ρ(τ−1)` and `ρ(τ) ≥ ρ(τ+1)`) with `ρ(τ) ≥ 0.9·ρ_max`. Because
/// `ρ` is normalised by `r(0)`, the result does not depend on the signal amplitude; taking
/// the first qualifying peak avoids reporting a sub-harmonic (multiples of the period give
/// almost equally high peaks). No interpolation is done, so `τ*` is an integer number of
/// samples. The cost is `O(N · fs/50)`.
///
/// # Arguments
///
/// - `data` - Non-empty slice of audio samples (mono, f64).
/// - `sample_rate` - Sampling rate in Hz. Should be finite and positive; other values
///   yield `Ok(None)`.
///
/// # Returns
///
/// * `Ok(Some(freq))` - Estimated fundamental frequency `fs / τ*` in Hz.
/// * `Ok(None)` - No periodic component (`ρ_max ≤ 0.1` or no qualifying local maximum), a
///   silent signal, or `N ≤ fs/50` (the signal is too short to cover the longest period).
///
/// # Errors
///
/// This function is currently infallible; the `Result` wrapper is reserved for future use.
#[inline]
pub fn detect_fundamental_autocorrelation(
    data: &NonEmptySlice<f64>,
    sample_rate: f64,
) -> AudioSampleResult<Option<f64>> {
    if data.len() < crate::nzu!(2) {
        return Ok(None);
    }

    let min_freq = 50.0; // Minimum fundamental frequency to detect
    let max_freq = 2000.0; // Maximum fundamental frequency to detect
    // Minimum normalised autocorrelation rho for a periodic component.
    const CORRELATION_THRESHOLD: f64 = 0.1;
    // A lag qualifies as the period if its rho is at least this fraction of
    // the highest rho in the search range.
    const PEAK_FRACTION: f64 = 0.9;

    let min_period = ((sample_rate / max_freq) as usize).max(1);
    let max_period = (sample_rate / min_freq) as usize;
    if max_period >= data.len().get() {
        return Ok(None);
    }

    // Mean square r(0); a silent signal has no period.
    let n = data.len().get();
    let r0 = data.iter().map(|&x| x * x).sum::<f64>() / n as f64;
    if r0.is_nan() || r0 <= 0.0 {
        return Ok(None);
    }

    // Normalised unbiased autocorrelation rho(tau) = r(tau) / r(0), with
    // r(tau) = (1 / (N - tau)) * sum_i x[i] x[i + tau]. Normalising by r(0)
    // makes the threshold independent of the signal amplitude.
    let last = max_period.min(n / 2);
    if last <= min_period + 1 {
        return Ok(None);
    }
    let lo = min_period.saturating_sub(1);
    let rho_at = |tau: usize| -> f64 {
        let count = n - tau;
        let acc: f64 = data[..count]
            .iter()
            .zip(data[tau..].iter())
            .map(|(&a, &b)| a * b)
            .sum();
        acc / count as f64 / r0
    };
    // rho[j] holds rho(lo + j) for tau in [lo, last].
    let rho: Vec<f64> = (lo..=last).map(rho_at).collect();

    // Highest peak in the search range [min_period, last).
    let max_rho = (min_period..last)
        .map(|tau| rho[tau - lo])
        .fold(f64::NEG_INFINITY, f64::max);
    if max_rho.is_nan() || max_rho <= CORRELATION_THRESHOLD {
        return Ok(None);
    }

    // Every multiple of the true period is (almost) as high as the first, so
    // taking the global maximum picks sub-harmonics. Instead take the FIRST
    // local maximum whose height is within PEAK_FRACTION of the highest one.
    for tau in min_period..last {
        let j = tau - lo;
        let v = rho[j];
        let is_local_max = v >= rho[j.saturating_sub(1)] && v >= rho[j + 1];
        if is_local_max && v >= PEAK_FRACTION * max_rho {
            return Ok(Some(sample_rate / tau as f64));
        }
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample_rate;
    use ndarray::array;
    use non_empty_slice::non_empty_vec;

    #[test]
    fn test_detect_silence_regions() {
        // Create a signal with silence regions
        let data = array![0.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let audio: AudioSamples<'_, f32> = AudioSamples::new_mono(data, sample_rate!(10)).unwrap(); // 10 Hz sample rate for easy calculation

        let silence_regions: Vec<(f64, f64)> =
            detect_silence_regions(&audio, 0.5_f32).expect("Failed to detect silence");

        assert!(!silence_regions.is_empty());
        // First silence region should be at the beginning
        assert_eq!(silence_regions[0].0, 0.0);
        assert_eq!(silence_regions[0].1, 0.3);
    }

    #[test]
    fn test_detect_dynamic_range() {
        // Create a signal with known dynamic range
        let data = non_empty_vec![0.1f32, 0.5, 1.0, 0.2, 0.8];
        let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(data, sample_rate!(10));

        let (peak, rms, dynamic_range) = detect_dynamic_range(&audio).unwrap();

        assert_eq!(peak, 1.0);
        assert!(rms > 0.0);
        assert!(dynamic_range > 0.0);
    }

    // detect_dynamic_range must use sqrt(mean(x^2)), not sqrt(mean(x)). For
    // zero-mean audio the latter is ~0 (giving a wrong/infinite dynamic range)
    // and NaN when the sample sum is negative. A sine of amplitude A has
    // RMS = A/sqrt(2).
    #[test]
    fn test_detect_dynamic_range_rms_correct_and_finite() {
        let sr = 44100u32;
        let amplitude = 0.8f64;
        let n = sr as usize; // 1 second
        let two_pi = std::f64::consts::PI * 2.0;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / sr as f64;
            data.push(amplitude * (two_pi * 440.0 * t).sin());
        }
        let arr = ndarray::Array1::from_vec(data);
        let audio: AudioSamples<'_, f64> =
            AudioSamples::new_mono(arr, sample_rate!(44100)).unwrap();

        let (peak, rms, dynamic_range) = detect_dynamic_range(&audio).unwrap();

        let expected_rms = amplitude / 2.0f64.sqrt();
        assert!(
            (rms - expected_rms).abs() < expected_rms * 0.01,
            "expected RMS ~{expected_rms}, got {rms}"
        );
        assert!(rms.is_finite() && dynamic_range.is_finite());
        assert!(peak > 0.0);
        // Dynamic range for a full sine: 20*log10(peak/rms) ~ 3.01 dB.
        assert!((dynamic_range - 20.0 * (peak / rms).log10()).abs() < 1e-9);
        assert!(dynamic_range > 0.0);
    }

    // The buggy fold(acc + x) summed raw samples; for a signal whose sample
    // sum is negative, sqrt of a negative number is NaN. The mean-of-squares
    // form is always non-negative, so the result must be finite.
    #[test]
    fn test_detect_dynamic_range_negative_sum_not_nan() {
        // Sample sum is clearly negative.
        let data = array![-0.9f64, -0.8, -0.7, 0.1, -0.5];
        let audio: AudioSamples<'_, f64> = AudioSamples::new_mono(data, sample_rate!(10)).unwrap();

        let (_, rms, dynamic_range) = detect_dynamic_range(&audio).unwrap();

        assert!(
            !rms.is_nan(),
            "RMS must not be NaN for a negative-sum signal"
        );
        assert!(rms.is_finite() && rms > 0.0);
        assert!(!dynamic_range.is_nan() && dynamic_range.is_finite());
    }

    #[test]
    fn test_detect_clipping() {
        // Create a signal with clipping
        let data = array![0.5f32, 1.0, 1.0, 1.0, 0.5, -1.0, -1.0, 0.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(8)).unwrap(); // 8 Hz sample rate for easy calculation

        let clipped_regions = detect_clipping(&audio, 0.99).expect("Failed to detect clipping");

        assert!(!clipped_regions.is_empty());
    }

    #[test]
    fn test_detect_fundamental_frequency() {
        // Create a sine wave with known frequency
        let sample_rate = crate::sample_rate!(44100);
        let frequency: f64 = 440.0; // A4
        let duration: f64 = 1.0; // 1 second

        let audio = crate::sine_wave::<f64>(
            frequency,
            std::time::Duration::from_secs_f64(duration),
            sample_rate,
            1.0f64,
        );

        let detected_freq: Option<f64> = detect_fundamental_frequency(&audio).unwrap();

        if let Some(freq) = detected_freq {
            // Integer-lag resolution at 440 Hz / 44.1 kHz is about 4.4 Hz; a
            // sub-harmonic (220 or 110 Hz) must not be reported.
            let tolerance: f64 = 0.01 * frequency;

            assert!(
                (freq - frequency).abs() < tolerance,
                "Expected: {} Hz, Detected: {} Hz, Difference: {} Hz",
                frequency,
                freq,
                (freq - frequency).abs()
            );
        } else {
            panic!("No frequency detected");
        }
    }
}
