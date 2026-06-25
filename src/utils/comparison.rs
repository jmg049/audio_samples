//! Audio signal comparison and similarity metrics.
//!
//! This module provides a small collection of signal-level comparison operators for
//! analysing similarity, error, and alignment between audio signals. The exposed functions
//! operate on [`AudioSamples`] and return scalar metrics or derived signals that are suitable
//! for validation, evaluation, diagnostics, and lightweight analysis workflows.
//!
//! The primary purpose of this module is to centralise common comparison semantics.
//!
//! # Usage
//!
//! Typical usage consists of passing two compatible audio signals into a metric function
//! and interpreting the returned scalar or aligned signal. All comparison functions require
//! matching dimensionality and compatible channel layouts unless explicitly documented
//! otherwise.
//!
//! ```rust
//! use audio_samples::utils::comparison::{correlation, mse};
//! use audio_samples::utils::generation::sine_wave;
//! use audio_samples::sample_rate;
//! use std::time::Duration;
//!
//! let sr = sample_rate!(44100);
//! let a = sine_wave::<f32>(440.0, Duration::from_millis(100), sr, 1.0);
//! let b = sine_wave::<f32>(440.0, Duration::from_millis(100), sr, 1.0);
//!
//! let corr: f64 = correlation(&a, &b).unwrap();
//! let error: f64 = mse(&a, &b).unwrap();
//!
//! assert!((corr - 1.0).abs() < 1e-6);
//! assert!(error < 1e-10);
//! ```
use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, LayoutError,
    ParameterError, traits::StandardSample,
};
use ndarray::{Array1, ArrayView1};

/// Computes the Pearson correlation coefficient between two audio signals.
///
/// This function measures linear similarity between two signals on a per-sample basis and
/// returns a scalar correlation coefficient in the range `[-1, 1]`. Positive values indicate
/// positive correlation, negative values indicate inverse correlation, and values near zero
/// indicate weak linear relationship.
///
/// For mono signals, the correlation is computed directly over the single channel.\
/// For multi-channel signals, the correlation is computed independently for each channel and
/// the results are averaged to produce a single scalar score.
///
/// # Arguments
///
/// * `a`\
///   The first audio signal.
///
/// * `b`\
///   The second audio signal.
///
/// Both signals must have identical channel counts and the same number of samples per
/// channel.
///
/// # Returns
///
/// A scalar correlation coefficient in the range `[-1, 1]`, expressed in the requested
/// floating-point type.
///
/// # Errors
///
/// Returns an error in the following cases:
///
/// * The two signals have different channel counts.
/// * The two signals have different numbers of samples per channel.
/// * The underlying sample layout is non-contiguous and cannot be viewed as a slice.
/// * The channel configuration of the two signals does not match (mono vs multi-channel).
///
/// # Behavioural Guarantees
///
/// * The returned value lies in the closed interval `[-1, 1]` for all finite inputs.
/// * For identical signals, the returned value is `1.0` (up to floating-point rounding).
/// * For multi-channel inputs, the result is the arithmetic mean of per-channel correlations.
/// * If either signal has zero variance over a channel, the correlation for that channel is
///   defined as `0.0`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::comparison::correlation;
/// use audio_samples::utils::generation::sine_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let sr = sample_rate!(44100);
/// let a = sine_wave::<f32>(440.0, Duration::from_millis(100), sr, 1.0);
/// let b = sine_wave::<f32>(440.0, Duration::from_millis(100), sr, 1.0);
///
/// let corr = correlation(&a, &b).unwrap();
/// assert!((corr - 1.0).abs() < 1e-6); // identical signals → correlation 1.0
/// ```
#[inline]
pub fn correlation<T>(a: &AudioSamples<T>, b: &AudioSamples<T>) -> AudioSampleResult<f64>
where
    T: StandardSample,
{
    if a.num_channels() != b.num_channels() || a.samples_per_channel() != b.samples_per_channel() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "audio_signals",
            "Signals must have the same dimensions for correlation",
        )));
    }

    let a_f = a.as_f64();
    let b_f = b.as_f64();

    match (a_f.as_mono(), b_f.as_mono()) {
        (Some(a_mono), Some(b_mono)) => {
            let corr = correlation_1d(&a_mono.view(), &b_mono.view())?;
            Ok(corr)
        }
        (Some(_), None) | (None, Some(_)) => {
            Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Signals must have the same channel configuration",
            )))
        }
        (None, None) => {
            // Multi-channel correlation - compute average correlation across channels
            let a_multi = a_f.as_multi_channel().ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Must be multi-channel audio",
                ))
            })?;
            let b_multi = b_f.as_multi_channel().ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Must be multi-channel audio",
                ))
            })?;

            let mut correlations = Vec::new();
            for i in 0..a_multi.nrows().get() {
                let a_channel = a_multi.row(i);
                let b_channel = b_multi.row(i);
                let corr = correlation_1d_slice(
                    a_channel.as_slice().ok_or_else(|| {
                        AudioSampleError::Layout(LayoutError::NonContiguous {
                            operation: "signal processing".to_string(),
                            layout_type: "non-contiguous multi-channel samples".to_string(),
                        })
                    })?,
                    b_channel.as_slice().ok_or_else(|| {
                        AudioSampleError::Layout(LayoutError::NonContiguous {
                            operation: "signal processing".to_string(),
                            layout_type: "non-contiguous multi-channel samples".to_string(),
                        })
                    })?,
                )?;
                correlations.push(corr);
            }

            Ok(correlations.iter().fold(0.0, |acc, x| acc + *x) / correlations.len() as f64)
        }
    }
}

/// Computes the mean squared error (MSE) between two audio signals.
///
/// This function measures the average squared per-sample difference between two signals.
/// Lower values indicate higher similarity, with a value of `0.0` indicating identical
/// signals under exact arithmetic. The metric is unnormalised and scales quadratically with
/// signal amplitude.
///
/// For mono signals, the MSE is computed directly over the single channel.\
/// For multi-channel signals, the MSE is computed independently for each channel and the
/// results are averaged to produce a single scalar value.
///
/// # Arguments
///
/// * `a`\
///   The first audio signal.
///
/// * `b`\
///   The second audio signal.
///
/// Both signals must have identical channel counts and the same number of samples per
/// channel.
///
/// # Returns
///
/// The mean squared error as a non-negative scalar value expressed in the requested
/// floating-point type.
///
/// # Errors
///
/// Returns an error in the following cases:
///
/// * The two signals have different channel counts.
/// * The two signals have different numbers of samples per channel.
/// * The underlying sample layout is non-contiguous and cannot be viewed as a slice.
/// * The channel configuration of the two signals does not match (mono vs multi-channel).
///
/// # Behavioural Guarantees
///
/// * The returned value is always greater than or equal to `0.0` for all finite inputs.
/// * For identical signals, the returned value is `0.0` up to floating-point rounding.
/// * For multi-channel inputs, the result is the arithmetic mean of per-channel MSE values.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::comparison::mse;
/// use audio_samples::utils::generation::sine_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let sr = sample_rate!(44100);
/// let a = sine_wave::<f32>(440.0, Duration::from_millis(100), sr, 1.0);
/// let b = sine_wave::<f32>(440.0, Duration::from_millis(100), sr, 1.0);
///
/// let error = mse(&a, &b).unwrap();
/// assert!(error < 1e-10); // identical signals → zero MSE
/// ```
#[inline]
pub fn mse<T>(a: &AudioSamples<T>, b: &AudioSamples<T>) -> AudioSampleResult<f64>
where
    T: StandardSample,
{
    if a.num_channels() != b.num_channels() || a.samples_per_channel() != b.samples_per_channel() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "audio_signals",
            "Signals must have the same dimensions for MSE",
        )));
    }

    let a_f = a.as_f64();
    let b_f = b.as_f64();

    match (a_f.as_mono(), b_f.as_mono()) {
        (Some(a_mono), Some(b_mono)) => mse_1d(&a_mono.view(), &b_mono.view()),
        (Some(_), None) | (None, Some(_)) => {
            Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Signals must have the same channel configuration",
            )))
        }
        (None, None) => {
            // Multi-channel MSE - compute average MSE across channels
            let a_multi = a_f.as_multi_channel().ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Must be multi-channel audio",
                ))
            })?;
            let b_multi = b_f.as_multi_channel().ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Must be multi-channel audio",
                ))
            })?;

            let mut mses = Vec::new();
            for i in 0..a_multi.nrows().get() {
                let a_channel = a_multi.row(i);
                let b_channel = b_multi.row(i);
                let mse = mse_1d_slice(
                    a_channel.as_slice().ok_or_else(|| {
                        AudioSampleError::Layout(LayoutError::NonContiguous {
                            operation: "signal processing".to_string(),
                            layout_type: "non-contiguous multi-channel samples".to_string(),
                        })
                    })?,
                    b_channel.as_slice().ok_or_else(|| {
                        AudioSampleError::Layout(LayoutError::NonContiguous {
                            operation: "signal processing".to_string(),
                            layout_type: "non-contiguous multi-channel samples".to_string(),
                        })
                    })?,
                )?;
                mses.push(mse);
            }

            Ok(mses.iter().fold(0.0, |acc, x| acc + *x) / mses.len() as f64)
        }
    }
}

/// Computes the signal-to-noise ratio (SNR) between a signal and a noise signal.
///
/// This function measures the ratio between the average power of a signal and the average
/// power of a corresponding noise signal, expressed in decibels. Higher values indicate
/// greater dominance of the signal relative to noise.
///
/// The function assumes that `signal` and `noise` are already aligned and represent
/// compatible components. No validation is performed to determine whether the inputs
/// actually correspond to a signal–noise decomposition.
///
/// For mono inputs, power is computed over the single channel.\
/// For multi-channel inputs, power is computed over all samples across all channels and
/// aggregated into a single scalar value. No per-channel SNR values are returned.
///
/// # Arguments
///
/// * `signal`\
///   The signal component.
///
/// * `noise`\
///   The noise component.
///
/// Both inputs must have identical channel counts and the same number of samples per
/// channel.
///
/// # Returns
///
/// The signal-to-noise ratio in decibels. Larger values indicate higher signal quality.
///
/// If the estimated noise power is zero, the function returns positive infinity.
///
/// # Errors
///
/// Returns an error in the following cases:
///
/// * The two inputs have different channel counts.
/// * The two inputs have different numbers of samples per channel.
/// * The underlying sample layout is non-contiguous and cannot be viewed as a slice.
/// * The channel configuration of the two signals does not match (mono vs multi-channel).
///
/// # Behavioural Guarantees
///
/// * For finite non-zero noise power, the returned value is finite.
/// * If the noise power is exactly zero, the returned value is positive infinity.
/// * The returned value increases monotonically with increasing signal power for fixed
///   noise power.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::comparison::snr;
/// use audio_samples::utils::generation::sine_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let sr = sample_rate!(44100);
/// // Signal with amplitude 1.0, noise with amplitude 0.01.
/// let signal = sine_wave::<f32>(440.0, Duration::from_millis(100), sr, 1.0);
/// let noise  = sine_wave::<f32>(880.0, Duration::from_millis(100), sr, 0.01);
///
/// let db = snr(&signal, &noise).unwrap();
/// assert!(db > 30.0); // high-power signal relative to low-power noise → high SNR
/// ```
#[inline]
pub fn snr<T>(signal: &AudioSamples<T>, noise: &AudioSamples<T>) -> AudioSampleResult<f64>
where
    T: StandardSample,
{
    if signal.num_channels() != noise.num_channels()
        || signal.samples_per_channel() != noise.samples_per_channel()
    {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "audio_signals",
            "Signal and noise must have the same dimensions for SNR",
        )));
    }

    let signal_f = signal.as_f64();
    let noise_f = noise.as_f64();

    // Calculate signal power
    let signal_power = if let Some(mono) = signal_f.as_mono() {
        mono.iter().map(|&x| x * x).fold(0.0, |acc, x| acc + x) / mono.len().get() as f64
    } else {
        let multi = signal_f.as_multi_channel().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Must be multi-channel audio",
            ))
        })?;
        multi.iter().map(|&x| x * x).fold(0.0, |acc, x| acc + x) / multi.len().get() as f64
    };

    // Calculate noise power
    let noise_power = if let Some(mono) = noise_f.as_mono() {
        mono.iter().map(|x| *x * *x).fold(0.0, |acc, x| acc + x) / mono.len().get() as f64
    } else {
        let multi = noise_f.as_multi_channel().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Must be multi-channel audio",
            ))
        })?;
        multi.iter().map(|&x| x * x).fold(0.0, |acc, x| acc + x) / multi.len().get() as f64
    };

    if noise_power == 0.0 {
        return Ok(f64::INFINITY);
    }

    let snr_db = 10.0 * (signal_power / noise_power).log10();
    Ok(snr_db)
}

/// Computes the Peak Signal-to-Noise Ratio (PSNR) between a reference and a test signal.
///
/// PSNR relates the peak amplitude of the reference signal to the power of the error between
/// the two signals, expressed in decibels:
///
/// ```text
/// PSNR = 20 · log10(peak / sqrt(MSE))
/// ```
///
/// where `peak` is the maximum absolute amplitude observed in `reference` and `MSE` is the
/// mean squared error between `reference` and `test` (see [`mse`]). Higher values indicate a
/// test signal that more closely reproduces the reference.
///
/// For mono signals the metric is computed over the single channel.\
/// For multi-channel signals the peak is taken across all channels and the MSE is the mean
/// across all channels, matching the aggregation performed by [`mse`].
///
/// # Arguments
///
/// * `reference`\
///   The reference signal whose peak amplitude defines the dynamic range.
///
/// * `test`\
///   The signal under evaluation.
///
/// Both signals must have identical channel counts and the same number of samples per
/// channel.
///
/// # Returns
///
/// The peak signal-to-noise ratio in decibels.
///
/// If the two signals are identical (`MSE == 0`), the function returns
/// [`f64::INFINITY`], since there is no error to measure.
///
/// # Errors
///
/// Returns an error in the following cases:
///
/// * The two signals have different channel counts.
/// * The two signals have different numbers of samples per channel.
/// * The underlying sample layout is non-contiguous and cannot be viewed as a slice.
/// * The channel configuration of the two signals does not match (mono vs multi-channel).
///
/// # Behavioural Guarantees
///
/// * For identical signals the returned value is [`f64::INFINITY`].
/// * For a fixed reference, the returned value decreases monotonically as the MSE increases.
/// * If the reference peak amplitude is zero while the signals differ, the returned value is
///   [`f64::NEG_INFINITY`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::comparison::psnr;
/// use audio_samples::utils::generation::sine_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let sr = sample_rate!(44100);
/// let a = sine_wave::<f64>(440.0, Duration::from_millis(100), sr, 1.0);
/// let b = sine_wave::<f64>(440.0, Duration::from_millis(100), sr, 1.0);
///
/// let db = psnr(&a, &b).unwrap();
/// assert!(db.is_infinite()); // identical signals → infinite PSNR
/// ```
#[inline]
pub fn psnr<T>(reference: &AudioSamples<T>, test: &AudioSamples<T>) -> AudioSampleResult<f64>
where
    T: StandardSample,
{
    if reference.num_channels() != test.num_channels()
        || reference.samples_per_channel() != test.samples_per_channel()
    {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "audio_signals",
            "Signals must have the same dimensions for PSNR",
        )));
    }

    // Reuse the existing MSE logic for the error term.
    let error = mse(reference, test)?;
    if error == 0.0 {
        return Ok(f64::INFINITY);
    }

    // Peak amplitude of the reference signal (across all channels).
    let ref_f = reference.as_f64();
    let peak = if let Some(mono) = ref_f.as_mono() {
        mono.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()))
    } else {
        let multi = ref_f.as_multi_channel().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Must be multi-channel audio",
            ))
        })?;
        multi.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()))
    };

    Ok(20.0 * (peak / error.sqrt()).log10())
}

/// Computes the mean segmental signal-to-noise ratio (segmental SNR) in decibels.
///
/// The signal and noise are split into consecutive, non-overlapping segments of
/// `segment_len` samples. For each segment the local SNR is computed as:
///
/// ```text
/// SNR_seg = 10 · log10( Σ signal² / Σ noise² )
/// ```
///
/// Each per-segment value is then clamped to the closed range `[-10, 35]` dB before
/// averaging. The clamp prevents near-silent segments (where the noise power dwarfs the
/// signal power, or vice versa) from dominating the mean with extreme values, which is the
/// standard treatment for segmental SNR in speech-quality literature. The final result is the
/// arithmetic mean of the clamped per-segment values.
///
/// A trailing partial segment shorter than `segment_len` is still evaluated.
///
/// For mono inputs the segmentation is applied to the single channel.\
/// For multi-channel inputs the per-sample energy is summed across all channels within each
/// segment, producing a single aggregate segmental SNR. No per-channel values are returned.
///
/// # Arguments
///
/// * `signal`\
///   The signal component.
///
/// * `noise`\
///   The noise component.
///
/// * `segment_len`\
///   The number of samples per segment (non-zero).
///
/// Both inputs must have identical channel counts and the same number of samples per
/// channel.
///
/// # Returns
///
/// The mean clamped per-segment SNR in decibels.
///
/// # Errors
///
/// Returns an error in the following cases:
///
/// * The two inputs have different channel counts.
/// * The two inputs have different numbers of samples per channel.
/// * The underlying sample layout is non-contiguous and cannot be viewed as a slice.
/// * The channel configuration of the two signals does not match (mono vs multi-channel).
///
/// # Behavioural Guarantees
///
/// * Every per-segment contribution lies in the closed interval `[-10, 35]` dB.
/// * A silent signal segment paired with non-zero noise clamps to the lower bound (`-10` dB).
/// * A non-zero signal segment paired with silent noise clamps to the upper bound (`35` dB).
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::comparison::segmental_snr;
/// use audio_samples::utils::generation::sine_wave;
/// use audio_samples::sample_rate;
/// use std::num::NonZeroUsize;
/// use std::time::Duration;
///
/// let sr = sample_rate!(44100);
/// let signal = sine_wave::<f64>(440.0, Duration::from_millis(100), sr, 1.0);
/// let noise  = sine_wave::<f64>(880.0, Duration::from_millis(100), sr, 0.01);
///
/// let db = segmental_snr(&signal, &noise, NonZeroUsize::new(256).unwrap()).unwrap();
/// assert!(db > 0.0);
/// ```
#[inline]
pub fn segmental_snr<T>(
    signal: &AudioSamples<T>,
    noise: &AudioSamples<T>,
    segment_len: core::num::NonZeroUsize,
) -> AudioSampleResult<f64>
where
    T: StandardSample,
{
    /// Lower clamp bound (dB) applied to each per-segment SNR.
    const SEG_SNR_MIN_DB: f64 = -10.0;
    /// Upper clamp bound (dB) applied to each per-segment SNR.
    const SEG_SNR_MAX_DB: f64 = 35.0;

    if signal.num_channels() != noise.num_channels()
        || signal.samples_per_channel() != noise.samples_per_channel()
    {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "audio_signals",
            "Signal and noise must have the same dimensions for segmental SNR",
        )));
    }

    let signal_f = signal.as_f64();
    let noise_f = noise.as_f64();

    // Flatten to per-sample energy series. For multi-channel input each "sample index"
    // sums the squared values across all channels (column-major energy).
    let (sig_sq, noise_sq): (Vec<f64>, Vec<f64>) = match (signal_f.as_mono(), noise_f.as_mono()) {
        (Some(s), Some(n)) => (
            s.iter().map(|&x| x * x).collect(),
            n.iter().map(|&x| x * x).collect(),
        ),
        (Some(_), None) | (None, Some(_)) => {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Signals must have the same channel configuration",
            )));
        }
        (None, None) => {
            let s_multi = signal_f.as_multi_channel().ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Must be multi-channel audio",
                ))
            })?;
            let n_multi = noise_f.as_multi_channel().ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Must be multi-channel audio",
                ))
            })?;

            let sig_sq: Vec<f64> = (0..s_multi.ncols().get())
                .map(|i| s_multi.column(i).iter().map(|&x| x * x).sum())
                .collect();
            let noise_sq: Vec<f64> = (0..n_multi.ncols().get())
                .map(|i| n_multi.column(i).iter().map(|&x| x * x).sum())
                .collect();
            (sig_sq, noise_sq)
        }
    };

    let seg = segment_len.get();
    let mut acc = 0.0;
    let mut count = 0usize;

    for (s_chunk, n_chunk) in sig_sq.chunks(seg).zip(noise_sq.chunks(seg)) {
        let sig_energy: f64 = s_chunk.iter().sum();
        let noise_energy: f64 = n_chunk.iter().sum();

        // Compute the raw per-segment SNR, handling degenerate (zero-power) segments by
        // saturating to the clamp bounds rather than producing NaN/±inf.
        let snr_db = if noise_energy == 0.0 {
            if sig_energy == 0.0 {
                // Both silent: treat as a neutral lower-bound contribution.
                SEG_SNR_MIN_DB
            } else {
                SEG_SNR_MAX_DB
            }
        } else if sig_energy == 0.0 {
            SEG_SNR_MIN_DB
        } else {
            10.0 * (sig_energy / noise_energy).log10()
        };

        acc += snr_db.clamp(SEG_SNR_MIN_DB, SEG_SNR_MAX_DB);
        count += 1;
    }

    Ok(acc / count as f64)
}

/// Aligns two audio signals by estimating a time offset that maximises correlation.
///
/// This function attempts to temporally align `signal` to `reference` by searching for a
/// non-negative sample offset that maximises cross-correlation between the two signals. The
/// returned signal is a shifted version of the input signal padded at the start, together
/// with the estimated offset in samples.
///
/// This is a heuristic alignment utility intended for coarse synchronisation, diagnostics,
/// and preprocessing. It does not guarantee globally optimal alignment, sub-sample accuracy,
/// or robustness under heavy noise, reverberation, or non-linear distortion.
///
/// # Channel Handling
///
/// * For mono signals, alignment is computed directly on the single channel.
/// * For multi-channel signals, all channels are averaged to a single derived signal before
///   alignment. The estimated offset is then applied uniformly to all channels.
///
/// No per-channel alignment is performed. Channel-specific delays are not detected.
///
/// # Search Behaviour
///
/// * Only non-negative offsets are considered. The signal is shifted forward in time only.
/// * The maximum offset searched is half of the shorter signal length.
/// * For each candidate offset, correlation is computed over the overlapping region only.
///
/// These limits are part of the public contract and constrain the class of alignments that
/// can be detected.
///
/// # Padding Semantics
///
/// When a positive offset is selected, the aligned signal is padded at the start using the
/// default value of the sample type. This is a structural padding operation rather than an
/// explicit silence model.
///
/// # Arguments
///
/// * `reference`\
///   The reference signal that defines the target alignment.
///
/// * `signal`\
///   The signal to be aligned to the reference.
///
/// Both signals must have identical channel counts.
///
/// # Returns
///
/// A tuple `(aligned_signal, offset_samples)` where:
///
/// * `aligned_signal` is a newly allocated signal shifted by the estimated offset.
/// * `offset_samples` is the number of samples by which the signal was shifted.
///
/// # Errors
///
/// Returns an error in the following cases:
///
/// * The two signals have different channel counts.
/// * The channel configuration of the two signals does not match (mono vs multi-channel).
/// * The underlying sample layout is non-contiguous and cannot be viewed as a slice.
/// * Internal array construction fails due to invalid shape assumptions.
///
/// # Behavioural Guarantees
///
/// * The returned offset is always greater than or equal to zero.
/// * If no offset improves correlation, the returned offset is zero and the original signal
///   is returned unchanged (cloned).
/// * The alignment decision is deterministic for fixed inputs.
/// * The function allocates a new signal when a non-zero offset is applied.
///
/// # Limitations
///
/// This function does not:
///
/// * Search negative offsets or bidirectional shifts.
/// * Perform sub-sample alignment or interpolation.
/// * Account for phase offsets, time stretching, or sample-rate mismatch.
/// * Preserve original sample padding semantics beyond default value initialisation.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::comparison::align_signals;
/// use audio_samples::AudioSamples;
///
/// fn example<T: audio_samples::traits::StandardSample>(
///     reference: AudioSamples<'static, T>,
///      signal: AudioSamples<'static, T>,
///  ) {
/// let (aligned, offset) = align_signals::<T>(&reference, &signal).unwrap();
///
/// println!("Estimated offset: {offset} samples");
/// assert!(offset >= 0);
///  }
/// ```
#[inline]
pub fn align_signals<T>(
    reference: &AudioSamples<'_, T>,
    signal: &AudioSamples<'_, T>,
) -> AudioSampleResult<(AudioSamples<'static, T>, usize)>
where
    T: StandardSample,
{
    if reference.num_channels() != signal.num_channels() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "audio_channels",
            "Signals must have the same number of channels for alignment",
        )));
    }

    let sample_rate = signal.sample_rate();
    let ref_f = reference.as_f64();
    let sig_f = signal.as_f64();

    // For simplicity, we'll work with the first channel for mono or the average for multi-channel
    let (ref_data, sig_data) = match (ref_f.as_mono(), sig_f.as_mono()) {
        (Some(ref_mono), Some(sig_mono)) => (ref_mono.to_vec(), sig_mono.to_vec()),
        (None, None) => {
            // Average across channels for multi-channel signals
            let ref_multi = ref_f.as_multi_channel().ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Must be multi-channel audio",
                ))
            })?;
            let sig_multi = sig_f.as_multi_channel().ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Must be multi-channel audio",
                ))
            })?;

            let ref_avg: Vec<f64> = (0..ref_multi.ncols().get())
                .map(|i| {
                    ref_multi.column(i).iter().fold(0.0, |acc, x| acc + *x)
                        / ref_multi.nrows().get() as f64
                })
                .collect();

            let sig_avg: Vec<f64> = (0..sig_multi.ncols().get())
                .map(|i| {
                    sig_multi.column(i).iter().fold(0.0, |acc, x| acc + *x)
                        / sig_multi.nrows().get() as f64
                })
                .collect();

            (ref_avg, sig_avg)
        }
        _ => {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Signals must have the same channel configuration",
            )));
        }
    };

    // Find the best alignment using cross-correlation
    let max_offset = ref_data.len().min(sig_data.len()) / 2;
    let mut best_offset = 0;
    let mut best_correlation = f64::NEG_INFINITY;

    for offset in 0..max_offset {
        let correlation = if offset < sig_data.len() {
            let end = (ref_data.len() - offset).min(sig_data.len() - offset);
            correlation_1d_slice(
                &ref_data[offset..offset + end],
                &sig_data[offset..offset + end],
            )?
        } else {
            0.0
        };

        if correlation > best_correlation {
            best_correlation = correlation;
            best_offset = offset;
        }
    }

    // Create aligned signal by shifting
    let aligned_signal = if best_offset > 0 {
        if let Some(mono) = signal.as_mono() {
            let mut aligned_data = vec![T::default(); best_offset];
            aligned_data.extend_from_slice(
                &mono.as_slice().ok_or_else(|| {
                    AudioSampleError::Layout(LayoutError::NonContiguous {
                        operation: "signal alignment".to_string(),
                        layout_type: "non-contiguous mono samples".to_string(),
                    })
                })?[..mono.len().get() - best_offset],
            );
            let aligned_array = Array1::from_vec(aligned_data);
            AudioSamples::new_mono(aligned_array, sample_rate)?
        } else {
            let multi = signal.as_multi_channel().ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Must be multi-channel audio",
                ))
            })?;
            let mut aligned_data = Vec::new();
            for i in 0..multi.nrows().get() {
                let mut row = vec![T::default(); best_offset];
                row.extend_from_slice(
                    &multi.row(i).as_slice().ok_or_else(|| {
                        AudioSampleError::Layout(LayoutError::NonContiguous {
                            operation: "signal alignment".to_string(),
                            layout_type: "non-contiguous multi-channel samples".to_string(),
                        })
                    })?[..multi.ncols().get() - best_offset],
                );
                aligned_data.push(row);
            }
            let aligned_array = ndarray::Array2::from_shape_vec(
                (aligned_data.len(), aligned_data[0].len()),
                aligned_data.into_iter().flatten().collect(),
            )
            .map_err(|e| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "array_shape",
                    format!("Array shape error: {e}"),
                ))
            })?;
            AudioSamples::new_multi_channel(aligned_array, sample_rate)?
        }
    } else {
        signal.clone().into_owned()
    };

    Ok((aligned_signal, best_offset))
}

/// Computes the RMS log-spectral distance (LSD) between two audio signals, in decibels.
///
/// The log-spectral distance compares the two signals in the (log) power-spectral domain. For
/// each frequency bin the difference of the log-power spectra is taken, and the root mean
/// square of those differences across bins yields the distance:
///
/// ```text
/// LSD = sqrt( mean_over_bins[ (10·log10(Pa + eps) - 10·log10(Pb + eps))² ] )
/// ```
///
/// where `Pa` and `Pb` are the power spectra (squared magnitude of the FFT) of the two signals
/// and `eps` is a small floor that keeps the logarithm finite for silent bins. The FFT length
/// equals the signal length, and the distance is averaged across all frequency bins of all
/// channels.
///
/// This function requires the `transforms` feature, since it relies on the crate's FFT.
///
/// # Arguments
///
/// * `a`\
///   The first audio signal.
///
/// * `b`\
///   The second audio signal.
///
/// Both signals must have identical channel counts and the same number of samples per
/// channel.
///
/// # Returns
///
/// The RMS log-spectral distance in decibels. Lower values indicate more similar spectra; a
/// value of `0.0` indicates identical power spectra.
///
/// # Errors
///
/// Returns an error in the following cases:
///
/// * The two signals have different channel counts.
/// * The two signals have different numbers of samples per channel.
/// * The underlying FFT computation fails.
///
/// # Behavioural Guarantees
///
/// * The returned value is greater than or equal to `0.0`.
/// * For identical signals the returned value is `0.0` up to floating-point rounding.
/// * The metric is symmetric in its arguments.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::comparison::log_spectral_distance;
/// use audio_samples::utils::generation::sine_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let sr = sample_rate!(44100);
/// let a = sine_wave::<f64>(440.0, Duration::from_millis(100), sr, 1.0);
/// let b = sine_wave::<f64>(440.0, Duration::from_millis(100), sr, 1.0);
///
/// let lsd = log_spectral_distance(&a, &b).unwrap();
/// assert!(lsd < 1e-6); // identical signals → ~0 distance
/// ```
#[cfg(feature = "transforms")]
#[inline]
pub fn log_spectral_distance<T>(
    a: &AudioSamples<T>,
    b: &AudioSamples<T>,
) -> AudioSampleResult<f64>
where
    T: StandardSample,
{
    use crate::operations::AudioTransforms;

    /// Power floor that keeps `log10` finite for silent frequency bins.
    const EPS: f64 = 1e-10;

    if a.num_channels() != b.num_channels() || a.samples_per_channel() != b.samples_per_channel() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "audio_signals",
            "Signals must have the same dimensions for log-spectral distance",
        )));
    }

    // FFT length is the full signal length; `fft` returns one row per channel.
    let n_fft = a.samples_per_channel();
    let spec_a = a.fft(n_fft)?;
    let spec_b = b.fft(n_fft)?;

    let mut sum_sq = 0.0;
    let mut bins = 0usize;
    for (ca, cb) in spec_a.rows().into_iter().zip(spec_b.rows().into_iter()) {
        for (za, zb) in ca.iter().zip(cb.iter()) {
            let la = 10.0 * (za.norm_sqr() + EPS).log10();
            let lb = 10.0 * (zb.norm_sqr() + EPS).log10();
            let d = la - lb;
            sum_sq += d * d;
            bins += 1;
        }
    }

    Ok((sum_sq / bins as f64).sqrt())
}

/// Computes the Pearson correlation coefficient for each channel independently.
///
/// This mirrors [`correlation`] but reports one value per channel instead of averaging across
/// channels. For mono input the returned vector contains a single element.
///
/// # Arguments
///
/// * `a`\
///   The first audio signal.
///
/// * `b`\
///   The second audio signal.
///
/// Both signals must have identical channel counts and the same number of samples per
/// channel.
///
/// # Returns
///
/// A vector of per-channel correlation coefficients, each in the range `[-1, 1]`, in channel
/// order. The vector length equals the channel count.
///
/// # Errors
///
/// Returns an error under the same conditions as [`correlation`]: mismatched dimensions,
/// mismatched channel configuration, or a non-contiguous sample layout.
///
/// # Behavioural Guarantees
///
/// * The returned vector has exactly one entry per channel.
/// * For mono input the returned vector has length one and its sole entry equals the value
///   returned by [`correlation`] on the same input.
/// * If a channel has zero variance, its correlation is defined as `0.0`.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::comparison::correlation_per_channel;
/// use audio_samples::utils::generation::sine_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let sr = sample_rate!(44100);
/// let a = sine_wave::<f64>(440.0, Duration::from_millis(100), sr, 1.0);
/// let b = sine_wave::<f64>(440.0, Duration::from_millis(100), sr, 1.0);
///
/// let per_channel = correlation_per_channel(&a, &b).unwrap();
/// assert_eq!(per_channel.len(), 1);
/// ```
#[inline]
pub fn correlation_per_channel<T>(
    a: &AudioSamples<T>,
    b: &AudioSamples<T>,
) -> AudioSampleResult<Vec<f64>>
where
    T: StandardSample,
{
    per_channel(a, b, "correlation", correlation_1d_slice)
}

/// Computes the mean squared error (MSE) for each channel independently.
///
/// This mirrors [`mse`] but reports one value per channel instead of averaging across
/// channels. For mono input the returned vector contains a single element.
///
/// # Arguments
///
/// * `a`\
///   The first audio signal.
///
/// * `b`\
///   The second audio signal.
///
/// Both signals must have identical channel counts and the same number of samples per
/// channel.
///
/// # Returns
///
/// A vector of per-channel MSE values, each non-negative, in channel order. The vector length
/// equals the channel count.
///
/// # Errors
///
/// Returns an error under the same conditions as [`mse`]: mismatched dimensions, mismatched
/// channel configuration, or a non-contiguous sample layout.
///
/// # Behavioural Guarantees
///
/// * The returned vector has exactly one entry per channel.
/// * For mono input the returned vector has length one and its sole entry equals the value
///   returned by [`mse`] on the same input.
/// * Every entry is greater than or equal to `0.0` for finite inputs.
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::comparison::mse_per_channel;
/// use audio_samples::utils::generation::sine_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let sr = sample_rate!(44100);
/// let a = sine_wave::<f64>(440.0, Duration::from_millis(100), sr, 1.0);
/// let b = sine_wave::<f64>(440.0, Duration::from_millis(100), sr, 1.0);
///
/// let per_channel = mse_per_channel(&a, &b).unwrap();
/// assert_eq!(per_channel.len(), 1);
/// ```
#[inline]
pub fn mse_per_channel<T>(a: &AudioSamples<T>, b: &AudioSamples<T>) -> AudioSampleResult<Vec<f64>>
where
    T: StandardSample,
{
    per_channel(a, b, "MSE", mse_1d_slice)
}

/// Computes the signal-to-noise ratio (SNR) in decibels for each channel independently.
///
/// This mirrors [`snr`] but reports one value per channel instead of aggregating power across
/// all channels. Each entry is `10 · log10(signal_power / noise_power)` for that channel. For
/// mono input the returned vector contains a single element.
///
/// # Arguments
///
/// * `signal`\
///   The signal component.
///
/// * `noise`\
///   The noise component.
///
/// Both inputs must have identical channel counts and the same number of samples per
/// channel.
///
/// # Returns
///
/// A vector of per-channel SNR values in decibels, in channel order. The vector length equals
/// the channel count. A channel with zero noise power yields [`f64::INFINITY`].
///
/// # Errors
///
/// Returns an error under the same conditions as [`snr`]: mismatched dimensions, mismatched
/// channel configuration, or a non-contiguous sample layout.
///
/// # Behavioural Guarantees
///
/// * The returned vector has exactly one entry per channel.
/// * For mono input the returned vector has length one and its sole entry equals the value
///   returned by [`snr`] on the same input.
/// * A channel with exactly zero noise power yields [`f64::INFINITY`].
///
/// # Examples
///
/// ```rust
/// use audio_samples::utils::comparison::snr_per_channel;
/// use audio_samples::utils::generation::sine_wave;
/// use audio_samples::sample_rate;
/// use std::time::Duration;
///
/// let sr = sample_rate!(44100);
/// let signal = sine_wave::<f64>(440.0, Duration::from_millis(100), sr, 1.0);
/// let noise  = sine_wave::<f64>(880.0, Duration::from_millis(100), sr, 0.01);
///
/// let per_channel = snr_per_channel(&signal, &noise).unwrap();
/// assert_eq!(per_channel.len(), 1);
/// ```
#[inline]
pub fn snr_per_channel<T>(
    signal: &AudioSamples<T>,
    noise: &AudioSamples<T>,
) -> AudioSampleResult<Vec<f64>>
where
    T: StandardSample,
{
    per_channel(signal, noise, "SNR", |sig, noise| {
        let n = sig.len();
        let signal_power = sig.iter().map(|&x| x * x).sum::<f64>() / n as f64;
        let noise_power = noise.iter().map(|&x| x * x).sum::<f64>() / n as f64;
        if noise_power == 0.0 {
            Ok(f64::INFINITY)
        } else {
            Ok(10.0 * (signal_power / noise_power).log10())
        }
    })
}

/// Shared driver for the `*_per_channel` metrics.
///
/// Validates that the two signals share dimensions and channel configuration, then applies
/// `metric` to each channel's contiguous sample slice, collecting one value per channel.
fn per_channel<T, F>(
    a: &AudioSamples<T>,
    b: &AudioSamples<T>,
    metric_name: &str,
    metric: F,
) -> AudioSampleResult<Vec<f64>>
where
    T: StandardSample,
    F: Fn(&[f64], &[f64]) -> AudioSampleResult<f64>,
{
    if a.num_channels() != b.num_channels() || a.samples_per_channel() != b.samples_per_channel() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "audio_signals",
            format!("Signals must have the same dimensions for {metric_name}"),
        )));
    }

    let a_f = a.as_f64();
    let b_f = b.as_f64();

    match (a_f.as_mono(), b_f.as_mono()) {
        (Some(a_mono), Some(b_mono)) => {
            let av = a_mono.as_slice().ok_or_else(|| {
                AudioSampleError::Layout(LayoutError::NonContiguous {
                    operation: "signal processing".to_string(),
                    layout_type: "non-contiguous mono samples".to_string(),
                })
            })?;
            let bv = b_mono.as_slice().ok_or_else(|| {
                AudioSampleError::Layout(LayoutError::NonContiguous {
                    operation: "signal processing".to_string(),
                    layout_type: "non-contiguous mono samples".to_string(),
                })
            })?;
            Ok(vec![metric(av, bv)?])
        }
        (Some(_), None) | (None, Some(_)) => {
            Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Signals must have the same channel configuration",
            )))
        }
        (None, None) => {
            let a_multi = a_f.as_multi_channel().ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Must be multi-channel audio",
                ))
            })?;
            let b_multi = b_f.as_multi_channel().ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Must be multi-channel audio",
                ))
            })?;

            let mut results = Vec::with_capacity(a_multi.nrows().get());
            for i in 0..a_multi.nrows().get() {
                let a_channel = a_multi.row(i);
                let b_channel = b_multi.row(i);
                results.push(metric(
                    a_channel.as_slice().ok_or_else(|| {
                        AudioSampleError::Layout(LayoutError::NonContiguous {
                            operation: "signal processing".to_string(),
                            layout_type: "non-contiguous multi-channel samples".to_string(),
                        })
                    })?,
                    b_channel.as_slice().ok_or_else(|| {
                        AudioSampleError::Layout(LayoutError::NonContiguous {
                            operation: "signal processing".to_string(),
                            layout_type: "non-contiguous multi-channel samples".to_string(),
                        })
                    })?,
                )?);
            }
            Ok(results)
        }
    }
}

// Helper functions for 1D correlation and MSE
fn correlation_1d(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> AudioSampleResult<f64> {
    correlation_1d_slice(
        a.as_slice().ok_or_else(|| {
            AudioSampleError::Layout(LayoutError::NonContiguous {
                operation: "correlation calculation".to_string(),
                layout_type: "non-contiguous mono samples".to_string(),
            })
        })?,
        b.as_slice().ok_or_else(|| {
            AudioSampleError::Layout(LayoutError::NonContiguous {
                operation: "correlation calculation".to_string(),
                layout_type: "non-contiguous mono samples".to_string(),
            })
        })?,
    )
}

fn correlation_1d_slice(a: &[f64], b: &[f64]) -> AudioSampleResult<f64> {
    if a.len() != b.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "array_length",
            "Arrays must have the same length for correlation",
        )));
    }

    let n = a.len();
    let mean_a = a.iter().fold(0.0, |acc, x| acc + *x) / n as f64;
    let mean_b = b.iter().fold(0.0, |acc, x| acc + *x) / n as f64;

    let mut num = 0.0;
    let mut den_a = 0.0;
    let mut den_b = 0.0;

    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        num += dx * dy;
        den_a += dx * dx;
        den_b += dy * dy;
    }

    let denominator = (den_a * den_b).sqrt();
    if denominator == 0.0 {
        Ok(0.0)
    } else {
        Ok(num / denominator)
    }
}

fn mse_1d(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> AudioSampleResult<f64> {
    if a.len() != b.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "array_length",
            "Arrays must have the same length for MSE",
        )));
    }

    let n = a.len();
    let sum_squared_diff: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .fold(0.0, |acc, val| acc + val);

    Ok(sum_squared_diff / n as f64)
}

fn mse_1d_slice(a: &[f64], b: &[f64]) -> AudioSampleResult<f64> {
    if a.len() != b.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "array_length",
            "Arrays must have the same length for MSE",
        )));
    }
    let n = a.len();

    let sum_squared_diff: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .fold(0.0, |acc, val| acc + val)
        / n as f64;

    Ok(sum_squared_diff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample_rate;
    use approx_eq::assert_approx_eq;
    use non_empty_slice::non_empty_vec;

    #[test]
    fn test_correlation_identical_signals() {
        let data: non_empty_slice::NonEmptyVec<f64> = non_empty_vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let audio1: AudioSamples<'static, f64> =
            AudioSamples::from_mono_vec::<f64>(data.clone(), sample_rate!(44100));
        let audio2: AudioSamples<'static, f64> =
            AudioSamples::from_mono_vec::<f64>(data, sample_rate!(44100));

        let corr: f64 = correlation(&audio1, &audio2).unwrap();
        assert_approx_eq!(corr, 1.0, 1e-10);
    }

    #[test]
    fn test_correlation_opposite_signals() {
        let audio1: AudioSamples<'static, f64> = AudioSamples::from_mono_vec::<f64>(
            non_empty_vec![1.0f64, 2.0, 3.0, 4.0, 5.0],
            sample_rate!(44100),
        );
        let audio2 = AudioSamples::from_mono_vec::<f64>(
            non_empty_vec![-1.0f64, -2.0, -3.0, -4.0, -5.0],
            sample_rate!(44100),
        );

        let corr: f64 = correlation(&audio1, &audio2).unwrap();
        assert_approx_eq!(corr, -1.0, 1e-10);
    }

    #[test]
    fn test_mse_identical_signals() {
        let audio1: AudioSamples<'static, f64> = AudioSamples::from_mono_vec::<f64>(
            non_empty_vec![1.0f64, 2.0, 3.0, 4.0, 5.0],
            sample_rate!(44100),
        );
        let audio2 = AudioSamples::from_mono_vec::<f64>(
            non_empty_vec![1.0f64, 2.0, 3.0, 4.0, 5.0],
            sample_rate!(44100),
        );

        let mse_val = mse(&audio1, &audio2).unwrap();
        assert_approx_eq!(mse_val, 0.0_f64, 1e-10);
    }

    #[test]
    fn test_snr_calculation() {
        let signal: AudioSamples<'static, f64> = AudioSamples::from_mono_vec::<f64>(
            non_empty_vec![1.0f64, 2.0, 3.0, 4.0, 5.0],
            sample_rate!(44100),
        );
        let noise = AudioSamples::from_mono_vec::<f64>(
            non_empty_vec![0.1f64, 0.2, 0.1, 0.2, 0.1],
            sample_rate!(44100),
        );

        let snr_val: f64 = snr(&signal, &noise).unwrap();
        assert!(snr_val > 0.0_f64); // Signal should have higher power than noise
    }

    #[test]
    fn test_align_signals_no_offset() {
        let data = non_empty_vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let reference = AudioSamples::from_mono_vec::<f64>(data.clone(), sample_rate!(44100));
        let signal = AudioSamples::from_mono_vec::<f64>(data, sample_rate!(44100));

        let (aligned, offset) = align_signals::<f64>(&reference, &signal).unwrap();
        assert_eq!(offset, 0);
        assert_eq!(aligned.samples_per_channel(), signal.samples_per_channel());
    }

    #[test]
    fn test_psnr_identical_signals_is_infinite() {
        let data = non_empty_vec![0.1f64, -0.5, 0.7, -0.2, 1.0];
        let reference: AudioSamples<'static, f64> =
            AudioSamples::from_mono_vec::<f64>(data.clone(), sample_rate!(44100));
        let test = AudioSamples::from_mono_vec::<f64>(data, sample_rate!(44100));

        let db = psnr(&reference, &test).unwrap();
        assert!(db.is_infinite() && db > 0.0);
    }

    #[test]
    fn test_psnr_known_noise() {
        // Reference peak = 1.0; constant error of 0.1 → MSE = 0.01,
        // PSNR = 20·log10(1.0 / sqrt(0.01)) = 20·log10(10) = 20 dB.
        let reference: AudioSamples<'static, f64> = AudioSamples::from_mono_vec::<f64>(
            non_empty_vec![1.0f64, 0.5, -0.5, 0.25, -1.0],
            sample_rate!(44100),
        );
        let test = AudioSamples::from_mono_vec::<f64>(
            non_empty_vec![1.1f64, 0.6, -0.4, 0.35, -0.9],
            sample_rate!(44100),
        );

        let db = psnr(&reference, &test).unwrap();
        assert_approx_eq!(db, 20.0, 1e-9);
    }

    #[test]
    fn test_segmental_snr_scaled_noise() {
        // Signal energy per segment is 100x noise energy → 10·log10(100) = 20 dB,
        // within the clamp range [-10, 35].
        let signal: AudioSamples<'static, f64> = AudioSamples::from_mono_vec::<f64>(
            non_empty_vec![1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            sample_rate!(44100),
        );
        let noise = AudioSamples::from_mono_vec::<f64>(
            non_empty_vec![0.1f64, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            sample_rate!(44100),
        );

        let db = segmental_snr(&signal, &noise, core::num::NonZeroUsize::new(4).unwrap()).unwrap();
        assert_approx_eq!(db, 20.0, 1e-9);
    }

    #[test]
    fn test_segmental_snr_clamp_engages_for_silent_segment() {
        // Segment 0: signal energy 100x noise → 20 dB.
        // Segment 1: signal silent, noise non-zero → clamps to lower bound -10 dB.
        // Mean = (20 + -10) / 2 = 5 dB.
        let signal: AudioSamples<'static, f64> = AudioSamples::from_mono_vec::<f64>(
            non_empty_vec![1.0f64, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            sample_rate!(44100),
        );
        let noise = AudioSamples::from_mono_vec::<f64>(
            non_empty_vec![0.1f64, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            sample_rate!(44100),
        );

        let db = segmental_snr(&signal, &noise, core::num::NonZeroUsize::new(4).unwrap()).unwrap();
        assert_approx_eq!(db, 5.0, 1e-9);
    }

    #[test]
    fn test_per_channel_mono_equals_aggregate() {
        let a: AudioSamples<'static, f64> = AudioSamples::from_mono_vec::<f64>(
            non_empty_vec![1.0f64, 2.0, 3.0, 4.0, 5.0],
            sample_rate!(44100),
        );
        let b = AudioSamples::from_mono_vec::<f64>(
            non_empty_vec![1.0f64, 2.5, 2.5, 4.5, 5.0],
            sample_rate!(44100),
        );

        let corr_pc = correlation_per_channel(&a, &b).unwrap();
        let mse_pc = mse_per_channel(&a, &b).unwrap();
        let snr_pc = snr_per_channel(&a, &b).unwrap();

        assert_eq!(corr_pc.len(), 1);
        assert_eq!(mse_pc.len(), 1);
        assert_eq!(snr_pc.len(), 1);

        assert_approx_eq!(corr_pc[0], correlation(&a, &b).unwrap(), 1e-12);
        assert_approx_eq!(mse_pc[0], mse(&a, &b).unwrap(), 1e-12);
        assert_approx_eq!(snr_pc[0], snr(&a, &b).unwrap(), 1e-12);
    }

    #[test]
    fn test_per_channel_distinct_values_for_two_channels() {
        // Channel 0 of `a` and `b` are identical (corr = 1); channel 1 differs.
        let a = AudioSamples::new_multi_channel(
            ndarray::array![[1.0f64, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
            sample_rate!(44100),
        )
        .unwrap();
        let b = AudioSamples::new_multi_channel(
            ndarray::array![[1.0f64, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
            sample_rate!(44100),
        )
        .unwrap();

        let mse_pc = mse_per_channel(&a, &b).unwrap();
        let corr_pc = correlation_per_channel(&a, &b).unwrap();

        assert_eq!(mse_pc.len(), 2);
        assert_eq!(corr_pc.len(), 2);

        // Channel 0 identical, channel 1 differs → distinct values.
        assert_approx_eq!(mse_pc[0], 0.0, 1e-12);
        assert!(mse_pc[1] > 0.0);
        assert!((mse_pc[0] - mse_pc[1]).abs() > 1e-6);

        assert_approx_eq!(corr_pc[0], 1.0, 1e-12);
        assert_approx_eq!(corr_pc[1], -1.0, 1e-12);
    }

    #[cfg(feature = "transforms")]
    #[test]
    fn test_log_spectral_distance_identical_and_different() {
        use crate::utils::generation::sine_wave;
        use std::time::Duration;

        let sr = sample_rate!(44100);
        let a = sine_wave::<f64>(440.0, Duration::from_millis(50), sr, 1.0);
        let a2 = sine_wave::<f64>(440.0, Duration::from_millis(50), sr, 1.0);
        let c = sine_wave::<f64>(880.0, Duration::from_millis(50), sr, 1.0);

        let lsd_identical = log_spectral_distance(&a, &a2).unwrap();
        assert!(lsd_identical < 1e-6, "identical → ~0, got {lsd_identical}");

        let lsd_diff = log_spectral_distance(&a, &c).unwrap();
        assert!(lsd_diff > 0.0, "different tones → > 0, got {lsd_diff}");
    }
}
