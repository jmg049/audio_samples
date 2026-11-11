//! Signal processing operations for audio samples.
//!
//! ## What
//!
//! This module provides the [`AudioProcessing`] trait implementation for
//! [`AudioSamples`]: normalization, gain (`scale`), DC removal, clipping,
//! windowing, FIR filtering by convolution (`apply_filter`), μ-law
//! companding, first-order low/high/band-pass filters and (with the
//! `resampling` feature) sample-rate conversion. Each transforming operation
//! has an in-place primitive (`op_in_place(&mut self, …)`) and a non-mutating
//! twin (`op(&self, …)`, provided by the trait default) that clones then
//! mutates - enabling both zero-allocation in-place pipelines and fluent
//! method chaining.
//!
//! ## Why
//!
//! Audio signals routinely need amplitude adjustment, spectral shaping and
//! offset removal before playback or further analysis. A single trait keeps
//! the API discoverable and composable across all supported sample types.
//!
//! ## How
//!
//! Import [`AudioProcessing`] and call methods directly on an [`AudioSamples`]
//! value. The infallible [`AudioProcessing::scale`] returns `Self` directly,
//! so it can be chained directly before fallible calls which return
//! [`AudioSampleResult`]. Gains, fades and peak normalisation are computed in
//! `f64`; for integer sample types they act on the normalised `[-1, 1]` scale
//! and the result is rounded and saturated back to the integer range.
//! Window and FIR coefficients are given as `f64` for every sample type
//! ([`AudioProcessing::apply_window_f64`], [`AudioProcessing::apply_filter_f64`]),
//! and every operation returns audio in the input's sample type.
//!
//! ### Convolution: full and valid
//!
//! An FIR filter with taps `h[0..M]` computes the convolution sum
//!
//! `y[n] = Σ_{k=0}^{M−1} h[k] · x[n − k]`.
//!
//! For an input of `N` samples the *full* convolution (all `n` where the
//! signal and the taps overlap at all, treating samples outside the input as
//! zero) has `N + M − 1` samples; this is what `AudioTransforms::convolve`
//! returns (feature `transforms`). The *valid* part keeps only the
//! `N − M + 1` outputs for which all `M` taps fall inside the input; this is
//! what [`AudioProcessing::apply_filter`] returns, so its output is shorter
//! than its input. Valid output index `i` corresponds to `n = i + M − 1` in
//! the formula above. Above 160 taps
//! `apply_filter` switches from direct summation (O(N·M)) to FFT convolution
//! (O(N log N)); both give the same result up to rounding.
//!
//! ```
//! use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
//! use ndarray::array;
//! use non_empty_slice::NonEmptySlice;
//!
//! // N = 5 samples, M = 2 taps: valid output has N - M + 1 = 4 samples.
//! let x = AudioSamples::new_mono(array![1.0f64, 2.0, 3.0, 4.0, 5.0], sample_rate!(8000)).unwrap();
//! let h = [1.0f64, 0.5];
//! let y = x.apply_filter(NonEmptySlice::new(&h).unwrap()).unwrap();
//! assert_eq!(y.samples_per_channel().get(), 4);
//! // y[0] = h[0]·x[1] + h[1]·x[0] = 2 + 0.5
//! assert_eq!(y[0], 2.5);
//! ```
//!
//! ### FIR group delay and linear phase
//!
//! An FIR filter has no feedback, so it is always stable. It has *linear
//! phase* (every frequency delayed by the same number of samples) only when
//! its taps are symmetric, `h[k] = h[M − 1 − k]`, or antisymmetric,
//! `h[k] = −h[M − 1 − k]`. The constant group delay is then `(M − 1)/2`
//! samples. An arbitrary FIR (for example `[1.0, 0.5]`) has a group delay
//! that varies with frequency. `apply_filter` does not compensate the delay.
//!
//! ### First-order low-, high- and band-pass filters
//!
//! [`low_pass_filter`](AudioProcessing::low_pass_filter),
//! [`high_pass_filter`](AudioProcessing::high_pass_filter) and
//! [`band_pass_filter`](AudioProcessing::band_pass_filter) are first-order
//! Butterworth designs made with the bilinear transform (the order-1 case of
//! the Butterworth filters of `AudioIirFiltering`). They are not ideal
//! "brick-wall" filters,
//! which are not realisable. With `fs` the sample rate, `f` the frequency in
//! Hz and `Ω = tan(π f / fs)` the pre-warped frequency, the magnitude
//! responses are
//!
//! - low-pass, cutoff `fc`, `Ωc = tan(π fc / fs)`:
//!   `|H(f)|² = 1 / (1 + (Ω/Ωc)²)`
//! - high-pass: `|H(f)|² = 1 / (1 + (Ωc/Ω)²)`
//! - band-pass, edges `fl < fh` (`Ωl`, `Ωh` pre-warped):
//!   `|H(f)|² = 1 / (1 + ((Ω² − Ωl·Ωh) / (Ω·(Ωh − Ωl)))²)`
//!
//! Each is −3 dB (`|H|² = 1/2`) at its cutoff or band edges, and rolls off at
//! 6 dB/octave well beyond them, so a band-pass still passes noticeable
//! energy just outside its band (about −7 to −8 dB one octave out). The low-pass
//! has a zero at the Nyquist frequency and the high-pass a zero at DC. For
//! steeper slopes use the `butterworth_*`, `chebyshev_*` or `elliptic`
//! designs of [`AudioIirFiltering`] with a higher order.
//!
//! ### DC removal
//!
//! [`remove_dc_offset`](AudioProcessing::remove_dc_offset) subtracts the
//! mean, `x'[n] = x[n] − x̄`. This removes exactly the 0 Hz component of the
//! buffer and changes no other DFT bin of that buffer. An offset that drifts
//! over time is better removed with a high-pass filter at a low cutoff
//! (about 5-20 Hz): its zero at DC blocks a constant exactly, and the cutoff
//! sets how fast a changing offset is followed (time constant about
//! `1/(2π fc)` seconds).
//!
//! ```
//! use audio_samples::{AudioSamples, AudioProcessing, sample_rate};
//! use ndarray::array;
//!
//! let audio = AudioSamples::new_mono(array![2.0f32, -3.0, 1.5], sample_rate!(44100))
//!     .unwrap()
//!     .scale(0.5)              // infallible - returns Self
//!     .clip(-1.0, 1.0)        // fallible - returns Result<Self>
//!     .unwrap();
//! assert!(audio[0] <= 1.0);
//! ```

use crate::operations::traits::AudioIirFiltering;
#[cfg(feature = "resampling")]
use crate::operations::types::ResamplingQuality;
use crate::operations::types::{IirFilterDesign, NormalizationConfig, NormalizationMethod};
use crate::repr::AudioData;
use crate::traits::{apply_gain_to_sample, raw_f64_to_sample};
use std::num::NonZeroUsize;

#[cfg(feature = "resampling")]
use crate::repr::SampleRate;

use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, ConvertTo, LayoutError, ParameterError,
    StandardSample,
    operations::traits::{AudioProcessing, AudioStatistics},
};

use ndarray::{Array2, Axis};
use non_empty_slice::NonEmptySlice;

/// Valid-region FIR convolution over flat slices, matching [`AudioProcessing::apply_filter`]'s
/// convention: `output[i] = Σ_{j=0}^{M-1} coeffs[j] · input[i + M - 1 - j]` for
/// `i in 0..output_len`, with `M = coeffs.len()` (true convolution, the "valid"
/// part; the taps are reversed, so this is not the correlation
/// `Σ_j input[i + j] · coeffs[j]`).
///
/// Working on contiguous `&[T]` slices (rather than enum-dispatched element
/// indexing) lets the compiler autovectorize the inner dot product.
///
/// `output_len` must equal `input.len() - coeffs.len() + 1`.
#[inline]
fn fir_valid_convolve<T>(input: &[T], coeffs: &[f64], output_len: usize) -> Vec<T>
where
    T: StandardSample,
{
    // True convolution, `y[i] = Σ_j h[j] · x[i + M - 1 - j]` (the "valid"
    // part): dot each window with the reversed taps.
    // Accumulates in f64 using the audio-aware sample conversion, so integer
    // samples are interpreted on the normalised [-1, 1] scale and cannot
    // overflow; the result is converted back with rounding and saturation.
    let h_rev: Vec<f64> = coeffs.iter().rev().copied().collect();
    let x: Vec<f64> = input.iter().map(|&s| -> f64 { s.convert_to() }).collect();
    let filter_len = h_rev.len();
    let mut out = Vec::with_capacity(output_len);
    // Blocks of eight outputs: each tap is broadcast over a contiguous
    // eight-sample input window, so the inner loop vectorises across outputs
    // and every tap is loaded once per block.
    const B: usize = 8;
    let blocks = output_len / B;
    for blk in 0..blocks {
        let i = blk * B;
        let mut acc = [0.0f64; B];
        for (j, &hj) in h_rev.iter().enumerate() {
            let w = &x[i + j..i + j + B];
            for k in 0..B {
                acc[k] += w[k] * hj;
            }
        }
        out.extend(acc.iter().map(|&v| -> T { v.convert_to() }));
    }
    for i in blocks * B..output_len {
        out.push(dot(&x[i..i + filter_len], &h_rev).convert_to());
    }
    out
}

/// Dot product with eight independent accumulators, so the reduction can be
/// vectorised (a single sequential f64 sum cannot be re-associated by the
/// compiler).
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    const L: usize = 8;
    let mut acc = [0.0f64; L];
    let (ca, ra) = a.as_chunks::<L>();
    let (cb, rb) = b.as_chunks::<L>();
    for (xa, xb) in ca.iter().zip(cb) {
        for k in 0..L {
            acc[k] += xa[k] * xb[k];
        }
    }
    let tail: f64 = ra.iter().zip(rb).map(|(p, q)| p * q).sum();
    acc.iter().sum::<f64>() + tail
}

/// Tap count at or above which FFT convolution beats direct convolution for
/// (measured crossover ≈ 150 taps on a 1 s, 44.1 kHz signal)
/// whole-signal FIR filtering. Below this the O(N·M) direct loop wins because
/// the FFT's constant overhead dominates; above it the O(N·log N) FFT path pulls
/// far ahead (e.g. ~120x at 4096 taps on a 1 s signal).
#[cfg(feature = "transforms")]
pub(crate) const FIR_FFT_MIN_TAPS: usize = 160;

/// Choose the fastest valid-region FIR convolution for the given filter length.
///
/// With `transforms` enabled, long filters route through FFT convolution
/// (`spectrograms::fft_convolve`); short filters and builds without `transforms`
/// use the direct slice loop. Both produce identical results up to floating-point
/// rounding and implement `apply_filter`'s valid-region convolution.
#[cfg(feature = "transforms")]
fn fir_valid_convolve_dispatch<T>(
    input: &[T],
    coeffs: &[f64],
    output_len: usize,
) -> AudioSampleResult<Vec<T>>
where
    T: StandardSample,
{
    if coeffs.len() >= FIR_FFT_MIN_TAPS {
        fir_valid_convolve_fft(input, coeffs, output_len)
    } else {
        Ok(fir_valid_convolve(input, coeffs, output_len))
    }
}

/// Direct-only dispatch used when the `transforms` feature (and thus the FFT
/// backend) is not available.
#[cfg(not(feature = "transforms"))]
fn fir_valid_convolve_dispatch<T>(
    input: &[T],
    coeffs: &[f64],
    output_len: usize,
) -> AudioSampleResult<Vec<T>>
where
    T: StandardSample,
{
    Ok(fir_valid_convolve(input, coeffs, output_len))
}

/// Valid-region FIR convolution via FFT, matching the direct path:
/// `output[i] = Σ_j coeffs[j] · input[i + M - 1 - j]` (the "valid" part).
///
/// Computation is in `f64` for headroom; results match the direct loop to
/// floating-point tolerance.
#[cfg(feature = "transforms")]
fn fir_valid_convolve_fft<T>(
    input: &[T],
    coeffs: &[f64],
    output_len: usize,
) -> AudioSampleResult<Vec<T>>
where
    T: StandardSample,
{
    use non_empty_slice::NonEmptySlice;

    let x: Vec<f64> = input.iter().map(|&s| -> f64 { s.convert_to() }).collect();

    let x_ne = NonEmptySlice::new(x.as_slice()).ok_or_else(|| {
        AudioSampleError::Parameter(ParameterError::invalid_value("audio", "empty signal"))
    })?;
    let h_ne = NonEmptySlice::new(coeffs).ok_or_else(|| {
        AudioSampleError::Parameter(ParameterError::invalid_value("filter", "empty filter"))
    })?;

    // Full linear convolution, length x.len() + coeffs.len() - 1.
    let full = spectrograms::fft_convolve(x_ne, h_ne)?.into_vec();

    // Valid region starts at coeffs.len() - 1 (see module note above).
    let start = coeffs.len() - 1;
    let out: Vec<T> = full[start..start + output_len]
        .iter()
        .map(|&v| -> T { v.convert_to() })
        .collect();
    Ok(out)
}

/// Checks that a window has one coefficient per sample (per channel).
fn check_window_len<T>(audio: &AudioSamples<'_, T>, window_len: usize) -> AudioSampleResult<()>
where
    T: StandardSample,
{
    let num_samples = audio.samples_per_channel().get();
    if window_len == num_samples {
        return Ok(());
    }
    Err(AudioSampleError::Layout(LayoutError::dimension_mismatch(
        format!("window length ({window_len})"),
        format!("audio length ({num_samples})"),
        "apply_window",
    )))
}

/// Valid-mode FIR convolution of every channel with the `f64` taps `coeffs`,
/// replacing the audio buffer (see [`AudioProcessing::apply_filter_f64`]).
fn apply_fir_in_place<T>(audio: &mut AudioSamples<'_, T>, coeffs: &[f64]) -> AudioSampleResult<()>
where
    T: StandardSample,
{
    let filter_len = coeffs.len();

    match audio.data_mut() {
        AudioData::Mono(arr) => {
            if arr.len().get() < filter_len {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_length",
                    format!(
                        "Audio length ({}) must be at least the filter length ({filter_len}): apply_filter returns the 'valid' convolution of length N - M + 1",
                        arr.len().get()
                    ),
                )));
            }

            let output_len = arr.len().get() - filter_len + 1;

            // Operate on a flat contiguous slice. The previous version indexed
            // `arr[i + j]`, which dispatched through the `MonoData` enum on every
            // single sample access; a raw slice lets the inner dot product
            // autovectorize and removes that per-element overhead entirely.
            let owned_fallback;
            let input: &[T] = match arr.as_slice() {
                Some(s) => s,
                None => {
                    owned_fallback = arr.to_vec();
                    &owned_fallback
                }
            };

            let output = fir_valid_convolve_dispatch(input, coeffs, output_len)?;
            *arr = ndarray::Array1::from_vec(output).try_into()?;
        }
        AudioData::Multi(arr) => {
            if arr.ncols().get() < filter_len {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_length",
                    format!(
                        "Audio length ({}) must be at least the filter length ({filter_len}): apply_filter returns the 'valid' convolution of length N - M + 1",
                        arr.ncols().get()
                    ),
                )));
            }

            let output_len = arr.ncols().get() - filter_len + 1;
            let num_channels = arr.nrows().get();

            let mut output = Array2::zeros((num_channels, output_len));

            for (ch, mut output_channel) in output.axis_iter_mut(Axis(0)).enumerate() {
                let row = arr.index_axis(Axis(0), ch);
                let owned_fallback;
                let input: &[T] = match row.as_slice() {
                    Some(s) => s,
                    None => {
                        owned_fallback = row.to_vec();
                        &owned_fallback
                    }
                };
                let filtered = fir_valid_convolve_dispatch(input, coeffs, output_len)?;
                for (dst, src) in output_channel.iter_mut().zip(filtered) {
                    *dst = src;
                }
            }

            *arr = output.try_into()?;
        }
    }
    Ok(())
}

/// Subtracts `offset` (on the raw scale of the sample values) from every
/// sample. The difference is computed in `f64`; integer results are rounded to
/// nearest and saturated (see [`raw_f64_to_sample`]), float results are cast.
#[inline]
fn subtract_raw_offset<T>(audio: &mut AudioSamples<'_, T>, offset: f64)
where
    T: StandardSample,
{
    let map = |x: T| -> T {
        let x: f64 = x.cast_into();
        raw_f64_to_sample(x - offset)
    };
    match audio.data_mut() {
        AudioData::Mono(arr) => arr.mapv_inplace(map),
        AudioData::Multi(arr) => arr.mapv_inplace(map),
    }
}

impl<T> AudioProcessing for AudioSamples<'_, T>
where
    T: StandardSample,
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
    /// # Returns
    /// The normalized audio samples.
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
    fn normalize_in_place(
        &mut self,
        config: NormalizationConfig<Self::Sample>,
    ) -> AudioSampleResult<()> {
        match config.method {
            NormalizationMethod::MinMax => {
                let min = config.min.unwrap_or(T::MIN);
                let max = config.max.unwrap_or(T::MAX);
                // Validate input range
                if min >= max {
                    return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                        "normalization_range",
                        format!("min ({min:?}) >= max ({max:?})"),
                        format!("{min:?}"),
                        format!("{max:?}"),
                        "min value must be less than max value for normalization",
                    )));
                }
                // Min-Max normalization: scale to [min, max] range
                let current_min = self.min_sample();
                let current_max = self.max_sample();

                // The affine map is evaluated in f64 so that integer sample
                // types neither overflow (e.g. `i16::MAX - i16::MIN`) nor
                // truncate the scale factor by integer division.
                let min_f: f64 = min.cast_into();
                let max_f: f64 = max.cast_into();
                let current_min: f64 = current_min.cast_into();
                let current_max: f64 = current_max.cast_into();

                // Avoid division by zero
                if current_min == current_max {
                    // All values are the same, set to middle of target range
                    let middle: T = raw_f64_to_sample(min_f + (max_f - min_f) / 2.0);
                    match self.data_mut() {
                        AudioData::Mono(arr) => arr.fill(middle),
                        AudioData::Multi(arr) => arr.fill(middle),
                    }
                    return Ok(());
                }

                let scale_factor = (max_f - min_f) / (current_max - current_min);
                let map = |x: T| -> T {
                    let x: f64 = x.cast_into();
                    raw_f64_to_sample(min_f + (x - current_min) * scale_factor)
                };

                match self.data_mut() {
                    AudioData::Mono(arr) => arr.mapv_inplace(map),
                    AudioData::Multi(arr) => arr.mapv_inplace(map),
                }
            }

            NormalizationMethod::Peak => {
                let target = config.target.unwrap_or({
                    // Default target is the maximum representable value for the sample type
                    Self::Sample::MAX
                });
                // Peak normalization: scale by peak value to target level
                let peak: Self::Sample = self.peak();
                if peak == Self::Sample::zero() {
                    return Ok(()); // No scaling needed for zero signal
                }

                let target_f64: f64 = target.convert_to();
                let target_peak = target_f64.abs();
                let peak_f64: f64 = peak.convert_to();
                let scale_factor = target_peak / peak_f64;

                match self.data_mut() {
                    AudioData::Mono(arr) => {
                        arr.mapv_inplace(|x| apply_gain_to_sample(x, scale_factor));
                    }
                    AudioData::Multi(arr) => {
                        arr.mapv_inplace(|x| apply_gain_to_sample(x, scale_factor));
                    }
                }
            }

            NormalizationMethod::Mean => {
                // Mean normalization: subtract the mean (of the raw sample
                // values) to center around zero.
                let mean: f64 = self.mean();
                subtract_raw_offset(self, mean);
            }

            NormalizationMethod::Median => {
                // Median normalization: subtract the median (the middle order
                // statistic of the sorted values; mean of the two central ones
                // for an even count) to center around zero.
                let mono = self.as_mono().ok_or_else(|| {
                    AudioSampleError::Parameter(ParameterError::InvalidValue {
                        parameter: "self".to_string(),
                        reason: "Self is not mono".to_string(),
                    })
                })?;
                let mut sorted: Vec<f64> = mono.iter().map(|&x| -> f64 { x.cast_into() }).collect();
                sorted.sort_unstable_by(f64::total_cmp);
                let n = sorted.len();
                let median: f64 = if n.is_multiple_of(2) {
                    0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
                } else {
                    sorted[n / 2]
                };
                subtract_raw_offset(self, median);
            }

            NormalizationMethod::ZScore => {
                // Z-scores are dimensionless values of a few units; an integer
                // sample type would round them to a handful of codes around
                // zero, which is silence, so integer input is rejected.
                if !T::SAMPLE_TYPE.is_float() {
                    return Err(AudioSampleError::unsupported(
                        "normalize (ZScore)",
                        format!(
                            "z-score output (zero mean, unit standard deviation) cannot be represented by the integer sample type {}; convert to f32 or f64 first (e.g. audio.to_format::<f64>())",
                            T::SAMPLE_TYPE
                        ),
                    ));
                }
                // Z-Score normalization: zero mean, unit variance
                let mean: f64 = self.mean();
                let std_dev: f64 = self.std_dev();

                if std_dev == 0.0f64 {
                    // All values are the same, just subtract mean
                    subtract_raw_offset(self, mean);
                } else {
                    let map = |x: T| -> T {
                        let x: f64 = x.cast_into();
                        Self::Sample::cast_from((x - mean) / std_dev)
                    };
                    match self.data_mut() {
                        AudioData::Mono(arr) => arr.mapv_inplace(map),
                        AudioData::Multi(arr) => arr.mapv_inplace(map),
                    }
                }
            }
        }

        Ok(())
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
    /// # Returns
    /// The scaled audio samples.
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
    #[inline]
    fn scale_in_place(&mut self, factor: f64) {
        match self.data_mut() {
            AudioData::Mono(arr) => arr.mapv_inplace(|x| apply_gain_to_sample(x, factor)),
            AudioData::Multi(arr) => arr.mapv_inplace(|x| apply_gain_to_sample(x, factor)),
        }
    }

    /// Removes DC offset by subtracting the mean value.
    ///
    /// Computes `x'[n] = x[n] − x̄`, where `x̄` is the mean over all samples
    /// (all channels together). This centers the audio around zero and
    /// removes any constant bias that may have been introduced during
    /// recording or processing. Only the 0 Hz bin of a DFT over the whole
    /// buffer changes; for a drifting offset use
    /// [`high_pass_filter`](AudioProcessing::high_pass_filter) instead.
    ///
    /// The subtraction is done in `f64` on the raw sample values; for integer
    /// sample types the result is rounded to nearest and saturated.
    ///
    /// # Returns
    /// The audio samples with the DC offset removed.
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
    #[inline]
    fn remove_dc_offset_in_place(&mut self) -> AudioSampleResult<()> {
        let mean: f64 = self.mean();
        subtract_raw_offset(self, mean);
        Ok(())
    }

    /// Clips all samples in-place to `[min_val, max_val]`.
    ///
    /// Any samples outside `[min_val, max_val]` are clamped to the nearest
    /// boundary. Useful for preventing digital clipping before output. This is
    /// the in-place primitive; [`clip`](AudioProcessing::clip) is the
    /// non-mutating twin.
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
    #[inline]
    fn clip_in_place(
        &mut self,
        min_val: Self::Sample,
        max_val: Self::Sample,
    ) -> AudioSampleResult<()> {
        if min_val > max_val {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "clipping_range",
                format!("min ({min_val:?}) > max ({max_val:?})"),
                format!("{min_val:?}"),
                format!("{max_val:?}"),
                "min value must be less than or equal to max value for clipping",
            )));
        }

        match self.data_mut() {
            AudioData::Mono(arr) => arr.mapv_inplace(|x| x.clamp_to(min_val, max_val)),
            AudioData::Multi(arr) => arr.mapv_inplace(|x| x.clamp_to(min_val, max_val)),
        }
        Ok(())
    }

    /// Multiplies each sample by the corresponding window coefficient, with
    /// the window given in the sample type.
    ///
    /// Deprecated: an integer sample type cannot hold window coefficients
    /// between 0 and 1. Use
    /// [`apply_window_f64_in_place`](AudioProcessing::apply_window_f64_in_place).
    /// The product is computed in the sample type, so for integer types it can
    /// overflow; for float types the result is unchanged from 2.0.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the window length does not
    ///   match the audio length.
    #[inline]
    fn apply_window_in_place(
        &mut self,
        window: &NonEmptySlice<Self::Sample>,
    ) -> AudioSampleResult<()> {
        check_window_len(self, window.len().get())?;
        match self.data_mut() {
            AudioData::Mono(arr) => {
                for (sample, &win_coeff) in arr.iter_mut().zip(window.iter()) {
                    *sample *= win_coeff;
                }
            }
            AudioData::Multi(arr) => {
                for mut channel in arr.axis_iter_mut(Axis(0)) {
                    for (sample, &win_coeff) in channel.iter_mut().zip(window.iter()) {
                        *sample *= win_coeff;
                    }
                }
            }
        }
        Ok(())
    }

    /// Multiplies each sample by the corresponding `f64` window coefficient.
    ///
    /// Each sample is converted to `f64` with the audio-aware conversion,
    /// multiplied by `w[n]` and converted back; integer results are rounded to
    /// nearest and saturated. The same window is applied to every channel.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if the window length does not
    ///   match the audio length.
    #[inline]
    fn apply_window_f64_in_place(&mut self, window: &NonEmptySlice<f64>) -> AudioSampleResult<()> {
        check_window_len(self, window.len().get())?;
        let mul = |sample: &mut T, w: f64| {
            let x: f64 = (*sample).convert_to();
            *sample = (x * w).convert_to();
        };
        match self.data_mut() {
            AudioData::Mono(arr) => {
                for (sample, &w) in arr.iter_mut().zip(window.iter()) {
                    mul(sample, w);
                }
            }
            AudioData::Multi(arr) => {
                for mut channel in arr.axis_iter_mut(Axis(0)) {
                    for (sample, &w) in channel.iter_mut().zip(window.iter()) {
                        mul(sample, w);
                    }
                }
            }
        }
        Ok(())
    }

    /// Applies an FIR filter whose taps are given in the sample type.
    ///
    /// The taps are converted to `f64` with the audio-aware conversion (an
    /// `i16` tap of `16384` means about 0.5) and the filter is then applied as
    /// by [`apply_filter_f64_in_place`](AudioProcessing::apply_filter_f64_in_place).
    /// See [`AudioProcessing::apply_filter`] for the convolution convention.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the audio is shorter than
    ///   the filter.
    #[inline]
    fn apply_filter_in_place(
        &mut self,
        filter_coeffs: &NonEmptySlice<Self::Sample>,
    ) -> AudioSampleResult<()> {
        let coeffs: Vec<f64> = filter_coeffs
            .iter()
            .map(|&c| -> f64 { c.convert_to() })
            .collect();
        apply_fir_in_place(self, &coeffs)
    }

    /// Applies an FIR filter with `f64` taps (valid-mode convolution).
    ///
    /// See [`AudioProcessing::apply_filter_f64`] for the convolution
    /// convention. Samples are converted to `f64` with the audio-aware
    /// conversion; integer results are rounded to nearest and saturated.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the audio is shorter than
    ///   the filter.
    #[inline]
    fn apply_filter_f64_in_place(
        &mut self,
        filter_coeffs: &NonEmptySlice<f64>,
    ) -> AudioSampleResult<()> {
        apply_fir_in_place(self, filter_coeffs.as_ref())
    }

    /// Applies μ-law compression to the audio samples.
    ///
    /// Each sample `x` (on the normalised `[-1, 1]` scale) is mapped to the
    /// continuous μ-law curve `sign(x) · ln(1 + μ|x|) / ln(1 + μ)`, without
    /// quantisation (see [`mu_law_encode`] for integer codes). Higher `mu` values
    /// produce stronger compression. No quantisation is done (ITU-T G.711 uses
    /// an 8-segment piecewise-linear approximation of this curve plus 8-bit
    /// coding); for the quantised (integer code) variant see [`mu_law_encode`].
    ///
    /// # Arguments
    /// - `mu` - Compression parameter (typically 255 for standard μ-law). Its
    ///   numeric value is used as-is for every sample type (`255` means
    ///   μ = 255 for `i16` audio too).
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
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.mu_compress_in_place(255.0).unwrap();
    /// assert!(audio[0] > 0.0); // Positive input stays positive
    /// assert!(audio[1] < 0.0); // Negative input stays negative
    /// ```
    #[inline]
    fn mu_compress_in_place(&mut self, mu: Self::Sample) -> AudioSampleResult<()> {
        // `mu` is a curve parameter, not a sample value: take its raw
        // numeric value (no audio-aware [-1, 1] rescaling).
        let mu: f64 = mu.cast_into();
        validate_mu(mu, "mu_compress")?;
        if Self::Sample::SAMPLE_TYPE == crate::repr::SampleType::F32 {
            // f32 samples: the curve is evaluated in f64 with a branch-free
            // logarithm that vectorises (see `mu_law_compress_f64_fast`) and
            // rounded to f32 once.
            let inv_den = 1.0 / mu.ln_1p();
            if let Some(xs) = self.as_slice_mut() {
                if std::any::TypeId::of::<Self::Sample>() == std::any::TypeId::of::<f32>() {
                    // SAFETY: the sample type is f32, so the slice is a &mut [f32].
                    let xs = unsafe { &mut *(std::ptr::from_mut(xs) as *mut [f32]) };
                    mu_compress_f32_slice(xs, mu, inv_den);
                    return Ok(());
                }
            }
            return self.apply_with_error_in_place(|x: Self::Sample| {
                let x: f64 = x.convert_to();
                Ok(Self::Sample::convert_from(mu_law_compress_f64_fast(
                    x, mu, inv_den,
                )))
            });
        }
        if std::any::TypeId::of::<Self::Sample>() == std::any::TypeId::of::<f64>() {
            let den = mu.ln_1p();
            if let Some(xs) = self.as_slice_mut() {
                // SAFETY: the sample type is f64, so the slice is a &mut [f64].
                let xs = unsafe { &mut *(std::ptr::from_mut(xs) as *mut [f64]) };
                mu_compress_f64_slice(xs, mu, den);
                return Ok(());
            }
        }
        let law = MuLaw::new(mu);
        self.apply_with_error_in_place(|x: Self::Sample| {
            let x: f64 = x.convert_to();
            Ok(Self::Sample::convert_from(law.compress(x)))
        })
    }

    /// Applies μ-law expansion (decompression) to the audio samples.
    ///
    /// Each sample `y` (on the normalised `[-1, 1]` scale) is mapped to
    /// `sign(y) · ((1 + μ)^|y| - 1) / μ`, the exact inverse of
    /// [`mu_compress`](AudioProcessing::mu_compress). For integer codes see
    /// [`mu_law_decode`].
    ///
    /// # Arguments
    /// - `mu` - Expansion parameter. Must match the value used for compression.
    ///   Its numeric value is used as-is for every sample type.
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
    /// let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// audio.mu_expand_in_place(255.0).unwrap();
    /// assert_eq!(audio[0], 0.0); // Zero maps to zero
    /// assert!(audio[1] > 0.0);   // Sign is preserved
    /// assert!(audio[2] < 0.0);   // Sign is preserved
    /// ```
    #[inline]
    fn mu_expand_in_place(&mut self, mu: Self::Sample) -> AudioSampleResult<()> {
        let mu: f64 = mu.cast_into();
        validate_mu(mu, "mu_expand")?;
        let law = MuLaw::new(mu);
        // Float samples with `1 + μ` a power of two (μ = 255 and the like):
        // the power by a branch-free `exp2` that vectorises, where the CPU
        // has FMA.
        if let Some(k) = law.log2_base
            && let Some(xs) = self.as_slice_mut()
        {
            let id = std::any::TypeId::of::<Self::Sample>();
            if id == std::any::TypeId::of::<f64>() {
                // SAFETY: the sample type is f64, so the slice is a &mut [f64].
                let xs = unsafe { &mut *(std::ptr::from_mut(xs) as *mut [f64]) };
                if xs.iter().all(|v| k * v.abs() <= EXP2_FAST_MAX) && mu_expand_slice(xs, k, mu) {
                    return Ok(());
                }
            } else if id == std::any::TypeId::of::<f32>() {
                // SAFETY: the sample type is f32, so the slice is a &mut [f32].
                let xs = unsafe { &mut *(std::ptr::from_mut(xs) as *mut [f32]) };
                if xs.iter().all(|v| k * f64::from(v.abs()) <= EXP2_FAST_MAX)
                    && mu_expand_slice(xs, k, mu)
                {
                    return Ok(());
                }
            }
        }
        self.apply_with_error_in_place(|x: Self::Sample| {
            let y: f64 = x.convert_to();
            Ok(Self::Sample::convert_from(law.expand(y)))
        })
    }

    /// Applies a first-order low-pass filter with the specified cutoff frequency.
    ///
    /// The filter is a first-order Butterworth low-pass designed with the
    /// bilinear transform: -3 dB at `cutoff_hz`, unity gain at DC, a zero at
    /// Nyquist and a 6 dB/octave roll-off. It is applied to each channel
    /// independently from zero initial state; it is the order-1 case of
    /// [`AudioIirFiltering::butterworth_lowpass`], which gives steeper slopes at
    /// higher orders.
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
    #[inline]
    fn low_pass_filter_in_place(&mut self, cutoff_hz: f64) -> AudioSampleResult<()> {
        let design = IirFilterDesign::butterworth_lowpass(NonZeroUsize::MIN, cutoff_hz);
        self.apply_iir_filter_in_place(&design)
    }

    /// Applies a first-order high-pass filter with the specified cutoff frequency.
    ///
    /// The filter is a first-order Butterworth high-pass designed with the
    /// bilinear transform: -3 dB at `cutoff_hz`, a zero at DC, unity gain at
    /// Nyquist and a 6 dB/octave slope. It is applied to each channel
    /// independently from zero initial state; it is the order-1 case of
    /// [`AudioIirFiltering::butterworth_highpass`], which gives steeper slopes at
    /// higher orders.
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
    #[inline]
    fn high_pass_filter_in_place(&mut self, cutoff_hz: f64) -> AudioSampleResult<()> {
        let design = IirFilterDesign::butterworth_highpass(NonZeroUsize::MIN, cutoff_hz);
        self.apply_iir_filter_in_place(&design)
    }

    /// Applies a band-pass filter that passes frequencies between the two cutoffs.
    ///
    /// The filter is the band-pass transform of a first-order Butterworth
    /// prototype (a single second-order section): unity gain at the geometric
    /// centre of the (pre-warped) band, -3 dB at both edges, and 6 dB/octave
    /// slopes outside the band. It is applied from zero initial state and is the
    /// order-1 case of [`AudioIirFiltering::butterworth_bandpass`], which gives
    /// steeper slopes at higher orders.
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
    #[inline]
    fn band_pass_filter_in_place(
        &mut self,
        low_cutoff_hz: f64,
        high_cutoff_hz: f64,
    ) -> AudioSampleResult<()> {
        let design =
            IirFilterDesign::butterworth_bandpass(NonZeroUsize::MIN, low_cutoff_hz, high_cutoff_hz);
        self.apply_iir_filter_in_place(&design)
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
    #[inline]
    fn resample(
        &self,
        target_sample_rate: SampleRate,
        quality: ResamplingQuality,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>> {
        use crate::resample;

        resample::<Self::Sample>(self, target_sample_rate, quality)
    }

    /// Resamples audio to a new sample rate in place, replacing the internal
    /// buffer with the resampled result.
    ///
    /// Explicit in-place twin of [`resample`](AudioProcessing::resample); see that
    /// method for details. Computes the resampled buffer and assigns it back to
    /// `self`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the target sample rate or
    ///   quality parameters are invalid.
    /// - [`crate::AudioSampleError::Layout`] if the input audio is empty.
    #[cfg(feature = "resampling")]
    #[inline]
    fn resample_in_place(
        &mut self,
        target_sample_rate: SampleRate,
        quality: ResamplingQuality,
    ) -> AudioSampleResult<()> {
        *self = self.resample(target_sample_rate, quality)?.into_owned();
        Ok(())
    }

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
    #[inline]
    fn resample_by_ratio(
        &self,
        ratio: f64,
        quality: ResamplingQuality,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>> {
        use crate::resample_by_ratio;

        resample_by_ratio(self, ratio, quality)
    }

    /// Resamples audio by a specific ratio in place, replacing the internal
    /// buffer with the resampled result.
    ///
    /// Explicit in-place twin of
    /// [`resample_by_ratio`](AudioProcessing::resample_by_ratio); see that method
    /// for details. Computes the resampled buffer and assigns it back to `self`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `ratio` is ≤ 0 or the
    ///   quality parameters are invalid.
    /// - [`crate::AudioSampleError::Layout`] if the input audio is empty.
    #[cfg(feature = "resampling")]
    #[inline]
    fn resample_by_ratio_in_place(
        &mut self,
        ratio: f64,
        quality: ResamplingQuality,
    ) -> AudioSampleResult<()> {
        *self = self.resample_by_ratio(ratio, quality)?.into_owned();
        Ok(())
    }
}

// =============================================================================
// μ-law companding
// =============================================================================

/// Checks that a μ-law parameter is finite and strictly positive.
fn validate_mu(mu: f64, operation: &str) -> AudioSampleResult<()> {
    if mu.is_finite() && mu > 0.0 {
        Ok(())
    } else {
        Err(AudioSampleError::Parameter(ParameterError::out_of_range(
            "mu",
            mu,
            "> 0",
            "finite",
            format!("{operation}: the mu-law parameter must be finite and strictly positive"),
        )))
    }
}

/// `sign(x) · ln(1 + μ|x|) / ln(1 + μ)`, evaluated as
/// `sign(x) * log1p(mu * |x|) / log1p(mu)`.
#[cfg(test)]
fn mu_law_compress_value(x: f64, mu: f64) -> f64 {
    MuLaw::new(mu).compress(x)
}

/// The μ-law curve for one `μ`, with the per-signal constants computed once.
#[derive(Clone, Copy)]
struct MuLaw {
    mu: f64,
    /// `ln(1 + μ)`.
    den: f64,
    /// `1 + μ`.
    base: f64,
    /// `log2(1 + μ)` when it is a power of two (`1 + μ = 2^(2^j)`, e.g.
    /// `μ = 255`), so that `(1 + μ)^a = exp2(log2(1 + μ) · a)` with an exact
    /// product.
    log2_base: Option<f64>,
}

impl MuLaw {
    fn new(mu: f64) -> Self {
        let base = 1.0 + mu;
        let bits = base.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i64 - 1023;
        let log2_base = (base.is_finite()
            && bits & 0x000f_ffff_ffff_ffff == 0
            && exponent > 0
            && (exponent as u64).is_power_of_two())
        .then_some(exponent as f64);
        Self {
            mu,
            den: mu.ln_1p(),
            base,
            log2_base,
        }
    }

    /// `sign(x) · ln(1 + μ|x|) / ln(1 + μ)`.
    #[inline]
    fn compress(&self, x: f64) -> f64 {
        let sign = if x == 0.0 { 0.0 } else { x.signum() };
        sign * (self.mu * x.abs()).ln_1p() / self.den
    }

    /// `sign(y) / μ · ((1 + μ)^|y| - 1)`, evaluated in that order, with the
    /// power as `exp2` when `1 + μ` is a power of two.
    #[inline]
    fn expand(&self, y: f64) -> f64 {
        let sign = if y == 0.0 { 0.0 } else { y.signum() };
        let p = match self.log2_base {
            Some(k) => (k * y.abs()).exp2(),
            None => self.base.powf(y.abs()),
        };
        sign / self.mu * (p - 1.0)
    }
}

/// Largest argument of [`exp2_fma`] (its result stays normal and finite
/// well inside this).
const EXP2_FAST_MAX: f64 = 1000.0;

/// `2^x` for `0 <= x <= EXP2_FAST_MAX` without branches or calls, so that
/// loops over it vectorise: `x = n + f` with `n` the nearest integer and
/// `|f| <= 1/2` (both exact), `2^f` by its Taylor series to `f^13`
/// (truncation below 5e-18 relative) in Horner form with fused
/// multiply-adds, and `2^n` put into the exponent. Measured against the C
/// library's `exp2`: within 1 ulp. Only fast where the CPU has FMA.
#[inline(always)]
fn exp2_fma(x: f64) -> f64 {
    // Adding 1.5 * 2^52 rounds to the nearest integer, which lands in the
    // low bits of the sum.
    const SHIFT: f64 = 6_755_399_441_055_744.0;
    const C: [f64; 14] = {
        let ln2 = std::f64::consts::LN_2;
        let mut c = [1.0; 14];
        let mut k = 1;
        while k < 14 {
            c[k] = c[k - 1] * ln2 / k as f64;
            k += 1;
        }
        c
    };
    let t = x + SHIFT;
    let n = t - SHIFT;
    let f = x - n;
    let mut p = C[13];
    let mut k = 13;
    while k > 0 {
        k -= 1;
        p = p.mul_add(f, C[k]);
    }
    let e = (t.to_bits() & 0xfff) + 1023;
    p * f64::from_bits(e << 52)
}

/// Sample types the μ-law expansion fast path runs on.
trait ExpandSample: Copy {
    fn widen(self) -> f64;
    fn narrow(v: f64) -> Self;
}

impl ExpandSample for f64 {
    #[inline(always)]
    fn widen(self) -> f64 {
        self
    }
    #[inline(always)]
    fn narrow(v: f64) -> Self {
        v
    }
}

impl ExpandSample for f32 {
    #[inline(always)]
    fn widen(self) -> f64 {
        f64::from(self)
    }
    #[inline(always)]
    fn narrow(v: f64) -> Self {
        v as f32
    }
}

/// μ-law expansion in place, `sign(y) / μ · (2^(k |y|) - 1)` evaluated in
/// `f64` as [`MuLaw::expand`] does, with [`exp2_fma`] for the power (`k =
/// log2(1 + μ)`, every `k |y|` at most [`EXP2_FAST_MAX`]), on CPUs with AVX2
/// and FMA. Returns `false` (and changes nothing) on other CPUs, which use
/// the C library's `exp2` instead (the results differ by at most an ulp of
/// the power).
fn mu_expand_slice<S: ExpandSample>(xs: &mut [S], k: f64, mu: f64) -> bool {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: the CPU supports AVX2 and FMA.
        unsafe { mu_expand_slice_avx2(xs, k, mu) };
        return true;
    }
    let _ = (xs, k, mu);
    false
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn mu_expand_slice_avx2<S: ExpandSample>(xs: &mut [S], k: f64, mu: f64) {
    for v in xs {
        let y = v.widen();
        let sign = if y == 0.0 { 0.0 } else { y.signum() };
        *v = S::narrow(sign / mu * (exp2_fma(k * y.abs()) - 1.0));
    }
}

/// Natural logarithm of a positive, finite, normal `x` by fdlibm's
/// `__ieee754_log` algorithm and constants (error below 1 ulp), written without branches
/// or calls so that loops over it vectorise.
#[inline(always)]
fn ln_branch_free(x: f64) -> f64 {
    const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000);
    const LN2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76);
    const LG1: f64 = f64::from_bits(0x3fe5_5555_5555_5593);
    const LG2: f64 = f64::from_bits(0x3fd9_9999_9997_fa04);
    const LG3: f64 = f64::from_bits(0x3fd2_4924_9422_9359);
    const LG4: f64 = f64::from_bits(0x3fcc_71c5_1d8e_78af);
    const LG5: f64 = f64::from_bits(0x3fc7_4664_96cb_03de);
    const LG6: f64 = f64::from_bits(0x3fc3_9a09_d078_c69f);
    const LG7: f64 = f64::from_bits(0x3fc2_f112_df3e_5244);
    let bits = x.to_bits();
    // x = 2^k m with m in [1, 2); fold m into [sqrt(2)/2, sqrt(2)).
    let k = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    let big = m > std::f64::consts::SQRT_2;
    let m = if big { m * 0.5 } else { m };
    let dk = f64::from(k + i32::from(big));
    let f = m - 1.0;
    let hfsq = 0.5 * f * f;
    let s = f / (2.0 + f);
    let z = s * s;
    let w = z * z;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    let r = t2 + t1;
    dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f)
}

/// `ln(1 + u)` for `u >= 0` without branches or calls, within about 1 ulp:
/// the series of `log1p` to `u^7` below `u = 2^-10` (truncation below 2^-60
/// relative), else [`ln_branch_free`] of `w = 1 + u` plus the first-order
/// correction `e / w` for the exact rounding error `e = (1 + u) - w`.
/// Infinite and NaN `u` return `u`.
#[inline(always)]
fn ln_1p_branch_free(u: f64) -> f64 {
    let w = 1.0 + u;
    // Fast2Sum error of 1 + u, ordering the operands by magnitude.
    let e = if u < 1.0 {
        u - (w - 1.0)
    } else {
        1.0 - (w - u)
    };
    let big = ln_branch_free(w) + e / w;
    let series = u
        * (1.0
            + u * (-0.5
                + u * (1.0 / 3.0 + u * (-0.25 + u * (0.2 + u * (-1.0 / 6.0 + u * (1.0 / 7.0)))))));
    if u >= 1.0 / 1024.0 {
        if w.is_finite() { big } else { w }
    } else if u.is_nan() {
        u
    } else {
        series
    }
}

/// μ-law companding of f64 samples in place, `sign(x) ln(1 + μ|x|) / den`
/// with `den = ln(1 + μ)` and [`ln_1p_branch_free`]; the loop vectorises,
/// with an AVX2 version selected at run time (the same arithmetic).
fn mu_compress_f64_slice(xs: &mut [f64], mu: f64, den: f64) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU supports AVX2.
        unsafe { mu_compress_f64_slice_avx2(xs, mu, den) };
        return;
    }
    mu_compress_f64_slice_portable(xs, mu, den);
}

#[inline(always)]
fn mu_compress_f64_slice_portable(xs: &mut [f64], mu: f64, den: f64) {
    for v in xs {
        let x = *v;
        let sign = if x == 0.0 { 0.0 } else { x.signum() };
        *v = sign * ln_1p_branch_free(mu * x.abs()) / den;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn mu_compress_f64_slice_avx2(xs: &mut [f64], mu: f64, den: f64) {
    mu_compress_f64_slice_portable(xs, mu, den);
}

/// [`mu_law_compress_f64_fast`] over f32 samples in place, with AVX2 when the
/// CPU has it (the arithmetic is the same: no fused operations).
fn mu_compress_f32_slice(xs: &mut [f32], mu: f64, inv_den: f64) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU supports AVX2.
        unsafe { mu_compress_f32_slice_avx2(xs, mu, inv_den) };
        return;
    }
    mu_compress_f32_slice_portable(xs, mu, inv_den);
}

#[inline(always)]
fn mu_compress_f32_slice_portable(xs: &mut [f32], mu: f64, inv_den: f64) {
    for v in xs {
        *v = mu_law_compress_f64_fast(f64::from(*v), mu, inv_den) as f32;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn mu_compress_f32_slice_avx2(xs: &mut [f32], mu: f64, inv_den: f64) {
    mu_compress_f32_slice_portable(xs, mu, inv_den);
}

/// `sign(x) ln(1 + mu |x|) / ln(1 + mu)` (as `... * inv_den`), for f32 samples
/// widened to f64 (so `u = mu |x|` is exact for f32 `mu`).
///
/// For `u >= 2^-10` the logarithm is `ln(1 + u)` by [`ln_branch_free`]
/// (`1 + u` is exact for `mu = 255` and otherwise rounds by at most 2^-53,
/// a relative error below 2^-43 in the logarithm); below, the Taylor series
/// of `log1p` to `u^7` (truncation below 2^-73 relative). Both are far more
/// accurate than the final rounding to f32. Infinite and NaN input give the
/// same result as `mu_law_compress_value`.
#[inline(always)]
fn mu_law_compress_f64_fast(x: f64, mu: f64, inv_den: f64) -> f64 {
    let u = mu * x.abs();
    let w = 1.0 + u;
    let big = ln_branch_free(w);
    let series = u
        * (1.0
            + u * (-0.5
                + u * (1.0 / 3.0 + u * (-0.25 + u * (0.2 + u * (-1.0 / 6.0 + u * (1.0 / 7.0)))))));
    let l = if u >= 1.0 / 1024.0 {
        if w == f64::INFINITY { w } else { big }
    } else {
        series
    };
    let sign = if x == 0.0 { 0.0 } else { x.signum() };
    sign * l * inv_den
}

/// Quantises samples to integer μ-law codes.
///
/// Each sample is first companded with `c = sign(x) · ln(1 + μ|x|) / ln(1 + μ)`
/// (see [`AudioProcessing::mu_compress`]) and then assigned to one of
/// `⌊1 + μ⌋` uniform bins on `[-1, 1]`: with bin edges
/// `e = linspace(-1, 1, ⌊1 + μ⌋)` the code is `#{j : e[j] < c} - ⌊(1 + μ) / 2⌋`.
/// For `μ = 255` the codes
/// are the 256 integers in `[-128, 127]`; `0.0` maps to `0`, `1.0` to `127` and
/// `-1.0` to `-128`.
///
/// Use this for WaveNet-style 8-bit class targets; [`mu_law_decode`] maps codes
/// back to amplitudes. For float-in / float-out companding of a whole signal use
/// [`AudioProcessing::mu_compress`].
///
/// # Arguments
/// - `x` - Samples on the normalised `[-1, 1]` scale (e.g. the output of
///   `audio.as_f64()`, one channel at a time).
/// - `mu` - Compression parameter (typically 255). Need not be an integer;
///   the number of codes is `⌊1 + μ⌋`.
///
/// # Returns
/// One `i64` code per input sample, in `[-⌊(1 + μ) / 2⌋, ⌊1 + μ⌋ - ⌊(1 + μ) / 2⌋ - 1]`.
///
/// # Errors
/// - [`crate::AudioSampleError::Parameter`] if `mu` is not finite and strictly
///   positive, `mu > 2^53`, or if any sample is outside `[-1, 1]` (or NaN).
///
/// # Examples
/// ```
/// use audio_samples::mu_law_encode;
///
/// let codes = mu_law_encode(&[-1.0, -0.5, 0.0, 0.5, 1.0], 255.0).unwrap();
/// assert_eq!(codes, vec![-128, -112, 0, 112, 127]);
/// ```
///
/// # References
/// - ITU-T Recommendation G.711.
#[inline]
pub fn mu_law_encode(x: &[f64], mu: f64) -> AudioSampleResult<Vec<i64>> {
    validate_mu(mu, "mu_law_encode")?;
    // Beyond 2^53 neither `1 + mu` nor the codes are exactly representable.
    if mu > 9_007_199_254_740_992.0 {
        return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
            "mu",
            mu,
            "> 0",
            "<= 2^53",
            "mu_law_encode: the number of codes must be exactly representable",
        )));
    }
    // Bin edges: `int(1 + mu)` points evenly spaced over [-1, 1], evaluated on
    // demand as `i * step + start` with the last edge pinned to `stop`, so that
    // samples landing on an edge get a reproducible code, without materialising
    // `1 + mu` edges.
    let num = (1.0 + mu).trunc() as usize;
    let step = if num > 1 { 2.0 / (num - 1) as f64 } else { 0.0 };
    let edge = |i: usize| -> f64 {
        if num == 1 {
            -1.0
        } else if i == num - 1 {
            1.0
        } else {
            i as f64 * step + -1.0
        }
    };
    // The bin index is the number of edges strictly below c.
    // Start from the analytic estimate and correct for rounding.
    let count_below = |c: f64| -> usize {
        let mut k = if num > 1 {
            (((c + 1.0) / step).floor().max(0.0) as usize).min(num)
        } else {
            0
        };
        while k < num && edge(k) < c {
            k += 1;
        }
        while k > 0 && edge(k - 1) >= c {
            k -= 1;
        }
        k
    };
    let offset = (num / 2) as i64;
    let law = MuLaw::new(mu);
    x.iter()
        .map(|&v| {
            if !(-1.0..=1.0).contains(&v) {
                return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                    "x",
                    v,
                    -1.0,
                    1.0,
                    "mu_law_encode: input samples must lie in [-1, 1]",
                )));
            }
            let c = law.compress(v);
            Ok(count_below(c) as i64 - offset)
        })
        .collect()
}

/// Maps integer μ-law codes back to amplitudes.
///
/// Each code `k` is rescaled to `y = 2k / (1 + μ)` and expanded with
/// `sign(y) / μ · ((1 + μ)^|y| - 1)` (see [`AudioProcessing::mu_expand`]).
/// This is not the exact inverse of [`mu_law_encode`]: codes
/// are bin indices, so a round trip carries quantisation error (at most about
/// 0.043 for `μ = 255`, and biased towards `-1`, e.g. code `127` decodes to
/// `0.9574`).
///
/// # Arguments
/// - `codes` - μ-law codes, typically from [`mu_law_encode`].
/// - `mu` - The μ used for encoding.
///
/// # Returns
/// One amplitude in `[-1, 1]` per code.
///
/// # Errors
/// - [`crate::AudioSampleError::Parameter`] if `mu` is not finite and strictly
///   positive, or if a code rescales to a value outside `[-1, 1]` (i.e.
///   `|2k / (1 + μ)| > 1`).
///
/// # Examples
/// ```
/// use audio_samples::{mu_law_decode, mu_law_encode};
///
/// let decoded = mu_law_decode(&[-128, 0, 127], 255.0).unwrap();
/// assert_eq!(decoded[0], -1.0);
/// assert_eq!(decoded[1], 0.0);
/// assert!((decoded[2] - 0.957_437).abs() < 1e-6);
///
/// let x = [0.25, -0.75];
/// let back = mu_law_decode(&mu_law_encode(&x, 255.0).unwrap(), 255.0).unwrap();
/// assert!((back[0] - x[0]).abs() < 0.05 && (back[1] - x[1]).abs() < 0.05);
/// ```
///
/// # References
/// - ITU-T Recommendation G.711.
#[inline]
pub fn mu_law_decode(codes: &[i64], mu: f64) -> AudioSampleResult<Vec<f64>> {
    validate_mu(mu, "mu_law_decode")?;
    let law = MuLaw::new(mu);
    let decode_one = |k: i64| law.expand(k as f64 * 2.0 / (1.0 + mu));
    let in_range = |k: i64| (-1.0..=1.0).contains(&(k as f64 * 2.0 / (1.0 + mu)));
    // Codes take few distinct values (256 for mu = 255): when every code is
    // valid and the code range is small, decode each value once into a table.
    if let (Some(&lo), Some(&hi)) = (codes.iter().min(), codes.iter().max()) {
        let span = hi.abs_diff(lo);
        if in_range(lo) && in_range(hi) && span < (codes.len() as u64).min(1 << 16) {
            let table: Vec<f64> = (lo..=hi).map(decode_one).collect();
            return Ok(codes
                .iter()
                .map(|&k| table[k.abs_diff(lo) as usize])
                .collect());
        }
    }
    codes
        .iter()
        .map(|&k| {
            let y = k as f64 * 2.0 / (1.0 + mu);
            if !(-1.0..=1.0).contains(&y) {
                return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                    "codes",
                    k,
                    -((1.0 + mu) / 2.0),
                    (1.0 + mu) / 2.0,
                    "mu_law_decode: code rescales outside [-1, 1]",
                )));
            }
            Ok(law.expand(y))
        })
        .collect()
}

// Helper methods for the AudioProcessing implementation
impl<T> AudioSamples<'_, T>
where
    T: StandardSample,
{
    /// Applies a fallible function to all samples in the audio data in place.
    /// This is a helper method for operations that can fail.
    ///
    /// # Examples
    /// ```rust,ignore
    /// // Apply a conversion that might fail
    /// audio.apply_with_error_in_place(|sample| {
    ///     if sample > max_value {
    ///         Err(AudioSampleError::Parameter(/* error */))
    ///     } else {
    ///         Ok(sample * 2.0)
    ///     }
    /// })?;
    /// ```
    ///
    /// # Errors
    ///
    /// - if the provided function errors for some reason
    #[inline]
    pub fn apply_with_error_in_place<F>(&mut self, f: F) -> AudioSampleResult<()>
    where
        F: Fn(T) -> AudioSampleResult<T>,
    {
        match self.data_mut() {
            AudioData::Mono(arr) => {
                for x in arr.iter_mut() {
                    *x = f(*x)?;
                }
            }
            AudioData::Multi(arr) => {
                for x in arr.iter_mut() {
                    *x = f(*x)?;
                }
            }
        }
        Ok(())
    }

    /// Applies a fallible function that uses an accumulator pattern.
    /// Useful for operations that need to maintain state across samples.
    ///
    /// # Examples
    /// ```rust,ignore
    /// let mut state = 0.0;
    /// audio.try_fold(|acc, sample| {
    ///     *acc += sample.abs();
    ///     if *acc > threshold {
    ///         Err(AudioSampleError::Parameter(/* error */))
    ///     } else {
    ///         Ok(sample)
    ///     }
    /// })?;
    /// ```
    ///
    ///  # Errors
    ///
    /// - if the provided function errors for some reason
    #[inline]
    pub fn try_fold<Acc, F>(&mut self, mut acc: Acc, mut f: F) -> AudioSampleResult<Acc>
    where
        F: FnMut(&mut Acc, T) -> AudioSampleResult<T>,
    {
        match self.data_mut() {
            AudioData::Mono(arr) => {
                for x in arr.iter_mut() {
                    *x = f(&mut acc, *x)?;
                }
                Ok(acc)
            }
            AudioData::Multi(arr) => {
                for x in arr.iter_mut() {
                    *x = f(&mut acc, *x)?;
                }
                Ok(acc)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample_rate;
    use crate::traits::AudioTypeConversion;
    use approx_eq::assert_approx_eq;
    use ndarray::array;

    use crate::AudioProcessing;

    /// The FFT fast path (filter >= FIR_FFT_MIN_TAPS) must match the direct
    /// slice convolution to floating-point tolerance, preserving apply_filter's
    /// valid-region / correlation convention.
    #[cfg(feature = "transforms")]
    #[test]
    fn apply_filter_fft_matches_direct_long_filter() {
        let n = 4000usize;
        let m = 512usize; // >= FIR_FFT_MIN_TAPS, so this exercises the FFT path
        assert!(m >= FIR_FFT_MIN_TAPS);

        let input: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.013).sin() + 0.4 * (i as f32 * 0.07).cos())
            .collect();
        let coeffs: Vec<f32> = (0..m)
            .map(|k| ((k as f32 * 0.05).sin()) / m as f32)
            .collect();

        // Reference: direct slice convolution (same convention).
        let output_len = n - m + 1;
        let h: Vec<f64> = coeffs.iter().map(|&c| f64::from(c)).collect();
        let reference = fir_valid_convolve(&input, &h, output_len);

        // Public API path (routes to FFT because m >= threshold).
        let audio: AudioSamples<f32> =
            AudioSamples::new_mono(ndarray::Array1::from_vec(input), sample_rate!(44100)).unwrap();
        let filtered = audio
            .apply_filter(NonEmptySlice::new(&coeffs).unwrap())
            .unwrap();

        assert_eq!(filtered.samples_per_channel().get(), output_len);
        for (i, &want) in reference.iter().enumerate() {
            let got = filtered[i];
            assert!(
                (got - want).abs() < 1e-3,
                "sample {i}: fft {got} vs direct {want}"
            );
        }
    }

    #[test]
    fn test_normalize_min_max() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio: AudioSamples<f32> = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let audio = audio
            .normalize(NormalizationConfig::min_max(-1.0, 1.0))
            .unwrap();

        assert_approx_eq!(audio.min_sample() as f64, -1.0);
        assert_approx_eq!(audio.max_sample() as f64, 1.0);
    }

    #[test]
    fn test_normalize_peak() {
        let data = array![-2.0f32, 1.0, 3.0, -1.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let audio = audio.normalize(NormalizationConfig::peak(1.0)).unwrap();

        assert_approx_eq!(audio.peak() as f64, 1.0);
    }

    #[test]
    fn test_scale() {
        let data = array![1.0f32, 2.0, 3.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let audio = audio.scale(2.0);

        let expected = array![2.0f32, 4.0, 6.0];
        match audio.data() {
            AudioData::Mono(arr) => {
                for (actual, expected) in arr.iter().zip(expected.iter()) {
                    assert_approx_eq!(*actual as f64, *expected as f64, 1e-6);
                }
            }
            _ => panic!("Expected mono data"),
        }
    }

    #[test]
    fn test_remove_dc_offset() {
        let data = array![3.0f32, 4.0, 5.0]; // Mean = 4.0
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let audio = audio.remove_dc_offset().unwrap();

        let mean: f64 = audio.mean();
        assert_approx_eq!(mean as f64, 0.0, 1e-6);
    }

    #[test]
    fn test_clip() {
        let data = array![-3.0f32, -1.0, 0.0, 1.0, 3.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let audio = audio.clip(-2.0, 2.0).unwrap();

        let expected = array![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        match audio.data() {
            AudioData::Mono(arr) => {
                for (actual, expected) in arr.iter().zip(expected.iter()) {
                    assert_approx_eq!(*actual as f64, *expected as f64, 1e-6);
                }
            }
            _ => panic!("Expected mono data"),
        }
    }

    #[test]
    #[allow(deprecated)] // checks that the deprecated sample-typed window keeps its float behaviour
    fn test_apply_window() {
        let data = array![1.0f32, 1.0, 1.0, 1.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
        let window = [0.5f32, 1.0, 1.0, 0.5];
        let window_slice = NonEmptySlice::new(&window).unwrap();
        let audio = audio.apply_window(window_slice).unwrap();

        let expected = array![0.5f32, 1.0, 1.0, 0.5];
        match audio.data() {
            AudioData::Mono(arr) => {
                for (actual, expected) in arr.iter().zip(expected.iter()) {
                    assert_approx_eq!(*actual as f64, *expected as f64, 1e-6);
                }
            }
            _ => panic!("Expected mono data"),
        }
    }

    #[test]
    fn test_multi_channel_normalize() {
        let data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let audio: AudioSamples<f32> =
            AudioSamples::new_multi_channel(data, sample_rate!(44100)).unwrap();

        let audio = audio
            .normalize(NormalizationConfig::min_max(-1.0, 1.0))
            .unwrap();

        assert_approx_eq!(audio.min_sample() as f64, -1.0);
        assert_approx_eq!(audio.max_sample() as f64, 1.0);
    }

    #[test]
    fn test_normalize_zscore() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let audio = audio.normalize(NormalizationConfig::zscore()).unwrap();

        // After Z-score normalization, mean should be ~0 and std dev should be ~1
        let mean: f64 = audio.mean();
        let std_dev: f64 = audio.std_dev();
        assert_approx_eq!(mean as f64, 0.0, 1e-6);
        assert_approx_eq!(std_dev as f64, 1.0, 1e-6);
    }

    #[test]
    fn test_direct_chaining() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        // Test chaining multiple operations using consuming pattern
        let audio = audio.scale(2.0).clip(-5.0, 5.0).unwrap();

        // Verify the result
        let expected = array![2.0f32, 4.0, 5.0, 5.0, 5.0]; // After scaling and clipping
        let result_data = audio.as_mono().unwrap();
        for (actual, expected) in result_data.iter().zip(expected.iter()) {
            assert_approx_eq!(*actual as f64, *expected as f64, 1e-6);
        }
    }

    #[test]
    fn test_chaining_error_handling() {
        let data = array![1.0f32, 2.0, 3.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        // Test that invalid operations return errors
        let result = audio.normalize(NormalizationConfig::min_max(2.0, 1.0)); // Invalid range (min > max)

        assert!(result.is_err());
    }

    #[test]
    fn test_multi_channel_chaining() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let audio = AudioSamples::new_multi_channel(data, sample_rate!(44100)).unwrap();

        let expected = array![[0.5f32, 1.0], [1.5, 2.0]];

        // Test chaining with multi-channel audio
        let audio = audio.scale(0.5);

        // Check the result using the public API instead of accessing data directly
        let result_data = audio.as_multi_channel().unwrap();
        for (actual_row, expected_row) in result_data
            .axis_iter(ndarray::Axis(0))
            .zip(expected.axis_iter(ndarray::Axis(0)))
        {
            for (actual, expected) in actual_row.iter().zip(expected_row.iter()) {
                assert_approx_eq!(*actual as f64, *expected as f64, 1e-6);
            }
        }
    }

    /// Helper: assert two mono signals are element-wise approximately equal.
    fn assert_mono_eq(a: &AudioSamples<f32>, b: &AudioSamples<f32>) {
        let a = a.as_mono().unwrap();
        let b = b.as_mono().unwrap();
        assert_eq!(a.len(), b.len(), "lengths differ");
        for (x, y) in a.iter().zip(b.iter()) {
            assert_approx_eq!(*x as f64, *y as f64, 1e-6);
        }
    }

    /// `normalize` (non-mutating) and `normalize_in_place` must produce identical
    /// results, and the non-mutating variant must leave the original untouched.
    #[test]
    fn test_normalize_dual_variant_equivalence() {
        let data = array![1.0f32, -3.0, 2.0, -1.0, 0.5];
        let original = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        // Non-mutating variant.
        let borrowed = original.normalize(NormalizationConfig::peak(1.0)).unwrap();

        // The original must be unchanged.
        assert_mono_eq(
            &original,
            &AudioSamples::new_mono(array![1.0f32, -3.0, 2.0, -1.0, 0.5], sample_rate!(44100))
                .unwrap(),
        );

        // In-place variant on a clone.
        let mut mutated = original.clone();
        mutated
            .normalize_in_place(NormalizationConfig::peak(1.0))
            .unwrap();

        // Both variants must agree.
        assert_mono_eq(&borrowed, &mutated);
    }

    /// `scale` (non-mutating, infallible) and `scale_in_place` must produce
    /// identical results, and the non-mutating variant must leave the original
    /// untouched.
    #[test]
    fn test_scale_dual_variant_equivalence() {
        let original =
            AudioSamples::new_mono(array![1.0f32, -2.0, 3.0, -4.0], sample_rate!(44100)).unwrap();

        let borrowed = original.scale(0.25);

        // Original unchanged.
        assert_mono_eq(
            &original,
            &AudioSamples::new_mono(array![1.0f32, -2.0, 3.0, -4.0], sample_rate!(44100)).unwrap(),
        );

        let mut mutated = original.clone();
        mutated.scale_in_place(0.25);

        assert_mono_eq(&borrowed, &mutated);
    }

    /// `clip` (non-mutating) and `clip_in_place` must produce identical results,
    /// and the non-mutating variant must leave the original untouched.
    #[test]
    fn test_clip_dual_variant_equivalence() {
        let original =
            AudioSamples::new_mono(array![2.0f32, -3.0, 1.5, -0.5], sample_rate!(44100)).unwrap();

        let borrowed = original.clip(-1.0, 1.0).unwrap();

        // Original unchanged.
        assert_mono_eq(
            &original,
            &AudioSamples::new_mono(array![2.0f32, -3.0, 1.5, -0.5], sample_rate!(44100)).unwrap(),
        );

        let mut mutated = original.clone();
        mutated.clip_in_place(-1.0, 1.0).unwrap();

        assert_mono_eq(&borrowed, &mutated);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "fma")]
    unsafe fn exp2_fma_checked(x: f64) -> f64 {
        exp2_fma(x)
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn fma_exp2_is_within_one_ulp_of_exp2() {
        if !std::arch::is_x86_feature_detected!("fma") {
            return;
        }
        let mut x = 0.0_f64;
        while x <= EXP2_FAST_MAX {
            for v in [x, x + 0.5, x + 0.499_999_999, x + 1e-9] {
                // SAFETY: only reached on CPUs with FMA (checked above).
                let (a, b) = (unsafe { exp2_fma_checked(v) }, v.exp2());
                let ulp = f64::EPSILON * b;
                assert!((a - b).abs() <= ulp, "exp2({v}): {a} vs {b}");
            }
            x += 0.013_7;
        }
    }

    /// The vectorised μ-law expansion (μ = 255) matches the scalar formula to
    /// within the rounding of the power, one ulp of `2^(k |y|)`, divided by μ
    /// (the subtraction of 1 keeps that absolute error).
    #[test]
    fn mu_expand_fast_path_matches_formula() {
        let x: Vec<f64> = (0..=4000).map(|i| -1.0 + f64::from(i) * 0.0005).collect();
        let audio =
            AudioSamples::new_mono(ndarray::Array1::from(x.clone()), sample_rate!(8000)).unwrap();
        let y = audio.mu_expand(255.0).unwrap();
        let law = MuLaw::new(255.0);
        for (i, v) in x.iter().enumerate() {
            let e = law.expand(*v);
            let tol = 2.0 * f64::EPSILON * (e.abs() + 1.0 / 255.0);
            assert!((y[i] - e).abs() <= tol, "{v}: {} vs {e}", y[i]);
        }
    }

    #[test]
    fn branch_free_ln_is_within_one_ulp_of_ln() {
        let mut x = 1.0e-300_f64;
        while x < 1.0e300 {
            for v in [
                x,
                x * std::f64::consts::SQRT_2,
                x * 1.414_213_562_373_096,
                x * 1.999_999,
            ] {
                let (a, b) = (ln_branch_free(v), v.ln());
                let ulp = f64::EPSILON * b.abs().max(f64::MIN_POSITIVE);
                assert!((a - b).abs() <= ulp, "ln({v}): {a} vs {b}");
            }
            x *= 1.37;
        }
    }

    #[test]
    fn branch_free_ln_1p_is_within_two_ulp_of_ln_1p() {
        let mut u = 1.0e-300_f64;
        while u < 1.0e300 {
            for v in [u, u * 1.000_001, u * 1.7] {
                let (a, b) = (ln_1p_branch_free(v), v.ln_1p());
                assert!(
                    (a - b).abs() <= 2.0 * f64::EPSILON * b,
                    "ln_1p({v}): {a} vs {b}"
                );
            }
            u *= 1.21;
        }
        assert_eq!(ln_1p_branch_free(0.0), 0.0);
        assert!(ln_1p_branch_free(f64::NAN).is_nan());
        assert_eq!(ln_1p_branch_free(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn fast_mu_compress_agrees_with_the_reference_formula() {
        let mut x = -1.0_f32;
        while x <= 1.0 {
            for v in [x, x * 1e-3, x * 1e-6, x * 1e-30] {
                let v = f64::from(v);
                let a = mu_law_compress_f64_fast(v, 255.0, 1.0 / 255.0_f64.ln_1p());
                let b = mu_law_compress_value(v, 255.0);
                assert!(
                    (a - b).abs() <= 4.0 * f64::EPSILON * b.abs(),
                    "{v}: {a} vs {b}"
                );
            }
            x += 0.000_37;
        }
        let den = 1.0 / 255.0_f64.ln_1p();
        assert_eq!(mu_law_compress_f64_fast(0.0, 255.0, den), 0.0);
        assert!(mu_law_compress_f64_fast(f64::NAN, 255.0, den).is_nan());
        assert_eq!(
            mu_law_compress_f64_fast(f64::NEG_INFINITY, 255.0, den),
            mu_law_compress_value(f64::NEG_INFINITY, 255.0)
        );
    }

    /// `mu_compress` follows the continuous μ-law curve
    /// `sign(x) ln(1 + μ|x|) / ln(1 + μ)`: 0 maps to 0 and ±1 to ±1.
    #[test]
    fn mu_compress_matches_reference_formula() {
        let x = array![-1.0f64, -0.5, 0.0, 0.5, 1.0];
        let audio = AudioSamples::new_mono(x, sample_rate!(8000)).unwrap();
        let y = audio.mu_compress(255.0).unwrap();
        // Continuous μ-law curve (μ = 255) at [-1, -0.5, 0, 0.5, 1].
        let expected = [-1.0, -0.875_703_07, 0.0, 0.875_703_07, 1.0];
        for (i, e) in expected.iter().enumerate() {
            assert!((y[i] - e).abs() < 1e-8, "sample {i}: {} vs {e}", y[i]);
        }
        assert_eq!(y[2], 0.0);
    }

    #[test]
    fn mu_expand_is_exact_inverse_of_mu_compress() {
        let x: Vec<f64> = (0..=200).map(|i| -1.0 + f64::from(i) * 0.01).collect();
        let audio =
            AudioSamples::new_mono(ndarray::Array1::from(x.clone()), sample_rate!(8000)).unwrap();
        for mu in [1.0, 15.0, 255.0, 1000.5] {
            let back = audio.mu_compress(mu).unwrap().mu_expand(mu).unwrap();
            for (i, v) in x.iter().enumerate() {
                assert!((back[i] - v).abs() < 1e-14, "mu {mu}: {} vs {v}", back[i]);
            }
        }
    }

    /// For integer sample types `mu` is the raw numeric value (255i16 means
    /// μ = 255, not the audio-scaled 255/32768): the result matches
    /// compressing the f64 conversion of the same audio to within 1 LSB.
    #[test]
    fn mu_law_integer_samples_use_numeric_mu() {
        let codes = array![-32768i16, -16384, 0, 8192, 16384, 32767];
        let audio = AudioSamples::new_mono(codes.clone(), sample_rate!(8000)).unwrap();
        let reference = audio.as_f64().mu_compress(255.0).unwrap();
        let compressed = audio.mu_compress(255).unwrap();
        for i in 0..codes.len() {
            let expected: i16 = reference[i].convert_to();
            assert!(
                (i32::from(compressed[i]) - i32::from(expected)).abs() <= 1,
                "sample {i}: {} vs {expected}",
                compressed[i]
            );
        }
        assert_eq!(compressed[2], 0);
        assert_eq!(compressed[0], -32768);
        // Round trip through i16: the slope of the expander is at most
        // ln(256) * 256 / 255 ≈ 5.6, so the 1-LSB rounding of the compressed
        // value grows to a few LSB.
        let back = compressed.mu_expand(255).unwrap();
        for i in 0..codes.len() {
            assert!(
                (i32::from(back[i]) - i32::from(codes[i])).abs() <= 8,
                "sample {i}: {} vs {}",
                back[i],
                codes[i]
            );
        }
        let i32_audio = AudioSamples::new_mono(array![0i32, 1 << 30], sample_rate!(8000)).unwrap();
        let c = i32_audio.mu_compress(255).unwrap();
        assert_eq!(c[0], 0);
        let expected: i32 = mu_law_compress_value(0.5, 255.0).convert_to();
        assert!((i64::from(c[1]) - i64::from(expected)).abs() <= 1);
    }

    #[test]
    fn mu_law_rejects_invalid_mu() {
        let audio = AudioSamples::new_mono(array![0.1f64, 0.2], sample_rate!(8000)).unwrap();
        assert!(audio.mu_compress(0.0).is_err());
        assert!(audio.mu_compress(-1.0).is_err());
        assert!(audio.mu_expand(f64::NAN).is_err());
        assert!(mu_law_encode(&[0.0], 0.0).is_err());
        assert!(mu_law_decode(&[0], -2.0).is_err());
    }

    #[test]
    fn mu_law_encode_matches_librosa_codes() {
        let x: Vec<f64> = (0..11).map(|i| -1.0 + f64::from(i) * 0.2).collect();
        // Codes of 11 points evenly spaced over [-1, 1], mu = 255.
        assert_eq!(
            mu_law_encode(&x, 255.0).unwrap(),
            vec![-128, -122, -116, -107, -91, 0, 91, 107, 116, 122, 127]
        );
        // The same points, mu = 15.
        assert_eq!(
            mu_law_encode(&x, 15.0).unwrap(),
            vec![-8, -7, -6, -5, -4, 0, 4, 5, 6, 7, 7]
        );
        assert!(mu_law_encode(&[1.5], 255.0).is_err());
        assert!(mu_law_encode(&[f64::NAN], 255.0).is_err());
    }

    /// The encoder computes each code per sample without materialising the
    /// `1 + mu` bin edges: codes equal the count of explicit edges
    /// `linspace(-1, 1, int(1 + mu))` below `c`, a huge `mu` (1e15) needs no
    /// allocation, and `mu` above 2^53 is an error rather than a panic.
    #[test]
    fn mu_law_encode_is_allocation_free_and_exact() {
        let reference = |x: &[f64], mu: f64| -> Vec<i64> {
            let num = (1.0 + mu).trunc() as usize;
            let edges: Vec<f64> = if num == 1 {
                vec![-1.0]
            } else {
                let step = 2.0 / (num - 1) as f64;
                let mut e: Vec<f64> = (0..num).map(|i| i as f64 * step + -1.0).collect();
                e[num - 1] = 1.0;
                e
            };
            x.iter()
                .map(|&v| {
                    let c = mu_law_compress_value(v, mu);
                    edges.partition_point(|&e| e < c) as i64 - (num / 2) as i64
                })
                .collect()
        };
        // Includes inputs that land exactly on bin edges (c = -1, 0, 1, and
        // the companded values of the edges themselves for mu = 1).
        let mut x: Vec<f64> = (0..=2000).map(|i| -1.0 + f64::from(i) * 0.001).collect();
        x.extend([-1.0, 1.0, 0.0, 1e-300, -1e-300, 0.5, -0.5]);
        for mu in [
            0.5,
            1.0,
            2.0,
            3.7,
            15.0,
            100.5,
            255.0,
            1023.0,
            65535.0,
            1e6 + 0.25,
        ] {
            assert_eq!(mu_law_encode(&x, mu).unwrap(), reference(&x, mu), "mu={mu}");
        }
        // Huge mu: no allocation, codes span the full range.
        let codes = mu_law_encode(&[-1.0, 0.0, 1.0], 1e15).unwrap();
        assert_eq!(codes[1], 0);
        assert!(codes[0] < -(1i64 << 48) && codes[2] > 1i64 << 48);
        assert!(mu_law_encode(&[0.5], 1e19).is_err());
        assert!(mu_law_encode(&[0.5], 1e300).is_err());
    }

    #[test]
    fn mu_law_decode_matches_librosa() {
        // Decoded values of codes [-128, -122, -116, 0, 116, 127], mu = 255.
        let y = mu_law_decode(&[-128, -122, -116, 0, 116, 127], 255.0).unwrap();
        let expected = [
            -1.0,
            -0.770_207_79,
            -0.593_013_77,
            0.0,
            0.593_013_77,
            0.957_437_02,
        ];
        for (a, e) in y.iter().zip(expected) {
            assert!((a - e).abs() < 1e-8, "{a} vs {e}");
        }
        assert!(mu_law_decode(&[129], 255.0).is_err());
    }
}
