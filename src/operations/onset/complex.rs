//! Complex domain onset detection functions.
//!
//! This module provides low-level functions for onset detection using both magnitude and phase
//! information from complex spectrograms. Complex domain methods analyze changes in both the
//! amplitude and phase of frequency components, providing more detailed information about
//! spectral changes than magnitude-only approaches.
//!
//! The primary functions compute:
//! - Magnitude differences between consecutive frames
//! - Phase deviations from expected phase progression
//! - Combined onset detection functions weighted by configuration
//!
//! These functions are used internally by the [`AudioOnsetDetection`] trait implementations
//! and are not typically called directly by users.
//!
//! [`AudioOnsetDetection`]: crate::operations::traits::AudioOnsetDetection

use super::ComplexOnsetConfig;
use ndarray::{Array2, ArrayView2, Axis, s};
use non_empty_slice::NonEmptyVec;
use num_complex::Complex;
#[cfg(feature = "simd")]
use wide::f64x4;

/// Computes frame-to-frame magnitude differences in a spectrogram.
///
/// For each frequency bin, calculates the difference between consecutive frames:
/// `diff[b, t] = mag[b, t] - mag[b, t-1]`. Positive values indicate magnitude increases,
/// negative values indicate decreases.
///
/// # Arguments
///
/// * `mag` — 2D magnitude spectrogram with shape `(frequency_bins, frames)`, must be in
///   standard (row-major) layout
///
/// # Returns
///
/// 2D array of magnitude differences with the same shape. The first frame (t=0) contains
/// all zeros since there is no previous frame.
///
/// # Example
///
/// ```rust,ignore
/// use ndarray::Array2;
/// use audio_samples::operations::onset::complex::magnitude_difference;
///
/// let mag = Array2::from_shape_vec((3, 4), vec![
///     1.0, 2.0, 3.0, 2.5,
///     0.5, 1.5, 2.0, 1.0,
///     2.0, 2.5, 3.5, 3.0,
/// ]).unwrap();
/// let diff = magnitude_difference(mag.view());
/// // diff[freq=0, time=1] = 2.0 - 1.0 = 1.0
/// ```
#[inline]
#[must_use]
pub fn magnitude_difference(mag: ArrayView2<f64>) -> Array2<f64> {
    let (bins, frames) = mag.dim();
    let mut out = Array2::zeros((bins, frames));

    let mag = mag.as_standard_layout();

    for (mut out_row, mag_row) in out.axis_iter_mut(Axis(0)).zip(mag.axis_iter(Axis(0))) {
        let curr = mag_row.slice(s![1..]);
        let prev = mag_row.slice(s![..frames - 1]);
        let mut out_slice = out_row.slice_mut(s![1..]);

        for ((o, &c), &p) in out_slice.iter_mut().zip(curr).zip(prev) {
            *o = c - p;
        }
    }

    out
}

/// Computes the wrapped phase difference between two phase values.
///
/// Calculates `a - b` and wraps the result to the range `(-π, π]` to handle phase wrapping
/// around ±π. This ensures that phase differences are always represented as the shortest
/// angular distance.
///
/// # Arguments
///
/// * `a` — First phase value in radians
/// * `b` — Second phase value in radians
///
/// # Returns
///
/// Phase difference in radians, wrapped to `(-π, π]`.
///
/// # Example
///
/// ```rust,ignore
/// use std::f64::consts::PI;
/// use audio_samples::operations::onset::complex::wrapped_phase_diff;
///
/// // Crossing the +π boundary: π - (-π) = 2π wraps to 0
/// let diff = wrapped_phase_diff(PI, -PI);
/// assert!((diff - 0.0).abs() < 1e-10);
/// ```
#[inline]
#[allow(dead_code)] // retained utility; superseded by `princarg` in `phase_deviation`
fn wrapped_phase_diff(a: f64, b: f64) -> f64 {
    let mut d = a - b;

    if d > std::f64::consts::PI {
        d -= std::f64::consts::TAU;
    } else if d < -std::f64::consts::PI {
        d += std::f64::consts::TAU;
    }

    d
}

/// Maps an angle (in radians) to its principal value in `(-π, π]`.
///
/// Unlike [`wrapped_phase_diff`], which only adjusts by a single ±2π, this
/// handles arbitrarily large inputs via modular arithmetic, which is required
/// when wrapping `phase_diff - expected` where `expected = 2π·f·hop/sr` can be
/// many multiples of 2π for high-frequency bins.
#[inline]
fn princarg(angle: f64) -> f64 {
    use std::f64::consts::{PI, TAU};
    // rem_euclid maps into [0, TAU); shift to (-π, π].
    let wrapped = (angle + PI).rem_euclid(TAU) - PI;
    // rem_euclid produces [-π, π); fold -π up to π so the range is (-π, π].
    if wrapped <= -PI { wrapped + TAU } else { wrapped }
}

/// Computes phase deviation from expected phase progression.
///
/// For each frequency bin, calculates the deviation between observed phase change and the
/// expected phase change based on the bin's center frequency. Large deviations indicate
/// transient events or changes in spectral content.
///
/// The expected phase change for a sinusoid at frequency `f` over hop size `H` is:
/// `2π * f * H / sample_rate`. Deviations from this indicate non-sinusoidal behavior.
///
/// # Arguments
///
/// * `spec` — 2D complex spectrogram with shape `(frequency_bins, frames)`
/// * `config` — Complex onset configuration (provides hop size and CQT parameters)
/// * `sample_rate` — Audio sample rate in Hz
///
/// # Returns
///
/// 2D array of absolute phase deviations with the same shape. The first frame (t=0) contains
/// all zeros since there is no previous frame. Values are always non-negative.
///
/// # Example
///
/// ```rust,ignore
/// use ndarray::Array2;
/// use num_complex::Complex;
/// use audio_samples::operations::onset::complex::phase_deviation;
/// use audio_samples::operations::onset::ComplexOnsetConfig;
///
/// let config = ComplexOnsetConfig::new();
/// let spec = Array2::from_elem((128, 100), Complex::new(1.0, 0.0));
/// let deviation = phase_deviation(spec.view(), &config, 44100.0);
/// ```
#[inline]
#[must_use]
pub fn phase_deviation(
    spec: ArrayView2<Complex<f64>>,
    config: &ComplexOnsetConfig,
    sample_rate: f64,
) -> Array2<f64> {
    let (bins, frames) = spec.dim();
    let mut out = Array2::zeros((bins, frames));

    let hop = config.hop_size.get() as f64;
    let tau = std::f64::consts::TAU;

    for (b, (mut out_row, spec_row)) in out
        .axis_iter_mut(Axis(0))
        .zip(spec.axis_iter(Axis(0)))
        .enumerate()
    {
        let f = config.cqt_config.bin_frequency(b);
        let expected = tau * f * hop / sample_rate;

        let mut prev_phase = spec_row[0].arg();

        for t in 1..frames {
            let phase = spec_row[t].arg();
            let diff = phase - prev_phase;
            // Standard phase-deviation ODF: wrap the FINAL quantity into (-π, π].
            // `expected` can exceed π for high-frequency bins, so wrapping the raw
            // phase difference before subtracting `expected` would leave a large,
            // never-re-wrapped residual that makes high bins dominate.
            out_row[t] = princarg(diff - expected).abs();
            prev_phase = phase;
        }
    }

    out
}
/// Combines magnitude differences and phase deviations into a single onset detection function.
///
/// For each frame, computes a weighted sum of magnitude and phase contributions across all
/// frequency bins. Optional rectification keeps only positive values, and optional logarithmic
/// compression reduces dynamic range.
///
/// The formula is:
/// `odf[t] = magnitude_weight * Σ mag[b,t] + phase_weight * Σ phase[b,t]`
///
/// If `log_compression > 0`, applies: `odf[t] = log(1 + log_compression * odf[t])`
///
/// # Arguments
///
/// * `mag_diff` — 2D magnitude difference array with shape `(frequency_bins, frames)`
/// * `phase_dev` — 2D phase deviation array with shape `(frequency_bins, frames)`
/// * `config` — Complex onset configuration (provides weights, rectification, compression)
///
/// # Returns
///
/// Onset detection function values for each frame, with length equal to number of frames.
///
/// # Example
///
/// ```rust,ignore
/// use ndarray::Array2;
/// use audio_samples::operations::onset::complex::combine_complex_odf;
/// use audio_samples::operations::onset::ComplexOnsetConfig;
///
/// let mag_diff = Array2::zeros((128, 100));
/// let phase_dev = Array2::zeros((128, 100));
/// let config = ComplexOnsetConfig::musical();
/// let odf = combine_complex_odf(&mag_diff, &phase_dev, &config);
/// ```
#[inline]
#[must_use]
pub fn combine_complex_odf(
    mag_diff: &Array2<f64>,
    phase_dev: &Array2<f64>,
    config: &ComplexOnsetConfig,
) -> NonEmptyVec<f64> {
    let (bins, frames) = mag_diff.dim();
    let mut odf = vec![0.0; frames];

    let mag_rect = config.magnitude_rectify;
    let phase_rect = config.phase_rectify;

    let mag_diff = mag_diff.as_standard_layout();
    let phase_dev = phase_dev.as_standard_layout();

    for (t, odf_val) in odf.iter_mut().enumerate().take(frames) {
        let mag_col = mag_diff.index_axis(Axis(1), t);
        let phase_col = phase_dev.index_axis(Axis(1), t);

        let mut b = 0;

        #[cfg(feature = "simd")]
        let mut mag_acc = f64x4::ZERO;
        #[cfg(feature = "simd")]
        let mut phase_acc = f64x4::ZERO;

        #[cfg(not(feature = "simd"))]
        let mut mag_acc = 0.0_f64;
        #[cfg(not(feature = "simd"))]
        let mut phase_acc = 0.0_f64;

        if mag_rect && phase_rect {
            while b + 4 <= bins {
                let m_view = mag_col.slice(s![b..b + 4]);
                let p_view = phase_col.slice(s![b..b + 4]);
                let m_slice = m_view.as_slice().unwrap_or_else(|| unreachable!("We made sure mag_diff array is contiguous, therefore any 1d slice from it should also be contiguous"));
                let p_slice = p_view.as_slice().unwrap_or_else(|| unreachable!("We made sure phase_dev array is contiguous, therefore any 1d slice from it should also be contiguous"));
                #[cfg(feature = "simd")]
                {
                    let m = f64x4::new([m_slice[0], m_slice[1], m_slice[2], m_slice[3]])
                        .max(f64x4::ZERO);
                    let p = f64x4::new([p_slice[0], p_slice[1], p_slice[2], p_slice[3]])
                        .max(f64x4::ZERO);
                    mag_acc += m;
                    phase_acc += p;
                }
                #[cfg(not(feature = "simd"))]
                {
                    for i in 0..4 {
                        mag_acc += m_slice[i].max(0.0);
                        phase_acc += p_slice[i].max(0.0);
                    }
                }
                b += 4;
            }
        } else {
            while b + 4 <= bins {
                let m_view = mag_col.slice(s![b..b + 4]);
                let p_view = phase_col.slice(s![b..b + 4]);
                let m_slice = m_view.as_slice().unwrap_or_else(|| unreachable!("We made sure mag_diff array is contiguous, therefore any 1d slice from it should also be contiguous"));
                let p_slice = p_view.as_slice().unwrap_or_else(|| unreachable!("We made sure phase_dev array is contiguous, therefore any 1d slice from it should also be contiguous"));
                #[cfg(feature = "simd")]
                {
                    let m = f64x4::new([m_slice[0], m_slice[1], m_slice[2], m_slice[3]]).abs();
                    let p = f64x4::new([p_slice[0], p_slice[1], p_slice[2], p_slice[3]]).abs();
                    mag_acc += m;
                    phase_acc += p;
                }
                #[cfg(not(feature = "simd"))]
                {
                    for i in 0..4 {
                        mag_acc += m_slice[i].abs();
                        phase_acc += p_slice[i].abs();
                    }
                }
                b += 4;
            }
        }

        #[cfg(feature = "simd")]
        let mut mag_sum = mag_acc.reduce_add();
        #[cfg(feature = "simd")]
        let mut phase_sum = phase_acc.reduce_add();

        #[cfg(not(feature = "simd"))]
        let mut mag_sum = mag_acc;
        #[cfg(not(feature = "simd"))]
        let mut phase_sum = phase_acc;

        for i in b..bins {
            mag_sum += if mag_rect {
                mag_col[i].max(0.0)
            } else {
                mag_col[i].abs()
            };
            phase_sum += if phase_rect {
                phase_col[i].max(0.0)
            } else {
                phase_col[i].abs()
            };
        }

        let mut v = config
            .magnitude_weight
            .mul_add(mag_sum, config.phase_weight * phase_sum);

        if config.log_compression > 0.0 {
            v = config.log_compression.mul_add(v, 1.0).ln();
        }

        *odf_val = v;
    }

    // safety: frames > 0 ensures odf is non-empty
    unsafe { NonEmptyVec::new_unchecked(odf) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for BUG 1: in the non-rectified combine path the magnitude
    /// contribution must come from `mag_diff`, not from `phase_dev`.
    ///
    /// We use `ComplexOnsetConfig::speech()` (`phase_rectify: false`), force the
    /// output to depend only on the magnitude term (magnitude_weight = 1,
    /// phase_weight = 0, no log compression), and feed magnitude and phase arrays
    /// with deliberately different values. Before the fix, `m_slice` was bound to
    /// the phase view, so the result reflected the phase data and this assertion
    /// failed.
    #[test]
    fn bug1_nonrectified_magnitude_comes_from_magnitude() {
        let mut config = ComplexOnsetConfig::speech();
        assert!(!config.phase_rectify, "test relies on non-rectified phase");
        config.magnitude_weight = 1.0;
        config.phase_weight = 0.0;
        config.log_compression = 0.0;
        // Ensure we hit the non-(mag&&phase)-rectify branch.
        config.magnitude_rectify = false;

        // 8 bins (two full 4-wide SIMD/blocked iterations), 1 frame.
        let bins = 8;
        let mag_val = 2.0;
        let phase_val = 100.0; // wildly different from magnitude
        let mag_diff = Array2::from_elem((bins, 1), mag_val);
        let phase_dev = Array2::from_elem((bins, 1), phase_val);

        let odf = combine_complex_odf(&mag_diff, &phase_dev, &config);

        // Output = magnitude_weight * Σ|mag| = 1.0 * (8 * 2.0) = 16.0
        let expected = bins as f64 * mag_val;
        assert!(
            (odf[0] - expected).abs() < 1e-9,
            "expected magnitude-derived sum {expected}, got {} (would be {} if read from phase)",
            odf[0],
            bins as f64 * phase_val
        );
    }

    /// Regression test for BUG 2: for a bin whose true phase advances by exactly
    /// `expected = 2π·f·hop/sr` each hop, the phase deviation must be ~0 even when
    /// `expected > π` (high-frequency bins). The buggy version wrapped the raw
    /// phase difference BEFORE subtracting `expected`, leaving a large residual.
    #[test]
    fn bug2_phase_deviation_zero_for_expected_advance() {
        let config = ComplexOnsetConfig::speech();
        let sample_rate = 44100.0;
        let hop = config.hop_size.get() as f64;
        let tau = std::f64::consts::TAU;

        // Find a high bin whose expected per-hop advance exceeds π, which is what
        // makes the bug observable (single-wrap is insufficient there).
        let bins = 64usize;
        let frames = 6usize;
        let mut high_bin = 0usize;
        for b in 0..bins {
            let f = config.cqt_config.bin_frequency(b);
            let expected = tau * f * hop / sample_rate;
            if expected > std::f64::consts::PI {
                high_bin = b;
                break;
            }
        }
        assert!(high_bin > 0, "expected to find a bin with advance > π");

        // Build a spec where every bin's phase advances by exactly its own
        // `expected` per frame (magnitude = 1).
        let mut spec = Array2::from_elem((bins, frames), Complex::new(0.0, 0.0));
        for b in 0..bins {
            let f = config.cqt_config.bin_frequency(b);
            let expected = tau * f * hop / sample_rate;
            for t in 0..frames {
                let ph = expected * t as f64;
                spec[[b, t]] = Complex::from_polar(1.0, ph);
            }
        }

        let dev = phase_deviation(spec.view(), &config, sample_rate);

        // The high bin's deviation must be ~0 for all t >= 1.
        for t in 1..frames {
            assert!(
                dev[[high_bin, t]].abs() < 1e-9,
                "high bin {high_bin} deviation at frame {t} should be ~0, got {}",
                dev[[high_bin, t]]
            );
        }
    }
}
