//! Complex-domain onset detection functions, which use both the magnitude and the
//! phase of each CQT bin: magnitude differences between frames, the deviation of
//! each bin's phase advance from that of its centre frequency, and their weighted
//! combination ([`ComplexOnsetConfig`]). The
//! [`AudioOnsetDetection`](crate::operations::traits::AudioOnsetDetection)
//! methods call these; use them directly to build a custom detector.

use super::ComplexOnsetConfig;
use ndarray::{Array2, ArrayView2, Axis, s};
use non_empty_slice::NonEmptyVec;
use num_complex::Complex;

/// Computes frame-to-frame magnitude differences in a spectrogram.
///
/// For each frequency bin, calculates the difference between consecutive frames:
/// `diff[b, t] = mag[b, t] - mag[b, t-1]`. Positive values indicate magnitude increases,
/// negative values indicate decreases.
///
/// # Arguments
///
/// * `mag` - 2D magnitude spectrogram with shape `(frequency_bins, frames)`, must be in
///   standard (row-major) layout
///
/// # Returns
///
/// 2D array of magnitude differences with the same shape. The first frame (t=0) contains
/// all zeros since there is no previous frame.
///
/// # Example
///
/// ```rust
/// use ndarray::array;
/// use audio_samples::operations::onset::complex::magnitude_difference;
///
/// let mag = array![
///     [1.0, 2.0, 3.0, 2.5],
///     [0.5, 1.5, 2.0, 1.0],
///     [2.0, 2.5, 3.5, 3.0],
/// ];
/// let diff = magnitude_difference(mag.view());
/// assert_eq!(diff.row(0).to_vec(), vec![0.0, 1.0, 1.0, -0.5]);
/// assert_eq!(diff[[1, 3]], -1.0);
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
/// * `a` - First phase value in radians
/// * `b` - Second phase value in radians
///
/// # Returns
///
/// Phase difference in radians, wrapped to `(-π, π]`.
///
/// For example, `wrapped_phase_diff(π, −π)` is 0: the difference 2π wraps to 0.
#[inline]
#[allow(dead_code)] // single-wrap helper; `phase_deviation` uses `princarg`
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
    if wrapped <= -PI {
        wrapped + TAU
    } else {
        wrapped
    }
}

/// Computes the deviation of each bin's phase advance from that of its centre frequency.
///
/// For bin b with centre frequency f_b (Hz), hop H (samples) and sample rate f_s:
///
/// ```text
/// dev[b, t] = | princarg( φ[b,t] − φ[b,t−1] − 2π·f_b·H/f_s ) |   (radians, in [0, π])
/// ```
///
/// where `princarg` wraps to (−π, π]. This is a first-order measure: it is zero only for a
/// stationary sinusoid exactly at the bin centre. A stationary sinusoid off the bin centre
/// gives a constant non-zero value, and empty (noise-only) bins give essentially random
/// values. Note this differs from the second-difference phase deviation
/// `princarg(φ_t − 2φ_{t−1} + φ_{t−2})` of Bello et al. (2005) and from the complex-domain
/// ODF `|X_t − X̂_t|`.
///
/// # Arguments
///
/// * `spec` - 2D complex spectrogram with shape `(frequency_bins, frames)`
/// * `config` - Complex onset configuration (provides hop size and CQT parameters)
/// * `sample_rate` - Audio sample rate in Hz
///
/// # Returns
///
/// 2D array of absolute phase deviations with the same shape. The first frame (t=0) contains
/// all zeros since there is no previous frame. Values are always non-negative.
///
/// # Example
///
/// ```rust
/// use ndarray::Array2;
/// use num_complex::Complex;
/// use audio_samples::operations::onset::complex::phase_deviation;
/// use audio_samples::operations::onset::ComplexOnsetConfig;
///
/// let config = ComplexOnsetConfig::default();
/// let sr = 44_100.0;
/// // Each bin advances by exactly its centre frequency's phase step: no deviation.
/// let (bins, frames) = (8, 5);
/// let spec = Array2::from_shape_fn((bins, frames), |(b, t)| {
///     let step = std::f64::consts::TAU * config.cqt_config.bin_frequency(b)
///         * config.hop_size.get() as f64 / sr;
///     Complex::from_polar(1.0, step * t as f64)
/// });
/// let deviation = phase_deviation(spec.view(), &config, sr);
/// assert!(deviation.iter().all(|&d| d < 1e-9));
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
/// frequency bins. Each term is rectified (`max(0, ·)`) when the corresponding
/// `*_rectify` flag is set and taken as an absolute value otherwise; optional logarithmic
/// compression reduces dynamic range.
///
/// The formula is:
/// `odf[t] = magnitude_weight · Σ_b R(mag[b,t]) + phase_weight · Σ_b R(phase[b,t])`
/// with `R(v) = max(0, v)` (rectify) or `|v|`.
///
/// If `log_compression > 0`, applies: `odf[t] = log(1 + log_compression * odf[t])`
///
/// # Arguments
///
/// * `mag_diff` - 2D magnitude difference array with shape `(frequency_bins, frames)`
/// * `phase_dev` - 2D phase deviation array with shape `(frequency_bins, frames)`
/// * `config` - Complex onset configuration (provides weights, rectification, compression)
///
/// # Returns
///
/// Onset detection function values for each frame, with length equal to number of frames.
///
/// # Example
///
/// ```rust
/// use ndarray::array;
/// use audio_samples::operations::onset::complex::combine_complex_odf;
/// use audio_samples::operations::onset::ComplexOnsetConfig;
///
/// let mut config = ComplexOnsetConfig::default();
/// config.magnitude_weight = 0.5;
/// config.phase_weight = 0.25;
/// config.log_compression = 0.0;
/// config.magnitude_rectify = true;
/// // Two bins, three frames.
/// let mag_diff = array![[0.0, 2.0, -1.0], [0.0, 1.0, 0.0]];
/// let phase_dev = array![[0.0, 0.4, 0.8], [0.0, 0.0, 0.4]];
/// let odf = combine_complex_odf(&mag_diff, &phase_dev, &config);
/// // 0.5 × (rectified magnitude sum) + 0.25 × (phase sum)
/// for (got, want) in odf.iter().zip([0.0, 1.6, 0.3]) {
///     assert!((got - want).abs() < 1e-12);
/// }
/// ```
#[inline]
#[must_use]
pub fn combine_complex_odf(
    mag_diff: &Array2<f64>,
    phase_dev: &Array2<f64>,
    config: &ComplexOnsetConfig,
) -> NonEmptyVec<f64> {
    debug_assert_eq!(mag_diff.dim(), phase_dev.dim());
    let frames = mag_diff.ncols();

    let mag_rect = config.magnitude_rectify;
    let phase_rect = config.phase_rectify;

    // Accumulate bin by bin into per-frame sums. Each bin is a row of the
    // `(bins, frames)` arrays, so this walks the data in memory order for the
    // standard layout and never needs a contiguous column (a column of a
    // row-major array is strided, so slicing columns would fail as soon as there
    // is more than one frame). Works for any layout without copying.
    let mut mag_sum = vec![0.0_f64; frames];
    let mut phase_sum = vec![0.0_f64; frames];
    for (mag_row, phase_row) in mag_diff.rows().into_iter().zip(phase_dev.rows()) {
        for ((m_acc, p_acc), (&m, &p)) in mag_sum
            .iter_mut()
            .zip(phase_sum.iter_mut())
            .zip(mag_row.iter().zip(phase_row.iter()))
        {
            *m_acc += if mag_rect { m.max(0.0) } else { m.abs() };
            *p_acc += if phase_rect { p.max(0.0) } else { p.abs() };
        }
    }

    let odf: Vec<f64> = mag_sum
        .iter()
        .zip(&phase_sum)
        .map(|(&m, &p)| {
            let v = config.magnitude_weight.mul_add(m, config.phase_weight * p);
            if config.log_compression > 0.0 {
                config.log_compression.mul_add(v, 1.0).ln()
            } else {
                v
            }
        })
        .collect();

    // safety: frames > 0 ensures odf is non-empty
    unsafe { NonEmptyVec::new_unchecked(odf) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// In the non-rectified combine path the magnitude
    /// contribution must come from `mag_diff`, not from `phase_dev`.
    ///
    /// The test uses `ComplexOnsetConfig::speech(..)` (`phase_rectify: false`), forces the
    /// output to depend only on the magnitude term (magnitude_weight = 1,
    /// phase_weight = 0, no log compression), and feeds magnitude and phase arrays
    /// with deliberately different values, so a result taken from the phase data
    /// fails the assertion.
    #[test]
    fn nonrectified_magnitude_comes_from_magnitude() {
        let mut config = ComplexOnsetConfig::speech(crate::sample_rate!(44100));
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

    /// `combine_complex_odf` must work for any number of frames and any memory
    /// layout, without requiring contiguous columns.
    #[test]
    fn combine_handles_many_frames_and_any_layout() {
        use ndarray::ShapeBuilder;

        let (bins, frames) = (7, 5);
        let mag =
            Array2::from_shape_fn((bins, frames), |(b, t)| (b as f64 - 3.0) * (t as f64 + 0.5));
        let phase = Array2::from_shape_fn((bins, frames), |(b, t)| ((b * 3 + t) % 5) as f64 - 1.5);
        for mut config in [
            ComplexOnsetConfig::musical(crate::sample_rate!(44100)),
            ComplexOnsetConfig::speech(crate::sample_rate!(44100)),
        ] {
            config.log_compression = 0.0;
            let expected: Vec<f64> = (0..frames)
                .map(|t| {
                    let (mut m, mut p) = (0.0, 0.0);
                    for b in 0..bins {
                        let (x, y) = (mag[[b, t]], phase[[b, t]]);
                        m += if config.magnitude_rectify {
                            x.max(0.0)
                        } else {
                            x.abs()
                        };
                        p += if config.phase_rectify {
                            y.max(0.0)
                        } else {
                            y.abs()
                        };
                    }
                    config.magnitude_weight * m + config.phase_weight * p
                })
                .collect();

            let standard = combine_complex_odf(&mag, &phase, &config);
            // Column-major copies: same values, transposed strides.
            let mut mag_f = Array2::zeros((bins, frames).f());
            mag_f.assign(&mag);
            let mut phase_f = Array2::zeros((bins, frames).f());
            phase_f.assign(&phase);
            assert!(!mag_f.is_standard_layout());
            let fortran = combine_complex_odf(&mag_f, &phase_f, &config);

            for t in 0..frames {
                assert!((standard[t] - expected[t]).abs() < 1e-12, "frame {t}");
                assert!((fortran[t] - expected[t]).abs() < 1e-12, "frame {t}");
            }
        }
    }

    /// For a bin whose true phase advances by exactly `expected = 2π·f·hop/sr` each
    /// hop, the phase deviation must be ~0 even when `expected > π` (high-frequency
    /// bins). Wrapping the raw phase difference before subtracting `expected` would
    /// leave a large residual.
    #[test]
    fn phase_deviation_zero_for_expected_advance() {
        let config = ComplexOnsetConfig::speech(crate::sample_rate!(44100));
        let sample_rate = 44100.0;
        let hop = config.hop_size.get() as f64;
        let tau = std::f64::consts::TAU;

        // Find a high bin whose expected per-hop advance exceeds π, where a single
        // wrap is not enough.
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
