//! Spectral flux kernels: the frame-to-frame change of a spectrogram, which rises at
//! onsets because a new note or hit adds energy across many bins.
//!
//! The four variants:
//! - **Energy flux** ([`energy_flux`]): `Σ_b max(0, X[b,t] − X[b,t−1])` on the values it is
//!   given (no squaring); with a magnitude spectrogram this is the standard half-wave-rectified
//!   spectral flux
//! - **Magnitude flux** ([`magnitude_flux`]): `Σ_b |X[b,t] − X[b,t−1]|` (absolute, not
//!   rectified: decreases count too)
//! - **Complex-domain flux** ([`complex_flux`]): `Σ_b |X[b,t] − X̂[b,t]|`, the distance from a
//!   prediction `X̂` that keeps the previous magnitude and the previous phase advance, so steady
//!   tones give nothing and both level and phase changes count
//! - **Rectified complex-domain flux** ([`rectified_complex_flux`]): the same sum over the bins
//!   whose magnitude has not fallen, so note offsets do not count
//!
//! All flux functions operate on 2D spectrograms with shape `(frequency_bins, frames)` and
//! return a 1D time series of flux values, one per frame. The first frame always has zero flux.

use ndarray::Array2;
use non_empty_slice::NonEmptyVec;
use num_complex::Complex;

/// Computes "energy" spectral flux (half-wave-rectified difference).
///
/// Calculates the sum of positive frame-to-frame differences across all frequency bins:
/// `flux[t] = Σ_b max(0, mag[b,t] − mag[b,t−1])`.
///
/// The input values are not squared. [`AudioOnsetDetection::spectral_flux`](crate::operations::traits::AudioOnsetDetection::spectral_flux)
/// passes a CQT **magnitude** spectrogram, so the result is the half-wave-rectified magnitude
/// flux (the standard spectral flux); pass a power spectrogram to obtain an energy flux.
///
/// # Arguments
///
/// * `mag` - 2D magnitude spectrogram with shape `(frequency_bins, frames)`
///
/// # Returns
///
/// Flux values for each frame. The first frame is always 0.0 (no previous frame to compare).
///
/// # Example
///
/// ```rust
/// use ndarray::array;
/// use audio_samples::operations::onset::flux::energy_flux;
///
/// let mag = array![[0.0, 1.0, 1.0, 0.5], [0.0, 0.0, 2.0, 2.0]];
/// let flux = energy_flux(&mag);
/// assert_eq!(flux.as_slice(), &[0.0, 1.0, 2.0, 0.0]);
/// ```
#[inline]
#[must_use]
pub fn energy_flux(mag: &Array2<f64>) -> NonEmptyVec<f64> {
    let (bins, frames) = mag.dim();
    let mut flux = Vec::with_capacity(frames);
    flux.push(0.0);

    for t in 1..frames {
        let mut acc = 0.0;
        for b in 0..bins {
            let diff = mag[[b, t]] - mag[[b, t - 1]];
            if diff > 0.0 {
                acc += diff;
            }
        }
        flux.push(acc);
    }

    // safety: frames > 1 ensures flux is non-empty

    unsafe { NonEmptyVec::new_unchecked(flux) }
}

/// Computes magnitude-based spectral flux.
///
/// Calculates the sum of absolute magnitude differences across all frequency bins:
/// `flux[t] = Σ_b |mag[b,t] − mag[b,t−1]|`.
///
/// Not rectified: energy decreases (note offsets) contribute as much as increases, and a
/// later rectification step has no effect because the values are already non-negative.
///
/// # Arguments
///
/// * `mag` - 2D magnitude spectrogram with shape `(frequency_bins, frames)`
///
/// # Returns
///
/// Flux values for each frame. The first frame is always 0.0 (no previous frame to compare).
///
/// # Example
///
/// ```rust
/// use ndarray::array;
/// use audio_samples::operations::onset::flux::magnitude_flux;
///
/// let mag = array![[0.0, 1.0, 1.0, 0.5], [0.0, 0.0, 2.0, 2.0]];
/// let flux = magnitude_flux(&mag);
/// // Decreases count too: frame 3 has |0.5 − 1.0| = 0.5.
/// assert_eq!(flux.as_slice(), &[0.0, 1.0, 2.0, 0.5]);
/// ```
#[inline]
#[must_use]
pub fn magnitude_flux(mag: &Array2<f64>) -> NonEmptyVec<f64> {
    let (bins, frames) = mag.dim();
    let mut flux = Vec::with_capacity(frames);
    flux.push(0.0);

    for t in 1..frames {
        let mut acc = 0.0;
        for b in 0..bins {
            let diff = (mag[[b, t]] - mag[[b, t - 1]]).abs();
            acc += diff;
        }
        flux.push(acc);
    }

    // safety:  frames > 1 ensures flux is non-empty

    unsafe { NonEmptyVec::new_unchecked(flux) }
}

/// Computes the complex-domain onset detection function (Duxbury et al., 2003; Bello et
/// al., 2004).
///
/// Each bin is predicted from the two previous frames as a stationary sinusoid: the
/// magnitude of frame `t−1` and the phase advanced by the previous frame-to-frame advance,
/// `X̂[b,t] = |X[b,t−1]|·e^{i(2φ[b,t−1] − φ[b,t−2])}`. The flux is the sum over bins of the
/// distance from the prediction:
///
/// ```text
/// flux[t] = Σ_b |X[b,t] − X̂[b,t]|
/// ```
///
/// A steady tone at any frequency gives zero; a change of level or of phase advance (a new
/// note, a hit, a change of pitch) does not. Decays also count; use
/// [`rectified_complex_flux`] to count only bins whose level has not fallen.
///
/// # Arguments
///
/// * `spec` - 2D complex spectrogram with shape `(frequency_bins, frames)`, for example
///   from [`cqt_frames`](super::complex::cqt_frames)
///
/// # Returns
///
/// Flux values for each frame. The first two frames are 0.0 (the prediction needs two
/// previous frames).
///
/// # Example
///
/// ```rust
/// use ndarray::Array2;
/// use num_complex::Complex;
/// use audio_samples::operations::onset::flux::complex_flux;
///
/// // One bin turning by 0.7 rad per frame at constant magnitude: perfectly predicted.
/// let mut spec = Array2::from_shape_fn((1, 6), |(_, t)| Complex::from_polar(1.0, 0.7 * t as f64));
/// assert!(complex_flux(&spec).iter().all(|&v| v.abs() < 1e-12));
///
/// // The magnitude doubles at frame 4: the flux there is |2 − 1| = 1.
/// for t in 4..6 {
///     spec[[0, t]] *= 2.0;
/// }
/// let flux = complex_flux(&spec);
/// assert!((flux[4] - 1.0).abs() < 1e-12);
/// ```
#[inline]
#[must_use]
pub fn complex_flux(spec: &Array2<Complex<f64>>) -> NonEmptyVec<f64> {
    complex_domain(spec, false)
}

/// Computes the rectified complex-domain onset detection function (Dixon, 2006).
///
/// The sum of [`complex_flux`] taken only over the bins whose magnitude has not fallen,
/// `|X[b,t]| ≥ |X[b,t−1]|`, so a decaying note contributes nothing and only onsets remain:
///
/// ```text
/// flux[t] = Σ_{b : |X[b,t]| ≥ |X[b,t−1]|} |X[b,t] − X̂[b,t]|
/// ```
///
/// # Arguments
///
/// * `spec` - 2D complex spectrogram with shape `(frequency_bins, frames)`
///
/// # Returns
///
/// Flux values for each frame. The first two frames are 0.0.
///
/// # Example
///
/// ```rust
/// use ndarray::Array2;
/// use num_complex::Complex;
/// use audio_samples::operations::onset::flux::{complex_flux, rectified_complex_flux};
///
/// // A tone that halves in level at frame 3 and doubles back at frame 5.
/// let spec = Array2::from_shape_fn((1, 7), |(_, t)| {
///     let level = if (3..5).contains(&t) { 0.5 } else { 1.0 };
///     Complex::from_polar(level, 0.4 * t as f64)
/// });
/// let full = complex_flux(&spec);
/// let rectified = rectified_complex_flux(&spec);
/// assert!(full[3] > 0.4 && rectified[3] == 0.0); // the drop counts only in the full flux
/// assert!((rectified[5] - full[5]).abs() < 1e-12); // the rise counts in both
/// ```
#[inline]
#[must_use]
pub fn rectified_complex_flux(spec: &Array2<Complex<f64>>) -> NonEmptyVec<f64> {
    complex_domain(spec, true)
}

/// The complex-domain sum `Σ_b |X[b,t] − X̂[b,t]|`, over every bin or, if `rectify`, over
/// the bins whose magnitude has not fallen.
fn complex_domain(spec: &Array2<Complex<f64>>, rectify: bool) -> NonEmptyVec<f64> {
    let frames = spec.ncols();
    let mut flux = vec![0.0; frames.max(1)];
    for row in spec.rows() {
        for t in 2..frames {
            let (now, prev, prev2) = (row[t], row[t - 1], row[t - 2]);
            if rectify && now.norm() < prev.norm() {
                continue;
            }
            let predicted = Complex::from_polar(prev.norm(), 2.0 * prev.arg() - prev2.arg());
            flux[t] += (now - predicted).norm();
        }
    }
    // safety: `flux` has at least one element
    unsafe { NonEmptyVec::new_unchecked(flux) }
}
