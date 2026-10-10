//! Spectral flux computation kernels.
//!
//! This module provides low-level functions for computing spectral flux-the rate of change in
//! a signal's frequency spectrum over time. Spectral flux is a fundamental feature for onset
//! detection, as sudden increases in spectral energy or magnitude typically correspond to
//! note onsets, percussive hits, or other transient events.
//!
//! Four spectral flux variants are provided, each with different characteristics:
//! - **Energy flux** ([`energy_flux`]): `Σ_b max(0, X[b,t] − X[b,t−1])` on the values it is
//!   given (no squaring); with a magnitude spectrogram this is the standard half-wave-rectified
//!   spectral flux
//! - **Magnitude flux** ([`magnitude_flux`]): `Σ_b |X[b,t] − X[b,t−1]|` (absolute, not
//!   rectified: decreases count too)
//! - **Complex flux** ([`complex_flux`]): `Σ_b |X[b,t] − X[b,t−1]|` on complex values, so phase
//!   changes contribute
//! - **Rectified complex flux** ([`rectified_complex_flux`]): `Σ_b max(0, |X[b,t]| − |X[b,t−1]|)`;
//!   phase is discarded, so it equals energy flux on `|X|`
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
/// ```rust,ignore
/// use ndarray::Array2;
/// use audio_samples::operations::onset::flux::energy_flux;
///
/// let mag = Array2::from_shape_vec((128, 100), vec![0.0; 12800]).unwrap();
/// let flux = energy_flux(&mag);
/// assert_eq!(flux.len().get(), 100);
/// assert_eq!(flux[0], 0.0);
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
/// ```rust,ignore
/// use ndarray::Array2;
/// use audio_samples::operations::onset::flux::magnitude_flux;
///
/// let mag = Array2::from_shape_vec((128, 100), vec![0.0; 12800]).unwrap();
/// let flux = magnitude_flux(&mag);
/// assert_eq!(flux.len().get(), 100);
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

/// Computes complex domain spectral flux using magnitude and phase.
///
/// Calculates the sum over bins of the distance between consecutive complex values:
/// `flux[t] = Σ_b |X[b,t] − X[b,t−1]|`, where X is complex-valued (an L1 sum of per-bin
/// moduli, not a Euclidean norm over bins). Because a stationary sinusoid's phase advances
/// each frame, even a steady tone gives non-zero complex flux.
///
/// Complex flux incorporates both magnitude and phase changes, providing the most complete
/// measure of spectral change. More computationally expensive but most accurate for
/// polyphonic music with complex timbres.
///
/// # Arguments
///
/// * `spec` - 2D complex spectrogram with shape `(frequency_bins, frames)`
///
/// # Returns
///
/// Flux values for each frame. The first frame is always 0.0 (no previous frame to compare).
///
/// # Example
///
/// ```rust,ignore
/// use ndarray::Array2;
/// use num_complex::Complex;
/// use audio_samples::operations::onset::flux::complex_flux;
///
/// let spec = Array2::from_elem((128, 100), Complex::new(0.0, 0.0));
/// let flux = complex_flux(&spec);
/// assert_eq!(flux.len().get(), 100);
/// ```
#[inline]
#[must_use]
pub fn complex_flux(spec: &Array2<Complex<f64>>) -> NonEmptyVec<f64> {
    let (bins, frames) = spec.dim();
    let mut flux = Vec::with_capacity(frames);
    flux.push(0.0);

    for t in 1..frames {
        let mut acc = 0.0;
        for b in 0..bins {
            let diff = spec[[b, t]] - spec[[b, t - 1]];
            acc += diff.norm();
        }
        flux.push(acc);
    }
    // safety: frames > 1 ensures flux is non-empty

    unsafe { NonEmptyVec::new_unchecked(flux) }
}

/// Computes rectified complex domain spectral flux.
///
/// Calculates the sum of positive magnitude differences:
/// `flux[t] = Σ_b max(0, |X[b,t]| − |X[b,t−1]|)`.
///
/// Only the moduli |X| are used; phase is discarded. The result is numerically identical to
/// [`energy_flux`] applied to `|X|` (half-wave-rectified magnitude flux).
///
/// # Arguments
///
/// * `spec` - 2D complex spectrogram with shape `(frequency_bins, frames)`
///
/// # Returns
///
/// Flux values for each frame. The first frame is always 0.0 (no previous frame to compare).
///
/// # Example
///
/// ```rust,ignore
/// use ndarray::Array2;
/// use num_complex::Complex;
/// use audio_samples::operations::onset::flux::rectified_complex_flux;
///
/// let spec = Array2::from_elem((128, 100), Complex::new(1.0, 0.0));
/// let flux = rectified_complex_flux(&spec);
/// assert_eq!(flux.len().get(), 100);
/// ```
#[inline]
#[must_use]
pub fn rectified_complex_flux(spec: &Array2<Complex<f64>>) -> NonEmptyVec<f64> {
    let (bins, frames) = spec.dim();
    let mut flux = Vec::with_capacity(frames);
    flux.push(0.0);

    for t in 1..frames {
        let mut acc = 0.0;
        for b in 0..bins {
            let diff = spec[[b, t]].norm() - spec[[b, t - 1]].norm();
            if diff > 0.0 {
                acc += diff;
            }
        }
        flux.push(acc);
    }

    // safety: frames > 1 ensures flux is non-empty

    unsafe { NonEmptyVec::new_unchecked(flux) }
}
