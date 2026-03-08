//! Spectral flux computation kernels.
//!
//! This module provides low-level functions for computing spectral flux—the rate of change in
//! a signal's frequency spectrum over time. Spectral flux is a fundamental feature for onset
//! detection, as sudden increases in spectral energy or magnitude typically correspond to
//! note onsets, percussive hits, or other transient events.
//!
//! Four spectral flux variants are provided, each with different characteristics:
//! - **Energy flux**: Emphasizes large amplitude changes (quadratic sensitivity)
//! - **Magnitude flux**: Linear sensitivity to amplitude changes
//! - **Complex flux**: Incorporates both magnitude and phase information
//! - **Rectified complex flux**: Focuses on magnitude increases with phase awareness
//!
//! All flux functions operate on 2D spectrograms with shape `(frequency_bins, frames)` and
//! return a 1D time series of flux values, one per frame. The first frame always has zero flux.

use ndarray::Array2;
use non_empty_slice::NonEmptyVec;
use num_complex::Complex;

/// Computes energy-based spectral flux.
///
/// Calculates the sum of positive energy differences across all frequency bins:
/// `flux[t] = Σ max(0, |X[b,t]|² - |X[b,t-1]|²)` for all bins b.
///
/// Energy flux emphasizes large amplitude changes due to squaring, making it particularly
/// effective for detecting percussive onsets with sharp transients.
///
/// # Arguments
///
/// * `mag` — 2D magnitude spectrogram with shape `(frequency_bins, frames)`
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
/// `flux[t] = Σ |mag[b,t] - mag[b,t-1]|` for all bins b.
///
/// Magnitude flux has linear sensitivity to amplitude changes, making it more sensitive to
/// subtle onsets compared to energy flux. Good for tonal instruments with gradual attacks.
///
/// # Arguments
///
/// * `mag` — 2D magnitude spectrogram with shape `(frequency_bins, frames)`
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
/// Calculates the Euclidean distance between consecutive complex spectra:
/// `flux[t] = Σ |X[b,t] - X[b,t-1]|` for all bins b, where X is complex-valued.
///
/// Complex flux incorporates both magnitude and phase changes, providing the most complete
/// measure of spectral change. More computationally expensive but most accurate for
/// polyphonic music with complex timbres.
///
/// # Arguments
///
/// * `spec` — 2D complex spectrogram with shape `(frequency_bins, frames)`
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
/// `flux[t] = Σ max(0, |X[b,t]| - |X[b,t-1]|)` for all bins b.
///
/// This variant considers phase information when computing the norm but only keeps positive
/// magnitude changes. Balances the robustness of complex methods with the interpretability
/// of rectified magnitude-based approaches.
///
/// # Arguments
///
/// * `spec` — 2D complex spectrogram with shape `(frequency_bins, frames)`
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
