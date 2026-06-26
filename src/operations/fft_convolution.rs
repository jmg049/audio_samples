//! 1-D FFT-based linear convolution and regularised spectral deconvolution.
//!
//! These helpers live here — rather than being re-exported from the
//! [`spectrograms`] crate — because the published `spectrograms` (1.4.3) only
//! ships a *2-D* image convolution (`convolve_fft`). It has no 1-D
//! `fft_convolve` / `fft_deconvolve`. Audio convolution/deconvolution
//! (`AudioProcessing::convolve` / `deconvolve` and the FIR FFT fast-path) need
//! the 1-D versions, so we implement them locally on top of `rustfft` (already
//! a dependency enabled by the `transforms` feature).
//!
//! Both functions operate in `f64` for headroom and match a direct
//! time-domain convolution to floating-point tolerance.

use non_empty_slice::{NonEmptySlice, NonEmptyVec};
use num_complex::Complex;
use rustfft::FftPlanner;

use crate::{AudioSampleError, AudioSampleResult, ParameterError};

/// Wraps a non-empty `Vec<f64>` into a [`NonEmptyVec`], mapping the
/// (unreachable for these functions) empty case to a `Parameter` error.
fn into_non_empty(v: Vec<f64>) -> AudioSampleResult<NonEmptyVec<f64>> {
    NonEmptyVec::new(v).map_err(|_| {
        AudioSampleError::Parameter(ParameterError::invalid_value(
            "fft_convolution",
            "empty result",
        ))
    })
}

/// Forward-FFTs a real signal zero-padded to length `len` into a complex
/// spectrum (unnormalised, as `rustfft` produces).
fn forward_padded(planner: &mut FftPlanner<f64>, signal: &[f64], len: usize) -> Vec<Complex<f64>> {
    let mut buf: Vec<Complex<f64>> = signal.iter().map(|&s| Complex::new(s, 0.0)).collect();
    buf.resize(len, Complex::new(0.0, 0.0));
    let fft = planner.plan_fft_forward(len);
    fft.process(&mut buf);
    buf
}

/// Full *linear* convolution of `a` and `b` via FFT.
///
/// The output length is `N = a.len() + b.len() - 1`. The transform length is
/// `L = N.next_power_of_two()`; both inputs are zero-padded to `L`, transformed,
/// multiplied bin-by-bin, inverse-transformed, normalised by `L` (rustfft is
/// unnormalised), and the first `N` real parts are returned.
pub(crate) fn fft_convolve(
    a: &NonEmptySlice<f64>,
    b: &NonEmptySlice<f64>,
) -> AudioSampleResult<NonEmptyVec<f64>> {
    let a = a.as_slice();
    let b = b.as_slice();

    let n = a.len() + b.len() - 1;
    let l = n.next_power_of_two();

    let mut planner = FftPlanner::<f64>::new();
    let mut spec_a = forward_padded(&mut planner, a, l);
    let spec_b = forward_padded(&mut planner, b, l);

    for (xa, xb) in spec_a.iter_mut().zip(spec_b.iter()) {
        *xa *= *xb;
    }

    let ifft = planner.plan_fft_inverse(l);
    ifft.process(&mut spec_a);

    let scale = 1.0 / l as f64;
    let out: Vec<f64> = spec_a.iter().take(n).map(|c| c.re * scale).collect();
    into_non_empty(out)
}

/// Regularised spectral deconvolution recovering the system that, convolved
/// with `b`, produced `a`.
///
/// Output length is `out_len = a.len().saturating_sub(b.len()) + 1` (always
/// `>= 1`). The transform length is `L = a.len().next_power_of_two()` (so
/// `L >= a.len()`, making circular convolution equal to linear convolution for
/// the recorded signal). Both inputs are zero-padded to `L` and transformed
/// into `A`, `B`. Each bin is recovered via Wiener-style division
/// `X = A * conj(B) / (|B|^2 + reg)`, inverse-transformed, normalised by `L`,
/// and the first `out_len` real parts are returned.
///
/// With `reg == 0.0` this is exact division `A / B` wherever `B` has no zero
/// bins; a literally-zero denominator is guarded with a tiny epsilon purely to
/// avoid `NaN` without perturbing well-conditioned results.
pub(crate) fn fft_deconvolve(
    a: &NonEmptySlice<f64>,
    b: &NonEmptySlice<f64>,
    regularization: f64,
) -> AudioSampleResult<NonEmptyVec<f64>> {
    let a = a.as_slice();
    let b = b.as_slice();

    let out_len = a.len().saturating_sub(b.len()) + 1;
    let l = a.len().next_power_of_two();

    let mut planner = FftPlanner::<f64>::new();
    let spec_a = forward_padded(&mut planner, a, l);
    let spec_b = forward_padded(&mut planner, b, l);

    let mut spec_x: Vec<Complex<f64>> = spec_a
        .iter()
        .zip(spec_b.iter())
        .map(|(&xa, &xb)| {
            let denom = xb.norm_sqr() + regularization;
            // Guard a literally-zero denominator (reg == 0 and |B| == 0) to
            // avoid NaN. The epsilon is far below the 1e-6 test tolerance.
            let denom = if denom == 0.0 { f64::EPSILON } else { denom };
            xa * xb.conj() / denom
        })
        .collect();

    let ifft = planner.plan_fft_inverse(l);
    ifft.process(&mut spec_x);

    let scale = 1.0 / l as f64;
    let out: Vec<f64> = spec_x.iter().take(out_len).map(|c| c.re * scale).collect();
    into_non_empty(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive O(n*m) direct linear convolution for cross-checking.
    fn naive_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
        let n = a.len() + b.len() - 1;
        let mut out = vec![0.0; n];
        for (i, &av) in a.iter().enumerate() {
            for (j, &bv) in b.iter().enumerate() {
                out[i + j] += av * bv;
            }
        }
        out
    }

    fn ne(v: &[f64]) -> &NonEmptySlice<f64> {
        NonEmptySlice::new(v).unwrap()
    }

    #[test]
    fn convolve_matches_naive_small() {
        let a = [1.0, 2.0, 3.0, -1.0, 0.5];
        let b = [0.5, -0.25, 1.0];
        let got = fft_convolve(ne(&a), ne(&b)).unwrap();
        let want = naive_convolve(&a, &b);
        assert_eq!(got.as_slice().len(), want.len());
        for (g, w) in got.as_slice().iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-9, "got {g}, want {w}");
        }
    }

    #[test]
    fn convolve_matches_naive_assorted() {
        let cases: &[(&[f64], &[f64])] = &[
            (&[1.0], &[1.0, 2.0, 3.0]),
            (&[0.3, -0.7, 1.1, 2.0, -0.4, 0.9], &[1.0, 0.0, -1.0]),
            (&[2.0, 0.0, 0.0, 5.0], &[0.0, 1.0, 0.0, 0.0, 3.0]),
        ];
        for (a, b) in cases {
            let got = fft_convolve(ne(a), ne(b)).unwrap();
            let want = naive_convolve(a, b);
            assert_eq!(got.as_slice().len(), want.len());
            for (g, w) in got.as_slice().iter().zip(want.iter()) {
                assert!((g - w).abs() < 1e-9, "a={a:?} b={b:?}: got {g}, want {w}");
            }
        }
    }

    #[test]
    fn deconvolve_round_trip() {
        let excitation = [1.0, 0.7, -0.3, 0.2, 0.9, -0.5, 0.1, 0.4];
        let system = [0.0, 0.0, 1.0, 0.5];
        let recorded = fft_convolve(ne(&excitation), ne(&system)).unwrap();
        let recorded = recorded.as_slice().to_vec();
        let recovered = fft_deconvolve(ne(&recorded), ne(&excitation), 0.0).unwrap();
        let r = recovered.as_slice();
        assert_eq!(r.len(), system.len());
        for (i, want) in system.iter().enumerate() {
            assert!(
                (r[i] - want).abs() < 1e-6,
                "tap {i}: got {}, want {want}",
                r[i]
            );
        }
    }
}
