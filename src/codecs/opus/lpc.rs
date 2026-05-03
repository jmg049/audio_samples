//! Linear Prediction Coding (LPC) primitives for the Opus SILK speech codec mode.
//!
//! ## What
//!
//! Implements the Levinson–Durbin algorithm for computing LPC coefficients from
//! the windowed autocorrelation of a signal frame, together with the matching
//! analysis (residual computation) and synthesis (reconstruction) filters.
//!
//! ## Why
//!
//! SILK, the speech-oriented half of Opus, models the vocal-tract as an all-pole
//! filter `H(z) = G / A(z)` where `A(z)` is the LPC polynomial. The encoder
//! transmits the LPC coefficients and the prediction residual; the decoder runs
//! the synthesis filter to recover the original waveform. This module provides
//! all the building blocks that `silk.rs` needs.
//!
//! ## Round-trip correctness
//!
//! When both encoder and decoder use **zero initial state** at the start of each
//! frame the analysis/synthesis pair forms a perfect inverse:
//!
//! ```text
//! Analysis:  e[n] = x[n] + Σ_{k=0}^{p-1} a[k] · x[n-1-k]
//! Synthesis: y[n] = e[n] − Σ_{k=0}^{p-1} a[k] · y[n-1-k]
//! ```
//!
//! Substituting: `y[0] = e[0] = x[0]`, `y[1] = e[1] − a[0]·y[0] = x[1]`, etc.
//! The reconstruction is exact up to floating-point precision; quantization of
//! the residual is the only lossy step.

/// Small diagonal loading factor applied to `R[0]` for numerical stability.
///
/// Biases the autocorrelation matrix slightly away from singularity, which
/// can occur for signals with near-zero energy. Value follows the SILK
/// reference code convention (0.001% perturbation of the zero-lag energy).
const DIAGONAL_LOADING_EPSILON: f64 = 1e-5;

/// Minimum prediction error to prevent numerical underflow in Levinson–Durbin.
///
/// Clamps the error after each Levinson step. At this scale, the filter is
/// essentially an all-pass and further iterations are numerically meaningless.
const MIN_PREDICTION_ERROR: f64 = 1e-15;

/// Default LPC predictor order used by SILK.
///
/// SILK uses a 16th-order LP filter for narrowband/wideband speech coding.
/// Higher orders capture more spectral detail at increased computational cost.
pub const SILK_LPC_ORDER: usize = 16;

// ── LpcCoefficients ───────────────────────────────────────────────────────────

/// LPC predictor coefficients and prediction error produced by Levinson–Durbin.
///
/// `coeffs[k]` is the coefficient `a[k+1]` in the all-zero analysis filter:
///
/// ```text
/// A(z) = 1 + a[1]·z⁻¹ + a[2]·z⁻² + … + a[p]·z⁻ᵖ
/// ```
///
/// The analysis filter computes `e[n] = x[n] + Σ a[k+1]·x[n-k-1]`, i.e. the
/// coefficient stored at `coeffs[k]` is the gain on `x[n-k-1]`.
#[derive(Debug, Clone)]
pub struct LpcCoefficients {
    /// Predictor coefficients `a[1..=order]`, zero-indexed.
    /// Length equals the order requested in [`lpc_analysis`].
    pub coeffs: Vec<f32>,
    /// Residual prediction error energy after the final Levinson–Durbin step.
    ///
    /// A value near zero indicates the predictor nearly perfectly models the
    /// signal. Always non-negative; protected against underflow to `1e-15`.
    pub prediction_error: f64,
}

// ── Autocorrelation ───────────────────────────────────────────────────────────

/// Computes the autocorrelation of `samples` up to lag `max_lag` (inclusive).
///
/// A Hamming window is applied to the samples before computing correlations.
/// This reduces spectral leakage and improves the conditioning of the Toeplitz
/// system solved by Levinson–Durbin, at the cost of slightly underestimating
/// the true autocorrelation.
///
/// Returns a `Vec<f64>` of length `max_lag + 1` where index `k` holds `R[k]`.
///
/// # Arguments
/// - `samples` – Input signal frame. If empty, returns all-zero autocorrelation.
/// - `max_lag` – Maximum lag to compute. Clamped to `samples.len() − 1`.
#[must_use]
pub fn compute_autocorrelation(samples: &[f32], max_lag: usize) -> Vec<f64> {
    let n = samples.len();
    if n == 0 {
        return vec![0.0; max_lag + 1];
    }

    // Apply a Hamming window for stability (avoids spectral leakage).
    let windowed: Vec<f64> = samples
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let cos_arg = if n > 1 {
                2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64
            } else {
                0.0
            };
            let w = 0.54 - 0.46 * cos_arg.cos();
            x as f64 * w
        })
        .collect();

    let effective_max = max_lag.min(n - 1);
    (0..=effective_max)
        .map(|lag| {
            (0..n - lag)
                .map(|i| windowed[i] * windowed[i + lag])
                .sum::<f64>()
        })
        .collect()
}

// ── Levinson–Durbin ───────────────────────────────────────────────────────────

/// Computes LPC coefficients from autocorrelation using the Levinson–Durbin algorithm.
///
/// Solves the Yule–Walker equations `R · a = −r` for the predictor coefficient
/// vector `a` of length `order`. Returns `None` if the autocorrelation is near
/// zero (silent or near-silent frame).
///
/// A small diagonal loading (`R[0] × 10⁻⁵`) is applied before the recursion to
/// prevent numerical instability in near-singular cases.
///
/// # Arguments
/// - `autocorr` – Autocorrelation values `R[0..=order]`. Must have length ≥ `order + 1`.
/// - `order` – Predictor order.
///
/// # Returns
/// `Some(LpcCoefficients)` on success, `None` for a zero-energy frame.
#[must_use]
pub fn levinson_durbin(autocorr: &[f64], order: usize) -> Option<LpcCoefficients> {
    if autocorr.len() < order + 1 || autocorr[0] < 1e-12 {
        return None;
    }

    // Diagonal loading for numerical stability.
    let mut r = autocorr.to_vec();
    r[0] *= 1.0 + DIAGONAL_LOADING_EPSILON;

    let mut a = vec![0.0f64; order];
    let mut a_prev = vec![0.0f64; order];
    let mut error = r[0];

    for m in 0..order {
        // Compute the reflection coefficient k_m.
        let mut num = r[m + 1];
        for i in 0..m {
            num += a_prev[i] * r[m - i];
        }
        let k = -num / error;

        // Update predictor coefficients.
        for i in 0..m {
            a[i] = a_prev[i] + k * a_prev[m - 1 - i];
        }
        a[m] = k;

        // Update prediction error; clamp against underflow.
        error = (error * (1.0 - k * k)).max(MIN_PREDICTION_ERROR);
        a_prev.clone_from(&a);
    }

    let coeffs = a.iter().map(|&v| v as f32).collect();
    Some(LpcCoefficients { coeffs, prediction_error: error })
}

// ── High-level analysis entry point ──────────────────────────────────────────

/// Performs LPC analysis on a signal frame.
///
/// Computes the windowed autocorrelation up to `order` lags and solves
/// Levinson–Durbin to obtain the predictor coefficients. If the frame is
/// silent or too short for the requested order, a zero-coefficient predictor
/// is returned.
///
/// The effective order is clamped to `min(order, samples.len() / 2)` so
/// that the autocorrelation matrix always has enough data to be well-posed.
///
/// # Arguments
/// - `samples` – Input signal frame. Must be non-empty.
/// - `order` – Predictor order. Use [`SILK_LPC_ORDER`] for SILK-standard quality.
#[must_use]
pub fn lpc_analysis(samples: &[f32], order: usize) -> LpcCoefficients {
    let effective_order = order.min(samples.len() / 2).max(1);
    let autocorr = compute_autocorrelation(samples, effective_order);
    levinson_durbin(&autocorr, effective_order).unwrap_or_else(|| LpcCoefficients {
        coeffs: vec![0.0; effective_order],
        prediction_error: 1.0,
    })
}

// ── Analysis filter ───────────────────────────────────────────────────────────

/// Applies the LPC analysis (all-zero) filter to produce the prediction residual.
///
/// Computes: `e[n] = x[n] + Σ_{k=0}^{order−1} coeffs[k] · x[n−1−k]`
///
/// Uses **zero initial state** (all past inputs are treated as 0 before the
/// frame starts). The decoder's [`lpc_synthesis`] uses the same convention,
/// which guarantees a perfect round-trip within a single frame.
///
/// # Arguments
/// - `samples` – Input signal frame.
/// - `coeffs` – LPC coefficients from [`lpc_analysis`].
///
/// # Returns
/// A `Vec<f32>` of the same length as `samples` containing the residual.
#[must_use]
pub fn lpc_residual(samples: &[f32], coeffs: &LpcCoefficients) -> Vec<f32> {
    let order = coeffs.coeffs.len();
    let mut residual = Vec::with_capacity(samples.len());
    for n in 0..samples.len() {
        let mut sum = 0.0f32;
        for k in 0..order.min(n) {
            sum += coeffs.coeffs[k] * samples[n - 1 - k];
        }
        residual.push(samples[n] + sum);
    }
    residual
}

// ── Synthesis filter ──────────────────────────────────────────────────────────

/// Applies the LPC synthesis (all-pole) filter to reconstruct the signal.
///
/// Computes: `y[n] = e[n] − Σ_{k=0}^{order−1} coeffs[k] · y[n−1−k]`
///
/// This is the exact inverse of [`lpc_residual`] when both use zero initial
/// state. The only reconstruction error comes from quantizing the residual
/// in the SILK encode step.
///
/// # Arguments
/// - `residual` – Prediction residual from [`lpc_residual`] (or its quantised
///   approximation from the SILK encoder).
/// - `coeffs` – LPC coefficients matching those used during analysis.
///
/// # Returns
/// A `Vec<f32>` of the same length as `residual`.
#[must_use]
pub fn lpc_synthesis(residual: &[f32], coeffs: &LpcCoefficients) -> Vec<f32> {
    let order = coeffs.coeffs.len();
    let mut output = Vec::with_capacity(residual.len());
    for n in 0..residual.len() {
        let mut sum = 0.0f32;
        for k in 0..order.min(n) {
            sum += coeffs.coeffs[k] * output[n - 1 - k];
        }
        output.push(residual[n] - sum);
    }
    output
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Verifies that the analysis/synthesis pair is a perfect inverse (no
    /// quantization) when both use zero initial state.
    #[test]
    fn round_trip_no_quantization() {
        let samples: Vec<f32> = (0..64)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();

        let coeffs = lpc_analysis(&samples, SILK_LPC_ORDER);
        let residual = lpc_residual(&samples, &coeffs);
        let recovered = lpc_synthesis(&residual, &coeffs);

        for (orig, rec) in samples.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() < 1e-4,
                "round-trip error {:.2e} > 1e-4",
                (orig - rec).abs()
            );
        }
    }

    /// Confirms Levinson–Durbin returns `None` for a zero-energy frame.
    #[test]
    fn levinson_silent_frame() {
        let autocorr = vec![0.0f64; SILK_LPC_ORDER + 1];
        assert!(levinson_durbin(&autocorr, SILK_LPC_ORDER).is_none());
    }
}
