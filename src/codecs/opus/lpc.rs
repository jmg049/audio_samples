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

use std::num::{NonZeroU32, NonZeroUsize};

use non_empty_slice::NonEmptyVec;

use crate::AudioSamples;
use crate::operations::traits::AudioStatistics;

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
/// A Hamming window is applied to the samples before computing correlations,
/// reducing spectral leakage and improving the conditioning of the Toeplitz
/// system solved by Levinson–Durbin.
///
/// The implementation delegates to [`AudioStatistics::autocorrelation`], which
/// uses an FFT-based O(n log n) algorithm. The trait method returns a normalised
/// (unbiased) estimator; this function multiplies each lag-k value back by
/// `(n − k)` to restore the **biased** sum-product form. The biased form
/// guarantees a positive semi-definite Toeplitz matrix, which is required for
/// Levinson–Durbin to remain numerically stable.
///
/// Returns a `Vec<f64>` of length `min(max_lag, n − 1) + 1` where index `k`
/// holds `R[k]`.
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

    // Apply a Hamming window before autocorrelation for spectral stability.
    let windowed: Vec<f32> = samples
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let w = if n > 1 {
                let t = 2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64;
                0.54 - 0.46 * t.cos()
            } else {
                1.0
            };
            (w * x as f64) as f32
        })
        .collect();

    let effective_max = max_lag.min(n - 1);

    // Wrap the windowed frame as a temporary mono AudioSamples so we can
    // call AudioStatistics::autocorrelation (FFT-based, O(n log n)).
    // Sample rate is irrelevant for autocorrelation; 1 Hz is a valid placeholder.
    let ne = NonEmptyVec::new(windowed).expect("n > 0 checked above");
    let audio = AudioSamples::<'static, f32>::from_mono_vec(ne, NonZeroU32::new(1).expect("1 > 0"));

    if let Some(lag_nz) = NonZeroUsize::new(effective_max) {
        if let Some(normalized) = audio.autocorrelation(lag_nz) {
            // `autocorrelation` divides each lag-k value by (n − k), yielding
            // the unbiased estimator. Multiplying back recovers the biased form
            // whose ratios R[k] / R[0] match the standard Yule–Walker equations.
            return (0..=effective_max)
                .map(|lag| normalized[lag] * (n - lag) as f64)
                .collect();
        }
    }

    // Fallback: effective_max == 0 (single-sample frame) or FFT failure.
    // Recover the windowed data from the contiguous mono storage and compute directly.
    // `from_mono_vec` always produces contiguous storage, so `as_slice()` succeeds.
    let w64: Vec<f64> = audio
        .as_slice()
        .expect("from_mono_vec is always contiguous")
        .iter()
        .map(|&x| x as f64)
        .collect();
    (0..=effective_max)
        .map(|lag| (0..n - lag).map(|i| w64[i] * w64[i + lag]).sum::<f64>())
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
        // Swap rather than clone: a_prev needs the current a for the next
        // iteration; a will be fully overwritten then, so stale values are fine.
        std::mem::swap(&mut a, &mut a_prev);
    }

    let coeffs = a.iter().map(|&v| v as f32).collect();
    Some(LpcCoefficients {
        coeffs,
        prediction_error: error,
    })
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

// ── Stateful analysis filter ──────────────────────────────────────────────────

/// Applies the LPC analysis (all-zero) filter with cross-frame state.
///
/// Identical to [`lpc_residual`] except that past-input samples from before
/// the frame start are drawn from `state` rather than being zero. This
/// eliminates boundary artefacts when encoding consecutive frames.
///
/// `state` must contain the last `order` input samples from the preceding frame
/// (zero-padded on the left if fewer are available). On return it holds the
/// last `order` samples of `samples`.
///
/// # Arguments
/// - `samples` – Input signal frame.
/// - `coeffs` – LPC coefficients from [`lpc_analysis`].
/// - `state` – Cross-frame input history, updated in place.
#[must_use]
pub fn lpc_residual_stateful(
    samples: &[f32],
    coeffs: &LpcCoefficients,
    state: &mut Vec<f32>,
) -> Vec<f32> {
    let order = coeffs.coeffs.len();
    normalise_state(state, order);

    let mut residual = Vec::with_capacity(samples.len());
    for n in 0..samples.len() {
        let mut sum = 0.0f32;
        for k in 0..order {
            let pos = n as isize - 1 - k as isize;
            let x = if pos >= 0 {
                samples[pos as usize]
            } else {
                state[(order as isize + pos) as usize]
            };
            sum += coeffs.coeffs[k] * x;
        }
        residual.push(samples[n] + sum);
    }

    update_state(state, samples, order);
    residual
}

// ── Stateful synthesis filter ─────────────────────────────────────────────────

/// Applies the LPC synthesis (all-pole) filter with cross-frame state.
///
/// Identical to [`lpc_synthesis`] except that past-output samples from before
/// the frame start are drawn from `state`. On return `state` holds the last
/// `order` output samples.
///
/// # Arguments
/// - `residual` – Prediction residual (quantised or exact).
/// - `coeffs` – LPC coefficients matching those used during analysis.
/// - `state` – Cross-frame output history, updated in place.
#[must_use]
pub fn lpc_synthesis_stateful(
    residual: &[f32],
    coeffs: &LpcCoefficients,
    state: &mut Vec<f32>,
) -> Vec<f32> {
    let order = coeffs.coeffs.len();
    normalise_state(state, order);

    let mut output = Vec::with_capacity(residual.len());
    for (n, &r) in residual.iter().enumerate() {
        let mut sum = 0.0f32;
        for k in 0..order {
            let pos = n as isize - 1 - k as isize;
            let y = if pos >= 0 {
                output[pos as usize]
            } else {
                state[(order as isize + pos) as usize]
            };
            sum += coeffs.coeffs[k] * y;
        }
        output.push(r - sum);
    }

    update_state(state, &output, order);
    output
}

// ── Pitch analysis (Long-Term Prediction) ────────────────────────────────────

/// Estimates the pitch period and long-term prediction (LTP) gain for a frame.
///
/// Searches the normalized autocorrelation of `samples` in the lag range
/// `[T_min, T_max]` where:
///
/// - `T_min = sample_rate / 500` (≈ 500 Hz maximum pitch)
/// - `T_max = min(sample_rate / 50, frame_length / 2)` (≈ 50 Hz minimum pitch)
///
/// Returns `Some((lag, gain))` where `lag` is the dominant pitch period and
/// `gain = R[lag] / R[0]` clamped to `[0.0, 0.9]` for filter stability.
/// Returns `None` when the frame is too short, the signal energy is negligible,
/// or no clear periodicity is found (gain < 0.3).
///
/// # Arguments
/// - `samples` – Signal frame (typically the LPC residual).
/// - `sample_rate` – Sample rate of the signal in Hz.
#[must_use]
pub fn estimate_pitch(samples: &[f32], sample_rate: u32) -> Option<(usize, f32)> {
    let n = samples.len();
    let fs = sample_rate as usize;

    let t_min = (fs / 500).max(2);
    let t_max = (fs / 50).min(n / 2);

    if t_max <= t_min || n <= t_max {
        return None;
    }

    let r0: f64 = samples.iter().map(|&x| (x as f64).powi(2)).sum();
    if r0 < 1e-10 {
        return None;
    }

    let (best_lag, best_r) = (t_min..=t_max)
        .map(|lag| {
            let r: f64 = (0..n - lag)
                .map(|i| samples[i] as f64 * samples[i + lag] as f64)
                .sum();
            (lag, r / r0)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;

    if best_r < 0.3 {
        return None;
    }

    Some((best_lag, (best_r as f32).clamp(0.0, 0.9)))
}

/// Applies the LTP analysis (FIR) filter: `d[n] = e[n] − gain × e[n − lag]`.
///
/// Uses zero initial state — samples before the frame start are treated as 0.
///
/// # Arguments
/// - `samples` – Short-term LPC residual.
/// - `lag` – Pitch period in samples.
/// - `gain` – LTP gain. Should be in `[0.0, 0.9]` for stability.
#[must_use]
pub fn ltp_residual(samples: &[f32], lag: usize, gain: f32) -> Vec<f32> {
    samples
        .iter()
        .enumerate()
        .map(|(n, &e)| e - gain * if n >= lag { samples[n - lag] } else { 0.0 })
        .collect()
}

/// Applies the LTP synthesis (IIR) filter: `e[n] = d[n] + gain × e[n − lag]`.
///
/// The inverse of [`ltp_residual`] when both use zero initial state.
/// Stable when `gain < 1.0`.
///
/// # Arguments
/// - `residual` – LTP residual (from the SILK encoder).
/// - `lag` – Pitch period in samples.
/// - `gain` – LTP gain matching the value used during encoding.
#[must_use]
pub fn ltp_synthesis(residual: &[f32], lag: usize, gain: f32) -> Vec<f32> {
    let mut output = vec![0.0f32; residual.len()];
    for n in 0..residual.len() {
        let prev = if n >= lag { output[n - lag] } else { 0.0 };
        output[n] = residual[n] + gain * prev;
    }
    output
}

// ── State helpers (private) ───────────────────────────────────────────────────

/// Ensures `state` has exactly `order` elements, zero-padding on the left.
fn normalise_state(state: &mut Vec<f32>, order: usize) {
    while state.len() < order {
        state.insert(0, 0.0);
    }
    if state.len() > order {
        let excess = state.len() - order;
        state.drain(0..excess);
    }
}

/// Updates `state` to hold the last `order` elements of `samples`.
fn update_state(state: &mut Vec<f32>, samples: &[f32], order: usize) {
    let n = samples.len();
    if n >= order {
        state.clear();
        state.extend_from_slice(&samples[n - order..]);
    } else {
        // Keep the tail of the old state and append the new samples.
        let tail: Vec<f32> = state[n..].to_vec();
        state.clear();
        state.extend_from_slice(&tail);
        state.extend_from_slice(samples);
    }
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

    /// Stateful analysis/synthesis is a perfect inverse within a single frame
    /// (same as the stateless pair when state starts at zero).
    #[test]
    fn stateful_round_trip_single_frame() {
        let samples: Vec<f32> = (0..64)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let coeffs = lpc_analysis(&samples, SILK_LPC_ORDER);
        let mut enc_state = Vec::new();
        let mut dec_state = Vec::new();
        let residual = lpc_residual_stateful(&samples, &coeffs, &mut enc_state);
        let recovered = lpc_synthesis_stateful(&residual, &coeffs, &mut dec_state);
        for (o, r) in samples.iter().zip(recovered.iter()) {
            assert!(
                (o - r).abs() < 1e-4,
                "stateful round-trip error {:.2e}",
                (o - r).abs()
            );
        }
    }

    /// Stateful filters carry state correctly across two consecutive frames so
    /// the reconstruction is continuous at the frame boundary.
    #[test]
    fn stateful_cross_frame_continuity() {
        let n = 64usize;
        let all: Vec<f32> = (0..n * 2)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let coeffs0 = lpc_analysis(&all[..n], SILK_LPC_ORDER);
        let coeffs1 = lpc_analysis(&all[n..], SILK_LPC_ORDER);

        let mut enc_state = Vec::new();
        let mut dec_state = Vec::new();
        let res0 = lpc_residual_stateful(&all[..n], &coeffs0, &mut enc_state);
        let res1 = lpc_residual_stateful(&all[n..], &coeffs1, &mut enc_state);
        let rec0 = lpc_synthesis_stateful(&res0, &coeffs0, &mut dec_state);
        let rec1 = lpc_synthesis_stateful(&res1, &coeffs1, &mut dec_state);

        let mut recovered = rec0;
        recovered.extend(rec1);
        for (o, r) in all.iter().zip(recovered.iter()) {
            assert!(
                (o - r).abs() < 1e-3,
                "cross-frame error {:.2e}",
                (o - r).abs()
            );
        }
    }

    /// LTP synthesis is the exact inverse of LTP residual (zero initial state).
    #[test]
    fn ltp_round_trip() {
        let samples: Vec<f32> = (0..200)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();
        let lag = 100;
        let gain = 0.7;
        let residual = ltp_residual(&samples, lag, gain);
        let recovered = ltp_synthesis(&residual, lag, gain);
        for (o, r) in samples.iter().zip(recovered.iter()) {
            assert!(
                (o - r).abs() < 1e-4,
                "LTP round-trip error {:.2e}",
                (o - r).abs()
            );
        }
    }

    /// estimate_pitch finds the correct period for a periodic signal.
    #[test]
    fn pitch_detection_sine() {
        let freq_hz = 220.0_f32;
        let sample_rate = 44100_u32;
        let expected_lag = (sample_rate as f32 / freq_hz).round() as usize; // 200 samples
        let samples: Vec<f32> = (0..882)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate as f32).sin())
            .collect();
        let result = estimate_pitch(&samples, sample_rate);
        assert!(result.is_some(), "pitch not detected for {freq_hz} Hz sine");
        let (lag, gain) = result.unwrap();
        // Allow ±2 samples tolerance.
        assert!(
            lag.abs_diff(expected_lag) <= 2,
            "detected lag {lag} expected ~{expected_lag}"
        );
        assert!(gain > 0.3, "LTP gain {gain:.2} too low");
    }
}
