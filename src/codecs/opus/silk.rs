//! SILK speech codec: LPC-based frame encode/decode for Opus's speech mode.
//!
//! ## What
//!
//! Implements one encode/decode cycle for a single Opus SILK audio frame using
//! the primitives from [`crate::codecs::opus::lpc`]:
//!
//! 1. **Encode** — LPC analysis → prediction residual → gain normalisation →
//!    16-bit residual quantisation.
//! 2. **Decode** — 16-bit dequantisation → scale by gain → LPC synthesis.
//!
//! ## Why
//!
//! SILK exploits the predictability of voiced speech: the LPC predictor removes
//! the spectral envelope, leaving a spectrally flat residual that is far smaller
//! in energy than the original. Quantising the residual at high resolution
//! (16 bits) achieves very high SNR for speech at low bitrates.
//!
//! For generic audio (music, noise) the LPC provides little prediction gain and
//! [`crate::codecs::opus::celt`] should be preferred via the auto-detection in
//! [`crate::codecs::opus::mode::detect_mode`].
//!
//! ## Sketch limitations
//!
//! - Each frame is encoded independently with **zero initial LPC state**.
//!   A complete implementation would carry the LPC filter state across frames to
//!   prevent boundary artifacts.
//! - LPC coefficients are stored as `f32` values. Real SILK transmits Line
//!   Spectral Frequency (LSF) parameters quantised to a codebook; the IO layer
//!   (`audio_samples_io`) is responsible for that packing.
//! - No pitch analysis or adaptive codebook is implemented.

use crate::{AudioSampleError, AudioSampleResult, ParameterError};

use super::lpc::{LpcCoefficients, SILK_LPC_ORDER, lpc_analysis, lpc_residual, lpc_synthesis};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Scale factor for 16-bit residual quantisation.
///
/// The normalised residual (in `[−1, 1]`) is multiplied by this constant before
/// rounding to `i16`, and divided by the same constant during dequantisation.
const SILK_RESIDUAL_SCALE: f32 = 32_767.0;

// ── SilkEncodedFrame ──────────────────────────────────────────────────────────

/// One SILK-encoded audio frame.
///
/// Stores the LPC coefficients, a 16-bit quantised prediction residual, and the
/// gain used to normalise the residual before quantisation. This struct is
/// self-contained: the decoder needs no external side-channel information.
///
/// ## Round-trip quality
///
/// The only error in the encode/decode round-trip is residual quantisation.
/// With `gain = max(|e[n]|)` the maximum per-sample error is `gain / 32767`.
/// For a 440 Hz sine at amplitude 0.5:
///
/// - The LPC residual energy is close to floating-point noise (≈ 10⁻⁵).
/// - Gain ≈ 10⁻⁵, so maximum error ≈ 3 × 10⁻¹⁰.
/// - Expected SNR > 50 dB.
///
/// For white noise the LPC provides no prediction gain (residual ≈ input),
/// but 16-bit quantisation still gives ≈ 90 dB dynamic range.
#[derive(Debug, Clone)]
pub struct SilkEncodedFrame {
    /// LPC predictor coefficients and prediction error from Levinson–Durbin.
    pub lpc_coeffs: LpcCoefficients,
    /// Residual `e[n]` normalised to `[−1, 1]` and quantised to 16-bit integers.
    pub residual_quantized: Vec<i16>,
    /// Peak absolute value of the prediction residual before normalisation.
    ///
    /// The decoder multiplies the dequantised residual by this value to restore
    /// the original amplitude scale.
    pub gain: f32,
    /// Number of PCM samples in the original frame.
    pub n_samples: usize,
}

// ── silk_encode_frame ─────────────────────────────────────────────────────────

/// Encodes a single SILK audio frame.
///
/// Steps:
/// 1. Compute an LPC predictor of order [`SILK_LPC_ORDER`] (or less for short frames).
/// 2. Apply the analysis filter to obtain the prediction residual.
/// 3. Compute `gain = max(|e[n]|)` and normalise: `e_norm[n] = e[n] / gain`.
/// 4. Quantise the normalised residual to 16-bit integers:
///    `q[n] = round(e_norm[n] × 32767)`.
///
/// # Arguments
/// - `samples` – PCM samples for the frame (f32, any amplitude range).
///
/// # Errors
/// Returns [`AudioSampleError::Parameter`] if `samples` is empty.
pub fn silk_encode_frame(samples: &[f32]) -> AudioSampleResult<SilkEncodedFrame> {
    if samples.is_empty() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "samples",
            "SILK frame must contain at least one sample",
        )));
    }

    let n_samples = samples.len();

    // LPC analysis — order clamped for very short frames.
    let lpc_coeffs = lpc_analysis(samples, SILK_LPC_ORDER);

    // Analysis filter → prediction residual.
    let residual = lpc_residual(samples, &lpc_coeffs);

    // Gain = peak absolute residual (ensures normalised residual fits in [−1, 1]).
    let gain = residual
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0_f32, f32::max)
        .max(1e-8); // prevent division by zero for silent frames

    // Quantise to i16.
    let residual_quantized: Vec<i16> = residual
        .iter()
        .map(|&r| {
            let scaled = r / gain * SILK_RESIDUAL_SCALE;
            scaled
                .round()
                .clamp(-SILK_RESIDUAL_SCALE, SILK_RESIDUAL_SCALE) as i16
        })
        .collect();

    Ok(SilkEncodedFrame {
        lpc_coeffs,
        residual_quantized,
        gain,
        n_samples,
    })
}

// ── silk_decode_frame ─────────────────────────────────────────────────────────

/// Decodes a SILK-encoded audio frame.
///
/// Steps:
/// 1. Dequantise: `e_hat[n] = q[n] / 32767 × gain`.
/// 2. Apply the LPC synthesis filter: `y[n] = e_hat[n] − Σ a[k]·y[n−1−k]`.
///
/// Both encoder and decoder use zero initial state, so the round-trip is exact
/// up to quantisation error (see [`SilkEncodedFrame`] for quality details).
///
/// # Arguments
/// - `frame` – A SILK frame produced by [`silk_encode_frame`].
///
/// # Returns
/// A `Vec<f32>` of `frame.n_samples` reconstructed PCM samples.
#[must_use]
pub fn silk_decode_frame(frame: &SilkEncodedFrame) -> Vec<f32> {
    // Dequantise residual.
    let residual: Vec<f32> = frame
        .residual_quantized
        .iter()
        .map(|&q| q as f32 / SILK_RESIDUAL_SCALE * frame.gain)
        .collect();

    // LPC synthesis filter.
    lpc_synthesis(&residual, &frame.lpc_coeffs)
}
