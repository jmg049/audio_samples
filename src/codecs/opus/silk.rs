//! SILK-style speech mode: one frame through LPC analysis and residual
//! quantisation, and back.
//!
//! The encoder computes an LPC predictor ([`lpc`](super::lpc)), takes the
//! prediction residual, normalises it by its peak and quantises it to 16 bits; the
//! decoder dequantises, rescales and runs the LPC synthesis filter.
//! [`silk_encode_frame_stateful`] also carries the LPC filter memory across frames
//! ([`SilkState`]) and applies a single-tap long-term predictor; this is the path
//! [`OpusCodec`](super::OpusCodec) uses. [`silk_encode_frame`] codes a frame on its
//! own, from zero filter state.
//!
//! The LPC predictor removes the spectral envelope of voiced speech, leaving a
//! residual with far less energy than the signal. Quantised at 16 bits per sample
//! (256 kbit/s at 16 kHz), it gives a high SNR but not a low bit rate. Real SILK
//! (RFC 6716) reaches 6 to 40 kbit/s with noise-shaped quantisation, LSF coding of
//! the coefficients and range coding, none of which this mode has. For music and
//! noise the LPC gains little; [`detect_mode`](super::mode::detect_mode) chooses
//! CELT for those.
//!
//! ## Limitations
//!
//! - LPC coefficients are stored as `f32` values, not as quantised line spectral
//!   frequencies.
//! - The LTP state does not carry across frames.

use crate::{AudioSampleError, AudioSampleResult, ParameterError};

use super::lpc::{
    LpcCoefficients, SILK_LPC_ORDER, estimate_pitch, lpc_analysis, lpc_residual,
    lpc_residual_stateful, lpc_synthesis, lpc_synthesis_stateful, ltp_residual, ltp_synthesis,
};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Scale factor for 16-bit residual quantisation.
///
/// The normalised residual (in `[-1, 1]`) is multiplied by this constant before
/// rounding to `i16`, and divided by the same constant during dequantisation.
const SILK_RESIDUAL_SCALE: f32 = 32_767.0;

/// Minimum allowed gain value, used to prevent division by zero for silent frames.
///
/// Corresponds to a residual peak amplitude of 10^-8 linear, which is well below
/// the noise floor of any practical audio system.
const MIN_GAIN_THRESHOLD: f32 = 1e-8;

// ── SilkState ─────────────────────────────────────────────────────────────────

/// Cross-frame LPC filter memory for the SILK codec.
///
/// Initialise with [`SilkState::default`] (all zeros) at the start of a
/// continuous signal. Pass the same `SilkState` to every successive
/// [`silk_encode_frame_stateful`] / [`silk_decode_frame_stateful`] call to
/// eliminate boundary artefacts between frames.
///
/// The encoder and decoder each maintain independent history buffers; the
/// encoder's history tracks the **input** samples, the decoder's history tracks
/// the **reconstructed output** samples.
#[derive(Debug, Clone, Default)]
pub struct SilkState {
    /// Last [`SILK_LPC_ORDER`] input samples from the preceding frame (encoder).
    pub encoder_lpc_history: Vec<f32>,
    /// Last [`SILK_LPC_ORDER`] reconstructed samples from the preceding frame (decoder).
    pub decoder_lpc_history: Vec<f32>,
}

// ── SilkEncodedFrame ──────────────────────────────────────────────────────────

/// One SILK-encoded audio frame.
///
/// Stores the LPC coefficients, a 16-bit quantised prediction residual, and the
/// gain used to normalise the residual before quantisation. This struct is
/// self-contained: the decoder needs no external side-channel information.
///
/// The decoded frame length equals `residual_quantized.len()`.
///
/// ## Round-trip quality
///
/// The only error in the encode/decode round-trip is residual quantisation.
/// With `gain = max(|e[n]|)` the maximum per-sample error is `gain / 32767`.
/// For a 440 Hz sine at amplitude 0.5:
///
/// - In steady state the LPC residual is tiny, but with zero initial filter state
///   the first `order` residual samples carry the start-up transient, so
///   `gain = max(|e[n]|)` is set by that transient (of order 10⁻²), not by the
///   steady-state residual.
/// - The maximum per-sample error is therefore about `gain / 32767` (≈ 10⁻⁶).
/// - Expected SNR > 50 dB.
///
/// For white noise the LPC provides no prediction gain (residual ≈ input),
/// but 16-bit quantisation still gives ≈ 90 dB dynamic range.
#[derive(Debug, Clone)]
pub struct SilkEncodedFrame {
    /// LPC predictor coefficients and prediction error from Levinson-Durbin.
    pub lpc_coeffs: LpcCoefficients,
    /// Residual `e[n]` normalised to `[-1, 1]` and quantised to 16-bit integers.
    pub residual_quantized: Vec<i16>,
    /// Peak absolute value of the prediction residual before normalisation.
    ///
    /// The decoder multiplies the dequantised residual by this value to restore
    /// the original amplitude scale.
    pub gain: f32,
    /// Pitch period in samples detected by the long-term predictor, or `None`
    /// if no clear periodicity was found or LTP was not used.
    pub pitch_lag: Option<usize>,
    /// Long-term prediction gain. Zero when `pitch_lag` is `None`.
    pub ltp_gain: f32,
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
/// - `samples` - PCM samples for the frame (f32, any amplitude range).
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

    // LPC analysis - order clamped for very short frames.
    let lpc_coeffs = lpc_analysis(samples, SILK_LPC_ORDER);

    // Analysis filter → prediction residual.
    let residual = lpc_residual(samples, &lpc_coeffs);

    // Gain = peak absolute residual (ensures normalised residual fits in [-1, 1]).
    let gain = residual
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0_f32, f32::max)
        .max(MIN_GAIN_THRESHOLD); // prevent division by zero for silent frames

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
        pitch_lag: None,
        ltp_gain: 0.0,
    })
}

// ── silk_decode_frame ─────────────────────────────────────────────────────────

/// Decodes a SILK-encoded audio frame.
///
/// Steps:
/// 1. Dequantise: `e_hat[n] = q[n] / 32767 × gain`.
/// 2. If `frame.pitch_lag` is `Some(lag)`, apply LTP synthesis:
///    `e_st[n] = e_lt[n] + ltp_gain × e_st[n - lag]`.
/// 3. Apply LPC synthesis: `y[n] = e_st[n] - Σ a[k]·y[n-1-k]`.
///
/// Both encoder and decoder use zero initial state, so the round-trip is exact
/// up to quantisation error (see [`SilkEncodedFrame`] for quality details).
///
/// For cross-frame continuity use [`silk_decode_frame_stateful`] instead.
///
/// # Arguments
/// - `frame` - A SILK frame produced by [`silk_encode_frame`] or
///   [`silk_encode_frame_stateful`].
///
/// # Returns
/// A `Vec<f32>` of reconstructed PCM samples.
#[must_use]
pub fn silk_decode_frame(frame: &SilkEncodedFrame) -> Vec<f32> {
    let residual: Vec<f32> = frame
        .residual_quantized
        .iter()
        .map(|&q| q as f32 / SILK_RESIDUAL_SCALE * frame.gain)
        .collect();

    let st_residual = match frame.pitch_lag {
        Some(lag) => ltp_synthesis(&residual, lag, frame.ltp_gain),
        None => residual,
    };

    lpc_synthesis(&st_residual, &frame.lpc_coeffs)
}

// ── silk_encode_frame_stateful ────────────────────────────────────────────────

/// Encodes a SILK frame with cross-frame LPC state and long-term prediction.
///
/// Extends [`silk_encode_frame`] in two ways:
///
/// 1. **Cross-frame LPC state** - the analysis filter uses `state.encoder_lpc_history`
///    to carry context from the previous frame, eliminating boundary artefacts for
///    consecutive frames.
///
/// 2. **Long-term prediction (LTP)** - after computing the short-term LPC
///    residual, [`estimate_pitch`] searches for a pitch period. When one is found,
///    a single-tap LTP filter (`d[n] = e[n] - ltp_gain × e[n - T]`) further
///    reduces the residual energy before quantisation.
///
/// # Arguments
/// - `samples` - PCM samples for the frame (f32).
/// - `sample_rate` - Signal sample rate in Hz (used for pitch lag bounds).
/// - `state` - Cross-frame state updated in place.
///
/// # Errors
/// Returns [`AudioSampleError::Parameter`] if `samples` is empty.
pub fn silk_encode_frame_stateful(
    samples: &[f32],
    sample_rate: u32,
    state: &mut SilkState,
) -> AudioSampleResult<SilkEncodedFrame> {
    if samples.is_empty() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "samples",
            "SILK frame must contain at least one sample",
        )));
    }

    let lpc_coeffs = lpc_analysis(samples, SILK_LPC_ORDER);

    // Stateful short-term analysis filter.
    let st_residual = lpc_residual_stateful(samples, &lpc_coeffs, &mut state.encoder_lpc_history);

    // Long-term prediction on the whitened residual.
    let (pitch_lag, ltp_gain, final_residual) = match estimate_pitch(&st_residual, sample_rate) {
        Some((lag, gain)) => {
            let lt_residual = ltp_residual(&st_residual, lag, gain);
            (Some(lag), gain, lt_residual)
        }
        None => (None, 0.0, st_residual),
    };

    let gain = final_residual
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0_f32, f32::max)
        .max(MIN_GAIN_THRESHOLD);

    let residual_quantized: Vec<i16> = final_residual
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
        pitch_lag,
        ltp_gain,
    })
}

// ── silk_decode_frame_stateful ────────────────────────────────────────────────

/// Decodes a SILK frame with cross-frame LPC state and LTP synthesis.
///
/// Mirror of [`silk_encode_frame_stateful`]. Uses `state.decoder_lpc_history`
/// to carry synthesis filter context across frame boundaries. Must be paired
/// with [`silk_encode_frame_stateful`] (same state sequence) for correct output.
///
/// # Arguments
/// - `frame` - A SILK frame produced by [`silk_encode_frame_stateful`].
/// - `state` - Cross-frame state updated in place.
///
/// # Returns
/// A `Vec<f32>` of reconstructed PCM samples.
#[must_use]
pub fn silk_decode_frame_stateful(frame: &SilkEncodedFrame, state: &mut SilkState) -> Vec<f32> {
    let residual: Vec<f32> = frame
        .residual_quantized
        .iter()
        .map(|&q| q as f32 / SILK_RESIDUAL_SCALE * frame.gain)
        .collect();

    let st_residual = match frame.pitch_lag {
        Some(lag) => ltp_synthesis(&residual, lag, frame.ltp_gain),
        None => residual,
    };

    lpc_synthesis_stateful(
        &st_residual,
        &frame.lpc_coeffs,
        &mut state.decoder_lpc_history,
    )
}
