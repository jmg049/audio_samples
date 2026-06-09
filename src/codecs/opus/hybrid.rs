//! Opus hybrid mode: SILK for the low band (0–8 kHz), CELT for the high band.
//!
//! ## What
//!
//! Implements the `Hybrid` operating mode of the Opus codec (RFC 6716 §2).
//! The signal is split at a crossover frequency (8 kHz) using a first-order
//! IIR filter; the low band is encoded with SILK (LPC speech codec) and the
//! high band with CELT (perceptual MDCT codec).
//!
//! ## Why
//!
//! Hybrid mode is designed for super-wideband speech: the vocal-tract model
//! (SILK) is appropriate for the low band where speech formants live, while
//! CELT handles the high-frequency breath and sibilance that SILK cannot model
//! efficiently.
//!
//! ## Reconstruction
//!
//! The first-order IIR crossover has the **perfect reconstruction** property:
//! `LP[n] + HP[n] = x[n]` exactly. The decoder therefore reconstructs the
//! original signal by summing the SILK-decoded low band and the CELT-decoded
//! high band; the only error is the independent quantisation noise from each codec.

use std::num::{NonZeroU32, NonZeroUsize};

use non_empty_slice::NonEmptyVec;
use spectrograms::WindowType;

use crate::codecs::perceptual::{BandLayout, PsychoacousticConfig};
use crate::{AudioSampleResult, AudioSamples};

use super::celt::{CeltEncodedFrame, celt_decode_frame, celt_encode_frame};
use super::silk::{
    SilkEncodedFrame, SilkState, silk_decode_frame_stateful, silk_encode_frame_stateful,
};

/// Crossover frequency for Opus hybrid mode in Hz.
///
/// SILK encodes 0–8 kHz; CELT encodes 8 kHz to Nyquist (RFC 6716 §2.1.2).
pub const HYBRID_CROSSOVER_HZ: f32 = 8_000.0;

// ── HybridEncodedFrame ────────────────────────────────────────────────────────

/// One hybrid-encoded Opus audio frame.
///
/// Stores a SILK-encoded low band and a CELT-encoded high band. The decoder
/// reconstructs the signal by summing both decoded bands; because the encoder
/// uses a perfect-reconstruction crossover, `LP_decoded + HP_decoded ≈ original`
/// up to codec quantisation error.
#[derive(Debug, Clone)]
pub struct HybridEncodedFrame {
    /// SILK payload: the low-band signal (0–[`HYBRID_CROSSOVER_HZ`]).
    pub silk_frame: SilkEncodedFrame,
    /// CELT payload: the high-band signal ([`HYBRID_CROSSOVER_HZ`]–Nyquist).
    pub celt_frame: CeltEncodedFrame,
    /// Number of PCM samples in the original frame.
    pub n_samples: usize,
}

// ── Crossover filter (private) ────────────────────────────────────────────────

/// Returns the IIR coefficient α such that `y[n] = α·y[n−1] + (1−α)·x[n]`
/// has its −3 dB point at `crossover_hz`.
///
/// Derived by solving `|H_LP(e^{jωc})|² = ½` for α:
/// `α = (2 − cos ωc) − √((2 − cos ωc)² − 1)`.
fn crossover_alpha(crossover_hz: f32, sample_rate_hz: f32) -> f32 {
    let omega = std::f32::consts::TAU * crossover_hz / sample_rate_hz;
    let c = 2.0_f32 - omega.cos();
    (c - (c * c - 1.0_f32).sqrt()).clamp(0.0, 1.0)
}

/// Splits `samples` into `(low_band, high_band)` with perfect reconstruction.
///
/// `low_band[n] + high_band[n] == samples[n]` exactly.
fn crossover_split(samples: &[f32], alpha: f32) -> (Vec<f32>, Vec<f32>) {
    let one_minus = 1.0 - alpha;
    let mut lp = Vec::with_capacity(samples.len());
    let mut hp = Vec::with_capacity(samples.len());
    let mut prev = 0.0_f32;
    for &x in samples {
        let y = alpha * prev + one_minus * x;
        lp.push(y);
        hp.push(x - y);
        prev = y;
    }
    (lp, hp)
}

// ── hybrid_encode_frame ───────────────────────────────────────────────────────

/// Encodes a single hybrid audio frame.
///
/// Splits the frame at [`HYBRID_CROSSOVER_HZ`] using a first-order IIR
/// crossover, encodes the low band with SILK (with cross-frame state), and
/// encodes the high band with CELT using the supplied perceptual configuration.
///
/// # Arguments
/// - `frame_samples` – Mono PCM f32 samples for this frame.
/// - `sample_rate` – Signal sample rate in Hz.
/// - `band_layout` – Perceptual band layout for the CELT high-band encoder.
/// - `psych_config` – Psychoacoustic configuration for CELT.
/// - `window` – MDCT window for CELT.
/// - `bit_budget` – CELT bit budget for the high band.
/// - `min_bits` – Minimum bits per band for CELT allocation.
/// - `silk_state` – Cross-frame SILK LPC state.
///
/// # Errors
/// Returns an error if the CELT encoder fails (e.g. incompatible `psych_config`).
///
/// # Panics
/// Panics if `frame_samples` is empty or `sample_rate` is zero (caller must
/// guarantee both — `OpusCodec` does this before dispatch).
pub fn hybrid_encode_frame(
    frame_samples: &[f32],
    sample_rate: u32,
    band_layout: &BandLayout,
    psych_config: &PsychoacousticConfig,
    window: WindowType,
    bit_budget: u32,
    min_bits: u8,
    silk_state: &mut SilkState,
) -> AudioSampleResult<HybridEncodedFrame> {
    let n_samples = frame_samples.len();
    let alpha = crossover_alpha(HYBRID_CROSSOVER_HZ, sample_rate as f32);
    let (low_band, high_band) = crossover_split(frame_samples, alpha);

    // Encode low band with SILK (stateful).
    let silk_frame = silk_encode_frame_stateful(&low_band, sample_rate, silk_state)?;

    // Encode high band with CELT.
    let sr =
        NonZeroU32::new(sample_rate).expect("sample_rate > 0 — guaranteed by OpusCodec dispatch");
    let ne = NonEmptyVec::new(high_band)
        .expect("high_band non-empty — frame_samples non-empty guaranteed by caller");
    let frame_audio: AudioSamples<'static, f32> = AudioSamples::from_mono_vec(ne, sr);

    // Round frame length down to even for MDCT (same as in codec.rs).
    let window_size = NonZeroUsize::new((n_samples / 2) * 2)
        .expect("n_samples >= 4 guaranteed by OpusCodec dispatch");

    let celt_frame = celt_encode_frame(
        &frame_audio,
        band_layout,
        psych_config,
        window,
        Some(window_size),
        bit_budget,
        min_bits,
    )?;

    Ok(HybridEncodedFrame {
        silk_frame,
        celt_frame,
        n_samples,
    })
}

// ── hybrid_decode_frame ───────────────────────────────────────────────────────

/// Decodes a hybrid audio frame.
///
/// Decodes the SILK low band and CELT high band independently, then sums them.
/// Perfect reconstruction is guaranteed up to codec quantisation noise.
///
/// # Arguments
/// - `frame` – A hybrid frame produced by [`hybrid_encode_frame`].
/// - `sample_rate` – Sample rate for CELT reconstruction.
/// - `silk_state` – Cross-frame SILK LPC state (must match encoder sequence).
///
/// # Errors
/// Returns an error if the CELT decoder fails.
pub fn hybrid_decode_frame(
    frame: HybridEncodedFrame,
    sample_rate: NonZeroU32,
    silk_state: &mut SilkState,
) -> AudioSampleResult<Vec<f32>> {
    let n = frame.n_samples;

    let low = silk_decode_frame_stateful(&frame.silk_frame, silk_state);
    let high = celt_decode_frame(frame.celt_frame, sample_rate)?;

    // Sum; truncate to the original frame length.
    let len = n.min(low.len()).min(high.len());
    Ok((0..len).map(|i| low[i] + high[i]).collect())
}
