//! CELT wideband audio codec: MDCT + psychoacoustic bit allocation for Opus's music mode.
//!
//! ## What
//!
//! Implements one encode/decode cycle for a single Opus CELT audio frame.
//! Each frame runs the full existing perceptual codec pipeline from
//! [`crate::codecs::perceptual`]:
//!
//! 1. MDCT analysis (window size = frame length).
//! 2. Psychoacoustic masking → per-band importance and allowed noise.
//! 3. Bit allocation: distribute the bit budget across bands proportional to
//!    perceptual importance.
//! 4. Scalar quantisation of MDCT coefficients per band.
//!
//! Decoding runs the inverse: dequantise → IMDCT with overlap-add.
//!
//! ## Why
//!
//! CELT is the wideband, low-latency half of Opus. It analyses and codes the
//! entire spectrum in one MDCT block matching the frame length, making it ideal
//! for music and generic audio. Unlike SILK, it makes no speech-specific
//! assumptions.
//!
//! ## Relationship to `PerceptualCodec`
//!
//! [`celt_encode_frame`] is essentially one call to the internal
//! `encode_segment` helper from `PerceptualCodec`, scoped to a single Opus
//! frame. The [`CeltEncodedFrame`] type mirrors
//! [`crate::codecs::perceptual::codec::EncodedSegment`] with the addition of
//! the per-frame sample count.

use std::num::{NonZeroU32, NonZeroUsize};

use non_empty_slice::NonEmptyVec;
use spectrograms::{MdctParams, WindowType};

use crate::codecs::perceptual::quantization::{BitAllocationResult, allocate_bits, dequantize, quantize};
use crate::codecs::perceptual::{BandLayout, PsychoacousticConfig, analyse_signal_with_window_size, reconstruct_signal};
use crate::{AudioSampleResult, AudioSamples, StandardSample};

// ── CeltEncodedFrame ──────────────────────────────────────────────────────────

/// One CELT-encoded Opus audio frame.
///
/// The in-memory representation is equivalent to
/// [`crate::codecs::perceptual::codec::EncodedSegment`] scoped to a single
/// Opus frame. The `window_size` equals the frame length, so `n_frames` is
/// typically 1 or 2 depending on how the MDCT hop interacts with the frame
/// boundary.
///
/// Everything needed to reconstruct the frame is self-contained: MDCT
/// parameters, per-band bit allocation, and the original sample count.
#[derive(Debug, Clone)]
pub struct CeltEncodedFrame {
    /// Quantised MDCT coefficients, row-major: index `k × n_frames + f`.
    pub quantized: NonEmptyVec<i32>,
    /// Number of MDCT bins per frame (`window_size / 2`).
    pub n_coefficients: NonZeroUsize,
    /// Number of MDCT analysis frames produced from this Opus frame.
    pub n_frames: NonZeroUsize,
    /// MDCT parameters used during analysis.
    pub mdct_params: MdctParams,
    /// Per-band bit allocation used for quantisation and dequantisation.
    pub allocation: BitAllocationResult,
    /// Number of PCM samples in the original Opus frame.
    pub n_samples: usize,
}

// ── celt_encode_frame ─────────────────────────────────────────────────────────

/// Encodes a single CELT audio frame.
///
/// The frame is analysed with the MDCT, processed through the psychoacoustic
/// model, and the resulting coefficients are quantised with the per-band
/// allocation from [`allocate_bits`].
///
/// # Arguments
/// - `frame` – Mono audio frame to encode.
/// - `band_layout` – Perceptual frequency-band partitioning (e.g. [`crate::BandLayout::celt`]).
/// - `psych_config` – Psychoacoustic masking configuration. Must have the same
///   number of weights as `band_layout.len()`.
/// - `window` – MDCT window function. [`spectrograms::WindowType::Hanning`] is a
///   reasonable default.
/// - `window_size` – Explicit MDCT window size (typically `= frame_length`, i.e.
///   the number of samples in `frame`). When `None`, an automatic size ≤ 2048 is
///   chosen.
/// - `bit_budget` – Total bits to allocate across all bands.
/// - `min_bits_per_band` – Minimum bits guaranteed to every band (typically 1).
///
/// # Errors
/// Returns [`crate::AudioSampleError::Parameter`] if `frame` is not mono, is
/// fewer than 4 samples, or `psych_config` is incompatible with `band_layout`.
pub fn celt_encode_frame<T: StandardSample>(
    frame: &AudioSamples<T>,
    band_layout: &BandLayout,
    psych_config: &PsychoacousticConfig,
    window: WindowType,
    window_size: Option<NonZeroUsize>,
    bit_budget: u32,
    min_bits_per_band: u8,
) -> AudioSampleResult<CeltEncodedFrame> {
    let n_samples = frame.samples_per_channel().get();

    let result =
        analyse_signal_with_window_size(frame, window, window_size, band_layout, psych_config)?;

    let allocation = allocate_bits(&result.band_metrics, bit_budget, min_bits_per_band);
    let quantized = quantize(
        result.coefficients.as_non_empty_slice(),
        result.n_coefficients,
        result.n_frames,
        &allocation,
    );

    Ok(CeltEncodedFrame {
        quantized,
        n_coefficients: result.n_coefficients,
        n_frames: result.n_frames,
        mdct_params: result.mdct_params,
        allocation,
        n_samples,
    })
}

// ── celt_decode_frame ─────────────────────────────────────────────────────────

/// Decodes a CELT-encoded Opus audio frame.
///
/// Dequantises the MDCT coefficients and applies the IMDCT with overlap-add to
/// reconstruct the time-domain signal.
///
/// # Arguments
/// - `frame` – A CELT frame produced by [`celt_encode_frame`].
/// - `sample_rate` – Sample rate for the returned audio.
///
/// # Errors
/// Returns [`crate::AudioSampleError`] if the IMDCT reconstruction fails.
///
/// # Returns
/// A `Vec<f32>` of `frame.n_samples` reconstructed PCM samples.
pub fn celt_decode_frame(
    frame: CeltEncodedFrame,
    sample_rate: NonZeroU32,
) -> AudioSampleResult<Vec<f32>> {
    let coefficients = dequantize(
        frame.quantized.as_non_empty_slice(),
        frame.n_coefficients,
        frame.n_frames,
        &frame.allocation,
    );

    let audio = reconstruct_signal(
        &coefficients,
        frame.n_coefficients,
        frame.n_frames,
        &frame.mdct_params,
        Some(frame.n_samples),
        sample_rate,
    )?;

    let ch = audio
        .channels()
        .next()
        .expect("reconstruct_signal always returns mono");
    Ok(ch
        .as_slice()
        .expect("mono channel is always contiguous")
        .to_vec())
}
