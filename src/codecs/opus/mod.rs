//! Opus codec skeleton: SILK (speech) and CELT (music) modes.
//!
//! ## Overview
//!
//! This module provides a structural sketch of the Opus codec (RFC 6716) within
//! the `audio_samples` perceptual codec framework. It is gated on the
//! `opus-codec` feature flag.
//!
//! The implementation splits into four sub-modules that mirror the real Opus
//! architecture:
//!
//! | Module | Purpose |
//! |--------|---------|
//! [`lpc`] | Levinson–Durbin LPC analysis/synthesis primitives used by SILK. |
//! [`mode`] | `OpusMode`, `OpusBandwidth`, `OpusConfig`, and `detect_mode`. |
//! [`silk`] | SILK frame encode/decode: LPC residual + 16-bit quantisation. |
//! [`celt`] | CELT frame encode/decode: delegates to the psychoacoustic pipeline. |
//! [`stereo`] | M/S stereo codec [`OpusStereoCodec`]. |
//! [`codec`] | [`OpusCodec`] and [`OpusEncodedAudio`] — the public entry points. |
//!
//! ## Design
//!
//! - CELT mode **reuses** the existing [`crate::codecs::perceptual`] pipeline:
//!   MDCT analysis → psychoacoustic masking → bit allocation → scalar
//!   quantisation. The CELT-specific band layout ([`crate::BandLayout::celt`])
//!   uses the RFC 6716 band edges.
//!
//! - SILK mode provides **new** LPC primitives in [`lpc`] and combines them in
//!   [`silk`] for frame-level encode/decode.
//!
//! - [`OpusCodec`] segments audio into fixed-length frames and dispatches each
//!   to SILK or CELT based on the mode from [`OpusConfig`] or automatic
//!   detection via [`detect_mode`].
//!
//! ## What lives in the IO crate
//!
//! Bitstream packing (range coding, Ogg/RIFF containers) and `.opus` file I/O
//! are responsibilities of `audio_samples_io`. This module provides only the
//! algorithmic core: analysis, quantisation, and reconstruction.
//!
//! ## Known sketch limitations
//!
//! - **Hybrid mode** is defined but falls back to CELT.
//! - **Cross-frame LPC state** is not tracked; each SILK frame restarts from
//!   zero initial state (introduces mild boundary artefacts for long signals).
//! - **No adaptive codebook** (pitch analysis/long-term prediction) for SILK.
//! - **No range/arithmetic coding** — residuals are stored as raw `Vec<i16>`.

pub mod celt;
pub mod codec;
pub mod lpc;
pub mod mode;
pub mod silk;
pub mod stereo;

pub use celt::{CeltEncodedFrame, celt_decode_frame, celt_encode_frame};
pub use codec::{OpusCodec, OpusEncodedAudio, OpusEncodedFrame, OpusFrameData};
pub use lpc::{
    LpcCoefficients, SILK_LPC_ORDER, compute_autocorrelation, levinson_durbin, lpc_analysis,
    lpc_residual, lpc_synthesis,
};
pub use mode::{OpusBandwidth, OpusConfig, OpusMode, detect_mode};
pub use silk::{SilkEncodedFrame, silk_decode_frame, silk_encode_frame};
pub use stereo::{OpusStereoCodec, OpusStereoEncodedAudio};
