//! Opus-inspired teaching codec: SILK-style (speech), CELT-style (music), and Hybrid modes.
//!
//! ## Overview
//!
//! An Opus-inspired teaching codec modelled on the SILK/CELT/Hybrid structure of
//! RFC 6716, built on the `audio_samples` perceptual codec framework. **It is not
//! the Opus codec and is not bitstream compatible with Opus:** there is no range
//! coder, no LSF/NLSF quantisation, no PVQ or band-energy coding as in CELT, and
//! no noise shaping. Its "CELT" mode is a generic perceptual MDCT codec, and its
//! mode decision is a simple heuristic (see [`detect_mode`](crate::codecs::opus::mode::detect_mode)),
//! not libopus's. Use it to study the structure of a hybrid speech/music codec, not
//! to produce or decode `.opus` data. It is gated on the `opus-codec` feature flag.
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`lpc`](crate::codecs::opus::lpc) | Levinson-Durbin LPC analysis/synthesis, pitch estimation, LTP primitives. |
//! | [`mode`](crate::codecs::opus::mode) | `OpusMode`, `OpusBandwidth`, `OpusConfig`, and `detect_mode`. |
//! | [`silk`](crate::codecs::opus::silk) | SILK frame encode/decode: stateful LPC + LTP + 16-bit quantisation. |
//! | [`celt`](crate::codecs::opus::celt) | CELT frame encode/decode: delegates to the psychoacoustic pipeline. |
//! | [`hybrid`](crate::codecs::opus::hybrid) | Hybrid mode: SILK low band + CELT high band with IIR crossover. |
//! | [`stereo`](crate::codecs::opus::stereo) | M/S stereo codec [`OpusStereoCodec`](crate::codecs::opus::OpusStereoCodec). |
//! | [`codec`](crate::codecs::opus::codec) | [`OpusCodec`](crate::codecs::opus::OpusCodec) and [`OpusEncodedAudio`](crate::codecs::opus::OpusEncodedAudio) - the public entry points. |
//!
//! ## Design
//!
//! - **CELT** reuses the existing [`crate::codecs::perceptual`] pipeline:
//!   MDCT → psychoacoustic masking → bit allocation → scalar quantisation.
//!
//! - **SILK** uses new LPC primitives in [`lpc`](crate::codecs::opus::lpc) with cross-frame state and a
//!   single-tap long-term predictor (LTP).
//!
//! - **Hybrid** splits the signal at 8 kHz via a first-order IIR crossover
//!   (perfect reconstruction), encodes the low band with SILK and the high band
//!   with CELT, and sums on decode.
//!
//! - [`OpusCodec`](crate::codecs::opus::OpusCodec) segments audio into fixed-length frames and dispatches each
//!   to SILK, CELT, or Hybrid based on [`OpusConfig`](crate::codecs::opus::OpusConfig) or per-frame detection.
//!
//! ## What belongs in the IO crate
//!
//! Bitstream packing (range coding, Ogg/RIFF containers) and `.opus` file I/O
//! are responsibilities of `audio_samples_io`. This module provides only the
//! algorithmic core: analysis, quantisation, and reconstruction.
//!
//! ## Remaining limitations
//!
//! - **No cross-frame LTP state** - LTP synthesis restarts at zero each frame.
//! - **No range/arithmetic coding** - residuals are stored as raw `Vec<i16>`.

pub mod celt;
pub mod codec;
pub mod hybrid;
pub mod lpc;
pub mod mode;
pub mod silk;
pub mod stereo;

pub use celt::{CeltEncodedFrame, celt_decode_frame, celt_encode_frame};
pub use codec::{OpusCodec, OpusEncodedAudio, OpusEncodedFrame, OpusFrameData};
pub use hybrid::{
    HYBRID_CROSSOVER_HZ, HybridEncodedFrame, hybrid_decode_frame, hybrid_encode_frame,
};
pub use lpc::{
    LpcCoefficients, SILK_LPC_ORDER, compute_autocorrelation, estimate_pitch, levinson_durbin,
    lpc_analysis, lpc_residual, lpc_residual_stateful, lpc_synthesis, lpc_synthesis_stateful,
    ltp_residual, ltp_synthesis,
};
pub use mode::{OpusBandwidth, OpusConfig, OpusMode, detect_mode};
pub use silk::{
    SilkEncodedFrame, SilkState, silk_decode_frame, silk_decode_frame_stateful, silk_encode_frame,
    silk_encode_frame_stateful,
};
pub use stereo::{OpusStereoCodec, OpusStereoEncodedAudio};
