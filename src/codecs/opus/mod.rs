//! A teaching codec with the structure of Opus: a SILK-style speech mode, a
//! CELT-style music mode and a hybrid of the two.
//!
//! It follows the SILK/CELT/hybrid layout of RFC 6716 and is built on the
//! [perceptual codec](crate::codecs::perceptual) framework. **It is not the Opus
//! codec and is not bitstream compatible with Opus:** there is no range coder, no
//! LSF/NLSF quantisation, no PVQ or band-energy coding as in CELT, and no noise
//! shaping. Its CELT mode is a generic perceptual MDCT codec, and its mode decision
//! is a simple heuristic ([`detect_mode`]).
//! Use it to study the structure of a hybrid speech and music codec, not to
//! produce or decode `.opus` data.
//!
//! [`OpusCodec`] cuts mono audio into fixed-length
//! frames and codes each with the mode set in
//! [`OpusConfig`], or the one detected per frame;
//! [`OpusStereoCodec`] codes two channels as
//! mid and side. Encoded audio is an in-memory structure.
//!
//! ```rust
//! use audio_samples::codecs::opus::{OpusCodec, OpusConfig, OpusMode};
//! use audio_samples::codecs::{decode, encode};
//! use audio_samples::{sample_rate, sine_wave};
//! use spectrograms::WindowType;
//! use std::time::Duration;
//!
//! let audio = sine_wave::<f32>(440.0, Duration::from_millis(100), sample_rate!(48000), 0.5);
//!
//! // Music mode, 128 kbit/s.
//! let codec = OpusCodec::new(OpusConfig::with_mode(OpusMode::Celt, 128_000), WindowType::Hanning);
//! let encoded = encode(&audio, codec).unwrap();
//! let decoded = decode::<OpusCodec, f32>(encoded).unwrap();
//! assert_eq!(decoded.samples_per_channel(), audio.samples_per_channel());
//! ```
//!
//! ## How it works
//!
//! - **CELT mode** runs the perceptual pipeline per frame: MDCT, psychoacoustic
//!   masking, bit allocation, scalar quantisation ([`celt`]).
//! - **SILK mode** predicts each sample from the previous ones with LPC
//!   ([`lpc`]), removes the pitch periodicity with a
//!   single-tap long-term predictor (LTP) and quantises the residual
//!   ([`silk`]).
//! - **Hybrid mode** splits the signal at 8 kHz with a first-order IIR crossover
//!   whose two bands sum back to the input exactly, codes the low band with SILK and
//!   the high band with CELT, and sums them on decode
//!   ([`hybrid`]).
//!
//! ## Limitations
//!
//! - The LPC filter state carries across frames, but the LTP state does not: LTP
//!   synthesis starts from zero in every frame.
//! - There is no entropy coding: quantised values are stored as plain integers.
//! - Frames have one length (`frame_size_ms`, 20 ms by default), and the CELT MDCT
//!   window equals the frame length, with no short blocks for transients.

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
