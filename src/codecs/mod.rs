//! Codec infrastructure: the [`AudioCodec`] trait, `encode`/`decode` free functions,
//! and built-in codec implementations (requires `feature = "psychoacoustic"`).
//!
//! ## What
//!
//! This module defines the codec abstraction layer for `audio_samples`. It provides
//! the [`AudioCodec`] trait — a common interface over encode/decode round-trips —
//! together with the [`PerceptualCodec`] and [`StereoPerceptualCodec`] implementations
//! that use psychoacoustic masking to drive perceptual quantization.
//!
//! ## Why
//!
//! `audio_samples` exposes the full signal-processing toolkit that perceptual codecs
//! need (MDCT, masking models, bit allocation, quantization), but leaves the codec
//! contract — how those building blocks are composed and how state is packaged for
//! transmission — to this layer. The [`AudioCodec`] trait is the glue.
//!
//! ## How
//!
//! ```rust,ignore
//! use audio_samples::codecs::{encode, decode, PerceptualCodec};
//! use audio_samples::{BandLayout, PsychoacousticConfig};
//! use spectrograms::WindowType;
//! use std::num::NonZeroUsize;
//!
//! let n_bands = NonZeroUsize::new(24).unwrap();
//! let layout  = BandLayout::bark(n_bands, 44100.0, NonZeroUsize::new(1024).unwrap());
//! let weights = PsychoacousticConfig::uniform_weights(n_bands);
//! let config  = PsychoacousticConfig::mpeg1(weights.as_non_empty_slice());
//!
//! let codec = PerceptualCodec::new(layout, config, WindowType::Hanning, 128_000, 1);
//! let encoded    = encode(&audio, codec)?;
//! let recovered  = decode::<PerceptualCodec, f32>(encoded)?;
//! ```

pub mod perceptual;

pub use perceptual::codec::{AudioCodec, decode, encode};
pub use perceptual::stereo::{StereoPerceptualCodec, StereoPerceptualEncodedAudio};
