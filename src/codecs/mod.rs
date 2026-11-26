//! Perceptual audio coding for study: psychoacoustic analysis, bit allocation,
//! quantisation and two complete codecs built from them. The codecs show how
//! masking drives compression; they are not production codecs, and their output is
//! an in-memory structure, not a file format.
//!
//! [`AudioCodec`] is the encode and decode contract, and [`encode`] and [`decode`]
//! call it. [`PerceptualCodec`](crate::PerceptualCodec) and
//! [`StereoPerceptualCodec`] are MDCT codecs that allocate bits from the
//! signal-to-mask ratio of each band ([`perceptual`]).
#![cfg_attr(
    feature = "opus-codec",
    doc = r#" [`opus`] adds a codec with the SILK, CELT and hybrid structure of Opus."#
)]
#![cfg_attr(
    not(feature = "opus-codec"),
    doc = r#" `opus` adds a codec with the SILK, CELT and hybrid structure of Opus."#
)]
//!
//! ```rust
//! use audio_samples::codecs::{decode, encode};
//! use audio_samples::{BandLayout, PerceptualCodec, PsychoacousticConfig, sample_rate, sine_wave};
//! use spectrograms::WindowType;
//! use std::num::NonZeroUsize;
//! use std::time::Duration;
//!
//! let audio = sine_wave::<f32>(440.0, Duration::from_millis(200), sample_rate!(44100), 0.5);
//!
//! // 24 Bark bands over 1024 MDCT bins, with the MPEG-1 masking model.
//! let n_bands = NonZeroUsize::new(24).unwrap();
//! let layout = BandLayout::bark(n_bands, 44100.0, NonZeroUsize::new(1024).unwrap());
//! let weights = PsychoacousticConfig::uniform_weights(n_bands);
//! let config = PsychoacousticConfig::mpeg1(weights.as_non_empty_slice());
//!
//! let codec = PerceptualCodec::new(layout, config, WindowType::Hanning, 128_000, 1);
//! let encoded = encode(&audio, codec).unwrap();
//! let decoded = decode::<PerceptualCodec, f32>(encoded).unwrap();
//! assert_eq!(decoded.samples_per_channel(), audio.samples_per_channel());
//! ```

pub mod perceptual;

#[cfg(feature = "opus-codec")]
pub mod opus;

pub use perceptual::codec::{AudioCodec, decode, encode};
pub use perceptual::stereo::{StereoPerceptualCodec, StereoPerceptualEncodedAudio};
