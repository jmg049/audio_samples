//! Mid/Side stereo encoding for Opus: [`OpusStereoCodec`] and helpers.
//!
//! ## What
//!
//! Extends the mono [`OpusCodec`] to two-channel audio using the same
//! Mid/Side (M/S) matrix coding that [`crate::codecs::perceptual::stereo`]
//! applies to the perceptual codec:
//!
//! - **Mid** = `(L + R) / 2` — the common (mono-compatible) content.
//! - **Side** = `(L − R) / 2` — the stereo difference signal.
//!
//! Each channel is then compressed independently with its own [`OpusCodec`],
//! allowing separate bit budgets. The side channel is typically less energetic
//! and can receive fewer bits with minimal perceptual harm.
//!
//! ## Why
//!
//! Stereo audio is often highly correlated. M/S decorrelates the two channels,
//! concentrating the signal energy in the mid channel and leaving a low-energy
//! side channel. The side channel can then be coded at a lower bitrate.
//!
//! ## How
//!
//! ```rust,ignore
//! use audio_samples::codecs::{encode, decode};
//! use audio_samples::codecs::opus::{OpusStereoCodec, OpusConfig, OpusMode};
//! use spectrograms::WindowType;
//!
//! let mid_config  = OpusConfig::with_mode(OpusMode::Celt, 96_000);
//! let side_config = OpusConfig::with_mode(OpusMode::Celt, 32_000);
//! let codec = OpusStereoCodec::new(mid_config, side_config, WindowType::Hanning);
//!
//! let encoded   = encode(&stereo_audio, codec)?;
//! let recovered = decode::<OpusStereoCodec, f32>(encoded)?;
//! ```

use std::num::NonZeroU32;

use non_empty_slice::NonEmptyVec;
use spectrograms::WindowType;

use crate::codecs::perceptual::codec::AudioCodec;
use crate::codecs::perceptual::stereo::{mid_side_decode, mid_side_encode};
use crate::traits::AudioTypeConversion;
use crate::{AudioSampleError, AudioSampleResult, AudioSamples, ParameterError, StandardSample};

use super::codec::{OpusCodec, OpusEncodedAudio};
use super::mode::OpusConfig;

// ── OpusStereoEncodedAudio ────────────────────────────────────────────────────

/// In-memory encoded audio produced by [`OpusStereoCodec`].
///
/// Holds independently encoded mid and side channels plus the original signal
/// metadata needed for reconstruction.
#[derive(Debug, Clone)]
pub struct OpusStereoEncodedAudio {
    /// Encoded mid (sum) channel.
    pub mid: OpusEncodedAudio,
    /// Encoded side (difference) channel.
    pub side: OpusEncodedAudio,
    /// Total original signal length in samples per channel.
    pub original_length: usize,
    /// Sample rate of the original signal.
    pub sample_rate: NonZeroU32,
}

// ── OpusStereoCodec ───────────────────────────────────────────────────────────

/// A two-channel Opus codec using Mid/Side matrix coding.
///
/// ## What
///
/// Takes a stereo (`2 × N`) signal, applies the M/S transform to decorrelate
/// the channels, and codes mid and side independently with separate
/// [`OpusCodec`] instances.
///
/// ## Invariants
///
/// - Input audio must have exactly two channels.
/// - `mid_config` and `side_config` are applied to the respective mono signals
///   produced by the M/S transform.
#[derive(Debug, Clone)]
pub struct OpusStereoCodec {
    /// Codec configuration for the mid (sum) channel.
    pub mid_config: OpusConfig,
    /// Codec configuration for the side (difference) channel.
    pub side_config: OpusConfig,
    /// MDCT window function shared by both channel codecs.
    pub window: WindowType,
}

impl OpusStereoCodec {
    /// Creates an `OpusStereoCodec`.
    ///
    /// # Arguments
    /// - `mid_config` – Configuration for the mid channel (common content).
    ///   Typically higher bitrate.
    /// - `side_config` – Configuration for the side channel (difference signal).
    ///   Typically lower bitrate since the side is usually less energetic.
    /// - `window` – MDCT window function for CELT frames on both channels.
    #[inline]
    #[must_use]
    pub fn new(mid_config: OpusConfig, side_config: OpusConfig, window: WindowType) -> Self {
        Self {
            mid_config,
            side_config,
            window,
        }
    }

    /// Builds the mid-channel [`OpusCodec`].
    fn mid_codec(&self) -> OpusCodec {
        OpusCodec::new(self.mid_config.clone(), self.window.clone())
    }

    /// Builds the side-channel [`OpusCodec`].
    fn side_codec(&self) -> OpusCodec {
        OpusCodec::new(self.side_config.clone(), self.window.clone())
    }
}

// ── AudioCodec impl ───────────────────────────────────────────────────────────

impl AudioCodec for OpusStereoCodec {
    type Encoded = OpusStereoEncodedAudio;

    fn encode<T: StandardSample>(
        self,
        audio: &AudioSamples<T>,
    ) -> AudioSampleResult<Self::Encoded> {
        if audio.num_channels().get() != 2 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio",
                format!(
                    "OpusStereoCodec requires exactly 2 channels, got {}",
                    audio.num_channels()
                ),
            )));
        }

        let sample_rate = audio.sample_rate();
        let original_length = audio.samples_per_channel().get();

        // Extract L/R as f32.
        let audio_f32 = audio.to_format::<f32>();
        let mut channels = audio_f32.channels();
        let left = channels
            .next()
            .expect("validated 2 channels")
            .as_slice()
            .expect("contiguous")
            .to_vec();
        let right = channels
            .next()
            .expect("validated 2 channels")
            .as_slice()
            .expect("contiguous")
            .to_vec();

        // M/S matrix encode.
        let (mid_samples, side_samples) = mid_side_encode(&left, &right);

        let mid_ne = NonEmptyVec::new(mid_samples).map_err(|_| AudioSampleError::EmptyData)?;
        let side_ne = NonEmptyVec::new(side_samples).map_err(|_| AudioSampleError::EmptyData)?;

        let mid_audio: AudioSamples<'static, f32> =
            AudioSamples::from_mono_vec(mid_ne, sample_rate);
        let side_audio: AudioSamples<'static, f32> =
            AudioSamples::from_mono_vec(side_ne, sample_rate);

        // Encode each channel independently.
        let mid = self.mid_codec().encode(&mid_audio)?;
        let side = self.side_codec().encode(&side_audio)?;

        Ok(OpusStereoEncodedAudio {
            mid,
            side,
            original_length,
            sample_rate,
        })
    }

    fn decode<U: StandardSample>(
        encoded: Self::Encoded,
    ) -> AudioSampleResult<AudioSamples<'static, U>>
    where
        f32: crate::ConvertFrom<U>,
    {
        let sample_rate = encoded.sample_rate;
        let target_length = encoded.original_length;

        // Decode mid and side as mono f32.
        let mid_f32 = OpusCodec::decode::<f32>(encoded.mid)?;
        let side_f32 = OpusCodec::decode::<f32>(encoded.side)?;

        let mid_ch = mid_f32
            .channels()
            .next()
            .expect("OpusCodec::decode returns mono");
        let side_ch = side_f32
            .channels()
            .next()
            .expect("OpusCodec::decode returns mono");

        let mid_samples = mid_ch.as_slice().expect("mono contiguous");
        let side_samples = side_ch.as_slice().expect("mono contiguous");

        // M/S matrix decode → L/R.
        let (left, right) = mid_side_decode(mid_samples, side_samples);

        // Interleave L/R → stereo AudioSamples, truncating to original length.
        let interleaved: Vec<f32> = left
            .iter()
            .zip(right.iter())
            .take(target_length)
            .flat_map(|(&l, &r)| [l, r])
            .collect();
        let interleaved_ne =
            NonEmptyVec::new(interleaved).map_err(|_| AudioSampleError::EmptyData)?;

        let stereo_f32: AudioSamples<'static, f32> = AudioSamples::from_interleaved_vec(
            interleaved_ne,
            NonZeroU32::new(2).expect("2 > 0"),
            sample_rate,
        )?;

        Ok(stereo_f32.to_format::<U>())
    }
}
