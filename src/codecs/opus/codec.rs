//! [`OpusCodec`], [`OpusEncodedAudio`], and the [`AudioCodec`] implementation.
//!
//! ## What
//!
//! `OpusCodec` segments a mono audio signal into fixed-length frames and encodes
//! each frame independently using either SILK (speech) or CELT (music/generic),
//! producing an [`OpusEncodedAudio`] value that carries all information needed
//! for reconstruction.
//!
//! ## How
//!
//! ```rust,ignore
//! use audio_samples::codecs::{encode, decode};
//! use audio_samples::codecs::opus::{OpusCodec, OpusConfig, OpusMode};
//! use spectrograms::WindowType;
//!
//! // Encode with CELT (music mode), 20 ms frames, 128 kbps budget.
//! let config  = OpusConfig::with_mode(OpusMode::Celt, 128_000);
//! let codec   = OpusCodec::new(config, WindowType::Hanning);
//! let encoded = encode(&audio, codec)?;
//!
//! // Decode back to f32.
//! let recovered = decode::<OpusCodec, f32>(encoded)?;
//! ```
//!
//! ## Sketch limitations
//!
//! - Hybrid mode falls back to CELT.
//! - SILK frames use zero initial state (no cross-frame LPC memory).
//! - No range coding or bitstream packing — that belongs in `audio_samples_io`.
//! - CELT window size equals the frame length (not Opus's fixed 20 ms / 960 sample block).

use std::num::{NonZeroU32, NonZeroUsize};

use non_empty_slice::NonEmptyVec;
use spectrograms::WindowType;

use crate::codecs::perceptual::codec::AudioCodec;
use crate::codecs::perceptual::{BandLayout, PsychoacousticConfig};
use crate::traits::AudioTypeConversion;
use crate::{AudioSampleError, AudioSampleResult, AudioSamples, ParameterError, StandardSample};

use super::celt::{CeltEncodedFrame, celt_decode_frame, celt_encode_frame};
use super::mode::{OpusConfig, OpusMode, detect_mode};
use super::silk::{SilkEncodedFrame, silk_decode_frame, silk_encode_frame};

// ── OpusFrameData ─────────────────────────────────────────────────────────────

/// Codec-specific payload for one encoded Opus frame.
///
/// The active variant matches the [`OpusEncodedFrame::mode`] field.
#[derive(Debug, Clone)]
pub enum OpusFrameData {
    /// SILK-encoded payload: LPC coefficients and quantised residual.
    Silk(SilkEncodedFrame),
    /// CELT-encoded payload: quantised MDCT coefficients and bit allocation.
    Celt(CeltEncodedFrame),
}

// ── OpusEncodedFrame ──────────────────────────────────────────────────────────

/// One encoded Opus audio frame.
///
/// Stores the operating mode, the encoded payload, and the number of PCM
/// samples in the original frame (needed for accurate truncation on decode).
#[derive(Debug, Clone)]
pub struct OpusEncodedFrame {
    /// Mode used to encode this frame.
    pub mode: OpusMode,
    /// Codec-specific encoded payload.
    pub data: OpusFrameData,
    /// Number of PCM samples in the original frame.
    pub n_samples: usize,
}

// ── OpusEncodedAudio ──────────────────────────────────────────────────────────

/// In-memory encoded audio produced by [`OpusCodec`].
///
/// Contains one [`OpusEncodedFrame`] per audio frame in temporal order.
/// Everything needed to reconstruct the original signal is embedded: LPC
/// coefficients, MDCT parameters, per-band quantisation step sizes, and the
/// total original signal length.
#[derive(Debug, Clone)]
pub struct OpusEncodedAudio {
    /// Encoded frames in temporal order.
    pub frames: NonEmptyVec<OpusEncodedFrame>,
    /// Total original signal length in samples.
    pub original_length: usize,
    /// Sample rate of the original signal.
    pub sample_rate: NonZeroU32,
}

// ── OpusCodec ─────────────────────────────────────────────────────────────────

/// An Opus-inspired perceptual codec supporting SILK (speech) and CELT (music) modes.
///
/// ## What
///
/// `OpusCodec` segments audio into fixed-length frames (default 20 ms) and
/// encodes each frame using either:
///
/// - **SILK** — order-16 LPC + 16-bit residual quantisation (speech).
/// - **CELT** — MDCT + CELT-band psychoacoustic masking (music/generic).
///
/// The mode is either fixed via [`OpusConfig::mode`] or detected per-frame with
/// a spectral-flatness heuristic ([`detect_mode`]).
///
/// ## Architecture
///
/// ```text
/// AudioSamples<T>
///     └── frame 0  ──► detect_mode ──► SILK encode ──► OpusEncodedFrame(Silk)
///     └── frame 1  ──► detect_mode ──► CELT encode ──► OpusEncodedFrame(Celt)
///     └── …
///     └── OpusEncodedAudio { frames, original_length, sample_rate }
/// ```
///
/// ## Core design decisions
///
/// - The CELT band layout is auto-derived from the frame size and signal sample
///   rate unless [`OpusCodec::with_perceptual_config`] provides an explicit one.
/// - Hybrid mode is defined in the type system but currently falls back to CELT.
///
/// ## What lives in the IO crate
///
/// Bitstream packing, range coding, Ogg encapsulation, and the `.opus` container
/// format are responsibilities of `audio_samples_io`, not this crate.
///
/// ## Example
///
/// ```rust,ignore
/// use audio_samples::codecs::{encode, decode};
/// use audio_samples::codecs::opus::{OpusCodec, OpusConfig, OpusMode};
/// use spectrograms::WindowType;
///
/// let config = OpusConfig::with_mode(OpusMode::Celt, 128_000);
/// let codec  = OpusCodec::new(config, WindowType::Hanning);
///
/// let encoded   = encode(&audio, codec)?;
/// let recovered = decode::<OpusCodec, f32>(encoded)?;
/// ```
#[derive(Debug, Clone)]
pub struct OpusCodec {
    /// Codec configuration: mode, bandwidth, bit budget, frame size.
    pub config: OpusConfig,
    /// Explicit CELT band layout. When `None`, a CELT-style layout is
    /// auto-constructed from the signal's sample rate and frame size.
    pub band_layout: Option<BandLayout>,
    /// Psychoacoustic masking config for CELT frames. When `None`, MPEG-1
    /// parameters with uniform weights are used.
    pub psych_config: Option<PsychoacousticConfig>,
    /// MDCT window function for CELT frames.
    pub window: WindowType,
}

impl OpusCodec {
    /// Creates an `OpusCodec` with automatic CELT band layout and MPEG-1
    /// psychoacoustic parameters.
    ///
    /// The CELT band layout is derived from the signal's sample rate and frame
    /// size at encode time, so it adapts to any input sample rate.
    ///
    /// # Arguments
    /// - `config` – Codec configuration (mode, bandwidth, bit budget, frame size).
    /// - `window` – MDCT window function for CELT frames.
    #[inline]
    #[must_use]
    pub fn new(config: OpusConfig, window: WindowType) -> Self {
        Self {
            config,
            band_layout: None,
            psych_config: None,
            window,
        }
    }

    /// Creates an `OpusCodec` with explicit CELT perceptual configuration.
    ///
    /// Use this variant when you need full control over the CELT band layout and
    /// psychoacoustic parameters (e.g., for a fixed target sample rate).
    ///
    /// # Arguments
    /// - `config` – Codec configuration.
    /// - `window` – MDCT window function.
    /// - `band_layout` – Explicit perceptual band layout for CELT frames.
    /// - `psych_config` – Psychoacoustic masking parameters. Must be compatible
    ///   with `band_layout`.
    #[inline]
    #[must_use]
    pub fn with_perceptual_config(
        config: OpusConfig,
        window: WindowType,
        band_layout: BandLayout,
        psych_config: PsychoacousticConfig,
    ) -> Self {
        Self {
            config,
            band_layout: Some(band_layout),
            psych_config: Some(psych_config),
            window,
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Computes the frame size in samples for `sample_rate` and `frame_size_ms`.
///
/// The result is rounded to the nearest even integer (MDCT requirement) with a
/// minimum of 4.
fn compute_frame_size(sample_rate: u32, frame_size_ms: f32) -> usize {
    let raw = (sample_rate as f32 * frame_size_ms / 1000.0).round() as usize;
    let even = if raw % 2 == 0 { raw } else { raw + 1 };
    even.max(4)
}

/// Returns the CELT band layout to use, falling back to auto-construction.
fn resolve_band_layout(
    band_layout: &Option<BandLayout>,
    sample_rate: u32,
    n_bins: NonZeroUsize,
) -> BandLayout {
    band_layout
        .clone()
        .unwrap_or_else(|| BandLayout::celt(sample_rate as f32, n_bins))
}

/// Returns the psychoacoustic config to use, falling back to MPEG-1 with
/// uniform weights for `n_bands` bands.
fn resolve_psych_config(
    psych_config: &Option<PsychoacousticConfig>,
    n_bands: NonZeroUsize,
) -> PsychoacousticConfig {
    psych_config.clone().unwrap_or_else(|| {
        let weights = PsychoacousticConfig::uniform_weights(n_bands);
        PsychoacousticConfig::mpeg1(weights.as_non_empty_slice())
    })
}

// ── AudioCodec impl ───────────────────────────────────────────────────────────

impl AudioCodec for OpusCodec {
    type Encoded = OpusEncodedAudio;

    fn encode<T: StandardSample>(
        self,
        audio: &AudioSamples<T>,
    ) -> AudioSampleResult<Self::Encoded> {
        if audio.num_channels().get() != 1 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio",
                "OpusCodec requires mono input; mix down or extract a channel first",
            )));
        }

        let sample_rate = audio.sample_rate();
        let original_length = audio.samples_per_channel().get();

        // Frame size and CELT band layout — computed once for all frames.
        let frame_size = compute_frame_size(sample_rate.get(), self.config.frame_size_ms);
        // `compute_frame_size` always returns an even number >= 4 (see its doc),
        // so `frame_size / 2 >= 2` and `NonZeroUsize::new` always succeeds here.
        let n_bins = NonZeroUsize::new(frame_size / 2)
            .expect("frame_size is always even and >= 4, so frame_size/2 >= 2");
        let band_layout = resolve_band_layout(&self.band_layout, sample_rate.get(), n_bins);
        let n_bands = band_layout.len();
        let psych_config = resolve_psych_config(&self.psych_config, n_bands);

        // Convert to f32 once.
        let audio_f32 = audio.to_format::<f32>();
        let channel = audio_f32
            .channels()
            .next()
            .expect("mono channel validated above");
        let all_samples: &[f32] = channel.as_slice().expect("mono is contiguous");

        let mut frames: Vec<OpusEncodedFrame> = Vec::new();
        let mut offset = 0;

        while offset < original_length {
            let end = (offset + frame_size).min(original_length);
            let frame_samples = &all_samples[offset..end];
            let n_samples = frame_samples.len();

            // Determine mode for this frame.
            let raw_mode = self.config.mode.unwrap_or_else(|| {
                detect_mode(frame_samples, sample_rate.get(), self.config.bandwidth)
            });

            // Hybrid falls back to CELT in this sketch;
            // frames shorter than 4 samples are forced to SILK to avoid MDCT issues.
            let effective_mode = if n_samples < 4 {
                OpusMode::Silk
            } else {
                match raw_mode {
                    OpusMode::Hybrid => OpusMode::Celt,
                    other => other,
                }
            };

            let encoded_frame = match effective_mode {
                OpusMode::Silk => {
                    let silk_frame = silk_encode_frame(frame_samples)?;
                    OpusEncodedFrame {
                        mode: OpusMode::Silk,
                        data: OpusFrameData::Silk(silk_frame),
                        n_samples,
                    }
                }
                OpusMode::Celt | OpusMode::Hybrid => {
                    // Build a temporary mono AudioSamples for this frame.
                    let ne = NonEmptyVec::new(frame_samples.to_vec())
                        .map_err(|_| AudioSampleError::EmptyData)?;
                    let frame_audio: AudioSamples<'static, f32> =
                        AudioSamples::from_mono_vec(ne, sample_rate);

                    // Window size = frame length (even, >= 4).
                    let window_size = NonZeroUsize::new(n_samples & !1_usize)
                        .unwrap_or_else(|| NonZeroUsize::new(4).expect("4 > 0"));

                    let celt_frame = celt_encode_frame(
                        &frame_audio,
                        &band_layout,
                        &psych_config,
                        self.window.clone(),
                        Some(window_size),
                        self.config.bit_budget,
                        self.config.min_bits_per_band,
                    )?;
                    OpusEncodedFrame {
                        mode: OpusMode::Celt,
                        data: OpusFrameData::Celt(celt_frame),
                        n_samples,
                    }
                }
            };

            frames.push(encoded_frame);
            offset = end;
        }

        let frames_ne = NonEmptyVec::new(frames).map_err(|_| AudioSampleError::EmptyData)?;

        Ok(OpusEncodedAudio {
            frames: frames_ne,
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

        let mut all_samples: Vec<f32> = Vec::with_capacity(target_length);

        for frame in encoded.frames.into_vec() {
            let frame_samples = match frame.data {
                OpusFrameData::Silk(silk_frame) => silk_decode_frame(&silk_frame),
                OpusFrameData::Celt(celt_frame) => celt_decode_frame(celt_frame, sample_rate)?,
            };
            all_samples.extend(frame_samples);
        }

        all_samples.truncate(target_length);

        let samples_ne = NonEmptyVec::new(all_samples).map_err(|_| AudioSampleError::EmptyData)?;
        let f32_audio: AudioSamples<'static, f32> =
            AudioSamples::from_mono_vec(samples_ne, sample_rate);
        Ok(f32_audio.to_format::<U>())
    }
}
