//! Mid/Side stereo encoding and the [`StereoPerceptualCodec`].
//!
//! [`StereoPerceptualCodec`] extends the mono [`PerceptualCodec`] to two channels
//! by mid/side coding. Mid = (L + R) / 2 carries the common content and
//! Side = (L − R) / 2 the difference. Each is coded by its own perceptual codec with
//! its own bit budget. Stereo recordings are usually highly correlated, so the side
//! signal is weak and can be quantised coarsely with little audible effect.
//! [`mid_side_encode`] and [`mid_side_decode`] are the matrix on its own.
//!
//! ```rust
//! use audio_samples::codecs::{StereoPerceptualCodec, decode, encode};
//! use audio_samples::{AudioSamples, BandLayout, PsychoacousticConfig, sample_rate};
//! use ndarray::Array2;
//! use spectrograms::WindowType;
//! use std::num::NonZeroUsize;
//!
//! // 0.1 s of a 440 Hz tone, slightly quieter on the right.
//! let stereo = AudioSamples::new_multi_channel(
//!     Array2::from_shape_fn((2, 4410), |(c, i)| {
//!         let x = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin();
//!         if c == 0 { 0.5 * x } else { 0.4 * x }
//!     }),
//!     sample_rate!(44100),
//! )
//! .unwrap();
//!
//! let n_bands = NonZeroUsize::new(24).unwrap();
//! let layout = BandLayout::bark(n_bands, 44100.0, NonZeroUsize::new(1024).unwrap());
//! let weights = PsychoacousticConfig::uniform_weights(n_bands);
//! let config = PsychoacousticConfig::mpeg1(weights.as_non_empty_slice());
//!
//! // 128 kbit for mid, 64 kbit for side.
//! let codec = StereoPerceptualCodec::new(layout, config, WindowType::Hanning, 128_000, 64_000, 1);
//! let encoded = encode(&stereo, codec).unwrap();
//! let decoded = decode::<StereoPerceptualCodec, f32>(encoded).unwrap();
//! assert_eq!(decoded.num_channels().get(), 2);
//! ```

use std::num::{NonZeroU32, NonZeroUsize};

use non_empty_slice::NonEmptyVec;
use spectrograms::WindowType;

use crate::traits::AudioTypeConversion;
use crate::{AudioSampleError, AudioSampleResult, AudioSamples, ParameterError, StandardSample};

use super::codec::{AudioCodec, PerceptualCodec, PerceptualEncodedAudio, decode as mono_decode};
use super::{BandLayout, PsychoacousticConfig};

/// Encodes left and right sample slices into mid and side components.
///
/// `mid[i]  = (left[i] + right[i]) / 2`
/// `side[i] = (left[i] - right[i]) / 2`
///
/// # Arguments
/// - `left` - Left channel samples.
/// - `right` - Right channel samples. Must have the same length as `left`.
///
/// # Returns
/// `(mid, side)` - two `Vec<f32>` of the same length as the inputs.
#[must_use]
pub fn mid_side_encode(left: &[f32], right: &[f32]) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(left.len(), right.len());
    let mid = left
        .iter()
        .zip(right)
        .map(|(&l, &r)| (l + r) * 0.5)
        .collect();
    let side = left
        .iter()
        .zip(right)
        .map(|(&l, &r)| (l - r) * 0.5)
        .collect();
    (mid, side)
}

/// Decodes mid and side components back into left and right samples.
///
/// `left[i]  = mid[i] + side[i]`
/// `right[i] = mid[i] - side[i]`
///
/// # Arguments
/// - `mid` - Mid channel samples.
/// - `side` - Side channel samples. Must have the same length as `mid`.
///
/// # Returns
/// `(left, right)` - two `Vec<f32>` of the same length as the inputs.
#[must_use]
pub fn mid_side_decode(mid: &[f32], side: &[f32]) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(mid.len(), side.len());
    let left = mid.iter().zip(side).map(|(&m, &s)| m + s).collect();
    let right = mid.iter().zip(side).map(|(&m, &s)| m - s).collect();
    (left, right)
}

/// In-memory encoded representation produced by [`StereoPerceptualCodec`].
///
/// Independently encoded mid and side channels, each as a full
/// [`PerceptualEncodedAudio`] (which may itself contain window-switched segments).
#[derive(Debug, Clone)]
pub struct StereoPerceptualEncodedAudio {
    /// Encoded mid (common) channel.
    pub mid: PerceptualEncodedAudio,
    /// Encoded side (difference) channel.
    pub side: PerceptualEncodedAudio,
}

/// A stereo perceptual codec using Mid/Side matrix coding.
///
/// ## What
///
/// `StereoPerceptualCodec` takes a two-channel input, applies M/S matrix coding
/// to decorrelate the channels, then encodes mid and side independently using
/// [`PerceptualCodec`] with separate bit budgets.
///
/// Window switching is supported: set `short_window_size` and `transient_threshold`
/// to enable it for both channels simultaneously.
///
/// ## Intended Usage
///
/// ```rust
/// # use audio_samples::{AudioSamples, sample_rate};
/// # let stereo_audio = AudioSamples::new_multi_channel(
/// #     ndarray::Array2::from_shape_fn((2, 4410), |(c, i)| (i as f32 * 0.06).sin() * (0.5 - 0.1 * c as f32)),
/// #     sample_rate!(44100),
/// # ).unwrap();
/// use audio_samples::{BandLayout, PsychoacousticConfig};
/// use audio_samples::codecs::{encode, decode, StereoPerceptualCodec};
/// use spectrograms::WindowType;
/// use std::num::NonZeroUsize;
///
/// let n_bands = NonZeroUsize::new(24).unwrap();
/// let layout  = BandLayout::bark(n_bands, 44100.0, NonZeroUsize::new(1024).unwrap());
/// let weights = PsychoacousticConfig::uniform_weights(n_bands);
/// let config  = PsychoacousticConfig::mpeg1(weights.as_non_empty_slice());
///
/// let codec = StereoPerceptualCodec::new(
///     layout, config, WindowType::Hanning,
///     128_000,  // mid bit budget (common content)
///     64_000,   // side bit budget (difference; usually less energetic)
///     1,
/// );
/// let encoded   = encode(&stereo_audio, codec)?;
/// let recovered = decode::<StereoPerceptualCodec, f32>(encoded)?;
/// assert_eq!(recovered.num_channels().get(), 2);
/// # Ok::<(), audio_samples::AudioSampleError>(())
/// ```
///
/// ## Invariants
///
/// Input audio must have exactly two channels. `config.band_count()` must equal
/// `band_layout.len()`; both are validated at encode time.
#[derive(Debug, Clone)]
pub struct StereoPerceptualCodec {
    /// Frequency band partitioning (same for both channels).
    pub band_layout: BandLayout,
    /// Psychoacoustic masking model configuration (same for both channels).
    pub config: PsychoacousticConfig,
    /// Window function applied to each MDCT frame.
    pub window: WindowType,
    /// Bit budget for the mid (common) channel.
    pub mid_bit_budget: u32,
    /// Bit budget for the side (difference) channel. Typically half of
    /// `mid_bit_budget`.
    pub side_bit_budget: u32,
    /// Minimum bits guaranteed to every band in both channels.
    pub min_bits_per_band: u8,
    /// Explicit MDCT window size. See [`PerceptualCodec::window_size`].
    pub window_size: Option<NonZeroUsize>,
    /// Short window size for transient frames. See [`PerceptualCodec::short_window_size`].
    pub short_window_size: Option<NonZeroUsize>,
    /// Transient detection threshold. See [`PerceptualCodec::transient_threshold`].
    pub transient_threshold: f32,
}

impl StereoPerceptualCodec {
    /// Creates a `StereoPerceptualCodec` without window switching.
    ///
    /// # Arguments
    /// - `band_layout` - Perceptual band partitioning.
    /// - `config` - Psychoacoustic masking model parameters.
    /// - `window` - MDCT window function.
    /// - `mid_bit_budget` - Bits for the mid (common) channel.
    /// - `side_bit_budget` - Bits for the side (difference) channel.
    /// - `min_bits_per_band` - Minimum bits per band (typically 1).
    #[inline]
    #[must_use]
    pub fn new(
        band_layout: BandLayout,
        config: PsychoacousticConfig,
        window: WindowType,
        mid_bit_budget: u32,
        side_bit_budget: u32,
        min_bits_per_band: u8,
    ) -> Self {
        Self {
            band_layout,
            config,
            window,
            mid_bit_budget,
            side_bit_budget,
            min_bits_per_band,
            window_size: None,
            short_window_size: None,
            transient_threshold: 8.0,
        }
    }

    /// Creates a `StereoPerceptualCodec` with window switching enabled.
    ///
    /// # Arguments
    /// - `band_layout`, `config`, `window`, `mid_bit_budget`, `side_bit_budget`,
    ///   `min_bits_per_band` - Same as [`StereoPerceptualCodec::new`].
    /// - `window_size` - Long MDCT window size.
    /// - `short_window_size` - Short MDCT window size for transient frames.
    /// - `transient_threshold` - Energy ratio threshold for transient detection.
    #[inline]
    #[must_use]
    pub fn with_window_switching(
        band_layout: BandLayout,
        config: PsychoacousticConfig,
        window: WindowType,
        mid_bit_budget: u32,
        side_bit_budget: u32,
        min_bits_per_band: u8,
        window_size: NonZeroUsize,
        short_window_size: NonZeroUsize,
        transient_threshold: f32,
    ) -> Self {
        Self {
            band_layout,
            config,
            window,
            mid_bit_budget,
            side_bit_budget,
            min_bits_per_band,
            window_size: Some(window_size),
            short_window_size: Some(short_window_size),
            transient_threshold,
        }
    }

    /// Builds the mid-channel [`PerceptualCodec`] from this stereo codec's parameters.
    fn mid_codec(&self) -> PerceptualCodec {
        PerceptualCodec {
            band_layout: self.band_layout.clone(),
            config: self.config.clone(),
            window: self.window.clone(),
            bit_budget: self.mid_bit_budget,
            min_bits_per_band: self.min_bits_per_band,
            window_size: self.window_size,
            short_window_size: self.short_window_size,
            transient_threshold: self.transient_threshold,
        }
    }

    /// Builds the side-channel [`PerceptualCodec`] from this stereo codec's parameters.
    fn side_codec(&self) -> PerceptualCodec {
        PerceptualCodec {
            band_layout: self.band_layout.clone(),
            config: self.config.clone(),
            window: self.window.clone(),
            bit_budget: self.side_bit_budget,
            min_bits_per_band: self.min_bits_per_band,
            window_size: self.window_size,
            short_window_size: self.short_window_size,
            transient_threshold: self.transient_threshold,
        }
    }
}

impl AudioCodec for StereoPerceptualCodec {
    type Encoded = StereoPerceptualEncodedAudio;

    fn encode<T: StandardSample>(
        self,
        audio: &AudioSamples<T>,
    ) -> AudioSampleResult<Self::Encoded> {
        if audio.num_channels().get() != 2 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio",
                format!(
                    "StereoPerceptualCodec requires exactly 2 channels, got {}",
                    audio.num_channels()
                ),
            )));
        }

        let sample_rate = audio.sample_rate();

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

        let mid_ne = NonEmptyVec::new(mid_samples)
            .map_err(|_| AudioSampleError::empty_data("mid_side_encode"))?;
        let side_ne = NonEmptyVec::new(side_samples)
            .map_err(|_| AudioSampleError::empty_data("mid_side_encode"))?;

        let mid_audio: AudioSamples<'static, f32> =
            AudioSamples::from_mono_vec(mid_ne, sample_rate);
        let side_audio: AudioSamples<'static, f32> =
            AudioSamples::from_mono_vec(side_ne, sample_rate);

        // Encode each channel with its own bit budget.
        let mid = self.mid_codec().encode(&mid_audio)?;
        let side = self.side_codec().encode(&side_audio)?;

        Ok(StereoPerceptualEncodedAudio { mid, side })
    }

    fn decode<U: StandardSample>(
        encoded: Self::Encoded,
    ) -> AudioSampleResult<AudioSamples<'static, U>>
    where
        f32: crate::ConvertFrom<U>,
    {
        // Decode each M/S channel to f32 via the mono codec.
        let mid_audio: AudioSamples<'static, f32> =
            mono_decode::<PerceptualCodec, f32>(encoded.mid)?;
        let side_audio: AudioSamples<'static, f32> =
            mono_decode::<PerceptualCodec, f32>(encoded.side)?;

        let mid_samples = mid_audio
            .channels()
            .next()
            .expect("mono")
            .as_slice()
            .expect("contiguous")
            .to_vec();
        let side_samples = side_audio
            .channels()
            .next()
            .expect("mono")
            .as_slice()
            .expect("contiguous")
            .to_vec();

        // M/S matrix decode.
        let (left, right) = mid_side_decode(&mid_samples, &side_samples);

        // Interleave L/R → stereo AudioSamples.
        let interleaved: Vec<f32> = left
            .iter()
            .zip(right.iter())
            .flat_map(|(&l, &r)| [l, r])
            .collect();
        let interleaved_ne = NonEmptyVec::new(interleaved)
            .map_err(|_| AudioSampleError::empty_data("mid_side_decode"))?;

        let stereo_f32: AudioSamples<'static, f32> = AudioSamples::from_interleaved_vec(
            interleaved_ne,
            NonZeroU32::new(2).expect("2 channels"),
            mid_audio.sample_rate(),
        )?;

        Ok(stereo_f32.to_format::<U>())
    }
}
