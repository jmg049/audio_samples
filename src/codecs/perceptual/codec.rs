//! [`AudioCodec`] trait, [`PerceptualCodec`], and encoded-audio types.

use std::num::{NonZeroU32, NonZeroUsize};

use non_empty_slice::{NonEmptySlice, NonEmptyVec};
use spectrograms::{MdctParams, WindowType};

use crate::traits::AudioTypeConversion;
use crate::{AudioSampleError, AudioSampleResult, AudioSamples, ParameterError, StandardSample};

use super::{
    BandLayout, PsychoacousticConfig, analyse_signal_with_window_size,
    bands::scale_band_layout,
    masking::detect_transient_windows,
    quantization::{BitAllocationResult, allocate_bits, dequantize, quantize},
    reconstruct_signal,
};

// ── AudioCodec trait ──────────────────────────────────────────────────────────

/// Abstraction over an audio codec's encode/decode round-trip.
///
/// A codec type serves as both the parameter container and the encode/decode
/// driver. Both `encode` and `decode` are generic over the sample type, so the
/// caller decides the input and output representations independently:
///
/// ```rust,ignore
/// use audio_samples::codecs::{encode, decode, PerceptualCodec};
///
/// // Encode from f32 audio.
/// let codec = PerceptualCodec::new(band_layout, config, WindowType::Hanning, 128_000, 1);
/// let encoded = encode(&audio_f32, codec)?;
///
/// // Decode to i16 (audio-aware scaling applied automatically).
/// let recovered = decode::<PerceptualCodec, i16>(encoded)?;
/// ```
pub trait AudioCodec: Sized {
    /// The in-memory encoded representation produced by [`encode`](AudioCodec::encode).
    type Encoded;

    /// Encodes `audio` into this codec's representation.
    ///
    /// The codec is consumed so that its parameters can be embedded in the
    /// returned [`Encoded`](AudioCodec::Encoded) value without cloning.
    ///
    /// # Errors
    /// Returns an error if `audio` is incompatible with the codec's parameters
    /// (e.g. multi-channel input for a mono codec, or a signal that is too short).
    fn encode<T: StandardSample>(self, audio: &AudioSamples<T>)
    -> AudioSampleResult<Self::Encoded>;

    /// Decodes an encoded representation into audio samples of type `U`.
    ///
    /// This is a static method — all information needed for decoding must be
    /// embedded in the [`Encoded`](AudioCodec::Encoded) value itself.
    ///
    /// The bound `f32: ConvertFrom<U>` is required because the audio-aware
    /// conversion layer must operate between `f32` (the codec's working format)
    /// and `U` in both directions. It is satisfied by all standard sample types.
    ///
    /// # Errors
    /// Returns an error if the encoded data is malformed or reconstruction fails.
    fn decode<U: StandardSample>(
        encoded: Self::Encoded,
    ) -> AudioSampleResult<AudioSamples<'static, U>>
    where
        f32: crate::ConvertFrom<U>;
}

// ── EncodedSegment ────────────────────────────────────────────────────────────

/// One encoded segment: a run of consecutive MDCT frames that all share the
/// same window size and band layout.
///
/// A [`PerceptualEncodedAudio`] normally contains a single segment (when window
/// switching is disabled). With window switching, transient regions produce
/// additional short-window segments interleaved with the long-window segments.
#[derive(Debug, Clone)]
pub struct EncodedSegment {
    /// Quantized MDCT coefficient indices for every frame in this segment,
    /// stored row-major: index `k * n_frames + f` for bin `k`, frame `f`.
    pub quantized: NonEmptyVec<i32>,
    /// Number of MDCT bins per frame in this segment (`window_size / 2`).
    pub n_coefficients: NonZeroUsize,
    /// Number of MDCT frames in this segment.
    pub n_frames: NonZeroUsize,
    /// MDCT parameters used for this segment (window shape, size, hop).
    pub mdct_params: MdctParams,
    /// Per-band bit allocation used to quantize and dequantize this segment.
    pub allocation: BitAllocationResult,
    /// Number of input samples covered by this segment; used to trim IMDCT
    /// output to the exact segment boundary during reconstruction.
    pub original_length: usize,
}

// ── PerceptualEncodedAudio ────────────────────────────────────────────────────

/// In-memory encoded representation produced by [`PerceptualCodec`].
///
/// Contains one or more [`EncodedSegment`] values in temporal order. A single
/// segment is produced when window switching is disabled; multiple segments
/// appear when transient frames are re-encoded with a shorter window.
///
/// Everything needed to reconstruct the original signal is embedded here:
/// MDCT parameters, quantization step sizes, and the original signal length.
#[derive(Debug, Clone)]
pub struct PerceptualEncodedAudio {
    /// Encoded segments in temporal order.
    pub segments: NonEmptyVec<EncodedSegment>,
    /// Total original signal length in samples.
    pub original_length: usize,
    /// Sample rate of the original signal.
    pub sample_rate: NonZeroU32,
}

// ── PerceptualCodec ───────────────────────────────────────────────────────────

/// A perceptual audio codec driven by MDCT + psychoacoustic masking.
///
/// ## What
///
/// `PerceptualCodec` implements the full perceptual codec pipeline:
/// analysis (MDCT → band energies → masking thresholds), bit allocation
/// (distribute `bit_budget` bits proportional to band importance), and
/// quantization (uniform scalar quantization per band). Decoding runs the
/// inverse: dequantization followed by IMDCT with overlap-add.
///
/// When `short_window_size` is set, the codec performs **window switching**:
/// transient frames are re-encoded with the shorter window for better time
/// resolution, preventing pre-echo artifacts. Non-transient frames continue
/// to use the long window for maximum frequency resolution.
///
/// ## Intended Usage
///
/// ```rust,ignore
/// use audio_samples::{BandLayout, PsychoacousticConfig};
/// use audio_samples::codecs::{encode, decode, PerceptualCodec};
/// use spectrograms::WindowType;
/// use std::num::NonZeroUsize;
///
/// let n_bands = NonZeroUsize::new(24).unwrap();
/// let n_bins  = NonZeroUsize::new(1024).unwrap();
/// let layout  = BandLayout::bark(n_bands, 44100.0, n_bins);
/// let weights = PsychoacousticConfig::uniform_weights(n_bands);
/// let config  = PsychoacousticConfig::mpeg1(weights.as_non_empty_slice());
///
/// // Basic codec (no window switching).
/// let codec = PerceptualCodec::new(layout.clone(), config.clone(), WindowType::Hanning, 128_000, 1);
/// let encoded = encode(&audio, codec)?;
/// let recovered = decode::<PerceptualCodec, f32>(encoded)?;
///
/// // With window switching: short window for transients.
/// let short_win = NonZeroUsize::new(256).unwrap();
/// let codec = PerceptualCodec::with_window_switching(
///     layout, config, WindowType::Hanning, 128_000, 1,
///     NonZeroUsize::new(2048).unwrap(), short_win, 8.0,
/// );
/// ```
///
/// ## Invariants
///
/// `config.band_count()` must equal `band_layout.len()`; validated at encode time.
#[derive(Debug, Clone)]
pub struct PerceptualCodec {
    /// Frequency band partitioning for analysis and bit allocation.
    pub band_layout: BandLayout,
    /// Psychoacoustic masking model configuration.
    pub config: PsychoacousticConfig,
    /// Window function applied to each MDCT frame.
    pub window: WindowType,
    /// Total bit budget to distribute across all bands per encode call.
    pub bit_budget: u32,
    /// Minimum bits guaranteed to every band regardless of importance.
    pub min_bits_per_band: u8,
    /// Explicit MDCT window size in samples (must be even, ≥ 4). When `None`,
    /// the codec auto-selects `min(2048, signal_length)` rounded to even.
    pub window_size: Option<NonZeroUsize>,
    /// Short window size for transient frames. When `Some`, window switching is
    /// enabled: frames whose energy exceeds the previous frame's by more than
    /// [`transient_threshold`](Self::transient_threshold) are re-encoded with
    /// this shorter window. Must be smaller than `window_size`.
    pub short_window_size: Option<NonZeroUsize>,
    /// Energy ratio threshold for transient detection. A frame is transient
    /// when `energy > prev_energy × threshold`. Typical range: 6.0–10.0.
    /// Only used when `short_window_size` is `Some`.
    pub transient_threshold: f32,
}

impl PerceptualCodec {
    /// Creates a `PerceptualCodec` without window switching.
    ///
    /// The MDCT window size is auto-selected as `min(2048, signal_length)`.
    /// Use [`PerceptualCodec::with_window_size`] or
    /// [`PerceptualCodec::with_window_switching`] when explicit control is needed.
    ///
    /// # Arguments
    /// - `band_layout` – Perceptual band partitioning (Bark, Mel, or ERB).
    /// - `config` – Psychoacoustic masking model parameters.
    /// - `window` – MDCT window function. [`WindowType::Hanning`] is a good default.
    /// - `bit_budget` – Total bits to allocate per encode call.
    /// - `min_bits_per_band` – Minimum bits guaranteed to every band (typically 1).
    #[inline]
    #[must_use]
    pub fn new(
        band_layout: BandLayout,
        config: PsychoacousticConfig,
        window: WindowType,
        bit_budget: u32,
        min_bits_per_band: u8,
    ) -> Self {
        Self {
            band_layout,
            config,
            window,
            bit_budget,
            min_bits_per_band,
            window_size: None,
            short_window_size: None,
            transient_threshold: 8.0,
        }
    }

    /// Creates a `PerceptualCodec` with an explicit MDCT window size but no
    /// window switching.
    ///
    /// The `window_size` must be even and ≥ 4; validated at encode time.
    #[inline]
    #[must_use]
    pub fn with_window_size(
        band_layout: BandLayout,
        config: PsychoacousticConfig,
        window: WindowType,
        bit_budget: u32,
        min_bits_per_band: u8,
        window_size: NonZeroUsize,
    ) -> Self {
        Self {
            band_layout,
            config,
            window,
            bit_budget,
            min_bits_per_band,
            window_size: Some(window_size),
            short_window_size: None,
            transient_threshold: 8.0,
        }
    }

    /// Creates a `PerceptualCodec` with window switching enabled.
    ///
    /// Transient frames (energy onset ratio above `transient_threshold`) are
    /// re-encoded with `short_window_size` for better time resolution. All
    /// other frames use `window_size`.
    ///
    /// # Arguments
    /// - `band_layout`, `config`, `window`, `bit_budget`, `min_bits_per_band` –
    ///   Same as [`PerceptualCodec::new`].
    /// - `window_size` – Long window size (even, ≥ 4). Used for steady-state frames.
    /// - `short_window_size` – Short window size for transient frames. Should
    ///   be a power-of-two fraction of `window_size` (e.g. `window_size / 8`).
    /// - `transient_threshold` – Energy ratio that triggers a transient. Typical: 8.0.
    #[inline]
    #[must_use]
    pub fn with_window_switching(
        band_layout: BandLayout,
        config: PsychoacousticConfig,
        window: WindowType,
        bit_budget: u32,
        min_bits_per_band: u8,
        window_size: NonZeroUsize,
        short_window_size: NonZeroUsize,
        transient_threshold: f32,
    ) -> Self {
        Self {
            band_layout,
            config,
            window,
            bit_budget,
            min_bits_per_band,
            window_size: Some(window_size),
            short_window_size: Some(short_window_size),
            transient_threshold,
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Groups a transient mask into runs of consecutive same-valued entries.
///
/// Returns a `NonEmptyVec` of `(frame_start, frame_end, is_transient)` triples
/// where `frame_start..frame_end` is the half-open range of frame indices in
/// the group.
fn group_by_type(mask: &NonEmptySlice<bool>) -> NonEmptyVec<(usize, usize, bool)> {
    let mut groups: Vec<(usize, usize, bool)> = Vec::new();
    let mut start = 0;
    let mut current = mask[0];

    for (i, &v) in mask.iter().enumerate().skip(1) {
        if v != current {
            groups.push((start, i, current));
            start = i;
            current = v;
        }
    }
    groups.push((start, mask.len().get(), current));

    // SAFETY: mask is non-empty, so at least one group is always produced.
    unsafe { NonEmptyVec::new_unchecked(groups) }
}

/// Encodes a sub-signal as a single [`EncodedSegment`].
fn encode_segment<T: StandardSample>(
    sub_audio: &AudioSamples<T>,
    window: WindowType,
    window_size: Option<NonZeroUsize>,
    band_layout: &BandLayout,
    config: &PsychoacousticConfig,
    bit_budget: u32,
    min_bits_per_band: u8,
) -> AudioSampleResult<EncodedSegment> {
    let original_length = sub_audio.samples_per_channel().get();

    let result =
        analyse_signal_with_window_size(sub_audio, window, window_size, band_layout, config)?;

    let allocation = allocate_bits(&result.band_metrics, bit_budget, min_bits_per_band);
    let quantized = quantize(
        result.coefficients.as_non_empty_slice(),
        result.n_coefficients,
        result.n_frames,
        &allocation,
    );

    Ok(EncodedSegment {
        quantized,
        n_coefficients: result.n_coefficients,
        n_frames: result.n_frames,
        mdct_params: result.mdct_params,
        allocation,
        original_length,
    })
}

// ── AudioCodec impl ───────────────────────────────────────────────────────────

impl AudioCodec for PerceptualCodec {
    type Encoded = PerceptualEncodedAudio;

    fn encode<T: StandardSample>(
        self,
        audio: &AudioSamples<T>,
    ) -> AudioSampleResult<Self::Encoded> {
        if !self.config.is_compatible_with(&self.band_layout) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "config",
                format!(
                    "PsychoacousticConfig has {} weights but BandLayout has {} bands",
                    self.config.band_count(),
                    self.band_layout.len(),
                ),
            )));
        }

        if audio.num_channels().get() != 1 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio",
                "PerceptualCodec requires mono input; mix down or extract a channel first",
            )));
        }

        let sample_rate = audio.sample_rate();
        let original_length = audio.samples_per_channel().get();

        match self.short_window_size {
            None => {
                // ── Single-segment path (no window switching) ──────────────
                let segment = encode_segment(
                    audio,
                    self.window,
                    self.window_size,
                    &self.band_layout,
                    &self.config,
                    self.bit_budget,
                    self.min_bits_per_band,
                )?;
                let segments = NonEmptyVec::new(vec![segment]).expect("single segment");
                Ok(PerceptualEncodedAudio {
                    segments,
                    original_length,
                    sample_rate,
                })
            }

            Some(short_ws) => {
                // ── Window-switching path ──────────────────────────────────
                let long_ws_val = self.window_size.map(|w| w.get()).unwrap_or_else(|| {
                    let raw = 2048_usize.min(original_length);
                    if raw % 2 == 0 { raw } else { raw - 1 }
                });

                if long_ws_val < 4 {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "signal",
                        format!("signal too short for analysis: {original_length} samples"),
                    )));
                }

                let long_ws = NonZeroUsize::new(long_ws_val).expect("validated >= 4");
                let long_hop = NonZeroUsize::new(long_ws_val / 2).expect("window >= 4");

                // Get f32 samples for transient detection.
                let audio_f32 = audio.to_format::<f32>();
                let mono = audio_f32.channels().next().expect("mono validated");
                let samples: &[f32] = mono.as_slice().expect("mono is contiguous");

                // SAFETY: original_length >= 1 (AudioSamples invariant).
                let samples_ne = unsafe { NonEmptySlice::new_unchecked(samples) };
                let transient_mask = detect_transient_windows(
                    samples_ne,
                    long_ws,
                    long_hop,
                    self.transient_threshold,
                );

                let groups = group_by_type(transient_mask.as_non_empty_slice());
                let n_groups = groups.len().get();

                // Short-window band layout (scaled from long-window layout).
                let long_n_bins = NonZeroUsize::new(long_ws_val / 2).expect("validated");
                let short_n_bins = NonZeroUsize::new(short_ws.get() / 2).max(NonZeroUsize::new(1));
                let short_n_bins = short_n_bins.expect("short_ws >= 2");
                let short_layout = scale_band_layout(&self.band_layout, long_n_bins, short_n_bins);

                let long_hop_val = long_hop.get();

                // Collect segment inputs sequentially (sample slicing preserves order).
                type SegInput = (AudioSamples<'static, f32>, Option<NonZeroUsize>, BandLayout);
                let mut seg_inputs: Vec<SegInput> = Vec::with_capacity(n_groups);

                for (g_idx, &(frame_start, frame_end, is_transient)) in groups.iter().enumerate() {
                    let sample_start = frame_start * long_hop_val;
                    let sample_end = if g_idx + 1 == n_groups {
                        original_length
                    } else {
                        (frame_end * long_hop_val).min(original_length)
                    };

                    if sample_start >= original_length {
                        break;
                    }

                    let seg_vec = samples[sample_start..sample_end].to_vec();
                    let seg_ne =
                        NonEmptyVec::new(seg_vec).map_err(|_| AudioSampleError::EmptyData)?;
                    let seg_audio: AudioSamples<'static, f32> =
                        AudioSamples::from_mono_vec(seg_ne, sample_rate);

                    let (win_size, layout) = if is_transient {
                        (Some(short_ws), short_layout.clone())
                    } else {
                        (self.window_size, self.band_layout.clone())
                    };
                    seg_inputs.push((seg_audio, win_size, layout));
                }

                // Encode segments — parallel when `parallel` feature is enabled.
                let window = self.window;
                let config = self.config;
                let bit_budget = self.bit_budget;
                let min_bits = self.min_bits_per_band;

                let process = |(seg_audio, win_size, layout): SegInput| {
                    encode_segment(
                        &seg_audio,
                        window.clone(),
                        win_size,
                        &layout,
                        &config,
                        bit_budget,
                        min_bits,
                    )
                };

                #[cfg(feature = "parallel")]
                let encoded_segments: AudioSampleResult<Vec<EncodedSegment>> = {
                    use rayon::prelude::*;
                    seg_inputs.into_par_iter().map(process).collect()
                };
                #[cfg(not(feature = "parallel"))]
                let encoded_segments: AudioSampleResult<Vec<EncodedSegment>> =
                    seg_inputs.into_iter().map(process).collect();

                let segments =
                    NonEmptyVec::new(encoded_segments?).map_err(|_| AudioSampleError::EmptyData)?;
                Ok(PerceptualEncodedAudio {
                    segments,
                    original_length,
                    sample_rate,
                })
            }
        }
    }

    fn decode<U: StandardSample>(
        encoded: Self::Encoded,
    ) -> AudioSampleResult<AudioSamples<'static, U>>
    where
        f32: crate::ConvertFrom<U>,
    {
        let sample_rate = encoded.sample_rate;
        let target_length = encoded.original_length;

        // Reconstruct each segment independently — order is preserved by rayon.
        let decode_seg = |seg: EncodedSegment| -> AudioSampleResult<Vec<f32>> {
            let coefficients = dequantize(
                seg.quantized.as_non_empty_slice(),
                seg.n_coefficients,
                seg.n_frames,
                &seg.allocation,
            );
            let seg_audio = reconstruct_signal(
                &coefficients,
                seg.n_coefficients,
                seg.n_frames,
                &seg.mdct_params,
                Some(seg.original_length),
                sample_rate,
            )?;
            let ch = seg_audio.channels().next().expect("mono");
            Ok(ch.as_slice().expect("mono is contiguous").to_vec())
        };

        #[cfg(feature = "parallel")]
        let segment_samples: AudioSampleResult<Vec<Vec<f32>>> = {
            use rayon::prelude::*;
            encoded
                .segments
                .into_vec()
                .into_par_iter()
                .map(decode_seg)
                .collect()
        };
        #[cfg(not(feature = "parallel"))]
        let segment_samples: AudioSampleResult<Vec<Vec<f32>>> = encoded
            .segments
            .into_vec()
            .into_iter()
            .map(decode_seg)
            .collect();

        let mut all_samples: Vec<f32> = Vec::with_capacity(target_length);
        for chunk in segment_samples? {
            all_samples.extend(chunk);
        }
        all_samples.truncate(target_length);

        let samples_ne = NonEmptyVec::new(all_samples).map_err(|_| AudioSampleError::EmptyData)?;
        let f32_audio: AudioSamples<'static, f32> =
            AudioSamples::from_mono_vec(samples_ne, sample_rate);
        Ok(f32_audio.to_format::<U>())
    }
}

// ── Free functions ────────────────────────────────────────────────────────────

/// Encodes an audio signal using the given codec.
///
/// The codec is consumed so that its parameters can be embedded in the
/// encoded output without cloning.
///
/// # Errors
/// Propagates errors from [`AudioCodec::encode`].
///
/// # Examples
///
/// ```rust,ignore
/// use audio_samples::codecs::{encode, PerceptualCodec};
///
/// let encoded = encode(&audio, PerceptualCodec::new(layout, config, window, 128_000, 1))?;
/// ```
#[inline]
pub fn encode<C: AudioCodec, T: StandardSample>(
    audio: &AudioSamples<T>,
    codec: C,
) -> AudioSampleResult<C::Encoded> {
    codec.encode(audio)
}

/// Decodes an encoded audio representation into audio samples of type `U`.
///
/// The output sample type is chosen at the call site.
///
/// # Errors
/// Propagates errors from [`AudioCodec::decode`].
///
/// # Examples
///
/// ```rust,ignore
/// use audio_samples::codecs::{decode, PerceptualCodec};
///
/// let recovered_f32 = decode::<PerceptualCodec, f32>(encoded_a)?;
/// let recovered_i16 = decode::<PerceptualCodec, i16>(encoded_b)?;
/// ```
#[inline]
pub fn decode<C: AudioCodec, U: StandardSample>(
    encoded: C::Encoded,
) -> AudioSampleResult<AudioSamples<'static, U>>
where
    f32: crate::ConvertFrom<U>,
{
    C::decode::<U>(encoded)
}
