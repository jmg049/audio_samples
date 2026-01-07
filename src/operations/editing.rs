//! Time-domain editing operations for [`AudioSamples`].
//!
//! ## What
//!
//! This module implements the [`AudioEditing`] trait, providing
//! time-domain editing operations: trimming, padding, fading, mixing,
//! concatenation, and more.
//!
//! ## Why
//!
//! Time-domain editing is the most common category of audio
//! manipulation — crop a recording, add silence, blend two sources.
//! Grouping all such operations behind a single trait keeps the
//! [`AudioSamples`] API organised and separates editing logic from the
//! core data type.
//!
//! ## How
//!
//! All operations are available on any [`AudioSamples<T>`] where `T`
//! is a supported sample type (`u8`, `i16`, `I24`, `i32`, `f32`,
//! `f64`).  Most operations work on both mono and multi-channel audio
//! transparently.
//!
//! ### Available operations
//!
//! | Method | Description |
//! |--------|-------------|
//! | [`reverse`](AudioEditing::reverse) | Time-reverse the signal |
//! | [`trim`](AudioEditing::trim) | Extract a time-bounded segment |
//! | [`pad`](AudioEditing::pad) | Add silence at start / end |
//! | [`pad_to_duration`](AudioEditing::pad_to_duration) | Pad to a target duration |
//! | [`split`](AudioEditing::split) | Divide into equal-length segments |
//! | [`concatenate`](AudioEditing::concatenate) | Join segments end-to-end |
//! | [`stack`](AudioEditing::stack) | Interleave mono sources into multi-channel |
//! | [`mix`](AudioEditing::mix) | Weighted sum of sources |
//! | [`fade_in`](AudioEditing::fade_in) / [`fade_out`](AudioEditing::fade_out) | Amplitude envelopes |
//! | [`repeat`](AudioEditing::repeat) | Tile the signal |
//! | [`trim_silence`](AudioEditing::trim_silence) | Remove leading/trailing silence |
//! | [`trim_all_silence`](AudioEditing::trim_all_silence) | Remove all silence regions |
//!
//! ## Example
//!
//! ```rust
//! use audio_samples::{AudioSamples, AudioEditing, sample_rate};
//! use ndarray::Array1;
//!
//! let data  = Array1::<f32>::zeros(100);
//! let audio = AudioSamples::new_mono(data, sample_rate!(100)).unwrap();
//!
//! // Trim to the middle 50 % of the signal (samples 25 … 75)
//! let trimmed = audio.trim(0.25, 0.75).unwrap();
//! assert_eq!(trimmed.samples_per_channel().get(), 50);
//! ```
//!
//! ## See Also
//!
//! - [`AudioSamples`]: The core audio data type.
//! - [`AudioEditing`]: The trait defining all editing operations.
//!
//! [`AudioEditing`]: crate::operations::traits::AudioEditing

use std::num::NonZeroUsize;
use std::time::Duration;

#[cfg(feature = "random-generation")]
use crate::brown_noise;
use crate::operations::traits::AudioEditing;
use crate::operations::types::{FadeCurve, NoiseColor, PadSide};

#[cfg(all(feature = "random-generation", feature = "iir-filtering"))]
use crate::operations::types::{PerturbationConfig, PerturbationMethod};
use crate::repr::{AudioData, ChannelCount};
use crate::utils::{samples_to_seconds, seconds_to_samples};
use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, AudioStatistics, AudioTypeConversion,
    ConvertTo, LayoutError, ParameterError, StandardSample,
};
#[cfg(feature = "random-generation")]
use crate::{pink_noise, white_noise};

use ndarray::{Array1, Array2, Axis, s};
use non_empty_slice::{NonEmptySlice, NonEmptyVec};

#[cfg(feature = "random-generation")]
use rand::Rng;

/// Validates time bounds for trim operations
fn validate_time_bounds(start: f64, end: f64, duration: f64) -> AudioSampleResult<()> {
    if start < 0.0 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "parameter",
            format!("Start time cannot be negative: {start}"),
        )));
    }
    if end <= start {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "parameter",
            format!("End time ({end}) must be greater than start time ({start})"),
        )));
    }
    if end > duration {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "parameter",
            format!("End time ({end}) exceeds audio duration ({duration})"),
        )));
    }
    Ok(())
}

/// Applies fade curve transformation to a position value [0.0, 1.0]
fn apply_fade_curve(curve: &FadeCurve, position: f64) -> f64 {
    match curve {
        FadeCurve::Linear => position,
        FadeCurve::Exponential => position * position,
        FadeCurve::Logarithmic => {
            if position <= 0.0 {
                0.0
            } else {
                position.ln_1p() / 2.0f64.ln()
            }
        }
        FadeCurve::SmoothStep => position * position * 2.0f64.mul_add(-position, 3.0),
    }
}

impl<T> AudioEditing for AudioSamples<'_, T>
where
    T: StandardSample,
    Self: AudioTypeConversion<Sample = T>,
{
    /// Returns a time-reversed copy of the signal.
    ///
    /// All channels are reversed independently.  The sample rate and
    /// channel count are preserved.
    ///
    /// # Returns
    /// A new [`AudioSamples`] with the sample order reversed.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     array![1.0f32, 2.0, 3.0], sample_rate!(44100),
    /// ).unwrap();
    /// let rev = audio.reverse();
    /// assert_eq!(rev[0], 3.0);
    /// assert_eq!(rev[1], 2.0);
    /// assert_eq!(rev[2], 1.0);
    /// ```
    fn reverse<'b>(&self) -> AudioSamples<'b, T> {
        match &self.data {
            AudioData::Mono(arr) => {
                // Reverse the 1D array using ndarray's reverse slicing
                let reversed = arr.slice(s![..;-1]).to_owned();

                match AudioSamples::new_mono(reversed, self.sample_rate()) {
                    Ok(audio) => audio,
                    Err(_) => unreachable!("self was valid, therefore reversed is valid"),
                }
            }
            AudioData::Multi(arr) => {
                // Reverse along the time axis (axis 1)
                let reversed = arr.slice(s![.., ..;-1]).to_owned();
                match AudioSamples::new_multi_channel(reversed, self.sample_rate()) {
                    Ok(audio) => audio,
                    Err(_) => unreachable!("self was valid, therefore reversed is valid"),
                }
            }
        }
    }

    /// Reverses the sample order in place.
    ///
    /// All channels are reversed independently.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Layout`] if an internal multi-channel
    ///   array row is not contiguous.
    fn reverse_in_place(&mut self) -> AudioSampleResult<()> {
        match &mut self.data {
            AudioData::Mono(arr) => {
                arr.as_slice_mut().reverse();
            }
            AudioData::Multi(arr) => {
                // Reverse along the time axis (axis 1)
                for mut channel in arr.axis_iter_mut(Axis(0)) {
                    channel
                        .as_slice_mut()
                        .ok_or(AudioSampleError::Layout(LayoutError::NonContiguous {
                            operation: "array access".to_string(),
                            layout_type: "Multi-channel samples must be contiguous".to_string(),
                        }))?
                        .reverse();
                }
            }
        }
        Ok(())
    }

    /// Extracts a segment of audio between two time boundaries.
    ///
    /// Works on both mono and multi-channel audio.  Time values are
    /// converted to sample indices using the signal's sample rate.
    ///
    /// # Arguments
    /// - `start_seconds` — start of the segment in seconds (`>= 0`).
    /// - `end_seconds` — end of the segment in seconds
    ///   (`> start_seconds`, `<= duration`).
    ///
    /// # Returns
    /// A new [`AudioSamples`] containing the requested segment.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `start_seconds < 0`,
    ///   `end_seconds <= start_seconds`, or `end_seconds` exceeds the
    ///   signal duration.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     Array1::<f32>::zeros(100), sample_rate!(100),
    /// ).unwrap();
    /// let trimmed = audio.trim(0.25, 0.75).unwrap();
    /// assert_eq!(trimmed.samples_per_channel().get(), 50);
    /// ```
    fn trim<'b>(
        &self,
        start_seconds: f64,
        end_seconds: f64,
    ) -> AudioSampleResult<AudioSamples<'b, T>> {
        let duration: f64 = self.duration_seconds();
        validate_time_bounds(start_seconds, end_seconds, duration)?;

        let start_sample = seconds_to_samples(start_seconds, self.sample_rate());
        let end_sample = seconds_to_samples(end_seconds, self.sample_rate());

        match &self.data {
            AudioData::Mono(arr) => {
                if end_sample > arr.len().get() {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "parameter",
                        format!(
                            "End sample {} exceeds audio length {}",
                            end_sample,
                            arr.len().get()
                        ),
                    )));
                }
                let trimmed = arr.slice(s![start_sample..end_sample]).to_owned();
                AudioSamples::new_mono(trimmed, self.sample_rate())
            }
            AudioData::Multi(arr) => {
                if end_sample > arr.ncols().get() {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "parameter",
                        format!(
                            "End sample {} exceeds audio length {}",
                            end_sample,
                            arr.ncols().get()
                        ),
                    )));
                }
                let trimmed = arr.slice(s![.., start_sample..end_sample]).to_owned();
                AudioSamples::new_multi_channel(trimmed, self.sample_rate())
            }
        }
    }

    /// Adds silence (or a constant value) at the beginning and/or end.
    ///
    /// Works on both mono and multi-channel audio.
    ///
    /// # Arguments
    /// - `pad_start_seconds` — duration of padding prepended, in seconds.
    ///   Must be `>= 0`.
    /// - `pad_end_seconds` — duration of padding appended, in seconds.
    ///   Must be `>= 0`.
    /// - `pad_value` — the sample value used for the padded region
    ///   (typically the zero value of `T` for silence).
    ///
    /// # Returns
    /// A new [`AudioSamples`] with the requested padding applied.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if either padding
    ///   duration is negative.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     Array1::from_elem(5, 1.0f32), sample_rate!(10),
    /// ).unwrap();
    /// // 0.2 s at 10 Hz = 2 samples each side
    /// let padded = audio.pad(0.2, 0.2, 0.0).unwrap();
    /// assert_eq!(padded.samples_per_channel().get(), 9);
    /// assert_eq!(padded[0], 0.0);
    /// assert_eq!(padded[2], 1.0);
    /// ```
    fn pad<'b>(
        &self,
        pad_start_seconds: f64,
        pad_end_seconds: f64,
        pad_value: T,
    ) -> AudioSampleResult<AudioSamples<'b, T>> {
        if pad_start_seconds < 0.0 || pad_end_seconds < 0.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "padding_durations",
                "Padding durations cannot be negative",
            )));
        }

        let start_samples = seconds_to_samples(pad_start_seconds, self.sample_rate());
        let end_samples = seconds_to_samples(pad_end_seconds, self.sample_rate());

        match &self.data {
            AudioData::Mono(arr) => {
                let total_length = start_samples + arr.len().get() + end_samples;
                let mut padded = Array1::from_elem(total_length, pad_value);

                // Copy original data to the middle
                padded
                    .slice_mut(s![start_samples..start_samples + arr.len().get()])
                    .assign(&arr.view());

                AudioSamples::new_mono(padded, self.sample_rate())
            }
            AudioData::Multi(arr) => {
                let total_length = start_samples + arr.ncols().get() + end_samples;
                let mut padded = Array2::from_elem((arr.nrows().get(), total_length), pad_value);

                // Copy original data to the middle
                padded
                    .slice_mut(s![.., start_samples..start_samples + arr.ncols().get()])
                    .assign(&arr.view());

                AudioSamples::new_multi_channel(padded, self.sample_rate())
            }
        }
    }

    /// Pads with a constant value on the right to reach a target sample count.
    ///
    /// If the signal already has `target_num_samples` or more samples per
    /// channel it is returned unchanged (as an owned clone).
    ///
    /// # Arguments
    /// - `target_num_samples` — desired number of samples per channel.
    /// - `pad_value` — the sample value used for the padded region.
    ///
    /// # Returns
    /// A new [`AudioSamples`] with at least `target_num_samples` samples
    /// per channel.
    ///
    /// # Errors
    /// Propagates any error from the underlying [`pad_to_duration`] call.
    ///
    /// [`pad_to_duration`]: Self::pad_to_duration
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     array![1.0f32, 2.0, 3.0], sample_rate!(44100),
    /// ).unwrap();
    /// let padded = audio.pad_samples_right(6, 0.0).unwrap();
    /// assert_eq!(padded.samples_per_channel().get(), 6);
    /// assert_eq!(padded[3], 0.0); // padded region is silent
    /// ```
    fn pad_samples_right<'b>(
        &self,
        target_num_samples: usize,
        pad_value: T,
    ) -> AudioSampleResult<AudioSamples<'b, T>> {
        let current_num_samples = self.samples_per_channel().get();
        if target_num_samples <= current_num_samples {
            return Ok(self.clone().into_owned());
        }

        let target_num_samples_seconds: f64 =
            samples_to_seconds(target_num_samples, self.sample_rate.get());

        self.pad_to_duration(target_num_samples_seconds, pad_value, PadSide::Right)
    }

    /// Pads the signal to reach a target duration.
    ///
    /// Padding is added on the side specified by `pad_side`.  If the
    /// signal is already at least as long as `target_duration_seconds`
    /// it is returned unchanged (as an owned clone).
    ///
    /// # Arguments
    /// - `target_duration_seconds` — desired length in seconds.
    /// - `pad_value` — the sample value used for the padded region.
    /// - `pad_side` — which end to pad ([`PadSide::Left`] or
    ///   [`PadSide::Right`]).
    ///
    /// # Returns
    /// A new [`AudioSamples`] at least `target_duration_seconds` long.
    ///
    /// # Errors
    /// Propagates any error from the underlying [`pad`] call.
    ///
    /// [`pad`]: Self::pad
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use audio_samples::operations::types::PadSide;
    /// use ndarray::Array1;
    ///
    /// // 100 samples at 100 Hz = 1.0 s
    /// let audio = AudioSamples::new_mono(
    ///     Array1::<f32>::ones(100), sample_rate!(100),
    /// ).unwrap();
    /// // Pad to 1.5 s on the right → 50 extra samples of silence
    /// let padded = audio.pad_to_duration(1.5, 0.0, PadSide::Right).unwrap();
    /// assert_eq!(padded.samples_per_channel().get(), 150);
    /// ```
    fn pad_to_duration<'b>(
        &self,
        target_duration_seconds: f64,
        pad_value: T,
        pad_side: PadSide,
    ) -> AudioSampleResult<AudioSamples<'b, T>> {
        let current_duration = self.duration_seconds();

        if target_duration_seconds <= current_duration {
            return Ok(self.clone().into_owned());
        }

        let total_target_samples =
            seconds_to_samples(target_duration_seconds, self.sample_rate.get());
        let current_samples = self.samples_per_channel().get();
        let total_padding_samples = total_target_samples - current_samples;

        let (pad_start_samples, pad_end_samples) = match pad_side {
            PadSide::Left => (total_padding_samples, 0),
            PadSide::Right => (0, total_padding_samples),
        };

        let padded = self.pad(
            pad_start_samples as f64 / f64::from(self.sample_rate.get()),
            pad_end_samples as f64 / f64::from(self.sample_rate.get()),
            pad_value,
        )?;

        Ok(padded)
    }

    /// Splits the signal into segments of a fixed duration.
    ///
    /// The last segment may be shorter than `segment_duration_seconds`
    /// if the signal does not divide evenly.  Works on both mono and
    /// multi-channel audio.
    ///
    /// # Arguments
    /// - `segment_duration_seconds` — target length of each segment in
    ///   seconds.  Must be `> 0` and must not exceed the signal length.
    ///
    /// # Returns
    /// A vector of segments in chronological order.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the duration is not
    ///   positive or exceeds the signal length.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::Array1;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     Array1::<f32>::zeros(100), sample_rate!(100),
    /// ).unwrap();
    /// let segments = audio.split(0.25).unwrap();
    /// assert_eq!(segments.len(), 4);
    /// assert_eq!(segments[0].samples_per_channel().get(), 25);
    /// ```
    fn split(
        &self,
        segment_duration_seconds: f64,
    ) -> AudioSampleResult<Vec<AudioSamples<'static, T>>> {
        if segment_duration_seconds <= 0.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Segment duration must be positive",
            )));
        }

        let segment_samples = seconds_to_samples(segment_duration_seconds, self.sample_rate.get());
        let total_samples = self.samples_per_channel().get();

        if segment_samples > total_samples {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Segment duration exceeds audio length",
            )));
        }

        let mut segments = Vec::new();
        let mut start = 0;

        while start < total_samples {
            let end = (start + segment_samples).min(total_samples);

            match &self.data {
                AudioData::Mono(arr) => {
                    let segment = arr.slice(s![start..end]).to_owned();
                    segments.push(AudioSamples::new_mono(segment, self.sample_rate())?);
                }
                AudioData::Multi(arr) => {
                    let segment = arr.slice(s![.., start..end]).to_owned();
                    segments.push(AudioSamples::new_multi_channel(
                        segment,
                        self.sample_rate(),
                    )?);
                }
            }

            start += segment_samples;
        }

        Ok(segments)
    }

    /// Joins multiple audio segments end-to-end.
    ///
    /// All segments must share the same sample rate and channel count.
    /// The output preserves the order of the input slice.
    ///
    /// # Arguments
    /// - `segments` — the segments to join, in order.
    ///
    /// # Returns
    /// A single [`AudioSamples`] containing all segments concatenated.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if any segment has a
    ///   different sample rate or channel count from the first.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::array;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let a = AudioSamples::new_mono(array![1.0f32, 2.0], sample_rate!(44100)).unwrap();
    /// let b = AudioSamples::new_mono(array![3.0f32, 4.0], sample_rate!(44100)).unwrap();
    /// let segments = NonEmptyVec::new(vec![a, b]).unwrap();
    /// let joined = AudioSamples::concatenate(&segments).unwrap();
    /// assert_eq!(joined.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    /// ```
    fn concatenate<'b>(
        segments: &'b NonEmptySlice<AudioSamples<'b, T>>,
    ) -> AudioSampleResult<AudioSamples<'b, T>> {
        // Validate all segments have the same sample rate and channel count
        let first_sample_rate = segments[0].sample_rate();
        let first_num_channels = segments[0].num_channels();
        for segment in segments.iter().skip(1) {
            if segment.sample_rate() != first_sample_rate {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "All segments must have the same sample rate",
                )));
            }
            if segment.num_channels() != first_num_channels {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "All segments must have the same number of channels",
                )));
            }
        }

        let first_sample_rate = segments[0].sample_rate();
        let first_is_mono = segments[0].is_mono();

        if first_is_mono {
            let mut all_samples = Vec::new();
            for segment in segments {
                let owned_segment = segment.clone().into_owned();

                let segment = owned_segment.as_mono().ok_or(AudioSampleError::Parameter(
                    ParameterError::invalid_value("input", "Expected mono audio data"),
                ))?;
                all_samples.extend_from_slice(segment.as_slice().ok_or(
                    AudioSampleError::Parameter(ParameterError::invalid_value(
                        "input",
                        "Mono samples must be contiguous",
                    )),
                )?);
            }

            let concatenated = Array1::from_vec(all_samples);
            AudioSamples::new_mono(concatenated, first_sample_rate)
        } else {
            let num_channels = segments[0].num_channels().get();
            let mut total_samples = 0;

            // Calculate total length
            for segment in segments {
                total_samples += segment.samples_per_channel().get();
            }

            // Create concatenated array
            let mut concatenated_data: Vec<T> =
                Vec::with_capacity(num_channels as usize * total_samples);

            // For each channel, concatenate all segments
            for channel_idx in 0..num_channels as usize {
                for segment in segments {
                    let owned_segment = segment.clone().into_owned();

                    let segment_multi =
                        owned_segment
                            .as_multi_channel()
                            .ok_or(AudioSampleError::Parameter(ParameterError::invalid_value(
                                "input",
                                "Expected multi-channel audio data",
                            )))?;
                    let channel_data = segment_multi.row(channel_idx);
                    concatenated_data.extend_from_slice(channel_data.as_slice().ok_or(
                        AudioSampleError::Parameter(ParameterError::invalid_value(
                            "input",
                            "Multi-channel samples must be contiguous",
                        )),
                    )?);
                }
            }

            let concatenated =
                Array2::from_shape_vec((num_channels as usize, total_samples), concatenated_data)?;

            AudioSamples::new_multi_channel(concatenated, first_sample_rate)
        }
    }

    /// Interleaves multiple mono signals into a single multi-channel signal.
    ///
    /// Each source becomes one channel in the output.  If only one source
    /// is provided it is returned unchanged (as an owned clone).
    ///
    /// # Arguments
    /// - `sources` — mono audio signals of equal length and sample rate.
    ///
    /// # Returns
    /// A multi-channel [`AudioSamples`] with one channel per source.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if any source is
    ///   multi-channel, or if the sources differ in sample rate or length.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::array;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let ch1 = AudioSamples::new_mono(array![1.0f32, 2.0], sample_rate!(44100)).unwrap();
    /// let ch2 = AudioSamples::new_mono(array![3.0f32, 4.0], sample_rate!(44100)).unwrap();
    /// let sources = NonEmptyVec::new(vec![ch1, ch2]).unwrap();
    /// let stereo = AudioSamples::stack(&sources).unwrap();
    /// assert_eq!(stereo.num_channels().get(), 2);
    /// ```
    fn stack(sources: &NonEmptySlice<Self>) -> AudioSampleResult<AudioSamples<'static, T>> {
        if sources.len() == NonZeroUsize::new(1).expect("1 is non-zero") {
            return Ok(sources[0].clone().into_owned());
        }

        // Validate all sources have the same sample rate and length
        let first: &AudioSamples<T> = &sources[0];
        if first.is_multi_channel() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Stacking is only supported for mono sources",
            )));
        }

        for (idx, source) in sources.iter().enumerate().skip(1) {
            if source.is_multi_channel() {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    format!(
                        "Stacking is only supported for mono sources. Audio at index {idx} is not mono"
                    ),
                )));
            }

            if source.sample_rate() != first.sample_rate() {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "All sources must have the same sample rate",
                )));
            }
            if source.samples_per_channel() != first.samples_per_channel() {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "All sources must have the same length",
                )));
            }
        }

        // Create stacked array

        let num_sources = sources.len();
        let num_samples = first.samples_per_channel();
        let mut stacked = Array2::zeros((num_sources.get(), num_samples.get()));
        for (i, source) in sources.iter().enumerate() {
            if let AudioData::Mono(arr) = &source.data {
                stacked.slice_mut(s![i, ..]).assign(&arr.view());
            }
        }

        AudioSamples::new_multi_channel(stacked, first.sample_rate())
    }

    /// Produces a weighted sum of multiple audio sources.
    ///
    /// All sources must have the same sample rate, channel count, and
    /// length.  When `weights` is `None`, equal weighting (1 / N per
    /// source) is applied.
    ///
    /// # Arguments
    /// - `sources` — the audio signals to mix.
    /// - `weights` — optional per-source gain factors.  Must have the
    ///   same length as `sources`.
    ///
    /// # Returns
    /// The mixed signal.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the sources differ in
    ///   sample rate, channel count, or length, or if the weights slice
    ///   length does not match the sources.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::Array1;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let a = AudioSamples::new_mono(Array1::from_elem(4, 1.0f32), sample_rate!(44100)).unwrap();
    /// let b = AudioSamples::new_mono(Array1::from_elem(4, 3.0f32), sample_rate!(44100)).unwrap();
    /// let sources = NonEmptyVec::new(vec![a, b]).unwrap();
    /// let mixed = AudioSamples::mix(&sources, None).unwrap();
    /// // Equal weighting: (1.0 + 3.0) / 2 = 2.0
    /// assert_eq!(mixed[0], 2.0);
    /// ```
    fn mix(
        sources: &NonEmptySlice<Self>,
        weights: Option<&NonEmptySlice<f64>>,
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        // Validate all sources have the same properties
        let first: &AudioSamples<T> = &sources[0];

        for source in sources.iter().skip(1) {
            if source.sample_rate() != first.sample_rate() {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "All sources must have the same sample rate",
                )));
            }
            if source.num_channels() != first.num_channels() {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "All sources must have the same number of channels",
                )));
            }
            if source.samples_per_channel() != first.samples_per_channel() {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "All sources must have the same length",
                )));
            }
        }

        // Validate weights if provided
        let mix_weights = if let Some(w) = weights {
            if w.len() != sources.len() {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "Number of weights must match number of sources",
                )));
            }
            w.as_slice()
        } else {
            // Equal weights
            &vec![1.0 / sources.len().get() as f64; sources.len().get()]
        };

        match &first.data {
            AudioData::Mono(_) => {
                let mut result = first.clone();
                if let AudioData::Mono(result_arr) = &mut result.data {
                    if let AudioData::Mono(_first_arr) = &first.data {
                        let weight: T = mix_weights[0].convert_to();
                        result_arr.mapv_inplace(|x| x * weight);
                    }

                    // Add remaining sources
                    for (i, source) in sources.iter().skip(1).enumerate() {
                        if let AudioData::Mono(source_arr) = &source.data {
                            let weight: T = mix_weights[i + 1].convert_to();
                            for (r, s) in result_arr.iter_mut().zip(source_arr.iter()) {
                                *r += *s * weight;
                            }
                        }
                    }
                }
                Ok(result.into_owned())
            }
            AudioData::Multi(_) => {
                let mut result = first.clone();

                if let AudioData::Multi(result_arr) = &mut result.data {
                    // Start with first source * weight
                    if let AudioData::Multi(_first_arr) = &first.data {
                        let weight: T = mix_weights[0].convert_to();
                        result_arr.mapv_inplace(|x| x * weight);
                    }

                    // Add remaining sources
                    for (i, source) in sources.iter().skip(1).enumerate() {
                        if let AudioData::Multi(source_arr) = &source.data {
                            let weight: T = mix_weights[i + 1].convert_to();
                            for (r, s) in result_arr.iter_mut().zip(source_arr.iter()) {
                                *r += *s * weight;
                            }
                        }
                    }
                }
                Ok(result.into_owned())
            }
        }
    }

    /// Applies a fade-in envelope to the beginning of the signal.
    ///
    /// The first `duration_seconds` of every channel are multiplied by
    /// an amplitude ramp from 0 to 1 shaped by `curve`.  If
    /// `duration_seconds` exceeds the signal length the entire signal
    /// is faded.
    ///
    /// # Arguments
    /// - `duration_seconds` — fade length in seconds.  Must be `> 0`.
    /// - `curve` — the envelope shape (see [`FadeCurve`]).
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `duration_seconds`
    ///   is not positive.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use audio_samples::operations::types::FadeCurve;
    /// use ndarray::Array1;
    ///
    /// let mut audio = AudioSamples::new_mono(
    ///     Array1::<f32>::ones(100), sample_rate!(100),
    /// ).unwrap();
    /// audio.fade_in(0.5, FadeCurve::Linear).unwrap();
    /// // First sample is at position 0 → gain 0
    /// assert_eq!(audio.as_slice().unwrap()[0], 0.0);
    /// ```
    fn fade_in(&mut self, duration_seconds: f64, curve: FadeCurve) -> AudioSampleResult<()> {
        if duration_seconds <= 0.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Fade duration must be positive",
            )));
        }

        let fade_samples = seconds_to_samples(duration_seconds, self.sample_rate());
        let total_samples = self.samples_per_channel().get();
        let actual_fade_samples = fade_samples.min(total_samples);

        match &mut self.data {
            AudioData::Mono(arr) => {
                for i in 0..actual_fade_samples {
                    let position = i as f64 / actual_fade_samples as f64;
                    let gain = apply_fade_curve(&curve, position as f64);
                    let gain_t: T = gain.convert_to();
                    arr[i] *= gain_t;
                }
            }
            AudioData::Multi(arr) => {
                for i in 0..actual_fade_samples {
                    let position = i as f64 / actual_fade_samples as f64;
                    let gain = apply_fade_curve(&curve, position as f64);
                    let gain_t: T = gain.convert_to();

                    for channel in 0..arr.nrows().get() {
                        arr[[channel, i]] *= gain_t;
                    }
                }
            }
        }

        Ok(())
    }

    /// Applies a fade-out envelope to the end of the signal.
    ///
    /// The last `duration_seconds` of every channel are multiplied by
    /// an amplitude ramp from 1 to 0 shaped by `curve`.  If
    /// `duration_seconds` exceeds the signal length the entire signal
    /// is faded.
    ///
    /// # Arguments
    /// - `duration_seconds` — fade length in seconds.  Must be `> 0`.
    /// - `curve` — the envelope shape (see [`FadeCurve`]).
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `duration_seconds`
    ///   is not positive.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use audio_samples::operations::types::FadeCurve;
    /// use ndarray::Array1;
    ///
    /// let mut audio = AudioSamples::new_mono(
    ///     Array1::<f32>::ones(100), sample_rate!(100),
    /// ).unwrap();
    /// audio.fade_out(0.5, FadeCurve::Linear).unwrap();
    /// // Last sample has position 0 in the fade ramp → gain ≈ 0
    /// let last = *audio.as_slice().unwrap().last().unwrap();
    /// assert!(last < 0.1);
    /// ```
    fn fade_out(&mut self, duration_seconds: f64, curve: FadeCurve) -> AudioSampleResult<()> {
        if duration_seconds <= 0.0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Fade duration must be positive",
            )));
        }

        let fade_samples = seconds_to_samples(duration_seconds, self.sample_rate());
        let total_samples = self.samples_per_channel().get();
        let actual_fade_samples = fade_samples.min(total_samples);
        let start_sample = total_samples - actual_fade_samples;

        match &mut self.data {
            AudioData::Mono(arr) => {
                for i in 0..actual_fade_samples {
                    let position = 1.0 - (i as f64 / actual_fade_samples as f64);
                    let gain = apply_fade_curve(&curve, position);
                    let gain_t: T = gain.convert_to();

                    arr[start_sample + i] *= gain_t;
                }
            }
            AudioData::Multi(arr) => {
                for i in 0..actual_fade_samples {
                    let position = 1.0 - (i as f64 / actual_fade_samples as f64);
                    let gain = apply_fade_curve(&curve, position);
                    let gain_t: T = gain.convert_to();
                    for channel in 0..arr.nrows().get() {
                        arr[[channel, start_sample + i]] *= gain_t;
                    }
                }
            }
        }

        Ok(())
    }

    /// Tiles the signal, repeating it end-to-end.
    ///
    /// Works on both mono and multi-channel audio.  A count of 1
    /// returns an owned clone of the original.
    ///
    /// # Arguments
    /// - `count` — number of repetitions.  Must be `>= 1`.
    ///
    /// # Returns
    /// A new [`AudioSamples`] whose length is `original_length × count`.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if `count` is 0.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(
    ///     array![1.0f32, 2.0], sample_rate!(44100),
    /// ).unwrap();
    /// let tiled = audio.repeat(3).unwrap();
    /// assert_eq!(tiled.as_slice().unwrap(), &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    /// ```
    fn repeat(&self, count: usize) -> AudioSampleResult<AudioSamples<'static, T>> {
        if count == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "parameter",
                "Repeat count must be greater than 0",
            )));
        }

        if count == 1 {
            return Ok(self.clone().into_owned());
        }

        match &self.data {
            AudioData::Mono(arr) => {
                let mut repeated = Array1::zeros(arr.len().get() * count);
                for i in 0..count {
                    let start = i * arr.len().get();
                    let end = start + arr.len().get();
                    repeated.slice_mut(s![start..end]).assign(&arr.view());
                }
                AudioSamples::new_mono(repeated, self.sample_rate())
            }
            AudioData::Multi(arr) => {
                let mut repeated = Array2::zeros((arr.nrows().get(), arr.ncols().get() * count));
                for i in 0..count {
                    let start = i * arr.ncols().get();
                    let end = start + arr.ncols().get();
                    repeated.slice_mut(s![.., start..end]).assign(&arr.view());
                }
                AudioSamples::new_multi_channel(repeated, self.sample_rate())
            }
        }
    }

    /// Removes leading and trailing silence from the signal.
    ///
    /// A sample (or, for multi-channel audio, an entire frame) is
    /// considered silent when its absolute value is at or below the
    /// linear equivalent of `threshold_db`.  If the entire signal is
    /// silent a zero-filled signal of the same length is returned.
    ///
    /// # Arguments
    /// - `threshold_db` — silence threshold in dB (typically negative,
    ///   e.g. `-40.0`).  Samples with amplitude at or below
    ///   `10^(threshold_db / 20)` are treated as silence.
    ///
    /// # Returns
    /// A new [`AudioSamples`] with leading and trailing silence removed.
    ///
    /// # Errors
    /// Cannot fail for valid input; errors are propagated from internal
    /// signal construction only.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::array;
    ///
    /// // 3 silent samples, 2 loud samples, 3 silent samples
    /// let data = array![0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(8)).unwrap();
    /// let trimmed = audio.trim_silence(-10.0).unwrap();
    /// assert_eq!(trimmed.samples_per_channel().get(), 2);
    /// ```
    fn trim_silence(&self, threshold_db: f64) -> AudioSampleResult<AudioSamples<'static, T>> {
        // Convert dB threshold to linear amplitude
        // dB = 20 * log10(x)  =>  x = 10^(dB / 20)
        let threshold_lin: f64 = 10.0f64.powf(threshold_db / 20.0);

        match &self.data {
            AudioData::Mono(arr) => {
                // Find first non-silent sample
                let mut start = 0usize;
                let mut found_start = false;

                for (idx, sample) in arr.iter().enumerate() {
                    let value: f64 = (*sample).convert_to();
                    if value.abs() > threshold_lin {
                        start = idx;
                        found_start = true;
                        break;
                    }
                }

                // If no non-silent sample was found, return silence
                if !found_start {
                    return Ok(AudioSamples::zeros_mono(arr.len(), self.sample_rate()));
                }

                // Find last non-silent sample
                let mut end = arr.len().get() - 1;
                for (idx, sample) in arr.iter().enumerate().rev() {
                    let value: f64 = (*sample).convert_to();
                    if value.abs() > threshold_lin {
                        end = idx;
                        break;
                    }
                }

                AudioSamples::new_mono(arr.slice(s![start..=end]).to_owned(), self.sample_rate())
            }

            AudioData::Multi(arr) => {
                let n_frames = arr.ncols().get();

                // Find first non-silent frame
                let mut start = 0usize;
                let mut found_start = false;
                for idx in 0..n_frames {
                    let col = arr.column(idx);
                    let is_silent = col.iter().all(|&x| {
                        let value: f64 = x.convert_to();
                        value <= threshold_lin
                    });
                    if !is_silent {
                        start = idx;
                        found_start = true;
                        break;
                    }
                }

                if !found_start {
                    return Ok(AudioSamples::zeros_multi(
                        ChannelCount::new(arr.nrows().get() as u32).expect("Guaranteed non-zero"),
                        arr.len(),
                        self.sample_rate(),
                    ));
                }

                // Find last non-silent frame
                let mut end = n_frames - 1;
                for idx in (0..n_frames).rev() {
                    let col = arr.column(idx);
                    let is_silent = col.iter().all(|&x| {
                        let value: f64 = x.convert_to();
                        value <= threshold_lin
                    });
                    if !is_silent {
                        end = idx;
                        break;
                    }
                }

                AudioSamples::new_multi_channel(
                    arr.slice(s![.., start..=end]).to_owned(),
                    self.sample_rate(),
                )
            }
        }
    }

    /// Returns a perturbed copy of the signal for data augmentation.
    ///
    /// Delegates to [`perturb_in_place`] on an owned clone.  The original signal
    /// is not modified.  Available only when the `random-generation`
    /// feature is enabled.
    ///
    /// # Arguments
    /// - `config` — perturbation configuration (method, parameters, and
    ///   optional deterministic seed). See [`PerturbationConfig`].
    ///
    /// # Returns
    /// A new [`AudioSamples`] with the perturbation applied.
    ///
    /// # Errors
    /// Propagates any error from validation or from the underlying
    /// perturbation method.
    ///
    /// [`perturb_in_place`]: Self::perturb_in_place
    #[cfg(all(feature = "random-generation", feature = "iir-filtering"))]
    fn perturb<'b>(&self, config: &PerturbationConfig) -> AudioSampleResult<AudioSamples<'b, T>> {
        let mut owned = self.clone();
        owned.perturb_in_place(config)?;
        Ok(owned.into_owned())
    }

    /// Applies a perturbation to the signal in place.
    ///
    /// Available only when the `random-generation` feature is enabled.
    /// When a seed is set in `config` the operation is fully
    /// deterministic.
    ///
    /// # Arguments
    /// - `config` — perturbation configuration (method, parameters, and
    ///   optional deterministic seed). See [`PerturbationConfig`].
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - Validation errors from [`PerturbationConfig`] (e.g. cutoff
    ///   above Nyquist).
    /// - Errors from the underlying perturbation method (filtering,
    ///   pitch shift, etc.).
    #[cfg(all(feature = "random-generation", feature = "iir-filtering"))]
    fn perturb_in_place(&mut self, config: &PerturbationConfig) -> AudioSampleResult<()> {
        config.validate(f64::from(self.sample_rate.get()))?;
        // Apply perturbation based on seed or use thread-local randomness
        if let Some(seed) = config.seed {
            use rand::{SeedableRng, rngs::StdRng};

            let mut rng = StdRng::seed_from_u64(seed);
            apply_perturbation_with_rng(self, &config.method, &mut rng)
        } else {
            let mut rng = rand::rng();
            apply_perturbation_with_rng(self, &config.method, &mut rng)
        }
    }

    /// Removes all silence regions throughout the signal.
    ///
    /// The signal is scanned for runs of silent samples (amplitude at
    /// or below `threshold_db`).  Any silence run at least
    /// `min_silence_duration_seconds` long is excised; the remaining
    /// non-silent segments are concatenated in order.
    ///
    /// # Arguments
    /// - `threshold_db` — silence threshold in dB (typically negative).
    /// - `min_silence_duration_seconds` — minimum length of a silence
    ///   region (in seconds) before it is removed.
    ///
    /// # Returns
    /// A new [`AudioSamples`] with qualifying silence regions removed.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the entire signal
    ///   consists of silence, resulting in an empty output.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::array;
    ///
    /// // loud(2) – silence(4) – loud(2)  at 10 Hz
    /// let data = array![1.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(10)).unwrap();
    /// // Remove silence runs >= 0.3 s (3 samples at 10 Hz)
    /// let trimmed = audio.trim_all_silence(-10.0, 0.3).unwrap();
    /// assert_eq!(trimmed.samples_per_channel().get(), 4);
    /// ```
    fn trim_all_silence(
        &self,
        threshold_db: f64,
        min_silence_duration_seconds: f64,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>> {
        let threshold_lin: f64 = 10.0f64.powf(threshold_db / 20.0);
        let sr = self.sample_rate().get();
        let min_silence_samples = (min_silence_duration_seconds * f64::from(sr)).round() as usize;

        match &self.data {
            AudioData::Mono(arr) => {
                let mut segments: Vec<(usize, usize)> = Vec::new();
                let mut in_silence = true;
                let mut silence_start = 0usize;
                let mut segment_start = 0usize;

                for (i, sample) in arr.iter().enumerate() {
                    let value: f64 = (*sample).convert_to();
                    let is_silent = value.abs() <= threshold_lin;

                    if in_silence && !is_silent {
                        // Entering a non-silent region
                        segment_start = i;
                        in_silence = false;
                    } else if !in_silence && is_silent {
                        // Entering silence
                        silence_start = i;
                        in_silence = true;
                    } else if in_silence {
                        // If silence has lasted long enough, finalise previous segment
                        if i - silence_start >= min_silence_samples && segment_start < silence_start
                        {
                            segments.push((segment_start, silence_start));
                            segment_start = silence_start;
                        }
                    }
                }

                // Handle trailing non-silent region
                if !in_silence && segment_start < arr.len().get() {
                    segments.push((segment_start, arr.len().get()));
                }

                // Concatenate non-silent segments
                let total_len: usize = segments.iter().map(|(s, e)| e - s).sum();
                let mut result = Array1::<T>::zeros(total_len);
                let mut offset = 0usize;

                for (s, e) in segments {
                    let len = e - s;
                    result
                        .slice_mut(s![offset..offset + len])
                        .assign(&arr.slice(s![s..e]));
                    offset += len;
                }

                AudioSamples::new_mono(result, self.sample_rate())
            }

            AudioData::Multi(arr) => {
                let n_channels = arr.nrows().get();
                let n_frames = arr.ncols().get();
                let mut segments: Vec<(usize, usize)> = Vec::new();
                let mut in_silence = true;
                let mut silence_start = 0usize;
                let mut segment_start = 0usize;

                for i in 0..n_frames {
                    let is_silent = arr.column(i).iter().all(|&x| {
                        let value: f64 = x.convert_to();
                        value <= threshold_lin
                    });

                    if in_silence && !is_silent {
                        segment_start = i;
                        in_silence = false;
                    } else if !in_silence && is_silent {
                        silence_start = i;
                        in_silence = true;
                    } else if in_silence
                        && i - silence_start >= min_silence_samples
                        && segment_start < silence_start
                    {
                        segments.push((segment_start, silence_start));
                        segment_start = silence_start;
                    }
                }

                if !in_silence && segment_start < n_frames {
                    segments.push((segment_start, n_frames));
                }

                let total_len: usize = segments.iter().map(|(s, e)| e - s).sum();
                let mut result = ndarray::Array2::<T>::zeros((n_channels, total_len));
                let mut offset = 0usize;

                for (s, e) in segments {
                    let len = e - s;
                    result
                        .slice_mut(s![.., offset..offset + len])
                        .assign(&arr.slice(s![.., s..e]));
                    offset += len;
                }

                AudioSamples::new_multi_channel(result, self.sample_rate())
            }
        }
    }

    /// Joins multiple owned audio segments end-to-end.
    ///
    /// Identical in behaviour to [`concatenate`] but accepts owned
    /// segments rather than borrowed ones.  All segments must share the
    /// same sample rate and channel count.
    ///
    /// # Arguments
    /// - `segments` — the owned segments to join, in order.
    ///
    /// # Returns
    /// A single [`AudioSamples`] containing all segments concatenated.
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if any segment has a
    ///   different sample rate or channel count from the first.
    ///
    /// [`concatenate`]: Self::concatenate
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioEditing, sample_rate};
    /// use ndarray::array;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let a = AudioSamples::new_mono(array![1.0f32, 2.0], sample_rate!(44100)).unwrap();
    /// let b = AudioSamples::new_mono(array![3.0f32, 4.0], sample_rate!(44100)).unwrap();
    /// let segments = NonEmptyVec::new(vec![a, b]).unwrap();
    /// let joined = AudioSamples::concatenate_owned(segments).unwrap();
    /// assert_eq!(joined.samples_per_channel().get(), 4);
    /// ```
    fn concatenate_owned<'b>(
        segments: NonEmptyVec<AudioSamples<'_, T>>,
    ) -> AudioSampleResult<AudioSamples<'b, T>>
    where
        Self: Sized,
    {
        // Validate all segments have the same sample rate and channel count
        let first_sample_rate = segments[0].sample_rate();
        let first_num_channels = segments[0].num_channels();
        for segment in segments.iter().skip(1) {
            if segment.sample_rate() != first_sample_rate {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "All segments must have the same sample rate",
                )));
            }
            if segment.num_channels() != first_num_channels {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    "All segments must have the same number of channels",
                )));
            }
        }

        let first_sample_rate = segments[0].sample_rate();
        let first_is_mono = segments[0].is_mono();

        if first_is_mono {
            let mut all_samples = Vec::new();
            for segment in &segments {
                let owned_segment = segment.clone().into_owned();

                let segment = owned_segment.as_mono().ok_or(AudioSampleError::Parameter(
                    ParameterError::invalid_value("input", "Expected mono audio data"),
                ))?;
                all_samples.extend_from_slice(segment.as_slice().ok_or(
                    AudioSampleError::Parameter(ParameterError::invalid_value(
                        "input",
                        "Mono samples must be contiguous",
                    )),
                )?);
            }

            let concatenated = Array1::from_vec(all_samples);
            AudioSamples::new_mono(concatenated, first_sample_rate)
        } else {
            let num_channels = segments[0].num_channels();
            let mut total_samples = 0;

            // Calculate total length
            for segment in &segments {
                total_samples += segment.samples_per_channel().get();
            }

            // Create concatenated array
            let mut concatenated_data: Vec<T> =
                Vec::with_capacity(num_channels.get() as usize * total_samples);

            // For each channel, concatenate all segments
            for channel_idx in 0..num_channels.get() as usize {
                for segment in &segments {
                    let owned_segment = segment.clone().into_owned();

                    let segment_multi =
                        owned_segment
                            .as_multi_channel()
                            .ok_or(AudioSampleError::Parameter(ParameterError::invalid_value(
                                "input",
                                "Expected multi-channel audio data",
                            )))?;
                    let channel_data = segment_multi.row(channel_idx);
                    concatenated_data.extend_from_slice(channel_data.as_slice().ok_or(
                        AudioSampleError::Parameter(ParameterError::invalid_value(
                            "input",
                            "Multi-channel samples must be contiguous",
                        )),
                    )?);
                }
            }

            let concatenated = Array2::from_shape_vec(
                (num_channels.get() as usize, total_samples),
                concatenated_data,
            )
            .map_err(|e| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "parameter",
                    format!("Array shape error: {e}"),
                ))
            })?;

            AudioSamples::new_multi_channel(concatenated, first_sample_rate)
        }
    }
}

/// Apply perturbation with a given RNG
#[cfg(all(feature = "random-generation", feature = "iir-filtering"))]
fn apply_perturbation_with_rng<T, R>(
    audio: &mut AudioSamples<T>,
    method: &PerturbationMethod,
    rng: &mut R,
) -> AudioSampleResult<()>
where
    R: Rng,
    T: StandardSample,
{
    match method {
        PerturbationMethod::GaussianNoise {
            target_snr_db,
            noise_color,
        } => apply_gaussian_noise_(audio, *target_snr_db, *noise_color, rng),
        PerturbationMethod::RandomGain {
            min_gain_db,
            max_gain_db,
        } => apply_random_gain_(audio, *min_gain_db, *max_gain_db, rng),
        PerturbationMethod::HighPassFilter {
            cutoff_hz,
            slope_db_per_octave,
        } => {
            use crate::operations::{AudioIirFiltering, types::IirFilterDesign};

            let order = match slope_db_per_octave {
                // safety: .max(1) ensures non-zero
                Some(slope) => unsafe {
                    NonZeroUsize::new_unchecked(((*slope / 6.02).ceil() as usize).max(1))
                },
                None => nzu!(2),
            };
            let butterworth_filter = IirFilterDesign::butterworth_highpass(order, *cutoff_hz);

            audio.apply_iir_filter(&butterworth_filter)
        }
        PerturbationMethod::LowPassFilter {
            cutoff_hz,
            slope_db_per_octave,
        } => {
            use crate::operations::{AudioIirFiltering, types::IirFilterDesign};

            let order = match slope_db_per_octave {
                // safety: .max(1) ensures non-zero
                Some(slope) => unsafe {
                    NonZeroUsize::new_unchecked(((*slope / 6.02).ceil() as usize).max(1))
                },
                None => nzu!(2),
            };
            let butterworth_filter = IirFilterDesign::butterworth_lowpass(order, *cutoff_hz);

            audio.apply_iir_filter(&butterworth_filter)
        }
        PerturbationMethod::PitchShift {
            semitones,
            preserve_formants,
        } => apply_pitch_shift_(audio, *semitones, *preserve_formants),
    }
}

/// Apply Gaussian noise to audio samples
#[cfg(feature = "random-generation")]
pub fn apply_gaussian_noise_<T, R>(
    audio: &mut AudioSamples<T>,
    target_snr_db: f64,
    noise_color: NoiseColor,
    rng: &mut R,
) -> AudioSampleResult<()>
where
    R: Rng,
    T: StandardSample,
{
    // Calculate signal RMS

    let signal_rms: f64 = audio.rms();

    // Calculate target noise RMS from SNR
    let target_noise_rms = signal_rms / 10.0f64.powf(target_snr_db / 20.0);

    let duration = audio.duration_seconds();
    let duration: Duration = Duration::from_secs_f64(duration);
    // Generate noise based on color
    let noise_audio: AudioSamples<T> = match noise_color {
        NoiseColor::White => {
            let mut noise = white_noise(duration, audio.sample_rate, target_noise_rms, None);
            // Use custom RNG for deterministic results
            apply_custom_noise_to_audio(&mut noise, target_noise_rms, rng)?;
            noise
        }
        NoiseColor::Pink => {
            let mut noise = pink_noise(duration, audio.sample_rate, target_noise_rms, None);
            apply_custom_noise_to_audio(&mut noise, target_noise_rms, rng)?;
            noise
        }
        NoiseColor::Brown => match &audio.data {
            AudioData::Mono(_) => {
                brown_noise(duration, audio.sample_rate, 0.02, target_noise_rms, None)?
            }
            AudioData::Multi(arr) => {
                let mut noise_arrays = Vec::new();
                for _ in 0..arr.nrows().get() {
                    let noise =
                        brown_noise(duration, audio.sample_rate, 0.02, target_noise_rms, None)?;
                    noise_arrays.push(noise);
                }
                let noise_arrays = NonEmptyVec::new(noise_arrays).expect("Guaranteed at least one");
                AudioSamples::stack(&noise_arrays)?
            }
        },
    };

    // Add noise to signal
    match (&mut audio.data, &noise_audio.data) {
        (AudioData::Mono(signal), AudioData::Mono(noise)) => {
            for (s, n) in signal.iter_mut().zip(noise.iter()) {
                *s += *n;
            }
        }
        (AudioData::Multi(signal), AudioData::Multi(noise)) => {
            for (s, n) in signal.iter_mut().zip(noise.iter()) {
                *s += *n;
            }
        }
        _ => {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "input",
                "Channel mismatch between signal and noise",
            )));
        }
    }

    Ok(())
}

/// Helper to apply custom noise generation using provided RNG
#[cfg(feature = "random-generation")]
fn apply_custom_noise_to_audio<T, R>(
    noise: &mut AudioSamples<T>,
    amplitude: f64,
    rng: &mut R,
) -> AudioSampleResult<()>
where
    R: Rng,
    T: StandardSample,
{
    use rand::RngExt;
    match &mut noise.data {
        AudioData::Mono(arr) => {
            
            for sample in arr.iter_mut() {
                let random_value: f64 = (rng.random_range(0.0..1.0) - 0.5) * 2.0;
                let random_value: f64 = random_value;
                *sample = (amplitude * random_value).convert_to();
            }
        }
        AudioData::Multi(arr) => {
            for sample in arr.iter_mut() {
                let random_value = (rng.random_range(0.0..1.0) - 0.5) * 2.0;
                let random_value: f64 = random_value;
                *sample = (amplitude * random_value).convert_to();
            }
        }
    }
    Ok(())
}

/// Apply random gain to audio samples
#[cfg(feature = "random-generation")]
pub fn apply_random_gain_<T, R>(
    audio: &mut AudioSamples<T>,
    min_gain_db: f64,
    max_gain_db: f64,
    rng: &mut R,
) -> AudioSampleResult<()>
where
    T: StandardSample,
    R: Rng,
{
    use rand::RngExt;

    let gain_db = rng.random_range(min_gain_db..=max_gain_db);
    let gain_linear: f64 = 10.0f64.powf(gain_db / 20.0);

    match &mut audio.data {
        AudioData::Mono(arr) => {
            for sample in arr.iter_mut() {
                let sample_f: f64 = (*sample).convert_to();
                *sample = (sample_f * gain_linear).convert_to();
            }
        }
        AudioData::Multi(arr) => {
            for sample in arr.iter_mut() {
                let sample_f: f64 = (*sample).convert_to();
                *sample = (sample_f * gain_linear).convert_to();
            }
        }
    }

    Ok(())
}

/// Apply pitch shift to audio samples (basic implementation)
pub fn apply_pitch_shift_<T>(
    audio: &mut AudioSamples<T>,
    semitones: f64,
    _preserve_formants: bool,
) -> AudioSampleResult<()>
where
    T: StandardSample,
{
    if semitones == 0.0 {
        return Ok(());
    }

    let pitch_ratio = (semitones / 12.0).exp2();

    // Simple time-domain pitch shifting using interpolation
    match &mut audio.data {
        AudioData::Mono(arr) => {
            let original_data: Vec<f64> = arr.iter().map(|&x| x.convert_to()).collect();

            for (i, sample) in arr.iter_mut().enumerate() {
                let source_index = i as f64 * pitch_ratio;
                let index_floor = source_index.floor() as usize;
                let index_frac = source_index - source_index.floor();

                if index_floor < original_data.len() {
                    let interpolated = if index_floor + 1 < original_data.len() {
                        original_data[index_floor].mul_add(
                            1.0 - index_frac,
                            original_data[index_floor + 1] * index_frac,
                        )
                    } else {
                        original_data[index_floor]
                    };
                    *sample = interpolated.convert_to();
                } else {
                    *sample = T::zero();
                }
            }
        }
        AudioData::Multi(arr) => {
            let original_data: Vec<Vec<f64>> = arr
                .outer_iter()
                .map(|channel| channel.iter().map(|&x| x.convert_to()).collect())
                .collect();

            for (ch_idx, mut channel) in arr.axis_iter_mut(Axis(0)).enumerate() {
                for (i, sample) in channel.iter_mut().enumerate() {
                    let source_index = i as f64 * pitch_ratio;
                    let index_floor = source_index.floor() as usize;
                    let index_frac = source_index - source_index.floor();

                    if index_floor < original_data[ch_idx].len() {
                        let interpolated = if index_floor + 1 < original_data[ch_idx].len() {
                            original_data[ch_idx][index_floor].mul_add(
                                1.0 - index_frac,
                                original_data[ch_idx][index_floor + 1] * index_frac,
                            )
                        } else {
                            original_data[ch_idx][index_floor]
                        };
                        *sample = interpolated.convert_to();
                    } else {
                        *sample = T::zero();
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::AudioSamples;
    use crate::sample_rate;
    use ndarray::Array1;

    #[test]
    fn test_reverse_mono_audio() {
        let samples = Array1::from(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let audio = AudioSamples::new_mono(samples.into(), sample_rate!(44100)).unwrap();

        let reversed = audio.reverse();

        if let AudioData::Mono(arr) = &reversed.data {
            let expected = vec![5.0, 4.0, 3.0, 2.0, 1.0];
            // Convert to vec for comparison since array might not be contiguous
            let actual: Vec<f32> = arr.iter().cloned().collect();
            assert_eq!(actual, expected);
        } else {
            panic!("Expected mono data");
        }
    }

    #[test]
    fn test_trim_with_time_bounds() {
        let samples = Array1::from(vec![1.0f32; 44100]); // 1 second at 44.1kHz
        let audio = AudioSamples::new_mono(samples.into(), sample_rate!(44100)).unwrap();

        let trimmed = audio.trim(0.25, 0.75).unwrap();

        // Should be 0.5 seconds = 22050 samples
        assert_eq!(trimmed.samples_per_channel().get(), 22050);
    }

    #[test]
    fn test_pad_with_silence() {
        let samples = Array1::from(vec![1.0f32; 1000]);
        let audio = AudioSamples::new_mono(samples.into(), sample_rate!(44100)).unwrap();

        let padded = audio.pad(0.1, 0.1, 0.0).unwrap(); // 0.1s = 4410 samples each side

        assert_eq!(padded.samples_per_channel().get(), 1000 + 4410 + 4410);

        if let AudioData::Mono(arr) = &padded.data {
            // Check padding is zeros
            assert_eq!(arr[0], 0.0);
            assert_eq!(arr[arr.len().get() - 1], 0.0);
            // Check original data is preserved
            assert_eq!(arr[4410], 1.0);
        }
    }

    #[test]
    fn test_split_into_segments() {
        let samples = Array1::from(vec![1.0f32; 8820]); // 0.2 seconds at 44.1kHz
        let audio = AudioSamples::new_mono(samples.into(), sample_rate!(44100)).unwrap();

        let segments = audio.split(0.05).unwrap(); // Split into 0.05s segments

        assert_eq!(segments.len(), 4); // 0.2s / 0.05s = 4 segments
        assert_eq!(
            segments[0].samples_per_channel(),
            NonZeroUsize::new(2205).unwrap()
        ); // 0.05s * 44100
    }

    #[test]
    fn test_concatenate_segments() {
        let samples1 = Array1::from(vec![1.0f32; 1000]);
        let samples2 = Array1::from(vec![2.0f32; 1000]);
        let audio1 = AudioSamples::new_mono(samples1.into(), sample_rate!(44100)).unwrap();
        let audio2 = AudioSamples::new_mono(samples2.into(), sample_rate!(44100)).unwrap();
        let audio = vec![audio1, audio2];
        let audio = NonEmptyVec::new(audio).unwrap();
        let concatenated = AudioSamples::concatenate(&audio).unwrap();

        assert_eq!(concatenated.samples_per_channel().get(), 2000);

        if let AudioData::Mono(arr) = &concatenated.data {
            assert_eq!(arr[500], 1.0); // First segment
            assert_eq!(arr[1500], 2.0); // Second segment
        }
    }

    #[test]
    fn test_mix_two_sources() {
        let samples1 = Array1::from(vec![1.0f32; 1000]);
        let samples2 = Array1::from(vec![2.0f32; 1000]);
        let audio1 = AudioSamples::new_mono(samples1.into(), sample_rate!(44100)).unwrap();
        let audio2 = AudioSamples::new_mono(samples2.into(), sample_rate!(44100)).unwrap();
        let v = NonEmptyVec::new(vec![audio1, audio2]).unwrap();
        let mixed = AudioSamples::mix(&v, None).unwrap();

        if let AudioData::Mono(arr) = &mixed.data {
            // Equal weighting: (1.0 + 2.0) / 2 = 1.5
            assert!((arr[0] - 1.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fade_operations() {
        let samples = Array1::from(vec![1.0f32; 1000]);
        let mut audio = AudioSamples::new_mono(samples.into(), sample_rate!(44100)).unwrap();

        // Test fade in
        audio.fade_in(0.01, FadeCurve::Linear).unwrap(); // 0.01s fade = 441 samples

        if let AudioData::Mono(arr) = &audio.data {
            assert_eq!(arr[0], 0.0); // Should start at 0
            assert!(arr[220] > 0.0 && arr[220] < 1.0); // Should be partially faded (middle of fade)
            // The fade applies to indices 0..441, so arr[440] should be close to but not exactly 1.0
            assert!(arr[440] > 0.99); // Should be nearly full at end of fade
        }

        // Reset and test fade out
        let samples = Array1::from(vec![1.0f32; 1000]);
        let mut audio = AudioSamples::new_mono(samples.into(), sample_rate!(44100)).unwrap();

        audio.fade_out(0.01, FadeCurve::Linear).unwrap();

        if let AudioData::Mono(arr) = &audio.data {
            // The last sample should be very close to 0 but not exactly 0 due to discrete sampling
            assert!(arr[arr.len().get() - 1] < 0.01); // Should end very close to 0
            let fade_start = arr.len().get() - 441;
            assert!(arr[fade_start + 220] > 0.0 && arr[fade_start + 220] < 1.0);
            // Should be partially faded
        }
    }

    #[test]
    fn test_repeat_audio() {
        let samples = Array1::from(vec![1.0f32, 2.0]);
        let audio = AudioSamples::new_mono(samples.into(), sample_rate!(44100)).unwrap();
        let repeated = audio.repeat(3).unwrap();

        assert_eq!(
            repeated.samples_per_channel(),
            NonZeroUsize::new(6).unwrap()
        ); // 2 * 3

        if let AudioData::Mono(arr) = &repeated.data {
            let expected = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
            assert_eq!(arr.as_slice().unwrap(), &expected);
        }
    }

    #[test]
    fn test_trim_silence() {
        let mut samples = vec![0.0f32; 1000];
        // Add some signal in the middle
        for i in 400..600 {
            samples[i] = 1.0;
        }
        let audio =
            AudioSamples::new_mono(Array1::from(samples).into(), sample_rate!(44100)).unwrap();

        let trimmed = audio.trim_silence(-10.0).unwrap();

        // Should trim silent portions at start and end
        assert_eq!(
            trimmed.samples_per_channel(),
            NonZeroUsize::new(200).unwrap()
        ); // Samples 400-599

        if let AudioData::Mono(arr) = &trimmed.data {
            assert!(arr.iter().all(|&x| x == 1.0));
        }
    }

    #[test]
    fn test_multi_source_mixing_with_weights() {
        let samples1 = Array1::from(vec![1.0f32; 100]);
        let samples2 = Array1::from(vec![2.0f32; 100]);
        let samples3 = Array1::from(vec![3.0f32; 100]);
        let audio1 = AudioSamples::new_mono(samples1.into(), sample_rate!(44100)).unwrap();
        let audio2 = AudioSamples::new_mono(samples2.into(), sample_rate!(44100)).unwrap();
        let audio3 = AudioSamples::new_mono(samples3.into(), sample_rate!(44100)).unwrap();

        let weights = NonEmptyVec::new(vec![0.5, 0.3, 0.2]).unwrap();
        let v = NonEmptyVec::new(vec![audio1, audio2, audio3]).unwrap();
        let mixed = AudioSamples::mix(&v, Some(&weights)).unwrap();

        if let AudioData::Mono(arr) = &mixed.data {
            // 1.0*0.5 + 2.0*0.3 + 3.0*0.2 = 0.5 + 0.6 + 0.6 = 1.7
            assert!((arr[0] - 1.7).abs() < 1e-6);
        }
    }

    #[cfg(all(feature = "random-generation", feature = "iir-filtering"))]
    #[test]
    fn test_perturbation_gaussian_noise() {
        use crate::operations::types::*;

        let samples = Array1::from(vec![1.0f32; 1000]);
        let audio = AudioSamples::new_mono(samples.into(), sample_rate!(44100)).unwrap();

        let config = PerturbationConfig::with_seed(
            PerturbationMethod::gaussian_noise(20.0, NoiseColor::White),
            12345,
        );

        let noisy_audio = audio.perturb(&config).unwrap();

        // The original signal should be different from the noisy signal
        if let (AudioData::Mono(original), AudioData::Mono(noisy)) =
            (&audio.data, &noisy_audio.data)
        {
            assert_ne!(original[0], noisy[0]);
            assert_eq!(original.len(), noisy.len());
        }
    }

    #[cfg(all(feature = "random-generation", feature = "iir-filtering"))]
    #[test]
    fn test_perturbation_random_gain() {
        use crate::operations::types::*;

        let samples = Array1::from(vec![1.0f32; 100]);
        let mut audio = AudioSamples::new_mono(samples.into(), sample_rate!(44100)).unwrap();

        let config =
            PerturbationConfig::with_seed(PerturbationMethod::random_gain(-3.0, 3.0), 54321);

        let original_sample = if let AudioData::Mono(arr) = &audio.data {
            arr[0]
        } else {
            panic!("Expected mono audio");
        };

        audio.perturb_in_place(&config).unwrap();

        let gained_sample = if let AudioData::Mono(arr) = &audio.data {
            arr[0]
        } else {
            panic!("Expected mono audio");
        };

        // Sample should be modified by gain
        assert_ne!(original_sample, gained_sample);
    }

    #[cfg(all(feature = "random-generation", feature = "iir-filtering"))]
    #[test]
    fn test_perturbation_high_pass_filter() {
        use crate::operations::types::*;

        let samples = Array1::from(vec![1.0f32; 100]);
        let mut audio = AudioSamples::new_mono(samples.into(), sample_rate!(44100)).unwrap();

        let config = PerturbationConfig::new(PerturbationMethod::high_pass_filter(80.0));

        audio.perturb_in_place(&config).unwrap();

        // After high-pass filtering, the signal should be modified
        // This is more of a smoke test to ensure no crashes
        if let AudioData::Mono(arr) = &audio.data {
            assert_eq!(arr.len(), NonZeroUsize::new(100).unwrap());
        }
    }

    #[cfg(all(feature = "random-generation", feature = "iir-filtering"))]
    #[test]
    fn test_perturbation_deterministic() {
        let samples = Array1::from(vec![1.0f32; 100]);
        let audio = AudioSamples::new_mono(samples.into(), sample_rate!(44100)).unwrap();

        let config = PerturbationConfig::with_seed(PerturbationMethod::random_gain(-1.0, 1.0), 42);

        let result1 = audio.perturb(&config).unwrap();
        let result2 = audio.perturb(&config).unwrap();

        // Results should be identical due to fixed seed
        if let (AudioData::Mono(arr1), AudioData::Mono(arr2)) = (&result1.data, &result2.data) {
            for (a, b) in arr1.iter().zip(arr2.iter()) {
                assert_eq!(a, b);
            }
        }
    }

    #[cfg(all(feature = "random-generation", feature = "iir-filtering"))]
    #[test]
    fn test_perturbation_validation() {
        let samples = Array1::from(vec![1.0f32; 100]);
        let mut audio = AudioSamples::new_mono(samples.into(), sample_rate!(44100)).unwrap();

        // Test invalid high-pass filter cutoff
        let invalid_config = PerturbationConfig::new(
            PerturbationMethod::high_pass_filter(50000.0), // Above Nyquist frequency
        );

        let result = audio.perturb_in_place(&invalid_config);
        assert!(result.is_err());
    }
}
