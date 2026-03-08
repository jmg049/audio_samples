//! Channel manipulation operations for [`AudioSamples`].
//!
//! This module implements the [`AudioChannelOps`] trait, which provides
//! mono/stereo conversions, channel extraction, panning, balancing,
//! interleaving, and per-channel transforms on [`AudioSamples`] of any
//! supported sample type.
//!
//! Multi-channel audio is the norm in production audio.  A single trait
//! with a uniform interface lets callers convert between layouts, isolate
//! individual channels, and apply channel-aware effects without needing
//! to know whether the underlying storage is mono, stereo, or surround.
//!
//! Bring [`AudioChannelOps`] into scope and call methods on any
//! [`AudioSamples`] value.  Conversion methods that produce new audio
//! (e.g. `to_mono`) return owned `AudioSamples<'static, T>`; in-place
//! methods (e.g. `pan`) mutate through `&mut self`.
//!
//! ```
//! use audio_samples::{AudioSamples, sample_rate};
//! use audio_samples::operations::traits::AudioChannelOps;
//! use non_empty_slice::NonEmptyVec;
//!
//! let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5, -1.0]).unwrap();
//! let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
//! let stereo = audio.duplicate_to_channels(2).unwrap();
//! assert_eq!(stereo.num_channels().get(), 2);
//! ```

use crate::{
    AudioChannelOps, AudioData, AudioSample, AudioSampleError, AudioSampleResult, AudioSamples,
    CastFrom, ParameterError, ProcessingError, StandardSample,
    operations::types::{MonoConversionMethod, StereoConversionMethod},
    repr::MonoData,
};
use ndarray::{Array1, Array2, Axis};

use non_empty_slice::NonEmptySlice;
use num_traits::FloatConst;

#[cfg(feature = "simd")]
use wide::{f32x8, f64x4, i16x16, i32x8};

impl<T> AudioChannelOps for AudioSamples<'_, T>
where
    T: StandardSample,
{
    /// Convert multi-channel audio to mono using the specified method.
    ///
    /// If the audio is already mono, a clone is returned without
    /// applying the conversion method.
    ///
    /// # Arguments
    /// - `method` – The downmix strategy.  Available variants:
    ///   - `Average` – arithmetic mean of all channels.
    ///   - `Left` – channel 0 only.
    ///   - `Right` – channel 1 only.
    ///   - `Weighted(weights)` – user-supplied per-channel weights.
    ///   - `Center` – surround center channel (≥ 6 ch), or average
    ///     of left and right for stereo.
    ///
    /// # Returns
    /// An owned mono [`AudioSamples`] at the same sample rate.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if `method` is `Weighted`
    ///   and the weights vector length does not match the channel count.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use audio_samples::operations::types::MonoConversionMethod;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5, -1.0]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let stereo = audio.duplicate_to_channels(2).unwrap();
    /// let mono = stereo.to_mono(MonoConversionMethod::Average).unwrap();
    /// assert!(mono.is_mono());
    /// ```
    fn to_mono(
        &self,
        method: MonoConversionMethod,
    ) -> AudioSampleResult<AudioSamples<'static, Self::Sample>> {
        if self.num_channels().get() == 1 {
            // return a borrowed version of self to avoid cloning
            return Ok(AudioSamples::from_borrowed(
                self.data.clone().into_owned(),
                self.sample_rate(),
            ));
        }

        match method {
            MonoConversionMethod::Average => match &self.data {
                AudioData::Mono(_) => {
                    unreachable!("We check for mono at the start of the function")
                }
                AudioData::Multi(data) => {
                    let mono_data = data.mean_axis(Axis(0));
                    Ok(AudioSamples::new_mono(mono_data, self.sample_rate())?)
                }
            },
            MonoConversionMethod::Left => match &self.data {
                AudioData::Mono(_) => {
                    // This should not happen as we check for mono above
                    unreachable!("We check for mono at the start of the function")
                }
                AudioData::Multi(data) => {
                    let left_channel = data.index_axis(Axis(0), 0).to_owned();
                    Ok(AudioSamples::new_mono(left_channel, self.sample_rate())?)
                }
            },
            MonoConversionMethod::Right => match &self.data {
                AudioData::Mono(_) => {
                    // This should not happen as we check for mono above
                    unreachable!("We check for mono at the start of the function")
                }
                AudioData::Multi(data) => {
                    let right_channel = data.index_axis(Axis(0), 1).to_owned();
                    Ok(AudioSamples::new_mono(right_channel, self.sample_rate())?)
                }
            },
            MonoConversionMethod::Weighted(weights) => {
                match &self.data {
                    AudioData::Mono(_) => Ok(self.clone().into_owned()),
                    AudioData::Multi(multi) => {
                        if weights.len() != multi.nrows().get() {
                            return Err(AudioSampleError::Parameter(
                                ParameterError::invalid_value(
                                    "weights",
                                    format!(
                                        "Weight count ({}) doesn't match channel count ({})",
                                        weights.len(),
                                        multi.nrows()
                                    ),
                                ),
                            ));
                        }

                        let samples_per_channel = multi.ncols().get();
                        let mut mono_samples = vec![T::default(); samples_per_channel];

                        // Apply weighted average across channels
                        for sample_idx in 0..samples_per_channel {
                            let mut weighted_sum = 0.0;
                            for (channel_idx, &weight) in weights.iter().enumerate() {
                                let sample_value: f64 =
                                    multi[(channel_idx, sample_idx)].convert_to();
                                weighted_sum += sample_value * weight;
                            }
                            mono_samples[sample_idx] =
                                <T as CastFrom<f64>>::cast_from(weighted_sum);
                        }

                        Ok(AudioSamples::new_mono(
                            Array1::from(mono_samples),
                            self.sample_rate(),
                        )?)
                    }
                }
            }
            MonoConversionMethod::Center => {
                match &self.data {
                    AudioData::Mono(_) => Ok(self.clone().into_owned()),
                    AudioData::Multi(multi) => {
                        let num_channels = multi.nrows().get();
                        let samples_per_channel = multi.ncols().get();

                        // Check if we have a center channel (5.1, 7.1 layouts typically have center as 3rd channel)
                        let mono_samples = if num_channels >= 6 {
                            // Assume 5.1+ surround: center channel is typically channel 2 (0-indexed)
                            (0..samples_per_channel).map(|i| multi[(2, i)]).collect()
                        } else if num_channels == 2 {
                            // Stereo: average left and right
                            let two: f64 = 2.0;
                            let mut mono_samples = vec![T::default(); samples_per_channel];
                            for sample_idx in 0..samples_per_channel {
                                let left: f64 = multi[(0, sample_idx)].convert_to();
                                let right: f64 = multi[(1, sample_idx)].convert_to();
                                let avg = (left + right) / two;
                                mono_samples[sample_idx] = <T as CastFrom<f64>>::cast_from(avg);
                            }
                            mono_samples
                        } else {
                            // Multi-channel but not stereo/surround: average all channels
                            let mut mono_samples = vec![T::default(); samples_per_channel];
                            for sample_idx in 0..samples_per_channel {
                                let mut sum = 0.0;
                                for channel_idx in 0..num_channels {
                                    let sample_value: f64 =
                                        multi[(channel_idx, sample_idx)].convert_to();
                                    sum += sample_value;
                                }
                                let avg = sum / num_channels as f64;
                                mono_samples[sample_idx] = <T as CastFrom<f64>>::cast_from(avg);
                            }
                            mono_samples
                        };

                        Ok(AudioSamples::new_mono(
                            Array1::from(mono_samples),
                            self.sample_rate(),
                        )?)
                    }
                }
            }
        }
    }

    /// Convert audio to a different stereo.
    ///
    /// Behaviour depends on the chosen method:
    /// - `Duplicate` – copies mono audio to both channels; multi-channel
    ///   input is returned unchanged.
    /// - `Pan(value)` – applies equal-power panning to mono input.
    ///   The value is clamped to \[-1, 1\]: −1 is hard left, 0 is
    ///   centre, +1 is hard right.  Multi-channel input is returned
    ///   unchanged.
    /// - `Left` – extracts channel 0 as a mono signal.
    /// - `Right` – extracts channel 1 as a mono signal.
    ///
    /// # Arguments
    /// - `method` – The conversion strategy.
    ///
    /// # Returns
    /// An owned [`AudioSamples`] at the same sample rate.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if `Left` or `Right` is
    ///   chosen and the requested channel does not exist.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use audio_samples::operations::types::StereoConversionMethod;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5, -1.0]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let stereo = audio.to_stereo(StereoConversionMethod::Duplicate).unwrap();
    /// assert_eq!(stereo.num_channels().get(), 2);
    /// ```
    fn to_stereo(
        &self,
        method: StereoConversionMethod,
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        match method {
            StereoConversionMethod::Duplicate => match &self.data {
                AudioData::Mono(mono_data) => {
                    let stereo_data =
                        ndarray::stack(Axis(0), &[mono_data.view(), mono_data.view()]).map_err(
                            |e| {
                                AudioSampleError::Processing(ProcessingError::algorithm_failure(
                                    "stereo_conversion",
                                    format!("Failed to duplicate mono to stereo: {e}"),
                                ))
                            },
                        )?;

                    let stereo_data = AudioData::try_from(stereo_data)?;

                    Ok(AudioSamples::new(stereo_data, self.sample_rate()))
                }
                AudioData::Multi(_) => Ok(self.clone().into_owned()),
            },
            StereoConversionMethod::Pan(pan) => {
                // Clamp pan value to [-1.0, 1.0]
                let pan = pan.clamp(-1.0, 1.0);

                // Calculate left and right gains using equal power panning
                let pan_radians = (pan + 1.0) * f64::PI() / 4.0;
                let left_gain = num_traits::Float::cos(pan_radians);
                let right_gain = num_traits::Float::sin(pan_radians);

                match &self.data {
                    AudioData::Mono(mono) => {
                        let samples_per_channel = mono.len().get();
                        let mut left_channel = vec![T::default(); samples_per_channel];
                        let mut right_channel = vec![T::default(); samples_per_channel];

                        for (i, &sample) in mono.iter().enumerate() {
                            let sample_f: f64 = sample.cast_into();
                            let left = sample_f * left_gain;
                            let right = sample_f * right_gain;
                            let left = <T as CastFrom<f64>>::cast_from(left);
                            let right = <T as CastFrom<f64>>::cast_from(right);
                            left_channel[i] = left;
                            right_channel[i] = right;
                        }

                        let mut stereo_matrix = Array2::<T>::zeros((2, samples_per_channel));
                        for (i, &sample) in left_channel.iter().enumerate() {
                            stereo_matrix[(0, i)] = sample;
                        }
                        for (i, &sample) in right_channel.iter().enumerate() {
                            stereo_matrix[(1, i)] = sample;
                        }

                        Ok(AudioSamples::new(
                            AudioData::Multi(stereo_matrix.try_into()?),
                            self.sample_rate(),
                        ))
                    }
                    AudioData::Multi(_) => {
                        // Already multi-channel, return as-is
                        Ok(self.clone().into_owned())
                    }
                }
            }
            StereoConversionMethod::Left => self.extract_channel(0),
            StereoConversionMethod::Right => self.extract_channel(1),
        }
    }

    /// Duplicate audio into an n-channel signal.
    ///
    /// For mono input the single channel is replicated into all output
    /// channels.  For multi-channel input only channel 0 is used as the
    /// source; the remaining input channels are ignored.
    ///
    /// # Arguments
    /// - `n_channels` – Target channel count; must be ≥ 1.
    ///
    /// # Returns
    /// An owned [`AudioSamples`] with `n_channels` identical channels
    /// at the same sample rate.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if `n_channels` is 0.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let quad = audio.duplicate_to_channels(4).unwrap();
    /// assert_eq!(quad.num_channels().get(), 4);
    /// ```
    fn duplicate_to_channels(
        &self,
        n_channels: usize,
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        if n_channels == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "n_channels",
                "must be at least 1",
            )));
        }

        // Get mono data (either directly or by extracting first channel)
        let mono_data = match &self.data {
            AudioData::Mono(data) => data.as_view().to_owned(),
            AudioData::Multi(data) => {
                // For multi-channel input, use first channel
                data.index_axis(Axis(0), 0).to_owned()
            }
        };

        if n_channels == 1 {
            return AudioSamples::new_mono(mono_data, self.sample_rate());
        }

        // Stack the mono data n_channels times
        let views: Vec<_> = (0..n_channels).map(|_| mono_data.view()).collect();
        let multi_data: AudioData<T> = ndarray::stack(Axis(0), &views)
            .map_err(|e| {
                AudioSampleError::Processing(ProcessingError::algorithm_failure(
                    "duplicate_to_channels",
                    format!("Failed to stack channels: {e}"),
                ))
            })?
            .try_into()?;

        Ok(AudioSamples::new(multi_data, self.sample_rate()))
    }

    /// Extract a single channel as an owned mono signal.
    ///
    /// If the audio is already mono, a clone is returned (only
    /// `channel_index` 0 is valid in that case).
    ///
    /// # Arguments
    /// - `channel_index` – Zero-based index of the channel to extract;
    ///   must be less than the number of channels.
    ///
    /// # Returns
    /// An owned mono [`AudioSamples`] at the same sample rate.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if `channel_index` is ≥ the
    ///   number of channels.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let stereo = audio.duplicate_to_channels(2).unwrap();
    /// let ch0 = stereo.extract_channel(0).unwrap();
    /// assert!(ch0.is_mono());
    /// ```
    fn extract_channel(&self, channel_index: u32) -> AudioSampleResult<AudioSamples<'static, T>> {
        if channel_index >= self.num_channels().get() {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "channel_index",
                channel_index.to_string(),
                "0",
                (self.num_channels().get() - 1).to_string(),
                "Channel index must be within available channels",
            )));
        }
        match &self.data {
            AudioData::Mono(_) => Ok(self.clone().into_owned()),
            AudioData::Multi(data) => {
                let channel: Array1<T> =
                    data.index_axis(Axis(0), channel_index as usize).to_owned();
                Ok(AudioSamples::new(
                    AudioData::Mono(channel.try_into()?),
                    self.sample_rate(),
                ))
            }
        }
    }

    /// Borrow a single channel as a zero-copy view.
    ///
    /// The returned [`AudioSamples`] shares memory with `self`; its
    /// lifetime is tied to the borrow of `self`.
    ///
    /// # Arguments
    /// - `channel_index` – Zero-based index of the channel to borrow;
    ///   must be less than the number of channels.
    ///
    /// # Returns
    /// A borrowed mono [`AudioSamples`] at the same sample rate.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if `channel_index` is ≥ the
    ///   number of channels.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let stereo = audio.duplicate_to_channels(2).unwrap();
    /// let ch1 = stereo.borrow_channel(1).unwrap();
    /// assert!(ch1.is_mono());
    /// ```
    fn borrow_channel(&self, channel_index: u32) -> AudioSampleResult<AudioSamples<'_, T>> {
        if channel_index >= self.num_channels().get() {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "channel_index",
                channel_index.to_string(),
                "0",
                (self.num_channels().get() - 1).to_string(),
                "Channel index must be within available channels",
            )));
        }
        match &self.data {
            AudioData::Mono(data) => {
                let channel = data.view();
                Ok(AudioSamples::new(
                    AudioData::Mono(MonoData::from_view(channel)?),
                    self.sample_rate(),
                ))
            }
            AudioData::Multi(data) => {
                let channel = data.index_axis(Axis(0), channel_index as usize);
                Ok(AudioSamples::new(
                    AudioData::Mono(MonoData::from_view(channel)?),
                    self.sample_rate(),
                ))
            }
        }
    }

    /// Swap two channels in place.
    ///
    /// For mono audio the only valid index is 0, making the swap a
    /// no-op.  Passing any other index for mono audio will trigger
    /// the out-of-range error below.
    ///
    /// # Arguments
    /// - `channel1` – Zero-based index of the first channel.
    /// - `channel2` – Zero-based index of the second channel.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if either index is ≥ the
    ///   number of channels.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let mut stereo = audio.duplicate_to_channels(2).unwrap();
    /// assert!(stereo.swap_channels(0, 1).is_ok());
    /// ```
    fn swap_channels(&mut self, channel1: u32, channel2: u32) -> AudioSampleResult<()> {
        if channel1 >= self.num_channels().get() || channel2 >= self.num_channels().get() {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "channel_indices",
                format!("{channel1}, {channel2}"),
                "0",
                (self.num_channels().get() - 1).to_string(),
                "Channel indices must be within available channels",
            )));
        }

        match &mut self.data {
            AudioData::Mono(_) => Ok(()), // No channels to swap in mono
            AudioData::Multi(data) => {
                data.swap_axes(channel1 as usize, channel2 as usize);
                Ok(())
            }
        }
    }

    /// Apply linear panning to a stereo signal in place.
    ///
    /// The left channel is scaled by `1 − pan_value` and the right
    /// channel by `1 + pan_value`, after clamping `pan_value` to
    /// \[-1, 1\].  A value of 0 leaves both channels unchanged;
    /// −1 silences the right channel; +1 silences the left channel.
    ///
    /// # Arguments
    /// - `pan_value` – Panning position in the range \[-1, 1\].
    ///   Values outside this range are clamped.  −1 is hard left,
    ///   0 is centre, +1 is hard right.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if the audio is mono or
    ///   has a channel count other than 2.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let mut stereo = audio.duplicate_to_channels(2).unwrap();
    /// stereo.pan(0.5).unwrap();
    /// ```
    fn pan(&mut self, pan_value: f64) -> AudioSampleResult<()> {
        match &mut self.data {
            AudioData::Mono(_) => Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_data",
                "Cannot pan mono audio",
            ))),
            AudioData::Multi(data) => {
                if data.shape()[0] != 2 {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "channels",
                        "Panning requires stereo audio",
                    )));
                }
                // left
                {
                    let left = &mut data.index_axis_mut(Axis(0), 0);
                    let left_gain = 1.0 - pan_value.clamp(-1.0, 1.0);
                    left.mapv_inplace(|x| {
                        let x: f64 = x.cast_into();
                        let diff = x * left_gain;
                        T::cast_from(diff)
                    });
                }

                // right
                {
                    let right = &mut data.index_axis_mut(Axis(0), 1);
                    let right_gain = 1.0 + pan_value.clamp(-1.0, 1.0);
                    right.mapv_inplace(|x| {
                        let x: f64 = x.cast_into();
                        let diff = x * right_gain;
                        T::cast_from(diff)
                    });
                }

                Ok(())
            }
        }
    }

    /// Adjust the left/right balance of a stereo signal in place.
    ///
    /// The formula applied is identical to [`AudioChannelOps::pan`]:
    /// the left channel is scaled by `1 − balance` and the right
    /// channel by `1 + balance`, after clamping to \[-1, 1\].
    ///
    /// # Arguments
    /// - `balance` – Balance position in the range \[-1, 1\].
    ///   Values outside this range are clamped.  −1 is full left,
    ///   0 is centre, +1 is full right.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if the audio is mono or
    ///   has a channel count other than 2.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let mut stereo = audio.duplicate_to_channels(2).unwrap();
    /// stereo.balance(-0.3).unwrap();
    /// ```
    fn balance(&mut self, balance: f64) -> AudioSampleResult<()> {
        match &mut self.data {
            AudioData::Mono(_) => Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_data",
                "Cannot balance mono audio",
            ))),
            AudioData::Multi(data) => {
                if data.shape()[0] != 2 {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "channels",
                        "Balancing requires stereo audio",
                    )));
                }

                // left
                {
                    let left = &mut data.index_axis_mut(Axis(0), 0);
                    let left_gain = 1.0 - balance.clamp(-1.0, 1.0);
                    left.mapv_inplace(|x| {
                        let x: f64 = x.cast_into();
                        let diff = x * left_gain;
                        T::cast_from(diff)
                    });
                }

                // right
                {
                    let right = &mut data.index_axis_mut(Axis(0), 1);
                    let right_gain = 1.0 + balance.clamp(-1.0, 1.0);
                    right.mapv_inplace(|x| {
                        let x: f64 = x.cast_into();
                        let diff = x * right_gain;
                        T::cast_from(diff)
                    });
                }

                Ok(())
            }
        }
    }

    /// Apply a closure to every sample in a single channel.
    ///
    /// For mono audio the closure is applied to the single channel
    /// and `channel_index` is ignored.  For multi-channel audio
    /// `channel_index` is validated and must be less than the number
    /// of channels.
    ///
    /// # Arguments
    /// - `channel_index` – Zero-based index of the target channel.
    ///   Ignored for mono audio.
    /// - `func` – A closure that maps each sample to a new value.
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if the audio is
    ///   multi-channel and `channel_index` is ≥ the number of
    ///   channels.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let mut stereo = audio.duplicate_to_channels(2).unwrap();
    /// stereo.apply_to_channel(1, |s| s * 0.5).unwrap();
    /// ```
    fn apply_to_channel<F>(&mut self, channel_index: u32, func: F) -> AudioSampleResult<()>
    where
        F: FnMut(T) -> T,
    {
        match &mut self.data {
            AudioData::Mono(array_base) => {
                array_base.mapv_inplace(func);
            }
            AudioData::Multi(array_base) => {
                if channel_index >= array_base.shape()[0] as u32 {
                    return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                        "channel_index",
                        channel_index.to_string(),
                        "0",
                        (array_base.shape()[0] - 1).to_string(),
                        "Channel index must be within available channels",
                    )));
                }
                let mut channel = array_base.index_axis_mut(Axis(0), channel_index as usize);
                channel.mapv_inplace(func);
            }
        }

        Ok(())
    }

    /// Combine multiple mono signals into a single multi-channel signal.
    ///
    /// All input signals must have the same number of samples.  The
    /// first signal becomes channel 0, the second becomes channel 1,
    /// and so on.  The output sample rate is taken from the first
    /// input signal.
    ///
    /// # Arguments
    /// - `channels` – A non-empty slice of mono [`AudioSamples`].
    ///   All elements must share the same sample count.
    ///
    /// # Returns
    /// An owned multi-channel [`AudioSamples`] with one channel per
    /// input signal.
    ///
    /// # Errors
    /// - [crate::AudioSampleError::Parameter] – if the input signals do
    ///   not all have the same sample count.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use non_empty_slice::{NonEmptySlice, NonEmptyVec};
    ///
    /// let s0 = NonEmptyVec::new(vec![1.0f32, 0.5]).unwrap();
    /// let s1 = NonEmptyVec::new(vec![-1.0f32, -0.5]).unwrap();
    /// let ch0: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(s0, sample_rate!(44100));
    /// let ch1: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(s1, sample_rate!(44100));
    /// let arr = [ch0, ch1];
    /// let channels = NonEmptySlice::new(&arr).unwrap();
    /// let stereo = <AudioSamples<'_, f32> as AudioChannelOps>::interleave_channels(channels).unwrap();
    /// assert_eq!(stereo.num_channels().get(), 2);
    /// ```
    fn interleave_channels(
        channels: &NonEmptySlice<AudioSamples<'_, T>>,
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        #[cfg(feature = "simd")]
        {
            interleave_channels_simd(channels)
        }

        #[cfg(not(feature = "simd"))]
        {
            interleave_channels_base(channels)
        }
    }

    /// Split a multi-channel signal into individual mono signals.
    ///
    /// Each output signal contains the samples from a single input
    /// channel.  Channel 0 becomes element 0, channel 1 becomes
    /// element 1, and so on.  For mono input a single-element vector
    /// is returned.
    ///
    /// # Returns
    /// A vector of owned mono [`AudioSamples`], one per input channel.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use audio_samples::operations::traits::AudioChannelOps;
    /// use non_empty_slice::NonEmptyVec;
    ///
    /// let samples = NonEmptyVec::new(vec![1.0f32, 0.5, -0.5]).unwrap();
    /// let audio: AudioSamples<'_, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// let stereo = audio.duplicate_to_channels(2).unwrap();
    /// let channels = stereo.deinterleave_channels().unwrap();
    /// assert_eq!(channels.len(), 2);
    /// assert!(channels[0].is_mono());
    /// ```
    fn deinterleave_channels(&self) -> AudioSampleResult<Vec<AudioSamples<'static, T>>> {
        let num_channels = self.num_channels().get();
        let mut result = Vec::with_capacity(num_channels as usize);

        for ch in 0..num_channels {
            let channel = self.extract_channel(ch)?;
            result.push(channel);
        }

        Ok(result)
    }
}

fn interleave_channels_base<'b, T>(
    channels: &NonEmptySlice<AudioSamples<'_, T>>,
) -> AudioSampleResult<AudioSamples<'b, T>>
where
    T: StandardSample,
{
    let num_channels = channels.len();
    let samples_per_channel = channels[0].samples_per_channel().get();
    for ch in channels {
        if ch.samples_per_channel().get() != samples_per_channel {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "channels",
                "All channels must have the same length",
            )));
        }
    }

    let mut interleaved = Vec::with_capacity(num_channels.get() * samples_per_channel);

    // Interleave loop
    for i in 0..samples_per_channel {
        for ch in channels {
            let slice = ch.data.as_slice().ok_or_else(|| {
                AudioSampleError::Processing(ProcessingError::algorithm_failure(
                    "data_access",
                    "Failed to get data as slice",
                ))
            })?;
            interleaved.push(slice[i]);
        }
    }

    let array =
        ndarray::Array2::from_shape_vec((num_channels.get(), samples_per_channel), interleaved)?;
    AudioSamples::new_multi_channel(array, channels[0].sample_rate())
}

#[cfg(feature = "simd")]
fn interleave_channels_simd<'b, T>(
    channels: &NonEmptySlice<AudioSamples<'_, T>>,
) -> AudioSampleResult<AudioSamples<'b, T>>
where
    T: StandardSample,
{
    let samples_per_channel = channels[0].samples_per_channel();

    // Validate all channels have same length
    for ch in channels {
        if ch.samples_per_channel() != samples_per_channel {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "channels",
                "All channels must have the same length",
            )));
        }
    }

    // For now, use specialized implementations for f32 and f64, fallback to base for others
    #[cfg(feature = "simd")]
    {
        use crate::SampleType;
        use std::any::TypeId;
        let channels = channels.as_slice();
        // Use SampleType::from_type_id for cleaner dispatch
        match SampleType::from_type_id(TypeId::of::<T>()) {
            Some(SampleType::I16) => {
                // SAFETY: We've confirmed via TypeId that T is i16. The transmute is sound because:
                // 1. AudioSamples<T> has the same memory layout for all T: AudioSample
                //    (the struct contains ndarray views/owned data which are type-erased pointers
                //     plus metadata, and the sample type only affects the element type)
                // 2. Both slices have the same length (we're reinterpreting in place)
                // 3. The lifetime 'a is preserved
                let i16_channels: &[AudioSamples<i16>] = unsafe { std::mem::transmute(channels) };
                let result = interleave_i16_simd_impl(i16_channels, samples_per_channel)?;
                // SAFETY: Same reasoning - T is i16, so converting AudioSamples<i16> to AudioSamples<T>
                // is a no-op at the memory level
                return Ok(unsafe {
                    std::mem::transmute::<AudioSamples<'_, i16>, AudioSamples<'_, T>>(result)
                });
            }
            Some(SampleType::I32) => {
                // SAFETY: Same as I16 case - TypeId confirms T is i32
                let i32_channels: &[AudioSamples<i32>] = unsafe { std::mem::transmute(channels) };
                let result = interleave_i32_simd_impl(i32_channels, samples_per_channel)?;
                return Ok(unsafe {
                    std::mem::transmute::<AudioSamples<'_, i32>, AudioSamples<'_, T>>(result)
                });
            }
            Some(SampleType::F32) => {
                // SAFETY: Same as I16 case - TypeId confirms T is f32
                let f32_channels: &[AudioSamples<f32>] = unsafe { std::mem::transmute(channels) };
                let result = interleave_f32_simd_impl(f32_channels, samples_per_channel)?;
                return Ok(unsafe {
                    std::mem::transmute::<AudioSamples<'_, f32>, AudioSamples<'_, T>>(result)
                });
            }
            Some(SampleType::F64) => {
                // SAFETY: Same as I16 case - TypeId confirms T is f64
                let f64_channels: &[AudioSamples<f64>] = unsafe { std::mem::transmute(channels) };
                let result = interleave_f64_simd_impl(f64_channels, samples_per_channel)?;
                return Ok(unsafe {
                    std::mem::transmute::<AudioSamples<'_, f64>, AudioSamples<'_, T>>(result)
                });
            }
            // I24 and Unknown fall through to base implementation (no SIMD optimization)
            _ => {}
        }
    }
    // Fallback to base implementation
    interleave_channels_base(channels)
}

#[cfg(feature = "simd")]
fn interleave_i16_simd_impl<'a>(
    channels: &[AudioSamples<'a, i16>],
    samples_per_channel: usize,
) -> AudioSampleResult<AudioSamples<'static, i16>> {
    let num_channels = channels.len();
    let mut interleaved = Vec::with_capacity(num_channels * samples_per_channel);

    // Optimized case for stereo (2 channels)
    if num_channels == 2 {
        let left_data = channels[0].data.as_slice().ok_or_else(|| {
            AudioSampleError::Processing(ProcessingError::algorithm_failure(
                "data_access",
                "Failed to get left channel data as slice",
            ))
        })?;
        let right_data = channels[1].data.as_slice().ok_or_else(|| {
            AudioSampleError::Processing(ProcessingError::algorithm_failure(
                "data_access",
                "Failed to get right channel data as slice",
            ))
        })?;

        // Process 16 samples at a time with SIMD (i16x16)
        let simd_chunks = samples_per_channel / 16;

        for chunk in 0..simd_chunks {
            let start_idx = chunk * 16;

            // Convert 16 samples from each channel for SIMD processing
            let mut left_i16 = [0i16; 16];
            let mut right_i16 = [0i16; 16];

            left_i16.copy_from_slice(&left_data[start_idx..(16 + start_idx)]);
            right_i16.copy_from_slice(&right_data[start_idx..(16 + start_idx)]);

            // Load into SIMD vectors
            let left_vec = i16x16::from(left_i16);
            let right_vec = i16x16::from(right_i16);

            // Interleave by extracting individual elements and alternating
            let left_array = left_vec.to_array();
            let right_array = right_vec.to_array();

            for i in 0..16 {
                interleaved.push(left_array[i]);
                interleaved.push(right_array[i]);
            }
        }

        // Handle remainder samples
        for i in simd_chunks * 16..samples_per_channel {
            interleaved.push(left_data[i]);
            interleaved.push(right_data[i]);
        }
    } else {
        // For non-stereo, fallback to scalar implementation
        for i in 0..samples_per_channel {
            for ch in channels {
                let slice = ch.data.as_slice().ok_or_else(|| {
                    AudioSampleError::Processing(ProcessingError::algorithm_failure(
                        "data_access",
                        "Failed to get data as slice",
                    ))
                })?;
                interleaved.push(slice[i]);
            }
        }
    }

    let array = ndarray::Array2::from_shape_vec((num_channels, samples_per_channel), interleaved)
        .map_err(|e| {
        AudioSampleError::Processing(ProcessingError::algorithm_failure(
            "array_creation",
            format!("Failed to create interleaved array: {}", e),
        ))
    })?;

    Ok(AudioSamples::new_multi_channel(
        array,
        channels[0].sample_rate(),
    ))
}

#[cfg(feature = "simd")]
fn interleave_i32_simd_impl<'a>(
    channels: &[AudioSamples<'a, i32>],
    samples_per_channel: usize,
) -> AudioSampleResult<AudioSamples<'static, i32>> {
    let num_channels = channels.len();
    let mut interleaved = Vec::with_capacity(num_channels * samples_per_channel);

    // Optimized case for stereo (2 channels)
    if num_channels == 2 {
        let left_data = channels[0].data.as_slice().ok_or_else(|| {
            AudioSampleError::Processing(ProcessingError::algorithm_failure(
                "data_access",
                "Failed to get left channel data as slice",
            ))
        })?;
        let right_data = channels[1].data.as_slice().ok_or_else(|| {
            AudioSampleError::Processing(ProcessingError::algorithm_failure(
                "data_access",
                "Failed to get right channel data as slice",
            ))
        })?;

        // Process 8 samples at a time with SIMD (i32x8)
        let simd_chunks = samples_per_channel / 8;

        for chunk in 0..simd_chunks {
            let start_idx = chunk * 8;

            // Convert 8 samples from each channel for SIMD processing
            let mut left_i32 = [0i32; 8];
            let mut right_i32 = [0i32; 8];

            left_i32.copy_from_slice(&left_data[start_idx..(8 + start_idx)]);
            right_i32.copy_from_slice(&right_data[start_idx..(8 + start_idx)]);

            // Load into SIMD vectors
            let left_vec = i32x8::from(left_i32);
            let right_vec = i32x8::from(right_i32);

            // Interleave by extracting individual elements and alternating
            let left_array = left_vec.to_array();
            let right_array = right_vec.to_array();

            for i in 0..8 {
                interleaved.push(left_array[i]);
                interleaved.push(right_array[i]);
            }
        }

        // Handle remainder samples
        for i in simd_chunks * 8..samples_per_channel {
            interleaved.push(left_data[i]);
            interleaved.push(right_data[i]);
        }
    } else {
        // For non-stereo, fallback to scalar implementation
        for i in 0..samples_per_channel {
            for ch in channels {
                let slice = ch.data.as_slice().ok_or_else(|| {
                    AudioSampleError::Processing(ProcessingError::algorithm_failure(
                        "data_access",
                        "Failed to get data as slice",
                    ))
                })?;
                interleaved.push(slice[i]);
            }
        }
    }

    let array = ndarray::Array2::from_shape_vec((num_channels, samples_per_channel), interleaved)
        .map_err(|e| {
        AudioSampleError::Processing(ProcessingError::algorithm_failure(
            "array_creation",
            format!("Failed to create interleaved array: {}", e),
        ))
    })?;

    Ok(AudioSamples::new_multi_channel(
        array,
        channels[0].sample_rate(),
    ))
}

#[cfg(feature = "simd")]
fn interleave_f32_simd_impl<'a>(
    channels: &[AudioSamples<'a, f32>],
    samples_per_channel: usize,
) -> AudioSampleResult<AudioSamples<'static, f32>> {
    let num_channels = channels.len();
    let mut interleaved = Vec::with_capacity(num_channels * samples_per_channel);

    // Optimized case for stereo (2 channels)
    if num_channels == 2 {
        let left_data = channels[0].data.as_slice().ok_or_else(|| {
            AudioSampleError::Processing(ProcessingError::algorithm_failure(
                "data_access",
                "Failed to get left channel data as slice",
            ))
        })?;
        let right_data = channels[1].data.as_slice().ok_or_else(|| {
            AudioSampleError::Processing(ProcessingError::algorithm_failure(
                "data_access",
                "Failed to get right channel data as slice",
            ))
        })?;

        // Process 8 samples at a time with SIMD
        let simd_chunks = samples_per_channel / 8;

        for chunk in 0..simd_chunks {
            let start_idx = chunk * 8;

            // Convert 8 samples from each channel to f32 for SIMD processing
            let mut left_f32 = [0.0f32; 8];
            let mut right_f32 = [0.0f32; 8];

            left_f32.copy_from_slice(&left_data[start_idx..(8 + start_idx)]);
            right_f32.copy_from_slice(&right_data[start_idx..(8 + start_idx)]);

            // Load into SIMD vectors
            let left_vec = f32x8::from(left_f32);
            let right_vec = f32x8::from(right_f32);

            // Interleave by extracting individual elements and alternating
            let left_array = left_vec.to_array();
            let right_array = right_vec.to_array();

            for i in 0..8 {
                interleaved.push(left_array[i]);
                interleaved.push(right_array[i]);
            }
        }

        // Handle remainder samples
        for i in simd_chunks * 8..samples_per_channel {
            interleaved.push(left_data[i]);
            interleaved.push(right_data[i]);
        }
    } else {
        // For non-stereo, fallback to scalar implementation
        for i in 0..samples_per_channel {
            for ch in channels {
                let slice = ch.data.as_slice().ok_or_else(|| {
                    AudioSampleError::Processing(ProcessingError::algorithm_failure(
                        "data_access",
                        "Failed to get data as slice",
                    ))
                })?;
                interleaved.push(slice[i]);
            }
        }
    }

    let array = ndarray::Array2::from_shape_vec((num_channels, samples_per_channel), interleaved)
        .map_err(|e| {
        AudioSampleError::Processing(ProcessingError::algorithm_failure(
            "array_creation",
            format!("Failed to create interleaved array: {}", e),
        ))
    })?;

    Ok(AudioSamples::new_multi_channel(
        array,
        channels[0].sample_rate(),
    ))
}

#[cfg(feature = "simd")]
fn interleave_f64_simd_impl<'a>(
    channels: &[AudioSamples<'a, f64>],
    samples_per_channel: usize,
) -> AudioSampleResult<AudioSamples<'static, f64>> {
    let num_channels = channels.len();
    let mut interleaved = Vec::with_capacity(num_channels * samples_per_channel);

    // Optimized case for stereo (2 channels)
    if num_channels == 2 {
        let left_data = channels[0].data.as_slice().ok_or_else(|| {
            AudioSampleError::Processing(ProcessingError::algorithm_failure(
                "data_access",
                "Failed to get left channel data as slice",
            ))
        })?;
        let right_data = channels[1].data.as_slice().ok_or_else(|| {
            AudioSampleError::Processing(ProcessingError::algorithm_failure(
                "data_access",
                "Failed to get right channel data as slice",
            ))
        })?;

        // Process 4 samples at a time with SIMD (f64x4)
        let simd_chunks = samples_per_channel / 4;

        for chunk in 0..simd_chunks {
            let start_idx = chunk * 4;

            // Convert 4 samples from each channel to f64 for SIMD processing
            let mut left_f64 = [0.0f64; 4];
            let mut right_f64 = [0.0f64; 4];

            left_f64.copy_from_slice(&left_data[start_idx..(4 + start_idx)]);
            right_f64.copy_from_slice(&right_data[start_idx..(4 + start_idx)]);

            // Load into SIMD vectors
            let left_vec = f64x4::from(left_f64);
            let right_vec = f64x4::from(right_f64);

            // Interleave by extracting individual elements and alternating
            let left_array = left_vec.to_array();
            let right_array = right_vec.to_array();

            for i in 0..4 {
                interleaved.push(left_array[i]);
                interleaved.push(right_array[i]);
            }
        }

        // Handle remainder samples
        for i in simd_chunks * 4..samples_per_channel {
            interleaved.push(left_data[i]);
            interleaved.push(right_data[i]);
        }
    } else {
        // For non-stereo, fallback to scalar implementation
        for i in 0..samples_per_channel {
            for ch in channels {
                let slice = ch.data.as_slice().ok_or_else(|| {
                    AudioSampleError::Processing(ProcessingError::algorithm_failure(
                        "data_access",
                        "Failed to get data as slice",
                    ))
                })?;
                interleaved.push(slice[i]);
            }
        }
    }

    let array = ndarray::Array2::from_shape_vec((num_channels, samples_per_channel), interleaved)
        .map_err(|e| {
        AudioSampleError::Processing(ProcessingError::algorithm_failure(
            "array_creation",
            format!("Failed to create interleaved array: {}", e),
        ))
    })?;

    Ok(AudioSamples::new_multi_channel(
        array,
        channels[0].sample_rate(),
    ))
}

/// Rearrange an interleaved sample buffer into planar layout in place.
///
/// An interleaved buffer stores frames contiguously:
/// `[L₀, R₀, L₁, R₁, …]`.  After deinterleaving the layout is
/// planar (channel-major): `[L₀, L₁, …, R₀, R₁, …]`.
///
/// # Arguments
/// - `samples` – Mutable buffer of interleaved audio samples.  Its
///   length must be a non-zero multiple of `num_channels`.
/// - `num_channels` – Number of channels in the interleaved data;
///   must be ≥ 1.  A value of 1 is a no-op.
///
/// # Returns
/// `Ok(())` on success.  The buffer is reordered in place.
///
/// # Errors
/// - [crate::AudioSampleError::Parameter] – if `samples` is empty,
///   `num_channels` is 0, or `samples.len()` is not evenly
///   divisible by `num_channels`.
///
/// # Examples
/// ```
/// use audio_samples::operations::channels::deinterleave;
///
/// // Stereo interleaved: [L0, R0, L1, R1, L2, R2]
/// let mut buf = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
/// deinterleave(&mut buf, 2).unwrap();
/// // Planar: [L0, L1, L2, R0, R1, R2]
/// assert_eq!(buf, [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
/// ```
#[inline]
pub fn deinterleave<T: AudioSample + 'static>(
    samples: &mut [T],
    num_channels: usize,
) -> AudioSampleResult<()> {
    if samples.is_empty() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "samples",
            "Cannot deinterleave an empty sample buffer",
        )));
    }

    if num_channels == 0 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "num_channels",
            "Number of channels must be > 0",
        )));
    }

    if !samples.len().is_multiple_of(num_channels) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "samples",
            "Sample buffer length must be a multiple of number of channels",
        )));
    }

    let num_frames = samples.len() / num_channels;

    // Fast path: already effectively mono
    if num_channels == 1 {
        return Ok(());
    }

    // Allocate a single contiguous temp buffer
    let mut tmp = vec![T::zero(); samples.len()];

    // Transpose (interleaved -> planar)
    for frame in 0..num_frames {
        let frame_base = frame * num_channels;
        for ch in 0..num_channels {
            tmp[ch * num_frames + frame] = samples[frame_base + ch];
        }
    }

    // Copy back
    samples.copy_from_slice(&tmp);

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;
    use crate::sample_rate;
    use ndarray::Array1;

    #[test]
    fn test_duplicate_to_channels_mono_to_stereo() {
        let mono = AudioSamples::new_mono(
            Array1::from(vec![0.1f32, 0.2, 0.3, 0.4]),
            sample_rate!(44100),
        )
        .unwrap();

        let stereo = mono.duplicate_to_channels(2).unwrap();

        assert_eq!(stereo.num_channels().get(), 2);
        assert_eq!(stereo.samples_per_channel(), NonZeroUsize::new(4).unwrap());

        // Both channels should have the same data via interleaved
        let interleaved = stereo.to_interleaved_vec();
        // For stereo interleaved: [L0, R0, L1, R1, L2, R2, L3, R3]
        assert_eq!(interleaved.len().get(), 8);
        // First check that L and R values at each position match
        for i in 0..4 {
            assert_eq!(interleaved[i * 2], interleaved[i * 2 + 1]);
        }
    }

    #[test]
    fn test_duplicate_to_channels_mono_to_surround() {
        let mono =
            AudioSamples::new_mono(Array1::from(vec![1.0f32, 2.0, 3.0]), sample_rate!(48000))
                .unwrap();

        let surround = mono.duplicate_to_channels(6).unwrap(); // 5.1

        assert_eq!(surround.num_channels().get(), 6);
        assert_eq!(
            surround.samples_per_channel(),
            NonZeroUsize::new(3).unwrap()
        );

        // Access via underlying array
        let multi = surround.as_multi_channel().unwrap();
        let expected = [1.0f32, 2.0, 3.0];
        for ch_idx in 0..6 {
            for (s_idx, &exp) in expected.iter().enumerate() {
                assert_eq!(multi[(ch_idx, s_idx)], exp);
            }
        }
    }

    #[test]
    fn test_duplicate_to_channels_single_channel() {
        let mono =
            AudioSamples::new_mono(Array1::from(vec![0.5f32, 0.6]), sample_rate!(44100)).unwrap();

        let result = mono.duplicate_to_channels(1).unwrap();

        assert_eq!(result.num_channels().get(), 1);
        assert!(result.is_mono());
    }

    #[test]
    fn test_duplicate_to_channels_zero_channels_error() {
        let mono =
            AudioSamples::new_mono(Array1::from(vec![0.1f32, 0.2]), sample_rate!(44100)).unwrap();

        let result = mono.duplicate_to_channels(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_to_channels_from_multi() {
        // When starting with multi-channel, it should use the first channel
        let mut data = ndarray::Array2::zeros((2, 3));
        data[[0, 0]] = 1.0f32;
        data[[0, 1]] = 2.0;
        data[[0, 2]] = 3.0;
        data[[1, 0]] = 10.0; // Different data in channel 1
        data[[1, 1]] = 20.0;
        data[[1, 2]] = 30.0;

        let stereo = AudioSamples::new_multi_channel(data, sample_rate!(44100)).unwrap();
        let quad = stereo.duplicate_to_channels(4).unwrap();

        assert_eq!(quad.num_channels().get(), 4);
        // All channels should have data from channel 0
        let multi = quad.as_multi_channel().unwrap();
        let expected = [1.0f32, 2.0, 3.0];
        for ch_idx in 0..4 {
            for (s_idx, &exp) in expected.iter().enumerate() {
                assert_eq!(multi[(ch_idx, s_idx)], exp);
            }
        }
    }
}
