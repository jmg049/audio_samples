//! Channel manipulation operations for AudioSamples.
//!
//! This module implements the AudioChannelOps trait, providing comprehensive
//! channel manipulation operations including mono/stereo conversions, channel
//! mixing, and multi-channel audio processing.
//!
//! ## Supported Operations
//!
//! - **Mono Conversion**: Convert multi-channel audio to mono using various methods
//! - **Stereo Conversion**: Convert mono audio to stereo with configurable panning
//! - **Channel Extraction**: Extract specific channels from multi-channel audio
//! - **Channel Mixing**: Combine channels with custom weighting factors
//!
//! ## Conversion Methods
//!
//! ### Mono Conversion
//! - `Average`: Simple arithmetic mean of all channels
//! - `LeftOnly`: Use only the left channel (channel 0)
//! - `RightOnly`: Use only the right channel (channel 1)
//! - `WeightedMix`: Custom weighted combination of channels
//!
//! ### Stereo Conversion
//! - `Duplicate`: Copy mono signal to both left and right channels
//! - `Pan`: Position mono signal in stereo field with configurable balance

use crate::{
    AudioData, AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion,
    CastFrom, CastInto, ConvertTo, I24, ParameterError, ProcessingError, RealFloat,
    operations::{StereoConversionMethod, traits::AudioChannelOps, types::MonoConversionMethod},
    repr::MonoData,
    to_precision,
};
use ndarray::{Array1, Array2, Axis};

use num_traits::FloatConst;
#[cfg(feature = "simd")]
use wide::{f32x8, f64x4, i16x16, i32x8};

impl<'a, T: AudioSample> AudioChannelOps<T> for AudioSamples<'a, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
{
    fn to_mono<F>(
        &self,
        method: MonoConversionMethod<F>,
    ) -> AudioSampleResult<AudioSamples<'static, T>>
    where
        T: CastFrom<F> + ConvertTo<F>,
        F: RealFloat + ConvertTo<T>,
    {
        if self.num_channels() == 1 {
            return Ok(self.clone().into_owned());
        }

        match method {
            MonoConversionMethod::Average => match &self.data {
                AudioData::Mono(_) => {
                    // This should not happen as we check for mono above
                    Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "audio_data",
                        "Audio is already mono",
                    )))
                }
                AudioData::Multi(data) => {
                    let mono_data = data.mean_axis(Axis(0)).ok_or_else(|| {
                        AudioSampleError::Processing(ProcessingError::algorithm_failure(
                            "channel_averaging",
                            "Failed to compute average for multi-channel audio",
                        ))
                    })?;
                    Ok(AudioSamples::new_mono(mono_data, self.sample_rate()))
                }
            },
            MonoConversionMethod::Left => match &self.data {
                AudioData::Mono(_) => {
                    // This should not happen as we check for mono above
                    Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "audio_data",
                        "Audio is already mono",
                    )))
                }
                AudioData::Multi(data) => {
                    let left_channel = data.index_axis(Axis(0), 0).to_owned();
                    Ok(AudioSamples::new_mono(left_channel, self.sample_rate()))
                }
            },
            MonoConversionMethod::Right => match &self.data {
                AudioData::Mono(_) => {
                    // This should not happen as we check for mono above
                    Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "audio_data",
                        "Audio is already mono",
                    )))
                }
                AudioData::Multi(data) => {
                    assert!(
                        data.nrows() >= 2,
                        "Right channel conversion requires at least 2 channels, got {}",
                        data.nrows()
                    );
                    let right_channel = data.index_axis(Axis(0), 1).to_owned();
                    Ok(AudioSamples::new_mono(right_channel, self.sample_rate()))
                }
            },
            MonoConversionMethod::Weighted(weights) => {
                match &self.data {
                    AudioData::Mono(_) => Ok(self.clone().into_owned()),
                    AudioData::Multi(multi) => {
                        if weights.len() != multi.nrows() {
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

                        let samples_per_channel = multi.ncols();
                        let mut mono_samples = vec![T::default(); samples_per_channel];

                        // Apply weighted average across channels
                        for sample_idx in 0..samples_per_channel {
                            let mut weighted_sum = F::zero();
                            for (channel_idx, &weight) in weights.iter().enumerate() {
                                let sample_value: F = multi[(channel_idx, sample_idx)].convert_to();
                                weighted_sum += sample_value * weight;
                            }
                            mono_samples[sample_idx] = T::cast_from(weighted_sum);
                        }

                        Ok(AudioSamples::new_mono(
                            Array1::from(mono_samples),
                            self.sample_rate(),
                        ))
                    }
                }
            }
            MonoConversionMethod::Center => {
                match &self.data {
                    AudioData::Mono(_) => Ok(self.clone().into_owned()),
                    AudioData::Multi(multi) => {
                        let num_channels = multi.nrows();
                        let samples_per_channel = multi.ncols();

                        // Check if we have a center channel (5.1, 7.1 layouts typically have center as 3rd channel)
                        let mono_samples = if num_channels >= 6 {
                            // Assume 5.1+ surround: center channel is typically channel 2 (0-indexed)
                            (0..samples_per_channel).map(|i| multi[(2, i)]).collect()
                        } else if num_channels == 2 {
                            // Stereo: average left and right
                            let two: F = to_precision::<F, _>(2.0);
                            let mut mono_samples = vec![T::default(); samples_per_channel];
                            for sample_idx in 0..samples_per_channel {
                                let left: F = multi[(0, sample_idx)].convert_to();
                                let right: F = multi[(1, sample_idx)].convert_to();
                                let avg = (left + right) / two;
                                mono_samples[sample_idx] = T::cast_from(avg);
                            }
                            mono_samples
                        } else {
                            // Multi-channel but not stereo/surround: average all channels
                            let mut mono_samples = vec![T::default(); samples_per_channel];
                            for sample_idx in 0..samples_per_channel {
                                let mut sum = F::zero();
                                for channel_idx in 0..num_channels {
                                    let sample_value: F =
                                        multi[(channel_idx, sample_idx)].convert_to();
                                    sum += sample_value;
                                }
                                let avg = sum / to_precision::<F, _>(num_channels);
                                mono_samples[sample_idx] = T::cast_from(avg);
                            }
                            mono_samples
                        };

                        Ok(AudioSamples::new_mono(
                            Array1::from(mono_samples),
                            self.sample_rate(),
                        ))
                    }
                }
            }
        }
    }

    fn to_stereo<F>(
        &self,
        method: StereoConversionMethod<F>,
    ) -> AudioSampleResult<AudioSamples<'static, T>>
    where
        T: CastFrom<F> + ConvertTo<F>,
        F: RealFloat + CastInto<T> + ConvertTo<T>,
    {
        match method {
            StereoConversionMethod::Duplicate => match &self.data {
                AudioData::Mono(mono_data) => {
                    let stereo_data =
                        ndarray::stack(Axis(0), &[mono_data.view(), mono_data.view()]).map_err(
                            |e| {
                                AudioSampleError::Processing(ProcessingError::algorithm_failure(
                                    "stereo_conversion",
                                    format!("Failed to duplicate mono to stereo: {}", e),
                                ))
                            },
                        )?;
                    Ok(AudioSamples::new(
                        AudioData::Multi(stereo_data.into()),
                        self.sample_rate(),
                    ))
                }
                AudioData::Multi(_) => Ok(self.clone().into_owned()),
            },
            StereoConversionMethod::Pan(pan) => {
                // Clamp pan value to [-1.0, 1.0]
                let pan = pan.clamp(-F::one(), F::one());

                // Calculate left and right gains using equal power panning
                let pan_radians =
                    (pan + F::one()) * <F as FloatConst>::PI() / to_precision::<F, _>(4.0);
                let left_gain = num_traits::Float::cos(pan_radians);
                let right_gain = num_traits::Float::sin(pan_radians);

                match &self.data {
                    AudioData::Mono(mono) => {
                        let samples_per_channel = mono.len();
                        let mut left_channel = vec![T::default(); samples_per_channel];
                        let mut right_channel = vec![T::default(); samples_per_channel];

                        for (i, &sample) in mono.iter().enumerate() {
                            let sample_f: F = to_precision::<F, _>(sample);
                            let left = sample_f * left_gain;
                            let right = sample_f * right_gain;
                            let left = T::cast_from(left);
                            let right = T::cast_from(right);
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
                            AudioData::Multi(stereo_matrix.into()),
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
            return Ok(AudioSamples::new_mono(mono_data, self.sample_rate()));
        }

        // Stack the mono data n_channels times
        let views: Vec<_> = (0..n_channels).map(|_| mono_data.view()).collect();
        let multi_data = ndarray::stack(Axis(0), &views).map_err(|e| {
            AudioSampleError::Processing(ProcessingError::algorithm_failure(
                "duplicate_to_channels",
                format!("Failed to stack channels: {}", e),
            ))
        })?;

        Ok(AudioSamples::new_multi_channel(
            multi_data,
            self.sample_rate(),
        ))
    }

    fn extract_channel(&self, channel_index: usize) -> AudioSampleResult<AudioSamples<'static, T>> {
        if channel_index >= self.num_channels() {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "channel_index",
                channel_index.to_string(),
                "0",
                (self.num_channels() - 1).to_string(),
                "Channel index must be within available channels",
            )));
        }
        match &self.data {
            AudioData::Mono(_) => Ok(self.clone().into_owned()),
            AudioData::Multi(data) => {
                let channel: Array1<T> = data.index_axis(Axis(0), channel_index).to_owned();
                Ok(AudioSamples::new(
                    AudioData::Mono(channel.into()),
                    self.sample_rate(),
                ))
            }
        }
    }

    fn borrow_channel(&self, channel_index: usize) -> AudioSampleResult<AudioSamples<'_, T>> {
        if channel_index >= self.num_channels() {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "channel_index",
                channel_index.to_string(),
                "0",
                (self.num_channels() - 1).to_string(),
                "Channel index must be within available channels",
            )));
        }
        match &self.data {
            AudioData::Mono(data) => {
                let channel = data.view();
                Ok(AudioSamples::new(
                    AudioData::Mono(MonoData::from_view(channel)),
                    self.sample_rate(),
                ))
            }
            AudioData::Multi(data) => {
                let channel = data.index_axis(Axis(0), channel_index);
                Ok(AudioSamples::new(
                    AudioData::Mono(MonoData::from_view(channel)),
                    self.sample_rate(),
                ))
            }
        }
    }

    fn swap_channels(&mut self, channel1: usize, channel2: usize) -> AudioSampleResult<()> {
        if channel1 >= self.num_channels() || channel2 >= self.num_channels() {
            return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                "channel_indices",
                format!("{}, {}", channel1, channel2),
                "0",
                (self.num_channels() - 1).to_string(),
                "Channel indices must be within available channels",
            )));
        }

        match &mut self.data {
            AudioData::Mono(_) => Ok(()), // No channels to swap in mono
            AudioData::Multi(data) => {
                data.swap_axes(channel1, channel2);
                Ok(())
            }
        }
    }

    fn pan<F>(&mut self, pan_value: F) -> AudioSampleResult<()>
    where
        T: CastFrom<F> + ConvertTo<F>,
        F: RealFloat + CastInto<T> + ConvertTo<T>,
    {
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
                    let left_gain = F::one() - pan_value.clamp(-F::one(), F::one());
                    left.mapv_inplace(|x| {
                        let x: F = to_precision::<F, _>(x);
                        let diff = x * left_gain;
                        T::cast_from(diff)
                    });
                }

                // right
                {
                    let right = &mut data.index_axis_mut(Axis(0), 1);
                    let right_gain = F::one() + pan_value.clamp(-F::one(), F::one());
                    right.mapv_inplace(|x| {
                        let x: F = to_precision::<F, _>(x);
                        let diff = x * right_gain;
                        T::cast_from(diff)
                    });
                }

                Ok(())
            }
        }
    }

    fn balance<F>(&mut self, balance: F) -> AudioSampleResult<()>
    where
        T: CastFrom<F> + ConvertTo<F>,
        F: RealFloat + CastInto<T> + ConvertTo<T>,
    {
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
                    let left_gain = F::one() - balance.clamp(-F::one(), F::one());
                    left.mapv_inplace(|x| {
                        let x: F = to_precision::<F, _>(x);
                        let diff = x * left_gain;
                        T::cast_from(diff)
                    });
                }

                // right
                {
                    let right = &mut data.index_axis_mut(Axis(0), 1);
                    let right_gain = F::one() + balance.clamp(-F::one(), F::one());
                    right.mapv_inplace(|x| {
                        let x: F = to_precision::<F, _>(x);
                        let diff = x * right_gain;
                        T::cast_from(diff)
                    });
                }

                Ok(())
            }
        }
    }

    fn apply_to_channel<F>(&mut self, channel_index: usize, func: F) -> AudioSampleResult<()>
    where
        F: FnMut(T) -> T,
        Self: Sized,
    {
        match &mut self.data {
            AudioData::Mono(array_base) => {
                array_base.mapv_inplace(func);
            }
            AudioData::Multi(array_base) => {
                if channel_index >= array_base.shape()[0] {
                    return Err(AudioSampleError::Parameter(ParameterError::out_of_range(
                        "channel_index",
                        channel_index.to_string(),
                        "0",
                        (array_base.shape()[0] - 1).to_string(),
                        "Channel index must be within available channels",
                    )));
                }
                let mut channel = array_base.index_axis_mut(Axis(0), channel_index);
                channel.mapv_inplace(func);
            }
        }

        Ok(())
    }

    fn interleave_channels(
        channels: &[AudioSamples<'_, T>],
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        if cfg!(feature = "simd") {
            interleave_channels_simd(channels)
        } else {
            interleave_channels_base(channels)
        }
    }

    fn deinterleave_channels(&self) -> AudioSampleResult<Vec<AudioSamples<'static, T>>> {
        // Simplified implementation to resolve lifetime issues
        let num_channels = self.num_channels();
        let mut result = Vec::with_capacity(num_channels);

        for ch in 0..num_channels {
            let channel = self.extract_channel(ch)?;
            result.push(channel);
        }

        Ok(result)
    }
}

fn interleave_channels_base<'b, T: AudioSample>(
    channels: &[AudioSamples<'_, T>],
) -> AudioSampleResult<AudioSamples<'b, T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    if channels.is_empty() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "channels",
            "Cannot interleave an empty channel list",
        )));
    }

    let num_channels = channels.len();
    let samples_per_channel = channels[0].samples_per_channel();
    for ch in channels {
        if ch.samples_per_channel() != samples_per_channel {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "channels",
                "All channels must have the same length",
            )));
        }
    }

    let mut interleaved = Vec::with_capacity(num_channels * samples_per_channel);

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

fn interleave_channels_simd<'b, T: AudioSample>(
    channels: &[AudioSamples<'_, T>],
) -> AudioSampleResult<AudioSamples<'b, T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    if channels.is_empty() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "channels",
            "Cannot interleave an empty channel list",
        )));
    }

    let _num_channels = channels.len();
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

/// Deinterleave in-place: converts interleaved samples to planar layout
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
    use super::*;
    use crate::sample_rate;
    use ndarray::Array1;

    #[test]
    fn test_duplicate_to_channels_mono_to_stereo() {
        let mono = AudioSamples::new_mono(
            Array1::from(vec![0.1f32, 0.2, 0.3, 0.4]),
            sample_rate!(44100),
        );

        let stereo = mono.duplicate_to_channels(2).unwrap();

        assert_eq!(stereo.num_channels(), 2);
        assert_eq!(stereo.samples_per_channel(), 4);

        // Both channels should have the same data via interleaved
        let interleaved = stereo.to_interleaved_vec();
        // For stereo interleaved: [L0, R0, L1, R1, L2, R2, L3, R3]
        assert_eq!(interleaved.len(), 8);
        // First check that L and R values at each position match
        for i in 0..4 {
            assert_eq!(interleaved[i * 2], interleaved[i * 2 + 1]);
        }
    }

    #[test]
    fn test_duplicate_to_channels_mono_to_surround() {
        let mono =
            AudioSamples::new_mono(Array1::from(vec![1.0f32, 2.0, 3.0]), sample_rate!(48000));

        let surround = mono.duplicate_to_channels(6).unwrap(); // 5.1

        assert_eq!(surround.num_channels(), 6);
        assert_eq!(surround.samples_per_channel(), 3);

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
        let mono = AudioSamples::new_mono(Array1::from(vec![0.5f32, 0.6]), sample_rate!(44100));

        let result = mono.duplicate_to_channels(1).unwrap();

        assert_eq!(result.num_channels(), 1);
        assert!(result.is_mono());
    }

    #[test]
    fn test_duplicate_to_channels_zero_channels_error() {
        let mono = AudioSamples::new_mono(Array1::from(vec![0.1f32, 0.2]), sample_rate!(44100));

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

        let stereo = AudioSamples::new_multi_channel(data, sample_rate!(44100));
        let quad = stereo.duplicate_to_channels(4).unwrap();

        assert_eq!(quad.num_channels(), 4);
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
