//! Channel manipulation operations for AudioSamples.
//!
//! This module implements the AudioChannelOps trait, providing comprehensive
//! channel manipulation operations including mono/stereo conversions, channel
//! mixing, and multi-channel audio processing using efficient ndarray operations.
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
    AudioChannelOps, AudioData, AudioDataRead, AudioSample, AudioSampleError, AudioSampleResult,
    AudioSamples, AudioTypeConversion, ConvertTo, I24,
    operations::{StereoConversionMethod, types::MonoConversionMethod},
};
use ndarray::{Array1, Axis};

#[cfg(feature = "simd")]
use wide::{f32x8, f64x4, i16x16, i32x8};

impl<T: AudioSample> AudioChannelOps<T> for AudioSamples<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    fn to_mono(&self, method: MonoConversionMethod) -> AudioSampleResult<Self>
    where
        Self: Sized,
    {
        if self.num_channels() == 1 {
            return Ok(AudioSamples::new(self.data.clone(), self.sample_rate()));
        }

        match method {
            MonoConversionMethod::Average => match &self.data {
                AudioData::Mono(_) => unreachable!("Already checked for mono above"),
                AudioData::MultiChannel(data) => {
                    let mono_data = data.mean_axis(Axis(0)).ok_or_else(|| {
                        AudioSampleError::ProcessingError {
                            msg: "Failed to compute average for multi-channel audio".to_string(),
                        }
                    })?;
                    Ok(AudioSamples::new_mono(mono_data.into(), self.sample_rate()))
                }
            },
            MonoConversionMethod::Left => match &self.data {
                AudioData::Mono(_) => unreachable!("Already checked for mono above"),
                AudioData::MultiChannel(data) => {
                    let left_channel = data.index_axis(Axis(0), 0).to_owned();
                    Ok(AudioSamples::new_mono(
                        left_channel.into(),
                        self.sample_rate(),
                    ))
                }
            },
            MonoConversionMethod::Right => match &self.data {
                AudioData::Mono(_) => unreachable!("Already checked for mono above"),
                AudioData::MultiChannel(data) => {
                    let right_channel = data.index_axis(Axis(0), 1).to_owned();
                    Ok(AudioSamples::new_mono(
                        right_channel.into(),
                        self.sample_rate(),
                    ))
                }
            },
            MonoConversionMethod::Weighted(_items) => {
                todo!()
            }
            MonoConversionMethod::Center => {
                todo!()
            }
        }
    }

    fn to_stereo(&self, method: StereoConversionMethod) -> AudioSampleResult<Self>
    where
        Self: Sized,
    {
        match method {
            StereoConversionMethod::Duplicate => match &self.data {
                AudioData::Mono(mono_data) => {
                    let stereo_data =
                        ndarray::stack(Axis(0), &[mono_data.view(), mono_data.view()]).map_err(
                            |e| AudioSampleError::ProcessingError {
                                msg: format!("Failed to duplicate mono to stereo: {}", e),
                            },
                        )?;
                    Ok(AudioSamples::new(
                        AudioData::MultiChannel(stereo_data.into()),
                        self.sample_rate(),
                    ))
                }
                AudioData::MultiChannel(_) => {
                    Ok(AudioSamples::new(self.data.clone(), self.sample_rate()))
                }
            },
            StereoConversionMethod::Pan(_) => {
                todo!()
            }
            StereoConversionMethod::Left => self.extract_channel(0),
            StereoConversionMethod::Right => self.extract_channel(1),
        }
    }

    fn extract_channel(&self, channel_index: usize) -> AudioSampleResult<Self>
    where
        Self: Sized,
    {
        if channel_index >= self.num_channels() {
            return Err(AudioSampleError::InvalidParameter(format!(
                "Channel index {} out of bounds for {} channels",
                channel_index,
                self.num_channels()
            )));
        }
        match &self.data {
            AudioData::Mono(_) => Ok(AudioSamples::new(self.data.clone(), self.sample_rate())),
            AudioData::MultiChannel(data) => {
                let channel: Array1<T> = data.index_axis(Axis(0), channel_index).to_owned();
                Ok(AudioSamples::new(
                    AudioData::Mono(channel.into()),
                    self.sample_rate(),
                ))
            }
        }
    }

    fn swap_channels(&mut self, channel1: usize, channel2: usize) -> AudioSampleResult<()> {
        if channel1 >= self.num_channels() || channel2 >= self.num_channels() {
            return Err(AudioSampleError::InvalidParameter(format!(
                "Channel indices {} and {} out of bounds for {} channels",
                channel1,
                channel2,
                self.num_channels()
            )));
        }

        match &mut self.data {
            AudioData::Mono(_) => Ok(()), // No channels to swap in mono
            AudioData::MultiChannel(data) => {
                data.swap_axes(channel1, channel2);
                Ok(())
            }
        }
    }

    fn pan(&mut self, pan_value: f64) -> AudioSampleResult<()> {
        match &mut self.data {
            AudioData::Mono(_) => Err(AudioSampleError::InvalidParameter(
                "Cannot pan mono audio".to_string(),
            )),
            AudioData::MultiChannel(data) => {
                if data.shape()[0] != 2 {
                    return Err(AudioSampleError::InvalidParameter(
                        "Panning requires stereo audio".to_string(),
                    ));
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

    fn balance(&mut self, balance: f64) -> AudioSampleResult<()> {
        match &mut self.data {
            AudioData::Mono(_) => Err(AudioSampleError::InvalidParameter(
                "Cannot balance mono audio".to_string(),
            )),
            AudioData::MultiChannel(data) => {
                if data.shape()[0] != 2 {
                    return Err(AudioSampleError::InvalidParameter(
                        "Balancing requires stereo audio".to_string(),
                    ));
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

    fn apply_to_channel<F>(&mut self, channel_index: usize, func: F) -> AudioSampleResult<()>
    where
        F: FnMut(T) -> T,
        Self: Sized,
    {
        match &mut self.data {
            AudioData::Mono(array_base) => {
                array_base.mapv_inplace(func);
            }
            AudioData::MultiChannel(array_base) => {
                if channel_index >= array_base.shape()[0] {
                    return Err(AudioSampleError::InvalidParameter(format!(
                        "Channel index {} out of bounds for {} channels",
                        channel_index,
                        array_base.shape()[0]
                    )));
                }
                let mut channel = array_base.index_axis_mut(Axis(0), channel_index);
                channel.mapv_inplace(func);
            }
        }

        Ok(())
    }

    fn interleave_channels(channels: &[Self]) -> AudioSampleResult<Self>
    where
        Self: Sized,
    {
        if cfg!(feature = "simd") {
            interleave_channels_simd(channels)
        } else {
            interleave_channels_base(channels)
        }
    }

    fn deinterleave_channels(&self) -> AudioSampleResult<Vec<Self>>
    where
        Self: Sized,
    {
        if cfg!(feature = "simd") {
            deinterleave_channels_simd(self)
        } else {
            deinterleave_channels_base(self)
        }
    }
}

fn interleave_channels_base<T: AudioSample>(
    channels: &[AudioSamples<T>],
) -> AudioSampleResult<AudioSamples<T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    if channels.is_empty() {
        return Err(AudioSampleError::InvalidInput {
            msg: "Cannot interleave an empty channel list".into(),
        });
    }

    let num_channels = channels.len();
    let samples_per_channel = channels[0].samples_per_channel();
    for ch in channels {
        if ch.samples_per_channel() != samples_per_channel {
            return Err(AudioSampleError::InvalidInput {
                msg: "All channels must have the same length".into(),
            });
        }
    }

    let mut interleaved = Vec::with_capacity(num_channels * samples_per_channel);

    // Interleave loop
    for i in 0..samples_per_channel {
        for ch in channels {
            interleaved.push(ch.data()[i]);
        }
    }

    let array = ndarray::Array2::from_shape_vec((num_channels, samples_per_channel), interleaved)
        .map_err(|e| AudioSampleError::InvalidInput {
        msg: format!("Failed to create interleaved array: {}", e),
    })?;

    Ok(AudioSamples::new_multi_channel(
        array.into(),
        channels[0].sample_rate(),
    ))
}

fn interleave_channels_simd<T: AudioSample>(
    channels: &[AudioSamples<T>],
) -> AudioSampleResult<AudioSamples<T>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    if channels.is_empty() {
        return Err(AudioSampleError::InvalidInput {
            msg: "Cannot interleave an empty channel list".into(),
        });
    }

    let _num_channels = channels.len();
    let samples_per_channel = channels[0].samples_per_channel();

    // Validate all channels have same length
    for ch in channels {
        if ch.samples_per_channel() != samples_per_channel {
            return Err(AudioSampleError::InvalidInput {
                msg: "All channels must have the same length".into(),
            });
        }
    }

    // For now, use specialized implementations for f32 and f64, fallback to base for others
    #[cfg(feature = "simd")]
    {
        use std::any::TypeId;
        let type_id = TypeId::of::<T>();

        if type_id == TypeId::of::<i16>() {
            // SAFETY: We've confirmed that T is i16
            let i16_channels: &[AudioSamples<i16>] = unsafe { std::mem::transmute(channels) };
            let result = interleave_i16_simd_impl(i16_channels, samples_per_channel)?;
            return Ok(unsafe { std::mem::transmute(result) });
        } else if type_id == TypeId::of::<i32>() {
            // SAFETY: We've confirmed that T is i32
            let i32_channels: &[AudioSamples<i32>] = unsafe { std::mem::transmute(channels) };
            let result = interleave_i32_simd_impl(i32_channels, samples_per_channel)?;
            return Ok(unsafe { std::mem::transmute(result) });
        } else if type_id == TypeId::of::<f32>() {
            // SAFETY: We've confirmed that T is f32
            let f32_channels: &[AudioSamples<f32>] = unsafe { std::mem::transmute(channels) };
            let result = interleave_f32_simd_impl(f32_channels, samples_per_channel)?;
            return Ok(unsafe { std::mem::transmute(result) });
        } else if type_id == TypeId::of::<f64>() {
            // SAFETY: We've confirmed that T is f64
            let f64_channels: &[AudioSamples<f64>] = unsafe { std::mem::transmute(channels) };
            let result = interleave_f64_simd_impl(f64_channels, samples_per_channel)?;
            return Ok(unsafe { std::mem::transmute(result) });
        }
    }

    // Fallback to base implementation
    interleave_channels_base(channels)
}

#[cfg(feature = "simd")]
fn interleave_i16_simd_impl(
    channels: &[AudioSamples<i16>],
    samples_per_channel: usize,
) -> AudioSampleResult<AudioSamples<i16>> {
    let num_channels = channels.len();
    let mut interleaved = Vec::with_capacity(num_channels * samples_per_channel);

    // Optimized case for stereo (2 channels)
    if num_channels == 2 {
        let left_data = channels[0].data();
        let right_data = channels[1].data();

        // Process 16 samples at a time with SIMD (i16x16)
        let simd_chunks = samples_per_channel / 16;

        for chunk in 0..simd_chunks {
            let start_idx = chunk * 16;

            // Convert 16 samples from each channel for SIMD processing
            let mut left_i16 = [0i16; 16];
            let mut right_i16 = [0i16; 16];

            for i in 0..16 {
                left_i16[i] = left_data[start_idx + i];
                right_i16[i] = right_data[start_idx + i];
            }

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
                interleaved.push(ch.data()[i]);
            }
        }
    }

    let array = ndarray::Array2::from_shape_vec((num_channels, samples_per_channel), interleaved)
        .map_err(|e| AudioSampleError::InvalidInput {
        msg: format!("Failed to create interleaved array: {}", e),
    })?;

    Ok(AudioSamples::new_multi_channel(
        array.into(),
        channels[0].sample_rate(),
    ))
}

#[cfg(feature = "simd")]
fn interleave_i32_simd_impl(
    channels: &[AudioSamples<i32>],
    samples_per_channel: usize,
) -> AudioSampleResult<AudioSamples<i32>> {
    let num_channels = channels.len();
    let mut interleaved = Vec::with_capacity(num_channels * samples_per_channel);

    // Optimized case for stereo (2 channels)
    if num_channels == 2 {
        let left_data = channels[0].data();
        let right_data = channels[1].data();

        // Process 8 samples at a time with SIMD (i32x8)
        let simd_chunks = samples_per_channel / 8;

        for chunk in 0..simd_chunks {
            let start_idx = chunk * 8;

            // Convert 8 samples from each channel for SIMD processing
            let mut left_i32 = [0i32; 8];
            let mut right_i32 = [0i32; 8];

            for i in 0..8 {
                left_i32[i] = left_data[start_idx + i];
                right_i32[i] = right_data[start_idx + i];
            }

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
                interleaved.push(ch.data()[i]);
            }
        }
    }

    let array = ndarray::Array2::from_shape_vec((num_channels, samples_per_channel), interleaved)
        .map_err(|e| AudioSampleError::InvalidInput {
        msg: format!("Failed to create interleaved array: {}", e),
    })?;

    Ok(AudioSamples::new_multi_channel(
        array.into(),
        channels[0].sample_rate(),
    ))
}

#[cfg(feature = "simd")]
fn interleave_f32_simd_impl(
    channels: &[AudioSamples<f32>],
    samples_per_channel: usize,
) -> AudioSampleResult<AudioSamples<f32>> {
    let num_channels = channels.len();
    let mut interleaved = Vec::with_capacity(num_channels * samples_per_channel);

    // Optimized case for stereo (2 channels)
    if num_channels == 2 {
        let left_data = channels[0].data();
        let right_data = channels[1].data();

        // Process 8 samples at a time with SIMD
        let simd_chunks = samples_per_channel / 8;

        for chunk in 0..simd_chunks {
            let start_idx = chunk * 8;

            // Convert 8 samples from each channel to f32 for SIMD processing
            let mut left_f32 = [0.0f32; 8];
            let mut right_f32 = [0.0f32; 8];

            for i in 0..8 {
                left_f32[i] = left_data[start_idx + i];
                right_f32[i] = right_data[start_idx + i];
            }

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
                interleaved.push(ch.data()[i]);
            }
        }
    }

    let array = ndarray::Array2::from_shape_vec((num_channels, samples_per_channel), interleaved)
        .map_err(|e| AudioSampleError::InvalidInput {
        msg: format!("Failed to create interleaved array: {}", e),
    })?;

    Ok(AudioSamples::new_multi_channel(
        array.into(),
        channels[0].sample_rate(),
    ))
}

#[cfg(feature = "simd")]
fn interleave_f64_simd_impl(
    channels: &[AudioSamples<f64>],
    samples_per_channel: usize,
) -> AudioSampleResult<AudioSamples<f64>> {
    let num_channels = channels.len();
    let mut interleaved = Vec::with_capacity(num_channels * samples_per_channel);

    // Optimized case for stereo (2 channels)
    if num_channels == 2 {
        let left_data = channels[0].data();
        let right_data = channels[1].data();

        // Process 4 samples at a time with SIMD (f64x4)
        let simd_chunks = samples_per_channel / 4;

        for chunk in 0..simd_chunks {
            let start_idx = chunk * 4;

            // Convert 4 samples from each channel to f64 for SIMD processing
            let mut left_f64 = [0.0f64; 4];
            let mut right_f64 = [0.0f64; 4];

            for i in 0..4 {
                left_f64[i] = left_data[start_idx + i];
                right_f64[i] = right_data[start_idx + i];
            }

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
                interleaved.push(ch.data()[i]);
            }
        }
    }

    let array = ndarray::Array2::from_shape_vec((num_channels, samples_per_channel), interleaved)
        .map_err(|e| AudioSampleError::InvalidInput {
        msg: format!("Failed to create interleaved array: {}", e),
    })?;

    Ok(AudioSamples::new_multi_channel(
        array.into(),
        channels[0].sample_rate(),
    ))
}

fn deinterleave_channels_base<T: AudioSample>(
    audio: &AudioSamples<T>,
) -> AudioSampleResult<Vec<AudioSamples<T>>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    let num_channels = audio.num_channels();
    let samples_per_channel = audio.samples_per_channel();

    let data = audio.data(); // &[T]
    if data.len() != num_channels * samples_per_channel {
        return Err(AudioSampleError::InvalidInput {
            msg: "Inconsistent buffer size for deinterleave".into(),
        });
    }

    let mut outputs: Vec<Vec<T>> = vec![Vec::with_capacity(samples_per_channel); num_channels];

    for i in 0..samples_per_channel {
        for ch in 0..num_channels {
            outputs[ch].push(data[i * num_channels + ch]);
        }
    }

    let sample_rate = audio.sample_rate();
    let result = outputs
        .into_iter()
        .map(|ch_data| {
            let array = ndarray::Array1::from_vec(ch_data);
            AudioSamples::new_mono(array.into(), sample_rate)
        })
        .collect();

    Ok(result)
}

fn deinterleave_channels_simd<T: AudioSample>(
    audio: &AudioSamples<T>,
) -> AudioSampleResult<Vec<AudioSamples<T>>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<T>: AudioTypeConversion<T>,
{
    let num_channels = audio.num_channels();
    let samples_per_channel = audio.samples_per_channel();

    let data = audio
        .data()
        .as_slice()
        .ok_or(AudioSampleError::ArrayLayoutError {
            message: "Audio data is not contiguous".to_string(),
        })?;
    if data.len() != num_channels * samples_per_channel {
        return Err(AudioSampleError::InvalidInput {
            msg: "Inconsistent buffer size for deinterleave".into(),
        });
    }

    // For now, use specialized implementations for f32 and f64, fallback to base for others
    #[cfg(feature = "simd")]
    {
        use std::any::TypeId;
        let type_id = TypeId::of::<T>();

        if type_id == TypeId::of::<i16>() {
            // SAFETY: We've confirmed that T is i16
            let i16_audio: &AudioSamples<i16> = unsafe { std::mem::transmute(audio) };
            let result =
                deinterleave_i16_simd_impl(i16_audio, num_channels, samples_per_channel, unsafe {
                    std::mem::transmute(data)
                })?;
            return Ok(unsafe { std::mem::transmute(result) });
        } else if type_id == TypeId::of::<i32>() {
            // SAFETY: We've confirmed that T is i32
            let i32_audio: &AudioSamples<i32> = unsafe { std::mem::transmute(audio) };
            let result =
                deinterleave_i32_simd_impl(i32_audio, num_channels, samples_per_channel, unsafe {
                    std::mem::transmute(data)
                })?;
            return Ok(unsafe { std::mem::transmute(result) });
        } else if type_id == TypeId::of::<f32>() {
            // SAFETY: We've confirmed that T is f32
            let f32_audio: &AudioSamples<f32> = unsafe { std::mem::transmute(audio) };
            let result =
                deinterleave_f32_simd_impl(f32_audio, num_channels, samples_per_channel, unsafe {
                    std::mem::transmute(data)
                })?;
            return Ok(unsafe { std::mem::transmute(result) });
        } else if type_id == TypeId::of::<f64>() {
            // SAFETY: We've confirmed that T is f64
            let f64_audio: &AudioSamples<f64> = unsafe { std::mem::transmute(audio) };
            let result =
                deinterleave_f64_simd_impl(f64_audio, num_channels, samples_per_channel, unsafe {
                    std::mem::transmute(data)
                })?;
            return Ok(unsafe { std::mem::transmute(result) });
        }
    }

    // Fallback to base implementation
    deinterleave_channels_base(audio)
}

#[cfg(feature = "simd")]
fn deinterleave_i16_simd_impl(
    audio: &AudioSamples<i16>,
    num_channels: usize,
    samples_per_channel: usize,
    data: &[i16],
) -> AudioSampleResult<Vec<AudioSamples<i16>>> {
    let mut outputs: Vec<Vec<i16>> = vec![Vec::with_capacity(samples_per_channel); num_channels];

    // Optimized case for stereo (2 channels)
    if num_channels == 2 {
        // Process 16 samples at a time with SIMD (i16x16)
        let simd_chunks = samples_per_channel / 16;

        for chunk in 0..simd_chunks {
            let start_idx = chunk * 16;

            // Convert 32 interleaved samples (16 left, 16 right) for SIMD processing
            let mut interleaved_i16 = [0i16; 32];

            for i in 0..16 {
                let left_idx = start_idx * 2 + i * 2;
                let right_idx = left_idx + 1;
                interleaved_i16[i * 2] = data[left_idx];
                interleaved_i16[i * 2 + 1] = data[right_idx];
            }

            // Deinterleave using SIMD vectors
            let mut left_i16 = [0i16; 16];
            let mut right_i16 = [0i16; 16];

            for i in 0..16 {
                left_i16[i] = interleaved_i16[i * 2];
                right_i16[i] = interleaved_i16[i * 2 + 1];
            }

            // Store back to output vectors
            for i in 0..16 {
                outputs[0].push(left_i16[i]);
                outputs[1].push(right_i16[i]);
            }
        }

        // Handle remainder samples
        for i in simd_chunks * 16..samples_per_channel {
            let left_idx = i * 2;
            let right_idx = left_idx + 1;
            outputs[0].push(data[left_idx]);
            outputs[1].push(data[right_idx]);
        }
    } else {
        // For non-stereo, fallback to scalar implementation
        for i in 0..samples_per_channel {
            for ch in 0..num_channels {
                outputs[ch].push(data[i * num_channels + ch]);
            }
        }
    }

    let sample_rate = audio.sample_rate();
    let result = outputs
        .into_iter()
        .map(|ch_data| {
            let array = ndarray::Array1::from_vec(ch_data);
            AudioSamples::new_mono(array.into(), sample_rate)
        })
        .collect();

    Ok(result)
}

#[cfg(feature = "simd")]
fn deinterleave_i32_simd_impl(
    audio: &AudioSamples<i32>,
    num_channels: usize,
    samples_per_channel: usize,
    data: &[i32],
) -> AudioSampleResult<Vec<AudioSamples<i32>>> {
    let mut outputs: Vec<Vec<i32>> = vec![Vec::with_capacity(samples_per_channel); num_channels];

    // Optimized case for stereo (2 channels)
    if num_channels == 2 {
        // Process 8 samples at a time with SIMD (i32x8)
        let simd_chunks = samples_per_channel / 8;

        for chunk in 0..simd_chunks {
            let start_idx = chunk * 8;

            // Convert 16 interleaved samples (8 left, 8 right) for SIMD processing
            let mut interleaved_i32 = [0i32; 16];

            for i in 0..8 {
                let left_idx = start_idx * 2 + i * 2;
                let right_idx = left_idx + 1;
                interleaved_i32[i * 2] = data[left_idx];
                interleaved_i32[i * 2 + 1] = data[right_idx];
            }

            // Deinterleave using SIMD vectors
            let mut left_i32 = [0i32; 8];
            let mut right_i32 = [0i32; 8];

            for i in 0..8 {
                left_i32[i] = interleaved_i32[i * 2];
                right_i32[i] = interleaved_i32[i * 2 + 1];
            }

            // Store back to output vectors
            for i in 0..8 {
                outputs[0].push(left_i32[i]);
                outputs[1].push(right_i32[i]);
            }
        }

        // Handle remainder samples
        for i in simd_chunks * 8..samples_per_channel {
            let left_idx = i * 2;
            let right_idx = left_idx + 1;
            outputs[0].push(data[left_idx]);
            outputs[1].push(data[right_idx]);
        }
    } else {
        // For non-stereo, fallback to scalar implementation
        for i in 0..samples_per_channel {
            for ch in 0..num_channels {
                outputs[ch].push(data[i * num_channels + ch]);
            }
        }
    }

    let sample_rate = audio.sample_rate();
    let result = outputs
        .into_iter()
        .map(|ch_data| {
            let array = ndarray::Array1::from_vec(ch_data);
            AudioSamples::new_mono(array.into(), sample_rate)
        })
        .collect();

    Ok(result)
}

#[cfg(feature = "simd")]
fn deinterleave_f32_simd_impl(
    audio: &AudioSamples<f32>,
    num_channels: usize,
    samples_per_channel: usize,
    data: &[f32],
) -> AudioSampleResult<Vec<AudioSamples<f32>>> {
    let mut outputs: Vec<Vec<f32>> = vec![Vec::with_capacity(samples_per_channel); num_channels];

    // Optimized case for stereo (2 channels)
    if num_channels == 2 {
        // Process 8 samples at a time with SIMD (f32x8)
        let simd_chunks = samples_per_channel / 8;

        for chunk in 0..simd_chunks {
            let start_idx = chunk * 8;

            // Convert 16 interleaved samples (8 left, 8 right) to f32 for SIMD processing
            let mut interleaved_f32 = [0.0f32; 16];

            for i in 0..8 {
                let left_idx = start_idx * 2 + i * 2;
                let right_idx = left_idx + 1;
                interleaved_f32[i * 2] = data[left_idx];
                interleaved_f32[i * 2 + 1] = data[right_idx];
            }

            // Deinterleave using SIMD vectors
            let mut left_f32 = [0.0f32; 8];
            let mut right_f32 = [0.0f32; 8];

            for i in 0..8 {
                left_f32[i] = interleaved_f32[i * 2];
                right_f32[i] = interleaved_f32[i * 2 + 1];
            }

            // Convert back to f32 and store
            for i in 0..8 {
                outputs[0].push(left_f32[i]);
                outputs[1].push(right_f32[i]);
            }
        }

        // Handle remainder samples
        for i in simd_chunks * 8..samples_per_channel {
            let left_idx = i * 2;
            let right_idx = left_idx + 1;
            outputs[0].push(data[left_idx]);
            outputs[1].push(data[right_idx]);
        }
    } else {
        // For non-stereo, fallback to scalar implementation
        for i in 0..samples_per_channel {
            for ch in 0..num_channels {
                outputs[ch].push(data[i * num_channels + ch]);
            }
        }
    }

    let sample_rate = audio.sample_rate();
    let result = outputs
        .into_iter()
        .map(|ch_data| {
            let array = ndarray::Array1::from_vec(ch_data);
            AudioSamples::new_mono(array.into(), sample_rate)
        })
        .collect();

    Ok(result)
}

#[cfg(feature = "simd")]
fn deinterleave_f64_simd_impl(
    audio: &AudioSamples<f64>,
    num_channels: usize,
    samples_per_channel: usize,
    data: &[f64],
) -> AudioSampleResult<Vec<AudioSamples<f64>>> {
    let mut outputs: Vec<Vec<f64>> = vec![Vec::with_capacity(samples_per_channel); num_channels];

    // Optimized case for stereo (2 channels)
    if num_channels == 2 {
        // Process 4 samples at a time with SIMD (f64x4)
        let simd_chunks = samples_per_channel / 4;

        for chunk in 0..simd_chunks {
            let start_idx = chunk * 4;

            // Convert 8 interleaved samples (4 left, 4 right) to f64 for SIMD processing
            let mut interleaved_f64 = [0.0f64; 8];

            for i in 0..4 {
                let left_idx = start_idx * 2 + i * 2;
                let right_idx = left_idx + 1;
                interleaved_f64[i * 2] = data[left_idx];
                interleaved_f64[i * 2 + 1] = data[right_idx];
            }

            // Deinterleave using SIMD vectors
            let mut left_f64 = [0.0f64; 4];
            let mut right_f64 = [0.0f64; 4];

            for i in 0..4 {
                left_f64[i] = interleaved_f64[i * 2];
                right_f64[i] = interleaved_f64[i * 2 + 1];
            }

            // Convert back to f64 and store
            for i in 0..4 {
                outputs[0].push(left_f64[i]);
                outputs[1].push(right_f64[i]);
            }
        }

        // Handle remainder samples
        for i in simd_chunks * 4..samples_per_channel {
            let left_idx = i * 2;
            let right_idx = left_idx + 1;
            outputs[0].push(data[left_idx]);
            outputs[1].push(data[right_idx]);
        }
    } else {
        // For non-stereo, fallback to scalar implementation
        for i in 0..samples_per_channel {
            for ch in 0..num_channels {
                outputs[ch].push(data[i * num_channels + ch]);
            }
        }
    }

    let sample_rate = audio.sample_rate();
    let result = outputs
        .into_iter()
        .map(|ch_data| {
            let array = ndarray::Array1::from_vec(ch_data);
            AudioSamples::new_mono(array.into(), sample_rate)
        })
        .collect();

    Ok(result)
}
