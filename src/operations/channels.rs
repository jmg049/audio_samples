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
    AudioChannelOps, AudioEditing, AudioSample, AudioSampleError, AudioSampleResult, AudioSamples,
    AudioTypeConversion, ConvertTo, I24, operations::MonoConversionMethod, repr::AudioData,
};
use ndarray::{Array1, Axis};

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
        match method {
            MonoConversionMethod::Average => match &self.data {
                AudioData::Mono(_) => Ok(AudioSamples::new(self.data.clone(), self.sample_rate())),
                AudioData::MultiChannel(data) => {
                    let mono_data = data.mean_axis(Axis(0)).ok_or_else(|| {
                        AudioSampleError::ProcessingError {
                            msg: "Failed to compute average for multi-channel audio".to_string(),
                        }
                    })?;
                    Ok(AudioSamples::new_mono(mono_data, self.sample_rate()))
                }
            },
            MonoConversionMethod::Left => match &self.data {
                AudioData::Mono(_) => Ok(AudioSamples::new(self.data.clone(), self.sample_rate())),
                AudioData::MultiChannel(data) => {
                    let left_channel = data.index_axis(Axis(0), 0).to_owned();
                    Ok(AudioSamples::new_mono(left_channel, self.sample_rate()))
                }
            },
            MonoConversionMethod::Right => match &self.data {
                AudioData::Mono(_) => Ok(AudioSamples::new(self.data.clone(), self.sample_rate())),
                AudioData::MultiChannel(data) => {
                    let right_channel = data.index_axis(Axis(0), 1).to_owned();
                    Ok(AudioSamples::new_mono(right_channel, self.sample_rate()))
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

    fn to_stereo(&self, method: super::StereoConversionMethod) -> crate::AudioSampleResult<Self>
    where
        Self: Sized,
    {
        match method {
            super::StereoConversionMethod::Duplicate => match &self.data {
                AudioData::Mono(mono_data) => {
                    let stereo_data =
                        ndarray::stack(Axis(0), &[mono_data.view(), mono_data.view()]).map_err(
                            |e| AudioSampleError::ProcessingError {
                                msg: format!("Failed to duplicate mono to stereo: {}", e),
                            },
                        )?;
                    Ok(AudioSamples::new(
                        AudioData::MultiChannel(stereo_data),
                        self.sample_rate(),
                    ))
                }
                AudioData::MultiChannel(_) => {
                    Ok(AudioSamples::new(self.data.clone(), self.sample_rate()))
                }
            },
            super::StereoConversionMethod::Pan(_) => {
                todo!()
            }
            super::StereoConversionMethod::Left => self.extract_channel(0),
            super::StereoConversionMethod::Right => self.extract_channel(1),
        }
    }

    /// TODO!
    fn to_channels(
        &self,
        _target_channels: usize,
        method: super::ChannelConversionMethod,
    ) -> crate::AudioSampleResult<Self>
    where
        Self: Sized,
    {
        match method {
            super::ChannelConversionMethod::Repeat => todo!(),
            super::ChannelConversionMethod::Smart => todo!(),
            super::ChannelConversionMethod::Custom(_items) => todo!(),
        }
    }

    fn extract_channel(&self, channel_index: usize) -> crate::AudioSampleResult<Self>
    where
        Self: Sized,
    {
        if channel_index >= self.num_channels() {
            return Err(crate::AudioSampleError::InvalidParameter(format!(
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
                    crate::repr::AudioData::Mono(channel),
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
}
