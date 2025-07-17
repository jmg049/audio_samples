use ndarray::{Array1, Axis};
use num_traits::{FromPrimitive, One, ToPrimitive, Zero};

use crate::{
    AudioChannelOps, AudioEditing, AudioSample, AudioSampleError, AudioSampleResult, AudioSamples,
    AudioTypeConversion, ConvertTo, I24, repr::AudioData,
};

impl<T: AudioSample> AudioChannelOps<T> for AudioSamples<T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<f64>: AudioTypeConversion<T>,
    T: FromPrimitive + ToPrimitive + Zero + One
{
    fn to_mono(&self, method: super::MonoConversionMethod) -> crate::AudioSampleResult<Self>
    where
        Self: Sized,
    {
        match method {
            super::MonoConversionMethod::Average => match &self.data {
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
            super::MonoConversionMethod::Left => match &self.data {
                AudioData::Mono(_) => Ok(AudioSamples::new(self.data.clone(), self.sample_rate())),
                AudioData::MultiChannel(data) => {
                    let left_channel = data.index_axis(Axis(0), 0).to_owned();
                    Ok(AudioSamples::new_mono(left_channel, self.sample_rate()))
                }
            },
            super::MonoConversionMethod::Right => match &self.data {
                AudioData::Mono(_) => Ok(AudioSamples::new(self.data.clone(), self.sample_rate())),
                AudioData::MultiChannel(data) => {
                    let right_channel = data.index_axis(Axis(0), 1).to_owned();
                    Ok(AudioSamples::new_mono(right_channel, self.sample_rate()))
                }
            },
            super::MonoConversionMethod::Weighted(items) => {
                todo!()
            }
            super::MonoConversionMethod::Center => {
                todo!()
            }
        }
    }

    fn to_stereo(&self, method: super::StereoConversionMethod) -> crate::AudioSampleResult<Self>
    where
        Self: Sized,
    {
        match method {
            super::StereoConversionMethod::Duplicate => self.repeat(2),
            super::StereoConversionMethod::Pan(_) => {
                todo!()
            }
            super::StereoConversionMethod::Left => self.extract_channel(0),
            super::StereoConversionMethod::Right => self.extract_channel(1),
        }
    }

    fn to_channels(
        &self,
        target_channels: usize,
        method: super::ChannelConversionMethod,
    ) -> crate::AudioSampleResult<Self>
    where
        Self: Sized,
    {
        match method {
            super::ChannelConversionMethod::Repeat => todo!(),
            super::ChannelConversionMethod::Smart => todo!(),
            super::ChannelConversionMethod::Custom(items) => todo!(),
        }
    }

    fn extract_channel(&self, channel_index: usize) -> crate::AudioSampleResult<Self>
    where
        Self: Sized,
    {
        if channel_index >= self.channels() {
            return Err(crate::AudioSampleError::InvalidParameter(format!(
                "Channel index {} out of bounds for {} channels",
                channel_index,
                self.channels()
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
        if channel1 >= self.channels() || channel2 >= self.channels() {
            return Err(AudioSampleError::InvalidParameter(format!(
                "Channel indices {} and {} out of bounds for {} channels",
                channel1,
                channel2,
                self.channels()
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

                let mut data_f64_samples = self.to_f64()?;
                let data_f64: &mut ndarray::ArrayBase<
                    ndarray::OwnedRepr<f64>,
                    ndarray::Dim<[usize; 2]>,
                > = match &mut data_f64_samples.data {
                    AudioData::Mono(_) => {
                        return Err(AudioSampleError::InvalidParameter(
                            "Cannot pan mono audio".to_string(),
                        ));
                    }
                    AudioData::MultiChannel(data_f64) => data_f64,
                };

                // left
                {
                    let left = &mut data_f64.index_axis_mut(Axis(0), 0);
                    let left_gain = 1.0 - pan_value.clamp(-1.0, 1.0);
                    left.mapv_inplace(|x| x * left_gain);
                }

                // right
                {
                    let right = &mut data_f64.index_axis_mut(Axis(0), 1);
                    let right_gain = 1.0 + pan_value.clamp(-1.0, 1.0);
                    right.mapv_inplace(|x| x * right_gain);
                }

                let converted = data_f64_samples.to_type::<T>()?;
                *self = converted;
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

                let mut data_f64_samples = self.to_f64()?;
                let data_f64: &mut ndarray::ArrayBase<
                    ndarray::OwnedRepr<f64>,
                    ndarray::Dim<[usize; 2]>,
                > = match &mut data_f64_samples.data {
                    AudioData::Mono(_) => {
                        return Err(AudioSampleError::InvalidParameter(
                            "Cannot balance mono audio".to_string(),
                        ));
                    }
                    AudioData::MultiChannel(data_f64) => data_f64,
                };

                // left
                {
                    let left = &mut data_f64.index_axis_mut(Axis(0), 0);
                    let left_gain = 1.0 - balance.clamp(-1.0, 1.0);
                    left.mapv_inplace(|x| x * left_gain);
                }

                // right
                {
                    let right = &mut data_f64.index_axis_mut(Axis(0), 1);
                    let right_gain = 1.0 + balance.clamp(-1.0, 1.0);
                    right.mapv_inplace(|x| x * right_gain);
                }

                let converted = data_f64_samples.to_type::<T>()?;
                *self = converted;
                Ok(())
            }
        }
    }
}
