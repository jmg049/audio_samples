//! Module for the representation of audio samples.
//! In essence, an enhanced wrapper around `ndarray` to represent audio samples and their operations.
use ndarray::{Array1, Array2};

use crate::{AudioSample, ChannelLayout};

/// Internal representation of audio data
#[derive(Debug, Clone, PartialEq)]
pub enum AudioData<T: AudioSample> {
    Mono(Array1<T>),         // Single channel audio samples
    MultiChannel(Array2<T>), // Multi-channel audio samples where each row is a channel
}

/// Represents audio samples in a format that can be used for various audio processing tasks.
/// This struct contains both the audio data and metadata like sample rate, channel information, etc.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioSamples<T: AudioSample> {
    pub(crate) data: AudioData<T>,
    sample_rate: u32,
    layout: ChannelLayout,
}

impl<T: AudioSample> AudioSamples<T> {
    /// Creates a new AudioSamples with the given data and sample rate
    pub fn new(data: AudioData<T>, sample_rate: u32) -> Self {
        Self {
            data,
            sample_rate,
            layout: ChannelLayout::Interleaved, // Default layout, can be changed later
        }
    }

    /// Creates a new mono AudioSamples with the given data and sample rate
    pub fn new_mono(data: Array1<T>, sample_rate: u32) -> Self {
        Self {
            data: AudioData::Mono(data),
            sample_rate,
            layout: ChannelLayout::Interleaved,
        }
    }

    /// Creates a new multi-channel AudioSamples with the given data and sample rate
    /// The data should be arranged with each row representing a channel
    pub fn new_multi_channel(data: Array2<T>, sample_rate: u32) -> Self {
        Self {
            data: AudioData::MultiChannel(data),
            sample_rate,
            layout: ChannelLayout::Interleaved,
        }
    }

    /// Creates a new mono AudioSamples filled with zeros
    pub fn zeros_mono(length: usize, sample_rate: u32) -> Self {
        Self {
            data: AudioData::Mono(Array1::zeros(length)),
            sample_rate,
            layout: ChannelLayout::Interleaved,
        }
    }

    /// Creates a new multi-channel AudioSamples filled with zeros
    pub fn zeros_multi(channels: usize, length: usize, sample_rate: u32) -> Self {
        Self {
            data: AudioData::MultiChannel(Array2::zeros((channels, length))),
            sample_rate,
            layout: ChannelLayout::Interleaved,
        }
    }

    /// Returns the sample rate in Hz
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the number of channels
    pub fn channels(&self) -> usize {
        match &self.data {
            AudioData::Mono(_) => 1,
            AudioData::MultiChannel(arr) => arr.nrows(),
        }
    }

    /// Returns the number of samples per channel
    pub fn samples_per_channel(&self) -> usize {
        match &self.data {
            AudioData::Mono(arr) => arr.len(),
            AudioData::MultiChannel(arr) => arr.ncols(),
        }
    }

    /// Returns the duration in seconds
    pub fn duration_seconds(&self) -> f64 {
        self.samples_per_channel() as f64 / self.sample_rate as f64
    }

    /// Returns the total number of samples across all channels
    pub fn total_samples(&self) -> usize {
        self.channels() * self.samples_per_channel()
    }

    /// Returns the number of bytes per sample for type T
    pub fn bytes_per_sample(&self) -> usize {
        std::mem::size_of::<T>()
    }

    pub fn sample_type() -> &'static str {
        std::any::type_name::<T>()
    }

    /// Returns the channel layout
    pub fn layout(&self) -> ChannelLayout {
        self.layout
    }

    /// Returns true if this is mono audio
    pub fn is_mono(&self) -> bool {
        matches!(self.data, AudioData::Mono(_))
    }

    /// Returns true if this is multi-channel audio
    pub fn is_multi_channel(&self) -> bool {
        matches!(self.data, AudioData::MultiChannel(_))
    }

    /// Returns the peak (maximum absolute value) of the audio samples in the native type
    pub fn peak_native(&self) -> T
    where
        T: PartialOrd + Copy + std::ops::Sub<Output = T>,
    {
        match &self.data {
            AudioData::Mono(arr) => {
                arr.iter()
                    .map(|&sample| {
                        // Simple absolute value: if sample < 0, return -sample, else sample
                        if sample < T::default() {
                            T::default() - sample
                        } else {
                            sample
                        }
                    })
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(T::default())
            }
            AudioData::MultiChannel(arr) => arr
                .iter()
                .map(|&sample| {
                    if sample < T::default() {
                        T::default() - sample
                    } else {
                        sample
                    }
                })
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(T::default()),
        }
    }

    /// Returns the minimum value in the audio samples
    pub fn min_native(&self) -> T
    where
        T: PartialOrd + Copy,
    {
        match &self.data {
            AudioData::Mono(arr) => arr
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .copied()
                .unwrap_or(T::default()),
            AudioData::MultiChannel(arr) => arr
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .copied()
                .unwrap_or(T::default()),
        }
    }

    /// Returns the maximum value in the audio samples
    pub fn max_native(&self) -> T
    where
        T: PartialOrd + Copy,
    {
        match &self.data {
            AudioData::Mono(arr) => arr
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .copied()
                .unwrap_or(T::default()),
            AudioData::MultiChannel(arr) => arr
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .copied()
                .unwrap_or(T::default()),
        }
    }

    /// Returns a reference to the underlying mono array, if this is mono audio
    pub fn as_mono(&self) -> Option<&Array1<T>> {
        match &self.data {
            AudioData::Mono(arr) => Some(arr),
            AudioData::MultiChannel(_) => None,
        }
    }

    /// Returns a reference to the underlying multi-channel array, if this is multi-channel audio
    pub fn as_multi_channel(&self) -> Option<&Array2<T>> {
        match &self.data {
            AudioData::Mono(_) => None,
            AudioData::MultiChannel(arr) => Some(arr),
        }
    }

    /// Returns a mutable reference to the underlying mono array, if this is mono audio
    pub fn as_mono_mut(&mut self) -> Option<&mut Array1<T>> {
        match &mut self.data {
            AudioData::Mono(arr) => Some(arr),
            AudioData::MultiChannel(_) => None,
        }
    }

    /// Returns a mutable reference to the underlying multi-channel array, if this is multi-channel audio
    pub fn as_multi_channel_mut(&mut self) -> Option<&mut Array2<T>> {
        match &mut self.data {
            AudioData::Mono(_) => None,
            AudioData::MultiChannel(arr) => Some(arr),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_new_mono_audio_samples() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioSamples::new_mono(data.clone(), 44100);

        assert_eq!(audio.sample_rate(), 44100);
        assert_eq!(audio.channels(), 1);
        assert_eq!(audio.samples_per_channel(), 5);
        assert!(audio.is_mono());
        assert!(!audio.is_multi_channel());
        assert_eq!(audio.as_mono().unwrap(), &data);
    }

    #[test]
    fn test_new_multi_channel_audio_samples() {
        let data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 2 channels, 3 samples each
        let audio = AudioSamples::new_multi_channel(data.clone(), 48000);

        assert_eq!(audio.sample_rate(), 48000);
        assert_eq!(audio.channels(), 2);
        assert_eq!(audio.samples_per_channel(), 3);
        assert_eq!(audio.total_samples(), 6);
        assert!(!audio.is_mono());
        assert!(audio.is_multi_channel());
        assert_eq!(audio.as_multi_channel().unwrap(), &data);
    }

    #[test]
    fn test_zeros_construction() {
        let mono_audio = AudioSamples::<f32>::zeros_mono(100, 44100);
        assert_eq!(mono_audio.channels(), 1);
        assert_eq!(mono_audio.samples_per_channel(), 100);
        assert_eq!(mono_audio.sample_rate(), 44100);

        let multi_audio = AudioSamples::<f32>::zeros_multi(2, 50, 48000);
        assert_eq!(multi_audio.channels(), 2);
        assert_eq!(multi_audio.samples_per_channel(), 50);
        assert_eq!(multi_audio.total_samples(), 100);
        assert_eq!(multi_audio.sample_rate(), 48000);
    }

    #[test]
    fn test_duration_seconds() {
        let audio = AudioSamples::<f32>::zeros_mono(44100, 44100);
        assert!((audio.duration_seconds() - 1.0).abs() < 1e-6);

        let audio2 = AudioSamples::<f32>::zeros_multi(2, 22050, 44100);
        assert!((audio2.duration_seconds() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_native_statistics() {
        let data = array![-3.0f32, -1.0, 0.0, 2.0, 4.0];
        let audio = AudioSamples::new_mono(data, 44100);

        assert_eq!(audio.min_native(), -3.0);
        assert_eq!(audio.max_native(), 4.0);
        assert_eq!(audio.peak_native(), 4.0); // abs(-3) = 3, abs(4) = 4, so peak is 4
    }

    #[test]
    fn test_multi_channel_statistics() {
        let data = array![[-2.0f32, 1.0], [3.0, -4.0]]; // 2 channels, 2 samples each
        let audio = AudioSamples::new_multi_channel(data, 44100);

        assert_eq!(audio.min_native(), -4.0);
        assert_eq!(audio.max_native(), 3.0);
        assert_eq!(audio.peak_native(), 4.0); // abs(-4) = 4 is the largest absolute value
    }
}
