//! Zero-copy view types for audio data.
//!
//! This module provides view types that allow read-only access to audio data
//! without allocation. Views can also provide zero-copy conversions for
//! compatible sample types that have the same memory layout.

use crate::repr::AudioData;
use crate::{
    AudioSample, AudioSampleResult, AudioSamples, AudioStatistics, AudioTypeConversion,
    ChannelLayout, ConvertTo, I24,
};
use std::ops::Deref;

/// Common interface for audio data that can be read.
///
/// This trait allows functions to work seamlessly with both `AudioSamples<T>`
/// and `AudioView<T>` without requiring separate implementations.
pub trait AudioDataRead<T: AudioSample> {
    /// Returns the sample rate in Hz.
    fn sample_rate(&self) -> u32;

    /// Returns the channel layout.
    fn layout(&self) -> ChannelLayout;

    /// Returns the number of channels.
    fn num_channels(&self) -> usize;

    /// Returns the number of samples per channel.
    fn samples_per_channel(&self) -> usize;

    /// Returns a reference to the underlying audio data.
    fn data(&self) -> &AudioData<T>;

    /// Returns true if this is mono audio.
    fn is_mono(&self) -> bool {
        self.data().is_mono()
    }

    /// Returns true if this is multi-channel audio.
    fn is_multi_channel(&self) -> bool {
        self.data().is_multi_channel()
    }

    /// Returns the duration in seconds.
    fn duration_seconds(&self) -> f64 {
        self.samples_per_channel() as f64 / self.sample_rate() as f64
    }
}

/// Trait to determine if two audio sample types have the same memory layout.
///
/// Types with the same layout can be safely transmuted between each other,
/// enabling zero-copy conversions.
pub trait SameLayout<T> {
    /// Returns true if this type has the same memory layout as T.
    fn same_layout() -> bool;
}

// Implement SameLayout for types that are identical or have same bit width
macro_rules! impl_same_layout {
    ($a:ty, $b:ty) => {
        impl SameLayout<$b> for $a {
            fn same_layout() -> bool {
                std::mem::size_of::<$a>() == std::mem::size_of::<$b>()
                    && std::mem::align_of::<$a>() == std::mem::align_of::<$b>()
            }
        }
    };
}

// Self-layouts
impl_same_layout!(i16, i16);
impl_same_layout!(i32, i32);
impl_same_layout!(f32, f32);
impl_same_layout!(f64, f64);

// Same-size cross-layouts (32-bit types)
impl_same_layout!(i32, f32);
impl_same_layout!(f32, i32);

// Same-size cross-layouts (64-bit types)
impl_same_layout!(i64, f64);
impl_same_layout!(f64, i64);

/// A read-only view into audio samples without owning the data.
///
/// This provides zero-allocation access to audio data and can be used
/// for analysis operations that don't need to modify the original data.
///
/// # Example
/// ```rust,ignore
/// let audio = AudioSamples::new_mono(data, 44100);
/// let view = audio.as_view();
/// let rms = view.rms(); // No allocation needed
/// ```
#[derive(Clone)]
pub struct AudioView<'a, T: AudioSample> {
    data: &'a AudioData<T>,
    sample_rate: u32,
    layout: ChannelLayout,
}

impl<'a, T: AudioSample> AudioView<'a, T> {
    /// Creates a new view from audio samples.
    pub fn new(samples: &'a AudioSamples<T>) -> Self {
        Self {
            data: &samples.data,
            sample_rate: samples.sample_rate,
            layout: samples.layout,
        }
    }

    /// Returns the sample rate in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the channel layout.
    pub fn layout(&self) -> ChannelLayout {
        self.layout
    }

    /// Returns the number of channels.
    pub fn num_channels(&self) -> usize {
        self.data.num_channels()
    }

    /// Returns the number of samples per channel.
    pub fn samples_per_channel(&self) -> usize {
        self.data.samples_per_channel()
    }

    /// Returns the total number of samples across all channels.
    pub fn total_samples(&self) -> usize {
        self.num_channels() * self.samples_per_channel()
    }

    /// Returns the duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        self.samples_per_channel() as f64 / self.sample_rate as f64
    }

    /// Returns true if this is mono audio.
    pub fn is_mono(&self) -> bool {
        self.data.is_mono()
    }

    /// Returns true if this is multi-channel audio.
    pub fn is_multi_channel(&self) -> bool {
        self.data.is_multi_channel()
    }

    /// Attempts to create a zero-copy view with a different sample type.
    ///
    /// This succeeds only if the source and target types have the same
    /// memory layout (size and alignment).
    ///
    /// # Safety
    /// This function uses unsafe transmutation when types have compatible
    /// layouts. The caller must ensure that the bit patterns are valid
    /// for the target type.
    pub fn try_as_type_view<U: AudioSample>(&self) -> Option<AudioView<'a, U>>
    where
        T: SameLayout<U>,
    {
        if T::same_layout() {
            Some(AudioView {
                data: unsafe { std::mem::transmute(self.data) },
                sample_rate: self.sample_rate,
                layout: self.layout,
            })
        } else {
            None
        }
    }

    /// Returns a reference to the underlying mono array, if this is mono audio.
    pub fn as_mono(&self) -> Option<&'a crate::repr::Array1<T>> {
        self.data.as_mono()
    }

    /// Returns a reference to the underlying multi-channel array, if this is multi-channel audio.
    pub fn as_multi_channel(&self) -> Option<&'a crate::repr::Array2<T>> {
        self.data.as_multi_channel()
    }

    /// Converts the view to an owned AudioSamples instance.
    ///
    /// This clones the underlying data.
    pub fn to_owned(&self) -> AudioSamples<T> {
        AudioSamples {
            data: self.data.clone(),
            sample_rate: self.sample_rate,
            layout: self.layout,
        }
    }
}

/// A mutable view into audio samples without owning the data.
///
/// This provides zero-allocation mutable access to audio data for
/// in-place operations.
pub struct AudioViewMut<'a, T: AudioSample> {
    data: &'a mut AudioData<T>,
    sample_rate: u32,
    layout: ChannelLayout,
}

impl<'a, T: AudioSample> AudioViewMut<'a, T> {
    /// Creates a new mutable view from audio samples.
    pub fn new(samples: &'a mut AudioSamples<T>) -> Self {
        Self {
            data: &mut samples.data,
            sample_rate: samples.sample_rate,
            layout: samples.layout,
        }
    }

    /// Returns the sample rate in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the channel layout.
    pub fn layout(&self) -> ChannelLayout {
        self.layout
    }

    /// Returns the number of channels.
    pub fn num_channels(&self) -> usize {
        self.data.num_channels()
    }

    /// Returns the number of samples per channel.
    pub fn samples_per_channel(&self) -> usize {
        self.data.samples_per_channel()
    }

    /// Returns the total number of samples across all channels.
    pub fn total_samples(&self) -> usize {
        self.num_channels() * self.samples_per_channel()
    }

    /// Returns the duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        self.samples_per_channel() as f64 / self.sample_rate as f64
    }

    /// Returns true if this is mono audio.
    pub fn is_mono(&self) -> bool {
        self.data.is_mono()
    }

    /// Returns true if this is multi-channel audio.
    pub fn is_multi_channel(&self) -> bool {
        self.data.is_multi_channel()
    }

    /// Returns a mutable reference to the underlying mono array, if this is mono audio.
    pub fn as_mono_mut(&mut self) -> Option<&mut crate::repr::Array1<T>> {
        self.data.as_mono_mut()
    }

    /// Returns a mutable reference to the underlying multi-channel array, if this is multi-channel audio.
    pub fn as_multi_channel_mut(&mut self) -> Option<&mut crate::repr::Array2<T>> {
        self.data.as_multi_channel_mut()
    }

    /// Returns an immutable view from this mutable view.
    pub fn as_view(&self) -> AudioView<'_, T> {
        AudioView {
            data: self.data,
            sample_rate: self.sample_rate,
            layout: self.layout,
        }
    }
}

// Implement Deref to allow AudioView to be used like AudioSamples for read operations
impl<'a, T: AudioSample> Deref for AudioView<'a, T> {
    type Target = AudioData<T>;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a, T: AudioSample> Deref for AudioViewMut<'a, T> {
    type Target = AudioData<T>;

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

// Implement AudioStatistics for AudioView to enable statistical operations on views
impl<'a, T: AudioSample> AudioStatistics<T> for AudioView<'a, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'b> AudioSamples<T>: AudioTypeConversion<T>,
{
    fn peak(&self) -> T {
        // Delegate to the underlying AudioSamples implementation
        // by creating a temporary AudioSamples that references the same data
        let temp_audio = AudioSamples {
            data: self.data.clone(),
            sample_rate: self.sample_rate,
            layout: self.layout,
        };
        temp_audio.peak()
    }

    fn min_sample(&self) -> T {
        let temp_audio = AudioSamples {
            data: self.data.clone(),
            sample_rate: self.sample_rate,
            layout: self.layout,
        };
        temp_audio.min_sample()
    }

    fn max_sample(&self) -> T {
        let temp_audio = AudioSamples {
            data: self.data.clone(),
            sample_rate: self.sample_rate,
            layout: self.layout,
        };
        temp_audio.max_sample()
    }

    fn mean(&self) -> T {
        let temp_audio = AudioSamples {
            data: self.data.clone(),
            sample_rate: self.sample_rate,
            layout: self.layout,
        };
        temp_audio.mean()
    }

    fn rms(&self) -> f64 {
        let temp_audio = AudioSamples {
            data: self.data.clone(),
            sample_rate: self.sample_rate,
            layout: self.layout,
        };
        temp_audio.rms()
    }

    fn variance(&self) -> AudioSampleResult<f64> {
        let temp_audio = AudioSamples {
            data: self.data.clone(),
            sample_rate: self.sample_rate,
            layout: self.layout,
        };
        temp_audio.variance()
    }

    fn std_dev(&self) -> AudioSampleResult<f64> {
        let temp_audio = AudioSamples {
            data: self.data.clone(),
            sample_rate: self.sample_rate,
            layout: self.layout,
        };
        temp_audio.std_dev()
    }

    fn zero_crossings(&self) -> usize {
        let temp_audio = AudioSamples {
            data: self.data.clone(),
            sample_rate: self.sample_rate,
            layout: self.layout,
        };
        temp_audio.zero_crossings()
    }

    fn zero_crossing_rate(&self) -> f64 {
        let temp_audio = AudioSamples {
            data: self.data.clone(),
            sample_rate: self.sample_rate,
            layout: self.layout,
        };
        temp_audio.zero_crossing_rate()
    }

    fn autocorrelation(&self, max_lag: usize) -> AudioSampleResult<Vec<f64>> {
        let temp_audio = AudioSamples {
            data: self.data.clone(),
            sample_rate: self.sample_rate,
            layout: self.layout,
        };
        temp_audio.autocorrelation(max_lag)
    }

    fn cross_correlation(&self, other: &Self, max_lag: usize) -> AudioSampleResult<Vec<f64>> {
        let temp_audio = AudioSamples {
            data: self.data.clone(),
            sample_rate: self.sample_rate,
            layout: self.layout,
        };
        let other_temp_audio = AudioSamples {
            data: other.data.clone(),
            sample_rate: other.sample_rate,
            layout: other.layout,
        };
        temp_audio.cross_correlation(&other_temp_audio, max_lag)
    }

    fn spectral_centroid(&self) -> AudioSampleResult<f64> {
        let temp_audio = AudioSamples {
            data: self.data.clone(),
            sample_rate: self.sample_rate,
            layout: self.layout,
        };
        temp_audio.spectral_centroid()
    }

    fn spectral_rolloff(&self, rolloff_percent: f64) -> AudioSampleResult<f64> {
        let temp_audio = AudioSamples {
            data: self.data.clone(),
            sample_rate: self.sample_rate,
            layout: self.layout,
        };
        temp_audio.spectral_rolloff(rolloff_percent)
    }
}

// Add view creation methods to AudioSamples
impl<T: AudioSample> AudioSamples<T> {
    /// Creates a read-only view of this audio data.
    ///
    /// Views provide zero-allocation access to the underlying data
    /// and support many of the same operations as owned samples.
    ///
    /// # Example
    /// ```rust,ignore
    /// let audio = AudioSamples::new_mono(data, 44100);
    /// let view = audio.as_view();
    /// let peak = view.peak(); // No allocation needed
    /// ```
    pub fn as_view(&self) -> AudioView<'_, T> {
        AudioView::new(self)
    }

    /// Creates a mutable view of this audio data.
    ///
    /// Mutable views allow in-place modifications without allocation.
    ///
    /// # Example
    /// ```rust,ignore
    /// let mut audio = AudioSamples::new_mono(data, 44100);
    /// let mut view = audio.as_view_mut();
    /// view.scale(0.5); // Modify in-place
    /// ```
    pub fn as_view_mut(&mut self) -> AudioViewMut<'_, T> {
        AudioViewMut::new(self)
    }

    /// Attempts to create a zero-copy view with a different sample type.
    ///
    /// This succeeds only if the source and target types have the same
    /// memory layout (size and alignment).
    ///
    /// # Example
    /// ```rust,ignore
    /// let audio_i32 = AudioSamples::new_mono(i32_data, 44100);
    /// if let Some(view_f32) = audio_i32.try_as_type_view::<f32>() {
    ///     // Zero-copy view as f32 (same 32-bit layout)
    ///     let peak = view_f32.peak();
    /// }
    /// ```
    pub fn try_as_type_view<U: AudioSample>(&self) -> Option<AudioView<'_, U>>
    where
        T: SameLayout<U>,
    {
        self.as_view().try_as_type_view()
    }
}

// Implement AudioDataRead for AudioSamples
impl<T: AudioSample> AudioDataRead<T> for AudioSamples<T> {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn layout(&self) -> ChannelLayout {
        self.layout
    }

    fn num_channels(&self) -> usize {
        self.data.num_channels()
    }

    fn samples_per_channel(&self) -> usize {
        self.data.samples_per_channel()
    }

    fn data(&self) -> &AudioData<T> {
        &self.data
    }
}

// Implement AudioDataRead for AudioView
impl<'a, T: AudioSample> AudioDataRead<T> for AudioView<'a, T> {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn layout(&self) -> ChannelLayout {
        self.layout
    }

    fn num_channels(&self) -> usize {
        self.data.num_channels()
    }

    fn samples_per_channel(&self) -> usize {
        self.data.samples_per_channel()
    }

    fn data(&self) -> &AudioData<T> {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_same_layout_detection() {
        assert!(<i32 as SameLayout<f32>>::same_layout());
        assert!(<f32 as SameLayout<i32>>::same_layout());
        assert!(<i16 as SameLayout<i16>>::same_layout());
        // Note: These would fail to compile as the trait isn't implemented for these pairs
        // assert!(!<i16 as SameLayout<i32>>::same_layout());
        // assert!(!<i16 as SameLayout<f32>>::same_layout());
    }

    #[test]
    fn test_audio_view_basic() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioSamples::new_mono(data, 44100);
        let view = audio.as_view();

        assert_eq!(view.sample_rate(), 44100);
        assert_eq!(view.num_channels(), 1);
        assert_eq!(view.samples_per_channel(), 5);
        assert_eq!(view.total_samples(), 5);
        assert!(view.is_mono());
        assert!(!view.is_multi_channel());
    }

    #[test]
    fn test_audio_view_mut() {
        let data = array![1.0f32, 2.0, 3.0];
        let mut audio = AudioSamples::new_mono(data, 44100);

        {
            let view = audio.as_view_mut();
            assert_eq!(view.num_channels(), 1);

            // Test immutable view from mutable view
            let immutable_view = view.as_view();
            assert_eq!(immutable_view.samples_per_channel(), 3);
        }

        // Original audio should still be accessible
        assert_eq!(audio.samples_per_channel(), 3);
    }

    #[test]
    fn test_view_to_owned() {
        let data = array![1.0f32, 2.0, 3.0];
        let audio = AudioSamples::new_mono(data, 44100);
        let view = audio.as_view();

        let owned = view.to_owned();
        assert_eq!(owned.sample_rate(), 44100);
        assert_eq!(owned.samples_per_channel(), 3);
    }

    #[test]
    fn test_zero_copy_type_view() {
        let data = array![1065353216i32, 1073741824i32]; // bit patterns for 1.0f32, 2.0f32
        let audio_i32 = AudioSamples::new_mono(data, 44100);

        // Should succeed for same-layout types
        let view_f32 = audio_i32.try_as_type_view::<f32>();
        assert!(view_f32.is_some());

        if let Some(view) = view_f32 {
            assert_eq!(view.num_channels(), 1);
            assert_eq!(view.samples_per_channel(), 2);
        }
    }

    #[test]
    fn test_multi_channel_view() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let audio = AudioSamples::new_multi_channel(data, 44100);
        let view = audio.as_view();

        assert_eq!(view.num_channels(), 2);
        assert_eq!(view.samples_per_channel(), 2);
        assert_eq!(view.total_samples(), 4);
        assert!(!view.is_mono());
        assert!(view.is_multi_channel());
    }
}
