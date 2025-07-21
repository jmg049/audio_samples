//! Module for the representation of audio samples.
//! In essence, an enhanced wrapper around `ndarray` to represent audio samples and their operations.
use ndarray::{Array1, Array2};

use crate::{AudioSample, AudioSampleError, ChannelLayout};

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
    pub const fn new(data: AudioData<T>, sample_rate: u32) -> Self {
        Self {
            data,
            sample_rate,
            layout: ChannelLayout::Interleaved, // Default layout, can be changed later
        }
    }

    /// Creates a new mono AudioSamples with the given data and sample rate
    pub const fn new_mono(data: Array1<T>, sample_rate: u32) -> Self {
        Self {
            data: AudioData::Mono(data),
            sample_rate,
            layout: ChannelLayout::Interleaved,
        }
    }

    /// Creates a new multi-channel AudioSamples with the given data and sample rate
    /// The data should be arranged with each row representing a channel
    pub const fn new_multi_channel(data: Array2<T>, sample_rate: u32) -> Self {
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
    pub const fn sample_rate(&self) -> u32 {
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
    pub const fn bytes_per_sample(&self) -> usize {
        std::mem::size_of::<T>()
    }

    pub fn sample_type() -> &'static str {
        std::any::type_name::<T>()
    }

    /// Returns the channel layout
    pub const fn layout(&self) -> ChannelLayout {
        self.layout
    }

    /// Returns true if this is mono audio
    pub const fn is_mono(&self) -> bool {
        matches!(self.data, AudioData::Mono(_))
    }

    /// Returns true if this is multi-channel audio
    pub const fn is_multi_channel(&self) -> bool {
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
    pub const fn as_mono(&self) -> Option<&Array1<T>> {
        match &self.data {
            AudioData::Mono(arr) => Some(arr),
            AudioData::MultiChannel(_) => None,
        }
    }

    /// Returns a reference to the underlying multi-channel array, if this is multi-channel audio
    pub const fn as_multi_channel(&self) -> Option<&Array2<T>> {
        match &self.data {
            AudioData::Mono(_) => None,
            AudioData::MultiChannel(arr) => Some(arr),
        }
    }

    /// Returns a mutable reference to the underlying mono array, if this is mono audio
    pub const fn as_mono_mut(&mut self) -> Option<&mut Array1<T>> {
        match &mut self.data {
            AudioData::Mono(arr) => Some(arr),
            AudioData::MultiChannel(_) => None,
        }
    }

    /// Returns a mutable reference to the underlying multi-channel array, if this is multi-channel audio
    pub const fn as_multi_channel_mut(&mut self) -> Option<&mut Array2<T>> {
        match &mut self.data {
            AudioData::Mono(_) => None,
            AudioData::MultiChannel(arr) => Some(arr),
        }
    }

    /// Applies a function to each sample in the audio data in-place.
    ///
    /// This method applies the given function to every sample in the audio data,
    /// modifying the samples in-place. The function receives each sample and
    /// should return the transformed sample.
    ///
    /// # Arguments
    /// * `f` - A function that takes a sample and returns a transformed sample
    ///
    /// # Returns
    /// A result indicating success or failure
    ///
    /// # Example
    /// ```rust,ignore
    /// // Halve the amplitude of all samples
    /// audio.apply(|sample| sample * 0.5)?;
    ///
    /// // Apply a simple distortion
    /// audio.apply(|sample| sample.clamp(-0.8, 0.8))?;
    /// ```
    pub fn apply<F>(&mut self, f: F) -> crate::AudioSampleResult<()>
    where
        F: Fn(T) -> T,
    {
        match &mut self.data {
            AudioData::Mono(arr) => {
                for sample in arr.iter_mut() {
                    *sample = f(*sample);
                }
            }
            AudioData::MultiChannel(arr) => {
                for sample in arr.iter_mut() {
                    *sample = f(*sample);
                }
            }
        }
        Ok(())
    }

    /// Applies a function to overlapping windows of samples in-place.
    ///
    /// This method processes the audio data in overlapping windows, applying
    /// the given function to each window. The function receives a slice of
    /// samples and should return a vector of transformed samples.
    ///
    /// # Arguments
    /// * `window_size` - Size of each window in samples
    /// * `hop_size` - Number of samples to advance between windows
    /// * `f` - A function that takes a window slice and returns transformed samples
    ///
    /// # Returns
    /// A result containing the processed audio or an error
    ///
    /// # Example
    /// ```rust,ignore
    /// // Apply a Hann window to overlapping frames
    /// audio.apply_windowed(1024, 512, |window| {
    ///     window.iter().enumerate()
    ///         .map(|(i, &sample)| sample * hann_window[i])
    ///         .collect()
    /// })?;
    /// ```
    pub fn apply_windowed<F>(
        &mut self,
        window_size: usize,
        hop_size: usize,
        f: F,
    ) -> crate::AudioSampleResult<()>
    where
        F: Fn(&[T]) -> Vec<T>,
    {
        if window_size == 0 || hop_size == 0 {
            return Err(crate::AudioSampleError::InvalidParameter(
                "Window size and hop size must be greater than 0".to_string(),
            ));
        }

        match &mut self.data {
            AudioData::Mono(arr) => {
                let mut result = Vec::new();
                let data = arr.as_slice().ok_or(AudioSampleError::ArrayLayoutError {
                    message: "Mono samples must be contiguous".to_string(),
                })?;

                // Process overlapping windows
                let mut pos = 0;
                while pos + window_size <= data.len() {
                    let window = &data[pos..pos + window_size];
                    let processed = f(window);

                    if processed.len() != window_size {
                        return Err(crate::AudioSampleError::InvalidParameter(format!(
                            "Window function must return {} samples, got {}",
                            window_size,
                            processed.len()
                        )));
                    }

                    if result.len() < pos + window_size {
                        result.resize(pos + window_size, T::default());
                    }

                    // Overlap-add the processed window
                    for (i, &sample) in processed.iter().enumerate() {
                        result[pos + i] = sample;
                    }

                    pos += hop_size;
                }

                // Handle remaining samples
                if pos < data.len() {
                    result.extend_from_slice(&data[pos..]);
                }

                *arr = Array1::from_vec(result);
            }
            AudioData::MultiChannel(arr) => {
                let (channels, _) = arr.dim();
                let mut result = Vec::new();

                // Process each channel separately
                for ch in 0..channels {
                    let channel_data = arr.row(ch).to_owned();
                    let data =
                        channel_data
                            .as_slice()
                            .ok_or(AudioSampleError::ArrayLayoutError {
                                message: "Multi-channel samples must be contiguous".to_string(),
                            })?;

                    let mut channel_result = Vec::new();
                    let mut pos = 0;

                    while pos + window_size <= data.len() {
                        let window = &data[pos..pos + window_size];
                        let processed = f(window);

                        if processed.len() != window_size {
                            return Err(crate::AudioSampleError::InvalidParameter(format!(
                                "Window function must return {} samples, got {}",
                                window_size,
                                processed.len()
                            )));
                        }

                        if channel_result.len() < pos + window_size {
                            channel_result.resize(pos + window_size, T::default());
                        }

                        // Overlap-add the processed window
                        for (i, &sample) in processed.iter().enumerate() {
                            channel_result[pos + i] = sample;
                        }

                        pos += hop_size;
                    }

                    // Handle remaining samples
                    if pos < data.len() {
                        channel_result.extend_from_slice(&data[pos..]);
                    }

                    result.extend(channel_result);
                }

                let new_samples = result.len() / channels;
                *arr = Array2::from_shape_vec((channels, new_samples), result).map_err(|e| {
                    crate::AudioSampleError::InvalidParameter(format!("Array shape error: {}", e))
                })?;
            }
        }

        Ok(())
    }

    /// Applies a function to each channel individually.
    ///
    /// This method applies the given function to each channel of the audio data.
    /// For mono audio, the function is applied to the single channel.
    /// For multi-channel audio, the function is applied to each channel separately.
    ///
    /// # Arguments
    /// * `f` - A function that takes a channel index and mutable slice of samples
    ///
    /// # Returns
    /// A result indicating success or failure
    ///
    /// # Example
    /// ```rust,ignore
    /// // Apply different gains to different channels
    /// audio.apply_channels(|channel, samples| {
    ///     let gain = match channel {
    ///         0 => 0.8, // Left channel
    ///         1 => 0.9, // Right channel
    ///         _ => 1.0, // Other channels
    ///     };
    ///     for sample in samples.iter_mut() {
    ///         *sample = *sample * gain;
    ///     }
    ///     Ok(())
    /// })?;
    /// ```
    pub fn apply_channels<F>(&mut self, f: F) -> crate::AudioSampleResult<()>
    where
        F: Fn(usize, &mut [T]) -> crate::AudioSampleResult<()>,
    {
        match &mut self.data {
            AudioData::Mono(arr) => {
                let mut_slice = arr
                    .as_slice_mut()
                    .ok_or(AudioSampleError::ArrayLayoutError {
                        message: "Mono samples must be contiguous".to_string(),
                    })?;
                f(0, mut_slice)?;
            }
            AudioData::MultiChannel(arr) => {
                for (ch, mut row) in arr.axis_iter_mut(ndarray::Axis(0)).enumerate() {
                    let mut_slice =
                        row.as_slice_mut()
                            .ok_or(AudioSampleError::ArrayLayoutError {
                                message: "Multi-channel samples must be contiguous".to_string(),
                            })?;
                    f(ch, mut_slice)?;
                }
            }
        }
        Ok(())
    }

    /// Applies a function to each sample and returns a new AudioSamples instance.
    ///
    /// This is a functional-style version of `apply` that doesn't modify the original
    /// audio data but returns a new instance with the transformed samples.
    ///
    /// # Arguments
    /// * `f` - A function that takes a sample and returns a transformed sample
    ///
    /// # Returns
    /// A new AudioSamples instance with the transformed samples
    ///
    /// # Example
    /// ```rust,ignore
    /// // Create a new audio instance with halved amplitude
    /// let quieter_audio = audio.map(|sample| sample * 0.5)?;
    ///
    /// // Create a new audio instance with clipped samples
    /// let clipped_audio = audio.map(|sample| sample.clamp(-0.8, 0.8))?;
    /// ```
    pub fn map<F>(self, f: F) -> crate::AudioSampleResult<Self>
    where
        F: Fn(T) -> T,
    {
        let mut result = self;
        result.apply(f)?;
        Ok(result)
    }

    /// Applies a function to each sample with access to the sample index.
    ///
    /// This method is similar to `apply` but provides the sample index to the function,
    /// allowing for position-dependent transformations.
    ///
    /// # Arguments
    /// * `f` - A function that takes a sample index and sample value, returns transformed sample
    ///
    /// # Returns
    /// A result indicating success or failure
    ///
    /// # Example
    /// ```rust,ignore
    /// // Apply a fade-in effect
    /// audio.apply_indexed(|index, sample| {
    ///     let fade_samples = 44100; // 1 second fade at 44.1kHz
    ///     let gain = if index < fade_samples {
    ///         index as f32 / fade_samples as f32
    ///     } else {
    ///         1.0
    ///     };
    ///     sample * gain
    /// })?;
    /// ```
    pub fn apply_indexed<F>(&mut self, f: F) -> crate::AudioSampleResult<()>
    where
        F: Fn(usize, T) -> T,
    {
        match &mut self.data {
            AudioData::Mono(arr) => {
                for (i, sample) in arr.iter_mut().enumerate() {
                    *sample = f(i, *sample);
                }
            }
            AudioData::MultiChannel(arr) => {
                for (i, sample) in arr.iter_mut().enumerate() {
                    *sample = f(i, *sample);
                }
            }
        }
        Ok(())
    }

    /// Applies a function to samples in chunks of a specified size.
    ///
    /// This method processes the audio data in non-overlapping chunks of the specified size,
    /// applying the given function to each chunk. This is useful for block-based processing.
    ///
    /// # Arguments
    /// * `chunk_size` - Size of each chunk in samples
    /// * `f` - A function that takes a chunk slice and returns transformed samples
    ///
    /// # Returns
    /// A result indicating success or failure
    ///
    /// # Example
    /// ```rust,ignore
    /// // Apply RMS normalization to each chunk
    /// audio.apply_chunks(1024, |chunk| {
    ///     let rms = (chunk.iter().map(|&x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
    ///     let target_rms = 0.5;
    ///     let gain = if rms > 0.0 { target_rms / rms } else { 1.0 };
    ///     chunk.iter().map(|&x| x * gain).collect()
    /// })?;
    /// ```
    pub fn apply_chunks<F>(&mut self, chunk_size: usize, f: F) -> crate::AudioSampleResult<()>
    where
        F: Fn(&[T]) -> Vec<T>,
    {
        if chunk_size == 0 {
            return Err(crate::AudioSampleError::InvalidParameter(
                "Chunk size must be greater than 0".to_string(),
            ));
        }

        match &mut self.data {
            AudioData::Mono(arr) => {
                let mut result = Vec::new();
                let data = arr.as_slice().ok_or(AudioSampleError::ArrayLayoutError {
                    message: "Mono samples must be contiguous".to_string(),
                })?;

                // Process chunks
                for chunk in data.chunks(chunk_size) {
                    let processed = f(chunk);
                    result.extend(processed);
                }

                *arr = Array1::from_vec(result);
            }
            AudioData::MultiChannel(arr) => {
                let (channels, _) = arr.dim();
                let mut result = Vec::new();

                // Process each channel separately
                for ch in 0..channels {
                    let channel_data = arr.row(ch).to_owned();
                    let data =
                        channel_data
                            .as_slice()
                            .ok_or(AudioSampleError::ArrayLayoutError {
                                message: "Multi-channel samples must be contiguous".to_string(),
                            })?;

                    let mut channel_result = Vec::new();
                    for chunk in data.chunks(chunk_size) {
                        let processed = f(chunk);
                        channel_result.extend(processed);
                    }

                    result.extend(channel_result);
                }

                let new_samples = result.len() / channels;
                *arr = Array2::from_shape_vec((channels, new_samples), result).map_err(|e| {
                    crate::AudioSampleError::InvalidParameter(format!("Array shape error: {}", e))
                })?;
            }
        }

        Ok(())
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

    #[test]
    fn test_apply_simple() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut audio = AudioSamples::new_mono(data, 44100);

        // Apply a simple scaling
        audio.apply(|sample| sample * 2.0).unwrap();

        let expected = array![2.0f32, 4.0, 6.0, 8.0, 10.0];
        assert_eq!(audio.as_mono().unwrap(), &expected);
    }

    #[test]
    fn test_apply_channels() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]]; // 2 channels, 2 samples each
        let mut audio = AudioSamples::new_multi_channel(data, 44100);

        // Apply different gains to different channels
        audio
            .apply_channels(|channel, samples| {
                let gain = match channel {
                    0 => 2.0,
                    1 => 3.0,
                    _ => 1.0,
                };
                for sample in samples.iter_mut() {
                    *sample = *sample * gain;
                }
                Ok(())
            })
            .unwrap();

        let expected = array![[2.0f32, 4.0], [9.0, 12.0]];
        assert_eq!(audio.as_multi_channel().unwrap(), &expected);
    }

    #[test]
    fn test_map_functional() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioSamples::new_mono(data, 44100);

        // Create a new audio instance with transformed samples
        let new_audio = audio.clone().map(|sample| sample * 0.5).unwrap();

        let expected = array![0.5f32, 1.0, 1.5, 2.0, 2.5];
        assert_eq!(new_audio.as_mono().unwrap(), &expected);

        // Original should be unchanged
        let original_expected = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(audio.as_mono().unwrap(), &original_expected);
    }

    #[test]
    fn test_apply_indexed() {
        let data = array![1.0f32, 1.0, 1.0, 1.0, 1.0];
        let mut audio = AudioSamples::new_mono(data, 44100);

        // Apply index-based transformation
        audio
            .apply_indexed(|index, sample| sample * (index + 1) as f32)
            .unwrap();

        let expected = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(audio.as_mono().unwrap(), &expected);
    }

    #[test]
    fn test_apply_chunks() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut audio = AudioSamples::new_mono(data, 44100);

        // Apply chunk-based transformation (double each chunk)
        audio
            .apply_chunks(2, |chunk| chunk.iter().map(|&x| x * 2.0).collect())
            .unwrap();

        let expected = array![2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0];
        assert_eq!(audio.as_mono().unwrap(), &expected);
    }

    #[test]
    fn test_apply_windowed() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut audio = AudioSamples::new_mono(data, 44100);

        // Apply windowed transformation (simple identity for testing)
        audio
            .apply_windowed(3, 2, |window| {
                window.to_vec() // Just return the window unchanged
            })
            .unwrap();

        // The length should be preserved or extended based on windowing
        assert!(audio.samples_per_channel() >= 6);
    }

    #[test]
    fn test_apply_error_handling() {
        let data = array![1.0f32, 2.0, 3.0];
        let mut audio = AudioSamples::new_mono(data, 44100);

        // Test error handling for invalid chunk size
        let result = audio.apply_chunks(0, |chunk| chunk.to_vec());
        assert!(result.is_err());

        // Test error handling for invalid window size
        let result = audio.apply_windowed(0, 1, |window| window.to_vec());
        assert!(result.is_err());

        // Test error handling for invalid hop size
        let result = audio.apply_windowed(2, 0, |window| window.to_vec());
        assert!(result.is_err());
    }
}
