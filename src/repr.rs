//! Core audio sample representation and data structures.
//!
//! This module provides the fundamental building blocks for representing audio data
//! within the `audio_samples` library. It defines an enhanced wrapper around `ndarray`
//! that combines raw audio samples with essential metadata like sample rate, channel
//! configuration, and layout information.
//!
//! # Architecture Overview
//!
//! The core types in this module form the foundation of all audio processing operations:
//!
//! - [`AudioData<T>`] - Internal enum for mono vs. multi-channel audio data storage
//! - [`AudioSamples<T>`] - Main struct combining audio data with metadata
//! - Generic over any type `T` that implements the [`AudioSample`] trait
//!
//! # Key Design Principles
//!
//! ## Type Safety
//! All audio data is strongly typed with the sample format (i16, i24, i32, f32, f64),
//! ensuring mathematical operations are performed with appropriate precision and range.
//!
//! ## Memory Efficiency  
//! Uses `ndarray` for efficient memory layout with both contiguous and strided access patterns.
//! Mono audio uses 1D arrays, while multi-channel audio uses 2D arrays with channels as rows.
//!
//! ## Metadata Integration
//! Audio samples are always paired with essential metadata (sample rate, channel layout)
//! to prevent common audio processing errors and enable automatic format conversions.
//!
//! # Examples
//!
//! ## Creating Audio Data
//!
//! ```rust
//! use audio_samples::{AudioSamples, AudioData};
//! use ndarray::{array, Array1, Array2};
//!
//! // Create mono audio from 1D array
//! let mono_data = array![0.1f32, 0.2, 0.3, 0.4, 0.5];
//! let mono_audio = AudioSamples::new_mono(mono_data, 44100);
//!
//! assert_eq!(mono_audio.num_channels(), 1);
//! assert_eq!(mono_audio.samples_per_channel(), 5);
//! assert_eq!(mono_audio.sample_rate(), 44100);
//! ```
//!
//! ## Multi-Channel Audio
//!
//! ```rust
//! use ndarray::array;
//! # use audio_samples::AudioSamples;
//!
//! // Create stereo audio (2 channels × 3 samples)
//! let stereo_data = array![
//!     [0.1f32, 0.2, 0.3],  // Left channel
//!     [0.4f32, 0.5, 0.6]   // Right channel
//! ];
//! let stereo_audio = AudioSamples::new_multi_channel(stereo_data, 48000);
//!
//! assert_eq!(stereo_audio.num_channels(), 2);
//! assert_eq!(stereo_audio.samples_per_channel(), 3);
//! assert_eq!(stereo_audio.total_samples(), 6);
//! ```
//!
//! ## Working with Different Sample Types
//!
//! ```rust
//! # use audio_samples::AudioSamples;
//! # use ndarray::array;
//!
//! // 16-bit integer audio (CD quality)
//! let cd_audio = AudioSamples::new_mono(array![1000i16, 2000, 3000], 44100);
//!
//! // High-precision floating point
//! let hifi_audio = AudioSamples::new_mono(array![0.1f64, 0.2, 0.3], 96000);
//!
//! // Both can be processed using the same trait methods
//! assert_eq!(cd_audio.duration_seconds(), hifi_audio.duration_seconds());
//! ```
//!
//! ## Advanced Sample Processing
//!
//! The [`AudioSamples`] struct provides powerful methods for sample-wise transformations:
//!
//! ```rust
//! # use audio_samples::AudioSamples;
//! # use ndarray::array;
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0], 44100);
//!
//! // Apply gain to all samples
//! audio.apply(|sample| sample * 0.5)?;
//!
//! // Apply position-dependent processing  
//! audio.apply_indexed(|index, sample| {
//!     sample * (1.0 - index as f32 * 0.1) // Fade out
//! })?;
//!
//! // Process in overlapping windows (useful for FFT)
//! audio.apply_windowed(1024, 512, |window| {
//!     window.to_vec() // Identity function for demonstration
//! })?;
//! # Ok(())
//! # }
//! ```
//!
//! # Memory Layout
//!
//! ## Mono Audio (`AudioData::Mono`)
//! - Stored as `Array1<T>` (1-dimensional array)
//! - Direct memory access with `as_mono()` / `as_mono_mut()`
//! - Optimal for single-channel processing
//!
//! ## Multi-Channel Audio (`AudioData::MultiChannel`)  
//! - Stored as `Array2<T>` with shape `(channels, samples_per_channel)`
//! - Each row represents one channel of audio data
//! - Enables efficient per-channel operations with `apply_channels()`
//!
//! # Integration with Operations
//!
//! This module provides the data foundation that all operations in the
//! [`operations`](crate::operations) module work with:
//!
//! ```rust,ignore
//! use audio_samples::{AudioSamples, operations::*};
//!
//! let mut audio = AudioSamples::new_mono(/* ... */, 44100);
//!
//! // All operations work on AudioSamples<T>
//! let rms = audio.rms()?;
//! audio.normalize(-1.0, 1.0, NormalizationMethod::Peak)?;
//! let spectrum = audio.fft(None)?;
//! ```
//!
//! See [`AudioSamples`] for the complete API documentation.
//!
//! [`AudioSample`]: crate::AudioSample
use std::ops::{Index, IndexMut, RangeBounds};

use ndarray::{Array1, Array2, s};

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
    pub fn num_channels(&self) -> usize {
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
        self.num_channels() * self.samples_per_channel()
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
                if arr.is_empty() {
                    return T::default();
                }
                
                // Use ndarray's vectorized operations for SIMD-optimized absolute value and max
                let abs_values = arr.mapv(|x| {
                    // Manual absolute value that works with existing trait bounds
                    if x < T::default() {
                        T::default() - x
                    } else {
                        x
                    }
                });
                
                // Use ndarray's efficient fold operation instead of iterator chains
                abs_values.fold(T::default(), |acc, &x| {
                    if x > acc { x } else { acc }
                })
            }
            AudioData::MultiChannel(arr) => {
                if arr.is_empty() {
                    return T::default();
                }
                
                // Vectorized absolute value and max across entire multi-channel array
                let abs_values = arr.mapv(|x| {
                    if x < T::default() {
                        T::default() - x
                    } else {
                        x
                    }
                });
                
                abs_values.fold(T::default(), |acc, &x| {
                    if x > acc { x } else { acc }
                })
            }
        }
    }

    /// Returns the minimum value in the audio samples using vectorized operations
    pub fn min_native(&self) -> T
    where
        T: PartialOrd + Copy,
    {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() {
                    return T::default();
                }
                // Use ndarray's efficient fold operation for vectorized minimum finding
                arr.fold(arr[0], |acc, &x| if x < acc { x } else { acc })
            }
            AudioData::MultiChannel(arr) => {
                if arr.is_empty() {
                    return T::default();
                }
                // Vectorized minimum across entire multi-channel array
                arr.fold(arr[[0, 0]], |acc, &x| if x < acc { x } else { acc })
            }
        }
    }

    /// Returns the maximum value in the audio samples using vectorized operations
    pub fn max_native(&self) -> T
    where
        T: PartialOrd + Copy,
    {
        match &self.data {
            AudioData::Mono(arr) => {
                if arr.is_empty() {
                    return T::default();
                }
                // Use ndarray's efficient fold operation for vectorized maximum finding
                arr.fold(arr[0], |acc, &x| if x > acc { x } else { acc })
            }
            AudioData::MultiChannel(arr) => {
                if arr.is_empty() {
                    return T::default();
                }
                // Vectorized maximum across entire multi-channel array
                arr.fold(arr[[0, 0]], |acc, &x| if x > acc { x } else { acc })
            }
        }
    }

    /// Returns a reference to the underlying mono array, if this is mono audio
    pub const fn as_mono(&self) -> Option<&Array1<T>> {
        match &self.data {
            AudioData::Mono(arr) => Some(arr),
            AudioData::MultiChannel(_) => None,
        }
    }

    /// Returns a reference to the underlying mono array, does this without checking
    pub const unsafe fn as_mono_unchecked(&self) -> &Array1<T> {
        match &self.data {
            AudioData::Mono(arr) => arr,
            AudioData::MultiChannel(_) => {
                panic!("Called as_mono_unchecked on multi-channel audio")
            }
        }
    }

    /// Returns a reference to the underlying multi-channel array, if this is multi-channel audio
    pub const fn as_multi_channel(&self) -> Option<&Array2<T>> {
        match &self.data {
            AudioData::Mono(_) => None,
            AudioData::MultiChannel(arr) => Some(arr),
        }
    }

    /// Returns a reference to the underlying multi-channel array, does this without checking
    pub const unsafe fn as_multi_channel_unchecked(&self) -> &Array2<T> {
        match &self.data {
            AudioData::Mono(_) => panic!("Called as_multi_channel_unchecked on mono audio"),
            AudioData::MultiChannel(arr) => arr,
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

    /// Applies a function to overlapping windows of samples in-place with optimized memory usage.
    ///
    /// This method processes the audio data in overlapping windows, applying
    /// the given function to each window. Uses pre-allocated buffers to avoid
    /// memory allocations during processing for real-time performance.
    ///
    /// # Arguments
    /// * `window_size` - Size of each window in samples
    /// * `hop_size` - Number of samples to advance between windows
    /// * `f` - A function that takes input window slice and output buffer slice
    ///
    /// # Returns
    /// A result containing the processed audio or an error
    ///
    /// # Example
    /// ```rust,ignore
    /// // Apply a Hann window to overlapping frames (zero-allocation version)
    /// audio.apply_windowed_inplace(1024, 512, |input, output| {
    ///     for (i, (&sample, out)) in input.iter().zip(output.iter_mut()).enumerate() {
    ///         *out = sample * hann_window[i];
    ///     }
    /// })?;
    /// ```
    pub fn apply_windowed_inplace<F>(
        &mut self,
        window_size: usize,
        hop_size: usize,
        f: F,
    ) -> crate::AudioSampleResult<()>
    where
        F: Fn(&[T], &mut [T]),
    {
        if window_size == 0 || hop_size == 0 {
            return Err(crate::AudioSampleError::InvalidParameter(
                "Window size and hop size must be greater than 0".to_string(),
            ));
        }

        match &mut self.data {
            AudioData::Mono(arr) => {
                let data = arr.as_slice().ok_or(AudioSampleError::ArrayLayoutError {
                    message: "Mono samples must be contiguous".to_string(),
                })?;

                // Pre-calculate output size to avoid reallocations
                let num_windows = if data.len() >= window_size {
                    (data.len() - window_size) / hop_size + 1
                } else {
                    0
                };
                
                if num_windows == 0 {
                    return Ok(()); // No processing needed
                }

                let output_len = (num_windows - 1) * hop_size + window_size;
                let mut result = vec![T::default(); output_len];
                let mut overlap_count = vec![0usize; output_len];
                
                // Pre-allocate working buffer for window processing
                let mut window_buffer = vec![T::default(); window_size];

                // Process each window with pre-allocated buffers
                let mut pos = 0;
                while pos + window_size <= data.len() {
                    let window = &data[pos..pos + window_size];
                    
                    // Process window in-place to avoid allocation
                    f(window, &mut window_buffer);

                    // Overlap-add with pre-allocated result buffer
                    for (i, &processed_sample) in window_buffer.iter().enumerate() {
                        if pos + i < result.len() {
                            result[pos + i] = result[pos + i] + processed_sample;
                            overlap_count[pos + i] += 1;
                        }
                    }

                    pos += hop_size;
                }

                // Normalize by overlap count
                for (sample, &count) in result.iter_mut().zip(overlap_count.iter()) {
                    if count > 1 {
                        // For overlapping regions, average the contributions
                        *sample = *sample / T::cast_from(count);
                    }
                }

                // Handle any remaining samples that weren't windowed
                if pos < data.len() && result.len() < data.len() {
                    result.extend_from_slice(&data[result.len()..]);
                }

                *arr = Array1::from_vec(result);
                Ok(())
            }
            AudioData::MultiChannel(arr) => {
                let (channels, samples_per_channel) = arr.dim();
                
                // Pre-calculate dimensions
                let num_windows = if samples_per_channel >= window_size {
                    (samples_per_channel - window_size) / hop_size + 1
                } else {
                    0
                };
                
                if num_windows == 0 {
                    return Ok(()); // No processing needed
                }

                let output_len = (num_windows - 1) * hop_size + window_size;
                let mut result = vec![T::default(); channels * output_len];
                let mut overlap_count = vec![0usize; output_len];
                
                // Pre-allocate working buffers (reused across channels)
                let mut window_buffer = vec![T::default(); window_size];
                let mut channel_buffer = vec![T::default(); samples_per_channel];

                // Process each channel with optimized memory usage
                for ch in 0..channels {
                    // Extract channel data without cloning the entire row
                    for (i, &sample) in arr.row(ch).iter().enumerate() {
                        channel_buffer[i] = sample;
                    }
                    
                    overlap_count.fill(0); // Reset overlap count for this channel
                    
                    let mut pos = 0;
                    while pos + window_size <= samples_per_channel {
                        let window = &channel_buffer[pos..pos + window_size];
                        
                        // Process window in-place
                        f(window, &mut window_buffer);

                        // Overlap-add to result buffer
                        let channel_offset = ch * output_len;
                        for (i, &processed_sample) in window_buffer.iter().enumerate() {
                            if pos + i < output_len {
                                let result_idx = channel_offset + pos + i;
                                result[result_idx] = result[result_idx] + processed_sample;
                                overlap_count[pos + i] += 1;
                            }
                        }

                        pos += hop_size;
                    }

                    // Normalize by overlap count for this channel
                    let channel_offset = ch * output_len;
                    for (i, &count) in overlap_count.iter().enumerate() {
                        if count > 1 && channel_offset + i < result.len() {
                            result[channel_offset + i] = result[channel_offset + i] / T::cast_from(count);
                        }
                    }

                    // Handle remaining samples for this channel
                    if pos < samples_per_channel && output_len < samples_per_channel {
                        let remaining_start = output_len;
                        let channel_offset = ch * samples_per_channel;
                        
                        // Extend result if needed
                        while result.len() < (ch + 1) * samples_per_channel {
                            result.push(T::default());
                        }
                        
                        // Copy remaining samples
                        for (i, &sample) in channel_buffer[remaining_start..].iter().enumerate() {
                            if channel_offset + remaining_start + i < result.len() {
                                result[channel_offset + remaining_start + i] = sample;
                            }
                        }
                    }
                }

                let final_samples_per_channel = result.len() / channels;
                *arr = Array2::from_shape_vec((channels, final_samples_per_channel), result)
                    .map_err(|e| {
                        crate::AudioSampleError::InvalidParameter(format!("Array shape error: {}", e))
                    })?;
                
                Ok(())
            }
        }
    }

    /// Legacy windowed operation with allocation per window (deprecated).
    ///
    /// This method is kept for backward compatibility but allocates memory
    /// for each window. Use `apply_windowed_inplace` for better performance.
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

    // ========================
    // Indexing and Slicing Methods - Leveraging ndarray
    // ========================

    /// Create a sliced view of the audio data using ndarray's SliceInfo.
    ///
    /// This method provides direct access to ndarray's powerful slicing capabilities,
    /// allowing you to use the `s!` macro for flexible array slicing.
    ///
    /// # Arguments
    /// * `info` - SliceInfo that can be created using the `s![]` macro
    ///
    /// # Returns
    /// A new `AudioSamples` instance containing the sliced data with preserved metadata
    ///
    /// # Examples
    /// ```rust
    /// use audio_samples::AudioSamples;
    /// use ndarray::{array, s};
    ///
    /// // Create test audio
    /// let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
    /// let audio = AudioSamples::new_mono(data, 44100);
    ///
    /// // Slice samples 1 through 3
    /// let sliced = audio.slice(s![1..4]).unwrap();
    /// assert_eq!(sliced.samples_per_channel(), 3);
    /// ```
    pub fn slice_1d<Si>(&self, info: Si) -> crate::AudioSampleResult<Self>
    where
        Si: ndarray::SliceArg<ndarray::Ix1, OutDim = ndarray::Ix1>,
    {
        match &self.data {
            AudioData::Mono(arr) => {
                let sliced = arr.slice(info);
                Ok(AudioSamples::new_mono(sliced.to_owned(), self.sample_rate))
            }
            AudioData::MultiChannel(_) => Err(crate::AudioSampleError::InvalidParameter(
                "slice_1d can only be used with mono audio. Use slice_2d for multi-channel audio.".to_string(),
            )),
        }
    }

    /// Create a sliced view of the audio data for 2D multi-channel arrays.
    ///
    /// This is a specialized version of `slice` for multi-channel audio that accepts
    /// 2D SliceInfo objects.
    ///
    /// # Arguments  
    /// * `info` - SliceInfo for 2D arrays (channels, samples)
    ///
    /// # Examples
    /// ```rust
    /// use audio_samples::AudioSamples;
    /// use ndarray::{array, s};
    ///
    /// let stereo_data = array![[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
    /// let audio = AudioSamples::new_multi_channel(stereo_data, 44100);
    ///
    /// // Slice both channels, samples 1-3
    /// let sliced = audio.slice_2d(s![.., 1..4]).unwrap();
    /// assert_eq!(sliced.samples_per_channel(), 3);
    /// assert_eq!(sliced.num_channels(), 2);
    /// ```
    pub fn slice_2d<Si>(&self, info: Si) -> crate::AudioSampleResult<Self>
    where
        Si: ndarray::SliceArg<ndarray::Ix2, OutDim = ndarray::Ix2>,
    {
        match &self.data {
            AudioData::Mono(_) => Err(crate::AudioSampleError::InvalidParameter(
                "slice_2d can only be used with multi-channel audio".to_string(),
            )),
            AudioData::MultiChannel(arr) => {
                let sliced = arr.slice(info);
                Ok(AudioSamples::new_multi_channel(sliced.to_owned(), self.sample_rate))
            }
        }
    }

    /// Slice only the samples dimension, keeping all channels.
    ///
    /// This is a convenience method for the common case of slicing along the time axis
    /// while preserving all channels.
    ///
    /// # Arguments
    /// * `range` - Range of samples to extract
    ///
    /// # Examples
    /// ```rust
    /// use audio_samples::AudioSamples;
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
    /// let audio = AudioSamples::new_mono(data, 44100);
    ///
    /// // Extract samples 1 through 3 (exclusive end)
    /// let sliced = audio.slice_samples(1..4).unwrap();
    /// assert_eq!(sliced.samples_per_channel(), 3);
    /// ```
    pub fn slice_samples<R>(&self, range: R) -> crate::AudioSampleResult<Self>
    where
        R: RangeBounds<usize> + Clone,
    {
        // Convert RangeBounds to actual start/end indices
        let len = self.samples_per_channel();
        let start = match range.start_bound() {
            std::ops::Bound::Included(&n) => n,
            std::ops::Bound::Excluded(&n) => n + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Included(&n) => n + 1,
            std::ops::Bound::Excluded(&n) => n,
            std::ops::Bound::Unbounded => len,
        };

        if start >= len || end > len || start >= end {
            return Err(crate::AudioSampleError::InvalidParameter(format!(
                "Invalid sample range {}..{} for audio with {} samples",
                start, end, len
            )));
        }

        match &self.data {
            AudioData::Mono(arr) => {
                let sliced = arr.slice(s![start..end]).to_owned();
                Ok(AudioSamples::new_mono(sliced, self.sample_rate))
            }
            AudioData::MultiChannel(arr) => {
                let sliced = arr.slice(s![.., start..end]).to_owned();
                Ok(AudioSamples::new_multi_channel(sliced, self.sample_rate))
            }
        }
    }

    /// Slice only the channels dimension for multi-channel audio.
    ///
    /// This extracts a subset of channels while keeping all samples.
    ///
    /// # Arguments  
    /// * `range` - Range of channels to extract
    ///
    /// # Examples
    /// ```rust
    /// use audio_samples::AudioSamples;
    /// use ndarray::array;
    ///
    /// // Create 3-channel audio
    /// let data = array![[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
    /// let audio = AudioSamples::new_multi_channel(data, 44100);
    ///
    /// // Extract channels 0 and 1 (first two channels)  
    /// let sliced = audio.slice_channels(0..2).unwrap();
    /// assert_eq!(sliced.num_channels(), 2);
    /// assert_eq!(sliced.samples_per_channel(), 2);
    /// ```
    pub fn slice_channels<R>(&self, range: R) -> crate::AudioSampleResult<Self>
    where
        R: RangeBounds<usize>,
    {
        match &self.data {
            AudioData::Mono(_) => Err(crate::AudioSampleError::InvalidParameter(
                "slice_channels can only be used with multi-channel audio".to_string(),
            )),
            AudioData::MultiChannel(arr) => {
                let len = self.num_channels();
                let start = match range.start_bound() {
                    std::ops::Bound::Included(&n) => n,
                    std::ops::Bound::Excluded(&n) => n + 1,
                    std::ops::Bound::Unbounded => 0,
                };
                let end = match range.end_bound() {
                    std::ops::Bound::Included(&n) => n + 1,
                    std::ops::Bound::Excluded(&n) => n,
                    std::ops::Bound::Unbounded => len,
                };

                if start >= len || end > len || start >= end {
                    return Err(crate::AudioSampleError::InvalidParameter(format!(
                        "Invalid channel range {}..{} for audio with {} channels",
                        start, end, len
                    )));
                }

                let sliced = arr.slice(s![start..end, ..]).to_owned();
                Ok(AudioSamples::new_multi_channel(sliced, self.sample_rate))
            }
        }
    }

    /// Slice both channels and samples dimensions simultaneously.
    ///
    /// This provides a convenient way to extract a rectangular region from multi-channel audio.
    ///
    /// # Arguments
    /// * `channel_range` - Range of channels to extract
    /// * `sample_range` - Range of samples to extract
    ///
    /// # Examples
    /// ```rust
    /// use audio_samples::AudioSamples;
    /// use ndarray::array;
    ///
    /// // Create 3x4 audio (3 channels, 4 samples each)
    /// let data = array![
    ///     [1.0f32, 2.0, 3.0, 4.0],
    ///     [5.0, 6.0, 7.0, 8.0],
    ///     [9.0, 10.0, 11.0, 12.0]
    /// ];
    /// let audio = AudioSamples::new_multi_channel(data, 44100);
    ///
    /// // Extract channels 1-2, samples 1-3
    /// let sliced = audio.slice_both(1..3, 1..4).unwrap();
    /// assert_eq!(sliced.num_channels(), 2);
    /// assert_eq!(sliced.samples_per_channel(), 3);
    /// ```
    pub fn slice_both<CR, SR>(
        &self,
        channel_range: CR,
        sample_range: SR,
    ) -> crate::AudioSampleResult<Self>
    where
        CR: RangeBounds<usize>,
        SR: RangeBounds<usize>,
    {
        match &self.data {
            AudioData::Mono(_) => Err(crate::AudioSampleError::InvalidParameter(
                "slice_both can only be used with multi-channel audio".to_string(),
            )),
            AudioData::MultiChannel(arr) => {
                let num_channels = self.num_channels();
                let num_samples = self.samples_per_channel();

                // Convert channel range
                let ch_start = match channel_range.start_bound() {
                    std::ops::Bound::Included(&n) => n,
                    std::ops::Bound::Excluded(&n) => n + 1,
                    std::ops::Bound::Unbounded => 0,
                };
                let ch_end = match channel_range.end_bound() {
                    std::ops::Bound::Included(&n) => n + 1,
                    std::ops::Bound::Excluded(&n) => n,
                    std::ops::Bound::Unbounded => num_channels,
                };

                // Convert sample range
                let s_start = match sample_range.start_bound() {
                    std::ops::Bound::Included(&n) => n,
                    std::ops::Bound::Excluded(&n) => n + 1,
                    std::ops::Bound::Unbounded => 0,
                };
                let s_end = match sample_range.end_bound() {
                    std::ops::Bound::Included(&n) => n + 1,
                    std::ops::Bound::Excluded(&n) => n,
                    std::ops::Bound::Unbounded => num_samples,
                };

                // Validate ranges
                if ch_start >= num_channels || ch_end > num_channels || ch_start >= ch_end {
                    return Err(crate::AudioSampleError::InvalidParameter(format!(
                        "Invalid channel range {}..{} for audio with {} channels",
                        ch_start, ch_end, num_channels
                    )));
                }

                if s_start >= num_samples || s_end > num_samples || s_start >= s_end {
                    return Err(crate::AudioSampleError::InvalidParameter(format!(
                        "Invalid sample range {}..{} for audio with {} samples",
                        s_start, s_end, num_samples
                    )));
                }

                let sliced = arr.slice(s![ch_start..ch_end, s_start..s_end]).to_owned();
                Ok(AudioSamples::new_multi_channel(sliced, self.sample_rate))
            }
        }
    }
}

// ========================
// Index and IndexMut implementations using ndarray delegation
// ========================

impl<T: AudioSample> Index<usize> for AudioSamples<T> {
    type Output = T;

    /// Index into mono audio samples by sample index.
    ///
    /// For mono audio, this returns the sample at the given index.
    /// For multi-channel audio, this will panic - use `[(channel, sample)]` instead.
    ///
    /// # Panics
    /// - If index is out of bounds
    /// - If used on multi-channel audio (use 2D indexing instead)
    ///
    /// # Examples
    /// ```rust
    /// use audio_samples::AudioSamples;
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
    /// let audio = AudioSamples::new_mono(data, 44100);
    ///
    /// assert_eq!(audio[0], 1.0);
    /// assert_eq!(audio[2], 3.0);
    /// ```
    fn index(&self, index: usize) -> &Self::Output {
        match &self.data {
            AudioData::Mono(arr) => &arr[index],
            AudioData::MultiChannel(_) => {
                panic!("Cannot use single index on multi-channel audio. Use (channel, sample) indexing instead.");
            }
        }
    }
}

impl<T: AudioSample> IndexMut<usize> for AudioSamples<T> {
    /// Mutable index into mono audio samples by sample index.
    ///
    /// For mono audio, this returns a mutable reference to the sample at the given index.
    /// For multi-channel audio, this will panic - use `[(channel, sample)]` instead.
    ///
    /// # Panics
    /// - If index is out of bounds
    /// - If used on multi-channel audio (use 2D indexing instead)
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match &mut self.data {
            AudioData::Mono(arr) => &mut arr[index],
            AudioData::MultiChannel(_) => {
                panic!("Cannot use single index on multi-channel audio. Use (channel, sample) indexing instead.");
            }
        }
    }
}

impl<T: AudioSample> Index<(usize, usize)> for AudioSamples<T> {
    type Output = T;

    /// Index into audio samples by (channel, sample) coordinates.
    ///
    /// This works for both mono and multi-channel audio:
    /// - For mono: only `(0, sample_index)` is valid
    /// - For multi-channel: `(channel_index, sample_index)`
    ///
    /// # Panics
    /// - If channel or sample index is out of bounds
    ///
    /// # Examples
    /// ```rust
    /// use audio_samples::AudioSamples;
    /// use ndarray::array;
    ///
    /// // Mono audio
    /// let mono_data = array![1.0f32, 2.0, 3.0];
    /// let mono_audio = AudioSamples::new_mono(mono_data, 44100);
    /// assert_eq!(mono_audio[(0, 1)], 2.0);
    ///
    /// // Multi-channel audio
    /// let stereo_data = array![[1.0f32, 2.0], [3.0, 4.0]];
    /// let stereo_audio = AudioSamples::new_multi_channel(stereo_data, 44100);
    /// assert_eq!(stereo_audio[(0, 1)], 2.0); // Channel 0, sample 1
    /// assert_eq!(stereo_audio[(1, 0)], 3.0); // Channel 1, sample 0
    /// ```
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (channel, sample) = index;
        match &self.data {
            AudioData::Mono(arr) => {
                if channel != 0 {
                    panic!("Channel index {} out of bounds for mono audio (only channel 0 exists)", channel);
                }
                &arr[sample]
            }
            AudioData::MultiChannel(arr) => &arr[(channel, sample)],
        }
    }
}

impl<T: AudioSample> IndexMut<(usize, usize)> for AudioSamples<T> {
    /// Mutable index into audio samples by (channel, sample) coordinates.
    ///
    /// This works for both mono and multi-channel audio:
    /// - For mono: only `(0, sample_index)` is valid  
    /// - For multi-channel: `(channel_index, sample_index)`
    ///
    /// # Panics
    /// - If channel or sample index is out of bounds
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (channel, sample) = index;
        match &mut self.data {
            AudioData::Mono(arr) => {
                if channel != 0 {
                    panic!("Channel index {} out of bounds for mono audio (only channel 0 exists)", channel);
                }
                &mut arr[sample]
            }
            AudioData::MultiChannel(arr) => &mut arr[(channel, sample)],
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
        assert_eq!(audio.num_channels(), 1);
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
        assert_eq!(audio.num_channels(), 2);
        assert_eq!(audio.samples_per_channel(), 3);
        assert_eq!(audio.total_samples(), 6);
        assert!(!audio.is_mono());
        assert!(audio.is_multi_channel());
        assert_eq!(audio.as_multi_channel().unwrap(), &data);
    }

    #[test]
    fn test_zeros_construction() {
        let mono_audio = AudioSamples::<f32>::zeros_mono(100, 44100);
        assert_eq!(mono_audio.num_channels(), 1);
        assert_eq!(mono_audio.samples_per_channel(), 100);
        assert_eq!(mono_audio.sample_rate(), 44100);

        let multi_audio = AudioSamples::<f32>::zeros_multi(2, 50, 48000);
        assert_eq!(multi_audio.num_channels(), 2);
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

    // ========================
    // Indexing and Slicing Tests
    // ========================

    #[test]
    fn test_index_mono_single() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioSamples::new_mono(data, 44100);

        assert_eq!(audio[0], 1.0);
        assert_eq!(audio[2], 3.0);
        assert_eq!(audio[4], 5.0);
    }

    #[test]
    #[should_panic(expected = "Cannot use single index on multi-channel audio")]
    fn test_index_mono_single_on_multichannel_panics() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let audio = AudioSamples::new_multi_channel(data, 44100);
        
        let _ = audio[0]; // Should panic
    }

    #[test]
    fn test_index_tuple() {
        // Test mono with tuple indexing
        let mono_data = array![1.0f32, 2.0, 3.0, 4.0];
        let mono_audio = AudioSamples::new_mono(mono_data, 44100);

        assert_eq!(mono_audio[(0, 1)], 2.0);
        assert_eq!(mono_audio[(0, 3)], 4.0);

        // Test multi-channel with tuple indexing
        let stereo_data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let stereo_audio = AudioSamples::new_multi_channel(stereo_data, 44100);

        assert_eq!(stereo_audio[(0, 0)], 1.0);
        assert_eq!(stereo_audio[(0, 2)], 3.0);
        assert_eq!(stereo_audio[(1, 0)], 4.0);
        assert_eq!(stereo_audio[(1, 2)], 6.0);
    }

    #[test]
    #[should_panic(expected = "Channel index 1 out of bounds for mono audio")]
    fn test_index_tuple_invalid_channel_mono() {
        let data = array![1.0f32, 2.0, 3.0];
        let audio = AudioSamples::new_mono(data, 44100);
        
        let _ = audio[(1, 0)]; // Should panic - mono only has channel 0
    }

    #[test]
    fn test_index_mut_mono() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut audio = AudioSamples::new_mono(data, 44100);

        audio[2] = 10.0;
        assert_eq!(audio[2], 10.0);
        assert_eq!(audio.as_mono().unwrap(), &array![1.0f32, 2.0, 10.0, 4.0, 5.0]);
    }

    #[test]
    fn test_index_mut_tuple() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio = AudioSamples::new_multi_channel(data, 44100);

        audio[(1, 0)] = 10.0;
        assert_eq!(audio[(1, 0)], 10.0);

        let expected = array![[1.0f32, 2.0], [10.0, 4.0]];
        assert_eq!(audio.as_multi_channel().unwrap(), &expected);
    }

    #[test]
    fn test_slice_samples() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let audio = AudioSamples::new_mono(data, 44100);

        // Test basic range slicing
        let sliced = audio.slice_samples(1..4).unwrap();
        assert_eq!(sliced.samples_per_channel(), 3);
        assert_eq!(sliced.as_mono().unwrap(), &array![2.0f32, 3.0, 4.0]);
        assert_eq!(sliced.sample_rate(), 44100); // Metadata preserved

        // Test with multi-channel
        let stereo_data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let stereo_audio = AudioSamples::new_multi_channel(stereo_data, 44100);

        let sliced_stereo = stereo_audio.slice_samples(1..3).unwrap();
        assert_eq!(sliced_stereo.num_channels(), 2);
        assert_eq!(sliced_stereo.samples_per_channel(), 2);
        
        let expected = array![[2.0f32, 3.0], [5.0, 6.0]];
        assert_eq!(sliced_stereo.as_multi_channel().unwrap(), &expected);
    }

    #[test]
    fn test_slice_samples_edge_cases() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioSamples::new_mono(data, 44100);

        // Test full range
        let full = audio.slice_samples(..).unwrap();
        assert_eq!(full.samples_per_channel(), 5);

        // Test from beginning
        let from_start = audio.slice_samples(..3).unwrap();
        assert_eq!(from_start.samples_per_channel(), 3);
        assert_eq!(from_start.as_mono().unwrap(), &array![1.0f32, 2.0, 3.0]);

        // Test to end
        let to_end = audio.slice_samples(2..).unwrap();
        assert_eq!(to_end.samples_per_channel(), 3);
        assert_eq!(to_end.as_mono().unwrap(), &array![3.0f32, 4.0, 5.0]);
    }

    #[test]
    fn test_slice_samples_errors() {
        let data = array![1.0f32, 2.0, 3.0];
        let audio = AudioSamples::new_mono(data, 44100);

        // Test out of bounds
        let result = audio.slice_samples(5..10);
        assert!(result.is_err());

        // Test empty range
        let result = audio.slice_samples(2..1);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_channels() {
        let data = array![
            [1.0f32, 2.0, 3.0],  // Channel 0
            [4.0, 5.0, 6.0],     // Channel 1
            [7.0, 8.0, 9.0]      // Channel 2
        ];
        let audio = AudioSamples::new_multi_channel(data, 44100);

        // Test single channel extraction
        let ch0 = audio.slice_channels(0..1).unwrap();
        assert_eq!(ch0.num_channels(), 1);
        assert_eq!(ch0.samples_per_channel(), 3);
        assert_eq!(ch0.as_multi_channel().unwrap(), &array![[1.0f32, 2.0, 3.0]]);

        // Test multiple channel extraction
        let ch01 = audio.slice_channels(0..2).unwrap();
        assert_eq!(ch01.num_channels(), 2);
        assert_eq!(ch01.samples_per_channel(), 3);
        
        let expected = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        assert_eq!(ch01.as_multi_channel().unwrap(), &expected);
    }

    #[test]
    fn test_slice_channels_error_on_mono() {
        let data = array![1.0f32, 2.0, 3.0];
        let audio = AudioSamples::new_mono(data, 44100);

        let result = audio.slice_channels(0..1);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_both() {
        let data = array![
            [1.0f32, 2.0, 3.0, 4.0],  // Channel 0
            [5.0, 6.0, 7.0, 8.0],     // Channel 1  
            [9.0, 10.0, 11.0, 12.0]   // Channel 2
        ];
        let audio = AudioSamples::new_multi_channel(data, 44100);

        // Extract channels 1-2, samples 1-3
        let sliced = audio.slice_both(1..3, 1..4).unwrap();
        assert_eq!(sliced.num_channels(), 2);
        assert_eq!(sliced.samples_per_channel(), 3);

        let expected = array![
            [6.0f32, 7.0, 8.0],    // Channel 1, samples 1-3
            [10.0, 11.0, 12.0]     // Channel 2, samples 1-3
        ];
        assert_eq!(sliced.as_multi_channel().unwrap(), &expected);
    }

    #[test]
    fn test_slice_both_error_on_mono() {
        let data = array![1.0f32, 2.0, 3.0];
        let audio = AudioSamples::new_mono(data, 44100);

        let result = audio.slice_both(0..1, 1..3);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_1d() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioSamples::new_mono(data, 44100);

        let sliced = audio.slice_1d(s![1..4]).unwrap();
        assert_eq!(sliced.samples_per_channel(), 3);
        assert_eq!(sliced.as_mono().unwrap(), &array![2.0f32, 3.0, 4.0]);
    }

    #[test]
    fn test_slice_1d_error_on_multichannel() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let audio = AudioSamples::new_multi_channel(data, 44100);

        let result = audio.slice_1d(s![0..2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_2d() {
        let data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let audio = AudioSamples::new_multi_channel(data, 44100);

        // Slice all channels, samples 1-3
        let sliced = audio.slice_2d(s![.., 1..3]).unwrap();
        assert_eq!(sliced.num_channels(), 2);
        assert_eq!(sliced.samples_per_channel(), 2);

        let expected = array![[2.0f32, 3.0], [5.0, 6.0]];
        assert_eq!(sliced.as_multi_channel().unwrap(), &expected);

        // Slice specific channels and samples
        let sliced2 = audio.slice_2d(s![0..1, 0..2]).unwrap();
        assert_eq!(sliced2.num_channels(), 1);
        assert_eq!(sliced2.samples_per_channel(), 2);

        let expected2 = array![[1.0f32, 2.0]];
        assert_eq!(sliced2.as_multi_channel().unwrap(), &expected2);
    }

    #[test]
    fn test_slice_2d_error_on_mono() {
        let data = array![1.0f32, 2.0, 3.0];
        let audio = AudioSamples::new_mono(data, 44100);

        let result = audio.slice_2d(s![0..1, 1..3]);
        assert!(result.is_err());
    }

    #[test] 
    fn test_slicing_preserves_metadata() {
        let data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let original_rate = 48000;
        let audio = AudioSamples::new_multi_channel(data, original_rate);

        // All slicing methods should preserve sample rate
        let samples_sliced = audio.slice_samples(1..3).unwrap();
        assert_eq!(samples_sliced.sample_rate(), original_rate);

        let channels_sliced = audio.slice_channels(0..1).unwrap();
        assert_eq!(channels_sliced.sample_rate(), original_rate);

        let both_sliced = audio.slice_both(0..2, 1..3).unwrap();
        assert_eq!(both_sliced.sample_rate(), original_rate);

        let nd_sliced = audio.slice_2d(s![.., 0..2]).unwrap();
        assert_eq!(nd_sliced.sample_rate(), original_rate);
    }

    #[test]
    fn test_slice_with_ndarray_patterns() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let audio = AudioSamples::new_mono(data, 44100);

        // Test step slicing (reverse)
        let sliced = audio.slice_1d(s![..;-1]).unwrap();
        assert_eq!(sliced.samples_per_channel(), 6);
        assert_eq!(sliced.as_mono().unwrap(), &array![6.0f32, 5.0, 4.0, 3.0, 2.0, 1.0]);

        // Test with multi-channel reverse
        let stereo_data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let stereo_audio = AudioSamples::new_multi_channel(stereo_data, 44100);

        let reversed_stereo = stereo_audio.slice_2d(s![.., ..;-1]).unwrap();
        let expected = array![[3.0f32, 2.0, 1.0], [6.0, 5.0, 4.0]];
        assert_eq!(reversed_stereo.as_multi_channel().unwrap(), &expected);
    }
}
