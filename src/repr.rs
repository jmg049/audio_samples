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
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};
use std::any::TypeId;
use std::fmt::Display;
use std::ops::{Add, Bound, Div, Index, IndexMut, Mul, Neg, RangeBounds, Sub};

use crate::operations::ProcessingBuilder;
use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioTypeConversion, ChainableResult,
    ChannelLayout, ConvertTo, I24,
};

pub type Array1<T> = ndarray::Array1<T>;
pub type Array2<T> = ndarray::Array2<T>;

/// Internal representation of audio data
#[derive(Debug, Clone, PartialEq)]
pub enum AudioData<T: AudioSample> {
    Mono(Array1<T>),         // Single channel audio samples
    MultiChannel(Array2<T>), // Multi-channel audio samples where each row is a channel
}

// Main implementation block for AudioData
impl<T: AudioSample> AudioData<T> {
    pub fn len(&self) -> usize {
        match self {
            AudioData::Mono(arr) => arr.len(),
            AudioData::MultiChannel(arr) => arr.len(),
        }
    }

    pub fn num_channels(&self) -> usize {
        match self {
            AudioData::Mono(_) => 1,
            AudioData::MultiChannel(arr) => arr.shape()[0],
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_mono(&self) -> bool {
        matches!(self, AudioData::Mono(_))
    }

    pub fn as_mono(&self) -> Option<&Array1<T>> {
        match self {
            AudioData::Mono(arr) => Some(arr),
            AudioData::MultiChannel(_) => None,
        }
    }

    pub unsafe fn as_mono_unchecked(&self) -> &Array1<T> {
        match self {
            AudioData::Mono(arr) => arr,
            AudioData::MultiChannel(_) => {
                panic!("Called as_mono_unchecked on non-mono audio")
            }
        }
    }

    pub fn as_mono_mut(&mut self) -> Option<&mut Array1<T>> {
        match self {
            AudioData::Mono(arr) => Some(arr),
            AudioData::MultiChannel(_) => None,
        }
    }

    pub unsafe fn as_mono_mut_unchecked(&mut self) -> &mut Array1<T> {
        match self {
            AudioData::Mono(arr) => arr,
            AudioData::MultiChannel(_) => {
                panic!("Called as_mono_mut_unchecked on non-mono audio")
            }
        }
    }

    pub fn is_multi_channel(&self) -> bool {
        matches!(self, AudioData::MultiChannel(_))
    }

    pub fn as_multi_channel(&self) -> Option<&Array2<T>> {
        match self {
            AudioData::Mono(_) => None,
            AudioData::MultiChannel(arr) => Some(arr),
        }
    }

    pub unsafe fn as_multi_channel_unchecked(&self) -> &Array2<T> {
        match self {
            AudioData::Mono(_) => panic!("Called as_multi_channel_unchecked on mono audio"),
            AudioData::MultiChannel(arr) => arr,
        }
    }

    pub fn as_multi_channel_mut(&mut self) -> Option<&mut Array2<T>> {
        match self {
            AudioData::Mono(_) => None,
            AudioData::MultiChannel(arr) => Some(arr),
        }
    }

    pub unsafe fn as_multi_channel_mut_unchecked(&mut self) -> &mut Array2<T> {
        match self {
            AudioData::Mono(_) => {
                panic!("Called as_multi_channel_mut_unchecked on mono audio")
            }
            AudioData::MultiChannel(arr) => arr,
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            AudioData::Mono(arr) => arr.shape(),
            AudioData::MultiChannel(arr) => arr.shape(),
        }
    }

    pub fn samples_per_channel(&self) -> usize {
        match &self {
            AudioData::Mono(arr) => arr.len(),
            AudioData::MultiChannel(arr) => arr.shape()[1],
        }
    }

    pub fn as_slice(&self) -> Option<&[T]> {
        match self {
            AudioData::Mono(arr) => arr.as_slice(),
            AudioData::MultiChannel(_) => None, // 2D arrays may not be contiguous
        }
    }

    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        match self {
            AudioData::Mono(arr) => arr.as_slice_mut(),
            AudioData::MultiChannel(_) => None, // 2D arrays may not be contiguous
        }
    }

    pub fn bytes_per_sample(&self) -> usize {
        match TypeId::of::<T>() {
            id if id == TypeId::of::<I24>() => 3,
            _ => std::mem::size_of::<T>(),
        }
    }

    pub fn as_bytes(&self) -> Vec<u8> {
        match self {
            AudioData::Mono(arr) => {
                let mut output = Vec::with_capacity(arr.len() * self.bytes_per_sample());

                for sample in arr.iter() {
                    output.extend_from_slice(&sample.to_bytes());
                }

                output
            }
            AudioData::MultiChannel(arr) => {
                let mut output = Vec::with_capacity(arr.len() * self.bytes_per_sample());

                for sample in arr.iter() {
                    output.extend_from_slice(&sample.to_bytes());
                }

                output
            }
        }
    }

    pub fn mapv<F, U>(&self, f: F) -> AudioData<U>
    where
        F: Fn(T) -> U,
        U: AudioSample,
    {
        match self {
            AudioData::Mono(arr) => AudioData::Mono(arr.mapv(f).into()),
            AudioData::MultiChannel(arr) => AudioData::MultiChannel(arr.mapv(f).into()),
        }
    }

    pub fn mapv_inplace<F>(&mut self, f: F)
    where
        F: Fn(T) -> T,
    {
        match self {
            AudioData::Mono(arr) => arr.mapv_inplace(f),
            AudioData::MultiChannel(arr) => arr.mapv_inplace(f),
        }
    }

    pub fn apply<F>(&mut self, func: F)
    where
        F: Fn(T) -> T,
    {
        match self {
            AudioData::Mono(arr) => arr.mapv_inplace(func),
            AudioData::MultiChannel(arr) => arr.mapv_inplace(func),
        }
    }

    pub fn apply_with_index<F>(&mut self, func: F)
    where
        F: Fn(usize, T) -> T,
    {
        match self {
            AudioData::Mono(arr) => {
                for (i, sample) in arr.iter_mut().enumerate() {
                    *sample = func(i, *sample);
                }
            }
            AudioData::MultiChannel(arr) => {
                for mut row in arr.rows_mut() {
                    for (i, sample) in row.iter_mut().enumerate() {
                        *sample = func(i, *sample);
                    }
                }
            }
        }
    }

    pub fn apply_windowed<F>(
        &mut self,
        window_size: usize,
        hop_size: usize,
        func: F,
    ) -> AudioSampleResult<()>
    where
        F: Fn(&[T], &[T]) -> Vec<T>,
    {
        if window_size == 0 || hop_size == 0 {
            return Err(AudioSampleError::InvalidParameter(
                "Window size and hop size must be greater than 0".to_string(),
            ));
        }

        match self {
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
                    func(window, &mut window_buffer);

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
                        func(window, &mut window_buffer);

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
                            result[channel_offset + i] =
                                result[channel_offset + i] / T::cast_from(count);
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
                        crate::AudioSampleError::InvalidParameter(format!(
                            "Array shape error: {}",
                            e
                        ))
                    })?;

                Ok(())
            }
        }
    }

    pub fn apply_to_all_channels<F>(&mut self, func: F)
    where
        F: Fn(T) -> T,
    {
        match self {
            AudioData::Mono(arr) => {
                arr.mapv_inplace(func);
            }
            AudioData::MultiChannel(arr) => {
                arr.mapv_inplace(func);
            }
        }
    }

    pub fn apply_to_channels<F>(&mut self, channels: &[usize], f: F)
    where
        F: Fn(T) -> T, // Function that returns a transformation for each channel
    {
        match self {
            AudioData::Mono(arr) => {
                // For mono, apply the transformation from channel 0
                arr.mapv_inplace(f);
            }
            AudioData::MultiChannel(arr) => {
                // Process each channel with vectorized operations
                for (ch, mut row) in arr.axis_iter_mut(ndarray::Axis(0)).enumerate() {
                    if !channels.contains(&ch) {
                        continue; // Skip channels not in the list
                    }
                    row.mapv_inplace(&f);
                }
            }
        }
    }

    pub fn convert_to<O: AudioSample>(&self) -> AudioData<O>
    where
        T: ConvertTo<O>,
    {
        match self {
            AudioData::Mono(arr) => {
                let converted = arr.mapv(|x| x.convert_to().unwrap_or_default());
                AudioData::Mono(converted.into())
            }
            AudioData::MultiChannel(arr) => {
                let converted = arr.mapv(|x| x.convert_to().unwrap_or_default());
                AudioData::MultiChannel(converted.into())
            }
        }
    }

    pub fn to_inverleaved_vec(self) -> Vec<T> {
        match self {
            AudioData::Mono(arr) => arr.to_vec(),
            AudioData::MultiChannel(arr) => {
                let (channels, samples_per_channel) = arr.dim();
                let mut interleaved = Vec::with_capacity(channels * samples_per_channel);

                for sample_idx in 0..samples_per_channel {
                    for ch in 0..channels {
                        interleaved.push(arr[[ch, sample_idx]]);
                    }
                }

                interleaved
            }
        }
    }

    pub fn as_interleaved_vec(&self) -> Vec<T> {
        match self {
            AudioData::Mono(arr) => arr.to_vec(),
            AudioData::MultiChannel(arr) => {
                let (channels, samples_per_channel) = arr.dim();
                let mut interleaved = Vec::with_capacity(channels * samples_per_channel);

                for sample_idx in 0..samples_per_channel {
                    for ch in 0..channels {
                        interleaved.push(arr[[ch, sample_idx]]);
                    }
                }

                interleaved
            }
        }
    }
}

// Conversion from ndarray views
impl<T: AudioSample> From<ArrayView1<'_, T>> for AudioData<T> {
    fn from(arr: ArrayView1<'_, T>) -> Self {
        AudioData::Mono(arr.to_owned())
    }
}

impl<T: AudioSample> From<ArrayViewMut1<'_, T>> for AudioData<T> {
    fn from(arr: ArrayViewMut1<'_, T>) -> Self {
        AudioData::Mono(arr.to_owned())
    }
}

impl<T: AudioSample> From<ArrayView2<'_, T>> for AudioData<T> {
    fn from(arr: ArrayView2<'_, T>) -> Self {
        AudioData::MultiChannel(arr.to_owned())
    }
}

impl<T: AudioSample> From<ArrayViewMut2<'_, T>> for AudioData<T> {
    fn from(arr: ArrayViewMut2<'_, T>) -> Self {
        AudioData::MultiChannel(arr.to_owned())
    }
}

// Indexing
impl<T: AudioSample> Index<usize> for AudioData<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            AudioData::Mono(arr) => &arr[index],
            AudioData::MultiChannel(arr) => {
                let (channels, samples_per_channel) = arr.dim();
                let total_samples = channels * samples_per_channel;
                if index >= total_samples {
                    panic!(
                        "Index {} out of bounds for total samples {}",
                        index, total_samples
                    );
                }
                let channel = index % channels;
                let sample_idx = index / channels;
                &arr[[channel, sample_idx]]
            }
        }
    }
}

// ---------------------
// OPS
// ---------------------

macro_rules! impl_audio_data_ops {
    ($(
        $trait:ident, $method:ident,
        $assign_trait:ident, $assign_method:ident,
        $op:tt,
        $mono_err:literal,
        $multi_err:literal,
        $mismatch_err:literal
    );+ $(;)?) => {
        $(
            // ================
            // Binary ops: Self
            // ================
            impl<T: AudioSample> std::ops::$trait<Self> for AudioData<T> {
                type Output = Self;

                fn $method(self, rhs: Self) -> Self::Output {
                    match (self, rhs) {
                        (AudioData::Mono(arr1), AudioData::Mono(arr2)) => {
                            if arr1.len() != arr2.len() {
                                panic!($mono_err);
                            }
                            AudioData::Mono(&arr1 $op &arr2)
                        }
                        (AudioData::MultiChannel(arr1), AudioData::MultiChannel(arr2)) => {
                            if arr1.dim() != arr2.dim() {
                                panic!($multi_err);
                            }
                            AudioData::MultiChannel(&arr1 $op &arr2)
                        }
                        _ => {
                            panic!($mismatch_err);
                        }
                    }
                }
            }

            // =================
            // Binary ops: Scalar
            // =================
            impl<T: AudioSample> std::ops::$trait<T> for AudioData<T> {
                type Output = Self;

                fn $method(self, rhs: T) -> Self::Output {
                    match self {
                        AudioData::Mono(arr) => AudioData::Mono(arr.mapv(|x| x $op rhs)),
                        AudioData::MultiChannel(arr) => AudioData::MultiChannel(arr.mapv(|x| x $op rhs)),
                    }
                }
            }

            // =====================
            // Assignment ops: Self
            // =====================
            impl<T: AudioSample> std::ops::$assign_trait<Self> for AudioData<T>
            where
                T: Clone,
            {
                fn $assign_method(&mut self, rhs: Self) {
                    *self = self.clone().$method(rhs);
                }
            }

            // ======================
            // Assignment ops: Scalar
            // ======================
            impl<T: AudioSample> std::ops::$assign_trait<T> for AudioData<T>
            where
                T: Clone,
            {
                fn $assign_method(&mut self, rhs: T) {
                    *self = self.clone().$method(rhs);
                }
            }
        )+
    };
}

impl_audio_data_ops!(
    Add, add, AddAssign, add_assign, +,
    "Cannot add mono audio with different lengths",
    "Cannot add multi-channel audio with different shapes",
    "Cannot add mono and multi-channel audio";
    Sub, sub, SubAssign, sub_assign, -,
    "Cannot subtract mono audio with different lengths",
    "Cannot subtract multi-channel audio with different shapes",
    "Cannot subtract mono and multi-channel audio";
    Mul, mul, MulAssign, mul_assign, *,
    "Cannot multiply mono audio with different lengths",
    "Cannot multiply multi-channel audio with different shapes",
    "Cannot multiply mono and multi-channel audio";
    Div, div, DivAssign, div_assign, /,
    "Cannot divide mono audio with different lengths",
    "Cannot divide multi-channel audio with different shapes",
    "Cannot divide mono and multi-channel audio";
);

// Negation
impl<T: AudioSample> Neg for AudioData<T>
where
    T: Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            AudioData::Mono(arr) => AudioData::Mono(arr.mapv(|x| -x)),
            AudioData::MultiChannel(arr) => AudioData::MultiChannel(arr.mapv(|x| -x)),
        }
    }
}

/// Represents audio samples in a format that can be used for various audio processing tasks.
/// This struct contains both the audio data and metadata like sample rate, channel information, etc.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioSamples<T: AudioSample> {
    pub data: AudioData<T>,
    pub sample_rate: u32,
    pub layout: ChannelLayout,
}

impl<T: AudioSample> Display for AudioSamples<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AudioSamples<{}>: {} channels, {} samples/channel, {} Hz, {:?} layout",
            std::any::type_name::<T>(),
            self.num_channels(),
            self.samples_per_channel(),
            self.sample_rate,
            self.layout
        )?;

        // display the first and last 5 samples of each channel
        match &self.data {
            AudioData::Mono(arr) => {
                let len = arr.len();
                let display_len = 5.min(len);
                write!(f, "\n[")?;
                for i in 0..display_len {
                    write!(f, "{:.4}", arr[i])?;
                    if i < display_len - 1 {
                        write!(f, ", ")?;
                    }
                }
                if len > display_len {
                    write!(f, ", ...")?;
                }

                // now print the last 5 samples
                if len > display_len {
                    write!(f, ", ")?;
                    for i in (len - display_len)..len {
                        write!(f, "{:.4}", arr[i])?;
                        if i < len - 1 {
                            write!(f, ", ")?;
                        }
                    }
                }

                write!(f, "]")
            }
            AudioData::MultiChannel(arr) => {
                let (channels, samples_per_channel) = arr.dim();
                for ch in 0..channels {
                    let len = samples_per_channel;
                    let display_len = 5.min(len);
                    write!(f, "\nChannel {} Samples: [", ch)?;
                    for i in 0..display_len {
                        write!(f, "{:.4}", arr[[ch, i]])?;
                        if i < display_len - 1 {
                            write!(f, ", ")?;
                        }
                    }
                    if len > display_len {
                        write!(f, ", ...")?;
                    }
                    // now print the last 5 samples
                    if len > display_len {
                        write!(f, ", ")?;
                        for i in (len - display_len)..len {
                            write!(f, "{:.4}", arr[[ch, i]])?;
                            if i < len - 1 {
                                write!(f, ", ")?;
                            }
                        }
                    }
                    write!(f, "]")?;
                }
                Ok(())
            }
        }
    }
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

    pub fn convert_to<O: AudioSample>(&self) -> AudioSamples<O>
    where
        T: ConvertTo<O>,
    {
        let data = self.data.convert_to();
        AudioSamples {
            data,
            sample_rate: self.sample_rate,
            layout: self.layout,
        }
    }

    /// Convert the AudioSamples struct into a vector of samples in interleaved format.
    pub fn to_interleaved_vec(&self) -> Vec<T> {
        self.data.as_interleaved_vec()
    }

    /// Creates a new mono AudioSamples with the given data and sample rate
    pub fn new_mono(data: Array1<T>, sample_rate: u32) -> Self {
        Self {
            data: AudioData::Mono(data),
            sample_rate,
            layout: ChannelLayout::NonInterleaved,
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
            layout: ChannelLayout::NonInterleaved,
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

    /// Creates a new mono AudioSamples filled with ones
    pub fn ones_mono(length: usize, sample_rate: u32) -> Self {
        Self {
            data: AudioData::Mono(Array1::ones(length)),
            sample_rate,
            layout: ChannelLayout::NonInterleaved,
        }
    }

    /// Creates a new multi-channel AudioSamples filled with ones
    pub fn ones_multi(channels: usize, length: usize, sample_rate: u32) -> Self {
        Self {
            data: AudioData::MultiChannel(Array2::ones((channels, length))),
            sample_rate,
            layout: ChannelLayout::Interleaved,
        }
    }

    /// Creates a new mono AudioSamples filled with the specified value
    pub fn uniform_mono(length: usize, sample_rate: u32, value: T) -> Self {
        Self {
            data: AudioData::Mono(Array1::from_elem(length, value)),
            sample_rate,
            layout: ChannelLayout::NonInterleaved,
        }
    }

    /// Creates a new multi-channel AudioSamples filled with the specified value
    pub fn uniform_multi(channels: usize, length: usize, sample_rate: u32, value: T) -> Self {
        Self {
            data: AudioData::MultiChannel(Array2::from_elem((channels, length), value)),
            sample_rate,
            layout: ChannelLayout::Interleaved,
        }
    }

    /// Returns a mutable reference to the channel layout
    pub fn layout_mut(&mut self) -> &mut ChannelLayout {
        &mut self.layout
    }

    /// Sets the channel layout
    pub fn set_layout(&mut self, layout: ChannelLayout) {
        self.layout = layout;
    }

    /// Returns the sample rate in Hz
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the number of channels
    pub fn num_channels(&self) -> usize {
        self.data.num_channels()
    }

    /// Returns the number of samples per channel
    pub fn samples_per_channel(&self) -> usize {
        self.data.samples_per_channel()
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
    pub fn bytes_per_sample(&self) -> usize {
        self.data.bytes_per_sample()
    }

    /// Returns the sample type as a string
    pub fn sample_type() -> &'static str {
        std::any::type_name::<T>()
    }

    /// Returns the channel layout
    pub const fn layout(&self) -> ChannelLayout {
        self.layout
    }

    /// Returns true if this is mono audio
    pub fn is_mono(&self) -> bool {
        self.data.is_mono()
    }

    /// Returns true if this is multi-channel audio
    pub fn is_multi_channel(&self) -> bool {
        self.data.is_multi_channel()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Returns a reference to the underlying mono array, if this is mono audio
    pub fn as_mono(&self) -> Option<&Array1<T>> {
        self.data.as_mono()
    }

    /// Returns a reference to the underlying mono array, does this without checking
    pub unsafe fn as_mono_unchecked(&self) -> &Array1<T> {
        unsafe { self.data.as_mono_unchecked() }
    }

    /// Returns a reference to the underlying multi-channel array, if this is multi-channel audio
    pub fn as_multi_channel(&self) -> Option<&Array2<T>> {
        self.data.as_multi_channel()
    }

    /// Returns a reference to the underlying multi-channel array, does this without checking
    pub unsafe fn as_multi_channel_unchecked(&self) -> &Array2<T> {
        unsafe { self.data.as_multi_channel_unchecked() }
    }

    /// Returns a mutable reference to the underlying mono array, if this is mono audio
    pub fn as_mono_mut(&mut self) -> Option<&mut Array1<T>> {
        self.data.as_mono_mut()
    }

    /// Returns a mutable reference to the underlying multi-channel array, if this is multi-channel audio
    pub fn as_multi_channel_mut(&mut self) -> Option<&mut Array2<T>> {
        self.data.as_multi_channel_mut()
    }

    /// Returns a mutable reference to the underlying mono array, does this without checking
    pub unsafe fn as_mono_mut_unchecked(&mut self) -> &mut Array1<T> {
        unsafe { self.data.as_mono_mut_unchecked() }
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
    pub fn apply<F>(&mut self, f: F)
    where
        F: Fn(T) -> T,
    {
        self.data.apply(f)
    }

    pub fn apply_to_channels<F>(&mut self, channels: &[usize], f: F)
    where
        F: Fn(T) -> T + Copy,
    {
        self.data.apply_to_channels(channels, f)
    }

    /// Maps a function to each sample and returns a new AudioSamples instance.
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
    pub fn map<F>(&self, f: F) -> AudioSampleResult<Self>
    where
        F: Fn(T) -> T,
    {
        let new_data = self.data.mapv(f);
        Ok(Self {
            data: new_data,
            sample_rate: self.sample_rate,
            layout: self.layout,
        })
    }

    pub fn map_into<O: AudioSample, F>(&self, f: F) -> AudioSampleResult<AudioSamples<O>>
    where
        F: Fn(T) -> O,
    {
        let new_data: AudioData<O> = self.data.mapv(f);
        Ok(AudioSamples {
            data: new_data,
            sample_rate: self.sample_rate,
            layout: self.layout,
        })
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
    /// audio.apply_with_index(|index, sample| {
    ///     let fade_samples = 44100; // 1 second fade at 44.1kHz
    ///     let gain = if index < fade_samples {
    ///         index as f32 / fade_samples as f32
    ///     } else {
    ///         1.0
    ///     };
    ///     sample * gain
    /// })?;
    /// ```
    pub fn apply_with_index<F>(&mut self, f: F)
    where
        F: Fn(usize, T) -> T,
    {
        self.data.apply_with_index(f)
    }

    /// Creates a processing builder for fluent operation chaining.
    ///
    /// This method provides a builder pattern for chaining multiple audio
    /// processing operations together and applying them all at once.
    ///
    /// # Example
    /// ```rust,ignore
    /// use audio_samples::{AudioSamples, operations::types::NormalizationMethod};
    /// use ndarray::array;
    ///
    /// let mut audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0], 44100);
    /// audio.processing()
    ///     .normalize(-1.0, 1.0, NormalizationMethod::Peak)
    ///     .scale(0.5)
    ///     .apply()?;
    /// ```
    pub fn processing(&mut self) -> ProcessingBuilder<'_, T>
    where
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
        for<'a> AudioSamples<T>: AudioTypeConversion<T>,
    {
        ProcessingBuilder::new(self)
    }

    // ========================
    // Chainable result methods for improved error handling ergonomics
    // ========================

    /// Convert to another audio sample type with chainable result.
    ///
    /// This provides a more ergonomic alternative to `as_type()` for method chaining.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, ChainableResult};
    /// # use ndarray::array;
    /// let audio = AudioSamples::new_mono(array![0.5f32, -0.3, 0.8], 44100);
    ///
    /// let result = audio.try_convert::<i16>()
    ///     .map(|converted| converted.channels())
    ///     .inspect(|channels| println!("Converted to {} channels", channels))
    ///     .into_result();
    /// ```
    pub fn try_convert<U: AudioSample>(self) -> ChainableResult<AudioSamples<U>>
    where
        T: ConvertTo<U>,
    {
        ChainableResult::ok(self.convert_to())
    }

    /// Apply a function to all samples with chainable result.
    ///
    /// Provides a chainable alternative to fallible sample processing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, ChainableResult};
    /// # use ndarray::array;
    /// let mut audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0], 44100);
    ///
    /// let result = audio.try_apply(|sample| sample * 0.5)
    ///     .and_then(|_| audio.try_apply(|sample| sample.clamp(-1.0, 1.0)))
    ///     .log_on_error("Sample processing failed")
    ///     .into_result();
    /// ```
    pub fn try_apply<F>(&mut self, f: F) -> ChainableResult<()>
    where
        F: Fn(T) -> T,
    {
        self.apply(f);
        ChainableResult::ok(())
    }

    /// Validate chainable result for audio operations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, ChainableResult};
    /// # use ndarray::array;
    /// let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0], 44100);
    ///
    /// let result = audio.try_validate()
    ///     .map(|validated| validated.samples_per_channel())
    ///     .inspect(|count| println!("Validated {} samples", count))
    ///     .into_result();
    /// ```
    pub fn try_validate(self) -> ChainableResult<Self> {
        if self.samples_per_channel() == 0 {
            ChainableResult::err(AudioSampleError::InvalidInput {
                msg: "Audio has zero samples".to_string(),
            })
        } else {
            ChainableResult::ok(self)
        }
    }

    /// Clone with chainable result.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, ChainableResult};
    /// # use ndarray::array;
    /// let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0], 44100);
    ///
    /// let result = audio.try_clone()
    ///     .map(|cloned| cloned.samples_per_channel())
    ///     .inspect(|count| println!("Cloned {} samples", count))
    ///     .into_result();
    /// ```
    pub fn try_clone(&self) -> ChainableResult<Self> {
        ChainableResult::ok(self.clone())
    }

    // ========================
    // Sample and channel slicing methods for Python bindings compatibility
    // ========================

    /// Slice audio by sample range, keeping all channels.
    ///
    /// Creates a new AudioSamples instance containing samples in the specified range.
    ///
    /// # Arguments
    /// * `sample_range` - Range of samples to extract (e.g., 100..200)
    ///
    /// # Returns
    /// A new AudioSamples instance with the sliced samples
    ///
    /// # Errors
    /// Returns an error if the range is out of bounds.
    pub fn slice_samples<R>(&self, sample_range: R) -> AudioSampleResult<Self>
    where
        R: RangeBounds<usize> + Clone,
    {
        let samples_per_channel = self.samples_per_channel();

        let start = match sample_range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n + 1,
            Bound::Unbounded => 0,
        };

        let end = match sample_range.end_bound() {
            Bound::Included(&n) => n + 1,
            Bound::Excluded(&n) => n,
            Bound::Unbounded => samples_per_channel,
        };

        if start >= samples_per_channel || end > samples_per_channel || start >= end {
            return Err(AudioSampleError::InvalidParameter(format!(
                "Sample range {}..{} out of bounds for {} samples",
                start, end, samples_per_channel
            )));
        }

        match &self.data {
            AudioData::Mono(arr) => {
                let sliced = arr.slice(ndarray::s![start..end]).to_owned();
                Ok(AudioSamples::new_mono(sliced.into(), self.sample_rate()))
            }
            AudioData::MultiChannel(arr) => {
                let sliced = arr.slice(ndarray::s![.., start..end]).to_owned();
                Ok(AudioSamples::new_multi_channel(
                    sliced.into(),
                    self.sample_rate(),
                ))
            }
        }
    }

    /// Slice audio by channel range, keeping all samples.
    ///
    /// Creates a new AudioSamples instance containing only the specified channels.
    ///
    /// # Arguments
    /// * `channel_range` - Range of channels to extract (e.g., 0..2 for stereo)
    ///
    /// # Returns
    /// A new AudioSamples instance with the sliced channels
    ///
    /// # Errors
    /// Returns an error if the range is out of bounds.
    pub fn slice_channels<R>(&self, channel_range: R) -> AudioSampleResult<Self>
    where
        R: RangeBounds<usize> + Clone,
    {
        let num_channels = self.num_channels();

        let start = match channel_range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n + 1,
            Bound::Unbounded => 0,
        };

        let end = match channel_range.end_bound() {
            Bound::Included(&n) => n + 1,
            Bound::Excluded(&n) => n,
            Bound::Unbounded => num_channels,
        };

        if start >= num_channels || end > num_channels || start >= end {
            return Err(AudioSampleError::InvalidParameter(format!(
                "Channel range {}..{} out of bounds for {} channels",
                start, end, num_channels
            )));
        }

        match &self.data {
            AudioData::Mono(arr) => {
                if start != 0 || end != 1 {
                    return Err(AudioSampleError::InvalidParameter(format!(
                        "Channel range {}..{} invalid for mono audio (only 0..1 allowed)",
                        start, end
                    )));
                }
                Ok(AudioSamples::new_mono(
                    arr.clone().into(),
                    self.sample_rate(),
                ))
            }
            AudioData::MultiChannel(arr) => {
                let sliced = arr.slice(ndarray::s![start..end, ..]).to_owned();
                if end - start == 1 {
                    // Single channel result - convert to mono
                    let mono_data = sliced.index_axis(ndarray::Axis(0), 0).to_owned();
                    Ok(AudioSamples::new_mono(mono_data.into(), self.sample_rate()))
                } else {
                    // Multi-channel result
                    Ok(AudioSamples::new_multi_channel(
                        sliced.into(),
                        self.sample_rate(),
                    ))
                }
            }
        }
    }

    /// Slice audio by both channel and sample ranges.
    ///
    /// Creates a new AudioSamples instance containing the intersection of
    /// the specified channel and sample ranges.
    ///
    /// # Arguments
    /// * `channel_range` - Range of channels to extract
    /// * `sample_range` - Range of samples to extract
    ///
    /// # Returns
    /// A new AudioSamples instance with the sliced data
    ///
    /// # Errors
    /// Returns an error if either range is out of bounds.
    pub fn slice_both<CR, SR>(&self, channel_range: CR, sample_range: SR) -> AudioSampleResult<Self>
    where
        CR: RangeBounds<usize> + Clone,
        SR: RangeBounds<usize> + Clone,
    {
        let num_channels = self.num_channels();
        let samples_per_channel = self.samples_per_channel();

        let ch_start = match channel_range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n + 1,
            Bound::Unbounded => 0,
        };

        let ch_end = match channel_range.end_bound() {
            Bound::Included(&n) => n + 1,
            Bound::Excluded(&n) => n,
            Bound::Unbounded => num_channels,
        };

        let s_start = match sample_range.start_bound() {
            Bound::Included(&n) => n,
            Bound::Excluded(&n) => n + 1,
            Bound::Unbounded => 0,
        };

        let s_end = match sample_range.end_bound() {
            Bound::Included(&n) => n + 1,
            Bound::Excluded(&n) => n,
            Bound::Unbounded => samples_per_channel,
        };

        if ch_start >= num_channels || ch_end > num_channels || ch_start >= ch_end {
            return Err(AudioSampleError::InvalidParameter(format!(
                "Channel range {}..{} out of bounds for {} channels",
                ch_start, ch_end, num_channels
            )));
        }

        if s_start >= samples_per_channel || s_end > samples_per_channel || s_start >= s_end {
            return Err(AudioSampleError::InvalidParameter(format!(
                "Sample range {}..{} out of bounds for {} samples",
                s_start, s_end, samples_per_channel
            )));
        }

        match &self.data {
            AudioData::Mono(arr) => {
                if ch_start != 0 || ch_end != 1 {
                    return Err(AudioSampleError::InvalidParameter(format!(
                        "Channel range {}..{} invalid for mono audio (only 0..1 allowed)",
                        ch_start, ch_end
                    )));
                }
                let sliced = arr.slice(ndarray::s![s_start..s_end]).to_owned();
                Ok(AudioSamples::new_mono(sliced.into(), self.sample_rate()))
            }
            AudioData::MultiChannel(arr) => {
                let sliced = arr
                    .slice(ndarray::s![ch_start..ch_end, s_start..s_end])
                    .to_owned();
                if ch_end - ch_start == 1 {
                    // Single channel result - convert to mono
                    let mono_data = sliced.index_axis(ndarray::Axis(0), 0).to_owned();
                    Ok(AudioSamples::new_mono(mono_data.into(), self.sample_rate()))
                } else {
                    // Multi-channel result
                    Ok(AudioSamples::new_multi_channel(
                        sliced.into(),
                        self.sample_rate(),
                    ))
                }
            }
        }
    }

    pub fn as_bytes(&self) -> Vec<u8> {
        self.data.as_bytes()
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
                panic!(
                    "Cannot use single index on multi-channel audio. Use (channel, sample) indexing instead."
                );
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
                panic!(
                    "Cannot use single index on multi-channel audio. Use (channel, sample) indexing instead."
                );
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
                    panic!(
                        "Channel index {} out of bounds for mono audio (only channel 0 exists)",
                        channel
                    );
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
                    panic!(
                        "Channel index {} out of bounds for mono audio (only channel 0 exists)",
                        channel
                    );
                }
                &mut arr[sample]
            }
            AudioData::MultiChannel(arr) => &mut arr[(channel, sample)],
        }
    }
}

impl<T: AudioSample> IntoIterator for AudioSamples<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    /// Consumes the AudioSamples and returns an iterator over the samples in interleaved order.
    ///
    /// For mono audio, this is simply the samples in order.
    /// For multi-channel audio, this interleaves samples from each channel.
    ///
    /// # Examples
    /// ```rust
    /// use audio_samples::AudioSamples;
    /// use ndarray::array;
    ///
    /// // Mono audio
    /// let mono_data = array![1.0f32, 2.0, 3.0];
    /// let mono_audio = AudioSamples::new_mono(mono_data, 44100);
    /// let mono_samples: Vec<f32> = mono_audio.into_iter().collect();
    /// assert_eq!(mono_samples, vec![1.0, 2.0, 3.0]);
    ///
    /// // Multi-channel audio
    /// let stereo_data = array![[1.0f32, 2.0], [3.0, 4.0]];
    /// let stereo_audio = AudioSamples::new_multi_channel(stereo_data, 44100);
    /// let stereo_samples: Vec<f32> = stereo_audio.into_iter().collect();
    /// assert_eq!(stereo_samples, vec![1.0, 3.0, 2.0, 4.0]); // Interleaved
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        self.to_interleaved_vec().into_iter()
    }
}

macro_rules! impl_audio_samples_ops {
    ($(
        $trait:ident, $method:ident,
        $assign_trait:ident, $assign_method:ident,
        $op:tt, $assign_op:tt
    );+ $(;)?) => {
        $(
            // Binary op with another AudioSamples<T>
            impl<T: AudioSample> std::ops::$trait<Self> for AudioSamples<T> {
                type Output = Self;

                fn $method(self, rhs: Self) -> Self::Output {
                    if self.sample_rate != rhs.sample_rate {
                        panic!(
                            concat!(
                                "Cannot ", stringify!($method),
                                " audio with different sample rates: {} vs {}"
                            ),
                            self.sample_rate, rhs.sample_rate
                        );
                    }
                    Self {
                        data: self.data $op rhs.data,
                        sample_rate: self.sample_rate,
                        layout: self.layout,
                    }
                }
            }

            // Binary op with scalar T
            impl<T: AudioSample> std::ops::$trait<T> for AudioSamples<T> {
                type Output = Self;

                fn $method(self, rhs: T) -> Self::Output {
                    Self {
                        data: self.data $op rhs,
                        sample_rate: self.sample_rate,
                        layout: self.layout,
                    }
                }
            }

            // Assign op with another AudioSamples<T>
            impl<T: AudioSample> std::ops::$assign_trait<Self> for AudioSamples<T> {
                fn $assign_method(&mut self, rhs: Self) {
                    if self.sample_rate != rhs.sample_rate {
                        panic!(
                            concat!(
                                "Cannot ", stringify!($assign_method),
                                " audio with different sample rates: {} vs {}"
                            ),
                            self.sample_rate, rhs.sample_rate
                        );
                    }
                    self.data $assign_op rhs.data;
                }
            }

            // Assign op with scalar T
            impl<T: AudioSample> std::ops::$assign_trait<T> for AudioSamples<T> {
                fn $assign_method(&mut self, rhs: T) {
                    self.data $assign_op rhs;
                }
            }
        )+
    };
}

impl_audio_samples_ops!(
    Add, add, AddAssign, add_assign, +, +=;
    Sub, sub, SubAssign, sub_assign, -, -=;
    Mul, mul, MulAssign, mul_assign, *, *=;
    Div, div, DivAssign, div_assign, /, /=;
);

// Negation
impl<T: AudioSample> Neg for AudioSamples<T>
where
    T: Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            data: -self.data,
            sample_rate: self.sample_rate,
            layout: self.layout,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayBase, array};

    #[test]
    fn test_new_mono_audio_samples() {
        let data: ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 1]>> =
            array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio: AudioSamples<f32> = AudioSamples::new_mono(data.clone().into(), 44100);

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
        let audio = AudioSamples::new_multi_channel(data.clone().into(), 48000);

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
    fn test_apply_simple() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut audio: AudioSamples<f32> = AudioSamples::new_mono(data.into(), 44100);

        // Apply a simple scaling
        audio.apply(|sample| sample * 2.0);
        let mono = audio.as_mono().unwrap();

        let expected = array![2.0f32, 4.0, 6.0, 8.0, 10.0];
        assert_eq!(mono, &expected);
    }

    #[test]
    fn test_apply_channels() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio: AudioSamples<f32> = AudioSamples::new_multi_channel(data.into(), 44100);

        {
            // Mutable borrow lives only within this block
            audio.apply_to_channels(&[0, 1], |sample| sample * sample);
        } // Mutable borrow ends here

        let expected = array![[1.0, 4.0], [9.0, 16.0]];
        let multichannel = audio.as_multi_channel().unwrap();

        assert_eq!(multichannel, &expected);
    }

    #[test]
    fn test_map_functional() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioSamples::new_mono(data.into(), 44100);

        // Create a new audio instance with transformed samples
        let new_audio = audio.map(|sample| sample * 0.5).unwrap();

        let expected = array![0.5f32, 1.0, 1.5, 2.0, 2.5];
        assert_eq!(new_audio.as_mono().unwrap(), &expected);

        // Original should be unchanged
        let original_expected = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(audio.as_mono().unwrap(), &original_expected);
    }

    #[test]
    fn test_apply_indexed() {
        let data = array![1.0f32, 1.0, 1.0, 1.0, 1.0];
        let mut audio = AudioSamples::new_mono(data.into(), 44100);

        // Apply index-based transformation
        audio.apply_with_index(|index, sample| sample * (index + 1) as f32);
        let expected = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(audio.as_mono().unwrap(), &expected);
    }

    // #[test]
    // fn test_apply_windowed() {
    //     let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    //     let mut audio: AudioSamples<'_, f32> = AudioSamples::new_mono(data.into(), 44100);

    //     // Apply windowed transformation (simple identity for testing)
    //     audio
    //         .apply_windowed(3, 2, |window| {
    //             window.to_vec() // Just return the window unchanged
    //         })
    //         .unwrap();

    //     // The length should be preserved or extended based on windowing
    //     assert!(audio.samples_per_channel() >= 6);
    // }

    // #[test]
    // fn test_apply_error_handling() {
    //     let data = array![1.0f32, 2.0, 3.0];
    //     let mut audio = AudioSamples::new_mono(data.into(), 44100);

    //     // Test error handling for invalid window size
    //     let result = audio.apply_windowed(0, 1, |window| window.to_vec());
    //     assert!(result.is_err());

    //     // Test error handling for invalid hop size
    //     let result = audio.apply_windowed(2, 0, |window| window.to_vec());
    //     assert!(result.is_err());
    // }

    // ========================
    // Indexing and Slicing Tests
    // ========================

    #[test]
    fn test_index_mono_single() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioSamples::new_mono(data.into(), 44100);

        assert_eq!(audio[0], 1.0);
        assert_eq!(audio[2], 3.0);
        assert_eq!(audio[4], 5.0);
    }

    #[test]
    #[should_panic(expected = "Cannot use single index on multi-channel audio")]
    fn test_index_mono_single_on_multichannel_panics() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let audio = AudioSamples::new_multi_channel(data.into(), 44100);

        let _ = audio[0]; // Should panic
    }

    #[test]
    fn test_index_tuple() {
        // Test mono with tuple indexing
        let mono_data = array![1.0f32, 2.0, 3.0, 4.0];
        let mono_audio = AudioSamples::new_mono(mono_data.into(), 44100);

        assert_eq!(mono_audio[(0, 1)], 2.0);
        assert_eq!(mono_audio[(0, 3)], 4.0);

        // Test multi-channel with tuple indexing
        let stereo_data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let stereo_audio = AudioSamples::new_multi_channel(stereo_data.into(), 44100);

        assert_eq!(stereo_audio[(0, 0)], 1.0);
        assert_eq!(stereo_audio[(0, 2)], 3.0);
        assert_eq!(stereo_audio[(1, 0)], 4.0);
        assert_eq!(stereo_audio[(1, 2)], 6.0);
    }

    #[test]
    #[should_panic(expected = "Channel index 1 out of bounds for mono audio")]
    fn test_index_tuple_invalid_channel_mono() {
        let data = array![1.0f32, 2.0, 3.0];
        let audio = AudioSamples::new_mono(data.into(), 44100);

        let _ = audio[(1, 0)]; // Should panic - mono only has channel 0
    }

    #[test]
    fn test_index_mut_mono() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut audio = AudioSamples::new_mono(data.into(), 44100);

        audio[2] = 10.0;
        assert_eq!(audio[2], 10.0);
        assert_eq!(
            audio.as_mono().unwrap(),
            &array![1.0f32, 2.0, 10.0, 4.0, 5.0]
        );
    }

    #[test]
    fn test_index_mut_tuple() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio = AudioSamples::new_multi_channel(data.into(), 44100);

        audio[(1, 0)] = 10.0;
        assert_eq!(audio[(1, 0)], 10.0);

        let expected = array![[1.0f32, 2.0], [10.0, 4.0]];
        assert_eq!(audio.as_multi_channel().unwrap(), &expected);
    }
}
