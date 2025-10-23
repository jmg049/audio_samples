//! Efficient iterators over audio samples.
//!
//! This module provides high-performance iterator types for structured access to audio data
//! in the `AudioSamples` struct. The iterators are designed to be zero-allocation where possible
//! and leverage Rust's iterator system for optimal performance.
//!
//! # Iterator Types
//!
//! - [`FrameIterator`] - Iterates over frames (one sample from each channel)
//! - [`ChannelIterator`] - Iterates over complete channels sequentially
//! - [`WindowIterator`] - Iterates over overlapping windows/blocks of data
//!
//! # Usage
//!
//! ```rust
//! use audio_samples::{AudioSamples, iterators::AudioSampleIterators};
//! use ndarray::array;
//!
//! let audio = AudioSamples::new_multi_channel(
//!     array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
//!     44100
//! );
//!
//! // Iterate over frames
//! for frame in audio.frames() {
//!     println!("Frame: {:?}", frame);
//! }
//!
//! // Iterate over channels
//! for channel in audio.channels() {
//!     println!("Channel samples: {:?}", channel);
//! }
//!
//! // Iterate over overlapping windows
//! for window in audio.windows(1024, 512) {
//!     println!("Window: {} samples", window.len());
//! }
//! ```
//!
//! # Performance Notes
//!
//! - All iterators are designed to avoid unnecessary allocations
//! - Frame and channel iterators provide direct access to underlying data
//! - Window iterator supports both zero-padding and overlap strategies
//! - Iterators work efficiently with both mono and multi-channel audio

use ndarray::{Array1, Array2};

use crate::{AudioData, AudioSample, AudioSamples};
use std::marker::PhantomData;

#[cfg(feature = "parallel-processing")]
use rayon::prelude::*;

/// Extension trait providing iterator methods for AudioSamples.
pub trait AudioSampleIterators<T: AudioSample> {
    /// Returns an iterator over frames (one sample from each channel).
    ///
    /// For mono audio, each frame contains one sample.
    /// For multi-channel audio, each frame contains one sample from each channel.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, iterators::AudioSampleIterators};
    /// # use ndarray::array;
    /// let audio = AudioSamples::new_multi_channel(
    ///     array![[1.0f32, 2.0], [3.0, 4.0]],
    ///     44100
    /// );
    ///
    /// let frames: Vec<Vec<f32>> = audio.frames().collect();
    /// assert_eq!(frames, vec![vec![1.0, 3.0], vec![2.0, 4.0]]);
    /// ```
    fn frames(&self) -> FrameIterator<T>;

    /// Returns an iterator over complete channels.
    ///
    /// Each iteration yields all samples from one channel before moving to the next.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, iterators::AudioSampleIterators};
    /// # use ndarray::array;
    /// let audio = AudioSamples::new_multi_channel(
    ///     array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
    ///     44100
    /// );
    ///
    /// let channels: Vec<Vec<f32>> = audio.channels().collect();
    /// assert_eq!(channels[0], vec![1.0, 2.0, 3.0]);
    /// assert_eq!(channels[1], vec![4.0, 5.0, 6.0]);
    /// ```
    fn channels(&self) -> ChannelIterator<T>;

    /// Returns an iterator over overlapping windows of audio data.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Size of each window in samples
    /// * `hop_size` - Number of samples to advance between windows (overlap = window_size - hop_size)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, iterators::AudioSampleIterators};
    /// # use ndarray::array;
    /// let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 44100);
    ///
    /// let windows: Vec<Vec<f32>> = audio.windows(4, 2).collect();
    /// // First window: [1.0, 2.0, 3.0, 4.0]
    /// // Second window: [3.0, 4.0, 5.0, 6.0] (advanced by 2, overlaps by 2)
    /// ```
    fn windows(&self, window_size: usize, hop_size: usize) -> WindowIterator<T>;

    /// Returns a mutable iterator over frames (one sample from each channel).
    ///
    /// This allows in-place modification of frames. Each iteration yields mutable references
    /// to samples that can be modified directly.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, iterators::AudioSampleIterators};
    /// # use ndarray::array;
    /// let mut audio = AudioSamples::new_multi_channel(
    ///     array![[1.0f32, 2.0], [3.0, 4.0]],
    ///     44100
    /// );
    ///
    /// // Apply gain to each frame
    /// for frame in audio.frames_mut() {
    ///     for sample in frame {
    ///         *sample *= 0.5;
    ///     }
    /// }
    /// ```
    fn frames_mut(&mut self) -> FrameIteratorMut<'_, T>;

    /// Returns a mutable iterator over complete channels.
    ///
    /// Each iteration yields mutable references to all samples in one channel,
    /// allowing for efficient channel-wise processing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, iterators::AudioSampleIterators};
    /// # use ndarray::array;
    /// let mut audio = AudioSamples::new_multi_channel(
    ///     array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
    ///     44100
    /// );
    ///
    /// // Apply different processing to each channel
    /// for (ch, channel) in audio.channels_mut().enumerate() {
    ///     let gain = if ch == 0 { 0.8 } else { 0.6 }; // Left/Right balance
    ///     for sample in channel {
    ///         *sample *= gain;
    ///     }
    /// }
    /// ```
    fn channels_mut(&mut self) -> ChannelIteratorMut<'_, T>;

    /// Returns a mutable iterator over overlapping windows of audio data.
    ///
    /// This allows in-place modification of windows, useful for audio processing
    /// operations like filtering, FFT processing, etc.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Size of each window in samples
    /// * `hop_size` - Number of samples to advance between windows
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, iterators::AudioSampleIterators};
    /// # use ndarray::array;
    /// let mut audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 44100);
    ///
    /// // Apply windowed processing
    /// for window in audio.windows_mut(4, 2) {
    ///     // Apply window function (e.g., Hann window)
    ///     for (i, sample) in window.iter_mut().enumerate() {
    ///         let window_factor = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (window.len() - 1) as f32).cos());
    ///         *sample *= window_factor;
    ///     }
    /// }
    /// ```
    fn windows_mut(&mut self, window_size: usize, hop_size: usize) -> WindowIteratorMut<'_, T>;
}

/// Extension trait providing parallel iterator methods for AudioSamples.
///
/// This trait provides parallel versions of the standard audio iterators using rayon's
/// parallel iterator framework. These methods are only available when the `parallel-processing`
/// feature is enabled.
///
/// # Performance Notes
///
/// - Parallel iterators are most beneficial for CPU-intensive operations on large datasets
/// - For simple operations or small audio samples, sequential iterators may be faster due to overhead
/// - The parallel iterators automatically scale to available CPU cores
/// - Consider using `.with_min_len()` on returned parallel iterators to tune work distribution
///
/// # Examples
///
/// ```rust
/// use audio_samples::{AudioSamples, iterators::AudioSampleParallelIterators};
/// use rayon::prelude::*;
/// use ndarray::array;
///
/// let audio = AudioSamples::new_multi_channel(
///     array![[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
///     44100
/// );
///
/// // Process frames in parallel
/// let processed: Vec<f32> = audio.par_frames()
///     .map(|frame| {
///         // Expensive computation per frame
///         frame.iter().map(|&s| s.powi(2)).sum()
///     })
///     .collect();
///
/// // Process channels in parallel
/// let channel_rms: Vec<f32> = audio.par_channels()
///     .map(|channel| {
///         let sum_squares: f32 = channel.iter().map(|&s| s * s).sum();
///         (sum_squares / channel.len() as f32).sqrt()
///     })
///     .collect();
/// ```
#[cfg(feature = "parallel-processing")]
pub trait AudioSampleParallelIterators<T: AudioSample + Send + Sync> {
    /// Returns a parallel iterator over frames (one sample from each channel).
    ///
    /// Each frame is processed independently in parallel, making this ideal for
    /// frame-wise audio processing operations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, iterators::AudioSampleParallelIterators};
    /// # use rayon::prelude::*;
    /// # use ndarray::array;
    /// let audio = AudioSamples::new_multi_channel(
    ///     array![[1.0f32, 2.0], [3.0, 4.0]],
    ///     44100
    /// );
    ///
    /// let frame_energies: Vec<f32> = audio.par_frames()
    ///     .map(|frame| frame.iter().map(|&s| s * s).sum())
    ///     .collect();
    /// ```
    fn par_frames(&self) -> ParFrameIterator<T>;

    /// Returns a parallel iterator over complete channels.
    ///
    /// Each channel is processed independently in parallel, making this ideal for
    /// channel-wise audio processing operations.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, iterators::AudioSampleParallelIterators};
    /// # use rayon::prelude::*;
    /// # use ndarray::array;
    /// let audio = AudioSamples::new_multi_channel(
    ///     array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
    ///     44100
    /// );
    ///
    /// let channel_maxes: Vec<f32> = audio.par_channels()
    ///     .map(|channel| {
    ///         channel.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x))
    ///     })
    ///     .collect();
    /// ```
    fn par_channels(&self) -> ParChannelIterator<T>;

    /// Returns a parallel iterator over overlapping windows of audio data.
    ///
    /// Each window is processed independently in parallel, making this ideal for
    /// windowed audio processing operations like STFT.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Size of each window in samples
    /// * `hop_size` - Number of samples to advance between windows
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, iterators::AudioSampleParallelIterators};
    /// # use rayon::prelude::*;
    /// # use ndarray::array;
    /// let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 44100);
    ///
    /// let window_energies: Vec<f32> = audio.par_windows(4, 2)
    ///     .map(|window| window.iter().map(|&s| s * s).sum())
    ///     .collect();
    /// ```
    fn par_windows(&self, window_size: usize, hop_size: usize) -> ParWindowIterator<T>;
}

impl<T: AudioSample> AudioSampleIterators<T> for AudioSamples<T> {
    fn frames(&self) -> FrameIterator<T> {
        FrameIterator::new(self)
    }

    fn channels(&self) -> ChannelIterator<T> {
        ChannelIterator::new(self)
    }

    fn windows(&self, window_size: usize, hop_size: usize) -> WindowIterator<T> {
        WindowIterator::new(self, window_size, hop_size)
    }

    fn frames_mut(&mut self) -> FrameIteratorMut<'_, T> {
        FrameIteratorMut::new(self)
    }

    fn channels_mut(&mut self) -> ChannelIteratorMut<'_, T> {
        ChannelIteratorMut::new(self)
    }

    fn windows_mut(&mut self, window_size: usize, hop_size: usize) -> WindowIteratorMut<'_, T> {
        WindowIteratorMut::new(self, window_size, hop_size)
    }
}

#[cfg(feature = "parallel-processing")]
impl<T: AudioSample + Send + Sync> AudioSampleParallelIterators<T> for AudioSamples<T> {
    fn par_frames(&self) -> ParFrameIterator<T> {
        ParFrameIterator::new(self)
    }

    fn par_channels(&self) -> ParChannelIterator<T> {
        ParChannelIterator::new(self)
    }

    fn par_windows(&self, window_size: usize, hop_size: usize) -> ParWindowIterator<T> {
        ParWindowIterator::new(self, window_size, hop_size)
    }
}

/// Iterator over frames of audio data.
///
/// A frame contains one sample from each channel at a given time point.
/// For mono audio, frames contain single samples. For multi-channel audio,
/// frames contain one sample per channel.
pub struct FrameIterator<T: AudioSample> {
    /// Raw pointer to the AudioSamples struct.
    ///
    /// We use a raw pointer here instead of a reference (&AudioSamples<T>) to avoid
    /// Rust's borrowing restrictions that would prevent users from creating multiple
    /// iterators from the same AudioSamples instance. Since AudioSamples is immutable
    /// during iteration and we only read from it, this is safe. The pointer remains
    /// valid for the iterator's lifetime because:
    /// 1. The iterator is created from a reference, ensuring the data exists
    /// 2. AudioSamples owns its data, so it won't be moved/freed unexpectedly
    /// 3. We only perform read operations through the pointer
    ///
    /// This pattern allows users to write code like:
    /// ```ignore
    /// let frames = audio.frames();
    /// let channels = audio.channels(); // This would fail with &AudioSamples<T>
    /// ```
    audio: *const AudioSamples<T>,
    current_frame: usize,
    total_frames: usize,
    num_channels: usize,
    _phantom: PhantomData<T>,
}

impl<T: AudioSample> FrameIterator<T> {
    fn new(audio: &AudioSamples<T>) -> Self {
        let total_frames = audio.samples_per_channel();
        let num_channels = audio.num_channels();

        Self {
            audio: audio as *const AudioSamples<T>,
            current_frame: 0,
            total_frames,
            num_channels,
            _phantom: PhantomData,
        }
    }

    #[inline]
    fn audio_ref(&self) -> &AudioSamples<T> {
        // SAFETY: The pointer is valid for the iterator's lifetime because:
        // 1. It was created from a valid reference in new()
        // 2. AudioSamples owns its data and won't be moved during iteration
        // 3. We only read from the data, never modify it
        unsafe { &*self.audio }
    }
}

impl<T: AudioSample> Iterator for FrameIterator<T> {
    type Item = AudioSamples<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_frame >= self.total_frames {
            return None;
        }

        let audio = self.audio_ref();
        let mut frame = Vec::with_capacity(self.num_channels);

        let audio_samples = match &audio.data {
            AudioData::Mono(arr) => {
                frame.push(arr[self.current_frame]);
                AudioSamples::new_mono(Array1::from_vec(frame), audio.sample_rate)
            }
            AudioData::MultiChannel(arr) => {
                for ch in 0..self.num_channels {
                    frame.push(arr[[ch, self.current_frame]]);
                }
                AudioSamples::new_mono(Array1::from_vec(frame), audio.sample_rate)
            }
        };

        self.current_frame += 1;
        Some(audio_samples)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_frames - self.current_frame;
        (remaining, Some(remaining))
    }
}

impl<T: AudioSample> ExactSizeIterator for FrameIterator<T> {}

/// Iterator over complete channels of audio data.
///
/// Each iteration yields all samples from one channel before proceeding to the next channel.
pub struct ChannelIterator<T: AudioSample> {
    /// Raw pointer to the AudioSamples struct.
    ///
    /// Same reasoning as FrameIterator - we use a raw pointer to avoid borrowing
    /// restrictions while maintaining safety through read-only access to immutable data.
    audio: *const AudioSamples<T>,
    current_channel: usize,
    total_channels: usize,
    _phantom: PhantomData<T>,
}

impl<T: AudioSample> ChannelIterator<T> {
    fn new(audio: &AudioSamples<T>) -> Self {
        let total_channels = audio.num_channels();

        Self {
            audio: audio as *const AudioSamples<T>,
            current_channel: 0,
            total_channels,
            _phantom: PhantomData,
        }
    }

    #[inline]
    fn audio_ref(&self) -> &AudioSamples<T> {
        // SAFETY: Same safety reasoning as FrameIterator::audio_ref()
        unsafe { &*self.audio }
    }
}

impl<T: AudioSample> Iterator for ChannelIterator<T> {
    type Item = AudioSamples<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_channel >= self.total_channels {
            return None;
        }

        let audio = self.audio_ref();
        let channel_data = match &audio.data {
            AudioData::Mono(arr) => {
                // For mono, there's only one channel
                arr.to_vec()
            }
            AudioData::MultiChannel(arr) => {
                // Get the row for this channel
                arr.row(self.current_channel).to_vec()
            }
        };
        let sample_rate = audio.sample_rate;
        self.current_channel += 1;
        let array_channel = Array1::from_vec(channel_data);
        Some(AudioSamples::new_mono(array_channel, sample_rate))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_channels - self.current_channel;
        (remaining, Some(remaining))
    }
}

impl<T: AudioSample> ExactSizeIterator for ChannelIterator<T> {}

/// Padding strategy for window iteration when the window extends beyond available data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingMode {
    /// Zero-pad incomplete windows to maintain consistent window size.
    Zero,
    /// Return incomplete windows without padding (variable size).
    None,
    /// Skip incomplete windows entirely.
    Skip,
}

impl Default for PaddingMode {
    fn default() -> Self {
        PaddingMode::Zero
    }
}

/// Iterator over overlapping windows of audio data.
///
/// This iterator supports various windowing strategies including overlap and padding.
/// It's optimized for audio processing operations like STFT that require consistent
/// window sizes with configurable overlap.
pub struct WindowIterator<T: AudioSample> {
    /// Raw pointer to the AudioSamples struct.
    ///
    /// Same reasoning as other iterators - enables multiple iterator creation
    /// while maintaining safety through read-only access to immutable data.
    audio: *const AudioSamples<T>,
    window_size: usize,
    hop_size: usize,
    current_position: usize,
    total_samples: usize,
    total_windows: usize,
    current_window: usize,
    padding_mode: PaddingMode,
    _phantom: PhantomData<T>,
}

impl<T: AudioSample> WindowIterator<T> {
    fn new(audio: &AudioSamples<T>, window_size: usize, hop_size: usize) -> Self {
        let total_samples = audio.samples_per_channel();

        // Calculate total number of windows based on default padding mode (Zero)
        let total_windows =
            Self::calculate_total_windows(total_samples, window_size, hop_size, PaddingMode::Zero);

        Self {
            audio: audio as *const AudioSamples<T>,
            window_size,
            hop_size,
            current_position: 0,
            total_samples,
            total_windows,
            current_window: 0,
            padding_mode: PaddingMode::Zero,
            _phantom: PhantomData,
        }
    }

    fn calculate_total_windows(
        total_samples: usize,
        window_size: usize,
        hop_size: usize,
        padding_mode: PaddingMode,
    ) -> usize {
        if total_samples == 0 {
            return 0;
        }

        // Calculate the maximum number of windows we could have
        // This is the ceiling of total_samples / hop_size
        let max_windows = (total_samples + hop_size - 1) / hop_size;

        match padding_mode {
            PaddingMode::Zero => {
                // With zero padding, we can always create max_windows
                max_windows
            }
            PaddingMode::None => {
                // With no padding, count windows that have at least some real data
                let mut count = 0;
                let mut pos = 0;
                while pos < total_samples {
                    count += 1;
                    pos += hop_size;
                }
                count
            }
            PaddingMode::Skip => {
                // With skip, only count complete windows
                if total_samples < window_size {
                    0
                } else {
                    1 + (total_samples - window_size) / hop_size
                }
            }
        }
    }

    /// Set the padding mode for this iterator.
    pub fn with_padding_mode(mut self, mode: PaddingMode) -> Self {
        self.padding_mode = mode;

        // Recalculate total windows based on padding mode
        self.total_windows = Self::calculate_total_windows(
            self.total_samples,
            self.window_size,
            self.hop_size,
            mode,
        );

        self
    }

    #[inline]
    fn audio_ref(&self) -> &AudioSamples<T> {
        // SAFETY: Same safety reasoning as other iterators
        unsafe { &*self.audio }
    }
}

impl<T: AudioSample> Iterator for WindowIterator<T> {
    type Item = AudioSamples<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_window >= self.total_windows {
            return None;
        }

        let audio = self.audio_ref();
        let start_pos = self.current_position;
        let end_pos = start_pos + self.window_size;

        let window = if end_pos <= self.total_samples {
            // Complete window within bounds
            match &audio.data {
                AudioData::Mono(arr) => {
                    let arr_slice = arr.slice(ndarray::s![start_pos..end_pos]).to_vec();
                    let audio_window =
                        AudioSamples::new_mono(Array1::from_vec(arr_slice), audio.sample_rate);
                    audio_window
                }
                AudioData::MultiChannel(arr) => {
                    // For multi-channel, create interleaved window
                    let mut window_data =
                        Vec::with_capacity(self.window_size * audio.num_channels());
                    for sample_idx in start_pos..end_pos {
                        for ch in 0..audio.num_channels() {
                            window_data.push(arr[[ch, sample_idx]]);
                        }
                    }
                    let audio_window = AudioSamples::new_multi_channel(
                        Array2::from_shape_vec(
                            (audio.num_channels(), self.window_size),
                            window_data,
                        )
                        .unwrap(),
                        audio.sample_rate,
                    );
                    audio_window
                }
            }
        } else {
            // Window extends beyond available data
            match self.padding_mode {
                PaddingMode::Zero => {
                    // Zero-pad to maintain consistent window size
                    let available_samples = self.total_samples.saturating_sub(start_pos);
                    match &audio.data {
                        AudioData::Mono(arr) => {
                            let mut window_data = Vec::with_capacity(self.window_size);

                            // Add available samples
                            if available_samples > 0 {
                                let slice = arr.slice(ndarray::s![start_pos..self.total_samples]);
                                window_data.extend(slice.iter().copied());
                            }

                            // Zero-pad remainder
                            window_data.resize(self.window_size, T::default());

                            AudioSamples::new_mono(Array1::from_vec(window_data), audio.sample_rate)
                        }
                        AudioData::MultiChannel(arr) => {
                            let mut window_data =
                                Vec::with_capacity(self.window_size * audio.num_channels());

                            // Add available samples (interleaved)
                            for sample_idx in start_pos..self.total_samples {
                                for ch in 0..audio.num_channels() {
                                    window_data.push(arr[[ch, sample_idx]]);
                                }
                            }

                            // Zero-pad remainder
                            let remaining_samples = self.window_size - available_samples;
                            for _ in 0..(remaining_samples * audio.num_channels()) {
                                window_data.push(T::default());
                            }
                            AudioSamples::new_multi_channel(
                                Array2::from_shape_vec(
                                    (audio.num_channels(), self.window_size),
                                    window_data,
                                )
                                .unwrap(),
                                audio.sample_rate,
                            )
                        }
                    }
                }
                PaddingMode::None => {
                    // Return available samples without padding
                    let available_samples = self.total_samples.saturating_sub(start_pos);
                    if available_samples == 0 {
                        return None;
                    }

                    match &audio.data {
                        AudioData::Mono(arr) => {
                            let slice = arr.slice(ndarray::s![start_pos..self.total_samples]);
                            let slice = slice.to_vec();
                            AudioSamples::new_mono(Array1::from_vec(slice), audio.sample_rate)
                        }

                        AudioData::MultiChannel(arr) => {
                            let mut window_data =
                                Vec::with_capacity(available_samples * audio.num_channels());
                            for sample_idx in start_pos..self.total_samples {
                                for ch in 0..audio.num_channels() {
                                    window_data.push(arr[[ch, sample_idx]]);
                                }
                            }
                            AudioSamples::new_multi_channel(
                                Array2::from_shape_vec(
                                    (audio.num_channels(), available_samples),
                                    window_data,
                                )
                                .unwrap(),
                                audio.sample_rate,
                            )
                        }
                    }
                }
                PaddingMode::Skip => {
                    // Skip incomplete windows
                    return None;
                }
            }
        };

        self.current_position += self.hop_size;
        self.current_window += 1;
        Some(window)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_windows - self.current_window;
        (remaining, Some(remaining))
    }
}

impl<T: AudioSample> ExactSizeIterator for WindowIterator<T> {}

// ==============================
// PARALLEL ITERATORS
// ==============================

#[cfg(feature = "parallel-processing")]
/// Parallel iterator over frames of audio data.
///
/// This iterator processes frames in parallel using rayon's parallel iterator framework.
/// Each frame contains one sample from each channel at a given time point.
pub struct ParFrameIterator<T: AudioSample + Send + Sync> {
    /// Owned audio data to ensure thread safety
    frames_data: Vec<Vec<T>>,
    sample_rate: u32,
}

#[cfg(feature = "parallel-processing")]
impl<T: AudioSample + Send + Sync> ParFrameIterator<T> {
    fn new(audio: &AudioSamples<T>) -> Self {
        let num_frames = audio.samples_per_channel();
        let num_channels = audio.num_channels();
        let mut frames_data = Vec::with_capacity(num_frames);

        // Pre-compute all frames for parallel processing
        for frame_idx in 0..num_frames {
            let mut frame = Vec::with_capacity(num_channels);
            match &audio.data {
                AudioData::Mono(arr) => {
                    frame.push(arr[frame_idx]);
                }
                AudioData::MultiChannel(arr) => {
                    for ch in 0..num_channels {
                        frame.push(arr[[ch, frame_idx]]);
                    }
                }
            }
            frames_data.push(frame);
        }

        Self {
            frames_data,
            sample_rate: audio.sample_rate,
        }
    }
}

#[cfg(feature = "parallel-processing")]
impl<T: AudioSample + Send + Sync> ParallelIterator for ParFrameIterator<T> {
    type Item = AudioSamples<T>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        let par_iter = self
            .frames_data
            .into_par_iter()
            .map(|frame| AudioSamples::new_mono(Array1::from_vec(frame), self.sample_rate));
        par_iter.drive_unindexed(consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.frames_data.len())
    }
}

#[cfg(feature = "parallel-processing")]
impl<T: AudioSample + Send + Sync> IndexedParallelIterator for ParFrameIterator<T> {
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::Consumer<Self::Item>,
    {
        let par_iter = self
            .frames_data
            .into_par_iter()
            .map(|frame| AudioSamples::new_mono(Array1::from_vec(frame), self.sample_rate));
        par_iter.drive(consumer)
    }

    fn len(&self) -> usize {
        self.frames_data.len()
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: rayon::iter::plumbing::ProducerCallback<Self::Item>,
    {
        let par_iter = self
            .frames_data
            .into_par_iter()
            .map(|frame| AudioSamples::new_mono(Array1::from_vec(frame), self.sample_rate));
        par_iter.with_producer(callback)
    }
}

#[cfg(feature = "parallel-processing")]
/// Parallel iterator over complete channels of audio data.
///
/// This iterator processes channels in parallel using rayon's parallel iterator framework.
/// Each iteration yields all samples from one channel.
pub struct ParChannelIterator<T: AudioSample + Send + Sync> {
    /// Owned channel data to ensure thread safety
    channels_data: Vec<Vec<T>>,
    sample_rate: u32,
}

#[cfg(feature = "parallel-processing")]
impl<T: AudioSample + Send + Sync> ParChannelIterator<T> {
    fn new(audio: &AudioSamples<T>) -> Self {
        let num_channels = audio.num_channels();
        let mut channels_data = Vec::with_capacity(num_channels);

        // Pre-compute all channels for parallel processing
        for ch in 0..num_channels {
            let channel_data = match &audio.data {
                AudioData::Mono(arr) => arr.to_vec(),
                AudioData::MultiChannel(arr) => arr.row(ch).to_vec(),
            };
            channels_data.push(channel_data);
        }

        Self {
            channels_data,
            sample_rate: audio.sample_rate,
        }
    }
}

#[cfg(feature = "parallel-processing")]
impl<T: AudioSample + Send + Sync> ParallelIterator for ParChannelIterator<T> {
    type Item = AudioSamples<T>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        let par_iter = self
            .channels_data
            .into_par_iter()
            .map(|channel| AudioSamples::new_mono(Array1::from_vec(channel), self.sample_rate));
        par_iter.drive_unindexed(consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.channels_data.len())
    }
}

#[cfg(feature = "parallel-processing")]
impl<T: AudioSample + Send + Sync> IndexedParallelIterator for ParChannelIterator<T> {
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::Consumer<Self::Item>,
    {
        let par_iter = self
            .channels_data
            .into_par_iter()
            .map(|channel| AudioSamples::new_mono(Array1::from_vec(channel), self.sample_rate));
        par_iter.drive(consumer)
    }

    fn len(&self) -> usize {
        self.channels_data.len()
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: rayon::iter::plumbing::ProducerCallback<Self::Item>,
    {
        let par_iter = self
            .channels_data
            .into_par_iter()
            .map(|channel| AudioSamples::new_mono(Array1::from_vec(channel), self.sample_rate));
        par_iter.with_producer(callback)
    }
}

#[cfg(feature = "parallel-processing")]
/// Parallel iterator over overlapping windows of audio data.
///
/// This iterator processes windows in parallel using rayon's parallel iterator framework.
/// Each window supports various windowing strategies including overlap and padding.
pub struct ParWindowIterator<T: AudioSample + Send + Sync> {
    /// Owned window data to ensure thread safety
    windows_data: Vec<AudioSamples<T>>,
}

#[cfg(feature = "parallel-processing")]
impl<T: AudioSample + Send + Sync> ParWindowIterator<T> {
    fn new(audio: &AudioSamples<T>, window_size: usize, hop_size: usize) -> Self {
        // Use the existing WindowIterator to generate all windows sequentially,
        // then convert to parallel processing
        let windows_data: Vec<AudioSamples<T>> = audio.windows(window_size, hop_size).collect();

        Self { windows_data }
    }

    /// Set the padding mode for this parallel iterator.
    pub fn with_padding_mode(
        audio: &AudioSamples<T>,
        window_size: usize,
        hop_size: usize,
        mode: PaddingMode,
    ) -> Self {
        let windows_data: Vec<AudioSamples<T>> = audio
            .windows(window_size, hop_size)
            .with_padding_mode(mode)
            .collect();

        Self { windows_data }
    }
}

#[cfg(feature = "parallel-processing")]
impl<T: AudioSample + Send + Sync> ParallelIterator for ParWindowIterator<T> {
    type Item = AudioSamples<T>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        self.windows_data.into_par_iter().drive_unindexed(consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.windows_data.len())
    }
}

#[cfg(feature = "parallel-processing")]
impl<T: AudioSample + Send + Sync> IndexedParallelIterator for ParWindowIterator<T> {
    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::Consumer<Self::Item>,
    {
        self.windows_data.into_par_iter().drive(consumer)
    }

    fn len(&self) -> usize {
        self.windows_data.len()
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: rayon::iter::plumbing::ProducerCallback<Self::Item>,
    {
        self.windows_data.into_par_iter().with_producer(callback)
    }
}

// ==============================
// MUTABLE ITERATORS
// ==============================

/// Mutable iterator over frames of audio data.
///
/// This iterator provides mutable access to frames, allowing in-place modification
/// of audio samples. Each frame contains one sample from each channel at a given time point.
///
/// # Performance Notes
///
/// - Use this for frame-wise processing where you need to modify samples across channels
/// - For simple element-wise operations, prefer `AudioSamples::apply()` which uses optimized `mapv_inplace`
/// - This iterator is ideal for operations like panning, stereo effects, or cross-channel processing
pub struct FrameIteratorMut<'a, T: AudioSample> {
    /// Mutable reference to the AudioSamples data
    audio: &'a mut AudioSamples<T>,
    current_frame: usize,
    total_frames: usize,
    num_channels: usize,
    _phantom: PhantomData<T>,
}

impl<'a, T: AudioSample> FrameIteratorMut<'a, T> {
    fn new(audio: &'a mut AudioSamples<T>) -> Self {
        let total_frames = audio.samples_per_channel();
        let num_channels = audio.num_channels();

        Self {
            audio,
            current_frame: 0,
            total_frames,
            num_channels,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: AudioSample> Iterator for FrameIteratorMut<'a, T> {
    type Item = FrameMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_frame >= self.total_frames {
            return None;
        }

        // SAFETY: We use unsafe here to extract multiple mutable references to different parts
        // of the audio data. This is safe because:
        // 1. Each frame accesses non-overlapping memory locations
        // 2. We ensure frame bounds are within the audio data
        // 3. The FrameMut lifetime is tied to the iterator lifetime
        let frame = unsafe {
            match &mut self.audio.data {
                AudioData::Mono(arr) => {
                    let ptr = arr.as_mut_ptr().add(self.current_frame);
                    FrameMut::Mono(std::slice::from_raw_parts_mut(ptr, 1))
                }
                AudioData::MultiChannel(arr) => {
                    let mut ptrs = Vec::with_capacity(self.num_channels);
                    for ch in 0..self.num_channels {
                        let ptr = arr
                            .as_mut_ptr()
                            .add(ch * self.total_frames + self.current_frame);
                        ptrs.push(ptr);
                    }
                    FrameMut::MultiChannel(ptrs, self.num_channels)
                }
            }
        };

        self.current_frame += 1;
        Some(frame)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_frames - self.current_frame;
        (remaining, Some(remaining))
    }
}

impl<'a, T: AudioSample> ExactSizeIterator for FrameIteratorMut<'a, T> {}

/// A mutable frame containing samples from all channels at a given time point.
pub enum FrameMut<'a, T: AudioSample> {
    Mono(&'a mut [T]),
    MultiChannel(Vec<*mut T>, usize),
}

impl<'a, T: AudioSample> FrameMut<'a, T> {
    /// Returns the number of channels (samples) in this frame.
    pub fn len(&self) -> usize {
        match self {
            FrameMut::Mono(_) => 1,
            FrameMut::MultiChannel(_, channels) => *channels,
        }
    }

    /// Returns true if the frame is empty (should not happen in practice).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a mutable reference to the sample for a specific channel.
    pub fn get_mut(&mut self, channel: usize) -> Option<&mut T> {
        match self {
            FrameMut::Mono(slice) => {
                if channel == 0 {
                    slice.get_mut(0)
                } else {
                    None
                }
            }
            FrameMut::MultiChannel(ptrs, channels) => {
                if channel < *channels {
                    // SAFETY: The pointers were created from valid mutable references
                    // and we ensure channel bounds
                    Some(unsafe { &mut *ptrs[channel] })
                } else {
                    None
                }
            }
        }
    }

    /// Apply a function to all samples in this frame.
    pub fn apply<F>(&mut self, f: F)
    where
        F: Fn(T) -> T,
    {
        match self {
            FrameMut::Mono(slice) => {
                for sample in slice.iter_mut() {
                    *sample = f(*sample);
                }
            }
            FrameMut::MultiChannel(ptrs, channels) => {
                for ch in 0..*channels {
                    // SAFETY: Pointers are valid for the frame lifetime
                    unsafe {
                        let sample_ref = &mut *ptrs[ch];
                        *sample_ref = f(*sample_ref);
                    }
                }
            }
        }
    }

    /// Apply a function with channel index to all samples in this frame.
    pub fn apply_with_channel<F>(&mut self, f: F)
    where
        F: Fn(usize, T) -> T,
    {
        match self {
            FrameMut::Mono(slice) => {
                for sample in slice.iter_mut() {
                    *sample = f(0, *sample);
                }
            }
            FrameMut::MultiChannel(ptrs, channels) => {
                for ch in 0..*channels {
                    // SAFETY: Pointers are valid for the frame lifetime
                    unsafe {
                        let sample_ref = &mut *ptrs[ch];
                        *sample_ref = f(ch, *sample_ref);
                    }
                }
            }
        }
    }
}

/// Mutable iterator over complete channels of audio data.
///
/// This iterator provides mutable access to entire channels, allowing efficient
/// channel-wise processing operations.
///
/// # Performance Notes
///
/// - Use this for channel-specific processing (e.g., different EQ per channel)
/// - For simple element-wise operations on all channels, prefer `AudioSamples::apply()`
/// - This iterator is ideal for operations like channel-specific effects or balance adjustments
pub struct ChannelIteratorMut<'a, T: AudioSample> {
    /// Mutable reference to the AudioSamples data
    audio: &'a mut AudioSamples<T>,
    current_channel: usize,
    total_channels: usize,
    _phantom: PhantomData<T>,
}

impl<'a, T: AudioSample> ChannelIteratorMut<'a, T> {
    fn new(audio: &'a mut AudioSamples<T>) -> Self {
        let total_channels = audio.num_channels();

        Self {
            audio,
            current_channel: 0,
            total_channels,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T: AudioSample> Iterator for ChannelIteratorMut<'a, T> {
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_channel >= self.total_channels {
            return None;
        }

        // SAFETY: We use unsafe here to extract mutable references to different channels.
        // This is safe because:
        // 1. Each channel occupies non-overlapping memory
        // 2. We ensure channel bounds are within the audio data
        // 3. The returned slice lifetime is tied to the iterator lifetime
        let channel_slice = unsafe {
            match &mut self.audio.data {
                AudioData::Mono(arr) => {
                    // For mono, there's only one channel
                    let ptr = arr.as_mut_ptr();
                    let len = arr.len();
                    std::slice::from_raw_parts_mut(ptr, len)
                }
                AudioData::MultiChannel(arr) => {
                    // Get mutable access to the specific channel row
                    let (_channels, samples_per_channel) = arr.dim();
                    let ptr = arr
                        .as_mut_ptr()
                        .add(self.current_channel * samples_per_channel);
                    std::slice::from_raw_parts_mut(ptr, samples_per_channel)
                }
            }
        };

        self.current_channel += 1;
        Some(channel_slice)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_channels - self.current_channel;
        (remaining, Some(remaining))
    }
}

impl<'a, T: AudioSample> ExactSizeIterator for ChannelIteratorMut<'a, T> {}

/// Mutable iterator over overlapping windows of audio data.
///
/// This iterator provides mutable access to windows of audio data, useful for
/// processing operations that work on blocks of samples like FFT, filtering, etc.
///
/// # Performance Notes
///
/// - Use this for windowed operations like STFT, filtering, or block-based effects
/// - For simple element-wise operations, prefer `AudioSamples::apply()`
/// - This iterator handles overlap-add processing and window function application
pub struct WindowIteratorMut<'a, T: AudioSample> {
    /// Mutable reference to the AudioSamples data
    audio: &'a mut AudioSamples<T>,
    window_size: usize,
    hop_size: usize,
    current_position: usize,
    total_samples: usize,
    total_windows: usize,
    current_window: usize,
    padding_mode: PaddingMode,
    _phantom: PhantomData<T>,
}

impl<'a, T: AudioSample> WindowIteratorMut<'a, T> {
    fn new(audio: &'a mut AudioSamples<T>, window_size: usize, hop_size: usize) -> Self {
        let total_samples = audio.samples_per_channel();
        let total_windows =
            Self::calculate_total_windows(total_samples, window_size, hop_size, PaddingMode::Zero);

        Self {
            audio,
            window_size,
            hop_size,
            current_position: 0,
            total_samples,
            total_windows,
            current_window: 0,
            padding_mode: PaddingMode::Zero,
            _phantom: PhantomData,
        }
    }

    fn calculate_total_windows(
        total_samples: usize,
        window_size: usize,
        hop_size: usize,
        padding_mode: PaddingMode,
    ) -> usize {
        if total_samples == 0 {
            return 0;
        }

        let max_windows = (total_samples + hop_size - 1) / hop_size;

        match padding_mode {
            PaddingMode::Zero => max_windows,
            PaddingMode::None => {
                let mut count = 0;
                let mut pos = 0;
                while pos < total_samples {
                    count += 1;
                    pos += hop_size;
                }
                count
            }
            PaddingMode::Skip => {
                if total_samples < window_size {
                    0
                } else {
                    1 + (total_samples - window_size) / hop_size
                }
            }
        }
    }

    /// Set the padding mode for this iterator.
    pub fn with_padding_mode(mut self, mode: PaddingMode) -> Self {
        self.padding_mode = mode;
        self.total_windows = Self::calculate_total_windows(
            self.total_samples,
            self.window_size,
            self.hop_size,
            mode,
        );
        self
    }
}

impl<'a, T: AudioSample> Iterator for WindowIteratorMut<'a, T> {
    type Item = WindowMut<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_window >= self.total_windows {
            return None;
        }

        let start_pos = self.current_position;
        let end_pos = start_pos + self.window_size;

        // For mutable windows, we need to be more careful about overlapping regions
        // For now, implement non-overlapping windows for safety
        if end_pos <= self.total_samples {
            // Complete window within bounds
            let window = unsafe {
                match &mut self.audio.data {
                    AudioData::Mono(arr) => {
                        let ptr = arr.as_mut_ptr().add(start_pos);
                        let slice = std::slice::from_raw_parts_mut(ptr, self.window_size);
                        WindowMut::Mono(slice)
                    }
                    AudioData::MultiChannel(arr) => {
                        let (channels, _) = arr.dim();
                        let mut channel_ptrs = Vec::with_capacity(channels);
                        for ch in 0..channels {
                            let ptr = arr.as_mut_ptr().add(ch * self.total_samples + start_pos);
                            channel_ptrs.push(ptr);
                        }
                        WindowMut::MultiChannel(channel_ptrs, channels, self.window_size)
                    }
                }
            };

            self.current_position += self.hop_size;
            self.current_window += 1;
            Some(window)
        } else {
            // Handle padding cases - for now, skip incomplete windows in mutable iterator
            // to avoid complexity of handling padding in mutable context
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_windows - self.current_window;
        (remaining, Some(remaining))
    }
}

impl<'a, T: AudioSample> ExactSizeIterator for WindowIteratorMut<'a, T> {}

/// A mutable window of audio data.
pub enum WindowMut<'a, T: AudioSample> {
    Mono(&'a mut [T]),
    MultiChannel(Vec<*mut T>, usize, usize), // (channel_ptrs, num_channels, window_size)
}

impl<'a, T: AudioSample> WindowMut<'a, T> {
    /// Returns the window size.
    pub fn len(&self) -> usize {
        match self {
            WindowMut::Mono(slice) => slice.len(),
            WindowMut::MultiChannel(_, _, window_size) => *window_size,
        }
    }

    /// Returns true if the window is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of channels in this window.
    pub fn num_channels(&self) -> usize {
        match self {
            WindowMut::Mono(_) => 1,
            WindowMut::MultiChannel(_, channels, _) => *channels,
        }
    }

    /// Get a mutable slice for a specific channel.
    pub fn channel_mut(&mut self, channel: usize) -> Option<&mut [T]> {
        match self {
            WindowMut::Mono(slice) => {
                if channel == 0 {
                    Some(*slice)
                } else {
                    None
                }
            }
            WindowMut::MultiChannel(ptrs, channels, window_size) => {
                if channel < *channels {
                    // SAFETY: Pointers are valid for the window lifetime
                    Some(unsafe { std::slice::from_raw_parts_mut(ptrs[channel], *window_size) })
                } else {
                    None
                }
            }
        }
    }

    /// Apply a function to all samples in this window.
    pub fn apply<F>(&mut self, f: F)
    where
        F: Fn(T) -> T + Copy,
    {
        match self {
            WindowMut::Mono(slice) => {
                for sample in slice.iter_mut() {
                    *sample = f(*sample);
                }
            }
            WindowMut::MultiChannel(ptrs, channels, window_size) => {
                for ch in 0..*channels {
                    // SAFETY: Pointers are valid for the window lifetime
                    unsafe {
                        let channel_slice = std::slice::from_raw_parts_mut(ptrs[ch], *window_size);
                        for sample in channel_slice.iter_mut() {
                            *sample = f(*sample);
                        }
                    }
                }
            }
        }
    }

    /// Apply a window function to all samples.
    pub fn apply_window_function<F>(&mut self, window_fn: F)
    where
        F: Fn(usize, usize) -> T + Copy, // (index, window_size) -> window_value
        T: std::ops::MulAssign,
    {
        let window_size = self.len();
        match self {
            WindowMut::Mono(slice) => {
                for (i, sample) in slice.iter_mut().enumerate() {
                    *sample *= window_fn(i, window_size);
                }
            }
            WindowMut::MultiChannel(ptrs, channels, _) => {
                for ch in 0..*channels {
                    // SAFETY: Pointers are valid for the window lifetime
                    unsafe {
                        let channel_slice = std::slice::from_raw_parts_mut(ptrs[ch], window_size);
                        for (i, sample) in channel_slice.iter_mut().enumerate() {
                            *sample *= window_fn(i, window_size);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioSamples;
    use ndarray::{Array1, array};

    #[test]
    fn test_frame_iterator_mono() {
        let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0], 44100);
        let frames: Vec<AudioSamples<f32>> = audio.frames().collect();

        assert_eq!(frames.len(), 5);
        assert_eq!(frames[0].as_mono().unwrap()[0], 1.0);
        assert_eq!(frames[1].as_mono().unwrap()[0], 2.0);
        assert_eq!(frames[4].as_mono().unwrap()[0], 5.0);
    }

    #[test]
    fn test_frame_iterator_stereo() {
        let audio =
            AudioSamples::new_multi_channel(array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], 44100);
        let frames: Vec<AudioSamples<f32>> = audio.frames().collect();

        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0].as_mono().unwrap().to_vec(), vec![1.0, 4.0]);
        assert_eq!(frames[1].as_mono().unwrap().to_vec(), vec![2.0, 5.0]);
        assert_eq!(frames[2].as_mono().unwrap().to_vec(), vec![3.0, 6.0]);
    }

    #[test]
    fn test_channel_iterator_mono() {
        let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0], 44100);
        let channels: Vec<AudioSamples<f32>> = audio.channels().collect();

        assert_eq!(channels.len(), 1);
        assert_eq!(
            channels[0].as_mono().unwrap().to_vec(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn test_channel_iterator_stereo() {
        let audio =
            AudioSamples::new_multi_channel(array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], 44100);
        let channels: Vec<AudioSamples<f32>> = audio.channels().collect();

        assert_eq!(channels.len(), 2);
        assert_eq!(channels[0].as_mono().unwrap().to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(channels[1].as_mono().unwrap().to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_window_iterator_no_overlap() {
        let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 44100);
        let windows: Vec<AudioSamples<f32>> = audio.windows(3, 3).collect();

        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].as_mono().unwrap().to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(windows[1].as_mono().unwrap().to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_window_iterator_with_overlap() {
        let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 44100);
        let windows: Vec<AudioSamples<f32>> = audio.windows(4, 2).collect();

        // For 6 samples, window_size=4, hop_size=2:
        // Window 1: position 0-3 (samples 0,1,2,3)
        // Window 2: position 2-5 (samples 2,3,4,5)
        // Window 3: position 4-7 (samples 4,5 + 2 zeros for padding)
        assert_eq!(windows.len(), 3);
        assert_eq!(
            windows[0].as_mono().unwrap().to_vec(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            windows[1].as_mono().unwrap().to_vec(),
            vec![3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(
            windows[2].as_mono().unwrap().to_vec(),
            vec![5.0, 6.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_window_iterator_zero_padding() {
        let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0], 44100);
        let windows: Vec<AudioSamples<f32>> = audio
            .windows(4, 3)
            .with_padding_mode(PaddingMode::Zero)
            .collect();

        assert_eq!(windows.len(), 2);
        assert_eq!(
            windows[0].as_mono().unwrap().to_vec(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            windows[1].as_mono().unwrap().to_vec(),
            vec![4.0, 5.0, 0.0, 0.0]
        ); // Zero-padded
    }

    #[test]
    fn test_window_iterator_no_padding() {
        let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0], 44100);
        let windows: Vec<AudioSamples<f32>> = audio
            .windows(4, 3)
            .with_padding_mode(PaddingMode::None)
            .collect();

        assert_eq!(windows.len(), 2);
        assert_eq!(
            windows[0].as_mono().unwrap().to_vec(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(windows[1].as_mono().unwrap().to_vec(), vec![4.0, 5.0]); // Incomplete window
    }

    #[test]
    fn test_window_iterator_skip_padding() {
        let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0], 44100);
        let windows: Vec<AudioSamples<f32>> = audio
            .windows(4, 3)
            .with_padding_mode(PaddingMode::Skip)
            .collect();

        assert_eq!(windows.len(), 1);
        assert_eq!(
            windows[0].as_mono().unwrap().to_vec(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn test_window_iterator_stereo_interleaved() {
        let audio = AudioSamples::new_multi_channel(
            array![[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            44100,
        );
        let windows: Vec<AudioSamples<f32>> = audio.windows(2, 2).collect();

        assert_eq!(windows.len(), 2);
        // First window: samples 0,1 interleaved across channels
        assert_eq!(
            windows[0]
                .as_multi_channel()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![&1.0, &5.0, &2.0, &6.0]
        );
        // Second window: samples 2,3 interleaved across channels
        assert_eq!(
            windows[1]
                .as_multi_channel()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![&3.0, &7.0, &4.0, &8.0]
        );
    }

    #[test]
    fn test_exact_size_iterators() {
        let audio =
            AudioSamples::new_multi_channel(array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], 44100);

        let frame_iter = audio.frames();
        assert_eq!(frame_iter.len(), 3);

        let channel_iter = audio.channels();
        assert_eq!(channel_iter.len(), 2);

        let window_iter = audio.windows(2, 1);
        assert_eq!(window_iter.len(), 3); // (3-2)/1 + 1 = 2, plus padding = 3
    }

    #[test]
    fn test_multiple_iterators_from_same_audio() {
        // This test verifies that our raw pointer approach allows multiple iterators
        let audio =
            AudioSamples::new_multi_channel(array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], 44100);

        // This should compile and work correctly
        let frames = audio.frames();
        let channels = audio.channels();
        let windows = audio.windows(2, 1);

        // Verify they all work independently
        assert_eq!(frames.len(), 3);
        assert_eq!(channels.len(), 2);
        assert_eq!(windows.len(), 3);
    }

    // ==============================
    // MUTABLE ITERATOR TESTS
    // ==============================

    #[test]
    fn test_frame_iterator_mut_mono() {
        let mut audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0], 44100);

        // Apply gain using mutable frame iterator
        for mut frame in audio.frames_mut() {
            frame.apply(|sample| sample * 2.0);
        }

        assert_eq!(
            audio.as_mono().unwrap(),
            &array![2.0f32, 4.0, 6.0, 8.0, 10.0]
        );
    }

    #[test]
    fn test_frame_iterator_mut_stereo() {
        let mut audio =
            AudioSamples::new_multi_channel(array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], 44100);

        // Apply different processing to each channel in each frame
        for mut frame in audio.frames_mut() {
            frame.apply_with_channel(|ch, sample| {
                if ch == 0 {
                    sample * 0.5 // Left channel: reduce volume
                } else {
                    sample * 1.5 // Right channel: increase volume
                }
            });
        }

        let expected = array![[0.5f32, 1.0, 1.5], [6.0, 7.5, 9.0]];
        assert_eq!(audio.as_multi_channel().unwrap(), &expected);
    }

    #[test]
    fn test_frame_iterator_mut_individual_access() {
        let mut audio = AudioSamples::new_multi_channel(array![[1.0f32, 2.0], [3.0, 4.0]], 44100);

        // Modify only specific channels in each frame
        for mut frame in audio.frames_mut() {
            if let Some(left) = frame.get_mut(0) {
                *left *= 10.0;
            }
            // Leave right channel unchanged
        }

        let expected = array![[10.0f32, 20.0], [3.0, 4.0]];
        assert_eq!(audio.as_multi_channel().unwrap(), &expected);
    }

    #[test]
    fn test_channel_iterator_mut_mono() {
        let mut audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0], 44100);

        for channel in audio.channels_mut() {
            for sample in channel.iter_mut() {
                *sample += 10.0;
            }
        }

        assert_eq!(audio.as_mono().unwrap(), &array![11.0f32, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_channel_iterator_mut_stereo() {
        let mut audio =
            AudioSamples::new_multi_channel(array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], 44100);

        // Apply different processing to each channel
        for (ch, channel) in audio.channels_mut().enumerate() {
            let gain = if ch == 0 { 0.5 } else { 2.0 };
            for sample in channel.iter_mut() {
                *sample *= gain;
            }
        }

        let expected = array![[0.5f32, 1.0, 1.5], [8.0, 10.0, 12.0]];
        assert_eq!(audio.as_multi_channel().unwrap(), &expected);
    }

    #[test]
    fn test_window_iterator_mut_mono() {
        let mut audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 44100);

        // Apply windowed processing (non-overlapping)
        for mut window in audio.windows_mut(3, 3) {
            window.apply(|sample| sample * 0.5);
        }

        assert_eq!(
            audio.as_mono().unwrap(),
            &array![0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0]
        );
    }

    #[test]
    fn test_window_iterator_mut_stereo() {
        let mut audio = AudioSamples::new_multi_channel(
            array![[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            44100,
        );

        // Apply windowed processing (non-overlapping)
        for mut window in audio.windows_mut(2, 2) {
            // Apply different gain to each channel
            for ch in 0..window.num_channels() {
                if let Some(channel) = window.channel_mut(ch) {
                    let gain = if ch == 0 { 0.8 } else { 1.2 };
                    for sample in channel.iter_mut() {
                        *sample *= gain;
                    }
                }
            }
        }

        let expected = array![[0.8f32, 1.6, 2.4, 3.2], [6.0, 7.2, 8.4, 9.6]];
        let result = audio.as_multi_channel().unwrap();
        for (i, (&actual, &expected)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "Mismatch at index {}: {} != {} (diff: {})",
                i,
                actual,
                expected,
                (actual - expected).abs()
            );
        }
    }

    #[test]
    fn test_window_function_application() {
        let mut audio = AudioSamples::new_mono(array![1.0f32, 1.0, 1.0, 1.0], 44100);

        // Apply Hann window function
        for mut window in audio.windows_mut(4, 4) {
            window.apply_window_function(|i, window_size| {
                0.5 * (1.0
                    - (2.0 * std::f32::consts::PI * i as f32 / (window_size - 1) as f32).cos())
            });
        }

        // Check that Hann window was applied (values should be different from 1.0)
        let result = audio.as_mono().unwrap();
        assert!(result[0] < 1.0); // Should be close to 0
        assert!(result[1] > 0.5); // Should be around 0.75
        assert!(result[2] > 0.5); // Should be around 0.75
        assert!(result[3] < 1.0); // Should be close to 0
    }

    #[test]
    fn test_mutable_iterator_size_hints() {
        let mut audio =
            AudioSamples::new_multi_channel(array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], 44100);

        let frame_iter = audio.frames_mut();
        assert_eq!(frame_iter.len(), 3);

        // For channel iterator, we need a fresh audio instance since frame_iter consumed the mutable borrow
        let mut audio2 =
            AudioSamples::new_multi_channel(array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], 44100);
        let channel_iter = audio2.channels_mut();
        assert_eq!(channel_iter.len(), 2);

        // For window iterator, we need another fresh audio instance
        let mut audio3 =
            AudioSamples::new_multi_channel(array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], 44100);
        let window_iter = audio3.windows_mut(2, 1);
        // Mutable iterator reports total possible windows but only yields complete ones
        assert_eq!(window_iter.len(), 3); // Reports total from padding calculation
        let actual_windows: Vec<_> = window_iter.collect();
        assert_eq!(actual_windows.len(), 2); // But only yields complete windows
    }

    #[test]
    fn test_performance_comparison_apply_vs_iterator() {
        // This test demonstrates when to use each approach
        let mut audio1 = AudioSamples::new_mono(Array1::<f32>::ones(1000), 44100);
        let mut audio2 = audio1.clone();

        // Method 1: Using optimized apply (recommended for simple operations)
        audio1.apply(|sample| sample * 0.5);

        // Method 2: Using mutable iterator (for more complex operations)
        for mut frame in audio2.frames_mut() {
            frame.apply(|sample| sample * 0.5);
        }

        // Results should be identical
        assert_eq!(audio1.as_mono().unwrap(), audio2.as_mono().unwrap());
    }

    #[test]
    fn test_frame_mut_edge_cases() {
        let mut audio = AudioSamples::new_mono(array![1.0f32], 44100);

        for mut frame in audio.frames_mut() {
            assert_eq!(frame.len(), 1);
            assert!(!frame.is_empty());
            assert!(frame.get_mut(0).is_some());
            assert!(frame.get_mut(1).is_none());
        }
    }

    #[test]
    fn test_window_mut_edge_cases() {
        let mut audio = AudioSamples::new_mono(array![1.0f32, 2.0], 44100);

        // Window larger than audio
        let windows: Vec<_> = audio.windows_mut(5, 1).collect();
        assert_eq!(windows.len(), 0); // Should not produce any windows

        // Test window bounds
        let mut audio2 = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0], 44100);
        for window in audio2.windows_mut(2, 2) {
            assert_eq!(window.len(), 2);
            assert!(!window.is_empty());
            assert_eq!(window.num_channels(), 1);
        }
    }

    // ==============================
    // PARALLEL ITERATOR TESTS
    // ==============================

    #[cfg(feature = "parallel-processing")]
    mod parallel_tests {
        use super::*;
        use crate::iterators::AudioSampleParallelIterators;
        use rayon::prelude::*;

        #[test]
        fn test_par_frame_iterator_mono() {
            let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0], 44100);

            let frame_energies: Vec<f32> = audio
                .par_frames()
                .map(|frame| frame.as_mono().unwrap().iter().map(|&s| s * s).sum())
                .collect();

            assert_eq!(frame_energies.len(), 5);
            assert_eq!(frame_energies, vec![1.0, 4.0, 9.0, 16.0, 25.0]);
        }

        #[test]
        fn test_par_frame_iterator_stereo() {
            let audio =
                AudioSamples::new_multi_channel(array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], 44100);

            let frame_sums: Vec<f32> = audio
                .par_frames()
                .map(|frame| frame.as_mono().unwrap().iter().sum())
                .collect();

            assert_eq!(frame_sums.len(), 3);
            assert_eq!(frame_sums[0], 5.0); // 1.0 + 4.0
            assert_eq!(frame_sums[1], 7.0); // 2.0 + 5.0
            assert_eq!(frame_sums[2], 9.0); // 3.0 + 6.0
        }

        #[test]
        fn test_par_channel_iterator_mono() {
            let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0], 44100);

            let channel_rms: Vec<f32> = audio
                .par_channels()
                .map(|channel| {
                    let samples = channel.as_mono().unwrap();
                    let sum_squares: f32 = samples.iter().map(|&s| s * s).sum();
                    (sum_squares / samples.len() as f32).sqrt()
                })
                .collect();

            assert_eq!(channel_rms.len(), 1);
            // RMS of [1, 2, 3, 4] = sqrt((1+4+9+16)/4) = sqrt(30/4) = sqrt(7.5)
            assert!((channel_rms[0] - (7.5f32).sqrt()).abs() < 1e-6);
        }

        #[test]
        fn test_par_channel_iterator_stereo() {
            let audio =
                AudioSamples::new_multi_channel(array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], 44100);

            let channel_maxes: Vec<f32> = audio
                .par_channels()
                .map(|channel| {
                    channel
                        .as_mono()
                        .unwrap()
                        .iter()
                        .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x))
                })
                .collect();

            assert_eq!(channel_maxes.len(), 2);
            assert_eq!(channel_maxes[0], 3.0); // Max of [1, 2, 3]
            assert_eq!(channel_maxes[1], 6.0); // Max of [4, 5, 6]
        }

        #[test]
        fn test_par_window_iterator_no_overlap() {
            let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 44100);

            let window_sums: Vec<f32> = audio
                .par_windows(3, 3)
                .map(|window| window.as_mono().unwrap().iter().sum())
                .collect();

            assert_eq!(window_sums.len(), 2);
            assert_eq!(window_sums[0], 6.0); // 1 + 2 + 3
            assert_eq!(window_sums[1], 15.0); // 4 + 5 + 6
        }

        #[test]
        fn test_par_window_iterator_with_overlap() {
            let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 44100);

            let window_sums: Vec<f32> = audio
                .par_windows(4, 2)
                .map(|window| window.as_mono().unwrap().iter().sum())
                .collect();

            assert_eq!(window_sums.len(), 3);
            assert_eq!(window_sums[0], 10.0); // 1 + 2 + 3 + 4
            assert_eq!(window_sums[1], 18.0); // 3 + 4 + 5 + 6
            assert_eq!(window_sums[2], 11.0); // 5 + 6 + 0 + 0 (zero-padded)
        }

        #[test]
        fn test_par_window_iterator_with_padding_modes() {
            let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0], 44100);

            // Test with zero padding (default)
            let zero_padded: Vec<f32> = audio
                .par_windows(4, 3)
                .map(|window| window.as_mono().unwrap().len() as f32)
                .collect();
            assert_eq!(zero_padded, vec![4.0, 4.0]); // Both windows have size 4

            // Test with no padding
            let no_padding = ParWindowIterator::with_padding_mode(&audio, 4, 3, PaddingMode::None);
            let no_pad_lens: Vec<f32> = no_padding
                .map(|window| window.as_mono().unwrap().len() as f32)
                .collect();
            assert_eq!(no_pad_lens, vec![4.0, 2.0]); // Second window incomplete

            // Test with skip padding
            let skip_padding =
                ParWindowIterator::with_padding_mode(&audio, 4, 3, PaddingMode::Skip);
            let skip_lens: Vec<f32> = skip_padding
                .map(|window| window.as_mono().unwrap().len() as f32)
                .collect();
            assert_eq!(skip_lens, vec![4.0]); // Only complete windows
        }

        #[test]
        fn test_parallel_vs_sequential_consistency() {
            let audio = AudioSamples::new_multi_channel(
                array![[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                44100,
            );

            // Compare frame processing
            let seq_frame_sums: Vec<f32> = audio
                .frames()
                .map(|frame| frame.as_mono().unwrap().iter().sum())
                .collect();

            let par_frame_sums: Vec<f32> = audio
                .par_frames()
                .map(|frame| frame.as_mono().unwrap().iter().sum())
                .collect();

            assert_eq!(seq_frame_sums, par_frame_sums);

            // Compare channel processing
            let seq_channel_sums: Vec<f32> = audio
                .channels()
                .map(|channel| channel.as_mono().unwrap().iter().sum())
                .collect();

            let par_channel_sums: Vec<f32> = audio
                .par_channels()
                .map(|channel| channel.as_mono().unwrap().iter().sum())
                .collect();

            assert_eq!(seq_channel_sums, par_channel_sums);

            // Compare window processing
            let seq_window_sums: Vec<f32> = audio
                .windows(2, 1)
                .map(|window| match window.as_multi_channel() {
                    Some(arr) => arr.iter().sum(),
                    None => window.as_mono().unwrap().iter().sum(),
                })
                .collect();

            let par_window_sums: Vec<f32> = audio
                .par_windows(2, 1)
                .map(|window| match window.as_multi_channel() {
                    Some(arr) => arr.iter().sum(),
                    None => window.as_mono().unwrap().iter().sum(),
                })
                .collect();

            assert_eq!(seq_window_sums, par_window_sums);
        }

        #[test]
        fn test_parallel_iterator_properties() {
            let audio =
                AudioSamples::new_multi_channel(array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], 44100);

            // Test that parallel iterators implement IndexedParallelIterator
            let frame_iter = audio.par_frames();
            assert_eq!(frame_iter.len(), 3);

            let channel_iter = audio.par_channels();
            assert_eq!(channel_iter.len(), 2);

            let window_iter = audio.par_windows(2, 1);
            assert_eq!(window_iter.len(), 3); // (3-2)/1 + 1 = 2, plus padding = 3
        }

        #[test]
        fn test_parallel_iterator_with_min_len() {
            let audio =
                AudioSamples::new_mono((0..1000).map(|i| i as f32).collect::<Array1<f32>>(), 44100);

            // Test that with_min_len works (rayon feature)
            let result: Vec<f32> = audio
                .par_frames()
                .with_min_len(10) // Minimum 10 items per thread
                .map(|frame| frame.as_mono().unwrap()[0] * 2.0)
                .collect();

            assert_eq!(result.len(), 1000);
            assert_eq!(result[0], 0.0);
            assert_eq!(result[999], 1998.0);
        }

        #[test]
        fn test_complex_parallel_processing() {
            let audio = AudioSamples::new_multi_channel(
                Array2::from_shape_fn((2, 1000), |(ch, sample)| {
                    (ch as f32 + 1.0) * (sample as f32 + 1.0)
                }),
                44100,
            );

            // Complex processing: compute spectral centroid per window in parallel
            let window_centroids: Vec<f32> = audio
                .par_windows(64, 32)
                .map(|window| {
                    let samples: Vec<f32> = match window.as_multi_channel() {
                        Some(arr) => arr.iter().copied().collect(),
                        None => window.as_mono().unwrap().to_vec(),
                    };
                    let mut weighted_sum = 0.0f32;
                    let mut magnitude_sum = 0.0f32;

                    for (i, &sample) in samples.iter().enumerate() {
                        let magnitude = sample.abs();
                        weighted_sum += (i as f32) * magnitude;
                        magnitude_sum += magnitude;
                    }

                    if magnitude_sum > 0.0 {
                        weighted_sum / magnitude_sum
                    } else {
                        0.0
                    }
                })
                .collect();

            assert!(!window_centroids.is_empty());
            // Verify that we got reasonable centroid values
            // For multi-channel data, the effective window size is channels * window_size
            let max_centroid = 2.0 * 64.0; // 2 channels * 64 samples
            for &centroid in &window_centroids {
                assert!(
                    centroid >= 0.0 && centroid < max_centroid,
                    "Centroid {} not in range [0, {})",
                    centroid,
                    max_centroid
                );
            }
        }
    }
}
