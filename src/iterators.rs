//! Structured iteration over audio sample data.
//!
//! This module defines the primary iteration abstractions for traversing
//! [`AudioSamples`] in semantically meaningful ways. Rather than exposing raw
//! indexing or layout-dependent access, the iterators in this module present
//! audio data in forms that reflect common signal-processing views, such as
//! frames, channels, and sliding windows.
//!
//! The purpose of this module is to provide **explicit and intention-revealing**
//! access patterns over audio data while preserving the invariants of
//! [`AudioSamples`]. By centralising iteration logic here, the crate ensures that
//! higher-level operations can reason about audio structure without duplicating
//! indexing, slicing, or boundary-handling logic.
//!
//! ## Conceptual Model
//!
//! Iteration is defined in terms of *audio structure*, not storage layout.
//! Depending on the iterator, each yielded item represents a coherent unit of
//! audio data, such as:
//!
//! - A frame containing one sample per channel at a given time index
//! - A complete channel viewed independently of others
//! - A fixed-size temporal window, optionally overlapping with neighbouring windows
//!
//! Some iterators yield borrowed views into the original data when this is
//! structurally valid. Others yield owned buffers when borrowing would be
//! incorrect or when padding or independent ownership is required. These
//! distinctions are part of the iterator’s contract and are documented at the
//! iterator level.
//!
//! ## Intended Usage
//!
//! The iterators in this module are intended to be used for read-oriented and
//! transformation-oriented workflows where the *shape* of the audio matters.
//! Typical use cases include feature extraction, windowed analysis, per-channel
//! processing, and frame-wise inspection.
//!
//! ```rust
//! use audio_samples::{AudioSamples, iterators::AudioSampleIterators};
//! use ndarray::array;
//!
//! let audio = AudioSamples::new_multi_channel(
//!     array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
//!     44_100,
//! );
//!
//! for frame in audio.frames() {
//!     println!("Frame: {:?}", frame);
//! }
//!
//! for channel in audio.channels() {
//!     println!("Channel samples: {:?}", channel);
//! }
//!
//! for window in audio.windows(1024, 512) {
//!     println!("Window length: {}", window.len());
//! }
//! ```
//!
//! ## Relationship to Other APIs
//!
//! This module complements higher-level processing methods on [`AudioSamples`]
//! by providing fine-grained control over iteration order and granularity.
//! Where in-place or overlapping mutation is required, specialised APIs such as
//! [`AudioSamples::apply_to_windows`] should be preferred, as overlapping mutable
//! iteration cannot be expressed safely through standard iterator interfaces.

#[cfg(feature = "editing")]
use non_empty_slice::{NonEmptyVec, non_empty_vec};

use crate::{
    AudioData, AudioSampleError, AudioSampleResult, AudioSamples, LayoutError,
    traits::StandardSample,
};

#[cfg(feature = "editing")]
use crate::AudioEditing;

use std::marker::PhantomData;

#[cfg(feature = "editing")]
use std::num::NonZeroUsize;

/// Extension trait providing iterator methods for AudioSamples.
pub trait AudioSampleIterators<'a, T>
where
    T: StandardSample,
{
    /// Returns an iterator over frames (one sample from each channel).
    ///
    /// For mono audio, each frame contains one sample.
    /// For multi-channel audio, each frame contains one sample from each channel.
    ///
    /// # Inputs
    /// - `&self`
    ///
    /// # Returns
    /// An iterator yielding an `AudioSamples` view for each frame. Each yielded item contains
    /// exactly one sample per channel.
    ///
    /// # Errors
    /// This iterator does not report errors. Internally it relies on valid in-bounds slicing.
    ///
    /// # Panics
    /// Does not panic.
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
    /// let frames: Vec<Vec<f32>> = audio.frames().map(|f| f.to_interleaved_vec()).collect();
    /// assert_eq!(frames, vec![vec![1.0, 3.0], vec![2.0, 4.0]]);
    /// ```
    fn frames(&'a self) -> FrameIterator<'a, T>;

    /// Returns an iterator over complete channels.
    ///
    /// Each iteration yields all samples from one channel before moving to the next.
    ///
    /// # Inputs
    /// - `&self`
    ///
    /// # Returns
    /// An iterator yielding owned `AudioSamples` values, one per channel.
    ///
    /// # Errors
    /// This iterator does not report errors. Internally it relies on valid in-bounds slicing.
    ///
    /// # Panics
    /// Does not panic.
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
    /// let channels: Vec<Vec<f32>> = audio.channels().map(|c| c.to_interleaved_vec()).collect();
    /// assert_eq!(channels[0], vec![1.0, 2.0, 3.0]);
    /// assert_eq!(channels[1], vec![4.0, 5.0, 6.0]);
    /// ```
    fn channels<'iter>(&'iter self) -> ChannelIterator<'iter, 'a, T>;

    #[cfg(feature = "editing")]
    /// Returns an iterator over overlapping windows of audio data.
    ///
    /// # Inputs
    /// - `window_size`: size of each window in samples (per channel)
    /// - `hop_size`: number of samples to advance between windows
    ///
    /// # Returns
    /// An iterator yielding owned `AudioSamples` windows.
    ///
    /// # Padding
    /// The default padding mode is [`PaddingMode::Zero`]. Use
    /// [`WindowIterator::with_padding_mode`] to change it.
    ///
    /// # Panics
    /// Does not panic. If `window_size == 0` or `hop_size == 0`, the iterator yields no items.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use audio_samples::{AudioSamples, iterators::AudioSampleIterators};
    /// # use ndarray::array;
    /// let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], 44100).unwrap();
    ///
    /// let windows: Vec<Vec<f32>> = audio
    ///     .windows(NonZeroUsize::new(4).unwrap(), NonZeroUsize::new(2).unwrap())
    ///     .map(|w| w.to_interleaved_vec())
    ///     .collect();
    /// assert_eq!(windows, vec![
    ///     vec![1.0, 2.0, 3.0, 4.0],
    ///     vec![3.0, 4.0, 5.0, 6.0],
    ///     vec![5.0, 6.0, 0.0, 0.0], // zero-padded tail
    /// ]);
    /// ```
    fn windows(&'a self, window_size: usize, hop_size: usize) -> WindowIterator<'a, T>;
}

impl<'a, T> AudioSamples<'a, T>
where
    T: StandardSample,
{
    /// Returns an iterator over frames (one sample from each channel).
    ///
    /// # Returns
    /// A [`FrameIterator`] yielding borrowed frame views.
    ///
    /// # Panics
    /// Does not panic.
    #[inline]
    #[must_use]
    pub fn frames(&'a self) -> FrameIterator<'a, T> {
        FrameIterator::new(self)
    }

    /// Returns an iterator over complete channels.
    ///
    /// # Returns
    /// A [`ChannelIterator`] yielding owned `AudioSamples` values (one per channel).
    ///
    /// # Panics
    /// Does not panic.
    #[inline]
    #[must_use]
    pub fn channels<'iter>(&'iter self) -> ChannelIterator<'iter, 'a, T> {
        ChannelIterator::new(self)
    }

    #[cfg(feature = "editing")]
    /// Returns an iterator over windows of audio data.
    ///
    /// The default padding mode is [`PaddingMode::Zero`].
    ///
    /// # Inputs
    /// - `window_size`: size of each window in samples (per channel)
    /// - `hop_size`: number of samples to advance between windows
    ///
    /// # Returns
    /// A [`WindowIterator`] yielding owned windows.
    #[inline]
    #[must_use]
    pub fn windows(
        &'a self,
        window_size: NonZeroUsize,
        hop_size: NonZeroUsize,
    ) -> WindowIterator<'a, T> {
        WindowIterator::new(self, window_size, hop_size)
    }

    /// Applies a function to each frame without borrowing issues.
    ///
    /// For mono audio, the callback receives a mutable slice of length 1.
    ///
    /// For multi-channel audio, the callback receives a temporary buffer containing exactly one
    /// sample per channel. The buffer is ordered by channel index and is copied back into the
    /// underlying audio after the callback returns.
    ///
    /// # Inputs
    /// - `f(frame_index, frame_samples)`
    ///
    /// # Returns
    /// `()`
    ///
    /// # Panics
    /// Does not panic.
    #[inline]
    pub fn apply_to_frames<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, &mut [T]), // (frame_index, frame_samples)
    {
        match &mut self.data {
            AudioData::Mono(arr) => {
                for (frame_idx, sample) in arr.iter_mut().enumerate() {
                    f(frame_idx, std::slice::from_mut(sample));
                }
            }
            AudioData::Multi(arr) => {
                let (channels, samples_per_channel) = arr.dim();

                for frame_idx in 0..samples_per_channel.get() {
                    let mut frame = Vec::with_capacity(channels.get());
                    for ch in 0..channels.get() {
                        frame.push(arr[[ch, frame_idx]]);
                    }

                    f(frame_idx, &mut frame);

                    for ch in 0..channels.get() {
                        arr[[ch, frame_idx]] = frame[ch];
                    }
                }
            }
        }
    }

    /// Applies a function to each channel's raw data without borrowing issues.
    ///
    /// # Inputs
    /// - `f(channel_index, channel_samples)`
    ///
    /// # Returns
    /// `Ok(())` on success.
    ///
    /// # Errors
    /// Returns [`AudioSampleError::Layout`] if the underlying storage is not contiguous.
    #[inline]
    pub fn try_apply_to_channel_data<F>(&mut self, mut f: F) -> AudioSampleResult<()>
    where
        F: FnMut(usize, &mut [T]), // (channel_index, channel_samples)
    {
        match &mut self.data {
            AudioData::Mono(arr) => {
                let slice = arr.as_slice_mut();
                f(0, slice);
            }
            AudioData::Multi(arr) => {
                let (channels, samples_per_channel) = arr.dim();
                let samples_per_channel = samples_per_channel.get();
                let slice = arr.as_slice_mut().ok_or_else(|| {
                    AudioSampleError::Layout(LayoutError::NonContiguous {
                        operation: "multi-channel iterator access".to_string(),
                        layout_type: "non-contiguous multi-channel data".to_string(),
                    })
                })?;

                for ch in 0..channels.get() {
                    let start_idx = ch * samples_per_channel;
                    let channel_slice = &mut slice[start_idx..start_idx + samples_per_channel];
                    f(ch, channel_slice);
                }
            }
        }
        Ok(())
    }

    /// Applies a function to each channel's raw data without borrowing issues.
    ///
    /// This is the infallible counterpart to [`AudioSamples::try_apply_to_channel_data`].
    ///
    /// # Panics
    /// Panics if the underlying storage is not contiguous in memory.
    #[inline]
    pub fn apply_to_channel_data<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, &mut [T]), // (channel_index, channel_samples)
    {
        self.try_apply_to_channel_data(|ch, data| f(ch, data))
            .expect("apply_to_channel_data requires contiguous storage; use try_apply_to_channel_data to handle non-contiguous inputs");
    }

    /// Applies a function to each window without borrowing issues.
    ///
    /// For mono audio, the callback receives a mutable slice into the underlying buffer.
    ///
    /// For multi-channel audio, the callback receives a temporary interleaved buffer of length
    /// `window_size * num_channels` that is copied back after the callback returns.
    ///
    /// # Inputs
    /// - `window_size`: size of each window in samples (per channel)
    /// - `hop_size`: number of samples to advance between windows
    /// - `f(window_index, window_samples)`
    ///
    /// # Panics
    /// Does not panic. If `window_size == 0` or the audio is empty, this method returns
    /// immediately.
    #[inline]
    pub fn apply_to_windows<F>(&mut self, window_size: usize, hop_size: usize, mut f: F)
    where
        F: FnMut(usize, &mut [T]), // (window_index, window_samples)
    {
        let total_samples = self.samples_per_channel().get();
        if total_samples == 0 || window_size == 0 {
            return;
        }

        match &mut self.data {
            AudioData::Mono(arr) => {
                let mut window_idx = 0;
                let mut pos = 0;

                while pos + window_size <= total_samples {
                    let slice = arr.as_slice_mut();
                    let window_slice = &mut slice[pos..pos + window_size];
                    f(window_idx, window_slice);
                    pos += hop_size;
                    window_idx += 1;
                }
            }
            AudioData::Multi(arr) => {
                let (rows, cols) = arr.dim();
                let rows = rows.get();
                let samples_per_channel = cols;

                let mut pos = 0;
                let mut window_idx = 0;

                while pos + window_size <= samples_per_channel.get() {
                    // Create a temporary buffer for the interleaved window
                    let mut window_data = vec![T::zero(); window_size * rows];

                    // Copy window data from each channel into interleaved buffer.
                    for ch in 0..rows {
                        for sample_idx in 0..window_size {
                            let dst_idx = sample_idx * rows + ch; // Interleaved layout
                            window_data[dst_idx] = arr[[ch, pos + sample_idx]];
                        }
                    }

                    // Call the user function
                    f(window_idx, &mut window_data);

                    // Copy modified data back to original channels.
                    for ch in 0..rows {
                        for sample_idx in 0..window_size {
                            let src_idx = sample_idx * rows + ch; // Interleaved layout
                            arr[[ch, pos + sample_idx]] = window_data[src_idx];
                        }
                    }

                    pos += hop_size;
                    window_idx += 1;
                }
            }
        }
    }
}

/// Iterates over time-aligned frames of an [`AudioSamples`] instance.
///
/// A *frame* represents the set of samples across all channels at a single
/// time index. For mono audio, each frame contains exactly one sample. For
/// multi-channel audio, each frame contains one sample per channel, preserving
/// channel alignment.
///
/// ## Purpose
///
/// `FrameIterator` provides a structured, time-centric view of audio data.
/// It is intended for algorithms that operate on synchronous samples across
/// channels, such as frame-wise feature extraction, analysis, or inspection.
///
/// The iterator yields immutable views into the underlying audio data. No
/// reordering, resampling, or interpolation is performed.
///
/// ## Invariants
///
/// - Frames are yielded in strictly increasing temporal order.
/// - Each yielded frame corresponds to exactly one time index.
/// - The number of frames is equal to the number of samples per channel.
/// - All channels are assumed to have equal length.
///
/// ## Assumptions and Limitations
///
/// This iterator assumes that the underlying [`AudioSamples`] instance is
/// channel-aligned and immutable for the lifetime of the iterator. It is not
/// suitable for in-place mutation or algorithms that require overlapping or
/// non-sequential access.
///
/// Use higher-level windowed or transformation APIs when temporal context
/// beyond a single frame is required.
pub struct FrameIterator<'a, T>
where
    T: StandardSample,
{
    /// The source audio data over which frames are iterated.
    audio: &'a AudioSamples<'a, T>,
    current_frame: usize,
    total_frames: usize,
    _phantom: PhantomData<T>,
}

impl<'a, T> FrameIterator<'a, T>
where
    T: StandardSample,
{
    /// Constructs a new frame iterator over the given audio.
    ///
    /// ## Purpose
    ///
    /// This constructor establishes a frame-wise traversal over the provided
    /// audio data, yielding one frame per time index.
    ///
    /// ## Arguments
    ///
    /// - `audio`: The source audio to iterate over. All channels must be
    ///   time-aligned.
    ///
    /// ## Behavioural Guarantees
    ///
    /// - The iterator will yield exactly `audio.samples_per_channel()` frames.
    /// - Frames are yielded in deterministic order.
    ///
    /// ## Panics
    ///
    /// This function does not panic.
    #[inline]
    #[must_use]
    pub fn new(audio: &'a AudioSamples<'a, T>) -> Self {
        let total_frames = audio.samples_per_channel().get();
        Self {
            audio,
            current_frame: 0,
            total_frames,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T> Iterator for FrameIterator<'a, T>
where
    T: StandardSample,
{
    type Item = AudioSamples<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_frame >= self.total_frames {
            return None;
        }

        let frame_range = self.current_frame..self.current_frame + 1;
        self.current_frame += 1;

        // Return a view of a single frame
        self.audio.slice_samples(frame_range).ok()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_frames - self.current_frame;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for FrameIterator<'_, T> where T: StandardSample {}

/// Iterates over complete channels of an [`AudioSamples`] instance.
///
/// Each iteration yields the full sequence of samples belonging to a single
/// channel, independent of other channels. Channels are yielded sequentially
/// in channel index order.
///
/// ## Purpose
///
/// `ChannelIterator` provides a channel-centric view of audio data. It is
/// intended for workflows that process or analyse channels independently,
/// such as per-channel filtering, statistics, or visualisation.
///
/// Unlike frame-based iteration, this iterator exposes the *entire temporal
/// extent* of one channel at a time.
///
/// ## Behaviour and Ownership
///
/// Each yielded item is an owned [`AudioSamples`] instance containing exactly
/// one channel. This reflects the fact that channel-wise slicing produces
/// independent audio objects rather than borrowed views.
///
/// As a result, channel iteration involves allocation and data copying.
/// Callers should take this into account when iterating over large audio
/// buffers or when allocation-free access is required.
///
/// ## Invariants
///
/// - Channels are yielded in increasing channel index order.
/// - Each channel is yielded exactly once.
/// - The number of yielded items is equal to the number of channels.
/// - All samples within a yielded item belong to the same channel.
///
/// ## Assumptions and Limitations
///
/// This iterator assumes that the underlying [`AudioSamples`] instance has a
/// fixed channel layout for the duration of iteration. It is not suitable for
/// algorithms that require simultaneous access to multiple channels or
/// fine-grained temporal alignment across channels.
pub struct ChannelIterator<'iter, 'data, T>
where
    T: StandardSample,
{
    /// The source audio from which channels are extracted.
    audio: &'iter AudioSamples<'data, T>,
    current_channel: usize,
    total_channels: usize,
}

impl<'iter, 'data, T> ChannelIterator<'iter, 'data, T>
where
    T: StandardSample,
{
    /// Constructs a new iterator over the channels of the given audio.
    ///
    /// ## Purpose
    ///
    /// This constructor establishes a channel-wise traversal over the provided
    /// audio data, yielding one complete channel per iteration.
    ///
    /// ## Arguments
    ///
    /// - `audio`: The source audio whose channels will be iterated.
    ///
    /// ## Behavioural Guarantees
    ///
    /// - The iterator will yield exactly `audio.num_channels()` items.
    /// - Channels are yielded in deterministic order.
    ///
    /// ## Panics
    ///
    /// This function does not panic.
    #[inline]
    #[must_use]
    pub fn new(audio: &'iter AudioSamples<'data, T>) -> Self {
        let total_channels = audio.num_channels().get();

        Self {
            audio,
            current_channel: 0,
            total_channels: total_channels as usize,
        }
    }
}

impl<T> Iterator for ChannelIterator<'_, '_, T>
where
    T: StandardSample,
{
    type Item = AudioSamples<'static, T>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_channel >= self.total_channels {
            return None;
        }

        let channel = match self
            .audio
            .clone()
            .into_owned()
            .slice_channels(self.current_channel..=self.current_channel)
        {
            Ok(ch) => ch,
            Err(e) => {
                eprintln!("Error slicing channel {}: {}", self.current_channel, e);
                return None;
            }
        };

        self.current_channel += 1;

        Some(channel)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_channels - self.current_channel;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for ChannelIterator<'_, '_, T> where T: StandardSample {}

/// Defines how window iteration behaves when a window extends beyond the
/// available audio data.
///
/// `PaddingMode` controls the treatment of trailing windows whose span exceeds
/// the number of samples per channel. The selected mode determines whether such
/// windows are padded, truncated, or omitted entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaddingMode {
    /// Pads incomplete windows with zeros so that all yielded windows have
    /// identical length.
    ///
    /// This mode guarantees a fixed window size and a deterministic number of
    /// windows, which is required by many spectral and frame-based algorithms.
    #[default]
    Zero,

    /// Yields trailing windows without padding.
    ///
    /// Windows near the end of the signal may be shorter than the configured
    /// window size. Callers must be prepared to handle variable-length windows.
    None,

    /// Omits any window that would extend beyond the available data.
    ///
    /// Only fully contained windows are yielded. This mode produces no padding
    /// and no partial windows.
    Skip,
}
/// Iterates over fixed-size temporal windows of audio data.
///
/// Each iteration yields a contiguous block of samples spanning all channels
/// over a fixed temporal extent. Successive windows are offset by a configurable
/// hop size and may overlap depending on the chosen parameters.
///
/// ## Purpose
///
/// `WindowIterator` provides a structured abstraction for windowed audio
/// processing. It is intended for algorithms that operate on local temporal
/// context, such as spectral analysis, feature extraction, and block-based
/// transformations.
///
/// Window iteration is defined in terms of *time*, not storage layout. All
/// windows preserve channel alignment and temporal ordering.
///
/// ## Window Boundaries and Padding
///
/// When a window would extend beyond the available data, its treatment is
/// determined by the configured [`PaddingMode`]. Depending on this mode,
/// trailing windows may be padded, truncated, or skipped entirely. This choice
/// directly affects both the number and shape of yielded windows.
///
/// ## Ownership and Allocation
///
/// Each yielded window is returned as an owned [`AudioSamples`] instance. This
/// allows windows to be processed independently but implies allocation and
/// copying **may** be performed. Whether or not this occurs is down to whether
/// the data is already owned or not. If not then yes, it will allocate,
/// otherwise the a borrow is used. For in-place or allocation-free processing,
/// prefer specialised higher-level APIs where available.
///
/// ## Invariants
///
/// - Windows are yielded in strictly increasing temporal order.
/// - All channels within a window remain time-aligned.
/// - The hop size between successive windows is constant.
/// - The iterator yields a finite, deterministic number of windows.
///
/// ## Assumptions and Limitations
///
/// This iterator assumes a fixed sampling rate and channel layout for the
/// lifetime of iteration. It is not suitable for overlapping mutable access or
/// algorithms that require shared ownership of window data.
#[cfg(feature = "editing")]
pub struct WindowIterator<'a, T>
where
    T: StandardSample,
{
    /// The source audio from which windows are extracted.
    audio: &'a AudioSamples<'a, T>,
    window_size: NonZeroUsize,
    hop_size: NonZeroUsize,
    current_position: usize,
    total_samples: NonZeroUsize,
    total_windows: NonZeroUsize,
    current_window: usize,
    padding_mode: PaddingMode,
    _phantom: PhantomData<T>,
}

#[cfg(feature = "editing")]
impl<'a, T> WindowIterator<'a, T>
where
    T: StandardSample,
{
    /// Constructs a new window iterator over the given audio.
    ///
    /// ## Purpose
    ///
    /// This constructor establishes a windowed traversal over the provided
    /// audio data using the specified window and hop sizes.
    ///
    /// ## Arguments
    ///
    /// - `audio`: The source audio to iterate over.
    /// - `window_size`: The number of samples per channel in each window.
    /// - `hop_size`: The number of samples between the starts of successive windows.
    ///
    /// ## Behavioural Guarantees
    ///
    /// - Windows are generated deterministically from the start of the signal.
    /// - The default padding mode is [`PaddingMode::Zero`].
    ///
    /// ## Degenerate Parameters
    ///
    /// If either `window_size` or `hop_size` is zero, the iterator yields no
    /// windows.
    ///
    /// ## Panics
    ///
    /// This function does not panic.
    fn new(
        audio: &'a AudioSamples<'a, T>,
        window_size: NonZeroUsize,
        hop_size: NonZeroUsize,
    ) -> Self {
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

    const fn calculate_total_windows(
        total_samples: NonZeroUsize,
        window_size: NonZeroUsize,
        hop_size: NonZeroUsize,
        padding_mode: PaddingMode,
    ) -> NonZeroUsize {
        // Calculate the maximum number of windows we could have
        // This is the ceiling of total_samples / hop_size
        let max_windows = total_samples.div_ceil(hop_size);

        match padding_mode {
            PaddingMode::Zero => {
                // With zero padding, we can always create max_windows
                max_windows
            }
            PaddingMode::None => {
                // With no padding, count windows that have at least some real data
                let mut count = 0;
                let mut pos = 0;
                while pos < total_samples.get() {
                    count += 1;
                    pos += hop_size.get();
                }
                // safety: count is at least 1 because total_samples is non-zero
                unsafe { NonZeroUsize::new_unchecked(count) }
            }
            PaddingMode::Skip => {
                // With skip, only count complete windows
                // safety: we have already checked that window_size and hop_size are non-zero
                unsafe {
                    NonZeroUsize::new_unchecked(
                        1 + (total_samples.get() - window_size.get()) / hop_size.get(),
                    )
                }
            }
        }
    }

    /// Sets the padding strategy used for trailing windows.
    ///
    /// ## Purpose
    ///
    /// This method allows callers to control how incomplete windows at the end
    /// of the signal are handled.
    ///
    /// Changing the padding mode affects both the number of yielded windows and
    /// the shape of the final windows.
    ///
    /// ## Behavioural Guarantees
    ///
    /// - The iterator’s internal window count is updated consistently with the
    ///   selected mode.
    #[inline]
    #[must_use]
    pub const fn with_padding_mode(mut self, mode: PaddingMode) -> Self {
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

#[cfg(feature = "editing")]
impl<T> Iterator for WindowIterator<'_, T>
where
    T: StandardSample,
{
    type Item = AudioSamples<'static, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_window >= self.total_windows.get() {
            return None;
        }

        let start_pos = self.current_position;
        let end_pos = start_pos + self.window_size.get();

        let window = if end_pos <= self.total_samples.get() {
            // Complete window within bounds
            self.audio
                .slice_samples(start_pos..end_pos)
                .ok()
                .map(super::repr::AudioSamples::into_owned)
        } else {
            // Window extends beyond available data
            match self.padding_mode {
                PaddingMode::Zero => {
                    // Zero-pad to maintain consistent window size
                    let available_samples = self.total_samples.get().saturating_sub(start_pos);
                    match &self.audio.data {
                        AudioData::Mono(_) => {
                            // Add available samples
                            let starting_slice = if available_samples > 0 {
                                let slice = self
                                    .audio
                                    .slice_samples(start_pos..self.total_samples.get())
                                    .ok()?
                                    .into_owned();
                                Some(slice)
                            } else {
                                None
                            };

                            let silence_samples = self.window_size.get() - available_samples;
                            let length = NonZeroUsize::new(silence_samples)?;
                            let silence = if silence_samples > 0 {
                                let silence =
                                    AudioSamples::<T>::zeros_mono(length, self.audio.sample_rate);
                                Some(silence)
                            } else {
                                return starting_slice;
                            };

                            match (starting_slice, silence) {
                                (None, None) => None,
                                (None, Some(silence)) => Some(silence),
                                (Some(starting_slice), None) => Some(starting_slice),
                                (Some(s), Some(z)) => {
                                    let slices = vec![s, z];
                                    let slices = NonEmptyVec::new(slices).ok()?;
                                    Some(AudioSamples::concatenate_owned(slices).ok()?)
                                }
                            }
                        }
                        AudioData::Multi(_) => {
                            let interleaved_slice = if available_samples > 0 {
                                let slice = self
                                    .audio
                                    .slice_samples(start_pos..self.total_samples.get())
                                    .ok()?
                                    .into_owned();
                                Some(slice)
                            } else {
                                None
                            };

                            // Zero-pad remainder
                            let remaining_samples = self.window_size.get() - available_samples;
                            if remaining_samples == 0 {
                                return interleaved_slice;
                            }

                            let length = NonZeroUsize::new(remaining_samples)?;

                            let silence = AudioSamples::<T>::zeros_multi_channel(
                                self.audio.num_channels(),
                                length,
                                self.audio.sample_rate,
                            );

                            match interleaved_slice {
                                None => Some(silence),
                                Some(slice) => {
                                    AudioSamples::concatenate_owned(non_empty_vec![slice, silence])
                                        .ok()
                                }
                            }
                        }
                    }
                }
                PaddingMode::None => {
                    // Return available samples without padding
                    let available_samples = self.total_samples.get().saturating_sub(start_pos);
                    if available_samples == 0 {
                        return None;
                    }

                    self.audio
                        .slice_samples(start_pos..self.total_samples.get())
                        .ok()
                        .map(super::repr::AudioSamples::into_owned)
                }
                PaddingMode::Skip => {
                    // Skip incomplete windows
                    return None;
                }
            }
        };

        self.current_position += self.hop_size.get();
        self.current_window += 1;
        window
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_windows.get() - self.current_window;
        (remaining, Some(remaining))
    }
}

#[cfg(feature = "editing")]
impl<T> ExactSizeIterator for WindowIterator<'_, T> where T: StandardSample {}
#[cfg(test)]
mod tests {
    // use super::*;
    use crate::PaddingMode;
    use crate::AudioSamples;
    use crate::sample_rate;
    use ndarray::{Array1, array};
    use non_empty_slice::non_empty_vec;

    #[test]
    fn test_frame_iterator_mono() {
        let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0], sample_rate!(44100))
            .unwrap();
        audio
            .frames()
            .zip([1.0f32, 2.0, 3.0, 4.0, 5.0])
            .for_each(|(f, x)| {
                assert_eq!(f.to_interleaved_vec(), non_empty_vec![x]);
            });
    }

    #[test]
    fn test_frame_iterator_stereo() {
        let audio = AudioSamples::new_multi_channel(
            array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
            sample_rate!(44100),
        )
        .unwrap();

        // Borrowing-first behavior: work with frames directly, don't collect into Vec
        let expected_frames = vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]];

        for (i, frame) in audio.frames().enumerate() {
            assert_eq!(frame.to_interleaved_vec().to_vec(), expected_frames[i]);
        }
    }

    #[test]
    fn test_channel_iterator_mono() {
        let audio =
            AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0], sample_rate!(44100)).unwrap();

        // Borrowing-first behavior: work with channels directly
        let mut channel_count = 0;
        for channel in audio.channels() {
            channel_count += 1;
            assert_eq!(
                channel.to_interleaved_vec(),
                non_empty_vec![1.0, 2.0, 3.0, 4.0]
            );
        }
        assert_eq!(channel_count, 1);
    }

    #[test]
    fn test_channel_iterator_stereo() {
        let audio = AudioSamples::new_multi_channel(
            array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
            sample_rate!(44100),
        )
        .unwrap();

        // Borrowing-first behavior: work with channels directly
        let expected_channels = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        for (i, channel) in audio.channels().enumerate() {
            assert_eq!(channel.as_mono().unwrap().to_vec(), expected_channels[i]);
        }
    }

    #[cfg(feature = "editing")]
    #[test]
    fn test_window_iterator_no_overlap() {
        let audio =
            AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], sample_rate!(44100))
                .unwrap();
        let windows: Vec<AudioSamples<f32>> =
            audio.windows(crate::nzu!(3), crate::nzu!(3)).collect();

        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].as_mono().unwrap().to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(windows[1].as_mono().unwrap().to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[cfg(feature = "editing")]
    #[test]
    fn test_window_iterator_with_overlap() {
        let audio =
            AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], sample_rate!(44100))
                .unwrap();
        let windows: Vec<AudioSamples<f32>> =
            audio.windows(crate::nzu!(4), crate::nzu!(2)).collect();

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

    #[cfg(feature = "editing")]
    #[test]
    fn test_window_iterator_zero_padding() {

        let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0], sample_rate!(44100))
            .unwrap();
        let windows: Vec<AudioSamples<f32>> = audio
            .windows(crate::nzu!(4), crate::nzu!(3))
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

    #[cfg(feature = "editing")]
    #[test]
    fn test_window_iterator_no_padding() {
        let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0], sample_rate!(44100))
            .unwrap();
        let windows: Vec<AudioSamples<f32>> = audio
            .windows(crate::nzu!(4), crate::nzu!(3))
            .with_padding_mode(PaddingMode::None)
            .collect();

        assert_eq!(windows.len(), 2);
        assert_eq!(
            windows[0].as_mono().unwrap().to_vec(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(windows[1].as_mono().unwrap().to_vec(), vec![4.0, 5.0]); // Incomplete window
    }

    #[cfg(feature = "editing")]
    #[test]
    fn test_window_iterator_skip_padding() {
        let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0], sample_rate!(44100))
            .unwrap();
        let windows: Vec<AudioSamples<f32>> = audio
            .windows(crate::nzu!(4), crate::nzu!(3))
            .with_padding_mode(PaddingMode::Skip)
            .collect();

        assert_eq!(windows.len(), 1);
        assert_eq!(
            windows[0].as_mono().unwrap().to_vec(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[cfg(feature = "editing")]
    #[test]
    fn test_window_iterator_stereo_interleaved() {
        let audio = AudioSamples::new_multi_channel(
            array![[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            sample_rate!(44100),
        )
        .unwrap();
        let windows: Vec<AudioSamples<f32>> =
            audio.windows(crate::nzu!(2), crate::nzu!(2)).collect();

        assert_eq!(windows.len(), 2);
        // First window: samples 0,1 interleaved across channels
        assert_eq!(
            windows[0].to_interleaved_vec(),
            non_empty_vec![1.0, 5.0, 2.0, 6.0]
        );
        // Second window: samples 2,3 interleaved across channels
        assert_eq!(
            windows[1].to_interleaved_vec(),
            non_empty_vec![3.0, 7.0, 4.0, 8.0]
        );
    }

    #[test]
    fn test_exact_size_iterators() {
        let audio = AudioSamples::new_multi_channel(
            array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
            sample_rate!(44100),
        )
        .unwrap();

        let frame_iter = audio.frames();
        assert_eq!(frame_iter.len(), 3);

        let channel_iter = audio.channels();
        assert_eq!(channel_iter.len(), 2);

        #[cfg(feature = "editing")]
        {
            let window_iter = audio.windows(crate::nzu!(2), crate::nzu!(1));
            assert_eq!(window_iter.len(), 3); // (3-2)/1 + 1 = 2, plus padding = 3
        }
    }

    #[test]
    fn test_multiple_iterators_from_same_audio() {
        // This test verifies that our raw pointer approach allows multiple iterators
        let audio = AudioSamples::new_multi_channel(
            array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
            sample_rate!(44100),
        )
        .unwrap();

        // This should compile and work correctly
        let frames = audio.frames();
        let channels = audio.channels();
        #[cfg(feature = "editing")]
        let windows = audio.windows(crate::nzu!(2), crate::nzu!(1));

        // Verify they all work independently
        assert_eq!(frames.len(), 3);
        assert_eq!(channels.len(), 2);
        #[cfg(feature = "editing")]
        assert_eq!(windows.len(), 3);
    }

    // ==============================
    // MUTABLE ITERATOR TESTS
    // ==============================

    #[test]
    fn test_frame_iterator_mut_stereo() {
        let mut audio = AudioSamples::new_multi_channel(
            array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
            sample_rate!(44100),
        )
        .unwrap();

        let expected = array![[0.5f32, 1.0, 1.5], [6.0, 7.5, 9.0]];

        // Apply different processing to each channel
        audio.apply_to_channel_data(|ch, channel_data| {
            let gain = if ch == 0 { 0.5 } else { 1.5 };
            for sample in channel_data {
                *sample *= gain;
            }
        });

        assert_eq!(audio.as_multi_channel().unwrap(), &expected);
    }

    #[test]
    fn test_frame_iterator_mut_individual_access() {
        let mut audio =
            AudioSamples::new_multi_channel(array![[1.0f32, 2.0], [3.0, 4.0]], sample_rate!(44100))
                .unwrap();

        let expected = array![[10.0f32, 20.0], [3.0, 4.0]];

        // Modify only the left channel (channel 0)
        audio.apply_to_channel_data(|ch, channel_data| {
            if ch == 0 {
                for sample in channel_data {
                    *sample *= 10.0;
                }
            }
            // Leave right channel unchanged
        });

        assert_eq!(audio.as_multi_channel().unwrap(), &expected);
    }

    #[test]
    fn test_channel_iterator_mut_mono() {
        let mut audio =
            AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0], sample_rate!(44100)).unwrap();

        audio.apply_to_channel_data(|_ch, channel_data| {
            for sample in channel_data {
                *sample += 10.0;
            }
        });

        assert_eq!(audio.as_mono().unwrap(), &array![11.0f32, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_channel_iterator_mut_stereo() {
        let mut audio = AudioSamples::new_multi_channel(
            array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
            sample_rate!(44100),
        )
        .unwrap();

        let expected = array![[0.5f32, 1.0, 1.5], [8.0, 10.0, 12.0]];

        // Apply different processing to each channel
        audio.apply_to_channel_data(|ch, channel_data| {
            let gain = if ch == 0 { 0.5 } else { 2.0 };
            for sample in channel_data {
                *sample *= gain;
            }
        });

        assert_eq!(audio.as_multi_channel().unwrap(), &expected);
    }

    #[test]
    fn test_window_iterator_mut_mono() {
        let mut audio =
            AudioSamples::new_mono(array![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], sample_rate!(44100))
                .unwrap();

        // Apply windowed processing (non-overlapping)
        audio.apply_to_windows(3, 3, |_window_idx, window_data| {
            for sample in window_data {
                *sample *= 0.5;
            }
        });

        assert_eq!(
            audio.as_mono().unwrap(),
            &array![0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0]
        );
    }

    #[test]
    fn test_window_iterator_mut_stereo() {
        let mut audio = AudioSamples::new_multi_channel(
            array![[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            sample_rate!(44100),
        )
        .unwrap();

        let expected = array![[0.8f32, 1.6, 2.4, 3.2], [6.0, 7.2, 8.4, 9.6]];

        // Apply windowed processing (non-overlapping)
        // For multi-channel, apply_to_windows provides interleaved data
        audio.apply_to_windows(2, 2, |_window_idx, window_data| {
            // window_data is interleaved: [L0, R0, L1, R1, ...] for 2-sample window
            let samples_per_channel = window_data.len() / 2; // 2 channels
            for sample_idx in 0..samples_per_channel {
                let left_idx = sample_idx * 2;
                let right_idx = sample_idx * 2 + 1;
                window_data[left_idx] *= 0.8; // Left channel gain
                window_data[right_idx] *= 1.2; // Right channel gain
            }
        });

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
        let mut audio =
            AudioSamples::new_mono(array![1.0f32, 1.0, 1.0, 1.0], sample_rate!(44100)).unwrap();

        // Apply Hann window function
        audio.apply_to_windows(4, 4, |_window_idx, window_data| {
            let window_size = window_data.len();
            for (i, sample) in window_data.iter_mut().enumerate() {
                let hann_weight = 0.5
                    * (1.0
                        - (2.0 * std::f32::consts::PI * i as f32 / (window_size - 1) as f32).cos());
                *sample *= hann_weight;
            }
        });

        // Check that Hann window was applied (values should be different from 1.0)
        let result = audio.as_mono().unwrap();
        assert!(result[0] < 1.0); // Should be close to 0
        assert!(result[1] > 0.5); // Should be around 0.75
        assert!(result[2] > 0.5); // Should be around 0.75
        assert!(result[3] < 1.0); // Should be close to 0
    }

    #[test]
    fn test_performance_comparison_apply_vs_iterator() {
        // This test demonstrates when to use each approach
        let mut audio1 =
            AudioSamples::new_mono(Array1::<f32>::ones(1000), sample_rate!(44100)).unwrap();
        let mut audio2 = audio1.clone();

        // Method 1: Using optimized apply (recommended for simple operations)
        audio1.apply(|sample| sample * 0.5);

        // Method 2: Using convenience method (alternative for complex operations)
        audio2.apply_to_frames(|_frame_idx, frame_data| {
            for sample in frame_data {
                *sample *= 0.5;
            }
        });

        // Results should be identical
        assert_eq!(audio1.as_mono().unwrap(), audio2.as_mono().unwrap());
    }
}
