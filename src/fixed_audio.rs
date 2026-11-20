//! Audio buffers whose frame count, and for multi-channel buffers channel count,
//! are const generic parameters.
//!
//! [`FixedSizeAudioSamples<T, N>`] holds `N` mono frames and
//! [`FixedSizeMultiChannelAudioSamples<T, N, C>`] holds `N` frames of `C`
//! channels. Use them where the block size is fixed when the program is written,
//! such as the processing block of a streaming pipeline, so that the size is part
//! of a function's signature. Both dereference to [`AudioSamples`], so every
//! operation is available on them, and [`swap`](FixedSizeAudioSamples::swap)
//! exchanges two buffers without copying (double buffering). The samples are
//! stored on the heap, like any other `AudioSamples`.
//!
//! ```rust
//! use audio_samples::{FixedSizeAudioSamples, FixedSizeMultiChannelAudioSamples, sample_rate};
//!
//! let mut front: FixedSizeAudioSamples<f32, 512> = FixedSizeAudioSamples::zeros(sample_rate!(48000));
//! let mut back = FixedSizeAudioSamples::<f32, 512>::zeros(sample_rate!(48000));
//! back[0] = 1.0;
//! front.swap(&mut back);
//! assert_eq!(front[0], 1.0);
//! assert_eq!(front.samples_per_channel().get(), 512);
//!
//! let stereo = FixedSizeMultiChannelAudioSamples::<f32, 256, 2>::zeros(sample_rate!(48000));
//! assert_eq!(stereo.capacity(), 512);
//! ```
//!
//! ## Limitations
//!
//! The size is fixed at construction. Mutable access goes through
//! `AudioSamples`, so an in-place operation that changes the length (trimming,
//! padding, resampling) leaves a buffer whose length no longer matches `N`.

use core::num::{NonZeroU32, NonZeroUsize};
use std::ops::{Deref, DerefMut};

use non_empty_slice::NonEmptySlice;

use crate::error::AudioSampleResult;
use crate::repr::{AudioSamples, SampleRate};
use crate::traits::StandardSample;

/// A mono [`AudioSamples`] buffer with a compile-time frame count `N`.
///
/// Encodes the buffer size in the type, making memory usage explicit and verifiable at
/// compile time. The underlying storage is heap-allocated. Use this type for fixed-size
/// mono buffers in streaming pipelines or DSP blocks with a known frame size.
pub struct FixedSizeAudioSamples<T, const N: usize>
where
    T: StandardSample,
{
    samples: AudioSamples<'static, T>,
}

impl<T: StandardSample, const N: usize> FixedSizeAudioSamples<T, N> {
    /// Creates a zero-filled mono buffer at the given sample rate.
    ///
    /// Panics at compile time if `N` is zero.
    #[inline]
    #[must_use]
    pub fn zeros(sample_rate: SampleRate) -> Self {
        const { assert!(N > 0, "N must be greater than 0") };
        // SAFETY: N > 0 asserted above
        let len = unsafe { NonZeroUsize::new_unchecked(N) };
        Self {
            samples: AudioSamples::zeros_mono(len, sample_rate),
        }
    }

    /// Creates a mono buffer from a slice, copying all elements into owned storage.
    #[inline]
    pub fn from_1d<D: AsRef<NonEmptySlice<T>>>(
        data: D,
        sample_rate: SampleRate,
    ) -> AudioSampleResult<Self> {
        let samples = AudioSamples::from_mono_vec(data.as_ref().to_non_empty_vec(), sample_rate);
        Ok(Self { samples })
    }

    /// The compile-time frame count of this buffer.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Returns a reference to the inner [`AudioSamples`].
    #[inline]
    #[must_use]
    pub fn samples(&self) -> &AudioSamples<'static, T> {
        &self.samples
    }

    /// Returns a mutable reference to the inner [`AudioSamples`].
    #[inline]
    pub fn samples_mut(&mut self) -> &mut AudioSamples<'static, T> {
        &mut self.samples
    }

    /// Swaps the underlying buffers of two instances.
    ///
    /// Safe because matching type parameters guarantee the same sample type and frame count.
    #[inline]
    pub fn swap(&mut self, other: &mut Self) {
        std::mem::swap(&mut self.samples, &mut other.samples);
    }
}

impl<T: StandardSample, const N: usize> Deref for FixedSizeAudioSamples<T, N> {
    type Target = AudioSamples<'static, T>;
    fn deref(&self) -> &Self::Target {
        &self.samples
    }
}

impl<T: StandardSample, const N: usize> DerefMut for FixedSizeAudioSamples<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.samples
    }
}

impl<T: StandardSample, const N: usize> AsRef<AudioSamples<'static, T>>
    for FixedSizeAudioSamples<T, N>
{
    fn as_ref(&self) -> &AudioSamples<'static, T> {
        &self.samples
    }
}

impl<T: StandardSample, const N: usize> AsMut<AudioSamples<'static, T>>
    for FixedSizeAudioSamples<T, N>
{
    fn as_mut(&mut self) -> &mut AudioSamples<'static, T> {
        &mut self.samples
    }
}

/// A multi-channel [`AudioSamples`] buffer with compile-time frame count `N` and channel
/// count `C`.
///
/// Both dimensions are encoded in the type, making buffer geometry explicit and verifiable
/// at compile time. Use this type for fixed-size multi-channel buffers in streaming
/// pipelines, where sample rate, channel count, and chunk size are all known at construction.
pub struct FixedSizeMultiChannelAudioSamples<T, const N: usize, const C: usize>
where
    T: StandardSample,
{
    samples: AudioSamples<'static, T>,
}

impl<T: StandardSample, const N: usize, const C: usize> FixedSizeMultiChannelAudioSamples<T, N, C> {
    /// Creates a zero-filled multi-channel buffer at the given sample rate.
    ///
    /// Panics at compile time if `N` (frames) or `C` (channels) is zero.
    #[inline]
    #[must_use]
    pub fn zeros(sample_rate: SampleRate) -> Self {
        const { assert!(N > 0, "N (frames per channel) must be greater than 0") };
        const { assert!(C > 0, "C (channel count) must be greater than 0") };
        // SAFETY: N > 0 and C > 0 asserted above
        let frames = unsafe { NonZeroUsize::new_unchecked(N) };
        // SAFETY: C > 0 asserted above, so C as u32 is non-zero
        let channels = unsafe { NonZeroU32::new_unchecked(C as u32) };
        Self {
            samples: AudioSamples::zeros_multi_channel(channels, frames, sample_rate),
        }
    }

    /// The compile-time number of frames per channel.
    #[inline]
    #[must_use]
    pub const fn frames(&self) -> usize {
        N
    }

    /// The compile-time number of channels.
    #[inline]
    #[must_use]
    pub const fn channels(&self) -> usize {
        C
    }

    /// Total sample capacity: `N * C`.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        N * C
    }

    /// Returns a reference to the inner [`AudioSamples`].
    #[inline]
    #[must_use]
    pub fn samples(&self) -> &AudioSamples<'static, T> {
        &self.samples
    }

    /// Returns a mutable reference to the inner [`AudioSamples`].
    #[inline]
    pub fn samples_mut(&mut self) -> &mut AudioSamples<'static, T> {
        &mut self.samples
    }

    /// Swaps the underlying buffers of two instances.
    ///
    /// Safe because matching type parameters guarantee the same geometry.
    #[inline]
    pub fn swap(&mut self, other: &mut Self) {
        std::mem::swap(&mut self.samples, &mut other.samples);
    }
}

impl<T: StandardSample, const N: usize, const C: usize> Deref
    for FixedSizeMultiChannelAudioSamples<T, N, C>
{
    type Target = AudioSamples<'static, T>;
    fn deref(&self) -> &Self::Target {
        &self.samples
    }
}

impl<T: StandardSample, const N: usize, const C: usize> DerefMut
    for FixedSizeMultiChannelAudioSamples<T, N, C>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.samples
    }
}

impl<T: StandardSample, const N: usize, const C: usize> AsRef<AudioSamples<'static, T>>
    for FixedSizeMultiChannelAudioSamples<T, N, C>
{
    fn as_ref(&self) -> &AudioSamples<'static, T> {
        &self.samples
    }
}

impl<T: StandardSample, const N: usize, const C: usize> AsMut<AudioSamples<'static, T>>
    for FixedSizeMultiChannelAudioSamples<T, N, C>
{
    fn as_mut(&mut self) -> &mut AudioSamples<'static, T> {
        &mut self.samples
    }
}
