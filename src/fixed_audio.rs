//! Audio buffers whose frame count, and for multi-channel buffers channel count,
//! are const generic parameters.
//!
//! [`FixedSizeAudioSamples<T, N>`] holds `N` mono frames and
//! [`FixedSizeMultiChannelAudioSamples<T, N, C>`] holds `N` frames of `C`
//! channels. Use them where the block size is fixed when the program is written,
//! such as the processing block of a streaming pipeline, so that the size is part
//! of a function's signature. The samples are stored on the heap, like any other
//! `AudioSamples`.
//!
//! Both dereference to [`AudioSamples`] for reading, so every operation that
//! takes `&self` is available on them. Mutable access cannot change the size:
//! indexing, [`view_mut`](FixedSizeAudioSamples::view_mut),
//! [`apply`](FixedSizeAudioSamples::apply) and
//! [`copy_from`](FixedSizeAudioSamples::copy_from) write samples in place, and
//! [`modify`](FixedSizeAudioSamples::modify) runs any in-place `AudioSamples`
//! operation and checks that the size is unchanged afterwards.
//! [`swap`](FixedSizeAudioSamples::swap) exchanges two buffers without copying
//! (double buffering).
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
//! let mut stereo = FixedSizeMultiChannelAudioSamples::<f32, 256, 2>::zeros(sample_rate!(48000));
//! stereo[(1, 0)] = 0.5;
//! stereo.apply(|x| x * 2.0);
//! assert_eq!(stereo[(1, 0)], 1.0);
//! assert_eq!(stereo.capacity(), 512);
//! ```
//!
//! Operations that change the length in place are not available, so the size in
//! the type always holds:
//!
//! ```rust,compile_fail,E0596
//! use audio_samples::{FixedSizeAudioSamples, sample_rate};
//!
//! let mut block = FixedSizeAudioSamples::<f32, 512>::zeros(sample_rate!(48000));
//! // `AudioSamples::replace_with_mono` could make the buffer 3 samples long, but the
//! // buffer only dereferences to `AudioSamples` for reading: this does not compile.
//! block.replace_with_mono(ndarray::Array1::zeros(3)).unwrap();
//! ```

use core::num::{NonZeroU32, NonZeroUsize};
use std::ops::{Deref, Index, IndexMut};

use ndarray::{ArrayViewMut1, ArrayViewMut2, s};
use non_empty_slice::NonEmptySlice;

use crate::error::{AudioSampleError, AudioSampleResult, LayoutError, ParameterError};
use crate::repr::{AudioSamples, SampleRate};
use crate::traits::StandardSample;

/// Error for a source whose shape differs from a fixed-size buffer's.
fn shape_error(
    operation: &str,
    expected: (usize, usize),
    actual: (usize, usize),
) -> AudioSampleError {
    AudioSampleError::Layout(LayoutError::dimension_mismatch(
        format!("({}, {})", expected.0, expected.1),
        format!("({}, {})", actual.0, actual.1),
        operation,
    ))
}

/// `(channels, frames)` of `audio`.
fn shape<T: StandardSample>(audio: &AudioSamples<'_, T>) -> (usize, usize) {
    (
        audio.num_channels().get() as usize,
        audio.samples_per_channel().get(),
    )
}

/// A mono [`AudioSamples`] buffer with a compile-time frame count `N`.
///
/// Encodes the buffer size in the type, making memory usage explicit and verifiable at
/// compile time. The underlying storage is heap-allocated. Use this type for fixed-size
/// mono buffers in streaming pipelines or DSP blocks with a known frame size.
///
/// The buffer always holds exactly `N` mono samples: it dereferences to
/// [`AudioSamples`] for reading only, and its mutable methods keep the length.
pub struct FixedSizeAudioSamples<T, const N: usize>
where
    T: StandardSample,
{
    samples: AudioSamples<'static, T>,
}

impl<T: StandardSample, const N: usize> FixedSizeAudioSamples<T, N> {
    /// Creates a zero-filled mono buffer at the given sample rate.
    ///
    /// Fails to compile if `N` is zero.
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

    /// Creates a mono buffer from a slice of exactly `N` samples, copying them into owned
    /// storage.
    ///
    /// # Errors
    ///
    /// Returns [`AudioSampleError::Parameter`] if `data` does not hold exactly `N` samples.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::{FixedSizeAudioSamples, sample_rate};
    /// use non_empty_slice::NonEmptySlice;
    ///
    /// let data = [0.1f32, 0.2, 0.3, 0.4];
    /// let block = FixedSizeAudioSamples::<f32, 4>::from_1d(
    ///     NonEmptySlice::new(&data).unwrap(), sample_rate!(8000)).unwrap();
    /// assert_eq!(block[3], 0.4);
    ///
    /// // Three samples do not fit a four-sample buffer.
    /// let short = NonEmptySlice::new(&data[..3]).unwrap();
    /// assert!(FixedSizeAudioSamples::<f32, 4>::from_1d(short, sample_rate!(8000)).is_err());
    /// ```
    #[inline]
    pub fn from_1d<D: AsRef<NonEmptySlice<T>>>(
        data: D,
        sample_rate: SampleRate,
    ) -> AudioSampleResult<Self> {
        let data = data.as_ref();
        if data.len().get() != N {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "data",
                format!("expected {N} samples, got {}", data.len()),
            )));
        }
        let samples = AudioSamples::from_mono_vec(data.to_non_empty_vec(), sample_rate);
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

    /// Consumes the buffer and returns the inner [`AudioSamples`].
    #[inline]
    #[must_use]
    pub fn into_inner(self) -> AudioSamples<'static, T> {
        self.samples
    }

    /// A mutable view of the `N` samples.
    #[inline]
    pub fn view_mut(&mut self) -> ArrayViewMut1<'_, T> {
        match self.samples.as_mono_mut() {
            Some(mono) => mono.slice_mut(s![..]),
            // The constructors make mono data and `modify` keeps it mono.
            None => unreachable!("FixedSizeAudioSamples holds mono data"),
        }
    }

    /// A mutable slice of the `N` samples, or `None` if the storage is not contiguous.
    #[inline]
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        self.samples.as_slice_mut()
    }

    /// Replaces each sample `x` with `f(x)`.
    #[inline]
    pub fn apply<F>(&mut self, f: F)
    where
        F: Fn(T) -> T,
    {
        self.samples.apply(f);
    }

    /// Replaces each sample `x` at index `i` with `f(i, x)`.
    #[inline]
    pub fn apply_with_index<F>(&mut self, f: F)
    where
        F: Fn(usize, T) -> T,
    {
        self.samples.apply_with_index(f);
    }

    /// Copies the samples of `source`, which must be mono with `N` samples, into this buffer.
    /// The sample rate is unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`AudioSampleError::Layout`] if `source` is not one channel of `N` samples;
    /// the buffer is then unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::time::Duration;
    /// use audio_samples::{FixedSizeAudioSamples, sample_rate, sine_wave};
    ///
    /// let tone = sine_wave::<f32>(440.0, Duration::from_millis(1), sample_rate!(48000), 0.5);
    /// let mut block = FixedSizeAudioSamples::<f32, 48>::zeros(sample_rate!(48000));
    /// block.copy_from(&tone).unwrap();
    /// assert_eq!(block[10], tone[10]);
    /// ```
    #[inline]
    pub fn copy_from(&mut self, source: &AudioSamples<'_, T>) -> AudioSampleResult<()> {
        let actual = shape(source);
        let src = match source.as_mono() {
            Some(mono) if actual.1 == N => mono.view(),
            _ => return Err(shape_error("copy_from", (1, N), actual)),
        };
        self.view_mut().assign(&src);
        Ok(())
    }

    /// Runs `f` with mutable access to the inner [`AudioSamples`], for in-place operations
    /// that keep the length (gain, filtering, normalisation, fades, ...).
    ///
    /// # Errors
    ///
    /// Returns [`AudioSampleError::Layout`] if `f` left anything other than `N` mono
    /// samples. The buffer is then reset to `N` zeros at its sample rate, so its size
    /// still matches the type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::{FixedSizeAudioSamples, sample_rate};
    ///
    /// let mut block = FixedSizeAudioSamples::<f32, 64>::zeros(sample_rate!(48000));
    /// block[0] = 0.25;
    /// # #[cfg(feature = "processing")] {
    /// use audio_samples::AudioProcessing;
    /// block.modify(|audio| audio.scale_in_place(2.0)).unwrap();
    /// assert_eq!(block[0], 0.5);
    /// # }
    ///
    /// // An in-place operation that changes the length is an error.
    /// let shorter = block.modify(|audio| audio.replace_with_mono(ndarray::Array1::zeros(3)));
    /// assert!(shorter.is_err());
    /// assert_eq!(block.samples_per_channel().get(), 64);
    /// ```
    #[inline]
    pub fn modify<R, F>(&mut self, f: F) -> AudioSampleResult<R>
    where
        F: FnOnce(&mut AudioSamples<'static, T>) -> R,
    {
        let out = f(&mut self.samples);
        let actual = shape(&self.samples);
        if self.samples.is_mono() && actual.1 == N {
            Ok(out)
        } else {
            let sample_rate = self.samples.sample_rate();
            *self = Self::zeros(sample_rate);
            Err(shape_error("modify", (1, N), actual))
        }
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

impl<T: StandardSample, const N: usize> AsRef<AudioSamples<'static, T>>
    for FixedSizeAudioSamples<T, N>
{
    fn as_ref(&self) -> &AudioSamples<'static, T> {
        &self.samples
    }
}

impl<T: StandardSample, const N: usize> Index<usize> for FixedSizeAudioSamples<T, N> {
    type Output = T;

    /// The sample at `index`.
    ///
    /// # Panics
    ///
    /// If `index >= N`, as for slices.
    fn index(&self, index: usize) -> &T {
        &self.samples[index]
    }
}

impl<T: StandardSample, const N: usize> IndexMut<usize> for FixedSizeAudioSamples<T, N> {
    /// The sample at `index`.
    ///
    /// # Panics
    ///
    /// If `index >= N`, as for slices.
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.samples[index]
    }
}

/// A multi-channel [`AudioSamples`] buffer with compile-time frame count `N` and channel
/// count `C`.
///
/// Both dimensions are encoded in the type, making buffer geometry explicit and verifiable
/// at compile time. Use this type for fixed-size multi-channel buffers in streaming
/// pipelines, where sample rate, channel count, and chunk size are all known at construction.
///
/// The buffer always holds `C` channels of `N` samples: it dereferences to
/// [`AudioSamples`] for reading only, and its mutable methods keep the shape.
pub struct FixedSizeMultiChannelAudioSamples<T, const N: usize, const C: usize>
where
    T: StandardSample,
{
    samples: AudioSamples<'static, T>,
}

impl<T: StandardSample, const N: usize, const C: usize> FixedSizeMultiChannelAudioSamples<T, N, C> {
    /// Creates a zero-filled multi-channel buffer at the given sample rate.
    ///
    /// Fails to compile if `N` (frames) or `C` (channels) is zero.
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

    /// Consumes the buffer and returns the inner [`AudioSamples`].
    #[inline]
    #[must_use]
    pub fn into_inner(self) -> AudioSamples<'static, T> {
        self.samples
    }

    /// A mutable `(C, N)` view of the samples (channels by frames).
    #[inline]
    pub fn view_mut(&mut self) -> ArrayViewMut2<'_, T> {
        match self.samples.as_multi_channel_mut() {
            Some(multi) => multi.view_mut(),
            // The constructor makes multi-channel data and `modify` keeps it so.
            None => unreachable!("FixedSizeMultiChannelAudioSamples holds multi-channel data"),
        }
    }

    /// A mutable slice of the `N * C` samples in planar order (all of channel 0, then
    /// channel 1, ...), or `None` if the storage is not contiguous.
    #[inline]
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        self.samples.as_slice_mut()
    }

    /// Replaces each sample `x` with `f(x)`.
    #[inline]
    pub fn apply<F>(&mut self, f: F)
    where
        F: Fn(T) -> T,
    {
        self.samples.apply(f);
    }

    /// Copies the samples of `source`, which must have `C` channels of `N` samples, into
    /// this buffer. The sample rate is unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`AudioSampleError::Layout`] if `source` has another shape; the buffer is
    /// then unchanged.
    #[inline]
    pub fn copy_from(&mut self, source: &AudioSamples<'_, T>) -> AudioSampleResult<()> {
        let actual = shape(source);
        if actual != (C, N) {
            return Err(shape_error("copy_from", (C, N), actual));
        }
        match (source.as_multi_channel(), source.as_mono()) {
            (Some(multi), _) => self.view_mut().assign(&multi.view()),
            (None, Some(mono)) => self.view_mut().row_mut(0).assign(&mono.view()),
            (None, None) => return Err(shape_error("copy_from", (C, N), actual)),
        }
        Ok(())
    }

    /// Runs `f` with mutable access to the inner [`AudioSamples`], for in-place operations
    /// that keep the shape (gain, filtering, normalisation, fades, ...).
    ///
    /// # Errors
    ///
    /// Returns [`AudioSampleError::Layout`] if `f` left anything other than `C` channels of
    /// `N` samples in multi-channel layout. The buffer is then reset to zeros at its sample
    /// rate, so its shape still matches the type.
    #[inline]
    pub fn modify<R, F>(&mut self, f: F) -> AudioSampleResult<R>
    where
        F: FnOnce(&mut AudioSamples<'static, T>) -> R,
    {
        let out = f(&mut self.samples);
        let actual = shape(&self.samples);
        if self.samples.is_multi_channel() && actual == (C, N) {
            Ok(out)
        } else {
            let sample_rate = self.samples.sample_rate();
            *self = Self::zeros(sample_rate);
            Err(shape_error("modify", (C, N), actual))
        }
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

impl<T: StandardSample, const N: usize, const C: usize> AsRef<AudioSamples<'static, T>>
    for FixedSizeMultiChannelAudioSamples<T, N, C>
{
    fn as_ref(&self) -> &AudioSamples<'static, T> {
        &self.samples
    }
}

impl<T: StandardSample, const N: usize, const C: usize> Index<(usize, usize)>
    for FixedSizeMultiChannelAudioSamples<T, N, C>
{
    type Output = T;

    /// The sample at `(channel, frame)`.
    ///
    /// # Panics
    ///
    /// If `channel >= C` or `frame >= N`, as for slices.
    fn index(&self, index: (usize, usize)) -> &T {
        &self.samples[index]
    }
}

impl<T: StandardSample, const N: usize, const C: usize> IndexMut<(usize, usize)>
    for FixedSizeMultiChannelAudioSamples<T, N, C>
{
    /// The sample at `(channel, frame)`.
    ///
    /// # Panics
    ///
    /// If `channel >= C` or `frame >= N`, as for slices.
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        &mut self.samples[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample_rate;

    #[test]
    fn from_1d_checks_the_length() {
        let data = [1.0f32, 2.0, 3.0];
        let slice = NonEmptySlice::new(&data).unwrap();
        assert!(FixedSizeAudioSamples::<f32, 3>::from_1d(slice, sample_rate!(8000)).is_ok());
        assert!(FixedSizeAudioSamples::<f32, 4>::from_1d(slice, sample_rate!(8000)).is_err());
        assert!(FixedSizeAudioSamples::<f32, 2>::from_1d(slice, sample_rate!(8000)).is_err());
    }

    #[test]
    fn modify_keeps_the_size() {
        let mut block = FixedSizeAudioSamples::<f32, 8>::zeros(sample_rate!(8000));
        block[1] = 1.0;
        block.modify(|a| a.apply(|x| x + 0.5)).unwrap();
        assert_eq!(block[1], 1.5);

        // An operation that shortens the buffer is an error and leaves N zeros.
        let err = block.modify(|a| {
            let shorter = ndarray::Array1::from_vec(vec![0.0f32; 3]);
            a.replace_with_mono(shorter)
        });
        assert!(err.is_err());
        assert_eq!(block.samples_per_channel().get(), 8);
        assert!(block.to_interleaved_vec().iter().all(|&x| x == 0.0));
        assert_eq!(block.sample_rate(), sample_rate!(8000));
    }

    #[test]
    fn multi_channel_modify_and_copy_keep_the_shape() {
        let mut block = FixedSizeMultiChannelAudioSamples::<f32, 4, 2>::zeros(sample_rate!(8000));
        block[(1, 3)] = 2.0;
        block.modify(|a| a.apply(|x| x * 2.0)).unwrap();
        assert_eq!(block[(1, 3)], 4.0);

        let err = block.modify(|a| {
            let one = ndarray::Array2::from_elem((1, 4), 0.0f32);
            *a = AudioSamples::new_multi_channel(one, sample_rate!(8000)).unwrap();
        });
        assert!(err.is_err());
        assert_eq!(block.num_channels().get(), 2);
        assert_eq!(block.samples_per_channel().get(), 4);

        let source = AudioSamples::new_multi_channel(
            ndarray::Array2::from_elem((2, 4), 0.25f32),
            sample_rate!(8000),
        )
        .unwrap();
        block.copy_from(&source).unwrap();
        assert!(block.to_interleaved_vec().iter().all(|&x| x == 0.25));
        let wrong = AudioSamples::new_multi_channel(
            ndarray::Array2::from_elem((2, 5), 0.5f32),
            sample_rate!(8000),
        )
        .unwrap();
        assert!(block.copy_from(&wrong).is_err());
        assert!(block.to_interleaved_vec().iter().all(|&x| x == 0.25));
    }

    #[test]
    fn mono_copy_from_checks_the_shape() {
        let mut block = FixedSizeAudioSamples::<f32, 4>::zeros(sample_rate!(8000));
        let five =
            AudioSamples::new_mono(ndarray::Array1::from_elem(5, 1.0f32), sample_rate!(8000))
                .unwrap();
        assert!(block.copy_from(&five).is_err());
        let four =
            AudioSamples::new_mono(ndarray::Array1::from_elem(4, 1.0f32), sample_rate!(8000))
                .unwrap();
        block.copy_from(&four).unwrap();
        assert!(block.to_interleaved_vec().iter().all(|&x| x == 1.0));
        block.view_mut()[2] = 3.0;
        assert_eq!(block[2], 3.0);
    }
}
