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
//! let mut audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0], 44100);
//!
//! // Apply gain to all samples
//! audio.apply(|sample| sample * 0.5);
//!
//! // Apply position-dependent processing  
//! audio.apply_with_index(|index, sample| {
//!     sample * (1.0 - index as f32 * 0.1) // Fade out
//! });
//!
//! // Process in overlapping windows (useful for FFT)
//! audio.apply_windowed(1024, 512, |window, _prev_window| {
//!     window.to_vec() // Identity function for demonstration
//! }).unwrap();
//! ```
//!
//! # Memory Layout
//!
//! ## Mono Audio (`AudioData::Mono`)
//! - Stored as `Array1<T>` (1-dimensional array)
//! - Direct memory access with `as_mono()` / `as_mono_mut()`
//! - Optimal for single-channel processing
//!
//! ## Multi-Channel Audio (`AudioData::Multi`)  
//! - Stored as `Array2<T>` with shape `(channels, samples_per_channel)`
//! - Each row represents one channel of audio data
//! - Enables efficient per-channel operations with `apply_to_channels()`
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
use core::fmt::{Display, Formatter, Result as FmtResult};

use core::num::{NonZeroU32, NonZeroUsize};
use ndarray::iter::AxisIterMut;
use ndarray::{
    Array1, Array2, ArrayView, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis, Ix1, Ix2, SliceArg
};
use std::any::TypeId;
use std::ops::{Bound, Deref, DerefMut, Index, IndexMut, Neg, RangeBounds};

/// Creates a [`NonZeroU32`] sample rate from a compile-time constant.
///
/// This macro provides zero-cost construction of sample rates with compile-time
/// validation that the value is non-zero.
///
/// # Examples
///
/// ```rust
/// use audio_samples::sample_rate;
///
/// // Common sample rates - validated at compile time
/// let cd_rate = sample_rate!(44100);
/// let dvd_rate = sample_rate!(48000);
/// let high_res = sample_rate!(96000);
///
/// // Use directly in AudioSamples constructors
/// use audio_samples::AudioSamples;
/// use ndarray::array;
/// let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0], sample_rate!(44100));
/// ```
///
/// # Compile-time Safety
///
/// The macro will cause a compile error if passed zero:
///
/// ```compile_fail
/// use audio_samples::sample_rate;
/// let invalid = sample_rate!(0); // Compile error!
/// ```
#[macro_export]
macro_rules! sample_rate {
    ($rate:expr) => {{
        const RATE: u32 = $rate;
        const { assert!(RATE > 0, "sample rate must be greater than 0") };
        // SAFETY: We just asserted RATE > 0 at compile time
        unsafe { ::core::num::NonZeroU32::new_unchecked(RATE) }
    }};
}

#[cfg(feature = "core-ops")]
use crate::operations::ProcessingBuilder;
use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioTypeConversion, ChannelLayout,
    ConvertTo, I24, LayoutError, ParameterError, RealFloat, to_precision,
};

/// Borrowed-or-owned byte view returned by `AudioData::bytes`.
#[derive(Debug, Clone)]
pub enum AudioBytes<'a> {
    /// Zero-copy view into the underlying contiguous buffer.
    Borrowed(&'a [u8]),
    /// Owned buffer when a borrow is impossible (e.g., I24 packing).
    Owned(Vec<u8>),
}

impl<'a> AudioBytes<'a> {
    /// Returns the byte slice regardless of ownership mode.
    pub const fn as_slice(&self) -> &[u8] {
        match self {
            AudioBytes::Borrowed(b) => b,
            AudioBytes::Owned(v) => v.as_slice(),
        }
    }

    /// Returns an owned `Vec<u8>`, cloning if necessary.
    pub fn into_owned(self) -> Vec<u8> {
        match self {
            AudioBytes::Borrowed(b) => b.to_vec(),
            AudioBytes::Owned(v) => v,
        }
    }
}

/// Audio sample data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SampleType {
    /// 16-bit signed integer
    I16,
    /// 24-bit signed integer
    I24,
    /// 32-bit signed integer
    I32,
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// Unknown or unsupported sample type
    Unknown,
}

impl SampleType {
    /// Returns the canonical short name (machine-oriented)
    pub const fn as_str(self) -> &'static str {
        match self {
            SampleType::I16 => "i16",
            SampleType::I24 => "i24",
            SampleType::I32 => "i32",
            SampleType::F32 => "f32",
            SampleType::F64 => "f64",
            SampleType::Unknown => "unknown",
        }
    }

    /// Returns a human-readable descriptive name
    pub const fn description(self) -> &'static str {
        match self {
            SampleType::I16 => "16-bit signed integer",
            SampleType::I24 => "24-bit signed integer",
            SampleType::I32 => "32-bit signed integer",
            SampleType::F32 => "32-bit floating point",
            SampleType::F64 => "64-bit floating point",
            SampleType::Unknown => "unknown or unsupported format",
        }
    }

    /// Returns the bit-depth of the format, if defined
    pub const fn bits(self) -> Option<u32> {
        match self {
            SampleType::I16 => Some(16),
            SampleType::I24 => Some(24),
            SampleType::I32 | SampleType::F32 => Some(32),
            SampleType::F64 => Some(64),
            SampleType::Unknown => None,
        }
    }

    /// Returns the size in bytes of one sample, if defined
    pub const fn bytes(self) -> Option<usize> {
        match self {
            SampleType::I16 => Some(2),
            SampleType::I24 => Some(3),
            SampleType::I32 | SampleType::F32 => Some(4),
            SampleType::F64 => Some(8),
            SampleType::Unknown => None,
        }
    }

    /// True if this is an integer-based sample format
    pub const fn is_integer(self) -> bool {
        matches!(self, SampleType::I16 | SampleType::I24 | SampleType::I32)
    }

    /// True if this is a floating-point sample format
    pub const fn is_float(self) -> bool {
        matches!(self, SampleType::F32 | SampleType::F64)
    }

    /// True if this format is usable for DSP operations without conversion
    pub const fn is_dsp_native(self) -> bool {
        matches!(self, SampleType::F32 | SampleType::F64)
    }
}

impl SampleType {
    /// Returns the SampleType corresponding to a TypeId, or None if unrecognized.
    ///
    /// This is useful for SIMD dispatch code that needs to determine the
    /// concrete sample type at runtime.
    ///
    /// # Example
    /// ```
    /// use audio_samples::SampleType;
    /// use std::any::TypeId;
    ///
    /// assert_eq!(SampleType::from_type_id(TypeId::of::<f32>()), Some(SampleType::F32));
    /// assert_eq!(SampleType::from_type_id(TypeId::of::<String>()), None);
    /// ```
    pub fn from_type_id(type_id: std::any::TypeId) -> Option<Self> {
        use std::any::TypeId;
        if type_id == TypeId::of::<i16>() {
            Some(SampleType::I16)
        } else if type_id == TypeId::of::<I24>() {
            Some(SampleType::I24)
        } else if type_id == TypeId::of::<i32>() {
            Some(SampleType::I32)
        } else if type_id == TypeId::of::<f32>() {
            Some(SampleType::F32)
        } else if type_id == TypeId::of::<f64>() {
            Some(SampleType::F64)
        } else {
            None
        }
    }

    /// Creates a sample type from a number of bits per sample.
    pub const fn from_bits(bits: u16) -> Self {
        match bits {
            16 => SampleType::I16,
            24 => SampleType::I24,
            32 => SampleType::I32,
            64 => SampleType::F64,
            _ => SampleType::Unknown,
        }
    }
}

impl Display for SampleType {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        if f.alternate() {
            // "{:#}" → human-readable description
            write!(f, "{}", self.description())
        } else {
            // "{}" → compact canonical form
            write!(f, "{}", self.as_str())
        }
    }
}

impl TryFrom<&str> for SampleType {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "i16" => Ok(SampleType::I16),
            "i24" => Ok(SampleType::I24),
            "i32" => Ok(SampleType::I32),
            "f32" => Ok(SampleType::F32),
            "f64" => Ok(SampleType::F64),
            _ => Err(()),
        }
    }
}

/// Storage wrapper for mono (single-channel) audio samples.
///
/// `MonoData` can either **borrow** an `ndarray::ArrayView1` / `ArrayViewMut1` or **own** an
/// `ndarray::Array1`. Methods that require mutation will promote borrowed data to an owned
/// buffer when necessary.
///
/// # Indexing
/// Indexing uses the sample index (`usize`).
///
/// # Allocation behavior
/// - Pure reads are zero-copy.
/// - Some mutable operations (e.g. indexing mutably when currently borrowed immutably) may
///   allocate by cloning into an owned buffer.
#[derive(Debug, PartialEq)]
pub struct MonoData<'a, T: AudioSample>(MonoRepr<'a, T>);

/// Storage wrapper for multi-channel (planar) audio samples.
///
/// `MultiData` can either **borrow** an `ndarray::ArrayView2` / `ArrayViewMut2` or **own** an
/// `ndarray::Array2`.
///
/// The shape is `(channels, samples_per_channel)` (channels are rows).
///
/// # Indexing
/// Indexing uses `(channel, sample)`.
///
/// # Allocation behavior
/// - Pure reads are zero-copy.
/// - Some mutable operations may allocate by cloning into an owned buffer.
#[derive(Debug, PartialEq)]
pub struct MultiData<'a, T: AudioSample>(MultiRepr<'a, T>);

// For MonoData
impl<'a, T: AudioSample> Index<usize> for MonoData<'a, T> {
    type Output = T;
    fn index(&self, idx: usize) -> &Self::Output {
        match &self.0 {
            MonoRepr::Borrowed(arr) => &arr[idx],
            MonoRepr::BorrowedMut(arr) => &arr[idx],
            MonoRepr::Owned(arr) => &arr[idx],
        }
    }
}

impl<'a, T: AudioSample> IndexMut<usize> for MonoData<'a, T> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        self.promote();
        match &mut self.0 {
            MonoRepr::BorrowedMut(arr) => &mut arr[idx],
            MonoRepr::Owned(arr) => &mut arr[idx],
            _ => unreachable!("Self should have been converted to owned by now"),
        }
    }
}

// For MultiData
impl<'a, T: AudioSample> Index<(usize, usize)> for MultiData<'a, T> {
    type Output = T;
    fn index(&self, (ch, s): (usize, usize)) -> &Self::Output {
        match &self.0 {
            MultiRepr::Borrowed(arr) => &arr[[ch, s]],
            MultiRepr::BorrowedMut(arr) => &arr[[ch, s]],
            MultiRepr::Owned(arr) => &arr[[ch, s]],
        }
    }
}

impl<'a, T: AudioSample> IndexMut<(usize, usize)> for MultiData<'a, T> {
    fn index_mut(&mut self, (ch, s): (usize, usize)) -> &mut Self::Output {
        self.promote();
        match &mut self.0 {
            MultiRepr::BorrowedMut(arr) => &mut arr[[ch, s]],
            MultiRepr::Owned(arr) => &mut arr[[ch, s]],
            _ => unreachable!("Self should have been converted to owned by now"),
        }
    }
}

// Support [usize; 2] indexing for MultiData
impl<'a, T: AudioSample> Index<[usize; 2]> for MultiData<'a, T> {
    type Output = T;
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        match &self.0 {
            MultiRepr::Borrowed(arr) => &arr[index],
            MultiRepr::BorrowedMut(arr) => &arr[index],
            MultiRepr::Owned(arr) => &arr[index],
        }
    }
}

impl<'a, T: AudioSample> IndexMut<[usize; 2]> for MultiData<'a, T> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        self.promote();
        match &mut self.0 {
            MultiRepr::BorrowedMut(arr) => &mut arr[index],
            MultiRepr::Owned(arr) => &mut arr[index],
            _ => unreachable!("Self should have been converted to owned by now"),
        }
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum MonoRepr<'a, T: AudioSample> {
    Borrowed(ArrayView1<'a, T>),
    BorrowedMut(ArrayViewMut1<'a, T>),
    Owned(Array1<T>),
}

#[derive(Debug, PartialEq)]
pub(crate) enum MultiRepr<'a, T: AudioSample> {
    Borrowed(ArrayView2<'a, T>),
    BorrowedMut(ArrayViewMut2<'a, T>),
    Owned(Array2<T>),
}

impl<'a, T: AudioSample> MonoData<'a, T> {
    #[inline]
    /// Returns an `ndarray` view of the samples.
    ///
    /// This is always a zero-copy operation.
    pub fn as_view(&self) -> ArrayView1<'_, T> {
        match &self.0 {
            MonoRepr::Borrowed(v) => *v,
            MonoRepr::BorrowedMut(v) => v.view(),
            MonoRepr::Owned(a) => a.view(),
        }
    }

    /// Promotes immutably-borrowed data to owned data.
    ///
    /// If this `MonoData` is already mutable-borrowed or owned, this is a no-op.
    /// If it is immutably borrowed, this allocates and clones the underlying samples.
    pub fn promote(&mut self) {
        if let MonoRepr::Borrowed(v) = &self.0 {
            self.0 = MonoRepr::Owned(v.to_owned());
        }
    }

    /// Create MonoData from an ArrayView1 (borrowed data)
    pub const fn from_view<'b>(view: ArrayView1<'b, T>) -> Self
    where
        'b: 'a,
    {
        MonoData(MonoRepr::Borrowed(view))
    }

    /// Create MonoData from an ArrayViewMut1 (borrowed data)
    pub const fn from_view_mut<'b>(view: ArrayViewMut1<'b, T>) -> Self
    where
        'b: 'a,
    {
        MonoData(MonoRepr::BorrowedMut(view))
    }

    /// Create MonoData from an owned Array1 (owned data)
    pub const fn from_owned(array: Array1<T>) -> Self {
        MonoData(MonoRepr::Owned(array))
    }

    #[inline]
    /// Get a mutable view of the audio data, converting to owned if necessary.
    fn to_mut(&mut self) -> ArrayViewMut1<'_, T> {
        if let MonoRepr::Borrowed(v) = self.0 {
            // Convert borrowed to owned for mutability
            self.0 = MonoRepr::Owned(v.to_owned());
        }

        match &mut self.0 {
            MonoRepr::BorrowedMut(view) => view.view_mut(), // If the data  is already mutable borrowed then we do not need to convert to owned, this variant says "we have mutable access"
            MonoRepr::Owned(a) => a.view_mut(),
            _ => unreachable!(),
        }
    }

    #[inline]
    fn into_owned<'b>(self) -> MonoData<'b, T> {
        match self.0 {
            MonoRepr::Borrowed(v) => MonoData(MonoRepr::Owned(v.to_owned())),
            MonoRepr::BorrowedMut(v) => MonoData(MonoRepr::Owned(v.to_owned())),
            MonoRepr::Owned(a) => MonoData(MonoRepr::Owned(a)),
        }
    }

    // Delegation methods for ndarray operations
    #[inline]
    /// Returns the number of samples.
    pub fn len(&self) -> usize {
        self.as_view().len()
    }

    /// Returns an `ndarray` view of the samples.
    pub fn view(&self) -> ArrayView1<'_, T> {
        self.as_view()
    }

    /// Returns the arithmetic mean of the samples, or `None` if empty.
    pub fn mean(&self) -> Option<T> {
        self.as_view().mean()
    }

    /// Returns the sum of the samples.
    pub fn sum(&self) -> T {
        self.as_view().sum()
    }

    /// Folds over the samples.
    pub fn fold<F>(&self, init: T, f: F) -> T
    where
        F: FnMut(T, &T) -> T,
    {
        self.iter().fold(init, f)
    }

    /// Returns a slice view of the audio data based on the provided slicing information.
    /// Returns a slice view of the audio data based on the provided slicing information.
    pub fn slice<I>(&self, info: I) -> ArrayView1<'_, T>
    where
        I: SliceArg<Ix1, OutDim = Ix1>,
        I: SliceArg<Ix1, OutDim = Ix1>,
    {
        match &self.0 {
            MonoRepr::Borrowed(v) => v.slice(info),
            MonoRepr::BorrowedMut(v) => v.slice(info),
            MonoRepr::Owned(a) => a.slice(info),
        }
    }

    /// Returns a mutable slice view of the audio data based on the provided slicing information.
    /// NOTE: This function promotes to owned data if the current representation is borrowed.
    /// Returns a mutable slice view of the audio data based on the provided slicing information.
    /// NOTE: This function promotes to owned data if the current representation is borrowed.
    pub fn slice_mut<I>(&mut self, info: I) -> ArrayViewMut1<'_, T>
    where
        I: ndarray::SliceArg<Ix1, OutDim = Ix1>,
        I: ndarray::SliceArg<Ix1, OutDim = Ix1>,
    {
        if let MonoRepr::Borrowed(v) = self.0 {
            // Convert borrowed to owned for mutability
            self.0 = MonoRepr::Owned(v.to_owned());
        }

        match &mut self.0 {
            MonoRepr::BorrowedMut(a) => a.slice_mut(info),
            MonoRepr::Owned(a) => a.slice_mut(info),
            _ => unreachable!("Self should have been converted to owned by now"),
        }
    }

    /// Returns an iterator over the samples.
    pub fn iter(&self) -> ndarray::iter::Iter<'_, T, Ix1> {
        match &self.0 {
            MonoRepr::Borrowed(v) => v.iter(),
            MonoRepr::BorrowedMut(v) => v.iter(),
            MonoRepr::Owned(a) => a.iter(),
        }
    }

    /// Returns a mutable iterator over the samples.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    pub fn iter_mut(&mut self) -> ndarray::iter::IterMut<'_, T, Ix1> {
        if let MonoRepr::Borrowed(a) = &mut self.0 {
            self.0 = MonoRepr::Owned(a.to_owned());
        }

        match &mut self.0 {
            MonoRepr::BorrowedMut(b) => b.iter_mut(),
            MonoRepr::Owned(a) => a.iter_mut(),
            _ => unreachable!("Self should have been converted to owned by now"),
        }
    }

    /// Applies a value-mapping function in-place.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    pub fn mapv_inplace<F>(&mut self, f: F)
    where
        F: FnMut(T) -> T,
    {
        self.to_mut().mapv_inplace(f);
    }

    // pub const fn swap_axes(&mut self, _a: usize, _b: usize) {
    //     // For 1D arrays, swap_axes is a no-op
    // }

    /// Returns a mutable slice of the underlying samples.
    ///
    /// # Panics
    /// Panics if the underlying `ndarray` storage is not contiguous.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.promote();
        match &mut self.0 {
            MonoRepr::BorrowedMut(a) => a
                .as_slice_mut()
                .expect("Structures backing audio samples should be contiguous in memory"),
            MonoRepr::Owned(a) => a
                .as_slice_mut()
                .expect("Structures backing audio samples should be contiguous in memory"),
            _ => unreachable!("Self should have been converted to owned by now"),
        }
    }

    /// Returns a shared slice if the underlying storage is contiguous.
    pub fn as_slice(&self) -> Option<&[T]> {
        match &self.0 {
            MonoRepr::Borrowed(v) => v.as_slice(),
            MonoRepr::BorrowedMut(v) => v.as_slice(),
            MonoRepr::Owned(a) => a.as_slice(),
        }
    }

    /// Returns `true` if there are no samples.
    pub fn is_empty(&self) -> bool {
        self.as_view().is_empty()
    }

    /// Returns the shape of the underlying `ndarray` buffer.
    pub fn shape(&self) -> &[usize] {
        match &self.0 {
            MonoRepr::Borrowed(v) => v.shape(),
            MonoRepr::BorrowedMut(v) => v.shape(),
            MonoRepr::Owned(a) => a.shape(),
        }
    }

    /// Maps each sample into a new `Array1`.
    pub fn mapv<F, U>(&self, f: F) -> Array1<U>
    where
        F: Fn(T) -> U,
        U: Clone,
    {
        self.as_view().mapv(f)
    }

    /// Returns a mutable pointer to the underlying buffer.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.to_mut().as_mut_ptr()
    }

    /// Collects the samples into a `Vec<T>`.
    pub fn to_vec(&self) -> Vec<T> {
        self.as_view().to_vec()
    }

    /// Fills all samples with `value`.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    pub fn fill(&mut self, value: T) {
        self.to_mut().fill(value);
    }

    /// Converts into a raw `Vec<T>` and an offset.
    ///
    /// For borrowed data, this allocates and clones.
    pub fn into_raw_vec_and_offset(self) -> (Vec<T>, usize) {
        match self.0 {
            MonoRepr::Borrowed(v) => {
                let (vec, offset) = v.to_owned().into_raw_vec_and_offset();
                (vec, offset.unwrap_or(0))
            }
            MonoRepr::BorrowedMut(v) => {
                let (vec, offset) = v.to_owned().into_raw_vec_and_offset();
                (vec, offset.unwrap_or(0))
            }
            MonoRepr::Owned(a) => {
                let (vec, offset) = a.into_raw_vec_and_offset();
                (vec, offset.unwrap_or(0))
            }
        }
    }

    /// Converts this wrapper into an owned `Array1<T>`.
    ///
    /// For borrowed data, this allocates and clones.
    pub fn take(self) -> Array1<T> {
        match self.0 {
            MonoRepr::Borrowed(v) => v.to_owned(),
            MonoRepr::BorrowedMut(v) => v.to_owned(),
            MonoRepr::Owned(a) => a,
        }
    }
}

// PartialEq implementations for testing compatibility
impl<'a, T: AudioSample> PartialEq<Array1<T>> for MonoData<'a, T> {
    fn eq(&self, other: &Array1<T>) -> bool {
        self.as_view() == other.view()
    }
}

impl<'a, T: AudioSample> PartialEq<MonoData<'a, T>> for Array1<T> {
    fn eq(&self, other: &MonoData<'a, T>) -> bool {
        self.view() == other.as_view()
    }
}

// IntoIterator for MonoData
impl<'a, T: AudioSample> IntoIterator for &'a MonoData<'_, T> {
    type Item = &'a T;
    type IntoIter = ndarray::iter::Iter<'a, T, ndarray::Ix1>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_view().into_iter()
    }
}

impl<'a, T: AudioSample> MultiData<'a, T> {
    #[inline]
    /// Returns an `ndarray` view of the samples with shape `(channels, samples_per_channel)`.
    ///
    /// This is always a zero-copy operation.
    pub fn as_view(&self) -> ArrayView2<'_, T> {
        match &self.0 {
            MultiRepr::Borrowed(a) => *a,
            MultiRepr::BorrowedMut(a) => a.view(),
            MultiRepr::Owned(a) => a.view(),
        }
    }

    /// Promotes immutably-borrowed data to owned data.
    ///
    /// If this `MultiData` is already mutable-borrowed or owned, this is a no-op.
    /// If it is immutably borrowed, this allocates and clones the underlying samples.
    fn promote(&mut self) {
        if let MultiRepr::Borrowed(v) = &self.0 {
            self.0 = MultiRepr::Owned(v.to_owned());
        }
    }

    pub const fn from_view<'b>(view: ArrayView2<'b, T>) -> Self
    where
        'b: 'a,
    {
        MultiData(MultiRepr::Borrowed(view))
    }

    /// Create MultiData from an ArrayViewMut2 (borrowed mutable data)
    pub const fn from_view_mut<'b>(view: ArrayViewMut2<'b, T>) -> Self
    where
        'b: 'a,
    {
        MultiData(MultiRepr::BorrowedMut(view))
    }

    pub const fn from_owned(array: Array2<T>) -> Self {
        MultiData(MultiRepr::Owned(array))
    }

    #[inline]
    fn to_mut(&mut self) -> ArrayViewMut2<'_, T> {
        self.promote();
        match &mut self.0 {
            MultiRepr::BorrowedMut(a) => a.view_mut(),
            MultiRepr::Owned(a) => a.view_mut(),
            _ => unreachable!("Self should have been converted to owned by now"),
        }
    }
    #[inline]
    fn into_owned<'b>(self) -> MultiData<'b, T> {
        match self.0 {
            MultiRepr::Borrowed(v) => MultiData(MultiRepr::Owned(v.to_owned())),
            MultiRepr::BorrowedMut(v) => MultiData(MultiRepr::Owned(v.to_owned())),
            MultiRepr::Owned(a) => MultiData(MultiRepr::Owned(a)),
        }
    }

    // Delegation methods for ndarray operations
    #[inline]
    /// Returns the number of channels (rows).
    pub fn nrows(&self) -> usize {
        self.as_view().nrows()
    }
    /// Returns the number of columns (samples per channel) in multi-channel audio data.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.as_view().ncols()
    }
    #[inline]
    /// Returns `(channels, samples_per_channel)`.
    pub fn dim(&self) -> (usize, usize) {
        self.as_view().dim()
    }

    /// Returns the arithmetic mean along `axis`, or `None` if empty.
    pub fn mean_axis(&self, axis: Axis) -> Option<Array1<T>> {
        self.as_view().mean_axis(axis)
    }

    /// Returns the arithmetic mean across all samples, or `None` if empty.
    pub fn mean(&self) -> Option<T> {
        self.as_view().mean()
    }

    /// Returns the sum across all samples.
    pub fn sum(&self) -> T {
        self.as_view().sum()
    }

    /// Returns a 1D view into `axis` at `index`.
    pub fn index_axis(&self, axis: Axis, index: usize) -> ArrayView1<'_, T> {
        match &self.0 {
            MultiRepr::Borrowed(a) => a.index_axis(axis, index),
            MultiRepr::BorrowedMut(a) => a.index_axis(axis, index),
            MultiRepr::Owned(a) => a.index_axis(axis, index),
        }
    }

    /// Returns a view of the column at `index`.
    pub fn column(&self, index: usize) -> ArrayView1<'_, T> {
        match &self.0 {
            MultiRepr::Borrowed(v) => v.column(index),
            MultiRepr::BorrowedMut(v) => v.column(index),
            MultiRepr::Owned(a) => a.column(index),
        }
    }

    /// Returns a sliced 2D view.
    pub fn slice<I>(&self, info: I) -> ArrayView2<'_, T>
    where
        I: ndarray::SliceArg<ndarray::Ix2, OutDim = ndarray::Ix2>,
    {
        match &self.0 {
            MultiRepr::Borrowed(v) => v.slice(info),
            MultiRepr::BorrowedMut(v) => v.slice(info),
            MultiRepr::Owned(a) => a.slice(info),
        }
    }

    /// Returns a mutable sliced 2D view.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    pub fn slice_mut<I>(&mut self, info: I) -> ArrayViewMut2<'_, T>
    where
        I: ndarray::SliceArg<ndarray::Ix2, OutDim = ndarray::Ix2>,
    {
        self.promote();

        self.promote();

        match &mut self.0 {
            MultiRepr::BorrowedMut(a) => a.slice_mut(info),
            MultiRepr::Owned(a) => a.slice_mut(info),
            _ => unreachable!("Self should have been converted to owned by now"),
        }
    }

    /// Returns a 2D view of the samples.
    pub fn view(&self) -> ArrayView2<'_, T> {
        self.as_view()
    }

    /// Returns a mutable 2D view of the samples.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    pub fn view_mut(&mut self) -> ArrayViewMut2<'_, T> {
        self.promote();
        match &mut self.0 {
            MultiRepr::BorrowedMut(a) => a.view_mut(),
            MultiRepr::Owned(a) => a.view_mut(),
            _ => unreachable!("Self should have been converted to owned by now"),
        }
    }

    /// Swaps two axes in-place.
    pub fn swap_axes(&mut self, a: usize, b: usize) {
        self.to_mut().swap_axes(a, b);
    }

    /// Returns a mutable 1D view into `axis` at `index`.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    pub fn index_axis_mut(&mut self, axis: Axis, index: usize) -> ArrayViewMut1<'_, T> {
        self.promote();
        self.promote();
        match &mut self.0 {
            MultiRepr::BorrowedMut(a) => a.index_axis_mut(axis, index),
            MultiRepr::Owned(a) => a.index_axis_mut(axis, index),
            _ => unreachable!("Self should have been converted to owned by now"),
        }
    }

    /// Returns the shape of the underlying `ndarray` buffer.
    pub fn shape(&self) -> &[usize] {
        match &self.0 {
            MultiRepr::Borrowed(v) => v.shape(),
            MultiRepr::BorrowedMut(v) => v.shape(),
            MultiRepr::Owned(a) => a.shape(),
        }
    }

    /// Applies a value-mapping function in-place.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    pub fn mapv_inplace<F>(&mut self, f: F)
    where
        F: FnMut(T) -> T,
    {
        self.to_mut().mapv_inplace(f);
    }

    /// Returns a mutable iterator over 1D lanes along `axis`.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    pub fn axis_iter_mut(&mut self, axis: Axis) -> AxisIterMut<'_, T, Ix1> {
        self.promote();
        match &mut self.0 {
            MultiRepr::BorrowedMut(a) => a.axis_iter_mut(axis),
            MultiRepr::Owned(a) => a.axis_iter_mut(axis),
            _ => unreachable!("Self should have been converted to owned by now"),
        }
    }

    /// Returns a view of the row at `index`.
    pub fn row(&self, index: usize) -> ArrayView1<'_, T> {
        match &self.0 {
            MultiRepr::Borrowed(v) => v.row(index),
            MultiRepr::BorrowedMut(v) => v.row(index),
            MultiRepr::Owned(a) => a.row(index),
        }
    }

    /// Returns an iterator over all samples (row-major).
    pub fn iter(&self) -> ndarray::iter::Iter<'_, T, Ix2> {
        match &self.0 {
            MultiRepr::Borrowed(v) => v.iter(),
            MultiRepr::BorrowedMut(v) => v.iter(),
            MultiRepr::Owned(a) => a.iter(),
        }
    }

    /// Returns a mutable iterator over all samples (row-major).
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    pub fn iter_mut(&mut self) -> ndarray::iter::IterMut<'_, T, Ix2> {
        self.promote();
        match &mut self.0 {
            MultiRepr::BorrowedMut(a) => a.iter_mut(),
            MultiRepr::Owned(a) => a.iter_mut(),
            _ => unreachable!("Self should have been converted to owned by now"),
        }
    }

    /// Returns the total number of samples across all channels.
    pub fn len(&self) -> usize {
        self.as_view().len()
    }

    /// Returns `true` if there are no samples.
    ///
    /// This should really only ever be false si
    pub fn is_empty(&self) -> bool {
        self.as_view().is_empty()
    }

    /// Maps each sample into a new `Array2`.
    pub fn mapv<F, U>(&self, f: F) -> Array2<U>
    where
        F: Fn(T) -> U,
        U: Clone,
    {
        self.as_view().mapv(f)
    }

    /// Returns a mutable pointer to the underlying buffer.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.to_mut().as_mut_ptr()
    }

    /// Returns a shared slice if the underlying storage is contiguous.
    pub fn as_slice(&self) -> Option<&[T]> {
        match &self.0 {
            MultiRepr::Borrowed(v) => v.as_slice(),
            MultiRepr::BorrowedMut(v) => v.as_slice(),
            MultiRepr::Owned(a) => a.as_slice(),
        }
    }

    /// Returns a mutable iterator over the outer axis (rows/channels).
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    pub fn outer_iter(&mut self) -> ndarray::iter::AxisIterMut<'_, T, ndarray::Ix1> {
        self.promote();
        match &mut self.0 {
            MultiRepr::BorrowedMut(a) => a.outer_iter_mut(),
            MultiRepr::Owned(a) => a.outer_iter_mut(),
            _ => unreachable!("Self should have been converted to owned by now"),
        }
    }

    /// Returns a mutable slice if the underlying storage is contiguous.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        self.promote();
        match &mut self.0 {
            MultiRepr::BorrowedMut(a) => a.as_slice_mut(),
            MultiRepr::Owned(a) => a.as_slice_mut(),
            _ => unreachable!("Self should have been converted to owned by now"),
        }
    }

    /// Returns an iterator over 1D lanes along `axis`.
    pub fn axis_iter(&self, axis: ndarray::Axis) -> ndarray::iter::AxisIter<'_, T, ndarray::Ix1> {
        match &self.0 {
            MultiRepr::Borrowed(v) => v.axis_iter(axis),
            MultiRepr::BorrowedMut(v) => v.axis_iter(axis),
            MultiRepr::Owned(a) => a.axis_iter(axis),
        }
    }

    /// Returns the raw dimension.
    pub fn raw_dim(&self) -> ndarray::Dim<[usize; 2]> {
        self.as_view().raw_dim()
    }

    /// Returns a mutable view of the row at `index`.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    pub fn row_mut(&mut self, index: usize) -> ndarray::ArrayViewMut1<'_, T> {
        self.promote();
        match &mut self.0 {
            MultiRepr::BorrowedMut(a) => a.row_mut(index),
            MultiRepr::Owned(a) => a.row_mut(index),
            _ => unreachable!("Self should have been converted to owned by now"),
        }
    }

    /// Fills all samples with `value`.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    pub fn fill(&mut self, value: T) {
        self.to_mut().fill(value);
    }

    /// Converts into a raw `Vec<T>` and an offset.
    ///
    /// For borrowed data, this allocates and clones.
    pub fn into_raw_vec_and_offset(self) -> (Vec<T>, usize) {
        match self.0 {
            MultiRepr::Borrowed(v) => {
                let (vec, offset) = v.to_owned().into_raw_vec_and_offset();
                (vec, offset.unwrap_or(0))
            }
            MultiRepr::BorrowedMut(v) => {
                let (vec, offset) = v.to_owned().into_raw_vec_and_offset();
                (vec, offset.unwrap_or(0))
            }
            MultiRepr::Owned(a) => {
                let (vec, offset) = a.into_raw_vec_and_offset();
                (vec, offset.unwrap_or(0))
            }
        }
    }

    /// Folds over all samples (row-major).
    pub fn fold<B, F>(&self, init: B, f: F) -> B
    where
        F: FnMut(B, &T) -> B,
    {
        self.as_view().iter().fold(init, f)
    }

    /// Converts this wrapper into an owned `Array2<T>`.
    ///
    /// For borrowed data, this allocates and clones.
    pub fn take(self) -> Array2<T> {
        match self.0 {
            MultiRepr::Borrowed(v) => v.to_owned(),
            MultiRepr::BorrowedMut(v) => v.to_owned(),
            MultiRepr::Owned(a) => a,
        }
    }
}

// PartialEq implementations for testing compatibility
impl<'a, T: AudioSample> PartialEq<Array2<T>> for MultiData<'a, T> {
    fn eq(&self, other: &Array2<T>) -> bool {
        self.as_view() == other.view()
    }
}

impl<'a, T: AudioSample> PartialEq<MultiData<'a, T>> for Array2<T> {
    fn eq(&self, other: &MultiData<'a, T>) -> bool {
        self.view() == other.as_view()
    }
}

// IntoIterator for MultiData
impl<'a, T: AudioSample> IntoIterator for &'a MultiData<'_, T> {
    type Item = &'a T;
    type IntoIter = ndarray::iter::Iter<'a, T, ndarray::Ix2>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_view().into_iter()
    }
}

/// Storage container for audio sample data.
///
/// Differentiates between mono (single-channel) and multi-channel audio data
/// while providing a unified interface for processing operations.
///
/// # Variants
/// - `Mono`: Single-channel audio data stored as 1D array
/// - `Multi`: Multi-channel audio data stored as 2D array (channels as rows)
///
/// # Example
/// ```rust
/// use audio_samples::{AudioData, MonoData, MultiData};
/// use ndarray::{array, Array1, Array2};
///
/// // Mono audio data
/// let mono: AudioData<f32> = AudioData::Mono(MonoData::from(array![0.1, 0.2, 0.3]));
///
/// // Stereo audio data
/// let stereo: AudioData<f32> = AudioData::Multi(MultiData::from(
///     array![[0.1, 0.2], [0.3, 0.4]]
/// ));
/// ```
#[derive(Debug, PartialEq)]
pub enum AudioData<'a, T: AudioSample> {
    /// Single-channel audio data
    Mono(MonoData<'a, T>),
    /// Multi-channel audio data
    Multi(MultiData<'a, T>),
}

impl<T: AudioSample> AudioData<'static, T> {
    /// Creates a new AudioData instance from owned data.
    pub fn from_owned(data: AudioData<'_, T>) -> Self {
        data.into_owned()
    }

    /// Computes the mean value across all samples in the audio data.
    pub fn mean(&self) -> Option<T> {
        match self {
            AudioData::Mono(m) => m.mean(),
            AudioData::Multi(m) => m.mean(),
        }
    }

    /// Consumes self and returns the underlying data as an owned Array1 or Array2.
    pub fn into_mono_data(self) -> Option<Array1<T>> {
        if let AudioData::Mono(m) = self {
            Some(m.take())
        } else {
            None
        }
    }

    /// Consumes self and returns the underlying data as an owned Array2.
    /// Returns None if the data is mono.
    pub fn into_multi_data(self) -> Option<Array2<T>> {
        if let AudioData::Multi(m) = self {
            Some(m.take())
        } else {
            None
        }
    }
}

impl<T: AudioSample> Clone for AudioData<'_, T> {
    fn clone(&self) -> Self {
        match self {
            AudioData::Mono(m) => AudioData::Mono(MonoData::from_owned(m.as_view().to_owned())),
            AudioData::Multi(m) => AudioData::Multi(MultiData::from_owned(m.as_view().to_owned())),
        }
    }
}

impl<T: AudioSample> AudioData<'_, T> {
    /// Creates a new mono AudioData from an owned Array1.
    pub const fn new_mono(data: Array1<T>) -> AudioData<'static, T> {
        AudioData::Mono(MonoData(MonoRepr::Owned(data)))
    }

    /// Creates a new multi-channel AudioData from an owned Array2.
    pub const fn new_multi(data: Array2<T>) -> AudioData<'static, T> {
        AudioData::Multi(MultiData(MultiRepr::Owned(data)))
    }

    /// Creates mono AudioData from a slice (borrowed).
    pub fn from_slice<'a>(slice: &'a [T]) -> AudioData<'a, T> {
        let view = ArrayView1::from(slice);
        AudioData::Mono(MonoData(MonoRepr::Borrowed(view)))
    }

    /// Creates mono AudioData from a mutable slice (borrowed).
    pub fn from_slice_mut<'a>(slice: &'a mut [T]) -> AudioData<'a, T> {
        let view = ArrayViewMut1::from(slice);
        AudioData::Mono(MonoData(MonoRepr::BorrowedMut(view)))
    }

    /// Creates multi-channel AudioData from a slice with specified channel count (borrowed).
    ///
    /// # Panics
    /// - If `slice.len()` is not divisible by `channels`
    /// - If the slice cannot be reshaped into the desired shape
    pub fn from_slice_multi<'a>(slice: &'a [T], channels: usize) -> AudioData<'a, T> {
        let total_samples = slice.len();
        assert!(
            total_samples.is_multiple_of(channels),
            "Slice length must be divisible by number of channels"
        );
        let samples_per_channel = total_samples / channels;
        let view = ArrayView2::from_shape((channels, samples_per_channel), slice)
            .expect("Failed to create ArrayView2 from slice");
        AudioData::Multi(MultiData(MultiRepr::Borrowed(view)))
    }

    /// Creates multi-channel AudioData from a mutable slice with specified channel count (borrowed).
    ///
    /// # Panics
    /// - If `slice.len()` is not divisible by `channels`
    /// - If the slice cannot be reshaped into the desired shape
    pub fn from_slice_multi_mut<'a>(slice: &'a mut [T], channels: usize) -> AudioData<'a, T> {
        let total_samples = slice.len();
        assert!(
            total_samples.is_multiple_of(channels),
            "Slice length must be divisible by number of channels"
        );
        let samples_per_channel = total_samples / channels;
        let view = ArrayViewMut2::from_shape((channels, samples_per_channel), slice)
            .expect("Failed to create ArrayViewMut2 from slice");
        AudioData::Multi(MultiData(MultiRepr::BorrowedMut(view)))
    }

    /// Creates mono AudioData from a Vec (owned).
    pub fn from_vec(vec: Vec<T>) -> AudioData<'static, T> {
        AudioData::Mono(MonoData(MonoRepr::Owned(Array1::from(vec))))
    }

    /// Creates multi-channel AudioData from a Vec with specified channel count (owned).
    ///
    /// # Panics
    /// - If `vec.len()` is not divisible by `channels`
    pub fn from_vec_multi(vec: Vec<T>, channels: usize) -> AudioData<'static, T> {
        let total_samples = vec.len();
        assert!(
            total_samples.is_multiple_of(channels),
            "Vec length must be divisible by number of channels"
        );
        let samples_per_channel = total_samples / channels;
        let arr = Array2::from_shape_vec((channels, samples_per_channel), vec)
            .expect("Failed to create Array2 from vec");
        AudioData::Multi(MultiData(MultiRepr::Owned(arr)))
    }
}

// Main implementation block for AudioData
impl<'a, T: AudioSample> AudioData<'a, T> {
    /// Converts this AudioData to owned data.
    pub fn into_owned<'b>(self) -> AudioData<'b, T> {
        match self {
            AudioData::Mono(m) => AudioData::Mono(m.into_owned()),
            AudioData::Multi(m) => AudioData::Multi(m.into_owned()),
        }
    }

    /// Creates a new AudioData instance from borrowed data.
    pub fn from_borrowed<'b>(&'b self) -> AudioData<'b, T> {
        match self {
            AudioData::Mono(m) => AudioData::Mono(MonoData(MonoRepr::Borrowed(m.as_view()))),
            AudioData::Multi(m) => AudioData::Multi(MultiData(MultiRepr::Borrowed(m.as_view()))),
        }
    }

    /// Creates AudioData from a borrowed mono array view.
    pub const fn from_borrowed_array1<'b>(view: ArrayView1<'b, T>) -> AudioData<'b, T> {
        AudioData::Mono(MonoData(MonoRepr::Borrowed(view)))
    }

    /// Creates AudioData from a borrowed mutable mono array view.
    pub const fn from_borrowed_array1_mut<'b>(view: ArrayViewMut1<'b, T>) -> AudioData<'b, T> {
        AudioData::Mono(MonoData(MonoRepr::BorrowedMut(view)))
    }

    /// Creates AudioData from a borrowed multi-channel array view.
    pub const fn from_borrowed_array2<'b>(view: ArrayView2<'b, T>) -> AudioData<'b, T> {
        AudioData::Multi(MultiData(MultiRepr::Borrowed(view)))
    }

    /// Creates AudioData from a borrowed mutable multi-channel array view.
    pub const fn from_borrowed_array2_mut<'b>(view: ArrayViewMut2<'b, T>) -> AudioData<'b, T> {
        AudioData::Multi(MultiData(MultiRepr::BorrowedMut(view)))
    }

    /// Returns the total number of samples in the audio data.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            AudioData::Mono(m) => m.len(),
            AudioData::Multi(m) => m.len(),
        }
    }
    /// Returns the number of channels in the audio data.
    #[inline]
    pub fn num_channels(&self) -> usize {
        match self {
            AudioData::Mono(_) => 1,
            AudioData::Multi(m) => m.shape()[0],
        }
    }
    /// Returns true if the audio data contains no samples.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Returns true if the audio data is mono (single channel).
    #[inline]
    pub const fn is_mono(&self) -> bool {
        matches!(self, AudioData::Mono(_))
    }
    /// Returns true if the audio data has multiple channels.
    #[inline]
    pub const fn is_multi_channel(&self) -> bool {
        matches!(self, AudioData::Multi(_))
    }

    /// Returns the shape of the underlying array data.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        match &self {
            AudioData::Mono(m) => m.shape(),
            AudioData::Multi(m) => m.shape(),
        }
    }
    /// Returns the number of samples per channel.
    #[inline]
    pub fn samples_per_channel(&self) -> usize {
        match self {
            AudioData::Mono(m) => m.as_view().len(),
            AudioData::Multi(m) => m.as_view().shape()[1],
        }
    }

    /// Returns the mean of the audio data along the specified axis.
    pub fn mean_axis(&self, axis: Axis) -> Option<Self> {
        match self {
            AudioData::Mono(m) => {
                let mean = m.mean()?;
                Some(AudioData::Mono(MonoData(MonoRepr::Owned(
                    ndarray::Array1::from_elem(1, mean),
                ))))
            }
            AudioData::Multi(m) => {
                let mean_array = m.mean_axis(axis)?;
                Some(AudioData::Mono(MonoData(MonoRepr::Owned(mean_array))))
            }
        }
    }

    /// Returns audio data as a slice if contiguous.
    pub fn as_slice(&self) -> Option<&[T]> {
        match &self {
            AudioData::Mono(m) => m.as_slice(),
            AudioData::Multi(m) => m.as_slice(),
        }
    }

    /// Returns a contiguous byte view when possible, falling back to I24 packing when required.
    pub fn bytes(&self) -> AudioSampleResult<AudioBytes<'_>> {
        let slice =
            self.as_slice()
                .ok_or(AudioSampleError::Layout(LayoutError::NonContiguous {
                    operation: "bytes view".to_string(),
                    layout_type: "non-contiguous audio data".to_string(),
                }))?;

        if TypeId::of::<T>() == TypeId::of::<I24>() {
            // I24 is 4 bytes in memory but 3 bytes on disk; pack to owned bytes.
            let i24_slice: &[I24] =
                unsafe { core::slice::from_raw_parts(slice.as_ptr() as *const I24, slice.len()) };
            let packed = I24::write_i24s_ne(i24_slice);
            return Ok(AudioBytes::Owned(packed));
        }

        let byte_len = std::mem::size_of_val(slice);
        let byte_ptr = slice.as_ptr() as *const u8;
        let bytes = unsafe { core::slice::from_raw_parts(byte_ptr, byte_len) };
        Ok(AudioBytes::Borrowed(bytes))
    }

    /// Returns the number of bytes per sample for the current sample type.
    #[inline]
    pub const fn bytes_per_sample(&self) -> usize {
        T::BYTES
    }

    /// Returns an owned byte buffer.
    ///
    /// Note: this always allocates a new `Vec<u8>`. Use [`Self::bytes`] for a
    /// potentially zero-copy view.
    pub fn into_bytes(&self) -> Vec<u8> {
        self.bytes()
            .map(AudioBytes::into_owned)
            .unwrap_or_else(|_| match self {
                AudioData::Mono(m) => m.as_view().iter().flat_map(|s| s.to_bytes()).collect(),
                AudioData::Multi(m) => m.as_view().iter().flat_map(|s| s.to_bytes()).collect(),
            })
    }

    /// Deprecated: use `bytes()` for borrowed access or `into_bytes()` for owned bytes.
    #[deprecated(note = "use bytes() for borrowed access or into_bytes() for owned bytes")]
    pub fn as_bytes(&self) -> Vec<u8> {
        self.into_bytes()
    }

    /// Maps a function over each sample, returning new owned audio data.
    pub fn mapv<F, U>(&self, f: F) -> AudioData<'static, U>
    where
        F: Fn(T) -> U,
        U: AudioSample,
    {
        match self {
            AudioData::Mono(m) => {
                let out = m.as_view().mapv(f);
                AudioData::Mono(MonoData(MonoRepr::Owned(out)))
            }
            AudioData::Multi(m) => {
                let out = m.as_view().mapv(f);
                AudioData::Multi(MultiData(MultiRepr::Owned(out)))
            }
        }
    }

    /// Maps a function over each sample in place.
    pub fn mapv_inplace<F>(&mut self, f: F)
    where
        F: Fn(T) -> T,
    {
        match self {
            AudioData::Mono(m) => m.to_mut().iter_mut().for_each(|x| *x = f(*x)),
            AudioData::Multi(m) => m.to_mut().iter_mut().for_each(|x| *x = f(*x)),
        }
    }

    /// Applies a function to each sample in place.
    pub fn apply<F>(&mut self, func: F)
    where
        F: Fn(T) -> T,
    {
        self.mapv_inplace(func)
    }

    /// Applies a function to each sample with its index in place.
    pub fn apply_with_index<F>(&mut self, func: F)
    where
        F: Fn(usize, T) -> T,
    {
        match self {
            AudioData::Mono(m) => {
                for (i, x) in m.to_mut().iter_mut().enumerate() {
                    *x = func(i, *x);
                }
            }
            AudioData::Multi(m) => {
                // index by frame within channel
                for mut row in m.to_mut().rows_mut() {
                    for (i, x) in row.iter_mut().enumerate() {
                        *x = func(i, *x);
                    }
                }
            }
        }
    }

    /// Applies a windowed function to the audio data with overlap processing.
    pub fn apply_windowed<F>(
        &mut self,
        window_size: usize,
        hop_size: usize,
        func: F,
    ) -> AudioSampleResult<()>
    where
        F: Fn(&[T], &[T]) -> Vec<T>, // (current_window, prev_window) -> processed_window
    {
        if window_size == 0 || hop_size == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "window_hop_size",
                "Window size and hop size must be greater than 0",
            )));
        }

        match self {
            AudioData::Mono(m) => {
                let data = m.as_view();
                let x = data.as_slice().ok_or(AudioSampleError::Layout(
                    LayoutError::NonContiguous {
                        operation: "mono processing".to_string(),
                        layout_type: "non-contiguous mono data".to_string(),
                    },
                ))?;

                let n = x.len();
                if n < window_size {
                    return Ok(());
                }

                let num_windows = (n - window_size) / hop_size + 1;
                let out_len = (num_windows - 1) * hop_size + window_size;

                let mut result = vec![T::default(); out_len];
                let mut overlap = vec![0usize; out_len];
                let mut prev = vec![T::default(); window_size];

                for w in 0..num_windows {
                    let pos = w * hop_size;
                    let win = &x[pos..pos + window_size];
                    let processed = func(win, &prev);

                    // overlap-add
                    for (i, &s) in processed.iter().enumerate() {
                        let idx = pos + i;
                        result[idx] += s;
                        overlap[idx] += 1;
                    }

                    prev.copy_from_slice(win);
                }

                // normalise overlaps
                for (y, &c) in result.iter_mut().zip(&overlap) {
                    if c > 1 {
                        *y /= T::cast_from(c);
                    }
                }

                // REPLACE THE VARIANT (don’t try to mutate inner binding)
                m.0 = MonoRepr::Owned(ndarray::Array1::from(result));
                Ok(())
            }

            AudioData::Multi(m) => {
                let view = m.as_view();
                let (ch, spc) = view.dim();
                if spc < window_size {
                    return Ok(());
                }

                let num_windows = (spc - window_size) / hop_size + 1;
                let out_len = (num_windows - 1) * hop_size + window_size;

                let mut out = ndarray::Array2::from_elem((ch, out_len), T::default());
                let mut cnt = vec![0usize; out_len];
                let mut prev = vec![T::default(); window_size];

                for c in 0..ch {
                    let row = view.row(c);
                    let x = row.as_slice().ok_or(AudioSampleError::Layout(
                        LayoutError::NonContiguous {
                            operation: "multi-channel row processing".to_string(),
                            layout_type: "non-contiguous row data".to_string(),
                        },
                    ))?;

                    cnt.fill(0);
                    prev.fill(T::default());

                    for w in 0..num_windows {
                        let pos = w * hop_size;
                        let win = &x[pos..pos + window_size];
                        let processed = func(win, &prev);

                        for (i, &s) in processed.iter().enumerate() {
                            let idx = pos + i;
                            out[[c, idx]] += s;
                            cnt[idx] += 1;
                        }
                        prev.copy_from_slice(win);
                    }

                    for i in 0..out_len {
                        if cnt[i] > 1 {
                            out[[c, i]] /= T::cast_from(cnt[i]);
                        }
                    }
                }

                // REPLACE THE VARIANT
                m.0 = MultiRepr::Owned(out);
                Ok(())
            }
        }
    }

    /// Applies a function to all samples in all channels.
    pub fn apply_to_all_channels<F>(&mut self, f: F)
    where
        F: Fn(T) -> T,
    {
        self.mapv_inplace(f)
    }

    /// Applies a function to samples in specified channels only.
    pub fn apply_to_channels<F>(&mut self, channels: &[usize], f: F)
    where
        F: Fn(T) -> T,
    {
        match self {
            AudioData::Mono(m) => m.to_mut().iter_mut().for_each(|x| *x = f(*x)),
            AudioData::Multi(m) => {
                let mut a = m.to_mut();
                for (ch_idx, mut row) in a.axis_iter_mut(Axis(0)).enumerate() {
                    if channels.contains(&ch_idx) {
                        for x in row.iter_mut() {
                            *x = f(*x);
                        }
                    }
                }
            }
        }
    }
    /// Converts the samples to another sample type.
    ///
    /// This is an audio-aware conversion using [`ConvertTo`], so it can clamp and scale
    /// as needed for the source/target sample formats.
    pub fn convert_to<O: AudioSample>(&self) -> AudioData<'static, O>
    where
        T: ConvertTo<O>,
    {
        match self {
            AudioData::Mono(m) => {
                let out = m.as_view().mapv(|x| x.convert_to());
                AudioData::Mono(MonoData(MonoRepr::Owned(out)))
            }
            AudioData::Multi(m) => {
                let out = m.as_view().mapv(|x| x.convert_to());
                AudioData::Multi(MultiData(MultiRepr::Owned(out)))
            }
        }
    }
    /// Converts the audio data to an interleaved vector, consuming the data.
    pub fn to_interleaved_vec(self) -> Vec<T> {
        match self {
            AudioData::Mono(m) => match m.0 {
                MonoRepr::Borrowed(v) => v.to_owned().to_vec(),
                MonoRepr::BorrowedMut(v) => v.to_owned().to_vec(),
                MonoRepr::Owned(a) => a.to_vec(),
            },
            AudioData::Multi(m) => {
                let (ch, _spc) = m.as_view().dim();
                // Get planar data as contiguous slice
                let planar: Vec<T> = if let Some(slice) = m.as_view().as_slice() {
                    slice.to_vec()
                } else {
                    // Array not contiguous in memory, must iterate
                    m.as_view().iter().copied().collect()
                };

                // Use optimized interleave
                crate::simd_conversions::interleave_multi_vec(planar, ch)
                    .expect("Interleave failed - this should not happen with valid input")
            }
        }
    }

    /// Returns the audio data as an interleaved vector without consuming the data.
    pub fn as_interleaved_vec(&self) -> Vec<T> {
        match self {
            AudioData::Mono(m) => m.as_view().to_vec(),
            AudioData::Multi(m) => {
                let v = m.as_view();
                let (ch, _spc) = v.dim();
                // Get planar data as contiguous slice
                let planar: Vec<T> = if let Some(slice) = v.as_slice() {
                    slice.to_vec()
                } else {
                    // Array not contiguous in memory, must iterate
                    v.iter().copied().collect()
                };

                // Use optimized interleave
                crate::simd_conversions::interleave_multi_vec(planar, ch)
                    .expect("Interleave failed - this should not happen with valid input")
            }
        }
    }
}

// Conversion from ndarray views
impl<'a, T: AudioSample> From<ArrayView1<'a, T>> for AudioData<'a, T> {
    fn from(arr: ArrayView1<'a, T>) -> Self {
        AudioData::Mono(MonoData(MonoRepr::Borrowed(arr)))
    }
}
impl<'a, T: AudioSample> From<ArrayViewMut1<'a, T>> for AudioData<'a, T> {
    fn from(arr: ArrayViewMut1<'a, T>) -> Self {
        AudioData::Mono(MonoData(MonoRepr::BorrowedMut(arr)))
    }
}

impl<'a, T: AudioSample> From<ArrayView2<'a, T>> for AudioData<'a, T> {
    fn from(arr: ArrayView2<'a, T>) -> Self {
        AudioData::Multi(MultiData(MultiRepr::Borrowed(arr)))
    }
}

impl<'a, T: AudioSample> From<ArrayViewMut2<'a, T>> for AudioData<'a, T> {
    fn from(arr: ArrayViewMut2<'a, T>) -> Self {
        AudioData::Multi(MultiData(MultiRepr::BorrowedMut(arr)))
    }
}

// Indexing
impl<'a, T: AudioSample> Index<usize> for AudioData<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            AudioData::Mono(arr) => &arr[index],
            AudioData::Multi(arr) => {
                let (channels, samples_per_channel) = arr.dim();
                let total_samples = channels * samples_per_channel;
                if index >= total_samples {
                    panic!(
                        "Index {} out of bounds for total samples {}",
                        index, total_samples
                    );
                }
                let channel = index / samples_per_channel;
                let sample_idx = index % samples_per_channel;
                &arr[(channel, sample_idx)]
            }
        }
    }
}

// ---------------------
// OPS
// ---------------------

impl<'a, T: AudioSample> From<Array1<T>> for MonoData<'a, T> {
    fn from(a: Array1<T>) -> Self {
        MonoData(MonoRepr::Owned(a))
    }
}
impl<'a, T: AudioSample> From<Array2<T>> for MultiData<'a, T> {
    fn from(a: Array2<T>) -> Self {
        MultiData(MultiRepr::Owned(a))
    }
}

impl<'a, T: AudioSample> From<Array1<T>> for AudioData<'a, T> {
    fn from(a: Array1<T>) -> Self {
        AudioData::Mono(a.into())
    }
}
impl<'a, T: AudioSample> From<Array2<T>> for AudioData<'a, T> {
    fn from(a: Array2<T>) -> Self {
        AudioData::Multi(a.into())
    }
}

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
            // =========================
            // Binary ops: AudioData ∘ AudioData  -> new AudioData
            // =========================
            impl<'a, T: AudioSample> std::ops::$trait<Self> for AudioData<'a, T> {
                type Output = Self;

                fn $method(self, rhs: Self) -> Self::Output {
                    match (self, rhs) {
                        (AudioData::Mono(lhs), AudioData::Mono(rhs)) => {
                            if lhs.len() != rhs.len() {
                                panic!($mono_err);
                            }
                            // operate on views; convert to owned to ensure operation works
                            AudioData::Mono((&lhs.as_view() $op &rhs.as_view()).into())
                        }
                        (AudioData::Multi(lhs), AudioData::Multi(rhs)) => {
                            if lhs.as_view().dim() != rhs.as_view().dim() {
                                panic!($multi_err);
                            }
                            AudioData::Multi((&lhs.as_view() $op &rhs.as_view()).into())
                        }
                        _ => panic!($mismatch_err),
                    }
                }
            }

            // =========================
            // Binary ops: AudioData ∘ scalar -> new AudioData
            // =========================
            impl<'a, T: AudioSample> std::ops::$trait<T> for AudioData<'a, T> {
                type Output = Self;

                fn $method(self, rhs: T) -> Self::Output {
                    match self {
                        AudioData::Mono(a) => AudioData::Mono(a.as_view().mapv(|x| x $op rhs).into()),
                        AudioData::Multi(a) => AudioData::Multi(a.as_view().mapv(|x| x $op rhs).into()),
                    }
                }
            }

            // =========================
            // Assignment ops: AudioData ∘= AudioData  (in-place, no Default)
            // =========================
            impl<'a, T: AudioSample> std::ops::$assign_trait<Self> for AudioData<'a, T>
            where
                T: Clone,
            {
                fn $assign_method(&mut self, rhs: Self) {
                    match (self, rhs) {
                        (AudioData::Mono(lhs), AudioData::Mono(rhs)) => {
                            if lhs.len() != rhs.len() {
                                panic!($mono_err);
                            }
                            // promote lhs to owned and apply in place against rhs view
                            let mut lhs_mut = lhs.to_mut();
                            let rhs_view = rhs.as_view();
                            // Use zip_mut_with for element-wise in-place operations
                            lhs_mut.zip_mut_with(&rhs_view, |a, &b| *a = *a $op b);
                        }
                        (AudioData::Multi(lhs), AudioData::Multi(rhs)) => {
                            if lhs.as_view().dim() != rhs.as_view().dim() {
                                panic!($multi_err);
                            }
                            let mut lhs_mut = lhs.to_mut();
                            let rhs_view = rhs.as_view();
                            lhs_mut.zip_mut_with(&rhs_view, |a, &b| *a = *a $op b);
                        }
                        _ => panic!($mismatch_err),
                    }
                }
            }

            // =========================
            // Assignment ops: AudioData ∘= scalar  (in-place)
            // =========================
            impl<'a, T: AudioSample> std::ops::$assign_trait<T> for AudioData<'a, T>
            where
                T: Clone,
            {
                fn $assign_method(&mut self, rhs: T) {
                    match self {
                        AudioData::Mono(lhs) => {
                            let mut lhs_mut = lhs.to_mut();
                            // Use element-wise iteration for scalar assignment operations
                            lhs_mut.iter_mut().for_each(|x| *x = *x $op rhs);
                        }
                        AudioData::Multi(lhs) => {
                            let mut lhs_mut = lhs.to_mut();
                            lhs_mut.iter_mut().for_each(|x| *x = *x $op rhs);
                        }
                    }
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
impl<'a, T: AudioSample> Neg for AudioData<'a, T>
where
    T: Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            AudioData::Mono(arr) => AudioData::Mono(arr.as_view().mapv(|x| -x).into()),
            AudioData::Multi(arr) => AudioData::Multi(arr.as_view().mapv(|x| -x).into()),
        }
    }
}

/// Represents homogeneous audio samples with associated metadata.
///
/// Primary container for audio data combining raw sample values with essential
/// metadata including sample rate, channel layout, and type information.
/// Supports both mono and multi-channel audio with unified interface.
///
/// # Fields
/// - `data`: Audio sample data in mono or multi-channel format
/// - `sample_rate`: Sampling frequency in Hz
/// - `layout`: Channel organization (interleaved or non-interleaved)
///
/// # Examples
/// ```rust
/// use audio_samples::{AudioSamples, sample_rate};
/// use ndarray::array;
///
/// let mono_audio = AudioSamples::new_mono(array![0.1f32, 0.2, 0.3], sample_rate!(44100));
/// assert_eq!(mono_audio.num_channels(), 1);
///
/// let stereo_data = array![[0.1f32, 0.2], [0.3f32, 0.4]];
/// let stereo_audio = AudioSamples::new_multi_channel(stereo_data, sample_rate!(48000));
/// assert_eq!(stereo_audio.num_channels(), 2);
/// ```
///
/// # Invariants
///
/// The following properties are guaranteed by construction and cannot be violated:
/// - `sample_rate` is always > 0 (stored as [`NonZeroU32`])
/// - `num_channels()` is always ≥ 1 (mono has 1 channel, multi has ≥ 1 rows)
/// - `samples_per_channel()` is always ≥ 1 (empty audio is not allowed)
///
/// These invariants eliminate the need for runtime validation in downstream code.
#[derive(Debug, PartialEq)]
pub struct AudioSamples<'a, T: AudioSample> {
    /// The audio sample data.
    pub data: AudioData<'a, T>,
    /// Sample rate in Hz (guaranteed non-zero).
    pub sample_rate: NonZeroU32,
    /// Channel layout metadata.
    ///
    /// Note: multi-channel sample data stored in [`AudioData::Multi`] is represented as a
    /// 2D array with shape `(channels, samples_per_channel)` (planar). This field records
    /// how the samples should be interpreted/serialized (e.g. when converting to/from an
    /// interleaved `Vec<T>`).
    pub layout: ChannelLayout,
}

impl<'a, T: AudioSample> Display for AudioSamples<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let type_name = std::any::type_name::<T>();
        let n_channels = self.num_channels();
        let n_samples = self.samples_per_channel();
        let rate = self.sample_rate;
        let layout = &self.layout;

        // Compact header (always shown)
        writeln!(
            f,
            "AudioSamples<{}>: {} ch × {} samples @ {} Hz ({:?})",
            type_name, n_channels, n_samples, rate, layout
        )?;

        // Alternate (#) gives full details; otherwise, concise
        if !f.alternate() {
            // Compact summary
            match &self.data {
                AudioData::Mono(arr) => {
                    let len = arr.len();
                    let preview = 5.min(len);
                    write!(f, "[")?;
                    for (i, val) in arr.as_view().iter().take(preview).enumerate() {
                        write!(f, "{:.4}", val)?;
                        if i < preview - 1 {
                            write!(f, ", ")?;
                        }
                    }
                    if len > preview {
                        write!(f, ", ...")?;
                    }
                    write!(f, "]")?;
                }
                AudioData::Multi(arr) => {
                    let (channels, _) = arr.dim();
                    for ch in 0..channels {
                        let ch_data = arr.index_axis(ndarray::Axis(0), ch);
                        let len = ch_data.len();
                        let preview = 3.min(len);
                        write!(f, "\nCh {}: [", ch)?;
                        for (i, val) in ch_data.iter().take(preview).enumerate() {
                            write!(f, "{:.4}", val)?;
                            if i < preview - 1 {
                                write!(f, ", ")?;
                            }
                        }
                        if len > preview {
                            write!(f, ", ...")?;
                        }
                        write!(f, "]")?;
                    }
                }
            }
        } else {
            // Detailed alternate view
            match &self.data {
                AudioData::Mono(arr) => {
                    let len = arr.len();
                    let display_len = 5.min(len);
                    write!(f, "Mono Channel\n  First {} samples: [", display_len)?;
                    for (i, val) in arr.iter().take(display_len).enumerate() {
                        write!(f, "{:.4}", val)?;
                        if i < display_len - 1 {
                            write!(f, ", ")?;
                        }
                    }
                    write!(f, "]")?;
                    if len > display_len {
                        write!(f, "\n  Last {} samples: [", display_len)?;
                        for (i, val) in arr.iter().rev().take(display_len).rev().enumerate() {
                            write!(f, "{:.4}", val)?;
                            if i < display_len - 1 {
                                write!(f, ", ")?;
                            }
                        }
                        write!(f, "]")?;
                    }
                }
                AudioData::Multi(arr) => {
                    let (channels, samples) = arr.dim();
                    for ch in 0..channels {
                        let ch_data = arr.index_axis(ndarray::Axis(0), ch);
                        let len = samples;
                        let display_len = 5.min(len);

                        write!(f, "\nChannel {}:", ch)?;
                        write!(f, "\n  First {} samples: [", display_len)?;
                        for (i, val) in ch_data.iter().take(display_len).enumerate() {
                            write!(f, "{:.4}", val)?;
                            if i < display_len - 1 {
                                write!(f, ", ")?;
                            }
                        }
                        write!(f, "]")?;

                        if len > display_len {
                            write!(f, "\n  Last {} samples: [", display_len)?;
                            for (i, val) in ch_data.iter().rev().take(display_len).rev().enumerate()
                            {
                                write!(f, "{:.4}", val)?;
                                if i < display_len - 1 {
                                    write!(f, ", ")?;
                                }
                            }
                            write!(f, "]")?;
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl<T: AudioSample> AudioSamples<'static, T> {
    /// Creates AudioSamples from owned data.
    pub fn from_owned(data: AudioData<'_, T>, sample_rate: NonZeroU32) -> Self {
        let owned = data.into_owned();

        let layout = match &owned {
            AudioData::Mono(_) => ChannelLayout::NonInterleaved,
            AudioData::Multi(_) => ChannelLayout::Interleaved,
        };

        Self {
            data: owned,
            sample_rate,
            layout,
        }
    }

    /// Consumes self and returns the underlying AudioData
    pub fn into_data(self) -> AudioData<'static, T> {
        self.data.into_owned()
    }

    /// Consumes self and returns the underlying mono Array1 if applicable
    pub fn into_array1(self) -> Option<Array1<T>> {
        match self.data {
            AudioData::Mono(m) => Some(m.take()),
            AudioData::Multi(_) => None,
        }
    }

    /// Consumes self and returns the underlying multi-channel Array2 if applicable
    pub fn into_array2(self) -> Option<Array2<T>> {
        match self.data {
            AudioData::Multi(m) => Some(m.take()),
            AudioData::Mono(_) => None,
        }
    }
}

impl<'a, T: AudioSample> AudioSamples<'a, T> {
    /// Creates a new AudioSamples with the given data and sample rate.
    ///
    /// This is a low-level constructor. Prefer `new_mono` or `new_multi_channel`
    /// for most use cases.
    pub const fn new(data: AudioData<'a, T>, sample_rate: NonZeroU32) -> Self {
        Self {
            data,
            sample_rate,
            layout: ChannelLayout::Interleaved, // Default layout, can be changed later
        }
    }

    /// Creates AudioSamples from borrowed data.
    pub const fn from_borrowed(data: AudioData<'a, T>, sample_rate: NonZeroU32) -> Self {
        let layout = match &data {
            AudioData::Mono(_) => ChannelLayout::NonInterleaved,
            AudioData::Multi(_) => ChannelLayout::Interleaved,
        };

        Self {
            data,
            sample_rate,
            layout,
        }
    }

    /// Creates AudioSamples from borrowed data with specified channel layout.
    pub const fn from_borrowed_with_layout(
        data: AudioData<'a, T>,
        sample_rate: NonZeroU32,
        layout: ChannelLayout,
    ) -> Self {
        Self {
            data,
            sample_rate,
            layout,
        }
    }

    /// Convert audio samples to another sample type.
    pub fn convert_to<O: AudioSample>(&self) -> AudioSamples<'static, O>
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

    /// Creates a new mono AudioSamples that owns its data.
    ///
    /// # Arguments
    /// * `data` - 1D array containing the audio samples
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// A new mono AudioSamples instance that owns the provided data.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 0.5, -0.5];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100));
    /// assert_eq!(audio.num_channels(), 1);
    /// assert_eq!(audio.sample_rate().get(), 44100);
    /// ```
    /// Creates a new mono AudioSamples with the given data and sample rate.
    ///
    /// # Panics
    /// - If `data` is empty
    pub fn new_mono<'b>(data: Array1<T>, sample_rate: NonZeroU32) -> AudioSamples<'b, T> {
        assert!(!data.is_empty(), "data must not be empty");
        AudioSamples {
            data: AudioData::Mono(MonoData(MonoRepr::Owned(data))),
            sample_rate,
            layout: ChannelLayout::NonInterleaved,
        }
    }

    /// Creates a new mono AudioSamples that owns its data, returning an error on invalid input.
    ///
    /// This is the fallible version of [`new_mono`](Self::new_mono). Use this when you want
    /// to handle validation errors gracefully instead of panicking.
    ///
    /// # Arguments
    /// * `data` - 1D array containing the audio samples
    /// * `sample_rate` - Sample rate in Hz (must be non-zero)
    ///
    /// # Returns
    /// `Ok(AudioSamples)` on success, or an error if:
    /// - `data` is empty
    /// - `sample_rate` is zero
    ///
    /// # Examples
    /// ```
    /// use audio_samples::AudioSamples;
    /// use ndarray::array;
    ///
    /// // Valid input
    /// let data = array![1.0f32, -1.0, 0.5, -0.5];
    /// let audio = AudioSamples::try_new_mono(data, 44100).unwrap();
    /// assert_eq!(audio.num_channels(), 1);
    ///
    /// // Invalid input (empty array)
    /// let empty = ndarray::Array1::<f32>::zeros(0);
    /// assert!(AudioSamples::try_new_mono(empty, 44100).is_err());
    ///
    /// // Invalid input (zero sample rate)
    /// let data = array![1.0f32, -1.0];
    /// assert!(AudioSamples::try_new_mono(data, 0).is_err());
    /// ```
    pub fn try_new_mono<'b>(
        data: Array1<T>,
        sample_rate: u32,
    ) -> crate::AudioSampleResult<AudioSamples<'b, T>> {
        if data.is_empty() {
            return Err(crate::AudioSampleError::Layout(
                crate::LayoutError::empty_data("try_new_mono"),
            ));
        }
        let sample_rate = NonZeroU32::new(sample_rate).ok_or_else(|| {
            crate::AudioSampleError::Parameter(crate::ParameterError::invalid_value(
                "sample_rate",
                "must be non-zero",
            ))
        })?;
        Ok(AudioSamples {
            data: AudioData::Mono(MonoData(MonoRepr::Owned(data))),
            sample_rate,
            layout: ChannelLayout::NonInterleaved,
        })
    }

    /// Creates a new multi-channel AudioSamples with the given data and sample rate.
    ///
    /// # Arguments
    /// * `data` - 2D array where each row represents a channel and each column a sample
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// A new multi-channel AudioSamples instance that owns the provided data.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![[1.0f32, -1.0], [0.5, -0.5]]; // 2 channels, 2 samples each
    /// let audio = AudioSamples::new_multi_channel(data, sample_rate!(44100));
    /// assert_eq!(audio.num_channels(), 2);
    /// assert_eq!(audio.samples_per_channel(), 2);
    /// ```
    /// Creates a new multi-channel AudioSamples with the given data and sample rate.
    ///
    /// # Panics
    /// - If `data` is empty (has zero samples per channel)
    pub fn new_multi_channel<'b>(data: Array2<T>, sample_rate: NonZeroU32) -> AudioSamples<'b, T> {
        assert!(
            data.ncols() > 0,
            "data must have at least one sample per channel"
        );
        AudioSamples {
            data: AudioData::Multi(MultiData(MultiRepr::Owned(data))),
            sample_rate,
            layout: ChannelLayout::Interleaved,
        }
    }

    /// Creates a new multi-channel AudioSamples, returning an error on invalid input.
    ///
    /// This is the fallible version of [`new_multi_channel`](Self::new_multi_channel). Use this
    /// when you want to handle validation errors gracefully instead of panicking.
    ///
    /// # Arguments
    /// * `data` - 2D array where each row represents a channel and each column a sample
    /// * `sample_rate` - Sample rate in Hz (must be non-zero)
    ///
    /// # Returns
    /// `Ok(AudioSamples)` on success, or an error if:
    /// - `data` has zero samples per channel (zero columns)
    /// - `sample_rate` is zero
    ///
    /// # Examples
    /// ```
    /// use audio_samples::AudioSamples;
    /// use ndarray::array;
    ///
    /// // Valid input
    /// let data = array![[1.0f32, -1.0], [0.5, -0.5]]; // 2 channels, 2 samples each
    /// let audio = AudioSamples::try_new_multi_channel(data, 44100).unwrap();
    /// assert_eq!(audio.num_channels(), 2);
    /// assert_eq!(audio.samples_per_channel(), 2);
    ///
    /// // Invalid input (zero columns)
    /// let empty = ndarray::Array2::<f32>::zeros((2, 0));
    /// assert!(AudioSamples::try_new_multi_channel(empty, 44100).is_err());
    ///
    /// // Invalid input (zero sample rate)
    /// let data = array![[1.0f32, -1.0], [0.5, -0.5]];
    /// assert!(AudioSamples::try_new_multi_channel(data, 0).is_err());
    /// ```
    pub fn try_new_multi_channel<'b>(
        data: Array2<T>,
        sample_rate: u32,
    ) -> crate::AudioSampleResult<AudioSamples<'b, T>> {
        if data.ncols() == 0 {
            return Err(crate::AudioSampleError::Layout(
                crate::LayoutError::empty_data("try_new_multi_channel"),
            ));
        }
        let sample_rate = NonZeroU32::new(sample_rate).ok_or_else(|| {
            crate::AudioSampleError::Parameter(crate::ParameterError::invalid_value(
                "sample_rate",
                "must be non-zero",
            ))
        })?;
        Ok(AudioSamples {
            data: AudioData::Multi(MultiData(MultiRepr::Owned(data))),
            sample_rate,
            layout: ChannelLayout::Interleaved,
        })
    }

    /// Creates a new mono AudioSamples filled with zeros.
    ///
    /// # Arguments
    /// * `length` - Number of samples to create
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// A new mono AudioSamples instance filled with zero values.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::AudioSamples;
    ///
    /// let audio = AudioSamples::<f32>::zeros_mono(1024, sample_rate!(44100));
    /// assert_eq!(audio.samples_per_channel(), 1024);
    /// assert_eq!(audio.num_channels(), 1);
    /// ```
    ///
    /// # Panics
    /// - If `length` is 0
    pub fn zeros_mono(length: usize, sample_rate: NonZeroU32) -> Self {
        assert!(length > 0, "length must be greater than 0");
        Self {
            data: AudioData::Mono(MonoData(MonoRepr::Owned(Array1::zeros(length)))),
            sample_rate,
            layout: ChannelLayout::NonInterleaved,
        }
    }

    /// Creates a new multi-channel AudioSamples filled with zeros.
    ///
    /// # Arguments
    /// * `channels` - Number of channels to create
    /// * `length` - Number of samples per channel
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Returns
    /// A new multi-channel AudioSamples instance filled with zero values.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    ///
    /// let audio = AudioSamples::<f32>::zeros_multi(2, 1024, sample_rate!(44100));
    /// assert_eq!(audio.num_channels(), 2);
    /// assert_eq!(audio.samples_per_channel(), 1024);
    /// ```
    ///
    /// # Panics
    /// - If `length` is 0
    pub fn zeros_multi(channels: usize, length: usize, sample_rate: NonZeroU32) -> Self {
        assert!(length > 0, "length must be greater than 0");
        Self {
            data: AudioData::Multi(MultiData(MultiRepr::Owned(Array2::zeros((
                channels, length,
            ))))),
            sample_rate,
            layout: ChannelLayout::Interleaved,
        }
    }

    /// Creates a new multi-channel AudioSamples filled with zeros (static version)
    ///
    /// # Panics
    /// - If `length` is 0
    pub fn zeros_multi_channel(
        channels: usize,
        length: usize,
        sample_rate: NonZeroU32,
    ) -> AudioSamples<'static, T> {
        assert!(length > 0, "length must be greater than 0");
        AudioSamples {
            data: AudioData::Multi(MultiData(MultiRepr::Owned(Array2::zeros((
                channels, length,
            ))))),
            sample_rate,
            layout: ChannelLayout::Interleaved,
        }
    }

    /// Creates a new mono AudioSamples filled with ones
    ///
    /// # Panics
    /// - If `length` is 0
    pub fn ones_mono(length: usize, sample_rate: NonZeroU32) -> Self {
        assert!(length > 0, "length must be greater than 0");
        Self {
            data: AudioData::Mono(MonoData(MonoRepr::Owned(Array1::ones(length)))),
            sample_rate,
            layout: ChannelLayout::NonInterleaved,
        }
    }

    /// Creates a new multi-channel AudioSamples filled with ones
    ///
    /// # Panics
    /// - If `length` is 0
    pub fn ones_multi(channels: usize, length: usize, sample_rate: NonZeroU32) -> Self {
        assert!(length > 0, "length must be greater than 0");
        Self {
            data: AudioData::Multi(MultiData(MultiRepr::Owned(Array2::ones((
                channels, length,
            ))))),
            sample_rate,
            layout: ChannelLayout::Interleaved,
        }
    }

    /// Creates a new mono AudioSamples filled with the specified value
    ///
    /// # Panics
    /// - If `length` is 0
    pub fn uniform_mono(length: usize, sample_rate: NonZeroU32, value: T) -> Self {
        assert!(length > 0, "length must be greater than 0");
        Self {
            data: AudioData::Mono(MonoData(MonoRepr::Owned(Array1::from_elem(length, value)))),
            sample_rate,
            layout: ChannelLayout::NonInterleaved,
        }
    }

    /// Creates a new multi-channel AudioSamples filled with the specified value
    ///
    /// # Panics
    /// - If `length` is 0
    pub fn uniform_multi(
        channels: usize,
        length: usize,
        sample_rate: NonZeroU32,
        value: T,
    ) -> Self {
        assert!(length > 0, "length must be greater than 0");
        Self {
            data: AudioData::Multi(MultiData(MultiRepr::Owned(Array2::from_elem(
                (channels, length),
                value,
            )))),
            sample_rate,
            layout: ChannelLayout::Interleaved,
        }
    }

    /// Returns a mutable reference to the channel layout
    pub const fn layout_mut(&mut self) -> &mut ChannelLayout {
        &mut self.layout
    }

    /// Sets the channel layout
    pub const fn set_layout(&mut self, layout: ChannelLayout) {
        self.layout = layout;
    }

    /// Returns basic info: (num_channels, samples_per_channel, duration_seconds, sample_rate, layout)
    pub fn info(&self) -> (usize, usize, f64, u32, ChannelLayout) {
        (
            self.num_channels(),
            self.samples_per_channel(),
            self.duration_seconds(),
            self.sample_rate.get(),
            self.layout,
        )
    }

    /// Returns the sample rate in Hz as a [`NonZeroU32`].
    ///
    /// This is guaranteed to be non-zero by construction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(array![1.0f32, 2.0], sample_rate!(44100));
    /// assert_eq!(audio.sample_rate().get(), 44100);
    /// ```
    #[inline]
    pub const fn sample_rate(&self) -> NonZeroU32 {
        self.sample_rate
    }

    /// Returns the sample rate as a plain `u32`.
    ///
    /// This is a convenience method equivalent to `self.sample_rate().get()`.
    #[inline]
    pub const fn sample_rate_hz(&self) -> u32 {
        self.sample_rate.get()
    }

    /// Returns the number of channels.
    ///
    /// This is guaranteed to be ≥ 1 by construction.
    #[inline]
    pub fn num_channels(&self) -> usize {
        self.data.num_channels()
    }

    /// Returns the number of channels as a [`NonZeroUsize`].
    ///
    /// This provides a compile-time guarantee that the value is non-zero,
    /// eliminating the need for runtime validation in downstream code.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(array![1.0f32, 2.0], sample_rate!(44100));
    /// let channels = audio.num_channels_nonzero();
    /// assert_eq!(channels.get(), 1);
    /// ```
    #[inline]
    pub fn num_channels_nonzero(&self) -> NonZeroUsize {
        // SAFETY: AudioSamples always has at least 1 channel (mono = 1, multi = rows ≥ 1)
        // This invariant is enforced by all constructors.
        unsafe { NonZeroUsize::new_unchecked(self.data.num_channels()) }
    }

    /// Returns the number of samples per channel.
    ///
    /// This is guaranteed to be ≥ 1 by construction (empty audio is not allowed).
    #[inline]
    pub fn samples_per_channel(&self) -> usize {
        self.data.samples_per_channel()
    }

    /// Returns the number of samples per channel as a [`NonZeroUsize`].
    ///
    /// This provides a compile-time guarantee that the value is non-zero,
    /// eliminating the need for runtime validation in downstream code.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0], sample_rate!(44100));
    /// let samples = audio.samples_per_channel_nonzero();
    /// assert_eq!(samples.get(), 3);
    /// ```
    #[inline]
    pub fn samples_per_channel_nonzero(&self) -> NonZeroUsize {
        // SAFETY: AudioSamples always has at least 1 sample per channel.
        // This invariant is enforced by all constructors (empty audio is rejected).
        unsafe { NonZeroUsize::new_unchecked(self.data.samples_per_channel()) }
    }

    /// Returns the total number of samples as a [`NonZeroUsize`].
    ///
    /// This is the product of channels × samples_per_channel, both guaranteed non-zero.
    #[inline]
    pub fn total_samples_nonzero(&self) -> NonZeroUsize {
        // SAFETY: Product of two non-zero values is non-zero
        unsafe { NonZeroUsize::new_unchecked(self.num_channels() * self.samples_per_channel()) }
    }

    /// Returns the duration in seconds
    pub fn duration_seconds<F: RealFloat>(&self) -> F {
        to_precision::<F, _>(self.samples_per_channel())
            / to_precision::<F, _>(self.sample_rate.get())
    }

    /// Returns the total number of samples across all channels
    pub fn total_samples(&self) -> usize {
        self.num_channels() * self.samples_per_channel()
    }

    /// Returns the number of bytes per sample for type T
    pub const fn bytes_per_sample(&self) -> usize {
        self.data.bytes_per_sample()
    }

    /// Returns the sample type as a string
    pub const fn sample_type() -> SampleType {
        T::SAMPLE_TYPE
    }

    /// Returns the channel layout
    pub const fn layout(&self) -> ChannelLayout {
        self.layout
    }

    /// Returns true if this is mono audio
    pub const fn is_mono(&self) -> bool {
        self.data.is_mono()
    }

    /// Returns true if this is multi-channel audio
    pub const fn is_multi_channel(&self) -> bool {
        self.data.is_multi_channel()
    }

    /// Returns the total number of samples.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if there are no samples.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the shape of the audio data.
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
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
    /// Nothing. This method is infallible.
    ///
    /// # Example
    /// ```rust,ignore
    /// // Halve the amplitude of all samples
    /// audio.apply(|sample| sample * 0.5);
    ///
    /// // Apply a simple distortion
    /// audio.apply(|sample| sample.clamp(-0.8, 0.8));
    /// ```
    pub fn apply<F>(&mut self, f: F)
    where
        F: Fn(T) -> T,
    {
        self.data.apply(f)
    }

    /// Apply a function to specific channels.
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
    /// let quieter_audio = audio.map(|sample| sample * 0.5);
    ///
    /// // Create a new audio instance with clipped samples
    /// let clipped_audio = audio.map(|sample| sample.clamp(-0.8, 0.8));
    /// ```
    pub fn map<F>(&self, f: F) -> AudioSamples<'static, T>
    where
        F: Fn(T) -> T,
    {
        let new_data = self.data.mapv(f);
        AudioSamples::from_owned(new_data, self.sample_rate)
    }

    /// Map each sample to a new type using a function.
    /// Does not care about in-domain or out-of-domain mapping.
    /// i.e. both convert_to and cast_from/into are acceptable.
    pub fn map_into<O: AudioSample, F>(&self, f: F) -> AudioSamples<'static, O>
    where
        F: Fn(T) -> O,
    {
        let new_data = AudioData::from_owned(self.data.mapv(f));
        AudioSamples::from_owned(new_data, self.sample_rate)
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
    /// Nothing. This method is infallible.
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
    /// });
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
    #[cfg(feature = "core-ops")]
    pub fn processing<'b>(&'b mut self) -> ProcessingBuilder<'b, T>
    where
        'b: 'a,
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
        AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
    {
        ProcessingBuilder::new(self)
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
    pub fn slice_samples<R>(&self, sample_range: R) -> AudioSampleResult<AudioSamples<'_, T>>
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
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "sample_range",
                format!(
                    "Sample range {}..{} out of bounds for {} samples",
                    start, end, samples_per_channel
                ),
            )));
        }

        match &self.data {
            AudioData::Mono(arr) => {
                let sliced = arr.slice(ndarray::s![start..end]).into();
                Ok(AudioSamples::from_borrowed_with_layout(
                    sliced,
                    self.sample_rate(),
                    self.layout,
                ))
            }
            AudioData::Multi(arr) => {
                let sliced = arr.slice(ndarray::s![.., start..end]).into();
                Ok(AudioSamples::from_borrowed_with_layout(
                    sliced,
                    self.sample_rate(),
                    self.layout,
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
    pub fn slice_channels<'iter, R>(&'iter self, channel_range: R) -> AudioSampleResult<AudioSamples<'static, T>>
    where
        'iter: 'a,
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
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "channel_range",
                format!(
                    "Channel range {}..{} out of bounds for {} channels",
                    start, end, num_channels
                ),
            )));
        }

        match &self.data {
            AudioData::Mono(arr) => {
                if start != 0 || end != 1 {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "channel_range",
                        format!(
                            "Channel range {}..{} invalid for mono audio (only 0..1 allowed)",
                            start, end
                        ),
                    )));
                }
                let audio_data = AudioData::from(arr.as_view());
                let audio = AudioSamples::from_owned(audio_data.into_owned(), self.sample_rate);
                Ok(audio)
            }
            AudioData::Multi(arr) => {
                let sliced = arr.slice(ndarray::s![start..end, ..]);
                if end - start == 1 {
                    // Single channel result - requires owned data due to temporary
                    let mono_data = sliced.index_axis(ndarray::Axis(0), 0).to_owned().into();
                    let audio = AudioSamples::from_owned(mono_data, self.sample_rate);
                    Ok(audio)
                } else {
                    // Multi-channel result - return owned data for consistency
                    let audio = AudioSamples::from_owned(sliced.to_owned().into(), self.sample_rate);
                    Ok(audio)
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
    pub fn slice_both<CR, SR>(
        &self,
        channel_range: CR,
        sample_range: SR,
    ) -> AudioSampleResult<AudioSamples<'static, T>>
    where
        CR: RangeBounds<isize> + Clone,
        SR: RangeBounds<isize> + Clone,
    {
        let num_channels = self.num_channels() as isize;
        let samples_per_channel = self.samples_per_channel() as isize;

        // --- Helper closure for normalising negative indices ---
        let norm = |idx: isize, len: isize| -> usize {
            if idx < 0 {
                (len + idx).max(0) as usize
            } else {
                idx.min(len) as usize
            }
        };

        // --- Channel range ---
        let ch_start = match channel_range.start_bound() {
            Bound::Included(&n) => norm(n, num_channels),
            Bound::Excluded(&n) => norm(n + 1, num_channels),
            Bound::Unbounded => 0,
        };
        let ch_end = match channel_range.end_bound() {
            Bound::Included(&n) => norm(n + 1, num_channels),
            Bound::Excluded(&n) => norm(n, num_channels),
            Bound::Unbounded => num_channels as usize,
        };

        // --- Sample range ---
        let s_start = match sample_range.start_bound() {
            Bound::Included(&n) => norm(n, samples_per_channel),
            Bound::Excluded(&n) => norm(n + 1, samples_per_channel),
            Bound::Unbounded => 0,
        };
        let s_end = match sample_range.end_bound() {
            Bound::Included(&n) => norm(n + 1, samples_per_channel),
            Bound::Excluded(&n) => norm(n, samples_per_channel),
            Bound::Unbounded => samples_per_channel as usize,
        };

        // --- Validate computed ranges ---
        if ch_start >= num_channels as usize || ch_end > num_channels as usize || ch_start >= ch_end
        {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "channel_range",
                format!(
                    "Channel range {}..{} out of bounds for {} channels",
                    ch_start, ch_end, num_channels
                ),
            )));
        }

        if s_start >= samples_per_channel as usize
            || s_end > samples_per_channel as usize
            || s_start >= s_end
        {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "sample_range",
                format!(
                    "Sample range {}..{} out of bounds for {} samples",
                    s_start, s_end, samples_per_channel
                ),
            )));
        }

        // --- Perform actual slicing ---
        match &self.data {
            AudioData::Mono(arr) => {
                if ch_start != 0 || ch_end != 1 {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "channel_range",
                        format!(
                            "Channel range {}..{} invalid for mono audio (only 0..1 allowed)",
                            ch_start, ch_end
                        ),
                    )));
                }
                let sliced = arr.slice(ndarray::s![s_start..s_end]).to_owned().into();
                Ok(AudioSamples::from_owned(sliced, self.sample_rate))
            }
            AudioData::Multi(arr) => {
                let sliced = arr.slice(ndarray::s![ch_start..ch_end, s_start..s_end]);
                if ch_end - ch_start == 1 {
                    let mono_data = sliced.index_axis(ndarray::Axis(0), 0).to_owned();
                    Ok(AudioSamples::from_owned(mono_data.into(), self.sample_rate))
                } else {
                    Ok(AudioSamples::from_owned(
                        sliced.to_owned().into(),
                        self.sample_rate,
                    ))
                }
            }
        }
    }

    /// Returns a contiguous byte view when possible.
    pub fn bytes(&self) -> AudioSampleResult<AudioBytes<'_>> {
        self.data.bytes()
    }

    /// Convert audio samples to raw bytes, preferring zero-copy when possible.
    pub fn into_bytes(&self) -> Vec<u8> {
        self.data.into_bytes()
    }

    /// Deprecated: use `bytes()` or `into_bytes()` instead.
    #[deprecated(note = "use bytes() for borrowed access or into_bytes() for owned bytes")]
    pub fn as_bytes(&self) -> Vec<u8> {
        self.into_bytes()
    }

    /// Returns the total size in bytes of all audio data.
    ///
    /// This is equivalent to `self.data.len() * self.bytes_per_sample()`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::AudioSamples;
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0], 44100);
    /// assert_eq!(audio.total_byte_size(), 12); // 3 samples × 4 bytes per f32
    /// ```
    pub fn total_byte_size(&self) -> usize {
        self.data.len() * self.bytes_per_sample()
    }

    /// Convert audio samples to bytes in the specified endianness.
    ///
    /// Returns a vector of bytes representing the audio data in the requested
    /// byte order. For cross-platform compatibility and when working with
    /// audio file formats that specify endianness.
    ///
    /// # Feature Gate
    /// This method requires the `serialization` feature.
    ///
    /// # Examples
    /// ```
    /// # #[cfg(feature = "serialization")]
    /// # {
    /// use audio_samples::{AudioSamples, operations::Endianness};
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(array![1000i16, 2000, 3000], 44100);
    ///
    /// let native_bytes = audio.as_bytes_with_endianness(Endianness::Native);
    /// let big_endian_bytes = audio.as_bytes_with_endianness(Endianness::Big);
    /// let little_endian_bytes = audio.as_bytes_with_endianness(Endianness::Little);
    ///
    /// assert_eq!(native_bytes.len(), 6); // 3 samples × 2 bytes per i16
    /// # }
    /// ```
    #[cfg(feature = "serialization")]
    pub fn as_bytes_with_endianness(&self, endianness: crate::operations::Endianness) -> Vec<u8> {
        use crate::operations::Endianness;

        match endianness {
            Endianness::Native => self.into_bytes(),
            Endianness::Big => {
                let mut result = Vec::with_capacity(self.total_byte_size());
                match &self.data {
                    AudioData::Mono(m) => {
                        for sample in m.iter() {
                            result.extend_from_slice(sample.to_be_bytes().as_ref());
                        }
                    }
                    AudioData::Multi(m) => {
                        for sample in m.iter() {
                            result.extend_from_slice(sample.to_be_bytes().as_ref());
                        }
                    }
                }
                result
            }
            Endianness::Little => {
                let mut result = Vec::with_capacity(self.total_byte_size());
                match &self.data {
                    AudioData::Mono(m) => {
                        for sample in m.iter() {
                            result.extend_from_slice(sample.to_le_bytes().as_ref());
                        }
                    }
                    AudioData::Multi(m) => {
                        for sample in m.iter() {
                            result.extend_from_slice(sample.to_le_bytes().as_ref());
                        }
                    }
                }
                result
            }
        }
    }

    /// Apply a windowed processing function to the audio data.
    ///
    /// This method processes the audio in overlapping windows, applying
    /// the provided function to each window and updating the audio data
    /// with the results.
    ///
    /// # Arguments
    /// * `window_size` - Size of each processing window in samples
    /// * `hop_size` - Number of samples to advance between windows
    /// * `func` - Function called for each window, receiving `(current_window, prev_window)`
    ///
    /// # Returns
    /// Returns `Ok(())` on success, or an error if parameters are invalid.
    pub fn apply_windowed<F>(
        &mut self,
        window_size: usize,
        hop_size: usize,
        func: F,
    ) -> AudioSampleResult<()>
    where
        F: Fn(&[T], &[T]) -> Vec<T>,
    {
        self.data.apply_windowed(window_size, hop_size, func)
    }

    /// Converts this AudioSamples into an owned version with 'static lifetime.
    ///
    /// This method takes ownership of the AudioSamples and ensures all data is owned,
    /// allowing it to be moved freely without lifetime constraints.
    pub fn into_owned<'b>(self) -> AudioSamples<'b, T> {
        AudioSamples {
            data: self.data.into_owned(),
            sample_rate: self.sample_rate,
            layout: self.layout,
        }
    }

    /// Replaces the audio data while preserving sample rate and channel layout.
    ///
    /// This method allows you to swap out the underlying audio data with new data
    /// of the same channel configuration, maintaining the existing sample rate and
    /// layout metadata.
    ///
    /// # Arguments
    /// * `new_data` - The new audio data to replace the current data
    ///
    /// # Returns
    /// Returns `Ok(())` on success, or an error if the new data has incompatible dimensions.
    ///
    /// # Errors
    /// - Returns `ParameterError` if the new data has a different number of channels
    /// - Returns `ParameterError` if mono/multi-channel layout mismatch occurs
    ///
    /// # Examples
    /// ```rust
    /// use audio_samples::{AudioSamples, AudioData};
    /// use ndarray::array;
    ///
    /// // Create initial audio
    /// let initial_data = array![1.0f32, 2.0, 3.0];
    /// let mut audio = AudioSamples::new_mono(initial_data, 44100);
    ///
    /// // Replace with new data
    /// let new_data = AudioData::new_mono(array![4.0f32, 5.0, 6.0, 7.0]);
    /// audio.replace_data(new_data)?;
    ///
    /// assert_eq!(audio.samples_per_channel(), 4);
    /// assert_eq!(audio.sample_rate(), 44100); // Preserved
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn replace_data(&mut self, new_data: AudioData<'a, T>) -> AudioSampleResult<()> {
        // Validate channel count compatibility
        let current_channels = self.data.num_channels();
        let new_channels = new_data.num_channels();

        if current_channels != new_channels {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "new_data",
                format!(
                    "Channel count mismatch: existing audio has {} channels, new data has {} channels",
                    current_channels, new_channels
                ),
            )));
        }

        // Validate mono/multi layout compatibility
        match (&self.data, &new_data) {
            (AudioData::Mono(_), AudioData::Multi(_)) => {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "new_data",
                    "Cannot replace mono audio data with multi-channel data",
                )));
            }
            (AudioData::Multi(_), AudioData::Mono(_)) => {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "new_data",
                    "Cannot replace multi-channel audio data with mono data",
                )));
            }
            _ => {} // Same layout types are fine
        }

        // Replace the data while preserving metadata
        self.data = new_data;
        Ok(())
    }

    /// Replaces the audio data with new mono data from an owned Array1.
    ///
    /// This is a convenience method for replacing mono audio data while preserving
    /// sample rate and layout metadata.
    ///
    /// # Arguments
    /// * `data` - New mono audio data as an owned Array1
    ///
    /// # Returns
    /// Returns `Ok(())` on success, or an error if the current audio is not mono.
    ///
    /// # Errors
    /// - Returns `ParameterError` if the current audio is not mono
    ///
    /// # Examples
    /// ```rust
    /// use audio_samples::AudioSamples;
    /// use ndarray::array;
    ///
    /// // Create initial mono audio
    /// let mut audio = AudioSamples::new_mono(array![1.0f32, 2.0], 44100);
    ///
    /// // Replace with new mono data
    /// audio.replace_with_mono(array![3.0f32, 4.0, 5.0])?;
    ///
    /// assert_eq!(audio.samples_per_channel(), 3);
    /// assert_eq!(audio.sample_rate(), 44100); // Preserved
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn replace_with_mono(&mut self, data: Array1<T>) -> AudioSampleResult<()> {
        if !self.is_mono() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_type",
                format!(
                    "Cannot replace multi-channel audio ({} channels) with mono data",
                    self.num_channels()
                ),
            )));
        }

        let new_data = AudioData::from(data).into_owned();
        self.replace_data(new_data)
    }

    /// Replaces the audio data with new multi-channel data from an owned Array2.
    ///
    /// This is a convenience method for replacing multi-channel audio data while preserving
    /// sample rate and layout metadata.
    ///
    /// # Arguments
    /// * `data` - New multi-channel audio data as an owned Array2 where rows are channels
    ///
    /// # Returns
    /// Returns `Ok(())` on success, or an error if channel count doesn't match.
    ///
    /// # Errors
    /// - Returns `ParameterError` if the current audio is mono
    /// - Returns `ParameterError` if the new data has different number of channels
    ///
    /// # Examples
    /// ```rust
    /// use audio_samples::AudioSamples;
    /// use ndarray::array;
    ///
    /// // Create initial stereo audio
    /// let mut audio = AudioSamples::new_multi_channel(
    ///     array![[1.0f32, 2.0], [3.0, 4.0]], 44100
    /// );
    ///
    /// // Replace with new stereo data (different length is OK)
    /// audio.replace_with_multi(array![[5.0f32, 6.0, 7.0], [8.0, 9.0, 10.0]])?;
    ///
    /// assert_eq!(audio.samples_per_channel(), 3);
    /// assert_eq!(audio.num_channels(), 2);
    /// assert_eq!(audio.sample_rate(), 44100); // Preserved
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn replace_with_multi(&mut self, data: Array2<T>) -> AudioSampleResult<()> {
        let new_channels = data.nrows();

        if self.is_mono() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_type",
                "Cannot replace mono audio with multi-channel data",
            )));
        }

        if self.num_channels() != new_channels {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "new_data",
                format!(
                    "Channel count mismatch: existing audio has {} channels, new data has {} channels",
                    self.num_channels(),
                    new_channels
                ),
            )));
        }

        let new_data = AudioData::from(data).into_owned();
        self.replace_data(new_data)
    }

    /// Replaces the audio data with new data from a Vec, maintaining current channel layout.
    ///
    /// This is a convenience method for replacing audio data with samples from a Vec.
    /// The method infers whether to create mono or multi-channel data based on the
    /// current AudioSamples configuration.
    ///
    /// # Arguments
    /// * `samples` - Vector of samples to replace the current audio data
    ///
    /// # Returns
    /// Returns `Ok(())` on success, or an error if the sample count is incompatible.
    ///
    /// # Errors
    /// - Returns `ParameterError` if the sample count is not divisible by current channel count
    /// - Returns `ParameterError` if the resulting array cannot be created
    ///
    /// # Examples
    /// ```rust
    /// use audio_samples::AudioSamples;
    /// use ndarray::array;
    ///
    /// // Replace mono audio
    /// let mut mono_audio = AudioSamples::new_mono(array![1.0f32, 2.0], 44100);
    /// mono_audio.replace_with_vec(vec![3.0f32, 4.0, 5.0, 6.0])?;
    /// assert_eq!(mono_audio.samples_per_channel(), 4);
    ///
    /// // Replace stereo audio (interleaved samples)
    /// let mut stereo_audio = AudioSamples::new_multi_channel(
    ///     array![[1.0f32, 2.0], [3.0, 4.0]], 44100
    /// );
    /// stereo_audio.replace_with_vec(vec![5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0])?;
    /// assert_eq!(stereo_audio.samples_per_channel(), 3); // 6 samples ÷ 2 channels
    /// assert_eq!(stereo_audio.num_channels(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn replace_with_vec(&mut self, samples: Vec<T>) -> AudioSampleResult<()> {
        let num_channels = self.num_channels();

        if !samples.len().is_multiple_of(num_channels) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "samples",
                format!(
                    "Sample count {} is not divisible by channel count {}",
                    samples.len(),
                    num_channels
                ),
            )));
        }

        match num_channels {
            1 => {
                // Mono case: create Array1 directly
                let array = Array1::from(samples);
                self.replace_with_mono(array)
            }
            _ => {
                // Multi-channel case: reshape into (channels, samples_per_channel)
                let samples_per_channel = samples.len() / num_channels;
                let array = Array2::from_shape_vec((num_channels, samples_per_channel), samples)
                    .map_err(|e| {
                        AudioSampleError::Parameter(ParameterError::invalid_value(
                            "samples",
                            format!(
                                "Failed to reshape samples into {}×{} array: {}",
                                num_channels, samples_per_channel, e
                            ),
                        ))
                    })?;
                self.replace_with_multi(array)
            }
        }
    }

    /// Replaces the audio data with new data from a Vec, validating against a specific channel count.
    ///
    /// This method is similar to `replace_with_vec` but validates the sample count against
    /// the provided `expected_channels` parameter rather than the current audio configuration.
    /// This is useful when the current audio might be a placeholder or when you want to
    /// ensure the data matches a specific channel configuration.
    ///
    /// # Arguments
    /// * `samples` - Vector of samples to replace the current audio data
    /// * `expected_channels` - Expected number of channels for validation
    ///
    /// # Returns
    /// Returns `Ok(())` on success, or an error if validation fails.
    ///
    /// # Errors
    /// - Returns `ParameterError` if the sample count is not divisible by `expected_channels`
    /// - Returns `ParameterError` if the current audio doesn't match `expected_channels`
    /// - Returns `ParameterError` if the resulting array cannot be created
    ///
    /// # Examples
    /// ```rust
    /// use audio_samples::AudioSamples;
    /// use ndarray::array;
    ///
    /// // Create placeholder stereo audio
    /// let mut audio = AudioSamples::new_multi_channel(
    ///     array![[0.0f32, 0.0], [0.0, 0.0]], 44100
    /// );
    ///
    /// // Replace with data that must be stereo (2 channels)
    /// let samples = vec![1.0f32, 2.0, 3.0, 4.0]; // 4 samples = 2 samples per channel
    /// audio.replace_with_vec_channels(samples, 2)?;
    ///
    /// assert_eq!(audio.samples_per_channel(), 2);
    /// assert_eq!(audio.num_channels(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn replace_with_vec_channels(
        &mut self,
        samples: Vec<T>,
        expected_channels: usize,
    ) -> AudioSampleResult<()> {
        // Validate sample count against expected channels
        if !samples.len().is_multiple_of(expected_channels) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "samples",
                format!(
                    "Sample count {} is not divisible by expected channel count {}",
                    samples.len(),
                    expected_channels
                ),
            )));
        }

        // Validate that current audio matches expected channels
        if self.num_channels() != expected_channels {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "expected_channels",
                format!(
                    "Current audio has {} channels, but expected {} channels",
                    self.num_channels(),
                    expected_channels
                ),
            )));
        }

        match expected_channels {
            1 => {
                // Mono case: create Array1 directly
                let array = Array1::from(samples);
                self.replace_with_mono(array)
            }
            _ => {
                // Multi-channel case: reshape into (channels, samples_per_channel)
                let samples_per_channel = samples.len() / expected_channels;
                let array =
                    Array2::from_shape_vec((expected_channels, samples_per_channel), samples)
                        .map_err(|e| {
                            AudioSampleError::Parameter(ParameterError::invalid_value(
                                "samples",
                                format!(
                                    "Failed to reshape samples into {}×{} array: {}",
                                    expected_channels, samples_per_channel, e
                                ),
                            ))
                        })?;
                self.replace_with_multi(array)
            }
        }
    }

    /// Returns a reference to the underlying mono array if this is mono audio.
    ///
    /// # Returns
    /// Some reference to the mono array if this is mono audio, None otherwise.
    pub const fn as_mono(&self) -> Option<&MonoData<'a, T>> {
        match &self.data {
            AudioData::Mono(arr) => Some(arr),
            AudioData::Multi(_) => None,
        }
    }

    /// Returns a reference to the underlying multi-channel array if this is multi-channel audio.
    ///
    /// # Returns
    /// Some reference to the multi-channel array if this is multi-channel audio, None otherwise.
    pub const fn as_multi_channel(&self) -> Option<&MultiData<'a, T>> {
        match &self.data {
            AudioData::Mono(_) => None,
            AudioData::Multi(arr) => Some(arr),
        }
    }

    /// Returns a mutable reference to the underlying mono array if this is mono audio.
    ///
    /// # Returns
    /// Some mutable reference to the mono array if this is mono audio, None otherwise.
    pub const fn as_mono_mut(&mut self) -> Option<&mut MonoData<'a, T>> {
        match &mut self.data {
            AudioData::Mono(arr) => Some(arr),
            AudioData::Multi(_) => None,
        }
    }

    /// Returns a mutable reference to the underlying multi-channel array if this is multi-channel audio.
    ///
    /// # Returns
    /// Some mutable reference to the multi-channel array if this is multi-channel audio, None otherwise.
    pub const fn as_multi_channel_mut(&mut self) -> Option<&mut MultiData<'a, T>> {
        match &mut self.data {
            AudioData::Mono(_) => None,
            AudioData::Multi(arr) => Some(arr),
        }
    }

    /// Creates a new mono AudioSamples from a slice.
    ///
    /// # Panics
    /// - If `slice` is empty
    pub fn new_mono_from_slice(slice: &'a [T], sample_rate: NonZeroU32) -> Self {
        assert!(!slice.is_empty(), "slice must not be empty");
        let arr = ArrayView1::from(slice);
        let mono_data = MonoData::from_view(arr);
        let audio_data = AudioData::Mono(mono_data);
        AudioSamples {
            data: audio_data,
            sample_rate,
            layout: ChannelLayout::NonInterleaved,
        }
    }

    /// Creates a new mono AudioSamples from a mutable slice.
    ///
    /// # Panics
    /// - If `slice` is empty
    pub fn new_mono_from_mut_slice(slice: &'a mut [T], sample_rate: NonZeroU32) -> Self {
        assert!(!slice.is_empty(), "slice must not be empty");
        let arr = ArrayViewMut1::from(slice);
        let mono_data = MonoData::from_view_mut(arr);
        let audio_data = AudioData::Mono(mono_data);
        AudioSamples {
            data: audio_data,
            sample_rate,
            layout: ChannelLayout::NonInterleaved,
        }
    }

    /// Creates a new multi-channel AudioSamples from a slice.
    ///
    /// # Panics
    /// - If `slice` is empty
    /// - If the length of the slice is not divisible by the number of channels.
    /// - If the slice cannot be reshaped into the desired shape.
    ///
    pub fn new_multi_channel_from_slice(
        slice: &'a [T],
        channels: usize,
        sample_rate: NonZeroU32,
    ) -> Self {
        assert!(!slice.is_empty(), "slice must not be empty");
        let total_samples = slice.len();
        assert!(
            total_samples.is_multiple_of(channels),
            "Slice length must be divisible by number of channels"
        );
        let samples_per_channel = total_samples / channels;
        let arr = ArrayView2::from_shape((channels, samples_per_channel), slice)
            .expect("Failed to create ArrayView2 from slice");
        let multi_data = MultiData::from_view(arr);
        let audio_data = AudioData::Multi(multi_data);
        AudioSamples {
            data: audio_data,
            sample_rate,
            layout: ChannelLayout::Interleaved,
        }
    }

    /// Creates a new multi-channel AudioSamples from a mutable slice.
    ///
    /// # Panics
    /// - If `slice` is empty
    /// - If the length of the slice is not divisible by the number of channels.
    /// - If the slice cannot be reshaped into the desired shape.
    ///
    pub fn new_multi_channel_from_mut_slice(
        slice: &'a mut [T],
        channels: usize,
        sample_rate: NonZeroU32,
    ) -> Self {
        assert!(!slice.is_empty(), "slice must not be empty");
        let total_samples = slice.len();
        assert!(
            total_samples.is_multiple_of(channels),
            "Slice length must be divisible by number of channels"
        );
        let samples_per_channel = total_samples / channels;
        let arr = ArrayViewMut2::from_shape((channels, samples_per_channel), slice)
            .expect("Failed to create ArrayViewMut2 from slice");
        let multi_data = MultiData::from_view_mut(arr);
        let audio_data = AudioData::Multi(multi_data);
        AudioSamples {
            data: audio_data,
            sample_rate,
            layout: ChannelLayout::Interleaved,
        }
    }

    /// Creates mono AudioSamples from a Vec (owned).
    ///
    /// # Panics
    /// - If `vec` is empty
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    ///
    /// let samples = vec![0.1f32, 0.2, 0.3, 0.4];
    /// let audio = AudioSamples::from_vec(samples, sample_rate!(44100));
    /// assert_eq!(audio.num_channels(), 1);
    /// assert_eq!(audio.samples_per_channel(), 4);
    /// ```
    pub fn from_vec(vec: Vec<T>, sample_rate: NonZeroU32) -> AudioSamples<'static, T> {
        assert!(!vec.is_empty(), "vec must not be empty");
        AudioSamples {
            data: AudioData::from_vec(vec),
            sample_rate,
            layout: ChannelLayout::NonInterleaved,
        }
    }

    /// Creates multi-channel AudioSamples from a Vec with specified channel count (owned).
    ///
    /// The samples in the Vec are assumed to be in row-major order (all samples for
    /// channel 0, then all samples for channel 1, etc.).
    ///
    /// # Panics
    /// - If `vec` is empty
    /// - If `vec.len()` is not divisible by `channels`
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    ///
    /// // 2 channels, 3 samples each: [ch0_s0, ch0_s1, ch0_s2, ch1_s0, ch1_s1, ch1_s2]
    /// let samples = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
    /// let audio = AudioSamples::from_vec_with_channels(samples, 2, sample_rate!(44100));
    /// assert_eq!(audio.num_channels(), 2);
    /// assert_eq!(audio.samples_per_channel(), 3);
    /// ```
    pub fn from_vec_with_channels(
        vec: Vec<T>,
        channels: usize,
        sample_rate: NonZeroU32,
    ) -> AudioSamples<'static, T> {
        assert!(!vec.is_empty(), "vec must not be empty");
        AudioSamples {
            data: AudioData::from_vec_multi(vec, channels),
            sample_rate,
            layout: ChannelLayout::Interleaved,
        }
    }

    /// Creates AudioSamples from interleaved sample data (owned).
    ///
    /// Interleaved format: [L0, R0, L1, R1, L2, R2, ...] for stereo.
    /// This method de-interleaves the data into the internal non-interleaved format.
    ///
    /// # Panics
    /// - If `samples` is empty
    /// - If `samples.len()` is not divisible by `channels`
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    ///
    /// // Interleaved stereo: [L0, R0, L1, R1]
    /// let interleaved = vec![0.1f32, 0.4, 0.2, 0.5];
    /// let audio = AudioSamples::from_interleaved_vec(interleaved, 2, sample_rate!(44100));
    /// assert_eq!(audio.num_channels(), 2);
    /// assert_eq!(audio.samples_per_channel(), 2);
    /// // After de-interleaving: channel 0 = [0.1, 0.2], channel 1 = [0.4, 0.5]
    /// ```
    pub fn from_interleaved_vec(
        samples: Vec<T>,
        channels: usize,
        sample_rate: NonZeroU32,
    ) -> AudioSamples<'static, T> {
        assert!(!samples.is_empty(), "samples must not be empty");
        assert!(
            samples.len().is_multiple_of(channels),
            "Sample count must be divisible by number of channels"
        );

        if channels == 1 {
            return AudioSamples::from_vec(samples, sample_rate);
        }

        let samples_per_channel = samples.len() / channels;

        // Use optimized deinterleave functions
        let deinterleaved = crate::simd_conversions::deinterleave_multi_vec(samples, channels)
            .expect("Deinterleave failed - this should not happen with valid input");

        let arr = Array2::from_shape_vec((channels, samples_per_channel), deinterleaved)
            .expect("Failed to create Array2 from deinterleaved samples");

        AudioSamples {
            data: AudioData::Multi(MultiData::from_owned(arr)),
            sample_rate,
            layout: ChannelLayout::Interleaved,
        }
    }

    /// Creates AudioSamples from interleaved slice data (borrowed).
    ///
    /// Note: This creates a borrowed view but requires de-interleaving, so it
    /// actually creates owned data internally. For zero-copy borrowed access
    /// to interleaved data, use `new_multi_channel_from_slice` with pre-deinterleaved data.
    ///
    /// # Panics
    /// - If `samples` is empty
    /// - If `samples.len()` is not divisible by `channels`
    pub fn from_interleaved_slice(
        samples: &[T],
        channels: usize,
        sample_rate: NonZeroU32,
    ) -> AudioSamples<'static, T> {
        // Must copy since we need to de-interleave
        AudioSamples::from_interleaved_vec(samples.to_vec(), channels, sample_rate)
    }

    /// Creates multi-channel AudioSamples from separate channel arrays.
    ///
    /// Each element in the vector represents one channel of audio data.
    /// All channels must have the same length.
    ///
    /// # Errors
    /// Returns an error if:
    /// - channels is empty
    /// - any channel is empty (has no samples)
    /// - channels have different lengths
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    ///
    /// let left = vec![0.1f32, 0.2, 0.3];
    /// let right = vec![0.4, 0.5, 0.6];
    /// let audio = AudioSamples::from_channels(vec![left, right], sample_rate!(44100)).unwrap();
    /// assert_eq!(audio.num_channels(), 2);
    /// assert_eq!(audio.samples_per_channel(), 3);
    /// ```
    pub fn from_channels(
        channels: Vec<Vec<T>>,
        sample_rate: NonZeroU32,
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        if channels.is_empty() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "channels",
                "Cannot create AudioSamples from empty channel list",
            )));
        }

        let num_channels = channels.len();
        let samples_per_channel = channels[0].len();

        if samples_per_channel == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "channels",
                "Channels must not be empty (no samples)",
            )));
        }

        // Validate all channels have the same length
        for (idx, ch) in channels.iter().enumerate() {
            if ch.len() != samples_per_channel {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "channels",
                    format!(
                        "Channel {} has {} samples, but channel 0 has {} samples",
                        idx,
                        ch.len(),
                        samples_per_channel
                    ),
                )));
            }
        }

        if num_channels == 1 {
            return Ok(AudioSamples::from_vec(
                channels.into_iter().next().unwrap(),
                sample_rate,
            ));
        }

        // Flatten channels into row-major order
        let flat: Vec<T> = channels.into_iter().flatten().collect();
        let arr =
            Array2::from_shape_vec((num_channels, samples_per_channel), flat).map_err(|e| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "channels",
                    format!("Failed to create multi-channel array: {}", e),
                ))
            })?;

        Ok(AudioSamples {
            data: AudioData::Multi(MultiData::from_owned(arr)),
            sample_rate,
            layout: ChannelLayout::NonInterleaved,
        })
    }

    /// Creates multi-channel AudioSamples from separate mono AudioSamples.
    ///
    /// All input AudioSamples must be mono and have the same sample rate and length.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Any input is not mono
    /// - Sample rates don't match
    /// - Lengths don't match
    /// - Input is empty
    ///
    /// # Examples
    /// ```
    /// use audio_samples::AudioSamples;
    /// use ndarray::array;
    ///
    /// let left = AudioSamples::new_mono(array![0.1f32, 0.2, 0.3], 44100);
    /// let right = AudioSamples::new_mono(array![0.4, 0.5, 0.6], 44100);
    /// let stereo = AudioSamples::from_mono_channels(vec![left, right]).unwrap();
    /// assert_eq!(stereo.num_channels(), 2);
    /// ```
    pub fn from_mono_channels(
        channels: Vec<AudioSamples<'_, T>>,
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        if channels.is_empty() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "channels",
                "Cannot create AudioSamples from empty channel list",
            )));
        }

        let sample_rate = channels[0].sample_rate();
        let samples_per_channel = channels[0].samples_per_channel();

        // Validate all channels
        for (idx, ch) in channels.iter().enumerate() {
            if !ch.is_mono() {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "channels",
                    format!(
                        "Channel {} is not mono (has {} channels)",
                        idx,
                        ch.num_channels()
                    ),
                )));
            }
            if ch.sample_rate() != sample_rate {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "channels",
                    format!(
                        "Channel {} has sample rate {}, but channel 0 has sample rate {}",
                        idx,
                        ch.sample_rate(),
                        sample_rate
                    ),
                )));
            }
            if ch.samples_per_channel() != samples_per_channel {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "channels",
                    format!(
                        "Channel {} has {} samples, but channel 0 has {} samples",
                        idx,
                        ch.samples_per_channel(),
                        samples_per_channel
                    ),
                )));
            }
        }

        // Extract data from each mono channel
        let channel_data: Vec<Vec<T>> = channels
            .into_iter()
            .map(|ch| match ch.data {
                AudioData::Mono(m) => m.to_vec(),
                AudioData::Multi(_) => unreachable!("Already validated as mono"),
            })
            .collect();

        AudioSamples::from_channels(channel_data, sample_rate)
    }

    /// Creates AudioSamples from an iterator of samples (mono, owned).
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    ///
    /// let audio = AudioSamples::from_iter((0..100).map(|i| i as f32 / 100.0), sample_rate!(44100));
    /// assert_eq!(audio.num_channels(), 1);
    /// assert_eq!(audio.samples_per_channel(), 100);
    /// ```
    pub fn from_iter<I>(iter: I, sample_rate: NonZeroU32) -> AudioSamples<'static, T>
    where
        I: IntoIterator<Item = T>,
    {
        let vec: Vec<T> = iter.into_iter().collect();
        AudioSamples::from_vec(vec, sample_rate)
    }

    /// Creates AudioSamples from an iterator with specified channel count (owned).
    ///
    /// Samples are collected and arranged in row-major order (all samples for
    /// channel 0, then channel 1, etc.).
    ///
    /// # Panics
    /// - If the collected sample count is not divisible by `channels`
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    ///
    /// // 2 channels, 50 samples each = 100 total samples
    /// let audio = AudioSamples::from_iter_with_channels(
    ///     (0..100).map(|i| i as f32 / 100.0),
    ///     2,
    ///     sample_rate!(44100)
    /// );
    /// assert_eq!(audio.num_channels(), 2);
    /// assert_eq!(audio.samples_per_channel(), 50);
    /// ```
    pub fn from_iter_with_channels<I>(
        iter: I,
        channels: usize,
        sample_rate: NonZeroU32,
    ) -> AudioSamples<'static, T>
    where
        I: IntoIterator<Item = T>,
    {
        let vec: Vec<T> = iter.into_iter().collect();
        AudioSamples::from_vec_with_channels(vec, channels, sample_rate)
    }
}

impl<'a, T: AudioSample> Clone for AudioSamples<'a, T> {
    fn clone(&self) -> Self {
        AudioSamples {
            data: self.data.clone(),
            sample_rate: self.sample_rate,
            layout: self.layout,
        }
    }
}

impl<'a, T: AudioSample> TryFrom<(u32, u32, Vec<T>)> for AudioSamples<'a, T> {
    type Error = AudioSampleError;
    /// Create AudioSamples from a sample rate and a vector of samples (assumed mono).
    fn try_from(
        (n_channels, sample_rate, samples): (u32, u32, Vec<T>),
    ) -> Result<Self, Self::Error> {
        let sample_rate = NonZeroU32::new(sample_rate).ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "sample_rate",
                "sample_rate must be greater than 0",
            ))
        })?;
        match n_channels {
            1 => {
                let arr = Array1::from(samples);
                Ok(AudioSamples::new_mono(arr, sample_rate))
            }
            _ => {
                let shape = (n_channels as usize, samples.len() / n_channels as usize);
                let arr = match Array2::from_shape_vec(shape, samples) {
                    Ok(arr) => arr,
                    Err(e) => {
                        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                            "samples",
                            format!(
                                "Failed to reshape samples into {} channels: {}",
                                n_channels, e
                            ),
                        )));
                    }
                };
                Ok(AudioSamples::new_multi_channel(arr, sample_rate))
            }
        }
    }
}

/// Create mono AudioSamples from (Vec<T>, sample_rate) tuple.
///
/// # Examples
/// ```
/// use audio_samples::AudioSamples;
/// use std::convert::TryFrom;
///
/// let audio = AudioSamples::try_from((vec![0.1f32, 0.2, 0.3], 44100_u32)).unwrap();
/// assert_eq!(audio.num_channels(), 1);
/// assert_eq!(audio.samples_per_channel(), 3);
/// ```
impl<T: AudioSample> TryFrom<(Vec<T>, u32)> for AudioSamples<'static, T> {
    type Error = AudioSampleError;

    fn try_from((samples, sample_rate): (Vec<T>, u32)) -> Result<Self, Self::Error> {
        if samples.is_empty() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "samples",
                "Cannot create AudioSamples from empty vector",
            )));
        }
        let sample_rate = NonZeroU32::new(sample_rate).ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "sample_rate",
                "sample_rate must be greater than 0",
            ))
        })?;
        Ok(AudioSamples::from_vec(samples, sample_rate))
    }
}

/// Create mono AudioSamples from (&[T], sample_rate) tuple (creates owned copy).
///
/// # Examples
/// ```
/// use audio_samples::AudioSamples;
/// use std::convert::TryFrom;
///
/// let data = [0.1f32, 0.2, 0.3];
/// let audio = AudioSamples::try_from((data.as_slice(), 44100_u32)).unwrap();
/// assert_eq!(audio.num_channels(), 1);
/// assert_eq!(audio.samples_per_channel(), 3);
/// ```
impl<T: AudioSample> TryFrom<(&[T], u32)> for AudioSamples<'static, T> {
    type Error = AudioSampleError;

    fn try_from((samples, sample_rate): (&[T], u32)) -> Result<Self, Self::Error> {
        if samples.is_empty() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "samples",
                "Cannot create AudioSamples from empty slice",
            )));
        }
        let sample_rate = NonZeroU32::new(sample_rate).ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "sample_rate",
                "sample_rate must be greater than 0",
            ))
        })?;
        Ok(AudioSamples::from_vec(samples.to_vec(), sample_rate))
    }
}

/// Create multi-channel AudioSamples from (Vec<T>, channels, sample_rate) tuple.
///
/// # Examples
/// ```
/// use audio_samples::AudioSamples;
/// use std::convert::TryFrom;
///
/// // 2 channels, 2 samples each
/// let audio = AudioSamples::try_from((vec![0.1f32, 0.2, 0.3, 0.4], 2_usize, 44100_u32)).unwrap();
/// assert_eq!(audio.num_channels(), 2);
/// assert_eq!(audio.samples_per_channel(), 2);
/// ```
impl<T: AudioSample> TryFrom<(Vec<T>, usize, u32)> for AudioSamples<'static, T> {
    type Error = AudioSampleError;

    fn try_from(
        (samples, channels, sample_rate): (Vec<T>, usize, u32),
    ) -> Result<Self, Self::Error> {
        if samples.is_empty() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "samples",
                "Cannot create AudioSamples from empty vector",
            )));
        }
        if channels == 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "channels",
                "Channel count must be at least 1",
            )));
        }
        if samples.len() % channels != 0 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "samples",
                format!(
                    "Sample count {} is not divisible by channel count {}",
                    samples.len(),
                    channels
                ),
            )));
        }
        let sample_rate = NonZeroU32::new(sample_rate).ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "sample_rate",
                "sample_rate must be greater than 0",
            ))
        })?;
        Ok(AudioSamples::from_vec_with_channels(
            samples,
            channels,
            sample_rate,
        ))
    }
}

// ========================
// Index and IndexMut implementations using ndarray delegation
// ========================

impl<'a, T: AudioSample> Index<usize> for AudioSamples<'a, T> {
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
            AudioData::Multi(_) => {
                panic!(
                    "Cannot use single index on multi-channel audio. Use (channel, sample) indexing instead."
                );
            }
        }
    }
}

impl<'a, T: AudioSample> IndexMut<usize> for AudioSamples<'a, T> {
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
            AudioData::Multi(_) => {
                panic!(
                    "Cannot use single index on multi-channel audio. Use (channel, sample) indexing instead."
                );
            }
        }
    }
}

impl<'a, T: AudioSample> Index<(usize, usize)> for AudioSamples<'a, T> {
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
            AudioData::Multi(arr) => &arr[(channel, sample)],
        }
    }
}

impl<'a, T: AudioSample> IndexMut<(usize, usize)> for AudioSamples<'a, T> {
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
            AudioData::Multi(arr) => &mut arr[(channel, sample)],
        }
    }
}

impl<'a, T: AudioSample> Index<[usize; 2]> for AudioSamples<'a, T> {
    type Output = T;

    /// Index into audio samples by [channel, sample] coordinates.
    ///
    /// This works for both mono and multi-channel audio:
    /// - For mono: only `[0, sample_index]` is valid
    /// - For multi-channel: `[channel_index, sample_index]`
    ///
    /// # Panics
    /// - If channel or sample index is out of bounds
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let channel = index[0];
        let sample = index[1];
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
            AudioData::Multi(arr) => &arr[(channel, sample)],
        }
    }
}

impl<'a, T: AudioSample> IntoIterator for AudioSamples<'a, T> {
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
            impl<T: AudioSample> std::ops::$trait<Self> for AudioSamples<'_, T> {
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
            impl<T: AudioSample> std::ops::$trait<T> for AudioSamples<'_, T> {
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
            impl<T: AudioSample> std::ops::$assign_trait<Self> for AudioSamples<'_, T> {
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
            impl<T: AudioSample> std::ops::$assign_trait<T> for AudioSamples<'_, T> {
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
impl<T: AudioSample> Neg for AudioSamples<'_, T>
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

/// A wrapper for AudioSamples that guarantees exactly 2 channels (stereo).
pub struct StereoAudioSamples<'a, T: AudioSample>(pub AudioSamples<'a, T>);

impl<'a, T: AudioSample> StereoAudioSamples<'a, T> {
    /// Creates a new `StereoAudioSamples` from audio data with exactly 2 channels.
    ///
    /// # Errors
    /// Returns an error if `stereo_data` is mono or has a channel count other than 2.
    pub fn new(stereo_data: AudioData<'a, T>, sample_rate: u32) -> AudioSampleResult<Self> {
        // Separated failure conditions which the following if statements check allow for more descriptive errors.

        match stereo_data.num_channels() {
            1 => {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "channels",
                    "Expected stereo data (2 channels), got mono (1 channel).",
                )));
            }
            2 => (),
            _ => {
                return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "channels",
                    format!(
                        "Expected stereo data (2 channels), got {} channels.",
                        stereo_data.num_channels()
                    ),
                )));
            }
        };

        // From now on we have guaranteed that the stereo_data does in fact have 2 channels;
        Ok(Self(AudioSamples::new(
            stereo_data,
            std::num::NonZeroU32::new(sample_rate).ok_or_else(|| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "sample_rate",
                    "Sample rate must be non-zero",
                ))
            })?,
        )))
    }

    /// Safely access individual channels with borrowed references for efficient processing.
    ///
    /// This method provides zero-copy access to the left and right channels, allowing
    /// efficient operations like STFT on individual channels without data duplication.
    ///
    /// # Arguments
    /// * `f` - Closure that receives borrowed left and right channel data
    ///
    /// # Returns
    /// The result of the closure operation
    ///
    /// # Example
    /// ```rust
    /// # use audio_samples::{AudioSamples, StereoAudioSamples};
    /// # use ndarray::array;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let stereo_data = array![[0.1f32, 0.2, 0.3], [0.4, 0.5, 0.6]];
    /// let audio = AudioSamples::new_multi_channel(stereo_data, 44100);
    /// let stereo: StereoAudioSamples<'static, f32> = StereoAudioSamples::try_from(audio)?;
    ///
    /// stereo.with_channels(|left, right| {
    ///     // left and right are borrowed AudioSamples<'_, f32>
    ///     println!("Left channel samples: {}", left.len());
    ///     println!("Right channel samples: {}", right.len());
    ///     Ok(())
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_channels<F, R>(&self, f: F) -> AudioSampleResult<R>
    where
        F: FnOnce(AudioSamples<'_, T>, AudioSamples<'_, T>) -> AudioSampleResult<R>,
    {
        match &self.0.data {
            AudioData::Multi(multi_data) => {
                // Extract left channel (row 0)
                let left_view = multi_data.index_axis(Axis(0), 0);
                let left_data = MonoData::from_view(left_view);
                let left_audio =
                    AudioSamples::new(AudioData::Mono(left_data), self.0.sample_rate());

                // Extract right channel (row 1)
                let right_view = multi_data.index_axis(Axis(0), 1);
                let right_data = MonoData::from_view(right_view);
                let right_audio =
                    AudioSamples::new(AudioData::Mono(right_data), self.0.sample_rate());

                f(left_audio, right_audio)
            }
            AudioData::Mono(_) => {
                unreachable!("StereoAudioSamples guarantees exactly 2 channels")
            }
        }
    }
}

impl<'a, T: AudioSample> Deref for StereoAudioSamples<'a, T> {
    type Target = AudioSamples<'a, T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: AudioSample> DerefMut for StereoAudioSamples<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, T: AudioSample> AsRef<AudioSamples<'a, T>> for StereoAudioSamples<'a, T> {
    fn as_ref(&self) -> &AudioSamples<'a, T> {
        &self.0
    }
}

impl<'a, T: AudioSample> AsMut<AudioSamples<'a, T>> for StereoAudioSamples<'a, T> {
    fn as_mut(&mut self) -> &mut AudioSamples<'a, T> {
        &mut self.0
    }
}

/// Zero-copy conversion from owned AudioSamples to StereoAudioSamples
impl<T: AudioSample> TryFrom<AudioSamples<'static, T>> for StereoAudioSamples<'static, T> {
    type Error = AudioSampleError;

    fn try_from(audio: AudioSamples<'static, T>) -> Result<Self, Self::Error> {
        if audio.num_channels() != 2 {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "channels",
                format!(
                    "Expected exactly 2 channels for stereo audio, but found {}",
                    audio.num_channels()
                ),
            )));
        }

        match audio.data {
            AudioData::Multi(_) => Ok(StereoAudioSamples(audio)),
            AudioData::Mono(_) => Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Cannot convert mono audio to stereo",
            ))),
        }
    }
}

impl<T: AudioSample> From<StereoAudioSamples<'static, T>> for AudioSamples<'static, T> {
    fn from(stereo: StereoAudioSamples<'static, T>) -> Self {
        stereo.0
    }
}

#[cfg(feature = "fixed-size-audio")]
/// Fixed-size audio samples buffer for stack-allocated audio data.
///
/// This type provides a way to work with audio samples that have a compile-time
/// known maximum size, allowing for stack allocation and zero-copy operations.
pub struct FixedSizeAudioSamples<T: AudioSample, const N: usize> {
    /// The underlying audio samples
    pub samples: AudioSamples<'static, T>,
}

#[cfg(feature = "fixed-size-audio")]
impl<T: AudioSample, const N: usize> FixedSizeAudioSamples<T, N> {
    /// Create a fixed-size audio buffer from a 1D array
    ///
    /// # Panics
    /// Panics if `sample_rate` is 0.
    pub fn from_1d(data: [T; N], sample_rate: u32) -> Self {
        let sample_rate = NonZeroU32::new(sample_rate).expect("sample_rate must be non-zero");
        let array = Array1::from_vec(data.to_vec());
        let mono_data = MonoData::from_owned(array);
        let audio_data = AudioData::Mono(mono_data);
        let audio = AudioSamples::from_owned(audio_data, sample_rate);
        Self { samples: audio }
    }

    /// Get the maximum capacity of this buffer
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Unsafe because it does have debug assertions to ensure the two AudioSamples have matching sample rates and channel counts.
    /// The caller is responsible for ensuring these conditions are met.
    pub unsafe fn swap_samples(&mut self, other: &mut Self) {
        debug_assert_eq!(
            self.samples.sample_rate(),
            other.samples.sample_rate(),
            "Sample rates must match for swap"
        );
        debug_assert_eq!(
            self.samples.num_channels(),
            other.samples.num_channels(),
            "Number of channels must match for swap"
        );

        std::mem::swap(&mut self.samples, &mut other.samples);
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
        let audio: AudioSamples<f32> = AudioSamples::new_mono(data, sample_rate!(44100));

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio.num_channels(), 1);
        assert_eq!(audio.samples_per_channel(), 5);
        assert!(audio.is_mono());
        assert!(!audio.is_multi_channel());
    }

    #[test]
    fn test_new_multi_channel_audio_samples() {
        let data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 2 channels, 3 samples each
        let audio = AudioSamples::new_multi_channel(data, sample_rate!(48000));

        assert_eq!(audio.sample_rate(), sample_rate!(48000));
        assert_eq!(audio.num_channels(), 2);
        assert_eq!(audio.samples_per_channel(), 3);
        assert_eq!(audio.total_samples(), 6);
        assert!(!audio.is_mono());
        assert!(audio.is_multi_channel());
    }

    #[test]
    fn test_zeros_construction() {
        let mono_audio = AudioSamples::<f32>::zeros_mono(100, sample_rate!(44100));
        assert_eq!(mono_audio.num_channels(), 1);
        assert_eq!(mono_audio.samples_per_channel(), 100);
        assert_eq!(mono_audio.sample_rate(), sample_rate!(44100));

        let multi_audio = AudioSamples::<f32>::zeros_multi(2, 50, sample_rate!(48000));
        assert_eq!(multi_audio.num_channels(), 2);
        assert_eq!(multi_audio.samples_per_channel(), 50);
        assert_eq!(multi_audio.total_samples(), 100);
        assert_eq!(multi_audio.sample_rate(), sample_rate!(48000));
    }

    #[cfg(feature = "spectral-analysis")]
    #[test]
    fn test_stereo_channel_processing_workflow() {
        use crate::operations::traits::AudioTransforms;
        use crate::operations::types::WindowType;

        // Create stereo test data (left = sine wave, right = cosine wave)
        let samples_per_channel = 1024;
        let sample_rate = 44100;
        let frequency = 440.0; // A4 note
        let duration = samples_per_channel as f64 / sample_rate as f64;

        let mut stereo_data = Array2::zeros((2, samples_per_channel));
        for i in 0..samples_per_channel {
            let t = i as f64 * duration / samples_per_channel as f64;
            let phase = 2.0 * std::f64::consts::PI * frequency * t;
            stereo_data[[0, i]] = (phase.sin() * 0.5) as f32; // Left channel
            stereo_data[[1, i]] = (phase.cos() * 0.5) as f32; // Right channel
        }

        let audio = AudioSamples::new_multi_channel(stereo_data, sample_rate!(44100));

        // Zero-copy conversion to StereoAudioSamples
        let stereo: StereoAudioSamples<'static, f32> = StereoAudioSamples::try_from(audio).unwrap();
        assert_eq!(stereo.num_channels(), 2);
        assert_eq!(stereo.samples_per_channel(), samples_per_channel);

        // Test the with_channels workflow (similar to the user's Python code)
        let result = stereo.with_channels(|left, right| {
            // These are borrowed AudioSamples<'_, f32> - zero copy!
            assert_eq!(left.num_channels(), 1);
            assert_eq!(right.num_channels(), 1);
            assert_eq!(left.samples_per_channel(), samples_per_channel);
            assert_eq!(right.samples_per_channel(), samples_per_channel);

            // Perform STFT on each channel (like librosa.stft in Python)
            let window_size = 512;
            let hop_size = 256;
            let left_stft = left.stft(window_size, hop_size, WindowType::<f32>::Hanning)?;
            let right_stft = right.stft(window_size, hop_size, WindowType::<f32>::Hanning)?;

            // Verify STFT results
            assert_eq!(left_stft.dim().0, window_size / 2 + 1); // Frequency bins
            assert_eq!(right_stft.dim().0, window_size / 2 + 1); // Frequency bins
            assert!(left_stft.dim().1 > 0); // Time frames
            assert!(right_stft.dim().1 > 0); // Time frames

            Ok(())
        });

        assert!(result.is_ok());
    }

    #[test]
    fn test_duration_seconds() {
        let audio: AudioSamples<'_, f32> =
            AudioSamples::<f32>::zeros_mono(44100, sample_rate!(44100));
        assert!((audio.duration_seconds::<f64>() - 1.0).abs() < 1e-6);

        let audio2: AudioSamples<'_, f32> =
            AudioSamples::<f32>::zeros_multi(2, 22050, sample_rate!(44100));
        assert!((audio2.duration_seconds::<f64>() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_apply_simple() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut audio: AudioSamples<f32> = AudioSamples::new_mono(data, sample_rate!(44100));

        // Apply a simple scaling
        audio.apply(|sample| sample * 2.0);

        let expected = array![2.0f32, 4.0, 6.0, 8.0, 10.0];
        let expected = AudioSamples::new_mono(expected, sample_rate!(44100));
        assert_eq!(
            audio, expected,
            "Applied audio samples do not match expected values"
        );
    }

    #[test]
    fn test_apply_channels() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio: AudioSamples<f32> =
            AudioSamples::new_multi_channel(data, sample_rate!(44100));

        {
            // Mutable borrow lives only within this block
            audio.apply_to_channels(&[0, 1], |sample| sample * sample);
        } // Mutable borrow ends here

        let expected = array![[1.0, 4.0], [9.0, 16.0]];
        let expected = AudioSamples::new_multi_channel(expected, sample_rate!(44100));
        assert_eq!(
            audio, expected,
            "Applied multi-channel audio samples do not match expected values"
        );
    }

    #[test]
    fn test_map_functional() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100));

        // Create a new audio instance with transformed samples
        let new_audio = audio.map(|sample| sample * 0.5);

        // Original should be unchanged
        let original_expected = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let original_expected = AudioSamples::new_mono(original_expected, sample_rate!(44100));
        assert_eq!(
            audio, original_expected,
            "Original audio samples should remain unchanged"
        );

        // New audio should contain transformed values
        let new_expected = array![0.5f32, 1.0, 1.5, 2.0, 2.5];
        let new_expected = AudioSamples::new_mono(new_expected, sample_rate!(44100));
        assert_eq!(
            new_audio, new_expected,
            "New audio should contain transformed samples"
        );
    }

    #[test]
    fn test_apply_indexed() {
        let data = array![1.0f32, 1.0, 1.0, 1.0, 1.0];
        let mut audio = AudioSamples::new_mono(data, sample_rate!(44100));

        // Apply index-based transformation
        audio.apply_with_index(|index, sample| sample * (index + 1) as f32);
        let expected = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let expected = AudioSamples::new_mono(expected, sample_rate!(44100));
        assert_eq!(
            audio, expected,
            "Indexed applied audio samples do not match expected values"
        );
    }

    // ========================
    // Indexing and Slicing Tests
    // ========================

    #[test]
    fn test_index_mono_single() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100));

        assert_eq!(audio[0], 1.0);
        assert_eq!(audio[2], 3.0);
        assert_eq!(audio[4], 5.0);
    }

    #[test]
    #[should_panic(expected = "Cannot use single index on multi-channel audio")]
    fn test_index_mono_single_on_multi_panics() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let audio = AudioSamples::new_multi_channel(data, sample_rate!(44100));

        let _ = audio[0]; // Should panic
    }

    #[test]
    fn test_index_tuple() {
        // Test mono with tuple indexing
        let mono_data = array![1.0f32, 2.0, 3.0, 4.0];
        let mono_audio = AudioSamples::new_mono(mono_data, sample_rate!(44100));

        assert_eq!(mono_audio[(0, 1)], 2.0);
        assert_eq!(mono_audio[(0, 3)], 4.0);

        // Test multi-channel with tuple indexing
        let stereo_data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let stereo_audio = AudioSamples::new_multi_channel(stereo_data, sample_rate!(44100));

        assert_eq!(stereo_audio[(0, 0)], 1.0);
        assert_eq!(stereo_audio[(0, 2)], 3.0);
        assert_eq!(stereo_audio[(1, 0)], 4.0);
        assert_eq!(stereo_audio[(1, 2)], 6.0);
    }

    #[test]
    #[should_panic(expected = "Channel index 1 out of bounds for mono audio")]
    fn test_index_tuple_invalid_channel_mono() {
        let data = array![1.0f32, 2.0, 3.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100));

        let _ = audio[(1, 0)]; // Should panic - mono only has channel 0
    }

    #[test]
    fn test_index_mut_mono() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut audio = AudioSamples::new_mono(data, sample_rate!(44100));

        audio[2] = 10.0;
        assert_eq!(audio[2], 10.0);
        assert_eq!(
            audio,
            AudioSamples::new_mono(array![1.0f32, 2.0, 10.0, 4.0, 5.0], sample_rate!(44100)),
            "Mutably indexed mono audio samples do not match expected values"
        );
    }

    #[test]
    fn test_index_mut_tuple() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio = AudioSamples::new_multi_channel(data, sample_rate!(44100));

        audio[(1, 0)] = 10.0;
        assert_eq!(audio[(1, 0)], 10.0);

        let expected = array![[1.0f32, 2.0], [10.0, 4.0]];
        let expected = AudioSamples::new_multi_channel(expected, sample_rate!(44100));
        assert_eq!(
            audio, expected,
            "Mutably indexed audio samples do not match expected values"
        );
    }

    // ========================
    // Audio Data Replacement Tests
    // ========================

    #[test]
    fn test_replace_data_mono_success() {
        let initial_data = array![1.0f32, 2.0, 3.0];
        let mut audio = AudioSamples::new_mono(initial_data, sample_rate!(44100));

        // Replace with new mono data of different length
        let new_data = AudioData::from(array![4.0f32, 5.0, 6.0, 7.0, 8.0]).into_owned();
        assert!(audio.replace_data(new_data).is_ok());

        assert_eq!(audio.samples_per_channel(), 5);
        assert_eq!(audio.num_channels(), 1);
        assert_eq!(audio.sample_rate(), sample_rate!(44100)); // Should be preserved
        assert_eq!(audio.layout(), ChannelLayout::NonInterleaved); // Should be preserved
    }

    #[test]
    fn test_replace_data_multi_success() {
        let initial_data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio = AudioSamples::new_multi_channel(initial_data, sample_rate!(44100));

        // Replace with new stereo data of different length
        let new_data = AudioData::from(array![[5.0f32, 6.0, 7.0], [8.0, 9.0, 10.0]]);
        assert!(audio.replace_data(new_data).is_ok());

        assert_eq!(audio.samples_per_channel(), 3);
        assert_eq!(audio.num_channels(), 2);
        assert_eq!(audio.sample_rate(), sample_rate!(44100)); // Should be preserved
        assert_eq!(audio.layout(), ChannelLayout::Interleaved); // Should be preserved
    }

    #[test]
    fn test_replace_data_channel_count_mismatch() {
        let initial_data = array![1.0f32, 2.0, 3.0];
        let mut audio = AudioSamples::new_mono(initial_data, sample_rate!(44100));

        // Try to replace mono with stereo data
        let new_data = AudioData::from(array![[4.0f32, 5.0], [6.0, 7.0]]);
        let result = audio.replace_data(new_data);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Channel count mismatch"));
        }
    }

    #[test]
    fn test_replace_data_layout_type_mismatch() {
        let initial_data = array![1.0f32, 2.0, 3.0];
        let mut audio = AudioSamples::new_mono(initial_data, sample_rate!(44100));

        // Try to replace mono with multi-channel data (even same channel count fails due to type)
        let new_data = AudioData::from(array![[4.0f32, 5.0, 6.0]]);
        let result = audio.replace_data(new_data);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string()
                    .contains("Cannot replace mono audio data with multi-channel data")
            );
        }
    }

    #[test]
    fn test_replace_with_mono_success() {
        let initial_data = array![1.0f32, 2.0];
        let mut audio = AudioSamples::new_mono(initial_data, sample_rate!(44100));

        // Replace with new mono data
        assert!(audio.replace_with_mono(array![3.0f32, 4.0, 5.0]).is_ok());

        assert_eq!(audio.samples_per_channel(), 3);
        assert_eq!(audio.num_channels(), 1);
        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio[0], 3.0);
        assert_eq!(audio[1], 4.0);
        assert_eq!(audio[2], 5.0);
    }

    #[test]
    fn test_replace_with_mono_on_multi_fails() {
        let initial_data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio = AudioSamples::new_multi_channel(initial_data, sample_rate!(44100));

        let result = audio.replace_with_mono(array![5.0f32, 6.0, 7.0]);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Cannot replace multi-channel audio"));
        }
    }

    #[test]
    fn test_replace_with_multi_success() {
        let initial_data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio = AudioSamples::new_multi_channel(initial_data, sample_rate!(44100));

        // Replace with new stereo data of different length
        assert!(
            audio
                .replace_with_multi(array![[5.0f32, 6.0, 7.0], [8.0, 9.0, 10.0]])
                .is_ok()
        );

        assert_eq!(audio.samples_per_channel(), 3);
        assert_eq!(audio.num_channels(), 2);
        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio[(0, 0)], 5.0);
        assert_eq!(audio[(0, 2)], 7.0);
        assert_eq!(audio[(1, 0)], 8.0);
        assert_eq!(audio[(1, 2)], 10.0);
    }

    #[test]
    fn test_replace_with_multi_on_mono_fails() {
        let initial_data = array![1.0f32, 2.0, 3.0];
        let mut audio = AudioSamples::new_mono(initial_data, sample_rate!(44100));

        let result = audio.replace_with_multi(array![[4.0f32, 5.0], [6.0, 7.0]]);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string()
                    .contains("Cannot replace mono audio with multi-channel data")
            );
        }
    }

    #[test]
    fn test_replace_with_multi_channel_count_mismatch() {
        let initial_data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio = AudioSamples::new_multi_channel(initial_data, sample_rate!(44100));

        // Try to replace stereo with tri-channel data
        let result = audio.replace_with_multi(array![[5.0f32, 6.0], [7.0, 8.0], [9.0, 10.0]]);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Channel count mismatch"));
        }
    }

    #[test]
    fn test_replace_with_vec_mono_success() {
        let initial_data = array![1.0f32, 2.0];
        let mut audio = AudioSamples::new_mono(initial_data, sample_rate!(44100));

        // Replace with vector
        assert!(audio.replace_with_vec(vec![3.0f32, 4.0, 5.0, 6.0]).is_ok());

        assert_eq!(audio.samples_per_channel(), 4);
        assert_eq!(audio.num_channels(), 1);
        assert_eq!(audio.sample_rate(), sample_rate!(44100));
    }

    #[test]
    fn test_replace_with_vec_multi_success() {
        let initial_data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio = AudioSamples::new_multi_channel(initial_data, sample_rate!(44100));

        // Replace with vector (6 samples = 3 samples per channel × 2 channels)
        assert!(
            audio
                .replace_with_vec(vec![5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0])
                .is_ok()
        );

        assert_eq!(audio.samples_per_channel(), 3);
        assert_eq!(audio.num_channels(), 2);
        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        // Data is arranged as: [ch0_sample0, ch0_sample1, ch0_sample2, ch1_sample0, ch1_sample1, ch1_sample2]
        assert_eq!(audio[(0, 0)], 5.0); // First channel, first sample
        assert_eq!(audio[(0, 1)], 6.0); // First channel, second sample
        assert_eq!(audio[(1, 0)], 8.0); // Second channel, first sample
    }

    #[test]
    fn test_replace_with_vec_not_divisible_fails() {
        let initial_data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio = AudioSamples::new_multi_channel(initial_data, sample_rate!(44100));

        // 5 samples is not divisible by 2 channels
        let result = audio.replace_with_vec(vec![5.0f32, 6.0, 7.0, 8.0, 9.0]);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string()
                    .contains("Sample count 5 is not divisible by channel count 2")
            );
        }
    }

    #[test]
    fn test_user_workflow_example() {
        // Simulate the user's use case
        let mut audio =
            AudioSamples::new_multi_channel(array![[0.0f32, 0.0], [0.0, 0.0]], sample_rate!(44100));

        // Simulate converted data from their function
        let converted_samples: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let num_channels = audio.num_channels();

        // This should work exactly like their use case
        assert!(converted_samples.len().is_multiple_of(num_channels));
        assert!(audio.replace_with_vec(converted_samples).is_ok());

        assert_eq!(audio.samples_per_channel(), 3);
        assert_eq!(audio.num_channels(), 2);
        assert_eq!(audio.sample_rate(), sample_rate!(44100)); // Preserved
        assert_eq!(audio.layout(), ChannelLayout::Interleaved); // Preserved
    }

    #[test]
    fn test_replace_with_vec_channels_success() {
        // Test with correct channel count
        let initial_data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio = AudioSamples::new_multi_channel(initial_data, sample_rate!(44100));

        // Replace with vector, validating against expected 2 channels
        assert!(
            audio
                .replace_with_vec_channels(vec![5.0f32, 6.0, 7.0, 8.0], 2)
                .is_ok()
        );

        assert_eq!(audio.samples_per_channel(), 2);
        assert_eq!(audio.num_channels(), 2);
        assert_eq!(audio.sample_rate(), sample_rate!(44100));
    }

    #[test]
    fn test_replace_with_vec_channels_sample_count_mismatch() {
        let initial_data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio = AudioSamples::new_multi_channel(initial_data, sample_rate!(44100));

        // Try with sample count not divisible by expected channels
        let result = audio.replace_with_vec_channels(vec![5.0f32, 6.0, 7.0], 2);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string()
                    .contains("Sample count 3 is not divisible by expected channel count 2")
            );
        }
    }

    #[test]
    fn test_replace_with_vec_channels_audio_channel_mismatch() {
        let initial_data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio = AudioSamples::new_multi_channel(initial_data, sample_rate!(44100));

        // Try with expected channel count different from current audio
        let result = audio.replace_with_vec_channels(vec![5.0f32, 6.0, 7.0], 3);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string()
                    .contains("Current audio has 2 channels, but expected 3 channels")
            );
        }
    }

    #[test]
    fn test_user_workflow_with_channels_validation() {
        // Simulate the user's corrected use case
        let mut audio =
            AudioSamples::new_multi_channel(array![[0.0f32, 0.0], [0.0, 0.0]], sample_rate!(44100));

        // Simulate converted data from their function
        let converted_samples: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let num_channels = 2; // This comes from the function parameter, not audio.num_channels()

        // Validate against the parameter, not the existing audio configuration
        assert!(converted_samples.len().is_multiple_of(num_channels));
        assert!(
            audio
                .replace_with_vec_channels(converted_samples, num_channels)
                .is_ok()
        );

        assert_eq!(audio.samples_per_channel(), 3); // 6 samples ÷ 2 channels
        assert_eq!(audio.num_channels(), 2);
        assert_eq!(audio.sample_rate(), sample_rate!(44100)); // Preserved
        assert_eq!(audio.layout(), ChannelLayout::Interleaved); // Preserved
    }

    #[test]
    fn test_metadata_preservation_across_replacements() {
        let mut audio = AudioSamples::new_mono(array![1.0f32, 2.0], sample_rate!(48000));
        audio.set_layout(ChannelLayout::NonInterleaved);

        let original_rate = audio.sample_rate();
        let original_layout = audio.layout();

        // Replace data multiple times
        assert!(audio.replace_with_mono(array![3.0f32, 4.0, 5.0]).is_ok());
        assert_eq!(audio.sample_rate(), original_rate);
        assert_eq!(audio.layout(), original_layout);

        assert!(audio.replace_with_vec(vec![6.0f32, 7.0, 8.0, 9.0]).is_ok());
        assert_eq!(audio.sample_rate(), original_rate);
        assert_eq!(audio.layout(), original_layout);

        let new_data = AudioData::from(array![10.0f32, 11.0]).into_owned();
        assert!(audio.replace_data(new_data).is_ok());
        assert_eq!(audio.sample_rate(), original_rate);
        assert_eq!(audio.layout(), original_layout);
    }
}
