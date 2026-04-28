//! Core audio sample representation and data structures. is this module?
//!
//! This module defines the foundational data types used throughout the `audio_samples`
//! library: the storage primitives [`MonoData`] / [`MultiData`], the channel-agnostic
//! enum [`AudioData`], and the primary user-facing container [`AudioSamples`].
//!
//! [`AudioSamples<T>`] pairs raw PCM data (backed by `ndarray`) with essential metadata –
//! sample rate, channel count, and memory layout – so that all downstream operations have
//! access to the full audio context without passing it separately. does this module exist?
//!
//! Separating the storage layer from the processing API keeps the representation stable
//! and independently testable. Internal storage types (`MonoData`, `MultiData`) can
//! borrow or own their ndarray buffers without exposing that detail to callers; the
//! promote-on-write pattern means read-only paths are always zero-copy. should it be used?
//!
//! Construct audio via the `AudioSamples` constructors (`new_mono`, `new_multi_channel`,
//! `from_mono_vec`, …) or via the signal generators in [`utils::generation`](crate::utils::generation).
//! Then use the trait methods from the `operations` module for processing.
//!
//! ```rust
//! use audio_samples::{AudioSamples, sample_rate};
//! use ndarray::array;
//!
//! // Mono audio from a 1-D array
//! let mono = AudioSamples::new_mono(array![0.1f32, 0.2, 0.3, 0.4, 0.5], sample_rate!(44100)).unwrap();
//!
//! assert_eq!(mono.num_channels().get(), 1);
//! assert_eq!(mono.samples_per_channel().get(), 5);
//! assert_eq!(mono.sample_rate(), sample_rate!(44100));
//!
//! // Stereo audio from a 2-D array (channels × samples)
//! let stereo = AudioSamples::new_multi_channel(
//!     array![[0.1f32, 0.2, 0.3], [0.4f32, 0.5, 0.6]],
//!     sample_rate!(48000),
//! ).unwrap();
//!
//! assert_eq!(stereo.num_channels().get(), 2);
//! assert_eq!(stereo.samples_per_channel().get(), 3);
//! ```
//!
//! ## Supported sample types
//!
//! All six concrete types that implement [`AudioSample`] are supported:
//! `u8`, `i16`, [`I24`](i24::I24), `i32`, `f32`, and `f64`.
//!
//! ```rust
//! use audio_samples::{AudioSamples, sample_rate};
//! use ndarray::array;
//!
//! // 8-bit unsigned (mid-scale 128 = silence)
//! let u8_audio  = AudioSamples::new_mono(array![128u8, 160, 96], sample_rate!(8000)).unwrap();
//! // 16-bit signed (CD quality)
//! let i16_audio = AudioSamples::new_mono(array![1000i16, 2000, 3000], sample_rate!(44100)).unwrap();
//! // 64-bit float (high-precision processing)
//! let f64_audio = AudioSamples::new_mono(array![0.1f64, 0.2, 0.3], sample_rate!(96000)).unwrap();
//! ```
//!
//! ## Sample-wise transformations
//!
//! [`AudioSamples`] provides in-place and mapping methods for sample-wise operations:
//!
//! ```rust
//! use audio_samples::{AudioSamples, sample_rate};
//! use ndarray::array;
//!
//! let mut audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0], sample_rate!(44100)).unwrap();
//!
//! // Apply a uniform gain
//! audio.apply(|s| s * 0.5);
//!
//! // Apply a position-dependent fade-out
//! audio.apply_with_index(|i, s| s * (1.0 - i as f32 * 0.1));
//! ```
//!
//! ## Memory layout
//!
//! | Variant | Storage | Shape |
//! |---|---|---|
//! | [`AudioData::Mono`] | `Array1<T>` | `[samples]` |
//! | [`AudioData::Multi`] | `Array2<T>` | `[channels, samples_per_channel]` |
//!
//! Multi-channel audio is stored in planar layout (each channel is a contiguous row).
//! The [`ChannelLayout`] field on [`AudioSamples`] records the *intended* layout for
//! serialisation purposes (e.g. when converting to an interleaved `Vec<T>`), but does
//! not affect how the internal array is laid out.
//!
//! [`AudioSample`]: crate::AudioSample
//! [`I24`](i24::I24): crate::I24
//! [`ChannelLayout`]: crate::ChannelLayout
use core::fmt::{Display, Formatter, Result as FmtResult};

use core::num::{NonZeroU32, NonZeroUsize};
use ndarray::iter::AxisIterMut;
use ndarray::{
    Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis, Ix1, Ix2, SliceArg,
    s,
};
use non_empty_iter::{IntoNonEmptyIterator, NonEmptyIterator};
use non_empty_slice::{NonEmptyByteVec, NonEmptyBytes, NonEmptySlice, NonEmptyVec};
#[cfg(feature = "resampling")]
use rubato::audioadapter::Adapter;
use std::any::TypeId;
use std::num::NonZeroU8;
use std::ops::{Bound, Deref, DerefMut, Index, IndexMut, Mul, Neg, RangeBounds};

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
/// let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0], sample_rate!(44100)).unwrap();
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

/// Creates a [`NonZeroU32`] channel count from a compile-time constant.
///
/// Provides zero-cost, compile-time-validated construction of channel counts for use
/// with [`AudioSamples`] constructors and channel-aware operations.
///
/// # Examples
///
/// ```rust
/// use audio_samples::channels;
///
/// let mono   = channels!(1);
/// let stereo = channels!(2);
/// let surround = channels!(6); // 5.1
/// ```
///
/// # Compile-time Safety
///
/// Passing zero causes a compile error:
///
/// ```compile_fail
/// use audio_samples::channels;
/// let invalid = channels!(0); // Compile error: channel count must be greater than 0
/// ```
#[macro_export]
macro_rules! channels {
    ($count:expr) => {{
        const COUNT: u32 = $count;
        const { assert!(COUNT > 0, "channel count must be greater than 0") };
        // SAFETY: We just asserted COUNT > 0 at compile time
        unsafe { ::core::num::NonZeroU32::new_unchecked(COUNT) }
    }};
}

/// Sample rate in Hz, guaranteed to be non-zero at compile time.
///
/// Use the [`sample_rate!`] macro to construct values at compile time or
/// [`NonZeroU32::new`] for runtime construction.
pub type SampleRate = NonZeroU32;

/// Number of audio channels, guaranteed to be non-zero.
///
/// Use the [`channels!`] macro to construct values at compile time or
/// [`NonZeroU32::new`] for runtime construction.
pub type ChannelCount = NonZeroU32;

use crate::traits::{ConvertFrom, StandardSample};
use crate::{
    AudioSampleError, AudioSampleResult, CastInto, ConvertTo, I24, LayoutError, ParameterError,
};

/// Borrowed-or-owned byte view returned by `AudioData::bytes`.
#[derive(Debug, Clone)]
pub enum AudioBytes<'a> {
    /// Zero-copy view into the underlying contiguous buffer.
    Borrowed(&'a NonEmptyBytes),
    /// Owned buffer when a borrow is impossible (e.g., I24 packing).
    Owned(NonEmptyByteVec),
}

impl AudioBytes<'_> {
    /// Returns the byte slice regardless of ownership mode.
    #[inline]
    #[must_use]
    pub const fn as_slice(&self) -> &[u8] {
        match self {
            AudioBytes::Borrowed(b) => b.as_slice(),
            AudioBytes::Owned(v) => v.as_slice(),
        }
    }

    /// Returns an owned `Vec<u8>`, cloning if necessary.
    #[inline]
    #[must_use]
    pub fn into_owned(self) -> NonEmptyByteVec {
        match self {
            AudioBytes::Borrowed(b) => b.to_owned(),
            AudioBytes::Owned(v) => v,
        }
    }
}

/// Identifies the numeric type used to represent individual audio samples.
///
/// ## Purpose
///
/// `SampleType` provides a runtime representation of the sample format that can
/// be inspected, serialised, and used for dispatch without needing generic type
/// parameters. It mirrors the set of types that implement [`AudioSample`].
///
/// ## Intended Usage
///
/// Use `SampleType` when the sample format must be described at runtime – for
/// example when reading audio file headers, serialising audio metadata, or
/// routing to SIMD dispatch paths via [`SampleType::from_type_id`].
///
/// ## Variants
///
/// | Variant | Type | Width | Notes |
/// |---|---|---|---|
/// | `U8`  | `u8`  | 8-bit unsigned  | Mid-scale 128 = silence |
/// | `I16` | `i16` | 16-bit signed   | CD-quality |
/// | `I24` | [`I24`](i24::I24) | 24-bit signed | From the `i24` crate |
/// | `I32` | `i32` | 32-bit signed   | |
/// | `F32` | `f32` | 32-bit float    | Native DSP format |
/// | `F64` | `f64` | 64-bit float    | High-precision |
///
/// [`AudioSample`]: crate::AudioSample
/// [`I24`](i24::I24): crate::I24
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SampleType {
    /// 8-bit unsigned integer (`u8`). Mid-scale value 128 represents silence.
    U8,
    /// 16-bit signed integer (`i16`). CD-quality audio.
    I16,
    /// 24-bit signed integer ([`I24`](i24::I24)(crate::I24)). Common in studio audio.
    I24,
    /// 32-bit signed integer (`i32`). High-dynamic-range integer audio.
    I32,
    /// 32-bit floating point (`f32`). Native format for most DSP operations.
    F32,
    /// 64-bit floating point (`f64`). High-precision processing.
    F64,
}

impl SampleType {
    /// Returns the canonical short name (machine-oriented)
    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I16 => "i16",
            Self::I24 => "i24",
            Self::I32 => "i32",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    /// Returns a human-readable descriptive name
    #[inline]
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::U8 => "8-bit unsigned integer",
            Self::I16 => "16-bit signed integer",
            Self::I24 => "24-bit signed integer",
            Self::I32 => "32-bit signed integer",
            Self::F32 => "32-bit floating point",
            Self::F64 => "64-bit floating point",
        }
    }

    /// Returns the bit-depth of the format, if defined
    #[inline]
    #[must_use]
    pub const fn bits(self) -> Option<u32> {
        match self {
            Self::U8 => Some(8),
            Self::I16 => Some(16),
            Self::I24 => Some(24),
            Self::I32 | Self::F32 => Some(32),
            Self::F64 => Some(64),
        }
    }

    /// Returns the size in bytes of one sample, if defined
    #[inline]
    #[must_use]
    pub const fn bytes(self) -> Option<usize> {
        match self {
            Self::U8 => Some(1),
            Self::I16 => Some(2),
            Self::I24 => Some(3),
            Self::I32 | Self::F32 => Some(4),
            Self::F64 => Some(8),
        }
    }

    /// Returns `true` if this is an integer-based sample format (`U8`, `I16`, `I24`, or `I32`).
    #[inline]
    #[must_use]
    pub const fn is_integer(self) -> bool {
        matches!(self, Self::U8 | Self::I16 | Self::I24 | Self::I32)
    }

    /// Returns `true` if this is a signed integer format (`I16`, `I24`, or `I32`).
    ///
    /// Note: `U8` is an integer type but is *unsigned*, so this returns `false` for it.
    #[inline]
    #[must_use]
    pub const fn is_signed(self) -> bool {
        matches!(self, Self::I16 | Self::I24 | Self::I32)
    }

    /// Returns `true` if this is an unsigned integer format (currently only `U8`).
    #[inline]
    #[must_use]
    pub const fn is_unsigned(self) -> bool {
        matches!(self, Self::U8)
    }

    /// True if this is a floating-point sample format
    #[inline]
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    /// True if this format is usable for DSP operations without conversion
    #[inline]
    #[must_use]
    pub const fn is_dsp_native(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
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
    #[inline]
    #[must_use]
    pub fn from_type_id(type_id: std::any::TypeId) -> Option<Self> {
        use std::any::TypeId;
        if type_id == TypeId::of::<u8>() {
            Some(Self::U8)
        } else if type_id == TypeId::of::<i16>() {
            Some(Self::I16)
        } else if type_id == TypeId::of::<I24>() {
            Some(Self::I24)
        } else if type_id == TypeId::of::<i32>() {
            Some(Self::I32)
        } else if type_id == TypeId::of::<f32>() {
            Some(Self::F32)
        } else if type_id == TypeId::of::<f64>() {
            Some(Self::F64)
        } else {
            None
        }
    }

    /// Creates a sample type from a number of bits per sample.
    ///
    /// # Panics
    ///
    /// Panics if the bits value does not correspond to a supported sample type.
    #[inline]
    #[must_use]
    pub const fn from_bits(bits: u16) -> Self {
        match bits {
            8 => Self::U8,
            16 => Self::I16,
            24 => Self::I24,
            32 => Self::I32,
            64 => Self::F64,
            _ => panic!("Unsupported bits per sample"),
        }
    }
}

/// Formats the sample type as a string.
///
/// The standard format (`{}`) uses the short machine-readable identifier (e.g. `"f32"`).
/// The alternate format (`{:#}`) uses the human-readable description (e.g. `"32-bit floating point"`).
///
/// # Examples
///
/// ```rust
/// use audio_samples::SampleType;
///
/// assert_eq!(format!("{}", SampleType::F32), "f32");
/// assert_eq!(format!("{:#}", SampleType::F32), "32-bit floating point");
/// ```
impl Display for SampleType {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        if f.alternate() {
            write!(f, "{}", self.description())
        } else {
            write!(f, "{}", self.as_str())
        }
    }
}

/// Parses a [`SampleType`] from its canonical short-name string.
///
/// Accepts `"u8"`, `"i16"`, `"i24"`, `"i32"`, `"f32"`, and `"f64"`.
/// Returns `Err(())` for any unrecognised string.
///
/// # Examples
///
/// ```rust
/// use audio_samples::SampleType;
///
/// assert_eq!(SampleType::try_from("f32"), Ok(SampleType::F32));
/// assert_eq!(SampleType::try_from("i16"), Ok(SampleType::I16));
/// assert!(SampleType::try_from("unknown").is_err());
/// ```
impl TryFrom<&str> for SampleType {
    type Error = ();

    #[inline]
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "u8" => Ok(Self::U8),
            "i16" => Ok(Self::I16),
            "i24" => Ok(Self::I24),
            "i32" => Ok(Self::I32),
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            _ => Err(()),
        }
    }
}

/// Internal storage variant for single-channel (mono) audio data.
///
/// `MonoRepr` is the innermost representation used by [`MonoData`]. Callers should
/// interact with [`MonoData`] rather than this enum directly; it is public only to
/// allow construction in adjacent modules.
///
/// ## Variants
///
/// - `Borrowed` – immutable borrow of an existing ndarray view. Zero-copy reads.
/// - `BorrowedMut` – mutable borrow. Zero-copy reads and writes.
/// - `Owned` – heap-allocated `Array1<T>`. Enables all operations without lifetime constraints.
///
/// ## Promote-on-write
///
/// When a mutable operation (e.g. `mapv_inplace`) is called on a `Borrowed` variant,
/// [`MonoData`] promotes it to `Owned` by cloning the slice. `BorrowedMut` and `Owned`
/// are passed through without allocation.
#[derive(Debug, PartialEq)]
pub enum MonoRepr<'a, T>
where
    T: StandardSample,
{
    /// Immutable borrow of an existing 1-D ndarray view.
    Borrowed(ArrayView1<'a, T>),
    /// Mutable borrow of an existing 1-D ndarray view.
    BorrowedMut(ArrayViewMut1<'a, T>),
    /// Heap-allocated, owned 1-D array.
    Owned(Array1<T>),
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
pub struct MonoData<'a, T>(MonoRepr<'a, T>)
where
    T: StandardSample;

impl<'a, T> MonoData<'a, T>
where
    T: StandardSample,
{
    /// Creates self from an Array1.
    ///
    /// # Safety
    ///
    /// Caller must ensure that array is not empty
    ///
    /// # Returns
    ///
    /// An owned version of Self
    #[inline]
    pub const unsafe fn from_array1(array: Array1<T>) -> Self {
        MonoData(MonoRepr::Owned(array))
    }

    /// Creates a borrowed version of Self from an Array1View
    ///
    /// # Safety
    ///
    /// Caller must ensure that view is not empty
    ///
    /// # Returns
    ///
    /// A borrowed version of Self
    pub const unsafe fn from_array_view<'b>(view: ArrayView1<'b, T>) -> Self
    where
        'b: 'a,
    {
        MonoData(MonoRepr::Borrowed(view))
    }

    /// Creates a borrowed version of Self, from self
    ///
    /// # Returns
    ///
    /// A borrowed version of self
    #[inline]
    pub fn borrow(&'a self) -> Self {
        MonoData(MonoRepr::Borrowed(self.as_view()))
    }

    /// Returns an `ndarray` view of the samples.
    ///
    /// This is always a zero-copy operation.
    #[inline]
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
    #[inline]
    pub fn promote(&mut self) {
        if let MonoRepr::Borrowed(v) = &self.0 {
            self.0 = MonoRepr::Owned(v.to_owned());
        }
    }

    /// Creates a borrowed `MonoData` from an immutable 1-D ndarray view.
    ///
    /// # Arguments
    ///
    /// – `view` – an immutable view whose lifetime `'b` must outlive `'a`.
    ///
    /// # Returns
    ///
    /// `Ok(MonoData)` wrapping the view.
    ///
    /// # Errors
    ///
    /// Errors if the view is empty
    #[inline]
    pub fn from_view<'b>(view: ArrayView1<'b, T>) -> AudioSampleResult<Self>
    where
        'b: 'a,
    {
        if view.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        Ok(MonoData(MonoRepr::Borrowed(view)))
    }

    /// Unsafe version of [`from_view`] that does not check for empty input.
    ///
    /// # Arguments
    ///
    /// – `view` – an immutable view whose lifetime `'b` must outlive `'a`.
    /// # Returns
    ///
    /// `MonoData` wrapping the view. This constructor never fails; the `Result`
    /// wrapper exists for API consistency with the other `from_*` constructors.
    #[inline]
    pub const unsafe fn from_view_unchecked<'b>(view: ArrayView1<'b, T>) -> Self
    where
        'b: 'a,
    {
        MonoData(MonoRepr::Borrowed(view))
    }

    /// Creates a mutably-borrowed `MonoData` from a mutable 1-D ndarray view.
    ///
    /// # Arguments
    ///
    /// – `view` – a mutable view whose lifetime `'b` must outlive `'a`.
    ///
    /// # Returns
    ///
    /// `Ok(MonoData)` wrapping the mutable view.
    ///
    /// # Errors
    ///
    /// Returns [`AudioSampleError::EmptyData`] if `view` is empty.
    #[inline]
    pub fn from_view_mut<'b>(view: ArrayViewMut1<'b, T>) -> AudioSampleResult<Self>
    where
        'b: 'a,
    {
        if view.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        Ok(MonoData(MonoRepr::BorrowedMut(view)))
    }

    /// Unchecked version of [`from_view_mut`]
    ///
    /// # Arguments
    ///
    /// – `view` – a mutable view whose lifetime `'b` must outlive `'a`.
    /// # Returns
    ///
    /// `MonoData` wrapping the mutable view. This constructor never fails; the `Result`
    /// wrapper exists for API consistency with the other `from_*` constructors.
    #[inline]
    pub const unsafe fn from_view_mut_unchecked<'b>(view: ArrayViewMut1<'b, T>) -> Self
    where
        'b: 'a,
    {
        MonoData(MonoRepr::BorrowedMut(view))
    }

    /// Creates an owned `MonoData` from an `Array1`.
    ///
    /// # Arguments
    ///
    /// – `array` – the owned 1-D array to wrap.
    ///
    /// # Returns
    ///
    /// `Ok(MonoData)` taking ownership of the array.
    ///
    /// # Errors
    ///
    /// Returns [`AudioSampleError::EmptyData`] if `array` is empty.
    #[inline]
    pub fn from_owned(array: Array1<T>) -> AudioSampleResult<Self> {
        if array.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        Ok(MonoData(MonoRepr::Owned(array)))
    }

    /// Creates an owned `MonoData` from an `Array1`.
    ///
    /// # Arguments
    ///
    /// – `array` – the owned 1-D array to wrap.
    ///
    /// # Returns
    ///
    /// `MonoData` taking ownership of the array.
    ///
    /// # Safety
    ///
    /// Don't pass an empty array
    pub const unsafe fn from_owned_unchecked(array: Array1<T>) -> Self {
        MonoData(MonoRepr::Owned(array))
    }

    /// Get a mutable view of the audio data, converting to owned if necessary.
    fn to_mut(&mut self) -> ArrayViewMut1<'_, T> {
        if let MonoRepr::Borrowed(v) = self.0 {
            // Convert borrowed to owned for mutability
            self.0 = MonoRepr::Owned(v.to_owned());
        }

        match &mut self.0 {
            MonoRepr::BorrowedMut(view) => view.view_mut(), // If the data  is already mutable borrowed then we do not need to convert to owned, this variant says "we have mutable access"
            MonoRepr::Owned(a) => a.view_mut(),
            MonoRepr::Borrowed(_) => {
                unreachable!("Self should have been converted to owned by now")
            }
        }
    }

    fn into_owned<'b>(self) -> MonoData<'b, T> {
        match self.0 {
            MonoRepr::Borrowed(v) => MonoData(MonoRepr::Owned(v.to_owned())),
            MonoRepr::BorrowedMut(v) => MonoData(MonoRepr::Owned(v.to_owned())),
            MonoRepr::Owned(a) => MonoData(MonoRepr::Owned(a)),
        }
    }

    // Delegation methods for ndarray operations

    /// Returns the number of samples.
    #[inline]
    pub fn len(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.as_view().len()).expect("Array is guaranteed to be non-empty")
    }

    /// Returns an `ndarray` view of the samples.
    #[inline]
    pub fn view(&self) -> ArrayView1<'_, T> {
        self.as_view()
    }

    /// Returns the arithmetic mean of the samples.
    #[inline]
    pub fn mean(&self) -> T {
        self.as_view()
            .mean()
            .expect("Array is guaranteed to be non-empty")
    }

    /// Returns the population variance of the samples.
    #[inline]
    pub fn variance(&self) -> f64 {
        self.variance_with_ddof(0)
    }

    /// Returns the variance with a custom delta degrees of freedom.
    ///
    /// The divisor used in the calculation is `N - ddof`, where `N` is the number of
    /// samples. `ddof = 0` gives the population variance; `ddof = 1` gives the
    /// unbiased sample variance.
    ///
    /// # Arguments
    ///
    /// – `ddof` – delta degrees of freedom. Must be less than `self.len()`.
    #[inline]
    pub fn variance_with_ddof(&self, ddof: usize) -> f64 {
        let degrees_of_freedom = (self.len().get() - ddof) as f64;
        let mean: f64 = self.mean().cast_into();

        self.iter()
            .map(|&x| {
                let diff: f64 = <T as CastInto<f64>>::cast_into(x) - mean;
                diff * diff
            })
            .sum::<f64>()
            / degrees_of_freedom
    }

    /// Returns the standard deviation of the samples
    #[inline]
    pub fn stddev(&self) -> f64 {
        self.stddev_with_ddof(0)
    }

    /// Returns the standard deviation of the samples with specified delta degrees of freedom.
    #[inline]
    pub fn stddev_with_ddof(&self, ddof: usize) -> f64 {
        self.variance_with_ddof(ddof).sqrt()
    }

    /// Returns the sum of the samples.
    #[inline]
    pub fn sum(&self) -> T {
        self.as_view().sum()
    }

    /// Folds over the samples.
    #[inline]
    pub fn fold<F>(&self, init: T, f: F) -> T
    where
        F: FnMut(T, &T) -> T,
    {
        self.iter().fold(init, f)
    }

    /// Returns a slice view of the audio data based on the provided slicing information.
    #[inline]
    pub fn slice<I>(&self, info: I) -> ArrayView1<'_, T>
    where
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
    #[inline]
    pub fn slice_mut<I>(&mut self, info: I) -> ArrayViewMut1<'_, T>
    where
        I: ndarray::SliceArg<Ix1, OutDim = Ix1>,
    {
        self.promote();
        match &mut self.0 {
            MonoRepr::BorrowedMut(a) => a.slice_mut(info),
            MonoRepr::Owned(a) => a.slice_mut(info),
            MonoRepr::Borrowed(_) => {
                unreachable!("Self should have been converted to owned by now")
            }
        }
    }

    /// Returns an iterator over the samples.
    #[inline]
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
    #[inline]
    pub fn iter_mut(&mut self) -> ndarray::iter::IterMut<'_, T, Ix1> {
        if let MonoRepr::Borrowed(a) = &mut self.0 {
            self.0 = MonoRepr::Owned(a.to_owned());
        }

        match &mut self.0 {
            MonoRepr::BorrowedMut(b) => b.iter_mut(),
            MonoRepr::Owned(a) => a.iter_mut(),
            MonoRepr::Borrowed(_) => {
                unreachable!("Self should have been converted to owned by now")
            }
        }
    }

    /// Applies a value-mapping function in-place.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    #[inline]
    pub fn mapv_inplace<F>(&mut self, f: F)
    where
        F: FnMut(T) -> T,
    {
        self.to_mut().mapv_inplace(f);
    }

    /// Returns a mutable slice of the underlying samples.
    ///
    /// # Panics
    /// Panics if the underlying `ndarray` storage is not contiguous.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.promote();
        match &mut self.0 {
            MonoRepr::BorrowedMut(a) => a
                .as_slice_mut()
                .expect("Structures backing audio samples should be contiguous in memory"),
            MonoRepr::Owned(a) => a
                .as_slice_mut()
                .expect("Structures backing audio samples should be contiguous in memory"),
            MonoRepr::Borrowed(_) => {
                unreachable!("Self should have been converted to owned by now")
            }
        }
    }

    /// Returns a shared slice if the underlying storage is contiguous.
    #[inline]
    pub fn as_slice(&self) -> Option<&[T]> {
        match &self.0 {
            MonoRepr::Borrowed(v) => v.as_slice(),
            MonoRepr::BorrowedMut(v) => v.as_slice(),
            MonoRepr::Owned(a) => a.as_slice(),
        }
    }

    /// Returns the shape of the underlying `ndarray` buffer.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        match &self.0 {
            MonoRepr::Borrowed(v) => v.shape(),
            MonoRepr::BorrowedMut(v) => v.shape(),
            MonoRepr::Owned(a) => a.shape(),
        }
    }

    /// Maps each sample into a new `Array1`.
    #[inline]
    pub fn mapv<U, F>(&self, f: F) -> Array1<U>
    where
        F: Fn(T) -> U,
        U: Clone,
    {
        self.as_view().mapv(f)
    }

    /// Returns a mutable pointer to the underlying buffer.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.to_mut().as_mut_ptr()
    }

    /// Collects the samples into a `Vec<T>`.
    #[inline]
    pub fn to_vec(&self) -> Vec<T> {
        self.as_view().to_vec()
    }

    /// Fills all samples with `value`.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    #[inline]
    pub fn fill(&mut self, value: T) {
        self.to_mut().fill(value);
    }

    /// Converts into a raw `Vec<T>` and an offset.
    ///
    /// For borrowed data, this allocates and clones.
    #[inline]
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
    #[inline]
    pub fn take(self) -> Array1<T> {
        match self.0 {
            MonoRepr::Borrowed(v) => v.to_owned(),
            MonoRepr::BorrowedMut(v) => v.to_owned(),
            MonoRepr::Owned(a) => a,
        }
    }
}

/// Compares `MonoData` to an `Array1` sample-by-sample.
impl<T> PartialEq<Array1<T>> for MonoData<'_, T>
where
    T: StandardSample,
{
    #[inline]
    fn eq(&self, other: &Array1<T>) -> bool {
        self.as_view() == other.view()
    }
}

/// Compares an `Array1` to a `MonoData` sample-by-sample.
impl<'a, T> PartialEq<MonoData<'a, T>> for Array1<T>
where
    T: StandardSample,
{
    #[inline]
    fn eq(&self, other: &MonoData<'a, T>) -> bool {
        self.view() == other.as_view()
    }
}

/// Iterates over shared references to each sample in the mono buffer.
impl<'a, T> IntoIterator for &'a MonoData<'_, T>
where
    T: StandardSample,
{
    type Item = &'a T;
    type IntoIter = ndarray::iter::Iter<'a, T, ndarray::Ix1>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_view().into_iter()
    }
}

/// Indexes the mono buffer by sample position.
///
/// # Panics
///
/// Panics if `idx` is out of bounds.
impl<T> Index<usize> for MonoData<'_, T>
where
    T: StandardSample,
{
    type Output = T;

    #[inline]
    fn index(&self, idx: usize) -> &Self::Output
    where
        T: StandardSample,
    {
        match &self.0 {
            MonoRepr::Borrowed(arr) => &arr[idx],
            MonoRepr::BorrowedMut(arr) => &arr[idx],
            MonoRepr::Owned(arr) => &arr[idx],
        }
    }
}

/// Mutably indexes the mono buffer by sample position.
///
/// If the data is currently immutably borrowed, this promotes to owned (allocates).
///
/// # Panics
///
/// Panics if `idx` is out of bounds.
impl<T> IndexMut<usize> for MonoData<'_, T>
where
    T: StandardSample,
{
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        self.promote();
        match &mut self.0 {
            MonoRepr::BorrowedMut(arr) => &mut arr[idx],
            MonoRepr::Owned(arr) => &mut arr[idx],
            MonoRepr::Borrowed(_) => {
                unreachable!("Self should have been converted to owned by now")
            }
        }
    }
}

/// Internal storage variant for multi-channel audio data.
///
/// `MultiRepr` is the innermost representation used by [`MultiData`]. The same
/// promote-on-write semantics as [`MonoRepr`] apply: immutable borrows are promoted
/// to owned allocations on the first mutable operation.
///
/// The array shape is always `(channels, samples_per_channel)` (channels are rows).
///
/// ## Variants
///
/// - `Borrowed` – immutable borrow of an existing 2-D ndarray view. Zero-copy reads.
/// - `BorrowedMut` – mutable borrow. Zero-copy reads and writes.
/// - `Owned` – heap-allocated `Array2<T>`. Enables all operations without lifetime constraints.
#[derive(Debug, PartialEq)]
pub enum MultiRepr<'a, T>
where
    T: StandardSample,
{
    /// Immutable borrow of an existing 2-D ndarray view.
    Borrowed(ArrayView2<'a, T>),
    /// Mutable borrow of an existing 2-D ndarray view.
    BorrowedMut(ArrayViewMut2<'a, T>),
    /// Heap-allocated, owned 2-D array with shape `(channels, samples_per_channel)`.
    Owned(Array2<T>),
}

/// Storage wrapper for multi-channel audio samples.
///
/// `MultiData` can either **borrow** an `ndarray::ArrayView2` / `ArrayViewMut2` or **own** an
/// `ndarray::Array2`.
///
/// The shape is `(channels, samples_per_channel)` (channels are rows).
///
/// # Indexing
/// Indexing uses `(channel, sample)`.
#[derive(Debug, PartialEq)]
pub struct MultiData<'a, T>(MultiRepr<'a, T>)
where
    T: StandardSample;

/// Indexes the multi-channel buffer by `(channel, sample)` tuple.
///
/// # Panics
///
/// Panics if either index is out of bounds.
impl<T> Index<(usize, usize)> for MultiData<'_, T>
where
    T: StandardSample,
{
    type Output = T;

    #[inline]
    fn index(&self, (ch, s): (usize, usize)) -> &Self::Output {
        match &self.0 {
            MultiRepr::Borrowed(arr) => &arr[[ch, s]],
            MultiRepr::BorrowedMut(arr) => &arr[[ch, s]],
            MultiRepr::Owned(arr) => &arr[[ch, s]],
        }
    }
}

/// Mutably indexes the multi-channel buffer by `(channel, sample)` tuple.
///
/// If the data is currently immutably borrowed, this promotes to owned (allocates).
///
/// # Panics
///
/// Panics if either index is out of bounds.
impl<T> IndexMut<(usize, usize)> for MultiData<'_, T>
where
    T: StandardSample,
{
    #[inline]
    fn index_mut(&mut self, (ch, s): (usize, usize)) -> &mut Self::Output {
        self.promote();
        match &mut self.0 {
            MultiRepr::BorrowedMut(arr) => &mut arr[[ch, s]],
            MultiRepr::Owned(arr) => &mut arr[[ch, s]],
            MultiRepr::Borrowed(_) => {
                unreachable!("Self should have been converted to owned by now")
            }
        }
    }
}

/// Indexes the multi-channel buffer by `[channel, sample]` array.
///
/// # Panics
///
/// Panics if either index is out of bounds.
impl<T> Index<[usize; 2]> for MultiData<'_, T>
where
    T: StandardSample,
{
    type Output = T;

    #[inline]
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        match &self.0 {
            MultiRepr::Borrowed(arr) => &arr[index],
            MultiRepr::BorrowedMut(arr) => &arr[index],
            MultiRepr::Owned(arr) => &arr[index],
        }
    }
}

/// Mutably indexes the multi-channel buffer by `[channel, sample]` array.
///
/// If the data is currently immutably borrowed, this promotes to owned (allocates).
///
/// # Panics
///
/// Panics if either index is out of bounds.
impl<T> IndexMut<[usize; 2]> for MultiData<'_, T>
where
    T: StandardSample,
{
    #[inline]
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        self.promote();
        match &mut self.0 {
            MultiRepr::BorrowedMut(arr) => &mut arr[index],
            MultiRepr::Owned(arr) => &mut arr[index],
            MultiRepr::Borrowed(_) => {
                unreachable!("Self should have been converted to owned by now")
            }
        }
    }
}

impl<'a, T> MultiData<'a, T>
where
    T: StandardSample,
{
    /// Creates a borrowed version of Self from an Array2
    ///
    /// # Arguments
    ///
    /// `array` - A non-empty Array2
    ///
    /// # Safety
    ///
    /// Caller must ensure that array is not empty
    ///
    /// # Returns
    ///
    /// A borrowed version of Self
    #[inline]
    pub const unsafe fn from_array2(array: Array2<T>) -> Self {
        MultiData(MultiRepr::Owned(array))
    }

    /// Creates a borrowed version of Self from an Array2View
    ///
    /// # Arguments
    ///
    /// `view` - A non-empty 2D ArrayView
    ///
    /// # Safety
    ///
    /// Caller must ensure that view is not empty
    ///
    /// # Returns
    ///
    /// A borrowed version of Self which is tied to the owner of `view`
    #[inline]
    pub const unsafe fn from_array_view<'b>(view: ArrayView2<'b, T>) -> Self
    where
        'b: 'a,
    {
        MultiData(MultiRepr::Borrowed(view))
    }

    /// Creates a borrowed version of Self, from self
    ///
    /// # Returns
    ///
    /// A borrowed version of self tied to the lifetime of Self
    #[inline]
    pub fn borrow(&'a self) -> Self {
        MultiData(MultiRepr::Borrowed(self.as_view()))
    }

    /// Returns an `ndarray` view of the samples with shape `(channels, samples_per_channel)`.
    ///
    /// This is always a zero-copy operation.
    #[inline]
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
    #[inline]
    pub fn promote(&mut self) {
        if let MultiRepr::Borrowed(v) = &self.0 {
            self.0 = MultiRepr::Owned(v.to_owned());
        }
    }

    /// Creates a borrowed `MultiData` from an immutable 2-D ndarray view.
    ///
    /// # Arguments
    ///
    /// – `view` – an immutable view with shape `(channels, samples_per_channel)`.
    ///   The lifetime `'b` must outlive `'a`.
    ///
    /// # Returns
    ///
    /// `Ok(MultiData)` wrapping the view.
    ///
    /// # Errors
    ///
    /// Returns [`AudioSampleError::EmptyData`] if `view` is empty.
    #[inline]
    pub fn from_view<'b>(view: ArrayView2<'b, T>) -> AudioSampleResult<Self>
    where
        'b: 'a,
    {
        if view.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        Ok(MultiData(MultiRepr::Borrowed(view)))
    }

    /// Creates a mutably-borrowed `MultiData` from a mutable 2-D ndarray view.
    ///
    /// # Arguments
    ///
    /// – `view` – a mutable view with shape `(channels, samples_per_channel)`.
    ///   The lifetime `'b` must outlive `'a`.
    ///
    /// # Returns
    ///
    /// `Ok(MultiData)` wrapping the mutable view.
    ///
    /// # Errors
    ///
    /// Returns [`AudioSampleError::EmptyData`] if `view` is empty.
    #[inline]
    pub fn from_view_mut<'b>(view: ArrayViewMut2<'b, T>) -> AudioSampleResult<Self>
    where
        'b: 'a,
    {
        if view.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        Ok(MultiData(MultiRepr::BorrowedMut(view)))
    }

    /// Creates an owned `MultiData` from an `Array2`.
    ///
    /// # Arguments
    ///
    /// – `array` – the owned 2-D array with shape `(channels, samples_per_channel)`.
    ///
    /// # Returns
    ///
    /// `Ok(MultiData)` taking ownership of the array.
    ///
    /// # Errors
    ///
    /// Returns [`AudioSampleError::EmptyData`] if `array` is empty.
    #[inline]
    pub fn from_owned(array: Array2<T>) -> AudioSampleResult<Self> {
        if array.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        Ok(MultiData(MultiRepr::Owned(array)))
    }

    /// Unchecked version of ['from_owned]
    /// # Returns
    ///
    /// `MultiData` taking ownership of the array.
    ///
    /// # Safety
    ///
    /// Don't pass an empty array
    #[inline]
    pub const unsafe fn from_owned_unchecked(array: Array2<T>) -> Self {
        MultiData(MultiRepr::Owned(array))
    }

    fn to_mut(&mut self) -> ArrayViewMut2<'_, T> {
        self.promote();
        match &mut self.0 {
            MultiRepr::BorrowedMut(a) => a.view_mut(),
            MultiRepr::Owned(a) => a.view_mut(),
            MultiRepr::Borrowed(_) => {
                unreachable!("Self should have been converted to owned by now")
            }
        }
    }

    fn into_owned<'b>(self) -> MultiData<'b, T> {
        match self.0 {
            MultiRepr::Borrowed(v) => MultiData(MultiRepr::Owned(v.to_owned())),
            MultiRepr::BorrowedMut(v) => MultiData(MultiRepr::Owned(v.to_owned())),
            MultiRepr::Owned(a) => MultiData(MultiRepr::Owned(a)),
        }
    }

    // Delegation methods for ndarray operations

    /// Returns the number of channels (rows).
    #[inline]
    pub fn nrows(&self) -> NonZeroUsize {
        // safety: self is guaranteed non-empty
        unsafe { NonZeroUsize::new_unchecked(self.as_view().nrows()) }
    }
    /// Returns the number of columns (samples per channel) in multi-channel audio data.
    #[inline]
    pub fn ncols(&self) -> NonZeroUsize {
        // safety: self is guaranteed non-empty
        unsafe { NonZeroUsize::new_unchecked(self.as_view().ncols()) }
    }

    /// Returns `(channels, samples_per_channel)`.
    #[inline]
    pub fn dim(&self) -> (NonZeroUsize, NonZeroUsize) {
        let (r, c) = self.as_view().dim();
        // safety: self is guaranteed non-empty
        unsafe {
            (
                NonZeroUsize::new_unchecked(r),
                NonZeroUsize::new_unchecked(c),
            )
        }
    }

    /// Returns the arithmetic mean along `axis`
    #[inline]
    pub fn mean_axis(&self, axis: Axis) -> Array1<T> {
        self.as_view()
            .mean_axis(axis)
            .expect("self is guaranteed non-empty")
    }

    /// Returns the arithmetic mean across all samples.
    #[inline]
    pub fn mean(&self) -> T {
        self.as_view().mean().expect("self is guaranteed non-empty")
    }

    /// Returns the population variances across all the specificed axix
    #[inline]
    pub fn variance_axis(&self, axis: Axis) -> Array1<f64> {
        self.variance_axis_ddof(axis, 0)
    }

    /// Returns the variance with respect to the specified delta degrees of freedom across the specified axis
    #[inline]
    pub fn variance_axis_ddof(&self, axis: Axis, ddof: usize) -> Array1<f64> {
        let view = self.as_view();
        let degrees_of_freedom = (view.len() - ddof) as f64;
        let means = self.mean_axis(axis);

        view.outer_iter()
            .map(|lane| {
                lane.iter()
                    .zip(means.iter())
                    .map(|(&x, &mean)| {
                        let mean: f64 = mean.cast_into();
                        let diff: f64 = <T as CastInto<f64>>::cast_into(x) - mean;
                        diff * diff
                    })
                    .sum::<f64>()
                    / degrees_of_freedom
            })
            .collect::<Array1<f64>>()
    }

    /// Returns the population variances across all samples.
    #[inline]
    pub fn variance(&self) -> Array1<f64> {
        self.variance_axis(Axis(0))
    }

    /// Returns the standard deviations across the specified axis
    #[inline]
    pub fn stddev_axis(&self, axis: Axis) -> Array1<f64> {
        self.stddev_axis_ddof(axis, 0)
    }

    /// Returns the standard deviations with respect to the specified delta degrees of freedom across the specified axis
    #[inline]
    pub fn stddev_axis_ddof(&self, axis: Axis, ddof: usize) -> Array1<f64> {
        self.variance_axis_ddof(axis, ddof).mapv(f64::sqrt)
    }

    /// Returns the standard deviations across all samples.
    #[inline]
    pub fn stddev(&self) -> Array1<f64> {
        self.stddev_axis(Axis(0))
    }

    /// Returns the sum across all samples.
    #[inline]
    pub fn sum(&self) -> T {
        self.as_view().sum()
    }

    /// Returns a 1D view into `axis` at `index`.
    #[inline]
    pub fn index_axis(&self, axis: Axis, index: usize) -> ArrayView1<'_, T> {
        match &self.0 {
            MultiRepr::Borrowed(a) => a.index_axis(axis, index),
            MultiRepr::BorrowedMut(a) => a.index_axis(axis, index),
            MultiRepr::Owned(a) => a.index_axis(axis, index),
        }
    }

    /// Returns a view of the column at `index`.
    #[inline]
    pub fn column(&self, index: usize) -> ArrayView1<'_, T> {
        match &self.0 {
            MultiRepr::Borrowed(v) => v.column(index),
            MultiRepr::BorrowedMut(v) => v.column(index),
            MultiRepr::Owned(a) => a.column(index),
        }
    }

    /// Returns a sliced 2D view.
    #[inline]
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
    #[inline]
    pub fn slice_mut<I>(&mut self, info: I) -> ArrayViewMut2<'_, T>
    where
        I: ndarray::SliceArg<ndarray::Ix2, OutDim = ndarray::Ix2>,
    {
        self.promote();

        self.promote();

        match &mut self.0 {
            MultiRepr::BorrowedMut(a) => a.slice_mut(info),
            MultiRepr::Owned(a) => a.slice_mut(info),
            MultiRepr::Borrowed(_) => {
                unreachable!("Self should have been converted to owned by now")
            }
        }
    }

    /// Returns a 2D view of the samples.
    #[inline]
    pub fn view(&self) -> ArrayView2<'_, T> {
        self.as_view()
    }

    /// Returns a mutable 2D view of the samples.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    #[inline]
    pub fn view_mut(&mut self) -> ArrayViewMut2<'_, T> {
        self.promote();
        match &mut self.0 {
            MultiRepr::BorrowedMut(a) => a.view_mut(),
            MultiRepr::Owned(a) => a.view_mut(),
            MultiRepr::Borrowed(_) => {
                unreachable!("Self should have been converted to owned by now")
            }
        }
    }

    /// Swaps two axes in-place.
    #[inline]
    pub fn swap_axes(&mut self, a: usize, b: usize) {
        self.to_mut().swap_axes(a, b);
    }

    /// Returns a mutable 1D view into `axis` at `index`.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    #[inline]
    pub fn index_axis_mut(&mut self, axis: Axis, index: usize) -> ArrayViewMut1<'_, T> {
        self.promote();
        self.promote();
        match &mut self.0 {
            MultiRepr::BorrowedMut(a) => a.index_axis_mut(axis, index),
            MultiRepr::Owned(a) => a.index_axis_mut(axis, index),
            MultiRepr::Borrowed(_) => {
                unreachable!("Self should have been converted to owned by now")
            }
        }
    }

    /// Returns the shape of the underlying `ndarray` buffer.
    #[inline]
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
    #[inline]
    pub fn mapv_inplace<F>(&mut self, f: F)
    where
        F: FnMut(T) -> T,
    {
        self.to_mut().mapv_inplace(f);
    }

    /// Returns a mutable iterator over 1D lanes along `axis`.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    #[inline]
    pub fn axis_iter_mut(&mut self, axis: Axis) -> AxisIterMut<'_, T, Ix1> {
        self.promote();
        match &mut self.0 {
            MultiRepr::BorrowedMut(a) => a.axis_iter_mut(axis),
            MultiRepr::Owned(a) => a.axis_iter_mut(axis),
            MultiRepr::Borrowed(_) => {
                unreachable!("Self should have been converted to owned by now")
            }
        }
    }

    /// Returns a view of the row at `index`.
    #[inline]
    pub fn row(&self, index: usize) -> ArrayView1<'_, T> {
        match &self.0 {
            MultiRepr::Borrowed(v) => v.row(index),
            MultiRepr::BorrowedMut(v) => v.row(index),
            MultiRepr::Owned(a) => a.row(index),
        }
    }

    /// Returns an iterator over all samples (row-major).
    #[inline]
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
    #[inline]
    pub fn iter_mut(&mut self) -> ndarray::iter::IterMut<'_, T, Ix2> {
        self.promote();
        match &mut self.0 {
            MultiRepr::BorrowedMut(a) => a.iter_mut(),
            MultiRepr::Owned(a) => a.iter_mut(),
            MultiRepr::Borrowed(_) => {
                unreachable!("Self should have been converted to owned by now")
            }
        }
    }

    /// Returns the total number of samples across all channels.
    #[inline]
    pub fn len(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.as_view().len()).expect("Array is guaranteed to be non-empty")
    }

    /// Maps each sample into a new `Array2`.
    #[inline]
    pub fn mapv<U, F>(&self, f: F) -> Array2<U>
    where
        F: Fn(T) -> U,
        U: Clone,
    {
        self.as_view().mapv(f)
    }

    /// Returns a mutable pointer to the underlying buffer.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.to_mut().as_mut_ptr()
    }

    /// Returns a shared slice if the underlying storage is contiguous.
    #[inline]
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
    #[inline]
    pub fn outer_iter(&mut self) -> ndarray::iter::AxisIterMut<'_, T, ndarray::Ix1> {
        self.promote();
        match &mut self.0 {
            MultiRepr::BorrowedMut(a) => a.outer_iter_mut(),
            MultiRepr::Owned(a) => a.outer_iter_mut(),
            MultiRepr::Borrowed(_) => {
                unreachable!("Self should have been converted to owned by now")
            }
        }
    }

    /// Returns a mutable slice if the underlying storage is contiguous.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    #[inline]
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        self.promote();
        match &mut self.0 {
            MultiRepr::BorrowedMut(a) => a.as_slice_mut(),
            MultiRepr::Owned(a) => a.as_slice_mut(),
            MultiRepr::Borrowed(_) => {
                unreachable!("Self should have been converted to owned by now")
            }
        }
    }

    /// Returns an iterator over 1D lanes along `axis`.
    #[inline]
    pub fn axis_iter(&self, axis: ndarray::Axis) -> ndarray::iter::AxisIter<'_, T, ndarray::Ix1> {
        match &self.0 {
            MultiRepr::Borrowed(v) => v.axis_iter(axis),
            MultiRepr::BorrowedMut(v) => v.axis_iter(axis),
            MultiRepr::Owned(a) => a.axis_iter(axis),
        }
    }

    /// Returns the raw dimension.
    #[inline]
    pub fn raw_dim(&self) -> ndarray::Dim<[usize; 2]> {
        self.as_view().raw_dim()
    }

    /// Returns a mutable view of the row at `index`.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    #[inline]
    pub fn row_mut(&mut self, index: usize) -> ndarray::ArrayViewMut1<'_, T> {
        self.promote();
        match &mut self.0 {
            MultiRepr::BorrowedMut(a) => a.row_mut(index),
            MultiRepr::Owned(a) => a.row_mut(index),
            MultiRepr::Borrowed(_) => {
                unreachable!("Self should have been converted to owned by now")
            }
        }
    }

    /// Fills all samples with `value`.
    ///
    /// If the data is currently immutably borrowed, this promotes to owned (allocates).
    #[inline]
    pub fn fill(&mut self, value: T) {
        self.to_mut().fill(value);
    }

    /// Converts into a raw `Vec<T>` and an offset.
    ///
    /// For borrowed data, this allocates and clones.
    #[inline]
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
    #[inline]
    pub fn fold<B, F>(&self, init: B, f: F) -> B
    where
        F: FnMut(B, &T) -> B,
    {
        self.as_view().iter().fold(init, f)
    }

    /// Converts this wrapper into an owned `Array2<T>`.
    ///
    /// For borrowed data, this allocates and clones.
    #[inline]
    pub fn take(self) -> Array2<T> {
        match self.0 {
            MultiRepr::Borrowed(v) => v.to_owned(),
            MultiRepr::BorrowedMut(v) => v.to_owned(),
            MultiRepr::Owned(a) => a,
        }
    }
}

/// Compares `MultiData` to an `Array2` element-by-element.
impl<T> PartialEq<Array2<T>> for MultiData<'_, T>
where
    T: StandardSample,
{
    #[inline]
    fn eq(&self, other: &Array2<T>) -> bool {
        self.as_view() == other.view()
    }
}

/// Compares an `Array2` to a `MultiData` element-by-element.
impl<'a, T> PartialEq<MultiData<'a, T>> for Array2<T>
where
    T: StandardSample,
{
    #[inline]
    fn eq(&self, other: &MultiData<'a, T>) -> bool {
        self.view() == other.as_view()
    }
}

/// Iterates over all samples in row-major order (`channel 0` first, then `channel 1`, …).
impl<'a, T> IntoIterator for &'a MultiData<'_, T>
where
    T: StandardSample,
{
    type Item = &'a T;
    type IntoIter = ndarray::iter::Iter<'a, T, ndarray::Ix2>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_view().into_iter()
    }
}

/// Channel-agnostic container for raw audio sample data.
///
/// ## Purpose
///
/// `AudioData` abstracts over the two storage shapes used by the library:
/// 1-D mono arrays and 2-D multi-channel arrays. All audio operations that
/// must work for both mono and multi-channel audio accept `AudioData<T>` or
/// the higher-level [`AudioSamples<T>`] that wraps it.
///
/// ## Intended Usage
///
/// Prefer constructing audio through [`AudioSamples`] constructors rather than
/// building `AudioData` directly. Use `AudioData` directly only when writing
/// low-level operations that need to inspect or replace the underlying array.
///
/// ## Invariants
///
/// - `Mono` always contains a non-empty `MonoData` (1-D array with ≥ 1 sample).
/// - `Multi` always contains a non-empty `MultiData` (2-D array with ≥ 1 channel
///   and ≥ 1 sample per channel).
///
/// ## Assumptions
///
/// Callers that use the unsafe constructors (`from_array1`, `from_array2`, …)
/// are responsible for upholding the non-empty invariant.
#[derive(Debug, PartialEq)]
#[allow(clippy::exhaustive_enums)]
pub enum AudioData<'a, T>
where
    T: StandardSample,
{
    /// Single-channel (mono) audio stored as a 1-D array.
    Mono(MonoData<'a, T>),
    /// Multi-channel audio stored as a 2-D array with shape `(channels, samples_per_channel)`.
    Multi(MultiData<'a, T>),
}

impl<T> AudioData<'static, T>
where
    T: StandardSample,
{
    /// Converts any `AudioData` into a `'static`-lifetime owned instance.
    ///
    /// Borrowed variants are promoted to owned by cloning the underlying buffer.
    /// If the input is already owned this is a zero-copy move.
    ///
    /// # Arguments
    ///
    /// – `data` – any `AudioData`, regardless of lifetime.
    ///
    /// # Returns
    ///
    /// An `AudioData<'static, T>` with owned storage.
    #[inline]
    #[must_use]
    pub fn from_owned(data: AudioData<'_, T>) -> Self {
        data.into_owned()
    }

    /// Returns the arithmetic mean across all samples.
    ///
    /// For mono data this is the mean of the single channel. For multi-channel data
    /// it is the mean computed across every sample in every channel.
    ///
    /// # Returns
    ///
    /// The mean value as type `T`.
    #[inline]
    #[must_use]
    pub fn mean(&self) -> T {
        match self {
            AudioData::Mono(m) => m.mean(),
            AudioData::Multi(m) => m.mean(),
        }
    }

    /// Consumes this `AudioData` and returns the underlying `Array1<T>` if it is mono.
    ///
    /// # Returns
    ///
    /// `Some(Array1<T>)` if the variant is `Mono`; `None` if it is `Multi`.
    #[inline]
    #[must_use]
    pub fn into_mono_data(self) -> Option<Array1<T>> {
        if let AudioData::Mono(m) = self {
            Some(m.take())
        } else {
            None
        }
    }

    /// Consumes this `AudioData` and returns the underlying `Array2<T>` if it is multi-channel.
    ///
    /// # Returns
    ///
    /// `Some(Array2<T>)` if the variant is `Multi`; `None` if it is `Mono`.
    #[inline]
    #[must_use]
    pub fn into_multi_data(self) -> Option<Array2<T>> {
        if let AudioData::Multi(m) = self {
            Some(m.take())
        } else {
            None
        }
    }
}

/// Clones `AudioData` into an owned instance.
///
/// Borrowed variants are promoted to owned by cloning the underlying ndarray buffer.
/// The resulting clone always uses owned storage regardless of the original variant.
impl<T> Clone for AudioData<'_, T>
where
    T: StandardSample,
{
    #[inline]
    fn clone(&self) -> Self {
        match self {
            AudioData::Mono(m) => AudioData::Mono(
                MonoData::from_owned(m.as_view().to_owned())
                    .expect("self has already guaranteed non-emptiness"),
            ),
            AudioData::Multi(m) => AudioData::Multi(
                MultiData::from_owned(m.as_view().to_owned())
                    .expect("self has already guaranteed non-emptiness"),
            ),
        }
    }
}

impl<T> AudioData<'_, T>
where
    T: StandardSample,
{
    /// Creates a new mono AudioData from an owned Array1.
    ///
    /// # Errors
    ///
    /// - If the data is empty
    #[inline]
    pub fn new_mono(data: Array1<T>) -> AudioSampleResult<AudioData<'static, T>> {
        if data.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        Ok(AudioData::Mono(MonoData(MonoRepr::Owned(data))))
    }

    /// Creates a new multi-channel AudioData from an owned Array2.
    ///
    /// # Errors
    ///
    /// - If the data is empty
    #[inline]
    pub fn new_multi(data: Array2<T>) -> AudioSampleResult<AudioData<'static, T>> {
        if data.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        Ok(AudioData::Multi(MultiData(MultiRepr::Owned(data))))
    }
    /// Create a borrowed version of self
    #[inline]
    #[must_use]
    pub fn borrow(&self) -> AudioData<'_, T> {
        match self {
            AudioData::Mono(mono_data) => {
                AudioData::Mono(MonoData(MonoRepr::Borrowed(mono_data.as_view())))
            }
            AudioData::Multi(multi_data) => {
                AudioData::Multi(MultiData(MultiRepr::Borrowed(multi_data.as_view())))
            }
        }
    }

    /// Returns all samples as a `Vec<T>`.
    ///
    /// For mono data the vec has `samples_per_channel` elements. For multi-channel
    /// data the samples are collected in row-major order: all samples for channel 0,
    /// then channel 1, etc. This allocates a new `Vec` on every call.
    ///
    /// # Returns
    ///
    /// A `Vec<T>` containing every sample.
    #[inline]
    #[must_use]
    pub fn as_vec(&self) -> Vec<T> {
        match &self {
            AudioData::Mono(mono_data) => mono_data.as_view().to_vec(),
            AudioData::Multi(multi_data) => multi_data.as_view().iter().copied().collect(),
        }
    }

    /// Returns the number of frames (time steps) in the audio data.
    ///
    /// A *frame* contains one sample per channel at a given point in time.
    /// For mono data this equals the total number of samples. For multi-channel
    /// data it equals `samples_per_channel`.
    ///
    /// # Returns
    ///
    /// The frame count as a non-zero `usize`.
    #[inline]
    #[must_use]
    pub fn total_frames(&self) -> NonZeroUsize {
        match self {
            AudioData::Mono(_) => self.len(),
            AudioData::Multi(multi_data) => multi_data.ncols(),
        }
    }

    /// Returns `true` if the underlying ndarray buffer uses standard (C) memory layout.
    ///
    /// Standard layout means elements are stored in row-major order with unit strides.
    /// Some operations (e.g. raw pointer access, SIMD paths) require standard layout.
    ///
    /// # Returns
    ///
    /// `true` if the storage is contiguous and row-major, `false` otherwise.
    #[inline]
    #[must_use]
    pub fn is_standard_layout(&self) -> bool {
        match &self {
            AudioData::Mono(m) => m.as_view().is_standard_layout(),
            AudioData::Multi(m) => m.as_view().is_standard_layout(),
        }
    }

    /// Creates mono AudioData from a slice (borrowed).
    #[inline]
    #[must_use]
    pub fn from_slice(slice: &NonEmptySlice<T>) -> AudioData<'_, T> {
        let view = ArrayView1::from(slice);
        AudioData::Mono(MonoData(MonoRepr::Borrowed(view)))
    }

    /// Creates mono AudioData from a mutable slice (borrowed).
    #[inline]
    #[must_use]
    pub fn from_slice_mut(slice: &mut NonEmptySlice<T>) -> AudioData<'_, T> {
        let view = ArrayViewMut1::from(slice);
        AudioData::Mono(MonoData(MonoRepr::BorrowedMut(view)))
    }

    /// Creates multi-channel AudioData from a slice with specified channel count (borrowed).
    ///
    /// # Errors
    /// - If the slice cannot be reshaped into the desired shape
    #[inline]
    pub fn from_slice_multi(
        slice: &NonEmptySlice<T>,
        channels: ChannelCount,
    ) -> AudioSampleResult<AudioData<'_, T>> {
        let total_samples = slice.len().get();
        let samples_per_channel = total_samples / channels.get() as usize;
        let view = ArrayView2::from_shape((channels.get() as usize, samples_per_channel), slice)?;
        Ok(AudioData::Multi(MultiData(MultiRepr::Borrowed(view))))
    }

    /// Creates multi-channel AudioData from a mutable slice with specified channel count (borrowed).
    ///
    /// # Errors
    ///
    /// - If the slice cannot be reshaped into the desired shape
    #[inline]
    pub fn from_slice_multi_mut(
        slice: &mut NonEmptySlice<T>,
        channels: ChannelCount,
    ) -> AudioSampleResult<AudioData<'_, T>> {
        let total_samples = slice.len().get();
        let samples_per_channel = total_samples / channels.get() as usize;
        let view =
            ArrayViewMut2::from_shape((channels.get() as usize, samples_per_channel), slice)?;
        Ok(AudioData::Multi(MultiData(MultiRepr::BorrowedMut(view))))
    }

    /// Creates mono AudioData from a Vec (owned).
    #[inline]
    #[must_use]
    pub fn from_vec(vec: NonEmptyVec<T>) -> AudioData<'static, T> {
        AudioData::Mono(MonoData(MonoRepr::Owned(Array1::from(vec.to_vec()))))
    }

    /// Creates multi-channel AudioData from a Vec with specified channel count (owned).
    ///
    /// # Errors
    ///
    /// - If the Vec cannot be reshaped into the desired shape
    #[inline]
    pub fn from_vec_multi(
        vec: NonEmptyVec<T>,
        channels: ChannelCount,
    ) -> AudioSampleResult<AudioData<'static, T>> {
        let total_samples = vec.len().get();
        let samples_per_channel = total_samples / channels.get() as usize;
        let arr =
            Array2::from_shape_vec((channels.get() as usize, samples_per_channel), vec.to_vec())?;
        Ok(AudioData::Multi(MultiData(MultiRepr::Owned(arr))))
    }
}

// Main implementation block for AudioData
impl<'a, T> AudioData<'a, T>
where
    T: StandardSample,
{
    /// Creates mono AudioData from an owned Array1.
    ///
    /// # Safety
    ///
    /// Caller must ensure that the array is non-empty.
    #[inline]
    #[must_use]
    pub const unsafe fn from_array1(arr: Array1<T>) -> Self {
        AudioData::Mono(MonoData(MonoRepr::Owned(arr)))
    }

    /// Creates multi-channel AudioData from an owned Array2.
    ///
    /// # Safety
    ///
    /// Caller must ensure that the array is non-empty.
    #[inline]
    #[must_use]
    pub const unsafe fn from_array2(arr: Array2<T>) -> Self {
        AudioData::Multi(MultiData(MultiRepr::Owned(arr)))
    }

    /// Wraps a pre-validated [`MonoData`] in `AudioData::Mono`.
    ///
    /// # Safety
    ///
    /// `data` must be non-empty. Callers that construct `MonoData` through the safe
    /// `from_owned` / `from_view` / `from_view_mut` constructors already uphold this
    /// invariant.
    #[inline]
    #[must_use]
    pub const unsafe fn from_mono_data(data: MonoData<'a, T>) -> Self {
        AudioData::Mono(data)
    }

    /// Wraps a pre-validated [`MultiData`] in `AudioData::Multi`.
    ///
    /// # Safety
    ///
    /// `data` must be non-empty. Callers that construct `MultiData` through the safe
    /// `from_owned` / `from_view` / `from_view_mut` constructors already uphold this
    /// invariant.
    #[inline]
    #[must_use]
    pub const unsafe fn from_multi_data(data: MultiData<'a, T>) -> Self {
        AudioData::Multi(data)
    }

    /// Creates a borrowed mono `AudioData` from a 1-D ndarray view.
    ///
    /// # Safety
    ///
    /// `view` must be non-empty.
    #[inline]
    #[must_use]
    pub const unsafe fn from_array1_view(view: ArrayView1<'a, T>) -> Self {
        AudioData::Mono(MonoData(MonoRepr::Borrowed(view)))
    }

    /// Creates a borrowed multi-channel `AudioData` from a 2-D ndarray view.
    ///
    /// # Safety
    ///
    /// `view` must be non-empty and have shape `(channels, samples_per_channel)`.
    #[inline]
    #[must_use]
    pub const unsafe fn from_array2_view(view: ArrayView2<'a, T>) -> Self {
        AudioData::Multi(MultiData(MultiRepr::Borrowed(view)))
    }

    /// Converts this AudioData to owned data.
    #[inline]
    #[must_use]
    pub fn into_owned<'b>(self) -> AudioData<'b, T> {
        match self {
            AudioData::Mono(m) => AudioData::Mono(m.into_owned()),
            AudioData::Multi(m) => AudioData::Multi(m.into_owned()),
        }
    }

    /// Creates a new AudioData instance from borrowed data.
    #[inline]
    #[must_use]
    pub fn from_borrowed(&self) -> AudioData<'_, T> {
        match self {
            AudioData::Mono(m) => AudioData::Mono(MonoData(MonoRepr::Borrowed(m.as_view()))),
            AudioData::Multi(m) => AudioData::Multi(MultiData(MultiRepr::Borrowed(m.as_view()))),
        }
    }

    /// Creates AudioData from a borrowed mono array view.
    ///
    /// # Errors
    ///
    /// - Returns [`AudioSampleError::EmptyData`] if `view` is empty.
    #[inline]
    pub fn from_borrowed_array1(view: ArrayView1<'_, T>) -> AudioSampleResult<AudioData<'_, T>> {
        if view.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        Ok(AudioData::Mono(MonoData(MonoRepr::Borrowed(view))))
    }

    /// Creates a borrowed mono `AudioData` from an immutable 1-D view, without checking for emptiness.
    ///
    /// Prefer [`from_borrowed_array1`](AudioData::from_borrowed_array1) unless the view is
    /// already guaranteed non-empty and the check is measurably expensive.
    ///
    /// # Safety
    ///
    /// `view` must be non-empty.
    #[inline]
    #[must_use]
    pub const unsafe fn from_borrowed_array1_unchecked(
        view: ArrayView1<'_, T>,
    ) -> AudioData<'_, T> {
        AudioData::Mono(MonoData(MonoRepr::Borrowed(view)))
    }

    /// Creates a mutably-borrowed mono `AudioData` from a mutable 1-D view.
    ///
    /// # Returns
    ///
    /// `Ok(AudioData::Mono)` wrapping the mutable view.
    ///
    /// # Errors
    ///
    /// Returns [`AudioSampleError::EmptyData`] if `view` is empty.
    #[inline]
    pub fn from_borrowed_array1_mut(
        view: ArrayViewMut1<'_, T>,
    ) -> AudioSampleResult<AudioData<'_, T>> {
        if view.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        Ok(AudioData::Mono(MonoData(MonoRepr::BorrowedMut(view))))
    }

    /// Creates a mutably-borrowed mono `AudioData` from a mutable 1-D view, without checking for emptiness.
    ///
    /// Prefer [`from_borrowed_array1_mut`](AudioData::from_borrowed_array1_mut) unless the
    /// view is already guaranteed non-empty.
    ///
    /// # Safety
    ///
    /// `view` must be non-empty.
    #[inline]
    #[must_use]
    pub const unsafe fn from_borrowed_array1_mut_unchecked(
        view: ArrayViewMut1<'_, T>,
    ) -> AudioData<'_, T> {
        AudioData::Mono(MonoData(MonoRepr::BorrowedMut(view)))
    }

    /// Creates a borrowed multi-channel `AudioData` from an immutable 2-D view.
    ///
    /// # Returns
    ///
    /// `Ok(AudioData::Multi)` wrapping the view.
    ///
    /// # Errors
    ///
    /// Returns [`AudioSampleError::EmptyData`] if `view` is empty.
    #[inline]
    pub fn from_borrowed_array2(view: ArrayView2<'_, T>) -> AudioSampleResult<AudioData<'_, T>> {
        if view.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        Ok(AudioData::Multi(MultiData(MultiRepr::Borrowed(view))))
    }

    /// Creates a borrowed multi-channel `AudioData` from an immutable 2-D view, without checking for emptiness.
    ///
    /// Prefer [`from_borrowed_array2`](AudioData::from_borrowed_array2) unless the view is
    /// already guaranteed non-empty.
    ///
    /// # Safety
    ///
    /// `view` must be non-empty and have shape `(channels, samples_per_channel)`.
    #[inline]
    #[must_use]
    pub const unsafe fn from_borrowed_array2_unchecked(
        view: ArrayView2<'_, T>,
    ) -> AudioData<'_, T> {
        AudioData::Multi(MultiData(MultiRepr::Borrowed(view)))
    }

    /// Creates a mutably-borrowed multi-channel `AudioData` from a mutable 2-D view.
    ///
    /// # Returns
    ///
    /// `Ok(AudioData::Multi)` wrapping the mutable view.
    ///
    /// # Errors
    ///
    /// Returns [`AudioSampleError::EmptyData`] if `view` is empty.
    #[inline]
    pub fn from_borrowed_array2_mut(
        view: ArrayViewMut2<'_, T>,
    ) -> AudioSampleResult<AudioData<'_, T>> {
        if view.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        Ok(AudioData::Multi(MultiData(MultiRepr::BorrowedMut(view))))
    }

    /// Creates a mutably-borrowed multi-channel `AudioData` from a mutable 2-D view, without checking for emptiness.
    ///
    /// Prefer [`from_borrowed_array2_mut`](AudioData::from_borrowed_array2_mut) unless the
    /// view is already guaranteed non-empty.
    ///
    /// # Safety
    ///
    /// `view` must be non-empty and have shape `(channels, samples_per_channel)`.
    #[inline]
    #[must_use]
    pub const unsafe fn from_borrowed_array2_mut_unchecked(
        view: ArrayViewMut2<'_, T>,
    ) -> AudioData<'_, T> {
        AudioData::Multi(MultiData(MultiRepr::BorrowedMut(view)))
    }

    /// Returns the total number of samples in the audio data.
    #[inline]
    #[must_use]
    pub fn len(&self) -> NonZeroUsize {
        match self {
            AudioData::Mono(m) => m.len(),
            AudioData::Multi(m) => m.len(),
        }
    }

    /// Returns the number of channels in the audio data.
    #[inline]
    #[must_use]
    pub fn num_channels(&self) -> ChannelCount {
        match self {
            AudioData::Mono(_) => channels!(1),
            // safety: self is non-empty therefore guaranteed to have at least one channel
            AudioData::Multi(m) => unsafe { ChannelCount::new_unchecked(m.shape()[0] as u32) },
        }
    }

    /// Returns true if the audio data is mono (single channel).
    #[inline]
    #[must_use]
    pub const fn is_mono(&self) -> bool {
        matches!(self, AudioData::Mono(_))
    }

    /// Returns true if the audio data has multiple channels.
    #[inline]
    #[must_use]
    pub const fn is_multi_channel(&self) -> bool {
        matches!(self, AudioData::Multi(_))
    }

    /// Returns the shape of the underlying array data.
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        match &self {
            AudioData::Mono(m) => m.shape(),
            AudioData::Multi(m) => m.shape(),
        }
    }

    /// Returns the number of samples per channel.
    #[inline]
    #[must_use]
    pub fn samples_per_channel(&self) -> NonZeroUsize {
        match self {
            AudioData::Mono(m) => {
                // safety: self is non-empty therefore len is NonZero
                unsafe { NonZeroUsize::new_unchecked(m.as_view().len()) }
            }
            // safety: self is non-empty therefore len is NonZero
            AudioData::Multi(m) => unsafe { NonZeroUsize::new_unchecked(m.as_view().shape()[1]) },
        }
    }

    /// Returns audio data as a slice if contiguous.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> Option<&[T]> {
        match &self {
            AudioData::Mono(m) => m.as_slice(),
            AudioData::Multi(m) => m.as_slice(),
        }
    }

    /// Returns a contiguous byte view when possible, falling back to I24 packing when required.
    ///
    /// # Errors
    ///
    /// - if the underlying data is not in a standard layout and cannot be safely viewed as bytes.
    #[inline]
    pub fn bytes(&self) -> AudioSampleResult<AudioBytes<'_>> {
        let slice = self.as_slice().ok_or_else(|| {
            AudioSampleError::Layout(LayoutError::NonContiguous {
                operation: "bytes view".to_string(),
                layout_type: "non-contiguous audio data".to_string(),
            })
        })?;

        if TypeId::of::<T>() == TypeId::of::<I24>() {
            // I24 is 4 bytes in memory but 3 bytes on disk; pack to owned bytes.
            // safety: self is non-empty
            let i24_slice: &[I24] =
                unsafe { core::slice::from_raw_parts(slice.as_ptr().cast::<I24>(), slice.len()) };
            let packed = I24::write_i24s_ne(i24_slice);
            // safety: self is non-empty
            let packed = unsafe { NonEmptyByteVec::new_unchecked(packed) };
            return Ok(AudioBytes::Owned(packed));
        }

        let byte_len = std::mem::size_of_val(slice);
        let byte_ptr = slice.as_ptr().cast::<u8>();
        // safety: self is non-empty
        let bytes = unsafe { core::slice::from_raw_parts(byte_ptr, byte_len) };
        // safety: self is non-empty
        let bytes = unsafe { NonEmptySlice::new_unchecked(bytes) };
        Ok(AudioBytes::Borrowed(bytes))
    }

    /// Returns the number of bytes per sample for the current sample type.
    #[inline]
    #[must_use]
    pub const fn bytes_per_sample(&self) -> NonZeroU32 {
        // Safety: T::BYTES is guaranteed to be non-zero for StandardSample types
        unsafe { NonZeroU32::new_unchecked(T::BYTES) }
    }

    /// Returns the number of bits per sample for the current sample type.
    #[inline]
    #[must_use]
    pub const fn bits_per_sample(&self) -> NonZeroU8 {
        // Safety: T::BITS is guaranteed to be non-zero for StandardSample types
        unsafe { NonZeroU8::new_unchecked(T::BITS) }
    }

    /// Returns an owned byte buffer.
    ///
    /// # Errors
    ///
    /// - If the underlying data is not in a standard layout and cannot be safely viewed as bytes.
    #[inline]
    pub fn into_bytes(&self) -> AudioSampleResult<NonEmptyByteVec> {
        self.bytes().map(AudioBytes::into_owned)
    }

    /// Maps a function over each sample, returning new owned audio data.
    ///
    /// Uses a raw contiguous slice when available so the compiler can
    /// auto-vectorise the element-wise conversion loop. Falls back to
    /// ndarray's `mapv` for non-contiguous views.
    #[inline]
    pub fn mapv<U, F>(&self, f: F) -> AudioData<'static, U>
    where
        F: Fn(T) -> U,
        U: StandardSample,
    {
        match self {
            AudioData::Mono(m) => {
                let out = m.as_slice().map_or_else(
                    || m.as_view().mapv(&f),
                    |slice| {
                        // Fast path: flat slice lets the compiler auto-vectorise.
                        let vec: Vec<U> = slice.iter().map(|&x| f(x)).collect();
                        Array1::from(vec)
                    },
                );
                AudioData::Mono(MonoData(MonoRepr::Owned(out)))
            }
            AudioData::Multi(m) => {
                let out = m.as_slice().map_or_else(|| m.as_view().mapv(&f), |slice| {
                    // Fast path: collect from flat C-order slice then reshape.
                    let (rows, cols) = m.as_view().dim();
                    let vec: Vec<U> = slice.iter().map(|&x| f(x)).collect();
                    Array2::from_shape_vec((rows, cols), vec).unwrap_or_else(|_| unreachable!("Output shape is guaranteed to match input shape since we're just mapping elementwise"))
                 });
                AudioData::Multi(MultiData(MultiRepr::Owned(out)))
            }
        }
    }

    /// Maps a function over each sample in place.
    ///
    /// # Arguments
    ///
    /// - `f` - a function that takes a sample of type `T` and returns a new sample of the same type.
    #[inline]
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
    ///
    /// # Arguments
    ///
    /// - `func` - a function that takes a sample of type `T` and returns a new sample of the same type.
    #[inline]
    pub fn apply<F>(&mut self, func: F)
    where
        F: Fn(T) -> T,
    {
        self.mapv_inplace(func);
    }

    /// Applies a function to each sample with its index in place.
    ///
    /// # Arguments
    ///
    /// - `func` - a function that takes the index of the sample and the sample value, and returns a new sample of the same type.
    #[inline]
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
    ///
    /// # Arguments
    ///
    /// - `window_size` - the size of the window to apply the function to.
    /// - `hop_size` - the number of samples to advance the window for each application
    /// - `func` - a function that takes the current window of samples and the previous window, and returns a new window of processed samples.
    ///
    /// # Errors
    ///
    /// - If the underlying data is not in a standard layout and cannot be safely viewed as slices.
    #[inline]
    pub fn apply_windowed<F>(
        &mut self,
        window_size: NonZeroUsize,
        hop_size: NonZeroUsize,
        func: F,
    ) -> AudioSampleResult<()>
    where
        F: Fn(&[T], &[T]) -> Vec<T>, // (current_window, prev_window) -> processed_window
    {
        let window_size = window_size.get();
        let hop_size = hop_size.get();

        match self {
            AudioData::Mono(m) => {
                let data = m.as_view();
                let x = data.as_slice().ok_or_else(|| {
                    AudioSampleError::Layout(LayoutError::NonContiguous {
                        operation: "mono processing".to_string(),
                        layout_type: "non-contiguous mono data".to_string(),
                    })
                })?;

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
                    let x = row.as_slice().ok_or_else(|| {
                        AudioSampleError::Layout(LayoutError::NonContiguous {
                            operation: "multi-channel row processing".to_string(),
                            layout_type: "non-contiguous row data".to_string(),
                        })
                    })?;

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
    ///
    /// # Arguments
    ///
    /// - `f` - a function that takes a sample of type `T` and returns a new sample of the same type.
    #[inline]
    pub fn apply_to_all_channels<F>(&mut self, f: F)
    where
        F: Fn(T) -> T,
    {
        self.mapv_inplace(f);
    }

    /// Applies a function to samples in specified channels only.
    ///
    /// # Arguments
    ///
    /// - `channels` - a slice of channel indices to apply the function to.
    /// - `f` - a function that takes a sample of type `T` and
    #[inline]
    pub fn apply_to_channels<F>(&mut self, channels: &[u32], f: F)
    where
        F: Fn(T) -> T,
    {
        match self {
            AudioData::Mono(m) => m.to_mut().iter_mut().for_each(|x| *x = f(*x)),
            AudioData::Multi(m) => {
                let mut a = m.to_mut();
                for (ch_idx, mut row) in a.axis_iter_mut(Axis(0)).enumerate() {
                    if channels.contains(&(ch_idx as u32)) {
                        for x in &mut row {
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
    ///
    /// # Returns
    ///
    /// A new `AudioData` instance with the same shape but samples converted to type `O`.
    #[inline]
    #[must_use]
    pub fn convert_to<O>(&self) -> AudioData<'static, O>
    where
        O: StandardSample + ConvertTo<T> + ConvertFrom<T>,
    {
        match self {
            AudioData::Mono(m) => {
                let out = m.as_view().mapv(super::traits::ConvertTo::convert_to);
                AudioData::Mono(MonoData(MonoRepr::Owned(out)))
            }
            AudioData::Multi(m) => {
                let out = m.as_view().mapv(super::traits::ConvertTo::convert_to);
                AudioData::Multi(MultiData(MultiRepr::Owned(out)))
            }
        }
    }
    /// Converts the audio data to an interleaved vector, consuming the data.
    ///
    /// # Returns
    ///
    /// A `NonEmptyVec<T>` containing the interleaved samples.
    ///
    /// # Panics
    ///
    /// Only panics if self.num_channels does not divide self.len() which at this point is guaranteed
    #[inline]
    #[must_use]
    pub fn to_interleaved_vec(self) -> NonEmptyVec<T> {
        match self {
            // safety: Self is guaranteed to be non-empty by construction, and to_vec() preserves length
            AudioData::Mono(m) => match m.0 {
                // safety: Self is guaranteed to be non-empty by construction, and to_vec() preserves length
                MonoRepr::Borrowed(v) => unsafe { NonEmptyVec::new_unchecked(v.to_vec()) },
                // safety: Self is guaranteed to be non-empty by construction, and to_vec() preserves length
                MonoRepr::BorrowedMut(v) => unsafe { NonEmptyVec::new_unchecked(v.to_vec()) },
                // safety: Self is guaranteed to be non-empty by construction, and to_vec() preserves length
                MonoRepr::Owned(a) => unsafe { NonEmptyVec::new_unchecked(a.to_vec()) },
            },
            AudioData::Multi(m) => {
                let (ch, _spc) = m.as_view().dim();
                // Get planar data as contiguous slice
                let planar: Vec<T> = m
                    .as_view()
                    .as_slice()
                    .map_or_else(|| m.as_view().iter().copied().collect(), <[T]>::to_vec);
                // safety: Self is guaranteed to be non-empty by construction
                let planar = unsafe { NonEmptyVec::new_unchecked(planar) };

                // safety: ch is non-zero by construction
                let ch = unsafe { NonZeroU32::new_unchecked(ch as u32) };
                // Use optimized interleave
                crate::simd_conversions::interleave_multi_vec(&planar, ch)
                    .expect("Interleave failed - this should not happen with valid input")
            }
        }
    }

    /// Returns the audio data as an interleaved vector without consuming the data.
    ///
    /// # Returns
    ///
    /// A `NonEmptyVec<T>` containing the interleaved samples.
    ///
    /// # Panics
    ///
    /// Will only ever panic if self.num_channel does not divide into self.len()
    /// In this part of the codebase, this is impossible.
    #[inline]
    #[must_use]
    pub fn as_interleaved_vec(&self) -> NonEmptyVec<T> {
        match self {
            // safety: self is guaranteed non-empty
            AudioData::Mono(m) => unsafe { NonEmptyVec::new_unchecked(m.as_view().to_vec()) },
            AudioData::Multi(m) => {
                let v = m.as_view();
                let (ch, _spc) = v.dim();
                // Get planar data as contiguous slice
                let planar: Vec<T> = v
                    .as_slice()
                    .map_or_else(|| v.iter().copied().collect(), <[T]>::to_vec);
                // safety: non-empty by design
                let planar = unsafe { NonEmptyVec::new_unchecked(planar) };
                // safety: channels non-zero by design
                let ch = unsafe { NonZeroU32::new_unchecked(ch as u32) };
                crate::simd_conversions::interleave_multi_vec(&planar, ch)
                    .expect("Interleave failed - this should not happen with valid input")
            }
        }
    }
}

/// Converts a borrowed 1-D view into a mono `AudioData`.
///
/// # Errors
///
/// Returns [`AudioSampleError::EmptyData`] if `arr` is empty.
impl<'a, T> TryFrom<ArrayView1<'a, T>> for AudioData<'a, T>
where
    T: StandardSample,
{
    type Error = AudioSampleError;

    #[inline]
    fn try_from(arr: ArrayView1<'a, T>) -> Result<Self, Self::Error> {
        if arr.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        Ok(AudioData::Mono(MonoData(MonoRepr::Borrowed(arr))))
    }
}

/// Converts a mutable borrowed 1-D view into a mono `AudioData`.
///
/// # Errors
///
/// Returns [`AudioSampleError::EmptyData`] if `arr` is empty.
impl<'a, T> TryFrom<ArrayViewMut1<'a, T>> for AudioData<'a, T>
where
    T: StandardSample,
{
    type Error = AudioSampleError;

    #[inline]
    fn try_from(arr: ArrayViewMut1<'a, T>) -> Result<Self, Self::Error> {
        if arr.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        Ok(AudioData::Mono(MonoData(MonoRepr::BorrowedMut(arr))))
    }
}

/// Converts a borrowed 2-D view into a multi-channel `AudioData`.
///
/// The view must have shape `(channels, samples_per_channel)`.
///
/// # Errors
///
/// Returns [`AudioSampleError::EmptyData`] if `arr` is empty.
impl<'a, T> TryFrom<ArrayView2<'a, T>> for AudioData<'a, T>
where
    T: StandardSample,
{
    type Error = AudioSampleError;

    #[inline]
    fn try_from(arr: ArrayView2<'a, T>) -> Result<Self, Self::Error> {
        if arr.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        Ok(AudioData::Multi(MultiData(MultiRepr::Borrowed(arr))))
    }
}

/// Converts a mutable borrowed 2-D view into a multi-channel `AudioData`.
///
/// The view must have shape `(channels, samples_per_channel)`.
///
/// # Errors
///
/// Returns [`AudioSampleError::EmptyData`] if `arr` is empty.
impl<'a, T> TryFrom<ArrayViewMut2<'a, T>> for AudioData<'a, T>
where
    T: StandardSample,
{
    type Error = AudioSampleError;

    #[inline]
    fn try_from(arr: ArrayViewMut2<'a, T>) -> Result<Self, Self::Error> {
        if arr.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        Ok(AudioData::Multi(MultiData(MultiRepr::BorrowedMut(arr))))
    }
}

/// Indexes `AudioData` by a flat sample index.
///
/// For mono data the index addresses samples directly. For multi-channel data the
/// index is linearised in row-major order: `sample[channel * samples_per_channel + i]`.
///
/// # Panics
///
/// Panics if `index` is out of bounds.
impl<T> Index<usize> for AudioData<'_, T>
where
    T: StandardSample,
{
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match self {
            AudioData::Mono(arr) => &arr[index],
            AudioData::Multi(arr) => {
                let (channels, samples_per_channel) = arr.dim();
                let total_samples = channels.get() * samples_per_channel.get();
                assert!(
                    index < total_samples,
                    "Index {index} out of bounds for total samples {total_samples}"
                );
                let channel = index / samples_per_channel;
                let sample_idx = index % samples_per_channel;
                &arr[(channel, sample_idx)]
            }
        }
    }
}

/// Converts an owned `Array1<T>` into a `MonoData`.
///
/// # Errors
///
/// Returns [`AudioSampleError::EmptyData`] if the array is empty.
impl<T> TryFrom<Array1<T>> for MonoData<'_, T>
where
    T: StandardSample,
{
    type Error = AudioSampleError;

    #[inline]
    fn try_from(a: Array1<T>) -> Result<Self, Self::Error> {
        if a.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        Ok(MonoData(MonoRepr::Owned(a)))
    }
}

/// Converts an owned `Array2<T>` into a `MultiData`.
///
/// The array must have shape `(channels, samples_per_channel)`.
///
/// # Errors
///
/// Returns [`AudioSampleError::EmptyData`] if the array is empty.
impl<T> TryFrom<Array2<T>> for MultiData<'_, T>
where
    T: StandardSample,
{
    type Error = AudioSampleError;

    #[inline]
    fn try_from(a: Array2<T>) -> Result<Self, Self::Error> {
        if a.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        Ok(MultiData(MultiRepr::Owned(a)))
    }
}

/// Converts an owned `Array1<T>` into a mono `AudioData`.
///
/// # Errors
///
/// Returns [`AudioSampleError::EmptyData`] if the array is empty.
impl<T> TryFrom<Array1<T>> for AudioData<'_, T>
where
    T: StandardSample,
{
    type Error = AudioSampleError;

    #[inline]
    fn try_from(a: Array1<T>) -> Result<Self, Self::Error> {
        if a.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        Ok(AudioData::Mono(a.try_into()?))
    }
}

/// Converts an owned `Array2<T>` into a multi-channel `AudioData`.
///
/// The array must have shape `(channels, samples_per_channel)`.
///
/// # Errors
///
/// Returns [`AudioSampleError::EmptyData`] if the array is empty.
impl<T> TryFrom<Array2<T>> for AudioData<'_, T>
where
    T: StandardSample,
{
    type Error = AudioSampleError;

    #[inline]
    fn try_from(a: Array2<T>) -> Result<Self, Self::Error> {
        if a.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        Ok(AudioData::Multi(a.try_into()?))
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
            impl<'a, T> std::ops::$trait<Self> for AudioData<'a, T>
                where
                    T: StandardSample,

            {
                type Output = Self;


                #[inline]
                fn $method(self, rhs: Self) -> Self::Output {
                    match (self, rhs) {
                        (AudioData::Mono(lhs), AudioData::Mono(rhs)) => {
                            if lhs.len() != rhs.len() {
                                panic!($mono_err);
                            }
                            // operate on views; convert to owned to ensure operation works
                            let arr: Array1<T> = &lhs.as_view() $op &rhs.as_view();
                            // safety: arr is already non-empty
                            unsafe {
                                AudioData::Mono(MonoData::from_array1(arr))
                            }
                        }
                        (AudioData::Multi(lhs), AudioData::Multi(rhs)) => {
                            if lhs.as_view().dim() != rhs.as_view().dim() {
                                panic!($multi_err);
                            }
                // safety: input array is not empty therefore output wont be

                            AudioData::Multi(unsafe { MultiData::from_array2(&lhs.as_view() $op &rhs.as_view()) })
                        }
                        _ => panic!($mismatch_err),
                    }
                }
            }

            // =========================
            // Binary ops: AudioData ∘ scalar -> new AudioData
            // =========================
            impl<'a, T> std::ops::$trait<T> for AudioData<'a, T>
                where
                    T: StandardSample,
            {
                type Output = Self;


                #[inline]
                fn $method(self, rhs: T) -> Self::Output {
                    match self {
                        // safety: input array is not empty therefore output wont be
                        AudioData::Mono(a) => AudioData::Mono(unsafe { MonoData::from_array1(a.as_view().mapv(|x| x $op rhs)) }),
                        // safety: input array is not empty therefore output wont be
                        AudioData::Multi(a) => AudioData::Multi(unsafe { MultiData::from_array2(a.as_view().mapv(|x| x $op rhs)) }),
                    }
                }
            }

            // =========================
            // Assignment ops: AudioData ∘= AudioData  (in-place, no Default)
            // =========================
            impl<'a, T> std::ops::$assign_trait<Self> for AudioData<'a, T>
            where
                T: StandardSample,
            {

                #[inline]
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
            impl<'a, T> std::ops::$assign_trait<T> for AudioData<'a, T>
            where
                T: StandardSample,
            {

                #[inline]
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

/// Negates every sample in the audio data.
///
/// Returns a new `AudioData` with all samples multiplied by −1. Requires the
/// sample type to implement `Neg`.
impl<T> Neg for AudioData<'_, T>
where
    T: StandardSample + Neg<Output = T> + ConvertTo<T> + ConvertFrom<T>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        match self {
            AudioData::Mono(arr) => {
                // safety: input array is not empty therefore output wont be
                AudioData::Mono(unsafe { MonoData::from_array1(arr.as_view().mapv(|x| -x)) })
            }
            AudioData::Multi(arr) => {
                // safety: input array is not empty therefore output wont be
                AudioData::Multi(unsafe { MultiData::from_array2(arr.as_view().mapv(|x| -x)) })
            }
        }
    }
}

/// Multiplies each sample in a mono `AudioSamples` by the corresponding value in a slice.
///
/// The multiplication is performed in the type `S`: each `T` sample is first converted to
/// `S` via [`ConvertTo`], multiplied, then converted back to `T` via [`ConvertFrom`].
///
/// # Returns
///
/// `Some(AudioSamples<T>)` on success; `None` if the audio is multi-channel or if the
/// slice length does not equal the number of samples.
impl<T, S> Mul<&[S]> for AudioSamples<'_, T>
where
    T: StandardSample + ConvertTo<S> + ConvertFrom<S>,
    S: StandardSample,
{
    type Output = Option<Self>;

    #[inline]
    fn mul(self, rhs: &[S]) -> Self::Output {
        if self.is_multi_channel() || self.len().get() != rhs.len() {
            return None;
        }
        let mut out = self.into_owned();
        out.apply_with_index(|idx, x| {
            let x: S = T::convert_to(x);
            let res = x * rhs[idx];
            let res: T = S::convert_to(res);
            res
        });

        Some(out)
    }
}

/// Represents homogeneous audio samples with associated metadata.
///
/// Primary container for audio data combining raw sample values with essential
/// metadata including sample rate, and type information.
/// Supports both mono and multi-channel audio with unified interface.
///
/// # Fields
/// - `data`: Audio sample data in mono or multi-channel format
/// - `sample_rate`: Sampling frequency in Hz
///
/// # Examples
///
/// ```rust
/// use audio_samples::{AudioSamples, sample_rate, channels};
/// use ndarray::array;
///
/// let mono = AudioSamples::new_mono(array![0.1f32, 0.2, 0.3], sample_rate!(44100)).unwrap();
/// assert_eq!(mono.num_channels(), channels!(1));
///
/// let stereo = AudioSamples::new_multi_channel(
///     array![[0.1f32, 0.2], [0.3f32, 0.4]],
///     sample_rate!(48000),
/// ).unwrap();
/// assert_eq!(stereo.num_channels(), channels!(2));
/// ```
///
/// # Invariants
///
/// The following properties are guaranteed by construction:
/// - `sample_rate` is always > 0 (stored as [`NonZeroU32`])
/// - `num_channels()` is always ≥ 1
/// - `samples_per_channel()` is always ≥ 1 (empty audio is not allowed)
///
/// These invariants eliminate the need for runtime null-checks in downstream code.
#[derive(Debug, PartialEq)]
#[allow(clippy::exhaustive_structs)] // `AudioSamples` will likely not change from this 
pub struct AudioSamples<'a, T>
where
    T: StandardSample,
{
    /// The audio sample data.
    pub data: AudioData<'a, T>,
    /// Sample rate in Hz (guaranteed non-zero).
    pub sample_rate: SampleRate,
    // pub layout: ChannelLayout,
}

/// Formats a human-readable summary of the audio samples.
///
/// The standard format (`{}`) prints a compact one-line header:
/// `AudioSamples<TYPE>: N ch × M samples @ R Hz (Layout)`.
///
/// The alternate format (`{:#}`) also includes the first and last 5 samples for
/// each channel, useful for quick inspection during debugging.
///
/// # Examples
///
/// ```rust
/// use audio_samples::{AudioSamples, sample_rate};
/// use ndarray::array;
///
/// let audio = AudioSamples::new_mono(array![0.1f32, 0.2, 0.3], sample_rate!(44100)).unwrap();
/// let s = format!("{}", audio);
/// assert!(s.contains("44100"));
/// ```
impl<T> Display for AudioSamples<'_, T>
where
    T: StandardSample,
{
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let type_name = std::any::type_name::<T>();
        let n_channels = self.num_channels();
        let n_samples = self.samples_per_channel();
        let rate = self.sample_rate;

        // Compact header (always shown)
        writeln!(
            f,
            "AudioSamples<{type_name}>: {n_channels} ch x {n_samples} samples @ {rate} Hz"
        )?;

        // Alternate (#) gives full details; otherwise, concise
        if f.alternate() {
            // Detailed alternate view
            match &self.data {
                AudioData::Mono(arr) => {
                    let len = arr.len();
                    let display_len = 5.min(len.get());
                    write!(f, "Mono Channel\n  First {display_len} samples: [")?;
                    for (i, val) in arr.iter().take(display_len).enumerate() {
                        write!(f, "{val:.4}")?;
                        if i < display_len - 1 {
                            write!(f, ", ")?;
                        }
                    }
                    write!(f, "]")?;
                    if len.get() > display_len {
                        write!(f, "\n  Last {display_len} samples: [")?;
                        for (i, val) in arr.iter().rev().take(display_len).rev().enumerate() {
                            write!(f, "{val:.4}")?;
                            if i < display_len - 1 {
                                write!(f, ", ")?;
                            }
                        }
                        write!(f, "]")?;
                    }
                }
                AudioData::Multi(arr) => {
                    let (channels, samples) = arr.dim();
                    for ch in 0..channels.get() {
                        let ch_data = arr.index_axis(ndarray::Axis(0), ch);
                        let len = samples.get();
                        let display_len = 5.min(len);

                        write!(f, "\nChannel {ch}:")?;
                        write!(f, "\n  First {display_len} samples: [")?;
                        for (i, val) in ch_data.iter().take(display_len).enumerate() {
                            write!(f, "{val:.4}")?;
                            if i < display_len - 1 {
                                write!(f, ", ")?;
                            }
                        }
                        write!(f, "]")?;

                        if len > display_len {
                            write!(f, "\n  Last {display_len} samples: [")?;
                            for (i, val) in ch_data.iter().rev().take(display_len).rev().enumerate()
                            {
                                write!(f, "{val:.4}")?;
                                if i < display_len - 1 {
                                    write!(f, ", ")?;
                                }
                            }
                            write!(f, "]")?;
                        }
                    }
                }
            }
        } else {
            // Compact summary
            match &self.data {
                AudioData::Mono(arr) => {
                    let len = arr.len();
                    let preview = 5.min(len.get());
                    write!(f, "[")?;
                    for (i, val) in arr.as_view().iter().take(preview).enumerate() {
                        write!(f, "{val:.4}")?;
                        if i < preview - 1 {
                            write!(f, ", ")?;
                        }
                    }
                    if len.get() > preview {
                        write!(f, ", ...")?;
                    }
                    write!(f, "]")?;
                }
                AudioData::Multi(arr) => {
                    let channels = arr.ncols().get();
                    for ch in 0..channels {
                        let ch_data = arr.index_axis(ndarray::Axis(0), ch);
                        let len = ch_data.len();
                        let preview = 3.min(len);
                        write!(f, "\nCh {ch}: [")?;
                        for (i, val) in ch_data.iter().take(preview).enumerate() {
                            write!(f, "{val:.4}")?;
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
        }
        Ok(())
    }
}

impl<T> AudioSamples<'static, T>
where
    T: StandardSample,
{
    /// Creates AudioSamples from owned data.
    #[inline]
    #[must_use]
    pub fn from_owned(data: AudioData<'_, T>, sample_rate: SampleRate) -> Self {
        let owned = data.into_owned();

        Self {
            data: owned,
            sample_rate,
        }
    }

    /// Consumes self and returns the underlying AudioData
    #[inline]
    #[must_use]
    pub fn into_data(self) -> AudioData<'static, T> {
        self.data.into_owned()
    }

    /// Consumes self and returns the underlying mono Array1 if applicable
    #[inline]
    #[must_use]
    pub fn into_array1(self) -> Option<Array1<T>> {
        match self.data {
            AudioData::Mono(m) => Some(m.take()),
            AudioData::Multi(_) => None,
        }
    }

    /// Consumes self and returns the underlying multi-channel Array2 if applicable
    #[inline]
    #[must_use]
    pub fn into_array2(self) -> Option<Array2<T>> {
        match self.data {
            AudioData::Multi(m) => Some(m.take()),
            AudioData::Mono(_) => None,
        }
    }
}

impl<'a, T> AudioSamples<'a, T>
where
    T: StandardSample,
{
    /// Creates a new AudioSamples with the given data and sample rate.
    ///
    /// This is a low-level constructor. Prefer `new_mono` or `new_multi_channel`
    /// for most use cases.
    #[inline]
    #[must_use]
    pub const fn new(data: AudioData<'a, T>, sample_rate: SampleRate) -> Self {
        Self { data, sample_rate }
    }

    /// Borrows the audio data as an AudioSamples with the same lifetime.
    #[inline]
    #[must_use]
    pub fn borrow(&self) -> AudioSamples<'_, T> {
        AudioSamples {
            data: self.data.borrow(),
            sample_rate: self.sample_rate,
        }
    }

    /// Returns all samples as a `Vec<T>`.
    ///
    /// For mono audio the vec contains `samples_per_channel` elements. For multi-channel
    /// audio the samples are in row-major order: all samples for channel 0, then channel 1, etc.
    /// This always allocates a new `Vec`.
    ///
    /// # Returns
    ///
    /// A `Vec<T>` containing every sample.
    #[inline]
    #[must_use]
    pub fn as_vec(&self) -> Vec<T> {
        self.data.as_vec()
    }

    /// Returns `true` if the underlying ndarray buffer uses standard (C/row-major) memory layout.
    ///
    /// Some low-level operations (raw pointer access, SIMD paths) require standard layout.
    ///
    /// # Returns
    ///
    /// `true` if the storage is contiguous and row-major, `false` otherwise.
    #[inline]
    #[must_use]
    pub fn is_standard_layout(&self) -> bool {
        self.data.is_standard_layout()
    }

    /// Creates AudioSamples from borrowed data.
    ///
    /// # Arguments
    ///
    /// - `data`: Audio sample data in mono or multi-channel format
    /// - `sample_rate`: Sample rate in Hz
    ///
    /// # Returns
    ///
    /// A new `AudioSamples` instance borrowing the provided data
    #[inline]
    #[must_use]
    pub const fn from_borrowed(data: AudioData<'a, T>, sample_rate: SampleRate) -> Self {
        Self { data, sample_rate }
    }

    /// Creates AudioSamples from borrowed data.
    ///
    /// # Arguments
    ///
    /// - `data`: Audio sample data in mono or multi-channel format
    /// - `sample_rate`: Sample rate in Hz
    ///
    /// # Returns
    ///
    /// A new `AudioSamples` instance borrowing the provided data.
    #[inline]
    #[must_use]
    pub const fn from_borrowed_with_layout(
        data: AudioData<'a, T>,
        sample_rate: SampleRate,
    ) -> Self {
        Self { data, sample_rate }
    }

    /// Convert audio samples to another sample type.
    ///
    /// # Returns
    ///
    /// A new `AudioSamples` instance with the same audio data converted to type `O`. The conversion is performed element-wise using the `ConvertTo` and `ConvertFrom` traits. This always allocates a new buffer for the converted samples.
    #[inline]
    pub fn convert_to<O>(&self) -> AudioSamples<'static, O>
    where
        T: ConvertTo<O>,
        O: StandardSample + ConvertFrom<T> + ConvertTo<O> + ConvertFrom<O>,
    {
        self.map_into(O::convert_from)
    }

    /// Convert the AudioSamples struct into a vector of samples in interleaved format.
    #[inline]
    #[must_use]
    pub fn to_interleaved_vec(&self) -> NonEmptyVec<T> {
        self.data.as_interleaved_vec()
    }

    /// Returns the number of frames (time steps) in the audio.
    ///
    /// A *frame* contains one sample per channel. For mono audio this equals
    /// `samples_per_channel()`. For multi-channel audio it equals `samples_per_channel()`.
    ///
    /// # Returns
    ///
    /// The total number of frames as a non-zero `usize`.
    #[inline]
    #[must_use]
    pub fn total_frames(&self) -> NonZeroUsize {
        self.data.total_frames()
    }

    /// Returns a slice of the audio samples if the data is contiguous. ``None`` otherwise
    ///
    /// # Returns
    ///
    /// `Some(&[T])` if the audio data is stored contiguously in memory, otherwise `None`.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> Option<&[T]> {
        match &self.data {
            AudioData::Mono(m) => m.as_slice(),
            AudioData::Multi(m) => m.as_slice(),
        }
    }

    /// Returns a mutable slice of the audio samples if the data is contiguous. ``None`` otherwise
    ///
    /// # Returns
    ///
    /// `Some(&mut [T])` if the audio data is stored contiguously in memory, otherwise `None`.
    #[inline]
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        match &mut self.data {
            AudioData::Mono(mono_data) => Some(mono_data.as_slice_mut()),
            AudioData::Multi(multi_data) => multi_data.as_slice_mut(),
        }
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
    /// # Errors
    ///
    /// - [`AudioSampleError::EmptyData`] if `data` is empty
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, -1.0, 0.5, -0.5];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    /// assert_eq!(audio.num_channels().get(), 1);
    /// assert_eq!(audio.sample_rate().get(), 44100);
    /// ```
    /// Creates a new mono AudioSamples with the given data and sample rate.
    ///
    /// # Panics
    /// - If `data` is empty
    #[inline]
    pub fn new_mono<'b>(
        data: Array1<T>,
        sample_rate: SampleRate,
    ) -> AudioSampleResult<AudioSamples<'b, T>> {
        if data.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }

        Ok(AudioSamples {
            data: AudioData::Mono(MonoData(MonoRepr::Owned(data))),
            sample_rate,
        })
    }

    /// Creates a new mono AudioSamples that owns its data without checking invariants.
    ///
    /// # Safety
    ///
    /// Make sure data is not empty
    #[inline]
    #[must_use]
    pub const unsafe fn new_mono_unchecked<'b>(
        data: Array1<T>,
        sample_rate: SampleRate,
    ) -> AudioSamples<'b, T> {
        AudioSamples {
            data: AudioData::Mono(MonoData(MonoRepr::Owned(data))),
            sample_rate,
        }
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
    /// let audio = AudioSamples::new_multi_channel(data, sample_rate!(44100)).unwrap();
    /// assert_eq!(audio.num_channels().get(), 2);
    /// assert_eq!(audio.samples_per_channel().get(), 2);
    /// ```
    /// Creates a new multi-channel AudioSamples with the given data and sample rate.
    ///
    /// # Errors
    /// - If `data` is empty
    #[inline]
    pub fn new_multi_channel<'b>(
        data: Array2<T>,
        sample_rate: SampleRate,
    ) -> AudioSampleResult<AudioSamples<'b, T>> {
        if data.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }

        Ok(AudioSamples {
            data: AudioData::Multi(MultiData(MultiRepr::Owned(data))),
            sample_rate,
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
    /// use audio_samples::{AudioSamples, sample_rate, nzu};
    ///
    /// let audio = AudioSamples::<f32>::zeros_mono(nzu!(1024), sample_rate!(44100));
    /// assert_eq!(audio.samples_per_channel().get(), 1024);
    /// assert_eq!(audio.num_channels().get(), 1);
    /// ```
    #[inline]
    #[must_use]
    pub fn zeros_mono(length: NonZeroUsize, sample_rate: SampleRate) -> Self {
        Self {
            data: AudioData::Mono(MonoData(MonoRepr::Owned(Array1::zeros(length.get())))),
            sample_rate,
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
    /// use audio_samples::{AudioSamples, sample_rate, channels, nzu};
    ///
    /// let audio = AudioSamples::<f32>::zeros_multi(channels!(2), nzu!(1024), sample_rate!(44100));
    /// assert_eq!(audio.num_channels().get(), 2);
    /// assert_eq!(audio.samples_per_channel().get(), 1024);
    /// ```
    #[inline]
    #[must_use]
    pub fn zeros_multi(
        channels: ChannelCount,
        length: NonZeroUsize,
        sample_rate: SampleRate,
    ) -> Self {
        Self {
            data: AudioData::Multi(MultiData(MultiRepr::Owned(Array2::zeros((
                channels.get() as usize,
                length.get(),
            ))))),
            sample_rate,
        }
    }

    /// Creates a new multi-channel AudioSamples filled with zeros (static version)
    #[inline]
    #[must_use]
    pub fn zeros_multi_channel(
        channels: ChannelCount,
        length: NonZeroUsize,
        sample_rate: SampleRate,
    ) -> AudioSamples<'static, T> {
        AudioSamples {
            data: AudioData::Multi(MultiData(MultiRepr::Owned(Array2::zeros((
                channels.get() as usize,
                length.get(),
            ))))),
            sample_rate,
        }
    }

    /// Creates a new mono AudioSamples filled with ones
    #[inline]
    #[must_use]
    pub fn ones_mono(length: NonZeroUsize, sample_rate: SampleRate) -> Self {
        Self {
            data: AudioData::Mono(MonoData(MonoRepr::Owned(Array1::ones(length.get())))),
            sample_rate,
        }
    }

    /// Creates a new multi-channel AudioSamples filled with ones
    #[inline]
    #[must_use]
    pub fn ones_multi(
        channels: ChannelCount,
        length: NonZeroUsize,
        sample_rate: SampleRate,
    ) -> Self {
        Self {
            data: AudioData::Multi(MultiData(MultiRepr::Owned(Array2::ones((
                channels.get() as usize,
                length.get(),
            ))))),
            sample_rate,
        }
    }

    /// Creates a new mono AudioSamples filled with the specified value
    #[inline]
    pub fn uniform_mono(length: NonZeroUsize, sample_rate: SampleRate, value: T) -> Self {
        Self {
            data: AudioData::Mono(MonoData(MonoRepr::Owned(Array1::from_elem(
                length.get(),
                value,
            )))),
            sample_rate,
        }
    }

    /// Creates a new multi-channel AudioSamples filled with the specified value
    #[inline]
    pub fn uniform_multi(
        channels: ChannelCount,
        length: NonZeroUsize,
        sample_rate: SampleRate,
        value: T,
    ) -> Self {
        Self {
            data: AudioData::Multi(MultiData(MultiRepr::Owned(Array2::from_elem(
                (channels.get() as usize, length.get()),
                value,
            )))),
            sample_rate,
        }
    }

    /// Returns basic info: (num_channels, samples_per_channel, duration_seconds, sample_rate, layout)
    ///
    /// # Returns
    ///
    /// A tuple containing the number of channels, the number of samples per channel,
    /// the duration in seconds, and the sample rate.
    #[inline]
    #[must_use]
    pub fn info(&self) -> (ChannelCount, NonZeroUsize, f64, NonZeroU32) {
        (
            self.num_channels(),
            self.samples_per_channel(),
            self.duration_seconds(),
            self.sample_rate,
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
    /// let audio = AudioSamples::new_mono(array![1.0f32, 2.0], sample_rate!(44100)).unwrap();
    /// assert_eq!(audio.sample_rate().get(), 44100);
    /// ```
    #[inline]
    #[must_use]
    pub const fn sample_rate(&self) -> SampleRate {
        self.sample_rate
    }

    /// Returns the sample rate as a plain `f64`.
    ///
    /// This is a convenience method equivalent to `self.sample_rate().get()`.
    #[inline]
    #[must_use]
    pub const fn sample_rate_hz(&self) -> f64 {
        self.sample_rate.get() as f64
    }

    /// Returns the number of channels.
    ///
    /// This is guaranteed to be ≥ 1 by construction.
    #[inline]
    #[must_use]
    pub fn num_channels(&self) -> ChannelCount {
        self.data.num_channels()
    }

    /// Returns the number of samples per channel.
    ///
    /// This is guaranteed to be ≥ 1 by construction (empty audio is not allowed).
    #[inline]
    #[must_use]
    pub fn samples_per_channel(&self) -> NonZeroUsize {
        self.data.samples_per_channel()
    }

    /// Returns the duration in seconds
    #[inline]
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        self.samples_per_channel().get() as f64 / self.sample_rate_hz()
    }

    /// Returns the total number of samples across all channels
    #[inline]
    #[must_use]
    pub fn total_samples(&self) -> NonZeroUsize {
        // Safety: A non-zero number * a non-zero number is always non-zero
        unsafe {
            NonZeroUsize::new_unchecked(
                self.num_channels().get() as usize * self.samples_per_channel().get(),
            )
        }
    }

    /// Returns the number of bytes per sample for type T
    #[inline]
    #[must_use]
    pub const fn bytes_per_sample(&self) -> NonZeroU32 {
        self.data.bytes_per_sample()
    }

    /// Returns the sample type as a string
    #[inline]
    #[must_use]
    pub const fn sample_type() -> SampleType {
        T::SAMPLE_TYPE
    }

    /// Returns true if this is mono audio
    #[inline]
    #[must_use]
    pub const fn is_mono(&self) -> bool {
        self.data.is_mono()
    }

    /// Returns true if this is multi-channel audio
    #[inline]
    #[must_use]
    pub const fn is_multi_channel(&self) -> bool {
        self.data.is_multi_channel()
    }

    /// Returns the total number of samples.
    #[inline]
    #[must_use]
    pub fn len(&self) -> NonZeroUsize {
        self.data.len()
    }

    /// Returns the shape of the audio data.
    #[inline]
    #[must_use]
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
    #[inline]
    pub fn apply<F>(&mut self, f: F)
    where
        F: Fn(T) -> T,
    {
        self.data.apply(f);
    }

    /// Apply a function to specific channels.
    #[inline]
    pub fn apply_to_channels<F>(&mut self, channels: &[u32], f: F)
    where
        F: Fn(T) -> T + Copy,
    {
        self.data.apply_to_channels(channels, f);
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
    #[inline]
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
    #[inline]
    pub fn map_into<O, F>(&self, f: F) -> AudioSamples<'static, O>
    where
        F: Fn(T) -> O,
        T: ConvertTo<O>,
        O: StandardSample,
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
    #[inline]
    pub fn apply_with_index<F>(&mut self, f: F)
    where
        F: Fn(usize, T) -> T,
    {
        self.data.apply_with_index(f);
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
    #[inline]
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
            Bound::Unbounded => samples_per_channel.get(),
        };

        if start >= samples_per_channel.get() || end > samples_per_channel.get() || start >= end {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "sample_range",
                format!(
                    "Sample range {start}..{end} out of bounds for {samples_per_channel} samples"
                ),
            )));
        }

        // no range condition
        if start == end {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "sample_range",
                "Start and end of sample range cannot be equal".to_string(),
            )));
        }
        // guarantees a non-empty slice

        match &self.data {
            AudioData::Mono(arr) => {
                // safety: slicing within bounds as we have guaranteed a non-empty slice
                let sliced =
                    unsafe { AudioData::from_array1_view(arr.slice(ndarray::s![start..end])) };
                Ok(AudioSamples::from_borrowed_with_layout(
                    sliced,
                    self.sample_rate(),
                ))
            }
            AudioData::Multi(arr) => {
                let sliced =
                    // safety: slicing within bounds as we have guaranteed a non-empty slice
                    unsafe { AudioData::from_array2_view(arr.slice(ndarray::s![.., start..end])) };
                Ok(AudioSamples::from_borrowed_with_layout(
                    sliced,
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
    #[inline]
    pub fn slice_channels<'iter, R>(
        &'iter self,
        channel_range: R,
    ) -> AudioSampleResult<AudioSamples<'static, T>>
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
            Bound::Unbounded => num_channels.get() as usize,
        };

        if start >= num_channels.get() as usize || end > num_channels.get() as usize || start >= end
        {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "channel_range",
                format!(
                    "Channel range {}..{} out of bounds for {} channels",
                    start,
                    end,
                    num_channels.get()
                ),
            )));
        }

        match &self.data {
            AudioData::Mono(arr) => {
                if start != 0 || end != 1 {
                    return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                        "channel_range",
                        format!(
                            "Channel range {start}..{end} invalid for mono audio (only 0..1 allowed)"
                        ),
                    )));
                }
                let audio_data = AudioData::try_from(arr.as_view())?;
                let audio = AudioSamples::from_owned(audio_data.into_owned(), self.sample_rate);
                Ok(audio)
            }
            AudioData::Multi(arr) => {
                let sliced = arr.slice(ndarray::s![start..end, ..]);
                if end - start == 1 {
                    // Single channel result - requires owned data due to temporary
                    let mono_data = sliced
                        .index_axis(ndarray::Axis(0), 0)
                        .to_owned()
                        .try_into()?;
                    let audio = AudioSamples::from_owned(mono_data, self.sample_rate);
                    Ok(audio)
                } else {
                    // Multi-channel result - return owned data for consistency
                    let audio =
                        AudioSamples::from_owned(sliced.to_owned().try_into()?, self.sample_rate);
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
    #[inline]
    pub fn slice_both<CR, SR>(
        &self,
        channel_range: CR,
        sample_range: SR,
    ) -> AudioSampleResult<AudioSamples<'static, T>>
    where
        CR: RangeBounds<isize> + Clone,
        SR: RangeBounds<isize> + Clone,
    {
        let num_channels = self.num_channels().get() as isize;
        let samples_per_channel = self.samples_per_channel().get().cast_signed();

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
                    "Channel range {ch_start}..{ch_end} out of bounds for {num_channels} channels"
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
                    "Sample range {s_start}..{s_end} out of bounds for {samples_per_channel} samples"
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
                            "Channel range {ch_start}..{ch_end} invalid for mono audio (only 0..1 allowed)"
                        ),
                    )));
                }
                let sliced = arr
                    .slice(ndarray::s![s_start..s_end])
                    .to_owned()
                    .try_into()?;
                Ok(AudioSamples::from_owned(sliced, self.sample_rate))
            }
            AudioData::Multi(arr) => {
                let sliced = arr.slice(ndarray::s![ch_start..ch_end, s_start..s_end]);
                if ch_end - ch_start == 1 {
                    let mono_data: AudioData<_> = sliced
                        .index_axis(ndarray::Axis(0), 0)
                        .to_owned()
                        .try_into()?;
                    Ok(AudioSamples::from_owned(mono_data, self.sample_rate))
                } else {
                    Ok(AudioSamples::from_owned(
                        sliced.to_owned().try_into()?,
                        self.sample_rate,
                    ))
                }
            }
        }
    }

    /// Returns a contiguous byte view when possible.
    ///
    /// # Errors
    ///
    /// - If the audio data is not stored contiguously in memory, an error is returned since a byte view cannot be created.
    #[inline]
    pub fn bytes(&self) -> AudioSampleResult<AudioBytes<'_>> {
        self.data.bytes()
    }

    /// Convert audio samples to raw bytes. If the underlying data is not contiguous,
    /// this will return an error
    ///
    /// # Errors
    ///
    /// - If the audio data is not stored contiguously in memory, an error is returned since a byte view cannot be created.
    #[inline]
    pub fn into_bytes(&self) -> AudioSampleResult<NonEmptyByteVec> {
        self.data.into_bytes()
    }

    /// Returns the total size in bytes of all audio data.
    ///
    /// This is equivalent to `self.data.len() * self.bytes_per_sample()`.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(array![1.0f32, 2.0, 3.0], sample_rate!(44100)).unwrap();
    /// assert_eq!(audio.total_byte_size().get(), 12); // 3 samples × 4 bytes per f32
    /// ```
    #[inline]
    #[must_use]
    pub fn total_byte_size(&self) -> NonZeroUsize {
        // safety: Totalbyte size will never be zero since both factors are non-zero
        unsafe {
            NonZeroUsize::new_unchecked(
                self.bytes_per_sample().get() as usize * self.data.len().get(),
            )
        }
    }

    // Convert audio samples to bytes in the specified endianness.
    //
    // Returns a vector of bytes representing the audio data in the requested
    // byte order. For cross-platform compatibility and when working with
    // audio file formats that specify endianness.
    //
    // # Feature Gate
    // This method requires the `serialization` feature.
    //
    // # Examples
    // ```
    // # #[cfg(feature = "serialization")]
    // # {
    // use audio_samples::{AudioSamples, operations::Endianness};
    // use ndarray::array;
    //
    // let audio = AudioSamples::new_mono(array![1000i16, 2000, 3000], 44100).unwrap();
    //
    // let native_bytes = audio.as_bytes_with_endianness(Endianness::Native);
    // let big_endian_bytes = audio.as_bytes_with_endianness(Endianness::Big);
    // let little_endian_bytes = audio.as_bytes_with_endianness(Endianness::Little);
    //
    // assert_eq!(native_bytes.len(), 6); // 3 samples × 2 bytes per i16
    // # }
    // ```
    // #[cfg(feature = "serialization")]
    //
    // pub fn as_bytes_with_endianness(
    //     &self,
    //     endianness: crate::operations::Endianness,
    // ) -> AudioSampleResult<NonEmptyByteVec> {
    //     use crate::operations::Endianness;

    //     match endianness {
    //         Endianness::Native => self.into_bytes(),
    //         Endianness::Big => {
    //             let mut result = Vec::with_capacity(self.total_byte_size().get());

    //             match &self.data {
    //                 AudioData::Mono(m) => {
    //                     for sample in m.iter() {
    //                         result.extend_from_slice(sample.to_be_bytes().as_ref());
    //                     }
    //                 }
    //                 AudioData::Multi(m) => {
    //                     for sample in m.iter() {
    //                         result.extend_from_slice(sample.to_be_bytes().as_ref());
    //                     }
    //                 }
    //             }

    //             NonEmptyByteVec::new(result).map_err(|_| AudioSampleError::EmptyData)
    //         }
    //         Endianness::Little => {
    //             let mut result = Vec::with_capacity(self.total_byte_size().get());
    //             match &self.data {
    //                 AudioData::Mono(m) => {
    //                     for sample in m.iter() {
    //                         result.extend_from_slice(sample.to_le_bytes().as_ref());
    //                     }
    //                 }
    //                 AudioData::Multi(m) => {
    //                     for sample in m.iter() {
    //                         result.extend_from_slice(sample.to_le_bytes().as_ref());
    //                     }
    //                 }
    //             }
    //             NonEmptyByteVec::new(result).map_err(|_| AudioSampleError::EmptyData)
    //         }
    //     }
    // }

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
    ///
    /// # Errors
    ///
    /// - If the underlying array data cannot be accessed contiguously
    #[inline]
    pub fn apply_windowed<F>(
        &mut self,
        window_size: NonZeroUsize,
        hop_size: NonZeroUsize,
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
    #[inline]
    #[must_use]
    pub fn into_owned<'b>(self) -> AudioSamples<'b, T> {
        AudioSamples {
            data: self.data.into_owned(),
            sample_rate: self.sample_rate,
        }
    }

    /// Replaces the audio data while preserving sample rate.
    ///
    /// This method allows you to swap out the underlying audio data with new data
    /// of the same channel configuration, maintaining the existing sample rate.
    ///
    /// # Arguments
    /// * `new_data` - The new audio data to replace the current data
    ///
    /// # Returns
    /// Returns `Ok(())` on success, or an error if the new data has incompatible dimensions.
    ///
    /// # Errors
    /// - Returns `ParameterError` if the new data has a different number of channels
    ///
    /// # Examples
    /// ```rust
    /// use audio_samples::{AudioSamples, AudioData, sample_rate};
    /// use ndarray::array;
    ///
    /// // Create initial audio
    /// let initial_data = array![1.0f32, 2.0, 3.0];
    /// let mut audio = AudioSamples::new_mono(initial_data, sample_rate!(44100)).unwrap();
    ///
    /// // Replace with new data
    /// let new_data = AudioData::new_mono(array![4.0f32, 5.0, 6.0, 7.0]).unwrap();
    /// audio.replace_data(new_data)?;
    ///
    /// assert_eq!(audio.samples_per_channel().get(), 4);
    /// assert_eq!(audio.sample_rate().get(), 44100); // Preserved
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn replace_data(&mut self, new_data: AudioData<'a, T>) -> AudioSampleResult<()> {
        // Validate channel count compatibility
        let current_channels = self.data.num_channels();
        let new_channels = new_data.num_channels();

        if current_channels != new_channels {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "new_data",
                format!(
                    "Channel count mismatch: existing audio has {current_channels} channels, new data has {new_channels} channels"
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
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::array;
    ///
    /// // Create initial mono audio
    /// let mut audio = AudioSamples::new_mono(array![1.0f32, 2.0], sample_rate!(44100)).unwrap();
    ///
    /// // Replace with new mono data
    /// audio.replace_with_mono(array![3.0f32, 4.0, 5.0])?;
    ///
    /// assert_eq!(audio.samples_per_channel().get(), 3);
    /// assert_eq!(audio.sample_rate().get(), 44100); // Preserved
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn replace_with_mono(&mut self, data: Array1<T>) -> AudioSampleResult<()> {
        if data.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        if !self.is_mono() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_type",
                format!(
                    "Cannot replace multi-channel audio ({} channels) with mono data",
                    self.num_channels()
                ),
            )));
        }

        let new_data = AudioData::try_from(data)?.into_owned();
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
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::array;
    ///
    /// // Create initial stereo audio
    /// let mut audio = AudioSamples::new_multi_channel(
    ///     array![[1.0f32, 2.0], [3.0, 4.0]], sample_rate!(44100)
    /// ).unwrap();
    ///
    /// // Replace with new stereo data (different length is OK)
    /// audio.replace_with_multi(array![[5.0f32, 6.0, 7.0], [8.0, 9.0, 10.0]])?;
    ///
    /// assert_eq!(audio.samples_per_channel().get(), 3);
    /// assert_eq!(audio.num_channels().get(), 2);
    /// assert_eq!(audio.sample_rate().get(), 44100); // Preserved
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn replace_with_multi(&mut self, data: Array2<T>) -> AudioSampleResult<()> {
        if data.is_empty() {
            return Err(AudioSampleError::EmptyData);
        }
        let new_channels = data.nrows();
        // safety: Channel count cannot be zero
        let new_channels = unsafe { ChannelCount::new_unchecked(new_channels as u32) };

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
                    new_channels.get()
                ),
            )));
        }

        let new_data = AudioData::try_from(data)?.into_owned();
        self.replace_data(new_data)
    }

    /// Replaces the audio data with new data from a Vec.
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
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use non_empty_slice::non_empty_vec;
    /// use ndarray::array;
    ///
    /// // Replace mono audio
    /// let mut mono_audio = AudioSamples::new_mono(array![1.0f32, 2.0], sample_rate!(44100)).unwrap();
    /// mono_audio.replace_with_vec(&non_empty_vec![3.0f32, 4.0, 5.0, 6.0])?;
    /// assert_eq!(mono_audio.samples_per_channel().get(), 4);
    ///
    /// // Replace stereo audio (interleaved samples)
    /// let mut stereo_audio = AudioSamples::new_multi_channel(
    ///     array![[1.0f32, 2.0], [3.0, 4.0]], sample_rate!(44100)
    /// ).unwrap();
    /// stereo_audio.replace_with_vec(&non_empty_vec![5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0])?;
    /// assert_eq!(stereo_audio.samples_per_channel().get(), 3); // 6 samples ÷ 2 channels
    /// assert_eq!(stereo_audio.num_channels().get(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn replace_with_vec(&mut self, samples: &NonEmptyVec<T>) -> AudioSampleResult<()> {
        let num_channels = self.num_channels().get() as usize;

        if !samples.len().get().is_multiple_of(num_channels) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "samples",
                format!(
                    "Sample count {} is not divisible by channel count {}",
                    samples.len(),
                    num_channels
                ),
            )));
        }

        if num_channels == 1 {
            // Mono case: create Array1 directly
            let array = Array1::from(samples.to_vec());
            self.replace_with_mono(array)
        } else {
            // Multi-channel case: reshape into (channels, samples_per_channel)
            let samples_per_channel = samples.len().get() / num_channels;
            let array =
                Array2::from_shape_vec((num_channels, samples_per_channel), samples.to_vec())
                    .map_err(|e| {
                    AudioSampleError::Parameter(ParameterError::invalid_value(
                        "samples",
                        format!(
                            "Failed to reshape samples into {num_channels}x{samples_per_channel} array: {e}"
                        ),
                    ))
                })?;
            self.replace_with_multi(array)
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
    /// use audio_samples::{AudioSamples, sample_rate, channels};
    /// use non_empty_slice::non_empty_vec;
    /// use ndarray::array;
    ///
    /// // Create placeholder stereo audio
    /// let mut audio = AudioSamples::new_multi_channel(
    ///     array![[0.0f32, 0.0], [0.0, 0.0]], sample_rate!(44100)
    /// ).unwrap();
    ///
    /// // Replace with data that must be stereo (2 channels)
    /// let samples = non_empty_vec![1.0f32, 2.0, 3.0, 4.0]; // 4 samples = 2 samples per channel
    /// audio.replace_with_vec_channels(&samples, channels!(2))?;
    ///
    /// assert_eq!(audio.samples_per_channel().get(), 2);
    /// assert_eq!(audio.num_channels().get(), 2);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[inline]
    pub fn replace_with_vec_channels(
        &mut self,
        samples: &NonEmptyVec<T>,
        expected_channels: ChannelCount,
    ) -> AudioSampleResult<()> {
        // Validate sample count against expected channels
        if !samples
            .len()
            .get()
            .is_multiple_of(expected_channels.get() as usize)
        {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "samples",
                format!(
                    "Sample count {} is not divisible by expected channel count {}",
                    samples.len(),
                    expected_channels.get()
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

        if expected_channels == channels!(1) {
            // Mono case: create Array1 directly
            let array: Array1<T> = Array1::from_vec(samples.to_vec());
            self.replace_with_mono(array)
        } else {
            // Multi-channel case: reshape into (channels, samples_per_channel)
            let samples_per_channel = self.samples_per_channel();
            let array = Array2::from_shape_vec(
                (expected_channels.get() as usize, samples_per_channel.get()),
                samples.to_vec(),
            )?;
            self.replace_with_multi(array)
        }
    }

    /// Returns a reference to the underlying mono array if this is mono audio.
    ///
    /// # Returns
    /// Some reference to the mono array if this is mono audio, None otherwise.
    #[inline]
    #[must_use]
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
    #[inline]
    #[must_use]
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
    #[inline]
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
    #[inline]
    pub const fn as_multi_channel_mut(&mut self) -> Option<&mut MultiData<'a, T>> {
        match &mut self.data {
            AudioData::Mono(_) => None,
            AudioData::Multi(arr) => Some(arr),
        }
    }

    /// Creates a new mono AudioSamples from a slice.
    ///
    /// # Arguments
    ///
    /// - `slice` - A non-empty slice containing the audio samples in row-major order (all samples for channel 0, then all samples for channel 1, etc.).
    /// - `sample_rate` - The sample rate of the audio.
    ///
    /// # Returns
    ///
    /// A new instance of `Self`
    #[inline]
    pub fn new_mono_from_slice(slice: &'a NonEmptySlice<T>, sample_rate: SampleRate) -> Self {
        let arr = ArrayView1::from(slice);
        // safety: We know slice to be non-empty
        let mono_data = unsafe { MonoData::from_view_unchecked(arr) };
        let audio_data = AudioData::Mono(mono_data);
        AudioSamples {
            data: audio_data,
            sample_rate,
        }
    }

    /// Creates a new mono AudioSamples from a mutable slice.
    ///
    /// # Arguments
    ///
    /// - `slice` - A non-empty slice containing the audio samples in row-major order (all samples for channel 0, then all samples for channel 1, etc.).
    /// - `sample_rate` - The sample rate of the audio.
    ///
    /// # Returns
    ///
    /// A new instance of `Self`
    #[inline]
    pub fn new_mono_from_mut_slice(
        slice: &'a mut NonEmptySlice<T>,
        sample_rate: SampleRate,
    ) -> Self {
        let arr = ArrayViewMut1::from(slice);
        // safety: We know the slice is non-empty
        let mono_data = unsafe { MonoData::from_view_mut_unchecked(arr) };
        let audio_data = AudioData::Mono(mono_data);
        AudioSamples {
            data: audio_data,
            sample_rate,
        }
    }

    /// Creates a new multi-channel AudioSamples from a slice.
    ///
    /// # Arguments
    ///
    /// - `slice` - A non-empty slice containing the audio samples in row-major order (all samples for channel 0, then all samples for channel 1, etc.).
    /// - `channels` - The number of channels in the audio data.
    /// - `sample_rate` - The sample rate of the audio data.
    ///
    /// # Returns
    ///
    /// Returns an `AudioSampleResult` containing the new `AudioSamples` instance if successful, or an error if the input data is invalid.
    ///
    /// # Errors
    ///
    /// - If the length of the slice is not divisible by the number of channels.
    /// - If the slice cannot be reshaped into the desired shape.
    #[inline]
    pub fn new_multi_channel_from_slice(
        slice: &'a NonEmptySlice<T>,
        channels: ChannelCount,
        sample_rate: SampleRate,
    ) -> AudioSampleResult<Self> {
        let total_samples = slice.len().get();
        let samples_per_channel = total_samples / channels.get() as usize;
        let arr = ArrayView2::from_shape((channels.get() as usize, samples_per_channel), slice)?;
        let multi_data = MultiData::from_view(arr)?;
        let audio_data = AudioData::Multi(multi_data);
        Ok(AudioSamples {
            data: audio_data,
            sample_rate,
        })
    }

    /// Creates a new multi-channel AudioSamples from a mutable slice.
    ///
    /// # Arguments
    ///
    /// - `slice` - A non-empty slice containing the audio samples in row-major order (all samples for channel 0, then all samples for channel 1, etc.).
    /// - `channels` - The number of channels in the audio data.
    /// - `sample_rate` - The sample rate of the audio data.
    ///
    /// # Returns
    ///
    /// Returns an `AudioSampleResult` containing the new `AudioSamples` instance if successful, or an error if the input data is invalid.
    ///
    /// # Errors
    /// - If the length of the slice is not divisible by the number of channels.
    /// - If the slice cannot be reshaped into the desired shape.
    #[inline]
    pub fn new_multi_channel_from_mut_slice(
        slice: &'a mut NonEmptySlice<T>,
        channels: ChannelCount,
        sample_rate: SampleRate,
    ) -> AudioSampleResult<Self> {
        let total_samples = slice.len().get();
        let samples_per_channel = total_samples / channels.get() as usize;
        let arr = ArrayViewMut2::from_shape((channels.get() as usize, samples_per_channel), slice)?;
        let multi_data = MultiData::from_view_mut(arr)?;
        let audio_data = AudioData::Multi(multi_data);
        Ok(AudioSamples {
            data: audio_data,
            sample_rate,
        })
    }

    /// Creates multi-channel AudioSamples from a flat non-empty vec (owned).
    ///
    /// # Arguments
    ///
    /// - `vec` - A non-empty vector containing the audio samples in row-major order (all samples for channel 0, then all samples for channel 1, etc.).
    /// - `channels` - The number of channels in the audio data.
    /// - `sample_rate` - The sample rate of the audio data.
    ///
    /// # Returns
    ///
    /// Returns an `AudioSampleResult` containing the new `AudioSamples` instance if successful, or an error if the input data is invalid.
    ///
    /// # Errors
    ///
    /// If the number of samples in the input Vec is not divisible by the number of channels, or if the resulting array cannot be created due to shape issues.
    #[inline]
    pub fn new_multi_channel_from_vec<O>(
        vec: NonEmptyVec<O>,
        channels: ChannelCount,
        sample_rate: SampleRate,
    ) -> AudioSampleResult<Self>
    where
        T: ConvertFrom<O>,
        O: StandardSample,
    {
        let total_samples = vec.len().get();
        if !total_samples.is_multiple_of(channels.get() as usize) {
            return Err(AudioSampleError::invalid_number_of_samples(
                total_samples,
                channels.get(),
            ));
        }
        let samples_per_channel = total_samples / channels.get() as usize;
        let vec: NonEmptyVec<T> = vec
            .into_non_empty_iter()
            .map(T::convert_from)
            .collect_non_empty();
        let arr =
            Array2::from_shape_vec((channels.get() as usize, samples_per_channel), vec.to_vec())?;

        let multi_data = MultiData::from_owned(arr)?;
        let audio_data = AudioData::Multi(multi_data);
        Ok(AudioSamples {
            data: audio_data,
            sample_rate,
        })
    }

    /// Creates mono AudioSamples from a Vec (owned).
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use non_empty_slice::non_empty_vec;
    ///
    /// let samples = non_empty_vec![0.1f32, 0.2, 0.3, 0.4];
    /// let audio: AudioSamples<'static, f32> = AudioSamples::from_mono_vec(samples, sample_rate!(44100));
    /// assert_eq!(audio.num_channels().get(), 1);
    /// assert_eq!(audio.samples_per_channel().get(), 4);
    /// ```
    #[inline]
    pub fn from_mono_vec<O>(
        vec: NonEmptyVec<O>,
        sample_rate: SampleRate,
    ) -> AudioSamples<'static, T>
    where
        O: StandardSample,
        T: ConvertFrom<O>,
    {
        let vec: NonEmptyVec<T> = vec
            .into_non_empty_iter()
            .map(ConvertFrom::convert_from)
            .collect_non_empty();
        AudioSamples {
            data: AudioData::from_vec(vec),
            sample_rate,
        }
    }

    /// Creates multi-channel AudioSamples from a Vec with specified channel count (owned).
    ///
    /// The samples in the Vec are assumed to be in row-major order (all samples for
    /// channel 0, then all samples for channel 1, etc.).
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate, channels};
    /// use non_empty_slice::non_empty_vec;
    ///
    /// // 2 channels, 3 samples each: [ch0_s0, ch0_s1, ch0_s2, ch1_s0, ch1_s1, ch1_s2]
    /// let samples = non_empty_vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
    /// let audio = AudioSamples::from_vec_with_channels(samples, channels!(2), sample_rate!(44100)).unwrap();
    /// assert_eq!(audio.num_channels().get(), 2);
    /// assert_eq!(audio.samples_per_channel().get(), 3);
    /// ```
    ///
    /// # Errors
    ///
    /// If the number of samples in the input Vec is not divisible by the number of channels, or if the resulting array cannot be created due to shape issues.
    #[inline]
    pub fn from_vec_with_channels(
        vec: NonEmptyVec<T>,
        channels: ChannelCount,
        sample_rate: SampleRate,
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        let total_samples = vec.len().get();

        if !total_samples.is_multiple_of(channels.get() as usize) {
            return Err(AudioSampleError::invalid_number_of_samples(
                total_samples,
                channels.get(),
            ));
        }

        Ok(AudioSamples {
            data: AudioData::from_vec_multi(vec, channels)?,
            sample_rate,
        })
    }

    /// Creates AudioSamples from interleaved sample data (owned).
    ///
    /// Interleaved format: [L0, R0, L1, R1, L2, R2, ...] for stereo.
    /// This method de-interleaves the data into the internal non-interleaved format.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate, channels};
    /// use non_empty_slice::non_empty_vec;
    ///
    /// // Interleaved stereo: [L0, R0, L1, R1]
    /// let interleaved = non_empty_vec![0.1f32, 0.4, 0.2, 0.5];
    /// let audio: AudioSamples<'static, f32> = AudioSamples::from_interleaved_vec(interleaved, channels!(2), sample_rate!(44100)).unwrap();
    /// assert_eq!(audio.num_channels().get(), 2);
    /// assert_eq!(audio.samples_per_channel().get(), 2);
    /// // After de-interleaving: channel 0 = [0.1, 0.2], channel 1 = [0.4, 0.5]
    /// ```
    ///
    /// # Errors
    ///
    /// If the number of samples in the input Vec is not divisible by the number of channels, or if the de-interleaving process fails due to shape issues.
    ///
    #[inline]
    pub fn from_interleaved_vec<O>(
        samples: NonEmptyVec<O>,
        channels: ChannelCount,
        sample_rate: SampleRate,
    ) -> AudioSampleResult<AudioSamples<'static, T>>
    where
        T: ConvertFrom<O>,
        O: StandardSample,
    {
        if channels.get() == 1 {
            return Ok(AudioSamples::from_mono_vec(samples, sample_rate));
        }

        let deinterleaved = crate::simd_conversions::deinterleave_multi_vec(&samples, channels)?;
        AudioSamples::new_multi_channel_from_vec::<O>(deinterleaved, channels, sample_rate)
    }

    /// Creates AudioSamples from interleaved slice data (borrowed).
    ///
    /// # Arguments
    ///
    /// - `samples`: A non-empty slice of interleaved audio samples.
    /// - `channels`: The number of channels in the interleaved data.
    /// - `sample_rate`: The sample rate of the audio data.
    ///
    /// # Errors
    ///
    /// If the `samples` slice cannot be turned into an 2D Array due to a shape error.
    #[inline]
    pub fn from_interleaved_slice(
        samples: &'a NonEmptySlice<T>,
        channels: ChannelCount,
        sample_rate: SampleRate,
    ) -> AudioSampleResult<Self> {
        if channels == channels!(1) {
            return Ok(AudioSamples::new_mono_from_slice(samples, sample_rate));
        }

        AudioSamples::new_multi_channel_from_slice(samples, channels, sample_rate)
    }

    /// Creates multi-channel AudioSamples from separate channel arrays.
    ///
    /// Each element in the vector represents one channel of audio data.
    /// All channels must have the same length.
    ///
    /// # Errors
    /// Returns an error if:
    /// - any channel is empty (has no samples)
    /// - channels have different lengths
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use non_empty_slice::non_empty_vec;
    ///
    /// let left = non_empty_vec![0.1f32, 0.2, 0.3];
    /// let right = non_empty_vec![0.4, 0.5, 0.6];
    /// let audio: AudioSamples<'static, f32> = AudioSamples::from_channels(non_empty_vec![left, right], sample_rate!(44100)).unwrap();
    /// assert_eq!(audio.num_channels().get(), 2);
    /// assert_eq!(audio.samples_per_channel().get(), 3);
    /// ```
    #[inline]
    pub fn from_channels<O>(
        channels: NonEmptyVec<NonEmptyVec<O>>,
        sample_rate: SampleRate,
    ) -> AudioSampleResult<AudioSamples<'static, T>>
    where
        T: ConvertFrom<O>,
        O: StandardSample,
    {
        let num_channels = channels.len().get();
        let samples_per_channel = channels[0].len().get();

        // Validate all channels have the same length
        for (idx, ch) in channels.iter().enumerate() {
            if ch.len().get() != samples_per_channel {
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
            return Ok(AudioSamples::from_mono_vec(
                channels[0].clone(),
                sample_rate,
            ));
        }

        // Flatten channels into row-major order
        let flat: NonEmptyVec<T> = channels
            .into_non_empty_iter()
            .flatten()
            .map(ConvertTo::convert_to)
            .collect_non_empty();
        let arr = Array2::from_shape_vec((num_channels, samples_per_channel), flat.to_vec())
            .map_err(|e| {
                AudioSampleError::Parameter(ParameterError::invalid_value(
                    "channels",
                    format!("Failed to create multi-channel array: {e}"),
                ))
            })?;

        Ok(AudioSamples {
            data: AudioData::Multi(MultiData::from_owned(arr)?),
            sample_rate,
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
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use non_empty_slice::non_empty_vec;
    /// use ndarray::array;
    ///
    /// let left = AudioSamples::new_mono(array![0.1f32, 0.2, 0.3], sample_rate!(44100)).unwrap();
    /// let right = AudioSamples::new_mono(array![0.4, 0.5, 0.6], sample_rate!(44100)).unwrap();
    /// let stereo: AudioSamples<'static, f32> = AudioSamples::from_mono_channels(non_empty_vec![left, right]).unwrap();
    /// assert_eq!(stereo.num_channels().get(), 2);
    /// ```
    #[inline]
    pub fn from_mono_channels<O>(
        channels: NonEmptyVec<AudioSamples<'_, O>>,
    ) -> AudioSampleResult<AudioSamples<'static, T>>
    where
        O: StandardSample + ConvertFrom<T> + ConvertTo<O> + ConvertFrom<O>,
        T: ConvertFrom<O>,
    {
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
        let channel_data: NonEmptyVec<NonEmptyVec<O>> = channels
            .into_non_empty_iter()
            .map(|ch| match ch.data {
                // Safety: m has already been verified as non-empty
                AudioData::Mono(m) => unsafe { NonEmptyVec::new_unchecked(m.to_vec()) },
                AudioData::Multi(_) => unreachable!("Already validated as mono"),
            })
            .collect_non_empty();

        AudioSamples::from_channels(channel_data, sample_rate)
    }
    /// Creates AudioSamples from an iterator of samples (mono, owned).
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use non_empty_slice::non_empty_vec;
    ///
    /// // NonEmptyVec implements IntoNonEmptyIterator
    /// let samples = non_empty_vec![0.0f32, 0.1, 0.2, 0.3, 0.4];
    /// let audio = AudioSamples::from_iter(samples, sample_rate!(44100));
    /// assert_eq!(audio.num_channels().get(), 1);
    /// assert_eq!(audio.samples_per_channel().get(), 5);
    /// ```
    #[inline]
    pub fn from_iter<I>(iter: I, sample_rate: SampleRate) -> AudioSamples<'static, T>
    where
        I: IntoNonEmptyIterator<Item = T>,
        NonEmptyVec<T>: non_empty_iter::FromNonEmptyIterator<T>,
    {
        // Convert the incoming type into a non-empty iterator
        let ne_iter = iter.into_non_empty_iter();

        // Collect into your non-empty vector
        let ne_vec: NonEmptyVec<T> = ne_iter.collect_non_empty();

        AudioSamples::from_mono_vec::<T>(ne_vec, sample_rate)
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
    /// use audio_samples::{AudioSamples, sample_rate, channels};
    /// use non_empty_slice::non_empty_vec;
    ///
    /// // 2 channels, 3 samples each — NonEmptyVec implements IntoNonEmptyIterator
    /// let samples = non_empty_vec![0.0f32, 0.1, 0.2, 0.3, 0.4, 0.5];
    /// let audio = AudioSamples::from_iter_with_channels(samples, channels!(2), sample_rate!(44100));
    /// assert_eq!(audio.num_channels().get(), 2);
    /// assert_eq!(audio.samples_per_channel().get(), 3);
    /// ```
    #[inline]
    pub fn from_iter_with_channels<I>(
        iter: I,
        channels: ChannelCount,
        sample_rate: SampleRate,
    ) -> AudioSamples<'static, T>
    where
        I: IntoNonEmptyIterator<Item = T>,
        NonEmptyVec<T>: non_empty_iter::FromNonEmptyIterator<T>,
    {
        // Convert the incoming type into a non-empty iterator
        let ne_iter = iter.into_non_empty_iter();

        // Collect into your non-empty vector
        let ne_vec: NonEmptyVec<T> = ne_iter.collect_non_empty();

        AudioSamples::from_vec_with_channels(ne_vec, channels, sample_rate)
            .expect("Collected samples should be valid for the given channel count")
    }

    /// Calculates the nyquist frequency of the signal
    #[inline]
    #[must_use]
    pub fn nyquist(&self) -> f64 {
        f64::from(self.sample_rate.get()) / 2.0
    }

    /// Raises each sample to the specified exponent, optionally applying modulo.
    ///
    /// # Arguments
    ///
    /// * `exponent` - The exponent to raise each sample to.
    /// * `modulo` - Optional modulo value to apply after exponentiation.
    ///
    /// # Returns
    ///
    /// A new AudioSamples instance with each sample raised to the specified exponent.
    #[inline]
    #[must_use]
    pub fn powf(&self, exponent: f64, modulo: Option<T>) -> Self {
        let new_data = match &self.data {
            AudioData::Mono(mono) => {
                let powered = mono.mapv(|sample: T| {
                    let base: f64 = sample.cast_into();
                    let result = base.powf(exponent);
                    let powered_sample: T = T::cast_from(result);
                    modulo.map_or(powered_sample, |mod_val| powered_sample % mod_val)
                });
                // safety: self has already been validated as non-empty, therefore powered is non-empty
                unsafe { AudioData::Mono(MonoData::from_owned_unchecked(powered)) }
            }
            AudioData::Multi(multi) => {
                let powered = multi.mapv(|sample| {
                    let base: f64 = sample.cast_into();
                    let result = base.powf(exponent);
                    let powered_sample: T = T::cast_from(result);
                    modulo.map_or(powered_sample, |mod_val| powered_sample % mod_val)
                });
                // safety: self has already been validated as non-empty, therefore powered is non-empty
                unsafe { AudioData::Multi(MultiData::from_owned_unchecked(powered)) }
            }
        };

        AudioSamples {
            data: new_data,
            sample_rate: self.sample_rate,
        }
    }

    /// Calls `f` with a mutable *view* of the window.
    ///
    /// For mono: `ArrayViewMut1<T>` of length `len`.
    /// For multi: `ArrayViewMut2<T>` with shape:
    /// - (channels, len) if SamplesAreAxis1
    /// - (len, channels) if SamplesAreAxis0
    #[inline]
    pub fn with_window_mut<R>(
        &mut self,
        start: usize,
        len: NonZeroUsize,
        f: impl FnOnce(WindowMut<'_, T>) -> R,
    ) -> Option<R> {
        let len = len.get();
        let total = self.samples_per_channel().get();
        if start >= total {
            return None;
        }

        let end = (start + len).min(total);

        let out = match &mut self.data {
            AudioData::Mono(mono_data) => {
                let view = mono_data.slice_mut(s![start..end]);
                f(WindowMut::Mono(view))
            }
            AudioData::Multi(multi_data) => {
                // (channels, samples): slice along Axis(1)
                let view = multi_data.slice_mut(s![.., start..end]);
                f(WindowMut::Multi(view))
            }
        };

        Some(out)
    }
}

/// Mutable window view  over ``ndarray::ArrayViewMut1`` and ``ndarray::ArrayViewMut2`` for `with_window_mut` method.
#[derive(Debug)]
pub enum WindowMut<'a, T> {
    Mono(ArrayViewMut1<'a, T>),
    Multi(ArrayViewMut2<'a, T>),
}

/// Indicates the time axis orientation in multi-channel audio data.
#[derive(Clone, Copy, Debug)]
#[allow(unused)]
pub enum TimeAxis {
    /// Shape: (channels, samples)
    SamplesAreAxis1,
    /// Shape: (samples, channels)
    SamplesAreAxis0,
}

impl<T> Clone for AudioSamples<'_, T>
where
    T: StandardSample,
{
    #[inline]
    fn clone(&self) -> Self {
        AudioSamples {
            data: self.data.clone(),
            sample_rate: self.sample_rate,
        }
    }
}

impl<T> TryInto<Array1<T>> for AudioSamples<'_, T>
where
    T: StandardSample,
{
    type Error = AudioSampleError;
    /// Convert mono AudioSamples into an owned Array1.
    ///
    /// # Errors
    /// Returns an error if the AudioSamples is not mono.
    #[inline]
    fn try_into(self) -> Result<Array1<T>, Self::Error> {
        match self.data {
            AudioData::Mono(mono) => Ok(mono.as_view().to_owned()),
            AudioData::Multi(_) => Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_type",
                "Cannot convert multi-channel AudioSamples into Array1",
            ))),
        }
    }
}

impl<T> TryInto<Array2<T>> for AudioSamples<'_, T>
where
    T: StandardSample,
{
    type Error = AudioSampleError;
    /// Convert multi-channel AudioSamples into an owned Array2.
    ///
    /// # Errors
    ///
    /// Returns an error if the AudioSamples is not multi-channel.
    #[inline]
    fn try_into(self) -> Result<Array2<T>, Self::Error> {
        match self.data {
            AudioData::Mono(_) => Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_type",
                "Cannot convert mono AudioSamples into Array2",
            ))),
            AudioData::Multi(multi) => Ok(multi.as_view().to_owned()),
        }
    }
}

impl<T> TryFrom<(ChannelCount, SampleRate, NonEmptyVec<T>)> for AudioSamples<'static, T>
where
    T: StandardSample,
{
    type Error = AudioSampleError;

    /// Create AudioSamples from a sample rate, the number of channels and a vector of samples .
    #[inline]
    fn try_from(
        (channels, sample_rate, samples): (ChannelCount, SampleRate, NonEmptyVec<T>),
    ) -> Result<Self, Self::Error> {
        if channels == channels!(1) {
            Ok(AudioSamples::from_mono_vec(samples, sample_rate))
        } else {
            AudioSamples::from_vec_with_channels(samples, channels, sample_rate)
        }
    }
}

/// Create owned mono AudioSamples from (`NonEmptyVec<T>`, sample_rate) tuple.
///
/// # Examples
/// ```
/// use audio_samples::{AudioSamples, sample_rate};
/// use non_empty_slice::non_empty_vec;
///
/// let audio = AudioSamples::from((non_empty_vec![0.1f32, 0.2, 0.3], sample_rate!(44100)));
/// assert_eq!(audio.num_channels().get(), 1);
/// assert_eq!(audio.samples_per_channel().get(), 3);
/// ```
impl<T> From<(NonEmptyVec<T>, SampleRate)> for AudioSamples<'static, T>
where
    T: StandardSample,
{
    #[inline]
    fn from((samples, sample_rate): (NonEmptyVec<T>, SampleRate)) -> Self {
        AudioSamples::from_mono_vec(samples, sample_rate)
    }
}
impl<T> From<(SampleRate, NonEmptyVec<T>)> for AudioSamples<'static, T>
where
    T: StandardSample,
{
    #[inline]
    fn from((sample_rate, samples): (SampleRate, NonEmptyVec<T>)) -> Self {
        AudioSamples::from_mono_vec(samples, sample_rate)
    }
}

/// Create mono AudioSamples from (`&NonEmptySlice<T>`, sample_rate) tuple (borrows the data).
///
/// # Examples
/// ```
/// use audio_samples::{AudioSamples, sample_rate};
/// use non_empty_slice::NonEmptySlice;
///
/// let data = [0.1f32, 0.2, 0.3];
/// let ne_slice = NonEmptySlice::new(&data).unwrap();
/// let audio = AudioSamples::from((ne_slice, sample_rate!(44100)));
/// assert_eq!(audio.num_channels().get(), 1);
/// assert_eq!(audio.samples_per_channel().get(), 3);
/// ```
impl<'a, T> From<(&'a NonEmptySlice<T>, SampleRate)> for AudioSamples<'a, T>
where
    T: StandardSample,
{
    #[inline]
    fn from((samples, sample_rate): (&'a NonEmptySlice<T>, SampleRate)) -> Self {
        AudioSamples::new_mono_from_slice(samples, sample_rate)
    }
}

// ========================
// Index and IndexMut implementations using ndarray delegation
// ========================

impl<T> Index<usize> for AudioSamples<'_, T>
where
    T: StandardSample,
{
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
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::array;
    ///
    /// let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
    /// let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
    ///
    /// assert_eq!(audio[0], 1.0);
    /// assert_eq!(audio[2], 3.0);
    /// ```
    #[inline]
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

impl<T> IndexMut<usize> for AudioSamples<'_, T>
where
    T: StandardSample,
{
    /// Mutable index into mono audio samples by sample index.
    ///
    /// For mono audio, this returns a mutable reference to the sample at the given index.
    /// For multi-channel audio, this will panic - use `[(channel, sample)]` instead.
    ///
    /// # Panics
    ///
    /// - If index is out of bounds
    /// - If used on multi-channel audio (use 2D indexing instead)
    #[inline]
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

impl<T> Index<(usize, usize)> for AudioSamples<'_, T>
where
    T: StandardSample,
{
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
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::array;
    ///
    /// // Mono audio
    /// let mono_data = array![1.0f32, 2.0, 3.0];
    /// let mono_audio = AudioSamples::new_mono(mono_data, sample_rate!(44100)).unwrap();
    /// assert_eq!(mono_audio[(0, 1)], 2.0);
    ///
    /// // Multi-channel audio
    /// let stereo_data = array![[1.0f32, 2.0], [3.0, 4.0]];
    /// let stereo_audio = AudioSamples::new_multi_channel(stereo_data, sample_rate!(44100)).unwrap();
    /// assert_eq!(stereo_audio[(0, 1)], 2.0); // Channel 0, sample 1
    /// assert_eq!(stereo_audio[(1, 0)], 3.0); // Channel 1, sample 0
    /// ```
    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (channel, sample) = index;
        match &self.data {
            AudioData::Mono(arr) => {
                assert!(
                    channel == 0,
                    "Channel index {channel} out of bounds for mono audio (only channel 0 exists)"
                );
                &arr[sample]
            }
            AudioData::Multi(arr) => &arr[(channel, sample)],
        }
    }
}

impl<T> IndexMut<(usize, usize)> for AudioSamples<'_, T>
where
    T: StandardSample,
{
    /// Mutable index into audio samples by (channel, sample) coordinates.
    ///
    /// This works for both mono and multi-channel audio:
    /// - For mono: only `(0, sample_index)` is valid  
    /// - For multi-channel: `(channel_index, sample_index)`
    ///
    /// # Panics
    /// - If channel or sample index is out of bounds
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (channel, sample) = index;
        match &mut self.data {
            AudioData::Mono(arr) => {
                assert!(
                    channel == 0,
                    "Channel index {channel} out of bounds for mono audio (only channel 0 exists)"
                );
                &mut arr[sample]
            }
            AudioData::Multi(arr) => &mut arr[(channel, sample)],
        }
    }
}

impl<T> Index<[usize; 2]> for AudioSamples<'_, T>
where
    T: StandardSample,
{
    type Output = T;

    /// Index into audio samples by [channel, sample] coordinates.
    ///
    /// This works for both mono and multi-channel audio:
    /// - For mono: only `[0, sample_index]` is valid
    /// - For multi-channel: `[channel_index, sample_index]`
    ///
    /// # Panics
    /// - If channel or sample index is out of bounds
    #[inline]
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let channel = index[0];
        let sample = index[1];
        match &self.data {
            AudioData::Mono(arr) => {
                assert!(
                    channel == 0,
                    "Channel index {channel} out of bounds for mono audio (only channel 0 exists)"
                );
                &arr[sample]
            }
            AudioData::Multi(arr) => &arr[(channel, sample)],
        }
    }
}

impl<T> IntoIterator for AudioSamples<'_, T>
where
    T: StandardSample,
{
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    /// Consumes the AudioSamples and returns an iterator over the samples in interleaved order.
    ///
    /// For mono audio, this is simply the samples in order.
    /// For multi-channel audio, this interleaves samples from each channel.
    ///
    /// # Examples
    /// ```rust
    /// use audio_samples::{AudioSamples, sample_rate};
    /// use ndarray::array;
    ///
    /// // Mono audio
    /// let mono_data = array![1.0f32, 2.0, 3.0];
    /// let mono_audio = AudioSamples::new_mono(mono_data, sample_rate!(44100)).unwrap();
    /// let mono_samples: Vec<f32> = mono_audio.into_iter().collect();
    /// assert_eq!(mono_samples, vec![1.0, 2.0, 3.0]);
    ///
    /// // Multi-channel audio
    /// let stereo_data = array![[1.0f32, 2.0], [3.0, 4.0]];
    /// let stereo_audio = AudioSamples::new_multi_channel(stereo_data, sample_rate!(44100)).unwrap();
    /// let stereo_samples: Vec<f32> = stereo_audio.into_iter().collect();
    /// assert_eq!(stereo_samples, vec![1.0, 3.0, 2.0, 4.0]); // Interleaved
    /// ```
    #[inline]
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
            impl<T> std::ops::$trait<Self> for AudioSamples<'_, T>
                where
                    T: StandardSample,

             {
                type Output = Self;

                #[inline]
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
                    }
                }
            }

            // Binary op with scalar T
            impl<T> std::ops::$trait<T> for AudioSamples<'_, T>
                where
                    T: StandardSample,

            {
                type Output = Self;

                #[inline]
                fn $method(self, rhs: T) -> Self::Output {
                    Self {
                        data: self.data $op rhs,
                        sample_rate: self.sample_rate,
                    }
                }
            }

            // Assign op with another AudioSamples<T>
            impl<T> std::ops::$assign_trait<Self> for AudioSamples<'_, T>
                where
                    T: StandardSample,

            {

                #[inline]
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
            impl<T> std::ops::$assign_trait<T> for AudioSamples<'_, T>
                where
                    T: StandardSample,

            {
                #[inline]
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
impl<T> Neg for AudioSamples<'_, T>
where
    T: StandardSample + Neg<Output = T> + ConvertTo<T> + ConvertFrom<T>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            data: -self.data,
            sample_rate: self.sample_rate,
        }
    }
}

#[cfg(feature = "resampling")]
impl<'a, T> Adapter<'a, T> for AudioSamples<'a, T>
where
    T: StandardSample,
{
    #[inline]
    unsafe fn read_sample_unchecked(&self, channel: usize, frame: usize) -> T {
        self[(channel, frame)]
    }

    #[inline]
    fn channels(&self) -> usize {
        self.num_channels().get() as usize
    }

    #[inline]
    fn frames(&self) -> usize {
        self.total_frames().get() as usize
    }
}

/// A newtype wrapper around [`AudioSamples`] that guarantees exactly two channels (stereo).
///
/// ## Purpose
///
/// `StereoAudioSamples` encodes the stereo invariant in the type system so that code
/// expecting a stereo signal can accept it without re-checking the channel count at
/// every call site.
///
/// ## Intended Usage
///
/// Construct via [`StereoAudioSamples::new`] or [`TryFrom<AudioSamples>`]. The inner
/// [`AudioSamples`] is accessible through `Deref` / `DerefMut` / `AsRef` / `AsMut`.
///
/// ## Invariants
///
/// - `num_channels()` always returns `2`.
/// - The inner data is always the `Multi` variant of [`AudioData`].
///
/// ## Assumptions
///
/// All constructors reject non-stereo input at runtime; there is no way to construct
/// a `StereoAudioSamples` with fewer or more than two channels through the public API.
#[non_exhaustive]
pub struct StereoAudioSamples<'a, T>(pub AudioSamples<'a, T>)
where
    T: StandardSample;

impl<'a, T> StereoAudioSamples<'a, T>
where
    T: StandardSample,
{
    /// Creates a new `StereoAudioSamples` from audio data with exactly 2 channels.
    ///
    /// # Errors
    /// Returns an error if `stereo_data` is mono or has a channel count other than 2.
    #[inline]
    pub fn new(stereo_data: AudioData<'a, T>, sample_rate: SampleRate) -> AudioSampleResult<Self> {
        // Separated failure conditions which the following if statements check allow for more descriptive errors.

        if stereo_data.is_mono() {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "channels",
                "Expected stereo data (2 channels), got mono (1 channel).",
            )));
        }

        if stereo_data.num_channels() != channels!(2) {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "channels",
                format!(
                    "Expected stereo data (2 channels), got {} channels.",
                    stereo_data.num_channels()
                ),
            )));
        }

        // From now on we have guaranteed that the stereo_data does in fact have 2 channels;
        Ok(Self(AudioSamples::new(stereo_data, sample_rate)))
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
    /// use audio_samples::{AudioSamples, StereoAudioSamples, sample_rate, AudioSampleResult};
    /// use ndarray::array;
    ///
    /// let stereo_data = array![[0.1f32, 0.2, 0.3], [0.4, 0.5, 0.6]];
    /// let audio = AudioSamples::new_multi_channel(stereo_data, sample_rate!(44100)).unwrap();
    /// let stereo: StereoAudioSamples<'static, f32> = StereoAudioSamples::try_from(audio).unwrap();
    ///
    /// stereo.with_channels(|left, right| -> AudioSampleResult<()> {
    ///     // left and right are borrowed AudioSamples<'_, f32>
    ///     println!("Left channel samples: {}", left.len());
    ///     println!("Right channel samples: {}", right.len());
    ///     Ok(())
    /// }).unwrap();
    /// ```
    ///
    /// # Errors
    ///
    #[inline]
    pub fn with_channels<R, F>(&self, f: F) -> AudioSampleResult<R>
    where
        F: FnOnce(AudioSamples<'_, T>, AudioSamples<'_, T>) -> AudioSampleResult<R>,
    {
        match &self.0.data {
            AudioData::Multi(multi_data) => {
                // Extract left channel (row 0)
                let left_view = multi_data.index_axis(Axis(0), 0);
                // safety: self guarantees non-empty at construction
                let left_data = unsafe { MonoData::from_view_unchecked(left_view) };
                let left_audio =
                    AudioSamples::new(AudioData::Mono(left_data), self.0.sample_rate());

                // Extract right channel (row 1)
                let right_view = multi_data.index_axis(Axis(0), 1);

                // safety: self guarantees non-empty at construction
                let right_data = unsafe { MonoData::from_view_unchecked(right_view) };
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

/// Provides transparent read access to the inner [`AudioSamples`].
impl<'a, T> Deref for StereoAudioSamples<'a, T>
where
    T: StandardSample,
{
    type Target = AudioSamples<'a, T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Provides transparent write access to the inner [`AudioSamples`].
///
/// Note: mutating the inner `AudioSamples` (e.g. via `replace_data`) can potentially
/// violate the stereo invariant. Prefer using [`StereoAudioSamples`] methods directly.
impl<T> DerefMut for StereoAudioSamples<'_, T>
where
    T: StandardSample,
{
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Provides a shared reference to the inner [`AudioSamples`].
impl<'a, T> AsRef<AudioSamples<'a, T>> for StereoAudioSamples<'a, T>
where
    T: StandardSample,
{
    #[inline]
    fn as_ref(&self) -> &AudioSamples<'a, T> {
        &self.0
    }
}

/// Provides a mutable reference to the inner [`AudioSamples`].
impl<'a, T> AsMut<AudioSamples<'a, T>> for StereoAudioSamples<'a, T>
where
    T: StandardSample,
{
    #[inline]
    fn as_mut(&mut self) -> &mut AudioSamples<'a, T> {
        &mut self.0
    }
}

/// Zero-copy conversion from an owned [`AudioSamples`] into a [`StereoAudioSamples`].
///
/// # Errors
///
/// Returns [crate::AudioSampleError::Parameter] if the audio does not have exactly 2 channels.
impl<T> TryFrom<AudioSamples<'static, T>> for StereoAudioSamples<'static, T>
where
    T: StandardSample,
{
    type Error = AudioSampleError;

    #[inline]
    fn try_from(audio: AudioSamples<'static, T>) -> Result<Self, Self::Error> {
        if audio.num_channels() != channels!(2) {
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

/// Unwraps a [`StereoAudioSamples`] back into the underlying [`AudioSamples`].
impl<T> From<StereoAudioSamples<'static, T>> for AudioSamples<'static, T>
where
    T: StandardSample,
{
    #[inline]
    fn from(stereo: StereoAudioSamples<'static, T>) -> Self {
        stereo.0
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayBase, array};
    use non_empty_slice::non_empty_vec;

    #[test]
    fn test_new_mono_audio_samples() {
        let data: ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 1]>> =
            array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio: AudioSamples<f32> = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio.num_channels(), channels!(1));
        assert_eq!(audio.samples_per_channel(), NonZeroUsize::new(5).unwrap());
        assert!(audio.is_mono());
        assert!(!audio.is_multi_channel());
    }

    #[test]
    fn test_new_multi_channel_audio_samples() {
        let data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]; // 2 channels, 3 samples each
        let audio = AudioSamples::new_multi_channel(data, sample_rate!(48000)).unwrap();

        assert_eq!(audio.sample_rate(), sample_rate!(48000));
        assert_eq!(audio.num_channels(), channels!(2));
        assert_eq!(audio.samples_per_channel(), NonZeroUsize::new(3).unwrap());
        assert_eq!(audio.total_samples(), NonZeroUsize::new(6).unwrap());
        assert!(!audio.is_mono());
        assert!(audio.is_multi_channel());
    }

    #[test]
    fn test_zeros_construction() {
        let mono_audio =
            AudioSamples::<f32>::zeros_mono(NonZeroUsize::new(100).unwrap(), sample_rate!(44100));
        assert_eq!(mono_audio.num_channels(), channels!(1));
        assert_eq!(
            mono_audio.samples_per_channel(),
            NonZeroUsize::new(100).unwrap()
        );
        assert_eq!(mono_audio.sample_rate(), sample_rate!(44100));

        let multi_audio = AudioSamples::<f32>::zeros_multi(
            channels!(2),
            NonZeroUsize::new(50).unwrap(),
            sample_rate!(48000),
        );
        assert_eq!(multi_audio.num_channels(), channels!(2));
        assert_eq!(
            multi_audio.samples_per_channel(),
            NonZeroUsize::new(50).unwrap()
        );
        assert_eq!(multi_audio.total_samples(), NonZeroUsize::new(100).unwrap());
        assert_eq!(multi_audio.sample_rate(), sample_rate!(48000));
    }

    #[test]
    fn test_duration_seconds() {
        let audio: AudioSamples<'_, f32> =
            AudioSamples::<f32>::zeros_mono(NonZeroUsize::new(44100).unwrap(), sample_rate!(44100));
        assert!((audio.duration_seconds() - 1.0).abs() < 1e-6);

        let audio2: AudioSamples<'_, f32> = AudioSamples::<f32>::zeros_multi(
            channels!(2),
            NonZeroUsize::new(22050).unwrap(),
            sample_rate!(44100),
        );
        assert!((audio2.duration_seconds() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_apply_simple() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut audio: AudioSamples<f32> =
            AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        // Apply a simple scaling
        audio.apply(|sample| sample * 2.0);

        let expected = array![2.0f32, 4.0, 6.0, 8.0, 10.0];
        let expected = AudioSamples::new_mono(expected, sample_rate!(44100)).unwrap();
        assert_eq!(
            audio, expected,
            "Applied audio samples do not match expected values"
        );
    }

    #[test]
    fn test_apply_channels() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio: AudioSamples<f32> =
            AudioSamples::new_multi_channel(data, sample_rate!(44100)).unwrap();

        {
            // Mutable borrow lives only within this block
            audio.apply_to_channels(&[0, 1], |sample| sample * sample);
        } // Mutable borrow ends here

        let expected = array![[1.0, 4.0], [9.0, 16.0]];
        let expected = AudioSamples::new_multi_channel(expected, sample_rate!(44100)).unwrap();
        assert_eq!(
            audio, expected,
            "Applied multi-channel audio samples do not match expected values"
        );
    }

    #[test]
    fn test_map_functional() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        // Create a new audio instance with transformed samples
        let new_audio = audio.map(|sample| sample * 0.5);

        // Original should be unchanged
        let original_expected = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let original_expected =
            AudioSamples::new_mono(original_expected, sample_rate!(44100)).unwrap();
        assert_eq!(
            audio, original_expected,
            "Original audio samples should remain unchanged"
        );

        // New audio should contain transformed values
        let new_expected = array![0.5f32, 1.0, 1.5, 2.0, 2.5];
        let new_expected = AudioSamples::new_mono(new_expected, sample_rate!(44100)).unwrap();
        assert_eq!(
            new_audio, new_expected,
            "New audio should contain transformed samples"
        );
    }

    #[test]
    fn test_apply_indexed() {
        let data = array![1.0f32, 1.0, 1.0, 1.0, 1.0];
        let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        // Apply index-based transformation
        audio.apply_with_index(|index, sample| sample * (index + 1) as f32);
        let expected = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let expected = AudioSamples::new_mono(expected, sample_rate!(44100)).unwrap();
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
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        assert_eq!(audio[0], 1.0);
        assert_eq!(audio[2], 3.0);
        assert_eq!(audio[4], 5.0);
    }

    #[test]
    #[should_panic(expected = "Cannot use single index on multi-channel audio")]
    fn test_index_mono_single_on_multi_panics() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let audio = AudioSamples::new_multi_channel(data, sample_rate!(44100)).unwrap();

        let _ = audio[0]; // Should panic
    }

    #[test]
    fn test_index_tuple() {
        // Test mono with tuple indexing
        let mono_data = array![1.0f32, 2.0, 3.0, 4.0];
        let mono_audio = AudioSamples::new_mono(mono_data, sample_rate!(44100)).unwrap();

        assert_eq!(mono_audio[(0, 1)], 2.0);
        assert_eq!(mono_audio[(0, 3)], 4.0);

        // Test multi-channel with tuple indexing
        let stereo_data = array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let stereo_audio =
            AudioSamples::new_multi_channel(stereo_data, sample_rate!(44100)).unwrap();

        assert_eq!(stereo_audio[(0, 0)], 1.0);
        assert_eq!(stereo_audio[(0, 2)], 3.0);
        assert_eq!(stereo_audio[(1, 0)], 4.0);
        assert_eq!(stereo_audio[(1, 2)], 6.0);
    }

    #[test]
    #[should_panic(expected = "Channel index 1 out of bounds for mono audio")]
    fn test_index_tuple_invalid_channel_mono() {
        let data = array![1.0f32, 2.0, 3.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let _ = audio[(1, 0)]; // Should panic - mono only has channel 0
    }

    #[test]
    fn test_index_mut_mono() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        audio[2] = 10.0;
        assert_eq!(audio[2], 10.0);
        assert_eq!(
            audio,
            AudioSamples::new_mono(array![1.0f32, 2.0, 10.0, 4.0, 5.0], sample_rate!(44100))
                .unwrap(),
            "Mutably indexed mono audio samples do not match expected values"
        );
    }

    #[test]
    fn test_index_mut_tuple() {
        let data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio = AudioSamples::new_multi_channel(data, sample_rate!(44100)).unwrap();

        audio[(1, 0)] = 10.0;
        assert_eq!(audio[(1, 0)], 10.0);

        let expected = array![[1.0f32, 2.0], [10.0, 4.0]];
        let expected = AudioSamples::new_multi_channel(expected, sample_rate!(44100)).unwrap();
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
        let mut audio = AudioSamples::new_mono(initial_data, sample_rate!(44100)).unwrap();

        // Replace with new mono data of different length
        let new_data = AudioData::try_from(array![4.0f32, 5.0, 6.0, 7.0, 8.0])
            .unwrap()
            .into_owned();
        assert!(audio.replace_data(new_data).is_ok());

        assert_eq!(audio.samples_per_channel(), NonZeroUsize::new(5).unwrap());
        assert_eq!(audio.num_channels(), channels!(1));
        assert_eq!(audio.sample_rate(), sample_rate!(44100)); // Should be preserved
    }

    #[test]
    fn test_replace_data_multi_success() {
        let initial_data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio = AudioSamples::new_multi_channel(initial_data, sample_rate!(44100)).unwrap();

        // Replace with new stereo data of different length
        let new_data = AudioData::try_from(array![[5.0f32, 6.0, 7.0], [8.0, 9.0, 10.0]]).unwrap();
        assert!(audio.replace_data(new_data).is_ok());

        assert_eq!(audio.samples_per_channel(), NonZeroUsize::new(3).unwrap());
        assert_eq!(audio.num_channels(), channels!(2));
        assert_eq!(audio.sample_rate(), sample_rate!(44100)); // Should be preserved
    }

    #[test]
    fn test_replace_data_channel_count_mismatch() {
        let initial_data = array![1.0f32, 2.0, 3.0];
        let mut audio = AudioSamples::new_mono(initial_data, sample_rate!(44100)).unwrap();

        // Try to replace mono with stereo data
        let new_data = AudioData::try_from(array![[4.0f32, 5.0], [6.0, 7.0]]).unwrap();
        let result = audio.replace_data(new_data);

        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Channel count mismatch"));
        }
    }

    #[test]
    fn test_replace_data_layout_type_mismatch() {
        let initial_data = array![1.0f32, 2.0, 3.0];
        let mut audio = AudioSamples::new_mono(initial_data, sample_rate!(44100)).unwrap();

        // Try to replace mono with multi-channel data (even same channel count fails due to type)
        let new_data = AudioData::try_from(array![[4.0f32, 5.0, 6.0]]).unwrap();
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
        let mut audio = AudioSamples::new_mono(initial_data, sample_rate!(44100)).unwrap();

        // Replace with new mono data
        assert!(audio.replace_with_mono(array![3.0f32, 4.0, 5.0]).is_ok());

        assert_eq!(audio.samples_per_channel(), NonZeroUsize::new(3).unwrap());
        assert_eq!(audio.num_channels(), channels!(1));
        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio[0], 3.0);
        assert_eq!(audio[1], 4.0);
        assert_eq!(audio[2], 5.0);
    }

    #[test]
    fn test_replace_with_mono_on_multi_fails() {
        let initial_data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio = AudioSamples::new_multi_channel(initial_data, sample_rate!(44100)).unwrap();

        let result = audio.replace_with_mono(array![5.0f32, 6.0, 7.0]);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Cannot replace multi-channel audio"));
        }
    }

    #[test]
    fn test_replace_with_multi_success() {
        let initial_data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio = AudioSamples::new_multi_channel(initial_data, sample_rate!(44100)).unwrap();

        // Replace with new stereo data of different length
        assert!(
            audio
                .replace_with_multi(array![[5.0f32, 6.0, 7.0], [8.0, 9.0, 10.0]])
                .is_ok()
        );

        assert_eq!(audio.samples_per_channel(), NonZeroUsize::new(3).unwrap());
        assert_eq!(audio.num_channels(), channels!(2));
        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        assert_eq!(audio[(0, 0)], 5.0);
        assert_eq!(audio[(0, 2)], 7.0);
        assert_eq!(audio[(1, 0)], 8.0);
        assert_eq!(audio[(1, 2)], 10.0);
    }

    #[test]
    fn test_replace_with_multi_on_mono_fails() {
        let initial_data = array![1.0f32, 2.0, 3.0];
        let mut audio = AudioSamples::new_mono(initial_data, sample_rate!(44100)).unwrap();

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
        let mut audio = AudioSamples::new_multi_channel(initial_data, sample_rate!(44100)).unwrap();

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
        let mut audio = AudioSamples::new_mono(initial_data, sample_rate!(44100)).unwrap();

        // Replace with vector
        assert!(
            audio
                .replace_with_vec(&non_empty_vec![3.0f32, 4.0, 5.0, 6.0])
                .is_ok()
        );

        assert_eq!(audio.samples_per_channel(), NonZeroUsize::new(4).unwrap());
        assert_eq!(audio.num_channels(), channels!(1));
        assert_eq!(audio.sample_rate(), sample_rate!(44100));
    }

    #[test]
    fn test_replace_with_vec_multi_success() {
        let initial_data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio = AudioSamples::new_multi_channel(initial_data, sample_rate!(44100)).unwrap();

        // Replace with vector (6 samples = 3 samples per channel × 2 channels)
        assert!(
            audio
                .replace_with_vec(&non_empty_vec![5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0])
                .is_ok()
        );

        assert_eq!(audio.samples_per_channel(), NonZeroUsize::new(3).unwrap());
        assert_eq!(audio.num_channels(), channels!(2));
        assert_eq!(audio.sample_rate(), sample_rate!(44100));
        // Data is arranged as: [ch0_sample0, ch0_sample1, ch0_sample2, ch1_sample0, ch1_sample1, ch1_sample2]
        assert_eq!(audio[(0, 0)], 5.0); // First channel, first sample
        assert_eq!(audio[(0, 1)], 6.0); // First channel, second sample
        assert_eq!(audio[(1, 0)], 8.0); // Second channel, first sample
    }

    #[test]
    fn test_replace_with_vec_not_divisible_fails() {
        let initial_data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio = AudioSamples::new_multi_channel(initial_data, sample_rate!(44100)).unwrap();

        // 5 samples is not divisible by 2 channels
        let result = audio.replace_with_vec(&non_empty_vec![5.0f32, 6.0, 7.0, 8.0, 9.0]);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.to_string()
                    .contains("Sample count 5 is not divisible by channel count 2")
            );
        }
    }

    #[test]
    fn test_replace_with_vec_channels_success() {
        // Test with correct channel count
        let initial_data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio = AudioSamples::new_multi_channel(initial_data, sample_rate!(44100)).unwrap();

        // Replace with vector, validating against expected 2 channels
        assert!(
            audio
                .replace_with_vec_channels(&non_empty_vec![5.0f32, 6.0, 7.0, 8.0], channels!(2))
                .is_ok()
        );

        assert_eq!(audio.samples_per_channel(), NonZeroUsize::new(2).unwrap());
        assert_eq!(audio.num_channels(), channels!(2));
        assert_eq!(audio.sample_rate(), sample_rate!(44100));
    }

    #[test]
    fn test_replace_with_vec_channels_sample_count_mismatch() {
        let initial_data = array![[1.0f32, 2.0], [3.0, 4.0]];
        let mut audio = AudioSamples::new_multi_channel(initial_data, sample_rate!(44100)).unwrap();

        // Try with sample count not divisible by expected channels
        let result =
            audio.replace_with_vec_channels(&non_empty_vec![5.0f32, 6.0, 7.0], channels!(2));
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
        let mut audio = AudioSamples::new_multi_channel(initial_data, sample_rate!(44100)).unwrap();

        // Try with expected channel count different from current audio
        let result =
            audio.replace_with_vec_channels(&non_empty_vec![5.0f32, 6.0, 7.0], channels!(3));
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
            AudioSamples::new_multi_channel(array![[0.0f32, 0.0], [0.0, 0.0]], sample_rate!(44100))
                .unwrap();

        // Simulate converted data from their function
        let converted_samples: NonEmptyVec<f32> = non_empty_vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let num_channels = channels!(2); // This comes from the function parameter, not audio.num_channels()

        // Validate against the parameter, not the existing audio configuration
        assert!(
            converted_samples
                .len()
                .get()
                .is_multiple_of(num_channels.get() as usize)
        );
        assert!(
            audio
                .replace_with_vec_channels(&converted_samples, num_channels)
                .is_err(),
            "audio has 2 channels with 2 samples per channel, but converted samples have 2 channels and 3 samples per channel (6/2)"
        );
    }

    #[test]
    fn test_metadata_preservation_across_replacements() {
        let mut audio = AudioSamples::new_mono(array![1.0f32, 2.0], sample_rate!(48000)).unwrap();

        let original_rate = audio.sample_rate();

        // Replace data multiple times
        assert!(audio.replace_with_mono(array![3.0f32, 4.0, 5.0]).is_ok());
        assert_eq!(audio.sample_rate(), original_rate);

        assert!(
            audio
                .replace_with_vec(&non_empty_vec![6.0f32, 7.0, 8.0, 9.0])
                .is_ok()
        );
        assert_eq!(audio.sample_rate(), original_rate);

        let new_data = AudioData::try_from(array![10.0f32, 11.0])
            .unwrap()
            .into_owned();
        assert!(audio.replace_data(new_data).is_ok());
        assert_eq!(audio.sample_rate(), original_rate);
    }
}
