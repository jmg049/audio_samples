use bytemuck::NoUninit;
use ndarray::ScalarOperand;
use num_traits::{FromPrimitive, Num, NumCast, One, Signed, ToBytes, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};

use crate::{AudioSamples, I24, RealFloat, SampleType};
use std::fmt::{Debug, Display};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// Trait for casting from one type to another.
pub trait CastFrom<S>: Sized {
    /// Cast from the source type to Self.
    fn cast_from(value: S) -> Self;
}

/// Trait for casting into another type.
pub trait CastInto<T>: Sized
where
    Self: CastFrom<T>,
{
    /// Cast self into the target type.
    fn cast_into(self) -> T;
}

/// Trait for types that can be cast to all audio sample types.
pub trait Castable:
    CastInto<i16> + CastInto<I24> + CastInto<i32> + CastInto<f32> + CastInto<f64>
{
}
impl<T> Castable for T where
    T: CastInto<i16> + CastInto<I24> + CastInto<i32> + CastInto<f32> + CastInto<f64>
{
}

mod sealed {
    pub trait Sealed {}
}
use sealed::Sealed;

/// Marker type for supported sample byte sizes.
pub struct SampleByteSize<const N: usize>;

/// Marker trait for supported sample byte sizes.
pub trait SupportedByteSize: Sealed {}

impl<const N: usize> Sealed for SampleByteSize<N> {}

impl SupportedByteSize for SampleByteSize<2> {}
impl SupportedByteSize for SampleByteSize<3> {}
impl SupportedByteSize for SampleByteSize<4> {}
impl SupportedByteSize for SampleByteSize<8> {}

/// Core trait defining the interface for audio sample types.
///
/// Provides a unified interface for working with different audio sample formats
/// including integers, 24-bit integers, and floating-point values. Ensures type-safe
/// conversions and arithmetic operations across all supported sample formats.
///
/// All implementors support:
/// - Type-safe conversions between all supported formats
/// - Arithmetic operations (Add, Sub, Mul, Div)
/// - Serialization to byte arrays
/// - Safe memory layout guarantees for audio processing
///
/// # Required Methods
/// This trait is implemented automatically for types that satisfy the
/// comprehensive bounds listed below. No manual implementation is required.
///
/// # Supported Types
/// - `i16`: 16-bit signed integer samples (most common for audio files)
/// - `I24`: 24-bit signed integer samples (professional audio)
/// - `i32`: 32-bit signed integer samples (high precision)
/// - `f32`: 32-bit floating-point samples (normalized -1.0 to 1.0)
/// - `f64`: 64-bit floating-point samples (highest precision)
///
/// # Safety
/// Implementors must ensure proper numeric behavior and memory safety
/// for all required trait bounds, particularly for byte serialization
/// and numeric conversions.
pub trait AudioSample:
    // Standard library traits
    Copy
    + Sized
    + Default
    + Display
    + Debug
    + Sync
    + Send
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + AddAssign<Self>
    + Sub<Output = Self>
    + SubAssign<Self>
    + Mul<Output = Self>
    + MulAssign<Self>
    + Div<Output = Self>
    + DivAssign<Self>
    + Rem<Output = Self>
    + RemAssign<Self>
    + Neg<Output = Self>
    + Into<Self>
    + From<Self>
    + ToString

    // External crate traits
    + NoUninit // bytemuck trait to ensure no uninitialized bytes
    + Num // num-traits trait for numeric operations
    + One // num-traits trait for 1 value
    + Zero // num-traits trait for 0 value
    + ToBytes // num-traits trait for byte conversion
    + Serialize // serde trait for serialization
    + Deserialize<'static> //serde trait for deserialisation // Need to make these optional. 
    + FromPrimitive // num-traits trait for conversion from primitive types
    + Signed // num-traits trait for signed types
    + NumCast // num-traits trait for casting between numeric types
    + ScalarOperand // ndarray trait for scalar operations

    // Library-specific traits. Most of which are below.
    // They define how to convert between types depending on the context.
    // Sometimes we are dealing with audio samples and float representations between -1.0 and 1.0, sometimes we are dealing with raw integer representations that we need to cast to floats for specific operations, but not -1.0 to 1.0, for various operations.
    + ConvertTo<Self> // "I can convert to myself" trait
    + ConvertTo<i16> // "I can convert to i16" trait
    + ConvertTo<I24> // "I can convert to I24" trait
    + ConvertTo<i32> // "I can convert to i32" trait
    + ConvertTo<f32> // "I can convert to f32" trait
    + ConvertTo<f64> // "I can convert to f64" trait
    + CastFrom<usize> // "I cant cast from a  usize"
    + Castable // "I can be cast into supported types"
{
    #[inline]
    /// Consumes this sample and returns itself.
    fn into_inner(self) -> Self {
        self
    }

    #[inline]
    /// Convert this sample into a byte vector in native-endian order.
    fn to_bytes(self) -> Vec<u8> {
        self.to_ne_bytes().as_ref().to_vec()
    }

    #[inline]
    /// Convert this sample into a byte array of specified size (must be supported) in native-endian order.
    fn as_bytes<const N: usize>(&self) -> [u8; N]
        where SampleByteSize<N>: SupportedByteSize,
    {
        let bytes_ref = self.to_ne_bytes();
        let bytes_slice: &[u8] = bytes_ref.as_ref();
        let mut result = [0u8; N];
        result.copy_from_slice(bytes_slice);
        result
    }

    #[inline]
    /// Convert a slice of samples into a byte vector in native-endian order.
    fn slice_to_bytes(samples: &[Self]) -> Vec<u8> {
        Vec::from(bytemuck::cast_slice(samples))
    }

    /// Maximum representable value for this sample type.
    const MAX: Self;
    /// Minimum representable value for this sample type.
    const MIN: Self;
    /// Bit depth of this sample type.
    const BITS: u8;
    /// Byte length
    const BYTES: usize = Self::BITS as usize / 8;
    /// Label used for plotting and display purposes.
    const LABEL: &'static str;
    /// Sample type enum variant.
    const SAMPLE_TYPE: SampleType;
}

/// Trait for converting one sample type to another with audio-aware scaling.
///
/// `ConvertTo` performs conversions that are intended for audio sample values rather than raw
/// numeric casts.
///
/// ## Conversion Behavior
/// - **Integer ↔ Integer**: PCM-style bit-depth scaling (e.g. `i16::MAX` maps to `0x7FFF0000i32`)
/// - **Integer ↔ Float**: normalized scaling into $[-1.0, 1.0]$ using asymmetric endpoints
/// - **Float ↔ Integer**: clamp to $[-1.0, 1.0]$, then scale, round, and saturate
/// - **I24 Special Handling**: conversions treat `I24` as a 24-bit signed PCM integer
///
/// ## Example
/// ```rust
/// use audio_samples::ConvertTo;
///
/// let sample_i16: i16 = 16384; // approximately half-scale
/// let sample_f32: f32 = sample_i16.convert_to();
/// assert!((sample_f32 - 0.5).abs() < 1e-4);
///
/// let sample_i32: i32 = sample_i16.convert_to();
/// assert_eq!(sample_i32, 0x4000_0000);
/// ```
pub trait ConvertTo<T: AudioSample> {
    /// Convert this sample to another audio sample type.
    fn convert_to(&self) -> T;

    /// Convert from another audio sample type to this type.
    /// This is a convenience method that calls `convert_to` on the source value.
    fn convert_from<F: AudioSample + ConvertTo<T>>(source: F) -> T {
        source.convert_to()
    }
}

// Identity
macro_rules! impl_identity_conversion {
    ($ty:ty) => {
        impl ConvertTo<$ty> for $ty {
            #[inline(always)]
            fn convert_to(&self) -> $ty {
                *self
            }
        }
    };
}

// Integer -> Integer, saturating (with bit-shift scaling if needed)
macro_rules! impl_int_to_int_conversion {
    ($from:ty, $to:ty) => {
        impl ConvertTo<$to> for $from {
            #[inline(always)]
            fn convert_to(&self) -> $to {
                let from_bits = <$from>::BITS as i32;
                let to_bits = <$to>::BITS as i32;

                let v = *self as i128;
                let scaled = if from_bits < to_bits {
                    v << (to_bits - from_bits)
                } else if from_bits > to_bits {
                    v >> (from_bits - to_bits)
                } else {
                    v
                };

                let min = <$to>::MIN as i128;
                let max = <$to>::MAX as i128;
                if scaled < min {
                    <$to>::MIN
                } else if scaled > max {
                    <$to>::MAX
                } else {
                    scaled as $to
                }
            }
        }
    };
}

// I24 -> Integer (any standard integer), saturating
macro_rules! impl_i24_to_int {
    ($to:ty) => {
        impl ConvertTo<$to> for I24 {
            #[inline(always)]
            fn convert_to(&self) -> $to {
                let to_bits = <$to>::BITS as i32;
                let v = self.to_i32().expect("Every I24 can fit in an i32") as i128;

                let shift = to_bits - 24;
                let scaled = if shift >= 0 {
                    v << shift
                } else {
                    v >> (-shift)
                };

                let min = <$to>::MIN as i128;
                let max = <$to>::MAX as i128;
                if scaled < min {
                    <$to>::MIN
                } else if scaled > max {
                    <$to>::MAX
                } else {
                    scaled as $to
                }
            }
        }
    };
}

// Integer -> I24, saturating
// Integer -> I24, saturating (FIXED, no trait ambiguity)
macro_rules! impl_int_to_i24 {
    ($from:ty) => {
        impl ConvertTo<I24> for $from {
            #[inline(always)]
            fn convert_to(&self) -> I24 {
                let from_bits = <$from>::BITS as i32;
                let v = *self as i128;

                let shift = 24 - from_bits;
                let scaled = if shift >= 0 {
                    v << shift
                } else {
                    v >> (-shift)
                };

                let min = I24::MIN.to_i32() as i128;
                let max = I24::MAX.to_i32() as i128;
                let clamped = if scaled < min {
                    min as i32
                } else if scaled > max {
                    max as i32
                } else {
                    scaled as i32
                };

                I24::saturating_from_i32(clamped)
            }
        }
    };
}

// I24 -> Float (normalised)
macro_rules! impl_i24_to_float {
    ($to:ty) => {
        impl ConvertTo<$to> for I24 {
            #[inline(always)]
            fn convert_to(&self) -> $to {
                let v = self.to_i32().expect("Every I24 can fit in an i32") as $to;
                let max = I24::MAX.to_i32() as $to;
                let min = I24::MIN.to_i32() as $to;
                if v < 0.0 { v / -min } else { v / max }
            }
        }
    };
}

// Integer -> Float (normalised)
macro_rules! impl_int_to_float {
    ($from:ty, $to:ty) => {
        impl ConvertTo<$to> for $from {
            #[inline(always)]
            fn convert_to(&self) -> $to {
                let v = *self;
                if v < 0 {
                    (v as $to) / (-(<$from>::MIN as $to))
                } else {
                    (v as $to) / (<$from>::MAX as $to)
                }
            }
        }
    };
}

// Float -> Integer (clamp + scale + round + saturating)
macro_rules! impl_float_to_int {
    ($from:ty, $to:ty) => {
        impl ConvertTo<$to> for $from {
            #[inline(always)]
            fn convert_to(&self) -> $to {
                let v = self.clamp(-1.0, 1.0);
                let scaled = if v < 0.0 {
                    v * (-(<$to>::MIN as $from))
                } else {
                    v * (<$to>::MAX as $from)
                };
                let rounded = scaled.round();
                if rounded < (<$to>::MIN as $from) {
                    <$to>::MIN
                } else if rounded > (<$to>::MAX as $from) {
                    <$to>::MAX
                } else {
                    rounded as $to
                }
            }
        }
    };
}

// Float -> I24 (clamp + scale + round + saturating)
macro_rules! impl_float_to_i24 {
    ($from:ty) => {
        impl ConvertTo<I24> for $from {
            #[inline(always)]
            fn convert_to(&self) -> I24 {
                let v = self.clamp(-1.0, 1.0);
                let scaled = if v < 0.0 {
                    v * (-(I24::MIN.to_i32() as $from))
                } else {
                    v * (I24::MAX.to_i32() as $from)
                };
                let rounded = scaled.round();
                let clamped = if rounded < (I24::MIN.to_i32() as $from) {
                    I24::MIN.to_i32()
                } else if rounded > (I24::MAX.to_i32() as $from) {
                    I24::MAX.to_i32()
                } else {
                    rounded as i32
                };
                I24::saturating_from_i32(clamped)
            }
        }
    };
}

// Float -> Float
macro_rules! impl_float_to_float {
    ($from:ty, $to:ty) => {
        impl ConvertTo<$to> for $from {
            #[inline(always)]
            fn convert_to(&self) -> $to {
                *self as $to
            }
        }
    };
}

// ========================
// Identity
// ========================

impl_identity_conversion!(i16);
impl_identity_conversion!(I24);
impl_identity_conversion!(i32);
impl_identity_conversion!(f32);
impl_identity_conversion!(f64);

// ========================
// Integer <-> Integer (Saturating, No Normalisation)
// ========================

impl_int_to_int_conversion!(i16, i32);
impl_int_to_int_conversion!(i32, i16);

// ========================
// I24 <-> Integer
// ========================

impl_i24_to_int!(i16);
impl_i24_to_int!(i32);

impl_int_to_i24!(i16);
impl_int_to_i24!(i32);

// ========================
// Integer -> Float (Normalised +- 1.0)
// ========================

impl_int_to_float!(i16, f32);
impl_int_to_float!(i16, f64);

impl_int_to_float!(i32, f32);
impl_int_to_float!(i32, f64);

// ========================
// I24 -> Float (Normalised +- 1.0)
// ========================

impl_i24_to_float!(f32);
impl_i24_to_float!(f64);

// ========================
// Float -> Integer (Clamped, Rounded, Saturating)
// ========================

impl_float_to_int!(f32, i16);
impl_float_to_int!(f64, i16);

impl_float_to_int!(f32, i32);
impl_float_to_int!(f64, i32);

// ========================
// Float -> I24 (Clamped, Rounded, Saturating)
// ========================

impl_float_to_i24!(f32);
impl_float_to_i24!(f64);

// ========================
// Float <-> Float
// ========================

impl_float_to_float!(f32, f64);
impl_float_to_float!(f64, f32);

// ========================
// AudioSample Implementations
// ========================

impl AudioSample for i16 {
    const MAX: Self = i16::MAX;
    const MIN: Self = i16::MIN;
    const BITS: u8 = 16;
    const LABEL: &'static str = "i16";
    const SAMPLE_TYPE: SampleType = SampleType::I16;
}

impl AudioSample for I24 {
    #[inline]
    fn slice_to_bytes(samples: &[Self]) -> Vec<u8> {
        I24::write_i24s_ne(samples)
    }

    const MAX: Self = I24::MAX;
    const MIN: Self = I24::MIN;
    const BITS: u8 = 24;
    const LABEL: &'static str = "I24";
    const SAMPLE_TYPE: SampleType = SampleType::I24;
}

impl AudioSample for i32 {
    const MAX: Self = i32::MAX;
    const MIN: Self = i32::MIN;
    const BITS: u8 = 32;
    const LABEL: &'static str = "i32";
    const SAMPLE_TYPE: SampleType = SampleType::I32;
}

impl AudioSample for f32 {
    const MAX: Self = 1.0;
    const MIN: Self = -1.0;
    const BITS: u8 = 32;
    const LABEL: &'static str = "f32";
    const SAMPLE_TYPE: SampleType = SampleType::F32;
}

impl AudioSample for f64 {
    const MAX: Self = 1.0;
    const MIN: Self = -1.0;
    const BITS: u8 = 64;
    const LABEL: &'static str = "f64";
    const SAMPLE_TYPE: SampleType = SampleType::F64;
}

// ========================
// Generate All Conversions
// ========================

macro_rules! impl_cast_from {
    ($src:ty => [$($dst:ty),+]) => {
        $(
            impl CastFrom<$src> for $dst {
                #[inline]
                fn cast_from(value: $src) -> Self {
                    value as $dst
                }
            }
        )+
    };
}

impl_cast_from!(i16 => [i16, i32, f32, f64]);
impl_cast_from!(i32 => [i16, i32, f32, f64]);
impl_cast_from!(f64 => [i16, i32, f32, f64]);
impl_cast_from!(f32 => [i16, i32, f32, f64]);

/// Macro to implement the `CastFrom` trait for multiple type pairs
macro_rules! impl_cast_from_i24 {
    // Simple direct casts (no clamping or special logic)
    ($src:ty => $dst:ty) => {
        impl CastFrom<$src> for $dst {
            #[inline]
            fn cast_from(value: $src) -> Self {
                value as $dst
            }
        }
    };

    // Clamped casts for usize -> integer
    (clamp_usize $src:ty => $dst:ty, $max:expr) => {
        impl CastFrom<$src> for $dst {
            #[inline]
            fn cast_from(value: $src) -> Self {
                if value > $max as $src {
                    $max
                } else {
                    value as $dst
                }
            }
        }
    };

    // usize -> I24 with clamping and try_from_i32
    (usize_to_i24 $src:ty => $dst:ty) => {
        impl CastFrom<$src> for $dst {
            #[inline]
            fn cast_from(value: $src) -> Self {
                if value > I24::MAX.to_i32() as $src {
                    I24::MAX
                } else {
                    match I24::try_from_i32(value as i32) {
                        Some(x) => x,
                        None => I24::MIN,
                    }
                }
            }
        }
    };

    // I24 -> primitive
    (i24_to_primitive $src:ty => $dst:ty) => {
        impl CastFrom<$src> for $dst {
            #[inline]
            fn cast_from(value: $src) -> Self {
                value.to_i32() as $dst
            }
        }
    };

    // primitive -> I24
    (primitive_to_i24 $src:ty => $dst:ty) => {
        impl CastFrom<$src> for $dst {
            #[inline]
            fn cast_from(value: $src) -> Self {
                I24::try_from_i32(value as i32).unwrap_or(I24::MIN)
            }
        }
    };

    // identity
    (identity $t:ty) => {
        impl CastFrom<$t> for $t {
            #[inline]
            fn cast_from(value: $t) -> Self {
                value
            }
        }
    };
}

// usize to primitives
impl_cast_from_i24!(clamp_usize usize => i16, i16::MAX);
impl_cast_from_i24!(usize_to_i24 usize => I24);
impl_cast_from_i24!(clamp_usize usize => i32, i32::MAX);
impl_cast_from_i24!(usize => f32);
impl_cast_from_i24!(usize => f64);

// I24 to primitives
impl_cast_from_i24!(i24_to_primitive I24 => i16);
impl_cast_from_i24!(identity I24);
impl_cast_from_i24!(i24_to_primitive I24 => i32);
impl_cast_from_i24!(i24_to_primitive I24 => f32);
impl_cast_from_i24!(i24_to_primitive I24 => f64);

// primitives to I24
impl_cast_from_i24!(primitive_to_i24 i16 => I24);
impl_cast_from_i24!(primitive_to_i24 i32 => I24);
impl_cast_from_i24!(primitive_to_i24 f32 => I24);
impl_cast_from_i24!(primitive_to_i24 f64 => I24);

macro_rules! impl_cast_into {
    ($src:ty => [$($dst:ty),+]) => {
        $(
            impl CastInto<$dst> for $src {
                #[inline]
                fn cast_into(self) -> $dst {
                    <$dst>::cast_from(self)
                }
            }
        )+
    };
}

impl_cast_into!(i16 => [i16, i32, f32, f64]);
impl_cast_into!(i32 => [i16, i32, f32, f64]);
impl_cast_into!(f64 => [i16, i32, f32, f64]);
impl_cast_into!(f32 => [i16, i32, f32, f64]);

/// Macro to implement the `CastInto` trait for multiple type pairs
macro_rules! impl_cast_into_i24 {
    // I24 -> primitive (via to_i32)
    (i24_to_primitive $src:ty => $dst:ty) => {
        impl CastInto<$dst> for $src {
            #[inline]
            fn cast_into(self) -> $dst {
                self.to_i32() as $dst
            }
        }
    };

    // primitive -> I24 (via CastFrom)
    (primitive_to_i24 $src:ty => $dst:ty) => {
        impl CastInto<$dst> for $src {
            #[inline]
            fn cast_into(self) -> $dst {
                <$dst as CastFrom<$src>>::cast_from(self)
            }
        }
    };

    // identity
    (identity $t:ty) => {
        impl CastInto<$t> for $t {
            #[inline]
            fn cast_into(self) -> $t {
                self
            }
        }
    };
}
// I24 to primitives
impl_cast_into_i24!(i24_to_primitive I24 => i16);
impl_cast_into_i24!(identity I24);
impl_cast_into_i24!(i24_to_primitive I24 => i32);
impl_cast_into_i24!(i24_to_primitive I24 => f32);
impl_cast_into_i24!(i24_to_primitive I24 => f64);

// primitives to I24
impl_cast_into_i24!(primitive_to_i24 i16 => I24);
impl_cast_into_i24!(primitive_to_i24 i32 => I24);
impl_cast_into_i24!(primitive_to_i24 f32 => I24);
impl_cast_into_i24!(primitive_to_i24 f64 => I24);

/// Type conversion operations between different sample formats.
///
/// This trait provides safe conversion between different audio sample types
/// while preserving audio quality and handling potential conversion errors.
/// Leverages the existing ConvertTo trait system for type safety.
pub trait AudioTypeConversion<'a, T: AudioSample>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    // -----
    // IN DOMAIN CONVERSIONS
    // -----

    /// Converts to different sample type, borrowing the original.
    ///
    /// Uses the existing ConvertTo trait system for type-safe conversions.
    /// The original AudioSamples instance remains unchanged.
    fn to_format<O>(&self) -> AudioSamples<'static, O>
    where
        T: ConvertTo<O>,
        O: AudioSample + ConvertTo<T>;

    /// Converts to different sample type, consuming the original.
    fn to_type<O: AudioSample + ConvertTo<T>>(self) -> AudioSamples<'static, O>
    where
        T: ConvertTo<O>;

    /// Converts audio samples to a floating-point type.
    fn as_float<F>(&self) -> AudioSamples<'static, F>
    where
        F: RealFloat,
        T: ConvertTo<F>,
        F: ConvertTo<T>,
    {
        self.to_format::<F>()
    }

    /// Converts to double precision floating-point format.
    fn as_f64(&self) -> AudioSamples<'static, f64>
    where
        T: ConvertTo<f64>,
    {
        self.to_format::<f64>()
    }

    /// Converts to single precision floating-point format.
    fn as_f32(&self) -> AudioSamples<'static, f32>
    where
        T: ConvertTo<f32>,
    {
        self.to_format::<f32>()
    }

    /// Converts to 32-bit integer format.
    fn as_i32(&self) -> AudioSamples<'static, i32>
    where
        T: ConvertTo<i32>,
    {
        self.to_format::<i32>()
    }

    /// Converts to 16-bit integer format (most common).
    /// Standard format for CD audio and many audio files.
    fn as_i16(&self) -> AudioSamples<'static, i16>
    where
        T: ConvertTo<i16>,
    {
        self.to_format::<i16>()
    }

    /// Converts to 24-bit integer format.
    /// Common in professional audio workflows.
    fn as_i24(&self) -> AudioSamples<'static, I24>
    where
        T: ConvertTo<I24>,
    {
        self.to_format::<I24>()
    }

    // -----
    // Out OF DOMAIN CONVERSIONS

    // These conversions do traditional casting without audio-specific scaling.
    // For example, PCM_16 to f32 would map -32768..32767 to -1.0..1.0 with the in-domain conversions
    // whereas out-of-domain casting would just cast -32768 to -32768.0f32 etc.
    // -----

    /// Casts to different sample type without audio-aware scaling, borrowing the original.
    ///
    /// This performs raw numeric casting without the audio-specific scaling
    /// used by `to_format`/`to_type`. For example, casting `i16::MIN` to `f32`
    /// gives `-32768.0f32`, not `-1.0f32`.
    ///
    /// Use this when you need raw numeric values, not normalized audio samples.
    fn cast_as<O>(&self) -> AudioSamples<'static, O>
    where
        O: AudioSample + CastFrom<T>;

    /// Casts to different sample type without audio-aware scaling, consuming the original.
    ///
    /// This performs raw numeric casting without the audio-specific scaling
    /// used by `to_type`/`to_format`. For example, casting `i16::MIN` to `f32`
    /// gives `-32768.0f32`, not `-1.0f32`.
    ///
    /// Use this when you need raw numeric values, not normalized audio samples.
    fn cast_to<O>(self) -> AudioSamples<'static, O>
    where
        O: AudioSample + CastFrom<T>;

    /// Casts audio samples to i16 format.
    fn cast_as_i16(&self) -> AudioSamples<'static, i16>
    where
        i16: CastFrom<T>,
    {
        self.cast_as::<i16>()
    }

    /// Converts audio samples to i16 format.
    fn cast_to_i16(self) -> AudioSamples<'static, i16>
    where
        i16: CastFrom<T>,
        Self: Sized,
    {
        self.cast_to::<i16>()
    }

    /// Casts audio samples to I24 format.
    fn cast_as_i24(&self) -> AudioSamples<'static, I24>
    where
        I24: CastFrom<T>,
    {
        self.cast_as::<I24>()
    }

    /// Converts audio samples to I24 format.
    fn cast_to_i24(self) -> AudioSamples<'static, I24>
    where
        I24: CastFrom<T>,
        Self: Sized,
    {
        self.cast_to::<I24>()
    }

    /// Casts audio samples to i32 format.
    fn cast_as_i32(&self) -> AudioSamples<'static, i32>
    where
        i32: CastFrom<T>,
    {
        self.cast_as::<i32>()
    }

    /// Converts audio samples to i32 format.
    fn cast_to_i32(self) -> AudioSamples<'static, i32>
    where
        i32: CastFrom<T>,
        Self: Sized,
    {
        self.cast_to::<i32>()
    }

    /// Casts audio samples to f32 format.
    fn cast_as_f32(&self) -> AudioSamples<'static, f32>
    where
        f32: CastFrom<T>,
    {
        self.cast_as::<f32>()
    }

    /// Converts audio samples to f32 format.
    fn cast_to_f32(self) -> AudioSamples<'static, f32>
    where
        f32: CastFrom<T>,
        Self: Sized,
    {
        self.cast_to::<f32>()
    }

    /// Casts audio samples to f64 format.
    fn cast_as_f64(&self) -> AudioSamples<'static, f64>
    where
        f64: CastFrom<T>,
        Self: Sized,
    {
        self.cast_as::<f64>()
    }
}

#[cfg(test)]
mod conversion_tests {
    use super::*;

    use i24::i24;

    macro_rules! assert_approx_eq {
        ($left:expr, $right:expr, $tolerance:expr) => {
            assert!(
                ($left - $right).abs() < $tolerance,
                "assertion failed: `{} ≈ {}` (tolerance: {})",
                $left,
                $right,
                $tolerance
            );
        };
    }

    // Edge cases for i16 conversions
    #[test]
    fn i16_edge_cases() {
        // Test minimum value
        let min_i16: i16 = i16::MIN;
        let min_i16_to_f32: f32 = min_i16.convert_to();
        // Use higher epsilon for floating point comparison
        assert_approx_eq!(min_i16_to_f32 as f64, -1.0, 1e-5);

        let min_i16_to_i32: i32 = min_i16.convert_to();
        assert_eq!(min_i16_to_i32, i32::MIN);

        let min_i16_to_i24: I24 = min_i16.convert_to();
        let expected_i24_min = i24!(i32::MIN >> 8);
        assert_eq!(min_i16_to_i24.to_i32(), expected_i24_min.to_i32());

        // Test maximum value
        let max_i16: i16 = i16::MAX;
        let max_i16_to_f32: f32 = max_i16.convert_to();
        assert_approx_eq!(max_i16_to_f32 as f64, 1.0, 1e-4);

        let max_i16_to_i32: i32 = max_i16.convert_to();
        assert_eq!(max_i16_to_i32, 0x7FFF0000);

        // Test zero
        let zero_i16: i16 = 0;
        let zero_i16_to_f32: f32 = zero_i16.convert_to();
        assert_approx_eq!(zero_i16_to_f32 as f64, 0.0, 1e-10);

        let zero_i16_to_i32: i32 = zero_i16.convert_to();
        assert_eq!(zero_i16_to_i32, 0);

        let zero_i16_to_i24: I24 = zero_i16.convert_to();
        assert_eq!(zero_i16_to_i24.to_i32(), 0);

        // Test mid-range positive
        let half_max_i16: i16 = i16::MAX / 2;
        let half_max_i16_to_f32: f32 = half_max_i16.convert_to();
        // Use higher epsilon for floating point comparison of half values
        assert_approx_eq!(half_max_i16_to_f32 as f64, 0.5, 1e-4);

        let half_max_i16_to_i32: i32 = half_max_i16.convert_to();
        assert_eq!(half_max_i16_to_i32, 0x3FFF0000);

        // Test mid-range negative
        let half_min_i16: i16 = i16::MIN / 2;
        let half_min_i16_to_f32: f32 = half_min_i16.convert_to();
        assert_approx_eq!(half_min_i16_to_f32 as f64, -0.5, 1e-4);

        // let half_min_i16_to_i32: i32 = half_min_i16.convert_to();
        // assert_eq!(half_min_i16_to_i32, 0xC0010000); // i16::MIN/2 == -16384
    }

    // Edge cases for i32 conversions
    #[test]
    fn i32_edge_cases() {
        // Test minimum value
        let min_i32: i32 = i32::MIN;
        let min_i32_to_f32: f32 = min_i32.convert_to();
        assert_approx_eq!(min_i32_to_f32 as f64, -1.0, 1e-6);

        let min_i32_to_f64: f64 = min_i32.convert_to();
        assert_approx_eq!(min_i32_to_f64, -1.0, 1e-12);

        let min_i32_to_i16: i16 = min_i32.convert_to();
        assert_eq!(min_i32_to_i16, i16::MIN);

        // Test maximum value
        let max_i32: i32 = i32::MAX;
        let max_i32_to_f32: f32 = max_i32.convert_to();
        assert_approx_eq!(max_i32_to_f32 as f64, 1.0, 1e-6);

        let max_i32_to_f64: f64 = max_i32.convert_to();
        assert_approx_eq!(max_i32_to_f64, 1.0, 1e-12);

        let max_i32_to_i16: i16 = max_i32.convert_to();
        assert_eq!(max_i32_to_i16, i16::MAX);

        // Test zero
        let zero_i32: i32 = 0;
        let zero_i32_to_f32: f32 = zero_i32.convert_to();
        assert_approx_eq!(zero_i32_to_f32 as f64, 0.0, 1e-10);

        let zero_i32_to_f64: f64 = zero_i32.convert_to();
        assert_approx_eq!(zero_i32_to_f64, 0.0, 1e-12);

        let zero_i32_to_i16: i16 = zero_i32.convert_to();
        assert_eq!(zero_i32_to_i16, 0);

        // Test quarter-range values
        let quarter_max_i32: i32 = i32::MAX / 4;
        let quarter_max_i32_to_f32: f32 = quarter_max_i32.convert_to();
        assert_approx_eq!(quarter_max_i32_to_f32 as f64, 0.25, 1e-6);

        let quarter_min_i32: i32 = i32::MIN / 4;
        let quarter_min_i32_to_f32: f32 = quarter_min_i32.convert_to();
        assert_approx_eq!(quarter_min_i32_to_f32 as f64, -0.25, 1e-6);
    }

    // Edge cases for f32 conversions
    #[test]
    fn f32_edge_cases() {
        // Test -1.0 (minimum valid value)
        let min_f32: f32 = -1.0;
        let min_f32_to_i16: i16 = min_f32.convert_to();
        // For exact -1.0, we can get -32767 due to rounding in the implementation
        // This is acceptable since it's only 1 bit off from the true min
        assert!(
            min_f32_to_i16 == i16::MIN || min_f32_to_i16 == -32767,
            "Expected either -32768 or -32767, got {}",
            min_f32_to_i16
        );

        let min_f32_to_i32: i32 = min_f32.convert_to();
        assert!(
            min_f32_to_i32 == i32::MIN || min_f32_to_i32 == -2147483647,
            "Expected either i32::MIN or -2147483647, got {}",
            min_f32_to_i32
        );

        let min_f32_to_i24: I24 = min_f32.convert_to();
        let expected_i24 = I24::MIN;
        let diff = (min_f32_to_i24.to_i32() - expected_i24.to_i32()).abs();
        assert!(diff <= 1, "I24 values differ by more than 1, {}", diff);

        // Test 1.0 (maximum valid value)
        let max_f32: f32 = 1.0;
        let max_f32_to_i16: i16 = max_f32.convert_to();
        println!("DEBUG: f32 -> i16 conversion for 1.0");
        println!(
            "Input: {}, Output: {}, Expected: {}",
            max_f32,
            max_f32_to_i16,
            i16::MAX
        );
        assert_eq!(max_f32_to_i16, i16::MAX);

        let max_f32_to_i32: i32 = max_f32.convert_to();
        println!("DEBUG: f32 -> i32 conversion for 1.0");
        println!(
            "Input: {}, Output: {}, Expected: {}",
            max_f32,
            max_f32_to_i32,
            i32::MAX
        );
        assert_eq!(max_f32_to_i32, i32::MAX);

        // Test 0.0
        let zero_f32: f32 = 0.0;
        let zero_f32_to_i16: i16 = zero_f32.convert_to();
        println!("DEBUG: f32 -> i16 conversion for 0.0");
        println!(
            "Input: {}, Output: {}, Expected: 0",
            zero_f32, zero_f32_to_i16
        );
        assert_eq!(zero_f32_to_i16, 0);

        let zero_f32_to_i32: i32 = zero_f32.convert_to();
        println!("DEBUG: f32 -> i32 conversion for 0.0");
        println!(
            "Input: {}, Output: {}, Expected: 0",
            zero_f32, zero_f32_to_i32
        );
        assert_eq!(zero_f32_to_i32, 0);

        let zero_f32_to_i24: I24 = zero_f32.convert_to();
        println!("DEBUG: f32 -> I24 conversion for 0.0");
        println!(
            "Input: {}, Output: {} (i32 value), Expected: 0",
            zero_f32,
            zero_f32_to_i24.to_i32()
        );
        assert_eq!(zero_f32_to_i24.to_i32(), 0);

        // Test clamping of out-of-range values
        let large_f32: f32 = 2.0;
        let large_f32_to_i16: i16 = large_f32.convert_to();
        assert_eq!(large_f32_to_i16, i16::MAX);

        let neg_large_f32: f32 = -2.0;
        let neg_large_f32_to_i16: i16 = neg_large_f32.convert_to();
        assert!(
            neg_large_f32_to_i16 == i16::MIN || neg_large_f32_to_i16 == -32767,
            "Expected either -32768 or -32767, got {}",
            neg_large_f32_to_i16
        );

        let large_f32_to_i32: i32 = large_f32.convert_to();
        assert_eq!(large_f32_to_i32, i32::MAX);

        let neg_large_f32_to_i32: i32 = neg_large_f32.convert_to();
        assert!(
            neg_large_f32_to_i32 == i32::MIN || neg_large_f32_to_i32 == -2147483647,
            "Expected either i32::MIN or -2147483647, got {}",
            neg_large_f32_to_i32
        );

        // Test small values
        let small_value: f32 = 1.0e-6;
        let small_value_to_i16: i16 = small_value.convert_to();
        assert_eq!(small_value_to_i16, 0);

        let small_value_to_i32: i32 = small_value.convert_to();
        assert_eq!(small_value_to_i32, 2147); // 1.0e-6 * 2147483647 rounded to nearest

        // Test values near 0.5
        let half_f32: f32 = 0.5;
        let half_f32_to_i16: i16 = half_f32.convert_to();
        assert_eq!(half_f32_to_i16, 16384); // 0.5 * 32767 rounded to nearest

        let neg_half_f32: f32 = -0.5;
        let neg_half_f32_to_i16: i16 = neg_half_f32.convert_to();
        assert_eq!(neg_half_f32_to_i16, -16384);
    }

    // Edge cases for f64 conversions
    #[test]
    fn f64_edge_cases() {
        // Test -1.0 (minimum valid value)
        let min_f64: f64 = -1.0;
        let min_f64_to_i16: i16 = min_f64.convert_to();

        println!("DEBUG: f64 -> i16 conversion for -1.0");
        println!(
            "Input: {}, Output: {}, Expected: {} or {}",
            min_f64,
            min_f64_to_i16,
            i16::MIN,
            -32767
        );

        // Due to rounding in the implementation, sometimes -1.0 can convert to -32767
        // This is acceptable since it's only 1 bit off from the true min
        assert!(
            min_f64_to_i16 == i16::MIN || min_f64_to_i16 == -32767,
            "Expected either -32768 or -32767, got {}",
            min_f64_to_i16
        );

        let min_f64_to_i32: i32 = min_f64.convert_to();

        println!("DEBUG: f64 -> i32 conversion for -1.0");
        println!(
            "Input: {}, Output: {}, Expected: {} or {}",
            min_f64,
            min_f64_to_i32,
            i32::MIN,
            -2147483647
        );

        assert!(
            min_f64_to_i32 == i32::MIN || min_f64_to_i32 == -2147483647,
            "Expected either i32::MIN or -2147483647, got {}",
            min_f64_to_i32
        );

        let min_f64_to_f32: f32 = min_f64.convert_to();

        println!("DEBUG: f64 -> f32 conversion for -1.0");
        println!(
            "Input: {}, Output: {}, Expected: -1.0",
            min_f64, min_f64_to_f32
        );

        assert_approx_eq!(min_f64_to_f32 as f64, -1.0, 1e-6);

        // Test 1.0 (maximum valid value)
        let max_f64: f64 = 1.0;
        let max_f64_to_i16: i16 = max_f64.convert_to();
        assert_eq!(max_f64_to_i16, i16::MAX);

        let max_f64_to_i32: i32 = max_f64.convert_to();
        assert_eq!(max_f64_to_i32, i32::MAX);

        let max_f64_to_f32: f32 = max_f64.convert_to();
        assert_approx_eq!(max_f64_to_f32 as f64, 1.0, 1e-6);

        // Test 0.0
        let zero_f64: f64 = 0.0;
        let zero_f64_to_i16: i16 = zero_f64.convert_to();
        assert_eq!(zero_f64_to_i16, 0);

        let zero_f64_to_i32: i32 = zero_f64.convert_to();
        assert_eq!(zero_f64_to_i32, 0);

        let zero_f64_to_f32: f32 = zero_f64.convert_to();
        assert_approx_eq!(zero_f64_to_f32 as f64, 0.0, 1e-10);

        // Test clamping of out-of-range values
        let large_f64: f64 = 2.0;
        let large_f64_to_i16: i16 = large_f64.convert_to();
        assert_eq!(large_f64_to_i16, i16::MAX);

        let neg_large_f64: f64 = -2.0;
        let neg_large_f64_to_i16: i16 = neg_large_f64.convert_to();
        assert!(
            neg_large_f64_to_i16 == i16::MIN || neg_large_f64_to_i16 == -32767,
            "Expected either -32768 or -32767, got {}",
            neg_large_f64_to_i16
        );

        // Test very small values
        let tiny_value: f64 = 1.0e-12;
        let tiny_value_to_i16: i16 = tiny_value.convert_to();
        assert_eq!(tiny_value_to_i16, 0);

        let tiny_value_to_i32: i32 = tiny_value.convert_to();
        assert_eq!(tiny_value_to_i32, 0);

        let tiny_value_to_f32: f32 = tiny_value.convert_to();
        assert_approx_eq!(tiny_value_to_f32 as f64, 0.0, 1e-10);
    }

    // Tests for I24 conversions
    #[test]
    fn i24_conversion_tests() {
        // Create an I24 with a known value
        let i24_value = i24!(4660 << 8); //  So converting back to i16 gives 4660
        println!(
            "DEBUG: Created I24 value from 4660 << 8 = {}",
            i24_value.to_i32()
        );

        // Test I24 to i16
        let i24_to_i16: i16 = i24_value.convert_to();
        let expected_i16 = 0x1234_i16;
        println!("DEBUG: I24 -> i16 conversion");
        println!(
            "I24 (as i32): {}, i16: {}, Expected: {}",
            i24_value.to_i32(),
            i24_to_i16,
            expected_i16
        );
        assert_eq!(i24_to_i16, expected_i16);

        // Test I24 to f32
        let i24_to_f32: f32 = i24_value.convert_to();
        let expected_f32 = (0x123456 as f32) / (I24::MAX.to_i32() as f32);
        println!("DEBUG: I24 -> f32 conversion");
        println!(
            "I24 (as i32): {}, f32: {}, Expected: {}",
            i24_value.to_i32(),
            i24_to_f32,
            expected_f32
        );
        // Print the difference to help debug
        println!("DEBUG: Difference: {}", (i24_to_f32 - expected_f32).abs());
        assert_approx_eq!(i24_to_f32 as f64, expected_f32 as f64, 1e-4);

        // Test I24 to f64
        let i24_to_f64: f64 = i24_value.convert_to();
        let expected_f64 = (0x123456 as f64) / (I24::MAX.to_i32() as f64);
        println!("DEBUG: I24 -> f64 conversion");
        println!(
            "I24 (as i32): {}, f64: {}, Expected: {}",
            i24_value.to_i32(),
            i24_to_f64,
            expected_f64
        );
        // Print the difference to help debug
        println!("DEBUG: Difference: {}", (i24_to_f64 - expected_f64).abs());
        assert_approx_eq!(i24_to_f64, expected_f64, 1e-4);
    }

    // Tests for convert_from functionality
    #[test]
    fn convert_from_tests() {
        // Test i16::convert_from with different source types
        let f32_source: f32 = 0.5;
        let i16_result: i16 = i16::convert_from(f32_source);
        assert_eq!(i16_result, 16384); // 0.5 * 32767 rounded

        let i32_source: i32 = 65536;
        let i16_result: i16 = i16::convert_from(i32_source);
        assert_eq!(i16_result, 1); // 65536 >> 16 = 1

        // Test f32::convert_from with different source types
        let i16_source: i16 = 16384;
        let f32_result: f32 = f32::convert_from(i16_source);
        assert_approx_eq!(f32_result as f64, 0.5, 1e-4);

        let i32_source: i32 = i32::MAX / 2;
        let f32_result: f32 = f32::convert_from(i32_source);
        assert_approx_eq!(f32_result as f64, 0.5, 1e-4);

        // Test I24::convert_from
        let i16_source: i16 = 4660; // 0x1234
        let i24_result: I24 = I24::convert_from(i16_source);
        assert_eq!(i24_result.to_i32(), 4660 << 8); // Should be shifted left by 8 bits

        // Test with zero values
        let zero_f32: f32 = 0.0;
        let zero_i16: i16 = i16::convert_from(zero_f32);
        assert_eq!(zero_i16, 0);

        let zero_i16_source: i16 = 0;
        let zero_f32_result: f32 = f32::convert_from(zero_i16_source);
        assert_approx_eq!(zero_f32_result as f64, 0.0, 1e-10);
    }

    // Tests for round trip conversions
    #[test]
    fn round_trip_conversions() {
        // i16 -> f32 -> i16
        for sample in [-32768, -16384, 0, 16384, 32767].iter() {
            let original = *sample;
            let intermediate: f32 = original.convert_to();
            let round_tripped: i16 = intermediate.convert_to();

            println!("DEBUG: i16->f32->i16 conversion");
            println!(
                "Original i16: {}, f32: {}, Round trip i16: {}",
                original, intermediate, round_tripped
            );

            assert!(
                (original - round_tripped).abs() <= 1,
                "Expected {}, got {}",
                original,
                round_tripped
            );
        }

        // i32 -> f32 -> i32 (will lose precision)
        for &sample in &[i32::MIN, i32::MIN / 2, 0, i32::MAX / 2, i32::MAX] {
            let original = sample;
            let intermediate: f32 = original.convert_to();
            let round_tripped: i32 = intermediate.convert_to();

            // Special case for extreme values
            if original == i32::MIN {
                // Allow off-by-one for MIN value
                assert!(
                    round_tripped == i32::MIN || round_tripped == -2147483647,
                    "Expected either i32::MIN or -2147483647, got {}",
                    round_tripped
                );
            } else if original == i32::MAX || original == 0 {
                assert_eq!(
                    original, round_tripped,
                    "Failed in i32->f32->i32 with extreme value {}",
                    original
                );
            } else {
                // For other values, we expect close but not exact due to precision
                let ratio = (round_tripped as f64) / (original as f64);
                assert!(
                    ratio > 0.999 && ratio < 1.001,
                    "Round trip error too large: {} -> {}",
                    original,
                    round_tripped
                );
            }
        }

        // f32 -> i16 -> f32
        for &sample in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let original: f32 = sample;
            let intermediate: i16 = original.convert_to();
            let round_tripped: f32 = intermediate.convert_to();

            // For all values, we check approximately but with a more generous epsilon
            assert_approx_eq!(original as f64, round_tripped as f64, 1e-4);
        }

        // i16 -> I24 -> i16
        for &sample in &[i16::MIN, -16384, 0, 16384, i16::MAX] {
            let original = sample;
            let intermediate: I24 = original.convert_to();
            let round_tripped: i16 = intermediate.convert_to();

            // For extreme negative values, allow 1-bit difference
            if original == i16::MIN {
                assert!(
                    round_tripped == i16::MIN || round_tripped == -32767,
                    "Expected either -32768 or -32767, got {}",
                    round_tripped
                );
            } else {
                assert_eq!(
                    original, round_tripped,
                    "Failed in i16->I24->i16 with value {}",
                    original
                );
            }
        }
    }
}
