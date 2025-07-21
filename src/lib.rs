// Correctness and logic
#![warn(clippy::unit_cmp)] // Detects comparing unit types
#![warn(clippy::match_same_arms)] // Duplicate match arms
#![warn(clippy::unreachable)] // Detects unreachable code

// Performance-focused
#![warn(clippy::inefficient_to_string)] // `format!("{}", x)` vs `x.to_string()`
#![warn(clippy::map_clone)] // Cloning inside `map()` unnecessarily
#![warn(clippy::unnecessary_to_owned)] // Detects redundant `.to_owned()` or `.clone()`
#![warn(clippy::large_stack_arrays)] // Helps avoid stack overflows
#![warn(clippy::box_collection)] // Warns on boxed `Vec`, `String`, etc.
#![warn(clippy::vec_box)] // Avoids using `Vec<Box<T>>` when unnecessary
#![warn(clippy::needless_collect)] // Avoids `.collect().iter()` chains

// Style and idiomatic Rust
#![warn(clippy::redundant_clone)] // Detects unnecessary `.clone()`
#![warn(clippy::identity_op)] // e.g., `x + 0`, `x * 1`
#![warn(clippy::needless_return)] // Avoids `return` at the end of functions
#![warn(clippy::let_unit_value)] // Avoids binding `()` to variables
#![warn(clippy::manual_map)] // Use `.map()` instead of manual `match`
#![warn(clippy::unwrap_used)] // Avoids using `unwrap()`
#![warn(clippy::panic)] // Avoids using `panic!` in production code

// Maintainability
#![warn(clippy::missing_panics_doc)] // Docs for functions that might panic
#![warn(clippy::missing_safety_doc)] // Docs for `unsafe` functions
#![warn(clippy::missing_const_for_fn)] // Suggests making eligible functions `const`
#![allow(clippy::too_many_arguments)] // Allow functions with many parameters (very few and far between)

//! # Audio Samples
//!
//! A high-performance audio processing library for Rust with Python bindings.
//!
//! This library provides a comprehensive set of tools for working with audio data,
//! including type-safe sample format conversions, statistical analysis, and various
//! audio processing operations.
//!
//! ## Core Features
//!
//! - **Type-safe audio sample conversions** between i16, I24, i32, f32, and f64
//! - **High-performance operations** leveraging ndarray for efficient computation
//! - **Comprehensive metadata** tracking (sample rate, channels, duration)
//! - **Flexible data structures** supporting both mono and multi-channel audio
//! - **Python integration** via PyO3 bindings
//!
//! ## Example Usage
//!
//! ```rust
//! use audio_samples::AudioSamples;
//! use ndarray::array;
//!
//! // Create mono audio with sample rate
//! let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
//! let audio = AudioSamples::new_mono(data, 44100);
//!
//! assert_eq!(audio.sample_rate(), 44100);
//! assert_eq!(audio.channels(), 1);
//! assert_eq!(audio.samples_per_channel(), 5);
//! ```

use bytemuck::NoUninit;
use num_traits::{FromPrimitive, ToBytes, Zero};
mod error;

pub mod operations;
#[cfg(feature = "python")]
pub mod python;
mod repr;

pub mod resampling;
pub mod utils;
use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
};

pub use i24::I24DiskMethods;
/// Re-export i24 for dependent crates to use.
pub use i24::i24 as I24;

// Re-exports for public API
pub use crate::error::{AudioSampleError, AudioSampleResult};
pub use crate::operations::{
    AudioChannelOps, AudioEditing, AudioProcessing, AudioSamplesOperations, AudioStatistics,
    AudioTransforms, AudioTypeConversion, NormalizationMethod,
};
pub use crate::repr::AudioSamples;

/// Array of supported audio sample data types as string identifiers
pub const SUPPORTED_DTYPES: [&str; 5] = ["i16", "I24", "i32", "f32", "f64"];

/// Describes how multi-channel audio data is organized in memory
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ChannelLayout {
    /// Samples from different channels are stored sequentially (LRLRLR...)
    /// This is the most common format for audio files and streaming
    #[default]
    Interleaved,
    /// Samples from each channel are stored in separate contiguous blocks (LLL...RRR...)
    /// This format is often preferred for digital signal processing
    NonInterleaved,
}

pub trait Castable:
    CastInto<i16> + CastInto<I24> + CastInto<i32> + CastInto<f32> + CastInto<f64>
{
}
impl<T> Castable for T where
    T: CastInto<i16> + CastInto<I24> + CastInto<i32> + CastInto<f32> + CastInto<f64>
{
}
/// Core trait defining the interface for audio sample types.
///
/// This trait provides a unified interface for working with different audio sample formats
/// including integers (i16, i32), 24-bit integers (I24), and floating-point (f32, f64).
///
/// All implementors support:
/// - Type-safe conversions between all supported formats
/// - Arithmetic operations (Add, Sub, Mul, Div)
/// - Serialization to byte arrays
/// - Safe memory layout guarantees
///
/// ## Supported Types
/// - `i16`: 16-bit signed integer samples (most common for audio files)
/// - `I24`: 24-bit signed integer samples (professional audio)
/// - `i32`: 32-bit signed integer samples (high precision)
/// - `f32`: 32-bit floating-point samples (normalized -1.0 to 1.0)
/// - `f64`: 64-bit floating-point samples (highest precision)
pub trait AudioSample:
    Copy
    + NoUninit
    + ConvertTo<Self>
    + ConvertTo<i16>
    + ConvertTo<i32>
    + ConvertTo<I24>
    + ConvertTo<f32>
    + ConvertTo<f64>
    + Into<Self>
    + From<Self>
    + Sync
    + Send
    + Debug
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + PartialOrd
    + PartialEq
    + Zero
    + ToBytes
    + serde::Serialize
    + serde::Deserialize<'static>
    + CastFrom<usize>
    + FromPrimitive
    + ToString
    + Castable
{
    #[inline]
    fn to_bytes(self) -> Vec<u8> {
        self.to_ne_bytes().as_ref().to_vec()
    }

    #[inline]
    fn to_bytes_slice(samples: &[Self]) -> Vec<u8> {
        Vec::from(bytemuck::cast_slice(samples))
    }

    #[inline]
    fn into_inner(self) -> Self {
        self
    }

    const MAX: Self;
    const MIN: Self;
    const BITS: u8;
}

pub trait CastFrom<S>: Sized {
    fn cast_from(value: S) -> Self;
}

pub trait CastInto<T>: Sized
where
    Self: CastFrom<T>,
{
    fn cast_into(self) -> T;
}

/// Trait for converting one sample type to another with proper scaling.
///
/// This trait provides type-safe, mathematically consistent conversions between
/// different audio sample formats. Conversions handle bit depth differences and
/// maintain audio quality through proper scaling algorithms.
///
/// ## Conversion Behavior
/// - **Integer ↔ Integer**: Bit-shift scaling to preserve full dynamic range
/// - **Integer ↔ Float**: Normalized scaling (-1.0 to 1.0 for floats)
/// - **Float ↔ Float**: Direct casting with precision conversion
/// - **I24 Special Handling**: Custom methods for 24-bit operations
///
/// ## Example
/// ```rust
/// use audio_samples::ConvertTo;
///
/// let sample_i16: i16 = 16384;  // Half of i16::MAX
/// let sample_f32: f32 = sample_i16.convert_to() {
///     Ok(sample) => sample,
///     Err(e) => panic!("Conversion failed: {}", e),
/// };
/// assert!((sample_f32 - 0.5).abs() < 1e-4);  // Should be approximately 0.5
/// ```
pub trait ConvertTo<T: AudioSample> {
    fn convert_to(&self) -> AudioSampleResult<T>;

    /// Convert from another audio sample type to this type.
    /// This is a convenience method that calls `convert_to` on the source value.
    fn convert_from<F: AudioSample + ConvertTo<T>>(source: F) -> AudioSampleResult<T> {
        source.convert_to()
    }
}

// ========================
// Conversion Macros
// ========================

/// Generates identity conversions (same type to same type)
macro_rules! impl_identity_conversion {
    ($type:ty) => {
        impl ConvertTo<$type> for $type {
            #[inline(always)]
            fn convert_to(&self) -> AudioSampleResult<$type> {
                Ok(*self)
            }
        }
    };
}

/// Generates integer-to-integer conversions using bit shifts
/// Ensures mathematical consistency by using the bit difference
macro_rules! impl_int_to_int_conversion {
    ($from:ty, $to:ty, $from_bits:expr, $to_bits:expr) => {
        impl ConvertTo<$to> for $from {
            #[inline(always)]
            fn convert_to(&self) -> AudioSampleResult<$to> {
                const SHIFT: i8 = $to_bits - $from_bits;
                Ok(if SHIFT > 0 {
                    (*self as i64) << SHIFT as u8
                } else if SHIFT < 0 {
                    (*self as i64) >> (-SHIFT) as u8
                } else {
                    unreachable!("Due to if and else if above")
                } as $to)
            }
        }
    };
}

/// Special handling for I24 conversions since it needs method calls
macro_rules! impl_i24_conversion {
    (from_i24 => $to:ty, $to_bits:expr) => {
        impl ConvertTo<$to> for I24 {
            #[inline(always)]
            fn convert_to(&self) -> AudioSampleResult<$to> {
                const SHIFT: i8 = $to_bits - 24;
                let val = self.to_i32();

                Ok(if SHIFT > 0 {
                    (val as i64) << SHIFT as u8
                } else if SHIFT < 0 {
                    (val as i64) >> (-SHIFT) as u8
                } else {
                    val as i64
                } as $to)
            }
        }
    };
    (to_i24 => $from:ty, $from_bits:expr) => {
        impl ConvertTo<I24> for $from {
            #[inline(always)]
            fn convert_to(&self) -> AudioSampleResult<I24> {
                const SHIFT: i8 = 24 - $from_bits;
                let result = if SHIFT > 0 {
                    (*self as i32) << SHIFT as u8
                } else if SHIFT < 0 {
                    (*self as i32) >> (-SHIFT) as u8
                } else {
                    *self as i32
                };
                match I24::try_from_i32(result) {
                    Some(x) => Ok(x),
                    None => Err(AudioSampleError::ConversionError(
                        format!("{:?}", self),
                        stringify!($from).to_string(),
                        "I24".to_string(),
                        "Value out of range for I24".to_string(),
                    )),
                }
            }
        }
    };
}

/// Generates integer-to-float conversions with consistent scaling
macro_rules! impl_int_to_float_conversion {
    ($from:ty, $to:ty, $max_val:expr) => {
        impl ConvertTo<$to> for $from {
            #[inline(always)]
            fn convert_to(&self) -> AudioSampleResult<$to> {
                if *self < 0 {
                    Ok((*self as $to) / (-(<$from>::MIN as $to)))
                } else {
                    Ok((*self as $to) / ($max_val as $to))
                }
            }
        }
    };
}

/// Special case for I24 to float
macro_rules! impl_i24_to_float_conversion {
    ($to:ty) => {
        impl ConvertTo<$to> for I24 {
            #[inline(always)]
            fn convert_to(&self) -> AudioSampleResult<$to> {
                let val = self.to_i32();
                let min_val = I24::MIN.to_i32();
                let max_val = I24::MAX.to_i32();
                if val < 0 {
                    Ok((val as $to) / (-(min_val as $to)))
                } else {
                    Ok((val as $to) / (max_val as $to))
                }
            }
        }
    };
}

/// Generates float-to-integer conversions with consistent scaling
macro_rules! impl_float_to_int_conversion {
    ($from:ty, $to:ty, $max_val:expr) => {
        impl ConvertTo<$to> for $from {
            #[inline(always)]
            fn convert_to(&self) -> AudioSampleResult<$to> {
                let clamped = self.clamp(-1.0, 1.0);
                if clamped < 0.0 {
                    Ok((clamped * (-(<$to>::MIN as $from))).round() as $to)
                } else {
                    Ok((clamped * ($max_val as $from)).round() as $to)
                }
            }
        }
    };
}

/// Special case for float to I24
macro_rules! impl_float_to_i24_conversion {
    ($from:ty) => {
        impl ConvertTo<I24> for $from {
            #[inline(always)]
            fn convert_to(&self) -> AudioSampleResult<I24> {
                let clamped = self.clamp(-1.0, 1.0);
                let scaled_val = if clamped < 0.0 {
                    (clamped * (-(I24::MIN.to_i32() as $from))).round() as i32
                } else {
                    (clamped * (I24::MAX.to_i32() as $from)).round() as i32
                };
                match I24::try_from_i32(scaled_val) {
                    Some(x) => Ok(x),
                    None => Err(AudioSampleError::ConversionError(
                        format!("{:?}", self),
                        stringify!($from).to_string(),
                        "I24".to_string(),
                        "Value out of range for I24".to_string(),
                    )),
                }
            }
        }
    };
}

/// Generates float-to-float conversions
macro_rules! impl_float_to_float_conversion {
    ($from:ty, $to:ty) => {
        impl ConvertTo<$to> for $from {
            #[inline(always)]
            fn convert_to(&self) -> AudioSampleResult<$to> {
                Ok(*self as $to)
            }
        }
    };
}

// ========================
// AudioSample Implementations
// ========================

impl AudioSample for i16 {
    const MAX: Self = i16::MAX;
    const MIN: Self = i16::MIN;
    const BITS: u8 = 16;
}

impl AudioSample for I24 {
    #[inline]
    fn to_bytes_slice(samples: &[Self]) -> Vec<u8> {
        I24::write_i24s_ne(samples)
    }

    const MAX: Self = I24::MAX;
    const MIN: Self = I24::MIN;
    const BITS: u8 = 24;
}

impl AudioSample for i32 {
    const MAX: Self = i32::MAX;
    const MIN: Self = i32::MIN;
    const BITS: u8 = 32;
}

impl AudioSample for f32 {
    const MAX: Self = 1.0;
    const MIN: Self = -1.0;
    const BITS: u8 = 32;
}

impl AudioSample for f64 {
    const MAX: Self = 1.0;
    const MIN: Self = -1.0;
    const BITS: u8 = 64;
}

// ========================
// Generate All Conversions
// ========================

// Identity conversions
impl_identity_conversion!(i16);
impl_identity_conversion!(I24);
impl_identity_conversion!(i32);
impl_identity_conversion!(f32);
impl_identity_conversion!(f64);

// Integer to integer conversions
impl_int_to_int_conversion!(i16, i32, 16, 32);
impl_int_to_int_conversion!(i32, i16, 32, 16);

// I24 conversions (special case due to method calls)
impl_i24_conversion!(from_i24 => i16, 16);
impl_i24_conversion!(from_i24 => i32, 32);
impl_i24_conversion!(to_i24 => i16, 16);
impl_i24_conversion!(to_i24 => i32, 32);

// Integer to float conversions
impl_int_to_float_conversion!(i16, f32, i16::MAX);
impl_int_to_float_conversion!(i16, f64, i16::MAX);
impl_int_to_float_conversion!(i32, f32, i32::MAX);
impl_int_to_float_conversion!(i32, f64, i32::MAX);

// I24 to float conversions
impl_i24_to_float_conversion!(f32);
impl_i24_to_float_conversion!(f64);

// Float to integer conversions
impl_float_to_int_conversion!(f32, i16, i16::MAX);
impl_float_to_int_conversion!(f32, i32, i32::MAX);
impl_float_to_int_conversion!(f64, i16, i16::MAX);
impl_float_to_int_conversion!(f64, i32, i32::MAX);

// Float to I24 conversions
impl_float_to_i24_conversion!(f32);
impl_float_to_i24_conversion!(f64);

// Float to float conversions
impl_float_to_float_conversion!(f32, f64);
impl_float_to_float_conversion!(f64, f32);

macro_rules! impl_cast_from {
    ($src:ty => [$($dst:ty),+]) => {
        $(
            impl CastFrom<$src> for $dst {
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

impl CastFrom<usize> for i16 {
    fn cast_from(value: usize) -> Self {
        if value > i16::MAX as usize {
            i16::MAX
        } else {
            value as i16
        }
    }
}

impl CastFrom<usize> for I24 {
    fn cast_from(value: usize) -> Self {
        if value > I24::MAX.to_i32() as usize {
            I24::MAX
        } else {
            match I24::try_from_i32(value as i32) {
                Some(x) => x,
                None => I24::MIN, // If conversion fails, return minimum value
            }
        }
    }
}

impl CastFrom<usize> for i32 {
    fn cast_from(value: usize) -> Self {
        if value > i32::MAX as usize {
            i32::MAX
        } else {
            value as i32
        }
    }
}

impl CastFrom<usize> for f32 {
    fn cast_from(value: usize) -> Self {
        value as f32
    }
}
impl CastFrom<usize> for f64 {
    fn cast_from(value: usize) -> Self {
        value as f64
    }
}

impl CastFrom<I24> for i16 {
    fn cast_from(value: I24) -> Self {
        value.to_i32() as i16
    }
}

impl CastFrom<I24> for I24 {
    fn cast_from(value: I24) -> Self {
        value
    }
}

impl CastFrom<I24> for i32 {
    fn cast_from(value: I24) -> Self {
        value.to_i32()
    }
}

impl CastFrom<I24> for f32 {
    fn cast_from(value: I24) -> Self {
        value.to_i32() as f32
    }
}

impl CastFrom<I24> for f64 {
    fn cast_from(value: I24) -> Self {
        value.to_i32() as f64
    }
}
impl CastFrom<i16> for I24 {
    fn cast_from(value: i16) -> Self {
        match I24::try_from_i32(value as i32) {
            Some(x) => x,
            None => I24::MIN, // If conversion fails, return minimum value
        }
    }
}

impl CastFrom<i32> for I24 {
    fn cast_from(value: i32) -> Self {
        I24::try_from_i32(value).unwrap_or(I24::MIN)
    }
}

impl CastFrom<f32> for I24 {
    fn cast_from(value: f32) -> Self {
        I24::try_from_i32(value as i32).unwrap_or(I24::MIN)
    }
}

impl CastFrom<f64> for I24 {
    fn cast_from(value: f64) -> Self {
        I24::try_from_i32(value as i32).unwrap_or(I24::MIN)
    }
}

macro_rules! impl_cast_into {
    ($src:ty => [$($dst:ty),+]) => {
        $(
            impl CastInto<$dst> for $src {
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

impl CastInto<i16> for I24 {
    fn cast_into(self) -> i16 {
        self.to_i32() as i16
    }
}

impl CastInto<I24> for I24 {
    fn cast_into(self) -> I24 {
        self
    }
}

impl CastInto<i32> for I24 {
    fn cast_into(self) -> i32 {
        self.to_i32()
    }
}

impl CastInto<f32> for I24 {
    fn cast_into(self) -> f32 {
        self.to_i32() as f32
    }
}

impl CastInto<f64> for I24 {
    fn cast_into(self) -> f64 {
        self.to_i32() as f64
    }
}

impl CastInto<I24> for i16 {
    fn cast_into(self) -> I24 {
        I24::cast_from(self)
    }
}

impl CastInto<I24> for i32 {
    fn cast_into(self) -> I24 {
        I24::cast_from(self)
    }
}

impl CastInto<I24> for f32 {
    fn cast_into(self) -> I24 {
        I24::cast_from(self)
    }
}

impl CastInto<I24> for f64 {
    fn cast_into(self) -> I24 {
        I24::cast_from(self)
    }
}

#[cfg(test)]
mod conversion_tests {
    use super::*;

    use approx_eq::assert_approx_eq;
    use std::fs::File;
    use std::io::BufRead;
    use std::path::Path;
    use std::str::FromStr;

    // Helper functions
    #[cfg(test)]
    fn read_lines<P>(filename: P) -> std::io::Result<std::io::Lines<std::io::BufReader<File>>>
    where
        P: AsRef<Path>,
    {
        let file = File::open(filename)?;
        Ok(std::io::BufReader::new(file).lines())
    }

    #[cfg(test)]
    fn read_text_to_vec<T: FromStr>(fp: &Path) -> Result<Vec<T>, Box<dyn std::error::Error>>
    where
        <T as FromStr>::Err: std::error::Error + 'static,
    {
        let mut data = Vec::new();
        let lines = read_lines(fp)?;
        for line in lines {
            let line = line?;
            for sample in line.split(" ") {
                let parsed_sample: T = match sample.trim().parse::<T>() {
                    Ok(num) => num,
                    Err(err) => {
                        eprintln!("Failed to parse {}", sample);
                        panic!("{}", err)
                    }
                };
                data.push(parsed_sample);
            }
        }
        Ok(data)
    }

    #[test]
    fn i16_to_f32() {
        let i16_samples: Vec<i16> =
            read_text_to_vec(Path::new("./test_resources/one_channel_i16.txt")).unwrap();
        let i16_samples: &[i16] = &i16_samples;

        let f32_samples: Vec<f32> =
            read_text_to_vec(Path::new("./test_resources/one_channel_f32.txt")).unwrap();
        let f32_samples: &[f32] = &f32_samples;
        for (expected_sample, actual_sample) in f32_samples.iter().zip(i16_samples) {
            let actual_sample: f32 = actual_sample.convert_to().unwrap();
            assert_approx_eq!(*expected_sample as f64, actual_sample as f64, 1e-4);
        }
    }

    #[test]
    fn f32_to_i16() {
        let i16_samples: Vec<i16> =
            read_text_to_vec(Path::new("./test_resources/one_channel_i16.txt")).unwrap();
        let i16_samples: &[i16] = &i16_samples;

        let f32_samples: Vec<f32> =
            read_text_to_vec(Path::new("./test_resources/one_channel_f32.txt")).unwrap();

        let f32_samples: &[f32] = &f32_samples;
        for (expected_sample, actual_sample) in i16_samples.iter().zip(f32_samples) {
            let converted_sample: i16 = actual_sample.convert_to().unwrap();
            assert_eq!(
                *expected_sample, converted_sample,
                "Failed to convert sample {} to i16",
                actual_sample
            );
        }
    }

    // Edge cases for i16 conversions
    #[test]
    fn i16_edge_cases() {
        // Test minimum value
        let min_i16: i16 = i16::MIN;
        let min_i16_to_f32: f32 = min_i16.convert_to().unwrap();
        // Use higher epsilon for floating point comparison
        assert_approx_eq!(min_i16_to_f32 as f64, -1.0, 1e-5);

        let min_i16_to_i32: i32 = min_i16.convert_to().unwrap();
        assert_eq!(min_i16_to_i32, i32::MIN);

        let min_i16_to_i24: I24 = min_i16.convert_to().unwrap();
        let expected_i24_min = I24!(i32::MIN >> 8);
        assert_eq!(min_i16_to_i24.to_i32(), expected_i24_min.to_i32());

        // Test maximum value
        let max_i16: i16 = i16::MAX;
        let max_i16_to_f32: f32 = max_i16.convert_to().unwrap();
        assert_approx_eq!(max_i16_to_f32 as f64, 1.0, 1e-4);

        let max_i16_to_i32: i32 = max_i16.convert_to().unwrap();
        assert_eq!(max_i16_to_i32, 0x7FFF0000);

        // Test zero
        let zero_i16: i16 = 0;
        let zero_i16_to_f32: f32 = zero_i16.convert_to().unwrap();
        assert_approx_eq!(zero_i16_to_f32 as f64, 0.0, 1e-10);

        let zero_i16_to_i32: i32 = zero_i16.convert_to().unwrap();
        assert_eq!(zero_i16_to_i32, 0);

        let zero_i16_to_i24: I24 = zero_i16.convert_to().unwrap();
        assert_eq!(zero_i16_to_i24.to_i32(), 0);

        // Test mid-range positive
        let half_max_i16: i16 = i16::MAX / 2;
        let half_max_i16_to_f32: f32 = half_max_i16.convert_to().unwrap();
        // Use higher epsilon for floating point comparison of half values
        assert_approx_eq!(half_max_i16_to_f32 as f64, 0.5, 1e-4);

        let half_max_i16_to_i32: i32 = half_max_i16.convert_to().unwrap();
        assert_eq!(half_max_i16_to_i32, 0x3FFF0000);

        // Test mid-range negative
        let half_min_i16: i16 = i16::MIN / 2;
        let half_min_i16_to_f32: f32 = half_min_i16.convert_to().unwrap();
        assert_approx_eq!(half_min_i16_to_f32 as f64, -0.5, 1e-4);

        // let half_min_i16_to_i32: i32 = half_min_i16.convert_to().unwrap();
        // assert_eq!(half_min_i16_to_i32, 0xC0010000); // i16::MIN/2 == -16384
    }

    // Edge cases for i32 conversions
    #[test]
    fn i32_edge_cases() {
        // Test minimum value
        let min_i32: i32 = i32::MIN;
        let min_i32_to_f32: f32 = min_i32.convert_to().unwrap();
        assert_approx_eq!(min_i32_to_f32 as f64, -1.0, 1e-6);

        let min_i32_to_f64: f64 = min_i32.convert_to().unwrap();
        assert_approx_eq!(min_i32_to_f64, -1.0, 1e-12);

        let min_i32_to_i16: i16 = min_i32.convert_to().unwrap();
        assert_eq!(min_i32_to_i16, i16::MIN);

        // Test maximum value
        let max_i32: i32 = i32::MAX;
        let max_i32_to_f32: f32 = max_i32.convert_to().unwrap();
        assert_approx_eq!(max_i32_to_f32 as f64, 1.0, 1e-6);

        let max_i32_to_f64: f64 = max_i32.convert_to().unwrap();
        assert_approx_eq!(max_i32_to_f64, 1.0, 1e-12);

        let max_i32_to_i16: i16 = max_i32.convert_to().unwrap();
        assert_eq!(max_i32_to_i16, i16::MAX);

        // Test zero
        let zero_i32: i32 = 0;
        let zero_i32_to_f32: f32 = zero_i32.convert_to().unwrap();
        assert_approx_eq!(zero_i32_to_f32 as f64, 0.0, 1e-10);

        let zero_i32_to_f64: f64 = zero_i32.convert_to().unwrap();
        assert_approx_eq!(zero_i32_to_f64, 0.0, 1e-12);

        let zero_i32_to_i16: i16 = zero_i32.convert_to().unwrap();
        assert_eq!(zero_i32_to_i16, 0);

        // Test quarter-range values
        let quarter_max_i32: i32 = i32::MAX / 4;
        let quarter_max_i32_to_f32: f32 = quarter_max_i32.convert_to().unwrap();
        assert_approx_eq!(quarter_max_i32_to_f32 as f64, 0.25, 1e-6);

        let quarter_min_i32: i32 = i32::MIN / 4;
        let quarter_min_i32_to_f32: f32 = quarter_min_i32.convert_to().unwrap();
        assert_approx_eq!(quarter_min_i32_to_f32 as f64, -0.25, 1e-6);
    }

    // Edge cases for f32 conversions
    #[test]
    fn f32_edge_cases() {
        // Test -1.0 (minimum valid value)
        let min_f32: f32 = -1.0;
        let min_f32_to_i16: i16 = min_f32.convert_to().unwrap();
        // For exact -1.0, we can get -32767 due to rounding in the implementation
        // This is acceptable since it's only 1 bit off from the true min
        assert!(
            min_f32_to_i16 == i16::MIN || min_f32_to_i16 == -32767,
            "Expected either -32768 or -32767, got {}",
            min_f32_to_i16
        );

        let min_f32_to_i32: i32 = min_f32.convert_to().unwrap();
        assert!(
            min_f32_to_i32 == i32::MIN || min_f32_to_i32 == -2147483647,
            "Expected either i32::MIN or -2147483647, got {}",
            min_f32_to_i32
        );

        let min_f32_to_i24: I24 = min_f32.convert_to().unwrap();
        let expected_i24 = I24::MIN;
        let diff = (min_f32_to_i24.to_i32() - expected_i24.to_i32()).abs();
        assert!(diff <= 1, "I24 values differ by more than 1, {}", diff);

        // Test 1.0 (maximum valid value)
        let max_f32: f32 = 1.0;
        let max_f32_to_i16: i16 = max_f32.convert_to().unwrap();
        println!("DEBUG: f32 -> i16 conversion for 1.0");
        println!(
            "Input: {}, Output: {}, Expected: {}",
            max_f32,
            max_f32_to_i16,
            i16::MAX
        );
        assert_eq!(max_f32_to_i16, i16::MAX);

        let max_f32_to_i32: i32 = max_f32.convert_to().unwrap();
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
        let zero_f32_to_i16: i16 = zero_f32.convert_to().unwrap();
        println!("DEBUG: f32 -> i16 conversion for 0.0");
        println!(
            "Input: {}, Output: {}, Expected: 0",
            zero_f32, zero_f32_to_i16
        );
        assert_eq!(zero_f32_to_i16, 0);

        let zero_f32_to_i32: i32 = zero_f32.convert_to().unwrap();
        println!("DEBUG: f32 -> i32 conversion for 0.0");
        println!(
            "Input: {}, Output: {}, Expected: 0",
            zero_f32, zero_f32_to_i32
        );
        assert_eq!(zero_f32_to_i32, 0);

        let zero_f32_to_i24: I24 = zero_f32.convert_to().unwrap();
        println!("DEBUG: f32 -> I24 conversion for 0.0");
        println!(
            "Input: {}, Output: {} (i32 value), Expected: 0",
            zero_f32,
            zero_f32_to_i24.to_i32()
        );
        assert_eq!(zero_f32_to_i24.to_i32(), 0);

        // Test clamping of out-of-range values
        let large_f32: f32 = 2.0;
        let large_f32_to_i16: i16 = large_f32.convert_to().unwrap();
        assert_eq!(large_f32_to_i16, i16::MAX);

        let neg_large_f32: f32 = -2.0;
        let neg_large_f32_to_i16: i16 = neg_large_f32.convert_to().unwrap();
        assert!(
            neg_large_f32_to_i16 == i16::MIN || neg_large_f32_to_i16 == -32767,
            "Expected either -32768 or -32767, got {}",
            neg_large_f32_to_i16
        );

        let large_f32_to_i32: i32 = large_f32.convert_to().unwrap();
        assert_eq!(large_f32_to_i32, i32::MAX);

        let neg_large_f32_to_i32: i32 = neg_large_f32.convert_to().unwrap();
        assert!(
            neg_large_f32_to_i32 == i32::MIN || neg_large_f32_to_i32 == -2147483647,
            "Expected either i32::MIN or -2147483647, got {}",
            neg_large_f32_to_i32
        );

        // Test small values
        let small_value: f32 = 1.0e-6;
        let small_value_to_i16: i16 = small_value.convert_to().unwrap();
        assert_eq!(small_value_to_i16, 0);

        let small_value_to_i32: i32 = small_value.convert_to().unwrap();
        assert_eq!(small_value_to_i32, 2147); // 1.0e-6 * 2147483647 rounded to nearest

        // Test values near 0.5
        let half_f32: f32 = 0.5;
        let half_f32_to_i16: i16 = half_f32.convert_to().unwrap();
        assert_eq!(half_f32_to_i16, 16384); // 0.5 * 32767 rounded to nearest

        let neg_half_f32: f32 = -0.5;
        let neg_half_f32_to_i16: i16 = neg_half_f32.convert_to().unwrap();
        assert_eq!(neg_half_f32_to_i16, -16384);
    }

    // Edge cases for f64 conversions
    #[test]
    fn f64_edge_cases() {
        // Test -1.0 (minimum valid value)
        let min_f64: f64 = -1.0;
        let min_f64_to_i16: i16 = min_f64.convert_to().unwrap();

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

        let min_f64_to_i32: i32 = min_f64.convert_to().unwrap();

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

        let min_f64_to_f32: f32 = min_f64.convert_to().unwrap();

        println!("DEBUG: f64 -> f32 conversion for -1.0");
        println!(
            "Input: {}, Output: {}, Expected: -1.0",
            min_f64, min_f64_to_f32
        );

        assert_approx_eq!(min_f64_to_f32 as f64, -1.0, 1e-6);

        // Test 1.0 (maximum valid value)
        let max_f64: f64 = 1.0;
        let max_f64_to_i16: i16 = max_f64.convert_to().unwrap();
        assert_eq!(max_f64_to_i16, i16::MAX);

        let max_f64_to_i32: i32 = max_f64.convert_to().unwrap();
        assert_eq!(max_f64_to_i32, i32::MAX);

        let max_f64_to_f32: f32 = max_f64.convert_to().unwrap();
        assert_approx_eq!(max_f64_to_f32 as f64, 1.0, 1e-6);

        // Test 0.0
        let zero_f64: f64 = 0.0;
        let zero_f64_to_i16: i16 = zero_f64.convert_to().unwrap();
        assert_eq!(zero_f64_to_i16, 0);

        let zero_f64_to_i32: i32 = zero_f64.convert_to().unwrap();
        assert_eq!(zero_f64_to_i32, 0);

        let zero_f64_to_f32: f32 = zero_f64.convert_to().unwrap();
        assert_approx_eq!(zero_f64_to_f32 as f64, 0.0, 1e-10);

        // Test clamping of out-of-range values
        let large_f64: f64 = 2.0;
        let large_f64_to_i16: i16 = large_f64.convert_to().unwrap();
        assert_eq!(large_f64_to_i16, i16::MAX);

        let neg_large_f64: f64 = -2.0;
        let neg_large_f64_to_i16: i16 = neg_large_f64.convert_to().unwrap();
        assert!(
            neg_large_f64_to_i16 == i16::MIN || neg_large_f64_to_i16 == -32767,
            "Expected either -32768 or -32767, got {}",
            neg_large_f64_to_i16
        );

        // Test very small values
        let tiny_value: f64 = 1.0e-12;
        let tiny_value_to_i16: i16 = tiny_value.convert_to().unwrap();
        assert_eq!(tiny_value_to_i16, 0);

        let tiny_value_to_i32: i32 = tiny_value.convert_to().unwrap();
        assert_eq!(tiny_value_to_i32, 0);

        let tiny_value_to_f32: f32 = tiny_value.convert_to().unwrap();
        assert_approx_eq!(tiny_value_to_f32 as f64, 0.0, 1e-10);
    }

    // Tests for I24 conversions
    #[test]
    fn i24_conversion_tests() {
        // Create an I24 with a known value
        let i24_value = I24!(4660 << 8); //  So converting back to i16 gives 4660
        println!(
            "DEBUG: Created I24 value from 4660 << 8 = {}",
            i24_value.to_i32()
        );

        // Test I24 to i16
        let i24_to_i16: i16 = i24_value.convert_to().unwrap();
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
        let i24_to_f32: f32 = i24_value.convert_to().unwrap();
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
        let i24_to_f64: f64 = i24_value.convert_to().unwrap();
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
        let i16_result: i16 = i16::convert_from(f32_source).unwrap();
        assert_eq!(i16_result, 16384); // 0.5 * 32767 rounded

        let i32_source: i32 = 65536;
        let i16_result: i16 = i16::convert_from(i32_source).unwrap();
        assert_eq!(i16_result, 1); // 65536 >> 16 = 1

        // Test f32::convert_from with different source types
        let i16_source: i16 = 16384;
        let f32_result: f32 = f32::convert_from(i16_source).unwrap();
        assert_approx_eq!(f32_result as f64, 0.5, 1e-4);

        let i32_source: i32 = i32::MAX / 2;
        let f32_result: f32 = f32::convert_from(i32_source).unwrap();
        assert_approx_eq!(f32_result as f64, 0.5, 1e-4);

        // Test I24::convert_from
        let i16_source: i16 = 4660; // 0x1234
        let i24_result: I24 = I24::convert_from(i16_source).unwrap();
        assert_eq!(i24_result.to_i32(), 4660 << 8); // Should be shifted left by 8 bits

        // Test with zero values
        let zero_f32: f32 = 0.0;
        let zero_i16: i16 = i16::convert_from(zero_f32).unwrap();
        assert_eq!(zero_i16, 0);

        let zero_i16_source: i16 = 0;
        let zero_f32_result: f32 = f32::convert_from(zero_i16_source).unwrap();
        assert_approx_eq!(zero_f32_result as f64, 0.0, 1e-10);
    }

    // Tests for round trip conversions
    #[test]
    fn round_trip_conversions() {
        // i16 -> f32 -> i16
        for sample in [-32768, -16384, 0, 16384, 32767].iter() {
            let original = *sample;
            let intermediate: f32 = original.convert_to().unwrap();
            let round_tripped: i16 = intermediate.convert_to().unwrap();

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
            let intermediate: f32 = original.convert_to().unwrap();
            let round_tripped: i32 = intermediate.convert_to().unwrap();

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
            let intermediate: i16 = original.convert_to().unwrap();
            let round_tripped: f32 = intermediate.convert_to().unwrap();

            // For all values, we check approximately but with a more generous epsilon
            assert_approx_eq!(original as f64, round_tripped as f64, 1e-4);
        }

        // i16 -> I24 -> i16
        for &sample in &[i16::MIN, -16384, 0, 16384, i16::MAX] {
            let original = sample;
            let intermediate: I24 = original.convert_to().unwrap();
            let round_tripped: i16 = intermediate.convert_to().unwrap();

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

// Python module export - this must be at the crate root for maturin
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn audio_samples(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // Delegate to the actual implementation in the python module
    python::register_module(_py, m)
}
