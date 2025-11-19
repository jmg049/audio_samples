//! # Audio Sample Type Conversions
//!
//! This module provides **high-performance, type-safe conversion operations**
//! between different audio sample representations for [`AudioSamples`].
//!
//! Conversions are exposed through the [`AudioTypeConversion`] trait,
//! which is automatically implemented for `AudioSamples<T>` where `T` is any supported sample type.
//! Users do not need to interact with the trait directly in most cases — the methods can be called
//! directly on an `AudioSamples` instance:
//!
//! ```rust
//! use audio_samples::AudioSamples;
//! use ndarray::array;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create mono audio data with f32 samples
//! let audio_f32 = AudioSamples::new_mono(array![0.5f32, -0.3, 0.8], 44_100);
//!
//! // Convert to i16
//! let audio_i16 = audio_f32.as_type::<i16>()?;
//!
//! // Consume and convert (avoids copying)
//! let audio_back = audio_i16.to_type::<f32>()?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Supported Conversions
//! Conversions are supported between the following types:
//!
//! - `i16`
//! - [`I24`] (24-bit PCM)
//! - `i32`
//! - `f32`
//! - `f64`
//!
//! Conversions use efficient **vectorised operations** internally (via `ndarray` and SIMD where
//! available) to minimise overhead. Extreme floating-point values are clamped when converting to
//! fixed-width integer formats to avoid overflow.
//!
//! ## Zero-Copy Conversion
//! For conversions between identical or layout-compatible types, the [`into_cow`](crate::AudioSamples::into_cow)
//! method can perform **zero-allocation conversions** using copy-on-write semantics:
//!
//! ```rust
//! use audio_samples::AudioSamples;
//! use ndarray::array;
//! use std::borrow::Cow;
//!
//! let audio_i32 = AudioSamples::new_mono(array![1i32, 2, 3], 44_100);
//! let cow = audio_i32.into_cow::<i32>().unwrap();
//!
//! match cow {
//!     Cow::Borrowed(_) => println!("Zero-copy conversion"),
//!     Cow::Owned(_) => println!("Conversion required allocation"),
//! }
//! ```
//!
//! ## Typical Use Cases
//! - Converting raw audio data from disk to a processing format (e.g. `i16` → `f32`)
//! - Preparing audio buffers for model input
//! - Reducing memory footprint after processing (`f64` → `i16`)
//! - Zero-copy conversions when the type is already correct
//!
//! ## Error Handling
//! Conversion methods return [`AudioSampleResult`], which can
//! contain an [`AudioSampleError`] if the conversion fails
//! (e.g. due to non-contiguous array layouts).
//!
//! ## API Summary
//! - [`as_type`](crate::AudioTypeConversion::as_type): Borrow and convert (copy)
//! - [`to_type`](crate::AudioTypeConversion::to_type): Consume and convert (no copy)
//! - [`cast_as`](crate::AudioTypeConversion::cast_as): Borrow and cast without safety checks
//! - [`cast_to`](crate::AudioTypeConversion::cast_to): Consume and cast without safety checks
//! - [`into_cow`](crate::AudioSamples::into_cow): Copy-on-write zero-copy conversion when possible
//!
//! These methods are available on all `AudioSamples` instances without additional imports.

use ndarray::{Array1, Array2};

use crate::repr::AudioData;
use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, CastFrom,
    ConvertTo, I24, LayoutError,
};

impl<'a, T: AudioSample> AudioTypeConversion<'a, T> for AudioSamples<'a, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
{
    /// Converts to different sample type, borrowing the original.
    ///
    /// Uses efficient vectorized operations via ndarray's mapv method
    /// combined with the existing ConvertTo trait system for type safety.
    ///
    /// # Example
    /// ```rust
    /// use audio_samples::{AudioSamples, operations::AudioTypeConversion};
    /// use ndarray::array;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let audio_f32 = AudioSamples::new_mono(array![0.5f32, -0.3, 0.8], 44100);
    /// let audio_i16 = audio_f32.as_type::<i16>()?;
    /// # Ok(())
    /// # }
    /// ```
    fn as_type<O>(&self) -> AudioSampleResult<AudioSamples<'static, O>>
    where
        T: ConvertTo<O>,
        O: AudioSample + ConvertTo<T>,
    {
        let sample_rate = self.sample_rate;
        match &self.data {
            AudioData::Mono(arr) => {
                let converted: ndarray::Array1<O> = arr
                    .iter()
                    .map(|s| s.convert_to())
                    .collect::<Result<Vec<_>, _>>()?
                    .into();
                Ok(AudioSamples::new_mono(converted, sample_rate))
            }
            AudioData::Multi(arr) => {
                let shape = arr.raw_dim();
                let converted_vec = arr
                    .iter()
                    .map(|s| s.convert_to())
                    .collect::<Result<Vec<_>, _>>()?;
                let converted =
                    ndarray::Array2::from_shape_vec(shape, converted_vec).map_err(|e| {
                        AudioSampleError::Layout(LayoutError::IncompatibleFormat {
                            operation: "array conversion".to_string(),
                            reason: e.to_string(),
                        })
                    })?;
                Ok(AudioSamples::new_multi_channel(converted, sample_rate))
            }
        }
    }

    /// Converts to different sample type, consuming the original.
    ///
    /// More efficient than as_type when the original is no longer needed
    /// since it avoids unnecessary copying.
    ///
    /// # Example
    /// ```rust
    /// use audio_samples::{AudioSamples, operations::AudioTypeConversion};
    /// use ndarray::array;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let audio_f32 = AudioSamples::new_mono(array![0.5f32, -0.3, 0.8], 44100);
    /// let audio_i16 = audio_f32.to_type::<i16>()?;
    /// // audio_f32 is no longer accessible here
    /// # Ok(())
    /// # }
    /// ```
    fn to_type<O: AudioSample + ConvertTo<T>>(self) -> AudioSampleResult<AudioSamples<'static, O>>
    where
        T: ConvertTo<O>,
    {
        let sample_rate = self.sample_rate;
        match self.data {
            AudioData::Mono(arr) => {
                let converted_data = arr
                    .iter()
                    .map(|sample| sample.convert_to())
                    .collect::<Result<Vec<O>, AudioSampleError>>()?;
                let converted_data: Array1<O> = Array1::from(converted_data);
                Ok(AudioSamples::new_mono(converted_data, sample_rate))
            }
            AudioData::Multi(arr) => {
                let shape: &[usize] = arr.shape();
                let converted_data = arr
                    .iter()
                    .map(|sample| sample.convert_to())
                    .collect::<Result<Vec<O>, AudioSampleError>>()?;

                let converted_data: Array2<O> =
                    Array2::from_shape_vec((shape[0], shape[1]), converted_data).map_err(|e| {
                        AudioSampleError::Layout(LayoutError::IncompatibleFormat {
                            operation: "array conversion".to_string(),
                            reason: e.to_string(),
                        })
                    })?;

                Ok(AudioSamples::new_multi_channel(converted_data, sample_rate))
            }
        }
    }

    fn cast_as<O>(&self) -> AudioSampleResult<AudioSamples<'static, O>>
    where
        O: AudioSample + CastFrom<T>,
    {
        Ok(self.map_into(|x| O::cast_from(x)))
    }

    fn cast_to<O>(self) -> AudioSampleResult<AudioSamples<'static, O>>
    where
        O: AudioSample + CastFrom<T>,
    {
        Ok(self.map_into(|x| O::cast_from(x)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;
    use ndarray::array;

    #[test]
    fn test_mono_f32_to_i16_conversion() {
        let data = array![0.5f32, -0.3, 0.0, 1.0, -1.0];
        let audio_f32 = AudioSamples::new_mono(data, 44100);

        let audio_i16 = audio_f32.as_type::<i16>().unwrap();

        assert_eq!(audio_i16.sample_rate(), 44100);
        assert_eq!(audio_i16.num_channels(), 1);
        assert_eq!(audio_i16.samples_per_channel(), 5);

        // Check specific conversions (allow for minor rounding differences)
        let converted_data = audio_i16;
        // 0.5 * 32767 can be either 16383 or 16384 depending on rounding
        assert!(
            (converted_data[0] - 16383).abs() <= 1,
            "Expected ~16383±1, got {}",
            converted_data[0]
        );
        assert_eq!(converted_data[2], 0); // 0.0 -> 0
        assert_eq!(converted_data[3], 32767); // 1.0 -> i16::MAX
        // -1.0 can map to either -32767 or -32768 depending on implementation
        assert!(
            converted_data[4] <= -32767,
            "Expected <= -32767, got {}",
            converted_data[4]
        );
    }

    #[test]
    fn test_multi_channel_i16_to_f32_conversion() {
        // Create 2 channels with 3 samples each: [[ch0_sample0, ch0_sample1, ch0_sample2], [ch1_sample0, ch1_sample1, ch1_sample2]]
        let data = array![[16384i16, 32767, 0], [-16384, -32768, 8192]];
        let audio_i16 = AudioSamples::new_multi_channel(data, 48000);

        let audio_f32 = audio_i16.as_type::<f32>().unwrap();
        assert_eq!(audio_f32.sample_rate(), 48000);
        assert_eq!(audio_f32.num_channels(), 2);
        assert_eq!(audio_f32.samples_per_channel(), 3);

        // Check specific conversions
        let converted_data = audio_f32;
        assert_approx_eq!(converted_data[[0, 0]] as f64, 0.5, 1e-4); // 16384/32767 ≈ 0.5
        assert_approx_eq!(converted_data[[0, 1]] as f64, 1.0, 1e-4); // 32767/32767 = 1.0
        assert_approx_eq!(converted_data[[0, 2]] as f64, 0.0, 1e-10); // 0/32767 = 0.0
        assert_approx_eq!(converted_data[[1, 0]] as f64, -0.5, 1e-4); // -16384/32768 ≈ -0.5
        assert_approx_eq!(converted_data[[1, 1]] as f64, -1.0, 1e-4); // -32768 conversion (asymmetric range)
        assert_approx_eq!(converted_data[[1, 2]] as f64, 0.25, 1e-4); // 8192/32768 = 0.25
    }

    #[test]
    fn test_consuming_conversion() {
        let data = array![0.1f32, 0.2, 0.3];
        let audio_f32 = AudioSamples::new_mono(data, 44100);

        // Test consuming conversion
        let audio_i16 = audio_f32.to_type::<i16>().unwrap();

        assert_eq!(audio_i16.num_channels(), 1);
        assert_eq!(audio_i16.samples_per_channel(), 3);

        // Verify conversion accuracy
        let converted_data = audio_i16;
        assert_eq!(converted_data[0], 3277); // 0.1 * 32767 ≈ 3277
        assert_eq!(converted_data[1], 6553); // 0.2 * 32767 ≈ 6553
        assert_eq!(converted_data[2], 9830); // 0.3 * 32767 ≈ 9830
    }

    #[test]
    fn test_convenience_methods() {
        let data = array![100i16, -200, 300];
        let audio_i16 = AudioSamples::new_mono(data, 44100);

        // Test convenience methods
        let audio_f32 = audio_i16.as_f32().unwrap();
        let audio_f64 = audio_i16.as_f64().unwrap();
        let audio_i32 = audio_i16.as_i32().unwrap();

        assert_eq!(audio_f32.num_channels(), 1);
        assert_eq!(audio_f64.num_channels(), 1);
        assert_eq!(audio_i32.num_channels(), 1);

        // Verify sample rate preservation
        assert_eq!(audio_f32.sample_rate(), 44100);
        assert_eq!(audio_f64.sample_rate(), 44100);
        assert_eq!(audio_i32.sample_rate(), 44100);
    }

    #[test]
    fn test_round_trip_conversion() {
        let original_data = array![0.123f32, -0.456, 0.789, -0.999, 0.0];
        let audio_original = AudioSamples::new_mono(original_data.clone(), 44100);

        // Convert f32 -> i16 -> f32
        let audio_i16 = audio_original.as_type::<i16>().unwrap();
        let audio_round_trip = audio_i16.as_type::<f32>().unwrap();

        // Verify structure preservation
        assert_eq!(audio_round_trip.num_channels(), 1);
        assert_eq!(audio_round_trip.samples_per_channel(), 5);
        assert_eq!(audio_round_trip.sample_rate(), 44100);

        // Check that values are approximately preserved (within i16 precision limits)
        let round_trip_data = audio_round_trip.as_mono().unwrap();
        for (original, round_trip) in original_data.iter().zip(round_trip_data.iter()) {
            assert_approx_eq!(*original as f64, *round_trip as f64, 5e-4); // i16 precision
        }
    }

    #[test]
    fn test_edge_cases() {
        // Test with minimum and maximum values
        let data = array![f32::MAX, f32::MIN, 0.0f32];
        let audio_f32 = AudioSamples::new_mono(data, 44100);

        // Convert to i16 (should clamp extreme values)
        let audio_i16 = audio_f32.as_type::<i16>().unwrap();
        let converted_data = audio_i16.as_mono().unwrap();

        // f32::MAX and f32::MIN should be clamped to i16 range
        assert_eq!(converted_data[0], i16::MAX);
        assert_eq!(converted_data[1], i16::MIN);
        assert_eq!(converted_data[2], 0);
    }
}
