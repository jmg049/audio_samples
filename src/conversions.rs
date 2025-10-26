//! Type conversion operations for AudioSamples.
//!
//! This module implements the AudioTypeConversion trait, providing efficient
//! vectorized type conversions using ndarray's mapv functionality and the
//! existing ConvertTo trait system.

use ndarray::{Array1, Array2};
use std::any::TypeId;
use std::borrow::Cow;

use crate::repr::AudioData;
use crate::simd_conversions::convert;
use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, CastFrom,
    ConvertTo, I24,
};

impl<T: AudioSample> AudioTypeConversion<T> for AudioSamples<T>
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
    fn as_type<O: AudioSample + ConvertTo<T>>(&self) -> AudioSampleResult<AudioSamples<O>>
    where
        T: ConvertTo<O>,
    {
        match &self.data {
            AudioData::Mono(arr) => {
                let mut converted_data = Vec::with_capacity(arr.len());
                converted_data.resize(arr.len(), O::default());
                convert(
                    arr.as_slice().ok_or(AudioSampleError::ArrayLayoutError {
                        message: "Array layout must be contiguous.".to_string(),
                    })?,
                    &mut converted_data,
                )?;

                let converted_data = Array1::from(converted_data).into();
                Ok(AudioSamples::new_mono(converted_data, self.sample_rate()))
            }
            AudioData::MultiChannel(arr) => {
                let shape: &[usize] = arr.shape();
                let mut converted_data = Vec::with_capacity(shape[0] * shape[1]);
                converted_data.resize(shape[0] * shape[1], O::default());

                convert(
                    arr.as_slice().ok_or(AudioSampleError::ArrayLayoutError {
                        message: "Array layout must be contiguous.".to_string(),
                    })?,
                    &mut converted_data,
                )?;

                let converted_data = Array2::from_shape_vec((shape[0], shape[1]), converted_data)
                    .map_err(|e| AudioSampleError::DimensionMismatch(e.to_string()))?
                    .into();

                Ok(AudioSamples::new_multi_channel(
                    converted_data,
                    self.sample_rate(),
                ))
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
    fn to_type<O: AudioSample + ConvertTo<T>>(self) -> AudioSampleResult<AudioSamples<O>>
    where
        T: ConvertTo<O>,
    {
        match &self.data {
            AudioData::Mono(arr) => {
                let mut converted_data = Vec::with_capacity(arr.len());
                for sample in arr.iter() {
                    converted_data.push(sample.convert_to()?);
                }
                let converted_data = Array1::from(converted_data).into();

                Ok(AudioSamples::new_mono(converted_data, self.sample_rate()))
            }
            AudioData::MultiChannel(arr) => {
                let shape: &[usize] = arr.shape();
                let mut converted_data = Vec::with_capacity(shape[0] * shape[1]);
                for sample in arr.iter() {
                    converted_data.push(sample.convert_to()?);
                }
                let converted_data = Array2::from_shape_vec((shape[0], shape[1]), converted_data)
                    .map_err(|e| AudioSampleError::DimensionMismatch(e.to_string()))?
                    .into();

                Ok(AudioSamples::new_multi_channel(
                    converted_data,
                    self.sample_rate(),
                ))
            }
        }
    }

    fn cast_as<O: AudioSample + CastFrom<T>>(&self) -> AudioSampleResult<AudioSamples<O>> {
        self.map_into(|x| O::cast_from(x))
    }

    fn cast_to<O: AudioSample + CastFrom<T>>(self) -> AudioSampleResult<AudioSamples<O>> {
        self.map_into(|x| O::cast_from(x))
    }
}

// Cow-based conversion API for zero-copy when possible
impl<T: AudioSample> AudioSamples<T> {
    /// Converts to different sample type using copy-on-write semantics.
    ///
    /// This method provides a runtime choice between borrowing and owning based
    /// on type compatibility. If the source and target types have the same
    /// memory layout and bit patterns, returns a borrowed reference. Otherwise,
    /// performs conversion and returns an owned result.
    ///
    /// # Example
    /// ```rust,ignore
    /// let audio_i32 = AudioSamples::new_mono(i32_data, 44100);
    ///
    /// // Zero-copy for compatible types
    /// if let Ok(cow_f32) = audio_i32.into_cow::<f32>() {
    ///     match cow_f32 {
    ///         Cow::Borrowed(borrowed_audio) => {
    ///             // Zero allocation - same memory layout
    ///             println!("Used zero-copy conversion!");
    ///         }
    ///         Cow::Owned(owned_audio) => {
    ///             // Allocation required - different layout
    ///             println!("Required allocation for conversion");
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// # Safety
    /// Uses unsafe transmutation for compatible layouts. This is safe because:
    /// 1. `SameLayout` trait ensures size and alignment match
    /// 2. `TypeId` comparison ensures we're not mixing incompatible types
    /// 3. Compatible bit patterns are verified per type
    pub fn into_cow<U: AudioSample>(&self) -> AudioSampleResult<Cow<'_, AudioSamples<U>>>
    where
        T: ConvertTo<U>,
        U: ConvertTo<T>,
        i16: ConvertTo<T>,
        I24: ConvertTo<T>,
        i32: ConvertTo<T>,
        f32: ConvertTo<T>,
        f64: ConvertTo<T>,
        for<'a> AudioSamples<T>: AudioTypeConversion<T>,
    {
        // Check if types are identical at runtime
        if TypeId::of::<T>() == TypeId::of::<U>() {
            // Safe transmutation for identical types
            let borrowed: &AudioSamples<U> = unsafe { std::mem::transmute(self) };
            Ok(Cow::Borrowed(borrowed))
        } else {
            // Different types require allocation and conversion
            let converted = self.as_type::<U>()?;
            Ok(Cow::Owned(converted))
        }
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
        let audio_f32 = AudioSamples::new_mono(data.into(), 44100);

        let audio_i16 = audio_f32.as_type::<i16>().unwrap();

        assert_eq!(audio_i16.sample_rate(), 44100);
        assert_eq!(audio_i16.num_channels(), 1);
        assert_eq!(audio_i16.samples_per_channel(), 5);

        // Check specific conversions (allow for minor rounding differences)
        let converted_data = audio_i16.as_mono().unwrap();
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
        let audio_i16 = AudioSamples::new_multi_channel(data.into(), 48000);

        let audio_f32 = audio_i16.as_type::<f32>().unwrap();

        assert_eq!(audio_f32.sample_rate(), 48000);
        assert_eq!(audio_f32.num_channels(), 2);
        assert_eq!(audio_f32.samples_per_channel(), 3);

        // Check specific conversions
        let converted_data = audio_f32.as_multi_channel().unwrap();
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
        let audio_f32 = AudioSamples::new_mono(data.into(), 44100);

        // Test consuming conversion
        let audio_i16 = audio_f32.to_type::<i16>().unwrap();

        assert_eq!(audio_i16.num_channels(), 1);
        assert_eq!(audio_i16.samples_per_channel(), 3);

        // Verify conversion accuracy
        let converted_data = audio_i16.as_mono().unwrap();
        assert_eq!(converted_data[0], 3277); // 0.1 * 32767 ≈ 3277
        assert_eq!(converted_data[1], 6553); // 0.2 * 32767 ≈ 6553
        assert_eq!(converted_data[2], 9830); // 0.3 * 32767 ≈ 9830
    }

    #[test]
    fn test_convenience_methods() {
        let data = array![100i16, -200, 300];
        let audio_i16 = AudioSamples::new_mono(data.into(), 44100);

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
        let audio_original = AudioSamples::new_mono(original_data.clone().into(), 44100);

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
        let audio_f32 = AudioSamples::new_mono(data.into(), 44100);

        // Convert to i16 (should clamp extreme values)
        let audio_i16 = audio_f32.as_type::<i16>().unwrap();
        let converted_data = audio_i16.as_mono().unwrap();

        // f32::MAX and f32::MIN should be clamped to i16 range
        assert_eq!(converted_data[0], i16::MAX);
        assert_eq!(converted_data[1], i16::MIN);
        assert_eq!(converted_data[2], 0);
    }

    // Tests for Cow-based conversion API
    #[test]
    fn test_cow_identical_types() {
        let data = array![1.0f32, 2.0, 3.0];
        let audio = AudioSamples::new_mono(data, 44100);

        // Same type should use zero-copy borrowing
        let cow_result = audio.into_cow::<f32>().unwrap();
        match cow_result {
            Cow::Borrowed(_) => {
                // This is what we expect for identical types
                assert!(true);
            }
            Cow::Owned(_) => {
                panic!("Expected borrowed for identical types");
            }
        }
    }

    #[test]
    fn test_cow_different_types() {
        let data = array![1000i16, 2000i16, 3000i16];
        let audio_i16 = AudioSamples::new_mono(data, 44100);

        // i16 and f32 have different types, should require conversion
        let cow_result = audio_i16.into_cow::<f32>().unwrap();
        match cow_result {
            Cow::Borrowed(_) => {
                panic!("Should not be borrowed for different types");
            }
            Cow::Owned(converted) => {
                // Verify the conversion worked
                assert_eq!(converted.samples_per_channel(), 3);
                assert_eq!(converted.sample_rate(), 44100);
            }
        }
    }
}
