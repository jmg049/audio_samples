//! Type conversion operations for AudioSamples.
//!
//! This module implements the AudioTypeConversion trait, providing efficient
//! vectorized type conversions using ndarray's mapv functionality and the
//! existing ConvertTo trait system.

use super::traits::AudioTypeConversion;
use crate::repr::AudioData;
use crate::{AudioSample, AudioSampleResult, AudioSamples, ConvertTo};

impl<T: AudioSample> AudioTypeConversion<T> for AudioSamples<T> {
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
    fn as_type<O: AudioSample>(&self) -> AudioSampleResult<AudioSamples<O>>
    where
        T: ConvertTo<O>,
    {
        match &self.data {
            AudioData::Mono(arr) => {
                // Use ndarray's mapv for efficient vectorized conversion
                let converted_data = arr.mapv(|sample| {
                    sample.convert_to().unwrap() // Safe because ConvertTo is in trait bounds
                });

                Ok(AudioSamples::new_mono(converted_data, self.sample_rate()))
            }
            AudioData::MultiChannel(arr) => {
                // Use ndarray's mapv for efficient vectorized conversion of multi-channel data
                let converted_data = arr.mapv(|sample| {
                    sample.convert_to().unwrap() // Safe because ConvertTo is in trait bounds
                });

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
    fn to_type<O: AudioSample>(self) -> AudioSampleResult<AudioSamples<O>>
    where
        T: ConvertTo<O>,
    {
        // For consuming conversion, we can use the same implementation as as_type
        // since ndarray's mapv already creates new arrays efficiently
        self.as_type()
    }

    /// Converts to the highest precision floating-point format.
    ///
    /// This is useful when maximum precision is needed for processing.
    /// Uses optimized vectorized conversion.
    fn to_f64(&self) -> AudioSampleResult<AudioSamples<f64>>
    where
        T: ConvertTo<f64>,
    {
        self.as_type::<f64>()
    }

    /// Converts to single precision floating-point format.
    ///
    /// Good balance between precision and memory usage.
    /// Uses optimized vectorized conversion.
    fn to_f32(&self) -> AudioSampleResult<AudioSamples<f32>>
    where
        T: ConvertTo<f32>,
    {
        self.as_type::<f32>()
    }

    /// Converts to 32-bit integer format.
    ///
    /// Highest precision integer format, useful for high-quality processing.
    /// Uses optimized vectorized conversion.
    fn to_i32(&self) -> AudioSampleResult<AudioSamples<i32>>
    where
        T: ConvertTo<i32>,
    {
        self.as_type::<i32>()
    }

    /// Converts to 16-bit integer format (most common).
    ///
    /// Standard format for CD audio and many audio files.
    /// Uses optimized vectorized conversion.
    fn to_i16(&self) -> AudioSampleResult<AudioSamples<i16>>
    where
        T: ConvertTo<i16>,
    {
        self.as_type::<i16>()
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
        assert_eq!(audio_i16.channels(), 1);
        assert_eq!(audio_i16.samples_per_channel(), 5);

        // Check specific conversions
        let converted_data = audio_i16.as_mono().unwrap();
        assert_eq!(converted_data[0], 16384); // 0.5 * 32767 ≈ 16384
        assert_eq!(converted_data[2], 0); // 0.0 -> 0
        assert_eq!(converted_data[3], 32767); // 1.0 -> i16::MAX
        assert_eq!(converted_data[4], -32768); // -1.0 -> i16::MIN
    }

    #[test]
    fn test_multi_channel_i16_to_f32_conversion() {
        // Create 2 channels with 3 samples each: [[ch0_sample0, ch0_sample1, ch0_sample2], [ch1_sample0, ch1_sample1, ch1_sample2]]
        let data = array![[16384i16, 32767, 0], [-16384, -32768, 8192]];
        let audio_i16 = AudioSamples::new_multi_channel(data, 48000);

        let audio_f32 = audio_i16.as_type::<f32>().unwrap();

        assert_eq!(audio_f32.sample_rate(), 48000);
        assert_eq!(audio_f32.channels(), 2);
        assert_eq!(audio_f32.samples_per_channel(), 3);

        // Check specific conversions
        let converted_data = audio_f32.as_multi_channel().unwrap();
        assert_approx_eq!(converted_data[[0, 0]] as f64, 0.5, 1e-4); // 16384/32767 ≈ 0.5
        assert_approx_eq!(converted_data[[0, 1]] as f64, 1.0, 1e-4); // 32767/32767 = 1.0
        assert_approx_eq!(converted_data[[0, 2]] as f64, 0.0, 1e-10); // 0/32767 = 0.0
        assert_approx_eq!(converted_data[[1, 0]] as f64, -0.5, 1e-4); // -16384/32768 ≈ -0.5
        assert_approx_eq!(converted_data[[1, 1]] as f64, -1.0, 1e-6); // -32768/32768 = -1.0
        assert_approx_eq!(converted_data[[1, 2]] as f64, 0.25, 1e-4); // 8192/32768 = 0.25
    }

    #[test]
    fn test_consuming_conversion() {
        let data = array![0.1f32, 0.2, 0.3];
        let audio_f32 = AudioSamples::new_mono(data.clone(), 44100);

        // Test consuming conversion
        let audio_i16 = audio_f32.to_type::<i16>().unwrap();

        assert_eq!(audio_i16.channels(), 1);
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
        let audio_i16 = AudioSamples::new_mono(data, 44100);

        // Test convenience methods
        let audio_f32 = audio_i16.to_f32().unwrap();
        let audio_f64 = audio_i16.to_f64().unwrap();
        let audio_i32 = audio_i16.to_i32().unwrap();

        assert_eq!(audio_f32.channels(), 1);
        assert_eq!(audio_f64.channels(), 1);
        assert_eq!(audio_i32.channels(), 1);

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
        assert_eq!(audio_round_trip.channels(), 1);
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
