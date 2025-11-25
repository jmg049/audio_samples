//! # Audio Sample Type Conversions
//!
//! This module provides **type-safe conversion operations**
//! between different audio sample representations for [`AudioSamples`].
//!
//! Conversions are exposed through the [`AudioTypeConversion`] trait,
//! which is automatically implemented for `AudioSamples<T>` where `T` is any supported sample type.
//! Users do not need to interact with the trait directly in most cases — the methods can be called
//! directly on an `AudioSamples` instance:
//!
//! ```rust
//! use audio_samples::{AudioSamples, AudioTypeConversion};
//! use ndarray::array;
//!
//! # fn example() {
//! // Create mono audio data with f32 samples
//! let audio_f32 = AudioSamples::new_mono(array![0.5f32, -0.3, 0.8], 44_100);
//!
//! // Convert to i16 (audio-aware conversion)
//! let audio_i16 = audio_f32.to_format::<i16>();
//!
//! // Consume and convert
//! let audio_back = audio_i16.to_type::<f32>();
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
//! Conversions are applied element-by-element via the crate's conversion traits.
//! When converting from floating-point to fixed-width integer formats, extreme values are clamped
//! to the destination range to avoid overflow.
//!
//! ## Typical Use Cases
//! - Converting audio data from disk to a processing format (e.g. `i16` → `f32`)
//! - Preparing audio buffers for model input
//! - Reducing memory footprint after processing (`f64` → `i16`)
//!
//! ## Allocation and Shape
//! All conversion methods allocate a new owned buffer and preserve:
//! - sample rate
//! - channel count and sample count
//! - sample ordering (mono, or `[channel, frame]` indexing for multi-channel)
//!
//! Maybe in future versions, zero-allocation conversions could be supported for certain cases where they share the same underlying representation.
//!
//! ## Error Handling
//! These conversion methods do not return `Result` and do not require contiguous storage.
//! They always produce a new owned `AudioSamples` value.
//!
//! ## Audio-aware conversion vs raw casting
//! - `to_format` / `to_type` use the audio-aware [`ConvertTo`] conversions (e.g. integer PCM to
//!   floating-point typically maps into $[-1.0, 1.0]$).
//! - `cast_as` / `cast_to` use raw numeric casting via [`CastFrom`] (no audio scaling).
//!
//! ## API Summary
//! - [`to_format`](crate::AudioTypeConversion::to_format): Borrow and convert (creates a new buffer)
//! - [`to_type`](crate::AudioTypeConversion::to_type): Consume and convert (creates a new buffer)
//! - [`cast_as`](crate::AudioTypeConversion::cast_as): Borrow and cast without audio-aware scaling
//! - [`cast_to`](crate::AudioTypeConversion::cast_to): Consume and cast without audio-aware scaling
use crate::{AudioSample, AudioSamples, AudioTypeConversion, CastFrom, ConvertTo, I24};

// Single blanket implementation that satisfies all requirements
impl<'a, T: AudioSample> AudioTypeConversion<'a, T> for AudioSamples<'a, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    T: ConvertTo<T>,
{
    fn to_format<O>(&self) -> AudioSamples<'static, O>
    where
        T: ConvertTo<O>,
        O: AudioSample + ConvertTo<T>,
    {
        self.map_into(O::convert_from)
    }

    fn to_type<O: AudioSample + ConvertTo<T>>(self) -> AudioSamples<'static, O>
    where
        T: ConvertTo<O>,
    {
        self.map_into(O::convert_from)
    }

    fn cast_as<O>(&self) -> AudioSamples<'static, O>
    where
        O: AudioSample + CastFrom<T>,
    {
        self.map_into(O::cast_from)
    }

    fn cast_to<O>(self) -> AudioSamples<'static, O>
    where
        O: AudioSample + CastFrom<T>,
    {
        self.map_into(O::cast_from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample_rate;
    use approx_eq::assert_approx_eq;
    use ndarray::array;

    #[test]
    fn test_mono_f32_to_i16_conversion() {
        let data = array![0.5f32, -0.3, 0.0, 1.0, -1.0];
        let audio_f32 = AudioSamples::new_mono(data, sample_rate!(44100));

        let audio_i16 = audio_f32.to_format::<i16>();

        assert_eq!(audio_i16.sample_rate(), sample_rate!(44100));
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
        let audio_i16 = AudioSamples::new_multi_channel(data, sample_rate!(48000));

        let audio_f32 = audio_i16.to_format::<f32>();
        assert_eq!(audio_f32.sample_rate(), sample_rate!(48000));
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
        let audio_f32 = AudioSamples::new_mono(data, sample_rate!(44100));

        // Test consuming conversion
        let audio_i16 = audio_f32.to_type::<i16>();

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
        let audio_i16 = AudioSamples::new_mono(data, sample_rate!(44100));

        // Test convenience methods
        let audio_f32 = audio_i16.as_f32();
        let audio_f64 = audio_i16.as_f64();
        let audio_i32 = audio_i16.as_i32();

        assert_eq!(audio_f32.num_channels(), 1);
        assert_eq!(audio_f64.num_channels(), 1);
        assert_eq!(audio_i32.num_channels(), 1);

        // Verify sample rate preservation
        assert_eq!(audio_f32.sample_rate(), sample_rate!(44100));
        assert_eq!(audio_f64.sample_rate(), sample_rate!(44100));
        assert_eq!(audio_i32.sample_rate(), sample_rate!(44100));
    }

    #[test]
    fn test_round_trip_conversion() {
        let original_data = array![0.123f32, -0.456, 0.789, -0.999, 0.0];
        let audio_original = AudioSamples::new_mono(original_data.clone(), sample_rate!(44100));

        // Convert f32 -> i16 -> f32
        let audio_i16 = audio_original.to_format::<i16>();
        let audio_round_trip = audio_i16.to_format::<f32>();

        // Verify structure preservation
        assert_eq!(audio_round_trip.num_channels(), 1);
        assert_eq!(audio_round_trip.samples_per_channel(), 5);
        assert_eq!(audio_round_trip.sample_rate(), sample_rate!(44100));

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
        let audio_f32 = AudioSamples::new_mono(data, sample_rate!(44100));

        // Convert to i16 (should clamp extreme values)
        let audio_i16 = audio_f32.to_format::<i16>();
        let converted_data = audio_i16.as_mono().unwrap();

        // f32::MAX and f32::MIN should be clamped to i16 range
        assert_eq!(converted_data[0], i16::MAX);
        assert_eq!(converted_data[1], i16::MIN);
        assert_eq!(converted_data[2], 0);
    }
}
