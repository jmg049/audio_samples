//! Audio sample type conversion utilities.
//!
//! This module defines the audio-aware type conversion facilities for
//! [`AudioSamples`]. It provides a consistent and explicit mechanism for
//! converting audio data between different sample representations while
//! preserving the structural properties of the signal.
//!
//! ## Purpose
//!
//! Audio data is commonly stored and processed using different numeric sample
//! types depending on context, such as fixed-width PCM formats for storage and
//! floating-point formats for analysis or learning-based models. This module
//! exists to centralise those conversions and to ensure that they are applied
//! correctly and consistently across the crate.
//!
//! Conversions are exposed via the [`AudioTypeConversion`] trait, which is
//! implemented for [`AudioSamples`] when the underlying sample type supports
//! the required conversion operations. In typical usage, the trait does not
//! need to be referenced directly, as its methods are available on
//! `AudioSamples` values.
//!
//! ## Intended Usage
//!
//! Conversion methods are designed for explicit, one-step transitions between
//! audio sample formats. They are commonly used when preparing audio for
//! processing, adapting data for downstream consumers, or reducing memory
//! usage after computation.
//!
//! ```rust
//! use audio_samples::{AudioSamples, AudioTypeConversion};
//! use ndarray::array;
//!
//! fn example() {
//! let audio_f32 = AudioSamples::new_mono(array![0.5f32, -0.3, 0.8], 44_100).unwrap();
//!
//! let audio_i16 = audio_f32.to_format::<i16>();
//! let audio_back = audio_i16.to_type::<f32>();
//! }
//! ```
//!
//! ## Conversion Semantics
//!
//! Conversions operate element-by-element and preserve the logical structure
//! of the audio signal. Sample rate, channel count, and sample ordering are
//! retained exactly. Mono and multi-channel layouts are preserved without
//! reordering.
//!
//! Two classes of conversion are supported:
//!
//! - *Audio-aware conversions*, which interpret numeric values as audio samples
//!   and apply appropriate scaling and clamping when converting between
//!   floating-point and integer representations.
//! - *Raw numeric casts*, which reinterpret values using standard numeric
//!   casting rules without audio-specific scaling.
//!
//! The distinction between these modes is explicit in the API and must be
//! chosen intentionally by the caller.
//!
//! ## Allocation and Ownership
//!
//! All conversion operations produce a new owned [`AudioSamples`] value. The
//! original audio data is never modified, and conversions do not require
//! contiguous storage. No conversion method performs in-place mutation.
//!
//! ## Error Handling
//!
//! Conversion methods do not return `Result`. All supported conversions are
//! total over their input domain and will always produce a valid output. When
//! converting from floating-point to fixed-width integer formats, values that
//! exceed the representable range are clamped to preserve numerical safety.
use crate::{AudioSamples, AudioTypeConversion, CastInto, ConvertTo, traits::StandardSample};

impl<T> AudioTypeConversion for AudioSamples<'_, T>
where
    T: StandardSample,
{
    type Sample = T;

    fn to_format<O>(&self) -> AudioSamples<'static, O>
    where
        T: ConvertTo<O>,
        O: StandardSample,
    {
        self.map_into(T::convert_to)
    }

    fn to_type<O>(self) -> AudioSamples<'static, O>
    where
        T: ConvertTo<O>,
        O: StandardSample,
    {
        self.map_into(T::convert_to)
    }

    fn cast_as<O>(&self) -> AudioSamples<'static, O>
    where
        T: CastInto<O> + ConvertTo<O>,
        O: StandardSample,
    {
        self.map_into(T::cast_into)
    }

    fn cast_to<O>(self) -> AudioSamples<'static, O>
    where
        T: CastInto<O> + ConvertTo<O>,
        O: StandardSample,
    {
        self.map_into(T::cast_into)
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;
    use crate::{channels, sample_rate};
    use approx_eq::assert_approx_eq;
    use ndarray::array;

    #[test]
    fn test_mono_f32_to_i16_conversion() {
        let data = array![0.5f32, -0.3, 0.0, 1.0, -1.0];
        let audio_f32 = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let audio_i16 = audio_f32.to_format::<i16>();

        assert_eq!(audio_i16.sample_rate(), sample_rate!(44100));
        assert_eq!(audio_i16.num_channels(), channels!(1));
        assert_eq!(
            audio_i16.samples_per_channel(),
            NonZeroUsize::new(5).unwrap()
        );

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
        let audio_i16 = AudioSamples::new_multi_channel(data, sample_rate!(48000)).unwrap();

        let audio_f32 = audio_i16.to_format::<f32>();
        assert_eq!(audio_f32.sample_rate(), sample_rate!(48000));
        assert_eq!(audio_f32.num_channels(), channels!(2));
        assert_eq!(
            audio_f32.samples_per_channel(),
            NonZeroUsize::new(3).unwrap()
        );

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
        let audio_f32 = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        // Test consuming conversion
        let audio_i16 = audio_f32.to_type::<i16>();

        assert_eq!(audio_i16.num_channels(), channels!(1));
        assert_eq!(
            audio_i16.samples_per_channel(),
            NonZeroUsize::new(3).unwrap()
        );

        // Verify conversion accuracy
        let converted_data = audio_i16;
        assert_eq!(converted_data[0], 3277); // 0.1 * 32767 ≈ 3277
        assert_eq!(converted_data[1], 6553); // 0.2 * 32767 ≈ 6553
        assert_eq!(converted_data[2], 9830); // 0.3 * 32767 ≈ 9830
    }

    #[test]
    fn test_convenience_methods() {
        let data = array![100i16, -200, 300];
        let audio_i16 = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        // Test convenience methods
        let audio_f32 = audio_i16.as_f32();
        let audio_f64 = audio_i16.as_f64();
        let audio_i32 = audio_i16.as_i32();

        assert_eq!(audio_f32.num_channels(), channels!(1));
        assert_eq!(audio_f64.num_channels(), channels!(1));
        assert_eq!(audio_i32.num_channels(), channels!(1));

        // Verify sample rate preservation
        assert_eq!(audio_f32.sample_rate(), sample_rate!(44100));
        assert_eq!(audio_f64.sample_rate(), sample_rate!(44100));
        assert_eq!(audio_i32.sample_rate(), sample_rate!(44100));
    }

    #[test]
    fn test_round_trip_conversion() {
        let original_data = array![0.123f32, -0.456, 0.789, -0.999, 0.0];
        let audio_original =
            AudioSamples::new_mono(original_data.clone(), sample_rate!(44100)).unwrap();

        // Convert f32 -> i16 -> f32
        let audio_i16 = audio_original.to_format::<i16>();
        let audio_round_trip = audio_i16.to_format::<f32>();

        // Verify structure preservation
        assert_eq!(audio_round_trip.num_channels(), channels!(1));
        assert_eq!(
            audio_round_trip.samples_per_channel(),
            NonZeroUsize::new(5).unwrap()
        );
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
        let audio_f32 = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        // Convert to i16 (should clamp extreme values)
        let audio_i16 = audio_f32.to_format::<i16>();
        let converted_data = audio_i16.as_mono().unwrap();

        // f32::MAX and f32::MIN should be clamped to i16 range
        assert_eq!(converted_data[0], i16::MAX);
        assert_eq!(converted_data[1], i16::MIN);
        assert_eq!(converted_data[2], 0);
    }
}
