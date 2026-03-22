//! Audio sample type conversion utilities. is this module?
//!
//! This module defines the audio-aware type conversion facilities for
//! [`AudioSamples`]. It provides a consistent and explicit mechanism for
//! converting audio data between different sample representations while
//! preserving the structural properties of the signal — sample rate, channel
//! count, and temporal ordering are always retained exactly.
//!
//! The supported sample types are `u8`, `i16`, `i24` ([`I24`](i24::I24)), `i32`, `f32`,
//! and `f64`. Two distinct conversion classes are provided:
//!
//! - **Audio-aware conversions** (`to_format`, `to_type`, `as_f32`, etc.) —
//!   interpret numeric values as audio samples and apply appropriate scaling,
//!   clamping, and rounding when converting between floating-point and integer
//!   representations. `u8` audio uses the unsigned PCM convention (mid-scale
//!   value `128` maps to silence / `0.0`).
//! - **Raw numeric casts** (`cast_as`, `cast_to`, `cast_as_f64`) — reinterpret
//!   values using standard Rust `as`-cast rules without any audio-specific
//!   scaling or normalisation. does this module exist?
//!
//! Audio data is routinely stored and processed using different sample types.
//! Fixed-width PCM formats (`i16`, `i24`, `i32`, `u8`) are used for storage
//! and interoperability with audio hardware; floating-point formats (`f32`,
//! `f64`) are preferred for analysis, effects, and machine-learning pipelines.
//! This module centralises those conversions so they are applied correctly and
//! consistently throughout the crate, without duplicating conversion logic in
//! every consumer.
//!
//! Conversions are exposed via the [`AudioTypeConversion`] trait, which is
//! implemented for [`AudioSamples`] whenever the underlying sample type
//! supports the required conversion operations. In typical usage the trait
//! does not need to be referenced directly — its methods are available on any
//! `AudioSamples` value. should it be used?
//!
//! Call conversion methods directly on an `AudioSamples` value. Use
//! `to_format` (borrows) or `to_type` (consumes) for audio-aware conversions
//! that preserve amplitude meaning. Use `cast_as` / `cast_to` only when you
//! need the raw numeric value without any amplitude normalisation.
//!
//! ```
//! use audio_samples::{AudioSamples, AudioTypeConversion, sample_rate};
//! use ndarray::array;
//!
//! // Convert f32 audio to i16 PCM (audio-aware: scales to ±32767)
//! let audio_f32 = AudioSamples::new_mono(array![0.5f32, -0.3, 0.8], sample_rate!(44100)).unwrap();
//! let audio_i16 = audio_f32.to_format::<i16>();
//! assert!((audio_i16[0] - 16384).abs() <= 1); // 0.5 × 32767 ≈ 16384
//!
//! // Convert back — audio-aware round-trip preserves amplitude to i16 precision
//! let audio_back: AudioSamples<'static, f32> = audio_i16.to_format::<f32>();
//! assert!((audio_back[0] - 0.5).abs() < 1e-3);
//! ```
//!
//! ## Allocation and ownership
//!
//! All conversion operations produce a new owned [`AudioSamples<'static, O>`]
//! value. The source audio is never modified. No conversion performs in-place
//! mutation, and no conversion requires contiguous storage.
//!
//! ## Error handling
//!
//! Conversion methods do not return `Result`. All supported conversions are
//! defined over their entire input domain. When converting from floating-point
//! to fixed-width integer formats, values outside the representable range are
//! clamped.
use crate::{AudioSamples, AudioTypeConversion, CastInto, ConvertTo, traits::StandardSample};

impl<T> AudioTypeConversion for AudioSamples<'_, T>
where
    T: StandardSample,
{
    type Sample = T;

    /// Converts the audio to a different sample type using audio-aware scaling.
    ///
    /// Performs an element-wise audio-aware conversion from `T` to `O`.
    /// Amplitude meaning is preserved: integer-to-float conversions normalise
    /// into `[-1.0, 1.0]`; float-to-integer conversions scale, clamp, and
    /// round; integer-to-integer conversions shift bit depth with saturation.
    /// For `u8`, the unsigned PCM convention applies (mid-scale `128` = silence).
    ///
    /// The source audio is not modified; a new owned value is returned.
    ///
    /// See [`AudioTypeConversion::to_format`] for the full contract.
    ///
    /// # Arguments
    /// – `O` — the target sample type; must implement [`StandardSample`] and
    ///   `T` must implement [`ConvertTo<O>`].
    ///
    /// # Returns
    /// A new owned [`AudioSamples<'static, O>`] with amplitude-preserving
    /// converted samples. Sample rate, channel count, and ordering are
    /// unchanged.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTypeConversion, sample_rate};
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(array![0.5f32, -1.0, 0.0], sample_rate!(44100)).unwrap();
    /// let audio_i16 = audio.to_format::<i16>();
    /// assert!((audio_i16[0] - 16384).abs() <= 1); // 0.5 × 32767 ≈ 16384
    /// assert_eq!(audio_i16[1], i16::MIN + 1);          // -1.0 → i16::MIN
    /// assert_eq!(audio_i16[2], 0);                 // 0.0 → 0
    /// ```
    #[inline]
    fn to_format<O>(&self) -> AudioSamples<'static, O>
    where
        T: ConvertTo<O>,
        O: StandardSample,
    {
        self.map_into(T::convert_to)
    }

    /// Converts the audio to a different sample type, consuming the source.
    ///
    /// This is the consuming counterpart to [`AudioTypeConversion::to_format`].
    /// It performs the same audio-aware conversion but takes ownership of the
    /// input, avoiding a clone when the source is no longer needed.
    ///
    /// See [`AudioTypeConversion::to_type`] for the full contract.
    ///
    /// # Arguments
    /// – `O` — the target sample type; must implement [`StandardSample`] and
    ///   `T` must implement [`ConvertTo<O>`].
    ///
    /// # Returns
    /// A new owned [`AudioSamples<'static, O>`] with amplitude-preserving
    /// converted samples. Sample rate, channel count, and ordering are
    /// unchanged.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTypeConversion, sample_rate};
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(array![0i16, i16::MAX], sample_rate!(44100)).unwrap();
    /// let audio_f32: AudioSamples<'static, f32> = audio.to_type::<f32>();
    /// assert_eq!(audio_f32[0], 0.0);
    /// assert!((audio_f32[1] - 1.0).abs() < 1e-4);
    /// ```
    #[inline]
    fn to_type<O>(self) -> AudioSamples<'static, O>
    where
        T: ConvertTo<O>,
        O: StandardSample,
    {
        self.map_into(T::convert_to)
    }

    /// Casts the audio to a different sample type without audio-aware scaling.
    ///
    /// Performs a raw element-wise numeric cast from `T` to `O`, equivalent to
    /// an `as` cast applied to every sample. No normalisation, clamping, or
    /// bit-depth scaling is applied — integer values are preserved as their raw
    /// numeric magnitude.
    ///
    /// Use [`AudioTypeConversion::to_format`] when amplitude meaning must be
    /// preserved across sample types.
    ///
    /// See [`AudioTypeConversion::cast_as`] for the full contract.
    ///
    /// # Arguments
    /// – `O` — the target sample type; must implement [`StandardSample`] and
    ///   `T` must implement [`CastInto<O>`].
    ///
    /// # Returns
    /// A new owned [`AudioSamples<'static, O>`] containing raw-cast samples.
    /// The source audio is unchanged. Sample rate, channel count, and ordering
    /// are preserved.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTypeConversion, sample_rate};
    /// use ndarray::array;
    ///
    /// // Raw cast: i16 value 1000 becomes f32 value 1000.0, not 0.030518...
    /// let audio = AudioSamples::new_mono(array![1000i16, -500], sample_rate!(44100)).unwrap();
    /// let raw = audio.cast_as::<f32>();
    /// assert_eq!(raw[0], 1000.0_f32);
    /// assert_eq!(raw[1], -500.0_f32);
    /// ```
    #[inline]
    fn cast_as<O>(&self) -> AudioSamples<'static, O>
    where
        T: CastInto<O> + ConvertTo<O>,
        O: StandardSample,
    {
        self.map_into(T::cast_into)
    }

    /// Casts the audio to a different sample type without audio-aware scaling,
    /// consuming the source.
    ///
    /// This is the consuming counterpart to [`AudioTypeConversion::cast_as`].
    /// It performs the same raw numeric cast but takes ownership of the input.
    ///
    /// See [`AudioTypeConversion::cast_to`] for the full contract.
    ///
    /// # Arguments
    /// – `O` — the target sample type; must implement [`StandardSample`] and
    ///   `T` must implement [`CastInto<O>`].
    ///
    /// # Returns
    /// A new owned [`AudioSamples<'static, O>`] containing raw-cast samples.
    /// Sample rate, channel count, and ordering are preserved.
    ///
    /// # Examples
    /// ```
    /// use audio_samples::{AudioSamples, AudioTypeConversion, sample_rate};
    /// use ndarray::array;
    ///
    /// let audio = AudioSamples::new_mono(array![255u8, 128u8, 0u8], sample_rate!(44100)).unwrap();
    /// let raw: AudioSamples<'static, i16> = audio.cast_to::<i16>();
    /// assert_eq!(raw[0], 255);
    /// assert_eq!(raw[1], 128);
    /// assert_eq!(raw[2], 0);
    /// ```
    #[inline]
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
        assert_eq!(converted_data[0], 3276); // 0.1 * 32767 = 3276.7, truncated to 3276
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

        // f32::MAX and f32::MIN are clamped to [-1.0, 1.0] then scaled by i16::MAX.
        // Symmetric scaling maps -1.0 to -32767 (not i16::MIN = -32768); this is
        // intentional — see impl_float_to_int! in traits.rs for rationale.
        assert_eq!(converted_data[0], i16::MAX);
        assert_eq!(converted_data[1], -32767i16);
        assert_eq!(converted_data[2], 0);
    }
}
