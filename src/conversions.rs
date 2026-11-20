//! Conversion of [`AudioSamples`] between sample formats.
//!
//! The methods of [`AudioTypeConversion`] change the sample type and keep the
//! sample rate, channel count and sample order. There are two kinds:
//!
//! - **Audio conversions** ([`to_format`](AudioTypeConversion::to_format), which
//!   borrows; [`to_type`](AudioTypeConversion::to_type), which consumes;
//!   `as_f32` and the other `as_*` shorthands) treat values as amplitudes and
//!   rescale, so full scale stays full scale. `u8` is offset binary: `128` is
//!   silence.
//! - **Numeric casts** ([`cast_as`](AudioTypeConversion::cast_as),
//!   [`cast_to`](AudioTypeConversion::cast_to), `cast_as_f64`) follow Rust's `as`
//!   rules with no scaling. Use them only when you need the raw number, such as an
//!   integer sample value in a float computation.
//!
//! ```
//! use audio_samples::{AudioSamples, AudioTypeConversion, sample_rate};
//! use ndarray::array;
//!
//! // To 16-bit PCM: negative values scale by 32768 and positive by 32767,
//! // so -1.0 becomes -32768 and +1.0 becomes 32767.
//! let audio_f32 = AudioSamples::new_mono(array![0.5f32, -0.3, 0.8], sample_rate!(44100)).unwrap();
//! let audio_i16 = audio_f32.to_format::<i16>();
//! assert!((audio_i16[0] - 16384).abs() <= 1); // 0.5 × 32767 ≈ 16384
//!
//! // Back to f32: the amplitude survives to 16-bit precision.
//! let audio_back: AudioSamples<'static, f32> = audio_i16.to_format::<f32>();
//! assert!((audio_back[0] - 0.5).abs() < 1e-3);
//! ```
//!
//! ## Scaling convention
//!
//! For a signed `B`-bit integer type the audio-aware mapping is piecewise
//! linear: negative values are divided by `2^(B−1)` and non-negative values by
//! `2^(B−1) − 1`, so `MIN ↔ −1.0` and `MAX ↔ +1.0` exactly. Float → integer
//! clamps to `[−1, 1]`, scales by the same factors and rounds half away from
//! zero, so integer → float → integer is the identity. `u8` is offset binary
//! (`128` ↔ `0.0`) with divisors 128 (below) and 127 (above).
//!
//! Because non-negative values are divided by `2^(B−1) − 1`, a positive `i16`
//! sample maps to a value up to about 3·10⁻⁵ larger (a relative gain of
//! 32768/32767) than under a mapping that divides both sides by `2^(B−1)`.
//!
//! Integer → integer conversions shift the bit pattern: widening is an exact
//! left shift (`i16::MAX` → `0x7FFF_0000` as `i32`, not `i32::MAX`), and
//! narrowing is an arithmetic right shift, which rounds toward −∞ (a bias of
//! −½ LSB of the target type) rather than to nearest.
//!
//! Requantising to fewer bits adds quantisation error of at most ½ LSB. For
//! busy signals it behaves like uniform noise, giving the familiar
//! signal-to-quantisation-noise ratio of about `6.02·B + 1.76` dB for a
//! full-scale sine. No dither is applied anywhere in this crate; for
//! low-level signals the error is correlated with the signal (distortion),
//! so add TPDF dither yourself before reducing bit depth if that matters.
//!
//! ## Allocation and errors
//!
//! Every conversion returns a new owned `AudioSamples<'static, O>` and leaves the
//! source unchanged; none needs contiguous storage. Conversions cannot fail: each
//! is defined over its whole input domain, and float values outside `[−1, 1]`
//! are clamped when converted to an integer format.
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
    /// round; integer-to-integer conversions shift bit depth (narrowing
    /// truncates toward −∞; see the module docs).
    /// For `u8`, the unsigned PCM convention applies (mid-scale `128` = silence).
    ///
    /// The source audio is not modified; a new owned value is returned.
    ///
    /// See [`AudioTypeConversion::to_format`] for the full contract.
    ///
    /// # Arguments
    /// - `O` - the target sample type; must implement [`StandardSample`] and
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
    /// assert_eq!(audio_i16[0], 16384);             // 0.5 × 32767 = 16383.5, rounded
    /// assert_eq!(audio_i16[1], i16::MIN);          // -1.0 → i16::MIN
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
    /// - `O` - the target sample type; must implement [`StandardSample`] and
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
    /// bit-depth scaling is applied - integer values are preserved as their raw
    /// numeric magnitude.
    ///
    /// Use [`AudioTypeConversion::to_format`] when amplitude meaning must be
    /// preserved across sample types.
    ///
    /// See [`AudioTypeConversion::cast_as`] for the full contract.
    ///
    /// # Arguments
    /// - `O` - the target sample type; must implement [`StandardSample`] and
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
        T: CastInto<O>,
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
    /// - `O` - the target sample type; must implement [`StandardSample`] and
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
        assert_eq!(converted_data[4], i16::MIN); // -1.0 -> i16::MIN
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
        assert_eq!(converted_data[0], 3277); // 0.1 * 32767 = 3276.7, rounded to 3277
        assert_eq!(converted_data[1], 6553); // 0.2 * 32767 = 6553.4, rounded to 6553
        assert_eq!(converted_data[2], 9830); // 0.3 * 32767 = 9830.1, rounded to 9830
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

        // f32::MAX and f32::MIN are clamped to [-1.0, 1.0]; the asymmetric
        // scaling (the inverse of i16 -> f32) maps +1.0 to i16::MAX and -1.0
        // to i16::MIN.
        assert_eq!(converted_data[0], i16::MAX);
        assert_eq!(converted_data[1], i16::MIN);
        assert_eq!(converted_data[2], 0);
    }
}
