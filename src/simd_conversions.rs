//! SIMD-optimized audio sample conversions.
//!
//! This module provides high-performance vectorized conversions between different
//! audio sample types using SIMD

#[cfg(feature = "simd")]
use wide::f32x8;

use crate::{AudioSample, AudioSampleError, AudioSampleResult, ConvertTo};

/// SIMD-optimized f32 to i16 conversion.
///
/// Processes 8 samples at a time using AVX2 when available,
/// falling back to scalar operations for remainder.
#[cfg(feature = "simd")]
pub fn convert_f32_to_i16_simd(input: &[f32], output: &mut [i16]) -> AudioSampleResult<()> {
    if input.len() != output.len() {
        return Err(AudioSampleError::InvalidParameter(
            "Input and output slices must have same length".to_string(),
        ));
    }

    let chunks = input.len() / 8;
    let remainder = input.len() % 8;

    // Process 8 samples at a time with SIMD
    for i in 0..chunks {
        let start_idx = i * 8;

        // Load 8 f32 values
        let f32_vec = f32x8::from([
            input[start_idx],
            input[start_idx + 1],
            input[start_idx + 2],
            input[start_idx + 3],
            input[start_idx + 4],
            input[start_idx + 5],
            input[start_idx + 6],
            input[start_idx + 7],
        ]);

        // Scale to i16 range and clamp
        let scaled = f32_vec * f32x8::splat(32767.0);
        let clamped = scaled
            .max(f32x8::splat(-32768.0))
            .min(f32x8::splat(32767.0));

        // Convert to i32 first, then to i16
        let as_array = clamped.to_array();
        for (j, &sample) in as_array.iter().enumerate() {
            output[start_idx + j] = sample as i16;
        }
    }

    // Handle remaining samples with scalar code
    let start_remainder = chunks * 8;
    for i in 0..remainder {
        let sample = input[start_remainder + i];
        let scaled = sample * 32767.0;
        output[start_remainder + i] = scaled.clamp(-32768.0, 32767.0) as i16;
    }

    Ok(())
}

/// SIMD-optimized i16 to f32 conversion.
///
/// Processes 8 samples at a time using AVX2 when available.
#[cfg(feature = "simd")]
pub fn convert_i16_to_f32_simd(input: &[i16], output: &mut [f32]) -> AudioSampleResult<()> {
    if input.len() != output.len() {
        return Err(AudioSampleError::InvalidParameter(
            "Input and output slices must have same length".to_string(),
        ));
    }

    let chunks = input.len() / 8;
    let remainder = input.len() % 8;

    // Process 8 samples at a time with SIMD
    for i in 0..chunks {
        let start_idx = i * 8;

        // Load 8 i16 values and convert to f32
        let f32_values = f32x8::from([
            input[start_idx] as f32,
            input[start_idx + 1] as f32,
            input[start_idx + 2] as f32,
            input[start_idx + 3] as f32,
            input[start_idx + 4] as f32,
            input[start_idx + 5] as f32,
            input[start_idx + 6] as f32,
            input[start_idx + 7] as f32,
        ]);

        // Normalize to [-1.0, 1.0] range
        let normalized = f32_values / f32x8::splat(32767.0);

        let as_array = normalized.to_array();
        for (j, &sample) in as_array.iter().enumerate() {
            output[start_idx + j] = sample;
        }
    }

    // Handle remaining samples with scalar code
    let start_remainder = chunks * 8;
    for i in 0..remainder {
        output[start_remainder + i] = input[start_remainder + i] as f32 / 32767.0;
    }

    Ok(())
}

/// SIMD-optimized f32 to i32 conversion.
#[cfg(feature = "simd")]
pub fn convert_f32_to_i32_simd(input: &[f32], output: &mut [i32]) -> AudioSampleResult<()> {
    if input.len() != output.len() {
        return Err(AudioSampleError::InvalidParameter(
            "Input and output slices must have same length".to_string(),
        ));
    }

    let chunks = input.len() / 8;
    let remainder = input.len() % 8;

    for i in 0..chunks {
        let start_idx = i * 8;

        let f32_vec = f32x8::from([
            input[start_idx],
            input[start_idx + 1],
            input[start_idx + 2],
            input[start_idx + 3],
            input[start_idx + 4],
            input[start_idx + 5],
            input[start_idx + 6],
            input[start_idx + 7],
        ]);

        // Scale to i32 range and clamp
        let scaled = f32_vec * f32x8::splat(2147483647.0);
        let clamped = scaled
            .max(f32x8::splat(-2147483648.0))
            .min(f32x8::splat(2147483647.0));

        let as_array = clamped.to_array();
        for (j, &sample) in as_array.iter().enumerate() {
            output[start_idx + j] = sample as i32;
        }
    }

    // Handle remainder
    let start_remainder = chunks * 8;
    for i in 0..remainder {
        let sample = input[start_remainder + i];
        let scaled = sample * 2147483647.0;
        output[start_remainder + i] = scaled.clamp(-2147483648.0, 2147483647.0) as i32;
    }

    Ok(())
}

/// Generic SIMD-optimized conversion dispatcher.
///
/// Routes to the appropriate SIMD conversion function based on types.
#[cfg(feature = "simd")]
pub fn convert_simd<T: AudioSample, U: AudioSample>(
    input: &[T],
    output: &mut [U],
) -> AudioSampleResult<()>
where
    T: ConvertTo<U>,
{
    use std::any::TypeId;

    // Dispatch to optimized conversions for common types
    if TypeId::of::<T>() == TypeId::of::<f32>() && TypeId::of::<U>() == TypeId::of::<i16>() {
        let input =
            unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f32, input.len()) };
        let output = unsafe {
            std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut i16, output.len())
        };
        return convert_f32_to_i16_simd(input, output);
    }

    if TypeId::of::<T>() == TypeId::of::<i16>() && TypeId::of::<U>() == TypeId::of::<f32>() {
        let input =
            unsafe { std::slice::from_raw_parts(input.as_ptr() as *const i16, input.len()) };
        let output = unsafe {
            std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f32, output.len())
        };
        return convert_i16_to_f32_simd(input, output);
    }

    if TypeId::of::<T>() == TypeId::of::<f32>() && TypeId::of::<U>() == TypeId::of::<i32>() {
        let input =
            unsafe { std::slice::from_raw_parts(input.as_ptr() as *const f32, input.len()) };
        let output = unsafe {
            std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut i32, output.len())
        };
        return convert_f32_to_i32_simd(input, output);
    }

    // Fallback to scalar conversion for unsupported type combinations
    for (i, o) in input.iter().zip(output.iter_mut()) {
        *o = i.convert_to()?;
    }

    Ok(())
}

/// Optimized scalar conversion with manual loop unrolling.
pub fn convert_scalar_unrolled<T: AudioSample, U: AudioSample>(
    input: &[T],
    output: &mut [U],
) -> AudioSampleResult<()>
where
    T: ConvertTo<U>,
{
    if input.len() != output.len() {
        return Err(AudioSampleError::InvalidParameter(
            "Input and output slices must have same length".to_string(),
        ));
    }

    let chunks = input.len() / 4;
    let remainder = input.len() % 4;

    // Process 4 samples at a time (manual unrolling)
    for i in 0..chunks {
        let base = i * 4;
        output[base] = input[base].convert_to()?;
        output[base + 1] = input[base + 1].convert_to()?;
        output[base + 2] = input[base + 2].convert_to()?;
        output[base + 3] = input[base + 3].convert_to()?;
    }

    // Handle remainder
    let start_remainder = chunks * 4;
    for i in 0..remainder {
        output[start_remainder + i] = input[start_remainder + i].convert_to()?;
    }

    Ok(())
}

/// High-level conversion function.
///
/// Automatically chooses between SIMD and scalar implementations based on
/// feature flags and runtime detection.
pub fn convert<T: AudioSample, U: AudioSample>(
    input: &[T],
    output: &mut [U],
) -> AudioSampleResult<()>
where
    T: ConvertTo<U>,
{
    #[cfg(feature = "simd")]
    {
        convert_simd(input, output)
    }

    #[cfg(not(feature = "simd"))]
    {
        convert_scalar_unrolled(input, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_conversion() {
        let _input = vec![0.5f32, -0.3, 0.8, 1.0, -1.0];
        let mut _output = vec![0i16; 5];

        // Test is compiled but only runs when simd feature is disabled
        #[cfg(not(feature = "simd"))]
        {
            // scalar_optimized::convert_scalar_unrolled(&_input, &mut _output).unwrap();

            // Verify conversion accuracy
            assert_eq!(_output[0], 16383); // 0.5 * 32767 = 16383.5 -> 16383
            assert_eq!(_output[1], (-0.3 * 32767.0) as i16);
            assert_eq!(_output[4], -32767); // Clamped to i16::MIN + 1
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_f32_to_i16_conversion() {
        let input = vec![0.5f32, -0.3, 0.8, 1.0, -1.0, 0.0, 0.1, -0.1, 0.25];
        let mut output = vec![0i16; 9];

        simd::convert_f32_to_i16_simd(&input, &mut output).unwrap();

        // Verify conversion accuracy
        assert_eq!(output[0], 16383); // 0.5 * 32767 = 16383.5 -> 16383
        assert_eq!(output[1], (-0.3 * 32767.0) as i16);
        assert_eq!(output[8], (0.25 * 32767.0) as i16); // Test remainder handling
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_i16_to_f32_conversion() {
        let input = vec![16383i16, -9830, 26214, 32767, -32768, 0, 3276, -3276];
        let mut output = vec![0.0f32; 8];

        simd::convert_i16_to_f32_simd(&input, &mut output).unwrap();

        // Verify conversion accuracy using approx_eq
        use approx_eq::assert_approx_eq;
        assert_approx_eq!(output[0] as f64, 16383.0 / 32767.0, 1e-5);
        assert_approx_eq!(output[1] as f64, -9830.0 / 32767.0, 1e-5);
        assert_approx_eq!(output[3] as f64, 1.0, 1e-5);
    }

    #[test]
    fn test_optimized_conversion_dispatch() {
        let input = vec![0.5f32, -0.3, 0.8];
        let mut output = vec![0i16; 3];

        convert(&input, &mut output).unwrap();

        // Should work regardless of SIMD feature
        assert_eq!(output.len(), 3);
        assert_eq!(output[0], 16383); // 0.5 * 32767 = 16383.5 -> 16383
    }
}
