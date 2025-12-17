//! SIMD-optimized audio sample conversions.
//!
//! This module provides faster, vectorized helpers for:
//! - converting between common sample types
//! - interleaving and deinterleaving planar/interleaved buffers
//!
//! ## Feature gating
//! Most SIMD conversion routines are behind the `simd` crate feature.
//! When the `simd` feature is disabled, [`convert`] falls back to a scalar implementation.
//!
//! ## Conversion semantics
//! The high-level [`convert`] API is intended to match the semantics of the crate's
//! [`ConvertTo`] conversions (clamp + scale + round + saturate for float → int, and asymmetric
//! scaling for int → float).

#[cfg(feature = "simd")]
use wide::f32x8;

use crate::{AudioSample, AudioSampleError, AudioSampleResult, ConvertTo, ParameterError};

/// SIMD-optimized `f32` → `i16` conversion.
///
/// # Behavior
/// Matches the crate's [`ConvertTo<i16>`] semantics for `f32`:
/// - clamps the input to $[-1.0, 1.0]$
/// - scales with asymmetric endpoints (`-1.0 → i16::MIN`, `1.0 → i16::MAX`)
/// - rounds to the nearest integer and saturates to the destination range
///
/// # Errors
/// Returns an error if `input.len() != output.len()`.
#[cfg(feature = "simd")]
pub fn convert_f32_to_i16_simd(input: &[f32], output: &mut [i16]) -> AudioSampleResult<()> {
    if input.len() != output.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "slice_lengths",
            "Input and output slices must have same length",
        )));
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

        // Clamp to [-1.0, 1.0]
        let clamped = f32_vec.max(f32x8::splat(-1.0)).min(f32x8::splat(1.0));

        // Convert lane-by-lane to exactly match ConvertTo semantics.
        let as_array = clamped.to_array();
        for (j, &v) in as_array.iter().enumerate() {
            let scaled = if v < 0.0 { v * 32768.0 } else { v * 32767.0 };
            let rounded = scaled.round();
            let clamped = rounded.clamp(i16::MIN as f32, i16::MAX as f32);
            output[start_idx + j] = clamped as i16;
        }
    }

    // Handle remaining samples with scalar code
    let start_remainder = chunks * 8;
    for i in 0..remainder {
        let v = input[start_remainder + i].clamp(-1.0, 1.0);
        let scaled = if v < 0.0 { v * 32768.0 } else { v * 32767.0 };
        let rounded = scaled.round();
        output[start_remainder + i] = rounded.clamp(i16::MIN as f32, i16::MAX as f32) as i16;
    }

    Ok(())
}

/// SIMD-optimized `i16` → `f32` conversion.
///
/// # Behavior
/// Matches the crate's [`ConvertTo<f32>`] semantics for `i16` (asymmetric scaling):
/// - negative values divide by `-(i16::MIN)` (32768)
/// - non-negative values divide by `i16::MAX` (32767)
///
/// # Errors
/// Returns an error if `input.len() != output.len()`.
#[cfg(feature = "simd")]
pub fn convert_i16_to_f32_simd(input: &[i16], output: &mut [f32]) -> AudioSampleResult<()> {
    if input.len() != output.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "slice_lengths",
            "Input and output slices must have same length",
        )));
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

        let as_array = f32_values.to_array();
        for (j, &v) in as_array.iter().enumerate() {
            output[start_idx + j] = if v < 0.0 { v / 32768.0 } else { v / 32767.0 };
        }
    }

    // Handle remaining samples with scalar code
    let start_remainder = chunks * 8;
    for i in 0..remainder {
        let v = input[start_remainder + i] as f32;
        output[start_remainder + i] = if v < 0.0 { v / 32768.0 } else { v / 32767.0 };
    }

    Ok(())
}

/// SIMD-optimized `f32` → `i32` conversion.
///
/// # Behavior
/// Matches the crate's [`ConvertTo<i32>`] semantics for `f32`:
/// - clamps the input to $[-1.0, 1.0]$
/// - scales with asymmetric endpoints (`-1.0 → i32::MIN`, `1.0 → i32::MAX`)
/// - rounds to the nearest integer and saturates to the destination range
///
/// # Errors
/// Returns an error if `input.len() != output.len()`.
#[cfg(feature = "simd")]
pub fn convert_f32_to_i32_simd(input: &[f32], output: &mut [i32]) -> AudioSampleResult<()> {
    if input.len() != output.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "slice_lengths",
            "Input and output slices must have same length",
        )));
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

        let clamped = f32_vec.max(f32x8::splat(-1.0)).min(f32x8::splat(1.0));

        let as_array = clamped.to_array();
        for (j, &v) in as_array.iter().enumerate() {
            let scaled = if v < 0.0 {
                v * 2147483648.0
            } else {
                v * 2147483647.0
            };
            let rounded = scaled.round();
            let clamped = rounded.clamp(i32::MIN as f32, i32::MAX as f32);
            output[start_idx + j] = clamped as i32;
        }
    }

    // Handle remainder
    let start_remainder = chunks * 8;
    for i in 0..remainder {
        let v = input[start_remainder + i].clamp(-1.0, 1.0);
        let scaled = if v < 0.0 {
            v * 2147483648.0
        } else {
            v * 2147483647.0
        };
        let rounded = scaled.round();
        output[start_remainder + i] = rounded.clamp(i32::MIN as f32, i32::MAX as f32) as i32;
    }

    Ok(())
}

/// Generic SIMD-optimized conversion dispatcher.
///
/// Routes to the appropriate SIMD conversion function based on types.
///
/// # Errors
/// Returns an error if `input.len() != output.len()`.
///
/// # Safety
/// This function uses `unsafe` slice re-interpretation for the specialized fast paths.
/// The casts are only taken when the runtime `TypeId` checks prove the types are exactly
/// the expected concrete types.
#[cfg(feature = "simd")]
pub fn convert_simd<T, U>(input: &[T], output: &mut [U]) -> AudioSampleResult<()>
where
    T: AudioSample + ConvertTo<U>,
    U: AudioSample,
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
        *o = i.convert_to();
    }

    Ok(())
}

/// Optimized scalar conversion with manual loop unrolling.
///
/// This is the non-SIMD fast path used by [`convert`] when the `simd` feature is disabled.
///
/// # Arguments
/// - `input`: source samples
/// - `output`: destination buffer (must have the same length as `input`)
///
/// # Errors
/// Returns an error if `input.len() != output.len()`.
///
/// # Panics
/// Does not panic.
pub fn convert_scalar_unrolled<T, U>(input: &[T], output: &mut [U]) -> AudioSampleResult<()>
where
    T: AudioSample + ConvertTo<U>,
    U: AudioSample,
{
    if input.len() != output.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "slice_lengths",
            "Input and output slices must have same length",
        )));
    }

    let chunks = input.len() / 4;
    let remainder = input.len() % 4;

    // Process 4 samples at a time (manual unrolling)
    for i in 0..chunks {
        let base = i * 4;
        output[base] = input[base].convert_to();
        output[base + 1] = input[base + 1].convert_to();
        output[base + 2] = input[base + 2].convert_to();
        output[base + 3] = input[base + 3].convert_to();
    }

    // Handle remainder
    let start_remainder = chunks * 4;
    for i in 0..remainder {
        output[start_remainder + i] = input[start_remainder + i].convert_to();
    }

    Ok(())
}

/// High-level conversion function.
///
/// Automatically chooses between SIMD and scalar implementations based on
/// feature flags and runtime detection.
///
/// # Arguments
/// - `input`: source samples
/// - `output`: destination buffer (must have the same length as `input`)
///
/// # Errors
/// Returns an error if `input.len() != output.len()`.
///
/// # Panics
/// Does not panic.
pub fn convert<T, U>(input: &[T], output: &mut [U]) -> AudioSampleResult<()>
where
    T: AudioSample + ConvertTo<U>,
    U: AudioSample,
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

// =============================================================================
// DEINTERLEAVE OPERATIONS
// =============================================================================

/// Deinterleave stereo audio data.
///
/// Converts interleaved stereo data `[L0, R0, L1, R1, ...]` to planar format
/// `[L0, L1, ..., R0, R1, ...]` in the output slice.
///
/// # Arguments
/// * `interleaved` - Input slice in interleaved format (length must be even)
/// * `output` - Output slice for planar data (same length as input)
///
/// # Errors
/// Returns error if slices have different lengths or input length is not even.
///
/// # Panics
/// Does not panic.
#[inline]
pub fn deinterleave_stereo<T: AudioSample>(
    interleaved: &[T],
    output: &mut [T],
) -> AudioSampleResult<()> {
    if interleaved.len() != output.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "slice_lengths",
            "Input and output slices must have same length",
        )));
    }
    if !interleaved.len().is_multiple_of(2) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "interleaved_length",
            "Interleaved stereo data must have even length",
        )));
    }

    #[cfg(feature = "simd")]
    {
        deinterleave_stereo_simd(interleaved, output)
    }

    #[cfg(not(feature = "simd"))]
    {
        deinterleave_stereo_scalar(interleaved, output)
    }
}

/// Scalar implementation of stereo deinterleave with loop unrolling.
#[inline]
fn deinterleave_stereo_scalar<T: AudioSample>(
    interleaved: &[T],
    output: &mut [T],
) -> AudioSampleResult<()> {
    let frames = interleaved.len() / 2;
    let (left_out, right_out) = output.split_at_mut(frames);

    // Process 4 frames at a time (8 samples)
    let chunks = frames / 4;
    let remainder = frames % 4;

    for i in 0..chunks {
        let base_frame = i * 4;
        let base_interleaved = base_frame * 2;

        // Unrolled: extract L and R for 4 consecutive frames
        left_out[base_frame] = interleaved[base_interleaved];
        right_out[base_frame] = interleaved[base_interleaved + 1];

        left_out[base_frame + 1] = interleaved[base_interleaved + 2];
        right_out[base_frame + 1] = interleaved[base_interleaved + 3];

        left_out[base_frame + 2] = interleaved[base_interleaved + 4];
        right_out[base_frame + 2] = interleaved[base_interleaved + 5];

        left_out[base_frame + 3] = interleaved[base_interleaved + 6];
        right_out[base_frame + 3] = interleaved[base_interleaved + 7];
    }

    // Handle remaining frames
    let start = chunks * 4;
    for i in 0..remainder {
        let frame_idx = start + i;
        let interleaved_idx = frame_idx * 2;
        left_out[frame_idx] = interleaved[interleaved_idx];
        right_out[frame_idx] = interleaved[interleaved_idx + 1];
    }

    Ok(())
}

/// SIMD-accelerated stereo deinterleave for f32 samples.
///
/// Uses f32x8 to process 4 stereo frames (8 samples) at a time.
#[cfg(feature = "simd")]
fn deinterleave_stereo_simd<T: AudioSample>(
    interleaved: &[T],
    output: &mut [T],
) -> AudioSampleResult<()> {
    use std::any::TypeId;

    // Dispatch to type-specific SIMD implementations
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        // SAFETY: We've verified T is f32
        let interleaved = unsafe {
            std::slice::from_raw_parts(interleaved.as_ptr() as *const f32, interleaved.len())
        };
        let output = unsafe {
            std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f32, output.len())
        };
        return deinterleave_stereo_f32_simd(interleaved, output);
    }

    // Fallback to scalar for other types (i16, i32, I24, f64)
    deinterleave_stereo_scalar(interleaved, output)
}

/// SIMD deinterleave specifically for f32 stereo data.
#[cfg(feature = "simd")]
fn deinterleave_stereo_f32_simd(interleaved: &[f32], output: &mut [f32]) -> AudioSampleResult<()> {
    let frames = interleaved.len() / 2;
    let (left_out, right_out) = output.split_at_mut(frames);

    // Process 4 frames at a time using f32x8
    // Load: [L0, R0, L1, R1, L2, R2, L3, R3]
    // We need to shuffle to: [L0, L1, L2, L3] and [R0, R1, R2, R3]
    let simd_chunks = frames / 4;
    let remainder_start = simd_chunks * 4;

    for i in 0..simd_chunks {
        let base_frame = i * 4;
        let base_interleaved = base_frame * 2;

        // Load 8 interleaved samples (4 stereo frames)
        let v = f32x8::from([
            interleaved[base_interleaved],
            interleaved[base_interleaved + 1],
            interleaved[base_interleaved + 2],
            interleaved[base_interleaved + 3],
            interleaved[base_interleaved + 4],
            interleaved[base_interleaved + 5],
            interleaved[base_interleaved + 6],
            interleaved[base_interleaved + 7],
        ]);

        let arr = v.to_array();
        // Extract left channel (even indices)
        left_out[base_frame] = arr[0];
        left_out[base_frame + 1] = arr[2];
        left_out[base_frame + 2] = arr[4];
        left_out[base_frame + 3] = arr[6];

        // Extract right channel (odd indices)
        right_out[base_frame] = arr[1];
        right_out[base_frame + 1] = arr[3];
        right_out[base_frame + 2] = arr[5];
        right_out[base_frame + 3] = arr[7];
    }

    // Handle remaining frames with scalar code
    for i in remainder_start..frames {
        let interleaved_idx = i * 2;
        left_out[i] = interleaved[interleaved_idx];
        right_out[i] = interleaved[interleaved_idx + 1];
    }

    Ok(())
}

/// Deinterleave multi-channel audio data.
///
/// Converts interleaved data `[ch0_f0, ch1_f0, ch2_f0, ch0_f1, ch1_f1, ...]`
/// to planar format `[ch0_f0, ch0_f1, ..., ch1_f0, ch1_f1, ...]`.
///
/// # Arguments
/// * `interleaved` - Input slice in interleaved format
/// * `output` - Output slice for planar data (same length as input)
/// * `num_channels` - Number of audio channels
///
/// # Errors
/// Returns error if lengths don't match or input isn't divisible by channel count.
///
/// # Panics
/// Does not panic.
#[inline]
pub fn deinterleave_multi<T: AudioSample>(
    interleaved: &[T],
    output: &mut [T],
    num_channels: usize,
) -> AudioSampleResult<()> {
    if interleaved.len() != output.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "slice_lengths",
            "Input and output slices must have same length",
        )));
    }
    if num_channels == 0 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "num_channels",
            "Number of channels must be greater than 0",
        )));
    }
    if !interleaved.len().is_multiple_of(num_channels) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "interleaved_length",
            "Interleaved data length must be divisible by channel count",
        )));
    }

    // Use optimized stereo path for 2 channels
    if num_channels == 2 {
        return deinterleave_stereo(interleaved, output);
    }

    // Generic multi-channel deinterleave (>2 channels)
    deinterleave_multi_scalar(interleaved, output, num_channels)
}

/// Scalar implementation of multi-channel deinterleave.
///
/// For >2 channels, we use a cache-friendly approach that processes
/// one channel at a time to maximize sequential writes.
fn deinterleave_multi_scalar<T: AudioSample>(
    interleaved: &[T],
    output: &mut [T],
    num_channels: usize,
) -> AudioSampleResult<()> {
    let frames = interleaved.len() / num_channels;

    // Process each channel sequentially for better cache locality on writes
    for ch in 0..num_channels {
        let out_start = ch * frames;
        let out_slice = &mut output[out_start..out_start + frames];

        for frame in 0..frames {
            out_slice[frame] = interleaved[frame * num_channels + ch];
        }
    }

    Ok(())
}

// =============================================================================
// INTERLEAVE OPERATIONS
// =============================================================================

/// Interleave stereo audio data.
///
/// Converts planar stereo data `[L0, L1, ..., R0, R1, ...]` to interleaved format
/// `[L0, R0, L1, R1, ...]` in the output slice.
///
/// # Arguments
/// * `planar` - Input slice in planar format (first half = left, second half = right)
/// * `output` - Output slice for interleaved data (same length as input)
///
/// # Errors
/// Returns error if slices have different lengths or input length is not even.
///
/// # Panics
/// Does not panic.
#[inline]
pub fn interleave_stereo<T: AudioSample>(planar: &[T], output: &mut [T]) -> AudioSampleResult<()> {
    if planar.len() != output.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "slice_lengths",
            "Input and output slices must have same length",
        )));
    }
    if !planar.len().is_multiple_of(2) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "planar_length",
            "Planar stereo data must have even length",
        )));
    }

    #[cfg(feature = "simd")]
    {
        interleave_stereo_simd(planar, output)
    }

    #[cfg(not(feature = "simd"))]
    {
        interleave_stereo_scalar(planar, output)
    }
}

/// Scalar implementation of stereo interleave with loop unrolling.
#[inline]
fn interleave_stereo_scalar<T: AudioSample>(
    planar: &[T],
    output: &mut [T],
) -> AudioSampleResult<()> {
    let frames = planar.len() / 2;
    let (left_in, right_in) = planar.split_at(frames);

    // Process 4 frames at a time
    let chunks = frames / 4;
    let remainder = frames % 4;

    for i in 0..chunks {
        let base_frame = i * 4;
        let base_interleaved = base_frame * 2;

        // Unrolled: interleave L and R for 4 consecutive frames
        output[base_interleaved] = left_in[base_frame];
        output[base_interleaved + 1] = right_in[base_frame];

        output[base_interleaved + 2] = left_in[base_frame + 1];
        output[base_interleaved + 3] = right_in[base_frame + 1];

        output[base_interleaved + 4] = left_in[base_frame + 2];
        output[base_interleaved + 5] = right_in[base_frame + 2];

        output[base_interleaved + 6] = left_in[base_frame + 3];
        output[base_interleaved + 7] = right_in[base_frame + 3];
    }

    // Handle remaining frames
    let start = chunks * 4;
    for i in 0..remainder {
        let frame_idx = start + i;
        let interleaved_idx = frame_idx * 2;
        output[interleaved_idx] = left_in[frame_idx];
        output[interleaved_idx + 1] = right_in[frame_idx];
    }

    Ok(())
}

/// SIMD-accelerated stereo interleave.
#[cfg(feature = "simd")]
fn interleave_stereo_simd<T: AudioSample>(planar: &[T], output: &mut [T]) -> AudioSampleResult<()> {
    use std::any::TypeId;

    // Dispatch to type-specific SIMD implementations
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        // SAFETY: We've verified T is f32
        let planar =
            unsafe { std::slice::from_raw_parts(planar.as_ptr() as *const f32, planar.len()) };
        let output = unsafe {
            std::slice::from_raw_parts_mut(output.as_mut_ptr() as *mut f32, output.len())
        };
        return interleave_stereo_f32_simd(planar, output);
    }

    // Fallback to scalar for other types
    interleave_stereo_scalar(planar, output)
}

/// SIMD interleave specifically for f32 stereo data.
#[cfg(feature = "simd")]
fn interleave_stereo_f32_simd(planar: &[f32], output: &mut [f32]) -> AudioSampleResult<()> {
    let frames = planar.len() / 2;
    let (left_in, right_in) = planar.split_at(frames);

    // Process 4 frames at a time
    let simd_chunks = frames / 4;
    let remainder_start = simd_chunks * 4;

    for i in 0..simd_chunks {
        let base_frame = i * 4;
        let base_interleaved = base_frame * 2;

        // Load 4 left samples and 4 right samples
        // Interleave them into [L0, R0, L1, R1, L2, R2, L3, R3]
        let l0 = left_in[base_frame];
        let l1 = left_in[base_frame + 1];
        let l2 = left_in[base_frame + 2];
        let l3 = left_in[base_frame + 3];

        let r0 = right_in[base_frame];
        let r1 = right_in[base_frame + 1];
        let r2 = right_in[base_frame + 2];
        let r3 = right_in[base_frame + 3];

        // Build interleaved output
        let interleaved = f32x8::from([l0, r0, l1, r1, l2, r2, l3, r3]);
        let arr = interleaved.to_array();

        output[base_interleaved] = arr[0];
        output[base_interleaved + 1] = arr[1];
        output[base_interleaved + 2] = arr[2];
        output[base_interleaved + 3] = arr[3];
        output[base_interleaved + 4] = arr[4];
        output[base_interleaved + 5] = arr[5];
        output[base_interleaved + 6] = arr[6];
        output[base_interleaved + 7] = arr[7];
    }

    // Handle remaining frames
    for i in remainder_start..frames {
        let interleaved_idx = i * 2;
        output[interleaved_idx] = left_in[i];
        output[interleaved_idx + 1] = right_in[i];
    }

    Ok(())
}

/// Interleave multi-channel audio data.
///
/// Converts planar data `[ch0_f0, ch0_f1, ..., ch1_f0, ch1_f1, ...]`
/// to interleaved format `[ch0_f0, ch1_f0, ch2_f0, ch0_f1, ch1_f1, ...]`.
///
/// # Arguments
/// * `planar` - Input slice in planar format (channel data stored sequentially)
/// * `output` - Output slice for interleaved data (same length as input)
/// * `num_channels` - Number of audio channels
///
/// # Errors
/// Returns error if lengths don't match or input isn't divisible by channel count.
///
/// # Panics
/// Does not panic.
#[inline]
pub fn interleave_multi<T: AudioSample>(
    planar: &[T],
    output: &mut [T],
    num_channels: usize,
) -> AudioSampleResult<()> {
    if planar.len() != output.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "slice_lengths",
            "Input and output slices must have same length",
        )));
    }
    if num_channels == 0 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "num_channels",
            "Number of channels must be greater than 0",
        )));
    }
    if !planar.len().is_multiple_of(num_channels) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "planar_length",
            "Planar data length must be divisible by channel count",
        )));
    }

    // Use optimized stereo path for 2 channels
    if num_channels == 2 {
        return interleave_stereo(planar, output);
    }

    // Generic multi-channel interleave (>2 channels)
    interleave_multi_scalar(planar, output, num_channels)
}

/// Scalar implementation of multi-channel interleave.
fn interleave_multi_scalar<T: AudioSample>(
    planar: &[T],
    output: &mut [T],
    num_channels: usize,
) -> AudioSampleResult<()> {
    let frames = planar.len() / num_channels;

    // Process frame by frame for correct interleaving
    for frame in 0..frames {
        let out_base = frame * num_channels;
        for ch in 0..num_channels {
            let in_idx = ch * frames + frame;
            output[out_base + ch] = planar[in_idx];
        }
    }

    Ok(())
}

// =============================================================================
// CONVENIENCE FUNCTIONS FOR Vec
// =============================================================================

/// Deinterleave stereo Vec, returning a new Vec in planar format.
///
/// This is a convenience function that allocates the output.
///
/// # Errors
/// Returns an error if `interleaved.len()` is not even.
#[inline]
pub fn deinterleave_stereo_vec<T: AudioSample>(interleaved: Vec<T>) -> AudioSampleResult<Vec<T>> {
    if !interleaved.len().is_multiple_of(2) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "interleaved_length",
            "Interleaved stereo data must have even length",
        )));
    }

    let mut output = vec![T::default(); interleaved.len()];
    deinterleave_stereo(&interleaved, &mut output)?;
    Ok(output)
}

/// Deinterleave multi-channel Vec, returning a new Vec in planar format.
///
/// # Errors
/// Returns an error if `num_channels == 0` or if `interleaved.len()` is not divisible by
/// `num_channels`.
#[inline]
pub fn deinterleave_multi_vec<T: AudioSample>(
    interleaved: Vec<T>,
    num_channels: usize,
) -> AudioSampleResult<Vec<T>> {
    if num_channels == 0 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "num_channels",
            "Number of channels must be greater than 0",
        )));
    }
    if !interleaved.len().is_multiple_of(num_channels) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "interleaved_length",
            "Interleaved data length must be divisible by channel count",
        )));
    }

    let mut output = vec![T::default(); interleaved.len()];
    deinterleave_multi(&interleaved, &mut output, num_channels)?;
    Ok(output)
}

/// Interleave stereo Vec, returning a new Vec in interleaved format.
///
/// # Errors
/// Returns an error if `planar.len()` is not even.
#[inline]
pub fn interleave_stereo_vec<T: AudioSample>(planar: Vec<T>) -> AudioSampleResult<Vec<T>> {
    if !planar.len().is_multiple_of(2) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "planar_length",
            "Planar stereo data must have even length",
        )));
    }

    let mut output = vec![T::default(); planar.len()];
    interleave_stereo(&planar, &mut output)?;
    Ok(output)
}

/// Interleave multi-channel Vec, returning a new Vec in interleaved format.
///
/// # Errors
/// Returns an error if `num_channels == 0` or if `planar.len()` is not divisible by
/// `num_channels`.
#[inline]
pub fn interleave_multi_vec<T: AudioSample>(
    planar: Vec<T>,
    num_channels: usize,
) -> AudioSampleResult<Vec<T>> {
    if num_channels == 0 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "num_channels",
            "Number of channels must be greater than 0",
        )));
    }
    if !planar.len().is_multiple_of(num_channels) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "planar_length",
            "Planar data length must be divisible by channel count",
        )));
    }

    let mut output = vec![T::default(); planar.len()];
    interleave_multi(&planar, &mut output, num_channels)?;
    Ok(output)
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
            convert_scalar_unrolled(&_input, &mut _output).unwrap();

            // Verify conversion accuracy
            let expected0: i16 = 0.5f32.convert_to();
            let expected1: i16 = (-0.3f32).convert_to();
            let expected4: i16 = (-1.0f32).convert_to();

            assert_eq!(_output[0], expected0);
            assert_eq!(_output[1], expected1);
            assert_eq!(_output[4], expected4);
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_f32_to_i16_conversion() {
        let input = vec![0.5f32, -0.3, 0.8, 1.0, -1.0, 0.0, 0.1, -0.1, 0.25];
        let mut output = vec![0i16; 9];

        convert_f32_to_i16_simd(&input, &mut output).unwrap();

        // Verify conversion accuracy
        let expected0: i16 = input[0].convert_to();
        let expected1: i16 = input[1].convert_to();
        let expected8: i16 = input[8].convert_to();

        assert_eq!(output[0], expected0);
        assert_eq!(output[1], expected1);
        assert_eq!(output[8], expected8); // Test remainder handling
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_i16_to_f32_conversion() {
        let input = vec![16383i16, -9830, 26214, 32767, -32768, 0, 3276, -3276];
        let mut output = vec![0.0f32; 8];

        convert_i16_to_f32_simd(&input, &mut output).unwrap();

        // Verify conversion accuracy using approx_eq
        use approx_eq::assert_approx_eq;
        let expected0: f32 = input[0].convert_to();
        let expected1: f32 = input[1].convert_to();
        let expected3: f32 = input[3].convert_to();

        assert_approx_eq!(output[0] as f64, expected0 as f64, 1e-5);
        assert_approx_eq!(output[1] as f64, expected1 as f64, 1e-5);
        assert_approx_eq!(output[3] as f64, expected3 as f64, 1e-5);
    }

    #[test]
    fn test_optimized_conversion_dispatch() {
        let input = vec![0.5f32, -0.3, 0.8];
        let mut output = vec![0i16; 3];

        convert(&input, &mut output).unwrap();

        // Should work regardless of SIMD feature
        assert_eq!(output.len(), 3);
        let expected0: i16 = input[0].convert_to();
        assert_eq!(output[0], expected0);
    }

    // =========================================================================
    // DEINTERLEAVE TESTS
    // =========================================================================

    #[test]
    fn test_deinterleave_stereo_f32() {
        // Interleaved: [L0, R0, L1, R1, L2, R2, L3, R3]
        let interleaved = vec![0.1f32, 0.5, 0.2, 0.6, 0.3, 0.7, 0.4, 0.8];
        let mut output = vec![0.0f32; 8];

        deinterleave_stereo(&interleaved, &mut output).unwrap();

        // Planar: [L0, L1, L2, L3, R0, R1, R2, R3]
        assert_eq!(output[0], 0.1); // L0
        assert_eq!(output[1], 0.2); // L1
        assert_eq!(output[2], 0.3); // L2
        assert_eq!(output[3], 0.4); // L3
        assert_eq!(output[4], 0.5); // R0
        assert_eq!(output[5], 0.6); // R1
        assert_eq!(output[6], 0.7); // R2
        assert_eq!(output[7], 0.8); // R3
    }

    #[test]
    fn test_deinterleave_stereo_i16() {
        let interleaved = vec![100i16, 200, 300, 400, 500, 600];
        let mut output = vec![0i16; 6];

        deinterleave_stereo(&interleaved, &mut output).unwrap();

        // Planar: [L0, L1, L2, R0, R1, R2]
        assert_eq!(output[0], 100); // L0
        assert_eq!(output[1], 300); // L1
        assert_eq!(output[2], 500); // L2
        assert_eq!(output[3], 200); // R0
        assert_eq!(output[4], 400); // R1
        assert_eq!(output[5], 600); // R2
    }

    #[test]
    fn test_deinterleave_stereo_remainder() {
        // Test with non-multiple-of-4 frames (5 frames = 10 samples)
        let interleaved = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut output = vec![0.0f32; 10];

        deinterleave_stereo(&interleaved, &mut output).unwrap();

        // Left channel: [1, 3, 5, 7, 9], Right channel: [2, 4, 6, 8, 10]
        assert_eq!(&output[0..5], &[1.0, 3.0, 5.0, 7.0, 9.0]);
        assert_eq!(&output[5..10], &[2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_deinterleave_multi_3ch() {
        // Interleaved 3-channel: [ch0_f0, ch1_f0, ch2_f0, ch0_f1, ch1_f1, ch2_f1]
        let interleaved = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![0.0f32; 6];

        deinterleave_multi(&interleaved, &mut output, 3).unwrap();

        // Planar: [ch0_f0, ch0_f1, ch1_f0, ch1_f1, ch2_f0, ch2_f1]
        assert_eq!(output[0], 1.0); // ch0_f0
        assert_eq!(output[1], 4.0); // ch0_f1
        assert_eq!(output[2], 2.0); // ch1_f0
        assert_eq!(output[3], 5.0); // ch1_f1
        assert_eq!(output[4], 3.0); // ch2_f0
        assert_eq!(output[5], 6.0); // ch2_f1
    }

    #[test]
    fn test_deinterleave_multi_6ch() {
        // 5.1 surround: 6 channels, 2 frames
        let interleaved: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let mut output = vec![0.0f32; 12];

        deinterleave_multi(&interleaved, &mut output, 6).unwrap();

        // Each channel should have 2 samples
        // ch0: [1, 7], ch1: [2, 8], ch2: [3, 9], ch3: [4, 10], ch4: [5, 11], ch5: [6, 12]
        assert_eq!(&output[0..2], &[1.0, 7.0]); // ch0
        assert_eq!(&output[2..4], &[2.0, 8.0]); // ch1
        assert_eq!(&output[4..6], &[3.0, 9.0]); // ch2
        assert_eq!(&output[6..8], &[4.0, 10.0]); // ch3
        assert_eq!(&output[8..10], &[5.0, 11.0]); // ch4
        assert_eq!(&output[10..12], &[6.0, 12.0]); // ch5
    }

    // =========================================================================
    // INTERLEAVE TESTS
    // =========================================================================

    #[test]
    fn test_interleave_stereo_f32() {
        // Planar: [L0, L1, L2, L3, R0, R1, R2, R3]
        let planar = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut output = vec![0.0f32; 8];

        interleave_stereo(&planar, &mut output).unwrap();

        // Interleaved: [L0, R0, L1, R1, L2, R2, L3, R3]
        assert_eq!(output[0], 0.1); // L0
        assert_eq!(output[1], 0.5); // R0
        assert_eq!(output[2], 0.2); // L1
        assert_eq!(output[3], 0.6); // R1
        assert_eq!(output[4], 0.3); // L2
        assert_eq!(output[5], 0.7); // R2
        assert_eq!(output[6], 0.4); // L3
        assert_eq!(output[7], 0.8); // R3
    }

    #[test]
    fn test_interleave_stereo_i16() {
        // Planar: [L0, L1, L2, R0, R1, R2]
        let planar = vec![100i16, 300, 500, 200, 400, 600];
        let mut output = vec![0i16; 6];

        interleave_stereo(&planar, &mut output).unwrap();

        // Interleaved: [L0, R0, L1, R1, L2, R2]
        assert_eq!(output[0], 100); // L0
        assert_eq!(output[1], 200); // R0
        assert_eq!(output[2], 300); // L1
        assert_eq!(output[3], 400); // R1
        assert_eq!(output[4], 500); // L2
        assert_eq!(output[5], 600); // R2
    }

    #[test]
    fn test_interleave_multi_3ch() {
        // Planar 3-channel: [ch0_f0, ch0_f1, ch1_f0, ch1_f1, ch2_f0, ch2_f1]
        let planar = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        let mut output = vec![0.0f32; 6];

        interleave_multi(&planar, &mut output, 3).unwrap();

        // Interleaved: [ch0_f0, ch1_f0, ch2_f0, ch0_f1, ch1_f1, ch2_f1]
        assert_eq!(output[0], 1.0); // ch0_f0
        assert_eq!(output[1], 2.0); // ch1_f0
        assert_eq!(output[2], 3.0); // ch2_f0
        assert_eq!(output[3], 4.0); // ch0_f1
        assert_eq!(output[4], 5.0); // ch1_f1
        assert_eq!(output[5], 6.0); // ch2_f1
    }

    // =========================================================================
    // ROUNDTRIP TESTS
    // =========================================================================

    #[test]
    fn test_stereo_roundtrip_f32() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut deinterleaved = vec![0.0f32; 8];
        let mut reinterleaved = vec![0.0f32; 8];

        deinterleave_stereo(&original, &mut deinterleaved).unwrap();
        interleave_stereo(&deinterleaved, &mut reinterleaved).unwrap();

        assert_eq!(original, reinterleaved);
    }

    #[test]
    fn test_stereo_roundtrip_i16() {
        let original: Vec<i16> = (1..=16).map(|x| x as i16 * 100).collect();
        let mut deinterleaved = vec![0i16; 16];
        let mut reinterleaved = vec![0i16; 16];

        deinterleave_stereo(&original, &mut deinterleaved).unwrap();
        interleave_stereo(&deinterleaved, &mut reinterleaved).unwrap();

        assert_eq!(original, reinterleaved);
    }

    #[test]
    fn test_multi_roundtrip_6ch() {
        let original: Vec<f32> = (1..=60).map(|x| x as f32).collect(); // 6 channels, 10 frames
        let mut deinterleaved = vec![0.0f32; 60];
        let mut reinterleaved = vec![0.0f32; 60];

        deinterleave_multi(&original, &mut deinterleaved, 6).unwrap();
        interleave_multi(&deinterleaved, &mut reinterleaved, 6).unwrap();

        assert_eq!(original, reinterleaved);
    }

    // =========================================================================
    // VEC CONVENIENCE FUNCTION TESTS
    // =========================================================================

    #[test]
    fn test_deinterleave_stereo_vec_fn() {
        let interleaved = vec![1.0f32, 2.0, 3.0, 4.0];
        let planar = deinterleave_stereo_vec(interleaved).unwrap();

        assert_eq!(planar, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_interleave_stereo_vec_fn() {
        let planar = vec![1.0f32, 3.0, 2.0, 4.0];
        let interleaved = interleave_stereo_vec(planar).unwrap();

        assert_eq!(interleaved, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // =========================================================================
    // ERROR HANDLING TESTS
    // =========================================================================

    #[test]
    fn test_deinterleave_stereo_length_mismatch() {
        let interleaved = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 2]; // Wrong size

        let result = deinterleave_stereo(&interleaved, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_deinterleave_stereo_odd_length() {
        let interleaved = vec![1.0f32, 2.0, 3.0]; // Odd length
        let mut output = vec![0.0f32; 3];

        let result = deinterleave_stereo(&interleaved, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_deinterleave_multi_zero_channels() {
        let interleaved = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];

        let result = deinterleave_multi(&interleaved, &mut output, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_deinterleave_multi_not_divisible() {
        let interleaved = vec![1.0f32, 2.0, 3.0, 4.0, 5.0]; // 5 samples, 3 channels
        let mut output = vec![0.0f32; 5];

        let result = deinterleave_multi(&interleaved, &mut output, 3);
        assert!(result.is_err());
    }
}
