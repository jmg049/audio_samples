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
//! The high-level [`convert`] API matches the semantics of the crate's
//! [`ConvertTo`] conversions (clamp + symmetric scale + truncate + saturate for
//! float → int, and asymmetric scaling for int → float).
#![allow(unused)]

use std::num::NonZeroU32;

use non_empty_slice::{NonEmptySlice, NonEmptyVec, non_empty_vec};
#[cfg(feature = "simd")]
use wide::f32x8;

use crate::{AudioSample, AudioSampleError, AudioSampleResult, ConvertTo, ParameterError};

/// SIMD-optimized `f32` → `i16` conversion.
///
/// # Behavior
/// Matches the crate's [`ConvertTo<i16>`] semantics for `f32`:
/// - clamps the input to $[-1.0, 1.0]$
/// - scales symmetrically by `i16::MAX` (`±1.0 → ±32767`)
/// - truncates toward zero and saturates to the destination range
///
/// # Errors
/// Returns an error if `input.len() != output.len()`.
#[cfg(feature = "simd")]
#[inline]
pub fn convert_f32_to_i16_simd(input: &[f32], output: &mut [i16]) -> AudioSampleResult<()> {
    if input.len() != output.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "slice_lengths",
            "Input and output slices must have same length",
        )));
    }

    let chunks = input.len() / 8;

    // Genuine vector path: clamp, scale by i16::MAX, truncate toward zero, narrow.
    //
    // Parity with the scalar `impl_float_to_int` (`traits.rs`): the scalar path
    // is `v = source.max(-1).min(1); scaled = v * 32767.0; (scaled.to_int_unchecked()) as i16`,
    // i.e. clamp + symmetric scale + truncate-toward-zero. `wide`'s `trunc_int`
    // truncates toward zero exactly like `to_int_unchecked`, and because the
    // clamped+scaled value is always in `[-32767, 32767]` the `i32 -> i16` narrow
    // is lossless and order-preserving (verified bit-identical for all inputs,
    // including values whose fractional part >= 0.5 that would round differently).
    for i in 0..chunks {
        let start_idx = i * 8;

        let chunk: [f32; 8] = input[start_idx..start_idx + 8].try_into().unwrap();
        let f32_vec = f32x8::from(chunk);
        let clamped = f32_vec.max(f32x8::splat(-1.0)).min(f32x8::splat(1.0));
        let scaled = clamped * f32x8::splat(i16::MAX as f32);
        let ints = scaled.trunc_int().to_array();
        for (j, &v) in ints.iter().enumerate() {
            // `v` is guaranteed in [-32767, 32767]; `as i16` matches scalar exactly.
            output[start_idx + j] = v as i16;
        }
    }

    // Handle remaining samples with the canonical scalar conversion.
    let start_remainder = chunks * 8;
    for i in start_remainder..input.len() {
        output[i] = input[i].convert_to();
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
#[inline]
pub fn convert_i16_to_f32_simd(input: &[i16], output: &mut [f32]) -> AudioSampleResult<()> {
    if input.len() != output.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "slice_lengths",
            "Input and output slices must have same length",
        )));
    }

    let chunks = input.len() / 8;

    // Genuine vector path: asymmetric scaling with a branchless blend.
    //
    // Parity with the scalar `impl_int_to_float` (`traits.rs`):
    // `if v < 0 { (v as f32) / 32768.0 } else { (v as f32) / 32767.0 }`.
    // We compute both quotients per-lane (IEEE division is bit-identical to the
    // scalar division) and select with the same `v < 0` predicate via `blend`,
    // so the result is bit-for-bit identical to scalar (verified across the full
    // input range, including negative / zero / i16::MIN / i16::MAX).
    let neg_div = f32x8::splat(-(i16::MIN as f32)); // 32768.0
    let pos_div = f32x8::splat(i16::MAX as f32); // 32767.0
    for i in 0..chunks {
        let start_idx = i * 8;

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

        let neg = f32_values / neg_div;
        let pos = f32_values / pos_div;
        let mask = f32_values.simd_lt(f32x8::splat(0.0));
        let result = mask.blend(neg, pos).to_array();
        output[start_idx..start_idx + 8].copy_from_slice(&result);
    }

    // Handle remaining samples with the canonical scalar conversion.
    let start_remainder = chunks * 8;
    for i in start_remainder..input.len() {
        output[i] = input[i].convert_to();
    }

    Ok(())
}

/// SIMD-optimized `f32` → `i32` conversion.
///
/// # Behavior
/// Matches the crate's [`ConvertTo<i32>`] semantics for `f32`:
/// - clamps the input to $[-1.0, 1.0]$
/// - scales symmetrically by `i32::MAX`
/// - truncates toward zero and saturates to the destination range
///
/// # Errors
/// Returns an error if `input.len() != output.len()`.
#[cfg(feature = "simd")]
#[inline]
pub fn convert_f32_to_i32_simd(input: &[f32], output: &mut [i32]) -> AudioSampleResult<()> {
    if input.len() != output.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "slice_lengths",
            "Input and output slices must have same length",
        )));
    }

    let chunks = input.len() / 8;

    // Genuine vector path: clamp, scale by i32::MAX, truncate toward zero (saturating).
    //
    // Parity with the scalar `impl_float_to_large_int` (`traits.rs`):
    // `v = source.max(-1).min(1); scaled = v * (i32::MAX as f32);
    //  scaled.clamp(i32::MIN as f32, i32::MAX as f32) as i32`.
    // The subtle case is `source == 1.0`: `1.0 * 2147483647.0` rounds up to
    // `2147483648.0` in f32, which the scalar clamps to `i32::MAX as f32`
    // (== 2147483648.0) and then `as i32` saturates to `2147483647`. `wide`'s
    // `trunc_int` is the SATURATING truncation (out-of-range -> i32::MIN/MAX),
    // so it yields exactly `2147483647` there — matching scalar. (Note:
    // `fast_trunc_int` would WRAP to i32::MIN here, so it must NOT be used.)
    // Verified bit-identical across the full input range.
    for i in 0..chunks {
        let start_idx = i * 8;

        let chunk: [f32; 8] = input[start_idx..start_idx + 8].try_into().unwrap();
        let f32_vec = f32x8::from(chunk);
        let clamped = f32_vec.max(f32x8::splat(-1.0)).min(f32x8::splat(1.0));
        let scaled = clamped * f32x8::splat(i32::MAX as f32);
        let ints = scaled.trunc_int().to_array();
        output[start_idx..start_idx + 8].copy_from_slice(&ints);
    }

    // Handle remainder with the canonical scalar conversion.
    let start_remainder = chunks * 8;
    for i in start_remainder..input.len() {
        output[i] = input[i].convert_to();
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
#[inline]
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
        *o = (*i).convert_to();
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
#[inline]
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
#[inline]
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
        Ok(deinterleave_stereo_simd(interleaved, output))
    }

    #[cfg(not(feature = "simd"))]
    {
        deinterleave_stereo_scalar(interleaved, output);
        Ok(())
    }
}

/// Scalar implementation of stereo deinterleave with loop unrolling.
#[inline]
fn deinterleave_stereo_scalar<T: AudioSample>(interleaved: &[T], output: &mut [T]) {
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
}

/// Stereo deinterleave under the `simd` feature.
///
/// Deinterleaving is pure data movement with no arithmetic, so there is no
/// vector-math win here: the `wide` crate exposes no shuffle/unpack intrinsic
/// that beats LLVM's own auto-vectorisation of the contiguous scalar copies.
/// The previous "SIMD" code loaded a vector only to immediately `.to_array()`
/// and scatter element-by-element — pure overhead. We therefore route to the
/// proven scalar implementation, which is bit-identical by construction and at
/// least as fast (DECISION RULE: route to scalar when there is no genuine
/// vector win).
#[cfg(feature = "simd")]
#[inline]
fn deinterleave_stereo_simd<T: AudioSample>(interleaved: &[T], output: &mut [T]) {
    deinterleave_stereo_scalar(interleaved, output)
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
    interleaved: &NonEmptySlice<T>,
    output: &mut NonEmptySlice<T>,
    num_channels: NonZeroU32,
) -> AudioSampleResult<()> {
    if interleaved.len() != output.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "slice_lengths",
            "Input and output slices must have same length",
        )));
    }
    if !interleaved
        .len()
        .get()
        .is_multiple_of(num_channels.get() as usize)
    {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "interleaved_length",
            "Interleaved data length must be divisible by channel count",
        )));
    }

    // Use optimized stereo path for 2 channels
    if num_channels.get() == 2 {
        return deinterleave_stereo(interleaved, output);
    }

    // Generic multi-channel deinterleave (>2 channels)
    deinterleave_multi_scalar(interleaved, output, num_channels);
    Ok(())
}

/// Scalar implementation of multi-channel deinterleave.
///
/// For >2 channels, we use a cache-friendly approach that processes
/// one channel at a time to maximize sequential writes.
fn deinterleave_multi_scalar<T: AudioSample>(
    interleaved: &NonEmptySlice<T>,
    output: &mut NonEmptySlice<T>,
    num_channels: NonZeroU32,
) {
    let frames = interleaved.len().get() / num_channels.get() as usize;
    // Process each channel sequentially for better cache locality on writes
    for ch in 0..num_channels.get() as usize {
        let out_start = ch * frames;
        let out_slice = &mut output[out_start..out_start + frames];

        for frame in 0..frames {
            out_slice[frame] = interleaved[frame * num_channels.get() as usize + ch];
        }
    }
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
        Ok(interleave_stereo_simd(planar, output))
    }

    #[cfg(not(feature = "simd"))]
    {
        interleave_stereo_scalar(planar, output);
        Ok(())
    }
}

/// Scalar implementation of stereo interleave with loop unrolling.
#[inline]
fn interleave_stereo_scalar<T: AudioSample>(planar: &[T], output: &mut [T]) {
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
}

/// Stereo interleave under the `simd` feature.
///
/// Like deinterleave, this is pure data movement; `wide` offers no interleave
/// shuffle that beats the auto-vectorised scalar copy, and the old vector
/// round-trip (`from([...]).to_array()` then scalar scatter) was pure overhead.
/// Route to the proven scalar implementation — bit-identical and no slower
/// (DECISION RULE: route to scalar when there is no genuine vector win).
#[cfg(feature = "simd")]
#[inline]
fn interleave_stereo_simd<T: AudioSample>(planar: &[T], output: &mut [T]) {
    interleave_stereo_scalar(planar, output)
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
    planar: &NonEmptySlice<T>,
    output: &mut NonEmptySlice<T>,
    num_channels: NonZeroU32,
) -> AudioSampleResult<()> {
    if planar.len() != output.len() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "slice_lengths",
            "Input and output slices must have same length",
        )));
    }
    if !planar
        .len()
        .get()
        .is_multiple_of(num_channels.get() as usize)
    {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "planar_length",
            "Planar data length must be divisible by channel count",
        )));
    }

    // Use optimized stereo path for 2 channels
    if num_channels.get() == 2 {
        return interleave_stereo(planar, output);
    }

    // Generic multi-channel interleave (>2 channels)
    interleave_multi_scalar(planar, output, num_channels);
    Ok(())
}

/// Scalar implementation of multi-channel interleave.
fn interleave_multi_scalar<T: AudioSample>(
    planar: &NonEmptySlice<T>,
    output: &mut NonEmptySlice<T>,
    num_channels: NonZeroU32,
) {
    let frames = planar.len().get() / num_channels.get() as usize;
    // Process frame by frame for correct interleaving
    for frame in 0..frames {
        let out_base = frame * num_channels.get() as usize;
        for ch in 0..num_channels.get() as usize {
            let in_idx = ch * frames + frame;
            output[out_base + ch] = planar[in_idx];
        }
    }
}

// CONVENIENCE FUNCTIONS FOR Vec
// =============================================================================

/// Deinterleave stereo Vec, returning a new Vec in planar format.
///
/// This is a convenience function that allocates the output.
///
/// # Errors
/// Returns an error if `interleaved.len()` is not even.
#[inline]
pub fn deinterleave_stereo_vec<T: AudioSample>(
    interleaved: &NonEmptyVec<T>,
) -> AudioSampleResult<NonEmptyVec<T>> {
    if !interleaved.len().get().is_multiple_of(2) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "interleaved_length",
            "Interleaved stereo data must have even length",
        )));
    }

    let mut output = non_empty_vec![T::default(); interleaved.len()];
    deinterleave_stereo(interleaved, &mut output)?;
    Ok(output)
}

/// Deinterleave multi-channel Vec, returning a new Vec in planar format.
///
/// # Errors
/// Returns an error if `num_channels == 0` or if `interleaved.len()` is not divisible by
/// `num_channels`.
#[inline]
pub fn deinterleave_multi_vec<T: AudioSample>(
    interleaved: &NonEmptySlice<T>,
    num_channels: NonZeroU32,
) -> AudioSampleResult<NonEmptyVec<T>> {
    if !interleaved
        .len()
        .get()
        .is_multiple_of(num_channels.get() as usize)
    {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "interleaved_length",
            "Interleaved data length must be divisible by channel count",
        )));
    }

    let mut output = non_empty_vec![T::default(); interleaved.len()];
    deinterleave_multi(interleaved, &mut output, num_channels)?;
    Ok(output)
}

/// Interleave stereo Vec, returning a new Vec in interleaved format.
///
/// # Errors
/// Returns an error if `planar.len()` is not even.
#[inline]
pub fn interleave_stereo_vec<T: AudioSample>(
    planar: &NonEmptyVec<T>,
) -> AudioSampleResult<NonEmptyVec<T>> {
    if !planar.len().get().is_multiple_of(2) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "planar_length",
            "Planar stereo data must have even length",
        )));
    }

    let mut output = non_empty_vec![T::default(); planar.len()];
    interleave_stereo(planar, &mut output)?;
    Ok(output)
}

/// Interleave multi-channel Vec, returning a new Vec in interleaved format.
///
/// # Errors
/// Returns an error if `num_channels == 0` or if `planar.len()` is not divisible by
/// `num_channels`.
#[inline]
pub fn interleave_multi_vec<T: AudioSample>(
    planar: &NonEmptyVec<T>,
    num_channels: NonZeroU32,
) -> AudioSampleResult<NonEmptyVec<T>> {
    if !planar
        .len()
        .get()
        .is_multiple_of(num_channels.get() as usize)
    {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "planar_length",
            "Planar data length must be divisible by channel count",
        )));
    }

    let mut output = non_empty_vec![T::default(); planar.len()];
    interleave_multi(planar, &mut output, num_channels)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use non_empty_iter::TryIntoNonEmptyIterator;

    use super::*;

    #[test]
    fn test_scalar_conversion() {
        let _input = non_empty_vec![0.5f32, -0.3, 0.8, 1.0, -1.0];
        let mut _output = non_empty_vec![0i16; crate::nzu!(5)];

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
        let input = non_empty_vec![0.5f32, -0.3, 0.8, 1.0, -1.0, 0.0, 0.1, -0.1, 0.25];
        let mut output = non_empty_vec![0i16; crate::nzu!(9)];

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
        let input = non_empty_vec![16383i16, -9830, 26214, 32767, -32768, 0, 3276, -3276];
        let mut output = non_empty_vec![0.0f32; crate::nzu!(8)];

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
        let input = non_empty_vec![0.5f32, -0.3, 0.8];
        let mut output = non_empty_vec![0i16; crate::nzu!(3)];

        convert(&input, &mut output).unwrap();

        // Should work regardless of SIMD feature
        assert_eq!(output.len().get(), 3);
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
        let interleaved = non_empty_vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = non_empty_vec![0.0f32; crate::nzu!(6)];

        deinterleave_multi(&interleaved, &mut output, NonZeroU32::new(3).unwrap());

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
        let interleaved = NonEmptyVec::new(interleaved).unwrap();
        let mut output = non_empty_vec![0.0f32; crate::nzu!(12)];

        deinterleave_multi(&interleaved, &mut output, NonZeroU32::new(6).unwrap()).unwrap();

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
        let planar = non_empty_vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let mut output = non_empty_vec![0.0f32; crate::nzu!(8)];

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
        let planar = non_empty_vec![100i16, 300, 500, 200, 400, 600];
        let mut output = non_empty_vec![0i16; crate::nzu!(6)];

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
        let planar = non_empty_vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        let mut output = non_empty_vec![0.0f32; crate::nzu!(6)];

        interleave_multi(&planar, &mut output, NonZeroU32::new(3).unwrap()).unwrap();

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
        let original = NonEmptyVec::new(original).unwrap();
        let mut deinterleaved = non_empty_vec![0.0f32; crate::nzu!(60)];
        let mut reinterleaved = non_empty_vec![0.0f32; crate::nzu!(60)];
        let num_channels = NonZeroU32::new(6).unwrap();

        deinterleave_multi(&original, &mut deinterleaved, num_channels).unwrap();
        interleave_multi(&deinterleaved, &mut reinterleaved, num_channels).unwrap();

        assert_eq!(original, reinterleaved);
    }

    // =========================================================================
    // VEC CONVENIENCE FUNCTION TESTS
    // =========================================================================

    #[test]
    fn test_deinterleave_stereo_vec_fn() {
        let interleaved = non_empty_vec![1.0f32, 2.0, 3.0, 4.0];
        let planar = deinterleave_stereo_vec(&interleaved).unwrap();

        assert_eq!(planar, non_empty_vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_interleave_stereo_vec_fn() {
        let planar = non_empty_vec![1.0f32, 3.0, 2.0, 4.0];
        let interleaved = interleave_stereo_vec(&planar).unwrap();

        assert_eq!(interleaved, non_empty_vec![1.0, 2.0, 3.0, 4.0]);
    }

    // =========================================================================
    // ERROR HANDLING TESTS
    // =========================================================================

    #[test]
    fn test_deinterleave_stereo_length_mismatch() {
        let interleaved = non_empty_vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output = non_empty_vec![0.0f32; crate::nzu!(2)]; // Wrong size

        let result = deinterleave_stereo(&interleaved, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_deinterleave_stereo_odd_length() {
        let interleaved = non_empty_vec![1.0f32, 2.0, 3.0]; // Odd length
        let mut output = non_empty_vec![0.0f32; crate::nzu!(3)];

        let result = deinterleave_stereo(&interleaved, &mut output);
        assert!(result.is_err());
    }

    #[test]
    fn test_deinterleave_multi_not_divisible() {
        let interleaved = non_empty_vec![1.0f32, 2.0, 3.0, 4.0, 5.0]; // 5 samples, 3 channels
        let mut output = non_empty_vec![0.0f32; crate::nzu!(5)];

        let result = deinterleave_multi(&interleaved, &mut output, NonZeroU32::new(3).unwrap());
        assert!(result.is_err());
    }
}

// =============================================================================
// SIMD <-> SCALAR PARITY (bit-identical) TESTS
// =============================================================================
//
// These tests assert that, with `feature = "simd"` enabled, every affected
// conversion and the interleave/deinterleave entry points produce results that
// are *bit-for-bit identical* to the scalar reference. Inputs deliberately
// include negative, zero, max, and fractional values that would ROUND
// differently than they TRUNCATE (e.g. x where x*MAX has fractional part
// >= 0.5), since the scalar path truncates toward zero.
#[cfg(all(test, feature = "simd"))]
mod simd_parity_tests {
    use super::*;

    /// Inline scalar reference for f32 -> i16 (mirrors `impl_float_to_int`).
    fn scalar_f32_to_i16(source: f32) -> i16 {
        let v = source.max(-1.0).min(1.0);
        let scaled = v * (i16::MAX as f32);
        let as_i32: i32 = unsafe { scaled.to_int_unchecked() };
        as_i32 as i16
    }

    /// Inline scalar reference for f32 -> i32 (mirrors `impl_float_to_large_int`).
    fn scalar_f32_to_i32(source: f32) -> i32 {
        let v = source.max(-1.0).min(1.0);
        let scaled = v * (i32::MAX as f32);
        scaled.clamp(i32::MIN as f32, i32::MAX as f32) as i32
    }

    /// Inline scalar reference for i16 -> f32 (mirrors `impl_int_to_float`).
    fn scalar_i16_to_f32(v: i16) -> f32 {
        if v < 0 {
            (v as f32) / (-(i16::MIN as f32))
        } else {
            (v as f32) / (i16::MAX as f32)
        }
    }

    /// A spread of f32 inputs covering negative, zero, max, sub-LSB, and values
    /// whose scaled fractional part is >= 0.5 (round-vs-truncate trap), plus
    /// out-of-range and NaN to exercise clamping.
    fn f32_probe_inputs() -> Vec<f32> {
        let mut v = vec![
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            0.25,
            -0.25,
            0.99999994, // largest f32 < 1.0
            -0.99999994,
            2.0,  // out of range (clamps to 1.0)
            -2.0, // out of range (clamps to -1.0)
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        // Values chosen so that x * 32767 / x * i32::MAX land near n + {0.5, 0.6, 0.9}
        // i.e. round-up but truncate-down.
        for k in 0..200i32 {
            let x = (k as f32) / 199.0; // 0.0 ..= 1.0
            v.push(x);
            v.push(-x);
            // nudge to create fractional parts >= 0.5 after scaling
            v.push(x + 0.4999999 / (i16::MAX as f32));
            v.push(-(x + 0.5000001 / (i16::MAX as f32)));
        }
        v
    }

    #[test]
    fn parity_f32_to_i16() {
        let input = f32_probe_inputs();
        let mut simd_out = vec![0i16; input.len()];
        convert_f32_to_i16_simd(&input, &mut simd_out).unwrap();
        for (i, &x) in input.iter().enumerate() {
            assert_eq!(
                simd_out[i],
                scalar_f32_to_i16(x),
                "f32->i16 mismatch at idx {i} for input {x} (bits {:08x})",
                x.to_bits()
            );
            // Also verify against the crate's canonical scalar conversion.
            let canonical: i16 = x.convert_to();
            assert_eq!(simd_out[i], canonical, "f32->i16 vs ConvertTo at idx {i}");
        }
    }

    #[test]
    fn parity_f32_to_i32() {
        let input = f32_probe_inputs();
        let mut simd_out = vec![0i32; input.len()];
        convert_f32_to_i32_simd(&input, &mut simd_out).unwrap();
        for (i, &x) in input.iter().enumerate() {
            assert_eq!(
                simd_out[i],
                scalar_f32_to_i32(x),
                "f32->i32 mismatch at idx {i} for input {x} (bits {:08x})",
                x.to_bits()
            );
            let canonical: i32 = x.convert_to();
            assert_eq!(simd_out[i], canonical, "f32->i32 vs ConvertTo at idx {i}");
        }
    }

    #[test]
    fn parity_i16_to_f32() {
        // Exhaustive over the entire i16 domain.
        let input: Vec<i16> = (i16::MIN..=i16::MAX).collect();
        let mut simd_out = vec![0.0f32; input.len()];
        convert_i16_to_f32_simd(&input, &mut simd_out).unwrap();
        for (i, &x) in input.iter().enumerate() {
            // Bit-for-bit (not approx): the vector blend must match the scalar
            // branch exactly.
            assert_eq!(
                simd_out[i].to_bits(),
                scalar_i16_to_f32(x).to_bits(),
                "i16->f32 mismatch at idx {i} for input {x}"
            );
            let canonical: f32 = x.convert_to();
            assert_eq!(
                simd_out[i].to_bits(),
                canonical.to_bits(),
                "i16->f32 vs ConvertTo at idx {i}"
            );
        }
    }

    #[test]
    fn parity_convert_dispatch() {
        // The high-level `convert` entry point (which dispatches to simd) must
        // match the scalar-unrolled path for the supported type pairs.
        let f32_in: Vec<f32> = f32_probe_inputs();

        let mut simd_i16 = vec![0i16; f32_in.len()];
        let mut scal_i16 = vec![0i16; f32_in.len()];
        convert(&f32_in, &mut simd_i16).unwrap();
        convert_scalar_unrolled(&f32_in, &mut scal_i16).unwrap();
        assert_eq!(simd_i16, scal_i16, "convert f32->i16 vs scalar unrolled");

        let mut simd_i32 = vec![0i32; f32_in.len()];
        let mut scal_i32 = vec![0i32; f32_in.len()];
        convert(&f32_in, &mut simd_i32).unwrap();
        convert_scalar_unrolled(&f32_in, &mut scal_i32).unwrap();
        assert_eq!(simd_i32, scal_i32, "convert f32->i32 vs scalar unrolled");

        let i16_in: Vec<i16> = (i16::MIN..=i16::MAX).step_by(7).collect();
        let mut simd_f32 = vec![0.0f32; i16_in.len()];
        let mut scal_f32 = vec![0.0f32; i16_in.len()];
        convert(&i16_in, &mut simd_f32).unwrap();
        convert_scalar_unrolled(&i16_in, &mut scal_f32).unwrap();
        for i in 0..i16_in.len() {
            assert_eq!(
                simd_f32[i].to_bits(),
                scal_f32[i].to_bits(),
                "convert i16->f32 vs scalar unrolled at idx {i}"
            );
        }
    }

    #[test]
    fn parity_interleave_deinterleave_stereo() {
        // Lengths chosen to exercise the SIMD chunk boundary and remainder.
        for &frames in &[1usize, 3, 4, 5, 8, 9, 17, 33] {
            let n = frames * 2;

            // f32
            let planar_f32: Vec<f32> = (0..n).map(|i| (i as f32) * 0.013 - 0.7).collect();
            let mut simd_out = vec![0.0f32; n];
            let mut scal_out = vec![0.0f32; n];
            interleave_stereo(&planar_f32, &mut simd_out).unwrap();
            interleave_stereo_scalar(&planar_f32, &mut scal_out);
            assert_eq!(simd_out, scal_out, "interleave_stereo f32 frames={frames}");

            let inter_f32 = simd_out.clone();
            let mut d_simd = vec![0.0f32; n];
            let mut d_scal = vec![0.0f32; n];
            deinterleave_stereo(&inter_f32, &mut d_simd).unwrap();
            deinterleave_stereo_scalar(&inter_f32, &mut d_scal);
            assert_eq!(d_simd, d_scal, "deinterleave_stereo f32 frames={frames}");

            // i16 (non-f32 type — used to hit the fallback path)
            let planar_i16: Vec<i16> = (0..n).map(|i| (i as i16) * 37 - 100).collect();
            let mut s16 = vec![0i16; n];
            let mut c16 = vec![0i16; n];
            interleave_stereo(&planar_i16, &mut s16).unwrap();
            interleave_stereo_scalar(&planar_i16, &mut c16);
            assert_eq!(s16, c16, "interleave_stereo i16 frames={frames}");
        }
    }

    #[test]
    fn parity_interleave_channels_entrypoint() {
        use crate::AudioSamples;
        use crate::operations::traits::AudioChannelOps;
        use crate::sample_rate;
        use ndarray::Array1;
        use non_empty_slice::NonEmptySlice;

        // Build per-type channel sets and compare the simd entry point against
        // the scalar base implementation for 2, 3, and 4 channels.
        macro_rules! check_type {
            ($t:ty, $gen:expr) => {{
                for &nch in &[2usize, 3, 4] {
                    let len = 37usize; // not a multiple of 16/8/4 -> exercises remainder
                    let chans: Vec<AudioSamples<'static, $t>> = (0..nch)
                        .map(|c| {
                            let data: Vec<$t> =
                                (0..len).map(|i| $gen(c, i)).collect();
                            AudioSamples::new_mono(Array1::from(data), sample_rate!(44100))
                                .unwrap()
                        })
                        .collect();
                    let slice = NonEmptySlice::new(&chans).unwrap();

                    let via_simd =
                        <AudioSamples<'_, $t> as AudioChannelOps>::interleave_channels(slice)
                            .unwrap();

                    // Inline scalar reference for interleaving: row ch holds the
                    // interleaved frame-major stream, identical to the scalar
                    // base implementation's semantics.
                    let a = via_simd.as_multi_channel().unwrap();
                    assert_eq!(a.shape(), &[nch, len]);
                    let mut expected: Vec<$t> = Vec::with_capacity(nch * len);
                    for i in 0..len {
                        for c in 0..nch {
                            expected.push(chans[c].as_slice().unwrap()[i]);
                        }
                    }
                    let mut k = 0usize;
                    for ch in 0..nch {
                        for s in 0..len {
                            assert_eq!(
                                a[(ch, s)],
                                expected[k],
                                "interleave_channels {} nch={} at ({},{})",
                                stringify!($t),
                                nch,
                                ch,
                                s
                            );
                            k += 1;
                        }
                    }
                }
            }};
        }

        check_type!(i16, |c: usize, i: usize| (c as i16) * 1000 + i as i16);
        check_type!(i32, |c: usize, i: usize| (c as i32) * 100_000 + i as i32);
        check_type!(f32, |c: usize, i: usize| (c as f32) + (i as f32) * 0.01);
        check_type!(f64, |c: usize, i: usize| (c as f64) + (i as f64) * 0.001);
    }
}
