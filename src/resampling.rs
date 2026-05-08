//! Module for handling audio sample resampling operations.
//! Uses linear interpolation for the Fast path and rubato's FFT resampler for Medium/High quality.

use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ParameterError,
    ProcessingError,
    operations::types::ResamplingQuality,
    repr::{ChannelCount, SampleRate},
    traits::StandardSample,
};
use audioadapter_buffers::direct::SequentialSliceOfVecs;
use non_empty_slice::NonEmptyVec;
use rubato::{Fft, FixedSync, Resampler};
use std::cell::RefCell;
use std::collections::HashMap;

// Chunk sizes for rubato's Fft synchronous resampler.
// The Fft resampler snaps to the nearest integer multiple of (in_sr / gcd(in_sr, out_sr)),
// so these are targets; the actual chunk will be slightly larger.
// Larger chunks amortise per-FFT overhead and improve throughput for long signals.
const FFT_MEDIUM_CHUNK: usize = 4096;
const FFT_HIGH_CHUNK: usize = 16384;

// ---------------------------------------------------------------------------
// Per-thread FFT resampler cache
// ---------------------------------------------------------------------------
//
// Building a rubato Fft resampler computes FFT plans and filter tables at
// construction time.  Caching one Fft<f32> per (in_sr, out_sr, channels,
// quality) tuple avoids that overhead on repeated calls with the same
// parameters (e.g. processing a batch of clips at the same sample rates).
//
// Thread-local storage means no locking overhead.  Fft<f32> does not
// implement Default, so we use HashMap::remove / re-insert to move ownership
// in and out of the cache around each call.  On reuse, Resampler::reset()
// clears overlap buffers and delay state while preserving FFT plans.

#[derive(Hash, Eq, PartialEq, Clone, Copy)]
struct ResamplerKey {
    in_sr: u32,
    out_sr: u32,
    channels: usize,
    quality: u8, // 1 = medium, 2 = high
}

thread_local! {
    static FFT_CACHE: RefCell<HashMap<ResamplerKey, Fft<f32>>>
        = RefCell::new(HashMap::new());
}

fn with_cached_fft_resampler<T>(
    key: ResamplerKey,
    make: impl FnOnce() -> AudioSampleResult<Fft<f32>>,
    f: impl FnOnce(&mut Fft<f32>) -> AudioSampleResult<T>,
) -> AudioSampleResult<T> {
    FFT_CACHE.with(|cell| -> AudioSampleResult<()> {
        let mut map = cell.borrow_mut();
        if !map.contains_key(&key) {
            map.insert(key, make()?);
        }
        Ok(())
    })?;

    let mut resampler = FFT_CACHE.with(|cell| cell.borrow_mut().remove(&key).unwrap());
    resampler.reset();
    let result = f(&mut resampler);
    FFT_CACHE.with(|cell| {
        cell.borrow_mut().insert(key, resampler);
    });
    result
}

// ---------------------------------------------------------------------------
// SSE2 SIMD helpers (x86_64 baseline — no runtime feature check needed)
// ---------------------------------------------------------------------------

/// Stride-2 downsample using SSE2 `_mm_shuffle_ps`.
///
/// Each SSE2 iteration loads 8 input f32s into two `__m128` registers and
/// extracts the even-indexed elements (indices 0, 2, 4, 6) with a single
/// shuffle, producing 4 output f32s.  This avoids the scalar stride-2 read
/// pattern that LLVM cannot auto-vectorise without AVX2 gather instructions.
///
/// `src.len()` must be ≥ 2 × `dst.len()` (caller ensures this).
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn downsample_by_2_simd(src: &[f32], dst: &mut [f32]) {
    use core::arch::x86_64::*;

    // Number of full 8-input / 4-output SIMD iterations that stay in bounds.
    let n_simd = (src.len() / 8).min(dst.len() / 4);

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    for i in 0..n_simd {
        // v0 = [a0, a1, b0, b1]   v1 = [c0, c1, d0, d1]
        // shuffle mask 0x88 = 0b10_00_10_00:
        //   result[0] = v0[0], result[1] = v0[2], result[2] = v1[0], result[3] = v1[2]
        // → [a0, b0, c0, d0]  (stride-2 downsampled)
        unsafe {
            let v0 = _mm_loadu_ps(src_ptr.add(i * 8));
            let v1 = _mm_loadu_ps(src_ptr.add(i * 8 + 4));
            let out = _mm_shuffle_ps(v0, v1, 0x88);
            _mm_storeu_ps(dst_ptr.add(i * 4), out);
        }
    }

    // Scalar tail for the remaining ≤ 3 output samples.
    let simd_out = n_simd * 4;
    let simd_in = n_simd * 8;
    for (j, out_s) in dst[simd_out..].iter_mut().enumerate() {
        *out_s = src[simd_in + j * 2];
    }
}

/// Stride-2 upsample (linear interpolation) using SSE2.
///
/// Each iteration reads 4 input samples plus 1 overlap sample, computing
/// midpoints with `_mm_add_ps` + `_mm_mul_ps`, then interleaving originals
/// and averages with `_mm_unpacklo/hi_ps` to produce 8 output samples.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn upsample_by_2_simd(src: &[f32], dst: &mut [f32]) {
    use core::arch::x86_64::*;

    // Each iteration accesses src[i*4 .. i*4+5], so need i*4+4 < src.len().
    // Maximum safe i = (src.len() - 2) / 4  (i.e. (src.len()-1)/4 in int-div).
    let n_simd = if src.len() >= 5 {
        ((src.len() - 1) / 4).min(dst.len() / 8)
    } else {
        0
    };

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    for i in 0..n_simd {
        unsafe {
            let half = _mm_set1_ps(0.5f32);
            let v_cur = _mm_loadu_ps(src_ptr.add(i * 4)); // [s0, s1, s2, s3]
            let v_next = _mm_loadu_ps(src_ptr.add(i * 4 + 1)); // [s1, s2, s3, s4]
            // v_avg = [(s0+s1)/2, (s1+s2)/2, (s2+s3)/2, (s3+s4)/2]
            let v_avg = _mm_mul_ps(_mm_add_ps(v_cur, v_next), half);
            // Interleave: lo = [s0, avg01, s1, avg12],  hi = [s2, avg23, s3, avg34]
            let lo = _mm_unpacklo_ps(v_cur, v_avg);
            let hi = _mm_unpackhi_ps(v_cur, v_avg);
            _mm_storeu_ps(dst_ptr.add(i * 8), lo);
            _mm_storeu_ps(dst_ptr.add(i * 8 + 4), hi);
        }
    }

    // Scalar tail.
    let simd_out = n_simd * 8;
    let simd_in = n_simd * 4;
    let last = src.len().saturating_sub(1);
    for (j, out_s) in dst[simd_out..].iter_mut().enumerate() {
        let in_i = simd_in + j / 2;
        let k = j & 1;
        if in_i < last {
            let s0 = src[in_i];
            let s1 = src[in_i + 1];
            *out_s = s0 + k as f32 * 0.5 * (s1 - s0);
        } else {
            *out_s = *src.last().unwrap_or(&0.0);
        }
    }
}

/// Resamples audio to a new sample rate.
///
/// # Arguments
/// * `audio` - The input audio samples
/// * `target_sample_rate` - Desired output sample rate in Hz
/// * `quality` - Quality/performance trade-off setting
///
/// # Returns
/// A new `AudioSamples` instance at the target sample rate.
///
/// # Errors
/// Returns an error if the resampling parameters are invalid or rubato fails internally.
///
/// # Example
/// ```rust,ignore
/// use audio_samples::{AudioSamples, operations::types::ResamplingQuality};
/// use ndarray::array;
///
/// let audio = AudioSamples::new_mono(array![1.0f32, 0.5, -0.5, -1.0], 44100).unwrap();
/// let resampled = resample(&audio, 48000, ResamplingQuality::High)?;
/// assert_eq!(resampled.sample_rate(), 48000);
/// ```
#[inline]
pub fn resample<T>(
    audio: &AudioSamples<T>,
    target_sample_rate: SampleRate,
    quality: ResamplingQuality,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    T: StandardSample,
{
    if audio.sample_rate() == target_sample_rate {
        return Ok(audio.clone().into_owned());
    }

    // Process at f32; sufficient precision for audio and halves memory traffic
    // versus f64.
    let input = audio.as_f32();

    match quality {
        ResamplingQuality::Fast => resample_linear(&input, target_sample_rate),
        ResamplingQuality::Medium => resample_medium(&input, target_sample_rate),
        ResamplingQuality::High => resample_high(&input, target_sample_rate),
    }
}

// ---------------------------------------------------------------------------
// Fast path — linear interpolation
// ---------------------------------------------------------------------------
//
// Zero construction overhead (no rubato resampler).  Each call allocates the
// output buffer and runs a tight per-channel loop.
//
// Integer ratios (e.g. 44100 → 22050, factor 2) are special-cased to avoid
// floating-point position tracking and allow the compiler to emit SIMD
// instructions for the inner loop.
//
// Quality trade-off: no anti-aliasing filter.  Aliasing is audible on signals
// with energy close to the Nyquist frequency of the output, which is
// acceptable for the Fast preset.

fn resample_linear<T>(
    audio: &AudioSamples<f32>,
    target_sample_rate: SampleRate,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    T: StandardSample,
{
    let in_sr = audio.sample_rate().get();
    let out_sr = target_sample_rate.get();
    let in_frames = audio.samples_per_channel().get();
    let out_frames =
        ((in_frames as u64 * out_sr as u64 + in_sr as u64 - 1) / in_sr as u64) as usize;
    let channels = audio.num_channels().get() as usize;

    let mut output: Vec<Vec<f32>> = (0..channels).map(|_| vec![0.0f32; out_frames]).collect();

    let process_channel = |src: &[f32], dst: &mut [f32]| {
        if in_sr % out_sr == 0 {
            // Integer downsampling by N: take the first sample of every
            // N-element chunk.
            let n = (in_sr / out_sr) as usize;

            // n=2 hot path: explicit SSE2 shuffle avoids the stride-2 read
            // pattern that LLVM cannot auto-vectorise without AVX2 gather
            // instructions.
            #[cfg(target_arch = "x86_64")]
            if n == 2 {
                // SAFETY: SSE2 is mandated by the x86_64 ABI.
                unsafe {
                    downsample_by_2_simd(src, dst);
                }
                return;
            }

            for (chunk, out_s) in src.chunks_exact(n).zip(dst.iter_mut()) {
                *out_s = chunk[0];
            }
        } else if out_sr % in_sr == 0 {
            // Integer upsampling by N.
            let n = (out_sr / in_sr) as usize;

            // n=2 hot path: SSE2 interleave + midpoint average, 8 output
            // samples per instruction pair.
            #[cfg(target_arch = "x86_64")]
            if n == 2 {
                // SAFETY: SSE2 is mandated by the x86_64 ABI.
                unsafe {
                    upsample_by_2_simd(src, dst);
                }
                return;
            }

            // General N: linear interpolation across N-wide output chunks.
            let step_f32 = 1.0f32 / n as f32;
            let safe_in = src.len().saturating_sub(1);
            let safe_out = safe_in * n;
            let split = safe_out.min(dst.len());
            let (main, tail) = dst.split_at_mut(split);

            for (in_i, out_chunk) in main.chunks_exact_mut(n).enumerate() {
                let s0 = src[in_i];
                let s1 = src[in_i + 1]; // always valid: in_i < safe_in
                let diff = s1 - s0;
                for (k, out_s) in out_chunk.iter_mut().enumerate() {
                    *out_s = s0 + k as f32 * step_f32 * diff;
                }
            }
            let last_s = src.last().copied().unwrap_or(0.0);
            tail.fill(last_s);
        } else {
            // General fractional ratio.
            let step = in_sr as f64 / out_sr as f64;
            let last = src.len() - 1;
            for (out_i, out_s) in dst.iter_mut().enumerate() {
                let pos = out_i as f64 * step;
                let i0 = pos as usize;
                let i1 = (i0 + 1).min(last);
                let frac = (pos - i0 as f64) as f32;
                *out_s = src[i0] + frac * (src[i1] - src[i0]);
            }
        }
    };

    if audio.is_mono() {
        let view = audio.as_mono().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Failed to get mono data",
            ))
        })?;
        // Prefer a zero-copy slice; fall back to an owned copy only if the
        // backing array is non-contiguous (rare in practice).
        let owned;
        let src: &[f32] = if let Some(s) = view.as_slice() {
            s
        } else {
            owned = view.to_vec();
            &owned
        };
        process_channel(src, &mut output[0]);
    } else {
        let view = audio.as_multi_channel().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Failed to get multi-channel data",
            ))
        })?;
        for ch in 0..channels {
            let row = view.row(ch);
            let owned;
            let src: &[f32] = if let Some(s) = row.as_slice() {
                s
            } else {
                owned = row.to_vec();
                &owned
            };
            process_channel(src, &mut output[ch]);
        }
    }

    assemble_output(
        output,
        out_frames,
        channels,
        audio.num_channels(),
        target_sample_rate,
    )
}

// ---------------------------------------------------------------------------
// Medium / High paths — rubato FFT synchronous resampler
// ---------------------------------------------------------------------------
//
// Both quality tiers use rubato's Fft<f32> synchronous resampler, which
// operates entirely in the spectral domain:
//   1. FFT the input chunk.
//   2. Extend or truncate the spectrum to the output length.
//   3. Apply a BlackmanHarris2 anti-aliasing window in the frequency domain.
//   4. IFFT to produce output samples.
//
// This is O(N log N) vs O(N × sinc_len) for the sinc resampler, giving
// significantly better throughput on the long signals typical in offline/batch
// use.  Quality is also slightly better in practice: the spectral approach
// avoids the sinc interpolation approximation error that the async resampler
// incurs between pre-computed filter table entries.
//
// The two tiers differ only in chunk size.  A larger chunk amortises per-FFT
// overhead more aggressively and gives marginally sharper roll-off at signal
// edges; both use the same BlackmanHarris2 anti-aliasing filter.
//
// The Fft resampler processes all channels in a single call, so no per-channel
// parallelism is needed here.

fn resample_medium<T>(
    audio: &AudioSamples<f32>,
    target_sample_rate: SampleRate,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    T: StandardSample,
{
    resample_fft(audio, target_sample_rate, FFT_MEDIUM_CHUNK, 1)
}

fn resample_high<T>(
    audio: &AudioSamples<f32>,
    target_sample_rate: SampleRate,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    T: StandardSample,
{
    resample_fft(audio, target_sample_rate, FFT_HIGH_CHUNK, 2)
}

fn resample_fft<T>(
    audio: &AudioSamples<f32>,
    target_sample_rate: SampleRate,
    chunk_size: usize,
    quality_id: u8,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    T: StandardSample,
{
    let in_sr = audio.sample_rate().get();
    let channels = audio.num_channels().get() as usize;
    let key = ResamplerKey {
        in_sr,
        out_sr: target_sample_rate.get(),
        channels,
        quality: quality_id,
    };

    let input_data = extract_channel_data(audio)?;
    let in_sr_u64 = in_sr as u64;
    let out_sr_u64 = target_sample_rate.get() as u64;
    let input_length = input_data[0].len();
    let expected_frames = ((input_length as u64 * out_sr_u64 + in_sr_u64 - 1) / in_sr_u64) as usize;

    with_cached_fft_resampler(
        key,
        || {
            Fft::<f32>::new(
                in_sr as usize,
                target_sample_rate.get() as usize,
                chunk_size,
                1,
                channels,
                FixedSync::Input,
            )
            .map_err(|e| {
                AudioSampleError::Processing(ProcessingError::algorithm_failure(
                    "fft_resampler",
                    format!("Failed to create FFT resampler: {e}"),
                ))
            })
        },
        |resampler| {
            resample_fft_blocked(
                resampler,
                input_data,
                expected_frames,
                channels,
                audio.num_channels(),
                target_sample_rate,
            )
        },
    )
}

/// Block-processing loop for all channels using a multi-channel rubato FFT resampler.
fn resample_fft_blocked<T>(
    resampler: &mut Fft<f32>,
    input_data: Vec<Vec<f32>>,
    expected_frames: usize,
    channels: usize,
    channel_count: ChannelCount,
    target_sample_rate: SampleRate,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    T: StandardSample,
{
    let chunk_in = resampler.input_frames_next();
    let chunk_out = resampler.output_frames_max();
    let input_length = input_data[0].len();

    let mut out: Vec<Vec<f32>> = (0..channels)
        .map(|_| Vec::with_capacity(expected_frames + chunk_out))
        .collect();

    // Multi-channel scratch buffers, reused across blocks.
    let mut in_buf: Vec<Vec<f32>> = (0..channels).map(|_| vec![0.0f32; chunk_in]).collect();
    let mut out_buf: Vec<Vec<f32>> = (0..channels).map(|_| vec![0.0f32; chunk_out]).collect();

    for block_start in (0..input_length).step_by(chunk_in) {
        let block_end = (block_start + chunk_in).min(input_length);
        let actual = block_end - block_start;

        for ch in 0..channels {
            in_buf[ch][..actual].copy_from_slice(&input_data[ch][block_start..block_end]);
            if actual < chunk_in {
                in_buf[ch][actual..].fill(0.0);
            }
        }

        let in_adapter = SequentialSliceOfVecs::new(&in_buf, channels, chunk_in).map_err(|e| {
            AudioSampleError::Processing(ProcessingError::algorithm_failure(
                "fft_resampler",
                format!("Input adapter error: {e}"),
            ))
        })?;
        let mut out_adapter = SequentialSliceOfVecs::new_mut(&mut out_buf, channels, chunk_out)
            .map_err(|e| {
                AudioSampleError::Processing(ProcessingError::algorithm_failure(
                    "fft_resampler",
                    format!("Output adapter error: {e}"),
                ))
            })?;

        let (_, frames_written) = resampler
            .process_into_buffer(&in_adapter, &mut out_adapter, None)
            .map_err(|e| {
                AudioSampleError::Processing(ProcessingError::algorithm_failure(
                    "fft_resampler",
                    format!("Resampling failed: {e}"),
                ))
            })?;

        for ch in 0..channels {
            out[ch].extend_from_slice(&out_buf[ch][..frames_written]);
        }
    }

    // Clamp to exact expected length; zero-padding the final block may
    // produce a few extra frames.
    for ch_data in &mut out {
        ch_data.truncate(expected_frames.min(ch_data.len()));
    }

    let total_frames = out[0].len();
    assemble_output(
        out,
        total_frames,
        channels,
        channel_count,
        target_sample_rate,
    )
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Extracts per-channel f32 data into owned `Vec<Vec<f32>>` for block processing.
fn extract_channel_data(audio: &AudioSamples<f32>) -> AudioSampleResult<Vec<Vec<f32>>> {
    if audio.is_mono() {
        let mono = audio.as_mono().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Failed to get mono data",
            ))
        })?;
        Ok(vec![mono.to_vec()])
    } else {
        let multi = audio.as_multi_channel().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Failed to get multi-channel data",
            ))
        })?;
        let channels = audio.num_channels().get() as usize;
        Ok((0..channels).map(|ch| multi.row(ch).to_vec()).collect())
    }
}

/// Assembles per-channel output `Vec`s into an [`AudioSamples<T>`].
///
/// For mono, the single channel `Vec` is used directly with no interleaving.
/// For multi-channel, channels are stored in channel-major (row-major) order
/// as required by [`AudioSamples::new_multi_channel_from_vec`].
fn assemble_output<T>(
    channel_data: Vec<Vec<f32>>,
    total_frames: usize,
    channels: usize,
    channel_count: ChannelCount,
    sample_rate: SampleRate,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    T: StandardSample,
{
    if channels == 1 {
        // SAFETY: caller guarantees total_frames > 0.
        let data = unsafe { NonEmptyVec::new_unchecked(channel_data.into_iter().next().unwrap()) };
        return Ok(AudioSamples::from_mono_vec::<f32>(data, sample_rate));
    }

    // new_multi_channel_from_vec stores data via Array2::from_shape_vec with
    // shape (channels, samples_per_channel), which interprets the flat vec in
    // row-major (C) order.  We therefore need channel-major layout:
    // [ch0[0..N], ch1[0..N], ...]  NOT interleaved [ch0[0], ch1[0], ch0[1], …].
    let mut flat = Vec::with_capacity(total_frames * channels);
    for ch_data in &channel_data {
        flat.extend_from_slice(ch_data);
    }

    let data = NonEmptyVec::new(flat).map_err(|_| {
        AudioSampleError::Processing(ProcessingError::algorithm_failure(
            "resampler",
            "Empty output after assembling channels",
        ))
    })?;

    AudioSamples::new_multi_channel_from_vec::<f32>(data, channel_count, sample_rate)
}

/// Resamples audio by a specific ratio.
///
/// # Arguments
/// * `audio` - The input audio samples
/// * `ratio` - Resampling ratio (output_rate / input_rate)
/// * `quality` - Quality/performance trade-off setting
///
/// # Returns
/// A new `AudioSamples` instance resampled by the given ratio.
///
/// # Errors
/// Returns an error if the ratio is not positive or the resampling fails.
#[inline]
pub fn resample_by_ratio<T>(
    audio: &AudioSamples<T>,
    ratio: f64,
    quality: ResamplingQuality,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    T: StandardSample,
{
    if ratio <= 0.0 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "ratio",
            format!("Invalid resampling ratio: {ratio}"),
        )));
    }

    let input_sample_rate = audio.sample_rate.get();
    let target_sample_rate = (f64::from(input_sample_rate) * ratio).round() as usize;
    let target_sample_rate = SampleRate::new(target_sample_rate as u32).ok_or_else(|| {
        AudioSampleError::Parameter(ParameterError::invalid_value(
            "target_sample_rate",
            "Calculated target sample rate is invalid",
        ))
    })?;

    resample(audio, target_sample_rate, quality)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioSamples;
    use crate::channels;
    use crate::sample_rate;
    use ndarray::{Array1, array};

    #[test]
    fn test_resample_mono() {
        let samples = (0..1024)
            .map(|x| (x as f32 * 2.0 * std::f32::consts::PI * 440.0 / 44100.0).sin())
            .collect::<Vec<f32>>();
        let data = Array1::from_vec(samples);
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let resampled = resample(&audio, sample_rate!(48000), ResamplingQuality::Medium).unwrap();
        assert_eq!(resampled.sample_rate(), sample_rate!(48000));
        assert_eq!(resampled.num_channels(), channels!(1));
    }

    #[test]
    fn test_resample_by_ratio() {
        let data = array![1.0f32, 0.0, -1.0, 0.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let upsampled = resample_by_ratio(&audio, 2.0, ResamplingQuality::Fast).unwrap();
        assert_eq!(upsampled.sample_rate(), sample_rate!(88200));

        let downsampled = resample_by_ratio(&audio, 0.5, ResamplingQuality::Fast).unwrap();
        assert_eq!(downsampled.sample_rate(), sample_rate!(22050));
    }

    #[test]
    fn test_no_resampling_needed() {
        let data = array![1.0f32, 0.0, -1.0];
        let audio = AudioSamples::new_mono(data.clone().into(), sample_rate!(44100)).unwrap();

        let result = resample(&audio, sample_rate!(44100), ResamplingQuality::High).unwrap();
        assert_eq!(result.sample_rate(), sample_rate!(44100));

        let original_mono = audio.as_mono().unwrap();
        let result_mono = result.as_mono().unwrap();
        assert_eq!(original_mono.len(), result_mono.len());
    }

    #[test]
    fn test_invalid_ratio() {
        let data = array![1.0f32, 0.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        assert!(resample_by_ratio(&audio, -1.0, ResamplingQuality::Fast).is_err());
        assert!(resample_by_ratio(&audio, 0.0, ResamplingQuality::Fast).is_err());
    }

    #[test]
    fn test_resample_correct_length_long_signal() {
        let samples = (0..220500)
            .map(|x| (x as f32 * 2.0 * std::f32::consts::PI * 440.0 / 44100.0).sin())
            .collect::<Vec<f32>>();
        let audio = AudioSamples::new_mono(Array1::from_vec(samples), sample_rate!(44100)).unwrap();

        for quality in [
            ResamplingQuality::Fast,
            ResamplingQuality::Medium,
            ResamplingQuality::High,
        ] {
            let result = resample(&audio, sample_rate!(22050), quality).unwrap();
            let got = result.samples_per_channel().get();
            let expected = 110250usize;
            assert!(
                got.abs_diff(expected) < 200,
                "{quality:?}: expected ~{expected} samples, got {got}"
            );
        }
    }
}
