//! Module for handling audio sample resampling operations.
//! Uses rubato for high-quality resampling.

use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ParameterError,
    ProcessingError,
    operations::types::ResamplingQuality,
    repr::{ChannelCount, SampleRate},
    traits::StandardSample,
};
use audioadapter_buffers::direct::SequentialSliceOfVecs;
use non_empty_slice::NonEmptyVec;
use rubato::{Async, Fft, FixedAsync, FixedSync, Resampler, SincInterpolationType, WindowFunction};

// todo add medium block
const FAST_BLOCK: usize = 4096;
const HIGH_BLOCK: usize = 8192;

fn block_size(quality: ResamplingQuality, input_len: usize) -> usize {
    let target = match quality {
        ResamplingQuality::Fast | ResamplingQuality::Medium => FAST_BLOCK,
        ResamplingQuality::High => HIGH_BLOCK,
    };
    input_len.min(target)
}

/// Resamples audio to a new sample rate using high-quality algorithms.
///
/// This function provides a convenient interface to rubato's resampling capabilities
/// with different quality/performance trade-offs.
///
/// # Arguments
/// * `audio` - The input audio samples
/// * `target_sample_rate` - Desired output sample rate in Hz
/// * `quality` - Quality/performance trade-off setting
///
/// # Returns
/// A new AudioSamples instance with the target sample rate
///
/// # Errors
/// Returns an error if:
/// - The resampling parameters are invalid
/// - The input audio is empty
/// - Rubato encounters an internal error
///
/// # Example
/// ```rust,ignore
/// use audio_samples::{AudioSamples<f64>, operations::types::ResamplingQuality};
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
    let input_sample_rate = audio.sample_rate();
    if input_sample_rate == target_sample_rate {
        // No resampling needed - convert directly
        return Ok(audio.clone().into_owned());
    }

    // Convert to f64 for processing (rubato works with f64)
    let input_float = audio.as_float();

    // Resample using appropriate quality settings
    let output: AudioSamples<T> = match quality {
        ResamplingQuality::Fast => resample_fast(&input_float, target_sample_rate)?,
        ResamplingQuality::Medium => resample_medium(&input_float, target_sample_rate)?,
        ResamplingQuality::High => resample_high(&input_float, target_sample_rate)?,
    };

    Ok(output)
}

/// Fast resampling using linear interpolation.
/// Good for real-time applications where speed is critical.
fn resample_fast<T>(
    audio: &AudioSamples<f64>,
    target_sample_rate: SampleRate,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    T: StandardSample,
{
    let input_sample_rate = audio.sample_rate().get() as usize;
    let channels = audio.num_channels();

    // Create resampler
    let mut resampler = Fft::<f64>::new(
        input_sample_rate,
        target_sample_rate.get() as usize,
        block_size(ResamplingQuality::Fast, audio.samples_per_channel().get()),
        1,
        channels.get() as usize,
        FixedSync::Both,
    )
    .map_err(|e| {
        AudioSampleError::Processing(ProcessingError::algorithm_failure(
            "fast_resampler",
            format!("Failed to create fast resampler: {e}"),
        ))
    })?;

    resample_with_resampler(audio, &mut resampler, target_sample_rate)
}

/// Medium quality resampling with balanced speed/quality.
/// Good general-purpose resampling for most applications.
fn resample_medium<T>(
    audio: &AudioSamples<f64>,
    target_sample_rate: SampleRate,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    T: StandardSample,
{
    let input_sample_rate = audio.sample_rate().get() as usize;
    let channels = audio.num_channels();

    // Create resampler with medium settings
    let mut resampler = Async::<f64>::new_sinc(
        f64::from(target_sample_rate.get()) / input_sample_rate as f64,
        2.0, // Oversampling factor
        &rubato::SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        },
        block_size(ResamplingQuality::Medium, audio.samples_per_channel().get()),
        channels.get() as usize,
        FixedAsync::Output,
    )
    .map_err(|e| {
        AudioSampleError::Processing(ProcessingError::algorithm_failure(
            "medium_resampler",
            format!("Failed to create medium quality resampler: {e}"),
        ))
    })?;

    resample_with_sinc_resampler(audio, &mut resampler, target_sample_rate)
}

/// High quality resampling with maximum quality.
/// Best for offline processing where quality is paramount.
fn resample_high<T>(
    audio: &AudioSamples<f64>,
    target_sample_rate: SampleRate,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    T: StandardSample,
{
    let channels = audio.num_channels();

    // Create high-quality resampler
    let mut resampler = Async::<f64>::new_sinc(
        f64::from(target_sample_rate.get()) / f64::from(audio.sample_rate().get()),
        2.0, // Oversampling factor
        &rubato::SincInterpolationParameters {
            sinc_len: 256, // Longer sinc filter for better quality
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic, // Higher quality interpolation
            oversampling_factor: 512,                    // Higher oversampling
            window: WindowFunction::BlackmanHarris2,
        },
        block_size(ResamplingQuality::High, audio.samples_per_channel().get()),
        channels.get() as usize,
        FixedAsync::Output,
    )
    .map_err(|e| {
        AudioSampleError::Processing(ProcessingError::algorithm_failure(
            "high_resampler",
            format!("Failed to create high quality resampler: {e}"),
        ))
    })?;

    resample_with_sinc_resampler(audio, &mut resampler, target_sample_rate)
}

/// Helper function to extract channel data from AudioSamples<f64>.
fn extract_channel_data_float(audio: &AudioSamples<f64>) -> AudioSampleResult<Vec<Vec<f64>>> {
    let channels = audio.num_channels().get();
    let mut input_data: Vec<Vec<f64>> = Vec::with_capacity(channels as usize);

    if audio.is_mono() {
        let mono_data = audio.as_mono().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Failed to get mono data",
            ))
        })?;
        input_data.push(mono_data.to_vec());
    } else {
        let multi_data = audio.as_multi_channel().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Failed to get multi-channel data",
            ))
        })?;

        for ch in 0..channels as usize {
            let channel_data: Vec<f64> = multi_data.row(ch).to_vec();
            input_data.push(channel_data);
        }
    }

    Ok(input_data)
}

/// Helper function to perform resampling with an FFT-based resampler.
fn resample_with_resampler<R, T>(
    audio: &AudioSamples<f64>,
    resampler: &mut R,
    target_sample_rate: SampleRate,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    R: Resampler<f64>,
    T: StandardSample,
{
    let channels = audio.num_channels().get() as usize;
    let input_data = extract_channel_data_float(audio)?;

    // Calculate output buffer size
    let input_frames = input_data[0].len();
    let output_frames = resampler.output_frames_max();

    // Create output buffer
    let mut output_data: Vec<Vec<f64>> = vec![vec![0.0; output_frames]; channels];

    // Wrap input and output with adapters
    let input_adapter =
        SequentialSliceOfVecs::new(&input_data, channels, input_frames).map_err(|e| {
            AudioSampleError::Processing(ProcessingError::algorithm_failure(
                "resampler",
                format!("Failed to create input adapter: {e}"),
            ))
        })?;
    let mut output_adapter =
        SequentialSliceOfVecs::new_mut(&mut output_data, channels, output_frames).map_err(|e| {
            AudioSampleError::Processing(ProcessingError::algorithm_failure(
                "resampler",
                format!("Failed to create output adapter: {e}"),
            ))
        })?;

    // Perform resampling
    let (_frames_read, frames_written) = resampler
        .process_into_buffer(&input_adapter, &mut output_adapter, None)
        .map_err(|e| {
            AudioSampleError::Processing(ProcessingError::algorithm_failure(
                "resampler",
                format!("Resampling failed: {e}"),
            ))
        })?;

    // Flatten output data (interleave channels)
    let mut output_flat: Vec<f64> = Vec::with_capacity(frames_written * channels);
    for frame_idx in 0..frames_written {
        for channel_data in &output_data {
            output_flat.push(channel_data[frame_idx]);
        }
    }

    // frames_written > 0 when resampling succeeds
    let output_flat = NonEmptyVec::new(output_flat).map_err(|_| {
        AudioSampleError::Processing(ProcessingError::algorithm_failure(
            "resampler",
            "No output frames produced",
        ))
    })?;

    // Convert output back to AudioSamples format
    convert_channel_data_to_audio_samples(output_flat, audio.num_channels(), target_sample_rate)
}

/// Helper function to perform resampling with a sinc-based (Async) resampler.
fn resample_with_sinc_resampler<T>(
    audio: &AudioSamples<f64>,
    resampler: &mut Async<f64>,
    target_sample_rate: SampleRate,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    T: StandardSample,
{
    let channels = audio.num_channels().get() as usize;

    // Extract channel data
    let input_data = extract_channel_data_float(audio)?;

    // For Async resamplers, we need to process in chunks
    let chunk_size = resampler.input_frames_max();
    let input_length = input_data[0].len();
    let mut all_output_data: Vec<Vec<f64>> = vec![Vec::new(); channels];

    // Preallocate reusable chunk buffer to reduce allocations
    let mut chunk_data = vec![vec![0.0; chunk_size]; channels];
    let output_chunk_size = resampler.output_frames_max();
    let mut output_data = vec![vec![0.0; output_chunk_size]; channels];

    // If input is smaller than chunk size, pad it to minimum required size
    if input_length < chunk_size {
        // Copy input data to chunk buffer and fill remaining with zeros
        for ch in 0..channels {
            let src = &input_data[ch];
            chunk_data[ch][..src.len()].copy_from_slice(src);
            chunk_data[ch][src.len()..].fill(0.0);
        }

        // Process the single padded chunk
        let input_adapter =
            SequentialSliceOfVecs::new(&chunk_data, channels, chunk_size).map_err(|e| {
                AudioSampleError::Processing(ProcessingError::algorithm_failure(
                    "resampler",
                    format!("Failed to create input adapter: {e}"),
                ))
            })?;
        let mut output_adapter =
            SequentialSliceOfVecs::new_mut(&mut output_data, channels, output_chunk_size).map_err(
                |e| {
                    AudioSampleError::Processing(ProcessingError::algorithm_failure(
                        "resampler",
                        format!("Failed to create output adapter: {e}"),
                    ))
                },
            )?;

        match resampler.process_into_buffer(&input_adapter, &mut output_adapter, None) {
            Ok((_frames_read, frames_written)) => {
                for ch in 0..channels {
                    all_output_data[ch].extend_from_slice(&output_data[ch][..frames_written]);
                }
            }
            Err(e) => {
                return Err(AudioSampleError::Processing(
                    ProcessingError::algorithm_failure(
                        "resampler",
                        format!(
                            "Single chunk resampling failed (input_len={input_length}, chunk_size={chunk_size}, channels={channels}): {e}"
                        ),
                    ),
                ));
            }
        }
    } else {
        // Process in chunks as normal
        for chunk_start in (0..input_length).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(input_length);
            let actual_chunk_size = chunk_end - chunk_start;

            // Copy chunk data using slice operations to avoid allocations
            for ch in 0..channels {
                let src = &input_data[ch][chunk_start..chunk_end];
                chunk_data[ch][..src.len()].copy_from_slice(src);
                // If this chunk is smaller than required (last chunk), pad it
                if actual_chunk_size < chunk_size {
                    chunk_data[ch][src.len()..].fill(0.0);
                }
            }

            // Resample this chunk
            let input_adapter = SequentialSliceOfVecs::new(&chunk_data, channels, chunk_size)
                .map_err(|e| {
                    AudioSampleError::Processing(ProcessingError::algorithm_failure(
                        "resampler",
                        format!("Failed to create input adapter: {e}"),
                    ))
                })?;
            let mut output_adapter =
                SequentialSliceOfVecs::new_mut(&mut output_data, channels, output_chunk_size)
                    .map_err(|e| {
                        AudioSampleError::Processing(ProcessingError::algorithm_failure(
                            "resampler",
                            format!("Failed to create output adapter: {e}"),
                        ))
                    })?;

            match resampler.process_into_buffer(&input_adapter, &mut output_adapter, None) {
                Ok((_frames_read, frames_written)) => {
                    for ch in 0..channels {
                        all_output_data[ch].extend_from_slice(&output_data[ch][..frames_written]);
                    }
                }
                Err(e) => {
                    return Err(AudioSampleError::Processing(
                        ProcessingError::algorithm_failure(
                            "resampler",
                            format!(
                                "Chunk resampling failed (chunk_start={chunk_start}, chunk_size={chunk_size}, actual_size={actual_chunk_size}, channels={channels}): {e}"
                            ),
                        ),
                    ));
                }
            }
        }
    }

    // Concatenate all output frames (interleave channels)
    if all_output_data[0].is_empty() {
        return Err(AudioSampleError::Processing(
            ProcessingError::algorithm_failure("resampler", "No output frames produced"),
        ));
    }

    let total_frames = all_output_data[0].len();
    let mut combined_output: Vec<f64> = Vec::with_capacity(total_frames * channels);
    for frame_idx in 0..total_frames {
        for channel_data in &all_output_data {
            combined_output.push(channel_data[frame_idx]);
        }
    }

    // safety: already checked for empty above
    let combined_output = unsafe { NonEmptyVec::new_unchecked(combined_output) };
    convert_channel_data_to_audio_samples(combined_output, audio.num_channels(), target_sample_rate)
}

/// Helper function to create AudioSamples from channel data using interleave_channels.
fn convert_channel_data_to_audio_samples<T>(
    channel_data: NonEmptyVec<f64>,
    channels: ChannelCount,
    sample_rate: SampleRate,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    T: StandardSample,
{
    if channels.get() == 1 {
        return Ok(AudioSamples::from_mono_vec::<f64>(
            channel_data,
            sample_rate,
        ));
    }
    AudioSamples::new_multi_channel_from_vec::<f64>(channel_data, channels, sample_rate)
}

/// Resamples audio by a specific ratio.
///
/// # Arguments
/// * `audio` - The input audio samples
/// * `ratio` - Resampling ratio (output_rate / input_rate)
/// * `quality` - Quality/performance trade-off setting
///
/// # Returns
/// A new AudioSamples instance resampled by the given ratio
///
/// # Errors
///
/// Returns an error if:
/// - The ratio is not positive
/// - The calculated target sample rate is invalid
/// - The resampling process fails for any reason
///
/// # Panics
///
/// Panics if target sample rate calculation results in a value that cannot be converted to usize.
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
        // Create a suitable mono signal at 44.1 kHz
        // at least 1024 samples to ensure chunking works
        let samples = (0..1024)
            .map(|x| (x as f32 * 2.0 * std::f32::consts::PI * 440.0 / 44100.0).sin())
            .collect::<Vec<f32>>();

        let data = Array1::from_vec(samples);

        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();
        println!("Original samples: {:?}", audio);
        // Resample to 48 kHz
        let resampled = resample(&audio, sample_rate!(48000), ResamplingQuality::Medium).unwrap();

        assert_eq!(resampled.sample_rate(), sample_rate!(48000));
        assert_eq!(resampled.num_channels(), channels!(1));
    }

    #[test]
    fn test_resample_by_ratio() {
        let data = array![1.0f32, 0.0, -1.0, 0.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        // Upsample by 2x
        let upsampled = resample_by_ratio(&audio, 2.0, ResamplingQuality::Fast).unwrap();
        assert_eq!(upsampled.sample_rate(), sample_rate!(88200));

        // Downsample by 0.5x
        let downsampled = resample_by_ratio(&audio, 0.5, ResamplingQuality::Fast).unwrap();
        assert_eq!(downsampled.sample_rate(), sample_rate!(22050));
    }

    #[test]
    fn test_no_resampling_needed() {
        let data = array![1.0f32, 0.0, -1.0];
        let audio = AudioSamples::new_mono(data.clone().into(), sample_rate!(44100)).unwrap();

        // "Resample" to same rate
        let result = resample(&audio, sample_rate!(44100), ResamplingQuality::High).unwrap();
        assert_eq!(result.sample_rate(), sample_rate!(44100));

        // Should be identical
        let original_mono = audio.as_mono().unwrap();
        let result_mono = result.as_mono().unwrap();
        assert_eq!(original_mono.len(), result_mono.len());
    }

    #[test]
    fn test_invalid_ratio() {
        let data = array![1.0f32, 0.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let result = resample_by_ratio(&audio, -1.0, ResamplingQuality::Fast);
        assert!(result.is_err());

        let result = resample_by_ratio(&audio, 0.0, ResamplingQuality::Fast);
        assert!(result.is_err());
    }
}
