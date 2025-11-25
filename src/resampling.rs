//! Module for handling audio sample resampling operations.
//! Uses rubato for high-quality resampling.

use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ConvertTo,
    I24, ParameterError, ProcessingError, RealFloat,
    operations::{traits::AudioChannelOps, types::ResamplingQuality},
    to_precision,
};
use rubato::{FftFixedInOut, Resampler, SincFixedIn, SincInterpolationType, WindowFunction};

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
/// use audio_samples::{AudioSamples, operations::types::ResamplingQuality};
/// use ndarray::array;
///
/// let audio = AudioSamples::new_mono(array![1.0f32, 0.5, -0.5, -1.0], 44100);
/// let resampled = resample(&audio, 48000, ResamplingQuality::High)?;
/// assert_eq!(resampled.sample_rate(), 48000);
/// ```
pub fn resample<F, T>(
    audio: &AudioSamples<T>,
    target_sample_rate: usize,
    quality: ResamplingQuality,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    F: RealFloat + ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
{
    if audio.total_samples() == 0 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "audio",
            "Cannot resample empty audio",
        )));
    }

    let input_sample_rate = audio.sample_rate().get() as usize;
    if input_sample_rate == target_sample_rate {
        // No resampling needed - convert directly
        return Ok(audio.clone().to_type::<T>());
    }

    // Convert to f64 for processing (rubato works with f64)
    let input_float: AudioSamples<F> = audio.as_float();

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
fn resample_fast<F, T>(
    audio: &AudioSamples<F>,
    target_sample_rate: usize,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    F: RealFloat + ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
{
    let input_sample_rate = audio.sample_rate().get() as usize;
    let channels = audio.num_channels();

    // Create resampler
    let mut resampler = FftFixedInOut::<F>::new(
        input_sample_rate,
        target_sample_rate,
        block_size(ResamplingQuality::Fast, audio.samples_per_channel()),
        channels,
    )
    .map_err(|e| {
        AudioSampleError::Processing(ProcessingError::algorithm_failure(
            "fast_resampler",
            format!("Failed to create fast resampler: {}", e),
        ))
    })?;

    resample_with_resampler(audio, &mut resampler, target_sample_rate)
}

/// Medium quality resampling with balanced speed/quality.
/// Good general-purpose resampling for most applications.
fn resample_medium<F, T>(
    audio: &AudioSamples<F>,
    target_sample_rate: usize,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    F: RealFloat + ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
{
    let input_sample_rate = audio.sample_rate().get() as usize;
    let channels = audio.num_channels();

    let target_sample_rate = to_precision::<F, _>(target_sample_rate);
    let input_sample_rate = to_precision::<F, _>(input_sample_rate);
    // Create resampler with medium settings
    let mut resampler = SincFixedIn::<F>::new(
        to_precision::<f64, F>(target_sample_rate / input_sample_rate),
        2.0, // Oversampling factor
        rubato::SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        },
        block_size(ResamplingQuality::Medium, audio.samples_per_channel()),
        channels,
    )
    .map_err(|e| {
        AudioSampleError::Processing(ProcessingError::algorithm_failure(
            "medium_resampler",
            format!("Failed to create medium quality resampler: {}", e),
        ))
    })?;

    resample_with_sinc_resampler(
        audio,
        &mut resampler,
        target_sample_rate.to_usize().expect("should not fail"),
    )
}

/// High quality resampling with maximum quality.
/// Best for offline processing where quality is paramount.
fn resample_high<F, T>(
    audio: &AudioSamples<F>,
    target_sample_rate: usize,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    F: RealFloat + ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
{
    let input_sample_rate = audio.sample_rate().get() as usize;
    let channels = audio.num_channels();

    let target_sample_rate = to_precision::<F, _>(target_sample_rate);
    let input_sample_rate = to_precision::<F, _>(input_sample_rate);

    // Create high-quality resampler
    let mut resampler = SincFixedIn::<F>::new(
        to_precision::<f64, F>(target_sample_rate / input_sample_rate),
        2.0, // Oversampling factor
        rubato::SincInterpolationParameters {
            sinc_len: 256, // Longer sinc filter for better quality
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic, // Higher quality interpolation
            oversampling_factor: 512,                    // Higher oversampling
            window: WindowFunction::BlackmanHarris2,
        },
        block_size(ResamplingQuality::High, audio.samples_per_channel()),
        channels,
    )
    .map_err(|e| {
        AudioSampleError::Processing(ProcessingError::algorithm_failure(
            "high_resampler",
            format!("Failed to create high quality resampler: {}", e),
        ))
    })?;

    resample_with_sinc_resampler(
        audio,
        &mut resampler,
        target_sample_rate.to_usize().expect("should not fail"),
    )
}

/// Helper function to extract channel data from AudioSamples<f64>.
fn extract_channel_data_float<F: RealFloat>(
    audio: &AudioSamples<F>,
) -> AudioSampleResult<Vec<Vec<F>>> {
    let channels = audio.num_channels();
    let mut input_data: Vec<Vec<F>> = Vec::with_capacity(channels);

    if audio.is_mono() {
        let mono_data =
            audio
                .as_mono()
                .ok_or(AudioSampleError::Parameter(ParameterError::invalid_value(
                    "audio_format",
                    "Failed to get mono data",
                )))?;
        input_data.push(mono_data.to_vec());
    } else {
        let multi_data = audio.as_multi_channel().ok_or(AudioSampleError::Parameter(
            ParameterError::invalid_value("audio_format", "Failed to get multi-channel data"),
        ))?;

        for ch in 0..channels {
            let channel_data: Vec<F> = multi_data.row(ch).to_vec();
            input_data.push(channel_data);
        }
    }

    Ok(input_data)
}

/// Helper function to perform resampling with an FFT-based resampler.
fn resample_with_resampler<F, R, T>(
    audio: &AudioSamples<F>,
    resampler: &mut R,
    target_sample_rate: usize,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    F: RealFloat + ConvertTo<T>,
    R: Resampler<F>,
    T: AudioSample + ConvertTo<F>,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
{
    // Extract channel data directly for f64 AudioSamples
    let input_data = extract_channel_data_float(audio)?;

    // Perform resampling
    let output_data = resampler.process(&input_data, None).map_err(|e| {
        AudioSampleError::Processing(ProcessingError::algorithm_failure(
            "resampler",
            format!("Resampling failed: {}", e),
        ))
    })?;

    // Convert output back to AudioSamples format using interleave_channels
    convert_channel_data_to_audio_samples(output_data, target_sample_rate)
}

/// Helper function to perform resampling with a sinc-based resampler.
fn resample_with_sinc_resampler<F, T>(
    audio: &AudioSamples<F>,
    resampler: &mut SincFixedIn<F>,
    target_sample_rate: usize,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    F: RealFloat + ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
{
    let channels = audio.num_channels();

    // Extract channel data
    let input_data = extract_channel_data_float(audio)?;

    // For SincFixedIn, we need to process in chunks
    let chunk_size = resampler.input_frames_max();
    let input_length = input_data[0].len();
    let mut output_chunks: Vec<Vec<Vec<F>>> = Vec::new();

    // Preallocate reusable chunk buffer to reduce allocations
    let mut chunk_data = vec![vec![F::zero(); chunk_size]; channels];

    // If input is smaller than chunk size, pad it to minimum required size
    if input_length < chunk_size {
        // Copy input data to chunk buffer and fill remaining with zeros
        for ch in 0..channels {
            let src = &input_data[ch];
            chunk_data[ch][..src.len()].copy_from_slice(src);
            chunk_data[ch][src.len()..].fill(F::zero());
        }

        // Process the single padded chunk
        match resampler.process(&chunk_data, None) {
            Ok(output_chunk) => output_chunks.push(output_chunk),
            Err(e) => {
                return Err(AudioSampleError::Processing(
                    ProcessingError::algorithm_failure(
                        "resampler",
                        format!(
                            "Single chunk resampling failed (input_len={}, chunk_size={}, channels={}): {}",
                            input_length, chunk_size, channels, e
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
                    chunk_data[ch][src.len()..].fill(F::zero());
                }
            }

            // Resample this chunk
            match resampler.process(&chunk_data, None) {
                Ok(output_chunk) => output_chunks.push(output_chunk),
                Err(e) => {
                    return Err(AudioSampleError::Processing(
                        ProcessingError::algorithm_failure(
                            "resampler",
                            format!(
                                "Chunk resampling failed (chunk_start={}, chunk_size={}, actual_size={}, channels={}): {}",
                                chunk_start, chunk_size, actual_chunk_size, channels, e
                            ),
                        ),
                    ));
                }
            }
        }
    }

    // Concatenate all output chunks
    if output_chunks.is_empty() {
        return Err(AudioSampleError::Processing(
            ProcessingError::algorithm_failure("resampler", "No output chunks produced"),
        ));
    }

    let mut combined_output: Vec<Vec<F>> = vec![Vec::new(); channels];
    for chunk in output_chunks {
        for (ch, channel_data) in chunk.into_iter().enumerate() {
            combined_output[ch].extend(channel_data.into_iter());
        }
    }

    convert_channel_data_to_audio_samples(combined_output, target_sample_rate)
}

/// Helper function to create AudioSamples from channel data using interleave_channels.
fn convert_channel_data_to_audio_samples<F: RealFloat, T: AudioSample>(
    channel_data: Vec<Vec<F>>,
    sample_rate: usize,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    F: ConvertTo<T>,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
    for<'a> AudioSamples<'a, T>: AudioChannelOps<T>,
{
    if channel_data.is_empty() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "channel_data",
            "No channel data provided",
        )));
    }

    // Convert each channel's data to AudioSamples<T> directly
    // hopefully more amicable to SIMD optimizations by using the chunks_exact method
    let mut channel_samples: Vec<AudioSamples<T>> = Vec::with_capacity(channel_data.len());
    for ch_data in channel_data {
        let mut converted_data: Vec<T> = Vec::with_capacity(ch_data.len());
        let channel_chunks = ch_data.chunks_exact(8);
        let remainder = channel_chunks.remainder();

        for chunk in channel_chunks {
            for sample in chunk {
                converted_data.push(sample.convert_to());
            }
        }

        for sample in remainder {
            converted_data.push(sample.convert_to());
        }

        let mono_array = ndarray::Array1::from_vec(converted_data);
        let audio_sample = AudioSamples::new_mono(
            mono_array,
            std::num::NonZeroU32::new(sample_rate as u32).expect("sample_rate should be non-zero"),
        );
        channel_samples.push(audio_sample);
    }

    // Use interleave_channels to combine them
    AudioSamples::interleave_channels(&channel_samples)
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
/// # Panics
///
/// Panics if target sample rate calculation results in a value that cannot be converted to usize.
pub fn resample_by_ratio<F, T>(
    audio: &AudioSamples<T>,
    ratio: F,
    quality: ResamplingQuality,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    F: RealFloat + ConvertTo<T>,
    T: AudioSample + ConvertTo<F>,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'a> AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
{
    if ratio <= F::zero() {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "ratio",
            format!("Invalid resampling ratio: {}", ratio),
        )));
    }

    let input_sample_rate = to_precision::<F, _>(audio.sample_rate.get());
    let target_sample_rate = (input_sample_rate * ratio)
        .round()
        .to_usize()
        .expect("Should not fail");

    resample::<F, T>(audio, target_sample_rate, quality)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioSamples;
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

        let audio = AudioSamples::new_mono(data, sample_rate!(44100));
        println!("Original samples: {:?}", audio);
        // Resample to 48 kHz
        let resampled = resample::<f64, _>(&audio, 48000, ResamplingQuality::Medium).unwrap();

        assert_eq!(resampled.sample_rate(), sample_rate!(48000));
        assert_eq!(resampled.num_channels(), 1);
        println!("Resampled samples: {:?}", resampled);
        assert!(resampled.samples_per_channel() > 0);
    }

    #[test]
    fn test_resample_by_ratio() {
        let data = array![1.0f32, 0.0, -1.0, 0.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100));

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
        let audio = AudioSamples::new_mono(data.clone().into(), sample_rate!(44100));

        // "Resample" to same rate
        let result = resample::<f64, _>(&audio, 44100, ResamplingQuality::High).unwrap();
        assert_eq!(result.sample_rate(), sample_rate!(44100));

        // Should be identical
        let original_mono = audio.as_mono().unwrap();
        let result_mono = result.as_mono().unwrap();
        assert_eq!(original_mono.len(), result_mono.len());
    }

    #[test]
    fn test_invalid_ratio() {
        let data = array![1.0f32, 0.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100));

        let result = resample_by_ratio(&audio, -1.0, ResamplingQuality::Fast);
        assert!(result.is_err());

        let result = resample_by_ratio(&audio, 0.0, ResamplingQuality::Fast);
        assert!(result.is_err());
    }
}
