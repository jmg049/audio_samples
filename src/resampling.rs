//! Module for handling audio sample resampling operations.
//! Uses rubato for high-quality resampling.

use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, ConvertTo, I24,
    operations::{traits::AudioTypeConversion, types::ResamplingQuality},
};

use rubato::{FftFixedInOut, Resampler, SincFixedIn, SincInterpolationType, WindowFunction};

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
pub fn resample<T>(
    audio: &AudioSamples<T>,
    target_sample_rate: usize,
    quality: ResamplingQuality,
) -> AudioSampleResult<AudioSamples<T>>
where
    T: crate::AudioSample
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive
        + crate::ConvertTo<f64>,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<f32>: AudioTypeConversion<T>,
{
    if audio.total_samples() == 0 {
        return Err(AudioSampleError::InvalidInput {
            msg: "Cannot resample empty audio".to_string(),
        });
    }

    let input_sample_rate = audio.sample_rate() as usize;
    if input_sample_rate == target_sample_rate {
        // No resampling needed
        return Ok(audio.clone());
    }

    // Convert to f64 for processing (rubato works with f64)
    let input_f32 = audio.as_type::<f32>()?;

    // Resample using appropriate quality settings
    let resampled_f64 = match quality {
        ResamplingQuality::Fast => resample_fast(&input_f32, target_sample_rate)?,
        ResamplingQuality::Medium => resample_medium(&input_f32, target_sample_rate)?,
        ResamplingQuality::High => resample_high(&input_f32, target_sample_rate)?,
    };

    // Convert back to original type
    resampled_f64.to_type::<T>()
}

/// Fast resampling using linear interpolation.
/// Good for real-time applications where speed is critical.
fn resample_fast(
    audio: &AudioSamples<f32>,
    target_sample_rate: usize,
) -> AudioSampleResult<AudioSamples<f32>> {
    let input_sample_rate = audio.sample_rate() as usize;
    let channels = audio.channels();

    // Create resampler
    let mut resampler = FftFixedInOut::<f32>::new(
        input_sample_rate,
        target_sample_rate,
        usize::min(audio.samples_per_channel(), 1024),
        channels,
    )
    .map_err(|e| AudioSampleError::ProcessingError {
        msg: format!("Failed to create fast resampler: {}", e),
    })?;

    resample_with_resampler(audio, &mut resampler, target_sample_rate)
}

/// Medium quality resampling with balanced speed/quality.
/// Good general-purpose resampling for most applications.
fn resample_medium(
    audio: &AudioSamples<f32>,
    target_sample_rate: usize,
) -> AudioSampleResult<AudioSamples<f32>> {
    let input_sample_rate = audio.sample_rate() as usize;
    let channels = audio.channels();

    // Create resampler with medium settings
    let mut resampler = SincFixedIn::<f32>::new(
        target_sample_rate as f64 / input_sample_rate as f64,
        2.0, // Oversampling factor
        rubato::SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        },
        usize::min(audio.samples_per_channel(), 1024),
        channels,
    )
    .map_err(|e| AudioSampleError::ProcessingError {
        msg: format!("Failed to create medium quality resampler: {}", e),
    })?;

    resample_with_sinc_resampler(audio, &mut resampler, target_sample_rate)
}

/// High quality resampling with maximum quality.
/// Best for offline processing where quality is paramount.
fn resample_high(
    audio: &AudioSamples<f32>,
    target_sample_rate: usize,
) -> AudioSampleResult<AudioSamples<f32>> {
    let input_sample_rate = audio.sample_rate() as usize;
    let channels = audio.channels();

    // Create high-quality resampler
    let mut resampler = SincFixedIn::<f32>::new(
        target_sample_rate as f64 / input_sample_rate as f64,
        2.0, // Oversampling factor
        rubato::SincInterpolationParameters {
            sinc_len: 256, // Longer sinc filter for better quality
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic, // Higher quality interpolation
            oversampling_factor: 512,                    // Higher oversampling
            window: WindowFunction::BlackmanHarris2,
        },
        usize::min(audio.samples_per_channel(), 2048),
        channels,
    )
    .map_err(|e| AudioSampleError::ProcessingError {
        msg: format!("Failed to create high quality resampler: {}", e),
    })?;

    resample_with_sinc_resampler(audio, &mut resampler, target_sample_rate)
}

/// Helper function to perform resampling with an FFT-based resampler.
fn resample_with_resampler<R: Resampler<f32>>(
    audio: &AudioSamples<f32>,
    resampler: &mut R,
    target_sample_rate: usize,
) -> AudioSampleResult<AudioSamples<f32>> {
    let channels = audio.channels();

    // Prepare input data - rubato expects Vec<Vec<f64>> (channels x samples)
    let mut input_data: Vec<Vec<f32>> = Vec::with_capacity(channels);

    if audio.is_mono() {
        let mono_data = audio.as_mono().ok_or(AudioSampleError::InvalidInput {
            msg: "Failed to get mono data".to_string(),
        })?;
        input_data.push(mono_data.to_vec());
    } else {
        let multi_data = audio
            .as_multi_channel()
            .ok_or(AudioSampleError::InvalidInput {
                msg: "Failed to get multi-channel data".to_string(),
            })?;

        for ch in 0..channels {
            let channel_data: Vec<f32> = multi_data.row(ch).to_vec();
            input_data.push(channel_data);
        }
    }

    // Perform resampling
    let output_data =
        resampler
            .process(&input_data, None)
            .map_err(|e| AudioSampleError::ProcessingError {
                msg: format!("Resampling failed: {}", e),
            })?;

    // Convert output back to AudioSamples format
    create_audio_samples_from_channels(output_data, target_sample_rate)
}

/// Helper function to perform resampling with a sinc-based resampler.
fn resample_with_sinc_resampler(
    audio: &AudioSamples<f32>,
    resampler: &mut SincFixedIn<f32>,
    target_sample_rate: usize,
) -> AudioSampleResult<AudioSamples<f32>> {
    let channels = audio.channels();

    // Prepare input data
    let mut input_data: Vec<Vec<f32>> = Vec::with_capacity(channels);

    if audio.is_mono() {
        let mono_data = audio.as_mono().ok_or(AudioSampleError::InvalidInput {
            msg: "Failed to get mono data".to_string(),
        })?;
        input_data.push(mono_data.to_vec());
    } else {
        let multi_data = audio
            .as_multi_channel()
            .ok_or(AudioSampleError::InvalidInput {
                msg: "Failed to get multi-channel data".to_string(),
            })?;

        for ch in 0..channels {
            let channel_data: Vec<f32> = multi_data.row(ch).to_vec();
            input_data.push(channel_data);
        }
    }

    // For SincFixedIn, we need to process in chunks
    let chunk_size = resampler.input_frames_max();
    let mut output_chunks: Vec<Vec<Vec<f32>>> = Vec::new();

    for chunk_start in (0..input_data[0].len()).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(input_data[0].len());

        // Extract chunk for each channel
        let mut chunk_data: Vec<Vec<f32>> = Vec::with_capacity(channels);
        for ch in 0..channels {
            chunk_data.push(input_data[ch][chunk_start..chunk_end].to_vec());
        }

        // Resample this chunk
        match resampler.process(&chunk_data, None) {
            Ok(output_chunk) => output_chunks.push(output_chunk),
            Err(e) => {
                return Err(AudioSampleError::ProcessingError {
                    msg: format!("Chunk resampling failed: {}", e),
                });
            }
        }
    }

    // Concatenate all output chunks
    if output_chunks.is_empty() {
        return Err(AudioSampleError::ProcessingError {
            msg: "No output chunks produced".to_string(),
        });
    }

    let mut combined_output: Vec<Vec<f32>> = vec![Vec::new(); channels];
    for chunk in output_chunks {
        for (ch, channel_data) in chunk.into_iter().enumerate() {
            combined_output[ch].extend(channel_data);
        }
    }

    create_audio_samples_from_channels(combined_output, target_sample_rate)
}

/// Helper function to create AudioSamples from channel data.
fn create_audio_samples_from_channels(
    channel_data: Vec<Vec<f32>>,
    sample_rate: usize,
) -> AudioSampleResult<AudioSamples<f32>> {
    if channel_data.is_empty() {
        return Err(AudioSampleError::InvalidInput {
            msg: "No channel data provided".to_string(),
        });
    }

    let channels = channel_data.len();

    if channels == 1 {
        // Mono case
        let mono_array = ndarray::Array1::from_vec(channel_data.into_iter().next().unwrap());
        Ok(AudioSamples::new_mono(mono_array, sample_rate as u32))
    } else {
        // Multi-channel case
        let samples_per_channel = channel_data[0].len();

        // Create interleaved data
        let mut interleaved = Vec::with_capacity(channels * samples_per_channel);
        for sample_idx in 0..samples_per_channel {
            for ch in 0..channels {
                interleaved.push(channel_data[ch][sample_idx]);
            }
        }

        // Reshape to (channels, samples_per_channel)
        let array = ndarray::Array2::from_shape_vec(
            (channels, samples_per_channel),
            channel_data.into_iter().flatten().collect(),
        )
        .map_err(|e| AudioSampleError::InvalidInput {
            msg: format!("Failed to create multi-channel array: {}", e),
        })?;

        Ok(AudioSamples::new_multi_channel(array, sample_rate as u32))
    }
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
pub fn resample_by_ratio<T>(
    audio: &AudioSamples<T>,
    ratio: f64,
    quality: ResamplingQuality,
) -> AudioSampleResult<AudioSamples<T>>
where
    T: crate::AudioSample
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive
        + crate::ConvertTo<f64>,
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<f32>: AudioTypeConversion<T>,
{
    if ratio <= 0.0 {
        return Err(AudioSampleError::InvalidInput {
            msg: format!("Invalid resampling ratio: {}", ratio),
        });
    }

    let input_sample_rate = audio.sample_rate() as f64;
    let target_sample_rate = (input_sample_rate * ratio).round() as usize;

    resample(audio, target_sample_rate, quality)
}

#[cfg(test)]

mod tests {
    use super::*;
    use crate::AudioSamples;
    use ndarray::{Array1, array};

    #[test]
    fn test_resample_mono() {
        // Create a suitable mono signal at 44.1 kHz
        // at least 1024 samples to ensure chunking works
        let samples = (0..1024)
            .map(|x| (x as f32 * 2.0 * std::f32::consts::PI * 440.0 / 44100.0).sin())
            .collect::<Vec<f32>>();

        let data = Array1::from_vec(samples);

        let audio = AudioSamples::new_mono(data, 44100);
        println!("Original samples: {:?}", audio);
        // Resample to 48 kHz
        let resampled = resample(&audio, 48000, ResamplingQuality::Medium).unwrap();

        assert_eq!(resampled.sample_rate(), 48000);
        assert_eq!(resampled.channels(), 1);
        println!("Resampled samples: {:?}", resampled);
        assert!(resampled.samples_per_channel() > 0);
    }

    #[test]
    fn test_resample_by_ratio() {
        let data = array![1.0f32, 0.0, -1.0, 0.0];
        let audio = AudioSamples::new_mono(data, 44100);

        // Upsample by 2x
        let upsampled = resample_by_ratio(&audio, 2.0, ResamplingQuality::Fast).unwrap();
        assert_eq!(upsampled.sample_rate(), 88200);

        // Downsample by 0.5x
        let downsampled = resample_by_ratio(&audio, 0.5, ResamplingQuality::Fast).unwrap();
        assert_eq!(downsampled.sample_rate(), 22050);
    }

    #[test]
    fn test_no_resampling_needed() {
        let data = array![1.0f32, 0.0, -1.0];
        let audio = AudioSamples::new_mono(data.clone(), 44100);

        // "Resample" to same rate
        let result = resample(&audio, 44100, ResamplingQuality::High).unwrap();
        assert_eq!(result.sample_rate(), 44100);

        // Should be identical
        let original_mono = audio.as_mono().unwrap();
        let result_mono = result.as_mono().unwrap();
        assert_eq!(original_mono.len(), result_mono.len());
    }

    #[test]
    fn test_invalid_ratio() {
        let data = array![1.0f32, 0.0];
        let audio = AudioSamples::new_mono(data, 44100);

        let result = resample_by_ratio(&audio, -1.0, ResamplingQuality::Fast);
        assert!(result.is_err());

        let result = resample_by_ratio(&audio, 0.0, ResamplingQuality::Fast);
        assert!(result.is_err());
    }
}
