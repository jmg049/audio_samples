//! Format detection and audio analysis utilities.
//!
//! This module provides functions for detecting and analyzing properties
//! of audio signals, such as sample rate, fundamental frequency, and
//! silence regions.

use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ConvertTo,
    I24,
};

use rustfft::{FftPlanner, num_complex::Complex};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Attempts to detect the sample rate of an audio signal based on its content.
///
/// This function analyzes the frequency content of the signal to estimate
/// the original sample rate. It's useful for validating sample rate metadata
/// or detecting resampled content.
///
/// # Arguments
/// * `audio` - The audio signal to analyze
///
/// # Returns
/// * `Some(sample_rate)` - The detected sample rate in Hz
/// * `None` - If sample rate cannot be reliably detected
pub fn detect_sample_rate<T: AudioSample>(audio: &AudioSamples<T>) -> AudioSampleResult<Option<u32>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'b> AudioSamples<T>: AudioTypeConversion<T>,
{
    let audio_f64 = audio.as_f64()?;

    // Use the first channel for analysis
    let data = match audio_f64.as_mono() {
        Some(mono) => mono
            .as_slice()
            .ok_or(AudioSampleError::ArrayLayoutError {
                message: "Mono samples must be contiguous".to_string(),
            })?
            .to_vec(),
        None => {
            // Use the first channel of multi-channel audio
            let multi = audio_f64
                .as_multi_channel()
                .ok_or(AudioSampleError::InvalidInput {
                    msg: "Audio must be multi-channel".to_string(),
                })?;
            multi
                .row(0)
                .as_slice()
                .ok_or(AudioSampleError::ArrayLayoutError {
                    message: "Multi-channel samples must be contiguous".to_string(),
                })?
                .to_vec()
        }
    };

    // Look for high-frequency cutoff patterns that might indicate resampling
    let spectrum = compute_spectrum(&data)?;
    let nyquist_freq = audio.sample_rate() as f64 / 2.0;

    // Analyze the spectrum for sharp cutoffs that might indicate resampling
    let detected_rate = analyze_spectrum_for_cutoff(&spectrum, nyquist_freq);

    Ok(detected_rate)
}

/// Detects the fundamental frequency of an audio signal.
///
/// This function uses spectral analysis to find the fundamental frequency
/// of the input signal. It's useful for pitch detection and harmonic analysis.
///
/// # Arguments
/// * `audio` - The audio signal to analyze
///
/// # Returns
/// * `Some(frequency)` - The detected fundamental frequency in Hz
/// * `None` - If no clear fundamental frequency is detected
pub fn detect_fundamental_frequency<T: AudioSample>(
    audio: &AudioSamples<T>,
) -> AudioSampleResult<Option<f64>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'b> AudioSamples<T>: AudioTypeConversion<T>,
{
    let audio_f64 = audio.as_f64()?;

    // Use the first channel for analysis
    let data = match audio_f64.as_mono() {
        Some(mono) => mono
            .as_slice()
            .ok_or(AudioSampleError::ArrayLayoutError {
                message: "Mono samples must be contiguous".to_string(),
            })?
            .to_vec(),
        None => {
            // Use the first channel of multi-channel audio
            let multi = audio_f64
                .as_multi_channel()
                .ok_or(AudioSampleError::InvalidInput {
                    msg: "Audio must be multi-channel".to_string(),
                })?;
            multi
                .row(0)
                .as_slice()
                .ok_or(AudioSampleError::ArrayLayoutError {
                    message: "Multi-channel samples must be contiguous".to_string(),
                })?
                .to_vec()
        }
    };

    if data.is_empty() {
        return Ok(None);
    }

    // Use autocorrelation method for fundamental frequency detection
    let fundamental = detect_fundamental_autocorrelation(&data, audio.sample_rate() as f64)?;

    Ok(fundamental)
}

/// Detects silence regions in an audio signal.
///
/// This function identifies time intervals where the audio signal
/// falls below a specified threshold, indicating silence or very quiet regions.
///
/// # Arguments
/// * `audio` - The audio signal to analyze
/// * `threshold` - The amplitude threshold below which samples are considered silence
///
/// # Returns
/// A vector of (start_time, end_time) tuples representing silence regions in seconds
pub fn detect_silence_regions<'a, T: AudioSample>(
    audio: &'a AudioSamples<T>,
    threshold: T,
) -> AudioSampleResult<Vec<(f64, f64)>> {
    let mut silence_regions = Vec::new();
    let mut in_silence = false;
    let mut silence_start = 0;

    let sample_rate = audio.sample_rate() as f64;
    let samples_per_second = sample_rate;
    let threshold: f64 = threshold.cast_into();

    // Helper function to check if a sample is below threshold (absolute value)
    let is_below_threshold = |sample: T| -> bool {
        let abs_val: f64 = sample.cast_into();
        let abs_val = abs_val.abs();
        abs_val < threshold
    };

    match audio.as_mono() {
        Some(mono) => {
            for (i, &sample) in mono.iter().enumerate() {
                if is_below_threshold(sample) {
                    if !in_silence {
                        silence_start = i;
                        in_silence = true;
                    }
                } else if in_silence {
                    let start_time = silence_start as f64 / samples_per_second;
                    let end_time = i as f64 / samples_per_second;
                    silence_regions.push((start_time, end_time));
                    in_silence = false;
                }
            }

            // Handle case where silence extends to the end
            if in_silence {
                let start_time = silence_start as f64 / samples_per_second;
                let end_time = mono.len() as f64 / samples_per_second;
                silence_regions.push((start_time, end_time));
            }
        }
        None => {
            // Multi-channel analysis - consider it silence only if ALL channels are below threshold
            let multi = audio
                .as_multi_channel()
                .ok_or(AudioSampleError::InvalidInput {
                    msg: "Audio must be multi-channel".to_string(),
                })?;
            for i in 0..multi.ncols() {
                let all_below_threshold =
                    (0..multi.nrows()).all(|ch| is_below_threshold(multi[(ch, i)]));

                if all_below_threshold {
                    if !in_silence {
                        silence_start = i;
                        in_silence = true;
                    }
                } else if in_silence {
                    let start_time = silence_start as f64 / samples_per_second;
                    let end_time = i as f64 / samples_per_second;
                    silence_regions.push((start_time, end_time));
                    in_silence = false;
                }
            }

            // Handle case where silence extends to the end
            if in_silence {
                let start_time = silence_start as f64 / samples_per_second;
                let end_time = multi.ncols() as f64 / samples_per_second;
                silence_regions.push((start_time, end_time));
            }
        }
    }

    Ok(silence_regions)
}

/// Detects the dynamic range of an audio signal.
///
/// This function analyzes the amplitude distribution of the signal
/// to determine its dynamic range characteristics.
///
/// # Arguments
/// * `audio` - The audio signal to analyze
///
/// # Returns
/// A tuple of (peak_amplitude, rms_amplitude, dynamic_range_db)
pub fn detect_dynamic_range<T: AudioSample>(
    audio: &AudioSamples<T>,
) -> AudioSampleResult<(f64, f64, f64)>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'b> AudioSamples<T>: AudioTypeConversion<T>,
{
    let audio_f64 = audio.as_f64()?;

    let data: Vec<f64> = match audio_f64.as_mono() {
        Some(mono) => mono
            .as_slice()
            .ok_or(AudioSampleError::ArrayLayoutError {
                message: "Mono samples must be contiguous".to_string(),
            })?
            .to_vec(),
        None => {
            // Flatten multi-channel audio
            let multi = audio_f64
                .as_multi_channel()
                .ok_or(AudioSampleError::InvalidInput {
                    msg: "Audio must be mult-channel".to_string(),
                })?;
            multi
                .as_slice()
                .ok_or(AudioSampleError::ArrayLayoutError {
                    message: "Multi-channels must be contiguous".to_string(),
                })?
                .to_vec()
        }
    };

    if data.is_empty() {
        return Ok((0.0, 0.0, 0.0));
    }

    // Calculate peak amplitude
    let peak_amplitude = data.iter().map(|&x| x.abs()).fold(0.0, f64::max);

    // Calculate RMS amplitude
    let rms_amplitude = (data.iter().map(|&x| x * x).sum::<f64>() / data.len() as f64).sqrt();

    // Calculate dynamic range in dB
    let dynamic_range_db = if rms_amplitude > 0.0_f64 {
        20.0 * (peak_amplitude / rms_amplitude).log10()
    } else {
        0.0
    };

    Ok((peak_amplitude, rms_amplitude, dynamic_range_db))
}

/// Detects clipping in an audio signal.
///
/// This function identifies regions where the audio signal is clipped
/// (reaches the maximum or minimum values and stays there).
///
/// # Arguments
/// * `audio` - The audio signal to analyze
/// * `threshold_ratio` - The ratio of max value to consider as clipping (default: 0.99)
///
/// # Returns
/// A vector of (start_time, end_time) tuples representing clipped regions in seconds
pub fn detect_clipping<'a, T: AudioSample>(
    audio: &'a AudioSamples<T>,
    threshold_ratio: f64,
) -> AudioSampleResult<Vec<(f64, f64)>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'b> AudioSamples<T>: AudioTypeConversion<T>,
{
    let mut clipped_regions = Vec::new();
    let mut in_clipped = false;
    let mut clipped_start = 0;

    let sample_rate = audio.sample_rate() as f64;

    // Determine clipping thresholds
    let max_val: f64 = T::MAX.cast_into();
    let min_val: f64 = T::MIN.cast_into();

    let upper_threshold: T = T::cast_from(max_val * threshold_ratio);
    let lower_threshold: T = T::cast_from(min_val * threshold_ratio);

    // TODO: Make constant function
    let is_clipped = |sample: T| -> AudioSampleResult<bool> {
        Ok(sample >= upper_threshold || sample <= lower_threshold)
    };

    match audio.as_mono() {
        Some(mono) => {
            for (i, &sample) in mono.iter().enumerate() {
                if is_clipped(sample)? {
                    if !in_clipped {
                        clipped_start = i;
                        in_clipped = true;
                    }
                } else if in_clipped {
                    let start_time = clipped_start as f64 / sample_rate;
                    let end_time = i as f64 / sample_rate;
                    clipped_regions.push((start_time, end_time));
                    in_clipped = false;
                }
            }

            // Handle case where clipping extends to the end
            if in_clipped {
                let start_time = clipped_start as f64 / sample_rate;
                let end_time = mono.len() as f64 / sample_rate;
                clipped_regions.push((start_time, end_time));
            }
        }
        None => {
            // Multi-channel analysis - consider it clipped if ANY channel is clipped
            let multi = audio
                .as_multi_channel()
                .ok_or(AudioSampleError::InvalidInput {
                    msg: "Audio must be mono or multi-channel".to_string(),
                })?;
            for i in 0..multi.ncols() {
                let mut any_clipped = false;
                for ch in 0..multi.nrows() {
                    if is_clipped(multi[(ch, i)])? {
                        any_clipped = true;
                        break;
                    }
                }

                if any_clipped {
                    if !in_clipped {
                        clipped_start = i;
                        in_clipped = true;
                    }
                } else if in_clipped {
                    let start_time = clipped_start as f64 / sample_rate;
                    let end_time = i as f64 / sample_rate;
                    clipped_regions.push((start_time, end_time));
                    in_clipped = false;
                }
            }

            // Handle case where clipping extends to the end
            if in_clipped {
                let start_time = clipped_start as f64 / sample_rate;
                let end_time = multi.ncols() as f64 / sample_rate;
                clipped_regions.push((start_time, end_time));
            }
        }
    }

    Ok(clipped_regions)
}

// ========================
// Enhanced Format Detection
// ========================

/// Audio format information detected from files or streams.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioFormat {
    pub format_type: AudioFormatType,
    pub sample_rate: u32,
    pub channels: u16,
    pub bit_depth: u16,
    pub codec: Option<String>,
    pub duration_samples: Option<u64>,
    pub container_format: Option<String>,
}

/// Supported audio format types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormatType {
    Wav,
    Flac,
    Mp3,
    Aac,
    Ogg,
    Raw,
    Unknown,
}

impl AudioFormatType {
    /// Get common file extensions for this format.
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            Self::Wav => &["wav", "wave"],
            Self::Flac => &["flac"],
            Self::Mp3 => &["mp3"],
            Self::Aac => &["aac", "m4a"],
            Self::Ogg => &["ogg", "oga"],
            Self::Raw => &["raw", "pcm"],
            Self::Unknown => &[],
        }
    }

    /// Get MIME type for this format.
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::Flac => "audio/flac",
            Self::Mp3 => "audio/mpeg",
            Self::Aac => "audio/aac",
            Self::Ogg => "audio/ogg",
            Self::Raw => "audio/raw",
            Self::Unknown => "application/octet-stream",
        }
    }
}

/// Detect audio format from file path and/or content.
pub fn detect_audio_format<P: AsRef<Path>>(path: P) -> AudioSampleResult<AudioFormat> {
    let path = path.as_ref();

    // First try to detect from file extension
    let format_type = detect_format_from_extension(path);

    // Then try to detect from file content if we can read it
    if let Ok(mut file) = std::fs::File::open(path) {
        let detected_format = detect_format_from_header(&mut file)?;
        Ok(detected_format)
    } else {
        // Fallback to extension-based detection with defaults
        Ok(AudioFormat {
            format_type,
            sample_rate: 44100,
            channels: 2,
            bit_depth: 16,
            codec: None,
            duration_samples: None,
            container_format: None,
        })
    }
}

/// Detect format from file extension.
pub fn detect_format_from_extension<P: AsRef<Path>>(path: P) -> AudioFormatType {
    let path = path.as_ref();
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    match extension.as_str() {
        "wav" | "wave" => AudioFormatType::Wav,
        "flac" => AudioFormatType::Flac,
        "mp3" => AudioFormatType::Mp3,
        "aac" | "m4a" => AudioFormatType::Aac,
        "ogg" | "oga" => AudioFormatType::Ogg,
        "raw" | "pcm" => AudioFormatType::Raw,
        _ => AudioFormatType::Unknown,
    }
}

/// Detect format from file header/magic bytes.
pub fn detect_format_from_header<R: Read + Seek>(reader: &mut R) -> AudioSampleResult<AudioFormat> {
    let mut header = [0u8; 12];
    reader
        .read_exact(&mut header)
        .map_err(|e| AudioSampleError::InvalidInput {
            msg: format!("Failed to read header: {}", e),
        })?;

    // Reset to beginning
    reader
        .seek(SeekFrom::Start(0))
        .map_err(|e| AudioSampleError::InvalidInput {
            msg: format!("Failed to seek: {}", e),
        })?;

    // Check magic bytes/signatures
    if &header[0..4] == b"RIFF" && &header[8..12] == b"WAVE" {
        parse_wav_header(reader)
    } else if &header[0..4] == b"fLaC" {
        parse_flac_header(reader)
    } else if header[0] == 0xFF && (header[1] & 0xE0) == 0xE0 {
        // MP3 frame header
        parse_mp3_header(reader)
    } else if &header[0..4] == b"OggS" {
        parse_ogg_header(reader)
    } else {
        // Unknown format, return basic info
        Ok(AudioFormat {
            format_type: AudioFormatType::Unknown,
            sample_rate: 44100,
            channels: 2,
            bit_depth: 16,
            codec: None,
            duration_samples: None,
            container_format: None,
        })
    }
}

/// Detect bit depth from raw audio stream analysis.
pub fn detect_bit_depth<T: AudioSample>(audio: &AudioSamples<T>) -> AudioSampleResult<u16>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'b> AudioSamples<T>: AudioTypeConversion<T>,
{
    let audio_f64 = audio.as_f64()?;

    let data: Vec<f64> = match audio_f64.as_mono() {
        Some(mono) => mono
            .as_slice()
            .ok_or(AudioSampleError::ArrayLayoutError {
                message: "Mono samples must be contiguous".to_string(),
            })?
            .to_vec(),
        None => {
            // Flatten multi-channel audio
            let multi = audio_f64
                .as_multi_channel()
                .ok_or(AudioSampleError::InvalidInput {
                    msg: "Audio must be multi-channel".to_string(),
                })?;
            multi
                .as_slice()
                .ok_or(AudioSampleError::ArrayLayoutError {
                    message: "Multi-channel samples must be contiguous".to_string(),
                })?
                .to_vec()
        }
    };

    if data.is_empty() {
        return Ok(16); // Default
    }

    // Analyze quantization patterns to estimate bit depth
    let unique_values: std::collections::HashSet<_> = data
        .iter()
        .map(|&x| (x * 65536.0) as i32) // Scale to detect quantization
        .collect();

    let unique_count = unique_values.len();
    let _total_samples = data.len();

    // Estimate bit depth based on unique value distribution
    let estimated_bits = if unique_count < 256 {
        8
    } else if unique_count < 65536 {
        16
    } else if unique_count < 16777216 {
        24
    } else {
        32
    };

    Ok(estimated_bits)
}

/// Detect channel configuration from audio stream.
pub fn detect_channel_config<'a, T: AudioSample>(
    audio: &'a AudioSamples<T>,
) -> ChannelConfiguration {
    let channels = audio.num_channels();

    match channels {
        1 => ChannelConfiguration::Mono,
        2 => {
            // Could be stereo or dual mono - analyze correlation
            if let Some(multi) = audio.as_multi_channel() {
                if multi.nrows() >= 2 {
                    let left = multi.row(0);
                    let right = multi.row(1);

                    // Calculate correlation between channels
                    let correlation = calculate_channel_correlation(
                        left.as_slice().unwrap_or(&[]),
                        right.as_slice().unwrap_or(&[]),
                    );

                    if correlation > 0.95 {
                        ChannelConfiguration::DualMono
                    } else {
                        ChannelConfiguration::Stereo
                    }
                } else {
                    ChannelConfiguration::Stereo
                }
            } else {
                ChannelConfiguration::Stereo
            }
        }
        3 => ChannelConfiguration::Surround3_0,
        4 => ChannelConfiguration::Surround4_0,
        5 => ChannelConfiguration::Surround5_0,
        6 => ChannelConfiguration::Surround5_1,
        8 => ChannelConfiguration::Surround7_1,
        _ => ChannelConfiguration::Other(channels),
    }
}

/// Channel configuration types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelConfiguration {
    Mono,
    Stereo,
    DualMono,
    Surround3_0,
    Surround4_0,
    Surround5_0,
    Surround5_1,
    Surround7_1,
    Other(usize),
}

// ========================
// Metadata Extraction
// ========================

/// Audio metadata extracted from files or streams.
#[derive(Debug, Clone, Default)]
pub struct AudioMetadata {
    pub format_info: Option<AudioFormat>,
    pub duration_seconds: Option<f64>,
    pub file_size_bytes: Option<u64>,
    pub bitrate_kbps: Option<u32>,
    pub quality_metrics: QualityMetrics,
    pub tags: HashMap<String, String>,
}

/// Quality metrics for audio analysis.
#[derive(Debug, Clone, Default)]
pub struct QualityMetrics {
    pub peak_amplitude: f64,
    pub rms_amplitude: f64,
    pub dynamic_range_db: f64,
    pub thd_plus_n_percent: Option<f64>,
    pub snr_db: Option<f64>,
    pub bit_depth_effective: Option<u16>,
    pub frequency_response_range: Option<(f64, f64)>,
    pub clipping_detected: bool,
    pub noise_floor_db: Option<f64>,
}

/// Extract comprehensive metadata from audio file.
pub fn extract_audio_metadata<P: AsRef<Path>>(path: P) -> AudioSampleResult<AudioMetadata> {
    let path = path.as_ref();
    let mut metadata = AudioMetadata::default();

    // Get file size
    if let Ok(file_metadata) = path.metadata() {
        metadata.file_size_bytes = Some(file_metadata.len());
    }

    // Detect format
    metadata.format_info = Some(detect_audio_format(path)?);

    // Calculate duration if we have format info
    if let Some(ref format) = metadata.format_info {
        if let Some(duration_samples) = format.duration_samples {
            metadata.duration_seconds = Some(duration_samples as f64 / format.sample_rate as f64);
        }

        // Estimate bitrate
        if let (Some(duration), Some(file_size)) =
            (metadata.duration_seconds, metadata.file_size_bytes)
        {
            if duration > 0.0 {
                let bitrate = (file_size * 8) as f64 / duration / 1000.0; // kbps
                metadata.bitrate_kbps = Some(bitrate as u32);
            }
        }
    }

    Ok(metadata)
}

/// Calculate quality metrics from audio samples.
pub fn calculate_quality_metrics<T: AudioSample>(
    audio: &AudioSamples<T>,
) -> AudioSampleResult<QualityMetrics>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'b> AudioSamples<T>: AudioTypeConversion<T>,
{
    let mut metrics = QualityMetrics::default();

    // Basic dynamic range analysis
    let (peak, rms, dynamic_range) = detect_dynamic_range(audio)?;
    metrics.peak_amplitude = peak;
    metrics.rms_amplitude = rms;
    metrics.dynamic_range_db = dynamic_range;

    // Clipping detection
    let clipped_regions = detect_clipping(audio, 0.99)?;
    metrics.clipping_detected = !clipped_regions.is_empty();

    // Effective bit depth
    metrics.bit_depth_effective = Some(detect_bit_depth(audio)?);

    // Estimate noise floor
    metrics.noise_floor_db = estimate_noise_floor(audio)?;

    // Frequency response analysis
    metrics.frequency_response_range = estimate_frequency_range(audio)?;

    Ok(metrics)
}

/// Estimate noise floor of audio signal.
fn estimate_noise_floor<T: AudioSample>(audio: &AudioSamples<T>) -> AudioSampleResult<Option<f64>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'b> AudioSamples<T>: AudioTypeConversion<T>,
{
    let audio_f64 = audio.as_f64()?;

    let data: Vec<f64> = match audio_f64.as_mono() {
        Some(mono) => mono
            .as_slice()
            .ok_or(AudioSampleError::ArrayLayoutError {
                message: "Mono samples must be contiguous".to_string(),
            })?
            .to_vec(),
        None => {
            // Use first channel
            let multi = audio_f64
                .as_multi_channel()
                .ok_or(AudioSampleError::InvalidInput {
                    msg: "Audio must be multi-channel".to_string(),
                })?;
            multi
                .row(0)
                .as_slice()
                .ok_or(AudioSampleError::ArrayLayoutError {
                    message: "Multi-channel samples must be contiguous".to_string(),
                })?
                .to_vec()
        }
    };

    if data.is_empty() {
        return Ok(None);
    }

    // Find quietest regions (bottom 10th percentile)
    let mut sorted_abs: Vec<f64> = data.iter().map(|&x| x.abs()).collect();
    sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let percentile_10 = sorted_abs.len() / 10;
    if percentile_10 > 0 {
        let noise_level = sorted_abs[percentile_10];
        if noise_level > 0.0 {
            let noise_floor_db = 20.0 * noise_level.log10();
            return Ok(Some(noise_floor_db));
        }
    }

    Ok(None)
}

/// Estimate frequency response range.
fn estimate_frequency_range<T: AudioSample>(
    audio: &AudioSamples<T>,
) -> AudioSampleResult<Option<(f64, f64)>>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    for<'b> AudioSamples<T>: AudioTypeConversion<T>,
{
    let audio_f64 = audio.as_f64()?;

    let data: Vec<f64> = match audio_f64.as_mono() {
        Some(mono) => mono
            .as_slice()
            .ok_or(AudioSampleError::ArrayLayoutError {
                message: "Mono samples must be contiguous".to_string(),
            })?
            .to_vec(),
        None => {
            // Use first channel
            let multi = audio_f64
                .as_multi_channel()
                .ok_or(AudioSampleError::InvalidInput {
                    msg: "Audio must be multi-channel".to_string(),
                })?;
            multi
                .row(0)
                .as_slice()
                .ok_or(AudioSampleError::ArrayLayoutError {
                    message: "Multi-channel samples must be contiguous".to_string(),
                })?
                .to_vec()
        }
    };

    if data.len() < 1024 {
        return Ok(None); // Too short for meaningful analysis
    }

    // Compute spectrum
    let spectrum = compute_spectrum(&data)?;
    let sample_rate = audio.sample_rate() as f64;
    let nyquist = sample_rate / 2.0;

    // Find frequency range with significant energy
    let threshold = spectrum.iter().fold(0.0_f64, |acc, &x| acc.max(x)) * 0.01; // 1% of peak

    let mut low_freq = None;
    let mut high_freq = None;

    for (i, &energy) in spectrum.iter().enumerate() {
        if energy > threshold {
            let freq = (i as f64 / spectrum.len() as f64) * nyquist;
            if low_freq.is_none() {
                low_freq = Some(freq);
            }
            high_freq = Some(freq);
        }
    }

    if let (Some(low), Some(high)) = (low_freq, high_freq) {
        Ok(Some((low, high)))
    } else {
        Ok(None)
    }
}

// ========================
// Format-specific parsers
// ========================

fn parse_wav_header<R: Read + Seek>(reader: &mut R) -> AudioSampleResult<AudioFormat> {
    let mut buffer = [0u8; 44]; // Standard WAV header size
    reader
        .read_exact(&mut buffer)
        .map_err(|e| AudioSampleError::InvalidInput {
            msg: format!("Failed to read WAV header: {}", e),
        })?;

    // Parse WAV header fields
    let sample_rate = u32::from_le_bytes([buffer[24], buffer[25], buffer[26], buffer[27]]);
    let channels = u16::from_le_bytes([buffer[22], buffer[23]]);
    let bits_per_sample = u16::from_le_bytes([buffer[34], buffer[35]]);

    // Calculate duration from chunk size
    let data_size = u32::from_le_bytes([buffer[40], buffer[41], buffer[42], buffer[43]]);
    let bytes_per_sample = (bits_per_sample / 8) as u32;
    let duration_samples = if bytes_per_sample > 0 {
        Some((data_size / bytes_per_sample / channels as u32) as u64)
    } else {
        None
    };

    Ok(AudioFormat {
        format_type: AudioFormatType::Wav,
        sample_rate,
        channels,
        bit_depth: bits_per_sample,
        codec: Some("PCM".to_string()),
        duration_samples,
        container_format: Some("RIFF".to_string()),
    })
}

fn parse_flac_header<R: Read + Seek>(_reader: &mut R) -> AudioSampleResult<AudioFormat> {
    // Simplified FLAC header parsing - real implementation would be more complex
    Ok(AudioFormat {
        format_type: AudioFormatType::Flac,
        sample_rate: 44100, // Default, should parse from STREAMINFO
        channels: 2,
        bit_depth: 16,
        codec: Some("FLAC".to_string()),
        duration_samples: None,
        container_format: Some("FLAC".to_string()),
    })
}

fn parse_mp3_header<R: Read + Seek>(_reader: &mut R) -> AudioSampleResult<AudioFormat> {
    // Simplified MP3 header parsing
    Ok(AudioFormat {
        format_type: AudioFormatType::Mp3,
        sample_rate: 44100,
        channels: 2,
        bit_depth: 16, // MP3 is lossy, but equivalent to ~16-bit
        codec: Some("MP3".to_string()),
        duration_samples: None,
        container_format: Some("MPEG".to_string()),
    })
}

fn parse_ogg_header<R: Read + Seek>(_reader: &mut R) -> AudioSampleResult<AudioFormat> {
    // Simplified OGG header parsing
    Ok(AudioFormat {
        format_type: AudioFormatType::Ogg,
        sample_rate: 44100,
        channels: 2,
        bit_depth: 16,
        codec: Some("Vorbis".to_string()),
        duration_samples: None,
        container_format: Some("OGG".to_string()),
    })
}

fn calculate_channel_correlation<T: AudioSample>(left: &[T], right: &[T]) -> f64 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }

    let left_f64: Vec<f64> = left.iter().map(|&x| x.cast_into()).collect();
    let right_f64: Vec<f64> = right.iter().map(|&x| x.cast_into()).collect();

    // Calculate Pearson correlation coefficient
    let n = left_f64.len() as f64;
    let mean_left = left_f64.iter().sum::<f64>() / n;
    let mean_right = right_f64.iter().sum::<f64>() / n;

    let numerator: f64 = left_f64
        .iter()
        .zip(right_f64.iter())
        .map(|(&l, &r)| (l - mean_left) * (r - mean_right))
        .sum();

    let sum_sq_left: f64 = left_f64.iter().map(|&l| (l - mean_left).powi(2)).sum();

    let sum_sq_right: f64 = right_f64.iter().map(|&r| (r - mean_right).powi(2)).sum();

    let denominator = (sum_sq_left * sum_sq_right).sqrt();

    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

// Helper functions

fn compute_spectrum(data: &[f64]) -> AudioSampleResult<Vec<f64>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(data.len());

    let mut buffer: Vec<Complex<f64>> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut buffer);

    let spectrum: Vec<f64> = buffer.iter().map(|c| c.norm()).collect();
    Ok(spectrum)
}

fn analyze_spectrum_for_cutoff(spectrum: &[f64], nyquist_freq: f64) -> Option<u32> {
    // Look for sharp cutoffs that might indicate resampling
    let len = spectrum.len();
    let half_len = len / 2;

    // Check for energy drops that might indicate filtering
    let mut candidates = Vec::new();

    // Common resampling target frequencies
    let common_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000, 88200, 96000];

    for &rate in &common_rates {
        let target_nyquist = rate as f64 / 2.0;
        if target_nyquist < nyquist_freq {
            // Check if there's a significant drop in energy around this frequency
            let bin_index = (target_nyquist / nyquist_freq * half_len as f64) as usize;
            if bin_index < spectrum.len() - 10 {
                let energy_before = spectrum[bin_index.saturating_sub(5)..bin_index]
                    .iter()
                    .sum::<f64>();
                let energy_after = spectrum[bin_index..bin_index + 10].iter().sum::<f64>();

                if energy_before > energy_after * 2.0 {
                    candidates.push(rate);
                }
            }
        }
    }

    // Return the most likely candidate
    candidates.first().copied()
}

fn detect_fundamental_autocorrelation(
    data: &[f64],
    sample_rate: f64,
) -> AudioSampleResult<Option<f64>> {
    if data.len() < 2 {
        return Ok(None);
    }

    let min_freq = 50.0; // Minimum fundamental frequency to detect
    let max_freq = 2000.0; // Maximum fundamental frequency to detect

    let min_period = (sample_rate / max_freq) as usize;
    let max_period = (sample_rate / min_freq) as usize;

    if max_period >= data.len() {
        return Ok(None);
    }

    let mut max_correlation = 0.0;
    let mut best_period = 0;

    for period in min_period..max_period.min(data.len() / 2) {
        let mut correlation = 0.0;
        let mut count = 0;

        for i in 0..(data.len() - period) {
            correlation += data[i] * data[i + period];
            count += 1;
        }

        if count > 0 {
            correlation /= count as f64;

            if correlation > max_correlation {
                max_correlation = correlation;
                best_period = period;
            }
        }
    }

    if best_period > 0 && max_correlation > 0.1 {
        let fundamental_freq = sample_rate / best_period as f64;
        Ok(Some(fundamental_freq))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::f64::consts::PI;

    #[test]
    fn test_detect_silence_regions() {
        // Create a signal with silence regions
        let data = array![0.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let audio = AudioSamples::new_mono(data.into(), 10); // 10 Hz sample rate for easy calculation

        let silence_regions =
            detect_silence_regions(&audio, 0.5).expect("Failed to detect silence");

        assert!(!silence_regions.is_empty());
        // First silence region should be at the beginning
        assert_eq!(silence_regions[0].0, 0.0);
        assert_eq!(silence_regions[0].1, 0.3);
    }

    #[test]
    fn test_detect_dynamic_range() {
        // Create a signal with known dynamic range
        let data = array![0.1f32, 0.5, 1.0, 0.2, 0.8];
        let audio = AudioSamples::new_mono(data, 44100);

        let (peak, rms, dynamic_range) = detect_dynamic_range(&audio).unwrap();

        assert_eq!(peak, 1.0);
        assert!(rms > 0.0);
        assert!(dynamic_range > 0.0);
    }

    #[test]
    fn test_detect_clipping() {
        // Create a signal with clipping
        let data = array![0.5f32, 1.0, 1.0, 1.0, 0.5, -1.0, -1.0, 0.0];
        let audio = AudioSamples::new_mono(data.into(), 8); // 8 Hz sample rate for easy calculation

        let clipped_regions = detect_clipping(&audio, 0.99).expect("Failed to detect clipping");

        assert!(!clipped_regions.is_empty());
    }

    #[test]
    fn test_detect_fundamental_frequency() {
        // Create a sine wave with known frequency
        let sample_rate = 44100.0;
        let frequency = 440.0; // A4
        let duration = 1.0; // 1 second

        let samples: Vec<f32> = (0..(sample_rate * duration) as usize)
            .map(|i| {
                let t = i as f64 / sample_rate;
                (2.0 * PI * frequency * t).sin() as f32
            })
            .collect();

        let data = ndarray::Array1::from_vec(samples);
        let audio = AudioSamples::new_mono(data, sample_rate as u32);

        let detected_freq = detect_fundamental_frequency(&audio).unwrap();

        if let Some(freq) = detected_freq {
            // Allow some tolerance in frequency detection
            assert!((freq - frequency).abs() < 10.0);
        }
    }
}
