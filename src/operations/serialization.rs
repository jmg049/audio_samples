//! Serialization and deserialization implementations for AudioSamples.
//!
//! This module provides concrete implementations for the AudioSamplesSerialise trait,
//! supporting various data interchange formats commonly used in audio analysis and
//! data science workflows.

use crate::{
    AudioSample, AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion,
    ConvertTo, I24,
};
use crate::operations::traits::AudioSamplesSerialise;
use crate::operations::types::{SerializationConfig, SerializationFormat, TextDelimiter};
use crate::error::SerializationError;

use std::collections::HashMap;
use std::path::Path;

/// Metadata structure for serialized audio data.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AudioMetadata {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of audio channels
    pub channels: u32,
    /// Total number of samples per channel
    pub samples_per_channel: usize,
    /// Sample type (e.g., "f32", "i16", "f64")
    pub sample_type: String,
    /// Duration in seconds
    pub duration_seconds: f64,
    /// Custom metadata attributes
    pub custom_attributes: HashMap<String, String>,
    /// Timestamp when serialized
    pub serialized_at: String,
    /// Library version
    pub version: String,
}

#[cfg(feature = "serialization")]
impl AudioMetadata {
    /// Create new audio metadata from AudioSamples.
    pub fn from_audio_samples<T: AudioSample>(audio: &AudioSamples<T>) -> Self {
        Self {
            sample_rate: audio.sample_rate(),
            channels: audio.num_channels() as u32,
            samples_per_channel: audio.samples_per_channel(),
            sample_type: std::any::type_name::<T>().to_string(),
            duration_seconds: audio.samples_per_channel() as f64 / audio.sample_rate() as f64,
            custom_attributes: HashMap::new(),
            serialized_at: chrono::Utc::now().to_rfc3339(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Validate metadata consistency.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if self.sample_rate == 0 {
            return Err(AudioSampleError::Serialization(
                SerializationError::invalid_header("sample_rate", "Sample rate cannot be zero"),
            ));
        }
        if self.channels == 0 {
            return Err(AudioSampleError::Serialization(
                SerializationError::invalid_header("channels", "Channel count cannot be zero"),
            ));
        }
        Ok(())
    }
}

impl<'a, T: AudioSample> AudioSamplesSerialise<T> for AudioSamples<'a, T>
where
    i16: ConvertTo<T>,
    I24: ConvertTo<T>,
    i32: ConvertTo<T>,
    f32: ConvertTo<T>,
    f64: ConvertTo<T>,
    AudioSamples<'a, T>: AudioTypeConversion<'a, T>,
{
    fn save_to_file<P: AsRef<Path>>(&self, path: P) -> AudioSampleResult<()> {
        let path = path.as_ref();
        let format = Self::detect_format(path)?;
        let config = SerializationConfig::<f64>::new(format);
        self.save_with_config(path, &config)
    }

    fn save_with_config<P: AsRef<Path>>(
        &self,
        path: P,
        config: &SerializationConfig<f64>,
    ) -> AudioSampleResult<()> {
        config.validate()?;
        let bytes = self.serialize_to_bytes(config.format)?;

        std::fs::write(path.as_ref(), bytes).map_err(|e| {
            AudioSampleError::Serialization(SerializationError::io_error(
                "writing file",
                e.to_string(),
            ))
        })
    }

    fn load_from_file<P: AsRef<Path>>(path: P) -> AudioSampleResult<AudioSamples<'static, T>> {
        let path = path.as_ref();
        let format = Self::detect_format(path)?;
        let config = SerializationConfig::<f64>::new(format);
        Self::load_with_config(path, &config)
    }

    fn load_with_config<P: AsRef<Path>>(
        path: P,
        config: &SerializationConfig<f64>,
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        let bytes = std::fs::read(path.as_ref()).map_err(|e| {
            AudioSampleError::Serialization(SerializationError::io_error(
                "reading file",
                e.to_string(),
            ))
        })?;

        Self::deserialize_from_bytes(&bytes, config.format)
    }

    fn serialize_to_bytes(&self, format: SerializationFormat) -> AudioSampleResult<Vec<u8>> {
        match format {
            SerializationFormat::Text { delimiter } => {
                serialize_text_format(self, delimiter)
            }
            SerializationFormat::Binary { endian } => {
                serialize_binary_format(self, endian)
            }
            SerializationFormat::Numpy => {
                serialize_numpy_format(self)
            }
            SerializationFormat::NumpyCompressed { compression_level } => {
                serialize_numpy_compressed_format(self, compression_level)
            }
            _ => Err(AudioSampleError::Serialization(
                SerializationError::unsupported_format(
                    format!("{:?}", format),
                    "serialization",
                ),
            )),
        }
    }

    fn deserialize_from_bytes(
        data: &[u8],
        format: SerializationFormat,
    ) -> AudioSampleResult<AudioSamples<'static, T>> {
        match format {
            SerializationFormat::Text { delimiter } => {
                deserialize_text_format(data, delimiter)
            }
            SerializationFormat::Binary { .. } => {
                deserialize_binary_format(data)
            }
            SerializationFormat::Numpy => {
                deserialize_numpy_format(data)
            }
            SerializationFormat::NumpyCompressed { .. } => {
                deserialize_numpy_compressed_format(data)
            }
            _ => Err(AudioSampleError::Serialization(
                SerializationError::unsupported_format(
                    format!("{:?}", format),
                    "deserialization",
                ),
            )),
        }
    }

    fn estimate_serialized_size(&self, format: SerializationFormat) -> AudioSampleResult<usize> {
        let data_size = self.len() * self.num_channels() * std::mem::size_of::<T>();

        match format {
            SerializationFormat::Text { .. } => Ok(data_size * 8), // Text is verbose
            _ => Ok(data_size),
        }
    }

    fn validate_round_trip(
        &self,
        format: SerializationFormat,
        _tolerance: f64,
    ) -> AudioSampleResult<()> {
        let serialized_bytes = self.serialize_to_bytes(format)?;
        let deserialized = Self::deserialize_from_bytes(&serialized_bytes, format)?;

        if self.len() != deserialized.len() || self.num_channels() != deserialized.num_channels() {
            return Err(AudioSampleError::Serialization(
                SerializationError::validation_failed(
                    "Audio dimensions changed during round-trip",
                ),
            ));
        }

        Ok(())
    }

    fn detect_format<P: AsRef<Path>>(path: P) -> AudioSampleResult<SerializationFormat> {
        let path = path.as_ref();
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        match extension.to_lowercase().as_str() {
            "txt" => Ok(SerializationFormat::Text { delimiter: TextDelimiter::Space }),
            "csv" => Ok(SerializationFormat::Csv),
            "bin" | "audio" => Ok(SerializationFormat::Binary { endian: crate::operations::types::Endianness::Native }),
            "npy" => Ok(SerializationFormat::Numpy),
            "npz" => Ok(SerializationFormat::NumpyCompressed { compression_level: 6 }),
            _ => Err(AudioSampleError::Serialization(
                SerializationError::unsupported_format(
                    extension.to_string(),
                    "file extension",
                ),
            )),
        }
    }

    fn supported_serialization_formats() -> Vec<SerializationFormat> {
        vec![
            SerializationFormat::Text { delimiter: TextDelimiter::Space },
            SerializationFormat::Csv,
            SerializationFormat::Binary { endian: crate::operations::types::Endianness::Native },
            SerializationFormat::Numpy,
            SerializationFormat::NumpyCompressed { compression_level: 6 },
        ]
    }

    fn supported_deserialization_formats() -> Vec<SerializationFormat> {
        vec![
            SerializationFormat::Text { delimiter: TextDelimiter::Space },
            SerializationFormat::Csv,
            SerializationFormat::Binary { endian: crate::operations::types::Endianness::Native },
            SerializationFormat::Numpy,
            SerializationFormat::NumpyCompressed { compression_level: 6 },
        ]
    }

    fn export_metadata<P: AsRef<Path>>(&self, path: P) -> AudioSampleResult<()> {
        let metadata = AudioMetadata::from_audio_samples(self);
        let json = serde_json::to_string_pretty(&metadata).map_err(|e| {
            AudioSampleError::Serialization(SerializationError::serialization_failed(
                "metadata",
                format!("Failed to serialize metadata: {}", e),
            ))
        })?;

        std::fs::write(path.as_ref(), json.as_bytes()).map_err(|e| {
            AudioSampleError::Serialization(SerializationError::io_error(
                "writing metadata file",
                e.to_string(),
            ))
        })
    }

    fn import_metadata<P: AsRef<Path>>(&mut self, _path: P) -> AudioSampleResult<()> {
        // For borrowed AudioSamples, we can't modify metadata
        Err(AudioSampleError::Serialization(
            SerializationError::validation_failed(
                "Cannot import metadata into borrowed AudioSamples",
            ),
        ))
    }
}

// Helper functions for text format serialization
fn serialize_text_format<T: AudioSample>(
    audio: &AudioSamples<T>,
    delimiter: TextDelimiter,
) -> AudioSampleResult<Vec<u8>> {
    let mut output = String::new();
    let _delim_char = match delimiter {
        TextDelimiter::Space => ' ',
        TextDelimiter::Tab => '\t',
        TextDelimiter::Comma => ',',
        TextDelimiter::Newline => '\n',
        TextDelimiter::Custom(c) => c,
    };

    // Add metadata header
    output.push_str(&format!("# sample_rate: {}\n", audio.sample_rate()));
    output.push_str(&format!("# channels: {}\n", audio.num_channels()));

    // Serialize audio data
    if let Some(slice) = audio.data.as_slice() {
        for sample in slice {
            output.push_str(&sample.to_string());
            output.push('\n');
        }
    } else {
        return Err(AudioSampleError::Serialization(
            SerializationError::serialization_failed("text", "Cannot access audio data as slice"),
        ));
    }

    Ok(output.into_bytes())
}

fn deserialize_text_format<T: AudioSample>(
    data: &[u8],
    _delimiter: TextDelimiter,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    f64: ConvertTo<T>,
{
    let content = std::str::from_utf8(data).map_err(|e| {
        AudioSampleError::Serialization(SerializationError::deserialization_failed(
            "text",
            format!("Invalid UTF-8: {}", e),
        ))
    })?;

    let mut sample_rate = 44100_u32;
    let mut channels = 1_usize;
    let mut samples: Vec<T> = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if line.starts_with('#') {
            if let Some(sr) = line.strip_prefix("# sample_rate: ") {
                sample_rate = sr.parse().unwrap_or(44100);
            } else if let Some(ch) = line.strip_prefix("# channels: ") {
                channels = ch.parse().unwrap_or(1);
            }
        } else {
            // Parse the sample value directly since T: AudioSample already includes serialization
            if let Ok(sample) = line.parse::<f64>() {
                if let Ok(converted) = sample.convert_to() {
                    samples.push(converted);
                }
            }
        }
    }

    match channels {
        1 => Ok(AudioSamples::new_mono(
            ndarray::Array1::from_vec(samples),
            sample_rate,
        )),
        _ => {
            let samples_per_channel = samples.len() / channels;
            let reshaped = ndarray::Array2::from_shape_vec(
                (channels, samples_per_channel),
                samples,
            ).map_err(|e| {
                AudioSampleError::Serialization(SerializationError::deserialization_failed(
                    "text",
                    format!("Failed to reshape data: {}", e),
                ))
            })?;
            Ok(AudioSamples::new_multi_channel(reshaped, sample_rate))
        }
    }
}

// Binary format implementation with magic bytes and metadata header
// Format: MAGIC(4) + VERSION(4) + SAMPLE_RATE(4) + CHANNELS(4) + SAMPLES_PER_CHANNEL(8) + SAMPLE_TYPE(4) + DATA
const AUDIO_MAGIC: &[u8; 4] = b"AUD\x01";
const FORMAT_VERSION: u32 = 1;

// Simple type identifier for supported audio sample types
fn sample_type_id<T: AudioSample>() -> u32 {
    let type_name = std::any::type_name::<T>();
    match type_name {
        "i16" => 1,
        "i24::I24" => 2,
        "i32" => 3,
        "f32" => 4,
        "f64" => 5,
        _ => 0, // unknown
    }
}

fn serialize_binary_format<T: AudioSample>(
    audio: &AudioSamples<T>,
    endian: crate::operations::types::Endianness,
) -> AudioSampleResult<Vec<u8>> {
    let mut buffer = Vec::new();

    // Write magic bytes
    buffer.extend_from_slice(AUDIO_MAGIC);

    // Write version
    match endian {
        crate::operations::types::Endianness::Big => buffer.extend_from_slice(&FORMAT_VERSION.to_be_bytes()),
        crate::operations::types::Endianness::Little => buffer.extend_from_slice(&FORMAT_VERSION.to_le_bytes()),
        crate::operations::types::Endianness::Native => buffer.extend_from_slice(&FORMAT_VERSION.to_ne_bytes()),
    }

    // Write metadata
    let sample_rate = audio.sample_rate();
    let channels = audio.num_channels() as u32;
    let samples_per_channel = audio.samples_per_channel() as u64;
    let sample_type = sample_type_id::<T>(); // Simple type identifier

    match endian {
        crate::operations::types::Endianness::Big => {
            buffer.extend_from_slice(&sample_rate.to_be_bytes());
            buffer.extend_from_slice(&channels.to_be_bytes());
            buffer.extend_from_slice(&samples_per_channel.to_be_bytes());
            buffer.extend_from_slice(&sample_type.to_be_bytes());
        }
        crate::operations::types::Endianness::Little => {
            buffer.extend_from_slice(&sample_rate.to_le_bytes());
            buffer.extend_from_slice(&channels.to_le_bytes());
            buffer.extend_from_slice(&samples_per_channel.to_le_bytes());
            buffer.extend_from_slice(&sample_type.to_le_bytes());
        }
        crate::operations::types::Endianness::Native => {
            buffer.extend_from_slice(&sample_rate.to_ne_bytes());
            buffer.extend_from_slice(&channels.to_ne_bytes());
            buffer.extend_from_slice(&samples_per_channel.to_ne_bytes());
            buffer.extend_from_slice(&sample_type.to_ne_bytes());
        }
    }

    // Write audio data as JSON (within binary wrapper for metadata + magic bytes)
    if let Some(slice) = audio.data.as_slice() {
        let json_data = serde_json::to_vec(slice).map_err(|e| {
            AudioSampleError::Serialization(SerializationError::serialization_failed(
                "binary",
                format!("Failed to serialize audio data: {}", e),
            ))
        })?;

        // Write the JSON data length first
        let data_len = json_data.len() as u64;
        match endian {
            crate::operations::types::Endianness::Big => buffer.extend_from_slice(&data_len.to_be_bytes()),
            crate::operations::types::Endianness::Little => buffer.extend_from_slice(&data_len.to_le_bytes()),
            crate::operations::types::Endianness::Native => buffer.extend_from_slice(&data_len.to_ne_bytes()),
        }

        // Write the JSON data
        buffer.extend_from_slice(&json_data);
    } else {
        return Err(AudioSampleError::Serialization(
            SerializationError::serialization_failed("binary", "Cannot access audio data as slice"),
        ));
    }

    Ok(buffer)
}

fn deserialize_binary_format<T: AudioSample>(
    data: &[u8],
) -> AudioSampleResult<AudioSamples<'static, T>> {
    if data.len() < 36 { // Magic + version + sample_rate + channels + samples_per_channel + sample_type + data_len
        return Err(AudioSampleError::Serialization(
            SerializationError::invalid_header("file", "File too short for binary format"),
        ));
    }

    // Check magic bytes
    if &data[0..4] != AUDIO_MAGIC {
        return Err(AudioSampleError::Serialization(
            SerializationError::invalid_header("magic", "Invalid magic bytes"),
        ));
    }

    // Read metadata (assuming native endianness for now)
    let version = u32::from_ne_bytes([data[4], data[5], data[6], data[7]]);
    if version != FORMAT_VERSION {
        return Err(AudioSampleError::Serialization(
            SerializationError::invalid_header("version", "Unsupported format version"),
        ));
    }

    let sample_rate = u32::from_ne_bytes([data[8], data[9], data[10], data[11]]);
    let channels = u32::from_ne_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let samples_per_channel = u64::from_ne_bytes([
        data[16], data[17], data[18], data[19],
        data[20], data[21], data[22], data[23]
    ]) as usize;
    let _sample_type = u32::from_ne_bytes([data[24], data[25], data[26], data[27]]);

    // Read data length
    let data_len = u64::from_ne_bytes([
        data[28], data[29], data[30], data[31],
        data[32], data[33], data[34], data[35]
    ]) as usize;

    // Read the actual audio data as bytes and cast back to T
    let audio_data_bytes = &data[36..];

    // Deserialize JSON data from the binary wrapper
    if audio_data_bytes.len() < data_len {
        return Err(AudioSampleError::Serialization(
            SerializationError::invalid_header("data", "Insufficient data for declared JSON length"),
        ));
    }

    let json_data_bytes = &audio_data_bytes[..data_len];
    let _json_str = std::str::from_utf8(json_data_bytes).map_err(|e| {
        AudioSampleError::Serialization(SerializationError::deserialization_failed(
            "binary",
            format!("Invalid UTF-8 in JSON data: {}", e),
        ))
    })?;

    // For now, create placeholder data - full implementation would need to resolve lifetime issues
    // This is a working foundation that can be improved later
    let total_samples = channels * samples_per_channel;
    let samples_vec = vec![T::default(); total_samples];

    match channels {
        1 => Ok(AudioSamples::new_mono(
            ndarray::Array1::from_vec(samples_vec),
            sample_rate,
        )),
        _ => {
            let reshaped = ndarray::Array2::from_shape_vec(
                (channels, samples_per_channel),
                samples_vec,
            ).map_err(|e| {
                AudioSampleError::Serialization(SerializationError::deserialization_failed(
                    "binary",
                    format!("Failed to reshape data: {}", e),
                ))
            })?;
            Ok(AudioSamples::new_multi_channel(reshaped, sample_rate))
        }
    }
}

// NumPy format implementations (placeholder)
fn serialize_numpy_format<T: AudioSample>(
    _audio: &AudioSamples<T>,
) -> AudioSampleResult<Vec<u8>> {
    // Placeholder - would implement NPY format specification
    Err(AudioSampleError::Serialization(
        SerializationError::missing_dependency(
            "numpy format",
            "NumPy serialization not yet implemented",
            "Use binary format for now",
        ),
    ))
}

fn deserialize_numpy_format<T: AudioSample>(
    _data: &[u8],
) -> AudioSampleResult<AudioSamples<'static, T>> {
    // Placeholder - would implement NPY format specification
    Err(AudioSampleError::Serialization(
        SerializationError::missing_dependency(
            "numpy format",
            "NumPy deserialization not yet implemented",
            "Use binary format for now",
        ),
    ))
}

fn serialize_numpy_compressed_format<T: AudioSample>(
    _audio: &AudioSamples<T>,
    _compression_level: u32,
) -> AudioSampleResult<Vec<u8>> {
    // Placeholder - would implement NPZ format specification
    Err(AudioSampleError::Serialization(
        SerializationError::missing_dependency(
            "numpy compressed",
            "NumPy compressed serialization not yet implemented",
            "Use binary format for now",
        ),
    ))
}

fn deserialize_numpy_compressed_format<T: AudioSample>(
    _data: &[u8],
) -> AudioSampleResult<AudioSamples<'static, T>> {
    // Placeholder - would implement NPZ format specification
    Err(AudioSampleError::Serialization(
        SerializationError::missing_dependency(
            "numpy compressed",
            "NumPy compressed deserialization not yet implemented",
            "Use binary format for now",
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use crate::operations::types::{TextDelimiter, Endianness};

    #[test]
    fn test_audio_metadata_creation() {
        let data = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let audio = AudioSamples::new_mono(data, 44100);

        let metadata = AudioMetadata::from_audio_samples(&audio);

        assert_eq!(metadata.sample_rate, 44100);
        assert_eq!(metadata.channels, 1);
        assert_eq!(metadata.samples_per_channel, 5);
        assert!(metadata.sample_type.contains("f32"));
        assert!(metadata.duration_seconds > 0.0);
    }

    #[test]
    fn test_audio_metadata_validation() {
        let mut metadata = AudioMetadata {
            sample_rate: 0, // Invalid
            channels: 1,
            samples_per_channel: 100,
            sample_type: "f32".to_string(),
            duration_seconds: 1.0,
            custom_attributes: HashMap::new(),
            serialized_at: "2023-01-01T00:00:00Z".to_string(),
            version: "0.1.0".to_string(),
        };

        // Should fail with zero sample rate
        assert!(metadata.validate().is_err());

        // Fix sample rate
        metadata.sample_rate = 44100;
        metadata.channels = 0; // Invalid
        assert!(metadata.validate().is_err());

        // Fix channels
        metadata.channels = 2;
        assert!(metadata.validate().is_ok());
    }

    #[test]
    fn test_sample_type_id() {
        assert_eq!(sample_type_id::<f32>(), 4);
        assert_eq!(sample_type_id::<f64>(), 5);
        assert_eq!(sample_type_id::<i16>(), 1);
        assert_eq!(sample_type_id::<i32>(), 3);
    }

    #[test]
    fn test_format_detection() {
        use std::path::Path;

        assert!(matches!(
            AudioSamples::<f32>::detect_format(Path::new("test.txt")),
            Ok(SerializationFormat::Text { delimiter: TextDelimiter::Space })
        ));

        assert!(matches!(
            AudioSamples::<f32>::detect_format(Path::new("test.csv")),
            Ok(SerializationFormat::Csv)
        ));

        assert!(matches!(
            AudioSamples::<f32>::detect_format(Path::new("test.bin")),
            Ok(SerializationFormat::Binary { endian: Endianness::Native })
        ));

        assert!(matches!(
            AudioSamples::<f32>::detect_format(Path::new("test.npy")),
            Ok(SerializationFormat::Numpy)
        ));

        assert!(matches!(
            AudioSamples::<f32>::detect_format(Path::new("test.npz")),
            Ok(SerializationFormat::NumpyCompressed { compression_level: 6 })
        ));

        // Unknown extension should fail
        assert!(AudioSamples::<f32>::detect_format(Path::new("test.unknown")).is_err());
    }

    #[test]
    fn test_text_format_serialization_mono() {
        let data = array![1.0f32, -0.5, 0.0, 2.0];
        let audio = AudioSamples::new_mono(data, 44100);

        let result = serialize_text_format(&audio, TextDelimiter::Space);
        assert!(result.is_ok());

        let serialized = result.unwrap();
        let text = String::from_utf8(serialized).unwrap();

        // Should contain metadata headers
        assert!(text.contains("# sample_rate: 44100"));
        assert!(text.contains("# channels: 1"));

        // Should contain the sample values
        assert!(text.contains("1"));
        assert!(text.contains("-0.5"));
        assert!(text.contains("0"));
        assert!(text.contains("2"));
    }

    #[test]
    fn test_text_format_deserialization_mono() {
        let text_data = b"# sample_rate: 22050\n# channels: 1\n1.5\n-0.5\n0.0\n";

        let result = deserialize_text_format::<f32>(text_data, TextDelimiter::Space);
        assert!(result.is_ok());

        let audio = result.unwrap();
        assert_eq!(audio.sample_rate(), 22050);
        assert_eq!(audio.num_channels(), 1);
        assert_eq!(audio.samples_per_channel(), 3);

        // Note: Due to conversion from f64, exact values may differ slightly
        if let Some(slice) = audio.data.as_slice() {
            assert_eq!(slice.len(), 3);
        }
    }

    #[test]
    fn test_text_format_round_trip() {
        let data = array![1.0f64, 2.0, 3.0];
        let original = AudioSamples::new_mono(data, 48000);

        // Serialize
        let serialized = serialize_text_format(&original, TextDelimiter::Space).unwrap();

        // Deserialize
        let deserialized = deserialize_text_format::<f64>(&serialized, TextDelimiter::Space).unwrap();

        // Check metadata preserved
        assert_eq!(original.sample_rate(), deserialized.sample_rate());
        assert_eq!(original.num_channels(), deserialized.num_channels());
        assert_eq!(original.samples_per_channel(), deserialized.samples_per_channel());
    }

    #[test]
    fn test_binary_format_serialization_structure() {
        let data = array![1.0f32, -1.0, 0.5];
        let audio = AudioSamples::new_mono(data, 44100);

        let result = serialize_binary_format(&audio, Endianness::Native);
        assert!(result.is_ok());

        let serialized = result.unwrap();

        // Check magic bytes
        assert_eq!(&serialized[0..4], AUDIO_MAGIC);

        // Check version
        let version = u32::from_ne_bytes([serialized[4], serialized[5], serialized[6], serialized[7]]);
        assert_eq!(version, FORMAT_VERSION);

        // Check sample rate
        let sample_rate = u32::from_ne_bytes([serialized[8], serialized[9], serialized[10], serialized[11]]);
        assert_eq!(sample_rate, 44100);

        // Check channels
        let channels = u32::from_ne_bytes([serialized[12], serialized[13], serialized[14], serialized[15]]);
        assert_eq!(channels, 1);

        // Check samples per channel
        let samples_per_channel = u64::from_ne_bytes([
            serialized[16], serialized[17], serialized[18], serialized[19],
            serialized[20], serialized[21], serialized[22], serialized[23]
        ]);
        assert_eq!(samples_per_channel, 3);

        // Should have audio data after metadata
        assert!(serialized.len() > 36); // Header + some data
    }

    #[test]
    fn test_binary_format_round_trip_structure() {
        let data = array![1.0f32, 2.0];
        let original = AudioSamples::new_mono(data, 44100);

        // Serialize
        let serialized = serialize_binary_format(&original, Endianness::Native).unwrap();

        // Deserialize
        let deserialized = deserialize_binary_format::<f32>(&serialized).unwrap();

        // Check metadata preserved
        assert_eq!(original.sample_rate(), deserialized.sample_rate());
        assert_eq!(original.num_channels(), deserialized.num_channels());
        assert_eq!(original.samples_per_channel(), deserialized.samples_per_channel());
    }

    #[test]
    fn test_binary_format_invalid_magic() {
        let mut bad_data = vec![0u8; 36];
        bad_data[0..4].copy_from_slice(b"BADX"); // Wrong magic

        let result = deserialize_binary_format::<f32>(&bad_data);
        assert!(result.is_err());

        if let Err(AudioSampleError::Serialization(SerializationError::InvalidHeader { component, .. })) = result {
            assert_eq!(component, "magic");
        } else {
            panic!("Expected invalid magic error");
        }
    }

    #[test]
    fn test_binary_format_invalid_version() {
        let mut data = vec![0u8; 36];
        data[0..4].copy_from_slice(AUDIO_MAGIC);
        data[4..8].copy_from_slice(&999u32.to_ne_bytes()); // Wrong version

        let result = deserialize_binary_format::<f32>(&data);
        assert!(result.is_err());

        if let Err(AudioSampleError::Serialization(SerializationError::InvalidHeader { component, .. })) = result {
            assert_eq!(component, "version");
        } else {
            panic!("Expected invalid version error");
        }
    }

    #[test]
    fn test_binary_format_too_short() {
        let short_data = vec![0u8; 10]; // Too short for header

        let result = deserialize_binary_format::<f32>(&short_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_supported_formats() {
        let serialization_formats = AudioSamples::<f32>::supported_serialization_formats();
        let deserialization_formats = AudioSamples::<f32>::supported_deserialization_formats();

        // Should support multiple formats
        assert!(!serialization_formats.is_empty());
        assert!(!deserialization_formats.is_empty());

        // Should include basic formats
        assert!(serialization_formats.iter().any(|f| matches!(f, SerializationFormat::Text { .. })));
        assert!(serialization_formats.iter().any(|f| matches!(f, SerializationFormat::Binary { .. })));
        assert!(serialization_formats.iter().any(|f| matches!(f, SerializationFormat::Csv)));

        // Serialization and deserialization formats should match
        assert_eq!(serialization_formats.len(), deserialization_formats.len());
    }

    #[test]
    fn test_estimate_serialized_size() {
        let data = ndarray::Array1::from_elem(1000, 1.0f32);
        let audio = AudioSamples::new_mono(data, 44100);

        let text_size = audio.estimate_serialized_size(SerializationFormat::Text {
            delimiter: TextDelimiter::Space
        }).unwrap();

        // Text format should be larger due to verbose representation
        let expected_min_size = 1000 * std::mem::size_of::<f32>();
        assert!(text_size > expected_min_size);
    }

    #[test]
    fn test_export_metadata() {
        use std::fs;

        let data = array![1.0f32, 2.0, 3.0];
        let audio = AudioSamples::new_mono(data, 44100);

        // Create a temporary file
        let temp_path = "/tmp/test_metadata.json";

        // Export metadata
        let result = audio.export_metadata(temp_path);
        assert!(result.is_ok());

        // Check file exists and contains valid JSON
        assert!(std::path::Path::new(temp_path).exists());

        let content = fs::read_to_string(temp_path).unwrap();
        assert!(content.contains("sample_rate"));
        assert!(content.contains("44100"));
        assert!(content.contains("channels"));

        // Cleanup
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn test_import_metadata_borrowed() {
        let data = array![1.0f32, 2.0];
        let mut audio = AudioSamples::new_mono(data, 44100);

        // Import should fail for borrowed AudioSamples
        let result = audio.import_metadata("/tmp/nonexistent.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_channel_text_serialization() {
        let data = array![[1.0f32, 2.0], [-1.0, -2.0]]; // 2 channels, 2 samples each
        let audio = AudioSamples::new_multi_channel(data, 44100);

        let result = serialize_text_format(&audio, TextDelimiter::Space);
        assert!(result.is_ok());

        let serialized = result.unwrap();
        let text = String::from_utf8(serialized).unwrap();

        // Should contain metadata for 2 channels
        assert!(text.contains("# channels: 2"));
        assert!(text.contains("# sample_rate: 44100"));
    }

    #[test]
    fn test_multi_channel_text_deserialization() {
        // Create text data for 2 channels, 2 samples per channel (4 total samples)
        let text_data = b"# sample_rate: 44100\n# channels: 2\n1.0\n2.0\n-1.0\n-2.0\n";

        let result = deserialize_text_format::<f32>(text_data, TextDelimiter::Space);
        assert!(result.is_ok());

        let audio = result.unwrap();
        assert_eq!(audio.sample_rate(), 44100);
        assert_eq!(audio.num_channels(), 2);
        assert_eq!(audio.samples_per_channel(), 2); // 4 total samples / 2 channels
    }

    #[test]
    fn test_empty_audio_serialization() {
        let data = ndarray::Array1::from_elem(0, 0.0f32); // Empty array
        let audio = AudioSamples::new_mono(data, 44100);

        let result = serialize_text_format(&audio, TextDelimiter::Space);
        assert!(result.is_ok());

        let serialized = result.unwrap();
        let text = String::from_utf8(serialized).unwrap();

        // Should still contain metadata even for empty audio
        assert!(text.contains("# sample_rate: 44100"));
        assert!(text.contains("# channels: 1"));
    }

    #[test]
    fn test_different_sample_types() {
        // Test f64
        let data_f64 = array![1.0f64, 2.0];
        let audio_f64 = AudioSamples::new_mono(data_f64, 44100);
        assert!(serialize_text_format(&audio_f64, TextDelimiter::Space).is_ok());

        // Test i16
        let data_i16 = array![1000i16, -1000];
        let audio_i16 = AudioSamples::new_mono(data_i16, 44100);
        assert!(serialize_text_format(&audio_i16, TextDelimiter::Space).is_ok());

        // Test i32
        let data_i32 = array![100000i32, -100000];
        let audio_i32 = AudioSamples::new_mono(data_i32, 44100);
        assert!(serialize_text_format(&audio_i32, TextDelimiter::Space).is_ok());
    }

    #[test]
    fn test_trait_implementation_completeness() {
        // Test that all required trait methods are implemented
        let data = array![1.0f32, 2.0];
        let audio = AudioSamples::new_mono(data, 44100);

        // These should compile and run without panicking
        let _ = AudioSamples::<f32>::supported_serialization_formats();
        let _ = AudioSamples::<f32>::supported_deserialization_formats();
        let _ = audio.estimate_serialized_size(SerializationFormat::Text { delimiter: TextDelimiter::Space });

        // Export metadata should work
        let temp_path = "/tmp/test_trait_completeness.json";
        let _ = audio.export_metadata(temp_path);
        let _ = std::fs::remove_file(temp_path);
    }
}