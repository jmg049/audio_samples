#[cfg(feature = "serialization")]
use std::time::Duration;

#[cfg(feature = "serialization")]
use audio_samples::{AudioSampleResult, AudioSamples, AudioStatistics, sine_wave};

#[cfg(feature = "serialization")]
use audio_samples::operations::traits::AudioSamplesSerialise;

#[cfg(feature = "serialization")]
use audio_samples::operations::types::SerializationFormat;

#[cfg(not(feature = "serialization"))]
fn main() {
    eprintln!("This example requires the 'serialization' feature.");
}

#[cfg(feature = "serialization")]
pub fn main() -> AudioSampleResult<()> {
    let audio: AudioSamples<'static, f32> =
        sine_wave::<f32, f32>(440.0, Duration::from_millis(200), 44_100, 0.8);

    let formats = AudioSamples::<f32>::supported_serialization_formats();
    println!("Supported formats: {:?}", formats);

    let format = SerializationFormat::Binary {
        endian: audio_samples::operations::Endianness::Little,
    };
    let estimated = audio.estimate_serialized_size(format)?;
    let bytes = audio.serialize_to_bytes(format)?;
    println!(
        "Binary bytes: actual={} estimated≈{}",
        bytes.len(),
        estimated
    );

    let decoded = AudioSamples::<f32>::deserialize_from_bytes(&bytes, format)?;
    println!(
        "Round-trip: peak_in={:.4} peak_out={:.4}",
        audio.peak(),
        decoded.peak()
    );

    // Validate round-trip within a reasonable tolerance.
    audio.validate_round_trip(format, 1e-5)?;

    Ok(())
}
