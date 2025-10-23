//! examples/oscillator_box.rs
//! ------------------------------------------------------------
//! This example demonstrates a realistic (but simple) signal chain for testing
//! embedded audio hardware or downstream processing systems.
//!
//! 1. Generate a clean sine wave (AudioSamples<f32>).
//! 2. Convert mono → stereo and attenuate the right channel.
//! 3. Inspect channel properties.
//! 4. Convert float samples to i16 for playback via a DAC (Digital-to-Analog Converter).
//!
//! Concepts shown: AudioSamples, AudioStatistics, channel ops, format conversion traits (ConvertTo).

use std::error::Error;

use audio_samples::operations::types::StereoConversionMethod;
use audio_samples::{AudioChannelOps, AudioSamples, AudioStatistics, RIGHT};

pub fn main() -> Result<(), Box<dyn Error>> {
    println!("-----------------------");

    // A signal engineer need to generate a clean sine wave to test frequency response of a downstream system.
    // They validate its amplitude and RMS values before sending it through the system.
    // Code Concepts: AudioSample, AudioSamples, utils::generation, statistics
    // Output Goals:
    // AudioSamples<f32> representing 1kHz tone, 1 second with an amplitude of 1.0 at 16kHz sample rate.
    // Print : mean, RMS, max amplitude.
    // Verify: Is the signal clean?
    let sine_wave: AudioSamples<f32> = audio_samples::sine_wave(1000.0, 1.0, 16000, 1.0)?;

    println!("{}", sine_wave);

    let mean = sine_wave.mean();
    let rms = sine_wave.rms();
    let max_amplitude = sine_wave.amplitude();

    println!("Sine Wave Statistics:");
    println!("\tMean: {:.4}", mean);
    println!("\tRMS: {:.4}", rms);
    println!("\tMax Amplitude: {:.2}", max_amplitude);
    println!(
        "\tMin: {:.2}, Max: {:.2}",
        sine_wave.min_sample(),
        sine_wave.max_sample()
    );

    println!("-----------------------");

    // The sine wave is "recorded" by two virtual microphones placed in a stereo field—left captures full signal, right captures attenuated version.
    // The engineer inspects and isolates each channel.

    let mut sine_wave = sine_wave.to_stereo(StereoConversionMethod::Duplicate)?;
    sine_wave.apply_to_channel(RIGHT, |sample| sample * 0.5)?;

    // Extract channels to get their individual amplitudes
    let left_channel =
        sine_wave.to_mono(audio_samples::operations::types::MonoConversionMethod::Left)?;
    let right_channel =
        sine_wave.to_mono(audio_samples::operations::types::MonoConversionMethod::Right)?;

    println!(
        "Left Channel Max Amplitude: {:.2}",
        left_channel.amplitude()
    );
    println!(
        "Right Channel Max Amplitude: {:.2}",
        right_channel.amplitude()
    );

    println!("{}", sine_wave);
    println!("-----------------------");

    // Before uploading the test tone to the firmware for playback via DAC,
    // the engineer converts the audio from f32 to i16, applying scaling
    // and clipping rules.

    let int16_audio: AudioSamples<i16> = sine_wave.convert_to();
    println!("Converted audio to i16");
    println!("{}", int16_audio);

    println!(
        "Min: {}, Max: {}",
        int16_audio.min_sample(),
        int16_audio.max_sample()
    );

    println!("-----------------------");

    Ok(())
}
