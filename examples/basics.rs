use std::time::Duration;

use audio_samples::{
    AudioSampleResult, AudioSamples, AudioStatistics, ConvertTo, ToneComponent, sample_rate,
};
use ndarray::array;

pub fn main() -> AudioSampleResult<()> {
    // At its core, audio_samples aims to provide a way of working with type-safe primitive audio samples.
    // ['AudioSample']s operate under slightly different conditions than regular primitive types, for example,
    // floats are expected to be normalised to [-1.0, +1.0].

    let i16_val: i16 = i16::MAX;
    let f32_val: f32 = i16_val.convert_to();

    println!("=> Converting i16 to f32...");
    println!("{}_i16 -> {:.2}_f32", i16_val, f32_val);
    println!("=> And back again to i16...");
    let i16_val: i16 = f32_val.convert_to();
    println!("{:.2}_f32 -> {}_i16", f32_val, i16_val);
    println!("\n-----\n");

    // The ['AudioSample'] trait underpins the entire audio system.

    // The more useful side of things though is the AudioSamples<'_, T: AudioSample> struct.
    // This is the core public data structure in the library. It represents a collection of
    // audio samples stored in an ndarray backed structure. It maintains properties such as sample rate automatically.
    // The samples can be mono or multi-channel.
    // The sample_rate macro is used to guarantee that sample rates are valid at compile time.

    let mono_audio = AudioSamples::<f32>::new_mono(array![0.1, 0.2, 0.3], sample_rate!(16000));
    let stereo_audio = AudioSamples::<f32>::new_multi_channel(
        array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        sample_rate!(44100),
    );

    println!("Mono audio: {:#}\n", mono_audio);
    println!("-----\n");
    println!("Stereo audio: {:#}\n", stereo_audio);

    println!("-----\n");

    // AudioSamples can be converted between different sample types easily due to the underlying AudioSampel and ConvertTo traits.
    let stereo_i16 = stereo_audio.convert_to::<i16>();
    println!("Converted stereo audio to i16: {:#}\n", stereo_i16);

    // Let's get some more useful audio data
    let sine_wave = audio_samples::sine_wave::<f32, f32>(
        440.0,                  // Frequency in Hz
        Duration::from_secs(2), // Duration
        44100,                  // Sample rate
        1.0,                    // Amplitude
    );

    println!("Generated sine wave audio: {:#}\n", sine_wave);
    println!("-----\n");

    // Sine waves are useful, but let's get something more practical.
    // Create 440 Hz with harmonics
    let components = [
        ToneComponent::new(440.0, 1.0),   // fundamental
        ToneComponent::new(880.0, 0.5),   // 2nd harmonic
        ToneComponent::new(1320.0, 0.25), // 3rd harmonic
    ];
    let audio =
        audio_samples::compound_tone::<f32, f64>(&components, Duration::from_secs(2), 44100);

    let (num_channels, samples_per_channel, duration_seconds, sample_rate, layout) = audio.info();
    println!("Compound Tone wave info:");
    println!("Number of channels: {}", num_channels);
    println!("Samples per channel: {}", samples_per_channel);
    println!("Duration (seconds): {:.2}", duration_seconds);
    println!("Sample rate: {} Hz", sample_rate);
    println!("Layout: {:?}", layout);

    println!("-----\n");

    println!("== Statistics ==");
    println!("Mean: {}", audio.mean::<f64>());
    println!("RMS: {}", audio.rms::<f64>());
    println!("Min: {}", audio.min_sample());
    println!("Max: {}", audio.max_sample());
    println!("Variance: {}", audio.variance::<f64>());
    println!("Standard Deviation: {}", audio.std_dev::<f64>());
    println!("Peak: {}", audio.peak());
    println!("Zero Crossings: {}", audio.zero_crossings());
    println!("Zero Crossing Rate: {}", audio.zero_crossing_rate::<f64>());
    #[cfg(feature = "fft")]
    {
        if let Some(ac) = audio.autocorrelation::<f64>(1) {
            println!("Autocorrelation (lag 1): {:?}", ac);
        }
    }

    // Create a new 880 Hz tone by shifting the frequency of the original audio
    let components = [
        ToneComponent::new(880.0, 1.0),   // fundamental
        ToneComponent::new(1760.0, 0.5),  // 2nd harmonic
        ToneComponent::new(2640.0, 0.25), // 3rd harmonic
    ];
    let other_audio =
        audio_samples::compound_tone::<f32, f64>(&components, Duration::from_secs(2), 44100);

    println!(
        "Cross-correlation with 880 Hz tone: {:?}",
        audio.cross_correlation::<f64>(&other_audio, 1)?
    );
    #[cfg(feature = "fft")]
    {
        println!("Spectral-centroid: {}", audio.spectral_centroid::<f64>()?);
        println!(
            "Spectral-rolloff (0.85): {}",
            audio.spectral_rolloff::<f64>(0.85)?
        );
    }
    Ok(())
}
