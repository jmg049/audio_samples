use std::time::Duration;

use audio_samples::{AudioSampleResult, AudioStatistics, ToneComponent};

pub fn main() -> AudioSampleResult<()> {
    let components = [
        ToneComponent::new(440.0, 1.0),   // fundamental
        ToneComponent::new(880.0, 0.5),   // 2nd harmonic
        ToneComponent::new(1320.0, 0.25), // 3rd harmonic
    ];
    let audio =
        audio_samples::compound_tone::<f32, f64>(&components, Duration::from_secs(2), 44100);

    let (num_channels, samples_per_channel, duration_seconds, sample_rate, layout) = audio.info();
    println!("Compound Tone wave info:");
    println!("|-Number of channels: {}", num_channels);
    println!("|-Samples per channel: {}", samples_per_channel);
    println!("|-Duration (seconds): {:.2}", duration_seconds);
    println!("|-Sample rate: {} Hz", sample_rate);
    println!("|-Layout: {:?}", layout);

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
        println!("Spectral-centroid: {}", audio.spectral_centroid::<f64>()?);
        println!(
            "Spectral-rolloff (0.85): {}",
            audio.spectral_rolloff::<f64>(0.85)?
        );
    }

    // Create a new 880 Hz tone by shifting the frequency of the original audio
    let components = [
        ToneComponent::new(880.0, 1.0),   // fundamental
        ToneComponent::new(1760.0, 0.5),  // 2nd harmonic
        ToneComponent::new(2640.0, 0.25), // 3rd harmonic
    ];
    let other_audio =
        audio_samples::compound_tone::<f32, f64>(&components, Duration::from_secs(2), 44100);
    let (num_channels, samples_per_channel, duration_seconds, sample_rate, layout) =
        other_audio.info();
    println!("\nCompound Tone wave info:");
    println!("|-Number of channels: {}", num_channels);
    println!("|-Samples per channel: {}", samples_per_channel);
    println!("|-Duration (seconds): {:.2}", duration_seconds);
    println!("|-Sample rate: {} Hz", sample_rate);
    println!("|-Layout: {:?}", layout);

    println!(
        "\nCross-correlation with 880 Hz tone: {:?}",
        audio.cross_correlation::<f64>(&other_audio, 1)?
    );

    Ok(())
}
