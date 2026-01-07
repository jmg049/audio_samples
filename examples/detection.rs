#[cfg(not(feature = "onset-detection"))]
pub fn main() {
    eprintln!("error: This example requires the `onset-detection` feature.");
    std::process::exit(1);
}

#[cfg(feature = "onset-detection")]
pub fn main() -> audio_samples::AudioSampleResult<()> {
    use audio_samples::{AudioSamples, sine_wave, utils::detection};
    use std::time::Duration;

    let audio: AudioSamples<'static, f64> = sine_wave::<f64>(
        440.0,
        Duration::from_millis(250),
        core::num::NonZeroU32::new(44_100).unwrap(),
        0.8,
    );

    let f0: Option<f64> = detection::detect_fundamental_frequency(&audio)?;
    println!("Fundamental frequency: {:?} Hz", f0);

    let silence = detection::detect_silence_regions::<f64>(&audio, 1e-4)?;
    println!("Silence regions: {:?}", silence);

    #[cfg(feature = "transforms")]
    {
        let sr = detection::detect_sample_rate::<f64>(&audio)?;
        println!("Detected sample rate: {:?}", sr);
    }

    Ok(())
}
