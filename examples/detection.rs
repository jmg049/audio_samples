use std::time::Duration;

use audio_samples::{AudioSampleResult, AudioSamples, sine_wave};

pub fn main() -> AudioSampleResult<()> {
    let audio: AudioSamples<'static, f64> =
        sine_wave::<f64, f64>(440.0, Duration::from_millis(250), 44_100, 0.8);

    let f0 = audio_samples::detection::detect_fundamental_frequency(&audio)?;
    println!("Fundamental frequency: {:?} Hz", f0);

    let silence = audio_samples::detection::detect_silence_regions::<f64, f64>(&audio, 1e-4)?;
    println!("Silence regions: {:?}", silence);

    #[cfg(feature = "fft")]
    {
        let sr = audio_samples::detection::detect_sample_rate(&audio)?;
        println!("Detected sample rate: {:?}", sr);
    }

    Ok(())
}
