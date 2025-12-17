#[cfg(feature = "spectral-analysis")]
use std::time::Duration;

#[cfg(feature = "spectral-analysis")]
use audio_samples::{AudioSampleResult, AudioSamples, sine_wave};

#[cfg(feature = "spectral-analysis")]
use audio_samples::operations::traits::AudioPitchAnalysis;

#[cfg(feature = "spectral-analysis")]
use audio_samples::operations::types::PitchDetectionMethod;

#[cfg(not(feature = "spectral-analysis"))]
fn main() {
    eprintln!("This example requires the 'spectral-analysis' feature.");
}

#[cfg(feature = "spectral-analysis")]
pub fn main() -> AudioSampleResult<()> {
    let sample_rate_hz = 44_100u32;

    let audio: AudioSamples<'static, f64> =
        sine_wave::<f64, f64>(440.0, Duration::from_secs(1), sample_rate_hz, 0.8);

    let yin = audio.detect_pitch_yin::<f64>(0.15, 80.0, 1_000.0)?;
    let ac = audio.detect_pitch_autocorr::<f64>(80.0, 1_000.0)?;
    println!("Pitch (YIN): {:?} Hz", yin);
    println!("Pitch (autocorr): {:?} Hz", ac);

    let contour =
        audio.track_pitch::<f64>(2048, 512, PitchDetectionMethod::Yin, 0.15, 80.0, 1_000.0)?;
    let voiced = contour.iter().filter(|(_, f)| f.is_some()).count();
    println!(
        "Pitch contour: {} frames ({} voiced)",
        contour.len(),
        voiced
    );

    let hnr = audio.harmonic_to_noise_ratio::<f64>(440.0, 8)?;
    println!("HNR (8 harmonics): {:.2} dB", hnr);

    let harmonics = audio.harmonic_analysis::<f64>(440.0, 6, 0.02)?;
    println!("Harmonic magnitudes (norm): {:?}", harmonics);

    let (key, confidence) = audio.estimate_key::<f64>(4096, 1024)?;
    println!("Estimated key index: {}  confidence={:.3}", key, confidence);

    Ok(())
}
