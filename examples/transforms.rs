#[cfg(feature = "spectral-analysis")]
use std::time::Duration;

#[cfg(feature = "spectral-analysis")]
use audio_samples::{AudioSampleResult, AudioSamples, AudioStatistics, AudioTransforms, sine_wave};

#[cfg(feature = "spectral-analysis")]
use audio_samples::operations::types::{CqtConfig, SpectrogramScale, WindowType};

#[cfg(not(feature = "spectral-analysis"))]
fn main() {
    eprintln!("This example requires the 'spectral-analysis' feature.");
}

#[cfg(feature = "spectral-analysis")]
pub fn main() -> AudioSampleResult<()> {
    let sample_rate_hz = 44_100u32;

    // A 440 Hz tone with a short duration keeps transforms fast.
    let audio: AudioSamples<'static, f64> =
        sine_wave::<f64, f64>(440.0, Duration::from_millis(200), sample_rate_hz, 0.8);

    let fft = audio.fft::<f64>()?;
    println!("FFT: shape={:?}", fft.dim());

    let stft = audio.stft::<f64>(1024, 256, WindowType::Hanning)?;
    println!("STFT: shape={:?}", stft.dim());

    let spec =
        audio.spectrogram::<f64>(1024, 256, WindowType::Hanning, SpectrogramScale::Log, true)?;
    println!("Spectrogram: shape={:?}", spec.dim());

    let mfcc = audio.mfcc::<f64>(13, 40, 20.0, 8_000.0)?;
    println!("MFCC: shape={:?}", mfcc.dim());

    let chroma = audio.chroma::<f64>(12)?;
    println!("Chroma: shape={:?}", chroma.dim());

    let (freqs, psd) = audio.power_spectral_density::<f64>(1024, 0.5)?;
    println!(
        "PSD: bins={}  psd_len={}  f0≈{:.1}Hz",
        freqs.len(),
        psd.len(),
        freqs[0]
    );

    // CQT (single-frame): demonstrate API and output shape.
    let config = CqtConfig::musical();
    let cqt = audio.constant_q_transform::<f64>(&config)?;
    println!("CQT: shape={:?}", cqt.dim());

    // Simple reconstruction from FFT just to validate round-trip.
    let reconstructed = audio.ifft::<f64>(&fft)?;
    println!(
        "iFFT reconstructed: peak={:.4} rms={:.4}",
        reconstructed.peak(),
        reconstructed.rms::<f64>()
    );

    Ok(())
}
