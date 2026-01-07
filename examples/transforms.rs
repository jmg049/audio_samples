#[cfg(not(feature = "transforms"))]
fn main() {
    eprintln!("error: This example requires the `transforms` feature.");
    std::process::exit(1);
}

#[cfg(feature = "transforms")]
pub fn main() -> audio_samples::AudioSampleResult<()> {
    use std::time::Duration;

    use audio_samples::{
        AudioSamples, AudioStatistics, AudioTransforms, nzu, sample_rate, sine_wave,
    };
    use spectrograms::{ChromaParams, CqtParams, MfccParams, StftParams, WindowType};

    let sample_rate_hz = sample_rate!(44100);

    // A 440 Hz tone with a short duration keeps transforms fast.
    let audio: AudioSamples<'static, f64> =
        sine_wave::<f64>(440.0, Duration::from_millis(200), sample_rate_hz, 0.8);

    println!("=== Basic Transforms ===");

    let fft = audio.fft(nzu!(8192))?;
    println!("FFT: shape={:?}", fft.dim());

    let stft_params = StftParams::new(nzu!(1024), nzu!(256), WindowType::Hanning, true)?;
    let stft = audio.stft(&stft_params)?;
    println!("STFT: shape={:?}", stft.data.dim());

    // MFCC
    let mfcc_params = MfccParams::speech_standard();
    let mfcc = audio.mfcc(&stft_params, nzu!(40), &mfcc_params)?;
    println!("MFCC: shape={:?}", mfcc.data.dim());

    // Chromagram
    let chroma_params = ChromaParams::music_standard();
    let chroma = audio.chromagram(&stft_params, &chroma_params)?;
    println!("Chroma: shape={:?}", chroma.data.dim());

    // PSD via Welch's method
    let (freqs, psd) = audio.power_spectral_density(nzu!(1024), 0.5)?;
    println!(
        "PSD: bins={} psd_len={} f0≈{:.1}Hz",
        freqs.len(),
        psd.len(),
        freqs[0]
    );

    // CQT
    let cqt_params = CqtParams::new(nzu!(12), nzu!(7), 32.7)?;
    let cqt = audio.constant_q_transform(&cqt_params, nzu!(256))?;
    println!("CQT: shape={:?}", cqt.data.dim());

    // ISTFT reconstruction
    println!("\n=== Reconstruction ===");
    let reconstructed = AudioSamples::<f64>::istft(stft)?;
    println!(
        "iSTFT reconstructed: peak={:.4} rms={:.4}",
        reconstructed.peak(),
        reconstructed.rms()
    );

    Ok(())
}
