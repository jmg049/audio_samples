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

    // FFT size must be >= the signal length (8820 samples for 200 ms @ 44.1 kHz).
    let n_fft = 16_384usize;
    let fft = audio.fft(nzu!(16_384))?;
    println!("FFT: shape={:?}", fft.dim());

    // --- Self-verification: the dominant FFT bin must be near 440 Hz --------
    // Magnitude over the first half (real-signal spectrum is symmetric).
    let mags: Vec<f64> = fft.iter().map(|c| c.norm()).collect();
    let half = mags.len() / 2;
    let (peak_bin, _) = mags[1..half]
        .iter()
        .enumerate()
        .map(|(i, &m)| (i + 1, m))
        .fold(
            (1usize, f64::MIN),
            |acc, (i, m)| if m > acc.1 { (i, m) } else { acc },
        );
    let bin_hz = peak_bin as f64 * sample_rate_hz.get() as f64 / n_fft as f64;
    println!("Dominant FFT bin: {peak_bin} (~{bin_hz:.1} Hz)");
    assert!(
        (bin_hz - 440.0).abs() < 15.0,
        "expected dominant FFT bin near 440 Hz, got {bin_hz:.1} Hz"
    );

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
    let psd = audio.power_spectral_density(nzu!(1024), 0.5)?;
    println!(
        "PSD: bins={} psd_len={} f0≈{:.1}Hz",
        psd.frequencies().len(),
        psd.density().len(),
        psd.frequencies()[0]
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

    // --- Self-verification: round-trip preserves signal energy -------------
    // The reconstructed signal should retain a comparable RMS to the input
    // (COLA windowing is energy-preserving up to edge effects).
    let in_rms = audio.rms();
    let out_rms = reconstructed.rms();
    assert!(out_rms > 0.0, "iSTFT must reconstruct a non-silent signal");
    assert!(
        (out_rms - in_rms).abs() / in_rms < 0.25,
        "iSTFT RMS {out_rms:.4} should be within 25% of input RMS {in_rms:.4}"
    );

    Ok(())
}
