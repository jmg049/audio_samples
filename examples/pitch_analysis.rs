//! Pitch analysis example: YIN vs autocorrelation, pitch tracking, spectral
//! HNR, harmonic magnitudes and key estimation.
//!
//! Learning goals:
//! - a frequency search range `[f_min, f_max]` is a lag range
//!   `τ = fs/f_max … fs/f_min` samples, and the analysed frame must be longer
//!   than two periods of `f_min`;
//! - autocorrelation picks an integer lag, so its answer lies on the grid
//!   `fs/k`; YIN refines the lag by parabolic interpolation.
//!
//! Run:
//! `cargo run --example pitch_analysis --features pitch-analysis`

#[cfg(not(feature = "pitch-analysis"))]
fn main() {
    eprintln!("This example requires the 'pitch-analysis' feature.");
}

#[cfg(feature = "pitch-analysis")]
pub fn main() -> audio_samples::AudioSampleResult<()> {
    use audio_samples::operations::traits::AudioPitchAnalysis;
    use audio_samples::operations::types::PitchDetectionMethod;
    use audio_samples::{AudioSamples, sample_rate, sine_wave};
    use spectrograms::StftParams;
    use std::time::Duration;

    let sr = sample_rate!(44100);
    let fs = f64::from(sr.get());

    let audio: AudioSamples<'static, f64> =
        sine_wave::<f64>(440.0, Duration::from_secs(1), sr, 0.8);

    // Search 80-1000 Hz → lags ⌊44100/1000⌋ = 44 … ⌊44100/80⌋ = 551 samples.
    let (f_min, f_max) = (80.0, 1_000.0);
    println!(
        "Search {f_min}-{f_max} Hz = lags {}..={} samples; frame must exceed {} samples",
        (fs / f_max) as usize,
        (fs / f_min) as usize,
        2 * (fs / f_min) as usize + 1
    );

    let yin = audio.detect_pitch_yin(0.15, f_min, f_max)?;
    let ac = audio.detect_pitch_autocorr(f_min, f_max)?;
    println!(
        "True f0           : 440.000 Hz (period {:.3} samples)",
        fs / 440.0
    );
    println!("Pitch (YIN)       : {:.3} Hz", yin.unwrap_or(f64::NAN));
    println!(
        "Pitch (autocorr)  : {:.3} Hz  (integer lag {} → fs/lag; neighbours {:.1} / {:.1} Hz)",
        ac.unwrap_or(f64::NAN),
        (fs / ac.unwrap_or(1.0)).round(),
        fs / 101.0,
        fs / 99.0
    );

    // --- Self-verification ---------------------------------------------------
    // YIN interpolates the period, so it is within 0.5 % of 440 Hz; the
    // autocorrelation estimate is the nearest integer-lag frequency fs/100 = 441 Hz.
    let f = yin.expect("a steady tone is voiced");
    assert!(
        (f - 440.0).abs() / 440.0 < 0.005,
        "YIN pitch {f} Hz should be ~440 Hz"
    );
    let a = ac.expect("a steady tone has a positive autocorrelation peak");
    assert!(
        (a - 441.0).abs() < 1e-9,
        "autocorrelation pitch {a} Hz should be fs/100"
    );

    // Pitch track: 2048-sample frames (> 2·551 samples), hop 512.
    let contour = audio.track_pitch(
        audio_samples::nzu!(2048),
        audio_samples::nzu!(512),
        PitchDetectionMethod::Yin,
        0.15,
        f_min,
        f_max,
    )?;
    let voiced = contour.voiced_frames().count();
    println!(
        "Pitch contour     : {} frames ({} voiced), mean {:.3} Hz",
        contour.len(),
        voiced,
        contour.mean_pitch().unwrap_or(f64::NAN)
    );
    assert!(voiced > 0, "a steady tone should have voiced frames");

    // Spectral HNR: harmonic bins are only those within ±1 bin of k·f0. Here
    // n_fft = 44100 (1 Hz bins) and 440 Hz falls exactly on a bin, so the Hann
    // main lobe puts power 1/4 in the centre bin and 1/16 in each neighbour;
    // only the centre bin counts as harmonic → 10·log10(0.25/0.125) ≈ 3.01 dB.
    // A pure tone therefore does NOT give a large HNR with this estimator.
    let hnr = audio.harmonic_to_noise_ratio(440.0, audio_samples::nzu!(8), None, None)?;
    println!("Spectral HNR      : {hnr:.2} dB (window-leakage limited, see docs)");
    assert!(hnr.is_finite(), "HNR must be finite, got {hnr}");

    let harmonics = audio.harmonic_analysis(440.0, audio_samples::nzu!(6), 0.02, None, None)?;
    println!("Harmonic magnitudes (relative to the fundamental): {harmonics:.3?}");

    // Key estimation needs tonal context, not a single tone: use a D major
    // chord (D4, F#4, A4).
    let chord = sine_wave::<f64>(293.66, Duration::from_secs(2), sr, 0.3)
        + sine_wave::<f64>(369.99, Duration::from_secs(2), sr, 0.3)
        + sine_wave::<f64>(440.0, Duration::from_secs(2), sr, 0.3);
    let stft_params = StftParams::builder()
        .n_fft(audio_samples::nzu!(4096))
        .hop_size(audio_samples::nzu!(1024))
        .build()
        .unwrap();
    let key = chord.estimate_key(&stft_params)?;
    println!(
        "Estimated key of a D major triad: {} {:?} (confidence {:.3})",
        key.tonic, key.mode, key.confidence
    );
    println!(
        "  (a bare triad is ambiguous between related keys; the Krumhansl-Schmuckler\n   \
         correlation needs a passage with a full scale for a reliable answer)"
    );

    Ok(())
}
