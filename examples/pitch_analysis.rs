#[cfg(not(feature = "pitch-analysis"))]
fn main() {
    eprintln!("This example requires the 'pitch-analysis' feature.");
}

#[cfg(feature = "pitch-analysis")]
pub fn main() -> audio_samples::AudioSampleResult<()> {
    use audio_samples::operations::traits::AudioPitchAnalysis;
    use audio_samples::operations::types::PitchDetectionMethod;
    use audio_samples::{AudioSamples, sine_wave};
    use spectrograms::StftParams;
    use std::time::Duration;

    let sample_rate_hz = core::num::NonZeroU32::new(44_100).unwrap();

    let audio: AudioSamples<'static, f64> =
        sine_wave::<f64>(440.0, Duration::from_secs(1), sample_rate_hz, 0.8);

    let yin = audio.detect_pitch_yin(0.15, 80.0, 1_000.0)?;
    let ac = audio.detect_pitch_autocorr(80.0, 1_000.0)?;
    println!("Pitch (YIN): {:?} Hz", yin);
    println!("Pitch (autocorr): {:?} Hz", ac);

    // --- Self-verification: both detectors should land near 440 Hz ---------
    // Both detectors quantise to a coarse grid here, so allow a modest tolerance.
    if let Some(f) = yin {
        assert!((f - 440.0).abs() < 20.0, "YIN pitch {f} Hz should be ~440 Hz");
    }
    if let Some(f) = ac {
        assert!(
            (f - 440.0).abs() < 20.0,
            "autocorr pitch {f} Hz should be ~440 Hz"
        );
    }

    let contour = audio.track_pitch(
        audio_samples::nzu!(2048),
        audio_samples::nzu!(512),
        PitchDetectionMethod::Yin,
        0.15,
        80.0,
        1_000.0,
    )?;
    let voiced = contour.voiced_frames().count();
    println!(
        "Pitch contour: {} frames ({} voiced, mean {:?} Hz)",
        contour.len(),
        voiced,
        contour.mean_pitch()
    );

    let hnr = audio.harmonic_to_noise_ratio(440.0, audio_samples::nzu!(8), None, None)?;
    println!("HNR (8 harmonics): {:.2} dB", hnr);

    let harmonics = audio.harmonic_analysis(440.0, audio_samples::nzu!(6), 0.02, None, None)?;
    println!("Harmonic magnitudes (norm): {:?}", harmonics);

    let stft_params = StftParams::builder()
        .n_fft(audio_samples::nzu!(4096))
        .hop_size(audio_samples::nzu!(1024))
        .build()
        .unwrap();
    let key = audio.estimate_key(&stft_params)?;
    println!(
        "Estimated key: {} {:?}  confidence={:.3}",
        key.tonic, key.mode, key.confidence
    );

    // A pure 440 Hz tone is strongly voiced, so the contour must report
    // at least one voiced frame, and HNR should be a finite number.
    assert!(voiced > 0, "a steady tone should have voiced frames");
    assert!(hnr.is_finite(), "HNR must be finite, got {hnr}");

    Ok(())
}
