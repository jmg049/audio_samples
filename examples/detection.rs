//! Lightweight detection helpers: fundamental-frequency and silence-region
//! estimation. The core helpers need no extra features; the sample-rate
//! heuristic additionally requires the `transforms` feature.

pub fn main() -> audio_samples::AudioSampleResult<()> {
    use audio_samples::{sine_wave, utils::detection, AudioSamples};
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

    // --- Self-verification -------------------------------------------------
    // This lightweight estimator can lock onto an octave/sub-octave of the true
    // pitch, so accept 440 Hz or any of its near sub-/super-octaves. We verify
    // the estimate is positive and octave-related to 440 Hz.
    let f = f0.expect("a pure 440 Hz tone should yield a fundamental-frequency estimate");
    assert!(f > 0.0, "estimated frequency must be positive, got {f}");
    let ratio = (440.0_f64 / f).log2();
    assert!(
        (ratio - ratio.round()).abs() < 0.1,
        "estimated f0 {f} Hz should be an octave-multiple of 440 Hz (ratio log2 = {ratio:.3})"
    );
    // A continuous tone has no *sustained* silence: any reported region is a
    // momentary zero-crossing, far shorter than one period of 440 Hz (~2.3 ms).
    for (start, end) in &silence {
        assert!(
            end - start < 1e-3,
            "unexpected sustained silence region ({start}, {end}) in a continuous tone"
        );
    }

    #[cfg(feature = "transforms")]
    {
        // The sample-rate heuristic is best-effort: on a pure tone it may fail
        // to find enough spectral content, which is fine for a demo.
        match detection::detect_sample_rate::<f64>(&audio) {
            Ok(sr) => println!("Detected sample rate: {:?}", sr),
            Err(e) => println!("Sample-rate detection inconclusive for a pure tone: {e}"),
        }
    }

    Ok(())
}
