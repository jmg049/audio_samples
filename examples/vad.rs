//! Voice Activity Detection (VAD) example.
//!
//! Learning goal: see how frame energy (RMS level in dBFS) and the
//! zero-crossing rate (ZCR, crossings per sample) separate a speech-like
//! segment from background noise, and why a pure low tone is rejected by the
//! combined energy + ZCR rule.
//!
//! Run:
//! `cargo run --example vad --features vad,editing`

#[cfg(not(feature = "vad"))]
fn main() {
    eprintln!("error: This example requires the `vad` feature.");
    std::process::exit(1);
}

#[cfg(feature = "vad")]
fn main() -> audio_samples::AudioSampleResult<()> {
    use audio_samples::{
        AudioSamples, AudioVoiceActivityDetection, nzu,
        operations::types::{VadChannelPolicy, VadConfig, VadMethod},
        sample_rate,
    };
    use non_empty_slice::NonEmptyVec;
    use std::f64::consts::PI;

    // ---------------------------------------------------------------------
    // Test signal at 16 kHz (1 s):
    //   0.00-0.25 s  background noise only (≈ −65 dBFS RMS)
    //   0.25-0.75 s  a speech-like "vowel": f0 = 125 Hz with harmonics
    //                weighted towards a 700 Hz formant, plus the noise
    //   0.75-1.00 s  background noise only
    // ---------------------------------------------------------------------
    let sr = sample_rate!(16000);
    let fs = f64::from(sr.get());
    let n = sr.get() as usize;
    let (on, off) = (n / 4, 3 * n / 4);

    let mut seed = 12345u64;
    let mut noise = || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((seed >> 33) as f64 / (1u64 << 30) as f64) - 1.0 // uniform in [-1, 1)
    };
    let vowel = |t: f64| -> f64 {
        (1..=24)
            .map(|h| {
                let f = 125.0 * h as f64;
                let a = 0.3 / (1.0 + ((f - 700.0) / 200.0).powi(2));
                a * (2.0 * PI * f * t).sin()
            })
            .sum()
    };
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            let speech = if (on..off).contains(&i) {
                vowel(t)
            } else {
                0.0
            };
            (speech + 1e-3 * noise()) as f32
        })
        .collect();
    let audio: AudioSamples<'static, f32> =
        AudioSamples::from_mono_vec(NonEmptyVec::new(data).unwrap(), sr);

    // The two per-frame features, computed by hand for one 25 ms frame of each
    // segment (the VAD computes exactly these internally):
    //   level = 20·log10( sqrt( (1/L) Σ x² ) )  dBFS
    //   ZCR   = sign changes / (L − 1)          crossings per sample (≈ 2f/fs)
    let frame_features = |start: usize, len: usize| -> (f64, f64) {
        let x = audio.as_slice().unwrap();
        let frame = &x[start..start + len];
        let rms = (frame.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / len as f64).sqrt();
        let crossings = frame
            .windows(2)
            .filter(|w| (w[0] > 0.0) != (w[1] > 0.0))
            .count();
        (20.0 * rms.log10(), crossings as f64 / (len - 1) as f64)
    };
    let frame_len = 400; // 25 ms
    let (lvl_noise, zcr_noise) = frame_features(1000, frame_len);
    let (lvl_vowel, zcr_vowel) = frame_features(on + 1000, frame_len);
    println!("Per-frame features (25 ms frames):");
    println!("  noise : level {lvl_noise:6.1} dBFS, ZCR {zcr_noise:.3} crossings/sample");
    println!("  vowel : level {lvl_vowel:6.1} dBFS, ZCR {zcr_vowel:.3} crossings/sample");

    // Combined rule: speech iff level ≥ −40 dBFS AND 0.02 ≤ ZCR ≤ 0.3.
    // At 16 kHz the ZCR window corresponds to dominant frequencies of
    // 0.02·fs/2 = 160 Hz to 0.3·fs/2 = 2400 Hz.
    let cfg = VadConfig::default()
        .with_method(VadMethod::Combined)
        .with_hop_size(nzu!(160))? // 10 ms hop (set first: hop must stay ≤ frame)
        .with_frame_size(nzu!(400))? // 25 ms frames
        .with_channel_policy(VadChannelPolicy::AverageToMono)
        .with_energy_threshold_db(-40.0);

    let mask = audio.voice_activity_mask(&cfg)?;
    let regions = audio.speech_regions(&cfg)?;
    let speech_frames = mask.iter().filter(|&&b| b).count();
    println!("\nCombined VAD (−40 dBFS, ZCR 0.02-0.30):");
    println!("  frames: {}, speech frames: {speech_frames}", mask.len());
    for &(s, e) in &regions {
        println!(
            "  speech region: samples {s}..{e}  ({:.3} s - {:.3} s; true 0.250 s - 0.750 s)",
            s as f64 / fs,
            e as f64 / fs
        );
    }
    println!(
        "  (regions extend past the true end by the hangover of {} frames plus up to one frame)",
        cfg.hangover_frames.get()
    );

    // --- Self-verification: exactly one region, close to the true segment.
    assert_eq!(
        regions.len(),
        1,
        "expected one speech region, got {regions:?}"
    );
    let (s, e) = regions[0];
    assert!(
        s.abs_diff(on) <= 800,
        "region start {s} should be near {on}"
    );
    assert!(
        e.abs_diff(off) <= 1600,
        "region end {e} should be near {off}"
    );

    // ---------------------------------------------------------------------
    // A loud pure 220 Hz tone at 44.1 kHz has ZCR ≈ 2·220/44100 ≈ 0.010,
    // below zcr_min = 0.02: the energy rule accepts it, the combined rule
    // rejects it.
    // ---------------------------------------------------------------------
    let sr44 = sample_rate!(44100);
    let tone: Vec<f32> = (0..44_100)
        .map(|i| (0.5 * (2.0 * PI * 220.0 * i as f64 / 44_100.0).sin()) as f32)
        .collect();
    let tone: AudioSamples<'static, f32> =
        AudioSamples::from_mono_vec(NonEmptyVec::new(tone).unwrap(), sr44);
    let cfg44 = VadConfig::default()
        .with_method(VadMethod::Combined)
        .with_energy_threshold_db(-40.0);
    let combined = tone
        .voice_activity_mask(&cfg44)?
        .iter()
        .filter(|&&b| b)
        .count();
    let energy = tone
        .voice_activity_mask(&cfg44.with_method(VadMethod::Energy))?
        .iter()
        .filter(|&&b| b)
        .count();
    println!(
        "\n220 Hz sine at 44.1 kHz (ZCR ≈ {:.4}):",
        2.0 * 220.0 / 44_100.0
    );
    println!("  Energy   method: {energy} active frames");
    println!("  Combined method: {combined} active frames (ZCR below zcr_min = 0.02)");
    assert!(energy > 0 && combined == 0);

    Ok(())
}
