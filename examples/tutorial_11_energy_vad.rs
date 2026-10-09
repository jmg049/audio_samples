//! Tutorial 11: Short-time energy, zero crossings and voice activity detection.
//!
//! Accompanies the guide chapter `audio_samples::guide::c11_energy_vad`.
//!
//! What this lab shows, with every number checked by an `assert!`:
//!
//! 1. A synthetic utterance at 16 kHz: 300 ms silence, 400 ms "vowel"
//!    (200 Hz harmonics shaped by a 700 Hz formant), 200 ms "fricative"
//!    (white noise), 300 ms silence, all over a -65 dBFS noise floor.
//! 2. Frame-wise short-time energy (dBFS) and zero-crossing rate (crossings
//!    per sample) separate silence, voiced and unvoiced frames.
//! 3. `voice_activity_mask` with the energy rule is reproduced exactly by a
//!    plain-Rust implementation (RMS, threshold, hangover).
//! 4. The combined energy + ZCR rule keeps the vowel and rejects the fricative.
//! 5. Hangover stretches the end of a region by whole hops.
//! 6. `VadConfig::explain` turns the settings into milliseconds and hertz.
//!
//! Run with:
//! `cargo run --release --example tutorial_11_energy_vad --features educational,full_no_plotting`

use std::f64::consts::PI;
use std::num::NonZeroUsize;
use std::time::Duration;

use audio_samples::educational::ExplainFor;
use audio_samples::operations::types::{VadConfig, VadMethod};
use audio_samples::utils::generation::{MonoSampleBuilder, ToneComponent};
use audio_samples::{
    AudioSampleResult, AudioSamples, AudioStatistics, AudioVoiceActivityDetection, compound_tone,
    nzu, sample_rate, white_noise,
};
use non_empty_slice::{NonEmptySlice, NonEmptyVec};

const FS: f64 = 16_000.0;
const FRAME: usize = 400; // 25 ms
const HOP: usize = 160; // 10 ms

/// The test utterance and the sample indices of its segment boundaries.
fn utterance() -> (AudioSamples<'static, f64>, [usize; 4]) {
    let sr = sample_rate!(16000);
    // Vowel: harmonics of 200 Hz, weighted by a resonance at 700 Hz.
    let partials: Vec<ToneComponent> = (1..=20)
        .map(|k| {
            let f = 200.0 * k as f64;
            ToneComponent::new(f, 0.3 / (1.0 + ((f - 700.0) / 200.0).powi(2)))
        })
        .collect();
    let vowel = compound_tone::<f64>(
        NonEmptySlice::from_slice(&partials).unwrap(),
        Duration::from_millis(400),
        sr,
    );
    let speech = MonoSampleBuilder::<f64>::new(sr)
        .silence(Duration::from_millis(300))
        .add_sample(vowel)
        .white_noise(Duration::from_millis(200), 0.1, Some(1)) // "s"
        .silence(Duration::from_millis(300))
        .build()
        .unwrap();
    // Background noise floor: uniform in [-1e-3, 1e-3), RMS 1e-3/sqrt(3) = -64.8 dBFS.
    let floor = white_noise::<f64>(Duration::from_millis(1200), sr, 1e-3, Some(2));
    let x: Vec<f64> = speech
        .as_slice()
        .unwrap()
        .iter()
        .zip(floor.as_slice().unwrap())
        .map(|(a, b)| a + b)
        .collect();
    let audio = AudioSamples::from_mono_vec(NonEmptyVec::new(x).unwrap(), sr);
    (audio, [4_800, 11_200, 14_400, 19_200])
}

/// Frame RMS and ZCR exactly as the library's VAD defines them (pad_end = true:
/// a partial last frame counts its missing samples as zeros; ZCR skips exact
/// zeros and divides by FRAME - 1).
fn frame_features(x: &[f64]) -> Vec<(f64, f64)> {
    (0..x.len().div_ceil(HOP))
        .map(|m| {
            let frame = &x[m * HOP..(m * HOP + FRAME).min(x.len())];
            let rms = (frame.iter().map(|v| v * v).sum::<f64>() / FRAME as f64).sqrt();
            let signs: Vec<f64> = frame
                .iter()
                .filter(|v| **v != 0.0)
                .map(|v| v.signum())
                .collect();
            let crossings = signs.windows(2).filter(|w| w[0] != w[1]).count();
            (rms, crossings as f64 / (FRAME - 1) as f64)
        })
        .collect()
}

/// Energy-only VAD with hangover, no smoothing, no minimum-run rules.
fn my_energy_vad(features: &[(f64, f64)], threshold_db: f64, hangover: usize) -> Vec<bool> {
    let thr = 10f64.powf(threshold_db / 20.0);
    let mut hold = 0;
    features
        .iter()
        .map(|&(rms, _)| {
            if rms >= thr {
                hold = hangover;
                true
            } else if hold > 0 {
                hold -= 1;
                true
            } else {
                false
            }
        })
        .collect()
}

fn db(x: f64) -> f64 {
    20.0 * x.log10()
}

fn plain_config(method: VadMethod, hangover: NonZeroUsize) -> AudioSampleResult<VadConfig> {
    Ok(VadConfig::default()
        .with_method(method)
        .with_hop_size(nzu!(160))? // set the hop first: it must stay <= frame size
        .with_frame_size(nzu!(400))?
        .with_energy_threshold_db(-40.0)
        .with_smooth_frames(nzu!(1))
        .with_min_speech_frames(0)
        .with_min_silence_frames(0)
        .with_hangover_frames(hangover))
}

fn main() -> AudioSampleResult<()> {
    println!("== Tutorial 11: short-time energy, ZCR and VAD ==\n");
    let (audio, [on, v_end, s_end, len]) = utterance();
    let x = audio.as_slice().unwrap();
    assert_eq!(x.len(), len);

    // ---------------------------------------------------------------- 1, 2
    let feats = frame_features(x);
    assert_eq!(feats.len(), 120); // ceil(19200 / 160)
    println!("1-2. Frame features (25 ms frames, 10 ms hop), one frame from each segment:");
    let pick = |t_ms: usize| feats[t_ms * 16 / HOP];
    let rows = [
        ("silence", pick(100)),
        ("vowel", pick(500)),
        ("fricative", pick(780)),
        ("silence", pick(1_050)),
    ];
    for (name, (rms, zcr)) in rows {
        println!(
            "     {name:<9} level {:>6.1} dBFS   ZCR {zcr:.3} crossings/sample (~{:.0} Hz)",
            db(rms),
            zcr * FS / 2.0
        );
    }
    let (sil, vow, fri) = (rows[0].1, rows[1].1, rows[2].1);
    // Noise floor: RMS 1e-3/sqrt(3) = -64.8 dBFS (random, so +-1 dB).
    assert!((db(sil.0) + 64.8).abs() < 1.0);
    // Vowel: RMS from the harmonic amplitudes, sqrt(sum a_k^2 / 2).
    let vowel_rms = ((1..=20)
        .map(|k| {
            let f = 200.0 * k as f64;
            (0.3 / (1.0 + ((f - 700.0) / 200.0).powi(2))).powi(2) / 2.0
        })
        .sum::<f64>())
    .sqrt();
    println!("     predicted vowel level {:.1} dBFS", db(vowel_rms));
    // A 25 ms frame holds exactly 5 periods of 200 Hz, so the frame RMS equals
    // the long-term RMS up to the tiny noise floor.
    assert!((db(vow.0) - db(vowel_rms)).abs() < 0.05);
    // Fricative: uniform noise of amplitude 0.1 -> RMS 0.1/sqrt(3) = -24.8 dBFS.
    assert!((db(fri.0) + 24.8).abs() < 0.5);
    // ZCR: white noise crosses zero on about half of the sample pairs; the
    // vowel's crossings follow its strongest partials (600-800 Hz).
    assert!((fri.1 - 0.5).abs() < 0.06 && (sil.1 - 0.5).abs() < 0.06);
    assert!(vow.1 > 0.06 && vow.1 < 0.11);
    println!();

    // Whole-signal ZCR from the library is in crossings per second.
    let per_second = audio.zero_crossing_rate();
    println!(
        "     library zero_crossing_rate() over the whole signal: {per_second:.0} crossings/s = {:.3} crossings/sample",
        per_second / FS
    );
    assert!((per_second - audio.zero_crossings() as f64 * FS / len as f64).abs() < 1e-9);

    // ---------------------------------------------------------------- 3
    println!("3. Energy VAD (-40 dBFS, hangover 1, no smoothing): library versus hand-written");
    let cfg = plain_config(VadMethod::Energy, nzu!(1))?;
    let lib_mask: Vec<bool> = audio.voice_activity_mask(&cfg)?.to_vec();
    let my_mask = my_energy_vad(&feats, -40.0, 1);
    assert_eq!(lib_mask, my_mask);
    let regions = audio.speech_regions(&cfg)?;
    println!(
        "     masks identical ({} frames); regions: {regions:?}",
        lib_mask.len()
    );
    assert_eq!(regions.len(), 1);
    let (s, e) = regions[0];
    // First frame overlapping the vowel starts at the first multiple of HOP
    // greater than on - FRAME; the region ends at the end of the first inactive
    // frame after the last active one (+ hangover).
    let first = (on - FRAME) / HOP + 1;
    let last = (s_end - 1) / HOP; // last frame that contains fricative samples
    assert_eq!(s, first * HOP);
    assert_eq!(e, (last + 1 + 1) * HOP + FRAME);
    println!(
        "     speech {:.3}-{:.3} s (true {:.3}-{:.3} s)\n",
        s as f64 / FS,
        e as f64 / FS,
        on as f64 / FS,
        s_end as f64 / FS
    );

    // ---------------------------------------------------------------- 4
    println!("4. Combined rule (energy AND 0.02 <= ZCR <= 0.30) drops the fricative");
    let cfg_c = plain_config(VadMethod::Combined, nzu!(1))?;
    let rc = audio.speech_regions(&cfg_c)?;
    println!("     regions: {rc:?}");
    assert_eq!(rc.len(), 1);
    let (cs, ce) = rc[0];
    // A frame that is mostly noise floor has a noise-like ZCR, so the combined
    // rule starts one or two hops later than the energy rule.
    assert!(cs >= s && cs.abs_diff(on) <= FRAME);
    // The region now ends near the vowel's end (11200), not the fricative's
    // (14400): within one frame plus the hangover.
    assert!(ce.abs_diff(v_end) <= FRAME + 2 * HOP);
    println!(
        "     speech {:.3}-{:.3} s (vowel {:.3}-{:.3} s)\n",
        cs as f64 / FS,
        ce as f64 / FS,
        on as f64 / FS,
        v_end as f64 / FS
    );

    // ---------------------------------------------------------------- 5
    println!("5. Hangover: region end versus hangover frames");
    let mut prev = None;
    for h in [1usize, 3, 6, 10] {
        let cfg_h = plain_config(VadMethod::Energy, NonZeroUsize::new(h).unwrap())?;
        let (_, end) = audio.speech_regions(&cfg_h)?[0];
        println!(
            "     hangover {h:>2} frames ({:>3} ms): region ends at {end} ({:.3} s)",
            h * 10,
            end as f64 / FS
        );
        if let Some((ph, pe)) = prev {
            // Each extra hangover frame moves the end by exactly one hop.
            assert_eq!(end - pe, (h - ph) * HOP);
        }
        prev = Some((h, end));
    }
    println!();

    // ---------------------------------------------------------------- 6
    println!("6. What the library says about the default configuration at 16 kHz:\n");
    let doc = audio.explain(&VadConfig::default());
    println!("{doc}");
    assert_eq!(doc.quantity("Frame duration").unwrap().value, 64.0); // 1024 samples
    assert_eq!(
        doc.quantity("ZCR lower bound as a tone").unwrap().value,
        160.0
    );
    assert!(doc.has_warning_containing("64 ms"));
    let doc = audio.explain(&cfg);
    assert_eq!(doc.quantity("Frame duration").unwrap().value, 25.0);
    assert_eq!(doc.quantity("Hop duration").unwrap().value, 10.0);
    assert_eq!(doc.quantity("Frames").unwrap().value, 120.0);

    // Bonus: a pure tone's ZCR is 2 f / fs.
    let tone: Vec<f64> = (0..FRAME)
        .map(|n| (2.0 * PI * 500.0 * (n as f64 + 0.5) / FS).sin())
        .collect();
    let z = frame_features(&tone)[0].1;
    println!(
        "500 Hz tone: ZCR {z:.4} (2 f / fs = {:.4})",
        2.0 * 500.0 / FS
    );
    assert!((z - 2.0 * 500.0 / FS).abs() < 1.0 / (FRAME - 1) as f64);

    println!("\nAll checks passed.");
    Ok(())
}
