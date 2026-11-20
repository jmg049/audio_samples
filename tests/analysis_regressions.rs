#![cfg(all(
    feature = "pitch-analysis",
    feature = "onset-detection",
    feature = "beat-tracking",
    feature = "peak-picking",
    feature = "vad"
))]
//! Regression tests for pitch, key, onset, beat and VAD analysis.

use std::f64::consts::PI;
use std::num::NonZeroUsize;

use audio_samples::operations::beat::{onset_strength_envelope, track_beats_core};
use audio_samples::operations::onset::{ComplexOnsetConfig, OnsetDetectionConfig};
use audio_samples::operations::pitch_analysis::{Mode, PitchClass, yin_pitch_detection};
use audio_samples::operations::traits::{
    AudioOnsetDetection, AudioPitchAnalysis, AudioVoiceActivityDetection,
};
use audio_samples::operations::types::{PitchDetectionMethod, VadConfig, VadMethod};
use audio_samples::{AudioSampleError, AudioSamples};
use ndarray::Array1;
use non_empty_slice::{NonEmptySlice, NonEmptyVec};
use spectrograms::StftParams;

fn mono(data: Vec<f64>, sr: u32) -> AudioSamples<'static, f64> {
    AudioSamples::new_mono(
        Array1::from_vec(data),
        std::num::NonZeroU32::new(sr).unwrap(),
    )
    .unwrap()
}

fn sine(f: f64, sr: u32, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (2.0 * PI * f * i as f64 / f64::from(sr)).sin())
        .collect()
}

// ---------------------------------------------------------------------------
// YIN (CMND from tau = 1, fixed window, parabolic interpolation)
// ---------------------------------------------------------------------------

/// Pure sines: the estimate must lie within 0.5 % of the true f0. An
/// integer-lag YIN gives 444.44 Hz (16 kHz), 450 Hz (44.1 kHz), 103.90 Hz and
/// 153.85 Hz on these tones.
#[test]
fn yin_pure_sines_within_half_percent() {
    let cases = [
        (16_000u32, 440.0f64),
        (44_100, 440.0),
        (16_000, 100.0),
        (16_000, 150.0),
    ];
    for (sr, f0) in cases {
        let audio = mono(sine(f0, sr, sr as usize), sr);
        let est = audio
            .detect_pitch_yin(0.1, 60.0, 1000.0)
            .unwrap()
            .expect("pure tone must be voiced");
        let rel = (est - f0).abs() / f0;
        assert!(
            rel < 0.005,
            "YIN at fs={sr}: {f0} Hz estimated as {est:.3} Hz ({:.3} %)",
            rel * 100.0
        );
    }
}

/// Harmonic tones: 2048-sample frames at 16 kHz, amplitudes [1, 0.6, 0.4, 0.3],
/// search 60-800 Hz. The estimate must lie within 0.5 % of the true f0.
#[test]
fn yin_harmonic_tones_match_reference() {
    let sr = 16_000u32;
    let amps = [1.0, 0.6, 0.4, 0.3];
    for f0 in [100.0, 155.0, 220.0, 311.3, 440.0] {
        let x: Vec<f64> = (0..2048)
            .map(|i| {
                amps.iter()
                    .enumerate()
                    .map(|(h, a)| {
                        a * (2.0 * PI * f0 * (h + 1) as f64 * i as f64 / f64::from(sr)).sin()
                    })
                    .sum()
            })
            .collect();
        let est = mono(x, sr)
            .detect_pitch_yin(0.1, 60.0, 800.0)
            .unwrap()
            .expect("harmonic tone must be voiced");
        let rel = (est - f0).abs() / f0;
        assert!(
            rel < 0.005,
            "{f0} Hz harmonic tone estimated as {est:.3} Hz"
        );
    }
}

/// The low-level primitive returns a fractional period, and `track_pitch`
/// (which shares it) is consistent with the single-shot estimate.
#[test]
fn yin_fractional_period_and_track_pitch_consistent() {
    let sr = 44_100u32;
    let f0 = 440.0;
    let x = sine(f0, sr, 8192);
    let slice = NonEmptySlice::from_slice(&x).unwrap();
    let tau = yin_pitch_detection(slice, 44, 551, 0.1).unwrap();
    let true_period = f64::from(sr) / f0; // 100.227...
    assert!(
        (tau - true_period).abs() < 0.1,
        "period {tau} vs {true_period}"
    );

    let audio = mono(sine(f0, sr, sr as usize / 2), sr);
    let contour = audio
        .track_pitch(
            NonZeroUsize::new(2048).unwrap(),
            NonZeroUsize::new(512).unwrap(),
            PitchDetectionMethod::Yin,
            0.1,
            80.0,
            1000.0,
        )
        .unwrap();
    for (_, hz) in contour.voiced_frames() {
        assert!((hz - f0).abs() / f0 < 0.005, "frame estimate {hz}");
    }
    assert!(contour.voiced_frames().count() > 0);
}

/// Silence stays unvoiced.
#[test]
fn yin_silence_is_none() {
    let audio = mono(vec![0.0; 16_000], 16_000);
    assert!(audio.detect_pitch_yin(0.1, 60.0, 800.0).unwrap().is_none());
}

// ---------------------------------------------------------------------------
// key-profile rotation
// ---------------------------------------------------------------------------

/// A D major chord (D, F#, A plus the other D-major scale tones weighted by
/// the Krumhansl major profile) must be estimated as D major. Rotating the
/// profile the wrong way (`profile[(i + tonic) % 12]`) mirrors keys about C
/// (D -> A#).
#[test]
fn key_d_major() {
    let sr = 22_050u32;
    let n = sr as usize * 2;
    // Krumhansl-Kessler major profile, index 0 = tonic.
    let profile = [
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
    ];
    let tonic = 2usize; // D
    let scale = [0usize, 2, 4, 5, 7, 9, 11];
    let mut x = vec![0.0f64; n];
    for &deg in &scale {
        let pc = (tonic + deg) % 12;
        // Octave 4: MIDI 60 + pc.
        let f = 440.0 * 2f64.powf((60.0 + pc as f64 - 69.0) / 12.0);
        let a = profile[deg] / 6.35 * 0.1;
        for (i, v) in x.iter_mut().enumerate() {
            *v += a * (2.0 * PI * f * i as f64 / f64::from(sr)).sin();
        }
    }
    let audio = mono(x, sr);
    let params = StftParams::builder()
        .n_fft(NonZeroUsize::new(4096).unwrap())
        .hop_size(NonZeroUsize::new(1024).unwrap())
        .build()
        .unwrap();
    let key = audio.estimate_key(&params).unwrap();
    assert_eq!(key.tonic, PitchClass::D, "got {} {:?}", key.tonic, key.mode);
    assert_eq!(key.mode, Mode::Major);
}

// ---------------------------------------------------------------------------
// complex onset detection must not panic on real input
// ---------------------------------------------------------------------------

fn click_track(sr: u32, secs: f64, first: f64, period: f64) -> Vec<f64> {
    let n = (f64::from(sr) * secs) as usize;
    let mut x = vec![0.0; n];
    let mut t = first;
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    while t < secs {
        let start = (t * f64::from(sr)) as usize;
        for k in 0..256 {
            if start + k < n {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let r = (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
                x[start + k] += 0.9 * (-(k as f64) / 24.0).exp() * r;
            }
        }
        t += period;
    }
    x
}

#[test]
fn complex_onset_detection_runs_on_clicks() {
    let sr = 44_100;
    let audio = mono(click_track(sr, 2.0, 0.25, 0.5), sr);
    let onsets = audio
        .complex_onset_detection(&ComplexOnsetConfig::default())
        .expect("complex onset detection must not fail");
    assert!(onsets.iter().all(|t| t.is_finite() && *t >= 0.0));
    let odf = audio
        .onset_detection_function_complex(&ComplexOnsetConfig::default())
        .unwrap();
    assert!(odf.iter().all(|v| v.is_finite()));
}

// ---------------------------------------------------------------------------
// envelope smoothing width is in frames, independent of the CQT
// window length in samples
// ---------------------------------------------------------------------------

#[test]
fn envelope_smoothing_independent_of_cqt_window() {
    let sr = 44_100;
    let audio = mono(click_track(sr, 3.0, 0.25, 0.5), sr);
    let mut config = OnsetDetectionConfig::default();
    config.window_size = Some(NonZeroUsize::new(4096).unwrap());

    let env = onset_strength_envelope(&audio, &config, None).unwrap();
    let (_, odf) = audio.onset_detection_function(&config).unwrap();

    // Expected: centred moving average with half-width 3 frames, then ln(1 + 0.5 x).
    let odf: Vec<f64> = odf.to_vec();
    let len = odf.len();
    for i in 0..len {
        let lo = i.saturating_sub(3);
        let hi = (i + 4).min(len);
        let mean = odf[lo..hi].iter().sum::<f64>() / (hi - lo) as f64;
        let expected = (0.5 * mean).ln_1p();
        assert!(
            (env[i] - expected).abs() < 1e-12,
            "frame {i}: envelope {} vs expected {expected}",
            env[i]
        );
    }
    // And it is not flattened (as averaging over ±4096 frames would do).
    let max = env.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = env.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(max - min > 1e-3, "envelope is flat: {min}..{max}");
}

// ---------------------------------------------------------------------------
// beat grid must not drift when the IBI is not an integer number of
// frames
// ---------------------------------------------------------------------------

#[test]
fn beat_grid_no_drift_over_120s() {
    let sr = 44_100.0;
    let hop = 512usize;
    let bpm = 120.0;
    let ibi = 60.0 / bpm * sr / hop as f64; // 43.066 frames
    let n_frames = (120.0 * sr / hop as f64) as usize;
    let mut env = vec![0.0; n_frames];
    let mut true_frames = Vec::new();
    let mut k = 0usize;
    loop {
        let f = (k as f64 * ibi).round() as usize;
        if f >= n_frames {
            break;
        }
        // Make the first beat the global maximum (the tracker's anchor).
        env[f] = if k == 0 { 2.0 } else { 1.0 };
        true_frames.push(f);
        k += 1;
    }
    let env = NonEmptyVec::new(env).unwrap();
    let mut beats = track_beats_core(&env, bpm, sr, NonZeroUsize::new(hop).unwrap(), None)
        .expect("beat tracking");
    beats.sort_by(f64::total_cmp);
    let hop_t = hop as f64 / sr;
    assert_eq!(beats.len(), true_frames.len());
    for (b, &f) in beats.iter().zip(&true_frames) {
        let frame = (b / hop_t).round() as i64;
        assert!(
            (frame - f as i64).abs() <= 1,
            "beat at frame {frame}, true beat at {f}"
        );
    }
}

// ---------------------------------------------------------------------------
// VadMethod::Spectral is not implemented and says so
// ---------------------------------------------------------------------------

#[test]
fn vad_spectral_returns_unsupported() {
    let audio = mono(sine(440.0, 16_000, 16_000), 16_000);
    let cfg = VadConfig::default().with_method(VadMethod::Spectral);
    let err = audio.voice_activity_mask(&cfg).unwrap_err();
    assert!(
        matches!(err, AudioSampleError::Unsupported { .. }),
        "expected Unsupported, got {err:?}"
    );
}

/// Pitch methods that are not implemented return an error from `track_pitch`.
#[test]
fn unimplemented_pitch_methods_error() {
    let audio = mono(sine(220.0, 16_000, 16_000), 16_000);
    for m in [
        PitchDetectionMethod::Cepstrum,
        PitchDetectionMethod::HarmonicProduct,
    ] {
        assert!(
            audio
                .track_pitch(
                    NonZeroUsize::new(1024).unwrap(),
                    NonZeroUsize::new(256).unwrap(),
                    m,
                    0.1,
                    60.0,
                    800.0
                )
                .is_err()
        );
    }
}

// ---------------------------------------------------------------------------
// examples/vad.rs: a speech-like (formant-weighted harmonic) burst between
// two low-level noise segments is detected by the Combined method.
// ---------------------------------------------------------------------------

#[test]
fn vad_combined_detects_speech_like_burst() {
    let sr = 16_000u32;
    let n = sr as usize; // 1 s
    let (on, off) = (4_000usize, 12_000usize);
    let mut x = vec![0.0f64; n];
    let mut seed = 12345u64;
    for (i, v) in x.iter_mut().enumerate() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = ((seed >> 33) as f64 / (1u64 << 30) as f64) - 1.0; // uniform in [-1, 1)
        *v = 1e-3 * r; // background noise ≈ −65 dBFS RMS
        if (on..off).contains(&i) {
            let t = i as f64 / f64::from(sr);
            // f0 = 125 Hz, harmonics weighted towards a 700 Hz "formant".
            for h in 1..=24 {
                let f = 125.0 * h as f64;
                let a = 0.3 / (1.0 + ((f - 700.0) / 200.0).powi(2));
                *v += a * (2.0 * PI * f * t).sin();
            }
        }
    }
    let audio = mono(x, sr);
    let cfg = VadConfig::default()
        .with_method(VadMethod::Combined)
        .with_hop_size(NonZeroUsize::new(160).unwrap())
        .unwrap()
        .with_frame_size(NonZeroUsize::new(400).unwrap())
        .unwrap();
    let regions = audio.speech_regions(&cfg).unwrap();
    assert_eq!(regions.len(), 1, "regions: {regions:?}");
    let (s, e) = regions[0];
    assert!(s.abs_diff(on) <= 800, "start {s}");
    assert!(e.abs_diff(off) <= 1600, "end {e}");
}

/// With `pad_end = false` and a signal shorter than one frame there are no
/// frames: this must be a proper error, not an (undefined-behaviour) empty
/// `NonEmptyVec`.
#[test]
fn vad_short_signal_without_padding_is_an_error() {
    let audio = mono(sine(440.0, 16_000, 100), 16_000);
    let cfg = VadConfig::default().with_pad_end(false); // frame_size 1024 > 100
    let err = audio.voice_activity_mask(&cfg).unwrap_err();
    assert!(matches!(err, AudioSampleError::Parameter(_)), "got {err:?}");
    assert!(audio.speech_regions(&cfg).is_err());
    // With padding the same signal yields one (zero-padded) frame.
    let ok = audio.voice_activity_mask(&VadConfig::default()).unwrap();
    assert_eq!(ok.len().get(), 1);
}
