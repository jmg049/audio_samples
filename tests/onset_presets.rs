#![cfg(all(feature = "onset-detection", feature = "peak-picking"))]
//! The onset presets are specified in seconds: their frame counts must span the
//! intended durations at every sample rate and hop size, and the presets must
//! find the onsets of the material they are named for.

use std::f64::consts::PI;
use std::num::{NonZeroU32, NonZeroUsize};

use audio_samples::AudioSamples;
use audio_samples::operations::onset::{
    ComplexOnsetConfig, OnsetDetectionConfig, SpectralFluxConfig,
};
use audio_samples::operations::traits::AudioOnsetDetection;
use audio_samples::operations::types::PeakPickingConfig;
use ndarray::Array1;

const RATES: [u32; 6] = [8_000, 16_000, 22_050, 44_100, 48_000, 96_000];
const HOPS: [usize; 5] = [64, 128, 256, 512, 1024];

/// Intended durations of each preset, in seconds: (threshold window, minimum
/// separation).
fn intended(name: &str) -> (f64, f64) {
    match name {
        "music" => (0.2, 0.1),
        "speech" => (0.2, 0.08),
        "drums" => (0.1, 0.03),
        _ => unreachable!(),
    }
}

fn preset(name: &str, sr: NonZeroU32, hop: NonZeroUsize) -> PeakPickingConfig {
    match name {
        "music" => PeakPickingConfig::music(sr, hop),
        "speech" => PeakPickingConfig::speech(sr, hop),
        "drums" => PeakPickingConfig::drums(sr, hop),
        _ => unreachable!(),
    }
}

/// A rounded length is within half a frame of the intended duration; a length
/// rounded to an odd count is within one frame. Durations shorter than one
/// frame give one frame. No preset uses the median filter.
#[test]
fn preset_lengths_span_the_intended_seconds() {
    for name in ["music", "speech", "drums"] {
        let (window, separation) = intended(name);
        for sr in RATES {
            for hop in HOPS {
                let frame = hop as f64 / f64::from(sr);
                let cfg = preset(
                    name,
                    NonZeroU32::new(sr).unwrap(),
                    NonZeroUsize::new(hop).unwrap(),
                );
                cfg.validate().unwrap();
                let ctx = format!("{name} at {sr} Hz, hop {hop}");

                let sep = cfg.min_peak_separation.get() as f64 * frame;
                if separation >= frame {
                    assert!(
                        (sep - separation).abs() <= frame / 2.0 + 1e-12,
                        "{ctx}: separation {sep} s"
                    );
                } else {
                    assert_eq!(cfg.min_peak_separation.get(), 1, "{ctx}");
                }

                let w = cfg.adaptive_threshold.window_size;
                assert_eq!(w % 2, 1, "{ctx}: window {w} frames is not odd");
                let span = w as f64 * frame;
                assert!(
                    (span - window).abs() <= frame + 1e-12 || w == 1,
                    "{ctx}: window {span} s"
                );

                assert!(!cfg.median_filter, "{ctx}");
            }
        }
    }
}

/// Spot checks of the frame counts at common settings.
#[test]
fn preset_lengths_at_common_settings() {
    let nz = |v: usize| NonZeroUsize::new(v).unwrap();
    let sr = |v: u32| NonZeroU32::new(v).unwrap();
    // 22.05 kHz, hop 512: 43.07 frames per second.
    let m = PeakPickingConfig::music(sr(22_050), nz(512));
    assert_eq!(
        (
            m.adaptive_threshold.window_size,
            m.min_peak_separation.get()
        ),
        (9, 4)
    );
    // 44.1 kHz, hop 256: 172.27 frames per second.
    let d = PeakPickingConfig::drums(sr(44_100), nz(256));
    assert_eq!(
        (
            d.adaptive_threshold.window_size,
            d.min_peak_separation.get()
        ),
        (17, 5)
    );
    // 16 kHz, hop 256: 62.5 frames per second.
    let s = PeakPickingConfig::speech(sr(16_000), nz(256));
    assert_eq!(
        (
            s.adaptive_threshold.window_size,
            s.min_peak_separation.get()
        ),
        (13, 5)
    );
    // The same durations at twice the rate span twice the frames.
    let a = PeakPickingConfig::music(sr(16_000), nz(160));
    let b = PeakPickingConfig::music(sr(32_000), nz(160));
    assert_eq!(a.min_peak_separation.get() * 2, b.min_peak_separation.get());
}

/// The onset-detector presets build their peak picker for their own hop size.
#[test]
fn detector_presets_use_their_hop() {
    for rate in [22_050, 44_100, 48_000] {
        let sr = NonZeroU32::new(rate).unwrap();
        let o = OnsetDetectionConfig::percussive(sr);
        assert_eq!(o.peak_picking, PeakPickingConfig::drums(sr, o.hop_size));
        let o = OnsetDetectionConfig::musical(sr);
        assert_eq!(o.peak_picking, PeakPickingConfig::music(sr, o.hop_size));
        let o = OnsetDetectionConfig::speech(sr);
        assert_eq!(o.peak_picking, PeakPickingConfig::speech(sr, o.hop_size));
        let f = SpectralFluxConfig::percussive(sr);
        assert_eq!(f.peak_picking, PeakPickingConfig::drums(sr, f.hop_size));
        let f = SpectralFluxConfig::musical(sr);
        assert_eq!(f.peak_picking, PeakPickingConfig::music(sr, f.hop_size));
        let c = ComplexOnsetConfig::percussive(sr);
        assert_eq!(c.peak_picking, PeakPickingConfig::drums(sr, c.hop_size));
        let c = ComplexOnsetConfig::musical(sr);
        assert_eq!(c.peak_picking, PeakPickingConfig::music(sr, c.hop_size));
        let c = ComplexOnsetConfig::speech(sr);
        assert_eq!(c.peak_picking, PeakPickingConfig::speech(sr, c.hop_size));
    }
}

// ---------------------------------------------------------------------------
// Peak picking with each preset behaves the same in seconds at every rate and
// hop size
// ---------------------------------------------------------------------------

/// An onset strength function at `fps` frames per second: Gaussian bumps of
/// standard deviation `width` seconds at `times` (heights `heights`) on a slow
/// ripple of up to 6 % of the strongest bump. The ripple is a function of time,
/// so it looks the same at every frame rate.
fn odf(fps: f64, secs: f64, times: &[f64], heights: &[f64], width: f64) -> Vec<f64> {
    let n = (secs * fps).round() as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / fps;
            let bumps: f64 = times
                .iter()
                .zip(heights)
                .map(|(&c, &h)| h * (-0.5 * ((t - c) / width).powi(2)).exp())
                .sum();
            let ripple = 0.03 * (1.0 + (2.0 * PI * 5.3 * t).sin() * (2.0 * PI * 11.7 * t).cos());
            bumps + ripple
        })
        .collect()
}

/// Each preset keeps onsets spaced well beyond its minimum separation and
/// merges a weaker second peak that follows within half the separation, at
/// every sample rate and hop size: the behaviour is set in seconds, not frames.
#[test]
fn preset_peak_picking_is_rate_and_hop_independent() {
    use audio_samples::operations::peak_picking::pick_peaks;
    use non_empty_slice::NonEmptySlice;

    let settings: [(u32, usize); 6] = [
        (16_000, 160),
        (22_050, 256),
        (22_050, 512),
        (44_100, 256),
        (44_100, 512),
        (48_000, 1024),
    ];
    for name in ["music", "speech", "drums"] {
        let (_, separation) = intended(name);
        let onsets = [0.3, 0.9, 1.5, 2.1];
        let mut times = onsets.to_vec();
        let mut heights = vec![1.0, 0.8, 1.0, 0.9];
        // A weaker echo of the second onset, half a separation later.
        times.push(0.9 + separation / 2.0);
        heights.push(0.5);
        for (rate, hop) in settings {
            let fps = f64::from(rate) / hop as f64;
            // Musical and speech onsets are spread over tens of milliseconds;
            // a drum hit lasts a few milliseconds but fills the frame it falls in.
            let width = if name == "drums" {
                0.004f64.max(0.6 / fps)
            } else {
                0.015
            };
            let cfg = preset(
                name,
                NonZeroU32::new(rate).unwrap(),
                NonZeroUsize::new(hop).unwrap(),
            );
            let strength = odf(fps, 2.5, &times, &heights, width);
            let peaks = pick_peaks(NonEmptySlice::from_slice(&strength).unwrap(), &cfg).unwrap();
            let found: Vec<f64> = peaks.iter().map(|&p| p as f64 / fps).collect();
            let ctx = format!("{name} at {rate} Hz, hop {hop}");
            assert_eq!(found.len(), onsets.len(), "{ctx}: found {found:.3?}");
            for (f, t) in found.iter().zip(onsets) {
                assert!(
                    (f - t).abs() <= 1.0 / fps + width,
                    "{ctx}: peak {f:.3} s vs {t} s"
                );
            }
        }
    }
}

/// The onset detectors run with every preset and return sorted times inside the
/// signal.
#[test]
fn detector_presets_run() {
    let fs = 22_050u32;
    let sr = NonZeroU32::new(fs).unwrap();
    let n = 2 * fs as usize;
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut x = vec![0.0; n];
    for start in (0..n).step_by(fs as usize / 4) {
        for k in 0..256.min(n - start) {
            x[start + k] = 0.9 * (-(k as f64) / 24.0).exp() * noise(&mut seed);
        }
    }
    let audio = AudioSamples::new_mono(Array1::from_vec(x), sr).unwrap();
    let check = |times: Vec<f64>, ctx: &str| {
        assert!(
            times.windows(2).all(|w| w[0] < w[1]),
            "{ctx}: {times:?} not increasing"
        );
        assert!(
            times.iter().all(|&t| (0.0..2.0).contains(&t)),
            "{ctx}: {times:?}"
        );
    };
    for c in [
        OnsetDetectionConfig::percussive(sr),
        OnsetDetectionConfig::musical(sr),
        OnsetDetectionConfig::speech(sr),
    ] {
        check(audio.detect_onsets(&c).unwrap(), "energy");
    }
    for c in [
        SpectralFluxConfig::percussive(sr),
        SpectralFluxConfig::musical(sr),
    ] {
        check(audio.detect_onsets_spectral_flux(&c).unwrap(), "flux");
    }
    for c in [
        ComplexOnsetConfig::percussive(sr),
        ComplexOnsetConfig::musical(sr),
        ComplexOnsetConfig::speech(sr),
    ] {
        check(audio.complex_onset_detection(&c).unwrap(), "complex");
    }
}

/// xorshift noise in [-1, 1).
fn noise(seed: &mut u64) -> f64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    (*seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// Noise clicks starting at `times` (seconds), with peak amplitudes `heights`.
fn clicks(fs: u32, secs: f64, times: &[f64], heights: &[f64]) -> AudioSamples<'static, f64> {
    let n = (secs * f64::from(fs)) as usize;
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut x = vec![0.0; n];
    for (&t, &h) in times.iter().zip(heights) {
        let start = (t * f64::from(fs)) as usize;
        for k in 0..256.min(n - start) {
            x[start + k] = h * (-(k as f64) / 24.0).exp() * noise(&mut seed);
        }
    }
    AudioSamples::new_mono(Array1::from_vec(x), NonZeroU32::new(fs).unwrap()).unwrap()
}

/// The global threshold, the minimum spacing and the pre-emphasis of
/// `detect_onsets` are the `peak_picking` settings, and each changes the result.
#[test]
fn peak_picking_sets_threshold_spacing_and_pre_emphasis() {
    use audio_samples::operations::peak_picking::pick_peaks;
    use non_empty_slice::NonEmptyVec;

    let base = || {
        let mut c = OnsetDetectionConfig::default();
        c.peak_picking.median_filter = false;
        c.peak_picking.pre_emphasis = false;
        c.peak_picking.adaptive_threshold.window_size = 11;
        c.peak_picking.min_peak_separation = NonZeroUsize::new(5).unwrap();
        c
    };
    let times = [0.25, 0.75, 1.25, 1.75];

    // Spacing: clicks 0.5 s apart are four onsets with a 5-frame (58 ms) minimum
    // spacing and two with a 50-frame (0.58 s) one.
    let even = clicks(44_100, 2.0, &times, &[0.9; 4]);
    assert_eq!(even.detect_onsets(&base()).unwrap().len(), 4);
    let mut spaced = base();
    spaced.peak_picking.min_peak_separation = NonZeroUsize::new(50).unwrap();
    assert_eq!(even.detect_onsets(&spaced).unwrap().len(), 2);

    // Threshold: a click at 2 % of the others' level falls below the default
    // floor (0.1 of the normalised strength) and is found with a floor of 0.005.
    let weak = clicks(44_100, 2.0, &times, &[0.9, 0.9, 0.018, 0.9]);
    assert_eq!(weak.detect_onsets(&base()).unwrap().len(), 3);
    let mut low_floor = base();
    low_floor.peak_picking.adaptive_threshold.min_threshold = 0.005;
    assert_eq!(weak.detect_onsets(&low_floor).unwrap().len(), 4);

    // Pre-emphasis: a slow rise and fall peaking at 1.0, and a one-frame spike of
    // 0.08. Without it the spike is below the floor; pre-emphasis
    // (y[n] = x[n] - 0.97 x[n-1]) flattens the slow slopes, so the spike becomes
    // the strongest peak and both are found.
    let mut odf = vec![0.0; 160];
    for k in 0..50 {
        odf[10 + k] = (k + 1) as f64 / 50.0;
        odf[60 + k] = (49 - k) as f64 / 50.0;
    }
    odf[140] = 0.08;
    let odf = NonEmptyVec::new(odf).unwrap();
    let mut picker = base().peak_picking;
    assert_eq!(pick_peaks(&odf, &picker).unwrap(), vec![59]);
    picker.pre_emphasis = true;
    picker.pre_emphasis_coeff = 0.97;
    assert_eq!(pick_peaks(&odf, &picker).unwrap(), vec![59, 140]);
}
