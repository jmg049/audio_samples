//! Shared body for the `AudioPitchAnalysis` bench targets
//! (`bench_pitch_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/pitch.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Catalog section: Pitch (lines 373-388 of CATALOG.md).
//! Methodology: bench_suite/METHODOLOGY.md.
//!
//! All trait methods in this section are mono-only and operate on `f64`
//! internally (the trait `to_format::<f64>()`s up front). Algorithms swept
//! here are YIN, autocorrelation, FFT-based HNR / harmonic analysis, and
//! chromagram-based key estimation. The `fixture_a440` deterministic 440 Hz
//! sine gives the pitch detectors a known fundamental.

use std::num::NonZeroUsize;

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use audio_samples::{AudioPitchAnalysis, I24};
use audio_samples::operations::pitch_analysis::{
    autocorr_pitch_detection, yin_pitch_detection,
};
use audio_samples::operations::types::PitchDetectionMethod;

use bench_suite_common::{
    DTYPES_DEFAULT, LENGTH_SWEEP_NO_XXXL, ParamLabel, SampleSizePolicy,
    fixture_a440, sample_size_for,
};

// Re-exports of spectrograms types used by the trait API. They come in via
// audio_samples' `transforms` (and hence `pitch-analysis`) feature.
use spectrograms::{StftParams, WindowType};

// All pitch ops are mono-only at the trait level.
const PITCH_CHANNELS: usize = 1;

// Frame-size sweep for windowed / framed analyses (track_pitch, estimate_key).
const FRAME_SIZES: &[usize] = &[1024, 2048, 4096];

// ===========================================================================
// Top-level entry point — wrappers call into this with their typed Criterion.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_pitch_001_detect_pitch_yin(c);
    bench_pitch_002_detect_pitch_autocorr(c);
    bench_pitch_003_track_pitch(c);
    bench_pitch_004_harmonic_to_noise_ratio(c);
    bench_pitch_005_harmonic_analysis(c);
    bench_pitch_006_estimate_key(c);
    bench_pitch_007_yin_pitch_detection(c);
    bench_pitch_008_autocorr_pitch_detection(c);
}

// ===========================================================================
// Dispatch macro — unary `audio.op()` over the four default dtypes on a 440
// Hz mono fixture.
// ===========================================================================

macro_rules! dispatch_mono_sine {
    ($group:expr, $id:expr, $dt:expr, $n:expr, |$audio:ident| $body:expr) => {{
        let id = $id;
        let n = $n;
        match $dt {
            "i16" => $group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, PITCH_CHANNELS),
                    |$audio| { black_box($body); },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched_ref(
                    || fixture_a440::<I24>(n, PITCH_CHANNELS),
                    |$audio| { black_box($body); },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched_ref(
                    || fixture_a440::<i32>(n, PITCH_CHANNELS),
                    |$audio| { black_box($body); },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, PITCH_CHANNELS),
                    |$audio| { black_box($body); },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

// Standard parameter-string + throughput tagging for the `(len, dt)` mono case.
fn label_and_throughput<M: Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
    catalog_id: &str,
    n: usize,
    dt: &str,
) -> BenchmarkId {
    group.throughput(Throughput::Elements(n as u64));
    let label = ParamLabel::new()
        .with("len", n).with("dt", dt).with("ch", PITCH_CHANNELS).build();
    BenchmarkId::new(catalog_id, label)
}

// ===========================================================================
// Pitch-001 detect_pitch_yin — NoFast; threshold axis.
//
// YIN's CMND step is O(n * max_tau). At fmin=80 Hz / sr=44.1 kHz, max_tau is
// ~551 samples; lengths below ~1102 short-circuit via the `max_tau >= n/2`
// guard but are still useful as cost-floor reference points.
// ===========================================================================

fn bench_pitch_001_detect_pitch_yin<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("pitch");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &threshold in &[0.10_f64, 0.20] {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements(len as u64));
                let label = ParamLabel::new()
                    .with("len", len).with("dt", dt).with("ch", PITCH_CHANNELS)
                    .with("threshold", format!("{threshold:.2}"))
                    .build();
                let id = BenchmarkId::new("Pitch-001_detect_pitch_yin", label);
                dispatch_mono_sine!(group, id, dt, len, |a| {
                    a.detect_pitch_yin(threshold, 80.0, 1000.0).ok()
                });
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Pitch-002 detect_pitch_autocorr — NoFast; O(n * max_tau) like YIN but
// without the normalisation pass. No threshold axis.
// ===========================================================================

fn bench_pitch_002_detect_pitch_autocorr<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("pitch");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Pitch-002_detect_pitch_autocorr", len, dt);
            dispatch_mono_sine!(group, id, dt, len, |a| {
                a.detect_pitch_autocorr(80.0, 1000.0).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Pitch-003 track_pitch — NoFast; frame_size axis × method axis. Hop = frame/4.
// Skip combinations where frame_size > signal length (trait would error out).
// ===========================================================================

fn bench_pitch_003_track_pitch<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("pitch");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &frame_size in FRAME_SIZES {
            if frame_size > len {
                continue;
            }
            let window = NonZeroUsize::new(frame_size).unwrap();
            let hop = NonZeroUsize::new((frame_size / 4).max(1)).unwrap();
            for &(method, method_label) in &[
                (PitchDetectionMethod::Yin, "yin"),
                (PitchDetectionMethod::Autocorrelation, "autocorr"),
            ] {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements(len as u64));
                    let label = ParamLabel::new()
                        .with("len", len).with("dt", dt).with("ch", PITCH_CHANNELS)
                        .with("frame_size", frame_size)
                        .with("hop", hop.get())
                        .with("method", method_label)
                        .build();
                    let id = BenchmarkId::new("Pitch-003_track_pitch", label);
                    dispatch_mono_sine!(group, id, dt, len, |a| {
                        a.track_pitch(window, hop, method, 0.10, 80.0, 1000.0).ok()
                    });
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Pitch-004 harmonic_to_noise_ratio — NoFast; FFT-bound. n_fft defaults to
// signal length when `None`; we let it default. 5 harmonics, Hanning window
// (default). Fundamental is 440 Hz to match the fixture.
// ===========================================================================

fn bench_pitch_004_harmonic_to_noise_ratio<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("pitch");
    let num_harmonics = NonZeroUsize::new(5).unwrap();
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(
                &mut group, "Pitch-004_harmonic_to_noise_ratio", len, dt,
            );
            dispatch_mono_sine!(group, id, dt, len, |a| {
                a.harmonic_to_noise_ratio(440.0, num_harmonics, None, Some(WindowType::Hanning)).ok()
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Pitch-005 harmonic_analysis — NoFast; FFT-bound; n_harmonics axis.
// tolerance=0.1; Hanning window; n_fft = signal length (default).
// ===========================================================================

fn bench_pitch_005_harmonic_analysis<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("pitch");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &nh in &[5usize, 10] {
            let num_harmonics = NonZeroUsize::new(nh).unwrap();
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements(len as u64));
                let label = ParamLabel::new()
                    .with("len", len).with("dt", dt).with("ch", PITCH_CHANNELS)
                    .with("n_harmonics", nh)
                    .build();
                let id = BenchmarkId::new("Pitch-005_harmonic_analysis", label);
                dispatch_mono_sine!(group, id, dt, len, |a| {
                    a.harmonic_analysis(
                        440.0, num_harmonics, 0.10, None, Some(WindowType::Hanning),
                    ).ok()
                });
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Pitch-006 estimate_key — NoFast; chromagram + correlation. frame_size axis
// (hop = frame/4). Skip combinations where frame_size > signal length.
// ===========================================================================

fn bench_pitch_006_estimate_key<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("pitch");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &frame_size in FRAME_SIZES {
            if frame_size > len {
                continue;
            }
            let n_fft = NonZeroUsize::new(frame_size).unwrap();
            let hop = NonZeroUsize::new((frame_size / 4).max(1)).unwrap();
            // StftParams cost is amortised across the bench — build once per
            // (frame_size, dt) combination at the iter-batched setup site.
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements(len as u64));
                let label = ParamLabel::new()
                    .with("len", len).with("dt", dt).with("ch", PITCH_CHANNELS)
                    .with("frame_size", frame_size)
                    .with("hop", hop.get())
                    .build();
                let id = BenchmarkId::new("Pitch-006_estimate_key", label);
                let stft_params = StftParams::new(n_fft, hop, WindowType::Hanning, true)
                    .expect("valid StftParams");
                // Re-use the dispatch macro: pass `&stft_params` by capture.
                dispatch_mono_sine!(group, id, dt, len, |a| {
                    a.estimate_key(&stft_params).ok()
                });
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Pitch-007 yin_pitch_detection — free fn kernel; f64 only; threshold axis.
// Uses the same min_tau/max_tau derivation the trait does: fmin=80, fmax=1000.
// ===========================================================================

fn bench_pitch_007_yin_pitch_detection<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("pitch");
    let sr_f = 44_100.0_f64;
    let min_tau = (sr_f / 1000.0) as usize; // 44
    let max_tau = (sr_f / 80.0) as usize;   // 551
    let dt = "f64";
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &threshold in &[0.10_f64, 0.20] {
            group.throughput(Throughput::Elements(len as u64));
            let label = ParamLabel::new()
                .with("len", len).with("dt", dt).with("ch", PITCH_CHANNELS)
                .with("threshold", format!("{threshold:.2}"))
                .build();
            let id = BenchmarkId::new("Pitch-007_yin_pitch_detection", label);
            group.bench_with_input(id, &(len, min_tau, max_tau, threshold),
                |b, &(n, mn_tau, mx_tau, thr)| {
                    b.iter_batched_ref(
                        || {
                            let audio = fixture_a440::<f64>(n, PITCH_CHANNELS);
                            let slice = audio.as_slice()
                                .expect("mono fixture is contiguous")
                                .to_vec();
                            slice
                        },
                        |v: &mut Vec<f64>| {
                            // SAFETY: v.len() == n > 0 (LENGTH_SWEEP_NO_XXXL entries are >= 256).
                            let s = unsafe {
                                non_empty_slice::NonEmptySlice::new_unchecked(v.as_slice())
                            };
                            black_box(yin_pitch_detection(s, mn_tau, mx_tau, thr));
                        },
                        BatchSize::LargeInput,
                    );
                });
        }
    }
    group.finish();
}

// ===========================================================================
// Pitch-008 autocorr_pitch_detection — free fn kernel; f64 only; raw `&[f64]`
// API (no NonEmptySlice wrapper). Same fmin/fmax-derived tau bounds.
// ===========================================================================

fn bench_pitch_008_autocorr_pitch_detection<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("pitch");
    let sr_f = 44_100.0_f64;
    let min_tau = (sr_f / 1000.0) as usize; // 44
    let max_tau = (sr_f / 80.0) as usize;   // 551
    let dt = "f64";
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        group.throughput(Throughput::Elements(len as u64));
        let label = ParamLabel::new()
            .with("len", len).with("dt", dt).with("ch", PITCH_CHANNELS)
            .build();
        let id = BenchmarkId::new("Pitch-008_autocorr_pitch_detection", label);
        group.bench_with_input(id, &(len, min_tau, max_tau),
            |b, &(n, mn_tau, mx_tau)| {
                b.iter_batched_ref(
                    || {
                        let audio = fixture_a440::<f64>(n, PITCH_CHANNELS);
                        audio.as_slice()
                            .expect("mono fixture is contiguous")
                            .to_vec()
                    },
                    |v: &mut Vec<f64>| {
                        black_box(autocorr_pitch_detection(v.as_slice(), mn_tau, mx_tau));
                    },
                    BatchSize::LargeInput,
                );
            });
    }
    group.finish();
}
