//! Shared body for the `AudioVoiceActivityDetection` bench targets
//! (`bench_vad_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/vad.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `VAD`
//! (lines 390-399).
//!
//! Notes:
//! - `VadMethod::Spectral` is not implemented (returns `FeatureError`); we
//!   only sweep `Energy` / `ZeroCrossing` / `Combined`. As a result the VAD
//!   impl currently has **no internal FFT** and therefore does not require
//!   the `transforms` feature.
//! - Energy threshold is swept as a `ParamLabel` axis to exercise the
//!   voiced/silence boundary on a 440 Hz sine fixture (which sits well
//!   above the noise floor at the default amplitude).

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use audio_samples::I24;
use audio_samples::operations::traits::AudioVoiceActivityDetection;
use audio_samples::operations::types::VadConfig;

use bench_suite_common::{
    CHANNELS_DEFAULT, DTYPES_DEFAULT, LENGTH_SWEEP_FULL, ParamLabel, SampleSizePolicy,
    fixture_a440, sample_size_for,
};

// ===========================================================================
// Top-level entry point — wrappers call into this with their typed Criterion.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_vad_001_voice_activity_mask(c);
    bench_vad_002_speech_regions(c);
}

// ===========================================================================
// Dispatch macro — copies `dispatch_unary_sine!` from stats.rs, but takes a
// `&VadConfig` from the routine setup so the bench body sees only the
// `audio.voice_activity_mask(&cfg)` shape.
// ===========================================================================

macro_rules! dispatch_unary_with_cfg {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, $cfg:expr, |$audio:ident, $cfg_ref:ident| $body:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        let cfg = $cfg;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |$audio| {
                        let $cfg_ref = &cfg;
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<I24>(n, ch),
                    |$audio| {
                        let $cfg_ref = &cfg;
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i32>(n, ch),
                    |$audio| {
                        let $cfg_ref = &cfg;
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |$audio| {
                        let $cfg_ref = &cfg;
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

// ===========================================================================
// VAD-001 voice_activity_mask — NoFast tier. Threshold axis sweeps the
// voiced/silence boundary: -60 dB (very sensitive) → -20 dB (strict).
// A 440 Hz sine at amplitude 1.0 has RMS ≈ -3 dB, so all three thresholds
// classify it as voiced; the threshold still drives a different branch
// fraction in the per-frame decision.
// ===========================================================================

fn bench_vad_001_voice_activity_mask<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("vad");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &threshold_db in &[-60.0_f64, -40.0, -20.0] {
                let mut cfg = VadConfig::energy_only();
                cfg.energy_threshold_db = threshold_db;
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("threshold_db", format!("{threshold_db:.1}"))
                        .build();
                    let id = BenchmarkId::new("VAD-001_voice_activity_mask", label);
                    dispatch_unary_with_cfg!(group, id, dt, len, ch, cfg, |a, c_ref| {
                        a.voice_activity_mask(c_ref).ok()
                    });
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// VAD-002 speech_regions — NoFast tier. Same threshold sweep as VAD-001
// (mask + region-grouping). The grouping step is O(n_frames) on top of
// the mask cost.
// ===========================================================================

fn bench_vad_002_speech_regions<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("vad");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &threshold_db in &[-60.0_f64, -40.0, -20.0] {
                let mut cfg = VadConfig::energy_only();
                cfg.energy_threshold_db = threshold_db;
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("threshold_db", format!("{threshold_db:.1}"))
                        .build();
                    let id = BenchmarkId::new("VAD-002_speech_regions", label);
                    dispatch_unary_with_cfg!(group, id, dt, len, ch, cfg, |a, c_ref| {
                        a.speech_regions(c_ref).ok()
                    });
                }
            }
        }
    }
    group.finish();
}
