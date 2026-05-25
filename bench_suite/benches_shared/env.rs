//! Shared body for the `AudioEnvelopes` bench targets
//! (`bench_env_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/env.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `Env`
//! (lines 293-305).

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::num::NonZeroUsize;

use audio_samples::I24;
use audio_samples::operations::dynamic_range::EnvelopeFollower;
use audio_samples::operations::traits::AudioEnvelopes;
use audio_samples::operations::types::DynamicRangeMethod;

use bench_suite_common::{
    BENCH_SAMPLE_RATE_HZ, CHANNELS_DEFAULT, DTYPES_DEFAULT, LENGTH_SWEEP_FULL,
    LENGTH_SWEEP_NO_XXXL, ParamLabel, SampleSizePolicy, fixture_a440, sample_size_for,
};

// ===========================================================================
// Top-level entry point — wrappers call into this with their typed Criterion.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_env_001_amplitude_envelope(c);
    bench_env_002_rms_envelope(c);
    bench_env_003_attack_decay_envelope(c);
    bench_env_004_analytic_envelope(c);
    bench_env_005_moving_average_envelope(c);
}

// ===========================================================================
// Dispatch macro — DTYPES_DEFAULT-typed expansion for a unary 440 Hz fixture.
// ===========================================================================

macro_rules! dispatch_unary_sine {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, |$audio:ident| $body:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |$audio| { black_box($body); },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<I24>(n, ch),
                    |$audio| { black_box($body); },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i32>(n, ch),
                    |$audio| { black_box($body); },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |$audio| { black_box($body); },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

fn label_and_throughput<M: Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
    catalog_id: &str,
    n: usize,
    dt: &str,
    ch: usize,
) -> BenchmarkId {
    group.throughput(Throughput::Elements((n * ch) as u64));
    let label = ParamLabel::new()
        .with("len", n).with("dt", dt).with("ch", ch).build();
    BenchmarkId::new(catalog_id, label)
}

// ===========================================================================
// Env-001 amplitude_envelope — per-sample abs; NoFast tier (allocates
// output array, mapv pass over every input sample).
// ===========================================================================

fn bench_env_001_amplitude_envelope<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("env");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(
                    &mut group, "Env-001_amplitude_envelope", len, dt, ch,
                );
                dispatch_unary_sine!(group, id, dt, len, ch, |a| a.amplitude_envelope());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Env-002 rms_envelope — block RMS; sweep length, window, hop.
// NoFast tier.
// ===========================================================================

fn bench_env_002_rms_envelope<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("env");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        // Sweep (window, hop) — small overlapping window, larger non-overlapping.
        // hop is bounded by len so we don't degenerate to one frame at tiny lengths.
        let configs = [
            (512usize, 256usize),  // overlapping
            (1024usize, 1024usize), // non-overlapping
        ];
        for &(window, hop) in &configs {
            // At very small lengths cap window/hop to length.
            let w = window.min(len).max(1);
            let h = hop.min(len).max(1);
            let window_nz = NonZeroUsize::new(w).unwrap();
            let hop_nz = NonZeroUsize::new(h).unwrap();
            for &ch in CHANNELS_DEFAULT {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len).with("dt", dt).with("ch", ch)
                        .with("window", w).with("hop", h).build();
                    let id = BenchmarkId::new("Env-002_rms_envelope", label);
                    dispatch_unary_sine!(
                        group, id, dt, len, ch, |a| a.rms_envelope(window_nz, hop_nz)
                    );
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Env-003 attack_decay_envelope — recursive single-pole follower;
// per-sample work. NoFast tier.
// ===========================================================================

fn bench_env_003_attack_decay_envelope<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("env");
    let follower = EnvelopeFollower::new(
        5.0,
        50.0,
        BENCH_SAMPLE_RATE_HZ as f64,
        DynamicRangeMethod::Peak,
    );
    let method = DynamicRangeMethod::Peak;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(
                    &mut group, "Env-003_attack_decay_envelope", len, dt, ch,
                );
                dispatch_unary_sine!(
                    group, id, dt, len, ch,
                    |a| a.attack_decay_envelope(&follower, method)
                );
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Env-004 analytic_envelope — Hilbert transform (FFT-based, rustfft).
// NoFast + LENGTH_SWEEP_NO_XXXL (skip the 4M point — FFT dominates).
// ===========================================================================

fn bench_env_004_analytic_envelope<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("env");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(
                    &mut group, "Env-004_analytic_envelope", len, dt, ch,
                );
                dispatch_unary_sine!(group, id, dt, len, ch, |a| a.analytic_envelope());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Env-005 moving_average_envelope — rectify + box filter. NoFast tier.
// ===========================================================================

fn bench_env_005_moving_average_envelope<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("env");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        let configs = [
            (512usize, 256usize),
            (1024usize, 1024usize),
        ];
        for &(window, hop) in &configs {
            let w = window.min(len).max(1);
            let h = hop.min(len).max(1);
            let window_nz = NonZeroUsize::new(w).unwrap();
            let hop_nz = NonZeroUsize::new(h).unwrap();
            for &ch in CHANNELS_DEFAULT {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len).with("dt", dt).with("ch", ch)
                        .with("window", w).with("hop", h).build();
                    let id = BenchmarkId::new("Env-005_moving_average_envelope", label);
                    dispatch_unary_sine!(
                        group, id, dt, len, ch,
                        |a| a.moving_average_envelope(window_nz, hop_nz)
                    );
                }
            }
        }
    }
    group.finish();
}
