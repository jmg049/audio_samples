//! Shared body for the `fixed_audio` bench targets
//! (`bench_fixed_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/fixed.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `Fixed`
//! (Fixed-001 .. Fixed-005). The trait methods inherited via `Deref` are
//! benched elsewhere; this module only covers the constructors and `swap`.
//!
//! Because the buffer length `N` is a const generic, the length sweep is
//! expressed as a per-length helper macro that emits one bench call per
//! `{256, 1024, 4096, 65536}` instantiation.

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use audio_samples::{
    FixedSizeAudioSamples, FixedSizeMultiChannelAudioSamples, sample_rate,
};
use non_empty_slice::NonEmptySlice;

use bench_suite_common::{ParamLabel, SampleSizePolicy, sample_size_for};

// ===========================================================================
// Top-level entry point.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_fixed_001_zeros(c);
    bench_fixed_002_from_1d(c);
    bench_fixed_003_swap(c);
    bench_fixed_004_multi_zeros(c);
    bench_fixed_005_multi_swap(c);
}

// ===========================================================================
// Helpers — sample-size tier for a given length index (matched to
// LENGTH_SWEEP_FULL positions 0, 1, 2, 4 → all in the FastSmall regime for
// these tiny constructors / swap calls).
// ===========================================================================

/// Per-length helper: emit a bench point for `FixedSizeAudioSamples<f32, N>::zeros`.
macro_rules! bench_zeros_at {
    ($group:expr, $catalog:expr, $n:expr, $idx:expr) => {{
        const N: usize = $n;
        $group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, $idx));
        $group.throughput(Throughput::Elements(N as u64));
        let label = ParamLabel::new().with("len", N).with("dt", "f32").build();
        let id = BenchmarkId::new($catalog, label);
        $group.bench_with_input(id, &N, |b, _| {
            b.iter(|| {
                let s = FixedSizeAudioSamples::<f32, N>::zeros(sample_rate!(44100));
                black_box(s);
            });
        });
    }};
}

/// Per-length helper: emit a bench point for `FixedSizeAudioSamples<f32, N>::from_1d`.
macro_rules! bench_from_1d_at {
    ($group:expr, $catalog:expr, $n:expr, $idx:expr) => {{
        const N: usize = $n;
        $group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, $idx));
        $group.throughput(Throughput::Elements(N as u64));
        let label = ParamLabel::new().with("len", N).with("dt", "f32").build();
        let id = BenchmarkId::new($catalog, label);
        $group.bench_with_input(id, &N, |b, _| {
            b.iter_batched_ref(
                || {
                    // Deterministic source buffer; the bench measures the
                    // constructor's clone-into-owned-storage path.
                    (0..N).map(|i| (i as f32) * 1e-4).collect::<Vec<f32>>()
                },
                |src: &mut Vec<f32>| {
                    // SAFETY: N ≥ 256 > 0 by construction.
                    let nes: &NonEmptySlice<f32> =
                        unsafe { NonEmptySlice::new_unchecked(&src[..]) };
                    let s = FixedSizeAudioSamples::<f32, N>::from_1d(nes, sample_rate!(44100))
                        .expect("from_1d on non-empty slice should not fail");
                    black_box(s);
                },
                BatchSize::LargeInput,
            );
        });
    }};
}

/// Per-length helper: emit a bench point for `FixedSizeAudioSamples::<f32, N>::swap`.
macro_rules! bench_swap_at {
    ($group:expr, $catalog:expr, $n:expr, $idx:expr) => {{
        const N: usize = $n;
        $group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, $idx));
        $group.throughput(Throughput::Elements(N as u64));
        let label = ParamLabel::new().with("len", N).with("dt", "f32").build();
        let id = BenchmarkId::new($catalog, label);
        $group.bench_with_input(id, &N, |b, _| {
            b.iter_batched_ref(
                || {
                    (
                        FixedSizeAudioSamples::<f32, N>::zeros(sample_rate!(44100)),
                        FixedSizeAudioSamples::<f32, N>::zeros(sample_rate!(44100)),
                    )
                },
                |pair| {
                    let (lhs, rhs) = pair;
                    lhs.swap(rhs);
                    black_box(&*lhs);
                    black_box(&*rhs);
                },
                BatchSize::LargeInput,
            );
        });
    }};
}

/// Per-length helper: emit a bench point for
/// `FixedSizeMultiChannelAudioSamples<f32, N, 2>::zeros` (stereo).
macro_rules! bench_multi_zeros_at {
    ($group:expr, $catalog:expr, $n:expr, $idx:expr) => {{
        const N: usize = $n;
        const C: usize = 2;
        $group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, $idx));
        $group.throughput(Throughput::Elements((N * C) as u64));
        let label = ParamLabel::new()
            .with("len", N)
            .with("ch", C)
            .with("dt", "f32")
            .build();
        let id = BenchmarkId::new($catalog, label);
        $group.bench_with_input(id, &N, |b, _| {
            b.iter(|| {
                let s = FixedSizeMultiChannelAudioSamples::<f32, N, C>::zeros(sample_rate!(44100));
                black_box(s);
            });
        });
    }};
}

/// Per-length helper: emit a bench point for
/// `FixedSizeMultiChannelAudioSamples::<f32, N, 2>::swap` (stereo).
macro_rules! bench_multi_swap_at {
    ($group:expr, $catalog:expr, $n:expr, $idx:expr) => {{
        const N: usize = $n;
        const C: usize = 2;
        $group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, $idx));
        $group.throughput(Throughput::Elements((N * C) as u64));
        let label = ParamLabel::new()
            .with("len", N)
            .with("ch", C)
            .with("dt", "f32")
            .build();
        let id = BenchmarkId::new($catalog, label);
        $group.bench_with_input(id, &N, |b, _| {
            b.iter_batched_ref(
                || {
                    (
                        FixedSizeMultiChannelAudioSamples::<f32, N, C>::zeros(sample_rate!(44100)),
                        FixedSizeMultiChannelAudioSamples::<f32, N, C>::zeros(sample_rate!(44100)),
                    )
                },
                |pair| {
                    let (lhs, rhs) = pair;
                    lhs.swap(rhs);
                    black_box(&*lhs);
                    black_box(&*rhs);
                },
                BatchSize::LargeInput,
            );
        });
    }};
}

// ===========================================================================
// Fixed-001 FixedSizeAudioSamples::<f32, N>::zeros
// Pure allocation cost; should track AudioSamples::zeros_mono.
// Lengths: {256, 1024, 4096, 65536}.
// ===========================================================================

fn bench_fixed_001_zeros<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("fixed");
    // Length-index mapping to LENGTH_SWEEP_FULL positions:
    //   256 → idx 0, 1024 → idx 1, 4096 → idx 2, 65_536 → idx 4.
    bench_zeros_at!(group, "Fixed-001_zeros", 256, 0);
    bench_zeros_at!(group, "Fixed-001_zeros", 1024, 1);
    bench_zeros_at!(group, "Fixed-001_zeros", 4096, 2);
    bench_zeros_at!(group, "Fixed-001_zeros", 65_536, 4);
    group.finish();
}

// ===========================================================================
// Fixed-002 FixedSizeAudioSamples::<f32, N>::from_1d
// Allocates + copies from a `NonEmptySlice`.
// ===========================================================================

fn bench_fixed_002_from_1d<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("fixed");
    bench_from_1d_at!(group, "Fixed-002_from_1d", 256, 0);
    bench_from_1d_at!(group, "Fixed-002_from_1d", 1024, 1);
    bench_from_1d_at!(group, "Fixed-002_from_1d", 4096, 2);
    bench_from_1d_at!(group, "Fixed-002_from_1d", 65_536, 4);
    group.finish();
}

// ===========================================================================
// Fixed-003 FixedSizeAudioSamples::swap
// Should be O(1) — just `mem::swap` on the inner `AudioSamples` (3-word
// move). The length axis only varies the size of the dropped buffer if
// either operand is dropped between iterations.
// ===========================================================================

fn bench_fixed_003_swap<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("fixed");
    bench_swap_at!(group, "Fixed-003_swap", 256, 0);
    bench_swap_at!(group, "Fixed-003_swap", 1024, 1);
    bench_swap_at!(group, "Fixed-003_swap", 4096, 2);
    bench_swap_at!(group, "Fixed-003_swap", 65_536, 4);
    group.finish();
}

// ===========================================================================
// Fixed-004 FixedSizeMultiChannelAudioSamples::<f32, N, 2>::zeros
// Stereo allocation; tracks AudioSamples::zeros_multi_channel.
// ===========================================================================

fn bench_fixed_004_multi_zeros<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("fixed");
    bench_multi_zeros_at!(group, "Fixed-004_multi_zeros", 256, 0);
    bench_multi_zeros_at!(group, "Fixed-004_multi_zeros", 1024, 1);
    bench_multi_zeros_at!(group, "Fixed-004_multi_zeros", 4096, 2);
    bench_multi_zeros_at!(group, "Fixed-004_multi_zeros", 65_536, 4);
    group.finish();
}

// ===========================================================================
// Fixed-005 FixedSizeMultiChannelAudioSamples::swap
// O(1) `mem::swap` — same shape as Fixed-003 but stereo.
// ===========================================================================

fn bench_fixed_005_multi_swap<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("fixed");
    bench_multi_swap_at!(group, "Fixed-005_multi_swap", 256, 0);
    bench_multi_swap_at!(group, "Fixed-005_multi_swap", 1024, 1);
    bench_multi_swap_at!(group, "Fixed-005_multi_swap", 4096, 2);
    bench_multi_swap_at!(group, "Fixed-005_multi_swap", 65_536, 4);
    group.finish();
}
