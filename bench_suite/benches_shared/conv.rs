//! Shared body for the `AudioTypeConversion` + SIMD-conversions bench
//! targets (`bench_conv_walltime`, `_instructions`, `_cycles`,
//! `_cache_misses`, `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/conv.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `Conv`
//! (lines 501..564 — Conv-001..Conv-034).
//!
//! ## (T_src, T_dst) pair subset
//!
//! The full 6×6 cross-product is 36 pairs × 4 ops = 144 bench targets,
//! which is too many for routine runs. Per CATALOG Open Q 6 we instead
//! pick a representative set that exercises:
//!
//! * **noop reference** — `i16→i16`, `f32→f32` (identity conversion,
//!   measures harness overhead + `map_into` cost).
//! * **SIMD fast paths** — `f32→i16`, `i16→f32`, `f32→i32`
//!   (the three pairs with explicit SIMD impls in `simd_conversions.rs`).
//! * **promotion / demotion** — `i16→f32`, `i16→i32`, `i32→f32`,
//!   `f32→f64`, `u8→f32` (canonical integer↔float and bit-depth shifts;
//!   `u8↔f32` exercises the unsigned-PCM mid-scale path).
//! * **I24 packed** — `i16→I24`, `I24→f32` (characterises the
//!   3-byte-packed-struct overhead vs. the integer-aligned pairs).
//!
//! That gives 11 pairs × 4 ops = 44 dense per-pair benches across
//! `to_format` / `to_type` / `cast_as` / `cast_to`. The default-impl
//! shortcut methods (`as_f32`, `as_i16`, …) get single-point spot-checks
//! since they trivially delegate to `to_format`.

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::mem::size_of;
use std::num::NonZeroU32;

use audio_samples::{
    AudioSample, AudioTypeConversion, CastInto, ConvertTo, I24, StandardSample,
    utils::generation::sine_wave,
};
use non_empty_slice::{NonEmptySlice, NonEmptyVec};

use bench_suite_common::{
    CHANNELS_DEFAULT, LENGTH_SWEEP_FULL, LENGTH_SWEEP_NO_XXXL,
    ParamLabel, SampleSizePolicy, fixture_a440, sample_size_for,
};

// ===========================================================================
// Top-level entry point.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    // Conv-001..Conv-004 — AudioTypeConversion methods, 11-pair subset.
    bench_conv_001_to_format(c);
    bench_conv_002_to_type(c);
    bench_conv_003_cast_as(c);
    bench_conv_004_cast_to(c);

    // Conv-005..Conv-012 — high-level shortcut methods (default-impl).
    bench_conv_005_cast_as_f64(c);
    bench_conv_006_as_float(c);
    bench_conv_007_as_f64(c);
    bench_conv_008_as_f32(c);
    bench_conv_009_as_i32(c);
    bench_conv_010_as_i16(c);
    bench_conv_011_as_i24(c);
    bench_conv_012_as_u8(c);

    // Conv-013..Conv-019 — per-sample scalar (Conv.Low).
    bench_conv_013_convert_to_scalar(c);
    bench_conv_014_cast_into_scalar(c);
    bench_conv_015_to_bytes(c);
    bench_conv_016_as_bytes(c);
    bench_conv_017_slice_to_bytes(c);
    bench_conv_018_as_float_scalar(c);
    bench_conv_019_clamp_to(c);

    // Conv-020 — AVX2 abs-max fast path on `f32`.
    bench_conv_020_avx2_abs_max(c);

    // Conv-021..Conv-026 — free SIMD / scalar conversion fns.
    #[cfg(feature = "simd")]
    {
        bench_conv_021_f32_to_i16_simd(c);
        bench_conv_022_i16_to_f32_simd(c);
        bench_conv_023_f32_to_i32_simd(c);
        bench_conv_024_convert_simd(c);
    }
    bench_conv_025_convert_scalar_unrolled(c);
    bench_conv_026_convert_dispatch(c);

    // Conv-027..Conv-034 — (de)interleave free fns.
    bench_conv_027_deinterleave_stereo(c);
    bench_conv_028_deinterleave_multi(c);
    bench_conv_029_interleave_stereo(c);
    bench_conv_030_interleave_multi(c);
    bench_conv_031_deinterleave_stereo_vec(c);
    bench_conv_032_deinterleave_multi_vec(c);
    bench_conv_033_interleave_stereo_vec(c);
    bench_conv_034_interleave_multi_vec(c);
}

// ===========================================================================
// (T_src, T_dst) pair subset — see module docs.
// Each pair is encoded as a `(&'static str, &'static str)` of dtype labels.
// ===========================================================================

const CONV_PAIRS: &[(&str, &str)] = &[
    // noop reference
    ("i16", "i16"),
    ("f32", "f32"),
    // SIMD fast paths
    ("f32", "i16"),
    ("i16", "f32"),
    ("f32", "i32"),
    // promotion / demotion
    ("i16", "i32"),
    ("i32", "f32"),
    ("f32", "f64"),
    ("u8",  "f32"),
    // I24 packed
    ("i16", "I24"),
    ("I24", "f32"),
];

/// Tier policy per (from, to) pair. Identity / cast paths are sub-µs scalar
/// copies; arithmetic-bearing promotions are not.
fn tier_for_pair(from: &str, to: &str) -> SampleSizePolicy {
    if from == to {
        SampleSizePolicy::FastSmall
    } else {
        SampleSizePolicy::NoFast
    }
}

// ===========================================================================
// Dispatch macros for Conv-001..Conv-004.
//
// `dispatch_to_format_ref!` — `audio.to_format::<O>()`, borrows; uses
// `iter_batched_ref`. Same shape as `cast_as`.
//
// `dispatch_to_type_consume!` — `audio.to_type::<O>()`, consumes self;
// uses `iter_batched` (no `_ref`) with a fresh moved value per iter.
// Same shape as `cast_to`.
//
// Each pair branches on the *destination* dtype string first (which
// determines the `<O>` turbofish argument), then on the *source* dtype
// (which determines the fixture builder).
// ===========================================================================

macro_rules! dispatch_pair_to_format {
    ($group:expr, $id:expr, $from:expr, $to:expr, $n:expr, $ch:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match ($from, $to) {
            ("i16", "i16") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |a| { black_box(a.to_format::<i16>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("f32", "f32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |a| { black_box(a.to_format::<f32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("f32", "i16") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |a| { black_box(a.to_format::<i16>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("i16", "f32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |a| { black_box(a.to_format::<f32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("f32", "i32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |a| { black_box(a.to_format::<i32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("i16", "i32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |a| { black_box(a.to_format::<i32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("i32", "f32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i32>(n, ch),
                    |a| { black_box(a.to_format::<f32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("f32", "f64") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |a| { black_box(a.to_format::<f64>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("u8", "f32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<u8>(n, ch),
                    |a| { black_box(a.to_format::<f32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("i16", "I24") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |a| { black_box(a.to_format::<I24>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("I24", "f32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<I24>(n, ch),
                    |a| { black_box(a.to_format::<f32>()); },
                    BatchSize::LargeInput,
                );
            }),
            (from, to) => unreachable!("CONV_PAIRS contains unhandled pair ({from}, {to})"),
        }
    }};
}

macro_rules! dispatch_pair_to_type {
    ($group:expr, $id:expr, $from:expr, $to:expr, $n:expr, $ch:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match ($from, $to) {
            ("i16", "i16") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i16>(n, ch),
                    |a| { black_box(a.to_type::<i16>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("f32", "f32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<f32>(n, ch),
                    |a| { black_box(a.to_type::<f32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("f32", "i16") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<f32>(n, ch),
                    |a| { black_box(a.to_type::<i16>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("i16", "f32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i16>(n, ch),
                    |a| { black_box(a.to_type::<f32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("f32", "i32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<f32>(n, ch),
                    |a| { black_box(a.to_type::<i32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("i16", "i32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i16>(n, ch),
                    |a| { black_box(a.to_type::<i32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("i32", "f32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i32>(n, ch),
                    |a| { black_box(a.to_type::<f32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("f32", "f64") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<f32>(n, ch),
                    |a| { black_box(a.to_type::<f64>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("u8", "f32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<u8>(n, ch),
                    |a| { black_box(a.to_type::<f32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("i16", "I24") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i16>(n, ch),
                    |a| { black_box(a.to_type::<I24>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("I24", "f32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<I24>(n, ch),
                    |a| { black_box(a.to_type::<f32>()); },
                    BatchSize::LargeInput,
                );
            }),
            (from, to) => unreachable!("CONV_PAIRS contains unhandled pair ({from}, {to})"),
        }
    }};
}

macro_rules! dispatch_pair_cast_as {
    ($group:expr, $id:expr, $from:expr, $to:expr, $n:expr, $ch:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match ($from, $to) {
            ("i16", "i16") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |a| { black_box(a.cast_as::<i16>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("f32", "f32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |a| { black_box(a.cast_as::<f32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("f32", "i16") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |a| { black_box(a.cast_as::<i16>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("i16", "f32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |a| { black_box(a.cast_as::<f32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("f32", "i32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |a| { black_box(a.cast_as::<i32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("i16", "i32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |a| { black_box(a.cast_as::<i32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("i32", "f32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i32>(n, ch),
                    |a| { black_box(a.cast_as::<f32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("f32", "f64") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |a| { black_box(a.cast_as::<f64>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("u8", "f32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<u8>(n, ch),
                    |a| { black_box(a.cast_as::<f32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("i16", "I24") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |a| { black_box(a.cast_as::<I24>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("I24", "f32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<I24>(n, ch),
                    |a| { black_box(a.cast_as::<f32>()); },
                    BatchSize::LargeInput,
                );
            }),
            (from, to) => unreachable!("CONV_PAIRS contains unhandled pair ({from}, {to})"),
        }
    }};
}

macro_rules! dispatch_pair_cast_to {
    ($group:expr, $id:expr, $from:expr, $to:expr, $n:expr, $ch:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match ($from, $to) {
            ("i16", "i16") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i16>(n, ch),
                    |a| { black_box(a.cast_to::<i16>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("f32", "f32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<f32>(n, ch),
                    |a| { black_box(a.cast_to::<f32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("f32", "i16") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<f32>(n, ch),
                    |a| { black_box(a.cast_to::<i16>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("i16", "f32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i16>(n, ch),
                    |a| { black_box(a.cast_to::<f32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("f32", "i32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<f32>(n, ch),
                    |a| { black_box(a.cast_to::<i32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("i16", "i32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i16>(n, ch),
                    |a| { black_box(a.cast_to::<i32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("i32", "f32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i32>(n, ch),
                    |a| { black_box(a.cast_to::<f32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("f32", "f64") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<f32>(n, ch),
                    |a| { black_box(a.cast_to::<f64>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("u8", "f32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<u8>(n, ch),
                    |a| { black_box(a.cast_to::<f32>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("i16", "I24") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i16>(n, ch),
                    |a| { black_box(a.cast_to::<I24>()); },
                    BatchSize::LargeInput,
                );
            }),
            ("I24", "f32") => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<I24>(n, ch),
                    |a| { black_box(a.cast_to::<f32>()); },
                    BatchSize::LargeInput,
                );
            }),
            (from, to) => unreachable!("CONV_PAIRS contains unhandled pair ({from}, {to})"),
        }
    }};
}

/// Throughput in bytes, computed from the source-type byte size. Lets the
/// harvester compute MB/s downstream.
fn bytes_throughput(from: &str, len: usize, ch: usize) -> Throughput {
    let bytes_per_sample = match from {
        "u8"  => size_of::<u8>(),
        "i16" => size_of::<i16>(),
        "I24" => 3, // I24 is packed 3 bytes
        "i32" => size_of::<i32>(),
        "f32" => size_of::<f32>(),
        "f64" => size_of::<f64>(),
        _ => unreachable!("unknown dtype label {from}"),
    };
    Throughput::Bytes((len * ch * bytes_per_sample) as u64)
}

fn pair_label(catalog_id: &str, from: &str, to: &str, n: usize, ch: usize) -> BenchmarkId {
    let label = ParamLabel::new()
        .with("len", n)
        .with("ch", ch)
        .with("from", from)
        .with("to", to)
        .build();
    BenchmarkId::new(catalog_id, label)
}

// ===========================================================================
// Conv-001 to_format — borrows; uses iter_batched_ref.
// ===========================================================================

fn bench_conv_001_to_format<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("conv");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        for &ch in CHANNELS_DEFAULT {
            for &(from, to) in CONV_PAIRS {
                group.sample_size(sample_size_for(tier_for_pair(from, to), i));
                group.throughput(bytes_throughput(from, len, ch));
                let id = pair_label("Conv-001_to_format", from, to, len, ch);
                dispatch_pair_to_format!(group, id, from, to, len, ch);
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Conv-002 to_type — consumes self; uses iter_batched (no _ref).
// ===========================================================================

fn bench_conv_002_to_type<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("conv");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        for &ch in CHANNELS_DEFAULT {
            for &(from, to) in CONV_PAIRS {
                group.sample_size(sample_size_for(tier_for_pair(from, to), i));
                group.throughput(bytes_throughput(from, len, ch));
                let id = pair_label("Conv-002_to_type", from, to, len, ch);
                dispatch_pair_to_type!(group, id, from, to, len, ch);
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Conv-003 cast_as — raw cast, borrows. Should be cheaper than to_format
// (no clamp/round/scale).
// ===========================================================================

fn bench_conv_003_cast_as<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("conv");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        for &ch in CHANNELS_DEFAULT {
            for &(from, to) in CONV_PAIRS {
                // cast_* are all sub-µs scalar copies — FastSmall regardless of pair.
                group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
                group.throughput(bytes_throughput(from, len, ch));
                let id = pair_label("Conv-003_cast_as", from, to, len, ch);
                dispatch_pair_cast_as!(group, id, from, to, len, ch);
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Conv-004 cast_to — consuming raw cast.
// ===========================================================================

fn bench_conv_004_cast_to<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("conv");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        for &ch in CHANNELS_DEFAULT {
            for &(from, to) in CONV_PAIRS {
                group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
                group.throughput(bytes_throughput(from, len, ch));
                let id = pair_label("Conv-004_cast_to", from, to, len, ch);
                dispatch_pair_cast_to!(group, id, from, to, len, ch);
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Conv-005..Conv-012 — default-impl shortcut methods. Spot-check at one
// representative length on the canonical `i16` source (or `f32` for
// `as_*` integer targets that exercise the float→int hot path).
//
// These all delegate trivially to `to_format::<O>()` (or `cast_as::<f64>`
// for `cast_as_f64`), so a single-point AlwaysDefault bench is enough to
// catch a regression in the wrapper itself.
// ===========================================================================

const SHORTCUT_LEN: usize = 65_536;
const SHORTCUT_CH:  usize = 1;

fn bench_conv_005_cast_as_f64<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("conv");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    group.throughput(bytes_throughput("i16", SHORTCUT_LEN, SHORTCUT_CH));
    let id = pair_label("Conv-005_cast_as_f64", "i16", "f64", SHORTCUT_LEN, SHORTCUT_CH);
    group.bench_with_input(id, &(SHORTCUT_LEN, SHORTCUT_CH), |b, &(n, ch)| {
        b.iter_batched_ref(
            || fixture_a440::<i16>(n, ch),
            |a| { black_box(a.cast_as_f64()); },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn bench_conv_006_as_float<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("conv");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    group.throughput(bytes_throughput("i16", SHORTCUT_LEN, SHORTCUT_CH));
    let id = pair_label("Conv-006_as_float", "i16", "f64", SHORTCUT_LEN, SHORTCUT_CH);
    group.bench_with_input(id, &(SHORTCUT_LEN, SHORTCUT_CH), |b, &(n, ch)| {
        b.iter_batched_ref(
            || fixture_a440::<i16>(n, ch),
            |a| { black_box(a.as_float()); },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn bench_conv_007_as_f64<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("conv");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    group.throughput(bytes_throughput("i16", SHORTCUT_LEN, SHORTCUT_CH));
    let id = pair_label("Conv-007_as_f64", "i16", "f64", SHORTCUT_LEN, SHORTCUT_CH);
    group.bench_with_input(id, &(SHORTCUT_LEN, SHORTCUT_CH), |b, &(n, ch)| {
        b.iter_batched_ref(
            || fixture_a440::<i16>(n, ch),
            |a| { black_box(a.as_f64()); },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn bench_conv_008_as_f32<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("conv");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    group.throughput(bytes_throughput("i16", SHORTCUT_LEN, SHORTCUT_CH));
    let id = pair_label("Conv-008_as_f32", "i16", "f32", SHORTCUT_LEN, SHORTCUT_CH);
    group.bench_with_input(id, &(SHORTCUT_LEN, SHORTCUT_CH), |b, &(n, ch)| {
        b.iter_batched_ref(
            || fixture_a440::<i16>(n, ch),
            |a| { black_box(a.as_f32()); },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn bench_conv_009_as_i32<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("conv");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    group.throughput(bytes_throughput("f32", SHORTCUT_LEN, SHORTCUT_CH));
    let id = pair_label("Conv-009_as_i32", "f32", "i32", SHORTCUT_LEN, SHORTCUT_CH);
    group.bench_with_input(id, &(SHORTCUT_LEN, SHORTCUT_CH), |b, &(n, ch)| {
        b.iter_batched_ref(
            || fixture_a440::<f32>(n, ch),
            |a| { black_box(a.as_i32()); },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn bench_conv_010_as_i16<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("conv");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    group.throughput(bytes_throughput("f32", SHORTCUT_LEN, SHORTCUT_CH));
    let id = pair_label("Conv-010_as_i16", "f32", "i16", SHORTCUT_LEN, SHORTCUT_CH);
    group.bench_with_input(id, &(SHORTCUT_LEN, SHORTCUT_CH), |b, &(n, ch)| {
        b.iter_batched_ref(
            || fixture_a440::<f32>(n, ch),
            |a| { black_box(a.as_i16()); },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn bench_conv_011_as_i24<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("conv");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    group.throughput(bytes_throughput("f32", SHORTCUT_LEN, SHORTCUT_CH));
    let id = pair_label("Conv-011_as_i24", "f32", "I24", SHORTCUT_LEN, SHORTCUT_CH);
    group.bench_with_input(id, &(SHORTCUT_LEN, SHORTCUT_CH), |b, &(n, ch)| {
        b.iter_batched_ref(
            || fixture_a440::<f32>(n, ch),
            |a| { black_box(a.as_i24()); },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn bench_conv_012_as_u8<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("conv");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    group.throughput(bytes_throughput("f32", SHORTCUT_LEN, SHORTCUT_CH));
    let id = pair_label("Conv-012_as_u8", "f32", "u8", SHORTCUT_LEN, SHORTCUT_CH);
    group.bench_with_input(id, &(SHORTCUT_LEN, SHORTCUT_CH), |b, &(n, ch)| {
        b.iter_batched_ref(
            || fixture_a440::<f32>(n, ch),
            |a| { black_box(a.as_u8()); },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

// ===========================================================================
// Conv-013..Conv-019 — per-sample microbenches.
//
// The per-sample functions are intended to be inlined into tight loops; we
// bench them in a `for` loop over a length-`n` slice fixture to reflect
// real-world use. Length sweep is restricted to LENGTH_SWEEP_NO_XXXL —
// scalar microbench at 4M points gains nothing.
// ===========================================================================

/// Build a flat `Vec<T>` of `n` deterministic samples, drawn from the same
/// 440 Hz sine fixture used elsewhere. Used by the per-sample microbenches
/// and by the SIMD free-fn benches that need raw slices.
fn flat_sine_vec<T>(n: usize) -> Vec<T>
where
    T: StandardSample + 'static,
{
    let sr = NonZeroU32::new(44_100).unwrap();
    let dur = std::time::Duration::from_secs_f64(n as f64 / 44_100.0);
    let audio = sine_wave::<T>(440.0, dur, sr, 1.0);
    // Mono sine_wave is contiguous by construction.
    audio.as_slice().expect("mono sine fixture should be contiguous").to_vec()
}

/// Convert a `Vec<T>` into a `NonEmptyVec<T>`, panicking if empty. The error
/// type `EmptyVec<T>` doesn't impl Debug, so use a `match` for the panic.
fn into_nev<T>(v: Vec<T>) -> NonEmptyVec<T> {
    match NonEmptyVec::try_from(v) {
        Ok(nev) => nev,
        Err(_) => panic!("bench fixture must be non-empty"),
    }
}

/// Conv-013: per-sample `T::convert_to::<U>()` inside a tight for-loop.
/// Subset: `f32→i16`, `i16→f32`, `f32→i32`, `i16→i32`.
fn bench_conv_013_convert_to_scalar<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("conv");
    let pairs: &[(&str, &str)] = &[
        ("f32", "i16"), ("i16", "f32"), ("f32", "i32"), ("i16", "i32"),
    ];
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &(from, to) in pairs {
            group.throughput(bytes_throughput(from, len, 1));
            let id = pair_label("Conv-013_convert_to_scalar", from, to, len, 1);
            match (from, to) {
                ("f32", "i16") => group.bench_with_input(id, &len, |b, &n| {
                    let src: Vec<f32> = flat_sine_vec::<f32>(n);
                    b.iter(|| {
                        let mut acc: i16 = 0;
                        for &s in &src {
                            acc = acc.wrapping_add(<f32 as ConvertTo<i16>>::convert_to(s));
                        }
                        black_box(acc);
                    });
                }),
                ("i16", "f32") => group.bench_with_input(id, &len, |b, &n| {
                    let src: Vec<i16> = flat_sine_vec::<i16>(n);
                    b.iter(|| {
                        let mut acc: f32 = 0.0;
                        for &s in &src {
                            acc += <i16 as ConvertTo<f32>>::convert_to(s);
                        }
                        black_box(acc);
                    });
                }),
                ("f32", "i32") => group.bench_with_input(id, &len, |b, &n| {
                    let src: Vec<f32> = flat_sine_vec::<f32>(n);
                    b.iter(|| {
                        let mut acc: i32 = 0;
                        for &s in &src {
                            acc = acc.wrapping_add(<f32 as ConvertTo<i32>>::convert_to(s));
                        }
                        black_box(acc);
                    });
                }),
                ("i16", "i32") => group.bench_with_input(id, &len, |b, &n| {
                    let src: Vec<i16> = flat_sine_vec::<i16>(n);
                    b.iter(|| {
                        let mut acc: i32 = 0;
                        for &s in &src {
                            acc = acc.wrapping_add(<i16 as ConvertTo<i32>>::convert_to(s));
                        }
                        black_box(acc);
                    });
                }),
                _ => unreachable!(),
            };
        }
    }
    group.finish();
}

/// Conv-014: per-sample `T::cast_into::<U>()` (raw cast).
fn bench_conv_014_cast_into_scalar<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("conv");
    let pairs: &[(&str, &str)] = &[
        ("f32", "i16"), ("i16", "f32"), ("f32", "i32"), ("i16", "i32"),
    ];
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &(from, to) in pairs {
            group.throughput(bytes_throughput(from, len, 1));
            let id = pair_label("Conv-014_cast_into_scalar", from, to, len, 1);
            match (from, to) {
                ("f32", "i16") => group.bench_with_input(id, &len, |b, &n| {
                    let src: Vec<f32> = flat_sine_vec::<f32>(n);
                    b.iter(|| {
                        let mut acc: i16 = 0;
                        for &s in &src {
                            acc = acc.wrapping_add(<f32 as CastInto<i16>>::cast_into(s));
                        }
                        black_box(acc);
                    });
                }),
                ("i16", "f32") => group.bench_with_input(id, &len, |b, &n| {
                    let src: Vec<i16> = flat_sine_vec::<i16>(n);
                    b.iter(|| {
                        let mut acc: f32 = 0.0;
                        for &s in &src {
                            acc += <i16 as CastInto<f32>>::cast_into(s);
                        }
                        black_box(acc);
                    });
                }),
                ("f32", "i32") => group.bench_with_input(id, &len, |b, &n| {
                    let src: Vec<f32> = flat_sine_vec::<f32>(n);
                    b.iter(|| {
                        let mut acc: i32 = 0;
                        for &s in &src {
                            acc = acc.wrapping_add(<f32 as CastInto<i32>>::cast_into(s));
                        }
                        black_box(acc);
                    });
                }),
                ("i16", "i32") => group.bench_with_input(id, &len, |b, &n| {
                    let src: Vec<i16> = flat_sine_vec::<i16>(n);
                    b.iter(|| {
                        let mut acc: i32 = 0;
                        for &s in &src {
                            acc = acc.wrapping_add(<i16 as CastInto<i32>>::cast_into(s));
                        }
                        black_box(acc);
                    });
                }),
                _ => unreachable!(),
            };
        }
    }
    group.finish();
}

/// Conv-015: `T::to_bytes()` — allocates a `Vec<u8>` per call. Expected to
/// be allocation-bound. Single dtype × single length spot-check.
fn bench_conv_015_to_bytes<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("conv");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    // Single-sample call repeated; not a slice operation.
    group.throughput(Throughput::Bytes(size_of::<i16>() as u64));
    let id = BenchmarkId::new(
        "Conv-015_to_bytes",
        ParamLabel::new().with("dt", "i16").build(),
    );
    group.bench_function(id, |b| {
        let s: i16 = 0x1234;
        b.iter(|| {
            black_box(s.to_bytes());
        });
    });
    group.finish();
}

/// Conv-016: `T::as_bytes::<N>()` — stack-allocated alternative; should be
/// dramatically cheaper than `to_bytes`.
fn bench_conv_016_as_bytes<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("conv");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    group.throughput(Throughput::Bytes(size_of::<i16>() as u64));
    let id = BenchmarkId::new(
        "Conv-016_as_bytes",
        ParamLabel::new().with("dt", "i16").build(),
    );
    group.bench_function(id, |b| {
        let s: i16 = 0x1234;
        b.iter(|| {
            let arr: [u8; 2] = s.as_bytes::<2>();
            black_box(arr);
        });
    });
    group.finish();
}

/// Conv-017: `T::slice_to_bytes(&[T])` — bulk bytemuck reinterpretation +
/// allocation. Bench at NoFast tier with the full length sweep over a
/// couple of representative dtypes.
fn bench_conv_017_slice_to_bytes<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("conv");
    let dtypes: &[&str] = &["i16", "i32", "f32"];
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in dtypes {
            group.throughput(bytes_throughput(dt, len, 1));
            let id = BenchmarkId::new(
                "Conv-017_slice_to_bytes",
                ParamLabel::new().with("len", len).with("ch", 1usize).with("dt", dt).build(),
            );
            match dt {
                "i16" => group.bench_with_input(id, &len, |b, &n| {
                    let src: Vec<i16> = flat_sine_vec::<i16>(n);
                    b.iter(|| { black_box(i16::slice_to_bytes(&src)); });
                }),
                "i32" => group.bench_with_input(id, &len, |b, &n| {
                    let src: Vec<i32> = flat_sine_vec::<i32>(n);
                    b.iter(|| { black_box(i32::slice_to_bytes(&src)); });
                }),
                "f32" => group.bench_with_input(id, &len, |b, &n| {
                    let src: Vec<f32> = flat_sine_vec::<f32>(n);
                    b.iter(|| { black_box(f32::slice_to_bytes(&src)); });
                }),
                _ => unreachable!(),
            };
        }
    }
    group.finish();
}

/// Conv-018: `T::as_float() -> f64` — scalar `cast_into::<f64>` shortcut.
fn bench_conv_018_as_float_scalar<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("conv");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    let len = 65_536usize;
    group.throughput(bytes_throughput("i16", len, 1));
    let id = BenchmarkId::new(
        "Conv-018_as_float_scalar",
        ParamLabel::new().with("len", len).with("ch", 1usize).with("dt", "i16").build(),
    );
    group.bench_with_input(id, &len, |b, &n| {
        let src: Vec<i16> = flat_sine_vec::<i16>(n);
        b.iter(|| {
            let mut acc = 0.0_f64;
            for &s in &src {
                acc += AudioSample::as_float(s);
            }
            black_box(acc);
        });
    });
    group.finish();
}

/// Conv-019: `T::clamp_to(min, max)` — branchless on float types, branchy
/// on integer types. Bench tight-loop scalar over a few dtypes.
fn bench_conv_019_clamp_to<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("conv");
    let dtypes: &[&str] = &["i16", "i32", "f32", "f64"];
    let len = 65_536usize;
    group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, 4));
    for &dt in dtypes {
        group.throughput(bytes_throughput(dt, len, 1));
        let id = BenchmarkId::new(
            "Conv-019_clamp_to",
            ParamLabel::new().with("len", len).with("ch", 1usize).with("dt", dt).build(),
        );
        match dt {
            "i16" => group.bench_with_input(id, &len, |b, &n| {
                let src: Vec<i16> = flat_sine_vec::<i16>(n);
                b.iter(|| {
                    let mut acc: i16 = 0;
                    for &s in &src {
                        acc = acc.wrapping_add(s.clamp_to(-1000, 1000));
                    }
                    black_box(acc);
                });
            }),
            "i32" => group.bench_with_input(id, &len, |b, &n| {
                let src: Vec<i32> = flat_sine_vec::<i32>(n);
                b.iter(|| {
                    let mut acc: i32 = 0;
                    for &s in &src {
                        acc = acc.wrapping_add(s.clamp_to(-1_000_000, 1_000_000));
                    }
                    black_box(acc);
                });
            }),
            "f32" => group.bench_with_input(id, &len, |b, &n| {
                let src: Vec<f32> = flat_sine_vec::<f32>(n);
                b.iter(|| {
                    let mut acc: f32 = 0.0;
                    for &s in &src {
                        acc += s.clamp_to(-0.5, 0.5);
                    }
                    black_box(acc);
                });
            }),
            "f64" => group.bench_with_input(id, &len, |b, &n| {
                let src: Vec<f64> = flat_sine_vec::<f64>(n);
                b.iter(|| {
                    let mut acc: f64 = 0.0;
                    for &s in &src {
                        acc += s.clamp_to(-0.5, 0.5);
                    }
                    black_box(acc);
                });
            }),
            _ => unreachable!(),
        };
    }
    group.finish();
}

// ===========================================================================
// Conv-020 — `f32::avx2_abs_max(&[f32])`.
//
// The trait default returns `None`; only `f32` on x86_64+AVX2 returns a
// `Some(_)`. On other hosts this measures the cheap-`None`-return cost.
// ===========================================================================

fn bench_conv_020_avx2_abs_max<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("conv");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        group.throughput(Throughput::Bytes((len * size_of::<f32>()) as u64));
        let id = BenchmarkId::new(
            "Conv-020_avx2_abs_max",
            ParamLabel::new().with("len", len).with("ch", 1usize).with("dt", "f32").build(),
        );
        group.bench_with_input(id, &len, |b, &n| {
            let src: Vec<f32> = flat_sine_vec::<f32>(n);
            b.iter(|| {
                black_box(<f32 as AudioSample>::avx2_abs_max(&src));
            });
        });
    }
    group.finish();
}

// ===========================================================================
// Conv-021..Conv-026 — free SIMD / scalar conversion fns.
//
// All take pre-allocated `&mut [U]` output; we hoist allocation out of the
// timed routine via `iter_batched_ref` with a `(Vec<T>, Vec<U>)` setup.
// ===========================================================================

#[cfg(feature = "simd")]
fn bench_conv_021_f32_to_i16_simd<M: Measurement>(c: &mut Criterion<M>) {
    use audio_samples::simd_conversions::convert_f32_to_i16_simd;
    let mut group = c.benchmark_group("conv");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        group.throughput(bytes_throughput("f32", len, 1));
        let id = BenchmarkId::new(
            "Conv-021_convert_f32_to_i16_simd",
            ParamLabel::new()
                .with("len", len).with("ch", 1usize)
                .with("from", "f32").with("to", "i16")
                .build(),
        );
        group.bench_with_input(id, &len, |b, &n| {
            b.iter_batched_ref(
                || (flat_sine_vec::<f32>(n), vec![0_i16; n]),
                |(input, output)| {
                    convert_f32_to_i16_simd(input, output).expect("conversion succeeds");
                    black_box(&output);
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

#[cfg(feature = "simd")]
fn bench_conv_022_i16_to_f32_simd<M: Measurement>(c: &mut Criterion<M>) {
    use audio_samples::simd_conversions::convert_i16_to_f32_simd;
    let mut group = c.benchmark_group("conv");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        group.throughput(bytes_throughput("i16", len, 1));
        let id = BenchmarkId::new(
            "Conv-022_convert_i16_to_f32_simd",
            ParamLabel::new()
                .with("len", len).with("ch", 1usize)
                .with("from", "i16").with("to", "f32")
                .build(),
        );
        group.bench_with_input(id, &len, |b, &n| {
            b.iter_batched_ref(
                || (flat_sine_vec::<i16>(n), vec![0.0_f32; n]),
                |(input, output)| {
                    convert_i16_to_f32_simd(input, output).expect("conversion succeeds");
                    black_box(&output);
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

#[cfg(feature = "simd")]
fn bench_conv_023_f32_to_i32_simd<M: Measurement>(c: &mut Criterion<M>) {
    use audio_samples::simd_conversions::convert_f32_to_i32_simd;
    let mut group = c.benchmark_group("conv");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        group.throughput(bytes_throughput("f32", len, 1));
        let id = BenchmarkId::new(
            "Conv-023_convert_f32_to_i32_simd",
            ParamLabel::new()
                .with("len", len).with("ch", 1usize)
                .with("from", "f32").with("to", "i32")
                .build(),
        );
        group.bench_with_input(id, &len, |b, &n| {
            b.iter_batched_ref(
                || (flat_sine_vec::<f32>(n), vec![0_i32; n]),
                |(input, output)| {
                    convert_f32_to_i32_simd(input, output).expect("conversion succeeds");
                    black_box(&output);
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

/// Conv-024 — `convert_simd<T, U>` dispatcher.  Bench at all four
/// `CONV_PAIRS` entries that have a specialised SIMD path, plus one pair
/// that falls through to the scalar fallback (`i16→i32`) so the dispatch
/// overhead vs. scalar baseline is visible.
#[cfg(feature = "simd")]
fn bench_conv_024_convert_simd<M: Measurement>(c: &mut Criterion<M>) {
    use audio_samples::simd_conversions::convert_simd;
    let mut group = c.benchmark_group("conv");
    let pairs: &[(&str, &str)] = &[
        ("f32", "i16"), ("i16", "f32"), ("f32", "i32"), ("i16", "i32"),
    ];
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &(from, to) in pairs {
            group.throughput(bytes_throughput(from, len, 1));
            let id = pair_label("Conv-024_convert_simd", from, to, len, 1);
            match (from, to) {
                ("f32", "i16") => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || (flat_sine_vec::<f32>(n), vec![0_i16; n]),
                        |(input, output)| {
                            convert_simd::<f32, i16>(input, output).expect("ok");
                            black_box(&output);
                        },
                        BatchSize::LargeInput,
                    );
                }),
                ("i16", "f32") => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || (flat_sine_vec::<i16>(n), vec![0.0_f32; n]),
                        |(input, output)| {
                            convert_simd::<i16, f32>(input, output).expect("ok");
                            black_box(&output);
                        },
                        BatchSize::LargeInput,
                    );
                }),
                ("f32", "i32") => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || (flat_sine_vec::<f32>(n), vec![0_i32; n]),
                        |(input, output)| {
                            convert_simd::<f32, i32>(input, output).expect("ok");
                            black_box(&output);
                        },
                        BatchSize::LargeInput,
                    );
                }),
                ("i16", "i32") => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || (flat_sine_vec::<i16>(n), vec![0_i32; n]),
                        |(input, output)| {
                            convert_simd::<i16, i32>(input, output).expect("ok");
                            black_box(&output);
                        },
                        BatchSize::LargeInput,
                    );
                }),
                _ => unreachable!(),
            };
        }
    }
    group.finish();
}

/// Conv-025 — `convert_scalar_unrolled<T, U>` — manually 4-unrolled scalar
/// baseline. Always available regardless of `simd` feature.
fn bench_conv_025_convert_scalar_unrolled<M: Measurement>(c: &mut Criterion<M>) {
    use audio_samples::simd_conversions::convert_scalar_unrolled;
    let mut group = c.benchmark_group("conv");
    let pairs: &[(&str, &str)] = &[
        ("f32", "i16"), ("i16", "f32"), ("f32", "i32"), ("i16", "i32"),
    ];
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &(from, to) in pairs {
            group.throughput(bytes_throughput(from, len, 1));
            let id = pair_label("Conv-025_convert_scalar_unrolled", from, to, len, 1);
            match (from, to) {
                ("f32", "i16") => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || (flat_sine_vec::<f32>(n), vec![0_i16; n]),
                        |(input, output)| {
                            convert_scalar_unrolled::<f32, i16>(input, output).expect("ok");
                            black_box(&output);
                        },
                        BatchSize::LargeInput,
                    );
                }),
                ("i16", "f32") => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || (flat_sine_vec::<i16>(n), vec![0.0_f32; n]),
                        |(input, output)| {
                            convert_scalar_unrolled::<i16, f32>(input, output).expect("ok");
                            black_box(&output);
                        },
                        BatchSize::LargeInput,
                    );
                }),
                ("f32", "i32") => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || (flat_sine_vec::<f32>(n), vec![0_i32; n]),
                        |(input, output)| {
                            convert_scalar_unrolled::<f32, i32>(input, output).expect("ok");
                            black_box(&output);
                        },
                        BatchSize::LargeInput,
                    );
                }),
                ("i16", "i32") => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || (flat_sine_vec::<i16>(n), vec![0_i32; n]),
                        |(input, output)| {
                            convert_scalar_unrolled::<i16, i32>(input, output).expect("ok");
                            black_box(&output);
                        },
                        BatchSize::LargeInput,
                    );
                }),
                _ => unreachable!(),
            };
        }
    }
    group.finish();
}

/// Conv-026 — `convert<T, U>` top-level dispatcher (SIMD-if-enabled).
/// Same pairs as Conv-024/025 for direct comparison.
fn bench_conv_026_convert_dispatch<M: Measurement>(c: &mut Criterion<M>) {
    use audio_samples::simd_conversions::convert;
    let mut group = c.benchmark_group("conv");
    let pairs: &[(&str, &str)] = &[
        ("f32", "i16"), ("i16", "f32"), ("f32", "i32"), ("i16", "i32"),
    ];
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &(from, to) in pairs {
            group.throughput(bytes_throughput(from, len, 1));
            let id = pair_label("Conv-026_convert_dispatch", from, to, len, 1);
            match (from, to) {
                ("f32", "i16") => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || (flat_sine_vec::<f32>(n), vec![0_i16; n]),
                        |(input, output)| {
                            convert::<f32, i16>(input, output).expect("ok");
                            black_box(&output);
                        },
                        BatchSize::LargeInput,
                    );
                }),
                ("i16", "f32") => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || (flat_sine_vec::<i16>(n), vec![0.0_f32; n]),
                        |(input, output)| {
                            convert::<i16, f32>(input, output).expect("ok");
                            black_box(&output);
                        },
                        BatchSize::LargeInput,
                    );
                }),
                ("f32", "i32") => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || (flat_sine_vec::<f32>(n), vec![0_i32; n]),
                        |(input, output)| {
                            convert::<f32, i32>(input, output).expect("ok");
                            black_box(&output);
                        },
                        BatchSize::LargeInput,
                    );
                }),
                ("i16", "i32") => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || (flat_sine_vec::<i16>(n), vec![0_i32; n]),
                        |(input, output)| {
                            convert::<i16, i32>(input, output).expect("ok");
                            black_box(&output);
                        },
                        BatchSize::LargeInput,
                    );
                }),
                _ => unreachable!(),
            };
        }
    }
    group.finish();
}

// ===========================================================================
// Conv-027..Conv-034 — (de)interleave free fns.
//
// `deinterleave_stereo` / `interleave_stereo` take plain `&[T]`.
// `deinterleave_multi` / `interleave_multi` take `NonEmptySlice<T>`.
//
// Only `f32` has a SIMD specialisation when `simd` is enabled — other dtypes
// fall through to the scalar path. We bench a representative subset of dtypes
// so the scalar/SIMD divergence is observable in the harvester output.
// ===========================================================================

fn deinterleave_dtypes() -> &'static [&'static str] {
    &["i16", "i32", "f32"]
}

fn bench_conv_027_deinterleave_stereo<M: Measurement>(c: &mut Criterion<M>) {
    use audio_samples::simd_conversions::deinterleave_stereo;
    let mut group = c.benchmark_group("conv");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &dt in deinterleave_dtypes() {
            // Total interleaved length = len * 2 (per-channel × 2 channels).
            let total = len * 2;
            group.throughput(bytes_throughput(dt, total, 1));
            let id = BenchmarkId::new(
                "Conv-027_deinterleave_stereo",
                ParamLabel::new()
                    .with("len", len).with("ch", 2usize).with("dt", dt)
                    .build(),
            );
            match dt {
                "i16" => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || (flat_sine_vec::<i16>(n * 2), vec![0_i16; n * 2]),
                        |(input, output)| {
                            deinterleave_stereo::<i16>(input, output).expect("ok");
                            black_box(&output);
                        },
                        BatchSize::LargeInput,
                    );
                }),
                "i32" => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || (flat_sine_vec::<i32>(n * 2), vec![0_i32; n * 2]),
                        |(input, output)| {
                            deinterleave_stereo::<i32>(input, output).expect("ok");
                            black_box(&output);
                        },
                        BatchSize::LargeInput,
                    );
                }),
                "f32" => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || (flat_sine_vec::<f32>(n * 2), vec![0.0_f32; n * 2]),
                        |(input, output)| {
                            deinterleave_stereo::<f32>(input, output).expect("ok");
                            black_box(&output);
                        },
                        BatchSize::LargeInput,
                    );
                }),
                _ => unreachable!(),
            };
        }
    }
    group.finish();
}

const MULTI_CHANNELS: &[usize] = &[4, 6]; // quad and 5.1
const STEREO_CHANNELS: usize = 2;

fn bench_conv_028_deinterleave_multi<M: Measurement>(c: &mut Criterion<M>) {
    use audio_samples::simd_conversions::deinterleave_multi;
    let mut group = c.benchmark_group("conv");
    // Cap at NO_XXXL: 4M × 6ch is ~96 MB per fixture, prohibitive for batched-ref.
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in MULTI_CHANNELS {
            for &dt in deinterleave_dtypes() {
                let total = len * ch;
                group.throughput(bytes_throughput(dt, total, 1));
                let id = BenchmarkId::new(
                    "Conv-028_deinterleave_multi",
                    ParamLabel::new()
                        .with("len", len).with("ch", ch).with("dt", dt)
                        .build(),
                );
                let nzu_ch = NonZeroU32::new(ch as u32).unwrap();
                match dt {
                    "i16" => group.bench_with_input(id, &(len, ch), |b, &(n, c_)| {
                        b.iter_batched_ref(
                            || {
                                let v = flat_sine_vec::<i16>(n * c_);
                                let o = vec![0_i16; n * c_];
                                (
                                    into_nev(v),
                                    into_nev(o),
                                )
                            },
                            |(input, output)| {
                                let inp: &NonEmptySlice<i16> = input.as_non_empty_slice();
                                let outp: &mut NonEmptySlice<i16> = output.as_non_empty_mut_slice();
                                deinterleave_multi::<i16>(inp, outp, nzu_ch).expect("ok");
                                black_box(&output);
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "i32" => group.bench_with_input(id, &(len, ch), |b, &(n, c_)| {
                        b.iter_batched_ref(
                            || {
                                let v = flat_sine_vec::<i32>(n * c_);
                                let o = vec![0_i32; n * c_];
                                (
                                    into_nev(v),
                                    into_nev(o),
                                )
                            },
                            |(input, output)| {
                                let inp: &NonEmptySlice<i32> = input.as_non_empty_slice();
                                let outp: &mut NonEmptySlice<i32> = output.as_non_empty_mut_slice();
                                deinterleave_multi::<i32>(inp, outp, nzu_ch).expect("ok");
                                black_box(&output);
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "f32" => group.bench_with_input(id, &(len, ch), |b, &(n, c_)| {
                        b.iter_batched_ref(
                            || {
                                let v = flat_sine_vec::<f32>(n * c_);
                                let o = vec![0.0_f32; n * c_];
                                (
                                    into_nev(v),
                                    into_nev(o),
                                )
                            },
                            |(input, output)| {
                                let inp: &NonEmptySlice<f32> = input.as_non_empty_slice();
                                let outp: &mut NonEmptySlice<f32> = output.as_non_empty_mut_slice();
                                deinterleave_multi::<f32>(inp, outp, nzu_ch).expect("ok");
                                black_box(&output);
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    _ => unreachable!(),
                };
            }
        }
    }
    group.finish();
}

fn bench_conv_029_interleave_stereo<M: Measurement>(c: &mut Criterion<M>) {
    use audio_samples::simd_conversions::interleave_stereo;
    let mut group = c.benchmark_group("conv");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &dt in deinterleave_dtypes() {
            let total = len * STEREO_CHANNELS;
            group.throughput(bytes_throughput(dt, total, 1));
            let id = BenchmarkId::new(
                "Conv-029_interleave_stereo",
                ParamLabel::new()
                    .with("len", len).with("ch", STEREO_CHANNELS).with("dt", dt)
                    .build(),
            );
            match dt {
                "i16" => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || (flat_sine_vec::<i16>(n * 2), vec![0_i16; n * 2]),
                        |(input, output)| {
                            interleave_stereo::<i16>(input, output).expect("ok");
                            black_box(&output);
                        },
                        BatchSize::LargeInput,
                    );
                }),
                "i32" => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || (flat_sine_vec::<i32>(n * 2), vec![0_i32; n * 2]),
                        |(input, output)| {
                            interleave_stereo::<i32>(input, output).expect("ok");
                            black_box(&output);
                        },
                        BatchSize::LargeInput,
                    );
                }),
                "f32" => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || (flat_sine_vec::<f32>(n * 2), vec![0.0_f32; n * 2]),
                        |(input, output)| {
                            interleave_stereo::<f32>(input, output).expect("ok");
                            black_box(&output);
                        },
                        BatchSize::LargeInput,
                    );
                }),
                _ => unreachable!(),
            };
        }
    }
    group.finish();
}

fn bench_conv_030_interleave_multi<M: Measurement>(c: &mut Criterion<M>) {
    use audio_samples::simd_conversions::interleave_multi;
    let mut group = c.benchmark_group("conv");
    // Cap at NO_XXXL: 4M × 6ch is ~96 MB per fixture, prohibitive for batched-ref.
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in MULTI_CHANNELS {
            for &dt in deinterleave_dtypes() {
                let total = len * ch;
                group.throughput(bytes_throughput(dt, total, 1));
                let id = BenchmarkId::new(
                    "Conv-030_interleave_multi",
                    ParamLabel::new()
                        .with("len", len).with("ch", ch).with("dt", dt)
                        .build(),
                );
                let nzu_ch = NonZeroU32::new(ch as u32).unwrap();
                match dt {
                    "i16" => group.bench_with_input(id, &(len, ch), |b, &(n, c_)| {
                        b.iter_batched_ref(
                            || {
                                let v = flat_sine_vec::<i16>(n * c_);
                                let o = vec![0_i16; n * c_];
                                (
                                    into_nev(v),
                                    into_nev(o),
                                )
                            },
                            |(input, output)| {
                                let inp: &NonEmptySlice<i16> = input.as_non_empty_slice();
                                let outp: &mut NonEmptySlice<i16> = output.as_non_empty_mut_slice();
                                interleave_multi::<i16>(inp, outp, nzu_ch).expect("ok");
                                black_box(&output);
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "i32" => group.bench_with_input(id, &(len, ch), |b, &(n, c_)| {
                        b.iter_batched_ref(
                            || {
                                let v = flat_sine_vec::<i32>(n * c_);
                                let o = vec![0_i32; n * c_];
                                (
                                    into_nev(v),
                                    into_nev(o),
                                )
                            },
                            |(input, output)| {
                                let inp: &NonEmptySlice<i32> = input.as_non_empty_slice();
                                let outp: &mut NonEmptySlice<i32> = output.as_non_empty_mut_slice();
                                interleave_multi::<i32>(inp, outp, nzu_ch).expect("ok");
                                black_box(&output);
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "f32" => group.bench_with_input(id, &(len, ch), |b, &(n, c_)| {
                        b.iter_batched_ref(
                            || {
                                let v = flat_sine_vec::<f32>(n * c_);
                                let o = vec![0.0_f32; n * c_];
                                (
                                    into_nev(v),
                                    into_nev(o),
                                )
                            },
                            |(input, output)| {
                                let inp: &NonEmptySlice<f32> = input.as_non_empty_slice();
                                let outp: &mut NonEmptySlice<f32> = output.as_non_empty_mut_slice();
                                interleave_multi::<f32>(inp, outp, nzu_ch).expect("ok");
                                black_box(&output);
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    _ => unreachable!(),
                };
            }
        }
    }
    group.finish();
}

// Allocating variants — cap length at the NO_XXXL ceiling: at 4M samples
// the alloc-bound bench dominates and we don't gain new information.

fn bench_conv_031_deinterleave_stereo_vec<M: Measurement>(c: &mut Criterion<M>) {
    use audio_samples::simd_conversions::deinterleave_stereo_vec;
    let mut group = c.benchmark_group("conv");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in deinterleave_dtypes() {
            let total = len * STEREO_CHANNELS;
            group.throughput(bytes_throughput(dt, total, 1));
            let id = BenchmarkId::new(
                "Conv-031_deinterleave_stereo_vec",
                ParamLabel::new()
                    .with("len", len).with("ch", STEREO_CHANNELS).with("dt", dt)
                    .build(),
            );
            match dt {
                "i16" => group.bench_with_input(id, &len, |b, &n| {
                    let input = into_nev(flat_sine_vec::<i16>(n * 2));
                    b.iter(|| { black_box(deinterleave_stereo_vec::<i16>(&input).unwrap()); });
                }),
                "i32" => group.bench_with_input(id, &len, |b, &n| {
                    let input = into_nev(flat_sine_vec::<i32>(n * 2));
                    b.iter(|| { black_box(deinterleave_stereo_vec::<i32>(&input).unwrap()); });
                }),
                "f32" => group.bench_with_input(id, &len, |b, &n| {
                    let input = into_nev(flat_sine_vec::<f32>(n * 2));
                    b.iter(|| { black_box(deinterleave_stereo_vec::<f32>(&input).unwrap()); });
                }),
                _ => unreachable!(),
            };
        }
    }
    group.finish();
}

fn bench_conv_032_deinterleave_multi_vec<M: Measurement>(c: &mut Criterion<M>) {
    use audio_samples::simd_conversions::deinterleave_multi_vec;
    let mut group = c.benchmark_group("conv");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in MULTI_CHANNELS {
            for &dt in deinterleave_dtypes() {
                let total = len * ch;
                group.throughput(bytes_throughput(dt, total, 1));
                let id = BenchmarkId::new(
                    "Conv-032_deinterleave_multi_vec",
                    ParamLabel::new()
                        .with("len", len).with("ch", ch).with("dt", dt)
                        .build(),
                );
                let nzu_ch = NonZeroU32::new(ch as u32).unwrap();
                match dt {
                    "i16" => group.bench_with_input(id, &(len, ch), |b, &(n, c_)| {
                        let input = into_nev(flat_sine_vec::<i16>(n * c_));
                        let input_slice: &NonEmptySlice<i16> = input.as_non_empty_slice();
                        b.iter(|| {
                            black_box(deinterleave_multi_vec::<i16>(input_slice, nzu_ch).unwrap());
                        });
                    }),
                    "i32" => group.bench_with_input(id, &(len, ch), |b, &(n, c_)| {
                        let input = into_nev(flat_sine_vec::<i32>(n * c_));
                        let input_slice: &NonEmptySlice<i32> = input.as_non_empty_slice();
                        b.iter(|| {
                            black_box(deinterleave_multi_vec::<i32>(input_slice, nzu_ch).unwrap());
                        });
                    }),
                    "f32" => group.bench_with_input(id, &(len, ch), |b, &(n, c_)| {
                        let input = into_nev(flat_sine_vec::<f32>(n * c_));
                        let input_slice: &NonEmptySlice<f32> = input.as_non_empty_slice();
                        b.iter(|| {
                            black_box(deinterleave_multi_vec::<f32>(input_slice, nzu_ch).unwrap());
                        });
                    }),
                    _ => unreachable!(),
                };
            }
        }
    }
    group.finish();
}

fn bench_conv_033_interleave_stereo_vec<M: Measurement>(c: &mut Criterion<M>) {
    use audio_samples::simd_conversions::interleave_stereo_vec;
    let mut group = c.benchmark_group("conv");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in deinterleave_dtypes() {
            let total = len * STEREO_CHANNELS;
            group.throughput(bytes_throughput(dt, total, 1));
            let id = BenchmarkId::new(
                "Conv-033_interleave_stereo_vec",
                ParamLabel::new()
                    .with("len", len).with("ch", STEREO_CHANNELS).with("dt", dt)
                    .build(),
            );
            match dt {
                "i16" => group.bench_with_input(id, &len, |b, &n| {
                    let input = into_nev(flat_sine_vec::<i16>(n * 2));
                    b.iter(|| { black_box(interleave_stereo_vec::<i16>(&input).unwrap()); });
                }),
                "i32" => group.bench_with_input(id, &len, |b, &n| {
                    let input = into_nev(flat_sine_vec::<i32>(n * 2));
                    b.iter(|| { black_box(interleave_stereo_vec::<i32>(&input).unwrap()); });
                }),
                "f32" => group.bench_with_input(id, &len, |b, &n| {
                    let input = into_nev(flat_sine_vec::<f32>(n * 2));
                    b.iter(|| { black_box(interleave_stereo_vec::<f32>(&input).unwrap()); });
                }),
                _ => unreachable!(),
            };
        }
    }
    group.finish();
}

fn bench_conv_034_interleave_multi_vec<M: Measurement>(c: &mut Criterion<M>) {
    use audio_samples::simd_conversions::interleave_multi_vec;
    let mut group = c.benchmark_group("conv");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in MULTI_CHANNELS {
            for &dt in deinterleave_dtypes() {
                let total = len * ch;
                group.throughput(bytes_throughput(dt, total, 1));
                let id = BenchmarkId::new(
                    "Conv-034_interleave_multi_vec",
                    ParamLabel::new()
                        .with("len", len).with("ch", ch).with("dt", dt)
                        .build(),
                );
                let nzu_ch = NonZeroU32::new(ch as u32).unwrap();
                match dt {
                    "i16" => group.bench_with_input(id, &(len, ch), |b, &(n, c_)| {
                        let input = into_nev(flat_sine_vec::<i16>(n * c_));
                        b.iter(|| {
                            black_box(interleave_multi_vec::<i16>(&input, nzu_ch).unwrap());
                        });
                    }),
                    "i32" => group.bench_with_input(id, &(len, ch), |b, &(n, c_)| {
                        let input = into_nev(flat_sine_vec::<i32>(n * c_));
                        b.iter(|| {
                            black_box(interleave_multi_vec::<i32>(&input, nzu_ch).unwrap());
                        });
                    }),
                    "f32" => group.bench_with_input(id, &(len, ch), |b, &(n, c_)| {
                        let input = into_nev(flat_sine_vec::<f32>(n * c_));
                        b.iter(|| {
                            black_box(interleave_multi_vec::<f32>(&input, nzu_ch).unwrap());
                        });
                    }),
                    _ => unreachable!(),
                };
            }
        }
    }
    group.finish();
}

