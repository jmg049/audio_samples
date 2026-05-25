//! Shared body for the `AudioProcessing` bench targets
//! (`bench_proc_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/proc.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `Proc`
//! (lines 118-141).

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use audio_samples::{
    AudioProcessing, I24, NormalizationConfig, NormalizationMethod, StandardSample,
};

use bench_suite_common::{
    CHANNELS_DEFAULT, DTYPES_DEFAULT, LENGTH_SWEEP_FULL, ParamLabel, SampleSizePolicy,
    fixture_a440, sample_size_for,
};

#[cfg(feature = "resampling")]
use audio_samples::operations::ResamplingQuality;
#[cfg(feature = "resampling")]
use std::num::NonZeroU32;

// ===========================================================================
// Top-level entry point — wrappers call into this with their typed Criterion.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_proc_001_normalize(c);
    bench_proc_002_scale(c);
    bench_proc_003_apply_window(c);
    bench_proc_004_apply_filter(c);
    bench_proc_005_mu_compress(c);
    bench_proc_006_mu_expand(c);
    bench_proc_007_low_pass_filter(c);
    bench_proc_008_high_pass_filter(c);
    bench_proc_009_band_pass_filter(c);
    bench_proc_010_remove_dc_offset(c);
    bench_proc_011_clip(c);
    bench_proc_012_clip_in_place(c);
    #[cfg(feature = "resampling")]
    {
        bench_proc_013_resample(c);
        bench_proc_014_resample_by_ratio(c);
    }
    bench_proc_015_apply_with_error(c);
    bench_proc_016_try_fold(c);
}

// ===========================================================================
// Dispatch macros
//
// `dispatch_proc_consume!`: bench an op that consumes `self` and returns
// `Result<Self>` (or `Self`). Builds a fresh fixture per iter via
// `iter_batched` (NOT `_ref`) so the move is sound.
//
// `dispatch_proc_ref!`: bench an op that takes `&self` (e.g. resample).
//
// For ops whose arguments need typed literals (mu_compress, clip,
// clip_in_place, …), each per-row function defines its own ad-hoc dispatch
// macro inline since the typed literals can't be fed through a generic
// dispatch.
// ===========================================================================

macro_rules! dispatch_proc_consume {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, |$audio:ident| $body:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i16>(n, ch),
                    |$audio| { black_box($body); },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<I24>(n, ch),
                    |$audio| { black_box($body); },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i32>(n, ch),
                    |$audio| { black_box($body); },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<f32>(n, ch),
                    |$audio| { black_box($body); },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

#[cfg(feature = "resampling")]
macro_rules! dispatch_proc_ref {
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

// Standard parameter-string + throughput tagging.
fn label_and_throughput<M: Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
    catalog_id: &str,
    n: usize,
    dt: &str,
    ch: usize,
) -> BenchmarkId {
    group.throughput(Throughput::Elements((n * ch) as u64));
    let label = ParamLabel::new()
        .with("len", n)
        .with("dt", dt)
        .with("ch", ch)
        .build();
    BenchmarkId::new(catalog_id, label)
}

// ===========================================================================
// Helper: build a `NormalizationConfig<T>` for a given method label, on each
// dtype. `min_max` and `peak` need typed bounds, so we use the type's
// `from_f64`-equivalent via `ConvertTo` — but for benches it's enough to use
// fixed config builders that work across all dtypes. We delegate via a tiny
// per-type config factory.
// ===========================================================================

trait NormCfgFactory: StandardSample + Sized {
    fn cfg_peak() -> NormalizationConfig<Self>;
    fn cfg_min_max() -> NormalizationConfig<Self>;
    fn cfg_zscore() -> NormalizationConfig<Self>;
}

macro_rules! norm_cfg_int {
    ($t:ty) => {
        impl NormCfgFactory for $t {
            fn cfg_peak() -> NormalizationConfig<Self> {
                // Half the positive range — well within i16/i24/i32 bounds.
                NormalizationConfig::peak(<$t>::MAX / 2)
            }
            fn cfg_min_max() -> NormalizationConfig<Self> {
                NormalizationConfig::min_max(<$t>::MIN / 2, <$t>::MAX / 2)
            }
            fn cfg_zscore() -> NormalizationConfig<Self> {
                NormalizationConfig::zscore()
            }
        }
    };
}

norm_cfg_int!(i16);
norm_cfg_int!(i32);

impl NormCfgFactory for I24 {
    fn cfg_peak() -> NormalizationConfig<Self> {
        NormalizationConfig::peak(I24::MAX / I24::try_from_i32(2).unwrap())
    }
    fn cfg_min_max() -> NormalizationConfig<Self> {
        NormalizationConfig::min_max(
            I24::MIN / I24::try_from_i32(2).unwrap(),
            I24::MAX / I24::try_from_i32(2).unwrap(),
        )
    }
    fn cfg_zscore() -> NormalizationConfig<Self> {
        NormalizationConfig::zscore()
    }
}

impl NormCfgFactory for f32 {
    fn cfg_peak() -> NormalizationConfig<Self> {
        NormalizationConfig::peak(0.5_f32)
    }
    fn cfg_min_max() -> NormalizationConfig<Self> {
        NormalizationConfig::min_max(-0.5_f32, 0.5_f32)
    }
    fn cfg_zscore() -> NormalizationConfig<Self> {
        NormalizationConfig::zscore()
    }
}

// ===========================================================================
// Proc-001 normalize — sweep three NormalizationMethod variants.
//
// `NoFast`: peak and z-score do two-pass work (find peak / mean+variance
// then scale). MinMax also requires peak detection.
// ===========================================================================

fn cfg_for<T: NormCfgFactory>(method: NormalizationMethod) -> NormalizationConfig<T> {
    match method {
        NormalizationMethod::Peak => T::cfg_peak(),
        NormalizationMethod::MinMax => T::cfg_min_max(),
        NormalizationMethod::ZScore => T::cfg_zscore(),
        // Mean/Median fall back to the safe defaults
        NormalizationMethod::Mean => NormalizationConfig::<T>::mean(),
        NormalizationMethod::Median => NormalizationConfig::<T>::median(),
        _ => T::cfg_peak(),
    }
}

macro_rules! dispatch_normalize {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, $method:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        let method = $method;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || (fixture_a440::<i16>(n, ch), cfg_for::<i16>(method)),
                    |(audio, cfg)| { black_box(audio.normalize(cfg).ok()); },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || (fixture_a440::<I24>(n, ch), cfg_for::<I24>(method)),
                    |(audio, cfg)| { black_box(audio.normalize(cfg).ok()); },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || (fixture_a440::<i32>(n, ch), cfg_for::<i32>(method)),
                    |(audio, cfg)| { black_box(audio.normalize(cfg).ok()); },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || (fixture_a440::<f32>(n, ch), cfg_for::<f32>(method)),
                    |(audio, cfg)| { black_box(audio.normalize(cfg).ok()); },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

fn bench_proc_001_normalize<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("proc");
    // Three representative methods covering the cost spectrum:
    // Peak (one-pass max + scale), MinMax (one-pass max + affine), ZScore
    // (two-pass mean+variance then scale).
    let methods: &[(NormalizationMethod, &str)] = &[
        (NormalizationMethod::Peak, "peak"),
        (NormalizationMethod::MinMax, "min_max"),
        (NormalizationMethod::ZScore, "zscore"),
    ];
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &(method, mlabel) in methods {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("method", mlabel)
                        .build();
                    let id = BenchmarkId::new("Proc-001_normalize", label);
                    dispatch_normalize!(group, id, dt, len, ch, method);
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Proc-002 scale — element-wise multiply by f64; consumes self.
// NoFast: integer types re-quantize through f64.
// ===========================================================================

fn bench_proc_002_scale<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("proc");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Proc-002_scale", len, dt, ch);
                dispatch_proc_consume!(group, id, dt, len, ch, |a| a.scale(0.5));
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Proc-003 apply_window — window length must equal samples-per-channel.
//
// We build a length-N window of ones (cheap; the cost of `apply_window` is
// pure element-wise multiply, the window contents don't matter for timing).
// ===========================================================================

macro_rules! dispatch_apply_window {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                let win_vec: Vec<i16> = vec![1i16; n];
                b.iter_batched(
                    || fixture_a440::<i16>(n, ch),
                    |audio| {
                        let win = non_empty_slice::NonEmptySlice::new(&win_vec).unwrap();
                        black_box(audio.apply_window(win).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                let win_vec: Vec<I24> = vec![I24::try_from_i32(1).unwrap(); n];
                b.iter_batched(
                    || fixture_a440::<I24>(n, ch),
                    |audio| {
                        let win = non_empty_slice::NonEmptySlice::new(&win_vec).unwrap();
                        black_box(audio.apply_window(win).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                let win_vec: Vec<i32> = vec![1i32; n];
                b.iter_batched(
                    || fixture_a440::<i32>(n, ch),
                    |audio| {
                        let win = non_empty_slice::NonEmptySlice::new(&win_vec).unwrap();
                        black_box(audio.apply_window(win).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                let win_vec: Vec<f32> = vec![1.0f32; n];
                b.iter_batched(
                    || fixture_a440::<f32>(n, ch),
                    |audio| {
                        let win = non_empty_slice::NonEmptySlice::new(&win_vec).unwrap();
                        black_box(audio.apply_window(win).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

fn bench_proc_003_apply_window<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("proc");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Proc-003_apply_window", len, dt, ch);
                dispatch_apply_window!(group, id, dt, len, ch);
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Proc-004 apply_filter — FIR convolution, O(N·M).
//
// Sweep filter_len ∈ {16, 64, 256}. Cap signal length at L = 65_536 (index 4)
// to keep the largest (256-tap × 65_536-sample) point under ~30 s.
// ===========================================================================

const PROC_004_LENGTHS: &[usize] = &[256, 1024, 4096, 16_384, 65_536];

macro_rules! dispatch_apply_filter {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, $fl:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        let fl = $fl;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch, fl), |b, &(n, ch, fl)| {
                let coeffs: Vec<i16> = vec![1i16; fl];
                b.iter_batched(
                    || fixture_a440::<i16>(n, ch),
                    |audio| {
                        let h = non_empty_slice::NonEmptySlice::new(&coeffs).unwrap();
                        black_box(audio.apply_filter(h).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch, fl), |b, &(n, ch, fl)| {
                let coeffs: Vec<I24> = vec![I24::try_from_i32(1).unwrap(); fl];
                b.iter_batched(
                    || fixture_a440::<I24>(n, ch),
                    |audio| {
                        let h = non_empty_slice::NonEmptySlice::new(&coeffs).unwrap();
                        black_box(audio.apply_filter(h).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch, fl), |b, &(n, ch, fl)| {
                let coeffs: Vec<i32> = vec![1i32; fl];
                b.iter_batched(
                    || fixture_a440::<i32>(n, ch),
                    |audio| {
                        let h = non_empty_slice::NonEmptySlice::new(&coeffs).unwrap();
                        black_box(audio.apply_filter(h).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch, fl), |b, &(n, ch, fl)| {
                let coeffs: Vec<f32> = vec![1.0f32 / (fl as f32); fl];
                b.iter_batched(
                    || fixture_a440::<f32>(n, ch),
                    |audio| {
                        let h = non_empty_slice::NonEmptySlice::new(&coeffs).unwrap();
                        black_box(audio.apply_filter(h).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

fn bench_proc_004_apply_filter<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("proc");
    let filter_lens: &[usize] = &[16, 64, 256];
    for (i, &len) in PROC_004_LENGTHS.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &fl in filter_lens {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("filter_len", fl)
                        .build();
                    let id = BenchmarkId::new("Proc-004_apply_filter", label);
                    dispatch_apply_filter!(group, id, dt, len, ch, fl);
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Proc-005 mu_compress — non-linear element-wise transform; consumes self.
//
// The `mu` argument has type `T` so we cannot fold the dtype match into the
// generic `dispatch_proc_consume!` macro: each dtype arm needs its own typed
// literal. Inline the iter_batched call per dtype.
// ===========================================================================

macro_rules! dispatch_mu_compress {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i16>(n, ch),
                    |audio| { black_box(audio.mu_compress(127i16).ok()); },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<I24>(n, ch),
                    |audio| {
                        black_box(audio.mu_compress(I24::try_from_i32(127).unwrap()).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i32>(n, ch),
                    |audio| { black_box(audio.mu_compress(127i32).ok()); },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<f32>(n, ch),
                    |audio| { black_box(audio.mu_compress(255.0f32).ok()); },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

fn bench_proc_005_mu_compress<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("proc");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Proc-005_mu_compress", len, dt, ch);
                dispatch_mu_compress!(group, id, dt, len, ch);
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Proc-006 mu_expand — inverse of mu_compress; consumes self.
// ===========================================================================

macro_rules! dispatch_mu_expand {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i16>(n, ch),
                    |audio| { black_box(audio.mu_expand(127i16).ok()); },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<I24>(n, ch),
                    |audio| {
                        black_box(audio.mu_expand(I24::try_from_i32(127).unwrap()).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i32>(n, ch),
                    |audio| { black_box(audio.mu_expand(127i32).ok()); },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<f32>(n, ch),
                    |audio| { black_box(audio.mu_expand(255.0f32).ok()); },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

fn bench_proc_006_mu_expand<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("proc");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Proc-006_mu_expand", len, dt, ch);
                dispatch_mu_expand!(group, id, dt, len, ch);
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Proc-007 low_pass_filter — single-pole IIR; consumes self.
// ===========================================================================

fn bench_proc_007_low_pass_filter<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("proc");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Proc-007_low_pass_filter", len, dt, ch);
                dispatch_proc_consume!(group, id, dt, len, ch, |a| a.low_pass_filter(1_000.0).ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Proc-008 high_pass_filter — single-pole IIR; consumes self.
// ===========================================================================

fn bench_proc_008_high_pass_filter<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("proc");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Proc-008_high_pass_filter", len, dt, ch);
                dispatch_proc_consume!(group, id, dt, len, ch, |a| a
                    .high_pass_filter(1_000.0)
                    .ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Proc-009 band_pass_filter — composed LP + HP; consumes self.
// ===========================================================================

fn bench_proc_009_band_pass_filter<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("proc");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Proc-009_band_pass_filter", len, dt, ch);
                dispatch_proc_consume!(group, id, dt, len, ch, |a| a
                    .band_pass_filter(300.0, 3_000.0)
                    .ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Proc-010 remove_dc_offset — two-pass (mean then subtract); consumes self.
// ===========================================================================

fn bench_proc_010_remove_dc_offset<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("proc");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id =
                    label_and_throughput(&mut group, "Proc-010_remove_dc_offset", len, dt, ch);
                dispatch_proc_consume!(group, id, dt, len, ch, |a| a.remove_dc_offset().ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Proc-011 clip — consumes self, allocates result.
// ===========================================================================

macro_rules! dispatch_clip {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i16>(n, ch),
                    |audio| { black_box(audio.clip(i16::MIN / 2, i16::MAX / 2).ok()); },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<I24>(n, ch),
                    |audio| {
                        let half = I24::try_from_i32(2).unwrap();
                        black_box(audio.clip(I24::MIN / half, I24::MAX / half).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i32>(n, ch),
                    |audio| { black_box(audio.clip(i32::MIN / 2, i32::MAX / 2).ok()); },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<f32>(n, ch),
                    |audio| { black_box(audio.clip(-0.5f32, 0.5f32).ok()); },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

fn bench_proc_011_clip<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("proc");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Proc-011_clip", len, dt, ch);
                dispatch_clip!(group, id, dt, len, ch);
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Proc-012 clip_in_place — takes &mut self; benches against Proc-011 to
// quantify allocation cost.
// ===========================================================================

macro_rules! dispatch_clip_in_place {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |audio| {
                        black_box(audio.clip_in_place(i16::MIN / 2, i16::MAX / 2).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<I24>(n, ch),
                    |audio| {
                        let half = I24::try_from_i32(2).unwrap();
                        black_box(audio.clip_in_place(I24::MIN / half, I24::MAX / half).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i32>(n, ch),
                    |audio| {
                        black_box(audio.clip_in_place(i32::MIN / 2, i32::MAX / 2).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |audio| {
                        black_box(audio.clip_in_place(-0.5f32, 0.5f32).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

fn bench_proc_012_clip_in_place<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("proc");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id =
                    label_and_throughput(&mut group, "Proc-012_clip_in_place", len, dt, ch);
                dispatch_clip_in_place!(group, id, dt, len, ch);
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Proc-013 resample — gated behind `resampling`. Sweep three qualities at
// representative lengths (cap at 262_144 since rubato is heavyweight).
// Output sample rate fixed to 48_000 (input 44_100).
//
// NOTE for orchestrator: the Resamp section in CATALOG.md is shallow (only
// 2 free fns); Proc-013/014 are the trait-method benches and are kept here
// behind a cfg gate.
// ===========================================================================

#[cfg(feature = "resampling")]
const PROC_013_LENGTHS: &[usize] = &[1024, 4096, 16_384, 65_536, 262_144];

#[cfg(feature = "resampling")]
fn bench_proc_013_resample<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("proc");
    let target_sr: NonZeroU32 = NonZeroU32::new(48_000).unwrap();
    let qualities: &[(ResamplingQuality, &str)] = &[
        (ResamplingQuality::Fast, "fast"),
        (ResamplingQuality::Medium, "medium"),
        (ResamplingQuality::High, "high"),
    ];
    for (i, &len) in PROC_013_LENGTHS.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &(quality, qlabel) in qualities {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("dst_sr", target_sr.get())
                        .with("quality", qlabel)
                        .build();
                    let id = BenchmarkId::new("Proc-013_resample", label);
                    dispatch_proc_ref!(group, id, dt, len, ch, |a| a
                        .resample(target_sr, quality)
                        .ok());
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Proc-014 resample_by_ratio — gated behind `resampling`. Sweep five ratios.
// ===========================================================================

#[cfg(feature = "resampling")]
fn bench_proc_014_resample_by_ratio<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("proc");
    let ratios: &[f64] = &[0.5, 1.5, 2.0, 0.999, 3.14159];
    let quality = ResamplingQuality::Medium;
    for (i, &len) in PROC_013_LENGTHS.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &ratio in ratios {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("ratio", format!("{ratio:.5}"))
                        .with("quality", "medium")
                        .build();
                    let id = BenchmarkId::new("Proc-014_resample_by_ratio", label);
                    dispatch_proc_ref!(group, id, dt, len, ch, |a| a
                        .resample_by_ratio(ratio, quality)
                        .ok());
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Proc-015 apply_with_error — inherent on AudioSamples; closure-driven
// element-wise mutation; consumes self.
//
// Closure: |x| Ok(x * 0.5). For integer types we cannot multiply by 0.5
// directly via StandardSample arithmetic, so we keep the closure simple by
// performing the half-scaling through ConvertTo<f64> round-trip — but the
// simplest portable choice is `Ok(x)` (identity). Catalog says "trivial
// closure" — use `Ok(x)` for portability across dtypes.
// ===========================================================================

macro_rules! dispatch_apply_with_error {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i16>(n, ch),
                    |audio| {
                        black_box(audio.apply_with_error(|x: i16| Ok(x)).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<I24>(n, ch),
                    |audio| {
                        black_box(audio.apply_with_error(|x: I24| Ok(x)).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i32>(n, ch),
                    |audio| {
                        black_box(audio.apply_with_error(|x: i32| Ok(x)).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<f32>(n, ch),
                    |audio| {
                        black_box(audio.apply_with_error(|x: f32| Ok(x * 0.5)).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

fn bench_proc_015_apply_with_error<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("proc");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id =
                    label_and_throughput(&mut group, "Proc-015_apply_with_error", len, dt, ch);
                dispatch_apply_with_error!(group, id, dt, len, ch);
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Proc-016 try_fold — inherent on AudioSamples; FnMut closure with mutable
// accumulator. Take &mut self.
//
// Closure: |acc, x| { *acc = *acc + 1; Ok(x) } — counts samples (cheap,
// portable across all dtypes).
// ===========================================================================

macro_rules! dispatch_try_fold {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |audio| {
                        black_box(audio.try_fold(0usize, |acc, x: i16| {
                            *acc += 1;
                            Ok(x)
                        }).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<I24>(n, ch),
                    |audio| {
                        black_box(audio.try_fold(0usize, |acc, x: I24| {
                            *acc += 1;
                            Ok(x)
                        }).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i32>(n, ch),
                    |audio| {
                        black_box(audio.try_fold(0usize, |acc, x: i32| {
                            *acc += 1;
                            Ok(x)
                        }).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |audio| {
                        black_box(audio.try_fold(0.0f32, |acc, x: f32| {
                            *acc += x;
                            Ok(x * 0.5)
                        }).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

fn bench_proc_016_try_fold<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("proc");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Proc-016_try_fold", len, dt, ch);
                dispatch_try_fold!(group, id, dt, len, ch);
            }
        }
    }
    group.finish();
}
