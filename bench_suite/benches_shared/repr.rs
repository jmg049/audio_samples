//! Shared body for the `AudioSamples` core representation bench targets
//! (`bench_repr_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/repr.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `Repr`
//! (rows Repr-001 .. Repr-009, lines 696-715).
//!
//! Strategy notes:
//! - Repr-001 / Repr-003 are *constructors* that consume an `Array1` / `Array2`
//!   / `NonEmptyVec`. We use `iter_batched` (not `_ref`) and rebuild the
//!   input in `setup` so each iteration sees a fresh moveable value.
//! - Repr-002 is also a constructor but takes only `(length, sr)`/
//!   `(channels, length, sr)` — no preexisting data; we just call it with
//!   `iter`. The interesting axis is the allocation size.
//! - Repr-004 / Repr-006 / Repr-007 / Repr-008 / Repr-009 are accessors on
//!   a long-lived `&AudioSamples` fixture. We allocate the fixture once
//!   outside the timed loop and reference it from the closure.
//! - Repr-005 (multi-channel axis reductions) is mono-incompatible so we
//!   pin `ch = 2` (and `ch = 6` if the bench is fast enough).

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::num::{NonZeroU32, NonZeroUsize};

use audio_samples::{AudioSamples, I24, sample_rate};
use ndarray::{Array1, Array2, Axis, s};
use non_empty_slice::NonEmptyVec;

use bench_suite_common::{
    CHANNELS_DEFAULT, DTYPES_DEFAULT, LENGTH_SWEEP_FULL, ParamLabel, SampleSizePolicy,
    fixture_a440, sample_size_for,
};

// ===========================================================================
// Top-level entry point — wrappers call into this with their typed Criterion.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_repr_001_new(c);
    bench_repr_002_zeros(c);
    bench_repr_003_from_mono_vec(c);
    bench_repr_004_mono_stats(c);
    bench_repr_005_multi_axis_stats(c);
    bench_repr_006_multi_bulk_stats(c);
    bench_repr_007_mono_iter(c);
    bench_repr_008_mono_mapv(c);
    bench_repr_009_mono_slice(c);
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Build the `ChannelCount` for an integer channel count. Bench helper —
/// panics on zero, which the bench sweeps never produce.
#[inline]
fn ch_count(ch: usize) -> NonZeroU32 {
    NonZeroU32::new(ch as u32).expect("channel count must be > 0 for benches")
}

/// Build the `NonZeroUsize` for an integer length. Bench helper.
#[inline]
fn len_nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("length must be > 0 for benches")
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
        .with("len", n).with("dt", dt).with("ch", ch).build();
    BenchmarkId::new(catalog_id, label)
}

// ===========================================================================
// Dispatch macros for the four DTYPES_DEFAULT types.
// ===========================================================================

/// Bench an `AudioSamples` accessor against a long-lived fixture, similar to
/// `dispatch_unary_sine!` in `stats.rs`. Use this when the bench routine
/// takes `&AudioSamples` (or `&mut`) and the fixture cost dominates if
/// rebuilt per-iter.
macro_rules! dispatch_accessor {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, |$audio:ident| $body:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                let fixture = fixture_a440::<i16>(n, ch);
                b.iter(|| {
                    let $audio = &fixture;
                    black_box($body);
                });
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                let fixture = fixture_a440::<I24>(n, ch);
                b.iter(|| {
                    let $audio = &fixture;
                    black_box($body);
                });
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                let fixture = fixture_a440::<i32>(n, ch);
                b.iter(|| {
                    let $audio = &fixture;
                    black_box($body);
                });
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                let fixture = fixture_a440::<f32>(n, ch);
                b.iter(|| {
                    let $audio = &fixture;
                    black_box($body);
                });
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

// ===========================================================================
// Repr-001 new_mono / new_multi_channel — constructor cost. The validation
// (`is_empty`) and the move-into-`AudioData` enum is what's timed. Setup
// builds the ndarray (clone of the fixture's underlying buffer) and the
// routine consumes it via `new_mono` / `new_multi_channel`.
// ===========================================================================

fn bench_repr_001_new<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("repr");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len).with("dt", dt).with("ch", ch).build();
                let id = BenchmarkId::new("Repr-001_new", label);
                dispatch_new(&mut group, id, dt, len, ch);
            }
        }
    }
    group.finish();
}

fn dispatch_new<M: Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
    id: BenchmarkId,
    dt: &str,
    n: usize,
    ch: usize,
) {
    let sr = sample_rate!(44_100);
    macro_rules! arm {
        ($t:ty) => {{
            if ch == 1 {
                group.bench_with_input(id, &(n, ch), |b, &(n, _ch)| {
                    b.iter_batched(
                        || Array1::<$t>::zeros(n),
                        |arr| {
                            black_box(AudioSamples::<$t>::new_mono(arr, sr).unwrap());
                        },
                        BatchSize::LargeInput,
                    );
                });
            } else {
                group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                    b.iter_batched(
                        || Array2::<$t>::zeros((ch, n)),
                        |arr| {
                            black_box(AudioSamples::<$t>::new_multi_channel(arr, sr).unwrap());
                        },
                        BatchSize::LargeInput,
                    );
                });
            }
        }};
    }
    match dt {
        "i16" => arm!(i16),
        "I24" => arm!(I24),
        "i32" => arm!(i32),
        "f32" => arm!(f32),
        _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
    }
}

// ===========================================================================
// Repr-002 zeros_mono / zeros_multi_channel — pure allocation. No setup;
// each iter allocates a fresh zeroed buffer of the requested geometry.
// ===========================================================================

fn bench_repr_002_zeros<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("repr");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len).with("dt", dt).with("ch", ch).build();
                let id = BenchmarkId::new("Repr-002_zeros", label);
                dispatch_zeros(&mut group, id, dt, len, ch);
            }
        }
    }
    group.finish();
}

fn dispatch_zeros<M: Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
    id: BenchmarkId,
    dt: &str,
    n: usize,
    ch: usize,
) {
    let sr = sample_rate!(44_100);
    let length = len_nz(n);
    let chc = ch_count(ch);
    macro_rules! arm {
        ($t:ty) => {{
            if ch == 1 {
                group.bench_with_input(id, &(n, ch), |b, &(_n, _ch)| {
                    b.iter(|| {
                        black_box(AudioSamples::<$t>::zeros_mono(length, sr));
                    });
                });
            } else {
                group.bench_with_input(id, &(n, ch), |b, &(_n, _ch)| {
                    b.iter(|| {
                        black_box(AudioSamples::<$t>::zeros_multi_channel(chc, length, sr));
                    });
                });
            }
        }};
    }
    match dt {
        "i16" => arm!(i16),
        "I24" => arm!(I24),
        "i32" => arm!(i32),
        "f32" => arm!(f32),
        _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
    }
}

// ===========================================================================
// Repr-003 from_mono_vec — wraps a `NonEmptyVec<T>` into mono `AudioSamples`.
// Mono only by API design. Bench sweeps length only.
// ===========================================================================

fn bench_repr_003_from_mono_vec<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("repr");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &dt in DTYPES_DEFAULT {
            group.throughput(Throughput::Elements(len as u64));
            let label = ParamLabel::new()
                .with("len", len).with("dt", dt).with("ch", ch).build();
            let id = BenchmarkId::new("Repr-003_from_mono_vec", label);
            dispatch_from_mono_vec(&mut group, id, dt, len);
        }
    }
    group.finish();
}

fn dispatch_from_mono_vec<M: Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
    id: BenchmarkId,
    dt: &str,
    n: usize,
) {
    let sr = sample_rate!(44_100);
    macro_rules! arm {
        ($t:ty) => {{
            group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched(
                    || NonEmptyVec::new(vec![<$t as Default>::default(); n])
                        .expect("non-empty vec for non-zero bench length"),
                    |v| {
                        let audio: AudioSamples<'static, $t> =
                            AudioSamples::<$t>::from_mono_vec(v, sr);
                        black_box(audio);
                    },
                    BatchSize::LargeInput,
                );
            });
        }};
    }
    match dt {
        "i16" => arm!(i16),
        "I24" => arm!(I24),
        "i32" => arm!(i32),
        "f32" => arm!(f32),
        _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
    }
}

// ===========================================================================
// Repr-004 MonoData::mean / variance / stddev — 1D reductions on the mono
// fixture. Access via `as_mono()`. Mono only.
// ===========================================================================

fn bench_repr_004_mono_stats<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("repr");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &dt in DTYPES_DEFAULT {
            for op in ["mean", "variance", "stddev"] {
                group.throughput(Throughput::Elements(len as u64));
                let label = ParamLabel::new()
                    .with("len", len).with("dt", dt).with("ch", ch)
                    .with("op", op)
                    .build();
                let id = BenchmarkId::new("Repr-004_mono_stats", label);
                match op {
                    "mean" => {
                        dispatch_accessor!(group, id, dt, len, ch, |a| {
                            a.as_mono().unwrap().mean()
                        });
                    }
                    "variance" => {
                        dispatch_accessor!(group, id, dt, len, ch, |a| {
                            a.as_mono().unwrap().variance()
                        });
                    }
                    "stddev" => {
                        dispatch_accessor!(group, id, dt, len, ch, |a| {
                            a.as_mono().unwrap().stddev()
                        });
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Repr-005 MultiData::{mean_axis, variance_axis, stddev_axis,
// variance_axis_ddof, stddev_axis_ddof} — 2D axis-wise reductions. Stereo
// only (mono has no multi data); axis sweep {Axis(0), Axis(1)}.
// ===========================================================================

fn bench_repr_005_multi_axis_stats<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("repr");
    let ch = 2; // multi-only
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &dt in DTYPES_DEFAULT {
            for &axis_n in &[0usize, 1usize] {
                let axis = Axis(axis_n);
                for op in ["mean_axis", "variance_axis", "stddev_axis",
                           "variance_axis_ddof", "stddev_axis_ddof"]
                {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len).with("dt", dt).with("ch", ch)
                        .with("axis", axis_n).with("op", op)
                        .build();
                    let id = BenchmarkId::new("Repr-005_multi_axis_stats", label);
                    match op {
                        "mean_axis" => {
                            dispatch_accessor!(group, id, dt, len, ch, |a| {
                                a.as_multi_channel().unwrap().mean_axis(axis)
                            });
                        }
                        "variance_axis" => {
                            dispatch_accessor!(group, id, dt, len, ch, |a| {
                                a.as_multi_channel().unwrap().variance_axis(axis)
                            });
                        }
                        "stddev_axis" => {
                            dispatch_accessor!(group, id, dt, len, ch, |a| {
                                a.as_multi_channel().unwrap().stddev_axis(axis)
                            });
                        }
                        "variance_axis_ddof" => {
                            dispatch_accessor!(group, id, dt, len, ch, |a| {
                                a.as_multi_channel().unwrap().variance_axis_ddof(axis, 1)
                            });
                        }
                        "stddev_axis_ddof" => {
                            dispatch_accessor!(group, id, dt, len, ch, |a| {
                                a.as_multi_channel().unwrap().stddev_axis_ddof(axis, 1)
                            });
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Repr-006 MultiData::{sum, mean, variance, stddev} — flattened bulk
// reductions over the full 2D buffer. Stereo only.
// ===========================================================================

fn bench_repr_006_multi_bulk_stats<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("repr");
    let ch = 2;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &dt in DTYPES_DEFAULT {
            for op in ["sum", "mean", "variance", "stddev"] {
                group.throughput(Throughput::Elements((len * ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len).with("dt", dt).with("ch", ch)
                    .with("op", op)
                    .build();
                let id = BenchmarkId::new("Repr-006_multi_bulk_stats", label);
                match op {
                    "sum" => {
                        dispatch_accessor!(group, id, dt, len, ch, |a| {
                            a.as_multi_channel().unwrap().sum()
                        });
                    }
                    "mean" => {
                        dispatch_accessor!(group, id, dt, len, ch, |a| {
                            a.as_multi_channel().unwrap().mean()
                        });
                    }
                    "variance" => {
                        dispatch_accessor!(group, id, dt, len, ch, |a| {
                            a.as_multi_channel().unwrap().variance()
                        });
                    }
                    "stddev" => {
                        dispatch_accessor!(group, id, dt, len, ch, |a| {
                            a.as_multi_channel().unwrap().stddev()
                        });
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Repr-007 MonoData::iter — ndarray iterator construction + full traversal
// (no-op accumulator). Mono only; this row is the per-element-iter overhead
// baseline that `frames()` / `apply_to_*` are compared against.
// ===========================================================================

fn bench_repr_007_mono_iter<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("repr");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Repr-007_mono_iter", len, dt, ch);
            dispatch_accessor!(group, id, dt, len, ch, |a| {
                a.as_mono().unwrap().iter().for_each(|s| { black_box(s); })
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Repr-008 MonoData::mapv — allocates a new `Array1<U>` by applying a
// (no-op) closure to every sample. Bench just the identity map; the cost
// is the per-element copy + the destination allocation.
// ===========================================================================

fn bench_repr_008_mono_mapv<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("repr");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Repr-008_mono_mapv", len, dt, ch);
            dispatch_accessor!(group, id, dt, len, ch, |a| {
                a.as_mono().unwrap().mapv(|x| x)
            });
        }
    }
    group.finish();
}

// ===========================================================================
// Repr-009 MonoData::slice — sub-view construction. Bench the (very cheap)
// O(1) slice operation across a half-length range. Mono only.
// ===========================================================================

fn bench_repr_009_mono_slice<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("repr");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Repr-009_mono_slice", len, dt, ch);
            let half = len / 2;
            dispatch_accessor!(group, id, dt, len, ch, |a| {
                a.as_mono().unwrap().slice(s![..half])
            });
        }
    }
    group.finish();
}
