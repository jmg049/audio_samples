//! Shared body for the `AudioSampleIterators` / `apply_to_*` bench targets
//! (`bench_iter_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/iter.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `Iter`
//! (rows Iter-001 .. Iter-008, lines 661-677).
//!
//! Strategy notes:
//! - Iter-001/002/003 are immutable iterators on `&AudioSamples`. We bench
//!   full traversal with a `for_each` no-op accumulator so the iterator is
//!   actually consumed; construction-only is also benched (Iter-001_construct)
//!   because per-row CATALOG specifies "creation cost (negligible) AND a full
//!   traversal".
//! - Iter-005/006/007/008 are `&mut self` methods. We use `iter_batched`
//!   (not `_ref`) so each iteration gets its own fixture: `apply_to_windows`
//!   mutates the underlying buffer (multiplies by 1.0 conceptually — the
//!   closure is a no-op black_box).
//! - Iter-003/004 (`windows()`, `with_padding_mode`) are gated on the
//!   `editing` feature.
//!
//! Compare with `src/iterators.rs:1624` (`test_performance_comparison_apply_vs_iterator`)
//! for the apply-vs-iterator delta this suite is intended to surface.

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

#[cfg(feature = "editing")]
use std::num::NonZeroUsize;

use audio_samples::I24;

#[cfg(feature = "editing")]
use audio_samples::PaddingMode;

use bench_suite_common::{
    CHANNELS_DEFAULT, DTYPES_DEFAULT, LENGTH_SWEEP_FULL, ParamLabel, SampleSizePolicy,
    fixture_a440, sample_size_for,
};

// ===========================================================================
// Top-level entry point — wrappers call into this with their typed Criterion.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_iter_001_frames(c);
    bench_iter_002_channels(c);
    #[cfg(feature = "editing")]
    {
        bench_iter_003_windows(c);
        bench_iter_004_windows_with_padding_mode(c);
    }
    bench_iter_005_apply_to_frames(c);
    bench_iter_006_try_apply_to_channel_data(c);
    bench_iter_007_apply_to_channel_data(c);
    bench_iter_008_apply_to_windows(c);
}

// ===========================================================================
// Dispatch macros — DTYPES_DEFAULT-typed expansion.
//
// `dispatch_iter_traverse!`: bench a `&AudioSamples` -> iterator -> full
// consumption pipeline. The audio fixture lives outside the timed loop
// (`iter_with_setup` style via `iter_batched_ref`); construction of the
// iterator is part of the timed body.
//
// `dispatch_iter_construct!`: bench just iterator construction, no traversal.
//
// `dispatch_apply_mut!`: bench a `&mut self` method with a no-op closure.
// Uses `iter_batched` because the closure may mutate state; each iter gets
// a fresh fixture.
// ===========================================================================

macro_rules! dispatch_iter_traverse {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, |$audio:ident| $body:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |$audio| { $body; },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<I24>(n, ch),
                    |$audio| { $body; },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i32>(n, ch),
                    |$audio| { $body; },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |$audio| { $body; },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

macro_rules! dispatch_apply_mut {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, |$audio:ident| $body:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i16>(n, ch),
                    |mut $audio| { $body; },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<I24>(n, ch),
                    |mut $audio| { $body; },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<i32>(n, ch),
                    |mut $audio| { $body; },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched(
                    || fixture_a440::<f32>(n, ch),
                    |mut $audio| { $body; },
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
        .with("len", n).with("dt", dt).with("ch", ch).build();
    BenchmarkId::new(catalog_id, label)
}

// ===========================================================================
// Iter-001 frames — full traversal of FrameIterator with no-op consumer.
// FastSmall tier; per-frame work is a borrow + return.
// ===========================================================================

fn bench_iter_001_frames<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("iter");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Iter-001_frames", len, dt, ch);
                dispatch_iter_traverse!(group, id, dt, len, ch, |a| {
                    a.frames().for_each(|frame| { black_box(frame); });
                });
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Iter-002 channels — full traversal of ChannelIterator. Per-channel work
// is the full per-channel slice yielded as an owned AudioSamples view. Cost
// is dominated by the number of channels, not samples, so we use FastSmall.
// ===========================================================================

fn bench_iter_002_channels<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("iter");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Iter-002_channels", len, dt, ch);
                dispatch_iter_traverse!(group, id, dt, len, ch, |a| {
                    a.channels().for_each(|chan| { black_box(chan); });
                });
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Iter-003 windows — gated on `editing`. Full traversal of WindowIterator
// (default PaddingMode::Zero). Sweep window/hop ratios via two divisors
// of the length so successive windows overlap at hop=window/2 and are
// non-overlapping at hop=window.
// ===========================================================================

#[cfg(feature = "editing")]
fn bench_iter_003_windows<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("iter");
    // Fixed window_size / hop_size ratios that stay safely within range for
    // every length in LENGTH_SWEEP_FULL.
    let configs: &[(usize, usize)] = &[(512, 512), (512, 256)];
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &(win, hop) in configs {
                if win > len {
                    continue;
                }
                let win_nz = NonZeroUsize::new(win).unwrap();
                let hop_nz = NonZeroUsize::new(hop).unwrap();
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len).with("dt", dt).with("ch", ch)
                        .with("win", win).with("hop", hop)
                        .build();
                    let id = BenchmarkId::new("Iter-003_windows", label);
                    dispatch_iter_traverse!(group, id, dt, len, ch, |a| {
                        a.windows(win_nz, hop_nz).for_each(|w| { black_box(w); });
                    });
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Iter-004 windows.with_padding_mode — gated on `editing`. Sweep the
// PaddingMode variants {Zero, None, Skip}. Length fixed at one
// representative point; padding affects only the trailing window.
// ===========================================================================

#[cfg(feature = "editing")]
fn bench_iter_004_windows_with_padding_mode<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("iter");
    // Use a window/hop that yields a non-trivial trailing-window decision so
    // the padding mode actually affects the iterator's output. Keep length
    // moderate — this row is about the per-iterator constructor branch, not
    // throughput.
    let win: usize = 1024;
    let hop: usize = 512;
    let win_nz = NonZeroUsize::new(win).unwrap();
    let hop_nz = NonZeroUsize::new(hop).unwrap();
    let modes: &[(&str, PaddingMode)] = &[
        ("zero", PaddingMode::Zero),
        ("none", PaddingMode::None),
        ("skip", PaddingMode::Skip),
    ];

    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        if len < win {
            continue;
        }
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &(mode_name, mode) in modes {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len).with("dt", dt).with("ch", ch)
                        .with("win", win).with("hop", hop)
                        .with("pad", mode_name)
                        .build();
                    let id = BenchmarkId::new("Iter-004_windows_with_padding_mode", label);
                    dispatch_iter_traverse!(group, id, dt, len, ch, |a| {
                        a.windows(win_nz, hop_nz)
                            .with_padding_mode(mode)
                            .for_each(|w| { black_box(w); });
                    });
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Iter-005 apply_to_frames — &mut self; closure is a no-op black_box.
// Bench measures dispatch overhead per frame plus the multi-channel
// temporary-buffer alloc/copy on stereo+.
// ===========================================================================

fn bench_iter_005_apply_to_frames<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("iter");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Iter-005_apply_to_frames", len, dt, ch);
                dispatch_apply_mut!(group, id, dt, len, ch, |a| {
                    a.apply_to_frames(|_idx, frame| { black_box(frame); });
                });
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Iter-006 try_apply_to_channel_data — fallible counterpart of Iter-007.
// Returns AudioSampleResult<()>. We discard the result; the fixture is
// always contiguous so the fallible path's Ok branch is what's timed.
// ===========================================================================

fn bench_iter_006_try_apply_to_channel_data<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("iter");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(
                    &mut group, "Iter-006_try_apply_to_channel_data", len, dt, ch,
                );
                dispatch_apply_mut!(group, id, dt, len, ch, |a| {
                    black_box(a.try_apply_to_channel_data(|_ch, samples| {
                        black_box(samples);
                    }).ok());
                });
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Iter-007 apply_to_channel_data — infallible variant; one closure call per
// channel with the full contiguous channel slice. This is the fast-path
// counterpart to the per-sample iterator and should drastically outperform
// `frames()`-style traversal for vectorised inner work.
// ===========================================================================

fn bench_iter_007_apply_to_channel_data<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("iter");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(
                    &mut group, "Iter-007_apply_to_channel_data", len, dt, ch,
                );
                dispatch_apply_mut!(group, id, dt, len, ch, |a| {
                    a.apply_to_channel_data(|_ch, samples| { black_box(samples); });
                });
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Iter-008 apply_to_windows — &mut self; bench non-overlapping windows
// (hop == window) and 50%-overlap windows (hop == window/2) to expose the
// dispatch overhead per window vs the per-sample mutation cost. The closure
// is a no-op black_box of the window slice.
// ===========================================================================

fn bench_iter_008_apply_to_windows<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("iter");
    let configs: &[(usize, usize)] = &[(512, 512), (512, 256)];
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &(win, hop) in configs {
                if win > len {
                    continue;
                }
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len).with("dt", dt).with("ch", ch)
                        .with("win", win).with("hop", hop)
                        .build();
                    let id = BenchmarkId::new("Iter-008_apply_to_windows", label);
                    dispatch_apply_mut!(group, id, dt, len, ch, |a| {
                        a.apply_to_windows(win, hop, |_idx, samples| {
                            black_box(samples);
                        });
                    });
                }
            }
        }
    }
    group.finish();
}
