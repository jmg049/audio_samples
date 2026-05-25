//! Shared body for the `AudioEditing` bench targets
//! (`bench_edit_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/edit.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `Edit`
//! (lines 181-209).
//!
//! ## Shape notes
//!
//! - Most `AudioEditing` methods either consume `self` semantically (return a
//!   new `AudioSamples`) or take `&mut self`. We use `iter_batched_ref` and
//!   call the method via `&` for non-mutating ops, and `iter_batched_ref`
//!   with `&mut` for in-place ops. The fixture clone happens in `setup`, so
//!   the per-iter cost reflects the op + its allocation (matching what
//!   callers pay; see METHODOLOGY §6).
//! - Static methods (`concatenate`, `concatenate_owned`, `mix`, `stack`)
//!   construct a `NonEmptyVec`/`NonEmptySlice` of segments in `setup`.
//!   `concatenate_owned` consumes the vec → uses `iter_batched`.
//! - Perturbation benches sweep representative `PerturbationMethod` variants
//!   (CATALOG Open Q 17): GaussianNoise (white), RandomGain, LowPassFilter,
//!   PitchShift. PitchShift is also pinned in CATALOG Open Q 17 as the most
//!   expensive — we exclude it from the largest two length points to keep
//!   the bench tractable.
//! - `Edit-018 trim_all_silence` uses a custom fixture with embedded silence
//!   runs (CATALOG Open Q 12). `Edit-014 trim_silence` uses the standard
//!   sine fixture (no leading/trailing silence to trim — this is a
//!   best-case scan cost).

use std::num::NonZeroU32;
use std::time::Duration;

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use audio_samples::operations::types::{FadeCurve, NoiseColor, PadSide};
#[cfg(all(feature = "random-generation", feature = "iir-filtering"))]
use audio_samples::operations::types::{PerturbationConfig, PerturbationMethod};
use audio_samples::utils::generation::{silence, sine_wave};
use audio_samples::{AudioEditing, AudioSamples, I24, StandardSample, sample_rate};
use non_empty_slice::{NonEmptySlice, NonEmptyVec};

#[cfg(all(feature = "random-generation", feature = "iir-filtering"))]
use bench_suite_common::BENCH_RNG_SEED;
use bench_suite_common::{
    CHANNELS_DEFAULT, DTYPES_DEFAULT, LENGTH_SWEEP_FULL, LENGTH_SWEEP_NO_XXXL, ParamLabel,
    SampleSizePolicy, fixture_a440, sample_size_for, seeded_rng,
};

// ===========================================================================
// Top-level entry point — wrappers call into this with their typed Criterion.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_edit_001_reverse(c);
    bench_edit_002_reverse_in_place(c);
    bench_edit_003_trim(c);
    bench_edit_004_pad(c);
    bench_edit_005_pad_samples_right(c);
    bench_edit_006_pad_to_duration(c);
    bench_edit_007_split(c);
    bench_edit_008_concatenate(c);
    bench_edit_009_concatenate_owned(c);
    bench_edit_010_mix(c);
    bench_edit_011_fade_in(c);
    bench_edit_012_fade_out(c);
    bench_edit_013_repeat(c);
    bench_edit_014_trim_silence(c);
    #[cfg(all(feature = "random-generation", feature = "iir-filtering"))]
    {
        bench_edit_015_perturb(c);
        bench_edit_016_perturb_in_place(c);
    }
    bench_edit_017_stack(c);
    bench_edit_018_trim_all_silence(c);
    bench_edit_019_apply_gaussian_noise(c);
    bench_edit_020_apply_random_gain(c);
    #[cfg(all(feature = "transforms", feature = "channels"))]
    bench_edit_021_apply_pitch_shift(c);
}

// ===========================================================================
// Dispatch macros — DTYPES_DEFAULT-typed expansion.
//
// `dispatch_edit_borrow!`: bench a non-mutating `(&audio).op(...)` call on a
//   fresh per-iter sine fixture. We use `iter_batched_ref` because the op
//   does not mutate the buffer, even though it produces a new owned output.
// `dispatch_edit_mut!`: bench an in-place `(&mut audio).op(...)` call. The
//   fixture is built fresh per iteration via `iter_batched_ref` and the
//   routine mutates it in place.
// `dispatch_edit_owned!`: bench a consuming op that takes the buffer by
//   value. Uses `iter_batched` (not `_ref`) with a fresh clone per iter.
// ===========================================================================

macro_rules! dispatch_edit_borrow {
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

macro_rules! dispatch_edit_mut {
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

/// Standard label + per-sample throughput tag.
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

/// Build a mono fixture with embedded silence runs for `trim_all_silence`:
/// `[burst | silence | burst | silence | burst]` where each segment is
/// `n / 5` samples. Yields ~40 % interior silence (two gaps of 20 % each)
/// plus three sine bursts.
fn fixture_with_silence_runs<T>(n: usize, sr: NonZeroU32) -> AudioSamples<'static, T>
where
    T: StandardSample + 'static,
{
    use audio_samples::AudioEditing;

    let seg_samples = n / 5;
    let seg_dur = Duration::from_secs_f64(seg_samples as f64 / f64::from(sr.get()));

    let burst = || sine_wave::<T>(440.0, seg_dur, sr, 1.0);
    let gap = || silence::<T>(seg_dur, sr);

    let segments = vec![burst(), gap(), burst(), gap(), burst()];
    // SAFETY: vec is non-empty by construction.
    let ne = NonEmptyVec::new(segments).expect("non-empty by construction");
    AudioSamples::concatenate_owned(ne).expect("uniform sample rate / channel count")
}

// ===========================================================================
// Edit-001 reverse — NoFast tier; allocates a new buffer.
// ===========================================================================

fn bench_edit_001_reverse<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("edit");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Edit-001_reverse", len, dt, ch);
                dispatch_edit_borrow!(group, id, dt, len, ch, |a| a.reverse());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Edit-002 reverse_in_place — FastSmall tier; pure in-place swap, no alloc.
// ===========================================================================

fn bench_edit_002_reverse_in_place<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("edit");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id =
                    label_and_throughput(&mut group, "Edit-002_reverse_in_place", len, dt, ch);
                dispatch_edit_mut!(group, id, dt, len, ch, |a| a.reverse_in_place().ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Edit-003 trim — FastSmall tier; slice + clone of half the buffer.
// Trim middle 50 % to keep cost predictable across lengths.
// ===========================================================================

fn bench_edit_003_trim<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("edit");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Edit-003_trim", len, dt, ch);
                let dur = len as f64 / 44_100.0;
                let start = dur * 0.25;
                let end = dur * 0.75;
                dispatch_edit_borrow!(group, id, dt, len, ch, |a| a.trim(start, end).ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Edit-004 pad — FastSmall tier; allocates `original_len + pad_len` samples.
// Pad 10 % on each side.
// ===========================================================================

fn bench_edit_004_pad<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("edit");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Edit-004_pad", len, dt, ch);
                let pad_each_side = (len as f64 * 0.1) / 44_100.0;
                match dt {
                    "i16" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || fixture_a440::<i16>(n, ch),
                            |a| {
                                black_box(a.pad(pad_each_side, pad_each_side, 0_i16).ok());
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "I24" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || fixture_a440::<I24>(n, ch),
                            |a| {
                                black_box(
                                    a.pad(pad_each_side, pad_each_side, I24::default()).ok(),
                                );
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "i32" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || fixture_a440::<i32>(n, ch),
                            |a| {
                                black_box(a.pad(pad_each_side, pad_each_side, 0_i32).ok());
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "f32" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || fixture_a440::<f32>(n, ch),
                            |a| {
                                black_box(a.pad(pad_each_side, pad_each_side, 0.0_f32).ok());
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

// ===========================================================================
// Edit-005 pad_samples_right — FastSmall tier; pads to 1.5× original length.
// ===========================================================================

fn bench_edit_005_pad_samples_right<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("edit");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id =
                    label_and_throughput(&mut group, "Edit-005_pad_samples_right", len, dt, ch);
                let target = (len as f64 * 1.5) as usize;
                match dt {
                    "i16" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || fixture_a440::<i16>(n, ch),
                            |a| { black_box(a.pad_samples_right(target, 0_i16).ok()); },
                            BatchSize::LargeInput,
                        );
                    }),
                    "I24" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || fixture_a440::<I24>(n, ch),
                            |a| {
                                black_box(a.pad_samples_right(target, I24::default()).ok());
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "i32" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || fixture_a440::<i32>(n, ch),
                            |a| { black_box(a.pad_samples_right(target, 0_i32).ok()); },
                            BatchSize::LargeInput,
                        );
                    }),
                    "f32" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || fixture_a440::<f32>(n, ch),
                            |a| { black_box(a.pad_samples_right(target, 0.0_f32).ok()); },
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

// ===========================================================================
// Edit-006 pad_to_duration — FastSmall tier; sweep PadSide {Left, Right}.
// Pads to 1.5× original duration.
// ===========================================================================

fn bench_edit_006_pad_to_duration<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("edit");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &(side_lbl, side) in &[("L", PadSide::Left), ("R", PadSide::Right)] {
                for &dt in DTYPES_DEFAULT {
                    let target_dur = (len as f64 * 1.5) / 44_100.0;
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("side", side_lbl)
                        .build();
                    let id = BenchmarkId::new("Edit-006_pad_to_duration", label);
                    match dt {
                        "i16" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                            b.iter_batched_ref(
                                || fixture_a440::<i16>(n, ch),
                                |a| {
                                    black_box(a.pad_to_duration(target_dur, 0_i16, side).ok());
                                },
                                BatchSize::LargeInput,
                            );
                        }),
                        "I24" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                            b.iter_batched_ref(
                                || fixture_a440::<I24>(n, ch),
                                |a| {
                                    black_box(
                                        a.pad_to_duration(target_dur, I24::default(), side)
                                            .ok(),
                                    );
                                },
                                BatchSize::LargeInput,
                            );
                        }),
                        "i32" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                            b.iter_batched_ref(
                                || fixture_a440::<i32>(n, ch),
                                |a| {
                                    black_box(a.pad_to_duration(target_dur, 0_i32, side).ok());
                                },
                                BatchSize::LargeInput,
                            );
                        }),
                        "f32" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                            b.iter_batched_ref(
                                || fixture_a440::<f32>(n, ch),
                                |a| {
                                    black_box(
                                        a.pad_to_duration(target_dur, 0.0_f32, side).ok(),
                                    );
                                },
                                BatchSize::LargeInput,
                            );
                        }),
                        _ => unreachable!(),
                    };
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Edit-007 split — NoFast tier; allocates Vec<AudioSamples> with N clones.
// Split into 8 segments → segment_duration = len / 8 / sr.
// ===========================================================================

fn bench_edit_007_split<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("edit");
    const N_SEGMENTS: usize = 8;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", ch)
                    .with("n_segments", N_SEGMENTS)
                    .build();
                let id = BenchmarkId::new("Edit-007_split", label);
                let seg_dur = (len as f64 / N_SEGMENTS as f64) / 44_100.0;
                dispatch_edit_borrow!(group, id, dt, len, ch, |a| a.split(seg_dur).ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Edit-008 concatenate — NoFast tier; 4 segments of `len / 4` each.
// Joining is a sum_len = len allocation + per-channel copy.
// ===========================================================================

fn bench_edit_008_concatenate<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("edit");
    const N_SEGMENTS: usize = 4;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", ch)
                    .with("n_segments", N_SEGMENTS)
                    .build();
                let id = BenchmarkId::new("Edit-008_concatenate", label);
                let seg_len = len / N_SEGMENTS;
                match dt {
                    "i16" => group.bench_with_input(id, &(seg_len, ch), |b, &(seg, ch)| {
                        b.iter_batched(
                            || {
                                let v: Vec<_> = (0..N_SEGMENTS)
                                    .map(|_| fixture_a440::<i16>(seg, ch))
                                    .collect();
                                NonEmptyVec::new(v).expect("non-empty")
                            },
                            |segs| {
                                black_box(
                                    <AudioSamples<'_, i16> as AudioEditing>::concatenate_owned(segs).ok(),
                                );
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "I24" => group.bench_with_input(id, &(seg_len, ch), |b, &(seg, ch)| {
                        b.iter_batched(
                            || {
                                let v: Vec<_> = (0..N_SEGMENTS)
                                    .map(|_| fixture_a440::<I24>(seg, ch))
                                    .collect();
                                NonEmptyVec::new(v).expect("non-empty")
                            },
                            |segs| {
                                black_box(
                                    <AudioSamples<'_, I24> as AudioEditing>::concatenate_owned(segs).ok(),
                                );
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "i32" => group.bench_with_input(id, &(seg_len, ch), |b, &(seg, ch)| {
                        b.iter_batched(
                            || {
                                let v: Vec<_> = (0..N_SEGMENTS)
                                    .map(|_| fixture_a440::<i32>(seg, ch))
                                    .collect();
                                NonEmptyVec::new(v).expect("non-empty")
                            },
                            |segs| {
                                black_box(
                                    <AudioSamples<'_, i32> as AudioEditing>::concatenate_owned(segs).ok(),
                                );
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "f32" => group.bench_with_input(id, &(seg_len, ch), |b, &(seg, ch)| {
                        b.iter_batched(
                            || {
                                let v: Vec<_> = (0..N_SEGMENTS)
                                    .map(|_| fixture_a440::<f32>(seg, ch))
                                    .collect();
                                NonEmptyVec::new(v).expect("non-empty")
                            },
                            |segs| {
                                black_box(
                                    <AudioSamples<'_, f32> as AudioEditing>::concatenate_owned(segs).ok(),
                                );
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

// ===========================================================================
// Edit-009 concatenate_owned — NoFast tier; consumes the input vec each iter.
// Uses `iter_batched` (not `_ref`) since the vec is moved.
// ===========================================================================

fn bench_edit_009_concatenate_owned<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("edit");
    const N_SEGMENTS: usize = 4;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", ch)
                    .with("n_segments", N_SEGMENTS)
                    .build();
                let id = BenchmarkId::new("Edit-009_concatenate_owned", label);
                let seg_len = len / N_SEGMENTS;
                match dt {
                    "i16" => group.bench_with_input(id, &(seg_len, ch), |b, &(seg, ch)| {
                        b.iter_batched(
                            || {
                                let v: Vec<_> = (0..N_SEGMENTS)
                                    .map(|_| fixture_a440::<i16>(seg, ch))
                                    .collect();
                                NonEmptyVec::new(v).expect("non-empty")
                            },
                            |segs| {
                                black_box(
                                    <AudioSamples<'_, i16> as AudioEditing>::concatenate_owned(
                                        segs,
                                    )
                                    .ok(),
                                );
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "I24" => group.bench_with_input(id, &(seg_len, ch), |b, &(seg, ch)| {
                        b.iter_batched(
                            || {
                                let v: Vec<_> = (0..N_SEGMENTS)
                                    .map(|_| fixture_a440::<I24>(seg, ch))
                                    .collect();
                                NonEmptyVec::new(v).expect("non-empty")
                            },
                            |segs| {
                                black_box(
                                    <AudioSamples<'_, I24> as AudioEditing>::concatenate_owned(
                                        segs,
                                    )
                                    .ok(),
                                );
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "i32" => group.bench_with_input(id, &(seg_len, ch), |b, &(seg, ch)| {
                        b.iter_batched(
                            || {
                                let v: Vec<_> = (0..N_SEGMENTS)
                                    .map(|_| fixture_a440::<i32>(seg, ch))
                                    .collect();
                                NonEmptyVec::new(v).expect("non-empty")
                            },
                            |segs| {
                                black_box(
                                    <AudioSamples<'_, i32> as AudioEditing>::concatenate_owned(
                                        segs,
                                    )
                                    .ok(),
                                );
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "f32" => group.bench_with_input(id, &(seg_len, ch), |b, &(seg, ch)| {
                        b.iter_batched(
                            || {
                                let v: Vec<_> = (0..N_SEGMENTS)
                                    .map(|_| fixture_a440::<f32>(seg, ch))
                                    .collect();
                                NonEmptyVec::new(v).expect("non-empty")
                            },
                            |segs| {
                                black_box(
                                    <AudioSamples<'_, f32> as AudioEditing>::concatenate_owned(
                                        segs,
                                    )
                                    .ok(),
                                );
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

// ===========================================================================
// Edit-010 mix — NoFast tier; 4 sources of equal length, equal weights.
// Per-sample cost is N_SOURCES f64 mul-add; throughput = len*ch (per output
// sample, all sources contribute).
// ===========================================================================

fn bench_edit_010_mix<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("edit");
    const N_SOURCES: usize = 4;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", ch)
                    .with("n_sources", N_SOURCES)
                    .build();
                let id = BenchmarkId::new("Edit-010_mix", label);
                match dt {
                    "i16" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || {
                                let v: Vec<_> = (0..N_SOURCES)
                                    .map(|_| fixture_a440::<i16>(n, ch))
                                    .collect();
                                NonEmptyVec::new(v).expect("non-empty")
                            },
                            |srcs| {
                                let ne = NonEmptySlice::from_slice(srcs.as_slice())
                                    .expect("non-empty");
                                black_box(
                                    <AudioSamples<'_, i16> as AudioEditing>::mix(ne, None).ok(),
                                );
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "I24" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || {
                                let v: Vec<_> = (0..N_SOURCES)
                                    .map(|_| fixture_a440::<I24>(n, ch))
                                    .collect();
                                NonEmptyVec::new(v).expect("non-empty")
                            },
                            |srcs| {
                                let ne = NonEmptySlice::from_slice(srcs.as_slice())
                                    .expect("non-empty");
                                black_box(
                                    <AudioSamples<'_, I24> as AudioEditing>::mix(ne, None).ok(),
                                );
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "i32" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || {
                                let v: Vec<_> = (0..N_SOURCES)
                                    .map(|_| fixture_a440::<i32>(n, ch))
                                    .collect();
                                NonEmptyVec::new(v).expect("non-empty")
                            },
                            |srcs| {
                                let ne = NonEmptySlice::from_slice(srcs.as_slice())
                                    .expect("non-empty");
                                black_box(
                                    <AudioSamples<'_, i32> as AudioEditing>::mix(ne, None).ok(),
                                );
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "f32" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || {
                                let v: Vec<_> = (0..N_SOURCES)
                                    .map(|_| fixture_a440::<f32>(n, ch))
                                    .collect();
                                NonEmptyVec::new(v).expect("non-empty")
                            },
                            |srcs| {
                                let ne = NonEmptySlice::from_slice(srcs.as_slice())
                                    .expect("non-empty");
                                black_box(
                                    <AudioSamples<'_, f32> as AudioEditing>::mix(ne, None).ok(),
                                );
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

// ===========================================================================
// Edit-011 fade_in — NoFast tier; sweep FadeCurve {Linear, SmoothStep}.
// Fade region = 25 % of signal.
// ===========================================================================

fn bench_edit_011_fade_in<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("edit");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &(curve_lbl, curve) in
                &[("linear", FadeCurve::Linear), ("smoothstep", FadeCurve::SmoothStep)]
            {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("curve", curve_lbl)
                        .build();
                    let id = BenchmarkId::new("Edit-011_fade_in", label);
                    let dur = (len as f64 * 0.25) / 44_100.0;
                    dispatch_edit_mut!(
                        group, id, dt, len, ch,
                        |a| a.fade_in(dur, curve).ok()
                    );
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Edit-012 fade_out — NoFast tier; sweep FadeCurve {Linear, SmoothStep}.
// ===========================================================================

fn bench_edit_012_fade_out<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("edit");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &(curve_lbl, curve) in
                &[("linear", FadeCurve::Linear), ("smoothstep", FadeCurve::SmoothStep)]
            {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("curve", curve_lbl)
                        .build();
                    let id = BenchmarkId::new("Edit-012_fade_out", label);
                    let dur = (len as f64 * 0.25) / 44_100.0;
                    dispatch_edit_mut!(
                        group, id, dt, len, ch,
                        |a| a.fade_out(dur, curve).ok()
                    );
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Edit-013 repeat — NoFast tier; cap at LENGTH_SWEEP_NO_XXXL to avoid
// allocating 4 * 4M samples per iter. count = 4.
// ===========================================================================

fn bench_edit_013_repeat<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("edit");
    const COUNT: usize = 4;
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * ch * COUNT) as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", ch)
                    .with("count", COUNT)
                    .build();
                let id = BenchmarkId::new("Edit-013_repeat", label);
                dispatch_edit_borrow!(group, id, dt, len, ch, |a| a.repeat(COUNT).ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Edit-014 trim_silence — NoFast tier; scans both ends of the buffer.
// Uses the standard sine fixture: there is no leading/trailing silence to
// trim, so this measures the best-case full scan cost (the impl scans until
// it finds a non-silent sample at each end). Threshold -120 dB ensures the
// sine survives the scan.
// ===========================================================================

fn bench_edit_014_trim_silence<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("edit");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Edit-014_trim_silence", len, dt, ch);
                dispatch_edit_borrow!(group, id, dt, len, ch, |a| a.trim_silence(-120.0).ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Edit-015 perturb — NoFast tier; clones then perturbs in place.
// Sweeps four PerturbationMethod variants (CATALOG Open Q 17). PitchShift
// is FFT-based and far more expensive than the others — cap its length
// at NO_XXXL to keep per-point under ~30 s.
// ===========================================================================

#[cfg(all(feature = "random-generation", feature = "iir-filtering"))]
fn bench_edit_015_perturb<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("edit");
    let methods: &[(&'static str, PerturbationMethod)] = &[
        (
            "gaussian",
            PerturbationMethod::gaussian_noise(20.0, NoiseColor::White),
        ),
        ("gain", PerturbationMethod::random_gain(-6.0, 6.0)),
        (
            "lowpass",
            PerturbationMethod::low_pass_filter(4000.0, None),
        ),
        #[cfg(all(feature = "transforms", feature = "channels"))]
        ("pitch", PerturbationMethod::pitch_shift(2.0, false)),
    ];

    for (method_lbl, method) in methods.iter().copied() {
        let lengths: &[usize] = if method_lbl == "pitch" {
            LENGTH_SWEEP_NO_XXXL
        } else {
            LENGTH_SWEEP_FULL
        };
        for (i, &len) in lengths.iter().enumerate() {
            group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
            for &ch in CHANNELS_DEFAULT {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("method", method_lbl)
                        .build();
                    let id = BenchmarkId::new("Edit-015_perturb", label);
                    let cfg = PerturbationConfig::with_seed(method, BENCH_RNG_SEED);
                    dispatch_edit_borrow!(group, id, dt, len, ch, |a| a.perturb(&cfg).ok());
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Edit-016 perturb_in_place — NoFast tier; sweep same four methods as 015.
// ===========================================================================

#[cfg(all(feature = "random-generation", feature = "iir-filtering"))]
fn bench_edit_016_perturb_in_place<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("edit");
    let methods: &[(&'static str, PerturbationMethod)] = &[
        (
            "gaussian",
            PerturbationMethod::gaussian_noise(20.0, NoiseColor::White),
        ),
        ("gain", PerturbationMethod::random_gain(-6.0, 6.0)),
        (
            "lowpass",
            PerturbationMethod::low_pass_filter(4000.0, None),
        ),
        #[cfg(all(feature = "transforms", feature = "channels"))]
        ("pitch", PerturbationMethod::pitch_shift(2.0, false)),
    ];

    for (method_lbl, method) in methods.iter().copied() {
        let lengths: &[usize] = if method_lbl == "pitch" {
            LENGTH_SWEEP_NO_XXXL
        } else {
            LENGTH_SWEEP_FULL
        };
        for (i, &len) in lengths.iter().enumerate() {
            group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
            for &ch in CHANNELS_DEFAULT {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("method", method_lbl)
                        .build();
                    let id = BenchmarkId::new("Edit-016_perturb_in_place", label);
                    let cfg = PerturbationConfig::with_seed(method, BENCH_RNG_SEED);
                    dispatch_edit_mut!(
                        group, id, dt, len, ch,
                        |a| a.perturb_in_place(&cfg).ok()
                    );
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Edit-017 stack — NoFast tier; N=2 (stereo) and N=6 (5.1) mono sources.
// Per-iter cost is dominated by the Array2 allocation + N row copies.
// ===========================================================================

fn bench_edit_017_stack<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("edit");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        // Each "source" is mono; n_channels in the output = n_sources.
        for &n_sources in &[2_usize, 6] {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * n_sources) as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", n_sources) // output channels = n_sources
                    .with("n_sources", n_sources)
                    .build();
                let id = BenchmarkId::new("Edit-017_stack", label);
                match dt {
                    "i16" => group.bench_with_input(id, &(len, n_sources), |b, &(n, ns)| {
                        b.iter_batched_ref(
                            || {
                                let v: Vec<_> = (0..ns)
                                    .map(|_| fixture_a440::<i16>(n, 1))
                                    .collect();
                                NonEmptyVec::new(v).expect("non-empty")
                            },
                            |srcs| {
                                let ne = NonEmptySlice::from_slice(srcs.as_slice())
                                    .expect("non-empty");
                                black_box(
                                    <AudioSamples<'_, i16> as AudioEditing>::stack(ne).ok(),
                                );
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "I24" => group.bench_with_input(id, &(len, n_sources), |b, &(n, ns)| {
                        b.iter_batched_ref(
                            || {
                                let v: Vec<_> = (0..ns)
                                    .map(|_| fixture_a440::<I24>(n, 1))
                                    .collect();
                                NonEmptyVec::new(v).expect("non-empty")
                            },
                            |srcs| {
                                let ne = NonEmptySlice::from_slice(srcs.as_slice())
                                    .expect("non-empty");
                                black_box(
                                    <AudioSamples<'_, I24> as AudioEditing>::stack(ne).ok(),
                                );
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "i32" => group.bench_with_input(id, &(len, n_sources), |b, &(n, ns)| {
                        b.iter_batched_ref(
                            || {
                                let v: Vec<_> = (0..ns)
                                    .map(|_| fixture_a440::<i32>(n, 1))
                                    .collect();
                                NonEmptyVec::new(v).expect("non-empty")
                            },
                            |srcs| {
                                let ne = NonEmptySlice::from_slice(srcs.as_slice())
                                    .expect("non-empty");
                                black_box(
                                    <AudioSamples<'_, i32> as AudioEditing>::stack(ne).ok(),
                                );
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "f32" => group.bench_with_input(id, &(len, n_sources), |b, &(n, ns)| {
                        b.iter_batched_ref(
                            || {
                                let v: Vec<_> = (0..ns)
                                    .map(|_| fixture_a440::<f32>(n, 1))
                                    .collect();
                                NonEmptyVec::new(v).expect("non-empty")
                            },
                            |srcs| {
                                let ne = NonEmptySlice::from_slice(srcs.as_slice())
                                    .expect("non-empty");
                                black_box(
                                    <AudioSamples<'_, f32> as AudioEditing>::stack(ne).ok(),
                                );
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

// ===========================================================================
// Edit-018 trim_all_silence — NoFast tier; mono only.
// Custom fixture with two embedded silence runs (~40 % of signal).
// Uses LENGTH_SWEEP_NO_XXXL because at XXXL the fixture construction itself
// (5-segment concat) takes seconds.
// ===========================================================================

fn bench_edit_018_trim_all_silence<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("edit");
    let ch = 1; // mono fixture
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            group.throughput(Throughput::Elements(len as u64));
            let label = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("fixture", "silence_runs")
                .build();
            let id = BenchmarkId::new("Edit-018_trim_all_silence", label);
            // min_silence at ~half a segment so both gaps are removed.
            let min_silence_s = (len as f64 / 5.0 / 2.0) / 44_100.0;
            let sr = sample_rate!(44_100);
            match dt {
                "i16" => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || fixture_with_silence_runs::<i16>(n, sr),
                        |a| {
                            black_box(a.trim_all_silence(-60.0, min_silence_s).ok());
                        },
                        BatchSize::LargeInput,
                    );
                }),
                "I24" => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || fixture_with_silence_runs::<I24>(n, sr),
                        |a| {
                            black_box(a.trim_all_silence(-60.0, min_silence_s).ok());
                        },
                        BatchSize::LargeInput,
                    );
                }),
                "i32" => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || fixture_with_silence_runs::<i32>(n, sr),
                        |a| {
                            black_box(a.trim_all_silence(-60.0, min_silence_s).ok());
                        },
                        BatchSize::LargeInput,
                    );
                }),
                "f32" => group.bench_with_input(id, &len, |b, &n| {
                    b.iter_batched_ref(
                        || fixture_with_silence_runs::<f32>(n, sr),
                        |a| {
                            black_box(a.trim_all_silence(-60.0, min_silence_s).ok());
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
// Edit-019 apply_gaussian_noise_ — NoFast tier; uses a seeded StdRng so
// the noise is deterministic across iters. Free function; bench mono +
// stereo for white noise (Pink/Brown are different cost regimes — left
// to a focused follow-up).
// ===========================================================================

#[cfg(feature = "random-generation")]
fn bench_edit_019_apply_gaussian_noise<M: Measurement>(c: &mut Criterion<M>) {
    use audio_samples::operations::editing::apply_gaussian_noise_;

    let mut group = c.benchmark_group("edit");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", ch)
                    .with("color", "white")
                    .build();
                let id = BenchmarkId::new("Edit-019_apply_gaussian_noise", label);
                match dt {
                    "i16" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || (fixture_a440::<i16>(n, ch), seeded_rng()),
                            |(a, rng)| {
                                black_box(
                                    apply_gaussian_noise_(a, 20.0, NoiseColor::White, rng).ok(),
                                );
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "I24" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || (fixture_a440::<I24>(n, ch), seeded_rng()),
                            |(a, rng)| {
                                black_box(
                                    apply_gaussian_noise_(a, 20.0, NoiseColor::White, rng).ok(),
                                );
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "i32" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || (fixture_a440::<i32>(n, ch), seeded_rng()),
                            |(a, rng)| {
                                black_box(
                                    apply_gaussian_noise_(a, 20.0, NoiseColor::White, rng).ok(),
                                );
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "f32" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || (fixture_a440::<f32>(n, ch), seeded_rng()),
                            |(a, rng)| {
                                black_box(
                                    apply_gaussian_noise_(a, 20.0, NoiseColor::White, rng).ok(),
                                );
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

// Stub used when random-generation is somehow off (shouldn't happen — `editing`
// pulls it in — but keeps the cfg surface tidy).
#[cfg(not(feature = "random-generation"))]
fn bench_edit_019_apply_gaussian_noise<M: Measurement>(_c: &mut Criterion<M>) {}

// ===========================================================================
// Edit-020 apply_random_gain_ — FastSmall tier; single multiply per sample
// after the RNG draw.
// ===========================================================================

#[cfg(feature = "random-generation")]
fn bench_edit_020_apply_random_gain<M: Measurement>(c: &mut Criterion<M>) {
    use audio_samples::operations::editing::apply_random_gain_;

    let mut group = c.benchmark_group("edit");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id =
                    label_and_throughput(&mut group, "Edit-020_apply_random_gain", len, dt, ch);
                match dt {
                    "i16" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || (fixture_a440::<i16>(n, ch), seeded_rng()),
                            |(a, rng)| {
                                apply_random_gain_(a, -6.0, 6.0, rng);
                                black_box(&*a);
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "I24" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || (fixture_a440::<I24>(n, ch), seeded_rng()),
                            |(a, rng)| {
                                apply_random_gain_(a, -6.0, 6.0, rng);
                                black_box(&*a);
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "i32" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || (fixture_a440::<i32>(n, ch), seeded_rng()),
                            |(a, rng)| {
                                apply_random_gain_(a, -6.0, 6.0, rng);
                                black_box(&*a);
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "f32" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || (fixture_a440::<f32>(n, ch), seeded_rng()),
                            |(a, rng)| {
                                apply_random_gain_(a, -6.0, 6.0, rng);
                                black_box(&*a);
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

#[cfg(not(feature = "random-generation"))]
fn bench_edit_020_apply_random_gain<M: Measurement>(_c: &mut Criterion<M>) {}

// ===========================================================================
// Edit-021 apply_pitch_shift_ — NoFast; FFT-based phase vocoder.
// Cap at LENGTH_SWEEP_NO_XXXL (XXXL would take minutes per point).
// Mono only — the impl loops over channels internally so the per-channel
// cost is identical to mono.
// ===========================================================================

#[cfg(all(feature = "transforms", feature = "channels"))]
fn bench_edit_021_apply_pitch_shift<M: Measurement>(c: &mut Criterion<M>) {
    use audio_samples::operations::editing::apply_pitch_shift_;

    let mut group = c.benchmark_group("edit");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        // Skip sizes too small for the STFT (n_fft = 2048).
        if len < 4096 {
            continue;
        }
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &semitones in &[2.0_f64, 7.0] {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements(len as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", ch)
                    .with("semitones", format!("{semitones:.1}"))
                    .build();
                let id = BenchmarkId::new("Edit-021_apply_pitch_shift", label);
                match dt {
                    "i16" => group.bench_with_input(id, &len, |b, &n| {
                        b.iter_batched_ref(
                            || fixture_a440::<i16>(n, ch),
                            |a| {
                                black_box(apply_pitch_shift_(a, semitones, false).ok());
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "I24" => group.bench_with_input(id, &len, |b, &n| {
                        b.iter_batched_ref(
                            || fixture_a440::<I24>(n, ch),
                            |a| {
                                black_box(apply_pitch_shift_(a, semitones, false).ok());
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "i32" => group.bench_with_input(id, &len, |b, &n| {
                        b.iter_batched_ref(
                            || fixture_a440::<i32>(n, ch),
                            |a| {
                                black_box(apply_pitch_shift_(a, semitones, false).ok());
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "f32" => group.bench_with_input(id, &len, |b, &n| {
                        b.iter_batched_ref(
                            || fixture_a440::<f32>(n, ch),
                            |a| {
                                black_box(apply_pitch_shift_(a, semitones, false).ok());
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

