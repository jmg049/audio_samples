//! Shared body for the `utils::comparison` bench targets
//! (`bench_comp_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/comp.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `Comp`
//! (Comp-001 .. Comp-004).

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::num::NonZeroU32;
use std::time::Duration;

use audio_samples::utils::comparison::{align_signals, correlation, mse, snr};
use audio_samples::utils::generation::{sine_wave, stereo_sine_wave};
use audio_samples::{AudioSamples, I24, StandardSample};

use bench_suite_common::{
    BENCH_SAMPLE_RATE_HZ, CHANNELS_DEFAULT, DTYPES_DEFAULT, LENGTH_SWEEP_FULL, ParamLabel,
    SampleSizePolicy, fixture_a440, sample_size_for,
};

// ===========================================================================
// Top-level entry point.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_comp_001_correlation(c);
    bench_comp_002_mse(c);
    bench_comp_003_snr(c);
    bench_comp_004_align_signals(c);
}

// ===========================================================================
// Fixture helper — build a second 880 Hz signal of the right dtype/channel
// count to compare against the 440 Hz primary fixture.
// ===========================================================================

fn fixture_b880<T>(n_samples: usize, channels: usize) -> AudioSamples<'static, T>
where
    T: StandardSample + 'static,
{
    // SAFETY: BENCH_SAMPLE_RATE_HZ is a non-zero constant.
    let sr = unsafe { NonZeroU32::new_unchecked(BENCH_SAMPLE_RATE_HZ) };
    let duration = Duration::from_secs_f64(n_samples as f64 / f64::from(BENCH_SAMPLE_RATE_HZ));
    match channels {
        1 => sine_wave::<T>(880.0, duration, sr, 1.0),
        2 => stereo_sine_wave::<T>(880.0, duration, sr, 1.0),
        n => panic!("fixture_b880: channel count {n} not supported (only 1 or 2)."),
    }
}

// ===========================================================================
// Dispatch macros — DTYPES_DEFAULT-typed expansion for `(a, b) -> scalar`
// comparison functions and for the two-output `align_signals` call.
// ===========================================================================

macro_rules! dispatch_binary_sine {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, |$a:ident, $b:ident| $body:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |bencher, &(n, ch)| {
                bencher.iter_batched_ref(
                    || (fixture_a440::<i16>(n, ch), fixture_b880::<i16>(n, ch)),
                    |inp| {
                        let ($a, $b) = inp;
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |bencher, &(n, ch)| {
                bencher.iter_batched_ref(
                    || (fixture_a440::<I24>(n, ch), fixture_b880::<I24>(n, ch)),
                    |inp| {
                        let ($a, $b) = inp;
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |bencher, &(n, ch)| {
                bencher.iter_batched_ref(
                    || (fixture_a440::<i32>(n, ch), fixture_b880::<i32>(n, ch)),
                    |inp| {
                        let ($a, $b) = inp;
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |bencher, &(n, ch)| {
                bencher.iter_batched_ref(
                    || (fixture_a440::<f32>(n, ch), fixture_b880::<f32>(n, ch)),
                    |inp| {
                        let ($a, $b) = inp;
                        black_box($body);
                    },
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
// Comp-001 correlation — Pearson correlation; NoFast tier (two-pass:
// mean then sum-of-products).
// ===========================================================================

fn bench_comp_001_correlation<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("comp");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Comp-001_correlation", len, dt, ch);
                dispatch_binary_sine!(group, id, dt, len, ch, |a, b| {
                    correlation(&*a, &*b).ok()
                });
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Comp-002 mse — mean squared error; NoFast tier (single-pass squared diff).
// ===========================================================================

fn bench_comp_002_mse<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("comp");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Comp-002_mse", len, dt, ch);
                dispatch_binary_sine!(group, id, dt, len, ch, |a, b| mse(&*a, &*b).ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Comp-003 snr — signal-to-noise ratio in dB; NoFast tier.
// ===========================================================================

fn bench_comp_003_snr<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("comp");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Comp-003_snr", len, dt, ch);
                dispatch_binary_sine!(group, id, dt, len, ch, |a, b| snr(&*a, &*b).ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Comp-004 align_signals — cross-correlation search up to len/2 offsets.
// O(n²) in worst case — sweep a deliberately small length subset so the
// largest point stays under ~30 s per dtype/channel.
// ===========================================================================

// Custom small length sweep for the O(n²) `align_signals` function.
// At len = 16_384 the inner loop performs ~67 M correlation operations
// (worst case), which is already in the seconds-per-iter regime for f64
// arithmetic. Anything beyond would dominate the bench wall-clock.
const COMP_ALIGN_LENGTHS: &[usize] = &[256, 1024, 4096, 16_384];

fn bench_comp_004_align_signals<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("comp");
    for (i, &len) in COMP_ALIGN_LENGTHS.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Comp-004_align_signals", len, dt, ch);
                dispatch_binary_sine!(group, id, dt, len, ch, |a, b| {
                    align_signals(&*a, &*b).ok()
                });
            }
        }
    }
    group.finish();
}
