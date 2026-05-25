//! Shared body for the `AudioStatistics` bench targets
//! (`bench_stats_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/stats.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `Stats`.

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::num::NonZeroUsize;

use audio_samples::{AudioStatistics, I24};

use bench_suite_common::{
    CHANNELS_DEFAULT, DTYPES_DEFAULT, LENGTH_SWEEP_FULL, LENGTH_SWEEP_NO_XXXL,
    ParamLabel, SampleSizePolicy, fixture_a440, sample_size_for,
};

// ===========================================================================
// Top-level entry point — wrappers call into this with their typed Criterion.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_stats_001_peak(c);
    bench_stats_002_amplitude(c);
    bench_stats_003_min_sample(c);
    bench_stats_004_max_sample(c);
    bench_stats_005_mean(c);
    bench_stats_006_median(c);
    bench_stats_007_rms(c);
    bench_stats_008_rms_and_peak(c);
    bench_stats_009_variance(c);
    bench_stats_010_std_dev(c);
    bench_stats_011_zero_crossings(c);
    bench_stats_012_zero_crossing_rate(c);
    bench_stats_013_cross_correlation(c);
    #[cfg(feature = "transforms")]
    {
        bench_stats_014_spectral_centroid(c);
        bench_stats_015_autocorrelation(c);
        bench_stats_016_spectral_rolloff(c);
    }
}

// ===========================================================================
// Dispatch macros — DTYPES_DEFAULT-typed expansion.
//
// `dispatch_unary_sine!`: bench a single `audio.op()` style call on a 440 Hz
// fixture, swept over the four default dtypes.
//
// `dispatch_xcorr_sine!`: bench a binary `a.cross_correlation(o, max_lag)`
// style call with a 440 Hz `a` and an 880 Hz `o`.
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

macro_rules! dispatch_xcorr_sine {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, $max_lag:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        let max_lag: NonZeroUsize = $max_lag;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch, max_lag), |b, &(n, ch, ml)| {
                b.iter_batched_ref(
                    || (fixture_a440::<i16>(n, ch), {
                        use audio_samples::utils::generation::sine_wave;
                        use audio_samples::sample_rate;
                        use std::time::Duration;
                        sine_wave::<i16>(880.0,
                            Duration::from_secs_f64(n as f64 / 44_100.0),
                            sample_rate!(44_100), 1.0)
                    }),
                    |inp| {
                        let (a, o) = inp;
                        black_box(a.cross_correlation(&*o, ml).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch, max_lag), |b, &(n, ch, ml)| {
                b.iter_batched_ref(
                    || (fixture_a440::<I24>(n, ch), {
                        use audio_samples::utils::generation::sine_wave;
                        use audio_samples::sample_rate;
                        use std::time::Duration;
                        sine_wave::<I24>(880.0,
                            Duration::from_secs_f64(n as f64 / 44_100.0),
                            sample_rate!(44_100), 1.0)
                    }),
                    |inp| {
                        let (a, o) = inp;
                        black_box(a.cross_correlation(&*o, ml).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch, max_lag), |b, &(n, ch, ml)| {
                b.iter_batched_ref(
                    || (fixture_a440::<i32>(n, ch), {
                        use audio_samples::utils::generation::sine_wave;
                        use audio_samples::sample_rate;
                        use std::time::Duration;
                        sine_wave::<i32>(880.0,
                            Duration::from_secs_f64(n as f64 / 44_100.0),
                            sample_rate!(44_100), 1.0)
                    }),
                    |inp| {
                        let (a, o) = inp;
                        black_box(a.cross_correlation(&*o, ml).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch, max_lag), |b, &(n, ch, ml)| {
                b.iter_batched_ref(
                    || (fixture_a440::<f32>(n, ch), {
                        use audio_samples::utils::generation::sine_wave;
                        use audio_samples::sample_rate;
                        use std::time::Duration;
                        sine_wave::<f32>(880.0,
                            Duration::from_secs_f64(n as f64 / 44_100.0),
                            sample_rate!(44_100), 1.0)
                    }),
                    |inp| {
                        let (a, o) = inp;
                        black_box(a.cross_correlation(&*o, ml).ok());
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
        .with("len", n).with("dt", dt).with("ch", ch).build();
    BenchmarkId::new(catalog_id, label)
}

// ===========================================================================
// Stats-001 peak — FastSmall tier; AVX2 fast path on f32 expected.
// ===========================================================================

fn bench_stats_001_peak<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("stats");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Stats-001_peak", len, dt, ch);
                dispatch_unary_sine!(group, id, dt, len, ch, |a| a.peak());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Stats-002 amplitude — default-impl alias; single point.
// ===========================================================================

fn bench_stats_002_amplitude<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("stats");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    let len = 65_536;
    let ch = 1;
    let dt = "f32";
    let id = label_and_throughput(&mut group, "Stats-002_amplitude", len, dt, ch);
    dispatch_unary_sine!(group, id, dt, len, ch, |a| a.amplitude());
    group.finish();
}

// ===========================================================================
// Stats-003 min_sample — FastSmall tier.
// ===========================================================================

fn bench_stats_003_min_sample<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("stats");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Stats-003_min_sample", len, dt, ch);
                dispatch_unary_sine!(group, id, dt, len, ch, |a| a.min_sample());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Stats-004 max_sample — FastSmall tier.
// ===========================================================================

fn bench_stats_004_max_sample<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("stats");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Stats-004_max_sample", len, dt, ch);
                dispatch_unary_sine!(group, id, dt, len, ch, |a| a.max_sample());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Stats-005 mean — FastSmall tier (single-pass sum/count).
// ===========================================================================

fn bench_stats_005_mean<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("stats");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Stats-005_mean", len, dt, ch);
                dispatch_unary_sine!(group, id, dt, len, ch, |a| a.mean());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Stats-006 median — mono only; NoFast tier (selection by index).
// ===========================================================================

fn bench_stats_006_median<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("stats");
    let ch = 1; // returns None for multi-channel
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Stats-006_median", len, dt, ch);
            dispatch_unary_sine!(group, id, dt, len, ch, |a| a.median());
        }
    }
    group.finish();
}

// ===========================================================================
// Stats-007 rms — NoFast tier (two-pass over the buffer).
// ===========================================================================

fn bench_stats_007_rms<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("stats");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Stats-007_rms", len, dt, ch);
                dispatch_unary_sine!(group, id, dt, len, ch, |a| a.rms());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Stats-008 rms_and_peak — NoFast; combines two passes from Stats-001 + 007.
// ===========================================================================

fn bench_stats_008_rms_and_peak<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("stats");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Stats-008_rms_and_peak", len, dt, ch);
                dispatch_unary_sine!(group, id, dt, len, ch, |a| a.rms_and_peak());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Stats-009 variance — NoFast tier.
// ===========================================================================

fn bench_stats_009_variance<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("stats");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Stats-009_variance", len, dt, ch);
                dispatch_unary_sine!(group, id, dt, len, ch, |a| a.variance());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Stats-010 std_dev — default-impl `sqrt(variance)`; single point.
// ===========================================================================

fn bench_stats_010_std_dev<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("stats");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    let len = 65_536;
    let ch = 1;
    let dt = "f32";
    let id = label_and_throughput(&mut group, "Stats-010_std_dev", len, dt, ch);
    dispatch_unary_sine!(group, id, dt, len, ch, |a| a.std_dev());
    group.finish();
}

// ===========================================================================
// Stats-011 zero_crossings — FastSmall tier.
// ===========================================================================

fn bench_stats_011_zero_crossings<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("stats");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Stats-011_zero_crossings", len, dt, ch);
                dispatch_unary_sine!(group, id, dt, len, ch, |a| a.zero_crossings());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Stats-012 zero_crossing_rate — default-impl wrapper; single point.
// ===========================================================================

fn bench_stats_012_zero_crossing_rate<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("stats");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    let len = 65_536;
    let ch = 1;
    let dt = "f32";
    let id = label_and_throughput(&mut group, "Stats-012_zero_crossing_rate", len, dt, ch);
    dispatch_unary_sine!(group, id, dt, len, ch, |a| a.zero_crossing_rate());
    group.finish();
}

// ===========================================================================
// Stats-013 cross_correlation — NoFast tier; extra max_lag axis;
// two inputs (440 Hz primary, 880 Hz secondary).
// ===========================================================================

fn bench_stats_013_cross_correlation<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("stats");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        // cross_correlation auto-broadcasts but expects same channel count both sides.
        // Bench mono only here — multi-channel xcorr is a different cost regime that
        // the trait routes through the same code path on first channels.
        let ch = 1;
        for &max_lag_div in &[64usize, 16, 4] {
            let max_lag = NonZeroUsize::new((len / max_lag_div).max(1)).unwrap();
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len).with("dt", dt).with("ch", ch)
                    .with("max_lag", max_lag.get())
                    .build();
                let id = BenchmarkId::new("Stats-013_cross_correlation", label);
                dispatch_xcorr_sine!(group, id, dt, len, ch, max_lag);
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Stats-014 spectral_centroid — transforms-gated, mono only, lengths
// capped at XXL.
// ===========================================================================

#[cfg(feature = "transforms")]
fn bench_stats_014_spectral_centroid<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("stats");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Stats-014_spectral_centroid", len, dt, ch);
            dispatch_unary_sine!(group, id, dt, len, ch, |a| a.spectral_centroid().ok());
        }
    }
    group.finish();
}

// ===========================================================================
// Stats-015 autocorrelation — transforms-gated, lengths capped at XXL,
// max_lag ∈ {len/16, len/4}. Multi-channel supported (uses first channel).
// ===========================================================================

#[cfg(feature = "transforms")]
fn bench_stats_015_autocorrelation<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("stats");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &max_lag_div in &[16usize, 4] {
                let max_lag = NonZeroUsize::new((len / max_lag_div).max(1)).unwrap();
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len).with("dt", dt).with("ch", ch)
                        .with("max_lag", max_lag.get())
                        .build();
                    let id = BenchmarkId::new("Stats-015_autocorrelation", label);
                    dispatch_unary_sine!(group, id, dt, len, ch, |a| a.autocorrelation(max_lag));
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Stats-016 spectral_rolloff — transforms-gated, mono only, lengths
// capped at XXL, p ∈ {0.85, 0.95}.
// ===========================================================================

#[cfg(feature = "transforms")]
fn bench_stats_016_spectral_rolloff<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("stats");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &p in &[0.85_f64, 0.95] {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements(len as u64));
                let label = ParamLabel::new()
                    .with("len", len).with("dt", dt).with("ch", ch)
                    .with("p", format!("{p:.2}"))
                    .build();
                let id = BenchmarkId::new("Stats-016_spectral_rolloff", label);
                dispatch_unary_sine!(group, id, dt, len, ch, |a| a.spectral_rolloff(p).ok());
            }
        }
    }
    group.finish();
}
