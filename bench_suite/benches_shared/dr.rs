//! Shared body for the `AudioDynamicRange` bench targets
//! (`bench_dr_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/dr.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Catalog: `bench_suite/CATALOG.md` section `DR` (DR-001 .. DR-012).

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use non_empty_slice::NonEmptySlice;
use std::hint::black_box;
use std::num::NonZeroUsize;

use audio_samples::operations::dynamic_range::{
    EnvelopeFollower, LookaheadBuffer, calculate_compression_gain, calculate_limiting_gain,
};
use audio_samples::operations::types::{
    CompressorConfig, DynamicRangeMethod, KneeType, LimiterConfig,
};
use audio_samples::{AudioDynamicRange, I24};

use bench_suite_common::{
    CHANNELS_DEFAULT, DTYPES_DEFAULT, LENGTH_SWEEP_NO_XXXL, ParamLabel, SampleSizePolicy,
    fixture_a440, sample_size_for,
};

// ===========================================================================
// Top-level entry point — wrappers call into this with their typed Criterion.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_dr_001_apply_compressor(c);
    bench_dr_002_apply_limiter(c);
    bench_dr_003_apply_compressor_sidechain(c);
    bench_dr_004_apply_limiter_sidechain(c);
    bench_dr_005_get_compression_curve(c);
    bench_dr_006_get_gain_reduction(c);
    bench_dr_007_apply_gate(c);
    bench_dr_008_apply_expander(c);
    bench_dr_009_envelope_follower_process(c);
    bench_dr_010_lookahead_buffer_process(c);
    bench_dr_011_calculate_compression_gain(c);
    bench_dr_012_calculate_limiting_gain(c);
}

// ===========================================================================
// Dispatch macros — DTYPES_DEFAULT-typed expansion. The shape here is
// `iter_batched_ref` with a fresh fixture per batch so the in-place mutation
// performed by `apply_*` doesn't leak across iters.
// ===========================================================================

macro_rules! dispatch_compressor_inplace {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, $cfg:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        let cfg = $cfg;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |a| {
                        let _ = black_box(a.apply_compressor(&cfg));
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<I24>(n, ch),
                    |a| {
                        let _ = black_box(a.apply_compressor(&cfg));
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i32>(n, ch),
                    |a| {
                        let _ = black_box(a.apply_compressor(&cfg));
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |a| {
                        let _ = black_box(a.apply_compressor(&cfg));
                    },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

macro_rules! dispatch_limiter_inplace {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, $cfg:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        let cfg = $cfg;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |a| {
                        let _ = black_box(a.apply_limiter(&cfg));
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<I24>(n, ch),
                    |a| {
                        let _ = black_box(a.apply_limiter(&cfg));
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i32>(n, ch),
                    |a| {
                        let _ = black_box(a.apply_limiter(&cfg));
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |a| {
                        let _ = black_box(a.apply_limiter(&cfg));
                    },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

// Sidechain APIs are mono-only; the second input is a 880 Hz sine to differ
// from the primary 440 Hz one.
macro_rules! dispatch_compressor_sidechain {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $cfg:expr) => {{
        let id = $id;
        let n = $n;
        let cfg = $cfg;
        match $dt {
            "i16" => $group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched_ref(
                    || (fixture_a440::<i16>(n, 1), make_sidechain::<i16>(n)),
                    |inp| {
                        let (a, sc) = inp;
                        let _ = black_box(a.apply_compressor_sidechain(&cfg, &*sc));
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched_ref(
                    || (fixture_a440::<I24>(n, 1), make_sidechain::<I24>(n)),
                    |inp| {
                        let (a, sc) = inp;
                        let _ = black_box(a.apply_compressor_sidechain(&cfg, &*sc));
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched_ref(
                    || (fixture_a440::<i32>(n, 1), make_sidechain::<i32>(n)),
                    |inp| {
                        let (a, sc) = inp;
                        let _ = black_box(a.apply_compressor_sidechain(&cfg, &*sc));
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched_ref(
                    || (fixture_a440::<f32>(n, 1), make_sidechain::<f32>(n)),
                    |inp| {
                        let (a, sc) = inp;
                        let _ = black_box(a.apply_compressor_sidechain(&cfg, &*sc));
                    },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

macro_rules! dispatch_limiter_sidechain {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $cfg:expr) => {{
        let id = $id;
        let n = $n;
        let cfg = $cfg;
        match $dt {
            "i16" => $group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched_ref(
                    || (fixture_a440::<i16>(n, 1), make_sidechain::<i16>(n)),
                    |inp| {
                        let (a, sc) = inp;
                        let _ = black_box(a.apply_limiter_sidechain(&cfg, &*sc));
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched_ref(
                    || (fixture_a440::<I24>(n, 1), make_sidechain::<I24>(n)),
                    |inp| {
                        let (a, sc) = inp;
                        let _ = black_box(a.apply_limiter_sidechain(&cfg, &*sc));
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched_ref(
                    || (fixture_a440::<i32>(n, 1), make_sidechain::<i32>(n)),
                    |inp| {
                        let (a, sc) = inp;
                        let _ = black_box(a.apply_limiter_sidechain(&cfg, &*sc));
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &n, |b, &n| {
                b.iter_batched_ref(
                    || (fixture_a440::<f32>(n, 1), make_sidechain::<f32>(n)),
                    |inp| {
                        let (a, sc) = inp;
                        let _ = black_box(a.apply_limiter_sidechain(&cfg, &*sc));
                    },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

// Helper: build an 880 Hz mono sidechain to pair with the 440 Hz primary.
fn make_sidechain<T>(n: usize) -> audio_samples::AudioSamples<'static, T>
where
    T: audio_samples::StandardSample + 'static,
{
    use audio_samples::sample_rate;
    use audio_samples::utils::generation::sine_wave;
    use std::time::Duration;
    sine_wave::<T>(
        880.0,
        Duration::from_secs_f64(n as f64 / 44_100.0),
        sample_rate!(44_100),
        1.0,
    )
}

// ===========================================================================
// DR-001 apply_compressor — NoFast; in-place; sweep DetectionMethod.
//
// `ratio` and `attack_ms` do not change per-sample cost (just FP constants),
// so we hold them at a representative pair (ratio=4, attack=5 ms) rather
// than sweeping them. DetectionMethod genuinely varies cost (Rms / Hybrid
// maintain a VecDeque-backed window).
// ===========================================================================

fn bench_dr_001_apply_compressor<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("dr");

    let methods = [
        ("peak", DynamicRangeMethod::Peak),
        ("rms", DynamicRangeMethod::Rms),
        ("hybrid", DynamicRangeMethod::Hybrid),
    ];

    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                for (method_label, method) in methods.iter().copied() {
                    let mut cfg = CompressorConfig::default();
                    cfg.threshold_db = -18.0;
                    cfg.ratio = 4.0;
                    cfg.attack_ms = 5.0;
                    cfg.release_ms = 100.0;
                    cfg.makeup_gain_db = 0.0;
                    cfg.knee_type = KneeType::Soft;
                    cfg.knee_width_db = 2.0;
                    cfg.detection_method = method;
                    cfg.lookahead_ms = 0.0;
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("method", method_label)
                        .build();
                    let id = BenchmarkId::new("DR-001_apply_compressor", label);
                    dispatch_compressor_inplace!(group, id, dt, len, ch, cfg);
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// DR-002 apply_limiter — NoFast; in-place; single representative
// (transparent preset, with 5 ms lookahead).
// ===========================================================================

fn bench_dr_002_apply_limiter<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("dr");
    let cfg = LimiterConfig::transparent();
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", ch)
                    .with("lookahead_ms", format!("{:.1}", cfg.lookahead_ms))
                    .build();
                let id = BenchmarkId::new("DR-002_apply_limiter", label);
                dispatch_limiter_inplace!(group, id, dt, len, ch, cfg);
            }
        }
    }
    group.finish();
}

// ===========================================================================
// DR-003 apply_compressor_sidechain — NoFast; mono-only; sidechain enabled.
// ===========================================================================

fn bench_dr_003_apply_compressor_sidechain<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("dr");
    let mut cfg = CompressorConfig::new();
    cfg.side_chain.enable();
    cfg.side_chain.high_pass_freq = None;
    cfg.side_chain.low_pass_freq = None;
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        let ch = 1usize;
        for &dt in DTYPES_DEFAULT {
            group.throughput(Throughput::Elements((len * ch) as u64));
            let label = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("lookahead_ms", format!("{:.1}", cfg.lookahead_ms))
                .build();
            let id = BenchmarkId::new("DR-003_apply_compressor_sidechain", label);
            dispatch_compressor_sidechain!(group, id, dt, len, cfg);
        }
    }
    group.finish();
}

// ===========================================================================
// DR-004 apply_limiter_sidechain — NoFast; mono-only; sidechain enabled.
// ===========================================================================

fn bench_dr_004_apply_limiter_sidechain<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("dr");
    let mut cfg = LimiterConfig::default();
    cfg.side_chain.enable();
    cfg.side_chain.high_pass_freq = None;
    cfg.side_chain.low_pass_freq = None;
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        let ch = 1usize;
        for &dt in DTYPES_DEFAULT {
            group.throughput(Throughput::Elements((len * ch) as u64));
            let label = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("lookahead_ms", format!("{:.1}", cfg.lookahead_ms))
                .build();
            let id = BenchmarkId::new("DR-004_apply_limiter_sidechain", label);
            dispatch_limiter_sidechain!(group, id, dt, len, cfg);
        }
    }
    group.finish();
}

// ===========================================================================
// DR-005 get_compression_curve — NoFast; analytical curve sampling.
// Cost is O(n_points); audio length is irrelevant. Sweep `n_points`.
// ===========================================================================

fn bench_dr_005_get_compression_curve<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("dr");
    let cfg = CompressorConfig::new();
    let ch = 1usize;
    let dt = "f32";
    let audio_len = 1024usize;
    // n_points sweep — log-spaced, cheap analytical fn.
    let n_points_sweep: &[usize] = &[16, 64, 256, 1024, 4096, 16_384];
    for (i, &n_points) in n_points_sweep.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        let levels: Vec<f64> = (0..n_points)
            .map(|k| -60.0 + (k as f64) * 60.0 / (n_points as f64))
            .collect();
        group.throughput(Throughput::Elements(n_points as u64));
        let label = ParamLabel::new()
            .with("n_points", n_points)
            .with("dt", dt)
            .with("ch", ch)
            .build();
        let id = BenchmarkId::new("DR-005_get_compression_curve", label);
        group.bench_with_input(id, &n_points, |b, &_np| {
            b.iter_batched_ref(
                || fixture_a440::<f32>(audio_len, ch),
                |a| {
                    let nes = NonEmptySlice::new(&levels).expect("non-empty levels");
                    black_box(a.get_compression_curve(&cfg, nes).ok());
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// ===========================================================================
// DR-006 get_gain_reduction — NoFast; reads audio, writes Vec<f64>.
// ===========================================================================

fn bench_dr_006_get_gain_reduction<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("dr");
    let cfg = CompressorConfig::new();
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", ch)
                    .build();
                let id = BenchmarkId::new("DR-006_get_gain_reduction", label);
                match dt {
                    "i16" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || fixture_a440::<i16>(n, ch),
                            |a| {
                                black_box(a.get_gain_reduction(&cfg).ok());
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "I24" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || fixture_a440::<I24>(n, ch),
                            |a| {
                                black_box(a.get_gain_reduction(&cfg).ok());
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "i32" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || fixture_a440::<i32>(n, ch),
                            |a| {
                                black_box(a.get_gain_reduction(&cfg).ok());
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "f32" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || fixture_a440::<f32>(n, ch),
                            |a| {
                                black_box(a.get_gain_reduction(&cfg).ok());
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
// DR-007 apply_gate — NoFast; in-place; representative gate config.
// ===========================================================================

fn bench_dr_007_apply_gate<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("dr");
    let threshold_db = -40.0_f64;
    let ratio = 10.0_f64;
    let attack_ms = 1.0_f64;
    let release_ms = 50.0_f64;
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", ch)
                    .build();
                let id = BenchmarkId::new("DR-007_apply_gate", label);
                match dt {
                    "i16" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || fixture_a440::<i16>(n, ch),
                            |a| {
                                let _ = black_box(a.apply_gate(
                                    threshold_db,
                                    ratio,
                                    attack_ms,
                                    release_ms,
                                ));
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "I24" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || fixture_a440::<I24>(n, ch),
                            |a| {
                                let _ = black_box(a.apply_gate(
                                    threshold_db,
                                    ratio,
                                    attack_ms,
                                    release_ms,
                                ));
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "i32" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || fixture_a440::<i32>(n, ch),
                            |a| {
                                let _ = black_box(a.apply_gate(
                                    threshold_db,
                                    ratio,
                                    attack_ms,
                                    release_ms,
                                ));
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "f32" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || fixture_a440::<f32>(n, ch),
                            |a| {
                                let _ = black_box(a.apply_gate(
                                    threshold_db,
                                    ratio,
                                    attack_ms,
                                    release_ms,
                                ));
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
// DR-008 apply_expander — NoFast; in-place; representative config.
//
// `ratio` and `attack_ms` do not change per-sample cost in the expander
// implementation; we hold them at a representative pair.
// ===========================================================================

fn bench_dr_008_apply_expander<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("dr");
    let threshold_db = -20.0_f64;
    let ratio = 4.0_f64;
    let attack_ms = 5.0_f64;
    let release_ms = 100.0_f64;
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", ch)
                    .build();
                let id = BenchmarkId::new("DR-008_apply_expander", label);
                match dt {
                    "i16" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || fixture_a440::<i16>(n, ch),
                            |a| {
                                let _ = black_box(a.apply_expander(
                                    threshold_db,
                                    ratio,
                                    attack_ms,
                                    release_ms,
                                ));
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "I24" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || fixture_a440::<I24>(n, ch),
                            |a| {
                                let _ = black_box(a.apply_expander(
                                    threshold_db,
                                    ratio,
                                    attack_ms,
                                    release_ms,
                                ));
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "i32" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || fixture_a440::<i32>(n, ch),
                            |a| {
                                let _ = black_box(a.apply_expander(
                                    threshold_db,
                                    ratio,
                                    attack_ms,
                                    release_ms,
                                ));
                            },
                            BatchSize::LargeInput,
                        );
                    }),
                    "f32" => group.bench_with_input(id, &(len, ch), |b, &(n, ch)| {
                        b.iter_batched_ref(
                            || fixture_a440::<f32>(n, ch),
                            |a| {
                                let _ = black_box(a.apply_expander(
                                    threshold_db,
                                    ratio,
                                    attack_ms,
                                    release_ms,
                                ));
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
// DR-009 EnvelopeFollower::process — per-sample atom; bench process_block-
// style over an f64 buffer, sweep length and DetectionMethod.
// ===========================================================================

fn bench_dr_009_envelope_follower_process<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("dr");
    let sample_rate = 44_100.0_f64;
    let attack_ms = 5.0_f64;
    let release_ms = 50.0_f64;
    let methods = [
        ("peak", DynamicRangeMethod::Peak),
        ("rms", DynamicRangeMethod::Rms),
        ("hybrid", DynamicRangeMethod::Hybrid),
    ];
    let dt = "f64";
    let ch = 1usize;
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for (method_label, method) in methods.iter().copied() {
            group.throughput(Throughput::Elements(len as u64));
            let label = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("method", method_label)
                .build();
            let id = BenchmarkId::new("DR-009_envelope_follower_process", label);
            group.bench_with_input(id, &len, |b, &n| {
                b.iter_batched_ref(
                    || {
                        // f64 sine buffer + fresh follower per batch
                        let buf: Vec<f64> = (0..n)
                            .map(|k| {
                                let phase = 2.0 * std::f64::consts::PI * 440.0 * (k as f64)
                                    / sample_rate;
                                phase.sin()
                            })
                            .collect();
                        let follower =
                            EnvelopeFollower::new(attack_ms, release_ms, sample_rate, method);
                        (buf, follower)
                    },
                    |inp| {
                        let (buf, follower) = inp;
                        let mut acc = 0.0_f64;
                        for &x in buf.iter() {
                            acc += follower.process(x, method);
                        }
                        black_box(acc);
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

// ===========================================================================
// DR-010 LookaheadBuffer::process — per-sample atom; bench over an f64
// buffer, sweep buffer length and `lookahead_samples`.
// ===========================================================================

fn bench_dr_010_lookahead_buffer_process<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("dr");
    let sample_rate = 44_100.0_f64;
    let dt = "f64";
    let ch = 1usize;
    // Lookahead sweep: small (1 sample) up to ~20 ms @ 44.1 kHz (882 samples).
    let lookahead_sweep: &[usize] = &[1, 64, 441, 882];
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &lh in lookahead_sweep {
            group.throughput(Throughput::Elements(len as u64));
            let label = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("lookahead_samples", lh)
                .build();
            let id = BenchmarkId::new("DR-010_lookahead_buffer_process", label);
            group.bench_with_input(id, &len, |b, &n| {
                b.iter_batched_ref(
                    || {
                        let buf: Vec<f64> = (0..n)
                            .map(|k| {
                                let phase = 2.0 * std::f64::consts::PI * 440.0 * (k as f64)
                                    / sample_rate;
                                phase.sin()
                            })
                            .collect();
                        let lab = LookaheadBuffer::new(NonZeroUsize::new(lh).unwrap());
                        (buf, lab)
                    },
                    |inp| {
                        let (buf, lab) = inp;
                        let mut acc = 0.0_f64;
                        for &x in buf.iter() {
                            acc += lab.process(x);
                        }
                        black_box(acc);
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

// ===========================================================================
// DR-011 calculate_compression_gain — scalar inner kernel; bench in a loop
// over an f64 buffer of input levels, sweep KneeType.
// ===========================================================================

fn bench_dr_011_calculate_compression_gain<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("dr");
    let threshold_db = -18.0_f64;
    let ratio = 4.0_f64;
    let knee_width_db = 4.0_f64;
    let knees = [("hard", KneeType::Hard), ("soft", KneeType::Soft)];
    let dt = "f64";
    let ch = 1usize;
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for (knee_label, knee) in knees.iter().copied() {
            group.throughput(Throughput::Elements(len as u64));
            let label = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("knee", knee_label)
                .build();
            let id = BenchmarkId::new("DR-011_calculate_compression_gain", label);
            group.bench_with_input(id, &len, |b, &n| {
                b.iter_batched_ref(
                    || {
                        // dB levels spanning the threshold so we exercise
                        // both the no-overshoot early-out and the
                        // ratio-applied branch.
                        let buf: Vec<f64> = (0..n)
                            .map(|k| -36.0 + (k as f64) * 36.0 / (n as f64))
                            .collect();
                        buf
                    },
                    |buf| {
                        let mut acc = 0.0_f64;
                        for &x in buf.iter() {
                            acc += calculate_compression_gain(
                                x,
                                threshold_db,
                                ratio,
                                knee,
                                knee_width_db,
                            );
                        }
                        black_box(acc);
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

// ===========================================================================
// DR-012 calculate_limiting_gain — scalar inner kernel; bench in a loop
// over an f64 buffer of input levels, sweep KneeType.
// ===========================================================================

fn bench_dr_012_calculate_limiting_gain<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("dr");
    let ceiling_db = -1.0_f64;
    let knee_width_db = 2.0_f64;
    let knees = [("hard", KneeType::Hard), ("soft", KneeType::Soft)];
    let dt = "f64";
    let ch = 1usize;
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for (knee_label, knee) in knees.iter().copied() {
            group.throughput(Throughput::Elements(len as u64));
            let label = ParamLabel::new()
                .with("len", len)
                .with("dt", dt)
                .with("ch", ch)
                .with("knee", knee_label)
                .build();
            let id = BenchmarkId::new("DR-012_calculate_limiting_gain", label);
            group.bench_with_input(id, &len, |b, &n| {
                b.iter_batched_ref(
                    || {
                        let buf: Vec<f64> = (0..n)
                            .map(|k| -6.0 + (k as f64) * 12.0 / (n as f64))
                            .collect();
                        buf
                    },
                    |buf| {
                        let mut acc = 0.0_f64;
                        for &x in buf.iter() {
                            acc += calculate_limiting_gain(x, ceiling_db, knee, knee_width_db);
                        }
                        black_box(acc);
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}
