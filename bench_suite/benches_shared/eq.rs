//! Shared body for the `AudioParametricEq` bench targets
//! (`bench_eq_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/eq.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `EQ`
//! (rows EQ-001 .. EQ-007).

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use audio_samples::operations::types::{EqBand, EqBandType, ParametricEq};
use audio_samples::{AudioParametricEq, I24};

use bench_suite_common::{
    CHANNELS_DEFAULT, DTYPES_DEFAULT, LENGTH_SWEEP_FULL, ParamLabel, SampleSizePolicy,
    fixture_a440, sample_size_for,
};

// ===========================================================================
// Top-level entry point — wrappers call into this with their typed Criterion.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_eq_001_apply_parametric_eq(c);
    bench_eq_002_apply_eq_band(c);
    bench_eq_003_apply_peak_filter(c);
    bench_eq_004_apply_low_shelf(c);
    bench_eq_005_apply_high_shelf(c);
    bench_eq_006_apply_three_band_eq(c);
    bench_eq_007_eq_frequency_response(c);
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Build a `ParametricEq` with `n_bands` enabled peak bands at logarithmically
/// spaced centre frequencies across the audible band. Pure data construction;
/// excluded from the timed routine via `iter_batched_ref` setup.
fn make_eq(n_bands: usize) -> ParametricEq {
    let mut eq = ParametricEq::new();
    // Spread centres across roughly 100 Hz .. 16 kHz logarithmically so each
    // band targets a distinct region of the spectrum. Q and gain are fixed.
    let lo = 100.0_f64.ln();
    let hi = 16_000.0_f64.ln();
    let denom = (n_bands.max(1) as f64).max(1.0);
    for i in 0..n_bands {
        let t = (i as f64 + 0.5) / denom;
        let freq = (lo + (hi - lo) * t).exp();
        eq.add_band(EqBand::peak(freq, 3.0, 2.0));
    }
    eq
}

/// Construct an [`EqBand`] for the given band type using the available
/// `EqBand::*` constructors and mutating `band_type` for `BandPass`/`BandStop`
/// (those variants have no dedicated constructor on `EqBand`). All public
/// fields, so mutation is sound.
fn band_for_type(bt: EqBandType) -> EqBand {
    match bt {
        EqBandType::Peak => EqBand::peak(1_000.0, 3.0, 2.0),
        EqBandType::LowShelf => EqBand::low_shelf(200.0, 3.0, 0.707),
        EqBandType::HighShelf => EqBand::high_shelf(8_000.0, 3.0, 0.707),
        EqBandType::LowPass => EqBand::low_pass(8_000.0, 0.707),
        EqBandType::HighPass => EqBand::high_pass(200.0, 0.707),
        EqBandType::BandPass => {
            let mut b = EqBand::low_pass(1_000.0, 2.0);
            b.band_type = EqBandType::BandPass;
            b
        }
        EqBandType::BandStop => {
            let mut b = EqBand::low_pass(1_000.0, 2.0);
            b.band_type = EqBandType::BandStop;
            b
        }
        _ => EqBand::peak(1_000.0, 3.0, 2.0),
    }
}

fn band_type_label(bt: EqBandType) -> &'static str {
    match bt {
        EqBandType::Peak => "peak",
        EqBandType::LowShelf => "low_shelf",
        EqBandType::HighShelf => "high_shelf",
        EqBandType::LowPass => "lowpass",
        EqBandType::HighPass => "highpass",
        EqBandType::BandPass => "bandpass",
        EqBandType::BandStop => "bandstop",
        _ => "unknown",
    }
}

// ===========================================================================
// Dispatch macros — DTYPES_DEFAULT-typed expansion.
//
// `dispatch_apply_eq!`: bench `audio.apply_parametric_eq(&eq)` swept over the
// four default dtypes. The EQ object is captured by reference and cloned
// per-iter only as a (trivial) `Vec<EqBand>` — actually we build it once in
// the setup closure to keep timed work to just the apply.
//
// `dispatch_unary_audio_op!`: generic single-call audio-mutating op (`&mut
// self`-style), no extra args beyond the closure body.
// ===========================================================================

macro_rules! dispatch_apply_eq {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, $eq:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        let eq_ref: &ParametricEq = $eq;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |a| {
                        black_box(a.apply_parametric_eq(eq_ref).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<I24>(n, ch),
                    |a| {
                        black_box(a.apply_parametric_eq(eq_ref).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i32>(n, ch),
                    |a| {
                        black_box(a.apply_parametric_eq(eq_ref).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |a| {
                        black_box(a.apply_parametric_eq(eq_ref).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

macro_rules! dispatch_unary_audio_op {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, |$audio:ident| $body:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |$audio| {
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<I24>(n, ch),
                    |$audio| {
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i32>(n, ch),
                    |$audio| {
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |$audio| {
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

macro_rules! dispatch_apply_band {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, $band:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        let band_ref: &EqBand = $band;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |a| {
                        black_box(a.apply_eq_band(band_ref).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<I24>(n, ch),
                    |a| {
                        black_box(a.apply_eq_band(band_ref).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i32>(n, ch),
                    |a| {
                        black_box(a.apply_eq_band(band_ref).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |a| {
                        black_box(a.apply_eq_band(band_ref).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

macro_rules! dispatch_freq_response {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, $eq:expr, $freqs:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        let eq_ref: &ParametricEq = $eq;
        let freqs_ref: &[f64] = $freqs;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |a| {
                        black_box(a.eq_frequency_response(eq_ref, freqs_ref).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<I24>(n, ch),
                    |a| {
                        black_box(a.eq_frequency_response(eq_ref, freqs_ref).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i32>(n, ch),
                    |a| {
                        black_box(a.eq_frequency_response(eq_ref, freqs_ref).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |a| {
                        black_box(a.eq_frequency_response(eq_ref, freqs_ref).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

// ===========================================================================
// EQ-001 apply_parametric_eq — NoFast tier; sweep length × channels × dtype
// × n_bands ∈ {3, 8, 16}.
// ===========================================================================

fn bench_eq_001_apply_parametric_eq<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("eq");
    let n_bands_sweep: &[usize] = &[3, 8, 16];
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &n_bands in n_bands_sweep {
            let eq = make_eq(n_bands);
            for &ch in CHANNELS_DEFAULT {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("n_bands", n_bands)
                        .build();
                    let id = BenchmarkId::new("EQ-001_apply_parametric_eq", label);
                    dispatch_apply_eq!(group, id, dt, len, ch, &eq);
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// EQ-002 apply_eq_band — NoFast tier; sweep band_type over all seven variants.
// ===========================================================================

fn bench_eq_002_apply_eq_band<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("eq");
    let band_types: &[EqBandType] = &[
        EqBandType::Peak,
        EqBandType::LowShelf,
        EqBandType::HighShelf,
        EqBandType::LowPass,
        EqBandType::HighPass,
        EqBandType::BandPass,
        EqBandType::BandStop,
    ];
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &bt in band_types {
            let band = band_for_type(bt);
            let bt_label = band_type_label(bt);
            for &ch in CHANNELS_DEFAULT {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("band_type", bt_label)
                        .build();
                    let id = BenchmarkId::new("EQ-002_apply_eq_band", label);
                    dispatch_apply_band!(group, id, dt, len, ch, &band);
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// EQ-003 apply_peak_filter — NoFast tier; convenience wrapper around one
// peak band.
// ===========================================================================

fn bench_eq_003_apply_peak_filter<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("eq");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", ch)
                    .build();
                let id = BenchmarkId::new("EQ-003_apply_peak_filter", label);
                dispatch_unary_audio_op!(group, id, dt, len, ch, |a| a
                    .apply_peak_filter(1_000.0, 3.0, 2.0)
                    .ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// EQ-004 apply_low_shelf — NoFast tier; single shelf band convenience.
// ===========================================================================

fn bench_eq_004_apply_low_shelf<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("eq");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", ch)
                    .build();
                let id = BenchmarkId::new("EQ-004_apply_low_shelf", label);
                dispatch_unary_audio_op!(group, id, dt, len, ch, |a| a
                    .apply_low_shelf(200.0, 3.0, 0.707)
                    .ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// EQ-005 apply_high_shelf — NoFast tier; single shelf band convenience.
// ===========================================================================

fn bench_eq_005_apply_high_shelf<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("eq");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", ch)
                    .build();
                let id = BenchmarkId::new("EQ-005_apply_high_shelf", label);
                dispatch_unary_audio_op!(group, id, dt, len, ch, |a| a
                    .apply_high_shelf(8_000.0, 3.0, 0.707)
                    .ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// EQ-006 apply_three_band_eq — NoFast tier; three-band cascade convenience.
// ===========================================================================

fn bench_eq_006_apply_three_band_eq<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("eq");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", ch)
                    .build();
                let id = BenchmarkId::new("EQ-006_apply_three_band_eq", label);
                dispatch_unary_audio_op!(group, id, dt, len, ch, |a| a
                    .apply_three_band_eq(200.0, -2.0, 1_000.0, 3.0, 2.0, 4_000.0, 1.0)
                    .ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// EQ-007 eq_frequency_response — NoFast tier; analytical, signal-length
// independent. Sweep n_bands ∈ {3, 8, 16} and n_freqs ∈ {32, 256, 2048}.
// Use a single small fixture (len/ch/dt are nominal — cost is driven by
// n_bands × n_freqs).
// ===========================================================================

fn bench_eq_007_eq_frequency_response<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("eq");
    // Fixture is just a host for the `&self` receiver; len/ch don't drive cost.
    let len = 1024_usize;
    let ch = 1_usize;
    let n_bands_sweep: &[usize] = &[3, 8, 16];
    let n_freqs_sweep: &[usize] = &[32, 256, 2048];
    group.sample_size(sample_size_for(SampleSizePolicy::NoFast, 0));
    for &n_bands in n_bands_sweep {
        let eq = make_eq(n_bands);
        for &n_freqs in n_freqs_sweep {
            // Logarithmically spaced frequencies across 20 Hz .. 20 kHz.
            let freqs: Vec<f64> = {
                let lo = 20.0_f64.ln();
                let hi = 20_000.0_f64.ln();
                (0..n_freqs)
                    .map(|i| {
                        let t = (i as f64 + 0.5) / n_freqs as f64;
                        (lo + (hi - lo) * t).exp()
                    })
                    .collect()
            };
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((n_bands * n_freqs) as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", ch)
                    .with("n_bands", n_bands)
                    .with("n_freqs", n_freqs)
                    .build();
                let id = BenchmarkId::new("EQ-007_eq_frequency_response", label);
                dispatch_freq_response!(group, id, dt, len, ch, &eq, &freqs);
            }
        }
    }
    group.finish();
}
