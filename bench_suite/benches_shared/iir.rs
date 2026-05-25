//! Shared body for the `AudioIirFiltering` bench targets
//! (`bench_iir_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Catalog section: IIR (lines 232–254 of CATALOG.md), trait
//! `AudioIirFiltering` plus the standalone [`IirFilter`] and [`SosFilter`]
//! structs in `src/operations/iir_filtering.rs`.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//!
//! ## Coverage and skips
//!
//! - **IIR-001** `apply_iir_filter` — swept over response × order.
//! - **IIR-002 / IIR-003 / IIR-004** `butterworth_{lowpass,highpass,bandpass}`
//!   — swept over order.
//! - **IIR-005** `chebyshev_i` — **skipped** (impl returns `Err` only — see
//!   CATALOG Open Q 2).
//! - **IIR-006** `frequency_response` (trait) — **skipped** (placeholder
//!   returning flat unity — CATALOG Open Q 3).
//! - **IIR-007** `IirFilter::process_sample` — **skipped**; benching a
//!   single-sample call is dominated by Criterion's measurement overhead
//!   and the per-sample work is covered by IIR-008/009 block APIs.
//! - **IIR-008 / IIR-009** `IirFilter::process_samples{,_in_place}` — full
//!   length sweep at a representative LP-2 design.
//! - **IIR-010** `IirFilter::frequency_response` — sweep n_freqs and the
//!   coefficient count via filter order.
//! - **IIR-011** `Biquad::process_sample` — **skipped**: `Biquad` is a
//!   private type inside `iir_filtering.rs` (not exported from the crate),
//!   so a bench cannot construct one without modifying library
//!   visibility.
//! - **IIR-012** `SosFilter::process_sample` — **skipped** in favour of
//!   block APIs (IIR-013/IIR-014) per the master brief.
//! - **IIR-013 / IIR-014** `SosFilter::process_samples{,_in_place}` —
//!   full length sweep at order ∈ {4,8} (which directly drives the
//!   number of biquad sections).
//! - **IIR-015** `SosFilter::frequency_response` — sweep n_freqs and the
//!   number of sections.
//!
//! ## Cost axes
//!
//! - `len`: per-channel sample count (`LENGTH_SWEEP_FULL`).
//! - `dt`: the **trait-level** benches (IIR-001..004) sweep
//!   `DTYPES_DEFAULT`; the standalone-struct benches operate on `f64`
//!   only (the structs' APIs take `&[f64]`).
//! - `ch`: `CHANNELS_DEFAULT` for trait-level benches; `1` for
//!   standalone-struct benches (which have no channel concept).
//! - `order`: filter order. The cost-relevant parameter for any IIR
//!   call. Sweep `{2, 4, 8}`. Order 2 takes the fast single-biquad path;
//!   orders > 2 take the SOS-cascade path. The implementation caps order
//!   at 12, so 8 is comfortably below the limit. (Bandpass: catalogue
//!   note says bandpass order is interpreted as the per-edge order; the
//!   impl validates `order ≤ 12` directly, so 2/4/8 are all valid.)
//! - `n_freqs`: query-grid size for the `frequency_response` benches.
//!
//! All benches use `SampleSizePolicy::NoFast`: even the smallest
//! length × order combination is well into the multi-µs regime due to
//! the inherent serial dependency of the difference equation.
//!
//! The representative cutoff is **1 kHz** for low-pass / high-pass, and
//! **300 Hz / 3 kHz** for band-pass, at the 44.1 kHz fixture rate. The
//! cutoff affects coefficient *values* but not the per-sample cost
//! (coefficient design is one-shot per call; the inner loop is fixed
//! length).

use std::hint::black_box;
use std::num::NonZeroUsize;

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};

use audio_samples::{AudioIirFiltering, I24};
use audio_samples::operations::iir_filtering::{IirFilter, SosFilter};
use audio_samples::operations::types::IirFilterDesign;

use bench_suite_common::{
    CHANNELS_DEFAULT, DTYPES_DEFAULT, LENGTH_SWEEP_FULL, ParamLabel,
    SampleSizePolicy, fixture_a440, sample_size_for,
};

// ===========================================================================
// Tuning knobs (representative single-point values).
// ===========================================================================

/// Cutoff used for every order-swept low-pass / high-pass bench.
const CUTOFF_HZ: f64 = 1_000.0;
/// Band-pass edges.
const BAND_LOW_HZ: f64 = 300.0;
const BAND_HIGH_HZ: f64 = 3_000.0;
/// Order sweep — applied wherever order is a cost axis.
const ORDER_SWEEP: &[usize] = &[2, 4, 8];
/// Frequency-response query-grid sweep.
const N_FREQS_SWEEP: &[usize] = &[16, 256, 4096];

// ===========================================================================
// Top-level entry point.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_iir_001_apply_iir_filter(c);
    bench_iir_002_butterworth_lowpass(c);
    bench_iir_003_butterworth_highpass(c);
    bench_iir_004_butterworth_bandpass(c);
    // IIR-005 chebyshev_i: unimplemented (returns Err) — skipped.
    // IIR-006 trait::frequency_response: placeholder — skipped.
    // IIR-007 IirFilter::process_sample: per-sample, subsumed by block APIs.
    bench_iir_008_iirfilter_process_samples(c);
    bench_iir_009_iirfilter_process_samples_in_place(c);
    bench_iir_010_iirfilter_frequency_response(c);
    // IIR-011 Biquad::process_sample: private type — skipped.
    // IIR-012 SosFilter::process_sample: per-sample, see brief — skipped.
    bench_iir_013_sosfilter_process_samples(c);
    bench_iir_014_sosfilter_process_samples_in_place(c);
    bench_iir_015_sosfilter_frequency_response(c);
}

// ===========================================================================
// Dispatch macro: trait-method bench over DTYPES_DEFAULT.
//
// Body shape: `|audio| audio.<method>(...)` where `audio` is `&mut AudioSamples<'_, T>`.
// Setup re-builds the fixture each iteration (LargeInput) because every
// invocation mutates the buffer in place.
// ===========================================================================

macro_rules! dispatch_unary_sine_mut {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, |$audio:ident| $body:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |$audio| { black_box($body).ok(); },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<I24>(n, ch),
                    |$audio| { black_box($body).ok(); },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i32>(n, ch),
                    |$audio| { black_box($body).ok(); },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |$audio| { black_box($body).ok(); },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

// ===========================================================================
// f64 buffer helper used by the standalone-struct benches (IirFilter,
// SosFilter). 440 Hz sine at the same nominal sample rate as the trait
// fixtures, deterministic.
// ===========================================================================

fn sine_buffer_f64(n: usize) -> Vec<f64> {
    let sr = 44_100.0_f64;
    let two_pi_f = 2.0 * std::f64::consts::PI * 440.0;
    (0..n)
        .map(|i| (two_pi_f * (i as f64) / sr).sin())
        .collect()
}

// ===========================================================================
// Coefficient builders for the IirFilter / SosFilter benches.
//
// We hand-roll trivial biquad coefficients here so that benches don't
// depend on the private helpers in `iir_filtering.rs`. Costs match the
// Butterworth path because every coefficient slot is non-zero.
// ===========================================================================

/// Build a single-biquad `IirFilter` with three b- and three a-coeffs
/// matching the order=2 layout used by the impl. Values are not a real
/// filter — they're only here to drive the difference equation at full
/// cost.
fn make_iir_filter(n_coeffs: usize) -> IirFilter {
    // n_coeffs covers both b_coeffs and a_coeffs lengths. Use the same
    // length on both sides so the inner-loop sums match a real DF-I/II
    // filter. `a[0]` is the divisor — set it to 1.0 for the standard
    // normalised form.
    let n = n_coeffs.max(1);
    let mut b = vec![0.0_f64; n];
    let mut a = vec![0.0_f64; n];
    // Identity-ish but with non-zero taps so the loop body runs in full.
    b[0] = 0.5;
    if n > 1 {
        b[1] = 0.25;
    }
    if n > 2 {
        for v in b.iter_mut().skip(2) {
            *v = 0.0625;
        }
    }
    a[0] = 1.0;
    if n > 1 {
        a[1] = -0.1;
    }
    if n > 2 {
        for v in a.iter_mut().skip(2) {
            *v = 0.01;
        }
    }
    IirFilter::new(b, a)
}

/// Build a SOS cascade of `n_sections` biquads (each a 3-tap section).
fn make_sos_filter(n_sections: usize) -> SosFilter {
    let sections = (0..n_sections.max(1))
        .map(|_| make_iir_filter(3))
        .collect();
    SosFilter::new(sections)
}

/// Frequency grid for IIR-010 / IIR-015 — log-spaced from 20 Hz to
/// 20 kHz inclusive (canonical audio-engineering range).
fn freq_grid(n: usize) -> Vec<f64> {
    let lo = 20.0_f64.ln();
    let hi = 20_000.0_f64.ln();
    (0..n)
        .map(|i| {
            let t = if n <= 1 { 0.0 } else { i as f64 / (n as f64 - 1.0) };
            (lo + (hi - lo) * t).exp()
        })
        .collect()
}

// ===========================================================================
// IIR-001 apply_iir_filter — response × order sweep.
//
// Variants benched:
//   - LowPass   @ cutoff = 1 kHz
//   - HighPass  @ cutoff = 1 kHz
//   - BandPass  @ low/high = 300 Hz / 3 kHz
// ===========================================================================

fn bench_iir_001_apply_iir_filter<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("iir");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &order in ORDER_SWEEP {
                let nz_order = NonZeroUsize::new(order).expect("order > 0");
                for (resp_label, design) in &[
                    (
                        "lp",
                        IirFilterDesign::butterworth_lowpass(nz_order, CUTOFF_HZ),
                    ),
                    (
                        "hp",
                        IirFilterDesign::butterworth_highpass(nz_order, CUTOFF_HZ),
                    ),
                    (
                        "bp",
                        IirFilterDesign::butterworth_bandpass(
                            nz_order,
                            BAND_LOW_HZ,
                            BAND_HIGH_HZ,
                        ),
                    ),
                ] {
                    for &dt in DTYPES_DEFAULT {
                        group.throughput(Throughput::Elements((len * ch) as u64));
                        let label = ParamLabel::new()
                            .with("len", len)
                            .with("dt", dt)
                            .with("ch", ch)
                            .with("order", order)
                            .with("resp", *resp_label)
                            .with("kind", "butter")
                            .build();
                        let id = BenchmarkId::new("IIR-001_apply_iir_filter", label);
                        let design = design.clone();
                        dispatch_unary_sine_mut!(
                            group, id, dt, len, ch,
                            |a| a.apply_iir_filter(&design)
                        );
                    }
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// IIR-002 butterworth_lowpass — order sweep at fixed cutoff.
// ===========================================================================

fn bench_iir_002_butterworth_lowpass<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("iir");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &order in ORDER_SWEEP {
                let nz_order = NonZeroUsize::new(order).expect("order > 0");
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("order", order)
                        .build();
                    let id = BenchmarkId::new("IIR-002_butterworth_lowpass", label);
                    dispatch_unary_sine_mut!(
                        group, id, dt, len, ch,
                        |a| a.butterworth_lowpass(nz_order, CUTOFF_HZ)
                    );
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// IIR-003 butterworth_highpass — order sweep at fixed cutoff.
// ===========================================================================

fn bench_iir_003_butterworth_highpass<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("iir");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &order in ORDER_SWEEP {
                let nz_order = NonZeroUsize::new(order).expect("order > 0");
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("order", order)
                        .build();
                    let id = BenchmarkId::new("IIR-003_butterworth_highpass", label);
                    dispatch_unary_sine_mut!(
                        group, id, dt, len, ch,
                        |a| a.butterworth_highpass(nz_order, CUTOFF_HZ)
                    );
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// IIR-004 butterworth_bandpass — order sweep at fixed band.
// ===========================================================================

fn bench_iir_004_butterworth_bandpass<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("iir");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &order in ORDER_SWEEP {
                let nz_order = NonZeroUsize::new(order).expect("order > 0");
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("order", order)
                        .build();
                    let id = BenchmarkId::new("IIR-004_butterworth_bandpass", label);
                    dispatch_unary_sine_mut!(
                        group, id, dt, len, ch,
                        |a| a.butterworth_bandpass(nz_order, BAND_LOW_HZ, BAND_HIGH_HZ)
                    );
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// IIR-008 IirFilter::process_samples — block API, allocates a new Vec.
//
// dt = f64 only (the struct API takes &[f64]). order axis controls the
// number of coefficients on each side of the difference equation.
// ===========================================================================

fn bench_iir_008_iirfilter_process_samples<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("iir");
    let ch = 1usize;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &order in ORDER_SWEEP {
            // n_coeffs ≈ order + 1 for direct-form layout.
            let n_coeffs = order + 1;
            group.throughput(Throughput::Elements((len * ch) as u64));
            let label = ParamLabel::new()
                .with("len", len)
                .with("dt", "f64")
                .with("ch", ch)
                .with("order", order)
                .build();
            let id = BenchmarkId::new("IIR-008_iirfilter_process_samples", label);
            group.bench_with_input(id, &(len, n_coeffs), |b, &(n, k)| {
                b.iter_batched_ref(
                    || (make_iir_filter(k), sine_buffer_f64(n)),
                    |inp| {
                        let (filter, buf) = inp;
                        black_box(filter.process_samples(buf));
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

// ===========================================================================
// IIR-009 IirFilter::process_samples_in_place — in-place block API.
// ===========================================================================

fn bench_iir_009_iirfilter_process_samples_in_place<M: Measurement>(
    c: &mut Criterion<M>,
) {
    let mut group = c.benchmark_group("iir");
    let ch = 1usize;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &order in ORDER_SWEEP {
            let n_coeffs = order + 1;
            group.throughput(Throughput::Elements((len * ch) as u64));
            let label = ParamLabel::new()
                .with("len", len)
                .with("dt", "f64")
                .with("ch", ch)
                .with("order", order)
                .build();
            let id = BenchmarkId::new("IIR-009_iirfilter_process_samples_in_place", label);
            group.bench_with_input(id, &(len, n_coeffs), |b, &(n, k)| {
                b.iter_batched_ref(
                    || (make_iir_filter(k), sine_buffer_f64(n)),
                    |inp| {
                        let (filter, buf) = inp;
                        filter.process_samples_in_place(buf);
                        black_box(&buf);
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

// ===========================================================================
// IIR-010 IirFilter::frequency_response — analytical evaluation, no audio
// signal cost. Sweep n_freqs and n_coeffs (via order).
//
// Uses LENGTH_SWEEP_FULL only for the n_freqs axis is *not* sensible here
// — instead use a dedicated `N_FREQS_SWEEP` and treat each combination as
// a single `AlwaysDefault` sample-size point.
// ===========================================================================

fn bench_iir_010_iirfilter_frequency_response<M: Measurement>(
    c: &mut Criterion<M>,
) {
    let mut group = c.benchmark_group("iir");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    for &order in ORDER_SWEEP {
        let n_coeffs = order + 1;
        for &n_freqs in N_FREQS_SWEEP {
            group.throughput(Throughput::Elements(n_freqs as u64));
            let label = ParamLabel::new()
                .with("dt", "f64")
                .with("n_freqs", n_freqs)
                .with("n_coeffs", n_coeffs)
                .with("order", order)
                .build();
            let id = BenchmarkId::new("IIR-010_iirfilter_frequency_response", label);
            group.bench_with_input(id, &(n_coeffs, n_freqs), |b, &(k, nf)| {
                b.iter_batched_ref(
                    || (make_iir_filter(k), freq_grid(nf)),
                    |inp| {
                        let (filter, freqs) = inp;
                        black_box(filter.frequency_response(freqs, 44_100.0));
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

// ===========================================================================
// IIR-013 SosFilter::process_samples — n_sections-swept block API.
// ===========================================================================

fn bench_iir_013_sosfilter_process_samples<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("iir");
    let ch = 1usize;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        // Order > 2 means SOS-cascade with order/2 sections (rounded up
        // for odd orders). Map order ∈ {2, 4, 8} → {1, 2, 4} sections.
        for &order in ORDER_SWEEP {
            let n_sections = order.div_ceil(2).max(1);
            group.throughput(Throughput::Elements((len * ch) as u64));
            let label = ParamLabel::new()
                .with("len", len)
                .with("dt", "f64")
                .with("ch", ch)
                .with("order", order)
                .with("n_sections", n_sections)
                .build();
            let id = BenchmarkId::new("IIR-013_sosfilter_process_samples", label);
            group.bench_with_input(id, &(len, n_sections), |b, &(n, sec)| {
                b.iter_batched_ref(
                    || (make_sos_filter(sec), sine_buffer_f64(n)),
                    |inp| {
                        let (filter, buf) = inp;
                        black_box(filter.process_samples(buf));
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

// ===========================================================================
// IIR-014 SosFilter::process_samples_in_place — in-place version.
// ===========================================================================

fn bench_iir_014_sosfilter_process_samples_in_place<M: Measurement>(
    c: &mut Criterion<M>,
) {
    let mut group = c.benchmark_group("iir");
    let ch = 1usize;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &order in ORDER_SWEEP {
            let n_sections = order.div_ceil(2).max(1);
            group.throughput(Throughput::Elements((len * ch) as u64));
            let label = ParamLabel::new()
                .with("len", len)
                .with("dt", "f64")
                .with("ch", ch)
                .with("order", order)
                .with("n_sections", n_sections)
                .build();
            let id = BenchmarkId::new("IIR-014_sosfilter_process_samples_in_place", label);
            group.bench_with_input(id, &(len, n_sections), |b, &(n, sec)| {
                b.iter_batched_ref(
                    || (make_sos_filter(sec), sine_buffer_f64(n)),
                    |inp| {
                        let (filter, buf) = inp;
                        filter.process_samples_in_place(buf);
                        black_box(&buf);
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

// ===========================================================================
// IIR-015 SosFilter::frequency_response — n_freqs × n_sections.
// ===========================================================================

fn bench_iir_015_sosfilter_frequency_response<M: Measurement>(
    c: &mut Criterion<M>,
) {
    let mut group = c.benchmark_group("iir");
    group.sample_size(sample_size_for(SampleSizePolicy::AlwaysDefault, 0));
    for &order in ORDER_SWEEP {
        let n_sections = order.div_ceil(2).max(1);
        for &n_freqs in N_FREQS_SWEEP {
            group.throughput(Throughput::Elements(n_freqs as u64));
            let label = ParamLabel::new()
                .with("dt", "f64")
                .with("n_freqs", n_freqs)
                .with("n_sections", n_sections)
                .with("order", order)
                .build();
            let id = BenchmarkId::new("IIR-015_sosfilter_frequency_response", label);
            group.bench_with_input(id, &(n_sections, n_freqs), |b, &(sec, nf)| {
                b.iter_batched_ref(
                    || (make_sos_filter(sec), freq_grid(nf)),
                    |inp| {
                        let (filter, freqs) = inp;
                        black_box(filter.frequency_response(freqs, 44_100.0));
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

