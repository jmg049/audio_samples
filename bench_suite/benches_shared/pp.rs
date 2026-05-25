//! Shared body for the `peak_picking` free-function bench targets
//! (`bench_pp_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/pp.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `PP`
//! (lines 336-349).
//!
//! Notes:
//! - There is no `AudioPeakPicking` trait; everything in `peak_picking`
//!   is exposed as free functions, so the dispatch boilerplate is simpler
//!   than e.g. `stats.rs`.
//! - All free functions operate on `&NonEmptySlice<f64>` (an onset-strength
//!   curve). Peak-picking is typically run on hop-spaced ODFs that are far
//!   shorter than the raw signal — we use a custom length sweep instead of
//!   `LENGTH_SWEEP_FULL` to reflect that.
//! - Single dtype (f64) — the input type is fixed by the function signatures.

use std::num::NonZeroUsize;

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use non_empty_slice::NonEmptySlice;

use audio_samples::operations::peak_picking::{
    adaptive_threshold, apply_median_filter, apply_pre_emphasis, normalize_onset_strength,
    pick_peaks, smooth_onset_strength,
};
use audio_samples::operations::types::{
    AdaptiveThresholdConfig, AdaptiveThresholdMethod, NormalizationMethod, PeakPickingConfig,
};

use bench_suite_common::{ParamLabel, SampleSizePolicy, sample_size_for};

// ===========================================================================
// Length sweep — onset-strength curves are short. 128 .. 32_768 covers
// realistic ODFs from a few seconds at hop ~1024 / sr=44.1 kHz up to many
// minutes of audio.
// ===========================================================================

const PP_LENGTH_SWEEP: &[usize] = &[128, 512, 2048, 8192, 32_768];

// ===========================================================================
// Top-level entry point — wrappers call into this with their typed Criterion.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_pp_001_adaptive_threshold(c);
    bench_pp_002_pick_peaks(c);
    bench_pp_003_apply_pre_emphasis(c);
    bench_pp_004_apply_median_filter(c);
    bench_pp_005_normalize_onset_strength(c);
    bench_pp_006_smooth_onset_strength(c);
}

// ===========================================================================
// Fixture: a deterministic synthetic onset-strength curve as a plain Vec<f64>.
//
// We build a half-wave-rectified sum of a low-frequency sine plus a sparse
// impulse train, then take its absolute value. This produces a non-negative
// curve with clear local maxima — the shape onset detectors actually emit
// downstream of spectral flux / energy.
// ===========================================================================

fn onset_curve(n: usize) -> Vec<f64> {
    let n = n.max(1);
    (0..n)
        .map(|i| {
            let phase = (i as f64) * 0.1;
            let envelope = phase.sin();
            let impulse = if i.is_multiple_of(13) { 0.5 } else { 0.0 };
            (envelope + impulse).abs()
        })
        .collect()
}

// ===========================================================================
// PP-001 adaptive_threshold — NoFast tier. Sweep method axis
// {Delta, Percentile, Combined}.
// ===========================================================================

fn bench_pp_001_adaptive_threshold<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("pp");
    for (i, &len) in PP_LENGTH_SWEEP.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &method in &[
            AdaptiveThresholdMethod::Delta,
            AdaptiveThresholdMethod::Percentile,
            AdaptiveThresholdMethod::Combined,
        ] {
            // Window of ~5 % of the curve length, clamped to [3, 64].
            let window_size = ((len / 20).max(3)).min(64);
            let mut cfg = AdaptiveThresholdConfig::default();
            cfg.method = method;
            cfg.window_size = window_size;

            group.throughput(Throughput::Elements(len as u64));
            let method_label = match method {
                AdaptiveThresholdMethod::Delta => "delta",
                AdaptiveThresholdMethod::Percentile => "percentile",
                AdaptiveThresholdMethod::Combined => "combined",
                _ => "other",
            };
            let label = ParamLabel::new()
                .with("len", len)
                .with("dt", "f64")
                .with("ch", 1)
                .with("method", method_label)
                .with("window_size", window_size)
                .build();
            let id = BenchmarkId::new("PP-001_adaptive_threshold", label);
            group.bench_with_input(id, &len, |b, &n| {
                b.iter_batched_ref(
                    || onset_curve(n),
                    |curve| {
                        let slice = NonEmptySlice::new(curve.as_slice()).unwrap();
                        black_box(adaptive_threshold(slice, &cfg).ok());
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

// ===========================================================================
// PP-002 pick_peaks — NoFast tier. Full pipeline; sweep min_peak_separation
// (short and long-spacing regimes).
// ===========================================================================

fn bench_pp_002_pick_peaks<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("pp");
    for (i, &len) in PP_LENGTH_SWEEP.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &sep_div in &[64usize, 8] {
            let sep = NonZeroUsize::new((len / sep_div).max(1)).unwrap();
            let window_size = ((len / 20).max(3)).min(64);
            let mut cfg = PeakPickingConfig::default();
            cfg.min_peak_separation = sep;
            cfg.adaptive_threshold.window_size = window_size;

            group.throughput(Throughput::Elements(len as u64));
            let label = ParamLabel::new()
                .with("len", len)
                .with("dt", "f64")
                .with("ch", 1)
                .with("min_peak_separation", sep.get())
                .with("window_size", window_size)
                .build();
            let id = BenchmarkId::new("PP-002_pick_peaks", label);
            group.bench_with_input(id, &len, |b, &n| {
                b.iter_batched_ref(
                    || onset_curve(n),
                    |curve| {
                        let slice = NonEmptySlice::new(curve.as_slice()).unwrap();
                        black_box(pick_peaks(slice, &cfg).ok());
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

// ===========================================================================
// PP-003 apply_pre_emphasis — NoFast tier. Single-pass O(n) filter.
// ===========================================================================

fn bench_pp_003_apply_pre_emphasis<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("pp");
    let coeff = 0.97_f64;
    for (i, &len) in PP_LENGTH_SWEEP.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        group.throughput(Throughput::Elements(len as u64));
        let label = ParamLabel::new()
            .with("len", len)
            .with("dt", "f64")
            .with("ch", 1)
            .with("coeff", format!("{coeff:.2}"))
            .build();
        let id = BenchmarkId::new("PP-003_apply_pre_emphasis", label);
        group.bench_with_input(id, &len, |b, &n| {
            b.iter_batched_ref(
                || onset_curve(n),
                |curve| {
                    let slice = NonEmptySlice::new(curve.as_slice()).unwrap();
                    black_box(apply_pre_emphasis(slice, coeff).ok());
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ===========================================================================
// PP-004 apply_median_filter — NoFast tier. Sweep kernel size (small vs
// medium); cost is O(n · k · log k) due to per-position sort.
// ===========================================================================

fn bench_pp_004_apply_median_filter<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("pp");
    for (i, &len) in PP_LENGTH_SWEEP.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &kernel in &[3usize, 11] {
            let kernel_nz = NonZeroUsize::new(kernel).unwrap();
            group.throughput(Throughput::Elements(len as u64));
            let label = ParamLabel::new()
                .with("len", len)
                .with("dt", "f64")
                .with("ch", 1)
                .with("kernel", kernel)
                .build();
            let id = BenchmarkId::new("PP-004_apply_median_filter", label);
            group.bench_with_input(id, &len, |b, &n| {
                b.iter_batched_ref(
                    || onset_curve(n),
                    |curve| {
                        let slice = NonEmptySlice::new(curve.as_slice()).unwrap();
                        black_box(apply_median_filter(slice, kernel_nz).ok());
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

// ===========================================================================
// PP-005 normalize_onset_strength — NoFast tier. Sweep method axis
// (Peak vs ZScore — different cost profiles: Peak is one pass, ZScore
// is two passes + sqrt).
// ===========================================================================

fn bench_pp_005_normalize_onset_strength<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("pp");
    for (i, &len) in PP_LENGTH_SWEEP.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &method in &[NormalizationMethod::Peak, NormalizationMethod::ZScore] {
            let method_label = match method {
                NormalizationMethod::Peak => "peak",
                NormalizationMethod::ZScore => "zscore",
                NormalizationMethod::MinMax => "minmax",
                NormalizationMethod::Mean => "mean",
                NormalizationMethod::Median => "median",
                _ => "other",
            };
            group.throughput(Throughput::Elements(len as u64));
            let label = ParamLabel::new()
                .with("len", len)
                .with("dt", "f64")
                .with("ch", 1)
                .with("method", method_label)
                .build();
            let id = BenchmarkId::new("PP-005_normalize_onset_strength", label);
            group.bench_with_input(id, &len, |b, &n| {
                b.iter_batched_ref(
                    || onset_curve(n),
                    |curve| {
                        let slice = NonEmptySlice::new(curve.as_slice()).unwrap();
                        black_box(normalize_onset_strength(slice, method));
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

// ===========================================================================
// PP-006 smooth_onset_strength — NoFast tier. Median filter + moving
// average. Sweep window size (the moving-average window); the median
// kernel is held at 3 to bracket the cheap-prefix regime.
// ===========================================================================

fn bench_pp_006_smooth_onset_strength<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("pp");
    let median_length = NonZeroUsize::new(3).unwrap();
    for (i, &len) in PP_LENGTH_SWEEP.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &window in &[3usize, 17] {
            let window_nz = NonZeroUsize::new(window).unwrap();
            group.throughput(Throughput::Elements(len as u64));
            let label = ParamLabel::new()
                .with("len", len)
                .with("dt", "f64")
                .with("ch", 1)
                .with("window", window)
                .with("median_length", median_length.get())
                .build();
            let id = BenchmarkId::new("PP-006_smooth_onset_strength", label);
            group.bench_with_input(id, &len, |b, &n| {
                b.iter_batched_ref(
                    || onset_curve(n),
                    |curve| {
                        let slice = NonEmptySlice::new(curve.as_slice()).unwrap();
                        black_box(smooth_onset_strength(slice, window_nz, median_length).ok());
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}
