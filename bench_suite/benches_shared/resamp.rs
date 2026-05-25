//! Shared body for the resampling bench targets
//! (`bench_resamp_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/resamp.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Catalog section: Resamp (lines 489-499 of CATALOG.md).
//! Methodology: `bench_suite/METHODOLOGY.md`.
//!
//! Resampling cost is dominated by (src_sr, dst_sr) and the quality preset:
//! - `Fast`     — linear interpolation, integer-ratio SSE2 fast paths.
//! - `Medium`   — rubato FFT synchronous resampler (4 KiB chunks).
//! - `High`     — rubato FFT synchronous resampler (16 KiB chunks).
//!
//! The Medium/High paths construct an FFT plan + filter table on first use
//! and then cache it thread-locally; the bench loop therefore amortises
//! plan-construction cost across iterations (Criterion's `iter_batched`
//! creates a fresh fixture per iter but reuses the cached resampler).

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::num::NonZeroU32;
use std::time::Duration;

use audio_samples::{
    AudioSamples, I24, StandardSample, resample, resample_by_ratio,
    operations::types::ResamplingQuality,
    utils::generation::{sine_wave, stereo_sine_wave},
};

use bench_suite_common::{
    CHANNELS_DEFAULT, DTYPES_DEFAULT, LENGTH_SWEEP_NO_XXXL,
    ParamLabel, SampleSizePolicy, sample_size_for,
};

// ===========================================================================
// Top-level entry point
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_resamp_001_resample(c);
    bench_resamp_002_resample_by_ratio(c);
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Sample-rate pairs swept by `Resamp-001`. A mix of common downsample and
/// upsample targets, plus the 44.1k <-> 22.05k integer-ratio Fast-path hot
/// path and the 48k -> 96k integer-ratio upsample.
const SR_PAIRS: &[(u32, u32)] = &[
    (44_100, 16_000),
    (44_100, 48_000),
    (16_000, 44_100),
    (48_000, 96_000),
    (44_100, 22_050),
    (22_050, 44_100),
];

/// Resampling ratios swept by `Resamp-002`.
const RATIOS: &[f64] = &[0.5, 1.5, 2.0, 0.999, 3.141_59];

/// Quality presets swept by both rows.
const QUALITIES: &[(ResamplingQuality, &str)] = &[
    (ResamplingQuality::Fast, "fast"),
    (ResamplingQuality::Medium, "medium"),
    (ResamplingQuality::High, "high"),
];

/// Build a 440 Hz fixture at an arbitrary sample rate (the standard
/// `fixture_a440` is locked to 44.1 kHz, but resampling needs the input SR
/// to vary). Mono uses `sine_wave`; stereo uses `stereo_sine_wave` (left =
/// sine, right = cosine).
#[inline]
fn fixture_at_sr<T>(n_samples: usize, channels: usize, src_sr_hz: u32) -> AudioSamples<'static, T>
where
    T: StandardSample + 'static,
{
    // SAFETY: SR_PAIRS / ratio-derived rates are all non-zero by construction.
    let sr = NonZeroU32::new(src_sr_hz).expect("src_sr must be non-zero");
    let sr_f = f64::from(src_sr_hz);
    let duration = Duration::from_secs_f64(n_samples as f64 / sr_f);
    match channels {
        1 => sine_wave::<T>(440.0, duration, sr, 1.0),
        2 => stereo_sine_wave::<T>(440.0, duration, sr, 1.0),
        n => panic!(
            "fixture_at_sr: channel count {n} not supported (only 1 or 2)."
        ),
    }
}

// ===========================================================================
// Dispatch macros — DTYPES_DEFAULT-typed expansion.
//
// `dispatch_resample_sr!`: bench `resample(audio, dst_sr, quality)` with a
// fixture built at `src_sr_hz`.
//
// `dispatch_resample_ratio!`: bench `resample_by_ratio(audio, ratio, quality)`
// with a fixture built at 44.1 kHz.
// ===========================================================================

macro_rules! dispatch_resample_sr {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, $src_sr:expr, $dst_sr:expr, $quality:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        let src_sr: u32 = $src_sr;
        let dst_sr_nz: NonZeroU32 = NonZeroU32::new($dst_sr).expect("dst_sr must be non-zero");
        let quality: ResamplingQuality = $quality;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_at_sr::<i16>(n, ch, src_sr),
                    |a| { let _ = black_box(resample::<i16>(&*a, dst_sr_nz, quality)); },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_at_sr::<I24>(n, ch, src_sr),
                    |a| { let _ = black_box(resample::<I24>(&*a, dst_sr_nz, quality)); },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_at_sr::<i32>(n, ch, src_sr),
                    |a| { let _ = black_box(resample::<i32>(&*a, dst_sr_nz, quality)); },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_at_sr::<f32>(n, ch, src_sr),
                    |a| { let _ = black_box(resample::<f32>(&*a, dst_sr_nz, quality)); },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

macro_rules! dispatch_resample_ratio {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, $ratio:expr, $quality:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        let ratio: f64 = $ratio;
        let quality: ResamplingQuality = $quality;
        // Fixture sample rate is fixed at 44.1 kHz; `resample_by_ratio`
        // scales target_sr = round(src_sr * ratio).
        const SRC_SR: u32 = 44_100;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_at_sr::<i16>(n, ch, SRC_SR),
                    |a| { let _ = black_box(resample_by_ratio::<i16>(&*a, ratio, quality)); },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_at_sr::<I24>(n, ch, SRC_SR),
                    |a| { let _ = black_box(resample_by_ratio::<I24>(&*a, ratio, quality)); },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_at_sr::<i32>(n, ch, SRC_SR),
                    |a| { let _ = black_box(resample_by_ratio::<i32>(&*a, ratio, quality)); },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_at_sr::<f32>(n, ch, SRC_SR),
                    |a| { let _ = black_box(resample_by_ratio::<f32>(&*a, ratio, quality)); },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

// ===========================================================================
// Resamp-001 resample — NoFast tier; sweep (src_sr, dst_sr) and quality.
//
// Lengths use LENGTH_SWEEP_NO_XXXL (cap at 1_048_576 frames); the FFT-based
// High path at 4M frames would push the per-point measurement budget past
// 30 s.
// ===========================================================================

fn bench_resamp_001_resample<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("resamp");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &(src_sr, dst_sr) in SR_PAIRS {
            for &(quality, q_label) in QUALITIES {
                for &ch in CHANNELS_DEFAULT {
                    for &dt in DTYPES_DEFAULT {
                        group.throughput(Throughput::Elements((len * ch) as u64));
                        let label = ParamLabel::new()
                            .with("len", len)
                            .with("dt", dt)
                            .with("ch", ch)
                            .with("src_sr", src_sr)
                            .with("dst_sr", dst_sr)
                            .with("quality", q_label)
                            .build();
                        let id = BenchmarkId::new("Resamp-001_resample", label);
                        dispatch_resample_sr!(
                            group, id, dt, len, ch, src_sr, dst_sr, quality
                        );
                    }
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Resamp-002 resample_by_ratio — NoFast tier; sweep ratio and quality.
//
// The fractional ratios (0.999, 3.14159) force the general-fractional Fast
// path; integer ratios (0.5, 2.0) hit the SSE2 integer fast paths; 1.5 falls
// into the general N=3-output-per-input upsample branch.
// ===========================================================================

fn bench_resamp_002_resample_by_ratio<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("resamp");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ratio in RATIOS {
            for &(quality, q_label) in QUALITIES {
                for &ch in CHANNELS_DEFAULT {
                    for &dt in DTYPES_DEFAULT {
                        group.throughput(Throughput::Elements((len * ch) as u64));
                        let label = ParamLabel::new()
                            .with("len", len)
                            .with("dt", dt)
                            .with("ch", ch)
                            .with("ratio", format!("{ratio:.5}"))
                            .with("quality", q_label)
                            .build();
                        let id = BenchmarkId::new("Resamp-002_resample_by_ratio", label);
                        dispatch_resample_ratio!(group, id, dt, len, ch, ratio, quality);
                    }
                }
            }
        }
    }
    group.finish();
}
