//! Shared body for the `utils::detection` bench targets
//! (`bench_det_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/det.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `Det`
//! (Det-001 .. Det-009).

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use audio_samples::utils::detection::{
    analyze_spectrum_for_cutoff, detect_clipping, detect_dynamic_range,
    detect_fundamental_autocorrelation, detect_fundamental_frequency, detect_silence_regions,
    estimate_noise_floor,
};
#[cfg(feature = "transforms")]
use audio_samples::utils::detection::{detect_sample_rate, estimate_frequency_range};
use audio_samples::{AudioSamples, CastFrom, I24};

use bench_suite_common::{
    CHANNELS_DEFAULT, DTYPES_DEFAULT, LENGTH_SWEEP_FULL, LENGTH_SWEEP_NO_XXXL, ParamLabel,
    SampleSizePolicy, fixture_a440, sample_size_for,
};
use non_empty_slice::{NonEmptySlice, NonEmptyVec};

// ===========================================================================
// Top-level entry point.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    #[cfg(feature = "transforms")]
    bench_det_001_detect_sample_rate(c);
    bench_det_002_detect_fundamental_frequency(c);
    bench_det_003_detect_silence_regions(c);
    bench_det_004_detect_dynamic_range(c);
    bench_det_005_detect_clipping(c);
    bench_det_006_estimate_noise_floor(c);
    #[cfg(feature = "transforms")]
    bench_det_007_estimate_frequency_range(c);
    bench_det_008_analyze_spectrum_for_cutoff(c);
    bench_det_009_detect_fundamental_autocorrelation(c);
}

// ===========================================================================
// Dispatch macros — DTYPES_DEFAULT-typed expansion for `audio.detect(...)`
// shape calls.
// ===========================================================================

macro_rules! dispatch_detect_sine {
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

// Variant for functions that need a typed sample threshold (e.g. detect_silence_regions).
macro_rules! dispatch_silence_sine {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |a: &mut AudioSamples<'_, i16>| {
                        black_box(detect_silence_regions(&*a, 1_000i16).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<I24>(n, ch),
                    |a: &mut AudioSamples<'_, I24>| {
                        // I24 threshold via CastFrom<i32>.
                        let thr: I24 = <I24 as CastFrom<i32>>::cast_from(1_000i32);
                        black_box(detect_silence_regions(&*a, thr).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i32>(n, ch),
                    |a: &mut AudioSamples<'_, i32>| {
                        black_box(detect_silence_regions(&*a, 1_000i32).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |a: &mut AudioSamples<'_, f32>| {
                        black_box(detect_silence_regions(&*a, 0.001_f32).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

// Variant for estimate_noise_floor (extra unused type param F).
macro_rules! dispatch_noise_floor_sine {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i16>(n, ch),
                    |a| {
                        black_box(estimate_noise_floor::<i16, ()>(&*a).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<I24>(n, ch),
                    |a| {
                        black_box(estimate_noise_floor::<I24, ()>(&*a).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<i32>(n, ch),
                    |a| {
                        black_box(estimate_noise_floor::<i32, ()>(&*a).ok());
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_a440::<f32>(n, ch),
                    |a| {
                        black_box(estimate_noise_floor::<f32, ()>(&*a).ok());
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
// Det-001 detect_sample_rate — `transforms`-gated; spectral peak heuristic.
// NoFast tier, lengths capped at NO_XXXL (FFT internally).
// ===========================================================================

#[cfg(feature = "transforms")]
fn bench_det_001_detect_sample_rate<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("det");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Det-001_detect_sample_rate", len, dt, ch);
                dispatch_detect_sine!(group, id, dt, len, ch, |a| {
                    detect_sample_rate(&*a).ok()
                });
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Det-002 detect_fundamental_frequency — autocorrelation-based pitch
// estimator; NoFast tier, lengths capped at NO_XXXL (search range × n).
// ===========================================================================

fn bench_det_002_detect_fundamental_frequency<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("det");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(
                    &mut group,
                    "Det-002_detect_fundamental_frequency",
                    len,
                    dt,
                    ch,
                );
                dispatch_detect_sine!(group, id, dt, len, ch, |a| {
                    detect_fundamental_frequency(&*a).ok()
                });
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Det-003 detect_silence_regions — FastSmall tier (single-pass with
// per-sample threshold compare).
// ===========================================================================

fn bench_det_003_detect_silence_regions<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("det");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id =
                    label_and_throughput(&mut group, "Det-003_detect_silence_regions", len, dt, ch);
                dispatch_silence_sine!(group, id, dt, len, ch);
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Det-004 detect_dynamic_range — two-pass (peak + RMS); NoFast tier.
// ===========================================================================

fn bench_det_004_detect_dynamic_range<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("det");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id =
                    label_and_throughput(&mut group, "Det-004_detect_dynamic_range", len, dt, ch);
                dispatch_detect_sine!(group, id, dt, len, ch, |a| {
                    detect_dynamic_range(&*a).ok()
                });
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Det-005 detect_clipping — FastSmall tier (single-pass threshold compare).
// ===========================================================================

fn bench_det_005_detect_clipping<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("det");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Det-005_detect_clipping", len, dt, ch);
                dispatch_detect_sine!(group, id, dt, len, ch, |a| {
                    detect_clipping(&*a, 0.99).ok()
                });
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Det-006 estimate_noise_floor — sort-based percentile; NoFast tier
// (O(n log n) allocation + sort).
// ===========================================================================

fn bench_det_006_estimate_noise_floor<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("det");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id =
                    label_and_throughput(&mut group, "Det-006_estimate_noise_floor", len, dt, ch);
                dispatch_noise_floor_sine!(group, id, dt, len, ch);
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Det-007 estimate_frequency_range — `transforms`-gated; FFT + linear scan.
// NoFast tier, lengths capped at NO_XXXL.
// ===========================================================================

#[cfg(feature = "transforms")]
fn bench_det_007_estimate_frequency_range<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("det");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_DEFAULT {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(
                    &mut group,
                    "Det-007_estimate_frequency_range",
                    len,
                    dt,
                    ch,
                );
                dispatch_detect_sine!(group, id, dt, len, ch, |a| {
                    estimate_frequency_range(&*a).ok()
                });
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Det-008 analyze_spectrum_for_cutoff — inner kernel; runs over a synthetic
// power-spectrum vector and a fixed 9-candidate list. Cost is essentially
// constant in `n_bins`, but the catalog asks for an n_bins sweep so we
// confirm that empirically. FastSmall tier.
// ===========================================================================

fn bench_det_008_analyze_spectrum_for_cutoff<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("det");
    // n_bins sweep: power-of-two spectrum lengths spanning practical FFT outputs.
    const NBINS_SWEEP: &[usize] = &[256, 1024, 4096, 16_384, 65_536];
    // The function only does ~9 candidate * ~15 reads of work; FastSmall fits.
    for (i, &n_bins) in NBINS_SWEEP.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        // Throughput in elements = spectrum bins inspected.
        group.throughput(Throughput::Elements(n_bins as u64));
        let label = ParamLabel::new().with("n_bins", n_bins).build();
        let id = BenchmarkId::new("Det-008_analyze_spectrum_for_cutoff", label);
        // Build a deterministic synthetic power spectrum (sloped envelope so
        // the heuristic has a chance to fire on at least one candidate rate).
        group.bench_with_input(id, &n_bins, |b, &n_bins| {
            b.iter_batched_ref(
                || -> (Vec<f64>, f64) {
                    let mut spectrum = Vec::with_capacity(n_bins);
                    for i in 0..n_bins {
                        // Exponential decay envelope across the bins.
                        let t = i as f64 / n_bins as f64;
                        spectrum.push((-3.0 * t).exp() + 1e-9);
                    }
                    // Nyquist for 48 kHz, plausible value for the heuristic.
                    (spectrum, 24_000.0)
                },
                |(spectrum, nyquist)| {
                    // SAFETY: spectrum length is `n_bins >= 256`, non-empty.
                    let nes = unsafe { NonEmptySlice::new_unchecked(&spectrum[..]) };
                    black_box(analyze_spectrum_for_cutoff(nes, *nyquist));
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ===========================================================================
// Det-009 detect_fundamental_autocorrelation — inner kernel; raw f64 buffer
// in, scalar out. Search range is fixed (50–2000 Hz at the bench sample
// rate ≈ 880 candidate periods); cost scales linearly with `len`. NoFast
// tier, lengths capped at NO_XXXL.
// ===========================================================================

fn bench_det_009_detect_fundamental_autocorrelation<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("det");
    let sample_rate_hz = 44_100.0_f64;
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        group.throughput(Throughput::Elements(len as u64));
        let label = ParamLabel::new().with("len", len).build();
        let id = BenchmarkId::new("Det-009_detect_fundamental_autocorrelation", label);
        group.bench_with_input(id, &len, |b, &len| {
            b.iter_batched_ref(
                || -> NonEmptyVec<f64> {
                    // 440 Hz sine at the bench sample rate, f64 directly so
                    // the kernel can be benched without any conversion cost.
                    let mut data = Vec::with_capacity(len);
                    let omega = 2.0 * std::f64::consts::PI * 440.0 / sample_rate_hz;
                    for n in 0..len {
                        data.push((omega * n as f64).sin());
                    }
                    // SAFETY: len ≥ 256 > 0.
                    unsafe { NonEmptyVec::new_unchecked(data) }
                },
                |data| {
                    black_box(
                        detect_fundamental_autocorrelation(&*data, sample_rate_hz).ok(),
                    );
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}
