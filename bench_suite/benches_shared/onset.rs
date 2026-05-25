//! Shared body for the `AudioOnsetDetection` bench targets
//! (`bench_onset_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Catalog section: Onset (lines 307-334 of CATALOG.md).
//! Methodology: bench_suite/METHODOLOGY.md.
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/onset.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Cost-axis notes:
//!
//! * Onset-001..008 — full trait pipelines with an STFT/CQT front-end. The
//!   `n_fft`/`hop_length` pair dominates per-frame cost, so each row sweeps
//!   `n_fft ∈ {1024, 2048}` with `hop = n_fft / 4` (75 % overlap).
//! * Onset-009..020 — leaf kernels that take FFT-frame inputs (Array2 of
//!   f64 magnitudes, Array2 of Complex<f64>, or NonEmptySlice<f64>). For
//!   these the cost axis is the number of frames, not the audio length;
//!   they get a short custom sweep `{64, 256, 1024, 4096}` frames with a
//!   fixed bin count.

use std::num::NonZeroUsize;

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use ndarray::Array2;
use non_empty_slice::{NonEmptySlice, NonEmptyVec};
use num_complex::Complex;
use std::hint::black_box;

use audio_samples::{
    AudioOnsetDetection, I24,
    operations::onset::{
        ComplexOnsetConfig, OnsetDetectionConfig, SpectralFluxConfig, SpectralFluxMethod,
        complex::{combine_complex_odf, magnitude_difference, phase_deviation},
        filters::{log_compress_inplace, median_filter, rectify_inplace},
        flux::{complex_flux, energy_flux, magnitude_flux, rectified_complex_flux},
        kernels::{apply_adaptive_threshold, energy_odf},
    },
    utils::generation::chirp,
};

use bench_suite_common::{
    CHANNELS_DEFAULT, DTYPES_DEFAULT, LENGTH_SWEEP_NO_XXXL, ParamLabel, SampleSizePolicy,
    fixture_a440, sample_size_for,
};

// ===========================================================================
// Top-level entry point — wrappers call into this with their typed Criterion.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    // Full-trait methods (Onset-001..008): STFT/CQT front-end + back-end.
    bench_onset_001_detect_onsets(c);
    bench_onset_002_onset_detection_function(c);
    bench_onset_003_detect_onsets_spectral_flux(c);
    bench_onset_004_spectral_flux(c);
    bench_onset_005_complex_onset_detection(c);
    bench_onset_006_onset_detection_function_complex(c);
    bench_onset_007_magnitude_difference_matrix(c);
    bench_onset_008_phase_deviation_matrix(c);

    // Leaf kernels (Onset-009..020): n_frames-driven cost.
    bench_onset_009_energy_flux(c);
    bench_onset_010_magnitude_flux(c);
    bench_onset_011_complex_flux(c);
    bench_onset_012_rectified_complex_flux(c);
    bench_onset_013_energy_odf(c);
    bench_onset_014_apply_adaptive_threshold(c);
    bench_onset_015_magnitude_difference(c);
    bench_onset_016_phase_deviation(c);
    bench_onset_017_combine_complex_odf(c);
    bench_onset_018_median_filter(c);
    bench_onset_019_rectify_inplace(c);
    bench_onset_020_log_compress_inplace(c);
}

// ===========================================================================
// Fixtures
// ===========================================================================

/// Frame-count sweep for leaf kernels — cost scales with `n_frames` rather
/// than audio sample length. Held mono and at a single bin count
/// (`KERNEL_N_BINS = 256`) so the only swept axis is frames.
const KERNEL_N_FRAMES: &[usize] = &[64, 256, 1_024, 4_096];

/// Fixed bin count for leaf-kernel synthetic inputs. Picked to be the
/// median of the real CQT bin counts produced by the trait benches at
/// `n_fft ∈ {1024, 2048}` (typically 128-512 bins per frame).
const KERNEL_N_BINS: usize = 256;

/// n_fft sweep for the trait benches. 1024 and 2048 are the two values
/// that dominate real onset-detection use; 4096 would be redundant given
/// we already sweep length.
const N_FFT_SWEEP: &[usize] = &[1_024, 2_048];

/// Build a real-valued magnitude-shaped `Array2<f64>` filled with a
/// deterministic ramp + sinusoidal modulation. Layout: `(bins, frames)`,
/// matching what the leaf kernels expect.
fn synth_magnitude(bins: usize, frames: usize) -> Array2<f64> {
    Array2::from_shape_fn((bins, frames), |(b, t)| {
        let base = (b as f64).mul_add(0.001, 1.0);
        let mod_ = ((t as f64 * 0.1).sin() + 1.0) * 0.5;
        base * mod_
    })
}

/// Build a complex spectrum-shaped `Array2<Complex<f64>>`. Layout:
/// `(bins, frames)`.
fn synth_complex(bins: usize, frames: usize) -> Array2<Complex<f64>> {
    Array2::from_shape_fn((bins, frames), |(b, t)| {
        let r = (b as f64).mul_add(0.001, 1.0);
        let theta = (b as f64 * 0.05) + (t as f64 * 0.1);
        Complex::new(r * theta.cos(), r * theta.sin())
    })
}

/// Build a 1-D non-empty f64 ramp signal. Used for `median_filter`,
/// `rectify_inplace`, `log_compress_inplace`, `apply_adaptive_threshold`.
fn synth_signal(n: usize) -> NonEmptyVec<f64> {
    debug_assert!(n > 0, "synth_signal requires n > 0");
    let v: Vec<f64> = (0..n)
        .map(|i| ((i as f64 * 0.1).sin() - 0.5) * (1.0 + (i % 7) as f64))
        .collect();
    NonEmptyVec::new(v).expect("synth_signal: n > 0 guarantees non-empty")
}

/// Convenience: construct an `OnsetDetectionConfig` based on the `percussive`
/// preset but with overridden window/hop. The window is set explicitly so
/// the bench measures the n_fft axis directly.
fn onset_cfg_with(n_fft: usize) -> OnsetDetectionConfig {
    let mut cfg = OnsetDetectionConfig::percussive();
    cfg.window_size = Some(NonZeroUsize::new(n_fft).expect("n_fft > 0"));
    cfg.hop_size = NonZeroUsize::new(n_fft / 4).expect("n_fft/4 > 0");
    cfg
}

fn flux_cfg_with(n_fft: usize, method: SpectralFluxMethod) -> SpectralFluxConfig {
    let mut cfg = SpectralFluxConfig::musical();
    cfg.window_size = Some(NonZeroUsize::new(n_fft).expect("n_fft > 0"));
    cfg.hop_size = NonZeroUsize::new(n_fft / 4).expect("n_fft/4 > 0");
    cfg.flux_method = method;
    cfg
}

fn complex_cfg_with(n_fft: usize) -> ComplexOnsetConfig {
    let mut cfg = ComplexOnsetConfig::musical();
    cfg.window_size = Some(NonZeroUsize::new(n_fft).expect("n_fft > 0"));
    cfg.hop_size = NonZeroUsize::new(n_fft / 4).expect("n_fft/4 > 0");
    cfg
}

// ===========================================================================
// Dispatch macros — DTYPES_DEFAULT-typed expansion.
// ===========================================================================

/// Bench an `audio.method(&cfg)` on a chirp fixture, swept over the four
/// default dtypes. The chirp gives spectral evolution so onset detectors
/// actually exercise the peak-picking branch (a pure sine produces no
/// onsets and short-circuits).
///
/// For mono we use `chirp`; for stereo we fall back to `fixture_a440`
/// (the `chirp` generator is mono-only). The cost is dominated by the
/// STFT/CQT and downstream kernels, which see both channels.
macro_rules! dispatch_trait_method {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, $cfg:expr, |$audio:ident, $c:ident| $body:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || onset_fixture::<i16>(n, ch),
                    |$audio| {
                        let $c = $cfg;
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || onset_fixture::<I24>(n, ch),
                    |$audio| {
                        let $c = $cfg;
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || onset_fixture::<i32>(n, ch),
                    |$audio| {
                        let $c = $cfg;
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || onset_fixture::<f32>(n, ch),
                    |$audio| {
                        let $c = $cfg;
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            _ => unreachable!("DTYPES_DEFAULT changed without updating dispatcher"),
        }
    }};
}

/// Build an onset-friendly fixture: mono uses [`chirp`] (linear sweep
/// 100→8000 Hz, which produces continuous spectral change for the ODF to
/// chew on); stereo falls back to the suite-standard `fixture_a440` since
/// `chirp` is mono-only.
fn onset_fixture<T>(n_samples: usize, channels: usize) -> audio_samples::AudioSamples<'static, T>
where
    T: audio_samples::StandardSample + 'static,
{
    use std::time::Duration;

    use audio_samples::sample_rate;

    if channels == 1 {
        let sr = sample_rate!(44_100);
        let duration = Duration::from_secs_f64(n_samples as f64 / 44_100.0);
        chirp::<T>(100.0, 8_000.0, duration, sr, 0.9)
    } else {
        fixture_a440::<T>(n_samples, channels)
    }
}

// ===========================================================================
// Onset-001 detect_onsets — full pipeline (STFT + ODF + peak picking).
// ===========================================================================

fn bench_onset_001_detect_onsets<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("onset");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &n_fft in N_FFT_SWEEP {
            for &ch in CHANNELS_DEFAULT {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("n_fft", n_fft)
                        .build();
                    let id = BenchmarkId::new("Onset-001_detect_onsets", label);
                    dispatch_trait_method!(
                        group,
                        id,
                        dt,
                        len,
                        ch,
                        onset_cfg_with(n_fft),
                        |a, cfg| a.detect_onsets(&cfg).ok()
                    );
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Onset-002 onset_detection_function — ODF curve only (no peak picking).
// ===========================================================================

fn bench_onset_002_onset_detection_function<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("onset");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &n_fft in N_FFT_SWEEP {
            for &ch in CHANNELS_DEFAULT {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("n_fft", n_fft)
                        .build();
                    let id = BenchmarkId::new("Onset-002_onset_detection_function", label);
                    dispatch_trait_method!(
                        group,
                        id,
                        dt,
                        len,
                        ch,
                        onset_cfg_with(n_fft),
                        |a, cfg| a.onset_detection_function(&cfg).ok()
                    );
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Onset-003 detect_onsets_spectral_flux — magnitude-flux pipeline.
// ===========================================================================

fn bench_onset_003_detect_onsets_spectral_flux<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("onset");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &n_fft in N_FFT_SWEEP {
            for &ch in CHANNELS_DEFAULT {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("n_fft", n_fft)
                        .build();
                    let id = BenchmarkId::new("Onset-003_detect_onsets_spectral_flux", label);
                    dispatch_trait_method!(
                        group,
                        id,
                        dt,
                        len,
                        ch,
                        flux_cfg_with(n_fft, SpectralFluxMethod::Magnitude),
                        |a, cfg| a.detect_onsets_spectral_flux(&cfg).ok()
                    );
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Onset-004 spectral_flux — flux curve only. Trait signature splits the
// config across four args (cqt_params + window + hop + method).
// ===========================================================================

fn bench_onset_004_spectral_flux<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("onset");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &n_fft in N_FFT_SWEEP {
            for &ch in CHANNELS_DEFAULT {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("n_fft", n_fft)
                        .build();
                    let id = BenchmarkId::new("Onset-004_spectral_flux", label);
                    dispatch_trait_method!(
                        group,
                        id,
                        dt,
                        len,
                        ch,
                        flux_cfg_with(n_fft, SpectralFluxMethod::Magnitude),
                        |a, cfg| a.spectral_flux(
                            &cfg.cqt_params,
                            cfg.window_size.expect("set above"),
                            cfg.hop_size,
                            cfg.flux_method,
                        )
                        .ok()
                    );
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Onset-005 complex_onset_detection — complex ODF + peak picking.
// ===========================================================================

fn bench_onset_005_complex_onset_detection<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("onset");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &n_fft in N_FFT_SWEEP {
            for &ch in CHANNELS_DEFAULT {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("n_fft", n_fft)
                        .build();
                    let id = BenchmarkId::new("Onset-005_complex_onset_detection", label);
                    dispatch_trait_method!(
                        group,
                        id,
                        dt,
                        len,
                        ch,
                        complex_cfg_with(n_fft),
                        |a, cfg| a.complex_onset_detection(&cfg).ok()
                    );
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Onset-006 onset_detection_function_complex — complex ODF curve only.
// ===========================================================================

fn bench_onset_006_onset_detection_function_complex<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("onset");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &n_fft in N_FFT_SWEEP {
            for &ch in CHANNELS_DEFAULT {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("n_fft", n_fft)
                        .build();
                    let id =
                        BenchmarkId::new("Onset-006_onset_detection_function_complex", label);
                    dispatch_trait_method!(
                        group,
                        id,
                        dt,
                        len,
                        ch,
                        complex_cfg_with(n_fft),
                        |a, cfg| a.onset_detection_function_complex(&cfg).ok()
                    );
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Onset-007 magnitude_difference_matrix — CQT mag + frame-to-frame diff.
// ===========================================================================

fn bench_onset_007_magnitude_difference_matrix<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("onset");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &n_fft in N_FFT_SWEEP {
            for &ch in CHANNELS_DEFAULT {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("n_fft", n_fft)
                        .build();
                    let id = BenchmarkId::new("Onset-007_magnitude_difference_matrix", label);
                    dispatch_trait_method!(
                        group,
                        id,
                        dt,
                        len,
                        ch,
                        complex_cfg_with(n_fft),
                        |a, cfg| a.magnitude_difference_matrix(&cfg).ok()
                    );
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Onset-008 phase_deviation_matrix — CQT complex + phase deviation.
// ===========================================================================

fn bench_onset_008_phase_deviation_matrix<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("onset");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &n_fft in N_FFT_SWEEP {
            for &ch in CHANNELS_DEFAULT {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("n_fft", n_fft)
                        .build();
                    let id = BenchmarkId::new("Onset-008_phase_deviation_matrix", label);
                    dispatch_trait_method!(
                        group,
                        id,
                        dt,
                        len,
                        ch,
                        complex_cfg_with(n_fft),
                        |a, cfg| a.phase_deviation_matrix(&cfg).ok()
                    );
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Leaf-kernel section. Cost axis = `n_frames`; fixed `KERNEL_N_BINS`.
//
// All leaf kernels operate on f64 data only (no integer dtype dispatch).
// Channels are 1 (no concept of channels at this layer).
// ===========================================================================

fn kernel_label(catalog_id: &str, frames: usize, bins: usize) -> BenchmarkId {
    let label = ParamLabel::new()
        .with("n_frames", frames)
        .with("n_bins", bins)
        .build();
    BenchmarkId::new(catalog_id, label)
}

// ----- Onset-009 energy_flux (mag spectrogram → flux curve) ------------------

fn bench_onset_009_energy_flux<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("onset");
    group.sample_size(sample_size_for(SampleSizePolicy::NoFast, 2));
    for &frames in KERNEL_N_FRAMES {
        group.throughput(Throughput::Elements((frames * KERNEL_N_BINS) as u64));
        let id = kernel_label("Onset-009_energy_flux", frames, KERNEL_N_BINS);
        group.bench_with_input(id, &frames, |b, &frames| {
            b.iter_batched_ref(
                || synth_magnitude(KERNEL_N_BINS, frames),
                |mag| {
                    black_box(energy_flux(mag));
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ----- Onset-010 magnitude_flux ----------------------------------------------

fn bench_onset_010_magnitude_flux<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("onset");
    group.sample_size(sample_size_for(SampleSizePolicy::NoFast, 2));
    for &frames in KERNEL_N_FRAMES {
        group.throughput(Throughput::Elements((frames * KERNEL_N_BINS) as u64));
        let id = kernel_label("Onset-010_magnitude_flux", frames, KERNEL_N_BINS);
        group.bench_with_input(id, &frames, |b, &frames| {
            b.iter_batched_ref(
                || synth_magnitude(KERNEL_N_BINS, frames),
                |mag| {
                    black_box(magnitude_flux(mag));
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ----- Onset-011 complex_flux (complex spectrum → flux curve) ----------------

fn bench_onset_011_complex_flux<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("onset");
    group.sample_size(sample_size_for(SampleSizePolicy::NoFast, 2));
    for &frames in KERNEL_N_FRAMES {
        group.throughput(Throughput::Elements((frames * KERNEL_N_BINS) as u64));
        let id = kernel_label("Onset-011_complex_flux", frames, KERNEL_N_BINS);
        group.bench_with_input(id, &frames, |b, &frames| {
            b.iter_batched_ref(
                || synth_complex(KERNEL_N_BINS, frames),
                |spec| {
                    black_box(complex_flux(spec));
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ----- Onset-012 rectified_complex_flux --------------------------------------

fn bench_onset_012_rectified_complex_flux<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("onset");
    group.sample_size(sample_size_for(SampleSizePolicy::NoFast, 2));
    for &frames in KERNEL_N_FRAMES {
        group.throughput(Throughput::Elements((frames * KERNEL_N_BINS) as u64));
        let id = kernel_label("Onset-012_rectified_complex_flux", frames, KERNEL_N_BINS);
        group.bench_with_input(id, &frames, |b, &frames| {
            b.iter_batched_ref(
                || synth_complex(KERNEL_N_BINS, frames),
                |spec| {
                    black_box(rectified_complex_flux(spec));
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ----- Onset-013 energy_odf (mag → ODF) --------------------------------------

fn bench_onset_013_energy_odf<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("onset");
    group.sample_size(sample_size_for(SampleSizePolicy::NoFast, 2));
    for &frames in KERNEL_N_FRAMES {
        group.throughput(Throughput::Elements((frames * KERNEL_N_BINS) as u64));
        let id = kernel_label("Onset-013_energy_odf", frames, KERNEL_N_BINS);
        group.bench_with_input(id, &frames, |b, &frames| {
            b.iter_batched_ref(
                || synth_magnitude(KERNEL_N_BINS, frames),
                |mag| {
                    black_box(energy_odf(mag));
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ----- Onset-014 apply_adaptive_threshold ------------------------------------
//
// Cost axis is `n_frames` only; no bin dimension. The kernel walks the
// signal once, comparing each sample to a precomputed median.

fn bench_onset_014_apply_adaptive_threshold<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("onset");
    group.sample_size(sample_size_for(SampleSizePolicy::NoFast, 2));
    for &frames in KERNEL_N_FRAMES {
        group.throughput(Throughput::Elements(frames as u64));
        let label = ParamLabel::new().with("n_frames", frames).build();
        let id = BenchmarkId::new("Onset-014_apply_adaptive_threshold", label);
        group.bench_with_input(id, &frames, |b, &frames| {
            b.iter_batched_ref(
                || (synth_signal(frames), synth_signal(frames)),
                |inp| {
                    let (sig, med) = inp;
                    apply_adaptive_threshold(sig, med, 1.5);
                    black_box(&sig);
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ----- Onset-015 magnitude_difference (Array2 → Array2) ---------------------

fn bench_onset_015_magnitude_difference<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("onset");
    group.sample_size(sample_size_for(SampleSizePolicy::NoFast, 2));
    for &frames in KERNEL_N_FRAMES {
        group.throughput(Throughput::Elements((frames * KERNEL_N_BINS) as u64));
        let id = kernel_label("Onset-015_magnitude_difference", frames, KERNEL_N_BINS);
        group.bench_with_input(id, &frames, |b, &frames| {
            b.iter_batched_ref(
                || synth_magnitude(KERNEL_N_BINS, frames),
                |mag| {
                    black_box(magnitude_difference(mag.view()));
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ----- Onset-016 phase_deviation --------------------------------------------
//
// Needs a `ComplexOnsetConfig` for bin frequencies + hop. We re-use the
// musical preset and a fixed sample_rate.

fn bench_onset_016_phase_deviation<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("onset");
    group.sample_size(sample_size_for(SampleSizePolicy::NoFast, 2));
    let sample_rate = 44_100.0_f64;
    let cfg = ComplexOnsetConfig::musical();
    for &frames in KERNEL_N_FRAMES {
        group.throughput(Throughput::Elements((frames * KERNEL_N_BINS) as u64));
        let id = kernel_label("Onset-016_phase_deviation", frames, KERNEL_N_BINS);
        group.bench_with_input(id, &frames, |b, &frames| {
            b.iter_batched_ref(
                || synth_complex(KERNEL_N_BINS, frames),
                |spec| {
                    black_box(phase_deviation(spec.view(), &cfg, sample_rate));
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ----- Onset-017 combine_complex_odf ----------------------------------------
//
// Takes two `Array2<f64>` (mag_diff and phase_dev) and a config. Cost
// scales with frames × bins.

fn bench_onset_017_combine_complex_odf<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("onset");
    group.sample_size(sample_size_for(SampleSizePolicy::NoFast, 2));
    let cfg = ComplexOnsetConfig::musical();
    for &frames in KERNEL_N_FRAMES {
        group.throughput(Throughput::Elements((frames * KERNEL_N_BINS) as u64));
        let id = kernel_label("Onset-017_combine_complex_odf", frames, KERNEL_N_BINS);
        group.bench_with_input(id, &frames, |b, &frames| {
            b.iter_batched_ref(
                || {
                    (
                        synth_magnitude(KERNEL_N_BINS, frames),
                        synth_magnitude(KERNEL_N_BINS, frames),
                    )
                },
                |inp| {
                    let (mag_diff, phase_dev) = inp;
                    black_box(combine_complex_odf(mag_diff, phase_dev, &cfg));
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ----- Onset-018 median_filter ----------------------------------------------
//
// O(N·k) naive impl. Bench with kernel sizes 3 and 11 (both odd, both
// within the typical adaptive-threshold range). Cost axis is `n_frames`
// only; no bin dimension.

fn bench_onset_018_median_filter<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("onset");
    group.sample_size(sample_size_for(SampleSizePolicy::NoFast, 2));
    for &kernel_size in &[3_usize, 11] {
        for &frames in KERNEL_N_FRAMES {
            group.throughput(Throughput::Elements(frames as u64));
            let label = ParamLabel::new()
                .with("n_frames", frames)
                .with("kernel", kernel_size)
                .build();
            let id = BenchmarkId::new("Onset-018_median_filter", label);
            let k = NonZeroUsize::new(kernel_size).expect("kernel_size > 0");
            group.bench_with_input(id, &frames, |b, &frames| {
                b.iter_batched_ref(
                    || synth_signal(frames),
                    |sig: &mut NonEmptyVec<f64>| {
                        let slice: &NonEmptySlice<f64> = sig;
                        black_box(median_filter(slice, k).ok());
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

// ----- Onset-019 rectify_inplace --------------------------------------------

fn bench_onset_019_rectify_inplace<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("onset");
    group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, 2));
    for &frames in KERNEL_N_FRAMES {
        group.throughput(Throughput::Elements(frames as u64));
        let label = ParamLabel::new().with("n_frames", frames).build();
        let id = BenchmarkId::new("Onset-019_rectify_inplace", label);
        group.bench_with_input(id, &frames, |b, &frames| {
            b.iter_batched_ref(
                || synth_signal(frames),
                |sig| {
                    rectify_inplace(sig);
                    black_box(&sig);
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

// ----- Onset-020 log_compress_inplace ---------------------------------------

fn bench_onset_020_log_compress_inplace<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("onset");
    group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, 2));
    for &frames in KERNEL_N_FRAMES {
        group.throughput(Throughput::Elements(frames as u64));
        let label = ParamLabel::new().with("n_frames", frames).build();
        let id = BenchmarkId::new("Onset-020_log_compress_inplace", label);
        group.bench_with_input(id, &frames, |b, &frames| {
            b.iter_batched_ref(
                || synth_signal(frames),
                |sig| {
                    log_compress_inplace(sig, 100.0);
                    black_box(&sig);
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}
