//! Shared body for the `AudioDecomposition` bench targets
//! (`bench_dec_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/dec.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `Dec`
//! (lines 363-371) — a single function (`hpss`).
//!
//! ## Cost model
//!
//! HPSS = STFT + 2 median filters over an `n_bins × n_frames` matrix +
//! ISTFT, where `n_bins ≈ n_fft / 2 + 1` and `n_frames ≈ ceil(len / hop)`.
//! The dominant term at non-trivial kernel sizes is the median filter,
//! whose naive cost is `O(n_bins · n_frames · kernel)` per axis.
//!
//! Because of this, lengths are capped by kernel size — see
//! [`DEC_LENGTHS_K17`] (kernel = 17) and [`DEC_LENGTHS_K31`] (kernel = 31).
//! Both caps target < 30 s/point wall-clock at the largest length.

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use audio_samples::I24;
use audio_samples::operations::hpss::HpssConfig;
use audio_samples::operations::traits::AudioDecomposition;

use bench_suite_common::{
    CHANNELS_DEFAULT, DTYPES_DEFAULT, ParamLabel, SAMPLE_SIZE_DEFAULT, SAMPLE_SIZE_SLOW,
    fixture_a440,
};

// ===========================================================================
// Length sweeps — narrower than LENGTH_SWEEP_NO_XXXL because HPSS scales
// super-linearly in `len` (more frames → larger median-filter input).
// ===========================================================================

/// Length sweep for kernel = 17 — caps at 262_144 (M).  At this point a
/// typical (n_fft=2048, hop=512) configuration produces ~512 frames, and
/// median(17) over a 1025×512 matrix is well under 30 s.
const DEC_LENGTHS_K17: &[usize] = &[16_384, 65_536, 262_144];

/// Length sweep for kernel = 31 — caps at 65_536 (L).  Doubling either
/// the kernel or n_frames roughly doubles the dominant median cost; this
/// keeps the worst-case point under the 30 s/point budget.
const DEC_LENGTHS_K31: &[usize] = &[16_384, 65_536];

// ===========================================================================
// Top-level entry point — wrappers call into this with their typed Criterion.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_dec_001_hpss(c);
}

// ===========================================================================
// Dispatch macro — DTYPES_DEFAULT-typed expansion for a unary 440 Hz fixture.
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

/// HPSS config: n_fft = 2048, hop = 512, both axes' median kernels = `kernel`.
fn hpss_cfg_with_kernel(kernel: usize) -> HpssConfig {
    // Start from the `musical()` preset (n_fft=2048, hop=512) and override
    // the kernels.  Public fields per HpssConfig definition.
    let mut cfg = HpssConfig::musical();
    cfg.median_filter_harmonic = kernel;
    cfg.median_filter_percussive = kernel;
    cfg
}

// ===========================================================================
// Dec-001 hpss — STFT + dual median-filter HPSS pipeline.
// Sweep `kernel_size ∈ {17, 31}` × a kernel-dependent length sweep.
// NoFast tier; SLOW sample size at the largest length per sweep.
// ===========================================================================

fn bench_dec_001_hpss<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("dec");
    for &kernel in &[17usize, 31usize] {
        let cfg = hpss_cfg_with_kernel(kernel);
        let sweep: &[usize] = if kernel <= 17 {
            DEC_LENGTHS_K17
        } else {
            DEC_LENGTHS_K31
        };
        let last_idx = sweep.len() - 1;
        for (i, &len) in sweep.iter().enumerate() {
            // The largest length per kernel sweep gets the SLOW sample size;
            // smaller lengths use DEFAULT.  We don't use `sample_size_for`
            // directly because our custom sweep doesn't align with the
            // canonical 8-point index space.
            let sample_size = if i == last_idx {
                SAMPLE_SIZE_SLOW
            } else {
                SAMPLE_SIZE_DEFAULT
            };
            group.sample_size(sample_size);
            for &ch in CHANNELS_DEFAULT {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len).with("dt", dt).with("ch", ch)
                        .with("n_fft", 2048usize)
                        .with("hop", 512usize)
                        .with("kernel", kernel)
                        .build();
                    let id = BenchmarkId::new("Dec-001_hpss", label);
                    dispatch_unary_sine!(
                        group, id, dt, len, ch, |a| a.hpss(&cfg).ok()
                    );
                }
            }
        }
    }
    group.finish();
}
