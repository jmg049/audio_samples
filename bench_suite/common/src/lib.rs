//! Shared helpers for the `audio_samples` benchmark suite.
//!
//! Encodes the conventions documented in `bench_suite/METHODOLOGY.md` —
//! length sweeps, sample-size tiers, parameter-string format, the canonical
//! Criterion config (WallTime + four PMU variants), and the deterministic
//! signal fixtures.
//!
//! See `bench_suite/METHODOLOGY.md` and `bench_suite/CATALOG.md` for the
//! design context. Changing a constant here ripples through every bench;
//! adjust the methodology document first.

#![allow(clippy::missing_panics_doc)]

use std::fmt::Write as _;
use std::num::NonZeroU32;
use std::path::Path;
use std::time::Duration;

use audio_samples::{
    AudioSamples, I24, StandardSample,
    utils::generation::{sine_wave, stereo_sine_wave},
};
use criterion::{Criterion, PlottingBackend};

// ===========================================================================
// Sample-size tiers (METHODOLOGY §3.4)
// ===========================================================================

/// Sample count for sub-µs functions. Tail focus: 500 samples gives ~5 data
/// points above empirical p99.
pub const SAMPLE_SIZE_FAST: usize = 500;

/// Default sample count for the suite. Doubles Criterion's stock default of
/// 100; tighter CIs at the cost of bench wall-clock time.
pub const SAMPLE_SIZE_DEFAULT: usize = 200;

/// Sample count for slow functions (per-iter ≳ 500 ms). Tail behaviour at
/// this regime is dominated by OS scheduling, not the algorithm.
pub const SAMPLE_SIZE_SLOW: usize = 75;

// ===========================================================================
// Length sweep presets (METHODOLOGY §5.1)
// ===========================================================================

/// Eight log-spaced power-of-two lengths, per-channel:
/// 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304.
pub const LENGTH_SWEEP_FULL: &[usize] = &[
    256, 1024, 4096, 16_384, 65_536, 262_144, 1_048_576, 4_194_304,
];

/// First seven of the full sweep. Use for FFT-cost-prohibitive functions
/// where the XXXL point would take minutes per parameter combination.
pub const LENGTH_SWEEP_NO_XXXL: &[usize] = &[
    256, 1024, 4096, 16_384, 65_536, 262_144, 1_048_576,
];

// ===========================================================================
// Dtype labels (METHODOLOGY §5.3)
// ===========================================================================

/// Four-dtype default sweep: integer PCM at three widths + DSP-native float.
pub const DTYPES_DEFAULT: &[&str] = &["i16", "I24", "i32", "f32"];

/// All six supported sample types. Use for conversion benches and SIMD
/// fast-path benches where `u8` and `f64` materially change the answer.
pub const DTYPES_ALL: &[&str] = &["u8", "i16", "I24", "i32", "f32", "f64"];

// ===========================================================================
// Channel layout presets (METHODOLOGY §5.4)
// ===========================================================================

/// `{mono, stereo}`.
pub const CHANNELS_DEFAULT: &[usize] = &[1, 2];

/// `{mono, stereo, 5.1}`. Use for channel-ops benches where 5.1 changes the
/// per-call cost more than linearly.
pub const CHANNELS_INCLUDING_SURROUND: &[usize] = &[1, 2, 6];

// ===========================================================================
// Bench sample rate (METHODOLOGY §5.2)
// ===========================================================================

/// Default sample rate for benches whose cost is driven by `total_samples`
/// rather than sample rate. Only resampling / coefficient-design benches
/// should deviate.
pub const BENCH_SAMPLE_RATE_HZ: u32 = 44_100;

/// Sample rate sweep for benches where SR genuinely affects cost.
pub const SAMPLE_RATE_SWEEP: &[u32] = &[16_000, 22_050, 44_100, 48_000, 96_000];

#[inline]
fn bench_sample_rate() -> NonZeroU32 {
    // SAFETY: BENCH_SAMPLE_RATE_HZ is a non-zero constant.
    unsafe { NonZeroU32::new_unchecked(BENCH_SAMPLE_RATE_HZ) }
}

// ===========================================================================
// Criterion config: WallTime (METHODOLOGY §3.1)
// ===========================================================================

/// Canonical Criterion config for WallTime benches. Headless (no plots),
/// 200 samples, 10 s measurement budget, 2 % noise threshold. Writes to
/// `target/criterion_walltime/` so PMU runs (which use the same
/// `benchmark_group("stats")` etc.) land in disjoint trees and don't
/// overwrite each other.
///
/// Always finish the chain with `configure_from_args()` so CLI overrides
/// apply on top.
#[must_use]
pub fn build_criterion() -> Criterion {
    Criterion::default()
        .output_directory(Path::new("target/criterion_walltime"))
        .without_plots()
        .plotting_backend(PlottingBackend::None)
        .sample_size(SAMPLE_SIZE_DEFAULT)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(10))
        .nresamples(100_000)
        .noise_threshold(0.02)
        .confidence_level(0.95)
        .significance_level(0.05)
        .configure_from_args()
}

// ===========================================================================
// Criterion config: PMU (METHODOLOGY §3.5) — Linux + perf_events feature
// ===========================================================================

#[cfg(all(feature = "perf_events", target_os = "linux"))]
mod perf {
    use std::path::Path;
    use std::time::Duration;

    use criterion::{Criterion, PlottingBackend};
    use criterion_perf_events::Perf;
    use perfcnt::linux::{
        CacheId, CacheOpId, CacheOpResultId, HardwareEventType,
        PerfCounterBuilderLinux as B,
    };

    fn base(perf: Perf, output_dir: &str) -> Criterion<Perf> {
        Criterion::default()
            .with_measurement(perf)
            .output_directory(Path::new(output_dir))
            .without_plots()
            .plotting_backend(PlottingBackend::None)
            .sample_size(super::SAMPLE_SIZE_DEFAULT)
            .warm_up_time(Duration::from_secs(3))
            .measurement_time(Duration::from_secs(10))
            .nresamples(100_000)
            .noise_threshold(0.02)
            .confidence_level(0.95)
            .significance_level(0.05)
            .configure_from_args()
    }

    #[must_use]
    pub fn build_instructions() -> Criterion<Perf> {
        base(
            Perf::new(B::from_hardware_event(HardwareEventType::Instructions)),
            "target/criterion_instructions",
        )
    }

    #[must_use]
    pub fn build_cycles() -> Criterion<Perf> {
        base(
            Perf::new(B::from_hardware_event(HardwareEventType::CPUCycles)),
            "target/criterion_cycles",
        )
    }

    #[must_use]
    pub fn build_cache_misses() -> Criterion<Perf> {
        base(
            Perf::new(B::from_cache_event(
                CacheId::L1D, CacheOpId::Read, CacheOpResultId::Miss,
            )),
            "target/criterion_cache_misses",
        )
    }

    #[must_use]
    pub fn build_branch_misses() -> Criterion<Perf> {
        base(
            Perf::new(B::from_hardware_event(HardwareEventType::BranchMisses)),
            "target/criterion_branch_misses",
        )
    }
}

#[cfg(all(feature = "perf_events", target_os = "linux"))]
pub use perf::{
    build_branch_misses as build_criterion_perf_branch_misses,
    build_cache_misses as build_criterion_perf_cache_misses,
    build_cycles as build_criterion_perf_cycles,
    build_instructions as build_criterion_perf_instructions,
};

// ===========================================================================
// dtype_label (METHODOLOGY §7.3)
// ===========================================================================

/// Canonical string label for a sample type `T`, as used in bench parameter
/// strings (`dt=` value).
#[must_use]
pub fn dtype_label<T: 'static>() -> &'static str {
    use std::any::TypeId;
    let tid = TypeId::of::<T>();
    if tid == TypeId::of::<u8>() {
        "u8"
    } else if tid == TypeId::of::<i16>() {
        "i16"
    } else if tid == TypeId::of::<I24>() {
        "I24"
    } else if tid == TypeId::of::<i32>() {
        "i32"
    } else if tid == TypeId::of::<f32>() {
        "f32"
    } else if tid == TypeId::of::<f64>() {
        "f64"
    } else {
        "unknown"
    }
}

// ===========================================================================
// ParamLabel — sorted key=value string builder (METHODOLOGY §7.3)
// ===========================================================================

/// Builder for the canonical parameter-string format
/// `key1=v1,key2=v2,...` (keys sorted lexicographically).
///
/// The format is consumed verbatim as Criterion's "value" string in
/// `BenchmarkId::new(function_id, label)`, flows into `raw.csv`'s `value`
/// column, and is parsed back out by the harvester into typed columns.
#[derive(Debug, Default)]
pub struct ParamLabel {
    entries: Vec<(&'static str, String)>,
}

impl ParamLabel {
    #[must_use]
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Add a `key=value` entry. `value` can be any `Display`.
    #[must_use]
    pub fn with<V: std::fmt::Display>(mut self, key: &'static str, value: V) -> Self {
        self.entries.push((key, value.to_string()));
        self
    }

    /// Render to the canonical `k=v,k=v,...` string.
    #[must_use]
    pub fn build(mut self) -> String {
        self.entries.sort_by_key(|(k, _)| *k);
        let mut out = String::with_capacity(self.entries.len() * 16);
        for (i, (k, v)) in self.entries.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            write!(out, "{k}={v}").expect("infallible write to String");
        }
        out
    }
}

// ===========================================================================
// Fixtures (METHODOLOGY §8.2)
// ===========================================================================

/// Deterministic sine-wave fixture. Mono uses [`sine_wave`]; stereo uses
/// [`stereo_sine_wave`] (left = sine, right = cosine — fixed phase offset
/// so the two channels are not bit-identical).
///
/// `n_samples` is the per-channel sample count. Channel counts other than
/// 1 or 2 are not supported by this helper; add a dedicated multichannel
/// fixture if a bench needs surround.
#[must_use]
pub fn fixture_sine<T>(
    freq_hz: f64,
    n_samples: usize,
    sr: NonZeroU32,
    channels: usize,
) -> AudioSamples<'static, T>
where
    T: StandardSample + 'static,
{
    let sr_f = f64::from(sr.get());
    let duration = Duration::from_secs_f64(n_samples as f64 / sr_f);
    match channels {
        1 => sine_wave::<T>(freq_hz, duration, sr, 1.0),
        2 => stereo_sine_wave::<T>(freq_hz, duration, sr, 1.0),
        n => panic!(
            "fixture_sine: channel count {n} not supported (only 1 or 2). \
             Add a multichannel helper if a bench needs it."
        ),
    }
}

/// Convenience: build a 440 Hz fixture at the default bench sample rate.
#[must_use]
pub fn fixture_a440<T>(n_samples: usize, channels: usize) -> AudioSamples<'static, T>
where
    T: StandardSample + 'static,
{
    fixture_sine::<T>(440.0, n_samples, bench_sample_rate(), channels)
}

// ===========================================================================
// Seeded RNG
// ===========================================================================

/// Deterministic RNG seed used by every bench that needs randomness.
pub const BENCH_RNG_SEED: u64 = 0xA5C3_F00D_DEAD_BEEF;

/// Returns a fresh `StdRng` seeded with [`BENCH_RNG_SEED`].
#[must_use]
pub fn seeded_rng() -> rand::rngs::StdRng {
    use rand::SeedableRng;
    rand::rngs::StdRng::seed_from_u64(BENCH_RNG_SEED)
}

// ===========================================================================
// Sample-size tier selector (METHODOLOGY §3.4)
// ===========================================================================

/// Sample-size policy. Each variant maps a length-index in `LENGTH_SWEEP_FULL`
/// (0 = XXS, 7 = XXXL) to the appropriate `SAMPLE_SIZE_*` constant.
#[derive(Copy, Clone, Debug)]
pub enum SampleSizePolicy {
    /// Fast tier for sub-µs ops: FAST at indices 0..=2, DEFAULT at 3..=5,
    /// SLOW at 6..=7.
    FastSmall,
    /// No FAST tier: DEFAULT at 0..=5, SLOW at 6..=7. Use for ops that are
    /// not sub-µs even at small lengths (two-pass / branching / FFT).
    NoFast,
    /// Always DEFAULT. Use when the entire sweep sits in the same regime.
    AlwaysDefault,
}

#[must_use]
pub fn sample_size_for(policy: SampleSizePolicy, length_idx: usize) -> usize {
    match policy {
        SampleSizePolicy::FastSmall => match length_idx {
            0..=2 => SAMPLE_SIZE_FAST,
            3..=5 => SAMPLE_SIZE_DEFAULT,
            _ => SAMPLE_SIZE_SLOW,
        },
        SampleSizePolicy::NoFast => match length_idx {
            0..=5 => SAMPLE_SIZE_DEFAULT,
            _ => SAMPLE_SIZE_SLOW,
        },
        SampleSizePolicy::AlwaysDefault => SAMPLE_SIZE_DEFAULT,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_label_sorts_and_joins() {
        let s = ParamLabel::new()
            .with("len", 1024)
            .with("ch", 2)
            .with("dt", "f32")
            .build();
        assert_eq!(s, "ch=2,dt=f32,len=1024");
    }

    #[test]
    fn dtype_labels_match_dtypes_default() {
        assert_eq!(dtype_label::<i16>(), "i16");
        assert_eq!(dtype_label::<I24>(), "I24");
        assert_eq!(dtype_label::<i32>(), "i32");
        assert_eq!(dtype_label::<f32>(), "f32");
    }

    #[test]
    fn length_sweep_is_powers_of_two() {
        for &len in LENGTH_SWEEP_FULL {
            assert!(len.is_power_of_two(), "{len} is not a power of two");
        }
    }

    #[test]
    fn sample_size_policy_fast_small() {
        assert_eq!(sample_size_for(SampleSizePolicy::FastSmall, 0), SAMPLE_SIZE_FAST);
        assert_eq!(sample_size_for(SampleSizePolicy::FastSmall, 2), SAMPLE_SIZE_FAST);
        assert_eq!(sample_size_for(SampleSizePolicy::FastSmall, 3), SAMPLE_SIZE_DEFAULT);
        assert_eq!(sample_size_for(SampleSizePolicy::FastSmall, 5), SAMPLE_SIZE_DEFAULT);
        assert_eq!(sample_size_for(SampleSizePolicy::FastSmall, 6), SAMPLE_SIZE_SLOW);
        assert_eq!(sample_size_for(SampleSizePolicy::FastSmall, 7), SAMPLE_SIZE_SLOW);
    }

    #[test]
    fn fixture_sine_produces_correct_length() {
        let audio = fixture_a440::<f32>(1024, 1);
        assert_eq!(audio.samples_per_channel().get(), 1024);
        assert_eq!(audio.num_channels().get(), 1);
    }

    #[test]
    fn fixture_sine_stereo() {
        let audio = fixture_a440::<f32>(1024, 2);
        assert_eq!(audio.samples_per_channel().get(), 1024);
        assert_eq!(audio.num_channels().get(), 2);
    }
}
