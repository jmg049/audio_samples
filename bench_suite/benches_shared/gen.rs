//! Shared body for the signal-generation bench targets
//! (`bench_gen_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/gen.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `Gen`
//! (lines 566-592).
//!
//! Note: these benches measure GENERATORS, not transformations of an
//! existing fixture. The work is the generator call (which allocates its
//! own output buffer); setup is trivial. We use plain `b.iter(|| ...)`
//! rather than `iter_batched_ref` since the allocator is part of what users
//! pay.

use std::hint::black_box;
use std::num::NonZeroU32;
use std::time::Duration;

use criterion::measurement::Measurement;
use criterion::{BenchmarkId, Criterion, Throughput};

use audio_samples::utils::generation::{
    am_signal, brown_noise, chirp, compound_tone, cosine_wave, exponential_bursts, impulse,
    multichannel_compound_tone, pink_noise, sawtooth_wave, silence, sine_wave, square_wave,
    stereo_chirp, stereo_silence, stereo_sine_wave, triangle_wave, white_noise, ToneComponent,
};
use audio_samples::I24;
use non_empty_slice::NonEmptySlice;

use bench_suite_common::{
    BENCH_RNG_SEED, BENCH_SAMPLE_RATE_HZ, DTYPES_DEFAULT, LENGTH_SWEEP_FULL, ParamLabel,
    SampleSizePolicy, sample_size_for,
};

// ===========================================================================
// Top-level entry point — wrappers call into this with their typed Criterion.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_gen_001_sine_wave(c);
    bench_gen_002_cosine_wave(c);
    bench_gen_003_square_wave(c);
    bench_gen_004_sawtooth_wave(c);
    bench_gen_005_triangle_wave(c);
    bench_gen_006_chirp(c);
    bench_gen_007_impulse(c);
    bench_gen_008_compound_tone(c);
    bench_gen_009_am_signal(c);
    bench_gen_010_exponential_bursts(c);
    bench_gen_011_silence(c);
    bench_gen_012_white_noise(c);
    bench_gen_013_pink_noise(c);
    bench_gen_014_brown_noise(c);
    bench_gen_015_stereo_sine_wave(c);
    bench_gen_016_stereo_chirp(c);
    bench_gen_017_stereo_silence(c);
    bench_gen_018_multichannel_compound_tone(c);
}

// ===========================================================================
// Helpers
// ===========================================================================

#[inline]
fn bench_sample_rate() -> NonZeroU32 {
    // SAFETY: BENCH_SAMPLE_RATE_HZ is a non-zero constant.
    unsafe { NonZeroU32::new_unchecked(BENCH_SAMPLE_RATE_HZ) }
}

/// Convert a per-channel sample count into a `Duration` at the bench sample
/// rate. Generators size their output via `duration * sample_rate`, so this
/// is how we drive the length axis.
#[inline]
fn duration_for(len: usize) -> Duration {
    Duration::from_secs_f64(len as f64 / f64::from(BENCH_SAMPLE_RATE_HZ))
}

// ===========================================================================
// Dispatch macro — generators allocate their own output, so we drive them
// with plain `b.iter(|| black_box(...))` and let the allocator be measured.
//
// The body block receives a type alias `T` for the per-arm dtype and must
// evaluate to an `AudioSamples<'static, T>` (the generator call).
// ===========================================================================

macro_rules! dispatch_gen_dtypes {
    ($group:expr, $id:expr, $dt:expr, || $body:expr) => {{
        let id = $id;
        match $dt {
            "i16" => $group.bench_function(id, |b| b.iter(|| { type T = i16; black_box::<audio_samples::AudioSamples<'static, T>>($body) })),
            "I24" => $group.bench_function(id, |b| b.iter(|| { type T = I24; black_box::<audio_samples::AudioSamples<'static, T>>($body) })),
            "i32" => $group.bench_function(id, |b| b.iter(|| { type T = i32; black_box::<audio_samples::AudioSamples<'static, T>>($body) })),
            "f32" => $group.bench_function(id, |b| b.iter(|| { type T = f32; black_box::<audio_samples::AudioSamples<'static, T>>($body) })),
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
// Gen-001 sine_wave — mono; phase accumulation per sample + dtype quantize.
// ===========================================================================

fn bench_gen_001_sine_wave<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("gen");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Gen-001_sine_wave", len, dt, ch);
            let dur = duration_for(len);
            let sr = bench_sample_rate();
            dispatch_gen_dtypes!(group, id, dt, || sine_wave::<T>(440.0, dur, sr, 1.0));
        }
    }
    group.finish();
}

// ===========================================================================
// Gen-002 cosine_wave — mono.
// ===========================================================================

fn bench_gen_002_cosine_wave<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("gen");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Gen-002_cosine_wave", len, dt, ch);
            let dur = duration_for(len);
            let sr = bench_sample_rate();
            dispatch_gen_dtypes!(group, id, dt, || cosine_wave::<T>(440.0, dur, sr, 1.0));
        }
    }
    group.finish();
}

// ===========================================================================
// Gen-003 square_wave — mono; naive (no anti-aliasing).
// ===========================================================================

fn bench_gen_003_square_wave<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("gen");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Gen-003_square_wave", len, dt, ch);
            let dur = duration_for(len);
            let sr = bench_sample_rate();
            dispatch_gen_dtypes!(group, id, dt, || square_wave::<T>(440.0, dur, sr, 1.0));
        }
    }
    group.finish();
}

// ===========================================================================
// Gen-004 sawtooth_wave — mono.
// ===========================================================================

fn bench_gen_004_sawtooth_wave<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("gen");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Gen-004_sawtooth_wave", len, dt, ch);
            let dur = duration_for(len);
            let sr = bench_sample_rate();
            dispatch_gen_dtypes!(group, id, dt, || sawtooth_wave::<T>(440.0, dur, sr, 1.0));
        }
    }
    group.finish();
}

// ===========================================================================
// Gen-005 triangle_wave — mono.
// ===========================================================================

fn bench_gen_005_triangle_wave<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("gen");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Gen-005_triangle_wave", len, dt, ch);
            let dur = duration_for(len);
            let sr = bench_sample_rate();
            dispatch_gen_dtypes!(group, id, dt, || triangle_wave::<T>(440.0, dur, sr, 1.0));
        }
    }
    group.finish();
}

// ===========================================================================
// Gen-006 chirp — mono; linear sweep 100 → 8 kHz.
// ===========================================================================

fn bench_gen_006_chirp<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("gen");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Gen-006_chirp", len, dt, ch);
            let dur = duration_for(len);
            let sr = bench_sample_rate();
            dispatch_gen_dtypes!(group, id, dt, || chirp::<T>(100.0, 8000.0, dur, sr, 1.0));
        }
    }
    group.finish();
}

// ===========================================================================
// Gen-007 impulse — mono; mostly zero-fill + one non-zero sample.
// ===========================================================================

fn bench_gen_007_impulse<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("gen");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Gen-007_impulse", len, dt, ch);
            let dur = duration_for(len);
            let sr = bench_sample_rate();
            // Position at midpoint of the buffer in seconds.
            let pos = dur.as_secs_f64() * 0.5;
            dispatch_gen_dtypes!(group, id, dt, || impulse::<T>(dur, sr, 1.0, pos));
        }
    }
    group.finish();
}

// ===========================================================================
// Gen-008 compound_tone — mono; cost ∝ len × n_components. Three-harmonic
// stack (fundamental + 2nd + 3rd) — extra axis recorded as `n_comp=3`.
// ===========================================================================

fn bench_gen_008_compound_tone<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("gen");
    let ch = 1;
    let raw_components = [
        ToneComponent::new(440.0, 1.0),
        ToneComponent::new(880.0, 0.5),
        ToneComponent::new(1320.0, 0.25),
    ];
    let n_comp = raw_components.len();
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            group.throughput(Throughput::Elements((len * ch) as u64));
            let label = ParamLabel::new()
                .with("len", len).with("dt", dt).with("ch", ch)
                .with("n_comp", n_comp)
                .build();
            let id = BenchmarkId::new("Gen-008_compound_tone", label);
            let dur = duration_for(len);
            let sr = bench_sample_rate();
            let comps = NonEmptySlice::from_slice(&raw_components).unwrap();
            dispatch_gen_dtypes!(group, id, dt, || compound_tone::<T>(comps, dur, sr));
        }
    }
    group.finish();
}

// ===========================================================================
// Gen-009 am_signal — mono; 440 Hz carrier, 2 Hz modulator, 50% depth.
// ===========================================================================

fn bench_gen_009_am_signal<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("gen");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Gen-009_am_signal", len, dt, ch);
            let dur = duration_for(len);
            let sr = bench_sample_rate();
            dispatch_gen_dtypes!(group, id, dt, || am_signal::<T>(440.0, 2.0, 0.5, dur, sr, 0.8));
        }
    }
    group.finish();
}

// ===========================================================================
// Gen-010 exponential_bursts — mono; requires `random-generation`. Uses
// `rand::random` internally (not RNG-seedable through the signature, so
// runs are reproducible to within the rand crate's thread-local RNG).
// ===========================================================================

fn bench_gen_010_exponential_bursts<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("gen");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Gen-010_exponential_bursts", len, dt, ch);
            let dur = duration_for(len);
            let sr = bench_sample_rate();
            dispatch_gen_dtypes!(group, id, dt, || exponential_bursts::<T>(2.0, 30.0, dur, sr, 0.8));
        }
    }
    group.finish();
}

// ===========================================================================
// Gen-011 silence — mono; pure allocation + zero-fill cost.
// ===========================================================================

fn bench_gen_011_silence<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("gen");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Gen-011_silence", len, dt, ch);
            let dur = duration_for(len);
            let sr = bench_sample_rate();
            dispatch_gen_dtypes!(group, id, dt, || silence::<T>(dur, sr));
        }
    }
    group.finish();
}

// ===========================================================================
// Gen-012 white_noise — mono; RNG-bound. Seeded with BENCH_RNG_SEED for
// reproducibility (`Some(seed)` triggers the seeded branch).
// ===========================================================================

fn bench_gen_012_white_noise<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("gen");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Gen-012_white_noise", len, dt, ch);
            let dur = duration_for(len);
            let sr = bench_sample_rate();
            let seed = Some(BENCH_RNG_SEED);
            dispatch_gen_dtypes!(group, id, dt, || white_noise::<T>(dur, sr, 0.5, seed));
        }
    }
    group.finish();
}

// ===========================================================================
// Gen-013 pink_noise — mono; filtered white noise via Paul Kellett IIR.
// Heavier per sample than white. Seeded.
// ===========================================================================

fn bench_gen_013_pink_noise<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("gen");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Gen-013_pink_noise", len, dt, ch);
            let dur = duration_for(len);
            let sr = bench_sample_rate();
            let seed = Some(BENCH_RNG_SEED);
            dispatch_gen_dtypes!(group, id, dt, || pink_noise::<T>(dur, sr, 0.5, seed));
        }
    }
    group.finish();
}

// ===========================================================================
// Gen-014 brown_noise — mono; random-walk integrator.
//
// NOTE: upstream `brown_noise` has a bug — the `Some(seed)` branch generates
// exactly ONE sample regardless of `num_samples` (the per-sample loop is
// missing in `src/utils/generation.rs::brown_noise`). To measure work that
// actually scales with `len`, we drive this bench with `seed = None`. The
// non-determinism is acceptable for benchmark timing (the RNG call cost is
// constant per sample); the call still uses `rand::random` and would benefit
// from being unified with the seeded path upstream. See the "Open questions"
// in the section report.
// ===========================================================================

fn bench_gen_014_brown_noise<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("gen");
    let ch = 1;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Gen-014_brown_noise", len, dt, ch);
            let dur = duration_for(len);
            let sr = bench_sample_rate();
            let seed: Option<u64> = None;
            dispatch_gen_dtypes!(group, id, dt, || brown_noise::<T>(dur, sr, 0.02, 0.5, seed));
        }
    }
    group.finish();
}

// ===========================================================================
// Gen-015 stereo_sine_wave — 2-channel (requires `channels`).
// ===========================================================================

fn bench_gen_015_stereo_sine_wave<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("gen");
    let ch = 2;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Gen-015_stereo_sine_wave", len, dt, ch);
            let dur = duration_for(len);
            let sr = bench_sample_rate();
            dispatch_gen_dtypes!(group, id, dt, || stereo_sine_wave::<T>(440.0, dur, sr, 1.0));
        }
    }
    group.finish();
}

// ===========================================================================
// Gen-016 stereo_chirp — 2-channel (requires `channels`).
// ===========================================================================

fn bench_gen_016_stereo_chirp<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("gen");
    let ch = 2;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Gen-016_stereo_chirp", len, dt, ch);
            let dur = duration_for(len);
            let sr = bench_sample_rate();
            dispatch_gen_dtypes!(group, id, dt, || stereo_chirp::<T>(100.0, 8000.0, dur, sr, 1.0));
        }
    }
    group.finish();
}

// ===========================================================================
// Gen-017 stereo_silence — 2-channel (requires `channels`).
// ===========================================================================

fn bench_gen_017_stereo_silence<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("gen");
    let ch = 2;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Gen-017_stereo_silence", len, dt, ch);
            let dur = duration_for(len);
            let sr = bench_sample_rate();
            dispatch_gen_dtypes!(group, id, dt, || stereo_silence::<T>(dur, sr));
        }
    }
    group.finish();
}

// ===========================================================================
// Gen-018 multichannel_compound_tone — sweep ch ∈ {2, 6, 8} (requires
// `channels`). Cost is mono compound_tone + duplicate_to_channels. Two
// components.
// ===========================================================================

fn bench_gen_018_multichannel_compound_tone<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("gen");
    let raw_components = [
        ToneComponent::new(440.0, 1.0),
        ToneComponent::new(880.0, 0.5),
    ];
    let n_comp = raw_components.len();
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in &[2usize, 6, 8] {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len).with("dt", dt).with("ch", ch)
                    .with("n_comp", n_comp)
                    .build();
                let id = BenchmarkId::new("Gen-018_multichannel_compound_tone", label);
                let dur = duration_for(len);
                let sr = bench_sample_rate();
                let comps = NonEmptySlice::from_slice(&raw_components).unwrap();
                dispatch_gen_dtypes!(group, id, dt, || multichannel_compound_tone::<T>(comps, dur, sr, ch));
            }
        }
    }
    group.finish();
}
