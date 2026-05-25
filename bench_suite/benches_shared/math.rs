//! Shared body for the `utils::audio_math` bench targets
//! (`bench_math_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/math.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `Math`
//! (lines 626-659).
//!
//! Strategy notes:
//! - Most rows are scalar `f64 -> f64` (or similar) functions whose per-call
//!   cost is dominated by Criterion's per-iter overhead. We wrap each in a
//!   tight loop over a fixed-size `Vec<f64>` (`SCALAR_LOOP_LEN = 1024`) and
//!   tag throughput as `Elements(1024)`. The reported time is therefore the
//!   amortised per-call cost times the loop length.
//! - Math-015..018 are string-parsing / allocating functions. They use short
//!   representative inputs ("A4", "F#3", 440.0) and report `Elements(1)`.
//! - Math-003, Math-021, Math-022, Math-023 allocate a `Vec<f64>` of the
//!   requested size and are swept across that size axis instead.
//! - Math-015 / Math-017 are benched with two-character ("A4") and
//!   three-character ("F#3") inputs to surface the alternate parser branch.

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

use audio_samples::utils::audio_math::{
    amplitude_to_db, cents_to_ratio, db_to_amplitude, db_to_power, fft_frequencies,
    frames_to_time, frequency_to_note, hz_to_mel, hz_to_midi, linspace, mel_frequencies,
    mel_scale, mel_to_hz, midi_to_hz, midi_to_note, ms_to_samples, note_to_frequency,
    note_to_midi, power_to_db, ratio_to_cents, samples_to_time, seconds_to_samples,
    time_to_frames,
};
use audio_samples::sample_rate;
use audio_samples::utils::{
    samples_to_seconds as mod_samples_to_seconds,
    seconds_to_samples as mod_seconds_to_samples,
};

use bench_suite_common::{ParamLabel, SampleSizePolicy, sample_size_for};

// ===========================================================================
// Tunable knobs
// ===========================================================================

/// Inner-loop length for scalar functions. Long enough to amortise away
/// Criterion's per-iter overhead, short enough to stay in L1.
const SCALAR_LOOP_LEN: usize = 1024;

/// Sizes swept for Vec-producing functions (`mel_scale`, `mel_frequencies`,
/// `linspace`, `fft_frequencies`).
const VEC_OUTPUT_SIZES: &[usize] = &[40, 128, 512, 2_048];

/// `n_fft` values for `fft_frequencies` (output size = `n_fft / 2 + 1`).
const FFT_SIZES: &[usize] = &[256, 1_024, 4_096, 16_384];

// ===========================================================================
// Top-level entry point
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_math_001_hz_to_mel(c);
    bench_math_002_mel_to_hz(c);
    bench_math_003_mel_scale(c);
    bench_math_004_hz_to_midi(c);
    bench_math_005_midi_to_hz(c);
    bench_math_006_amplitude_to_db(c);
    bench_math_007_db_to_amplitude(c);
    bench_math_008_power_to_db(c);
    bench_math_009_db_to_power(c);
    bench_math_010_frames_to_time(c);
    bench_math_011_time_to_frames(c);
    bench_math_012_samples_to_time(c);
    bench_math_013_seconds_to_samples(c);
    bench_math_014_ms_to_samples(c);
    bench_math_015_note_to_midi(c);
    bench_math_016_midi_to_note(c);
    bench_math_017_note_to_frequency(c);
    bench_math_018_frequency_to_note(c);
    bench_math_019_cents_to_ratio(c);
    bench_math_020_ratio_to_cents(c);
    bench_math_021_fft_frequencies(c);
    bench_math_022_mel_frequencies(c);
    bench_math_023_linspace(c);
    bench_math_024_utils_seconds_to_samples(c);
    bench_math_025_utils_samples_to_seconds(c);
}

// ===========================================================================
// Fixtures + dispatch helpers
// ===========================================================================

/// Build the input vector for the scalar-loop benches. Values cover a
/// representative dynamic range (positive, finite, non-trivial) so each
/// function exercises its real cost rather than a fast path on `0.0`.
fn scalar_inputs() -> Vec<f64> {
    // Use a deterministic, branch-friendly mix: spread across a few decades.
    // Avoid 0.0 so the `> 0.0` branches in `amplitude_to_db` / `power_to_db`
    // take the log10 path rather than the `-80.0` floor.
    (0..SCALAR_LOOP_LEN)
        .map(|i| 1.0 + (i as f64) * 0.5)
        .collect()
}

/// Bench a unary `f64 -> f64` function over `SCALAR_LOOP_LEN` inputs.
fn bench_scalar_loop<M, F>(
    c: &mut Criterion<M>,
    catalog_id: &str,
    f: F,
)
where
    M: Measurement,
    F: Fn(f64) -> f64 + Copy,
{
    let mut group = c.benchmark_group("math");
    group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, 0));
    group.throughput(Throughput::Elements(SCALAR_LOOP_LEN as u64));
    let label = ParamLabel::new()
        .with("len", SCALAR_LOOP_LEN)
        .with("dt", "f64")
        .build();
    let id = BenchmarkId::new(catalog_id, label);
    let inputs = scalar_inputs();
    group.bench_with_input(id, &inputs, |b, inp| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for &x in inp {
                acc += f(black_box(x));
            }
            black_box(acc)
        });
    });
    group.finish();
}

/// Bench a unary `f64 -> usize` function over `SCALAR_LOOP_LEN` inputs.
fn bench_scalar_loop_usize<M, F>(
    c: &mut Criterion<M>,
    catalog_id: &str,
    f: F,
)
where
    M: Measurement,
    F: Fn(f64) -> usize + Copy,
{
    let mut group = c.benchmark_group("math");
    group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, 0));
    group.throughput(Throughput::Elements(SCALAR_LOOP_LEN as u64));
    let label = ParamLabel::new()
        .with("len", SCALAR_LOOP_LEN)
        .with("dt", "f64")
        .build();
    let id = BenchmarkId::new(catalog_id, label);
    let inputs = scalar_inputs();
    group.bench_with_input(id, &inputs, |b, inp| {
        b.iter(|| {
            let mut acc = 0_usize;
            for &x in inp {
                acc = acc.wrapping_add(f(black_box(x)));
            }
            black_box(acc)
        });
    });
    group.finish();
}

// ===========================================================================
// Math-001 hz_to_mel — scalar loop
// ===========================================================================

fn bench_math_001_hz_to_mel<M: Measurement>(c: &mut Criterion<M>) {
    bench_scalar_loop(c, "Math-001_hz_to_mel", hz_to_mel);
}

// ===========================================================================
// Math-002 mel_to_hz — scalar loop
// ===========================================================================

fn bench_math_002_mel_to_hz<M: Measurement>(c: &mut Criterion<M>) {
    bench_scalar_loop(c, "Math-002_mel_to_hz", mel_to_hz);
}

// ===========================================================================
// Math-003 mel_scale — allocates `Vec<f64>` of length `n_mels`.
// Swept across n_mels ∈ {40, 128, 512, 2048}.
// ===========================================================================

fn bench_math_003_mel_scale<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("math");
    let fmin = 0.0_f64;
    let fmax = 8_000.0_f64;
    for &n_mels in VEC_OUTPUT_SIZES {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, 0));
        group.throughput(Throughput::Elements(n_mels as u64));
        let label = ParamLabel::new()
            .with("dt", "f64")
            .with("n_mels", n_mels)
            .build();
        let id = BenchmarkId::new("Math-003_mel_scale", label);
        group.bench_with_input(id, &n_mels, |b, &n| {
            b.iter(|| black_box(mel_scale(black_box(n), black_box(fmin), black_box(fmax))));
        });
    }
    group.finish();
}

// ===========================================================================
// Math-004 hz_to_midi — scalar loop
// ===========================================================================

fn bench_math_004_hz_to_midi<M: Measurement>(c: &mut Criterion<M>) {
    bench_scalar_loop(c, "Math-004_hz_to_midi", hz_to_midi);
}

// ===========================================================================
// Math-005 midi_to_hz — scalar loop
// ===========================================================================

fn bench_math_005_midi_to_hz<M: Measurement>(c: &mut Criterion<M>) {
    bench_scalar_loop(c, "Math-005_midi_to_hz", midi_to_hz);
}

// ===========================================================================
// Math-006 amplitude_to_db — scalar loop (branchy: `>0` path vs floor)
// ===========================================================================

fn bench_math_006_amplitude_to_db<M: Measurement>(c: &mut Criterion<M>) {
    bench_scalar_loop(c, "Math-006_amplitude_to_db", amplitude_to_db);
}

// ===========================================================================
// Math-007 db_to_amplitude — scalar loop
// ===========================================================================

fn bench_math_007_db_to_amplitude<M: Measurement>(c: &mut Criterion<M>) {
    bench_scalar_loop(c, "Math-007_db_to_amplitude", db_to_amplitude);
}

// ===========================================================================
// Math-008 power_to_db — scalar loop
// ===========================================================================

fn bench_math_008_power_to_db<M: Measurement>(c: &mut Criterion<M>) {
    bench_scalar_loop(c, "Math-008_power_to_db", power_to_db);
}

// ===========================================================================
// Math-009 db_to_power — scalar loop
// ===========================================================================

fn bench_math_009_db_to_power<M: Measurement>(c: &mut Criterion<M>) {
    bench_scalar_loop(c, "Math-009_db_to_power", db_to_power);
}

// ===========================================================================
// Math-010 frames_to_time — scalar loop (3-arg, frame idx swept)
// ===========================================================================

fn bench_math_010_frames_to_time<M: Measurement>(c: &mut Criterion<M>) {
    let sr = 44_100.0_f64;
    let hop = 512_usize;
    bench_scalar_loop(c, "Math-010_frames_to_time", move |x| {
        frames_to_time(x as usize, sr, hop)
    });
}

// ===========================================================================
// Math-011 time_to_frames — scalar loop (f64 -> usize)
// ===========================================================================

fn bench_math_011_time_to_frames<M: Measurement>(c: &mut Criterion<M>) {
    let sr = 44_100.0_f64;
    let hop = 512_usize;
    bench_scalar_loop_usize(c, "Math-011_time_to_frames", move |t| {
        time_to_frames(t, sr, hop)
    });
}

// ===========================================================================
// Math-012 samples_to_time — scalar loop (sample-idx swept)
// ===========================================================================

fn bench_math_012_samples_to_time<M: Measurement>(c: &mut Criterion<M>) {
    let sr = 44_100.0_f64;
    bench_scalar_loop(c, "Math-012_samples_to_time", move |x| {
        samples_to_time(x as usize, sr)
    });
}

// ===========================================================================
// Math-013 seconds_to_samples (audio_math version)
// ===========================================================================

fn bench_math_013_seconds_to_samples<M: Measurement>(c: &mut Criterion<M>) {
    let sr = 44_100.0_f64;
    bench_scalar_loop_usize(c, "Math-013_seconds_to_samples", move |t| {
        seconds_to_samples(t, sr)
    });
}

// ===========================================================================
// Math-014 ms_to_samples
// ===========================================================================

fn bench_math_014_ms_to_samples<M: Measurement>(c: &mut Criterion<M>) {
    let sr = 44_100.0_f64;
    bench_scalar_loop_usize(c, "Math-014_ms_to_samples", move |ms| {
        ms_to_samples(ms, sr)
    });
}

// ===========================================================================
// Math-015 note_to_midi — string parse, short input. Bench both 2-char
// ("A4") and 3-char with accidental ("F#3") to cover the alternate branch.
// Allocator-bound — single-call, Elements(1).
// ===========================================================================

fn bench_math_015_note_to_midi<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("math");
    group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, 0));
    for &input in &["A4", "F#3"] {
        group.throughput(Throughput::Elements(1));
        let label = ParamLabel::new()
            .with("dt", "str")
            .with("input", input)
            .build();
        let id = BenchmarkId::new("Math-015_note_to_midi", label);
        group.bench_with_input(id, &input, |b, &name| {
            b.iter(|| black_box(note_to_midi(black_box(name)).ok()));
        });
    }
    group.finish();
}

// ===========================================================================
// Math-016 midi_to_note — allocates a `String`. Single representative input.
// ===========================================================================

fn bench_math_016_midi_to_note<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("math");
    group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, 0));
    for &midi in &[60_u8, 69, 100] {
        group.throughput(Throughput::Elements(1));
        let label = ParamLabel::new()
            .with("dt", "u8")
            .with("midi", midi)
            .build();
        let id = BenchmarkId::new("Math-016_midi_to_note", label);
        group.bench_with_input(id, &midi, |b, &m| {
            b.iter(|| black_box(midi_to_note(black_box(m)).ok()));
        });
    }
    group.finish();
}

// ===========================================================================
// Math-017 note_to_frequency — note_to_midi + midi_to_hz composition.
// Short string input; both branches of the parser exercised.
// ===========================================================================

fn bench_math_017_note_to_frequency<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("math");
    group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, 0));
    for &input in &["A4", "F#3"] {
        group.throughput(Throughput::Elements(1));
        let label = ParamLabel::new()
            .with("dt", "str")
            .with("input", input)
            .build();
        let id = BenchmarkId::new("Math-017_note_to_frequency", label);
        group.bench_with_input(id, &input, |b, &name| {
            b.iter(|| black_box(note_to_frequency(black_box(name)).ok()));
        });
    }
    group.finish();
}

// ===========================================================================
// Math-018 frequency_to_note — composes hz_to_midi + midi_to_note;
// allocates a `String`. Single representative input (A4 = 440 Hz).
// ===========================================================================

fn bench_math_018_frequency_to_note<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("math");
    group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, 0));
    for &freq in &[261.63_f64, 440.0, 1_760.0] {
        group.throughput(Throughput::Elements(1));
        let label = ParamLabel::new()
            .with("dt", "f64")
            .with("freq", format!("{freq:.2}"))
            .build();
        let id = BenchmarkId::new("Math-018_frequency_to_note", label);
        group.bench_with_input(id, &freq, |b, &f| {
            b.iter(|| black_box(frequency_to_note(black_box(f)).ok()));
        });
    }
    group.finish();
}

// ===========================================================================
// Math-019 cents_to_ratio — scalar loop
// ===========================================================================

fn bench_math_019_cents_to_ratio<M: Measurement>(c: &mut Criterion<M>) {
    bench_scalar_loop(c, "Math-019_cents_to_ratio", cents_to_ratio);
}

// ===========================================================================
// Math-020 ratio_to_cents — scalar loop
// ===========================================================================

fn bench_math_020_ratio_to_cents<M: Measurement>(c: &mut Criterion<M>) {
    bench_scalar_loop(c, "Math-020_ratio_to_cents", ratio_to_cents);
}

// ===========================================================================
// Math-021 fft_frequencies — allocates `Vec<f64>` of length `n_fft/2 + 1`.
// Swept across n_fft ∈ {256, 1024, 4096, 16384}.
// ===========================================================================

fn bench_math_021_fft_frequencies<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("math");
    let sr = 44_100.0_f64;
    for &n_fft in FFT_SIZES {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, 0));
        let n_bins = n_fft / 2 + 1;
        group.throughput(Throughput::Elements(n_bins as u64));
        let label = ParamLabel::new()
            .with("dt", "f64")
            .with("n_fft", n_fft)
            .build();
        let id = BenchmarkId::new("Math-021_fft_frequencies", label);
        group.bench_with_input(id, &n_fft, |b, &n| {
            b.iter(|| black_box(fft_frequencies(black_box(n), black_box(sr))));
        });
    }
    group.finish();
}

// ===========================================================================
// Math-022 mel_frequencies — allocates `Vec<f64>` of length `n_mels`.
// ===========================================================================

fn bench_math_022_mel_frequencies<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("math");
    let fmin = 0.0_f64;
    let fmax = 8_000.0_f64;
    for &n_mels in VEC_OUTPUT_SIZES {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, 0));
        group.throughput(Throughput::Elements(n_mels as u64));
        let label = ParamLabel::new()
            .with("dt", "f64")
            .with("n_mels", n_mels)
            .build();
        let id = BenchmarkId::new("Math-022_mel_frequencies", label);
        group.bench_with_input(id, &n_mels, |b, &n| {
            b.iter(|| black_box(mel_frequencies(black_box(n), black_box(fmin), black_box(fmax))));
        });
    }
    group.finish();
}

// ===========================================================================
// Math-023 linspace — allocates `Vec<f64>` of length `num`.
// ===========================================================================

fn bench_math_023_linspace<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("math");
    let start = 0.0_f64;
    let end = 1.0_f64;
    for &num in VEC_OUTPUT_SIZES {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, 0));
        group.throughput(Throughput::Elements(num as u64));
        let label = ParamLabel::new()
            .with("dt", "f64")
            .with("num", num)
            .build();
        let id = BenchmarkId::new("Math-023_linspace", label);
        group.bench_with_input(id, &num, |b, &n| {
            b.iter(|| black_box(linspace(black_box(start), black_box(end), black_box(n))));
        });
    }
    group.finish();
}

// ===========================================================================
// Math-024 utils::mod::seconds_to_samples — generic `impl Into<u32>` API.
// Bench with `NonZeroU32` (the canonical bench-time sample-rate type) wrapped
// via the `sample_rate!` macro.
// ===========================================================================

fn bench_math_024_utils_seconds_to_samples<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("math");
    group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, 0));
    group.throughput(Throughput::Elements(SCALAR_LOOP_LEN as u64));
    let label = ParamLabel::new()
        .with("len", SCALAR_LOOP_LEN)
        .with("dt", "f64")
        .build();
    let id = BenchmarkId::new("Math-024_utils_seconds_to_samples", label);
    let inputs = scalar_inputs();
    let sr = sample_rate!(44_100);
    group.bench_with_input(id, &inputs, |b, inp| {
        b.iter(|| {
            let mut acc = 0_usize;
            for &t in inp {
                acc = acc.wrapping_add(mod_seconds_to_samples(black_box(t), sr));
            }
            black_box(acc)
        });
    });
    group.finish();
}

// ===========================================================================
// Math-025 utils::mod::samples_to_seconds — sibling of Math-024, inverse.
// ===========================================================================

fn bench_math_025_utils_samples_to_seconds<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("math");
    group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, 0));
    group.throughput(Throughput::Elements(SCALAR_LOOP_LEN as u64));
    let label = ParamLabel::new()
        .with("len", SCALAR_LOOP_LEN)
        .with("dt", "usize")
        .build();
    let id = BenchmarkId::new("Math-025_utils_samples_to_seconds", label);
    // usize inputs covering a representative span.
    let inputs: Vec<usize> = (0..SCALAR_LOOP_LEN).map(|i| i * 31 + 1).collect();
    let sr = sample_rate!(44_100);
    group.bench_with_input(id, &inputs, |b, inp| {
        b.iter(|| {
            let mut acc = 0.0_f64;
            for &n in inp {
                acc += mod_samples_to_seconds(black_box(n), sr);
            }
            black_box(acc)
        });
    });
    group.finish();
}
