//! Shared body for the `AudioChannelOps` bench targets
//! (`bench_chan_walltime`, `_instructions`, `_cycles`, `_cache_misses`,
//! `_branch_misses`).
//!
//! Each per-measurement bench-target file under `bench_suite/benches/`
//! includes this module via `#[path = "../benches_shared/chan.rs"]` and
//! invokes [`bench_all`] with its `Criterion<M>` instance.
//!
//! Conventions: see `bench_suite/METHODOLOGY.md`.
//! Function inventory: see `bench_suite/CATALOG.md` section `Chan`
//! (rows Chan-001 .. Chan-012, lines 211-230).
//!
//! Notes on signatures: although the catalog rows summarise `to_mono`,
//! `to_stereo`, and `duplicate_to_channels` as `(self, ...)`, the actual
//! trait methods take `&self` (see `src/operations/traits.rs:4075/4114/4146`).
//! That makes `iter_batched_ref` (no clone needed) the right shape for the
//! whole section.

use criterion::measurement::Measurement;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use std::num::NonZeroU32;
use std::time::Duration;

use audio_samples::operations::channels as channel_ops;
use audio_samples::operations::types::{MonoConversionMethod, StereoConversionMethod};
use audio_samples::utils::generation::{ToneComponent, multichannel_compound_tone, sine_wave};
use audio_samples::{AudioChannelOps, AudioSamples, I24, StandardSample};

use bench_suite_common::{
    BENCH_SAMPLE_RATE_HZ, DTYPES_DEFAULT, LENGTH_SWEEP_FULL, LENGTH_SWEEP_NO_XXXL, ParamLabel,
    SampleSizePolicy, fixture_a440, sample_size_for,
};
use non_empty_slice::NonEmptySlice;

// ===========================================================================
// Section-local channel layouts. The shared `CHANNELS_DEFAULT` covers only
// {1, 2}; channel ops have meaningful 5.1 cost so we include 6. Some ops
// don't accept ch=1 (pan/balance) or don't differentiate it from a clone
// (to_mono, deinterleave_channels), so per-bench subsets are declared
// inline at each call site and noted in the comment block above the fn.
// ===========================================================================

const CHANNELS_INCLUDING_SURROUND: &[usize] = &[1, 2, 6];
const CHANNELS_MULTI: &[usize] = &[2, 6];

#[inline]
fn bench_sr() -> NonZeroU32 {
    NonZeroU32::new(BENCH_SAMPLE_RATE_HZ).expect("BENCH_SAMPLE_RATE_HZ is non-zero")
}

// ===========================================================================
// Multichannel fixture helper.
//
// `fixture_a440` only supports channel counts ∈ {1, 2}. For 5.1-style work
// we use `multichannel_compound_tone`, which builds a mono compound tone
// and duplicates it across N channels (identical content per channel —
// good enough for cost benching, since the channel-loop iteration count is
// what we care about, not content).
// ===========================================================================

fn fixture_multichannel<T>(n_samples: usize, channels: usize) -> AudioSamples<'static, T>
where
    T: StandardSample + 'static,
{
    if channels <= 2 {
        return fixture_a440::<T>(n_samples, channels);
    }
    let sr_f = f64::from(BENCH_SAMPLE_RATE_HZ);
    let duration = Duration::from_secs_f64(n_samples as f64 / sr_f);
    let components = [ToneComponent::new(440.0, 1.0)];
    let nes = NonEmptySlice::new(&components).expect("non-empty tone component slice");
    multichannel_compound_tone::<T>(nes, duration, bench_sr(), channels)
}

// ===========================================================================
// Top-level entry point — wrappers call into this with their typed Criterion.
// ===========================================================================

pub fn bench_all<M: Measurement>(c: &mut Criterion<M>) {
    bench_chan_001_to_mono(c);
    bench_chan_002_to_stereo(c);
    bench_chan_003_duplicate_to_channels(c);
    bench_chan_004_extract_channel(c);
    bench_chan_005_borrow_channel(c);
    bench_chan_006_swap_channels(c);
    bench_chan_007_pan(c);
    bench_chan_008_balance(c);
    bench_chan_009_apply_to_channel(c);
    bench_chan_010_interleave_channels(c);
    bench_chan_011_deinterleave_channels(c);
    bench_chan_012_deinterleave_free(c);
}

// ===========================================================================
// Dispatch macros — DTYPES_DEFAULT-typed expansion over channel-ops fixtures.
//
// `dispatch_chan_ref!`: bench `audio.op()` (immutable) using the multichannel
// fixture. The routine receives `&mut AudioSamples` (from iter_batched_ref);
// for `&self` ops it autoderefs.
//
// `dispatch_chan_mut!`: identical batching, but the macro body explicitly
// runs a `&mut self` op (pan / balance / swap_channels / apply_to_channel).
// Same expansion shape as `dispatch_chan_ref!`; kept as a separate name to
// keep call sites self-documenting.
// ===========================================================================

macro_rules! dispatch_chan_ref {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, |$audio:ident| $body:expr) => {{
        let id = $id;
        let n = $n;
        let ch = $ch;
        match $dt {
            "i16" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_multichannel::<i16>(n, ch),
                    |$audio| {
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            "I24" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_multichannel::<I24>(n, ch),
                    |$audio| {
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            "i32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_multichannel::<i32>(n, ch),
                    |$audio| {
                        black_box($body);
                    },
                    BatchSize::LargeInput,
                );
            }),
            "f32" => $group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || fixture_multichannel::<f32>(n, ch),
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

// Reuse the same body for `&mut self` ops — `iter_batched_ref` already
// gives the routine `&mut Output`, so the dispatch is identical.
macro_rules! dispatch_chan_mut {
    ($group:expr, $id:expr, $dt:expr, $n:expr, $ch:expr, |$audio:ident| $body:expr) => {{
        dispatch_chan_ref!($group, $id, $dt, $n, $ch, |$audio| $body)
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
// Chan-001 to_mono — NoFast tier. Allocates a new mono buffer; the
// stereo fast path has its own f32 cast loop. Mono input is a clone, so
// we restrict ch to {2, 6} where the downmix actually runs.
//
// Method axis: Average (default, most expensive), Left (channel 0 copy),
// Center (5.1-aware path). `Weighted` is skipped because it requires a
// length-matched Vec<f64> for every channel count; the cost is essentially
// the same as Average's inner loop.
// ===========================================================================

fn bench_chan_001_to_mono<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("chan");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_MULTI {
            for method_name in &["Average", "Left", "Center"] {
                for &dt in DTYPES_DEFAULT {
                    group.throughput(Throughput::Elements((len * ch) as u64));
                    let label = ParamLabel::new()
                        .with("len", len)
                        .with("dt", dt)
                        .with("ch", ch)
                        .with("method", method_name)
                        .build();
                    let id = BenchmarkId::new("Chan-001_to_mono", label);
                    let method = match *method_name {
                        "Average" => MonoConversionMethod::Average,
                        "Left" => MonoConversionMethod::Left,
                        "Center" => MonoConversionMethod::Center,
                        _ => unreachable!(),
                    };
                    dispatch_chan_ref!(group, id, dt, len, ch, |a| a
                        .to_mono(method.clone())
                        .ok());
                }
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Chan-002 to_stereo — NoFast tier. Source must be mono (ch=1) for the
// upmix path to do work; multi-channel input returns a clone. Method axis:
// Duplicate (cheap memcpy), Pan(0.0) (per-sample gain on both channels).
//
// `Left`/`Right` variants on a mono source call `extract_channel(1)` which
// errors out — we skip them here.
// ===========================================================================

fn bench_chan_002_to_stereo<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("chan");
    let ch = 1usize;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for method_name in &["Duplicate", "Pan0"] {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * 2) as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", ch)
                    .with("method", method_name)
                    .build();
                let id = BenchmarkId::new("Chan-002_to_stereo", label);
                let method = match *method_name {
                    "Duplicate" => StereoConversionMethod::Duplicate,
                    "Pan0" => StereoConversionMethod::Pan(0.0),
                    _ => unreachable!(),
                };
                dispatch_chan_ref!(group, id, dt, len, ch, |a| a.to_stereo(method).ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Chan-003 duplicate_to_channels — NoFast tier. Source is mono; target
// n_channels ∈ {2, 6}. Cost is one `clone` of the mono buffer + (n-1) views
// stacked into an Array2.
// ===========================================================================

fn bench_chan_003_duplicate_to_channels<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("chan");
    let src_ch = 1usize;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &tgt_ch in &[2usize, 6] {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * tgt_ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", src_ch)
                    .with("tgt_ch", tgt_ch)
                    .build();
                let id = BenchmarkId::new("Chan-003_duplicate_to_channels", label);
                dispatch_chan_ref!(group, id, dt, len, src_ch, |a| a
                    .duplicate_to_channels(tgt_ch)
                    .ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Chan-004 extract_channel — NoFast tier. Allocates an owned mono buffer.
// Restrict ch to {2, 6} (mono->idx 0 is a trivial clone).
// ===========================================================================

fn bench_chan_004_extract_channel<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("chan");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_MULTI {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Chan-004_extract_channel", len, dt, ch);
                dispatch_chan_ref!(group, id, dt, len, ch, |a| a.extract_channel(0).ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Chan-005 borrow_channel — FastSmall tier. Returns a zero-copy view; the
// per-call cost is index validation + view construction, length-independent.
// Bench the full sweep anyway to confirm the constancy.
// ===========================================================================

fn bench_chan_005_borrow_channel<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("chan");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_MULTI {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Chan-005_borrow_channel", len, dt, ch);
                dispatch_chan_ref!(group, id, dt, len, ch, |a| a.borrow_channel(0).ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Chan-006 swap_channels — FastSmall tier. In-place row swap on the
// ndarray; cost = 2 × `len` element copies through a temp.
// ===========================================================================

fn bench_chan_006_swap_channels<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("chan");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::FastSmall, i));
        for &ch in CHANNELS_MULTI {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Chan-006_swap_channels", len, dt, ch);
                dispatch_chan_mut!(group, id, dt, len, ch, |a| a.swap_channels(0, 1).ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Chan-007 pan — NoFast tier. Stereo only; per-sample f64 cast + gain.
// ===========================================================================

fn bench_chan_007_pan<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("chan");
    let ch = 2usize;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Chan-007_pan", len, dt, ch);
            dispatch_chan_mut!(group, id, dt, len, ch, |a| a.pan(0.3).ok());
        }
    }
    group.finish();
}

// ===========================================================================
// Chan-008 balance — NoFast tier. Same formula as pan; stereo only.
// ===========================================================================

fn bench_chan_008_balance<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("chan");
    let ch = 2usize;
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &dt in DTYPES_DEFAULT {
            let id = label_and_throughput(&mut group, "Chan-008_balance", len, dt, ch);
            dispatch_chan_mut!(group, id, dt, len, ch, |a| a.balance(-0.3).ok());
        }
    }
    group.finish();
}

// ===========================================================================
// Chan-009 apply_to_channel — NoFast tier. Closure call overhead per sample
// + the body. We use a trivial `|s| s` so the bench isolates the channel
// dispatch + ndarray iteration, not the user closure cost.
// ===========================================================================

fn bench_chan_009_apply_to_channel<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("chan");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_INCLUDING_SURROUND {
            for &dt in DTYPES_DEFAULT {
                let id = label_and_throughput(&mut group, "Chan-009_apply_to_channel", len, dt, ch);
                dispatch_chan_mut!(group, id, dt, len, ch, |a| a.apply_to_channel(0, |s| s).ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Chan-010 interleave_channels — NoFast tier. Static method: takes a
// NonEmptySlice of mono AudioSamples and packs them into a single
// multichannel AudioSamples. We build N mono `sine_wave` inputs in setup,
// hand them to the trait via NonEmptySlice. Lengths capped at NO_XXXL —
// the function allocates a fresh Array2 per call and 6 × 4_194_304 × 4 B
// per iter starts to swamp the bench harness.
// ===========================================================================

fn bench_chan_010_interleave_channels<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("chan");
    for (i, &len) in LENGTH_SWEEP_NO_XXXL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_INCLUDING_SURROUND {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", ch)
                    .build();
                let id = BenchmarkId::new("Chan-010_interleave_channels", label);
                interleave_dispatch(&mut group, id, dt, len, ch);
            }
        }
    }
    group.finish();
}

fn interleave_dispatch<M: Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
    id: BenchmarkId,
    dt: &str,
    n: usize,
    ch: usize,
) {
    macro_rules! one {
        ($t:ty) => {{
            group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || {
                        // Build `ch` mono sine fixtures, all length `n`.
                        let duration = Duration::from_secs_f64(n as f64 / f64::from(BENCH_SAMPLE_RATE_HZ));
                        let mut v: Vec<AudioSamples<'static, $t>> = Vec::with_capacity(ch);
                        for k in 0..ch {
                            // Slightly different frequency per channel so the
                            // mono inputs aren't bit-identical (matches the
                            // stereo fixture convention from common's fixture).
                            let freq = 440.0 + k as f64 * 55.0;
                            v.push(sine_wave::<$t>(freq, duration, bench_sr(), 1.0));
                        }
                        v
                    },
                    |inputs| {
                        let nes = NonEmptySlice::new(inputs.as_slice())
                            .expect("at least one channel");
                        black_box(
                            <AudioSamples<'_, $t> as AudioChannelOps>::interleave_channels(nes)
                                .ok(),
                        );
                    },
                    BatchSize::LargeInput,
                );
            });
        }};
    }
    match dt {
        "i16" => one!(i16),
        "I24" => one!(I24),
        "i32" => one!(i32),
        "f32" => one!(f32),
        _ => unreachable!("DTYPES_DEFAULT changed without updating interleave dispatch"),
    }
}

// ===========================================================================
// Chan-011 deinterleave_channels — NoFast tier. Allocates one
// `AudioSamples<'static, T>` per channel.
// ===========================================================================

fn bench_chan_011_deinterleave_channels<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("chan");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_INCLUDING_SURROUND {
            for &dt in DTYPES_DEFAULT {
                let id =
                    label_and_throughput(&mut group, "Chan-011_deinterleave_channels", len, dt, ch);
                dispatch_chan_ref!(group, id, dt, len, ch, |a| a.deinterleave_channels().ok());
            }
        }
    }
    group.finish();
}

// ===========================================================================
// Chan-012 operations::channels::deinterleave — NoFast tier.
//
// The catalog hand-summary says `deinterleave(&[T], usize) -> Vec<Vec<T>>`
// but the actual signature is `deinterleave(&mut [T], usize) -> Result<()>`
// (`src/operations/channels.rs:1336`). It transposes an interleaved buffer
// into planar layout in place via a single temp allocation.
//
// Fixture: a flat Vec<T> built by interleaving `ch` channels of a 440 Hz
// sine. We use the stereo `fixture_a440` data for ch=2 and the
// multichannel helper for ch=6, then flatten to interleaved order in the
// setup phase. The `interleave_channels` step runs once per iter and is
// outside `black_box`, so the bench measures only the in-place transpose.
// ===========================================================================

fn bench_chan_012_deinterleave_free<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("chan");
    for (i, &len) in LENGTH_SWEEP_FULL.iter().enumerate() {
        group.sample_size(sample_size_for(SampleSizePolicy::NoFast, i));
        for &ch in CHANNELS_INCLUDING_SURROUND {
            for &dt in DTYPES_DEFAULT {
                group.throughput(Throughput::Elements((len * ch) as u64));
                let label = ParamLabel::new()
                    .with("len", len)
                    .with("dt", dt)
                    .with("ch", ch)
                    .build();
                let id = BenchmarkId::new("Chan-012_deinterleave_free", label);
                deinterleave_free_dispatch(&mut group, id, dt, len, ch);
            }
        }
    }
    group.finish();
}

fn deinterleave_free_dispatch<M: Measurement>(
    group: &mut criterion::BenchmarkGroup<'_, M>,
    id: BenchmarkId,
    dt: &str,
    n: usize,
    ch: usize,
) {
    macro_rules! one {
        ($t:ty) => {{
            group.bench_with_input(id, &(n, ch), |b, &(n, ch)| {
                b.iter_batched_ref(
                    || build_interleaved_buf::<$t>(n, ch),
                    |buf: &mut Vec<$t>| {
                        black_box(channel_ops::deinterleave::<$t>(buf.as_mut_slice(), ch).ok());
                    },
                    BatchSize::LargeInput,
                );
            });
        }};
    }
    match dt {
        "i16" => one!(i16),
        "I24" => one!(I24),
        "i32" => one!(i32),
        "f32" => one!(f32),
        _ => unreachable!("DTYPES_DEFAULT changed without updating deinterleave dispatch"),
    }
}

// Build a flat interleaved buffer `[ch0_0, ch1_0, ..., chN_0, ch0_1, ...]`
// of total length `n * ch`. For ch=1 we just use the mono fixture; for
// ch∈{2,6} we build per-channel sines with distinct frequencies (so the
// content isn't bit-identical across channels, matching the spirit of the
// stereo fixture in common's fixture_a440).
fn build_interleaved_buf<T>(n: usize, ch: usize) -> Vec<T>
where
    T: StandardSample + 'static,
{
    let duration = Duration::from_secs_f64(n as f64 / f64::from(BENCH_SAMPLE_RATE_HZ));
    let mut channels: Vec<AudioSamples<'static, T>> = Vec::with_capacity(ch);
    for k in 0..ch {
        let freq = 440.0 + k as f64 * 55.0;
        channels.push(sine_wave::<T>(freq, duration, bench_sr(), 1.0));
    }
    // Manual interleave: out[frame * ch + c] = channels[c].sample(frame).
    let mut out = vec![T::default(); n * ch];
    for c in 0..ch {
        let mono = channels[c]
            .as_mono()
            .expect("sine_wave returns mono");
        for frame in 0..n {
            out[frame * ch + c] = mono[frame];
        }
    }
    out
}
