//! Type-conversion throughput benchmarks for the SIMD-sensitive paths.
//!
//! Covers the audio-aware sample-type conversions (`to_type`/`as_f32`) and the
//! channel interleave/deinterleave paths that the 2.0 SIMD work touched.
//!
//! NOTE: the SIMD code path is selected at compile time by `--features simd`.
//! To compare the scalar fallback against the SIMD path, run this bench BOTH
//! with and without that feature, e.g.
//!   cargo bench --bench conversions --features "statistics channels"
//!   cargo bench --bench conversions --features "statistics channels simd"
//!
//! Run with:
//!   cargo bench --bench conversions --features "statistics channels"

use std::num::NonZeroU32;
use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

use audio_samples::{AudioSamples, AudioTypeConversion};
use ndarray::{Array1, Array2};

const SR: u32 = 44_100;
const SECONDS: usize = 2;

fn signal(len: usize) -> Array1<f32> {
    // Deterministic two-tone + a little broadband-ish content, in [-1, 1].
    let mut v = Vec::with_capacity(len);
    for n in 0..len {
        let t = n as f32 / SR as f32;
        let s = 0.6 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
            + 0.3 * (2.0 * std::f32::consts::PI * 6_000.0 * t).sin()
            + 0.1 * ((n * 2654435761usize) as f32 / u32::MAX as f32 - 0.5);
        v.push(s.clamp(-1.0, 1.0));
    }
    Array1::from_vec(v)
}

fn make_f32(len: usize) -> AudioSamples<'static, f32> {
    AudioSamples::new_mono(signal(len), NonZeroU32::new(SR).unwrap()).unwrap()
}

fn bench_to_type(c: &mut Criterion) {
    let len = SR as usize * SECONDS;
    let mut g = c.benchmark_group("convert_to_type_f32_src");
    g.throughput(Throughput::Elements(len as u64));

    let base = make_f32(len);

    g.bench_function("f32_to_i16", |b| {
        b.iter_batched(
            || base.clone(),
            |audio| std::hint::black_box(audio.to_type::<i16>()),
            BatchSize::SmallInput,
        )
    });

    g.bench_function("f32_to_i32", |b| {
        b.iter_batched(
            || base.clone(),
            |audio| std::hint::black_box(audio.to_type::<i32>()),
            BatchSize::SmallInput,
        )
    });

    // as_f32() on an f32 signal: exercises the (identity-typed) conversion path.
    g.bench_function("f32_as_f32", |b| {
        b.iter_batched(
            || base.clone(),
            |audio| std::hint::black_box(audio.as_f32()),
            BatchSize::SmallInput,
        )
    });

    g.finish();
}

fn bench_round_trip(c: &mut Criterion) {
    let len = SR as usize * SECONDS;
    let mut g = c.benchmark_group("convert_round_trip");
    g.throughput(Throughput::Elements(len as u64));

    let base_f32 = make_f32(len);
    let base_i16 = base_f32.clone().to_type::<i16>();

    // f32 -> i16 -> f32
    g.bench_function("f32_i16_f32", |b| {
        b.iter_batched(
            || base_f32.clone(),
            |audio| {
                let as_i16 = audio.to_type::<i16>();
                std::hint::black_box(as_i16.to_type::<f32>())
            },
            BatchSize::SmallInput,
        )
    });

    // i16 -> f32 (the decode-to-float direction the SIMD work targeted)
    g.bench_function("i16_to_f32", |b| {
        b.iter_batched(
            || base_i16.clone(),
            |audio| std::hint::black_box(audio.to_type::<f32>()),
            BatchSize::SmallInput,
        )
    });

    g.finish();
}

#[cfg(feature = "channels")]
fn bench_channels(c: &mut Criterion) {
    use audio_samples::operations::AudioChannelOps;
    use non_empty_slice::NonEmptySlice;

    let per_channel = SR as usize * SECONDS;
    let mut g = c.benchmark_group("channels_interleave");
    g.throughput(Throughput::Elements((per_channel * 2) as u64));

    // Two independent mono channels for interleave.
    let ch0 = make_f32(per_channel);
    let ch1 = make_f32(per_channel);

    g.bench_function("interleave_stereo", |b| {
        let arr = [ch0.clone(), ch1.clone()];
        let channels = NonEmptySlice::from_slice(&arr).unwrap();
        b.iter(|| {
            std::hint::black_box(
                <AudioSamples<'_, f32> as AudioChannelOps>::interleave_channels(channels).unwrap(),
            )
        })
    });

    // A multi-channel buffer to deinterleave back into mono signals.
    let stereo_data = {
        let s = signal(per_channel);
        let mut m = Array2::<f32>::zeros((2, per_channel));
        m.row_mut(0).assign(&s);
        m.row_mut(1).assign(&s);
        m
    };
    let stereo =
        AudioSamples::new_multi_channel(stereo_data, NonZeroU32::new(SR).unwrap()).unwrap();

    g.bench_function("deinterleave_stereo", |b| {
        b.iter(|| std::hint::black_box(stereo.deinterleave_channels().unwrap()))
    });

    g.finish();
}

#[cfg(not(feature = "channels"))]
fn bench_channels(_c: &mut Criterion) {}

criterion_group! {
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_millis(500)).measurement_time(Duration::from_secs(3));
    targets = bench_to_type, bench_round_trip, bench_channels
}
criterion_main!(benches);
