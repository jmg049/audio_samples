//! Dynamic-range processing throughput benchmarks.
//!
//! Exercises the compressor ring-buffer / envelope path and friends. The
//! `_in_place` variants are used so the measurement is the actual sample
//! processing, not a clone of the buffer (each iteration restores a fresh copy
//! via `iter_batched`, but the timed body is the in-place op).
//!
//! Run with:
//!   cargo bench --bench dynamics --features "dynamic-range statistics"

use std::num::NonZeroU32;
use std::time::Duration;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

use audio_samples::{AudioDynamicRange, AudioSamples, CompressorConfig, GateConfig, LimiterConfig};
use ndarray::Array1;

const SR: u32 = 44_100;
const SECONDS: usize = 3;

fn signal(len: usize) -> Array1<f32> {
    // Two-tone with bursty amplitude so the detector/envelope actually moves.
    let mut v = Vec::with_capacity(len);
    for n in 0..len {
        let t = n as f32 / SR as f32;
        let env = 0.3 + 0.7 * (2.0 * std::f32::consts::PI * 2.0 * t).sin().abs();
        let s = env
            * (0.7 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 4_000.0 * t).sin());
        v.push(s.clamp(-1.0, 1.0));
    }
    Array1::from_vec(v)
}

fn make_audio(len: usize) -> AudioSamples<'static, f32> {
    AudioSamples::new_mono(signal(len), NonZeroU32::new(SR).unwrap()).unwrap()
}

fn bench_dynamics(c: &mut Criterion) {
    let len = SR as usize * SECONDS;
    let base = make_audio(len);

    let mut g = c.benchmark_group("dynamic_range_in_place_f32");
    g.throughput(Throughput::Elements(len as u64));

    let comp = CompressorConfig::vocal();
    g.bench_function("compressor_vocal", |b| {
        b.iter_batched(
            || base.clone(),
            |mut audio| {
                audio.apply_compressor_in_place(&comp).unwrap();
                std::hint::black_box(audio)
            },
            BatchSize::SmallInput,
        )
    });

    let lim = LimiterConfig::transparent();
    g.bench_function("limiter_transparent", |b| {
        b.iter_batched(
            || base.clone(),
            |mut audio| {
                audio.apply_limiter_in_place(&lim).unwrap();
                std::hint::black_box(audio)
            },
            BatchSize::SmallInput,
        )
    });

    let gate = GateConfig::noise_gate();
    g.bench_function("gate_noise", |b| {
        b.iter_batched(
            || base.clone(),
            |mut audio| {
                audio.apply_gate_in_place(&gate).unwrap();
                std::hint::black_box(audio)
            },
            BatchSize::SmallInput,
        )
    });

    g.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_millis(500)).measurement_time(Duration::from_secs(3));
    targets = bench_dynamics
}
criterion_main!(benches);
