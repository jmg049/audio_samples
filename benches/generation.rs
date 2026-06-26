//! Signal-generator throughput benchmarks at a couple of durations.
//!
//! Covers the deterministic oscillators/sweeps (core, no extra features) and
//! the noise generators (require `random-generation`). Each generator returns
//! a freshly allocated `AudioSamples`, so each iteration allocates + fills the
//! whole buffer.
//!
//! Run with:
//!   cargo bench --bench generation --features "random-generation"
//! (the noise group is compiled out without `random-generation`; the rest are
//! core and always run.)

use std::num::NonZeroU32;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use audio_samples::{
    AudioSamples, ToneComponent, chirp, compound_tone, exponential_chirp, fm_signal,
    sawtooth_wave_bandlimited, sine_wave, square_wave_bandlimited, triangle_wave_bandlimited,
};
use non_empty_slice::NonEmptySlice;

const SR: u32 = 44_100;

fn sr() -> NonZeroU32 {
    NonZeroU32::new(SR).unwrap()
}

/// Durations exercised: 1s and 10s.
const DURATIONS: [(&str, u64); 2] = [("1s", 1), ("10s", 10)];

fn bench_simple(c: &mut Criterion) {
    let mut g = c.benchmark_group("generation_simple");
    for (label, secs) in DURATIONS {
        let dur = Duration::from_secs(secs);
        g.throughput(Throughput::Elements((SR as u64) * secs));

        g.bench_function(format!("sine_wave_{label}"), |b| {
            b.iter(|| {
                let a: AudioSamples<'static, f32> = sine_wave(440.0, dur, sr(), 0.8);
                std::hint::black_box(a)
            })
        });

        g.bench_function(format!("chirp_{label}"), |b| {
            b.iter(|| {
                let a: AudioSamples<'static, f32> = chirp(100.0, 8_000.0, dur, sr(), 0.8);
                std::hint::black_box(a)
            })
        });

        g.bench_function(format!("exponential_chirp_{label}"), |b| {
            b.iter(|| {
                let a: AudioSamples<'static, f32> =
                    exponential_chirp(100.0, 8_000.0, dur, sr(), 0.8);
                std::hint::black_box(a)
            })
        });

        g.bench_function(format!("fm_signal_{label}"), |b| {
            b.iter(|| {
                let a: AudioSamples<'static, f32> = fm_signal(440.0, 110.0, 5.0, dur, sr(), 0.8);
                std::hint::black_box(a)
            })
        });

        g.bench_function(format!("compound_tone_{label}"), |b| {
            let comps = [
                ToneComponent::new(440.0, 1.0),
                ToneComponent::new(880.0, 0.5),
                ToneComponent::new(1320.0, 0.25),
            ];
            let slice = NonEmptySlice::from_slice(&comps).unwrap();
            b.iter(|| {
                let a: AudioSamples<'static, f32> = compound_tone(slice, dur, sr());
                std::hint::black_box(a)
            })
        });
    }
    g.finish();
}

fn bench_bandlimited(c: &mut Criterion) {
    // Band-limited oscillators do additive harmonic synthesis -> heavier per sample.
    let mut g = c.benchmark_group("generation_bandlimited");
    for (label, secs) in DURATIONS {
        let dur = Duration::from_secs(secs);
        g.throughput(Throughput::Elements((SR as u64) * secs));

        g.bench_function(format!("square_bandlimited_{label}"), |b| {
            b.iter(|| {
                let a: AudioSamples<'static, f32> = square_wave_bandlimited(110.0, dur, sr(), 0.8);
                std::hint::black_box(a)
            })
        });

        g.bench_function(format!("sawtooth_bandlimited_{label}"), |b| {
            b.iter(|| {
                let a: AudioSamples<'static, f32> =
                    sawtooth_wave_bandlimited(110.0, dur, sr(), 0.8);
                std::hint::black_box(a)
            })
        });

        g.bench_function(format!("triangle_bandlimited_{label}"), |b| {
            b.iter(|| {
                let a: AudioSamples<'static, f32> =
                    triangle_wave_bandlimited(110.0, dur, sr(), 0.8);
                std::hint::black_box(a)
            })
        });
    }
    g.finish();
}

#[cfg(feature = "random-generation")]
fn bench_noise(c: &mut Criterion) {
    use audio_samples::{pink_noise, white_noise};

    let mut g = c.benchmark_group("generation_noise");
    for (label, secs) in DURATIONS {
        let dur = Duration::from_secs(secs);
        g.throughput(Throughput::Elements((SR as u64) * secs));

        g.bench_function(format!("white_noise_{label}"), |b| {
            b.iter(|| {
                let a: AudioSamples<'static, f32> = white_noise(dur, sr(), 0.5, Some(12345));
                std::hint::black_box(a)
            })
        });

        g.bench_function(format!("pink_noise_{label}"), |b| {
            b.iter(|| {
                let a: AudioSamples<'static, f32> = pink_noise(dur, sr(), 0.5, Some(12345));
                std::hint::black_box(a)
            })
        });
    }
    g.finish();
}

#[cfg(not(feature = "random-generation"))]
fn bench_noise(_c: &mut Criterion) {}

criterion_group! {
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_millis(500)).measurement_time(Duration::from_secs(3));
    targets = bench_simple, bench_bandlimited, bench_noise
}
criterion_main!(benches);
