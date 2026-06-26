//! FFT / spectral-transform throughput benchmarks.
//!
//! These exercise the thread-local FFT-planner cache added in the 2.0 work:
//! `spectrograms::FftPlanner` memoizes plans (twiddles, scratch) by size, and
//! the cache is reused across calls within a thread. A single call would not
//! show plan reuse, so each case below performs MANY same-size transforms per
//! iteration (per-frame FFTs or N repeated op calls) so the benefit of reusing
//! the cached plan over per-call planning is actually measured.
//!
//! Run with:
//!   cargo bench --bench transforms --features "transforms statistics"

use std::num::NonZeroUsize;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use audio_samples::operations::types::ChannelReduction;
use audio_samples::{AudioSamples, AudioStatistics, AudioTransforms, sine_wave};
use spectrograms::{StftParams, WindowType};

const SR: u32 = 44_100;

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

fn signal(seconds: u64) -> AudioSamples<'static, f64> {
    // Single tone; content does not affect transform timing materially. f64 is
    // the transform working dtype.
    let sr = std::num::NonZeroU32::new(SR).unwrap();
    sine_wave::<f64>(440.0, Duration::from_secs(seconds), sr, 0.8)
}

fn bench_fft_frames(c: &mut Criterion) {
    // Compute an FFT per frame across a 1s signal: ~repeated same-size FFTs that
    // hammer the planner cache (the per-call-planning regression this fixed).
    let n_fft = 2048usize;
    let hop = 512usize;
    let audio = signal(1);
    let total = audio.samples_per_channel().get();
    let n_frames = (total.saturating_sub(n_fft)) / hop + 1;

    let mut g = c.benchmark_group("fft_repeated_frames");
    g.throughput(Throughput::Elements((n_frames * n_fft) as u64));

    g.bench_function(format!("fft_{n_fft}_x{n_frames}_frames"), |b| {
        b.iter(|| {
            // Many same-size FFTs in one iteration -> plan reuse is exercised.
            for _ in 0..n_frames {
                let spec = audio.fft(nz(n_fft)).unwrap();
                std::hint::black_box(&spec);
            }
        })
    });
    g.finish();
}

fn bench_stft(c: &mut Criterion) {
    let audio = signal(2);
    let params = StftParams::new(nz(1024), nz(256), WindowType::Hanning, true).unwrap();

    let mut g = c.benchmark_group("stft");
    g.throughput(Throughput::Elements(
        audio.samples_per_channel().get() as u64
    ));

    g.bench_function("stft_1024_hop256_hann", |b| {
        b.iter(|| std::hint::black_box(audio.stft(&params).unwrap()))
    });
    g.finish();
}

fn bench_psd(c: &mut Criterion) {
    let audio = signal(2);

    let mut g = c.benchmark_group("power_spectral_density");
    g.throughput(Throughput::Elements(
        audio.samples_per_channel().get() as u64
    ));

    g.bench_function("psd_welch_1024_overlap50", |b| {
        b.iter(|| std::hint::black_box(audio.power_spectral_density(nz(1024), 0.5).unwrap()))
    });
    g.finish();
}

fn bench_spectral_centroid(c: &mut Criterion) {
    // spectral_centroid does one FFT-based pass per call; call it N times per
    // iteration so repeated same-size planning hits the cached planner.
    let audio = signal(1);
    let calls = 32usize;

    let mut g = c.benchmark_group("spectral_centroid");
    g.throughput(Throughput::Elements(
        (audio.samples_per_channel().get() * calls) as u64,
    ));

    g.bench_function(format!("centroid_first_x{calls}"), |b| {
        b.iter(|| {
            for _ in 0..calls {
                let v = audio.spectral_centroid(ChannelReduction::First).unwrap();
                std::hint::black_box(v);
            }
        })
    });
    g.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_millis(500)).measurement_time(Duration::from_secs(3));
    targets = bench_fft_frames, bench_stft, bench_psd, bench_spectral_centroid
}
criterion_main!(benches);
