//! Filtering throughput benchmarks (IIR + FIR), f32 real-time room-correction scenarios.
//!
//! Run with:
//!   cargo bench --bench filtering --features "iir-filtering processing editing statistics transforms"

use std::num::NonZeroU32;
use std::num::NonZeroUsize;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};

use audio_samples::operations::traits::{AudioIirFiltering, AudioProcessing};
use audio_samples::operations::types::{FilterResponse, IirFilterDesign};
use audio_samples::AudioSamples;
use ndarray::Array1;
use non_empty_slice::NonEmptySlice;

const SR: u32 = 44_100;
const SECONDS: usize = 1;

fn signal(len: usize) -> Array1<f32> {
    // Deterministic two-tone + a little broadband-ish content.
    let mut v = Vec::with_capacity(len);
    for n in 0..len {
        let t = n as f32 / SR as f32;
        let s = 0.6 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
            + 0.3 * (2.0 * std::f32::consts::PI * 6_000.0 * t).sin()
            + 0.1 * ((n * 2654435761usize) as f32 / u32::MAX as f32 - 0.5);
        v.push(s);
    }
    Array1::from_vec(v)
}

fn make_audio(len: usize) -> AudioSamples<'static, f32> {
    AudioSamples::new_mono(signal(len), NonZeroU32::new(SR).unwrap()).unwrap()
}

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

fn bench_iir_whole(c: &mut Criterion) {
    let len = SR as usize * SECONDS;
    let mut g = c.benchmark_group("iir_whole_signal_f32");
    g.throughput(Throughput::Elements(len as u64));

    let cases: Vec<(&str, IirFilterDesign)> = vec![
        ("butter_lp_order2_single", IirFilterDesign::butterworth_lowpass(nz(2), 1_000.0)),
        ("butter_lp_order8_cascade", IirFilterDesign::butterworth_lowpass(nz(8), 1_000.0)),
        ("cheby1_hp_order6_cascade", IirFilterDesign::chebyshev_i(FilterResponse::HighPass, nz(6), 2_000.0, 1.0)),
    ];

    for (name, design) in cases {
        let base = make_audio(len);
        g.bench_function(name, |b| {
            b.iter_batched(
                || base.clone(),
                |mut audio| {
                    audio.apply_iir_filter(&design).unwrap();
                    audio
                },
                BatchSize::SmallInput,
            )
        });
    }
    g.finish();
}

fn bench_iir_streaming(c: &mut Criterion) {
    // Simulate real-time block processing: many small apply_iir_filter calls.
    // This captures the per-call filter *design* cost, which dominates here.
    let total = SR as usize * SECONDS;
    let block = 256usize;
    let mut g = c.benchmark_group("iir_streaming_256_f32");
    g.throughput(Throughput::Elements(total as u64));

    let design = IirFilterDesign::butterworth_lowpass(nz(8), 1_000.0);
    let full = signal(total);

    g.bench_function("butter_lp_order8_per_block_redesign", |b| {
        b.iter_batched(
            || full.clone(),
            |buf| {
                let mut start = 0;
                while start < total {
                    let end = (start + block).min(total);
                    let chunk = buf.slice(ndarray::s![start..end]).to_owned();
                    let mut audio =
                        AudioSamples::new_mono(chunk, NonZeroU32::new(SR).unwrap()).unwrap();
                    audio.apply_iir_filter(&design).unwrap();
                    start = end;
                }
            },
            BatchSize::SmallInput,
        )
    });
    g.finish();
}

fn bench_fir(c: &mut Criterion) {
    let len = SR as usize * SECONDS;
    let mut g = c.benchmark_group("fir_whole_signal_f32");
    g.throughput(Throughput::Elements(len as u64));

    for taps in [64usize, 1024, 4096] {
        // Simple normalized low-pass-ish coefficients (box filter); values irrelevant to timing.
        let coeffs: Vec<f32> = vec![1.0 / taps as f32; taps];
        let base = make_audio(len);
        g.bench_function(format!("apply_filter_{taps}_taps"), |b| {
            b.iter_batched(
                || base.clone(),
                |audio| {
                    let c = NonEmptySlice::from_slice(&coeffs).unwrap();
                    audio.apply_filter(c).unwrap()
                },
                BatchSize::SmallInput,
            )
        });
    }
    g.finish();
}

/// Real-time room-correction streaming path: block=1024, taps=4096 @ 44.1k.
/// Mirrors the user's app config (10.17 ms/buffer, RT 0.48x with direct conv).
/// Compares direct streaming FIR vs the overlap-save FFT convolver.
fn bench_room_correction_streaming(c: &mut Criterion) {
    use spectrograms::OverlapSaveConvolver;

    let total = SR as usize * SECONDS;
    let block = 1024usize;
    let taps = 4096usize;
    let n_blocks = total / block;
    let processed = n_blocks * block;

    let x = signal(total);
    let xv: Vec<f32> = x.to_vec();
    // Linear-phase-ish symmetric IR (values irrelevant to timing).
    let ir: Vec<f32> = (0..taps)
        .map(|k| {
            let c = (taps / 2) as f32;
            let d = (k as f32 - c) / c;
            (1.0 - d * d) / taps as f32
        })
        .collect();

    let mut g = c.benchmark_group("room_correction_streaming_b1024_t4096");
    g.throughput(Throughput::Elements(processed as u64));

    // Baseline: direct streaming FIR with history (what a no-FFT real-time path pays).
    g.bench_function("direct_streaming", |b| {
        b.iter(|| {
            let mut hist = vec![0.0f32; taps - 1];
            let mut out = vec![0.0f32; block];
            let mut start = 0;
            for _ in 0..n_blocks {
                let blk = &xv[start..start + block];
                for i in 0..block {
                    let mut acc = 0.0f32;
                    for (k, &h) in ir.iter().enumerate() {
                        let idx = i as isize - k as isize;
                        let s = if idx >= 0 {
                            blk[idx as usize]
                        } else {
                            hist[(hist.len() as isize + idx) as usize]
                        };
                        acc += h * s;
                    }
                    out[i] = acc;
                }
                // update history with last taps-1 samples of this block
                if block >= taps - 1 {
                    hist.copy_from_slice(&blk[block - (taps - 1)..]);
                }
                start += block;
                criterion::black_box(&out);
            }
        })
    });

    // New: overlap-save FFT convolver.
    g.bench_function("overlap_save_fft", |b| {
        let mut conv = OverlapSaveConvolver::new(&ir, NonZeroUsize::new(block).unwrap()).unwrap();
        let mut out = vec![0.0f32; block];
        b.iter(|| {
            conv.reset();
            let mut start = 0;
            for _ in 0..n_blocks {
                conv.process_block(&xv[start..start + block], &mut out).unwrap();
                start += block;
                criterion::black_box(&out);
            }
        })
    });

    g.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().warm_up_time(Duration::from_millis(500)).measurement_time(Duration::from_secs(3));
    targets = bench_iir_whole, bench_iir_streaming, bench_fir, bench_room_correction_streaming
}
criterion_main!(benches);
