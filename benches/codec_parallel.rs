//! Benchmark: sequential vs parallel perceptual codec encode/decode.
//!
//! Three scenarios:
//! - **no_switching** — single segment, no window switching (baseline).
//! - **switching_sine** — window switching on a pure sine; sine has no transients
//!   so all frames collapse into one segment — measures window-switching overhead.
//! - **switching_transient** — window switching on a signal with amplitude jumps;
//!   each jump triggers segment splits, giving rayon something to actually parallelise.
//!
//! ## Running
//!
//! ```bash
//! # Sequential
//! cargo bench --bench codec_parallel \
//!   --no-default-features --features psychoacoustic
//!
//! # Parallel
//! cargo bench --bench codec_parallel \
//!   --no-default-features --features "psychoacoustic,parallel"
//! ```

use std::num::NonZeroUsize;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

use audio_samples::{
    BandLayout, PerceptualCodec, PsychoacousticConfig,
    codecs::{decode, encode},
    sample_rate, silence, sine_wave,
};
use spectrograms::WindowType;

const SR: u32 = 44100;
const N_BANDS: usize = 24;
const N_BINS: usize = 1024;
const LONG_WIN: usize = 2048;
const SHORT_WIN: usize = 256;
const TRANSIENT_THRESHOLD: f32 = 8.0;

fn codec_no_switch() -> PerceptualCodec {
    let n_bands = NonZeroUsize::new(N_BANDS).unwrap();
    let n_bins = NonZeroUsize::new(N_BINS).unwrap();
    let layout = BandLayout::bark(n_bands, SR as f32, n_bins);
    let weights = PsychoacousticConfig::uniform_weights(n_bands);
    let config = PsychoacousticConfig::mpeg1(weights.as_non_empty_slice());
    PerceptualCodec::with_window_size(
        layout,
        config,
        WindowType::Hanning,
        128_000,
        1,
        NonZeroUsize::new(LONG_WIN).unwrap(),
    )
}

fn codec_switching() -> PerceptualCodec {
    let n_bands = NonZeroUsize::new(N_BANDS).unwrap();
    let n_bins = NonZeroUsize::new(N_BINS).unwrap();
    let layout = BandLayout::bark(n_bands, SR as f32, n_bins);
    let weights = PsychoacousticConfig::uniform_weights(n_bands);
    let config = PsychoacousticConfig::mpeg1(weights.as_non_empty_slice());
    PerceptualCodec::with_window_switching(
        layout,
        config,
        WindowType::Hanning,
        128_000,
        1,
        NonZeroUsize::new(LONG_WIN).unwrap(),
        NonZeroUsize::new(SHORT_WIN).unwrap(),
        TRANSIENT_THRESHOLD,
    )
}

/// Sine wave — no transients, produces one segment even with window switching.
fn signal_sine(ms: u64) -> audio_samples::AudioSamples<'static, f32> {
    sine_wave::<f32>(440.0, Duration::from_millis(ms), sample_rate!(44100), 0.5)
}

/// Alternating loud tones and silence — each onset is a clear transient.
/// Produces ~2 × n_bursts segments under window switching.
fn signal_transient(n_bursts: usize) -> audio_samples::AudioSamples<'static, f32> {
    use audio_samples::AudioSamples;
    let burst_ms = 50u64;
    let gap_ms = 50u64;
    let sr = sample_rate!(44100);

    let burst = sine_wave::<f32>(880.0, Duration::from_millis(burst_ms), sr, 1.0);
    let gap = silence::<f32>(Duration::from_millis(gap_ms), sr);

    let burst_samples = burst.as_slice().expect("contiguous").to_vec();
    let gap_samples = gap.as_slice().expect("contiguous").to_vec();

    let mut flat: Vec<f32> =
        Vec::with_capacity((burst_samples.len() + gap_samples.len()) * n_bursts);
    for _ in 0..n_bursts {
        flat.extend_from_slice(&burst_samples);
        flat.extend_from_slice(&gap_samples);
    }

    // Safety: flat has n_bursts * (burst + gap) samples, always non-empty.
    let nev = unsafe { non_empty_slice::NonEmptyVec::new_unchecked(flat) };
    AudioSamples::<'static, f32>::from_mono_vec(nev, sr)
}

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec/encode");

    for &ms in &[200u64, 500, 1000] {
        let sine = signal_sine(ms);

        group.bench_with_input(
            BenchmarkId::new("no_switching", format!("{ms}ms")),
            &ms,
            |b, _| b.iter(|| encode(&sine, codec_no_switch()).unwrap()),
        );

        group.bench_with_input(
            BenchmarkId::new("switching_sine", format!("{ms}ms")),
            &ms,
            |b, _| b.iter(|| encode(&sine, codec_switching()).unwrap()),
        );
    }

    // Transient signal: vary number of burst/gap pairs (each pair = ~2 segments).
    for &n_bursts in &[4usize, 8, 16] {
        let transient = signal_transient(n_bursts);
        group.bench_with_input(
            BenchmarkId::new("switching_transient", format!("{n_bursts}_bursts")),
            &n_bursts,
            |b, _| b.iter(|| encode(&transient, codec_switching()).unwrap()),
        );
    }

    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("codec/decode");

    for &ms in &[200u64, 500, 1000] {
        let sine = signal_sine(ms);
        let enc_no_switch = encode(&sine, codec_no_switch()).unwrap();
        let enc_switching = encode(&sine, codec_switching()).unwrap();

        // iter_batched: clone happens outside the measured region.
        group.bench_with_input(
            BenchmarkId::new("no_switching", format!("{ms}ms")),
            &ms,
            |b, _| {
                b.iter_batched(
                    || enc_no_switch.clone(),
                    |enc| decode::<PerceptualCodec, f32>(enc).unwrap(),
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("switching_sine", format!("{ms}ms")),
            &ms,
            |b, _| {
                b.iter_batched(
                    || enc_switching.clone(),
                    |enc| decode::<PerceptualCodec, f32>(enc).unwrap(),
                    BatchSize::SmallInput,
                )
            },
        );
    }

    for &n_bursts in &[4usize, 8, 16] {
        let transient = signal_transient(n_bursts);
        let enc_transient = encode(&transient, codec_switching()).unwrap();

        group.bench_with_input(
            BenchmarkId::new("switching_transient", format!("{n_bursts}_bursts")),
            &n_bursts,
            |b, _| {
                b.iter_batched(
                    || enc_transient.clone(),
                    |enc| decode::<PerceptualCodec, f32>(enc).unwrap(),
                    BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode);
criterion_main!(benches);
