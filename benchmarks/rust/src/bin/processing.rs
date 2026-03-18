use audio_samples_benchmarks::{Args, print_csv_header, print_csv_row, run_timed};

use audio_samples::{
    AudioProcessing,
    operations::types::NormalizationConfig,
    sample_rate, sine_wave,
};
use std::time::Duration;

fn bench_scale(dur_s: u64, iters: usize, warmup: usize) {
    let audio = sine_wave::<f32>(440.0, Duration::from_secs(dur_s), sample_rate!(44100), 0.8);
    let n = audio.samples_per_channel().get();

    // scale() consumes self; clone inside the timed region matches the C benchmark
    // which does memcpy + in-place scale.
    let s = run_timed(iters, warmup, || {
        let out = audio.clone().scale(0.5f64);
        std::hint::black_box(out);
    });

    let op = format!("scale_by_half_{dur_s}s");
    print_csv_row(&op, "audio_samples", dur_s, n, iters, warmup, &s);
}

fn bench_clip(dur_s: u64, iters: usize, warmup: usize) {
    let audio = sine_wave::<f32>(440.0, Duration::from_secs(dur_s), sample_rate!(44100), 0.8);
    let n = audio.samples_per_channel().get();

    // clone() + clip() matches the C benchmark: memcpy + in-place clamp.
    let s = run_timed(iters, warmup, || {
        let out = audio.clone().clip(-0.5f32, 0.5f32)
            .expect("clip should not fail");
        std::hint::black_box(out);
    });

    let op = format!("clip_{dur_s}s");
    print_csv_row(&op, "audio_samples", dur_s, n, iters, warmup, &s);
}

/// Benchmark for clip_in_place — measures pure compute with no allocation.
/// Resets the work buffer each iteration using clone_from (reuses allocation).
fn bench_clip_inplace(dur_s: u64, iters: usize, warmup: usize) {
    let source = sine_wave::<f32>(440.0, Duration::from_secs(dur_s), sample_rate!(44100), 0.8);
    let n = source.samples_per_channel().get();
    let mut work = source.clone();

    let s = run_timed(iters, warmup, || {
        // Reset work to original data (reuses existing allocation via clone_from).
        work.clone_from(&source);
        work.clip_in_place(-0.5f32, 0.5f32)
            .expect("clip_in_place should not fail");
        std::hint::black_box(&work);
    });

    let op = format!("clip_inplace_{dur_s}s");
    print_csv_row(&op, "audio_samples_inplace", dur_s, n, iters, warmup, &s);
}

fn bench_normalize(dur_s: u64, iters: usize, warmup: usize) {
    let audio = sine_wave::<f32>(440.0, Duration::from_secs(dur_s), sample_rate!(44100), 0.8);
    let n = audio.samples_per_channel().get();

    let s = run_timed(iters, warmup, || {
        let out = audio.clone()
            .normalize(NormalizationConfig::peak(1.0f32))
            .expect("normalize should not fail");
        std::hint::black_box(out);
    });

    let op = format!("normalize_peak_{dur_s}s");
    print_csv_row(&op, "audio_samples", dur_s, n, iters, warmup, &s);
}

fn main() {
    audio_samples_benchmarks::bench_init();
    let a = Args::parse_with_defaults(1000, 100);
    print_csv_header();
    bench_scale(a.duration_s, a.iterations, a.warmup);
    bench_clip(a.duration_s, a.iterations, a.warmup);
    bench_clip_inplace(a.duration_s, a.iterations, a.warmup);
    bench_normalize(a.duration_s, a.iterations, a.warmup);
}
