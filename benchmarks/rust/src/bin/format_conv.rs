use audio_samples_benchmarks::{Args, print_csv_header, print_csv_row, run_timed};

use audio_samples::{AudioTypeConversion, sample_rate, sine_wave};
use std::time::Duration;

fn bench_f32_to_i16(dur_s: u64, iters: usize, warmup: usize) {
    let audio = sine_wave::<f32>(440.0, Duration::from_secs(dur_s), sample_rate!(44100), 0.8);
    let n = audio.samples_per_channel().get();

    let s = run_timed(iters, warmup, || {
        let out = audio.to_format::<i16>();
        std::hint::black_box(out);
    });

    let op = format!("format_f32_to_i16_{dur_s}s");
    print_csv_row(&op, "audio_samples", dur_s, n, iters, warmup, &s);
}

fn bench_i16_to_f32(dur_s: u64, iters: usize, warmup: usize) {
    // Pre-convert to i16 outside the timing loop.
    let sine = sine_wave::<f32>(440.0, Duration::from_secs(dur_s), sample_rate!(44100), 0.8);
    let audio_i16 = sine.to_format::<i16>();
    let n = audio_i16.samples_per_channel().get();

    let s = run_timed(iters, warmup, || {
        let out = audio_i16.to_format::<f32>();
        std::hint::black_box(out);
    });

    let op = format!("format_i16_to_f32_{dur_s}s");
    print_csv_row(&op, "audio_samples", dur_s, n, iters, warmup, &s);
}

fn main() {
    audio_samples_benchmarks::bench_init();
    let a = Args::parse_with_defaults(1000, 100);
    print_csv_header();
    bench_f32_to_i16(a.duration_s, a.iterations, a.warmup);
    bench_i16_to_f32(a.duration_s, a.iterations, a.warmup);
}
