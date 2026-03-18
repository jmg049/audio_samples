use audio_samples_benchmarks::{Args, print_csv_header, print_csv_row, run_timed};

use audio_samples::{AudioStatistics, sample_rate, sine_wave};
use std::time::Duration;

fn bench(dur_s: u64, iters: usize, warmup: usize) {
    let audio = sine_wave::<f32>(440.0, Duration::from_secs(dur_s), sample_rate!(44100), 0.5);
    let n = audio.samples_per_channel().get();

    // Use rms_and_peak() for a fair single-pass comparison with the C benchmark,
    // which also computes both in one loop (see benchmarks/c/volume.c).
    let s = run_timed(iters, warmup, || {
        std::hint::black_box(audio.rms_and_peak());
    });

    let op = format!("rms_and_peak_{dur_s}s");
    print_csv_row(&op, "audio_samples", dur_s, n, iters, warmup, &s);
}

fn main() {
    audio_samples_benchmarks::bench_init();
    let a = Args::parse_with_defaults(1000, 100);
    print_csv_header();
    bench(a.duration_s, a.iterations, a.warmup);
}
