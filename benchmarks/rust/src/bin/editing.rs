use audio_samples_benchmarks::{Args, print_csv_header, print_csv_row, run_timed};

use audio_samples::{
    AudioEditing,
    operations::types::FadeCurve,
    sample_rate, sine_wave,
};
use std::time::Duration;

fn bench_trim(dur_s: u64, iters: usize, warmup: usize) {
    let audio = sine_wave::<f32>(440.0, Duration::from_secs(dur_s), sample_rate!(44100), 0.8);
    let dur_f = dur_s as f64;
    let n = audio.samples_per_channel().get();

    // trim() borrows &self and returns a new owned AudioSamples.
    let s = run_timed(iters, warmup, || {
        let out = audio.trim(dur_f * 0.25, dur_f * 0.75)
            .expect("trim should not fail");
        std::hint::black_box(out);
    });

    let op = format!("trim_middle_{dur_s}s");
    print_csv_row(&op, "audio_samples", dur_s, n, iters, warmup, &s);
}

fn bench_pad(dur_s: u64, iters: usize, warmup: usize) {
    let audio = sine_wave::<f32>(440.0, Duration::from_secs(dur_s), sample_rate!(44100), 0.8);
    let n = audio.samples_per_channel().get();

    // pad() borrows &self and returns a new owned AudioSamples (n + sr/2 samples).
    let s = run_timed(iters, warmup, || {
        let out = audio.pad(0.0, 0.5, 0.0f32)
            .expect("pad should not fail");
        std::hint::black_box(out);
    });

    let op = format!("pad_end_{dur_s}s");
    print_csv_row(&op, "audio_samples", dur_s, n, iters, warmup, &s);
}

fn bench_fade_in(dur_s: u64, iters: usize, warmup: usize) {
    let audio = sine_wave::<f32>(440.0, Duration::from_secs(dur_s), sample_rate!(44100), 0.8);
    let n = audio.samples_per_channel().get();

    // fade_in() mutates &mut self, so clone into a fresh copy each iteration.
    let s = run_timed(iters, warmup, || {
        let mut a = audio.clone();
        a.fade_in(0.5, FadeCurve::Linear).expect("fade_in should not fail");
        std::hint::black_box(a);
    });

    let op = format!("fade_in_{dur_s}s");
    print_csv_row(&op, "audio_samples", dur_s, n, iters, warmup, &s);
}

fn main() {
    audio_samples_benchmarks::bench_init();
    let a = Args::parse_with_defaults(1000, 100);
    print_csv_header();
    bench_trim(a.duration_s, a.iterations, a.warmup);
    bench_pad(a.duration_s, a.iterations, a.warmup);
    bench_fade_in(a.duration_s, a.iterations, a.warmup);
}
