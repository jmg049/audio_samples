use audio_samples_benchmarks::{Args, print_csv_header, print_csv_row, run_timed};

use audio_samples::{
    AudioChannelOps,
    operations::types::MonoConversionMethod,
    sample_rate, sine_wave, stereo_sine_wave,
};
use std::time::Duration;

fn bench_stereo_to_mono(dur_s: u64, iters: usize, warmup: usize) {
    let stereo = stereo_sine_wave::<f32>(440.0, Duration::from_secs(dur_s), sample_rate!(44100), 0.8);
    let n = stereo.samples_per_channel().get();

    let s = run_timed(iters, warmup, || {
        let mono = stereo.to_mono(MonoConversionMethod::Average)
            .expect("to_mono should not fail");
        std::hint::black_box(mono);
    });

    let op = format!("stereo_to_mono_{dur_s}s");
    print_csv_row(&op, "audio_samples", dur_s, n, iters, warmup, &s);
}

fn bench_mono_to_stereo(dur_s: u64, iters: usize, warmup: usize) {
    let mono = sine_wave::<f32>(440.0, Duration::from_secs(dur_s), sample_rate!(44100), 0.8);
    let n = mono.samples_per_channel().get();
    let s = run_timed(iters, warmup, || {
        let stereo = mono.duplicate_to_channels(2)
            .expect("duplicate_to_channels should not fail");
        std::hint::black_box(stereo);
    });

    let op = format!("mono_to_stereo_{dur_s}s");
    print_csv_row(&op, "audio_samples", dur_s, n, iters, warmup, &s);
}

fn main() {
    audio_samples_benchmarks::bench_init();
    let a = Args::parse_with_defaults(1000, 100);
    print_csv_header();
    bench_stereo_to_mono(a.duration_s, a.iterations, a.warmup);
    bench_mono_to_stereo(a.duration_s, a.iterations, a.warmup);
}
