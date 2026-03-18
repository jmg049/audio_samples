use audio_samples_benchmarks::{Args, print_csv_header, print_csv_row, run_timed};

use audio_samples::{resample, sample_rate, sine_wave, operations::ResamplingQuality};
use std::time::Duration;

fn bench(dur_s: u64, iters: usize, warmup: usize, dst_hz: u32,
         impl_name: &str, quality: ResamplingQuality) {
    let audio = sine_wave::<f32>(440.0, Duration::from_secs(dur_s), sample_rate!(44100), 0.8);
    let dst_sr = std::num::NonZeroU32::new(dst_hz).unwrap();
    let n = audio.samples_per_channel().get();

    let s = run_timed(iters, warmup, || {
        let _ = resample(&audio, dst_sr, quality)
            .expect("resample should not fail");
    });

    let op = format!("resample_44100_to_{dst_hz}_{dur_s}s");
    print_csv_row(&op, impl_name, dur_s, n, iters, warmup, &s);
}

fn main() {
    audio_samples_benchmarks::bench_init();
    let a = Args::parse_with_defaults(1000, 100);
    print_csv_header();
    bench(a.duration_s, a.iterations, a.warmup, 16000, "audio_samples_fast",   ResamplingQuality::Fast);
    bench(a.duration_s, a.iterations, a.warmup, 16000, "audio_samples_medium", ResamplingQuality::Medium);
    bench(a.duration_s, a.iterations, a.warmup, 16000, "audio_samples_high",   ResamplingQuality::High);
}
