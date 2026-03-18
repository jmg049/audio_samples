use audio_samples_benchmarks::{Args, print_csv_header, print_csv_row, run_timed};

use audio_samples::{
    AudioIirFiltering,
    operations::types::IirFilterDesign,
    sample_rate, sine_wave,
};
use std::{num::NonZeroUsize, time::Duration};

fn bench(dur_s: u64, iters: usize, warmup: usize,
         design: &IirFilterDesign, op_prefix: &str, impl_name: &str) {
    let original = sine_wave::<f32>(440.0, Duration::from_secs(dur_s), sample_rate!(44100), 0.8);
    let n = original.samples_per_channel().get();

    let s = run_timed(iters, warmup, || {
        // Clone each iteration: apply_iir_filter mutates in place, so we need
        // a fresh copy to avoid accumulating filter state across iterations.
        let mut audio = original.clone();
        audio.apply_iir_filter(design).expect("filter should not fail");
        std::hint::black_box(audio);
    });

    let op = format!("{op_prefix}_{dur_s}s");
    print_csv_row(&op, impl_name, dur_s, n, iters, warmup, &s);
}

fn main() {
    audio_samples_benchmarks::bench_init();
    let a = Args::parse_with_defaults(200, 20);
    let order = NonZeroUsize::new(2).unwrap();

    print_csv_header();

    let lp = IirFilterDesign::butterworth_lowpass(order, 1000.0);
    bench(a.duration_s, a.iterations, a.warmup,
          &lp, "lowpass_1000hz_order2", "audio_samples_butterworth");

    let hp = IirFilterDesign::butterworth_highpass(order, 1000.0);
    bench(a.duration_s, a.iterations, a.warmup,
          &hp, "highpass_1000hz_order2", "audio_samples_butterworth");
}
