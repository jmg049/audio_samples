//! Cache-misses PMU bench target for the `AudioVoiceActivityDetection` trait.

#[path = "../benches_shared/vad.rs"]
mod body;

use bench_suite_common::build_criterion_perf_cache_misses;
use criterion::{criterion_group, criterion_main};

criterion_group! {
    name    = benches;
    config  = build_criterion_perf_cache_misses();
    targets = body::bench_all,
}
criterion_main!(benches);
