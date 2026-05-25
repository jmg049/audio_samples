//! Cycles PMU bench target for the `AudioPlotting` trait.

#[path = "../benches_shared/plot.rs"]
mod body;

use bench_suite_common::build_criterion_perf_cycles;
use criterion::{criterion_group, criterion_main};

criterion_group! {
    name    = benches;
    config  = build_criterion_perf_cycles();
    targets = body::bench_all,
}
criterion_main!(benches);
