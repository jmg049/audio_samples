//! PMU bench target — branch mispredictions — for type conversion.

#[path = "../benches_shared/conv.rs"]
mod body;

use bench_suite_common::build_criterion_perf_branch_misses;
use criterion::{criterion_group, criterion_main};

criterion_group! {
    name    = benches;
    config  = build_criterion_perf_branch_misses();
    targets = body::bench_all,
}
criterion_main!(benches);
