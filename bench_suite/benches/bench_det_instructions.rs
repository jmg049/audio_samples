//! PMU bench target — retired instructions — for `utils::detection`.

#[path = "../benches_shared/det.rs"]
mod body;

use bench_suite_common::build_criterion_perf_instructions;
use criterion::{criterion_group, criterion_main};

criterion_group! {
    name    = benches;
    config  = build_criterion_perf_instructions();
    targets = body::bench_all,
}
criterion_main!(benches);
