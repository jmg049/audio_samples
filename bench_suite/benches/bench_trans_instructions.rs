//! PMU bench target — retired instructions — for `AudioTransforms`.
//!
//! Linux only. Requires `perf_event_paranoid <= 1` or `CAP_PERFMON`.
//! Methodology: `bench_suite/METHODOLOGY.md` §3.5.

#[path = "../benches_shared/trans.rs"]
mod body;

use bench_suite_common::build_criterion_perf_instructions;
use criterion::{criterion_group, criterion_main};

criterion_group! {
    name    = benches;
    config  = build_criterion_perf_instructions();
    targets = body::bench_all,
}
criterion_main!(benches);
