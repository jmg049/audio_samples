//! PMU bench target — L1 data cache read misses — for `AudioStatistics`.
//!
//! Linux only. Requires `perf_event_paranoid <= 1` or `CAP_PERFMON`.
//! Methodology: `bench_suite/METHODOLOGY.md` §3.5.

#[path = "../benches_shared/stats.rs"]
mod body;

use bench_suite_common::build_criterion_perf_cache_misses;
use criterion::{criterion_group, criterion_main};

criterion_group! {
    name    = benches;
    config  = build_criterion_perf_cache_misses();
    targets = body::bench_all,
}
criterion_main!(benches);
