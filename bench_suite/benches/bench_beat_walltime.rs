//! WallTime bench target for the `AudioBeatTracking` trait.
//!
//! Methodology: `bench_suite/METHODOLOGY.md` §3.5 (WallTime baseline).
//! Catalog: `bench_suite/CATALOG.md` Beat-001 .. Beat-003.

#[path = "../benches_shared/beat.rs"]
mod body;

use bench_suite_common::build_criterion;
use criterion::{criterion_group, criterion_main};

criterion_group! {
    name    = benches;
    config  = build_criterion();
    targets = body::bench_all,
}
criterion_main!(benches);
