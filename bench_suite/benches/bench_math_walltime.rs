//! WallTime bench target for the `utils::audio_math` functions.
//!
//! Methodology: `bench_suite/METHODOLOGY.md` §3.5 (WallTime baseline).
//! Catalog: `bench_suite/CATALOG.md` Math-001 .. Math-025.

#[path = "../benches_shared/math.rs"]
mod body;

use bench_suite_common::build_criterion;
use criterion::{criterion_group, criterion_main};

criterion_group! {
    name    = benches;
    config  = build_criterion();
    targets = body::bench_all,
}
criterion_main!(benches);
