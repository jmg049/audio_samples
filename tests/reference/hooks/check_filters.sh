#!/usr/bin/env bash
# Run the filter unit tests and the golden-reference suite.
set -euo pipefail
cd "$(dirname "$0")/../../.."
FEATURES="iir-filtering,parametric-eq,processing,channels"
cargo test --quiet --release --features "$FEATURES" --test filter_reference
cargo test --quiet --release --features "$FEATURES" --lib -- iir_filtering iir_design parametric_eq processing
