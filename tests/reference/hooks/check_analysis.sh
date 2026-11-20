#!/usr/bin/env bash
# Run the analysis golden-fixture suite.
set -euo pipefail
cd "$(dirname "$0")/../../.."
cargo test --quiet --release --features full_no_plotting --test analysis_reference
