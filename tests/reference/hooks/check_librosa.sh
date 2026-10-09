#!/usr/bin/env bash
# Run the librosa golden-reference suite.
set -euo pipefail
cd "$(dirname "$0")/../../.."
cargo test --quiet --release --features full_no_plotting --test librosa_reference
