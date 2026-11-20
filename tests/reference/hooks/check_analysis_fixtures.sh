#!/usr/bin/env bash
# Fail if the committed analysis fixtures differ from what the generator
# produces. Requires `uv` (the generator's pinned dependencies are installed on
# first use).
set -euo pipefail
cd "$(dirname "$0")/../../.."
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; skipping analysis fixture freshness check" >&2
  exit 0
fi
uv run --quiet --project tests/reference/analysis tests/reference/analysis/generate_analysis_fixtures.py --check
