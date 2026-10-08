#!/usr/bin/env bash
# Fail if the committed filter fixtures differ from what the generator
# produces. Requires `uv` (Python deps are installed on first use) and ffmpeg.
# If the C/C++ harness (tests/reference/c) is not built, the DSPFilters /
# liquid-dsp cross references are not compared.
set -euo pipefail
cd "$(dirname "$0")/../../.."
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; skipping fixture freshness check" >&2
  exit 0
fi
uv run --quiet --project tests/reference tests/reference/generate_filter_fixtures.py --check
