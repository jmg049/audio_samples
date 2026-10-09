#!/usr/bin/env bash
# Fail if the committed librosa fixtures differ from what the generator
# produces. Requires `uv` (librosa and its pinned dependencies are installed on
# first use).
set -euo pipefail
cd "$(dirname "$0")/../../.."
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; skipping librosa fixture freshness check" >&2
  exit 0
fi
uv run --quiet --project tests/reference/librosa tests/reference/librosa/generate_librosa_fixtures.py --check
