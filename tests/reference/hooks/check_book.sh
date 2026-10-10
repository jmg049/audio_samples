#!/usr/bin/env bash
# Guard the mdBook of the learning guide (book/).
#
# Builds the figures crate (release, incremental, reusing book/figures/target),
# re-runs the figure programs whose binaries changed (their assertions keep the
# figures in line with the numbers the chapters state), builds the book with
# mdbook + mdbook-katex and checks the output: no KaTeX errors, no broken
# relative links, no missing or empty figures, and every figure marker in the
# chapters resolves (the preprocessor fails the build otherwise).
#
# Usage: tests/reference/hooks/check_book.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."

missing=()
command -v mdbook >/dev/null 2>&1 || missing+=(mdbook)
command -v mdbook-katex >/dev/null 2>&1 || missing+=(mdbook-katex)
if [ "${#missing[@]}" -gt 0 ]; then
    cat >&2 <<MSG
check_book: ${missing[*]} not found on PATH. Install the book tools with:
    cargo install mdbook --version 0.5.4 --locked
    cargo install mdbook-katex --version 0.10.0
(see book/README.md)
MSG
    exit 1
fi

cargo fmt --manifest-path book/figures/Cargo.toml -- --check
book/build.sh
