#!/usr/bin/env bash
# Guard the teaching material: the learning guide (doctests), the explain API
# and the tutorial labs. Every number a chapter or lab claims is asserted in
# code, so these runs are the ground-truth check for the course.
#
# Usage (pre-commit passes the changed files):
#   tests/reference/hooks/check_educational.sh [changed files...]
# With no arguments every tutorial lab is run.
set -euo pipefail
cd "$(dirname "$0")/../../.."
FEATURES="educational,full_no_plotting"

# Guide chapters and educational/explain doctests.
cargo test --quiet --doc --features "$FEATURES" -- guide educational
# Explain API unit + integration tests.
cargo test --quiet --features "$FEATURES" --lib -- educational
cargo test --quiet --features "$FEATURES" --test explain_api --test educational_integration

# Tutorial labs: run the ones whose lab or chapter changed (all if no args).
labs=()
if [ "$#" -eq 0 ]; then
    for f in examples/tutorial_*.rs; do labs+=("$(basename "$f" .rs)"); done
else
    for f in "$@"; do
        case "$f" in
            examples/tutorial_*.rs) labs+=("$(basename "$f" .rs)") ;;
            documentation/guide/[0-9][0-9]_*.md)
                n="$(basename "$f" | cut -c1-2)"
                for e in examples/tutorial_"$n"_*.rs; do
                    [ -e "$e" ] && labs+=("$(basename "$e" .rs)")
                done ;;
        esac
    done
fi
if [ "${#labs[@]}" -gt 0 ]; then
    # Deduplicate, then run each lab (each asserts its own numbers).
    mapfile -t labs < <(printf '%s\n' "${labs[@]}" | sort -u)
    for lab in "${labs[@]}"; do
        echo "running $lab"
        cargo run --quiet --release --features "$FEATURES" --example "$lab" >/dev/null
    done
fi
