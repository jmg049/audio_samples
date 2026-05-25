#!/usr/bin/env bash
# run_and_collect.sh — run a bench target, harvest its raw data into the
# unified CSV layout under bench_suite/results/<run_id>/.
#
# Usage:
#   bench_suite/scripts/run_and_collect.sh \
#       <feature_set_tag> \
#       <comma-separated-features> \
#       <bench_target_name> \
#       [-- <extra cargo bench args>]
#
# Example — WallTime smoke run of Stats-001 only:
#   bench_suite/scripts/run_and_collect.sh \
#       stats_smoke \
#       statistics \
#       bench_stats_walltime \
#       -- --sample-size 30 --measurement-time 2 "Stats-001"
#
# Example — full PMU instructions run:
#   bench_suite/scripts/run_and_collect.sh \
#       stats_instructions \
#       statistics,transforms,perf_events \
#       bench_stats_instructions

set -euo pipefail

FEATURE_SET_TAG="${1:?feature_set_tag required}"
FEATURES="${2:?features required}"
BENCH_TARGET="${3:?bench target name required}"
shift 3
# strip the optional leading -- separator
if [[ "${1:-}" == "--" ]]; then shift; fi
EXTRA_ARGS=("$@")

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD 2>/dev/null || echo unknown)_${FEATURE_SET_TAG}"
OUT_DIR="bench_suite/results/${RUN_ID}"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo " run_id:        ${RUN_ID}"
echo " bench_target:  ${BENCH_TARGET}"
echo " features:      ${FEATURES}"
echo " extra args:    ${EXTRA_ARGS[*]}"
echo " output:        ${OUT_DIR}"
echo "============================================================"

# 1) Write the manifest BEFORE the run so we capture the intended environment
#    even if the bench fails partway.
bash bench_suite/scripts/write_manifest.sh \
    "$RUN_ID" "$FEATURES" "$FEATURE_SET_TAG" \
    > "$OUT_DIR/manifest.json"

# 2) Run the bench. Single-core pinning + native target for repeatability.
#
# RUSTC_BOOTSTRAP=1 is set whenever a PMU bench target is invoked, because
# perfcnt 0.8.0 pulls in nom 4.2.3 which mis-detects nightly and tries to
# `feature(test)` on stable. The bootstrap env var lets stable rustc accept
# unstable features — harmless for our code (which uses none).
#
# Drop the workaround once perfcnt updates its nom pin.
BOOTSTRAP=""
case "$BENCH_TARGET" in
    *_instructions|*_cycles|*_cache_misses|*_branch_misses)
        BOOTSTRAP="RUSTC_BOOTSTRAP=1"
        ;;
esac

echo ">>> running bench…"
env $BOOTSTRAP RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}" \
    taskset -c 0 \
    cargo bench --bench "$BENCH_TARGET" --features "$FEATURES" -- "${EXTRA_ARGS[@]}"

# 3) Harvest the target/criterion_<measurement>/ subtree(s) into the unified CSVs.
echo ">>> harvesting…"
python3 bench_suite/scripts/harvest.py target "$OUT_DIR" --run-id "$RUN_ID"

echo
echo "done. results in: $OUT_DIR"
ls -la "$OUT_DIR"
