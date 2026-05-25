#!/usr/bin/env bash
# write_manifest.sh — emit the per-run manifest.json described in
# bench_suite/METHODOLOGY.md §9.6.
#
# Usage:
#   write_manifest.sh <run_id> <comma-separated-features> <feature_set_tag>

set -euo pipefail

RUN_ID="${1:?run_id required}"
FEATURES="${2:?features required}"
FEATURE_SET_TAG="${3:?feature_set_tag required}"

# host info
KERNEL="$(uname -srm)"
CPU_MODEL="$(awk -F': ' '/^model name/ {print $2; exit}' /proc/cpuinfo 2>/dev/null || echo unknown)"
CPU_COUNT="$(nproc 2>/dev/null || echo 0)"
RAM_KB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
RAM_GB=$(( RAM_KB / 1024 / 1024 ))

# git info
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
COMMIT_FULL="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
DIRTY=false
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    DIRTY=true
fi

# toolchain
RUSTC_VER="$(rustc --version 2>/dev/null || echo unknown)"
RUSTFLAGS_VAL="${RUSTFLAGS:-}"

DATE_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# audio_samples version from root Cargo.toml
AS_VER="$(awk -F'"' '/^version = / {print $2; exit}' Cargo.toml 2>/dev/null || echo unknown)"

# perf_event_paranoid value (Linux only)
PEP="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo n/a)"

# Convert features csv into json array
IFS=',' read -ra FEATS <<< "$FEATURES"
FEAT_JSON="["
SEP=""
for f in "${FEATS[@]}"; do
    FEAT_JSON+="${SEP}\"$f\""
    SEP=", "
done
FEAT_JSON+="]"

cat <<EOF
{
  "run_id": "${RUN_ID}",
  "feature_set_tag": "${FEATURE_SET_TAG}",
  "commit": "${COMMIT}",
  "commit_full": "${COMMIT_FULL}",
  "commit_dirty": ${DIRTY},
  "branch": "${BRANCH}",
  "date_utc": "${DATE_UTC}",
  "host": {
    "kernel": "${KERNEL}",
    "cpu_model": "${CPU_MODEL}",
    "cpu_count": ${CPU_COUNT},
    "ram_gb": ${RAM_GB},
    "perf_event_paranoid": "${PEP}"
  },
  "rustc": "${RUSTC_VER}",
  "rustflags": "${RUSTFLAGS_VAL}",
  "features": ${FEAT_JSON},
  "criterion": "0.8.2",
  "audio_samples": "${AS_VER}"
}
EOF
