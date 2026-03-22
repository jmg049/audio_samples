#!/usr/bin/env bash
# run.sh — build both sides, run all benchmarks, print a comparison table.
#
# Usage:
#   bash benchmarks/run.sh [--duration 1] [--iterations 1000] [--warmup 100]
#   bash benchmarks/run.sh --duration 10 --iterations 500 --warmup 50
#
# Run from the audio_samples repo root.

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$BENCH_DIR/.." && pwd)"

DURATIONS=(1)
ITERATIONS="${ITERATIONS:-1000}"
WARMUP="${WARMUP:-100}"
FILTER_ITER="${FILTER_ITER:-200}"
FILTER_WARMUP="${FILTER_WARMUP:-20}"

# Parse flags.  --duration accepts one or more values: --duration 1 5 10
while [[ $# -gt 0 ]]; do
  case $1 in
    --duration)
      shift; DURATIONS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do DURATIONS+=("$1"); shift; done
      ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --warmup)     WARMUP="$2";     shift 2 ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

# --------------------------------------------------------------------------
# Detect compiler and print header
# --------------------------------------------------------------------------
GCC_VER=$(gcc --version 2>/dev/null | head -1 || echo "gcc not found")
RUSTC_VER=$(rustc --version 2>/dev/null || echo "rustc not found")
FFMPEG_VER=$(ffmpeg -version 2>/dev/null | head -1 || echo "ffmpeg not found")

echo "============================================================"
echo " audio_samples vs FFmpeg — Head-to-Head Benchmark"
echo "============================================================"
echo " C    : $GCC_VER"
echo "        Flags: -O3 -march=native -ffast-math"
echo " Rust : $RUSTC_VER"
echo "        Flags: opt-level=3 lto=thin RUSTFLAGS=-C target-cpu=native"
echo " $FFMPEG_VER"
echo ""
echo " NOTE: FFmpeg ships hand-tuned SIMD assembly for resampling and"
echo "       filtering (libswresample, libavfilter).  audio_samples relies"
echo "       on auto-vectorisation only (the experimental 'simd' feature"
echo "       is not enabled).  This is a declared asymmetry, not a flaw"
echo "       in the comparison."
echo ""
echo " Filter note: FFmpeg's 'lowpass' filter is capped at 2 poles."
echo "       Both sides use 2nd-order Butterworth for the filter benchmark."
echo "============================================================"
echo ""

# --------------------------------------------------------------------------
# Build C
# --------------------------------------------------------------------------
echo ">>> Building C programs..."
make -C "$BENCH_DIR" --no-print-directory
echo ""

# --------------------------------------------------------------------------
# Build Rust
# --------------------------------------------------------------------------
echo ">>> Building Rust programs..."
RUSTFLAGS="-C target-cpu=native" cargo build --release \
  --manifest-path "$BENCH_DIR/rust/Cargo.toml" \
  --quiet
echo ""

C_BIN="$BENCH_DIR/build"
R_BIN="$BENCH_DIR/rust/target/release"

# --------------------------------------------------------------------------
# CSV results file in benchmarks/
# --------------------------------------------------------------------------
RESULTS_FILE="$BENCH_DIR/results.csv"
echo "operation,implementation,duration_s,n_samples,iterations,warmup,min_us,mean_us,median_us,max_us,stddev_us" > "$RESULTS_FILE"

# --------------------------------------------------------------------------
# Helper: run one pair, append to RESULTS_FILE, show progress
# --------------------------------------------------------------------------
run_pair() {
  local label="$1"  # display name
  local name="$2"   # binary name
  local c_args="$3"
  local r_args="$4"

  printf "  %-20s " "$label..."

  # C side — skip CSV header (first line), append data rows
  "$C_BIN/$name" $c_args 2>/dev/null | tail -n +2 >> "$RESULTS_FILE"

  # Rust side — skip CSV header, append data rows
  "$R_BIN/$name"  $r_args 2>/dev/null | tail -n +2 >> "$RESULTS_FILE"

  echo "done"
}

DUR_LIST="${DURATIONS[*]}"  # for display
echo ">>> Running benchmarks (durations=${DUR_LIST}s, iterations=${ITERATIONS}, warmup=${WARMUP})..."
echo ""

for DUR in "${DURATIONS[@]}"; do
  echo "  --- ${DUR}s ---"
  COMMON="--duration $DUR --iterations $ITERATIONS --warmup $WARMUP"
  FARGS="--duration $DUR --iterations $FILTER_ITER --warmup $FILTER_WARMUP"
  run_pair "Resample"    resample    "$COMMON" "$COMMON"
  run_pair "Filter"      filter      "$FARGS"  "$FARGS"
  run_pair "Volume"      volume      "$COMMON" "$COMMON"
  run_pair "Channels"    channels    "$COMMON" "$COMMON"
  run_pair "Format conv" format_conv "$COMMON" "$COMMON"
  run_pair "Processing"  processing  "$COMMON" "$COMMON"
  run_pair "Editing"     editing     "$COMMON" "$COMMON"
  echo ""
done

echo "  Results saved to: $RESULTS_FILE"
echo ""

# --------------------------------------------------------------------------
# Pretty-print via the Rust show_results binary
# --------------------------------------------------------------------------
"$R_BIN/show_results" "$RESULTS_FILE"
