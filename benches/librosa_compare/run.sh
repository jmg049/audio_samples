#!/usr/bin/env bash
# Run the librosa comparison benchmark end to end.
#
#   benches/librosa_compare/run.sh OUT_DIR [stage ...]
#
# Environment:
#   PY      Python with librosa, memray and the audio_samples bindings
#           (default: ../audio_samples_python/.venv/bin/python)
#   CORE    CPU to pin the single-thread runs to (default 10)
#   JOBS    parallel RSS probes (default 6)
#   REPEATS fresh-process repeats per RSS probe (default 5)
#   SEL     extra selection arguments for bench_py.py, e.g. "--groups spectrum"
#
# Stages (default: all, in this order):
#   build     build the Rust harness (release)
#   signals   write the test signals to target/librosa_bench/signals
#   rust      native timing + allocation counts, pinned, 1 thread
#   py        librosa and audio_samples-through-Python timing, pinned, 1 thread
#   mem       tracemalloc and memray per call (both Python paths)
#   rss       peak-RSS growth per call in fresh processes (both Python paths),
#             exact (rss_shim.c), REPEATS processes per case and input
#   default   timing with each library's default threading, unpinned
#   ties      rows within 3 % in time (py): both libraries again in one
#             process, alternating rounds; native rows at 0.97 or above:
#             five more processes, median kept; RSS rows above librosa:
#             ten more processes per library (needs a report stage first)
#   report    tables (report.py)
#
# Each stage appends JSON lines to OUT_DIR/<stage-file>.jsonl; delete the
# file to re-run a stage from scratch.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT="${1:?usage: run.sh OUT_DIR [stage ...]}"
shift || true
STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(build signals rust py mem rss default report)

PY="${PY:-$REPO/../audio_samples_python/.venv/bin/python}"
CORE="${CORE:-10}"
JOBS="${JOBS:-6}"
REPEATS="${REPEATS:-5}"
SEL="${SEL:-}"
SIG="$REPO/target/librosa_bench/signals"
# Same release settings as the Python bindings (thin LTO, one codegen unit), so
# that "ours (py)" minus "ours (rust)" is the binding overhead; separate target
# directory so the overrides do not invalidate target/release.
TARGET_DIR="$REPO/target/librosa_bench/build"
BIN="$TARGET_DIR/release/examples/librosa_bench"
ONE=(env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
     RAYON_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c "$CORE")

mkdir -p "$OUT"
# shellcheck disable=SC2086
for stage in "${STAGES[@]}"; do
  echo "== $stage" >&2
  case "$stage" in
    build)
      (cd "$REPO" && CARGO_PROFILE_RELEASE_LTO=thin CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
        cargo build --release --example librosa_bench --features full_no_plotting --target-dir "$TARGET_DIR")
      ;;
    signals)
      "$PY" -c "import sys; sys.path.insert(0, '$HERE'); import common; common.write_signals('$SIG')"
      ;;
    rust)
      "$PY" "$HERE/bench_py.py" --list --tsv $SEL > "$OUT/pairs.tsv"
      "${ONE[@]}" "$BIN" --signals "$SIG" --pairs "$OUT/pairs.tsv" >> "$OUT/rust_time.jsonl"
      ;;
    py)
      for lib in librosa ours; do
        "${ONE[@]}" "$PY" "$HERE/bench_py.py" --lib "$lib" --signals "$SIG" --threads 1 $SEL \
          --out "$OUT/${lib}_time.jsonl"
      done
      ;;
    mem)
      for lib in librosa ours; do
        "${ONE[@]}" "$PY" "$HERE/bench_py.py" --lib "$lib" --signals "$SIG" --threads 1 --mode mem $SEL \
          --out "$OUT/${lib}_mem.jsonl"
      done
      ;;
    rss)
      # Exact peak-RSS tracker (see rss_shim.c), preloaded into every probe.
      mkdir -p "$REPO/target/librosa_bench"
      cc -O2 -Wall -shared -fPIC -o "$REPO/target/librosa_bench/rss_shim.so" "$HERE/rss_shim.c"
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
        "$PY" "$HERE/rss_probe.py" --signals "$SIG" --jobs "$JOBS" --repeats "$REPEATS" \
        --shim "$REPO/target/librosa_bench/rss_shim.so" $SEL --out "$OUT/rss.jsonl"
      ;;
    default)
      "$PY" "$HERE/bench_py.py" --list --tsv --config-set default $SEL > "$OUT/pairs_default.tsv"
      "$BIN" --signals "$SIG" --pairs "$OUT/pairs_default.tsv" --no-alloc >> "$OUT/rust_default.jsonl"
      for lib in librosa ours; do
        "$PY" "$HERE/bench_py.py" --lib "$lib" --signals "$SIG" --threads default --config-set default $SEL \
          --out "$OUT/${lib}_default.jsonl"
      done
      ;;
    ties)
      # Rows whose separate librosa and ours runs are within 3 %: time both
      # again in one process, alternating (bench_py.py --mode pair). Needs
      # merged.json from a report stage.
      "$PY" -c "
import json
for r in json.load(open('$OUT/merged.json')):
    x = r.get('ratio_py')
    if x is not None and 0.97 <= x <= 1.03:
        print(r['case'], r['config'])
" | while read -r case cfg; do
        "${ONE[@]}" "$PY" "$HERE/bench_py.py" --mode pair --signals "$SIG" --threads 1 --cases "$case" \
          --configs "$cfg" --out "$OUT/ours_time.jsonl" --out-librosa "$OUT/librosa_time.jsonl"
      done
      # Native rows at 0.97 or above: five more processes each; the record
      # keeps the median of the five process medians.
      "$PY" -c "
import json
for r in json.load(open('$OUT/merged.json')):
    x = r.get('ratio_rust')
    if x is not None and x >= 0.97:
        print(r['case'], r['config'])
" > "$OUT/pairs_ties_rust.tsv"
      if [ -s "$OUT/pairs_ties_rust.tsv" ]; then
        for i in 1 2 3 4 5; do
          "${ONE[@]}" "$BIN" --signals "$SIG" --pairs "$OUT/pairs_ties_rust.tsv"
        done | "$PY" -c "
import json, statistics, sys
runs = {}
for line in sys.stdin:
    if line.startswith('{'):
        r = json.loads(line)
        runs.setdefault((r['case'], r['config']), []).append(r)
for recs in runs.values():
    recs.sort(key=lambda r: r['median_ms'])
    rec = dict(recs[len(recs) // 2])
    rec['process_medians_ms'] = [r['median_ms'] for r in recs]
    print(json.dumps(rec))
" >> "$OUT/rust_time.jsonl"
      fi
      # RSS rows where ours' median is above librosa's: ten more fresh
      # processes per library (the table uses every repeat).
      cases=$("$PY" -c "
import json
rows = json.load(open('$OUT/merged.json'))
print(','.join(sorted({r['case'] for r in rows if r.get('ours_rss_kb') is not None
                       and r.get('librosa_rss_kb') is not None and r['ours_rss_kb'] > r['librosa_rss_kb']})))
")
      if [ -n "$cases" ]; then
        mkdir -p "$REPO/target/librosa_bench"
        cc -O2 -Wall -shared -fPIC -o "$REPO/target/librosa_bench/rss_shim.so" "$HERE/rss_shim.c"
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
          "$PY" "$HERE/rss_probe.py" --signals "$SIG" --jobs "$JOBS" --repeats 10 \
          --shim "$REPO/target/librosa_bench/rss_shim.so" --cases "$cases" --out "$OUT/rss.jsonl"
      fi
      ;;
    report)
      "$PY" "$HERE/report.py" "$OUT"
      ;;
    *)
      echo "unknown stage $stage" >&2
      exit 2
      ;;
  esac
done
