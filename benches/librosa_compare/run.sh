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
# Not part of the default list (file reading, start-up and install costs; see
# the header of each script):
#   load      load_bench.py: librosa.load / librosa.stream against the read path:
#             time pinned, memray peak, peak RSS growth in fresh processes
#   startup   startup_bench.py: import time, first-call latency, process start
#             to first MFCC, pinned and default threading, warm and cold cache
#   footprint footprint.py: wheel size, fresh environments, dependency count
#             (needs network)
#   guide     guide_numbers.py, guide_tables.py: every figure quoted in the librosa guide,
#             and the tables its page includes
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
    load)
      "$PY" "$HERE/load_bench.py" --write-files
      for lib in librosa ours; do
        "${ONE[@]}" "$PY" -W ignore "$HERE/load_bench.py" --lib "$lib" --out "$OUT/load_time.jsonl"
        "${ONE[@]}" "$PY" -W ignore "$HERE/load_bench.py" --lib "$lib" --mode mem --out "$OUT/load_mem.jsonl"
      done
      # Rows at 0.97 or above: both libraries again in one process, alternating.
      "$PY" -c "
import json
t = {}
for line in open('$OUT/load_time.jsonl'):
    r = json.loads(line)
    if 'median_ms' in r:
        t[(r['row'], r['input'], r['lib'])] = r['median_ms']
for (row, inp, lib), v in t.items():
    if lib == 'ours' and row != 'defaults' and (row, inp, 'librosa') in t and v / t[(row, inp, 'librosa')] >= 0.97:
        print(row, inp)
" | while read -r row inp; do
        "${ONE[@]}" "$PY" -W ignore "$HERE/load_bench.py" --mode pair --rows "$row" --inputs "$inp" --out "$OUT/load_time.jsonl"
      done
      mkdir -p "$REPO/target/librosa_bench"
      cc -O2 -Wall -shared -fPIC -o "$REPO/target/librosa_bench/rss_shim.so" "$HERE/rss_shim.c"
      "$PY" -W ignore "$HERE/load_bench.py" --mode rss-run --repeats "$REPEATS" --jobs "$JOBS" \
        --shim "$REPO/target/librosa_bench/rss_shim.so" --inputs wav16_60s,flac16_60s,wavf32_60s --out "$OUT/load_rss.jsonl"
      ;;
    startup)
      "$PY" "$HERE/load_bench.py" --write-files
      "$PY" "$HERE/startup_bench.py" --core "$CORE" --out "$OUT/startup.jsonl"
      "$PY" "$HERE/startup_bench.py" --core "" --stages import,mfcc,batch --out "$OUT/startup_default.jsonl"
      ;;
    footprint)
      "$PY" "$HERE/footprint.py" --python-repo "$REPO/../audio_samples_python" --out "$OUT/footprint.json"
      ;;
    guide)
      "$PY" "$HERE/guide_numbers.py" "$OUT"
      "$PY" "$HERE/guide_tables.py" "$OUT" "$REPO/../audio_samples_python/docs/source/_guide_tables"
      ;;
    *)
      echo "unknown stage $stage" >&2
      exit 2
      ;;
  esac
done
