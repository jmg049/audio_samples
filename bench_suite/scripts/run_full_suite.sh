#!/usr/bin/env bash
# run_full_suite.sh — single-command full-coverage bench runner.
#
# Run this with NO arguments. You get:
#   - every catalog section (26)
#   - every measurement per section (walltime + 4 PMU = 5)
#   - every gated extra-coverage variant (simd for conv, parallel for psy)
#   - = 140 individual bench-target invocations
#   - a single .tar.gz at the end containing all results + run logs
#   - an end-of-run audit confirming whether the picture is complete
#
# Resumability: jobs whose results directory already exists AND has a
# non-empty raw_unified.csv are skipped. Crashed / partial runs are NOT
# considered done and are re-executed automatically. There is no way to
# "lose" data by re-running.
#
# Usage:
#   bench_suite/scripts/run_full_suite.sh             # do everything
#
# Surgical flags (only needed for tuning, not for completeness):
#   --sections LIST       comma-separated section short names
#   --measurements LIST   comma-separated measurement names
#   --skip-pmu            shorthand for --measurements walltime
#   --no-extras           skip the simd / parallel coverage variants
#   --force               re-run sections that already have complete results
#   --no-tar              skip the final tarball (just leave the dirs)
#   --dry-run             print the planned invocations and exit
#   --tag-prefix S        change the "suite" prefix in feature_set_tag
#   --bench-args "..."    extra args forwarded inside cargo bench
#   -h, --help            this help

set -uo pipefail
# NOTE: intentionally NOT using `-e`. Any single bench failure is logged and
# the loop continues so we capture as much data as possible from a long run.

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------

ALL_SECTIONS=(stats proc trans edit chan iir eq dr env onset pp beat dec pitch
              vad plot psy resamp conv gen comp det math iter fixed repr)
ALL_MEASUREMENTS=(walltime instructions cycles cache_misses branch_misses)

# Per-section Cargo features. Empty = no features required.
declare -A SECTION_FEATURES=(
    [stats]="statistics"          [proc]="processing"
    [trans]="transforms"          [edit]="editing"
    [chan]="channels"             [iir]="iir-filtering"
    [eq]="parametric-eq,iir-filtering"
    [dr]="dynamic-range"          [env]="envelopes"
    [onset]="onset-detection"     [pp]="peak-picking"
    [beat]="beat-tracking,onset-detection"
    [dec]="decomposition"         [pitch]="pitch-analysis"
    [vad]="vad"                   [plot]="plotting"
    [psy]="psychoacoustic"        [resamp]="resampling"
    [conv]=""                     [gen]="random-generation,channels"
    [comp]=""                     [det]="transforms"
    [math]=""                     [iter]="editing"
    [fixed]="fixed-size-audio"    [repr]=""
)

# Extra-coverage runs (one per measurement, in addition to the base run).
declare -A EXTRA_VARIANTS=(
    [conv]="simd"        # cfg-gates Conv-021..Conv-024 SIMD fast paths
    [psy]="parallel"     # cfg-gates the rayon path in PerceptualCodec::encode
)

# --------------------------------------------------------------------------
# Arg parsing (all flags optional — defaults capture everything)
# --------------------------------------------------------------------------

SECTIONS=("${ALL_SECTIONS[@]}")
MEASUREMENTS=("${ALL_MEASUREMENTS[@]}")
INCLUDE_EXTRAS=1
FORCE=0
DRY_RUN=0
TAG_PREFIX="suite"
DO_TAR=1
BENCH_ARGS=""

usage() {
    sed -n '/^# Usage:/,/^# *-h/p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sections)      IFS=',' read -ra SECTIONS <<< "$2"; shift 2 ;;
        --measurements)  IFS=',' read -ra MEASUREMENTS <<< "$2"; shift 2 ;;
        --skip-pmu)      MEASUREMENTS=(walltime); shift ;;
        --no-extras)     INCLUDE_EXTRAS=0; shift ;;
        --force)         FORCE=1; shift ;;
        --no-tar)        DO_TAR=0; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --tag-prefix)    TAG_PREFIX="$2"; shift 2 ;;
        --bench-args)    BENCH_ARGS="$2"; shift 2 ;;
        -h|--help)       usage ;;
        *) echo "ERROR: unknown flag: $1" >&2; echo "Try --help" >&2; exit 1 ;;
    esac
done

for s in "${SECTIONS[@]}"; do
    if [[ -z "${SECTION_FEATURES[$s]+x}" ]]; then
        echo "ERROR: unknown section: '$s'  valid: ${ALL_SECTIONS[*]}" >&2
        exit 1
    fi
done
for m in "${MEASUREMENTS[@]}"; do
    case "$m" in
        walltime|instructions|cycles|cache_misses|branch_misses) ;;
        *) echo "ERROR: unknown measurement: '$m'  valid: ${ALL_MEASUREMENTS[*]}" >&2; exit 1 ;;
    esac
done

# --------------------------------------------------------------------------
# Pre-flight: confirm the build environment is sane BEFORE we start a 140-job
# run. Catches the "host setup is broken, all 140 jobs would fail" case.
# --------------------------------------------------------------------------

preflight_fatal() { echo "PREFLIGHT FAIL: $*" >&2; exit 2; }
preflight_warn()  { echo "PREFLIGHT WARN: $*" >&2; }

command -v cargo    >/dev/null 2>&1 || preflight_fatal "cargo not on PATH"
command -v python3  >/dev/null 2>&1 || preflight_fatal "python3 not on PATH"
command -v tar      >/dev/null 2>&1 || preflight_fatal "tar not on PATH"
command -v taskset  >/dev/null 2>&1 || preflight_warn  "taskset missing — pinning will be skipped (bench numbers noisier)"
command -v git      >/dev/null 2>&1 || preflight_warn  "git missing — run_id will use 'unknown' commit"

[[ -x bench_suite/scripts/run_and_collect.sh ]] || preflight_fatal "bench_suite/scripts/run_and_collect.sh not found or not executable"
[[ -f bench_suite/scripts/harvest.py ]]          || preflight_fatal "bench_suite/scripts/harvest.py missing"

# PMU prereq — only matters if PMU measurements are in the plan. If the host
# isn't ready, auto-invoke host_setup.sh via sudo. The user runs this script
# once, gives sudo their password (or has NOPASSWD), and never thinks about
# host config again.
NEEDS_PMU=0
for m in "${MEASUREMENTS[@]}"; do [[ "$m" != "walltime" ]] && NEEDS_PMU=1; done

setup_host_if_needed() {
    local pep="?"
    [[ -r /proc/sys/kernel/perf_event_paranoid ]] && pep=$(cat /proc/sys/kernel/perf_event_paranoid)
    local gov="?"
    [[ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]] \
        && gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)

    local need_setup=0
    [[ $NEEDS_PMU -eq 1 ]] && [[ "$pep" != "?" ]] && [[ "$pep" -gt 1 ]] && need_setup=1
    [[ "$gov" != "?" && "$gov" != "performance" ]] && need_setup=1

    if [[ $need_setup -eq 0 ]]; then
        echo "Pre-flight: host already configured (perf_event_paranoid=$pep, governor=$gov)."
        return 0
    fi

    echo "Pre-flight: host needs configuration (perf_event_paranoid=$pep, governor=$gov)."
    echo "Pre-flight: auto-invoking bench_suite/scripts/host_setup.sh via sudo …"
    echo "            (this needs root once; subsequent runs will skip this step.)"

    if [[ $EUID -eq 0 ]]; then
        bash bench_suite/scripts/host_setup.sh
    elif command -v sudo >/dev/null 2>&1; then
        if ! sudo -n true 2>/dev/null; then
            echo "Pre-flight: sudo requires a password — you'll be prompted now."
        fi
        sudo bash bench_suite/scripts/host_setup.sh
    else
        preflight_warn "no sudo available — cannot run host_setup.sh automatically."
        preflight_warn "  Run manually: sudo bash bench_suite/scripts/host_setup.sh"
        return 1
    fi
}

setup_host_if_needed || preflight_warn "host setup didn't complete cleanly; PMU jobs may fail."

# --------------------------------------------------------------------------
# Suite-run identification and logging (created BEFORE prebuild so prebuild
# can log there).
# --------------------------------------------------------------------------

START_EPOCH=$(date +%s)
SUITE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short HEAD 2>/dev/null || echo unknown)_${TAG_PREFIX}"
SUITE_LOG_DIR="bench_suite/results/_suite_logs/${SUITE_RUN_ID}"
mkdir -p "$SUITE_LOG_DIR"
SUMMARY_FILE="$SUITE_LOG_DIR/summary.tsv"
COVERAGE_FILE="$SUITE_LOG_DIR/coverage.json"
echo -e "section\tmeasurement\tvariant\tfeature_set_tag\tstatus\telapsed_s\trun_id" > "$SUMMARY_FILE"

# Pre-build every bench target up-front so the first job in the loop doesn't
# also pay the audio_samples + bench_suite_common compile cost (~3–5 min cold,
# plus another ~30s when PMU deps come in). After this `cargo build --benches`
# completes, each per-job `cargo bench` invocation is a near-instant
# incremental no-op — it just runs the already-built binary.
#
# Union feature set:
#   - "full"          = full_no_plotting + plotting (covers all 26 sections)
#   - "perf_events"   = PMU bench targets (140 jobs total)
#   - "simd"          = cfg-gated Conv-021..Conv-024
#   - "parallel"      = rayon path in PerceptualCodec::encode (psy extra)
#
# This is the ONE place where build cost is paid in the entire fire-and-forget
# run. Time it explicitly so the operator can see the build wasn't free.

PREBUILD_FEATURES="full,perf_events,simd,parallel"
PREBUILD_LOG="$SUITE_LOG_DIR/prebuild.log"
mkdir -p "$(dirname "$PREBUILD_LOG")" 2>/dev/null || true

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Pre-flight: (dry-run) would cargo build --release --benches --features '$PREBUILD_FEATURES'"
    PREBUILD_ELAPSED=0
else

echo "Pre-flight: cargo build --release --benches --features '$PREBUILD_FEATURES'"
echo "            (this is the only compile step; ~3–8 min cold, then every"
echo "             bench in the loop runs from cache.)"

PREBUILD_START=$(date +%s)
# RUSTC_BOOTSTRAP=1 because perfcnt → nom 4.2.3 mis-detects nightly on stable
# rustc. Documented in bench_suite/README.md.
if RUSTC_BOOTSTRAP=1 RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}" \
        cargo build --release --benches --features "$PREBUILD_FEATURES" \
        > "$PREBUILD_LOG" 2>&1; then
    PREBUILD_ELAPSED=$(( $(date +%s) - PREBUILD_START ))
    echo "Pre-flight: build succeeded in ${PREBUILD_ELAPSED}s"
else
    PREBUILD_ELAPSED=$(( $(date +%s) - PREBUILD_START ))
    preflight_warn "Pre-flight cargo build FAILED after ${PREBUILD_ELAPSED}s. See $PREBUILD_LOG"
    preflight_warn "Aborting — running the bench loop now would mean every job repeats this failure."
    preflight_warn "Fix the build (or invoke with a narrower --sections / --measurements) and re-run."
    exit 3
fi
fi  # end !DRY_RUN guard around prebuild

# --------------------------------------------------------------------------
# Planning — enumerate every (section, measurement, variant) triple
# --------------------------------------------------------------------------

declare -a JOBS=()

for s in "${SECTIONS[@]}"; do
    for m in "${MEASUREMENTS[@]}"; do
        JOBS+=("${s}|${m}|")
    done
done

if [[ $INCLUDE_EXTRAS -eq 1 ]]; then
    for s in "${!EXTRA_VARIANTS[@]}"; do
        if [[ " ${SECTIONS[*]} " == *" ${s} "* ]]; then
            for variant in ${EXTRA_VARIANTS[$s]}; do
                for m in "${MEASUREMENTS[@]}"; do
                    JOBS+=("${s}|${m}|${variant}")
                done
            done
        fi
    done
fi

N_JOBS=${#JOBS[@]}

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

PMU_COUNT=$((N_JOBS - ${#SECTIONS[@]}))
[[ "${MEASUREMENTS[*]}" == "walltime" ]] && PMU_COUNT=0
echo "============================================================"
echo " audio_samples bench_suite — full sweep"
echo "------------------------------------------------------------"
echo " suite_run_id:    $SUITE_RUN_ID"
echo " jobs to run:     $N_JOBS"
echo "   sections:      ${#SECTIONS[@]} (${SECTIONS[*]})"
echo "   measurements:  ${#MEASUREMENTS[@]} (${MEASUREMENTS[*]})"
echo "   extras:        $([ $INCLUDE_EXTRAS -eq 1 ] && echo "yes — conv-simd, psy-parallel" || echo "no")"
echo " mode:            $([ $DRY_RUN -eq 1 ] && echo "DRY RUN" || echo "live")"
echo " force re-run:    $([ $FORCE -eq 1 ] && echo "yes" || echo "no")"
echo " auto-tar at end: $([ $DO_TAR -eq 1 ] && echo "yes" || echo "no")"
echo " summary file:    $SUMMARY_FILE"
echo " coverage audit:  $COVERAGE_FILE"
echo "============================================================"
echo

# --------------------------------------------------------------------------
# Skip-existing predicate: only skip if the previous run looks complete.
# Definition of complete: results dir exists AND contains raw_unified.csv
# with at least 2 lines (header + ≥ 1 data row).
# --------------------------------------------------------------------------

is_complete() {
    local dir="$1"
    [[ -d "$dir" ]] || return 1
    local csv="$dir/raw_unified.csv"
    [[ -f "$csv" ]] || return 1
    local lines
    lines=$(wc -l < "$csv" 2>/dev/null || echo 0)
    [[ "$lines" -ge 2 ]] || return 1
    return 0
}

# --------------------------------------------------------------------------
# Run loop
# --------------------------------------------------------------------------

N_PASS=0
N_FAIL=0
N_SKIP=0
NEW_DIRS=()
declare -a FAILED_TAGS=()

for idx in "${!JOBS[@]}"; do
    IFS='|' read -r section measurement variant <<< "${JOBS[$idx]}"
    pos=$((idx + 1))

    # Compose features list
    section_feats="${SECTION_FEATURES[$section]}"
    feats_list=()
    [[ -n "$section_feats" ]] && feats_list+=("$section_feats")
    [[ -n "$variant" ]]       && feats_list+=("$variant")
    [[ "$measurement" != "walltime" ]] && feats_list+=("perf_events")
    features="$(IFS=,; echo "${feats_list[*]}")"

    variant_tag=""
    [[ -n "$variant" ]] && variant_tag="_${variant}"
    feature_set_tag="${TAG_PREFIX}_${section}_${measurement}${variant_tag}"
    bench_target="bench_${section}_${measurement}"

    # Resume: skip only if a COMPLETE results dir already exists.
    if [[ $FORCE -eq 0 ]]; then
        # Find newest matching dir
        existing_dir=$(find bench_suite/results -maxdepth 1 -type d -name "*_${feature_set_tag}" 2>/dev/null \
                       | sort -r | head -1)
        if [[ -n "$existing_dir" ]] && is_complete "$existing_dir"; then
            printf "[%3d/%-3d] %-7s %-15s %-10s SKIP — complete: %s\n" \
                "$pos" "$N_JOBS" "$section" "$measurement" "${variant:-base}" "$(basename "$existing_dir")"
            echo -e "${section}\t${measurement}\t${variant}\t${feature_set_tag}\tskip\t0\t$(basename "$existing_dir")" \
                >> "$SUMMARY_FILE"
            N_SKIP=$((N_SKIP + 1))
            continue
        elif [[ -n "$existing_dir" ]]; then
            # Stale / partial directory exists — log it but proceed.
            printf "          INCOMPLETE prior dir detected (%s); will re-run.\n" "$(basename "$existing_dir")"
        fi
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        printf "[%3d/%-3d] %-7s %-15s %-10s DRY  features=%s\n" \
            "$pos" "$N_JOBS" "$section" "$measurement" "${variant:-base}" "${features:-(none)}"
        continue
    fi

    printf "[%3d/%-3d] %-7s %-15s %-10s RUN  features=%s\n" \
        "$pos" "$N_JOBS" "$section" "$measurement" "${variant:-base}" "${features:-(none)}"

    job_start=$(date +%s)
    job_log="$SUITE_LOG_DIR/${feature_set_tag}.log"

    if [[ -n "$BENCH_ARGS" ]]; then
        # shellcheck disable=SC2086
        bash bench_suite/scripts/run_and_collect.sh \
            "$feature_set_tag" "$features" "$bench_target" \
            -- $BENCH_ARGS \
            > "$job_log" 2>&1
        rc=$?
    else
        bash bench_suite/scripts/run_and_collect.sh \
            "$feature_set_tag" "$features" "$bench_target" \
            > "$job_log" 2>&1
        rc=$?
    fi

    job_end=$(date +%s)
    job_elapsed=$((job_end - job_start))

    run_dir=$(find bench_suite/results -maxdepth 1 -type d -name "*_${feature_set_tag}" -printf "%T@ %p\n" 2>/dev/null \
              | sort -nr | head -1 | awk '{print $2}')
    run_id_name="$(basename "${run_dir:-unknown}")"

    if [[ $rc -eq 0 ]] && is_complete "$run_dir"; then
        printf "          PASS  elapsed=%ds  -> %s\n" "$job_elapsed" "$run_id_name"
        N_PASS=$((N_PASS + 1))
        [[ -n "$run_dir" ]] && NEW_DIRS+=("$run_dir")
        echo -e "${section}\t${measurement}\t${variant}\t${feature_set_tag}\tpass\t${job_elapsed}\t${run_id_name}" >> "$SUMMARY_FILE"
    else
        printf "          FAIL  elapsed=%ds  exit=%d  log=%s\n" "$job_elapsed" "$rc" "$job_log"
        N_FAIL=$((N_FAIL + 1))
        FAILED_TAGS+=("$feature_set_tag")
        echo -e "${section}\t${measurement}\t${variant}\t${feature_set_tag}\tfail\t${job_elapsed}\t${run_id_name}" >> "$SUMMARY_FILE"
    fi
done

# --------------------------------------------------------------------------
# Coverage audit — write a JSON manifest of every expected job → actual status
# --------------------------------------------------------------------------

python3 - "$SUMMARY_FILE" "$COVERAGE_FILE" "$N_JOBS" "$N_PASS" "$N_FAIL" "$N_SKIP" <<'PYEOF'
import csv, json, sys
summary_path, coverage_path, n_jobs, n_pass, n_fail, n_skip = sys.argv[1:7]
rows = []
with open(summary_path) as f:
    rdr = csv.DictReader(f, delimiter="\t")
    rows = list(rdr)
out = {
    "n_jobs":       int(n_jobs),
    "n_pass":       int(n_pass),
    "n_fail":       int(n_fail),
    "n_skip":       int(n_skip),
    "complete":     int(n_fail) == 0,
    "jobs":         rows,
}
with open(coverage_path, "w") as f:
    json.dump(out, f, indent=2)
PYEOF

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

END_EPOCH=$(date +%s)
TOTAL_ELAPSED=$((END_EPOCH - START_EPOCH))
TOTAL_H=$((TOTAL_ELAPSED / 3600))
TOTAL_M=$(((TOTAL_ELAPSED % 3600) / 60))
TOTAL_S=$((TOTAL_ELAPSED % 60))

echo
echo "============================================================"
if [[ $N_FAIL -eq 0 ]]; then
    echo " ✓ SUITE COMPLETE — every requested job has data."
else
    echo " ✗ SUITE PARTIAL  — $N_FAIL jobs FAILED (see logs)."
fi
echo "------------------------------------------------------------"
printf " pass:   %3d\n" "$N_PASS"
printf " fail:   %3d\n" "$N_FAIL"
printf " skip:   %3d  (already complete from prior run)\n" "$N_SKIP"
printf " total:  %3d\n" "$N_JOBS"
printf " wall:   %dh %dm %ds\n" "$TOTAL_H" "$TOTAL_M" "$TOTAL_S"
echo " summary: $SUMMARY_FILE"
echo " audit:   $COVERAGE_FILE"
echo "============================================================"

if [[ $N_FAIL -gt 0 ]]; then
    echo
    echo "FAILED jobs (re-run with --force --sections X --measurements Y):"
    for t in "${FAILED_TAGS[@]}"; do
        echo "  - $t   log: $SUITE_LOG_DIR/${t}.log"
    done
fi

# --------------------------------------------------------------------------
# Auto-tar
# --------------------------------------------------------------------------

if [[ $DO_TAR -eq 1 && $DRY_RUN -eq 0 ]]; then
    TARBALL="bench_suite/results/${SUITE_RUN_ID}.tar.gz"
    echo
    if [[ ${#NEW_DIRS[@]} -eq 0 ]] && [[ $N_SKIP -eq $N_JOBS ]]; then
        # Nothing actually ran — everything was skipped. Skip the tar; the
        # data from the prior run is presumably already tarred.
        echo "No new data produced (all jobs were skipped). Not creating a tarball."
    else
        echo "Packaging results → $TARBALL"
        # Include: every NEW_DIR plus the suite log directory.
        # If --force was used we may have many NEW_DIRs; if nothing was
        # skipped we have everything; either way this captures what this
        # invocation produced.
        if [[ ${#NEW_DIRS[@]} -gt 0 ]]; then
            tar -czf "$TARBALL" "${NEW_DIRS[@]}" "$SUITE_LOG_DIR"
        else
            tar -czf "$TARBALL" "$SUITE_LOG_DIR"
        fi
        echo "  $(du -h "$TARBALL" | cut -f1)  $TARBALL"
        echo
        echo "scp this single file off the box:"
        echo "  scp $(hostname):$REPO_ROOT/$TARBALL ./"
    fi
fi

echo
if [[ $N_FAIL -eq 0 ]]; then
    echo "All requested benchmark data has been captured."
else
    echo "WARNING: $N_FAIL job(s) failed. See $COVERAGE_FILE for details."
fi

exit $([ $N_FAIL -eq 0 ] && echo 0 || echo 1)
