#!/usr/bin/env python3
"""Harvest Criterion bench data into a single tidy CSV.

Walks every `target/criterion_<measurement>/` tree under the given target
directory, reads each bench's `new/raw.csv` and `new/benchmark.json`, parses
the parameter string, and writes three CSVs to the output directory:

  - raw_unified.csv     one row per Criterion sample
  - benchmarks_meta.csv one row per (group, function, value) triple
  - estimates.csv       one row per (group, function, value) triple

Schema is documented in bench_suite/METHODOLOGY.md §9.

Usage:
    python3 harvest.py <target-dir> <output-dir> [--run-id RUN_ID]

Stdlib only — no third-party deps required to harvest.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

CATALOG_ID_RE = re.compile(r"^([A-Z][a-z]+-\d{3})_")

# Map raw.csv `unit` values to canonical measurement labels.
# WallTime emits "ns"; criterion-perf-events emits various per-event labels.
UNIT_TO_MEASUREMENT: dict[str, str] = {
    "ns":              "walltime",
    "us":              "walltime",
    "ms":              "walltime",
    "s":               "walltime",
    "instructions":    "instructions",
    "cycles":          "cycles",
    "cpu-cycles":      "cycles",
    "cache-misses":    "cache_misses",
    "cache-references":"cache_references",
    "branches":        "branches",
    "branch-misses":   "branch_misses",
}


def parse_param_string(value: str) -> tuple[dict[str, str], dict[str, str]]:
    """Split a `k=v,k=v,...` parameter string into known + extra columns.

    Returns (known, extra) where known contains the well-known columns
    (`len`, `dt`, `ch`) and extra is everything else.
    """
    known: dict[str, str] = {"len": "", "dt": "", "ch": ""}
    extra: dict[str, str] = {}
    if not value:
        return known, extra
    for part in value.split(","):
        if "=" not in part:
            continue
        k, _, v = part.partition("=")
        if k in known:
            known[k] = v
        else:
            extra[k] = v
    return known, extra


def measurement_from_dir(criterion_dir: Path) -> str:
    """Derive the measurement label from the criterion_<measurement>/ name."""
    name = criterion_dir.name  # e.g. "criterion_walltime"
    if name.startswith("criterion_"):
        return name[len("criterion_"):]
    return "walltime"   # fallback for plain "criterion/" (legacy)


def measurement_from_unit(unit: str) -> str:
    """Best-effort guess of the measurement from raw.csv's unit column.

    Used as a cross-check against the directory name. The directory name
    wins on conflict.
    """
    return UNIT_TO_MEASUREMENT.get(unit, unit)


def extract_catalog_id(function: str) -> str:
    """Pull `Stats-001` out of `Stats-001_peak`. Empty if not a catalog id."""
    m = CATALOG_ID_RE.match(function)
    return m.group(1) if m else ""


def walk_bench_dirs(target_dir: Path):
    """Yield every (criterion_root, benchmark.json, raw.csv) under target/.

    Looks for `target/criterion_*/` subtrees (per-measurement output dirs)
    AND the legacy `target/criterion/` tree.
    """
    candidates = []
    legacy = target_dir / "criterion"
    if legacy.is_dir():
        candidates.append(legacy)
    for entry in sorted(target_dir.glob("criterion_*")):
        if entry.is_dir():
            candidates.append(entry)
    for crit_root in candidates:
        for raw_csv in crit_root.rglob("new/raw.csv"):
            bench_json = raw_csv.parent / "benchmark.json"
            estimates_json = raw_csv.parent / "estimates.json"
            yield crit_root, bench_json, raw_csv, estimates_json


# ----------------------------------------------------------------------------
# Harvest
# ----------------------------------------------------------------------------

RAW_HEADER = [
    "run_id", "measurement",
    "group", "catalog_id", "function", "value",
    "len", "dt", "ch", "extra_params_json",
    "throughput_num", "throughput_type",
    "sample_measured_value", "unit", "iteration_count",
]

META_HEADER = [
    "run_id", "measurement",
    "group", "catalog_id", "function", "value",
    "len", "dt", "ch", "extra_params_json",
    "throughput_num", "throughput_type",
    "full_id", "directory_name",
]

EST_HEADER = [
    "run_id", "measurement",
    "group", "catalog_id", "function", "value",
    "len", "dt", "ch", "extra_params_json",
    "mean_point", "mean_ci_lo", "mean_ci_hi", "mean_std_err",
    "median_point", "median_ci_lo", "median_ci_hi", "median_std_err",
    "std_dev_point", "median_abs_dev_point",
    "slope_point",
]


def harvest(target_dir: Path, out_dir: Path, run_id: str) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path  = out_dir / "raw_unified.csv"
    meta_path = out_dir / "benchmarks_meta.csv"
    est_path  = out_dir / "estimates.csv"

    n_meta = 0
    n_raw  = 0

    with raw_path.open("w", newline="") as raw_f, \
         meta_path.open("w", newline="") as meta_f, \
         est_path.open("w", newline="") as est_f:
        raw_w  = csv.writer(raw_f)
        meta_w = csv.writer(meta_f)
        est_w  = csv.writer(est_f)
        raw_w.writerow(RAW_HEADER)
        meta_w.writerow(META_HEADER)
        est_w.writerow(EST_HEADER)

        for crit_root, bench_json, raw_csv, estimates_json in walk_bench_dirs(target_dir):
            measurement = measurement_from_dir(crit_root)

            # ---- benchmark.json ----
            try:
                with bench_json.open() as f:
                    bj = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"WARN: failed to read {bench_json}: {e}", file=sys.stderr)
                continue

            group_id    = bj.get("group_id", "")
            function_id = bj.get("function_id", "")
            value_str   = bj.get("value_str", "") or ""
            full_id     = bj.get("full_id", "")
            dir_name    = bj.get("directory_name", "")
            throughput_obj = bj.get("throughput")
            throughput_num, throughput_type = "", ""
            if isinstance(throughput_obj, dict) and throughput_obj:
                # e.g. {"Elements": 4096}, {"Bytes": 8192}
                ttype, tnum = next(iter(throughput_obj.items()))
                throughput_type = ttype
                throughput_num  = str(tnum)
            elif isinstance(throughput_obj, dict):
                # ElementsAndBytes variant — keep both
                throughput_type = "ElementsAndBytes"
                throughput_num  = json.dumps(throughput_obj)

            known, extra = parse_param_string(value_str)
            catalog_id = extract_catalog_id(function_id)
            extra_json = json.dumps(extra, separators=(",", ":")) if extra else ""

            meta_w.writerow([
                run_id, measurement,
                group_id, catalog_id, function_id, value_str,
                known["len"], known["dt"], known["ch"], extra_json,
                throughput_num, throughput_type,
                full_id, dir_name,
            ])
            n_meta += 1

            # ---- raw.csv ----
            try:
                with raw_csv.open() as f:
                    rdr = csv.DictReader(f)
                    for row in rdr:
                        # Cross-check directory measurement vs unit-derived one.
                        unit = row.get("unit", "")
                        m_unit = measurement_from_unit(unit)
                        # Directory wins, but warn on inconsistency once.
                        raw_w.writerow([
                            run_id, measurement,
                            row.get("group", ""),
                            catalog_id,
                            row.get("function", ""),
                            row.get("value", ""),
                            known["len"], known["dt"], known["ch"], extra_json,
                            row.get("throughput_num", ""),
                            row.get("throughput_type", ""),
                            row.get("sample_measured_value", ""),
                            unit,
                            row.get("iteration_count", ""),
                        ])
                        n_raw += 1
            except OSError as e:
                print(f"WARN: failed to read {raw_csv}: {e}", file=sys.stderr)

            # ---- estimates.json ----
            try:
                with estimates_json.open() as f:
                    ej = json.load(f)
            except (OSError, json.JSONDecodeError):
                ej = None
            if isinstance(ej, dict):
                def pt(k: str, default: str = "") -> str:
                    v = ej.get(k) or {}
                    p = v.get("point_estimate")
                    return "" if p is None else str(p)
                def ci(k: str, side: str, default: str = "") -> str:
                    v = ej.get(k) or {}
                    c = v.get("confidence_interval") or {}
                    p = c.get(f"{side}_bound")
                    return "" if p is None else str(p)
                def se(k: str) -> str:
                    v = ej.get(k) or {}
                    p = v.get("standard_error")
                    return "" if p is None else str(p)
                est_w.writerow([
                    run_id, measurement,
                    group_id, catalog_id, function_id, value_str,
                    known["len"], known["dt"], known["ch"], extra_json,
                    pt("mean"),         ci("mean", "lower"),   ci("mean", "upper"),   se("mean"),
                    pt("median"),       ci("median", "lower"), ci("median", "upper"), se("median"),
                    pt("std_dev"),      pt("median_abs_dev"),
                    pt("slope"),
                ])

    return n_meta, n_raw


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target_dir", type=Path, help="path containing criterion_<m>/ subtrees (usually `target`)")
    ap.add_argument("out_dir",    type=Path, help="where to write the unified CSVs")
    ap.add_argument("--run-id",   default="unknown", help="run identifier; written to every row")
    args = ap.parse_args()

    if not args.target_dir.is_dir():
        print(f"ERROR: target dir {args.target_dir} not found", file=sys.stderr)
        return 1

    n_meta, n_raw = harvest(args.target_dir, args.out_dir, args.run_id)
    print(f"wrote {args.out_dir}/raw_unified.csv ({n_raw} samples)")
    print(f"wrote {args.out_dir}/benchmarks_meta.csv ({n_meta} bench points)")
    print(f"wrote {args.out_dir}/estimates.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
