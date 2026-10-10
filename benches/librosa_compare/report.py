"""Turn the JSON lines of a benchmark run into Markdown tables.

    report.py OUT_DIR [--canonical 22050_10s_mono_f32]

Writes, into OUT_DIR:

* ``table_full.md``: every case and configuration;
* ``table_main.md``: one row per case at the canonical configuration;
* ``slower.md``: every row where audio_samples is slower or uses more memory;
* ``aggregates.md``: geometric-mean time ratios by input property;
* ``rss.md``, ``default_threads.md``;
* ``merged.json``: the merged records.

Time ratio = audio_samples / librosa (below 1: audio_samples is faster). The
like-for-like ratio is the one through Python ("py"); "rust" is the native
call. Winner: ratio below 0.95 or above 1.05 (time), or a peak-memory ratio
outside the same band; otherwise "tie".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--canonical", default="22050_10s_mono_f32")
args = ap.parse_args()

BAND = 0.05


def load(name):
    path = os.path.join(args.out, name)
    recs = {}
    if not os.path.exists(path):
        return recs
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                r = json.loads(line)
                recs[(r["case"], r["config"], r.get("lib"))] = r  # last one wins
    return recs


def by_key(recs, lib):
    return {(c, cfg): r for (c, cfg, lb), r in recs.items() if lb == lib}


rust = by_key(load("rust_time.jsonl"), "rust")
lib_t = by_key(load("librosa_time.jsonl"), "librosa")
our_t = by_key(load("ours_time.jsonl"), "ours")
lib_m = by_key(load("librosa_mem.jsonl"), "librosa")
our_m = by_key(load("ours_mem.jsonl"), "ours")
rss = load("rss.jsonl")
lib_rss = by_key(rss, "librosa")
our_rss = by_key(rss, "ours")
rust_d = by_key(load("rust_default.jsonl"), "rust")
lib_d = by_key(load("librosa_default.jsonl"), "librosa")
our_d = by_key(load("ours_default.jsonl"), "ours")

try:
    from cases import all_cases

    order = {c.name: i for i, c in enumerate(all_cases())}
    func = {c.name: (c.func or c.name) for c in all_cases()}
    group = {c.name: c.group for c in all_cases()}
    notes = {c.name: c.note for c in all_cases() if c.note}
except Exception:  # report without the registry (e.g. librosa not importable)
    order, func, group, notes = {}, {}, {}, {}

keys = sorted(set(lib_t) | set(our_t) | set(rust),
              key=lambda k: (order.get(k[0], 1e9), k[0], k[1]))


def med(d, k):
    r = d.get(k)
    return None if r is None or "median_ms" not in r else r["median_ms"]


def spread(d, k):
    r = d.get(k)
    if not r or "median_ms" not in r or r["median_ms"] == 0:
        return None
    return (r["p75_ms"] - r["p25_ms"]) / r["median_ms"]


def fms(x):
    if x is None:
        return "–"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    if x >= 1:
        return f"{x:.2f}"
    if x >= 0.01:
        return f"{x:.3f}"
    return f"{x:.2g}"


def fbytes(b):
    if b is None:
        return "–"
    b = float(b)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(b) < 1024 or unit == "GB":
            return f"{b:.0f} {unit}" if unit == "B" else f"{b:.1f} {unit}"
        b /= 1024


def fratio(x):
    return "–" if x is None else f"{x:.2f}"


def ratio(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def winner(r):
    if r is None:
        return "–"
    if r < 1 - BAND:
        return "ours"
    if r > 1 + BAND:
        return "librosa"
    return "tie"


rows = []
for k in keys:
    case, cfg = k
    lt, ot, rt = med(lib_t, k), med(our_t, k), med(rust, k)
    lm, om = lib_m.get(k, {}), our_m.get(k, {})
    rr = rust.get(k, {})
    row = {
        "case": case,
        "func": func.get(case, case),
        "group": group.get(case, ""),
        "config": cfg,
        "librosa_ms": lt,
        "ours_py_ms": ot,
        "rust_ms": rt,
        "librosa_spread": spread(lib_t, k),
        "ours_py_spread": spread(our_t, k),
        "rust_spread": spread(rust, k),
        "ratio_py": ratio(ot, lt),
        "ratio_rust": ratio(rt, lt),
        "binding_overhead_ms": None if ot is None or rt is None else ot - rt,
        "librosa_peak": lm.get("mr_peak"),
        "ours_peak": om.get("mr_peak"),
        "rust_peak": rr.get("peak_bytes"),
        "librosa_allocs": lm.get("mr_allocs"),
        "ours_allocs": om.get("mr_allocs"),
        "rust_allocs": rr.get("allocs"),
        "rust_alloc_bytes": rr.get("alloc_bytes"),
        "librosa_tracemalloc": lm.get("tm_peak"),
        "ours_tracemalloc": om.get("tm_peak"),
        "librosa_rss_kb": lib_rss.get(k, {}).get("rss_delta_kb"),
        "ours_rss_kb": our_rss.get(k, {}).get("rss_delta_kb"),
        "librosa_default_ms": med(lib_d, k),
        "ours_default_ms": med(our_d, k),
        "rust_default_ms": med(rust_d, k),
        "out_librosa": lib_t.get(k, {}).get("out"),
        "out_ours": our_t.get(k, {}).get("out"),
        "error": "; ".join(
            f"{n}: {d[k]['error']}" for n, d in (("librosa", lib_t), ("ours", our_t)) if k in d and "error" in d[k]
        ) or None,
    }
    row["mem_ratio"] = ratio(row["ours_peak"], row["librosa_peak"])
    row["time_winner"] = winner(row["ratio_py"])
    row["mem_winner"] = winner(row["mem_ratio"])
    rows.append(row)

with open(os.path.join(args.out, "merged.json"), "w") as f:
    json.dump(rows, f, indent=1)

HEADER = ("| function | input | librosa ms | ours (py) ms | ours (rust) ms | ratio py (rust) "
          "| librosa peak | ours peak (py / rust) | allocs librosa / ours py / rust | winner time / mem |\n"
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")


def table_row(r):
    def bold(s, cond):
        return f"**{s}**" if cond else s

    rp = r["ratio_py"]
    mr = r["mem_ratio"]
    allocs = " / ".join(str(r[x]) if r[x] is not None else "–"
                        for x in ("librosa_allocs", "ours_allocs", "rust_allocs"))
    return (
        f"| {r['func']}{' [' + r['case'] + ']' if r['case'] != r['func'] else ''} | {r['config']} "
        f"| {fms(r['librosa_ms'])} | {fms(r['ours_py_ms'])} | {fms(r['rust_ms'])} "
        f"| {bold(fratio(rp), rp is not None and rp > 1 + BAND)} ({fratio(r['ratio_rust'])}) "
        f"| {fbytes(r['librosa_peak'])} "
        f"| {bold(fbytes(r['ours_peak']), mr is not None and mr > 1 + BAND)} / {fbytes(r['rust_peak'])} "
        f"| {allocs} | {r['time_winner']} / {r['mem_winner']} |\n"
    )


def write_table(name, title, rs, intro=""):
    with open(os.path.join(args.out, name), "w") as f:
        f.write(f"# {title}\n\n{intro}")
        f.write(HEADER)
        for r in rs:
            f.write(table_row(r))
        errs = [r for r in rs if r["error"]]
        if errs:
            f.write("\nErrors:\n\n")
            for r in errs:
                f.write(f"- {r['case']} {r['config']}: {r['error']}\n")


INTRO = (
    "Times are medians over repeated calls after warm-up, one thread, pinned to one core. "
    "Ratio = audio_samples / librosa; the first is through the Python bindings (like for like), "
    "the one in brackets is the native Rust call. Peak memory is memray's peak heap growth during "
    "one call (numpy and Rust allocations alike) for the two Python paths, and the counting "
    "allocator's peak live bytes for the native call. Allocation counts: memray (librosa, ours "
    "through Python) and the counting allocator (rust). Bold: audio_samples slower or larger by "
    "more than 5%.\n\n"
)

write_table("table_full.md", "All cases and inputs", rows, INTRO)

# one row per case at the canonical configuration (or the closest one it has)
by_case = defaultdict(list)
for r in rows:
    by_case[r["case"]].append(r)


def pick(rs):
    prefs = [args.canonical, args.canonical.replace("f32", "f64"), "22050_10s_mono_f32",
             "vec_f32", "seq_f32", "const_f64", "vec_f64", "seq_f64"]
    for p in prefs:
        for r in rs:
            if r["config"] == p:
                return r
    return rs[0]


main_rows = [pick(rs) for rs in by_case.values()]
write_table("table_main.md", f"One row per case ({args.canonical} where the case has it)", main_rows, INTRO)

slower = [r for r in rows if (r["ratio_py"] or 0) > 1 + BAND or (r["ratio_rust"] or 0) > 1 + BAND
          or (r["mem_ratio"] or 0) > 1 + BAND]
write_table("slower.md", "Rows where audio_samples is slower or larger", slower, INTRO)


# aggregates
def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def prop(cfg):
    p = cfg.split("_")
    if len(p) == 4:
        return {"rate": p[0], "duration": p[1], "layout": p[2], "dtype": p[3]}
    return {"kind": p[0], "dtype": p[1]}


with open(os.path.join(args.out, "aggregates.md"), "w") as f:
    f.write("# Geometric-mean time ratios (audio_samples / librosa)\n\n")
    f.write("| grouping | value | n | py | rust | memory (py) |\n|---|---|---:|---:|---:|---:|\n")
    for dim in ("duration", "rate", "layout", "dtype", "kind"):
        groups = defaultdict(list)
        for r in rows:
            v = prop(r["config"]).get(dim)
            if v:
                groups[v].append(r)
        for v, rs in sorted(groups.items()):
            f.write(f"| {dim} | {v} | {len(rs)} | {fratio(gmean(r['ratio_py'] for r in rs))} "
                    f"| {fratio(gmean(r['ratio_rust'] for r in rs))} "
                    f"| {fratio(gmean(r['mem_ratio'] for r in rs))} |\n")
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    for v, rs in sorted(groups.items()):
        f.write(f"| group | {v} | {len(rs)} | {fratio(gmean(r['ratio_py'] for r in rs))} "
                f"| {fratio(gmean(r['ratio_rust'] for r in rs))} "
                f"| {fratio(gmean(r['mem_ratio'] for r in rs))} |\n")
    tw = defaultdict(int)
    mw = defaultdict(int)
    for r in rows:
        tw[r["time_winner"]] += 1
        mw[r["mem_winner"]] += 1
    f.write(f"\nTime winner over all rows: {dict(tw)}\n\nMemory winner over all rows: {dict(mw)}\n")
    tw = defaultdict(int)
    for r in main_rows:
        tw[r["time_winner"]] += 1
    f.write(f"\nTime winner, one row per case: {dict(tw)}\n")

with open(os.path.join(args.out, "rss.md"), "w") as f:
    f.write("# Peak RSS growth per call (fresh process, after a warm-up on 1 s of input)\n\n")
    f.write("| function | input | librosa | ours (py) | ratio | tracemalloc librosa / ours |\n"
            "|---|---|---:|---:|---:|---:|\n")
    for r in rows:
        if r["librosa_rss_kb"] is None and r["ours_rss_kb"] is None:
            continue
        lk, ok = r["librosa_rss_kb"], r["ours_rss_kb"]
        f.write(f"| {r['func']} [{r['case']}] | {r['config']} | {fbytes(lk * 1024 if lk is not None else None)} "
                f"| {fbytes(ok * 1024 if ok is not None else None)} | {fratio(ratio(ok, lk) if lk else None)} "
                f"| {fbytes(r['librosa_tracemalloc'])} / {fbytes(r['ours_tracemalloc'])} |\n")

with open(os.path.join(args.out, "default_threads.md"), "w") as f:
    f.write("# Default threading (no thread limits, not pinned)\n\n")
    f.write("| function | input | librosa ms (1 thread) | librosa ms (default) | ours py (1) | ours py (default) "
            "| rust (1) | rust (default) |\n|---|---|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        if r["librosa_default_ms"] is None and r["ours_default_ms"] is None:
            continue
        f.write(f"| {r['func']} [{r['case']}] | {r['config']} | {fms(r['librosa_ms'])} | {fms(r['librosa_default_ms'])} "
                f"| {fms(r['ours_py_ms'])} | {fms(r['ours_default_ms'])} | {fms(r['rust_ms'])} "
                f"| {fms(r['rust_default_ms'])} |\n")

print(f"{len(rows)} rows; {len(slower)} with audio_samples slower or larger; written to {args.out}")
