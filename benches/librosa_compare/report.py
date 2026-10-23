"""Turn the JSON lines of a benchmark run into Markdown tables.

    report.py OUT_DIR [--canonical 22050_10s_mono_f32]

Writes, into OUT_DIR:

* ``table_full.md``: every case and configuration;
* ``table_main.md``: one row per case at the canonical configuration;
* ``slower.md``: every row where audio_samples is slower or uses more memory by more than 5%;
* ``not_faster.md``: every row with any ratio above 1;
* ``before_after.md`` (with ``--baseline``): one row per case, before and after;
* ``aggregates.md``: geometric-mean time ratios by input property;
* ``rss.md`` (median and range over the fresh-process repeats), ``default_threads.md``;
* ``comparison.md``: per measure, the rows where audio_samples is below, equal
  to or above librosa, and every row where it is not strictly below;
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
ap.add_argument("--baseline", default=None, help="an earlier run's directory (with merged.json): adds before/after tables")
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
def load_rss(name):
    """Every repeat of the RSS probe: {(case, config, lib): [rss_delta_kb, ...]}."""
    path = os.path.join(args.out, name)
    out = defaultdict(list)
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("{"):
                    r = json.loads(line)
                    if "rss_delta_kb" in r:
                        out[(r["case"], r["config"], r.get("lib"))].append(r["rss_delta_kb"])
    return out


def median_of(xs):
    xs = sorted(xs)
    n = len(xs)
    return None if n == 0 else (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2)


rss_all = load_rss("rss.jsonl")
lib_rss = {(c, cfg): v for (c, cfg, lb), v in rss_all.items() if lb == "librosa"}
our_rss = {(c, cfg): v for (c, cfg, lb), v in rss_all.items() if lb == "ours"}
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
        "librosa_rss_kb": median_of(lib_rss.get(k, [])),
        "ours_rss_kb": median_of(our_rss.get(k, [])),
        "librosa_rss_all": sorted(lib_rss.get(k, [])),
        "ours_rss_all": sorted(our_rss.get(k, [])),
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
def larger(a, b):
    return a is not None and b is not None and a > b


# Any measure above librosa, including a peak above librosa's 0 B (no ratio).
strict = [r for r in rows if larger(r["ours_py_ms"], r["librosa_ms"]) or larger(r["rust_ms"], r["librosa_ms"])
          or larger(r["ours_peak"], r["librosa_peak"])]
write_table("not_faster.md", "Rows where audio_samples is not faster or not smaller (any ratio above 1)",
            strict, INTRO)


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

def fkb(xs):
    """Median KB and range of the repeats."""
    if not xs:
        return "–"
    m = median_of(xs)
    rng = f" ({xs[0]}-{xs[-1]})" if xs[0] != xs[-1] else ""
    return f"{m:g}{rng}"


with open(os.path.join(args.out, "rss.md"), "w") as f:
    f.write("# Peak RSS growth per call (fresh process, after a warm-up on 1 s of input)\n\n"
            "Exact peak resident size during the call minus the resident size before it (rss_shim.c), "
            "in KB: median over the fresh-process repeats, range in brackets. "
            "Ratio = ours median / librosa median.\n\n")
    f.write("| function | input | repeats librosa / ours | librosa KB | ours (py) KB | ratio |\n"
            "|---|---|---:|---:|---:|---:|\n")
    for r in rows:
        la, oa = r["librosa_rss_all"], r["ours_rss_all"]
        if not la and not oa:
            continue
        lk, ok = r["librosa_rss_kb"], r["ours_rss_kb"]
        f.write(f"| {r['func']} [{r['case']}] | {r['config']} | {len(la)} / {len(oa)} | {fkb(la)} | {fkb(oa)} "
                f"| {fratio(ratio(ok, lk) if lk else None)} |\n")

# Every measure, ours against librosa: below / equal / above, and the rows not strictly below.
MEASURES = [
    ("time, Python (ms)", "librosa_ms", "ours_py_ms", fms),
    ("time, native Rust (ms)", "librosa_ms", "rust_ms", fms),
    ("memray peak, Python (bytes)", "librosa_peak", "ours_peak", lambda x: f"{x:.0f}"),
    ("native peak vs librosa memray peak (bytes)", "librosa_peak", "rust_peak", lambda x: f"{x:.0f}"),
    ("time, default threads, Python (ms)", "librosa_default_ms", "ours_default_ms", fms),
    ("time, default threads, native Rust (ms)", "librosa_default_ms", "rust_default_ms", fms),
    ("peak RSS growth, median of repeats (KB)", "librosa_rss_kb", "ours_rss_kb", lambda x: f"{x:g}"),
]
with open(os.path.join(args.out, "comparison.md"), "w") as f:
    f.write("# audio_samples against librosa, every measure\n\n"
            "Counts of rows where ours is below, equal to or above librosa (exact comparison of the "
            "recorded values, no tolerance band), then every row where ours is not strictly below.\n\n"
            "| measure | rows | ours lower | equal | ours higher |\n|---|---:|---:|---:|---:|\n")
    detail = []
    for title, lk, ok, fmt in MEASURES:
        lo = eq = hi = 0
        for r in rows:
            a, b = r[ok], r[lk]
            if a is None or b is None:
                continue
            if a < b:
                lo += 1
            elif a == b:
                eq += 1
                detail.append((title, r, fmt(a), fmt(b), "equal"))
            else:
                hi += 1
                detail.append((title, r, fmt(a), fmt(b), "OURS HIGHER"))
        f.write(f"| {title} | {lo + eq + hi} | {lo} | {eq} | {hi} |\n")
    f.write("\n## Rows where ours is not strictly lower\n\n"
            "| measure | function | input | ours | librosa | result |\n|---|---|---|---:|---:|---|\n")
    for title, r, a, b, res in detail:
        f.write(f"| {title} | {r['func']} [{r['case']}] | {r['config']} | ours {a} | librosa {b} | {res} |\n")

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

if args.baseline:
    base = {(r["case"], r["config"]): r for r in json.load(open(os.path.join(args.baseline, "merged.json")))}

    def ba(key, r, b):
        x, y = b.get(key) if b else None, r.get(key)
        return f"{fms(x)} -> {fms(y)}"

    with open(os.path.join(args.out, "before_after.md"), "w") as f:
        f.write("# Before (baseline) and after, one row per case\n\n"
                "Median ms and memray peak; librosa from the current run. Ratio = ours (py) / librosa.\n\n"
                "| function | input | librosa ms | ours py ms before -> after | ours rust ms before -> after "
                "| ratio py before -> after | peak py before -> after | librosa peak |\n"
                "|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in main_rows:
            b = base.get((r["case"], r["config"]))
            rb = b.get("ratio_py") if b else None
            f.write(f"| {r['func']} [{r['case']}] | {r['config']} | {fms(r['librosa_ms'])} "
                    f"| {ba('ours_py_ms', r, b)} | {ba('rust_ms', r, b)} | {fratio(rb)} -> {fratio(r['ratio_py'])} "
                    f"| {fbytes(b.get('ours_peak') if b else None)} -> {fbytes(r['ours_peak'])} "
                    f"| {fbytes(r['librosa_peak'])} |\n")
        both = [(base[(r["case"], r["config"])], r) for r in rows if (r["case"], r["config"]) in base]
        f.write("\nGeometric-mean ratios over the rows present in both runs (py, rust, memory py): before "
                f"{fratio(gmean(b['ratio_py'] for b, _ in both))}, {fratio(gmean(b['ratio_rust'] for b, _ in both))}, "
                f"{fratio(gmean(b['mem_ratio'] for b, _ in both))}; after "
                f"{fratio(gmean(r['ratio_py'] for _, r in both))}, {fratio(gmean(r['ratio_rust'] for _, r in both))}, "
                f"{fratio(gmean(r['mem_ratio'] for _, r in both))}\n")

print(f"{len(rows)} rows; {len(slower)} with audio_samples slower or larger; written to {args.out}")
