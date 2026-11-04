"""Every figure quoted in the librosa guide, recomputed from one run's records.

    guide_numbers.py OUT_DIR

Reads ``merged.json`` (report.py), ``load_*.jsonl``, ``startup*.jsonl`` and
``footprint.json`` from OUT_DIR and writes ``guide_numbers.md`` there. Ratios are
ours / librosa: below 1 means audio_samples is faster or smaller.
"""

from __future__ import annotations

import json
import math
import os
import statistics as st
import sys
from collections import defaultdict

out_dir = sys.argv[1]
L = []


def w(s=""):
    L.append(s)


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def table(head, rows):
    w("| " + " | ".join(head) + " |")
    w("|" + "---|" * len(head))
    for r in rows:
        w("| " + " | ".join(str(c) for c in r) + " |")
    w()


def ms(x):
    return f"{x:.3g}" if x < 100 else f"{x:.0f}"


def mem(x):
    if x < 1024:
        return f"{x:.0f} B"
    if x < 1024**2:
        return f"{x / 1024:.1f} KB"
    return f"{x / 1024**2:.1f} MB"


def counts(rows, ours, lib):
    lo = eq = hi = 0
    for r in rows:
        a, b = r.get(ours), r.get(lib)
        if a is None or b is None:
            continue
        lo += a < b
        eq += a == b
        hi += a > b
    return lo, eq, hi


rows = json.load(open(os.path.join(out_dir, "merged.json")))
w("# Figures for the librosa guide")
w()
w("## Whole benchmark")
w()
w(f"{len(rows)} rows, {len({r['case'] for r in rows})} cases.")
w()
measures = [
    ("time through Python, one core", "ours_py_ms", "librosa_ms"),
    ("time native Rust, one core", "rust_ms", "librosa_ms"),
    ("time through Python, default threads", "ours_default_ms", "librosa_default_ms"),
    ("time native Rust, default threads", "rust_default_ms", "librosa_default_ms"),
    ("peak allocation through Python (memray)", "ours_peak", "librosa_peak"),
    ("peak allocation native", "rust_peak", "librosa_peak"),
    ("peak RSS growth", "ours_rss_kb", "librosa_rss_kb"),
]
table(["measure", "rows", "ours lower", "equal", "ours higher"],
      [(n, sum(counts(rows, a, b)), *counts(rows, a, b)) for n, a, b in measures])


def ratios(rs, ours, lib):
    return [r[ours] / r[lib] for r in rs if r.get(ours) and r.get(lib)]


def spread(rs, ours, lib):
    x = sorted(ratios(rs, ours, lib))
    return dict(n=len(x), geo=geomean(x), med=st.median(x), p90=x[int(0.9 * (len(x) - 1))], worst=x[-1], best=x[0])


groups = sorted({r["group"] for r in rows})
w("Ratios by area (geometric mean, median, 90th percentile, worst row):")
w()
hdr = ["area", "rows", "time geo", "time median", "time p90", "time worst", "native geo", "memory geo", "memory median", "memory worst"]
out = []
for g in [*groups, None]:
    rs = [r for r in rows if g is None or r["group"] == g]
    t, n, m = spread(rs, "ours_py_ms", "librosa_ms"), spread(rs, "rust_ms", "librosa_ms"), spread(rs, "ours_peak", "librosa_peak")
    out.append((g or "all", len(rs), f"{t['geo']:.2f}", f"{t['med']:.2f}", f"{t['p90']:.2f}", f"{t['worst']:.3f}", f"{n['geo']:.2f}",
                f"{m['geo']:.2f}", f"{m['med']:.2f}", f"{m['worst']:.3f}"))
table(hdr, out)

w("Weakest 15 rows by time through Python:")
w()
weak = sorted((r for r in rows if r.get("ratio_py")), key=lambda r: -r["ratio_py"])[:15]
table(["case", "input", "ours ms", "librosa ms", "ratio"],
      [(r["case"], r["config"], f"{r['ours_py_ms']:.4g}", f"{r['librosa_ms']:.4g}", f"{r['ratio_py']:.3f}") for r in weak])
w("Weakest 15 rows by peak allocation (librosa above 0):")
w()
weak = sorted((r for r in rows if r.get("ours_peak") is not None and r.get("librosa_peak")), key=lambda r: -(r["ours_peak"] / r["librosa_peak"]))[:15]
table(["case", "input", "ours", "librosa", "ratio"],
      [(r["case"], r["config"], mem(r["ours_peak"]), mem(r["librosa_peak"]), f"{r['ours_peak'] / r['librosa_peak']:.3f}") for r in weak])

w("Common functions, 10 s mono float32 at 22050 Hz, one core:")
w()
common = ["stft", "istft", "melspectrogram", "mfcc", "chroma_stft", "chroma_cqt", "cqt", "spectral_centroid", "rms", "zero_crossing_rate",
          "onset_strength", "onset_detect", "tempo", "beat_track", "yin", "pyin", "piptrack", "hpss", "harmonic", "time_stretch",
          "pitch_shift", "griffinlim", "trim", "split", "resample_16000", "resample", "power_to_db", "amplitude_to_db", "dtw", "viterbi", "recurrence_matrix"]
by = defaultdict(dict)
for r in rows:
    by[r["case"]][r["config"]] = r
tab = []
for c in common:
    for cfg in ("22050_10s_mono_f32", "seq_f32"):
        r = by.get(c, {}).get(cfg)
        if r:
            tab.append((c, cfg, f"{r['ours_py_ms']:.3g}", f"{r['rust_ms']:.3g}", f"{r['librosa_ms']:.3g}", f"{r['ratio_py']:.2f}",
                        mem(r["ours_peak"]), mem(r["librosa_peak"])))
table(["case", "input", "ours Python ms", "ours native ms", "librosa ms", "ratio", "ours peak", "librosa peak"], tab)
missing = [c for c in common if c not in by]
w(f"Names not found as cases: {missing}")
w()
w("All case names: " + ", ".join(sorted(by)))
w()


def load_jsonl(name):
    p = os.path.join(out_dir, name)
    if not os.path.exists(p):
        return []
    return [json.loads(x) for x in open(p)]


# ---------------------------------------------------------------------------
w("## Reading files")
w()
time_recs = {}
for r in load_jsonl("load_time.jsonl"):
    if "median_ms" in r:
        time_recs[(r["row"], r["input"], r["lib"])] = r  # the last record wins (near ties are re-timed interleaved)
mem_recs = {(r["row"], r["input"], r["lib"]): r for r in load_jsonl("load_mem.jsonl") if "mr_peak" in r}
rss = defaultdict(list)
for r in load_jsonl("load_rss.jsonl"):
    if "rss_delta_kb" in r:
        rss[(r["row"], r["input"], r["lib"])].append(r["rss_delta_kb"])
keys = sorted({k[:2] for k in time_recs})
tab = []
tot = dict(time=[0, 0, 0], mem=[0, 0, 0], rss=[0, 0, 0])
for row, inp in keys:
    a, b = time_recs.get((row, inp, "ours")), time_recs.get((row, inp, "librosa"))
    if not a or not b:
        continue
    ma, mb = mem_recs.get((row, inp, "ours")), mem_recs.get((row, inp, "librosa"))
    ra, rb = rss.get((row, inp, "ours")), rss.get((row, inp, "librosa"))
    def bump(key, x, y):
        tot[key][0 if x < y else (1 if x == y else 2)] += 1
    if row != "defaults":
        bump("time", a["median_ms"], b["median_ms"])
        if ma and mb:
            bump("mem", ma["mr_peak"], mb["mr_peak"])
        if ra and rb:
            bump("rss", st.median(ra), st.median(rb))
    tab.append((row, inp, b["out"] if row == "defaults" else a["out"], f"{a['median_ms']:.3g}", f"{b['median_ms']:.3g}", f"{a['median_ms'] / b['median_ms']:.2f}",
                mem(ma["mr_peak"]) if ma else "", mem(mb["mr_peak"]) if mb else "", f"{st.median(ra):.0f}" if ra else "", f"{st.median(rb):.0f}" if rb else ""))
table(["row", "input", "out", "ours ms", "librosa ms", "ratio", "ours peak", "librosa peak", "ours RSS KB", "librosa RSS KB"], tab)
w(f"Excluding `defaults` (different work): ours lower / equal / higher: time {tot['time']}, memray {tot['mem']}, RSS {tot['rss']}.")
w()
near = [(k, time_recs[(*k, 'ours')]['median_ms'] / time_recs[(*k, 'librosa')]['median_ms']) for k in keys if (*k, 'ours') in time_recs and (*k, 'librosa') in time_recs]
w("Rows at 0.97 or above: " + ", ".join(f"{k[0]} {k[1]} {v:.3f}" for k, v in near if v >= 0.97))
w()

# ---------------------------------------------------------------------------
for fname, title in (("startup.jsonl", "## Start-up, pinned to one core, one thread"), ("startup_default.jsonl", "## Start-up, default threading, unpinned")):
    recs = load_jsonl(fname)
    if not recs:
        continue
    w(title)
    w()
    def rng(r):
        return f"{r['median_ms']:.1f} ({r['min_ms']:.1f}-{r['max_ms']:.1f})"
    imp = [r for r in recs if r["stage"] == "import"]
    table(["python -c", "median ms (range)", "runs"], [(r["name"], rng(r), r["runs"]) for r in imp])
    cold = [r for r in recs if r["stage"] == "cold_import"]
    if cold:
        w("Cold page cache:")
        w()
        table(["python -c", "median ms (range)", "runs"], [(r["name"], rng(r), r["runs"]) for r in cold])
    fc = defaultdict(dict)
    for r in recs:
        if r["stage"] == "first":
            fc[r["name"]][r["lib"]] = r
    if fc:
        w("First call in a fresh process (median ms, range), then the third call:")
        w()
        tb = []
        for name, d in fc.items():
            a, b = d["ours"], d["librosa"]
            tb.append((name, rng(a), f"{a['third_ms']:.3g}", rng(b), f"{b['third_ms']:.3g}", f"{a['median_ms'] / b['median_ms']:.3f}"))
        table(["function", "ours first", "ours third", "librosa first", "librosa third", "first-call ratio"], tb)
    for st_name, label in (("mfcc", "Process start to MFCC on a file (whole process)"), ("cold_mfcc", "The same with a cold page cache")):
        rs = [r for r in recs if r["stage"] == st_name]
        if rs:
            w(label + ":")
            w()
            d = defaultdict(dict)
            for r in rs:
                d[(r["input"], r["name"])][r["lib"]] = r
            table(["input", "settings", "ours ms", "librosa ms", "ratio"],
                  [(*k, rng(v["ours"]), rng(v["librosa"]), f"{v['ours']['median_ms'] / v['librosa']['median_ms']:.2f}") for k, v in d.items()])

# ---------------------------------------------------------------------------
p = os.path.join(out_dir, "footprint.json")
if os.path.exists(p):
    f = json.load(open(p))
    w("## Install footprint")
    w()
    table(["", "ours", "librosa"], [
        ("wheel file", f["ours_wheel"]["file"], f["librosa_wheel"]["file"]),
        ("wheel bytes", f["ours_wheel"]["bytes"], f["librosa_wheel"]["bytes"]),
        ("site-packages bytes", f["ours"]["site_packages_bytes"], f["librosa"]["site_packages_bytes"]),
        ("packages installed", f["ours"]["package_count"], f["librosa"]["package_count"]),
        ("numpy alone, bytes", f["numpy_only"]["site_packages_bytes"], f["numpy_only"]["site_packages_bytes"]),
    ])
    w("librosa environment: " + ", ".join(f"{k} {v}" for k, v in f["librosa"]["packages"].items()))
    w()
    w("ours environment: " + ", ".join(f"{k} {v}" for k, v in f["ours"]["packages"].items()))
    w()
    w("Largest entries, librosa environment: " + ", ".join(f"{k} {mem(v)}" for k, v in f["librosa"]["largest"].items()))
    w()

with open(os.path.join(out_dir, "guide_numbers.md"), "w") as fh:
    fh.write("\n".join(L) + "\n")
print(os.path.join(out_dir, "guide_numbers.md"))
