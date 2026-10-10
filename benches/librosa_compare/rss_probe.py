"""Peak-RSS growth per call, each measured in a fresh Python process.

For every case and configuration of the ``rss`` set (see ``common.RSS_CONFIGS``)
and each library, starts ``bench_py.py --mode rss`` in a new process (see that
mode's description) and collects the JSON lines it prints.

    rss_probe.py --out rss.jsonl [--jobs 6] [--cases a,b] [--groups g]

The probes run in parallel (``--jobs``); peak RSS does not depend on timing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench_py.py")

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--jobs", type=int, default=6)
ap.add_argument("--signals", default=None)
ap.add_argument("--cases", default=None)
ap.add_argument("--groups", default=None)
ap.add_argument("--libs", default="librosa,ours")
args = ap.parse_args()

sel = []
if args.cases:
    sel += ["--cases", args.cases]
if args.groups:
    sel += ["--groups", args.groups]
sig = ["--signals", args.signals] if args.signals else []

pairs = json.loads(
    subprocess.check_output([sys.executable, BENCH, "--list", "--config-set", "rss", *sel, *sig])
)
jobs = [(lib, p["case"], p["config"]) for p in pairs for lib in args.libs.split(",")]


def probe(job):
    lib, case, cfg = job
    r = subprocess.run(
        [sys.executable, BENCH, "--mode", "rss", "--lib", lib, "--cases", case, "--configs", cfg, *sig],
        capture_output=True,
        text=True,
    )
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    if lines:
        return lines[-1]
    err = (r.stderr.strip().splitlines() or ["no output"])[-1]
    return json.dumps({"lib": lib, "case": case, "config": cfg, "mode": "rss", "error": err})


with open(args.out, "a") as out, ThreadPoolExecutor(args.jobs) as pool:
    for i, line in enumerate(pool.map(probe, jobs)):
        out.write(line + "\n")
        out.flush()
        rec = json.loads(line)
        print(f"[{i + 1}/{len(jobs)}] {rec['lib']:8} {rec['case']:34} {rec['config']:22} "
              f"{rec.get('rss_delta_kb', rec.get('error'))}", file=sys.stderr)
