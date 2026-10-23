"""Peak-RSS growth per call, each measured in a fresh Python process.

For every case and configuration of the ``rss`` set (see ``common.RSS_CONFIGS``)
and each library, starts ``bench_py.py --mode rss`` in a new process (see that
mode's description) ``--repeats`` times and collects the JSON lines it prints
(one per process, with a ``rep`` field).

    rss_probe.py --out rss.jsonl [--jobs 6] [--repeats 5] [--shim rss_shim.so]
                 [--cases a,b] [--groups g]

``--shim`` preloads the exact peak-RSS tracker built from ``rss_shim.c``
(``run.sh`` builds it); without it the probes fall back to the kernel's
``VmHWM``, which can be off by a few hundred KB. Every probe runs with
glibc's mmap threshold fixed at 128 KiB (see the comment at ``env``).

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
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--shim", default=None, help="rss_shim.so to preload for exact peaks")
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
jobs = [(lib, p["case"], p["config"], rep)
        for rep in range(args.repeats) for p in pairs for lib in args.libs.split(",")]
env = dict(os.environ)
# A fixed malloc mmap threshold (glibc's default 128 KiB, without its dynamic
# raising): every allocation of 128 KiB or more gets its own mapping, freed
# when released. Otherwise whether a large result comes from the heap, and
# so whether its first page or a 2 MiB huge page around it is already
# resident, depends on what the process freed before, which differs between
# the two libraries' processes for the same work.
env["GLIBC_TUNABLES"] = ":".join(
    t for t in (env.get("GLIBC_TUNABLES"), "glibc.malloc.mmap_threshold=131072") if t
)
if args.shim:
    env["LD_PRELOAD"] = os.path.abspath(args.shim)


def probe(job):
    lib, case, cfg, rep = job
    r = subprocess.run(
        [sys.executable, BENCH, "--mode", "rss", "--lib", lib, "--cases", case, "--configs", cfg, *sig],
        capture_output=True,
        text=True,
        env=env,
    )
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    if lines:
        rec = json.loads(lines[-1])
    else:
        err = (r.stderr.strip().splitlines() or ["no output"])[-1]
        rec = {"lib": lib, "case": case, "config": cfg, "mode": "rss", "error": err}
    rec["rep"] = rep
    return json.dumps(rec)


with open(args.out, "a") as out, ThreadPoolExecutor(args.jobs) as pool:
    for i, line in enumerate(pool.map(probe, jobs)):
        out.write(line + "\n")
        out.flush()
        rec = json.loads(line)
        print(f"[{i + 1}/{len(jobs)}] {rec['lib']:8} {rec['case']:34} {rec['config']:22} "
              f"{rec.get('rss_delta_kb', rec.get('error'))}", file=sys.stderr)
