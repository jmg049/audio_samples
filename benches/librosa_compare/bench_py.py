"""Time and measure librosa, or audio_samples through its Python bindings.

Usage (from any directory; use the Python environment that has librosa and
the audio_samples bindings installed):

    bench_py.py --lib librosa|ours --signals DIR --out FILE.jsonl
                [--mode time|mem|rss] [--threads 1|default]
                [--cases a,b] [--groups g,h] [--configs id,id | --config-set rss|default]
    bench_py.py --list            # case/config pairs as JSON, for the Rust harness

Modes:

* ``time``: wall time per call (see ``common.time_call``).
* ``mem``: after two warm-up calls, one call under ``tracemalloc`` (Python and
  numpy allocations; it does not see allocations made inside the Rust
  extension) and one call under ``memray`` (every ``malloc``/``mmap`` in the
  process, numpy and Rust alike): peak heap bytes and allocation count.
* ``rss``: run by ``rss_probe.py`` in a fresh process per case and
  configuration: warm up on a 1 s input of the same rate, layout and dtype,
  reset the peak-RSS counter (``/proc/self/clear_refs``), make one call on the
  real input and report ``VmHWM - VmRSS(before)``.

``--threads 1`` sets the BLAS/OpenMP/numba/rayon thread counts to 1 before numpy
is imported (pin the process to one core with ``taskset`` as well);
``--threads default`` leaves every library's default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import THREAD_VARS, Config, parse_config, summarise, time_call  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lib", choices=("librosa", "ours"))
ap.add_argument("--signals", default=os.path.join(HERE, "..", "..", "target", "librosa_bench", "signals"))
ap.add_argument("--out", default=None)
ap.add_argument("--mode", default="time", choices=("time", "mem", "rss"))
ap.add_argument("--threads", default="1")
ap.add_argument("--cases", default=None)
ap.add_argument("--groups", default=None)
ap.add_argument("--configs", default=None)
ap.add_argument("--config-set", default=None, choices=("rss", "default"))
ap.add_argument("--list", action="store_true")
ap.add_argument("--tsv", action="store_true", help="with --list: `case config` lines")
ap.add_argument("--min-time", type=float, default=None)
args = ap.parse_args()

if args.threads != "default":
    for v in THREAD_VARS:
        os.environ[v] = args.threads

import numpy as np  # noqa: E402

import common  # noqa: E402
from cases import Ctx, all_cases  # noqa: E402


def selected():
    cases = all_cases()
    if args.cases:
        want = set(args.cases.split(","))
        unknown = want - {c.name for c in cases}
        if unknown:
            sys.exit(f"unknown cases: {sorted(unknown)}")
        cases = [c for c in cases if c.name in want]
    if args.groups:
        g = set(args.groups.split(","))
        cases = [c for c in cases if c.group in g]
    cfg_filter = None
    if args.configs:
        cfg_filter = set(args.configs.split(","))
    elif args.config_set == "rss":
        cfg_filter = common.RSS_CONFIGS
    elif args.config_set == "default":
        cfg_filter = common.DEFAULT_THREAD_CONFIGS
    for case in cases:
        for cfg in case.config_list():
            if cfg_filter is None or cfg.id in cfg_filter:
                yield case, cfg


if args.list and args.tsv:
    for c, cfg in selected():
        print(f"{c.name} {cfg.id}")
    sys.exit(0)
if args.list:
    print(json.dumps([{"case": c.name, "group": c.group, "config": cfg.id} for c, cfg in selected()]))
    sys.exit(0)

if args.lib is None:
    sys.exit("--lib is required")


def describe(r):
    """dtype and shape of a result (first array of a tuple)."""
    if isinstance(r, tuple) and r:
        r = r[0]
    if hasattr(r, "__array__") and not isinstance(r, np.ndarray):
        try:
            r = np.asarray(r)
        except Exception:
            return type(r).__name__
    if isinstance(r, np.ndarray):
        return f"{r.dtype}{list(r.shape)}"
    return type(r).__name__


def builder(case):
    return case.librosa if args.lib == "librosa" else case.ours


def emit(rec, out):
    line = json.dumps(rec)
    if out:
        out.write(line + "\n")
        out.flush()
    else:
        print(line)


def rss_kb(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    raise RuntimeError(field)


def malloc_trim():
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def run():
    out = open(args.out, "a") if args.out else None
    ctxs: dict[str, Ctx] = {}
    for case, cfg in selected():
        build = builder(case)
        rec = {"lib": args.lib, "case": case.name, "group": case.group, "config": cfg.id,
               "mode": args.mode, "threads": args.threads}
        if build is None:
            continue
        ctx = ctxs.get(cfg.id)
        if ctx is None:
            ctxs.clear()  # keep one configuration's inputs alive at a time
            ctx = ctxs[cfg.id] = Ctx(cfg, args.signals)
        try:
            if args.mode == "time":
                f = build(ctx)
                rec["out"] = describe(f())
                kw = {} if args.min_time is None else {"min_time": args.min_time}
                rec.update(summarise(time_call(f, **kw)))
            elif args.mode == "mem":
                rec.update(measure_mem(build(ctx)))
            else:
                rec.update(measure_rss(case, cfg, build, ctx))
        except Exception as e:  # recorded, not fatal
            rec["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc(file=sys.stderr)
        print(f"{args.lib:8} {case.name:34} {cfg.id:22} "
              + (f"{rec['median_ms']:10.3f} ms" if "median_ms" in rec else str({k: v for k, v in rec.items() if k in ('tm_peak', 'mr_peak', 'mr_allocs', 'rss_delta_kb', 'error')})),
              file=sys.stderr)
        emit(rec, out)


def measure_mem(f):
    import tempfile
    import tracemalloc

    import memray

    f()
    f()
    tracemalloc.start()
    r = f()
    _, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del r
    fd, path = tempfile.mkstemp(suffix=".memray", dir=os.environ.get("TMPDIR"))
    os.close(fd)
    os.unlink(path)
    try:
        with memray.Tracker(path, native_traces=False):
            r = f()
            del r
        md = memray.FileReader(path).metadata
        return {"tm_peak": tm_peak, "mr_peak": md.peak_memory, "mr_allocs": md.total_allocations}
    finally:
        if os.path.exists(path):
            os.unlink(path)


def measure_rss(case, cfg, build, ctx):
    import gc

    # warm-up on a short input: JIT compilation, FFT plans, filterbank caches
    if cfg.kind == "audio":
        wcfg = Config("audio", cfg.sr, 1, cfg.channels, cfg.dtype)
        wf = build(Ctx(wcfg, args.signals))
    else:
        wf = build(ctx)
    wf()
    wf()
    del wf
    f = build(ctx)
    gc.collect()
    malloc_trim()
    with open("/proc/self/clear_refs", "w") as fh:
        fh.write("5")
    before = rss_kb("VmRSS")
    t = time.perf_counter()
    r = f()
    dt = time.perf_counter() - t
    peak = rss_kb("VmHWM")
    del r
    return {"rss_delta_kb": peak - before, "rss_before_kb": before, "cold_ms": dt * 1e3}


if __name__ == "__main__":
    run()
