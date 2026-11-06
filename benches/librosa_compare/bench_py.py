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
  ``--processes N`` measures each case and configuration in N fresh processes
  instead, each of which first warms up both libraries' calls (``--warm-both``),
  and records the median peak. A call that allocates many small Python objects
  (plotting) can need a new 1 MiB pymalloc arena or fit in the free space of the
  arenas already mapped; which happens varies between processes (hash seeds,
  address layout) and moves the peak by about 1 MiB, for either library.
* ``pair``: librosa and audio_samples in one process, timed in alternating
  rounds (``--rounds``, the order swapped every round), for rows whose two
  separate ``time`` runs are within a few percent: both see the same clock
  and cache state. Each library's record has the median over all its timed
  calls; ``--out`` takes audio_samples' records and ``--out-librosa``
  librosa's.
* ``rss``: run by ``rss_probe.py`` in a fresh process per case and
  configuration: warm up on a 1 s input of the same rate, layout and dtype,
  make one call on the real input and report the peak resident size during
  the call minus the resident size before it. With ``rss_shim.so`` preloaded
  (``rss_probe.py`` does this) the peak is exact (see ``rss_shim.c``);
  without it the kernel's ``VmHWM`` is used, which can be off by a few
  hundred KB.

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
ap.add_argument("--mode", default="time", choices=("time", "mem", "rss", "pair"))
ap.add_argument("--out-librosa", default=None, help="with --mode pair: file for librosa's records")
ap.add_argument("--rounds", type=int, default=40, help="with --mode pair: alternating rounds")
ap.add_argument("--threads", default="1")
ap.add_argument("--cases", default=None)
ap.add_argument("--groups", default=None)
ap.add_argument("--configs", default=None)
ap.add_argument("--config-set", default=None, choices=("rss", "default"))
ap.add_argument("--list", action="store_true")
ap.add_argument("--tsv", action="store_true", help="with --list: `case config` lines")
ap.add_argument("--min-time", type=float, default=None)
ap.add_argument("--warm-both", action="store_true",
                help="with --mode mem: warm up both libraries' calls before measuring")
ap.add_argument("--processes", type=int, default=1,
                help="with --mode mem: median over this many fresh processes per row")
args = ap.parse_args()

if args.mode == "rss":
    # Count resident memory in 4 KiB pages only: with transparent huge pages
    # a first touch can map a whole 2 MiB page, depending on alignment and on
    # whether the kernel has a free huge page at that moment, which made the
    # same call differ by up to 1.4 MB between processes. Same for both
    # libraries; set before any large mapping is touched.
    import ctypes

    PR_SET_THP_DISABLE = 41
    ctypes.CDLL(None, use_errno=True).prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0)

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

if args.lib is None and args.mode != "pair":
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
            elif args.mode == "mem" and args.processes > 1:
                rec.update(measure_mem_processes(case, cfg))
            elif args.mode == "mem":
                if args.warm_both:
                    for other in (case.librosa, case.ours):
                        if other is not None and other is not build:
                            g = other(ctx)
                            g()
                            g()
                            del g
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


def run_pair():
    """``--mode pair``: alternating rounds of both libraries in this process."""
    outs = {"ours": open(args.out, "a") if args.out else None,
            "librosa": open(args.out_librosa, "a") if args.out_librosa else None}
    for case, cfg in selected():
        if case.librosa is None or case.ours is None:
            continue
        ctx = Ctx(cfg, args.signals)
        fs = {"librosa": case.librosa(ctx), "ours": case.ours(ctx)}
        times = {lib: [] for lib in fs}
        outs_desc = {lib: describe(f()) for lib, f in fs.items()}
        for k in range(args.rounds):
            order = ("librosa", "ours") if k % 2 == 0 else ("ours", "librosa")
            for lib in order:
                times[lib] += time_call(fs[lib], min_time=0.01, min_reps=5, max_reps=30)
        for lib in fs:
            rec = {"lib": lib, "case": case.name, "group": case.group, "config": cfg.id, "mode": "time",
                   "threads": args.threads, "out": outs_desc[lib], "interleaved_rounds": args.rounds}
            rec.update(summarise(times[lib]))
            print(f"{lib:8} {case.name:34} {cfg.id:22} {rec['median_ms']:10.4f} ms (interleaved)", file=sys.stderr)
            emit(rec, outs[lib])


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


def measure_mem_processes(case, cfg):
    """``--mode mem`` for one row in ``--processes`` fresh processes, each warming up both
    libraries first: the median peaks, and every process's memray peak."""
    import statistics
    import subprocess

    recs = []
    for _ in range(args.processes):
        cmd = [sys.executable, os.path.abspath(__file__), "--lib", args.lib, "--mode", "mem",
               "--warm-both", "--signals", args.signals, "--threads", args.threads,
               "--cases", case.name, "--configs", cfg.id]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        recs += [json.loads(line) for line in res.stdout.splitlines() if line.startswith("{")]
    recs = [r for r in recs if "mr_peak" in r]
    if not recs:
        raise RuntimeError("no process measured the call")
    return {"tm_peak": statistics.median_low([r["tm_peak"] for r in recs]),
            "mr_peak": statistics.median_low([r["mr_peak"] for r in recs]),
            "mr_allocs": statistics.median_low([r["mr_allocs"] for r in recs]),
            "process_mr_peaks": [r["mr_peak"] for r in recs], "warm_both": True}


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
    shim = rss_shim()
    with open("/proc/self/clear_refs", "w") as fh:
        fh.write("5")
    if shim is None:
        before = rss_kb("VmRSS")
    else:
        before = shim.rss_shim_begin()
    t = time.perf_counter()
    r = f()
    dt = time.perf_counter() - t
    # End the exact tracking before reading VmHWM: reading /proc/self/status
    # from Python allocates.
    peak = None if shim is None else shim.rss_shim_end()
    hwm = rss_kb("VmHWM")
    if peak is None:
        peak = hwm
    del r
    return {"rss_delta_kb": peak - before, "rss_before_kb": before, "cold_ms": dt * 1e3,
            "rss_method": "VmHWM" if shim is None else "exact", "hwm_delta_kb": hwm - before}


def rss_shim():
    """The exact peak-RSS tracker (``rss_shim.c``) if it is preloaded, else None."""
    import ctypes

    try:
        lib = ctypes.CDLL(None)
        begin, end = lib.rss_shim_begin, lib.rss_shim_end
    except (OSError, AttributeError):
        return None
    begin.restype = end.restype = ctypes.c_long
    return lib


if __name__ == "__main__":
    if args.mode == "pair":
        run_pair()
    else:
        run()
