"""Batch throughput against the number of Python threads, on any interpreter.

    thread_scaling.py --out results.jsonl [--lib ours|librosa] [--threads 1,2,4,8,16]
                      [--work DIR] [--repeats 5] [--files 32] [--rounds 4]

Three strong-scaling workloads with a fixed amount of work, each run through
one ``ThreadPoolExecutor``:

``mfcc``
    MFCC of a directory of files: read, mix to mono, resample to 22050 Hz, MFCC.
    The files are 30 s of 44100 Hz stereo 16-bit WAV, each with its own signal.
``stft``
    STFT of 60 s of float32 at 22050 Hz, one array per file.
``resample``
    22050 Hz to 16000 Hz at the highest quality of each library (``"high"``,
    ``soxr_hq``), 60 s of float32, one array per file.

A pass runs every job ``--rounds`` times. Each thread count gets its own pool. The first pass over the jobs warms the
per-thread caches and is not timed. Both the median and the fastest of ``--repeats``
timed passes are recorded; on a machine shared with other work the fastest is the
less disturbed one, and ``--quiet-below`` waits for a calm moment before each pass.
Speed-ups are relative to the fastest one-thread pass. One record is written per workload and
thread count, with the interpreter, the GIL state after the run, library
versions, the CPUs the process may use, the load average around the run and how
busy other processes kept those CPUs during it.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import sysconfig
import time
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--lib", default="ours", choices=("ours", "librosa"))
ap.add_argument("--threads", default="1,2,4,8,16")
ap.add_argument("--workloads", default="mfcc,stft,resample")
ap.add_argument("--work", default=None, help="directory for the generated files")
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--files", type=int, default=32)
ap.add_argument("--rounds", type=int, default=4, help="times each file or array appears in a pass")
ap.add_argument("--quiet-below", type=float, default=None, help="wait before each pass until other processes keep fewer than this many CPUs busy")
ap.add_argument("--quiet-wait", type=float, default=60.0, help="longest wait per pass, in seconds")
ap.add_argument("--note", default="")
args = ap.parse_args()

# Native libraries must not add their own threads on top of the Python ones.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np  # noqa: E402


def signal(seed: int, seconds: float, sr: int) -> np.ndarray:
    """Harmonic tones with an amplitude envelope and noise; each seed differs."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    y = np.zeros_like(t)
    for f in rng.uniform(80.0, 1200.0, 6):
        y += np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.2, 3.0) * t))
    y = 0.6 * y / np.max(np.abs(y)) + 0.05 * rng.standard_normal(t.size)
    return y.astype(np.float32)


def write_files(d: str, n: int) -> list[str]:
    import wave

    os.makedirs(d, exist_ok=True)
    paths = []
    for i in range(n):
        p = os.path.join(d, f"{i:03d}.wav")
        paths.append(p)
        if os.path.exists(p):
            continue
        left, right = signal(i, 30.0, 44100), signal(i + 1000, 30.0, 44100)
        pcm = (np.stack([left, right], axis=1) * 32767.0).astype("<i2")
        with wave.open(p, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(44100)
            w.writeframes(pcm.tobytes())
    return paths


if args.lib == "ours":
    import audio_samples as aus

    versions = {"audio_samples": aus.__version__}
    F32 = aus.SampleType.F32

    def mfcc(p):
        return aus.io.read(p, as_type=F32).to_mono().resample(22050, "high").mfcc_frames(top_db=80.0)

    def stft(y):
        return aus.spectral.stft(y)

    def resample(a):
        return a.resample(16000, "high")

    def make_resample_input(y):
        return aus.AudioSamples.new_mono(y, 22050)
else:
    import librosa
    import scipy

    versions = {"librosa": librosa.__version__, "scipy": scipy.__version__}
    try:
        import numba

        versions["numba"] = numba.__version__
    except ImportError:
        versions["numba"] = None

    def mfcc(p):
        y, sr = librosa.load(p)
        return librosa.feature.mfcc(y=y, sr=sr)

    def stft(y):
        return librosa.stft(y)

    def resample(y):
        return librosa.resample(y, orig_sr=22050, target_sr=16000)

    def make_resample_input(y):
        return y

versions["numpy"] = np.__version__

work_dir = args.work or os.path.join(os.path.dirname(args.out) or ".", "scaling_files")
workloads = {}
wanted = args.workloads.split(",")
if "mfcc" in wanted:
    workloads["mfcc"] = (mfcc, write_files(work_dir, args.files))
if {"stft", "resample"} & set(wanted):
    arrays = [signal(i, 60.0, 22050) for i in range(args.files)]
    if "stft" in wanted:
        workloads["stft"] = (stft, arrays)
    if "resample" in wanted:
        workloads["resample"] = (resample, [make_resample_input(y) for y in arrays])


def wait_until_quiet():
    if args.quiet_below is None:
        return
    deadline = time.monotonic() + args.quiet_wait
    while time.monotonic() < deadline:
        b0 = cpu_busy_s()
        time.sleep(1.0)
        if cpu_busy_s() - b0 < args.quiet_below:
            return


def timed_pass(fn, jobs, pool):
    t = time.perf_counter()
    # Results are dropped as they arrive, as a batch job that saves them would.
    for _ in pool.map(fn, jobs * args.rounds):
        pass
    return time.perf_counter() - t


def rusage_cpu():
    r = os.times()
    return r.user + r.system


def rusage_sys():
    return os.times().system


def cpu_busy_s():
    """Busy seconds since boot, summed over the CPUs this process may use (Linux only)."""
    allowed = os.sched_getaffinity(0)
    tick = os.sysconf("SC_CLK_TCK")
    total = 0
    try:
        with open("/proc/stat") as fh:
            for line in fh:
                head, *fields = line.split()
                if head.startswith("cpu") and head != "cpu" and int(head[3:]) in allowed:
                    user, nice, system, _idle, _iowait, irq, softirq, steal = map(int, fields[:8])
                    total += user + nice + system + irq + softirq + steal
    except OSError:
        return float("nan")
    return total / tick


one_thread = {}
for name, (fn, jobs) in workloads.items():
    for n in [int(x) for x in args.threads.split(",")]:
        with ThreadPoolExecutor(n) as pool:
            timed_pass(fn, jobs, pool)  # warm-up
            load_before = os.getloadavg()[0]
            cpu0, sys0 = rusage_cpu(), rusage_sys()
            walls, others = [], []
            for _ in range(args.repeats):
                wait_until_quiet()
                c0, b0 = rusage_cpu(), cpu_busy_s()
                wall_i = timed_pass(fn, jobs, pool)
                walls.append(wall_i)
                others.append(((cpu_busy_s() - b0) - (rusage_cpu() - c0)) / wall_i)
            cpu1, sys1 = rusage_cpu(), rusage_sys()
        wall = min(walls)
        if n == int(args.threads.split(",")[0]):
            one_thread[name] = wall
        rec = {
            "workload": name,
            "lib": args.lib,
            "threads": n,
            "jobs": len(jobs) * args.rounds,
            "wall_s": wall,
            "wall_median_s": statistics.median(walls),
            "wall_all_s": walls,
            "speedup": one_thread[name] / wall,
            "jobs_per_s": len(jobs) * args.rounds / wall,
            "cpu_s_per_pass": (cpu1 - cpu0) / args.repeats,
            "sys_s_per_pass": (sys1 - sys0) / args.repeats,
            # CPU time other processes used on the same CPUs, per second of each pass.
            "other_cpus_busy": others[walls.index(wall)],
            "other_cpus_busy_all": others,
            "python": sys.version.split()[0],
            "free_threaded_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
            "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
            "cpus": sorted(os.sched_getaffinity(0)),
            "load_before": load_before,
            "load_after": os.getloadavg()[0],
            "versions": versions,
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "note": args.note,
        }
        print(f"{name:9s} {args.lib:8s} {n:3d} threads {wall:8.3f} s x{rec['speedup']:5.2f} cpu/pass {rec['cpu_s_per_pass']:7.2f} sys {rec['sys_s_per_pass']:6.2f} others {rec['other_cpus_busy']:.2f} cpus", flush=True)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
