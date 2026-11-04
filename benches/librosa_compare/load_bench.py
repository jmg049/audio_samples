"""File reading: ``librosa.load`` and ``librosa.stream`` against the
audio_samples read path through its Python bindings.

    load_bench.py --write-files DIR
    load_bench.py --lib librosa|ours --files DIR --mode time|mem|rss --out FILE.jsonl
                  [--rows a,b] [--inputs x,y] [--threads 1|default]
    load_bench.py --list
    load_bench.py --mode rss-run --files DIR --out FILE.jsonl [--repeats N] [--jobs N] [--shim PATH]
    load_bench.py --mode pair --files DIR --out FILE.jsonl --rows a --inputs x

Every row names work both libraries do in the same way: same file, same channels,
same output rate, same sample type. A call of each library's defaults is a row
too (``defaults``), and differs in work: ``librosa.load(path)`` resamples to
22050 Hz mono float32, ``aus.io.read(path)`` keeps the file's rate, channels and
sample type.

Modes as in ``bench_py.py``: ``time`` is the wall time of one call, ``mem`` the
memray peak of one call after two warm-up calls, ``rss-run`` the peak resident
growth of one call in fresh processes (exact with ``rss_shim.so``), ``pair`` both
libraries timed in alternating rounds in one process, for near ties.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import THREAD_VARS, summarise, test_signal, time_call  # noqa: E402

RATE = 44100
# (name, soundfile format, subtype, native dtype name or None, seconds)
FILES = [
    ("wav16_10s", "WAV", "PCM_16", "int16", 10),
    ("wav16_60s", "WAV", "PCM_16", "int16", 60),
    ("wav16_600s", "WAV", "PCM_16", "int16", 600),
    ("wavf32_60s", "WAV", "FLOAT", "float32", 60),
    ("wav24_60s", "WAV", "PCM_24", None, 60),
    ("flac16_10s", "FLAC", "PCM_16", "int16", 10),
    ("flac16_60s", "FLAC", "PCM_16", "int16", 60),
    ("flac16_600s", "FLAC", "PCM_16", "int16", 600),
    ("flac24_60s", "FLAC", "PCM_24", None, 60),
]
SPEC = {f[0]: f for f in FILES}


def file_path(d, name):
    return os.path.join(d, name + (".flac" if SPEC[name][1] == "FLAC" else ".wav"))


def write_files(d):
    """44100 Hz stereo files of the benchmark's test signal. 600 s repeats 60 s."""
    import numpy as np
    import soundfile as sf

    os.makedirs(d, exist_ok=True)
    base = {s: np.stack([test_signal(RATE, s, c) for c in (0, 1)], axis=1) for s in (10, 60)}
    for name, fmt, sub, _, sec in FILES:
        p = file_path(d, name)
        if os.path.exists(p):
            continue
        x = np.concatenate([base[60]] * 10) if sec == 600 else base[sec]
        sf.write(p, x, RATE, format=fmt, subtype=sub)


# ---------------------------------------------------------------------------
# Rows: name -> (inputs, librosa call, ours call). Calls run `f(path)`.

STREAM_BLOCK = 256
STREAM_FRAME = 2048
STREAM_HOP = 512
# A librosa stream block of `block_length` frames covers this many samples and
# advances by this many.
STREAM_WINDOW = STREAM_FRAME + (STREAM_BLOCK - 1) * STREAM_HOP
STREAM_STEP = STREAM_BLOCK * STREAM_HOP
INT16 = {n for n, f in SPEC.items() if f[3] == "int16"}
ALL = set(SPEC)
SHORT = {n for n in SPEC if "600s" not in n}


def rows():
    import numpy as np

    L = None
    if args.lib == "librosa":
        import librosa as L
    import audio_samples as aus

    F32 = aus.SampleType.F32
    F64 = aus.SampleType.F64
    I16 = aus.SampleType.I16

    def stream_l(p, mono):
        n = 0
        for b in L.stream(p, block_length=STREAM_BLOCK, frame_length=STREAM_FRAME, hop_length=STREAM_HOP, mono=mono):
            n += b.size
        return n

    def stream_o(p, mono):
        n = 0
        with aus.io.StreamedAudioReader(p) as r:
            for b in r.windows(STREAM_WINDOW, STREAM_STEP, F32):
                if mono:
                    b = b.to_mono()
                n += b.total_samples
        return n

    def chunk_l(p):
        n = 0
        for b in L.stream(p, block_length=1, frame_length=4096, hop_length=4096, mono=False):
            n += b.size
        return n

    def chunk_o(p):
        n = 0
        with aus.io.StreamedAudioReader(p) as r:
            for b in r.frames(4096, F32):
                n += b.total_samples
        return n

    def part_o(p):
        with aus.io.StreamedAudioReader(p) as r:
            r.seek_to_frame(5 * RATE)
            return r.read_frames(2 * RATE, F32)

    # Writing: soundfile is what librosa users write with; librosa has no write function.
    written = {}
    work = tempfile.gettempdir()

    def save_s(p):
        import soundfile as sf

        if p not in written:
            written[p] = (*sf.read(p, dtype=SPEC[os.path.basename(p).split(".")[0]][3] or "int32"), sf.info(p).subtype)
        data, rate, subtype = written[p]
        out = os.path.join(work, f"save_librosa_{os.getpid()}" + os.path.splitext(p)[1])
        sf.write(out, data, rate, subtype=subtype)

    def save_o(p):
        if p not in written:
            written[p] = aus.io.read(p)
        out = os.path.join(work, f"save_ours_{os.getpid()}" + os.path.splitext(p)[1])
        aus.io.save(out, written[p])

    return {
        "save": (SHORT, save_s, save_o),
        # Whole file, nothing changed but the sample type asked for.
        "read_f32": (ALL, lambda p: L.load(p, sr=None, mono=False), lambda p: aus.io.read(p, as_type=F32)),
        "read_f64": (SHORT, lambda p: L.load(p, sr=None, mono=False, dtype=np.float64), lambda p: aus.io.read(p, as_type=F64)),
        "read_i16": (INT16, lambda p: L.load(p, sr=None, mono=False, dtype=np.int16), lambda p: aus.io.read(p, as_type=I16)),
        # Mix-down, then both libraries' float32.
        "read_mono_f32": (SHORT, lambda p: L.load(p, sr=None, mono=True), lambda p: aus.io.read(p, as_type=F32).to_mono()),
        # librosa.load(path) with nothing else: mono, 22050 Hz, float32.
        "load_default": (SHORT, lambda p: L.load(p), lambda p: aus.io.read(p, as_type=F32).to_mono().resample(22050, "high")),
        # The same in librosa's explicit spelling against read_and_resample, stereo.
        "read_resample_stereo": (SHORT, lambda p: L.load(p, sr=22050, mono=False), lambda p: aus.io.read_and_resample(p, 22050, as_type=F32)),
        # 2 s from 5 s in.
        "read_part": (SHORT, lambda p: L.load(p, sr=None, mono=False, offset=5.0, duration=2.0), part_o),
        # Block streams, consumed fully.
        "stream_blocks_mono": (SHORT, lambda p: stream_l(p, True), lambda p: stream_o(p, True)),
        "stream_blocks_stereo": (SHORT, lambda p: stream_l(p, False), lambda p: stream_o(p, False)),
        "stream_chunks_stereo": (SHORT, chunk_l, chunk_o),
        # Header only.
        "get_samplerate": (SHORT, lambda p: L.get_samplerate(p), lambda p: aus.io.info(p).sample_rate),
        "get_duration": (SHORT, lambda p: L.get_duration(path=p), lambda p: aus.io.info(p).duration),
        # Each library's own default call; different work (see the module docstring).
        "defaults": (SHORT, lambda p: L.load(p), lambda p: aus.io.read(p)),
    }


def selected():
    want_rows = set(args.rows.split(",")) if args.rows else None
    want_inputs = set(args.inputs.split(",")) if args.inputs else None
    for row, (inputs, lf, of) in rows().items():
        if want_rows and row not in want_rows:
            continue
        for name in sorted(inputs):
            if want_inputs and name not in want_inputs:
                continue
            yield row, name, (lf if args.lib == "librosa" else of)


def describe(r):
    import numpy as np

    if isinstance(r, tuple):
        r = r[0]
    if hasattr(r, "to_numpy"):
        r = r.to_numpy()
    if isinstance(r, np.ndarray):
        return f"{r.dtype}{list(r.shape)}"
    return repr(r)


def emit(rec, out):
    print(f"{rec['lib']:8} {rec['row']:22} {rec['input']:12} "
          + " ".join(f"{k}={rec[k]}" for k in ("median_ms", "mr_peak", "rss_delta_kb", "out", "error") if k in rec),
          file=sys.stderr)
    line = json.dumps(rec)
    if out:
        with open(out, "a") as fh:
            fh.write(line + "\n")
    else:
        print(line)


def measure_mem(f):
    import memray

    f()
    f()
    fd, path = tempfile.mkstemp(suffix=".memray")
    os.close(fd)
    os.unlink(path)
    try:
        with memray.Tracker(path, native_traces=False):
            r = f()
            del r
        md = memray.FileReader(path).metadata
        return {"mr_peak": md.peak_memory, "mr_allocs": md.total_allocations}
    finally:
        if os.path.exists(path):
            os.unlink(path)


def measure_rss(f):
    """One process, one row: warm up on the 10 s file of the same encoding, then one call."""
    import ctypes
    import gc

    lib = ctypes.CDLL(None)
    begin, end = lib.rss_shim_begin, lib.rss_shim_end
    begin.restype = end.restype = ctypes.c_long
    warm = args.warm_path
    r = f(warm)
    r = f(warm)
    del r
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    before = begin()
    r = f(args.path)
    peak = end()
    del r
    return {"rss_delta_kb": peak - before, "rss_before_kb": before}


def run():
    for row, name, f in selected():
        p = file_path(args.files, name)
        rec = {"lib": args.lib, "row": row, "input": name, "mode": args.mode, "threads": args.threads}
        call = lambda f=f, p=p: f(p)  # noqa: E731
        try:
            if args.mode == "time":
                rec["out"] = describe(call())
                rec.update(summarise(time_call(call)))
            elif args.mode == "mem":
                rec.update(measure_mem(call))
        except Exception as e:  # recorded, not fatal
            rec["error"] = f"{type(e).__name__}: {e}"
        emit(rec, args.out)


def run_pair():
    """Rows within a few percent: both libraries in one process, in alternating
    rounds, so that both see the same clock and cache state."""
    import gc

    fs = {}
    for lib in ("librosa", "ours"):
        args.lib = lib
        for row, name, f in selected():
            fs.setdefault((row, name), {})[lib] = f
    for (row, name), pair in fs.items():
        p = file_path(args.files, name)
        times = {lib: [] for lib in pair}
        out = {lib: describe(f(p)) for lib, f in pair.items()}
        for k in range(args.rounds):
            order = ("librosa", "ours") if k % 2 == 0 else ("ours", "librosa")
            for lib in order:
                gc.collect()
                times[lib] += time_call(lambda f=pair[lib]: f(p), min_time=0.01, min_reps=5, max_reps=30)
        for lib in pair:
            rec = {"lib": lib, "row": row, "input": name, "mode": "time", "threads": args.threads, "out": out[lib],
                   "interleaved_rounds": args.rounds}
            rec.update(summarise(times[lib]))
            emit(rec, args.out)


def rss_one():
    for row, name, f in selected():
        args.path = file_path(args.files, name)
        args.warm_path = args.path
        # A shorter file of the same encoding for the warm-up, where one exists.
        short = name.rsplit("_", 1)[0] + "_10s"
        if short in SPEC:
            args.warm_path = file_path(args.files, short)
        rec = {"lib": args.lib, "row": row, "input": name, "mode": "rss", "threads": args.threads}
        rec.update(measure_rss(f))
        print(json.dumps(rec))


def rss_run():
    """Fresh processes per row and library; 4 KiB pages and a fixed mmap
    threshold, as ``rss_probe.py``."""
    jobs = []
    for lib in (args.libs.split(",")):
        args.lib = lib
        for row, name, _ in selected():
            for _ in range(args.repeats):
                jobs.append((lib, row, name))

    def one(job):
        lib, row, name = job
        env = dict(os.environ)
        env["LD_PRELOAD"] = args.shim
        env["GLIBC_TUNABLES"] = "glibc.malloc.mmap_threshold=131072"
        for v in THREAD_VARS:
            env[v] = "1"
        cp = subprocess.run([sys.executable, os.path.abspath(__file__), "--lib", lib, "--files", args.files,
                             "--mode", "rss", "--rows", row, "--inputs", name], env=env, capture_output=True, text=True)
        lines = [ln for ln in cp.stdout.splitlines() if ln.startswith("{")]
        if not lines:
            return json.dumps({"lib": lib, "row": row, "input": name, "mode": "rss", "error": cp.stderr[-400:]})
        return lines[0]

    with ThreadPoolExecutor(args.jobs) as ex, open(args.out, "a") as fh:
        for line in ex.map(one, jobs):
            fh.write(line + "\n")
            fh.flush()


ap = argparse.ArgumentParser()
ap.add_argument("--lib", choices=("librosa", "ours"))
ap.add_argument("--files", default=os.path.join(HERE, "..", "..", "target", "librosa_bench", "files"))
ap.add_argument("--write-files", action="store_true")
ap.add_argument("--mode", default="time", choices=("time", "mem", "rss", "rss-run", "pair"))
ap.add_argument("--rounds", type=int, default=40)
ap.add_argument("--out", default=None)
ap.add_argument("--rows", default=None)
ap.add_argument("--inputs", default=None)
ap.add_argument("--threads", default="1")
ap.add_argument("--list", action="store_true")
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--jobs", type=int, default=4)
ap.add_argument("--shim", default=None)
ap.add_argument("--libs", default="librosa,ours")
args = ap.parse_args()

if args.mode == "rss":
    import ctypes

    ctypes.CDLL(None, use_errno=True).prctl(41, 1, 0, 0, 0)  # PR_SET_THP_DISABLE
if args.threads != "default":
    for v in THREAD_VARS:
        os.environ[v] = args.threads

if __name__ == "__main__":
    if args.write_files:
        write_files(args.files)
    elif args.list:
        args.lib = "ours"
        for row, name, _ in selected():
            print(row, name)
    elif args.mode == "rss":
        rss_one()
    elif args.mode == "rss-run":
        rss_run()
    elif args.mode == "pair":
        run_pair()
    else:
        if args.lib is None:
            sys.exit("--lib is required")
        run()
