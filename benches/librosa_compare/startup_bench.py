"""Start-up costs: every measurement is in fresh processes.

    startup_bench.py --files DIR --signals DIR --out FILE.jsonl [--repeats N] [--stages import,first,mfcc,cold]

Stages:

* ``import``: wall time of ``python -c "import X"`` from process start to exit.
* ``first``: in a fresh process per function and library, the time of the first,
  second and third call on 10 s of float32 mono at 22050 Hz, after the import.
  The first call pays for lazy imports, numba compilation and table building.
* ``mfcc``: wall time of a whole process that imports the library, reads a file
  and computes MFCCs with librosa's settings.
* ``cold``: ``import`` and ``mfcc`` with the Python installation's files evicted
  from the page cache before each process (``posix_fadvise``, no root needed).

Set ``CORE`` to pin the processes (``taskset``), as the other stages do; the
thread variables are set to 1 when it is given.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import THREAD_VARS  # noqa: E402

IMPORTS = {
    "python": "pass",
    "numpy": "import numpy",
    "librosa": "import librosa",
    "audio_samples": "import audio_samples",
    # librosa loads its submodules on first access.
    "librosa_feature": "import librosa; librosa.feature.mfcc",
    # A script that also uses NumPy, which both libraries need for arrays.
    "numpy_librosa": "import numpy, librosa",
    "numpy_audio_samples": "import numpy, audio_samples",
}

# name -> (librosa call, ours call). `y` is float32 mono, `a` the same as AudioSamples,
# `sr` 22050, `X` a 40 x 200 float32 matrix, `P` a 12 x 200 probability matrix.
FIRST = {
    "stft": ("librosa.stft(y)", "a.stft_matrix()"),
    "melspectrogram": ("librosa.feature.melspectrogram(y=y, sr=sr)", "a.melspectrogram()"),
    "mfcc": ("librosa.feature.mfcc(y=y, sr=sr)", "a.mfcc_frames(top_db=80.0)"),
    "chroma_stft": ("librosa.feature.chroma_stft(y=y, sr=sr)", "a.chroma_stft(tuning=None)"),
    "cqt": ("librosa.cqt(y, sr=sr)", "a.cqt()"),
    "onset_strength": ("librosa.onset.onset_strength(y=y, sr=sr)", "a.onset_strength()"),
    "onset_detect": ("librosa.onset.onset_detect(y=y, sr=sr)", "a.onset_detect()"),
    "tempo": ("librosa.feature.tempo(y=y, sr=sr)", "a.tempo()"),
    "beat_track": ("librosa.beat.beat_track(y=y, sr=sr)", "a.beat_track(trim=True)"),
    "yin": ("librosa.yin(y, fmin=65.4, fmax=2093.0, sr=sr)", "a.yin(fmin=65.4, fmax=2093.0)"),
    "pyin": ("librosa.pyin(y, fmin=65.4, fmax=2093.0, sr=sr)", "a.pyin(fmin=65.4, fmax=2093.0)"),
    "piptrack": ("librosa.piptrack(y=y, sr=sr)", "a.piptrack()"),
    "hpss": ("librosa.effects.hpss(y)", "a.harmonic_percussive()"),
    "time_stretch": ("librosa.effects.time_stretch(y, rate=1.25)", "a.time_stretch(1.25)"),
    "pitch_shift": ("librosa.effects.pitch_shift(y, sr=sr, n_steps=2)", 'a.pitch_shift(2.0, res_type="soxr_hq")'),
    "resample": ("librosa.resample(y, orig_sr=sr, target_sr=16000)", 'a.resample(16000, "high")'),
    "griffinlim": ("librosa.griffinlim(S, n_iter=4)", "aus.effects.griffinlim(S, n_iter=4)"),
    "trim": ("librosa.effects.trim(y)", "a.trim_db(top_db=60.0)"),
    "dtw": ("librosa.sequence.dtw(X, X)", "aus.sequence.dtw(X, X)"),
    "viterbi": ("librosa.sequence.viterbi(P, T)", "aus.sequence.viterbi(P, T)"),
    "recurrence_matrix": ("librosa.segment.recurrence_matrix(X)", "aus.sequence.recurrence_matrix(X)"),
}

FIRST_CHILD = """
import sys, time
t_start = time.perf_counter()
import numpy as np
LIB
t_import = time.perf_counter()
y = np.fromfile(SIG, dtype="<f8").astype(np.float32)
sr = 22050
rng = np.random.default_rng(0)
X = rng.random((40, 200)).astype(np.float32)
P = rng.random((12, 200)).astype(np.float32)
T = np.full((12, 12), 1.0 / 12.0, dtype=np.float32)
S = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(y[:40000], 2048)[::512] * np.hanning(2049)[:-1], axis=1)).T.astype(np.float32)
SETUP
import json
times = []
for _ in range(3):
    t = time.perf_counter()
    CALL
    times.append(time.perf_counter() - t)
print(json.dumps({"import_s": t_import - t_start, "times_s": times}))
"""

# Whole processes: import, read a file, MFCC with librosa's settings (top_db=80).
MFCC = {
    # librosa.load's defaults: mono at 22050 Hz.
    "mono_22050": (
        "import librosa\ny, sr = librosa.load(P)\nm = librosa.feature.mfcc(y=y, sr=sr)",
        "import audio_samples as aus\na = aus.io.read(P, as_type=aus.SampleType.F32).to_mono().resample(22050, 'high')\nm = a.mfcc_frames(top_db=80.0)",
    ),
    # The file's own rate, mono.
    "mono_native": (
        "import librosa\ny, sr = librosa.load(P, sr=None)\nm = librosa.feature.mfcc(y=y, sr=sr)",
        "import audio_samples as aus\na = aus.io.read(P, as_type=aus.SampleType.F32).to_mono()\nm = a.mfcc_frames(top_db=80.0)",
    ),
}
MFCC_INPUTS = ("wav16_10s.wav", "wav16_60s.wav", "flac16_60s.flac")


def child_env(pinned):
    env = dict(os.environ)
    if pinned:
        for v in THREAD_VARS:
            env[v] = "1"
    return env


def prefix():
    return ["taskset", "-c", args.core] if args.core else []


def wall(code, env):
    t = time.perf_counter()
    subprocess.run(prefix() + [sys.executable, "-W", "ignore", "-c", code], env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.perf_counter() - t


def evict(root_dirs):
    """Drop the files under ``root_dirs`` from the page cache."""
    for root in root_dirs:
        for d, _, names in os.walk(root):
            for n in names:
                p = os.path.join(d, n)
                try:
                    fd = os.open(p, os.O_RDONLY)
                except OSError:
                    continue
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)


def python_roots():
    import numpy
    import sysconfig

    return [os.path.dirname(os.path.dirname(numpy.__file__)), sysconfig.get_paths()["stdlib"]]


def record(rec, values_ms):
    values_ms = sorted(values_ms)
    rec.update(median_ms=statistics.median(values_ms), min_ms=values_ms[0], max_ms=values_ms[-1],
               runs=len(values_ms), values_ms=[round(v, 3) for v in values_ms])
    print(rec["stage"], {k: v for k, v in rec.items() if k not in ("stage", "values_ms", "runs")}, file=sys.stderr)
    with open(args.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def interleaved_walls(codes, repeats, env, before=None):
    """Runs of each code in turn, so that neither side gets a quieter machine."""
    out = {k: [] for k in codes}
    for code in codes.values():  # warm the page cache and write .pyc files
        if before is None:
            wall(code, env)
    for _ in range(repeats):
        for k, code in codes.items():
            if before is not None:
                before()
            out[k].append(wall(code, env) * 1e3)
    return out


def stage_import():
    env = child_env(bool(args.core))
    res = interleaved_walls(IMPORTS, args.repeats, env)
    for k, v in res.items():
        record({"stage": "import", "name": k, "pinned": bool(args.core)}, v)


def stage_first():
    env = child_env(bool(args.core))
    sig = os.path.join(args.signals, "sig_22050_10s_1ch.f64")
    for name, (lc, oc) in FIRST.items():
        for lib, call in (("librosa", lc), ("ours", oc)):
            firsts, seconds, thirds, imports = [], [], [], []
            for _ in range(args.first_repeats):
                lib_line = "import librosa" if lib == "librosa" else "import audio_samples as aus"
                setup = "" if lib == "librosa" else "a = aus.AudioSamples.new_mono(y, sr)"
                code = FIRST_CHILD.replace("LIB", lib_line).replace("SIG", repr(sig)).replace("SETUP", setup).replace("CALL", call)
                cp = subprocess.run(prefix() + [sys.executable, "-W", "ignore", "-c", code], env=env,
                                    capture_output=True, text=True)
                if cp.returncode != 0:
                    raise SystemExit(f"{lib} {name}: {cp.stderr[-600:]}")
                r = json.loads(cp.stdout.strip().splitlines()[-1])
                firsts.append(r["times_s"][0] * 1e3)
                seconds.append(r["times_s"][1] * 1e3)
                thirds.append(r["times_s"][2] * 1e3)
                imports.append(r["import_s"] * 1e3)
            record({"stage": "first", "name": name, "lib": lib, "second_ms": statistics.median(seconds),
                    "third_ms": statistics.median(thirds), "import_ms": statistics.median(imports),
                    "pinned": bool(args.core)}, firsts)


def mfcc_codes(inp):
    p = repr(os.path.join(args.files, inp))
    out = {}
    for name, (lc, oc) in MFCC.items():
        out[(name, "librosa")] = lc.replace("(P", "(" + p)
        out[(name, "ours")] = oc.replace("(P", "(" + p)
    return out


def stage_mfcc():
    env = child_env(bool(args.core))
    for inp in MFCC_INPUTS:
        res = interleaved_walls(mfcc_codes(inp), args.repeats, env)
        for (name, lib), v in res.items():
            record({"stage": "mfcc", "name": name, "lib": lib, "input": inp, "pinned": bool(args.core)}, v)


def stage_cold():
    env = child_env(bool(args.core))
    roots = python_roots()
    before = lambda: evict(roots)  # noqa: E731
    keep = {k: IMPORTS[k] for k in ("python", "numpy", "librosa", "audio_samples")}
    res = interleaved_walls(keep, args.cold_repeats, env, before)
    for k, v in res.items():
        record({"stage": "cold_import", "name": k, "pinned": bool(args.core)}, v)
    res = interleaved_walls(mfcc_codes("wav16_10s.wav"), args.cold_repeats, env, before)
    for (name, lib), v in res.items():
        record({"stage": "cold_mfcc", "name": name, "lib": lib, "input": "wav16_10s.wav", "pinned": bool(args.core)}, v)


ap = argparse.ArgumentParser()
ap.add_argument("--files", default=os.path.join(HERE, "..", "..", "target", "librosa_bench", "files"))
ap.add_argument("--signals", default=os.path.join(HERE, "..", "..", "target", "librosa_bench", "signals"))
ap.add_argument("--out", required=True)
ap.add_argument("--repeats", type=int, default=30)
ap.add_argument("--first-repeats", type=int, default=7)
ap.add_argument("--cold-repeats", type=int, default=10)
ap.add_argument("--stages", default="import,first,mfcc,cold")
ap.add_argument("--core", default=os.environ.get("CORE", ""))
args = ap.parse_args()

if __name__ == "__main__":
    for stage in args.stages.split(","):
        {"import": stage_import, "first": stage_first, "mfcc": stage_mfcc, "cold": stage_cold}[stage]()
