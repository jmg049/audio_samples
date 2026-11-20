#!/usr/bin/env python3
"""Generate golden reference fixtures for the audio_samples analysis suite.

librosa (pinned in ``pyproject.toml`` next to this file) is run on a small set of
deterministic test signals, and its outputs are written to
``tests/fixtures/analysis/*.json``. The Rust tests in ``tests/analysis_reference/``
and the Python binding tests (``audio_samples_python/tests/test_librosa_*.py``)
compare audio_samples against them.

Layout
------
* ``signals.json``: the shared input signals, stored exactly as 16-bit PCM
  integers ``k`` (sample value ``k / pcm_scale``), so they are exact in i16,
  f32 and f64 audio.
* ``<area>.json``: one file per case module in ``cases/``. A case module defines
  ``NAME`` and ``build(signals, sr) -> dict``.

Where librosa itself delegates to another implementation (``soxr`` for
resampling, ``scipy.fft``/``numpy.fft`` for transforms, ``scipy.signal`` for
filters) the case module says so next to the case.

Usage
-----
    uv run --project tests/reference/analysis tests/reference/analysis/generate_analysis_fixtures.py
    uv run --project tests/reference/analysis tests/reference/analysis/generate_analysis_fixtures.py --check
    uv run --project tests/reference/analysis tests/reference/analysis/generate_analysis_fixtures.py --only spectrum

``--check`` regenerates in memory and fails if the committed fixtures differ by
more than a relative 1e-9 (librosa/numba/BLAS reductions are not bit-identical
across machines, so an exact comparison would be flaky). The generator always
runs single-threaded, so the result does not depend on how many CPUs it can use.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import pkgutil
import platform
import subprocess
import sys
import warnings
from pathlib import Path

# BLAS splits a matrix product into one block per thread, and the rounding of a
# float32 product depends on those blocks. With OpenBLAS's default of one thread
# per available CPU, the fixtures changed by up to 2e-7 relative with the CPU
# affinity of the process (24 CPUs against 8). These must be set before NumPy
# loads, and the child processes below inherit them.
for _variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_variable] = "1"
# The kernels differ in rounding too, so x86-64 machines all use the AVX2 ones.
if platform.machine() in ("x86_64", "AMD64"):
    os.environ["OPENBLAS_CORETYPE"] = "Haswell"

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "analysis"
sys.path.insert(0, str(HERE))

import librosa  # noqa: E402
import scipy  # noqa: E402

import cases  # noqa: E402

SR = 22050
RNG_SEED = 20261009
CHECK_RTOL = 1e-9
CHECK_ATOL = 1e-12


# ---------------------------------------------------------------------------
# Test signals
# ---------------------------------------------------------------------------


PCM_SCALE = 32768.0


def _f32(x: np.ndarray) -> np.ndarray:
    """Quantise to the 16-bit PCM grid (k / 32768).

    Every sample is then exact in f32, f64 and i16 audio, and is stored in
    ``signals.json`` as a small integer, which keeps the file compact.
    """
    k = np.clip(np.round(np.asarray(x, dtype=np.float64) * PCM_SCALE), -32768, 32767)
    return k / PCM_SCALE


def make_signals(sr: int = SR) -> dict[str, np.ndarray]:
    """Deterministic test signals. Keep them short: fixtures must stay small."""
    rng = np.random.default_rng(RNG_SEED)
    n = sr  # 1 s
    t = np.arange(n) / sr
    sig: dict[str, np.ndarray] = {}

    # Harmonic mixture + linear chirp: the general-purpose spectral signal.
    tones = 0.3 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 330.0 * t + 0.3)
    tones += 0.1 * np.sin(2 * np.pi * 1760.0 * t + 1.1)
    chirp = 0.15 * np.sin(2 * np.pi * (100.0 * t + 0.5 * (4000.0 - 100.0) * t**2))
    sig["mix"] = _f32(tones + chirp)

    # White noise (seeded).
    sig["noise"] = _f32(0.25 * rng.uniform(-1.0, 1.0, n))

    # Note sequence: 8 decaying harmonic notes, 0.125 s apart (onsets / chroma / pitch).
    notes_hz = [261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 493.88, 523.25]
    seq = np.zeros(n)
    step = n // len(notes_hz)
    for i, f in enumerate(notes_hz):
        tt = np.arange(n - i * step) / sr
        env = np.exp(-tt * 12.0)
        tone = sum((0.5 / k) * np.sin(2 * np.pi * k * f * tt) for k in (1, 2, 3))
        seq[i * step :] += env * tone
    sig["notes"] = _f32(0.4 * seq / np.max(np.abs(seq)))

    # Click track at 120 BPM over 3 s (beat / tempo), plus a little noise.
    n3 = 3 * sr
    clicks = librosa.clicks(times=np.arange(0.25, 3.0, 0.5), sr=sr, length=n3, click_duration=0.05)
    sig["clicks120"] = _f32(0.5 * clicks + 0.01 * rng.uniform(-1.0, 1.0, n3))

    # Speech-like: silence - burst - silence (trim / split).
    burst = np.zeros(n)
    a, b = int(0.3 * sr), int(0.7 * sr)
    burst[a:b] = 0.5 * np.sin(2 * np.pi * 300.0 * t[: b - a]) * np.hanning(b - a)
    burst += 1e-4 * rng.uniform(-1.0, 1.0, n)
    sig["burst"] = _f32(burst)

    # Stereo: left = mix, right = noise (multichannel broadcasting).
    sig["stereo_l"] = sig["mix"]
    sig["stereo_r"] = sig["noise"]
    return sig


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def to_jsonable(obj):
    """Convert numpy containers to JSON. Complex arrays become {"re": .., "im": ..}."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if np.iscomplexobj(obj):
            return {"re": to_jsonable(obj.real), "im": to_jsonable(obj.imag)}
        if obj.dtype == bool:
            return obj.tolist()
        if np.issubdtype(obj.dtype, np.integer):
            return obj.astype(np.int64).tolist()
        return np.asarray(obj, dtype=np.float64).tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, complex) or isinstance(obj, np.complexfloating):
        return {"re": float(obj.real), "im": float(obj.imag)}
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError("non-finite value in fixture")
    return obj


def dumps(obj) -> str:
    return json.dumps(to_jsonable(obj), separators=(",", ":"), allow_nan=False) + "\n"


def close(a, b, path="") -> list[str]:
    """Recursive comparison with CHECK_RTOL / CHECK_ATOL for floats."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys differ"]
        out = []
        for k in a:
            out += close(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        if a and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in a) and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in b
        ):
            xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            if not np.allclose(xa, xb, rtol=CHECK_RTOL, atol=CHECK_ATOL * max(1.0, np.max(np.abs(xb)))):
                return [f"{path}: max abs diff {np.max(np.abs(xa - xb)):.3e}"]
            return []
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += close(x, y, f"{path}[{i}]")
        return out
    if isinstance(a, float) or isinstance(b, float):
        if not math.isclose(a, b, rel_tol=CHECK_RTOL, abs_tol=CHECK_ATOL):
            return [f"{path}: {a} != {b}"]
        return []
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if committed fixtures are out of date")
    ap.add_argument("--only", action="append", default=[], help="only build these case modules")
    ap.add_argument("--emit-module", help=argparse.SUPPRESS)
    args = ap.parse_args()

    warnings.simplefilter("ignore", FutureWarning)
    meta = {
        "generator": "tests/reference/analysis/generate_analysis_fixtures.py",
        "librosa": librosa.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sr": SR,
    }
    signals = make_signals(SR)
    pcm = {k: np.round(v * PCM_SCALE).astype(np.int64) for k, v in signals.items()}
    outputs: dict[str, dict] = {
        "signals.json": {"meta": meta, "sr": SR, "pcm_scale": PCM_SCALE, "signals": pcm}
    }

    modules = sorted(m.name for m in pkgutil.iter_modules(cases.__path__) if not m.name.startswith("_"))
    if args.emit_module:
        # Child process: build one case module and print its JSON on stdout.
        mod = importlib.import_module(f"cases.{args.emit_module}")
        sys.stdout.write(dumps(mod.build(signals, SR)))
        return 0

    for mod_name in modules:
        if args.only and mod_name not in args.only:
            continue
        mod = importlib.import_module(f"cases.{mod_name}")
        name = getattr(mod, "NAME", mod_name)
        print(f"building {name} ...", file=sys.stderr)
        # Each case module runs in a fresh interpreter: librosa / numba /
        # scipy.fft keep process-wide caches, and building one module first
        # was measured to perturb another (icqt) by ~2e-7, which made the
        # fixtures depend on generation order.
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--emit-module", mod_name],
            capture_output=True,
            text=True,
            check=False,
        )
        if child.returncode != 0:
            sys.stderr.write(child.stderr)
            raise SystemExit(f"case module {mod_name} failed")
        outputs[f"{name}.json"] = {"meta": meta, **json.loads(child.stdout)}

    stale = []
    FIXTURES.mkdir(parents=True, exist_ok=True)
    total = 0
    for fname, obj in outputs.items():
        path = FIXTURES / fname
        text = dumps(obj)
        total += len(text)
        if args.check:
            if not path.exists():
                stale.append(f"{fname} (missing)")
                continue
            committed = json.loads(path.read_text())
            fresh = json.loads(text)
            committed.pop("meta", None)
            fresh.pop("meta", None)
            diffs = close(fresh, committed, fname)
            if diffs:
                stale.append(f"{fname}: " + "; ".join(diffs[:5]))
        else:
            path.write_text(text)
            print(f"wrote {path.relative_to(ROOT)} ({len(text) / 1e6:.2f} MB)")
    print(f"total fixture size {total / 1e6:.2f} MB", file=sys.stderr)
    if stale:
        print("stale fixtures:\n  " + "\n  ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
