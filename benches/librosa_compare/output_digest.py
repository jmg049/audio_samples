"""Digest of every case's audio_samples output, to check that an optimisation
does not change results.

    output_digest.py --out before.json [--cases a,b] [--groups g]
    output_digest.py --out after.json
    output_digest.py --compare before.json after.json

Runs the binding call ("ours") of each selected case on a few inputs (f32
and f64, mono and stereo, 10 s at 22050 Hz, plus the fixed inputs) and stores
a SHA-256 of each result's bytes together with its dtype and shape.
``--compare`` lists the cases whose output changed, with the largest
peak-relative difference when both runs saved the arrays (``--save-arrays
DIR``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out")
ap.add_argument("--signals", default=os.path.join(HERE, "..", "..", "target", "librosa_bench", "signals"))
ap.add_argument("--cases", default=None)
ap.add_argument("--groups", default=None)
ap.add_argument("--save-arrays", default=None, help="directory for .npy copies of the outputs")
ap.add_argument("--compare", nargs=2, default=None)
args = ap.parse_args()

CONFIGS = ("22050_10s_mono_f32", "22050_10s_mono_f64", "22050_10s_stereo_f32", "22050_10s_stereo_f64",
           "const_f64", "vec_f32", "vec_f64", "seq_f32", "seq_f64")


def arrays(r):
    """Flatten a result into a list of numpy arrays."""
    if isinstance(r, (tuple, list)):
        out = []
        for x in r:
            out += arrays(x)
        return out
    if isinstance(r, dict):
        out = []
        for k in sorted(r):
            out += arrays(r[k])
        return out
    if hasattr(r, "__array__") or np.isscalar(r):
        return [np.asarray(r)]
    for attr in ("f0", "voiced_flag", "voiced_prob", "tempo", "beats", "path", "cost", "states"):
        if hasattr(r, attr):
            return [np.asarray(getattr(r, a)) for a in
                    ("f0", "voiced_flag", "voiced_prob", "tempo", "beats", "path", "cost", "states") if hasattr(r, a)]
    # A colormap's repr holds its address; its name and colours identify it.
    if hasattr(r, "name") and hasattr(r, "N") and callable(r):
        return [np.asarray(r.name), np.asarray(r(np.linspace(0.0, 1.0, r.N)))]
    return [np.asarray(repr(r))]


def digest(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str((a.dtype.str, a.shape)).encode())
    if a.dtype.kind in "OU":
        h.update(repr(a.tolist()).encode())
    else:
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def run():
    from cases import Ctx, all_cases
    from common import CONFIG_SETS

    cases = all_cases()
    if args.cases:
        cases = [c for c in cases if c.name in set(args.cases.split(","))]
    if args.groups:
        cases = [c for c in cases if c.group in set(args.groups.split(","))]
    res = {}
    for case in cases:
        if case.ours is None:
            continue
        for cfg in CONFIG_SETS[case.configs]:
            if cfg.id not in CONFIGS:
                continue
            key = f"{case.name}@{cfg.id}"
            try:
                outs = arrays(case.ours(Ctx(cfg, args.signals))())
            except Exception as e:
                res[key] = {"error": f"{type(e).__name__}: {e}"}
                continue
            res[key] = {"digest": [digest(a) for a in outs], "dtype": [a.dtype.str for a in outs],
                        "shape": [list(a.shape) for a in outs]}
            if args.save_arrays:
                os.makedirs(args.save_arrays, exist_ok=True)
                for i, a in enumerate(outs):
                    if a.dtype.kind not in "OU":
                        np.save(os.path.join(args.save_arrays, f"{key}.{i}.npy"), np.ascontiguousarray(a))
            print(key, res[key]["digest"], file=sys.stderr)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


def compare():
    a_path, b_path = args.compare
    a, b = json.load(open(a_path)), json.load(open(b_path))
    da = os.path.join(os.path.dirname(a_path), os.path.basename(a_path).rsplit(".", 1)[0])
    db = os.path.join(os.path.dirname(b_path), os.path.basename(b_path).rsplit(".", 1)[0])
    changed = 0
    common = set(a) & set(b)
    for key in sorted(common):
        ra, rb = a.get(key), b.get(key)
        if ra == rb:
            continue
        changed += 1
        line = f"{key}: "
        if ra is None or rb is None or "error" in ra or "error" in rb:
            line += f"{ra} -> {rb}"
        else:
            parts = []
            for i, (x, y) in enumerate(zip(ra["digest"], rb["digest"])):
                if x == y:
                    continue
                fa, fb = os.path.join(da, f"{key}.{i}.npy"), os.path.join(db, f"{key}.{i}.npy")
                if os.path.exists(fa) and os.path.exists(fb):
                    xa, xb = np.load(fa), np.load(fb)
                    if xa.shape != xb.shape:
                        parts.append(f"[{i}] shape {xa.shape} -> {xb.shape}")
                    else:
                        fin = np.isfinite(xa) & np.isfinite(xb)
                        same_nan = np.array_equal(np.isnan(xa), np.isnan(xb))
                        den = float(np.max(np.abs(xa[fin]))) if fin.any() else 1.0
                        rel = float(np.max(np.abs(xa[fin].astype(np.complex128) - xb[fin]))) / (den or 1.0)
                        parts.append(f"[{i}] {ra['dtype'][i]} rel {rel:.2e}{'' if same_nan else ' NaN pattern differs'}")
                else:
                    parts.append(f"[{i}] digest differs")
            line += "; ".join(parts)
        print(line)
    print(f"{changed} of {len(common)} outputs changed "
          f"({len(set(a) - set(b))} only in the first run, {len(set(b) - set(a))} only in the second)")


if args.compare:
    compare()
else:
    run()
