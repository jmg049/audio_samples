"""Check that each case's two calls compute the same thing.

Runs the librosa and the audio_samples call of every selected case once on one
configuration and prints the result shapes and the peak-relative difference
``max|a - b| / max|b|`` of the first array of each result. A case whose shapes
differ, or whose difference is large, is not a like-for-like comparison.

    check_cases.py [--cases a,b] [--groups g] [--config 22050_10s_mono_f32]
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from cases import Ctx, all_cases  # noqa: E402
from common import CONFIG_SETS, parse_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--signals", default=os.path.join(HERE, "..", "..", "target", "librosa_bench", "signals"))
ap.add_argument("--cases", default=None)
ap.add_argument("--groups", default=None)
ap.add_argument("--config", default="22050_10s_mono_f32",
                help="audio config to use; cases whose config set lacks it use their first config")
args = ap.parse_args()


def first_array(r):
    while isinstance(r, (tuple, list)) and r and not np.isscalar(r[0]):
        r = r[0]
    if hasattr(r, "__array__") and not isinstance(r, np.ndarray):
        r = np.asarray(r)
    return np.asarray(r)


def main():
    cases = all_cases()
    if args.cases:
        want = set(args.cases.split(","))
        cases = [c for c in cases if c.name in want]
    if args.groups:
        cases = [c for c in cases if c.group in set(args.groups.split(","))]
    bad = 0
    for case in cases:
        cfgs = CONFIG_SETS[case.configs]
        cfg = next((c for c in cfgs if c.id == args.config), cfgs[0])
        if case.ours is None:
            print(f"{case.name:34} {cfg.id:22} (no binding)")
            continue
        try:
            a = first_array(case.librosa(Ctx(cfg, args.signals))())
            b = first_array(case.ours(Ctx(cfg, args.signals))())
        except Exception as e:
            print(f"{case.name:34} {cfg.id:22} ERROR {type(e).__name__}: {e}")
            bad += 1
            continue
        if a.dtype.kind in "USO" or b.dtype.kind in "USO":
            diff = "equal" if np.array_equal(np.squeeze(a), np.squeeze(b)) else "DIFF"
        elif a.size and b.size:
            sa, sb = np.squeeze(a), np.squeeze(b)
            if sa.shape == sb.shape:
                denom = float(np.max(np.abs(sa))) or 1.0
                rel = float(np.max(np.abs(sa.astype(np.complex128) - sb.astype(np.complex128)))) / denom
                diff = f"rel {rel:.2e}"
            else:
                diff = "SHAPE MISMATCH"
                bad += 1
        else:
            diff = "-"
        print(f"{case.name:34} {cfg.id:22} librosa {a.dtype}{list(a.shape)}  ours {b.dtype}{list(b.shape)}  {diff}")
    sys.exit(1 if bad else 0)


main()
