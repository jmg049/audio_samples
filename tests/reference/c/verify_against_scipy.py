#!/usr/bin/env python3
"""Verify the filter_ref harness parameter mappings against SciPy.

For every (lib, family, response) over a parameter grid this compares the
harness output with scipy.signal.{butter,cheby1,cheby2,ellip,bessel}(...,
output='sos', fs=fs) and, for the RBJ biquads, with the Audio EQ Cookbook
formulas (https://www.w3.org/TR/audio-eq-cookbook/).

Metrics per case (maximum over the grid is reported per group):
  imp    max |h_harness[n] - h_ref[n]| over the 2048-sample impulse response
         (h_ref = scipy.signal.sosfilt(ref_sos, impulse))
  cond   conditioning gauge: max |sosfilt(ref_sos reversed) - h_ref| (an
         impulse error of the same size as `cond` is numerical noise, not a
         design difference)
  |dH|   max | |H_harness(w)| - |H_ref(w)| |  (linear) over 4096 sosfreqz bins,
         H_harness evaluated from the harness' own SOS
  dBpb   max |dB difference| over bins where the reference is >= -15 dB
  dB60   max |dB difference| with both responses floored at -60 dB
  dB100  same, floored at -100 dB
  pole   max distance (z-plane) from a harness pole to the nearest reference pole
  out    max |y_harness - sosfilt(ref_sos, x)| for a white-noise input
         (only evaluated on every OUT_EVERY-th case)

Usage:  .venv/bin/python verify_against_scipy.py [--quick]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "build", "filter_ref")
NFREQ = 4096
NIMP = 2048
OUT_EVERY = 7
BATCH = 300

FS_LIST = [44100.0, 48000.0]
CUTOFFS = [50.0, 1000.0, 10000.0, 18000.0]
BANDS = [(300.0, 3400.0), (1000.0, 1200.0), (20.0, 20000.0)]
RP_LIST = [0.1, 0.5, 1.0, 3.0]
RS_LIST = [20.0, 40.0, 60.0, 80.0]
ORDERS = list(range(1, 9))
RESPONSES = ["lowpass", "highpass", "bandpass", "bandstop"]
BESSEL_FAMILIES = ["bessel", "bessel_mag", "bessel_delay", "bessel_native"]
BESSEL_NORMS = ["phase", "mag", "delay"]


# --------------------------------------------------------------- references --

def scipy_sos(family, response, order, fs, f1, f2, rp, rs, norm=None):
    wn = f1 if response in ("lowpass", "highpass") else [f1, f2]
    if family == "butter":
        return signal.butter(order, wn, btype=response, output="sos", fs=fs)
    if family == "cheby1":
        return signal.cheby1(order, rp, wn, btype=response, output="sos", fs=fs)
    if family == "cheby2":
        return signal.cheby2(order, rs, wn, btype=response, output="sos", fs=fs)
    if family == "ellip":
        return signal.ellip(order, rp, rs, wn, btype=response, output="sos", fs=fs)
    if family.startswith("bessel"):
        return signal.bessel(order, wn, btype=response, output="sos", fs=fs, norm=norm)
    raise ValueError(family)


def rbj_sos(family, fs, f0, q, gain_db):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    cs, sn = np.cos(w0), np.sin(w0)
    al = sn / (2 * q)
    if family == "rbj_lowpass":
        b = [(1 - cs) / 2, 1 - cs, (1 - cs) / 2]; a = [1 + al, -2 * cs, 1 - al]
    elif family == "rbj_highpass":
        b = [(1 + cs) / 2, -(1 + cs), (1 + cs) / 2]; a = [1 + al, -2 * cs, 1 - al]
    elif family == "rbj_bandpass":  # constant 0 dB peak gain
        b = [al, 0, -al]; a = [1 + al, -2 * cs, 1 - al]
    elif family == "rbj_bandstop":
        b = [1, -2 * cs, 1]; a = [1 + al, -2 * cs, 1 - al]
    elif family == "rbj_peak":
        b = [1 + al * A, -2 * cs, 1 - al * A]; a = [1 + al / A, -2 * cs, 1 - al / A]
    elif family == "rbj_lowshelf":
        s = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) - (A - 1) * cs + s), 2 * A * ((A - 1) - (A + 1) * cs),
             A * ((A + 1) - (A - 1) * cs - s)]
        a = [(A + 1) + (A - 1) * cs + s, -2 * ((A - 1) + (A + 1) * cs),
             (A + 1) + (A - 1) * cs - s]
    elif family == "rbj_highshelf":
        s = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) + (A - 1) * cs + s), -2 * A * ((A - 1) + (A + 1) * cs),
             A * ((A + 1) + (A - 1) * cs - s)]
        a = [(A + 1) - (A - 1) * cs + s, 2 * ((A - 1) - (A + 1) * cs),
             (A + 1) - (A - 1) * cs - s]
    else:
        raise ValueError(family)
    b = np.array(b) / a[0]
    a = np.array(a) / a[0]
    return np.array([[*b, *a]])


# ----------------------------------------------------------------- metrics --

def metrics(res, ref_sos, x):
    sos = np.array(res["sos"], dtype=float)
    imp = np.zeros(NIMP); imp[0] = 1.0
    h_ref = signal.sosfilt(ref_sos, imp)
    h = np.array(res["impulse"], dtype=float)
    _, H = signal.sosfreqz(sos, worN=NFREQ)
    _, Hr = signal.sosfreqz(ref_sos, worN=NFREQ)
    mag, magr = np.abs(H), np.abs(Hr)
    db = 20 * np.log10(np.maximum(mag, 1e-300))
    dbr = 20 * np.log10(np.maximum(magr, 1e-300))
    p = np.concatenate([np.roots([1, s[4], s[5]]) for s in sos])
    pr = np.concatenate([np.roots(s[3:]) for s in ref_sos])
    out = {
        "imp": float(np.max(np.abs(h - h_ref))),
        # conditioning gauge: same reference SOS, sections applied in reverse
        # order (mathematically identical, numerically not)
        "cond": float(np.max(np.abs(signal.sosfilt(np.ascontiguousarray(ref_sos[::-1]), imp) - h_ref))),
        "dH": float(np.max(np.abs(mag - magr))),
        # dB difference where the reference is >= -15 dB (passband / transition)
        "dBpb": float(np.max(np.abs(db - dbr)[dbr >= -15])) if (dbr >= -15).any() else 0.0,
        # dB difference with both responses floored at -60 / -100 dB
        "dB60": float(np.max(np.abs(np.maximum(db, -60) - np.maximum(dbr, -60)))),
        "dB100": float(np.max(np.abs(np.maximum(db, -100) - np.maximum(dbr, -100)))),
        # max distance from each harness pole to the nearest reference pole
        "pole": float(max(np.min(np.abs(pr - q)) for q in p)) if len(p) == len(pr) else float("inf"),
    }
    if x is not None:
        y_ref = signal.sosfilt(ref_sos, x)
        out["out"] = float(np.max(np.abs(np.array(res["output"], dtype=float) - y_ref)))
    return out


# ------------------------------------------------------------------- cases --

def build_cases(quick):
    orders = [1, 2, 3, 4, 8] if quick else ORDERS
    fss = FS_LIST[:1] if quick else FS_LIST
    rps = [0.5, 3.0] if quick else RP_LIST
    rss = [40.0, 80.0] if quick else RS_LIST
    cases = []
    for lib in ("dspfilters", "liquid"):
        fams = [("butter", [(0.0, 0.0)]),
                ("cheby1", [(rp, 0.0) for rp in rps]),
                ("cheby2", [(0.0, rs) for rs in rss]),
                ("ellip", [(rp, rs) for rp in rps for rs in rss])]
        fams += [(bf, [(0.0, 0.0)]) for bf in BESSEL_FAMILIES]
        for (fam, ripples), resp, order, fs in itertools.product(fams, RESPONSES, orders, fss):
            freqs = [(c, 0.0) for c in CUTOFFS] if resp in ("lowpass", "highpass") else BANDS
            for (f1, f2), (rp, rs) in itertools.product(freqs, ripples):
                cases.append(dict(lib=lib, family=fam, response=resp, order=order, fs=fs,
                                  f1=f1, f2=f2, rp=rp, rs=rs, gain_db=0.0, q=0.0))
    # RBJ biquads (DSPFilters only; liquid has none and must return ok=false)
    for lib in ("dspfilters", "liquid"):
        for fam in ("rbj_lowpass", "rbj_highpass", "rbj_bandpass", "rbj_bandstop",
                    "rbj_peak", "rbj_lowshelf", "rbj_highshelf"):
            gains = [-12.0, -3.0, 0.0, 6.0, 15.0] if fam in ("rbj_peak", "rbj_lowshelf",
                                                             "rbj_highshelf") else [0.0]
            libfs = fss if lib == "dspfilters" else fss[:1]
            for fs, f0, q, g in itertools.product(libfs, CUTOFFS, [0.5, 0.7071067811865476, 2.0, 10.0], gains):
                cases.append(dict(lib=lib, family=fam, response="-", order=2, fs=fs,
                                  f1=f0, f2=0.0, rp=0.0, rs=0.0, gain_db=g, q=q))
    for i, c in enumerate(cases):
        c["id"] = f"c{i}"
    return cases


def well_conditioned(c):
    """Subset away from the numerically hard corners (very low cutoff, very
    narrow or very wide bands) where float32 coefficient quantisation, not
    the parameter mapping, dominates the error."""
    if c["family"].startswith("rbj_"):
        return True
    if c["response"] in ("lowpass", "highpass"):
        return c["f1"] in (1000.0, 10000.0)
    return (c["f1"], c["f2"]) == (300.0, 3400.0)


def run_harness(lines):
    p = subprocess.run([HARNESS], input="\n".join(lines) + "\n", capture_output=True, text=True,
                       check=True)
    return [json.loads(l) for l in p.stdout.splitlines() if l.strip()]


def fmt(v):
    return f"{v:9.2e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller grid")
    args = ap.parse_args()
    if not os.path.exists(HARNESS):
        sys.exit(f"harness not built: run {HERE}/build.sh")

    rng = np.random.default_rng(1234)
    noise = rng.standard_normal(4096)
    tmp = tempfile.NamedTemporaryFile(suffix=".f64", delete=False)
    noise.astype("<f8").tofile(tmp.name)

    cases = build_cases(args.quick)
    groups = defaultdict(lambda: defaultdict(float))   # exact_convention == true
    flagged = defaultdict(lambda: defaultdict(float))  # exact_convention == false
    wellc = defaultdict(lambda: defaultdict(float))    # exact, well-conditioned subset
    counts = defaultdict(lambda: [0, 0, 0])  # ok, fail, inexact
    errors = defaultdict(set)
    bessel = defaultdict(lambda: defaultdict(float))  # (lib,fam,resp) -> norm -> max dB60
    worst = {}

    for start in range(0, len(cases), BATCH):
        chunk = cases[start:start + BATCH]
        lines = []
        for k, c in enumerate(chunk):
            use_x = (start + k) % OUT_EVERY == 0
            c["_x"] = use_x
            lines.append(" ".join(str(v) for v in (
                c["id"], c["lib"], c["family"], c["response"], c["order"], repr(c["fs"]),
                repr(c["f1"]), repr(c["f2"]), repr(c["rp"]), repr(c["rs"]), repr(c["gain_db"]),
                repr(c["q"]), tmp.name if use_x else "-")))
        results = run_harness(lines)
        assert len(results) == len(chunk)
        for c, res in zip(chunk, results):
            assert res["id"] == c["id"]
            key = (c["lib"], c["family"], c["response"])
            if not res["ok"]:
                counts[key][1] += 1
                errors[key].add(res["error"])
                continue
            counts[key][0] += 1
            exact = res["exact_convention"]
            if not exact:
                counts[key][2] += 1
            x = noise if c["_x"] else None
            if c["family"].startswith("rbj_"):
                ref = rbj_sos(c["family"], c["fs"], c["f1"], c["q"], c["gain_db"])
                mm = metrics(res, ref, x)
            elif c["family"].startswith("bessel"):
                target = {"bessel": "phase", "bessel_mag": "mag", "bessel_delay": "delay",
                          "bessel_native": None}[c["family"]]
                mm = None
                for norm in BESSEL_NORMS:
                    ref = scipy_sos("bessel", c["response"], c["order"], c["fs"], c["f1"],
                                    c["f2"], 0, 0, norm=norm)
                    m = metrics(res, ref, x if norm == target else None)
                    bessel[key][norm] = max(bessel[key][norm], m["dB60"])
                    if norm == target:
                        mm = m
                if mm is None:  # native: no scipy target; skip main table
                    continue
            else:
                ref = scipy_sos(c["family"], c["response"], c["order"], c["fs"], c["f1"],
                                c["f2"], c["rp"], c["rs"])
                mm = metrics(res, ref, x)
            if exact and well_conditioned(c):
                gw = wellc[key]
                for k, v in mm.items():
                    gw[k] = max(gw[k], v)
            g = groups[key] if exact else flagged[key]
            for k, v in mm.items():
                if v > g[k]:
                    g[k] = v
                    if k == "dB60" and exact:
                        worst[key] = {kk: c[kk] for kk in ("order", "fs", "f1", "f2", "rp", "rs",
                                                          "q", "gain_db")}
                        worst[key]["note"] = res["note"]
        print(f"  {min(start + BATCH, len(cases))}/{len(cases)} cases", file=sys.stderr)
    os.unlink(tmp.name)

    print(f"\nscipy {__import__('scipy').__version__}, numpy {np.__version__}; "
          f"{len(cases)} cases\n")
    cols = ("imp", "cond", "dH", "dBpb", "dB60", "dB100", "pole", "out")
    hdr = (f"{'lib':10} {'family':14} {'response':9} {'ok':>4} {'fail':>4} {'inex':>4} " +
           " ".join(f"{c:>9}" for c in cols))
    print("Cases with exact_convention == true (max over grid):")
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(counts):
        ok, fail, inex = counts[key]
        g = groups.get(key)
        vals = " ".join(fmt(g.get(k, float("nan"))) for k in cols) if g else "        -" * len(cols)
        print(f"{key[0]:10} {key[1]:14} {key[2]:9} {ok:4d} {fail:4d} {inex:4d} {vals}")
    if flagged:
        print("\nCases with exact_convention == false (flagged by the harness), max over grid:")
        print(hdr)
        print("-" * len(hdr))
        for key in sorted(flagged):
            ok, fail, inex = counts[key]
            g = flagged[key]
            vals = " ".join(fmt(g.get(k, float("nan"))) for k in cols)
            print(f"{key[0]:10} {key[1]:14} {key[2]:9} {ok:4d} {fail:4d} {inex:4d} {vals}")

    print("\nWell-conditioned subset (exact_convention == true; LP/HP cutoff 1 kHz or 10 kHz, "
          "BP/BS band [300, 3400] Hz; all orders/ripples):")
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(wellc):
        ok, fail, inex = counts[key]
        g = wellc[key]
        vals = " ".join(fmt(g.get(k, float("nan"))) for k in cols)
        print(f"{key[0]:10} {key[1]:14} {key[2]:9} {ok:4d} {fail:4d} {inex:4d} {vals}")

    print("\nBessel: max dB60 difference vs each scipy norm")
    print(f"{'lib':10} {'family':14} {'response':9} " + " ".join(f"{n:>9}" for n in BESSEL_NORMS))
    for key in sorted(bessel):
        print(f"{key[0]:10} {key[1]:14} {key[2]:9} " +
              " ".join(fmt(bessel[key][n]) for n in BESSEL_NORMS))

    print("\nWorst-case parameters (by dB60) per group:")
    for key in sorted(worst):
        print(f"  {key}: {worst[key]}")
    if errors:
        print("\nErrors:")
        for key in sorted(errors):
            for e in sorted(errors[key]):
                print(f"  {key}: {e}")


if __name__ == "__main__":
    main()
