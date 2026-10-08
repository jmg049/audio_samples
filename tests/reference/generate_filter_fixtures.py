#!/usr/bin/env python3
"""Generate golden reference fixtures for the audio_samples filter test suite.

Every fixture is computed by long-standing, independently maintained
implementations and written to ``tests/fixtures/filters/*.json``. The Rust test
``tests/filter_reference.rs`` compares audio_samples against them.

References
----------
* SciPy ``scipy.signal`` (``iirfilter``/``butter``/``cheby1``/``cheby2``/
  ``ellip``/``bessel``, ``sosfilt``, ``sosfiltfilt``, ``sosfreqz``, ``lfilter``)
  -- the primary oracle; its kernels are C/Cython.
* FFmpeg ``af_biquads`` (C) -- RBJ-cookbook biquads (equalizer, shelves,
  low/high/band-pass, notch), run on the CLI with ``precision=f64``.
* DSPFilters (Vinnie Falco, C++, double precision) and liquid-dsp (C, single
  precision) via the harness in ``tests/reference/c`` -- independent IIR design
  implementations (only when the harness has been built; see its README).
* The RBJ Audio EQ Cookbook formulas, transcribed directly below.

Before writing anything the script cross-checks the references against each
other and aborts if they disagree, so a fixture is only produced when at least
two independent implementations agree on the expected value.

Usage
-----
    uv run --project tests/reference tests/reference/generate_filter_fixtures.py
    uv run --project tests/reference tests/reference/generate_filter_fixtures.py --check

``--check`` regenerates in memory and fails if the committed fixtures differ.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy
import scipy.signal as sg

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "filters"
HARNESS = ROOT / "tests" / "reference" / "c" / "build" / "filter_ref"

SIGNAL_LEN = 1024
TD_LEN = 512  # samples stored per time-domain reference
IMPULSE_LEN = 256
RNG_SEED = 20261008


# ---------------------------------------------------------------------------
# Test signals
# ---------------------------------------------------------------------------


def make_signals() -> dict[str, list[float]]:
    """Deterministic broadband test signals (stored in the fixtures)."""
    rng = np.random.default_rng(RNG_SEED)
    n = np.arange(SIGNAL_LEN)
    noise = rng.uniform(-0.5, 0.5, SIGNAL_LEN)
    # Log chirp 20 Hz -> 0.45 fs (normalised to fs = 1) plus noise and a DC
    # offset so DC handling and initial conditions are exercised.
    chirp = 0.4 * sg.chirp(n, f0=20 / 48000, t1=SIGNAL_LEN, f1=0.45, method="logarithmic")
    mix = chirp + 0.3 * noise + 0.1
    impulse = np.zeros(SIGNAL_LEN)
    impulse[0] = 1.0
    return {
        "mix": mix.tolist(),
        "impulse": impulse.tolist(),
    }


# ---------------------------------------------------------------------------
# IIR design cases
# ---------------------------------------------------------------------------

FAMILY_PARAMS = {
    "butter": [{}],
    "cheby1": [{"rp": 0.1}, {"rp": 1.0}, {"rp": 3.0}],
    "cheby2": [{"rs": 20.0}, {"rs": 40.0}, {"rs": 80.0}],
    "ellip": [{"rp": 0.1, "rs": 40.0}, {"rp": 1.0, "rs": 60.0}, {"rp": 3.0, "rs": 80.0}],
    "bessel": [{"norm": "mag"}, {"norm": "phase"}, {"norm": "delay"}],
}

# (fs, single-edge frequencies, band edges)
FREQ_CONFIGS = [
    (44100.0, [30.0, 1000.0, 15000.0, 21000.0], [[300.0, 3400.0], [1000.0, 1200.0], [20.0, 20000.0]]),
    (48000.0, [100.0, 5000.0], [[50.0, 60.0], [5000.0, 15000.0]]),
    (8000.0, [300.0, 3400.0], [[300.0, 3400.0]]),
    (96000.0, [20.0, 40000.0], [[20.0, 40000.0]]),
]


def scipy_design(case: dict) -> np.ndarray:
    fam, order, fs, wn = case["family"], case["order"], case["fs"], case["wn"]
    btype = case["response"]
    w = wn[0] if len(wn) == 1 else wn
    if fam == "butter":
        return sg.butter(order, w, btype, fs=fs, output="sos")
    if fam == "cheby1":
        return sg.cheby1(order, case["rp"], w, btype, fs=fs, output="sos")
    if fam == "cheby2":
        return sg.cheby2(order, case["rs"], w, btype, fs=fs, output="sos")
    if fam == "ellip":
        return sg.ellip(order, case["rp"], case["rs"], w, btype, fs=fs, output="sos")
    if fam == "bessel":
        return sg.bessel(order, w, btype, norm=case["norm"], fs=fs, output="sos")
    raise ValueError(fam)


def design_cases() -> list[dict]:
    cases: list[dict] = []

    def add(family, response, order, fs, wn, params):
        cid = f"{family}_{response}_o{order}_fs{int(fs)}_" + "-".join(f"{x:g}" for x in wn)
        for k, v in params.items():
            cid += f"_{k}{v:g}" if isinstance(v, float) else f"_{k}{v}"
        cases.append(
            {"id": cid, "family": family, "response": response, "order": order, "fs": fs, "wn": wn, **params}
        )

    for family, plist in FAMILY_PARAMS.items():
        for response in ("lowpass", "highpass", "bandpass", "bandstop"):
            band = response in ("bandpass", "bandstop")
            base_wn = [300.0, 3400.0] if band else [1000.0]
            # 1. Order sweep at a base configuration (first parameter set).
            for order in list(range(1, 9)) + [10, 12, 16, 20, 24]:
                if band and order > 12:
                    continue  # band designs double the order
                add(family, response, order, 44100.0, base_wn, plist[0])
            # 2. Frequency / sample-rate sweep at orders 2 and 5.
            for fs, singles, bands in FREQ_CONFIGS:
                for wn in (bands if band else [[f] for f in singles]):
                    for order in (2, 5):
                        if fs == 44100.0 and wn == base_wn:
                            continue
                        add(family, response, order, fs, wn, plist[0])
            # 3. Parameter sweep (ripple / attenuation / Bessel norm) at order 4 and 7.
            for params in plist[1:]:
                for order in (4, 7):
                    add(family, response, order, 48000.0, [100.0, 5000.0] if band else [5000.0], params)
    return cases


def response_grid(fs: float) -> np.ndarray:
    nyq = fs / 2
    grid = np.unique(np.concatenate([[0.0], np.geomspace(5.0, nyq * 0.999, 46), [nyq]]))
    return grid


# ---------------------------------------------------------------------------
# C/C++ harness (DSPFilters, liquid-dsp)
# ---------------------------------------------------------------------------


USE_HARNESS = True


def run_harness(lines: list[str]) -> dict[str, dict]:
    if not USE_HARNESS or not HARNESS.exists():
        return {}
    proc = subprocess.run([str(HARNESS)], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
    out = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def harness_family(case: dict) -> str:
    if case["family"] == "bessel":
        return {"mag": "bessel_mag", "phase": "bessel_phase", "delay": "bessel_delay"}[case["norm"]]
    return case["family"]


def harness_line(cid: str, lib: str, case: dict) -> str:
    wn = case["wn"]
    f1 = wn[0]
    f2 = wn[1] if len(wn) > 1 else 0.0
    return " ".join(
        str(x)
        for x in [
            cid,
            lib,
            harness_family(case),
            case["response"],
            case["order"],
            case["fs"],
            f1,
            f2,
            case.get("rp", 0.0),
            case.get("rs", 0.0),
            0.0,
            0.0,
            "-",
        ]
    )


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-300))


def magnitude_agreement_db(h_ref: np.ndarray, h_other: np.ndarray, floor_db: float) -> float:
    """Max |dB difference| over the points where the reference is above floor."""
    ref_db, oth_db = db(h_ref), db(h_other)
    mask = ref_db > floor_db
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(ref_db[mask] - oth_db[mask])))


# Cross-library agreement on |H| (dB), evaluated where the SciPy response is
# above FLOOR_DB. Tolerances reflect each library's documented precision:
# DSPFilters is double precision (elliptic poles come from a Bairstow solver
# stopping at 1e-6, hence the looser elliptic bound); liquid-dsp is float32.
CROSS_FLOOR_DB = {"dspfilters": -60.0, "liquid": -40.0}


def liquid_well_conditioned(case: dict) -> bool:
    """liquid-dsp designs and filters in float32; its prototypes lose accuracy
    for edges very close to DC / Nyquist and for extremely narrow or wide bands
    (see tests/reference/c/README.md), so it is only used as a reference for
    well-conditioned specifications."""
    rel = [f / case["fs"] for f in case["wn"]]
    if not all(0.005 <= r <= 0.4 for r in rel):
        return False
    if len(rel) == 2 and not (rel[1] / rel[0] <= 20 and rel[1] - rel[0] >= 0.005):
        return False
    # Band designs double the order; keep the float32 cascade short.
    return case["order"] <= (4 if len(rel) == 2 else 8)


def cross_tol_db(lib: str, family: str) -> float:
    if lib == "dspfilters":
        return 2e-3 if family == "ellip" else 1e-6
    return {"butter": 5e-3, "bessel": 5e-3, "cheby1": 1e-2}.get(family, 5e-2)


def build_iir_designs(time_domain_ids: set[str]) -> tuple[dict, dict]:
    signals = make_signals()
    cases = design_cases()

    harness_lines = []
    for case in cases:
        for lib in ("dspfilters", "liquid"):
            harness_lines.append(harness_line(f"{case['id']}@{lib}", lib, case))
    harness = run_harness(harness_lines)

    out_cases = []
    td_cases = []
    agreement = {"dspfilters": [], "liquid": []}
    for case in cases:
        sos = scipy_design(case)
        freqs = response_grid(case["fs"])
        _, h = sg.sosfreqz(sos, worN=freqs, fs=case["fs"])
        rec = dict(case)
        rec["sos"] = sos.tolist()
        rec["freqs"] = freqs.tolist()
        rec["h_re"] = h.real.tolist()
        rec["h_im"] = h.imag.tolist()
        refs = {}
        for lib in ("dspfilters", "liquid"):
            hr = harness.get(f"{case['id']}@{lib}")
            if not hr or not hr.get("ok"):
                continue
            if lib == "liquid" and not liquid_well_conditioned(case):
                continue
            hsos = np.asarray(hr["sos"], dtype=float)
            _, h_lib = sg.sosfreqz(hsos, worN=freqs, fs=case["fs"])
            dev = magnitude_agreement_db(h, h_lib, CROSS_FLOOR_DB[lib])
            exact = bool(hr.get("exact_convention", False))
            tol = cross_tol_db(lib, case["family"])
            agreement[lib].append((case["id"], dev, exact, tol, case["family"]))
            if exact:
                refs[lib] = {"mag_db": db(h_lib).tolist(), "max_dev_vs_scipy_db": dev, "tol_db": tol,
                             "floor_db": CROSS_FLOOR_DB[lib]}
        if refs:
            rec["cross_refs"] = refs
        out_cases.append(rec)

        if case["id"] in time_domain_ids:
            td = {"id": case["id"], "design": {k: case[k] for k in case if k != "id"}}
            mix = np.asarray(signals["mix"][:TD_LEN])
            td["sosfilt_mix"] = sg.sosfilt(sos, mix).tolist()
            td["sosfilt_impulse"] = sg.sosfilt(sos, np.asarray(signals["impulse"][:IMPULSE_LEN])).tolist()
            td["sosfiltfilt_mix"] = sg.sosfiltfilt(sos, mix).tolist()
            td_cases.append(td)

    # Abort if an exact-convention cross reference disagrees with SciPy.
    failures = [
        (lib, cid, dev, tol)
        for lib, rows in agreement.items()
        for cid, dev, exact, tol, _fam in rows
        if exact and dev > tol
    ]
    summary = {}
    for lib, rows in agreement.items():
        per_family = {}
        for _cid, dev, exact, _tol, fam in rows:
            f = per_family.setdefault(fam, {"compared": 0, "exact_convention": 0, "max_dev_db": 0.0})
            f["compared"] += 1
            if exact:
                f["exact_convention"] += 1
                f["max_dev_db"] = max(f["max_dev_db"], dev)
        summary[lib] = per_family
    if failures:
        for f in failures:
            print("CROSS-CHECK FAILURE", f, file=sys.stderr)
        raise SystemExit(f"{len(failures)} cross-reference disagreements; refusing to write fixtures")

    designs = {
        "description": "IIR designs: SciPy SOS coefficients and complex frequency response, "
        "with independent DSPFilters / liquid-dsp magnitude responses where their parameter "
        "conventions match SciPy exactly.",
        "cross_check_summary": summary,
        "cases": out_cases,
    }
    time_domain = {
        "description": "scipy.signal.sosfilt / sosfiltfilt outputs for a subset of designs.",
        "signals": signals,
        "cases": td_cases,
    }
    return designs, time_domain


def time_domain_subset(cases: list[dict]) -> set[str]:
    """Representative designs for time-domain comparisons."""
    chosen = set()
    for c in cases:
        fam, resp, order, fs, wn = c["family"], c["response"], c["order"], c["fs"], c["wn"]
        if fs == 44100.0 and wn in ([1000.0], [300.0, 3400.0]) and order in (1, 2, 3, 4, 8, 12):
            if fam == "butter" or order in (2, 3, 8):
                chosen.add(c["id"])
        if fs == 48000.0 and wn in ([100.0], [50.0, 60.0]) and order == 5:
            chosen.add(c["id"])  # low / narrow: poles very close to z = 1
        if fs == 44100.0 and wn == [21000.0] and order == 5:
            chosen.add(c["id"])  # near Nyquist
    return chosen


# ---------------------------------------------------------------------------
# RBJ cookbook biquads (parametric EQ)
# ---------------------------------------------------------------------------


def rbj(kind: str, f0: float, fs: float, q: float, gain_db: float) -> tuple[list[float], list[float]]:
    """Coefficients straight from the RBJ Audio EQ Cookbook (W3C note)."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * f0 / fs
    c, s = math.cos(w0), math.sin(w0)
    alpha = s / (2 * q)
    sa = 2 * math.sqrt(A) * alpha
    if kind == "lowpass":
        b = [(1 - c) / 2, 1 - c, (1 - c) / 2]
        a = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
        a = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == "bandpass":  # constant 0 dB peak gain
        b = [alpha, 0.0, -alpha]
        a = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == "notch":
        b = [1.0, -2 * c, 1.0]
        a = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * c, 1 - alpha * A]
        a = [1 + alpha / A, -2 * c, 1 - alpha / A]
    elif kind == "lowshelf":
        b = [A * ((A + 1) - (A - 1) * c + sa), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sa)]
        a = [(A + 1) + (A - 1) * c + sa, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sa]
    elif kind == "highshelf":
        b = [A * ((A + 1) + (A - 1) * c + sa), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sa)]
        a = [(A + 1) - (A - 1) * c + sa, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sa]
    else:
        raise ValueError(kind)
    a0 = a[0]
    return [x / a0 for x in b], [x / a0 for x in a]


FFMPEG_FILTER = {
    "lowpass": "lowpass=f={f}:p=2:t=q:w={q}",
    "highpass": "highpass=f={f}:p=2:t=q:w={q}",
    "bandpass": "bandpass=f={f}:t=q:w={q}:csg=0",
    "notch": "bandreject=f={f}:t=q:w={q}",
    "peak": "equalizer=f={f}:t=q:w={q}:g={g}",
    "lowshelf": "lowshelf=f={f}:t=q:w={q}:g={g}",
    "highshelf": "highshelf=f={f}:t=q:w={q}:g={g}",
}


def ffmpeg_biquad(kind: str, f0: float, fs: float, q: float, g: float, x: np.ndarray) -> np.ndarray | None:
    if shutil.which("ffmpeg") is None:
        return None
    flt = FFMPEG_FILTER[kind].format(f=repr(f0), q=repr(q), g=repr(g)) + ":precision=f64"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "f64le", "-ar", str(int(fs)), "-ac", "1", "-i", "pipe:0",
        "-af", flt, "-f", "f64le", "-ar", str(int(fs)), "-ac", "1", "pipe:1",
    ]
    proc = subprocess.run(cmd, input=x.astype("<f8").tobytes(), capture_output=True, check=True)
    y = np.frombuffer(proc.stdout, dtype="<f8")
    return y if len(y) == len(x) else None


RBJ_HARNESS_FAMILY = {
    "lowpass": "rbj_lowpass",
    "highpass": "rbj_highpass",
    "bandpass": "rbj_bandpass",
    "notch": "rbj_bandstop",
    "peak": "rbj_peak",
    "lowshelf": "rbj_lowshelf",
    "highshelf": "rbj_highshelf",
}


def build_rbj() -> dict:
    signals = make_signals()
    x = np.asarray(signals["mix"])
    imp = np.asarray(signals["impulse"])
    cases = []
    hl = []
    for fs in (44100.0, 48000.0):
        for kind in ("lowpass", "highpass", "bandpass", "notch", "peak", "lowshelf", "highshelf"):
            gains = [-12.0, -3.0, 6.0, 15.0] if kind in ("peak", "lowshelf", "highshelf") else [0.0]
            for f0 in (50.0, 1000.0, 10000.0, 18000.0):
                for q in (0.3, 0.7071067811865476, 2.0, 10.0):
                    for g in gains:
                        cid = f"rbj_{kind}_fs{int(fs)}_f{f0:g}_q{q:.4g}_g{g:g}"
                        cases.append({"id": cid, "kind": kind, "fs": fs, "f0": f0, "q": q, "gain_db": g})
                        hl.append(" ".join(str(v) for v in [cid, "dspfilters", RBJ_HARNESS_FAMILY[kind], "-", 2, fs, f0, 0.0, 0.0, 0.0, g, q, "-"]))
    harness = run_harness(hl)

    out = []
    ffmpeg_dev = 0.0
    dsp_dev = 0.0
    for c in cases:
        b, a = rbj(c["kind"], c["f0"], c["fs"], c["q"], c["gain_db"])
        y = sg.lfilter(b, a, x)
        h_imp = sg.lfilter(b, a, imp)
        rec = dict(c)
        rec.update({"b": b, "a": a})
        refs = []
        yf = ffmpeg_biquad(c["kind"], c["f0"], c["fs"], c["q"], c["gain_db"], x)
        if yf is not None:
            dev = float(np.max(np.abs(yf - y)))
            ffmpeg_dev = max(ffmpeg_dev, dev)
            if dev > 1e-9:
                raise SystemExit(f"ffmpeg disagrees with RBJ cookbook for {c['id']}: {dev}")
            rec["ffmpeg_max_abs_dev"] = dev
            refs.append("ffmpeg")
        # Time-domain reference (ffmpeg and the cookbook agree to < 1e-9, so a
        # single array represents both) for a representative subset.
        if c["q"] in (0.7071067811865476, 2.0):
            rec["lfilter_mix"] = y[:TD_LEN].tolist()
        hr = harness.get(c["id"])
        if hr and hr.get("ok") and hr.get("exact_convention"):
            dev = float(np.max(np.abs(np.asarray(hr["impulse"])[: len(h_imp)] - h_imp[: len(hr["impulse"])])))
            dsp_dev = max(dsp_dev, dev)
            if dev > 1e-9:
                raise SystemExit(f"DSPFilters disagrees with RBJ cookbook for {c['id']}: {dev}")
            refs.append("dspfilters")
        rec["agreeing_references"] = refs
        out.append(rec)
    return {
        "description": "RBJ Audio EQ Cookbook biquads: cookbook coefficients (a0-normalised) and "
        "lfilter output on the 'mix' signal; every case is cross-checked against FFmpeg af_biquads "
        "(precision=f64) and, when built, DSPFilters RBJ.",
        "max_dev": {"ffmpeg_vs_cookbook": ffmpeg_dev, "dspfilters_vs_cookbook": dsp_dev},
        "signals": {"mix": signals["mix"][:TD_LEN]},
        "cases": out,
    }


# ---------------------------------------------------------------------------
# Simple filters, FIR, filtfilt edge cases
# ---------------------------------------------------------------------------


def build_misc() -> dict:
    signals = make_signals()
    x = np.asarray(signals["mix"])
    simple = []
    for fs in (8000.0, 44100.0, 48000.0):
        for fc in (20.0, 1000.0, fs * 0.45):
            for kind, btype in (("low_pass", "lowpass"), ("high_pass", "highpass")):
                sos = sg.butter(1, fc, btype, fs=fs, output="sos")
                simple.append({"kind": kind, "fs": fs, "cutoff": fc, "y": sg.sosfilt(sos, x).tolist()})
        for lo, hi in ((100.0, 3000.0), (1000.0, 1100.0), (20.0, fs * 0.45)):
            sos = sg.butter(1, [lo, hi], "bandpass", fs=fs, output="sos")
            simple.append({"kind": "band_pass", "fs": fs, "low": lo, "high": hi, "y": sg.sosfilt(sos, x).tolist()})

    fir = []
    kernels = {
        "first_difference": [1.0, -1.0],
        "asymmetric_3": [0.5, 0.3, 0.2],
        "firwin_31_lowpass": sg.firwin(31, 0.2).tolist(),
        "firwin_301_bandpass": sg.firwin(301, [0.1, 0.3], pass_zero=False).tolist(),
        "minimum_phase_63": sg.minimum_phase(sg.firwin(125, 0.25), method="homomorphic").tolist(),
    }
    for name, h in kernels.items():
        fir.append({"name": name, "h": h, "valid": np.convolve(x, h, mode="valid").tolist()})

    filtfilt_short = []
    for n in (2, 5, 13, 40):
        xs = x[:n]
        sos = sg.butter(4, 1000, fs=44100, output="sos")
        # SciPy requires len > padlen (= 27 here); for shorter inputs
        # audio_samples pads with len-1 samples, i.e. padlen = n - 1.
        padlen = min(3 * (2 * len(sos) + 1), n - 1)
        filtfilt_short.append({"n": n, "padlen": padlen, "y": sg.sosfiltfilt(sos, xs, padlen=padlen).tolist()})

    return {
        "description": "First-order Butterworth simple filters, FIR valid convolution, short-signal sosfiltfilt.",
        "signals": {"mix": signals["mix"]},
        "simple_filters": simple,
        "fir": fir,
        "filtfilt_short": filtfilt_short,
    }


# ---------------------------------------------------------------------------


def dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if committed fixtures are out of date")
    ap.add_argument("--require-harness", action="store_true", help="fail if the C/C++ harness is not built")
    ap.add_argument("--no-harness", action="store_true", help="skip the C/C++ harness even if it is built")
    args = ap.parse_args()
    global USE_HARNESS
    USE_HARNESS = not args.no_harness

    if args.require_harness and not HARNESS.exists():
        print(f"C/C++ harness not built: {HARNESS} (run tests/reference/c/build.sh)", file=sys.stderr)
        return 2
    if not HARNESS.exists():
        print("warning: C/C++ harness not built; DSPFilters/liquid-dsp cross references omitted", file=sys.stderr)

    meta = {
        "generator": "tests/reference/generate_filter_fixtures.py",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "ffmpeg": subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
        if shutil.which("ffmpeg")
        else None,
        "harness": USE_HARNESS and HARNESS.exists(),
    }

    cases = design_cases()
    designs, time_domain = build_iir_designs(time_domain_subset(cases))
    outputs = {
        "iir_designs.json": {"meta": meta, **designs},
        "iir_time_domain.json": {"meta": meta, **time_domain},
        "rbj_biquads.json": {"meta": meta, **build_rbj()},
        "misc_filters.json": {"meta": meta, **build_misc()},
    }

    stale = []
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, obj in outputs.items():
        path = FIXTURES / name
        text = dumps(obj)
        if args.check:
            if not path.exists():
                stale.append(name)
                continue
            committed = json.loads(path.read_text())
            fresh = json.loads(text)
            for obj_ in (committed, fresh):
                obj_.pop("meta", None)
                if not meta["harness"]:
                    # Without the C/C++ harness the cross references cannot be
                    # regenerated; compare everything else.
                    obj_.pop("cross_check_summary", None)
                    obj_.get("max_dev", {}).pop("dspfilters_vs_cookbook", None)
                    for c in obj_.get("cases", []):
                        c.pop("cross_refs", None)
                        c.pop("agreeing_references", None)
            if committed != fresh:
                stale.append(name)
        else:
            path.write_text(text)
            print(f"wrote {path.relative_to(ROOT)} ({len(text) / 1e6:.2f} MB)")
    if stale:
        print("stale fixtures: " + ", ".join(stale), file=sys.stderr)
        return 1
    print("summary:", json.dumps(designs["cross_check_summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
