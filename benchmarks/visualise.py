#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Visualise audio_samples vs FFmpeg/C benchmark results.

Reads results.csv produced by benchmarks/run.sh and outputs:
  - A PNG chart (results.png) with one subplot per operation
  - A Markdown summary table printed to stdout

Usage:
    python benchmarks/visualise.py [results.csv]
"""

import csv
import sys
import math
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Map raw operation prefix → human label (also defines display order)
OP_LABELS = {
    "rms_and_peak":            "rms + peak",
    "stereo_to_mono":          "stereo → mono",
    "mono_to_stereo":          "mono → stereo",
    "format_f32_to_i16":       "f32 → i16",
    "format_i16_to_f32":       "i16 → f32",
    "lowpass_1000hz_order2":   "lowpass (order-2)",
    "highpass_1000hz_order2":  "highpass (order-2)",
    "resample_44100_to_16000": "resample 44.1 → 16 kHz",
    "scale_by_half":           "scale × 0.5",
    "clip":                    "clip [−0.5, 0.5]",
    "normalize_peak":          "normalize (peak)",
    "trim_middle":             "trim (middle 50 %)",
    "pad_end":                 "pad (+ 0.5 s end)",
    "fade_in":                 "fade-in (0.5 s)",
}

# For each operation: list of (our_impl, ref_impl) pairs used in the
# markdown table.  For multi-quality ops (resampling) each quality level
# gets its own row, paired against the most comparable FFmpeg engine.
COMPARISONS = {
    "rms_and_peak":            [("audio_samples",             "c_native")],
    "stereo_to_mono":          [("audio_samples",             "ffmpeg_swresample")],
    "mono_to_stereo":          [("audio_samples",             "ffmpeg_swresample")],
    "format_f32_to_i16":       [("audio_samples",             "ffmpeg_swresample")],
    "format_i16_to_f32":       [("audio_samples",             "ffmpeg_swresample")],
    "lowpass_1000hz_order2":   [("audio_samples_butterworth", "ffmpeg_lowpass_order2")],
    "highpass_1000hz_order2":  [("audio_samples_butterworth", "ffmpeg_highpass_order2")],
    "resample_44100_to_16000": [
        ("audio_samples_fast",   "ffmpeg_default"),   # comparable quality
        ("audio_samples_medium", "ffmpeg_soxr"),
        ("audio_samples_high",   "ffmpeg_soxr"),
    ],
    "scale_by_half":           [("audio_samples", "c_native")],
    "clip":                    [("audio_samples", "c_native")],
    "normalize_peak":          [("audio_samples", "c_native")],
    "trim_middle":             [("audio_samples", "c_native")],
    "pad_end":                 [("audio_samples", "c_native")],
    "fade_in":                 [("audio_samples", "c_native")],
}

# Ordered list of implementations to show per op in the bar chart.
# Ref implementations first, then audio_samples variants.
IMPL_ORDER = {
    "rms_and_peak":            ["c_native", "audio_samples"],
    "stereo_to_mono":          ["ffmpeg_swresample", "audio_samples"],
    "mono_to_stereo":          ["ffmpeg_swresample", "audio_samples"],
    "format_f32_to_i16":       ["ffmpeg_swresample", "audio_samples"],
    "format_i16_to_f32":       ["ffmpeg_swresample", "audio_samples"],
    "lowpass_1000hz_order2":   ["ffmpeg_lowpass_order2", "audio_samples_butterworth"],
    "highpass_1000hz_order2":  ["ffmpeg_highpass_order2", "audio_samples_butterworth"],
    "resample_44100_to_16000": [
        "ffmpeg_default", "ffmpeg_soxr",
        "audio_samples_fast", "audio_samples_medium", "audio_samples_high",
    ],
    "scale_by_half":           ["c_native", "audio_samples"],
    "clip":                    ["c_native", "audio_samples"],
    "normalize_peak":          ["c_native", "audio_samples"],
    "trim_middle":             ["c_native", "audio_samples"],
    "pad_end":                 ["c_native", "audio_samples"],
    "fade_in":                 ["c_native", "audio_samples"],
}

OUR_IMPL_PREFIX = "audio_samples"

# Human-readable labels for each implementation
IMPL_LABELS = {
    "c_native":                  "C (gcc -O3 -march=native -ffast-math)",
    "ffmpeg_swresample":         "FFmpeg (libswresample)",
    "ffmpeg_lowpass_order2":     "FFmpeg (libavfilter)",
    "ffmpeg_highpass_order2":    "FFmpeg (libavfilter)",
    "ffmpeg_default":            "FFmpeg (default)",
    "ffmpeg_soxr":               "FFmpeg (SoXr)",
    "audio_samples":             "audio_samples",
    "audio_samples_butterworth": "audio_samples",
    "audio_samples_fast":        "audio_samples (fast)",
    "audio_samples_medium":      "audio_samples (medium)",
    "audio_samples_high":        "audio_samples (high)",
}

# Bar colours: warm tones for reference/C, cool blues for audio_samples
IMPL_COLORS = {
    "c_native":                  "#2ca02c",
    "ffmpeg_swresample":         "#DD8452",
    "ffmpeg_lowpass_order2":     "#c45e28",
    "ffmpeg_highpass_order2":    "#c45e28",
    "ffmpeg_default":            "#DD8452",
    "ffmpeg_soxr":               "#a04020",
    "audio_samples":             "#4C72B0",
    "audio_samples_butterworth": "#4C72B0",
    "audio_samples_fast":        "#8ab0d8",
    "audio_samples_medium":      "#4C72B0",
    "audio_samples_high":        "#2d4f82",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            row["duration_s"]  = int(row["duration_s"])
            row["n_samples"]   = int(row["n_samples"])
            row["iterations"]  = int(row["iterations"])
            row["warmup"]      = int(row["warmup"])
            row["min_us"]      = float(row["min_us"])
            row["mean_us"]     = float(row["mean_us"])
            row["median_us"]   = float(row["median_us"])
            row["max_us"]      = float(row["max_us"])
            row["stddev_us"]   = float(row["stddev_us"])
            rows.append(row)
    return rows


def op_prefix(operation: str) -> str:
    """Strip trailing _Ns duration suffix to get the operation prefix."""
    parts = operation.rsplit("_", 1)
    if len(parts) == 2 and parts[1].endswith("s") and parts[1][:-1].isdigit():
        return parts[0]
    return operation


def group(rows: list[dict]):
    """Group rows by (op_prefix, duration_s, implementation)."""
    data: dict[str, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        prefix = op_prefix(row["operation"])
        if prefix not in OP_LABELS:
            continue
        data[prefix][row["duration_s"]][row["implementation"]] = row
    return data


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------

def fmt_us(v: float) -> str:
    if v >= 1000:
        return f"{v/1000:.1f} ms"
    if v >= 10:
        return f"{v:.0f} µs"
    return f"{v:.2f} µs"


def ratio_str(ours: float, ref: float) -> str:
    r = ours / ref
    if r <= 1.05:
        label = f"**{1/r:.1f}× faster**" if r < 0.95 else "**≈ tied**"
    else:
        label = f"{r:.1f}× slower"
    return label


def _quality_suffix(impl: str) -> str:
    """Return ' (fast)' / ' (medium)' / ' (high)' for multi-quality impls."""
    for q in ("_fast", "_medium", "_high"):
        if impl.endswith(q):
            return f" ({q[1:]})"
    return ""


def markdown_table(data) -> str:
    lines = [
        "## Benchmark Summary\n",
        "Metric: **median latency** (lower is better).\n",
        "| Operation | Duration | audio_samples | Reference | vs Reference |",
        "|---|---|---|---|---|",
    ]
    for prefix in OP_LABELS:
        if prefix not in data:
            continue
        op_label = OP_LABELS[prefix]
        pairs = COMPARISONS.get(prefix, [])
        dur_map = data[prefix]

        for dur in sorted(dur_map):
            impls = dur_map[dur]
            for our_impl, ref_impl in pairs:
                our_row = impls.get(our_impl)
                ref_row = impls.get(ref_impl)
                if our_row is None or ref_row is None:
                    continue
                our_med = our_row["median_us"]
                ref_med = ref_row["median_us"]
                ref_name = IMPL_LABELS.get(ref_impl, ref_impl)
                row_label = op_label + _quality_suffix(our_impl)
                lines.append(
                    f"| {row_label} | {dur}s | {fmt_us(our_med)} "
                    f"| {fmt_us(ref_med)} ({ref_name}) "
                    f"| {ratio_str(our_med, ref_med)} |"
                )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _impl_color(impl: str) -> str:
    return IMPL_COLORS.get(impl, "#888888")


def plot(data, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping plot (pip install matplotlib)", file=sys.stderr)
        return

    prefixes = [p for p in OP_LABELS if p in data]
    n_ops = len(prefixes)
    ncols = 2
    nrows = math.ceil(n_ops / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.5 * nrows))
    axes = axes.flatten() if n_ops > 1 else [axes]

    for ax, prefix in zip(axes, prefixes):
        dur_map   = data[prefix]
        durations = sorted(dur_map)

        # Determine which implementations to plot, in order
        order = IMPL_ORDER.get(prefix)
        if order is None:
            # Fallback: all implementations found across any duration
            seen = {}
            for d in durations:
                for k in dur_map[d]:
                    seen[k] = True
            order = list(seen)

        # Filter to only impls that actually exist in the data
        avail = {impl for d in durations for impl in dur_map[d]}
        impls = [im for im in order if im in avail]
        n_impls = len(impls)

        width = min(0.3, 0.75 / n_impls)
        offsets = np.arange(n_impls) * width - (n_impls - 1) * width / 2
        x = np.arange(len(durations))
        labels = [f"{d}s" for d in durations]

        for impl, offset in zip(impls, offsets):
            medians = [dur_map[d].get(impl, {}).get("median_us", float("nan")) for d in durations]
            errs    = [dur_map[d].get(impl, {}).get("stddev_us", 0)           for d in durations]
            color   = _impl_color(impl)
            label   = IMPL_LABELS.get(impl, impl)
            ax.bar(x + offset, medians, width,
                   label=label, color=color, alpha=0.87)

        # For simple 2-impl ops: annotate ratio between the two bars
        if n_impls == 2:
            ref_impl, our_impl = impls[0], impls[1]
            for xi, dur in enumerate(durations):
                rm = dur_map[dur].get(ref_impl, {}).get("median_us", float("nan"))
                om = dur_map[dur].get(our_impl, {}).get("median_us", float("nan"))
                if math.isnan(rm) or math.isnan(om) or rm == 0:
                    continue
                r = om / rm
                ymax = max(rm, om)
                label = f"{1/r:.1f}×↑" if r < 0.95 else (f"{r:.1f}×↓" if r > 1.05 else "≈")
                color = _impl_color(our_impl) if r < 0.95 else ("red" if r > 1.05 else "gray")
                ax.text(xi, ymax * 1.06, label, ha="center", va="bottom",
                        fontsize=8, color=color, fontweight="bold")

        ax.set_title(OP_LABELS[prefix], fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Median latency (µs)")
        ax.legend(fontsize=7, loc="upper left")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{v/1000:.0f} ms" if v >= 1000 else f"{v:.0f} µs"
        ))
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Hide any unused subplots
    for ax in axes[n_ops:]:
        ax.set_visible(False)

    fig.suptitle("audio_samples vs FFmpeg/C  —  median latency (lower is better)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    
    print(f"Chart saved to {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results.csv"
    if not csv_path.exists():
        sys.exit(f"results.csv not found at {csv_path}. Run benchmarks/run.sh first.")

    rows = load(csv_path)
    data = group(rows)

    # Markdown table → stdout
    print(markdown_table(data))

    # Chart → results.png next to the CSV
    out_png = csv_path.with_suffix(".png")
    plot(data, out_png)


if __name__ == "__main__":
    main()
