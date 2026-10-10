"""Shared pieces of the librosa comparison benchmark: input configurations,
test signals and the timing rule.

The Rust half (`examples/librosa_bench.rs`) reads the signals this module
writes, so both languages time the same samples.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

SAMPLE_RATES = (22050, 16000)
DURATIONS = (1, 10, 60)
DTYPES = ("f32", "f64")


@dataclass(frozen=True)
class Config:
    """One input configuration. `kind` selects what the case is given:

    * ``audio``: a signal of ``seconds`` at ``sr`` with ``channels`` channels;
    * ``const``: no signal (filterbanks, windows, frequency tables);
    * ``vec``: a 1-D array of values (unit conversions);
    * ``seq``: fixed-size matrices (sequence alignment and decoding).

    Cases that work on a spectrogram or envelope are ``audio`` cases whose
    prepared input is computed from the signal.
    """

    kind: str
    sr: int = 22050
    seconds: int = 10
    channels: int = 1
    dtype: str = "f64"

    @property
    def id(self) -> str:
        if self.kind == "audio":
            ch = "mono" if self.channels == 1 else "stereo"
            return f"{self.sr}_{self.seconds}s_{ch}_{self.dtype}"
        return f"{self.kind}_{self.dtype}"

    @property
    def np_dtype(self):
        return np.float32 if self.dtype == "f32" else np.float64

    @property
    def n(self) -> int:
        return self.sr * self.seconds


def parse_config(cid: str) -> Config:
    parts = cid.split("_")
    if len(parts) == 2:
        return Config(kind=parts[0], dtype=parts[1])
    sr, sec, ch, dt = parts
    return Config("audio", int(sr), int(sec[:-1]), 1 if ch == "mono" else 2, dt)


def audio_configs(stereo: bool = True, durations=DURATIONS, rates=SAMPLE_RATES):
    """Mono at every rate and duration, both dtypes; stereo at 22050 Hz, 10 s."""
    out = [
        Config("audio", sr, d, 1, dt) for sr in rates for d in durations for dt in DTYPES
    ]
    if stereo:
        out += [Config("audio", 22050, 10, 2, dt) for dt in DTYPES]
    return out


# Config sets named by the cases.
CONFIG_SETS = {
    # every audio size, mono and stereo
    "full": audio_configs(),
    # every audio size, mono only (librosa has no multichannel form)
    "mono": audio_configs(stereo=False),
    # expensive iterative ops: 1 s and 10 s
    "short": audio_configs(durations=(1, 10)),
    "short_mono": audio_configs(durations=(1, 10), stereo=False),
    # stereo-only operations (to_mono, remix of channels)
    "stereo": [Config("audio", sr, d, 2, dt) for sr in SAMPLE_RATES for d in DURATIONS for dt in DTYPES],
    # one input rate (resampling from that rate to a fixed target)
    "rate_22050": audio_configs(rates=(22050,)),
    "rate_16000": audio_configs(stereo=False, rates=(16000,)),
    # derived arrays (spectrograms, envelopes) from 10 s and 60 s mono at 22050 Hz
    "derived": [Config("audio", 22050, d, 1, dt) for d in (10, 60) for dt in DTYPES],
    "derived_f64": [Config("audio", 22050, d, 1, "f64") for d in (10, 60)],
    "const": [Config("const", dtype="f64")],
    "vec": [Config("vec", dtype=dt) for dt in DTYPES],
    "vec_f64": [Config("vec", dtype="f64")],
    "seq": [Config("seq", dtype=dt) for dt in DTYPES],
    "seq_f64": [Config("seq", dtype="f64")],
}

# Configs used for the fresh-subprocess RSS probe and the default-threading run.
RSS_CONFIGS = {"22050_10s_mono_f32", "22050_60s_mono_f32", "22050_60s_mono_f64", "22050_10s_stereo_f32",
               "const_f64", "vec_f32", "vec_f64", "seq_f32", "seq_f64"}
DEFAULT_THREAD_CONFIGS = {"22050_10s_mono_f32", "22050_60s_mono_f32", "22050_10s_stereo_f32",
                          "const_f64", "vec_f32", "seq_f32", "seq_f64"}


# ---------------------------------------------------------------------------
# Signals


def test_signal(sr: int, seconds: int, channel: int = 0) -> np.ndarray:
    """Decaying harmonic notes every 0.5 s (pitch walking in fifths from
    110 Hz, or 165 Hz on the second channel) plus 0.02 uniform noise, with
    0.5 s of digital silence at each end."""
    n = sr * seconds
    rng = np.random.default_rng(12345 + channel)
    noise = rng.uniform(-1.0, 1.0, n)
    t = np.arange(n) / sr
    seg = np.floor(t / 0.5)
    base = 110.0 * (1.5 if channel else 1.0)
    f = base * 2.0 ** (((seg.astype(np.uint64) * 7) % 24) / 12.0)
    env = np.exp(-3.0 * (t - seg * 0.5))
    tau = 2 * np.pi
    x = env * (
        0.5 * np.sin(tau * f * t) + 0.25 * np.sin(tau * 2 * f * t) + 0.12 * np.sin(tau * 3 * f * t)
    ) + 0.02 * noise
    if seconds > 1:
        x[(t < 0.5) | (t >= seconds - 0.5)] = 0.0
    return x


def signal_path(sig_dir: str, sr: int, seconds: int, channels: int) -> str:
    return os.path.join(sig_dir, f"sig_{sr}_{seconds}s_{channels}ch.f64")


def write_signals(sig_dir: str) -> None:
    """Write every signal as little-endian f64, channel-major ``(channels, n)``."""
    os.makedirs(sig_dir, exist_ok=True)
    for sr in SAMPLE_RATES:
        for d in DURATIONS:
            for ch in (1, 2):
                p = signal_path(sig_dir, sr, d, ch)
                if os.path.exists(p):
                    continue
                x = np.stack([test_signal(sr, d, c) for c in range(ch)])
                x.astype("<f8").tofile(p)


def load_signal(sig_dir: str, cfg: Config) -> np.ndarray:
    """``(n,)`` for mono, ``(channels, n)`` otherwise, in the config's dtype."""
    x = np.fromfile(signal_path(sig_dir, cfg.sr, cfg.seconds, cfg.channels), dtype="<f8")
    x = x.reshape(cfg.channels, -1)
    if cfg.channels == 1:
        x = x[0]
    return np.ascontiguousarray(x.astype(cfg.np_dtype))


# ---------------------------------------------------------------------------
# Timing rule (the Rust harness uses the same one)

MIN_TIME = 0.3  # seconds of timed calls, at least
MIN_REPS = 5
MAX_REPS = 50


def time_call(f, min_time=MIN_TIME, min_reps=MIN_REPS, max_reps=MAX_REPS):
    """Warm up (2 calls, 1 if a call takes over a second), then repeat until
    ``min_time`` and ``min_reps`` are both reached, at most ``max_reps``.
    Returns per-call times in milliseconds."""
    t0 = time.perf_counter()
    f()
    if time.perf_counter() - t0 < 1.0:
        f()
    times = []
    start = time.perf_counter()
    while len(times) < max_reps and (len(times) < min_reps or time.perf_counter() - start < min_time):
        t = time.perf_counter_ns()
        f()
        times.append((time.perf_counter_ns() - t) / 1e6)
    return times


def summarise(times):
    a = np.sort(np.asarray(times))
    return {
        "median_ms": float(np.median(a)),
        "p25_ms": float(np.percentile(a, 25)),
        "p75_ms": float(np.percentile(a, 75)),
        "min_ms": float(a[0]),
        "max_ms": float(a[-1]),
        "reps": int(a.size),
    }


THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
