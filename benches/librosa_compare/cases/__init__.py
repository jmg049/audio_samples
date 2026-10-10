"""Benchmark case registry.

A case pairs one librosa function with its audio_samples equivalent through
the Python bindings. The Rust harness (`examples/librosa_bench/`) has a case of
the same name for the native call.

Each case gives two *builders*, ``librosa`` and ``ours``. A builder takes a
:class:`Ctx` (the input for one configuration) and returns a zero-argument
callable that performs exactly one call of the function. Everything that is
not part of the call (loading the signal, building the ``AudioSamples``,
computing a spectrogram that the function takes as input) happens in the
builder, outside the timed region. ``functools.partial`` is convenient
because it evaluates the arguments when the builder runs.

Prepared inputs (``ctx.D``, ``ctx.mel``, ...) are computed once per
configuration with librosa in the configuration's dtype and handed to both
libraries, so both see identical arrays.

Both libraries are called with librosa 1.0's defaults. Where a binding's
default differs from librosa's (for example ``power_to_db``'s ``top_db``), the
case passes librosa's value explicitly.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from common import CONFIG_SETS, Config, load_signal


@dataclass
class Case:
    name: str
    group: str
    configs: str  # key of common.CONFIG_SETS
    librosa: Callable[["Ctx"], Callable[[], Any]]
    ours: Optional[Callable[["Ctx"], Callable[[], Any]]]
    # librosa function this case times (for the report), e.g. "feature.mfcc"
    func: str = ""
    note: str = ""

    def config_list(self) -> list[Config]:
        return CONFIG_SETS[self.configs]


class Ctx:
    """Input for one configuration; derived inputs are cached properties."""

    def __init__(self, cfg: Config, sig_dir: str):
        self.cfg = cfg
        self.sig_dir = sig_dir
        self.sr = cfg.sr
        self.dtype = cfg.np_dtype
        self._cache: dict[str, Any] = {}

    def cached(self, key: str, make: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    # --- audio ---------------------------------------------------------------
    @property
    def y(self) -> np.ndarray:
        """``(n,)`` mono or ``(channels, n)``, in the configuration's dtype."""
        return self.cached("y", lambda: load_signal(self.sig_dir, self.cfg))

    @property
    def n(self) -> int:
        return self.y.shape[-1]

    @property
    def a(self):
        """The same samples as an ``audio_samples.AudioSamples``."""
        import audio_samples as aus

        def make():
            y = self.y
            if y.ndim == 1:
                return aus.AudioSamples.new_mono(y, self.sr)
            return aus.AudioSamples.new_multi(y, self.sr)

        return self.cached("a", make)

    # --- derived inputs, computed with librosa -------------------------------
    @property
    def D(self) -> np.ndarray:
        import librosa

        return self.cached("D", lambda: librosa.stft(self.y))

    @property
    def S_mag(self) -> np.ndarray:
        return self.cached("S_mag", lambda: np.abs(self.D))

    @property
    def S_pow(self) -> np.ndarray:
        return self.cached("S_pow", lambda: np.abs(self.D) ** 2)

    @property
    def mel(self) -> np.ndarray:
        import librosa

        return self.cached("mel", lambda: librosa.feature.melspectrogram(y=self.y, sr=self.sr))

    @property
    def mel_db(self) -> np.ndarray:
        import librosa

        return self.cached("mel_db", lambda: librosa.power_to_db(self.mel))

    @property
    def mfcc(self) -> np.ndarray:
        import librosa

        return self.cached("mfcc", lambda: librosa.feature.mfcc(y=self.y, sr=self.sr))

    @property
    def chroma(self) -> np.ndarray:
        import librosa

        return self.cached("chroma", lambda: librosa.feature.chroma_stft(y=self.y, sr=self.sr))

    @property
    def onset_env(self) -> np.ndarray:
        import librosa

        return self.cached("onset_env", lambda: librosa.onset.onset_strength(y=self.y, sr=self.sr))

    @property
    def C(self) -> np.ndarray:
        import librosa

        return self.cached("C", lambda: librosa.cqt(self.y, sr=self.sr))

    @property
    def freqs(self) -> np.ndarray:
        import librosa

        return self.cached("freqs", lambda: librosa.fft_frequencies(sr=self.sr, n_fft=2048))


# Size of the 1-D arrays of the ``vec`` configurations (unit conversions).
VEC_N = 100_000


def vec_values(lo: float, hi: float, dtype, n: int = VEC_N) -> np.ndarray:
    """``n`` values spread over ``[lo, hi]`` (log-spaced when ``lo > 0``)."""
    if lo > 0:
        return np.geomspace(lo, hi, n).astype(dtype)
    return np.linspace(lo, hi, n).astype(dtype)


def lcg_matrix(rows: int, cols: int, seed: int, dtype=np.float64) -> np.ndarray:
    """Deterministic uniform ``[0, 1)`` matrix (numpy PCG64)."""
    return np.random.default_rng(seed).random((rows, cols)).astype(dtype)


partial = functools.partial

REGISTRY: list[Case] = []


def register(*cases: Case) -> None:
    REGISTRY.extend(cases)


def all_cases() -> list[Case]:
    if not REGISTRY:
        from . import (  # noqa: F401
            constant_q,
            convert,
            effects,
            features,
            pitch,
            rhythm,
            sequence,
            spectrum,
            tonal,
        )
    names = [c.name for c in REGISTRY]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f"duplicate case names: {dupes}"
    return REGISTRY
