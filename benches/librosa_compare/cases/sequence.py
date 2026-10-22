"""librosa.sequence and librosa.segment (DTW, RQA, Viterbi decoding,
transition matrices, recurrence and cross-similarity, lag conversions), plus
``util.sync``, ``util.normalize`` and ``util.frame``.

``seq`` configs time fixed-size matrices (``lcg_matrix``; the Rust half uses
its own generator: same shapes and distribution, different values). ``derived``
configs use chroma / MFCC / magnitude spectrograms of the 10 s and 60 s
signals, computed with librosa and handed to both libraries.

Skipped rows: none.

float32 differences (documented under "Output precision" in
LIBROSA_MIGRATION.md): ``dtw`` costs and ``viterbi`` log-probabilities
accumulate in float32 here and in float64 in librosa, so on ``seq_f32`` the
2000-step ``viterbi`` / ``viterbi_discriminative`` paths can differ at a step
(1 of 2000 on this input); ``float64`` agrees exactly. Recurrence /
cross-similarity of float32 chroma can pick different neighbours among
near-tied distances (238 of 186k cells on the 10 s input; 0 at 60 s and in
float64).
"""

import librosa
import numpy as np

from audio_samples import features as F
from audio_samples import sequence as Q
from audio_samples import spectral as S

from . import Case, lcg_matrix, partial, register

G = "sequence"

# Viterbi: 100 states x 2000 steps, banded (width-9 triangle) row-stochastic transitions.
N_STATES, N_STEPS, BAND = 100, 2000, 9
# Transition-matrix constructors.
N_TRANS = 1000
# util.sync: one segment every 21 frames (~0.5 s at 22050 Hz, hop 512).
SYNC_STEP = 21


def _dtw_xy(c, n, m):
    return c.cached(f"dtw_{n}_{m}", lambda: (lcg_matrix(12, n, 1, c.dtype), lcg_matrix(12, m, 2, c.dtype)))


def _dtw_steps(c):
    def make():
        x, y = _dtw_xy(c, 430, 500)
        return librosa.sequence.dtw(x, y, return_steps=True)[2]

    return c.cached("dtw_steps", make)


def _rqa_sim(c):
    # Binary, recurrence-like: about 20% of the cells set.
    return c.cached("rqa_sim", lambda: (lcg_matrix(500, 600, 3) > 0.8).astype(c.dtype))


def _band():
    return librosa.sequence.transition_local(N_STATES, BAND)


def _prob(c):
    return c.cached("prob", lambda: lcg_matrix(N_STATES, N_STEPS, 4, c.dtype))


def _posterior(c):
    def make():
        p = _prob(c)
        return (p / p.sum(axis=0, keepdims=True)).astype(c.dtype)

    return c.cached("posterior", make)


def _rec(c):
    return c.cached("rec", lambda: librosa.segment.recurrence_matrix(c.chroma))


def _lag(c):
    return c.cached("lag", lambda: librosa.segment.recurrence_to_lag(_rec(c)))


def _halves(c):
    def make():
        h = c.chroma.shape[1] // 2
        return np.ascontiguousarray(c.chroma[:, :h]), np.ascontiguousarray(c.chroma[:, h:])

    return c.cached("halves", make)


def _sync_idx(c):
    return list(range(0, c.mfcc.shape[1], SYNC_STEP))


register(
    # --- DTW ----------------------------------------------------------------------
    Case(
        "dtw", G, "seq",
        librosa=lambda c: partial(librosa.sequence.dtw, *_dtw_xy(c, 430, 500)),
        ours=lambda c: partial(Q.dtw, *_dtw_xy(c, 430, 500)),
        func="sequence.dtw",
        note="X 12x430, Y 12x500 (10 s of chroma), euclidean, with backtracking",
    ),
    Case(
        "dtw_large", G, "seq",
        librosa=lambda c: partial(librosa.sequence.dtw, *_dtw_xy(c, 1300, 1500)),
        ours=lambda c: partial(Q.dtw, *_dtw_xy(c, 1300, 1500)),
        func="sequence.dtw",
        note="X 12x1300, Y 12x1500 (30 s of chroma), euclidean, with backtracking",
    ),
    Case(
        "dtw_backtracking", G, "seq",
        librosa=lambda c: partial(librosa.sequence.dtw_backtracking, _dtw_steps(c)),
        ours=lambda c: partial(Q.dtw_backtracking, _dtw_steps(c)),
        func="sequence.dtw_backtracking",
        note="steps of the 430x500 dtw case",
    ),
    Case(
        "rqa", G, "seq",
        librosa=lambda c: partial(librosa.sequence.rqa, _rqa_sim(c)),
        ours=lambda c: partial(Q.rqa, _rqa_sim(c)),
        func="sequence.rqa",
        note="500x600 binary similarity (20% set), with backtracking",
    ),
    # --- Viterbi --------------------------------------------------------------------
    Case(
        "viterbi", G, "seq",
        librosa=lambda c: partial(librosa.sequence.viterbi, _prob(c), _band()),
        ours=lambda c: partial(Q.viterbi, _prob(c), _band()),
        func="sequence.viterbi",
        note="100 states x 2000 steps, transition_local(100, 9)",
    ),
    Case(
        "viterbi_discriminative", G, "seq",
        librosa=lambda c: partial(librosa.sequence.viterbi_discriminative, _posterior(c), _band()),
        ours=lambda c: partial(Q.viterbi_discriminative, _posterior(c), _band()),
        func="sequence.viterbi_discriminative",
        note="100 states x 2000 steps (columns sum to 1), transition_local(100, 9)",
    ),
    Case(
        "viterbi_binary", G, "seq",
        librosa=lambda c: partial(
            librosa.sequence.viterbi_binary, _prob(c), librosa.sequence.transition_loop(2, 0.9)
        ),
        ours=lambda c: partial(Q.viterbi_binary, _prob(c), librosa.sequence.transition_loop(2, 0.9)),
        func="sequence.viterbi_binary",
        note="100 independent states x 2000 steps, shared transition_loop(2, 0.9)",
    ),
    Case(
        "transition_uniform", G, "const",
        librosa=lambda c: partial(librosa.sequence.transition_uniform, N_TRANS),
        ours=lambda c: partial(Q.transition_uniform, N_TRANS),
        func="sequence.transition_uniform",
        note="1000 states",
    ),
    Case(
        "transition_loop", G, "const",
        librosa=lambda c: partial(librosa.sequence.transition_loop, N_TRANS, 0.9),
        ours=lambda c: partial(Q.transition_loop, N_TRANS, 0.9),
        func="sequence.transition_loop",
        note="1000 states",
    ),
    Case(
        "transition_cycle", G, "const",
        librosa=lambda c: partial(librosa.sequence.transition_cycle, N_TRANS, 0.9),
        ours=lambda c: partial(Q.transition_cycle, N_TRANS, 0.9),
        func="sequence.transition_cycle",
        note="1000 states",
    ),
    Case(
        "transition_local", G, "const",
        librosa=lambda c: partial(librosa.sequence.transition_local, N_TRANS, BAND),
        ours=lambda c: partial(Q.transition_local, N_TRANS, BAND),
        func="sequence.transition_local",
        note="1000 states, width 9, triangle window",
    ),
    # --- recurrence ----------------------------------------------------------------------
    Case(
        "recurrence_matrix", G, "derived",
        librosa=lambda c: partial(librosa.segment.recurrence_matrix, c.chroma),
        ours=lambda c: partial(Q.recurrence_matrix, c.chroma),
        func="segment.recurrence_matrix",
        note="chroma_stft (12 x frames), connectivity",
    ),
    Case(
        "recurrence_matrix_affinity", G, "derived",
        librosa=lambda c: partial(librosa.segment.recurrence_matrix, c.mfcc, mode="affinity", sym=True),
        ours=lambda c: partial(Q.recurrence_matrix, c.mfcc, mode="affinity", sym=True),
        func="segment.recurrence_matrix",
        note="mfcc (20 x frames), mode='affinity', sym=True",
    ),
    Case(
        "cross_similarity", G, "derived",
        librosa=lambda c: partial(librosa.segment.cross_similarity, *_halves(c)),
        ours=lambda c: partial(Q.cross_similarity, *_halves(c)),
        func="segment.cross_similarity",
        note="first half of the chroma against the second half, connectivity",
    ),
    Case(
        "recurrence_to_lag", G, "derived",
        librosa=lambda c: partial(librosa.segment.recurrence_to_lag, _rec(c)),
        ours=lambda c: partial(Q.recurrence_to_lag, _rec(c)),
        func="segment.recurrence_to_lag",
        note="the bool recurrence_matrix of the chroma",
    ),
    Case(
        "lag_to_recurrence", G, "derived",
        librosa=lambda c: partial(librosa.segment.lag_to_recurrence, _lag(c)),
        ours=lambda c: partial(Q.lag_to_recurrence, _lag(c)),
        func="segment.lag_to_recurrence",
        note="recurrence_to_lag of the bool chroma recurrence matrix",
    ),
    # --- util --------------------------------------------------------------------------------
    Case(
        "sync", G, "derived",
        librosa=lambda c: partial(librosa.util.sync, c.mfcc, _sync_idx(c)),
        ours=lambda c: partial(Q.sync, c.mfcc, _sync_idx(c)),
        func="util.sync",
        note="mfcc, a boundary every 21 frames, mean",
    ),
    Case(
        "normalize", G, "derived",
        librosa=lambda c: partial(librosa.util.normalize, c.S_mag),
        ours=lambda c: partial(F.normalize, c.S_mag),
        func="util.normalize",
        note="|stft| (1025 x frames), max norm over axis 0",
    ),
    Case(
        "frame", G, "mono",
        librosa=lambda c: partial(librosa.util.frame, c.y, frame_length=2048, hop_length=512),
        ours=lambda c: partial(S.frame, c.y, frame_length=2048, hop_length=512, copy=False),
        func="util.frame",
        note="both return a read-only strided view (audio_samples with copy=False; its default copies)",
    ),
    Case(
        "frame_copy", G, "mono",
        librosa=lambda c: lambda: librosa.util.frame(c.y, frame_length=2048, hop_length=512).copy(),
        ours=lambda c: partial(S.frame, c.y, frame_length=2048, hop_length=512),
        func="util.frame",
        note="owned frames: librosa's view copied with .copy(); audio_samples' default (copy=True)",
    ),
)
