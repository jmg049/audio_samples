"""librosa.util array helpers and ``blocks_to_*``.

Every helper has a row on a large input (``vec``: VEC_N values, ``seq``: a
SEQ_R x SEQ_C matrix) and a ``_tiny`` row on 64 values or an 8 x 8 matrix
(``const``), where binding overhead dominates. Scalars-only helpers have a
``const`` row.

``stack`` and the float32 ``seq`` rows of the integer helpers use the same
arrays on both sides. ``interp_broadcast`` uses ``kind="cubic"`` as the
expensive kind and ``kind="linear"`` as the cheap one. ``find_files`` reads a
directory tree of 2000 empty files created outside the timed call.
"""

import os
import tempfile

import librosa
import numpy as np

from audio_samples import arrays as A
from audio_samples import utils as U

from . import VEC_N, Case, lcg_matrix, partial, register, vec_values

G = "array_ops"
SEQ_R, SEQ_C = 500, 600
TINY = 64
BL, HOP, SR = 16, 512, 22050


def _v(c):
    return c.cached("v", lambda: np.sin(vec_values(0.0, 3000.0, c.dtype)))


def _tv(c):
    return c.cached("tv", lambda: np.sin(np.linspace(0.0, 30.0, TINY)))


def _m(c):
    return c.cached("m", lambda: lcg_matrix(SEQ_R, SEQ_C, 11, c.dtype))


def _tm(c):
    return c.cached("tm", lambda: lcg_matrix(8, 8, 11))


def _z(c):
    cdt = np.complex64 if c.dtype == np.float32 else np.complex128
    return c.cached("z", lambda: (_v(c) + 1j * _v(c)[::-1]).astype(cdt))


def _tz(c):
    return c.cached("tz", lambda: _tv(c) + 1j * _tv(c)[::-1])


def _ints(c):
    return c.cached("ints", lambda: (lcg_matrix(SEQ_R, SEQ_C, 12) * 50).astype(np.int64))


def _events(n, seed):
    return lcg_matrix(1, n, seed)[0] * 1000.0


def _intervals(n, seed):
    s = np.sort(lcg_matrix(1, n, seed)[0] * 1000.0)
    return np.c_[s, s + 5.0]


def _frames():
    return (lcg_matrix(1, VEC_N, 13)[0] * 1e6).astype(np.int64)


def _files():
    d = os.path.join(tempfile.gettempdir(), "aus_bench_files")
    if not os.path.isdir(d):
        for sub in range(20):
            p = os.path.join(d, f"s{sub}")
            os.makedirs(p, exist_ok=True)
            for i in range(100):
                open(os.path.join(p, f"f{i}.wav"), "w").close()
    return d


def pair(name, configs, lib, ours, func, note=""):
    return Case(name, G, configs, librosa=lib, ours=ours, func=func, note=note)


register(
    pair("localmax", "vec", lambda c: partial(librosa.util.localmax, _v(c)), lambda c: partial(A.localmax, _v(c)), "util.localmax"),
    pair("localmax_tiny", "const", lambda c: partial(librosa.util.localmax, _tv(c)), lambda c: partial(A.localmax, _tv(c)), "util.localmax"),
    pair("localmin", "vec", lambda c: partial(librosa.util.localmin, _v(c)), lambda c: partial(A.localmin, _v(c)), "util.localmin"),
    pair("localmin_tiny", "const", lambda c: partial(librosa.util.localmin, _tv(c)), lambda c: partial(A.localmin, _tv(c)), "util.localmin"),
    pair("match_events", "const", lambda c: partial(librosa.util.match_events, _events(VEC_N, 1), _events(5000, 2)), lambda c: partial(A.match_events, _events(VEC_N, 1), _events(5000, 2)), "util.match_events"),
    pair("match_events_tiny", "const", lambda c: partial(librosa.util.match_events, _events(TINY, 1), _events(16, 2)), lambda c: partial(A.match_events, _events(TINY, 1), _events(16, 2)), "util.match_events"),
    pair("match_intervals", "const", lambda c: partial(librosa.util.match_intervals, _intervals(2000, 3), _intervals(2000, 4)), lambda c: partial(A.match_intervals, _intervals(2000, 3), _intervals(2000, 4)), "util.match_intervals"),
    pair("match_intervals_tiny", "const", lambda c: partial(librosa.util.match_intervals, _intervals(16, 3), _intervals(16, 4), strict=False), lambda c: partial(A.match_intervals, _intervals(16, 3), _intervals(16, 4), strict=False), "util.match_intervals"),
    pair("abs2", "vec", lambda c: partial(librosa.util.abs2, _z(c)), lambda c: partial(A.abs2, _z(c)), "util.abs2"),
    pair("abs2_tiny", "const", lambda c: partial(librosa.util.abs2, _tz(c)), lambda c: partial(A.abs2, _tz(c)), "util.abs2"),
    pair("phasor", "vec", lambda c: partial(librosa.util.phasor, _v(c)), lambda c: partial(A.phasor, _v(c)), "util.phasor"),
    pair("phasor_tiny", "const", lambda c: partial(librosa.util.phasor, _tv(c)), lambda c: partial(A.phasor, _tv(c)), "util.phasor"),
    pair("tiny", "const", lambda c: partial(librosa.util.tiny, _tv(c)), lambda c: partial(A.tiny, _tv(c)), "util.tiny"),
    pair("expand_to", "const", lambda c: partial(librosa.util.expand_to, _tm(c), ndim=4, axes=[1, 3]), lambda c: partial(A.expand_to, _tm(c), ndim=4, axes=[1, 3]), "util.expand_to"),
    pair("stack", "seq", lambda c: partial(librosa.util.stack, [_m(c), _m(c)], axis=0), lambda c: partial(A.stack, [_m(c), _m(c)], axis=0), "util.stack"),
    pair("stack_tiny", "const", lambda c: partial(librosa.util.stack, [_tm(c), _tm(c)], axis=0), lambda c: partial(A.stack, [_tm(c), _tm(c)], axis=0), "util.stack"),
    pair("pad_center", "vec", lambda c: partial(librosa.util.pad_center, _v(c), size=2 * VEC_N), lambda c: partial(A.pad_center, _v(c), size=2 * VEC_N), "util.pad_center"),
    pair("pad_center_tiny", "const", lambda c: partial(librosa.util.pad_center, _tv(c), size=128), lambda c: partial(A.pad_center, _tv(c), size=128), "util.pad_center"),
    pair("pad_center_reflect", "vec", lambda c: partial(librosa.util.pad_center, _v(c), size=2 * VEC_N, mode="reflect"), lambda c: partial(A.pad_center, _v(c), size=2 * VEC_N, mode="reflect"), "util.pad_center"),
    pair("fix_frames", "const", lambda c: partial(librosa.util.fix_frames, _frames(), x_max=500000), lambda c: partial(A.fix_frames, _frames(), x_max=500000), "util.fix_frames"),
    pair("fix_frames_tiny", "const", lambda c: partial(librosa.util.fix_frames, np.arange(TINY), x_max=50), lambda c: partial(A.fix_frames, np.arange(TINY), x_max=50), "util.fix_frames"),
    pair("index_to_slice", "const", lambda c: partial(librosa.util.index_to_slice, _frames()[:5000], idx_max=1000000), lambda c: partial(A.index_to_slice, _frames()[:5000], idx_max=1000000), "util.index_to_slice"),
    pair("index_to_slice_tiny", "const", lambda c: partial(librosa.util.index_to_slice, np.arange(TINY), idx_max=50), lambda c: partial(A.index_to_slice, np.arange(TINY), idx_max=50), "util.index_to_slice"),
    pair("axis_sort", "seq", lambda c: partial(librosa.util.axis_sort, _m(c)), lambda c: partial(A.axis_sort, _m(c)), "util.axis_sort"),
    pair("axis_sort_tiny", "const", lambda c: partial(librosa.util.axis_sort, _tm(c)), lambda c: partial(A.axis_sort, _tm(c)), "util.axis_sort"),
    pair("shear", "seq", lambda c: partial(librosa.util.shear, _m(c), factor=1), lambda c: partial(A.shear, _m(c), factor=1), "util.shear"),
    pair("shear_tiny", "const", lambda c: partial(librosa.util.shear, _tm(c), factor=1), lambda c: partial(A.shear, _tm(c), factor=1), "util.shear"),
    pair("sparsify_rows", "seq", lambda c: partial(librosa.util.sparsify_rows, _m(c), quantile=0.1), lambda c: partial(A.sparsify_rows, _m(c), quantile=0.1), "util.sparsify_rows"),
    pair("sparsify_rows_tiny", "const", lambda c: partial(librosa.util.sparsify_rows, _tm(c), quantile=0.1), lambda c: partial(A.sparsify_rows, _tm(c), quantile=0.1), "util.sparsify_rows"),
    pair("cyclic_gradient", "seq", lambda c: partial(librosa.util.cyclic_gradient, _m(c)), lambda c: partial(A.cyclic_gradient, _m(c)), "util.cyclic_gradient"),
    pair("cyclic_gradient_tiny", "const", lambda c: partial(librosa.util.cyclic_gradient, _tm(c)), lambda c: partial(A.cyclic_gradient, _tm(c)), "util.cyclic_gradient"),
    pair("fill_off_diagonal", "seq", lambda c: partial(librosa.util.fill_off_diagonal, _m(c).copy(), radius=0.2, value=0.0), lambda c: partial(A.fill_off_diagonal, _m(c).copy(), radius=0.2, value=0.0), "util.fill_off_diagonal"),
    pair("fill_off_diagonal_tiny", "const", lambda c: partial(librosa.util.fill_off_diagonal, _tm(c).copy(), radius=0.25, value=0.0), lambda c: partial(A.fill_off_diagonal, _tm(c).copy(), radius=0.25, value=0.0), "util.fill_off_diagonal"),
    pair("count_unique", "const", lambda c: partial(librosa.util.count_unique, _ints(c)), lambda c: partial(A.count_unique, _ints(c)), "util.count_unique"),
    pair("count_unique_tiny", "const", lambda c: partial(librosa.util.count_unique, (_tm(c) * 4).astype(np.int64)), lambda c: partial(A.count_unique, (_tm(c) * 4).astype(np.int64)), "util.count_unique"),
    pair("is_unique", "const", lambda c: partial(librosa.util.is_unique, _ints(c)), lambda c: partial(A.is_unique, _ints(c)), "util.is_unique"),
    pair("is_unique_tiny", "const", lambda c: partial(librosa.util.is_unique, (_tm(c) * 4).astype(np.int64)), lambda c: partial(A.is_unique, (_tm(c) * 4).astype(np.int64)), "util.is_unique"),
    pair(
        "interp_broadcast", "seq",
        lambda c: partial(librosa.util.interp_broadcast, x1=_m(c), x1_pos=np.arange(SEQ_C) * 1.0, x2=_m(c)[:, ::2], x2_pos=np.arange(SEQ_C // 2) * 2.0, axis=-1),
        lambda c: partial(A.interp_broadcast, x1=_m(c), x1_pos=np.arange(SEQ_C) * 1.0, x2=_m(c)[:, ::2], x2_pos=np.arange(SEQ_C // 2) * 2.0, axis=-1),
        "util.interp_broadcast",
    ),
    pair(
        "interp_broadcast_cubic", "seq",
        lambda c: partial(librosa.util.interp_broadcast, x1=_m(c), x1_pos=np.arange(SEQ_C) * 1.0, x2=_m(c)[:, ::2], x2_pos=np.arange(SEQ_C // 2) * 2.0, kind="cubic", axis=-1),
        lambda c: partial(A.interp_broadcast, x1=_m(c), x1_pos=np.arange(SEQ_C) * 1.0, x2=_m(c)[:, ::2], x2_pos=np.arange(SEQ_C // 2) * 2.0, kind="cubic", axis=-1),
        "util.interp_broadcast",
    ),
    pair(
        "interp_broadcast_tiny", "const",
        lambda c: partial(librosa.util.interp_broadcast, x1=_tm(c), x1_pos=np.arange(8) * 1.0, x2=_tm(c)[:, ::2], x2_pos=np.arange(4) * 2.0, axis=-1),
        lambda c: partial(A.interp_broadcast, x1=_tm(c), x1_pos=np.arange(8) * 1.0, x2=_tm(c)[:, ::2], x2_pos=np.arange(4) * 2.0, axis=-1),
        "util.interp_broadcast",
    ),
    pair("dtype_c2r", "const", lambda c: partial(librosa.util.dtype_c2r, np.complex64), lambda c: partial(A.dtype_c2r, np.complex64), "util.dtype_c2r"),
    pair("dtype_r2c", "const", lambda c: partial(librosa.util.dtype_r2c, np.float64), lambda c: partial(A.dtype_r2c, np.float64), "util.dtype_r2c"),
    pair("buf_to_float", "const", lambda c: partial(librosa.util.buf_to_float, (_events(VEC_N, 5).astype(np.int16)).tobytes()), lambda c: partial(A.buf_to_float, (_events(VEC_N, 5).astype(np.int16)).tobytes()), "util.buf_to_float"),
    pair("buf_to_float_tiny", "const", lambda c: partial(librosa.util.buf_to_float, (_events(TINY, 5).astype(np.int16)).tobytes()), lambda c: partial(A.buf_to_float, (_events(TINY, 5).astype(np.int16)).tobytes()), "util.buf_to_float"),
    pair("find_files", "const", lambda c: partial(librosa.util.find_files, _files()), lambda c: partial(A.find_files, _files()), "util.find_files", "2000 files in 20 directories"),
    pair("valid_audio", "vec", lambda c: partial(librosa.util.valid_audio, _v(c)), lambda c: partial(A.valid_audio, _v(c)), "util.valid_audio"),
    pair("valid_audio_tiny", "const", lambda c: partial(librosa.util.valid_audio, _tv(c)), lambda c: partial(A.valid_audio, _tv(c)), "util.valid_audio"),
    pair("valid_intervals", "const", lambda c: partial(librosa.util.valid_intervals, _intervals(VEC_N, 6)), lambda c: partial(A.valid_intervals, _intervals(VEC_N, 6)), "util.valid_intervals"),
    pair("valid_intervals_tiny", "const", lambda c: partial(librosa.util.valid_intervals, _intervals(16, 6)), lambda c: partial(A.valid_intervals, _intervals(16, 6)), "util.valid_intervals"),
    pair("is_positive_int", "const", lambda c: partial(librosa.util.is_positive_int, 5), lambda c: partial(A.is_positive_int, 5), "util.is_positive_int"),
    pair("valid_int", "const", lambda c: partial(librosa.util.valid_int, 5.5), lambda c: partial(A.valid_int, 5.5), "util.valid_int"),
    pair("blocks_to_frames", "const", lambda c: partial(librosa.blocks_to_frames, np.arange(VEC_N), block_length=BL), lambda c: partial(U.blocks_to_frames, np.arange(VEC_N), block_length=BL), "blocks_to_frames"),
    pair("blocks_to_frames_tiny", "const", lambda c: partial(librosa.blocks_to_frames, 7, block_length=BL), lambda c: partial(U.blocks_to_frames, 7, block_length=BL), "blocks_to_frames"),
    pair("blocks_to_samples", "const", lambda c: partial(librosa.blocks_to_samples, np.arange(VEC_N), block_length=BL, hop_length=HOP), lambda c: partial(U.blocks_to_samples, np.arange(VEC_N), block_length=BL, hop_length=HOP), "blocks_to_samples"),
    pair("blocks_to_samples_tiny", "const", lambda c: partial(librosa.blocks_to_samples, 7, block_length=BL, hop_length=HOP), lambda c: partial(U.blocks_to_samples, 7, block_length=BL, hop_length=HOP), "blocks_to_samples"),
    pair("blocks_to_time", "const", lambda c: partial(librosa.blocks_to_time, np.arange(VEC_N), block_length=BL, hop_length=HOP, sr=SR), lambda c: partial(U.blocks_to_time, np.arange(VEC_N), block_length=BL, hop_length=HOP, sample_rate=SR), "blocks_to_time"),
    pair("blocks_to_time_tiny", "const", lambda c: partial(librosa.blocks_to_time, 7, block_length=BL, hop_length=HOP, sr=SR), lambda c: partial(U.blocks_to_time, 7, block_length=BL, hop_length=HOP, sample_rate=SR), "blocks_to_time"),
)
