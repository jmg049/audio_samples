"""References for the array helpers of ``librosa.util`` and ``blocks_to_*``.

Inputs are stored next to the outputs. ``match_intervals`` non-strict cases use
queries whose preceding interval is the nearer one, because librosa's search for
the following interval starts one element too late. ``interp_broadcast`` is
produced by ``scipy.interpolate.interp1d`` inside librosa, once per kind.
"""

from __future__ import annotations

import numpy as np

import librosa

NAME = "array_ops"


def _l(a):
    return np.asarray(a).tolist()


def build(signals, sr) -> dict:
    rng = np.random.default_rng(20261104)
    out: dict = {}

    x = np.round(rng.normal(size=(6, 40)), 3)
    x[2, 5:9] = x[2, 4]  # plateaus
    out["local"] = {
        "x": _l(x),
        "max_axis0": _l(librosa.util.localmax(x, axis=0).astype(int)),
        "max_axis1": _l(librosa.util.localmax(x, axis=1).astype(int)),
        "min_axis0": _l(librosa.util.localmin(x, axis=0).astype(int)),
        "min_axis1": _l(librosa.util.localmin(x, axis=1).astype(int)),
    }

    src = np.sort(rng.uniform(0, 100, size=50))
    dst = rng.uniform(0, 100, size=30)
    dst_in = np.r_[dst, src[:3]]
    out["match_events"] = {
        "from": _l(src),
        "to": _l(dst),
        "both": _l(librosa.util.match_events(src, dst)),
        "from_inner": _l(src[(src > dst.min()) & (src < dst.max())]),
        "no_left": _l(librosa.util.match_events(src[(src > dst.min()) & (src < dst.max())], dst, left=False)),
        "no_right": _l(librosa.util.match_events(src[(src > dst.min()) & (src < dst.max())], dst, right=False)),
        "subset_from": _l(src[:3]),
        "subset_to": _l(dst_in),
        "exact": _l(librosa.util.match_events(src[:3], dst_in, left=False, right=False)),
    }

    starts = np.cumsum(rng.uniform(0.2, 2.0, size=20))
    to = np.c_[starts, starts + rng.uniform(0.1, 3.0, size=20)]
    fs = np.sort(rng.uniform(1.0, float(starts[-1]), size=25))
    frm = np.c_[fs, fs + rng.uniform(0.01, 1.0, size=25)]
    frm = frm[[bool(np.any((to[:, 0] <= q[1]) & (to[:, 1] >= q[0]))) for q in frm]]
    gap_to = np.array([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    gap_from = np.array([[1.5, 2.0], [11.5, 12.0], [22.0, 23.0], [0.5, 0.75]])
    out["match_intervals"] = {
        "from": _l(frm),
        "to": _l(to),
        "strict": _l(librosa.util.match_intervals(frm, to, strict=True)),
        "gap_from": _l(gap_from),
        "gap_to": _l(gap_to),
        "gap_non_strict": _l(librosa.util.match_intervals(gap_from, gap_to, strict=False)),
    }

    z = rng.normal(size=(5, 7)) + 1j * rng.normal(size=(5, 7))
    ang = rng.uniform(-np.pi, np.pi, size=(5, 7))
    mag = rng.uniform(0, 2, size=(5, 7))
    out["complex"] = {
        "z_re": _l(z.real),
        "z_im": _l(z.imag),
        "abs2": _l(librosa.util.abs2(z)),
        "abs2_real": _l(librosa.util.abs2(z.real)),
        "angles": _l(ang),
        "mag": _l(mag),
        "phasor_re": _l(librosa.util.phasor(ang).real),
        "phasor_im": _l(librosa.util.phasor(ang).imag),
        "phasor_mag_re": _l(librosa.util.phasor(ang, mag=mag).real),
        "phasor_mag_im": _l(librosa.util.phasor(ang, mag=mag).imag),
    }
    out["tiny"] = {
        "f32": float(librosa.util.tiny(np.float32(1))),
        "f64": float(librosa.util.tiny(np.float64(1))),
        "c64": float(librosa.util.tiny(np.complex64(1))),
        "int": float(librosa.util.tiny(1)),
    }

    v = np.arange(1.0, 6.0)
    out["pad_center"] = {
        "x": _l(v),
        **{
            mode: _l(librosa.util.pad_center(v, size=18, mode=mode))
            for mode in ("constant", "edge", "reflect", "symmetric", "wrap")
        },
        "matrix": _l(x[:2, :3]),
        "matrix_axis0": _l(librosa.util.pad_center(x[:2, :3], size=5, axis=0)),
        "matrix_axis1": _l(librosa.util.pad_center(x[:2, :3], size=6, axis=-1)),
    }

    frames = [3, 7, 7, 20, 41, 0, 12]
    out["fix_frames"] = {
        "frames": frames,
        "default": _l(librosa.util.fix_frames(frames)),
        "bounded": _l(librosa.util.fix_frames(frames, x_min=5, x_max=25)),
        "no_pad": _l(librosa.util.fix_frames(frames, x_min=5, x_max=25, pad=False)),
        "none": _l(librosa.util.fix_frames(frames, x_min=None, x_max=None)),
        "slices": [[s.start, s.stop] for s in librosa.util.index_to_slice(frames, idx_min=2, idx_max=30)],
    }

    s = rng.uniform(size=(8, 8))
    s = s + 5 * np.eye(8)[rng.permutation(8)]
    sorted_cols, idx_cols = librosa.util.axis_sort(s, index=True)
    sorted_rows, idx_rows = librosa.util.axis_sort(s, axis=0, index=True)
    out["axis_sort"] = {
        "s": _l(s),
        "columns": _l(sorted_cols),
        "columns_idx": _l(idx_cols),
        "rows": _l(sorted_rows),
        "rows_idx": _l(idx_rows),
        "argmin": _l(librosa.util.axis_sort(s, value=np.argmin)),
    }

    e = rng.normal(size=(6, 9))
    out["shear"] = {
        "x": _l(e),
        "columns": _l(librosa.util.shear(e, factor=1)),
        "rows": _l(librosa.util.shear(e, factor=-2, axis=0)),
    }

    sp = rng.laplace(size=(6, 30))
    sp[1] = 0.0
    out["sparsify_rows"] = {
        "x": _l(sp),
        "q01": _l(librosa.util.sparsify_rows(sp, quantile=0.01).toarray()),
        "q25": _l(librosa.util.sparsify_rows(sp, quantile=0.25).toarray()),
        "one_d": _l(librosa.util.sparsify_rows(sp[0], quantile=0.1).toarray()),
    }

    g = rng.normal(size=(4, 11))
    out["cyclic_gradient"] = {
        "x": _l(g),
        "axis1": _l(librosa.util.cyclic_gradient(g)),
        "axis0_order2": _l(librosa.util.cyclic_gradient(g, axis=0, edge_order=2)),
    }

    fill: dict = {}
    for shape in [(8, 8), (5, 12), (12, 5)]:
        for radius in (0.25, 0.5, 0.0625):
            m = np.ones(shape)
            librosa.util.fill_off_diagonal(m, radius=radius, value=-1.0)
            fill[f"{shape[0]}x{shape[1]}_{radius}"] = _l(m)
    out["fill_off_diagonal"] = fill

    u = rng.integers(0, 4, size=(6, 5))
    out["unique"] = {
        "x": _l(u),
        "count_axis1": _l(librosa.util.count_unique(u)),
        "count_axis0": _l(librosa.util.count_unique(u, axis=0)),
        "is_axis1": _l(librosa.util.is_unique(u).astype(int)),
        "is_axis0": _l(librosa.util.is_unique(u, axis=0).astype(int)),
    }

    xi = rng.normal(size=(3, 12))
    pos1 = np.sort(rng.uniform(0, 10, size=12))
    xj = rng.normal(size=(3, 9))
    pos2 = np.sort(rng.uniform(1, 9, size=9))
    q = np.linspace(0, 10, 25)
    kinds: dict = {}
    for kind in ("linear", "nearest", "nearest-up", "zero", "slinear", "quadratic", "cubic", "previous", "next"):
        y1, y2 = librosa.util.interp_broadcast(
            x1=xi, x1_pos=pos1, x2=xj, x2_pos=pos2, interp_pos=q, op=None, kind=kind, fill_value=-7.0, axis=-1
        )
        kinds[kind] = {"y1": _l(y1), "y2": _l(y2)}
    out["interp"] = {
        "x1": _l(xi),
        "x1_pos": _l(pos1),
        "x2": _l(xj),
        "x2_pos": _l(pos2),
        "q": _l(q),
        "kinds": kinds,
        "product": _l(
            librosa.util.interp_broadcast(x1=xi, x1_pos=pos1, x2=xj, x2_pos=pos2, interp_pos=q, axis=-1)
        ),
    }

    raw = rng.integers(0, 256, size=64, dtype=np.uint8)
    out["buf_to_float"] = {
        "bytes": _l(raw),
        **{
            f"n{n}": _l(librosa.util.buf_to_float(raw.tobytes(), n_bytes=n, dtype=np.float64))
            for n in (1, 2, 4, 8)
        },
        "n2_f32": _l(librosa.util.buf_to_float(raw.tobytes(), n_bytes=2)),
    }

    blocks = [0, 1, 7, 1000]
    out["blocks"] = {
        "blocks": blocks,
        "frames": _l(librosa.blocks_to_frames(blocks, block_length=16)),
        "samples": _l(librosa.blocks_to_samples(blocks, block_length=16, hop_length=512)),
        "time": _l(librosa.blocks_to_time(blocks, block_length=16, hop_length=512, sr=22050)),
    }
    y = rng.normal(size=64)
    y[10:14] = [1e-12, -1e-12, 0.0, -0.0]
    y2 = np.stack([y, -y])
    out["zero_crossings"] = {
        "y": _l(y),
        "default": _l(librosa.zero_crossings(y).astype(float)),
        "no_pad": _l(librosa.zero_crossings(y, pad=False).astype(float)),
        "zero_sign": _l(librosa.zero_crossings(y, zero_pos=False, threshold=0.5).astype(float)),
        "ref": _l(librosa.zero_crossings(y, threshold=0.1, ref_magnitude=np.max).astype(float)),
        "axis0": _l(librosa.zero_crossings(y2, axis=0).astype(float)),
    }
    out["generators"] = {
        "tone": _l(librosa.tone(441.0, sr=22050, length=512)),
        "tone_phi": _l(librosa.tone(441.0, sr=22050, length=512, phi=0.7)),
        "chirp_log": _l(librosa.chirp(fmin=110.0, fmax=3520.0, sr=22050, length=2048)),
        "chirp_lin": _l(librosa.chirp(fmin=110.0, fmax=3520.0, sr=22050, length=2048, linear=True, phi=0.3)),
    }
    out["note_names"] = {
        "midi": [60.0, 61.0, 69.25, 70.0, 12.0, 3.0],
        "c": [str(librosa.midi_to_note(m, unicode=False)) for m in [60.0, 61.0, 69.25, 70.0, 12.0, 3.0]],
        "f_major_no_octave": [str(librosa.midi_to_note(m, octave=False, key="F:maj", unicode=False)) for m in [60.0, 61.0, 70.0]],
        "unicode": [str(librosa.midi_to_note(m, key="Db:maj")) for m in [61.0, 70.0]],
        "cents": [str(librosa.midi_to_note(m, cents=True, unicode=False)) for m in [69.25, 69.0, 68.75]],
        "fractional_notes": ["C4+25", "A4-50", "Db3", "G#5+10"],
        "fractional": [float(librosa.note_to_midi(n, round_midi=False)) for n in ["C4+25", "A4-50", "Db3", "G#5+10"]],
        "rounded": [float(librosa.note_to_midi(n, round_midi=True)) for n in ["C4+25", "A4-50", "Db3", "G#5+10"]],
        "hz": _l(librosa.note_to_hz(["C4+25", "A4-50", "Db3", "G#5+10"])),
        "hz_rounded": _l(librosa.note_to_hz(["C4+25", "A4-50", "Db3", "G#5+10"], round_midi=True)),
    }
    return out
