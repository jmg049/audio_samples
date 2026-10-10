"""librosa.sequence / librosa.segment / librosa.util.sync reference cases.

All inputs are small deterministic matrices (seeded random, or chroma of the
shared test signals) and are stored in the fixture next to librosa's output, so
the Rust side does not need any feature extractor to reproduce them.

Non-finite values (``inf`` cells of DTW accumulated-cost matrices) are stored
as ``null``.

librosa delegates:
* DTW pairwise distances to ``scipy.spatial.distance.cdist``;
* nearest-neighbour search in ``recurrence_matrix`` / ``cross_similarity`` to
  ``sklearn.neighbors.NearestNeighbors`` (kd-tree or brute force, chosen by
  sklearn);
* transition windows to ``scipy.signal.get_window(..., fftbins=False)``.
"""

from __future__ import annotations

import numpy as np

import librosa

NAME = "sequence"


def _finite_or_none(a: np.ndarray):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 0:
        return float(a) if np.isfinite(a) else None
    return [_finite_or_none(x) for x in a]


def _dtw_cases(signals, sr, rng) -> dict:
    """DTW cases. Inputs are stored once in ``inputs`` and referenced by name."""
    cases = {}

    notes = signals["notes"]
    slow = librosa.effects.time_stretch(notes, rate=0.8)
    cx = librosa.feature.chroma_stft(y=notes, sr=sr)
    cy = librosa.feature.chroma_stft(y=slow, sr=sr)
    xa = rng.standard_normal((5, 14))
    ya = rng.standard_normal((5, 19))
    xs = rng.standard_normal((5, 6))
    c = rng.uniform(0.0, 1.0, (9, 13))
    inputs = {
        "cx": cx,
        "cy": cy,
        "cx_mid": cx[:, 10:22],
        "xa": xa,
        "ya": ya,
        "ya18": ya[:, :18],
        "xs": xs,
        "c": c,
        "c4": c[:4],
        "c_tall": c[:, :5].copy(),
    }

    def run(name, X=None, Y=None, C=None, **kw):
        if C is None:
            D, wp, steps = librosa.sequence.dtw(inputs[X], inputs[Y], return_steps=True, **kw)
            refs = {"X": X, "Y": Y}
        else:
            D, wp, steps = librosa.sequence.dtw(C=inputs[C], return_steps=True, **kw)
            refs = {"C": C}
        cfg = {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in kw.items()}
        cases[name] = {**refs, "config": cfg, "D": _finite_or_none(D), "wp": wp, "steps": steps}

    run("chroma_default", "cx", "cy")
    run("chroma_subseq", "cx_mid", "cy", subseq=True)

    for metric in ("euclidean", "sqeuclidean", "cosine", "cityblock", "chebyshev"):
        run(f"rand_{metric}", "xa", "ya", metric=metric)
    run("rand_weights_mul", "xa", "ya", weights_mul=np.array([2.0, 1.0, 1.0]))
    run("rand_weights_add", "xa", "ya", weights_add=np.array([0.0, 0.3, 0.2]))
    run(
        "rand_custom_steps",
        "xa",
        "ya18",
        step_sizes_sigma=np.array([[2, 1], [1, 2]]),
        weights_add=np.array([0.1, 0.2]),
        weights_mul=np.array([1.5, 1.0]),
    )
    run("rand_custom_steps_noweights", "xa", "ya", step_sizes_sigma=np.array([[1, 2], [2, 1], [1, 1]]))
    run("rand_band_10", "xa", "ya", global_constraints=True, band_rad=0.1)
    run("rand_band_tall", "ya", "xa", global_constraints=True, band_rad=0.3)

    run("rand_subseq", "xs", "ya", subseq=True)
    # X longer than Y: librosa transposes C and flips the path back.
    run("rand_subseq_transposed", "ya", "xs", subseq=True)
    run("rand_subseq_band", "xs", "ya", subseq=True, global_constraints=True, band_rad=0.5)

    run("cost_matrix", C="c")
    run("cost_matrix_subseq", C="c4", subseq=True)
    # Precomputed C with more rows than columns under subseq: librosa flips the
    # path columns without transposing C.
    run("cost_matrix_subseq_tall", C="c_tall", subseq=True)
    run("cost_matrix_band", C="c", global_constraints=True, band_rad=0.2)
    return {"inputs": inputs, "cases": cases}


def _backtracking_cases(rng) -> dict:
    out = {}
    c = rng.uniform(0.0, 1.0, (6, 15))
    D, steps = librosa.sequence.dtw(C=c, subseq=True, backtrack=False, return_steps=True)
    for start in (3, 9, 14):
        out[f"subseq_start_{start}"] = {
            "steps": steps,
            "subseq": True,
            "start": start,
            "step_sizes_sigma": None,
            "wp": librosa.sequence.dtw_backtracking(steps, subseq=True, start=start),
        }
    c2 = rng.uniform(0.0, 1.0, (10, 13))
    sigma = np.array([[2, 1], [1, 2]])
    _, steps2 = librosa.sequence.dtw(C=c2, step_sizes_sigma=sigma, backtrack=False, return_steps=True)
    out["custom_full"] = {
        "steps": steps2,
        "subseq": False,
        "start": None,
        "step_sizes_sigma": sigma,
        "wp": librosa.sequence.dtw_backtracking(steps2, step_sizes_sigma=sigma),
    }
    return out


def _viterbi_cases(rng) -> dict:
    out = {}

    # Wikipedia example from the librosa docstring.
    p_init = np.array([0.6, 0.4])
    p_emit = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
    p_trans = np.array([[0.7, 0.3], [0.4, 0.6]])
    s, lp = librosa.sequence.viterbi(p_emit, p_trans, p_init=p_init, return_logp=True)
    out["wikipedia"] = {"prob": p_emit, "transition": p_trans, "p_init": p_init, "min_prob": None, "states": s, "logp": float(lp[0])}

    n_states, n_steps = 6, 40
    prob = rng.uniform(0.0, 1.0, (n_states, n_steps))
    trans = librosa.sequence.transition_loop(n_states, 0.8)
    s, lp = librosa.sequence.viterbi(prob, trans, return_logp=True)
    out["loop_uniform_init"] = {"prob": prob, "transition": trans, "p_init": None, "min_prob": None, "states": s, "logp": float(lp[0])}

    pi = rng.uniform(0.0, 1.0, n_states)
    pi /= pi.sum()
    trans2 = rng.uniform(0.0, 1.0, (n_states, n_states))
    trans2 /= trans2.sum(axis=1, keepdims=True)
    s, lp = librosa.sequence.viterbi(prob, trans2, p_init=pi, return_logp=True)
    out["random_trans_init"] = {"prob": prob, "transition": trans2, "p_init": pi, "min_prob": None, "states": s, "logp": float(lp[0])}

    # Zero probabilities exercise the log(x + tiny) clipping.
    prob0 = prob.copy()
    prob0[prob0 < 0.2] = 0.0
    tl = librosa.sequence.transition_local(n_states, 3)
    s, lp = librosa.sequence.viterbi(prob0, tl, return_logp=True)
    out["local_zeros"] = {"prob": prob0, "transition": tl, "p_init": None, "min_prob": None, "states": s, "logp": float(lp[0])}

    s, lp = librosa.sequence.viterbi(prob, trans2, return_logp=True, transition_min_prob=0.15)
    out["min_prob"] = {"prob": prob, "transition": trans2, "p_init": None, "min_prob": 0.15, "states": s, "logp": float(lp[0])}

    disc = {}
    pd = rng.uniform(0.0, 1.0, (n_states, n_steps))
    pd /= pd.sum(axis=0, keepdims=True)
    tc = librosa.sequence.transition_cycle(n_states, 0.7)
    s, lp = librosa.sequence.viterbi_discriminative(pd, trans, return_logp=True)
    disc["default"] = {"prob": pd, "transition": trans, "p_state": None, "p_init": None, "min_prob": None, "states": s, "logp": float(lp)}
    ps = rng.uniform(0.1, 1.0, n_states)
    ps /= ps.sum()
    s, lp = librosa.sequence.viterbi_discriminative(pd, tc, p_state=ps, p_init=pi, return_logp=True)
    disc["cycle_state_init"] = {"prob": pd, "transition": tc, "p_state": ps, "p_init": pi, "min_prob": None, "states": s, "logp": float(lp)}
    s, lp = librosa.sequence.viterbi_discriminative(pd, trans2, p_state=ps, return_logp=True, transition_min_prob=0.1)
    disc["min_prob"] = {"prob": pd, "transition": trans2, "p_state": ps, "p_init": None, "min_prob": 0.1, "states": s, "logp": float(lp)}

    binary = {}
    pb = rng.uniform(0.0, 1.0, (4, 30))
    t2 = librosa.sequence.transition_loop(2, 0.9)
    s, lp = librosa.sequence.viterbi_binary(pb, t2, return_logp=True)
    binary["shared"] = {"prob": pb, "transition": t2[None], "p_state": None, "p_init": None, "min_prob": None, "states": s, "logp": lp}
    tper = np.stack([librosa.sequence.transition_loop(2, [a, b]) for a, b in ((0.9, 0.8), (0.6, 0.7), (0.95, 0.5), (0.7, 0.7))])
    bstate = np.array([0.3, 0.5, 0.2, 0.8])
    binit = np.array([0.1, 0.9, 0.5, 0.4])
    s, lp = librosa.sequence.viterbi_binary(pb, tper, p_state=bstate, p_init=binit, return_logp=True)
    binary["per_state"] = {"prob": pb, "transition": tper, "p_state": bstate, "p_init": binit, "min_prob": None, "states": s, "logp": lp}
    return {"viterbi": out, "viterbi_discriminative": disc, "viterbi_binary": binary}


def _transition_cases() -> dict:
    tr = librosa.sequence
    cases = {
        "uniform_5": ({"kind": "uniform", "n": 5}, tr.transition_uniform(5)),
        "loop_5_scalar": ({"kind": "loop", "n": 5, "prob": [0.7]}, tr.transition_loop(5, 0.7)),
        "loop_4_vector": ({"kind": "loop", "n": 4, "prob": [0.1, 0.5, 0.9, 1.0]}, tr.transition_loop(4, [0.1, 0.5, 0.9, 1.0])),
        "cycle_5_scalar": ({"kind": "cycle", "n": 5, "prob": [0.6]}, tr.transition_cycle(5, 0.6)),
        "cycle_3_vector": ({"kind": "cycle", "n": 3, "prob": [0.2, 0.5, 0.0]}, tr.transition_cycle(3, [0.2, 0.5, 0.0])),
    }
    for window in ("triangle", "hann", "hamming", "blackman", "boxcar"):
        for wrap in (False, True):
            for width in ([3], [5], [1, 2, 3, 4, 5, 6, 7, 8]):
                if window in ("hann", "blackman") and width != [5]:
                    # symmetric hann/blackman of width <= 2 are all-zero rows
                    continue
                w = width[0] if len(width) == 1 else width
                key = f"local_{window}_{'wrap' if wrap else 'nowrap'}_{'x'.join(map(str, width)) if len(width) == 1 else 'vec'}"
                cases[key] = (
                    {"kind": "local", "n": 8, "width": width, "window": window, "wrap": wrap},
                    tr.transition_local(8, w, window=window, wrap=wrap),
                )
    return {k: {"args": a, "T": t} for k, (a, t) in cases.items()}


def _rqa_cases(rng) -> dict:
    out = {}
    sims = {}
    X = rng.standard_normal((3, 25))
    sim_bin = librosa.segment.recurrence_matrix(X, width=1, k=6, sym=False).astype(float)
    Y = np.cumsum(rng.standard_normal((3, 30)), axis=1)
    sim_aff = librosa.segment.cross_similarity(Y, Y[:, ::-1][:, :22] + 0.3 * rng.standard_normal((3, 22)), mode="affinity", k=5)
    sims = {"binary": sim_bin, "affinity": sim_aff}
    for sname, sim in sims.items():
        for knight in (True, False):
            gaps = ((1.0, 1.0), (0.5, 0.25), (3.0, 0.0)) if sname == "binary" else ((1.0, 1.0), (0.5, 0.25))
            for go, ge in gaps:
                score, path = librosa.sequence.rqa(sim, gap_onset=go, gap_extend=ge, knight_moves=knight)
                key = f"{sname}_{'knight' if knight else 'diag'}_{go}_{ge}"
                out[key] = {
                    "sim": sname,
                    "gap_onset": _finite_or_none(go),
                    "gap_extend": _finite_or_none(ge),
                    "knight_moves": knight,
                    "score": score,
                    "path": path.astype(np.int64),
                }
    return {"sims": sims, "cases": out}


def _segment_cases(rng) -> dict:
    rec_out = {}
    t = 30
    X = rng.standard_normal((4, t))
    configs = [
        ("connectivity", {}),
        ("connectivity_w3_sym", {"width": 3, "sym": True}),
        ("connectivity_self", {"self": True, "k": 5}),
        ("distance", {"mode": "distance"}),
        ("distance_sym_k4", {"mode": "distance", "sym": True, "k": 4}),
        ("distance_full", {"mode": "distance", "full": True}),
        ("affinity", {"mode": "affinity"}),
        ("affinity_self_sym", {"mode": "affinity", "self": True, "sym": True}),
        ("affinity_mean_k", {"mode": "affinity", "bandwidth": "mean_k"}),
        ("affinity_gmean_k", {"mode": "affinity", "bandwidth": "gmean_k"}),
        ("affinity_mean_k_avg", {"mode": "affinity", "bandwidth": "mean_k_avg"}),
        ("affinity_mean_k_avg_and_pair", {"mode": "affinity", "bandwidth": "mean_k_avg_and_pair", "width": 2}),
        ("affinity_full", {"mode": "affinity", "full": True}),
        ("cosine_distance", {"mode": "distance", "metric": "cosine"}),
        ("cityblock_affinity", {"mode": "affinity", "metric": "cityblock", "k": 6}),
    ]
    for name, kw in configs:
        rec_out[name] = {"config": kw, "R": librosa.segment.recurrence_matrix(X, **kw).astype(float)}

    xs_out = {}
    ref = rng.standard_normal((4, 30))
    data = rng.standard_normal((4, 22))
    xconfigs = [
        ("connectivity", {}),
        ("distance_k3", {"mode": "distance", "k": 3}),
        ("affinity", {"mode": "affinity"}),
        ("affinity_bw", {"mode": "affinity", "bandwidth": 1.5}),
        ("affinity_full", {"mode": "affinity", "full": True}),
        ("cosine_connectivity", {"metric": "cosine", "k": 4}),
    ]
    for name, kw in xconfigs:
        xs_out[name] = {"config": kw, "R": librosa.segment.cross_similarity(data, ref, **kw).astype(float)}
    # Per-pair bandwidths need n_ref <= n in librosa (they index per-row statistics by column).
    xs_out["affinity_mean_k_short_ref"] = {
        "config": {"mode": "affinity", "bandwidth": "mean_k", "ref_cols": 18},
        "R": librosa.segment.cross_similarity(data, ref[:, :18], mode="affinity", bandwidth="mean_k").astype(float),
    }

    lag_out = {}
    rec = librosa.segment.recurrence_matrix(X, mode="affinity")
    for pad in (True, False):
        lag = librosa.segment.recurrence_to_lag(rec, pad=pad)
        # lag_to_recurrence inverts recurrence_to_lag exactly; the Rust side checks the round trip.
        assert np.array_equal(librosa.segment.lag_to_recurrence(lag), rec)
        lag_out[f"pad_{pad}"] = {"pad": pad, "lag": lag}

    return {
        "segment_input": X,
        "cross_data": data,
        "cross_ref": ref,
        "recurrence": rec_out,
        "cross_similarity": xs_out,
        "lag": {"rec": rec, **lag_out},
    }


def _sync_cases(rng) -> dict:
    out = {}
    data = rng.standard_normal((3, 20))
    aggs = {"mean": np.mean, "median": np.median, "max": np.max, "min": np.min, "sum": np.sum}
    idx_sets = {
        "basic": [2, 5, 9, 15],
        "unsorted_dupes": [9, 2, 5, 5, 15, 2],
        "beyond_end": [4, 12, 25],
        "with_bounds": [0, 7, 20],
        "empty": [],
    }
    for iname, idx in idx_sets.items():
        for pad in (True, False):
            for aname, agg in aggs.items():
                if not pad and len(idx) < 2:
                    continue
                res = librosa.util.sync(data, np.asarray(idx, dtype=int), aggregate=agg, pad=pad)
                out[f"{iname}_{aname}_{'pad' if pad else 'nopad'}"] = {
                    "idx": idx,
                    "pad": pad,
                    "aggregate": aname,
                    "out": res,
                }
    return {"sync_input": data, "sync": out}


def _float32_cases(rng) -> dict:
    """DTW and Viterbi on float32 inputs.

    Inputs are rounded to float32 and stored (exact as float64 JSON values).
    librosa computes the DTW cost through ``scipy.spatial.distance.cdist``,
    which works in float64, and accumulates Viterbi log-likelihoods in float64
    (``log(prob + tiny(float32))`` is float32). The transition matrix stays
    float64 here, as librosa's builders return; the Rust side rounds it to f32.
    """
    x = rng.standard_normal((12, 40)).astype(np.float32)
    y = rng.standard_normal((12, 55)).astype(np.float32)
    D, wp = librosa.sequence.dtw(x, y)
    prob = rng.uniform(0.0, 1.0, (8, 120)).astype(np.float32)
    trans = librosa.sequence.transition_local(8, 3)
    states, logp = librosa.sequence.viterbi(prob, trans, return_logp=True)
    trans_loop = librosa.sequence.transition_loop(8, 0.9)
    states2, logp2 = librosa.sequence.viterbi(prob, trans_loop, return_logp=True)
    return {
        "dtw": {"X": x, "Y": y, "D": D, "wp": wp},
        "viterbi": {
            "prob": prob,
            "local": {"transition": trans, "states": states, "logp": float(logp[0])},
            "loop": {"transition": trans_loop, "states": states2, "logp": float(logp2[0])},
        },
    }


def build(signals, sr) -> dict:
    rng = np.random.default_rng(20261009)
    out = {"dtw": _dtw_cases(signals, sr, rng), "dtw_backtracking": _backtracking_cases(rng)}
    out.update(_viterbi_cases(rng))
    out["transition"] = _transition_cases()
    out["rqa"] = _rqa_cases(rng)
    out.update(_segment_cases(rng))
    out.update(_sync_cases(rng))
    out["float32"] = _float32_cases(rng)
    return out
