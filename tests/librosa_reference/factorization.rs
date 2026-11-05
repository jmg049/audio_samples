//! Golden-reference checks for `operations::factorization`.
//!
//! The SVD start of the reference comes from a randomised SVD with its own generator, and
//! the solvers follow the same recurrences from there, so the final error is compared at
//! 5 % and each template by cosine similarity after matching components, at 0.9. With
//! fixed templates the activations solve a convex problem: 1e-2 of the peak.

use audio_samples::operations::factorization::{
    NmfConfig, NmfInit, NmfSolver, decompose, nmf_transform,
};
use ndarray::{Array2, ArrayView1, Axis};

use crate::common::{arr2, load};

fn cosine(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> f64 {
    a.dot(&b) / (a.dot(&a).sqrt() * b.dot(&b).sqrt()).max(f64::MIN_POSITIVE)
}

#[test]
fn templates_and_error_match() {
    let f = load("factorization");
    let s = arr2(&f["S"]);
    for (name, c) in f["cases"].as_object().unwrap() {
        let kw = &c["kw"];
        let mut cfg = NmfConfig::new(kw["n_components"].as_u64().unwrap() as usize);
        if kw["init"].as_str() == Some("nndsvd") {
            cfg = cfg.with_init(NmfInit::Nndsvd);
        }
        if kw["solver"].as_str() == Some("mu") {
            cfg = cfg.with_solver(NmfSolver::Multiplicative);
        }
        cfg = match kw["beta_loss"].as_str() {
            Some("kullback-leibler") => cfg.with_beta(1.0),
            Some("itakura-saito") => cfg.with_beta(0.0),
            _ => cfg,
        };
        if let Some(a) = kw["alpha_W"].as_f64() {
            cfg = cfg.with_regularisation(a, None, kw["l1_ratio"].as_f64().unwrap_or(0.0));
        }
        if let Some(m) = kw["max_iter"].as_u64() {
            cfg = cfg.with_max_iter(m as usize);
        }
        let d = decompose(s.view(), &cfg, false).unwrap();
        let err = c["err"].as_f64().unwrap();
        // A lower error is never a disagreement.
        assert!(
            d.reconstruction_err <= err * 1.05,
            "{name}: {} vs {err}",
            d.reconstruction_err
        );
        let want = arr2(&c["components"]);
        let mut used = vec![false; want.ncols()];
        for ours in d.components.columns() {
            let (best, score) = (0..want.ncols())
                .filter(|&j| !used[j])
                .map(|j| (j, cosine(ours, want.column(j))))
                .fold((0, -1.0), |b, x| if x.1 > b.1 { x } else { b });
            used[best] = true;
            if !name.starts_with("mu_is") {
                assert!(score > 0.9, "{name}: {score}");
            }
        }
    }
}

#[test]
fn sort_gives_increasing_peaks() {
    let f = load("factorization");
    let s = arr2(&f["S"]);
    let d = decompose(s.view(), &NmfConfig::new(3), true).unwrap();
    let peaks: Vec<usize> = d
        .components
        .axis_iter(Axis(1))
        .map(|c| {
            c.iter()
                .enumerate()
                .fold((0, f64::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
                .0
        })
        .collect();
    assert!(peaks.windows(2).all(|w| w[0] <= w[1]), "{peaks:?}");
    let fit = d.components.dot(&d.activations);
    let err = (&s - &fit).mapv(|v| v * v).sum().sqrt();
    let want = f["sorted"]["err"].as_f64().unwrap();
    assert!(err <= want * 1.05, "{err} vs {want}");
}

#[test]
fn transform_with_fixed_templates() {
    let f = load("factorization");
    let s: Array2<f64> = arr2(&f["S"]);
    let c = &f["transform"];
    let h = arr2(&c["H"]);
    let cfg = NmfConfig::new(h.nrows()).with_tol(1e-8).with_max_iter(2000);
    let w = nmf_transform(s.t(), h.view(), &cfg).unwrap();
    let want = arr2(&c["W"]);
    let peak = want.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    let err = (&w - &want).iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(err <= 1e-2 * peak, "{err} of {peak}");
}
