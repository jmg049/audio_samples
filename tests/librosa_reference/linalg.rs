//! Golden-reference checks for `operations::linalg`.
//!
//! A vector right-hand side has a unique solution, so `nnls` is held to 1e-9 of the peak
//! against SciPy's active-set solver. On a matrix right-hand side the reference stops
//! L-BFGS-B early on a problem with many solutions, so the check is that the objective
//! is not above the reference's. Eigenvalues and singular values agree at round-off;
//! the Gram route loses half the digits for the smallest singular vectors, whose
//! products are compared at 1e-9.

use audio_samples::nzu;
use audio_samples::operations::filterbanks::{MelFilterbankConfig, mel_filterbank};
use audio_samples::operations::linalg::{NnlsConfig, eigh, nnls, nnls_matrix, pinv, svd};
use ndarray::Array1;

use crate::common::{arr1, arr2, load};

fn fx() -> serde_json::Value {
    load("linalg")
}

fn close(a: &[f64], b: &[f64], rtol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    let peak = b
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    let err = a
        .iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    assert!(err <= rtol * peak, "{what}: {err:e} > {rtol:e} of {peak}");
}

#[test]
fn nnls_vector_matches_active_set() {
    let f = fx();
    for key in ["nnls_vector", "nnls_vector_exact"] {
        let c = &f[key];
        let x: Array1<f64> = nnls(arr2(&c["a"]).view(), arr1(&c["b"]).view()).unwrap();
        close(
            x.as_slice().unwrap(),
            arr1(&c["x"]).as_slice().unwrap(),
            1e-9,
            key,
        );
    }
}

#[test]
fn nnls_matrix_objective_is_no_worse() {
    let f = fx();
    let c = &f["nnls_matrix"];
    let mel = mel_filterbank(
        22050.0,
        nzu!(512),
        &MelFilterbankConfig::new().with_n_mels(nzu!(32)),
    )
    .unwrap();
    let b = arr2(&c["b"]);
    let x = nnls_matrix(mel.view(), b.view(), &NnlsConfig::new()).unwrap();
    assert!(x.iter().all(|&v| v >= 0.0));
    let objective = 0.5 * (&mel.dot(&x) - &b).mapv(|v| v * v).sum();
    let reference = c["objective"].as_f64().unwrap();
    assert!(
        objective <= reference * 1.0001,
        "{objective} vs {reference}"
    );
}

#[test]
fn pinv_matches() {
    let f = fx();
    let c = &f["pinv"];
    let p = pinv(arr2(&c["a"]).view(), 1e-15).unwrap();
    close(
        p.as_slice().unwrap(),
        arr2(&c["p"]).as_slice().unwrap(),
        1e-11,
        "pinv",
    );
}

#[test]
fn eigh_matches() {
    let f = fx();
    let c = &f["eigh"];
    let (w, v) = eigh(arr2(&c["a"]).view()).unwrap();
    close(
        w.as_slice().unwrap(),
        arr1(&c["w"]).as_slice().unwrap(),
        1e-13,
        "values",
    );
    // Each eigenvector is defined up to its sign.
    let want = arr2(&c["v"]);
    for j in 0..v.ncols() {
        let d = v.column(j).dot(&want.column(j)).abs();
        assert!((d - 1.0).abs() < 1e-11, "{j}: {d}");
    }
}

#[test]
fn svd_matches() {
    let f = fx();
    let c = &f["svd"];
    let a = arr2(&c["a"]);
    let d = svd(a.view()).unwrap();
    close(
        d.s.as_slice().unwrap(),
        arr1(&c["s"]).as_slice().unwrap(),
        1e-13,
        "s",
    );
    let back = d.u.dot(&ndarray::Array2::from_diag(&d.s)).dot(&d.vt);
    close(
        back.as_slice().unwrap(),
        a.as_slice().unwrap(),
        1e-9,
        "product",
    );
}

#[test]
fn single_precision_vector_solution() {
    let f = fx();
    let c = &f["nnls_vector"];
    let a = arr2(&c["a"]).mapv(|v| v as f32);
    let b = arr1(&c["b"]).mapv(|v| v as f32);
    let x = nnls(a.view(), b.view()).unwrap();
    let want = arr1(&c["x"]);
    // Single-precision inputs change the solution by the rounding of the data.
    for (p, q) in x.iter().zip(want.iter()) {
        assert!((f64::from(*p) - q).abs() < 1e-5);
    }
}
