//! Golden-reference checks for `operations::segment` and `sequence::path_to_steps`.
//!
//! Kernels and convolutions are the same weighted sums as the reference, so they agree to
//! 1e-9 of the peak; rotation angles and spline weights differ at round-off, which is
//! amplified by the normalisation. Segment boundaries must be exact.

use audio_samples::operations::filterbanks::Window;
use audio_samples::operations::segment::{
    BoundaryMode, PathEnhanceConfig, agglomerative, diagonal_filter, path_enhance, subsegment,
    timelag_filter,
};
use audio_samples::operations::sequence::path_to_steps;
use ndarray::{Array2, Axis};
use serde_json::Value;

use crate::common::{arr2, f64s, load};

fn fx() -> Value {
    load("segment")
}

fn close(a: &Array2<f64>, b: &Array2<f64>, rtol: f64, what: &str) {
    assert_eq!(a.dim(), b.dim(), "{what}");
    let peak = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let err = (a - b).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(err <= rtol * peak, "{what}: {err:e} of {peak:e}");
}

fn usizes(v: &Value) -> Vec<usize> {
    f64s(v).into_iter().map(|x| x as usize).collect()
}

fn window(v: &Value) -> Window {
    match v.as_str() {
        Some("hann") => Window::Hann,
        Some("hamming") => Window::Hamming,
        Some("boxcar") => Window::Rectangular,
        _ => Window::Kaiser {
            beta: v[1].as_f64().unwrap(),
        },
    }
}

#[test]
fn diagonal_filters() {
    let f = fx();
    for (name, c) in f["diagonal_filter"].as_object().unwrap() {
        let kw = &c["kw"];
        let k = diagonal_filter(
            &window(&c["window"]),
            c["n"].as_u64().unwrap() as usize,
            kw["slope"].as_f64().unwrap_or(1.0),
            kw["angle"].as_f64(),
            kw["zero_mean"].as_bool().unwrap_or(false),
        )
        .unwrap();
        close(&k, &arr2(&c["k"]), 1e-9, name);
    }
}

#[test]
fn path_enhancement() {
    let f = fx();
    let c = &f["path_enhance"];
    let r = arr2(&c["R"]);
    for (name, case) in c.as_object().unwrap() {
        if name == "R" {
            continue;
        }
        let kw = &case["kw"];
        let mut cfg = PathEnhanceConfig::new().with_ratios(
            kw["max_ratio"].as_f64().unwrap_or(2.0),
            kw["min_ratio"].as_f64(),
            kw["n_filters"].as_u64().unwrap_or(7) as usize,
        );
        cfg = cfg
            .with_zero_mean(kw["zero_mean"].as_bool().unwrap_or(false))
            .with_clip(kw["clip"].as_bool().unwrap_or(true));
        if kw["window"].as_str() == Some("boxcar") {
            cfg = cfg.with_window(Window::Rectangular);
        }
        if let Some(o) = kw["origin"].as_i64() {
            cfg = cfg.with_origin((o as isize, o as isize));
        }
        cfg = cfg.with_mode(match kw["mode"].as_str() {
            Some("constant") => BoundaryMode::Constant(kw["cval"].as_f64().unwrap()),
            Some("nearest") => BoundaryMode::Nearest,
            Some("mirror") => BoundaryMode::Mirror,
            Some("wrap") => BoundaryMode::Wrap,
            _ => BoundaryMode::Reflect,
        });
        let e = path_enhance(r.view(), kw["n"].as_u64().unwrap() as usize, &cfg).unwrap();
        close(&e, &arr2(&case["out"]), 1e-9, name);
    }
}

#[test]
fn clustering() {
    let f = fx();
    let c = &f["agglomerative"];
    let x = arr2(&c["x"]);
    for k in [1_usize, 2, 3, 5, 12, 80] {
        let b = agglomerative(x.view(), k).unwrap();
        assert_eq!(b, usizes(&c[k.to_string()]), "{k}");
    }
    let row = x.row(0).insert_axis(Axis(1)).reversed_axes();
    assert_eq!(agglomerative(row.view(), 4).unwrap(), usizes(&c["one_d"]));
    let s = &f["subsegment"];
    let frames = [10_i64, 40, 70];
    assert_eq!(
        subsegment(x.view(), &frames, 4).unwrap(),
        usizes(&s["default"])
    );
    assert_eq!(subsegment(x.view(), &frames, 2).unwrap(), usizes(&s["n2"]));
    assert_eq!(
        subsegment(x.view(), &[3, 5], 4).unwrap(),
        usizes(&s["many"])
    );
    let x32 = x.mapv(|v| v as f32);
    assert_eq!(agglomerative(x32.view(), 3).unwrap(), usizes(&c["3"]));
}

#[test]
fn timelag_with_a_running_median() {
    let f = fx();
    let c = &f["timelag"];
    let rec = arr2(&c["rec"]);
    let got = timelag_filter(rec.view(), true, |lag| {
        // A 1 x 3 median along time with mirrored ends.
        let (n, t) = lag.dim();
        let mut out = Array2::zeros((n, t));
        for r in 0..n {
            for j in 0..t {
                let at = |i: isize| {
                    let q = i.rem_euclid(2 * t as isize);
                    let q = if q < t as isize {
                        q
                    } else {
                        2 * t as isize - 1 - q
                    };
                    lag[[r, q as usize]]
                };
                let mut v = [at(j as isize - 1), at(j as isize), at(j as isize + 1)];
                v.sort_by(f64::total_cmp);
                out[[r, j]] = v[1];
            }
        }
        Ok(out)
    })
    .unwrap();
    close(&got, &arr2(&c["median"]), 1e-12, "median");
}

#[test]
fn steps() {
    let f = fx();
    let c = &f["path_to_steps"];
    let parse = |v: &Value| -> Vec<[usize; 2]> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let u = usizes(p);
                [u[0], u[1]]
            })
            .collect()
    };
    let path = parse(&c["path"]);
    assert_eq!(
        path_to_steps(&path, false).unwrap().to_vec(),
        f64s(&c["forward"])
    );
    assert_eq!(
        path_to_steps(&path, true).unwrap().to_vec(),
        f64s(&c["inverse"])
    );
    let gap = parse(&c["gap_path"]);
    let g = path_to_steps(&gap, false).unwrap().to_vec();
    let want = f64s(&c["gap_forward"]);
    for (a, b) in g.iter().zip(&want) {
        assert!((a - b).abs() < 1e-12);
    }
    assert_eq!(g.len(), want.len());
    let g = path_to_steps(&gap, true).unwrap().to_vec();
    let want = f64s(&c["gap_inverse"]);
    for (a, b) in g.iter().zip(&want) {
        assert!((a - b).abs() < 1e-12);
    }
    assert!(path_to_steps(&[[0, 0]], false).is_err());
}
