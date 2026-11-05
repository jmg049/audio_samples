//! Golden-reference checks for `operations::mellin`.
//!
//! The grid, the spline and the transform are the same computations as the reference, so
//! the results agree to 1e-9 of the peak; the spline's linear system is solved by a
//! different elimination and the grid by a different power function.

use std::num::NonZeroUsize;

use audio_samples::operations::mellin::{MellinConfig, mellin_transform};
use audio_samples::utils::array_ops::InterpKind;
use ndarray::{Array1, Array2, Axis};
use num_complex::Complex;
use serde_json::Value;

use crate::common::{f64s, load};

fn complex(v: &Value) -> Vec<Complex<f64>> {
    let (re, im) = (f64s(&v["re"]), f64s(&v["im"]));
    re.into_iter()
        .zip(im)
        .map(|(r, i)| Complex::new(r, i))
        .collect()
}

fn close(a: &[Complex<f64>], b: &[Complex<f64>], rtol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    let peak = b.iter().fold(0.0_f64, |m, v| m.max(v.norm()));
    let err = a
        .iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).norm()));
    assert!(err <= rtol * peak, "{what}: {err:e} of {peak:e}");
}

#[test]
fn cases() {
    let f = load("mellin");
    let y = Array1::from(f64s(&f["y"])).into_dyn();
    for (name, c) in f["cases"].as_object().unwrap() {
        let kw = &c["kw"];
        let mut cfg = MellinConfig::new();
        if let Some(v) = kw["beta"].as_f64() {
            cfg = cfg.with_beta(v);
        }
        if let Some(v) = kw["t_min"].as_f64() {
            cfg = cfg.with_t_min(v);
        }
        if let Some(v) = kw["n_fmt"].as_u64() {
            cfg = cfg.with_n_fmt(Some(NonZeroUsize::new(v as usize).unwrap().get()));
        }
        if let Some(v) = kw["over_sample"].as_f64() {
            cfg = cfg.with_over_sample(v);
        }
        cfg = cfg.with_kind(match kw["kind"].as_str() {
            Some("linear") => InterpKind::Linear,
            Some("quadratic") => InterpKind::Quadratic,
            _ => InterpKind::Cubic,
        });
        let r = mellin_transform(y.view(), &cfg, Axis(0)).unwrap();
        assert_eq!(r.len() as u64, c["n"].as_u64().unwrap(), "{name}");
        let got: Vec<_> = r.iter().take(64).copied().collect();
        close(&got, &complex(&c["out"]), 1e-9, name);
    }
}

#[test]
fn stereo_along_the_last_axis() {
    let f = load("mellin");
    let y = f64s(&f["y"]);
    let rev: Vec<f64> = y.iter().rev().copied().collect();
    let mut data = y.clone();
    data.extend(rev);
    let s = Array2::from_shape_vec((2, y.len()), data)
        .unwrap()
        .into_dyn();
    let r = mellin_transform(
        s.view(),
        &MellinConfig::new().with_n_fmt(Some(128)),
        Axis(1),
    )
    .unwrap();
    let got: Vec<_> = r.iter().copied().collect();
    let want = &f["stereo"];
    let re = crate::common::arr2(&want["re"]);
    let im = crate::common::arr2(&want["im"]);
    let w: Vec<_> = re
        .iter()
        .zip(im.iter())
        .map(|(a, b)| Complex::new(*a, *b))
        .collect();
    close(&got, &w, 1e-9, "stereo");
}

#[test]
fn single_precision_output() {
    let f = load("mellin");
    let y: Array1<f32> = f64s(&f["y"]).iter().map(|&v| v as f32).collect();
    let r = mellin_transform(y.into_dyn().view(), &MellinConfig::new(), Axis(0)).unwrap();
    assert!(r.iter().all(|z: &Complex<f32>| z.re.is_finite()));
}
