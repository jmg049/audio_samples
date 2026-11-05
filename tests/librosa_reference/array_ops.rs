//! Golden-reference checks for `utils::array_ops` and the block conversions.
//!
//! Index and boolean outputs must match exactly. The value outputs are single
//! arithmetic operations on the same inputs, so they agree to round-off: 1e-15 of the
//! peak; spline interpolation solves a linear system by a different elimination, and is
//! held to 1e-10 of the peak.

use std::num::NonZeroUsize;

use audio_samples::utils::array_ops::*;
use audio_samples::utils::audio_math::{blocks_to_frames, blocks_to_samples, blocks_to_time};
use ndarray::{Array2, ArrayD, Axis};
use num_complex::Complex;
use serde_json::Value;

use crate::common::{arr2, f64s, load};

const RTOL: f64 = 1e-15;
const SPLINE_RTOL: f64 = 1e-10;

fn fixture() -> Value {
    load("array_ops")
}

fn close(a: &[f64], b: &[f64], rtol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    let peak = b
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= rtol * peak, "{what}: {x} vs {y}");
    }
}

fn bools(a: &Array2<bool>) -> Array2<f64> {
    a.mapv(|v| if v { 1.0 } else { 0.0 })
}

fn usizes(v: &Value) -> Vec<usize> {
    f64s(v).into_iter().map(|x| x as usize).collect()
}

fn i64s(v: &Value) -> Vec<i64> {
    f64s(v).into_iter().map(|x| x as i64).collect()
}

#[test]
fn local_extrema() {
    let f = &fixture()["local"];
    let x = arr2(&f["x"]);
    assert_eq!(
        bools(&localmax(&x, Axis(0)).unwrap()),
        arr2(&f["max_axis0"])
    );
    assert_eq!(
        bools(&localmax(&x, Axis(1)).unwrap()),
        arr2(&f["max_axis1"])
    );
    assert_eq!(
        bools(&localmin(&x, Axis(0)).unwrap()),
        arr2(&f["min_axis0"])
    );
    assert_eq!(
        bools(&localmin(&x, Axis(1)).unwrap()),
        arr2(&f["min_axis1"])
    );
}

#[test]
fn events() {
    let f = &fixture()["match_events"];
    let (src, dst) = (f64s(&f["from"]), f64s(&f["to"]));
    assert_eq!(
        match_events(&src, &dst, true, true).unwrap(),
        usizes(&f["both"])
    );
    let inner = f64s(&f["from_inner"]);
    assert_eq!(
        match_events(&inner, &dst, false, true).unwrap(),
        usizes(&f["no_left"])
    );
    assert_eq!(
        match_events(&inner, &dst, true, false).unwrap(),
        usizes(&f["no_right"])
    );
    assert_eq!(
        match_events(
            &f64s(&f["subset_from"]),
            &f64s(&f["subset_to"]),
            false,
            false
        )
        .unwrap(),
        usizes(&f["exact"])
    );
}

#[test]
fn intervals() {
    let f = &fixture()["match_intervals"];
    let m = match_intervals(arr2(&f["from"]).view(), arr2(&f["to"]).view(), true).unwrap();
    assert_eq!(m, usizes(&f["strict"]));
    let g = match_intervals(
        arr2(&f["gap_from"]).view(),
        arr2(&f["gap_to"]).view(),
        false,
    )
    .unwrap();
    assert_eq!(g, usizes(&f["gap_non_strict"]));
}

#[test]
fn complex_helpers() {
    let f = &fixture()["complex"];
    let (re, im) = (arr2(&f["z_re"]), arr2(&f["z_im"]));
    let z = ndarray::Zip::from(&re)
        .and(&im)
        .map_collect(|&r, &i| Complex::new(r, i));
    let a = abs2(&z);
    close(
        a.as_slice().unwrap(),
        arr2(&f["abs2"]).as_slice().unwrap(),
        RTOL * 4.0,
        "abs2",
    );
    let ang = arr2(&f["angles"]);
    let p = phasor(&ang);
    close(
        p.mapv(|v| v.re).as_slice().unwrap(),
        arr2(&f["phasor_re"]).as_slice().unwrap(),
        RTOL,
        "phasor re",
    );
    close(
        p.mapv(|v| v.im).as_slice().unwrap(),
        arr2(&f["phasor_im"]).as_slice().unwrap(),
        RTOL,
        "phasor im",
    );
    let t = &fixture()["tiny"];
    assert_eq!(f64::from(tiny::<f32>()), t["f32"].as_f64().unwrap());
    assert_eq!(tiny::<f64>(), t["f64"].as_f64().unwrap());
}

#[test]
fn padding() {
    let f = &fixture()["pad_center"];
    let x = ndarray::Array1::from(f64s(&f["x"]));
    for (name, mode) in [
        ("constant", PadMode::Constant(0.0)),
        ("edge", PadMode::Edge),
        ("reflect", PadMode::Reflect),
        ("symmetric", PadMode::Symmetric),
        ("wrap", PadMode::Wrap),
    ] {
        let y = pad_center(&x, 18, Axis(0), mode).unwrap();
        assert_eq!(y.to_vec(), f64s(&f[name]), "{name}");
    }
    let m = arr2(&f["matrix"]);
    assert_eq!(
        pad_center(&m, 5, Axis(0), PadMode::Constant(0.0)).unwrap(),
        arr2(&f["matrix_axis0"])
    );
    assert_eq!(
        pad_center(&m, 6, Axis(1), PadMode::Constant(0.0)).unwrap(),
        arr2(&f["matrix_axis1"])
    );
}

#[test]
fn frames_and_slices() {
    let f = &fixture()["fix_frames"];
    let frames = i64s(&f["frames"]);
    assert_eq!(
        fix_frames(&frames, Some(0), None, true).unwrap(),
        i64s(&f["default"])
    );
    assert_eq!(
        fix_frames(&frames, Some(5), Some(25), true).unwrap(),
        i64s(&f["bounded"])
    );
    assert_eq!(
        fix_frames(&frames, Some(5), Some(25), false).unwrap(),
        i64s(&f["no_pad"])
    );
    assert_eq!(
        fix_frames(&frames, None, None, true).unwrap(),
        i64s(&f["none"])
    );
    let s = index_to_slice(&frames, Some(2), Some(30), None, true).unwrap();
    let want = f["slices"].as_array().unwrap();
    assert_eq!(s.len(), want.len());
    for (a, b) in s.iter().zip(want) {
        assert_eq!(vec![a.start, a.stop], i64s(b));
    }
}

#[test]
fn sorting() {
    let f = &fixture()["axis_sort"];
    let s = arr2(&f["s"]);
    let (c, ci) = axis_sort(s.view(), Axis(1), AxisSortKey::Argmax).unwrap();
    assert_eq!(c, arr2(&f["columns"]));
    assert_eq!(ci, usizes(&f["columns_idx"]));
    let (r, ri) = axis_sort(s.view(), Axis(0), AxisSortKey::Argmax).unwrap();
    assert_eq!(r, arr2(&f["rows"]));
    assert_eq!(ri, usizes(&f["rows_idx"]));
}

#[test]
fn shears() {
    let f = &fixture()["shear"];
    let x = arr2(&f["x"]);
    assert_eq!(shear(x.view(), 1, Axis(1)).unwrap(), arr2(&f["columns"]));
    assert_eq!(shear(x.view(), -2, Axis(0)).unwrap(), arr2(&f["rows"]));
}

#[test]
fn sparse_rows() {
    let f = &fixture()["sparsify_rows"];
    let x = arr2(&f["x"]);
    assert_eq!(
        sparsify_rows(x.view(), 0.01).unwrap().to_dense(),
        arr2(&f["q01"])
    );
    assert_eq!(
        sparsify_rows(x.view(), 0.25).unwrap().to_dense(),
        arr2(&f["q25"])
    );
    let row = x.row(0).insert_axis(Axis(0));
    assert_eq!(
        sparsify_rows(row, 0.1).unwrap().to_dense(),
        arr2(&f["one_d"])
    );
}

#[test]
fn gradients() {
    let f = &fixture()["cyclic_gradient"];
    let x = arr2(&f["x"]);
    let g = cyclic_gradient(&x, Axis(1)).unwrap();
    close(
        g.as_slice().unwrap(),
        arr2(&f["axis1"]).as_slice().unwrap(),
        RTOL,
        "axis1",
    );
    let g = cyclic_gradient(&x, Axis(0)).unwrap();
    close(
        g.as_slice().unwrap(),
        arr2(&f["axis0_order2"]).as_slice().unwrap(),
        RTOL,
        "axis0",
    );
}

#[test]
fn off_diagonal() {
    let f = &fixture()["fill_off_diagonal"];
    for (key, want) in f.as_object().unwrap() {
        let want = arr2(want);
        let radius: f64 = key.split('_').nth(1).unwrap().parse().unwrap();
        let mut m = Array2::from_elem(want.dim(), 1.0);
        fill_off_diagonal(m.view_mut(), radius, -1.0).unwrap();
        assert_eq!(m, want, "{key}");
    }
}

#[test]
fn uniques() {
    let f = &fixture()["unique"];
    let x = arr2(&f["x"]).mapv(|v| v as i64);
    assert_eq!(
        count_unique(&x, Axis(1)).unwrap().to_vec(),
        i64s(&f["count_axis1"])
    );
    assert_eq!(
        count_unique(&x, Axis(0)).unwrap().to_vec(),
        i64s(&f["count_axis0"])
    );
    let to_u = |a: ndarray::Array1<bool>| a.iter().map(|&b| usize::from(b)).collect::<Vec<_>>();
    assert_eq!(
        to_u(is_unique(&x, Axis(1)).unwrap()),
        usizes(&f["is_axis1"])
    );
    assert_eq!(
        to_u(is_unique(&x, Axis(0)).unwrap()),
        usizes(&f["is_axis0"])
    );
}

#[test]
fn interpolation() {
    let f = &fixture()["interp"];
    let (x1, x2) = (arr2(&f["x1"]).into_dyn(), arr2(&f["x2"]).into_dyn());
    let (p1, p2) = (f64s(&f["x1_pos"]), f64s(&f["x2_pos"]));
    let q: ArrayD<f64> = ndarray::Array1::from(f64s(&f["q"])).into_dyn();
    for (name, kind) in [
        ("linear", InterpKind::Linear),
        ("nearest", InterpKind::Nearest),
        ("nearest-up", InterpKind::NearestUp),
        ("zero", InterpKind::Zero),
        ("slinear", InterpKind::SLinear),
        ("quadratic", InterpKind::Quadratic),
        ("cubic", InterpKind::Cubic),
        ("previous", InterpKind::Previous),
        ("next", InterpKind::Next),
    ] {
        let (y1, y2) = interp_broadcast(
            x1.view(),
            &p1,
            x2.view(),
            &p2,
            Some(q.view()),
            kind,
            -7.0,
            -1,
        )
        .unwrap();
        let w = &f["kinds"][name];
        let (w1, w2) = (arr2(&w["y1"]), arr2(&w["y2"]));
        close(
            y1.as_slice().unwrap(),
            w1.as_slice().unwrap(),
            SPLINE_RTOL,
            name,
        );
        close(
            y2.as_slice().unwrap(),
            w2.as_slice().unwrap(),
            SPLINE_RTOL,
            name,
        );
    }
}

#[test]
fn decoding() {
    let f = &fixture()["buf_to_float"];
    let raw: Vec<u8> = f64s(&f["bytes"]).iter().map(|&v| v as u8).collect();
    for n in [1_usize, 2, 4, 8] {
        let y: Vec<f64> = buf_to_float(&raw, n).unwrap();
        assert_eq!(y, f64s(&f[format!("n{n}")]), "{n}");
    }
    let y: Vec<f32> = buf_to_float(&raw, 2).unwrap();
    let want: Vec<f32> = f64s(&f["n2_f32"]).iter().map(|&v| v as f32).collect();
    assert_eq!(y, want);
}

#[test]
fn blocks() {
    let f = &fixture()["blocks"];
    let b = i64s(&f["blocks"]);
    let (bl, hop) = (
        NonZeroUsize::new(16).unwrap(),
        NonZeroUsize::new(512).unwrap(),
    );
    let frames: Vec<i64> = b.iter().map(|&v| blocks_to_frames(v, bl)).collect();
    assert_eq!(frames, i64s(&f["frames"]));
    let samples: Vec<i64> = b.iter().map(|&v| blocks_to_samples(v, bl, hop)).collect();
    assert_eq!(samples, i64s(&f["samples"]));
    let time: Vec<f64> = b
        .iter()
        .map(|&v| blocks_to_time(v, bl, hop, 22050.0))
        .collect();
    assert_eq!(time, f64s(&f["time"]));
}

#[test]
fn zero_crossings_match() {
    let f = &fixture()["zero_crossings"];
    let y = ndarray::Array1::from(f64s(&f["y"]));
    let check = |z: ndarray::Array1<bool>, key: &str| {
        let got: Vec<f64> = z.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect();
        assert_eq!(got, f64s(&f[key]), "{key}");
    };
    check(
        zero_crossings(&y, 1e-10, true, true, Axis(0)).unwrap(),
        "default",
    );
    check(
        zero_crossings(&y, 1e-10, false, true, Axis(0)).unwrap(),
        "no_pad",
    );
    check(
        zero_crossings(&y, 0.5, true, false, Axis(0)).unwrap(),
        "zero_sign",
    );
    let peak = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    check(
        zero_crossings(&y, 0.1 * peak, true, true, Axis(0)).unwrap(),
        "ref",
    );
    let negative = y.mapv(|v| -v);
    let both = ndarray::stack(Axis(0), &[y.view(), negative.view()]).unwrap();
    let z = zero_crossings(&both, 1e-10, true, true, Axis(0)).unwrap();
    assert_eq!(bools(&z), arr2(&f["axis0"]));
}

#[test]
fn generators_match() {
    let f = &fixture()["generators"];
    let start = -std::f64::consts::FRAC_PI_2;
    close(
        &tone::<f64>(441.0, 22050.0, 512, start),
        &f64s(&f["tone"]),
        1e-12,
        "tone",
    );
    close(
        &tone::<f64>(441.0, 22050.0, 512, 0.7),
        &f64s(&f["tone_phi"]),
        1e-12,
        "tone phi",
    );
    let sweep = chirp::<f64>(110.0, 3520.0, 22050.0, 2048, false, start).unwrap();
    close(&sweep, &f64s(&f["chirp_log"]), 1e-12, "chirp");
    let sweep = chirp::<f64>(110.0, 3520.0, 22050.0, 2048, true, 0.3).unwrap();
    close(&sweep, &f64s(&f["chirp_lin"]), 1e-12, "linear chirp");
}

#[test]
fn note_names_match() {
    use audio_samples::utils::notation::midi_name;
    let f = &fixture()["note_names"];
    let text = |v: &Value| -> Vec<String> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|e| e.as_str().unwrap().to_owned())
            .collect()
    };
    let midi = f64s(&f["midi"]);
    let c: Vec<String> = midi
        .iter()
        .map(|&m| midi_name(m, true, false, "C:maj", false).unwrap())
        .collect();
    assert_eq!(c, text(&f["c"]));
    let plain: Vec<String> = [60.0, 61.0, 70.0]
        .iter()
        .map(|&m| midi_name(m, false, false, "F:maj", false).unwrap())
        .collect();
    assert_eq!(plain, text(&f["f_major_no_octave"]));
    let unicode: Vec<String> = [61.0, 70.0]
        .iter()
        .map(|&m| midi_name(m, true, false, "Db:maj", true).unwrap())
        .collect();
    assert_eq!(unicode, text(&f["unicode"]));
    let cents: Vec<String> = [69.25, 69.0, 68.75]
        .iter()
        .map(|&m| midi_name(m, true, true, "C:maj", false).unwrap())
        .collect();
    assert_eq!(cents, text(&f["cents"]));
    let notes = text(&f["fractional_notes"]);
    let fractional: Vec<f64> = notes
        .iter()
        .map(|n| audio_samples::utils::audio_math::note_to_midi_fractional(n).unwrap())
        .collect();
    close(
        &fractional,
        &f64s(&f["fractional"]),
        1e-14,
        "fractional midi",
    );
}
