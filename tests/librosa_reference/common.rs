//! Shared helpers for the librosa reference tests: fixture loading, array
//! parsing and tolerance assertions with informative failure messages.

use std::path::PathBuf;
use std::sync::OnceLock;

use audio_samples::AudioSamples;
use ndarray::{Array1, Array2, Array3};
use num_complex::Complex;
use serde_json::Value;

/// Load `tests/fixtures/librosa/<name>.json`.
pub fn load(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/librosa")
        .join(format!("{name}.json"));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", path.display()));
    serde_json::from_str(&text).expect("fixture is valid JSON")
}

fn signals() -> &'static Value {
    static SIGNALS: OnceLock<Value> = OnceLock::new();
    SIGNALS.get_or_init(|| load("signals"))
}

/// Sample rate shared by all fixture signals.
pub fn sr() -> u32 {
    signals()["sr"].as_u64().expect("sr") as u32
}

/// A shared test signal as f64 samples (exact: stored as 16-bit PCM codes).
pub fn signal(name: &str) -> Vec<f64> {
    let s = signals();
    let scale = s["pcm_scale"].as_f64().expect("pcm_scale");
    s["signals"][name]
        .as_array()
        .unwrap_or_else(|| panic!("unknown signal {name}"))
        .iter()
        .map(|k| k.as_f64().expect("pcm code") / scale)
        .collect()
}

/// A shared test signal as mono f64 audio at the fixture sample rate.
pub fn mono(name: &str) -> AudioSamples<'static, f64> {
    let rate = std::num::NonZeroU32::new(sr()).expect("sr > 0");
    AudioSamples::new_mono(Array1::from(signal(name)), rate).expect("valid mono audio")
}

/// The stereo test signal (`stereo_l` / `stereo_r`) as f64 audio.
pub fn stereo() -> AudioSamples<'static, f64> {
    let l = signal("stereo_l");
    let r = signal("stereo_r");
    let n = l.len();
    let mut data = Array2::zeros((2, n));
    for i in 0..n {
        data[[0, i]] = l[i];
        data[[1, i]] = r[i];
    }
    let rate = std::num::NonZeroU32::new(sr()).expect("sr > 0");
    AudioSamples::new_multi_channel(data, rate).expect("valid stereo audio")
}

pub fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("expected array, got {v}"))
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

pub fn arr1(v: &Value) -> Array1<f64> {
    Array1::from(f64s(v))
}

pub fn arr2(v: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = v.as_array().expect("2-D array").iter().map(f64s).collect();
    let (r, c) = (rows.len(), rows.first().map_or(0, Vec::len));
    Array2::from_shape_vec((r, c), rows.into_iter().flatten().collect()).expect("rectangular")
}

pub fn arr3(v: &Value) -> Array3<f64> {
    let planes: Vec<Array2<f64>> = v.as_array().expect("3-D array").iter().map(arr2).collect();
    let (p, r, c) = (planes.len(), planes[0].nrows(), planes[0].ncols());
    let flat: Vec<f64> = planes.iter().flat_map(|a| a.iter().copied()).collect();
    Array3::from_shape_vec((p, r, c), flat).expect("rectangular")
}

/// A complex 2-D array stored as `{"re": [[..]], "im": [[..]]}`.
pub fn carr2(v: &Value) -> Array2<Complex<f64>> {
    let re = arr2(&v["re"]);
    let im = arr2(&v["im"]);
    ndarray::Zip::from(&re)
        .and(&im)
        .map_collect(|&a, &b| Complex::new(a, b))
}

/// A complex 3-D array stored as `{"re": [[[..]]], "im": [[[..]]]}`.
pub fn carr3(v: &Value) -> Array3<Complex<f64>> {
    let re = arr3(&v["re"]);
    let im = arr3(&v["im"]);
    ndarray::Zip::from(&re)
        .and(&im)
        .map_collect(|&a, &b| Complex::new(a, b))
}

/// Maximum absolute error and maximum error relative to `max|expected|`.
pub fn errors<'a>(
    actual: impl IntoIterator<Item = &'a f64>,
    expected: impl IntoIterator<Item = &'a f64>,
) -> (f64, f64) {
    let a: Vec<f64> = actual.into_iter().copied().collect();
    let e: Vec<f64> = expected.into_iter().copied().collect();
    assert_eq!(a.len(), e.len(), "length mismatch");
    let scale = e
        .iter()
        .fold(0.0f64, |m, x| m.max(x.abs()))
        .max(f64::MIN_POSITIVE);
    let abs = a
        .iter()
        .zip(&e)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
    (abs, abs / scale)
}

/// Assert `actual` ≈ `expected`: every element satisfies
/// `|a - e| <= atol + rtol * max|expected|` (a peak-relative bound, the right
/// measure for spectra whose small bins are dominated by round-off).
#[track_caller]
pub fn assert_close<'a>(
    what: &str,
    actual: impl IntoIterator<Item = &'a f64>,
    expected: impl IntoIterator<Item = &'a f64>,
    rtol: f64,
    atol: f64,
) {
    let a: Vec<f64> = actual.into_iter().copied().collect();
    let e: Vec<f64> = expected.into_iter().copied().collect();
    assert_eq!(
        a.len(),
        e.len(),
        "{what}: length {} != expected {}",
        a.len(),
        e.len()
    );
    let scale = e.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let bound = atol + rtol * scale;
    let (mut worst, mut at) = (0.0f64, 0usize);
    for (i, (x, y)) in a.iter().zip(&e).enumerate() {
        assert!(x.is_finite(), "{what}: non-finite value {x} at {i}");
        let d = (x - y).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    assert!(
        worst <= bound,
        "{what}: max abs error {worst:.3e} at flat index {at} (actual {}, expected {}) exceeds \
         bound {bound:.3e} (rtol {rtol:e} x peak {scale:.3e} + atol {atol:e})",
        a[at],
        e[at]
    );
}

/// Assert two shapes are equal.
#[track_caller]
pub fn assert_shape(what: &str, actual: &[usize], expected: &[usize]) {
    assert_eq!(actual, expected, "{what}: shape mismatch");
}
