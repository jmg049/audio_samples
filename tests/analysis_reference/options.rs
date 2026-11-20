//! Golden-reference checks for wavelet bandwidths and tuned variable-Q bins.
//!
//! The wavelet lengths and filters are the same arithmetic as the reference, so they agree
//! to round-off. The variable-Q responses resample with a different filter, whose passband
//! ripple sets the difference: 1e-5 of the peak over all octaves.

use audio_samples::operations::constant_q::{
    CqtConfig, VqtConfig, WaveletConfig, vqt_with_intervals, wavelet_lengths_with_alpha,
    wavelet_with_alpha,
};
use audio_samples::{AudioSamples, sample_rate};
use ndarray::{Array1, Array2};
use num_complex::Complex;
use serde_json::Value;
use std::num::NonZeroUsize;

use crate::common::{arr2, f64s, load};

fn complex2(v: &Value) -> Array2<Complex<f64>> {
    let (re, im) = (arr2(&v["re"]), arr2(&v["im"]));
    ndarray::Zip::from(&re)
        .and(&im)
        .map_collect(|&r, &i| Complex::new(r, i))
}

fn peak_error(
    a: ndarray::ArrayView2<'_, Complex<f64>>,
    b: ndarray::ArrayView2<'_, Complex<f64>>,
) -> f64 {
    let peak = b.iter().fold(0.0_f64, |m, v| m.max(v.norm()));
    a.iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).norm()))
        / peak
}

#[test]
fn wavelet_bandwidths() {
    let f = &load("options")["wavelet_alpha"];
    let (freqs, alpha) = (f64s(&f["freqs"]), f64s(&f["alpha"]));
    let cfg = WaveletConfig::default();
    let (lengths, cutoff) = wavelet_lengths_with_alpha(&freqs, 22050.0, &cfg, &alpha).unwrap();
    for (a, b) in lengths.iter().zip(f64s(&f["lengths"])) {
        assert!((a - b).abs() <= 1e-12 * b, "{a} vs {b}");
    }
    assert!((cutoff - f["cutoff"].as_f64().unwrap()).abs() < 1e-9);
    let (filters, _) = wavelet_with_alpha(&freqs, 22050.0, &cfg, &alpha).unwrap();
    assert!(peak_error(filters.view(), complex2(&f["filters"]).view()) < 1e-9);
}

#[test]
fn tuned_variable_q() {
    let all = load("options");
    let y = Array1::from(f64s(&all["y"]));
    let f = &all["vqt_intervals"];
    let audio = AudioSamples::new_mono(y, sample_rate!(22050)).unwrap();
    let cqt = CqtConfig::new()
        .with_fmin(Some(110.0))
        .with_n_bins(NonZeroUsize::new(48))
        .with_tuning(Some(0.0));
    let cfg = VqtConfig::new().with_cqt(cqt).with_gamma(Some(0.0));
    let v = vqt_with_intervals(&audio, &cfg, &f64s(&f["ratios"])).unwrap();
    let got = v.index_axis(ndarray::Axis(0), 0);
    let expected = complex2(&f["v"]);
    assert_eq!(got.dim(), expected.dim());
    let whole = peak_error(got.view(), expected.view());
    assert!(whole < 1e-5, "all octaves {whole}");
}
