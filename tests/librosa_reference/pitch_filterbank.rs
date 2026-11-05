//! Golden-reference checks for `operations::pitch_filterbank`.
//!
//! Designs come from the same closed forms as the reference, section by section, so they
//! agree to 1e-9 of the peak coefficient; the narrow bands' polynomial form loses digits
//! and agrees to 1e-6 of the peak. The top rate of `iirt` needs no resampling: the band
//! energies agree to 2e-5 of the peak over the input (the padding of forward-backward
//! filtering meets the same samples). The lower rates pass through a different
//! resampler, for which the bound is 5e-3 of the peak.

use std::num::NonZeroUsize;

use audio_samples::operations::pitch_filterbank::{
    BandFilter, FilterLayout, IirtConfig, MultirateFilterConfig, iirt, mr_frequencies,
    semitone_filterbank,
};
use audio_samples::operations::types::IirFilterType;
use audio_samples::utils::array_ops::PadMode;
use ndarray::{Array2, Axis};
use serde_json::Value;

use crate::common::{arr2, f64s, load, signal};

fn fx() -> Value {
    load("pitch_filterbank")
}

fn close(a: &[f64], b: &[f64], rtol: f64, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    let peak = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let err = a
        .iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    assert!(err <= rtol * peak, "{what}: {err:e} > {rtol:e} of {peak:e}");
}

fn sos(f: &BandFilter) -> Vec<f64> {
    let BandFilter::Sos(s) = f else {
        unreachable!("asked for sections")
    };
    s.iter().flatten().copied().collect()
}

#[test]
fn frequencies() {
    let f = fx();
    for (tuning, key) in [(0.0, "mr_0.0"), (0.25, "mr_0.25")] {
        let (freqs, rates) = mr_frequencies(tuning);
        close(&freqs, &f64s(&f[key]["freqs"]), 1e-13, key);
        assert_eq!(rates.to_vec(), f64s(&f[key]["rates"]));
    }
}

#[test]
fn designs() {
    let f = fx();
    let bank = &f["bank"];
    let (freqs, rates) = mr_frequencies(0.0);
    let base = MultirateFilterConfig::new();
    for (key, cfg) in [
        ("ellip", base),
        (
            "butter",
            MultirateFilterConfig {
                filter_type: IirFilterType::Butterworth,
                ..base
            },
        ),
        (
            "cheby1",
            MultirateFilterConfig {
                filter_type: IirFilterType::ChebyshevI,
                ..base
            },
        ),
        (
            "cheby2",
            MultirateFilterConfig {
                filter_type: IirFilterType::ChebyshevII,
                ..base
            },
        ),
        (
            "q10",
            MultirateFilterConfig {
                q: 10.0,
                passband_ripple: 0.5,
                stopband_attenuation: 30.0,
                ..base
            },
        ),
    ] {
        let ours = semitone_filterbank(&freqs, &rates, &cfg, FilterLayout::Sos).unwrap();
        for (band, want) in bank[key].as_object().unwrap() {
            let want = arr2(want);
            let got = sos(&ours[band.parse::<usize>().unwrap()]);
            close(
                &got,
                want.as_slice().unwrap(),
                1e-9,
                &format!("{key} {band}"),
            );
        }
    }
    let ours = semitone_filterbank(&freqs, &rates, &base, FilterLayout::Ba).unwrap();
    for (band, want) in bank["ba"].as_object().unwrap() {
        let BandFilter::Ba { b, a } = &ours[band.parse::<usize>().unwrap()] else {
            unreachable!("asked for polynomials")
        };
        close(b, &f64s(&want["b"]), 1e-6, band);
        close(a, &f64s(&want["a"]), 1e-6, band);
    }
    let c = &bank["custom"];
    let ours = semitone_filterbank(
        &f64s(&c["freqs"]),
        &f64s(&c["rates"]),
        &base,
        FilterLayout::Sos,
    )
    .unwrap();
    for (o, w) in ours.iter().zip(c["sos"].as_array().unwrap()) {
        close(&sos(o), arr2(w).as_slice().unwrap(), 1e-9, "custom");
    }
}

fn case_config(kw: &Value) -> IirtConfig {
    let mut cfg = IirtConfig::new();
    if let Some(t) = kw["tuning"].as_f64() {
        cfg = cfg.with_tuning(t);
    }
    if let Some(w) = kw["win_length"].as_u64() {
        cfg = cfg.with_win_length(NonZeroUsize::new(w as usize).unwrap());
    }
    if let Some(h) = kw["hop_length"].as_u64() {
        cfg = cfg.with_hop_length(NonZeroUsize::new(h as usize));
    }
    if kw["center"].as_bool() == Some(false) {
        cfg = cfg.with_center(false);
    }
    if kw["pad_mode"].as_str() == Some("reflect") {
        cfg = cfg.with_pad_mode(PadMode::Reflect);
    }
    if kw["flayout"].as_str() == Some("ba") {
        cfg = cfg.with_layout(FilterLayout::Ba);
    }
    cfg
}

#[test]
fn iirt_top_rate_matches() {
    let f = fx();
    let y = Array2::from_shape_vec((1, signal("mix").len()), signal("mix")).unwrap();
    for (name, c) in f["iirt"].as_object().unwrap() {
        let p = iirt(y.view(), 22050, &case_config(&c["kw"])).unwrap();
        let p = p.index_axis(Axis(0), 0);
        let want = arr2(&c["power"]);
        let ours = p.slice(ndarray::s![70.., ..20]).to_owned();
        assert_eq!(ours.dim(), want.dim(), "{name}");
        let rtol = if name == "ba" { 1e-3 } else { 2e-5 };
        close(
            ours.as_standard_layout().as_slice().unwrap(),
            want.as_slice().unwrap(),
            rtol,
            name,
        );
    }
}

#[test]
fn iirt_all_rates_within_resampler_bound() {
    let f = fx();
    let y = Array2::from_shape_vec((1, signal("mix").len()), signal("mix")).unwrap();
    let c = &f["iirt"]["default"];
    let p = iirt(y.view(), 22050, &IirtConfig::new()).unwrap();
    let ours = p
        .index_axis(Axis(0), 0)
        .slice(ndarray::s![.., ..;8])
        .to_owned();
    let want = arr2(&c["all"]);
    close(
        ours.as_standard_layout().as_slice().unwrap(),
        want.as_slice().unwrap(),
        5e-3,
        "all",
    );
}

#[test]
fn iirt_single_precision() {
    let v: Vec<f32> = signal("mix").iter().map(|&x| x as f32).collect();
    let y = Array2::from_shape_vec((1, v.len()), v).unwrap();
    let p = iirt(y.view(), 22050, &IirtConfig::new()).unwrap();
    assert_eq!(p.dim().1, 85);
    assert!(p.iter().all(|x| x.is_finite() && *x >= 0.0));
}
