//! Golden-reference tests for the filter implementations.
//!
//! The fixtures in `tests/fixtures/filters/` are produced by
//! `tests/reference/generate_filter_fixtures.py` from long-standing reference
//! implementations - SciPy `scipy.signal`, FFmpeg `af_biquads`, and (when the
//! C/C++ harness is built) DSPFilters and liquid-dsp - and are cross-checked
//! against each other before being written. These tests require audio_samples
//! to reproduce them:
//!
//! - IIR designs: SOS coefficients equal SciPy's (same section pairing and
//!   ordering) and the complex frequency response matches; magnitude responses
//!   also match the independent C/C++ designs where those use the same
//!   parameter conventions.
//! - Filtering: `apply_iir_filter` ≡ `sosfilt`, `filtfilt` ≡ `sosfiltfilt`,
//!   streaming `SosFilter` ≡ `sosfilt`.
//! - Parametric EQ: RBJ cookbook coefficients and `lfilter` output (which FFmpeg
//!   reproduces to < 1e-9).
//! - Simple first-order filters, FIR convolution and short-signal `filtfilt`.
//!
//! Regenerate the fixtures with
//! `uv run --project tests/reference tests/reference/generate_filter_fixtures.py`.

use std::num::NonZeroUsize;
use std::path::PathBuf;

use audio_samples::operations::iir_filtering::SosFilter;
use audio_samples::operations::traits::{AudioIirFiltering, AudioParametricEq, AudioProcessing};
use audio_samples::operations::types::{
    BesselNorm, EqBand, FilterResponse, IirFilterDesign, IirFilterType,
};
use audio_samples::{AudioSamples, sample_rate};
use ndarray::Array1;
use non_empty_slice::NonEmptySlice;
use num_complex::Complex;
use serde_json::Value;

fn load(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/filters")
        .join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", path.display()));
    serde_json::from_str(&text).expect("fixture is valid JSON")
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn sos_rows(v: &Value) -> Vec<[f64; 6]> {
    v.as_array()
        .expect("sos array")
        .iter()
        .map(|row| {
            let r = f64s(row);
            [r[0], r[1], r[2], r[3], r[4], r[5]]
        })
        .collect()
}

fn nz(n: u64) -> NonZeroUsize {
    NonZeroUsize::new(n as usize).expect("order > 0")
}

fn response_of(name: &str) -> FilterResponse {
    match name {
        "lowpass" => FilterResponse::LowPass,
        "highpass" => FilterResponse::HighPass,
        "bandpass" => FilterResponse::BandPass,
        "bandstop" => FilterResponse::BandStop,
        other => panic!("unknown response {other}"),
    }
}

/// Build the audio_samples design equivalent to a fixture case.
fn design_of(case: &Value) -> IirFilterDesign {
    let order = nz(case["order"].as_u64().unwrap());
    let response = response_of(case["response"].as_str().unwrap());
    let wn = f64s(&case["wn"]);
    let rp = case.get("rp").and_then(Value::as_f64);
    let rs = case.get("rs").and_then(Value::as_f64);
    let band = wn.len() == 2;
    let mut d = match case["family"].as_str().unwrap() {
        "butter" => {
            if band {
                IirFilterDesign::butterworth_bandpass(order, wn[0], wn[1])
            } else {
                IirFilterDesign::butterworth_lowpass(order, wn[0])
            }
        }
        "cheby1" => {
            if band {
                IirFilterDesign::chebyshev_i_band(response, order, wn[0], wn[1], rp.unwrap())
            } else {
                IirFilterDesign::chebyshev_i(response, order, wn[0], rp.unwrap())
            }
        }
        "cheby2" => {
            if band {
                IirFilterDesign::chebyshev_ii_band(response, order, wn[0], wn[1], rs.unwrap())
            } else {
                IirFilterDesign::chebyshev_ii(response, order, wn[0], rs.unwrap())
            }
        }
        "ellip" => {
            if band {
                IirFilterDesign::elliptic_band(
                    response,
                    order,
                    wn[0],
                    wn[1],
                    rp.unwrap(),
                    rs.unwrap(),
                )
            } else {
                IirFilterDesign::elliptic(response, order, wn[0], rp.unwrap(), rs.unwrap())
            }
        }
        "bessel" => {
            let norm = match case["norm"].as_str().unwrap() {
                "mag" => BesselNorm::Magnitude,
                "phase" => BesselNorm::Phase,
                "delay" => BesselNorm::Delay,
                other => panic!("unknown norm {other}"),
            };
            if band {
                IirFilterDesign::bessel_band(response, order, wn[0], wn[1])
            } else {
                IirFilterDesign::bessel(response, order, wn[0])
            }
            .with_bessel_norm(norm)
        }
        other => panic!("unknown family {other}"),
    };
    d.response = response;
    d
}

fn sosfreqz(sos: &[[f64; 6]], freqs: &[f64], fs: f64) -> Vec<Complex<f64>> {
    freqs
        .iter()
        .map(|&f| {
            let w = 2.0 * std::f64::consts::PI * f / fs;
            let z1 = Complex::new(w.cos(), -w.sin());
            let z2 = z1 * z1;
            sos.iter().fold(Complex::new(1.0, 0.0), |acc, s| {
                acc * (s[0] + z1 * s[1] + z2 * s[2]) / (s[3] + z1 * s[4] + z2 * s[5])
            })
        })
        .collect()
}

fn db(x: f64) -> f64 {
    20.0 * x.abs().max(1e-300).log10()
}

fn mono_f64(x: &[f64], fs: f64) -> AudioSamples<'static, f64> {
    AudioSamples::new_mono(
        Array1::from_vec(x.to_vec()),
        std::num::NonZeroU32::new(fs as u32).unwrap(),
    )
    .unwrap()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------
// IIR design
// ---------------------------------------------------------------------------

/// Coefficient tolerance relative to the largest coefficient of the row.
///
/// The design pipeline mirrors SciPy's operation order (including the
/// `zpk2sos` pairing), so coefficients agree to a few ULPs.
fn coeff_tol(_case: &Value) -> f64 {
    // Observed worst case is ~4e-12 (order-24 elliptic); allow headroom for
    // libm differences between platforms.
    1e-10
}

#[test]
fn iir_designs_match_scipy() {
    let fixtures = load("iir_designs.json");
    let cases = fixtures["cases"].as_array().unwrap();
    assert!(cases.len() > 500, "fixture unexpectedly small");

    let mut failures = Vec::new();
    let mut worst_coeff = (0.0f64, String::new());
    let mut worst_resp = (0.0f64, String::new());
    let mut n_cross = 0usize;
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let fs = case["fs"].as_f64().unwrap();
        let expected = sos_rows(&case["sos"]);
        let design = design_of(case);
        let got = match design.sos_coefficients(fs) {
            Ok(s) => s,
            Err(e) => {
                failures.push(format!("{id}: design failed: {e}"));
                continue;
            }
        };

        // 1. Coefficients (section count, order and values).
        if got.len() != expected.len() {
            failures.push(format!(
                "{id}: {} sections, SciPy has {}",
                got.len(),
                expected.len()
            ));
            continue;
        }
        let tol = coeff_tol(case);
        for (si, (g, e)) in got.iter().zip(&expected).enumerate() {
            let scale = e.iter().fold(1.0f64, |m, v| m.max(v.abs()));
            for k in 0..6 {
                let rel = (g[k] - e[k]).abs() / scale;
                if rel > worst_coeff.0 {
                    worst_coeff = (rel, format!("{id} s{si} c{k}"));
                }
                if rel > tol {
                    failures.push(format!(
                        "{id}: section {si} coeff {k}: {} vs SciPy {} (rel {rel:.2e} > {tol:.0e})",
                        g[k], e[k]
                    ));
                }
            }
        }

        // 2. Complex frequency response (independent of section ordering).
        let freqs = f64s(&case["freqs"]);
        let h_re = f64s(&case["h_re"]);
        let h_im = f64s(&case["h_im"]);
        let h = sosfreqz(&got, &freqs, fs);
        let peak = h_re
            .iter()
            .zip(&h_im)
            .map(|(r, i)| r.hypot(*i))
            .fold(0.0f64, f64::max);
        for (k, hk) in h.iter().enumerate() {
            let e = Complex::new(h_re[k], h_im[k]);
            let err = (hk - e).norm() / peak.max(1.0);
            if err > worst_resp.0 {
                worst_resp = (err, format!("{id} @ {} Hz", freqs[k]));
            }
            if err > 1e-9 {
                failures.push(format!(
                    "{id}: H({} Hz) = {hk} vs SciPy {e} (err {err:.2e})",
                    freqs[k]
                ));
            }
        }

        // 3. Independent C/C++ implementations, where present.
        if let Some(refs) = case.get("cross_refs").and_then(Value::as_object) {
            for (lib, r) in refs {
                // Tolerances recorded by the generator from each library's
                // precision (DSPFilters: double; liquid-dsp: float32), plus a
                // small margin for audio_samples' own (~1e-12) deviation.
                let tol_db = r["tol_db"].as_f64().unwrap() + 1e-9;
                let floor_db = r["floor_db"].as_f64().unwrap();
                n_cross += 1;
                let mag = f64s(&r["mag_db"]);
                for (k, hk) in h.iter().enumerate() {
                    if mag[k] > floor_db && (db(hk.norm()) - mag[k]).abs() > tol_db {
                        failures.push(format!(
                            "{id}: |H({} Hz)| = {:.9} dB vs {lib} {:.9} dB",
                            freqs[k],
                            db(hk.norm()),
                            mag[k]
                        ));
                    }
                }
            }
        }
    }
    assert!(
        n_cross > 500,
        "expected C/C++ cross references in the fixture"
    );
    println!(
        "iir designs: {} cases ({n_cross} C/C++ cross-reference comparisons); worst coefficient rel err {:.2e} ({}); worst response err {:.2e} ({})",
        cases.len(),
        worst_coeff.0,
        worst_coeff.1,
        worst_resp.0,
        worst_resp.1
    );
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures
            .iter()
            .take(60)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn iir_family_shapes_match_textbook() {
    // Analytic properties that hold regardless of any reference library.
    let fs = 48000.0;
    let butter = IirFilterDesign::butterworth_lowpass(nz(5), 1000.0);
    let (m, _) = butter
        .frequency_response(&[0.0, 1000.0, 2000.0], fs)
        .unwrap();
    assert!((m[0] - 1.0).abs() < 1e-12);
    assert!((db(m[1]) + 3.0103).abs() < 1e-3, "Butterworth -3 dB at fc");

    let hp = IirFilterDesign::butterworth_highpass(nz(4), 100.0);
    let (m, _) = hp.frequency_response(&[100.0, 24000.0], fs).unwrap();
    assert!(
        (db(m[0]) + 3.0103).abs() < 1e-3,
        "HP -3 dB at fc, not mirrored"
    );
    assert!((m[1] - 1.0).abs() < 1e-12, "HP unity at Nyquist");

    for order in 1..=8 {
        let ch = IirFilterDesign::chebyshev_i(FilterResponse::LowPass, nz(order), 1000.0, 1.0);
        let (m, _) = ch.frequency_response(&[0.0, 1000.0], fs).unwrap();
        // Odd order: 0 dB at DC; even order: -rp at DC. Always -rp at the edge.
        let dc = if order % 2 == 1 { 0.0 } else { -1.0 };
        assert!((db(m[0]) - dc).abs() < 1e-9, "cheby1 order {order} DC");
        assert!((db(m[1]) + 1.0).abs() < 1e-9, "cheby1 order {order} edge");
    }

    let c2 = IirFilterDesign::chebyshev_ii(FilterResponse::LowPass, nz(6), 2000.0, 60.0);
    let (m, _) = c2
        .frequency_response(&[2000.0, 3000.0, 10000.0], fs)
        .unwrap();
    assert!(
        (db(m[0]) + 60.0).abs() < 1e-6,
        "cheby2 -rs at stopband edge"
    );
    assert!(m.iter().all(|&v| db(v) <= -60.0 + 1e-6));
}

// ---------------------------------------------------------------------------
// Filtering (time domain)
// ---------------------------------------------------------------------------

#[test]
fn iir_filtering_matches_sosfilt_and_sosfiltfilt() {
    let fixtures = load("iir_time_domain.json");
    let mix_full = f64s(&fixtures["signals"]["mix"]);
    let impulse_full = f64s(&fixtures["signals"]["impulse"]);
    let cases = fixtures["cases"].as_array().unwrap();
    assert!(cases.len() > 50);

    let mut failures = Vec::new();
    let mut worst = [
        (0.0f64, String::new()),
        (0.0, String::new()),
        (0.0, String::new()),
        (0.0, String::new()),
    ];
    let mut note = |slot: usize, err: f64, id: &str| {
        if err > worst[slot].0 {
            worst[slot] = (err, id.to_string());
        }
    };
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let d = &case["design"];
        let fs = d["fs"].as_f64().unwrap();
        let design = design_of(d);
        let want_mix = f64s(&case["sosfilt_mix"]);
        let want_imp = f64s(&case["sosfilt_impulse"]);
        let want_ff = f64s(&case["sosfiltfilt_mix"]);
        let mix = &mix_full[..want_mix.len()];
        let imp = &impulse_full[..want_imp.len()];
        let scale = want_mix.iter().fold(1.0f64, |m, v| m.max(v.abs()));

        // apply_iir_filter == sosfilt
        let mut a = mono_f64(mix, fs);
        a.apply_iir_filter_in_place(&design).unwrap();
        let e = max_abs_diff(a.as_slice().unwrap(), &want_mix) / scale;
        note(0, e, id);
        if e > 2e-10 {
            failures.push(format!("{id}: apply_iir_filter vs sosfilt: {e:.2e}"));
        }
        let mut a = mono_f64(imp, fs);
        a.apply_iir_filter_in_place(&design).unwrap();
        let e = max_abs_diff(a.as_slice().unwrap(), &want_imp);
        if e > 2e-10 {
            failures.push(format!("{id}: impulse response vs sosfilt: {e:.2e}"));
        }

        // filtfilt == sosfiltfilt
        let mut a = mono_f64(mix, fs);
        a.filtfilt_in_place(&design).unwrap();
        let ff_scale = want_ff.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        let e = max_abs_diff(a.as_slice().unwrap(), &want_ff) / ff_scale;
        note(1, e, id);
        if e > 1e-9 {
            failures.push(format!("{id}: filtfilt vs sosfiltfilt: {e:.2e}"));
        }

        // Streaming SosFilter (direct form I sections), fed in uneven blocks.
        let mut sos = SosFilter::from_design(&design, fs).unwrap();
        let mut buf = mix.to_vec();
        let (head, tail) = buf.split_at_mut(37);
        sos.process_block(head);
        sos.process_block(tail);
        let e = max_abs_diff(&buf, &want_mix) / scale;
        note(2, e, id);
        if e > 1e-9 {
            failures.push(format!("{id}: streaming SosFilter vs sosfilt: {e:.2e}"));
        }

        // f32 audio path: same filter, single-precision samples.
        let x32: Vec<f32> = mix.iter().map(|&v| v as f32).collect();
        let mut a32: AudioSamples<'static, f32> = AudioSamples::new_mono(
            Array1::from_vec(x32),
            std::num::NonZeroU32::new(fs as u32).unwrap(),
        )
        .unwrap();
        a32.apply_iir_filter_in_place(&design).unwrap();
        let y32: Vec<f64> = a32
            .as_slice()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let e = max_abs_diff(&y32, &want_mix) / scale;
        note(3, e, id);
        // Input quantisation to f32 (~6e-8) amplified by the filter's noise
        // gain; narrow / near-DC designs have large gain, hence the margin.
        if e > 5e-4 {
            failures.push(format!("{id}: f32 path vs sosfilt: {e:.2e}"));
        }
    }
    println!(
        "time domain: sosfilt {:.2e} ({}); sosfiltfilt {:.2e} ({}); streaming {:.2e} ({}); f32 {:.2e} ({})",
        worst[0].0,
        worst[0].1,
        worst[1].0,
        worst[1].1,
        worst[2].0,
        worst[2].1,
        worst[3].0,
        worst[3].1
    );
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn multichannel_filters_each_channel_independently() {
    let fixtures = load("iir_time_domain.json");
    let mix = f64s(&fixtures["signals"]["mix"]);
    let case = &fixtures["cases"].as_array().unwrap()[0];
    let design = design_of(&case["design"]);
    let fs = case["design"]["fs"].as_f64().unwrap();
    let n = f64s(&case["sosfilt_mix"]).len();
    let want = f64s(&case["sosfilt_mix"]);

    // Channel 0: the reference signal; channel 1: its negation.
    let mut data = ndarray::Array2::<f64>::zeros((2, n));
    for i in 0..n {
        data[[0, i]] = mix[i];
        data[[1, i]] = -mix[i];
    }
    let mut audio =
        AudioSamples::new_multi_channel(data, std::num::NonZeroU32::new(fs as u32).unwrap())
            .unwrap();
    audio.apply_iir_filter_in_place(&design).unwrap();
    let out = audio.as_multi_channel().unwrap();
    for i in 0..n {
        assert!((out[[0, i]] - want[i]).abs() < 1e-11);
        assert!((out[[1, i]] + want[i]).abs() < 1e-11);
    }
}

// ---------------------------------------------------------------------------
// Parametric EQ (RBJ cookbook)
// ---------------------------------------------------------------------------

fn eq_band(case: &Value) -> EqBand {
    let f0 = case["f0"].as_f64().unwrap();
    let q = case["q"].as_f64().unwrap();
    let g = case["gain_db"].as_f64().unwrap();
    match case["kind"].as_str().unwrap() {
        "peak" => EqBand::peak(f0, g, q),
        "lowshelf" => EqBand::low_shelf(f0, g, q),
        "highshelf" => EqBand::high_shelf(f0, g, q),
        "lowpass" => EqBand::low_pass(f0, q),
        "highpass" => EqBand::high_pass(f0, q),
        "bandpass" => EqBand::band_pass(f0, q),
        "notch" => EqBand::band_stop(f0, q),
        other => panic!("unknown kind {other}"),
    }
}

#[test]
fn parametric_eq_matches_rbj_cookbook_and_ffmpeg() {
    let fixtures = load("rbj_biquads.json");
    let mix = f64s(&fixtures["signals"]["mix"]);
    let cases = fixtures["cases"].as_array().unwrap();
    assert!(cases.len() >= 512);

    let mut failures = Vec::new();
    let mut worst_resp = 0.0f64;
    let mut worst_td = 0.0f64;
    let mut checked_td = 0;
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let fs = case["fs"].as_f64().unwrap();
        let band = eq_band(case);
        assert!(
            case["agreeing_references"]
                .as_array()
                .unwrap()
                .iter()
                .any(|r| r == "ffmpeg"),
            "{id}: fixture lacks FFmpeg confirmation"
        );

        // Frequency response vs the cookbook coefficients.
        let b = f64s(&case["b"]);
        let a = f64s(&case["a"]);
        let freqs: Vec<f64> = (0..64).map(|k| k as f64 * fs / 2.0 / 63.0).collect();
        let want = sosfreqz(&[[b[0], b[1], b[2], a[0], a[1], a[2]]], &freqs, fs);
        let audio = mono_f64(&[0.0; 8], fs);
        let mut eq = audio_samples::operations::types::ParametricEq::new();
        eq.add_band(band);
        let (mag, _) = audio.eq_frequency_response(&eq, &freqs).unwrap();
        for (k, w) in want.iter().enumerate() {
            let err = (mag[k] - w.norm()).abs() / w.norm().max(1.0);
            worst_resp = worst_resp.max(err);
            // Low-frequency RBJ designs have near-cancelling coefficient sums
            // (|1 + a1 + a2| ~ 1e-5), so last-bit coefficient differences
            // appear amplified at DC.
            if err > 1e-9 {
                failures.push(format!(
                    "{id}: |H({} Hz)| {} vs {}",
                    freqs[k],
                    mag[k],
                    w.norm()
                ));
            }
        }

        // Time domain vs scipy.lfilter (== FFmpeg to < 1e-9).
        if let Some(want_y) = case.get("lfilter_mix") {
            let want_y = f64s(want_y);
            let mut a = mono_f64(&mix[..want_y.len()], fs);
            a.apply_eq_band_in_place(&band).unwrap();
            let scale = want_y.iter().fold(1.0f64, |m, v| m.max(v.abs()));
            let e = max_abs_diff(a.as_slice().unwrap(), &want_y) / scale;
            worst_td = worst_td.max(e);
            checked_td += 1;
            if e > 1e-11 {
                failures.push(format!("{id}: apply_eq_band vs lfilter: {e:.2e}"));
            }
        }
    }
    println!(
        "rbj: {} cases; worst |H| err {worst_resp:.2e}; worst time-domain err {worst_td:.2e} over {checked_td} cases",
        cases.len()
    );
    assert!(checked_td > 100);
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn parametric_eq_cascade_matches_band_by_band() {
    let fixtures = load("rbj_biquads.json");
    let mix = f64s(&fixtures["signals"]["mix"]);
    let fs = 48000.0;
    let mut eq = audio_samples::operations::types::ParametricEq::new();
    eq.add_band(EqBand::low_shelf(
        100.0,
        4.0,
        std::f64::consts::FRAC_1_SQRT_2,
    ));
    eq.add_band(EqBand::peak(1000.0, -6.0, 2.0));
    eq.add_band(EqBand::high_shelf(
        8000.0,
        3.0,
        std::f64::consts::FRAC_1_SQRT_2,
    ));
    eq.set_output_gain(-2.0);

    let mut cascade = mono_f64(&mix, fs);
    cascade.apply_parametric_eq_in_place(&eq).unwrap();

    let mut sequential = mono_f64(&mix, fs);
    for band in &eq.bands {
        sequential.apply_eq_band_in_place(band).unwrap();
    }
    let g = 10f64.powf(-2.0 / 20.0);
    let seq: Vec<f64> = sequential
        .as_slice()
        .unwrap()
        .iter()
        .map(|v| v * g)
        .collect();
    assert!(max_abs_diff(cascade.as_slice().unwrap(), &seq) < 1e-12);
}

// ---------------------------------------------------------------------------
// Simple filters, FIR, short filtfilt
// ---------------------------------------------------------------------------

#[test]
fn simple_filters_match_first_order_butterworth() {
    let fixtures = load("misc_filters.json");
    let mix = f64s(&fixtures["signals"]["mix"]);
    for case in fixtures["simple_filters"].as_array().unwrap() {
        let fs = case["fs"].as_f64().unwrap();
        let want = f64s(&case["y"]);
        let mut a = mono_f64(&mix, fs);
        let kind = case["kind"].as_str().unwrap();
        match kind {
            "low_pass" => a
                .low_pass_filter_in_place(case["cutoff"].as_f64().unwrap())
                .unwrap(),
            "high_pass" => a
                .high_pass_filter_in_place(case["cutoff"].as_f64().unwrap())
                .unwrap(),
            "band_pass" => a
                .band_pass_filter_in_place(
                    case["low"].as_f64().unwrap(),
                    case["high"].as_f64().unwrap(),
                )
                .unwrap(),
            other => panic!("{other}"),
        }
        let e = max_abs_diff(a.as_slice().unwrap(), &want);
        assert!(e < 1e-12, "{kind} {case:?}: {e:.2e}");
    }
}

#[test]
fn simple_filters_work_on_integer_samples() {
    // Regression: the high-pass used to compute in the sample type, so for
    // i16 audio its coefficient truncated to 0 and the output was silence.
    let fixtures = load("misc_filters.json");
    let mix = f64s(&fixtures["signals"]["mix"]);
    let case = fixtures["simple_filters"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["kind"] == "high_pass" && c["fs"] == 44100.0 && c["cutoff"] == 1000.0)
        .unwrap();
    let want = f64s(&case["y"]);
    let x: Vec<i16> = mix.iter().map(|&v| (v * 32767.0).round() as i16).collect();
    let mut a: AudioSamples<'static, i16> =
        AudioSamples::new_mono(Array1::from_vec(x), sample_rate!(44100)).unwrap();
    a.high_pass_filter_in_place(1000.0).unwrap();
    let y: Vec<f64> = a
        .as_slice()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v) / 32767.0)
        .collect();
    // Quantisation of input and output to 16 bits: a few LSB.
    let e = max_abs_diff(&y, &want);
    assert!(e < 4.0 / 32767.0, "i16 high-pass error {e}");
}

#[test]
fn fir_apply_filter_is_numpy_convolve_valid() {
    let fixtures = load("misc_filters.json");
    let mix = f64s(&fixtures["signals"]["mix"]);
    for case in fixtures["fir"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let h = f64s(&case["h"]);
        let want = f64s(&case["valid"]);
        let a = mono_f64(&mix, 48000.0);
        let y = a.apply_filter(NonEmptySlice::new(&h).unwrap()).unwrap();
        assert_eq!(y.samples_per_channel().get(), want.len(), "{name}");
        let e = max_abs_diff(y.as_slice().unwrap(), &want);
        assert!(e < 1e-12, "{name}: {e:.2e}");
    }
}

#[test]
fn filtfilt_short_signals_match_sosfiltfilt_with_clamped_padlen() {
    let fixtures = load("misc_filters.json");
    let mix = f64s(&fixtures["signals"]["mix"]);
    let design = IirFilterDesign::butterworth_lowpass(nz(4), 1000.0);
    for case in fixtures["filtfilt_short"].as_array().unwrap() {
        let n = case["n"].as_u64().unwrap() as usize;
        let want = f64s(&case["y"]);
        let mut a = mono_f64(&mix[..n], 44100.0);
        a.filtfilt_in_place(&design).unwrap();
        let e = max_abs_diff(a.as_slice().unwrap(), &want);
        assert!(e < 1e-10, "n={n}: {e:.2e}");
    }
}

#[test]
fn sos_coefficients_round_trip_through_scipy_layout() {
    let design = IirFilterDesign::elliptic(FilterResponse::BandPass, nz(1), 1.0, 1.0, 40.0);
    // A band design given through the single-edge constructor is rejected.
    assert!(design.sos_coefficients(48000.0).is_err());

    let design =
        IirFilterDesign::chebyshev_ii_band(FilterResponse::BandStop, nz(5), 900.0, 1100.0, 50.0);
    let rows = design.sos_coefficients(48000.0).unwrap();
    let sos = SosFilter::from_sos_coefficients(&rows).unwrap();
    assert_eq!(sos.sos_coefficients().unwrap(), rows);
    assert_eq!(design.filter_type, IirFilterType::ChebyshevII);
}

// ---------------------------------------------------------------------------
// Integer sample paths
// ---------------------------------------------------------------------------

#[test]
fn i16_audio_through_unity_filter_is_bit_identical() {
    // A 0 dB peaking band has identical numerator and denominator, so it must
    // be transparent. Regression: float -> i16 used to truncate with a
    // symmetric scale, shifting every negative sample by one LSB.
    let x: Vec<i16> = (0..4096)
        .map(|i| ((i as f64 * 0.731).sin() * 30000.0) as i16)
        .chain([i16::MIN, i16::MAX, -1, 0, 1])
        .collect();
    let mut a: AudioSamples<'static, i16> =
        AudioSamples::new_mono(Array1::from_vec(x.clone()), sample_rate!(48000)).unwrap();
    a.apply_eq_band_in_place(&EqBand::peak(1000.0, 0.0, 1.0))
        .unwrap();
    assert_eq!(a.as_slice().unwrap(), x.as_slice());
}

#[test]
fn i16_filtering_matches_f64_reference_without_bias() {
    let fixtures = load("iir_time_domain.json");
    let mix = f64s(&fixtures["signals"]["mix"]);
    let design = IirFilterDesign::butterworth_highpass(nz(4), 100.0);
    let fs = 44100.0;
    let n = mix.len();
    let x: Vec<i16> = mix
        .iter()
        .map(|&v| (v * 0.9 * 32767.0).round() as i16)
        .collect();
    // f64 reference on the dequantised input (the f64 path is checked against
    // scipy elsewhere), then requantised by the documented rule.
    let xf: Vec<f64> = x
        .iter()
        .map(|&v| {
            if v < 0 {
                f64::from(v) / 32768.0
            } else {
                f64::from(v) / 32767.0
            }
        })
        .collect();
    let mut r = mono_f64(&xf, fs);
    r.apply_iir_filter_in_place(&design).unwrap();
    let want: Vec<i16> = r
        .as_slice()
        .unwrap()
        .iter()
        .map(|&v| {
            let v = v.clamp(-1.0, 1.0);
            (if v < 0.0 { v * 32768.0 } else { v * 32767.0 }).round() as i16
        })
        .collect();
    let mut a: AudioSamples<'static, i16> =
        AudioSamples::new_mono(Array1::from_vec(x), sample_rate!(44100)).unwrap();
    a.apply_iir_filter_in_place(&design).unwrap();
    let got = a.as_slice().unwrap();
    let mut max_diff = 0i32;
    let mut sum_diff = 0i64;
    for i in 0..n {
        let d = i32::from(got[i]) - i32::from(want[i]);
        max_diff = max_diff.max(d.abs());
        sum_diff += i64::from(d);
    }
    assert!(max_diff <= 1, "max LSB difference {max_diff}");
    assert_eq!(sum_diff, 0, "requantisation must be unbiased");
}

#[test]
fn non_biquad_sections_are_rejected_by_sos_export() {
    use audio_samples::operations::iir_filtering::IirFilter;
    let sos = SosFilter::new(vec![IirFilter::new(vec![0.25; 4], vec![1.0])]);
    assert!(sos.sos_coefficients().is_err());
    let mut x = vec![1.0; 64];
    assert!(sos.filtfilt_in_place(&mut x).is_err());
    assert!(x.iter().all(|&v| v == 1.0), "signal untouched on error");
}

#[test]
fn degenerate_designs_error_instead_of_panicking_or_nan() {
    // Vanishing ripple makes the Chebyshev prototype poles non-finite.
    for order in 1..=4 {
        let d = IirFilterDesign::chebyshev_i(FilterResponse::LowPass, nz(order), 1000.0, 1e-17);
        assert!(d.sos_coefficients(48000.0).is_err());
        let mut a = mono_f64(&[0.5; 32], 48000.0);
        assert!(a.apply_iir_filter_in_place(&d).is_err());
    }
    // Edges within 1e-5 of Nyquist used to overflow the gain computation and
    // silently produce zero / NaN output; SciPy designs these fine.
    let d = IirFilterDesign::butterworth_lowpass(nz(24), 23999.99);
    let (m, _) = d.frequency_response(&[0.0, 1000.0], 48000.0).unwrap();
    assert!((m[0] - 1.0).abs() < 1e-9 && (m[1] - 1.0).abs() < 1e-9);
    let d = IirFilterDesign::butterworth_bandstop(nz(17), 23990.0, 23999.9);
    let (m, _) = d.frequency_response(&[0.0, 1000.0], 48000.0).unwrap();
    assert!((m[0] - 1.0).abs() < 1e-9 && (m[1] - 1.0).abs() < 1e-9);
}
