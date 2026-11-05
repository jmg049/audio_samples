//! Golden-reference checks for `utils::notation`. Names and factorisations must match
//! exactly. Ratios are powers of 2 of the same logarithms, so they agree to 1e-14.

use audio_samples::utils::notation::*;
use serde_json::Value;

use crate::common::{f64s, load};

fn fx() -> Value {
    load("notation")
}

fn close(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}");
    for (x, y) in a.iter().zip(b) {
        assert!(
            (x - y).abs() <= 1e-14 * y.abs().max(1.0),
            "{what}: {x} vs {y}"
        );
    }
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_owned())
        .collect()
}

fn factors(v: &Value) -> Vec<Vec<(u32, i32)>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|d| {
            let mut f: Vec<(u32, i32)> = d
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, e)| (k.parse().unwrap(), e.as_i64().unwrap() as i32))
                .collect();
            f.sort_unstable();
            f
        })
        .collect()
}

fn sorted(mut f: Vec<Vec<(u32, i32)>>) -> Vec<Vec<(u32, i32)>> {
    for v in &mut f {
        v.sort_unstable();
    }
    f
}

#[test]
fn tuning_systems() {
    let f = fx();
    let t = &f["tuning"];
    for b in [7_usize, 12, 24, 53] {
        for sort in [true, false] {
            let name = if sort { "True" } else { "False" };
            close(
                &pythagorean_intervals(b, sort),
                &f64s(&t[format!("pyth_{b}_{name}")]),
                "pythagorean",
            );
            assert_eq!(
                sorted(pythagorean_factors(b, sort)),
                factors(&t[format!("pyth_factors_{b}_{name}")]),
                "{b} {sort}"
            );
        }
    }
    let cases: [(&[u32], usize); 6] = [
        (&[3], 12),
        (&[3, 5], 12),
        (&[3, 5], 24),
        (&[3, 5, 7], 16),
        (&[3, 11], 9),
        (&[5], 5),
    ];
    for (primes, b) in cases {
        for sort in [true, false] {
            let flag = if sort { "True" } else { "False" };
            let key = primes
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join("_");
            let key = format!("{key}_{b}_{flag}");
            close(
                &plimit_intervals(primes, b, sort).unwrap(),
                &f64s(&t[format!("plimit_{key}")]),
                &key,
            );
            assert_eq!(
                sorted(plimit_factors(primes, b, sort).unwrap()),
                factors(&t[format!("plimit_factors_{key}")]),
                "{key}"
            );
        }
    }
}

#[test]
fn frequencies() {
    let f = fx();
    let c = &f["frequencies"];
    for (name, system) in [
        ("equal", TuningSystem::Equal),
        ("pythagorean", TuningSystem::Pythagorean),
        ("ji3", TuningSystem::Ji3),
        ("ji5", TuningSystem::Ji5),
        ("ji7", TuningSystem::Ji7),
    ] {
        for sort in [true, false] {
            let flag = if sort { "True" } else { "False" };
            let v = interval_frequencies(30, 55.0, &system, 12, 0.0, sort).unwrap();
            close(&v, &f64s(&c[format!("{name}_{flag}")]), name);
        }
    }
    let v = interval_frequencies(30, 55.0, &TuningSystem::Equal, 12, 0.25, true).unwrap();
    close(&v, &f64s(&c["equal_tuned"]), "tuned");
    let v = interval_frequencies(
        7,
        100.0,
        &TuningSystem::Custom(vec![1.0, 1.25, 1.5]),
        12,
        0.0,
        true,
    )
    .unwrap();
    close(&v, &f64s(&c["custom"]), "custom");
}

#[test]
fn fifths() {
    let f = fx();
    for c in f["fifths"].as_array().unwrap() {
        let got = fifths_to_note(
            c["unison"].as_str().unwrap(),
            c["fifths"].as_i64().unwrap(),
            c["unicode"].as_bool().unwrap(),
        )
        .unwrap();
        assert_eq!(got, c["out"].as_str().unwrap(), "{c}");
    }
}

#[test]
fn every_table_ratio_spells_the_same() {
    let f = fx();
    let c = &f["fjs_table"];
    let ratios = f64s(&c["intervals"]);
    let (uni, ascii) = (strings(&c["unicode"]), strings(&c["ascii"]));
    for (i, &r) in ratios.iter().enumerate() {
        assert_eq!(
            interval_to_fjs(r, "C", 65.0 / 63.0, true).unwrap(),
            uni[i],
            "{r}"
        );
        assert_eq!(
            interval_to_fjs(r, "F#", 65.0 / 63.0, false).unwrap(),
            ascii[i],
            "{r}"
        );
    }
    let c = &f["fjs_octaves"];
    for (r, want) in f64s(&c["intervals"]).iter().zip(strings(&c["out"])) {
        assert_eq!(
            interval_to_fjs(*r, "C", 65.0 / 63.0, true).unwrap(),
            want,
            "{r}"
        );
    }
    let c = &f["hz_to_fjs"];
    let hz = f64s(&c["hz"]);
    let d = strings(&c["default"]);
    let e = strings(&c["fmin"]);
    for (i, &h) in hz.iter().enumerate() {
        assert_eq!(hz_to_fjs(h, 220.0, None, false).unwrap(), d[i]);
        assert_eq!(hz_to_fjs(h, 110.0, Some("D"), true).unwrap(), e[i]);
    }
}

#[test]
fn thaat_and_mela() {
    let f = fx();
    assert_eq!(list_thaat().to_vec(), strings(&f["list_thaat"]));
    for (name, want) in f["thaat_degrees"].as_object().unwrap() {
        let got: Vec<f64> = thaat_to_degrees(name)
            .unwrap()
            .iter()
            .map(|&d| f64::from(d))
            .collect();
        assert_eq!(got, f64s(want), "{name}");
    }
    for (i, name) in list_mela().iter().enumerate() {
        assert_eq!(
            f["list_mela"][*name].as_u64().unwrap() as usize,
            i + 1,
            "{name}"
        );
        assert_eq!(usize::from(mela_number(name).unwrap()), i + 1);
    }
    for (i, want) in f["mela_degrees"].as_array().unwrap().iter().enumerate() {
        let got: Vec<f64> = mela_to_degrees(i as u8 + 1)
            .unwrap()
            .iter()
            .map(|&d| f64::from(d))
            .collect();
        assert_eq!(got, f64s(want), "{i}");
    }
    for (key, table) in f["mela_svara"].as_object().unwrap() {
        let abbr = key.starts_with("True");
        let unicode = key.ends_with("True");
        for (i, want) in table.as_array().unwrap().iter().enumerate() {
            assert_eq!(
                mela_to_svara(i as u8 + 1, abbr, unicode).unwrap(),
                strings(want),
                "{key} {i}"
            );
        }
    }
}

#[test]
fn svara() {
    let f = fx();
    let c = &f["svara"];
    let midi: Vec<f64> = c["midi"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap_or(f64::NAN))
        .collect();
    for abbr in [true, false] {
        for octave in [true, false] {
            for unicode in [true, false] {
                let t = |b: bool| if b { "True" } else { "False" };
                let key = format!("{}_{}_{}", t(abbr), t(octave), t(unicode));
                let h = strings(&c[format!("h_{key}")]);
                let k = strings(&c[format!("c_{key}")]);
                for (i, &m) in midi.iter().enumerate() {
                    assert_eq!(
                        midi_to_svara_h(m, 60.0, abbr, octave, unicode),
                        h[i],
                        "{key} {m}"
                    );
                    assert_eq!(
                        midi_to_svara_c(m, 60.0, 31, abbr, octave, unicode).unwrap(),
                        k[i],
                        "{key} {m}"
                    );
                }
            }
        }
    }
    let hz = f64s(&c["hz"]);
    let (hh, hc) = (strings(&c["hz_h"]), strings(&c["hz_c"]));
    let kosalam = mela_number("kosalam").unwrap();
    for (i, &v) in hz.iter().enumerate() {
        assert_eq!(hz_to_svara_h(v, 261.63, true, true, true), hh[i]);
        assert_eq!(
            hz_to_svara_c(v, 261.63, kosalam, true, true, true).unwrap(),
            hc[i]
        );
    }
    let notes = strings(&c["notes"]);
    let (nh, nc) = (strings(&c["note_h"]), strings(&c["note_c"]));
    for (i, n) in notes.iter().enumerate() {
        assert_eq!(
            note_to_svara_h(n, "C4", true, true, true).unwrap(),
            nh[i],
            "{n}"
        );
        assert_eq!(
            note_to_svara_c(n, "C4", 1, true, true, true).unwrap(),
            nc[i],
            "{n}"
        );
    }
}
