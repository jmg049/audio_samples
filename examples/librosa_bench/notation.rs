//! Native half of the `notation` benchmark group. Display rows have no native half.

use audio_samples::AudioFloat;
use audio_samples::utils::notation as n;

use crate::{AudioCtx, ConstCtx, SeqCtx, Thunk, VecCtx, thunk};

const N_NAMES: usize = 1000;
const RATIOS: [f64; 8] = [
    1.0,
    1.25,
    1.5,
    1.75,
    1.125,
    5.0 / 3.0,
    4.0 / 3.0,
    16.0 / 15.0,
];
const FJS_TOLERANCE: f64 = 65.0 / 63.0;

fn midi() -> Vec<f64> {
    (0..N_NAMES)
        .map(|i| 48.0 + 48.0 * i as f64 / N_NAMES as f64)
        .collect()
}

fn hz() -> Vec<f64> {
    (0..N_NAMES)
        .map(|i| 110.0 * 2f64.powf(i as f64 * 3.0 / N_NAMES as f64))
        .collect()
}

fn notes() -> Vec<&'static str> {
    ["C4", "D4", "E4", "G4", "A4", "C5", "F#3", "Bb4"]
        .iter()
        .copied()
        .cycle()
        .take(N_NAMES)
        .collect()
}

pub fn audio<'a, F: AudioFloat>(_case: &str, _x: &'a AudioCtx<F>) -> Option<Thunk<'a>> {
    None
}

pub fn constant<'a, F: AudioFloat>(case: &str, _x: &'a ConstCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "pythagorean_intervals" => thunk(|| n::pythagorean_intervals(53, true)),
        "plimit_intervals" => thunk(|| n::plimit_intervals(&[3, 5], 24, true).unwrap()),
        "interval_frequencies" => thunk(|| {
            n::interval_frequencies(84, 32.7, &n::TuningSystem::Ji5, 12, 0.0, true).unwrap()
        }),
        "fifths_to_note" => thunk(|| n::fifths_to_note("C", 7, true).unwrap()),
        "interval_to_fjs" => thunk(|| {
            RATIOS
                .iter()
                .map(|&r| n::interval_to_fjs(r, "C", FJS_TOLERANCE, true).unwrap())
                .collect::<Vec<_>>()
        }),
        "interval_to_fjs_tiny" => {
            thunk(|| n::interval_to_fjs(1.5, "C", FJS_TOLERANCE, true).unwrap())
        }
        "hz_to_fjs" => thunk(|| {
            RATIOS
                .iter()
                .map(|&r| n::hz_to_fjs(220.0 * r, 220.0, Some("A"), false).unwrap())
                .collect::<Vec<_>>()
        }),
        "list_mela" => thunk(n::list_mela),
        "list_thaat" => thunk(n::list_thaat),
        "thaat_to_degrees" => thunk(|| n::thaat_to_degrees("kafi").unwrap()),
        "mela_to_degrees" => thunk(|| n::mela_to_degrees(29).unwrap()),
        "mela_to_svara" => thunk(|| n::mela_to_svara(29, true, true).unwrap()),
        "midi_to_svara_h" => {
            let m = midi();
            thunk(move || {
                m.iter()
                    .map(|&v| n::midi_to_svara_h(v, 60.0, true, true, true))
                    .collect::<Vec<_>>()
            })
        }
        "midi_to_svara_h_tiny" => thunk(|| n::midi_to_svara_h(67.0, 60.0, true, true, true)),
        "midi_to_svara_c" => {
            let m = midi();
            thunk(move || {
                m.iter()
                    .map(|&v| n::midi_to_svara_c(v, 60.0, 29, true, true, true).unwrap())
                    .collect::<Vec<_>>()
            })
        }
        "hz_to_svara_h" => {
            let f = hz();
            thunk(move || {
                f.iter()
                    .map(|&v| n::hz_to_svara_h(v, 261.63, true, true, true))
                    .collect::<Vec<_>>()
            })
        }
        "hz_to_svara_c" => {
            let f = hz();
            thunk(move || {
                f.iter()
                    .map(|&v| n::hz_to_svara_c(v, 261.63, 29, true, true, true).unwrap())
                    .collect::<Vec<_>>()
            })
        }
        "note_to_svara_h" => {
            let t = notes();
            thunk(move || {
                t.iter()
                    .map(|v| n::note_to_svara_h(v, "C4", true, true, true).unwrap())
                    .collect::<Vec<_>>()
            })
        }
        "note_to_svara_c" => {
            let t = notes();
            thunk(move || {
                t.iter()
                    .map(|v| n::note_to_svara_c(v, "C4", 29, true, true, true).unwrap())
                    .collect::<Vec<_>>()
            })
        }
        _ => None,
    }
}

pub fn vec<'a, F: AudioFloat>(_case: &str, _x: &'a VecCtx<F>) -> Option<Thunk<'a>> {
    None
}

pub fn seq<'a, F: AudioFloat>(_case: &str, _x: &'a SeqCtx<F>) -> Option<Thunk<'a>> {
    None
}
