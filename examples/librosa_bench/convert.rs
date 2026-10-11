//! librosa.convert: unit conversions and frequency tables.
//!
//! The conversions are scalar `f64` functions (the frame/sample/time helpers
//! also have `_slice` forms); array cases map them over the `vec` values. The
//! note-name and key cases loop over the same inputs as the Python half.

use std::num::NonZeroUsize;

use audio_samples::operations::rhythm::{fourier_tempo_frequencies, tempo_frequencies};
use audio_samples::utils::audio_math::{
    MelScale, a4_to_tuning_slice, cqt_frequencies, fft_frequencies, frames_to_samples_slice,
    frames_to_time_with_offset_slice, frequency_to_note_static, hz_to_mel_with_scale,
    hz_to_midi_slice, hz_to_octs_slice, key_to_degrees, key_to_notes_static,
    mel_frequencies_with_scale, mel_to_hz_with_scale, midi_to_hz, midi_to_note,
    midi_to_note_static, note_to_frequency, note_to_midi, octs_to_hz, samples_like,
    samples_to_frames_slice, samples_to_time, time_to_frames_floor_slice, time_to_samples_slice,
    times_like, tuning_to_a4,
};
use audio_samples::{AudioFloat, nzu};

use crate::ctx::VEC_N;
use crate::{AudioCtx, ConstCtx, SeqCtx, Thunk, VecCtx, thunk};

const SR: f64 = 22050.0;
const HOP: NonZeroUsize = nzu!(512);
const BPO: NonZeroUsize = nzu!(12);
/// Inputs of the string conversions.
const N_STR: usize = 1000;
/// `note_to_hz('C1')`.
const C1: f64 = 32.703_195_662_574_83;
const PITCHES: [&str; 12] = [
    "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
];

/// `np.geomspace(lo, hi, n)`.
fn geomspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (lo.ln() + i as f64 / (n - 1) as f64 * (hi.ln() - lo.ln())).exp())
        .collect()
}

fn midis() -> Vec<u8> {
    (0..N_STR).map(|i| (i % 128) as u8).collect()
}

/// The 24 major and minor keys, repeated to `N_STR` entries.
fn keys() -> Vec<String> {
    let all: Vec<String> = ["maj", "min"]
        .iter()
        .flat_map(|m| PITCHES.iter().map(move |p| format!("{p}:{m}")))
        .collect();
    (0..N_STR).map(|i| all[i % all.len()].clone()).collect()
}

fn f64s<F: AudioFloat>(x: &VecCtx<F>, lo: f64, hi: f64) -> Vec<f64> {
    x.values(lo, hi).iter().map(|v| v.as_f64()).collect()
}

pub fn audio<'a, F: AudioFloat>(_case: &str, _x: &'a AudioCtx<F>) -> Option<Thunk<'a>> {
    None
}

pub fn constant<'a, F: AudioFloat>(case: &str, _x: &'a ConstCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "fft_frequencies" => thunk(|| fft_frequencies(2048, SR)),
        "mel_frequencies" => {
            thunk(|| mel_frequencies_with_scale(128, 0.0, 11025.0, MelScale::Slaney))
        }
        "cqt_frequencies" => thunk(|| cqt_frequencies(84, C1, BPO, 0.0)),
        "tempo_frequencies" => thunk(|| tempo_frequencies(384, HOP, SR)),
        "fourier_tempo_frequencies" => thunk(|| fourier_tempo_frequencies(SR, HOP, nzu!(384))),
        "hz_to_note" => {
            let hz = geomspace(30.0, 4000.0, N_STR);
            // Note names as &'static str (frequency_to_note_static): no
            // allocation per name, as librosa's array of fixed-width strings.
            thunk(move || {
                hz.iter()
                    .map(|&f| frequency_to_note_static(f).unwrap())
                    .collect::<Vec<_>>()
            })
        }
        "midi_to_note" => {
            let m = midis();
            // Note names as &'static str (midi_to_note_static).
            thunk(move || {
                m.iter()
                    .map(|&v| midi_to_note_static(v).unwrap())
                    .collect::<Vec<_>>()
            })
        }
        "note_to_hz" | "note_to_midi" => {
            let notes: Vec<String> = midis().iter().map(|&v| midi_to_note(v).unwrap()).collect();
            if case == "note_to_hz" {
                thunk(move || {
                    notes
                        .iter()
                        .map(|n| note_to_frequency(n).unwrap())
                        .collect::<Vec<_>>()
                })
            } else {
                thunk(move || {
                    notes
                        .iter()
                        .map(|n| note_to_midi(n).unwrap())
                        .collect::<Vec<_>>()
                })
            }
        }
        "key_to_notes" => {
            let k = keys();
            // Cached tables (key_to_notes_static), as librosa's lru-cached
            // key_to_notes: the loop holds references, not new strings.
            thunk(move || {
                k.iter()
                    .map(|k| key_to_notes_static(k, true, false).unwrap())
                    .collect::<Vec<_>>()
            })
        }
        "key_to_degrees" => {
            let k = keys();
            thunk(move || {
                k.iter()
                    .map(|k| key_to_degrees(k).unwrap())
                    .collect::<Vec<_>>()
            })
        }
        _ => None,
    }
}

pub fn vec<'a, F: AudioFloat>(case: &str, x: &'a VecCtx<F>) -> Option<Thunk<'a>> {
    let frames = || (0..VEC_N as i64).collect::<Vec<i64>>();
    let samples = || (0..VEC_N as i64).map(|i| i * 13).collect::<Vec<i64>>();
    let map = |v: Vec<f64>, f: fn(f64) -> f64| {
        thunk(move || v.iter().map(|&a| f(a)).collect::<Vec<f64>>())
    };
    match case {
        "frames_to_samples" => {
            let fr = frames();
            thunk(move || frames_to_samples_slice(&fr, HOP, None))
        }
        "frames_to_time" => {
            let fr = frames();
            thunk(move || frames_to_time_with_offset_slice(&fr, SR, HOP, None))
        }
        "samples_to_frames" => {
            let s = samples();
            thunk(move || samples_to_frames_slice(&s, HOP, None))
        }
        "samples_to_time" => {
            let s: Vec<usize> = samples().iter().map(|&v| v as usize).collect();
            thunk(move || {
                s.iter()
                    .map(|&v| samples_to_time(v, SR))
                    .collect::<Vec<f64>>()
            })
        }
        "time_to_frames" => {
            let t = f64s(x, 0.0, 60.0);
            thunk(move || time_to_frames_floor_slice(&t, SR, HOP, None))
        }
        "time_to_samples" => {
            let t = f64s(x, 0.0, 60.0);
            thunk(move || time_to_samples_slice(&t, SR))
        }
        "times_like" => thunk(|| times_like(VEC_N, SR, HOP, None)),
        "samples_like" => thunk(|| samples_like(VEC_N, HOP, None)),
        "hz_to_mel" => map(f64s(x, 20.0, 11025.0), |v| {
            hz_to_mel_with_scale(v, MelScale::Slaney)
        }),
        "mel_to_hz" => map(f64s(x, 0.0, 45.0), |v| {
            mel_to_hz_with_scale(v, MelScale::Slaney)
        }),
        "hz_to_midi" => {
            let v = f64s(x, 20.0, 11025.0);
            thunk(move || hz_to_midi_slice(&v))
        }
        "midi_to_hz" => map(f64s(x, 0.0, 127.0), midi_to_hz),
        "hz_to_octs" => {
            let v = f64s(x, 20.0, 11025.0);
            thunk(move || hz_to_octs_slice(&v, 0.0, BPO))
        }
        "octs_to_hz" => map(f64s(x, 0.0, 9.0), |v| octs_to_hz(v, 0.0, BPO)),
        "A4_to_tuning" => {
            let v = f64s(x, 430.0, 450.0);
            thunk(move || a4_to_tuning_slice(&v, BPO))
        }
        "tuning_to_A4" => map(f64s(x, -0.5, 0.5), |v| tuning_to_a4(v, BPO)),
        _ => None,
    }
}

pub fn seq<'a, F: AudioFloat>(_case: &str, _x: &'a SeqCtx<F>) -> Option<Thunk<'a>> {
    None
}
