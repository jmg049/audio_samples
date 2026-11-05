//! Native half of the `options` benchmark group. Rows that use array forms have no native
//! half: they are NumPy and existing kernels.

use audio_samples::AudioFloat;
use audio_samples::operations::constant_q::{
    CqtConfig, VqtConfig, WaveletConfig, vqt_with_intervals, wavelet_lengths_with_alpha,
    wavelet_with_alpha,
};
use audio_samples::utils::array_ops::{chirp, tone, zero_crossings};
use audio_samples::utils::audio_math::note_to_midi_fractional;
use audio_samples::utils::notation::{midi_name, plimit_intervals};
use ndarray::Axis;
use std::num::NonZeroUsize;

use crate::{AudioCtx, ConstCtx, SeqCtx, Thunk, VecCtx, thunk};

const SR: f64 = 22050.0;

fn freqs() -> Vec<f64> {
    (0..24)
        .map(|i| 110.0 * 2f64.powf(i as f64 / 12.0))
        .collect()
}

fn alpha() -> Vec<f64> {
    (0..24).map(|i| 0.05 + 0.002 * i as f64).collect()
}

pub fn audio<'a, F: AudioFloat>(case: &str, x: &'a AudioCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "vqt_intervals" => {
            let ratios = plimit_intervals(&[3, 5], 12, true).unwrap();
            let cqt = CqtConfig::new()
                .with_fmin(Some(110.0))
                .with_n_bins(NonZeroUsize::new(48))
                .with_tuning(Some(0.0));
            let c = VqtConfig::new().with_cqt(cqt).with_gamma(Some(0.0));
            thunk(move || vqt_with_intervals(&x.audio, &c, &ratios).unwrap())
        }
        _ => None,
    }
}

pub fn constant<'a, F: AudioFloat>(case: &str, _x: &'a ConstCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "tone_phi" => thunk(|| tone::<f64>(440.0, SR, 220_500, 0.3)),
        "chirp_phi" => thunk(|| chirp::<f64>(110.0, 3520.0, SR, 220_500, false, 0.3).unwrap()),
        "midi_to_note_key" => thunk(|| {
            (0..1000)
                .map(|i| {
                    midi_name(
                        48.0 + 48.0 * f64::from(i) / 1000.0,
                        true,
                        true,
                        "F:maj",
                        false,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>()
        }),
        "hz_to_note_key" => thunk(|| {
            (0..1000)
                .map(|i| {
                    let f = 110.0 * 2f64.powf(f64::from(i) * 3.0 / 1000.0);
                    let midi = 12.0 * (f.log2() - 440f64.log2()) + 69.0;
                    midi_name(midi, false, false, "F:maj", false).unwrap()
                })
                .collect::<Vec<_>>()
        }),
        "note_to_midi_fractional" => thunk(|| {
            ["C4+25", "D4", "E4-10", "G4", "A4", "C5", "F#3", "Bb4"]
                .iter()
                .cycle()
                .take(1000)
                .map(|n| note_to_midi_fractional(n).unwrap())
                .collect::<Vec<_>>()
        }),
        "wavelet_lengths_alpha" => {
            let (f, a) = (freqs(), alpha());
            let c = WaveletConfig::default();
            thunk(move || wavelet_lengths_with_alpha(&f, SR, &c, &a).unwrap())
        }
        "wavelet_alpha" => {
            let (f, a) = (freqs(), alpha());
            let c = WaveletConfig::default();
            thunk(move || wavelet_with_alpha(&f, SR, &c, &a).unwrap())
        }
        _ => None,
    }
}

pub fn vec<'a, F: AudioFloat>(case: &str, x: &'a VecCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "zero_crossings_opt" => {
            let v = x.values(0.0, 20.0).mapv(|e| e.sin());
            let t = F::of(0.1);
            thunk(move || zero_crossings(&v, t, true, false, Axis(0)).unwrap())
        }
        _ => None,
    }
}

pub fn seq<'a, F: AudioFloat>(_case: &str, _x: &'a SeqCtx<F>) -> Option<Thunk<'a>> {
    None
}
