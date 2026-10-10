//! librosa pitch tracking and harmonics: yin, pyin, salience, interp_harmonics,
//! f0_harmonics.
//!
//! The harmonic functions take `(channels, bins, frames)` spectrograms
//! (librosa receives the 2-D `(bins, frames)` of a mono signal).

use audio_samples::AudioFloat;
use audio_samples::operations::pitch_tracking::{
    AudioPitchTracking, FrequencyGrid, PyinConfig, SalienceConfig, YinConfig, f0_harmonics,
    interp_harmonics, salience,
};
use ndarray::Array1;

use crate::{AudioCtx, ConstCtx, SeqCtx, Thunk, VecCtx, thunk};

/// `note_to_hz('C2')`, `note_to_hz('C7')`: yin / pyin `fmin`, `fmax`.
const FMIN: f64 = 65.406_391_325_149_66;
const FMAX: f64 = 2_093.004_522_404_789;
/// librosa requires `harmonics`; the Python half uses the same.
const HARMONICS: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0];

/// FFT bin centre frequencies for librosa's default `n_fft = 2048`.
fn fft_freqs<F: AudioFloat>(x: &AudioCtx<F>) -> Array1<F> {
    let n_fft = 2048;
    Array1::from_shape_fn(n_fft / 2 + 1, |k| {
        F::of(k as f64 * f64::from(x.sr) / n_fft as f64)
    })
}

pub fn audio<'a, F: AudioFloat>(case: &str, x: &'a AudioCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "yin" => {
            let c = YinConfig::librosa(FMIN, FMAX);
            thunk(move || x.audio.yin(&c).unwrap())
        }
        "pyin" => {
            let c = PyinConfig::librosa(FMIN, FMAX);
            thunk(move || x.audio.pyin(&c).unwrap())
        }
        "salience" => {
            let s = x.s_mag();
            let freqs = fft_freqs(x);
            let c = SalienceConfig::librosa();
            thunk(move || {
                salience(s.view(), FrequencyGrid::Fixed(freqs.view()), &HARMONICS, &c).unwrap()
            })
        }
        "interp_harmonics" => {
            let s = x.s_mag();
            let freqs = fft_freqs(x);
            thunk(move || {
                interp_harmonics(
                    s.view(),
                    FrequencyGrid::Fixed(freqs.view()),
                    &HARMONICS,
                    0.0,
                )
                .unwrap()
            })
        }
        "f0_harmonics" => {
            let s = x.s_mag();
            let freqs = fft_freqs(x);
            let f0 = x.audio.yin(&YinConfig::librosa(FMIN, FMAX)).unwrap();
            thunk(move || {
                f0_harmonics(
                    s.view(),
                    f0.view(),
                    FrequencyGrid::Fixed(freqs.view()),
                    &HARMONICS,
                    0.0,
                )
                .unwrap()
            })
        }
        _ => None,
    }
}

pub fn constant<'a, F: AudioFloat>(_case: &str, _x: &'a ConstCtx<F>) -> Option<Thunk<'a>> {
    None
}

pub fn vec<'a, F: AudioFloat>(_case: &str, _x: &'a VecCtx<F>) -> Option<Thunk<'a>> {
    None
}

pub fn seq<'a, F: AudioFloat>(_case: &str, _x: &'a SeqCtx<F>) -> Option<Thunk<'a>> {
    None
}
