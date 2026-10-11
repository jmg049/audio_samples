//! librosa core spectral representations (STFT, magphase, reassigned
//! spectrogram, phase vocoder, Griffin-Lim, PCEN, dB conversions, frequency
//! weightings), the mel spectrogram and the `librosa.filters` window and mel
//! functions.
//!
//! Spectrogram inputs are `(channels, bins, frames)`, as the free functions
//! take them (librosa receives the 2-D `(bins, frames)` of a mono signal).

use audio_samples::operations::effects::{
    GriffinLimConfig, GriffinLimInit, griffinlim, phase_vocoder,
};
use audio_samples::operations::features::{
    FrequencyWeighting, PcenConfig, ReassignedSpectrogramConfig, a_weighting, b_weighting,
    c_weighting, d_weighting, frequency_weighting, multi_frequency_weighting, pcen,
    perceptual_weighting, z_weighting,
};
use audio_samples::operations::filterbanks::{
    MelFilterbankConfig, get_window, mel_filterbank, window_bandwidth, window_sumsquare_as,
};
use audio_samples::operations::spectrum::{
    DbConfig, MelSpectrogramConfig, StftConfig, amplitude_to_db, db_to_amplitude, db_to_power,
    melspectrogram_from_spectrogram, power_to_db,
};
use audio_samples::{
    AudioFeatures, AudioFloat, AudioSamples, AudioSpectrum, AudioTransforms, Window, nzu,
};
use ndarray::{Array1, Array3, Axis};
use num_complex::Complex;
use std::num::NonZeroUsize;

use crate::{AudioCtx, ConstCtx, SeqCtx, Thunk, VecCtx, thunk};

/// Phase-vocoder stretch rate (the Python half uses the same).
const PV_RATE: f64 = 1.5;
/// Frames of a 10 s signal at 22050 Hz with librosa's default STFT.
const WSS_FRAMES: usize = 431;

/// `melspectrogram(y)` with librosa's defaults.
pub fn mel<F: AudioFloat>(x: &AudioCtx<F>) -> Array3<F> {
    x.audio
        .melspectrogram(&MelSpectrogramConfig::librosa())
        .unwrap()
}

/// `power_to_db(melspectrogram(y))` with librosa's defaults.
pub fn mel_db<F: AudioFloat>(x: &AudioCtx<F>) -> Array3<F> {
    power_to_db(&mel(x), &DbConfig::librosa_power()).unwrap()
}

/// FFT bin centre frequencies for `n_fft` at the context's rate.
pub fn fft_freqs<F: AudioFloat>(x: &AudioCtx<F>, n_fft: usize) -> Vec<f64> {
    (0..=n_fft / 2)
        .map(|k| k as f64 * f64::from(x.sr) / n_fft as f64)
        .collect()
}

/// `values` as `f64` (the frequency weightings take `&[f64]`).
pub fn to_f64<F: AudioFloat>(values: &Array1<F>) -> Vec<f64> {
    values.iter().map(|v| v.as_f64()).collect()
}

pub fn audio<'a, F: AudioFloat>(case: &str, x: &'a AudioCtx<F>) -> Option<Thunk<'a>> {
    let sr = f64::from(x.sr);
    match case {
        "stft" => {
            let c = StftConfig::librosa();
            thunk(move || x.audio.stft_matrix(&c).unwrap())
        }
        "istft" => {
            let c = StftConfig::librosa();
            let d = x.stft();
            let n = NonZeroUsize::new(x.n());
            let rate = x.rate();
            thunk(move || AudioSamples::<F>::istft_matrix(d, &c, rate, n).unwrap())
        }
        "magphase" => {
            // The API takes a 2-D complex128 matrix: channel 0, widened to f64.
            let d = x
                .stft()
                .index_axis(Axis(0), 0)
                .mapv(|c| Complex::new(c.re.as_f64(), c.im.as_f64()));
            thunk(move || AudioSamples::<f64>::magphase(&d, None))
        }
        "reassigned_spectrogram" => {
            let c = ReassignedSpectrogramConfig::librosa();
            thunk(move || x.audio.reassigned_spectrogram(&c).unwrap())
        }
        "phase_vocoder" => {
            let d = x.stft();
            thunk(move || phase_vocoder(d.view(), PV_RATE).unwrap())
        }
        "griffinlim" => {
            let s = x.s_mag();
            let c = GriffinLimConfig::librosa().with_init(GriffinLimInit::ZeroPhase);
            thunk(move || griffinlim(s.view(), &c).unwrap())
        }
        "pcen" => {
            let s = mel(x).mapv(|v| v * F::of(2f64.powi(31)));
            let c = PcenConfig::librosa();
            thunk(move || pcen(s.view(), sr, &c).unwrap())
        }
        "melspectrogram" => {
            let c = MelSpectrogramConfig::librosa();
            thunk(move || x.audio.melspectrogram(&c).unwrap())
        }
        "melspectrogram_S" => {
            let s = x.s_pow();
            let c = MelFilterbankConfig::librosa();
            thunk(move || melspectrogram_from_spectrogram(s.view(), sr, nzu!(2048), &c).unwrap())
        }
        "power_to_db" => {
            let m = mel(x);
            let c = DbConfig::librosa_power();
            thunk(move || power_to_db(&m, &c).unwrap())
        }
        "amplitude_to_db" => {
            let s = x.s_mag();
            let c = DbConfig::librosa_amplitude();
            thunk(move || amplitude_to_db(&s, &c).unwrap())
        }
        "db_to_power" => {
            let s = mel_db(x);
            thunk(move || db_to_power(&s, 1.0))
        }
        "db_to_amplitude" => {
            let s = amplitude_to_db(&x.s_mag(), &DbConfig::librosa_amplitude()).unwrap();
            thunk(move || db_to_amplitude(&s, 1.0))
        }
        "perceptual_weighting" => {
            let s = x.s_pow();
            let f = fft_freqs(x, 2048);
            let c = DbConfig::librosa_power();
            thunk(move || perceptual_weighting(s.view(), &f, FrequencyWeighting::A, &c).unwrap())
        }
        _ => None,
    }
}

pub fn constant<'a, F: AudioFloat>(case: &str, _x: &'a ConstCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "get_window" => thunk(|| get_window(&Window::Hann, 2048, true).unwrap()),
        "filters_mel" => {
            let c = MelFilterbankConfig::librosa();
            thunk(move || mel_filterbank(22050.0, nzu!(2048), &c).unwrap())
        }
        "window_bandwidth" => thunk(|| window_bandwidth(&Window::Hann, nzu!(1000)).unwrap()),
        "window_sumsquare" => thunk(|| {
            // librosa's default dtype is float32
            window_sumsquare_as::<f32>(&Window::Hann, WSS_FRAMES, nzu!(512), None, nzu!(2048))
                .unwrap()
        }),
        _ => None,
    }
}

pub fn vec<'a, F: AudioFloat>(case: &str, x: &'a VecCtx<F>) -> Option<Thunk<'a>> {
    let weighting: fn(&[f64], Option<f64>) -> Vec<f64> = match case {
        "A_weighting" => a_weighting,
        "B_weighting" => b_weighting,
        "C_weighting" => c_weighting,
        "D_weighting" => d_weighting,
        "Z_weighting" => z_weighting,
        "frequency_weighting" => {
            let f = to_f64(&x.values(10.0, 20000.0));
            return thunk(move || frequency_weighting(&f, FrequencyWeighting::A, Some(-80.0)));
        }
        "multi_frequency_weighting" => {
            let f = to_f64(&x.values(10.0, 20000.0));
            let kinds = [
                FrequencyWeighting::Z,
                FrequencyWeighting::A,
                FrequencyWeighting::C,
            ];
            return thunk(move || multi_frequency_weighting(&f, &kinds, Some(-80.0)));
        }
        _ => return None,
    };
    let f = to_f64(&x.values(10.0, 20000.0));
    thunk(move || weighting(&f, Some(-80.0)))
}

pub fn seq<'a, F: AudioFloat>(_case: &str, _x: &'a SeqCtx<F>) -> Option<Thunk<'a>> {
    None
}
