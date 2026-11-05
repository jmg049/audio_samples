//! Native half of the `inverse` benchmark group: mel and MFCC inversion, non-negative
//! least squares and matrix factorisation, on the shapes of the Python half.

use audio_samples::operations::effects::GriffinLimConfig;
use audio_samples::operations::factorization::{NmfConfig, decompose};
use audio_samples::operations::filterbanks::{MelFilterbankConfig, mel_filterbank};
use audio_samples::operations::inverse::{
    MelInversionConfig, MfccInversionConfig, mel_to_audio, mel_to_stft, mfcc_to_audio, mfcc_to_mel,
};
use audio_samples::operations::linalg::{NnlsConfig, nnls, nnls_matrix};
use audio_samples::operations::spectrum::{MelSpectrogramConfig, MfccConfig, StftConfig};
use audio_samples::{AudioFloat, AudioSpectrum, nzu};
use ndarray::{Array2, Array3, Axis};

use crate::{AudioCtx, ConstCtx, SeqCtx, Thunk, VecCtx, thunk};

const SR: f64 = 22050.0;

fn uniform(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    Array2::from_shape_simple_fn((rows, cols), || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 11) as f64 / (1_u64 << 53) as f64
    })
}

fn mel<F: AudioFloat>(x: &AudioCtx<F>) -> Array3<F> {
    let cfg =
        MelSpectrogramConfig::librosa().with_mel(MelFilterbankConfig::new().with_n_mels(nzu!(64)));
    x.audio.melspectrogram(&cfg).unwrap()
}

fn mel_cfg() -> MelInversionConfig {
    MelInversionConfig::new(SR)
}

fn griffin() -> GriffinLimConfig {
    GriffinLimConfig::new().with_n_iter(8)
}

pub fn audio<'a, F: AudioFloat>(case: &str, x: &'a AudioCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "mel_to_stft" => {
            let (m, c) = (mel(x), mel_cfg());
            thunk(move || mel_to_stft(m.view(), &c).unwrap())
        }
        "mel_to_audio" => {
            let (m, c, g) = (mel(x), mel_cfg(), griffin());
            thunk(move || mel_to_audio(m.view(), &c, &g).unwrap())
        }
        "mfcc_to_mel" | "mfcc_to_audio" => {
            let cfg = MfccConfig::librosa().with_n_mfcc(nzu!(20));
            let cfg = cfg.with_mel_spectrogram(
                MelSpectrogramConfig::librosa()
                    .with_mel(MelFilterbankConfig::new().with_n_mels(nzu!(64))),
            );
            let m = x.audio.mfcc_frames(&cfg).unwrap();
            let inv = MfccInversionConfig::new().with_n_mels(nzu!(64));
            if case == "mfcc_to_mel" {
                thunk(move || mfcc_to_mel(m.view(), &inv).unwrap())
            } else {
                let (c, g) = (mel_cfg(), griffin());
                thunk(move || mfcc_to_audio(m.view(), &inv, &c, &g).unwrap())
            }
        }
        "nnls_matrix" => {
            let basis = mel_filterbank(
                SR,
                nzu!(2048),
                &MelFilterbankConfig::new().with_n_mels(nzu!(64)),
            )
            .unwrap()
            .mapv(|v| <F as num_traits::NumCast>::from(v).unwrap());
            let m = mel(x).index_axis_move(Axis(0), 0);
            let c = NnlsConfig::new();
            thunk(move || nnls_matrix(basis.view(), m.view(), &c).unwrap())
        }
        "decompose" => {
            let s = StftConfig::new()
                .with_n_fft(nzu!(512))
                .with_hop_length(Some(nzu!(256)));
            let mag = x
                .audio
                .stft_matrix(&s)
                .unwrap()
                .mapv(num_complex::Complex::norm)
                .index_axis_move(Axis(0), 0);
            let c = NmfConfig::new(8);
            thunk(move || decompose(mag.view(), &c, false).unwrap())
        }
        _ => None,
    }
}

pub fn constant<'a, F: AudioFloat>(case: &str, _x: &'a ConstCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "nnls_vector" => {
            let (a, b) = (
                uniform(200, 60, 23),
                uniform(200, 1, 24).column(0).to_owned(),
            );
            thunk(move || nnls(a.view(), b.view()).unwrap())
        }
        "nnls_vector_tiny" => {
            let (a, b) = (uniform(16, 8, 21), uniform(16, 4, 22).column(0).to_owned());
            thunk(move || nnls(a.view(), b.view()).unwrap())
        }
        "nnls_matrix_tiny" => {
            let (a, b) = (uniform(16, 8, 21), uniform(16, 4, 22));
            thunk(move || nnls_matrix(a.view(), b.view(), &NnlsConfig::new()).unwrap())
        }
        "decompose_tiny" => {
            let a = uniform(16, 8, 21);
            thunk(move || decompose(a.view(), &NmfConfig::new(3), false).unwrap())
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
