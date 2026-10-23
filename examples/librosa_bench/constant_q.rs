//! librosa constant-Q functions: cqt / vqt and their inverses, CQT chroma
//! features and the wavelet filterbank.

use std::num::NonZeroUsize;

use audio_samples::operations::constant_q::{
    ChromaCensConfig, ChromaCqtConfig, ChromaVqtConfig, TonnetzConfig, cq_to_chroma,
    tonnetz_from_chroma, wavelet, wavelet_lengths_into,
};
use audio_samples::utils::audio_math::cqt_frequencies;
use audio_samples::{
    AudioConstantQ, AudioFloat, AudioSamples, CqtConfig, VqtConfig, WaveletConfig, nzu,
};

use crate::{AudioCtx, ConstCtx, SeqCtx, Thunk, VecCtx, thunk};

/// `note_to_hz('C1')`, the default CQT's lowest frequency.
const C1: f64 = 32.703_195_662_574_83;

/// The default CQT's 84 centre frequencies (from C1, 12 per octave).
fn cqt_freqs() -> Vec<f64> {
    cqt_frequencies(84, C1, nzu!(12), 0.0)
}

pub fn audio<'a, F: AudioFloat>(case: &str, x: &'a AudioCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "cqt" => {
            let c = CqtConfig::librosa();
            thunk(move || x.audio.cqt(&c).unwrap())
        }
        "vqt" => {
            let c = VqtConfig::librosa();
            thunk(move || x.audio.vqt(&c).unwrap())
        }
        "pseudo_cqt" => {
            let c = CqtConfig::librosa();
            thunk(move || x.audio.pseudo_cqt(&c).unwrap())
        }
        "hybrid_cqt" => {
            let c = CqtConfig::librosa();
            thunk(move || x.audio.hybrid_cqt(&c).unwrap())
        }
        "icqt" => {
            let c = CqtConfig::librosa();
            let cq = x.audio.cqt(&c).unwrap();
            let rate = x.rate();
            let n = NonZeroUsize::new(x.n());
            thunk(move || AudioSamples::<F>::icqt(&cq, &c, rate, n).unwrap())
        }
        "griffinlim_cqt" => {
            // librosa's defaults (32 iterations, momentum 0.99) with init=None.
            let c = CqtConfig::librosa();
            let mag = x.audio.cqt(&c).unwrap().mapv(|z| z.norm());
            let rate = x.rate();
            let n = NonZeroUsize::new(x.n());
            thunk(move || AudioSamples::<F>::griffinlim_cqt(&mag, &c, rate, 32, 0.99, n).unwrap())
        }
        "chroma_cqt" => {
            let c = ChromaCqtConfig::librosa();
            thunk(move || x.audio.chroma_cqt(&c).unwrap())
        }
        "chroma_cens" => {
            let c = ChromaCensConfig::librosa();
            thunk(move || x.audio.chroma_cens(&c).unwrap())
        }
        "chroma_vqt" => {
            let c = ChromaVqtConfig::librosa();
            thunk(move || x.audio.chroma_vqt(&c).unwrap())
        }
        "tonnetz" => {
            let c = TonnetzConfig::librosa();
            thunk(move || x.audio.tonnetz(&c).unwrap())
        }
        "tonnetz_chroma" => {
            let chroma = x.audio.chroma_cqt(&ChromaCqtConfig::librosa()).unwrap();
            thunk(move || tonnetz_from_chroma(&chroma).unwrap())
        }
        _ => None,
    }
}

pub fn constant<'a, F: AudioFloat>(case: &str, _x: &'a ConstCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "wavelet" => {
            let freqs = cqt_freqs();
            let c = WaveletConfig::librosa();
            thunk(move || wavelet(&freqs, 22050.0, &c).unwrap())
        }
        "wavelet_lengths" => {
            let freqs = cqt_freqs();
            let c = WaveletConfig::librosa();
            // librosa's 672-byte lengths come from NumPy's small-block cache,
            // with no allocation during the call; the native form writes into
            // a buffer allocated once.
            let mut lengths = vec![0.0; freqs.len()];
            thunk(move || wavelet_lengths_into(&freqs, 22050.0, &c, &mut lengths).unwrap())
        }
        "cq_to_chroma" => thunk(|| cq_to_chroma(84, nzu!(12), nzu!(12), None, true).unwrap()),
        _ => None,
    }
}

pub fn vec<'a, F: AudioFloat>(_case: &str, _x: &'a VecCtx<F>) -> Option<Thunk<'a>> {
    None
}

pub fn seq<'a, F: AudioFloat>(_case: &str, _x: &'a SeqCtx<F>) -> Option<Thunk<'a>> {
    None
}
