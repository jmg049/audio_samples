//! Tonal analysis: `feature.chroma_stft`, `filters.chroma`, `piptrack`,
//! `estimate_tuning` and `pitch_tuning`, from audio and (`_S` cases) from a
//! spectrogram.
//!
//! Spectrogram inputs are `(channels, bins, frames)`, as the free functions
//! take them (librosa receives the 2-D form of a mono signal).

use audio_samples::operations::tonal::{
    AudioTonal, ChromaFilterbankConfig, ChromaStftConfig, PiptrackConfig, TuningConfig,
    chroma_filterbank, chroma_stft_from_spectrogram, estimate_tuning_from_spectrogram,
    piptrack_from_spectrogram, pitch_tuning,
};
use audio_samples::{AudioFloat, nzu};

use crate::spectrum::to_f64;
use crate::{AudioCtx, ConstCtx, SeqCtx, Thunk, VecCtx, thunk};

pub fn audio<'a, F: AudioFloat>(case: &str, x: &'a AudioCtx<F>) -> Option<Thunk<'a>> {
    let sr = f64::from(x.sr);
    match case {
        "chroma_stft" => {
            let c = ChromaStftConfig::librosa();
            thunk(move || x.audio.chroma_stft(&c).unwrap())
        }
        "chroma_stft_S" => {
            let s = x.s_pow();
            let c = ChromaStftConfig::librosa();
            thunk(move || chroma_stft_from_spectrogram(s.view(), sr, &c).unwrap())
        }
        "piptrack" => {
            let c = PiptrackConfig::librosa();
            thunk(move || x.audio.piptrack(&c).unwrap())
        }
        "piptrack_S" => {
            let s = x.s_mag();
            let c = PiptrackConfig::librosa();
            thunk(move || piptrack_from_spectrogram(s.view(), sr, &c).unwrap())
        }
        "estimate_tuning" => {
            let c = TuningConfig::librosa();
            thunk(move || x.audio.estimate_tuning(&c).unwrap())
        }
        "estimate_tuning_S" => {
            let s = x.s_mag();
            let c = TuningConfig::librosa();
            thunk(move || estimate_tuning_from_spectrogram(s.view(), sr, &c).unwrap())
        }
        _ => None,
    }
}

pub fn constant<'a, F: AudioFloat>(case: &str, _x: &'a ConstCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "filters_chroma" => {
            let c = ChromaFilterbankConfig::librosa();
            thunk(move || chroma_filterbank(22050.0, nzu!(2048), &c).unwrap())
        }
        _ => None,
    }
}

pub fn vec<'a, F: AudioFloat>(case: &str, x: &'a VecCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "pitch_tuning" => {
            let f = to_f64(&x.values(50.0, 4000.0));
            thunk(move || pitch_tuning(&f, 0.01, nzu!(12)).unwrap())
        }
        _ => None,
    }
}

pub fn seq<'a, F: AudioFloat>(_case: &str, _x: &'a SeqCtx<F>) -> Option<Thunk<'a>> {
    None
}
