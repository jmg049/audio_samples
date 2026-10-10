//! `librosa.feature`: MFCC, deltas, frame-wise spectral features, RMS and
//! zero-crossing rate, from audio and (`_S` cases) from a spectrogram.
//!
//! Spectrogram and feature inputs are `(channels, rows, frames)`, as the free
//! functions take them (librosa receives the 2-D form of a mono signal).

use audio_samples::operations::features::{
    AudioFeatures, PolyFeaturesConfig, RmsConfig, SpectralBandwidthConfig, SpectralCentroidConfig,
    SpectralContrastConfig, SpectralFlatnessConfig, SpectralRolloffConfig, ZeroCrossingRateConfig,
    poly_features_from_spectrogram, rms_from_spectrogram, spectral_bandwidth_from_spectrogram,
    spectral_centroid_from_spectrogram, spectral_contrast_from_spectrogram,
    spectral_flatness_from_spectrogram, spectral_rolloff_from_spectrogram,
};
use audio_samples::operations::spectrum::{
    DeltaConfig, MfccConfig, delta, mfcc_from_mel_db, stack_memory,
};
use audio_samples::{AudioFloat, AudioSpectrum, nzu};

use crate::spectrum::{fft_freqs, mel_db};
use crate::{AudioCtx, ConstCtx, SeqCtx, Thunk, VecCtx, thunk};

pub fn audio<'a, F: AudioFloat>(case: &str, x: &'a AudioCtx<F>) -> Option<Thunk<'a>> {
    let sr = f64::from(x.sr);
    match case {
        "mfcc" => {
            let c = MfccConfig::librosa();
            thunk(move || x.audio.mfcc_frames(&c).unwrap())
        }
        "mfcc_S" => {
            let s = mel_db(x);
            let c = MfccConfig::librosa();
            thunk(move || mfcc_from_mel_db(&s, &c).unwrap())
        }
        "delta" => {
            let m = x.audio.mfcc_frames(&MfccConfig::librosa()).unwrap();
            let c = DeltaConfig::librosa();
            thunk(move || delta(&m, &c).unwrap())
        }
        "stack_memory" => {
            let m = x.audio.mfcc_frames(&MfccConfig::librosa()).unwrap();
            thunk(move || stack_memory(&m, nzu!(2), 1).unwrap())
        }
        "spectral_centroid" => {
            let c = SpectralCentroidConfig::librosa();
            thunk(move || x.audio.spectral_centroid_frames(&c).unwrap())
        }
        "spectral_centroid_S" => {
            let s = x.s_mag();
            let f = fft_freqs(x, 2048);
            thunk(move || spectral_centroid_from_spectrogram(s.view(), &f).unwrap())
        }
        "spectral_bandwidth" => {
            let c = SpectralBandwidthConfig::librosa();
            thunk(move || x.audio.spectral_bandwidth_frames(&c).unwrap())
        }
        "spectral_bandwidth_S" => {
            let s = x.s_mag();
            let f = fft_freqs(x, 2048);
            let c = SpectralBandwidthConfig::librosa();
            thunk(move || spectral_bandwidth_from_spectrogram(s.view(), &f, None, &c).unwrap())
        }
        "spectral_contrast" => {
            let c = SpectralContrastConfig::librosa();
            thunk(move || x.audio.spectral_contrast_frames(&c).unwrap())
        }
        "spectral_contrast_S" => {
            let s = x.s_mag();
            let f = fft_freqs(x, 2048);
            let c = SpectralContrastConfig::librosa();
            thunk(move || spectral_contrast_from_spectrogram(s.view(), &f, sr, &c).unwrap())
        }
        "spectral_flatness" => {
            let c = SpectralFlatnessConfig::librosa();
            thunk(move || x.audio.spectral_flatness_frames(&c).unwrap())
        }
        "spectral_flatness_S" => {
            let s = x.s_mag();
            let c = SpectralFlatnessConfig::librosa();
            thunk(move || spectral_flatness_from_spectrogram(s.view(), &c).unwrap())
        }
        "spectral_rolloff" => {
            let c = SpectralRolloffConfig::librosa();
            thunk(move || x.audio.spectral_rolloff_frames(&c).unwrap())
        }
        "spectral_rolloff_S" => {
            let s = x.s_mag();
            let f = fft_freqs(x, 2048);
            let c = SpectralRolloffConfig::librosa();
            thunk(move || spectral_rolloff_from_spectrogram(s.view(), &f, &c).unwrap())
        }
        "poly_features" => {
            let c = PolyFeaturesConfig::librosa();
            thunk(move || x.audio.poly_features(&c).unwrap())
        }
        "poly_features_S" => {
            let s = x.s_mag();
            let f = fft_freqs(x, 2048);
            let c = PolyFeaturesConfig::librosa();
            thunk(move || poly_features_from_spectrogram(s.view(), &f, &c).unwrap())
        }
        "rms" => {
            let c = RmsConfig::librosa();
            thunk(move || x.audio.rms_frames(&c).unwrap())
        }
        "rms_S" => {
            let s = x.s_mag();
            thunk(move || rms_from_spectrogram(s.view(), nzu!(2048)).unwrap())
        }
        "zero_crossing_rate" => {
            let c = ZeroCrossingRateConfig::librosa();
            thunk(move || x.audio.zero_crossing_rate_frames(&c).unwrap())
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
