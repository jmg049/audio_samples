//! librosa rhythm functions: onset strength and detection, tempograms, tempo,
//! beat tracking, PLP, metrogram and `util.peak_pick`.
//!
//! The plain names call the `AudioRhythm` method on the audio. The `_S` /
//! `_env` / `_tg` variants call the free function on an input computed here
//! with audio_samples (the log-power mel spectrogram, the onset envelope, the
//! autocorrelation tempogram), outside the timed call. Envelopes are passed as
//! `(channels, n)` and tempograms as `(channels, bins, frames)`, the shapes
//! these functions take.

use std::num::NonZeroUsize;

use audio_samples::operations::rhythm::{
    Aggregate, AudioRhythm, BeatTrackConfig, FourierTempogramConfig, HybridTempogramConfig,
    OnsetDetectConfig, OnsetStrengthConfig, PeakPickConfig, PeakPickMethod, PlpConfig, TempoConfig,
    TempogramConfig, TempogramRatioConfig, beat_track_from_envelope, fourier_tempo_frequencies,
    fourier_tempogram_from_envelope, hybrid_tempogram_from_envelope, metrogram, onset_backtrack,
    onset_detect_from_envelope, onset_strength_from_db, onset_strength_multi_from_db, peak_pick,
    plp_from_envelope, tempo_from_envelope, tempo_from_tempogram, tempogram_from_envelope,
    tempogram_ratio_from_tempogram,
};
use audio_samples::operations::spectrum::{DbConfig, MelSpectrogramConfig, power_to_db};
use audio_samples::{AudioFloat, AudioSpectrum, nzu};
use ndarray::{Array2, Array3, s};

use crate::{AudioCtx, ConstCtx, SeqCtx, Thunk, VecCtx, thunk};

const HOP: NonZeroUsize = nzu!(512);
/// Mel sub-band boundaries of `onset_strength_multi_S`.
const BANDS: [usize; 5] = [0, 16, 32, 64, 128];

/// `power_to_db(melspectrogram(y))`, `(channels, n_mels, frames)`.
fn mel_db<F: AudioFloat>(x: &AudioCtx<F>) -> Array3<F> {
    let mel = x
        .audio
        .melspectrogram(&MelSpectrogramConfig::librosa())
        .unwrap();
    power_to_db(&mel, &DbConfig::librosa_power()).unwrap()
}

/// The onset envelope, `(channels, frames)`.
fn envelope<F: AudioFloat>(x: &AudioCtx<F>) -> Array2<F> {
    x.audio
        .onset_strength(&OnsetStrengthConfig::librosa())
        .unwrap()
}

/// The autocorrelation tempogram of the onset envelope.
fn tempogram_of<F: AudioFloat>(x: &AudioCtx<F>) -> Array3<F> {
    tempogram_from_envelope(envelope(x).view(), &TempogramConfig::librosa()).unwrap()
}

pub fn audio<'a, F: AudioFloat>(case: &str, x: &'a AudioCtx<F>) -> Option<Thunk<'a>> {
    let sr = f64::from(x.sr);
    match case {
        // --- on audio ---------------------------------------------------------
        "onset_strength" => {
            let c = OnsetStrengthConfig::librosa();
            thunk(move || x.audio.onset_strength(&c).unwrap())
        }
        "onset_strength_multi" => {
            let c = OnsetStrengthConfig::librosa();
            thunk(move || x.audio.onset_strength_multi(&c, None).unwrap())
        }
        "onset_detect" => {
            let c = OnsetDetectConfig::librosa();
            thunk(move || x.audio.onset_detect(&c).unwrap())
        }
        "tempogram" => {
            let c = TempogramConfig::librosa();
            thunk(move || x.audio.tempogram(&c).unwrap())
        }
        "fourier_tempogram" => {
            let c = FourierTempogramConfig::librosa();
            thunk(move || x.audio.fourier_tempogram(&c).unwrap())
        }
        "hybrid_tempogram" => {
            let c = HybridTempogramConfig::librosa();
            thunk(move || x.audio.hybrid_tempogram(&c).unwrap())
        }
        "tempogram_ratio" => {
            let c = TempogramRatioConfig::librosa();
            thunk(move || x.audio.tempogram_ratio(&c).unwrap())
        }
        "tempo" => {
            let c = TempoConfig::librosa();
            thunk(move || x.audio.tempo(&c).unwrap())
        }
        "beat_track" => {
            let c = BeatTrackConfig::librosa();
            thunk(move || x.audio.beat_track(&c).unwrap())
        }
        "plp" => {
            let c = PlpConfig::librosa();
            thunk(move || x.audio.plp(&c).unwrap())
        }
        // --- on prepared inputs -----------------------------------------------------
        "onset_strength_S" => {
            let db = mel_db(x);
            let c = OnsetStrengthConfig::librosa();
            thunk(move || onset_strength_from_db(db.view(), &c).unwrap())
        }
        "onset_strength_multi_S" => {
            let db = mel_db(x);
            let c = OnsetStrengthConfig::librosa();
            thunk(move || onset_strength_multi_from_db(db.view(), None, &c, Some(&BANDS)).unwrap())
        }
        "onset_detect_env" => {
            let env = envelope(x).row(0).to_vec();
            let c = OnsetDetectConfig::librosa();
            thunk(move || onset_detect_from_envelope(&env, sr, HOP, &c, None).unwrap())
        }
        "onset_backtrack" => {
            let env = envelope(x).row(0).to_vec();
            let events =
                onset_detect_from_envelope(&env, sr, HOP, &OnsetDetectConfig::librosa(), None)
                    .unwrap();
            thunk(move || onset_backtrack(&events, &env))
        }
        "peak_pick" => {
            let env = envelope(x).row(0).to_vec();
            let c = PeakPickConfig {
                pre_max: 3,
                post_max: 3,
                pre_avg: 3,
                post_avg: 5,
                delta: 0.5,
                wait: 10,
                method: PeakPickMethod::Greedy,
            };
            thunk(move || peak_pick(&env, &c).unwrap())
        }
        "tempogram_env" => {
            let env = envelope(x);
            let c = TempogramConfig::librosa();
            thunk(move || tempogram_from_envelope(env.view(), &c).unwrap())
        }
        "fourier_tempogram_env" => {
            let env = envelope(x);
            let c = FourierTempogramConfig::librosa();
            thunk(move || fourier_tempogram_from_envelope(env.view(), &c).unwrap())
        }
        "hybrid_tempogram_env" => {
            let env = envelope(x);
            let c = HybridTempogramConfig::librosa();
            thunk(move || hybrid_tempogram_from_envelope(env.view(), sr, HOP, &c).unwrap())
        }
        "tempogram_ratio_tg" => {
            let tg = tempogram_of(x);
            let c = TempogramRatioConfig::librosa();
            thunk(move || tempogram_ratio_from_tempogram(tg.view(), sr, HOP, &c, None).unwrap())
        }
        "metrogram" => {
            let ft = fourier_tempogram_from_envelope(
                envelope(x).view(),
                &FourierTempogramConfig::librosa(),
            )
            .unwrap();
            let n = ft.dim().2 - 1;
            let mag = ft.slice(s![.., .., ..n]).mapv(|z| z.norm());
            let freqs = fourier_tempo_frequencies(sr, HOP, nzu!(384));
            thunk(move || metrogram(mag.view(), &freqs, None, Aggregate::Sum, 0.0).unwrap())
        }
        "tempo_env" => {
            let env = envelope(x);
            let c = TempoConfig::librosa();
            thunk(move || tempo_from_envelope(env.view(), sr, HOP, &c).unwrap())
        }
        "tempo_tg" => {
            let tg = tempogram_of(x);
            let c = TempoConfig::librosa();
            thunk(move || tempo_from_tempogram(tg.view(), sr, HOP, &c).unwrap())
        }
        "beat_track_env" => {
            let env = envelope(x).row(0).to_vec();
            let c = BeatTrackConfig::librosa();
            thunk(move || beat_track_from_envelope(&env, sr, HOP, &c).unwrap())
        }
        "plp_env" => {
            let env = envelope(x);
            let c = PlpConfig::librosa();
            thunk(move || plp_from_envelope(env.view(), sr, HOP, &c).unwrap())
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
