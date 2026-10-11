//! librosa core audio functions (resample, channel conversion, lpc, signal
//! generation, mu-law), `librosa.effects`, `librosa.decompose` and
//! `util.softmask` / `util.fix_length`.
//!
//! Spectrogram inputs are `(channels, bins, frames)`, as the free functions
//! take them (librosa receives the 2-D `(bins, frames)` of a mono signal).

use std::time::Duration;

use audio_samples::operations::effects::{
    ClickPositions, ClicksConfig, HarmonicPercussiveConfig, HpssMaskConfig, NnAggregate,
    PitchShiftConfig, SilenceConfig, clicks, hpss_magnitude, hpss_spectrogram,
    nn_filter_recurrence, softmask,
};
use audio_samples::operations::sequence::RecurrenceConfig;
use audio_samples::operations::spectrum::StftConfig;
use audio_samples::operations::tonal::ChromaStftConfig;
use audio_samples::operations::types::{
    MonoConversionMethod, ResamplingQuality, StereoConversionMethod,
};
use audio_samples::{
    AudioChannelOps, AudioEditing, AudioEffects, AudioFloat, AudioProcessing, AudioSamples,
    AudioStatistics, AudioTonal, chirp, exponential_chirp, mu_law_decode, mu_law_encode, nzu,
    sample_rate, sine_wave,
};
use ndarray::Axis;
use non_empty_slice::NonEmptySlice;
use std::num::NonZeroU32;

use crate::{AudioCtx, ConstCtx, SeqCtx, Thunk, VecCtx, thunk};

/// The Python half's settings (the previous benchmark's).
const STRETCH_RATE: f64 = 1.25;
const PITCH_STEPS: f64 = 4.0;
const LPC_ORDER: usize = 16;
const MU: f64 = 255.0;

/// The pitch-shift resampling ratio for `n_steps = 4`: 2^(-4/12).
fn shift_ratio() -> f64 {
    2f64.powf(-PITCH_STEPS / 12.0)
}

pub fn audio<'a, F: AudioFloat>(case: &str, x: &'a AudioCtx<F>) -> Option<Thunk<'a>> {
    let high = ResamplingQuality::High;
    let resample_to = |target: u32| -> Option<Thunk<'a>> {
        let rate = NonZeroU32::new(target).unwrap();
        thunk(move || x.audio.resample(rate, high).unwrap())
    };
    match case {
        // --- core: audio ----------------------------------------------------
        "resample_22050_16000" => resample_to(16000),
        "resample_22050_44100" => resample_to(44100),
        "resample_22050_11025" => resample_to(11025),
        "resample_16000_22050" => resample_to(22050),
        "resample_by_ratio" => {
            let ratio = shift_ratio();
            thunk(move || x.audio.resample_by_ratio(ratio, high).unwrap())
        }
        "to_mono" => thunk(move || x.audio.to_mono(MonoConversionMethod::Average).unwrap()),
        "to_stereo" => thunk(move || {
            x.audio
                .to_stereo(StereoConversionMethod::Duplicate)
                .unwrap()
        }),
        "to_multi" => {
            let rate = x.rate();
            let chans: Vec<AudioSamples<'static, F>> =
                x.y.axis_iter(Axis(0))
                    .map(|row| AudioSamples::new_mono(row.to_owned(), rate).unwrap())
                    .collect();
            thunk(move || AudioSamples::stack(NonEmptySlice::from_slice(&chans).unwrap()).unwrap())
        }
        "autocorrelate" => thunk(move || x.audio.autocorrelation(nzu!(2047)).unwrap()),
        "lpc" => {
            let order = std::num::NonZeroUsize::new(LPC_ORDER).unwrap();
            thunk(move || x.audio.lpc(order).unwrap())
        }
        "zero_crossings" => thunk(move || x.audio.zero_crossings()),
        "mu_compress" => {
            // mu_law_encode takes f64 samples: converted outside the timed call.
            let v: Vec<f64> = x.y.iter().map(|s| s.as_f64()).collect();
            thunk(move || mu_law_encode(&v, MU).unwrap())
        }
        "mu_compress_float" => thunk(move || x.audio.mu_compress(F::of(MU)).unwrap()),
        "mu_expand" => {
            let v: Vec<f64> = x.y.iter().map(|s| s.as_f64()).collect();
            let codes = mu_law_encode(&v, MU).unwrap();
            thunk(move || mu_law_decode(&codes, MU).unwrap())
        }
        "mu_expand_float" => {
            let comp = x.audio.mu_compress(F::of(MU)).unwrap();
            thunk(move || comp.mu_expand(F::of(MU)).unwrap())
        }
        // --- librosa.effects ------------------------------------------------
        "time_stretch" => {
            let c = StftConfig::librosa();
            thunk(move || x.audio.time_stretch(STRETCH_RATE, &c).unwrap())
        }
        "pitch_shift" => {
            let c = PitchShiftConfig::librosa();
            thunk(move || x.audio.pitch_shift(PITCH_STEPS, &c).unwrap())
        }
        "preemphasis" => thunk(move || x.audio.preemphasis(0.97).unwrap()),
        "deemphasis" => thunk(move || x.audio.deemphasis(0.97).unwrap()),
        "trim" => {
            let c = SilenceConfig::librosa();
            thunk(move || x.audio.trim_db(&c).unwrap())
        }
        "split" => {
            let c = SilenceConfig::librosa();
            thunk(move || x.audio.split_db(&c).unwrap())
        }
        "remix" => {
            // split(y, top_db=8) (about one interval per note), reversed.
            let mut iv = x.audio.split_db(&SilenceConfig::new(8.0)).unwrap();
            iv.reverse();
            thunk(move || x.audio.remix(&iv, true).unwrap())
        }
        "hpss" => {
            let c = HarmonicPercussiveConfig::librosa();
            thunk(move || x.audio.harmonic_percussive(&c).unwrap())
        }
        "harmonic" => {
            let c = HarmonicPercussiveConfig::librosa();
            thunk(move || x.audio.harmonic(&c).unwrap())
        }
        "percussive" => {
            let c = HarmonicPercussiveConfig::librosa();
            thunk(move || x.audio.percussive(&c).unwrap())
        }
        // --- librosa.decompose ----------------------------------------------
        "decompose_hpss" => {
            let d = x.stft();
            let c = HpssMaskConfig::librosa();
            thunk(move || hpss_spectrogram(d.view(), &c).unwrap())
        }
        "nn_filter" => {
            let chroma = x
                .audio
                .chroma_stft(&ChromaStftConfig::librosa())
                .unwrap()
                .index_axis_move(Axis(0), 0);
            let c = RecurrenceConfig::librosa();
            thunk(move || nn_filter_recurrence(chroma.view(), &c, NnAggregate::Mean).unwrap())
        }
        // --- librosa.util ---------------------------------------------------
        "softmask" => {
            let s = x.s_mag();
            let (h, p) = hpss_magnitude(s.view(), &HpssMaskConfig::librosa()).unwrap();
            thunk(move || softmask(&h, &p, 1.0, false).unwrap())
        }
        "fix_length" => {
            let size = x.n() + x.sr as usize;
            thunk(move || x.audio.pad_samples_right(size, F::of(0.0)).unwrap())
        }
        _ => None,
    }
}

/// 120 clicks over 60 s at 22050 Hz (the Python half's `CLICK_TIMES`).
fn click_times() -> Vec<f64> {
    (0..120).map(|i| f64::from(i) * 0.5).collect()
}

pub fn constant<'a, F: AudioFloat>(case: &str, _x: &'a ConstCtx<F>) -> Option<Thunk<'a>> {
    let sr = sample_rate!(22050);
    let secs = Duration::from_secs(10);
    match case {
        "clicks" => {
            let times = click_times();
            let c = ClicksConfig {
                length: Some(60 * 22050),
                ..ClicksConfig::librosa()
            };
            thunk(move || clicks::<f32>(ClickPositions::Times(&times), sr, &c).unwrap())
        }
        "tone" => thunk(move || sine_wave::<f64>(440.0, secs, sr, 1.0)),
        "chirp" => thunk(move || exponential_chirp::<f64>(110.0, 8000.0, secs, sr, 1.0)),
        "chirp_linear" => thunk(move || chirp::<f64>(110.0, 8000.0, secs, sr, 1.0)),
        _ => None,
    }
}

pub fn vec<'a, F: AudioFloat>(_case: &str, _x: &'a VecCtx<F>) -> Option<Thunk<'a>> {
    None
}

pub fn seq<'a, F: AudioFloat>(_case: &str, _x: &'a SeqCtx<F>) -> Option<Thunk<'a>> {
    None
}
