//! Native half of the `pitch_filterbank` benchmark group.

use audio_samples::AudioFloat;
use audio_samples::operations::mellin::{MellinConfig, mellin_transform};
use audio_samples::operations::pitch_filterbank::{
    AudioPitchFilterbank, FilterLayout, IirtConfig, MultirateFilterConfig, mr_frequencies,
    semitone_filterbank,
};
use ndarray::{Array1, Axis};

use crate::{AudioCtx, ConstCtx, SeqCtx, Thunk, VecCtx, thunk};

const TINY: usize = 64;

pub fn audio<'a, F: AudioFloat>(case: &str, x: &'a AudioCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "iirt" => {
            let c = IirtConfig::new();
            thunk(move || x.audio.iirt(&c).unwrap())
        }
        _ => None,
    }
}

pub fn constant<'a, F: AudioFloat>(case: &str, _x: &'a ConstCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "fmt_tiny" => {
            let t: Array1<f64> = (0..TINY)
                .map(|i| (-4.0 * i as f64 / (TINY - 1) as f64).exp())
                .collect();
            let t = t.into_dyn();
            let c = MellinConfig::new();
            thunk(move || mellin_transform(t.view(), &c, Axis(0)).unwrap())
        }
        "mr_frequencies" => thunk(|| mr_frequencies(0.0)),
        "semitone_filterbank" => {
            let (f, r) = mr_frequencies(0.0);
            let c = MultirateFilterConfig::new();
            thunk(move || semitone_filterbank(&f, &r, &c, FilterLayout::Sos).unwrap())
        }
        _ => None,
    }
}

pub fn vec<'a, F: AudioFloat>(case: &str, x: &'a VecCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "fmt" => {
            let v = x.values(0.0, 20.0).mapv(|e| (-e).exp()).into_dyn();
            let c = MellinConfig::new().with_n_fmt(Some(16384));
            thunk(move || mellin_transform(v.view(), &c, Axis(0)).unwrap())
        }
        _ => None,
    }
}

pub fn seq<'a, F: AudioFloat>(_case: &str, _x: &'a SeqCtx<F>) -> Option<Thunk<'a>> {
    None
}
