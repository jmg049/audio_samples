//! Native half of the `array_ops` benchmark group.

use audio_samples::AudioFloat;

use crate::{AudioCtx, ConstCtx, SeqCtx, Thunk, VecCtx};

pub fn audio<'a, F: AudioFloat>(_case: &str, _x: &'a AudioCtx<F>) -> Option<Thunk<'a>> {
    None
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
