//! Native half of the `segment` benchmark group.

use audio_samples::AudioFloat;
use audio_samples::operations::filterbanks::Window;
use audio_samples::operations::segment::{
    PathEnhanceConfig, agglomerative, diagonal_filter, path_enhance, subsegment, timelag_filter,
};
use audio_samples::operations::sequence::path_to_steps;

use crate::{AudioCtx, ConstCtx, SeqCtx, Thunk, VecCtx, thunk};

const N: usize = 300;

fn uniform(rows: usize, cols: usize, seed: u64) -> ndarray::Array2<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    ndarray::Array2::from_shape_simple_fn((rows, cols), || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 11) as f64 / (1_u64 << 53) as f64
    })
}

fn path(n: usize) -> Vec<[usize; 2]> {
    (0..n).rev().map(|i| [i, i / 2 * 2]).collect()
}

pub fn audio<'a, F: AudioFloat>(_case: &str, _x: &'a AudioCtx<F>) -> Option<Thunk<'a>> {
    None
}

pub fn constant<'a, F: AudioFloat>(case: &str, _x: &'a ConstCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "diagonal_filter" => {
            thunk(|| diagonal_filter(&Window::Hann, 51, 1.5, None, false).unwrap())
        }
        "diagonal_filter_tiny" => {
            thunk(|| diagonal_filter(&Window::Hann, 5, 1.0, None, false).unwrap())
        }
        "path_enhance_tiny" => {
            let r = uniform(12, 12, 31);
            let c = PathEnhanceConfig::new().with_ratios(2.0, None, 2);
            thunk(move || path_enhance(r.view(), 3, &c).unwrap())
        }
        "timelag_filter_tiny" => {
            let r = uniform(12, 12, 31);
            thunk(move || timelag_filter(r.view(), true, |m| Ok(m.to_owned())).unwrap())
        }
        "agglomerative_tiny" => {
            let x = uniform(4, 24, 32);
            thunk(move || agglomerative(x.view(), 3).unwrap())
        }
        "subsegment_tiny" => {
            let x = uniform(4, 24, 32);
            thunk(move || subsegment(x.view(), &[8, 16], 2).unwrap())
        }
        "path_to_steps" => {
            let p = path(20000);
            thunk(move || path_to_steps(&p, false).unwrap())
        }
        "path_to_steps_tiny" => {
            let p = path(16);
            thunk(move || path_to_steps(&p, false).unwrap())
        }
        _ => None,
    }
}

pub fn vec<'a, F: AudioFloat>(_case: &str, _x: &'a VecCtx<F>) -> Option<Thunk<'a>> {
    None
}

pub fn seq<'a, F: AudioFloat>(case: &str, x: &'a SeqCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "path_enhance" => {
            let r = x.matrix(N, N, 31);
            let c = PathEnhanceConfig::new();
            thunk(move || path_enhance(r.view(), 15, &c).unwrap())
        }
        "timelag_filter" => {
            let r = x.matrix(N, N, 31);
            thunk(move || timelag_filter(r.view(), true, |m| Ok(m.to_owned())).unwrap())
        }
        "agglomerative" => {
            let d = x.matrix(12, 2000, 32);
            thunk(move || agglomerative(d.view(), 20).unwrap())
        }
        "subsegment" => {
            let d = x.matrix(12, 2000, 32);
            let f: Vec<i64> = (0..10).map(|i| i * 200).collect();
            thunk(move || subsegment(d.view(), &f, 4).unwrap())
        }
        _ => None,
    }
}
