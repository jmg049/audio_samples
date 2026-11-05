//! Native half of the `array_ops` benchmark group: array helpers and block
//! conversions on the same shapes as the Python half.

use std::num::NonZeroUsize;

use audio_samples::utils::array_ops as ops;
use audio_samples::utils::audio_math::{
    blocks_scale_into, blocks_to_frames, blocks_to_samples, blocks_to_time,
};
use audio_samples::{AudioFloat, nzu};
use ndarray::{Array1, Array2, Axis, Ix2};
use num_complex::Complex;

use crate::ctx::VEC_N;
use crate::{AudioCtx, ConstCtx, SeqCtx, Thunk, VecCtx, thunk};

const SEQ_R: usize = 500;
const SEQ_C: usize = 600;
const TINY: usize = 64;
const BL: NonZeroUsize = nzu!(16);
const HOP: NonZeroUsize = nzu!(512);

/// A deterministic uniform `[0, 1)` sequence.
fn uniform(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (s >> 11) as f64 / (1_u64 << 53) as f64
        })
        .collect()
}

fn tiny_v() -> Array1<f64> {
    Array1::from_iter((0..TINY).map(|i| (30.0 * i as f64 / (TINY - 1) as f64).sin()))
}

fn tiny_m() -> Array2<f64> {
    Array2::from_shape_vec((8, 8), uniform(64, 11)).unwrap()
}

fn events(n: usize, seed: u64) -> Vec<f64> {
    uniform(n, seed).into_iter().map(|v| v * 1000.0).collect()
}

fn intervals(n: usize, seed: u64) -> Array2<f64> {
    let mut s = events(n, seed);
    s.sort_by(f64::total_cmp);
    Array2::from_shape_fn((n, 2), |(i, j)| s[i] + 5.0 * j as f64)
}

fn int_m(rows: usize, cols: usize, scale: f64) -> Array2<i64> {
    Array2::from_shape_vec(
        (rows, cols),
        uniform(rows * cols, 12)
            .into_iter()
            .map(|v| (v * scale) as i64)
            .collect(),
    )
    .unwrap()
}

fn frames() -> Vec<i64> {
    uniform(VEC_N, 13)
        .into_iter()
        .map(|v| (v * 1e6) as i64)
        .collect()
}

fn pcm(n: usize) -> Vec<u8> {
    events(n, 5)
        .into_iter()
        .flat_map(|v| (v as i16).to_le_bytes())
        .collect()
}

fn file_tree() -> std::path::PathBuf {
    let d = std::env::temp_dir().join("aus_bench_files");
    if !d.is_dir() {
        for sub in 0..20 {
            let p = d.join(format!("s{sub}"));
            std::fs::create_dir_all(&p).unwrap();
            for i in 0..100 {
                std::fs::write(p.join(format!("f{i}.wav")), b"").unwrap();
            }
        }
    }
    d
}

pub fn audio<'a, F: AudioFloat>(_case: &str, _x: &'a AudioCtx<F>) -> Option<Thunk<'a>> {
    None
}

pub fn constant<'a, F: AudioFloat>(case: &str, _x: &'a ConstCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "localmax_tiny" => {
            let v = tiny_v();
            thunk(move || ops::localmax(&v, Axis(0)).unwrap())
        }
        "localmin_tiny" => {
            let v = tiny_v();
            thunk(move || ops::localmin(&v, Axis(0)).unwrap())
        }
        "match_events" => {
            let (a, b) = (events(VEC_N, 1), events(5000, 2));
            thunk(move || ops::match_events(&a, &b, true, true).unwrap())
        }
        "match_events_tiny" => {
            let (a, b) = (events(TINY, 1), events(16, 2));
            thunk(move || ops::match_events(&a, &b, true, true).unwrap())
        }
        "match_intervals" => {
            let (a, b) = (intervals(2000, 3), intervals(2000, 4));
            thunk(move || ops::match_intervals(a.view(), b.view(), true).unwrap())
        }
        "match_intervals_tiny" => {
            let (a, b) = (intervals(16, 3), intervals(16, 4));
            thunk(move || ops::match_intervals(a.view(), b.view(), false).unwrap())
        }
        "abs2_tiny" => {
            let v = tiny_v();
            let z = v.mapv(|e| Complex::new(e, 1.0 - e));
            let mut out = Array1::<f64>::zeros(TINY);
            thunk(move || ops::abs2_into(z.view(), out.view_mut()).unwrap())
        }
        "phasor_tiny" => {
            let v = tiny_v();
            thunk(move || ops::phasor(&v))
        }
        "tiny" => thunk(ops::tiny::<F>),
        "expand_to" => {
            let m = tiny_m();
            thunk(move || {
                ops::expand_to(m.view().into_dyn(), 4, &[1, 3])
                    .unwrap()
                    .len()
            })
        }
        "stack_tiny" => {
            let m = tiny_m().into_dyn();
            thunk(move || ops::stack(&[m.view(), m.view()], Axis(0)).unwrap())
        }
        "pad_center_tiny" => {
            let v = tiny_v();
            thunk(move || ops::pad_center(&v, 128, Axis(0), ops::PadMode::Constant(0.0)).unwrap())
        }
        "fix_frames" => {
            let f = frames();
            thunk(move || ops::fix_frames(&f, Some(0), Some(500_000), true).unwrap())
        }
        "fix_frames_tiny" => {
            let f: Vec<i64> = (0..TINY as i64).collect();
            thunk(move || ops::fix_frames(&f, Some(0), Some(50), true).unwrap())
        }
        "index_to_slice" => {
            let f: Vec<i64> = frames()[..5000].to_vec();
            thunk(move || ops::index_to_slice(&f, None, Some(1_000_000), None, true).unwrap())
        }
        "index_to_slice_tiny" => {
            let f: Vec<i64> = (0..TINY as i64).collect();
            thunk(move || ops::index_to_slice(&f, None, Some(50), None, true).unwrap())
        }
        "axis_sort_tiny" => {
            let m = tiny_m();
            thunk(move || ops::axis_sort(m.view(), Axis(1), ops::AxisSortKey::Argmax).unwrap())
        }
        "shear_tiny" => {
            let m = tiny_m();
            thunk(move || ops::shear(m.view(), 1, Axis(1)).unwrap())
        }
        "sparsify_rows_tiny" => {
            let m = tiny_m();
            thunk(move || ops::sparsify_rows(m.view(), 0.1).unwrap())
        }
        "cyclic_gradient_tiny" => {
            let m = tiny_m();
            thunk(move || ops::cyclic_gradient(&m, Axis(1)).unwrap())
        }
        "fill_off_diagonal_tiny" => {
            let mut m = tiny_m();
            thunk(move || ops::fill_off_diagonal(m.view_mut(), 0.25, 0.0).unwrap())
        }
        "count_unique" => {
            let m = int_m(SEQ_R, SEQ_C, 50.0);
            thunk(move || ops::count_unique(&m, Axis(1)).unwrap())
        }
        "count_unique_tiny" => {
            let m = int_m(8, 8, 4.0);
            thunk(move || ops::count_unique(&m, Axis(1)).unwrap())
        }
        "is_unique" => {
            let m = int_m(SEQ_R, SEQ_C, 50.0);
            thunk(move || ops::is_unique(&m, Axis(1)).unwrap())
        }
        "is_unique_tiny" => {
            let m = int_m(8, 8, 4.0);
            thunk(move || ops::is_unique(&m, Axis(1)).unwrap())
        }
        "interp_broadcast_tiny" => {
            let m = tiny_m();
            let half = m.slice(ndarray::s![.., ..;2]).to_owned();
            let p1: Vec<f64> = (0..8).map(f64::from).collect();
            let p2: Vec<f64> = (0..4).map(|i| f64::from(i) * 2.0).collect();
            thunk(move || {
                let (mut a, b) = ops::interp_broadcast(
                    m.view().into_dyn(),
                    &p1,
                    half.view().into_dyn(),
                    &p2,
                    None,
                    ops::InterpKind::Linear,
                    0.0,
                    -1,
                )
                .unwrap();
                a *= &b;
                a
            })
        }
        "dtype_c2r" | "dtype_r2c" => thunk(std::mem::size_of::<Complex<F>>),
        "buf_to_float" => {
            let b = pcm(VEC_N);
            thunk(move || ops::buf_to_float::<f32>(&b, 2).unwrap())
        }
        "buf_to_float_tiny" => {
            let b = pcm(TINY);
            let mut out = vec![0.0_f32; TINY];
            thunk(move || ops::buf_to_float_into::<f32>(&b, 2, &mut out).unwrap())
        }
        "find_files" => {
            let d = file_tree();
            let cfg = ops::FindFilesConfig::new();
            thunk(move || {
                let mut count = 0_usize;
                ops::find_files_each(&d, &cfg, |p| count += p.as_os_str().len()).unwrap();
                count
            })
        }
        "valid_audio_tiny" => {
            let v = tiny_v();
            thunk(move || ops::valid_audio(&v).unwrap())
        }
        "valid_intervals" => {
            let v = intervals(VEC_N, 6);
            thunk(move || ops::valid_intervals(v.view()).unwrap())
        }
        "valid_intervals_tiny" => {
            let v = intervals(16, 6);
            thunk(move || ops::valid_intervals(v.view()).unwrap())
        }
        "is_positive_int" => thunk(|| ops::is_positive_int(std::hint::black_box(5))),
        "valid_int" => {
            thunk(|| ops::valid_int(std::hint::black_box(5.5), ops::IntCast::Floor).unwrap())
        }
        "blocks_to_frames" => {
            let b: Vec<i64> = (0..VEC_N as i64).collect();
            let mut out = vec![0_i64; VEC_N];
            thunk(move || blocks_scale_into(&b, BL.get() as i64, &mut out))
        }
        "blocks_to_frames_tiny" => thunk(|| blocks_to_frames(std::hint::black_box(7), BL)),
        "blocks_to_samples" => {
            let b: Vec<i64> = (0..VEC_N as i64).collect();
            let mut out = vec![0_i64; VEC_N];
            thunk(move || blocks_scale_into(&b, (BL.get() * HOP.get()) as i64, &mut out))
        }
        "blocks_to_samples_tiny" => thunk(|| blocks_to_samples(std::hint::black_box(7), BL, HOP)),
        "blocks_to_time" => {
            let b: Vec<i64> = (0..VEC_N as i64).collect();
            thunk(move || {
                b.iter()
                    .map(|&v| blocks_to_time(v, BL, HOP, 22050.0))
                    .collect::<Vec<_>>()
            })
        }
        "blocks_to_time_tiny" => {
            thunk(|| blocks_to_time(std::hint::black_box(7), BL, HOP, 22050.0))
        }
        _ => None,
    }
}

pub fn vec<'a, F: AudioFloat>(case: &str, x: &'a VecCtx<F>) -> Option<Thunk<'a>> {
    let v = || x.values(0.0, 3000.0).mapv(num_traits::Float::sin);
    match case {
        "localmax" => {
            let v = v();
            thunk(move || ops::localmax(&v, Axis(0)).unwrap())
        }
        "localmin" => {
            let v = v();
            thunk(move || ops::localmin(&v, Axis(0)).unwrap())
        }
        "abs2" => {
            let v = v();
            let r: Array1<F> = v.iter().rev().copied().collect();
            let z = ndarray::Zip::from(&v)
                .and(&r)
                .map_collect(|&a, &b| Complex::new(a, b));
            thunk(move || ops::abs2(&z))
        }
        "phasor" => {
            let v = v();
            thunk(move || ops::phasor(&v))
        }
        "pad_center" => {
            let v = v();
            thunk(move || {
                ops::pad_center(&v, 2 * VEC_N, Axis(0), ops::PadMode::Constant(F::zero())).unwrap()
            })
        }
        "pad_center_reflect" => {
            let v = v();
            thunk(move || ops::pad_center(&v, 2 * VEC_N, Axis(0), ops::PadMode::Reflect).unwrap())
        }
        "valid_audio" => {
            let v = v();
            thunk(move || ops::valid_audio(&v).unwrap())
        }
        _ => None,
    }
}

pub fn seq<'a, F: AudioFloat>(case: &str, x: &'a SeqCtx<F>) -> Option<Thunk<'a>> {
    let m = || x.matrix(SEQ_R, SEQ_C, 11);
    match case {
        "stack" => {
            let m = m().into_dyn();
            thunk(move || ops::stack(&[m.view(), m.view()], Axis(0)).unwrap())
        }
        "axis_sort" => {
            let m = m();
            thunk(move || ops::axis_sort(m.view(), Axis(1), ops::AxisSortKey::Argmax).unwrap())
        }
        "shear" => {
            let m = m();
            thunk(move || ops::shear(m.view(), 1, Axis(1)).unwrap())
        }
        "sparsify_rows" => {
            let m = m();
            if std::mem::size_of::<F>() == 4 {
                let d = m.mapv(|v| v.to_f32().unwrap_or(0.0));
                thunk(move || ops::sparsify_rows(d.view(), 0.1).unwrap())
            } else {
                let d = m.mapv(|v| v.to_f64().unwrap_or(0.0));
                thunk(move || ops::sparsify_rows(d.view(), 0.1).unwrap())
            }
        }
        "cyclic_gradient" => {
            let m = m();
            thunk(move || ops::cyclic_gradient(&m, Axis(1)).unwrap())
        }
        "fill_off_diagonal" => {
            let mut m = m();
            thunk(move || ops::fill_off_diagonal(m.view_mut(), 0.2, F::zero()).unwrap())
        }
        "interp_broadcast" | "interp_broadcast_cubic" => {
            let kind = if case.ends_with("cubic") {
                ops::InterpKind::Cubic
            } else {
                ops::InterpKind::Linear
            };
            let m = m();
            let half = m.slice(ndarray::s![.., ..;2]).to_owned();
            let p1: Vec<f64> = (0..SEQ_C).map(|i| i as f64).collect();
            let p2: Vec<f64> = (0..SEQ_C / 2).map(|i| i as f64 * 2.0).collect();
            thunk(move || {
                let (a, b) = ops::interp_broadcast(
                    m.view().into_dyn(),
                    &p1,
                    half.view().into_dyn(),
                    &p2,
                    None,
                    kind,
                    F::zero(),
                    -1,
                )
                .unwrap();
                let a = a.into_dimensionality::<Ix2>().unwrap();
                let b = b.into_dimensionality::<Ix2>().unwrap();
                ndarray::Zip::from(&a).and(&b).map_collect(|&p, &q| p * q)
            })
        }
        _ => None,
    }
}
