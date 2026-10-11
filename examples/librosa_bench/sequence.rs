//! librosa.sequence and librosa.segment (DTW, RQA, Viterbi decoding,
//! transition matrices, recurrence and cross-similarity, lag conversions),
//! plus `util.sync`, `util.normalize` and `util.frame`.
//!
//! `seq` cases use `SeqCtx::matrix` inputs of the Python half's shapes (other
//! values, same distribution). Audio cases compute chroma, MFCC and `|stft|`
//! with audio_samples outside the timed call and pass the first channel as a
//! 2-D `(features, frames)` array, as librosa receives it.

use audio_samples::operations::sequence::{
    CrossSimilarityConfig, DtwConfig, RecurrenceConfig, RecurrenceMode, RqaConfig, SyncAggregate,
    TransitionWindow, cross_similarity, cross_similarity_bool, dtw, dtw_backtracking,
    lag_to_recurrence, recurrence_matrix, recurrence_matrix_bool, recurrence_to_lag, rqa, sync,
    transition_cycle, transition_local, transition_loop, transition_uniform, viterbi,
    viterbi_binary, viterbi_binary_u16, viterbi_discriminative,
};
use audio_samples::operations::spectrum::{MfccConfig, frame, frame_view};
use audio_samples::operations::tonal::{ChromaStftConfig, NormalizeConfig, normalize_array};
use audio_samples::{AudioFloat, AudioSpectrum, AudioTonal, nzu};
use ndarray::{Array2, Axis, s};

use crate::{AudioCtx, ConstCtx, SeqCtx, Thunk, VecCtx, thunk};

/// Viterbi: 100 states x 2000 steps, width-9 triangle (banded) transitions.
const N_STATES: usize = 100;
const N_STEPS: usize = 2000;
const BAND: usize = 9;
/// Transition-matrix constructors.
const N_TRANS: usize = 1000;
/// `util.sync`: one segment every 21 frames.
const SYNC_STEP: usize = 21;

fn chroma<F: AudioFloat>(x: &AudioCtx<F>) -> Array2<F> {
    x.audio
        .chroma_stft(&ChromaStftConfig::librosa())
        .unwrap()
        .index_axis_move(Axis(0), 0)
}

fn mfcc<F: AudioFloat>(x: &AudioCtx<F>) -> Array2<F> {
    x.audio
        .mfcc_frames(&MfccConfig::librosa())
        .unwrap()
        .index_axis_move(Axis(0), 0)
}

/// The bool connectivity recurrence matrix of the chroma (librosa's lag
/// conversion input is its bool recurrence matrix).
fn rec<F: AudioFloat>(x: &AudioCtx<F>) -> Array2<bool> {
    recurrence_matrix_bool(chroma(x).view(), &RecurrenceConfig::librosa()).unwrap()
}

pub fn audio<'a, F: AudioFloat>(case: &str, x: &'a AudioCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "recurrence_matrix" => {
            let c = chroma(x);
            let cfg = RecurrenceConfig::librosa();
            thunk(move || recurrence_matrix(c.view(), &cfg).unwrap())
        }
        "recurrence_matrix_affinity" => {
            let m = mfcc(x);
            let cfg = RecurrenceConfig::librosa()
                .with_mode(RecurrenceMode::Affinity)
                .with_sym(true);
            thunk(move || recurrence_matrix(m.view(), &cfg).unwrap())
        }
        "recurrence_matrix_bool" => {
            let c = chroma(x);
            let cfg = RecurrenceConfig::librosa();
            thunk(move || recurrence_matrix_bool(c.view(), &cfg).unwrap())
        }
        "cross_similarity_bool" => {
            let c = chroma(x);
            let h = c.ncols() / 2;
            let (a, b) = (
                c.slice(s![.., ..h]).to_owned(),
                c.slice(s![.., h..]).to_owned(),
            );
            let cfg = CrossSimilarityConfig::librosa();
            thunk(move || cross_similarity_bool(a.view(), b.view(), &cfg).unwrap())
        }
        "cross_similarity" => {
            let c = chroma(x);
            let h = c.ncols() / 2;
            let (a, b) = (
                c.slice(s![.., ..h]).to_owned(),
                c.slice(s![.., h..]).to_owned(),
            );
            let cfg = CrossSimilarityConfig::librosa();
            thunk(move || cross_similarity(a.view(), b.view(), &cfg).unwrap())
        }
        "recurrence_to_lag" => {
            let r = rec(x);
            thunk(move || recurrence_to_lag(r.view(), true).unwrap())
        }
        "lag_to_recurrence" => {
            let lag = recurrence_to_lag(rec(x).view(), true).unwrap();
            thunk(move || lag_to_recurrence(lag.view()).unwrap())
        }
        "sync" => {
            let m = mfcc(x);
            let idx: Vec<usize> = (0..m.ncols()).step_by(SYNC_STEP).collect();
            thunk(move || sync(m.view(), &idx, SyncAggregate::Mean, true).unwrap())
        }
        "normalize" => {
            let mag = x.s_mag().index_axis_move(Axis(0), 0);
            let cfg = NormalizeConfig::librosa();
            thunk(move || normalize_array(&mag, &cfg).unwrap())
        }
        "frame_copy" => {
            let y = x.mono().to_vec();
            thunk(move || frame(&y, nzu!(2048), nzu!(512)).unwrap())
        }
        "frame" => {
            let y = x.mono().to_vec();
            // librosa returns a strided view; frame_view is the zero-copy
            // equivalent (frame makes an owned copy).
            thunk(move || frame_view(&y, nzu!(2048), nzu!(512)).unwrap().dim())
        }
        _ => None,
    }
}

pub fn constant<'a, F: AudioFloat>(case: &str, _x: &'a ConstCtx<F>) -> Option<Thunk<'a>> {
    match case {
        "transition_uniform" => thunk(|| transition_uniform::<F>(N_TRANS).unwrap()),
        "transition_loop" => thunk(|| transition_loop::<F>(N_TRANS, &[0.9]).unwrap()),
        "transition_cycle" => thunk(|| transition_cycle::<F>(N_TRANS, &[0.9]).unwrap()),
        "transition_local" => thunk(|| {
            transition_local::<F>(N_TRANS, &[BAND], TransitionWindow::Triangle, false).unwrap()
        }),
        _ => None,
    }
}

pub fn vec<'a, F: AudioFloat>(_case: &str, _x: &'a VecCtx<F>) -> Option<Thunk<'a>> {
    None
}

pub fn seq<'a, F: AudioFloat>(case: &str, x: &'a SeqCtx<F>) -> Option<Thunk<'a>> {
    let band =
        || transition_local::<F>(N_STATES, &[BAND], TransitionWindow::Triangle, false).unwrap();
    match case {
        "dtw" | "dtw_large" => {
            let (n, m) = if case == "dtw" {
                (430, 500)
            } else {
                (1300, 1500)
            };
            let (a, b) = (x.matrix(12, n, 1), x.matrix(12, m, 2));
            let cfg = DtwConfig::librosa();
            thunk(move || dtw(a.view(), b.view(), &cfg).unwrap())
        }
        "dtw_backtracking" => {
            let steps = dtw(
                x.matrix(12, 430, 1).view(),
                x.matrix(12, 500, 2).view(),
                &DtwConfig::librosa(),
            )
            .unwrap()
            .steps;
            thunk(move || dtw_backtracking(steps.view(), None, false, None).unwrap())
        }
        "rqa" => {
            // Binary, recurrence-like: about 20% of the cells set.
            let sim = x.matrix(500, 600, 3).mapv(|v| {
                if v.as_f64() > 0.8 {
                    F::one()
                } else {
                    F::zero()
                }
            });
            let cfg = RqaConfig::librosa();
            thunk(move || rqa(sim.view(), &cfg).unwrap())
        }
        "viterbi" => {
            let prob = x.matrix(N_STATES, N_STEPS, 4);
            let t = band();
            thunk(move || viterbi(prob.view(), t.view(), None, None).unwrap())
        }
        "viterbi_discriminative" => {
            let mut prob = x.matrix(N_STATES, N_STEPS, 4);
            let sums = prob.sum_axis(Axis(0));
            prob /= &sums;
            let t = band();
            thunk(move || viterbi_discriminative(prob.view(), t.view(), None, None, None).unwrap())
        }
        "viterbi_binary_u16" => {
            let prob = x.matrix(N_STATES, N_STEPS, 4);
            let t = transition_loop::<F>(2, &[0.9])
                .unwrap()
                .insert_axis(Axis(0));
            thunk(move || viterbi_binary_u16(prob.view(), t.view(), None, None, None).unwrap())
        }
        "viterbi_binary" => {
            let prob = x.matrix(N_STATES, N_STEPS, 4);
            let t = transition_loop::<F>(2, &[0.9])
                .unwrap()
                .insert_axis(Axis(0));
            thunk(move || viterbi_binary(prob.view(), t.view(), None, None, None).unwrap())
        }
        _ => None,
    }
}
