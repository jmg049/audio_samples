//! Golden-reference checks for the option values reached through `*_with` and `*_slices`
//! functions: fractional `magphase` powers, time-varying `freq`, slice channels and
//! indices, the per-bin metrogram, sparse and custom `nn_filter`, custom `trim` and
//! `split`, and the quality of a random-phase `griffinlim_cqt`.
//!
//! The results are the same arithmetic as the reference on the same input, in `f64`, and
//! agree to 1e-9 of the peak. The fit of `poly_features` uses a different solver.

use audio_samples::operations::constant_q::{AudioConstantQ, CqtConfig, griffinlim_cqt_random};
use audio_samples::operations::effects::{
    ChannelAggregate, NnAggregate, SilenceConfig, SparseRecurrence, nn_filter_sparse,
    nn_filter_sparse_with, split_db_with, trim_db_with,
};
use audio_samples::operations::features::{
    PolyFeaturesConfig, SpectralBandwidthConfig, SpectralRolloffConfig,
    poly_features_from_spectrogram_with_freqs, spectral_bandwidth_from_spectrogram_with_freqs,
    spectral_centroid_from_spectrogram_with_freqs, spectral_rolloff_from_spectrogram_with_freqs,
};
use audio_samples::operations::rhythm::{
    Aggregate, OnsetStrengthConfig, metrogram_per_bin, onset_strength_multi_from_db_slices,
};
use audio_samples::operations::sequence::{SyncAggregate, sync_slices};
use audio_samples::operations::transforms::magphase_powf;
use audio_samples::{AudioSamples, sample_rate};
use ndarray::{Array1, Array2, Array3, Axis, Slice, array, stack};
use serde_json::Value;
use std::num::NonZeroUsize;

use crate::common::{arr2, carr2, f64s, load};

fn fixture() -> Value {
    load("option_values")
}

fn close<D: ndarray::Dimension>(
    what: &str,
    a: ndarray::ArrayView<'_, f64, D>,
    b: ndarray::ArrayView<'_, f64, D>,
    rtol: f64,
) {
    assert_eq!(a.shape(), b.shape(), "{what}");
    let peak = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let worst = a
        .iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    assert!(worst <= rtol * peak, "{what}: {worst:e} vs peak {peak:e}");
}

/// A `(rows, frames)` fixture as `(1, rows, frames)`.
fn with_channel(v: &Value) -> Array3<f64> {
    arr2(v).insert_axis(Axis(0))
}

fn first_channel(a: &Array3<f64>) -> Array2<f64> {
    a.index_axis(Axis(0), 0).to_owned()
}

#[test]
fn magphase_with_a_fractional_power() {
    let f = &fixture()["magphase"];
    let d = carr2(&f["D"]);
    for (name, p) in [("mag_0_5", 0.5), ("mag_1_5", 1.5), ("mag_0_25", 0.25)] {
        let (m, _) = magphase_powf(d.view(), p);
        close(name, m.view(), arr2(&f[name]).view(), 1e-13);
    }
    // A zero magnitude with a negative power is infinite; the fixture writes it as -1.
    let (m, _) = magphase_powf(d.view(), -0.5);
    assert!(m[[3, 2]].is_infinite());
    close(
        "mag_m0_5",
        m.mapv(|v| if v.is_infinite() { -1.0 } else { v }).view(),
        arr2(&f["mag_m0_5"]).view(),
        1e-13,
    );
    let (_, phase) = magphase_powf(d.view(), 1.5);
    let expected = carr2(&f["phase"]);
    for (a, b) in phase.iter().zip(expected.iter()) {
        assert!((a - b).norm() < 1e-13);
    }
}

#[test]
fn time_varying_frequencies() {
    let f = &fixture()["time_varying_freq"];
    let (s, freq) = (with_channel(&f["S"]), with_channel(&f["freq"]));
    let c = spectral_centroid_from_spectrogram_with_freqs(s.view(), freq.view()).unwrap();
    close(
        "centroid",
        first_channel(&c).view(),
        arr2(&f["centroid"]).view(),
        1e-12,
    );
    let b = spectral_bandwidth_from_spectrogram_with_freqs(
        s.view(),
        freq.view(),
        None,
        &SpectralBandwidthConfig::default(),
    )
    .unwrap();
    close(
        "bandwidth",
        first_channel(&b).view(),
        arr2(&f["bandwidth"]).view(),
        1e-12,
    );
    let cfg = SpectralBandwidthConfig {
        p: 3.0,
        norm: false,
        ..SpectralBandwidthConfig::default()
    };
    let b3 =
        spectral_bandwidth_from_spectrogram_with_freqs(s.view(), freq.view(), None, &cfg).unwrap();
    close(
        "bandwidth p=3",
        first_channel(&b3).view(),
        arr2(&f["bandwidth_p3_nonorm"]).view(),
        1e-11,
    );
    let r = spectral_rolloff_from_spectrogram_with_freqs(
        s.view(),
        freq.view(),
        &SpectralRolloffConfig::new().with_roll_percent(0.7),
    )
    .unwrap();
    close(
        "rolloff",
        first_channel(&r).view(),
        arr2(&f["rolloff"]).view(),
        1e-12,
    );
    let cfg = PolyFeaturesConfig {
        order: 2,
        ..PolyFeaturesConfig::default()
    };
    let p = poly_features_from_spectrogram_with_freqs(s.view(), freq.view(), &cfg).unwrap();
    let expected = arr2(&f["poly2"]);
    // Row by row: the coefficients of different powers have very different scales.
    for k in 0..3 {
        close(
            "poly_features",
            p.index_axis(Axis(0), 0).row(k),
            expected.row(k),
            1e-6,
        );
    }
    // One set of frequencies for every channel.
    let two = stack(
        Axis(0),
        &[s.index_axis(Axis(0), 0), s.index_axis(Axis(0), 0)],
    )
    .unwrap();
    let c2 = spectral_centroid_from_spectrogram_with_freqs(two.view(), freq.view()).unwrap();
    assert_eq!(c2.dim().0, 2);
    assert_eq!(c2.index_axis(Axis(0), 1), c.index_axis(Axis(0), 0));
    assert!(
        spectral_centroid_from_spectrogram_with_freqs(
            s.view(),
            freq.slice(ndarray::s![.., 1.., ..])
        )
        .is_err()
    );
}

fn onset_config(aggregate: Aggregate) -> OnsetStrengthConfig {
    let mut cfg = OnsetStrengthConfig::librosa();
    cfg.mel.stft.n_fft = NonZeroUsize::new(512).unwrap();
    cfg.mel.stft.hop_length = NonZeroUsize::new(128);
    cfg.mel.stft.win_length = NonZeroUsize::new(512);
    cfg.aggregate = Some(aggregate);
    cfg
}

#[test]
fn onset_strength_with_slices() {
    let f = &fixture()["slices"];
    let s = with_channel(&f["Sdb"]);
    let bands = [
        Slice::new(0, Some(6), 1),
        Slice::new(4, Some(20), 3),
        Slice::new(20, None, 1),
        // slice(10, 2, -2): bins 10, 8, 6, 4.
        Slice::new(3, Some(11), -2),
    ];
    for (name, agg, lag, detrend) in [
        ("onset_mean", Aggregate::Mean, 1, false),
        ("onset_median", Aggregate::Median, 1, false),
        ("onset_max_detrend", Aggregate::Max, 2, true),
    ] {
        let mut cfg = onset_config(agg);
        cfg.lag = NonZeroUsize::new(lag).unwrap();
        cfg.detrend = detrend;
        let env = onset_strength_multi_from_db_slices(s.view(), None, &cfg, &bands).unwrap();
        close(
            name,
            first_channel(&env).view(),
            arr2(&f[name]).view(),
            1e-12,
        );
    }
    // No maximum of nothing; NaN for the mean, as numpy.
    let empty = [Slice::new(5, Some(5), 1)];
    let cfg = onset_config(Aggregate::Max);
    assert!(onset_strength_multi_from_db_slices(s.view(), None, &cfg, &empty).is_err());
    let cfg = onset_config(Aggregate::Mean);
    let env = onset_strength_multi_from_db_slices(s.view(), None, &cfg, &empty).unwrap();
    assert!(env[[0, 0, 10]].is_nan());
}

#[test]
fn sync_with_slices() {
    let f = &fixture()["slices"];
    let s = arr2(&f["Sdb"]);
    let slices = [
        Slice::new(0, Some(5), 1),
        Slice::new(3, Some(12), 2),
        // slice(30, 8, -3): frames 30, 27, ..., 9.
        Slice::new(9, Some(31), -3),
    ];
    for (name, agg) in [
        ("sync_mean", SyncAggregate::Mean),
        ("sync_median", SyncAggregate::Median),
        ("sync_max", SyncAggregate::Max),
    ] {
        let out = sync_slices(s.view(), &slices, agg).unwrap();
        close(name, out.view(), arr2(&f[name]).view(), 1e-12);
    }
    // Out-of-range bounds are clipped; nothing is left of this one.
    let empty = [Slice::new(1_000_000, None, 1)];
    assert!(sync_slices(s.view(), &empty, SyncAggregate::Max).is_err());
    let nan = sync_slices(s.view(), &empty, SyncAggregate::Median).unwrap();
    assert!(nan[[0, 0]].is_nan());
    let zero = sync_slices(s.view(), &empty, SyncAggregate::Sum).unwrap();
    assert_eq!(zero[[0, 0]], 0.0);
    assert!(
        sync_slices(
            s.view(),
            &[Slice {
                start: 0,
                end: None,
                step: 0
            }],
            SyncAggregate::Mean
        )
        .is_err()
    );
}

fn nested(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().flat_map(nested).collect(),
        _ => vec![v.as_f64().expect("a number")],
    }
}

#[test]
fn metrogram_for_every_bin() {
    let f = &fixture()["metrogram"];
    let tg = with_channel(&f["tg"]);
    let freqs = f64s(&f["freqs"]);
    let m = metrogram_per_bin(tg.view(), &freqs, None, 0.0).unwrap();
    let (n_f, n_t) = (tg.dim().1, tg.dim().2);
    assert_eq!(m.dim(), (1, 4, n_f, n_t));
    close(
        "per_bin",
        Array1::from_iter(m.iter().copied()).view(),
        Array1::from(nested(&f["per_bin"])).view(),
        1e-12,
    );
    let m = metrogram_per_bin(tg.view(), &freqs, Some(&[0.5, 0.25]), 0.25).unwrap();
    assert_eq!(m.dim(), (1, 2, n_f, n_t));
    close(
        "per_bin_factors",
        Array1::from_iter(m.iter().copied()).view(),
        Array1::from(nested(&f["per_bin_factors"])).view(),
        1e-12,
    );
}

#[test]
fn nn_filter_on_a_sparse_matrix() {
    let f = &fixture()["nn_filter"];
    let data = arr2(&f["S"]);
    let indptr: Vec<i32> = f64s(&f["indptr"]).iter().map(|&v| v as i32).collect();
    let indices: Vec<i32> = f64s(&f["indices"]).iter().map(|&v| v as i32).collect();
    let values = f64s(&f["values"]);
    let rec = SparseRecurrence {
        indptr: &indptr,
        indices: &indices,
        values: Some(&values),
    };
    for (name, agg) in [
        ("mean", NnAggregate::Mean),
        ("median", NnAggregate::Median),
        ("average", NnAggregate::WeightedAverage),
    ] {
        let out = nn_filter_sparse(data.view(), &rec, agg).unwrap();
        close(name, out.view(), arr2(&f[name]).view(), 1e-12);
    }
    let reduce = |name: &str, g: fn(f64, f64) -> f64, init: f64| {
        let out = nn_filter_sparse_with(data.view(), &rec, |n, frame: &mut [f64]| {
            for (o, col) in frame.iter_mut().zip(n.columns()) {
                *o = col.iter().copied().fold(init, g);
            }
        })
        .unwrap();
        close(name, out.view(), arr2(&f[name]).view(), 1e-12);
    };
    reduce("max", f64::max, f64::NEG_INFINITY);
    reduce("min", f64::min, f64::INFINITY);
    reduce("sum", |a, b| a + b, 0.0);
    let bad = SparseRecurrence::<f64, i32> {
        indptr: &indptr[1..],
        indices: &indices,
        values: None,
    };
    assert!(nn_filter_sparse(data.view(), &bad, NnAggregate::Mean).is_err());
    let no_values = SparseRecurrence::<f64, i32> {
        indptr: &indptr,
        indices: &indices,
        values: None,
    };
    assert!(nn_filter_sparse(data.view(), &no_values, NnAggregate::WeightedAverage).is_err());
}

fn median_of(mut v: Vec<f64>) -> f64 {
    v.sort_unstable_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

#[test]
fn trim_and_split_with_closures() {
    let f = &fixture()["silence"];
    let y = f64s(&f["y"]);
    let rate = sample_rate!(22050);
    let median_ref =
        |rms: ndarray::ArrayView2<'_, f64>| Ok(median_of(rms.iter().copied().collect()));
    let keep = |db: ndarray::ArrayView2<'_, f64>| Ok(db.index_axis(Axis(0), 0).to_owned());
    let median_channels =
        |db: ndarray::ArrayView2<'_, f64>| Ok(db.map_axis(Axis(0), |c| median_of(c.to_vec())));
    let mono = AudioSamples::new_mono(Array1::from(y.clone()), rate).unwrap();
    let cfg = SilenceConfig::new(20.0);
    let (kept, range) = trim_db_with(&mono, &cfg, median_ref, keep).unwrap();
    let expected = f64s(&f["trim_median_ref"]);
    assert_eq!([range.0 as f64, range.1 as f64], [expected[0], expected[1]]);
    assert_eq!(kept.samples_per_channel().get(), range.1 - range.0);
    let iv = split_db_with(&mono, &cfg, median_ref, keep).unwrap();
    let iv: Array2<f64> = Array2::from_shape_fn((iv.len(), 2), |(i, j)| {
        if j == 0 {
            iv[i].0 as f64
        } else {
            iv[i].1 as f64
        }
    });
    close(
        "split_median_ref",
        iv.view(),
        arr2(&f["split_median_ref"]).view(),
        0.0,
    );

    let right: Vec<f64> = y.iter().rev().map(|v| v * 0.25).collect();
    let stereo_rows = stack(Axis(0), &[ndarray::aview1(&y), ndarray::aview1(&right)]).unwrap();
    let stereo = AudioSamples::new_multi_channel(stereo_rows, rate).unwrap();
    let cfg = SilenceConfig::new(25.0);
    let max_ref = |rms: ndarray::ArrayView2<'_, f64>| {
        Ok(rms.iter().copied().fold(f64::NEG_INFINITY, f64::max))
    };
    let iv = split_db_with(&stereo, &cfg, max_ref, median_channels).unwrap();
    let iv: Array2<f64> = Array2::from_shape_fn((iv.len(), 2), |(i, j)| {
        if j == 0 {
            iv[i].0 as f64
        } else {
            iv[i].1 as f64
        }
    });
    close(
        "split_stereo_median",
        iv.view(),
        arr2(&f["split_stereo_median"]).view(),
        0.0,
    );
    let (_, range) = trim_db_with(&stereo, &cfg, median_ref, median_channels).unwrap();
    let expected = f64s(&f["trim_stereo_both"]);
    assert_eq!([range.0 as f64, range.1 as f64], [expected[0], expected[1]]);
    // A wrong number of levels is an error, not a truncated answer.
    let short = |_db: ndarray::ArrayView2<'_, f64>| Ok(array![0.0]);
    assert!(split_db_with(&stereo, &cfg, max_ref, short).is_err());
    // The closures agree with the configuration they stand for.
    let builtin = cfg.with_aggregate(ChannelAggregate::Max);
    let with_closures = split_db_with(&stereo, &builtin, max_ref, |db| {
        Ok(db.map_axis(Axis(0), |c| {
            c.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        }))
    })
    .unwrap();
    use audio_samples::operations::effects::AudioEffects;
    assert_eq!(with_closures, stereo.split_db(&builtin).unwrap());
}

#[test]
fn griffinlim_cqt_with_random_phases() {
    let root = fixture();
    let f = &root["griffinlim_cqt_random"];
    let y = f64s(&root["y"]);
    let n = y.len();
    let rate = sample_rate!(22050);
    let audio = AudioSamples::new_mono(Array1::from(y), rate).unwrap();
    let cfg = CqtConfig::librosa()
        .with_n_bins(NonZeroUsize::new(f["n_bins"].as_u64().unwrap() as usize))
        .with_tuning(Some(0.0));
    let mag = audio.cqt(&cfg).unwrap().mapv(|z| z.norm());
    let n_iter = f["n_iter"].as_u64().unwrap() as usize;
    let reconstruct = |seed| {
        griffinlim_cqt_random::<f64>(
            &mag,
            &cfg,
            rate,
            n_iter,
            0.99,
            NonZeroUsize::new(n),
            Some(seed),
        )
        .unwrap()
    };
    let a = reconstruct(0);
    assert_eq!(a.samples_per_channel().get(), n);
    // Reproducible for a seed, different for another.
    assert_eq!(a.to_interleaved_vec(), reconstruct(0).to_interleaved_vec());
    assert_ne!(a.to_interleaved_vec(), reconstruct(1).to_interleaved_vec());
    // As close to the target magnitude as the reference's random start, within the
    // spread of one random start against another.
    let rec = a.cqt(&cfg).unwrap().mapv(|z| z.norm());
    let error = (&rec - &mag).mapv(|v| v * v).sum().sqrt() / mag.mapv(|v| v * v).sum().sqrt();
    let reference = f["spectral_convergence"].as_f64().unwrap();
    assert!(error < 1.25 * reference, "{error} vs {reference}");
    assert!(griffinlim_cqt_random::<f64>(&mag, &cfg, rate, 1, -1.0, None, Some(0)).is_err());
}
