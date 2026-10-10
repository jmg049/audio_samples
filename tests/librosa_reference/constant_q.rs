//! `librosa.cqt` / `vqt` / `pseudo_cqt` / `hybrid_cqt` / `icqt`,
//! `filters.cq_to_chroma`, `feature.chroma_cqt` / `chroma_cens` / `chroma_vqt` /
//! `tonnetz`.
//!
//! Error sources, and the tolerances they justify (all bounds are relative to
//! the peak magnitude of the compared matrix):
//!
//! 1. **Top octave** (computed at the input rate, no resampling): the algorithm
//!    is a straight port, so against librosa with complex128 filters
//!    (`cqt_f64_basis_mix_default`) it agrees to round-off (~1e-14). Stock
//!    librosa builds its filters in complex64; that alone moves the output by up
//!    to ~4e-8 of the peak (measured in the generator environment), hence 1e-7
//!    for comparisons against stock librosa where no resampling is involved.
//! 2. **Lower octaves**: librosa decimates by 2 per octave with soxr (`soxr_hq`);
//!    we use the crate's windowed-sinc resampler (`ResamplingQuality::High`). The
//!    two have different transition bands and edge behaviour, and the error
//!    compounds down the octave cascade. On `mix` it is largest in the two
//!    lowest octaves, where the decimated signals are a few hundred samples long
//!    and the abrupt signal start/end dominates. For scale, librosa's own
//!    `soxr_hq` and `soxr_vhq` differ by up to 4.7e-4 of the peak there. Bounds
//!    for whole resampled transforms are set just above the measured value
//!    (printed by the tests) with the per-case reasoning below.

use audio_samples::nzu;
use audio_samples::operations::constant_q::{
    AudioConstantQ, ChromaCensConfig, ChromaCqtConfig, ChromaVqtConfig, CqtConfig, CqtMode,
    TonnetzConfig, VqtConfig, cq_to_chroma,
};
use audio_samples::operations::spectrum::StftPadMode;
use audio_samples::{AudioSamples, sample_rate};
use ndarray::{Array2, Array3, Axis, s};
use num_complex::Complex;
use serde_json::Value;
use std::num::NonZeroUsize;

use crate::common::{arr2, arr3, assert_close, carr2, carr3, f64s, load, mono, stereo};

fn nz(v: &Value) -> NonZeroUsize {
    NonZeroUsize::new(v.as_u64().unwrap() as usize).unwrap()
}

/// Max |a - e| / max |e| over complex arrays.
fn rel_err_c<'a>(
    a: impl IntoIterator<Item = &'a Complex<f64>>,
    e: impl IntoIterator<Item = &'a Complex<f64>>,
) -> f64 {
    let a: Vec<_> = a.into_iter().copied().collect();
    let e: Vec<_> = e.into_iter().copied().collect();
    assert_eq!(a.len(), e.len(), "length mismatch");
    let peak = e.iter().fold(0.0f64, |m, z| m.max(z.norm()));
    let err = a
        .iter()
        .zip(&e)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).norm()));
    err / peak
}

#[track_caller]
fn check_c(what: &str, actual: &Array2<Complex<f64>>, expected: &Array2<Complex<f64>>, tol: f64) {
    assert_eq!(actual.dim(), expected.dim(), "{what}: shape");
    let e = rel_err_c(actual, expected);
    eprintln!("{what}: max |error| / peak = {e:.3e} (bound {tol:.0e})");
    assert!(e <= tol, "{what}: {e:.3e} > {tol:e}");
}

/// Per-octave error (octave 0 = top, computed without resampling), relative to
/// the global peak.
fn octave_errors(
    actual: &Array2<Complex<f64>>,
    expected: &Array2<Complex<f64>>,
    bpo: usize,
) -> Errs {
    let n = expected.nrows();
    let peak = expected.iter().fold(0.0f64, |m, z| m.max(z.norm()));
    (0..n.div_ceil(bpo))
        .map(|i| {
            let hi = n - bpo * i;
            let lo = hi.saturating_sub(bpo);
            let a = actual.slice(s![lo..hi, ..]);
            let e = expected.slice(s![lo..hi, ..]);
            a.iter()
                .zip(e.iter())
                .fold(0.0f64, |m, (x, y)| m.max((x - y).norm()))
                / peak
        })
        .collect::<Vec<_>>()
        .into()
}

fn mono_rows(c: &Array3<Complex<f64>>, step: usize) -> Array2<Complex<f64>> {
    c.index_axis(Axis(0), 0).slice(s![.., ..;step]).to_owned()
}

fn cqt_config(params: &Value) -> CqtConfig {
    let mut cfg = CqtConfig::librosa();
    if let Some(v) = params.get("fmin") {
        cfg.fmin = Some(v.as_f64().unwrap());
    }
    if let Some(v) = params.get("bins_per_octave") {
        cfg.bins_per_octave = nz(v);
    }
    if let Some(v) = params.get("n_bins") {
        cfg.n_bins = Some(nz(v));
    }
    if let Some(v) = params.get("filter_scale") {
        cfg.filter_scale = v.as_f64().unwrap();
    }
    if let Some(v) = params.get("scale") {
        cfg.scale = v.as_bool().unwrap();
    }
    if let Some(v) = params.get("norm") {
        cfg.norm = Some(v.as_f64().unwrap());
    }
    if let Some(v) = params.get("hop_length") {
        cfg.hop_length = nz(v);
    }
    if let Some(v) = params.get("sparsity") {
        cfg.sparsity = v.as_f64().unwrap();
    }
    if let Some(v) = params.get("pad_mode") {
        assert_eq!(v.as_str(), Some("reflect"));
        cfg.pad_mode = StftPadMode::Reflect;
    }
    cfg
}

#[test]
fn cqt_default_per_octave() {
    let fx = load("constant_q");
    let ours = mono_rows(&mono("mix").cqt(&CqtConfig::librosa()).unwrap(), 1);
    let stock = carr2(&fx["cqt_full_mix_default"]);
    let f64_basis = carr2(&fx["cqt_f64_basis_mix_default"]);
    assert_eq!(ours.dim(), stock.dim());

    let oct_f64 = octave_errors(&ours, &f64_basis, 12);
    let oct_stock = octave_errors(&ours, &stock, 12);
    eprintln!("per-octave error vs librosa (complex128 filters), top first: {oct_f64:.2e}");
    eprintln!("per-octave error vs librosa (stock complex64 filters), top first: {oct_stock:.2e}");
    // Top octave: no resampling -> round-off against complex128 filters,
    // complex64 filter rounding against stock librosa (source 1).
    assert!(
        oct_f64[0] < 1e-12,
        "top octave vs f64 filters: {:e}",
        oct_f64[0]
    );
    assert!(
        oct_stock[0] < 1e-7,
        "top octave vs stock: {:e}",
        oct_stock[0]
    );
    // Resampled octaves (source 2): see the measured values printed above.
    for (i, &e) in oct_f64.iter().enumerate().skip(1) {
        assert!(e < 2e-3, "octave {i}: {e:e}");
    }
}

/// Error list printed in scientific notation.
struct Errs(Vec<f64>);

impl From<Vec<f64>> for Errs {
    fn from(v: Vec<f64>) -> Self {
        Self(v)
    }
}

impl std::ops::Index<usize> for Errs {
    type Output = f64;
    fn index(&self, i: usize) -> &f64 {
        &self.0[i]
    }
}

impl Errs {
    fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.0.iter()
    }
}

impl std::fmt::LowerExp for Errs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let parts: Vec<String> = self.0.iter().map(|e| format!("{e:.2e}")).collect();
        write!(f, "[{}]", parts.join(", "))
    }
}

/// Bound for a whole CQT/VQT that goes through the resampling cascade (source 2).
const RESAMPLED: f64 = 5e-4;
/// Bound for results with no resampling against stock librosa (source 1:
/// complex64 filters; pseudo_cqt measured 1.7e-8 / 3.0e-8).
const NO_RESAMPLING: f64 = 1e-7;

/// (whole-matrix bound, middle-half-of-the-signal bound) for each CQT case.
///
/// `RESAMPLED` (5e-4) covers the plain 2:1 cascade: measured 1.2e-5 .. 1.7e-4
/// on `mix` / `notes`; librosa's own `soxr_hq` vs `soxr_vhq` differ by 4.7e-4 on
/// the default case. The two `early_downsample`-type cases decimate by 32 (and
/// 16) in one step before the first octave: `mix`'s 330 Hz partial then lies in
/// the transition band of the decimator (Nyquist 344.5 Hz), where soxr and our
/// filter attenuate differently, and the abrupt start/end of that partial leaks
/// into the low bins of the first and last frames. Measured whole-matrix errors
/// 0.28 and 4.4e-3 (librosa itself moves by 0.33 for `early_downsample` when
/// switching to `res_type="polyphase"`); away from the signal edges the
/// agreement is 1e-5 .. 4e-4, which is what the second bound checks.
fn cqt_bounds(name: &str) -> (f64, f64) {
    match name {
        "early_downsample" => (0.3, 1e-3),
        "fmin65_bpo24_n48" => (6e-3, 1e-3),
        _ => (RESAMPLED, RESAMPLED),
    }
}

/// Error over the subset columns whose frames lie in the middle half of the signal.
fn interior_err(
    actual: &Array2<Complex<f64>>,
    expected: &Array2<Complex<f64>>,
    step: usize,
    hop: usize,
    n: usize,
) -> f64 {
    let peak = expected.iter().fold(0.0f64, |m, z| m.max(z.norm()));
    let mut worst = 0.0f64;
    for j in 0..expected.ncols() {
        let centre = j * step * hop;
        if 4 * centre < n || 4 * centre > 3 * n {
            continue;
        }
        for k in 0..expected.nrows() {
            worst = worst.max((actual[[k, j]] - expected[[k, j]]).norm());
        }
    }
    worst / peak
}

#[test]
fn cqt_variants() {
    let fx = load("constant_q");
    let step = fx["frame_step"].as_u64().unwrap() as usize;
    for (name, case) in fx["cqt"].as_object().unwrap() {
        let cfg = cqt_config(&case["params"]);
        let (whole, interior) = cqt_bounds(name);
        let shape: Vec<usize> = case["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        if name == "stereo_default" {
            let ours = stereo().cqt(&cfg).unwrap();
            assert_eq!(ours.shape(), &shape[..]);
            let exp = carr3(&case["subset"]);
            let ours = ours.slice(s![.., .., ..;step]).to_owned();
            let e = rel_err_c(&ours, &exp);
            eprintln!("cqt {name}: max |error| / peak = {e:.3e} (bound {whole:.0e})");
            assert!(e <= whole, "{name}: {e:e}");
            continue;
        }
        let signal = if name.starts_with("notes") {
            "notes"
        } else {
            "mix"
        };
        let audio = mono(signal);
        let ours = audio.cqt(&cfg).unwrap();
        assert_eq!(&ours.shape()[1..], &shape[..], "{name}: shape");
        let (ours, exp) = (mono_rows(&ours, step), carr2(&case["subset"]));
        check_c(&format!("cqt {name}"), &ours, &exp, whole);
        let n = audio.samples_per_channel().get();
        let e = interior_err(&ours, &exp, step, cfg.hop_length.get(), n);
        eprintln!("cqt {name}: middle half {e:.3e} (bound {interior:.0e})");
        assert!(e <= interior, "{name} interior: {e:e}");
    }
}

#[test]
fn vqt_variants() {
    let fx = load("constant_q");
    let step = fx["frame_step"].as_u64().unwrap() as usize;
    for (name, case) in fx["vqt"].as_object().unwrap() {
        let p = &case["params"];
        let gamma = p.get("gamma").map(|v| v.as_f64().unwrap());
        let cfg = VqtConfig::librosa()
            .with_cqt(cqt_config(p))
            .with_gamma(gamma);
        let ours = mono("mix").vqt(&cfg).unwrap();
        check_c(
            &format!("vqt {name}"),
            &mono_rows(&ours, step),
            &carr2(&case["subset"]),
            RESAMPLED,
        );
    }
}

#[track_caller]
fn check_r(what: &str, actual: &[f64], expected: &[f64], tol: f64) {
    let (_, rel) = crate::common::errors(actual, expected);
    eprintln!("{what}: max |error| / peak = {rel:.3e} (bound {tol:.0e})");
    assert_close(what, actual, expected, tol, 0.0);
}

fn mono_real(x: &Array3<f64>, step: usize) -> Vec<f64> {
    x.index_axis(Axis(0), 0)
        .slice(s![.., ..;step])
        .iter()
        .copied()
        .collect()
}

#[test]
fn pseudo_and_hybrid_cqt() {
    let fx = load("constant_q");
    let step = fx["frame_step"].as_u64().unwrap() as usize;
    let mix = mono("mix");
    // pseudo_cqt: a single STFT, no resampling (source 1).
    let p = mix.pseudo_cqt(&CqtConfig::librosa()).unwrap();
    check_r(
        "pseudo_cqt default",
        &mono_real(&p, step),
        arr2(&fx["pseudo_cqt"]["default"]).as_slice().unwrap(),
        NO_RESAMPLING,
    );
    let cfg = CqtConfig::librosa()
        .with_fmin(Some(220.0))
        .with_n_bins(Some(nzu!(48)))
        .with_scale(false);
    let p = mix.pseudo_cqt(&cfg).unwrap();
    check_r(
        "pseudo_cqt fmin220 scale=false",
        &mono_real(&p, step),
        arr2(&fx["pseudo_cqt"]["fmin220_scale_false"])
            .as_slice()
            .unwrap(),
        NO_RESAMPLING,
    );

    let h = mix.hybrid_cqt(&CqtConfig::librosa()).unwrap();
    check_r(
        "hybrid_cqt default",
        &mono_real(&h, step),
        arr2(&fx["hybrid_cqt"]["default"]).as_slice().unwrap(),
        RESAMPLED,
    );
    let cfg = CqtConfig::librosa()
        .with_bins_per_octave(nzu!(36))
        .with_n_bins(Some(nzu!(252)));
    let h = mono("notes").hybrid_cqt(&cfg).unwrap();
    check_r(
        "hybrid_cqt notes bpo36",
        &mono_real(&h, step),
        arr2(&fx["hybrid_cqt"]["notes_bpo36"]).as_slice().unwrap(),
        RESAMPLED,
    );
}

/// icqt bounds: (whole signal, excluding the first and last `ICQT_EDGE` samples).
///
/// Each octave is inverted at its own rate and upsampled (by up to 64) to the
/// input rate; librosa uses soxr, we use our resampler. Inside the signal the two
/// agree to ~2e-7 .. 6e-5 of the peak (librosa's `soxr_hq` vs `soxr_vhq`: 6.3e-5
/// away from the edges). At the abrupt signal start/end the reconstructed
/// low-octave signals step from zero, and the two interpolation filters ring
/// differently there (the resample fixture shows the same head/tail effect):
/// measured 8.7e-3 within the first ~1000 samples (librosa `soxr_hq` vs
/// `soxr_vhq`: 8.7e-4).
const ICQT_WHOLE: f64 = 1.2e-2;
const ICQT_INTERIOR: f64 = 5e-5;
const ICQT_EDGE: usize = 2048;

fn check_icqt(what: &str, y: &[f64], expected: &[f64], step: usize) {
    check_r(what, y, expected, ICQT_WHOLE);
    let skip = ICQT_EDGE / step;
    let n = expected.len();
    let peak = expected.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let (_, inner) = crate::common::errors(&y[skip..n - skip], &expected[skip..n - skip]);
    let inner = inner
        * expected[skip..n - skip]
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()))
        / peak;
    eprintln!("{what}: away from the edges {inner:.3e} (bound {ICQT_INTERIOR:.0e})");
    assert!(inner <= ICQT_INTERIOR, "{what} interior: {inner:e}");
}

#[test]
fn icqt_of_librosa_cqt() {
    let fx = load("constant_q");
    let istep = fx["icqt"]["step"].as_u64().unwrap() as usize;
    // Input: librosa's own CQT (stored in full), so only the inverse is compared.
    let c = carr2(&fx["cqt_full_mix_default"]).insert_axis(Axis(0));
    let n = mono("mix").samples_per_channel().get();
    let y = AudioSamples::<f64>::icqt(
        &c,
        &CqtConfig::librosa(),
        sample_rate!(22050),
        NonZeroUsize::new(n),
    )
    .unwrap();
    assert_eq!(y.samples_per_channel().get(), n);
    let y: Vec<f64> = y
        .as_mono()
        .unwrap()
        .iter()
        .step_by(istep)
        .copied()
        .collect();
    check_icqt("icqt default", &y, &f64s(&fx["icqt"]["default"]), istep);

    // Round trip through our own forward transform (scale = false).
    let cfg = CqtConfig::librosa().with_scale(false);
    let c = mono("mix").cqt(&cfg).unwrap();
    let y = AudioSamples::<f64>::icqt(&c, &cfg, sample_rate!(22050), NonZeroUsize::new(n)).unwrap();
    let y: Vec<f64> = y
        .as_mono()
        .unwrap()
        .iter()
        .step_by(istep)
        .copied()
        .collect();
    check_icqt(
        "icqt(cqt) scale=false",
        &y,
        &f64s(&fx["icqt"]["noscale"]),
        istep,
    );
}

#[test]
fn cq_to_chroma_matrices() {
    let fx = load("constant_q");
    let m = &fx["cq_to_chroma"];
    // (name, n_input, bins_per_octave, n_chroma, fmin, base_c)
    type Case<'a> = (&'a str, usize, usize, usize, Option<f64>, bool);
    let cases: [Case; 5] = [
        ("default_84", 84, 12, 12, None, true),
        ("bpo36_108", 108, 36, 12, None, true),
        ("fmin65_bpo24_50", 50, 24, 12, Some(65.0), true),
        ("base_a_fmin100", 40, 12, 12, Some(100.0), false),
        ("n24_bpo24", 48, 24, 24, None, true),
    ];
    for (name, n, bpo, nc, fmin, base_c) in cases {
        let ours = cq_to_chroma(
            n,
            NonZeroUsize::new(bpo).unwrap(),
            NonZeroUsize::new(nc).unwrap(),
            fmin,
            base_c,
        )
        .unwrap();
        // Exact 0/1 matrices.
        assert_eq!(ours, arr2(&m[name]), "cq_to_chroma {name}");
    }
}

/// CENS quantises the L1-normalised chroma at 0.05 .. 0.4, so CQT differences of
/// ~1e-4 either leave it unchanged (round-off only, measured < 1e-15) or flip a
/// quantisation step, which would show as an error of order 0.1.
const CENS: f64 = 1e-12;

fn chroma_cfg(tuning: f64) -> ChromaCqtConfig {
    ChromaCqtConfig::librosa().with_tuning(Some(tuning))
}

#[test]
fn chroma_features() {
    let fx = load("constant_q");
    let t_notes = fx["estimated_tuning"]["notes"].as_f64().unwrap();
    // The tuning estimate itself (owned by the tonal module) must agree for the
    // tuning=None cases below to be comparable.
    let est = audio_samples::operations::tonal::estimate_tuning_from_signal(
        notes_rows().view(),
        22050.0,
        &audio_samples::operations::tonal::TuningConfig::librosa().with_bins_per_octave(nzu!(36)),
    )
    .unwrap();
    eprintln!("estimate_tuning(notes, bpo=36): ours {est}, librosa {t_notes}");
    assert!((est - t_notes).abs() < 1e-9);
    let notes = mono("notes");
    let mix = mono("mix");
    let cc = &fx["chroma_cqt"];
    let flat = |x: &Array3<f64>| x.iter().copied().collect::<Vec<_>>();
    let a2 = |v: &Value| arr2(v).iter().copied().collect::<Vec<_>>();

    check_r(
        "chroma_cqt notes tuning=0",
        &flat(&notes.chroma_cqt(&chroma_cfg(0.0)).unwrap()),
        &a2(&cc["notes_tuning0"]),
        RESAMPLED,
    );
    // tuning=None (the default): estimated from the signal, as librosa does.
    check_r(
        "chroma_cqt notes tuning=None",
        &flat(&notes.chroma_cqt(&ChromaCqtConfig::librosa()).unwrap()),
        &a2(&cc["notes_tuning_none"]),
        RESAMPLED,
    );
    // ... and with librosa's estimate supplied explicitly.
    check_r(
        "chroma_cqt notes tuning=librosa estimate",
        &flat(&notes.chroma_cqt(&chroma_cfg(t_notes)).unwrap()),
        &a2(&cc["notes_tuning_none"]),
        RESAMPLED,
    );
    check_r(
        "chroma_cqt mix tuning=0",
        &flat(&mix.chroma_cqt(&chroma_cfg(0.0)).unwrap()),
        &a2(&cc["mix_tuning0"]),
        RESAMPLED,
    );
    let cfg = chroma_cfg(0.0)
        .with_cqt_mode(CqtMode::Hybrid)
        .with_norm(Some(2.0))
        .with_bins_per_octave(Some(nzu!(12)));
    check_r(
        "chroma_cqt notes hybrid l2 bpo12",
        &flat(&notes.chroma_cqt(&cfg).unwrap()),
        &a2(&cc["notes_hybrid_l2_bpo12"]),
        RESAMPLED,
    );
    check_r(
        "chroma_cqt stereo",
        &flat(&stereo().chroma_cqt(&chroma_cfg(0.0)).unwrap()),
        &arr3(&cc["stereo_tuning0"])
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        RESAMPLED,
    );

    let ce = &fx["chroma_cens"];
    let cens = |c: ChromaCqtConfig| ChromaCensConfig::librosa().with_chroma(c);
    check_r(
        "chroma_cens notes tuning=0",
        &flat(&notes.chroma_cens(&cens(chroma_cfg(0.0))).unwrap()),
        &a2(&ce["notes_tuning0"]),
        CENS,
    );
    check_r(
        "chroma_cens notes tuning=None",
        &flat(&notes.chroma_cens(&ChromaCensConfig::librosa()).unwrap()),
        &a2(&ce["notes_tuning_none"]),
        CENS,
    );
    let cfg = cens(chroma_cfg(0.0)).with_win_len_smooth(Some(nzu!(10)));
    check_r(
        "chroma_cens mix win10",
        &flat(&mix.chroma_cens(&cfg).unwrap()),
        &a2(&ce["mix_tuning0_win10"]),
        CENS,
    );

    let cv = &fx["chroma_vqt"];
    check_r(
        "chroma_vqt mix",
        &flat(&mix.chroma_vqt(&ChromaVqtConfig::librosa()).unwrap()),
        &a2(&cv["mix_equal"]),
        RESAMPLED,
    );
    check_r(
        "chroma_vqt notes gamma=None",
        &flat(
            &notes
                .chroma_vqt(&ChromaVqtConfig::librosa().with_gamma(None))
                .unwrap(),
        ),
        &a2(&cv["notes_gamma_none"]),
        RESAMPLED,
    );

    let tz = &fx["tonnetz"];
    let tcfg = |t: f64| TonnetzConfig::librosa().with_chroma(chroma_cfg(t));
    check_r(
        "tonnetz notes tuning=0",
        &flat(&notes.tonnetz(&tcfg(0.0)).unwrap()),
        &a2(&tz["notes_tuning0"]),
        RESAMPLED,
    );
    check_r(
        "tonnetz notes tuning=None",
        &flat(&notes.tonnetz(&TonnetzConfig::librosa()).unwrap()),
        &a2(&tz["notes_tuning_none"]),
        RESAMPLED,
    );
}

fn notes_rows() -> Array2<f64> {
    Array2::from_shape_vec(
        (1, crate::common::signal("notes").len()),
        crate::common::signal("notes"),
    )
    .unwrap()
}

/// A shared test signal as mono `f32` audio (exact: the signals are 16-bit PCM).
fn mono_f32(name: &str) -> AudioSamples<'static, f32> {
    let x: Vec<f32> = crate::common::signal(name)
        .iter()
        .map(|&v| v as f32)
        .collect();
    AudioSamples::new_mono(ndarray::Array1::from(x), sample_rate!(22050)).unwrap()
}

/// `f32` audio against librosa run on the same samples cast to `float32`
/// (complex64 CQT), and against this crate's `f64` path.
///
/// - Top octave (no resampling): f32 FFT and filter-product round-off. Measured
///   3.4e-8 of the peak against librosa and 2.9e-8 against the f64 path; bound
///   2e-7 (1.7 f32 eps).
/// - Decimated octaves against librosa: the resampler difference of the f64
///   comparison (`cqt_default_per_octave`), measured up to 1.3e-4; bound 2e-3 as
///   there.
/// - Every octave against the f64 path: the resampler runs in f64 for both, so
///   only f32 round-off remains; measured <= 2.5e-7, bound 1e-6 (8 eps).
/// - chroma_cqt: against librosa the resampled octaves dominate (measured
///   5.7e-5, bound `RESAMPLED`); against the f64 path measured 1.7e-7, bound
///   1e-6.
#[test]
fn float32_input() {
    let fx = load("constant_q");
    let f = &fx["float32"];
    let c32 = mono_f32("mix").cqt(&CqtConfig::librosa()).unwrap();
    let ours = mono_rows(
        &c32.mapv(|z| Complex::new(f64::from(z.re), f64::from(z.im))),
        1,
    );
    let lib32 = carr2(&f["cqt_mix_default"]);
    let ours64 = mono_rows(&mono("mix").cqt(&CqtConfig::librosa()).unwrap(), 1);
    let oct_lib = octave_errors(&ours, &lib32, 12);
    let oct_64 = octave_errors(&ours, &ours64, 12);
    eprintln!("f32 cqt per-octave error vs librosa float32, top first: {oct_lib:.2e}");
    eprintln!("f32 cqt per-octave error vs f64 path, top first: {oct_64:.2e}");
    assert!(oct_lib[0] < 2e-7, "top octave vs librosa: {:e}", oct_lib[0]);
    for (i, &e) in oct_lib.iter().enumerate().skip(1) {
        assert!(e < 2e-3, "octave {i} vs librosa: {e:e}");
    }
    for (i, &e) in oct_64.iter().enumerate() {
        assert!(e < 1e-6, "octave {i} vs f64: {e:e}");
    }
    let flat = |x: &Array3<f64>| x.iter().copied().collect::<Vec<_>>();
    let a2 = |v: &Value| arr2(v).iter().copied().collect::<Vec<_>>();
    for (name, sig) in [
        ("chroma_cqt_notes_tuning0", "notes"),
        ("chroma_cqt_mix_tuning0", "mix"),
    ] {
        let cfg = chroma_cfg(0.0);
        let got = mono_f32(sig).chroma_cqt(&cfg).unwrap().mapv(f64::from);
        check_r(
            &format!("f32 {name} vs librosa float32"),
            &flat(&got),
            &a2(&f[name]),
            RESAMPLED,
        );
        let got64 = mono(sig).chroma_cqt(&cfg).unwrap();
        check_r(
            &format!("f32 {name} vs f64"),
            &flat(&got),
            &flat(&got64),
            1e-6,
        );
    }
}
