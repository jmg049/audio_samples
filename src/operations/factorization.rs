//! Non-negative matrix factorisation of spectrograms. Non-negative factors of a
//! magnitude spectrogram read as parts: a spectral template per note or drum, with
//! its gain over time. This is the usual starting point for source separation and
//! transcription.
//!
//! ```rust
//! use audio_samples::operations::factorization::{NmfConfig, decompose};
//! use ndarray::Array2;
//!
//! // Two patterns that alternate frame by frame.
//! let s = Array2::from_shape_fn((16, 8), |(f, t)| {
//!     if f < 8 { (t % 2) as f64 + 0.1 } else { ((t + 1) % 2) as f64 + 0.1 }
//! });
//! let d = decompose(s.view(), &NmfConfig::new(2), false).unwrap();
//! assert_eq!(d.components.dim(), (16, 2));
//! assert_eq!(d.activations.dim(), (2, 8));
//! ```
//!
//! [`nmf`] approximates a non-negative matrix `X` by `W H` with both factors
//! non-negative, by cyclic coordinate descent or multiplicative updates.
//! [`nmf_transform`] finds `W` for a fixed `H`. [`decompose`] applies this to a
//! spectrogram and returns spectral templates and their activations over time.
//!
//! ## Conventions
//!
//! `X` has one row per sample and one column per feature. For a `(bins, frames)`
//! spectrogram, [`decompose`] uses frames as samples, so `W` holds the activations
//! and `H` the templates. The computation runs in the input's float type, except
//! the initial singular value decomposition, which is computed in `f64` and
//! rounded once.
//!
//! ## References
//!
//! - D. D. Lee and H. S. Seung, *Algorithms for non-negative matrix
//!   factorization*, NeurIPS 2001.
//! - C.-J. Hsieh and I. S. Dhillon, *Fast coordinate descent methods with variable
//!   selection for non-negative matrix factorization*, KDD 2011.
//! - C. Févotte and J. Idier, *Algorithms for nonnegative matrix factorization
//!   with the β-divergence*, Neural Computation 23, 2011.
//! - C. Boutsidis and E. Gallopoulos, *SVD based initialization: a head start for
//!   nonnegative matrix factorization*, Pattern Recognition 41, 2008.

use ndarray::{Array1, Array2, ArrayView2, Axis, Zip};

use crate::operations::linalg::{SmallRng, truncated_svd};
use crate::utils::array_ops::{AxisSortKey, axis_sort};
use crate::{AudioFloat, AudioSampleResult, ParameterError};
use ndarray::linalg::general_mat_mul;

fn invalid<T>(parameter: &str, reason: impl Into<String>) -> AudioSampleResult<T> {
    Err(ParameterError::invalid_value(parameter, reason.into()).into())
}

fn of<F: AudioFloat>(v: f64) -> F {
    <F as num_traits::NumCast>::from(v).unwrap_or_else(F::nan)
}

/// How [`nmf`] starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NmfInit {
    /// [`NmfInit::Nndsvda`] when the number of components fits the matrix, otherwise
    /// [`NmfInit::Random`].
    #[default]
    Auto,
    /// Scaled non-negative random values.
    Random,
    /// Non-negative double SVD; keeps zeros, so suited to sparse factors.
    Nndsvd,
    /// Non-negative double SVD with zeros filled by the mean of `X`.
    Nndsvda,
    /// Non-negative double SVD with zeros filled by small random values.
    Nndsvdar,
}

/// The update rule of [`nmf`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NmfSolver {
    /// Cyclic coordinate descent; minimises the Frobenius norm.
    #[default]
    CoordinateDescent,
    /// Multiplicative updates; minimises any β-divergence.
    Multiplicative,
}

/// Parameters of [`nmf`] and [`decompose`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NmfConfig {
    /// Number of components.
    pub n_components: usize,
    /// The starting point. Default [`NmfInit::Auto`].
    pub init: NmfInit,
    /// The update rule. Default coordinate descent.
    pub solver: NmfSolver,
    /// β of the divergence: 2 is the squared Frobenius norm, 1 Kullback-Leibler,
    /// 0 Itakura-Saito. Default 2.
    pub beta: f64,
    /// Relative stopping tolerance. Default `1e-4`.
    pub tol: f64,
    /// Most iterations. Default 200.
    pub max_iter: usize,
    /// Seeds the random parts of the initialisation and any shuffling. Default 0.
    pub seed: u64,
    /// Regularisation strength of `W`. Default 0.
    pub alpha_w: f64,
    /// Regularisation strength of `H`; `None` uses `alpha_w`. Default `None`.
    pub alpha_h: Option<f64>,
    /// Share of L1 in the regularisation, in `[0, 1]`. Default 0.
    pub l1_ratio: f64,
    /// Visit the coordinates in random order. Default false.
    pub shuffle: bool,
}

impl NmfConfig {
    /// Automatic start, coordinate descent, Frobenius norm, tolerance `1e-4`, 200 iterations.
    #[must_use]
    pub const fn new(n_components: usize) -> Self {
        Self {
            n_components,
            init: NmfInit::Auto,
            solver: NmfSolver::CoordinateDescent,
            beta: 2.0,
            tol: 1e-4,
            max_iter: 200,
            seed: 0,
            alpha_w: 0.0,
            alpha_h: None,
            l1_ratio: 0.0,
            shuffle: false,
        }
    }

    /// Settings matching librosa 1.0's defaults; equal to
    /// [`new`](Self::new).
    #[must_use]
    pub const fn librosa(n_components: usize) -> Self {
        Self::new(n_components)
    }

    /// Sets the starting point.
    #[must_use]
    pub const fn with_init(mut self, init: NmfInit) -> Self {
        self.init = init;
        self
    }

    /// Sets the update rule.
    #[must_use]
    pub const fn with_solver(mut self, solver: NmfSolver) -> Self {
        self.solver = solver;
        self
    }

    /// Sets β of the divergence.
    #[must_use]
    pub const fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta;
        self
    }

    /// Sets the stopping tolerance.
    #[must_use]
    pub const fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the most iterations.
    #[must_use]
    pub const fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the seed.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the regularisation of `W`, of `H` (`None` to share) and the L1 share.
    #[must_use]
    pub const fn with_regularisation(
        mut self,
        alpha_w: f64,
        alpha_h: Option<f64>,
        l1_ratio: f64,
    ) -> Self {
        self.alpha_w = alpha_w;
        self.alpha_h = alpha_h;
        self.l1_ratio = l1_ratio;
        self
    }

    /// Visits coordinates in random order or not.
    #[must_use]
    pub const fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    /// [`ParameterError`] if a parameter is out of range, or coordinate descent is asked
    /// for a divergence other than the Frobenius norm.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if self.n_components == 0 {
            return invalid("n_components", "must be positive");
        }
        if !self.tol.is_finite() || self.tol < 0.0 {
            return invalid("tol", "must be finite and non-negative");
        }
        if !self.beta.is_finite() {
            return invalid("beta", "must be finite");
        }
        if self.solver == NmfSolver::CoordinateDescent && self.beta != 2.0 {
            return invalid(
                "beta",
                "coordinate descent minimises only the Frobenius norm (beta = 2)",
            );
        }
        if !self.alpha_w.is_finite()
            || self.alpha_w < 0.0
            || self.alpha_h.is_some_and(|a| !a.is_finite() || a < 0.0)
        {
            return invalid("alpha", "must be finite and non-negative");
        }
        if !(0.0..=1.0).contains(&self.l1_ratio) {
            return invalid("l1_ratio", "must lie in [0, 1]");
        }
        Ok(())
    }
}

/// The result of [`nmf`].
#[derive(Debug, Clone, PartialEq)]
pub struct NmfResult<F> {
    /// `(samples, components)`.
    pub w: Array2<F>,
    /// `(components, features)`.
    pub h: Array2<F>,
    /// Iterations run.
    pub n_iter: usize,
    /// The divergence between `X` and `W H` at the end: the Frobenius norm for `beta = 2`,
    /// otherwise the square root of twice the divergence.
    pub reconstruction_err: f64,
}

fn check_x<F: AudioFloat>(x: ArrayView2<'_, F>) -> AudioSampleResult<()> {
    if x.is_empty() {
        return invalid("x", "must not be empty");
    }
    if !x.fold(true, |ok, v| ok && v.is_finite() && *v >= F::zero()) {
        return invalid("x", "must be finite and non-negative");
    }
    Ok(())
}

/// Non-negative double SVD (Boutsidis and Gallopoulos, 2008).
fn nndsvd<F: AudioFloat>(
    x: ArrayView2<'_, F>,
    k: usize,
    init: NmfInit,
    seed: u64,
) -> AudioSampleResult<(Array2<F>, Array2<F>)> {
    let small_rank = (k as f64) < 0.1 * x.nrows().min(x.ncols()) as f64;
    // The factors overwrite the singular vectors they are made from.
    let (mut w, s_values, mut h) = truncated_svd(x, k, 10, if small_rank { 7 } else { 4 }, seed)?;
    let root = of::<F>(s_values[0].sqrt());
    w.column_mut(0).mapv_inplace(|v| root * v.abs());
    h.row_mut(0).mapv_inplace(|v| root * v.abs());
    // Norms of the positive and negative parts, without storing the parts.
    let norms = |v: ndarray::ArrayView1<'_, F>| -> (F, F) {
        let (mut pos, mut neg) = (F::zero(), F::zero());
        for &e in v {
            if e > F::zero() {
                pos += e * e;
            } else {
                neg += e * e;
            }
        }
        (pos.sqrt(), neg.sqrt())
    };
    for (j, &singular) in s_values.iter().enumerate().take(k).skip(1) {
        let (up, un) = norms(w.column(j));
        let (vp, vn) = norms(h.row(j));
        let (mp, mn) = (up * vp, un * vn);
        let positive = mp > mn;
        let (sigma, nu, nv) = if positive { (mp, up, vp) } else { (mn, un, vn) };
        let lambda = of::<F>((singular * sigma.to_f64().unwrap_or(0.0)).sqrt());
        let take = move |e: F, norm: F| -> F {
            let part = if positive { e } else { F::zero() - e };
            if part > F::zero() && norm > F::zero() {
                lambda * part / norm
            } else {
                F::zero()
            }
        };
        w.column_mut(j).mapv_inplace(|e| take(e, nu));
        h.row_mut(j).mapv_inplace(|e| take(e, nv));
    }
    let small = of::<F>(1e-6);
    w.mapv_inplace(|v| if v < small { F::zero() } else { v });
    h.mapv_inplace(|v| if v < small { F::zero() } else { v });
    let mean = x.fold(0.0, |a, v| a + v.to_f64().unwrap_or(0.0)) / x.len() as f64;
    match init {
        NmfInit::Nndsvda => {
            let m = of::<F>(mean);
            w.mapv_inplace(|v| if v == F::zero() { m } else { v });
            h.mapv_inplace(|v| if v == F::zero() { m } else { v });
        }
        NmfInit::Nndsvdar => {
            let mut rng = SmallRng::new(seed.wrapping_add(1));
            let mut fill = |a: &mut Array2<F>| {
                a.mapv_inplace(|v| {
                    if v == F::zero() {
                        of::<F>(mean * rng.normal().abs() / 100.0)
                    } else {
                        v
                    }
                });
            };
            fill(&mut w);
            fill(&mut h);
        }
        _ => {}
    }
    Ok((w, h))
}

fn random_init<F: AudioFloat>(x: ArrayView2<'_, F>, k: usize, seed: u64) -> (Array2<F>, Array2<F>) {
    let mean = x.fold(0.0, |a, v| a + v.to_f64().unwrap_or(0.0)) / x.len() as f64;
    let avg = (mean / k as f64).sqrt();
    let mut rng = SmallRng::new(seed);
    let (rows, cols) = x.dim();
    let h = Array2::from_shape_simple_fn((k, cols), || of::<F>(avg * rng.normal().abs()));
    let w = Array2::from_shape_simple_fn((rows, k), || of::<F>(avg * rng.normal().abs()));
    (w, h)
}

/// Work space of the coordinate sweeps: one factor by columns, and one gradient per row.
struct SweepBuffers<F> {
    columns: Vec<F>,
    gradient: Vec<F>,
}

impl<F: AudioFloat> SweepBuffers<F> {
    fn new(rows: usize, k: usize) -> Self {
        Self {
            columns: vec![F::zero(); rows * k],
            gradient: vec![F::zero(); rows],
        }
    }
}

/// One sweep of coordinate descent over `w`, given `hht` and `xht`; returns the sum of
/// the projected gradient magnitudes.
///
/// The rows of `w` do not interact within the update of one coordinate, so each coordinate
/// is updated for all of them at once, on `w` held by columns.
fn coordinate_sweep<F: AudioFloat>(
    w: &mut Array2<F>,
    hht: &Array2<F>,
    xht: &Array2<F>,
    order: &[usize],
    buffers: &mut SweepBuffers<F>,
) -> F {
    let (rows, k) = w.dim();
    let hht = hht.as_standard_layout();
    let (Some(by_row), Some(products), Some(targets)) =
        (w.as_slice_mut(), hht.as_slice(), xht.as_slice())
    else {
        unreachable!("the coordinates and their products are in standard layout");
    };
    let columns = &mut buffers.columns[..rows * k];
    let gradient = &mut buffers.gradient[..rows];
    for (c, column) in columns.chunks_exact_mut(rows).enumerate() {
        for (slot, row) in column.iter_mut().zip(by_row.chunks_exact(k)) {
            *slot = row[c];
        }
    }
    #[cfg(target_arch = "x86_64")]
    let violation = if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU supports AVX2.
        unsafe { sweep_avx2(columns, gradient, products, targets, order, k) }
    } else {
        sweep_columns(columns, gradient, products, targets, order, k)
    };
    #[cfg(not(target_arch = "x86_64"))]
    let violation = sweep_columns(columns, gradient, products, targets, order, k);
    for (c, column) in columns.chunks_exact(rows).enumerate() {
        for (value, row) in column.iter().zip(by_row.chunks_exact_mut(k)) {
            row[c] = *value;
        }
    }
    violation
}

/// [`sweep_columns`] compiled for AVX2. No operation is fused, so the values do not depend
/// on the CPU.
///
/// # Safety
/// The CPU must support `avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sweep_avx2<F: AudioFloat>(
    columns: &mut [F],
    gradient: &mut [F],
    products: &[F],
    targets: &[F],
    order: &[usize],
    k: usize,
) -> F {
    sweep_columns(columns, gradient, products, targets, order, k)
}

#[inline(always)]
fn sweep_columns<F: AudioFloat>(
    columns: &mut [F],
    gradient: &mut [F],
    products: &[F],
    targets: &[F],
    order: &[usize],
    k: usize,
) -> F {
    const LANES: usize = 8;
    let rows = gradient.len();
    let mut violation = F::zero();
    for &t in order {
        let hess = products[t * k + t];
        for (g, row) in gradient.iter_mut().zip(targets.chunks_exact(k)) {
            *g = F::zero() - row[t];
        }
        for (column, &product) in columns
            .chunks_exact(rows)
            .zip(&products[t * k..(t + 1) * k])
        {
            for (g, &v) in gradient.iter_mut().zip(column) {
                *g += product * v;
            }
        }
        let column = &mut columns[t * rows..(t + 1) * rows];
        let mut lanes = [F::zero(); LANES];
        let (whole, rest) = gradient.as_chunks::<LANES>();
        let (values, last) = column.as_chunks_mut::<LANES>();
        for (g, v) in whole.iter().zip(values) {
            for (lane, (&g, v)) in lanes.iter_mut().zip(g.iter().zip(v)) {
                *lane += coordinate_update(g, v, hess);
            }
        }
        for lane in lanes {
            violation += lane;
        }
        for (&g, v) in rest.iter().zip(last) {
            violation += coordinate_update(g, v, hess);
        }
    }
    violation
}

/// Moves one coordinate along its gradient, staying non-negative, and returns the
/// magnitude of the projected gradient.
#[inline(always)]
fn coordinate_update<F: AudioFloat>(g: F, value: &mut F, hess: F) -> F {
    let v = *value;
    let negative = if g < F::zero() { g } else { F::zero() };
    let projected = if v == F::zero() { negative } else { g };
    let moved = v - g / hess;
    let kept = if moved > F::zero() { moved } else { F::zero() };
    *value = if hess == F::zero() { v } else { kept };
    projected.abs()
}

struct Regularisation {
    l1_w: f64,
    l2_w: f64,
    l1_h: f64,
    l2_h: f64,
}

impl Regularisation {
    fn new(cfg: &NmfConfig, rows: usize, cols: usize) -> Self {
        let alpha_h = cfg.alpha_h.unwrap_or(cfg.alpha_w);
        Self {
            l1_w: cols as f64 * cfg.alpha_w * cfg.l1_ratio,
            l2_w: cols as f64 * cfg.alpha_w * (1.0 - cfg.l1_ratio),
            l1_h: rows as f64 * alpha_h * cfg.l1_ratio,
            l2_h: rows as f64 * alpha_h * (1.0 - cfg.l1_ratio),
        }
    }
}

fn update_coordinates<F: AudioFloat>(
    target: &mut Array2<F>,
    other: ArrayView2<'_, F>,
    x_other_t: &mut Array2<F>,
    l1: f64,
    l2: f64,
    order: &[usize],
    buffers: &mut SweepBuffers<F>,
) -> F {
    let mut oot = other.dot(&other.t());
    if l2 != 0.0 {
        let l2 = of::<F>(l2);
        for t in 0..oot.nrows() {
            oot[[t, t]] += l2;
        }
    }
    if l1 != 0.0 {
        let l1 = of::<F>(l1);
        x_other_t.mapv_inplace(|v| v - l1);
    }
    coordinate_sweep(target, &oot, x_other_t, order, buffers)
}

fn shuffled(k: usize, rng: &mut SmallRng) -> Vec<usize> {
    let mut order: Vec<usize> = (0..k).collect();
    for i in (1..k).rev() {
        let j = (rng.uniform() * (i + 1) as f64) as usize;
        order.swap(i, j);
    }
    order
}

fn run_coordinate_descent<F: AudioFloat>(
    x: ArrayView2<'_, F>,
    w: &mut Array2<F>,
    h: &mut Array2<F>,
    cfg: &NmfConfig,
    update_h: bool,
) -> usize {
    let reg = Regularisation::new(cfg, x.nrows(), x.ncols());
    let k = cfg.n_components;
    let mut rng = SmallRng::new(cfg.seed.wrapping_add(2));
    let in_order: Vec<usize> = (0..k).collect();
    let tol = of::<F>(cfg.tol);
    let mut first: Option<F> = None;
    let mut iter = 0;
    // H is updated through its transpose, whose rows are the coordinates of each frame.
    let mut ht = h.t().as_standard_layout().into_owned();
    *h = Array2::zeros((0, 0));
    let mut xht = Array2::<F>::zeros((x.nrows(), k));
    let mut xtw = Array2::<F>::zeros(if update_h { (x.ncols(), k) } else { (0, 0) });
    let mut buffers = SweepBuffers::new(x.nrows().max(if update_h { x.ncols() } else { 0 }), k);
    while iter < cfg.max_iter {
        iter += 1;
        let order_w = if cfg.shuffle {
            shuffled(k, &mut rng)
        } else {
            in_order.clone()
        };
        general_mat_mul(F::one(), &x, &ht, F::zero(), &mut xht);
        let mut violation = update_coordinates(
            w,
            ht.t(),
            &mut xht,
            reg.l1_w,
            reg.l2_w,
            &order_w,
            &mut buffers,
        );
        if update_h {
            let order_h = if cfg.shuffle {
                shuffled(k, &mut rng)
            } else {
                in_order.clone()
            };
            general_mat_mul(F::one(), &x.t(), &*w, F::zero(), &mut xtw);
            violation += update_coordinates(
                &mut ht,
                w.t(),
                &mut xtw,
                reg.l1_h,
                reg.l2_h,
                &order_h,
                &mut buffers,
            );
        }
        match first {
            None => {
                if violation == F::zero() {
                    break;
                }
                first = Some(violation);
            }
            Some(v0) => {
                if violation / v0 <= tol {
                    break;
                }
            }
        }
    }
    *h = ht.t().as_standard_layout().into_owned();
    iter
}

/// The β-divergence between `x` and `wh`.
fn beta_divergence<F: AudioFloat>(x: ArrayView2<'_, F>, wh: &Array2<F>, beta: f64) -> f64 {
    const EPS: f64 = 1.192_092_9e-7;
    let mut sum = 0.0;
    Zip::from(&x).and(wh).for_each(|&xv, &wv| {
        let a = xv.to_f64().unwrap_or(0.0);
        let b = wv.to_f64().unwrap_or(0.0);
        sum += if beta == 2.0 {
            (a - b) * (a - b) * 0.5
        } else if beta == 1.0 {
            if a > EPS {
                a * (a / b.max(EPS)).ln() - a + b
            } else {
                b
            }
        } else if beta == 0.0 {
            if a > EPS {
                a / b.max(EPS) - (a / b.max(EPS)).ln() - 1.0
            } else {
                0.0
            }
        } else if a > EPS {
            (a.powf(beta) + (beta - 1.0) * b.max(EPS).powf(beta)
                - beta * a * b.max(EPS).powf(beta - 1.0))
                / (beta * (beta - 1.0))
        } else {
            b.max(EPS).powf(beta) / beta
        };
    });
    sum
}

/// One multiplicative update of `w` with `h` fixed.
fn multiplicative_w<F: AudioFloat>(
    x: ArrayView2<'_, F>,
    w: &mut Array2<F>,
    h: &Array2<F>,
    beta: f64,
    gamma: f64,
    l1: f64,
    l2: f64,
) {
    let eps = of::<F>(1.192_092_9e-7);
    let (mut numerator, mut denominator) = if beta == 2.0 {
        let hht = h.dot(&h.t());
        (x.dot(&h.t()), w.dot(&hht))
    } else {
        let wh = w.dot(h);
        let safe = wh.mapv(|v| if v == F::zero() { eps } else { v });
        if beta == 1.0 {
            let ratio = Zip::from(&x).and(&safe).map_collect(|&a, &b| a / b);
            let sums: Array1<F> = h
                .sum_axis(Axis(1))
                .mapv(|v| if v == F::zero() { eps } else { v });
            let mut den = Array2::zeros(w.raw_dim());
            for mut row in den.rows_mut() {
                row.assign(&sums);
            }
            (ratio.dot(&h.t()), den)
        } else {
            let (bm2, bm1) = (of::<F>(beta - 2.0), of::<F>(beta - 1.0));
            let ratio = Zip::from(&x)
                .and(&safe)
                .map_collect(|&a, &b| a * b.powf(bm2));
            let powered = safe.mapv(|b| b.powf(bm1));
            (ratio.dot(&h.t()), powered.dot(&h.t()))
        }
    };
    if l1 != 0.0 {
        let l1 = of::<F>(l1);
        denominator.mapv_inplace(|v| v + l1);
    }
    if l2 != 0.0 {
        let l2 = of::<F>(l2);
        Zip::from(&mut denominator)
            .and(&*w)
            .for_each(|d, &v| *d += l2 * v);
    }
    denominator.mapv_inplace(|v| if v == F::zero() { eps } else { v });
    Zip::from(&mut numerator)
        .and(&denominator)
        .for_each(|n, &d| *n /= d);
    if gamma != 1.0 {
        let g = of::<F>(gamma);
        numerator.mapv_inplace(|v| v.powf(g));
    }
    Zip::from(&mut *w).and(&numerator).for_each(|a, &d| *a *= d);
}

fn run_multiplicative<F: AudioFloat>(
    x: ArrayView2<'_, F>,
    w: &mut Array2<F>,
    h: &mut Array2<F>,
    cfg: &NmfConfig,
    update_h: bool,
) -> usize {
    let reg = Regularisation::new(cfg, x.nrows(), x.ncols());
    let beta = cfg.beta;
    let gamma = if beta < 1.0 {
        1.0 / (2.0 - beta)
    } else if beta > 2.0 {
        1.0 / (beta - 1.0)
    } else {
        1.0
    };
    let xt = x.t();
    let initial = beta_divergence(x, &w.dot(&*h), beta);
    let scale = (2.0 * initial).sqrt();
    let mut previous = scale;
    let mut iter = 0;
    let mut ht = h.t().to_owned();
    while iter < cfg.max_iter {
        iter += 1;
        let current_h = ht.t().to_owned();
        multiplicative_w(x, w, &current_h, beta, gamma, reg.l1_w, reg.l2_w);
        if beta < 1.0 {
            w.mapv_inplace(|v| if v < F::epsilon() { F::zero() } else { v });
        }
        if update_h {
            let wt = w.t().to_owned();
            multiplicative_w(xt, &mut ht, &wt, beta, gamma, reg.l1_h, reg.l2_h);
            if beta <= 1.0 {
                ht.mapv_inplace(|v| if v < F::epsilon() { F::zero() } else { v });
            }
        }
        if cfg.tol > 0.0 && iter % 10 == 0 {
            let error = (2.0 * beta_divergence(x, &w.dot(&ht.t()), beta)).sqrt();
            if (previous - error) / scale < cfg.tol {
                break;
            }
            previous = error;
        }
    }
    h.assign(&ht.t());
    iter
}

/// Factorises a non-negative matrix as `W H`.
///
/// # Arguments
/// - `x`: the `(samples, features)` matrix.
/// - `config`: the number of components, the starting point and the update rule.
///
/// # Returns
/// The factors, the iterations run and the final error.
///
/// # Errors
/// [`ParameterError`] for an invalid configuration, or if `x` is empty, non-finite or has
/// a negative entry, or if an SVD start asks for more components than the matrix has.
///
/// # Examples
/// ```
/// use audio_samples::operations::factorization::{NmfConfig, nmf};
/// use ndarray::array;
///
/// let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]];
/// let r = nmf(x.view(), &NmfConfig::new(2)).unwrap();
/// assert!(r.reconstruction_err < 1e-3);
/// ```
pub fn nmf<F: AudioFloat>(
    x: ArrayView2<'_, F>,
    config: &NmfConfig,
) -> AudioSampleResult<NmfResult<F>> {
    config.validate()?;
    check_x(x)?;
    let k = config.n_components;
    let fits = k <= x.nrows().min(x.ncols());
    let (w, h) = match config.init {
        NmfInit::Auto if fits => nndsvd(x, k, NmfInit::Nndsvda, config.seed)?,
        NmfInit::Auto | NmfInit::Random => random_init(x, k, config.seed),
        svd_init => {
            if !fits {
                return invalid(
                    "n_components",
                    format!("an SVD start allows at most {}", x.nrows().min(x.ncols())),
                );
            }
            nndsvd(x, k, svd_init, config.seed)?
        }
    };
    nmf_from(x, w, h, config)
}

/// [`nmf`] from a given starting point.
///
/// # Arguments
/// - `x`: the `(samples, features)` matrix.
/// - `w`: the starting `(samples, components)` factor.
/// - `h`: the starting `(components, features)` factor.
/// - `config`: the update rule; `init` is not used.
///
/// # Returns
/// As for [`nmf`].
///
/// # Errors
/// [`ParameterError`] as for [`nmf`], and if the shapes disagree or a factor has a
/// negative entry.
pub fn nmf_from<F: AudioFloat>(
    x: ArrayView2<'_, F>,
    mut w: Array2<F>,
    mut h: Array2<F>,
    config: &NmfConfig,
) -> AudioSampleResult<NmfResult<F>> {
    config.validate()?;
    check_x(x)?;
    let k = config.n_components;
    if w.dim() != (x.nrows(), k) || h.dim() != (k, x.ncols()) {
        return invalid(
            "w",
            "the starting factors do not fit the matrix and n_components",
        );
    }
    if w.iter()
        .chain(h.iter())
        .any(|v| !v.is_finite() || *v < F::zero())
    {
        return invalid("w", "the starting factors must be finite and non-negative");
    }
    let n_iter = match config.solver {
        NmfSolver::CoordinateDescent => run_coordinate_descent(x, &mut w, &mut h, config, true),
        NmfSolver::Multiplicative => run_multiplicative(x, &mut w, &mut h, config, true),
    };
    let reconstruction_err = (2.0 * beta_divergence(x, &w.dot(&h), config.beta)).sqrt();
    Ok(NmfResult {
        w,
        h,
        n_iter,
        reconstruction_err,
    })
}

/// Finds `W` for fixed `H`.
///
/// # Arguments
/// - `x`: the `(samples, features)` matrix.
/// - `h`: the `(components, features)` factor.
/// - `config`: the update rule. `n_components` is taken from `h`.
///
/// # Returns
/// The `(samples, components)` factor.
///
/// # Errors
/// [`ParameterError`] as for [`nmf`], and if `h` does not fit `x`.
///
/// # Examples
/// ```
/// use audio_samples::operations::factorization::{NmfConfig, nmf_transform};
/// use ndarray::array;
///
/// let x = array![[2.0_f64, 0.0], [0.0, 3.0]];
/// let h = array![[1.0, 0.0], [0.0, 1.0]];
/// let w = nmf_transform(x.view(), h.view(), &NmfConfig::new(2)).unwrap();
/// assert!((w[[0, 0]] - 2.0).abs() < 1e-6);
/// ```
pub fn nmf_transform<F: AudioFloat>(
    x: ArrayView2<'_, F>,
    h: ArrayView2<'_, F>,
    config: &NmfConfig,
) -> AudioSampleResult<Array2<F>> {
    let mut cfg = *config;
    cfg.n_components = h.nrows();
    cfg.validate()?;
    check_x(x)?;
    if h.ncols() != x.ncols() {
        return invalid(
            "h",
            format!("{} features for a matrix of {}", h.ncols(), x.ncols()),
        );
    }
    let mean = x.fold(0.0, |a, v| a + v.to_f64().unwrap_or(0.0)) / x.len() as f64;
    let avg = of::<F>((mean / cfg.n_components as f64).sqrt());
    let mut w = Array2::from_elem((x.nrows(), cfg.n_components), avg);
    let mut fixed = h.to_owned();
    match cfg.solver {
        NmfSolver::CoordinateDescent => {
            run_coordinate_descent(x, &mut w, &mut fixed, &cfg, false);
        }
        NmfSolver::Multiplicative => {
            run_multiplicative(x, &mut w, &mut fixed, &cfg, false);
        }
    }
    Ok(w)
}

/// A spectrogram's templates and activations.
#[derive(Debug, Clone, PartialEq)]
pub struct Decomposition<F> {
    /// `(bins, components)` spectral templates.
    pub components: Array2<F>,
    /// `(components, frames)` gains over time.
    pub activations: Array2<F>,
    /// Iterations run.
    pub n_iter: usize,
    /// The final error.
    pub reconstruction_err: f64,
}

/// Factorises a spectrogram into spectral templates and activations.
///
/// # Arguments
/// - `s`: a non-negative `(bins, frames)` spectrogram.
/// - `config`: as for [`nmf`].
/// - `sort`: order the components by the bin of their largest value.
///
/// # Returns
/// The templates and activations, with `s` close to `components * activations`.
///
/// # Errors
/// As for [`nmf`].
///
/// # Examples
/// See the [module documentation](self).
pub fn decompose<F: AudioFloat>(
    s: ArrayView2<'_, F>,
    config: &NmfConfig,
    sort: bool,
) -> AudioSampleResult<Decomposition<F>> {
    let r = nmf(s.t(), config)?;
    let components = r.h.reversed_axes();
    let activations = r.w.reversed_axes();
    let (components, activations) = if sort {
        let (sorted, idx) = axis_sort(components.view(), Axis(1), AxisSortKey::Argmax)?;
        (sorted, activations.select(Axis(0), &idx))
    } else {
        (components, activations)
    };
    Ok(Decomposition {
        components,
        activations,
        n_iter: r.n_iter,
        reconstruction_err: r.reconstruction_err,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::linalg::SmallRng;

    fn low_rank(rows: usize, cols: usize, k: usize, seed: u64) -> Array2<f64> {
        let mut rng = SmallRng::new(seed);
        let w = Array2::from_shape_simple_fn((rows, k), || rng.uniform());
        let h = Array2::from_shape_simple_fn((k, cols), || rng.uniform());
        w.dot(&h)
    }

    #[test]
    fn recovers_low_rank_with_every_start_and_solver() {
        let x = low_rank(40, 30, 3, 1);
        let peak = x.iter().fold(0.0_f64, |a, &b| a.max(b));
        for init in [
            NmfInit::Auto,
            NmfInit::Random,
            NmfInit::Nndsvd,
            NmfInit::Nndsvda,
            NmfInit::Nndsvdar,
        ] {
            let cfg = NmfConfig::new(3)
                .with_init(init)
                .with_max_iter(2000)
                .with_tol(1e-7);
            let r = nmf(x.view(), &cfg).unwrap();
            assert!(
                r.reconstruction_err < 0.05 * peak * 30.0,
                "{init:?} {}",
                r.reconstruction_err
            );
        }
        for beta in [2.0, 1.0, 0.0, 0.5, 3.0] {
            let cfg = NmfConfig::new(3)
                .with_solver(NmfSolver::Multiplicative)
                .with_beta(beta)
                .with_max_iter(500);
            let r = nmf(x.view(), &cfg).unwrap();
            assert!(r.w.iter().all(|&v| v >= 0.0) && r.h.iter().all(|&v| v >= 0.0));
            assert!(r.reconstruction_err.is_finite(), "{beta}");
        }
    }

    #[test]
    fn transform_recovers_activations() {
        let x = low_rank(20, 25, 2, 2);
        let cfg = NmfConfig::new(2).with_max_iter(2000).with_tol(1e-8);
        let r = nmf(x.view(), &cfg).unwrap();
        let w = nmf_transform(x.view(), r.h.view(), &cfg).unwrap();
        let fit = w.dot(&r.h);
        let err = (&fit - &x).mapv(f64::abs).sum() / x.sum();
        assert!(err < 1e-2, "{err}");
    }

    #[test]
    fn regularisation_shrinks_the_factors() {
        let x = low_rank(20, 25, 2, 3);
        let plain = nmf(x.view(), &NmfConfig::new(2)).unwrap();
        let cfg = NmfConfig::new(2).with_regularisation(0.1, None, 1.0);
        let reg = nmf(x.view(), &cfg).unwrap();
        assert!(reg.h.sum() + reg.w.sum() < plain.h.sum() + plain.w.sum());
    }

    #[test]
    fn rejects_bad_input() {
        let x = low_rank(4, 5, 1, 4);
        assert!(nmf(x.view(), &NmfConfig::new(0)).is_err());
        assert!(nmf((-&x).view(), &NmfConfig::new(1)).is_err());
        let cfg = NmfConfig::new(9).with_init(NmfInit::Nndsvd);
        assert!(nmf(x.view(), &cfg).is_err());
        let cfg = NmfConfig::new(1).with_beta(1.0);
        assert!(nmf(x.view(), &cfg).is_err());
        assert!(nmf_transform(x.view(), Array2::zeros((1, 3)).view(), &NmfConfig::new(1)).is_err());
    }

    #[test]
    fn sort_orders_components_by_peak_bin() {
        let x = low_rank(30, 20, 3, 5);
        let d = decompose(x.view(), &NmfConfig::new(3), true).unwrap();
        let peaks: Vec<usize> = d
            .components
            .columns()
            .into_iter()
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold((0, f64::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b })
                    .0
            })
            .collect();
        assert!(peaks.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn single_precision_runs() {
        let x = low_rank(10, 12, 2, 6).mapv(|v| v as f32);
        let r = nmf(x.view(), &NmfConfig::new(2)).unwrap();
        assert_eq!(r.w.dim(), (10, 2));
    }
}
