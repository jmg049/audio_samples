//! Dense linear algebra for the analysis modules: the symmetric eigenproblem,
//! singular value decomposition, the pseudo-inverse and non-negative least squares.
//!
//! # What
//!
//! - [`eigh`]: eigenvalues and eigenvectors of a symmetric matrix.
//! - [`svd`] and [`randomized_svd`]: the thin and the truncated singular value
//!   decomposition.
//! - [`pinv`]: the Moore-Penrose pseudo-inverse.
//! - [`nnls`]: the exact non-negative least squares solution for one right-hand side.
//! - [`nnls_matrix`]: non-negative least squares for many right-hand sides that share
//!   a matrix, starting from the projected pseudo-inverse solution.
//!
//! # Why
//!
//! Inverting a mel spectrogram and factorising a spectrogram need these operations,
//! and the crate does not depend on a linear algebra library.
//!
//! # How
//!
//! Functions take `ndarray` views and return owned arrays. Decompositions are computed
//! in `f64` and rounded once to the input's float type. Non-finite input is rejected
//! with a [`ParameterError`].
//!
//! ```
//! use audio_samples::operations::linalg::{nnls, pinv};
//! use ndarray::array;
//!
//! let a = array![[1.0_f64, 0.0], [0.0, 2.0], [0.0, 0.0]];
//! let x = nnls(a.view(), array![1.0, -2.0, 0.0].view()).unwrap();
//! assert_eq!(x.to_vec(), vec![1.0, 0.0]);
//!
//! let p = pinv(a.view(), 1e-15).unwrap();
//! assert!((p[[1, 1]] - 0.5).abs() < 1e-12);
//! ```
//!
//! References: G. H. Golub and C. F. Van Loan, *Matrix Computations*, 4th ed., 2013
//! (Householder tridiagonalisation and implicit QL); C. L. Lawson and R. J. Hanson,
//! *Solving Least Squares Problems*, 1974 (active-set non-negative least squares);
//! V. Franc, V. Hlaváč and M. Navara, *Sequential coordinate-wise algorithm for the
//! non-negative least squares problem*, CAIP 2005; N. Halko, P.-G. Martinsson and
//! J. A. Tropp, *Finding structure with randomness*, SIAM Review 53, 2011.

use ndarray::linalg::general_mat_mul;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};

use crate::{AudioFloat, AudioSampleResult, ParameterError};

fn invalid<T>(parameter: &str, reason: impl Into<String>) -> AudioSampleResult<T> {
    Err(ParameterError::invalid_value(parameter, reason.into()).into())
}

fn to_f64<F: AudioFloat>(a: ArrayView2<'_, F>, name: &str) -> AudioSampleResult<Array2<f64>> {
    if !a.fold(true, |ok, v| ok && v.is_finite()) {
        return invalid(name, "must be finite");
    }
    Ok(a.mapv(|v| v.to_f64().unwrap_or(f64::NAN)))
}

fn of<F: AudioFloat>(v: f64) -> F {
    <F as num_traits::NumCast>::from(v).unwrap_or_else(F::nan)
}

/// A small deterministic generator (PCG-XSH-RR, M. E. O'Neill, 2014).
#[derive(Debug, Clone)]
pub struct SmallRng {
    state: u64,
}

impl SmallRng {
    /// A generator with the given seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut rng = Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        };
        rng.next_u32();
        rng
    }

    /// The next 32 random bits.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let shifted = (((old >> 18) ^ old) >> 27) as u32;
        shifted.rotate_right((old >> 59) as u32)
    }

    /// A uniform value in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        let hi = u64::from(self.next_u32()) >> 5;
        let lo = u64::from(self.next_u32()) >> 6;
        (hi as f64 * 67_108_864.0 + lo as f64) / 9_007_199_254_740_992.0
    }

    /// A standard normal value (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u = 1.0 - self.uniform();
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// Eigenvalues and eigenvectors of a symmetric matrix.
///
/// Only the lower triangle of `a` is read.
///
/// # Arguments
/// - `a`: a symmetric `(n, n)` matrix.
///
/// # Returns
/// Eigenvalues in increasing order, and a matrix whose columns are the matching
/// orthonormal eigenvectors.
///
/// # Errors
/// [`ParameterError`] if `a` is not square or not finite, or if the iteration fails to
/// converge.
///
/// # Examples
/// ```
/// use audio_samples::operations::linalg::eigh;
/// use ndarray::array;
///
/// let (w, v) = eigh(array![[2.0, 0.0], [0.0, 1.0]].view()).unwrap();
/// assert_eq!(w.to_vec(), vec![1.0, 2.0]);
/// assert_eq!(v[[1, 0]].abs(), 1.0);
/// ```
pub fn eigh(a: ArrayView2<'_, f64>) -> AudioSampleResult<(Array1<f64>, Array2<f64>)> {
    let n = a.nrows();
    if a.ncols() != n {
        return invalid("a", "must be square");
    }
    if !a.fold(true, |ok, v| ok && v.is_finite()) {
        return invalid("a", "must be finite");
    }
    if n == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }
    let mut z = a.to_owned();
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];
    tridiagonalise(&mut z, &mut d, &mut e);
    implicit_ql(&mut d, &mut e, &mut z)?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| d[i].total_cmp(&d[j]));
    let values = order.iter().map(|&i| d[i]).collect();
    Ok((values, z.select(Axis(1), &order)))
}

/// Householder reduction to tridiagonal form, accumulating the transformation in `z`.
fn tridiagonalise(z: &mut Array2<f64>, d: &mut [f64], e: &mut [f64]) {
    let n = d.len();
    for i in (1..n).rev() {
        let mut h = 0.0;
        if i > 1 {
            let scale: f64 = (0..i).map(|k| z[[i, k]].abs()).sum();
            if scale == 0.0 {
                e[i] = z[[i, i - 1]];
            } else {
                for k in 0..i {
                    z[[i, k]] /= scale;
                    h += z[[i, k]] * z[[i, k]];
                }
                let f = z[[i, i - 1]];
                let g = if f >= 0.0 { -h.sqrt() } else { h.sqrt() };
                e[i] = scale * g;
                h -= f * g;
                z[[i, i - 1]] = f - g;
                let mut f_sum = 0.0;
                for j in 0..i {
                    z[[j, i]] = z[[i, j]] / h;
                    let mut g_val = 0.0;
                    for k in 0..=j {
                        g_val += z[[j, k]] * z[[i, k]];
                    }
                    for k in j + 1..i {
                        g_val += z[[k, j]] * z[[i, k]];
                    }
                    e[j] = g_val / h;
                    f_sum += e[j] * z[[i, j]];
                }
                let hh = f_sum / (h + h);
                for j in 0..i {
                    let f_j = z[[i, j]];
                    let g_j = e[j] - hh * f_j;
                    e[j] = g_j;
                    for k in 0..=j {
                        z[[j, k]] -= f_j * e[k] + g_j * z[[i, k]];
                    }
                }
            }
        } else {
            e[i] = z[[i, i - 1]];
        }
        d[i] = h;
    }
    d[0] = 0.0;
    e[0] = 0.0;
    for i in 0..n {
        if d[i] != 0.0 {
            for j in 0..i {
                let mut g = 0.0;
                for k in 0..i {
                    g += z[[i, k]] * z[[k, j]];
                }
                for k in 0..i {
                    z[[k, j]] -= g * z[[k, i]];
                }
            }
        }
        d[i] = z[[i, i]];
        z[[i, i]] = 1.0;
        for j in 0..i {
            z[[j, i]] = 0.0;
            z[[i, j]] = 0.0;
        }
    }
}

/// Implicit QL iteration on a tridiagonal matrix, rotating the columns of `z`.
fn implicit_ql(d: &mut [f64], e: &mut [f64], z: &mut Array2<f64>) -> AudioSampleResult<()> {
    let n = d.len();
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    for l in 0..n {
        let mut iterations = 0;
        loop {
            let mut m = l;
            while m + 1 < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= f64::EPSILON * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iterations += 1;
            if iterations > 60 {
                return Err(ParameterError::invalid_value(
                    "a",
                    "the QL iteration did not converge".to_string(),
                )
                .into());
            }
            let mut g = (d[l + 1] - d[l]) / (2.0 * e[l]);
            let mut r = g.hypot(1.0);
            g = d[m] - d[l] + e[l] / (g + if g >= 0.0 { r } else { -r });
            let (mut s_val, mut c, mut p) = (1.0, 1.0, 0.0);
            let mut underflow = false;
            for i in (l..m).rev() {
                let mut f = s_val * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == 0.0 {
                    d[i + 1] -= p;
                    e[m] = 0.0;
                    underflow = true;
                    break;
                }
                s_val = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s_val + 2.0 * c * b;
                p = s_val * r;
                d[i + 1] = g + p;
                g = c * r - b;
                for k in 0..n {
                    f = z[[k, i + 1]];
                    z[[k, i + 1]] = s_val * z[[k, i]] + c * f;
                    z[[k, i]] = c * z[[k, i]] - s_val * f;
                }
            }
            if underflow {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = 0.0;
        }
    }
    Ok(())
}

/// A thin singular value decomposition `a = u * diag(s) * vt`.
#[derive(Debug, Clone, PartialEq)]
pub struct Svd {
    /// Left singular vectors in the columns, `(m, k)`.
    pub u: Array2<f64>,
    /// Singular values in decreasing order, length `k = min(m, n)`.
    pub s: Array1<f64>,
    /// Right singular vectors in the rows, `(k, n)`.
    pub vt: Array2<f64>,
}

/// Thin singular value decomposition through the eigenproblem of the smaller Gram matrix.
///
/// Squaring the matrix limits relative accuracy: singular values below about
/// `1.5e-8` of the largest are reported as zero and their vectors are not meaningful.
///
/// # Arguments
/// - `a`: an `(m, n)` matrix.
///
/// # Returns
/// The decomposition with `k = min(m, n)`.
///
/// # Errors
/// [`ParameterError`] if `a` is not finite or the iteration fails to converge.
///
/// # Examples
/// ```
/// use audio_samples::operations::linalg::svd;
/// use ndarray::array;
///
/// let d = svd(array![[3.0, 0.0], [0.0, 4.0], [0.0, 0.0]].view()).unwrap();
/// assert!((d.s[0] - 4.0).abs() < 1e-12);
/// assert!((d.s[1] - 3.0).abs() < 1e-12);
/// ```
pub fn svd<F: AudioFloat>(a: ArrayView2<'_, F>) -> AudioSampleResult<Svd> {
    let a = to_f64(a, "a")?;
    let (m, n) = a.dim();
    let k = m.min(n);
    let wide = m < n;
    let gram = if wide { a.dot(&a.t()) } else { a.t().dot(&a) };
    let (values, vectors) = eigh(gram.view())?;
    let top = values[k.saturating_sub(1)].max(0.0);
    let floor = top * 2.0 * f64::EPSILON;
    let mut s = Array1::zeros(k);
    let mut small = Array2::zeros((vectors.nrows(), k));
    for j in 0..k {
        let from = k - 1 - j;
        let v = values[from];
        s[j] = if v > floor { v.sqrt() } else { 0.0 };
        small.column_mut(j).assign(&vectors.column(from));
    }
    let other = if wide {
        a.t().dot(&small)
    } else {
        a.dot(&small)
    };
    let mut large = other;
    for j in 0..k {
        let mut c = large.column_mut(j);
        if s[j] > 0.0 {
            c.mapv_inplace(|v| v / s[j]);
        } else {
            c.fill(0.0);
        }
    }
    Ok(if wide {
        Svd {
            u: small,
            s,
            vt: large.reversed_axes(),
        }
    } else {
        Svd {
            u: large,
            s,
            vt: small.reversed_axes(),
        }
    })
}

/// Orthonormalises the columns of `q` in place by `passes` rounds of modified Gram-Schmidt.
fn orthonormalise<F: AudioFloat>(q: &mut Array2<F>, passes: usize) {
    let cols = q.ncols();
    for _ in 0..passes {
        for j in 0..cols {
            let (done, mut rest) = q.view_mut().split_at(Axis(1), j);
            let mut col = rest.column_mut(0);
            for i in 0..j {
                let prev = done.column(i);
                let proj = prev
                    .iter()
                    .zip(col.iter())
                    .fold(F::zero(), |acc, (&p, &c)| acc + p * c);
                col.zip_mut_with(&prev, |c, &p| *c -= proj * p);
            }
            let norm = col.iter().fold(F::zero(), |acc, &c| acc + c * c).sqrt();
            if norm > F::zero() {
                col.mapv_inplace(|v| v / norm);
            }
        }
    }
}

/// The leading `k` singular triplets in `a`'s float type, found by randomised range
/// finding.
///
/// # Errors
/// [`ParameterError`] if `a` is not finite or `k` is out of range.
pub(crate) fn truncated_svd<F: AudioFloat>(
    a: ArrayView2<'_, F>,
    k: usize,
    oversampling: usize,
    n_iter: usize,
    seed: u64,
) -> AudioSampleResult<(Array2<F>, Vec<f64>, Array2<F>)> {
    let (m, n) = a.dim();
    if k == 0 || k > m.min(n) {
        return invalid("k", format!("must lie in 1..={}, got {k}", m.min(n)));
    }
    if !a.fold(true, |ok, v| ok && v.is_finite()) {
        return invalid("a", "must be finite");
    }
    let size = (k + oversampling).min(m.min(n));
    let mut rng = SmallRng::new(seed);
    let omega = Array2::from_shape_simple_fn((n, size), || of::<F>(rng.normal()));
    let mut q = a.dot(&omega);
    drop(omega);
    orthonormalise(&mut q, 1);
    // One round keeps the power iterations stable; the last result gets a second one.
    let mut z = Array2::<F>::zeros((n, size));
    for _ in 0..n_iter {
        general_mat_mul(F::one(), &a.t(), &q, F::zero(), &mut z);
        orthonormalise(&mut z, 1);
        general_mat_mul(F::one(), &a, &z, F::zero(), &mut q);
        orthonormalise(&mut q, 1);
    }
    drop(z);
    orthonormalise(&mut q, 1);
    let b = q.t().dot(&a);
    let small = svd(b.view())?;
    drop(b);
    let u = q.dot(&small.u.slice(s![.., ..k]).mapv(of::<F>));
    let vt = small.vt.slice(s![..k, ..]).mapv(of::<F>);
    Ok((u, small.s.iter().take(k).copied().collect(), vt))
}

/// Truncated singular value decomposition by randomised range finding.
///
/// # Arguments
/// - `a`: an `(m, n)` matrix.
/// - `k`: the number of singular triplets to keep; at most `min(m, n)`.
/// - `oversampling`: extra basis vectors for the range finder.
/// - `n_iter`: power iterations.
/// - `seed`: seeds the random test matrix.
///
/// # Returns
/// The leading `k` triplets, deterministic for a given seed.
///
/// # Errors
/// [`ParameterError`] if `a` is not finite or `k` is out of range.
///
/// # Examples
/// ```
/// use audio_samples::operations::linalg::randomized_svd;
/// use ndarray::array;
///
/// let a = array![[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.0]];
/// let d = randomized_svd(a.view(), 1, 2, 4, 0).unwrap();
/// assert!((d.s[0] - 3.0).abs() < 1e-9);
/// ```
pub fn randomized_svd<F: AudioFloat>(
    a: ArrayView2<'_, F>,
    k: usize,
    oversampling: usize,
    n_iter: usize,
    seed: u64,
) -> AudioSampleResult<Svd> {
    let (u, s_values, vt) = truncated_svd(a, k, oversampling, n_iter, seed)?;
    let widen = |m: Array2<F>| m.mapv(|v| v.to_f64().unwrap_or(f64::NAN));
    Ok(Svd {
        u: widen(u),
        s: Array1::from(s_values),
        vt: widen(vt),
    })
}

/// The Moore-Penrose pseudo-inverse.
///
/// # Arguments
/// - `a`: an `(m, n)` matrix.
/// - `rcond`: singular values below `rcond` times the largest are treated as zero.
///   Values below about `1.5e-8` act as that limit (see [`svd`]).
///
/// # Returns
/// The `(n, m)` pseudo-inverse.
///
/// # Errors
/// [`ParameterError`] if `a` is not finite or `rcond` is negative.
///
/// # Examples
/// ```
/// use audio_samples::operations::linalg::pinv;
/// use ndarray::array;
///
/// let p = pinv(array![[2.0_f64, 0.0]].view(), 1e-15).unwrap();
/// assert!((p[[0, 0]] - 0.5).abs() < 1e-12);
/// assert_eq!(p[[1, 0]], 0.0);
/// ```
pub fn pinv<F: AudioFloat>(a: ArrayView2<'_, F>, rcond: f64) -> AudioSampleResult<Array2<F>> {
    if rcond.is_nan() || rcond < 0.0 {
        return invalid("rcond", "must not be negative");
    }
    let d = svd(a)?;
    let top = d.s.first().copied().unwrap_or(0.0);
    let mut scaled = d.vt.t().to_owned();
    for (j, &sv) in d.s.iter().enumerate() {
        let scale = if sv > rcond * top && sv > 0.0 {
            1.0 / sv
        } else {
            0.0
        };
        scaled.column_mut(j).mapv_inplace(|v| v * scale);
    }
    Ok(scaled.dot(&d.u.t()).mapv(of::<F>))
}

/// Solves `min ||a x - b||` subject to `x >= 0` for one right-hand side.
///
/// The solution is exact (the active-set algorithm of Lawson and Hanson). When `a` has
/// more columns than rows the solution is one of many, with few non-zero entries.
///
/// # Arguments
/// - `a`: an `(m, n)` matrix.
/// - `b`: the right-hand side, length `m`.
///
/// # Returns
/// The solution `x`, length `n`.
///
/// # Errors
/// [`ParameterError`] if a dimension disagrees or an input is not finite.
///
/// # Examples
/// ```
/// use audio_samples::operations::linalg::nnls;
/// use ndarray::array;
///
/// let x = nnls(array![[1.0, 1.0], [0.0, 1.0]].view(), array![1.0, -1.0].view()).unwrap();
/// assert_eq!(x.to_vec(), vec![1.0, 0.0]);
/// ```
pub fn nnls<F: AudioFloat>(
    a: ArrayView2<'_, F>,
    b: ArrayView1<'_, F>,
) -> AudioSampleResult<Array1<F>> {
    let m = a.nrows();
    if b.len() != m {
        return invalid("b", format!("length {} for a matrix of {m} rows", b.len()));
    }
    if b.iter().any(|v| !v.is_finite()) {
        return invalid("b", "must be finite");
    }
    if !a.fold(true, |ok, v| ok && v.is_finite()) {
        return invalid("a", "must be finite");
    }
    let b: Array1<f64> = b.mapv(|v| v.to_f64().unwrap_or(f64::NAN));
    let strides = a.strides();
    if std::any::TypeId::of::<F>() == std::any::TypeId::of::<f64>()
        && strides.iter().all(|&v| v >= 0)
    {
        // SAFETY: F is f64 (the same type, so the same layout); the shape and the
        // non-negative strides are those of `a`, whose elements outlive the call.
        let view = unsafe {
            ArrayView2::from_shape_ptr(
                ndarray::ShapeBuilder::strides(
                    a.raw_dim(),
                    ndarray::Ix2(strides[0] as usize, strides[1] as usize),
                ),
                a.as_ptr().cast::<f64>(),
            )
        };
        return Ok(active_set(view, &b).mapv(of::<F>));
    }
    let copy = a.mapv(|v| v.to_f64().unwrap_or(f64::NAN));
    Ok(active_set(copy.view(), &b).mapv(of::<F>))
}

/// The lower Cholesky factor of the Gram matrix of the passive columns, in packed rows,
/// grown one column at a time.
struct PassiveFactor {
    passive: Vec<usize>,
    l: Vec<f64>,
}

impl PassiveFactor {
    fn with_capacity(columns: usize) -> Self {
        Self {
            passive: Vec::with_capacity(columns),
            l: Vec::with_capacity(columns * (columns + 1) / 2),
        }
    }

    /// Adds column `t`. Returns false if it is numerically dependent on the others.
    fn push(&mut self, gram: &Array2<f64>, t: usize) -> bool {
        let p = self.passive.len();
        let start = self.l.len();
        self.l.resize(start + p + 1, 0.0);
        let mut offset = 0;
        for i in 0..p {
            let mut v = gram[[self.passive[i], t]];
            for k in 0..i {
                v -= self.l[offset + k] * self.l[start + k];
            }
            self.l[start + i] = v / self.l[offset + i];
            offset += i + 1;
        }
        let diag = gram[[t, t]];
        let d2 = diag - self.l[start..start + p].iter().map(|v| v * v).sum::<f64>();
        if d2.is_nan() || d2 <= diag * 1e-14 {
            self.l.truncate(start);
            return false;
        }
        self.l[start + p] = d2.sqrt();
        self.passive.push(t);
        true
    }

    fn rebuild(&mut self, gram: &Array2<f64>, keep: &[usize]) {
        self.passive.clear();
        self.l.clear();
        for &t in keep {
            self.push(gram, t);
        }
    }

    /// Solves the normal equations for the right-hand side `rhs` of the passive set.
    fn solve(&self, rhs: &mut [f64]) {
        let p = self.passive.len();
        let mut offset = 0;
        for i in 0..p {
            let mut v = rhs[i];
            for (l, r) in self.l[offset..offset + i].iter().zip(rhs.iter()) {
                v -= l * r;
            }
            rhs[i] = v / self.l[offset + i];
            offset += i + 1;
        }
        // Back substitution walks the packed rows in reverse, column by column.
        for i in (0..p).rev() {
            offset -= i + 1;
            let v = rhs[i] / self.l[offset + i];
            rhs[i] = v;
            for (l, r) in self.l[offset..offset + i].iter().zip(rhs.iter_mut()) {
                *r -= l * v;
            }
        }
    }
}

fn active_set(a: ArrayView2<'_, f64>, b: &Array1<f64>) -> Array1<f64> {
    let (m, n) = a.dim();
    let capacity = m.min(n);
    let mut x = Array1::<f64>::zeros(n);
    let mut factor = PassiveFactor::with_capacity(capacity);
    // 1 for a column in the passive set, 2 for one rejected until the set next changes.
    let mut state = vec![0_u8; n];
    let mut at_b = Array1::<f64>::zeros(n);
    let mut gram = Array2::<f64>::zeros((n, n));
    // One row at a time, the upper triangle only; the lower is mirrored below.
    if let (Some(g), Some(atb)) = (gram.as_slice_mut(), at_b.as_slice_mut()) {
        let mut scratch = vec![0.0; n];
        for (row, &bi) in a.rows().into_iter().zip(b.iter()) {
            let values = row.as_slice().unwrap_or_else(|| {
                for (o, &v) in scratch.iter_mut().zip(row.iter()) {
                    *o = v;
                }
                &scratch
            });
            for (o, &v) in atb.iter_mut().zip(values) {
                *o += bi * v;
            }
            for (p, &rp) in values.iter().enumerate() {
                if rp != 0.0 {
                    let target = &mut g[p * n + p..(p + 1) * n];
                    for (t, &v) in target.iter_mut().zip(&values[p..]) {
                        *t += rp * v;
                    }
                }
            }
        }
    }
    for p in 0..n {
        for q in 0..p {
            gram[[p, q]] = gram[[q, p]];
        }
    }
    let tol = {
        let mut norm1 = vec![0.0_f64; n];
        for row in a.rows() {
            for (s, v) in norm1.iter_mut().zip(row.iter()) {
                *s += v.abs();
            }
        }
        let largest = norm1.iter().copied().fold(0.0, f64::max);
        10.0 * m.max(n) as f64 * f64::EPSILON * largest.max(f64::MIN_POSITIVE)
    };
    // The gradient follows from the Gram matrix, without the residual.
    let mut w = at_b.clone();
    let mut s: Vec<f64> = Vec::with_capacity(capacity);
    let max_iter = 3 * n.max(1);
    let mut iter = 0;
    let solve_passive = |factor: &PassiveFactor, s: &mut Vec<f64>| {
        s.clear();
        s.extend(factor.passive.iter().map(|&t| at_b[t]));
        factor.solve(s);
    };
    while factor.passive.len() < capacity && iter < max_iter {
        iter += 1;
        let mut best: Option<(usize, f64)> = None;
        for (t, (&w_t, &flag)) in w.iter().zip(&state).enumerate() {
            if flag == 0 && w_t > tol && best.is_none_or(|(_, b)| w_t > b) {
                best = Some((t, w_t));
            }
        }
        let Some((t, _)) = best else {
            break;
        };
        if !factor.push(&gram, t) {
            state[t] = 2;
            continue;
        }
        state[t] = 1;
        for flag in &mut state {
            if *flag == 2 {
                *flag = 0;
            }
        }
        loop {
            solve_passive(&factor, &mut s);
            if s.iter().all(|&v| v > 0.0) {
                for (i, &j) in factor.passive.iter().enumerate() {
                    x[j] = s[i];
                }
                break;
            }
            let mut alpha = f64::INFINITY;
            for (i, &j) in factor.passive.iter().enumerate() {
                if s[i] <= 0.0 {
                    alpha = alpha.min(x[j] / (x[j] - s[i]));
                }
            }
            let keep: Vec<usize> = factor
                .passive
                .iter()
                .enumerate()
                .filter_map(|(i, &j)| {
                    x[j] += alpha * (s[i] - x[j]);
                    if x[j] > tol * f64::EPSILON && s[i] > 0.0 {
                        Some(j)
                    } else {
                        x[j] = 0.0;
                        state[j] = 0;
                        None
                    }
                })
                .collect();
            factor.rebuild(&gram, &keep);
            if factor.passive.is_empty() {
                break;
            }
        }
        w.assign(&at_b);
        if let (Some(left), Some(g)) = (w.as_slice_mut(), gram.as_slice()) {
            for &j in &factor.passive {
                let weight = x[j];
                for (o, &v) in left.iter_mut().zip(&g[j * n..(j + 1) * n]) {
                    *o -= weight * v;
                }
            }
        }
    }
    // One step of iterative refinement recovers what the normal equations lose.
    if !factor.passive.is_empty() {
        let mut residual = b.clone();
        for &j in &factor.passive {
            residual.scaled_add(-x[j], &a.column(j));
        }
        let mut d: Vec<f64> = factor
            .passive
            .iter()
            .map(|&t| a.column(t).dot(&residual))
            .collect();
        factor.solve(&mut d);
        for (&j, v) in factor.passive.iter().zip(&d) {
            let refined = x[j] + v;
            if refined > 0.0 {
                x[j] = refined;
            }
        }
    }
    x
}

/// Stopping rule of [`nnls_matrix`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NnlsConfig {
    /// Most sweeps over the unknowns of one right-hand side. Default 200.
    pub max_iter: usize,
    /// Stop a right-hand side when no sweep changes `a x` by more than this fraction of
    /// `max |b|`. Default `1e-5`.
    pub tol: f64,
}

impl NnlsConfig {
    /// 200 sweeps and a tolerance of `1e-5`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-5,
        }
    }

    /// The configuration used for comparison with librosa; currently equal to
    /// [`new`](Self::new).
    #[must_use]
    pub const fn librosa() -> Self {
        Self::new()
    }

    /// Sets the most sweeps per right-hand side.
    #[must_use]
    pub const fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the stopping tolerance.
    #[must_use]
    pub const fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    /// [`ParameterError`] if `tol` is negative or not finite.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if !self.tol.is_finite() || self.tol < 0.0 {
            return invalid("tol", "must be finite and non-negative");
        }
        Ok(())
    }
}

impl Default for NnlsConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Solves `min ||a x - b||` subject to `x >= 0` for each column of `b`.
///
/// Each column starts from the pseudo-inverse solution with negative entries set to
/// zero, and is refined by cyclic coordinate descent on the objective. When `a` has
/// more columns than rows, this selects a solution near the minimum-norm one, smooth
/// where the exact active-set vertex of [`nnls`] would be spiky. The stored non-zeros
/// of `a` are what each step touches, so a sparse `a`, such as a filterbank, is cheap.
///
/// # Arguments
/// - `a`: an `(m, n)` matrix.
/// - `b`: `(m, t)` right-hand sides in the columns.
/// - `config`: the stopping rule.
///
/// # Returns
/// The `(n, t)` solution.
///
/// # Errors
/// [`ParameterError`] if a dimension disagrees or an input is not finite.
///
/// # Examples
/// ```
/// use audio_samples::operations::linalg::{NnlsConfig, nnls_matrix};
/// use ndarray::array;
///
/// let a = array![[1.0_f64, 1.0, 0.0], [0.0, 1.0, 1.0]];
/// let b = array![[2.0], [2.0]];
/// let x = nnls_matrix(a.view(), b.view(), &NnlsConfig::new()).unwrap();
/// let fit = a.dot(&x);
/// assert!((fit[[0, 0]] - 2.0).abs() < 1e-3);
/// assert!(x.iter().all(|&v| v >= 0.0));
/// ```
pub fn nnls_matrix<F: AudioFloat>(
    a: ArrayView2<'_, F>,
    b: ArrayView2<'_, F>,
    config: &NnlsConfig,
) -> AudioSampleResult<Array2<F>> {
    NnlsSolver::new(a, config)?.solve(b)
}

/// [`nnls_matrix`] with the work that depends only on the matrix kept for several
/// right-hand side matrices.
#[derive(Debug, Clone)]
pub struct NnlsSolver<F> {
    pseudo_inverse: Array2<F>,
    pointers: Vec<usize>,
    rows: Vec<u32>,
    values: Vec<F>,
    norms: Vec<F>,
    config: NnlsConfig,
}

impl<F: AudioFloat> NnlsSolver<F> {
    /// Prepares the pseudo-inverse and the compressed columns of `a`.
    ///
    /// # Arguments
    /// - `a`: an `(m, n)` matrix.
    /// - `config`: the stopping rule.
    ///
    /// # Errors
    /// [`ParameterError`] if `a` is not finite or the configuration is invalid.
    pub fn new(a: ArrayView2<'_, F>, config: &NnlsConfig) -> AudioSampleResult<Self> {
        config.validate()?;
        let n = a.ncols();
        let pseudo_inverse = pinv(a, 1e-15)?;
        let mut pointers = Vec::with_capacity(n + 1);
        let mut rows: Vec<u32> = Vec::new();
        let mut values: Vec<F> = Vec::new();
        let mut norms = vec![F::zero(); n];
        pointers.push(0_usize);
        for (j, col) in a.columns().into_iter().enumerate() {
            for (i, &v) in col.iter().enumerate() {
                if v != F::zero() {
                    rows.push(i as u32);
                    values.push(v);
                    norms[j] += v * v;
                }
            }
            pointers.push(rows.len());
        }
        Ok(Self {
            pseudo_inverse,
            pointers,
            rows,
            values,
            norms,
            config: *config,
        })
    }

    /// Solves for each column of `b`.
    ///
    /// # Arguments
    /// - `b`: `(m, t)` right-hand sides in the columns.
    ///
    /// # Returns
    /// The `(n, t)` solution.
    ///
    /// # Errors
    /// [`ParameterError`] if `b` has the wrong number of rows or is not finite.
    pub fn solve(&self, b: ArrayView2<'_, F>) -> AudioSampleResult<Array2<F>> {
        let m = self.pseudo_inverse.ncols();
        let n = self.norms.len();
        if b.nrows() != m {
            return invalid("b", format!("{} rows for a matrix of {m} rows", b.nrows()));
        }
        if b.iter().any(|v| !v.is_finite()) {
            return invalid("b", "must be finite");
        }
        let (pointers, rows, values, norms) =
            (&self.pointers, &self.rows, &self.values, &self.norms);
        let mut x = self.pseudo_inverse.dot(&b);
        x.mapv_inplace(|v| if v > F::zero() { v } else { F::zero() });
        let tol = of::<F>(self.config.tol);
        let mut residual = vec![F::zero(); m];
        let peak_all = b.iter().fold(F::zero(), |p, &v| p.max(v.abs()));
        let floor = F::min_positive_value() + peak_all * F::epsilon();
        for c in 0..b.ncols() {
            let target = b.column(c);
            let mut xc = x.column_mut(c);
            for (r, &v) in residual.iter_mut().zip(target.iter()) {
                *r = v;
            }
            let mut peak = F::zero();
            for &v in target.iter() {
                peak = peak.max(v.abs());
            }
            for (j, &v) in xc.iter().enumerate() {
                if v != F::zero() {
                    for k in pointers[j]..pointers[j + 1] {
                        let r = rows[k] as usize;
                        residual[r] -= values[k] * v;
                    }
                }
            }
            let limit = (tol * peak).max(floor);
            for _ in 0..self.config.max_iter {
                let mut largest = F::zero();
                for j in 0..n {
                    if norms[j] == F::zero() {
                        continue;
                    }
                    let mut grad = F::zero();
                    for k in pointers[j]..pointers[j + 1] {
                        grad += values[k] * residual[rows[k] as usize];
                    }
                    let old = xc[j];
                    let new = old + grad / norms[j];
                    let new = if new > F::zero() { new } else { F::zero() };
                    let delta = new - old;
                    if delta != F::zero() {
                        xc[j] = new;
                        for k in pointers[j]..pointers[j + 1] {
                            let r = rows[k] as usize;
                            residual[r] -= values[k] * delta;
                        }
                        let change = delta.abs() * norms[j].sqrt();
                        if change > largest {
                            largest = change;
                        }
                    }
                }
                if largest <= limit {
                    break;
                }
            }
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn random(m: usize, n: usize, seed: u64) -> Array2<f64> {
        let mut rng = SmallRng::new(seed);
        Array2::from_shape_simple_fn((m, n), || rng.normal())
    }

    #[test]
    fn eigh_reconstructs() {
        let r = random(30, 30, 1);
        let a = &r + &r.t();
        let (w, v) = eigh(a.view()).unwrap();
        let back = v.dot(&Array2::from_diag(&w)).dot(&v.t());
        for (x, y) in back.iter().zip(a.iter()) {
            assert!((x - y).abs() < 1e-11);
        }
        assert!(w.windows(2).into_iter().all(|p| p[0] <= p[1]));
        let id = v.t().dot(&v);
        for i in 0..30 {
            assert!((id[[i, i]] - 1.0).abs() < 1e-12);
        }
        assert!(eigh(random(2, 3, 1).view()).is_err());
    }

    #[test]
    fn svd_reconstructs_both_shapes() {
        for (m, n) in [(40, 12), (12, 40), (9, 9)] {
            let a = random(m, n, 2);
            let d = svd(a.view()).unwrap();
            let back = d.u.dot(&Array2::from_diag(&d.s)).dot(&d.vt);
            for (x, y) in back.iter().zip(a.iter()) {
                assert!((x - y).abs() < 1e-10, "{m}x{n}");
            }
        }
    }

    #[test]
    fn randomized_svd_finds_the_leading_triplets() {
        let a = random(100, 40, 3);
        let exact = svd(a.view()).unwrap();
        let r = randomized_svd(a.view(), 5, 10, 6, 0).unwrap();
        for i in 0..5 {
            assert!((r.s[i] - exact.s[i]).abs() < 1e-2 * exact.s[i]);
        }
        assert!(randomized_svd(a.view(), 0, 10, 6, 0).is_err());
    }

    #[test]
    fn pinv_is_a_generalised_inverse() {
        let a = random(10, 25, 4);
        let p = pinv(a.view(), 1e-15).unwrap();
        let back = a.dot(&p).dot(&a);
        for (x, y) in back.iter().zip(a.iter()) {
            assert!((x - y).abs() < 1e-10);
        }
        assert!(pinv(a.view(), -1.0).is_err());
    }

    #[test]
    fn nnls_satisfies_the_optimality_conditions() {
        let a = random(40, 15, 5);
        let b = random(40, 1, 6).column(0).to_owned();
        let x = nnls(a.view(), b.view()).unwrap();
        let w = a.t().dot(&(&b - &a.dot(&x)));
        for (xi, wi) in x.iter().zip(w.iter()) {
            assert!(*xi >= 0.0);
            assert!(*wi < 1e-9);
            if *xi > 0.0 {
                assert!(wi.abs() < 1e-9);
            }
        }
    }

    #[test]
    fn nnls_handles_dependent_columns_and_zero_input() {
        let a = array![[1.0_f64, 1.0, 2.0], [1.0, 1.0, 2.0]];
        let x = nnls(a.view(), array![2.0, 2.0].view()).unwrap();
        let fit = a.dot(&x);
        assert!((fit[0] - 2.0).abs() < 1e-12);
        let z = nnls(a.view(), array![0.0, 0.0].view()).unwrap();
        assert_eq!(z.to_vec(), vec![0.0, 0.0, 0.0]);
        assert!(nnls(a.view(), array![1.0].view()).is_err());
        assert!(nnls(a.view(), array![f64::NAN, 1.0].view()).is_err());
    }

    #[test]
    fn nnls_matrix_is_not_worse_than_its_start_and_stays_non_negative() {
        let a = random(10, 30, 7).mapv(f64::abs);
        let x_true = random(30, 6, 8).mapv(f64::abs);
        let b = a.dot(&x_true);
        let x = nnls_matrix(a.view(), b.view(), &NnlsConfig::new()).unwrap();
        let start = pinv(a.view(), 1e-15).unwrap().dot(&b).mapv(|v| v.max(0.0));
        let err = |e: &Array2<f64>| (&a.dot(e) - &b).mapv(|v| v * v).sum();
        assert!(err(&x) <= err(&start) + 1e-12);
        assert!(err(&x) < 1e-3 * b.mapv(|v| v * v).sum());
        assert!(x.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn single_precision_matches_double_precision() {
        let a = random(10, 30, 9).mapv(f64::abs);
        let b = random(10, 3, 10).mapv(f64::abs);
        let x64 = nnls_matrix(a.view(), b.view(), &NnlsConfig::new()).unwrap();
        let a32 = a.mapv(|v| v as f32);
        let b32 = b.mapv(|v| v as f32);
        let x32 = nnls_matrix(a32.view(), b32.view(), &NnlsConfig::new()).unwrap();
        let fit64 = a.dot(&x64);
        let fit32 = a32.dot(&x32);
        for (p, q) in fit64.iter().zip(fit32.iter()) {
            assert!((p - f64::from(*q)).abs() < 1e-2);
        }
    }
}
