//! Sequence alignment, decoding and self-similarity over feature matrices.
//!
//! # What
//!
//! This module collects the dynamic-programming and structure-analysis tools
//! that operate on *sequences of feature vectors* (chroma, MFCC, onset
//! envelopes, class probabilities, ...) rather than on audio directly:
//!
//! - **Alignment**: [`dtw`] / [`dtw_with_cost`] (dynamic time warping) and
//!   [`dtw_backtracking`], plus [`rqa`] (recurrence quantification analysis,
//!   local alignment on a similarity matrix).
//! - **Decoding**: [`viterbi`], [`viterbi_discriminative`] and
//!   [`viterbi_binary`] (hidden Markov model decoding), with the transition
//!   matrix builders [`transition_uniform`], [`transition_loop`],
//!   [`transition_cycle`] and [`transition_local`].
//! - **Self-similarity**: [`recurrence_matrix`], [`cross_similarity`],
//!   [`recurrence_to_lag`] and [`lag_to_recurrence`].
//! - **Aggregation**: [`sync`] / [`sync_with`] (e.g. beat-synchronous
//!   features).
//!
//! # Why
//!
//! Many music-analysis tasks reduce to "find the best path through a matrix":
//! aligning two performances, smoothing frame-wise pitch or chord estimates,
//! or finding repeated sections. These functions reproduce
//! `librosa.sequence`, `librosa.segment` and `librosa.util.sync`
//! (librosa 1.0.0) so that librosa pipelines can be ported one-to-one; every
//! config's `Default` equals librosa's defaults.
//!
//! # How
//!
//! All functions are free functions over `ndarray` arrays. Feature matrices
//! follow librosa's layout `(n_features, n_frames)`: one column per frame.
//! Probability matrices are `(n_states, n_steps)`. Results are reported through
//! [`AudioSampleResult`]; invalid input yields a [`ParameterError`], never a
//! panic.
//!
//! The functions are generic over the element type ([`AudioFloat`]: `f32` or
//! `f64`). The computation runs in the element type of the input matrices and
//! the float outputs (costs, scores, log-likelihoods, similarity matrices) have
//! that type, so `f32` features (for example from `f32` or `i16` audio) are
//! processed without conversion. Index outputs (paths, state sequences, step
//! matrices) are `usize`. Parameters in the config structs stay `f64` and are
//! rounded to the element type once. The transition-matrix builders
//! ([`transition_loop`], ...) compute in `f64` and round each entry to the
//! requested type once.
//!
//! ```
//! use audio_samples::operations::sequence::{dtw, DtwConfig};
//! use ndarray::array;
//!
//! // Two 1-D feature sequences (1 feature x n frames); `y` repeats a frame.
//! let x = array![[0.0, 1.0, 2.0, 3.0]];
//! let y = array![[0.0, 1.0, 1.0, 2.0, 3.0]];
//! let result = dtw(x.view(), y.view(), &DtwConfig::default()).unwrap();
//!
//! // A perfect alignment exists, so the total cost is zero.
//! assert_eq!(result.cost[[3, 4]], 0.0);
//! // The path runs from the end back to the start, as in librosa.
//! let path = result.path.unwrap();
//! assert_eq!(path.first(), Some(&[3, 4]));
//! assert_eq!(path.last(), Some(&[0, 0]));
//! ```
//!
//! # Differences from librosa
//!
//! - Only 2-D inputs are accepted (librosa also broadcasts over leading
//!   axes); loop over channels yourself.
//! - `recurrence_matrix` / `cross_similarity` return dense matrices only
//!   (no `sparse=True`), and connectivity matrices are floats (`0.0`/`1.0`)
//!   rather than `bool`. Only librosa's default `axis=-1` layout is supported.
//! - `sync` with an empty index list and `pad = true` returns one segment
//!   (librosa returns none).
//! - `rqa` follows the move actually taken when backtracking through the
//!   second column (librosa stores a wrong pointer there).
//! - With `f32` input, the Viterbi decoders accumulate log-likelihoods in
//!   `f32`; librosa accumulates them in `float64` for every input type.
//!   DTW costs of `f32` features are computed in `f32`; librosa computes
//!   them with SciPy's `float64` `cdist`.
//! - `path_enhance`, `timelag_filter`, `agglomerative`, `subsegment` and
//!   `path_to_steps` are not provided.

use ndarray::{
    Array, Array1, Array2, ArrayView, ArrayView1, ArrayView2, ArrayView3, Axis, Dimension,
    RemoveAxis, Slice, Zip,
};

use crate::{AudioFloat, AudioSampleResult, ParameterError};

/// `numpy.finfo(dtype).tiny` (the smallest positive normal value of `F`), the
/// clipping constant librosa adds before logs (`util.tiny(prob)`).
fn tiny<F: AudioFloat>() -> F {
    F::min_positive_value()
}

/// `numpy.allclose` for scalars with its default tolerances.
fn allclose(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
}

/// The sum of `x` in `f64` (for validation: rows of `f32` matrices are
/// checked against `numpy.allclose` without `f32` summation error).
fn sum_f64<'a, F: AudioFloat>(x: impl IntoIterator<Item = &'a F>) -> f64 {
    x.into_iter().map(|v| v.as_f64()).sum()
}

fn err<T>(parameter: &str, reason: impl ToString) -> AudioSampleResult<T> {
    Err(ParameterError::invalid_value(parameter, reason).into())
}

// ---------------------------------------------------------------------------
// Distances
// ---------------------------------------------------------------------------

/// Distance between two feature vectors.
///
/// Used by [`dtw`] to build the local cost matrix and by
/// [`recurrence_matrix`] / [`cross_similarity`] for nearest-neighbour search.
/// The names follow `scipy.spatial.distance` (which librosa passes them to).
///
/// The default is [`DistanceMetric::Euclidean`], librosa's default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    /// `sqrt(sum((u - v)^2))`.
    #[default]
    Euclidean,
    /// `sum((u - v)^2)`.
    SqEuclidean,
    /// `1 - u.v / (|u| |v|)`. Undefined (NaN) when either vector is all zeros,
    /// which makes [`dtw`] return an error (as librosa does).
    Cosine,
    /// `sum(|u - v|)` (Manhattan / L1).
    Cityblock,
    /// `max(|u - v|)` (L-infinity).
    Chebyshev,
}

impl DistanceMetric {
    /// Distance between two equally long vectors.
    ///
    /// # Arguments
    ///
    /// - `u`, `v` - Feature vectors of the same length.
    ///
    /// # Returns
    ///
    /// The distance under this metric, in the element type `F` of the vectors.
    /// For [`DistanceMetric::Cosine`] the result is NaN if either vector has
    /// zero norm.
    ///
    /// # Panics
    ///
    /// Panics if `u` and `v` have different lengths (programmer error).
    ///
    /// # Examples
    ///
    /// ```
    /// use audio_samples::operations::sequence::DistanceMetric;
    /// use ndarray::array;
    ///
    /// let u = array![0.0, 3.0];
    /// let v = array![4.0, 0.0];
    /// assert_eq!(DistanceMetric::Euclidean.distance(u.view(), v.view()), 5.0);
    /// assert_eq!(DistanceMetric::Cityblock.distance(u.view(), v.view()), 7.0);
    /// assert_eq!(DistanceMetric::Chebyshev.distance(u.view(), v.view()), 4.0);
    /// ```
    #[must_use]
    pub fn distance<F: AudioFloat>(self, u: ArrayView1<F>, v: ArrayView1<F>) -> F {
        assert_eq!(u.len(), v.len(), "vectors must have equal length");
        let zero = F::zero();
        match self {
            Self::Euclidean => Zip::from(&u)
                .and(&v)
                .fold(zero, |acc, &a, &b| acc + (a - b) * (a - b))
                .sqrt(),
            Self::SqEuclidean => Zip::from(&u)
                .and(&v)
                .fold(zero, |acc, &a, &b| acc + (a - b) * (a - b)),
            Self::Cityblock => Zip::from(&u)
                .and(&v)
                .fold(zero, |acc, &a, &b| acc + (a - b).abs()),
            Self::Chebyshev => Zip::from(&u)
                .and(&v)
                .fold(zero, |acc: F, &a, &b| acc.max((a - b).abs())),
            Self::Cosine => {
                let (mut dot, mut nu, mut nv) = (zero, zero, zero);
                Zip::from(&u).and(&v).for_each(|&a, &b| {
                    dot += a * b;
                    nu += a * a;
                    nv += b * b;
                });
                F::one() - dot / (nu.sqrt() * nv.sqrt())
            }
        }
    }
}

/// Distances from single feature vectors to every column of `y`, one row of
/// a `scipy.spatial.distance.cdist` result at a time.
///
/// Each distance is accumulated feature by feature in the same order as
/// [`DistanceMetric::distance`] (so the results are identical), but across
/// all columns of `y` at once, which vectorises.
struct RowDistances<'a, F: AudioFloat> {
    y: ndarray::CowArray<'a, F, ndarray::Ix2>,
    metric: DistanceMetric,
    /// Squared norms of the columns of `y` (cosine only), summed as in
    /// `distance`.
    nv: Vec<F>,
}

impl<'a, F: AudioFloat> RowDistances<'a, F> {
    fn new(y: ArrayView2<'a, F>, metric: DistanceMetric) -> Self {
        let y = if y.is_standard_layout() {
            ndarray::CowArray::from(y)
        } else {
            ndarray::CowArray::from(y.as_standard_layout().into_owned())
        };
        let nv = if metric == DistanceMetric::Cosine {
            y.columns()
                .into_iter()
                .map(|v| v.iter().fold(F::zero(), |acc, &b| acc + b * b))
                .collect()
        } else {
            Vec::new()
        };
        Self { y, metric, nv }
    }

    /// Writes the distances from `xi` to each column of `y` into `row`.
    fn row(&self, xi: ArrayView1<F>, row: &mut [F]) {
        let ny = self.y.ncols();
        row.fill(F::zero());
        if ny == 0 {
            return;
        }
        let ys = self.y.as_slice().expect("standard layout");
        let metric = self.metric;
        for (&a, yf) in xi.iter().zip(ys.chunks_exact(ny)) {
            match metric {
                DistanceMetric::Euclidean | DistanceMetric::SqEuclidean => {
                    for (o, &b) in row.iter_mut().zip(yf) {
                        *o += (a - b) * (a - b);
                    }
                }
                DistanceMetric::Cityblock => {
                    for (o, &b) in row.iter_mut().zip(yf) {
                        *o += (a - b).abs();
                    }
                }
                DistanceMetric::Chebyshev => {
                    for (o, &b) in row.iter_mut().zip(yf) {
                        *o = o.max((a - b).abs());
                    }
                }
                DistanceMetric::Cosine => {
                    for (o, &b) in row.iter_mut().zip(yf) {
                        *o += a * b;
                    }
                }
            }
        }
        match metric {
            DistanceMetric::Euclidean => row.iter_mut().for_each(|o| *o = o.sqrt()),
            DistanceMetric::Cosine => {
                let nu = xi.iter().fold(F::zero(), |acc, &a| acc + a * a);
                for (o, &nv) in row.iter_mut().zip(&self.nv) {
                    *o = F::one() - *o / (nu.sqrt() * nv.sqrt());
                }
            }
            _ => {}
        }
    }
}

/// Pairwise distances between the columns of `x` (rows of the result) and the
/// columns of `y` (columns of the result), like `scipy.spatial.distance.cdist`.
fn pairwise<F: AudioFloat>(
    x: ArrayView2<F>,
    y: ArrayView2<F>,
    metric: DistanceMetric,
) -> Array2<F> {
    let mut out = Array2::zeros((x.ncols(), y.ncols()));
    let dist = RowDistances::new(y, metric);
    for (xi, mut row) in x.columns().into_iter().zip(out.rows_mut()) {
        dist.row(xi, row.as_slice_mut().expect("standard layout"));
    }
    out
}

fn check_finite<F: AudioFloat>(
    name: &str,
    a: impl IntoIterator<Item = F>,
) -> AudioSampleResult<()> {
    if a.into_iter().all(F::is_finite) {
        Ok(())
    } else {
        err(name, "must contain only finite values")
    }
}

// ---------------------------------------------------------------------------
// DTW
// ---------------------------------------------------------------------------

/// Full step list with additive and multiplicative weights.
type ResolvedSteps = (Vec<[usize; 2]>, Vec<f64>, Vec<f64>);

/// librosa's default DTW step set: diagonal, horizontal, vertical.
const DEFAULT_STEPS: [[usize; 2]; 3] = [[1, 1], [0, 1], [1, 0]];

/// Configuration for [`dtw`] and [`dtw_with_cost`].
///
/// Mirrors the keyword arguments of `librosa.sequence.dtw`.
/// [`DtwConfig::librosa`] gives librosa's defaults: Euclidean metric, steps
/// `(1,1), (0,1), (1,0)` with unit weights, full (not subsequence) alignment,
/// backtracking enabled, no global constraint. `Default` is currently the same.
///
/// The weights and `band_rad` are `f64`; [`dtw`] rounds the weights to the
/// element type of its input once.
///
/// # Invariants
///
/// [`DtwConfig::validate`] checks that the weight vectors match the step set,
/// that no custom step is `[0, 0]` (it would make backtracking loop forever),
/// that weights are not NaN and that `band_rad` is finite and non-negative.
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::{DistanceMetric, DtwConfig};
///
/// let cfg = DtwConfig::new()
///     .with_metric(DistanceMetric::Cosine)
///     .with_subseq(true);
/// assert!(cfg.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct DtwConfig {
    /// Metric for the local cost matrix when it is computed from features
    /// ([`dtw`]). Ignored by [`dtw_with_cost`].
    pub metric: DistanceMetric,
    /// Additional step sizes `[row_step, col_step]`. `None` uses librosa's
    /// default steps. As in librosa, custom steps are *appended* to the
    /// defaults, and the defaults then get infinite weights so that they are
    /// never preferred; step indices in [`DtwResult::steps`] count the three
    /// defaults first.
    pub step_sizes_sigma: Option<Vec<[usize; 2]>>,
    /// Additive weight per step (`cost = D[prev] + weights_mul * C + weights_add`).
    /// `None` means zeros. When given, its length must equal the number of
    /// custom steps (or 3 when `step_sizes_sigma` is `None`).
    pub weights_add: Option<Vec<f64>>,
    /// Multiplicative weight per step. `None` means ones. Same length rule as
    /// `weights_add`.
    pub weights_mul: Option<Vec<f64>>,
    /// Subsequence alignment: the shorter sequence may match any contiguous
    /// part of the longer one (free start and end along the longer axis).
    pub subseq: bool,
    /// Compute the optimal warping path ([`DtwResult::path`]).
    pub backtrack: bool,
    /// Apply a Sakoe-Chiba band: cells further than `band_rad` from the
    /// diagonal get infinite cost.
    pub global_constraints: bool,
    /// Band half-width as a fraction of the shorter sequence length. Default
    /// 0.25 (librosa's default). Only used when `global_constraints` is true.
    pub band_rad: f64,
}

impl Default for DtwConfig {
    fn default() -> Self {
        Self {
            metric: DistanceMetric::Euclidean,
            step_sizes_sigma: None,
            weights_add: None,
            weights_mul: None,
            subseq: false,
            backtrack: true,
            global_constraints: false,
            band_rad: 0.25,
        }
    }
}

impl DtwConfig {
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// librosa 1.0's defaults for `librosa.sequence.dtw`.
    ///
    /// Currently the same as [`Default`].
    #[must_use]
    pub fn librosa() -> Self {
        Self::default()
    }

    /// Sets [`DtwConfig::metric`].
    #[must_use]
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Sets [`DtwConfig::step_sizes_sigma`].
    #[must_use]
    pub fn with_step_sizes_sigma(mut self, steps: Option<Vec<[usize; 2]>>) -> Self {
        self.step_sizes_sigma = steps;
        self
    }

    /// Sets [`DtwConfig::weights_add`].
    #[must_use]
    pub fn with_weights_add(mut self, weights: Option<Vec<f64>>) -> Self {
        self.weights_add = weights;
        self
    }

    /// Sets [`DtwConfig::weights_mul`].
    #[must_use]
    pub fn with_weights_mul(mut self, weights: Option<Vec<f64>>) -> Self {
        self.weights_mul = weights;
        self
    }

    /// Sets [`DtwConfig::subseq`].
    #[must_use]
    pub fn with_subseq(mut self, subseq: bool) -> Self {
        self.subseq = subseq;
        self
    }

    /// Sets [`DtwConfig::backtrack`].
    #[must_use]
    pub fn with_backtrack(mut self, backtrack: bool) -> Self {
        self.backtrack = backtrack;
        self
    }

    /// Sets [`DtwConfig::global_constraints`].
    #[must_use]
    pub fn with_global_constraints(mut self, global_constraints: bool) -> Self {
        self.global_constraints = global_constraints;
        self
    }

    /// Sets [`DtwConfig::band_rad`].
    #[must_use]
    pub fn with_band_rad(mut self, band_rad: f64) -> Self {
        self.band_rad = band_rad;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ParameterError`] if a weight vector has the wrong length or
    /// contains NaN, a custom step is `[0, 0]`, or `band_rad` is negative or
    /// not finite.
    pub fn validate(&self) -> AudioSampleResult<()> {
        self.resolve().map(|_| ())
    }

    /// Full step set and weights, librosa style (defaults first).
    fn resolve(&self) -> AudioSampleResult<ResolvedSteps> {
        if !self.band_rad.is_finite() || self.band_rad < 0.0 {
            return Err(ParameterError::out_of_range(
                "band_rad",
                self.band_rad,
                0.0,
                f64::INFINITY,
                "the band radius must be finite and non-negative",
            )
            .into());
        }
        let n_custom = self.step_sizes_sigma.as_ref().map_or(3, Vec::len);
        for (name, w) in [
            ("weights_add", &self.weights_add),
            ("weights_mul", &self.weights_mul),
        ] {
            if let Some(w) = w {
                if w.len() != n_custom {
                    return err(
                        name,
                        format!(
                            "length {} must equal the number of steps in step_sizes_sigma ({n_custom})",
                            w.len()
                        ),
                    );
                }
                if w.iter().any(|x| x.is_nan()) {
                    return err(name, "must not contain NaN");
                }
            }
        }
        let w_add = |default: f64| {
            self.weights_add
                .clone()
                .unwrap_or_else(|| vec![default; n_custom])
        };
        let w_mul = |default: f64| {
            self.weights_mul
                .clone()
                .unwrap_or_else(|| vec![default; n_custom])
        };
        match &self.step_sizes_sigma {
            None => Ok((DEFAULT_STEPS.to_vec(), w_add(0.0), w_mul(1.0))),
            Some(custom) => {
                if custom.iter().any(|s| s[0] == 0 && s[1] == 0) {
                    return err(
                        "step_sizes_sigma",
                        "a step of [0, 0] is not allowed (it never advances)",
                    );
                }
                let mut steps = DEFAULT_STEPS.to_vec();
                steps.extend_from_slice(custom);
                let mut wa = vec![f64::INFINITY; 3];
                wa.extend(w_add(0.0));
                let mut wm = vec![f64::INFINITY; 3];
                wm.extend(w_mul(1.0));
                Ok((steps, wa, wm))
            }
        }
    }
}

/// Output of [`dtw`] / [`dtw_with_cost`].
///
/// `F` is the element type of the input (`f32` or `f64`).
///
/// # Invariants
///
/// `cost` and `steps` have the shape of the local cost matrix (`(N, M)` for
/// `X` with `N` frames and `Y` with `M` frames, transposed when a subsequence
/// alignment swapped the roles, as in librosa).
#[derive(Debug, Clone, PartialEq)]
pub struct DtwResult<F = f64> {
    /// Accumulated cost matrix `D` (librosa's first return value).
    /// Unreachable cells are infinite.
    pub cost: Array2<F>,
    /// Optimal warping path as `[index_in_x, index_in_y]` pairs, **from the
    /// end of the alignment back to its start** (librosa order). `None` when
    /// [`DtwConfig::backtrack`] is false.
    pub path: Option<Vec<[usize; 2]>>,
    /// Index of the step (into librosa's combined step list, defaults first)
    /// that reached each cell; feed it to [`dtw_backtracking`] to extract
    /// other paths (librosa's `return_steps=True`).
    pub steps: Array2<usize>,
}

/// Dynamic time warping between two feature sequences.
///
/// Builds the local cost matrix `C[n, m] = metric(x[:, n], y[:, m])` and
/// accumulates it with dynamic programming:
/// `D[n, m] = min_k D[n - s_k0, m - s_k1] + w_mul_k * C[n, m] + w_add_k`.
/// Equivalent to `librosa.sequence.dtw(X, Y, ...)`. The costs are computed in
/// the element type `F` of `x` and `y` (`f32` or `f64`).
///
/// # Arguments
///
/// - `x` - First sequence, `(n_features, N)`.
/// - `y` - Second sequence, `(n_features, M)`; same number of features as `x`.
/// - `cfg` - Step sizes, weights, subsequence mode, band constraint.
///
/// # Returns
///
/// The accumulated cost matrix, the step matrix and (if requested) the
/// warping path. For a subsequence alignment with `N > M`, librosa transposes
/// the problem; `cost` is then `(M, N)` while the path is still reported as
/// `[index_in_x, index_in_y]`.
///
/// # Errors
///
/// - [`ParameterError`] if the configuration is invalid, either sequence is
///   empty, the feature counts differ, the cost matrix contains NaN (e.g.
///   cosine distance with an all-zero frame), or no valid path exists with
///   the given steps/band.
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::{dtw, DtwConfig};
/// use ndarray::array;
///
/// // Find where a short query best matches inside a longer reference.
/// let query = array![[1.0, 2.0, 3.0]];
/// let reference = array![[9.0, 9.0, 1.0, 2.0, 3.0, 9.0]];
/// let cfg = DtwConfig::new().with_subseq(true);
/// let result = dtw(query.view(), reference.view(), &cfg).unwrap();
/// let path = result.path.unwrap();
/// assert_eq!(path, vec![[2, 4], [1, 3], [0, 2]]);
/// ```
///
/// # References
///
/// Meinard Müller, *Fundamentals of Music Processing*, Springer 2015, ch. 3.2
/// and 7.2; `librosa.sequence.dtw`.
pub fn dtw<F: AudioFloat>(
    x: ArrayView2<F>,
    y: ArrayView2<F>,
    cfg: &DtwConfig,
) -> AudioSampleResult<DtwResult<F>> {
    if x.nrows() != y.nrows() {
        return err(
            "y",
            format!(
                "x has {} features but y has {}; both must be (n_features, n_frames)",
                x.nrows(),
                y.nrows()
            ),
        );
    }
    if x.ncols() == 0 || y.ncols() == 0 {
        return err("x / y", "sequences must contain at least one frame");
    }
    // A subsequence alignment of a longer `x` runs on the transposed cost
    // matrix (as in librosa); every metric is symmetric, so it is computed
    // directly as the distances from `y` to `x`.
    let transposed = cfg.subseq && x.ncols() > y.ncols();
    let c = if transposed {
        pairwise(y, x, cfg.metric)
    } else {
        pairwise(x, y, cfg.metric)
    };
    dtw_core(c, cfg, transposed)
}

/// Dynamic time warping on a precomputed local cost matrix.
///
/// Equivalent to `librosa.sequence.dtw(C=c, ...)`; [`DtwConfig::metric`] is
/// ignored. Infinite costs are allowed (they forbid cells). The accumulation
/// runs in the element type `F` of `c`.
///
/// # Arguments
///
/// - `c` - Local cost matrix `(N, M)`, no NaN.
/// - `cfg` - Step sizes, weights, subsequence mode, band constraint.
///
/// # Returns
///
/// See [`DtwResult`]. Like librosa, when `subseq` is true and `c` has more rows
/// than columns, the path columns are swapped (`[col, row]`) although `c` is
/// not transposed; pass `c` with `N <= M` to avoid this quirk.
///
/// # Errors
///
/// [`ParameterError`] for an invalid configuration, an empty or NaN-containing
/// `c`, or when no valid path exists.
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::{dtw_with_cost, DtwConfig};
/// use ndarray::array;
///
/// let c = array![[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]];
/// let r = dtw_with_cost(c.view(), &DtwConfig::default()).unwrap();
/// assert_eq!(r.cost[[1, 2]], 1.0);
/// assert_eq!(r.path.unwrap(), vec![[1, 2], [1, 1], [0, 0]]);
/// ```
pub fn dtw_with_cost<F: AudioFloat>(
    c: ArrayView2<F>,
    cfg: &DtwConfig,
) -> AudioSampleResult<DtwResult<F>> {
    if c.is_empty() {
        return err("c", "cost matrix must be non-empty");
    }
    dtw_core(c.to_owned(), cfg, false)
}

/// librosa's `fill_off_diagonal(x, radius, value)` (Sakoe-Chiba band).
fn fill_off_diagonal<F: AudioFloat>(x: &mut Array2<F>, radius: f64, value: F) {
    let (nx, ny) = x.dim();
    let r = (radius * nx.min(ny) as f64).round_ties_even() as i64;
    let offset = nx.abs_diff(ny) as i64;
    for ((i, j), v) in x.indexed_iter_mut() {
        let d = j as i64 - i as i64;
        let off = if nx < ny {
            d >= r + offset || d <= -r
        } else {
            d >= r || d <= -r - offset
        };
        if off {
            *v = value;
        }
    }
}

fn dtw_core<F: AudioFloat>(
    c: Array2<F>,
    cfg: &DtwConfig,
    transposed: bool,
) -> AudioSampleResult<DtwResult<F>> {
    let (sigma, w_add, w_mul) = cfg.resolve()?;
    let w_add: Vec<F> = w_add.into_iter().map(F::of).collect();
    let w_mul: Vec<F> = w_mul.into_iter().map(F::of).collect();
    if c.iter().any(|v| v.is_nan()) {
        return err("c", "DTW cost matrix contains NaN values");
    }
    let mut d = if c.is_standard_layout() {
        c
    } else {
        c.as_standard_layout().into_owned()
    };
    if cfg.global_constraints {
        fill_off_diagonal(&mut d, cfg.band_rad, F::infinity());
    }
    let (n, m) = d.dim();
    let mut steps = Array2::<usize>::zeros((n, m));
    accumulate(
        d.as_slice_mut().expect("standard layout"),
        steps.as_slice_mut().expect("standard layout"),
        m,
        &sigma,
        &w_add,
        &w_mul,
        cfg.subseq,
    );

    let path = if cfg.backtrack {
        let mut wp = if cfg.subseq {
            let last = d.row(n - 1);
            if last.iter().all(|v| v.is_infinite()) {
                return err(
                    "step_sizes_sigma",
                    "no valid sub-sequence warping path could be constructed with the given step sizes",
                );
            }
            let start = argmin_first(last.iter().copied());
            backtrack_path(steps.view(), &sigma, true, Some(start))?
        } else {
            if d[[n - 1, m - 1]].is_infinite() {
                return err(
                    "step_sizes_sigma",
                    "no valid warping path could be constructed with the given step sizes / band",
                );
            }
            let wp = backtrack_path(steps.view(), &sigma, false, None)?;
            if wp.last() != Some(&[0, 0]) {
                return err(
                    "subseq",
                    "unable to compute a full DTW warping path; try subseq = true",
                );
            }
            wp
        };
        if cfg.subseq && (transposed || n > m) {
            for p in &mut wp {
                p.swap(0, 1);
            }
        }
        Some(wp)
    } else {
        None
    };
    Ok(DtwResult {
        cost: d,
        path,
        steps,
    })
}

/// librosa's `__dtw_calc_accu_cost`, in place: `d` holds the local cost
/// matrix (`m` columns, row-major) on entry and the accumulated cost on exit;
/// `steps` (zeros on entry) receives the index of the step taken into each
/// cell.
///
/// librosa pads the accumulated matrix with infinite rows and columns so that
/// every step has a predecessor. A step from the padding adds to `inf` and can
/// never improve a cell (the sum is `inf` or NaN, and the comparison is a
/// strict `<`), so here such steps are skipped instead, which gives the same
/// values without the padded copies.
fn accumulate<F: AudioFloat>(
    d: &mut [F],
    steps: &mut [usize],
    m: usize,
    sigma: &[[usize; 2]],
    w_add: &[F],
    w_mul: &[F],
    subseq: bool,
) {
    let inf = F::infinity();
    let n = d.len() / m;
    // Cell (r, j): its local cost is still in d[r * m + j] when it is reached.
    // The first row starts from the local cost in a subsequence alignment, and
    // cell (0, 0) always does; every other cell starts from infinity.
    let start = |r: usize, j: usize, local: F| {
        if r == 0 && (subseq || j == 0) {
            local
        } else {
            inf
        }
    };
    let generic_cell = |d: &mut [F], steps: &mut [usize], r: usize, j: usize| {
        let idx = r * m + j;
        let local = d[idx];
        let mut best = start(r, j, local);
        let mut arg = 0;
        for (k, s) in sigma.iter().enumerate() {
            if r >= s[0] && j >= s[1] {
                let cur_d = d[idx - s[0] * m - s[1]];
                let mut cur_c = w_mul[k] * local;
                cur_c += w_add[k];
                let cost = cur_d + cur_c;
                if cost < best {
                    best = cost;
                    arg = k;
                }
            }
        }
        d[idx] = best;
        steps[idx] = arg;
    };
    if sigma != DEFAULT_STEPS {
        for r in 0..n {
            for j in 0..m {
                generic_cell(d, steps, r, j);
            }
        }
        return;
    }
    // librosa's default steps: diagonal (1, 1), horizontal (0, 1), vertical
    // (1, 0). The first row and column have fewer predecessors.
    for j in 0..m {
        generic_cell(d, steps, 0, j);
    }
    let (wm0, wm1, wm2) = (w_mul[0], w_mul[1], w_mul[2]);
    let (wa0, wa1, wa2) = (w_add[0], w_add[1], w_add[2]);
    for r in 1..n {
        generic_cell(d, steps, r, 0);
        let (above, rest) = d.split_at_mut(r * m);
        let prev = &above[(r - 1) * m..];
        let cur = &mut rest[..m];
        let st = &mut steps[r * m..(r + 1) * m];
        let mut left = cur[0];
        for j in 1..m {
            let local = cur[j];
            let mut best = inf;
            let mut arg = 0;
            let mut c0 = wm0 * local;
            c0 += wa0;
            let cost = prev[j - 1] + c0;
            if cost < best {
                best = cost;
            }
            let mut c1 = wm1 * local;
            c1 += wa1;
            let cost = left + c1;
            if cost < best {
                best = cost;
                arg = 1;
            }
            let mut c2 = wm2 * local;
            c2 += wa2;
            let cost = prev[j] + c2;
            if cost < best {
                best = cost;
                arg = 2;
            }
            cur[j] = best;
            st[j] = arg;
            left = best;
        }
    }
}

/// Index of the first minimum (`numpy.argmin`); NaN-free input assumed.
fn argmin_first<F: AudioFloat>(it: impl Iterator<Item = F>) -> usize {
    let mut best = (0, F::infinity());
    let mut first = true;
    for (i, v) in it.enumerate() {
        if first || v < best.1 {
            best = (i, v);
            first = false;
        }
    }
    best.0
}

/// Index of the first maximum (`numpy.argmax`); NaN-free input assumed.
fn argmax_first<F: AudioFloat>(it: impl Iterator<Item = F>) -> usize {
    let mut best = (0, F::neg_infinity());
    let mut first = true;
    for (i, v) in it.enumerate() {
        if first || v > best.1 {
            best = (i, v);
            first = false;
        }
    }
    best.0
}

fn backtrack_path<S: Copy + std::fmt::Display + num_traits::ToPrimitive>(
    steps: ArrayView2<S>,
    sigma: &[[usize; 2]],
    subseq: bool,
    start: Option<usize>,
) -> AudioSampleResult<Vec<[usize; 2]>> {
    let (rows, cols) = steps.dim();
    let mut cur = [rows - 1, start.unwrap_or(cols - 1)];
    let mut wp = vec![cur];
    while (subseq && cur[0] > 0) || (!subseq && cur != [0, 0]) {
        let k = steps[cur];
        let Some(s) = k.to_usize().and_then(|i| sigma.get(i)) else {
            return err(
                "steps",
                format!(
                    "step index {k} at {cur:?} is outside the step list (0..{})",
                    sigma.len()
                ),
            );
        };
        if s[0] > cur[0] || s[1] > cur[1] {
            break;
        }
        cur = [cur[0] - s[0], cur[1] - s[1]];
        wp.push(cur);
    }
    Ok(wp)
}

/// Backtracks a warping path from a DTW step matrix.
///
/// Equivalent to `librosa.sequence.dtw_backtracking`. Useful to extract a
/// subsequence match ending at a column other than the global optimum.
///
/// # Arguments
///
/// - `steps` - Step matrix from [`DtwResult::steps`], or the same indices in
///   any primitive integer type (for example librosa's `int32` matrix).
/// - `step_sizes_sigma` - The custom steps used for the DTW (`None` for
///   librosa's defaults); appended to the defaults exactly as in [`dtw`].
/// - `subseq` - Whether the DTW was a subsequence alignment.
/// - `start` - Column of the last row to start from (subsequence only;
///   `None` starts at the bottom-right cell).
///
/// # Returns
///
/// The path from `start` back to the first row (subsequence) or to `[0, 0]`,
/// in librosa order (end first).
///
/// # Errors
///
/// [`ParameterError`] if `steps` is empty, `start` is given without `subseq`
/// or is out of range, a custom step is `[0, 0]`, or the path meets an index
/// in `steps` outside the step list (negative indices included).
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::{dtw_backtracking, dtw_with_cost, DtwConfig};
/// use ndarray::array;
///
/// let c = array![[0.0, 1.0, 0.5, 1.0], [1.0, 0.0, 1.0, 0.2]];
/// let cfg = DtwConfig::new().with_subseq(true);
/// let r = dtw_with_cost(c.view(), &cfg).unwrap();
/// let alt = dtw_backtracking(r.steps.view(), None, true, Some(1)).unwrap();
/// assert_eq!(alt, vec![[1, 1], [0, 0]]);
/// ```
pub fn dtw_backtracking<S: Copy + std::fmt::Display + num_traits::ToPrimitive>(
    steps: ArrayView2<S>,
    step_sizes_sigma: Option<&[[usize; 2]]>,
    subseq: bool,
    start: Option<usize>,
) -> AudioSampleResult<Vec<[usize; 2]>> {
    if !subseq && start.is_some() {
        return err("start", "start is only allowed when subseq is true");
    }
    if steps.is_empty() {
        return err("steps", "step matrix must be non-empty");
    }
    if let Some(s) = start {
        if s >= steps.ncols() {
            return Err(ParameterError::out_of_range(
                "start",
                s,
                0,
                steps.ncols() - 1,
                "start must index a column of the step matrix",
            )
            .into());
        }
    }
    let mut sigma = DEFAULT_STEPS.to_vec();
    if let Some(custom) = step_sizes_sigma {
        if custom.iter().any(|s| s[0] == 0 && s[1] == 0) {
            return err("step_sizes_sigma", "a step of [0, 0] is not allowed");
        }
        sigma.extend_from_slice(custom);
    }
    backtrack_path(steps, &sigma, subseq, start)
}

// ---------------------------------------------------------------------------
// RQA
// ---------------------------------------------------------------------------

/// Configuration for [`rqa`].
///
/// [`RqaConfig::librosa`] gives librosa's defaults (unit gap penalties, knight
/// moves, backtracking); `Default` is currently the same. The penalties are
/// `f64` and are rounded to the element type of the similarity matrix once.
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::RqaConfig;
///
/// let cfg = RqaConfig::new().with_knight_moves(false).with_gap_onset(2.0);
/// assert!(cfg.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RqaConfig {
    /// Penalty for starting a gap (a non-recurrent cell right after a
    /// recurrent one). Finite and `>= 0`. Default 1 (librosa's default).
    pub gap_onset: f64,
    /// Penalty for extending a gap. Finite and `>= 0`. Default 1 (librosa's
    /// default).
    pub gap_extend: f64,
    /// Allow knight moves `(1, 2)` / `(2, 1)` in addition to the diagonal,
    /// tolerating small tempo deviations. Default `true` (librosa's default).
    pub knight_moves: bool,
    /// Compute the optimal path ([`RqaResult::path`]). Default `true`
    /// (librosa's default).
    pub backtrack: bool,
}

impl Default for RqaConfig {
    fn default() -> Self {
        Self {
            gap_onset: 1.0,
            gap_extend: 1.0,
            knight_moves: true,
            backtrack: true,
        }
    }
}

impl RqaConfig {
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// librosa 1.0's defaults for `librosa.sequence.rqa`.
    ///
    /// Currently the same as [`Default`].
    #[must_use]
    pub fn librosa() -> Self {
        Self::default()
    }

    /// Sets [`RqaConfig::gap_onset`].
    #[must_use]
    pub fn with_gap_onset(mut self, v: f64) -> Self {
        self.gap_onset = v;
        self
    }

    /// Sets [`RqaConfig::gap_extend`].
    #[must_use]
    pub fn with_gap_extend(mut self, v: f64) -> Self {
        self.gap_extend = v;
        self
    }

    /// Sets [`RqaConfig::knight_moves`].
    #[must_use]
    pub fn with_knight_moves(mut self, v: bool) -> Self {
        self.knight_moves = v;
        self
    }

    /// Sets [`RqaConfig::backtrack`].
    #[must_use]
    pub fn with_backtrack(mut self, v: bool) -> Self {
        self.backtrack = v;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if a gap penalty is negative or not finite.
    pub fn validate(&self) -> AudioSampleResult<()> {
        for (name, v) in [
            ("gap_onset", self.gap_onset),
            ("gap_extend", self.gap_extend),
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(ParameterError::out_of_range(
                    name,
                    v,
                    0.0,
                    f64::INFINITY,
                    "gap penalties must be finite and non-negative",
                )
                .into());
            }
        }
        Ok(())
    }
}

/// Output of [`rqa`]. `F` is the element type of the similarity matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct RqaResult<F = f64> {
    /// Alignment score matrix, same shape and element type as the similarity
    /// matrix.
    pub score: Array2<F>,
    /// Optimal local alignment path as `[row, col]` pairs **from start to end**
    /// (librosa order for `rqa`). Empty if there is no recurrence. `None` when
    /// [`RqaConfig::backtrack`] is false.
    pub path: Option<Vec<[usize; 2]>>,
}

/// Recurrence quantification analysis: optimal local alignment in a
/// (cross-)similarity matrix.
///
/// A Smith-Waterman-like dynamic program that rewards runs of recurrent cells
/// along diagonals (optionally with knight moves) and penalises gaps.
/// Equivalent to `librosa.sequence.rqa`. The scores are computed in the
/// element type `F` of `sim`.
///
/// # Arguments
///
/// - `sim` - Similarity / recurrence matrix (binary or non-negative
///   affinity), e.g. from [`cross_similarity`]. No NaN.
/// - `cfg` - Gap penalties, knight moves, backtracking.
///
/// # Returns
///
/// The score matrix and (optionally) the best-scoring path.
///
/// # Errors
///
/// [`ParameterError`] for an invalid config, an empty `sim` or NaN entries.
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::{rqa, RqaConfig};
/// use ndarray::Array2;
///
/// // A diagonal run of recurrences from (1, 2) to (4, 5).
/// let mut sim = Array2::<f64>::zeros((6, 7));
/// for i in 1..5 {
///     sim[[i, i + 1]] = 1.0;
/// }
/// let r = rqa(sim.view(), &RqaConfig::default()).unwrap();
/// assert_eq!(r.score[[4, 5]], 4.0);
/// assert_eq!(r.path.unwrap(), vec![[1, 2], [2, 3], [3, 4], [4, 5]]);
/// ```
///
/// # Differences from librosa
///
/// librosa stores the wrong back-pointer for a `(-2, -1)` move into the second
/// column (`j == 1`), which makes its backtracking step outside the matrix.
/// The scores are identical; the path here follows the move that was actually
/// taken.
///
/// # References
///
/// J. Serrà, X. Serra and R. G. Andrzejak, "Cross recurrence quantification
/// for cover song identification", New Journal of Physics 11 (2009).
pub fn rqa<F: AudioFloat>(sim: ArrayView2<F>, cfg: &RqaConfig) -> AudioSampleResult<RqaResult<F>> {
    cfg.validate()?;
    if sim.is_empty() {
        return err("sim", "similarity matrix must be non-empty");
    }
    if sim.iter().any(|v| v.is_nan()) {
        return err("sim", "similarity matrix must not contain NaN");
    }
    let (score, ptr) = rqa_dp(
        sim,
        F::of(cfg.gap_onset),
        F::of(cfg.gap_extend),
        cfg.knight_moves,
    );
    let path = cfg.backtrack.then(|| rqa_backtrack(&score, &ptr));
    Ok(RqaResult { score, path })
}

/// Back-pointer codes: -2 path start, -1 no path, 0 diag, 1 (-1,-2), 2 (-2,-1).
fn rqa_dp<F: AudioFloat>(
    sim: ArrayView2<F>,
    gap_onset: F,
    gap_extend: F,
    knight: bool,
) -> (Array2<F>, Array2<i8>) {
    let zero = F::zero();
    let (rows, cols) = sim.dim();
    let mut score = Array2::<F>::zeros((rows, cols));
    let mut bt = Array2::<i8>::zeros((rows, cols));
    let (init_limit, limit) = if knight { (2, 3) } else { (1, 1) };
    score.row_mut(0).assign(&sim.row(0));
    score.column_mut(0).assign(&sim.column(0));
    for i in 0..rows {
        bt[[i, 0]] = if sim[[i, 0]] != zero { -2 } else { -1 };
    }
    for j in 0..cols {
        bt[[0, j]] = if sim[[0, j]] != zero { -2 } else { -1 };
    }
    if rows == 1 || cols == 1 {
        return (score, bt);
    }
    let penalty = |prev_sim: F| {
        if prev_sim > zero {
            gap_onset
        } else {
            gap_extend
        }
    };

    if sim[[1, 1]] > zero {
        score[[1, 1]] = score[[0, 0]] + sim[[1, 1]];
        bt[[1, 1]] = 0;
    } else {
        let v = score[[0, 0]] - penalty(sim[[0, 0]]);
        score[[1, 1]] = if v > zero { v } else { zero };
        bt[[1, 1]] = if score[[1, 1]] > zero { 0 } else { -1 };
    }

    // One cell update: candidates are (score, sim, pointer code) per move.
    let update =
        |score: &mut Array2<F>, bt: &mut Array2<i8>, i: usize, j: usize, cands: &[(F, F, i8)]| {
            if sim[[i, j]] > zero {
                let k = argmax_first(cands.iter().map(|c| c.0));
                bt[[i, j]] = cands[k].2;
                score[[i, j]] = cands[k].0 + sim[[i, j]];
            } else {
                let gapped: Vec<F> = cands.iter().map(|c| c.0 - penalty(c.1)).collect();
                let k = argmax_first(gapped.iter().copied());
                let v = gapped[k];
                score[[i, j]] = if v > zero { v } else { zero };
                bt[[i, j]] = if score[[i, j]] == zero {
                    -1
                } else {
                    cands[k].2
                };
            }
        };

    let i = 1;
    for j in 2..cols {
        let cands = [
            (score[[i - 1, j - 1]], sim[[i - 1, j - 1]], 0i8),
            (score[[i - 1, j - 2]], sim[[i - 1, j - 2]], 1i8),
        ];
        update(&mut score, &mut bt, i, j, &cands[..init_limit]);
    }
    let j = 1;
    for i in 2..rows {
        let cands = [
            (score[[i - 1, j - 1]], sim[[i - 1, j - 1]], 0i8),
            (score[[i - 2, j - 1]], sim[[i - 2, j - 1]], 2i8),
        ];
        update(&mut score, &mut bt, i, j, &cands[..init_limit]);
    }
    for i in 2..rows {
        for j in 2..cols {
            let cands = [
                (score[[i - 1, j - 1]], sim[[i - 1, j - 1]], 0i8),
                (score[[i - 1, j - 2]], sim[[i - 1, j - 2]], 1i8),
                (score[[i - 2, j - 1]], sim[[i - 2, j - 1]], 2i8),
            ];
            update(&mut score, &mut bt, i, j, &cands[..limit]);
        }
    }
    (score, bt)
}

fn rqa_backtrack<F: AudioFloat>(score: &Array2<F>, ptr: &Array2<i8>) -> Vec<[usize; 2]> {
    let cols = score.ncols();
    let flat = argmax_first(score.iter().copied());
    let mut idx = [flat / cols, flat % cols];
    let mut path = Vec::new();
    loop {
        let p = ptr[idx];
        if p == -1 {
            break;
        }
        path.push(idx);
        if p == -2 {
            break;
        }
        let (di, dj) = match p {
            0 => (1, 1),
            1 => (1, 2),
            _ => (2, 1),
        };
        if idx[0] < di || idx[1] < dj {
            // Cannot happen for pointers produced by `rqa_dp`; stop defensively.
            break;
        }
        idx = [idx[0] - di, idx[1] - dj];
    }
    path.reverse();
    path
}

// ---------------------------------------------------------------------------
// Viterbi
// ---------------------------------------------------------------------------

/// Output of [`viterbi`] / [`viterbi_discriminative`]. `F` is the element
/// type of the probability matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct ViterbiResult<F = f64> {
    /// Most likely state index per time step (`n_steps` entries).
    pub states: Vec<usize>,
    /// Log-likelihood of `states` (librosa's `logp` with `return_logp=True`).
    /// Unnormalised for the discriminative variant.
    pub log_prob: F,
}

/// Output of [`viterbi_binary`]. `F` is the element type of the probability
/// matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct ViterbiBinaryResult<F = f64> {
    /// `(n_states, n_steps)` matrix of activations per state: `true` where the
    /// state is on.
    pub states: Array2<bool>,
    /// Log-likelihood of each state's binary sequence (`n_states` entries).
    pub log_prob: Array1<F>,
}

fn check_square_transition<F: AudioFloat>(
    transition: ArrayView2<F>,
    n_states: usize,
) -> AudioSampleResult<()> {
    if transition.dim() != (n_states, n_states) {
        return err(
            "transition",
            format!(
                "shape {:?} must be (n_states, n_states) = ({n_states}, {n_states})",
                transition.dim()
            ),
        );
    }
    check_stochastic_rows(transition)
}

fn check_stochastic_rows<F: AudioFloat>(transition: ArrayView2<F>) -> AudioSampleResult<()> {
    for row in transition.rows() {
        if row.iter().any(|&v| !v.is_finite() || v < F::zero()) || !allclose(sum_f64(&row), 1.0) {
            return err(
                "transition",
                "invalid transition matrix: must be non-negative and sum to 1 on each row",
            );
        }
    }
    Ok(())
}

fn check_distribution<F: AudioFloat>(
    name: &str,
    p: ArrayView1<F>,
    n_states: usize,
) -> AudioSampleResult<()> {
    if p.len() != n_states
        || p.iter().any(|&v| !v.is_finite() || v < F::zero())
        || !allclose(sum_f64(&p), 1.0)
    {
        return err(
            name,
            format!("must be a non-negative distribution of length {n_states} summing to 1"),
        );
    }
    Ok(())
}

fn log_threshold<F: AudioFloat>(transition_min_prob: Option<f64>) -> AudioSampleResult<Option<F>> {
    match transition_min_prob {
        None | Some(0.0) => Ok(None),
        Some(p) if p > 0.0 && p.is_finite() => Ok(Some((F::of(p) + tiny::<F>()).ln())),
        Some(p) => Err(ParameterError::out_of_range(
            "transition_min_prob",
            p,
            0.0,
            f64::INFINITY,
            "must be None or a finite non-negative probability",
        )
        .into()),
    }
}

fn check_prob_shape<F: AudioFloat>(prob: ArrayView2<F>) -> AudioSampleResult<()> {
    if prob.nrows() == 0 || prob.ncols() == 0 {
        return err("prob", "must have at least one state and one time step");
    }
    Ok(())
}

/// Candidate predecessors of one target state in [`viterbi_core`].
enum Live {
    /// The contiguous range `lo..hi`.
    Range(usize, usize),
    /// An arbitrary (sorted) list.
    List(Vec<usize>),
}

/// Largest `a[i] + b[i]` (NaN sums never win; `-inf` for empty input). The
/// eight independent accumulators let the compiler vectorise the reduction;
/// the result is exact because a maximum does not round.
#[inline]
fn max_of_sums<F: AudioFloat>(a: &[F], b: &[F]) -> F {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let ninf = F::neg_infinity();
    let mut acc = [ninf; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for i in 0..8 {
            let c = x[i] + y[i];
            acc[i] = if c > acc[i] { c } else { acc[i] };
        }
    }
    let mut m = ninf;
    for &v in &acc {
        if v > m {
            m = v;
        }
    }
    for (&x, &y) in ca.remainder().iter().zip(cb.remainder()) {
        let c = x + y;
        if c > m {
            m = c;
        }
    }
    m
}

/// librosa's inner loop over the candidates `lo..hi`: the first index `k`
/// with the largest `prev[k] + lt[k]` that exceeds `-inf`, and that sum.
#[inline(always)]
fn best_in_range<F: AudioFloat>(prev: &[F], lt: &[F], lo: usize, hi: usize) -> (F, Option<usize>) {
    let (p, l) = (&prev[lo..hi], &lt[lo..hi]);
    if p.len() <= 16 {
        // Short ranges (banded matrices): one branch-free pass.
        let mut best = F::neg_infinity();
        let mut arg = usize::MAX;
        for (i, (&a, &b)) in p.iter().zip(l).enumerate() {
            let c = a + b;
            let gt = c > best;
            best = if gt { c } else { best };
            arg = if gt { i } else { arg };
        }
        return (best, (arg != usize::MAX).then(|| lo + arg));
    }
    let m = max_of_sums(p, l);
    if m > F::neg_infinity() {
        let i = p.iter().zip(l).position(|(&a, &b)| a + b == m);
        (m, i.map(|i| lo + i))
    } else {
        (m, None)
    }
}

/// librosa's inner loop over an explicit candidate list.
#[inline]
fn best_in_list<F: AudioFloat>(prev: &[F], lt: &[F], ks: &[usize]) -> (F, Option<usize>) {
    let mut best = F::neg_infinity();
    let mut arg = None;
    for &k in ks {
        let cost = prev[k] + lt[k];
        if cost > best {
            best = cost;
            arg = Some(k);
        }
    }
    (best, arg)
}

/// The core log-domain Viterbi recursion (librosa's `_viterbi`).
///
/// `fill_log_prob(t, col)` writes the `n_states` observation log-likelihoods of
/// step `t` into `col`; `log_trans_t[j * n_states + k]` is the log-transition
/// `k -> j` (the transposed log matrix, so that the candidates of one target
/// state are contiguous). Ties are broken towards the lowest state index, as in
/// librosa.
///
/// Transitions whose log value is exactly `ln(tiny)` (zero probabilities) all
/// add the same constant, so their best total is bounded by the best previous
/// value among them plus that constant. They are only scanned when that bound
/// reaches the best total of the other candidates; otherwise none of them can
/// be the (first) maximum and skipping them gives librosa's result exactly.
fn viterbi_core<F: AudioFloat>(
    n_states: usize,
    n_steps: usize,
    mut fill_log_prob: impl FnMut(usize, &mut [F]),
    log_trans_t: &[F],
    log_p_init: &[F],
    threshold: Option<F>,
) -> AudioSampleResult<ViterbiResult<F>> {
    let n = n_states;
    let ninf = F::neg_infinity();
    let l0 = tiny::<F>().ln();
    // Candidates of each target state (librosa's `pred_states` with a
    // threshold, every state otherwise), split into the live ones and a flag
    // for structural zeros among them.
    let mut cands: Vec<Vec<usize>> = Vec::new();
    let mut live: Vec<Live> = Vec::with_capacity(n);
    let mut has_zero = vec![false; n];
    for j in 0..n {
        let lt = &log_trans_t[j * n..(j + 1) * n];
        let c: Vec<usize> = match threshold {
            Some(th) => {
                let c: Vec<usize> = (0..n).filter(|&k| lt[k] >= th).collect();
                if c.is_empty() {
                    return err(
                        "transition_min_prob",
                        format!(
                            "no transition into state {j} reaches the threshold; reduce transition_min_prob"
                        ),
                    );
                }
                c
            }
            None => Vec::new(),
        };
        let all = threshold.is_none();
        let is_live = |k: usize| lt[k] != l0;
        let (n_cand, n_live, first, last) = if all {
            let first = (0..n).find(|&k| is_live(k));
            let last = (0..n).rev().find(|&k| is_live(k));
            let n_live = (0..n).filter(|&k| is_live(k)).count();
            (n, n_live, first, last)
        } else {
            let first = c.iter().copied().find(|&k| is_live(k));
            let last = c.iter().copied().rev().find(|&k| is_live(k));
            let n_live = c.iter().filter(|&&k| is_live(k)).count();
            (c.len(), n_live, first, last)
        };
        has_zero[j] = n_live < n_cand;
        live.push(match (first, last) {
            (Some(lo), Some(hi)) if hi - lo + 1 == n_live => Live::Range(lo, hi + 1),
            _ if all => Live::List((0..n).filter(|&k| is_live(k)).collect()),
            _ => Live::List(c.iter().copied().filter(|&k| is_live(k)).collect()),
        });
        if !all {
            cands.push(c);
        }
    }
    let any_zero = has_zero.iter().any(|&z| z);

    let mut ptr = vec![0u32; n_steps * n];
    let mut col = vec![F::zero(); n];
    let mut prev = vec![F::zero(); n];
    let mut cur = vec![F::zero(); n];
    // pre[i] = max(prev[..i]), suf[i] = max(prev[i..]).
    let mut pre = vec![ninf; n + 1];
    let mut suf = vec![ninf; n + 1];
    fill_log_prob(0, &mut col);
    for j in 0..n {
        prev[j] = col[j] + log_p_init[j];
    }
    for t in 1..n_steps {
        fill_log_prob(t, &mut col);
        if any_zero {
            for k in 0..n {
                pre[k + 1] = if prev[k] > pre[k] { prev[k] } else { pre[k] };
            }
            for k in (0..n).rev() {
                suf[k] = if prev[k] > suf[k + 1] {
                    prev[k]
                } else {
                    suf[k + 1]
                };
            }
        }
        let row = &mut ptr[t * n..(t + 1) * n];
        for j in 0..n {
            let lt = &log_trans_t[j * n..(j + 1) * n];
            let (mut best, mut arg) = match &live[j] {
                Live::Range(lo, hi) => best_in_range(&prev, lt, *lo, *hi),
                Live::List(ks) => best_in_list(&prev, lt, ks),
            };
            if has_zero[j] {
                let zmax = match (&live[j], threshold) {
                    (Live::Range(lo, hi), None) => {
                        if pre[*lo] > suf[*hi] {
                            pre[*lo]
                        } else {
                            suf[*hi]
                        }
                    }
                    _ => suf[0],
                };
                let zeros_lose = zmax + l0 < best;
                if !zeros_lose {
                    (best, arg) = match threshold {
                        None => best_in_range(&prev, lt, 0, n),
                        Some(_) => best_in_list(&prev, lt, &cands[j]),
                    };
                }
            }
            // `n_states` fits in u32: the `n_states x n_states` transition
            // matrix is in memory.
            row[j] = arg.unwrap_or(0) as u32;
            cur[j] = col[j] + best;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let mut states = vec![0usize; n_steps];
    states[n_steps - 1] = argmax_first(prev.iter().copied());
    for t in (0..n_steps - 1).rev() {
        states[t] = ptr[(t + 1) * n + states[t + 1]] as usize;
    }
    let log_prob = prev[states[n_steps - 1]];
    Ok(ViterbiResult { states, log_prob })
}

/// `ln(transition + tiny)`, transposed and flattened for [`viterbi_core`].
fn log_transition_t<F: AudioFloat>(transition: ArrayView2<F>) -> Vec<F> {
    let tiny = tiny::<F>();
    let t = transition.t();
    t.iter().map(|&v| (v + tiny).ln()).collect()
}
/// Viterbi decoding from observation likelihoods.
///
/// Computes the most likely hidden-state sequence of an HMM given
/// `prob[s, t] = P[observation t | state s]`, the transition matrix and the
/// initial distribution, in the log domain (each probability is clipped as
/// `ln(p + tiny)`, where `tiny` is the smallest positive normal value of `F`).
/// Equivalent to `librosa.sequence.viterbi(prob, transition, p_init=...,
/// return_logp=True, transition_min_prob=...)`.
///
/// The recursion runs in the element type `F` of `prob` (`f32` or `f64`).
/// librosa accumulates the path log-likelihoods in `float64` for every input
/// type; with `f32` input this function accumulates in `f32`, so paths whose
/// log-likelihoods differ by less than the `f32` resolution of the running
/// totals (about `1e-7` relative) may be decoded differently.
///
/// # Arguments
///
/// - `prob` - `(n_states, n_steps)` likelihoods in `[0, 1]`.
/// - `transition` - `(n_states, n_states)` row-stochastic matrix,
///   `transition[i, j] = P[i -> j]`. Rows must sum to 1 (within `numpy.allclose`
///   tolerance).
/// - `p_init` - Initial distribution (`n_states`, sums to 1); `None` = uniform.
/// - `transition_min_prob` - If `Some(p)` with `p > 0`, only transitions with
///   probability `>= p` are searched (faster, approximate). `None` or `0` is
///   an exact search.
///
/// # Returns
///
/// The state sequence and its log-likelihood. Ties are broken towards the
/// lowest state index, exactly as librosa's `argmax`.
///
/// # Errors
///
/// [`ParameterError`] if shapes mismatch, `prob` is empty or outside `[0, 1]`,
/// `transition` / `p_init` are not valid distributions, or the threshold
/// leaves a state without predecessors.
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::viterbi;
/// use ndarray::array;
///
/// // The Wikipedia "healthy / fever" example.
/// let p_init = array![0.6_f64, 0.4];
/// let p_emit = array![[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]];
/// let p_trans = array![[0.7, 0.3], [0.4, 0.6]];
/// let r = viterbi(p_emit.view(), p_trans.view(), Some(p_init.view()), None).unwrap();
/// assert_eq!(r.states, vec![0, 0, 1]);
/// assert!((r.log_prob - (-4.19173690823075)).abs() < 1e-12);
/// ```
///
/// # References
///
/// A. Viterbi, "Error bounds for convolutional codes and an asymptotically
/// optimum decoding algorithm", IEEE Trans. Inf. Theory 13(2), 1967.
pub fn viterbi<F: AudioFloat>(
    prob: ArrayView2<F>,
    transition: ArrayView2<F>,
    p_init: Option<ArrayView1<F>>,
    transition_min_prob: Option<f64>,
) -> AudioSampleResult<ViterbiResult<F>> {
    check_prob_shape(prob)?;
    let n_states = prob.nrows();
    check_square_transition(transition, n_states)?;
    if prob.iter().any(|&v| !(F::zero()..=F::one()).contains(&v)) {
        return err("prob", "probability values must be between 0 and 1");
    }
    let tiny = tiny::<F>();
    let log_p_init: Vec<F> = match p_init {
        Some(p) => {
            check_distribution("p_init", p, n_states)?;
            p.iter().map(|&v| (v + tiny).ln()).collect()
        }
        None => vec![(F::one() / F::of_usize(n_states) + tiny).ln(); n_states],
    };
    let threshold = log_threshold(transition_min_prob)?;
    let log_trans_t = log_transition_t(transition);
    let fill = |t: usize, col: &mut [F]| {
        for (c, &p) in col.iter_mut().zip(prob.column(t)) {
            *c = (p + tiny).ln();
        }
    };
    let (n_states, n_steps) = prob.dim();
    viterbi_core(
        n_states,
        n_steps,
        fill,
        &log_trans_t,
        &log_p_init,
        threshold,
    )
}

/// Viterbi decoding from discriminative (posterior) state predictions.
///
/// `prob[s, t] = P[state s | observation t]` (each column a distribution) is
/// turned into a scaled likelihood by dividing by the marginal `p_state`,
/// then decoded as in [`viterbi`] (in the element type `F` of `prob`).
/// Equivalent to `librosa.sequence.viterbi_discriminative(...,
/// return_logp=True)`.
///
/// # Arguments
///
/// - `prob` - `(n_states, n_steps)`, non-negative, each column sums to 1.
/// - `transition` - `(n_states, n_states)` row-stochastic matrix.
/// - `p_state` - Marginal state distribution; `None` = uniform.
/// - `p_init` - Initial distribution; `None` = uniform.
/// - `transition_min_prob` - See [`viterbi`].
///
/// # Returns
///
/// State sequence and its (unnormalised) log-likelihood.
///
/// # Errors
///
/// [`ParameterError`] for shape mismatches or invalid distributions.
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::{transition_loop, viterbi_discriminative};
/// use ndarray::array;
///
/// // Frame-wise posteriors flicker to state 1 at t = 2; a sticky HMM smooths it out.
/// let prob = array![[0.9, 0.8, 0.4, 0.8, 0.9], [0.1, 0.2, 0.6, 0.2, 0.1]];
/// let trans = transition_loop(2, &[0.95]).unwrap();
/// let r = viterbi_discriminative(prob.view(), trans.view(), None, None, None).unwrap();
/// assert_eq!(r.states, vec![0, 0, 0, 0, 0]);
/// ```
pub fn viterbi_discriminative<F: AudioFloat>(
    prob: ArrayView2<F>,
    transition: ArrayView2<F>,
    p_state: Option<ArrayView1<F>>,
    p_init: Option<ArrayView1<F>>,
    transition_min_prob: Option<f64>,
) -> AudioSampleResult<ViterbiResult<F>> {
    check_prob_shape(prob)?;
    let n_states = prob.nrows();
    check_square_transition(transition, n_states)?;
    for col in prob.columns() {
        if col.iter().any(|&v| !v.is_finite() || v < F::zero()) || !allclose(sum_f64(&col), 1.0) {
            return err("prob", "each column must be non-negative and sum to 1");
        }
    }
    let uniform = || Array1::from_elem(n_states, F::one() / F::of_usize(n_states));
    let p_state = match p_state {
        Some(p) => {
            check_distribution("p_state", p, n_states)?;
            p.to_owned()
        }
        None => uniform(),
    };
    let p_init = match p_init {
        Some(p) => {
            check_distribution("p_init", p, n_states)?;
            p.to_owned()
        }
        None => uniform(),
    };
    let threshold = log_threshold(transition_min_prob)?;
    let tiny = tiny::<F>();
    let log_p_init: Vec<F> = p_init.iter().map(|&v| (v + tiny).ln()).collect();
    let log_trans_t = log_transition_t(transition);
    let log_marginal: Vec<F> = p_state.iter().map(|&v| (v + tiny).ln()).collect();
    let fill = |t: usize, col: &mut [F]| {
        for ((c, &p), &m) in col.iter_mut().zip(prob.column(t)).zip(&log_marginal) {
            *c = (p + tiny).ln() - m;
        }
    };
    let (n_states, n_steps) = prob.dim();
    viterbi_core(
        n_states,
        n_steps,
        fill,
        &log_trans_t,
        &log_p_init,
        threshold,
    )
}

/// Viterbi decoding for multi-label (independent binary) state activations.
///
/// Each of the `n_states` rows of `prob` is treated as an independent
/// two-state (off/on) HMM with `P[on | observation t] = prob[s, t]` and decoded
/// with [`viterbi_discriminative`] (in the element type `F` of `prob`). Equivalent to
/// `librosa.sequence.viterbi_binary(..., return_logp=True)`. Typical use:
/// smoothing frame-wise voice-activity or note-activity probabilities.
///
/// # Arguments
///
/// - `prob` - `(n_states, n_steps)` activation probabilities in `[0, 1]`.
/// - `transition` - `(1, 2, 2)` (shared by all states) or `(n_states, 2, 2)`;
///   each `2 x 2` block is row-stochastic (`[off, on]` order). A plain `2 x 2`
///   matrix `t` can be passed as `t.view().insert_axis(Axis(0))`.
/// - `p_state` - Marginal `P[on]` per state (`n_states` values in `[0, 1]`);
///   `None` = 0.5.
/// - `p_init` - Initial `P[on]` per state; `None` = 0.5.
/// - `transition_min_prob` - See [`viterbi`].
///
/// # Returns
///
/// A `(n_states, n_steps)` `bool` matrix (`true` = on) and per-state
/// log-likelihoods.
///
/// # Errors
///
/// [`ParameterError`] for shape mismatches or invalid probabilities.
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::{transition_loop, viterbi_binary};
/// use ndarray::{array, Axis};
///
/// let prob = array![[0.1, 0.2, 0.9, 0.3, 0.8, 0.9, 0.9, 0.2]];
/// let trans = transition_loop(2, &[0.8]).unwrap();
/// let r = viterbi_binary(prob.view(), trans.view().insert_axis(Axis(0)), None, None, None)
///     .unwrap();
/// let on: Vec<bool> = [0, 0, 1, 1, 1, 1, 1, 0].iter().map(|&v| v == 1).collect();
/// assert_eq!(r.states.row(0).to_vec(), on);
/// ```
///
/// # Differences from librosa
///
/// librosa returns the states as `uint16` `0`/`1`; this function returns
/// `bool` (one byte per entry).
pub fn viterbi_binary<F: AudioFloat>(
    prob: ArrayView2<F>,
    transition: ArrayView3<F>,
    p_state: Option<ArrayView1<F>>,
    p_init: Option<ArrayView1<F>>,
    transition_min_prob: Option<f64>,
) -> AudioSampleResult<ViterbiBinaryResult<F>> {
    viterbi_binary_core(prob, transition, p_state, p_init, transition_min_prob)
}

/// The body of [`viterbi_binary`].
fn viterbi_binary_core<F: AudioFloat>(
    prob: ArrayView2<F>,
    transition: ArrayView3<F>,
    p_state: Option<ArrayView1<F>>,
    p_init: Option<ArrayView1<F>>,
    transition_min_prob: Option<f64>,
) -> AudioSampleResult<ViterbiBinaryResult<F>> {
    check_prob_shape(prob)?;
    let (n_states, n_steps) = prob.dim();
    let (nt, r, c) = transition.dim();
    if (r, c) != (2, 2) || (nt != 1 && nt != n_states) {
        return err(
            "transition",
            format!(
                "shape {:?} must be (1, 2, 2) or (n_states, 2, 2) = ({n_states}, 2, 2)",
                transition.dim()
            ),
        );
    }
    for t in transition.outer_iter() {
        check_stochastic_rows(t)?;
    }
    let unit = F::zero()..=F::one();
    if prob.iter().any(|v| !unit.contains(v)) {
        return err("prob", "probability values must be between 0 and 1");
    }
    let per_state = |name: &str, p: Option<ArrayView1<F>>| -> AudioSampleResult<Array1<F>> {
        match p {
            None => Ok(Array1::from_elem(n_states, F::of(0.5))),
            Some(p) => {
                if p.len() != n_states || p.iter().any(|v| !unit.contains(v)) {
                    return err(name, format!("must have {n_states} values, each in [0, 1]"));
                }
                Ok(p.to_owned())
            }
        }
    };
    let p_state = per_state("p_state", p_state)?;
    let p_init = per_state("p_init", p_init)?;

    let mut states = Array2::<bool>::from_elem((n_states, n_steps), false);
    let mut logp = Array1::<F>::zeros(n_states);
    let mut prob_bin = Array2::<F>::zeros((2, n_steps));
    for s in 0..n_states {
        for t in 0..n_steps {
            prob_bin[[0, t]] = F::one() - prob[[s, t]];
            prob_bin[[1, t]] = prob[[s, t]];
        }
        let ps = ndarray::arr1(&[F::one() - p_state[s], p_state[s]]);
        let pi = ndarray::arr1(&[F::one() - p_init[s], p_init[s]]);
        let trans = transition.index_axis(Axis(0), if nt == 1 { 0 } else { s });
        let r = viterbi_discriminative(
            prob_bin.view(),
            trans,
            Some(ps.view()),
            Some(pi.view()),
            transition_min_prob,
        )?;
        for (t, &v) in r.states.iter().enumerate() {
            states[[s, t]] = v != 0;
        }
        logp[s] = r.log_prob;
    }
    Ok(ViterbiBinaryResult {
        states,
        log_prob: logp,
    })
}

// ---------------------------------------------------------------------------
// Transition matrices
// ---------------------------------------------------------------------------

/// Uniform transition matrix: every transition has probability `1 / n_states`.
///
/// Equivalent to `librosa.sequence.transition_uniform`. The entries are
/// computed in `f64` and rounded to `F` (`f32` or `f64`, usually inferred from
/// the decoder call) once.
///
/// # Errors
///
/// [`ParameterError`] if `n_states == 0` or the `n_states x n_states`
/// matrix would exceed 2^28 elements.
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::transition_uniform;
///
/// let t = transition_uniform::<f64>(4).unwrap();
/// assert!(t.iter().all(|&p| p == 0.25));
/// ```
pub fn transition_uniform<F: AudioFloat>(n_states: usize) -> AudioSampleResult<Array2<F>> {
    if n_states == 0 {
        return err("n_states", "must be a positive integer");
    }
    check_n_states(n_states)?;
    Ok(Array2::from_elem(
        (n_states, n_states),
        F::of(1.0 / n_states as f64),
    ))
}

/// Rejects transition matrices too large to allocate (`n_states**2` above
/// [`MAX_TABLE_ELEMENTS`](crate::utils::MAX_TABLE_ELEMENTS)).
fn check_n_states(n_states: usize) -> AudioSampleResult<()> {
    let n = n_states as f64;
    crate::utils::check_derived_len(
        "n_states",
        "a transition matrix",
        n * n,
        crate::utils::MAX_TABLE_ELEMENTS,
    )
    .map(|_| ())
}

fn per_state_probs(n_states: usize, prob: &[f64]) -> AudioSampleResult<Vec<f64>> {
    if n_states < 2 {
        return err("n_states", "must be an integer > 1");
    }
    check_n_states(n_states)?;
    let p = match prob.len() {
        1 => vec![prob[0]; n_states],
        n if n == n_states => prob.to_vec(),
        n => {
            return err(
                "prob",
                format!("must have length 1 or n_states = {n_states}, got {n}"),
            );
        }
    };
    if p.iter().any(|v| !(0.0..=1.0).contains(v)) {
        return err("prob", "values must be in the range [0, 1]");
    }
    Ok(p)
}

/// Self-loop transition matrix: stay with probability `prob[i]`, move to any
/// other state uniformly otherwise.
///
/// Equivalent to `librosa.sequence.transition_loop`. Useful to discourage
/// rapid switching (e.g. chord or voicing smoothing). The entries are computed
/// in `f64` and rounded to `F` once.
///
/// # Arguments
///
/// - `n_states` - Number of states, `> 1`.
/// - `prob` - Self-loop probability: one value for all states, or one per
///   state. Each in `[0, 1]`.
///
/// # Errors
///
/// [`ParameterError`] if `n_states < 2`, the matrix would exceed 2^28
/// elements, `prob` has the wrong length or a value outside `[0, 1]`.
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::transition_loop;
///
/// let t = transition_loop::<f64>(3, &[0.5]).unwrap();
/// assert_eq!(t.row(0).to_vec(), vec![0.5, 0.25, 0.25]);
/// ```
pub fn transition_loop<F: AudioFloat>(
    n_states: usize,
    prob: &[f64],
) -> AudioSampleResult<Array2<F>> {
    let p = per_state_probs(n_states, prob)?;
    let mut t = Array2::zeros((n_states, n_states));
    for (i, &pi) in p.iter().enumerate() {
        t.row_mut(i).fill(F::of((1.0 - pi) / (n_states - 1) as f64));
        t[[i, i]] = F::of(pi);
    }
    Ok(t)
}

/// Cyclic transition matrix: stay with probability `prob[i]`, otherwise move
/// to state `(i + 1) % n_states`.
///
/// Equivalent to `librosa.sequence.transition_cycle` (e.g. beat-position or
/// metrical-cycle models). The entries are computed in `f64` and rounded to
/// `F` once.
///
/// # Arguments
///
/// - `n_states` - Number of states, `> 1`.
/// - `prob` - Self-loop probability, one value or one per state, in `[0, 1]`.
///
/// # Errors
///
/// As [`transition_loop`].
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::transition_cycle;
///
/// let t = transition_cycle::<f64>(3, &[0.9]).unwrap();
/// assert_eq!(t.row(2).to_vec(), vec![1.0 - 0.9, 0.0, 0.9]);
/// ```
pub fn transition_cycle<F: AudioFloat>(
    n_states: usize,
    prob: &[f64],
) -> AudioSampleResult<Array2<F>> {
    let p = per_state_probs(n_states, prob)?;
    let mut t = Array2::zeros((n_states, n_states));
    for (i, &pi) in p.iter().enumerate() {
        t[[i, (i + 1) % n_states]] = F::of(1.0 - pi);
        t[[i, i]] = F::of(pi);
    }
    Ok(t)
}

/// Window shape for [`transition_local`] (symmetric, as
/// `scipy.signal.get_window(name, width, fftbins=False)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransitionWindow {
    /// `scipy.signal.windows.triang` (librosa's default `'triangle'`).
    #[default]
    Triangle,
    /// Symmetric Hann window. Widths 1 and 2 give degenerate rows (2 is all
    /// zeros and is rejected).
    Hann,
    /// Symmetric Hamming window.
    Hamming,
    /// Symmetric Blackman window.
    Blackman,
    /// Rectangular window (uniform over the neighbourhood).
    Boxcar,
}

impl TransitionWindow {
    /// Symmetric window of length `m`.
    fn values(self, m: usize) -> Vec<f64> {
        if m <= 1 {
            return vec![1.0; m];
        }
        let cosine = |a: &[f64]| -> Vec<f64> {
            let step = 2.0 * std::f64::consts::PI / (m - 1) as f64;
            (0..m)
                .map(|i| {
                    let fac = if i == m - 1 {
                        std::f64::consts::PI
                    } else {
                        i as f64 * step - std::f64::consts::PI
                    };
                    let mut w = 0.0;
                    for (k, &ak) in a.iter().enumerate() {
                        w += ak * (k as f64 * fac).cos();
                    }
                    w
                })
                .collect()
        };
        match self {
            Self::Boxcar => vec![1.0; m],
            Self::Hann => cosine(&[0.5, 0.5]),
            Self::Hamming => cosine(&[0.54, 0.46]),
            Self::Blackman => cosine(&[0.42, 0.50, 0.08]),
            Self::Triangle => {
                let half: Vec<f64> = (1..=m.div_ceil(2))
                    .map(|n| {
                        let n = n as f64;
                        if m.is_multiple_of(2) {
                            (2.0 * n - 1.0) / m as f64
                        } else {
                            2.0 * n / (m as f64 + 1.0)
                        }
                    })
                    .collect();
                let mut w = half.clone();
                if m.is_multiple_of(2) {
                    w.extend(half.iter().rev());
                } else {
                    w.extend(half.iter().rev().skip(1));
                }
                w
            }
        }
    }
}

/// Localised transition matrix: from state `i`, transitions are allowed only
/// to states within `width[i] / 2` of `i`, weighted by a window.
///
/// Equivalent to `librosa.sequence.transition_local(n_states, width,
/// window=..., wrap=...)`. Typical for pitch tracking (pYIN), where the
/// pitch may only move a few bins per frame. Each row is computed and
/// normalised in `f64` and rounded to `F` once.
///
/// # Arguments
///
/// - `n_states` - Number of states, `> 1`.
/// - `width` - Neighbourhood width (one value, or one per state), each in
///   `1..=n_states`.
/// - `window` - Window shape centred on the current state.
/// - `wrap` - If true, neighbourhoods wrap around the ends (circular state
///   space, e.g. chroma); otherwise they are truncated at the edges.
///
/// # Returns
///
/// A row-stochastic `(n_states, n_states)` matrix.
///
/// # Errors
///
/// [`ParameterError`] if `n_states < 2`, the matrix would exceed 2^28
/// elements, `width` has the wrong length or a
/// value outside `1..=n_states`, or a row has zero total weight (e.g. a
/// width-2 Hann window).
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::{transition_local, TransitionWindow};
///
/// let t = transition_local::<f64>(5, &[3], TransitionWindow::Triangle, false).unwrap();
/// // Interior state 2 moves to 1, 2 or 3 with weights 1/4, 1/2, 1/4.
/// assert_eq!(t.row(2).to_vec(), vec![0.0, 0.25, 0.5, 0.25, 0.0]);
/// // Edge state 0 is truncated and renormalised.
/// assert!((t[[0, 0]] - 2.0 / 3.0).abs() < 1e-15);
/// ```
pub fn transition_local<F: AudioFloat>(
    n_states: usize,
    width: &[usize],
    window: TransitionWindow,
    wrap: bool,
) -> AudioSampleResult<Array2<F>> {
    if n_states < 2 {
        return err("n_states", "must be an integer > 1");
    }
    check_n_states(n_states)?;
    let widths = match width.len() {
        1 => vec![width[0]; n_states],
        n if n == n_states => width.to_vec(),
        n => {
            return err(
                "width",
                format!("must have length 1 or n_states = {n_states}, got {n}"),
            );
        }
    };
    let mut t = Array2::<F>::zeros((n_states, n_states));
    for (i, &w) in widths.iter().enumerate() {
        if w < 1 || w > n_states {
            return Err(ParameterError::out_of_range(
                "width",
                w,
                1,
                n_states,
                "each width must be at least 1 and at most n_states",
            )
            .into());
        }
        // pad_center(window, n_states), then roll by n_states // 2 + i + 1.
        let win = window.values(w);
        let lpad = (n_states - w) / 2;
        let mut padded = vec![0.0; n_states];
        padded[lpad..lpad + w].copy_from_slice(&win);
        let shift = (n_states / 2 + i + 1) % n_states;
        let mut row = vec![0.0; n_states];
        for (r, &v) in padded.iter().enumerate() {
            row[(r + shift) % n_states] = v;
        }
        if !wrap {
            let hi = n_states.min(i + w / 2 + 1);
            row[hi..].fill(0.0);
            let lo = i.saturating_sub(w / 2);
            row[..lo].fill(0.0);
        }
        let sum: f64 = row.iter().sum();
        if sum.is_nan() || sum <= 0.0 {
            return err(
                "window",
                format!("the transition row for state {i} has zero total weight"),
            );
        }
        for (j, v) in row.into_iter().enumerate() {
            t[[i, j]] = F::of(v / sum);
        }
    }
    Ok(t)
}

// ---------------------------------------------------------------------------
// Recurrence / cross-similarity
// ---------------------------------------------------------------------------

/// How a recurrence / cross-similarity matrix encodes links.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecurrenceMode {
    /// `1.0` for a link, `0.0` otherwise (librosa default; librosa returns `bool`).
    #[default]
    Connectivity,
    /// The distance between the linked frames.
    Distance,
    /// `exp(-distance / bandwidth)` (a Gaussian-like kernel on the distance).
    Affinity,
}

/// Bandwidth for [`RecurrenceMode::Affinity`] (librosa's `bandwidth`).
///
/// The `*K*` heuristics follow librosa's `__affinity_bandwidth`, using the
/// distances from each frame to its `k` nearest linked neighbours.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Bandwidth {
    /// Median over frames of the distance to the k-th nearest neighbour
    /// (librosa default, `None` / `'med_k_scalar'`).
    #[default]
    MedKScalar,
    /// Per link: mean of both frames' k-th-neighbour distances (`'mean_k'`).
    MeanK,
    /// Per link: geometric mean of both frames' k-th-neighbour distances (`'gmean_k'`).
    GmeanK,
    /// Per link: mean of both frames' average distance to their k nearest
    /// neighbours (`'mean_k_avg'`).
    MeanKAvg,
    /// Per link: geometric mean of the average k-nearest distances (`'gmean_k_avg'`).
    GmeanKAvg,
    /// Per link: mean of both frames' average k-nearest distances and the
    /// link's own distance (`'mean_k_avg_and_pair'`).
    MeanKAvgAndPair,
    /// A fixed positive bandwidth (rounded to the element type of the data).
    Scalar(f64),
    /// A per-link bandwidth matrix, all entries `> 0`, indexed
    /// `[query_frame, neighbour_frame]` as in librosa, i.e. the **transpose**
    /// of the output layout (`(t, t)` for a recurrence matrix, `(n, n_ref)`
    /// for a cross-similarity matrix). Entries are rounded to the element type
    /// of the data.
    Matrix(Array2<f64>),
}

/// Configuration for [`recurrence_matrix`].
///
/// [`RecurrenceConfig::librosa`] gives librosa's defaults; `Default` is
/// currently the same.
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::{RecurrenceConfig, RecurrenceMode};
///
/// let cfg = RecurrenceConfig::new()
///     .with_mode(RecurrenceMode::Affinity)
///     .with_sym(true)
///     .with_width(3);
/// assert_eq!(cfg.k, None);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrenceConfig {
    /// Number of nearest neighbours per frame. `None` = librosa's
    /// `2 * ceil(sqrt(t - 2 * width + 1))`. Must be `>= 1` when given.
    pub k: Option<usize>,
    /// Frames closer than `width` in time are never linked (suppresses the
    /// trivial near-diagonal). `1 <= width < (t - 1) / 2`. Default 1 (librosa's
    /// default).
    pub width: usize,
    /// Distance between feature vectors.
    pub metric: DistanceMetric,
    /// Keep only mutual links (`i` in the neighbours of `j` and vice versa).
    pub sym: bool,
    /// Output encoding.
    pub mode: RecurrenceMode,
    /// Bandwidth for [`RecurrenceMode::Affinity`].
    pub bandwidth: Bandwidth,
    /// Link each frame to itself (librosa's `self`): 1 in connectivity mode,
    /// affinity 1 in affinity mode; no effect in distance mode.
    pub self_links: bool,
    /// Link every pair (no k-NN sparsification, no width band) in distance /
    /// affinity mode (librosa's `full`).
    pub full: bool,
}

impl Default for RecurrenceConfig {
    fn default() -> Self {
        Self {
            k: None,
            width: 1,
            metric: DistanceMetric::Euclidean,
            sym: false,
            mode: RecurrenceMode::Connectivity,
            bandwidth: Bandwidth::MedKScalar,
            self_links: false,
            full: false,
        }
    }
}

impl RecurrenceConfig {
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// librosa 1.0's defaults for `librosa.segment.recurrence_matrix`.
    ///
    /// Currently the same as [`Default`].
    #[must_use]
    pub fn librosa() -> Self {
        Self::default()
    }

    /// Sets [`RecurrenceConfig::k`].
    #[must_use]
    pub fn with_k(mut self, k: Option<usize>) -> Self {
        self.k = k;
        self
    }

    /// Sets [`RecurrenceConfig::width`].
    #[must_use]
    pub fn with_width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Sets [`RecurrenceConfig::metric`].
    #[must_use]
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Sets [`RecurrenceConfig::sym`].
    #[must_use]
    pub fn with_sym(mut self, sym: bool) -> Self {
        self.sym = sym;
        self
    }

    /// Sets [`RecurrenceConfig::mode`].
    #[must_use]
    pub fn with_mode(mut self, mode: RecurrenceMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets [`RecurrenceConfig::bandwidth`].
    #[must_use]
    pub fn with_bandwidth(mut self, bandwidth: Bandwidth) -> Self {
        self.bandwidth = bandwidth;
        self
    }

    /// Sets [`RecurrenceConfig::self_links`].
    #[must_use]
    pub fn with_self_links(mut self, self_links: bool) -> Self {
        self.self_links = self_links;
        self
    }

    /// Sets [`RecurrenceConfig::full`].
    #[must_use]
    pub fn with_full(mut self, full: bool) -> Self {
        self.full = full;
        self
    }

    /// Checks the configuration for a sequence of `n_frames` frames.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `width` is outside `1..(n_frames - 1) / 2`,
    /// `k == Some(0)`, or a scalar bandwidth is not positive.
    pub fn validate(&self, n_frames: usize) -> AudioSampleResult<()> {
        let max_w = n_frames.saturating_sub(1) / 2;
        if self.width < 1 || self.width >= max_w {
            return Err(ParameterError::out_of_range(
                "width",
                self.width,
                1,
                max_w.saturating_sub(1),
                format!("width must be at least 1 and less than (n_frames - 1) / 2 = {max_w}"),
            )
            .into());
        }
        if self.k == Some(0) {
            return err("k", "must be at least 1");
        }
        validate_bandwidth(&self.bandwidth)
    }
}

fn validate_bandwidth(b: &Bandwidth) -> AudioSampleResult<()> {
    match b {
        Bandwidth::Scalar(v) if !v.is_finite() || *v <= 0.0 => Err(ParameterError::out_of_range(
            "bandwidth",
            v,
            0.0,
            f64::INFINITY,
            "a scalar bandwidth must be strictly positive and finite",
        )
        .into()),
        Bandwidth::Matrix(m) if m.iter().any(|v| v.is_nan() || *v <= 0.0) => err(
            "bandwidth",
            "all bandwidth matrix entries must be strictly positive",
        ),
        _ => Ok(()),
    }
}

/// Configuration for [`cross_similarity`].
///
/// [`CrossSimilarityConfig::librosa`] gives librosa's defaults; `Default` is
/// currently the same.
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::{CrossSimilarityConfig, RecurrenceMode};
///
/// let cfg = CrossSimilarityConfig::new().with_mode(RecurrenceMode::Distance).with_k(Some(3));
/// assert!(cfg.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CrossSimilarityConfig {
    /// Nearest reference frames per query frame. `None` = librosa's
    /// `min(n_ref, 2 * ceil(sqrt(n_ref)))`. Must be `>= 1` when given.
    pub k: Option<usize>,
    /// Distance between feature vectors.
    pub metric: DistanceMetric,
    /// Output encoding.
    pub mode: RecurrenceMode,
    /// Bandwidth for [`RecurrenceMode::Affinity`]. The per-link heuristics
    /// reproduce librosa, which indexes the per-query-frame statistics by the
    /// *reference* frame index; they therefore require `n_ref <= n`.
    pub bandwidth: Bandwidth,
    /// In distance / affinity mode, link each query frame to its
    /// `min(n_ref, n)` nearest reference frames (librosa's `full`).
    pub full: bool,
}

impl Default for CrossSimilarityConfig {
    fn default() -> Self {
        Self {
            k: None,
            metric: DistanceMetric::Euclidean,
            mode: RecurrenceMode::Connectivity,
            bandwidth: Bandwidth::MedKScalar,
            full: false,
        }
    }
}

impl CrossSimilarityConfig {
    /// Returns the default configuration (same as [`Default`]).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// librosa 1.0's defaults for `librosa.segment.cross_similarity`.
    ///
    /// Currently the same as [`Default`].
    #[must_use]
    pub fn librosa() -> Self {
        Self::default()
    }

    /// Sets [`CrossSimilarityConfig::k`].
    #[must_use]
    pub fn with_k(mut self, k: Option<usize>) -> Self {
        self.k = k;
        self
    }

    /// Sets [`CrossSimilarityConfig::metric`].
    #[must_use]
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Sets [`CrossSimilarityConfig::mode`].
    #[must_use]
    pub fn with_mode(mut self, mode: RecurrenceMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets [`CrossSimilarityConfig::bandwidth`].
    #[must_use]
    pub fn with_bandwidth(mut self, bandwidth: Bandwidth) -> Self {
        self.bandwidth = bandwidth;
        self
    }

    /// Sets [`CrossSimilarityConfig::full`].
    #[must_use]
    pub fn with_full(mut self, full: bool) -> Self {
        self.full = full;
        self
    }

    /// Checks the configuration.
    ///
    /// # Errors
    ///
    /// [`ParameterError`] if `k == Some(0)` or a scalar bandwidth is not positive.
    pub fn validate(&self) -> AudioSampleResult<()> {
        if self.k == Some(0) {
            return err("k", "must be at least 1");
        }
        validate_bandwidth(&self.bandwidth)
    }
}

/// Writes the nearest-neighbour distances from `xi` to the columns of `y`
/// into `row`, following sklearn's conventions (librosa delegates to
/// `sklearn.neighbors.NearestNeighbors`): cosine distances of zero vectors
/// are 1 and results are clipped to `[0, 2]`.
fn knn_row<F: AudioFloat>(dist: &RowDistances<'_, F>, xi: ArrayView1<F>, row: &mut [F]) {
    dist.row(xi, row);
    if dist.metric == DistanceMetric::Cosine {
        let two = F::of(2.0);
        for v in row.iter_mut() {
            *v = if v.is_nan() {
                F::one()
            } else if *v < F::zero() {
                F::zero()
            } else if *v > two {
                two
            } else {
                *v
            };
        }
    }
}

/// `f64::total_cmp` for `F` (the conversion to `f64` is exact).
fn total_cmp<F: AudioFloat>(a: F, b: F) -> std::cmp::Ordering {
    a.as_f64().total_cmp(&b.as_f64())
}

/// Puts into `idx` the indices of the `n` nearest entries of `row` (excluding
/// `skip`) under the (distance, index) order, in no particular order.
///
/// The order is total, so the selected set is the same as the first `n`
/// of a full sort.
fn nearest<F: AudioFloat>(row: &[F], n: usize, skip: Option<usize>, idx: &mut Vec<usize>) {
    idx.clear();
    idx.extend((0..row.len()).filter(|&j| Some(j) != skip));
    if n < idx.len() {
        if n > 0 {
            idx.select_nth_unstable_by(n - 1, |&a, &b| total_cmp(row[a], row[b]).then(a.cmp(&b)));
        }
        idx.truncate(n);
    }
}

/// Element of a recurrence / cross-similarity matrix: a float (`0` = no
/// link) or a `bool` connectivity flag.
trait RecCell: Copy + PartialEq {
    fn zero() -> Self;
    /// Sort key of a stored value for [`trim_rows`] (the value itself; every
    /// `bool` link is equal).
    fn key(self) -> f64;
    /// Symmetrisation of a pair of entries (`min`, `&&` for flags).
    fn sym(self, other: Self) -> Self;
}

impl<F: AudioFloat> RecCell for F {
    fn zero() -> Self {
        F::zero()
    }
    fn key(self) -> f64 {
        self.as_f64()
    }
    fn sym(self, other: Self) -> Self {
        self.min(other)
    }
}

impl RecCell for bool {
    fn zero() -> Self {
        false
    }
    fn key(self) -> f64 {
        1.0
    }
    fn sym(self, other: Self) -> Self {
        self && other
    }
}

/// Keep the `k` smallest stored (non-zero) values of each row, ties by column
/// (librosa's stable `argsort` over the row's links).
fn trim_rows<C: RecCell>(rec: &mut Array2<C>, k: usize) {
    let mut links: Vec<usize> = Vec::new();
    for mut row in rec.rows_mut() {
        links.clear();
        links.extend((0..row.len()).filter(|&j| row[j] != C::zero()));
        if links.len() <= k {
            continue;
        }
        links.sort_by(|&a, &b| row[a].key().total_cmp(&row[b].key()).then(a.cmp(&b)));
        for &j in links.iter().skip(k) {
            row[j] = C::zero();
        }
    }
}

/// Transposes a square row-major matrix in place.
fn transpose_square_in_place<F: Copy>(a: &mut [F], n: usize) {
    const B: usize = 32;
    for bi in (0..n).step_by(B) {
        for bj in (bi..n).step_by(B) {
            for i in bi..(bi + B).min(n) {
                let j0 = if bi == bj { i + 1 } else { bj };
                for j in j0..(bj + B).min(n) {
                    a.swap(i * n + j, j * n + i);
                }
            }
        }
    }
}

/// Stored links of a recurrence or cross-similarity matrix, row by row (row
/// = query frame): a dense matrix, where a stored value is non-zero, or
/// [`RecurrenceLinks`]. A stored zero distance is negative.
trait LinkRows<F: AudioFloat> {
    /// `(rows, columns)`.
    fn shape(&self) -> (usize, usize);
    /// Appends the stored values of row `i` to `out`, in column order.
    fn stored_values(&self, i: usize, out: &mut Vec<F>);
    /// Replaces each stored value `v` at `(i, j)` by `f(i, j, v)`, row by row
    /// in column order; unstored cells become zero.
    fn map_stored(
        &mut self,
        f: impl FnMut(usize, usize, F) -> AudioSampleResult<F>,
    ) -> AudioSampleResult<()>;
}

impl<F: AudioFloat> LinkRows<F> for ndarray::ArrayViewMut2<'_, F> {
    fn shape(&self) -> (usize, usize) {
        self.dim()
    }
    fn stored_values(&self, i: usize, out: &mut Vec<F>) {
        out.extend(self.row(i).iter().copied().filter(|&v| v != F::zero()));
    }
    fn map_stored(
        &mut self,
        mut f: impl FnMut(usize, usize, F) -> AudioSampleResult<F>,
    ) -> AudioSampleResult<()> {
        let (t, ncols) = self.dim();
        for i in 0..t {
            for j in 0..ncols {
                let v = self[[i, j]];
                self[[i, j]] = if v == F::zero() {
                    F::zero()
                } else {
                    f(i, j, v)?
                };
            }
        }
        Ok(())
    }
}

/// Per-row statistics over the `k` smallest stored (clipped) values: the
/// `k`-th smallest, the mean, and whether the row has no links.
fn link_row_stats<F: AudioFloat>(rec: &impl LinkRows<F>, k: usize) -> (Vec<F>, Vec<F>, Vec<bool>) {
    let t = rec.shape().0;
    let zero = F::zero();
    let mut dist_to_k = vec![F::nan(); t];
    let mut avg = vec![F::nan(); t];
    let mut empty = vec![false; t];
    let mut vals: Vec<F> = Vec::new();
    for i in 0..t {
        vals.clear();
        rec.stored_values(i, &mut vals);
        if vals.is_empty() {
            empty[i] = true;
            continue;
        }
        for v in &mut vals {
            if *v < zero {
                *v = zero;
            }
        }
        vals.sort_by(|a, b| total_cmp(*a, *b));
        vals.truncate(k);
        dist_to_k[i] = vals[vals.len() - 1];
        avg[i] = vals.iter().copied().sum::<F>() / F::of_usize(vals.len());
    }
    (dist_to_k, avg, empty)
}

/// Converts stored distances to affinities in place (`rec` in librosa's
/// pre-transpose orientation; stored = non-zero, negative = stored zero).
fn apply_affinity<F: AudioFloat>(
    rec: &mut impl LinkRows<F>,
    bandwidth: &Bandwidth,
    k: usize,
) -> AudioSampleResult<()> {
    let zero = F::zero();
    let (t, ncols) = rec.shape();
    let clip = |v: F| if v < zero { zero } else { v };

    match bandwidth {
        Bandwidth::Scalar(bw) => {
            let bw = F::of(*bw);
            rec.map_stored(|_, _, v| Ok((clip(v) / -bw).exp()))
        }
        Bandwidth::Matrix(m) => {
            if m.dim() != (t, ncols) {
                return err(
                    "bandwidth",
                    format!(
                        "bandwidth matrix shape {:?} must be {:?} ([query, neighbour] layout)",
                        m.dim(),
                        (t, ncols)
                    ),
                );
            }
            rec.map_stored(|i, j, v| Ok((clip(v) / -F::of(m[[i, j]])).exp()))
        }
        Bandwidth::MedKScalar => {
            let (dist_to_k, _, _) = link_row_stats(rec, k);
            let mut finite: Vec<F> = dist_to_k.into_iter().filter(|v| !v.is_nan()).collect();
            if finite.is_empty() {
                return err("bandwidth", "cannot estimate bandwidth from an empty graph");
            }
            finite.sort_by(|a, b| total_cmp(*a, *b));
            let n = finite.len();
            let med = if n % 2 == 1 {
                finite[n / 2]
            } else {
                (finite[n / 2 - 1] + finite[n / 2]) / F::of(2.0)
            };
            rec.map_stored(|_, _, v| Ok((clip(v) / -med).exp()))
        }
        heuristic => {
            let (dist_to_k, avg, empty) = link_row_stats(rec, k);
            if let Some(i) = empty.iter().position(|&e| e) {
                return err(
                    "bandwidth",
                    format!("the sample at time point {i} has no neighbors"),
                );
            }
            let stat = match heuristic {
                Bandwidth::MeanK | Bandwidth::GmeanK => &dist_to_k,
                _ => &avg,
            };
            rec.map_stored(|i, j, v| {
                let Some(&sj) = stat.get(j) else {
                    return err(
                        "bandwidth",
                        "per-link bandwidth heuristics need n_ref <= n (librosa indexes \
                         per-query statistics by reference frame)",
                    );
                };
                let si = stat[i];
                let d = clip(v);
                let bw = match heuristic {
                    Bandwidth::MeanK | Bandwidth::MeanKAvg => (si + sj) / F::of(2.0),
                    Bandwidth::GmeanK | Bandwidth::GmeanKAvg => (si * sj).sqrt(),
                    _ => (si + sj + d) / F::of(3.0),
                };
                Ok((d / -bw).exp())
            })
        }
    }
}

/// The links of a recurrence matrix in librosa's pre-transpose orientation,
/// row by row (compressed sparse rows, columns ascending): row `i` holds the
/// neighbours of frame `i`, which after librosa's transpose are column `i`.
/// A stored value may be zero only after an affinity underflows; the dense
/// matrix holds no link there.
#[derive(Debug, Clone)]
pub(crate) struct RecurrenceLinks<F> {
    ptr: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<F>,
}

impl<F: AudioFloat> RecurrenceLinks<F> {
    /// Number of frames.
    pub(crate) fn n_frames(&self) -> usize {
        self.ptr.len() - 1
    }

    /// The neighbours of frame `i` and their values, columns ascending.
    pub(crate) fn row(&self, i: usize) -> (&[usize], &[F]) {
        let r = self.ptr[i]..self.ptr[i + 1];
        (&self.cols[r.clone()], &self.vals[r])
    }

    fn lookup(&self, i: usize, j: usize) -> Option<F> {
        let (cols, vals) = self.row(i);
        cols.binary_search(&j).ok().map(|p| vals[p])
    }
}

impl<F: AudioFloat> LinkRows<F> for RecurrenceLinks<F> {
    fn shape(&self) -> (usize, usize) {
        (self.n_frames(), self.n_frames())
    }
    fn stored_values(&self, i: usize, out: &mut Vec<F>) {
        out.extend(self.row(i).1.iter().copied().filter(|&v| v != F::zero()));
    }
    fn map_stored(
        &mut self,
        mut f: impl FnMut(usize, usize, F) -> AudioSampleResult<F>,
    ) -> AudioSampleResult<()> {
        for i in 0..self.n_frames() {
            for p in self.ptr[i]..self.ptr[i + 1] {
                let v = self.vals[p];
                if v != F::zero() {
                    self.vals[p] = f(i, self.cols[p], v)?;
                }
            }
        }
        Ok(())
    }
}

/// The recurrence links of [`recurrence_matrix_weighted`] without the dense
/// `(t, t)` matrix: memory `O(t k)`. The values are bit-identical to the
/// dense matrix's non-zero entries (`rec[j, i]` is the value at column `j`
/// of row `i`).
///
/// Returns `None` in `full` distance / affinity mode, where every pair is
/// linked and the dense matrix is the smaller representation.
pub(crate) fn recurrence_links<F: AudioFloat>(
    data: ArrayView2<F>,
    cfg: &RecurrenceConfig,
) -> AudioSampleResult<Option<RecurrenceLinks<F>>> {
    if cfg.full && cfg.mode != RecurrenceMode::Connectivity {
        return Ok(None);
    }
    let (t, w, k, bandwidth_k, n_nb) = recurrence_params(data, cfg)?;
    let mode = cfg.mode;
    let one = F::one();
    let link = |d: F| match mode {
        RecurrenceMode::Connectivity => one,
        _ => d,
    };
    // The diagonal entry (always zero before this step).
    let diag = match (cfg.self_links, mode) {
        (true, RecurrenceMode::Connectivity) => one,
        (true, RecurrenceMode::Affinity) => -one,
        _ => F::zero(),
    };
    let per_row = if cfg.full { n_nb } else { k.min(n_nb) } + 1;
    let mut links = RecurrenceLinks {
        ptr: Vec::with_capacity(t + 1),
        cols: Vec::with_capacity(t * per_row),
        vals: Vec::with_capacity(t * per_row),
    };
    links.ptr.push(0);
    {
        let dists = RowDistances::new(data, cfg.metric);
        let mut drow = vec![F::zero(); t];
        let mut idx = Vec::with_capacity(t);
        let mut row: Vec<(usize, F)> = Vec::with_capacity(n_nb + 1);
        for i in 0..t {
            knn_row(&dists, data.column(i), &mut drow);
            nearest(&drow, n_nb, Some(i), &mut idx);
            row.clear();
            // A zero link value is no link (as in the dense matrix); outside
            // `full`, frames within `width` of `i` are dropped.
            row.extend(
                idx.iter()
                    .filter(|&&j| cfg.full || j.abs_diff(i) >= w)
                    .map(|&j| (j, link(drow[j])))
                    .filter(|&(_, v)| v != F::zero()),
            );
            if !cfg.full && row.len() > k {
                // `trim_rows`: the k smallest values, ties by column.
                row.sort_by(|a, b| a.1.key().total_cmp(&b.1.key()).then(a.0.cmp(&b.0)));
                row.truncate(k);
            }
            if diag != F::zero() {
                row.push((i, diag));
            }
            row.sort_unstable_by_key(|&(j, _)| j);
            links.cols.extend(row.iter().map(|&(j, _)| j));
            links.vals.extend(row.iter().map(|&(_, v)| v));
            links.ptr.push(links.cols.len());
        }
    }
    if cfg.sym {
        // Keep the mutual links, with the smaller value (`RecCell::sym`).
        let mut sym = RecurrenceLinks {
            ptr: Vec::with_capacity(t + 1),
            cols: Vec::with_capacity(links.cols.len()),
            vals: Vec::with_capacity(links.cols.len()),
        };
        sym.ptr.push(0);
        for i in 0..t {
            let (cols, vals) = links.row(i);
            for (&j, &a) in cols.iter().zip(vals) {
                let v = if j == i {
                    a
                } else {
                    a.sym(links.lookup(j, i).unwrap_or_else(F::zero))
                };
                if v != F::zero() {
                    sym.cols.push(j);
                    sym.vals.push(v);
                }
            }
            sym.ptr.push(sym.cols.len());
        }
        links = sym;
    }
    if mode == RecurrenceMode::Affinity {
        apply_affinity(&mut links, &cfg.bandwidth, bandwidth_k)?;
    }
    Ok(Some(links))
}

/// Validates `cfg` and `data` and returns `(t, width, k, bandwidth_k, n_nb)`:
/// the frame count, the width, the number of links kept per frame, the `k`
/// for bandwidth estimation, and the number of nearest neighbours searched.
fn recurrence_params<F: AudioFloat>(
    data: ArrayView2<F>,
    cfg: &RecurrenceConfig,
) -> AudioSampleResult<(usize, usize, usize, usize, usize)> {
    let t = data.ncols();
    cfg.validate(t)?;
    check_finite("data", data.iter().copied())?;
    let w = cfg.width;
    let k = cfg
        .k
        .unwrap_or_else(|| 2 * ((t - 2 * w + 1) as f64).sqrt().ceil() as usize);
    let bandwidth_k = k;
    let k = if cfg.full && cfg.mode != RecurrenceMode::Connectivity {
        t
    } else {
        k
    };
    let n_nb = (t - 1).min(k + 2 * w);
    Ok((t, w, k, bandwidth_k, n_nb))
}

/// Recurrence (self-similarity) matrix of a feature sequence, as links.
///
/// Frame `j` is linked to frame `i` if it is among the `k` nearest neighbours
/// of `i` and at least `width` frames away. Equivalent to
/// `librosa.segment.recurrence_matrix(data, ..., mode='connectivity',
/// sparse=False)`, including its `bool` element type (one byte per entry).
/// Distances are computed in the element type `F` of `data`. For distance or
/// affinity weights use [`recurrence_matrix_weighted`].
///
/// # Arguments
///
/// - `data` - Features `(n_features, t)`, finite.
/// - `cfg` - Neighbourhood size, width, metric, symmetry. `cfg.mode` must be
///   [`RecurrenceMode::Connectivity`] (the default).
///
/// # Returns
///
/// A `(t, t)` matrix `R` with `R[j, i]` true iff frame `j` is a neighbour of
/// frame `i` (librosa's orientation).
///
/// # Errors
///
/// [`ParameterError`] for an invalid config (see
/// [`RecurrenceConfig::validate`]), non-finite data, or a `cfg.mode` other
/// than [`RecurrenceMode::Connectivity`].
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::{recurrence_matrix, RecurrenceConfig};
/// use ndarray::Array2;
///
/// let data = Array2::from_shape_fn((1, 12), |(_, t)| (t % 6) as f64 + 0.01 * (t / 6) as f64);
/// let cfg = RecurrenceConfig::new().with_k(Some(2));
/// let rec = recurrence_matrix(data.view(), &cfg).unwrap();
/// assert_eq!(rec.dim(), (12, 12));
/// // Every frame has exactly k = 2 links, and none to itself.
/// for i in 0..12 {
///     assert_eq!(rec.column(i).iter().filter(|&&v| v).count(), 2);
///     assert!(!rec[[i, i]]);
/// }
/// ```
///
/// # Differences from librosa
///
/// - Dense output only.
/// - Brute-force exact neighbour search (librosa uses sklearn, whose
///   brute-force Euclidean distances can differ by round-off).
/// - Among the `k + 2 * width` nearest neighbours outside the width band,
///   librosa keeps the `k` with the *lowest frame index* (its sort over
///   equal link values preserves index order). This function reproduces that
///   behaviour exactly.
pub fn recurrence_matrix<F: AudioFloat>(
    data: ArrayView2<F>,
    cfg: &RecurrenceConfig,
) -> AudioSampleResult<Array2<bool>> {
    if cfg.mode != RecurrenceMode::Connectivity {
        return err(
            "mode",
            "recurrence_matrix returns connectivity links; use recurrence_matrix_weighted for distance or affinity weights",
        );
    }
    let self_links = cfg.self_links;
    let (mut rec, _) = recurrence_core(data, cfg, |_| true, |_| self_links)?;
    let t = rec.nrows();
    transpose_square_in_place(rec.as_slice_mut().expect("standard layout"), t);
    Ok(rec)
}

/// Recurrence matrix with distance or affinity weights.
///
/// The same neighbourhood as [`recurrence_matrix`], but each link stores a
/// value of type `F`: the distance ([`RecurrenceMode::Distance`]),
/// `exp(-distance / bandwidth)` ([`RecurrenceMode::Affinity`]), or `1.0`
/// ([`RecurrenceMode::Connectivity`]). Equivalent to
/// `librosa.segment.recurrence_matrix(data, ..., sparse=False)` in those
/// modes. Computed in the element type `F` of `data`.
///
/// # Arguments
///
/// - `data` - Features `(n_features, t)`, finite.
/// - `cfg` - Neighbourhood size, width, metric, symmetry, output mode.
///
/// # Returns
///
/// A `(t, t)` matrix `R` with `R[j, i] != 0` iff frame `j` is a neighbour of
/// frame `i` (librosa's orientation).
///
/// # Errors
///
/// [`ParameterError`] for an invalid config (see
/// [`RecurrenceConfig::validate`]), non-finite data, or an affinity bandwidth
/// that cannot be estimated.
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::{
///     recurrence_matrix_weighted, RecurrenceConfig, RecurrenceMode,
/// };
/// use ndarray::Array2;
///
/// let data = Array2::from_shape_fn((1, 12), |(_, t)| (t % 6) as f64 + 0.01 * (t / 6) as f64);
/// let cfg = RecurrenceConfig::new()
///     .with_k(Some(1))
///     .with_mode(RecurrenceMode::Distance);
/// let rec = recurrence_matrix_weighted(data.view(), &cfg).unwrap();
/// // Each frame's nearest neighbour is its repetition, 0.01 away.
/// assert!((rec[[6, 0]] - 0.01).abs() < 1e-12);
/// assert_eq!(rec.column(0).iter().filter(|&&v| v != 0.0).count(), 1);
/// ```
///
/// # Differences from librosa
///
/// As [`recurrence_matrix`]. In connectivity mode librosa returns `bool`;
/// this function returns `1.0`/`0.0` (use [`recurrence_matrix`] for `bool`).
pub fn recurrence_matrix_weighted<F: AudioFloat>(
    data: ArrayView2<F>,
    cfg: &RecurrenceConfig,
) -> AudioSampleResult<Array2<F>> {
    let one = F::one();
    let mode = cfg.mode;
    let link = |d: F| match mode {
        RecurrenceMode::Connectivity => one,
        _ => d,
    };
    let diag = |v: F| match (cfg.self_links, mode) {
        (true, RecurrenceMode::Connectivity) => one,
        (true, RecurrenceMode::Affinity) => -one,
        (true, RecurrenceMode::Distance) => v,
        (false, _) => F::zero(),
    };
    let (mut rec, bandwidth_k) = recurrence_core(data, cfg, link, diag)?;
    // Connectivity entries are already 0 or 1, distances need no change.
    if mode == RecurrenceMode::Affinity {
        apply_affinity(&mut rec.view_mut(), &cfg.bandwidth, bandwidth_k)?;
    }
    let t = rec.nrows();
    transpose_square_in_place(rec.as_slice_mut().expect("standard layout"), t);
    Ok(rec)
}

/// The shared part of [`recurrence_matrix`] and [`recurrence_matrix_weighted`]:
/// the `(t, t)` link matrix in librosa's pre-transpose orientation (row =
/// query frame) and the `k` for bandwidth estimation. `link(d)` is the entry
/// stored for a neighbour at distance `d`; `diag(v)` the diagonal entry given
/// its current value.
fn recurrence_core<F: AudioFloat, C: RecCell>(
    data: ArrayView2<F>,
    cfg: &RecurrenceConfig,
    link: impl Fn(F) -> C,
    diag: impl Fn(C) -> C,
) -> AudioSampleResult<(Array2<C>, usize)> {
    let (t, w, k, bandwidth_k, n_nb) = recurrence_params(data, cfg)?;

    // The distances are computed one row at a time, straight into the output
    // (no t x t distance matrix).
    let mut rec = Array2::<C>::from_elem((t, t), C::zero());
    {
        let dists = RowDistances::new(data, cfg.metric);
        let mut drow = vec![F::zero(); t];
        let mut idx = Vec::with_capacity(t);
        for (i, mut row) in rec.rows_mut().into_iter().enumerate() {
            knn_row(&dists, data.column(i), &mut drow);
            nearest(&drow, n_nb, Some(i), &mut idx);
            for &j in &idx {
                row[j] = link(drow[j]);
            }
        }
    }
    if !cfg.full {
        for i in 0..t {
            for j in i.saturating_sub(w - 1)..(i + w).min(t) {
                rec[[i, j]] = C::zero();
            }
        }
        trim_rows(&mut rec, k);
    }
    for i in 0..t {
        rec[[i, i]] = diag(rec[[i, i]]);
    }
    if cfg.sym {
        for i in 0..t {
            for j in i + 1..t {
                let (a, b) = (rec[[i, j]], rec[[j, i]]);
                rec[[i, j]] = a.sym(b);
                rec[[j, i]] = b.sym(a);
            }
        }
    }
    Ok((rec, bandwidth_k))
}

/// Cross-similarity matrix between a query and a reference feature sequence,
/// as links.
///
/// Each query frame is linked to its `k` nearest reference frames.
/// Equivalent to `librosa.segment.cross_similarity(data, data_ref, ...,
/// mode='connectivity', sparse=False)`, including its `bool` element type.
/// For distance or affinity weights use [`cross_similarity_weighted`].
///
/// # Arguments
///
/// - `data` - Query features `(n_features, n)`, finite.
/// - `data_ref` - Reference features `(n_features, n_ref)`, finite.
/// - `cfg` - Neighbourhood size and metric. `cfg.mode` must be
///   [`RecurrenceMode::Connectivity`] (the default).
///
/// # Returns
///
/// A `(n_ref, n)` matrix; entry `[r, q]` is true iff reference frame `r` is
/// linked to query frame `q`.
///
/// # Errors
///
/// [`ParameterError`] if the feature counts differ, either input is empty or
/// non-finite, the config is invalid, or `cfg.mode` is not
/// [`RecurrenceMode::Connectivity`].
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::{cross_similarity, CrossSimilarityConfig};
/// use ndarray::array;
///
/// let query = array![[0.0, 5.0]];
/// let reference = array![[5.1, 0.1, 9.0]];
/// let cfg = CrossSimilarityConfig::new().with_k(Some(1));
/// let x = cross_similarity(query.view(), reference.view(), &cfg).unwrap();
/// assert_eq!(x.dim(), (3, 2));
/// assert_eq!(x.column(0).to_vec(), vec![false, true, false]);
/// assert_eq!(x.column(1).to_vec(), vec![true, false, false]);
/// ```
pub fn cross_similarity<F: AudioFloat>(
    data: ArrayView2<F>,
    data_ref: ArrayView2<F>,
    cfg: &CrossSimilarityConfig,
) -> AudioSampleResult<Array2<bool>> {
    if cfg.mode != RecurrenceMode::Connectivity {
        return err(
            "mode",
            "cross_similarity returns connectivity links; use cross_similarity_weighted for distance or affinity weights",
        );
    }
    Ok(cross_similarity_core(data, data_ref, cfg, |_| true)?.0)
}

/// Cross-similarity matrix with distance or affinity weights.
///
/// The same links as [`cross_similarity`], but each stores a value of type
/// `F`: the distance, the affinity `exp(-distance / bandwidth)`, or `1.0` in
/// connectivity mode. Equivalent to `librosa.segment.cross_similarity(data,
/// data_ref, ..., sparse=False)` in those modes. Computed in the element type
/// `F` of the inputs.
///
/// # Arguments
///
/// - `data` - Query features `(n_features, n)`, finite.
/// - `data_ref` - Reference features `(n_features, n_ref)`, finite.
/// - `cfg` - Neighbourhood size, metric, mode, bandwidth.
///
/// # Returns
///
/// A `(n_ref, n)` matrix; entry `[r, q]` weights the link from reference
/// frame `r` to query frame `q` (`0` where there is none).
///
/// # Errors
///
/// [`ParameterError`] if the feature counts differ, either input is empty or
/// non-finite, the config is invalid, or a bandwidth cannot be computed.
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::{
///     cross_similarity_weighted, CrossSimilarityConfig, RecurrenceMode,
/// };
/// use ndarray::array;
///
/// let query = array![[0.0_f64, 5.0]];
/// let reference = array![[5.1_f64, 0.1, 9.0]];
/// let cfg = CrossSimilarityConfig::new()
///     .with_k(Some(1))
///     .with_mode(RecurrenceMode::Distance);
/// let x = cross_similarity_weighted(query.view(), reference.view(), &cfg).unwrap();
/// assert!((x[[1, 0]] - 0.1).abs() < 1e-12);
/// assert!((x[[0, 1]] - 0.1).abs() < 1e-12);
/// ```
pub fn cross_similarity_weighted<F: AudioFloat>(
    data: ArrayView2<F>,
    data_ref: ArrayView2<F>,
    cfg: &CrossSimilarityConfig,
) -> AudioSampleResult<Array2<F>> {
    let one = F::one();
    let mode = cfg.mode;
    let link = |d: F| match mode {
        RecurrenceMode::Connectivity => one,
        _ => d,
    };
    let (mut out, bandwidth_k) = cross_similarity_core(data, data_ref, cfg, link)?;
    // Connectivity entries are already 0 or 1, distances need no change.
    if mode == RecurrenceMode::Affinity {
        apply_affinity(
            &mut out.view_mut().reversed_axes(),
            &cfg.bandwidth,
            bandwidth_k,
        )?;
    }
    Ok(out)
}

/// The shared part of [`cross_similarity`] and [`cross_similarity_weighted`]:
/// the `(n_ref, n)` link matrix (librosa's output orientation) and the `k`
/// for bandwidth estimation. `link(d)` is the entry stored for a neighbour
/// at distance `d`.
fn cross_similarity_core<F: AudioFloat, C: RecCell>(
    data: ArrayView2<F>,
    data_ref: ArrayView2<F>,
    cfg: &CrossSimilarityConfig,
    link: impl Fn(F) -> C,
) -> AudioSampleResult<(Array2<C>, usize)> {
    cfg.validate()?;
    if data.nrows() != data_ref.nrows() {
        return err(
            "data_ref",
            format!(
                "data has {} features but data_ref has {}",
                data.nrows(),
                data_ref.nrows()
            ),
        );
    }
    let (n, n_ref) = (data.ncols(), data_ref.ncols());
    if n == 0 || n_ref == 0 {
        return err("data / data_ref", "must contain at least one frame");
    }
    check_finite("data", data.iter().copied())?;
    check_finite("data_ref", data_ref.iter().copied())?;
    let k = cfg
        .k
        .unwrap_or_else(|| n_ref.min(2 * (n_ref as f64).sqrt().ceil() as usize));
    let bandwidth_k = k;
    let k = if cfg.full && cfg.mode != RecurrenceMode::Connectivity {
        n
    } else {
        k
    };
    let n_nb = n_ref.min(k);
    // librosa returns the transpose of the (query, reference) matrix built
    // here, so it is built directly in the transposed storage, one row of
    // distances at a time (no distance matrix, no transposed copy). Each row
    // holds at most n_nb <= k links, so librosa's trimming to the k nearest
    // (when `full` is false) never removes any.
    let mut out = Array2::<C>::from_elem((n_ref, n), C::zero());
    {
        let mut xsim = out.view_mut().reversed_axes();
        let dists = RowDistances::new(data_ref, cfg.metric);
        let mut drow = vec![F::zero(); n_ref];
        let mut idx = Vec::with_capacity(n_ref);
        for i in 0..n {
            knn_row(&dists, data.column(i), &mut drow);
            nearest(&drow, n_nb, None, &mut idx);
            for &j in &idx {
                xsim[[i, j]] = link(drow[j]);
            }
        }
    }
    Ok((out, bandwidth_k))
}

/// The `(rows, cols)` matrix `out[i, j] = src[(i - j) mod n, j]` for an
/// `(n, cols)` source, `rows <= n`, `cols <= n` (the shear of
/// [`lag_to_recurrence`]).
///
/// The source is read along diagonals, so the output is filled in blocks of
/// 64 columns: consecutive output rows of a block then read overlapping,
/// cache-resident row segments of the source. Each output element is written
/// once.
fn shear<T: Copy>(src: ArrayView2<T>, rows: usize) -> Array2<T> {
    const B: usize = 64;
    let (n, cols) = src.dim();
    let src = src.as_standard_layout();
    let s = src.as_slice().expect("standard layout");
    let mut out = Array2::<T>::uninit((rows, cols));
    {
        let o = out.as_slice_mut().expect("standard layout");
        for cb in (0..cols).step_by(B) {
            let ce = (cb + B).min(cols);
            for i in 0..rows {
                let orow = &mut o[i * cols + cb..i * cols + ce];
                for (dst, j) in orow.iter_mut().zip(cb..ce) {
                    // Source row (i - j) mod n, with i < n and j < n.
                    let r = if i >= j { i - j } else { i + n - j };
                    *dst = std::mem::MaybeUninit::new(s[r * cols + j]);
                }
            }
        }
    }
    // SAFETY: the loops above write every element (each column block, every
    // row).
    unsafe { out.assume_init() }
}

/// Converts a recurrence matrix into a time-lag matrix.
///
/// `lag[l, i] = rec[(i + l) mod T, i]` with `T = 2t` when `pad` is true (the
/// recurrence matrix is zero-padded below, so negative and positive lags do
/// not wrap onto each other) and `T = t` otherwise. Equivalent to
/// `librosa.segment.recurrence_to_lag(rec, pad=..., axis=-1)` for dense input.
///
/// # Arguments
///
/// - `rec` - Square `(t, t)` recurrence matrix.
/// - `pad` - Zero-pad to `(2t, t)` before shearing (librosa default true).
///
/// # Errors
///
/// [`ParameterError`] if `rec` is not square.
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::{lag_to_recurrence, recurrence_to_lag};
/// use ndarray::Array2;
///
/// let rec = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f64);
/// let lag = recurrence_to_lag(rec.view(), true).unwrap();
/// assert_eq!(lag.dim(), (8, 4));
/// assert_eq!(lag_to_recurrence(lag.view()).unwrap(), rec);
/// ```
pub fn recurrence_to_lag<T: Copy + Default>(
    rec: ArrayView2<T>,
    pad: bool,
) -> AudioSampleResult<Array2<T>> {
    let (r, c) = rec.dim();
    if r != c {
        return err(
            "rec",
            format!("non-square recurrence matrix shape: ({r}, {c})"),
        );
    }
    let n = if pad { 2 * r } else { r };
    // shear(factor = -1): rec[row, col] moves to lag[(row - col) mod n, col];
    // the rows no element moves to stay zero (vec! of a zero-valued
    // primitive is a zeroed allocation, so only the moved elements are
    // written). The source is read in blocks of 64 columns, so that the
    // diagonal writes of consecutive rows hit the same cache lines.
    const B: usize = 64;
    let mut out = vec![T::default(); n * c];
    let dst = |row: usize, col: usize| {
        let d = if row >= col { row - col } else { row + n - col };
        d * c + col
    };
    if let Some(src) = rec.as_slice() {
        for cb in (0..c).step_by(B) {
            let ce = (cb + B).min(c);
            for row in 0..r {
                for col in cb..ce {
                    out[dst(row, col)] = src[row * c + col];
                }
            }
        }
    } else if let Some(src) = rec.t().as_slice() {
        // Fortran order (librosa's recurrence matrices): each column is
        // contiguous; read it in place instead of copying the matrix.
        for cb in (0..c).step_by(B) {
            let ce = (cb + B).min(c);
            for row in 0..r {
                for col in cb..ce {
                    out[dst(row, col)] = src[col * r + row];
                }
            }
        }
    } else {
        let rec = rec.as_standard_layout();
        let src = rec.as_slice().expect("standard layout");
        for cb in (0..c).step_by(B) {
            let ce = (cb + B).min(c);
            for row in 0..r {
                for col in cb..ce {
                    out[dst(row, col)] = src[row * c + col];
                }
            }
        }
    }
    Ok(Array2::from_shape_vec((n, c), out).expect("n * c elements"))
}

/// Converts a time-lag matrix back into a recurrence matrix.
///
/// Inverse of [`recurrence_to_lag`] (padded or not). Equivalent to
/// `librosa.segment.lag_to_recurrence(lag, axis=-1)` for dense input.
///
/// # Arguments
///
/// - `lag` - `(t, t)` or `(2t, t)` lag matrix.
///
/// # Returns
///
/// The `(t, t)` recurrence matrix.
///
/// # Errors
///
/// [`ParameterError`] if the shape is neither `(t, t)` nor `(2t, t)`.
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::lag_to_recurrence;
/// use ndarray::array;
///
/// // Lag 1 everywhere (with wrap-around): each frame recurs one frame later.
/// let lag = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]];
/// let rec = lag_to_recurrence(lag.view()).unwrap();
/// assert_eq!(rec, array![[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
/// ```
pub fn lag_to_recurrence<T: Copy + Default>(lag: ArrayView2<T>) -> AudioSampleResult<Array2<T>> {
    let (n, t) = lag.dim();
    if n != t && n != 2 * t {
        return err(
            "lag",
            format!("invalid lag matrix shape ({n}, {t}): must be (t, t) or (2t, t)"),
        );
    }
    // shear(factor = +1): rec[row, col] = lag[(row - col) mod n, col].
    // rec[row, col] = lag[(row - col) mod n, col].
    Ok(shear(lag, t))
}

// ---------------------------------------------------------------------------
// sync
// ---------------------------------------------------------------------------

/// Built-in aggregation for [`sync`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncAggregate {
    /// Arithmetic mean (librosa default, `np.mean`).
    #[default]
    Mean,
    /// Median (`np.median`; mean of the two middle values for even counts).
    Median,
    /// Maximum (`np.max`).
    Max,
    /// Minimum (`np.min`).
    Min,
    /// Sum (`np.sum`).
    Sum,
}

impl SyncAggregate {
    fn apply<F: AudioFloat>(self, x: ArrayView1<F>) -> F {
        if x.iter().any(|v| v.is_nan()) {
            // numpy reductions propagate NaN.
            return F::nan();
        }
        match self {
            Self::Mean => x.sum() / F::of_usize(x.len()),
            Self::Sum => x.sum(),
            Self::Max => x.iter().copied().fold(F::neg_infinity(), F::max),
            Self::Min => x.iter().copied().fold(F::infinity(), F::min),
            Self::Median => {
                let mut v = x.to_vec();
                v.sort_by(|a, b| total_cmp(*a, *b));
                let n = v.len();
                if n % 2 == 1 {
                    v[n / 2]
                } else {
                    (v[n / 2 - 1] + v[n / 2]) / F::of(2.0)
                }
            }
        }
    }
}

/// Segment boundaries as librosa's `index_to_slice(idx, idx_min=0,
/// idx_max=n, pad=pad)`.
fn sync_boundaries(idx: &[usize], n: usize, pad: bool) -> Vec<usize> {
    // One allocation of the final size (the two padding entries included).
    let mut b = Vec::with_capacity(idx.len() + 2);
    b.extend(idx.iter().map(|&i| if pad { i.min(n) } else { i }));
    if pad {
        b.push(0);
        b.push(n);
    }
    b.retain(|&i| i <= n);
    b.sort_unstable();
    b.dedup();
    b
}

/// Aggregates a feature array between frame boundaries (e.g. beats).
///
/// The last axis of `data` is the frame axis. `idx` lists boundary frames;
/// they are sorted and de-duplicated, and (with `pad`) `0` and `n_frames` are
/// added and out-of-range indices clipped, so that the segments cover all
/// frames. Each segment `[idx[i], idx[i+1])` is reduced to one column.
/// Equivalent to `librosa.util.sync(data, idx, aggregate=..., pad=...)`.
///
/// # Arguments
///
/// - `data` - Array of any dimension `>= 1`, frames on the last axis (e.g. a
///   `(n_features, n_frames)` feature matrix or a
///   `(channels, n_features, n_frames)` stack).
/// - `idx` - Boundary frame indices (any order, duplicates allowed).
/// - `aggregate` - Reduction applied to each segment.
/// - `pad` - Add `0` and `n_frames` to the boundaries (librosa default true).
///   Without padding, indices beyond `n_frames` are dropped.
///
/// # Returns
///
/// `data` with the last axis replaced by one entry per segment, in the element
/// type `F` of `data` (`f32` or `f64`).
///
/// # Errors
///
/// [`ParameterError`] if `data` is zero-dimensional.
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::{sync, SyncAggregate};
/// use ndarray::array;
///
/// let chroma = array![[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
/// let beats = [2, 4];
/// let synced = sync(chroma.view(), &beats, SyncAggregate::Mean, true).unwrap();
/// // Segments [0, 2), [2, 4), [4, 6).
/// assert_eq!(synced, array![[1.5, 3.5, 5.5]]);
/// ```
///
/// # Differences from librosa
///
/// With an empty `idx` and `pad = true` the output has one column (the
/// segment `[0, n_frames)`); librosa treats an empty list as a list of zero
/// slices and returns zero columns. Only frame indices are accepted (librosa
/// also takes a list of slices).
pub fn sync<F, D>(
    data: ArrayView<F, D>,
    idx: &[usize],
    aggregate: SyncAggregate,
    pad: bool,
) -> AudioSampleResult<Array<F, D>>
where
    F: AudioFloat,
    D: Dimension + RemoveAxis,
{
    sync_with(data, idx, pad, |x| aggregate.apply(x))
}

/// [`sync`] with a custom aggregation function.
///
/// `aggregate` receives the frames of one segment for one feature (never
/// empty) and returns the aggregated value.
///
/// # Errors
///
/// [`ParameterError`] if `data` is zero-dimensional.
///
/// # Examples
///
/// ```
/// use audio_samples::operations::sequence::sync_with;
/// use ndarray::array;
///
/// let x = array![[1.0, -2.0, 3.0, -4.0]];
/// // Peak absolute value per segment [0, 2), [2, 4).
/// let peak = sync_with(x.view(), &[2], true, |s| s.fold(0.0, |m: f64, v| m.max(v.abs()))).unwrap();
/// assert_eq!(peak, array![[2.0, 4.0]]);
/// ```
pub fn sync_with<F, D, A>(
    data: ArrayView<F, D>,
    idx: &[usize],
    pad: bool,
    mut aggregate: A,
) -> AudioSampleResult<Array<F, D>>
where
    F: AudioFloat,
    D: Dimension + RemoveAxis,
    A: FnMut(ArrayView1<F>) -> F,
{
    if data.ndim() == 0 {
        return err("data", "must have at least one dimension (frames last)");
    }
    let axis = Axis(data.ndim() - 1);
    // Segment s is [bounds[s], bounds[s + 1]).
    let bounds = sync_boundaries(idx, data.len_of(axis), pad);
    let mut shape = data.raw_dim();
    shape[axis.index()] = bounds.len().saturating_sub(1);
    let mut out = Array::<F, D>::zeros(shape);
    for (s, w) in bounds.windows(2).enumerate() {
        let (a, b) = (w[0], w[1]);
        let seg = data.slice_axis(axis, Slice::from(a..b));
        let out_s = out.index_axis_mut(axis, s);
        Zip::from(out_s)
            .and(seg.lanes(axis))
            .for_each(|o, lane| *o = aggregate(lane));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, ShapeBuilder, array};

    #[test]
    fn huge_transition_matrices_are_parameter_errors() {
        let n = 1usize << 31;
        let is_param = |r: AudioSampleResult<Array2<f64>>| {
            matches!(r, Err(crate::AudioSampleError::Parameter(_)))
        };
        assert!(is_param(transition_uniform(n)));
        assert!(is_param(transition_uniform(usize::MAX)));
        assert!(is_param(transition_loop(n, &[0.5])));
        assert!(is_param(transition_cycle(n, &[0.5])));
        assert!(is_param(transition_local::<f64>(
            n,
            &[3],
            TransitionWindow::default(),
            true
        )));
        assert!(transition_uniform::<f32>(1 << 14).is_ok());
    }

    #[test]
    fn dtw_identical_sequences_have_zero_cost_diagonal_path() {
        let x = array![[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 1.0, 0.0]];
        let r = dtw(x.view(), x.view(), &DtwConfig::default()).unwrap();
        assert_eq!(r.cost[[3, 3]], 0.0);
        assert_eq!(r.path.unwrap(), vec![[3, 3], [2, 2], [1, 1], [0, 0]]);
    }

    #[test]
    fn dtw_hand_checked_cost() {
        // C = [[0, 1], [1, 0], [2, 1]] (1-D euclidean between [0,1,2] and [0,1]).
        let x = array![[0.0, 1.0, 2.0]];
        let y = array![[0.0, 1.0]];
        let r = dtw(x.view(), y.view(), &DtwConfig::default()).unwrap();
        assert_eq!(r.cost, array![[0.0, 1.0], [1.0, 0.0], [3.0, 1.0]]);
        assert_eq!(r.path.unwrap(), vec![[2, 1], [1, 1], [0, 0]]);
        assert_eq!(r.steps[[2, 1]], 2); // vertical step (1, 0)
    }

    #[test]
    fn dtw_errors() {
        let x = array![[0.0, 1.0]];
        let y = array![[0.0, 1.0], [1.0, 1.0]];
        assert!(dtw(x.view(), y.view(), &DtwConfig::default()).is_err());
        let empty = Array2::<f64>::zeros((1, 0));
        assert!(dtw(x.view(), empty.view(), &DtwConfig::default()).is_err());
        let bad = DtwConfig::default().with_weights_add(Some(vec![0.0]));
        assert!(dtw(x.view(), x.view(), &bad).is_err());
        let zero_step = DtwConfig::default().with_step_sizes_sigma(Some(vec![[0, 0]]));
        assert!(zero_step.validate().is_err());
        let band = DtwConfig::default().with_band_rad(-1.0);
        assert!(band.validate().is_err());
        let c = array![[f64::NAN, 1.0]];
        assert!(dtw_with_cost(c.view(), &DtwConfig::default()).is_err());
        // Cosine distance of an all-zero frame is NaN.
        let z = array![[0.0, 1.0], [0.0, 1.0]];
        let cos = DtwConfig::default().with_metric(DistanceMetric::Cosine);
        assert!(dtw(z.view(), z.view(), &cos).is_err());
        // Unreachable end with only (2, 1)/(1, 2) steps.
        let c = Array2::<f64>::ones((2, 2));
        let knight = DtwConfig::default().with_step_sizes_sigma(Some(vec![[2, 1], [1, 2]]));
        assert!(dtw_with_cost(c.view(), &knight).is_err());
    }

    #[test]
    fn dtw_backtracking_validation() {
        let steps = Array2::<usize>::zeros((2, 2));
        assert!(dtw_backtracking(steps.view(), None, false, Some(0)).is_err());
        assert!(dtw_backtracking(steps.view(), None, true, Some(5)).is_err());
        let bad = Array2::<usize>::from_elem((2, 2), 7);
        assert!(dtw_backtracking(bad.view(), None, false, None).is_err());
        let ok = dtw_backtracking(steps.view(), None, false, None).unwrap();
        assert_eq!(ok, vec![[1, 1], [0, 0]]);
    }

    #[test]
    fn fill_off_diagonal_matches_librosa_square() {
        let mut x = Array2::<f64>::zeros((4, 4));
        fill_off_diagonal(&mut x, 0.25, 1.0);
        // radius = round(1.0) = 1: only the main diagonal survives.
        assert_eq!(
            x,
            Array2::from_shape_fn((4, 4), |(i, j)| if i == j { 0.0 } else { 1.0 })
        );
    }

    #[test]
    fn viterbi_validation() {
        let p = array![[0.5, 0.5], [0.5, 0.5]];
        let bad_t = array![[0.5, 0.6], [0.5, 0.5]];
        assert!(viterbi(p.view(), bad_t.view(), None, None).is_err());
        let t = array![[0.5, 0.5], [0.5, 0.5]];
        let bad_p = array![[1.5, 0.5], [0.5, 0.5]];
        assert!(viterbi(bad_p.view(), t.view(), None, None).is_err());
        let bad_init = array![0.2, 0.2];
        assert!(viterbi(p.view(), t.view(), Some(bad_init.view()), None).is_err());
        assert!(viterbi(p.view(), t.view(), None, Some(-1.0)).is_err());
        let empty = Array2::<f64>::zeros((2, 0));
        assert!(viterbi(empty.view(), t.view(), None, None).is_err());
        // Threshold above every transition into a state.
        let tc = array![[1.0, 0.0], [1.0, 0.0]];
        assert!(viterbi(p.view(), tc.view(), None, Some(0.5)).is_err());
        let nan = array![[f64::NAN, 0.5], [0.5, 0.5]];
        assert!(viterbi(nan.view(), t.view(), None, None).is_err());
    }

    #[test]
    fn viterbi_ties_pick_lowest_state() {
        let p = Array2::from_elem((3, 4), 0.5);
        let t = transition_uniform(3).unwrap();
        let r = viterbi(p.view(), t.view(), None, None).unwrap();
        assert_eq!(r.states, vec![0, 0, 0, 0]);
    }

    #[test]
    fn viterbi_binary_shapes_and_validation() {
        let p = array![[0.9, 0.9, 0.1], [0.1, 0.2, 0.9]];
        let t = transition_loop(2, &[0.5]).unwrap();
        let t3 = t.view().insert_axis(Axis(0));
        let r = viterbi_binary(p.view(), t3, None, None, None).unwrap();
        assert_eq!(r.states, array![[true, true, false], [false, false, true]]);
        assert_eq!(r.log_prob.len(), 2);
        let wrong = Array3::<f64>::from_elem((3, 2, 2), 0.5);
        assert!(viterbi_binary(p.view(), wrong.view(), None, None, None).is_err());
        let ps = array![0.5];
        assert!(viterbi_binary(p.view(), t3, Some(ps.view()), None, None).is_err());
    }

    #[test]
    fn transitions_are_row_stochastic() {
        for t in [
            transition_uniform::<f64>(4).unwrap(),
            transition_loop(4, &[0.3]).unwrap(),
            transition_cycle(4, &[0.1, 0.2, 0.3, 0.4]).unwrap(),
            transition_local(6, &[3], TransitionWindow::Hann, true).unwrap(),
            transition_local(6, &[1, 2, 3, 4, 5, 6], TransitionWindow::Boxcar, false).unwrap(),
        ] {
            for row in t.rows() {
                assert!((row.sum() - 1.0).abs() < 1e-12);
            }
        }
        assert!(transition_uniform::<f64>(0).is_err());
        assert!(transition_loop::<f64>(1, &[0.5]).is_err());
        assert!(transition_loop::<f64>(3, &[0.5, 0.5]).is_err());
        assert!(transition_cycle::<f64>(3, &[1.5]).is_err());
        assert!(transition_local::<f64>(4, &[0], TransitionWindow::Triangle, false).is_err());
        assert!(transition_local::<f64>(4, &[5], TransitionWindow::Triangle, false).is_err());
        assert!(transition_local::<f64>(4, &[2], TransitionWindow::Hann, false).is_err());
    }

    #[test]
    fn triangle_window_matches_scipy() {
        assert_eq!(TransitionWindow::Triangle.values(1), vec![1.0]);
        assert_eq!(TransitionWindow::Triangle.values(3), vec![0.5, 1.0, 0.5]);
        assert_eq!(
            TransitionWindow::Triangle.values(4),
            vec![0.25, 0.75, 0.75, 0.25]
        );
    }

    #[test]
    fn rqa_hand_checked() {
        let sim = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let r = rqa(sim.view(), &RqaConfig::default()).unwrap();
        assert_eq!(r.score[[2, 2]], 3.0);
        assert_eq!(r.path.unwrap(), vec![[0, 0], [1, 1], [2, 2]]);
        let zeros = Array2::<f64>::zeros((3, 3));
        let r = rqa(zeros.view(), &RqaConfig::default()).unwrap();
        assert_eq!(r.path.unwrap(), Vec::<[usize; 2]>::new());
        assert!(rqa(sim.view(), &RqaConfig::default().with_gap_onset(-1.0)).is_err());
    }

    #[test]
    fn rqa_column_one_knight_move_backtracks_inside() {
        // Move (-2, -1) into column 1: (0,0) -> (2,1).
        let mut sim = Array2::<f64>::zeros((3, 3));
        sim[[0, 0]] = 1.0;
        sim[[2, 1]] = 1.0;
        let r = rqa(sim.view(), &RqaConfig::default()).unwrap();
        assert_eq!(r.score[[2, 1]], 2.0);
        assert_eq!(r.path.unwrap(), vec![[0, 0], [2, 1]]);
    }

    #[test]
    fn recurrence_validation_and_shapes() {
        let data = Array2::from_shape_fn((2, 10), |(f, t)| (f + t) as f64);
        assert!(recurrence_matrix(data.view(), &RecurrenceConfig::new().with_width(4)).is_err());
        assert!(recurrence_matrix(data.view(), &RecurrenceConfig::new().with_k(Some(0))).is_err());
        let r = recurrence_matrix(data.view(), &RecurrenceConfig::new().with_sym(true)).unwrap();
        assert_eq!(r.dim(), (10, 10));
        assert_eq!(r, r.t());
        let x = cross_similarity(
            data.view(),
            data.slice(ndarray::s![.., ..4]),
            &CrossSimilarityConfig::default(),
        )
        .unwrap();
        assert_eq!(x.dim(), (4, 10));
        assert!(recurrence_to_lag(x.view(), true).is_err());
        assert!(lag_to_recurrence(Array2::<f64>::zeros((3, 2)).view()).is_err());
    }

    #[test]
    fn sync_matches_hand_computation() {
        let x = array![[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]];
        let m = sync(x.view(), &[3, 1, 3], SyncAggregate::Max, true).unwrap();
        assert_eq!(m, array![[1.0, 3.0, 5.0], [5.0, 4.0, 2.0]]);
        let med = sync(x.view(), &[1, 3], SyncAggregate::Median, false).unwrap();
        assert_eq!(med, array![[2.5], [3.5]]);
        let none = sync(x.view(), &[9], SyncAggregate::Mean, false).unwrap();
        assert_eq!(none.dim(), (2, 0));
        let x3 = Array3::<f64>::ones((2, 3, 4));
        let s3 = sync(x3.view(), &[2], SyncAggregate::Sum, true).unwrap();
        assert_eq!(s3, Array3::from_elem((2, 3, 2), 2.0));
    }

    /// The straightforward recursion (the previous implementation), as the
    /// reference for the pruned one in `viterbi_core`.
    fn reference_viterbi<F: AudioFloat>(
        log_prob: &Array2<F>,
        log_trans: &Array2<F>,
        log_p_init: &Array1<F>,
        threshold: Option<F>,
    ) -> ViterbiResult<F> {
        let (n_states, n_steps) = log_prob.dim();
        let mut value = Array2::<F>::zeros((n_steps, n_states));
        let mut ptr = Array2::<usize>::zeros((n_steps, n_states));
        for j in 0..n_states {
            value[[0, j]] = log_prob[[j, 0]] + log_p_init[j];
        }
        let preds: Vec<Vec<usize>> = (0..n_states)
            .map(|j| {
                (0..n_states)
                    .filter(|&k| threshold.is_none_or(|th| log_trans[[k, j]] >= th))
                    .collect()
            })
            .collect();
        for t in 1..n_steps {
            for j in 0..n_states {
                let mut best = F::neg_infinity();
                for &k in &preds[j] {
                    let cost = value[[t - 1, k]] + log_trans[[k, j]];
                    if cost > best {
                        ptr[[t, j]] = k;
                        best = cost;
                    }
                }
                value[[t, j]] = log_prob[[j, t]] + best;
            }
        }
        let mut states = vec![0usize; n_steps];
        states[n_steps - 1] = argmax_first(value.row(n_steps - 1).iter().copied());
        for t in (0..n_steps - 1).rev() {
            states[t] = ptr[[t + 1, states[t + 1]]];
        }
        let log_prob = value[[n_steps - 1, states[n_steps - 1]]];
        ViterbiResult { states, log_prob }
    }

    /// Deterministic uniform values in `[0, 1)`.
    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 11) as f64 / (1u64 << 53) as f64
    }

    fn check_viterbi_matches_reference<F: AudioFloat>() {
        let mut seed = 7u64;
        let tiny = tiny::<F>();
        for &n in &[1usize, 2, 3, 7, 9, 17, 40] {
            for kind in 0..5 {
                // Transition: dense, banded, sparse random, cycle, and one
                // with a zero column (a state nothing can reach).
                let mut t = Array2::<f64>::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        let keep = match kind {
                            0 => true,
                            1 => i.abs_diff(j) <= 2,
                            2 => i == j || lcg(&mut seed) < 0.3,
                            3 => j == i || j == (i + 1) % n,
                            _ => j != n / 2 || n == 1,
                        };
                        if keep {
                            t[[i, j]] = 0.05 + lcg(&mut seed);
                        }
                    }
                    let s: f64 = t.row(i).sum();
                    t.row_mut(i).mapv_inplace(|v| v / s);
                }
                let t = t.mapv(F::of);
                // Likelihoods with exact zeros and tiny values, so that the
                // pruning bound sometimes fails.
                let steps = 60;
                let prob = Array2::from_shape_fn((n, steps), |_| {
                    let u = lcg(&mut seed);
                    F::of(if u < 0.15 {
                        0.0
                    } else if u < 0.25 {
                        1e-30
                    } else {
                        u
                    })
                });
                for thr in [None, Some(1e-3), Some(0.2)] {
                    let lt = t.mapv(|v| (v + tiny).ln());
                    let th = log_threshold::<F>(thr).unwrap();
                    let lp = prob.mapv(|v| (v + tiny).ln());
                    let init = Array1::from_elem(n, (F::one() / F::of_usize(n) + tiny).ln());
                    let got = viterbi(prob.view(), t.view(), None, thr);
                    let feasible =
                        th.is_none_or(|th| (0..n).all(|j| (0..n).any(|k| lt[[k, j]] >= th)));
                    if !feasible {
                        assert!(got.is_err());
                        continue;
                    }
                    let want = reference_viterbi(&lp, &lt, &init, th);
                    let got = got.unwrap();
                    assert_eq!(got.states, want.states, "n={n} kind={kind} thr={thr:?}");
                    assert_eq!(
                        got.log_prob.as_f64().to_bits(),
                        want.log_prob.as_f64().to_bits(),
                        "n={n} kind={kind} thr={thr:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn viterbi_matches_reference_recursion() {
        check_viterbi_matches_reference::<f32>();
        check_viterbi_matches_reference::<f64>();
    }

    /// The previous DTW implementation (padded accumulation, per-pair
    /// distances), as the reference for `dtw_core` / `pairwise`.
    fn reference_dtw<F: AudioFloat>(
        x: Option<(ArrayView2<F>, ArrayView2<F>)>,
        cost: Option<ArrayView2<F>>,
        cfg: &DtwConfig,
    ) -> (Array2<F>, Array2<usize>) {
        let mut c = match (x, cost) {
            (Some((x, y)), _) => {
                let mut out = Array2::zeros((x.ncols(), y.ncols()));
                for (i, xi) in x.columns().into_iter().enumerate() {
                    for (j, yj) in y.columns().into_iter().enumerate() {
                        out[[i, j]] = cfg.metric.distance(xi, yj);
                    }
                }
                if cfg.subseq && x.ncols() > y.ncols() {
                    out = out.reversed_axes().as_standard_layout().into_owned();
                }
                out
            }
            (None, Some(c)) => c.to_owned(),
            _ => unreachable!(),
        };
        let (sigma, w_add, w_mul) = cfg.resolve().unwrap();
        let w_add: Vec<F> = w_add.into_iter().map(F::of).collect();
        let w_mul: Vec<F> = w_mul.into_iter().map(F::of).collect();
        if cfg.global_constraints {
            fill_off_diagonal(&mut c, cfg.band_rad, F::infinity());
        }
        let (n, m) = c.dim();
        let max0 = sigma.iter().map(|s| s[0]).max().unwrap();
        let max1 = sigma.iter().map(|s| s[1]).max().unwrap();
        let mut d = Array2::from_elem((n + max0, m + max1), F::infinity());
        d[[max0, max1]] = c[[0, 0]];
        if cfg.subseq {
            for j in 0..m {
                d[[max0, max1 + j]] = c[[0, j]];
            }
        }
        let mut steps = Array2::<usize>::zeros(d.dim());
        for cur_n in max0..d.nrows() {
            for cur_m in max1..d.ncols() {
                let local = c[[cur_n - max0, cur_m - max1]];
                for (k, s) in sigma.iter().enumerate() {
                    let cur_d = d[[cur_n - s[0], cur_m - s[1]]];
                    let mut cur_c = w_mul[k] * local;
                    cur_c += w_add[k];
                    let cur_cost = cur_d + cur_c;
                    if cur_cost < d[[cur_n, cur_m]] {
                        d[[cur_n, cur_m]] = cur_cost;
                        steps[[cur_n, cur_m]] = k;
                    }
                }
            }
        }
        (
            d.slice(ndarray::s![max0.., max1..]).to_owned(),
            steps.slice(ndarray::s![max0.., max1..]).to_owned(),
        )
    }

    fn bits<F: AudioFloat>(a: &Array2<F>) -> Vec<u64> {
        a.iter().map(|v| v.as_f64().to_bits()).collect()
    }

    fn check_dtw_matches_reference<F: AudioFloat>() {
        let mut seed = 11u64;
        let metrics = [
            DistanceMetric::Euclidean,
            DistanceMetric::SqEuclidean,
            DistanceMetric::Cosine,
            DistanceMetric::Cityblock,
            DistanceMetric::Chebyshev,
        ];
        let cfgs = [
            DtwConfig::new(),
            DtwConfig::new().with_subseq(true),
            DtwConfig::new()
                .with_global_constraints(true)
                .with_band_rad(0.3),
            DtwConfig::new()
                .with_step_sizes_sigma(Some(vec![[2, 1], [1, 2]]))
                .with_weights_add(Some(vec![0.5, 0.25]))
                .with_weights_mul(Some(vec![2.0, 1.5])),
            DtwConfig::new()
                .with_weights_add(Some(vec![0.1, -0.2, 0.0]))
                .with_weights_mul(Some(vec![1.0, 2.0, 0.0])),
            DtwConfig::new()
                .with_subseq(true)
                .with_step_sizes_sigma(Some(vec![[1, 2]])),
        ];
        for &(nx, ny) in &[(1usize, 1usize), (1, 5), (5, 1), (7, 9), (13, 6), (20, 20)] {
            let x = Array2::from_shape_fn((3, nx), |_| F::of(lcg(&mut seed) + 0.01));
            let y = Array2::from_shape_fn((3, ny), |_| F::of(lcg(&mut seed) + 0.01));
            for metric in metrics {
                for cfg in &cfgs {
                    let cfg = cfg.clone().with_metric(metric).with_backtrack(false);
                    let (wd, ws) = reference_dtw(Some((x.view(), y.view())), None, &cfg);
                    let r = dtw(x.view(), y.view(), &cfg).unwrap();
                    assert_eq!(bits(&r.cost), bits(&wd), "{nx}x{ny} {cfg:?}");
                    assert_eq!(r.steps, ws, "{nx}x{ny} {cfg:?}");
                }
            }
            // Precomputed costs with infinities, in Fortran order.
            let mut c = Array2::from_shape_fn((nx, ny).f(), |_| F::of(lcg(&mut seed) - 0.2));
            if nx * ny > 2 {
                c[[nx / 2, ny / 2]] = F::infinity();
            }
            for cfg in &cfgs {
                let cfg = cfg.clone().with_backtrack(false);
                let (wd, ws) = reference_dtw(None, Some(c.view()), &cfg);
                let r = dtw_with_cost(c.view(), &cfg).unwrap();
                assert_eq!(bits(&r.cost), bits(&wd), "cost {nx}x{ny} {cfg:?}");
                assert_eq!(r.steps, ws, "cost {nx}x{ny} {cfg:?}");
            }
        }
    }

    #[test]
    fn dtw_matches_reference_accumulation() {
        check_dtw_matches_reference::<f32>();
        check_dtw_matches_reference::<f64>();
    }

    #[test]
    fn lag_conversions_take_bool_matrices() {
        let rec = Array2::from_shape_fn((5, 5), |(i, j)| (i * 3 + j) % 4 == 0);
        let as_f = rec.mapv(|b| if b { 1.0f32 } else { 0.0 });
        for pad in [false, true] {
            let lag = recurrence_to_lag(rec.view(), pad).unwrap();
            let lag_f = recurrence_to_lag(as_f.view(), pad).unwrap();
            assert_eq!(lag.mapv(|b| if b { 1.0f32 } else { 0.0 }), lag_f);
            assert_eq!(lag_to_recurrence(lag.view()).unwrap(), rec);
            // Fortran-order and strided input give the same lag matrix.
            let mut f_order = Array2::from_elem((5, 5).f(), false);
            f_order.assign(&rec);
            assert_eq!(recurrence_to_lag(f_order.view(), pad).unwrap(), lag);
            let big = Array2::from_shape_fn((5, 10), |(i, j)| j % 2 == 0 && rec[[i, j / 2]]);
            let strided = big.slice(ndarray::s![.., ..;2]);
            assert_eq!(recurrence_to_lag(strided, pad).unwrap(), lag);
        }
    }

    #[test]
    fn bool_recurrence_matches_float_connectivity() {
        let mut seed = 21u64;
        for &t in &[8usize, 17, 40] {
            let data = Array2::from_shape_fn((3, t), |_| (lcg(&mut seed) * 4.0).floor() / 4.0);
            for sym in [false, true] {
                for self_links in [false, true] {
                    for full in [false, true] {
                        for (width, k) in [(1usize, None), (2, Some(3))] {
                            let cfg = RecurrenceConfig::new()
                                .with_sym(sym)
                                .with_self_links(self_links)
                                .with_full(full)
                                .with_width(width)
                                .with_k(k);
                            let b = recurrence_matrix(data.view(), &cfg).unwrap();
                            let f = recurrence_matrix_weighted(data.view(), &cfg).unwrap();
                            assert_eq!(b.mapv(|v| if v { 1.0 } else { 0.0 }), f);
                            let h = t / 2;
                            let ccfg = CrossSimilarityConfig::new().with_full(full).with_k(k);
                            let (q, r) = (
                                data.slice(ndarray::s![.., ..h]),
                                data.slice(ndarray::s![.., h..]),
                            );
                            let b = cross_similarity(q, r, &ccfg).unwrap();
                            let f = cross_similarity_weighted(q, r, &ccfg).unwrap();
                            assert_eq!(b.mapv(|v| if v { 1.0 } else { 0.0 }), f);
                        }
                    }
                }
            }
        }
        let dist = RecurrenceConfig::new().with_mode(RecurrenceMode::Distance);
        let data = Array2::<f64>::zeros((1, 8));
        assert!(recurrence_matrix(data.view(), &dist).is_err());
    }
}
