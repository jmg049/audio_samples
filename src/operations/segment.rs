//! Structure analysis over feature matrices: path enhancement, temporally constrained
//! clustering and subsegmentation.
//!
//! # What
//!
//! - [`diagonal_filter`]: a smoothing kernel along a line through the origin.
//! - [`path_enhance`]: the largest response of a recurrence matrix to diagonal kernels of
//!   several slopes.
//! - [`timelag_filter`]: applies a filter to a recurrence matrix in lag coordinates.
//! - [`agglomerative`]: Ward clustering of frames, where only neighbouring segments may
//!   merge, returning the segment boundaries.
//! - [`subsegment`]: the same inside each interval of a coarser segmentation.
//!
//! # Why
//!
//! Repeated musical passages appear as diagonal stripes in a recurrence matrix. Smoothing
//! along the stripes, at slopes that allow for small tempo differences, makes them stand
//! out, and linear temporal clustering turns frames into sections.
//!
//! # How
//!
//! Matrices have one column per frame. The kernel of [`diagonal_filter`] is a window on the
//! main diagonal, rotated by quintic B-spline interpolation. [`agglomerative`] keeps a heap
//! of the Ward distances of neighbouring clusters, so it takes `O(n log n)` merges'
//! worth of work on `n` frames.
//!
//! ```
//! use audio_samples::operations::segment::agglomerative;
//! use ndarray::array;
//!
//! // Three frames near 0, then two near 5.
//! let x = array![[0.0, 0.1, 0.0, 5.0, 5.1]];
//! assert_eq!(agglomerative(x.view(), 2).unwrap(), vec![0, 3]);
//! ```
//!
//! References: J. H. Ward, *Hierarchical grouping to optimize an objective function*,
//! JASA 58, 1963; G. N. Lance and W. T. Williams, *A general theory of classificatory
//! sorting strategies*, The Computer Journal 9, 1967; J. Foote, *Visualizing music and
//! audio using self-similarity*, ACM Multimedia 1999; M. Müller and F. Kurth, *Enhancing
//! similarity matrices for music audio analysis*, ICASSP 2006; M. Unser, *Splines: a
//! perfect fit for signal and image processing*, IEEE SPM 16, 1999.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use ndarray::{Array2, ArrayView2, s};

use crate::operations::filterbanks::{Window, get_window};
use crate::operations::sequence::{lag_to_recurrence, recurrence_to_lag};
use crate::utils::array_ops::fix_frames;
use crate::{AudioFloat, AudioSampleResult, ParameterError};

fn invalid<T>(parameter: &str, reason: impl Into<String>) -> AudioSampleResult<T> {
    Err(ParameterError::invalid_value(parameter, reason.into()).into())
}

fn of<F: AudioFloat>(v: f64) -> F {
    <F as num_traits::NumCast>::from(v).unwrap_or_else(F::nan)
}

/// The quintic cardinal B-spline at distance `d >= 0` from its centre.
fn quintic(d: f64) -> f64 {
    let p5 = |v: f64| v * v * v * v * v;
    let v = if d < 1.0 {
        p5(3.0 - d) - 6.0 * p5(2.0 - d) + 15.0 * p5(1.0 - d)
    } else if d < 2.0 {
        p5(3.0 - d) - 6.0 * p5(2.0 - d)
    } else if d < 3.0 {
        p5(3.0 - d)
    } else {
        0.0
    };
    v / 120.0
}

/// Index `i` reflected into `0..n` about the first and last sample.
fn mirror(mut i: isize, n: isize) -> usize {
    if n == 1 {
        return 0;
    }
    let period = 2 * n - 2;
    i = i.rem_euclid(period);
    (if i < n { i } else { period - i }) as usize
}

fn is_close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
}

/// Rotates a square matrix by `angle` degrees about its centre with quintic B-spline
/// weights applied directly to the samples. The result is large enough to hold the
/// rotated corners. Points outside the input are zero.
fn rotate_quintic(input: &Array2<f64>, angle: f64) -> Array2<f64> {
    let (ny, nx) = input.dim();
    let radians = angle.to_radians();
    let (c, s_a) = (radians.cos(), radians.sin());
    let (iy, ix) = (ny as f64, nx as f64);
    let corners_y = [0.0, c * ix * 0.0 + s_a * ix, c * iy, c * iy + s_a * ix];
    let corners_x = [0.0, c * ix, -s_a * iy, -s_a * iy + c * ix];
    let ptp = |v: &[f64; 4]| {
        let (lo, hi) = v
            .iter()
            .fold((f64::MAX, f64::MIN), |(l, h), &e| (l.min(e), h.max(e)));
        hi - lo
    };
    let (oy, ox) = (
        (ptp(&corners_y) + 0.5).trunc() as usize,
        (ptp(&corners_x) + 0.5).trunc() as usize,
    );
    let (out_cy, out_cx) = ((oy as f64 - 1.0) / 2.0, (ox as f64 - 1.0) / 2.0);
    let off_y = (iy - 1.0) / 2.0 - (c * out_cy + s_a * out_cx);
    let off_x = (ix - 1.0) / 2.0 - (-s_a * out_cy + c * out_cx);
    let mut out = Array2::<f64>::zeros((oy, ox));
    let (ny_i, nx_i) = (ny as isize, nx as isize);
    for yo in 0..oy {
        for xo in 0..ox {
            let y = c * yo as f64 + s_a * xo as f64 + off_y;
            let x = -s_a * yo as f64 + c * xo as f64 + off_x;
            if y < 0.0 || y > iy - 1.0 || x < 0.0 || x > ix - 1.0 {
                continue;
            }
            let (fy, fx) = (y.floor(), x.floor());
            let mut wy = [0.0; 6];
            let mut wx = [0.0; 6];
            for j in 0..6 {
                wy[j] = quintic((y - (fy - 2.0 + j as f64)).abs());
                wx[j] = quintic((x - (fx - 2.0 + j as f64)).abs());
            }
            let (sy, sx) = (fy as isize - 2, fx as isize - 2);
            let mut sum = 0.0;
            for (jy, wyv) in wy.iter().enumerate() {
                let row = input.row(mirror(sy + jy as isize, ny_i));
                for (jx, wxv) in wx.iter().enumerate() {
                    sum += wyv * wxv * row[mirror(sx + jx as isize, nx_i)];
                }
            }
            out[[yo, xo]] = sum;
        }
    }
    out
}

/// A normalised smoothing kernel along a line through the origin.
///
/// # Arguments
/// - `window`: the window along the line.
/// - `n`: its length, before rotation.
/// - `slope`: the line's slope. Ignored when `angle` is given. 1 is the main diagonal.
/// - `angle`: the line's angle in radians.
/// - `zero_mean`: subtract the mean, so that the kernel removes a constant.
///
/// # Returns
/// The kernel, summing to 1 before `zero_mean`.
///
/// # Errors
/// [`ParameterError`] if `n` is 0, the window is invalid or the kernel sums to zero.
///
/// # Examples
/// ```
/// use audio_samples::operations::filterbanks::Window;
/// use audio_samples::operations::segment::diagonal_filter;
///
/// let k = diagonal_filter(&Window::Hann, 5, 1.0, None, false).unwrap();
/// assert_eq!(k.dim(), (5, 5));
/// assert!((k.sum() - 1.0).abs() < 1e-12);
/// assert_eq!(k[[0, 1]], 0.0);
/// ```
pub fn diagonal_filter(
    window: &Window,
    n: usize,
    slope: f64,
    angle: Option<f64>,
    zero_mean: bool,
) -> AudioSampleResult<Array2<f64>> {
    if n == 0 {
        return invalid("n", "must be positive");
    }
    crate::utils::check_derived_len("n", "a diagonal kernel", (n * n) as f64, 1e9)?;
    let angle = angle.unwrap_or_else(|| slope.atan());
    if !angle.is_finite() {
        return invalid("angle", "must be finite");
    }
    let w = get_window(window, n, false)?;
    let mut kernel = Array2::zeros((n, n));
    for i in 0..n {
        kernel[[i, i]] = w[i];
    }
    if !is_close(angle, std::f64::consts::FRAC_PI_4) {
        kernel = rotate_quintic(&kernel, 45.0 - angle.to_degrees());
    }
    kernel.mapv_inplace(|v| v.max(0.0));
    let total = kernel.sum();
    if total.is_nan() || total <= 0.0 {
        return invalid("window", "the kernel sums to zero");
    }
    kernel.mapv_inplace(|v| v / total);
    if zero_mean {
        let mean = kernel.sum() / kernel.len() as f64;
        kernel.mapv_inplace(|v| v - mean);
    }
    Ok(kernel)
}

/// How [`path_enhance`] extends the matrix beyond its edges.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum BoundaryMode {
    /// Mirror, repeating the edge (`d c b a | a b c d | d c b a`).
    #[default]
    Reflect,
    /// A constant value.
    Constant(f64),
    /// The edge value (`a a a a | a b c d | d d d d`).
    Nearest,
    /// Mirror about the edge sample (`d c b | a b c d | c b a`).
    Mirror,
    /// Periodic (`a b c d | a b c d | a b c d`).
    Wrap,
}

fn boundary_index(i: isize, n: isize, mode: BoundaryMode) -> Option<usize> {
    if (0..n).contains(&i) {
        return Some(i as usize);
    }
    match mode {
        BoundaryMode::Constant(_) => None,
        BoundaryMode::Nearest => Some(i.clamp(0, n - 1) as usize),
        BoundaryMode::Wrap => Some(i.rem_euclid(n) as usize),
        BoundaryMode::Mirror => Some(mirror(i, n)),
        BoundaryMode::Reflect => {
            let q = i.rem_euclid(2 * n);
            Some((if q < n { q } else { 2 * n - 1 - q }) as usize)
        }
    }
}

/// Parameters of [`path_enhance`].
#[derive(Debug, Clone, PartialEq)]
pub struct PathEnhanceConfig {
    /// The window along each diagonal. Default Hann.
    pub window: Window,
    /// Largest slope. Default 2.
    pub max_ratio: f64,
    /// Smallest slope; `None` is `1 / max_ratio`. Default `None`.
    pub min_ratio: Option<f64>,
    /// Number of slopes, spaced logarithmically. Default 7.
    pub n_filters: usize,
    /// Subtract each kernel's mean. Default false.
    pub zero_mean: bool,
    /// Set negative results to zero. Default true.
    pub clip: bool,
    /// How the matrix extends beyond its edges. Default [`BoundaryMode::Reflect`].
    pub mode: BoundaryMode,
    /// Shifts each kernel from its centre, in samples along `(rows, columns)`. Default 0.
    pub origin: (isize, isize),
}

impl PathEnhanceConfig {
    /// A Hann window, 7 slopes between 1/2 and 2, mirrored edges and clipping.
    #[must_use]
    pub fn new() -> Self {
        Self {
            window: Window::Hann,
            max_ratio: 2.0,
            min_ratio: None,
            n_filters: 7,
            zero_mean: false,
            clip: true,
            mode: BoundaryMode::Reflect,
            origin: (0, 0),
        }
    }

    /// librosa's defaults; currently equal to [`new`](Self::new).
    #[must_use]
    pub fn librosa() -> Self {
        Self::new()
    }

    /// Sets the kernel's shift from its centre.
    #[must_use]
    pub fn with_origin(mut self, origin: (isize, isize)) -> Self {
        self.origin = origin;
        self
    }

    /// Sets the window.
    #[must_use]
    pub fn with_window(mut self, window: Window) -> Self {
        self.window = window;
        self
    }

    /// Sets the slope range and count.
    #[must_use]
    pub fn with_ratios(mut self, max_ratio: f64, min_ratio: Option<f64>, n_filters: usize) -> Self {
        self.max_ratio = max_ratio;
        self.min_ratio = min_ratio;
        self.n_filters = n_filters;
        self
    }

    /// Sets whether each kernel has zero mean.
    #[must_use]
    pub fn with_zero_mean(mut self, zero_mean: bool) -> Self {
        self.zero_mean = zero_mean;
        self
    }

    /// Sets whether negative results are set to zero.
    #[must_use]
    pub fn with_clip(mut self, clip: bool) -> Self {
        self.clip = clip;
        self
    }

    /// Sets the edge handling.
    #[must_use]
    pub fn with_mode(mut self, mode: BoundaryMode) -> Self {
        self.mode = mode;
        self
    }
}

impl Default for PathEnhanceConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Adds `weight` times the row of `source` shifted by `dx` to `out`, extending the row
/// beyond its ends by `mode`.
fn add_shifted_row<F: AudioFloat>(
    out: &mut [F],
    source: Option<&[F]>,
    dx: isize,
    weight: F,
    mode: BoundaryMode,
    fill: F,
) {
    let cols = out.len() as isize;
    let Some(row) = source else {
        for o in out.iter_mut() {
            *o += weight * fill;
        }
        return;
    };
    // Columns whose source lies inside the row: 0 <= column - dx < cols.
    let start = dx.max(0).min(cols);
    let end = (cols + dx.min(0)).max(start);
    for column in (0..start).chain(end..cols) {
        let value = match boundary_index(column - dx, cols, mode) {
            Some(i) => row[i],
            None => fill,
        };
        out[column as usize] += weight * value;
    }
    let (first, last) = (start as usize, end as usize);
    let shift = (start - dx) as usize;
    for (o, &v) in out[first..last]
        .iter_mut()
        .zip(&row[shift..shift + (last - first)])
    {
        *o += weight * v;
    }
}

/// Convolves `r` with the non-zero `kernel` entries into `out`.
fn convolve_rows<F: AudioFloat>(
    r: ArrayView2<'_, F>,
    kernel: &[(isize, isize, F)],
    mode: BoundaryMode,
    fill: F,
    out: &mut Array2<F>,
) {
    let (rows, cols) = out.dim();
    out.fill(F::zero());
    let Some(source) = r.to_slice() else {
        return;
    };
    let Some(target) = out.as_slice_mut() else {
        return;
    };
    for &(dy, dx, w) in kernel {
        for (row, chunk) in target.chunks_exact_mut(cols).enumerate() {
            let line = boundary_index(row as isize - dy, rows as isize, mode)
                .map(|i| &source[i * cols..(i + 1) * cols]);
            add_shifted_row(chunk, line, dx, w, mode, fill);
        }
    }
}

/// Enhances the diagonal paths of a recurrence matrix.
///
/// # Arguments
/// - `r`: a square or lag matrix.
/// - `n`: the length of each diagonal kernel, before rotation.
/// - `config`: the slopes and the window.
///
/// # Returns
/// The largest response over the slopes, the shape of `r`.
///
/// # Errors
/// [`ParameterError`] if the ratios are not positive or `min_ratio` exceeds `max_ratio`,
/// `n_filters` is 0 or a kernel cannot be built.
///
/// # Examples
/// ```
/// use audio_samples::operations::segment::{PathEnhanceConfig, path_enhance};
/// use ndarray::Array2;
///
/// let r = Array2::from_shape_fn((16, 16), |(i, j)| if i == j { 1.0 } else { 0.0 });
/// let e = path_enhance(r.view(), 5, &PathEnhanceConfig::new()).unwrap();
/// assert!(e[[8, 8]] > e[[8, 2]]);
/// ```
pub fn path_enhance<F: AudioFloat>(
    r: ArrayView2<'_, F>,
    n: usize,
    config: &PathEnhanceConfig,
) -> AudioSampleResult<Array2<F>> {
    if config.n_filters == 0 {
        return invalid("n_filters", "must be positive");
    }
    let min_ratio = match config.min_ratio {
        None => 1.0 / config.max_ratio,
        Some(m) => {
            if m > config.max_ratio {
                return invalid("min_ratio", "cannot exceed max_ratio");
            }
            m
        }
    };
    if !(min_ratio > 0.0
        && config.max_ratio > 0.0
        && min_ratio.is_finite()
        && config.max_ratio.is_finite())
    {
        return invalid("max_ratio", "the ratios must be positive and finite");
    }
    if r.is_empty() {
        return invalid("r", "must not be empty");
    }
    let (lo, hi) = (min_ratio.log2(), config.max_ratio.log2());
    let mut kernels: Vec<Vec<(isize, isize, F)>> = Vec::with_capacity(config.n_filters);
    for i in 0..config.n_filters {
        let t = if config.n_filters == 1 {
            0.0
        } else {
            i as f64 / (config.n_filters - 1) as f64
        };
        let ratio = 2f64.powf(lo + (hi - lo) * t);
        let k = diagonal_filter(&config.window, n, ratio, None, config.zero_mean)?;
        let (h, w) = k.dim();
        let (cy, cx) = (
            h as isize / 2 + config.origin.0,
            w as isize / 2 + config.origin.1,
        );
        let mut entries = Vec::new();
        for ((y, x), &v) in k.indexed_iter() {
            if v != 0.0 {
                entries.push((y as isize - cy, x as isize - cx, of::<F>(v)));
            }
        }
        kernels.push(entries);
    }
    let (rows, cols) = r.dim();
    let fill = match config.mode {
        BoundaryMode::Constant(c) => of::<F>(c),
        _ => F::zero(),
    };
    let standard = r.as_standard_layout();
    let mut out = Array2::<F>::zeros((rows, cols));
    let mut scratch = Array2::<F>::zeros(if kernels.len() > 1 {
        (rows, cols)
    } else {
        (0, 0)
    });
    for (i, entries) in kernels.iter().enumerate() {
        if i == 0 {
            convolve_rows(standard.view(), entries, config.mode, fill, &mut out);
        } else {
            convolve_rows(standard.view(), entries, config.mode, fill, &mut scratch);
            out.zip_mut_with(&scratch, |a, &b| {
                if b > *a {
                    *a = b;
                }
            });
        }
    }
    if config.clip {
        out.mapv_inplace(|v| if v > F::zero() { v } else { F::zero() });
    }
    Ok(out)
}

/// Applies a filter to a recurrence matrix in lag coordinates.
///
/// Converts to a time-lag matrix, applies `filter`, and converts back. In lag coordinates a
/// repeated passage is a horizontal line, so an ordinary median or mean filter follows it.
///
/// # Arguments
/// - `r`: a square recurrence matrix.
/// - `pad`: pad the lag matrix to twice the length, so that lags do not wrap.
/// - `filter`: the filter; it must return a matrix of the shape it is given.
///
/// # Returns
/// The filtered recurrence matrix.
///
/// # Errors
/// [`ParameterError`] if `r` is not square or the filter changes the shape, and any error
/// from `filter`.
///
/// # Examples
/// ```
/// use audio_samples::operations::segment::timelag_filter;
/// use ndarray::Array2;
///
/// let r = Array2::from_shape_fn((4, 4), |(i, j)| (i * 4 + j) as f64);
/// let same = timelag_filter(r.view(), true, Ok).unwrap();
/// assert_eq!(same, r);
/// ```
pub fn timelag_filter<T: Copy + Default>(
    r: ArrayView2<'_, T>,
    pad: bool,
    filter: impl FnOnce(Array2<T>) -> AudioSampleResult<Array2<T>>,
) -> AudioSampleResult<Array2<T>> {
    let lag = recurrence_to_lag(r, pad)?;
    let dim = lag.dim();
    let filtered = filter(lag)?;
    if filtered.dim() != dim {
        return invalid("filter", "must keep the shape of its input");
    }
    lag_to_recurrence(filtered.view())
}

#[derive(Clone, Copy)]
struct Merge {
    distance: f64,
    first: u32,
    second: u32,
}

impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Merge {}
impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Merge {
    /// The heap's largest is the smallest distance, then the oldest clusters.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .total_cmp(&self.distance)
            .then_with(|| other.first.cmp(&self.first))
            .then_with(|| other.second.cmp(&self.second))
    }
}

/// Segment boundaries from Ward clustering in which only neighbouring segments merge.
///
/// # Arguments
/// - `data`: `(features, frames)`, finite.
/// - `k`: the number of segments, from 1 to the number of frames.
///
/// # Returns
/// The first frame of each segment, starting with 0.
///
/// # Errors
/// [`ParameterError`] if `k` is out of range, `data` has no frames or is not finite.
///
/// # Examples
/// See the [module documentation](self).
pub fn agglomerative<F: AudioFloat>(
    data: ArrayView2<'_, F>,
    k: usize,
) -> AudioSampleResult<Vec<usize>> {
    let (d, n) = data.dim();
    if n == 0 {
        return invalid("data", "has no frames");
    }
    if k == 0 || k > n {
        return invalid("k", format!("must lie in 1..={n}, got {k}"));
    }
    if data.iter().any(|v| !v.is_finite()) {
        return invalid("data", "must be finite");
    }
    if k == n {
        return Ok((0..n).collect());
    }
    // Node storage: leaves, then one new node per merge.
    let capacity = 2 * n - k;
    // Means, then sizes; short inputs need no allocation for them.
    let mut small = [0.0_f64; 256];
    let mut large = Vec::new();
    let store = if capacity * (d + 1) <= small.len() {
        &mut small[..capacity * (d + 1)]
    } else {
        large.resize(capacity * (d + 1), 0.0);
        &mut large[..]
    };
    let (means, sizes) = store.split_at_mut(capacity * d);
    for (j, column) in data.columns().into_iter().enumerate() {
        for (m, v) in means[j * d..(j + 1) * d].iter_mut().zip(column.iter()) {
            *m = v.to_f64().unwrap_or(f64::NAN);
        }
    }
    let mut alive = vec![false; capacity];
    let mut left = vec![u32::MAX; capacity];
    let mut right = vec![u32::MAX; capacity];
    let mut first_frame = vec![0_u32; capacity];
    for j in 0..n {
        sizes[j] = 1.0;
        alive[j] = true;
        first_frame[j] = j as u32;
        if j > 0 {
            left[j] = j as u32 - 1;
        }
        if j + 1 < n {
            right[j] = j as u32 + 1;
        }
    }
    let ward = |means: &[f64], sizes: &[f64], a: usize, b: usize| -> f64 {
        let (ma, mb) = (&means[a * d..(a + 1) * d], &means[b * d..(b + 1) * d]);
        let dist: f64 = ma.iter().zip(mb).map(|(x, y)| (x - y) * (x - y)).sum();
        dist * sizes[a] * sizes[b] / (sizes[a] + sizes[b])
    };
    let mut heap = BinaryHeap::with_capacity(2 * n);
    for j in 0..n - 1 {
        heap.push(Merge {
            distance: ward(means, sizes, j, j + 1),
            first: j as u32,
            second: j as u32 + 1,
        });
    }
    let mut next = n;
    while next < capacity {
        let Some(m) = heap.pop() else {
            break;
        };
        let (a, b) = (m.first as usize, m.second as usize);
        if !alive[a] || !alive[b] {
            continue;
        }
        let new = next;
        next += 1;
        let total = sizes[a] + sizes[b];
        let (wa, wb) = (sizes[a] / total, sizes[b] / total);
        for i in 0..d {
            means[new * d + i] = wa * means[a * d + i] + wb * means[b * d + i];
        }
        sizes[new] = total;
        alive[a] = false;
        alive[b] = false;
        alive[new] = true;
        // The merged pair is adjacent in time, in some order.
        let (before, after) = if right[a] == b as u32 {
            (left[a], right[b])
        } else {
            (left[b], right[a])
        };
        first_frame[new] = first_frame[a].min(first_frame[b]);
        left[new] = before;
        right[new] = after;
        for (neighbour, is_left) in [(before, true), (after, false)] {
            if neighbour != u32::MAX {
                let w = neighbour as usize;
                if is_left {
                    right[w] = new as u32;
                } else {
                    left[w] = new as u32;
                }
                heap.push(Merge {
                    distance: ward(means, sizes, new, w),
                    first: new as u32,
                    second: neighbour,
                });
            }
        }
    }
    let mut boundaries: Vec<usize> = alive
        .iter()
        .enumerate()
        .filter(|&(_, &a)| a)
        .map(|(i, _)| first_frame[i] as usize)
        .collect();
    boundaries.sort_unstable();
    Ok(boundaries)
}

/// Subdivides each interval of a segmentation with [`agglomerative`].
///
/// # Arguments
/// - `data`: `(features, frames)`.
/// - `frames`: the segment boundaries; they are normalised with
///   [`fix_frames`](crate::utils::array_ops::fix_frames) to include the ends.
/// - `n_segments`: the most subsegments per interval.
///
/// # Returns
/// The first frame of each subsegment.
///
/// # Errors
/// [`ParameterError`] if `n_segments` is 0, an index is negative or `data` is not finite.
///
/// # Examples
/// ```
/// use audio_samples::operations::segment::subsegment;
/// use ndarray::array;
///
/// let x = array![[0.0, 0.0, 5.0, 5.0, 9.0, 9.0, 2.0, 2.0]];
/// assert_eq!(subsegment(x.view(), &[4], 2).unwrap(), vec![0, 2, 4, 6]);
/// ```
pub fn subsegment<F: AudioFloat>(
    data: ArrayView2<'_, F>,
    frames: &[i64],
    n_segments: usize,
) -> AudioSampleResult<Vec<usize>> {
    if n_segments == 0 {
        return invalid("n_segments", "must be positive");
    }
    let n = data.ncols() as i64;
    let bounds = fix_frames(frames, Some(0), Some(n), true)?;
    let mut out = Vec::new();
    for pair in bounds.windows(2) {
        let (start, end) = (pair[0] as usize, pair[1] as usize);
        let part = data.slice(s![.., start..end]);
        for b in agglomerative(part, (end - start).min(n_segments))? {
            out.push(start + b);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn rejects_bad_parameters() {
        let x = array![[1.0, 2.0]];
        assert!(agglomerative(x.view(), 0).is_err());
        assert!(agglomerative(x.view(), 3).is_err());
        assert!(agglomerative(array![[f64::NAN, 1.0]].view(), 1).is_err());
        assert!(subsegment(x.view(), &[1], 0).is_err());
        assert!(diagonal_filter(&Window::Hann, 0, 1.0, None, false).is_err());
        let r = Array2::<f64>::zeros((4, 4));
        let bad = PathEnhanceConfig::new().with_ratios(1.0, Some(2.0), 3);
        assert!(path_enhance(r.view(), 3, &bad).is_err());
        let bad = PathEnhanceConfig::new().with_ratios(2.0, None, 0);
        assert!(path_enhance(r.view(), 3, &bad).is_err());
    }

    #[test]
    fn all_frames_are_segments_at_k_equal_n() {
        let x = array![[1.0, 2.0, 3.0]];
        assert_eq!(agglomerative(x.view(), 3).unwrap(), vec![0, 1, 2]);
        assert_eq!(agglomerative(x.view(), 1).unwrap(), vec![0]);
    }

    #[test]
    fn boundary_modes() {
        assert_eq!(boundary_index(-1, 4, BoundaryMode::Reflect), Some(0));
        assert_eq!(boundary_index(-2, 4, BoundaryMode::Reflect), Some(1));
        assert_eq!(boundary_index(-1, 4, BoundaryMode::Mirror), Some(1));
        assert_eq!(boundary_index(4, 4, BoundaryMode::Nearest), Some(3));
        assert_eq!(boundary_index(4, 4, BoundaryMode::Wrap), Some(0));
        assert_eq!(boundary_index(4, 4, BoundaryMode::Constant(0.0)), None);
    }

    #[test]
    fn timelag_filter_checks_the_shape() {
        let r = Array2::<f64>::zeros((3, 3));
        let e = timelag_filter(r.view(), true, |_| Ok(Array2::zeros((1, 1))));
        assert!(e.is_err());
    }
}
