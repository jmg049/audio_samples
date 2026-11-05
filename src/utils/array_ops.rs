//! Array helpers: local extrema, event and interval matching, one-dimensional
//! interpolation, padding, reordering, compressed sparse rows and argument checks.
//!
//! # What
//!
//! Small, general operations on `ndarray` arrays and index lists that feature
//! pipelines keep needing: [`localmax`] / [`localmin`], [`match_events`] /
//! [`match_intervals`], [`fix_frames`] / [`index_to_slice`], [`axis_sort`],
//! [`shear`], [`cyclic_gradient`], [`fill_off_diagonal`], [`count_unique`] /
//! [`is_unique`], [`pad_center`], [`stack`], [`expand_to`], [`abs2`], [`phasor`],
//! [`sparsify_rows`], [`interp1d`] / [`interp_broadcast`], [`buf_to_float`],
//! [`find_files`], and checks such as [`valid_audio`] and [`valid_intervals`].
//!
//! # Why
//!
//! Each of these is a few lines to write once and easy to get subtly wrong at the
//! edges: the first and last element of a local-extremum test, ties when matching
//! events, a wrapped gradient, spline end conditions. Keeping them in one tested
//! place lets analysis code and users share one definition.
//!
//! # How
//!
//! All functions are free functions over `ndarray` views or slices. Functions on
//! floats are generic over [`AudioFloat`] and return the input's element type.
//! Invalid input gives a [`ParameterError`]; nothing panics on bad arguments.
//!
//! ```
//! use audio_samples::utils::array_ops::{localmax, match_events};
//! use ndarray::{Axis, array};
//!
//! let peaks = localmax(&array![1.0, 0.0, 1.0, 2.0, -1.0, 0.0, -2.0, 1.0], Axis(0)).unwrap();
//! assert_eq!(peaks.to_vec(), vec![false, false, false, true, false, true, false, true]);
//!
//! // Each source event goes to the nearest target event.
//! let matched = match_events(&[1.0, 4.9, 9.0], &[0.0, 5.0, 10.0], true, true).unwrap();
//! assert_eq!(matched, vec![0, 1, 2]);
//! ```
//!
//! The spline kinds of [`interp1d`] use not-a-knot B-spline interpolation
//! (C. de Boor, *A Practical Guide to Splines*, Springer, 1978). Its collocation
//! matrix is totally positive, so band elimination needs no pivoting (de Boor and
//! Pinkus, *Backward error analysis for totally positive linear systems*,
//! Numerische Mathematik 27, 1977).

use std::cmp::Ordering;
use std::path::{Component, Path, PathBuf};

use ndarray::{
    Array, Array2, ArrayBase, ArrayD, ArrayView1, ArrayView2, ArrayViewD, ArrayViewMut2, Axis,
    Data, Dimension, IxDyn, RemoveAxis, Slice, Zip,
};
use num_complex::Complex;

use crate::{AudioFloat, AudioSampleResult, ParameterError};

/// Suggested upper limit in bytes for one block of a batched computation: 256 KiB.
pub const MAX_MEM_BLOCK: usize = 1 << 18;

fn invalid<T>(parameter: &str, reason: impl Into<String>) -> AudioSampleResult<T> {
    Err(ParameterError::invalid_value(parameter, reason.into()).into())
}

fn check_axis(parameter: &str, axis: Axis, ndim: usize) -> AudioSampleResult<()> {
    if axis.index() >= ndim {
        return invalid(
            parameter,
            format!(
                "axis {} is out of range for {ndim} dimensions",
                axis.index()
            ),
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Local extrema
// ---------------------------------------------------------------------------

fn local_extrema<A, S, D>(
    x: &ArrayBase<S, D>,
    axis: Axis,
    strict: impl Fn(&A, &A) -> bool,
    loose: impl Fn(&A, &A) -> bool,
) -> AudioSampleResult<Array<bool, D>>
where
    S: Data<Elem = A>,
    D: Dimension + RemoveAxis,
{
    check_axis("axis", axis, x.ndim())?;
    let mut out = Array::from_elem(x.raw_dim(), false);
    Zip::from(out.lanes_mut(axis))
        .and(x.lanes(axis))
        .for_each(|mut o, lane| {
            let n = lane.len();
            if n < 2 {
                return;
            }
            for i in 1..n - 1 {
                o[i] = strict(&lane[i], &lane[i - 1]) && loose(&lane[i], &lane[i + 1]);
            }
            o[n - 1] = strict(&lane[n - 1], &lane[n - 2]);
        });
    Ok(out)
}

/// Marks the local maxima of an array along one axis.
///
/// `x[i]` is a maximum when `x[i] > x[i - 1]` and `x[i] >= x[i + 1]`. The first
/// element is never a maximum; the last one is when `x[n - 1] > x[n - 2]`.
///
/// # Arguments
/// - `x`: the array, of any dimension and ordered element type.
/// - `axis`: the axis to compare along.
///
/// # Returns
/// A boolean array of the same shape.
///
/// # Errors
/// [`ParameterError`] if `axis` is out of range.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::localmax;
/// use ndarray::{Axis, array};
///
/// let m = localmax(&array![[1.0, 3.0, 2.0], [2.0, 1.0, 5.0]], Axis(1)).unwrap();
/// assert_eq!(m, array![[false, true, false], [false, false, true]]);
/// ```
pub fn localmax<A, S, D>(x: &ArrayBase<S, D>, axis: Axis) -> AudioSampleResult<Array<bool, D>>
where
    A: PartialOrd,
    S: Data<Elem = A>,
    D: Dimension + RemoveAxis,
{
    local_extrema(x, axis, |a, b| a > b, |a, b| a >= b)
}

/// Marks the local minima of an array along one axis.
///
/// `x[i]` is a minimum when `x[i] < x[i - 1]` and `x[i] <= x[i + 1]`. The first
/// element is never a minimum; the last one is when `x[n - 1] < x[n - 2]`.
///
/// # Arguments
/// - `x`: the array, of any dimension and ordered element type.
/// - `axis`: the axis to compare along.
///
/// # Returns
/// A boolean array of the same shape.
///
/// # Errors
/// [`ParameterError`] if `axis` is out of range.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::localmin;
/// use ndarray::{Axis, array};
///
/// let m = localmin(&array![3.0, 1.0, 2.0, 0.0], Axis(0)).unwrap();
/// assert_eq!(m, array![false, true, false, true]);
/// ```
pub fn localmin<A, S, D>(x: &ArrayBase<S, D>, axis: Axis) -> AudioSampleResult<Array<bool, D>>
where
    A: PartialOrd,
    S: Data<Elem = A>,
    D: Dimension + RemoveAxis,
{
    local_extrema(x, axis, |a, b| a < b, |a, b| a <= b)
}

// ---------------------------------------------------------------------------
// Event and interval matching
// ---------------------------------------------------------------------------

fn sorted_index(values: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(Ordering::Equal));
    idx
}

/// Matches each source event to the nearest target event.
///
/// # Arguments
/// - `events_from`: the source events, in any order.
/// - `events_to`: the target events, in any order.
/// - `left`: a source event may match a target before it.
/// - `right`: a source event may match a target after it.
///
/// # Returns
/// For each source event, the index in `events_to` of its match. A source event
/// half-way between two targets goes to the later one.
///
/// # Errors
/// [`ParameterError`] if either list is empty, or if the `left` / `right`
/// restrictions leave a source event with no possible match.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::match_events;
///
/// let m = match_events(&[1.0, 6.0], &[0.0, 5.0, 10.0], false, true).unwrap();
/// assert_eq!(m, vec![1, 2]);
/// ```
pub fn match_events(
    events_from: &[f64],
    events_to: &[f64],
    left: bool,
    right: bool,
) -> AudioSampleResult<Vec<usize>> {
    if events_from.is_empty() || events_to.is_empty() {
        return invalid("events", "cannot match an empty event list");
    }
    let to_idx = sorted_index(events_to);
    let sorted: Vec<f64> = to_idx.iter().map(|&i| events_to[i]).collect();
    let (lo, hi) = (sorted[0], sorted[sorted.len() - 1]);
    let mut out = Vec::with_capacity(events_from.len());
    for &event in events_from {
        let at = sorted.partition_point(|&t| t < event);
        let exact = at < sorted.len() && sorted[at] == event;
        if !left && !right && !exact {
            return invalid(
                "events_from",
                "with left and right both false, every source event must be a target event",
            );
        }
        if !left && hi < event {
            return invalid(
                "events_from",
                "with left false, a source event follows every target",
            );
        }
        if !right && lo > event {
            return invalid(
                "events_from",
                "with right false, a source event precedes every target",
            );
        }
        let chosen = if exact {
            at
        } else if at == sorted.len() {
            at - 1
        } else if at == 0 {
            0
        } else if !left {
            at
        } else if !right || (event - sorted[at - 1]) < (sorted[at] - event) {
            at - 1
        } else {
            at
        };
        out.push(to_idx[chosen]);
    }
    Ok(out)
}

/// Checks that `intervals` has shape `(n, 2)` and that no interval ends before it
/// starts.
///
/// # Arguments
/// - `intervals`: one `[start, end]` row per interval.
///
/// # Errors
/// [`ParameterError`] if the shape is wrong or a duration is negative.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::valid_intervals;
/// use ndarray::array;
///
/// assert!(valid_intervals(array![[0.0, 1.0]].view()).is_ok());
/// assert!(valid_intervals(array![[1.0, 0.0]].view()).is_err());
/// ```
pub fn valid_intervals(intervals: ArrayView2<'_, f64>) -> AudioSampleResult<()> {
    if intervals.ncols() != 2 {
        return invalid("intervals", "must have shape (n, 2)");
    }
    let reversed = if let Some(flat) = intervals.as_slice() {
        flat.chunks_exact(2)
            .fold(false, |bad, r| bad | (r[0] > r[1]))
    } else {
        intervals.rows().into_iter().any(|r| r[0] > r[1])
    };
    if reversed {
        return invalid("intervals", "must have non-negative durations");
    }
    Ok(())
}

/// Matches each source interval to the target interval it overlaps most.
///
/// Overlap is the Jaccard index, the length of the intersection over the length
/// of the union; on a tie the lowest target index wins.
///
/// # Arguments
/// - `intervals_from`: source intervals, `(n, 2)`.
/// - `intervals_to`: target intervals, `(m, 2)`.
/// - `strict`: when true, a source interval that overlaps nothing is an error;
///   when false it matches the nearest target (the earlier one at equal distance).
///
/// # Returns
/// For each source interval, an index into `intervals_to`.
///
/// # Errors
/// [`ParameterError`] if a list is empty or invalid, or with `strict` when a
/// source interval overlaps no target.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::match_intervals;
/// use ndarray::array;
///
/// let from = array![[0.0, 1.0], [2.5, 3.0]];
/// let to = array![[0.0, 2.0], [2.0, 4.0]];
/// assert_eq!(match_intervals(from.view(), to.view(), true).unwrap(), vec![0, 1]);
/// ```
pub fn match_intervals(
    intervals_from: ArrayView2<'_, f64>,
    intervals_to: ArrayView2<'_, f64>,
    strict: bool,
) -> AudioSampleResult<Vec<usize>> {
    if intervals_from.is_empty() || intervals_to.is_empty() {
        return invalid("intervals", "cannot match an empty interval list");
    }
    valid_intervals(intervals_from)?;
    valid_intervals(intervals_to)?;
    let starts: Vec<f64> = intervals_to.column(0).to_vec();
    let ends: Vec<f64> = intervals_to.column(1).to_vec();
    let by_start = sorted_index(&starts);
    let by_end = sorted_index(&ends);
    let sorted_starts: Vec<f64> = by_start.iter().map(|&i| starts[i]).collect();
    let sorted_ends: Vec<f64> = by_end.iter().map(|&i| ends[i]).collect();

    let mut out = Vec::with_capacity(intervals_from.nrows());
    for query in intervals_from.rows() {
        let (qs, qe) = (query[0], query[1]);
        // Targets that start no later than the query ends.
        let started = sorted_starts.partition_point(|&s| s <= qe);
        let mut best: Option<(f64, usize)> = None;
        for &i in &by_start[..started] {
            if ends[i] < qs {
                continue;
            }
            let intersection = (ends[i].min(qe) - starts[i].max(qs)).max(0.0);
            let union = ends[i].max(qe) - starts[i].min(qs);
            let score = if union > 0.0 {
                intersection / union
            } else {
                0.0
            };
            let better = match best {
                None => true,
                Some((s, b)) => score > s || (score == s && i < b),
            };
            if better {
                best = Some((score, i));
            }
        }
        if let Some((_, i)) = best {
            out.push(i);
        } else if strict {
            return invalid(
                "intervals_from",
                "an interval overlaps no target with strict = true",
            );
        } else {
            let ended = sorted_ends.partition_point(|&e| e < qs);
            let before = (ended > 0).then(|| (qs - sorted_ends[ended - 1], by_end[ended - 1]));
            let after = (started < sorted_starts.len())
                .then(|| (sorted_starts[started] - qe, by_start[started]));
            out.push(match (before, after) {
                (Some((db, b)), Some((da, a))) => {
                    if db <= da {
                        b
                    } else {
                        a
                    }
                }
                (Some((_, b)), None) => b,
                (None, Some((_, a))) => a,
                (None, None) => {
                    return invalid("intervals_to", "cannot match an empty interval list");
                }
            });
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Complex helpers
// ---------------------------------------------------------------------------

/// Squared magnitude of complex values: `re * re + im * im`.
///
/// # Arguments
/// - `x`: complex values of any shape.
///
/// # Returns
/// The squared magnitudes in the real type of `x`.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::abs2;
/// use ndarray::array;
/// use num_complex::Complex;
///
/// let y = abs2(&array![Complex::new(3.0_f64, 4.0)]);
/// assert_eq!(y[0], 25.0);
/// ```
#[must_use]
pub fn abs2<F, S, D>(x: &ArrayBase<S, D>) -> Array<F, D>
where
    F: AudioFloat,
    S: Data<Elem = Complex<F>>,
    D: Dimension,
{
    x.mapv(|z| z.re * z.re + z.im * z.im)
}

/// [`abs2`] writing into an existing array of the same shape.
///
/// # Arguments
/// - `x`: complex values.
/// - `out`: receives the squared magnitudes.
///
/// # Errors
/// [`ParameterError`] if the shapes differ.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::abs2_into;
/// use ndarray::{Array1, array};
/// use num_complex::Complex;
///
/// let mut out = Array1::zeros(1);
/// abs2_into(array![Complex::new(3.0_f64, 4.0)].view(), out.view_mut()).unwrap();
/// assert_eq!(out[0], 25.0);
/// ```
pub fn abs2_into<F, D>(
    x: ndarray::ArrayView<'_, Complex<F>, D>,
    mut out: ndarray::ArrayViewMut<'_, F, D>,
) -> AudioSampleResult<()>
where
    F: AudioFloat,
    D: Dimension,
{
    if x.shape() != out.shape() {
        return invalid("out", "must have the shape of x");
    }
    Zip::from(&mut out)
        .and(&x)
        .for_each(|o, z| *o = z.re * z.re + z.im * z.im);
    Ok(())
}

/// Unit phasors `cos(angle) + i sin(angle)`.
///
/// # Arguments
/// - `angles`: angles in radians, of any shape.
///
/// # Returns
/// Complex values of the same shape and precision.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::phasor;
/// use ndarray::array;
///
/// let z = phasor(&array![0.0_f64]);
/// assert_eq!(z[0].re, 1.0);
/// assert_eq!(z[0].im, 0.0);
/// ```
#[must_use]
pub fn phasor<F, S, D>(angles: &ArrayBase<S, D>) -> Array<Complex<F>, D>
where
    F: AudioFloat,
    S: Data<Elem = F>,
    D: Dimension,
{
    angles.mapv(|a| {
        let (s, c) = a.sin_cos();
        Complex::new(c, s)
    })
}

/// The smallest positive normal value of a float type.
///
/// # Returns
/// `F::MIN_POSITIVE`, the floor that keeps a division by a small magnitude finite.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::tiny;
///
/// assert_eq!(tiny::<f32>(), f32::MIN_POSITIVE);
/// ```
#[must_use]
pub fn tiny<F: AudioFloat>() -> F {
    F::min_positive_value()
}

// ---------------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------------

/// Places the axes of `x` at given positions of a higher-dimensional view; all
/// other axes have length 1.
///
/// # Arguments
/// - `x`: the array.
/// - `ndim`: dimensions of the result.
/// - `axes`: for each axis of `x`, its position in the result; negative positions
///   count from the end.
///
/// # Returns
/// A view of the same data. Positions out of increasing order reorder the axes, so
/// axis `i` of `x` is always axis `axes[i]` of the result.
///
/// # Errors
/// [`ParameterError`] if `axes` does not have one position per axis of `x`, a
/// position is out of range or repeated, or `ndim` is below `x.ndim()`.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::expand_to;
/// use ndarray::Array2;
///
/// let x = Array2::<f64>::zeros((2, 3));
/// let y = expand_to(x.view().into_dyn(), 4, &[1, -1]).unwrap();
/// assert_eq!(y.shape(), &[1, 2, 1, 3]);
/// ```
pub fn expand_to<'a, A>(
    x: ArrayViewD<'a, A>,
    ndim: usize,
    axes: &[isize],
) -> AudioSampleResult<ArrayViewD<'a, A>> {
    let mut positions = [0_usize; MAX_DIMS];
    expand_positions(x.ndim(), ndim, axes, &mut positions)?;
    let positions = &positions[..axes.len()];
    let mut order = [0_usize; MAX_DIMS];
    for (i, o) in order.iter_mut().enumerate().take(axes.len()) {
        *o = i;
    }
    let order = &mut order[..axes.len()];
    order.sort_unstable_by_key(|&i| positions[i]);
    let mut view = x.permuted_axes(IxDyn(order));
    for p in 0..ndim {
        if !positions.contains(&p) {
            view = view.insert_axis(Axis(p));
        }
    }
    Ok(view)
}

/// Most axes [`expand_to`] handles.
pub const MAX_DIMS: usize = 64;

/// Resolves the positions of [`expand_to`] into `out`, with the same checks.
///
/// # Arguments
/// - `input_ndim`: dimensions of the input.
/// - `ndim`: dimensions of the result.
/// - `axes`: the requested position of each input axis; negative counts from the end.
/// - `out`: receives the resolved positions, one per input axis.
///
/// # Errors
/// As for [`expand_to`], and [`ParameterError`] if `out` is too short.
pub fn expand_positions(
    input_ndim: usize,
    ndim: usize,
    axes: &[isize],
    out: &mut [usize],
) -> AudioSampleResult<()> {
    if axes.len() != input_ndim {
        return invalid(
            "axes",
            format!("expected {input_ndim} positions, got {}", axes.len()),
        );
    }
    if ndim < input_ndim {
        return invalid(
            "ndim",
            format!("cannot expand {input_ndim} dimensions to {ndim}"),
        );
    }
    if out.len() < axes.len() {
        return invalid("axes", "too many axes");
    }
    for (i, &a) in axes.iter().enumerate() {
        let p = if a < 0 { a + ndim as isize } else { a };
        if p < 0 || p as usize >= ndim {
            return invalid(
                "axes",
                format!("position {a} is out of range for {ndim} dimensions"),
            );
        }
        if out[..i].contains(&(p as usize)) {
            return invalid("axes", "positions must be distinct");
        }
        out[i] = p as usize;
    }
    Ok(())
}

/// Stacks arrays of one shape along a new axis.
///
/// # Arguments
/// - `arrays`: the arrays, all the same shape.
/// - `axis`: position of the new axis.
///
/// # Returns
/// A new array with one more dimension.
///
/// # Errors
/// [`ParameterError`] if the list is empty, the shapes differ or `axis` is out of
/// range.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::stack;
/// use ndarray::{Axis, array};
///
/// let a = array![1.0, 2.0].into_dyn();
/// let b = array![3.0, 4.0].into_dyn();
/// let s = stack(&[a.view(), b.view()], Axis(0)).unwrap();
/// assert_eq!(s.shape(), &[2, 2]);
/// ```
pub fn stack<A: Clone>(arrays: &[ArrayViewD<'_, A>], axis: Axis) -> AudioSampleResult<ArrayD<A>> {
    let Some(first) = arrays.first() else {
        return invalid("arrays", "at least one array must be provided");
    };
    if arrays.iter().any(|a| a.shape() != first.shape()) {
        return invalid("arrays", "all arrays must have the same shape");
    }
    if axis.index() > first.ndim() {
        return invalid("axis", format!("axis {} is out of range", axis.index()));
    }
    ndarray::stack(axis, arrays).or_else(|e| invalid("arrays", e.to_string()))
}

/// How [`pad_center`] fills the new elements.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PadMode<A> {
    /// A constant value.
    Constant(A),
    /// The nearest edge value.
    Edge,
    /// Mirror without repeating the edge value (`d c b | a b c d | c b a`).
    Reflect,
    /// Mirror repeating the edge value (`b a | a b c d | d c`).
    Symmetric,
    /// Periodic extension (`c d | a b c d | a b`).
    Wrap,
}

/// Copies `source`, forwards or reversed, into `out`, which is walked outwards from the
/// data: from its start, or from its end when `backwards`. Returns what is left of `out`.
fn copy_run<'a, A: Clone>(
    out: &'a mut [A],
    source: &[A],
    reversed: bool,
    backwards: bool,
) -> &'a mut [A] {
    let len = source.len().min(out.len());
    let (segment, rest) = if backwards {
        let at = out.len() - len;
        let (rest, segment) = out.split_at_mut(at);
        (segment, rest)
    } else {
        out.split_at_mut(len)
    };
    let source = if reversed {
        &source[source.len() - len..]
    } else {
        &source[..len]
    };
    match (backwards, reversed) {
        (false, false) | (true, true) => segment.clone_from_slice(source),
        _ => {
            segment.clone_from_slice(source);
            segment.reverse();
        }
    }
    rest
}

/// Fills `out` going outwards from the data: `Reflect` and `Symmetric` bounce between the
/// ends, `Wrap` cycles, and `Edge` repeats. `backwards` fills towards index 0.
fn pad_side<A: Clone>(out: &mut [A], data: &[A], kind: PadSide, backwards: bool) {
    let n = data.len();
    match kind {
        PadSide::Edge => out.fill(if backwards {
            data[0].clone()
        } else {
            data[n - 1].clone()
        }),
        PadSide::Wrap => {
            // Going right repeats from the start, in order; going left repeats from the end.
            let mut rest = out;
            while !rest.is_empty() {
                rest = if backwards {
                    copy_run(rest, data, true, true)
                } else {
                    copy_run(rest, data, false, false)
                };
            }
        }
        PadSide::Mirror { repeat_edge } => {
            // Each run walks from one end of the data to the other. The first run leaves
            // the edge being extended, so it goes away from it.
            let trim = usize::from(!repeat_edge && n > 1);
            let inner = &data[trim..];
            let outer = &data[..n - trim];
            let mut rest = out;
            let mut towards_far_end = true;
            while !rest.is_empty() {
                rest = match (backwards, towards_far_end) {
                    // Extending the start: the first run reads forwards from the start.
                    (true, true) => copy_run(rest, inner, false, true),
                    (true, false) => copy_run(rest, outer, true, true),
                    // Extending the end: the first run reads backwards from the end.
                    (false, true) => copy_run(rest, outer, true, false),
                    (false, false) => copy_run(rest, inner, false, false),
                };
                towards_far_end = !towards_far_end;
            }
        }
    }
}

#[derive(Clone, Copy)]
enum PadSide {
    Edge,
    Wrap,
    Mirror { repeat_edge: bool },
}

/// Pads an array along one axis to a given length, keeping the data centred.
///
/// With an odd number of new elements the extra one goes after the data.
///
/// # Arguments
/// - `data`: the array.
/// - `size`: the length of `axis` in the result.
/// - `axis`: the axis to pad.
/// - `mode`: how to fill the new elements.
///
/// # Returns
/// A new array.
///
/// # Errors
/// [`ParameterError`] if `size` is smaller than the current length, the axis is out
/// of range, or a non-constant mode is used on an empty axis.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::{PadMode, pad_center};
/// use ndarray::{Axis, array};
///
/// let y = pad_center(&array![1.0, 2.0], 5, Axis(0), PadMode::Constant(0.0)).unwrap();
/// assert_eq!(y, array![0.0, 1.0, 2.0, 0.0, 0.0]);
/// ```
pub fn pad_center<A, S, D>(
    data: &ArrayBase<S, D>,
    size: usize,
    axis: Axis,
    mode: PadMode<A>,
) -> AudioSampleResult<Array<A, D>>
where
    A: Clone + Default,
    S: Data<Elem = A>,
    D: Dimension + RemoveAxis,
{
    check_axis("axis", axis, data.ndim())?;
    let n = data.len_of(axis);
    if size < n {
        return invalid(
            "size",
            format!("target size {size} is below the input size {n}"),
        );
    }
    let lpad = (size - n) / 2;
    let mut dim = data.raw_dim();
    dim[axis.index()] = size;
    crate::utils::check_derived_len("size", "padding", dim.size() as f64, 1e12)?;
    if let PadMode::Constant(value) = mode {
        let mut out = Array::from_elem(dim, value);
        match (out.as_slice_mut(), data.as_slice(), data.ndim()) {
            (Some(target), Some(source), 1) => {
                target[lpad..lpad + n].clone_from_slice(source);
            }
            _ => {
                out.slice_axis_mut(axis, Slice::from(lpad..lpad + n))
                    .assign(data);
            }
        }
        return Ok(out);
    }
    if data.is_empty() {
        return invalid("data", "cannot extend an empty axis with this pad mode");
    }
    let kind = match mode {
        PadMode::Edge => PadSide::Edge,
        PadMode::Wrap => PadSide::Wrap,
        PadMode::Reflect => PadSide::Mirror { repeat_edge: false },
        _ => PadSide::Mirror { repeat_edge: true },
    };
    // A zero element is allocated without touching the pages, which the runs then fill.
    let mut out = Array::from_elem(dim, A::default());
    Zip::from(out.lanes_mut(axis))
        .and(data.lanes(axis))
        .for_each(|mut o, lane| {
            if let (Some(os), Some(ls)) = (o.as_slice_mut(), lane.as_slice()) {
                let (head, rest) = os.split_at_mut(lpad);
                let (mid, tail) = rest.split_at_mut(n);
                mid.clone_from_slice(ls);
                pad_side(head, ls, kind, true);
                pad_side(tail, ls, kind, false);
            } else {
                let copied: Vec<A> = lane.iter().cloned().collect();
                let mut buffer: Vec<A> = vec![A::default(); size];
                {
                    let (head, rest) = buffer.split_at_mut(lpad);
                    let (mid, tail) = rest.split_at_mut(n);
                    mid.clone_from_slice(&copied);
                    pad_side(head, &copied, kind, true);
                    pad_side(tail, &copied, kind, false);
                }
                for (slot, v) in o.iter_mut().zip(buffer) {
                    *slot = v;
                }
            }
        });
    Ok(out)
}

/// Normalises a set of frame indices: clips, optionally adds the bounds, removes
/// duplicates and sorts.
///
/// # Arguments
/// - `frames`: non-negative frame indices.
/// - `x_min`: lower bound; with `pad`, values below it become `x_min` and `x_min`
///   joins the set, otherwise they are dropped.
/// - `x_max`: upper bound, likewise.
/// - `pad`: clip to the bounds and include them, rather than only filter.
///
/// # Returns
/// Sorted, unique indices.
///
/// # Errors
/// [`ParameterError`] if an index is negative.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::fix_frames;
///
/// let f = fix_frames(&[3, 7, 7, 20], Some(0), Some(10), true).unwrap();
/// assert_eq!(f, vec![0, 3, 7, 10]);
/// ```
pub fn fix_frames(
    frames: &[i64],
    x_min: Option<i64>,
    x_max: Option<i64>,
    pad: bool,
) -> AudioSampleResult<Vec<i64>> {
    if frames.iter().any(|&f| f < 0) {
        return invalid("frames", "negative frame index");
    }
    let mut out: Vec<i64> = Vec::with_capacity(frames.len() + 2);
    if pad {
        out.extend(frames.iter().map(|&f| {
            let f = x_min.map_or(f, |m| f.max(m));
            x_max.map_or(f, |m| f.min(m))
        }));
        out.extend(x_min);
        out.extend(x_max);
    } else {
        out.extend_from_slice(frames);
    }
    out.retain(|&f| x_min.is_none_or(|m| f >= m) && x_max.is_none_or(|m| f <= m));
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// One slice of [`index_to_slice`]: the half-open range `start..stop`, with a step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndexSlice {
    /// First index.
    pub start: i64,
    /// One past the last index.
    pub stop: i64,
    /// Step, if one was given.
    pub step: Option<i64>,
}

/// Converts boundary indices to the slices between consecutive boundaries.
///
/// The indices are normalised with [`fix_frames`] first.
///
/// # Arguments
/// - `idx`: boundary indices.
/// - `idx_min`, `idx_max`, `pad`: as for [`fix_frames`].
/// - `step`: step recorded in each slice.
///
/// # Returns
/// One [`IndexSlice`] per pair of consecutive boundaries.
///
/// # Errors
/// [`ParameterError`] if an index is negative.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::index_to_slice;
///
/// let s = index_to_slice(&[2, 5], Some(0), Some(8), None, true).unwrap();
/// assert_eq!((s[0].start, s[0].stop), (0, 2));
/// assert_eq!(s.len(), 3);
/// ```
pub fn index_to_slice(
    idx: &[i64],
    idx_min: Option<i64>,
    idx_max: Option<i64>,
    step: Option<i64>,
    pad: bool,
) -> AudioSampleResult<Vec<IndexSlice>> {
    let fixed = fix_frames(idx, idx_min, idx_max, pad)?;
    Ok(fixed
        .windows(2)
        .map(|w| IndexSlice {
            start: w[0],
            stop: w[1],
            step,
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Reordering
// ---------------------------------------------------------------------------

/// The key [`axis_sort`] sorts by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AxisSortKey {
    /// Position of the largest value.
    #[default]
    Argmax,
    /// Position of the smallest value.
    Argmin,
}

/// Sorts the rows or columns of a matrix by the position of their peak.
///
/// # Arguments
/// - `s`: the matrix.
/// - `axis`: `Axis(1)` reorders columns, `Axis(0)` reorders rows.
/// - `key`: the position that orders them.
///
/// # Returns
/// The sorted matrix and the permutation applied. Equal keys keep their order.
///
/// # Errors
/// [`ParameterError`] if `axis` is out of range.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::{AxisSortKey, axis_sort};
/// use ndarray::{Axis, array};
///
/// let s = array![[0.0, 1.0], [1.0, 0.0]];
/// let (sorted, idx) = axis_sort(s.view(), Axis(1), AxisSortKey::Argmax).unwrap();
/// assert_eq!(idx, vec![1, 0]);
/// assert_eq!(sorted, array![[1.0, 0.0], [0.0, 1.0]]);
/// ```
pub fn axis_sort<A: PartialOrd + Clone>(
    s: ArrayView2<'_, A>,
    axis: Axis,
    key: AxisSortKey,
) -> AudioSampleResult<(Array2<A>, Vec<usize>)> {
    check_axis("axis", axis, 2)?;
    let (rows, cols) = s.dim();
    let count = s.len_of(axis);
    let better = |v: &A, b: &A| match key {
        AxisSortKey::Argmax => v > b,
        AxisSortKey::Argmin => v < b,
    };
    let mut keys = vec![0_u32; count];
    if axis.index() == 1 {
        // Walk along the rows, keeping the best value of each column, to stay in cache.
        if let Some(first) = s.rows().into_iter().next() {
            let mut best: Vec<A> = first.to_vec();
            for (r, row) in s.rows().into_iter().enumerate().skip(1) {
                for ((key_slot, best_value), v) in
                    keys.iter_mut().zip(best.iter_mut()).zip(row.iter())
                {
                    if better(v, best_value) {
                        *best_value = v.clone();
                        *key_slot = r as u32;
                    }
                }
            }
        }
    } else {
        for (slot, row) in keys.iter_mut().zip(s.rows()) {
            let mut best = 0;
            for (j, v) in row.iter().enumerate() {
                if better(v, &row[best]) {
                    best = j;
                }
            }
            *slot = best as u32;
        }
    }
    let mut idx: Vec<u32> = (0..count as u32).collect();
    idx.sort_by_key(|&i| keys[i as usize]);
    drop(keys);
    let mut flat: Vec<A> = Vec::with_capacity(s.len());
    if axis.index() == 1 {
        for row in s.rows() {
            if let Some(slice) = row.as_slice() {
                flat.extend(idx.iter().map(|&i| slice[i as usize].clone()));
            } else {
                flat.extend(idx.iter().map(|&i| row[i as usize].clone()));
            }
        }
    } else {
        for &i in &idx {
            flat.extend(s.row(i as usize).iter().cloned());
        }
    }
    let sorted =
        Array2::from_shape_vec((rows, cols), flat).or_else(|e| invalid("s", e.to_string()))?;
    Ok((sorted, idx.into_iter().map(|i| i as usize).collect()))
}

/// Circularly shifts each column of a matrix by `factor` times its index.
///
/// `factor = 1` on a recurrence matrix gives the lag matrix.
///
/// # Arguments
/// - `x`: the matrix.
/// - `factor`: shift per column; negative shifts go the other way.
/// - `axis`: `Axis(1)` shifts columns by their column index; `Axis(0)` shifts rows
///   by their row index.
///
/// # Returns
/// The sheared matrix.
///
/// # Errors
/// [`ParameterError`] if `axis` is out of range.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::shear;
/// use ndarray::{Axis, array};
///
/// let y = shear(array![[1, 0], [0, 1]].view(), 1, Axis(1)).unwrap();
/// assert_eq!(y, array![[1, 1], [0, 0]]);
/// ```
pub fn shear<A: Clone>(
    x: ArrayView2<'_, A>,
    factor: i64,
    axis: Axis,
) -> AudioSampleResult<Array2<A>> {
    check_axis("axis", axis, 2)?;
    if x.is_empty() {
        return Ok(x.to_owned());
    }
    let n = x.len_of(Axis(axis.index() ^ 1)) as i64;
    let mut out = x.to_owned();
    for (i, (mut o, lane)) in out.axis_iter_mut(axis).zip(x.axis_iter(axis)).enumerate() {
        let roll = (factor.wrapping_mul(i as i64)).rem_euclid(n) as usize;
        for (j, v) in lane.iter().enumerate() {
            let at = j + roll;
            o[if at >= n as usize {
                at - n as usize
            } else {
                at
            }] = v.clone();
        }
    }
    Ok(out)
}

/// A matrix in compressed sparse row form.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix<A> {
    /// `(rows, columns)`.
    pub shape: (usize, usize),
    /// Stored values, row by row.
    pub data: Vec<A>,
    /// Column of each stored value.
    pub indices: Vec<usize>,
    /// For each row, where its values start in `data`; one extra entry at the end.
    pub indptr: Vec<usize>,
}

impl<A: Clone + num_traits::Zero> CsrMatrix<A> {
    /// The dense form of the matrix.
    ///
    /// # Returns
    /// A `(rows, columns)` array with zeros at the entries not stored.
    #[must_use]
    pub fn to_dense(&self) -> Array2<A> {
        let mut out = Array2::zeros(self.shape);
        for r in 0..self.shape.0 {
            for k in self.indptr[r]..self.indptr[r + 1] {
                out[[r, self.indices[k]]] = self.data[k].clone();
            }
        }
        out
    }
}

/// The element types [`sparsify_rows`] can compute a magnitude for.
pub trait Magnitude: Copy + num_traits::Zero {
    /// The magnitude's float type.
    type Real: AudioFloat;
    /// `|self|`.
    fn magnitude(self) -> Self::Real;
}

impl Magnitude for f32 {
    type Real = f32;
    fn magnitude(self) -> f32 {
        self.abs()
    }
}
impl Magnitude for f64 {
    type Real = f64;
    fn magnitude(self) -> f64 {
        self.abs()
    }
}
impl Magnitude for Complex<f32> {
    type Real = f32;
    fn magnitude(self) -> f32 {
        self.norm()
    }
}
impl Magnitude for Complex<f64> {
    type Real = f64;
    fn magnitude(self) -> f64 {
        self.norm()
    }
}

/// Removes the smallest values of each row, keeping a sparse matrix.
///
/// In each row the sorted magnitudes are divided by the row's total and summed
/// cumulatively; entries below the first magnitude whose running total reaches
/// `quantile` are dropped.
///
/// # Arguments
/// - `x`: the matrix.
/// - `quantile`: the fraction of each row's total magnitude that may be discarded,
///   in `[0, 1)`.
///
/// # Returns
/// The kept entries as a [`CsrMatrix`]. Zero entries are not stored.
///
/// # Errors
/// [`ParameterError`] if `quantile` is outside `[0, 1)`.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::sparsify_rows;
/// use ndarray::array;
///
/// let m = sparsify_rows(array![[0.01, 1.0, 2.0]].view(), 0.1).unwrap();
/// assert_eq!(m.indices, vec![1, 2]);
/// ```
pub fn sparsify_rows<A: Magnitude>(
    x: ArrayView2<'_, A>,
    quantile: f64,
) -> AudioSampleResult<CsrMatrix<A>> {
    if !(0.0..1.0).contains(&quantile) {
        return invalid("quantile", format!("must lie in [0, 1), got {quantile}"));
    }
    let q: A::Real = to_float(quantile);
    let zero = <A::Real as num_traits::Zero>::zero();
    let (rows, cols) = x.dim();
    let mut sorted: Vec<A::Real> = Vec::with_capacity(cols);
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(rows + 1);
    indptr.push(0);
    for row in x.rows() {
        sorted.clear();
        sorted.extend(row.iter().map(|v| v.magnitude()));
        let norm = sorted.iter().fold(zero, |a, &b| a + b);
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let mut cumulative = zero;
        // No running total reaches the quantile: everything is kept.
        let mut threshold = sorted.first().copied().unwrap_or(zero);
        for &m in &sorted {
            cumulative += m / norm;
            // A NaN total (a row of zeros) stops at the first entry.
            if num_traits::Float::is_nan(cumulative) || cumulative >= q {
                threshold = m;
                break;
            }
        }
        for (j, v) in row.iter().enumerate() {
            let m = v.magnitude();
            if m >= threshold && m != zero {
                data.push(*v);
                indices.push(j);
            }
        }
        indptr.push(data.len());
    }
    Ok(CsrMatrix {
        shape: (rows, cols),
        data,
        indices,
        indptr,
    })
}

/// Central-difference gradient of periodic data along one axis.
///
/// The element after the last is the first, so the result is
/// `(x[i + 1] - x[i - 1]) / 2` with wrapped indices.
///
/// # Arguments
/// - `data`: the array.
/// - `axis`: the periodic axis.
///
/// # Returns
/// An array of the same shape.
///
/// # Errors
/// [`ParameterError`] if `axis` is out of range.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::cyclic_gradient;
/// use ndarray::{Axis, array};
///
/// let g = cyclic_gradient(&array![0.0, 1.0, 2.0, 3.0], Axis(0)).unwrap();
/// assert_eq!(g, array![-1.0, 1.0, 1.0, -1.0]);
/// ```
pub fn cyclic_gradient<F, S, D>(
    data: &ArrayBase<S, D>,
    axis: Axis,
) -> AudioSampleResult<Array<F, D>>
where
    F: AudioFloat,
    S: Data<Elem = F>,
    D: Dimension + RemoveAxis,
{
    check_axis("axis", axis, data.ndim())?;
    let half: F = to_float(0.5);
    let mut out = Array::from_elem(data.raw_dim(), F::zero());
    Zip::from(out.lanes_mut(axis))
        .and(data.lanes(axis))
        .for_each(|mut o, lane| {
            let n = lane.len();
            for i in 0..n {
                let up = lane[if i + 1 == n { 0 } else { i + 1 }];
                let down = lane[if i == 0 { n - 1 } else { i - 1 }];
                o[i] = (up - down) * half;
            }
        });
    Ok(out)
}

/// Sets the entries of a matrix outside a diagonal band to a value (the
/// Sakoe-Chiba band of dynamic time warping).
///
/// # Arguments
/// - `x`: the matrix, changed in place.
/// - `radius`: half-width of the band as a fraction of the shorter side.
/// - `value`: the value written outside the band.
///
/// # Errors
/// [`ParameterError`] if `radius` is negative or not finite.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::fill_off_diagonal;
/// use ndarray::Array2;
///
/// let mut x = Array2::from_elem((4, 4), 1.0);
/// fill_off_diagonal(x.view_mut(), 0.25, 0.0).unwrap();
/// assert_eq!(x[[0, 0]], 1.0);
/// assert_eq!(x[[0, 1]], 0.0);
/// ```
pub fn fill_off_diagonal<A: Clone>(
    mut x: ArrayViewMut2<'_, A>,
    radius: f64,
    value: A,
) -> AudioSampleResult<()> {
    if !radius.is_finite() || radius < 0.0 {
        return invalid(
            "radius",
            format!("must be finite and non-negative, got {radius}"),
        );
    }
    let (nx, ny) = x.dim();
    let r = round_half_even(radius * nx.min(ny) as f64) as i64;
    let offset = nx.abs_diff(ny) as i64;
    let (upper, lower) = if nx < ny {
        (r + offset, r)
    } else {
        (r, r + offset)
    };
    for (i, mut row) in x.rows_mut().into_iter().enumerate() {
        let i = i as i64;
        // Columns at or below the diagonal `-lower`, and at or above the diagonal `upper`.
        let lo_end = (i - lower + 1).clamp(0, ny as i64) as usize;
        let hi_start = (i + upper).clamp(0, ny as i64) as usize;
        for v in row.iter_mut().take(lo_end) {
            *v = value.clone();
        }
        for v in row.iter_mut().skip(hi_start) {
            *v = value.clone();
        }
    }
    Ok(())
}

fn round_half_even(v: f64) -> f64 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - v.signum()
    } else {
        r
    }
}

/// The element types [`count_unique`] and [`is_unique`] can compare.
pub trait UniqueKey: Copy + Default {
    /// Total order in which equal values, and all NaNs, are adjacent.
    fn order(&self, other: &Self) -> Ordering;
    /// Whether two values count as one; NaNs do.
    fn same(&self, other: &Self) -> bool;
}

macro_rules! unique_float {
    ($($t:ty),*) => {$(
        impl UniqueKey for $t {
            fn order(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or_else(|| self.is_nan().cmp(&other.is_nan()))
            }
            fn same(&self, other: &Self) -> bool {
                self == other || (self.is_nan() && other.is_nan())
            }
        }
    )*};
}
macro_rules! unique_ord {
    ($($t:ty),*) => {$(
        impl UniqueKey for $t {
            fn order(&self, other: &Self) -> Ordering {
                self.cmp(other)
            }
            fn same(&self, other: &Self) -> bool {
                self == other
            }
        }
    )*};
}
unique_float!(f32, f64);
unique_ord!(bool, i8, i16, i32, i64, u8, u16, u32, u64);

impl<F: UniqueKey> UniqueKey for Complex<F> {
    fn order(&self, other: &Self) -> Ordering {
        self.re
            .order(&other.re)
            .then_with(|| self.im.order(&other.im))
    }
    fn same(&self, other: &Self) -> bool {
        self.re.same(&other.re) && self.im.same(&other.im)
    }
}

/// Lanes up to this length are sorted on the stack.
const STACK_LANE: usize = 512;

fn distinct_values<A: UniqueKey>(lane: ArrayView1<'_, A>, buffer: &mut Vec<A>) -> usize {
    fn count_sorted<A: UniqueKey>(values: &mut [A]) -> usize {
        values.sort_unstable_by(A::order);
        let mut count = usize::from(!values.is_empty());
        for pair in values.windows(2) {
            count += usize::from(!pair[0].same(&pair[1]));
        }
        count
    }
    if lane.len() <= STACK_LANE {
        let mut stack = [A::default(); STACK_LANE];
        let used = &mut stack[..lane.len()];
        for (slot, v) in used.iter_mut().zip(lane.iter()) {
            *slot = *v;
        }
        return count_sorted(used);
    }
    buffer.clear();
    buffer.extend(lane.iter().copied());
    count_sorted(buffer)
}

/// Counts the distinct values in each lane of an array along one axis.
///
/// # Arguments
/// - `data`: the array.
/// - `axis`: the axis of the lanes.
///
/// # Returns
/// An array with `axis` removed. All NaNs count as one value.
///
/// # Errors
/// [`ParameterError`] if `axis` is out of range.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::count_unique;
/// use ndarray::{Axis, array};
///
/// let c = count_unique(&array![[0, 1, 1], [2, 2, 2]], Axis(1)).unwrap();
/// assert_eq!(c, array![2, 1]);
/// ```
pub fn count_unique<A, S, D>(
    data: &ArrayBase<S, D>,
    axis: Axis,
) -> AudioSampleResult<Array<i64, D::Smaller>>
where
    A: UniqueKey,
    S: Data<Elem = A>,
    D: Dimension + RemoveAxis,
{
    check_axis("axis", axis, data.ndim())?;
    let mut buffer = Vec::new();
    let mut out = Array::from_elem(data.raw_dim().remove_axis(axis), 0_i64);
    Zip::from(&mut out)
        .and(data.lanes(axis))
        .for_each(|o, lane| *o = distinct_values(lane, &mut buffer) as i64);
    Ok(out)
}

/// Tells for each lane along an axis whether all its values are distinct.
///
/// # Arguments
/// - `data`: the array.
/// - `axis`: the axis of the lanes.
///
/// # Returns
/// A boolean array with `axis` removed.
///
/// # Errors
/// [`ParameterError`] if `axis` is out of range.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::is_unique;
/// use ndarray::{Axis, array};
///
/// let u = is_unique(&array![[0, 1, 2], [2, 2, 2]], Axis(1)).unwrap();
/// assert_eq!(u, array![true, false]);
/// ```
pub fn is_unique<A, S, D>(
    data: &ArrayBase<S, D>,
    axis: Axis,
) -> AudioSampleResult<Array<bool, D::Smaller>>
where
    A: UniqueKey,
    S: Data<Elem = A>,
    D: Dimension + RemoveAxis,
{
    check_axis("axis", axis, data.ndim())?;
    let n = data.len_of(axis);
    let mut buffer = Vec::new();
    let mut out = Array::from_elem(data.raw_dim().remove_axis(axis), false);
    Zip::from(&mut out)
        .and(data.lanes(axis))
        .for_each(|o, lane| *o = distinct_values(lane, &mut buffer) == n);
    Ok(out)
}

// ---------------------------------------------------------------------------
// One-dimensional interpolation
// ---------------------------------------------------------------------------

/// The interpolant of [`interp1d`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpKind {
    /// Piecewise linear.
    #[default]
    Linear,
    /// Nearest sample; a point half-way between two samples takes the earlier.
    Nearest,
    /// Nearest sample; a point half-way between two samples takes the later.
    NearestUp,
    /// Piecewise constant B-spline of degree zero.
    Zero,
    /// Linear B-spline.
    SLinear,
    /// Quadratic not-a-knot B-spline.
    Quadratic,
    /// Cubic not-a-knot B-spline.
    Cubic,
    /// The sample at or before each point.
    Previous,
    /// The sample at or after each point.
    Next,
}

impl InterpKind {
    /// Least number of sample positions the interpolant needs.
    #[must_use]
    pub const fn min_points(self) -> usize {
        match self {
            Self::Linear | Self::SLinear | Self::Zero => 2,
            Self::Nearest | Self::NearestUp | Self::Previous | Self::Next => 1,
            Self::Quadratic => 3,
            Self::Cubic => 4,
        }
    }
}

/// The values of the `k + 1` B-splines of degree `k` that are non-zero at `x` in
/// knot span `span` (the Cox-de Boor recurrence).
fn bspline_basis(t: &[f64], k: usize, span: usize, x: f64, out: &mut [f64; 4]) {
    let mut left = [0.0_f64; 4];
    let mut right = [0.0_f64; 4];
    out[0] = 1.0;
    for j in 1..=k {
        left[j] = x - t[span + 1 - j];
        right[j] = t[span + j] - x;
        let mut saved = 0.0;
        for r in 0..j {
            let tmp = out[r] / (right[r + 1] + left[j - r]);
            out[r] = saved + right[r + 1] * tmp;
            saved = left[j - r] * tmp;
        }
        out[j] = saved;
    }
}

fn knot_span(t: &[f64], k: usize, n: usize, x: f64) -> usize {
    let at = t[k..=n].partition_point(|&v| v <= x) + k;
    at.saturating_sub(1).clamp(k, n - 1)
}

/// A not-a-knot interpolating spline of degree 2 or 3, ready to solve for any number
/// of sample vectors.
struct SplineSolver {
    knots: Vec<f64>,
    k: usize,
    n: usize,
    /// Band of the factored collocation matrix, `(2k + 1)` values per row.
    band: Vec<f64>,
}

impl SplineSolver {
    fn new(x: &[f64], k: usize) -> AudioSampleResult<Self> {
        let n = x.len();
        if x.windows(2).any(|w| w[0] == w[1]) {
            return invalid("x", "spline interpolation needs distinct sample positions");
        }
        let mut knots = Vec::with_capacity(n + k + 1);
        knots.extend(std::iter::repeat_n(x[0], k + 1));
        if k == 3 {
            knots.extend_from_slice(&x[2..n - 2]);
        } else {
            knots.extend((1..n - 2).map(|i| 0.5 * (x[i] + x[i + 1])));
        }
        knots.extend(std::iter::repeat_n(x[n - 1], k + 1));

        let width = 2 * k + 1;
        let mut band = vec![0.0; n * width];
        let mut basis = [0.0; 4];
        for (i, &xi) in x.iter().enumerate() {
            let span = knot_span(&knots, k, n, xi);
            bspline_basis(&knots, k, span, xi, &mut basis);
            for (j, &b) in basis.iter().enumerate().take(k + 1) {
                let col = span + j - k;
                band[i * width + col + k - i] = b;
            }
        }
        // Band elimination without pivoting: the matrix is totally positive.
        for p in 0..n {
            let pivot = band[p * width + k];
            if pivot == 0.0 {
                return invalid("x", "the spline system is singular");
            }
            for r in p + 1..(p + k + 1).min(n) {
                let f = band[r * width + p + k - r] / pivot;
                if f == 0.0 {
                    continue;
                }
                band[r * width + p + k - r] = f;
                for c in p + 1..(p + k + 1).min(n) {
                    band[r * width + c + k - r] -= f * band[p * width + c + k - p];
                }
            }
        }
        Ok(Self { knots, k, n, band })
    }

    fn solve(&self, c: &mut [f64]) {
        let (k, n) = (self.k, self.n);
        let width = 2 * k + 1;
        for r in 1..n {
            let mut s = c[r];
            for (p, v) in c.iter().enumerate().take(r).skip(r.saturating_sub(k)) {
                s -= self.band[r * width + p + k - r] * v;
            }
            c[r] = s;
        }
        for r in (0..n).rev() {
            let mut s = c[r];
            for (p, v) in c.iter().enumerate().take((r + k + 1).min(n)).skip(r + 1) {
                s -= self.band[r * width + p + k - r] * v;
            }
            c[r] = s / self.band[r * width + k];
        }
    }
}

/// `(span, basis)` of each query point, or `None` to fill.
type SplinePoints = Vec<Option<(usize, [f64; 4])>>;

/// Where each query point falls, independent of the sample values.
enum Plan {
    /// `(index of the sample below, fraction towards the next)`, or `None` to fill.
    Linear(Vec<Option<(usize, f64)>>),
    /// Index of the sample to copy, or `None` to fill.
    Copy(Vec<Option<usize>>),
    /// The solver, and with more than one lane `(span, basis)` per point; a single lane
    /// evaluates its points as it goes.
    Spline(SplineSolver, Option<SplinePoints>),
}

fn build_plan(x: &[f64], xi: &[f64], kind: InterpKind) -> AudioSampleResult<Plan> {
    let n = x.len();
    let (lo, hi) = (x[0], x[n - 1]);
    let inside = |v: f64| v >= lo && v <= hi;
    Ok(match kind {
        InterpKind::Linear | InterpKind::SLinear => Plan::Linear(
            xi.iter()
                .map(|&v| {
                    inside(v).then(|| {
                        let at = x.partition_point(|&s| s < v).clamp(1, n - 1);
                        (at - 1, (v - x[at - 1]) / (x[at] - x[at - 1]))
                    })
                })
                .collect(),
        ),
        InterpKind::Zero => Plan::Copy(
            xi.iter()
                .map(|&v| {
                    inside(v).then(|| {
                        // The last sample closes the last interval, whose value it takes.
                        x.partition_point(|&s| s <= v).saturating_sub(1).min(n - 2)
                    })
                })
                .collect(),
        ),
        InterpKind::Nearest | InterpKind::NearestUp => {
            let up = kind == InterpKind::NearestUp;
            let mids: Vec<f64> = x.windows(2).map(|w| 0.5 * w[0] + 0.5 * w[1]).collect();
            Plan::Copy(
                xi.iter()
                    .map(|&v| {
                        inside(v)
                            .then(|| mids.partition_point(|&m| if up { m <= v } else { m < v }))
                    })
                    .collect(),
            )
        }
        InterpKind::Previous => Plan::Copy(
            xi.iter()
                .map(|&v| inside(v).then(|| x.partition_point(|&s| s <= v).saturating_sub(1)))
                .collect(),
        ),
        InterpKind::Next => Plan::Copy(
            xi.iter()
                .map(|&v| inside(v).then(|| x.partition_point(|&s| s < v).min(n - 1)))
                .collect(),
        ),
        InterpKind::Quadratic | InterpKind::Cubic => {
            let k = if kind == InterpKind::Cubic { 3 } else { 2 };
            Plan::Spline(SplineSolver::new(x, k)?, None)
        }
    })
}

/// Interpolates sampled data along one axis.
///
/// # Arguments
/// - `x`: sample positions, in any order (they are sorted together with the data).
///   Spline kinds need distinct positions.
/// - `y`: the samples; `axis` has one element per position.
/// - `xi`: query positions, of any shape.
/// - `kind`: the interpolant.
/// - `fill_value`: result outside `x[0]..=x[n - 1]`.
/// - `axis`: the axis of `y` that follows `x`.
///
/// # Returns
/// An array shaped like `y` with `axis` replaced by the axes of `xi`. A NaN query
/// gives NaN.
///
/// # Errors
/// [`ParameterError`] if the lengths disagree, there are too few positions for
/// `kind`, a position is not finite, or `axis` is out of range.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::{InterpKind, interp1d};
/// use ndarray::{Axis, array};
///
/// let y = array![0.0_f64, 1.0, 8.0, 27.0].into_dyn();
/// let xi = array![1.5].into_dyn();
/// let v = interp1d(&[0.0, 1.0, 2.0, 3.0], y.view(), xi.view(), InterpKind::Cubic, 0.0, Axis(0)).unwrap();
/// assert!((v[[0]] - 3.375).abs() < 1e-12);
/// ```
pub fn interp1d<F: AudioFloat>(
    x: &[f64],
    y: ArrayViewD<'_, F>,
    xi: ArrayViewD<'_, f64>,
    kind: InterpKind,
    fill_value: F,
    axis: Axis,
) -> AudioSampleResult<ArrayD<F>> {
    check_axis("axis", axis, y.ndim())?;
    if x.len() != y.len_of(axis) {
        return invalid(
            "x",
            format!(
                "{} positions for {} samples on the axis",
                x.len(),
                y.len_of(axis)
            ),
        );
    }
    if x.len() < kind.min_points() {
        return invalid(
            "x",
            format!(
                "{kind:?} interpolation needs {} positions",
                kind.min_points()
            ),
        );
    }
    if x.iter().any(|v| !v.is_finite()) {
        return invalid("x", "positions must be finite");
    }
    let is_sorted = x.windows(2).all(|w| w[0] <= w[1]);
    let sorted_y;
    let sorted_x: Vec<f64>;
    let (x, y) = if is_sorted {
        (x, y)
    } else {
        let order = sorted_index(x);
        sorted_x = order.iter().map(|&i| x[i]).collect();
        sorted_y = y.select(axis, &order);
        (&sorted_x[..], sorted_y.view())
    };

    let flat: Vec<f64>;
    let xi_flat: &[f64] = if let Some(s) = xi.as_slice() {
        s
    } else {
        flat = xi.iter().copied().collect();
        &flat
    };
    let mut plan = build_plan(x, xi_flat, kind)?;
    if let Plan::Spline(solver, points) = &mut plan
        && y.len() != y.len_of(axis)
    {
        let (first, last) = (x[0], x[x.len() - 1]);
        let k = solver.k;
        *points = Some(
            xi_flat
                .iter()
                .map(|&v| {
                    (v >= first && v <= last).then(|| {
                        let span = knot_span(&solver.knots, k, solver.n, v);
                        let mut basis = [0.0; 4];
                        bspline_basis(&solver.knots, k, span, v, &mut basis);
                        (span, basis)
                    })
                })
                .collect(),
        );
    }
    let nans: Vec<usize> = xi_flat
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.is_nan().then_some(i))
        .collect();

    let mut shape = y.shape().to_vec();
    shape[axis.index()] = xi_flat.len();
    let mut out = ArrayD::from_elem(IxDyn(&shape), fill_value);
    let mut coefficients = vec![0.0; x.len()];
    Zip::from(out.lanes_mut(axis))
        .and(y.lanes(axis))
        .for_each(|mut o, lane| {
            match &plan {
                Plan::Linear(points) => {
                    for (slot, p) in o.iter_mut().zip(points) {
                        if let Some((i, frac)) = *p {
                            *slot = lane[i] + (lane[i + 1] - lane[i]) * to_float::<F>(frac);
                        }
                    }
                }
                Plan::Copy(points) => {
                    for (slot, p) in o.iter_mut().zip(points) {
                        if let Some(i) = *p {
                            *slot = lane[i];
                        }
                    }
                }
                Plan::Spline(solver, Some(points)) => {
                    for (c, v) in coefficients.iter_mut().zip(lane.iter()) {
                        *c = v.to_f64().unwrap_or(f64::NAN);
                    }
                    solver.solve(&mut coefficients);
                    let k = solver.k;
                    for (slot, point) in o.iter_mut().zip(points) {
                        if let Some((span, basis)) = point {
                            let mut sum = 0.0;
                            for (j, b) in basis.iter().enumerate().take(k + 1) {
                                sum += b * coefficients[span + j - k];
                            }
                            *slot = to_float::<F>(sum);
                        }
                    }
                }
                Plan::Spline(solver, None) => {
                    for (c, v) in coefficients.iter_mut().zip(lane.iter()) {
                        *c = v.to_f64().unwrap_or(f64::NAN);
                    }
                    solver.solve(&mut coefficients);
                    let k = solver.k;
                    let mut basis = [0.0; 4];
                    let mut span = k;
                    for (slot, &v) in o.iter_mut().zip(xi_flat) {
                        if v >= x[0] && v <= x[x.len() - 1] {
                            // Queries are usually in order, so try the last span first.
                            if !(solver.knots[span] <= v && v < solver.knots[span + 1]) {
                                span = knot_span(&solver.knots, k, solver.n, v);
                            }
                            bspline_basis(&solver.knots, k, span, v, &mut basis);
                            let mut sum = 0.0;
                            for (j, b) in basis.iter().enumerate().take(k + 1) {
                                sum += b * coefficients[span + j - k];
                            }
                            *slot = to_float::<F>(sum);
                        }
                    }
                }
            }
            for &i in &nans {
                o[i] = F::nan();
            }
        });
    if xi.ndim() == 1 {
        return Ok(out);
    }
    let mut final_shape = y.shape()[..axis.index()].to_vec();
    final_shape.extend_from_slice(xi.shape());
    final_shape.extend_from_slice(&y.shape()[axis.index() + 1..]);
    out.into_shape_with_order(IxDyn(&final_shape))
        .or_else(|e| invalid("xi", e.to_string()))
}

/// Interpolates two arrays sampled on different grids onto one common grid.
///
/// # Arguments
/// - `x1`, `x1_pos`: the first array and its sample positions along `axis`.
/// - `x2`, `x2_pos`: the second array and its positions.
/// - `interp_pos`: the common positions; `None` uses `x1_pos`.
/// - `kind`, `fill_value`, `axis`: as for [`interp1d`]. `axis` may be negative.
///
/// # Returns
/// The two interpolated arrays, ready to combine element by element.
///
/// # Errors
/// As for [`interp1d`], and [`ParameterError`] if `axis` is out of range for the
/// smaller array.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::{InterpKind, interp_broadcast};
/// use ndarray::array;
///
/// let a = array![1.0, 2.0].into_dyn();
/// let b = array![10.0, 30.0, 50.0].into_dyn();
/// let (ya, yb) = interp_broadcast(a.view(), &[0.0, 1.0], b.view(), &[0.0, 0.5, 1.0],
///     None, InterpKind::Linear, 0.0, -1).unwrap();
/// assert_eq!(ya.as_slice().unwrap(), &[1.0, 2.0]);
/// assert_eq!(yb.as_slice().unwrap(), &[10.0, 50.0]);
/// ```
pub fn interp_broadcast<F: AudioFloat>(
    x1: ArrayViewD<'_, F>,
    x1_pos: &[f64],
    x2: ArrayViewD<'_, F>,
    x2_pos: &[f64],
    interp_pos: Option<ArrayViewD<'_, f64>>,
    kind: InterpKind,
    fill_value: F,
    axis: isize,
) -> AudioSampleResult<(ArrayD<F>, ArrayD<F>)> {
    let min_ndim = x1.ndim().min(x2.ndim()) as isize;
    if axis < -min_ndim || axis >= min_ndim {
        return invalid(
            "axis",
            format!("{axis} is out of range for {min_ndim} dimensions"),
        );
    }
    let resolve = |ndim: usize| Axis(if axis < 0 { axis + ndim as isize } else { axis } as usize);
    let xi = interp_pos.unwrap_or_else(|| ArrayView1::from(x1_pos).into_dyn());
    let y1 = interp1d(
        x1_pos,
        x1.view(),
        xi.view(),
        kind,
        fill_value,
        resolve(x1.ndim()),
    )?;
    let y2 = interp1d(
        x2_pos,
        x2.view(),
        xi.view(),
        kind,
        fill_value,
        resolve(x2.ndim()),
    )?;
    Ok((y1, y2))
}

// ---------------------------------------------------------------------------
// Decoding and checks
// ---------------------------------------------------------------------------

/// Decodes little-endian signed integer PCM bytes to float samples in `[-1, 1)`.
///
/// The scale is `2^-(8 n - 1)`, so the most negative code maps to exactly `-1`.
///
/// # Arguments
/// - `buffer`: the bytes.
/// - `n_bytes`: bytes per sample: 1, 2, 3, 4 or 8.
///
/// # Returns
/// One float per sample.
///
/// # Errors
/// [`ParameterError`] if `n_bytes` is another value or `buffer` is not a whole
/// number of samples.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::buf_to_float;
///
/// let y: Vec<f32> = buf_to_float(&[0x00, 0x80, 0x00, 0x40], 2).unwrap();
/// assert_eq!(y, vec![-1.0, 0.5]);
/// ```
pub fn buf_to_float<F: AudioFloat>(buffer: &[u8], n_bytes: usize) -> AudioSampleResult<Vec<F>> {
    check_pcm(buffer, n_bytes)?;
    let mut out = vec![F::zero(); buffer.len() / n_bytes];
    buf_to_float_into(buffer, n_bytes, &mut out)?;
    Ok(out)
}

fn check_pcm(buffer: &[u8], n_bytes: usize) -> AudioSampleResult<()> {
    if !matches!(n_bytes, 1 | 2 | 3 | 4 | 8) {
        return invalid("n_bytes", format!("must be 1, 2, 3, 4 or 8, got {n_bytes}"));
    }
    if !buffer.len().is_multiple_of(n_bytes) {
        return invalid(
            "buffer",
            format!(
                "{} bytes is not a whole number of {n_bytes}-byte samples",
                buffer.len()
            ),
        );
    }
    Ok(())
}

/// [`buf_to_float`] writing into an existing slice.
///
/// # Arguments
/// - `buffer`: the bytes.
/// - `n_bytes`: bytes per sample: 1, 2, 3, 4 or 8.
/// - `out`: receives one float per sample.
///
/// # Errors
/// As for [`buf_to_float`], and [`ParameterError`] if `out` has the wrong length.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::buf_to_float_into;
///
/// let mut y = [0.0_f32; 1];
/// buf_to_float_into(&[0x00, 0x40], 2, &mut y).unwrap();
/// assert_eq!(y[0], 0.5);
/// ```
pub fn buf_to_float_into<F: AudioFloat>(
    buffer: &[u8],
    n_bytes: usize,
    out: &mut [F],
) -> AudioSampleResult<()> {
    check_pcm(buffer, n_bytes)?;
    if out.len() != buffer.len() / n_bytes {
        return invalid("out", "must have one element per sample");
    }
    let scale = F::one() / to_float::<F>((1_u64 << (8 * n_bytes - 1)) as f64);
    let chunks = buffer.chunks_exact(n_bytes);
    match n_bytes {
        1 => {
            for (o, c) in out.iter_mut().zip(chunks) {
                *o = to_float::<F>(f64::from(c[0] as i8)) * scale;
            }
        }
        2 => {
            for (o, c) in out.iter_mut().zip(chunks) {
                *o = to_float::<F>(f64::from(i16::from_le_bytes([c[0], c[1]]))) * scale;
            }
        }
        3 => {
            for (o, c) in out.iter_mut().zip(chunks) {
                let v = i32::from_le_bytes([0, c[0], c[1], c[2]]) >> 8;
                *o = to_float::<F>(f64::from(v)) * scale;
            }
        }
        4 => {
            for (o, c) in out.iter_mut().zip(chunks) {
                let v = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                *o = <F as num_traits::NumCast>::from(v).unwrap_or_else(F::zero) * scale;
            }
        }
        _ => {
            for (o, c) in out.iter_mut().zip(chunks) {
                let mut b = [0_u8; 8];
                b.copy_from_slice(c);
                *o = <F as num_traits::NumCast>::from(i64::from_le_bytes(b))
                    .unwrap_or_else(F::zero)
                    * scale;
            }
        }
    }
    Ok(())
}

fn to_float<F: AudioFloat>(v: f64) -> F {
    <F as num_traits::NumCast>::from(v).unwrap_or_else(F::nan)
}

/// Checks that audio samples are at least one-dimensional and finite everywhere.
///
/// # Arguments
/// - `y`: the samples.
///
/// # Errors
/// [`ParameterError`] on a zero-dimensional array or a NaN or infinite sample.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::valid_audio;
/// use ndarray::array;
///
/// assert!(valid_audio(&array![0.0, 1.0]).is_ok());
/// assert!(valid_audio(&array![0.0, f64::NAN]).is_err());
/// ```
pub fn valid_audio<F, S, D>(y: &ArrayBase<S, D>) -> AudioSampleResult<()>
where
    F: AudioFloat,
    S: Data<Elem = F>,
    D: Dimension,
{
    if y.ndim() == 0 {
        return invalid("y", "audio data must be at least one-dimensional");
    }
    let finite = y
        .as_slice_memory_order()
        .map_or_else(|| y.iter().all(|v| v.is_finite()), all_finite);
    if !finite {
        return invalid("y", "audio buffer is not finite everywhere");
    }
    Ok(())
}

/// `x - x` is zero for a finite value and NaN otherwise, so the bits of the difference
/// can be merged without a branch per element, which the compiler vectorises.
#[allow(clippy::eq_op)]
fn all_finite<F: AudioFloat>(values: &[F]) -> bool {
    use std::any::TypeId;
    const BLOCK: usize = 1024;
    if TypeId::of::<F>() == TypeId::of::<f32>() {
        // SAFETY: F is f32, so the layout is the same.
        let v = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<f32>(), values.len()) };
        return v
            .chunks(BLOCK)
            .all(|c| c.iter().fold(0_u32, |a, x| a | (x - x).to_bits()) == 0);
    }
    if TypeId::of::<F>() == TypeId::of::<f64>() {
        // SAFETY: F is f64, so the layout is the same.
        let v = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<f64>(), values.len()) };
        return v
            .chunks(BLOCK)
            .all(|c| c.iter().fold(0_u64, |a, x| a | (x - x).to_bits()) == 0);
    }
    values.iter().all(|v| v.is_finite())
}

/// Marks the sign changes of a signal along one axis.
///
/// Values within `threshold` of zero count as zero. An element is marked when its sign
/// differs from the previous element's.
///
/// # Arguments
/// - `y`: the signal, of any dimension.
/// - `threshold`: magnitudes up to this are zero; not negative.
/// - `pad`: the mark of the first element.
/// - `zero_pos`: zero counts as positive; otherwise it is a sign of its own.
/// - `axis`: the time axis.
///
/// # Returns
/// A boolean array of the same shape.
///
/// # Errors
/// [`ParameterError`] if `axis` is out of range or `threshold` is negative or not finite.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::zero_crossings;
/// use ndarray::{Axis, array};
///
/// let z = zero_crossings(&array![1.0, -1.0, 2.0], 0.0, true, true, Axis(0)).unwrap();
/// assert_eq!(z, array![true, true, true]);
/// ```
pub fn zero_crossings<F, S, D>(
    y: &ArrayBase<S, D>,
    threshold: F,
    pad: bool,
    zero_pos: bool,
    axis: Axis,
) -> AudioSampleResult<Array<bool, D>>
where
    F: AudioFloat,
    S: Data<Elem = F>,
    D: Dimension + RemoveAxis,
{
    check_axis("axis", axis, y.ndim())?;
    if !threshold.is_finite() || threshold < F::zero() {
        return invalid("threshold", "must be finite and not negative");
    }
    let sign = |v: F| -> i8 {
        let v = if v.abs() <= threshold { F::zero() } else { v };
        if zero_pos {
            if v.is_sign_negative() && v != F::zero() {
                -1
            } else {
                1
            }
        } else if v > F::zero() {
            1
        } else if v < F::zero() {
            -1
        } else {
            0
        }
    };
    let mut out = Array::from_elem(y.raw_dim(), false);
    Zip::from(out.lanes_mut(axis))
        .and(y.lanes(axis))
        .for_each(|mut o, lane| {
            let mut previous = None;
            for (slot, &v) in o.iter_mut().zip(lane.iter()) {
                let current = sign(v);
                *slot = previous.map_or(pad, |p| p != current);
                previous = Some(current);
            }
        });
    Ok(out)
}

/// A cosine of a given frequency and phase.
///
/// # Arguments
/// - `frequency`: in Hz.
/// - `sample_rate`: in Hz.
/// - `length`: samples.
/// - `phi`: phase offset in radians; `-pi / 2` starts like a sine.
///
/// # Returns
/// The samples.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::tone;
///
/// let y: Vec<f64> = tone(100.0, 400.0, 4, 0.0);
/// assert!((y[0] - 1.0).abs() < 1e-12);
/// assert!(y[1].abs() < 1e-12);
/// ```
#[must_use]
pub fn tone<F: AudioFloat>(frequency: f64, sample_rate: f64, length: usize, phi: f64) -> Vec<F> {
    let w = std::f64::consts::TAU * frequency / sample_rate;
    (0..length)
        .map(|i| to_float((w * i as f64 + phi).cos()))
        .collect()
}

/// A swept cosine from `fmin` to `fmax` over its length.
///
/// # Arguments
/// - `fmin`: starting frequency in Hz.
/// - `fmax`: frequency at the end in Hz.
/// - `sample_rate`: in Hz.
/// - `length`: samples.
/// - `linear`: sweep linearly; otherwise exponentially.
/// - `phi`: phase offset in radians.
///
/// # Returns
/// The samples.
///
/// # Errors
/// [`ParameterError`] if an exponential sweep has a frequency that is not positive.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::chirp;
///
/// let y: Vec<f64> = chirp(110.0, 880.0, 8000.0, 800, false, 0.0).unwrap();
/// assert_eq!(y.len(), 800);
/// assert!((y[0] - 1.0).abs() < 1e-12);
/// ```
pub fn chirp<F: AudioFloat>(
    fmin: f64,
    fmax: f64,
    sample_rate: f64,
    length: usize,
    linear: bool,
    phi: f64,
) -> AudioSampleResult<Vec<F>> {
    if !linear && (fmin.is_nan() || fmax.is_nan() || fmin <= 0.0 || fmax <= 0.0) {
        return invalid("fmin", "an exponential sweep needs positive frequencies");
    }
    let duration = length as f64 / sample_rate;
    let tau = std::f64::consts::TAU;
    let ratio = fmax / fmin;
    Ok((0..length)
        .map(|i| {
            let t = i as f64 / sample_rate;
            let phase = if linear {
                tau * (fmin * t + 0.5 * (fmax - fmin) / duration * t * t)
            } else if fmin == fmax {
                tau * fmin * t
            } else {
                tau * fmin * duration / ratio.ln() * (ratio.powf(t / duration) - 1.0)
            };
            to_float((phase + phi).cos())
        })
        .collect())
}

/// Whether an integer is positive.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::is_positive_int;
///
/// assert!(is_positive_int(3));
/// assert!(!is_positive_int(0));
/// ```
#[must_use]
pub const fn is_positive_int(x: i64) -> bool {
    x > 0
}

/// How [`valid_int`] rounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IntCast {
    /// Towards negative infinity.
    #[default]
    Floor,
    /// Towards positive infinity.
    Ceil,
    /// To the nearest integer, halves away from zero.
    Round,
    /// Towards zero.
    Trunc,
}

/// Converts a float to an integer with a chosen rounding.
///
/// # Arguments
/// - `x`: the value.
/// - `cast`: the rounding.
///
/// # Errors
/// [`ParameterError`] if `x` is not finite or does not fit `i64`.
///
/// # Examples
/// ```
/// use audio_samples::utils::array_ops::{IntCast, valid_int};
///
/// assert_eq!(valid_int(2.7, IntCast::Floor).unwrap(), 2);
/// ```
pub fn valid_int(x: f64, cast: IntCast) -> AudioSampleResult<i64> {
    let r = match cast {
        IntCast::Floor => x.floor(),
        IntCast::Ceil => x.ceil(),
        IntCast::Round => x.round(),
        IntCast::Trunc => x.trunc(),
    };
    if !r.is_finite() || r.abs() >= 9.2e18 {
        return invalid("x", format!("{x} does not fit a 64-bit integer"));
    }
    Ok(r as i64)
}

/// What [`find_files`] looks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FindFilesConfig {
    /// File extensions without the dot.
    pub extensions: Vec<String>,
    /// Search subdirectories.
    pub recurse: bool,
    /// Compare extensions exactly; otherwise any letter case matches.
    pub case_sensitive: bool,
    /// Keep at most this many paths after the offset.
    pub limit: Option<usize>,
    /// Skip this many paths.
    pub offset: usize,
}

impl FindFilesConfig {
    /// Common audio extensions, searching subdirectories, ignoring case.
    #[must_use]
    pub fn new() -> Self {
        Self {
            extensions: ["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"]
                .map(String::from)
                .to_vec(),
            recurse: true,
            case_sensitive: false,
            limit: None,
            offset: 0,
        }
    }

    /// librosa's defaults; currently equal to [`new`](Self::new).
    #[must_use]
    pub fn librosa() -> Self {
        Self::new()
    }
}

impl Default for FindFilesConfig {
    fn default() -> Self {
        Self::new()
    }
}

fn normalise_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Lists files with given extensions under a directory.
///
/// A leading `~` expands to the home directory. Hidden files (a name beginning with a
/// dot) and directories are left out. Symbolic links to directories are not followed.
///
/// # Arguments
/// - `directory`: where to look.
/// - `config`: extensions, recursion, case handling and the slice of results to keep.
///
/// # Returns
/// Sorted absolute paths. Unreadable or missing directories are skipped.
///
/// # Errors
/// [`ParameterError`] if the current directory cannot be read to make the path
/// absolute.
///
/// # Examples
/// ```no_run
/// use audio_samples::utils::array_ops::{FindFilesConfig, find_files};
///
/// let wavs = find_files("~/Music", &FindFilesConfig::new()).unwrap();
/// println!("{} files", wavs.len());
/// ```
pub fn find_files(
    directory: impl AsRef<Path>,
    config: &FindFilesConfig,
) -> AudioSampleResult<Vec<PathBuf>> {
    let mut out = Vec::new();
    find_files_each(directory, config, |p| out.push(p.to_path_buf()))?;
    Ok(out)
}

/// One file found by [`find_files_each`]: its directory and a range of the name buffer.
struct FoundFile {
    directory: u32,
    start: u32,
    end: u32,
}

/// [`find_files`] calling `visit` with each path in sorted order, keeping only the
/// directory names and file names in memory.
///
/// # Arguments
/// - `directory`, `config`: as for [`find_files`].
/// - `visit`: called once per path.
///
/// # Errors
/// As for [`find_files`].
///
/// # Examples
/// ```no_run
/// use audio_samples::utils::array_ops::{FindFilesConfig, find_files_each};
///
/// let mut n = 0;
/// find_files_each("~/Music", &FindFilesConfig::new(), |_| n += 1).unwrap();
/// println!("{n} files");
/// ```
pub fn find_files_each(
    directory: impl AsRef<Path>,
    config: &FindFilesConfig,
    mut visit: impl FnMut(&Path),
) -> AudioSampleResult<()> {
    use std::ffi::OsStr;
    let directory = directory.as_ref();
    let expanded = match directory.strip_prefix("~") {
        Ok(rest) => std::env::var_os("HOME")
            .map_or_else(|| directory.to_owned(), |h| Path::new(&h).join(rest)),
        Err(_) => directory.to_owned(),
    };
    let root = std::path::absolute(&expanded)
        .map_err(|e| ParameterError::invalid_value("directory", e.to_string()))?;
    let root = normalise_path(&root);

    let suffixes: Vec<Vec<u8>> = config
        .extensions
        .iter()
        .map(|e| {
            let mut suffix = b".".to_vec();
            suffix.extend_from_slice(e.as_bytes());
            suffix
        })
        .collect();
    let case_sensitive = config.case_sensitive;
    let matches = |name: &[u8]| -> bool {
        if name.first() == Some(&b'.') {
            return false;
        }
        suffixes.iter().any(|suffix| {
            name.len() >= suffix.len() && {
                let tail = &name[name.len() - suffix.len()..];
                if case_sensitive {
                    tail == &suffix[..]
                } else {
                    tail.eq_ignore_ascii_case(suffix)
                }
            }
        })
    };

    let mut directories = vec![root];
    let mut names: Vec<u8> = Vec::new();
    let mut files: Vec<FoundFile> = Vec::new();
    let mut next = 0;
    while next < directories.len() {
        let current = next;
        next += 1;
        let Ok(entries) = std::fs::read_dir(&directories[current]) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_dir() {
                if config.recurse {
                    directories.push(entry.path());
                }
                continue;
            }
            let name = entry.file_name();
            let bytes = name.as_encoded_bytes();
            if matches(bytes) && !(kind.is_symlink() && entry.path().is_dir()) {
                let start = names.len() as u32;
                names.extend_from_slice(bytes);
                files.push(FoundFile {
                    directory: current as u32,
                    start,
                    end: names.len() as u32,
                });
            }
        }
    }
    let sep = std::path::MAIN_SEPARATOR as u8;
    let parts = |f: &FoundFile| {
        directories[f.directory as usize]
            .as_os_str()
            .as_encoded_bytes()
            .iter()
            .copied()
            .chain(std::iter::once(sep))
            .chain(names[f.start as usize..f.end as usize].iter().copied())
    };
    files.sort_unstable_by(|a, b| parts(a).cmp(parts(b)));
    let limit = config.limit.unwrap_or(usize::MAX);
    let mut path = PathBuf::new();
    for f in files.iter().skip(config.offset).take(limit) {
        path.clear();
        path.push(&directories[f.directory as usize]);
        let bytes = &names[f.start as usize..f.end as usize];
        // SAFETY: the bytes were copied unchanged from an `OsStr` read from the directory.
        path.push(unsafe { OsStr::from_encoded_bytes_unchecked(bytes) });
        visit(&path);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn local_extrema_on_short_input_are_false() {
        assert_eq!(localmax(&array![1.0], Axis(0)).unwrap(), array![false]);
        assert_eq!(localmin(&array![1.0], Axis(0)).unwrap(), array![false]);
        assert!(localmax(&array![1.0], Axis(1)).is_err());
    }

    #[test]
    fn matching_respects_direction() {
        assert_eq!(
            match_events(&[4.0], &[0.0, 5.0], false, true).unwrap(),
            vec![1]
        );
        assert_eq!(
            match_events(&[4.0], &[0.0, 5.0], true, false).unwrap(),
            vec![0]
        );
        assert_eq!(
            match_events(&[2.5], &[0.0, 5.0], true, true).unwrap(),
            vec![1]
        );
        assert!(match_events(&[6.0], &[0.0, 5.0], false, true).is_err());
        assert!(match_events(&[-1.0], &[0.0, 5.0], true, false).is_err());
        assert!(match_events(&[1.0], &[0.0, 5.0], false, false).is_err());
        assert_eq!(
            match_events(&[5.0], &[0.0, 5.0], false, false).unwrap(),
            vec![1]
        );
        assert!(match_events(&[], &[0.0], true, true).is_err());
    }

    #[test]
    fn non_strict_intervals_take_the_true_nearest() {
        let to = array![[0.0, 1.0], [5.0, 6.0], [20.0, 30.0]];
        let m = match_intervals(array![[3.5, 4.0]].view(), to.view(), false).unwrap();
        assert_eq!(m, vec![1]);
        let m = match_intervals(array![[1.5, 2.0]].view(), to.view(), false).unwrap();
        assert_eq!(m, vec![0]);
        let m = match_intervals(array![[40.0, 50.0]].view(), to.view(), false).unwrap();
        assert_eq!(m, vec![2]);
        assert!(match_intervals(array![[3.5, 4.0]].view(), to.view(), true).is_err());
    }

    #[test]
    fn pad_modes_extend_over_several_periods() {
        let x = array![1, 2];
        let e = pad_center(&x, 6, Axis(0), PadMode::Edge).unwrap();
        assert_eq!(e, array![1, 1, 1, 2, 2, 2]);
        let w = pad_center(&x, 6, Axis(0), PadMode::Wrap).unwrap();
        assert_eq!(w, array![1, 2, 1, 2, 1, 2]);
        let s = pad_center(&x, 6, Axis(0), PadMode::Symmetric).unwrap();
        assert_eq!(s, array![2, 1, 1, 2, 2, 1]);
        let r = pad_center(&x, 6, Axis(0), PadMode::Reflect).unwrap();
        assert_eq!(r, array![1, 2, 1, 2, 1, 2]);
        assert!(pad_center(&x, 1, Axis(0), PadMode::Edge).is_err());
    }

    #[test]
    fn expand_to_reorders_out_of_order_positions() {
        let x = ndarray::Array2::<f64>::zeros((2, 3));
        let y = expand_to(x.view().into_dyn(), 3, &[2, 0]).unwrap();
        assert_eq!(y.shape(), &[3, 1, 2]);
        assert!(expand_to(x.view().into_dyn(), 1, &[0, 1]).is_err());
        assert!(expand_to(x.view().into_dyn(), 3, &[1, 1]).is_err());
        assert!(expand_to(x.view().into_dyn(), 3, &[0]).is_err());
    }

    #[test]
    fn interpolation_kinds_on_known_values() {
        let y = array![0.0_f64, 1.0, 4.0, 9.0, 16.0].into_dyn();
        let x = [0.0, 1.0, 2.0, 3.0, 4.0];
        let xi = array![0.5, 3.5, 5.0, f64::NAN].into_dyn();
        let run = |k| interp1d(&x, y.view(), xi.view(), k, -1.0, Axis(0)).unwrap();
        // A quadratic and a cubic spline reproduce x^2 exactly.
        for k in [InterpKind::Quadratic, InterpKind::Cubic] {
            let v = run(k);
            assert!((v[[0]] - 0.25).abs() < 1e-12);
            assert!((v[[1]] - 12.25).abs() < 1e-12);
            assert_eq!(v[[2]], -1.0);
            assert!(v[[3]].is_nan());
        }
        assert_eq!(run(InterpKind::Linear)[[0]], 0.5);
        assert_eq!(run(InterpKind::Previous)[[0]], 0.0);
        assert_eq!(run(InterpKind::Next)[[0]], 1.0);
        assert_eq!(run(InterpKind::Nearest)[[0]], 0.0);
        assert_eq!(run(InterpKind::NearestUp)[[0]], 1.0);
        assert_eq!(run(InterpKind::Zero)[[1]], 9.0);
        assert!(
            interp1d(
                &[0.0, 1.0],
                y.view(),
                xi.view(),
                InterpKind::Linear,
                0.0,
                Axis(0)
            )
            .is_err()
        );
    }

    #[test]
    fn unsorted_positions_are_sorted_with_the_data() {
        let y = array![4.0, 0.0, 2.0].into_dyn();
        let xi = array![0.5].into_dyn();
        let v = interp1d(
            &[2.0, 0.0, 1.0],
            y.view(),
            xi.view(),
            InterpKind::Linear,
            0.0,
            Axis(0),
        )
        .unwrap();
        assert_eq!(v[[0]], 1.0);
    }

    #[test]
    fn buffers_decode_every_width() {
        let a: Vec<f32> = buf_to_float(&[0x80], 1).unwrap();
        assert_eq!(a, vec![-1.0]);
        let b: Vec<f64> = buf_to_float(&[0, 0, 0x80], 3).unwrap();
        assert_eq!(b, vec![-1.0]);
        let c: Vec<f64> = buf_to_float(&[0, 0, 0, 0x80], 4).unwrap();
        assert_eq!(c, vec![-1.0]);
        let d: Vec<f64> = buf_to_float(&[0, 0, 0, 0, 0, 0, 0, 0x80], 8).unwrap();
        assert_eq!(d, vec![-1.0]);
        assert!(buf_to_float::<f32>(&[0, 0, 0], 2).is_err());
        assert!(buf_to_float::<f32>(&[0], 5).is_err());
    }

    #[test]
    fn sparse_rows_round_trip() {
        let m = sparsify_rows(array![[0.0_f64, 0.0], [1.0, -2.0]].view(), 0.01).unwrap();
        assert_eq!(m.to_dense(), array![[0.0, 0.0], [1.0, -2.0]]);
        assert!(sparsify_rows(array![[1.0_f64]].view(), 1.0).is_err());
    }

    #[test]
    fn unique_counts_treat_nans_as_one_value() {
        let c = count_unique(&array![[f64::NAN, f64::NAN, 1.0]], Axis(1)).unwrap();
        assert_eq!(c, array![2]);
        let u = is_unique(&array![[f64::NAN, f64::NAN, 1.0]], Axis(1)).unwrap();
        assert_eq!(u, array![false]);
    }

    #[test]
    fn find_files_filters_and_slices() {
        let dir = std::env::temp_dir().join(format!("aus-find-{}", std::process::id()));
        let sub = dir.join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        for f in ["a.wav", "b.WAV", "c.txt", ".hidden.wav"] {
            std::fs::write(dir.join(f), b"").unwrap();
        }
        std::fs::write(sub.join("d.flac"), b"").unwrap();
        let cfg = FindFilesConfig::new();
        let names = |v: Vec<PathBuf>| -> Vec<String> {
            v.iter()
                .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
                .collect()
        };
        assert_eq!(
            names(find_files(&dir, &cfg).unwrap()),
            ["a.wav", "b.WAV", "d.flac"]
        );
        let flat = FindFilesConfig {
            recurse: false,
            ..cfg.clone()
        };
        assert_eq!(names(find_files(&dir, &flat).unwrap()), ["a.wav", "b.WAV"]);
        let exact = FindFilesConfig {
            case_sensitive: true,
            extensions: vec!["wav".into()],
            ..cfg
        };
        assert_eq!(names(find_files(&dir, &exact).unwrap()), ["a.wav"]);
        let sliced = FindFilesConfig {
            limit: Some(1),
            offset: 1,
            ..cfg
        };
        assert_eq!(names(find_files(&dir, &sliced).unwrap()), ["b.WAV"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
