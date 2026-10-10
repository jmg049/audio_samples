//! Memory-order traversal of 2-D `(bins, frames)` arrays.
//!
//! Spectrogram kernels reduce over the bin axis independently for every
//! frame. They are written against tiles of consecutive frames whose rows (or
//! columns) are contiguous slices, so that the inner loops are unit-stride for
//! either memory order: a C-order input is passed through unchanged, and any
//! other layout (for example the Fortran-order arrays that librosa's `stft`
//! returns) is copied one tile at a time into a small buffer. The tiles keep
//! the logical element order, so a kernel computes the same values, in the
//! same order of operations, for every layout.

use ndarray::{ArrayView2, ShapeBuilder};

/// Target size in bytes of one copied tile (inside L2).
const TILE_BYTES: usize = 256 * 1024;

/// Number of frames per copied tile for `rows` rows of `T`.
fn tile_width<T>(rows: usize, cols: usize) -> usize {
    let w = TILE_BYTES / (rows.max(1) * size_of::<T>().max(1));
    // A multiple of 8 frames (a full vector of f32 lanes), at least 32 so
    // that the row-wise loops of the kernels are long enough to vectorise.
    (w / 8 * 8).max(32).min(cols)
}

/// Calls `f(t0, tile)` for consecutive tiles of columns of `x`, where `tile`
/// is the `(rows, w)` block of columns `t0..t0 + w` and each of its rows is a
/// contiguous slice (see [`row`]).
///
/// When the rows of `x` are already contiguous, `f` is called once with `x`
/// itself; otherwise the tiles are copied.
pub(crate) fn for_each_row_tile<T: Copy>(
    x: ArrayView2<'_, T>,
    mut f: impl FnMut(usize, ArrayView2<'_, T>),
) {
    let (rows, cols) = x.dim();
    if rows == 0 || cols == 0 {
        return;
    }
    if cols == 1 || x.strides()[1] == 1 {
        f(0, x);
        return;
    }
    let w = tile_width::<T>(rows, cols);
    let mut buf: Vec<T> = Vec::with_capacity(rows * w);
    let mut t0 = 0;
    while t0 < cols {
        let wt = w.min(cols - t0);
        buf.clear();
        buf.resize(rows * wt, x[[0, 0]]);
        transpose_into(
            &x.slice(ndarray::s![.., t0..t0 + wt]).reversed_axes(),
            &mut buf,
            wt,
        );
        f(
            t0,
            ArrayView2::from_shape((rows, wt), &buf).expect("buffer holds rows x wt values"),
        );
        t0 += wt;
    }
}

/// Calls `f(t0, tile)` for consecutive tiles of columns of `x`, where `tile`
/// is the `(rows, w)` block of columns `t0..t0 + w` and each of its columns
/// is a contiguous slice (see [`col`]).
///
/// When the columns of `x` are already contiguous, `f` is called once with
/// `x` itself; otherwise the tiles are copied.
pub(crate) fn for_each_col_tile<T: Copy>(
    x: ArrayView2<'_, T>,
    mut f: impl FnMut(usize, ArrayView2<'_, T>),
) {
    let (rows, cols) = x.dim();
    if rows == 0 || cols == 0 {
        return;
    }
    if rows == 1 || x.strides()[0] == 1 {
        f(0, x);
        return;
    }
    let w = tile_width::<T>(rows, cols);
    let mut buf: Vec<T> = Vec::with_capacity(rows * w);
    let mut t0 = 0;
    while t0 < cols {
        let wt = w.min(cols - t0);
        buf.clear();
        buf.resize(rows * wt, x[[0, 0]]);
        transpose_into(&x.slice(ndarray::s![.., t0..t0 + wt]), &mut buf, rows);
        f(
            t0,
            ArrayView2::from_shape((rows, wt).f(), &buf).expect("buffer holds rows x wt values"),
        );
        t0 += wt;
    }
}

/// Writes `src[[i, j]]` to `dst[j * stride + i]` (`dst` holds `src.ncols()`
/// rows of `stride >= src.nrows()` values), reading `src` along its rows when
/// they are contiguous. Works in blocks of 8 source rows so that the writes
/// are runs of 8 consecutive values.
fn transpose_into<T: Copy>(src: &ArrayView2<'_, T>, dst: &mut [T], stride: usize) {
    const B: usize = 8;
    let (n_i, n_j) = src.dim();
    let mut i0 = 0;
    while i0 < n_i {
        let bi = B.min(n_i - i0);
        let rows: [Option<&[T]>; B] = std::array::from_fn(|q| {
            if q < bi {
                src.row(i0 + q).to_slice()
            } else {
                None
            }
        });
        if bi == B && rows.iter().all(Option::is_some) {
            let r: [&[T]; B] = std::array::from_fn(|q| rows[q].expect("checked"));
            for j in 0..n_j {
                let d = &mut dst[j * stride + i0..j * stride + i0 + B];
                for q in 0..B {
                    d[q] = r[q][j];
                }
            }
        } else {
            for (q, i) in (i0..i0 + bi).enumerate() {
                for (j, &v) in src.row(i).iter().enumerate() {
                    dst[j * stride + i0 + q] = v;
                }
            }
        }
        i0 += bi;
    }
}

/// Row `k` of a tile from [`for_each_row_tile`] as a slice.
pub(crate) fn row<'a, T>(tile: &ArrayView2<'a, T>, k: usize) -> &'a [T] {
    let (rows, cols) = tile.dim();
    assert!(k < rows);
    if cols == 0 {
        return &[];
    }
    // SAFETY: the rows of a tile are contiguous (unit column stride, or a
    // single column), so row `k` is `cols` consecutive elements starting at
    // the address of element `[k, 0]`, which lives as long as the view.
    unsafe { std::slice::from_raw_parts(tile.uget((k, 0)) as *const T, cols) }
}

/// Column `t` of a tile from [`for_each_col_tile`] as a slice.
pub(crate) fn col<'a, T>(tile: &ArrayView2<'a, T>, t: usize) -> &'a [T] {
    let (rows, cols) = tile.dim();
    assert!(t < cols);
    if rows == 0 {
        return &[];
    }
    // SAFETY: as for `row`, with unit row stride (or a single row).
    unsafe { std::slice::from_raw_parts(tile.uget((0, t)) as *const T, rows) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, ShapeBuilder};

    fn check_row_tiles(x: ArrayView2<'_, f64>) {
        let mut seen = Array2::<f64>::zeros(x.dim());
        for_each_row_tile(x, |t0, tile| {
            for k in 0..tile.nrows() {
                for (j, &v) in row(&tile, k).iter().enumerate() {
                    seen[[k, t0 + j]] = v;
                }
            }
        });
        assert_eq!(seen, x);
    }

    fn check_col_tiles(x: ArrayView2<'_, f64>) {
        let mut seen = Array2::<f64>::zeros(x.dim());
        for_each_col_tile(x, |t0, tile| {
            for j in 0..tile.ncols() {
                for (k, &v) in col(&tile, j).iter().enumerate() {
                    seen[[k, t0 + j]] = v;
                }
            }
        });
        assert_eq!(seen, x);
    }

    #[test]
    fn tiles_cover_every_element_in_any_layout() {
        for &(r, c) in &[(1, 1), (1, 50), (50, 1), (3, 7), (1025, 431), (20, 9000)] {
            let c_order = Array2::from_shape_fn((r, c), |(i, j)| (i * 10_000 + j) as f64);
            let f_order = Array2::from_shape_fn((r, c).f(), |(i, j)| (i * 10_000 + j) as f64);
            check_row_tiles(c_order.view());
            check_row_tiles(f_order.view());
            check_col_tiles(c_order.view());
            check_col_tiles(f_order.view());
            let big = Array2::from_shape_fn((2 * r, 2 * c), |(i, j)| (i * 10_000 + j) as f64);
            let strided = big.slice(ndarray::s![..;2, ..;2]);
            check_row_tiles(strided);
            check_col_tiles(strided);
            let rev = c_order.slice(ndarray::s![..;-1, ..;-1]);
            check_row_tiles(rev);
            check_col_tiles(rev);
        }
    }
}
