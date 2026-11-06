//! Complex FFTs of `n = r · m` points as one radix-`r` pass and `r`
//! transforms of `m` points, with the spectrum in a permuted order.
//!
//! Forward (decimation in frequency): with `x[n2 + m n1]` (`n1 < r`,
//! `n2 < m`), row `k1` of the result is the `m`-point transform of
//! `u[k1][n2] = e^(-2πi n2 k1 / n) Σ_n1 x[n2 + m n1] e^(-2πi n1 k1 / r)`,
//! so position `k2` of row `k1` holds bin `k1 + r k2`. The inverse undoes
//! exactly that: `m`-point inverse transforms of the rows, then the twiddles
//! and the radix-`r` pass, giving the signal in natural order.
//!
//! Products of spectra do not depend on the order of the bins, so the
//! resampler keeps its filter spectra in this order and never sorts the
//! bins. Rows are `stride` elements apart (`m` plus a little padding, which
//! stays zero), so that the radix pass's `r` loads and stores do not all
//! fall into the same cache sets when `m` is a multiple of 256.
//!
//! The `m`-point transforms are small enough to stay in the L1 cache with
//! their scratch, which the single `n`-point transform is not: on the
//! efficiency cores of a hybrid x86-64 CPU, `f64` transforms of 1024 to 4096
//! points take 10-20 % less time this way.

use super::RsFloat;
use realfft::num_complex::Complex;
use rustfft::Fft;
use std::sync::Arc;

/// Storage order of a spectrum: `r` rows of `m` bins, `stride` apart;
/// position `k2` of row `k1` holds bin `k1 + r k2`. The natural order is
/// one row (`r = 1`, `stride = m`).
///
/// For two spectra of `n` and `f n` bins in layouts of the same `r`, bin
/// `k mod n` of the shorter is in the same row as bin `k` of the longer, at
/// position `k2 mod (n / r)`: a spectrum repeated `f` times is each row
/// repeated `f` times, and the `f` aliases of a spectrum summed are each
/// row's `f` segments summed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::resampling) struct Layout {
    pub r: usize,
    pub m: usize,
    pub stride: usize,
}

impl Layout {
    /// One row of `n` bins in natural order.
    pub(super) const fn natural(n: usize) -> Self {
        Self {
            r: 1,
            m: n,
            stride: n,
        }
    }

    /// Elements a spectrum occupies (padding included).
    pub(super) const fn len(&self) -> usize {
        self.r * self.stride
    }

    /// Number of bins.
    pub(super) const fn bins(&self) -> usize {
        self.r * self.m
    }

    /// Storage position of bin `k`.
    pub(super) const fn pos(&self, k: usize) -> usize {
        (k % self.r) * self.stride + k / self.r
    }

    /// Converts a spectrum in natural order into this layout (padding zero).
    pub(super) fn arrange<T: Copy>(&self, natural: &[T], zero: T) -> Vec<T> {
        debug_assert_eq!(natural.len(), self.bins());
        let mut out = vec![zero; self.len()];
        for (k, &v) in natural.iter().enumerate() {
            out[self.pos(k)] = v;
        }
        out
    }
}

/// Padding (elements) after each row of a permuted spectrum.
const PAD: usize = 4;

/// A transform of `r · m` points in [`Layout`] order (see the module
/// documentation).
pub(super) struct FourStep<F> {
    layout: Layout,
    /// `e^(-2πi n2 k1 / n)` for `k1 >= 1` at [`tw_pos`]`(r, k1, n2)`.
    twiddles: Vec<Complex<F>>,
    fwd: Arc<dyn Fft<F>>,
    inv: Arc<dyn Fft<F>>,
}

impl<F: RsFloat> FourStep<F> {
    /// The transform of `n` points, if `n` is one this form serves:
    /// powers of two from 512 to 4096 (radix 4 for 512, 8 above).
    pub(super) fn new(n: usize) -> Option<Self> {
        let r = match n {
            512 => 4,
            1024 | 2048 | 4096 => 8,
            _ => return None,
        };
        let m = n / r;
        let mut twiddles = vec![Complex::new(F::zero(), F::zero()); (r - 1) * m];
        for k1 in 1..r {
            for n2 in 0..m {
                let a = -std::f64::consts::TAU * ((n2 * k1) % n) as f64 / n as f64;
                twiddles[tw_pos(r, k1, n2)] = Complex::new(F::of(a.cos()), F::of(a.sin()));
            }
        }
        let (fwd, inv) = F::with_planner(|p| (p.plan_fft_forward(m), p.plan_fft_inverse(m)));
        Some(Self {
            layout: Layout {
                r,
                m,
                stride: m + PAD,
            },
            twiddles,
            fwd,
            inv,
        })
    }

    pub(super) fn layout(&self) -> Layout {
        self.layout
    }

    /// Scratch elements [`FourStep::forward`] and [`FourStep::inverse`] need.
    pub(super) fn scratch_len(&self) -> usize {
        self.fwd
            .get_inplace_scratch_len()
            .max(self.inv.get_inplace_scratch_len())
    }

    /// Forward transform of `src` (natural order, `n` points) into `dst`
    /// (`layout().len()`, permuted). `simd`: AVX2 + FMA code (only when the
    /// CPU has them).
    pub(super) fn forward(
        &self,
        src: &[Complex<F>],
        dst: &mut [Complex<F>],
        scratch: &mut [Complex<F>],
        simd: bool,
    ) {
        let Layout { r, m, stride } = self.layout;
        F::radix_pass(r, false, simd, src, dst, &self.twiddles, m, m, stride);
        for row in dst.chunks_exact_mut(stride) {
            self.fwd.process_with_scratch(&mut row[..m], scratch);
        }
    }

    /// Inverse transform (unscaled) of `buf` (permuted, overwritten) into
    /// `dst`: natural order (`n` points), or with `padded` the `r` blocks of
    /// `m` samples `stride` apart (`layout().len()` elements), which keeps
    /// the radix pass's stores out of each other's cache sets.
    pub(super) fn inverse(
        &self,
        buf: &mut [Complex<F>],
        dst: &mut [Complex<F>],
        scratch: &mut [Complex<F>],
        simd: bool,
        padded: bool,
    ) {
        let Layout { r, m, stride } = self.layout;
        for row in buf.chunks_exact_mut(stride) {
            self.inv.process_with_scratch(&mut row[..m], scratch);
        }
        let ds = if padded { stride } else { m };
        F::radix_pass(r, true, simd, buf, dst, &self.twiddles, m, stride, ds);
    }
}

/// Position of the twiddle of row `k1 >= 1` at `n2` for radix `r`: the
/// `r - 1` twiddles of `n2` and `n2 + 1` (`n2` even) are read together,
/// as `r - 1` consecutive pairs.
#[inline(always)]
const fn tw_pos(r: usize, k1: usize, n2: usize) -> usize {
    (n2 / 2) * 2 * (r - 1) + (k1 - 1) * 2 + n2 % 2
}

/// The radix-`r` pass of a [`FourStep`] transform, portable. Forward
/// (`INV = false`): for each `n2 < m`, the `r`-point DFT of
/// `src[n2 + ss · n1]` times the twiddles, into `dst[n2 + ds · k1]`.
/// Inverse: twiddles conjugated and applied first, then the inverse DFT.
#[allow(clippy::too_many_arguments)]
pub(super) fn radix_pass_portable<F: RsFloat, const R: usize, const INV: bool>(
    src: &[Complex<F>],
    dst: &mut [Complex<F>],
    tw: &[Complex<F>],
    m: usize,
    ss: usize,
    ds: usize,
) {
    // e^(∓2πi j / R)
    let root: [Complex<F>; R] = std::array::from_fn(|j| {
        let a = std::f64::consts::TAU * j as f64 / R as f64;
        let s = if INV { a.sin() } else { -a.sin() };
        Complex::new(F::of(a.cos()), F::of(s))
    });
    for n2 in 0..m {
        let a: [Complex<F>; R] = std::array::from_fn(|j| {
            let v = src[n2 + ss * j];
            if INV && j > 0 {
                v * tw[tw_pos(R, j, n2)].conj()
            } else {
                v
            }
        });
        for k in 0..R {
            let mut acc = Complex::new(F::zero(), F::zero());
            for (j, &v) in a.iter().enumerate() {
                acc += v * root[(j * k) % R];
            }
            dst[n2 + ds * k] = if INV || k == 0 {
                acc
            } else {
                acc * tw[tw_pos(R, k, n2)]
            };
        }
    }
}

/// AVX2 + FMA radix passes for `f64` (two complex values per vector).
#[cfg(target_arch = "x86_64")]
pub(super) mod simd {
    use realfft::num_complex::Complex;
    use std::arch::x86_64::*;

    /// `a · b` for two complex values per vector.
    #[inline(always)]
    unsafe fn mul(a: __m256d, b: __m256d) -> __m256d {
        // SAFETY: arithmetic only (the caller has AVX2 and FMA).
        unsafe {
            let br = _mm256_movedup_pd(b);
            let bi = _mm256_permute_pd::<0b1111>(b);
            let sw = _mm256_permute_pd::<0b0101>(a);
            _mm256_fmaddsub_pd(a, br, _mm256_mul_pd(sw, bi))
        }
    }

    /// `a · conj(b)`.
    #[inline(always)]
    unsafe fn mul_conj(a: __m256d, b: __m256d) -> __m256d {
        // SAFETY: arithmetic only (the caller has AVX2 and FMA).
        unsafe {
            let br = _mm256_movedup_pd(b);
            let bi = _mm256_permute_pd::<0b1111>(b);
            let sw = _mm256_permute_pd::<0b0101>(a);
            _mm256_fmsubadd_pd(a, br, _mm256_mul_pd(sw, bi))
        }
    }

    /// `a · (-i)` (forward) or `a · i` (inverse).
    #[inline(always)]
    unsafe fn rot<const INV: bool>(a: __m256d) -> __m256d {
        // SAFETY: arithmetic only (the caller has AVX2 and FMA).
        unsafe {
            let sw = _mm256_permute_pd::<0b0101>(a);
            if INV {
                _mm256_xor_pd(sw, _mm256_set_pd(0.0, -0.0, 0.0, -0.0))
            } else {
                _mm256_xor_pd(sw, _mm256_set_pd(-0.0, 0.0, -0.0, 0.0))
            }
        }
    }

    #[inline(always)]
    unsafe fn dft2(a: [__m256d; 2]) -> [__m256d; 2] {
        // SAFETY: arithmetic only (the caller has AVX2 and FMA).
        unsafe { [_mm256_add_pd(a[0], a[1]), _mm256_sub_pd(a[0], a[1])] }
    }

    #[inline(always)]
    unsafe fn dft4<const INV: bool>(a: [__m256d; 4]) -> [__m256d; 4] {
        // SAFETY: arithmetic only (the caller has AVX2).
        unsafe {
            let s02 = _mm256_add_pd(a[0], a[2]);
            let d02 = _mm256_sub_pd(a[0], a[2]);
            let s13 = _mm256_add_pd(a[1], a[3]);
            let d13 = rot::<INV>(_mm256_sub_pd(a[1], a[3]));
            [
                _mm256_add_pd(s02, s13),
                _mm256_add_pd(d02, d13),
                _mm256_sub_pd(s02, s13),
                _mm256_sub_pd(d02, d13),
            ]
        }
    }

    #[inline(always)]
    unsafe fn dft8<const INV: bool>(a: [__m256d; 8]) -> [__m256d; 8] {
        // SAFETY: arithmetic only (the caller has AVX2).
        unsafe {
            let h = _mm256_set1_pd(std::f64::consts::FRAC_1_SQRT_2);
            let b = [0, 1, 2, 3].map(|j| _mm256_add_pd(a[j], a[j + 4]));
            let c = [0, 1, 2, 3].map(|j| _mm256_sub_pd(a[j], a[j + 4]));
            // c[j] · e^(∓2πi j / 8): (1 ∓ i) / √2, ∓i, ∓i (1 ∓ i) / √2.
            let c1 = _mm256_mul_pd(_mm256_add_pd(c[1], rot::<INV>(c[1])), h);
            let c2 = rot::<INV>(c[2]);
            let c3 = rot::<INV>(_mm256_mul_pd(_mm256_add_pd(c[3], rot::<INV>(c[3])), h));
            let e = dft4::<INV>(b);
            let o = dft4::<INV>([c[0], c1, c2, c3]);
            [e[0], o[0], e[1], o[1], e[2], o[2], e[3], o[3]]
        }
    }

    #[inline(always)]
    unsafe fn dft<const R: usize, const INV: bool>(a: [__m256d; R]) -> [__m256d; R] {
        // SAFETY: arithmetic only (the caller has AVX2).
        unsafe {
            let mut y = a;
            match R {
                2 => y.copy_from_slice(&dft2([a[0], a[1]])),
                4 => y.copy_from_slice(&dft4::<INV>([a[0], a[1], a[2], a[3]])),
                8 => y.copy_from_slice(&dft8::<INV>(std::array::from_fn(|i| a[i]))),
                _ => unreachable!("radix 2, 4 or 8"),
            }
            y
        }
    }

    /// [`super::radix_pass_portable`] for `f64`, two values of `n2` at a time.
    ///
    /// # Safety
    /// The CPU must support AVX2 and FMA. `m` is even, `src` holds
    /// `ss · (R - 1) + m` elements, `dst` `ds · (R - 1) + m`, `tw` `(R - 1) · m`.
    #[target_feature(enable = "avx2,fma")]
    pub(in crate::resampling) unsafe fn radix_pass_f64<const R: usize, const INV: bool>(
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
        tw: &[Complex<f64>],
        m: usize,
        ss: usize,
        ds: usize,
    ) {
        assert!(m.is_multiple_of(2) && tw.len() >= (R - 1) * m);
        assert!(src.len() >= ss * (R - 1) + m && dst.len() >= ds * (R - 1) + m);
        let (ps, pd, pt) = (
            src.as_ptr().cast::<f64>(),
            dst.as_mut_ptr().cast::<f64>(),
            tw.as_ptr().cast::<f64>(),
        );
        let mut n2 = 0;
        while n2 < m {
            // SAFETY: rows j < R at n2, n2 + 1 < m are in bounds (asserted).
            unsafe {
                let load = |j: usize| _mm256_loadu_pd(ps.add(2 * (n2 + ss * j)));
                let twid = |k: usize| _mm256_loadu_pd(pt.add(2 * super::tw_pos(R, k, n2)));
                if INV {
                    let a: [__m256d; R] = std::array::from_fn(|k| {
                        if k == 0 {
                            load(0)
                        } else {
                            mul_conj(load(k), twid(k))
                        }
                    });
                    let y = dft::<R, true>(a);
                    for (j, v) in y.into_iter().enumerate() {
                        _mm256_storeu_pd(pd.add(2 * (n2 + ds * j)), v);
                    }
                } else {
                    let y = dft::<R, false>(std::array::from_fn(load));
                    for (k, v) in y.into_iter().enumerate() {
                        let v = if k == 0 { v } else { mul(v, twid(k)) };
                        _mm256_storeu_pd(pd.add(2 * (n2 + ds * k)), v);
                    }
                }
            }
            n2 += 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check<F: RsFloat>(n: usize, simd: bool, tol: f64) {
        let fs = FourStep::<F>::new(n).unwrap();
        let lay = fs.layout();
        let x: Vec<Complex<F>> = (0..n)
            .map(|i| {
                Complex::new(
                    F::of((i as f64 * 0.37).sin()),
                    F::of((i as f64 * 0.11).cos()),
                )
            })
            .collect();
        let full = F::with_planner(|p| p.plan_fft_forward(n));
        let mut want = x.clone();
        full.process(&mut want);
        let zero = Complex::new(F::zero(), F::zero());
        let mut got = vec![zero; lay.len()];
        let mut scratch = vec![zero; fs.scratch_len()];
        fs.forward(&x, &mut got, &mut scratch, simd);
        for (k, w) in want.iter().enumerate() {
            let d = got[lay.pos(k)] - w;
            assert!(
                d.re.as_f64().hypot(d.im.as_f64()) < tol * n as f64,
                "{n} bin {k}"
            );
        }
        let mut back = vec![zero; n];
        fs.inverse(&mut got.clone(), &mut back, &mut scratch, simd, false);
        let mut rows = vec![zero; lay.len()];
        fs.inverse(&mut got, &mut rows, &mut scratch, simd, true);
        for (i, b) in back.iter().enumerate() {
            assert_eq!(rows[i / lay.m * lay.stride + i % lay.m], *b);
        }
        for (b, v) in back.iter().zip(&x) {
            let d = *b - *v * F::of(n as f64);
            assert!(d.re.as_f64().hypot(d.im.as_f64()) < tol * n as f64);
        }
    }

    /// Forward bins match a single transform's, and the inverse gives the
    /// input back, for every size, portable and AVX2.
    #[test]
    fn four_step_matches_single_transform() {
        let simd = [false, super::super::super::has_avx2_fma_cached()];
        for n in [512, 1024, 2048, 4096] {
            for s in simd {
                check::<f64>(n, s, 1e-14);
                check::<f32>(n, false, 1e-5);
            }
        }
    }
}
