//! Whole-buffer evaluation of a resampling [`Plan`] in the sample type's float
//! type (`f32` or `f64`).
//!
//! Stage A (the long, sharp low-pass) is evaluated by FFT overlap-save; stage
//! B (the short interpolator, if any) by direct dot products over a sliding
//! buffer of stage A's output. Both compute the same sums as the streaming
//! path ([`super::PolyphaseState`]), so the two agree up to floating-point
//! rounding.
//!
//! # FFT stage
//!
//! Stage A is a linear-phase FIR `g` at the convolution rate `up · in_rate`,
//! applied to the zero-stuffed input and then decimated by `down` (one of
//! `up`, `down` is 1). For a block of `hop` convolution-rate outputs starting
//! at `m0`, the input segment around it is transformed, multiplied by the
//! filter spectrum and transformed back:
//!
//! * zero-stuffing by `up` is an `up`-fold repetition of the spectrum of the
//!   segment at the input rate, so the forward FFT has size `n / up`;
//! * decimation by `down` is a `down`-fold aliasing sum of the spectrum, so
//!   the inverse FFT has size `n / down`.
//!
//! The filter and the stuffing / decimation are real, so two consecutive
//! blocks are processed together as the real and imaginary parts of one
//! complex signal: one complex forward and one complex inverse FFT per two
//! blocks, with no real-FFT packing passes.

use super::{Kernel, Plan};
use crate::AudioFloat;
use realfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::cell::RefCell;
use std::mem::MaybeUninit;
use std::sync::Arc;

/// The float types the resampler computes in (`f32`, `f64`).
pub(super) trait RsFloat: AudioFloat + rustfft::FftNum {
    /// Dot product of two slices whose length is a multiple of the SIMD
    /// width (8 `f32`, 4 `f64`). `FMA`: AVX2 + FMA code, only used from
    /// [`run_avx2_fma`].
    fn dot<const FMA: bool>(a: &[Self], b: &[Self]) -> Self;
    /// `(1 - frac) · (a · w) + frac · (b · w)` in one pass over `w`.
    fn interp<const FMA: bool>(a: &[Self], b: &[Self], w: &[Self], frac: Self) -> Self;
    /// Complex products `dst[i] = a[i] · b[i]` (`ACC`: `dst[i] += ...`).
    fn cmul<const ACC: bool>(dst: &mut [Complex<Self>], a: &[Complex<Self>], b: &[Complex<Self>]);
    /// Runs `f` with this thread's complex FFT planner for this type.
    fn with_planner<R>(f: impl FnOnce(&mut FftPlanner<Self>) -> R) -> R;
    /// Runs `f` with this thread's engine cache for this type.
    fn with_engines<R>(f: impl FnOnce(&mut EngineCache<Self>) -> R) -> R;
}

/// Recently used engines, most recent first (`None` for plans evaluated by
/// the direct single-stage path).
pub(super) type EngineCache<F> = Vec<(super::KernelKey, Option<Arc<Engine<F>>>)>;

thread_local! {
    static PLANNER_F32: RefCell<FftPlanner<f32>> = RefCell::new(FftPlanner::new());
    static PLANNER_F64: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
    static ENGINES_F32: RefCell<EngineCache<f32>> = const { RefCell::new(Vec::new()) };
    static ENGINES_F64: RefCell<EngineCache<f64>> = const { RefCell::new(Vec::new()) };
}

macro_rules! rs_float {
    ($t:ty, $dot:ident, $interp:ident, $cmul:ident, $planner:ident, $engines:ident) => {
        impl RsFloat for $t {
            #[inline(always)]
            fn dot<const FMA: bool>(a: &[Self], b: &[Self]) -> Self {
                #[cfg(target_arch = "x86_64")]
                if FMA {
                    // SAFETY: `FMA` code only runs from `run_avx2_fma`, which is
                    // only called on CPUs with AVX2 and FMA.
                    return unsafe { simd::$dot(a, b) };
                }
                dot_portable(a, b)
            }

            #[inline(always)]
            fn interp<const FMA: bool>(a: &[Self], b: &[Self], w: &[Self], frac: Self) -> Self {
                #[cfg(target_arch = "x86_64")]
                if FMA {
                    // SAFETY: as in `dot`.
                    return unsafe { simd::$interp(a, b, w, frac) };
                }
                (1.0 - frac) * dot_portable(a, w) + frac * dot_portable(b, w)
            }

            #[inline(always)]
            fn cmul<const ACC: bool>(
                dst: &mut [Complex<Self>],
                a: &[Complex<Self>],
                b: &[Complex<Self>],
            ) {
                #[cfg(target_arch = "x86_64")]
                if super::has_avx2_fma_cached() {
                    // SAFETY: AVX2 and FMA support was checked just above.
                    unsafe { simd::$cmul::<ACC>(dst, a, b) };
                    return;
                }
                cmul_portable::<Self, ACC>(dst, a, b);
            }

            fn with_planner<R>(f: impl FnOnce(&mut FftPlanner<Self>) -> R) -> R {
                $planner.with(|p| f(&mut p.borrow_mut()))
            }

            fn with_engines<R>(f: impl FnOnce(&mut EngineCache<Self>) -> R) -> R {
                $engines.with(|c| f(&mut c.borrow_mut()))
            }
        }
    };
}

rs_float!(f32, dot_f32, interp_f32, cmul_f32, PLANNER_F32, ENGINES_F32);
rs_float!(f64, dot_f64, interp_f64, cmul_f64, PLANNER_F64, ENGINES_F64);

/// Most engines kept per thread and float type.
const ENGINE_CACHE_ENTRIES: usize = 16;
/// Most engine memory (filter spectra and interpolator tables) kept per
/// thread and float type.
const ENGINE_CACHE_BYTES: usize = 32 << 20;

/// Returns the (per-thread cached) engine for a conversion, or `None` when
/// the plan is a direct single-stage one.
///
/// # Errors
/// See [`super::plan`].
pub(super) fn cached_engine<F: RsFloat>(
    in_sr: u32,
    out_sr: u32,
    quality: crate::operations::types::ResamplingQuality,
) -> crate::AudioSampleResult<Option<Arc<Engine<F>>>> {
    let key = super::KernelKey {
        in_sr,
        out_sr,
        quality,
        stage: super::Stage::Single,
    };
    if let Some(hit) = F::with_engines(|cache| {
        let pos = cache.iter().position(|(k, _)| *k == key)?;
        let entry = cache.remove(pos);
        let engine = entry.1.clone();
        cache.insert(0, entry);
        Some(engine)
    }) {
        return Ok(hit);
    }
    // Built outside the cache borrow (plan building uses other thread-locals).
    let engine = Engine::new(&super::plan(in_sr, out_sr, quality)?).map(Arc::new);
    F::with_engines(|cache| {
        cache.insert(0, (key, engine.clone()));
        let mut total = 0;
        let keep = cache
            .iter()
            .position(|(_, e)| {
                total += e.as_ref().map_or(0, |e| e.size_bytes());
                total > ENGINE_CACHE_BYTES
            })
            .unwrap_or(cache.len())
            .min(ENGINE_CACHE_ENTRIES);
        cache.truncate(keep);
    });
    Ok(engine)
}

/// A plan prepared for evaluation in `F`.
pub(super) struct Engine<F> {
    a: FftStage<F>,
    b: Option<Interp<F>>,
    avx2_fma: bool,
}

impl<F: RsFloat> Engine<F> {
    /// `None` for a [`Plan::Direct`].
    fn new(plan: &Plan) -> Option<Self> {
        let (a, b) = match plan {
            Plan::Direct(_) => return None,
            Plan::Fft { kernel, up, down } => (FftStage::new(kernel, *up, *down), None),
            Plan::TwoStage { a, up, b } => (FftStage::new(a, *up, 1), Some(Interp::new(b))),
        };
        Some(Self {
            a,
            b,
            avx2_fma: super::has_avx2_fma_cached(),
        })
    }

    fn size_bytes(&self) -> usize {
        self.a.spec.len() * std::mem::size_of::<Complex<F>>()
            + self
                .b
                .as_ref()
                .map_or(0, |b| b.coeffs.len() * std::mem::size_of::<F>())
    }

    /// Resamples one channel `x` into `out`, whose length is the number of
    /// output samples to produce (at most the full output length). Every
    /// element of `out` is written.
    pub(super) fn run<T: Copy>(
        &self,
        x: &[T],
        out: &mut [MaybeUninit<T>],
        to_f: &impl Fn(T) -> F,
        from_f: &impl Fn(F) -> T,
    ) {
        #[cfg(target_arch = "x86_64")]
        if self.avx2_fma {
            // SAFETY: `avx2_fma` is only true when the CPU supports AVX2 and FMA.
            unsafe { run_avx2_fma(self, x, out, to_f, from_f) };
            return;
        }
        run_impl::<T, F, false>(self, x, out, to_f, from_f);
    }
}

/// [`run_impl`] compiled for AVX2 + FMA.
///
/// # Safety
/// The CPU must support `avx2` and `fma`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn run_avx2_fma<T: Copy, F: RsFloat>(
    e: &Engine<F>,
    x: &[T],
    out: &mut [MaybeUninit<T>],
    to_f: &impl Fn(T) -> F,
    from_f: &impl Fn(F) -> T,
) {
    run_impl::<T, F, true>(e, x, out, to_f, from_f);
}

#[inline(always)]
fn run_impl<T: Copy, F: RsFloat, const FMA: bool>(
    e: &Engine<F>,
    x: &[T],
    out: &mut [MaybeUninit<T>],
    to_f: &impl Fn(T) -> F,
    from_f: &impl Fn(F) -> T,
) {
    let a = &e.a;
    let mut bufs = PairBufs::new(a);
    // Stage-A outputs per block.
    let blk = a.hop / a.down;
    let n_out = out.len() as i64;
    let Some(b) = &e.b else {
        // Stage A is the whole conversion: write its blocks straight out.
        let mut k0 = 0i64;
        while k0 < n_out {
            a.pair(x, to_f, k0 * a.down as i64, &mut bufs);
            let start = k0 as usize;
            let n_a = (n_out - k0).min(blk as i64) as usize;
            for (o, &v) in out[start..start + n_a].iter_mut().zip(&bufs.ya) {
                o.write(from_f(v));
            }
            if k0 + (blk as i64) < n_out {
                let start = start + blk;
                let n_b = (n_out - k0 - blk as i64).min(blk as i64) as usize;
                for (o, &v) in out[start..start + n_b].iter_mut().zip(&bufs.yb) {
                    o.write(from_f(v));
                }
            }
            k0 += 2 * blk as i64;
        }
        return;
    };
    if n_out == 0 {
        return;
    }
    // Two stages: stage A's output `y1` (at `up · in_rate`) is produced block
    // by block into `buf` (holding `y1[buf_start..]`); stage B consumes it.
    let half_b = b.half as i64;
    let (l, m) = (b.l, b.m);
    let last_n0 = ((n_out as u128 - 1) * u128::from(m) / u128::from(l)) as i64;
    // The last window ends at `last_n0 + half_b`, plus the rows' zero padding.
    let k_hi = last_n0 + half_b + 1 + (b.taps - 2 * b.half) as i64;
    let up = a.up as i64;
    // First `y1` index: the first output's window starts at `1 - half_b`;
    // blocks start at multiples of `up`.
    let mut k0 = (-half_b).div_euclid(up) * up;
    let mut buf: Vec<F> = Vec::with_capacity(2 * blk + b.taps + 1);
    let mut buf_start = k0;
    let (step_int, step_frac) = ((m / l) as i64, m % l);
    let mut st = BState { j: 0, n0: 0, p: 0 };
    let n_out = n_out as usize;
    while st.j < n_out && k0 < k_hi {
        a.pair(x, to_f, k0, &mut bufs);
        let take = |from: i64| (k_hi - from).clamp(0, blk as i64) as usize;
        buf.extend_from_slice(&bufs.ya[..take(k0)]);
        buf.extend_from_slice(&bufs.yb[..take(k0 + blk as i64)]);
        k0 += 2 * blk as i64;
        st = run_b::<T, F, FMA>(b, &buf, buf_start, out, from_f, st, step_int, step_frac);
        let drop = ((st.n0 - half_b + 1 - buf_start).max(0) as usize).min(buf.len());
        buf.drain(..drop);
        buf_start += drop as i64;
    }
    debug_assert_eq!(st.j, n_out);
    // Not reached, but `out` must end up initialised whatever happens.
    for o in &mut out[st.j..] {
        o.write(from_f(F::zero()));
    }
}

/// Stage B's position: next output `j`, its input index `n0` and phase `p`.
#[derive(Clone, Copy)]
struct BState {
    j: usize,
    n0: i64,
    p: u64,
}

/// Runs stage B over the buffered stage-A output (see [`Interp::run`]),
/// dispatching on the tap count so that common lengths get fully unrolled
/// dot products.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn run_b<T, F: RsFloat, const FMA: bool>(
    b: &Interp<F>,
    buf: &[F],
    buf_start: i64,
    out: &mut [MaybeUninit<T>],
    from_f: &impl Fn(F) -> T,
    st: BState,
    step_int: i64,
    step_frac: u64,
) -> BState {
    macro_rules! go {
        ($exact:literal, $($n:literal)*) => {
            match b.taps {
                $($n => b.run::<T, FMA, $exact, $n>(buf, buf_start, out, from_f, st, step_int, step_frac),)*
                _ => b.run::<T, FMA, $exact, 0>(buf, buf_start, out, from_f, st, step_int, step_frac),
            }
        };
    }
    if b.exact {
        go!(true, 12 16 20 24 28 32 36 40 44 48)
    } else {
        go!(false, 12 16 20 24 28 32 36 40 44 48)
    }
}

// ---------------------------------------------------------------------------
// Stage A: FFT overlap-save with integer up- or down-sampling
// ---------------------------------------------------------------------------

/// Stage A evaluated by FFT (see the [module documentation](self)).
struct FftStage<F> {
    /// Zero-stuffing factor (convolution rate = `up · in_rate`).
    up: usize,
    /// Decimation factor after the convolution.
    down: usize,
    /// FFT size at the convolution rate.
    n: usize,
    /// The filter spans `[-g, g]` (convolution-rate samples); a multiple of
    /// `max(up, down)`.
    g: usize,
    /// Convolution-rate outputs per block (a multiple of `max(up, down)`).
    hop: usize,
    /// Spectrum of the filter (`n` bins), scaled by `1 / n`.
    spec: Vec<Complex<F>>,
    /// Forward FFT of size `n / up`.
    fwd: Arc<dyn Fft<F>>,
    /// Inverse FFT of size `n / down`.
    inv: Arc<dyn Fft<F>>,
}

/// FFT size for a filter spanning `[-g, g]`: minimises a cost model (FFT
/// work and spectrum products per valid output; sizes above 2048 are
/// penalised for leaving the L1 cache).
fn fft_size(g: usize, up: usize, down: usize) -> usize {
    let unit = up.max(down);
    let fft = |s: usize| {
        let lg = (s as f64).log2();
        s as f64 * lg * (1.0 + 0.1 * (lg - 11.0).max(0.0))
    };
    (6..=20)
        .map(|e| 1usize << e)
        .filter(|&n| n >= 4 * g && n / unit >= 64)
        .map(|n| {
            let hop = (n - 2 * g) / unit * unit;
            let cost = (fft(n / up) + fft(n / down) + 4.0 * n as f64) / hop as f64;
            (n, cost)
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(1 << 20, |t| t.0)
}

impl<F: RsFloat> FftStage<F> {
    /// The FFT form of `kernel`, an exact table with `up` rows and `down`
    /// input samples per output (`kernel.l == up`, `kernel.m == down`).
    fn new(kernel: &Kernel, up: usize, down: usize) -> Self {
        debug_assert!(kernel.exact && kernel.l as usize == up && kernel.m as usize == down);
        debug_assert!(up == 1 || down == 1);
        let unit = up.max(down);
        let g = (up * kernel.half).next_multiple_of(unit);
        let n = fft_size(g, up, down);
        let hop = (n - 2 * g) / unit * unit;
        // Output k of row p uses x[n0 - half + 1 + j] with weight row[j]: in
        // convolution-rate samples, tap j sits at d = p - up (j - half + 1).
        let scale = 1.0 / n as f64;
        let mut gw = vec![Complex::new(0.0f64, 0.0); n];
        let taps = kernel.taps;
        for p in 0..up {
            for (j, &c) in kernel.coeffs[p * taps..(p + 1) * taps].iter().enumerate() {
                let d = p as i64 - up as i64 * (j as i64 - kernel.half as i64 + 1);
                gw[d.rem_euclid(n as i64) as usize] = Complex::new(c * scale, 0.0);
            }
        }
        let plan = f64::with_planner(|p| p.plan_fft_forward(n));
        plan.process(&mut gw);
        let spec = gw
            .iter()
            .map(|c| Complex::new(F::of(c.re), F::of(c.im)))
            .collect();
        let (fwd, inv) =
            F::with_planner(|p| (p.plan_fft_forward(n / up), p.plan_fft_inverse(n / down)));
        Self {
            up,
            down,
            n,
            g,
            hop,
            spec,
            fwd,
            inv,
        }
    }

    /// Computes two consecutive blocks, the convolution-rate outputs
    /// `[m0, m0 + hop)` and `[m0 + hop, m0 + 2 hop)` decimated by `down`,
    /// into `bufs.ya` and `bufs.yb`. `m0` must be a multiple of
    /// `max(up, down)`.
    #[inline(always)]
    fn pair<T: Copy>(&self, x: &[T], to_f: &impl Fn(T) -> F, m0: i64, bufs: &mut PairBufs<F>) {
        let up = self.up as i64;
        let start = (m0 - self.g as i64).div_euclid(up);
        load_pair(x, to_f, start, start + self.hop as i64 / up, &mut bufs.z);
        // Out of place: the in-place transforms copy through scratch.
        self.fwd
            .process_outofplace_with_scratch(&mut bufs.z, &mut bufs.zf, &mut bufs.scratch);
        let PairBufs {
            zf,
            w,
            y,
            ya,
            yb,
            scratch,
            ..
        } = bufs;
        if self.down == 1 {
            // Zero-stuffing by `up` repeats the spectrum `up` times.
            let seg = zf.len();
            for (wc, sc) in w.chunks_exact_mut(seg).zip(self.spec.chunks_exact(seg)) {
                F::cmul::<false>(wc, zf, sc);
            }
        } else {
            // Decimation by `down` sums the `down` aliases of the spectrum.
            let seg = w.len();
            for i in 0..self.down {
                let range = i * seg..(i + 1) * seg;
                if i == 0 {
                    F::cmul::<false>(w, &zf[range.clone()], &self.spec[range]);
                } else {
                    F::cmul::<true>(w, &zf[range.clone()], &self.spec[range]);
                }
            }
        }
        self.inv.process_outofplace_with_scratch(w, y, scratch);
        let first = self.g / self.down;
        let valid = &y[first..first + ya.len()];
        for ((a, b), v) in ya.iter_mut().zip(yb.iter_mut()).zip(valid) {
            *a = v.re;
            *b = v.im;
        }
    }
}

/// Work buffers of [`FftStage::pair`].
struct PairBufs<F> {
    /// Input segments (`n / up`).
    z: Vec<Complex<F>>,
    /// Their spectrum.
    zf: Vec<Complex<F>>,
    /// Product spectrum (`n / down`).
    w: Vec<Complex<F>>,
    /// Inverse transform.
    y: Vec<Complex<F>>,
    /// The two output blocks (`hop / down` samples each).
    ya: Vec<F>,
    yb: Vec<F>,
    scratch: Vec<Complex<F>>,
}

impl<F: RsFloat> PairBufs<F> {
    fn new(a: &FftStage<F>) -> Self {
        let zero = Complex::new(F::zero(), F::zero());
        let scratch = a
            .fwd
            .get_outofplace_scratch_len()
            .max(a.inv.get_outofplace_scratch_len());
        Self {
            z: vec![zero; a.n / a.up],
            zf: vec![zero; a.n / a.up],
            w: vec![zero; a.n / a.down],
            y: vec![zero; a.n / a.down],
            ya: vec![F::zero(); a.hop / a.down],
            yb: vec![F::zero(); a.hop / a.down],
            scratch: vec![zero; scratch],
        }
    }
}

/// Complex products `dst[i] (+)= a[i] · b[i]` (`ACC`: accumulate), portable.
#[inline(always)]
fn cmul_portable<F: RsFloat, const ACC: bool>(
    dst: &mut [Complex<F>],
    a: &[Complex<F>],
    b: &[Complex<F>],
) {
    for ((o, a), b) in dst.iter_mut().zip(a).zip(b) {
        let re = a.re * b.re - a.im * b.im;
        let im = a.re * b.im + a.im * b.re;
        if ACC {
            o.re += re;
            o.im += im;
        } else {
            *o = Complex::new(re, im);
        }
    }
}

/// Fills `z[i]` with `(x[start_a + i], x[start_b + i])`, zero outside `x`.
#[inline(always)]
fn load_pair<T: Copy, F: RsFloat>(
    x: &[T],
    to_f: &impl Fn(T) -> F,
    start_a: i64,
    start_b: i64,
    z: &mut [Complex<F>],
) {
    let len = z.len() as i64;
    let nx = x.len() as i64;
    if start_a >= 0 && start_b >= 0 && start_a + len <= nx && start_b + len <= nx {
        let xa = &x[start_a as usize..(start_a + len) as usize];
        let xb = &x[start_b as usize..(start_b + len) as usize];
        for ((c, &a), &b) in z.iter_mut().zip(xa).zip(xb) {
            *c = Complex::new(to_f(a), to_f(b));
        }
        return;
    }
    load_lane::<T, F, false>(x, to_f, start_a, z);
    load_lane::<T, F, true>(x, to_f, start_b, z);
}

/// Writes `x[start + i]` (zero outside `x`) into the real (`IM = false`) or
/// imaginary part of `z[i]`.
#[inline(always)]
fn load_lane<T: Copy, F: RsFloat, const IM: bool>(
    x: &[T],
    to_f: &impl Fn(T) -> F,
    start: i64,
    z: &mut [Complex<F>],
) {
    let len = z.len() as i64;
    let lo = (-start).clamp(0, len) as usize;
    let hi = ((x.len() as i64 - start).clamp(0, len) as usize).max(lo);
    let set = |c: &mut Complex<F>, v: F| {
        if IM {
            c.im = v;
        } else {
            c.re = v;
        }
    };
    for c in &mut z[..lo] {
        set(c, F::zero());
    }
    if hi > lo {
        let src = &x[(start + lo as i64) as usize..(start + hi as i64) as usize];
        for (c, &s) in z[lo..hi].iter_mut().zip(src) {
            set(c, to_f(s));
        }
    }
    for c in &mut z[hi..] {
        set(c, F::zero());
    }
}

// ---------------------------------------------------------------------------
// Stage B: short polyphase interpolator
// ---------------------------------------------------------------------------

/// Stage B's coefficient table in `F` (see [`Kernel`] for the layout), each
/// row padded with zeros to whole SIMD registers (8 `f32` or 4 `f64`); the
/// padding multiplies the samples just after the window.
struct Interp<F> {
    l: u64,
    m: u64,
    half: usize,
    /// Padded row length.
    taps: usize,
    exact: bool,
    phases: usize,
    /// `phases / l`: row position per unit of phase.
    pos_scale: f64,
    coeffs: Vec<F>,
}

impl<F: RsFloat> Interp<F> {
    fn new(kernel: &Kernel) -> Self {
        let lanes = 32 / std::mem::size_of::<F>();
        let taps = kernel.taps.next_multiple_of(lanes);
        let rows = kernel.coeffs.len() / kernel.taps;
        let mut coeffs = Vec::with_capacity(rows * taps);
        for row in kernel.coeffs.chunks_exact(kernel.taps) {
            coeffs.extend(row.iter().map(|&c| F::of(c)));
            coeffs.resize(coeffs.len() + taps - kernel.taps, F::zero());
        }
        Self {
            l: kernel.l,
            m: kernel.m,
            half: kernel.half,
            taps,
            exact: kernel.exact,
            phases: kernel.phases,
            pos_scale: kernel.phases as f64 / kernel.l as f64,
            coeffs,
        }
    }

    /// Produces outputs `st.j..` into `out` while their windows lie inside
    /// `buf` (which holds `y1[buf_start..]`), as [`Kernel::eval`] does for
    /// one output; returns the state for the next output. `TAPS` is
    /// `self.taps` as a constant (or 0).
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn run<T, const FMA: bool, const EXACT: bool, const TAPS: usize>(
        &self,
        buf: &[F],
        buf_start: i64,
        out: &mut [MaybeUninit<T>],
        from_f: &impl Fn(F) -> T,
        st: BState,
        step_int: i64,
        step_frac: u64,
    ) -> BState {
        let BState { mut j, n0, mut p } = st;
        let taps = if TAPS > 0 { TAPS } else { self.taps };
        debug_assert_eq!(taps, self.taps);
        let l = self.l;
        let last_row = self.phases as i64 - 1;
        let pos_scale = self.pos_scale;
        let coeffs = &self.coeffs[..];
        // Window start of output j, relative to `buf`.
        let mut s = n0 - self.half as i64 + 1 - buf_start;
        let s_end = buf.len() as i64 - taps as i64;
        while j < out.len() && s <= s_end {
            let window = &buf[s as usize..s as usize + taps];
            let v = if EXACT {
                let r = p as usize * taps;
                F::dot::<FMA>(&coeffs[r..r + taps], window)
            } else {
                let pos = p as f64 * pos_scale;
                let i = (pos as i64).min(last_row);
                let frac = F::of(pos - i as f64);
                let r = i as usize * taps;
                F::interp::<FMA>(
                    &coeffs[r..r + taps],
                    &coeffs[r + taps..r + 2 * taps],
                    window,
                    frac,
                )
            };
            out[j].write(from_f(v));
            j += 1;
            // Branch-free phase step (the wrap pattern is irregular).
            p += step_frac;
            let wrap = u64::from(p >= l);
            p -= wrap * l;
            s += step_int + wrap as i64;
        }
        BState {
            j,
            n0: s + buf_start + self.half as i64 - 1,
            p,
        }
    }
}

#[inline(always)]
fn hsum8<F: RsFloat>(a: [F; 8]) -> F {
    ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]))
}

/// Dot product of two equally long slices (no FMA).
#[inline(always)]
fn dot_portable<F: RsFloat>(a: &[F], b: &[F]) -> F {
    let mut acc = [F::zero(); 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail = ca
        .remainder()
        .iter()
        .zip(cb.remainder())
        .fold(F::zero(), |s, (x, y)| s + *x * *y);
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    hsum8(acc) + tail
}

/// AVX2 + FMA kernels: dot products (equal slice lengths, multiples of 8 for
/// `f32` and 4 for `f64`) and complex products.
#[cfg(target_arch = "x86_64")]
mod simd {
    use realfft::num_complex::Complex;
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn hsum_ps(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)))
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn hsum_pd(v: __m256d) -> f64 {
        let s = _mm_add_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd::<1>(v));
        _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)))
    }

    /// # Safety
    /// CPU with AVX2 + FMA; `a.len() == b.len()`, a multiple of 8.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        debug_assert!(a.len() == b.len() && a.len().is_multiple_of(8));
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i < a.len() {
            // SAFETY: i + 8 <= len for both slices.
            unsafe {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc);
            }
            i += 8;
        }
        hsum_ps(acc)
    }

    /// # Safety
    /// CPU with AVX2 + FMA; equal lengths, a multiple of 8.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn interp_f32(a: &[f32], b: &[f32], w: &[f32], frac: f32) -> f32 {
        debug_assert!(a.len() == w.len() && b.len() == w.len() && w.len().is_multiple_of(8));
        let (pa, pb, pw) = (a.as_ptr(), b.as_ptr(), w.as_ptr());
        let mut acc_a = _mm256_setzero_ps();
        let mut acc_b = _mm256_setzero_ps();
        let mut i = 0;
        while i < w.len() {
            // SAFETY: i + 8 <= len for all three slices.
            unsafe {
                let v = _mm256_loadu_ps(pw.add(i));
                acc_a = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), v, acc_a);
                acc_b = _mm256_fmadd_ps(_mm256_loadu_ps(pb.add(i)), v, acc_b);
            }
            i += 8;
        }
        let d = _mm256_sub_ps(acc_b, acc_a);
        hsum_ps(_mm256_fmadd_ps(_mm256_set1_ps(frac), d, acc_a))
    }

    /// # Safety
    /// CPU with AVX2 + FMA; equal lengths, a multiple of 4.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
        debug_assert!(a.len() == b.len() && a.len().is_multiple_of(4));
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut acc0 = _mm256_setzero_pd();
        let mut acc1 = _mm256_setzero_pd();
        let mut i = 0;
        while i + 8 <= a.len() {
            // SAFETY: i + 8 <= len for both slices.
            unsafe {
                acc0 =
                    _mm256_fmadd_pd(_mm256_loadu_pd(pa.add(i)), _mm256_loadu_pd(pb.add(i)), acc0);
                acc1 = _mm256_fmadd_pd(
                    _mm256_loadu_pd(pa.add(i + 4)),
                    _mm256_loadu_pd(pb.add(i + 4)),
                    acc1,
                );
            }
            i += 8;
        }
        if i < a.len() {
            // SAFETY: i + 4 <= len (the lengths are multiples of 4).
            unsafe {
                acc0 =
                    _mm256_fmadd_pd(_mm256_loadu_pd(pa.add(i)), _mm256_loadu_pd(pb.add(i)), acc0);
            }
        }
        hsum_pd(_mm256_add_pd(acc0, acc1))
    }

    /// # Safety
    /// CPU with AVX2 + FMA; equal lengths, a multiple of 4.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn interp_f64(a: &[f64], b: &[f64], w: &[f64], frac: f64) -> f64 {
        debug_assert!(a.len() == w.len() && b.len() == w.len() && w.len().is_multiple_of(4));
        let (pa, pb, pw) = (a.as_ptr(), b.as_ptr(), w.as_ptr());
        let mut acc_a = _mm256_setzero_pd();
        let mut acc_b = _mm256_setzero_pd();
        let mut i = 0;
        while i < w.len() {
            // SAFETY: i + 4 <= len for all three slices.
            unsafe {
                let v = _mm256_loadu_pd(pw.add(i));
                acc_a = _mm256_fmadd_pd(_mm256_loadu_pd(pa.add(i)), v, acc_a);
                acc_b = _mm256_fmadd_pd(_mm256_loadu_pd(pb.add(i)), v, acc_b);
            }
            i += 4;
        }
        let d = _mm256_sub_pd(acc_b, acc_a);
        hsum_pd(_mm256_fmadd_pd(_mm256_set1_pd(frac), d, acc_a))
    }

    /// `dst[i] (+)= a[i] · b[i]` over the common length.
    ///
    /// # Safety
    /// The CPU must support AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn cmul_f64<const ACC: bool>(
        dst: &mut [Complex<f64>],
        a: &[Complex<f64>],
        b: &[Complex<f64>],
    ) {
        let n = dst.len().min(a.len()).min(b.len());
        // `Complex<f64>` is `repr(C)`: (re, im) pairs.
        let pd = dst.as_mut_ptr().cast::<f64>();
        let (pa, pb) = (a.as_ptr().cast::<f64>(), b.as_ptr().cast::<f64>());
        let mut i = 0;
        while i + 2 <= n {
            // SAFETY: elements i and i + 1 are in bounds of all three slices.
            unsafe {
                let va = _mm256_loadu_pd(pa.add(2 * i));
                let vb = _mm256_loadu_pd(pb.add(2 * i));
                let br = _mm256_movedup_pd(vb);
                let bi = _mm256_permute_pd::<0b1111>(vb);
                let sw = _mm256_permute_pd::<0b0101>(va);
                let mut r = _mm256_fmaddsub_pd(va, br, _mm256_mul_pd(sw, bi));
                if ACC {
                    r = _mm256_add_pd(r, _mm256_loadu_pd(pd.add(2 * i)));
                }
                _mm256_storeu_pd(pd.add(2 * i), r);
            }
            i += 2;
        }
        super::cmul_portable::<f64, ACC>(&mut dst[i..n], &a[i..n], &b[i..n]);
    }

    /// `dst[i] (+)= a[i] · b[i]` over the common length.
    ///
    /// # Safety
    /// The CPU must support AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn cmul_f32<const ACC: bool>(
        dst: &mut [Complex<f32>],
        a: &[Complex<f32>],
        b: &[Complex<f32>],
    ) {
        let n = dst.len().min(a.len()).min(b.len());
        let pd = dst.as_mut_ptr().cast::<f32>();
        let (pa, pb) = (a.as_ptr().cast::<f32>(), b.as_ptr().cast::<f32>());
        let mut i = 0;
        while i + 4 <= n {
            // SAFETY: elements i .. i + 4 are in bounds of all three slices.
            unsafe {
                let va = _mm256_loadu_ps(pa.add(2 * i));
                let vb = _mm256_loadu_ps(pb.add(2 * i));
                let br = _mm256_moveldup_ps(vb);
                let bi = _mm256_movehdup_ps(vb);
                let sw = _mm256_permute_ps::<0b1011_0001>(va);
                let mut r = _mm256_fmaddsub_ps(va, br, _mm256_mul_ps(sw, bi));
                if ACC {
                    r = _mm256_add_ps(r, _mm256_loadu_ps(pd.add(2 * i)));
                }
                _mm256_storeu_ps(pd.add(2 * i), r);
            }
            i += 4;
        }
        super::cmul_portable::<f32, ACC>(&mut dst[i..n], &a[i..n], &b[i..n]);
    }
}
