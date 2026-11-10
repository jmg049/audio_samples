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

mod four_step;

use super::{Kernel, Plan};
use crate::AudioFloat;
use four_step::{FourStep, Layout};
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
    /// Four [`RsFloat::dot`] products at once, each with the same result as
    /// `dot` (the same operations in the same order).
    fn dot4<const FMA: bool>(a: [&[Self]; 4], b: [&[Self]; 4]) -> [Self; 4];
    /// [`RsFloat::dot`] of one row `a` with four windows, each with the same
    /// result as `dot`.
    fn dot_rows4<const FMA: bool>(a: &[Self], b: [&[Self]; 4]) -> [Self; 4];
    /// [`RsFloat::dot`] of one row `a` with two windows, each with the same
    /// result as `dot`.
    fn dot_rows2<const FMA: bool>(a: &[Self], b: [&[Self]; 2]) -> [Self; 2];
    /// `a · f` and `a · reverse(r)` over the first `r.len()` entries of `a`
    /// (`f` at least that long): a row against a window and, reversed, the
    /// mirror output's window.
    fn dot_mirror<const FMA: bool>(a: &[Self], f: &[Self], r: &[Self]) -> [Self; 2];
    /// `(1 - frac) · (a · w) + frac · (b · w)` in one pass over `w`.
    fn interp<const FMA: bool>(a: &[Self], b: &[Self], w: &[Self], frac: Self) -> Self;
    /// The 2 : 1 polyphase spectrum (see [`FftStage::pair_down2`]):
    /// `w[k] = A[k] P[k] + B[k] Q[k]` with `A = Za + i Zb` and
    /// `B[k] = conj(Za[-k]) + i conj(Zb[-k])`, all spectra in `lay` order.
    fn down2_bins(
        w: &mut [Complex<Self>],
        za: &[Complex<Self>],
        zb: &[Complex<Self>],
        t: &Bins<Self>,
        lay: Layout,
    );
    /// [`RsFloat::down2_bins`] with `Zb = 0`: `w = Z · P + conj(Z[-k]) · Q`
    /// (see [`FftStage::pair_fir2`]).
    fn fir2_bins(w: &mut [Complex<Self>], z: &[Complex<Self>], t: &Bins<Self>, lay: Layout);
    /// The radix-`r` pass of a [`FourStep`] transform (see
    /// [`four_step::radix_pass_portable`]); `simd`: AVX2 + FMA code where
    /// there is some (only when the CPU has them).
    #[allow(clippy::too_many_arguments)]
    fn radix_pass(
        r: usize,
        inv: bool,
        simd: bool,
        src: &[Complex<Self>],
        dst: &mut [Complex<Self>],
        tw: &[Complex<Self>],
        m: usize,
        ss: usize,
        ds: usize,
    );
    /// The bins of a block-wise conversion (see [`DftStage`]):
    /// `w[k] = C1 Z[k] + C2 conj(Z[-k]) + C3 conj(Z[M2 - k]) + C4 Z[M - M2 + k]`
    /// for `k < w.len() = M2`, `Z` of length `M`, with the coefficients in
    /// split form in `tab0` (bin 0) and `tab` (see [`DftStage::tab`]).
    /// `simd`: AVX2 + FMA code (only when the CPU has them).
    fn dft_bins(
        w: &mut [Complex<Self>],
        z: &[Complex<Self>],
        tab0: &[Complex<Self>; 8],
        tab: &[Complex<Self>],
        simd: bool,
    );
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
    ($t:ty, $dot:ident, $dot4:ident, $rows4:ident, $rows2:ident, $mirror:ident, $interp:ident, $cmul:ident, $down2:ident, $dft:ident, $radix:ident, $planner:ident, $engines:ident) => {
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
            fn dot4<const FMA: bool>(a: [&[Self]; 4], b: [&[Self]; 4]) -> [Self; 4] {
                #[cfg(target_arch = "x86_64")]
                if FMA {
                    // SAFETY: as in `dot`.
                    return unsafe { simd::$dot4(a, b) };
                }
                [0, 1, 2, 3].map(|t| dot_portable(a[t], b[t]))
            }

            #[inline(always)]
            fn dot_rows4<const FMA: bool>(a: &[Self], b: [&[Self]; 4]) -> [Self; 4] {
                #[cfg(target_arch = "x86_64")]
                if FMA {
                    // SAFETY: as in `dot`.
                    return unsafe { simd::$rows4(a, b) };
                }
                [0, 1, 2, 3].map(|t| dot_portable(a, b[t]))
            }

            #[inline(always)]
            fn dot_rows2<const FMA: bool>(a: &[Self], b: [&[Self]; 2]) -> [Self; 2] {
                #[cfg(target_arch = "x86_64")]
                if FMA {
                    // SAFETY: as in `dot`.
                    return unsafe { simd::$rows2(a, b) };
                }
                [dot_portable(a, b[0]), dot_portable(a, b[1])]
            }

            #[inline(always)]
            fn dot_mirror<const FMA: bool>(a: &[Self], f: &[Self], r: &[Self]) -> [Self; 2] {
                #[cfg(target_arch = "x86_64")]
                if FMA {
                    // SAFETY: as in `dot`.
                    return unsafe { simd::$mirror(a, f, r) };
                }
                let n = r.len();
                let fwd = a[..n]
                    .iter()
                    .zip(&f[..n])
                    .fold(0.0, |s: Self, (x, y)| s + *x * *y);
                let rev = a[..n]
                    .iter()
                    .zip(r.iter().rev())
                    .fold(0.0, |s: Self, (x, y)| s + *x * *y);
                [fwd, rev]
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
            fn down2_bins(
                w: &mut [Complex<Self>],
                za: &[Complex<Self>],
                zb: &[Complex<Self>],
                t: &Bins<Self>,
                lay: Layout,
            ) {
                #[cfg(target_arch = "x86_64")]
                if super::has_avx2_fma_cached() {
                    // SAFETY: AVX2 and FMA support was checked just above.
                    unsafe { simd::$down2::<true>(w, za, zb, t, lay) };
                    return;
                }
                for k1 in 0..lay.r {
                    down2_bins_portable::<_, true>(w, za, zb, t, lay, k1, 0, lay.m);
                }
            }

            #[inline(always)]
            fn fir2_bins(
                w: &mut [Complex<Self>],
                z: &[Complex<Self>],
                t: &Bins<Self>,
                lay: Layout,
            ) {
                #[cfg(target_arch = "x86_64")]
                if super::has_avx2_fma_cached() {
                    // SAFETY: AVX2 and FMA support was checked just above.
                    unsafe { simd::$down2::<false>(w, z, z, t, lay) };
                    return;
                }
                for k1 in 0..lay.r {
                    down2_bins_portable::<_, false>(w, z, z, t, lay, k1, 0, lay.m);
                }
            }

            #[inline(always)]
            fn radix_pass(
                r: usize,
                inv: bool,
                simd: bool,
                src: &[Complex<Self>],
                dst: &mut [Complex<Self>],
                tw: &[Complex<Self>],
                m: usize,
                ss: usize,
                ds: usize,
            ) {
                $radix(r, inv, simd, src, dst, tw, m, ss, ds);
            }

            #[inline(always)]
            fn dft_bins(
                w: &mut [Complex<Self>],
                z: &[Complex<Self>],
                tab0: &[Complex<Self>; 8],
                tab: &[Complex<Self>],
                simd: bool,
            ) {
                #[cfg(target_arch = "x86_64")]
                if simd {
                    // SAFETY: `simd` is only set on CPUs with AVX2 and FMA.
                    unsafe { simd::$dft(w, z, tab0, tab) };
                    return;
                }
                let _ = simd;
                dft_bins_portable(w, z, tab0, tab, 0, w.len());
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

rs_float!(
    f32,
    dot_f32,
    dot4_f32,
    rows4_f32,
    rows2_f32,
    mirror_f32,
    interp_f32,
    cmul_f32,
    down2_f32,
    dft_f32,
    radix_pass_f32,
    PLANNER_F32,
    ENGINES_F32
);
rs_float!(
    f64,
    dot_f64,
    dot4_f64,
    rows4_f64,
    rows2_f64,
    mirror_f64,
    interp_f64,
    cmul_f64,
    down2_f64,
    dft_f64,
    radix_pass_f64,
    PLANNER_F64,
    ENGINES_F64
);

/// [`RsFloat::radix_pass`] for `f32`: portable.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn radix_pass_f32(
    r: usize,
    inv: bool,
    _simd: bool,
    src: &[Complex<f32>],
    dst: &mut [Complex<f32>],
    tw: &[Complex<f32>],
    m: usize,
    ss: usize,
    ds: usize,
) {
    radix_portable(r, inv, src, dst, tw, m, ss, ds);
}

/// [`RsFloat::radix_pass`] for `f64`: AVX2 + FMA when `simd`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn radix_pass_f64(
    r: usize,
    inv: bool,
    simd: bool,
    src: &[Complex<f64>],
    dst: &mut [Complex<f64>],
    tw: &[Complex<f64>],
    m: usize,
    ss: usize,
    ds: usize,
) {
    #[cfg(target_arch = "x86_64")]
    if simd {
        use four_step::simd::radix_pass_f64 as pass;
        // SAFETY: `simd` is only set on CPUs with AVX2 and FMA; the sizes
        // are checked by the kernel.
        unsafe {
            match (r, inv) {
                (2, false) => pass::<2, false>(src, dst, tw, m, ss, ds),
                (2, true) => pass::<2, true>(src, dst, tw, m, ss, ds),
                (4, false) => pass::<4, false>(src, dst, tw, m, ss, ds),
                (4, true) => pass::<4, true>(src, dst, tw, m, ss, ds),
                (8, false) => pass::<8, false>(src, dst, tw, m, ss, ds),
                (8, true) => pass::<8, true>(src, dst, tw, m, ss, ds),
                _ => unreachable!("radix 2, 4 or 8"),
            }
        }
        return;
    }
    let _ = simd;
    radix_portable(r, inv, src, dst, tw, m, ss, ds);
}

#[allow(clippy::too_many_arguments)]
fn radix_portable<F: RsFloat>(
    r: usize,
    inv: bool,
    src: &[Complex<F>],
    dst: &mut [Complex<F>],
    tw: &[Complex<F>],
    m: usize,
    ss: usize,
    ds: usize,
) {
    use four_step::radix_pass_portable as pass;
    match (r, inv) {
        (2, false) => pass::<F, 2, false>(src, dst, tw, m, ss, ds),
        (2, true) => pass::<F, 2, true>(src, dst, tw, m, ss, ds),
        (4, false) => pass::<F, 4, false>(src, dst, tw, m, ss, ds),
        (4, true) => pass::<F, 4, true>(src, dst, tw, m, ss, ds),
        (8, false) => pass::<F, 8, false>(src, dst, tw, m, ss, ds),
        (8, true) => pass::<F, 8, true>(src, dst, tw, m, ss, ds),
        _ => unreachable!("radix 2, 4 or 8"),
    }
}

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
    stages: Stages<F>,
    avx2_fma: bool,
}

/// The evaluation of a plan: FFT stage A and the optional stage B, or the
/// block-wise frequency-domain conversion.
enum Stages<F> {
    Fft {
        a: Box<FftStage<F>>,
        b: Option<Interp<F>>,
    },
    Dft(DftStage<F>),
}

impl<F: RsFloat> Engine<F> {
    /// `None` for a [`Plan::Direct`].
    fn new(plan: &Plan) -> Option<Self> {
        let stages = match plan {
            Plan::Direct(_) => return None,
            Plan::Fft { kernel, up, down } => Stages::Fft {
                a: Box::new(FftStage::new(kernel, *up, *down)),
                b: None,
            },
            Plan::TwoStage { a, up, b } => Stages::Fft {
                a: Box::new(FftStage::new(a, *up, 1)),
                b: Some(Interp::new(b)),
            },
            Plan::Dft(spec) => Stages::Dft(DftStage::new(spec)),
        };
        Some(Self {
            stages,
            avx2_fma: super::has_avx2_fma_cached(),
        })
    }

    fn size_bytes(&self) -> usize {
        match &self.stages {
            Stages::Fft { a, b } => {
                a.spec.len() * std::mem::size_of::<Complex<F>>()
                    + [&a.down2, &a.fir2]
                        .iter()
                        .map(|t| t.as_ref().map_or(0, Bins::size_bytes))
                        .sum::<usize>()
                    + b.as_ref()
                        .map_or(0, |b| b.coeffs.len() * std::mem::size_of::<F>())
            }
            Stages::Dft(d) => {
                d.tab.len() * std::mem::size_of::<Complex<F>>()
                    + d.up
                        .as_ref()
                        .map_or(0, |(a, b)| a.size_bytes() + b.size_bytes())
            }
        }
    }

    /// [`Engine::run`] for samples of type `F`, each output `y` written as
    /// `y · gain` computed in `f64` (1 leaves it unchanged).
    pub(super) fn run_native(&self, x: &[F], out: &mut [MaybeUninit<F>], gain: f64) {
        if gain == 1.0 {
            self.run_with(x, Some(x), out, &|v| v, &|v| v);
        } else {
            self.run_with(x, Some(x), out, &|v| v, &|v| F::of(v.as_f64() * gain));
        }
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
        self.run_with(x, None, out, to_f, from_f);
    }

    /// [`Engine::run`]; `native` is `x` itself when `T` is `F` (blocks inside
    /// it are then transformed in place instead of copied).
    fn run_with<T: Copy>(
        &self,
        x: &[T],
        native: Option<&[F]>,
        out: &mut [MaybeUninit<T>],
        to_f: &impl Fn(T) -> F,
        from_f: &impl Fn(F) -> T,
    ) {
        if let Stages::Dft(d) = &self.stages {
            d.run(x, native, out, to_f, from_f, self.avx2_fma);
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if self.avx2_fma {
                // SAFETY: `avx2_fma` is only true when the CPU supports AVX2 and FMA.
                unsafe { run_avx2_fma(self, x, native, out, to_f, from_f) };
                return;
            }
            // x86-64 CPUs without AVX2 run one portable body per float type, on a
            // converted copy of the channel (the same values as converting on the fly).
            let xf: Vec<F> = x.iter().map(|&s| to_f(s)).collect();
            let mut of = vec![MaybeUninit::<F>::uninit(); out.len()];
            run_portable(self, &xf, &mut of);
            for (o, v) in out.iter_mut().zip(of) {
                // SAFETY: `run_portable` wrote every element.
                o.write(from_f(unsafe { v.assume_init() }));
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        run_impl::<T, F, false>(self, x, native, out, to_f, from_f);
    }
}

/// [`run_impl`] without AVX2 on float input and output, once per float type.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
fn run_portable<F: RsFloat>(e: &Engine<F>, x: &[F], out: &mut [MaybeUninit<F>]) {
    run_impl::<F, F, false>(e, x, Some(x), out, &|v| v, &|v| v);
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
    native: Option<&[F]>,
    out: &mut [MaybeUninit<T>],
    to_f: &impl Fn(T) -> F,
    from_f: &impl Fn(F) -> T,
) {
    run_impl::<T, F, true>(e, x, native, out, to_f, from_f);
}

#[inline(always)]
fn run_impl<T: Copy, F: RsFloat, const FMA: bool>(
    e: &Engine<F>,
    x: &[T],
    native: Option<&[F]>,
    out: &mut [MaybeUninit<T>],
    to_f: &impl Fn(T) -> F,
    from_f: &impl Fn(F) -> T,
) {
    let Stages::Fft { a, b } = &e.stages else {
        unreachable!("block-wise plans run in `DftStage::run`")
    };
    let mut arena = PairArena::new(a);
    let mut bufs = arena.bufs();
    // Stage-A outputs per block.
    let blk = a.hop / a.down;
    let n_out = out.len() as i64;
    let Some(b) = b else {
        // Stage A is the whole conversion: its blocks go straight to `out`.
        for (k0, pair) in out.chunks_mut(2 * blk).enumerate() {
            let (ya, yb) = pair.split_at_mut(blk.min(pair.len()));
            let m0 = (k0 * 2 * blk * a.down) as i64;
            a.pair(x, native, to_f, m0, &mut bufs, ya, yb, from_f);
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
    // A short signal fills less than the two blocks and the window that `buf` can hold.
    let mut buf = vec![F::zero(); (2 * blk + b.taps + 1).min((k_hi - k0).max(0) as usize)];
    let mut filled = 0;
    let mut buf_start = k0;
    let (step_int, step_frac) = ((m / l) as i64, m % l);
    let mut st = BState { j: 0, n0: 0, p: 0 };
    let n_out = n_out as usize;
    while st.j < n_out && k0 < k_hi {
        // Stage A writes its two blocks straight after what `buf` holds.
        let take = |from: i64| (k_hi - from).clamp(0, blk as i64) as usize;
        let (n_a, n_b) = (take(k0), take(k0 + blk as i64));
        let end = filled + n_a + n_b;
        if end > buf.len() {
            buf.resize(end, F::zero());
        }
        // SAFETY: `MaybeUninit<F>` has the layout of `F`, and only values
        // are written through it.
        let fresh =
            unsafe { &mut *(std::ptr::from_mut(&mut buf[filled..end]) as *mut [MaybeUninit<F>]) };
        let (ya, yb) = fresh.split_at_mut(n_a);
        a.pair(x, native, to_f, k0, &mut bufs, ya, yb, &|v| v);
        k0 += 2 * blk as i64;
        st = run_b::<T, F, FMA>(
            b,
            &buf[..end],
            buf_start,
            out,
            from_f,
            st,
            (step_int, step_frac),
            k0 >= k_hi,
        );
        let drop = ((st.n0 - half_b + 1 - buf_start).max(0) as usize).min(end);
        buf.copy_within(drop..end, 0);
        filled = end - drop;
        buf_start += drop as i64;
    }
    debug_assert_eq!(st.j, n_out);
    // Not reached, but `out` must end up initialised whatever happens.
    for o in &mut out[st.j..] {
        o.write(from_f(F::zero()));
    }
}

/// Largest input step `m` per cycle of an exact stage B evaluated a few
/// cycles at a time (see [`Interp::run_exact`]); the buffered stage-A output
/// is then at most `8 m` samples.
const CHUNK_MAX_STEP: usize = 1 << 16;
/// Exact stage-B tables above this size (bytes) are evaluated up to eight
/// cycles at a time instead of four, so that they are read half as often.
const CHUNK_BIG_TABLE: usize = 1 << 20;

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
    step: (i64, u64),
    last: bool,
) -> BState {
    macro_rules! go {
        ($exact:literal, $($n:literal)*) => {
            match b.taps {
                $($n => b.run::<T, FMA, $exact, $n>(buf, buf_start, out, from_f, st, step, last),)*
                _ => b.run::<T, FMA, $exact, 0>(buf, buf_start, out, from_f, st, step, last),
            }
        };
    }
    if b.exact {
        go!(true, 12 16 20 24 28 32 36 40 44 48 52)
    } else {
        go!(false, 12 16 20 24 28 32 36 40 44 48 52)
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
    /// Spectrum of the filter (`n` bins), scaled by `1 / n`, in the
    /// order of the larger transform's bins.
    spec: Vec<Complex<F>>,
    /// Forward FFT of size `n / up`.
    fwd: Xform<F>,
    /// Inverse FFT of size `n / down`.
    inv: Xform<F>,
    /// 2 : 1 decimation in polyphase form (see [`FftStage::pair_down2`]):
    /// `spec` is then empty and these are `P = (G_e - i G_o) / 2` and
    /// `Q = (G_e + i G_o) / 2` (`n / 2` bins each).
    down2: Option<Down2<F>>,
    /// Plain FIR (`up = down = 1`, `f64`) in polyphase form (see
    /// [`FftStage::pair_fir2`]): `spec` is then empty and these are
    /// `P = G_e + i (G_o - G_o') / 2` and `Q = i (G_o + G_o') / 2`.
    fir2: Option<Down2<F>>,
}

/// The `P` and `Q` spectra of a polyphase stage, split for multiply-adds:
/// bin `k` of `pr` is `(P.re, P.re)`, of `pi` `(-P.im, P.im)`, and likewise
/// `qr`, `qi` for `Q`, so that `a · P = a ⊙ pr + swap(a) ⊙ pi` elementwise
/// (`swap` exchanges the real and imaginary parts).
pub(super) struct Bins<F> {
    pr: Vec<Complex<F>>,
    pi: Vec<Complex<F>>,
    qr: Vec<Complex<F>>,
    qi: Vec<Complex<F>>,
}

impl<F: RsFloat> Bins<F> {
    fn new(p: &[Complex<F>], q: &[Complex<F>]) -> Self {
        let re = |v: &[Complex<F>]| v.iter().map(|c| Complex::new(c.re, c.re)).collect();
        let im = |v: &[Complex<F>]| v.iter().map(|c| Complex::new(-c.im, c.im)).collect();
        Self {
            pr: re(p),
            pi: im(p),
            qr: re(q),
            qi: im(q),
        }
    }

    fn size_bytes(&self) -> usize {
        4 * self.pr.len() * std::mem::size_of::<Complex<F>>()
    }
}

type Down2<F> = Bins<F>;

/// A complex FFT whose spectrum is in natural order (`Plain`) or in a
/// [`FourStep`] permutation.
enum Xform<F> {
    Plain(Arc<dyn Fft<F>>),
    FourStep(Box<FourStep<F>>),
}

impl<F: RsFloat> Xform<F> {
    /// The forward (`inverse = false`) or inverse transform of `n` points:
    /// in four steps when `four_step` and `n` is a size that form serves.
    fn new(n: usize, inverse: bool, four_step: bool) -> Self {
        if four_step && let Some(t) = FourStep::new(n) {
            return Self::FourStep(Box::new(t));
        }
        Self::Plain(F::with_planner(|p| {
            if inverse {
                p.plan_fft_inverse(n)
            } else {
                p.plan_fft_forward(n)
            }
        }))
    }

    /// The forward and inverse transforms of `n_fwd` and `n_inv` points,
    /// in four steps when `four_step` allows it for both with the same
    /// radix (so that their bins line up row by row, see [`Layout`]).
    fn pair(n_fwd: usize, n_inv: usize, four_step: bool) -> (Self, Self) {
        let (f, i) = (
            Self::new(n_fwd, false, four_step),
            Self::new(n_inv, true, four_step),
        );
        if f.layout(n_fwd).r == i.layout(n_inv).r {
            (f, i)
        } else {
            (
                Self::new(n_fwd, false, false),
                Self::new(n_inv, true, false),
            )
        }
    }

    /// Order of the spectrum of this `n`-point transform.
    fn layout(&self, n: usize) -> Layout {
        match self {
            Self::Plain(_) => Layout::natural(n),
            Self::FourStep(t) => t.layout(),
        }
    }

    fn scratch_len(&self) -> usize {
        match self {
            Self::Plain(t) => t
                .get_outofplace_scratch_len()
                .max(t.get_immutable_scratch_len()),
            Self::FourStep(t) => t.scratch_len(),
        }
    }

    /// Forward transform of `src`, which is left unchanged.
    #[inline(always)]
    fn forward(&self, src: &[Complex<F>], dst: &mut [Complex<F>], scratch: &mut [Complex<F>]) {
        match self {
            Self::Plain(t) => t.process_immutable_with_scratch(src, dst, scratch),
            Self::FourStep(t) => t.forward(src, dst, scratch, super::has_avx2_fma_cached()),
        }
    }

    /// Forward transform of `src`, which may be overwritten (faster for
    /// large plain transforms).
    #[inline(always)]
    fn forward_mut(
        &self,
        src: &mut [Complex<F>],
        dst: &mut [Complex<F>],
        scratch: &mut [Complex<F>],
    ) {
        match self {
            Self::Plain(t) => t.process_outofplace_with_scratch(src, dst, scratch),
            Self::FourStep(t) => t.forward(src, dst, scratch, super::has_avx2_fma_cached()),
        }
    }

    /// Inverse transform of `src` (overwritten) into `dst`, natural order.
    #[inline(always)]
    fn inverse(&self, src: &mut [Complex<F>], dst: &mut [Complex<F>], scratch: &mut [Complex<F>]) {
        match self {
            Self::Plain(t) => t.process_outofplace_with_scratch(src, dst, scratch),
            Self::FourStep(t) => {
                t.inverse(src, dst, scratch, super::has_avx2_fma_cached(), false);
            }
        }
    }

    /// Inverse transform of `src` (overwritten) of `n` points into `dst`,
    /// as blocks of `.m` samples `.stride` apart (natural order: one block).
    #[inline(always)]
    fn inverse_rows(
        &self,
        n: usize,
        src: &mut [Complex<F>],
        dst: &mut [Complex<F>],
        scratch: &mut [Complex<F>],
    ) -> Layout {
        match self {
            Self::Plain(t) => {
                t.process_outofplace_with_scratch(src, dst, scratch);
                Layout::natural(n)
            }
            Self::FourStep(t) => {
                t.inverse(src, dst, scratch, super::has_avx2_fma_cached(), true);
                t.layout()
            }
        }
    }
}

/// Whether `f64` stages use [`FourStep`] transforms: on CPUs with AVX2 and
/// FMA (the radix pass is written for them).
fn four_step_for<F: RsFloat>() -> bool {
    std::mem::size_of::<F>() == 8 && super::has_avx2_fma_cached()
}

/// FFT size for a filter spanning `[-g, g]`: minimises a cost model (FFT
/// work and spectrum products per valid output; sizes above 2048 are
/// penalised for leaving the L1 cache).
fn fft_size(g: usize, up: usize, down: usize) -> usize {
    let unit = up.max(down);
    let fft = |s: usize| {
        let lg = (s as f64).log2();
        s as f64 * lg * (1.0 + 0.2 * (lg - 11.0).max(0.0))
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
        let to_f = |c: &Complex<f64>| Complex::new(F::of(c.re), F::of(c.im));
        if up == 1 && down == 2 {
            // Polyphase: c[2t] = (g_e * s_e)[t] + (g_o * s_o)[t] with
            // g_e[e] = g[2e], g_o[e] = g[2e - 1], s_e[k] = s[2k], s_o[k] = s[2k + 1].
            let half_n = n / 2;
            let mut ge = vec![Complex::new(0.0f64, 0.0); half_n];
            let mut go = vec![Complex::new(0.0f64, 0.0); half_n];
            for (i, c) in gw.iter().enumerate() {
                // `gw` holds g[d] at d mod n (already scaled by 1 / n).
                let d = if i < n / 2 {
                    i as i64
                } else {
                    i as i64 - n as i64
                };
                let v = Complex::new(c.re * 2.0, 0.0); // scale 1 / (n / 2)
                if d.rem_euclid(2) == 0 {
                    ge[(d / 2).rem_euclid(half_n as i64) as usize] = v;
                } else {
                    go[((d + 1) / 2).rem_euclid(half_n as i64) as usize] = v;
                }
            }
            let plan = f64::with_planner(|p| p.plan_fft_forward(half_n));
            plan.process(&mut ge);
            plan.process(&mut go);
            let i = Complex::new(0.0, 1.0);
            let pq: (Vec<_>, Vec<_>) = ge
                .iter()
                .zip(&go)
                .map(|(e, o)| (to_f(&((e - i * o) * 0.5)), to_f(&((e + i * o) * 0.5))))
                .unzip();
            let (fwd, inv) = Xform::pair(half_n, half_n, four_step_for::<F>());
            let lay = fwd.layout(half_n);
            let zero = Complex::new(F::zero(), F::zero());
            return Self {
                up,
                down,
                n,
                g,
                hop,
                spec: Vec::new(),
                fwd,
                inv,
                down2: Some(Bins::new(
                    &lay.arrange(&pq.0, zero),
                    &lay.arrange(&pq.1, zero),
                )),
                fir2: None,
            };
        }
        // f64 only: there the size-n transforms cost ~2.5x the size-n/2 ones
        // (L1 cache), so two half-size transforms per block win; in f32 the
        // ratio is ~2.2 and the extra spectrum pass eats the gain.
        if up == 1 && down == 1 && std::mem::size_of::<F>() == 8 {
            // Polyphase FIR: with x_e[k] = x[2k], x_o[k] = x[2k + 1],
            // g_e[e] = g[2e], g_o[e] = g[2e + 1], g_o'[e] = g[2e - 1]:
            // y_e = g_e * x_e + g_o' * x_o and y_o = g_o * x_e + g_e * x_o,
            // so W = Y_e + i Y_o = Z P + Z* Q with Z = FFT(x_e + i x_o).
            let half_n = n / 2;
            let zero = Complex::new(0.0f64, 0.0);
            let (mut ge, mut go, mut gp) =
                (vec![zero; half_n], vec![zero; half_n], vec![zero; half_n]);
            for (i, c) in gw.iter().enumerate() {
                let d = if i < n / 2 {
                    i as i64
                } else {
                    i as i64 - n as i64
                };
                let v = Complex::new(c.re * 2.0, 0.0); // scale 1 / (n / 2)
                let at = |e: i64| e.rem_euclid(half_n as i64) as usize;
                if d.rem_euclid(2) == 0 {
                    ge[at(d / 2)] = v;
                } else {
                    go[at((d - 1) / 2)] = v;
                    gp[at((d + 1) / 2)] = v;
                }
            }
            let plan = f64::with_planner(|p| p.plan_fft_forward(half_n));
            for v in [&mut ge, &mut go, &mut gp] {
                plan.process(v);
            }
            let i = Complex::new(0.0, 1.0);
            let pq: (Vec<_>, Vec<_>) = (0..half_n)
                .map(|k| {
                    let p = ge[k] + i * (go[k] - gp[k]) * 0.5;
                    let q = i * (go[k] + gp[k]) * 0.5;
                    (to_f(&p), to_f(&q))
                })
                .unzip();
            let (fwd, inv) = Xform::pair(half_n, half_n, four_step_for::<F>());
            let lay = fwd.layout(half_n);
            let zero = Complex::new(F::zero(), F::zero());
            return Self {
                up,
                down,
                n,
                g,
                hop,
                spec: Vec::new(),
                fwd,
                inv,
                down2: None,
                fir2: Some(Bins::new(
                    &lay.arrange(&pq.0, zero),
                    &lay.arrange(&pq.1, zero),
                )),
            };
        }
        let plan = f64::with_planner(|p| p.plan_fft_forward(n));
        plan.process(&mut gw);
        let spec: Vec<_> = gw.iter().map(to_f).collect();
        // Products of the spectra of one radix line up row by row (see
        // `pair`); other decimation factors keep the natural order.
        let (fwd, inv) = Xform::pair(n / up, n / down, four_step_for::<F>());
        let spec = if up == 1 {
            fwd.layout(n)
                .arrange(&spec, Complex::new(F::zero(), F::zero()))
        } else {
            inv.layout(n)
                .arrange(&spec, Complex::new(F::zero(), F::zero()))
        };
        Self {
            up,
            down,
            n,
            g,
            hop,
            spec,
            fwd,
            inv,
            down2: None,
            fir2: None,
        }
    }

    /// Computes two consecutive blocks, the convolution-rate outputs
    /// `[m0, m0 + hop)` and `[m0 + hop, m0 + 2 hop)` decimated by `down`,
    /// into `bufs.ya` and `bufs.yb`. `m0` must be a multiple of
    /// `max(up, down)`.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn pair<T: Copy, U>(
        &self,
        x: &[T],
        native: Option<&[F]>,
        to_f: &impl Fn(T) -> F,
        m0: i64,
        bufs: &mut PairBufs<F>,
        ya: &mut [MaybeUninit<U>],
        yb: &mut [MaybeUninit<U>],
        put: &impl Fn(F) -> U,
    ) {
        if let Some(t) = &self.down2 {
            self.pair_down2(x, native, to_f, m0, bufs, t, ya, yb, put);
            return;
        }
        if let Some(t) = &self.fir2 {
            self.pair_fir2(x, to_f, m0, bufs, t, ya, yb, put);
            return;
        }
        let up = self.up as i64;
        let start = (m0 - self.g as i64).div_euclid(up);
        load_pair(x, to_f, start, start + self.hop as i64 / up, bufs.z);
        // Out of place: the in-place transforms copy through scratch.
        self.fwd.forward_mut(bufs.z, bufs.zf, bufs.scratch);
        let PairBufs {
            zf, w, y, scratch, ..
        } = bufs;
        let (lz, lw) = (
            self.fwd.layout(self.n / self.up),
            self.inv.layout(self.n / self.down),
        );
        // Row by row (see `Layout`); one row in natural order.
        if self.down == 1 {
            // Zero-stuffing by `up` repeats the spectrum `up` times.
            let seg = lz.m;
            for k1 in 0..lw.r {
                let z = &zf[k1 * lz.stride..][..seg];
                let (wr, sr) = (&mut w[k1 * lw.stride..], &self.spec[k1 * lw.stride..]);
                for h in 0..self.up {
                    F::cmul::<false>(&mut wr[h * seg..][..seg], z, &sr[h * seg..][..seg]);
                }
            }
        } else {
            // Decimation by `down` sums the `down` aliases of the spectrum.
            let seg = lw.m;
            for k1 in 0..lz.r {
                let wr = &mut w[k1 * lw.stride..][..seg];
                let (zr, sr) = (&zf[k1 * lz.stride..], &self.spec[k1 * lz.stride..]);
                for i in 0..self.down {
                    let range = i * seg..(i + 1) * seg;
                    if i == 0 {
                        F::cmul::<false>(wr, &zr[range.clone()], &sr[range]);
                    } else {
                        F::cmul::<true>(wr, &zr[range.clone()], &sr[range]);
                    }
                }
            }
        }
        let rows = self.inv.inverse_rows(self.n / self.down, w, y, scratch);
        split_rows(y, rows, self.g / self.down, ya, yb, put);
    }
}

impl<F: RsFloat> FftStage<F> {
    /// [`FftStage::pair`] for 2 : 1 decimation in polyphase form. Each block's
    /// segment `s` is read as the complex signal `s_e + i s_o` (even and odd
    /// samples), so its transform has half the size; with `Z = FFT(s_e + i s_o)`
    /// and `Z*[k] = conj(Z[-k])`, the decimated block's spectrum is
    /// `Z · P + Z* · Q`. Two blocks share the inverse transform as real and
    /// imaginary parts.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn pair_down2<T: Copy, U>(
        &self,
        x: &[T],
        native: Option<&[F]>,
        to_f: &impl Fn(T) -> F,
        m0: i64,
        bufs: &mut PairBufs<F>,
        t: &Bins<F>,
        ya: &mut [MaybeUninit<U>],
        yb: &mut [MaybeUninit<U>],
        put: &impl Fn(F) -> U,
    ) {
        let half_n = self.n / 2;
        let lay = self.fwd.layout(half_n);
        let start = m0 - self.g as i64;
        let (za, zb) = bufs.z.split_at_mut(half_n);
        let (fa, fb) = bufs.zf.split_at_mut(lay.len());
        for (from, z, f) in [
            (start, &mut *za, &mut *fa),
            (start + self.hop as i64, &mut *zb, &mut fb[..lay.len()]),
        ] {
            match in_place(native, from, half_n) {
                Some(seg) => self.fwd.forward(seg, f, bufs.scratch),
                None => {
                    load_even_odd(x, to_f, from, z);
                    self.fwd.forward(z, f, bufs.scratch);
                }
            }
        }
        let w = &mut bufs.w[..lay.len()];
        F::down2_bins(w, fa, &fb[..lay.len()], t, lay);
        let rows = self.inv.inverse_rows(half_n, w, bufs.y, bufs.scratch);
        split_rows(bufs.y, rows, self.g / 2, ya, yb, put);
    }
}

impl<F: RsFloat> FftStage<F> {
    /// [`FftStage::pair`] for a plain FIR in polyphase form: each block's
    /// segment is read as `x_e + i x_o` (half-size transform) and its even and
    /// odd outputs come out of one half-size inverse transform as real and
    /// imaginary parts, i.e. interleaved in memory.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn pair_fir2<T: Copy, U>(
        &self,
        x: &[T],
        to_f: &impl Fn(T) -> F,
        m0: i64,
        bufs: &mut PairBufs<F>,
        t: &Bins<F>,
        ya: &mut [MaybeUninit<U>],
        yb: &mut [MaybeUninit<U>],
        put: &impl Fn(F) -> U,
    ) {
        let half_n = self.n / 2;
        let first = self.g / 2;
        for (blk, dst) in [(0, ya), (1, yb)] {
            if dst.is_empty() {
                continue;
            }
            let dst_len = dst.len();
            let start = m0 - self.g as i64 + blk * self.hop as i64;
            let lay = self.fwd.layout(half_n);
            let z = &mut bufs.z[..half_n];
            load_even_odd(x, to_f, start, z);
            let zf = &mut bufs.zf[..lay.len()];
            self.fwd.forward_mut(z, zf, bufs.scratch);
            let w = &mut bufs.w[..lay.len()];
            F::fir2_bins(w, zf, t, lay);
            let y = &mut bufs.y[..half_n];
            self.inv.inverse(w, y, bufs.scratch);
            // Even and odd outputs are the real and imaginary parts.
            let (pairs, rest) = dst.as_chunks_mut::<2>();
            for (d, v) in pairs.iter_mut().zip(&y[first..]) {
                d[0].write(put(v.re));
                d[1].write(put(v.im));
            }
            if let [d] = rest {
                d.write(put(y[first + dst_len / 2].re));
            }
        }
    }
}

/// Writes the real parts of `y` into `a` and the imaginary parts into `b`
/// (as many as each holds), through `put`.
#[inline(always)]
fn split_re_im<F: RsFloat, U>(
    y: &[Complex<F>],
    a: &mut [MaybeUninit<U>],
    b: &mut [MaybeUninit<U>],
    put: &impl Fn(F) -> U,
) {
    if a.len() == b.len() {
        for ((a, b), v) in a.iter_mut().zip(b.iter_mut()).zip(y) {
            a.write(put(v.re));
            b.write(put(v.im));
        }
    } else {
        for (a, v) in a.iter_mut().zip(y) {
            a.write(put(v.re));
        }
        for (b, v) in b.iter_mut().zip(y) {
            b.write(put(v.im));
        }
    }
}

/// [`split_re_im`] of `y[first..]`, `y` stored in blocks of `rows.m`
/// values `rows.stride` apart.
#[inline(always)]
fn split_rows<F: RsFloat, U>(
    y: &[Complex<F>],
    rows: Layout,
    first: usize,
    a: &mut [MaybeUninit<U>],
    b: &mut [MaybeUninit<U>],
    put: &impl Fn(F) -> U,
) {
    if rows.r == 1 {
        split_re_im(&y[first..], a, b, put);
        return;
    }
    let total = a.len().max(b.len());
    let (mut done, mut i) = (0, first);
    while done < total {
        let (row, col) = (i / rows.m, i % rows.m);
        let len = (rows.m - col).min(total - done);
        let seg = &y[row * rows.stride + col..][..len];
        let part = |v: &mut [MaybeUninit<U>]| {
            let lo = done.min(v.len());
            (lo, (done + len).min(v.len()))
        };
        let ((al, ah), (bl, bh)) = (part(a), part(b));
        split_re_im(seg, &mut a[al..ah], &mut b[bl..bh], put);
        done += len;
        i += len;
    }
}

/// `native[start..start + 2 n]` as `n` complex values, when `native` is
/// given and holds that range.
#[inline(always)]
fn in_place<F: RsFloat>(native: Option<&[F]>, start: i64, n: usize) -> Option<&[Complex<F>]> {
    let seg = native?.get(usize::try_from(start).ok()?..)?.get(..2 * n)?;
    // SAFETY: `Complex<F>` is `repr(C)` with two `F` fields, so `2 n` values
    // of `F` read as `n` complex values (with the alignment of `F`).
    Some(unsafe { std::slice::from_raw_parts(seg.as_ptr().cast::<Complex<F>>(), n) })
}

/// Fills `z[k]` with `(x[start + 2k], x[start + 2k + 1])`, zero outside `x`.
#[inline(always)]
fn load_even_odd<T: Copy, F: RsFloat>(
    x: &[T],
    to_f: &impl Fn(T) -> F,
    start: i64,
    z: &mut [Complex<F>],
) {
    let len = 2 * z.len() as i64;
    if start >= 0 && start + len <= x.len() as i64 {
        let src = &x[start as usize..(start + len) as usize];
        for (c, pair) in z.iter_mut().zip(src.as_chunks::<2>().0) {
            *c = Complex::new(to_f(pair[0]), to_f(pair[1]));
        }
        return;
    }
    let get = |i: i64| {
        if i >= 0 && i < x.len() as i64 {
            to_f(x[i as usize])
        } else {
            F::zero()
        }
    };
    for (k, c) in z.iter_mut().enumerate() {
        let i = start + 2 * k as i64;
        *c = Complex::new(get(i), get(i + 1));
    }
}

/// Work buffers of [`FftStage::pair`].
struct PairBufs<'a, F> {
    /// Input segments (`n / up`).
    z: &'a mut [Complex<F>],
    /// Their spectrum.
    zf: &'a mut [Complex<F>],
    /// Product spectrum (`n / down`).
    w: &'a mut [Complex<F>],
    /// Inverse transform.
    y: &'a mut [Complex<F>],
    scratch: &'a mut [Complex<F>],
}

/// One allocation holding the complex buffers of [`PairBufs`], each
/// starting on a 64-byte (cache-line) boundary. With separate vectors, which start wherever
/// the allocator puts them, the conversion runs 5-8 % slower (16 kHz to
/// 22.05 kHz and 22.05 kHz to 16 kHz by ratio, `f64`).
struct PairArena<F> {
    mem: Vec<Complex<F>>,
    /// Start and length of `z`, `zf`, `w`, `y` and `scratch`.
    parts: [(usize, usize); 5],
}

impl<F: RsFloat> PairArena<F> {
    fn new(a: &FftStage<F>) -> Self {
        let scratch = a.fwd.scratch_len().max(a.inv.scratch_len());
        // The polyphase FIR runs every transform at half size; 2 : 1
        // decimation transforms two segments of `n / 2` points.
        let lens = if a.fir2.is_some() {
            let h = a.fwd.layout(a.n / 2).len();
            [a.n / 2, h, h, a.n / 2, scratch]
        } else if a.down2.is_some() {
            let h = a.fwd.layout(a.n / 2).len();
            [a.n, 2 * h, h, h, scratch]
        } else {
            let w = a.inv.layout(a.n / a.down).len();
            [a.n / a.up, a.fwd.layout(a.n / a.up).len(), w, w, scratch]
        };
        let size = std::mem::size_of::<Complex<F>>();
        // Elements per cache line.
        let line = 64 / size;
        let total: usize = lens.iter().map(|&l| l + line).sum();
        let mem = vec![Complex::new(F::zero(), F::zero()); total];
        // Position of `mem[0]` within its cache line, in elements.
        let base = (mem.as_ptr() as usize / size) % line;
        let mut parts = [(0, 0); 5];
        let mut pos = 0usize;
        for (part, &len) in parts.iter_mut().zip(&lens) {
            let start = (base + pos).div_ceil(line) * line - base;
            *part = (start, len);
            pos = start + len;
        }
        debug_assert!(pos <= total);
        Self { mem, parts }
    }

    fn bufs(&mut self) -> PairBufs<'_, F> {
        let mut rest: &mut [Complex<F>] = &mut self.mem;
        let mut done = 0;
        let mut take = |(start, len): (usize, usize)| {
            let r = std::mem::take(&mut rest);
            let (_, r) = r.split_at_mut(start - done);
            let (part, r) = r.split_at_mut(len);
            rest = r;
            done = start + len;
            part
        };
        let [pz, pzf, pw, py, ps] = self.parts;
        PairBufs {
            z: take(pz),
            zf: take(pzf),
            w: take(pw),
            y: take(py),
            scratch: take(ps),
        }
    }
}

/// [`RsFloat::down2_bins`] for positions `from..to` of row `k1` of `lay`,
/// portable. The mirror `-k` of the bin at position `k2` of row `k1` is at
/// position `-k2 mod m` of row 0 when `k1 = 0`, else at `m - 1 - k2` of row
/// `r - k1`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn down2_bins_portable<F: RsFloat, const TWO: bool>(
    w: &mut [Complex<F>],
    za: &[Complex<F>],
    zb: &[Complex<F>],
    t: &Bins<F>,
    lay: Layout,
    k1: usize,
    from: usize,
    to: usize,
) {
    let Layout { r, m, stride } = lay;
    let (row, mrow) = (k1 * stride, (r - k1) % r * stride);
    for k2 in from..to {
        let kr = mrow + if k1 == 0 { (m - k2) % m } else { m - 1 - k2 };
        let k = row + k2;
        let zero = Complex::new(F::zero(), F::zero());
        let (a, b) = (za[k], if TWO { zb[k] } else { zero });
        let (ar, br) = (za[kr], if TWO { zb[kr] } else { zero });
        let av = Complex::new(a.re - b.im, a.im + b.re);
        let bv = Complex::new(ar.re + br.im, br.re - ar.im);
        let (pr, pi, qr, qi) = (t.pr[k], t.pi[k], t.qr[k], t.qi[k]);
        // The order of the AVX2 kernel's products and sums.
        w[k] = Complex::new(
            (av.re * pr.re + av.im * pi.re) + (bv.re * qr.re + bv.im * qi.re),
            (av.im * pr.im + av.re * pi.im) + (bv.im * qr.im + bv.re * qi.im),
        );
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
// Block-wise conversion in the frequency domain
// ---------------------------------------------------------------------------

/// A [`Plan::Dft`] prepared for evaluation in `F`.
///
/// Block `b`'s `n1` input samples (from `b · hop - half` on) are read as
/// `n1 / 2` complex values `z[t] = x[2t] + i x[2t + 1]` and transformed:
/// `Z`, `M = n1 / 2` points. The real spectrum is
/// `X[k] = (Z[k] + conj(Z[-k])) / 2 + e^(-2πik/n1) (Z[k] - conj(Z[-k])) / 2i`,
/// the filtered one `Y[k] = X[k] G[k]` (`G`: the filter's spectrum, shifted
/// by `half` samples so that the block's first output comes first), and the
/// `n2` output-rate samples come from one inverse transform of `M2 = n2 / 2`
/// points of `W[k] = (1 + i e^(2πik/n2)) Y[k] + (1 - i e^(2πik/n2)) conj(Y[M2 - k])`,
/// as `y[2u] + i y[2u + 1]`. All of it folds into four products per bin
/// with fixed coefficients ([`RsFloat::dft_bins`]).
pub(super) struct DftStage<F> {
    /// Input samples per block (`a · m`).
    n1: usize,
    /// Outputs per inverse transform (`a · l`).
    n2: usize,
    /// The filter spans `[-half, half]` input samples.
    half: usize,
    /// Input samples per block step (a multiple of `m`).
    hop: usize,
    /// Outputs per block (`hop · l / m`).
    hop_out: usize,
    /// Forward transform, `n1 / 2` points.
    fwd: Arc<dyn Fft<F>>,
    /// Inverse transform, `n2 / 2` points.
    inv: Arc<dyn Fft<F>>,
    /// Bin 0's coefficients in split form (see [`DftStage::tab`]).
    tab0: [Complex<F>; 8],
    /// Bins `1..M2` in groups of `G` bins (one SIMD register: 2 `f64` or 4
    /// `f32` complex values): for group `g`, eight runs of `G` values, the
    /// split forms `(c.re, c.re)`, `(-c.im, c.im)` of `C1` and `C4` and
    /// `(c.re, -c.re)`, `(c.im, c.im)` of `C2` and `C3`, in the order
    /// `C1, C1, C2, C2, C3, C3, C4, C4`. Each multiplies one of `v`, `swap(v)`
    /// elementwise (`swap` exchanges the real and imaginary parts), which
    /// gives `c · v` or `c · conj(v)`.
    tab: Vec<Complex<F>>,
    /// Upsampling (`n2 > n1`): `Y` (the filtered input band, `M` bins) from
    /// `Z` as `Y[k] = A[k] Z[k] + B[k] conj(Z[-k])`, then `W` from `Y`
    /// zero-extended to `M2` bins as `W[k] = p[k] Y[k] + q[k] conj(Y[M2 - k])`,
    /// both by [`RsFloat::fir2_bins`]; `tab0` and `tab` are then unused.
    up: Option<(Bins<F>, Bins<F>)>,
    /// Upsampling by 2 or more (`n2 >= 2 n1`): `up` holds instead the
    /// products giving `W[..M]` and `W[M2 - M..]` from `Z` directly.
    split_up: bool,
    simd: bool,
}

/// Complex values per SIMD register of `F` (bins per group of
/// [`DftStage::tab`]).
const fn dft_group<F>() -> usize {
    16 / std::mem::size_of::<F>()
}

impl<F: RsFloat> DftStage<F> {
    pub(super) fn new(spec: &super::DftSpec) -> Self {
        let (l, m, a, half) = (spec.l, spec.m, spec.a, spec.half);
        let (n1, n2) = (a * m, a * l);
        let (big_m, m2) = (n1 / 2, n2 / 2);
        let hop = (n1 - 2 * half) / m * m;
        let zero = Complex::new(0.0f64, 0.0);
        let mut g = vec![zero; n1];
        for (i, &t) in spec.taps.iter().enumerate() {
            let d = i as i64 - half as i64;
            g[d.rem_euclid(n1 as i64) as usize] = Complex::new(t, 0.0);
        }
        f64::with_planner(|p| p.plan_fft_forward(n1)).process(&mut g);
        let tau = std::f64::consts::TAU;
        // e^(2πi num / den), with the angle reduced exactly first.
        let turn = |num: usize, den: usize| {
            Complex::from_polar(1.0, tau * (num % den) as f64 / den as f64)
        };
        let i = Complex::new(0.0, 1.0);
        // A[k], B[k]: Y[k] = A[k] Z[k] + B[k] conj(Z[-k]), for k up to M2
        // (downsampling) or below M (upsampling).
        let (ak, bk): (Vec<_>, Vec<_>) = (0..=m2.min(big_m))
            .map(|k| {
                let gk = g[k] * turn(k * half, n1) / n1 as f64;
                let w1 = turn(n1 - k % n1, n1);
                (gk * (1.0 - i * w1) * 0.5, gk * (1.0 + i * w1) * 0.5)
            })
            .unzip();
        let coeffs = |k: usize| {
            let v = turn(k, n2);
            let (p, q) = (1.0 + i * v, 1.0 - i * v);
            [
                p * ak[k],
                p * bk[k],
                q * ak[m2 - k].conj(),
                q * bk[m2 - k].conj(),
            ]
        };
        let to_f = |re: f64, im: f64| Complex::new(F::of(re), F::of(im));
        let (fwd, inv) = F::with_planner(|p| (p.plan_fft_forward(big_m), p.plan_fft_inverse(m2)));
        if m2 >= 2 * big_m {
            // The two halves of `W` do not overlap: `W[k] = p Y[k]` for
            // `k < M` and `W[M2 - M + j] = q conj(Y[M - j])` for `0 < j < M`,
            // each two products on `Z` (zero between them and at `j = 0`).
            let cf = |c: Complex<f64>| to_f(c.re, c.im);
            let (lo_p, lo_q): (Vec<_>, Vec<_>) = (0..big_m)
                .map(|k| {
                    let p = 1.0 + i * turn(k, n2);
                    (cf(p * ak[k]), cf(p * bk[k]))
                })
                .unzip();
            let (hi_p, hi_q): (Vec<_>, Vec<_>) = (0..big_m)
                .map(|j| {
                    if j == 0 {
                        return (cf(Complex::new(0.0, 0.0)), cf(Complex::new(0.0, 0.0)));
                    }
                    let q = 1.0 - i * turn(m2 - big_m + j, n2);
                    (cf(q * bk[big_m - j].conj()), cf(q * ak[big_m - j].conj()))
                })
                .unzip();
            return Self {
                n1,
                n2,
                half,
                hop,
                hop_out: hop / m * l,
                fwd,
                inv,
                tab0: [Complex::new(F::zero(), F::zero()); 8],
                tab: Vec::new(),
                up: Some((Bins::new(&lo_p, &lo_q), Bins::new(&hi_p, &hi_q))),
                split_up: true,
                simd: super::has_avx2_fma_cached(),
            };
        }
        if m2 > big_m {
            let cf = |c: Complex<f64>| to_f(c.re, c.im);
            let (a, b): (Vec<_>, Vec<_>) = (0..big_m).map(|k| (cf(ak[k]), cf(bk[k]))).unzip();
            // `Y[M2]`, the output Nyquist bin, is zero: `q[0]` is not needed.
            let (p, q): (Vec<_>, Vec<_>) = (0..m2)
                .map(|k| {
                    let v = turn(k, n2);
                    let q = if k == 0 {
                        Complex::new(0.0, 0.0)
                    } else {
                        1.0 - i * v
                    };
                    (cf(1.0 + i * v), cf(q))
                })
                .unzip();
            return Self {
                n1,
                n2,
                half,
                hop,
                hop_out: hop / m * l,
                fwd,
                inv,
                tab0: [Complex::new(F::zero(), F::zero()); 8],
                tab: Vec::new(),
                up: Some((Bins::new(&a, &b), Bins::new(&p, &q))),
                split_up: false,
                simd: super::has_avx2_fma_cached(),
            };
        }
        let split = |c: [Complex<f64>; 4]| {
            [
                to_f(c[0].re, c[0].re),
                to_f(-c[0].im, c[0].im),
                to_f(c[1].re, -c[1].re),
                to_f(c[1].im, c[1].im),
                to_f(c[2].re, -c[2].re),
                to_f(c[2].im, c[2].im),
                to_f(c[3].re, c[3].re),
                to_f(-c[3].im, c[3].im),
            ]
        };
        let tab0 = split(coeffs(0));
        let group = dft_group::<F>();
        let groups = (m2 - 1).div_ceil(group);
        let mut tab = vec![Complex::new(F::zero(), F::zero()); groups * 8 * group];
        for k in 1..m2 {
            let (gi, lane) = ((k - 1) / group, (k - 1) % group);
            for (r, v) in split(coeffs(k)).into_iter().enumerate() {
                tab[(gi * 8 + r) * group + lane] = v;
            }
        }
        Self {
            n1,
            n2,
            half,
            hop,
            hop_out: hop / m * l,
            fwd,
            inv,
            tab0,
            tab,
            up: None,
            split_up: false,
            simd: super::has_avx2_fma_cached(),
        }
    }

    /// Input samples per block.
    pub(super) fn block_len(&self) -> usize {
        self.n1
    }

    /// Input samples from one block's start to the next.
    pub(super) fn hop(&self) -> usize {
        self.hop
    }

    /// Outputs per block.
    pub(super) fn hop_out(&self) -> usize {
        self.hop_out
    }

    /// Input samples of the first block before the signal start.
    pub(super) fn half(&self) -> usize {
        self.half
    }

    /// Converts the block `z` (`n1` samples as `n1 / 2` pairs); its outputs
    /// are then `bufs.y[u] = (y[2u], y[2u + 1])` for `2u < hop_out`.
    fn transform(&self, z: &[Complex<F>], bufs: &mut DftBufs<F>, simd: bool) {
        self.fwd
            .process_immutable_with_scratch(z, &mut bufs.zf, &mut bufs.scratch);
        if let (Some((lo, hi)), true) = (&self.up, self.split_up) {
            let (m, m2) = (bufs.zf.len(), bufs.w.len());
            F::fir2_bins(&mut bufs.w[..m], &bufs.zf, lo, Layout::natural(m));
            bufs.w[m..m2 - m].fill(Complex::new(F::zero(), F::zero()));
            F::fir2_bins(&mut bufs.w[m2 - m..], &bufs.zf, hi, Layout::natural(m));
        } else if let Some((a, b)) = &self.up {
            // `bufs.band[M..]` stays zero.
            let m = bufs.zf.len();
            F::fir2_bins(&mut bufs.band[..m], &bufs.zf, a, Layout::natural(m));
            let lay = Layout::natural(bufs.w.len());
            F::fir2_bins(&mut bufs.w, &bufs.band, b, lay);
        } else {
            F::dft_bins(&mut bufs.w, &bufs.zf, &self.tab0, &self.tab, simd);
        }
        self.inv
            .process_outofplace_with_scratch(&mut bufs.w, &mut bufs.y, &mut bufs.scratch);
    }

    /// Converts the block `seg` (`n1` samples) into `out` (at most
    /// `hop_out` samples).
    pub(super) fn block(&self, seg: &[F], bufs: &mut DftBufs<F>, out: &mut [F]) {
        debug_assert_eq!(seg.len(), self.n1);
        let mut z = std::mem::take(&mut bufs.z);
        load_even_odd(seg, &|v| v, 0, &mut z);
        self.transform(&z, bufs, self.simd);
        bufs.z = z;
        for (d, v) in out.as_chunks_mut::<2>().0.iter_mut().zip(&bufs.y) {
            *d = [v.re, v.im];
        }
        if out.len() % 2 == 1 {
            out[out.len() - 1] = bufs.y[out.len() / 2].re;
        }
    }

    /// Resamples one channel `x` into `out` (its length is the number of
    /// outputs to produce, at most the full output length), block by block.
    /// `native`: `x` itself when `T` is `F`; blocks inside it are then
    /// transformed in place instead of copied.
    fn run<T: Copy>(
        &self,
        x: &[T],
        native: Option<&[F]>,
        out: &mut [MaybeUninit<T>],
        to_f: &impl Fn(T) -> F,
        from_f: &impl Fn(F) -> T,
        simd: bool,
    ) {
        let mut bufs = DftBufs::new(self);
        let mut z = std::mem::take(&mut bufs.z);
        for (b, dst) in out.chunks_mut(self.hop_out).enumerate() {
            let start = (b * self.hop) as i64 - self.half as i64;
            match in_place(native, start, self.n1 / 2) {
                Some(seg) => self.transform(seg, &mut bufs, simd && self.simd),
                None => {
                    load_even_odd(x, to_f, start, &mut z);
                    self.transform(&z, &mut bufs, simd && self.simd);
                }
            }
            let (pairs, rest) = dst.as_chunks_mut::<2>();
            for (d, v) in pairs.iter_mut().zip(&bufs.y) {
                d[0].write(from_f(v.re));
                d[1].write(from_f(v.im));
            }
            if let [d] = rest {
                d.write(from_f(bufs.y[pairs.len()].re));
            }
        }
    }
}

/// Work buffers of [`DftStage`].
pub(super) struct DftBufs<F> {
    z: Vec<Complex<F>>,
    zf: Vec<Complex<F>>,
    /// Upsampling: the filtered input band, zero-extended to `n2 / 2` bins.
    band: Vec<Complex<F>>,
    w: Vec<Complex<F>>,
    y: Vec<Complex<F>>,
    scratch: Vec<Complex<F>>,
}

impl<F: RsFloat> DftBufs<F> {
    pub(super) fn new(d: &DftStage<F>) -> Self {
        let zero = Complex::new(F::zero(), F::zero());
        let scratch = d
            .fwd
            .get_immutable_scratch_len()
            .max(d.inv.get_outofplace_scratch_len());
        Self {
            z: vec![zero; d.n1 / 2],
            zf: vec![zero; d.n1 / 2],
            band: if d.up.is_some() && !d.split_up {
                vec![zero; d.n2 / 2]
            } else {
                Vec::new()
            },
            w: vec![zero; d.n2 / 2],
            y: vec![zero; d.n2 / 2],
            scratch: vec![zero; scratch],
        }
    }
}

/// [`RsFloat::dft_bins`] for bins `from..to`, portable (the products and
/// sums in the order of the AVX2 kernels).
#[inline(always)]
fn dft_bins_portable<F: RsFloat>(
    w: &mut [Complex<F>],
    z: &[Complex<F>],
    tab0: &[Complex<F>; 8],
    tab: &[Complex<F>],
    from: usize,
    to: usize,
) {
    let (big_m, m2) = (z.len(), w.len());
    let group = dft_group::<F>();
    let swap = |c: Complex<F>| Complex::new(c.im, c.re);
    for k in from..to {
        let v = [z[k], z[(big_m - k) % big_m], z[m2 - k], z[big_m - m2 + k]];
        let t = |r: usize| {
            if k == 0 {
                tab0[r]
            } else {
                tab[(((k - 1) / group) * 8 + r) * group + (k - 1) % group]
            }
        };
        // c_j · v_j for each j, then summed pairwise.
        let p = [0, 1, 2, 3].map(|j| {
            let (a, b, s) = (t(2 * j), t(2 * j + 1), swap(v[j]));
            Complex::new(v[j].re * a.re + s.re * b.re, v[j].im * a.im + s.im * b.im)
        });
        w[k] = (p[0] + p[1]) + (p[2] + p[3]);
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
    /// Row length before padding (`2 · half`).
    kernel_taps: usize,
    exact: bool,
    phases: usize,
    /// `phases / l`: row position per unit of phase.
    pos_scale: f64,
    /// Exact tables: rows in visiting order, followed by copies of the first
    /// three (so that four consecutive outputs read consecutive rows).
    coeffs: Vec<F>,
    /// Exact tables: window start of output `q` of a cycle of `l` outputs
    /// relative to the cycle's first, `q · m / l` rounded down, for
    /// `q < l + 3`.
    offsets: Vec<usize>,
}

impl<F: RsFloat> Interp<F> {
    fn new(kernel: &Kernel) -> Self {
        let lanes = 32 / std::mem::size_of::<F>();
        let taps = kernel.taps.next_multiple_of(lanes);
        let rows = kernel.coeffs.len() / kernel.taps;
        let mut coeffs = Vec::with_capacity(rows * taps);
        let mut push = |row: &[f64]| {
            coeffs.extend(row.iter().map(|&c| F::of(c)));
            coeffs.resize(coeffs.len() + taps - kernel.taps, F::zero());
        };
        let mut offsets = Vec::new();
        if kernel.exact {
            // Output k uses phase (k · m) mod l: store the rows in that order,
            // so that consecutive outputs read consecutive rows.
            let mut p = 0u64;
            for _ in 0..rows + 3 {
                let r = p as usize * kernel.taps;
                push(&kernel.coeffs[r..r + kernel.taps]);
                p = (p + kernel.m) % kernel.l;
            }
            offsets = (0..kernel.l + 3)
                .map(|q| (u128::from(q) * u128::from(kernel.m) / u128::from(kernel.l)) as usize)
                .collect();
        } else {
            kernel.coeffs.chunks_exact(kernel.taps).for_each(&mut push);
        }
        Self {
            l: kernel.l,
            m: kernel.m,
            half: kernel.half,
            taps,
            kernel_taps: kernel.taps,
            exact: kernel.exact,
            phases: kernel.phases,
            pos_scale: kernel.phases as f64 / kernel.l as f64,
            coeffs,
            offsets,
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
        (step_int, step_frac): (i64, u64),
        last: bool,
    ) -> BState {
        let BState { mut j, n0, mut p } = st;
        let taps = if TAPS > 0 { TAPS } else { self.taps };
        debug_assert_eq!(taps, self.taps);
        // Window start of output j, relative to `buf`.
        let mut s = n0 - self.half as i64 + 1 - buf_start;
        let s_end = buf.len() as i64 - taps as i64;
        if EXACT {
            return self.run_exact::<T, FMA, TAPS>(buf, buf_start, out, from_f, j, s, last);
        }
        let l = self.l;
        let last_row = self.phases as i64 - 1;
        let pos_scale = self.pos_scale;
        let coeffs = &self.coeffs[..];
        while j < out.len() && s <= s_end {
            let window = &buf[s as usize..s as usize + taps];
            let pos = p as f64 * pos_scale;
            let i = (pos as i64).min(last_row);
            let frac = F::of(pos - i as f64);
            let r = i as usize * taps;
            let v = F::interp::<FMA>(
                &coeffs[r..r + taps],
                &coeffs[r + taps..r + 2 * taps],
                window,
                frac,
            );
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

    /// [`Interp::run`] for an exact table: output `j` uses row `j mod l` and
    /// the window at `j · m / l` (rounded down), so the positions within a
    /// cycle of `l` outputs come from [`Interp::offsets`]. `s` is output `j`'s
    /// window start relative to `buf`, `s_end` the last start that fits.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn run_exact<T, const FMA: bool, const TAPS: usize>(
        &self,
        buf: &[F],
        buf_start: i64,
        out: &mut [MaybeUninit<T>],
        from_f: &impl Fn(F) -> T,
        mut j: usize,
        s: i64,
        last: bool,
    ) -> BState {
        let taps = if TAPS > 0 { TAPS } else { self.taps };
        let s_end = buf.len() as i64 - taps as i64;
        let (l, m) = (self.l as usize, self.m as usize);
        let coeffs = &self.coeffs[..(l + 3) * taps];
        let offsets = &self.offsets[..l + 3];
        let mut q = (j as u64 % self.l) as usize;
        // Window start of the cycle's first output, relative to `buf` (it
        // can be negative; the windows of outputs `j..` start at `s >= 0` or
        // later).
        let mut base = s - offsets[q] as i64;
        // Whole cycles, up to `most` at a time, row by row: each row is read
        // once for every cycle of the chunk. Tables larger than the caches are
        // worth longer chunks (more buffered stage-A output per pass). At the
        // end, the remaining whole cycles form one shorter chunk.
        let most = if l * taps * std::mem::size_of::<F>() > CHUNK_BIG_TABLE {
            8
        } else {
            4
        };
        // Signals shorter than four cycles run in output order (in chunks,
        // all of stage A's output would wait for the end).
        if q == 0 && m <= CHUNK_MAX_STEP && out.len() >= 4 * l {
            // Window start of the last output of `c` cycles, from `base`.
            let span = |c: usize| (c * m + offsets[l - 1]) as i64 - m as i64;
            loop {
                let whole = ((out.len() - j) / l).min(most);
                let Some(c) = (1..=whole).rev().find(|&c| base + span(c) <= s_end) else {
                    break;
                };
                if c < most && !last {
                    break;
                }
                for (r, row) in coeffs[..l * taps].chunks_exact(taps).enumerate() {
                    let w = (base + offsets[r] as i64) as usize;
                    let mut k = 0;
                    while k + 4 <= c {
                        let w = w + k * m;
                        let v = F::dot_rows4::<FMA>(
                            row,
                            [0, 1, 2, 3].map(|i| &buf[w + i * m..w + i * m + taps]),
                        );
                        for (i, v) in v.into_iter().enumerate() {
                            out[j + (k + i) * l + r].write(from_f(v));
                        }
                        k += 4;
                    }
                    for k in k..c {
                        let w = w + k * m;
                        out[j + k * l + r].write(from_f(F::dot::<FMA>(row, &buf[w..w + taps])));
                    }
                }
                j += c * l;
                base += (c * m) as i64;
            }
            if !last {
                // The rest waits for more of stage A's output.
                return BState {
                    j,
                    n0: base + buf_start + self.half as i64 - 1,
                    p: 0,
                };
            }
        } else if q == 0 && j == 0 && m <= CHUNK_MAX_STEP && out.len() >= l && out.len() < 4 * l {
            // One to three whole cycles: once all of stage A's output is
            // here, each row serves output `r` and, reversed, output `l - r`
            // (the filter is even), so the table is read once for both.
            if !last {
                return BState {
                    j,
                    n0: base + buf_start + self.half as i64 - 1,
                    p: 0,
                };
            }
            let c = out.len() / l;
            if base + ((c - 1) * m + offsets[l - 1]) as i64 <= s_end && base >= 0 {
                self.mirrored_cycles::<T, FMA>(buf, base as usize, c, out, from_f, j);
                j += c * l;
                base += (c * m) as i64;
            }
        }
        // Four outputs at a time while all four windows lie in `buf`.
        while j + 4 <= out.len() && base + offsets[q + 3] as i64 <= s_end {
            let at = [0, 1, 2, 3].map(|i| (base + offsets[q + i] as i64) as usize);
            let v = F::dot4::<FMA>(
                [0, 1, 2, 3].map(|i| &coeffs[(q + i) * taps..(q + i + 1) * taps]),
                at.map(|a| &buf[a..a + taps]),
            );
            for (o, v) in out[j..j + 4].iter_mut().zip(v) {
                o.write(from_f(v));
            }
            j += 4;
            q += 4;
            if q >= l {
                base += (q / l * m) as i64;
                q %= l;
            }
        }
        loop {
            let s = base + offsets[q] as i64;
            if j >= out.len() || s > s_end {
                break;
            }
            let s = s as usize;
            out[j].write(from_f(F::dot::<FMA>(
                &coeffs[q * taps..(q + 1) * taps],
                &buf[s..s + taps],
            )));
            j += 1;
            q += 1;
            if q == l {
                q = 0;
                base += m as i64;
            }
        }
        BState {
            j,
            n0: base + offsets[q] as i64 + buf_start + self.half as i64 - 1,
            p: (q as u64 * self.m) % self.l,
        }
    }
}

impl<F: RsFloat> Interp<F> {
    /// Outputs `j .. j + c l` (`c` whole cycles from output `j`, `j` a
    /// multiple of `l`) of an exact table, the first window at `base` in
    /// `buf`: output `l - r` of a cycle uses row `r` reversed (the filter is
    /// even), so each row is read once for both outputs.
    #[inline(always)]
    fn mirrored_cycles<T, const FMA: bool>(
        &self,
        buf: &[F],
        base: usize,
        c: usize,
        out: &mut [MaybeUninit<T>],
        from_f: &impl Fn(F) -> T,
        j: usize,
    ) {
        let (l, m, taps, kt) = (
            self.l as usize,
            self.m as usize,
            self.taps,
            self.kernel_taps,
        );
        for r in 0..=l / 2 {
            let row = &self.coeffs[r * taps..(r + 1) * taps];
            let mirror = r > 0 && 2 * r != l;
            let mut k = 0;
            if !mirror {
                while k + 2 <= c {
                    let w = base + self.offsets[r] + k * m;
                    let v =
                        F::dot_rows2::<FMA>(row, [&buf[w..w + taps], &buf[w + m..w + m + taps]]);
                    out[j + k * l + r].write(from_f(v[0]));
                    out[j + (k + 1) * l + r].write(from_f(v[1]));
                    k += 2;
                }
            }
            for k in k..c {
                let w = base + self.offsets[r] + k * m;
                if mirror {
                    let wm = base + self.offsets[l - r] + k * m;
                    let v = F::dot_mirror::<FMA>(row, &buf[w..w + taps], &buf[wm..wm + kt]);
                    out[j + k * l + r].write(from_f(v[0]));
                    out[j + k * l + l - r].write(from_f(v[1]));
                } else {
                    out[j + k * l + r].write(from_f(F::dot::<FMA>(row, &buf[w..w + taps])));
                }
            }
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
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    let tail = ra.iter().zip(rb).fold(F::zero(), |s, (x, y)| s + *x * *y);
    for (x, y) in ca.iter().zip(cb) {
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

    /// [`dot_f32`] of `a` with each of `b`.
    ///
    /// # Safety
    /// As [`dot_f32`], for each pair.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn rows4_f32(a: &[f32], b: [&[f32]; 4]) -> [f32; 4] {
        let n = a.len();
        debug_assert!(n.is_multiple_of(8) && b.iter().all(|s| s.len() == n));
        let pa = a.as_ptr();
        let pb = b.map(<[f32]>::as_ptr);
        let mut acc = [_mm256_setzero_ps(); 4];
        let mut i = 0;
        while i < n {
            // SAFETY: i + 8 <= n for every slice.
            unsafe {
                let c = _mm256_loadu_ps(pa.add(i));
                for t in 0..4 {
                    acc[t] = _mm256_fmadd_ps(c, _mm256_loadu_ps(pb[t].add(i)), acc[t]);
                }
            }
            i += 8;
        }
        acc.map(|v| hsum_ps(v))
    }

    /// [`dot_f64`] of `a` with each of `b`.
    ///
    /// # Safety
    /// As [`dot_f64`], for each pair.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn rows4_f64(a: &[f64], b: [&[f64]; 4]) -> [f64; 4] {
        let n = a.len();
        debug_assert!(n.is_multiple_of(4) && b.iter().all(|s| s.len() == n));
        let pa = a.as_ptr();
        let pb = b.map(<[f64]>::as_ptr);
        let mut acc0 = [_mm256_setzero_pd(); 4];
        let mut acc1 = [_mm256_setzero_pd(); 4];
        let mut i = 0;
        while i + 8 <= n {
            // SAFETY: i + 8 <= n for every slice.
            unsafe {
                let c0 = _mm256_loadu_pd(pa.add(i));
                let c1 = _mm256_loadu_pd(pa.add(i + 4));
                for t in 0..4 {
                    acc0[t] = _mm256_fmadd_pd(c0, _mm256_loadu_pd(pb[t].add(i)), acc0[t]);
                    acc1[t] = _mm256_fmadd_pd(c1, _mm256_loadu_pd(pb[t].add(i + 4)), acc1[t]);
                }
            }
            i += 8;
        }
        if i < n {
            // SAFETY: i + 4 <= n (the lengths are multiples of 4).
            unsafe {
                let c0 = _mm256_loadu_pd(pa.add(i));
                for t in 0..4 {
                    acc0[t] = _mm256_fmadd_pd(c0, _mm256_loadu_pd(pb[t].add(i)), acc0[t]);
                }
            }
        }
        sum4_pd([0, 1, 2, 3].map(|t| _mm256_add_pd(acc0[t], acc1[t])))
    }

    /// [`dot_f32`] of `a` with each of `b`.
    ///
    /// # Safety
    /// As [`dot_f32`], for each pair.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn rows2_f32(a: &[f32], b: [&[f32]; 2]) -> [f32; 2] {
        let n = a.len();
        debug_assert!(n.is_multiple_of(8) && b.iter().all(|s| s.len() == n));
        let pa = a.as_ptr();
        let pb = b.map(<[f32]>::as_ptr);
        let mut acc = [_mm256_setzero_ps(); 2];
        let mut i = 0;
        while i < n {
            // SAFETY: i + 8 <= n for every slice.
            unsafe {
                let c = _mm256_loadu_ps(pa.add(i));
                for t in 0..2 {
                    acc[t] = _mm256_fmadd_ps(c, _mm256_loadu_ps(pb[t].add(i)), acc[t]);
                }
            }
            i += 8;
        }
        [hsum_ps(acc[0]), hsum_ps(acc[1])]
    }

    /// [`dot_f64`] of `a` with each of `b`.
    ///
    /// # Safety
    /// As [`dot_f64`], for each pair.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn rows2_f64(a: &[f64], b: [&[f64]; 2]) -> [f64; 2] {
        let n = a.len();
        debug_assert!(n.is_multiple_of(4) && b.iter().all(|s| s.len() == n));
        let pa = a.as_ptr();
        let pb = b.map(<[f64]>::as_ptr);
        let mut acc0 = [_mm256_setzero_pd(); 2];
        let mut acc1 = [_mm256_setzero_pd(); 2];
        let mut i = 0;
        while i + 8 <= n {
            // SAFETY: i + 8 <= n for every slice.
            unsafe {
                let c0 = _mm256_loadu_pd(pa.add(i));
                let c1 = _mm256_loadu_pd(pa.add(i + 4));
                for t in 0..2 {
                    acc0[t] = _mm256_fmadd_pd(c0, _mm256_loadu_pd(pb[t].add(i)), acc0[t]);
                    acc1[t] = _mm256_fmadd_pd(c1, _mm256_loadu_pd(pb[t].add(i + 4)), acc1[t]);
                }
            }
            i += 8;
        }
        if i < n {
            // SAFETY: i + 4 <= n (the lengths are multiples of 4).
            unsafe {
                let c0 = _mm256_loadu_pd(pa.add(i));
                for t in 0..2 {
                    acc0[t] = _mm256_fmadd_pd(c0, _mm256_loadu_pd(pb[t].add(i)), acc0[t]);
                }
            }
        }
        [
            hsum_pd(_mm256_add_pd(acc0[0], acc1[0])),
            hsum_pd(_mm256_add_pd(acc0[1], acc1[1])),
        ]
    }

    /// `super::RsFloat::dot_mirror` for `f32`.
    ///
    /// # Safety
    /// CPU with AVX2 + FMA; `a` and `f` at least `r.len()` long.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn mirror_f32(a: &[f32], f: &[f32], r: &[f32]) -> [f32; 2] {
        let n = r.len();
        assert!(a.len() >= n && f.len() >= n);
        let (pa, pf, pr) = (a.as_ptr(), f.as_ptr(), r.as_ptr());
        let (mut af, mut ar) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let mut i = 0;
        while i + 8 <= n {
            // SAFETY: i + 8 <= n; the reversed chunk is r[n - i - 8 .. n - i].
            unsafe {
                let c = _mm256_loadu_ps(pa.add(i));
                af = _mm256_fmadd_ps(c, _mm256_loadu_ps(pf.add(i)), af);
                let q = pr.add(n - i - 8);
                // Halves swapped, then each reversed (no lane-crossing shuffle).
                let v = _mm256_permute_ps::<0b00_01_10_11>(_mm256_loadu2_m128(q, q.add(4)));
                ar = _mm256_fmadd_ps(c, v, ar);
            }
            i += 8;
        }
        let (mut sf, mut sr) = (hsum_ps(af), hsum_ps(ar));
        for i in i..n {
            sf += a[i] * f[i];
            sr += a[i] * r[n - 1 - i];
        }
        [sf, sr]
    }

    /// `super::RsFloat::dot_mirror` for `f64`.
    ///
    /// # Safety
    /// CPU with AVX2 + FMA; `a` and `f` at least `r.len()` long.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn mirror_f64(a: &[f64], f: &[f64], r: &[f64]) -> [f64; 2] {
        let n = r.len();
        assert!(a.len() >= n && f.len() >= n);
        let (pa, pf, pr) = (a.as_ptr(), f.as_ptr(), r.as_ptr());
        let (mut af, mut ar) = (_mm256_setzero_pd(), _mm256_setzero_pd());
        let mut i = 0;
        while i + 4 <= n {
            // SAFETY: i + 4 <= n; the reversed chunk is r[n - i - 4 .. n - i].
            unsafe {
                let c = _mm256_loadu_pd(pa.add(i));
                af = _mm256_fmadd_pd(c, _mm256_loadu_pd(pf.add(i)), af);
                let q = pr.add(n - i - 4);
                // Halves swapped, then each reversed (no lane-crossing shuffle).
                let v = _mm256_permute_pd::<0b0101>(_mm256_loadu2_m128d(q, q.add(2)));
                ar = _mm256_fmadd_pd(c, v, ar);
            }
            i += 4;
        }
        let (mut sf, mut sr) = (hsum_pd(af), hsum_pd(ar));
        for i in i..n {
            sf += a[i] * f[i];
            sr += a[i] * r[n - 1 - i];
        }
        [sf, sr]
    }

    /// Four [`dot_f32`] products.
    ///
    /// # Safety
    /// As [`dot_f32`], for each pair.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot4_f32(a: [&[f32]; 4], b: [&[f32]; 4]) -> [f32; 4] {
        // SAFETY: forwarded.
        [0, 1, 2, 3].map(|t| unsafe { dot_f32(a[t], b[t]) })
    }

    /// Four [`dot_f64`] products.
    ///
    /// # Safety
    /// As [`dot_f64`], for each pair.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot4_f64(a: [&[f64]; 4], b: [&[f64]; 4]) -> [f64; 4] {
        let n = a[0].len();
        debug_assert!(n.is_multiple_of(4) && a.iter().chain(&b).all(|s| s.len() == n));
        let pa = a.map(<[f64]>::as_ptr);
        let pb = b.map(<[f64]>::as_ptr);
        // Two accumulators per product (even and odd groups of four), as in
        // `dot_f64`, so that the sums are the same.
        let mut acc0 = [_mm256_setzero_pd(); 4];
        let mut acc1 = [_mm256_setzero_pd(); 4];
        let mut i = 0;
        while i + 8 <= n {
            for t in 0..4 {
                // SAFETY: i + 8 <= n for every slice.
                unsafe {
                    acc0[t] = _mm256_fmadd_pd(
                        _mm256_loadu_pd(pa[t].add(i)),
                        _mm256_loadu_pd(pb[t].add(i)),
                        acc0[t],
                    );
                    acc1[t] = _mm256_fmadd_pd(
                        _mm256_loadu_pd(pa[t].add(i + 4)),
                        _mm256_loadu_pd(pb[t].add(i + 4)),
                        acc1[t],
                    );
                }
            }
            i += 8;
        }
        if i < n {
            for t in 0..4 {
                // SAFETY: i + 4 <= n (the lengths are multiples of 4).
                unsafe {
                    acc0[t] = _mm256_fmadd_pd(
                        _mm256_loadu_pd(pa[t].add(i)),
                        _mm256_loadu_pd(pb[t].add(i)),
                        acc0[t],
                    );
                }
            }
        }
        sum4_pd([0, 1, 2, 3].map(|t| _mm256_add_pd(acc0[t], acc1[t])))
    }

    /// The lane sums of four vectors, each as [`hsum_pd`] adds them.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn sum4_pd(v: [__m256d; 4]) -> [f64; 4] {
        // Lanes (0 + 2) + (1 + 3) of each, as `hsum_pd` adds them.
        let half = |x: __m256d, y: __m256d| {
            _mm256_add_pd(
                _mm256_permute2f128_pd::<0x20>(x, y),
                _mm256_permute2f128_pd::<0x31>(x, y),
            )
        };
        // (s0, s2, s1, s3)
        let s = _mm256_hadd_pd(half(v[0], v[1]), half(v[2], v[3]));
        let mut r = [0.0; 4];
        // SAFETY: `r` holds four f64.
        unsafe { _mm256_storeu_pd(r.as_mut_ptr(), _mm256_permute4x64_pd::<0b11_01_10_00>(s)) };
        r
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

    /// `super::RsFloat::down2_bins` for `f64`, two bins per vector.
    ///
    /// # Safety
    /// The CPU must support AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn down2_f64<const TWO: bool>(
        w: &mut [Complex<f64>],
        za: &[Complex<f64>],
        zb: &[Complex<f64>],
        t: &super::Bins<f64>,
        lay: super::Layout,
    ) {
        let n = lay.len();
        assert!(w.len() == n && za.len() == n && zb.len() == n);
        assert!(t.pr.len() == n && t.pi.len() == n && t.qr.len() == n && t.qi.len() == n);
        for k1 in 0..lay.r {
            // SAFETY: as for this function.
            unsafe { down2_row_f64::<TWO>(w, za, zb, t, lay, k1) };
        }
    }

    /// [`down2_f64`] for row `k1` of `lay` (see
    /// `super::down2_bins_portable` for the mirror positions).
    ///
    /// # Safety
    /// The CPU must support AVX2 and FMA; all slices have `lay.len()` elements.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn down2_row_f64<const TWO: bool>(
        w: &mut [Complex<f64>],
        za: &[Complex<f64>],
        zb: &[Complex<f64>],
        t: &super::Bins<f64>,
        lay: super::Layout,
        k1: usize,
    ) {
        let super::Layout { r, m, stride } = lay;
        let (row, mrow) = (k1 * stride, (r - k1) % r * stride);
        // Position k's mirror is at mrow + c - k (from position `first` on).
        let (first, c) = if k1 == 0 { (1, m) } else { (0, m - 1) };
        super::down2_bins_portable::<_, TWO>(w, za, zb, t, lay, k1, 0, first.min(m));
        let (pw, pa, pb) = (
            w.as_mut_ptr().cast::<f64>(),
            za.as_ptr().cast::<f64>(),
            zb.as_ptr().cast::<f64>(),
        );
        let tab = [&t.pr, &t.pi, &t.qr, &t.qi].map(|v| v.as_ptr().cast::<f64>());
        // Imaginary parts negated: conj.
        let conj = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
        let mut k = first;
        while k + 2 <= m {
            let (i, ir) = (row + k, mrow + c - k);
            // SAFETY: positions k, k + 1 of the row and their mirrors
            // ir - 1, ir are in bounds (k + 2 <= m, ir - 1 >= mrow).
            unsafe {
                let a = _mm256_loadu_pd(pa.add(2 * i));
                let b = if TWO {
                    _mm256_loadu_pd(pb.add(2 * i))
                } else {
                    _mm256_setzero_pd()
                };
                // Mirrors of k, k + 1 as two 128-bit loads (no lane-crossing
                // shuffle).
                let ar = _mm256_loadu2_m128d(pa.add(2 * (ir - 1)), pa.add(2 * ir));
                let br = if TWO {
                    _mm256_loadu2_m128d(pb.add(2 * (ir - 1)), pb.add(2 * ir))
                } else {
                    _mm256_setzero_pd()
                };
                // A = a + i b = (a.re - b.im, a.im + b.re).
                let av = if TWO {
                    _mm256_addsub_pd(a, _mm256_permute_pd::<0b0101>(b))
                } else {
                    a
                };
                // B = conj(ar) + i conj(br) = (ar.re + br.im, br.re - ar.im).
                let bv = if TWO {
                    _mm256_add_pd(_mm256_xor_pd(ar, conj), _mm256_permute_pd::<0b0101>(br))
                } else {
                    _mm256_xor_pd(ar, conj)
                };
                let at = |j: usize| _mm256_loadu_pd(tab[j].add(2 * i));
                let ra = _mm256_fmadd_pd(
                    _mm256_permute_pd::<0b0101>(av),
                    at(1),
                    _mm256_mul_pd(av, at(0)),
                );
                let rb = _mm256_fmadd_pd(
                    _mm256_permute_pd::<0b0101>(bv),
                    at(3),
                    _mm256_mul_pd(bv, at(2)),
                );
                _mm256_storeu_pd(pw.add(2 * i), _mm256_add_pd(ra, rb));
            }
            k += 2;
        }
        super::down2_bins_portable::<_, TWO>(w, za, zb, t, lay, k1, k, m);
    }

    /// `super::RsFloat::down2_bins` for `f32`, four bins per vector.
    ///
    /// # Safety
    /// The CPU must support AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn down2_f32<const TWO: bool>(
        w: &mut [Complex<f32>],
        za: &[Complex<f32>],
        zb: &[Complex<f32>],
        t: &super::Bins<f32>,
        lay: super::Layout,
    ) {
        let n = lay.len();
        assert!(w.len() == n && za.len() == n && zb.len() == n);
        assert!(t.pr.len() == n && t.pi.len() == n && t.qr.len() == n && t.qi.len() == n);
        for k1 in 0..lay.r {
            // SAFETY: as for this function.
            unsafe { down2_row_f32::<TWO>(w, za, zb, t, lay, k1) };
        }
    }

    /// [`down2_f32`] for row `k1` of `lay`.
    ///
    /// # Safety
    /// The CPU must support AVX2 and FMA; all slices have `lay.len()` elements.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn down2_row_f32<const TWO: bool>(
        w: &mut [Complex<f32>],
        za: &[Complex<f32>],
        zb: &[Complex<f32>],
        t: &super::Bins<f32>,
        lay: super::Layout,
        k1: usize,
    ) {
        let super::Layout { r, m, stride } = lay;
        let (row, mrow) = (k1 * stride, (r - k1) % r * stride);
        let (first, c) = if k1 == 0 { (1, m) } else { (0, m - 1) };
        super::down2_bins_portable::<_, TWO>(w, za, zb, t, lay, k1, 0, first.min(m));
        let (pw, pa, pb) = (
            w.as_mut_ptr().cast::<f32>(),
            za.as_ptr().cast::<f32>(),
            zb.as_ptr().cast::<f32>(),
        );
        let tab = [&t.pr, &t.pi, &t.qr, &t.qi].map(|v| v.as_ptr().cast::<f32>());
        let conj = _mm256_set_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
        // Bins j, j - 1, j - 2, j - 3: two 128-bit loads, each pair swapped
        // within its lane (no lane-crossing shuffle).
        let mirror = |p: *const f32| {
            // SAFETY: the caller passes bins j - 3 ..= j in bounds.
            let v = unsafe { _mm256_loadu2_m128(p.sub(6), p.sub(2)) };
            _mm256_permute_ps::<0b0100_1110>(v)
        };
        let mut k = first;
        while k + 4 <= m {
            let (i, ir) = (row + k, mrow + c - k);
            // SAFETY: positions k .. k + 4 of the row and their mirrors
            // ir - 3 ..= ir are in bounds (k + 4 <= m, ir - 3 >= mrow).
            unsafe {
                let a = _mm256_loadu_ps(pa.add(2 * i));
                let b = if TWO {
                    _mm256_loadu_ps(pb.add(2 * i))
                } else {
                    _mm256_setzero_ps()
                };
                let ar = mirror(pa.add(2 * ir));
                let br = if TWO {
                    mirror(pb.add(2 * ir))
                } else {
                    _mm256_setzero_ps()
                };
                let av = if TWO {
                    _mm256_addsub_ps(a, _mm256_permute_ps::<0b1011_0001>(b))
                } else {
                    a
                };
                let bv = if TWO {
                    _mm256_add_ps(
                        _mm256_xor_ps(ar, conj),
                        _mm256_permute_ps::<0b1011_0001>(br),
                    )
                } else {
                    _mm256_xor_ps(ar, conj)
                };
                let at = |j: usize| _mm256_loadu_ps(tab[j].add(2 * i));
                let ra = _mm256_fmadd_ps(
                    _mm256_permute_ps::<0b1011_0001>(av),
                    at(1),
                    _mm256_mul_ps(av, at(0)),
                );
                let rb = _mm256_fmadd_ps(
                    _mm256_permute_ps::<0b1011_0001>(bv),
                    at(3),
                    _mm256_mul_ps(bv, at(2)),
                );
                _mm256_storeu_ps(pw.add(2 * i), _mm256_add_ps(ra, rb));
            }
            k += 4;
        }
        super::down2_bins_portable::<_, TWO>(w, za, zb, t, lay, k1, k, m);
    }

    /// `super::RsFloat::dft_bins` for `f64`, two bins per vector.
    ///
    /// # Safety
    /// The CPU must support AVX2 and FMA; `tab` holds the groups of bins
    /// `1..w.len()` and `w.len() < z.len()`.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dft_f64(
        w: &mut [Complex<f64>],
        z: &[Complex<f64>],
        tab0: &[Complex<f64>; 8],
        tab: &[Complex<f64>],
    ) {
        let (big_m, m2) = (z.len(), w.len());
        assert!(m2 < big_m && tab.len() >= (m2.max(1) - 1).div_ceil(2) * 16);
        super::dft_bins_portable(w, z, tab0, tab, 0, 1.min(m2));
        let (pw, pz, pt) = (
            w.as_mut_ptr().cast::<f64>(),
            z.as_ptr().cast::<f64>(),
            tab.as_ptr().cast::<f64>(),
        );
        let swap = |v: __m256d| _mm256_permute_pd::<0b0101>(v);
        // Bins j, j - 1 (two 128-bit loads: no lane-crossing shuffle).
        let mirror = |p: *const f64| {
            // SAFETY: the caller passes bins j - 1 and j in bounds.
            unsafe { _mm256_loadu2_m128d(p.sub(2), p) }
        };
        let mut k = 1;
        while k + 2 <= m2 {
            // SAFETY: bins k, k + 1 < m2 < big_m, their mirrors big_m - k - 1 ..=
            // big_m - k and m2 - k - 1 ..= m2 - k, and big_m - m2 + k + 1 < big_m
            // are in bounds of `z`; group (k - 1) / 2 of `tab` is (asserted).
            unsafe {
                let t = pt.add((k - 1) / 2 * 32);
                let at = |r: usize| _mm256_loadu_pd(t.add(4 * r));
                let v = [
                    _mm256_loadu_pd(pz.add(2 * k)),
                    mirror(pz.add(2 * (big_m - k))),
                    mirror(pz.add(2 * (m2 - k))),
                    _mm256_loadu_pd(pz.add(2 * (big_m - m2 + k))),
                ];
                // Four independent products, then a tree of sums.
                let r = [0, 1, 2, 3].map(|j| {
                    _mm256_fmadd_pd(swap(v[j]), at(2 * j + 1), _mm256_mul_pd(v[j], at(2 * j)))
                });
                let r = _mm256_add_pd(_mm256_add_pd(r[0], r[1]), _mm256_add_pd(r[2], r[3]));
                _mm256_storeu_pd(pw.add(2 * k), r);
            }
            k += 2;
        }
        super::dft_bins_portable(w, z, tab0, tab, k, m2);
    }

    /// `super::RsFloat::dft_bins` for `f32`, four bins per vector.
    ///
    /// # Safety
    /// As [`dft_f64`].
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dft_f32(
        w: &mut [Complex<f32>],
        z: &[Complex<f32>],
        tab0: &[Complex<f32>; 8],
        tab: &[Complex<f32>],
    ) {
        let (big_m, m2) = (z.len(), w.len());
        assert!(m2 < big_m && tab.len() >= (m2.max(1) - 1).div_ceil(4) * 32);
        super::dft_bins_portable(w, z, tab0, tab, 0, 1.min(m2));
        let (pw, pz, pt) = (
            w.as_mut_ptr().cast::<f32>(),
            z.as_ptr().cast::<f32>(),
            tab.as_ptr().cast::<f32>(),
        );
        let swap = |v: __m256| _mm256_permute_ps::<0b1011_0001>(v);
        // Bins j, j - 1, j - 2, j - 3: two 128-bit loads, each pair swapped
        // within its lane (no lane-crossing shuffle).
        let mirror = |p: *const f32| {
            // SAFETY: the caller passes bins j - 3 ..= j in bounds.
            let v = unsafe { _mm256_loadu2_m128(p.sub(6), p.sub(2)) };
            _mm256_permute_ps::<0b0100_1110>(v)
        };
        let mut k = 1;
        while k + 4 <= m2 {
            // SAFETY: as in `dft_f64`, for bins k .. k + 4.
            unsafe {
                let t = pt.add((k - 1) / 4 * 64);
                let at = |r: usize| _mm256_loadu_ps(t.add(8 * r));
                let v = [
                    _mm256_loadu_ps(pz.add(2 * k)),
                    mirror(pz.add(2 * (big_m - k))),
                    mirror(pz.add(2 * (m2 - k))),
                    _mm256_loadu_ps(pz.add(2 * (big_m - m2 + k))),
                ];
                let r = [0, 1, 2, 3].map(|j| {
                    _mm256_fmadd_ps(swap(v[j]), at(2 * j + 1), _mm256_mul_ps(v[j], at(2 * j)))
                });
                let r = _mm256_add_ps(_mm256_add_ps(r[0], r[1]), _mm256_add_ps(r[2], r[3]));
                _mm256_storeu_ps(pw.add(2 * k), r);
            }
            k += 4;
        }
        super::dft_bins_portable(w, z, tab0, tab, k, m2);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::types::ResamplingQuality;

    /// The block-wise plan's portable bins give the AVX2 result up to
    /// rounding, in both float types.
    #[test]
    fn block_plan_portable_matches_avx2() {
        if !super::super::has_avx2_fma_cached() {
            return;
        }
        let plan = super::super::plan(22050, 8000, ResamplingQuality::High).unwrap();
        assert!(matches!(plan, Plan::Dft(_)));
        let x: Vec<f64> = (0..30011)
            .map(|i| (f64::from(i) * 0.07).sin() * 0.8 + (f64::from(i) * 0.9).cos() * 0.1)
            .collect();
        let n = 30011 * 8000 / 22050;
        fn run<F: RsFloat>(plan: &Plan, x: &[f64], n: usize, simd: bool) -> Vec<f64> {
            let mut e = Engine::<F>::new(plan).unwrap();
            e.avx2_fma = simd;
            let mut out = vec![MaybeUninit::<f64>::uninit(); n];
            e.run(x, &mut out, &|v| F::of(v), &|v: F| v.as_f64());
            // SAFETY: `run` writes every element.
            out.into_iter()
                .map(|v| unsafe { v.assume_init() })
                .collect()
        }
        for (a, b, tol) in [
            (
                run::<f64>(&plan, &x, n, true),
                run::<f64>(&plan, &x, n, false),
                1e-13,
            ),
            (
                run::<f32>(&plan, &x, n, true),
                run::<f32>(&plan, &x, n, false),
                1e-5,
            ),
        ] {
            assert!(a.iter().any(|v| v.abs() > 0.5));
            let err = a
                .iter()
                .zip(&b)
                .fold(0.0f64, |m, (u, v)| m.max((u - v).abs()));
            assert!(err < tol, "{err:e}");
        }
    }

    /// The portable body (CPUs without AVX2) gives the AVX2 result up to rounding,
    /// for integer and float input.
    #[test]
    fn portable_matches_avx2() {
        if !super::super::has_avx2_fma_cached() {
            return;
        }
        let plan = super::super::plan(16000, 22050, ResamplingQuality::High).unwrap();
        let fast = Engine::<f32>::new(&plan).unwrap();
        let mut slow = Engine::<f32>::new(&plan).unwrap();
        slow.avx2_fma = false;
        let x: Vec<i16> = (0..5000)
            .map(|i| ((f64::from(i) * 0.07).sin() * 20000.0) as i16)
            .collect();
        let n = 5000 * 22050 / 16000;
        let to_f = |s: i16| f32::from(s) / 32768.0;
        let from_f = |v: f32| (v * 32768.0).round() as i16;
        let run = |e: &Engine<f32>| {
            let mut out = vec![MaybeUninit::<i16>::uninit(); n];
            e.run(&x, &mut out, &to_f, &from_f);
            // SAFETY: `run` writes every element.
            out.into_iter()
                .map(|v| i32::from(unsafe { v.assume_init() }))
                .collect::<Vec<_>>()
        };
        let (a, b) = (run(&fast), run(&slow));
        assert!(a.iter().any(|v| v.abs() > 10000));
        for (u, v) in a.iter().zip(&b) {
            assert!((u - v).abs() <= 1, "{u} vs {v}");
        }
    }
}
