//! Sample-rate conversion for [`AudioSamples`].
//!
//! This module converts audio between sample rates, either all at once
//! ([`resample`], [`resample_by_ratio`]) or chunk by chunk for live sources
//! ([`StreamingResampler`]).
//!
//! Sample-rate conversion is needed whenever audio from different sources is
//! combined, or when an analysis expects a fixed rate (e.g. 16 kHz speech
//! models, librosa's 22.05 kHz default). Done naively it either aliases
//! (dropping samples) or shifts the signal in time (a filter delay that is not
//! compensated); both corrupt any later sample-accurate comparison.
//!
//! All quality tiers except `Fast` upsampling use a Kaiser-windowed sinc
//! interpolator evaluated at the exact rational position of every output
//! sample: output `k` is the band-limited reconstruction of the input at time
//! `k · in_rate / out_rate`. The filter is symmetric (linear phase) and centred
//! on that position, so the output is time-aligned with the input (no delay to
//! trim), input before the start and after the end is treated as zeros (so the
//! tail is flushed), and the output always has exactly
//! `ceil(n · out_rate / in_rate)` samples. The filter cut-off is placed just
//! below the Nyquist frequency of the lower of the two rates, so downsampling
//! is anti-aliased and upsampling is anti-imaged. Processing is done in `f64`.
//!
//! | Quality | Passband edge | Stopband | Use |
//! |---|---|---|---|
//! | `Fast` (downsampling) | 0.80 · Nyquist | 70 dB from Nyquist | cheap anti-aliased decimation |
//! | `Fast` (upsampling) | linear interpolation | none | cheapest; images are not suppressed |
//! | `Medium` | 0.90 · Nyquist | 100 dB from Nyquist | general use |
//! | `High` | 0.913 · Nyquist | 125 dB from Nyquist | analysis; close to `librosa.resample` (`soxr_hq`) |
//!
//! (Nyquist of the lower of the two rates.) Where the ratio
//! `out_rate / in_rate = L / M` (in lowest terms) has a small enough `L`, one
//! exact filter phase is precomputed per output phase; otherwise the filter is
//! tabulated at 4096 phases per sample (proportionally fewer when
//! downsampling, where the filter is wider) and linearly interpolated (error
//! below -120 dB). Filter tables are cached per thread (a few recent ratios,
//! at most 64 MiB). Ratios needing a table of more than 2^25 coefficients
//! (decimation by more than ~10^5 : 1) are rejected with an error.
//!
//! # Differences from librosa
//!
//! `librosa.resample` (default `res_type="soxr_hq"`) delegates to the soxr C
//! library, a different filter design; `High` follows soxr-hq's passband
//! (91.3 % of Nyquist) and attenuation, so the two agree closely on band-limited
//! content (see `tests/librosa_reference/resample.rs` for measured agreement)
//! but are not bit-identical. librosa computes the output length as
//! `ceil(n * target_sr / orig_sr)` in floating point, which can exceed the exact
//! value by one sample (e.g. 96001 instead of 96000 for 88200 samples from
//! 44100 Hz to 48000 Hz); this module uses exact integer arithmetic. librosa
//! accepts fractional sample rates and a `scale` option; this module does not.
//!
//! ```
//! use audio_samples::{AudioSamples, resample, sample_rate};
//! use audio_samples::operations::types::ResamplingQuality;
//! use ndarray::Array1;
//!
//! let x = Array1::from_iter((0..44100).map(|i| (i as f64 * 0.01).sin()));
//! let audio = AudioSamples::new_mono(x, sample_rate!(44100)).unwrap();
//! let y = resample(&audio, sample_rate!(16000), ResamplingQuality::High).unwrap();
//! assert_eq!(y.samples_per_channel().get(), 16000);
//! ```

use crate::{
    AudioSampleError, AudioSampleResult, AudioSamples, AudioTypeConversion, ConvertTo,
    ParameterError,
    operations::types::ResamplingQuality,
    repr::{ChannelCount, SampleRate},
    traits::StandardSample,
};
use non_empty_slice::NonEmptyVec;
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::cell::RefCell;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Windowed-sinc interpolation kernel
// ---------------------------------------------------------------------------

/// Largest `L` (output phases per input period) for which every phase gets its
/// own exact filter row.
const MAX_EXACT_PHASES: u64 = 4096;
/// Upper bound on the size of an exact polyphase table (entries).
const MAX_TABLE_ENTRIES: u64 = 1 << 22;
/// Phase resolution (rows per input sample) of the interpolated table used for
/// awkward ratios when upsampling; downsampling by `lower = L / M` uses
/// `ceil(INTERP_PHASES · lower)` rows, which gives the same interpolation
/// error because the filter is `1 / lower` times wider.
const INTERP_PHASES: usize = 4096;
/// Hard limit on the coefficients of any filter table (256 MiB). Only reached
/// for decimation by more than ~10^5 : 1, which is rejected with an error
/// instead of attempting the allocation.
const MAX_KERNEL_ENTRIES: u64 = 1 << 25;

/// Low-pass specification, relative to the Nyquist frequency of the lower of
/// the two sample rates.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FilterSpec {
    /// End of the passband (fraction of Nyquist).
    passband: f64,
    /// Start of the stopband (fraction of Nyquist).
    stopband: f64,
    /// Stopband attenuation in dB (sets the Kaiser window's beta and length).
    attenuation_db: f64,
}

/// What kind of interpolation kernel a quality tier uses for a given ratio.
#[derive(Debug, Clone, Copy, PartialEq)]
enum KernelKind {
    /// Kaiser-windowed sinc.
    Sinc(FilterSpec),
    /// Two-tap linear interpolation (`Fast` upsampling). The end of the signal
    /// is extended by holding the last sample rather than with zeros.
    Linear,
}

fn kernel_kind(quality: ResamplingQuality, in_sr: u32, out_sr: u32) -> KernelKind {
    match quality {
        ResamplingQuality::Fast if out_sr >= in_sr => KernelKind::Linear,
        ResamplingQuality::Fast => KernelKind::Sinc(FilterSpec {
            passband: 0.80,
            stopband: 1.0,
            attenuation_db: 70.0,
        }),
        ResamplingQuality::Medium => KernelKind::Sinc(FilterSpec {
            passband: 0.90,
            stopband: 1.0,
            attenuation_db: 100.0,
        }),
        ResamplingQuality::High => KernelKind::Sinc(FilterSpec {
            passband: 0.913,
            stopband: 1.0,
            attenuation_db: 125.0,
        }),
    }
}

/// Zeroth-order modified Bessel function of the first kind (power series).
fn bessel_i0(x: f64) -> f64 {
    let q = x * x / 4.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut k = 1.0;
    while term > sum * 1e-17 {
        term *= q / (k * k);
        sum += term;
        k += 1.0;
    }
    sum
}

/// Kaiser's empirical beta for a given stopband attenuation (dB).
fn kaiser_beta(attenuation_db: f64) -> f64 {
    if attenuation_db > 50.0 {
        0.1102 * (attenuation_db - 8.7)
    } else if attenuation_db >= 21.0 {
        0.5842 * (attenuation_db - 21.0).powf(0.4) + 0.07886 * (attenuation_db - 21.0)
    } else {
        0.0
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Polyphase interpolation table for one `(in_rate, out_rate, quality)`.
///
/// Output sample `k` sits at input position `t = k · M / L`. With
/// `n0 = floor(t)` and phase `p = (k · M) mod L`, it is the dot product of the
/// `taps` input samples `x[n0 - half + 1 ..= n0 + half]` with the row for `p`,
/// where row entry `j` is `h(p / L - (j - half + 1))`.
#[derive(Debug)]
struct Kernel {
    /// Output samples per period (`out_rate / gcd`).
    l: u64,
    /// Input samples per period (`in_rate / gcd`).
    m: u64,
    /// Taps on each side of the interpolation point.
    half: usize,
    /// Row length (`2 · half`).
    taps: usize,
    /// `true`: one row per phase `p`; `false`: `phases + 1` rows over
    /// `[0, 1]`, linearly interpolated.
    exact: bool,
    /// Rows per input sample of the interpolated table (unused when `exact`).
    phases: usize,
    /// Row-major coefficients.
    coeffs: Vec<f64>,
    /// Whether the signal end is extended by holding the last sample.
    hold_last: bool,
    /// Use the AVX2 + FMA dot product.
    avx2_fma: bool,
}

impl Kernel {
    /// Builds the table for `in_sr -> out_sr`.
    ///
    /// # Errors
    /// [`ParameterError`] if the table would exceed [`MAX_KERNEL_ENTRIES`]
    /// coefficients (decimation by more than ~10^5 : 1).
    fn new(in_sr: u32, out_sr: u32, kind: KernelKind) -> AudioSampleResult<Self> {
        let g = gcd(u64::from(in_sr), u64::from(out_sr));
        let l = u64::from(out_sr) / g;
        let m = u64::from(in_sr) / g;
        // Bandwidth of the lower rate relative to the input rate.
        let lower = (l as f64 / m as f64).min(1.0);

        // Continuous kernel h(tau), tau in input samples, and its support.
        let (half, h): (usize, Box<dyn Fn(f64) -> f64>) = match kind {
            KernelKind::Linear => (1, Box::new(|tau: f64| (1.0 - tau.abs()).max(0.0))),
            KernelKind::Sinc(spec) => {
                // -6 dB point midway through the transition band, as a
                // fraction of the input Nyquist frequency.
                let cutoff = 0.5 * (spec.passband + spec.stopband) * lower;
                // Kaiser length estimate (in lower-rate samples) for the
                // requested attenuation and transition width.
                let transition = std::f64::consts::PI * (spec.stopband - spec.passband);
                let length = (spec.attenuation_db - 7.95) / (2.285 * transition);
                let half_lower = (length / 2.0).ceil();
                let support = half_lower / lower;
                let beta = kaiser_beta(spec.attenuation_db);
                let i0_beta = bessel_i0(beta);
                let h = move |tau: f64| -> f64 {
                    let r = tau / support;
                    if r.abs() >= 1.0 {
                        return 0.0;
                    }
                    let window = bessel_i0(beta * (1.0 - r * r).sqrt()) / i0_beta;
                    let x = std::f64::consts::PI * cutoff * tau;
                    let sinc = if x == 0.0 { 1.0 } else { x.sin() / x };
                    cutoff * sinc * window
                };
                // Round the half-width up to a multiple of 8 so rows are a
                // whole number of 16-wide SIMD blocks (the extra taps are 0).
                // `support` is at most ~4e11 (u32 rates), so this cannot wrap
                // on 64-bit targets; the size check below rejects it anyway.
                ((support.ceil() as usize).next_multiple_of(8), Box::new(h))
            }
        };
        let taps = 2 * half;
        // The interpolated table needs fewer rows per input sample when the
        // filter is wider (downsampling): its error depends on the row
        // spacing relative to the filter's bandwidth.
        let phases = ((INTERP_PHASES as f64 * lower).ceil() as usize).clamp(1, INTERP_PHASES);
        let exact = l <= MAX_EXACT_PHASES && l.saturating_mul(taps as u64) <= MAX_TABLE_ENTRIES;
        let (rows, denom) = if exact {
            (l as usize, l as f64)
        } else {
            (phases + 1, phases as f64)
        };
        let entries = (rows as u64).saturating_mul(taps as u64);
        if entries > MAX_KERNEL_ENTRIES {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "target_sample_rate",
                format!(
                    "resampling {in_sr} Hz -> {out_sr} Hz needs a {taps}-tap filter table of \
                     {entries} coefficients (limit {MAX_KERNEL_ENTRIES}); decimate in stages"
                ),
            )));
        }
        let mut coeffs = Vec::with_capacity(rows * taps);
        for row in 0..rows {
            let phi = row as f64 / denom;
            for j in 0..taps {
                let offset = j as f64 - (half as f64 - 1.0);
                coeffs.push(h(phi - offset));
            }
        }
        Ok(Self {
            l,
            m,
            half,
            taps,
            exact,
            phases,
            coeffs,
            hold_last: matches!(kind, KernelKind::Linear),
            avx2_fma: has_avx2_fma(),
        })
    }

    /// Integer part and phase of output `k`'s input position.
    #[inline]
    fn position(&self, k: u64) -> (u64, u64) {
        let num = u128::from(k) * u128::from(self.m);
        let l = u128::from(self.l);
        ((num / l) as u64, (num % l) as u64)
    }

    /// Number of output samples for `n` input samples: `ceil(n · L / M)`.
    #[inline]
    fn output_len(&self, n: u64) -> u64 {
        let num = u128::from(n) * u128::from(self.l);
        num.div_ceil(u128::from(self.m)) as u64
    }

    /// Interpolates one output sample from `window` (`taps` input samples
    /// starting at `n0 - half + 1`) at phase `p`.
    #[inline]
    fn eval(&self, window: &[f64], p: u64) -> f64 {
        let taps = self.taps;
        if self.exact {
            let row = &self.coeffs[p as usize * taps..(p as usize + 1) * taps];
            dot(row, window, self.avx2_fma)
        } else {
            let pos = p as f64 / self.l as f64 * self.phases as f64;
            let i = (pos.floor() as usize).min(self.phases - 1);
            let frac = pos - i as f64;
            let r0 = &self.coeffs[i * taps..(i + 1) * taps];
            let r1 = &self.coeffs[(i + 1) * taps..(i + 2) * taps];
            (1.0 - frac) * dot(r0, window, self.avx2_fma) + frac * dot(r1, window, self.avx2_fma)
        }
    }
}

/// Dot product with eight independent accumulators (vectorises well).
#[inline(always)]
fn dot_portable(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0f64; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut sum = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (x, y) in ra.iter().zip(rb) {
        sum += x * y;
    }
    sum
}

/// [`dot_portable`] compiled for AVX2 + FMA (fused multiply-adds, four `f64`
/// lanes per register).
///
/// # Safety
/// The CPU must support `avx2` and `fma` (checked once in [`Kernel::new`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_fma(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0f64; 16];
    let ca = a.chunks_exact(16);
    let cb = b.chunks_exact(16);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..16 {
            acc[i] = x[i].mul_add(y[i], acc[i]);
        }
    }
    let mut sum = 0.0;
    for v in acc {
        sum += v;
    }
    for (x, y) in ra.iter().zip(rb) {
        sum = x.mul_add(*y, sum);
    }
    sum
}

/// Whether the AVX2 + FMA dot product can be used on this CPU.
fn has_avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64], avx2_fma: bool) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if avx2_fma {
        // SAFETY: `avx2_fma` is only true when the CPU supports AVX2 and FMA.
        return unsafe { dot_avx2_fma(a, b) };
    }
    let _ = avx2_fma;
    dot_portable(a, b)
}

#[derive(Hash, Eq, PartialEq, Clone, Copy)]
struct KernelKey {
    in_sr: u32,
    out_sr: u32,
    quality: ResamplingQuality,
}

/// Most kernels kept per thread.
const KERNEL_CACHE_ENTRIES: usize = 8;
/// Most coefficient memory kept per thread. A kernel larger than this on its
/// own is used for the call that built it and then dropped, never cached.
const KERNEL_CACHE_BYTES: usize = 64 << 20;

thread_local! {
    /// Recently used kernels, most recent first. Small and bounded: every
    /// distinct ratio (e.g. pitch shifts by arbitrary steps) builds a new
    /// table of up to tens of MB, so an unbounded map grows without limit.
    static KERNEL_CACHE: RefCell<Vec<(KernelKey, Arc<Kernel>)>> = const { RefCell::new(Vec::new()) };
}

impl Kernel {
    /// Heap size of the coefficient table in bytes.
    fn size_bytes(&self) -> usize {
        self.coeffs.len() * std::mem::size_of::<f64>()
    }
}

/// Returns the (per-thread cached) kernel for a rate pair and quality.
///
/// The cache is a small LRU list: a hit moves the entry to the front; a miss
/// builds the kernel, inserts it at the front and evicts from the back until
/// at most [`KERNEL_CACHE_ENTRIES`] kernels and [`KERNEL_CACHE_BYTES`] bytes
/// of coefficients remain.
///
/// # Errors
/// See [`Kernel::new`].
fn cached_kernel(
    in_sr: u32,
    out_sr: u32,
    quality: ResamplingQuality,
) -> AudioSampleResult<Arc<Kernel>> {
    let key = KernelKey {
        in_sr,
        out_sr,
        quality,
    };
    KERNEL_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        if let Some(pos) = cache.iter().position(|(k, _)| *k == key) {
            let entry = cache.remove(pos);
            let kernel = Arc::clone(&entry.1);
            cache.insert(0, entry);
            return Ok(kernel);
        }
        let kernel = Arc::new(Kernel::new(
            in_sr,
            out_sr,
            kernel_kind(quality, in_sr, out_sr),
        )?);
        cache.insert(0, (key, Arc::clone(&kernel)));
        let mut total = 0;
        let keep = cache
            .iter()
            .position(|(_, k)| {
                total += k.size_bytes();
                total > KERNEL_CACHE_BYTES
            })
            .unwrap_or(cache.len())
            .min(KERNEL_CACHE_ENTRIES);
        cache.truncate(keep);
        Ok(kernel)
    })
}

/// Incremental polyphase resampling state shared by the whole-buffer and the
/// streaming paths, so both produce identical output for the same input.
///
/// Each channel buffer holds input samples from absolute index `buf_start`
/// on; the signal before index 0 is zeros (pre-filled).
#[derive(Debug)]
struct PolyphaseState {
    kernel: Arc<Kernel>,
    buffers: Vec<Vec<f64>>,
    /// Absolute input index of `buffers[c][0]`.
    buf_start: i64,
    /// Next output index to produce.
    next_out: u64,
    /// Total input samples received per channel.
    received: u64,
    /// Last input sample per channel (for `hold_last` kernels).
    last: Vec<f64>,
}

impl PolyphaseState {
    fn new(kernel: Arc<Kernel>, channels: usize) -> Self {
        let half = kernel.half;
        Self {
            buffers: vec![vec![0.0; half]; channels],
            buf_start: -(half as i64),
            next_out: 0,
            received: 0,
            last: vec![0.0; channels],
            kernel,
        }
    }

    fn reset(&mut self) {
        let half = self.kernel.half;
        for b in &mut self.buffers {
            b.clear();
            b.resize(half, 0.0);
        }
        self.buf_start = -(half as i64);
        self.next_out = 0;
        self.received = 0;
        self.last.fill(0.0);
    }

    /// Appends one block of input (one slice per channel, equal lengths).
    fn push(&mut self, input: &[&[f64]]) {
        for ((buf, src), last) in self.buffers.iter_mut().zip(input).zip(&mut self.last) {
            buf.extend_from_slice(src);
            if let Some(&v) = src.last() {
                *last = v;
            }
        }
        self.received += input.first().map_or(0, |c| c.len()) as u64;
    }

    /// Produces every output sample whose input window lies before absolute
    /// index `available` (exclusive), stopping at output index `limit`, then
    /// discards input that is no longer needed.
    fn produce(&mut self, available: i64, limit: Option<u64>) -> Vec<Vec<f64>> {
        let kernel = Arc::clone(&self.kernel);
        let half = kernel.half as i64;
        let taps = kernel.taps;
        let mut out: Vec<Vec<f64>> = vec![Vec::new(); self.buffers.len()];
        // Step the position incrementally: output k + 1 is M / L input
        // samples after output k.
        let (step_int, step_frac) = ((kernel.m / kernel.l) as i64, kernel.m % kernel.l);
        let (n0, mut p) = kernel.position(self.next_out);
        let mut n0 = n0 as i64;
        loop {
            if limit.is_some_and(|lim| self.next_out >= lim) || n0 + half >= available {
                break;
            }
            let start = (n0 - half + 1 - self.buf_start) as usize;
            for (buf, o) in self.buffers.iter().zip(out.iter_mut()) {
                o.push(kernel.eval(&buf[start..start + taps], p));
            }
            self.next_out += 1;
            n0 += step_int;
            p += step_frac;
            if p >= kernel.l {
                p -= kernel.l;
                n0 += 1;
            }
        }
        // Keep from the first sample the next output needs.
        let keep_from = (n0 - half + 1).max(self.buf_start);
        let drop = ((keep_from - self.buf_start) as usize).min(self.buffers[0].len());
        if drop > 0 {
            for buf in &mut self.buffers {
                buf.drain(..drop);
            }
            self.buf_start += drop as i64;
        }
        out
    }

    /// Produces what the input so far allows (the look-ahead of `half`
    /// samples is held back until more input or [`finish`](Self::finish)).
    fn process(&mut self, input: &[&[f64]]) -> Vec<Vec<f64>> {
        self.push(input);
        let available = self.received as i64;
        self.produce(available, None)
    }

    /// Flushes the tail: extends the input (zeros, or the held last sample for
    /// linear interpolation) and produces the remaining outputs, so that the
    /// total output is exactly `ceil(received · L / M)` samples.
    fn finish(&mut self) -> Vec<Vec<f64>> {
        let half = self.kernel.half;
        let hold = self.kernel.hold_last;
        for (buf, &last) in self.buffers.iter_mut().zip(&self.last) {
            let pad = if hold { last } else { 0.0 };
            buf.extend(std::iter::repeat_n(pad, half));
        }
        let available = self.received as i64 + half as i64;
        let total = self.kernel.output_len(self.received);
        self.produce(available, Some(total))
    }
}

// ---------------------------------------------------------------------------
// FFT evaluation of the polyphase filter (small L and M)
// ---------------------------------------------------------------------------

/// Largest `L · M` for which the FFT path is considered (one filter spectrum
/// is precomputed per `(output phase, input residue)` pair).
const FFT_MAX_SPECTRA: usize = 64;
/// Inputs shorter than this always use the direct path.
const FFT_MIN_INPUT: usize = 8192;

thread_local! {
    static RESAMPLE_FFT_PLANNER: RefCell<RealFftPlanner<f64>> = RefCell::new(RealFftPlanner::new());
}

/// Exact FFT (overlap-save) evaluation of a whole-buffer polyphase resample,
/// for ratios `L / M` with small `L` and `M` (e.g. 2:1 decimation, 1:2
/// interpolation, 3:1, 2:3).
///
/// It computes the same sums as [`Kernel::eval`] with the same coefficients,
/// only in a different order, so the result equals the direct path up to
/// floating-point round-off. Output `k = q · L + i` (sub-stream `i`) is
///
/// ```text
/// y[q L + i] = Σ_j c[p_i][j] · x[q M + d_i − half + 1 + j]
///            = Σ_r Σ_t G_{i,r}[t] · x_r[q + t0 + t]
/// ```
///
/// with `p_i = i M mod L`, `d_i = ⌊i M / L⌋`, `x_r[v] = x[M v + r]` the input
/// split into its `M` residue streams, and `G_{i,r}` the taps of row `p_i`
/// that land on residue `r`. Each block of every sub-stream is therefore a sum
/// of `M` correlations at the input rate divided by `M`, which share the
/// residue streams' forward FFTs: per block, `M` forward and `L` inverse real
/// FFTs replace `L · block · taps` multiply-adds.
struct FftPolyphase {
    l: usize,
    m: usize,
    /// FFT size.
    fft_len: usize,
    /// Taps per `(i, r)` filter.
    filter_len: usize,
    /// Valid outputs per sub-stream per block (`fft_len - filter_len + 1`).
    block: usize,
    /// Residue-stream index of the first filter tap (may be negative).
    t0: i64,
    /// Spectra of the time-reversed `G_{i,r}`, scaled by `1 / fft_len`;
    /// index `i * m + r`.
    spectra: Vec<Vec<Complex<f64>>>,
    forward: Arc<dyn RealToComplex<f64>>,
    inverse: Arc<dyn ComplexToReal<f64>>,
}

impl FftPolyphase {
    /// Builds the FFT evaluator when it is applicable and estimated to be
    /// faster than the direct dot products for an input of `n` samples.
    fn new(kernel: &Kernel, n: usize) -> Option<Self> {
        if !kernel.exact || kernel.hold_last || n < FFT_MIN_INPUT {
            return None;
        }
        let (l, m) = (kernel.l as usize, kernel.m as usize);
        if l.saturating_mul(m) > FFT_MAX_SPECTRA {
            return None;
        }
        let (half, taps) = (kernel.half as i64, kernel.taps);
        let a_len = taps.div_ceil(m);
        let mi = m as i64;
        // Per sub-stream: phase row, input offset of tap 0, first residue index.
        let subs: Vec<(usize, i64, i64)> = (0..l)
            .map(|i| {
                let p = (i * m) % l;
                let b = (i * m / l) as i64 - half + 1;
                (p, b, b.div_euclid(mi))
            })
            .collect();
        let t0 = subs.iter().map(|s| s.2).min()?;
        let t_end = subs.iter().map(|s| s.2 + a_len as i64 + 1).max()?;
        let filter_len = (t_end - t0) as usize;

        // Cost model (CPU cycles per output sample, measured on AVX2 + FMA):
        // the direct path costs ~0.35 cycles per tap; a real FFT of size F
        // ~0.45 F log2 F, a complex multiply-add per bin ~1.
        let direct = 0.35 * taps as f64 + 10.0;
        let (fft_len, cost) = (8..=16)
            .map(|e| 1usize << e)
            .filter(|&f| f >= 2 * filter_len)
            .map(|f| {
                let block = (f - filter_len + 1) as f64;
                let fft = 0.45 * f as f64 * (f as f64).log2();
                let macs = (l * m * (f / 2 + 1)) as f64;
                (f, ((l + m) as f64 * fft + macs) / (l as f64 * block))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))?;
        if cost >= 0.8 * direct {
            return None;
        }

        let (forward, inverse) = RESAMPLE_FFT_PLANNER.with(|p| {
            let mut p = p.borrow_mut();
            (p.plan_fft_forward(fft_len), p.plan_fft_inverse(fft_len))
        });
        let scale = 1.0 / fft_len as f64;
        let mut buf = vec![0.0; fft_len];
        let mut spectra = Vec::with_capacity(l * m);
        for &(p, b, _) in &subs {
            let row = &kernel.coeffs[p * taps..(p + 1) * taps];
            let mut g = vec![vec![0.0; filter_len]; m];
            for s in 0..m {
                let u = b + s as i64;
                let (r, e) = (u.rem_euclid(mi) as usize, u.div_euclid(mi));
                for a in 0..a_len {
                    if let Some(&c) = row.get(m * a + s) {
                        g[r][(e - t0) as usize + a] = c;
                    }
                }
            }
            for g_r in g {
                buf.fill(0.0);
                for (dst, &v) in buf.iter_mut().zip(g_r.iter().rev()) {
                    *dst = v * scale;
                }
                let mut spec = forward.make_output_vec();
                forward.process(&mut buf, &mut spec).ok()?;
                spectra.push(spec);
            }
        }
        Some(Self {
            l,
            m,
            fft_len,
            filter_len,
            block: fft_len - filter_len + 1,
            t0,
            spectra,
            forward,
            inverse,
        })
    }

    /// Fills `dst` with `x[M (t0 + q0 + v) + r]` for `v in 0..dst.len()`
    /// (zero where the index falls outside `x`).
    fn gather(&self, x: &[f64], r: usize, q0: usize, dst: &mut [f64]) {
        let m = self.m as i64;
        let start = (self.t0 + q0 as i64) * m + r as i64;
        // Leading entries with a negative input index.
        let lead = if start < 0 {
            usize::try_from((-start + m - 1) / m).unwrap_or(usize::MAX)
        } else {
            0
        }
        .min(dst.len());
        dst[..lead].fill(0.0);
        let first = start + lead as i64 * m;
        let mut filled = lead;
        if let Ok(first) = usize::try_from(first)
            && first < x.len()
        {
            for (d, &v) in dst[lead..]
                .iter_mut()
                .zip(x[first..].iter().step_by(self.m))
            {
                *d = v;
                filled += 1;
            }
        }
        dst[filled..].fill(0.0);
    }

    /// Resamples one channel, producing `n_out` samples.
    fn process(&self, x: &[f64], n_out: usize) -> Vec<f64> {
        let (l, m, f) = (self.l, self.m, self.fft_len);
        let n_q = n_out.div_ceil(l);
        let n_blocks = n_q.div_ceil(self.block);
        let n_bins = f / 2 + 1;
        let mut y = vec![0.0; n_out];
        let mut time = vec![0.0; f];
        let mut fwd_scratch = self.forward.make_scratch_vec();
        let mut inv_scratch = self.inverse.make_scratch_vec();
        let mut x_spec = vec![vec![Complex::new(0.0, 0.0); n_bins]; m];
        let mut acc = vec![Complex::new(0.0, 0.0); n_bins];
        for blk in 0..n_blocks {
            let q0 = blk * self.block;
            for (r, spec) in x_spec.iter_mut().enumerate() {
                // Residue stream r: time[v] = x[M (t0 + q0 + v) + r], zero outside x.
                self.gather(x, r, q0, &mut time);
                // Infallible: buffer lengths come from the plan itself.
                let _ = self
                    .forward
                    .process_with_scratch(&mut time, spec, &mut fwd_scratch);
            }
            for i in 0..l {
                acc.fill(Complex::new(0.0, 0.0));
                for (r, spec) in x_spec.iter().enumerate() {
                    for ((a, &xv), &hv) in acc.iter_mut().zip(spec).zip(&self.spectra[i * m + r]) {
                        *a += xv * hv;
                    }
                }
                // Real inputs: the DC and Nyquist bins are real.
                acc[0].im = 0.0;
                acc[n_bins - 1].im = 0.0;
                let _ = self
                    .inverse
                    .process_with_scratch(&mut acc, &mut time, &mut inv_scratch);
                let valid = &time[self.filter_len - 1..];
                for (qq, &v) in valid.iter().enumerate().take(self.block) {
                    let k = (q0 + qq) * l + i;
                    if k >= n_out {
                        break;
                    }
                    y[k] = v;
                }
            }
        }
        y
    }
}

// ---------------------------------------------------------------------------
// Whole-buffer resampling
// ---------------------------------------------------------------------------

/// Resamples audio to a new sample rate.
///
/// The output is time-aligned with the input (no filter delay) and has exactly
/// `ceil(n · target_rate / input_rate)` samples per channel. See the
/// [module documentation](self) for the filter used by each quality tier.
///
/// # Arguments
/// - `audio` - The input audio samples (any channel count).
/// - `target_sample_rate` - Desired output sample rate in Hz.
/// - `quality` - Quality/performance trade-off setting.
///
/// # Returns
/// A new `AudioSamples` instance at the target sample rate (a copy of the input
/// if the rates are equal).
///
/// # Errors
/// - [`crate::AudioSampleError::Parameter`] if the rate ratio is too extreme
///   for a filter table (decimation by more than ~10^5 : 1), or if the channel
///   data cannot be read (never for well-formed audio).
///
/// # Examples
/// ```
/// use audio_samples::{AudioSamples, resample, sample_rate};
/// use audio_samples::operations::types::ResamplingQuality;
/// use ndarray::array;
///
/// let audio = AudioSamples::new_mono(array![1.0f32, 0.5, -0.5, -1.0], sample_rate!(44100)).unwrap();
/// let resampled = resample(&audio, sample_rate!(48000), ResamplingQuality::High).unwrap();
/// assert_eq!(resampled.sample_rate(), sample_rate!(48000));
/// assert_eq!(resampled.samples_per_channel().get(), 5); // ceil(4 * 48000 / 44100)
/// ```
///
/// # Differences from librosa
/// See the [module documentation](self#differences-from-librosa).
#[inline]
pub fn resample<T>(
    audio: &AudioSamples<T>,
    target_sample_rate: SampleRate,
    quality: ResamplingQuality,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    T: StandardSample,
{
    if audio.sample_rate() == target_sample_rate {
        return Ok(audio.clone().into_owned());
    }

    let in_sr = audio.sample_rate().get();
    let out_sr = target_sample_rate.get();
    match kernel_kind(quality, in_sr, out_sr) {
        // Linear interpolation: dedicated (SIMD) loops, f32.
        KernelKind::Linear => resample_linear(&audio.as_f32(), target_sample_rate),
        KernelKind::Sinc(_) => {
            let data = channel_data_f64(audio)?;
            let kernel = cached_kernel(in_sr, out_sr, quality)?;
            let n_in = data.first().map_or(0, Vec::len);
            if let Some(fft) = FftPolyphase::new(&kernel, n_in) {
                let n_out = kernel.output_len(n_in as u64) as usize;
                let out: Vec<Vec<f64>> = data.iter().map(|x| fft.process(x, n_out)).collect();
                return assemble_output(out, n_out, audio.num_channels(), target_sample_rate);
            }
            let mut state = PolyphaseState::new(kernel, data.len());
            let slices: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
            let mut out = state.process(&slices);
            for (o, tail) in out.iter_mut().zip(state.finish()) {
                o.extend(tail);
            }
            let frames = out[0].len();
            assemble_output(out, frames, audio.num_channels(), target_sample_rate)
        }
    }
}

// ---------------------------------------------------------------------------
// Fast upsampling - linear interpolation
// ---------------------------------------------------------------------------
//
// Zero construction overhead. Output sample `k` is the straight-line
// interpolation of the input at position `k · in_rate / out_rate`, holding the
// last sample past the end. Integer ratios (e.g. 22050 → 44100) are
// special-cased so the inner loop vectorises. Only used for upsampling: the
// triangle kernel does not suppress images, which is acceptable for the Fast
// preset; downsampling always goes through the anti-aliased sinc kernel.

fn resample_linear<T>(
    audio: &AudioSamples<f32>,
    target_sample_rate: SampleRate,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    T: StandardSample,
{
    let in_sr = audio.sample_rate().get();
    let out_sr = target_sample_rate.get();
    let in_frames = audio.samples_per_channel().get();
    let out_frames = (in_frames as u64 * out_sr as u64).div_ceil(in_sr as u64) as usize;

    let process_channel = |src: &[f32], dst: &mut [f32]| {
        if out_sr.is_multiple_of(in_sr) {
            // Integer upsampling by N.
            let n = (out_sr / in_sr) as usize;

            // n=2 hot path: SSE2 interleave + midpoint average, 8 output
            // samples per instruction pair.
            #[cfg(target_arch = "x86_64")]
            if n == 2 {
                // SAFETY: SSE2 is mandated by the x86_64 ABI.
                unsafe {
                    upsample_by_2_simd(src, dst);
                }
                return;
            }

            // General N: linear interpolation across N-wide output chunks.
            let step_f32 = 1.0f32 / n as f32;
            let safe_in = src.len().saturating_sub(1);
            let safe_out = safe_in * n;
            let split = safe_out.min(dst.len());
            let (main, tail) = dst.split_at_mut(split);

            for (in_i, out_chunk) in main.chunks_exact_mut(n).enumerate() {
                let s0 = src[in_i];
                let s1 = src[in_i + 1]; // always valid: in_i < safe_in
                let diff = s1 - s0;
                for (k, out_s) in out_chunk.iter_mut().enumerate() {
                    *out_s = s0 + k as f32 * step_f32 * diff;
                }
            }
            let last_s = src.last().copied().unwrap_or(0.0);
            tail.fill(last_s);
        } else {
            // General fractional ratio.
            let step = in_sr as f64 / out_sr as f64;
            let last = src.len() - 1;
            for (out_i, out_s) in dst.iter_mut().enumerate() {
                let pos = out_i as f64 * step;
                let i0 = pos as usize;
                let i1 = (i0 + 1).min(last);
                let frac = (pos - i0 as f64) as f32;
                *out_s = src[i0] + frac * (src[i1] - src[i0]);
            }
        }
    };

    let input = extract_channel_data(audio)?;
    let output: Vec<Vec<f32>> = input
        .iter()
        .map(|src| {
            let mut dst = vec![0.0f32; out_frames];
            process_channel(src, &mut dst);
            dst
        })
        .collect();

    assemble_output(output, out_frames, audio.num_channels(), target_sample_rate)
}

// ---------------------------------------------------------------------------
// Streaming (chunk-by-chunk) resampling
// ---------------------------------------------------------------------------

/// Stateful resampler for continuous, chunk-by-chunk resampling of a live audio
/// stream.
///
/// Construct once per stream with [`new`](Self::new) and feed it via
/// [`process_chunk`](Self::process_chunk) as chunks arrive. Each call returns
/// every output sample whose input is already known: the interpolation filter
/// needs a short look-ahead (up to a few hundred input samples for `High`), so
/// the first call(s) may return `Ok(None)` and every call holds back that much
/// input until more arrives. Output chunk sizes need not match input chunk
/// sizes.
///
/// The stream uses the same filters and the same exact sample positions as
/// [`resample`]: the output is time-aligned with the input (no delay), and
/// after [`finish`](Self::finish) the concatenated output equals
/// `resample(whole_input, ...)` up to floating-point rounding, with exactly
/// `ceil(n · out_rate / in_rate)` samples in total.
///
/// [`reset`](Self::reset) clears all state - call it after a seek, since the
/// buffered input no longer corresponds to a contiguous position in the stream.
///
/// # Examples
/// ```
/// use audio_samples::{AudioSamples, StreamingResampler, resample, sample_rate, channels};
/// use audio_samples::operations::types::ResamplingQuality;
/// use ndarray::Array1;
///
/// let x: Vec<f64> = (0..10_000).map(|i| (i as f64 * 0.05).sin()).collect();
/// let mut stream = StreamingResampler::<f64>::new(
///     sample_rate!(44100), sample_rate!(16000), channels!(1), ResamplingQuality::Medium,
/// ).unwrap();
/// let mut total = 0;
/// for chunk in x.chunks(1000) {
///     let piece = AudioSamples::new_mono(Array1::from(chunk.to_vec()), sample_rate!(44100)).unwrap();
///     if let Some(out) = stream.process_chunk(&piece).unwrap() {
///         total += out.samples_per_channel().get();
///     }
/// }
/// if let Some(out) = stream.finish().unwrap() {
///     total += out.samples_per_channel().get();
/// }
/// assert_eq!(total, 3629); // ceil(10000 * 16000 / 44100)
/// ```
#[derive(Debug)]
pub struct StreamingResampler<T: StandardSample> {
    state: PolyphaseState,
    channels: usize,
    channel_count: ChannelCount,
    out_sample_rate: SampleRate,
    _marker: std::marker::PhantomData<T>,
}

impl<T: StandardSample> StreamingResampler<T> {
    /// Create a streaming resampler from `in_sample_rate` to `out_sample_rate` for
    /// `channels` channels of audio, at the given quality tier.
    ///
    /// # Errors
    /// [`crate::AudioSampleError::Parameter`] if the rate ratio is too extreme
    /// for a filter table (decimation by more than ~10^5 : 1).
    pub fn new(
        in_sample_rate: SampleRate,
        out_sample_rate: SampleRate,
        channels: ChannelCount,
        quality: ResamplingQuality,
    ) -> AudioSampleResult<Self> {
        let nbr_channels = channels.get() as usize;
        let kernel = cached_kernel(in_sample_rate.get(), out_sample_rate.get(), quality)?;
        Ok(Self {
            state: PolyphaseState::new(kernel, nbr_channels),
            channels: nbr_channels,
            channel_count: channels,
            out_sample_rate,
            _marker: std::marker::PhantomData,
        })
    }

    /// Sample rate this resampler produces.
    pub fn output_sample_rate(&self) -> SampleRate {
        self.out_sample_rate
    }

    /// Feed the next chunk of input audio (must have the channel count this
    /// resampler was constructed with). Returns `Ok(None)` if no output sample
    /// could be completed yet (the filter's look-ahead is not yet filled).
    ///
    /// # Errors
    /// - [`crate::AudioSampleError::Parameter`] if the chunk's channel count
    ///   differs from the configured one.
    pub fn process_chunk(
        &mut self,
        chunk: &AudioSamples<T>,
    ) -> AudioSampleResult<Option<AudioSamples<'static, T>>> {
        if chunk.num_channels().get() as usize != self.channels {
            return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
                "channels",
                "chunk channel count does not match the resampler's configured channel count",
            )));
        }
        let input = extract_channel_data(&chunk.as_f64())?;
        let slices: Vec<&[f64]> = input.iter().map(Vec::as_slice).collect();
        let out = self.state.process(&slices);
        self.emit(out)
    }

    /// Flush the held-back tail (the input is extended with zeros, or by holding
    /// the last sample for `Fast` upsampling). Call exactly once when the source
    /// is exhausted. Returns `Ok(None)` if there is nothing left to emit.
    ///
    /// After this call the stream has produced exactly
    /// `ceil(n · out_rate / in_rate)` samples for `n` input samples. Call
    /// [`reset`](Self::reset) before reusing the resampler.
    ///
    /// # Errors
    /// Currently infallible; the `Result` is kept for API stability.
    pub fn finish(&mut self) -> AudioSampleResult<Option<AudioSamples<'static, T>>> {
        let out = self.state.finish();
        self.emit(out)
    }

    /// Reset the resampler and discard any pending input. Call this after a
    /// seek - buffered state no longer corresponds to a contiguous stream
    /// position.
    pub fn reset(&mut self) {
        self.state.reset();
    }

    fn emit(&self, out: Vec<Vec<f64>>) -> AudioSampleResult<Option<AudioSamples<'static, T>>> {
        let frames = out.first().map_or(0, Vec::len);
        if frames == 0 {
            return Ok(None);
        }
        assemble_output(out, frames, self.channel_count, self.out_sample_rate).map(Some)
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Extracts per-channel data into owned `Vec`s.
fn extract_channel_data<S: StandardSample>(
    audio: &AudioSamples<S>,
) -> AudioSampleResult<Vec<Vec<S>>> {
    if audio.is_mono() {
        let mono = audio.as_mono().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Failed to get mono data",
            ))
        })?;
        Ok(vec![mono.to_vec()])
    } else {
        let multi = audio.as_multi_channel().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Failed to get multi-channel data",
            ))
        })?;
        let channels = audio.num_channels().get() as usize;
        Ok((0..channels).map(|ch| multi.row(ch).to_vec()).collect())
    }
}

/// Extracts per-channel data converted to `f64` (audio-aware, as
/// [`AudioTypeConversion::as_f64`]) in a single pass.
fn channel_data_f64<S: StandardSample>(
    audio: &AudioSamples<S>,
) -> AudioSampleResult<Vec<Vec<f64>>> {
    let to_f64 = |s: &S| -> f64 { ConvertTo::<f64>::convert_to(*s) };
    if let Some(mono) = audio.as_mono() {
        Ok(vec![match mono.as_slice() {
            Some(x) => x.iter().map(to_f64).collect(),
            None => mono.iter().map(to_f64).collect(),
        }])
    } else {
        let multi = audio.as_multi_channel().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Failed to get multi-channel data",
            ))
        })?;
        let channels = audio.num_channels().get() as usize;
        Ok((0..channels)
            .map(|ch| multi.row(ch).iter().map(to_f64).collect())
            .collect())
    }
}

/// Assembles per-channel output `Vec`s into an [`AudioSamples<T>`].
///
/// Multi-channel data is laid out channel-major (`[ch0[..], ch1[..], ...]`), as
/// required by [`AudioSamples::new_multi_channel_from_vec`].
fn assemble_output<S, T>(
    channel_data: Vec<Vec<S>>,
    total_frames: usize,
    channel_count: ChannelCount,
    sample_rate: SampleRate,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    S: StandardSample,
    T: StandardSample + crate::ConvertFrom<S>,
{
    if channel_data.len() == 1 {
        let data = NonEmptyVec::new(channel_data.into_iter().next().unwrap_or_default())
            .map_err(|_| AudioSampleError::empty_data("resample"))?;
        return Ok(AudioSamples::from_mono_vec::<S>(data, sample_rate));
    }
    let mut flat = Vec::with_capacity(total_frames * channel_data.len());
    for ch_data in &channel_data {
        flat.extend_from_slice(ch_data);
    }
    let data = NonEmptyVec::new(flat).map_err(|_| AudioSampleError::empty_data("resample"))?;
    AudioSamples::new_multi_channel_from_vec::<S>(data, channel_count, sample_rate)
}

/// Resamples audio by a specific ratio.
///
/// The target rate is `round(input_rate · ratio)`; the conversion is then the
/// same as [`resample`] to that rate.
///
/// # Arguments
/// - `audio` - The input audio samples
/// - `ratio` - Resampling ratio (output_rate / input_rate)
/// - `quality` - Quality/performance trade-off setting
///
/// # Returns
/// A new `AudioSamples` instance resampled by the given ratio.
///
/// # Errors
/// - [`crate::AudioSampleError::Parameter`] if the ratio is not positive or the
///   resulting target rate rounds to zero.
///
/// # Examples
/// ```
/// use audio_samples::{AudioSamples, resample_by_ratio, sample_rate};
/// use audio_samples::operations::types::ResamplingQuality;
/// use ndarray::array;
///
/// let audio = AudioSamples::new_mono(array![1.0f32, 0.0, -1.0, 0.0], sample_rate!(44100)).unwrap();
/// let half = resample_by_ratio(&audio, 0.5, ResamplingQuality::Medium).unwrap();
/// assert_eq!(half.sample_rate(), sample_rate!(22050));
/// assert_eq!(half.samples_per_channel().get(), 2);
/// ```
#[inline]
pub fn resample_by_ratio<T>(
    audio: &AudioSamples<T>,
    ratio: f64,
    quality: ResamplingQuality,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    T: StandardSample,
{
    if ratio.is_nan() || ratio <= 0.0 {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "ratio",
            format!("Invalid resampling ratio: {ratio}"),
        )));
    }

    let input_sample_rate = audio.sample_rate().get();
    let target_sample_rate = (f64::from(input_sample_rate) * ratio).round() as usize;
    let target_sample_rate = SampleRate::new(target_sample_rate as u32).ok_or_else(|| {
        AudioSampleError::Parameter(ParameterError::invalid_value(
            "target_sample_rate",
            "Calculated target sample rate is invalid",
        ))
    })?;

    resample(audio, target_sample_rate, quality)
}

/// Stride-2 upsample (linear interpolation) using SSE2.
///
/// Each iteration reads 4 input samples plus 1 overlap sample, computing
/// midpoints with `_mm_add_ps` + `_mm_mul_ps`, then interleaving originals
/// and averages with `_mm_unpacklo/hi_ps` to produce 8 output samples.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn upsample_by_2_simd(src: &[f32], dst: &mut [f32]) {
    use core::arch::x86_64::*;

    // Each iteration accesses src[i*4 .. i*4+5], so need i*4+4 < src.len().
    // Maximum safe i = (src.len() - 2) / 4  (i.e. (src.len()-1)/4 in int-div).
    let n_simd = if src.len() >= 5 {
        ((src.len() - 1) / 4).min(dst.len() / 8)
    } else {
        0
    };

    let src_ptr = src.as_ptr();
    let dst_ptr = dst.as_mut_ptr();

    for i in 0..n_simd {
        // SAFETY: n_simd is bounded so that i*4+4 < src.len() (reads src_ptr.add(i*4 ..
        // i*4+5)) and i*8+8 <= dst.len() (writes dst_ptr.add(i*8 .. i*8+8)) for every
        // i < n_simd; _mm_loadu/storeu_ps are unaligned, and SSE2 is in the x86_64 baseline.
        unsafe {
            let half = _mm_set1_ps(0.5f32);
            let v_cur = _mm_loadu_ps(src_ptr.add(i * 4)); // [s0, s1, s2, s3]
            let v_next = _mm_loadu_ps(src_ptr.add(i * 4 + 1)); // [s1, s2, s3, s4]
            // v_avg = [(s0+s1)/2, (s1+s2)/2, (s2+s3)/2, (s3+s4)/2]
            let v_avg = _mm_mul_ps(_mm_add_ps(v_cur, v_next), half);
            // Interleave: lo = [s0, avg01, s1, avg12],  hi = [s2, avg23, s3, avg34]
            let lo = _mm_unpacklo_ps(v_cur, v_avg);
            let hi = _mm_unpackhi_ps(v_cur, v_avg);
            _mm_storeu_ps(dst_ptr.add(i * 8), lo);
            _mm_storeu_ps(dst_ptr.add(i * 8 + 4), hi);
        }
    }

    // Scalar tail.
    let simd_out = n_simd * 8;
    let simd_in = n_simd * 4;
    let last = src.len().saturating_sub(1);
    for (j, out_s) in dst[simd_out..].iter_mut().enumerate() {
        let in_i = simd_in + j / 2;
        let k = j & 1;
        if in_i < last {
            let s0 = src[in_i];
            let s1 = src[in_i + 1];
            *out_s = s0 + k as f32 * 0.5 * (s1 - s0);
        } else {
            *out_s = *src.last().unwrap_or(&0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AudioSamples;
    use crate::channels;
    use crate::sample_rate;
    use ndarray::{Array1, Array2, array};

    fn sine(n: usize, freq: f64, sr: f64, phase: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr + phase).sin())
            .collect()
    }

    fn mono(x: Vec<f64>, sr: SampleRate) -> AudioSamples<'static, f64> {
        AudioSamples::new_mono(Array1::from(x), sr).unwrap()
    }

    /// SNR (dB) of `actual` against `expected` over `range`.
    fn snr_db(actual: &[f64], expected: &[f64], range: std::ops::Range<usize>) -> f64 {
        let (mut sig, mut err) = (0.0, 0.0);
        for i in range {
            sig += expected[i] * expected[i];
            err += (actual[i] - expected[i]).powi(2);
        }
        10.0 * (sig / err.max(1e-300)).log10()
    }

    #[test]
    fn test_resample_mono() {
        let samples = (0..1024)
            .map(|x| (x as f32 * 2.0 * std::f32::consts::PI * 440.0 / 44100.0).sin())
            .collect::<Vec<f32>>();
        let data = Array1::from_vec(samples);
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let resampled = resample(&audio, sample_rate!(48000), ResamplingQuality::Medium).unwrap();
        assert_eq!(resampled.sample_rate(), sample_rate!(48000));
        assert_eq!(resampled.num_channels(), channels!(1));
        assert_eq!(resampled.samples_per_channel().get(), 1115); // ceil(1024 * 48000 / 44100)
    }

    #[test]
    fn test_resample_by_ratio() {
        let data = array![1.0f32, 0.0, -1.0, 0.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let upsampled = resample_by_ratio(&audio, 2.0, ResamplingQuality::Fast).unwrap();
        assert_eq!(upsampled.sample_rate(), sample_rate!(88200));
        assert_eq!(upsampled.samples_per_channel().get(), 8);

        let downsampled = resample_by_ratio(&audio, 0.5, ResamplingQuality::Fast).unwrap();
        assert_eq!(downsampled.sample_rate(), sample_rate!(22050));
        assert_eq!(downsampled.samples_per_channel().get(), 2);
    }

    #[test]
    fn test_no_resampling_needed() {
        let data = array![1.0f32, 0.0, -1.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        let result = resample(&audio, sample_rate!(44100), ResamplingQuality::High).unwrap();
        assert_eq!(result.sample_rate(), sample_rate!(44100));

        let original_mono = audio.as_mono().unwrap();
        let result_mono = result.as_mono().unwrap();
        assert_eq!(original_mono.len(), result_mono.len());
    }

    #[test]
    fn test_invalid_ratio() {
        let data = array![1.0f32, 0.0];
        let audio = AudioSamples::new_mono(data, sample_rate!(44100)).unwrap();

        assert!(resample_by_ratio(&audio, -1.0, ResamplingQuality::Fast).is_err());
        assert!(resample_by_ratio(&audio, 0.0, ResamplingQuality::Fast).is_err());
        assert!(resample_by_ratio(&audio, f64::NAN, ResamplingQuality::Fast).is_err());
    }

    /// Output length is exactly `ceil(n * out / in)` for every quality (this
    /// test used to allow 200 samples of slack, which hid the lost tail).
    #[test]
    fn test_resample_correct_length_long_signal() {
        let samples = (0..220500)
            .map(|x| (x as f32 * 2.0 * std::f32::consts::PI * 440.0 / 44100.0).sin())
            .collect::<Vec<f32>>();
        let audio = AudioSamples::new_mono(Array1::from_vec(samples), sample_rate!(44100)).unwrap();

        for quality in [
            ResamplingQuality::Fast,
            ResamplingQuality::Medium,
            ResamplingQuality::High,
        ] {
            for (out_sr, expected) in [(22050u32, 110250usize), (16000, 80000), (48000, 240000)] {
                let rate = SampleRate::new(out_sr).unwrap();
                let got = resample(&audio, rate, quality)
                    .unwrap()
                    .samples_per_channel()
                    .get();
                assert_eq!(got, expected, "{quality:?} 44100 -> {out_sr}");
            }
        }
        // Exact integer length (librosa's float ceil gives 96001 here).
        let audio = mono(vec![0.1; 88200], sample_rate!(44100));
        let y = resample(&audio, sample_rate!(48000), ResamplingQuality::High).unwrap();
        assert_eq!(y.samples_per_channel().get(), 96000);
    }

    /// Regression: Medium/High output was delayed by the rubato filter delay
    /// (64/128 input samples) and lost the tail. A band-limited tone must now
    /// line up with the analytic tone at the output rate.
    #[test]
    fn test_resample_is_time_aligned() {
        let cases = [
            (44100u32, 16000u32, 1000.0),
            (22050, 44100, 3000.0),
            (22050, 16000, 2500.0),
            (44100, 48000, 5000.0),
            (48000, 44100, 7000.0),
        ];
        for (in_sr, out_sr, f) in cases {
            let x = sine(in_sr as usize, f, f64::from(in_sr), 0.3);
            let audio = mono(x, SampleRate::new(in_sr).unwrap());
            let expected = sine(out_sr as usize, f, f64::from(out_sr), 0.3);
            // Ignore the edges, where the input is implicitly zero-padded.
            let interior = (out_sr as usize / 20)..(out_sr as usize * 19 / 20);
            for (quality, min_snr) in [
                (ResamplingQuality::High, 130.0),
                (ResamplingQuality::Medium, 110.0),
            ] {
                let y = resample(&audio, SampleRate::new(out_sr).unwrap(), quality).unwrap();
                let y = y.as_mono().unwrap().to_vec();
                assert_eq!(y.len(), expected.len());
                let snr = snr_db(&y, &expected, interior.clone());
                assert!(
                    snr > min_snr,
                    "{quality:?} {in_sr} -> {out_sr}, {f} Hz: SNR {snr:.1} dB"
                );
            }
        }
    }

    /// Regression: Fast downsampling was plain decimation, so content above
    /// the new Nyquist frequency aliased at full level.
    #[test]
    fn test_fast_downsampling_is_anti_aliased() {
        // 13 kHz is above the 11.025 kHz Nyquist of 22.05 kHz.
        let x = sine(44100, 13000.0, 44100.0, 0.0);
        let audio = mono(x, sample_rate!(44100));
        let y = resample(&audio, sample_rate!(22050), ResamplingQuality::Fast).unwrap();
        let y = y.as_mono().unwrap().to_vec();
        let mid = &y[1000..y.len() - 1000];
        let rms = (mid.iter().map(|v| v * v).sum::<f64>() / mid.len() as f64).sqrt();
        let level_db = 20.0 * (rms / std::f64::consts::FRAC_1_SQRT_2).log10();
        assert!(level_db < -70.0, "alias level {level_db:.1} dB");

        // A passband tone survives.
        let x = sine(44100, 2000.0, 44100.0, 0.0);
        let y = resample(
            &mono(x, sample_rate!(44100)),
            sample_rate!(22050),
            ResamplingQuality::Fast,
        )
        .unwrap();
        let expected = sine(22050, 2000.0, 22050.0, 0.0);
        let snr = snr_db(&y.as_mono().unwrap().to_vec(), &expected, 1000..21000);
        assert!(snr > 90.0, "Fast passband SNR {snr:.1} dB");
    }

    #[test]
    fn test_resample_multichannel_matches_per_channel() {
        let l = sine(4410, 440.0, 44100.0, 0.0);
        let r = sine(4410, 1250.0, 44100.0, 1.0);
        let mut data = Array2::<f64>::zeros((2, 4410));
        for i in 0..4410 {
            data[[0, i]] = l[i];
            data[[1, i]] = r[i];
        }
        let stereo = AudioSamples::new_multi_channel(data, sample_rate!(44100)).unwrap();
        for quality in [
            ResamplingQuality::Fast,
            ResamplingQuality::Medium,
            ResamplingQuality::High,
        ] {
            let y = resample(&stereo, sample_rate!(16000), quality).unwrap();
            let yl = resample(
                &mono(l.clone(), sample_rate!(44100)),
                sample_rate!(16000),
                quality,
            )
            .unwrap();
            let yr = resample(
                &mono(r.clone(), sample_rate!(44100)),
                sample_rate!(16000),
                quality,
            )
            .unwrap();
            let multi = y.as_multi_channel().unwrap();
            assert_eq!(multi.row(0).to_vec(), yl.as_mono().unwrap().to_vec());
            assert_eq!(multi.row(1).to_vec(), yr.as_mono().unwrap().to_vec());
        }
    }

    /// The FFT evaluation computes the same filter sums as the direct
    /// polyphase loop (only the summation order differs).
    #[test]
    fn test_fft_path_matches_direct_path() {
        let cases = [
            (44100, 22050),
            (22050, 44100),
            (2, 1),
            (48000, 16000),
            (16000, 48000),
            (44100, 29400),
            (29400, 44100),
            (48000, 8000),
        ];
        for (in_sr, out_sr) in cases {
            for quality in [
                ResamplingQuality::Fast,
                ResamplingQuality::Medium,
                ResamplingQuality::High,
            ] {
                let kernel = cached_kernel(in_sr, out_sr, quality).unwrap();
                for n in [FFT_MIN_INPUT, 20011, 65537] {
                    let x: Vec<f64> = (0..n)
                        .map(|i| (i as f64 * 0.37).sin() + 0.3 * (i as f64 * 0.0123).cos())
                        .collect();
                    let Some(fft) = FftPolyphase::new(&kernel, n) else {
                        assert!(
                            kernel.hold_last
                                || n == FFT_MIN_INPUT
                                || quality != ResamplingQuality::High,
                            "FFT path not used for {in_sr} -> {out_sr} {quality:?} (n={n})"
                        );
                        continue;
                    };
                    let mut state = PolyphaseState::new(Arc::clone(&kernel), 1);
                    let mut direct = state.process(&[&x]).remove(0);
                    direct.extend(state.finish().remove(0));
                    let fast = fft.process(&x, direct.len());
                    assert_eq!(fast.len(), direct.len());
                    let err = fast
                        .iter()
                        .zip(&direct)
                        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
                    assert!(
                        err < 1e-13,
                        "{in_sr} -> {out_sr} {quality:?} n={n}: {err:e}"
                    );
                }
            }
        }
    }

    /// The per-thread kernel cache stays bounded however many distinct ratios
    /// are used, and keeps recently used kernels.
    #[test]
    fn test_kernel_cache_is_bounded_lru() {
        let len = || KERNEL_CACHE.with(|c| c.borrow().len());
        let bytes = || {
            KERNEL_CACHE.with(|c| {
                c.borrow()
                    .iter()
                    .map(|(_, k)| k.size_bytes())
                    .sum::<usize>()
            })
        };
        let first = cached_kernel(44100, 22050, ResamplingQuality::High).unwrap();
        for out_sr in 30000..30040 {
            let _ = cached_kernel(44100, out_sr, ResamplingQuality::High).unwrap();
            // Keep the first kernel recently used: it must survive.
            assert!(Arc::ptr_eq(
                &first,
                &cached_kernel(44100, 22050, ResamplingQuality::High).unwrap()
            ));
            assert!(len() <= KERNEL_CACHE_ENTRIES);
            assert!(bytes() <= KERNEL_CACHE_BYTES);
        }
        // An entry not used for a while is evicted and rebuilt.
        let old = cached_kernel(44100, 30000, ResamplingQuality::High).unwrap();
        for out_sr in 31000..31000 + KERNEL_CACHE_ENTRIES as u32 {
            let _ = cached_kernel(44100, out_sr, ResamplingQuality::Fast).unwrap();
        }
        assert!(!Arc::ptr_eq(
            &old,
            &cached_kernel(44100, 30000, ResamplingQuality::High).unwrap()
        ));
    }

    #[test]
    fn test_interpolated_table_for_awkward_ratio() {
        // L = 44101 phases is too many for an exact table.
        let kernel = Kernel::new(
            44100,
            44101,
            kernel_kind(ResamplingQuality::High, 44100, 44101),
        )
        .unwrap();
        assert!(!kernel.exact);
        let x = sine(44100, 3000.0, 44100.0, 0.2);
        let y = resample(
            &mono(x, sample_rate!(44100)),
            sample_rate!(44101),
            ResamplingQuality::High,
        )
        .unwrap();
        let expected = sine(44101, 3000.0, 44101.0, 0.2);
        let snr = snr_db(&y.as_mono().unwrap().to_vec(), &expected, 2000..42000);
        assert!(snr > 130.0, "SNR {snr:.1} dB");
    }

    /// Regression: strongly decimating awkward ratios built a 4097-row table
    /// of ~10^5 taps (48000 -> 101 Hz: 2.9 GB, 12 s, kept in the cache;
    /// 192000 -> 101 Hz aborted on allocation). The interpolated table now has
    /// `ceil(4096 · L / M)` rows, oversized kernels are not cached, and
    /// impossible sizes are an error.
    #[test]
    fn test_steep_decimation_table_is_bounded() {
        for (in_sr, out_sr) in [(48000u32, 101u32), (192_000, 101), (96000, 7919)] {
            let kernel = Kernel::new(
                in_sr,
                out_sr,
                kernel_kind(ResamplingQuality::High, in_sr, out_sr),
            )
            .unwrap();
            assert!(
                kernel.size_bytes() < 64 << 20,
                "{in_sr} -> {out_sr}: {} bytes",
                kernel.size_bytes()
            );
        }
        // A band-limited input survives decimation (2 s of a 20 Hz tone,
        // well inside the 50.5 Hz output Nyquist).
        let in_sr = 48000u32;
        let x = sine(2 * in_sr as usize, 20.0, f64::from(in_sr), 0.4);
        let y = resample(
            &mono(x, SampleRate::new(in_sr).unwrap()),
            sample_rate!(101),
            ResamplingQuality::High,
        )
        .unwrap();
        let y = y.as_mono().unwrap().to_vec();
        assert_eq!(y.len(), 202);
        let expected = sine(202, 20.0, 101.0, 0.4);
        // The filter spans ~4.6 s of input, so only the middle is free of
        // edge effects; compare there.
        let snr = snr_db(&y, &expected, 95..107);
        assert!(snr > 120.0, "SNR {snr:.1} dB");
        // Impossible ratios are an error, not an allocation failure.
        let tiny = mono(vec![0.5; 16], SampleRate::new(4_000_000_000).unwrap());
        assert!(resample(&tiny, sample_rate!(1), ResamplingQuality::High).is_err());
        assert!(
            StreamingResampler::<f64>::new(
                SampleRate::new(4_000_000_000).unwrap(),
                sample_rate!(1),
                channels!(1),
                ResamplingQuality::Medium,
            )
            .is_err()
        );
    }

    /// A kernel larger than the cache budget is not retained.
    #[test]
    fn test_oversized_kernel_is_not_cached() {
        // 192 kHz -> 8 Hz needs a ~72 MB table.
        let k = cached_kernel(192_000, 8, ResamplingQuality::High).unwrap();
        assert!(k.size_bytes() > KERNEL_CACHE_BYTES);
        KERNEL_CACHE.with(|c| assert!(c.borrow().iter().all(|(_, e)| !Arc::ptr_eq(e, &k))));
    }

    fn sine_mono(n: usize, sample_rate: SampleRate) -> AudioSamples<'static, f32> {
        let samples = (0..n)
            .map(|x| {
                (x as f32 * 2.0 * std::f32::consts::PI * 440.0 / sample_rate.get() as f32).sin()
            })
            .collect::<Vec<f32>>();
        AudioSamples::new_mono(Array1::from_vec(samples), sample_rate).unwrap()
    }

    /// The streaming path uses the same kernel and positions as the offline
    /// path: same total length, same samples (no delay in either).
    #[test]
    fn test_streaming_resampler_matches_offline() {
        let in_sr = sample_rate!(44100);
        let total_frames = 44100; // 1 second
        let x = sine(total_frames, 440.0, 44100.0, 0.1);
        let audio = mono(x.clone(), in_sr);

        for out_sr in [
            sample_rate!(22050),
            sample_rate!(48000),
            sample_rate!(16000),
        ] {
            for quality in [
                ResamplingQuality::Fast,
                ResamplingQuality::Medium,
                ResamplingQuality::High,
            ] {
                let mut streaming =
                    StreamingResampler::<f64>::new(in_sr, out_sr, channels!(1), quality).unwrap();
                // Deliberately odd chunk size.
                const CHUNK: usize = 333;
                let mut streamed: Vec<f64> = Vec::new();
                for chunk in x.chunks(CHUNK) {
                    let piece = mono(chunk.to_vec(), in_sr);
                    if let Some(out) = streaming.process_chunk(&piece).unwrap() {
                        streamed.extend(out.as_mono().unwrap().iter());
                    }
                }
                if let Some(out) = streaming.finish().unwrap() {
                    streamed.extend(out.as_mono().unwrap().iter());
                }

                let offline = resample(&audio, out_sr, quality).unwrap();
                let offline = offline.as_mono().unwrap().to_vec();
                assert_eq!(streamed.len(), offline.len(), "{quality:?} -> {out_sr}");
                // Fast upsampling offline runs in f32.
                let tol = if quality == ResamplingQuality::Fast {
                    1e-6
                } else {
                    1e-12
                };
                for (i, (a, b)) in streamed.iter().zip(&offline).enumerate() {
                    assert!(
                        (a - b).abs() < tol,
                        "{quality:?} -> {out_sr} at {i}: {a} vs {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_streaming_resampler_channel_mismatch_errors() {
        let mut streaming = StreamingResampler::<f32>::new(
            sample_rate!(44100),
            sample_rate!(48000),
            channels!(1),
            ResamplingQuality::Fast,
        )
        .unwrap();

        let stereo = AudioSamples::new_multi_channel(
            ndarray::array![[0.0f32, 0.0], [0.0, 0.0]],
            sample_rate!(44100),
        )
        .unwrap();

        assert!(streaming.process_chunk(&stereo).is_err());
    }

    #[test]
    fn test_streaming_resampler_reset_allows_reuse() {
        let in_sr = sample_rate!(44100);
        let out_sr = sample_rate!(48000);
        let mut streaming =
            StreamingResampler::<f32>::new(in_sr, out_sr, channels!(1), ResamplingQuality::Medium)
                .unwrap();

        let chunk = sine_mono(2048, in_sr);
        let first: Vec<f32> = streaming
            .process_chunk(&chunk)
            .unwrap()
            .map(|a| a.as_mono().unwrap().to_vec())
            .unwrap_or_default();

        streaming.reset();

        // After reset, the same chunk gives the same output as a fresh stream.
        let again: Vec<f32> = streaming
            .process_chunk(&chunk)
            .unwrap()
            .map(|a| a.as_mono().unwrap().to_vec())
            .unwrap_or_default();
        assert_eq!(first, again);
        assert!(!first.is_empty());
    }
}
