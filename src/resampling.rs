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
//! is anti-aliased and upsampling is anti-imaged. [`resample`] computes in the
//! sample type's float type ([`AudioSample::Float`]: `f32` for `u8`, `i16`,
//! `I24` and `f32`, `f64` for `i32` and `f64`), reading the input in place and
//! writing the output buffer directly; [`StreamingResampler`] computes in `f64`.
//!
//! | Quality | Passband edge | Stopband | Use |
//! |---|---|---|---|
//! | `Fast` (downsampling) | 0.80 · Nyquist | 70 dB from Nyquist | cheap anti-aliased decimation |
//! | `Fast` (upsampling) | linear interpolation | none | cheapest; images are not suppressed |
//! | `Medium` | 0.90 · Nyquist | 100 dB from Nyquist | general use |
//! | `High` | 0.913 · Nyquist | 125 dB from Nyquist | analysis; close to `librosa.resample` (`soxr_hq`) |
//!
//! (Nyquist of the lower of the two rates.) The filter runs in two stages:
//! the low-pass with the passband, transition band and attenuation above at
//! the input rate or twice it, evaluated by FFT, then a short interpolator
//! (12-48 taps, 30 dB more attenuation) from that rate to the exact output
//! positions. 1 : 2 upsampling and decimation by a power of two up to 64 are
//! the first stage alone. The interpolator has one exact filter phase per
//! output phase when the ratio `L / M` (in lowest terms) has at most 2^15
//! output phases, otherwise a table of 4096 phases per sample, linearly
//! interpolated. Ratios whose low-pass is longer than 2^14 taps (decimation by
//! more than ~170 : 1) run as one filter evaluated directly (exact phases for
//! small `L`, otherwise the interpolated table). Filters and tables are cached
//! per thread (a few recent ratios, bounded in size). Ratios needing a table of
//! more than 2^25 coefficients (decimation by more than ~10^5 : 1) are
//! rejected with an error.
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
    AudioFloat, AudioSample, AudioSampleError, AudioSampleResult, AudioSamples,
    AudioTypeConversion, ConvertTo, ParameterError,
    operations::types::ResamplingQuality,
    repr::{ChannelCount, SampleRate, SampleType},
    traits::StandardSample,
};
use non_empty_slice::NonEmptyVec;
use std::cell::RefCell;
use std::sync::Arc;

mod engine;

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

/// Passband edge (fraction of the lower Nyquist frequency) and stopband
/// attenuation in dB of a quality tier's windowed-sinc filter, or `None` when
/// the tier uses linear interpolation (`Fast` upsampling). Used by the
/// `educational` explanations so they report the filter that actually runs.
#[cfg(feature = "educational")]
pub(crate) fn quality_filter_spec(
    quality: ResamplingQuality,
    in_sr: u32,
    out_sr: u32,
) -> Option<(f64, f64)> {
    match kernel_kind(quality, in_sr, out_sr) {
        KernelKind::Sinc(spec) => Some((spec.passband, spec.attenuation_db)),
        KernelKind::Linear => None,
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
    /// Identity of an interpolated table (`None` for exact tables), so
    /// kernels for different ratios can share it.
    table_key: Option<TableKey>,
    /// Row-major coefficients.
    coeffs: Arc<[f64]>,
    /// Whether the signal end is extended by holding the last sample.
    hold_last: bool,
    /// Use the AVX2 + FMA dot product.
    avx2_fma: bool,
}

/// Identifies an interpolated (non-exact) coefficient table. Such a table
/// depends only on the quality tier, the relative bandwidth `min(L/M, 1)` and
/// the row count, not on the exact ratio: every upsampling ratio of one
/// quality shares one table (e.g. pitch shifts down by any amount).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TableKey {
    quality: ResamplingQuality,
    stage: Stage,
    /// `lower` (single-stage tables) or the transition band (stage-B
    /// tables), as `f64` bits.
    param_bits: u64,
    /// Half-width (stage-B tables; 0 otherwise).
    half: usize,
    phases: usize,
}

/// Samples per lower-rate sample of the prototype filter table.
const PROTO_RES: usize = 8192;

/// Continuous Kaiser-windowed sinc low-pass for relative bandwidth `lower`:
/// its half-length in lower-rate samples and `h(tau)`, `tau` in input samples.
///
/// `h_lower(tau) = lower · h_1(lower · tau)`: the filter for any ratio is the
/// `lower = 1` prototype stretched in time.
fn sinc_kernel(spec: FilterSpec, lower: f64) -> (f64, impl Fn(f64) -> f64) {
    // -6 dB point midway through the transition band, as a fraction of the
    // input Nyquist frequency.
    let cutoff = 0.5 * (spec.passband + spec.stopband) * lower;
    // Kaiser length estimate (in lower-rate samples) for the requested
    // attenuation and transition width.
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
    (half_lower, h)
}

/// The `lower = 1` filter of a quality tier sampled at `1 / PROTO_RES` over
/// `[0, half_lower]` (it is even). The window does not reach zero at the edge
/// of the support (the filter steps to zero there), so the last entry holds
/// the left limit, not the value 0 at the edge itself. Built once per process
/// and quality (≈ 6 MB for `High`).
fn prototype(quality: ResamplingQuality, spec: FilterSpec) -> &'static [f64] {
    static TABLES: [std::sync::OnceLock<Vec<f64>>; 3] = [const { std::sync::OnceLock::new() }; 3];
    let slot = match quality {
        ResamplingQuality::Fast => 0,
        ResamplingQuality::Medium => 1,
        ResamplingQuality::High => 2,
    };
    TABLES[slot].get_or_init(|| {
        let (half_lower, h) = sinc_kernel(spec, 1.0);
        let n = half_lower as usize * PROTO_RES;
        let mut g: Vec<f64> = (0..n).map(|i| h(i as f64 / PROTO_RES as f64)).collect();
        g.push(h(half_lower * (1.0 - f64::EPSILON)));
        g
    })
}

/// Linear interpolation in a [`prototype`] table (zero from the edge of the
/// support on, like the filter itself).
#[inline]
fn prototype_eval(g: &[f64], u: f64) -> f64 {
    let a = u.abs() * PROTO_RES as f64;
    let i = a as usize;
    if i + 1 >= g.len() {
        return 0.0;
    }
    let f = a - i as f64;
    g[i] + f * (g[i + 1] - g[i])
}

impl Kernel {
    /// Builds the table for `in_sr -> out_sr`. An interpolated table is taken
    /// from `shared` when it returns one for the table's [`TableKey`].
    ///
    /// Exact tables evaluate the filter directly. Interpolated tables are
    /// sampled from the quality's [`prototype`] (two linear interpolations
    /// instead of a Bessel series per coefficient; each adds < 1e-8 per
    /// coefficient), which makes building one cheap.
    ///
    /// # Errors
    /// [`ParameterError`] if the table would exceed [`MAX_KERNEL_ENTRIES`]
    /// coefficients (decimation by more than ~10^5 : 1).
    fn new(
        in_sr: u32,
        out_sr: u32,
        quality: ResamplingQuality,
        shared: &dyn Fn(TableKey) -> Option<Arc<[f64]>>,
    ) -> AudioSampleResult<Self> {
        let kind = kernel_kind(quality, in_sr, out_sr);
        let g = gcd(u64::from(in_sr), u64::from(out_sr));
        let l = u64::from(out_sr) / g;
        let m = u64::from(in_sr) / g;
        // Bandwidth of the lower rate relative to the input rate.
        let lower = (l as f64 / m as f64).min(1.0);

        let half = match kind {
            KernelKind::Linear => 1,
            KernelKind::Sinc(spec) => {
                let support = sinc_kernel(spec, lower).0 / lower;
                // Round the half-width up to a multiple of 8 so rows are a
                // whole number of 16-wide SIMD blocks (the extra taps are 0).
                // `support` is at most ~4e11 (u32 rates), so this cannot wrap
                // on 64-bit targets; the size check below rejects it anyway.
                (support.ceil() as usize).next_multiple_of(8)
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
        let table_key = (!exact).then_some(TableKey {
            quality,
            stage: Stage::Single,
            param_bits: lower.to_bits(),
            half: 0,
            phases,
        });
        let build = |h: &dyn Fn(f64) -> f64| -> Arc<[f64]> {
            let mut coeffs = Vec::with_capacity(rows * taps);
            for row in 0..rows {
                let phi = row as f64 / denom;
                for j in 0..taps {
                    let offset = j as f64 - (half as f64 - 1.0);
                    coeffs.push(h(phi - offset));
                }
            }
            coeffs.into()
        };
        let coeffs = match (table_key.and_then(shared), kind) {
            (Some(table), _) => table,
            (None, KernelKind::Linear) => build(&|tau: f64| (1.0 - tau.abs()).max(0.0)),
            (None, KernelKind::Sinc(spec)) if exact => build(&sinc_kernel(spec, lower).1),
            (None, KernelKind::Sinc(spec)) => {
                let g = prototype(quality, spec);
                build(&|tau: f64| lower * prototype_eval(g, lower * tau))
            }
        };
        Ok(Self {
            l,
            m,
            half,
            taps,
            exact,
            phases,
            table_key,
            coeffs,
            hold_last: matches!(kind, KernelKind::Linear),
            avx2_fma: has_avx2_fma_cached(),
        })
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

// ---------------------------------------------------------------------------
// Two-stage conversion
// ---------------------------------------------------------------------------
//
// A single polyphase filter evaluated directly costs `taps` multiply-adds per
// output sample, and `taps` is large (≈ 190 / lower for `High`) because the
// transition band is narrow. Most ratios therefore run in two stages:
//
// * Stage A is the same Kaiser-windowed sinc low-pass (same passband edge,
//   transition band and attenuation) sampled at `up · in_rate` (`up` = 1 or
//   2): a plain FIR or a 1 : 2 interpolator. It runs by FFT, so its cost
//   hardly depends on its length.
// * Stage B interpolates stage A's output `y1` at the exact output positions.
//   `y1` has no content (above the attenuation) between the lower Nyquist
//   frequency `f_low / 2` and its first image at `up · in_rate - f_low / 2`,
//   so stage B only needs a transition band that wide: a short filter
//   (12-48 taps), flat over all of stage A's band, with its cut-off at
//   `y1`'s Nyquist frequency.
//
// The cascade has stage A's passband edge and transition band; its stopband
// is stage A's attenuation below the image band and stage B's (set higher,
// see [`STAGE_B_EXTRA_DB`]) in it. Stage A's output is computed outside the
// signal too (the zero-padded input filtered by stage A), so the edges match
// the single filter as well.

/// Which filter of a conversion a kernel is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Stage {
    /// The whole conversion in one polyphase filter.
    Single,
    /// Stage A of a two-stage conversion (the low-pass at `up · in_rate`).
    A,
    /// Stage B of a two-stage conversion (the interpolator to `out_rate`).
    B,
}

/// Stage B's stopband attenuation above the quality tier's. Stage B's
/// passband ripple adds to stage A's; 30 dB more keeps the cascade's
/// passband error at the single filter's level (measured on tones across the
/// passband).
const STAGE_B_EXTRA_DB: f64 = 30.0;
/// Cost of one stage-A output sample (FFT work, spectrum products and
/// copies) in units of one stage-B multiply-add, measured with AVX2: used to
/// choose stage A's rate.
const STAGE_A_COST: f64 = 20.0;
/// Decimation factors (powers of two) up to this run as a single FFT stage.
const FFT_MAX_FACTOR: u64 = 64;
/// Largest stage-A half-length (convolution-rate samples) evaluated by FFT;
/// longer filters (decimation by more than ~170 : 1) run directly.
const FFT_MAX_HALF: usize = 1 << 14;

/// How a conversion is evaluated.
#[derive(Debug, Clone)]
enum Plan {
    /// One polyphase filter evaluated directly (linear interpolation, and
    /// ratios whose stage-A filter is too long for the FFT path).
    Direct(Arc<Kernel>),
    /// One filter, 1 : 2 upsampling or downsampling by a power of two
    /// (`kernel` has `l = up`, `m = down`), evaluated by FFT.
    Fft {
        kernel: Arc<Kernel>,
        up: usize,
        down: usize,
    },
    /// Stage A at `up · in_rate` (FFT), then stage B.
    TwoStage {
        a: Arc<Kernel>,
        up: usize,
        b: Arc<Kernel>,
    },
}

/// Shape of stage B when stage A runs at `up · in_sr`: `(l, m, half,
/// transition)` with `l / m = out_sr / (up · in_sr)` in lowest terms, the
/// half-width (a multiple of 2) and the transition band in radians per
/// stage-A sample.
fn stage_b_shape(in_sr: u32, out_sr: u32, spec: FilterSpec, up: u64) -> (u64, u64, usize, f64) {
    let fs_a = up * u64::from(in_sr);
    let g = gcd(fs_a, u64::from(out_sr));
    let (l, m) = (u64::from(out_sr) / g, fs_a / g);
    let r = f64::from(in_sr.min(out_sr)) / fs_a as f64;
    let attenuation = spec.attenuation_db + STAGE_B_EXTRA_DB;
    let transition = 2.0 * std::f64::consts::PI * (1.0 - r);
    let length = (attenuation - 7.95) / (2.285 * transition);
    let half = ((length / 2.0).ceil() as usize).next_multiple_of(2);
    (l, m, half, transition)
}

/// Most phases of an exact stage-B table. Its rows are stored in the order
/// the outputs visit them (see `engine::Interp`), so even a table larger
/// than the L2 cache is read sequentially.
const STAGE_B_MAX_EXACT_PHASES: u64 = 1 << 15;
/// Most coefficients of an exact stage-B table (8 MiB in `f64`).
const STAGE_B_MAX_EXACT_ENTRIES: u64 = 1 << 20;

/// Whether stage B with `l` phases and `taps` taps gets an exact table.
fn stage_b_exact(l: u64, taps: usize) -> bool {
    l <= STAGE_B_MAX_EXACT_PHASES && l.saturating_mul(taps as u64) <= STAGE_B_MAX_EXACT_ENTRIES
}

/// Stage A's rate factor for a two-stage conversion: 2 when upsampling;
/// when downsampling, 1 or 2, whichever the cost model prefers (close to
/// 1 : 1, stage B at the input rate needs many taps).
fn stage_a_up(in_sr: u32, out_sr: u32, spec: FilterSpec) -> u64 {
    if out_sr > in_sr {
        return 2;
    }
    let ratio = f64::from(out_sr) / f64::from(in_sr);
    let cost = |up: u64| {
        let (l, _, half, _) = stage_b_shape(in_sr, out_sr, spec, up);
        let rows = if stage_b_exact(l, 2 * half) { 1.0 } else { 2.0 };
        up as f64 * STAGE_A_COST + ratio * (2 * half) as f64 * rows
    };
    if cost(2) < cost(1) { 2 } else { 1 }
}

/// Half-width (input samples, a multiple of 8) of the single-stage filter
/// for the ratio `l / m`, as `f64` so that absurd ratios do not overflow.
fn stage_a_half(spec: FilterSpec, l: u64, m: u64) -> f64 {
    let lower = (l as f64 / m as f64).min(1.0);
    let half = (sinc_kernel(spec, lower).0 / lower).ceil();
    (half / 8.0).ceil() * 8.0
}

/// Chooses how to evaluate `in_sr -> out_sr` at `quality`.
///
/// # Errors
/// See [`Kernel::new`].
fn plan(in_sr: u32, out_sr: u32, quality: ResamplingQuality) -> AudioSampleResult<Plan> {
    let KernelKind::Sinc(spec) = kernel_kind(quality, in_sr, out_sr) else {
        return Ok(Plan::Direct(cached_kernel(in_sr, out_sr, quality)?));
    };
    let g = gcd(u64::from(in_sr), u64::from(out_sr));
    let (l, m) = (u64::from(out_sr) / g, u64::from(in_sr) / g);
    let power = |f: u64| f.is_power_of_two() && (2..=FFT_MAX_FACTOR).contains(&f);
    // Upsampling by more than 2 runs as 1 : 2 then a short interpolator:
    // a single FFT stage would need FFT sizes (and work buffers) of several
    // times the filter length at the output rate, more than short inputs
    // need.
    if (m == 1 && l == 2) || (l == 1 && power(m)) {
        let kernel = cached_kernel(in_sr, out_sr, quality)?;
        if kernel.half * l as usize <= FFT_MAX_HALF {
            return Ok(Plan::Fft {
                kernel,
                up: l as usize,
                down: m as usize,
            });
        }
        return Ok(Plan::Direct(kernel));
    }
    if stage_a_half(spec, l, m) * stage_a_up(in_sr, out_sr, spec) as f64 > FFT_MAX_HALF as f64 {
        return Ok(Plan::Direct(cached_kernel(in_sr, out_sr, quality)?));
    }
    let a = cached_stage_kernel(in_sr, out_sr, quality, Stage::A)?;
    let b = cached_stage_kernel(in_sr, out_sr, quality, Stage::B)?;
    Ok(Plan::TwoStage {
        up: a.l as usize,
        a,
        b,
    })
}

impl Kernel {
    /// Stage A of `in_sr -> out_sr`: the quality's low-pass for the lower of
    /// the two rates, with `up` exact rows (`l = up`, `m = 1`). For
    /// upsampling this is the single-stage kernel of `in_sr -> 2 in_sr`.
    fn stage_a(in_sr: u32, out_sr: u32, quality: ResamplingQuality) -> Self {
        let KernelKind::Sinc(spec) = kernel_kind(quality, in_sr, out_sr) else {
            unreachable!("two-stage plans are only made for windowed-sinc kernels")
        };
        let up = stage_a_up(in_sr, out_sr, spec);
        let g = gcd(u64::from(in_sr), u64::from(out_sr));
        let (l, m) = (u64::from(out_sr) / g, u64::from(in_sr) / g);
        let lower = (l as f64 / m as f64).min(1.0);
        let (_, h) = sinc_kernel(spec, lower);
        let half = stage_a_half(spec, l, m) as usize;
        let taps = 2 * half;
        let mut coeffs = Vec::with_capacity(up as usize * taps);
        for row in 0..up {
            let phi = row as f64 / up as f64;
            for j in 0..taps {
                coeffs.push(h(phi - (j as f64 - (half as f64 - 1.0))));
            }
        }
        Self {
            l: up,
            m: 1,
            half,
            taps,
            exact: true,
            phases: 1,
            table_key: None,
            coeffs: coeffs.into(),
            hold_last: false,
            avx2_fma: has_avx2_fma_cached(),
        }
    }

    /// Stage B of `in_sr -> out_sr`: from stage A's rate `up · in_sr` to
    /// `out_sr`, flat up to the lower Nyquist frequency `f_low / 2` and
    /// attenuating from `up · in_sr - f_low / 2` on. Its cut-off (−6 dB) is
    /// therefore stage A's Nyquist frequency; its half-width follows from
    /// Kaiser's length estimate (see [`stage_b_shape`]).
    fn stage_b(
        in_sr: u32,
        out_sr: u32,
        quality: ResamplingQuality,
        shared: &dyn Fn(TableKey) -> Option<Arc<[f64]>>,
    ) -> Self {
        let KernelKind::Sinc(spec) = kernel_kind(quality, in_sr, out_sr) else {
            unreachable!("two-stage plans are only made for windowed-sinc kernels")
        };
        let up = stage_a_up(in_sr, out_sr, spec);
        let (l, m, half, transition) = stage_b_shape(in_sr, out_sr, spec, up);
        // The rounded-up length allows more attenuation for the same
        // transition band: use all of it (Kaiser's estimate solved for A).
        let attenuation = 2.285 * transition * (2 * half) as f64 + 7.95;
        let taps = 2 * half;
        let exact = stage_b_exact(l, taps);
        let phases = INTERP_PHASES;
        let (rows, denom) = if exact {
            (l as usize, l as f64)
        } else {
            (phases + 1, phases as f64)
        };
        let table_key = (!exact).then_some(TableKey {
            quality,
            stage: Stage::B,
            param_bits: transition.to_bits(),
            half,
            phases,
        });
        let coeffs = table_key.and_then(shared).unwrap_or_else(|| {
            let support = half as f64;
            let beta = kaiser_beta(attenuation);
            let i0_beta = bessel_i0(beta);
            let h = |tau: f64| -> f64 {
                let r = tau / support;
                if r.abs() >= 1.0 {
                    return 0.0;
                }
                let x = std::f64::consts::PI * tau;
                let sinc = if x == 0.0 { 1.0 } else { x.sin() / x };
                sinc * bessel_i0(beta * (1.0 - r * r).sqrt()) / i0_beta
            };
            let mut coeffs = Vec::with_capacity(rows * taps);
            for row in 0..rows {
                let phi = row as f64 / denom;
                for j in 0..taps {
                    coeffs.push(h(phi - (j as f64 - (half as f64 - 1.0))));
                }
            }
            coeffs.into()
        });
        Self {
            l,
            m,
            half,
            taps,
            exact,
            phases,
            table_key,
            coeffs,
            hold_last: false,
            avx2_fma: has_avx2_fma_cached(),
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

/// [`has_avx2_fma`], detected once per process.
fn has_avx2_fma_cached() -> bool {
    static DETECTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DETECTED.get_or_init(has_avx2_fma)
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

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
struct KernelKey {
    in_sr: u32,
    out_sr: u32,
    quality: ResamplingQuality,
    stage: Stage,
}

/// Most kernels kept per thread (kernels sharing an interpolated table cost
/// only a few bytes each; [`KERNEL_CACHE_BYTES`] is the real bound).
const KERNEL_CACHE_ENTRIES: usize = 32;
/// Most coefficient memory kept per thread. A kernel larger than this on its
/// own is used for the call that built it and then dropped, never cached.
const KERNEL_CACHE_BYTES: usize = 64 << 20;

thread_local! {
    /// Recently used kernels, most recent first. Small and bounded: distinct
    /// ratios (e.g. pitch shifts by arbitrary steps) can each need a table of
    /// several MB, so an unbounded map grows without limit.
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
/// of coefficients remain (a table shared by several kernels counts once).
/// A miss that needs an interpolated table already held by a cached kernel
/// (same [`TableKey`]) reuses it instead of building a new one.
///
/// # Errors
/// See [`Kernel::new`].
fn cached_kernel(
    in_sr: u32,
    out_sr: u32,
    quality: ResamplingQuality,
) -> AudioSampleResult<Arc<Kernel>> {
    cached_stage_kernel(in_sr, out_sr, quality, Stage::Single)
}

/// [`cached_kernel`] for one stage of a conversion.
///
/// # Errors
/// See [`Kernel::new`].
fn cached_stage_kernel(
    in_sr: u32,
    out_sr: u32,
    quality: ResamplingQuality,
    stage: Stage,
) -> AudioSampleResult<Arc<Kernel>> {
    let key = KernelKey {
        in_sr,
        out_sr,
        quality,
        stage,
    };
    KERNEL_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        if let Some(pos) = cache.iter().position(|(k, _)| *k == key) {
            let entry = cache.remove(pos);
            let kernel = Arc::clone(&entry.1);
            cache.insert(0, entry);
            return Ok(kernel);
        }
        let shared = |table: TableKey| {
            cache
                .iter()
                .find(|(_, k)| k.table_key == Some(table))
                .map(|(_, k)| Arc::clone(&k.coeffs))
        };
        let kernel = Arc::new(match stage {
            Stage::Single => Kernel::new(in_sr, out_sr, quality, &shared)?,
            Stage::A => Kernel::stage_a(in_sr, out_sr, quality),
            Stage::B => Kernel::stage_b(in_sr, out_sr, quality, &shared),
        });
        cache.insert(0, (key, Arc::clone(&kernel)));
        let mut total = 0;
        let mut counted: Vec<*const f64> = Vec::new();
        let keep = cache
            .iter()
            .position(|(_, k)| {
                let table = k.coeffs.as_ptr();
                if !counted.contains(&table) {
                    counted.push(table);
                    total += k.size_bytes();
                }
                total > KERNEL_CACHE_BYTES
            })
            .unwrap_or(cache.len())
            .min(KERNEL_CACHE_ENTRIES);
        cache.truncate(keep);
        Ok(kernel)
    })
}

/// Incremental direct evaluation of one polyphase kernel: the streaming path,
/// and the whole-buffer path for [`Plan::Direct`].
///
/// Each channel buffer holds input samples from absolute index `buf_start`
/// on. The first pushed sample has index `in_start`; `prefill` zeros before
/// it stand for the signal before the start. Outputs are numbered from
/// `first_out` (negative for a stage A that also produces the part of its
/// output before the signal start, which stage B needs).
#[derive(Debug)]
struct PolyphaseState {
    kernel: Arc<Kernel>,
    buffers: Vec<Vec<f64>>,
    /// Absolute input index of `buffers[c][0]`.
    buf_start: i64,
    /// Next output index to produce.
    next_out: i64,
    /// Absolute index one past the last input sample received.
    in_end: i64,
    /// Input samples received per channel.
    received: u64,
    /// Last input sample per channel (for `hold_last` kernels).
    last: Vec<f64>,
    in_start: i64,
    first_out: i64,
    prefill: usize,
}

impl PolyphaseState {
    /// State for a signal starting at index 0 (zeros before it), producing
    /// outputs from index 0.
    fn new(kernel: Arc<Kernel>, channels: usize) -> Self {
        let half = kernel.half;
        Self::with_layout(kernel, channels, 0, 0, half)
    }

    fn with_layout(
        kernel: Arc<Kernel>,
        channels: usize,
        in_start: i64,
        first_out: i64,
        prefill: usize,
    ) -> Self {
        let mut state = Self {
            buffers: vec![Vec::new(); channels],
            buf_start: 0,
            next_out: 0,
            in_end: 0,
            received: 0,
            last: vec![0.0; channels],
            kernel,
            in_start,
            first_out,
            prefill,
        };
        state.reset();
        state
    }

    fn reset(&mut self) {
        for b in &mut self.buffers {
            b.clear();
            b.resize(self.prefill, 0.0);
        }
        self.buf_start = self.in_start - self.prefill as i64;
        self.next_out = self.first_out;
        self.in_end = self.in_start;
        self.received = 0;
        self.last.fill(0.0);
    }

    /// Appends one block of input (one slice per channel, equal lengths).
    fn push<S: AsRef<[f64]>>(&mut self, input: &[S]) {
        for ((buf, src), last) in self.buffers.iter_mut().zip(input).zip(&mut self.last) {
            let src = src.as_ref();
            buf.extend_from_slice(src);
            if let Some(&v) = src.last() {
                *last = v;
            }
        }
        let n = input.first().map_or(0, |c| c.as_ref().len());
        self.received += n as u64;
        self.in_end += n as i64;
    }

    /// Produces every output sample whose input window lies before absolute
    /// index `available` (exclusive), stopping at output index `limit`, then
    /// discards input that is no longer needed.
    fn produce(&mut self, available: i64, limit: Option<i64>) -> Vec<Vec<f64>> {
        let kernel = Arc::clone(&self.kernel);
        let half = kernel.half as i64;
        let taps = kernel.taps;
        let mut out: Vec<Vec<f64>> = vec![Vec::new(); self.buffers.len()];
        // Step the position incrementally: output k + 1 is M / L input
        // samples after output k.
        let (step_int, step_frac) = ((kernel.m / kernel.l) as i64, kernel.m % kernel.l);
        let num = i128::from(self.next_out) * i128::from(kernel.m);
        let l = i128::from(kernel.l);
        let (mut n0, mut p) = (num.div_euclid(l) as i64, num.rem_euclid(l) as u64);
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
    fn process<S: AsRef<[f64]>>(&mut self, input: &[S]) -> Vec<Vec<f64>> {
        self.push(input);
        self.produce(self.in_end, None)
    }

    /// Flushes the tail: extends the input (zeros, or the held last sample for
    /// linear interpolation) and produces the remaining outputs, so that the
    /// total output is exactly `ceil(received · L / M)` samples.
    fn finish(&mut self) -> Vec<Vec<f64>> {
        let total = self.kernel.output_len(self.received) as i64;
        self.finish_with(self.kernel.half, total)
    }

    /// Extends the input by `pad` samples (zeros, or the held last sample)
    /// and produces the outputs before index `end`.
    fn finish_with(&mut self, pad: usize, end: i64) -> Vec<Vec<f64>> {
        let hold = self.kernel.hold_last;
        for (buf, &last) in self.buffers.iter_mut().zip(&self.last) {
            let v = if hold { last } else { 0.0 };
            buf.extend(std::iter::repeat_n(v, pad));
        }
        self.produce(self.in_end + pad as i64, Some(end))
    }
}

/// Streaming evaluation of a [`Plan::TwoStage`]: stage A's output (from index
/// `-half_b` on, i.e. including the part before the signal start) feeds
/// stage B.
#[derive(Debug)]
struct ChainState {
    a: PolyphaseState,
    b: PolyphaseState,
    /// Overall output samples per period.
    l: u64,
    /// Overall input samples per period.
    m: u64,
}

impl ChainState {
    fn new(a: Arc<Kernel>, b: Arc<Kernel>, channels: usize, l: u64, m: u64) -> Self {
        let up = a.l as i64;
        let first = -(b.half as i64);
        // Stage A output `first` sits at input position floor(first / up) and
        // needs the input from `half_a - 1` samples before that.
        let prefill = (a.half as i64 - 1 - first.div_euclid(up)) as usize;
        Self {
            a: PolyphaseState::with_layout(a, channels, 0, first, prefill),
            b: PolyphaseState::with_layout(b, channels, first, 0, 0),
            l,
            m,
        }
    }

    fn reset(&mut self) {
        self.a.reset();
        self.b.reset();
    }

    fn process(&mut self, input: &[&[f64]]) -> Vec<Vec<f64>> {
        let y1 = self.a.process(input);
        self.b.process(&y1)
    }

    fn finish(&mut self) -> Vec<Vec<f64>> {
        let n = self.a.received;
        let up = self.a.kernel.l;
        let half_b = self.b.kernel.half as u64;
        // Stage B's last output needs stage A's output up to
        // `up · n + half_b - 1`, which needs the input up to
        // `n + ceil(half_b / up) + half_a - 1`.
        let end_a = (up * n + half_b) as i64;
        let pad = self.a.kernel.half + half_b.div_ceil(up) as usize + 1;
        let y1 = self.a.finish_with(pad, end_a);
        self.b.push(&y1);
        let total = (u128::from(n) * u128::from(self.l)).div_ceil(u128::from(self.m)) as i64;
        self.b.produce(self.b.in_end, Some(total))
    }
}

/// Streaming state for any [`Plan`].
#[derive(Debug)]
enum StreamState {
    Single(PolyphaseState),
    Chain(ChainState),
}

impl StreamState {
    fn new(plan: Plan, channels: usize, in_sr: u32, out_sr: u32) -> Self {
        match plan {
            Plan::Direct(kernel) | Plan::Fft { kernel, .. } => {
                Self::Single(PolyphaseState::new(kernel, channels))
            }
            Plan::TwoStage { a, b, .. } => {
                let g = gcd(u64::from(in_sr), u64::from(out_sr));
                let (l, m) = (u64::from(out_sr) / g, u64::from(in_sr) / g);
                Self::Chain(ChainState::new(a, b, channels, l, m))
            }
        }
    }

    fn process(&mut self, input: &[&[f64]]) -> Vec<Vec<f64>> {
        match self {
            Self::Single(s) => s.process(input),
            Self::Chain(c) => c.process(input),
        }
    }

    fn finish(&mut self) -> Vec<Vec<f64>> {
        match self {
            Self::Single(s) => s.finish(),
            Self::Chain(c) => c.finish(),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Single(s) => s.reset(),
            Self::Chain(c) => c.reset(),
        }
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
        KernelKind::Sinc(_) => match <T::Float as AudioSample>::SAMPLE_TYPE {
            SampleType::F32 => resample_sinc::<T, f32>(audio, target_sample_rate, quality),
            _ => resample_sinc::<T, f64>(audio, target_sample_rate, quality),
        },
    }
}

/// Number of output samples for `n` input samples: `ceil(n · out / in)`.
fn output_len(in_sr: u32, out_sr: u32, n: usize) -> usize {
    (n as u128 * u128::from(out_sr)).div_ceil(u128::from(in_sr)) as usize
}

/// Windowed-sinc resampling in `F` (the sample type's float): the cached
/// engine for the plan writes every channel straight into the output buffer.
fn resample_sinc<T, F>(
    audio: &AudioSamples<T>,
    target_sample_rate: SampleRate,
    quality: ResamplingQuality,
) -> AudioSampleResult<AudioSamples<'static, T>>
where
    T: StandardSample,
    F: engine::RsFloat,
{
    let in_sr = audio.sample_rate().get();
    let out_sr = target_sample_rate.get();
    let Some(engine) = engine::cached_engine::<F>(in_sr, out_sr, quality)? else {
        return resample_direct(audio, target_sample_rate, quality);
    };
    let n_out = output_len(in_sr, out_sr, audio.samples_per_channel().get());
    let channels = audio.num_channels().get() as usize;
    // Every sample is written by the engine: no zero-initialisation pass.
    let mut flat: Vec<T> = Vec::with_capacity(channels * n_out);
    let to_f = |s: T| F::of(s.to_float().as_f64());
    let from_f = |v: F| T::from_float(<T::Float as AudioFloat>::of(v.as_f64()));
    let spare = &mut flat.spare_capacity_mut()[..channels * n_out];
    let mut run = |c: usize, x: &[T]| {
        engine.run(x, &mut spare[c * n_out..(c + 1) * n_out], &to_f, &from_f);
    };
    if let Some(mono) = audio.as_mono() {
        match mono.as_slice() {
            Some(x) => run(0, x),
            None => run(0, &mono.to_vec()),
        }
    } else {
        let multi = audio.as_multi_channel().ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "audio_format",
                "Failed to get multi-channel data",
            ))
        })?;
        for c in 0..channels {
            let row = multi.row(c);
            match row.as_slice() {
                Some(x) => run(c, x),
                None => run(c, &row.to_vec()),
            }
        }
    }
    // SAFETY: `Engine::run` wrote every element of each channel's range, and
    // the ranges cover the first `channels * n_out` elements.
    unsafe { flat.set_len(channels * n_out) };
    if channels == 1 {
        AudioSamples::new_mono(ndarray::Array1::from_vec(flat), target_sample_rate)
    } else {
        let data = ndarray::Array2::from_shape_vec((channels, n_out), flat).map_err(|e| {
            AudioSampleError::Parameter(ParameterError::invalid_value("audio", e.to_string()))
        })?;
        AudioSamples::new_multi_channel(data, target_sample_rate)
    }
}

/// Direct single-stage evaluation in `f64` ([`Plan::Direct`]).
fn resample_direct<T: StandardSample>(
    audio: &AudioSamples<T>,
    target_sample_rate: SampleRate,
    quality: ResamplingQuality,
) -> AudioSampleResult<AudioSamples<'static, T>> {
    let in_sr = audio.sample_rate().get();
    let out_sr = target_sample_rate.get();
    let data = channel_data_f64(audio)?;
    let kernel = cached_kernel(in_sr, out_sr, quality)?;
    let mut state = PolyphaseState::new(kernel, data.len());
    let slices: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
    let mut out = state.process(&slices);
    for (o, tail) in out.iter_mut().zip(state.finish()) {
        o.extend(tail);
    }
    let frames = out[0].len();
    assemble_output(out, frames, audio.num_channels(), target_sample_rate)
}

/// Resamples `(channels, samples)` float rows from `in_sr` to `out_sr`
/// without wrapping them in [`AudioSamples`] and without converting them:
/// the same result as [`resample`] on audio of sample type `F`, truncated or
/// zero-padded to `n_out` samples per row when given
/// (`librosa.util.fix_length`), computing only the samples kept.
///
/// # Errors
/// See [`resample`]; also if a rate is zero or `rows` is empty.
pub(crate) fn resample_float_rows<F: AudioFloat>(
    rows: ndarray::ArrayView2<'_, F>,
    in_sr: u32,
    out_sr: u32,
    quality: ResamplingQuality,
    n_out: Option<usize>,
) -> AudioSampleResult<ndarray::Array2<F>> {
    match F::SAMPLE_TYPE {
        SampleType::F32 => resample_rows_in::<F, f32>(rows, in_sr, out_sr, quality, n_out),
        _ => resample_rows_in::<F, f64>(rows, in_sr, out_sr, quality, n_out),
    }
}

fn resample_rows_in<T: AudioFloat, F: engine::RsFloat>(
    rows: ndarray::ArrayView2<'_, T>,
    in_sr: u32,
    out_sr: u32,
    quality: ResamplingQuality,
    n_out: Option<usize>,
) -> AudioSampleResult<ndarray::Array2<T>> {
    let rate = |v: u32| {
        SampleRate::new(v).ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "sample_rate",
                "must be positive",
            ))
        })
    };
    let (in_rate, out_rate) = (rate(in_sr)?, rate(out_sr)?);
    if rows.is_empty() {
        return Err(AudioSampleError::empty_data("resample"));
    }
    let natural = output_len(in_sr, out_sr, rows.ncols());
    let n_out = n_out.unwrap_or(natural);
    let engine =
        if in_sr == out_sr || matches!(kernel_kind(quality, in_sr, out_sr), KernelKind::Linear) {
            None
        } else {
            engine::cached_engine::<F>(in_sr, out_sr, quality)?
        };
    let Some(engine) = engine else {
        // Identity, linear interpolation and direct plans: through `resample`.
        let audio =
            crate::audio_float::float_rows_to_audio::<T>(rows.to_owned(), in_rate, "resample")?;
        let y = resample(&audio, out_rate, quality)?;
        let y = crate::audio_float::audio_float_rows(&y);
        let mut out = ndarray::Array2::<T>::zeros((rows.nrows(), n_out));
        let n = n_out.min(y.ncols());
        out.slice_mut(ndarray::s![.., ..n])
            .assign(&y.slice(ndarray::s![.., ..n]));
        return Ok(out);
    };
    let mut out = ndarray::Array2::<T>::zeros((rows.nrows(), n_out));
    let keep = n_out.min(natural);
    let to_f = |s: T| F::of(s.as_f64());
    let from_f = |v: F| T::of(v.as_f64());
    for (row, mut dst) in rows.outer_iter().zip(out.outer_iter_mut()) {
        let dst = dst
            .as_slice_mut()
            .expect("rows of a new standard-layout array are contiguous");
        // SAFETY: `MaybeUninit<T>` has the layout of `T`, and the engine only
        // writes valid values.
        let dst = unsafe {
            &mut *(std::ptr::from_mut(&mut dst[..keep]) as *mut [std::mem::MaybeUninit<T>])
        };
        match row.as_slice() {
            Some(x) => engine.run(x, dst, &to_f, &from_f),
            None => engine.run(&row.to_vec(), dst, &to_f, &from_f),
        }
    }
    Ok(out)
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
    state: StreamState,
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
        let (in_sr, out_sr) = (in_sample_rate.get(), out_sample_rate.get());
        let plan = plan(in_sr, out_sr, quality)?;
        Ok(Self {
            state: StreamState::new(plan, nbr_channels, in_sr, out_sr),
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
/// - [`crate::AudioSampleError::Parameter`] if the ratio is not finite and
///   positive, or the resulting target rate rounds to zero or exceeds `u32::MAX`
///   Hz (or see [`resample`]).
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
    if !(ratio.is_finite() && ratio > 0.0) {
        return Err(AudioSampleError::Parameter(ParameterError::invalid_value(
            "ratio",
            format!("Invalid resampling ratio: {ratio}"),
        )));
    }

    let input_sample_rate = audio.sample_rate().get();
    let target = (f64::from(input_sample_rate) * ratio).round();
    // Checked before the integer conversion (which used to wrap for ratios
    // above ~97 000 at 44.1 kHz).
    let target_sample_rate = (target <= f64::from(u32::MAX))
        .then(|| SampleRate::new(target as u32))
        .flatten()
        .ok_or_else(|| {
            AudioSampleError::Parameter(ParameterError::invalid_value(
                "ratio",
                format!(
                    "resampling {input_sample_rate} Hz by {ratio} gives a target rate of \
                     {target} Hz, outside 1..=4294967295"
                ),
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
        // Regression: the target rate was cast through `usize as u32` and
        // wrapped for huge ratios (e.g. 44100 * 1e5 -> a small, wrong rate).
        assert!(resample_by_ratio(&audio, 1e5, ResamplingQuality::Fast).is_err());
        assert!(resample_by_ratio(&audio, f64::INFINITY, ResamplingQuality::Fast).is_err());
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

    /// The whole-buffer evaluation (FFT stage A, stage B over blocks)
    /// computes the same sums as the direct streaming evaluation of the same
    /// plan (only the summation order differs).
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
            (22050, 16000),
            (16000, 22050),
            (22050, 17501),
            (16000, 12699),
            (48000, 44100),
            (44100, 44101),
            (1, 64),
        ];
        for (in_sr, out_sr) in cases {
            for quality in [
                ResamplingQuality::Fast,
                ResamplingQuality::Medium,
                ResamplingQuality::High,
            ] {
                if matches!(kernel_kind(quality, in_sr, out_sr), KernelKind::Linear) {
                    continue;
                }
                for n in [1, 37, 8192, 20011] {
                    let x: Vec<f64> = (0..n)
                        .map(|i| (i as f64 * 0.37).sin() + 0.3 * (i as f64 * 0.0123).cos())
                        .collect();
                    let mut state =
                        StreamState::new(plan(in_sr, out_sr, quality).unwrap(), 1, in_sr, out_sr);
                    let mut direct = state.process(&[&x]).remove(0);
                    direct.extend(state.finish().remove(0));
                    let audio = mono(x, SampleRate::new(in_sr).unwrap());
                    let fast = resample(&audio, SampleRate::new(out_sr).unwrap(), quality).unwrap();
                    let fast = fast.as_mono().unwrap().to_vec();
                    assert_eq!(fast.len(), direct.len(), "{in_sr} -> {out_sr} n={n}");
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

    /// The `f32` engine (its own SIMD kernels) computes the same filter as the
    /// `f64` one, to `f32` round-off.
    #[test]
    fn test_f32_engine_matches_f64() {
        for (in_sr, out_sr) in [
            (44100, 22050),
            (22050, 44100),
            (22050, 16000),
            (16000, 22050),
            (22050, 17501),
            (1, 8),
            (8, 1),
        ] {
            let n = 20011;
            let x: Vec<f64> = (0..n)
                .map(|i| 0.5 * (i as f64 * 0.37).sin() + 0.3 * (i as f64 * 0.0123).cos())
                .collect();
            let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
            let r64 = resample(
                &mono(x, SampleRate::new(in_sr).unwrap()),
                SampleRate::new(out_sr).unwrap(),
                ResamplingQuality::High,
            )
            .unwrap();
            let a32 =
                AudioSamples::new_mono(Array1::from(x32), SampleRate::new(in_sr).unwrap()).unwrap();
            let r32 = resample(
                &a32,
                SampleRate::new(out_sr).unwrap(),
                ResamplingQuality::High,
            )
            .unwrap();
            let (y64, y32) = (
                r64.as_mono().unwrap().to_vec(),
                r32.as_mono().unwrap().to_vec(),
            );
            assert_eq!(y64.len(), y32.len());
            let err = y64
                .iter()
                .zip(&y32)
                .fold(0.0f64, |m, (a, b)| m.max((a - f64::from(*b)).abs()));
            assert!(err < 2e-6, "{in_sr} -> {out_sr}: {err:e}");
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
        let kernel = Kernel::new(44100, 44101, ResamplingQuality::High, &|_| None).unwrap();
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
            let kernel = Kernel::new(in_sr, out_sr, ResamplingQuality::High, &|_| None).unwrap();
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

    /// Regression (pitch_shift with many distinct steps): every awkward
    /// upsampling ratio built its own 4097-row table (~30 ms each) and the
    /// LRU thrashed. Upsampling tables do not depend on the ratio, so they
    /// are now shared; downsampling tables are sampled from a per-quality
    /// prototype instead of evaluating the window per coefficient.
    #[test]
    fn test_interpolated_tables_are_shared_and_accurate() {
        // Pitch-shift style rates (2^31 and a nearby integer).
        let base = 1u32 << 31;
        let a = cached_kernel(base - 12345, base, ResamplingQuality::High).unwrap();
        let b = cached_kernel(base - 999_999, base, ResamplingQuality::High).unwrap();
        assert!(!a.exact && !b.exact);
        assert!(Arc::ptr_eq(&a.coeffs, &b.coeffs));
        // Different quality: different table.
        let c = cached_kernel(base - 12345, base, ResamplingQuality::Medium).unwrap();
        assert!(!Arc::ptr_eq(&a.coeffs, &c.coeffs));

        // Tables sampled from the prototype match the directly evaluated
        // filter to well below the interpolation error budget.
        for (in_sr, out_sr) in [(44100u32, 44101u32), (44101, 44100), (48000, 7919)] {
            for quality in [
                ResamplingQuality::Fast,
                ResamplingQuality::Medium,
                ResamplingQuality::High,
            ] {
                let k = Kernel::new(in_sr, out_sr, quality, &|_| None).unwrap();
                let KernelKind::Sinc(spec) = kernel_kind(quality, in_sr, out_sr) else {
                    continue;
                };
                assert!(!k.exact);
                let lower = (f64::from(out_sr) / f64::from(in_sr)).min(1.0);
                let (_, h) = sinc_kernel(spec, lower);
                let mut err = 0.0f64;
                for row in 0..=k.phases {
                    let phi = row as f64 / k.phases as f64;
                    for j in 0..k.taps {
                        let tau = phi - (j as f64 - (k.half as f64 - 1.0));
                        err = err.max((k.coeffs[row * k.taps + j] - h(tau)).abs());
                    }
                }
                assert!(err < 1e-8, "{in_sr} -> {out_sr} {quality:?}: {err:e}");
            }
        }
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
