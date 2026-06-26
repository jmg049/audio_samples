//! Bit allocation and uniform scalar quantization for perceptual audio codecs.
//!
//! ## What
//!
//! Provides the quantization half of the perceptual codec pipeline:
//! - **Bit allocation** — distribute a fixed bit budget across frequency bands
//!   in proportion to each band's perceptual importance.
//! - **Quantization** — map `f32` MDCT coefficients to integer indices with a
//!   step size derived from the psychoacoustic masking threshold.
//! - **Dequantization** — recover `f32` coefficients from integer indices.
//!
//! ## Why
//!
//! Psychoacoustic analysis ([`super::analyse_signal`]) produces per-band
//! `energy`, `importance`, and `allowed_noise` scores. This module bridges
//! analysis and coding: [`allocate_bits`] distributes the bit budget across
//! bands (weighted by energy and perceptual importance), and
//! [`refine_step_sizes`] turns that budget into per-band quantiser step sizes
//! via optimal bit allocation, so the bit budget genuinely controls
//! reconstruction quality. `allowed_noise` seeds the initial perceptual step.
//!
//! ## How
//!
//! ```rust,ignore
//! // After running analyse_signal:
//! let allocation = allocate_bits(&result.band_metrics, 128_000, 1);
//! let quantized  = quantize(
//!     result.coefficients.as_non_empty_slice(),
//!     result.n_coefficients,
//!     result.n_frames,
//!     &allocation,
//! );
//! // … transmit or store `quantized` …
//! let recovered  = dequantize(
//!     quantized.as_non_empty_slice(),
//!     result.n_coefficients,
//!     result.n_frames,
//!     &allocation,
//! );
//! ```

use std::num::NonZeroUsize;

use non_empty_slice::{NonEmptySlice, NonEmptyVec};

use super::BandMetrics;

/// Bit allocation and quantization step size for a single frequency band.
#[derive(Debug, Clone, PartialEq)]
pub struct BandAllocation {
    /// First spectral bin index (inclusive) from the source [`Band`](super::Band).
    pub start_bin: usize,
    /// One past the last spectral bin index (exclusive).
    pub end_bin: usize,
    /// Total bits allocated to this band across all its coefficients
    /// (0 = band is not coded).
    pub bits: u32,
    /// Per-coefficient quantiser word length in bits.
    ///
    /// Every quantised index in this band must fit in a signed `word_length`-bit
    /// word, i.e. lie in `±(2^(word_length−1) − 1)` (see
    /// [`max_index_for_word_length`]). [`allocate_bits`] seeds it from the band's
    /// bit budget divided by its coefficient count; [`refine_step_sizes`] then
    /// overwrites it with the rate-distortion word length it actually used to set
    /// `step_size`, keeping the clamp consistent with the chosen resolution.
    pub word_length: u32,
    /// Uniform quantization step size in linear amplitude.
    ///
    /// Initialised from the psychoacoustic masking threshold
    /// (`10^(allowed_noise_db / 20) × √12`) and then refined by
    /// [`refine_step_sizes`] so that bands granted more bits are quantised more
    /// finely — letting the bit budget actually control reconstruction quality.
    pub step_size: f32,
}

/// Per-band bit allocation derived from [`BandMetrics`].
#[derive(Debug, Clone, PartialEq)]
pub struct BitAllocationResult {
    /// One [`BandAllocation`] per frequency band, in the same order as the
    /// source [`BandMetrics`].
    pub allocations: NonEmptyVec<BandAllocation>,
}

/// Computes the quantization step size for a band from its allowed-noise budget.
///
/// Uses the relation between uniform quantization noise and step size:
/// `noise_rms = step_size / sqrt(12)`. Setting `noise_rms` equal to the linear
/// amplitude corresponding to `allowed_noise_db` gives:
///
/// `step_size = 10^(allowed_noise_db / 20) × √12`
///
/// # Arguments
/// - `allowed_noise_db` – Allowed noise level in dB (from [`BandMetric::allowed_noise`]).
///
/// # Returns
/// Step size in linear amplitude, clamped to a minimum of `1e-6` to prevent
/// divide-by-zero in quantization.
///
/// [`BandMetric::allowed_noise`]: super::BandMetric::allowed_noise
#[inline]
#[must_use]
pub fn step_size_from_allowed_noise(allowed_noise_db: f32) -> f32 {
    let noise_amplitude = 10.0_f32.powf(allowed_noise_db / 20.0);
    (noise_amplitude * 12.0_f32.sqrt()).max(1e-6)
}

/// Allocates a fixed bit budget across frequency bands proportional to their
/// perceptual importance.
///
/// Steps:
/// 1. Reserve `min_bits_per_band` bits for every band.
/// 2. Distribute the remaining budget proportionally to `band.importance`
///    (clamped to ≥ 0). Bands with zero importance receive no extra bits.
/// 3. Derive a quantization step size for each band from
///    [`step_size_from_allowed_noise`].
///
/// # Arguments
/// - `band_metrics` – Per-band psychoacoustic metrics from
///   [`compute_band_metrics`](super::masking::compute_band_metrics).
/// - `total_bits` – Total bit budget to distribute.
/// - `min_bits_per_band` – Minimum bits guaranteed to every band.
///
/// # Returns
/// A [`BitAllocationResult`] with one [`BandAllocation`] per band.
#[must_use]
pub fn allocate_bits(
    band_metrics: &BandMetrics,
    total_bits: u32,
    min_bits_per_band: u8,
) -> BitAllocationResult {
    let n_bands = band_metrics.metrics.len().get();
    let reserved = (n_bands as u32).saturating_mul(min_bits_per_band as u32);
    let remaining = total_bits.saturating_sub(reserved);

    // Weight each band by its linear energy (`energy` is in dB:
    // `10·log10(power)`) boosted by its perceptual importance. Energy ensures
    // every signal-bearing band is coded even when it is perceptually masked
    // (e.g. self-masking noise), which keeps objective reconstruction quality
    // scaling with the bit budget; importance biases extra bits toward the
    // bands the ear cares about most.
    let band_weight = |m: &super::BandMetric| -> f32 {
        let energy_linear = 10.0_f32.powf(m.energy / 10.0);
        energy_linear * (1.0 + m.importance.max(0.0))
    };
    let total_weight: f32 = band_metrics.metrics.iter().map(band_weight).sum();

    let allocations: Vec<BandAllocation> = band_metrics
        .metrics
        .iter()
        .map(|m| {
            let extra_bits = if total_weight > 0.0 {
                let fraction = band_weight(m) / total_weight;
                (fraction * remaining as f32).round().max(0.0) as u32
            } else {
                0
            };
            let bits = u32::from(min_bits_per_band).saturating_add(extra_bits);
            let step_size = step_size_from_allowed_noise(m.allowed_noise);
            // Seed the per-coefficient word length from the band's bit budget
            // spread over its bins. `refine_step_sizes` overwrites this with the
            // rate-distortion word length it actually uses; this seed only
            // applies when `quantize` is called without `refine_step_sizes`.
            let band_width = m.band.end_bin.saturating_sub(m.band.start_bin).max(1) as u32;
            let word_length = bits / band_width;
            BandAllocation {
                start_bin: m.band.start_bin,
                end_bin: m.band.end_bin,
                bits,
                word_length,
                step_size,
            }
        })
        .collect();

    // SAFETY: n_bands >= 1 (BandMetrics invariant).
    let allocations = unsafe { NonEmptyVec::new_unchecked(allocations) };
    BitAllocationResult { allocations }
}

/// Largest signed magnitude representable in a `bits`-bit two's-complement word.
///
/// A `w`-bit signed index spans `[−2^(w−1), 2^(w−1) − 1]`. To keep quantisation
/// symmetric we clamp to `±(2^(w−1) − 1)`, the largest magnitude that fits in
/// both directions. A word length of `0` codes only the value `0`. Word lengths
/// at or above `i32`'s 31 usable magnitude bits impose no clamp (the value
/// always fits), so we saturate the cap at [`i32::MAX`].
#[inline]
#[must_use]
pub fn max_index_for_word_length(bits: u32) -> i32 {
    if bits == 0 {
        0
    } else if bits >= 31 {
        i32::MAX
    } else {
        (1i32 << (bits - 1)) - 1
    }
}

/// Uniformly quantizes a slice of `f32` MDCT coefficients to `i32` indices,
/// clamping each index to the band's per-coefficient word length.
///
/// `q[i] = clamp(round(c[i] / step_size), ±(2^(bits_per_coeff−1) − 1))`
///
/// The clamp guarantees every index fits in the `bits_per_coeff`-bit word the
/// bit allocator budgeted for this band, so a single outlier coefficient cannot
/// blow the band's bit budget. In-range values (the normal case) are unchanged.
///
/// # Arguments
/// - `coefficients` – Float coefficients for a single band.
/// - `step_size` – Quantization step size (from [`BandAllocation::step_size`]).
/// - `bits_per_coeff` – Word length per coefficient; the index is clamped to
///   `±(2^(bits_per_coeff−1) − 1)`. Pass `0` to force every index to `0`, or a
///   large value (≥ 31) to disable clamping.
#[inline]
#[must_use]
pub fn quantize_band(coefficients: &[f32], step_size: f32, bits_per_coeff: u32) -> Vec<i32> {
    let max_index = max_index_for_word_length(bits_per_coeff);
    coefficients
        .iter()
        .map(|&c| {
            let q = (c / step_size).round() as i32;
            q.clamp(-max_index, max_index)
        })
        .collect()
}

/// Dequantizes `i32` indices back to `f32` coefficients.
///
/// `c[i] = q[i] × step_size`
///
/// # Arguments
/// - `quantized` – Integer indices from [`quantize_band`].
/// - `step_size` – The same step size used during quantization.
#[inline]
#[must_use]
pub fn dequantize_band(quantized: &[i32], step_size: f32) -> Vec<f32> {
    quantized.iter().map(|&q| q as f32 * step_size).collect()
}

/// Quantizes all MDCT coefficients using the given band allocation.
///
/// Iterates over every band in `allocation` and applies uniform scalar
/// quantization to the corresponding coefficient bins across all frames.
/// Bins that fall outside any band allocation are set to 0.
///
/// Coefficients are expected in row-major layout: index `k * n_frames + f`
/// for spectral bin `k`, frame `f` — matching the layout of
/// [`PerceptualAnalysisResult::coefficients`].
///
/// # Arguments
/// - `coefficients` – Flat coefficient array with `n_coefficients × n_frames` elements.
/// - `n_coefficients` – Number of spectral bins per frame.
/// - `n_frames` – Number of MDCT frames.
/// - `allocation` – Per-band bit allocation from [`allocate_bits`].
///
/// # Returns
/// A `NonEmptyVec<i32>` with the same shape as `coefficients`.
///
/// # Panics
///
/// Panics (debug) if `coefficients.len() != n_coefficients × n_frames`.
///
/// [`PerceptualAnalysisResult::coefficients`]: super::PerceptualAnalysisResult::coefficients
#[must_use]
pub fn quantize(
    coefficients: &NonEmptySlice<f32>,
    n_coefficients: NonZeroUsize,
    n_frames: NonZeroUsize,
    allocation: &BitAllocationResult,
) -> NonEmptyVec<i32> {
    let nc = n_coefficients.get();
    let nf = n_frames.get();
    debug_assert_eq!(coefficients.len().get(), nc * nf);

    let mut out = vec![0i32; nc * nf];

    for alloc in allocation.allocations.iter() {
        let step = alloc.step_size;
        let k_end = alloc.end_bin.min(nc);
        if alloc.start_bin >= k_end {
            continue;
        }
        // Clamp each index to the band's per-coefficient word length so a single
        // large coefficient cannot overrun the bit budget the band was granted.
        let max_index = max_index_for_word_length(alloc.word_length);
        for k in alloc.start_bin..k_end {
            for f in 0..nf {
                let idx = k * nf + f;
                let q = (coefficients[idx] / step).round() as i32;
                out[idx] = q.clamp(-max_index, max_index);
            }
        }
    }

    // SAFETY: nc * nf >= 1 since both nc >= 1 and nf >= 1.
    unsafe { NonEmptyVec::new_unchecked(out) }
}

/// Dequantizes all MDCT coefficient bands using the given allocation.
///
/// Inverse of [`quantize`]: reconstructs `f32` coefficients from integer
/// indices. Bins outside any band in `allocation` are reconstructed as 0.
///
/// # Arguments
/// - `quantized` – Integer coefficient array with `n_coefficients × n_frames` elements.
/// - `n_coefficients` – Number of spectral bins per frame.
/// - `n_frames` – Number of MDCT frames.
/// - `allocation` – The same allocation used during [`quantize`].
///
/// # Returns
/// A `NonEmptyVec<f32>` with the same shape as `quantized`.
///
/// # Panics
///
/// Panics (debug) if `quantized.len() != n_coefficients × n_frames`.
#[must_use]
pub fn dequantize(
    quantized: &NonEmptySlice<i32>,
    n_coefficients: NonZeroUsize,
    n_frames: NonZeroUsize,
    allocation: &BitAllocationResult,
) -> NonEmptyVec<f32> {
    let nc = n_coefficients.get();
    let nf = n_frames.get();
    debug_assert_eq!(quantized.len().get(), nc * nf);

    let mut out = vec![0.0f32; nc * nf];

    for alloc in allocation.allocations.iter() {
        let step = alloc.step_size;
        for k in alloc.start_bin..alloc.end_bin.min(nc) {
            for f in 0..nf {
                let idx = k * nf + f;
                out[idx] = quantized[idx] as f32 * step;
            }
        }
    }

    // SAFETY: nc * nf >= 1 since both nc >= 1 and nf >= 1.
    unsafe { NonEmptyVec::new_unchecked(out) }
}

/// Maximum per-coefficient quantiser word length, in bits.
///
/// Caps the resolution at a level that is effectively lossless for normalised
/// MDCT coefficients while keeping quantised indices comfortably inside `i32`.
const MAX_WORDLENGTH_BITS: u32 = 24;

/// Refines per-band quantisation step sizes from the bit budget using optimal
/// bit allocation.
///
/// [`allocate_bits`] seeds each band's `step_size` from its psychoacoustic
/// masking threshold, which alone makes reconstruction quality independent of
/// the bit budget. This pass replaces those steps with a rate-driven assignment
/// so the budget actually controls quality.
///
/// It applies the classic high-resolution optimal bit-allocation rule: with an
/// average of `R` bits per coefficient and per-band coefficient variance
/// `σ²_k`, the rate-distortion-optimal word length for band `k` is
///
/// ```text
/// w_k = R + ½·log₂( σ²_k / geomean(σ²) )
/// ```
///
/// i.e. give every coefficient the average word length, then shift bits toward
/// louder bands and away from quieter ones. For a flat spectrum (white noise)
/// every band gets `R`, so the whole budget is used and SNR scales with the
/// budget; for a tone, the budget concentrates on the few loud bands. The step
/// size then covers each band's peak with `2^w_k` levels.
///
/// # Arguments
/// - `allocation` – Allocation from [`allocate_bits`]; `step_size` is updated in place.
///   Only the *sum* of `bits` (the total budget) is used here.
/// - `coefficients` – The MDCT coefficients about to be quantised (row-major,
///   `k * n_frames + f`).
/// - `n_coefficients` – Number of spectral bins per frame.
/// - `n_frames` – Number of MDCT frames.
pub fn refine_step_sizes(
    allocation: &mut BitAllocationResult,
    coefficients: &NonEmptySlice<f32>,
    n_coefficients: NonZeroUsize,
    n_frames: NonZeroUsize,
) {
    let nc = n_coefficients.get();
    let nf = n_frames.get();

    // Per-band statistics: peak magnitude, mean-square (variance proxy), and
    // coefficient count.
    struct BandStat {
        peak: f32,
        variance: f32,
        n_coeffs: usize,
    }

    let stats: Vec<BandStat> = allocation
        .allocations
        .iter()
        .map(|alloc| {
            let k_end = alloc.end_bin.min(nc);
            if alloc.start_bin >= k_end {
                return BandStat {
                    peak: 0.0,
                    variance: 0.0,
                    n_coeffs: 0,
                };
            }
            let mut peak = 0.0f32;
            let mut sum_sq = 0.0f64;
            let mut n = 0usize;
            for k in alloc.start_bin..k_end {
                for f in 0..nf {
                    let v = coefficients[k * nf + f];
                    let a = v.abs();
                    if a > peak {
                        peak = a;
                    }
                    sum_sq += f64::from(v) * f64::from(v);
                    n += 1;
                }
            }
            let variance = if n > 0 {
                (sum_sq / n as f64) as f32
            } else {
                0.0
            };
            BandStat {
                peak,
                variance,
                n_coeffs: n,
            }
        })
        .collect();

    // Total budget (sum of allocated bits) and total coded coefficients.
    let total_bits: u64 = allocation
        .allocations
        .iter()
        .map(|a| u64::from(a.bits))
        .sum();
    let total_coeffs: usize = stats.iter().map(|s| s.n_coeffs).sum();
    if total_coeffs == 0 {
        return;
    }
    let r_avg = total_bits as f32 / total_coeffs as f32;

    // Geometric mean of the (positive) per-band variances, in log2 space.
    let mut log_sum = 0.0f32;
    let mut active = 0usize;
    for s in &stats {
        if s.variance > 0.0 {
            log_sum += s.variance.log2();
            active += 1;
        }
    }
    let geomean_log2 = if active > 0 {
        log_sum / active as f32
    } else {
        0.0
    };

    for (alloc, s) in allocation.allocations.iter_mut().zip(stats.iter()) {
        if s.peak <= 0.0 || s.variance <= 0.0 {
            // Silent band: a coarse step quantises its ~zero data to zero.
            alloc.step_size = (2.0 * s.peak).max(1e-6);
            // A single sign bit is enough for the (clamped-to-zero) data.
            alloc.word_length = 1;
            continue;
        }

        // Optimal word length, clamped to a representable, near-lossless range.
        let w = (r_avg + 0.5 * (s.variance.log2() - geomean_log2))
            .clamp(0.0, MAX_WORDLENGTH_BITS as f32);
        if w < 0.5 {
            // Fewer than ~one level: drop the band (its energy is negligible).
            alloc.step_size = (2.0 * s.peak).max(1e-6);
            alloc.word_length = 1;
        } else {
            let levels = 2.0_f32.powf(w);
            alloc.step_size = ((2.0 * s.peak) / levels).max(1e-6);
            // The step covers `2·peak` with `2^w` levels, so the peak quantises
            // to index ≈ ±2^(w−1). Record `w` so the quantiser clamps to exactly
            // the resolution this step size implies (and not the coarser budget
            // seed). `+ 1` keeps the unbiased rounded peak inside the word.
            alloc.word_length = (w.ceil() as u32).saturating_add(1).min(MAX_WORDLENGTH_BITS + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_index_for_word_length_bounds() {
        assert_eq!(max_index_for_word_length(0), 0);
        assert_eq!(max_index_for_word_length(1), 0); // ±(2^0 − 1) = 0
        assert_eq!(max_index_for_word_length(2), 1); // ±(2^1 − 1) = 1
        assert_eq!(max_index_for_word_length(4), 7); // ±(2^3 − 1) = 7
        assert_eq!(max_index_for_word_length(8), 127);
        assert_eq!(max_index_for_word_length(31), i32::MAX);
        assert_eq!(max_index_for_word_length(64), i32::MAX);
    }

    #[test]
    fn quantize_band_clamps_to_word_length() {
        // A huge coefficient relative to the step would overflow a small word.
        // With a 4-bit word, indices must lie in ±7.
        let coeffs = [100.0_f32, -100.0, 3.0, -3.0, 0.0];
        let q = quantize_band(&coeffs, 1.0, 4);
        for &idx in &q {
            assert!((-7..=7).contains(&idx), "index {idx} escaped ±7 (4-bit word)");
        }
        // In-range values are untouched.
        assert_eq!(q[2], 3);
        assert_eq!(q[3], -3);
        assert_eq!(q[4], 0);
        // Out-of-range values are clamped to the boundary, not wrapped.
        assert_eq!(q[0], 7);
        assert_eq!(q[1], -7);
    }

    #[test]
    fn quantize_band_unbounded_word_is_lossless_rounding() {
        let coeffs = [1000.0_f32, -1000.0, 12.4, -12.6];
        // word_length >= 31 disables clamping.
        let q = quantize_band(&coeffs, 1.0, 31);
        assert_eq!(q, vec![1000, -1000, 12, -13]);
    }
}
